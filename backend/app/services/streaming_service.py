"""
Sentinel Streaming Ingestion Service (Phase 6)
Coordinates event deserialization, schema validation, deduplication,
pre-authorization inference, retry orchestration, and DLQ dispatch.
"""

from collections import OrderedDict
from datetime import datetime, timezone
import json
import logging
import threading
import time
from typing import Any, Dict, Optional, Set, Tuple

from backend.app.core.config import settings
from backend.app.schemas.streaming import (
    TransactionStreamEvent,
    TransactionResultEvent,
    DeadLetterEvent
)
from backend.app.services.inference_service import inference_service
from backend.app.workers.kafka_client import SentinelKafkaProducer

logger = logging.getLogger("sentinel.services.streaming")


class StreamingIngestionService:
    """
    Core orchestrator for Kafka transaction event ingestion and prediction dispatch.
    """
    def __init__(
        self,
        producer: Optional[SentinelKafkaProducer] = None,
        max_cache_size: int = 50000
    ):
        self.producer = producer or SentinelKafkaProducer(
            bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
            use_mock=settings.USE_MOCK_STREAMING
        )
        self._processed_ids: OrderedDict[str, float] = OrderedDict()
        self._cache_lock = threading.Lock()
        self.max_cache_size = max_cache_size

    def is_duplicate(self, deduplication_key: str) -> bool:
        """Checks if a transaction_id or idempotency_key has already been processed."""
        with self._cache_lock:
            return deduplication_key in self._processed_ids

    def record_processed(self, deduplication_key: str) -> None:
        """Records key in thread-safe bounded LRU deduplication cache."""
        with self._cache_lock:
            self._processed_ids[deduplication_key] = time.time()
            if len(self._processed_ids) > self.max_cache_size:
                self._processed_ids.popitem(last=False)

    def clear_cache(self) -> None:
        """Clears deduplication cache (useful for test isolation)."""
        with self._cache_lock:
            self._processed_ids.clear()

    def process_raw_message(self, raw_value: Any) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        Processes a single message consumed from the Kafka ingestion topic.
        Returns (outcome_status, event_dict).
        Outcome statuses:
        - 'PROCESSED'
        - 'DUPLICATE_IGNORED'
        - 'REJECTED_VALIDATION'
        - 'REJECTED_TEMPORAL_LOOKAHEAD'
        - 'FAILED_INFERENCE'
        """
        # 1. Parse JSON
        if isinstance(raw_value, bytes):
            try:
                raw_str = raw_value.decode("utf-8")
                payload = json.loads(raw_str)
            except Exception as parse_err:
                logger.warning("Corrupt JSON message rejected: %s", str(parse_err))
                self._route_to_dlq(
                    event_id="CORRUPT_EVENT",
                    original_payload=str(raw_value)[:1000],
                    error_type="CORRUPT_PAYLOAD",
                    error_message=f"JSON decoding failure: {str(parse_err)}"
                )
                return "REJECTED_VALIDATION", None
        elif isinstance(raw_value, dict):
            payload = raw_value
        else:
            self._route_to_dlq(
                event_id="INVALID_TYPE",
                original_payload=str(raw_value)[:500],
                error_type="CORRUPT_PAYLOAD",
                error_message=f"Unsupported message type: {type(raw_value)}"
            )
            return "REJECTED_VALIDATION", None

        # 2. Validate Schema
        try:
            event = TransactionStreamEvent.model_validate(payload)
        except Exception as val_err:
            raw_id = payload.get("transaction_id", "UNKNOWN") if isinstance(payload, dict) else "UNKNOWN"
            logger.warning("Transaction event %s failed schema validation: %s", raw_id, str(val_err))
            self._route_to_dlq(
                event_id=raw_id,
                original_payload=payload,
                error_type="VALIDATION_ERROR",
                error_message=str(val_err)
            )
            return "REJECTED_VALIDATION", None

        # 3. Check Deduplication & Idempotency
        dedup_key = event.idempotency_key or event.transaction_id
        if self.is_duplicate(dedup_key):
            logger.info("Duplicate transaction event %s ignored.", dedup_key)
            result = TransactionResultEvent(
                transaction_id=event.transaction_id,
                status="DUPLICATE_IGNORED",
                error_reason=f"Event with key '{dedup_key}' was previously processed."
            )
            self._publish_result(result)
            return "DUPLICATE_IGNORED", result.model_dump()

        # 4. Execute Pre-Authorization Inference with Bounded Retries
        txn_req = event.to_transaction_request()
        max_retries = max(1, settings.KAFKA_MAX_RETRIES)
        last_error = None

        for attempt in range(max_retries):
            try:
                pred_resp = inference_service.predict_transaction(
                    txn=txn_req,
                    transaction_id=event.transaction_id
                )
                # Success: record deduplication and publish result
                self.record_processed(dedup_key)
                result = TransactionResultEvent(
                    transaction_id=event.transaction_id,
                    status="PROCESSED",
                    hybrid_risk_score=pred_resp.hybrid_risk_score,
                    assigned_risk_tier=pred_resp.assigned_risk_tier,
                    recommended_action=pred_resp.recommended_action,
                    latency_ms=pred_resp.latency_ms
                )
                self._publish_result(result)

                # Route high/critical alerts to dedicated topic
                if pred_resp.assigned_risk_tier in ("CRITICAL", "HIGH"):
                    self._publish_alert(pred_resp)

                return "PROCESSED", result.model_dump()

            except ValueError as ve:
                # Non-transient temporal lookahead or validation error from inference service
                err_msg = str(ve)
                logger.warning("Temporal boundary violation for transaction %s: %s", event.transaction_id, err_msg)
                self._route_to_dlq(
                    event_id=event.transaction_id,
                    original_payload=payload,
                    error_type="TEMPORAL_LOOKAHEAD_ERROR",
                    error_message=err_msg
                )
                result = TransactionResultEvent(
                    transaction_id=event.transaction_id,
                    status="REJECTED_TEMPORAL_LOOKAHEAD",
                    error_reason=err_msg
                )
                self._publish_result(result)
                return "REJECTED_TEMPORAL_LOOKAHEAD", result.model_dump()

            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Inference attempt %d/%d failed for transaction %s: %s",
                    attempt + 1, max_retries, event.transaction_id, str(exc)
                )
                if attempt + 1 < max_retries:
                    backoff = (settings.KAFKA_RETRY_BACKOFF_MS / 1000.0) * (2 ** attempt)
                    time.sleep(backoff)

        # 5. Retry Exhaustion
        logger.error("Inference retry exhausted for transaction %s: %s", event.transaction_id, str(last_error))
        self._route_to_dlq(
            event_id=event.transaction_id,
            original_payload=payload,
            error_type="RETRY_EXHAUSTED",
            error_message=f"Inference failed after {max_retries} attempts: {str(last_error)}",
            retry_count=max_retries
        )
        result = TransactionResultEvent(
            transaction_id=event.transaction_id,
            status="FAILED",
            error_reason=f"Inference failure: {str(last_error)}"
        )
        self._publish_result(result)
        return "FAILED_INFERENCE", result.model_dump()

    def _publish_result(self, result: TransactionResultEvent) -> None:
        try:
            self.producer.produce_json(
                topic=settings.KAFKA_RESULTS_TOPIC,
                payload=result.model_dump(),
                key=result.transaction_id
            )
        except Exception as e:
            logger.warning("Failed to publish prediction outcome to Kafka: %s", str(e))

    def _publish_alert(self, pred_resp: Any) -> None:
        try:
            alert_payload = {
                "transaction_id": pred_resp.transaction_id,
                "hybrid_risk_score": pred_resp.hybrid_risk_score,
                "assigned_risk_tier": pred_resp.assigned_risk_tier,
                "recommended_action": pred_resp.recommended_action,
                "latency_ms": pred_resp.latency_ms,
                "alert_timestamp": datetime.now(timezone.utc).isoformat()
            }
            self.producer.produce_json(
                topic=settings.KAFKA_ALERTS_TOPIC,
                payload=alert_payload,
                key=pred_resp.transaction_id
            )
        except Exception as e:
            logger.warning("Failed to publish fraud alert to Kafka: %s", str(e))

    def _route_to_dlq(
        self,
        event_id: str,
        original_payload: Any,
        error_type: str,
        error_message: str,
        retry_count: int = 0
    ) -> None:
        try:
            dlq_event = DeadLetterEvent(
                event_id=event_id,
                original_payload=original_payload,
                error_type=error_type,
                error_message=error_message[:1000],
                retry_count=retry_count
            )
            self.producer.produce_json(
                topic=settings.KAFKA_DLQ_TOPIC,
                payload=dlq_event.model_dump(),
                key=event_id
            )
        except Exception as e:
            logger.warning("Failed to route failed event to DLQ: %s", str(e))


streaming_service = StreamingIngestionService()
