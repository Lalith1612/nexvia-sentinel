"""
Sentinel Synthetic Transaction Stream Producer (Phase 6)
Streams synthetic or historical parquet transaction events to sentinel.transactions topic.
"""

from datetime import datetime, timezone
import logging
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Optional
import uuid

import pyarrow.parquet as pq

from backend.app.core.config import settings
from backend.app.schemas.streaming import TransactionStreamEvent
from backend.app.workers.kafka_client import SentinelKafkaProducer

logger = logging.getLogger("sentinel.workers.producer")

TRANSACTION_TYPES = ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"]


def generate_synthetic_event(
    step: int = 625,
    transaction_id: Optional[str] = None
) -> Dict[str, Any]:
    """Generates a realistic synthetic transaction event."""
    txn_id = transaction_id or f"TXN_SYNTH_{uuid.uuid4().hex[:12]}"
    ttype = random.choice(TRANSACTION_TYPES)
    amount = round(random.uniform(10.0, 50000.0), 2)
    old_orig = round(random.uniform(amount, amount * 2.5), 2)
    new_orig = round(max(0.0, old_orig - amount), 2)
    old_dest = round(random.uniform(0.0, 10000.0), 2)
    new_dest = round(old_dest + amount, 2)

    is_merchant = random.random() < 0.3
    dest_prefix = "M" if is_merchant else "C"
    name_orig = f"C{random.randint(1000000, 9999999)}"
    name_dest = f"{dest_prefix}{random.randint(1000000, 9999999)}"

    return {
        "transaction_id": txn_id,
        "step": step,
        "type": ttype,
        "amount": amount,
        "nameOrig": name_orig,
        "oldbalanceOrg": old_orig,
        "newbalanceOrig": new_orig,
        "nameDest": name_dest,
        "oldbalanceDest": old_dest,
        "newbalanceDest": new_dest,
        "event_timestamp": datetime.now(timezone.utc).isoformat(),
        "idempotency_key": txn_id
    }


def stream_events_from_parquet(
    parquet_path: Path,
    producer: SentinelKafkaProducer,
    topic: str = settings.KAFKA_TRANSACTIONS_TOPIC,
    max_records: int = 100,
    min_step: int = 621,
    interval_sec: float = 0.05
) -> int:
    """Streams up to max_records from a parquet dataset slice into Kafka."""
    parquet_file = pq.ParquetFile(str(parquet_path))
    produced_count = 0

    for batch in parquet_file.iter_batches(batch_size=min(max_records, 1024)):
        batch_df = batch.to_pandas()
        for idx, row in batch_df.iterrows():
            step = int(row["step"])
            if step < min_step:
                continue

            txn_id = f"TXN_STREAM_PQT_{produced_count}_S{step}"
            payload = {
                "transaction_id": txn_id,
                "step": step,
                "type": str(row["type"]),
                "amount": float(row["amount"]),
                "nameOrig": str(row["nameOrig"]),
                "oldbalanceOrg": float(row["oldbalanceOrg"]),
                "newbalanceOrig": float(row["newbalanceOrig"]) if "newbalanceOrig" in row and row["newbalanceOrig"] is not None else None,
                "nameDest": str(row["nameDest"]),
                "oldbalanceDest": float(row["oldbalanceDest"]),
                "newbalanceDest": float(row["newbalanceDest"]) if "newbalanceDest" in row and row["newbalanceDest"] is not None else None,
                "event_timestamp": datetime.now(timezone.utc).isoformat(),
                "idempotency_key": txn_id
            }

            producer.produce_json(topic=topic, payload=payload, key=txn_id)
            produced_count += 1
            if interval_sec > 0:
                time.sleep(interval_sec)

            if produced_count >= max_records:
                producer.flush(timeout=5.0)
                return produced_count

    producer.flush(timeout=5.0)
    return produced_count


def stream_synthetic_events(
    producer: SentinelKafkaProducer,
    topic: str = settings.KAFKA_TRANSACTIONS_TOPIC,
    count: int = 50,
    interval_sec: float = 0.05,
    step: int = 625
) -> int:
    """Publishes a batch of synthetic transaction events into Kafka."""
    for i in range(count):
        payload = generate_synthetic_event(step=step)
        producer.produce_json(topic=topic, payload=payload, key=payload["transaction_id"])
        if interval_sec > 0:
            time.sleep(interval_sec)
    producer.flush(timeout=5.0)
    return count
