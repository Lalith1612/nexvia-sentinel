"""
Sentinel Background Kafka Consumer Worker (Phase 6)
Continuously polls incoming transactions topic and invokes streaming ingestion service.
Supports graceful cooperative shutdown and offset management.
"""

import logging
import threading
import time
from typing import Optional

from backend.app.core.config import settings
from backend.app.services.streaming_service import StreamingIngestionService, streaming_service
from backend.app.workers.kafka_client import SentinelKafkaConsumer

logger = logging.getLogger("sentinel.workers.consumer")


class SentinelConsumerWorker:
    """
    Background worker managing consumer lifecycle, polling loop, and offset commits.
    """
    def __init__(
        self,
        consumer: Optional[SentinelKafkaConsumer] = None,
        service: Optional[StreamingIngestionService] = None
    ):
        self._consumer = consumer
        self._service = service or streaming_service
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._is_running = False

    @property
    def is_running(self) -> bool:
        return self._is_running

    def start(self) -> None:
        """Starts background consumer thread."""
        if self._is_running:
            logger.warning("Consumer worker is already running.")
            return

        if self._consumer is None:
            self._consumer = SentinelKafkaConsumer(
                bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
                group_id=settings.KAFKA_CONSUMER_GROUP,
                topics=[settings.KAFKA_TRANSACTIONS_TOPIC],
                auto_offset_reset=settings.KAFKA_AUTO_OFFSET_RESET,
                use_mock=settings.USE_MOCK_STREAMING
            )

        self._stop_event.clear()
        self._is_running = True
        self._thread = threading.Thread(
            target=self._run_loop,
            name="sentinel-kafka-consumer-worker",
            daemon=True
        )
        self._thread.start()
        logger.info("Started Sentinel Kafka Consumer Worker in background thread.")

    def _run_loop(self) -> None:
        """Core polling loop running in background thread."""
        poll_timeout = settings.KAFKA_POLL_TIMEOUT_SEC
        while not self._stop_event.is_set():
            try:
                msg = self._consumer.poll(timeout=poll_timeout)
                if msg is None:
                    continue

                if hasattr(msg, "error") and msg.error():
                    logger.warning("Kafka consumer error received: %s", msg.error())
                    continue

                raw_val = msg.value() if hasattr(msg, "value") else None
                if raw_val is not None:
                    self._service.process_raw_message(raw_val)

                # Commit offset post-processing / DLQ routing (at-least-once delivery)
                self._consumer.commit(message=msg)

            except Exception as e:
                logger.exception("Unexpected error in Kafka consumer polling loop: %s", str(e))
                time.sleep(0.5)

        logger.info("Sentinel Kafka Consumer Worker loop terminated.")
        self._is_running = False

    def stop(self, timeout: float = 5.0) -> None:
        """Signals stop event and waits for background thread to terminate."""
        if not self._is_running:
            return

        logger.info("Stopping Sentinel Kafka Consumer Worker...")
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

        if self._consumer:
            self._consumer.close()

        # Flush any pending producer outcome events
        if self._service and self._service.producer:
            self._service.producer.flush(timeout=2.0)

        self._is_running = False
        logger.info("Sentinel Kafka Consumer Worker stopped successfully.")


consumer_worker = SentinelConsumerWorker()
