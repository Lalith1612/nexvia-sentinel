"""
Kafka Client Abstraction & In-Memory Test Mock (Phase 6)
Supports both confluent-kafka and in-memory mock brokers for deterministic unit testing.
"""

from collections import deque
import json
import logging
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("sentinel.kafka.client")


class KafkaMessage:
    """Standard message representation abstraction across real and mock clients."""
    def __init__(
        self,
        topic: str,
        value: bytes,
        key: Optional[bytes] = None,
        offset: int = 0,
        partition: int = 0,
        error: Optional[Any] = None
    ):
        self._topic = topic
        self._value = value
        self._key = key
        self._offset = offset
        self._partition = partition
        self._error = error

    def topic(self) -> str:
        return self._topic

    def value(self) -> Optional[bytes]:
        return self._value

    def key(self) -> Optional[bytes]:
        return self._key

    def offset(self) -> int:
        return self._offset

    def partition(self) -> int:
        return self._partition

    def error(self) -> Optional[Any]:
        return self._error


class MockKafkaBroker:
    """
    Thread-safe in-memory broker supporting topic queues for deterministic offline testing.
    """
    _instance: Optional["MockKafkaBroker"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._topics: Dict[str, deque] = {}
        self._offsets: Dict[str, int] = {}
        self._lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "MockKafkaBroker":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        with cls._lock:
            cls._instance = cls()

    def produce(self, topic: str, value: bytes, key: Optional[bytes] = None) -> int:
        with self._lock:
            if topic not in self._topics:
                self._topics[topic] = deque()
                self._offsets[topic] = 0
            offset = self._offsets[topic]
            self._offsets[topic] += 1
            msg = KafkaMessage(topic=topic, value=value, key=key, offset=offset)
            self._topics[topic].append(msg)
            return offset

    def consume(self, topic: str, timeout: float = 0.1) -> Optional[KafkaMessage]:
        with self._lock:
            if topic in self._topics and len(self._topics[topic]) > 0:
                return self._topics[topic].popleft()
            return None

    def get_messages(self, topic: str) -> List[KafkaMessage]:
        with self._lock:
            if topic in self._topics:
                return list(self._topics[topic])
            return []

    def clear(self, topic: Optional[str] = None) -> None:
        with self._lock:
            if topic:
                self._topics.pop(topic, None)
                self._offsets.pop(topic, None)
            else:
                self._topics.clear()
                self._offsets.clear()


class SentinelKafkaProducer:
    """Producer wrapper delegating to confluent-kafka or in-memory mock."""
    def __init__(self, bootstrap_servers: str, use_mock: bool = False):
        self.bootstrap_servers = bootstrap_servers
        self.use_mock = use_mock
        self._producer = None

        if not self.use_mock:
            try:
                import confluent_kafka
                conf = {
                    "bootstrap.servers": bootstrap_servers,
                    "client.id": "sentinel-producer",
                    "acks": "all",
                    "retries": 3,
                    "retry.backoff.ms": 200
                }
                self._producer = confluent_kafka.Producer(conf)
                logger.info("Initialized Confluent Kafka Producer connected to %s", bootstrap_servers)
            except Exception as e:
                logger.warning("Failed to initialize Confluent Kafka Producer (%s); falling back to in-memory mock.", str(e))
                self.use_mock = True

    def produce_json(
        self,
        topic: str,
        payload: Dict[str, Any],
        key: Optional[str] = None,
        callback: Optional[Callable] = None
    ) -> None:
        raw_val = json.dumps(payload).encode("utf-8")
        raw_key = key.encode("utf-8") if key else None

        if self.use_mock:
            MockKafkaBroker.get_instance().produce(topic, raw_val, raw_key)
            if callback:
                callback(None, KafkaMessage(topic=topic, value=raw_val, key=raw_key))
            return

        assert self._producer is not None
        self._producer.produce(
            topic=topic,
            value=raw_val,
            key=raw_key,
            on_delivery=callback
        )
        self._producer.poll(0)

    def flush(self, timeout: float = 5.0) -> int:
        if self.use_mock or self._producer is None:
            return 0
        return self._producer.flush(timeout)


class SentinelKafkaConsumer:
    """Consumer wrapper delegating to confluent-kafka or in-memory mock."""
    def __init__(
        self,
        bootstrap_servers: str,
        group_id: str,
        topics: List[str],
        auto_offset_reset: str = "latest",
        use_mock: bool = False
    ):
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self.topics = topics
        self.auto_offset_reset = auto_offset_reset
        self.use_mock = use_mock
        self._consumer = None

        if not self.use_mock:
            try:
                import confluent_kafka
                conf = {
                    "bootstrap.servers": bootstrap_servers,
                    "group.id": group_id,
                    "auto.offset.reset": auto_offset_reset,
                    "enable.auto.commit": False
                }
                self._consumer = confluent_kafka.Consumer(conf)
                self._consumer.subscribe(topics)
                logger.info("Initialized Confluent Kafka Consumer (%s) subscribed to %s", group_id, topics)
            except Exception as e:
                logger.warning("Failed to initialize Confluent Kafka Consumer (%s); falling back to in-memory mock.", str(e))
                self.use_mock = True

    def poll(self, timeout: float = 1.0) -> Optional[Any]:
        if self.use_mock:
            broker = MockKafkaBroker.get_instance()
            for t in self.topics:
                msg = broker.consume(t, timeout=timeout)
                if msg is not None:
                    return msg
            return None

        assert self._consumer is not None
        msg = self._consumer.poll(timeout)
        return msg

    def commit(self, message: Optional[Any] = None) -> None:
        if self.use_mock or self._consumer is None:
            return
        try:
            self._consumer.commit(message=message, asynchronous=False)
        except Exception as e:
            logger.warning("Failed to commit offset: %s", str(e))

    def close(self) -> None:
        if self._consumer is not None:
            try:
                self._consumer.close()
                logger.info("Closed Confluent Kafka Consumer (%s).", self.group_id)
            except Exception as e:
                logger.warning("Error closing Kafka consumer: %s", str(e))
