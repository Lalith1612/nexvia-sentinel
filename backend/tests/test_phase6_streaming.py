"""
Sentinel Phase 6: Kafka-Based Real-Time Transaction Streaming Tests

Verifies:
1. Valid transaction event publication, consumption, and prediction outcome generation.
2. Malformed JSON and Pydantic validation rejection to DLQ (sentinel.transactions.dlq).
3. Temporal lookahead violation rejection (step <= 620) to DLQ without model inference.
4. Duplicate event detection and idempotency deduplication (DUPLICATE_IGNORED).
5. Transient inference failure retry with exponential backoff and eventual recovery.
6. Retry exhaustion on non-recoverable error routing to DLQ with RETRY_EXHAUSTED.
7. Background consumer worker lifecycle, polling loop, and graceful shutdown.
8. Application startup behavior when KAFKA_ENABLED is False (no Kafka initialization).
9. Total non-interference: streaming does not mutate database tables, settlements, or replay engine.
"""

import json
from pathlib import Path
import time
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.core.config import settings
from backend.app.core.database import init_db, count_alerts, count_settlements
from backend.app.schemas.streaming import TransactionStreamEvent, TransactionResultEvent, DeadLetterEvent
from backend.app.services.inference_service import inference_service
from backend.app.services.streaming_service import StreamingIngestionService
from backend.app.workers.kafka_client import MockKafkaBroker, SentinelKafkaProducer, SentinelKafkaConsumer
from backend.app.workers.kafka_consumer import SentinelConsumerWorker
from backend.app.workers.kafka_producer import generate_synthetic_event, stream_synthetic_events


@pytest.fixture(scope="module", autouse=True)
def initialize_inference_artifacts(tmp_path_factory):
    """Initializes models and graph state once for the test module."""
    test_db = tmp_path_factory.mktemp("db") / "test_streaming_session.db"
    init_db(test_db)
    original_db = getattr(inference_service, "db_path", None)
    inference_service.db_path = test_db
    with TestClient(app):
        yield
    if original_db:
        inference_service.db_path = original_db


@pytest.fixture
def client(tmp_path):
    """TestClient configured with an isolated test ledger."""
    test_db = tmp_path / "test_client_ledger.db"
    init_db(test_db)
    original_db = getattr(inference_service, "db_path", None)
    inference_service.db_path = test_db
    with TestClient(app) as test_client:
        yield test_client
    if original_db:
        inference_service.db_path = original_db


@pytest.fixture(autouse=True)
def reset_streaming_state():
    """Resets mock broker state and deduplication cache before and after every test."""
    MockKafkaBroker.reset_instance()
    yield
    MockKafkaBroker.reset_instance()


@pytest.fixture
def streaming_env(tmp_path):
    """Provides an isolated test database and configured mock streaming service."""
    db_file = tmp_path / "test_streaming_ledger.db"
    init_db(db_file)
    original_db = getattr(inference_service, "db_path", None)
    inference_service.db_path = db_file

    mock_producer = SentinelKafkaProducer(
        bootstrap_servers="mock://localhost:9092",
        use_mock=True
    )
    svc = StreamingIngestionService(producer=mock_producer)

    yield {
        "db_file": db_file,
        "service": svc,
        "producer": mock_producer,
        "broker": MockKafkaBroker.get_instance()
    }

    if original_db:
        inference_service.db_path = original_db


def test_valid_event_scoring_and_result_publication(streaming_env):
    """
    Verifies that a valid transaction event is processed, scored by the hybrid model,
    and published to the results topic with valid risk tier, score, and latency.
    """
    svc = streaming_env["service"]
    broker = streaming_env["broker"]

    event_payload = {
        "transaction_id": "TXN_STREAM_001",
        "step": 625,
        "type": "PAYMENT",
        "amount": 250.0,
        "nameOrig": "C123456789",
        "oldbalanceOrg": 1000.0,
        "newbalanceOrig": 750.0,
        "nameDest": "M987654321",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 250.0
    }

    status, result = svc.process_raw_message(event_payload)
    assert status == "PROCESSED"
    assert result is not None
    assert result["transaction_id"] == "TXN_STREAM_001"
    assert result["status"] == "PROCESSED"
    assert 0.0 <= result["hybrid_risk_score"] <= 1.0
    assert result["assigned_risk_tier"] in ("CRITICAL", "HIGH", "MEDIUM", "LOW")
    assert result["recommended_action"] in (
        "AUTOMATED_BLOCK",
        "STEP_UP_AUTHENTICATION",
        "MANUAL_ANALYST_REVIEW",
        "AUTOMATED_CLEAR_AND_SETTLE"
    )
    assert result["latency_ms"] >= 0.0

    # Verify message in results topic
    results_msgs = broker.get_messages(settings.KAFKA_RESULTS_TOPIC)
    assert len(results_msgs) == 1
    published_res = json.loads(results_msgs[0].value().decode("utf-8"))
    assert published_res["transaction_id"] == "TXN_STREAM_001"
    assert published_res["status"] == "PROCESSED"

    # DLQ must be empty
    assert len(broker.get_messages(settings.KAFKA_DLQ_TOPIC)) == 0


def test_malformed_json_and_validation_errors_routed_to_dlq(streaming_env):
    """
    Verifies that corrupt bytes or invalid Pydantic schemas are rejected and routed to the DLQ.
    """
    svc = streaming_env["service"]
    broker = streaming_env["broker"]

    # 1. Corrupt non-JSON bytes
    status1, res1 = svc.process_raw_message(b"INVALID_CORRUPT_BYTES_{{[")
    assert status1 == "REJECTED_VALIDATION"
    assert res1 is None

    dlq_msgs = broker.get_messages(settings.KAFKA_DLQ_TOPIC)
    assert len(dlq_msgs) == 1
    dlq1 = json.loads(dlq_msgs[0].value().decode("utf-8"))
    assert dlq1["error_type"] == "CORRUPT_PAYLOAD"

    # 2. Schema violation (negative amount, invalid type)
    bad_payload = {
        "transaction_id": "TXN_BAD_001",
        "step": 625,
        "type": "INVALID_TRANSACTION_TYPE",
        "amount": -500.0,
        "nameOrig": "C1",  # Too short (min_length=2)
        "oldbalanceOrg": 0.0,
        "nameDest": "M2",
        "oldbalanceDest": 0.0
    }

    status2, res2 = svc.process_raw_message(bad_payload)
    assert status2 == "REJECTED_VALIDATION"

    dlq_msgs = broker.get_messages(settings.KAFKA_DLQ_TOPIC)
    assert len(dlq_msgs) == 2
    dlq2 = json.loads(dlq_msgs[1].value().decode("utf-8"))
    assert dlq2["error_type"] == "VALIDATION_ERROR"
    assert dlq2["event_id"] == "TXN_BAD_001"


def test_temporal_lookahead_violation_routed_to_dlq_without_model_scoring(streaming_env):
    """
    Verifies that an event with step <= snapshot cutoff (620) is rejected as a temporal lookahead
    violation, published to the DLQ with TEMPORAL_LOOKAHEAD_ERROR, and marked REJECTED_TEMPORAL_LOOKAHEAD.
    """
    svc = streaming_env["service"]
    broker = streaming_env["broker"]

    stale_payload = {
        "transaction_id": "TXN_STALE_001",
        "step": 600,  # <= 620 cutoff!
        "type": "TRANSFER",
        "amount": 10000.0,
        "nameOrig": "C_STALE_SRC",
        "oldbalanceOrg": 10000.0,
        "nameDest": "C_STALE_DST",
        "oldbalanceDest": 0.0
    }

    status, result = svc.process_raw_message(stale_payload)
    assert status == "REJECTED_TEMPORAL_LOOKAHEAD"
    assert result["status"] == "REJECTED_TEMPORAL_LOOKAHEAD"
    assert result["hybrid_risk_score"] is None

    # DLQ must receive the violation record
    dlq_msgs = broker.get_messages(settings.KAFKA_DLQ_TOPIC)
    assert len(dlq_msgs) == 1
    dlq_data = json.loads(dlq_msgs[0].value().decode("utf-8"))
    assert dlq_data["error_type"] == "TEMPORAL_LOOKAHEAD_ERROR"
    assert "Historical lookahead violation" in dlq_data["error_message"]


def test_duplicate_events_and_idempotency_deduplication(streaming_env):
    """
    Verifies that identical events received multiple times are deduplicated,
    returning DUPLICATE_IGNORED on subsequent occurrences without re-scoring.
    """
    svc = streaming_env["service"]
    broker = streaming_env["broker"]

    payload = {
        "transaction_id": "TXN_IDEMPOTENT_001",
        "step": 625,
        "type": "TRANSFER",
        "amount": 5000.0,
        "nameOrig": "C_IDEMP_SRC",
        "oldbalanceOrg": 5000.0,
        "nameDest": "C_IDEMP_DST",
        "oldbalanceDest": 0.0,
        "idempotency_key": "IDEMP_KEY_ABC"
    }

    # First delivery: PROCESSED
    status1, res1 = svc.process_raw_message(payload)
    assert status1 == "PROCESSED"
    assert res1["status"] == "PROCESSED"

    # Second delivery: DUPLICATE_IGNORED
    status2, res2 = svc.process_raw_message(payload)
    assert status2 == "DUPLICATE_IGNORED"
    assert res2["status"] == "DUPLICATE_IGNORED"
    assert "previously processed" in res2["error_reason"]

    # Results topic contains 2 records (one PROCESSED, one DUPLICATE_IGNORED)
    res_msgs = broker.get_messages(settings.KAFKA_RESULTS_TOPIC)
    assert len(res_msgs) == 2
    assert json.loads(res_msgs[0].value().decode("utf-8"))["status"] == "PROCESSED"
    assert json.loads(res_msgs[1].value().decode("utf-8"))["status"] == "DUPLICATE_IGNORED"


def test_transient_failure_retries_and_recovery(streaming_env, monkeypatch):
    """
    Verifies that transient inference errors trigger exponential backoff retries
    and successfully recover if a subsequent attempt succeeds.
    """
    svc = streaming_env["service"]
    original_predict = inference_service.predict_transaction
    attempts = 0

    def mock_flaky_predict(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise RuntimeError("Transient CUDA/Memory resource glitch")
        return original_predict(*args, **kwargs)

    monkeypatch.setattr(inference_service, "predict_transaction", mock_flaky_predict)

    payload = {
        "transaction_id": "TXN_FLAKY_RECOVER_001",
        "step": 625,
        "type": "PAYMENT",
        "amount": 100.0,
        "nameOrig": "C_FLAKY_SRC",
        "oldbalanceOrg": 500.0,
        "nameDest": "M_FLAKY_DST",
        "oldbalanceDest": 0.0
    }

    status, result = svc.process_raw_message(payload)
    assert status == "PROCESSED"
    assert attempts == 2
    assert result["status"] == "PROCESSED"


def test_retry_exhaustion_routes_to_dlq_with_failed_status(streaming_env, monkeypatch):
    """
    Verifies that permanent runtime inference failures exhaust retries
    and are cleanly routed to the DLQ with RETRY_EXHAUSTED.
    """
    svc = streaming_env["service"]
    broker = streaming_env["broker"]

    def mock_perma_fail(*args, **kwargs):
        raise RuntimeError("Unrecoverable inference failure")

    monkeypatch.setattr(inference_service, "predict_transaction", mock_perma_fail)

    payload = {
        "transaction_id": "TXN_PERMA_FAIL_001",
        "step": 625,
        "type": "PAYMENT",
        "amount": 100.0,
        "nameOrig": "C_FAIL_SRC",
        "oldbalanceOrg": 500.0,
        "nameDest": "M_FAIL_DST",
        "oldbalanceDest": 0.0
    }

    status, result = svc.process_raw_message(payload)
    assert status == "FAILED_INFERENCE"
    assert result["status"] == "FAILED"

    # DLQ receives RETRY_EXHAUSTED record
    dlq_msgs = broker.get_messages(settings.KAFKA_DLQ_TOPIC)
    assert len(dlq_msgs) == 1
    dlq = json.loads(dlq_msgs[0].value().decode("utf-8"))
    assert dlq["error_type"] == "RETRY_EXHAUSTED"
    assert dlq["retry_count"] == settings.KAFKA_MAX_RETRIES


def test_consumer_worker_polling_loop_and_graceful_shutdown(streaming_env):
    """
    Verifies that the background consumer worker polls incoming events,
    dispatches them through the streaming service, and terminates cleanly upon stop().
    """
    broker = streaming_env["broker"]
    svc = streaming_env["service"]

    # Seed 3 events into ingestion topic
    for i in range(3):
        event = generate_synthetic_event(step=625, transaction_id=f"TXN_WORKER_TEST_{i}")
        broker.produce(settings.KAFKA_TRANSACTIONS_TOPIC, json.dumps(event).encode("utf-8"))

    consumer = SentinelKafkaConsumer(
        bootstrap_servers="mock://localhost:9092",
        group_id="test-consumer-group",
        topics=[settings.KAFKA_TRANSACTIONS_TOPIC],
        use_mock=True
    )
    worker = SentinelConsumerWorker(consumer=consumer, service=svc)

    worker.start()
    assert worker.is_running is True

    # Allow worker thread to drain messages
    t_start = time.time()
    while len(broker.get_messages(settings.KAFKA_RESULTS_TOPIC)) < 3 and time.time() - t_start < 5.0:
        time.sleep(0.05)

    worker.stop(timeout=2.0)
    assert worker.is_running is False

    # All 3 events must have been scored and published to results topic
    results_msgs = broker.get_messages(settings.KAFKA_RESULTS_TOPIC)
    assert len(results_msgs) == 3


def test_disabled_kafka_startup_behavior(client):
    """
    Verifies that when KAFKA_ENABLED is False (the default), the FastAPI application
    starts cleanly without attempting broker connections or launching background consumer threads.
    """
    assert settings.KAFKA_ENABLED is False
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    assert res.json()["status"] == "HEALTHY"


def test_streaming_non_interference_with_ledger_alerts_and_replay(streaming_env, client):
    """
    Verifies that streaming inference leaves durable settlements, alerts,
    and replay engine completely unmutated (DRY_RUN isolation).
    """
    svc = streaming_env["service"]
    db_file = streaming_env["db_file"]

    pre_settlements = count_settlements(db_file)
    pre_alerts = count_alerts(db_file)

    # Process batch of events (including high-amount transfers)
    events = [
        generate_synthetic_event(step=625, transaction_id=f"TXN_NON_INTERFERE_{i}")
        for i in range(10)
    ]
    for ev in events:
        svc.process_raw_message(ev)

    post_settlements = count_settlements(db_file)
    post_alerts = count_alerts(db_file)

    # DB records must not be mutated
    assert post_settlements == pre_settlements
    assert post_alerts == pre_alerts

    # Replay engine must remain operable
    replay_res = client.post("/api/v1/replay/jobs", json={
        "records": [
            {
                "step": 625,
                "type": "PAYMENT",
                "amount": 50.0,
                "nameOrig": "C_TEST_REP",
                "oldbalanceOrg": 100.0,
                "nameDest": "M_TEST_REP",
                "oldbalanceDest": 0.0
            }
        ]
    })
    assert replay_res.status_code == 201
    assert replay_res.json()["mode"] == "DRY_RUN"
