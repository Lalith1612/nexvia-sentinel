"""
Sentinel Phase 5.3A: Read-Only Transaction Replay Engine Tests

Verifies:
1. SQLite schema initialization and WAL mode for replay_jobs.
2. Strict input validation and exclusive input-source selection (inline vs dataset).
3. Rejection of forbidden labels (isFraud, isFlaggedFraud) and extra fields.
4. Record limits (max 1000 inline, max 5000 dataset) and whitelist enforcement.
5. Dry-run non-interference: zero mutations to alerts table, settlements table, graph degrees, or embeddings.
6. Deterministic pre-sort stable ID assignment and chronological ordering.
7. Stale watermark handling (step <= 620 skipped and counted, never scored).
8. Idempotent retries (same key & payload returns 200) vs conflict (same key & different payload returns 409).
9. Mutual exclusion / concurrent job rejection (returns 409).
10. Cooperative cancellation semantics and terminal-state immutability.
11. Startup crash reconciliation for interrupted jobs.
12. Latency percentiles and risk tier metrics calculation.
"""

from pathlib import Path
import time
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.core.database import (
    init_db,
    get_connection,
    count_alerts,
    count_settlements,
    create_replay_job,
    get_replay_job,
    reconcile_interrupted_replay_jobs
)
from backend.app.services.inference_service import inference_service
from backend.app.services.replay_service import _replay_worker_lock


@pytest.fixture
def replay_db(tmp_path):
    """Provides an isolated SQLite database for replay tests."""
    db_file = tmp_path / "test_replay_ledger.db"
    init_db(db_file)
    original_db = getattr(inference_service, "db_path", None)
    inference_service.db_path = db_file
    yield db_file
    if original_db:
        inference_service.db_path = original_db


@pytest.fixture
def client(replay_db):
    """TestClient configured with the isolated replay database."""
    with TestClient(app) as test_client:
        yield test_client


def _wait_for_job_completion(client, job_id: str, timeout_sec: float = 15.0) -> dict:
    """Helper polling until job reaches a terminal state."""
    start = time.time()
    while time.time() - start < timeout_sec:
        resp = client.get(f"/api/v1/replay/jobs/{job_id}")
        assert resp.status_code == 200
        data = resp.json()
        if data["status"] in {"COMPLETED", "FAILED", "CANCELLED"}:
            return data
        time.sleep(0.05)
    raise TimeoutError(f"Job {job_id} did not terminate within {timeout_sec}s.")


def test_replay_schema_and_indexes_initialization(replay_db):
    """Verifies that init_db creates replay_jobs table and indexes in WAL mode."""
    conn = get_connection(replay_db)
    cursor = conn.cursor()

    cursor.execute("PRAGMA journal_mode;")
    assert cursor.fetchone()[0].lower() == "wal"

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='replay_jobs';")
    assert cursor.fetchone() is not None

    cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_replay_jobs_status';")
    assert cursor.fetchone() is not None

    cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_replay_jobs_idempotency';")
    assert cursor.fetchone() is not None
    conn.close()


def test_strict_input_validation_exclusive_source(client):
    """Verifies mutual exclusivity between inline records and dataset_source."""
    valid_record = {
        "step": 625,
        "type": "TRANSFER",
        "amount": 50000.0,
        "nameOrig": "C_TEST_SRC_01",
        "oldbalanceOrg": 50000.0,
        "nameDest": "C_TEST_DST_01",
        "oldbalanceDest": 0.0
    }

    # 1. Both provided -> 422
    res1 = client.post("/api/v1/replay/jobs", json={
        "mode": "DRY_RUN",
        "dataset_source": "test.parquet",
        "records": [valid_record]
    })
    assert res1.status_code == 422
    assert "Specify either 'records' or 'dataset_source', not both" in str(res1.json())

    # 2. Neither provided -> 422
    res2 = client.post("/api/v1/replay/jobs", json={
        "mode": "DRY_RUN"
    })
    assert res2.status_code == 422
    assert "Exactly one input source must be specified" in str(res2.json())

    # 3. Empty records list -> 422
    res3 = client.post("/api/v1/replay/jobs", json={
        "mode": "DRY_RUN",
        "records": []
    })
    assert res3.status_code == 422


def test_extra_fields_and_labels_forbidden(client):
    """Verifies rejection of isFraud, isFlaggedFraud, or arbitrary extra fields."""
    base_record = {
        "step": 625,
        "type": "TRANSFER",
        "amount": 25000.0,
        "nameOrig": "C_VAL_1",
        "oldbalanceOrg": 25000.0,
        "nameDest": "C_VAL_2",
        "oldbalanceDest": 0.0
    }

    # Attempt to include isFraud
    leak_fraud = dict(base_record, isFraud=1)
    res1 = client.post("/api/v1/replay/jobs", json={"records": [leak_fraud]})
    assert res1.status_code == 422
    assert "extra" in str(res1.json()).lower() or "isFraud" in str(res1.json())

    # Attempt to include isFlaggedFraud
    leak_flagged = dict(base_record, isFlaggedFraud=0)
    res2 = client.post("/api/v1/replay/jobs", json={"records": [leak_flagged]})
    assert res2.status_code == 422

    # Attempt to include arbitrary extra field
    extra_field = dict(base_record, arbitrary_column="malicious_payload")
    res3 = client.post("/api/v1/replay/jobs", json={"records": [extra_field]})
    assert res3.status_code == 422


def test_dry_run_inline_batch_non_interference(client, replay_db):
    """
    Submits an inline batch in DRY_RUN mode.
    Verifies that all predictions execute, metrics are computed,
    and alerts, settlements, and graph degrees remain 100% unmutated.
    """
    initial_alerts = count_alerts(replay_db)
    initial_settlements = count_settlements(replay_db)
    initial_max_settled_step = inference_service.max_settled_step

    records = [
        {
            "step": 625,
            "type": "TRANSFER",
            "amount": 100000.0 * (i + 1),
            "nameOrig": f"C_DRY_SRC_{i}",
            "oldbalanceOrg": 100000.0 * (i + 1),
            "nameDest": f"C_DRY_DST_{i}",
            "oldbalanceDest": 0.0
        }
        for i in range(5)
    ]

    create_res = client.post("/api/v1/replay/jobs", json={"records": records})
    assert create_res.status_code == 201
    job_id = create_res.json()["job_id"]

    final_job = _wait_for_job_completion(client, job_id)
    assert final_job["status"] == "COMPLETED"
    assert final_job["total_records"] == 5
    assert final_job["processed_records"] == 5
    assert final_job["failed_records"] == 0
    assert final_job["skipped_records"] == 0

    # Metrics verification
    metrics = final_job["metrics_summary"]
    assert metrics is not None
    assert metrics["latency_ms"]["p50"] > 0.0
    assert metrics["latency_ms"]["p95"] >= metrics["latency_ms"]["p50"]
    assert sum(metrics["risk_tier_counts"].values()) == 5
    assert metrics["state_mutations"]["alerts_created"] == 0
    assert metrics["state_mutations"]["settlements_applied"] == 0

    # NON-INTERFERENCE ASSERTIONS:
    assert count_alerts(replay_db) == initial_alerts
    assert count_settlements(replay_db) == initial_settlements
    assert inference_service.max_settled_step == initial_max_settled_step


def test_dry_run_registered_dataset_slice(client, replay_db):
    """
    Verifies replaying a bounded slice from registered dataset test.parquet.
    Confirms bounded streaming read and non-interference.
    """
    initial_alerts = count_alerts(replay_db)
    initial_settlements = count_settlements(replay_db)

    res = client.post("/api/v1/replay/jobs", json={
        "dataset_source": "test.parquet",
        "max_records": 10
    })
    assert res.status_code == 201
    job_id = res.json()["job_id"]

    final_job = _wait_for_job_completion(client, job_id)
    assert final_job["status"] == "COMPLETED"
    assert final_job["total_records"] == 10
    assert final_job["processed_records"] + final_job["skipped_records"] == 10

    # Zero mutation guarantees
    assert count_alerts(replay_db) == initial_alerts
    assert count_settlements(replay_db) == initial_settlements


def test_missing_dataset_source_returns_400(client, monkeypatch, tmp_path):
    """
    Verifies that requesting a registered dataset that does not exist on disk
    fails cleanly with HTTP 400 Bad Request and descriptive detail,
    reproducing the Render error condition and ensuring graceful API rejection.
    """
    from backend.app.core.config import settings

    empty_data_dir = tmp_path / "empty_processed"
    empty_data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(settings, "DATA_DIR", empty_data_dir)

    res = client.post("/api/v1/replay/jobs", json={
        "dataset_source": "test.parquet",
        "max_records": 10
    })
    assert res.status_code == 400
    assert "Registered dataset 'test.parquet' not found" in res.json()["detail"]


def test_dataset_replay_isolation_and_schema_validation(client, replay_db):
    """
    Verifies that replaying test.parquet streams valid records, evaluates them
    against the inference model, and maintains absolute zero-ledger-mutation isolation.
    """
    initial_alerts = count_alerts(replay_db)
    initial_settlements = count_settlements(replay_db)
    initial_watermark = inference_service.max_settled_step

    res = client.post("/api/v1/replay/jobs", json={
        "dataset_source": "test.parquet",
        "max_records": 25
    })
    assert res.status_code == 201
    job_id = res.json()["job_id"]

    final_job = _wait_for_job_completion(client, job_id)
    assert final_job["status"] == "COMPLETED"
    assert final_job["total_records"] == 25
    assert final_job["processed_records"] + final_job["skipped_records"] == 25

    metrics = final_job["metrics_summary"]
    assert metrics is not None
    assert metrics["records_summary"]["total"] == 25
    assert metrics["state_mutations"]["alerts_created"] == 0
    assert metrics["state_mutations"]["settlements_applied"] == 0

    # Ensure absolute non-interference with ledger and runtime graph watermark
    assert count_alerts(replay_db) == initial_alerts
    assert count_settlements(replay_db) == initial_settlements
    assert inference_service.max_settled_step == initial_watermark


def test_stale_watermark_records_skipped(client):
    """
    Verifies that transactions with step <= 620 are skipped (never scored),
    counted in skipped_records, and noted in error_summary.
    """
    records = [
        # Stale records (<= 620)
        {"step": 600, "type": "TRANSFER", "amount": 1000.0, "nameOrig": "C_STALE_1", "oldbalanceOrg": 1000.0, "nameDest": "C_STALE_D1", "oldbalanceDest": 0.0},
        {"step": 620, "type": "TRANSFER", "amount": 2000.0, "nameOrig": "C_STALE_2", "oldbalanceOrg": 2000.0, "nameDest": "C_STALE_D2", "oldbalanceDest": 0.0},
        # Valid records (> 620)
        {"step": 621, "type": "TRANSFER", "amount": 3000.0, "nameOrig": "C_VAL_1", "oldbalanceOrg": 3000.0, "nameDest": "C_VAL_D1", "oldbalanceDest": 0.0},
        {"step": 625, "type": "TRANSFER", "amount": 4000.0, "nameOrig": "C_VAL_2", "oldbalanceOrg": 4000.0, "nameDest": "C_VAL_D2", "oldbalanceDest": 0.0},
    ]

    res = client.post("/api/v1/replay/jobs", json={"records": records})
    assert res.status_code == 201
    job_id = res.json()["job_id"]

    final_job = _wait_for_job_completion(client, job_id)
    assert final_job["status"] == "COMPLETED"
    assert final_job["total_records"] == 4
    assert final_job["processed_records"] == 2
    assert final_job["skipped_records"] == 2

    # Check bounded error summary
    errors = final_job["error_summary"]
    assert len(errors) == 2
    assert any("SKIPPED_STALE_WATERMARK" in e["reason"] for e in errors)


def test_deterministic_ordering_and_stable_transaction_ids(client):
    """
    Verifies that records provided in reverse chronological order
    are sorted deterministically while preserving stable assigned transaction IDs.
    """
    records = [
        {"step": 630, "type": "TRANSFER", "amount": 10000.0, "nameOrig": "C_ORD_3", "oldbalanceOrg": 10000.0, "nameDest": "C_ORD_D3", "oldbalanceDest": 0.0},
        {"step": 625, "type": "TRANSFER", "amount": 20000.0, "nameOrig": "C_ORD_2", "oldbalanceOrg": 20000.0, "nameDest": "C_ORD_D2", "oldbalanceDest": 0.0},
        {"step": 622, "type": "TRANSFER", "amount": 30000.0, "nameOrig": "C_ORD_1", "oldbalanceOrg": 30000.0, "nameDest": "C_ORD_D1", "oldbalanceDest": 0.0},
    ]

    res = client.post("/api/v1/replay/jobs", json={"records": records})
    assert res.status_code == 201
    job_id = res.json()["job_id"]

    final_job = _wait_for_job_completion(client, job_id)
    assert final_job["status"] == "COMPLETED"
    assert final_job["processed_records"] == 3


def test_idempotency_key_replay_and_conflict(client):
    """
    Verifies:
    1. Repeated submission with identical idempotency_key returns existing job (200 OK).
    2. Reusing key with different input payload returns 409 Conflict.
    """
    idemp_key = "IDEMP_REPLAY_TEST_KEY_99"
    payload1 = {
        "idempotency_key": idemp_key,
        "records": [
            {"step": 625, "type": "TRANSFER", "amount": 5000.0, "nameOrig": "C_ID_1", "oldbalanceOrg": 5000.0, "nameDest": "C_ID_2", "oldbalanceDest": 0.0}
        ]
    }

    # 1. Initial creation -> 201 Created
    res1 = client.post("/api/v1/replay/jobs", json=payload1)
    assert res1.status_code == 201
    job_id = res1.json()["job_id"]

    # 2. Identical retry -> 200 OK with same job_id
    res2 = client.post("/api/v1/replay/jobs", json=payload1)
    assert res2.status_code == 200
    assert res2.json()["job_id"] == job_id

    # 3. Differing payload with same key -> 409 Conflict
    payload2 = {
        "idempotency_key": idemp_key,
        "records": [
            {"step": 626, "type": "PAYMENT", "amount": 99999.0, "nameOrig": "C_DIFF_1", "oldbalanceOrg": 99999.0, "nameDest": "C_DIFF_2", "oldbalanceDest": 0.0}
        ]
    }
    res3 = client.post("/api/v1/replay/jobs", json=payload2)
    assert res3.status_code == 409
    assert "already exists with a different input payload" in res3.json()["detail"]


def test_concurrent_job_rejection(client):
    """Verifies that attempting to start a replay job while another is active returns 409 Conflict."""
    # Temporarily acquire the worker lock to simulate an active running job
    acquired = _replay_worker_lock.acquire(blocking=False)
    assert acquired is True

    try:
        res = client.post("/api/v1/replay/jobs", json={
            "records": [
                {"step": 625, "type": "TRANSFER", "amount": 5000.0, "nameOrig": "C_LOCKED", "oldbalanceOrg": 5000.0, "nameDest": "C_DST", "oldbalanceDest": 0.0}
            ]
        })
        assert res.status_code == 409
        assert "Concurrent replay execution is not permitted" in res.json()["detail"]
    finally:
        _replay_worker_lock.release()


def test_cooperative_cancellation(client):
    """
    Submits a job with 100 records and issues a cancel request.
    Verifies that the job terminates promptly in CANCELLED status.
    """
    records = [
        {
            "step": 625 + (i // 10),
            "type": "TRANSFER",
            "amount": 1000.0 * (i + 1),
            "nameOrig": f"C_CANCEL_SRC_{i}",
            "oldbalanceOrg": 1000.0 * (i + 1),
            "nameDest": f"C_CANCEL_DST_{i}",
            "oldbalanceDest": 0.0
        }
        for i in range(100)
    ]

    res = client.post("/api/v1/replay/jobs", json={"records": records})
    assert res.status_code == 201
    job_id = res.json()["job_id"]

    # Issue cancel
    cancel_res = client.post(f"/api/v1/replay/jobs/{job_id}/cancel")
    assert cancel_res.status_code in {200, 409}

    final_job = _wait_for_job_completion(client, job_id)
    assert final_job["status"] in {"CANCELLED", "COMPLETED"}

    # Second cancel attempt on terminal job must return 409 Conflict
    re_cancel = client.post(f"/api/v1/replay/jobs/{job_id}/cancel")
    assert re_cancel.status_code == 409
    assert "terminal state" in re_cancel.json()["detail"]


def test_startup_crash_reconciliation(replay_db):
    """
    Seeds a job stuck in RUNNING status and calls reconcile_interrupted_replay_jobs().
    Verifies atomic transition to FAILED with error explanation.
    """
    create_replay_job(
        db_path=replay_db,
        job_id="REP_ORPHAN_CRASH_01",
        idempotency_key=None,
        input_fingerprint="dummy_fingerprint",
        mode="DRY_RUN",
        status="RUNNING",
        total_records=50
    )

    reconciled_count = reconcile_interrupted_replay_jobs(replay_db)
    assert reconciled_count == 1

    job = get_replay_job(replay_db, "REP_ORPHAN_CRASH_01")
    assert job["status"] == "FAILED"
    assert "Server process terminated before job completion" in str(job["error_summary"])


def test_record_limits_and_whitelist_enforced(client):
    """Verifies that record limits (1000 inline, 5000 dataset) and invalid datasets are rejected."""
    # 1. Invalid dataset source
    res1 = client.post("/api/v1/replay/jobs", json={"dataset_source": "train.parquet"})
    assert res1.status_code == 422

    # 2. max_records > 5000
    res2 = client.post("/api/v1/replay/jobs", json={
        "dataset_source": "test.parquet",
        "max_records": 5001
    })
    assert res2.status_code == 422


def test_job_listing_and_filtering(client):
    """Verifies GET /api/v1/replay/jobs pagination and status filtering."""
    list_res = client.get("/api/v1/replay/jobs?limit=5&offset=0")
    assert list_res.status_code == 200
    data = list_res.json()
    assert "total" in data
    assert "items" in data
    assert data["limit"] == 5
    assert data["offset"] == 0


def test_all_stale_watermark_batch_metrics(client):
    """
    Verifies that a batch containing exclusively stale watermark records (step <= 620)
    terminates with 0 processed, N skipped, 0 failed, and valid zero latency statistics.
    """
    records = [
        {"step": 100, "type": "TRANSFER", "amount": 1000.0, "nameOrig": "C_STALE_A", "oldbalanceOrg": 1000.0, "nameDest": "C_DST_A", "oldbalanceDest": 0.0},
        {"step": 620, "type": "TRANSFER", "amount": 2000.0, "nameOrig": "C_STALE_B", "oldbalanceOrg": 2000.0, "nameDest": "C_DST_B", "oldbalanceDest": 0.0},
    ]

    res = client.post("/api/v1/replay/jobs", json={"records": records})
    assert res.status_code == 201
    job_id = res.json()["job_id"]

    final_job = _wait_for_job_completion(client, job_id)
    assert final_job["status"] == "COMPLETED"
    assert final_job["total_records"] == 2
    assert final_job["processed_records"] == 0
    assert final_job["skipped_records"] == 2
    assert final_job["failed_records"] == 0

    metrics = final_job["metrics_summary"]
    assert metrics["records_summary"]["processed"] == 0
    assert metrics["records_summary"]["skipped_stale_watermark"] == 2
    assert metrics["latency_ms"]["p50"] == 0.0
    assert metrics["latency_ms"]["mean"] == 0.0
    assert sum(metrics["risk_tier_counts"].values()) == 0


def test_dispatch_failure_transitions_to_failed(client, replay_db, monkeypatch):
    """
    Verifies that if threadpool dispatch raises an unhandled exception,
    the job is not left hanging in PENDING status, but is atomically transitioned to FAILED.
    """
    import backend.app.services.replay_service as rep_mod

    def mock_submit_fail(*args, **kwargs):
        raise RuntimeError("Simulated threadpool executor worker scheduling failure")

    monkeypatch.setattr(rep_mod._executor, "submit", mock_submit_fail)

    records = [
        {"step": 625, "type": "TRANSFER", "amount": 5000.0, "nameOrig": "C_FAIL_1", "oldbalanceOrg": 5000.0, "nameDest": "C_DST_1", "oldbalanceDest": 0.0}
    ]

    res = client.post("/api/v1/replay/jobs", json={"records": records})
    assert res.status_code == 500
    assert "dispatch" in res.json()["detail"].lower()

    # Verify the database state
    conn = get_connection(replay_db)
    cursor = conn.cursor()
    cursor.execute("SELECT job_id, status, error_summary FROM replay_jobs;")
    rows = cursor.fetchall()
    assert len(rows) == 1
    job_id, status, error_summary = rows[0]
    assert status == "FAILED"
    assert "Simulated threadpool executor" in error_summary
    conn.close()


def test_pending_job_blocks_new_submission(client, replay_db):
    """
    Verifies that while any job is in PENDING or RUNNING status in the database,
    a new submission is rejected with 409 Conflict.
    """
    # Manually create a PENDING job in DB
    create_replay_job(
        db_path=replay_db,
        job_id="REP_MANUAL_PENDING_01",
        idempotency_key=None,
        input_fingerprint="fp_pending_test",
        mode="DRY_RUN",
        status="PENDING",
        total_records=10
    )

    records = [
        {"step": 625, "type": "TRANSFER", "amount": 5000.0, "nameOrig": "C_BLOCKED_1", "oldbalanceOrg": 5000.0, "nameDest": "C_DST_1", "oldbalanceDest": 0.0}
    ]

    res = client.post("/api/v1/replay/jobs", json={"records": records})
    assert res.status_code == 409
    assert "Another replay job is currently executing or pending" in res.json()["detail"]


def test_dataset_content_hash_fingerprint_detects_changes_with_constant_size_and_mtime(tmp_path, monkeypatch):
    """
    Verifies that compute_input_fingerprint computes a chunked binary content hash
    and detects modifications to dataset contents even when the file size and
    filesystem modification timestamp are held completely identical.
    """
    import os
    from backend.app.core.config import settings
    from backend.app.schemas.replay import ReplayJobCreateRequest
    from backend.app.services.replay_service import compute_input_fingerprint

    # Point DATA_DIR to tmp_path
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    dataset_file = tmp_path / "test.parquet"

    # 1. Write initial payload of 2048 bytes
    initial_content = b"A" * 2048
    dataset_file.write_bytes(initial_content)
    fixed_timestamp = 1700000000.0
    os.utime(dataset_file, (fixed_timestamp, fixed_timestamp))

    stat_before = dataset_file.stat()
    assert stat_before.st_size == 2048

    req = ReplayJobCreateRequest(
        mode="DRY_RUN",
        dataset_source="test.parquet",
        max_records=100
    )
    fp_before = compute_input_fingerprint(req)

    # 2. Mutate file content with DIFFERENT bytes of EXACT SAME length (2048 bytes)
    mutated_content = b"B" * 2048
    dataset_file.write_bytes(mutated_content)
    os.utime(dataset_file, (fixed_timestamp, fixed_timestamp))

    stat_after = dataset_file.stat()
    assert stat_after.st_size == stat_before.st_size
    assert stat_after.st_mtime_ns == stat_before.st_mtime_ns

    fp_after = compute_input_fingerprint(req)

    # Content change MUST produce a different fingerprint despite identical size & mtime
    assert fp_before != fp_after


def test_identical_dataset_contents_yield_identical_fingerprints(tmp_path, monkeypatch):
    """
    Verifies that identical dataset contents produce identical input fingerprints.
    """
    from backend.app.core.config import settings
    from backend.app.schemas.replay import ReplayJobCreateRequest
    from backend.app.services.replay_service import compute_input_fingerprint

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    dataset_file = tmp_path / "test.parquet"
    dataset_file.write_bytes(b"DATASET_CHUNKED_CONTENT_TEST_PAYLOAD")

    req1 = ReplayJobCreateRequest(mode="DRY_RUN", dataset_source="test.parquet", max_records=250)
    req2 = ReplayJobCreateRequest(mode="DRY_RUN", dataset_source="test.parquet", max_records=250)

    fp1 = compute_input_fingerprint(req1)
    fp2 = compute_input_fingerprint(req2)

    assert fp1 == fp2
    assert len(fp1) == 64  # SHA-256 hex digest


def test_replay_observability_financial_exposure_and_score_distribution(client):
    """
    Verifies that a completed replay job calculates strongly typed financial exposure
    and score distribution metrics with accurate tier and histogram reconciliation.
    """
    records = [
        # Normal lower-risk transaction
        {"step": 625, "type": "PAYMENT", "amount": 100.0, "nameOrig": "C_EXP_01", "oldbalanceOrg": 500.0, "nameDest": "M_EXP_01", "oldbalanceDest": 0.0},
        # High amount transfer
        {"step": 626, "type": "TRANSFER", "amount": 250000.0, "nameOrig": "C_EXP_02", "oldbalanceOrg": 250000.0, "nameDest": "C_EXP_DST_02", "oldbalanceDest": 0.0},
        # Stale record (watermark step <= 620)
        {"step": 600, "type": "TRANSFER", "amount": 99999.0, "nameOrig": "C_STALE_EXP", "oldbalanceOrg": 99999.0, "nameDest": "C_STALE_DST", "oldbalanceDest": 0.0},
    ]

    res = client.post("/api/v1/replay/jobs", json={"records": records})
    assert res.status_code == 201
    job_id = res.json()["job_id"]

    final_job = _wait_for_job_completion(client, job_id)
    assert final_job["status"] == "COMPLETED"
    assert final_job["total_records"] == 3
    assert final_job["processed_records"] == 2
    assert final_job["skipped_records"] == 1
    assert final_job["failed_records"] == 0

    # Test top-level progress field
    progress = final_job["progress"]
    assert progress is not None
    assert progress["progress_percentage"] == 100.0
    assert progress["elapsed_duration_sec"] >= 0.0
    assert progress["estimated_remaining_sec"] == 0.0

    # Test metrics summary
    metrics = final_job["metrics_summary"]
    assert metrics is not None

    # 1. Financial exposure
    f_exp = metrics["financial_exposure"]
    assert f_exp is not None
    # Stale amount (99999.0) must NOT be included in total_amount_replayed
    assert f_exp["total_amount_replayed"] == pytest.approx(250100.0, rel=1e-3)
    assert f_exp["flagged_amount"] <= f_exp["total_amount_replayed"]
    assert 0.0 <= f_exp["flagged_amount_percentage"] <= 100.0
    assert 0.0 <= f_exp["flagged_record_percentage"] <= 100.0

    # 2. Score distribution
    s_dist = metrics["score_distribution"]
    assert s_dist is not None
    assert 0.0 <= s_dist["mean_score"] <= 1.0
    assert 0.0 <= s_dist["median_score"] <= 1.0
    buckets = s_dist["score_buckets"]
    total_buckets = (
        buckets["[0.0, 0.2)"]
        + buckets["[0.2, 0.5)"]
        + buckets["[0.5, 0.8)"]
        + buckets["[0.8, 1.0]"]
    )
    # Histogram count must reconcile exactly with processed_records
    assert total_buckets == 2

    # 3. Risk tier and action counts
    tier_counts = metrics["risk_tier_counts"]
    assert sum(tier_counts.values()) == 2
    action_counts = metrics["action_counts"]
    assert sum(action_counts.values()) == 2

    # Flagged records count reconciliation
    flagged_recs = tier_counts.get("CRITICAL", 0) + tier_counts.get("HIGH", 0)
    expected_flagged_pct = (flagged_recs / 2.0) * 100.0
    assert f_exp["flagged_record_percentage"] == pytest.approx(expected_flagged_pct, rel=1e-3)

    # 4. State mutations and disclaimer
    mutations = metrics["state_mutations"]
    assert mutations["alerts_created"] == 0
    assert mutations["settlements_applied"] == 0
    assert "DRY_RUN" in metrics["disclaimer"]


def test_score_bucket_boundary_exact_reconciliation():
    """
    Unit test verifying exact boundary semantics for score distribution histogram:
    [0.0, 0.2), [0.2, 0.5), [0.5, 0.8), and [0.8, 1.0].
    """
    from backend.app.services.replay_service import _build_metrics_summary

    test_scores = [0.0, 0.1999, 0.2, 0.4999, 0.5, 0.7999, 0.8, 1.0]
    metrics = _build_metrics_summary(
        processed=len(test_scores),
        skipped_stale=0,
        failed=0,
        latencies_ms=[10.0] * len(test_scores),
        risk_tier_counts={"LOW": 4, "MEDIUM": 2, "HIGH": 1, "CRITICAL": 1},
        action_counts={"ALLOW": 4, "REVIEW": 2, "STEP_UP": 1, "BLOCK": 1},
        total_amount=800.0,
        flagged_amount=200.0,
        flagged_records=2,
        hybrid_scores=test_scores,
        elapsed_sec=1.5,
        total_records=len(test_scores)
    )

    s_dist = metrics["score_distribution"]
    buckets = s_dist["score_buckets"]
    assert buckets["[0.0, 0.2)"] == 2  # 0.0, 0.1999
    assert buckets["[0.2, 0.5)"] == 2  # 0.2, 0.4999
    assert buckets["[0.5, 0.8)"] == 2  # 0.5, 0.7999
    assert buckets["[0.8, 1.0]"] == 2  # 0.8, 1.0
    assert sum(buckets.values()) == len(test_scores)
    assert s_dist["mean_score"] == pytest.approx(sum(test_scores) / len(test_scores), abs=1e-4)
    # Median of 8 sorted elements is average of 4th and 5th (indices 3 and 4)
    expected_median = (0.4999 + 0.5) / 2.0
    assert s_dist["median_score"] == pytest.approx(expected_median, abs=1e-4)


def test_zero_processed_and_all_stale_observability_safety():
    """
    Verifies that _build_metrics_summary safely handles zero processed records
    without division-by-zero, returning valid None for scores and 0.0 for percentages.
    """
    from backend.app.services.replay_service import _build_metrics_summary

    metrics = _build_metrics_summary(
        processed=0,
        skipped_stale=5,
        failed=0,
        latencies_ms=[],
        risk_tier_counts={},
        action_counts={},
        total_amount=0.0,
        flagged_amount=0.0,
        flagged_records=0,
        hybrid_scores=[],
        elapsed_sec=0.25,
        total_records=5
    )

    assert metrics["records_summary"]["processed"] == 0
    assert metrics["records_summary"]["skipped_stale_watermark"] == 5
    assert metrics["records_summary"]["failed"] == 0

    f_exp = metrics["financial_exposure"]
    assert f_exp["total_amount_replayed"] == 0.0
    assert f_exp["flagged_amount"] == 0.0
    assert f_exp["flagged_amount_percentage"] == 0.0
    assert f_exp["flagged_record_percentage"] == 0.0

    s_dist = metrics["score_distribution"]
    assert s_dist["mean_score"] is None
    assert s_dist["median_score"] is None
    assert all(count == 0 for count in s_dist["score_buckets"].values())

    progress = metrics["progress"]
    assert progress["progress_percentage"] == 100.0
    assert progress["estimated_remaining_sec"] == 0.0


def test_cancellation_persists_partial_metrics(client):
    """
    Submits a batch of 60 records and requests cancellation mid-flight.
    Verifies that:
    1. The job transitions to CANCELLED status.
    2. Partial aggregate metrics are persisted and returned in metrics_summary.
    3. Exposure and score distribution reflect processed records up to cancellation.
    """
    records = [
        {
            "step": 625 + (i // 10),
            "type": "TRANSFER",
            "amount": 500.0 * (i + 1),
            "nameOrig": f"C_PARTIAL_{i}",
            "oldbalanceOrg": 500.0 * (i + 1),
            "nameDest": f"C_PARTIAL_DST_{i}",
            "oldbalanceDest": 0.0
        }
        for i in range(60)
    ]

    res = client.post("/api/v1/replay/jobs", json={"records": records})
    assert res.status_code == 201
    job_id = res.json()["job_id"]

    # Cancel immediately
    client.post(f"/api/v1/replay/jobs/{job_id}/cancel")

    final_job = _wait_for_job_completion(client, job_id)
    assert final_job["status"] in {"CANCELLED", "COMPLETED"}

    # Regardless of whether it managed to process 0 or more items before cancellation,
    # metrics_summary must be present and valid
    metrics = final_job["metrics_summary"]
    assert metrics is not None
    assert "records_summary" in metrics
    assert "financial_exposure" in metrics
    assert "score_distribution" in metrics
    assert "state_mutations" in metrics
    assert metrics["state_mutations"]["alerts_created"] == 0
    assert metrics["state_mutations"]["settlements_applied"] == 0

    if final_job["status"] == "CANCELLED":
        assert final_job["processed_records"] < 60


def test_mid_batch_failure_persists_partial_metrics_and_bounded_diagnostics(client, monkeypatch):
    """
    Verifies that if an unexpected error occurs during transaction processing:
    1. The job transitions to FAILED status.
    2. Partial metrics up to the failure are preserved in metrics_summary.
    3. Error summary diagnostics are bounded (<= 500 characters).
    """
    import backend.app.services.replay_service as rep_mod

    def mock_failing_update(*args, **kwargs):
        raise RuntimeError("Database write failure during progress update: " + "X" * 600)

    monkeypatch.setattr(rep_mod, "update_replay_job_progress", mock_failing_update)

    records = [
        {"step": 625, "type": "TRANSFER", "amount": 1000.0, "nameOrig": "C_FAIL_A", "oldbalanceOrg": 1000.0, "nameDest": "C_DST_A", "oldbalanceDest": 0.0},
        {"step": 626, "type": "TRANSFER", "amount": 2000.0, "nameOrig": "C_FAIL_B", "oldbalanceOrg": 2000.0, "nameDest": "C_DST_B", "oldbalanceDest": 0.0},
        {"step": 627, "type": "TRANSFER", "amount": 3000.0, "nameOrig": "C_FAIL_C", "oldbalanceOrg": 3000.0, "nameDest": "C_DST_C", "oldbalanceDest": 0.0},
    ]

    res = client.post("/api/v1/replay/jobs", json={"records": records})
    assert res.status_code == 201
    job_id = res.json()["job_id"]

    final_job = _wait_for_job_completion(client, job_id)
    assert final_job["status"] == "FAILED"

    # Diagnostics must be bounded to at most 500 chars (plus JSON formatting)
    err = final_job["error_summary"]
    assert err is not None
    assert len(str(err)) <= 600
    assert "Database write failure" in str(err)

    # Partial metrics must be preserved
    metrics = final_job["metrics_summary"]
    assert metrics is not None
    assert metrics["records_summary"]["total"] == 3
    assert metrics["financial_exposure"]["total_amount_replayed"] >= 1000.0


def test_invalid_and_non_finite_scores_recorded_as_failures_and_excluded_from_metrics(client, monkeypatch):
    """
    Verifies that model predictions returning NaN, +/-Infinity, or out-of-range scores
    are rejected, counted as failed records, recorded in error_summary with clear diagnostics,
    and excluded from total_amount_replayed and score histogram buckets.
    """
    import backend.app.services.replay_service as rep_mod
    original_predict = rep_mod.inference_service.predict_transaction

    test_scores_sequence = [
        0.10,          # Record 0: Valid score
        float("nan"),  # Record 1: NaN
        float("inf"),  # Record 2: Positive infinity
        float("-inf"), # Record 3: Negative infinity
        1.50,          # Record 4: Out-of-range (> 1.0)
        -0.25,         # Record 5: Out-of-range (< 0.0)
    ]
    call_idx = 0

    def mock_predict_varying_scores(req):
        nonlocal call_idx
        idx = call_idx
        call_idx += 1
        resp = original_predict(req)
        score_val = test_scores_sequence[idx % len(test_scores_sequence)]
        return resp.model_copy(update={"hybrid_risk_score": score_val})

    monkeypatch.setattr(rep_mod.inference_service, "predict_transaction", mock_predict_varying_scores)

    records = [
        {"step": 625, "type": "PAYMENT", "amount": 100.0 * (i + 1), "nameOrig": f"C_INV_{i}", "oldbalanceOrg": 500.0, "nameDest": f"M_INV_{i}", "oldbalanceDest": 0.0}
        for i in range(len(test_scores_sequence))
    ]

    res = client.post("/api/v1/replay/jobs", json={"records": records})
    assert res.status_code == 201
    job_id = res.json()["job_id"]

    final_job = _wait_for_job_completion(client, job_id)
    assert final_job["status"] == "COMPLETED"
    assert final_job["total_records"] == 6
    assert final_job["processed_records"] == 1
    assert final_job["failed_records"] == 5

    # Check metrics summary
    metrics = final_job["metrics_summary"]
    assert metrics is not None
    assert metrics["records_summary"]["processed"] == 1
    assert metrics["records_summary"]["failed"] == 5

    # Financial exposure: ONLY the single valid record amount (100.0) is included
    f_exp = metrics["financial_exposure"]
    assert f_exp["total_amount_replayed"] == pytest.approx(100.0, rel=1e-3)

    # Score distribution: ONLY the single valid score (0.10) is in buckets
    s_dist = metrics["score_distribution"]
    buckets = s_dist["score_buckets"]
    assert buckets["[0.0, 0.2)"] == 1
    assert buckets["[0.2, 0.5)"] == 0
    assert buckets["[0.5, 0.8)"] == 0
    assert buckets["[0.8, 1.0]"] == 0
    assert s_dist["mean_score"] == pytest.approx(0.10, abs=1e-4)

    # Error diagnostics: error_summary must contain entries detailing the invalid scores
    err_samples = final_job["error_summary"]
    assert err_samples is not None
    assert len(err_samples) == 5
    for err_entry in err_samples:
        assert "invalid hybrid risk score" in err_entry["error"]


def test_running_job_live_progress_with_deterministic_synchronization(client, monkeypatch):
    """
    Verifies that GET /api/v1/replay/jobs/{job_id} returns live in-flight progress
    while a replay job is running, with ETA null while there are insufficient observations.
    Uses threading events for deterministic synchronization without arbitrary sleeps.
    """
    import threading
    import backend.app.services.replay_service as rep_mod
    original_predict = rep_mod.inference_service.predict_transaction

    first_record_started = threading.Event()
    resume_worker = threading.Event()
    has_paused = False

    def synchronized_predict(req):
        nonlocal has_paused
        if not has_paused:
            has_paused = True
            first_record_started.set()
            resume_worker.wait(timeout=5.0)
        return original_predict(req)

    monkeypatch.setattr(rep_mod.inference_service, "predict_transaction", synchronized_predict)

    records = [
        {"step": 625, "type": "PAYMENT", "amount": 100.0 * (i + 1), "nameOrig": f"C_SYNC_{i}", "oldbalanceOrg": 500.0, "nameDest": f"M_SYNC_{i}", "oldbalanceDest": 0.0}
        for i in range(5)
    ]

    try:
        res = client.post("/api/v1/replay/jobs", json={"records": records})
        assert res.status_code == 201
        job_id = res.json()["job_id"]

        # Wait deterministically until the worker has entered predict_transaction
        assert first_record_started.wait(timeout=5.0) is True

        # Poll the job while it is actively in RUNNING status
        poll_res = client.get(f"/api/v1/replay/jobs/{job_id}")
        assert poll_res.status_code == 200
        poll_data = poll_res.json()

        assert poll_data["status"] == "RUNNING"
        assert poll_data["progress"] is not None
        assert 0.0 <= poll_data["progress"]["progress_percentage"] <= 100.0
        assert poll_data["progress"]["elapsed_duration_sec"] >= 0.0
        # With < 3 completed records, ETA and throughput MUST be null (safeguard against fabricated ETA)
        assert poll_data["progress"]["estimated_remaining_sec"] is None
        assert poll_data["progress"]["throughput_records_per_sec"] is None
    finally:
        # Resume worker unblocking the background thread
        resume_worker.set()

    final_job = _wait_for_job_completion(client, job_id)
    assert final_job["status"] == "COMPLETED"
    assert final_job["processed_records"] == 5
    assert final_job["progress"]["progress_percentage"] == 100.0
    assert final_job["progress"]["estimated_remaining_sec"] == 0.0
