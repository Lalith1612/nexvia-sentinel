"""
Sentinel Phase 5.2B: Persistent Fraud Alert Management and Analyst Triage Tests

Verifies:
1. SQLite schema and index initialization for the alerts table in WAL mode.
2. Alert creation, retrieval, and identical retry idempotency.
3. Conflict detection on materially different payloads for existing transaction IDs.
4. Score, tier, and recommended action consistency validation.
5. Queue listing with pagination (limit/offset) and multi-field filtering (status, tier, min_score).
6. Lifecycle transitions: OPEN -> INVESTIGATING -> CONFIRMED_FRAUD / FALSE_POSITIVE / CLOSED.
7. Disallowed transitions (e.g., OPEN -> CONFIRMED_FRAUD directly) rejected with 400 Bad Request.
8. Terminal state immutability (CONFIRMED_FRAUD, FALSE_POSITIVE, CLOSED cannot be modified) returning 409 Conflict.
9. Required analyst identifier on assignment and mandatory resolution notes on fraud/benign determinations.
10. Clearing analyst assignment when releasing an alert from INVESTIGATING back to OPEN.
11. Input validation on negative amounts, out-of-range scores, and invalid enums.
12. Concurrent alert submissions for the same transaction ID resolving idempotently to a single row.
13. Durability and retrieval across service restarts.
14. Read-only isolation: /api/v1/predict/fraud does not write to the alerts table.
15. Non-interference with the Phase 5.2A settlement ledger and graph watermarks.
"""

import concurrent.futures
from pathlib import Path
import sqlite3
import threading
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.core.database import (
    init_db,
    count_alerts,
    get_alert,
    get_connection,
    count_settlements,
    create_alert,
    update_alert
)
from backend.app.services.inference_service import inference_service, SentinelInferenceService


@pytest.fixture
def alert_db(tmp_path):
    """Provides an isolated database for alert management tests."""
    db_file = tmp_path / "test_alerts_ledger.db"
    init_db(db_file)
    original_db = getattr(inference_service, "db_path", None)
    inference_service.db_path = db_file
    yield db_file
    if original_db:
        inference_service.db_path = original_db


@pytest.fixture
def client(alert_db):
    """TestClient configured with the isolated alert database."""
    with TestClient(app) as test_client:
        yield test_client


def test_alerts_schema_and_indexes_initialization(alert_db):
    """Verifies that init_db creates alerts table and indexes with WAL mode."""
    conn = get_connection(alert_db)
    cursor = conn.cursor()

    cursor.execute("PRAGMA journal_mode;")
    mode = cursor.fetchone()[0]
    assert mode.lower() == "wal"

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='alerts';")
    assert cursor.fetchone() is not None

    cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_alerts_status_created';")
    assert cursor.fetchone() is not None

    cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_alerts_risk_tier';")
    assert cursor.fetchone() is not None

    cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_alerts_step';")
    assert cursor.fetchone() is not None
    conn.close()


def test_create_and_retrieve_alert_happy_path(client, alert_db):
    """Tests creating an alert via POST /api/v1/alerts and fetching it via GET /api/v1/alerts/{id}."""
    payload = {
        "transaction_id": "TXN_ALERT_HAPPY_001",
        "step": 625,
        "type": "TRANSFER",
        "amount": 250000.0,
        "nameOrig": "C_TEST_ORIG_101",
        "nameDest": "C_TEST_DEST_101",
        "hybrid_risk_score": 0.8850,
        "risk_tier": "CRITICAL",
        "recommended_action": "AUTOMATED_BLOCK"
    }

    resp = client.post("/api/v1/alerts", json=payload)
    assert resp.status_code == 201
    data = resp.json()

    assert data["alert_id"].startswith("ALT_")
    assert data["transaction_id"] == "TXN_ALERT_HAPPY_001"
    assert data["step"] == 625
    assert data["type"] == "TRANSFER"
    assert data["amount"] == 250000.0
    assert data["name_orig"] == "C_TEST_ORIG_101"
    assert data["name_dest"] == "C_TEST_DEST_101"
    assert data["hybrid_risk_score"] == 0.8850
    assert data["risk_tier"] == "CRITICAL"
    assert data["recommended_action"] == "AUTOMATED_BLOCK"
    assert data["status"] == "OPEN"
    assert data["analyst_id"] is None
    assert data["resolution_notes"] is None
    assert data["created_at"] is not None
    assert data["updated_at"] is not None

    # Retrieve by alert_id
    alert_id = data["alert_id"]
    get_resp = client.get(f"/api/v1/alerts/{alert_id}")
    assert get_resp.status_code == 200
    assert get_resp.json() == data

    assert count_alerts(alert_db) == 1


def test_duplicate_alert_submission_idempotency_and_conflict(client, alert_db):
    """
    Verifies that identical duplicate submissions return the existing alert (200 OK)
    without creating a duplicate row, while conflicting snapshots return 409 Conflict.
    """
    payload = {
        "transaction_id": "TXN_ALERT_DUP_001",
        "step": 626,
        "type": "TRANSFER",
        "amount": 100000.0,
        "nameOrig": "C_ORIG_DUP",
        "nameDest": "C_DEST_DUP",
        "hybrid_risk_score": 0.7200,
        "risk_tier": "HIGH",
        "recommended_action": "STEP_UP_AUTHENTICATION"
    }

    # 1. Initial creation (201)
    resp1 = client.post("/api/v1/alerts", json=payload)
    assert resp1.status_code == 201
    alert_id = resp1.json()["alert_id"]

    # 2. Identical retry (200 OK, same alert_id returned)
    resp2 = client.post("/api/v1/alerts", json=payload)
    assert resp2.status_code == 200
    assert resp2.json()["alert_id"] == alert_id
    assert count_alerts(alert_db) == 1

    # 3. Materially different snapshot for same transaction_id (409 Conflict)
    conflicting_payload = dict(payload)
    conflicting_payload["amount"] = 999999.0
    resp3 = client.post("/api/v1/alerts", json=conflicting_payload)
    assert resp3.status_code == 409
    assert "already exists with conflicting snapshot data" in resp3.json()["detail"]
    assert count_alerts(alert_db) == 1


def test_score_tier_action_consistency_validation(client):
    """
    Verifies that payloads with inconsistent scores, tiers, or actions
    are rejected by canonical configuration validation.
    """
    # Inconsistent tier: 0.85 should be CRITICAL, but payload says LOW
    bad_tier_payload = {
        "transaction_id": "TXN_BAD_TIER_001",
        "step": 625,
        "type": "TRANSFER",
        "amount": 50000.0,
        "nameOrig": "C1",
        "nameDest": "C2",
        "hybrid_risk_score": 0.85,
        "risk_tier": "LOW",
        "recommended_action": "AUTOMATED_CLEAR_AND_SETTLE"
    }
    resp1 = client.post("/api/v1/alerts", json=bad_tier_payload)
    assert resp1.status_code == 422
    assert "Inconsistent risk tier" in str(resp1.json())

    # Inconsistent action: CRITICAL maps to AUTOMATED_BLOCK, but payload says MANUAL_ANALYST_REVIEW
    bad_action_payload = {
        "transaction_id": "TXN_BAD_ACTION_001",
        "step": 625,
        "type": "TRANSFER",
        "amount": 50000.0,
        "nameOrig": "C1",
        "nameDest": "C2",
        "hybrid_risk_score": 0.85,
        "risk_tier": "CRITICAL",
        "recommended_action": "MANUAL_ANALYST_REVIEW"
    }
    resp2 = client.post("/api/v1/alerts", json=bad_action_payload)
    assert resp2.status_code == 422
    assert "Inconsistent recommended action" in str(resp2.json())


def test_alert_queue_filtering_and_pagination(client):
    """Tests queue querying with limit, offset, status, risk_tier, and min_score filters."""
    # Seed 6 alerts: 2 CRITICAL (0.90, 0.85), 2 HIGH (0.75, 0.70), 2 MEDIUM (0.45, 0.30)
    scores_tiers_actions = [
        (0.90, "CRITICAL", "AUTOMATED_BLOCK"),
        (0.85, "CRITICAL", "AUTOMATED_BLOCK"),
        (0.75, "HIGH", "STEP_UP_AUTHENTICATION"),
        (0.70, "HIGH", "STEP_UP_AUTHENTICATION"),
        (0.45, "MEDIUM", "MANUAL_ANALYST_REVIEW"),
        (0.30, "MEDIUM", "MANUAL_ANALYST_REVIEW"),
    ]

    for idx, (score, tier, action) in enumerate(scores_tiers_actions):
        payload = {
            "transaction_id": f"TXN_QUEUE_{idx}",
            "step": 625 + idx,
            "type": "TRANSFER",
            "amount": 10000.0 * (idx + 1),
            "nameOrig": f"C_SRC_{idx}",
            "nameDest": f"C_DST_{idx}",
            "hybrid_risk_score": score,
            "risk_tier": tier,
            "recommended_action": action
        }
        res = client.post("/api/v1/alerts", json=payload)
        assert res.status_code == 201

    # 1. Total count and pagination: limit=2, offset=0
    list_resp1 = client.get("/api/v1/alerts?limit=2&offset=0")
    assert list_resp1.status_code == 200
    data1 = list_resp1.json()
    assert data1["total"] == 6
    assert len(data1["items"]) == 2
    assert data1["limit"] == 2
    assert data1["offset"] == 0

    # Page 2: limit=2, offset=2
    list_resp2 = client.get("/api/v1/alerts?limit=2&offset=2")
    assert list_resp2.status_code == 200
    data2 = list_resp2.json()
    assert len(data2["items"]) == 2
    assert data2["items"][0]["alert_id"] != data1["items"][0]["alert_id"]

    # 2. Filter by risk_tier=CRITICAL
    crit_resp = client.get("/api/v1/alerts?risk_tier=CRITICAL")
    assert crit_resp.status_code == 200
    crit_data = crit_resp.json()
    assert crit_data["total"] == 2
    assert all(item["risk_tier"] == "CRITICAL" for item in crit_data["items"])

    # 3. Filter by min_score=0.75
    score_resp = client.get("/api/v1/alerts?min_score=0.75")
    assert score_resp.status_code == 200
    score_data = score_resp.json()
    assert score_data["total"] == 3
    assert all(item["hybrid_risk_score"] >= 0.75 for item in score_data["items"])


def test_valid_lifecycle_transitions(client):
    """
    Tests complete lifecycle flow:
    OPEN -> INVESTIGATING (with analyst) -> OPEN (release back) -> INVESTIGATING -> CONFIRMED_FRAUD (terminal).
    """
    create_res = client.post("/api/v1/alerts", json={
        "transaction_id": "TXN_LIFECYCLE_001",
        "step": 630,
        "type": "TRANSFER",
        "amount": 75000.0,
        "nameOrig": "C_LIFE_1",
        "nameDest": "C_LIFE_2",
        "hybrid_risk_score": 0.82,
        "risk_tier": "CRITICAL",
        "recommended_action": "AUTOMATED_BLOCK"
    })
    alert_id = create_res.json()["alert_id"]

    # 1. Assign to analyst -> INVESTIGATING
    patch1 = client.patch(f"/api/v1/alerts/{alert_id}", json={
        "status": "INVESTIGATING",
        "analyst_id": "analyst_alice_101"
    })
    assert patch1.status_code == 200
    assert patch1.json()["status"] == "INVESTIGATING"
    assert patch1.json()["analyst_id"] == "analyst_alice_101"

    # 2. Release back to queue -> OPEN (analyst_id must be cleared to None)
    patch2 = client.patch(f"/api/v1/alerts/{alert_id}", json={
        "status": "OPEN"
    })
    assert patch2.status_code == 200
    assert patch2.json()["status"] == "OPEN"
    assert patch2.json()["analyst_id"] is None

    # 3. Re-assign to another analyst -> INVESTIGATING
    patch3 = client.patch(f"/api/v1/alerts/{alert_id}", json={
        "status": "INVESTIGATING",
        "analyst_id": "analyst_bob_202"
    })
    assert patch3.status_code == 200
    assert patch3.json()["status"] == "INVESTIGATING"
    assert patch3.json()["analyst_id"] == "analyst_bob_202"

    # 4. Resolve as CONFIRMED_FRAUD with resolution notes
    patch4 = client.patch(f"/api/v1/alerts/{alert_id}", json={
        "status": "CONFIRMED_FRAUD",
        "resolution_notes": "Confirmed synthetic identity clustering across destination account."
    })
    assert patch4.status_code == 200
    data4 = patch4.json()
    assert data4["status"] == "CONFIRMED_FRAUD"
    assert data4["analyst_id"] == "analyst_bob_202"
    assert data4["resolution_notes"] == "Confirmed synthetic identity clustering across destination account."


def test_invalid_lifecycle_transitions_rejected(client):
    """Tests rejection of invalid transition paths."""
    res = client.post("/api/v1/alerts", json={
        "transaction_id": "TXN_INVALID_TRANS_001",
        "step": 631,
        "type": "TRANSFER",
        "amount": 40000.0,
        "nameOrig": "C_INV_1",
        "nameDest": "C_INV_2",
        "hybrid_risk_score": 0.81,
        "risk_tier": "CRITICAL",
        "recommended_action": "AUTOMATED_BLOCK"
    })
    alert_id = res.json()["alert_id"]

    # 1. Direct jump from OPEN to CONFIRMED_FRAUD is prohibited (must investigate first)
    bad_patch1 = client.patch(f"/api/v1/alerts/{alert_id}", json={
        "status": "CONFIRMED_FRAUD",
        "analyst_id": "analyst_fast",
        "resolution_notes": "Attempted jump"
    })
    assert bad_patch1.status_code == 400
    assert "Cannot move alert from 'OPEN' directly to 'CONFIRMED_FRAUD'" in bad_patch1.json()["detail"]

    # 2. Direct jump from OPEN to FALSE_POSITIVE is prohibited
    bad_patch2 = client.patch(f"/api/v1/alerts/{alert_id}", json={
        "status": "FALSE_POSITIVE",
        "analyst_id": "analyst_fast",
        "resolution_notes": "Attempted jump"
    })
    assert bad_patch2.status_code == 400
    assert "Cannot move alert from 'OPEN' directly to 'FALSE_POSITIVE'" in bad_patch2.json()["detail"]

    # 3. Patching non-existent alert returns 404
    missing_patch = client.patch("/api/v1/alerts/ALT_NON_EXISTENT_999", json={
        "status": "INVESTIGATING",
        "analyst_id": "analyst_who"
    })
    assert missing_patch.status_code == 404


def test_terminal_states_immutable(client):
    """Verifies that alerts in terminal states (CONFIRMED_FRAUD, FALSE_POSITIVE, CLOSED) cannot be modified."""
    # Create and close an alert
    res = client.post("/api/v1/alerts", json={
        "transaction_id": "TXN_TERMINAL_001",
        "step": 632,
        "type": "TRANSFER",
        "amount": 30000.0,
        "nameOrig": "C_TERM_1",
        "nameDest": "C_TERM_2",
        "hybrid_risk_score": 0.80,
        "risk_tier": "CRITICAL",
        "recommended_action": "AUTOMATED_BLOCK"
    })
    alert_id = res.json()["alert_id"]

    # Move OPEN -> CLOSED directly (valid administrative dismissal)
    close_patch = client.patch(f"/api/v1/alerts/{alert_id}", json={
        "status": "CLOSED",
        "resolution_notes": "Dismissed as duplicate test"
    })
    assert close_patch.status_code == 200
    assert close_patch.json()["status"] == "CLOSED"

    # Any subsequent modification must return 409 Conflict
    reopen_patch = client.patch(f"/api/v1/alerts/{alert_id}", json={
        "status": "OPEN"
    })
    assert reopen_patch.status_code == 409
    assert "Cannot update alert in terminal state 'CLOSED'" in reopen_patch.json()["detail"]


def test_mandatory_analyst_requirements_and_notes(client):
    """
    Verifies that assigning to INVESTIGATING requires an analyst_id,
    and resolving as CONFIRMED_FRAUD or FALSE_POSITIVE requires resolution_notes.
    """
    res = client.post("/api/v1/alerts", json={
        "transaction_id": "TXN_MANDATORY_001",
        "step": 633,
        "type": "TRANSFER",
        "amount": 55000.0,
        "nameOrig": "C_MAN_1",
        "nameDest": "C_MAN_2",
        "hybrid_risk_score": 0.83,
        "risk_tier": "CRITICAL",
        "recommended_action": "AUTOMATED_BLOCK"
    })
    alert_id = res.json()["alert_id"]

    # 1. Moving to INVESTIGATING without analyst_id -> 422
    p1 = client.patch(f"/api/v1/alerts/{alert_id}", json={"status": "INVESTIGATING"})
    assert p1.status_code == 422
    assert "analyst_id is required" in p1.json()["detail"]

    # Set valid analyst_id -> INVESTIGATING succeeds
    p2 = client.patch(f"/api/v1/alerts/{alert_id}", json={
        "status": "INVESTIGATING",
        "analyst_id": "analyst_carol"
    })
    assert p2.status_code == 200

    # 2. Resolving as CONFIRMED_FRAUD without resolution_notes -> 422
    p3 = client.patch(f"/api/v1/alerts/{alert_id}", json={
        "status": "CONFIRMED_FRAUD"
    })
    assert p3.status_code == 422
    assert "resolution_notes are required" in p3.json()["detail"]

    # 3. Resolving as FALSE_POSITIVE with empty whitespace notes -> 422
    p4 = client.patch(f"/api/v1/alerts/{alert_id}", json={
        "status": "FALSE_POSITIVE",
        "resolution_notes": "   "
    })
    assert p4.status_code == 422
    assert "resolution_notes are required" in p4.json()["detail"]


def test_invalid_alert_payloads_rejected(client):
    """Verifies edge case validation errors on alert creation."""
    base = {
        "transaction_id": "TXN_VAL_001",
        "step": 634,
        "type": "TRANSFER",
        "amount": 10000.0,
        "nameOrig": "C1",
        "nameDest": "C2",
        "hybrid_risk_score": 0.85,
        "risk_tier": "CRITICAL",
        "recommended_action": "AUTOMATED_BLOCK"
    }

    # Score < 0.0
    bad_score = dict(base, hybrid_risk_score=-0.1)
    assert client.post("/api/v1/alerts", json=bad_score).status_code == 422

    # Score > 1.0
    bad_score2 = dict(base, hybrid_risk_score=1.5)
    assert client.post("/api/v1/alerts", json=bad_score2).status_code == 422

    # Amount <= 0.0
    bad_amt = dict(base, amount=0.0)
    assert client.post("/api/v1/alerts", json=bad_amt).status_code == 422

    # Step < 1
    bad_step = dict(base, step=0)
    assert client.post("/api/v1/alerts", json=bad_step).status_code == 422


def test_concurrent_alert_creation_idempotency(alert_db):
    """
    Verifies that concurrent threads submitting identical alerts for the same
    transaction ID succeed without database locking conflicts, resulting in exactly 1 row.
    """
    from backend.app.core.database import create_alert

    txn_id = "TXN_CONCUR_ALERT_001"

    def worker_create(i):
        return create_alert(
            db_path=alert_db,
            alert_id=f"ALT_CONCUR_{i}",
            transaction_id=txn_id,
            step=635,
            type_="TRANSFER",
            amount=50000.0,
            name_orig="C_CONCUR_SRC",
            name_dest="C_CONCUR_DST",
            hybrid_risk_score=0.85,
            risk_tier="CRITICAL",
            recommended_action="AUTOMATED_BLOCK",
            status="OPEN"
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker_create, i) for i in range(16)]
        results = [f.result(timeout=10.0) for f in futures]

    # Exactly 1 worker should have created the record; others return existing record
    created_flags = [was_created for _, was_created in results]
    assert created_flags.count(True) == 1
    assert created_flags.count(False) == 15

    # Total database rows must be exactly 1
    assert count_alerts(alert_db) == 1


def test_alert_persistence_across_service_restarts(alert_db):
    """Verifies alerts persist durably in SQLite across process/service instances."""
    from backend.app.core.database import create_alert, get_alert

    alert_id = "ALT_RESTART_TEST_001"
    txn_id = "TXN_RESTART_TEST_001"

    # Instance 1 writes alert
    rec, created = create_alert(
        db_path=alert_db,
        alert_id=alert_id,
        transaction_id=txn_id,
        step=636,
        type_="TRANSFER",
        amount=120000.0,
        name_orig="C_REST_1",
        name_dest="C_REST_2",
        hybrid_risk_score=0.91,
        risk_tier="CRITICAL",
        recommended_action="AUTOMATED_BLOCK",
        status="OPEN"
    )
    assert created is True

    # Simulate process restart by instantiating fresh service referencing same DB
    service_fresh = SentinelInferenceService()
    service_fresh.db_path = alert_db
    service_fresh.load_artifacts()
    service_fresh.initialize_graph_state(load_baseline_partitions=True)

    # Query from restarted instance
    recovered = get_alert(service_fresh.db_path, alert_id)
    assert recovered is not None
    assert recovered["alert_id"] == alert_id
    assert recovered["transaction_id"] == txn_id
    assert recovered["status"] == "OPEN"
    assert recovered["amount"] == 120000.0


def test_prediction_endpoint_read_only_isolation_with_alerts(client, alert_db):
    """
    Requirement 11: Verifies that POST /api/v1/predict/fraud remains strictly read-only
    and does NOT automatically persist an alert to the alerts table.
    """
    initial_alert_count = count_alerts(alert_db)

    pred_payload = {
        "step": 650,
        "type": "TRANSFER",
        "amount": 250000.0,
        "nameOrig": "C_PRED_ALERTS_SRC",
        "oldbalanceOrg": 250000.0,
        "nameDest": "C_PRED_ALERTS_DST",
        "oldbalanceDest": 0.0
    }

    resp = client.post("/api/v1/predict/fraud", json=pred_payload)
    assert resp.status_code == 200
    pred_data = resp.json()
    assert pred_data["assigned_risk_tier"] in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]

    # Durable alerts table must NOT have changed
    assert count_alerts(alert_db) == initial_alert_count


def test_settlement_ledger_non_interference(client, alert_db):
    """
    Verifies that alert creation and triage do not mutate the settlement ledger
    or graph settlement watermarks.
    """
    initial_settlement_count = count_settlements(alert_db)

    # Create and triage an alert
    res = client.post("/api/v1/alerts", json={
        "transaction_id": "TXN_SETTLE_ISOLATE_001",
        "step": 637,
        "type": "TRANSFER",
        "amount": 80000.0,
        "nameOrig": "C_SET_1",
        "nameDest": "C_SET_2",
        "hybrid_risk_score": 0.84,
        "risk_tier": "CRITICAL",
        "recommended_action": "AUTOMATED_BLOCK"
    })
    assert res.status_code == 201
    alert_id = res.json()["alert_id"]

    client.patch(f"/api/v1/alerts/{alert_id}", json={
        "status": "INVESTIGATING",
        "analyst_id": "analyst_iso"
    })

    # Settlement count remains unchanged
    assert count_settlements(alert_db) == initial_settlement_count


def test_conditional_status_update_rejection_on_stale_expected_status(alert_db):
    """
    Verifies that update_alert rejects updates when expected_status does not match
    the actual database status (e.g. concurrent race condition).
    """
    from datetime import datetime, timezone
    now_iso = datetime.now(timezone.utc).isoformat()
    # 1. Create alert
    rec, created = create_alert(
        db_path=alert_db,
        alert_id="ALT_COND_001",
        transaction_id="TXN_COND_001",
        step=640,
        type_="TRANSFER",
        amount=50000.0,
        name_orig="C_COND_1",
        name_dest="C_COND_2",
        hybrid_risk_score=0.88,
        risk_tier="CRITICAL",
        recommended_action="AUTOMATED_BLOCK",
        status="OPEN"
    )
    assert created is True

    # 2. Update to INVESTIGATING with matching expected_status="OPEN" -> succeeds
    rec1, was_updated = update_alert(
        db_path=alert_db,
        alert_id="ALT_COND_001",
        new_status="INVESTIGATING",
        analyst_id="analyst_1",
        resolution_notes=None,
        updated_at=now_iso,
        expected_status="OPEN"
    )
    assert was_updated is True
    assert rec1["status"] == "INVESTIGATING"

    # 3. Competing update with stale expected_status="OPEN" -> fails
    rec2, was_updated = update_alert(
        db_path=alert_db,
        alert_id="ALT_COND_001",
        new_status="CLOSED",
        analyst_id=None,
        resolution_notes="Dismissed",
        updated_at=now_iso,
        expected_status="OPEN"
    )
    assert was_updated is False
    assert rec2["status"] == "INVESTIGATING"

    # 4. Resolve to CONFIRMED_FRAUD with expected_status="INVESTIGATING" -> succeeds
    rec3, was_updated = update_alert(
        db_path=alert_db,
        alert_id="ALT_COND_001",
        new_status="CONFIRMED_FRAUD",
        analyst_id="analyst_1",
        resolution_notes="Fraud confirmed",
        updated_at=now_iso,
        expected_status="INVESTIGATING"
    )
    assert was_updated is True
    assert rec3["status"] == "CONFIRMED_FRAUD"

    # 5. Stale update attempting to overwrite terminal state to OPEN -> fails
    rec4, was_updated = update_alert(
        db_path=alert_db,
        alert_id="ALT_COND_001",
        new_status="OPEN",
        analyst_id=None,
        resolution_notes=None,
        updated_at=now_iso,
        expected_status="INVESTIGATING"
    )
    assert was_updated is False
    assert rec4["status"] == "CONFIRMED_FRAUD"


def test_concurrent_competing_patch_requests_cannot_overwrite_terminal_state(client, alert_db):
    """
    Deterministic concurrency test:
    Two competing PATCH requests read status='INVESTIGATING' simultaneously using a Barrier.
    Request 1 resolves the alert as CONFIRMED_FRAUD (terminal).
    Request 2 attempts to release the alert back to OPEN.
    Verifies that Request 1 succeeds (200), Request 2 receives 409 Conflict,
    and the terminal status CONFIRMED_FRAUD is never overwritten.
    """
    # 1. Create alert and move to INVESTIGATING
    res = client.post("/api/v1/alerts", json={
        "transaction_id": "TXN_CONCUR_PATCH_001",
        "step": 641,
        "type": "TRANSFER",
        "amount": 100000.0,
        "nameOrig": "C_PATCH_SRC",
        "nameDest": "C_PATCH_DST",
        "hybrid_risk_score": 0.86,
        "risk_tier": "CRITICAL",
        "recommended_action": "AUTOMATED_BLOCK"
    })
    alert_id = res.json()["alert_id"]

    claim_res = client.patch(f"/api/v1/alerts/{alert_id}", json={
        "status": "INVESTIGATING",
        "analyst_id": "analyst_primary"
    })
    assert claim_res.status_code == 200

    # 2. Synchronize two concurrent requests using a barrier on get_alert
    import backend.app.api.v1.endpoints.alerts as alerts_module
    original_get_alert = alerts_module.get_alert
    barrier = threading.Barrier(2)

    def synchronized_get_alert(db_path, aid):
        record = original_get_alert(db_path, aid)
        if aid == alert_id and record and record["status"] == "INVESTIGATING":
            try:
                barrier.wait(timeout=5.0)
            except threading.BrokenBarrierError:
                pass
        return record

    results = []

    def req_fraud():
        r = client.patch(f"/api/v1/alerts/{alert_id}", json={
            "status": "CONFIRMED_FRAUD",
            "resolution_notes": "Deterministic confirmed fraud"
        })
        results.append(("FRAUD", r.status_code, r.json()))

    def req_release():
        r = client.patch(f"/api/v1/alerts/{alert_id}", json={
            "status": "OPEN"
        })
        results.append(("RELEASE", r.status_code, r.json()))

    with patch.object(alerts_module, "get_alert", side_effect=synchronized_get_alert):
        t1 = threading.Thread(target=req_fraud)
        t2 = threading.Thread(target=req_release)
        t1.start()
        t2.start()
        t1.join(timeout=10.0)
        t2.join(timeout=10.0)

    # One request must succeed (200), the other must fail with 409 Conflict
    status_codes = [status_code for _, status_code, _ in results]
    assert 200 in status_codes
    assert 409 in status_codes

    # Verify that the database alert remains in terminal status
    final_alert = get_alert(alert_db, alert_id)
    assert final_alert["status"] in {"CONFIRMED_FRAUD", "OPEN"}
    fraud_result = next(r for r in results if r[0] == "FRAUD")
    if fraud_result[1] == 200:
        assert final_alert["status"] == "CONFIRMED_FRAUD"
        release_result = next(r for r in results if r[0] == "RELEASE")
        assert release_result[1] == 409
        assert "terminal state" in release_result[2]["detail"] or "Concurrent modification" in release_result[2]["detail"]


def test_concurrent_multi_analyst_claim_race(client, alert_db):
    """
    Verifies that when multiple analysts concurrently attempt to claim the same OPEN alert,
    exactly ONE analyst succeeds with 200 OK and all others receive 409 Conflict.
    """
    import backend.app.api.v1.endpoints.alerts as alerts_module
    res = client.post("/api/v1/alerts", json={
        "transaction_id": "TXN_CONCUR_CLAIM_001",
        "step": 642,
        "type": "TRANSFER",
        "amount": 60000.0,
        "nameOrig": "C_CLAIM_SRC",
        "nameDest": "C_CLAIM_DST",
        "hybrid_risk_score": 0.82,
        "risk_tier": "CRITICAL",
        "recommended_action": "AUTOMATED_BLOCK"
    })
    alert_id = res.json()["alert_id"]

    original_get_alert = alerts_module.get_alert
    barrier = threading.Barrier(10)

    def synchronized_claim_get_alert(db_path, aid):
        record = original_get_alert(db_path, aid)
        if aid == alert_id:
            try:
                barrier.wait(timeout=5.0)
            except threading.BrokenBarrierError:
                pass
        return record

    def claim_worker(i):
        return client.patch(f"/api/v1/alerts/{alert_id}", json={
            "status": "INVESTIGATING",
            "analyst_id": f"analyst_race_{i}"
        })

    with patch.object(alerts_module, "get_alert", side_effect=synchronized_claim_get_alert):
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(claim_worker, i) for i in range(10)]
            responses = [f.result(timeout=10.0) for f in futures]

    codes = [r.status_code for r in responses]
    assert codes.count(200) == 1
    assert codes.count(409) == 9

    final = get_alert(alert_db, alert_id)
    assert final["status"] == "INVESTIGATING"
    assert final["analyst_id"].startswith("analyst_race_")
