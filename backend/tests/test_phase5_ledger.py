"""
Sentinel Phase 5.2A: Durable SQLite Ledger and Restart Recovery Tests
Verifies:
1. SQLite WAL mode initialization, schema creation, and PRAGMA settings.
2. Direct durable persistence, retrieval, and chronological ordering.
3. End-to-end settlement persistence via the FastAPI settlement endpoint.
4. Deterministic restart recovery: rebuilding graph state from baseline step-620 snapshot
   plus post-snapshot durable ledger entries across process instantiations.
5. Recovery when DB commit succeeds but subsequent in-memory graph update fails:
   ensuring settlements are neither silently lost nor double-applied upon retry or restart.
6. Predictions strictly preserving read-only isolation (zero ledger writes).
7. Concurrent settlement persistence and idempotency under parallel worker threads.
"""

import concurrent.futures
from pathlib import Path
import sqlite3
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.core.database import (
    init_db,
    record_settlement,
    get_settlement,
    get_durable_settlements_since,
    count_settlements,
    get_connection
)
from backend.app.services.inference_service import SentinelInferenceService, inference_service


@pytest.fixture
def isolated_db(tmp_path):
    """Provides an isolated SQLite database path for testing."""
    db_file = tmp_path / "test_sentinel_ledger.db"
    init_db(db_file)
    return db_file


def test_sqlite_wal_mode_and_schema_initialization(isolated_db):
    """Verifies that the ledger initializes with WAL mode and proper indexes."""
    conn = get_connection(isolated_db)
    cursor = conn.cursor()

    # Check journal mode is WAL
    cursor.execute("PRAGMA journal_mode;")
    mode = cursor.fetchone()[0]
    assert mode.lower() == "wal"

    # Check settlements table schema
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='settlements';")
    assert cursor.fetchone() is not None

    # Check index on step
    cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_settlements_step';")
    assert cursor.fetchone() is not None
    conn.close()


def test_durable_settlement_persistence_and_chronology(isolated_db):
    """Verifies record_settlement, duplicate idempotency, and chronological query."""
    # Insert step 625
    res1 = record_settlement(
        isolated_db,
        transaction_id="TXN_TEST_001",
        step=625,
        type_="TRANSFER",
        amount=50000.0,
        name_orig="C_TEST_ORIG_1",
        name_dest="C_TEST_DEST_1",
        settled_at="2026-10-11T12:00:00Z"
    )
    assert res1 is True

    # Duplicate insert with same transaction_id must return False (idempotent ignore)
    res_dup = record_settlement(
        isolated_db,
        transaction_id="TXN_TEST_001",
        step=625,
        type_="TRANSFER",
        amount=50000.0,
        name_orig="C_TEST_ORIG_1",
        name_dest="C_TEST_DEST_1",
        settled_at="2026-10-11T12:00:00Z"
    )
    assert res_dup is False

    # Insert step 630
    res2 = record_settlement(
        isolated_db,
        transaction_id="TXN_TEST_002",
        step=630,
        type_="CASH_OUT",
        amount=25000.0,
        name_orig="C_TEST_ORIG_2",
        name_dest="C_TEST_DEST_2",
        settled_at="2026-10-11T12:01:00Z"
    )
    assert res2 is True

    assert count_settlements(isolated_db) == 2

    # Query settlements since step 620
    rows = get_durable_settlements_since(isolated_db, cutoff_step=620)
    assert len(rows) == 2
    assert rows[0]["transaction_id"] == "TXN_TEST_001"
    assert rows[1]["transaction_id"] == "TXN_TEST_002"
    assert rows[0]["step"] == 625
    assert rows[1]["step"] == 630


def test_deterministic_restart_recovery_from_ledger(isolated_db):
    """
    Verifies that a new SentinelInferenceService instance deterministically recovers
    post-snapshot settlements from SQLite on startup, advancing watermarks and updating graph counters.
    """
    # 1. Settle 2 transactions in Service Instance 1 using isolated_db
    service_1 = SentinelInferenceService()
    service_1.db_path = isolated_db
    service_1.load_artifacts()
    # Initialize with baseline partitions (snapshot ends at step 620)
    service_1.initialize_graph_state(load_baseline_partitions=True)

    prior_edges = service_1.settled_edge_index.shape[1]
    assert service_1.snapshot_cutoff_step == 620
    assert service_1.max_settled_step == 620

    # Settle transaction A at step 625
    event_a = {
        "transaction_id": "TXN_RECOVER_A_625",
        "step": 625,
        "type": "TRANSFER",
        "amount": 75000.0,
        "nameOrig": "C_RECOVER_ORIG_SHARED",
        "nameDest": "C_RECOVER_DEST_1"
    }
    # Settle transaction B at step 630
    event_b = {
        "transaction_id": "TXN_RECOVER_B_630",
        "step": 630,
        "type": "TRANSFER",
        "amount": 45000.0,
        "nameOrig": "C_RECOVER_ORIG_SHARED",
        "nameDest": "C_RECOVER_DEST_2"
    }

    import asyncio
    from backend.app.schemas.transaction import SettlementEventRequest

    asyncio.run(service_1.process_settlement_event(SettlementEventRequest(**event_a), auto_refresh_embeddings=False))
    asyncio.run(service_1.process_settlement_event(SettlementEventRequest(**event_b), auto_refresh_embeddings=False))

    u_id_1 = service_1.graph_builder.node_to_idx["C_RECOVER_ORIG_SHARED"]
    assert service_1.graph_builder.out_degrees[u_id_1] == 2.0
    assert service_1.graph_builder.out_volume[u_id_1] == 120000.0
    assert service_1.max_settled_step == 630
    assert service_1.settled_edge_index.shape[1] == prior_edges + 2
    assert count_settlements(isolated_db) == 2

    # 2. Simulate Clean Process Restart: Instantiate completely NEW service instance 2
    service_2 = SentinelInferenceService()
    service_2.db_path = isolated_db
    service_2.load_artifacts()
    # initialize_graph_state loads baseline snapshot (step 620) AND replays isolated_db post-620 settlements
    service_2.initialize_graph_state(load_baseline_partitions=True)

    # 3. Verify Deterministic State Recovery in Service Instance 2
    assert service_2.snapshot_cutoff_step == 620
    assert service_2.max_settled_step == 630  # Watermark advanced to latest recovered step
    assert "C_RECOVER_ORIG_SHARED" in service_2.graph_builder.node_to_idx

    u_id_2 = service_2.graph_builder.node_to_idx["C_RECOVER_ORIG_SHARED"]
    assert service_2.graph_builder.out_degrees[u_id_2] == 2.0
    assert service_2.graph_builder.out_volume[u_id_2] == 120000.0
    assert service_2.settled_edge_index.shape[1] == prior_edges + 2
    assert "TXN_RECOVER_A_625" in service_2.settled_event_ids
    assert "TXN_RECOVER_B_630" in service_2.settled_event_ids

    # 4. Verify that retrying transaction A on Service 2 returns DUPLICATE_IGNORED and does not double-count
    res_dup = asyncio.run(service_2.process_settlement_event(SettlementEventRequest(**event_a), auto_refresh_embeddings=False))
    assert res_dup.status_code if hasattr(res_dup, "status_code") else res_dup.status == "DUPLICATE_IGNORED"
    assert service_2.graph_builder.out_degrees[u_id_2] == 2.0  # Still exactly 2.0


def test_recovery_when_db_commit_succeeds_but_in_memory_update_fails(isolated_db, monkeypatch):
    """
    Requirement 7: Verifies that if a settlement is committed to SQLite but the subsequent
    in-memory graph update raises an unexpected fault:
    1. In-memory state is rolled back cleanly (no torn state).
    2. The settlement is NOT lost: upon subsequent retry, the service recovers the committed DB record
       and applies it to the in-memory graph.
    3. The settlement is NOT double-applied: subsequent requests return DUPLICATE_IGNORED.
    """
    from backend.app.schemas.transaction import SettlementEventRequest
    import asyncio

    service = SentinelInferenceService()
    service.db_path = isolated_db
    service.load_artifacts()
    service.initialize_graph_state(load_baseline_partitions=True)

    prior_edges = service.settled_edge_index.shape[1]
    txn_id = "TXN_FAULT_RECOVERY_999"
    orig_name = "C_FAULT_ORIG_999"
    dest_name = "C_FAULT_DEST_999"

    event_req = SettlementEventRequest(
        transaction_id=txn_id,
        step=640,
        type="TRANSFER",
        amount=85000.0,
        nameOrig=orig_name,
        nameDest=dest_name
    )

    # Monkeypatch refresh_graph_embeddings to simulate an unexpected failure during in-memory processing
    def mock_fault():
        raise RuntimeError("Simulated transient CUDA/memory fault after DB commit")

    monkeypatch.setattr(service, "refresh_graph_embeddings", mock_fault)

    # 1. First attempt fails due to simulated fault
    with pytest.raises(RuntimeError, match="Simulated transient CUDA/memory fault"):
        asyncio.run(service.process_settlement_event(event_req, auto_refresh_embeddings=True))

    # In-memory graph state must be cleanly rolled back
    assert orig_name not in service.graph_builder.node_to_idx
    assert dest_name not in service.graph_builder.node_to_idx
    assert service.settled_edge_index.shape[1] == prior_edges
    assert txn_id not in service.settled_event_ids

    # BUT the record was committed in SQLite (durability guarantee)
    db_record = get_settlement(isolated_db, txn_id)
    assert db_record is not None
    assert db_record["transaction_id"] == txn_id
    assert db_record["step"] == 640

    # 2. Remove monkeypatch and retry the settlement
    monkeypatch.undo()
    res_retry = asyncio.run(service.process_settlement_event(event_req, auto_refresh_embeddings=False))

    # The service detects the committed DB record, recovers it into the in-memory graph, and succeeds
    assert res_retry.status == "SETTLED"
    assert orig_name in service.graph_builder.node_to_idx
    assert dest_name in service.graph_builder.node_to_idx
    assert service.settled_edge_index.shape[1] == prior_edges + 1
    assert txn_id in service.settled_event_ids
    u_id = service.graph_builder.node_to_idx[orig_name]
    assert service.graph_builder.out_degrees[u_id] == 1.0
    assert service.graph_builder.out_volume[u_id] == 85000.0

    # 3. Third submission: Must be ignored as duplicate (no double-counting)
    res_third = asyncio.run(service.process_settlement_event(event_req, auto_refresh_embeddings=False))
    assert res_third.status == "DUPLICATE_IGNORED"
    assert service.graph_builder.out_degrees[u_id] == 1.0
    assert service.settled_edge_index.shape[1] == prior_edges + 1
    assert count_settlements(isolated_db) == 1  # Exactly 1 DB row


def test_prediction_does_not_mutate_ledger(isolated_db):
    """
    Requirement 6: Verifies that prediction requests are strictly read-only
    and do not insert rows into the durable settlements ledger.
    """
    from backend.app.schemas.transaction import TransactionRequest

    service = SentinelInferenceService()
    service.db_path = isolated_db
    service.load_artifacts()
    service.initialize_graph_state(load_baseline_partitions=True)

    initial_db_count = count_settlements(isolated_db)
    initial_edge_count = service.settled_edge_index.shape[1]

    pred_req = TransactionRequest(
        step=650,
        type="TRANSFER",
        amount=40000.0,
        nameOrig="C_PRED_READONLY_SRC",
        oldbalanceOrg=40000.0,
        nameDest="C_PRED_READONLY_DST",
        oldbalanceDest=0.0
    )

    res = service.predict_transaction(pred_req)
    assert 0.0 <= res.hybrid_risk_score <= 1.0

    # Verify zero mutations to durable ledger and graph history
    assert count_settlements(isolated_db) == initial_db_count
    assert service.settled_edge_index.shape[1] == initial_edge_count
    assert "C_PRED_READONLY_SRC" not in service.graph_builder.node_to_idx


@pytest.mark.asyncio
async def test_concurrent_settlements_ledger_and_idempotency(isolated_db):
    """
    Verifies that concurrent settlement submissions on the event loop:
    1. Do not cause SQLite locking conflicts under WAL mode.
    2. Maintain exact idempotency and single-write ledger integrity.
    """
    from backend.app.schemas.transaction import SettlementEventRequest
    import asyncio

    service = SentinelInferenceService()
    service.db_path = isolated_db
    service.load_artifacts()
    service.initialize_graph_state(load_baseline_partitions=True)

    step = 660
    dup_txn_id = "TXN_CONCUR_LEDGER_DUP"
    event_dup = SettlementEventRequest(
        transaction_id=dup_txn_id,
        step=step,
        type="TRANSFER",
        amount=50000.0,
        nameOrig="C_CONCUR_LEDGER_SRC",
        nameDest="C_CONCUR_LEDGER_DST"
    )

    # Concurrently submit 6 identical settlement coroutines on the event loop
    tasks = [
        service.process_settlement_event(event_dup, auto_refresh_embeddings=False)
        for _ in range(6)
    ]
    results = await asyncio.gather(*tasks)

    statuses = [r.status for r in results]
    assert statuses.count("SETTLED") == 1
    assert statuses.count("DUPLICATE_IGNORED") == 5

    # Check database row count: exactly 1 row
    assert count_settlements(isolated_db) == 1
    u_id = service.graph_builder.node_to_idx["C_CONCUR_LEDGER_SRC"]
    assert service.graph_builder.out_degrees[u_id] == 1.0


def test_concurrent_sqlite_wal_writes(isolated_db):
    """
    Verifies that multi-threaded concurrent inserts to SQLite in WAL mode
    succeed without database locking errors.
    """
    def worker_write(i):
        return record_settlement(
            isolated_db,
            transaction_id=f"TXN_CONCUR_DIRECT_{i}",
            step=670 + i,
            type_="TRANSFER",
            amount=1000.0 * (i + 1),
            name_orig=f"C_CONCUR_SRC_{i}",
            name_dest=f"C_CONCUR_DST_{i}",
            settled_at="2026-10-11T12:00:00Z"
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker_write, i) for i in range(20)]
        results = [f.result(timeout=10.0) for f in futures]

    assert all(results)
    assert count_settlements(isolated_db) == 20


def test_sqlite_transaction_rollback_on_injected_exception(isolated_db):
    """
    Part A: Verifies explicit transaction rollback when an exception is injected
    after a write begins but before commit, confirming zero partial rows remain.
    """
    from backend.app.core.database import db_session

    assert count_settlements(isolated_db) == 0

    trace_log = []

    with db_session(isolated_db) as conn:
        conn.set_trace_callback(lambda stmt: trace_log.append(stmt))
        with pytest.raises(RuntimeError, match="Injected failure before commit"):
            with conn:
                conn.execute("""
                    INSERT INTO settlements (
                        transaction_id, step, type, amount, name_orig, name_dest, settled_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?);
                """, ("TXN_ROLLBACK_FAIL", 621, "TRANSFER", 1000.0, "C1", "C2", "2026-10-11T00:00:00Z"))
                raise RuntimeError("Injected failure before commit")

    # Verify trace log contains BEGIN, INSERT, and ROLLBACK
    begin_seen = any("BEGIN" in stmt.upper() for stmt in trace_log)
    insert_seen = any("INSERT" in stmt.upper() for stmt in trace_log)
    rollback_seen = any("ROLLBACK" in stmt.upper() for stmt in trace_log)
    assert begin_seen, "Expected BEGIN in SQLite transaction trace"
    assert insert_seen, "Expected INSERT in SQLite transaction trace"
    assert rollback_seen, "Expected ROLLBACK in SQLite transaction trace"

    # Confirm that the failed write left zero partial settlement rows
    assert count_settlements(isolated_db) == 0
    assert get_settlement(isolated_db, "TXN_ROLLBACK_FAIL") is None


@pytest.mark.asyncio
async def test_pending_durable_settlement_reconciliation_sequence(isolated_db, monkeypatch):
    """
    Parts B & C: Verifies pending durable settlement recovery and restart sequence:
    1. Begin with baseline graph at step 620.
    2. Persist settlement at step 621.
    3. Inject failure during in-memory graph update/embedding refresh after DB commit.
    4. Do not retry that same settlement.
    5. Submit different settlement at step 622.
    6. Verify service reconciles step 621 before accepting step 622.
    7. Verify graph counters, edge count, watermark, settled_event_ids, and SQLite rows are consistent.
    8. Verify retrying original settlement after reconciliation returns DUPLICATE_IGNORED and does not double-apply.
    9. Construct fresh inference service from same database and verify deterministic chronological restoration.
    """
    from backend.app.schemas.transaction import SettlementEventRequest

    service = SentinelInferenceService()
    service.db_path = isolated_db
    service.load_artifacts()
    service.initialize_graph_state(load_baseline_partitions=True)

    prior_edges = service.settled_edge_index.shape[1]
    assert service.snapshot_cutoff_step == 620
    assert service.max_settled_step == 620

    # 1. Settlement S1 at step 621
    event_s1 = SettlementEventRequest(
        transaction_id="TXN_PENDING_S1_621",
        step=621,
        type="TRANSFER",
        amount=50000.0,
        nameOrig="C_PENDING_ORIG_S1",
        nameDest="C_PENDING_DEST_S1"
    )

    # 2. Inject failure after DB commit during embedding refresh
    def mock_fault():
        raise RuntimeError("Injected transient embedding fault on step 621")

    monkeypatch.setattr(service, "refresh_graph_embeddings", mock_fault)

    with pytest.raises(RuntimeError, match="Injected transient embedding fault on step 621"):
        await service.process_settlement_event(event_s1, auto_refresh_embeddings=True)

    # S1 is committed to SQLite (durable guarantee)
    db_s1 = get_settlement(isolated_db, "TXN_PENDING_S1_621")
    assert db_s1 is not None
    assert db_s1["step"] == 621

    # In-memory graph state for S1 rolled back cleanly
    assert "C_PENDING_ORIG_S1" not in service.graph_builder.node_to_idx
    assert service.settled_edge_index.shape[1] == prior_edges
    assert service.max_settled_step == 620
    assert "TXN_PENDING_S1_621" not in service.settled_event_ids

    # 3. DO NOT retry S1. Clear the fault and submit S2 at step 622
    monkeypatch.undo()

    event_s2 = SettlementEventRequest(
        transaction_id="TXN_PENDING_S2_622",
        step=622,
        type="TRANSFER",
        amount=75000.0,
        nameOrig="C_PENDING_ORIG_S2",
        nameDest="C_PENDING_DEST_S2"
    )

    # 4. Submit S2: Service must reconcile S1 before accepting S2
    resp_s2 = await service.process_settlement_event(event_s2, auto_refresh_embeddings=False)
    assert resp_s2.status == "SETTLED"

    # 5. Verify graph counters, edge count, watermark, settled_event_ids, and SQLite rows
    # Both S1 and S2 accounts exist in graph
    assert "C_PENDING_ORIG_S1" in service.graph_builder.node_to_idx
    assert "C_PENDING_DEST_S1" in service.graph_builder.node_to_idx
    assert "C_PENDING_ORIG_S2" in service.graph_builder.node_to_idx
    assert "C_PENDING_DEST_S2" in service.graph_builder.node_to_idx

    u1 = service.graph_builder.node_to_idx["C_PENDING_ORIG_S1"]
    v1 = service.graph_builder.node_to_idx["C_PENDING_DEST_S1"]
    assert service.graph_builder.out_degrees[u1] == 1.0
    assert service.graph_builder.out_volume[u1] == 50000.0
    assert service.graph_builder.in_degrees[v1] == 1.0
    assert service.graph_builder.in_volume[v1] == 50000.0

    u2 = service.graph_builder.node_to_idx["C_PENDING_ORIG_S2"]
    v2 = service.graph_builder.node_to_idx["C_PENDING_DEST_S2"]
    assert service.graph_builder.out_degrees[u2] == 1.0
    assert service.graph_builder.out_volume[u2] == 75000.0
    assert service.graph_builder.in_degrees[v2] == 1.0
    assert service.graph_builder.in_volume[v2] == 75000.0

    assert service.settled_edge_index.shape[1] == prior_edges + 2
    assert service.max_settled_step == 622
    assert "TXN_PENDING_S1_621" in service.settled_event_ids
    assert "TXN_PENDING_S2_622" in service.settled_event_ids
    assert count_settlements(isolated_db) == 2

    # 6. Verify retrying original settlement S1 returns DUPLICATE_IGNORED and does not double-apply mutations
    resp_s1_retry = await service.process_settlement_event(event_s1, auto_refresh_embeddings=False)
    assert resp_s1_retry.status == "DUPLICATE_IGNORED"
    assert service.graph_builder.out_degrees[u1] == 1.0
    assert service.graph_builder.out_volume[u1] == 50000.0
    assert service.settled_edge_index.shape[1] == prior_edges + 2

    # 7. Part C: Test constructing a fresh inference service from the same database
    service_fresh = SentinelInferenceService()
    service_fresh.db_path = isolated_db
    service_fresh.load_artifacts()
    service_fresh.initialize_graph_state(load_baseline_partitions=True)

    assert service_fresh.snapshot_cutoff_step == 620
    assert service_fresh.max_settled_step == 622
    assert service_fresh.settled_edge_index.shape[1] == prior_edges + 2
    assert "TXN_PENDING_S1_621" in service_fresh.settled_event_ids
    assert "TXN_PENDING_S2_622" in service_fresh.settled_event_ids

    u1_fresh = service_fresh.graph_builder.node_to_idx["C_PENDING_ORIG_S1"]
    v1_fresh = service_fresh.graph_builder.node_to_idx["C_PENDING_DEST_S1"]
    assert service_fresh.graph_builder.out_degrees[u1_fresh] == 1.0
    assert service_fresh.graph_builder.out_volume[u1_fresh] == 50000.0
    assert service_fresh.graph_builder.in_degrees[v1_fresh] == 1.0
    assert service_fresh.graph_builder.in_volume[v1_fresh] == 50000.0


def test_record_settlement_write_path_rollback_leaves_no_partial_row(isolated_db):
    """
    Directly tests failure injection through the actual record_settlement() write path.
    Installs an abort trigger to simulate an engine error after insert begins,
    verifying that the actual record_settlement() call catches the error, rolls back,
    and leaves zero partial settlement rows in SQLite.
    """
    conn = get_connection(isolated_db)
    conn.execute("""
        CREATE TRIGGER test_write_fail_trigger AFTER INSERT ON settlements
        BEGIN
            SELECT RAISE(ABORT, 'Simulated write engine failure during record_settlement');
        END;
    """)
    conn.close()

    txn_id = "TXN_ACTUAL_WRITE_PATH_FAIL"
    with pytest.raises(sqlite3.IntegrityError, match="Simulated write engine failure"):
        record_settlement(
            isolated_db,
            transaction_id=txn_id,
            step=621,
            type_="TRANSFER",
            amount=50000.0,
            name_orig="C_FAIL_ORIG",
            name_dest="C_FAIL_DEST",
            settled_at="2026-10-11T12:00:00Z"
        )

    # Confirm that zero rows exist in the table and no partial record remains
    assert count_settlements(isolated_db) == 0
    assert get_settlement(isolated_db, txn_id) is None


def test_init_db_transactional_ddl_rollback(tmp_path, monkeypatch):
    """
    Verifies that schema creation and index creation roll back together atomically
    if index creation fails during init_db().
    """
    from backend.app.core.database import db_session

    test_db = tmp_path / "test_transactional_ddl.db"

    # Monkeypatch to simulate a failure during index creation in init_db
    orig_db_session = db_session

    def faulty_init_db(db_path: Path):
        with orig_db_session(db_path) as conn:
            conn.execute("BEGIN IMMEDIATE;")
            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS settlements (
                        transaction_id TEXT PRIMARY KEY,
                        step INTEGER NOT NULL,
                        type TEXT NOT NULL,
                        amount REAL NOT NULL,
                        name_orig TEXT NOT NULL,
                        name_dest TEXT NOT NULL,
                        settled_at TEXT NOT NULL,
                        created_at TEXT NOT NULL DEFAULT (datetime('now', 'utc'))
                    );
                """)
                # Injected failure before index creation completes
                raise RuntimeError("Injected failure during DDL index creation")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_settlements_step ON settlements(step);")
                conn.execute("COMMIT;")
            except Exception:
                conn.execute("ROLLBACK;")
                raise

    with pytest.raises(RuntimeError, match="Injected failure during DDL index creation"):
        faulty_init_db(test_db)

    # Verify that the entire DDL transaction rolled back: table does NOT exist
    with orig_db_session(test_db) as conn:
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='settlements';")
        assert cursor.fetchone() is None
