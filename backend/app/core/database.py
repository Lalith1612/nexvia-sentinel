"""
Sentinel SQLite Durable Ledger & WAL Connection Manager
Provides crash-resilient persistence for confirmed settlements.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sqlite3
from typing import Any, Dict, Generator, List, Optional, Tuple

logger = logging.getLogger("sentinel.database")


def get_connection(db_path: Path) -> sqlite3.Connection:
    """
    Creates and configures a thread-local SQLite connection with WAL mode
    and strict foreign keys.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        str(db_path),
        timeout=10.0,
        check_same_thread=False
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


@contextmanager
def db_session(db_path: Path) -> Generator[sqlite3.Connection, None, None]:
    """Context manager for SQLite connections with transaction management."""
    conn = get_connection(db_path)
    try:
        yield conn
    finally:
        conn.close()


def init_db(db_path: Path) -> None:
    """
    Initializes durable tables and indexes in SQLite.
    Uses explicit transactional DDL (BEGIN IMMEDIATE ... COMMIT / ROLLBACK)
    to guarantee that schema and index creation roll back together atomically
    if any step fails.
    Also executes startup crash reconciliation for interrupted replay jobs.
    """
    with db_session(db_path) as conn:
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
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_settlements_step ON settlements(step);
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id TEXT PRIMARY KEY,
                    transaction_id TEXT NOT NULL UNIQUE,
                    step INTEGER NOT NULL CHECK (step >= 1),
                    type TEXT NOT NULL,
                    amount REAL NOT NULL CHECK (amount > 0.0),
                    name_orig TEXT NOT NULL,
                    name_dest TEXT NOT NULL,
                    hybrid_risk_score REAL NOT NULL CHECK (hybrid_risk_score >= 0.0 AND hybrid_risk_score <= 1.0),
                    risk_tier TEXT NOT NULL CHECK (risk_tier IN ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW')),
                    recommended_action TEXT NOT NULL CHECK (recommended_action IN (
                        'AUTOMATED_BLOCK',
                        'STEP_UP_AUTHENTICATION',
                        'MANUAL_ANALYST_REVIEW',
                        'AUTOMATED_CLEAR_AND_SETTLE'
                    )),
                    status TEXT NOT NULL CHECK (status IN (
                        'OPEN',
                        'INVESTIGATING',
                        'CONFIRMED_FRAUD',
                        'FALSE_POSITIVE',
                        'CLOSED'
                    )) DEFAULT 'OPEN',
                    analyst_id TEXT,
                    resolution_notes TEXT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now', 'utc')),
                    updated_at TEXT NOT NULL DEFAULT (datetime('now', 'utc'))
                );
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_status_created ON alerts(status, created_at DESC);
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_risk_tier ON alerts(risk_tier, hybrid_risk_score DESC);
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_step ON alerts(step ASC);
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS replay_jobs (
                    job_id TEXT PRIMARY KEY,
                    idempotency_key TEXT UNIQUE,
                    input_fingerprint TEXT NOT NULL,
                    mode TEXT NOT NULL CHECK (mode = 'DRY_RUN'),
                    status TEXT NOT NULL CHECK (status IN ('PENDING', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED')),
                    total_records INTEGER NOT NULL,
                    processed_records INTEGER NOT NULL DEFAULT 0,
                    failed_records INTEGER NOT NULL DEFAULT 0,
                    skipped_records INTEGER NOT NULL DEFAULT 0,
                    metrics_summary TEXT,
                    error_summary TEXT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now', 'utc')),
                    updated_at TEXT NOT NULL DEFAULT (datetime('now', 'utc'))
                );
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_replay_jobs_status ON replay_jobs(status, created_at DESC);
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_replay_jobs_idempotency ON replay_jobs(idempotency_key);
            """)
            # Startup crash reconciliation for interrupted jobs
            conn.execute("""
                UPDATE replay_jobs
                SET status = 'FAILED',
                    error_summary = '[{"error": "Server process terminated before job completion (startup crash reconciliation)"}]',
                    updated_at = datetime('now', 'utc')
                WHERE status IN ('PENDING', 'RUNNING');
            """)
            conn.execute("COMMIT;")
        except Exception:
            conn.execute("ROLLBACK;")
            raise
    logger.info("Initialized durable SQLite ledger, alerts, and replay schema at %s", db_path)


def record_settlement(
    db_path: Path,
    transaction_id: str,
    step: int,
    type_: str,
    amount: float,
    name_orig: str,
    name_dest: str,
    settled_at: str
) -> bool:
    """
    Persists a settlement into the ledger.
    Returns True if newly inserted, False if transaction_id was already present (idempotent).
    """
    with db_session(db_path) as conn:
        with conn:
            cursor = conn.execute("""
                INSERT OR IGNORE INTO settlements (
                    transaction_id, step, type, amount, name_orig, name_dest, settled_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?);
            """, (transaction_id, step, type_, amount, name_orig, name_dest, settled_at))
            return cursor.rowcount > 0


def get_settlement(db_path: Path, transaction_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves a single settlement record by transaction_id."""
    with db_session(db_path) as conn:
        cursor = conn.execute(
            "SELECT * FROM settlements WHERE transaction_id = ?;",
            (transaction_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None


def get_durable_settlements_since(db_path: Path, cutoff_step: int) -> List[Dict[str, Any]]:
    """
    Retrieves all confirmed settlements strictly after cutoff_step,
    ordered chronologically for deterministic graph reconstruction.
    """
    with db_session(db_path) as conn:
        cursor = conn.execute("""
            SELECT * FROM settlements
            WHERE step > ?
            ORDER BY step ASC, rowid ASC;
        """, (cutoff_step,))
        return [dict(row) for row in cursor.fetchall()]


def count_settlements(db_path: Path) -> int:
    """Returns total count of durable settlements."""
    with db_session(db_path) as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM settlements;")
        return int(cursor.fetchone()[0])


def create_alert(
    db_path: Path,
    alert_id: str,
    transaction_id: str,
    step: int,
    type_: str,
    amount: float,
    name_orig: str,
    name_dest: str,
    hybrid_risk_score: float,
    risk_tier: str,
    recommended_action: str,
    status: str = "OPEN",
    analyst_id: Optional[str] = None,
    resolution_notes: Optional[str] = None,
    created_at: Optional[str] = None,
    updated_at: Optional[str] = None
) -> Tuple[Dict[str, Any], bool]:
    """
    Persists an alert into SQLite.
    Returns (record_dict, was_created).
    If transaction_id already exists, returns (existing_record_dict, False).
    """
    if created_at is None:
        created_at = datetime.now(timezone.utc).isoformat()
    if updated_at is None:
        updated_at = created_at

    with db_session(db_path) as conn:
        with conn:
            # Check if alert already exists for this transaction_id
            cursor = conn.execute("SELECT * FROM alerts WHERE transaction_id = ?;", (transaction_id,))
            row = cursor.fetchone()
            if row:
                return dict(row), False

            try:
                conn.execute("""
                    INSERT INTO alerts (
                        alert_id, transaction_id, step, type, amount, name_orig, name_dest,
                        hybrid_risk_score, risk_tier, recommended_action, status,
                        analyst_id, resolution_notes, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, (
                    alert_id, transaction_id, step, type_, amount, name_orig, name_dest,
                    hybrid_risk_score, risk_tier, recommended_action, status,
                    analyst_id, resolution_notes, created_at, updated_at
                ))
            except sqlite3.IntegrityError:
                # Concurrent insertion won race condition
                cursor = conn.execute("SELECT * FROM alerts WHERE transaction_id = ?;", (transaction_id,))
                row = cursor.fetchone()
                if row:
                    return dict(row), False
                raise

            cursor = conn.execute("SELECT * FROM alerts WHERE alert_id = ?;", (alert_id,))
            return dict(cursor.fetchone()), True


def get_alert(db_path: Path, alert_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves an alert by alert_id."""
    with db_session(db_path) as conn:
        cursor = conn.execute("SELECT * FROM alerts WHERE alert_id = ?;", (alert_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def get_alert_by_transaction_id(db_path: Path, transaction_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves an alert by transaction_id."""
    with db_session(db_path) as conn:
        cursor = conn.execute("SELECT * FROM alerts WHERE transaction_id = ?;", (transaction_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def list_alerts(
    db_path: Path,
    status: Optional[str] = None,
    risk_tier: Optional[str] = None,
    min_score: Optional[float] = None,
    limit: int = 50,
    offset: int = 0
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Lists alerts with optional filters and pagination.
    Returns (items, total_count).
    """
    where_clauses = []
    params: List[Any] = []
    if status is not None:
        where_clauses.append("status = ?")
        params.append(status)
    if risk_tier is not None:
        where_clauses.append("risk_tier = ?")
        params.append(risk_tier)
    if min_score is not None:
        where_clauses.append("hybrid_risk_score >= ?")
        params.append(min_score)

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    with db_session(db_path) as conn:
        count_cursor = conn.execute(f"SELECT COUNT(*) FROM alerts {where_sql};", params)
        total = int(count_cursor.fetchone()[0])

        query_sql = f"""
            SELECT * FROM alerts
            {where_sql}
            ORDER BY created_at DESC, alert_id ASC
            LIMIT ? OFFSET ?;
        """
        cursor = conn.execute(query_sql, params + [limit, offset])
        items = [dict(r) for r in cursor.fetchall()]
        return items, total


def update_alert(
    db_path: Path,
    alert_id: str,
    new_status: str,
    analyst_id: Optional[str],
    resolution_notes: Optional[str],
    updated_at: str,
    expected_status: Optional[str] = None
) -> Tuple[Optional[Dict[str, Any]], bool]:
    """
    Atomically updates alert status, analyst assignment, notes, and timestamp.
    If expected_status is provided, enforces conditional status matching:
    WHERE alert_id = ? AND status = ?
    Returns (record_dict, was_updated).
    If record does not exist, returns (None, False).
    If status did not match expected_status, returns (latest_record_dict, False).
    """
    with db_session(db_path) as conn:
        with conn:
            if expected_status is not None:
                cursor = conn.execute("""
                    UPDATE alerts
                    SET status = ?, analyst_id = ?, resolution_notes = ?, updated_at = ?
                    WHERE alert_id = ? AND status = ?;
                """, (new_status, analyst_id, resolution_notes, updated_at, alert_id, expected_status))
            else:
                cursor = conn.execute("""
                    UPDATE alerts
                    SET status = ?, analyst_id = ?, resolution_notes = ?, updated_at = ?
                    WHERE alert_id = ?;
                """, (new_status, analyst_id, resolution_notes, updated_at, alert_id))

            if cursor.rowcount == 0:
                cursor = conn.execute("SELECT * FROM alerts WHERE alert_id = ?;", (alert_id,))
                row = cursor.fetchone()
                if not row:
                    return None, False
                return dict(row), False

            cursor = conn.execute("SELECT * FROM alerts WHERE alert_id = ?;", (alert_id,))
            row = cursor.fetchone()
            return dict(row), True


def count_alerts(db_path: Path) -> int:
    """Returns total count of alerts."""
    with db_session(db_path) as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM alerts;")
        return int(cursor.fetchone()[0])


def _parse_replay_job_row(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
    if not row:
        return None
    d = dict(row)
    if d.get("metrics_summary") and isinstance(d["metrics_summary"], str):
        try:
            d["metrics_summary"] = json.loads(d["metrics_summary"])
        except Exception:
            pass
    if d.get("error_summary") and isinstance(d["error_summary"], str):
        try:
            d["error_summary"] = json.loads(d["error_summary"])
        except Exception:
            pass
    return d


def create_replay_job(
    db_path: Path,
    job_id: str,
    idempotency_key: Optional[str],
    input_fingerprint: str,
    mode: str,
    status: str,
    total_records: int,
    created_at: Optional[str] = None,
    updated_at: Optional[str] = None
) -> Tuple[Dict[str, Any], bool]:
    """
    Persists a new replay job into SQLite.
    Returns (job_dict, was_created).
    If idempotency_key already exists, returns (existing_job_dict, False).
    """
    if created_at is None:
        created_at = datetime.now(timezone.utc).isoformat()
    if updated_at is None:
        updated_at = created_at

    with db_session(db_path) as conn:
        with conn:
            if idempotency_key:
                cursor = conn.execute("SELECT * FROM replay_jobs WHERE idempotency_key = ?;", (idempotency_key,))
                row = cursor.fetchone()
                if row:
                    return _parse_replay_job_row(row), False

            try:
                conn.execute("""
                    INSERT INTO replay_jobs (
                        job_id, idempotency_key, input_fingerprint, mode, status,
                        total_records, processed_records, failed_records, skipped_records,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 0, 0, 0, ?, ?);
                """, (
                    job_id, idempotency_key, input_fingerprint, mode, status,
                    total_records, created_at, updated_at
                ))
            except sqlite3.IntegrityError:
                if idempotency_key:
                    cursor = conn.execute("SELECT * FROM replay_jobs WHERE idempotency_key = ?;", (idempotency_key,))
                    row = cursor.fetchone()
                    if row:
                        return _parse_replay_job_row(row), False
                raise

            cursor = conn.execute("SELECT * FROM replay_jobs WHERE job_id = ?;", (job_id,))
            return _parse_replay_job_row(cursor.fetchone()), True


def get_replay_job(db_path: Path, job_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves a replay job by job_id."""
    with db_session(db_path) as conn:
        cursor = conn.execute("SELECT * FROM replay_jobs WHERE job_id = ?;", (job_id,))
        row = cursor.fetchone()
        return _parse_replay_job_row(row)


def get_replay_job_by_idempotency_key(db_path: Path, idempotency_key: str) -> Optional[Dict[str, Any]]:
    """Retrieves a replay job by idempotency_key."""
    with db_session(db_path) as conn:
        cursor = conn.execute("SELECT * FROM replay_jobs WHERE idempotency_key = ?;", (idempotency_key,))
        row = cursor.fetchone()
        return _parse_replay_job_row(row)


def count_active_replay_jobs(db_path: Path) -> int:
    """Returns count of replay jobs currently in PENDING or RUNNING status."""
    with db_session(db_path) as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM replay_jobs WHERE status IN ('PENDING', 'RUNNING');")
        return int(cursor.fetchone()[0])


def update_replay_job_progress(
    db_path: Path,
    job_id: str,
    processed_records: int,
    failed_records: int,
    skipped_records: int,
    updated_at: str,
    expected_status: str = "RUNNING"
) -> bool:
    """
    Updates in-flight record counters for a running replay job.
    Returns True if update succeeded, False if job status changed.
    """
    with db_session(db_path) as conn:
        with conn:
            cursor = conn.execute("""
                UPDATE replay_jobs
                SET processed_records = ?, failed_records = ?, skipped_records = ?, updated_at = ?
                WHERE job_id = ? AND status = ?;
            """, (processed_records, failed_records, skipped_records, updated_at, job_id, expected_status))
            return cursor.rowcount > 0


def update_replay_job_status(
    db_path: Path,
    job_id: str,
    new_status: str,
    expected_status: Optional[str] = None,
    processed_records: Optional[int] = None,
    failed_records: Optional[int] = None,
    skipped_records: Optional[int] = None,
    metrics_summary: Optional[Dict[str, Any]] = None,
    error_summary: Optional[List[Dict[str, Any]]] = None,
    updated_at: Optional[str] = None
) -> Tuple[Optional[Dict[str, Any]], bool]:
    """
    Atomically updates replay job status, counters, metrics, and error summaries.
    Enforces status-conditional updates (WHERE job_id = ? AND status = ?).
    Returns (job_dict, was_updated).
    """
    if updated_at is None:
        updated_at = datetime.now(timezone.utc).isoformat()
    metrics_json = json.dumps(metrics_summary) if metrics_summary is not None else None
    error_json = json.dumps(error_summary) if error_summary is not None else None

    with db_session(db_path) as conn:
        with conn:
            where_clause = "WHERE job_id = ?"
            params = [new_status, updated_at]

            set_clauses = ["status = ?", "updated_at = ?"]
            if processed_records is not None:
                set_clauses.append("processed_records = ?")
                params.append(processed_records)
            if failed_records is not None:
                set_clauses.append("failed_records = ?")
                params.append(failed_records)
            if skipped_records is not None:
                set_clauses.append("skipped_records = ?")
                params.append(skipped_records)
            if metrics_json is not None:
                set_clauses.append("metrics_summary = ?")
                params.append(metrics_json)
            if error_json is not None:
                set_clauses.append("error_summary = ?")
                params.append(error_json)

            params.append(job_id)
            if expected_status is not None:
                where_clause += " AND status = ?"
                params.append(expected_status)

            sql = f"UPDATE replay_jobs SET {', '.join(set_clauses)} {where_clause};"
            cursor = conn.execute(sql, params)
            if cursor.rowcount == 0:
                cursor = conn.execute("SELECT * FROM replay_jobs WHERE job_id = ?;", (job_id,))
                row = cursor.fetchone()
                return _parse_replay_job_row(row), False

            cursor = conn.execute("SELECT * FROM replay_jobs WHERE job_id = ?;", (job_id,))
            row = cursor.fetchone()
            return _parse_replay_job_row(row), True


def reconcile_interrupted_replay_jobs(db_path: Path) -> int:
    """
    Reconciles any replay jobs left in PENDING or RUNNING status after process crash/restart.
    Transitions them atomically to FAILED with explanatory error summary.
    Returns count of reconciled jobs.
    """
    error_summary_json = json.dumps([{
        "error": "Server process terminated before job completion (startup crash reconciliation)"
    }])
    with db_session(db_path) as conn:
        with conn:
            cursor = conn.execute("""
                UPDATE replay_jobs
                SET status = 'FAILED',
                    error_summary = ?,
                    updated_at = datetime('now', 'utc')
                WHERE status IN ('PENDING', 'RUNNING');
            """, (error_summary_json,))
            return cursor.rowcount


def list_replay_jobs(
    db_path: Path,
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Lists replay jobs with optional status filter and pagination.
    Returns (items, total_count).
    """
    where_sql = "WHERE status = ?" if status is not None else ""
    params = [status] if status is not None else []

    with db_session(db_path) as conn:
        count_cursor = conn.execute(f"SELECT COUNT(*) FROM replay_jobs {where_sql};", params)
        total = int(count_cursor.fetchone()[0])

        query_sql = f"""
            SELECT * FROM replay_jobs
            {where_sql}
            ORDER BY created_at DESC, job_id ASC
            LIMIT ? OFFSET ?;
        """
        cursor = conn.execute(query_sql, params + [limit, offset])
        items = [_parse_replay_job_row(r) for r in cursor.fetchall()]
        return items, total
