"""
Sentinel SQLite Durable Ledger & WAL Connection Manager
Provides crash-resilient persistence for confirmed settlements.
"""

from contextlib import contextmanager
import logging
from pathlib import Path
import sqlite3
from typing import Any, Dict, Generator, List, Optional

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
            conn.execute("COMMIT;")
        except Exception:
            conn.execute("ROLLBACK;")
            raise
    logger.info("Initialized durable SQLite ledger at %s", db_path)


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
