"""
Sentinel Transaction Replay Engine Service (Phase 5.3A)
Executes deterministic, bounded, read-only transaction replay in DRY_RUN mode.

Guarantees:
1. Reuses existing singleton SentinelInferenceService without reloading or fitting models.
2. Read-only isolation: DRY_RUN never mutates graph state, node degrees, edge tensors,
   cached node embeddings, the settlements ledger, or the alerts table.
3. Scores against the static, frozen graph snapshot (step 620); does not simulate graph evolution.
4. Deterministic stable transaction ID derivation based on pre-sort source indexes.
5. Deterministic chronological sorting by (step, nameOrig, -amount, transaction_id).
6. Strict watermark enforcement: transactions with step <= 620 are skipped and counted.
7. Background worker thread execution protecting the FastAPI asyncio event loop.
8. Mutual exclusion: exactly one replay job may execute at any given moment.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq

from backend.app.core.config import settings
from backend.app.core.database import (
    create_replay_job,
    get_replay_job,
    get_replay_job_by_idempotency_key,
    count_active_replay_jobs,
    update_replay_job_progress,
    update_replay_job_status,
    list_replay_jobs
)
from backend.app.schemas.replay import (
    ReplayJobCreateRequest,
    ReplayTransactionInput
)
from backend.app.schemas.transaction import TransactionRequest
from backend.app.services.inference_service import inference_service

logger = logging.getLogger("sentinel.replay")

# Background threadpool with exactly 1 worker for serialized background processing
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sentinel_replay_worker")

# Mutual exclusion lock ensuring only one replay job runs at a time
_replay_worker_lock = threading.Lock()

# Submission lock ensuring atomic submission checks and serialized dispatch
_submission_lock = threading.Lock()

# Cooperative cancellation flags keyed by job_id
_cancellation_flags: Dict[str, bool] = {}

# Active in-flight execution progress dictionary keyed by job_id
_active_job_progress: Dict[str, Dict[str, Any]] = {}


def hash_file_contents(file_path: Path, chunk_size: int = 65536) -> str:
    """
    Computes a deterministic SHA-256 hash of a file's binary contents using chunked reads,
    avoiding loading the entire dataset into memory.
    """
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_input_fingerprint(request: ReplayJobCreateRequest) -> str:
    """
    Computes a deterministic SHA-256 fingerprint for a replay job submission.
    Includes input mode, dataset identity, max record limit, chunked content hash,
    or canonical record payload.
    """
    if request.dataset_source is not None:
        dataset_path = settings.DATA_DIR / request.dataset_source
        if not dataset_path.exists():
            raise FileNotFoundError(f"Registered dataset '{request.dataset_source}' not found at {dataset_path}")
        content_hash = hash_file_contents(dataset_path)
        identity = (
            f"mode={request.mode};"
            f"source={request.dataset_source};"
            f"max_records={request.max_records};"
            f"content_hash={content_hash}"
        )
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()
    else:
        assert request.records is not None
        records_dump = [r.model_dump() for r in request.records]
        canonical_json = json.dumps(records_dump, sort_keys=True)
        identity = f"mode={request.mode};count={len(request.records)};records={canonical_json}"
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _read_registered_dataset_slice(
    parquet_path: Path,
    dataset_name: str,
    max_records: int
) -> List[Dict[str, Any]]:
    """
    Streams up to max_records from parquet file using PyArrow batch reading
    without loading the entire dataset into memory.
    Assigns stable transaction IDs based on original pre-sort source row index.
    """
    parquet_file = pq.ParquetFile(str(parquet_path))
    records: List[Dict[str, Any]] = []
    source_row_idx = 0

    for batch in parquet_file.iter_batches(batch_size=min(max_records, 1024)):
        batch_df = batch.to_pandas()
        for _, row in batch_df.iterrows():
            step = int(row["step"])
            # Derive deterministic transaction ID from pre-sort source row position
            txn_id = f"TXN_DS_{dataset_name}_{source_row_idx}_S{step}"
            rec = {
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
                "_source_row_idx": source_row_idx
            }
            records.append(rec)
            source_row_idx += 1
            if len(records) >= max_records:
                return records

    return records


def _prepare_inline_records(
    inputs: List[ReplayTransactionInput]
) -> List[Dict[str, Any]]:
    """
    Prepares inline transaction records, assigning stable transaction IDs
    based on original pre-sort array index if caller omitted transaction_id.
    """
    records: List[Dict[str, Any]] = []
    for idx, r in enumerate(inputs):
        if r.transaction_id:
            txn_id = r.transaction_id
        else:
            fields_hash = hashlib.sha256(
                f"{r.step}_{r.type}_{r.amount}_{r.nameOrig}_{r.nameDest}".encode("utf-8")
            ).hexdigest()[:8]
            txn_id = f"TXN_PAYLOAD_{idx}_S{r.step}_{fields_hash}"

        rec = {
            "transaction_id": txn_id,
            "step": r.step,
            "type": r.type,
            "amount": r.amount,
            "nameOrig": r.nameOrig,
            "oldbalanceOrg": r.oldbalanceOrg,
            "newbalanceOrig": r.newbalanceOrig,
            "nameDest": r.nameDest,
            "oldbalanceDest": r.oldbalanceDest,
            "newbalanceDest": r.newbalanceDest,
            "_source_row_idx": idx
        }
        records.append(rec)
    return records


def _build_metrics_summary(
    total_records: int,
    processed_count: int = 0,
    failed_count: int = 0,
    skipped_count: int = 0,
    latencies: Optional[List[float]] = None,
    tier_counts: Optional[Dict[str, int]] = None,
    action_counts: Optional[Dict[str, int]] = None,
    total_amount_replayed: float = 0.0,
    flagged_amount: float = 0.0,
    scores: Optional[List[float]] = None,
    score_buckets: Optional[Dict[str, int]] = None,
    t_job_duration: float = 0.0,
    is_cancelled: bool = False,
    is_failed: bool = False,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Constructs a strongly typed, mathematically consistent metrics summary
    for completed, cancelled, or failed replay jobs.
    """
    if "processed" in kwargs:
        processed_count = kwargs["processed"]
    if "skipped_stale" in kwargs:
        skipped_count = kwargs["skipped_stale"]
    if "failed" in kwargs:
        failed_count = kwargs["failed"]
    if "latencies_ms" in kwargs:
        latencies = kwargs["latencies_ms"]
    if "risk_tier_counts" in kwargs:
        tier_counts = kwargs["risk_tier_counts"]
    if "action_counts" in kwargs:
        action_counts = kwargs["action_counts"]
    if "total_amount" in kwargs:
        total_amount_replayed = kwargs["total_amount"]
    if "hybrid_scores" in kwargs:
        scores = kwargs["hybrid_scores"]
    if "elapsed_sec" in kwargs:
        t_job_duration = kwargs["elapsed_sec"]

    if latencies is None:
        latencies = []
    if tier_counts is None:
        tier_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    if action_counts is None:
        action_counts = {
            "AUTOMATED_BLOCK": 0,
            "STEP_UP_AUTHENTICATION": 0,
            "MANUAL_ANALYST_REVIEW": 0,
            "AUTOMATED_CLEAR_AND_SETTLE": 0
        }
    if scores is None:
        scores = []

    # 1. Latency statistics
    if latencies:
        lat_stats = {
            "min": round(float(min(latencies)), 2),
            "max": round(float(max(latencies)), 2),
            "mean": round(float(np.mean(latencies)), 2),
            "p50": round(float(np.percentile(latencies, 50)), 2),
            "p95": round(float(np.percentile(latencies, 95)), 2),
            "total_duration_sec": round(float(t_job_duration), 2)
        }
    else:
        lat_stats = {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "total_duration_sec": round(float(t_job_duration), 2)
        }

    # 2. Financial exposure
    if total_amount_replayed > 0.0:
        flagged_amt_pct = round((flagged_amount / total_amount_replayed) * 100.0, 2)
    else:
        flagged_amt_pct = 0.0

    if processed_count > 0:
        flagged_rec_count = tier_counts.get("CRITICAL", 0) + tier_counts.get("HIGH", 0)
        flagged_rec_pct = round((flagged_rec_count / processed_count) * 100.0, 2)
    else:
        flagged_rec_pct = 0.0

    financial_exposure = {
        "total_amount_replayed": round(float(total_amount_replayed), 2),
        "flagged_amount": round(float(flagged_amount), 2),
        "flagged_amount_percentage": flagged_amt_pct,
        "flagged_record_percentage": flagged_rec_pct
    }

    # 3. Score distribution
    if score_buckets:
        norm_buckets = {
            "[0.0, 0.2)": score_buckets.get("[0.0, 0.2)", score_buckets.get("bucket_0_0_to_0_2", score_buckets.get("0.0-0.2", 0))),
            "[0.2, 0.5)": score_buckets.get("[0.2, 0.5)", score_buckets.get("bucket_0_2_to_0_5", score_buckets.get("0.2-0.5", 0))),
            "[0.5, 0.8)": score_buckets.get("[0.5, 0.8)", score_buckets.get("bucket_0_5_to_0_8", score_buckets.get("0.5-0.8", 0))),
            "[0.8, 1.0]": score_buckets.get("[0.8, 1.0]", score_buckets.get("[0.8, 1.0)", score_buckets.get("bucket_0_8_to_1_0", score_buckets.get("0.8-1.0", 0)))),
        }
    else:
        norm_buckets = {
            "[0.0, 0.2)": 0,
            "[0.2, 0.5)": 0,
            "[0.5, 0.8)": 0,
            "[0.8, 1.0]": 0
        }
        for s in scores:
            if s < 0.2:
                norm_buckets["[0.0, 0.2)"] += 1
            elif s < 0.5:
                norm_buckets["[0.2, 0.5)"] += 1
            elif s < 0.8:
                norm_buckets["[0.5, 0.8)"] += 1
            else:
                norm_buckets["[0.8, 1.0]"] += 1

    if scores:
        mean_score = round(float(np.mean(scores)), 4)
        median_score = round(float(np.median(scores)), 4)
    else:
        mean_score = None
        median_score = None

    score_dist = {
        "mean_score": mean_score,
        "median_score": median_score,
        "mean_hybrid_score": mean_score,
        "median_hybrid_score": median_score,
        "score_buckets": norm_buckets
    }

    # 4. Progress status at termination
    completed = processed_count + failed_count + skipped_count
    progress_pct = round(min(100.0, (completed / total_records) * 100.0), 1) if total_records > 0 else 100.0
    throughput = round(completed / t_job_duration, 2) if (completed >= 1 and t_job_duration > 0.05) else None
    eta = 0.0 if not (is_cancelled or is_failed) else None

    progress_summary = {
        "progress_percentage": progress_pct,
        "elapsed_duration_sec": round(float(t_job_duration), 2),
        "throughput_records_per_sec": throughput,
        "estimated_remaining_sec": eta
    }

    # 5. Operational disclaimer
    if is_cancelled:
        disclaimer = (
            "Replay job was CANCELLED before completion. Partial metrics reflect observations processed prior to cancellation. "
            "Evaluated against static frozen graph snapshot (step 620) without state mutation."
        )
    elif is_failed:
        disclaimer = (
            "Replay job FAILED before completion. Partial metrics reflect observations processed prior to failure. "
            "Evaluated against static frozen graph snapshot (step 620) without state mutation."
        )
    else:
        disclaimer = (
            "DRY_RUN execution evaluates transactions against a static frozen graph snapshot (step 620) "
            "and does not simulate dynamic graph evolution or mutate persistence. Latency and risk tier "
            "tallies reflect pipeline throughput, not real-world fraud confirmation."
        )

    return {
        "records_summary": {
            "total": total_records,
            "processed": processed_count,
            "failed": failed_count,
            "skipped_stale_watermark": skipped_count
        },
        "latency_ms": lat_stats,
        "progress": progress_summary,
        "financial_exposure": financial_exposure,
        "score_distribution": score_dist,
        "risk_tier_counts": tier_counts,
        "action_counts": action_counts,
        "state_mutations": {
            "alerts_created": 0,
            "settlements_applied": 0
        },
        "disclaimer": disclaimer
    }


def execute_replay_worker(
    db_path: Path,
    job_id: str,
    raw_records: List[Dict[str, Any]]
) -> None:
    """
    Background worker thread function executing the DRY_RUN replay sequence.
    Acquires _replay_worker_lock to prevent concurrent worker execution.
    """
    acquired = _replay_worker_lock.acquire(blocking=False)
    if not acquired:
        logger.warning("Replay worker could not acquire lock for job %s; another job is active.", job_id)
        update_replay_job_status(
            db_path=db_path,
            job_id=job_id,
            new_status="FAILED",
            expected_status="PENDING",
            error_summary=[{"error": "Failed to acquire worker lock; another replay job is active."}]
        )
        return

    try:
        # 1. Claim job atomically (PENDING -> RUNNING)
        claimed_job, was_claimed = update_replay_job_status(
            db_path=db_path,
            job_id=job_id,
            new_status="RUNNING",
            expected_status="PENDING"
        )
        if not was_claimed:
            logger.info("Job %s could not be claimed into RUNNING (status may be CANCELLED). Aborting.", job_id)
            return

        logger.info("Replay job %s claimed by worker. Processing %d records...", job_id, len(raw_records))

        # 2. Sort records deterministically by simulation step and stable tie-breakers
        # Tie-breakers: step ASC, nameOrig ASC, amount DESC, transaction_id ASC
        sorted_records = sorted(
            raw_records,
            key=lambda x: (x["step"], x["nameOrig"], -x["amount"], x["transaction_id"])
        )

        total_recs = len(sorted_records)
        processed_count = 0
        failed_count = 0
        skipped_count = 0
        latencies: List[float] = []
        tier_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        action_counts = {
            "AUTOMATED_BLOCK": 0,
            "STEP_UP_AUTHENTICATION": 0,
            "MANUAL_ANALYST_REVIEW": 0,
            "AUTOMATED_CLEAR_AND_SETTLE": 0
        }
        error_samples: List[Dict[str, Any]] = []

        total_amount_replayed = 0.0
        flagged_amount = 0.0
        scores: List[float] = []
        score_buckets = {"[0.0, 0.2)": 0, "[0.2, 0.5)": 0, "[0.5, 0.8)": 0, "[0.8, 1.0]": 0}

        t_job_start = time.perf_counter()
        last_progress_write_time = t_job_start

        # Dynamic progress cadence: shorter batches update more frequently
        progress_interval = min(25, max(5, total_recs // 10)) if total_recs > 20 else max(1, total_recs // 4)

        for idx, rec in enumerate(sorted_records):
            # Check for cooperative cancellation
            if _cancellation_flags.get(job_id, False):
                logger.info("Replay job %s cancellation detected at record %d. Halting worker.", job_id, idx)
                t_duration = time.perf_counter() - t_job_start
                partial_metrics = _build_metrics_summary(
                    total_records=total_recs,
                    processed_count=processed_count,
                    failed_count=failed_count,
                    skipped_count=skipped_count,
                    latencies=latencies,
                    tier_counts=tier_counts,
                    action_counts=action_counts,
                    total_amount_replayed=total_amount_replayed,
                    flagged_amount=flagged_amount,
                    scores=scores,
                    score_buckets=score_buckets,
                    t_job_duration=t_duration,
                    is_cancelled=True
                )
                update_replay_job_status(
                    db_path=db_path,
                    job_id=job_id,
                    new_status="CANCELLED",
                    expected_status="RUNNING",
                    processed_records=processed_count,
                    failed_records=failed_count,
                    skipped_records=skipped_count,
                    metrics_summary=partial_metrics,
                    error_summary=error_samples
                )
                _active_job_progress.pop(job_id, None)
                return

            # Watermark check: step <= 620 is prior to or within graph snapshot cutoff
            # Must be skipped and never scored
            if rec["step"] <= 620:
                skipped_count += 1
                if len(error_samples) < 10:
                    error_samples.append({
                        "transaction_id": rec["transaction_id"],
                        "step": rec["step"],
                        "reason": "SKIPPED_STALE_WATERMARK: step <= 620 is prior to or within graph snapshot cutoff (step 620)."
                    })
            else:
                try:
                    txn_req = TransactionRequest(
                        step=rec["step"],
                        type=rec["type"],
                        amount=rec["amount"],
                        nameOrig=rec["nameOrig"],
                        oldbalanceOrg=rec["oldbalanceOrg"],
                        newbalanceOrig=rec.get("newbalanceOrig"),
                        nameDest=rec["nameDest"],
                        oldbalanceDest=rec["oldbalanceDest"],
                        newbalanceDest=rec.get("newbalanceDest")
                    )

                    t0 = time.perf_counter()
                    pred_resp = inference_service.predict_transaction(txn_req)
                    lat_ms = (time.perf_counter() - t0) * 1000.0

                    score = float(pred_resp.hybrid_risk_score)
                    if not math.isfinite(score) or not (0.0 <= score <= 1.0):
                        raise ValueError(
                            f"Model produced invalid hybrid risk score: {score} "
                            f"(must be finite float in [0.0, 1.0]) for transaction {rec['transaction_id']}"
                        )

                    # Only accumulate metrics once prediction and score are confirmed valid
                    latencies.append(lat_ms)
                    tier = pred_resp.assigned_risk_tier
                    action = pred_resp.recommended_action
                    amt = float(rec["amount"])

                    total_amount_replayed += amt
                    scores.append(score)

                    # Bucket intervals: [0.0, 0.2), [0.2, 0.5), [0.5, 0.8), [0.8, 1.0]
                    if score < 0.2:
                        score_buckets["[0.0, 0.2)"] += 1
                    elif score < 0.5:
                        score_buckets["[0.2, 0.5)"] += 1
                    elif score < 0.8:
                        score_buckets["[0.5, 0.8)"] += 1
                    else:
                        score_buckets["[0.8, 1.0]"] += 1

                    if tier in ("CRITICAL", "HIGH"):
                        flagged_amount += amt

                    if tier in tier_counts:
                        tier_counts[tier] += 1
                    if action in action_counts:
                        action_counts[action] += 1

                    processed_count += 1

                except Exception as e:
                    failed_count += 1
                    if len(error_samples) < 10:
                        error_samples.append({
                            "transaction_id": rec["transaction_id"],
                            "step": rec["step"],
                            "error": str(e)[:500]
                        })

            # Real-time progress tracking
            completed = processed_count + failed_count + skipped_count
            now_t = time.perf_counter()
            elapsed = now_t - t_job_start
            pct = round(min(100.0, (completed / total_recs) * 100.0), 1) if total_recs > 0 else 100.0

            # Guard against fabricated throughput: require >= 3 records and >= 0.2s elapsed
            throughput = None
            eta = None
            if completed >= 3 and elapsed >= 0.2:
                throughput = round(completed / elapsed, 2)
                remaining = max(0, total_recs - completed)
                if throughput > 0.0 and remaining > 0:
                    eta = round(remaining / throughput, 1)
                elif remaining == 0:
                    eta = 0.0

            progress_data = {
                "progress_percentage": pct,
                "elapsed_duration_sec": round(elapsed, 2),
                "throughput_records_per_sec": throughput,
                "estimated_remaining_sec": eta
            }
            _active_job_progress[job_id] = progress_data

            # Throttled SQLite persistence (cadence-based or minimum 2.0s interval)
            time_since_last_write = now_t - last_progress_write_time
            if ((idx + 1) % progress_interval == 0 and time_since_last_write >= 0.5) or time_since_last_write >= 2.0 or (idx + 1 == total_recs):
                now_iso = datetime.now(timezone.utc).isoformat()
                update_replay_job_progress(
                    db_path=db_path,
                    job_id=job_id,
                    processed_records=processed_count,
                    failed_records=failed_count,
                    skipped_records=skipped_count,
                    updated_at=now_iso,
                    expected_status="RUNNING"
                )
                last_progress_write_time = now_t

        t_job_duration = time.perf_counter() - t_job_start

        # 3. Compute final metrics
        final_metrics = _build_metrics_summary(
            total_records=total_recs,
            processed_count=processed_count,
            failed_count=failed_count,
            skipped_count=skipped_count,
            latencies=latencies,
            tier_counts=tier_counts,
            action_counts=action_counts,
            total_amount_replayed=total_amount_replayed,
            flagged_amount=flagged_amount,
            scores=scores,
            score_buckets=score_buckets,
            t_job_duration=t_job_duration,
            is_cancelled=False
        )

        # 4. Atomically complete the job
        update_replay_job_status(
            db_path=db_path,
            job_id=job_id,
            new_status="COMPLETED",
            expected_status="RUNNING",
            processed_records=processed_count,
            failed_records=failed_count,
            skipped_records=skipped_count,
            metrics_summary=final_metrics,
            error_summary=error_samples
        )
        _active_job_progress.pop(job_id, None)
        logger.info(
            "Replay job %s completed successfully: %d processed, %d skipped, %d failed in %.2fs.",
            job_id, processed_count, skipped_count, failed_count, t_job_duration
        )

    except Exception as exc:
        logger.exception("Unexpected error in replay worker for job %s: %s", job_id, str(exc))
        t_job_duration = time.perf_counter() - t_job_start
        failed_metrics = _build_metrics_summary(
            total_records=len(raw_records),
            processed_count=processed_count,
            failed_count=failed_count,
            skipped_count=skipped_count,
            latencies=latencies,
            tier_counts=tier_counts,
            action_counts=action_counts,
            total_amount_replayed=total_amount_replayed,
            flagged_amount=flagged_amount,
            scores=scores,
            score_buckets=score_buckets,
            t_job_duration=t_job_duration,
            is_failed=True
        )
        update_replay_job_status(
            db_path=db_path,
            job_id=job_id,
            new_status="FAILED",
            expected_status="RUNNING",
            metrics_summary=failed_metrics,
            error_summary=[{"error": f"Unhandled worker exception: {str(exc)[:500]}"}]
        )
        _active_job_progress.pop(job_id, None)
    finally:
        _active_job_progress.pop(job_id, None)
        _cancellation_flags.pop(job_id, None)
        _replay_worker_lock.release()


class SentinelReplayService:
    """Orchestration service for submitting, tracking, and cancelling replay jobs."""

    def submit_job(
        self,
        db_path: Path,
        request: ReplayJobCreateRequest
    ) -> Tuple[Dict[str, Any], int]:
        """
        Validates submission, performs idempotency checks, records PENDING job,
        and dispatches background worker.
        Returns (job_dict, http_status_code).
        """
        fingerprint = compute_input_fingerprint(request)

        with _submission_lock:
            # 1. Idempotency Check
            if request.idempotency_key:
                existing = get_replay_job_by_idempotency_key(db_path, request.idempotency_key)
                if existing:
                    if existing["input_fingerprint"] == fingerprint:
                        logger.info("Idempotent retry for key '%s'; returning existing job %s.", request.idempotency_key, existing["job_id"])
                        return existing, 200
                    else:
                        logger.warning("Conflict for key '%s': payload fingerprint differs from existing job %s.", request.idempotency_key, existing["job_id"])
                        raise ValueError(
                            f"Idempotency key '{request.idempotency_key}' already exists with a different input payload or configuration."
                        )

            # 2. Check if a replay job is already actively running or pending in DB
            active_count = count_active_replay_jobs(db_path)
            if active_count > 0 or _replay_worker_lock.locked():
                raise RuntimeError("Another replay job is currently executing or pending. Concurrent replay execution is not permitted.")

            # 3. Read and normalize records
            if request.dataset_source is not None:
                dataset_path = settings.DATA_DIR / request.dataset_source
                records = _read_registered_dataset_slice(
                    parquet_path=dataset_path,
                    dataset_name=request.dataset_source,
                    max_records=request.max_records
                )
            else:
                assert request.records is not None
                records = _prepare_inline_records(request.records)

            import uuid
            job_id = f"REP_{uuid.uuid4().hex[:12]}"

            # 4. Persist PENDING job
            job_dict, was_created = create_replay_job(
                db_path=db_path,
                job_id=job_id,
                idempotency_key=request.idempotency_key,
                input_fingerprint=fingerprint,
                mode=request.mode,
                status="PENDING",
                total_records=len(records)
            )
            if not was_created:
                # Race condition on idempotency key
                if job_dict["input_fingerprint"] == fingerprint:
                    return job_dict, 200
                else:
                    raise ValueError(
                        f"Idempotency key '{request.idempotency_key}' already exists with a different input payload or configuration."
                    )

            # 5. Dispatch background worker thread with safety fallback
            try:
                _executor.submit(execute_replay_worker, db_path, job_id, records)
            except Exception as dispatch_err:
                logger.exception("Failed to dispatch background replay worker for job %s: %s", job_id, str(dispatch_err))
                update_replay_job_status(
                    db_path=db_path,
                    job_id=job_id,
                    new_status="FAILED",
                    expected_status="PENDING",
                    error_summary=[{"error": f"Failed to dispatch background worker: {str(dispatch_err)}"}]
                )
                raise RuntimeError(f"Failed to dispatch background worker: {str(dispatch_err)}") from dispatch_err

            return job_dict, 201

    def get_job(
        self,
        db_path: Path,
        job_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieves a replay job by job_id, attaching live in-flight progress metadata if running.
        """
        job = get_replay_job(db_path, job_id)
        if not job:
            return None
        if job["status"] == "RUNNING" and job_id in _active_job_progress:
            job["progress"] = _active_job_progress[job_id]
        return job

    def cancel_job(
        self,
        db_path: Path,
        job_id: str
    ) -> Tuple[Dict[str, Any], int]:
        """
        Signals cooperative cancellation for an in-flight or pending job.
        Returns (job_dict, http_status_code).
        """
        job = get_replay_job(db_path, job_id)
        if not job:
            raise KeyError(f"Replay job '{job_id}' not found.")

        current_status = job["status"]
        if current_status in {"COMPLETED", "FAILED", "CANCELLED"}:
            raise ValueError(f"Cannot cancel job '{job_id}' in terminal state '{current_status}'.")

        if current_status == "PENDING":
            # Direct cancel before worker claims
            empty_metrics = _build_metrics_summary(
                total_records=job.get("total_records", 0),
                is_cancelled=True
            )
            updated, was_updated = update_replay_job_status(
                db_path=db_path,
                job_id=job_id,
                new_status="CANCELLED",
                expected_status="PENDING",
                metrics_summary=empty_metrics
            )
            return updated or job, 200

        elif current_status == "RUNNING":
            # Signal worker cooperative cancellation
            _cancellation_flags[job_id] = True
            if job_id in _active_job_progress:
                job["progress"] = _active_job_progress[job_id]
            return job, 200

        return job, 200


replay_service = SentinelReplayService()
