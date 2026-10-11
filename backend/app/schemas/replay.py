"""
Pydantic Schemas for Sentinel Transaction Replay Engine (Phase 5.3A)
"""

import math
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _validate_finite(value: Optional[float], field_name: str) -> Optional[float]:
    if value is not None and (math.isnan(value) or math.isinf(value)):
        raise ValueError(f"{field_name} must be a finite numerical value.")
    return value


class ReplayTransactionInput(BaseModel):
    """Transaction record for inline replay batches."""
    model_config = ConfigDict(extra="forbid")

    transaction_id: Optional[str] = Field(
        None, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_\-\.]+$",
        description="Optional caller-supplied transaction identifier"
    )
    step: int = Field(..., ge=1, description="Simulation step / time index (1-indexed)")
    type: Literal["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"] = Field(
        ..., description="Transaction payment type"
    )
    amount: float = Field(..., gt=0.0, description="Transaction amount in currency units")
    nameOrig: str = Field(..., min_length=2, max_length=64, description="Origin account identifier")
    oldbalanceOrg: float = Field(..., ge=0.0, description="Origin balance prior to transaction")
    newbalanceOrig: Optional[float] = Field(
        None, ge=0.0, description="Optional post-settlement origin balance (ignored by pre-auth scoring)"
    )
    nameDest: str = Field(..., min_length=2, max_length=64, description="Destination account identifier")
    oldbalanceDest: float = Field(..., ge=0.0, description="Destination balance prior to transaction")
    newbalanceDest: Optional[float] = Field(
        None, ge=0.0, description="Optional post-settlement destination balance (ignored by pre-auth scoring)"
    )

    @field_validator("amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest")
    @classmethod
    def check_finite(cls, v, info):
        return _validate_finite(v, info.field_name)


class ReplayJobCreateRequest(BaseModel):
    """Submission request for launching a transaction replay job."""
    model_config = ConfigDict(extra="forbid")

    mode: Literal["DRY_RUN"] = Field(
        default="DRY_RUN",
        description="Execution mode. Only 'DRY_RUN' is supported in Phase 5.3A."
    )
    idempotency_key: Optional[str] = Field(
        None, min_length=1, max_length=128,
        description="Client idempotency key for safe retries"
    )
    dataset_source: Optional[Literal["test.parquet"]] = Field(
        None,
        description="Whitelisted registered dataset source to replay (mutually exclusive with 'records')"
    )
    max_records: int = Field(
        default=500, ge=1, le=5000,
        description="Maximum records to read from registered dataset (max 5000)"
    )
    records: Optional[List[ReplayTransactionInput]] = Field(
        None, max_length=1000,
        description="Inline batch transaction records (max 1000, mutually exclusive with 'dataset_source')"
    )

    @model_validator(mode="after")
    def verify_exclusive_input_source(self) -> "ReplayJobCreateRequest":
        has_dataset = self.dataset_source is not None
        has_records = self.records is not None and len(self.records) > 0

        if has_dataset and has_records:
            raise ValueError(
                "Conflicting input sources: Specify either 'records' or 'dataset_source', not both."
            )
        if not has_dataset and not has_records:
            raise ValueError(
                "Missing input source: Exactly one input source must be specified ('records' or 'dataset_source')."
            )
        return self


class ReplayRecordsSummary(BaseModel):
    """Execution breakdown across input records."""
    model_config = ConfigDict(extra="forbid")

    total: int = Field(..., ge=0, description="Total input records in replay job")
    processed: int = Field(..., ge=0, description="Records successfully evaluated by inference engine")
    failed: int = Field(..., ge=0, description="Records that encountered errors during prediction")
    skipped_stale_watermark: int = Field(..., ge=0, description="Records skipped due to watermark step <= 620")


class LatencyMetrics(BaseModel):
    """Inference engine latency distribution metrics in milliseconds."""
    model_config = ConfigDict(extra="forbid")

    min: float = Field(..., ge=0.0, description="Minimum transaction prediction latency in milliseconds")
    max: float = Field(..., ge=0.0, description="Maximum transaction prediction latency in milliseconds")
    mean: float = Field(..., ge=0.0, description="Mean transaction prediction latency in milliseconds")
    p50: float = Field(..., ge=0.0, description="Median (50th percentile) prediction latency in milliseconds")
    p95: float = Field(..., ge=0.0, description="95th percentile prediction latency in milliseconds")
    total_duration_sec: float = Field(..., ge=0.0, description="Total replay job execution duration in seconds")


class ReplayProgressStatus(BaseModel):
    """Real-time or terminal execution progress metrics."""
    model_config = ConfigDict(extra="forbid")

    progress_percentage: float = Field(..., ge=0.0, le=100.0, description="Overall completion percentage (0.0 to 100.0)")
    elapsed_duration_sec: float = Field(..., ge=0.0, description="Elapsed execution duration in seconds")
    throughput_records_per_sec: Optional[float] = Field(None, ge=0.0, description="Current processing rate in records per second (null if insufficient samples)")
    estimated_remaining_sec: Optional[float] = Field(None, ge=0.0, description="Estimated seconds to completion (null until reliable throughput established)")


class FinancialExposureMetrics(BaseModel):
    """Monetary volume and risk exposure metrics across scored transactions."""
    model_config = ConfigDict(extra="forbid")

    total_amount_replayed: float = Field(..., ge=0.0, description="Total currency volume across successfully scored transactions")
    flagged_amount: float = Field(..., ge=0.0, description="Total currency volume assigned to HIGH or CRITICAL risk tiers")
    flagged_amount_percentage: float = Field(..., ge=0.0, le=100.0, description="Percentage of scored currency volume flagged as HIGH or CRITICAL")
    flagged_record_percentage: float = Field(..., ge=0.0, le=100.0, description="Percentage of scored transactions assigned to HIGH or CRITICAL risk tiers")


class ScoreDistributionMetrics(BaseModel):
    """Model fraud risk score distributions and histogram buckets."""
    model_config = ConfigDict(extra="ignore")

    mean_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Mean hybrid risk score across successfully scored transactions (null if no records scored)")
    median_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Median hybrid risk score across successfully scored transactions (null if no records scored)")
    mean_hybrid_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Alias for mean_score")
    median_hybrid_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Alias for median_score")
    score_buckets: Dict[str, int] = Field(
        ...,
        description="Distribution of scored transactions across risk intervals: [0.0, 0.2), [0.2, 0.5), [0.5, 0.8), [0.8, 1.0]"
    )


class StateMutationsSummary(BaseModel):
    """Audit of mutations to persistent state (always zero in DRY_RUN mode)."""
    model_config = ConfigDict(extra="ignore")

    alerts_created: int = Field(0, ge=0, description="Fraud alerts persisted to database (0 in DRY_RUN)")
    settlements_applied: int = Field(0, ge=0, description="Settlement ledger events written (0 in DRY_RUN)")


class ReplayMetricsSummary(BaseModel):
    """Comprehensive aggregate metrics summary for a transaction replay job."""
    model_config = ConfigDict(extra="forbid")

    records_summary: ReplayRecordsSummary = Field(..., description="Summary of record evaluation counts")
    latency_ms: LatencyMetrics = Field(..., description="Latency statistics across scored transactions")
    progress: Optional[ReplayProgressStatus] = Field(None, description="Progress and duration metrics at completion or cancellation")
    financial_exposure: FinancialExposureMetrics = Field(..., description="Currency volume and risk exposure metrics")
    score_distribution: ScoreDistributionMetrics = Field(..., description="Model fraud score distribution and bucket histogram")
    risk_tier_counts: Dict[str, int] = Field(..., description="Counts per assigned risk tier (CRITICAL, HIGH, MEDIUM, LOW)")
    action_counts: Dict[str, int] = Field(..., description="Counts per recommended operational action")
    state_mutations: StateMutationsSummary = Field(
        default_factory=StateMutationsSummary,
        description="State mutation audit (alerts_created: 0, settlements_applied: 0)"
    )
    disclaimer: str = Field(..., description="Operational replay limitations and static graph snapshot disclaimer")


class ReplayJobResponse(BaseModel):
    """Detailed metadata and progress response for a replay job."""
    model_config = ConfigDict(extra="ignore")

    job_id: str = Field(..., description="Unique replay job identifier")
    idempotency_key: Optional[str] = Field(None, description="Client idempotency key")
    mode: Literal["DRY_RUN"] = Field(..., description="Replay execution mode")
    status: Literal["PENDING", "RUNNING", "COMPLETED", "FAILED", "CANCELLED"] = Field(
        ..., description="Current job lifecycle status"
    )
    total_records: int = Field(..., ge=0, description="Total records in replay job")
    processed_records: int = Field(..., ge=0, description="Records successfully evaluated")
    failed_records: int = Field(..., ge=0, description="Records that failed during processing")
    skipped_records: int = Field(..., ge=0, description="Records skipped (e.g. step <= 620 watermark)")
    progress: Optional[ReplayProgressStatus] = Field(
        None, description="Real-time or terminal execution progress metrics"
    )
    metrics_summary: Optional[ReplayMetricsSummary] = Field(
        None, description="Strongly typed aggregate latency, exposure, and score metrics"
    )
    error_summary: Optional[List[Dict[str, Any]]] = Field(
        None, description="Bounded sample of record-level failures or skipped items"
    )
    created_at: str = Field(..., description="UTC ISO-8601 creation timestamp")
    updated_at: str = Field(..., description="UTC ISO-8601 last-updated timestamp")

    @model_validator(mode="before")
    @classmethod
    def resolve_progress(cls, values: Any) -> Any:
        if isinstance(values, dict):
            # 1. If progress already provided explicitly
            if values.get("progress") is not None:
                return values
            # 2. If progress is inside metrics_summary
            metrics = values.get("metrics_summary")
            if isinstance(metrics, dict) and metrics.get("progress") is not None:
                values["progress"] = metrics["progress"]
                return values
            # 3. Otherwise compute progress from counters
            total = values.get("total_records", 0)
            processed = values.get("processed_records", 0)
            failed = values.get("failed_records", 0)
            skipped = values.get("skipped_records", 0)
            completed = processed + failed + skipped
            pct = 100.0 if total == 0 else round(min(100.0, (completed / total) * 100.0), 1)

            elapsed = 0.0
            if isinstance(metrics, dict) and isinstance(metrics.get("latency_ms"), dict):
                elapsed = float(metrics["latency_ms"].get("total_duration_sec", 0.0))

            status = values.get("status", "PENDING")
            eta = 0.0 if status == "COMPLETED" else None

            values["progress"] = {
                "progress_percentage": pct,
                "elapsed_duration_sec": elapsed,
                "throughput_records_per_sec": None,
                "estimated_remaining_sec": eta
            }
        return values


class ReplayJobListResponse(BaseModel):
    """Paginated list of replay jobs."""
    total: int = Field(..., ge=0, description="Total number of replay jobs matching query")
    limit: int = Field(..., ge=1, description="Pagination limit")
    offset: int = Field(..., ge=0, description="Pagination offset")
    items: List[ReplayJobResponse] = Field(..., description="List of replay jobs")
