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


class ReplayJobResponse(BaseModel):
    """Detailed metadata and progress response for a replay job."""
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
    metrics_summary: Optional[Dict[str, Any]] = Field(
        None, description="Aggregate latency percentiles and risk tier tallies"
    )
    error_summary: Optional[List[Dict[str, Any]]] = Field(
        None, description="Bounded sample of record-level failures or skipped items"
    )
    created_at: str = Field(..., description="UTC ISO-8601 creation timestamp")
    updated_at: str = Field(..., description="UTC ISO-8601 last-updated timestamp")


class ReplayJobListResponse(BaseModel):
    """Paginated list of replay jobs."""
    total: int = Field(..., ge=0, description="Total number of replay jobs matching query")
    limit: int = Field(..., ge=1, description="Pagination limit")
    offset: int = Field(..., ge=0, description="Pagination offset")
    items: List[ReplayJobResponse] = Field(..., description="List of replay jobs")
