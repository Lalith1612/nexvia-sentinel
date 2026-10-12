"""
Pydantic Schemas for Sentinel Kafka Real-Time Streaming Layer (Phase 6)
"""

from datetime import datetime, timezone
import math
from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel, Field, field_validator

from backend.app.schemas.transaction import TransactionRequest


def _validate_finite(value: Optional[float], field_name: str) -> Optional[float]:
    if value is not None and (math.isnan(value) or math.isinf(value)):
        raise ValueError(f"{field_name} must be a finite numerical value.")
    return value


class TransactionStreamEvent(BaseModel):
    """
    Schema for transaction events ingested from Kafka stream (sentinel.transactions).
    Compatible with TransactionRequest while carrying stream metadata.
    """
    transaction_id: str = Field(..., min_length=1, max_length=128, description="Unique transaction event identifier")
    step: int = Field(..., ge=1, description="Simulation step / time index (1-indexed)")
    type: Literal["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"] = Field(..., description="Transaction type")
    amount: float = Field(..., gt=0.0, description="Transaction amount in currency units")
    nameOrig: str = Field(..., min_length=2, max_length=64, description="Origin account identifier")
    oldbalanceOrg: float = Field(..., ge=0.0, description="Origin balance immediately prior to execution")
    newbalanceOrig: Optional[float] = Field(None, ge=0.0, description="Optional post-settlement origin balance")
    nameDest: str = Field(..., min_length=2, max_length=64, description="Destination account identifier")
    oldbalanceDest: float = Field(..., ge=0.0, description="Destination balance immediately prior to execution")
    newbalanceDest: Optional[float] = Field(None, ge=0.0, description="Optional post-settlement destination balance")
    event_timestamp: Optional[str] = Field(None, description="ISO-8601 event generation timestamp")
    idempotency_key: Optional[str] = Field(None, max_length=128, description="Optional idempotency key for deduplication")

    @field_validator("amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest")
    @classmethod
    def check_finite(cls, v, info):
        return _validate_finite(v, info.field_name)

    def to_transaction_request(self) -> TransactionRequest:
        """Converts stream event to core TransactionRequest model for inference service."""
        return TransactionRequest(
            step=self.step,
            type=self.type,
            amount=self.amount,
            nameOrig=self.nameOrig,
            oldbalanceOrg=self.oldbalanceOrg,
            newbalanceOrig=self.newbalanceOrig,
            nameDest=self.nameDest,
            oldbalanceDest=self.oldbalanceDest,
            newbalanceDest=self.newbalanceDest
        )


class TransactionResultEvent(BaseModel):
    """
    Structured outcome emitted to sentinel.fraud-predictions topic post-scoring.
    """
    transaction_id: str
    status: Literal["PROCESSED", "REJECTED_TEMPORAL_LOOKAHEAD", "FAILED", "DUPLICATE_IGNORED"]
    hybrid_risk_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    assigned_risk_tier: Optional[Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"]] = None
    recommended_action: Optional[Literal[
        "AUTOMATED_BLOCK",
        "STEP_UP_AUTHENTICATION",
        "MANUAL_ANALYST_REVIEW",
        "AUTOMATED_CLEAR_AND_SETTLE"
    ]] = None
    latency_ms: Optional[float] = Field(None, ge=0.0)
    processed_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    error_reason: Optional[str] = None


class DeadLetterEvent(BaseModel):
    """
    Dead-letter queue record capturing unprocessable, malformed, or retry-exhausted events.
    Emitted to sentinel.transactions.dlq topic.
    """
    event_id: str
    original_payload: Any
    error_type: Literal[
        "VALIDATION_ERROR",
        "TEMPORAL_LOOKAHEAD_ERROR",
        "INFERENCE_ERROR",
        "CORRUPT_PAYLOAD",
        "RETRY_EXHAUSTED"
    ]
    error_message: str
    retry_count: int = 0
    failed_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
