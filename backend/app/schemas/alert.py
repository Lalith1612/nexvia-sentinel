"""
Pydantic Schemas for Sentinel Fraud Alert Management & Analyst Triage
"""

import math
from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator

from backend.app.core.config import settings


def _validate_finite(value: Optional[float], field_name: str) -> Optional[float]:
    if value is not None and (math.isnan(value) or math.isinf(value)):
        raise ValueError(f"{field_name} must be a finite numerical value.")
    return value


def get_canonical_tier_and_action(score: float) -> tuple[str, str]:
    """Derives canonical risk tier and recommended action from hybrid risk score."""
    if score >= settings.THRESHOLD_CRITICAL:
        return "CRITICAL", "AUTOMATED_BLOCK"
    elif score >= settings.THRESHOLD_HIGH:
        return "HIGH", "STEP_UP_AUTHENTICATION"
    elif score >= settings.THRESHOLD_MEDIUM:
        return "MEDIUM", "MANUAL_ANALYST_REVIEW"
    else:
        return "LOW", "AUTOMATED_CLEAR_AND_SETTLE"


class AlertCreateRequest(BaseModel):
    transaction_id: str = Field(..., min_length=1, max_length=128, description="Unique transaction identifier")
    step: int = Field(..., ge=1, description="Simulation step / time index (1-indexed)")
    type: Literal["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"] = Field(
        ..., description="Transaction payment type"
    )
    amount: float = Field(..., gt=0.0, description="Transaction amount in currency units")
    nameOrig: str = Field(..., min_length=2, max_length=64, description="Origin account identifier")
    nameDest: str = Field(..., min_length=2, max_length=64, description="Destination account identifier")
    hybrid_risk_score: float = Field(..., ge=0.0, le=1.0, description="Hybrid calibrated fraud risk score")
    risk_tier: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"] = Field(
        ..., description="Risk tier assigned to the transaction"
    )
    recommended_action: Literal[
        "AUTOMATED_BLOCK",
        "STEP_UP_AUTHENTICATION",
        "MANUAL_ANALYST_REVIEW",
        "AUTOMATED_CLEAR_AND_SETTLE"
    ] = Field(..., description="Recommended operational action")

    @field_validator("amount", "hybrid_risk_score")
    @classmethod
    def check_finite(cls, v, info):
        return _validate_finite(v, info.field_name)

    @model_validator(mode="after")
    def verify_consistency(self) -> "AlertCreateRequest":
        expected_tier, expected_action = get_canonical_tier_and_action(self.hybrid_risk_score)
        if self.risk_tier != expected_tier:
            raise ValueError(
                f"Inconsistent risk tier: hybrid_risk_score {self.hybrid_risk_score:.4f} corresponds to "
                f"'{expected_tier}', but '{self.risk_tier}' was provided."
            )
        if self.recommended_action != expected_action:
            raise ValueError(
                f"Inconsistent recommended action: risk_tier '{self.risk_tier}' maps to "
                f"'{expected_action}', but '{self.recommended_action}' was provided."
            )
        return self


class AlertUpdateRequest(BaseModel):
    status: Literal["OPEN", "INVESTIGATING", "CONFIRMED_FRAUD", "FALSE_POSITIVE", "CLOSED"] = Field(
        ..., description="Target alert lifecycle status"
    )
    analyst_id: Optional[str] = Field(
        None, min_length=1, max_length=64,
        description="Unverified client-supplied analyst identifier (authentication is not implemented)"
    )
    resolution_notes: Optional[str] = Field(
        None, max_length=1024,
        description="Analyst investigation findings or resolution rationale"
    )

    @field_validator("analyst_id", "resolution_notes")
    @classmethod
    def strip_whitespace(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            stripped = v.strip()
            return stripped if stripped else None
        return None


class AlertResponse(BaseModel):
    alert_id: str = Field(..., description="Unique alert identifier")
    transaction_id: str = Field(..., description="Unique transaction identifier")
    step: int = Field(..., ge=1, description="Transaction simulation step")
    type: str = Field(..., description="Transaction payment type")
    amount: float = Field(..., gt=0.0, description="Transaction amount")
    name_orig: str = Field(..., description="Origin account identifier")
    name_dest: str = Field(..., description="Destination account identifier")
    hybrid_risk_score: float = Field(..., ge=0.0, le=1.0, description="Hybrid calibrated risk score")
    risk_tier: str = Field(..., description="Assigned risk tier")
    recommended_action: str = Field(..., description="Recommended action")
    status: Literal["OPEN", "INVESTIGATING", "CONFIRMED_FRAUD", "FALSE_POSITIVE", "CLOSED"] = Field(
        ..., description="Current alert lifecycle status"
    )
    analyst_id: Optional[str] = Field(
        None, description="Unverified client-supplied analyst identifier (authentication is not implemented)"
    )
    resolution_notes: Optional[str] = Field(None, description="Analyst triage or resolution notes")
    created_at: str = Field(..., description="UTC ISO-8601 creation timestamp")
    updated_at: str = Field(..., description="UTC ISO-8601 last-updated timestamp")


class AlertListResponse(BaseModel):
    total: int = Field(..., ge=0, description="Total number of alerts matching query filters")
    limit: int = Field(..., ge=1, description="Pagination page limit")
    offset: int = Field(..., ge=0, description="Pagination offset")
    items: List[AlertResponse] = Field(..., description="Paginated list of alerts")
