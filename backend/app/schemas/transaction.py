"""
Pydantic Schemas for Sentinel Fraud Inference API
"""

import math
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


def _validate_finite(value: Optional[float], field_name: str) -> Optional[float]:
    if value is not None and (math.isnan(value) or math.isinf(value)):
        raise ValueError(f"{field_name} must be a finite numerical value.")
    return value


class TransactionRequest(BaseModel):
    step: int = Field(..., ge=1, description="Simulation step / time index (1-indexed)")
    type: Literal["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"] = Field(
        ..., description="Transaction type"
    )
    amount: float = Field(..., gt=0.0, description="Transaction amount in currency units")
    nameOrig: str = Field(..., min_length=2, max_length=64, description="Origin account identifier (e.g. C123456789)")
    oldbalanceOrg: float = Field(..., ge=0.0, description="Origin balance immediately prior to transaction execution")
    newbalanceOrig: Optional[float] = Field(
        None, ge=0.0, description="Optional post-settlement origin balance (ignored during pre-authorization scoring)"
    )
    nameDest: str = Field(..., min_length=2, max_length=64, description="Destination account identifier (e.g. M987654321)")
    oldbalanceDest: float = Field(..., ge=0.0, description="Destination balance immediately prior to transaction execution")
    newbalanceDest: Optional[float] = Field(
        None, ge=0.0, description="Optional post-settlement destination balance (ignored during pre-authorization scoring)"
    )

    @field_validator("amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest")
    @classmethod
    def check_finite(cls, v, info):
        return _validate_finite(v, info.field_name)


class SettlementEventRequest(BaseModel):
    transaction_id: str = Field(..., min_length=1, max_length=128, description="Unique transaction settlement identifier")
    step: int = Field(..., ge=1, description="Simulation step / settlement timestamp index")
    type: Literal["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"] = Field(
        ..., description="Transaction type"
    )
    amount: float = Field(..., gt=0.0, description="Settled amount in currency units")
    nameOrig: str = Field(..., min_length=2, max_length=64, description="Origin account identifier")
    nameDest: str = Field(..., min_length=2, max_length=64, description="Destination account identifier")

    @field_validator("amount")
    @classmethod
    def check_amount_finite(cls, v):
        return _validate_finite(v, "amount")


class SettlementEventResponse(BaseModel):
    transaction_id: str
    status: Literal["SETTLED", "DUPLICATE_IGNORED", "REJECTED"]
    settled_at: str
    message: str
    graph_out_degree_orig: float
    graph_in_degree_dest: float
    refresh_latency_ms: Optional[float] = None


class ScoreComponent(BaseModel):
    raw_score: float = Field(..., ge=0.0, le=1.0)
    calibrated_probability: float = Field(..., ge=0.0, le=1.0)
    ensemble_weight: float = Field(..., ge=0.0, le=1.0)
    weighted_contribution: float = Field(..., ge=0.0, le=1.0)


class FeatureAttribution(BaseModel):
    feature: str
    value: float
    shap_contribution: float
    impact: Literal["INCREASES_FRAUD_RISK", "REDUCES_FRAUD_RISK"]


class GraphTopologicalContext(BaseModel):
    account_orig_masked: str
    account_dest_masked: str
    is_destination_merchant: bool
    orig_historical_out_degree: float
    orig_historical_out_volume: float
    dest_historical_in_degree: float
    dest_historical_in_volume: float
    raw_gnn_anomaly_score: float
    calibrated_gnn_risk: float
    relational_weight_in_ensemble: float
    node_orig_status: Literal["SETTLED_SNAPSHOT", "COLD_START_INDUCTIVE", "SETTLED_PENDING_REFRESH"]
    node_dest_status: Literal["SETTLED_SNAPSHOT", "COLD_START_INDUCTIVE", "SETTLED_PENDING_REFRESH"]


class FraudPredictionResponse(BaseModel):
    transaction_id: str
    hybrid_risk_score: float = Field(..., ge=0.0, le=1.0)
    assigned_risk_tier: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    recommended_action: Literal[
        "AUTOMATED_BLOCK",
        "STEP_UP_AUTHENTICATION",
        "MANUAL_ANALYST_REVIEW",
        "AUTOMATED_CLEAR_AND_SETTLE"
    ]
    score_components: Dict[str, ScoreComponent]
    tabular_shap_attributions: List[FeatureAttribution]
    graph_topological_context: GraphTopologicalContext
    regulatory_disclaimers: List[str]
    latency_ms: float = Field(..., ge=0.0)


class HealthResponse(BaseModel):
    status: Literal["HEALTHY", "DEGRADED", "UNHEALTHY"]
    service_name: str
    version: str
    artifacts_loaded: bool
    graph_state_initialized: bool
    snapshot_cutoff_step: int
    max_settled_step: int
    total_indexed_nodes: int
    settled_edges_count: int
    torch_device: str
    memory_rss_mb: float


class ModelInfoResponse(BaseModel):
    model_architecture: str
    ensemble_weights: Dict[str, float]
    risk_tier_thresholds: Dict[str, float]
    tier_actions: Dict[str, str]
    calibration_method: str
    validation_dataset: str
    governance_disclaimers: List[str]
    operational_prototype_boundary: str
