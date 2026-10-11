"""
Alert Management Endpoints: /api/v1/alerts
Provides persistent fraud alert queuing, retrieval, and analyst triage lifecycle operations.
"""

from datetime import datetime, timezone
import logging
from typing import Optional
import uuid

from fastapi import APIRouter, HTTPException, Query, Response, status

from backend.app.core.config import settings
from backend.app.core.database import (
    create_alert,
    get_alert,
    get_alert_by_transaction_id,
    list_alerts,
    update_alert
)
from backend.app.schemas.alert import (
    AlertCreateRequest,
    AlertUpdateRequest,
    AlertResponse,
    AlertListResponse
)
from backend.app.services.inference_service import inference_service

logger = logging.getLogger("sentinel.api.alerts")
router = APIRouter()

TERMINAL_STATUSES = {"CONFIRMED_FRAUD", "FALSE_POSITIVE", "CLOSED"}


def _get_active_db_path():
    """Retrieves active SQLite database path, supporting dynamic test overrides."""
    return getattr(inference_service, "db_path", settings.LEDGER_DB_PATH)


@router.post(
    "",
    response_model=AlertResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Fraud Alert",
    description="Persists a fraud alert snapshot for analyst triage. Idempotent for identical retries."
)
def create_fraud_alert(payload: AlertCreateRequest, response: Response) -> AlertResponse:
    db_path = _get_active_db_path()

    # Check if an alert already exists for this transaction_id
    existing = get_alert_by_transaction_id(db_path, payload.transaction_id)
    if existing:
        is_identical = (
            existing["step"] == payload.step and
            existing["type"] == payload.type and
            abs(existing["amount"] - payload.amount) < 1e-4 and
            existing["name_orig"] == payload.nameOrig and
            existing["name_dest"] == payload.nameDest and
            abs(existing["hybrid_risk_score"] - payload.hybrid_risk_score) < 1e-4 and
            existing["risk_tier"] == payload.risk_tier and
            existing["recommended_action"] == payload.recommended_action
        )
        if is_identical:
            logger.info("Idempotent alert submission for transaction %s; returning existing record.", payload.transaction_id)
            response.status_code = status.HTTP_200_OK
            return AlertResponse(**existing)
        else:
            logger.warning("Conflicting alert submission for transaction %s.", payload.transaction_id)
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Alert for transaction '{payload.transaction_id}' already exists with conflicting snapshot data."
            )

    alert_id = f"ALT_{uuid.uuid4().hex[:12]}"
    created_row, was_created = create_alert(
        db_path=db_path,
        alert_id=alert_id,
        transaction_id=payload.transaction_id,
        step=payload.step,
        type_=payload.type,
        amount=payload.amount,
        name_orig=payload.nameOrig,
        name_dest=payload.nameDest,
        hybrid_risk_score=payload.hybrid_risk_score,
        risk_tier=payload.risk_tier,
        recommended_action=payload.recommended_action,
        status="OPEN"
    )

    if not was_created:
        # Race condition resolution: check identical vs conflicting
        is_identical = (
            created_row["step"] == payload.step and
            created_row["type"] == payload.type and
            abs(created_row["amount"] - payload.amount) < 1e-4 and
            created_row["name_orig"] == payload.nameOrig and
            created_row["name_dest"] == payload.nameDest and
            abs(created_row["hybrid_risk_score"] - payload.hybrid_risk_score) < 1e-4 and
            created_row["risk_tier"] == payload.risk_tier and
            created_row["recommended_action"] == payload.recommended_action
        )
        if is_identical:
            response.status_code = status.HTTP_200_OK
            return AlertResponse(**created_row)
        else:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Alert for transaction '{payload.transaction_id}' already exists with conflicting snapshot data."
            )

    logger.info("Created fraud alert %s for transaction %s (tier: %s).", alert_id, payload.transaction_id, payload.risk_tier)
    return AlertResponse(**created_row)


@router.get(
    "",
    response_model=AlertListResponse,
    status_code=status.HTTP_200_OK,
    summary="List Fraud Alerts",
    description="Queries alerts queue with optional filtering by status, risk tier, or minimum risk score."
)
def list_fraud_alerts(
    status: Optional[str] = Query(None, description="Filter by alert status"),
    risk_tier: Optional[str] = Query(None, description="Filter by risk tier"),
    min_score: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum hybrid risk score"),
    limit: int = Query(50, ge=1, le=100, description="Page limit"),
    offset: int = Query(0, ge=0, description="Page offset")
) -> AlertListResponse:
    db_path = _get_active_db_path()
    items, total = list_alerts(
        db_path=db_path,
        status=status,
        risk_tier=risk_tier,
        min_score=min_score,
        limit=limit,
        offset=offset
    )
    return AlertListResponse(
        total=total,
        limit=limit,
        offset=offset,
        items=[AlertResponse(**row) for row in items]
    )


@router.get(
    "/{alert_id}",
    response_model=AlertResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Fraud Alert",
    description="Retrieves a specific fraud alert by unique alert identifier."
)
def get_fraud_alert(alert_id: str) -> AlertResponse:
    db_path = _get_active_db_path()
    record = get_alert(db_path, alert_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Alert '{alert_id}' not found."
        )
    return AlertResponse(**record)


@router.patch(
    "/{alert_id}",
    response_model=AlertResponse,
    status_code=status.HTTP_200_OK,
    summary="Update Alert Status & Analyst Triage",
    description="Performs atomic status transitions, assigns analyst, or records resolution decisions."
)
def update_fraud_alert(alert_id: str, update: AlertUpdateRequest) -> AlertResponse:
    db_path = _get_active_db_path()
    current = get_alert(db_path, alert_id)
    if not current:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Alert '{alert_id}' not found."
        )

    current_status = current["status"]
    target_status = update.status

    # 1. Enforce terminal state immutability
    if current_status in TERMINAL_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Cannot update alert in terminal state '{current_status}'. Reopening is not permitted."
        )

    analyst_id = update.analyst_id if update.analyst_id is not None else current["analyst_id"]
    resolution_notes = update.resolution_notes if update.resolution_notes is not None else current["resolution_notes"]

    # 2. Lifecycle transition rules
    if current_status == target_status:
        if current_status == "INVESTIGATING":
            if update.analyst_id and update.analyst_id != current["analyst_id"]:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Alert '{alert_id}' is already claimed by analyst '{current['analyst_id']}'. "
                           f"It must be released back to 'OPEN' before being re-assigned."
                )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Alert '{alert_id}' is already in status '{current_status}'."
        )

    if current_status == "OPEN":
        if target_status in {"CONFIRMED_FRAUD", "FALSE_POSITIVE"}:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid transition: Cannot move alert from 'OPEN' directly to '{target_status}'. "
                       f"Alert must be claimed into 'INVESTIGATING' status first."
            )
        elif target_status == "INVESTIGATING":
            if not analyst_id or not analyst_id.strip():
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="analyst_id is required when assigning an alert to 'INVESTIGATING'."
                )

    elif current_status == "INVESTIGATING":
        if target_status == "OPEN":
            # Releasing alert back to queue: analyst assignment must be cleared
            analyst_id = None
        elif target_status in {"CONFIRMED_FRAUD", "FALSE_POSITIVE"}:
            if not analyst_id or not analyst_id.strip():
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"analyst_id is required when resolving an alert as '{target_status}'."
                )
            if not update.resolution_notes or not update.resolution_notes.strip():
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"resolution_notes are required when resolving an alert as '{target_status}'."
                )
            resolution_notes = update.resolution_notes.strip()

    updated_at = datetime.now(timezone.utc).isoformat()
    updated_record, was_updated = update_alert(
        db_path=db_path,
        alert_id=alert_id,
        new_status=target_status,
        analyst_id=analyst_id,
        resolution_notes=resolution_notes,
        updated_at=updated_at,
        expected_status=current_status
    )

    if not was_updated:
        if not updated_record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Alert '{alert_id}' not found during update."
            )
        latest_status = updated_record["status"]
        if latest_status in TERMINAL_STATUSES:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Cannot update alert '{alert_id}'. It transitioned to terminal state '{latest_status}' concurrently."
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Concurrent modification detected on alert '{alert_id}'. Expected status '{current_status}', but current status is '{latest_status}'."
            )

    logger.info("Updated alert %s status from %s to %s (analyst: %s).", alert_id, current_status, target_status, analyst_id)
    return AlertResponse(**updated_record)
