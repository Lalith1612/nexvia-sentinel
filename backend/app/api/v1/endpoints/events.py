"""
Settlement Events Endpoint: POST /api/v1/events/settlement
"""

import logging
from fastapi import APIRouter, HTTPException, status
from backend.app.schemas.transaction import SettlementEventRequest, SettlementEventResponse
from backend.app.services.inference_service import inference_service

logger = logging.getLogger("sentinel.api.events")
router = APIRouter()


@router.post(
    "/settlement",
    response_model=SettlementEventResponse,
    status_code=status.HTTP_200_OK,
    summary="Process Confirmed Settlement Event",
    description="Updates graph degrees and settled edges post-authorization. Idempotent across repeated transaction_id notifications."
)
async def process_settlement(event: SettlementEventRequest) -> SettlementEventResponse:
    try:
        response = await inference_service.process_settlement_event(event)
        return response
    except ValueError as ve:
        logger.warning("Validation error processing settlement event: %s", str(ve))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        logger.exception("Failed to process settlement event: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update graph state with settlement event."
        )
