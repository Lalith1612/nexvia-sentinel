"""
Model Info Endpoint: GET /api/v1/model/info
"""

import logging
from fastapi import APIRouter, HTTPException, status
from backend.app.schemas.transaction import ModelInfoResponse
from backend.app.services.inference_service import inference_service

logger = logging.getLogger("sentinel.api.model")
router = APIRouter()


@router.get(
    "/info",
    response_model=ModelInfoResponse,
    status_code=status.HTTP_200_OK,
    summary="Model Metadata & Governance Info",
    description="Exposes ensemble weights, risk tier boundaries, calibration info, and regulatory disclaimers."
)
def get_model_info() -> ModelInfoResponse:
    try:
        return inference_service.get_model_info()
    except Exception as e:
        logger.exception("Failed to retrieve model info: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model metadata is currently unavailable."
        )
