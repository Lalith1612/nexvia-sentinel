"""
Health Endpoint: GET /api/v1/health
"""

from fastapi import APIRouter, status
from backend.app.schemas.transaction import HealthResponse
from backend.app.services.inference_service import inference_service

router = APIRouter()


@router.get(
    "",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Service Health & Diagnostic Status",
    description="Returns liveness, readiness, artifact loading verification, and memory RSS usage."
)
def get_health() -> HealthResponse:
    return inference_service.get_health_status()
