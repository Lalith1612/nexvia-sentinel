"""
Predict Endpoint: POST /api/v1/predict/fraud
"""

import logging
from fastapi import APIRouter, HTTPException, status
from backend.app.schemas.transaction import TransactionRequest, FraudPredictionResponse
from backend.app.services.inference_service import inference_service

logger = logging.getLogger("sentinel.api.predict")
router = APIRouter()


@router.post(
    "/fraud",
    response_model=FraudPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict Transaction Fraud Risk",
    description="Real-time pre-authorization fraud inference combining Tabular XGBoost, Relational GraphSAGE, and Platt calibration."
)
def predict_fraud(transaction: TransactionRequest) -> FraudPredictionResponse:
    try:
        response = inference_service.predict_transaction(transaction)
        return response
    except ValueError as ve:
        logger.warning("Validation violation during inference: %s", str(ve))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except RuntimeError as re:
        logger.error("Service runtime error during inference: %s", str(re))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference service is not ready or models are unavailable."
        )
    except Exception as e:
        logger.exception("Unexpected error during fraud prediction: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal inference failure. Transaction could not be scored."
        )
