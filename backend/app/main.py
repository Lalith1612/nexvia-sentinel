"""
Sentinel Real-Time Fraud Detection Platform — FastAPI Application
"""

from contextlib import asynccontextmanager
import logging
from typing import Dict, Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from backend.app.core.config import settings
from backend.app.api.v1.api import api_router
from backend.app.services.inference_service import inference_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
)
logger = logging.getLogger("sentinel.app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Loads models, pipelines, calibrators, and graph snapshot once at startup.
    Ensures zero retraining, re-fitting, or disk dataset re-reads inside request path.
    """
    logger.info("Starting Sentinel Fraud Inference Service (v%s)...", settings.VERSION)
    try:
        inference_service.load_artifacts()
        inference_service.initialize_graph_state()
        logger.info("Sentinel artifacts and graph state snapshot loaded successfully.")
    except Exception as exc:
        logger.critical("Failed to load Sentinel artifacts during startup: %s", str(exc), exc_info=True)
        inference_service.initialization_error = str(exc)

    # Optional background Kafka consumer worker (Phase 6)
    if settings.KAFKA_ENABLED:
        try:
            from backend.app.workers.kafka_consumer import consumer_worker
            consumer_worker.start()
            logger.info("Sentinel Kafka background consumer worker started.")
        except Exception as kafka_err:
            logger.error("Failed to start Kafka consumer worker: %s", str(kafka_err))

    yield

    if settings.KAFKA_ENABLED:
        try:
            from backend.app.workers.kafka_consumer import consumer_worker
            consumer_worker.stop()
            logger.info("Sentinel Kafka background consumer worker stopped.")
        except Exception as kafka_stop_err:
            logger.warning("Error stopping Kafka consumer worker: %s", str(kafka_stop_err))

    logger.info("Shutting down Sentinel Fraud Inference Service.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Real-Time Financial Fraud Detection Platform combining XGBoost, PyTorch Geometric GraphSAGE, and Platt Calibration.",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS middleware for local frontend dashboard and authorized origins
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API v1 endpoints
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Sanitized validation error response with structured details."""
    errors = []
    for err in exc.errors():
        loc = " -> ".join([str(x) for x in err.get("loc", []) if x != "body"])
        errors.append({
            "field": loc or "payload",
            "message": err.get("msg", "Invalid value"),
            "type": err.get("type", "value_error")
        })
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Unprocessable Entity",
            "message": "Input validation failed for transaction payload.",
            "details": errors
        }
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Sanitized internal server error handler avoiding secret and file path leakage."""
    logger.exception("Unhandled server exception on %s %s: %s", request.method, request.url.path, str(exc))
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred while processing the request."
        }
    )


@app.get("/", tags=["Root"])
def root_status() -> Dict[str, Any]:
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs_url": "/docs",
        "health_endpoint": f"{settings.API_V1_STR}/health",
        "predict_endpoint": f"{settings.API_V1_STR}/predict/fraud",
        "model_info_endpoint": f"{settings.API_V1_STR}/model/info"
    }
