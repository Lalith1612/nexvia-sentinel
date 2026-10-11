"""
API v1 Router Aggregator
"""

from fastapi import APIRouter
from backend.app.api.v1.endpoints import predict, health, model, events, alerts, replay

api_router = APIRouter()
api_router.include_router(predict.router, prefix="/predict", tags=["Prediction"])
api_router.include_router(health.router, prefix="/health", tags=["Health"])
api_router.include_router(model.router, prefix="/model", tags=["Model"])
api_router.include_router(events.router, prefix="/events", tags=["Events"])
api_router.include_router(alerts.router, prefix="/alerts", tags=["Alerts"])
api_router.include_router(replay.router, prefix="/replay", tags=["Replay"])
