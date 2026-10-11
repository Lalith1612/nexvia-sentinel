"""
Replay API Endpoints: /api/v1/replay
Provides submission, status monitoring, and cooperative cancellation for transaction replay jobs.
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Query, Response, status

from backend.app.core.config import settings
from backend.app.core.database import (
    get_replay_job,
    list_replay_jobs
)
from backend.app.schemas.replay import (
    ReplayJobCreateRequest,
    ReplayJobResponse,
    ReplayJobListResponse
)
from backend.app.services.inference_service import inference_service
from backend.app.services.replay_service import replay_service

logger = logging.getLogger("sentinel.api.replay")
router = APIRouter()


def _get_active_db_path():
    """Retrieves active SQLite database path, supporting dynamic test overrides."""
    return getattr(inference_service, "db_path", settings.LEDGER_DB_PATH)


@router.post(
    "/jobs",
    response_model=ReplayJobResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit Transaction Replay Job",
    description="Submits a bounded, read-only DRY_RUN transaction replay job. Processes transactions chronologically."
)
def create_replay_job_endpoint(
    payload: ReplayJobCreateRequest,
    response: Response
) -> ReplayJobResponse:
    db_path = _get_active_db_path()

    try:
        job_dict, status_code = replay_service.submit_job(
            db_path=db_path,
            request=payload
        )
        response.status_code = status_code
        return ReplayJobResponse(**job_dict)

    except ValueError as ve:
        # Idempotency conflict
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(ve)
        )
    except RuntimeError as re:
        msg = str(re)
        if "dispatch" in msg.lower():
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=msg
            )
        # Concurrent replay execution conflict
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=msg
        )
    except FileNotFoundError as fe:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(fe)
        )
    except Exception as exc:
        logger.exception("Unexpected error creating replay job: %s", str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to initialize transaction replay job."
        )


@router.get(
    "/jobs",
    response_model=ReplayJobListResponse,
    status_code=status.HTTP_200_OK,
    summary="List Replay Jobs",
    description="Queries replay jobs with optional status filter and pagination."
)
def list_replay_jobs_endpoint(
    status: Optional[str] = Query(None, description="Filter by job status (PENDING, RUNNING, COMPLETED, FAILED, CANCELLED)"),
    limit: int = Query(50, ge=1, le=100, description="Page limit"),
    offset: int = Query(0, ge=0, description="Page offset")
) -> ReplayJobListResponse:
    db_path = _get_active_db_path()
    items, total = list_replay_jobs(
        db_path=db_path,
        status=status,
        limit=limit,
        offset=offset
    )
    for row in items:
        if row["status"] == "RUNNING" and row["job_id"] in replay_service._active_job_progress:
            row["progress"] = replay_service._active_job_progress[row["job_id"]]
    return ReplayJobListResponse(
        total=total,
        limit=limit,
        offset=offset,
        items=[ReplayJobResponse(**row) for row in items]
    )


@router.get(
    "/jobs/{job_id}",
    response_model=ReplayJobResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Replay Job Status & Metrics",
    description="Retrieves status, progress counters, and aggregate latency metrics for a specific replay job."
)
def get_replay_job_endpoint(job_id: str) -> ReplayJobResponse:
    db_path = _get_active_db_path()
    job = replay_service.get_job(db_path, job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Replay job '{job_id}' not found."
        )
    return ReplayJobResponse(**job)


@router.post(
    "/jobs/{job_id}/cancel",
    response_model=ReplayJobResponse,
    status_code=status.HTTP_200_OK,
    summary="Cancel Replay Job",
    description="Signals cooperative cancellation to an in-flight or pending replay job."
)
def cancel_replay_job_endpoint(job_id: str) -> ReplayJobResponse:
    db_path = _get_active_db_path()
    try:
        job, status_code = replay_service.cancel_job(db_path, job_id)
        return ReplayJobResponse(**job)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Replay job '{job_id}' not found."
        )
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(ve)
        )
    except Exception as exc:
        logger.exception("Unexpected error cancelling replay job %s: %s", job_id, str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cancel replay job."
        )
