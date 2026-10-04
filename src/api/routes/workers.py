"""
Worker control and telemetry REST API routes.
Part 6: JWT authentication + role-based access control.
"""

from typing import Any, Dict, List
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from src.auth.dependencies import get_current_user, require_role
from src.workers.worker_manager import WorkerManager, get_worker_manager

router = APIRouter(prefix="/api/workers", tags=["Workers"])


class WorkerSpeedSchema(BaseModel):
    speed: float = Field(1.0, ge=0.25, le=8.0)


@router.get("/status", response_model=List[Dict[str, Any]])
def get_worker_statuses(
    worker_mgr: WorkerManager = Depends(get_worker_manager),
    current_user=Depends(get_current_user),
):
    """Get real-time operational telemetry for all camera workers. Requires authentication."""
    return worker_mgr.get_statuses()


@router.get("/{camera_id}/status", response_model=Dict[str, Any])
def get_single_worker_status(
    camera_id: str,
    worker_mgr: WorkerManager = Depends(get_worker_manager),
    current_user=Depends(get_current_user),
):
    """Get telemetry for a specific camera worker (e.g. C1). Requires authentication."""
    st = worker_mgr.get_worker_status(camera_id)
    if not st:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Worker for camera '{camera_id}' not found",
        )
    return st


@router.post("/start-all", response_model=Dict[str, bool])
def start_all_workers(
    worker_mgr: WorkerManager = Depends(get_worker_manager),
    current_user=require_role("ADMIN", "POLICE"),
):
    """Start all configured camera worker threads. Requires ADMIN or POLICE role."""
    return worker_mgr.start_all()


@router.post("/stop-all", response_model=Dict[str, str])
def stop_all_workers(
    worker_mgr: WorkerManager = Depends(get_worker_manager),
    current_user=require_role("ADMIN"),
):
    """Stop all running camera worker threads. Requires ADMIN role."""
    worker_mgr.stop_all()
    return {"message": "All camera workers stopped"}


@router.post("/{camera_id}/start", response_model=Dict[str, str])
def start_worker(
    camera_id: str,
    worker_mgr: WorkerManager = Depends(get_worker_manager),
    current_user=require_role("ADMIN", "POLICE"),
):
    """Start specific camera worker thread. Requires ADMIN or POLICE role."""
    success = worker_mgr.start_worker(camera_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not start worker '{camera_id}'",
        )
    return {"message": f"Worker '{camera_id}' started"}


@router.post("/{camera_id}/stop", response_model=Dict[str, str])
def stop_worker(
    camera_id: str,
    worker_mgr: WorkerManager = Depends(get_worker_manager),
    current_user=require_role("ADMIN"),
):
    """Stop specific camera worker thread. Requires ADMIN role."""
    success = worker_mgr.stop_worker(camera_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Worker '{camera_id}' not found",
        )
    return {"message": f"Worker '{camera_id}' stopped"}


@router.post("/{camera_id}/speed", response_model=Dict[str, Any])
def set_worker_speed(
    camera_id: str,
    payload: WorkerSpeedSchema,
    worker_mgr: WorkerManager = Depends(get_worker_manager),
    current_user=require_role("ADMIN", "POLICE"),
):
    """Set playback speed on a camera worker (1.0x, 2.0x, etc.)."""
    success = worker_mgr.set_worker_playback_speed(camera_id, payload.speed)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Worker '{camera_id}' not active or not found",
        )
    return {"message": f"Worker '{camera_id}' playback speed set to {payload.speed}x", "speed": payload.speed}
