"""
Worker control and telemetry REST API routes.
"""

from typing import Any, Dict, List
from fastapi import APIRouter, Depends, HTTPException, status

from src.workers.worker_manager import WorkerManager, get_worker_manager

router = APIRouter(prefix="/api/workers", tags=["Workers"])


@router.get("/status", response_model=List[Dict[str, Any]])
def get_worker_statuses(worker_mgr: WorkerManager = Depends(get_worker_manager)):
    """Get real-time operational telemetry for all camera workers."""
    return worker_mgr.get_statuses()


@router.get("/{camera_id}/status", response_model=Dict[str, Any])
def get_single_worker_status(
    camera_id: str,
    worker_mgr: WorkerManager = Depends(get_worker_manager),
):
    """Get telemetry for a specific camera worker (e.g. C1)."""
    st = worker_mgr.get_worker_status(camera_id)
    if not st:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Worker for camera '{camera_id}' not found",
        )
    return st


@router.post("/start-all", response_model=Dict[str, bool])
def start_all_workers(worker_mgr: WorkerManager = Depends(get_worker_manager)):
    """Start all configured camera worker threads."""
    return worker_mgr.start_all()


@router.post("/stop-all", response_model=Dict[str, str])
def stop_all_workers(worker_mgr: WorkerManager = Depends(get_worker_manager)):
    """Stop all running camera worker threads."""
    worker_mgr.stop_all()
    return {"message": "All camera workers stopped"}


@router.post("/{camera_id}/start", response_model=Dict[str, str])
def start_worker(camera_id: str, worker_mgr: WorkerManager = Depends(get_worker_manager)):
    """Start specific camera worker thread."""
    success = worker_mgr.start_worker(camera_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not start worker '{camera_id}'",
        )
    return {"message": f"Worker '{camera_id}' started"}


@router.post("/{camera_id}/stop", response_model=Dict[str, str])
def stop_worker(camera_id: str, worker_mgr: WorkerManager = Depends(get_worker_manager)):
    """Stop specific camera worker thread."""
    success = worker_mgr.stop_worker(camera_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Worker '{camera_id}' not found",
        )
    return {"message": f"Worker '{camera_id}' stopped"}

