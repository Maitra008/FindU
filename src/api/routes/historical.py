"""
Historical Video Upload & Analysis REST API routes.
Enables asynchronous background search and replay-as-camera functionality for offline/recorded footage.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

import cv2
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy.orm import Session

from src.auth.dependencies import get_current_user, require_role
from src.config import PROJECT_ROOT
from src.db.database import get_db
from src.db.repositories.cameras import CameraRepository
from src.db.repositories.historical import HistoricalJobRepository
from src.services.audit_service import get_audit_service
from src.services.historical_service import HistoricalService, get_historical_service
from src.workers.worker_manager import get_worker_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/historical", tags=["Historical Footage"])

HISTORICAL_UPLOADS_DIR = PROJECT_ROOT / "data" / "historical_uploads"
HISTORICAL_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
MAX_VIDEO_SIZE_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB limit


def _safe_resolve_upload(filename: str) -> Path:
    """Sanitize and ensure path remains strictly inside HISTORICAL_UPLOADS_DIR."""
    clean_name = Path(filename).name
    dest = (HISTORICAL_UPLOADS_DIR / clean_name).resolve()
    if not str(dest).startswith(str(HISTORICAL_UPLOADS_DIR.resolve())):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file path trajectory detected",
        )
    return dest


@router.post("/upload", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
async def upload_historical_footage(
    file: UploadFile = File(...),
    camera_id: str = Form("C5"),
    camera_name: Optional[str] = Form(None),
    location: Optional[str] = Form(None),
    mode: str = Form("replay"),  # 'replay' or 'search'
    sample_fps: float = Form(2.0),
    db: Session = Depends(get_db),
    historical_svc: HistoricalService = Depends(get_historical_service),
    current_user=require_role("ADMIN"),
):
    """
    Upload historical CCTV video footage (ADMIN only).
    - If mode == 'replay': Configures camera for synchronized live replay and starts CameraWorker.
    - If mode == 'search': Dispatches asynchronous background processing job.
    """
    ext = Path(file.filename or "footage.mp4").suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported video format '{ext}'. Allowed: {ALLOWED_VIDEO_EXTENSIONS}",
        )

    job_id = f"JOB-{uuid.uuid4().hex[:8].upper()}"
    saved_filename = f"{job_id}_{Path(file.filename).name}"
    target_path = _safe_resolve_upload(saved_filename)

    # Stream file to disk
    bytes_written = 0
    with open(target_path, "wb") as out:
        while chunk := await file.read(1024 * 1024):  # 1MB chunks
            bytes_written += len(chunk)
            if bytes_written > MAX_VIDEO_SIZE_BYTES:
                out.close()
                target_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Video exceeds maximum allowed size of {MAX_VIDEO_SIZE_BYTES // (1024*1024)} MB",
                )
            out.write(chunk)

    rel_path = str(target_path.relative_to(PROJECT_ROOT)).replace("\\", "/")

    # Get duration
    cap = cv2.VideoCapture(str(target_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration_sec = total_frames / fps if fps > 0 else 0.0
    cap.release()

    # Create HistoricalJob record
    job_repo = HistoricalJobRepository(db)
    job = job_repo.create(
        job_id=job_id,
        camera_id=camera_id,
        file_path=rel_path,
        sample_fps=sample_fps,
        total_duration_sec=duration_sec,
    )

    get_audit_service().log(
        event="historical.uploaded",
        actor_username=current_user.username,
        resource_type="historical_job",
        resource_id=job_id,
        detail={"camera_id": camera_id, "mode": mode, "filename": file.filename},
    )

    if mode == "search":
        # Launch background search thread
        historical_svc.start_background_search(
            job_id=job_id,
            camera_id=camera_id,
            file_path=rel_path,
            sample_fps=sample_fps,
        )
        return {
            "mode": "search",
            "job": job.to_dict(),
            "message": f"Historical search job '{job_id}' dispatched in background",
        }
    else:
        # Replay-as-Camera: register camera and launch worker
        cam_repo = CameraRepository(db)
        cam = cam_repo.create_or_update(
            camera_id=camera_id,
            name=camera_name or f"Historical Replay ({camera_id})",
            source=rel_path,
            source_type="file",
            location=location or "Recorded Archive",
            enabled=True,
            sample_fps=sample_fps,
            status="RUNNING",
        )

        worker_mgr = get_worker_manager()
        worker_mgr.add_camera_worker(
            camera_id=camera_id,
            source=rel_path,
            source_type="file",
            name=cam.name,
            sample_fps=sample_fps,
            loop_video=True,
        )
        worker_mgr.start_worker(camera_id)

        return {
            "mode": "replay",
            "camera": cam.to_dict(),
            "job": job.to_dict(),
            "message": f"Camera '{camera_id}' registered and replay started successfully",
        }


@router.get("/jobs", response_model=List[Dict[str, Any]])
def list_historical_jobs(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """List historical video processing jobs. Requires authentication."""
    repo = HistoricalJobRepository(db)
    jobs = repo.get_all(limit=limit, offset=offset)
    return [j.to_dict() for j in jobs]


@router.get("/jobs/{job_id}", response_model=Dict[str, Any])
def get_historical_job(
    job_id: str,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Get status and progress telemetry for a specific historical job."""
    repo = HistoricalJobRepository(db)
    job = repo.get_by_job_id(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job '{job_id}' not found")
    return job.to_dict()


@router.post("/jobs/{job_id}/cancel", response_model=Dict[str, Any])
def cancel_historical_job(
    job_id: str,
    historical_svc: HistoricalService = Depends(get_historical_service),
    current_user=require_role("ADMIN"),
):
    """Cancel a running or queued historical analysis job (ADMIN only)."""
    job = historical_svc.cancel_job(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Historical job '{job_id}' not found")
    return {"message": f"Historical job '{job_id}' cancelled", "job": job}

