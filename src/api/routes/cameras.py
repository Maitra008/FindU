"""
Camera management REST API routes.
Part 6 & Camera Command Center: JWT authentication, RBAC, CRUD, and Live Connection Testing.
"""

import logging
import time
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.auth.dependencies import get_current_user, require_role
from src.cameras.config import CameraSourceConfig, RTSPTransport, SourceType
from src.cameras.sources import create_camera_source
from src.db.database import get_db
from src.db.repositories.cameras import CameraRepository
from src.services.audit_service import get_audit_service
from src.workers.worker_manager import get_worker_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/cameras", tags=["Cameras"])


class CameraCreateSchema(BaseModel):
    camera_id: str = Field(..., min_length=1, max_length=32)
    name: str = Field(..., min_length=1, max_length=128)
    source: str = Field(..., min_length=1)
    source_type: str = Field("file", description="Source type: 'usb', 'rtsp', or 'file'")
    location: Optional[str] = None
    enabled: bool = True
    sample_fps: float = Field(2.0, ge=0.1, le=30.0)
    stream_fps: Optional[float] = Field(None, ge=1.0, le=120.0)
    transport: str = Field("tcp", description="RTSP transport: 'tcp' or 'udp'")


class CameraUpdateSchema(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=128)
    source: Optional[str] = Field(None, min_length=1)
    source_type: Optional[str] = Field(None, description="Source type: 'usb', 'rtsp', or 'file'")
    location: Optional[str] = None
    enabled: Optional[bool] = None
    sample_fps: Optional[float] = Field(None, ge=0.1, le=30.0)
    stream_fps: Optional[float] = Field(None, ge=1.0, le=120.0)
    transport: Optional[str] = None


class TestSourceSchema(BaseModel):
    source: str
    source_type: str = "file"
    transport: str = "tcp"


@router.get("", response_model=List[Dict[str, Any]])
def list_cameras(
    enabled_only: bool = False,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """List all registered cameras with live runtime status and metrics."""
    repo = CameraRepository(db)
    cams = repo.get_all(enabled_only=enabled_only)
    
    # Enrich with live worker metrics if running
    worker_mgr = get_worker_manager()
    worker_statuses = worker_mgr.get_all_statuses()
    worker_map = {w["camera_id"]: w for w in worker_statuses}

    result = []
    for c in cams:
        cd = c.to_dict()
        if c.camera_id in worker_map:
            wm = worker_map[c.camera_id]
            cd["status"] = wm.get("status", cd.get("status", "OFFLINE"))
            cd["fps"] = wm.get("fps", cd.get("stream_fps", 0.0))
            cd["frames_processed"] = wm.get("frames_processed", 0)
            cd["alerts_emitted"] = wm.get("alerts_emitted", 0)
            cd["tracks_created"] = wm.get("tracks_created", 0)
        result.append(cd)

    return result


@router.get("/{camera_id}", response_model=Dict[str, Any])
def get_camera(
    camera_id: str,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Get single camera configuration by camera_id (e.g. 'C1'). Requires authentication."""
    repo = CameraRepository(db)
    cam = repo.get_by_camera_id(camera_id)
    if not cam:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera '{camera_id}' not found",
        )
    cd = cam.to_dict()

    worker_mgr = get_worker_manager()
    wm = worker_mgr.get_worker_status(camera_id)
    if wm:
        cd["status"] = wm.get("status", cd.get("status", "OFFLINE"))
        cd["fps"] = wm.get("fps", cd.get("stream_fps", 0.0))
        cd["frames_processed"] = wm.get("frames_processed", 0)
        cd["alerts_emitted"] = wm.get("alerts_emitted", 0)
        cd["tracks_created"] = wm.get("tracks_created", 0)

    return cd


@router.post("", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
def create_camera(
    payload: CameraCreateSchema,
    db: Session = Depends(get_db),
    current_user=require_role("ADMIN"),
):
    """Create a new camera configuration. ADMIN role required."""
    repo = CameraRepository(db)
    existing = repo.get_by_camera_id(payload.camera_id)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Camera with ID '{payload.camera_id}' already exists",
        )

    cam = repo.create_or_update(
        camera_id=payload.camera_id,
        name=payload.name,
        source=payload.source,
        source_type=payload.source_type,
        location=payload.location,
        enabled=payload.enabled,
        sample_fps=payload.sample_fps,
        status="OFFLINE",
    )

    # Register with WorkerManager if enabled
    if payload.enabled:
        worker_mgr = get_worker_manager()
        worker_mgr.add_camera_worker(
            camera_id=payload.camera_id,
            source=payload.source,
            source_type=payload.source_type,
            name=payload.name,
            sample_fps=payload.sample_fps,
        )

    # Audit log
    get_audit_service().log(
        event="camera.created",
        actor_username=current_user.username,
        resource_type="camera",
        resource_id=payload.camera_id,
        detail={"name": payload.name, "source_type": payload.source_type},
    )

    return cam.to_dict()


@router.patch("/{camera_id}", response_model=Dict[str, Any])
def update_camera(
    camera_id: str,
    payload: CameraUpdateSchema,
    db: Session = Depends(get_db),
    current_user=require_role("ADMIN"),
):
    """Update camera configuration. ADMIN role required."""
    repo = CameraRepository(db)
    cam = repo.get_by_camera_id(camera_id)
    if not cam:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera '{camera_id}' not found",
        )

    updates = {k: v for k, v in payload.model_dump().items() if v is not None}
    updated = repo.update(camera_id, **updates)

    # Update worker state if enabled/source changed
    worker_mgr = get_worker_manager()
    if payload.enabled is False:
        worker_mgr.stop_worker(camera_id)
    elif payload.enabled is True or any(k in updates for k in ["source", "source_type", "sample_fps"]):
        worker_mgr.stop_worker(camera_id)
        if updated.enabled:
            worker_mgr.add_camera_worker(
                camera_id=updated.camera_id,
                source=updated.source,
                source_type=updated.source_type if hasattr(updated, "source_type") and updated.source_type else "file",
                name=updated.name,
                sample_fps=updated.sample_fps,
            )

    get_audit_service().log(
        event="camera.updated",
        actor_username=current_user.username,
        resource_type="camera",
        resource_id=camera_id,
        detail={"updated_fields": list(updates.keys())},
    )

    return updated.to_dict()


@router.delete("/{camera_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_camera(
    camera_id: str,
    db: Session = Depends(get_db),
    current_user=require_role("ADMIN"),
):
    """Delete camera configuration and stop associated worker. ADMIN role required."""
    repo = CameraRepository(db)
    cam = repo.get_by_camera_id(camera_id)
    if not cam:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera '{camera_id}' not found",
        )

    worker_mgr = get_worker_manager()
    worker_mgr.stop_worker(camera_id)

    repo.delete(camera_id)

    get_audit_service().log(
        event="camera.deleted",
        actor_username=current_user.username,
        resource_type="camera",
        resource_id=camera_id,
        detail={"name": cam.name},
    )
    return None


@router.post("/test-source", response_model=Dict[str, Any])
def test_camera_source(
    payload: TestSourceSchema,
    current_user=Depends(get_current_user),
):
    """
    Test camera source connection without saving to registry.
    Available to ADMIN and POLICE roles.
    """
    st_str = payload.source_type.lower()
    st = st_str if st_str in ("usb", "rtsp", "file") else "file"

    tp_str = payload.transport.lower()
    transport = tp_str if tp_str in ("tcp", "udp") else "tcp"

    cfg = CameraSourceConfig(
        source=payload.source,
        source_type=st,
        transport=transport,
        timeout_sec=5.0,
    )

    start_time = time.monotonic()
    source = create_camera_source(cfg)
    try:
        success = source.open()
        latency_ms = round((time.monotonic() - start_time) * 1000.0, 1)

        if not success:
            return {
                "success": False,
                "latency_ms": latency_ms,
                "error": source.last_error or "Failed to connect to video stream",
                "masked_source": cfg.masked_source,
            }

        # Try to read 1 frame
        read_success, frame, _ = source.read_frame()
        meta = source.get_metadata()

        return {
            "success": read_success,
            "latency_ms": latency_ms,
            "fps": meta.get("fps", 0.0),
            "width": meta.get("width", 0),
            "height": meta.get("height", 0),
            "error": None if read_success else "Stream opened but failed to capture frame",
            "masked_source": cfg.masked_source,
        }
    finally:
        source.close()


@router.post("/{camera_id}/test", response_model=Dict[str, Any])
def test_existing_camera(
    camera_id: str,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Test connection for an existing registered camera."""
    repo = CameraRepository(db)
    cam = repo.get_by_camera_id(camera_id)
    if not cam:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera '{camera_id}' not found",
        )

    return test_camera_source(
        TestSourceSchema(
            source=cam.source,
            source_type=getattr(cam, "source_type", "file") or "file",
            transport="tcp",
        ),
        current_user=current_user,
    )
