"""
Camera management REST API routes.
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.db.database import get_db
from src.db.repositories.cameras import CameraRepository

router = APIRouter(prefix="/api/cameras", tags=["Cameras"])


class CameraCreateSchema(BaseModel):
    camera_id: str
    name: str
    source: str
    location: Optional[str] = None
    enabled: bool = True
    sample_fps: float = 2.0


class CameraUpdateSchema(BaseModel):
    name: Optional[str] = None
    source: Optional[str] = None
    location: Optional[str] = None
    enabled: Optional[bool] = None
    sample_fps: Optional[float] = None


@router.get("", response_model=List[Dict[str, Any]])
def list_cameras(enabled_only: bool = False, db: Session = Depends(get_db)):
    """List all registered cameras."""
    repo = CameraRepository(db)
    cams = repo.get_all(enabled_only=enabled_only)
    return [c.to_dict() for c in cams]


@router.get("/{camera_id}", response_model=Dict[str, Any])
def get_camera(camera_id: str, db: Session = Depends(get_db)):
    """Get single camera configuration by camera_id (e.g. 'C1')."""
    repo = CameraRepository(db)
    cam = repo.get_by_camera_id(camera_id)
    if not cam:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Camera '{camera_id}' not found")
    return cam.to_dict()


@router.post("", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
def create_or_update_camera(payload: CameraCreateSchema, db: Session = Depends(get_db)):
    """Create or update camera configuration."""
    repo = CameraRepository(db)
    cam = repo.create_or_update(
        camera_id=payload.camera_id,
        name=payload.name,
        source=payload.source,
        location=payload.location,
        enabled=payload.enabled,
        sample_fps=payload.sample_fps,
    )
    return cam.to_dict()

