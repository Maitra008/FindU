"""
Camera database repository operations.
Provides CRUD operations for configured cameras and video feeds.
"""

import logging
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session

from src.db.models import Camera, get_utc_now

logger = logging.getLogger(__name__)


class CameraRepository:
    """Repository handling CRUD operations for Camera feeds."""

    def __init__(self, session: Session):
        self.session = session

    def get_all(self, enabled_only: bool = False) -> List[Camera]:
        """Fetch all cameras or only enabled cameras."""
        query = self.session.query(Camera)
        if enabled_only:
            query = query.filter(Camera.enabled == True)
        return query.order_by(Camera.camera_id).all()

    def get_by_camera_id(self, camera_id: str) -> Optional[Camera]:
        """Fetch camera record by camera_id (e.g. 'C1')."""
        return self.session.query(Camera).filter(Camera.camera_id == camera_id).first()

    def create_or_update(
        self,
        camera_id: str,
        name: str,
        source: str,
        source_type: str = "file",
        location: Optional[str] = None,
        enabled: bool = True,
        sample_fps: float = 2.0,
        status: str = "OFFLINE",
        stream_fps: Optional[float] = 30.0,
    ) -> Camera:
        """Create or update a camera configuration."""
        cam = self.get_by_camera_id(camera_id)
        if cam is None:
            cam = Camera(
                camera_id=camera_id,
                name=name,
                source=source,
                source_type=source_type,
                location=location,
                enabled=enabled,
                sample_fps=sample_fps,
                status=status,
                stream_fps=stream_fps,
            )
            self.session.add(cam)
        else:
            cam.name = name
            cam.source = source
            cam.source_type = source_type
            cam.location = location
            cam.enabled = enabled
            cam.sample_fps = sample_fps
            if status:
                cam.status = status
            if stream_fps:
                cam.stream_fps = stream_fps

        self.session.commit()
        self.session.refresh(cam)
        logger.info("Saved camera configuration for '%s' (%s)", camera_id, name)
        return cam

    def update(self, camera_id: str, **kwargs: Any) -> Optional[Camera]:
        """Update specific fields of an existing camera."""
        cam = self.get_by_camera_id(camera_id)
        if not cam:
            return None

        allowed = {
            "name",
            "source",
            "source_type",
            "location",
            "enabled",
            "sample_fps",
            "status",
            "stream_fps",
            "ai_fps",
            "latency_ms",
            "reconnect_count",
            "error_message",
        }

        for k, v in kwargs.items():
            if k in allowed and hasattr(cam, k):
                setattr(cam, k, v)

        cam.updated_at = get_utc_now()
        self.session.commit()
        self.session.refresh(cam)
        return cam

    def update_metrics(
        self,
        camera_id: str,
        status: Optional[str] = None,
        stream_fps: Optional[float] = None,
        ai_fps: Optional[float] = None,
        latency_ms: Optional[float] = None,
        reconnect_count: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> Optional[Camera]:
        """Update runtime telemetry for a camera."""
        cam = self.get_by_camera_id(camera_id)
        if not cam:
            return None

        if status is not None:
            cam.status = status
        if stream_fps is not None:
            cam.stream_fps = stream_fps
        if ai_fps is not None:
            cam.ai_fps = ai_fps
        if latency_ms is not None:
            cam.latency_ms = latency_ms
        if reconnect_count is not None:
            cam.reconnect_count = reconnect_count
        if error_message is not None:
            cam.error_message = error_message

        cam.last_frame_at = get_utc_now()
        self.session.commit()
        self.session.refresh(cam)
        return cam

    def set_enabled(self, camera_id: str, enabled: bool) -> Optional[Camera]:
        """Enable or disable a camera stream."""
        cam = self.get_by_camera_id(camera_id)
        if cam is None:
            return None
        cam.enabled = enabled
        self.session.commit()
        self.session.refresh(cam)
        return cam

    def delete(self, camera_id: str) -> bool:
        """Delete a camera configuration."""
        cam = self.get_by_camera_id(camera_id)
        if not cam:
            return False
        self.session.delete(cam)
        self.session.commit()
        return True
