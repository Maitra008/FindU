"""
Camera database repository operations.
Provides CRUD operations for configured cameras and video feeds.
"""

import logging
from typing import List, Optional
from sqlalchemy.orm import Session

from src.db.models import Camera

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
        location: Optional[str] = None,
        enabled: bool = True,
        sample_fps: float = 2.0,
    ) -> Camera:
        """Create or update a camera configuration."""
        cam = self.get_by_camera_id(camera_id)
        if cam is None:
            cam = Camera(
                camera_id=camera_id,
                name=name,
                source=source,
                location=location,
                enabled=enabled,
                sample_fps=sample_fps,
            )
            self.session.add(cam)
        else:
            cam.name = name
            cam.source = source
            cam.location = location
            cam.enabled = enabled
            cam.sample_fps = sample_fps

        self.session.commit()
        self.session.refresh(cam)
        logger.info("Saved camera configuration for '%s' (%s)", camera_id, name)
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

