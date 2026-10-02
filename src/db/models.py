"""
SQLAlchemy Database Models for Missing Person Recognition System.
Defines schemas for Person, Camera, Track, and Alert entities.
"""

from datetime import datetime, timezone
import json
from typing import Any, Dict, List, Optional
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def get_utc_now() -> datetime:
    """Return timezone-aware current UTC datetime."""
    return datetime.now(timezone.utc)


class Person(Base):
    """Registered person / missing person identity."""
    __tablename__ = "persons"

    id = Column(Integer, primary_key=True, autoincrement=True)
    person_id = Column(String(64), unique=True, nullable=False, index=True)
    name = Column(String(128), nullable=False)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=get_utc_now, nullable=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "person_id": self.person_id,
            "name": self.name,
            "notes": self.notes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Camera(Base):
    """Camera / video source feed configuration."""
    __tablename__ = "cameras"

    id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(String(32), unique=True, nullable=False, index=True)
    name = Column(String(128), nullable=False)
    location = Column(String(256), nullable=True)
    source = Column(String(512), nullable=False)  # file path, camera index (e.g. '0'), or rtsp uri
    enabled = Column(Boolean, default=True, nullable=False)
    sample_fps = Column(Float, default=2.0, nullable=False)
    created_at = Column(DateTime, default=get_utc_now, nullable=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "camera_id": self.camera_id,
            "name": self.name,
            "location": self.location,
            "source": self.source,
            "enabled": self.enabled,
            "sample_fps": self.sample_fps,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class TrackRecord(Base):
    """Historical and active track state associated with a specific camera."""
    __tablename__ = "tracks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    track_id = Column(String(64), nullable=False, index=True)
    camera_id = Column(String(32), nullable=False, index=True)
    person_id = Column(String(64), nullable=True, index=True)
    person_name = Column(String(128), default="Unknown", nullable=False)
    first_seen = Column(Float, default=0.0, nullable=False)
    last_seen = Column(Float, default=0.0, nullable=False)
    first_frame = Column(Integer, default=0, nullable=False)
    last_frame = Column(Integer, default=0, nullable=False)
    detection_count = Column(Integer, default=1, nullable=False)
    max_similarity = Column(Float, default=0.0, nullable=False)
    mean_similarity = Column(Float, default=0.0, nullable=False)
    top_k_similarity = Column(Float, default=0.0, nullable=False)
    threshold_matches = Column(Integer, default=0, nullable=False)
    alert_triggered = Column(Boolean, default=False, nullable=False)
    status = Column(String(32), default="ACTIVE", nullable=False)  # ACTIVE, TERMINATED
    created_at = Column(DateTime, default=get_utc_now, nullable=False)
    updated_at = Column(DateTime, default=get_utc_now, onupdate=get_utc_now, nullable=False)

    __table_args__ = (
        UniqueConstraint("camera_id", "track_id", name="uq_camera_track"),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "track_id": self.track_id,
            "camera_id": self.camera_id,
            "person_id": self.person_id,
            "person_name": self.person_name,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "detection_count": self.detection_count,
            "max_similarity": round(self.max_similarity, 4),
            "mean_similarity": round(self.mean_similarity, 4),
            "top_k_similarity": round(self.top_k_similarity, 4),
            "threshold_matches": self.threshold_matches,
            "alert_triggered": self.alert_triggered,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class AlertRecord(Base):
    """Verified or potential match alert dispatched for a person on a track."""
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    alert_id = Column(String(64), unique=True, nullable=False, index=True)
    camera_id = Column(String(32), nullable=False, index=True)
    track_id = Column(String(64), nullable=False, index=True)
    person_id = Column(String(64), nullable=False, index=True)
    person_name = Column(String(128), nullable=False)
    similarity = Column(Float, nullable=False)
    max_similarity = Column(Float, nullable=False)
    mean_similarity = Column(Float, nullable=False)
    threshold = Column(Float, default=0.89, nullable=False)
    timestamp = Column(Float, nullable=False)
    frame_idx = Column(Integer, default=0, nullable=False)
    bbox = Column(Text, nullable=False)  # JSON encoded [x1, y1, x2, y2]
    status = Column(String(32), default="NEW", nullable=False)  # NEW, VERIFIED, DISMISSED
    snapshot_path = Column(String(512), nullable=True)
    created_at = Column(DateTime, default=get_utc_now, nullable=False)

    __table_args__ = (
        UniqueConstraint("camera_id", "track_id", name="uq_camera_track_alert"),
    )

    def to_dict(self) -> Dict[str, Any]:
        try:
            bbox_list = json.loads(self.bbox) if isinstance(self.bbox, str) else self.bbox
        except Exception:
            bbox_list = self.bbox

        return {
            "id": self.id,
            "alert_id": self.alert_id,
            "camera_id": self.camera_id,
            "track_id": self.track_id,
            "person_id": self.person_id,
            "person_name": self.person_name,
            "similarity": round(self.similarity, 4),
            "max_similarity": round(self.max_similarity, 4),
            "mean_similarity": round(self.mean_similarity, 4),
            "threshold": round(self.threshold, 4),
            "timestamp": round(self.timestamp, 2),
            "frame_idx": self.frame_idx,
            "bbox": bbox_list,
            "status": self.status,
            "snapshot_path": self.snapshot_path,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

