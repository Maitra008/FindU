"""
SQLAlchemy Database Models for Missing Person Recognition System.
Defines schemas for Person, Camera, Track, Alert, User, and AuditLog entities.
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
    """Registered person / missing person identity and case record."""
    __tablename__ = "persons"

    id = Column(Integer, primary_key=True, autoincrement=True)
    person_id = Column(String(64), unique=True, nullable=False, index=True)
    name = Column(String(128), nullable=False)
    case_id = Column(String(64), nullable=True, index=True)
    age = Column(Integer, nullable=True)
    gender = Column(String(32), nullable=True)
    date_last_seen = Column(String(64), nullable=True)
    last_known_location = Column(String(256), nullable=True)
    notes = Column(Text, nullable=True)
    status = Column(String(32), default="ACTIVE", nullable=False)  # ACTIVE, FOUND, CLOSED
    photo_paths = Column(Text, nullable=True)  # JSON list of relative photo paths
    embedding_path = Column(String(512), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=get_utc_now, nullable=False)
    updated_at = Column(DateTime, default=get_utc_now, onupdate=get_utc_now, nullable=False)

    def to_dict(self) -> Dict[str, Any]:
        photos = []
        if self.photo_paths:
            try:
                photos = json.loads(self.photo_paths) if isinstance(self.photo_paths, str) else self.photo_paths
            except Exception:
                photos = [self.photo_paths]

        return {
            "id": self.id,
            "person_id": self.person_id,
            "name": self.name,
            "case_id": self.case_id,
            "age": self.age,
            "gender": self.gender,
            "date_last_seen": self.date_last_seen,
            "last_known_location": self.last_known_location,
            "notes": self.notes,
            "status": self.status,
            "photo_paths": photos,
            "embedding_path": self.embedding_path,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class Camera(Base):
    """Camera / video source feed configuration."""
    __tablename__ = "cameras"

    id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(String(32), unique=True, nullable=False, index=True)
    name = Column(String(128), nullable=False)
    location = Column(String(256), nullable=True)
    source_type = Column(String(32), default="file", nullable=False)  # file, usb, rtsp
    source = Column(String(512), nullable=False)  # file path, camera index (e.g. '0'), or rtsp uri
    enabled = Column(Boolean, default=True, nullable=False)
    sample_fps = Column(Float, default=2.0, nullable=False)
    status = Column(String(32), default="OFFLINE", nullable=False)  # ONLINE, CONNECTING, RECONNECTING, OFFLINE, PROCESSING, ERROR
    stream_fps = Column(Float, default=30.0, nullable=True)
    ai_fps = Column(Float, default=2.0, nullable=True)
    latency_ms = Column(Float, default=0.0, nullable=True)
    reconnect_count = Column(Integer, default=0, nullable=False)
    last_connected_at = Column(DateTime, nullable=True)
    last_frame_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=get_utc_now, nullable=False)
    updated_at = Column(DateTime, default=get_utc_now, onupdate=get_utc_now, nullable=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "camera_id": self.camera_id,
            "name": self.name,
            "location": self.location,
            "source_type": self.source_type or "file",
            "source": self.source,
            "enabled": self.enabled,
            "sample_fps": self.sample_fps,
            "status": self.status or ("ONLINE" if self.enabled else "OFFLINE"),
            "stream_fps": self.stream_fps or 30.0,
            "ai_fps": self.ai_fps or self.sample_fps,
            "latency_ms": self.latency_ms or 0.0,
            "reconnect_count": self.reconnect_count or 0,
            "last_connected_at": self.last_connected_at.isoformat() if self.last_connected_at else None,
            "last_frame_at": self.last_frame_at.isoformat() if self.last_frame_at else None,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class HistoricalJob(Base):
    """Background asynchronous processing job for long/historical video files."""
    __tablename__ = "historical_jobs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(64), unique=True, nullable=False, index=True)
    camera_id = Column(String(32), nullable=False, index=True)
    file_path = Column(String(512), nullable=False)
    sample_fps = Column(Float, default=2.0, nullable=False)
    status = Column(String(32), default="QUEUED", nullable=False)  # QUEUED, PROCESSING, PAUSED, COMPLETED, FAILED, CANCELLED
    total_duration_sec = Column(Float, default=0.0, nullable=False)
    processed_duration_sec = Column(Float, default=0.0, nullable=False)
    progress_percent = Column(Float, default=0.0, nullable=False)
    frames_sampled = Column(Integer, default=0, nullable=False)
    faces_detected = Column(Integer, default=0, nullable=False)
    tracks_created = Column(Integer, default=0, nullable=False)
    potential_matches = Column(Integer, default=0, nullable=False)
    verified_matches = Column(Integer, default=0, nullable=False)
    processing_speed_fps = Column(Float, default=0.0, nullable=False)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=get_utc_now, nullable=False)
    completed_at = Column(DateTime, nullable=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "job_id": self.job_id,
            "camera_id": self.camera_id,
            "file_path": self.file_path,
            "sample_fps": self.sample_fps,
            "status": self.status,
            "total_duration_sec": round(self.total_duration_sec, 2),
            "processed_duration_sec": round(self.processed_duration_sec, 2),
            "progress_percent": round(self.progress_percent, 1),
            "frames_sampled": self.frames_sampled,
            "faces_detected": self.faces_detected,
            "tracks_created": self.tracks_created,
            "potential_matches": self.potential_matches,
            "verified_matches": self.verified_matches,
            "processing_speed_fps": round(self.processing_speed_fps, 1),
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
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


class User(Base):
    """
    Operator user account for FindU portal authentication.

    Roles:
        ADMIN    — full access including audit log and user management
        POLICE   — can view alerts and cameras, confirm/dismiss
        HOSPITAL — read-only access to alerts
        NGO      — read-only access to alerts
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(64), unique=True, nullable=False, index=True)
    password_hash = Column(String(256), nullable=False)
    role = Column(String(32), nullable=False, default="POLICE")  # ADMIN, POLICE, HOSPITAL, NGO
    enabled = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=get_utc_now, nullable=False)

    def to_dict(self) -> Dict[str, Any]:
        """Return safe serialization — NEVER includes password_hash."""
        return {
            "id": self.id,
            "username": self.username,
            "role": self.role,
            "enabled": self.enabled,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class AuditLog(Base):
    """
    Immutable audit trail for security-relevant operator actions.

    NOTE: NEVER store passwords, raw tokens, face embeddings, or raw video frames here.
    """
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event = Column(String(64), nullable=False, index=True)          # e.g. "alert.verified", "login.success"
    actor_username = Column(String(64), nullable=True, index=True)  # None for system events
    resource_type = Column(String(32), nullable=True)               # e.g. "alert", "camera", "user"
    resource_id = Column(String(128), nullable=True)                # e.g. alert_id, camera_id
    detail = Column(Text, nullable=True)                            # JSON-safe extra context (no secrets)
    created_at = Column(DateTime, default=get_utc_now, nullable=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "event": self.event,
            "actor_username": self.actor_username,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "detail": self.detail,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
