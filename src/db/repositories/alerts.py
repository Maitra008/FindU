"""
Alert and Track database repository operations.
Provides thread-safe persistence, filtering, retrieval, and status updates.
"""

import json
import logging
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import desc
from sqlalchemy.exc import IntegrityError

from src.db.models import AlertRecord, TrackRecord

logger = logging.getLogger(__name__)


class AlertRepository:
    """Repository handling CRUD operations for alerts."""

    def __init__(self, session: Session):
        self.session = session

    def create_alert(
        self,
        alert_id: str,
        camera_id: str,
        track_id: str,
        person_id: str,
        person_name: str,
        similarity: float,
        max_similarity: float,
        mean_similarity: float,
        threshold: float,
        timestamp: float,
        frame_idx: int,
        bbox: List[int],
        snapshot_path: Optional[str] = None,
        status: str = "NEW",
    ) -> AlertRecord:
        """
        Create and persist a new AlertRecord in the database.
        """
        bbox_str = json.dumps(bbox) if isinstance(bbox, list) else str(bbox)
        
        # Check for existing alert for this (camera_id, track_id) to ensure idempotency
        existing = (
            self.session.query(AlertRecord)
            .filter_by(camera_id=camera_id, track_id=track_id)
            .first()
        )
        if existing is not None:
            logger.debug(
                "Alert already exists for camera %s, track %s (id=%d)",
                camera_id,
                track_id,
                existing.id,
            )
            return existing

        alert = AlertRecord(
            alert_id=alert_id,
            camera_id=camera_id,
            track_id=track_id,
            person_id=person_id,
            person_name=person_name,
            similarity=similarity,
            max_similarity=max_similarity,
            mean_similarity=mean_similarity,
            threshold=threshold,
            timestamp=timestamp,
            frame_idx=frame_idx,
            bbox=bbox_str,
            status=status,
            snapshot_path=snapshot_path,
        )
        self.session.add(alert)
        try:
            self.session.commit()
        except IntegrityError:
            self.session.rollback()
            existing = (
                self.session.query(AlertRecord)
                .filter_by(camera_id=camera_id, track_id=track_id)
                .first()
            )
            if existing is not None:
                logger.debug(
                    "Concurrent duplicate alert resolved to existing alert %s",
                    existing.alert_id,
                )
                return existing
            raise

        self.session.refresh(alert)
        logger.info("Persisted alert %s (id=%d) for track %s on camera %s", alert.alert_id, alert.id, track_id, camera_id)
        return alert

    def get_by_id(self, alert_id_or_pk: Any) -> Optional[AlertRecord]:
        """Fetch alert by integer primary key or string alert_id."""
        if isinstance(alert_id_or_pk, int) or (isinstance(alert_id_or_pk, str) and alert_id_or_pk.isdigit()):
            pk = int(alert_id_or_pk)
            return self.session.query(AlertRecord).filter(AlertRecord.id == pk).first()
        return self.session.query(AlertRecord).filter(AlertRecord.alert_id == str(alert_id_or_pk)).first()

    def get_alerts(
        self,
        camera_id: Optional[str] = None,
        person_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[AlertRecord]:
        """Query alerts with optional filtering."""
        query = self.session.query(AlertRecord)
        if camera_id:
            query = query.filter(AlertRecord.camera_id == camera_id)
        if person_id:
            query = query.filter(AlertRecord.person_id == person_id)
        if status:
            query = query.filter(AlertRecord.status == status.upper())

        return query.order_by(desc(AlertRecord.id)).offset(offset).limit(limit).all()

    def update_status(self, alert_id_or_pk: Any, status: str) -> Optional[AlertRecord]:
        """Update verification status (NEW, VERIFIED, DISMISSED)."""
        alert = self.get_by_id(alert_id_or_pk)
        if alert is None:
            return None

        alert.status = status.upper()
        self.session.commit()
        self.session.refresh(alert)
        logger.info("Updated alert %s status to '%s'", alert.alert_id, alert.status)
        return alert


class TrackRepository:
    """Repository handling CRUD operations for tracks."""

    def __init__(self, session: Session):
        self.session = session

    def upsert_track(
        self,
        camera_id: str,
        track_id: str,
        person_id: Optional[str],
        person_name: str,
        first_seen: float,
        last_seen: float,
        first_frame: int,
        last_frame: int,
        detection_count: int,
        max_similarity: float,
        mean_similarity: float,
        top_k_similarity: float,
        threshold_matches: int,
        alert_triggered: bool,
        status: str = "ACTIVE",
    ) -> TrackRecord:
        """Create or update track state in database."""
        track = (
            self.session.query(TrackRecord)
            .filter_by(camera_id=camera_id, track_id=track_id)
            .first()
        )
        if track is None:
            track = TrackRecord(
                camera_id=camera_id,
                track_id=track_id,
                person_id=person_id,
                person_name=person_name,
                first_seen=first_seen,
                last_seen=last_seen,
                first_frame=first_frame,
                last_frame=last_frame,
                detection_count=detection_count,
                max_similarity=max_similarity,
                mean_similarity=mean_similarity,
                top_k_similarity=top_k_similarity,
                threshold_matches=threshold_matches,
                alert_triggered=alert_triggered,
                status=status,
            )
            self.session.add(track)
        else:
            track.person_id = person_id
            track.person_name = person_name
            track.last_seen = last_seen
            track.last_frame = last_frame
            track.detection_count = detection_count
            track.max_similarity = max_similarity
            track.mean_similarity = mean_similarity
            track.top_k_similarity = top_k_similarity
            track.threshold_matches = threshold_matches
            track.alert_triggered = alert_triggered
            track.status = status

        self.session.commit()
        self.session.refresh(track)
        return track

    def get_tracks(
        self,
        camera_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[TrackRecord]:
        """Fetch track records with optional filters."""
        query = self.session.query(TrackRecord)
        if camera_id:
            query = query.filter(TrackRecord.camera_id == camera_id)
        if status:
            query = query.filter(TrackRecord.status == status.upper())
        return query.order_by(desc(TrackRecord.id)).limit(limit).all()

