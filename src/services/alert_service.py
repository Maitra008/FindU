"""
Central Alert Service.
Validates recognition alerts, persists them to PostgreSQL/Database,
and broadcasts verified alert events to connected WebSocket clients.
"""

import logging
from typing import Any, Dict, List, Optional
import asyncio
from sqlalchemy.orm import Session

from src.config import DEFAULT_SIMILARITY_THRESHOLD
from src.api.websocket import WebSocketConnectionManager, ws_manager
from src.db.database import get_session_factory
from src.db.models import AlertRecord, TrackRecord
from src.db.repositories.alerts import AlertRepository, TrackRepository
from src.tracker import TrackAlert

logger = logging.getLogger(__name__)


class AlertService:
    """
    Central service orchestrating alert validation, database persistence,
    and real-time WebSocket dispatch.
    """

    def __init__(
        self,
        session_factory=None,
        websocket_manager: Optional[WebSocketConnectionManager] = None,
    ):
        self._session_factory = session_factory
        self.ws_manager = websocket_manager or ws_manager

    @property
    def session_factory(self):
        return self._session_factory or get_session_factory()

    def create_alert_from_track_alert(
        self,
        camera_id: str,
        track_alert: TrackAlert,
        snapshot_path: Optional[str] = None,
    ) -> AlertRecord:
        """
        Ingest a Part 3 TrackAlert from a CameraWorker.

        Args:
            camera_id: Identifier of the camera source (e.g. 'C1').
            track_alert: Part 3 TrackAlert instance.
            snapshot_path: Optional path to cropped face / frame snapshot image.

        Returns:
            Persisted AlertRecord.
        """
        alert_id = f"ALERT-{camera_id}-{track_alert.track_id}-{track_alert.frame_idx}"
        return self.create_alert(
            camera_id=camera_id,
            alert_id=alert_id,
            track_id=track_alert.track_id,
            person_id=track_alert.person_id,
            person_name=track_alert.person_name,
            similarity=track_alert.similarity,
            max_similarity=track_alert.max_similarity,
            mean_similarity=track_alert.mean_similarity,
            threshold=track_alert.threshold,
            timestamp=track_alert.timestamp_sec,
            frame_idx=track_alert.frame_idx,
            bbox=track_alert.bbox,
            snapshot_path=snapshot_path,
        )

    def create_alert(
        self,
        camera_id: str,
        track_id: str,
        person_id: str,
        person_name: str,
        similarity: float,
        timestamp: float,
        bbox: List[int],
        alert_id: Optional[str] = None,
        max_similarity: Optional[float] = None,
        mean_similarity: Optional[float] = None,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        frame_idx: int = 0,
        snapshot_path: Optional[str] = None,
        status: str = "NEW",
    ) -> AlertRecord:
        """
        Validate, persist in database, and broadcast alert event over WebSocket.

        1. Validate required fields.
        2. Persist alert in database and commit.
        3. Convert stored alert into serializable payload.
        4. Broadcast to WebSocket clients.
        5. Return created AlertRecord.
        """
        # 1. Validation
        if not camera_id or not str(camera_id).strip():
            raise ValueError("camera_id is required and cannot be empty")
        if not track_id or not str(track_id).strip():
            raise ValueError("track_id is required and cannot be empty")
        if not person_id or not str(person_id).strip():
            raise ValueError("person_id is required and cannot be empty")
        if similarity is None or similarity < 0.0:
            raise ValueError(f"Invalid similarity score: {similarity}")
        if not bbox or len(bbox) != 4:
            raise ValueError(f"Invalid bounding box: {bbox}")

        effective_alert_id = alert_id or f"ALERT-{camera_id}-{track_id}-{frame_idx}"
        effective_max_sim = max_similarity if max_similarity is not None else similarity
        effective_mean_sim = mean_similarity if mean_similarity is not None else similarity

        # 2. Database Persistence
        with self.session_factory() as session:
            repo = AlertRepository(session)
            alert = repo.create_alert(
                alert_id=effective_alert_id,
                camera_id=camera_id,
                track_id=track_id,
                person_id=person_id,
                person_name=person_name,
                similarity=similarity,
                max_similarity=effective_max_sim,
                mean_similarity=effective_mean_sim,
                threshold=threshold,
                timestamp=timestamp,
                frame_idx=frame_idx,
                bbox=bbox,
                snapshot_path=snapshot_path,
                status=status,
            )
            alert_dict = alert.to_dict()

        # 3. WebSocket Broadcast (AFTER successful database commit)
        ws_payload = {
            "event": "alert.created",
            "alert": alert_dict,
        }
        self.ws_manager.broadcast_sync(ws_payload)
        logger.info("Queued alert %s for WebSocket clients.", effective_alert_id)

        return alert

    def update_track_state(
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
        """Persist or update track state in database."""
        with self.session_factory() as session:
            track_repo = TrackRepository(session)
            track = track_repo.upsert_track(
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
            return track

    def get_alerts(
        self,
        camera_id: Optional[str] = None,
        person_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Retrieve alerts from database."""
        with self.session_factory() as session:
            repo = AlertRepository(session)
            alerts = repo.get_alerts(
                camera_id=camera_id,
                person_id=person_id,
                status=status,
                limit=limit,
                offset=offset,
            )
            return [a.to_dict() for a in alerts]

    def get_alert_by_id(self, alert_id_or_pk: Any) -> Optional[Dict[str, Any]]:
        """Retrieve specific alert by ID."""
        with self.session_factory() as session:
            repo = AlertRepository(session)
            alert = repo.get_by_id(alert_id_or_pk)
            return alert.to_dict() if alert else None

    def update_alert_status(self, alert_id_or_pk: Any, status: str) -> Optional[Dict[str, Any]]:
        """Update alert status (VERIFIED, DISMISSED, NEW) and broadcast status update."""
        with self.session_factory() as session:
            repo = AlertRepository(session)
            alert = repo.update_status(alert_id_or_pk, status)
            if not alert:
                return None
            alert_dict = alert.to_dict()

        # Broadcast update to clients
        ws_payload = {
            "event": "alert.updated",
            "alert": alert_dict,
        }
        self.ws_manager.broadcast_sync(ws_payload)
        return alert_dict


_alert_service_instance: Optional[AlertService] = None


def get_alert_service() -> AlertService:
    """Get singleton AlertService instance."""
    global _alert_service_instance
    if _alert_service_instance is None:
        _alert_service_instance = AlertService()
    return _alert_service_instance

