"""
Admin & Environment Management Service.
Handles controlled test environment resets, ensuring safety across:
- In-memory workers & historical background threads
- Database tables (Persons, Alerts, Tracks, Historical Jobs)
- Filesystem storage (reference photos, dynamic embeddings, historical uploads)
- Canonical FAISS index synchronization
- Tamper-evident audit logging
"""

import logging
import os
import shutil
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from sqlalchemy.orm import Session

from src.api.websocket import ws_manager
from src.config import EMBEDDINGS_DIR, INDEX_PATH, METADATA_PATH, PROJECT_ROOT
from src.db.database import get_session_factory
from src.db.models import AlertRecord, AuditLog, Camera, HistoricalJob, Person, TrackRecord, User
from src.index import get_face_index
from src.services.audit_service import get_audit_service
from src.services.historical_service import get_historical_service
from src.workers.worker_manager import get_worker_manager

logger = logging.getLogger(__name__)

REFERENCE_PHOTOS_DIR = PROJECT_ROOT / "data" / "reference_photos"
HISTORICAL_UPLOADS_DIR = PROJECT_ROOT / "data" / "historical_uploads"

# Explicitly preserved baseline/static identities required for evaluation and demo functionality
BASELINE_EMBEDDING_FILES: Set[str] = {
    "person_01_embedding.npz",
    "person_02_embedding.npz",
    "person_03_embedding.npz",
    "person_04_embedding.npz",
    "person_05_embedding.npz",
    "person_w_embedding.npz",
    "person_x_embedding.npz",
    "person_y_embedding.npz",
    "person_z_embedding.npz",
    "partho_embedding.npz",
}


class AdminService:
    """
    Coordinates safe, atomic administrative resets of the test environment.
    """

    def __init__(self, session_factory=None):
        self.session_factory = session_factory or get_session_factory()
        self._reset_lock = threading.Lock()

    def reset_test_environment(
        self,
        actor_username: str,
        session: Optional[Session] = None,
    ) -> Dict[str, Any]:
        """
        Safely and idempotently reset test data:
        1. Acquire application-level reset lock.
        2. Cancel active historical jobs.
        3. Pause/stop active CameraWorkers.
        4. Clear test database records (Persons, Alerts, Tracks, Historical Jobs).
        5. Clear test filesystem artifacts (reference photos, dynamic embeddings, historical uploads).
        6. Synchronize canonical FAISS FaceIndex & persist to disk.
        7. Restore CameraWorkers with the clean index.
        8. Record audit log for reset operation.
        9. Broadcast WebSocket notification.
        """
        if not self._reset_lock.acquire(blocking=False):
            raise RuntimeError("Another environment reset operation is already in progress.")

        try:
            logger.info("Starting controlled test environment reset initiated by '%s'", actor_username)

            # 1. Stop/cancel active historical jobs
            historical_svc = get_historical_service()
            active_job_ids = list(historical_svc._active_jobs.keys())
            for jid in active_job_ids:
                event = historical_svc._active_jobs.get(jid)
                if event:
                    event.set()
            historical_svc._active_jobs.clear()
            logger.info("Cancelled %d active historical background jobs.", len(active_job_ids))

            # 2. Stop camera workers
            worker_mgr = get_worker_manager()
            worker_mgr.stop_all(timeout=3.0)
            for w in worker_mgr.workers.values():
                w.tracker = None
                w.metrics.faces_detected = 0
                w.metrics.tracks_created = 0
                w.metrics.alerts_emitted = 0

            # 3. Database cleanup in controlled transaction
            def _execute_db_reset(s: Session) -> Dict[str, int]:
                count_alerts = s.query(AlertRecord).delete()
                count_tracks = s.query(TrackRecord).delete()
                count_persons = s.query(Person).delete()
                count_jobs = s.query(HistoricalJob).delete()
                
                # Verify infrastructure counts
                count_cameras = s.query(Camera).count()
                count_users = s.query(User).count()

                return {
                    "alerts": count_alerts,
                    "tracks": count_tracks,
                    "persons": count_persons,
                    "historical_jobs": count_jobs,
                    "cameras_preserved": count_cameras,
                    "users_preserved": count_users,
                }

            if session is not None:
                db_counts = _execute_db_reset(session)
                session.commit()
            else:
                with self.session_factory() as s:
                    db_counts = _execute_db_reset(s)
                    s.commit()

            logger.info("Deleted from DB: %s", db_counts)

            def _safe_remove(p: Path) -> bool:
                try:
                    if p.is_dir():
                        shutil.rmtree(p, ignore_errors=True)
                        return True
                    elif p.is_file():
                        p.unlink(missing_ok=True)
                        return True
                except Exception as exc:
                    logger.warning("Notice: could not delete '%s' during reset: %s", p, exc)
                    return False
                return False

            # 4. Filesystem cleanup
            count_ref_photos = 0
            if REFERENCE_PHOTOS_DIR.exists():
                for p in list(REFERENCE_PHOTOS_DIR.glob("*")):
                    if _safe_remove(p):
                        count_ref_photos += 1
            REFERENCE_PHOTOS_DIR.mkdir(parents=True, exist_ok=True)

            count_hist_files = 0
            if HISTORICAL_UPLOADS_DIR.exists():
                for p in list(HISTORICAL_UPLOADS_DIR.glob("*")):
                    if _safe_remove(p):
                        count_hist_files += 1
            HISTORICAL_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

            count_embeddings_deleted = 0
            if EMBEDDINGS_DIR.exists():
                for p in list(EMBEDDINGS_DIR.glob("*.npz")):
                    if p.name not in BASELINE_EMBEDDING_FILES:
                        if _safe_remove(p):
                            count_embeddings_deleted += 1
            EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

            # 5. Canonical FAISS FaceIndex synchronization & persistence
            face_index = get_face_index()
            face_index.build_from_directory(EMBEDDINGS_DIR)
            face_index.save(INDEX_PATH, METADATA_PATH)
            logger.info("Rebuilt and persisted canonical FaceIndex with %d baseline identities.", face_index.total_identities)

            # 6. Propagate clean index and restart enabled camera workers
            worker_mgr.index = face_index
            for worker in worker_mgr.workers.values():
                worker.index = face_index

            worker_mgr.load_cameras_from_db(enabled_only=True, auto_start=True)

            # 7. Record tamper-evident audit log
            audit = get_audit_service()
            audit.log(
                event="admin.reset_test_environment",
                actor_username=actor_username,
                resource_type="system",
                resource_id="test_environment",
                detail={
                    "deleted_counts": {
                        "persons": db_counts["persons"],
                        "alerts": db_counts["alerts"],
                        "tracks": db_counts["tracks"],
                        "historical_jobs": db_counts["historical_jobs"],
                        "historical_files": count_hist_files,
                        "reference_photos": count_ref_photos,
                        "embeddings": count_embeddings_deleted,
                    },
                    "preserved_counts": {
                        "cameras": db_counts["cameras_preserved"],
                        "users": db_counts["users_preserved"],
                        "baseline_identities": face_index.total_identities,
                    },
                },
                session=session,
            )
            if session is not None:
                session.commit()

            # 8. Broadcast WebSocket notification
            ws_manager.broadcast_sync({
                "event": "system.reset",
                "message": "Test environment has been reset by administrator.",
                "actor": actor_username,
            })

            result = {
                "success": True,
                "message": "Test environment reset successfully.",
                "deleted": {
                    "persons": db_counts["persons"],
                    "alerts": db_counts["alerts"],
                    "tracks": db_counts["tracks"],
                    "historical_jobs": db_counts["historical_jobs"],
                    "historical_files": count_hist_files,
                    "reference_photos": count_ref_photos,
                    "embeddings": count_embeddings_deleted,
                },
                "preserved": {
                    "cameras": db_counts["cameras_preserved"],
                    "users": db_counts["users_preserved"],
                    "baseline_identities": face_index.total_identities,
                },
            }
            logger.info("Test environment reset completed successfully: %s", result)
            return result

        finally:
            self._reset_lock.release()


_admin_service_instance: Optional[AdminService] = None


def get_admin_service() -> AdminService:
    """Singleton getter for AdminService."""
    global _admin_service_instance
    if _admin_service_instance is None:
        _admin_service_instance = AdminService()
    return _admin_service_instance
