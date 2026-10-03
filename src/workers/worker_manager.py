"""
Worker Manager.
Orchestrates lifecycle (start, stop, monitor) for multiple concurrent CameraWorkers (C1 to C4).
Provides dynamic FAISS face index updates upon person registration.
"""

import logging
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session

from src.config import (
    DEFAULT_SAMPLE_FPS,
    DEFAULT_SIMILARITY_THRESHOLD,
    EMBEDDINGS_DIR,
    INDEX_PATH,
    METADATA_PATH,
)
from src.db.database import get_session_factory
from src.db.models import Camera
from src.db.repositories.cameras import CameraRepository
from src.face_engine import FaceEngine
from src.index import FaceIndex
from src.services.alert_service import AlertService, get_alert_service
from src.workers.camera_worker import CameraWorker

logger = logging.getLogger(__name__)


class WorkerManager:
    """
    Manages concurrent camera worker streams (C1..C4).
    Provides start, stop, restart, telemetry query, and dynamic face index updates.
    """

    def __init__(
        self,
        face_engine: Optional[FaceEngine] = None,
        face_index: Optional[FaceIndex] = None,
        alert_service: Optional[AlertService] = None,
        session_factory=None,
    ):
        self.engine = face_engine
        self.index = face_index
        self.alert_service = alert_service or get_alert_service()
        self.session_factory = session_factory or get_session_factory()
        self.workers: Dict[str, CameraWorker] = {}

    def _get_engine(self) -> FaceEngine:
        if self.engine is None:
            self.engine = FaceEngine()
        return self.engine

    def _get_index(self) -> FaceIndex:
        if self.index is None:
            self.index = FaceIndex()
            if INDEX_PATH.exists() and METADATA_PATH.exists():
                self.index.load(INDEX_PATH, METADATA_PATH)
            elif EMBEDDINGS_DIR.exists():
                self.index.build_from_directory(EMBEDDINGS_DIR)
        return self.index

    def register_face_identity(
        self,
        embedding: Any,
        person_id: str,
        name: str,
        npz_path: str = "",
    ) -> None:
        """
        Dynamically update shared FAISS index with newly registered identity
        and propagate to all active CameraWorkers.
        """
        index = self._get_index()
        existing = any(m.get("person_id") == person_id for m in index.metadata)
        if existing and EMBEDDINGS_DIR.exists():
            index.build_from_directory(EMBEDDINGS_DIR)
        else:
            index.add_identity(
                embedding=embedding,
                person_id=person_id,
                name=name,
                num_images=1,
                npz_file=npz_path,
            )

        # Propagate updated index reference to all camera workers
        for worker in self.workers.values():
            worker.index = index
        logger.info("Propagated updated FaceIndex (%d identities) to %d workers.", index.total_identities, len(self.workers))

    def add_camera_worker(
        self,
        camera_id: str,
        source: str,
        name: Optional[str] = None,
        sample_fps: float = DEFAULT_SAMPLE_FPS,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        loop_video: bool = False,
    ) -> CameraWorker:
        """
        Register a new camera worker or replace an existing stopped worker.
        """
        if camera_id in self.workers and self.workers[camera_id].is_running:
            logger.warning("Camera worker '%s' is already running; stopping it first.", camera_id)
            self.workers[camera_id].stop()

        worker = CameraWorker(
            camera_id=camera_id,
            source=source,
            name=name,
            sample_fps=sample_fps,
            threshold=threshold,
            face_engine=self._get_engine(),
            face_index=self._get_index(),
            alert_service=self.alert_service,
            loop_video=loop_video,
        )
        self.workers[camera_id] = worker
        return worker

    def load_cameras_from_db(self, enabled_only: bool = True) -> int:
        """
        Scan database for configured cameras (e.g. C1..C4) and initialize workers.
        """
        with self.session_factory() as session:
            repo = CameraRepository(session)
            cameras: List[Camera] = repo.get_all(enabled_only=enabled_only)
            for cam in cameras:
                if cam.camera_id not in self.workers:
                    self.add_camera_worker(
                        camera_id=cam.camera_id,
                        source=cam.source,
                        name=cam.name,
                        sample_fps=cam.sample_fps,
                    )
            logger.info("Loaded %d camera workers from database.", len(cameras))
            return len(cameras)

    def start_worker(self, camera_id: str) -> bool:
        """Start an individual camera worker thread."""
        if camera_id not in self.workers:
            logger.error("Camera '%s' not registered in WorkerManager.", camera_id)
            return False

        worker = self.workers[camera_id]
        if worker.is_running:
            logger.info("CameraWorker '%s' already running.", camera_id)
            return True

        # Re-instantiate worker thread if already finished
        if not worker.is_alive() and worker.metrics.frames_read > 0:
            worker = self.add_camera_worker(
                camera_id=worker.camera_id,
                source=worker.source,
                name=worker.camera_name,
                sample_fps=worker.sample_fps,
                threshold=worker.threshold,
                loop_video=worker.loop_video,
            )

        worker.start()
        logger.info("Started CameraWorker thread '%s'.", camera_id)
        return True

    def stop_worker(self, camera_id: str, timeout: float = 5.0) -> bool:
        """Stop an individual camera worker thread."""
        if camera_id not in self.workers:
            return False
        self.workers[camera_id].stop(timeout=timeout)
        return True

    def start_all(self) -> Dict[str, bool]:
        """Start all registered camera workers."""
        results = {}
        for cid in list(self.workers.keys()):
            results[cid] = self.start_worker(cid)
        return results

    def stop_all(self, timeout: float = 5.0) -> None:
        """Stop all running camera workers."""
        for worker in self.workers.values():
            if worker.is_running:
                worker.stop(timeout=timeout)

    def get_statuses(self) -> List[Dict[str, Any]]:
        """Return real-time telemetry metrics for all managed workers."""
        return [w.metrics.to_dict() for w in self.workers.values()]

    def get_worker_status(self, camera_id: str) -> Optional[Dict[str, Any]]:
        """Return metrics for specific camera worker."""
        worker = self.workers.get(camera_id)
        return worker.metrics.to_dict() if worker else None


_worker_manager_instance: Optional[WorkerManager] = None


def get_worker_manager() -> WorkerManager:
    """Get singleton WorkerManager instance."""
    global _worker_manager_instance
    if _worker_manager_instance is None:
        _worker_manager_instance = WorkerManager()
    return _worker_manager_instance
