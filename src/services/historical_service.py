"""
Historical Video Processing Service.
Runs asynchronous background searches over long video files without blocking HTTP requests.
Feeds SCRFD + ArcFace + FAISS + FaceTracker + AlertService and emits WebSocket progress events.
"""

import logging
import threading
import time
from pathlib import Path
from typing import Optional
import cv2
import numpy as np

from src.api.websocket import ws_manager
from src.config import (
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_MAX_MISSED_FRAMES,
    DEFAULT_SAMPLE_FPS,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_TRACK_TOP_K,
    PROJECT_ROOT,
)
from src.db.database import get_session_factory
from src.db.repositories.historical import HistoricalJobRepository
from src.face_engine import DetectedFace, FaceEngine, get_face_engine
from src.index import FaceIndex, MatchResult, get_face_index
from src.pipeline import FrameRecognitionResult, recognize_frame
from src.services.alert_service import AlertService, get_alert_service
from src.tracker import DetectionItem, FaceTracker

logger = logging.getLogger(__name__)


class HistoricalService:
    """Service orchestrating background historical video analysis jobs using the canonical recognition pipeline."""

    def __init__(
        self,
        face_engine: Optional[FaceEngine] = None,
        face_index: Optional[FaceIndex] = None,
        alert_service: Optional[AlertService] = None,
        session_factory=None,
    ):
        self._engine = face_engine
        self._index = face_index
        self.alert_service = alert_service or get_alert_service()
        self.session_factory = session_factory or get_session_factory()
        self._active_jobs: dict[str, threading.Event] = {}

    def _get_engine(self) -> FaceEngine:
        if self._engine is not None:
            return self._engine
        return get_face_engine()

    def _get_index(self) -> FaceIndex:
        if self._index is not None:
            return self._index
        return get_face_index()

    def start_background_search(
        self,
        job_id: str,
        camera_id: str,
        file_path: str,
        sample_fps: float = DEFAULT_SAMPLE_FPS,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> None:
        """Spawn background worker thread for video analysis."""
        cancel_event = threading.Event()
        self._active_jobs[job_id] = cancel_event

        thread = threading.Thread(
            target=self._run_job,
            args=(job_id, camera_id, file_path, sample_fps, threshold, cancel_event),
            name=f"HistoricalJob-{job_id}",
            daemon=True,
        )
        thread.start()
        logger.info("Dispatched background historical search thread for job '%s'", job_id)

    def cancel_job(self, job_id: str) -> Optional[dict]:
        """Cancel an ongoing, queued, or existing background analysis job."""
        if job_id in self._active_jobs:
            self._active_jobs[job_id].set()
            logger.info("Signal sent to cancel active historical job thread '%s'", job_id)

        with self.session_factory() as session:
            repo = HistoricalJobRepository(session)
            job = repo.get_by_job_id(job_id)
            if not job:
                return None
            if job.status in ("COMPLETED", "FAILED", "CANCELLED"):
                return job.to_dict()

            updated_job = repo.update_progress(
                job_id=job_id,
                status="CANCELLED",
                processed_duration_sec=job.processed_duration_sec or 0.0,
                progress_percent=job.progress_percent or 0.0,
                frames_sampled=job.frames_sampled or 0,
                faces_detected=job.faces_detected or 0,
                tracks_created=job.tracks_created or 0,
                potential_matches=job.potential_matches or 0,
                completed=True,
            )
            if updated_job:
                ws_manager.broadcast_sync({
                    "event": "job.completed",
                    "job": updated_job.to_dict(),
                })
                return updated_job.to_dict()
            return job.to_dict()

    def _run_job(
        self,
        job_id: str,
        camera_id: str,
        file_path: str,
        sample_fps: float,
        threshold: float,
        cancel_event: threading.Event,
    ) -> None:
        """Background thread execution loop for video search."""
        src_path = Path(file_path)
        if not src_path.is_absolute():
            src_path = PROJECT_ROOT / file_path

        cap = cv2.VideoCapture(str(src_path))
        if not cap.isOpened():
            err_msg = f"Failed to open video file: {file_path}"
            logger.error("[HistoricalJob %s] %s", job_id, err_msg)
            with self.session_factory() as session:
                repo = HistoricalJobRepository(session)
                repo.update_progress(
                    job_id=job_id,
                    status="FAILED",
                    processed_duration_sec=0.0,
                    progress_percent=0.0,
                    frames_sampled=0,
                    faces_detected=0,
                    tracks_created=0,
                    potential_matches=0,
                    error_message=err_msg,
                    completed=True,
                )
            self._active_jobs.pop(job_id, None)
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_duration_sec = total_frames / fps if total_frames > 0 else 0.0

        with self.session_factory() as session:
            repo = HistoricalJobRepository(session)
            repo.update_progress(
                job_id=job_id,
                status="PROCESSING",
                processed_duration_sec=0.0,
                progress_percent=0.0,
                frames_sampled=0,
                faces_detected=0,
                tracks_created=0,
                potential_matches=0,
            )

        tracker = FaceTracker(
            iou_threshold=DEFAULT_IOU_THRESHOLD,
            max_missed_frames=DEFAULT_MAX_MISSED_FRAMES,
            top_k=DEFAULT_TRACK_TOP_K,
        )

        sample_interval = 1.0 / max(0.1, sample_fps)
        next_sample_time = 0.0
        frame_idx = 0
        frames_sampled = 0
        faces_detected = 0
        potential_matches = 0
        start_mono = time.monotonic()
        last_progress_broadcast = start_mono

        logger.info(
            "[HistoricalJob %s] Executing SEARCH pipeline on '%s' (camera_id='%s', sample_fps=%.1f, threshold=%.2f) via Canonical FaceEngine + FAISS",
            job_id,
            src_path.name,
            camera_id,
            sample_fps,
            threshold,
        )

        try:
            while not cancel_event.is_set():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                current_timestamp = frame_idx / fps
                if current_timestamp + 1e-5 >= next_sample_time:
                    frames_sampled += 1
                    next_sample_time = current_timestamp + sample_interval

                    # Canonical Recognition Pipeline (SCRFD -> ArcFace -> FAISS -> FaceTracker -> AlertService)
                    res: FrameRecognitionResult = recognize_frame(
                        frame=frame,
                        frame_idx=frame_idx,
                        timestamp_sec=current_timestamp,
                        camera_id=f"{camera_id}_historical",
                        tracker=tracker,
                        face_engine=self._get_engine(),
                        face_index=self._get_index(),
                        alert_service=self.alert_service,
                        threshold=threshold,
                        dispatch_alerts=True,
                    )
                    faces_detected += res.faces_detected
                    potential_matches += res.potential_matches

                frame_idx += 1

                # Broadcast progress every 1 second
                now = time.monotonic()
                if now - last_progress_broadcast >= 1.0:
                    elapsed = max(0.01, now - start_mono)
                    speed_fps = frames_sampled / elapsed
                    processed_sec = frame_idx / fps
                    progress_pct = (frame_idx / total_frames * 100.0) if total_frames > 0 else 0.0

                    with self.session_factory() as session:
                        repo = HistoricalJobRepository(session)
                        job = repo.update_progress(
                            job_id=job_id,
                            status="PROCESSING",
                            processed_duration_sec=processed_sec,
                            progress_percent=progress_pct,
                            frames_sampled=frames_sampled,
                            faces_detected=faces_detected,
                            tracks_created=len(tracker.get_all_tracks()),
                            potential_matches=potential_matches,
                            processing_speed_fps=speed_fps,
                        )
                        if job:
                            ws_manager.broadcast_sync({
                                "event": "job.progress",
                                "job": job.to_dict(),
                            })
                    last_progress_broadcast = now

            # Finalize analysis
            all_tracks = tracker.finalize()
            final_status = "CANCELLED" if cancel_event.is_set() else "COMPLETED"
            total_elapsed = max(0.01, time.monotonic() - start_mono)
            final_speed = frames_sampled / total_elapsed

            with self.session_factory() as session:
                repo = HistoricalJobRepository(session)
                job = repo.update_progress(
                    job_id=job_id,
                    status=final_status,
                    processed_duration_sec=total_duration_sec,
                    progress_percent=100.0 if final_status == "COMPLETED" else (frame_idx / total_frames * 100.0 if total_frames > 0 else 0.0),
                    frames_sampled=frames_sampled,
                    faces_detected=faces_detected,
                    tracks_created=len(all_tracks),
                    potential_matches=potential_matches,
                    processing_speed_fps=final_speed,
                    completed=True,
                )
                if job:
                    ws_manager.broadcast_sync({
                        "event": "job.completed",
                        "job": job.to_dict(),
                    })
            logger.info("[HistoricalJob %s] Finished with status '%s' in %.1fs", job_id, final_status, total_elapsed)

        except Exception as e:
            logger.exception("[HistoricalJob %s] Error during execution: %s", job_id, e)
            with self.session_factory() as session:
                repo = HistoricalJobRepository(session)
                repo.update_progress(
                    job_id=job_id,
                    status="FAILED",
                    processed_duration_sec=frame_idx / fps if fps > 0 else 0.0,
                    progress_percent=0.0,
                    frames_sampled=frames_sampled,
                    faces_detected=faces_detected,
                    tracks_created=len(tracker.get_all_tracks()),
                    potential_matches=potential_matches,
                    error_message=str(e),
                    completed=True,
                )
        finally:
            cap.release()
            self._active_jobs.pop(job_id, None)


_historical_service_instance: Optional[HistoricalService] = None


def get_historical_service() -> HistoricalService:
    """Singleton getter for HistoricalService."""
    global _historical_service_instance
    if _historical_service_instance is None:
        _historical_service_instance = HistoricalService()
    return _historical_service_instance
