"""
Camera Worker Thread.
Runs a dedicated video ingestion and recognition pipeline for a single camera (e.g. C1, C2, C3, C4).
Processes frames with SCRFD face detection, ArcFace embedding, FAISS matching, FaceTracker,
and forwards track alerts to the central AlertService.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np

from src.config import (
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_MAX_MISSED_FRAMES,
    DEFAULT_SAMPLE_FPS,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_TRACK_TOP_K,
    EMBEDDINGS_DIR,
    INDEX_PATH,
    METADATA_PATH,
)
from src.face_engine import DetectedFace, FaceEngine
from src.index import FaceIndex, MatchResult
from src.services.alert_service import AlertService, get_alert_service
from src.tracker import DetectionItem, FaceTracker, Track, TrackAlert

logger = logging.getLogger(__name__)


@dataclass
class CameraWorkerMetrics:
    """Real-time runtime metrics for a camera worker."""
    camera_id: str
    name: str
    source: str
    status: str  # 'STOPPED', 'STARTING', 'RUNNING', 'COMPLETED', 'ERROR'
    frames_read: int = 0
    frames_processed: int = 0
    faces_detected: int = 0
    tracks_created: int = 0
    alerts_emitted: int = 0
    fps: float = 0.0
    last_frame_timestamp: float = 0.0
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "name": self.name,
            "source": self.source,
            "status": self.status,
            "frames_read": self.frames_read,
            "frames_processed": self.frames_processed,
            "faces_detected": self.faces_detected,
            "tracks_created": self.tracks_created,
            "alerts_emitted": self.alerts_emitted,
            "fps": round(self.fps, 1),
            "last_frame_timestamp": round(self.last_frame_timestamp, 2),
            "error_message": self.error_message,
        }


class CameraWorker(threading.Thread):
    """
    Dedicated worker thread for an individual camera stream.
    Reuses the Part 3 Recognition and Tracking Core.
    """

    def __init__(
        self,
        camera_id: str,
        source: Union[str, int, Path],
        name: Optional[str] = None,
        source_type: str = "file",
        sample_fps: float = DEFAULT_SAMPLE_FPS,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        max_missed_frames: int = DEFAULT_MAX_MISSED_FRAMES,
        track_top_k: int = DEFAULT_TRACK_TOP_K,
        face_engine: Optional[FaceEngine] = None,
        face_index: Optional[FaceIndex] = None,
        alert_service: Optional[AlertService] = None,
        loop_video: bool = False,
    ):
        super().__init__(name=f"CameraWorker-{camera_id}", daemon=True)
        self.camera_id = camera_id
        self.source = str(source)
        self.source_type = source_type.upper()
        self.camera_name = name or f"Camera {camera_id}"
        self.sample_fps = max(0.1, sample_fps)
        self.threshold = threshold
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames
        self.track_top_k = track_top_k
        self.loop_video = loop_video

        # Core Engines
        self.engine = face_engine or FaceEngine()
        if face_index is not None:
            self.index = face_index
        else:
            self.index = FaceIndex()
            if INDEX_PATH.exists() and METADATA_PATH.exists():
                self.index.load(INDEX_PATH, METADATA_PATH)
            elif EMBEDDINGS_DIR.exists():
                self.index.build_from_directory(EMBEDDINGS_DIR)
        self.alert_service = alert_service or get_alert_service()

        # Thread Control
        self._stop_event = threading.Event()
        self.metrics = CameraWorkerMetrics(
            camera_id=self.camera_id,
            name=self.camera_name,
            source=self.source,
            status="STOPPED",
        )
        self.tracker: Optional[FaceTracker] = None

    def stop(self, timeout: float = 5.0) -> None:
        """Signal worker to stop and wait for termination."""
        self._stop_event.set()
        logger.info("Signaled CameraWorker '%s' to stop.", self.camera_id)
        if self.is_alive():
            self.join(timeout=timeout)

    @property
    def is_running(self) -> bool:
        return not self._stop_event.is_set() and self.is_alive()

    def run(self) -> None:
        """Main camera ingestion and face recognition loop."""
        self.metrics.status = "RUNNING"
        self._stop_event.clear()

        # Parse source (camera index vs video file)
        src = int(self.source) if self.source.isdigit() else self.source
        cap = cv2.VideoCapture(src)

        if not cap.isOpened():
            err = f"Failed to open video source: {self.source}"
            logger.error("[%s] %s", self.camera_id, err)
            self.metrics.status = "ERROR"
            self.metrics.error_message = err
            return

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0 or np.isnan(video_fps):
            video_fps = 30.0

        self.tracker = FaceTracker(
            iou_threshold=self.iou_threshold,
            max_missed_frames=self.max_missed_frames,
            top_k=self.track_top_k,
        )

        sample_interval = 1.0 / self.sample_fps
        next_sample_time = 0.0
        frame_idx = 0
        start_mono = time.monotonic()
        last_metric_update = start_mono

        logger.info(
            "[%s] Started CameraWorker (%s source: '%s', FPS: %.1f, Sample: %.1f FPS, Threshold: %.2f) via Canonical FaceEngine + FAISS",
            self.camera_id,
            self.source_type,
            self.source,
            video_fps,
            self.sample_fps,
            self.threshold,
        )

        try:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret or frame is None:
                    if self.loop_video and not self._stop_event.is_set():
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        logger.info("[%s] Video stream ended.", self.camera_id)
                        self.metrics.status = "COMPLETED"
                        break

                self.metrics.frames_read += 1
                current_timestamp = frame_idx / video_fps
                self.metrics.last_frame_timestamp = current_timestamp

                # Sampling interval check
                if current_timestamp + 1e-5 >= next_sample_time:
                    self.metrics.frames_processed += 1
                    next_sample_time = current_timestamp + sample_interval

                    self._process_single_frame(frame, frame_idx, current_timestamp)

                frame_idx += 1

                # Periodic FPS calculation
                now = time.monotonic()
                if now - last_metric_update >= 2.0:
                    elapsed = max(0.01, now - start_mono)
                    self.metrics.fps = self.metrics.frames_read / elapsed
                    last_metric_update = now

        except Exception as e:
            logger.exception("[%s] Error during camera worker execution: %s", self.camera_id, e)
            self.metrics.status = "ERROR"
            self.metrics.error_message = str(e)
        finally:
            cap.release()
            if self.tracker:
                all_tracks = self.tracker.finalize()
                self.metrics.tracks_created = len(all_tracks)
                for t in all_tracks:
                    self.alert_service.update_track_state(
                        camera_id=self.camera_id,
                        track_id=t.track_id,
                        person_id=t.matched_person_id,
                        person_name=t.matched_person_name,
                        first_seen=t.first_timestamp,
                        last_seen=t.last_timestamp,
                        first_frame=t.first_frame,
                        last_frame=t.last_frame,
                        detection_count=t.detection_count,
                        max_similarity=t.max_similarity,
                        mean_similarity=t.mean_similarity,
                        top_k_similarity=t.top_k_mean_similarity,
                        threshold_matches=t.threshold_match_count,
                        alert_triggered=t.alert_triggered,
                        status="TERMINATED",
                    )
            if self.metrics.status != "ERROR":
                if self.metrics.status == "RUNNING":
                    self.metrics.status = "STOPPED"
            logger.info("[%s] CameraWorker stopped.", self.camera_id)

    def _process_single_frame(self, frame: np.ndarray, frame_idx: int, current_timestamp: float) -> None:
        """
        Execute Part 3 pipeline for one sampled frame:
        SCRFD detection -> ArcFace embedding -> FAISS search -> FaceTracker -> AlertService.
        """
        faces: List[DetectedFace] = self.engine.detect_and_embed(frame)
        self.metrics.faces_detected += len(faces)

        detection_items: List[DetectionItem] = []
        for face_idx, face in enumerate(faces):
            search_results = self.index.search(
                query_embedding=face.normalized_embedding,
                k=1,
                threshold=self.threshold,
            )
            match = search_results[0] if search_results else MatchResult(None, "Unknown", 0.0, False, -1)
            is_match = match.is_match and (match.person_id is not None)

            det_item = DetectionItem(
                bbox=face.bbox,
                confidence=face.confidence,
                similarity=match.similarity,
                person_id=match.person_id if is_match else None,
                person_name=match.name if is_match else "Unknown",
                is_match=is_match,
                embedding=face.normalized_embedding,
                face_idx=face_idx,
            )
            detection_items.append(det_item)

        # Update temporal tracker
        assigned_tracks, new_alerts = self.tracker.update(
            detections=detection_items,
            frame_idx=frame_idx,
            timestamp_sec=current_timestamp,
            threshold=self.threshold,
            source_name=self.camera_id,
        )

        self.metrics.tracks_created = len(self.tracker.get_all_tracks())

        # Forward new track alerts to AlertService
        for alert in new_alerts:
            self.metrics.alerts_emitted += 1
            try:
                self.alert_service.create_alert_from_track_alert(
                    camera_id=self.camera_id,
                    track_alert=alert,
                )
            except Exception as e:
                logger.error("[%s] Failed to persist/broadcast alert: %s", self.camera_id, e)

        # Periodically update track records in database
        for t in assigned_tracks:
            self.alert_service.update_track_state(
                camera_id=self.camera_id,
                track_id=t.track_id,
                person_id=t.matched_person_id,
                person_name=t.matched_person_name,
                first_seen=t.first_timestamp,
                last_seen=t.last_timestamp,
                first_frame=t.first_frame,
                last_frame=t.last_frame,
                detection_count=t.detection_count,
                max_similarity=t.max_similarity,
                mean_similarity=t.mean_similarity,
                top_k_similarity=t.top_k_mean_similarity,
                threshold_matches=t.threshold_match_count,
                alert_triggered=t.alert_triggered,
                status="ACTIVE" if t.active else "TERMINATED",
            )

