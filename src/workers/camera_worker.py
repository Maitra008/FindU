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
    PROJECT_ROOT,
)
from src.face_engine import DetectedFace, FaceEngine, get_face_engine
from src.index import FaceIndex, MatchResult, get_face_index
from src.pipeline import FrameRecognitionResult, recognize_frame
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
    Executes the canonical recognition pipeline.
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

        # Canonical Singletons
        self.engine = face_engine or get_face_engine()
        self.index = face_index or get_face_index()
        self.alert_service = alert_service or get_alert_service()

        # Thread & Playback Control
        self.playback_speed: float = 1.0
        self._latest_jpeg: Optional[bytes] = None
        self._frame_lock = threading.Lock()
        self._stop_event = threading.Event()
        self.metrics = CameraWorkerMetrics(
            camera_id=self.camera_id,
            name=self.camera_name,
            source=self.source,
            status="STOPPED",
        )
        self.tracker: Optional[FaceTracker] = None

    def set_playback_speed(self, speed: float) -> None:
        """Set replay playback speed multiplier (e.g. 1.0 or 2.0)."""
        self.playback_speed = max(0.25, min(8.0, speed))
        logger.info("[%s] Playback speed updated to %.2fx", self.camera_id, self.playback_speed)

    def get_latest_jpeg(self) -> Optional[bytes]:
        """Return the latest frame encoded as JPEG bytes for MJPEG streaming."""
        with self._frame_lock:
            return self._latest_jpeg

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

        # Parse source (USB index vs RTSP stream vs video file)
        if self.source.isdigit():
            src_idx = int(self.source)
            cap = cv2.VideoCapture(src_idx, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(src_idx)
        else:
            src_path = Path(self.source)
            if not src_path.is_absolute():
                src_path = PROJECT_ROOT / self.source
            if src_path.exists():
                cap = cv2.VideoCapture(str(src_path))
            else:
                cap = cv2.VideoCapture(self.source)

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
        is_file_source = self.source_type == "FILE" or self.loop_video

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
                frame_start_time = time.monotonic()
                ret, frame = cap.read()
                if not ret or frame is None:
                    if self.loop_video and not self._stop_event.is_set():
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        frame_idx = 0
                        continue
                    else:
                        logger.info("[%s] Video stream ended.", self.camera_id)
                        self.metrics.status = "COMPLETED"
                        break

                # Store latest JPEG for browser streaming preview
                try:
                    encode_success, buffer = cv2.imencode(
                        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75]
                    )
                    if encode_success:
                        with self._frame_lock:
                            self._latest_jpeg = buffer.tobytes()
                except Exception:
                    pass

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

                # Real-time pacing for file playback
                if is_file_source and not self._stop_event.is_set():
                    target_delay = (1.0 / max(1.0, video_fps * self.playback_speed))
                    proc_time = time.monotonic() - frame_start_time
                    sleep_time = target_delay - proc_time
                    if sleep_time > 0.001:
                        time.sleep(sleep_time)

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
        Execute the canonical recognition pipeline for one sampled frame:
        SCRFD detection -> ArcFace embedding -> FAISS search -> FaceTracker -> AlertService.
        """
        if self.tracker is None:
            return

        result: FrameRecognitionResult = recognize_frame(
            frame=frame,
            frame_idx=frame_idx,
            timestamp_sec=current_timestamp,
            camera_id=self.camera_id,
            tracker=self.tracker,
            face_engine=self.engine,
            face_index=self.index,
            alert_service=self.alert_service,
            threshold=self.threshold,
            dispatch_alerts=True,
        )

        self.metrics.faces_detected += result.faces_detected
        self.metrics.tracks_created = len(self.tracker.get_all_tracks())
        self.metrics.alerts_emitted += len(result.new_alerts)

        # Periodically update track records in database
        for t in result.assigned_tracks:
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
                status="ACTIVE",
            )
