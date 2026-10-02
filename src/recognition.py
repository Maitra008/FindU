"""
Video and Live Camera Recognition pipeline for Recognition Core (Part 3 Integrated).
Samples frames from a video stream/file or live webcam, detects faces with SCRFD,
extracts ArcFace embeddings, searches FAISS index, associates detections with FaceTracker,
aggregates track-level scores, renders persistent visual overlays, and guarantees
single-alert-per-track dispatch.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from src.config import (
    DEFAULT_COOLDOWN_SECONDS,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_MAX_MISSED_FRAMES,
    DEFAULT_SAMPLE_FPS,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_TRACK_TOP_K,
)
from src.face_engine import DetectedFace, FaceEngine, normalize_embedding
from src.index import FaceIndex, MatchResult
from src.tracker import DetectionItem, FaceTracker, Track, TrackAlert

logger = logging.getLogger(__name__)


@dataclass
class FaceRecognitionEvent:
    """Individual face recognition evaluation event."""
    detection_id: str
    frame_idx: int
    timestamp_sec: float
    face_idx: int
    bbox: List[int]
    detection_confidence: float
    similarity: float
    person_id: Optional[str]
    name: str
    is_match: bool
    is_suppressed: bool
    track_id: str = ""
    track_first_frame: int = 0
    track_last_frame: int = 0
    track_max_similarity: float = 0.0
    track_mean_similarity: float = 0.0
    track_threshold_match_count: int = 0
    alert_triggered: bool = False


@dataclass
class OverlayDetection:
    """Detection item cached for persistent display between sampled frames."""
    bbox: List[int]
    name: str
    similarity: float
    is_match: bool
    detection_id: str
    track_id: str
    timestamp: float


@dataclass
class VideoProcessingSummary:
    """Summary of video or webcam recognition execution."""
    source_name: str
    video_fps: float
    total_frames: int
    duration_sec: float
    frames_processed: int
    faces_detected: int
    potential_matches: int
    unknown_faces: int
    events: List[FaceRecognitionEvent] = field(default_factory=list)
    tracks: List[Track] = field(default_factory=list)
    alerts: List[TrackAlert] = field(default_factory=list)
    total_tracks: int = 0
    alerted_tracks: int = 0
    max_simultaneous_tracks: int = 0


class VideoRecognizer:
    """
    Orchestrates video/webcam frame sampling, face detection, embedding extraction,
    FAISS similarity search, FaceTracker association, visual overlay rendering,
    and single-alert-per-track dispatch.
    """

    def __init__(
        self,
        face_engine: Optional[FaceEngine] = None,
        face_index: Optional[FaceIndex] = None,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        sample_fps: float = DEFAULT_SAMPLE_FPS,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        max_missed_frames: int = DEFAULT_MAX_MISSED_FRAMES,
        track_top_k: int = DEFAULT_TRACK_TOP_K,
        verbose_debug: bool = False,
    ):
        """
        Initialize VideoRecognizer.

        Args:
            face_engine: Centralized FaceEngine instance.
            face_index: Populated FAISS FaceIndex instance.
            threshold: Cosine similarity cutoff for potential matches (default: 0.89).
            sample_fps: Frame sampling rate in frames per second (default: 2.0).
            cooldown_seconds: Secondary/legacy duplicate suppression window.
            iou_threshold: Minimum IoU overlap for tracking association (default: 0.30).
            max_missed_frames: Consecutive missed frames before track termination (default: 5).
            track_top_k: Top-k scores to average for track aggregation (default: 3).
            verbose_debug: Enable detailed frame/face debugging console output.
        """
        self.engine = face_engine or FaceEngine()
        self.index = face_index or FaceIndex()
        self.threshold = threshold
        self.sample_fps = sample_fps
        self.cooldown_seconds = cooldown_seconds
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames
        self.track_top_k = track_top_k
        self.verbose_debug = verbose_debug

    def _process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        current_timestamp: float,
        tracker: FaceTracker,
        effective_thresh: float,
        source_name: str = "",
        start_detection_id: int = 0,
    ) -> Tuple[List[FaceRecognitionEvent], List[OverlayDetection], List[TrackAlert], int, int, int]:
        """
        Shared recognition and tracking logic for a single video or camera frame:
        frame -> SCRFD detect -> ArcFace embed -> FAISS match -> FaceTracker update -> TrackAlert.

        Args:
            frame: OpenCV BGR image array.
            frame_idx: Integer index of the current frame.
            current_timestamp: Elapsed seconds since start of stream.
            tracker: Active FaceTracker instance.
            effective_thresh: Cosine similarity threshold.
            source_name: Source video/camera identifier.
            start_detection_id: Current cumulative face detection count.

        Returns:
            Tuple of:
              - new_events: List[FaceRecognitionEvent]
              - new_overlays: List[OverlayDetection]
              - frame_alerts: List[TrackAlert]
              - faces_count: int
              - matches_count: int
              - unknown_count: int
        """
        faces: List[DetectedFace] = self.engine.detect_and_embed(frame)

        new_events: List[FaceRecognitionEvent] = []
        new_overlays: List[OverlayDetection] = []
        faces_count = len(faces)
        matches_count = 0
        unknown_count = 0

        # Step 1: FAISS Search for all detected faces
        detection_items: List[DetectionItem] = []
        face_matches: List[MatchResult] = []

        for face_idx, face in enumerate(faces):
            search_results = self.index.search(
                query_embedding=face.normalized_embedding,
                k=1,
                threshold=effective_thresh,
            )
            match = search_results[0] if search_results else MatchResult(None, "Unknown", 0.0, False, -1)
            face_matches.append(match)

            is_match = match.is_match and (match.person_id is not None)
            sim_val = match.similarity
            identity_name = match.name if is_match else "Unknown"
            person_id = match.person_id if is_match else None

            det_item = DetectionItem(
                bbox=face.bbox,
                confidence=face.confidence,
                similarity=sim_val,
                person_id=person_id,
                person_name=identity_name,
                is_match=is_match,
                embedding=face.normalized_embedding,
                face_idx=face_idx,
            )
            detection_items.append(det_item)

        # Step 2: Temporal Tracker Association & Single-Alert Dispatch
        assigned_tracks, frame_alerts = tracker.update(
            detections=detection_items,
            frame_idx=frame_idx,
            timestamp_sec=current_timestamp,
            threshold=effective_thresh,
            source_name=source_name,
        )

        alerted_track_ids = {a.track_id for a in frame_alerts}

        # Step 3: Package Events and Overlays
        for face_idx, (face, match, det_item) in enumerate(zip(faces, face_matches, detection_items)):
            current_det_id = start_detection_id + face_idx + 1
            face_id_str = f"{current_det_id:03d}"

            assigned_track = assigned_tracks[face_idx] if face_idx < len(assigned_tracks) else None
            track_id_str = assigned_track.track_id if assigned_track else "TRACK-UNKNOWN"
            track_first_f = assigned_track.first_frame if assigned_track else frame_idx
            track_last_f = assigned_track.last_frame if assigned_track else frame_idx
            track_max_s = assigned_track.max_similarity if assigned_track else det_item.similarity
            track_mean_s = assigned_track.mean_similarity if assigned_track else det_item.similarity
            track_thresh_cnt = assigned_track.threshold_match_count if assigned_track else (1 if det_item.similarity >= effective_thresh else 0)
            track_alert_trig = assigned_track.alert_triggered if assigned_track else False

            is_match = det_item.is_match
            sim_val = det_item.similarity
            identity_name = det_item.person_name
            person_id = det_item.person_id

            # Duplicate suppression: suppressed if the track previously triggered an alert
            # and is not currently emitting a new alert in this frame
            is_suppressed = False
            if is_match and person_id:
                if track_id_str in alerted_track_ids:
                    matches_count += 1
                else:
                    is_suppressed = True
            else:
                unknown_count += 1
                if self.verbose_debug:
                    print(f"unknown, similarity {sim_val:.2f}, timestamp {current_timestamp:.1f}s, track {track_id_str}")

            if self.verbose_debug:
                print(
                    f"  [DEBUG] face_id={face_id_str} track={track_id_str} frame={frame_idx} face_idx={face_idx} "
                    f"bbox={face.bbox} det_conf={face.confidence:.2f} identity={identity_name} "
                    f"similarity={sim_val:.2f} alert_trig={track_alert_trig} suppressed={is_suppressed}"
                )

            new_overlays.append(
                OverlayDetection(
                    bbox=face.bbox,
                    name=identity_name,
                    similarity=sim_val,
                    is_match=is_match,
                    detection_id=face_id_str,
                    track_id=track_id_str,
                    timestamp=current_timestamp,
                )
            )

            event = FaceRecognitionEvent(
                detection_id=face_id_str,
                frame_idx=frame_idx,
                timestamp_sec=current_timestamp,
                face_idx=face_idx,
                bbox=face.bbox,
                detection_confidence=face.confidence,
                similarity=sim_val,
                person_id=person_id,
                name=identity_name,
                is_match=is_match,
                is_suppressed=is_suppressed,
                track_id=track_id_str,
                track_first_frame=track_first_f,
                track_last_frame=track_last_f,
                track_max_similarity=track_max_s,
                track_mean_similarity=track_mean_s,
                track_threshold_match_count=track_thresh_cnt,
                alert_triggered=track_alert_trig,
            )
            new_events.append(event)

        # Print alert notifications to terminal
        for alert in frame_alerts:
            print(
                f"potential match: {alert.person_name}, similarity {alert.similarity:.2f}, "
                f"timestamp {alert.timestamp_sec:.1f}s, track {alert.track_id}"
            )

        return new_events, new_overlays, frame_alerts, faces_count, matches_count, unknown_count

    def process_video(
        self,
        video_path: Union[str, Path],
        threshold: Optional[float] = None,
        sample_fps: Optional[float] = None,
        cooldown_seconds: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        max_missed_frames: Optional[int] = None,
        display: bool = False,
    ) -> VideoProcessingSummary:
        """
        Process a video file with frame sampling, face recognition, and temporal tracking.

        Args:
            video_path: Path to the video file.
            threshold: Optional override for similarity threshold (default: 0.89).
            sample_fps: Optional override for sample FPS (default: 2.0).
            cooldown_seconds: Optional override for duplicate alert cooldown.
            iou_threshold: Optional override for tracking IoU threshold (default: 0.30).
            max_missed_frames: Optional override for track termination threshold (default: 5).
            display: Whether to render a visual display window.

        Returns:
            VideoProcessingSummary with statistics, detected events, tracks, and alerts.
        """
        video_p = Path(video_path)
        if not video_p.exists() or not video_p.is_file():
            raise FileNotFoundError(f"Video file not found: {video_p}")

        effective_thresh = threshold if threshold is not None else self.threshold
        effective_sample_fps = sample_fps if sample_fps is not None else self.sample_fps
        effective_cooldown = cooldown_seconds if cooldown_seconds is not None else self.cooldown_seconds
        effective_iou = iou_threshold if iou_threshold is not None else self.iou_threshold
        effective_missed = max_missed_frames if max_missed_frames is not None else self.max_missed_frames

        cap = cv2.VideoCapture(str(video_p))
        if not cap.isOpened():
            raise IOError(f"Failed to open video file: {video_p}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0 or np.isnan(video_fps):
            video_fps = 30.0
            logger.warning("Video FPS could not be determined from metadata; defaulting to %.1f", video_fps)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / video_fps if total_frames > 0 else 0.0

        print(f"Processing video: {video_p.name}")
        print(f"FPS: {video_fps:.1f}")
        print(f"Sampling: {effective_sample_fps:.1f} FPS (Threshold: {effective_thresh:.2f}, Tracking IoU: {effective_iou:.2f})")
        print()

        logger.info(
            "Starting video recognition for '%s' (FPS: %.2f, Frames: %d, Sample FPS: %.2f, Threshold: %.2f)",
            video_p.name,
            video_fps,
            total_frames,
            effective_sample_fps,
            effective_thresh,
        )

        tracker = FaceTracker(
            iou_threshold=effective_iou,
            max_missed_frames=effective_missed,
            top_k=self.track_top_k,
        )

        frames_processed = 0
        faces_detected = 0
        potential_matches = 0
        unknown_count = 0
        events: List[FaceRecognitionEvent] = []
        all_alerts: List[TrackAlert] = []
        cached_overlays: List[OverlayDetection] = []

        sample_interval_sec = 1.0 / max(0.01, effective_sample_fps)
        next_sample_timestamp = 0.0
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                current_timestamp = frame_idx / video_fps

                if current_timestamp + 1e-5 >= next_sample_timestamp:
                    frames_processed += 1
                    next_sample_timestamp = current_timestamp + sample_interval_sec

                    (
                        new_ev,
                        new_ov,
                        new_alerts,
                        f_count,
                        m_count,
                        u_count,
                    ) = self._process_frame(
                        frame=frame,
                        frame_idx=frame_idx,
                        current_timestamp=current_timestamp,
                        tracker=tracker,
                        effective_thresh=effective_thresh,
                        source_name=video_p.name,
                        start_detection_id=faces_detected,
                    )

                    faces_detected += f_count
                    potential_matches += m_count
                    unknown_count += u_count
                    events.extend(new_ev)
                    all_alerts.extend(new_alerts)
                    cached_overlays = new_ov

                if display:
                    display_frame = frame.copy()
                    self._draw_overlays(
                        frame=display_frame,
                        overlays=cached_overlays,
                        current_timestamp=current_timestamp,
                        sample_fps=effective_sample_fps,
                        total_matches=len(all_alerts),
                        active_tracks_count=len(tracker.active_tracks),
                    )
                    cv2.imshow("Recognition Core - Video Playback (Press 'q' to stop)", display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == 27:
                        break

                frame_idx += 1

        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()

        all_tracks = tracker.finalize()
        alerted_tracks = [t for t in all_tracks if t.alert_triggered]

        print()
        print("============================================================")
        print("              VIDEO RECOGNITION SUMMARY")
        print("============================================================")
        print(f"Video source:          {video_p.name}")
        print(f"Total frames:          {frame_idx}")
        print(f"Frames processed:      {frames_processed}")
        print(f"Faces detected:        {faces_detected}")
        print(f"Potential matches:     {len(all_alerts)}")
        print(f"Match threshold:       {effective_thresh:.2f}")
        print(f"Cooldown period:       {effective_cooldown:.1f}s")
        print(f"Track count:           {len(all_tracks)}")
        print(f"Tracks alerted:        {len(alerted_tracks)}")
        print(f"Max simultaneous tracks: {tracker.max_simultaneous_tracks}")
        print("------------------------------------------------------------")
        print("                    TRACKING SUMMARY")
        print("------------------------------------------------------------")
        for track in all_tracks:
            alert_str = "YES" if track.alert_triggered else "NO"
            pid_str = f" ({track.matched_person_id})" if track.matched_person_id else ""
            print(f"{track.track_id}:")
            print(f"  Identity:            {track.matched_person_name}{pid_str}")
            print(f"  Frame span:          {track.first_frame} -> {track.last_frame} (span: {track.last_frame - track.first_frame + 1} frames)")
            print(f"  Detections:          {track.detection_count}")
            print(f"  Max similarity:      {track.max_similarity:.4f}")
            print(f"  Mean similarity:     {track.mean_similarity:.4f}")
            print(f"  Top-{self.track_top_k} similarity:    {track.top_k_mean_similarity:.4f}")
            print(f"  Threshold matches (>=0.89): {track.threshold_match_count}")
            print(f"  Alert triggered:     {alert_str}")
            print()
        print("============================================================")

        return VideoProcessingSummary(
            source_name=video_p.name,
            video_fps=video_fps,
            total_frames=frame_idx,
            duration_sec=duration_sec,
            frames_processed=frames_processed,
            faces_detected=faces_detected,
            potential_matches=len(all_alerts),
            unknown_faces=unknown_count,
            events=events,
            tracks=all_tracks,
            alerts=all_alerts,
            total_tracks=len(all_tracks),
            alerted_tracks=len(alerted_tracks),
            max_simultaneous_tracks=tracker.max_simultaneous_tracks,
        )

    def process_camera(
        self,
        camera_index: int = 0,
        threshold: Optional[float] = None,
        sample_fps: Optional[float] = None,
        cooldown_seconds: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        max_missed_frames: Optional[int] = None,
        display: bool = True,
        persistence_timeout: float = 1.5,
    ) -> VideoProcessingSummary:
        """
        Process a live camera feed with responsive frame capture, configurable inference sampling,
        persistent visual annotations, temporal tracking, and single-alert-per-track dispatch.

        Args:
            camera_index: Webcam device index (default: 0).
            threshold: Optional override for similarity threshold (default: 0.89).
            sample_fps: Optional override for sample FPS (default: 2.0).
            cooldown_seconds: Optional override for duplicate alert cooldown.
            iou_threshold: Optional override for tracking IoU threshold (default: 0.30).
            max_missed_frames: Optional override for track termination threshold (default: 5).
            display: Whether to display the live OpenCV GUI window (default: True).
            persistence_timeout: Duration in seconds to retain bounding boxes between inferences.

        Returns:
            VideoProcessingSummary with statistics, detected events, tracks, and alerts.
        """
        effective_thresh = threshold if threshold is not None else self.threshold
        effective_sample_fps = sample_fps if sample_fps is not None else self.sample_fps
        effective_cooldown = cooldown_seconds if cooldown_seconds is not None else self.cooldown_seconds
        effective_iou = iou_threshold if iou_threshold is not None else self.iou_threshold
        effective_missed = max_missed_frames if max_missed_frames is not None else self.max_missed_frames

        source_name = f"Camera #{camera_index}"
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise IOError(f"Failed to open webcam at index: {camera_index}")

        print(f"Processing camera: {source_name}")
        print(f"Sampling: {effective_sample_fps:.1f} FPS (Threshold: {effective_thresh:.2f}, Tracking IoU: {effective_iou:.2f})")
        if display:
            print("Visual feed active: Press 'q' or 'ESC' in the webcam window to stop.")
        print()

        logger.info(
            "Starting live camera recognition on '%s' (Sample FPS: %.2f, Threshold: %.2f, IoU: %.2f)",
            source_name,
            effective_sample_fps,
            effective_thresh,
            effective_iou,
        )

        tracker = FaceTracker(
            iou_threshold=effective_iou,
            max_missed_frames=effective_missed,
            top_k=self.track_top_k,
        )

        frames_processed = 0
        faces_detected = 0
        potential_matches = 0
        unknown_count = 0
        events: List[FaceRecognitionEvent] = []
        all_alerts: List[TrackAlert] = []
        cached_overlays: List[OverlayDetection] = []

        sample_interval_sec = 1.0 / max(0.01, effective_sample_fps)
        next_sample_timestamp = 0.0
        frame_idx = 0
        start_time = time.monotonic()

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    logger.warning("Failed to grab frame from camera.")
                    break

                current_timestamp = time.monotonic() - start_time

                # Continuous capture; execute ML recognition only at sample intervals
                if current_timestamp + 1e-5 >= next_sample_timestamp:
                    frames_processed += 1
                    next_sample_timestamp = current_timestamp + sample_interval_sec

                    (
                        new_ev,
                        new_ov,
                        new_alerts,
                        f_count,
                        m_count,
                        u_count,
                    ) = self._process_frame(
                        frame=frame,
                        frame_idx=frame_idx,
                        current_timestamp=current_timestamp,
                        tracker=tracker,
                        effective_thresh=effective_thresh,
                        source_name=source_name,
                        start_detection_id=faces_detected,
                    )

                    faces_detected += f_count
                    potential_matches += m_count
                    unknown_count += u_count
                    events.extend(new_ev)
                    all_alerts.extend(new_alerts)
                    cached_overlays = new_ov

                if display:
                    display_frame = frame.copy()
                    valid_overlays = [
                        ov for ov in cached_overlays
                        if (current_timestamp - ov.timestamp) <= persistence_timeout
                    ]

                    self._draw_overlays(
                        frame=display_frame,
                        overlays=valid_overlays,
                        current_timestamp=current_timestamp,
                        sample_fps=effective_sample_fps,
                        total_matches=len(all_alerts),
                        active_tracks_count=len(tracker.active_tracks),
                    )
                    cv2.imshow("Recognition Core - Live Webcam (Press 'q' to stop)", display_frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == 27:
                        logger.info("Webcam stopped by user key press.")
                        break

                frame_idx += 1

        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()

        elapsed_duration = time.monotonic() - start_time
        all_tracks = tracker.finalize()
        alerted_tracks = [t for t in all_tracks if t.alert_triggered]

        print()
        print("============================================================")
        print("              CAMERA RECOGNITION SUMMARY")
        print("============================================================")
        print(f"Source:                {source_name}")
        print(f"Total frames:          {frame_idx}")
        print(f"Frames processed:      {frames_processed}")
        print(f"Faces detected:        {faces_detected}")
        print(f"Potential matches:     {len(all_alerts)}")
        print(f"Match threshold:       {effective_thresh:.2f}")
        print(f"Track count:           {len(all_tracks)}")
        print(f"Tracks alerted:        {len(alerted_tracks)}")
        print(f"Max simultaneous tracks: {tracker.max_simultaneous_tracks}")
        print("------------------------------------------------------------")
        print("                    TRACKING SUMMARY")
        print("------------------------------------------------------------")
        for track in all_tracks:
            alert_str = "YES" if track.alert_triggered else "NO"
            pid_str = f" ({track.matched_person_id})" if track.matched_person_id else ""
            print(f"{track.track_id}:")
            print(f"  Identity:            {track.matched_person_name}{pid_str}")
            print(f"  Frame span:          {track.first_frame} -> {track.last_frame} (span: {track.last_frame - track.first_frame + 1} frames)")
            print(f"  Detections:          {track.detection_count}")
            print(f"  Max similarity:      {track.max_similarity:.4f}")
            print(f"  Mean similarity:     {track.mean_similarity:.4f}")
            print(f"  Top-{self.track_top_k} similarity:    {track.top_k_mean_similarity:.4f}")
            print(f"  Threshold matches (>=0.89): {track.threshold_match_count}")
            print(f"  Alert triggered:     {alert_str}")
            print()
        print("============================================================")

        return VideoProcessingSummary(
            source_name=source_name,
            video_fps=frame_idx / max(0.01, elapsed_duration),
            total_frames=frame_idx,
            duration_sec=elapsed_duration,
            frames_processed=frames_processed,
            faces_detected=faces_detected,
            potential_matches=len(all_alerts),
            unknown_faces=unknown_count,
            events=events,
            tracks=all_tracks,
            alerts=all_alerts,
            total_tracks=len(all_tracks),
            alerted_tracks=len(alerted_tracks),
            max_simultaneous_tracks=tracker.max_simultaneous_tracks,
        )

    @staticmethod
    def _draw_overlays(
        frame: np.ndarray,
        overlays: List[OverlayDetection],
        current_timestamp: float,
        sample_fps: float,
        total_matches: int,
        active_tracks_count: int = 0,
    ) -> None:
        """
        Draw visual bounding boxes and multiline labels:
        - Potential match:
            TRACK-0001
            Person X
            0.93 POTENTIAL MATCH
        - Unknown:
            TRACK-0002
            Unknown
            0.12
        """
        h, w, _ = frame.shape

        # Top HUD Banner
        hud_text = (
            f"Time: {current_timestamp:.1f}s | Sample: {sample_fps:.1f} FPS | "
            f"Tracks: {active_tracks_count} | Alerts: {total_matches} | Press 'q' to stop"
        )
        cv2.rectangle(frame, (0, 0), (w, 30), (25, 25, 25), -1)
        cv2.putText(
            frame,
            hud_text,
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )

        for ov in overlays:
            x1, y1, x2, y2 = ov.bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            box_color = (0, 220, 0) if ov.is_match else (0, 70, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            # Multi-line text label content
            if ov.is_match:
                lines = [
                    ov.track_id,
                    ov.name,
                    f"{ov.similarity:.2f} POTENTIAL MATCH",
                ]
            else:
                lines = [
                    ov.track_id,
                    ov.name,
                    f"{ov.similarity:.2f}",
                ]

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.50
            thickness = 1
            line_height = 18
            pad = 6

            max_text_w = 0
            for line in lines:
                (t_w, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
                max_text_w = max(max_text_w, t_w)

            total_h = len(lines) * line_height + pad

            # Position label box above face if space permits, else below
            if y1 - total_h >= 32:
                bg_y1 = y1 - total_h
                bg_y2 = y1
            else:
                bg_y1 = y2
                bg_y2 = min(h, y2 + total_h)

            bg_x1 = x1
            bg_x2 = min(w, x1 + max_text_w + pad * 2)

            # Draw dark background plate for readability
            cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (20, 20, 20), -1)
            cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), box_color, 1)

            # Draw label lines
            for i, line in enumerate(lines):
                text_y = bg_y1 + pad + (i + 1) * line_height - 4
                text_color = (0, 255, 0) if (ov.is_match and i == 2) else (255, 255, 255)
                cv2.putText(
                    frame,
                    line,
                    (bg_x1 + pad, text_y),
                    font,
                    font_scale,
                    text_color,
                    thickness,
                    cv2.LINE_AA,
                )
