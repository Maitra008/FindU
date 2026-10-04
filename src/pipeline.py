"""
Canonical Frame Recognition Pipeline for FindU.
Enforces the single canonical pipeline invariant:
ONE RECOGNITION ENGINE. ONE EXACT PIPELINE.

Input Frame
    ↓
Canonical Frame Sampling
    ↓
Canonical FaceEngine (Singleton)
    ↓
SCRFD Face Detection
    ↓
ArcFace 512-d Feature Embedding
    ↓
Strict L2 Unit Normalization
    ↓
Canonical FaceIndex (Shared Dynamic FAISS Cosine Index)
    ↓
Canonical Similarity Threshold (0.89)
    ↓
FaceTracker (IoU Temporal Association + Top-K Aggregation)
    ↓
AlertService (Single Alert per Target Track)
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from src.config import DEFAULT_SIMILARITY_THRESHOLD
from src.face_engine import DetectedFace, FaceEngine, get_face_engine
from src.index import FaceIndex, MatchResult, get_face_index
from src.services.alert_service import AlertService, get_alert_service
from src.tracker import DetectionItem, FaceTracker, Track, TrackAlert

logger = logging.getLogger(__name__)


@dataclass
class FrameRecognitionResult:
    """Standardized output of the canonical frame recognition pipeline."""
    faces: List[DetectedFace]
    detection_items: List[DetectionItem]
    assigned_tracks: List[Track]
    new_alerts: List[TrackAlert]
    faces_detected: int
    potential_matches: int


def recognize_frame(
    frame: np.ndarray,
    frame_idx: int,
    timestamp_sec: float,
    camera_id: str,
    tracker: FaceTracker,
    face_engine: Optional[FaceEngine] = None,
    face_index: Optional[FaceIndex] = None,
    alert_service: Optional[AlertService] = None,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    dispatch_alerts: bool = True,
) -> FrameRecognitionResult:
    """
    Execute the single canonical recognition pipeline for an individual frame.

    All input sources (USB webcam, RTSP network streams, video files, replay streams,
    and historical background search) MUST invoke this exact function.

    Args:
        frame: OpenCV BGR image numpy array.
        frame_idx: Frame sequence index.
        timestamp_sec: Elapsed video/stream time in seconds.
        camera_id: Camera identifier context (e.g. 'C1', 'C_USB_01', 'JOB-XXX').
        tracker: FaceTracker instance maintaining temporal track state.
        face_engine: Canonical FaceEngine instance (defaults to shared singleton).
        face_index: Shared FAISS FaceIndex instance (defaults to shared singleton).
        alert_service: AlertService for alert persistence & dispatch (defaults to singleton).
        threshold: Cosine similarity threshold for identity match (default: 0.89).
        dispatch_alerts: Whether to persist/broadcast alerts via AlertService.

    Returns:
        FrameRecognitionResult containing detected faces, tracks, alerts, and metrics.
    """
    engine = face_engine or get_face_engine()
    index = face_index or get_face_index()

    # Step 1-3: SCRFD Detection + ArcFace Embedding + L2 Normalization
    faces: List[DetectedFace] = engine.detect_and_embed(frame)

    # Step 4-5: FAISS Cosine Similarity Search against registered identities
    detection_items: List[DetectionItem] = []
    potential_matches = 0

    for face_idx, face in enumerate(faces):
        search_results = index.search(
            query_embedding=face.normalized_embedding,
            k=1,
            threshold=threshold,
        )
        match = search_results[0] if search_results else MatchResult(None, "Unknown", 0.0, False, -1)
        is_match = match.is_match and (match.person_id is not None)
        if is_match:
            potential_matches += 1

        detection_items.append(
            DetectionItem(
                bbox=face.bbox,
                confidence=face.confidence,
                similarity=match.similarity,
                person_id=match.person_id if is_match else None,
                person_name=match.name if is_match else "Unknown",
                is_match=is_match,
                embedding=face.normalized_embedding,
                face_idx=face_idx,
            )
        )

    # Step 6: Temporal Tracker Association & Top-K scoring
    assigned_tracks, new_alerts = tracker.update(
        detections=detection_items,
        frame_idx=frame_idx,
        timestamp_sec=timestamp_sec,
        threshold=threshold,
        source_name=camera_id,
    )

    # Step 7: Single-Alert-Per-Track dispatch to AlertService
    if dispatch_alerts and new_alerts:
        svc = alert_service or get_alert_service()
        for alert in new_alerts:
            try:
                svc.create_alert_from_track_alert(
                    camera_id=camera_id,
                    track_alert=alert,
                )
            except Exception as e:
                logger.error("[%s] Failed to dispatch alert for track %s: %s", camera_id, alert.track_id, e)

    return FrameRecognitionResult(
        faces=faces,
        detection_items=detection_items,
        assigned_tracks=assigned_tracks,
        new_alerts=new_alerts,
        faces_detected=len(faces),
        potential_matches=potential_matches,
    )
