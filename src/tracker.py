"""
Temporal Face Tracking and Score Aggregation Module (Part 3).
Associates face detections across consecutive video/camera frames using IoU matching,
maintains persistent track states, aggregates similarity metrics, and enforces
single-alert-per-track dispatch.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.config import DEFAULT_SIMILARITY_THRESHOLD, DEFAULT_TEMPORAL_CONFIRMATIONS

logger = logging.getLogger(__name__)


def compute_iou(bbox1: List[int], bbox2: List[int]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    Boxes are in format [x1, y1, x2, y2].

    Args:
        bbox1: [x1, y1, x2, y2]
        bbox2: [x1, y1, x2, y2]

    Returns:
        IoU float in range [0.0, 1.0].
    """
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    intersection_w = max(0, x2 - x1)
    intersection_h = max(0, y2 - y1)
    intersection_area = intersection_w * intersection_h

    area1 = max(0, bbox1[2] - bbox1[0]) * max(0, bbox1[3] - bbox1[1])
    area2 = max(0, bbox2[2] - bbox2[0]) * max(0, bbox2[3] - bbox2[1])

    union_area = area1 + area2 - intersection_area
    if union_area <= 0:
        return 0.0
    return float(intersection_area / union_area)


@dataclass
class DetectionItem:
    """Detection item passed to tracker for a single frame."""
    bbox: List[int]
    confidence: float
    similarity: float
    person_id: Optional[str]
    person_name: str
    is_match: bool
    embedding: Optional[Any] = None
    face_idx: int = 0
    quality_passed: bool = True
    rejection_reason: Optional[str] = None
    quality_reasons: List[str] = field(default_factory=list)


@dataclass
class TrackAlert:
    """Alert event dispatched by a track upon first threshold crossing."""
    track_id: str
    person_id: str
    person_name: str
    similarity: float
    max_similarity: float
    mean_similarity: float
    threshold: float
    frame_idx: int
    timestamp_sec: float
    bbox: List[int]
    detection_count: int
    source_name: str = ""
    threshold_match_count: int = 1
    match_frames: int = 1
    alert_triggered: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to serializable dictionary."""
        return {
            "track_id": self.track_id,
            "person_id": self.person_id,
            "person_name": self.person_name,
            "similarity": round(self.similarity, 4),
            "max_similarity": round(self.max_similarity, 4),
            "mean_similarity": round(self.mean_similarity, 4),
            "threshold": round(self.threshold, 4),
            "frame_idx": self.frame_idx,
            "timestamp_sec": round(self.timestamp_sec, 2),
            "bbox": self.bbox,
            "source_name": self.source_name,
            "detection_count": self.detection_count,
            "threshold_match_count": self.threshold_match_count,
            "alert_triggered": self.alert_triggered,
        }


@dataclass
class Track:
    """Persistent temporal track representing a single person across frames."""
    track_id: str
    bbox: List[int]
    first_frame: int
    last_frame: int
    first_timestamp: float
    last_timestamp: float
    detection_count: int = 1
    matched_person_id: Optional[str] = None
    matched_person_name: str = "Unknown"
    similarity_scores: List[float] = field(default_factory=list)
    detection_history: List[Dict[str, Any]] = field(default_factory=list)
    max_similarity: float = 0.0
    mean_similarity: float = 0.0
    top_k_mean_similarity: float = 0.0
    threshold_match_count: int = 0
    alert_triggered: bool = False
    missed_frames: int = 0
    active: bool = True
    best_match_frame: int = 0
    best_match_timestamp: float = 0.0
    best_match_bbox: List[int] = field(default_factory=list)

    @property
    def match_frames_above_threshold(self) -> int:
        """Compatibility property for threshold_match_count."""
        return self.threshold_match_count

    @match_frames_above_threshold.setter
    def match_frames_above_threshold(self, val: int) -> None:
        self.threshold_match_count = val

    def update(
        self,
        det: DetectionItem,
        frame_idx: int,
        timestamp_sec: float,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        top_k: int = 3,
        source_name: str = "",
        min_confirmations: int = DEFAULT_TEMPORAL_CONFIRMATIONS,
    ) -> Optional[TrackAlert]:
        """
        Update track with a newly associated detection and recalculate aggregate scores.

        Args:
            det: Matched DetectionItem.
            frame_idx: Current video frame index.
            timestamp_sec: Elapsed video time in seconds.
            threshold: Operating recognition similarity threshold.
            top_k: Number of highest scores to average for top_k_mean_similarity.
            source_name: Name/identifier of video or camera source.
            min_confirmations: Required matching detections before firing alert.

        Returns:
            Optional TrackAlert if this update triggered a new potential-match alert.
        """
        self.bbox = det.bbox
        self.last_frame = frame_idx
        self.last_timestamp = timestamp_sec
        self.detection_count += 1
        self.missed_frames = 0
        self.active = True

        sim = det.similarity
        self.similarity_scores.append(sim)

        if det.is_match and det.quality_passed and sim >= threshold:
            self.threshold_match_count += 1

        # Update best identity attribution
        if det.is_match and det.person_id:
            if sim >= self.max_similarity or self.matched_person_id is None:
                self.matched_person_id = det.person_id
                self.matched_person_name = det.person_name

        if sim > self.max_similarity or not self.best_match_bbox:
            self.max_similarity = sim
            self.best_match_frame = frame_idx
            self.best_match_timestamp = timestamp_sec
            self.best_match_bbox = det.bbox

        # Score aggregations
        self.mean_similarity = float(sum(self.similarity_scores) / len(self.similarity_scores))
        sorted_scores = sorted(self.similarity_scores, reverse=True)
        top_scores = sorted_scores[:top_k]
        self.top_k_mean_similarity = float(sum(top_scores) / len(top_scores))

        self.detection_history.append({
            "frame_idx": frame_idx,
            "timestamp_sec": timestamp_sec,
            "bbox": det.bbox,
            "similarity": sim,
            "person_id": det.person_id,
            "person_name": det.person_name,
            "is_match": det.is_match,
            "quality_passed": det.quality_passed,
        })

        # Check alert trigger: exactly ONE alert upon reaching min_confirmations
        alert_event = None
        if (
            det.is_match
            and det.quality_passed
            and det.person_id
            and self.threshold_match_count >= min_confirmations
            and not self.alert_triggered
        ):
            self.alert_triggered = True
            alert_event = TrackAlert(
                track_id=self.track_id,
                person_id=det.person_id,
                person_name=det.person_name,
                similarity=sim,
                max_similarity=self.max_similarity,
                mean_similarity=self.mean_similarity,
                threshold=threshold,
                frame_idx=frame_idx,
                timestamp_sec=timestamp_sec,
                bbox=det.bbox,
                detection_count=self.detection_count,
                source_name=source_name,
                threshold_match_count=self.threshold_match_count,
                match_frames=self.threshold_match_count,
                alert_triggered=True,
            )

        return alert_event


class FaceTracker:
    """
    IoU-based temporal face tracker with score aggregation, temporal confirmation, and single-alert dispatch.
    """

    def __init__(
        self,
        iou_threshold: float = 0.30,
        max_missed_frames: int = 5,
        top_k: int = 3,
        min_confirmations: int = DEFAULT_TEMPORAL_CONFIRMATIONS,
    ):
        """
        Initialize FaceTracker.

        Args:
            iou_threshold: Minimum IoU overlap required to match a detection to an existing track.
            max_missed_frames: Consecutive frames without detection before a track is terminated.
            top_k: Number of highest scores to average for track score aggregation.
            min_confirmations: Required matching detections before firing an alert.
        """
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames
        self.top_k = top_k
        self.min_confirmations = min_confirmations
        self.active_tracks: List[Track] = []
        self.terminated_tracks: List[Track] = []
        self.next_track_num: int = 1
        self.max_simultaneous_tracks: int = 0

    def update(
        self,
        detections: List[DetectionItem],
        frame_idx: int,
        timestamp_sec: float,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        source_name: str = "",
        min_confirmations: Optional[int] = None,
    ) -> Tuple[List[Track], List[TrackAlert]]:
        """
        Process frame detections, update active tracks, spawn new tracks, terminate dead tracks.

        Args:
            detections: List of DetectionItem from current frame.
            frame_idx: Current integer frame index.
            timestamp_sec: Current video timestamp in seconds.
            threshold: Recognition similarity threshold (default: 0.55).
            source_name: Optional video/camera stream identifier.
            min_confirmations: Optional override for confirmation count requirement.

        Returns:
            Tuple of:
              - assigned_tracks: List of Track instances corresponding 1-to-1 with input detections.
              - new_alerts: List of TrackAlert dispatched during this frame.
        """
        if min_confirmations is None:
            min_confirmations = self.min_confirmations

        new_alerts: List[TrackAlert] = []
        assigned_tracks: List[Optional[Track]] = [None] * len(detections)

        if not self.active_tracks:
            # All detections start new tracks
            for d_idx, det in enumerate(detections):
                track = self._create_new_track(det, frame_idx, timestamp_sec, threshold, self.top_k)
                if (
                    det.is_match
                    and det.quality_passed
                    and det.person_id
                    and track.threshold_match_count >= min_confirmations
                    and not track.alert_triggered
                ):
                    track.alert_triggered = True
                    alert = TrackAlert(
                        track_id=track.track_id,
                        person_id=det.person_id,
                        person_name=det.person_name,
                        similarity=det.similarity,
                        max_similarity=track.max_similarity,
                        mean_similarity=track.mean_similarity,
                        threshold=threshold,
                        frame_idx=frame_idx,
                        timestamp_sec=timestamp_sec,
                        bbox=det.bbox,
                        detection_count=track.detection_count,
                        source_name=source_name,
                        threshold_match_count=track.threshold_match_count,
                        match_frames=track.threshold_match_count,
                        alert_triggered=True,
                    )
                    new_alerts.append(alert)
                self.active_tracks.append(track)
                assigned_tracks[d_idx] = track

            self.max_simultaneous_tracks = max(self.max_simultaneous_tracks, len(self.active_tracks))
            return [t for t in assigned_tracks if t is not None], new_alerts

        # Compute IoU matching matrix between active tracks and detections
        matched_tracks, matched_dets = self._match_iou(detections)

        # 1. Update matched active tracks
        for track_idx, det_idx in zip(matched_tracks, matched_dets):
            track = self.active_tracks[track_idx]
            det = detections[det_idx]
            alert = track.update(
                det=det,
                frame_idx=frame_idx,
                timestamp_sec=timestamp_sec,
                threshold=threshold,
                top_k=self.top_k,
                source_name=source_name,
                min_confirmations=min_confirmations,
            )
            if alert:
                new_alerts.append(alert)
            assigned_tracks[det_idx] = track

        # 2. Handle unmatched tracks BEFORE adding new tracks.
        # Track indices refer to the pre-update active-track list. New tracks
        # must never be counted as missed on their creation frame.
        matched_track_set = set(matched_tracks)
        surviving_tracks: List[Track] = []
        for i, track in enumerate(self.active_tracks):
            if i not in matched_track_set:
                track.missed_frames += 1
                if track.missed_frames > self.max_missed_frames:
                    track.active = False
                    self.terminated_tracks.append(track)
                    logger.debug("Terminated track %s at frame %d", track.track_id, frame_idx)
                    continue
            surviving_tracks.append(track)

        # 3. Spawn new tracks for unmatched detections.
        unmatched_det_indices = [i for i in range(len(detections)) if i not in matched_dets]
        for det_idx in unmatched_det_indices:
            det = detections[det_idx]
            track = self._create_new_track(det, frame_idx, timestamp_sec, threshold, self.top_k)
            if (
                det.is_match
                and det.quality_passed
                and det.person_id
                and track.threshold_match_count >= min_confirmations
                and not track.alert_triggered
            ):
                track.alert_triggered = True
                alert = TrackAlert(
                    track_id=track.track_id,
                    person_id=det.person_id,
                    person_name=det.person_name,
                    similarity=det.similarity,
                    max_similarity=track.max_similarity,
                    mean_similarity=track.mean_similarity,
                    threshold=threshold,
                    frame_idx=frame_idx,
                    timestamp_sec=timestamp_sec,
                    bbox=det.bbox,
                    detection_count=track.detection_count,
                    source_name=source_name,
                    threshold_match_count=track.threshold_match_count,
                    match_frames=track.threshold_match_count,
                    alert_triggered=True,
                )
                new_alerts.append(alert)
            surviving_tracks.append(track)
            assigned_tracks[det_idx] = track

        self.active_tracks = surviving_tracks
        self.max_simultaneous_tracks = max(self.max_simultaneous_tracks, len(self.active_tracks))
        return [t for t in assigned_tracks if t is not None], new_alerts

    def _match_iou(self, detections: List[DetectionItem]) -> Tuple[List[int], List[int]]:
        """
        Greedy bipartite matching between active tracks and detections based on IoU overlap.

        Returns:
            Tuple of (matched_track_indices, matched_detection_indices).
        """
        if not self.active_tracks or not detections:
            return [], []

        # Build candidate pairs with IoU >= iou_threshold
        candidates = []
        for t_idx, track in enumerate(self.active_tracks):
            for d_idx, det in enumerate(detections):
                iou = compute_iou(track.bbox, det.bbox)
                if iou >= self.iou_threshold:
                    candidates.append((iou, t_idx, d_idx))

        # Sort candidates descending by IoU
        candidates.sort(key=lambda x: x[0], reverse=True)

        matched_tracks: List[int] = []
        matched_dets: List[int] = []
        used_tracks = set()
        used_dets = set()

        for iou, t_idx, d_idx in candidates:
            if t_idx not in used_tracks and d_idx not in used_dets:
                used_tracks.add(t_idx)
                used_dets.add(d_idx)
                matched_tracks.append(t_idx)
                matched_dets.append(d_idx)

        return matched_tracks, matched_dets

    def _create_new_track(
        self,
        det: DetectionItem,
        frame_idx: int,
        timestamp_sec: float,
        threshold: float,
        top_k: int = 3,
    ) -> Track:
        """Instantiate a new Track object with a stable ID."""
        track_id = f"TRACK-{self.next_track_num:04d}"
        self.next_track_num += 1

        sim = det.similarity
        is_threshold_match = 1 if (det.is_match and det.quality_passed and sim >= threshold) else 0
        matched_id = det.person_id if (det.is_match and det.person_id) else None
        matched_name = det.person_name if (det.is_match and det.person_id) else "Unknown"

        track = Track(
            track_id=track_id,
            bbox=det.bbox,
            first_frame=frame_idx,
            last_frame=frame_idx,
            first_timestamp=timestamp_sec,
            last_timestamp=timestamp_sec,
            detection_count=1,
            matched_person_id=matched_id,
            matched_person_name=matched_name,
            similarity_scores=[sim],
            detection_history=[{
                "frame_idx": frame_idx,
                "timestamp_sec": timestamp_sec,
                "bbox": det.bbox,
                "similarity": sim,
                "person_id": det.person_id,
                "person_name": det.person_name,
                "is_match": det.is_match,
                "quality_passed": det.quality_passed,
            }],
            max_similarity=sim,
            mean_similarity=sim,
            top_k_mean_similarity=sim,
            threshold_match_count=is_threshold_match,
            alert_triggered=False,
            missed_frames=0,
            active=True,
            best_match_frame=frame_idx,
            best_match_timestamp=timestamp_sec,
            best_match_bbox=det.bbox,
        )
        return track

    def get_all_tracks(self) -> List[Track]:
        """Return all tracks (active and terminated)."""
        return self.active_tracks + self.terminated_tracks

    def finalize(self) -> List[Track]:
        """Finalize tracking by moving all remaining active tracks to terminated list."""
        for t in self.active_tracks:
            t.active = False
            self.terminated_tracks.append(t)
        self.active_tracks = []
        return self.terminated_tracks
