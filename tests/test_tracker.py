"""
Unit and Integration Tests for Temporal Face Tracking and Score Aggregation Module (Part 3).
Verifies:
- Test A: Tracker unit tests & IoU boundary calculations.
- Test B: Single person track persistence across motion and time (one track ID).
- Test C: Significant movement association between sampled frames.
- Test D: Temporary disappearance survival (within max_missed_frames <= 5).
- Test E: Permanent disappearance track termination (exceeding max_missed_frames > 5).
- Test F: Multi-person distinct track assignment and isolation (two distinct track IDs).
- Test G: Repeated matching frames guarantee exactly ONE alert per track.
- Test H: Unknown person tracking without alert trigger.
- Test I: Similarity below 0.89 does NOT trigger potential-match alert.
- Test J: Similarity >= 0.89 triggers potential-match alert.
- Test K: CLI & VideoRecognizer integration with tracking parameters.
"""

from pathlib import Path
import cv2
import numpy as np
import pytest

from src.config import RecognitionConfig
from src.tracker import (
    DetectionItem,
    FaceTracker,
    Track,
    TrackAlert,
    compute_iou,
)
from src.recognition import VideoRecognizer, FaceRecognitionEvent
from src.index import FaceIndex


def test_iou_calculation_boundaries():
    """Test A: Verify compute_iou with exact overlaps, partial overlaps, and non-overlapping boxes."""
    # Exact overlap
    b1 = [100, 100, 200, 200]
    b2 = [100, 100, 200, 200]
    assert compute_iou(b1, b2) == pytest.approx(1.0, abs=1e-4)

    # 50% horizontal shift -> IoU = (50 * 100) / (100 * 100 + 100 * 100 - 50 * 100) = 5000 / 15000 = 1/3 ~ 0.3333
    b3 = [150, 100, 250, 200]
    assert compute_iou(b1, b3) == pytest.approx(1.0 / 3.0, abs=1e-3)

    # No overlap (disjoint)
    b4 = [300, 300, 400, 400]
    assert compute_iou(b1, b4) == 0.0

    # Degenerate zero-area boxes
    b5 = [100, 100, 100, 100]
    assert compute_iou(b1, b5) == 0.0


def test_single_moving_person_single_track_id():
    """Test B: One moving person across consecutive frames produces one persistent TRACK ID."""
    tracker = FaceTracker(iou_threshold=0.30, max_missed_frames=5)

    # Simulate person walking across screen from x=100 to x=180 in steps of 8px per frame
    for f_idx in range(10):
        x = 100 + f_idx * 8
        det = DetectionItem(
            bbox=[x, 100, x + 100, 200],
            confidence=0.95,
            similarity=0.92,
            person_id="person_01",
            person_name="Person 01",
            is_match=True,
        )
        assigned_tracks, alerts = tracker.update([det], frame_idx=f_idx, timestamp_sec=f_idx * 0.5, threshold=0.89)

        assert len(assigned_tracks) == 1
        assert assigned_tracks[0].track_id == "TRACK-0001"
        assert assigned_tracks[0].detection_count == f_idx + 1

    all_tracks = tracker.finalize()
    assert len(all_tracks) == 1
    assert all_tracks[0].track_id == "TRACK-0001"
    assert all_tracks[0].detection_count == 10
    assert all_tracks[0].first_frame == 0
    assert all_tracks[0].last_frame == 9


def test_movement_integration_significant_motion():
    """Test C: Significant movement between sampled frames maintains the same track when IoU >= threshold."""
    tracker = FaceTracker(iou_threshold=0.30, max_missed_frames=5)

    # Frame 0: Box [100, 100, 200, 200]
    det0 = DetectionItem(
        bbox=[100, 100, 200, 200],
        confidence=0.95,
        similarity=0.93,
        person_id="person_01",
        person_name="Person 01",
        is_match=True,
    )
    assigned, _ = tracker.update([det0], frame_idx=0, timestamp_sec=0.0, threshold=0.89)
    assert assigned[0].track_id == "TRACK-0001"

    # Frame 1: Shifts right by 40px -> IoU = (60*100)/(100*100+100*100-60*100) = 6000/14000 = 0.428 >= 0.30
    det1 = DetectionItem(
        bbox=[140, 100, 240, 200],
        confidence=0.95,
        similarity=0.94,
        person_id="person_01",
        person_name="Person 01",
        is_match=True,
    )
    assigned, _ = tracker.update([det1], frame_idx=1, timestamp_sec=0.5, threshold=0.89)
    assert assigned[0].track_id == "TRACK-0001"
    assert assigned[0].detection_count == 2


def test_person_disappears_briefly_survives():
    """Test D: Person disappears for <= 5 frames (within max_missed_frames=5), track survives with same ID."""
    tracker = FaceTracker(iou_threshold=0.30, max_missed_frames=5)

    # Frames 0-2: Visible
    for f in range(3):
        det = DetectionItem(
            bbox=[100, 100, 200, 200],
            confidence=0.92,
            similarity=0.91,
            person_id="person_01",
            person_name="Person 01",
            is_match=True,
        )
        tracker.update([det], frame_idx=f, timestamp_sec=f * 0.5, threshold=0.89)

    # Frames 3-5: Disappears (0 detections) for 3 frames (<= 5)
    for f in range(3, 6):
        assigned, _ = tracker.update([], frame_idx=f, timestamp_sec=f * 0.5, threshold=0.89)
        assert len(tracker.active_tracks) == 1
        assert tracker.active_tracks[0].missed_frames == f - 2

    # Frame 6: Reappears at same location
    det_reappear = DetectionItem(
        bbox=[105, 100, 205, 200],
        confidence=0.94,
        similarity=0.93,
        person_id="person_01",
        person_name="Person 01",
        is_match=True,
    )
    assigned, _ = tracker.update([det_reappear], frame_idx=6, timestamp_sec=3.0, threshold=0.89)

    assert len(assigned) == 1
    assert assigned[0].track_id == "TRACK-0001"
    assert assigned[0].missed_frames == 0
    assert assigned[0].detection_count == 4


def test_person_disappears_long_terminates():
    """Test E: Person disappears for > 5 frames (exceeding max_missed_frames=5), track terminates."""
    tracker = FaceTracker(iou_threshold=0.30, max_missed_frames=5)

    # Frame 0: Appears
    det = DetectionItem(
        bbox=[100, 100, 200, 200],
        confidence=0.90,
        similarity=0.92,
        person_id="person_01",
        person_name="Person 01",
        is_match=True,
    )
    tracker.update([det], frame_idx=0, timestamp_sec=0.0, threshold=0.89)

    # Frames 1-5: 5 missed frames (still active)
    for f in range(1, 6):
        tracker.update([], frame_idx=f, timestamp_sec=f * 0.5, threshold=0.89)
    assert len(tracker.active_tracks) == 1

    # Frame 6: 6th missed frame (> 5) -> track terminates
    tracker.update([], frame_idx=6, timestamp_sec=3.0, threshold=0.89)
    assert len(tracker.active_tracks) == 0
    assert len(tracker.terminated_tracks) == 1
    assert tracker.terminated_tracks[0].track_id == "TRACK-0001"
    assert not tracker.terminated_tracks[0].active

    # Frame 7: Reappears after termination -> gets new track ID TRACK-0002
    assigned, _ = tracker.update([det], frame_idx=7, timestamp_sec=3.5, threshold=0.89)
    assert len(assigned) == 1
    assert assigned[0].track_id == "TRACK-0002"


def test_two_people_distinct_track_ids():
    """Test F: Two people in the frame receive distinct, persistent track IDs."""
    tracker = FaceTracker(iou_threshold=0.30, max_missed_frames=5)

    # Frame 0: Two faces at distinct locations
    det_a = DetectionItem(
        bbox=[50, 50, 150, 150],
        confidence=0.90,
        similarity=0.93,
        person_id="person_01",
        person_name="Person 01",
        is_match=True,
    )
    det_b = DetectionItem(
        bbox=[400, 100, 500, 200],
        confidence=0.88,
        similarity=0.91,
        person_id="person_02",
        person_name="Person 02",
        is_match=True,
    )

    assigned, _ = tracker.update([det_a, det_b], frame_idx=0, timestamp_sec=0.0, threshold=0.89)
    assert len(assigned) == 2
    track_ids = {t.track_id for t in assigned}
    assert track_ids == {"TRACK-0001", "TRACK-0002"}

    # Frame 1: Both move slightly
    det_a2 = DetectionItem(
        bbox=[55, 52, 155, 152],
        confidence=0.92,
        similarity=0.94,
        person_id="person_01",
        person_name="Person 01",
        is_match=True,
    )
    det_b2 = DetectionItem(
        bbox=[405, 102, 505, 202],
        confidence=0.89,
        similarity=0.90,
        person_id="person_02",
        person_name="Person 02",
        is_match=True,
    )

    assigned2, _ = tracker.update([det_a2, det_b2], frame_idx=1, timestamp_sec=0.5, threshold=0.89)
    assert len(assigned2) == 2
    assert {t.track_id for t in assigned2} == {"TRACK-0001", "TRACK-0002"}


def test_one_matched_person_exactly_one_alert():
    """Test G: Track exceeding threshold across multiple frames triggers exactly ONE alert."""
    tracker = FaceTracker(iou_threshold=0.30, max_missed_frames=5)
    all_dispatched_alerts = []

    # 10 consecutive frames, all matching person_01 above threshold 0.89
    for f in range(10):
        det = DetectionItem(
            bbox=[100, 100, 200, 200],
            confidence=0.95,
            similarity=0.91 + (f * 0.005),  # 0.91 to 0.955
            person_id="person_01",
            person_name="Person 01",
            is_match=True,
        )
        _, alerts = tracker.update([det], frame_idx=f, timestamp_sec=f * 0.5, threshold=0.89)
        all_dispatched_alerts.extend(alerts)

    # Crucial guarantee: Exactly ONE alert dispatched for TRACK-0001
    assert len(all_dispatched_alerts) == 1
    alert = all_dispatched_alerts[0]
    assert alert.track_id == "TRACK-0001"
    assert alert.person_id == "person_01"
    assert alert.person_name == "Person 01"
    assert alert.frame_idx == 0
    assert alert.timestamp_sec == 0.0
    assert alert.alert_triggered is True


def test_unknown_person_track_exists_no_alert():
    """Test H: Unknown person creates a track but never triggers a potential-match alert."""
    tracker = FaceTracker(iou_threshold=0.30, max_missed_frames=5)
    all_alerts = []

    for f in range(8):
        det = DetectionItem(
            bbox=[100, 100, 200, 200],
            confidence=0.85,
            similarity=0.15,
            person_id=None,
            person_name="Unknown",
            is_match=False,
        )
        assigned, alerts = tracker.update([det], frame_idx=f, timestamp_sec=f * 0.5, threshold=0.89)
        all_alerts.extend(alerts)

    assert len(all_alerts) == 0
    all_tracks = tracker.finalize()
    assert len(all_tracks) == 1
    assert all_tracks[0].track_id == "TRACK-0001"
    assert not all_tracks[0].alert_triggered
    assert all_tracks[0].matched_person_name == "Unknown"
    assert all_tracks[0].threshold_match_count == 0


def test_threshold_below_089_no_alert():
    """Test I: Similarity below 0.89 (e.g. 0.85) must NOT trigger a potential-match alert."""
    tracker = FaceTracker(iou_threshold=0.30, max_missed_frames=5)

    det = DetectionItem(
        bbox=[100, 100, 200, 200],
        confidence=0.95,
        similarity=0.85,
        person_id="person_01",
        person_name="Person 01",
        is_match=False,
    )

    assigned, alerts = tracker.update([det], frame_idx=0, timestamp_sec=0.0, threshold=0.89)
    assert len(alerts) == 0
    assert assigned[0].alert_triggered is False
    assert assigned[0].threshold_match_count == 0


def test_threshold_above_089_triggers_alert():
    """Test J: Similarity >= 0.89 (e.g. 0.93) triggers a potential-match alert."""
    tracker = FaceTracker(iou_threshold=0.30, max_missed_frames=5)

    det = DetectionItem(
        bbox=[100, 100, 200, 200],
        confidence=0.95,
        similarity=0.93,
        person_id="person_01",
        person_name="Person 01",
        is_match=True,
    )

    assigned, alerts = tracker.update([det], frame_idx=0, timestamp_sec=0.0, threshold=0.89)
    assert len(alerts) == 1
    assert alerts[0].track_id == "TRACK-0001"
    assert alerts[0].person_id == "person_01"
    assert alerts[0].similarity == pytest.approx(0.93)
    assert alerts[0].alert_triggered is True
    assert assigned[0].alert_triggered is True


def test_score_aggregation_and_top_k():
    """Verify max_similarity, mean_similarity, and top_k_mean_similarity calculations."""
    track = Track(
        track_id="TRACK-0001",
        bbox=[100, 100, 200, 200],
        first_frame=0,
        last_frame=0,
        first_timestamp=0.0,
        last_timestamp=0.0,
        similarity_scores=[0.80],
        max_similarity=0.80,
        mean_similarity=0.80,
        top_k_mean_similarity=0.80,
    )

    scores = [0.90, 0.70, 0.95, 0.85]
    for i, s in enumerate(scores, start=1):
        det = DetectionItem(
            bbox=[100, 100, 200, 200],
            confidence=0.9,
            similarity=s,
            person_id="person_01",
            person_name="Person 01",
            is_match=True,
        )
        track.update(det, frame_idx=i, timestamp_sec=i * 0.5, threshold=0.89, top_k=3)

    all_scores = [0.80, 0.90, 0.70, 0.95, 0.85]
    expected_max = max(all_scores)  # 0.95
    expected_mean = sum(all_scores) / len(all_scores)  # 0.84
    # Top 3 scores: 0.95, 0.90, 0.85 -> mean = 0.90
    expected_top_3_mean = (0.95 + 0.90 + 0.85) / 3.0

    assert track.max_similarity == pytest.approx(expected_max, abs=1e-4)
    assert track.mean_similarity == pytest.approx(expected_mean, abs=1e-4)
    assert track.top_k_mean_similarity == pytest.approx(expected_top_3_mean, abs=1e-4)
    assert track.threshold_match_count == 2  # 0.90 and 0.95 >= 0.89


def test_cli_and_recognizer_integration_defaults():
    """Test K: Verify VideoRecognizer uses default tracking parameters and threshold 0.89."""
    config = RecognitionConfig()
    assert config.threshold == pytest.approx(0.89)
    assert config.iou_threshold == pytest.approx(0.30)
    assert config.max_missed_frames == 5
    assert config.track_top_k == 3

    recognizer = VideoRecognizer()
    assert recognizer.threshold == pytest.approx(0.89)
    assert recognizer.iou_threshold == pytest.approx(0.30)
    assert recognizer.max_missed_frames == 5
    assert recognizer.track_top_k == 3
