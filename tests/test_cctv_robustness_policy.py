"""
Unit and regression tests for CCTV-Robust Recognition Policy:
1. Face Quality Gate validation (size, confidence, blur, illumination).
2. FAISS similarity threshold operating at calibrated 0.55 cutoff.
3. Temporal confirmation mechanics (>= 2 match confirmations required before alert).
4. Single-alert-per-track invariant preservation.
5. Independent track state isolation.
"""

import numpy as np
import pytest

from src.config import (
    DEFAULT_FACE_MIN_BLUR_VARIANCE,
    DEFAULT_FACE_MIN_CONFIDENCE,
    DEFAULT_FACE_MIN_SIZE,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_TEMPORAL_CONFIRMATIONS,
)
from src.face_engine import DetectedFace, normalize_embedding
from src.face_quality import FaceQualityGate, FaceQualityResult
from src.index import FaceIndex
from src.tracker import DetectionItem, FaceTracker, TrackAlert


# =====================================================================
# 1. Face Quality Gate Tests
# =====================================================================

class TestFaceQualityGate:
    """Test suite for FaceQualityGate rejection rules and edge conditions."""

    @pytest.fixture
    def quality_gate(self):
        return FaceQualityGate(
            min_size=70,
            min_confidence=0.70,
            min_blur_variance=25.0,
            min_brightness=30.0,
            enabled=True,
        )

    def _create_synthetic_frame(self, h=480, w=640, is_blurry=False, is_dark=False):
        """Helper to create synthetic test frames with controlled sharpness/brightness."""
        if is_dark:
            return np.full((h, w, 3), 10, dtype=np.uint8)
        if is_blurry:
            # Uniform image has Laplacian variance of 0.0
            return np.full((h, w, 3), 128, dtype=np.uint8)
        # Sharp textured image (checkerboard / high variance)
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[::4, ::4] = 255
        frame[1::4, 1::4] = 200
        frame[2::4, 2::4] = 100
        return frame

    def test_gate_rejects_undersized_face(self, quality_gate):
        """Faces smaller than 70px on either dimension must be rejected."""
        frame = self._create_synthetic_frame()
        face = DetectedFace(
            bbox=[100, 100, 150, 160],  # w=50, h=60 (< 70px)
            confidence=0.95,
            raw_embedding=np.random.randn(512).astype(np.float32),
            normalized_embedding=normalize_embedding(np.random.randn(512).astype(np.float32)),
        )
        res = quality_gate.evaluate_face(face, frame)
        assert not res.passed
        assert any("small" in r.lower() or "size" in r.lower() for r in res.rejection_reasons)
        assert res.min_dimension == 50

    def test_gate_rejects_low_confidence(self, quality_gate):
        """SCRFD detections with confidence < 0.70 must be rejected."""
        frame = self._create_synthetic_frame()
        face = DetectedFace(
            bbox=[100, 100, 200, 200],  # w=100, h=100
            confidence=0.65,  # < 0.70
            raw_embedding=np.random.randn(512).astype(np.float32),
            normalized_embedding=normalize_embedding(np.random.randn(512).astype(np.float32)),
        )
        res = quality_gate.evaluate_face(face, frame)
        assert not res.passed
        assert any("confidence" in r.lower() for r in res.rejection_reasons)

    def test_gate_rejects_blurry_face(self, quality_gate):
        """Faces with Laplacian variance < 25.0 must be rejected."""
        frame = self._create_synthetic_frame(is_blurry=True)
        face = DetectedFace(
            bbox=[100, 100, 200, 200],  # w=100, h=100
            confidence=0.95,
            raw_embedding=np.random.randn(512).astype(np.float32),
            normalized_embedding=normalize_embedding(np.random.randn(512).astype(np.float32)),
        )
        res = quality_gate.evaluate_face(face, frame)
        assert not res.passed
        assert any("blur" in r.lower() for r in res.rejection_reasons)
        assert res.blur_variance < 25.0

    def test_gate_rejects_dark_face(self, quality_gate):
        """Severely underexposed faces (brightness < 30) must be rejected."""
        frame = self._create_synthetic_frame(is_dark=True)
        face = DetectedFace(
            bbox=[100, 100, 200, 200],
            confidence=0.95,
            raw_embedding=np.random.randn(512).astype(np.float32),
            normalized_embedding=normalize_embedding(np.random.randn(512).astype(np.float32)),
        )
        res = quality_gate.evaluate_face(face, frame)
        assert not res.passed
        assert any("underexposure" in r.lower() or "brightness" in r.lower() or "dark" in r.lower() for r in res.rejection_reasons)

    def test_gate_passes_high_quality_face(self, quality_gate):
        """High-resolution, sharp, high-confidence face must pass all checks."""
        frame = self._create_synthetic_frame(is_blurry=False)
        face = DetectedFace(
            bbox=[100, 100, 220, 220],  # w=120, h=120
            confidence=0.98,
            raw_embedding=np.random.randn(512).astype(np.float32),
            normalized_embedding=normalize_embedding(np.random.randn(512).astype(np.float32)),
        )
        res = quality_gate.evaluate_face(face, frame)
        assert res.passed
        assert len(res.rejection_reasons) == 0
        assert res.min_dimension >= 70
        assert res.confidence >= 0.70
        assert res.blur_variance >= 25.0
        assert res.brightness >= 30.0


# =====================================================================
# 2. Similarity Threshold Calibration Tests (0.55)
# =====================================================================

class TestThresholdCalibration:
    """Validate that FAISS searching respects calibrated 0.55 cutoff."""

    @pytest.fixture
    def populated_index(self):
        index = FaceIndex()
        # Create a unit base vector
        v = np.zeros(512, dtype=np.float32)
        v[0] = 1.0
        index.add_identity(embedding=v, person_id="p_target", name="Target Subject")
        return index, v

    def test_default_config_threshold_is_0_55(self):
        assert DEFAULT_SIMILARITY_THRESHOLD == 0.55
        assert DEFAULT_TEMPORAL_CONFIRMATIONS == 2
        assert DEFAULT_FACE_MIN_SIZE == 70
        assert DEFAULT_FACE_MIN_CONFIDENCE == 0.70
        assert DEFAULT_FACE_MIN_BLUR_VARIANCE == 3.0

    def test_threshold_boundary_0_54_rejected(self, populated_index):
        index, v = populated_index
        q = np.zeros(512, dtype=np.float32)
        q[0] = 0.54
        q[1] = float(np.sqrt(1.0 - 0.54**2))

        matches = index.search(q, threshold=0.55)
        assert len(matches) == 1
        assert not matches[0].is_match  # Below threshold 0.55
        assert matches[0].similarity < 0.55

    def test_threshold_boundary_0_55_accepted(self, populated_index):
        index, v = populated_index
        q = np.zeros(512, dtype=np.float32)
        q[0] = 0.5501
        q[1] = float(np.sqrt(1.0 - 0.5501**2))

        matches = index.search(q, threshold=0.55)
        assert len(matches) == 1
        assert matches[0].is_match
        assert matches[0].person_id == "p_target"
        assert matches[0].similarity >= 0.55

    def test_threshold_boundary_0_56_accepted(self, populated_index):
        index, v = populated_index
        q = np.zeros(512, dtype=np.float32)
        q[0] = 0.56
        q[1] = float(np.sqrt(1.0 - 0.56**2))

        matches = index.search(q, threshold=0.55)
        assert len(matches) == 1
        assert matches[0].is_match
        assert matches[0].person_id == "p_target"
        assert matches[0].similarity >= 0.56


# =====================================================================
# 3. Temporal Confirmation & Single-Alert Invariant Tests
# =====================================================================

class TestTemporalConfirmationAndAlerts:
    """Validate 2-observation temporal confirmation and duplicate prevention."""

    @pytest.fixture
    def tracker(self):
        return FaceTracker(
            iou_threshold=0.30,
            max_missed_frames=5,
            top_k=3,
            min_confirmations=2,
        )

    def _make_det(self, bbox, sim, is_match, quality_passed=True, name="Target", pid="p_target"):
        return DetectionItem(
            bbox=bbox,
            confidence=0.95,
            similarity=sim,
            person_id=pid if is_match else None,
            person_name=name if is_match else "Unknown",
            is_match=is_match,
            embedding=np.ones(512, dtype=np.float32),
            face_idx=0,
            quality_passed=quality_passed,
        )

    def test_single_observation_does_not_trigger_alert(self, tracker):
        """Frame 1: 1 match detection must NOT trigger an alert yet."""
        det = self._make_det([100, 100, 200, 200], sim=0.60, is_match=True, quality_passed=True)
        assigned, alerts = tracker.update([det], frame_idx=0, timestamp_sec=0.0, threshold=0.55)

        assert len(assigned) == 1
        assert assigned[0].threshold_match_count == 1
        assert not assigned[0].alert_triggered
        assert len(alerts) == 0  # No alert on 1st observation

    def test_second_observation_triggers_exactly_one_alert(self, tracker):
        """Frame 2: 2nd consecutive match detection on same track MUST trigger alert."""
        det1 = self._make_det([100, 100, 200, 200], sim=0.60, is_match=True, quality_passed=True)
        tracker.update([det1], frame_idx=0, timestamp_sec=0.0, threshold=0.55)

        det2 = self._make_det([105, 105, 205, 205], sim=0.62, is_match=True, quality_passed=True)
        assigned, alerts = tracker.update([det2], frame_idx=1, timestamp_sec=0.5, threshold=0.55)

        assert len(assigned) == 1
        assert assigned[0].threshold_match_count == 2
        assert assigned[0].alert_triggered
        assert len(alerts) == 1
        assert alerts[0].person_id == "p_target"
        assert alerts[0].track_id == assigned[0].track_id

    def test_subsequent_observations_do_not_produce_duplicate_alerts(self, tracker):
        """Frames 3, 4, 5: Subsequent matches on alerted track must emit 0 new alerts."""
        det1 = self._make_det([100, 100, 200, 200], sim=0.60, is_match=True)
        tracker.update([det1], frame_idx=0, timestamp_sec=0.0, threshold=0.55)

        det2 = self._make_det([102, 102, 202, 202], sim=0.62, is_match=True)
        _, alerts2 = tracker.update([det2], frame_idx=1, timestamp_sec=0.5, threshold=0.55)
        assert len(alerts2) == 1

        # Frame 3
        det3 = self._make_det([104, 104, 204, 204], sim=0.65, is_match=True)
        assigned3, alerts3 = tracker.update([det3], frame_idx=2, timestamp_sec=1.0, threshold=0.55)
        assert len(alerts3) == 0
        assert assigned3[0].threshold_match_count == 3
        assert assigned3[0].alert_triggered

        # Frame 4
        det4 = self._make_det([106, 106, 206, 206], sim=0.58, is_match=True)
        assigned4, alerts4 = tracker.update([det4], frame_idx=3, timestamp_sec=1.5, threshold=0.55)
        assert len(alerts4) == 0
        assert assigned4[0].threshold_match_count == 4

    def test_quality_rejected_frame_does_not_increment_match_count(self, tracker):
        """A blurry/small frame on an active track must not count toward temporal confirmation."""
        # Frame 1: Valid high quality match
        det1 = self._make_det([100, 100, 200, 200], sim=0.60, is_match=True, quality_passed=True)
        tracker.update([det1], frame_idx=0, timestamp_sec=0.0, threshold=0.55)

        # Frame 2: Low quality / rejected face (e.g. motion blur)
        det2 = self._make_det([102, 102, 202, 202], sim=0.60, is_match=False, quality_passed=False)
        assigned2, alerts2 = tracker.update([det2], frame_idx=1, timestamp_sec=0.5, threshold=0.55)

        assert len(alerts2) == 0
        assert assigned2[0].threshold_match_count == 1  # Still 1, NOT incremented
        assert not assigned2[0].alert_triggered

        # Frame 3: Valid high quality match -> Now reaches 2 confirmations
        det3 = self._make_det([104, 104, 204, 204], sim=0.61, is_match=True, quality_passed=True)
        assigned3, alerts3 = tracker.update([det3], frame_idx=2, timestamp_sec=1.0, threshold=0.55)

        assert len(alerts3) == 1
        assert assigned3[0].threshold_match_count == 2
        assert assigned3[0].alert_triggered

    def test_independent_tracks_maintain_separate_confirmation_counters(self, tracker):
        """Two distinct spatial tracks must confirm and alert independently."""
        # Track A: (100, 100) -> Person A
        # Track B: (400, 400) -> Person B
        det_a1 = self._make_det([100, 100, 200, 200], sim=0.60, is_match=True, pid="p_A", name="Person A")
        det_b1 = self._make_det([400, 400, 500, 500], sim=0.70, is_match=True, pid="p_B", name="Person B")
        assigned1, alerts1 = tracker.update([det_a1, det_b1], frame_idx=0, timestamp_sec=0.0, threshold=0.55)
        assert len(alerts1) == 0
        assert len(assigned1) == 2

        # Frame 2: Only Track A appears again
        det_a2 = self._make_det([105, 105, 205, 205], sim=0.62, is_match=True, pid="p_A", name="Person A")
        assigned2, alerts2 = tracker.update([det_a2], frame_idx=1, timestamp_sec=0.5, threshold=0.55)
        assert len(alerts2) == 1
        assert alerts2[0].person_id == "p_A"

        # Frame 3: Track B appears again -> reaches 2 confirmations and alerts
        det_b2 = self._make_det([405, 405, 505, 505], sim=0.72, is_match=True, pid="p_B", name="Person B")
        assigned3, alerts3 = tracker.update([det_b2], frame_idx=2, timestamp_sec=1.0, threshold=0.55)
        assert len(alerts3) == 1
        assert alerts3[0].person_id == "p_B"

    def test_intermittent_quality_sequence_valid_rejected_valid_valid(self, tracker):
        """Intermittent quality: valid -> rejected -> valid -> valid.
        Alert MUST fire strictly on the 2nd valid observation (Frame 3), not on Frame 1 or 2,
        and must not fire again on Frame 4.
        """
        # Frame 1: Valid high quality match
        det1 = self._make_det([100, 100, 200, 200], sim=0.60, is_match=True, quality_passed=True)
        assigned1, alerts1 = tracker.update([det1], frame_idx=0, timestamp_sec=0.0, threshold=0.55)
        assert len(alerts1) == 0
        assert assigned1[0].threshold_match_count == 1
        assert not assigned1[0].alert_triggered

        # Frame 2: Rejected low quality (e.g. motion blur / low confidence)
        det2 = self._make_det([102, 102, 202, 202], sim=0.60, is_match=False, quality_passed=False)
        assigned2, alerts2 = tracker.update([det2], frame_idx=1, timestamp_sec=0.5, threshold=0.55)
        assert len(alerts2) == 0
        assert assigned2[0].threshold_match_count == 1
        assert not assigned2[0].alert_triggered

        # Frame 3: Valid high quality match -> Reaches 2 confirmations -> Fires 1 alert
        det3 = self._make_det([104, 104, 204, 204], sim=0.65, is_match=True, quality_passed=True)
        assigned3, alerts3 = tracker.update([det3], frame_idx=2, timestamp_sec=1.0, threshold=0.55)
        assert len(alerts3) == 1
        assert assigned3[0].threshold_match_count == 2
        assert assigned3[0].alert_triggered

        # Frame 4: Valid high quality match -> Already alerted -> 0 alerts
        det4 = self._make_det([106, 106, 206, 206], sim=0.70, is_match=True, quality_passed=True)
        assigned4, alerts4 = tracker.update([det4], frame_idx=3, timestamp_sec=1.5, threshold=0.55)
        assert len(alerts4) == 0
        assert assigned4[0].threshold_match_count == 3
        assert assigned4[0].alert_triggered

    def test_initial_frame_creation_does_not_alert_even_high_similarity(self, tracker):
        """Initial track creation must never trigger an immediate alert, even with 0.99 similarity."""
        det = self._make_det([100, 100, 200, 200], sim=0.99, is_match=True, quality_passed=True)
        assigned, alerts = tracker.update([det], frame_idx=0, timestamp_sec=0.0, threshold=0.55)
        assert len(alerts) == 0
        assert len(assigned) == 1
        assert assigned[0].threshold_match_count == 1
        assert not assigned[0].alert_triggered


# =====================================================================
# 4. Canonical Pipeline Quality Gate Integration Tests
# =====================================================================

class TestCanonicalPipelineQualityIntegration:
    """Validate that recognize_frame() passes detections through FaceQualityGate."""

    def test_pipeline_filters_low_quality_faces(self, monkeypatch):
        """Pipeline must mark low-quality faces as quality_passed=False and is_match=False."""
        from unittest.mock import MagicMock
        from src.pipeline import recognize_frame

        mock_engine = MagicMock()
        mock_index = MagicMock()
        mock_gate = MagicMock()
        tracker = FaceTracker(min_confirmations=2)

        # Mock 1 detected face (undersized)
        face = DetectedFace(
            bbox=[10, 10, 50, 50],  # 40x40 px
            confidence=0.95,
            raw_embedding=np.random.randn(512).astype(np.float32),
            normalized_embedding=normalize_embedding(np.random.randn(512).astype(np.float32)),
        )
        mock_engine.detect_and_embed.return_value = [face]
        mock_gate.evaluate_face.return_value = FaceQualityResult(
            passed=False,
            face_width=40,
            face_height=40,
            min_dimension=40,
            detection_confidence=0.95,
            blur_score=50.0,
            brightness=100.0,
            rejection_reasons=["face_too_small: 40px < 70px"],
        )

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = recognize_frame(
            frame=dummy_frame,
            frame_idx=0,
            timestamp_sec=0.0,
            camera_id="cam_test",
            tracker=tracker,
            face_engine=mock_engine,
            face_index=mock_index,
            face_quality_gate=mock_gate,
            threshold=0.55,
        )

        assert len(result.detection_items) == 1
        item = result.detection_items[0]
        assert not item.quality_passed
        assert not item.is_match
        assert item.rejection_reason == "face_too_small: 40px < 70px"
        # FaceIndex search should NOT even be called for rejected faces
        mock_index.search.assert_not_called()
        # No alerts triggered
        assert len(result.new_alerts) == 0
