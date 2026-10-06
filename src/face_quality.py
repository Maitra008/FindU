"""
Face Quality Gate Module for Recognition Pipeline.
Evaluates detected face observations before accepting recognition results.

Evaluates:
- Minimum face pixel dimensions (width & height)
- SCRFD detection confidence score
- Focus / sharpness via Laplacian variance
- Illumination / luminance level

Provides structured diagnostics explaining pass/fail reasons.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np

from src.config import (
    DEFAULT_FACE_MIN_BLUR_VARIANCE,
    DEFAULT_FACE_MIN_CONFIDENCE,
    DEFAULT_FACE_MIN_SIZE,
    DEFAULT_QUALITY_GATE_ENABLED,
)

logger = logging.getLogger(__name__)


@dataclass
class FaceQualityResult:
    """Structured diagnostic result returned by the Face Quality Gate."""
    passed: bool
    face_width: int
    face_height: int
    min_dimension: int
    detection_confidence: float
    blur_score: float
    brightness: float
    rejection_reasons: List[str] = field(default_factory=list)

    @property
    def primary_rejection_reason(self) -> Optional[str]:
        """Return primary rejection reason string if failed, or None if passed."""
        if not self.passed and self.rejection_reasons:
            return self.rejection_reasons[0]
        return None

    @property
    def blur_variance(self) -> float:
        """Alias for blur_score."""
        return self.blur_score

    @property
    def confidence(self) -> float:
        """Alias for detection_confidence."""
        return self.detection_confidence

    def to_dict(self) -> dict:
        """Convert quality result to serializable dictionary."""
        return {
            "passed": self.passed,
            "face_width": self.face_width,
            "face_height": self.face_height,
            "min_dimension": self.min_dimension,
            "detection_confidence": round(self.detection_confidence, 4),
            "blur_score": round(self.blur_score, 2),
            "brightness": round(self.brightness, 2),
            "rejection_reasons": self.rejection_reasons,
        }


class FaceQualityGate:
    """
    Configurable Face Quality Gate.
    Filters out degraded, distant, or blurry face detections before ArcFace recognition.
    """

    def __init__(
        self,
        min_size: int = DEFAULT_FACE_MIN_SIZE,
        min_confidence: float = DEFAULT_FACE_MIN_CONFIDENCE,
        min_blur_variance: float = DEFAULT_FACE_MIN_BLUR_VARIANCE,
        min_brightness: float = 30.0,
        enabled: bool = DEFAULT_QUALITY_GATE_ENABLED,
    ):
        """
        Initialize FaceQualityGate with configurable thresholds.

        Args:
            min_size: Minimum face bounding box width/height in pixels (default: 70).
            min_confidence: Minimum SCRFD detector score (default: 0.70).
            min_blur_variance: Minimum Laplacian variance for focus sharpness (default: 25.0).
            min_brightness: Minimum average grayscale luminance (default: 30.0).
            enabled: Master switch to enable/disable quality filtering.
        """
        self.min_size = min_size
        self.min_confidence = min_confidence
        self.min_blur_variance = min_blur_variance
        self.min_brightness = min_brightness
        self.enabled = enabled

    @staticmethod
    def calculate_blur(crop: np.ndarray) -> float:
        """Calculate focus sharpness via variance of Laplacian."""
        if crop is None or crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def calculate_brightness(crop: np.ndarray) -> float:
        """Calculate average luminance across face crop."""
        if crop is None or crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        return float(np.mean(gray))

    def evaluate_face(
        self,
        face_or_frame: Any = None,
        frame_or_bbox: Any = None,
        confidence: Optional[float] = None,
        bbox: Optional[List[int]] = None,
        frame: Optional[np.ndarray] = None,
        face: Optional[Any] = None,
    ) -> FaceQualityResult:
        """
        Evaluate quality metrics for an individual detected face observation.
        Supports:
          - evaluate_face(face, frame)
          - evaluate_face(frame=frame, bbox=bbox, confidence=confidence)
          - evaluate_face(face=face, frame=frame)
          - evaluate_face(frame, bbox, confidence)

        Returns:
            FaceQualityResult detailing pass/fail status and exact metrics.
        """
        if face is not None or (hasattr(face_or_frame, "bbox") and hasattr(face_or_frame, "confidence")):
            target_face = face if face is not None else face_or_frame
            target_frame = frame if frame is not None else frame_or_bbox
            target_bbox = target_face.bbox
            target_confidence = float(target_face.confidence)
        else:
            target_frame = frame if frame is not None else face_or_frame
            target_bbox = bbox if bbox is not None else frame_or_bbox
            target_confidence = float(confidence if confidence is not None else 1.0)

        # Calculate bounding box dimensions
        x1, y1, x2, y2 = target_bbox[:4] if target_bbox is not None else [0, 0, 0, 0]
        width = max(0, x2 - x1)
        height = max(0, y2 - y1)
        min_dim = min(width, height)

        rejection_reasons = []

        # If gate is disabled, pass automatically with measured metrics
        if not self.enabled:
            blur = 0.0
            brightness = 0.0
            if target_frame is not None and target_frame.size > 0 and width > 0 and height > 0:
                h, w = target_frame.shape[:2]
                crop = target_frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                blur = self.calculate_blur(crop)
                brightness = self.calculate_brightness(crop)

            return FaceQualityResult(
                passed=True,
                face_width=width,
                face_height=height,
                min_dimension=min_dim,
                detection_confidence=target_confidence,
                blur_score=blur,
                brightness=brightness,
                rejection_reasons=[],
            )

        # 1. Minimum Face Dimension Check
        if min_dim < self.min_size:
            rejection_reasons.append(
                f"rejected: face too small ({min_dim}px < {self.min_size}px minimum)"
            )

        # 2. Detector Confidence Check
        if target_confidence < self.min_confidence:
            rejection_reasons.append(
                f"rejected: low detector confidence ({target_confidence:.3f} < {self.min_confidence:.2f})"
            )

        # Extract face crop for image-level quality checks
        blur = 0.0
        brightness = 0.0
        if target_frame is not None and target_frame.size > 0 and width > 0 and height > 0:
            h, w = target_frame.shape[:2]
            crop = target_frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            
            if crop.size > 0:
                blur = self.calculate_blur(crop)
                brightness = self.calculate_brightness(crop)

                # 3. Blur Sharpness Check
                if blur < self.min_blur_variance:
                    rejection_reasons.append(
                        f"rejected: excessive blur ({blur:.1f} < {self.min_blur_variance:.1f} variance)"
                    )

                # 4. Illumination Check
                if brightness < self.min_brightness:
                    rejection_reasons.append(
                        f"rejected: severe underexposure ({brightness:.1f} < {self.min_brightness:.1f} luminance)"
                    )
        else:
            rejection_reasons.append("rejected: invalid or empty face crop")

        passed = (len(rejection_reasons) == 0)

        return FaceQualityResult(
            passed=passed,
            face_width=width,
            face_height=height,
            min_dimension=min_dim,
            detection_confidence=target_confidence,
            blur_score=blur,
            brightness=brightness,
            rejection_reasons=rejection_reasons,
        )


_default_quality_gate: Optional[FaceQualityGate] = None


def get_face_quality_gate(
    min_size: int = DEFAULT_FACE_MIN_SIZE,
    min_confidence: float = DEFAULT_FACE_MIN_CONFIDENCE,
    min_blur_variance: float = DEFAULT_FACE_MIN_BLUR_VARIANCE,
    enabled: bool = DEFAULT_QUALITY_GATE_ENABLED,
) -> FaceQualityGate:
    """Singleton getter for the centralized FaceQualityGate."""
    global _default_quality_gate
    if _default_quality_gate is None:
        _default_quality_gate = FaceQualityGate(
            min_size=min_size,
            min_confidence=min_confidence,
            min_blur_variance=min_blur_variance,
            enabled=enabled,
        )
    return _default_quality_gate
