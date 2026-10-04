"""
FaceEngine module for face detection and embedding extraction using InsightFace.
Wraps SCRFD (detection) and ArcFace (recognition) with centralized model management.
"""

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

warnings.filterwarnings("ignore", category=FutureWarning)

import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

logger = logging.getLogger(__name__)


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """
    Perform L2 normalization on a 1D or 2D feature vector.
    
    Args:
        embedding: Raw embedding array.
        
    Returns:
        L2-normalized float32 embedding.
    """
    embedding = np.asarray(embedding, dtype=np.float32)
    norm = np.linalg.norm(embedding, axis=-1, keepdims=True)
    norm = np.where(norm == 0, 1e-12, norm)
    return (embedding / norm).astype(np.float32)


@dataclass
class DetectedFace:
    """Represents a single detected face with bounding box, score, and ArcFace embedding."""
    bbox: List[int]  # [x1, y1, x2, y2] integer coordinates
    confidence: float
    raw_embedding: np.ndarray
    normalized_embedding: np.ndarray
    landmarks: Optional[np.ndarray] = None


class FaceEngine:
    """
    Centralized face detection and embedding extraction engine.
    Initializes InsightFace FaceAnalysis once and reuses it across frames.
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        providers: Optional[List[str]] = None,
        det_size: Tuple[int, int] = (640, 640),
        det_thresh: float = 0.50,
    ):
        """
        Initialize the InsightFace FaceAnalysis pipeline.

        Args:
            model_name: Model pack name (default: buffalo_l).
            providers: Execution providers list (default: ['CPUExecutionProvider']).
            det_size: SCRFD input resolution (default: (640, 640)).
            det_thresh: Minimum face detection confidence threshold.
        """
        self.model_name = model_name
        self.providers = providers or ["CPUExecutionProvider"]
        self.det_size = det_size
        self.det_thresh = det_thresh

        logger.info("Initializing FaceEngine with model='%s', providers=%s", self.model_name, self.providers)
        
        self.app = FaceAnalysis(
            name=self.model_name,
            providers=self.providers,
            allowed_modules=["detection", "recognition"],
        )
        self.app.prepare(ctx_id=0, det_size=self.det_size, det_thresh=self.det_thresh)
        logger.info("FaceEngine initialized successfully.")

    def detect_and_embed(self, frame: Optional[np.ndarray]) -> List[DetectedFace]:
        """
        Detect all faces in the provided frame and extract ArcFace embeddings.

        Args:
            frame: BGR image numpy array from OpenCV.

        Returns:
            List of DetectedFace instances, or empty list if no valid faces found.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            logger.warning("Invalid or empty frame provided to detect_and_embed.")
            return []

        try:
            # FaceAnalysis expect BGR format as read by cv2
            raw_faces = self.app.get(frame)
        except Exception as e:
            logger.error("Error during face detection/embedding: %s", e, exc_info=True)
            return []

        detected: List[DetectedFace] = []
        for face in raw_faces:
            if face.embedding is None:
                continue

            score = float(getattr(face, "det_score", 0.0))
            if score < self.det_thresh:
                continue

            bbox_raw = face.bbox.astype(int).tolist()
            raw_emb = np.asarray(face.embedding, dtype=np.float32)
            norm_emb = normalize_embedding(raw_emb)
            kps = getattr(face, "kps", None)

            detected.append(
                DetectedFace(
                    bbox=bbox_raw,
                    confidence=score,
                    raw_embedding=raw_emb,
                    normalized_embedding=norm_emb,
                    landmarks=kps,
                )
            )

        return detected

    @staticmethod
    def load_image(image_path: Union[str, Path]) -> Optional[np.ndarray]:
        """
        Load an image safely from disk using OpenCV.

        Args:
            image_path: Path to the image file.

        Returns:
            BGR numpy image array, or None if reading fails.
        """
        path = Path(image_path)
        if not path.exists() or not path.is_file():
            logger.error("Image path does not exist or is not a file: %s", path)
            return None

        # cv2.imread might fail on non-ASCII paths in Windows, use imdecode as a robust fallback
        try:
            image = cv2.imread(str(path))
            if image is None:
                # Try reading via byte buffer
                with open(path, "rb") as f:
                    file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
                    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if image is None:
                logger.error("Failed to decode image: %s", path)
                return None

            return image
        except Exception as e:
            logger.error("Exception loading image '%s': %s", path, e)
            return None


_face_engine_instance: Optional[FaceEngine] = None


def get_face_engine(
    model_name: str = "buffalo_l",
    providers: Optional[List[str]] = None,
    det_size: Tuple[int, int] = (640, 640),
    det_thresh: float = 0.50,
) -> FaceEngine:
    """Singleton getter for the centralized FaceEngine."""
    global _face_engine_instance
    if _face_engine_instance is None:
        _face_engine_instance = FaceEngine(
            model_name=model_name,
            providers=providers,
            det_size=det_size,
            det_thresh=det_thresh,
        )
    return _face_engine_instance
