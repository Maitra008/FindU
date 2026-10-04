"""
Recognition Core - Missing-Person Alert Platform
"""

__version__ = "1.0.0"

from src.face_engine import DetectedFace, FaceEngine, get_face_engine, normalize_embedding
from src.index import FaceIndex, MatchResult, get_face_index
from src.pipeline import FrameRecognitionResult, recognize_frame
