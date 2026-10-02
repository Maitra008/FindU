"""
Configuration module for Recognition Core.
Centralizes all default parameters, paths, and settings.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple

# Base Project Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REGISTRATION_DIR = DATA_DIR / "registration"
VIDEOS_DIR = DATA_DIR / "videos"

MODELS_DIR = PROJECT_ROOT / "models"
EMBEDDINGS_DIR = MODELS_DIR / "embeddings"
INDEX_PATH = MODELS_DIR / "face_index.faiss"
METADATA_PATH = MODELS_DIR / "face_metadata.json"

# InsightFace Model Configuration
MODEL_NAME = "buffalo_l"
EXECUTION_PROVIDERS = ["CPUExecutionProvider"]
DETECTION_SIZE = (640, 640)
DET_THRESH = 0.50
EMBEDDING_DIM = 512

# Recognition & Sampling Defaults
DEFAULT_SIMILARITY_THRESHOLD = 0.89
DEFAULT_SAMPLE_FPS = 2.0

# Tracking Defaults (Part 3)
DEFAULT_IOU_THRESHOLD = 0.30
DEFAULT_MAX_MISSED_FRAMES = 5
DEFAULT_TRACK_TOP_K = 3


@dataclass
class RecognitionConfig:
    """Runtime configuration container allowing overrides from CLI or scripts."""
    model_name: str = MODEL_NAME
    providers: List[str] = field(default_factory=lambda: list(EXECUTION_PROVIDERS))
    det_size: Tuple[int, int] = DETECTION_SIZE
    det_thresh: float = DET_THRESH
    embedding_dim: int = EMBEDDING_DIM
    
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    sample_fps: float = DEFAULT_SAMPLE_FPS
    # Tracking parameters
    iou_threshold: float = DEFAULT_IOU_THRESHOLD
    max_missed_frames: int = DEFAULT_MAX_MISSED_FRAMES
    track_top_k: int = DEFAULT_TRACK_TOP_K
    
    models_dir: Path = MODELS_DIR
    embeddings_dir: Path = EMBEDDINGS_DIR
    index_path: Path = INDEX_PATH
    metadata_path: Path = METADATA_PATH
    registration_dir: Path = REGISTRATION_DIR
    videos_dir: Path = VIDEOS_DIR

    def ensure_directories(self) -> None:
        """Create required directories if they don't exist."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        self.registration_dir.mkdir(parents=True, exist_ok=True)
        self.videos_dir.mkdir(parents=True, exist_ok=True)

