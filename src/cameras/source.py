"""
Abstract Camera Source Interface.
Provides standard contract for frame ingestion across file, USB, and RTSP feeds.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import numpy as np

from src.cameras.config import CameraSourceConfig


class CameraSource(ABC):
    """
    Abstract Base Class for video/camera stream ingestion.
    Encapsulates lifecycle, frame retrieval, and source metadata.
    """

    def __init__(self, config: CameraSourceConfig):
        self.config = config
        self.is_opened = False
        self.last_error: Optional[str] = None
        self.reconnect_count = 0
        self.stream_fps = 30.0
        self.frame_width = 0
        self.frame_height = 0

    @abstractmethod
    def open(self) -> bool:
        """
        Open connection or file for the video source.
        Returns True if successfully initialized, False otherwise.
        """
        pass

    @abstractmethod
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """
        Retrieve next video frame.
        Returns:
            (success: bool, frame: Optional[np.ndarray], timestamp_sec: float)
        """
        pass

    @property
    def is_open(self) -> bool:
        """Check whether the underlying stream/file is currently open."""
        return self.is_opened

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """
        Return stream metadata including resolution, stream FPS, and source type.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Release hardware, network handles, or file descriptors.
        """
        pass

