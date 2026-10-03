"""
Concrete Camera Source Implementations: FileSource, USBSource, and RTSPSource.
Includes automatic backoff reconnection for RTSP and playback pacing for video files.
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from src.cameras.config import CameraSourceConfig
from src.cameras.source import CameraSource

logger = logging.getLogger(__name__)


class FileSource(CameraSource):
    """
    Video File Source.
    Supports playback of prerecorded files (e.g. mp4, avi, mkv) with optional realtime pacing and looping.
    """

    def __init__(self, config: CameraSourceConfig):
        super().__init__(config)
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_idx = 0
        self.total_frames = 0
        self.start_mono: Optional[float] = None
        self.last_frame_mono: Optional[float] = None

    def open(self) -> bool:
        src_path = Path(self.config.source)
        if not src_path.exists():
            # Check relative to project root
            from src.config import PROJECT_ROOT
            alt_path = PROJECT_ROOT / self.config.source
            if alt_path.exists():
                src_path = alt_path
            else:
                self.last_error = f"Failed to open video source: Video file not found: {self.config.source}"
                logger.error("[FileSource] %s", self.last_error)
                self.is_opened = False
                return False

        self.cap = cv2.VideoCapture(str(src_path))
        if not self.cap.isOpened():
            self.last_error = f"Failed to open video source: OpenCV failed to open video file: {src_path}"
            logger.error("[FileSource] %s", self.last_error)
            self.is_opened = False
            return False

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.stream_fps = fps if (fps > 0 and not np.isnan(fps)) else 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_idx = 0
        self.start_mono = time.monotonic()
        self.last_frame_mono = self.start_mono
        self.is_opened = True
        self.last_error = None
        logger.info("[FileSource] Opened '%s' (%.1f FPS, %dx%d, %d frames)", src_path.name, self.stream_fps, self.frame_width, self.frame_height, self.total_frames)
        return True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], float]:
        if not self.is_opened or self.cap is None:
            return False, None, 0.0

        # Realtime playback pacing
        if self.config.realtime and self.stream_fps > 0:
            target_interval = 1.0 / self.stream_fps
            now = time.monotonic()
            elapsed = now - (self.last_frame_mono or now)
            if elapsed < target_interval:
                time.sleep(target_interval - elapsed)
            self.last_frame_mono = time.monotonic()

        ret, frame = self.cap.read()
        if not ret or frame is None:
            if self.config.loop_file and self.is_opened:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.frame_idx = 0
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    return False, None, 0.0
            else:
                return False, None, float(self.frame_idx) / self.stream_fps

        timestamp_sec = float(self.frame_idx) / self.stream_fps
        self.frame_idx += 1
        return True, frame, timestamp_sec

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "source_type": "file",
            "source": self.config.source,
            "fps": round(self.stream_fps, 2),
            "stream_fps": round(self.stream_fps, 2),
            "width": self.frame_width,
            "height": self.frame_height,
            "total_frames": self.total_frames,
            "current_frame": self.frame_idx,
            "duration_sec": round(self.total_frames / self.stream_fps, 2) if self.stream_fps > 0 else 0.0,
        }

    def close(self) -> None:
        self.is_opened = False
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        logger.info("[FileSource] Closed '%s'.", self.config.source)


class USBSource(CameraSource):
    """
    USB / Hardware Webcam Source.
    Connects to local video device index (e.g. 0, 1).
    """

    def __init__(self, config: CameraSourceConfig):
        super().__init__(config)
        self.cap: Optional[cv2.VideoCapture] = None
        self.device_index = 0
        try:
            self.device_index = int(config.source)
        except ValueError:
            self.device_index = 0
        self.start_time = 0.0

    def open(self) -> bool:
        # Use DSHOW on Windows for fast device capture when available
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(self.device_index, backend)
        if not self.cap.isOpened():
            # Fallback to default backend
            self.cap = cv2.VideoCapture(self.device_index)

        if not self.cap.isOpened():
            self.last_error = f"Failed to open USB camera device index: {self.device_index}"
            logger.error("[USBSource] %s", self.last_error)
            self.is_opened = False
            return False

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.stream_fps = fps if (fps > 0 and not np.isnan(fps)) else 30.0
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.start_time = time.time()
        self.is_opened = True
        self.last_error = None
        logger.info("[USBSource] Opened USB camera %d (%.1f FPS, %dx%d)", self.device_index, self.stream_fps, self.frame_width, self.frame_height)
        return True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], float]:
        if not self.is_opened or self.cap is None:
            return False, None, 0.0

        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.last_error = "Failed to grab frame from USB camera"
            return False, None, 0.0

        timestamp_sec = time.time() - self.start_time
        return True, frame, timestamp_sec

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "source_type": "usb",
            "source": str(self.device_index),
            "fps": round(self.stream_fps, 2),
            "stream_fps": round(self.stream_fps, 2),
            "width": self.frame_width,
            "height": self.frame_height,
        }

    def close(self) -> None:
        self.is_opened = False
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        logger.info("[USBSource] Closed USB camera %d.", self.device_index)


class RTSPSource(CameraSource):
    """
    RTSP / IP Camera Stream Source.
    Supports TCP/UDP transport configuration, timeout safety, and exponential backoff auto-reconnect.
    """

    def __init__(self, config: CameraSourceConfig):
        super().__init__(config)
        self.cap: Optional[cv2.VideoCapture] = None
        self.last_frame_time = 0.0
        self.current_backoff = config.reconnect_delay_sec

    def _configure_env(self) -> None:
        """Set RTSP transport options for FFmpeg backend."""
        transport_val = "tcp" if self.config.transport == "tcp" else "udp"
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{transport_val}|stimeout;{int(self.config.timeout_sec * 1000000)}"

    def open(self) -> bool:
        self._configure_env()
        # Sanitize log output to not log passwords
        safe_url = self._mask_credentials(self.config.source)
        logger.info("[RTSPSource] Connecting to RTSP stream '%s' (Transport: %s, Timeout: %.1fs)...", safe_url, self.config.transport, self.config.timeout_sec)

        self.cap = cv2.VideoCapture(self.config.source, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            self.last_error = f"Failed to connect to RTSP stream: {safe_url}"
            logger.warning("[RTSPSource] %s", self.last_error)
            self.is_opened = False
            return False

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.stream_fps = fps if (fps > 0 and not np.isnan(fps)) else 25.0
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.last_frame_time = time.time()
        self.is_opened = True
        self.last_error = None
        self.current_backoff = self.config.reconnect_delay_sec
        logger.info("[RTSPSource] Connected to '%s' (%.1f FPS, %dx%d)", safe_url, self.stream_fps, self.frame_width, self.frame_height)
        return True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], float]:
        if not self.is_opened or self.cap is None:
            if self.config.reconnect_enabled:
                reconnected = self._attempt_reconnect()
                if not reconnected:
                    return False, None, 0.0
            else:
                return False, None, 0.0

        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.last_error = "RTSP read timeout or connection dropped"
            self.is_opened = False
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None

            if self.config.reconnect_enabled:
                self._attempt_reconnect()
            return False, None, 0.0

        self.last_frame_time = time.time()
        return True, frame, self.last_frame_time

    def _attempt_reconnect(self) -> bool:
        """Perform non-blocking or backoff reconnection."""
        self.reconnect_count += 1
        safe_url = self._mask_credentials(self.config.source)
        logger.warning("[RTSPSource] Reconnecting to '%s' (Attempt #%d, backoff: %.1fs)...", safe_url, self.reconnect_count, self.current_backoff)
        time.sleep(min(self.current_backoff, self.config.max_reconnect_delay_sec))
        self.current_backoff = min(self.current_backoff * 1.5, self.config.max_reconnect_delay_sec)
        return self.open()

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "source_type": "rtsp",
            "source": self._mask_credentials(self.config.source),
            "fps": round(self.stream_fps, 2),
            "stream_fps": round(self.stream_fps, 2),
            "width": self.frame_width,
            "height": self.frame_height,
            "transport": self.config.transport,
            "reconnect_count": self.reconnect_count,
        }

    @staticmethod
    def _mask_credentials(url: str) -> str:
        """Mask credentials in RTSP URL (e.g. rtsp://user:pass@host -> rtsp://***:***@host)."""
        if "@" in url and "://" in url:
            try:
                protocol, rest = url.split("://", 1)
                creds, host = rest.split("@", 1)
                return f"{protocol}://***:***@{host}"
            except Exception:
                return "rtsp://***:***@..."
        return url

    def close(self) -> None:
        self.is_opened = False
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        logger.info("[RTSPSource] Closed stream.")


def create_camera_source(config: CameraSourceConfig) -> CameraSource:
    """Factory function creating the appropriate CameraSource instance."""
    st = config.source_type.lower()
    if st == "usb":
        return USBSource(config)
    elif st == "rtsp":
        return RTSPSource(config)
    else:
        # Default: file or pre-recorded video replay
        return FileSource(config)

