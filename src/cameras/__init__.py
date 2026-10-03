"""
Camera sources module.
Provides CameraSource interface, FileSource, USBSource, RTSPSource, and configuration classes.
"""

from src.cameras.config import CameraSourceConfig, SourceType, RTSPTransport
from src.cameras.source import CameraSource
from src.cameras.sources import FileSource, USBSource, RTSPSource, create_camera_source

__all__ = [
    "CameraSourceConfig",
    "SourceType",
    "RTSPTransport",
    "CameraSource",
    "FileSource",
    "USBSource",
    "RTSPSource",
    "create_camera_source",
]

