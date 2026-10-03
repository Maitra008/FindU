"""
Camera Source Configuration Data Classes.
Supports file, usb, and rtsp camera source types.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

SourceType = Literal["file", "usb", "rtsp"]
RTSPTransport = Literal["tcp", "udp"]


@dataclass
class CameraSourceConfig:
    """Configuration for an individual camera stream source."""
    source_type: SourceType = "file"
    source: str = ""  # File path, USB device index string (e.g. "0"), or RTSP URI
    sample_fps: float = 2.0  # AI processing sample rate
    realtime: bool = True  # If True, paces file playback at natural stream FPS
    reconnect_enabled: bool = True  # Auto-reconnect for RTSP/network feeds
    reconnect_delay_sec: float = 2.0  # Initial reconnect backoff delay
    max_reconnect_delay_sec: float = 30.0  # Max reconnect backoff ceiling
    transport: RTSPTransport = "tcp"  # Preferred RTSP transport (TCP reduces packet drop)
    timeout_sec: float = 5.0  # Stream read/connect timeout in seconds
    loop_file: bool = False  # Loop video file continuously for simulation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_type": self.source_type,
            "source": self.source,
            "sample_fps": self.sample_fps,
            "realtime": self.realtime,
            "reconnect_enabled": self.reconnect_enabled,
            "reconnect_delay_sec": self.reconnect_delay_sec,
            "max_reconnect_delay_sec": self.max_reconnect_delay_sec,
            "transport": self.transport,
            "timeout_sec": self.timeout_sec,
            "loop_file": self.loop_file,
        }

    @property
    def masked_source(self) -> str:
        """Mask credentials in RTSP URLs for safe display and logging."""
        if not self.source or self.source_type != "rtsp":
            return self.source
        import re
        return re.sub(r"://([^:@]+):([^@]+)@", r"://\1:***@", self.source)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CameraSourceConfig":
        return cls(
            source_type=data.get("source_type", "file"),
            source=str(data.get("source", "")),
            sample_fps=float(data.get("sample_fps", 2.0)),
            realtime=bool(data.get("realtime", True)),
            reconnect_enabled=bool(data.get("reconnect_enabled", True)),
            reconnect_delay_sec=float(data.get("reconnect_delay_sec", 2.0)),
            max_reconnect_delay_sec=float(data.get("max_reconnect_delay_sec", 30.0)),
            transport=data.get("transport", "tcp"),
            timeout_sec=float(data.get("timeout_sec", 5.0)),
            loop_file=bool(data.get("loop_file", False)),
        )

