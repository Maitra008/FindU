"""
API route routers.
"""

from src.api.routes.alerts import router as alerts_router
from src.api.routes.cameras import router as cameras_router
from src.api.routes.health import router as health_router
from src.api.routes.workers import router as workers_router

__all__ = ["health_router", "cameras_router", "alerts_router", "workers_router"]

