"""
FastAPI Application Entry Point.
Provides REST API endpoints, database initialization, and real-time WebSocket alert feeds.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes.alerts import router as alerts_router
from src.api.routes.cameras import router as cameras_router
from src.api.routes.health import router as health_router
from src.api.routes.workers import router as workers_router
from src.api.websocket import ws_manager
from src.db.database import get_engine, init_db
from src.workers.worker_manager import get_worker_manager

logger = logging.getLogger(__name__)


def create_app(
    database_url: Optional[str] = None,
    init_database: bool = True,
    load_workers: bool = True,
) -> FastAPI:
    """
    Application factory for the Missing Person Finder backend.
    """
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        loop = asyncio.get_running_loop()
        ws_manager.set_event_loop(loop)

        if init_database:
            engine = get_engine(database_url)
            init_db(engine=engine, seed_defaults=True)

        worker_mgr = get_worker_manager()
        if load_workers:
            worker_mgr.load_cameras_from_db(enabled_only=True)

        logger.info("FastAPI application started successfully.")
        yield

        # Shutdown
        worker_mgr = get_worker_manager()
        worker_mgr.stop_all()
        logger.info("FastAPI application shutdown complete.")

    configured_origins = [
        origin.strip()
        for origin in os.getenv(
            "FRONTEND_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173",
        ).split(",")
        if origin.strip()
    ]

    app = FastAPI(
        title="Missing Person Recognition & Alert System",
        description="Part 4: Multi-Camera Ingestion, Tracking, PostgreSQL, and WebSocket Alerts",
        version="4.0.0",
        lifespan=lifespan,
    )

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=configured_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include REST Routers
    app.include_router(health_router)
    app.include_router(cameras_router)
    app.include_router(alerts_router)
    app.include_router(workers_router)

    # WebSocket Real-Time Alert Feed Endpoint
    @app.websocket("/ws/alerts")
    async def websocket_alerts_endpoint(websocket: WebSocket):
        """
        WebSocket stream for live alert broadcasting.
        Receives new alerts in real-time as they are persisted to the database.
        """
        await ws_manager.connect(websocket)
        try:
            # Keep connection open and handle client messages/heartbeats
            while True:
                data = await websocket.receive_text()
                # Optional ping-pong / client heartbeat
                if data == "ping":
                    await websocket.send_text('{"event": "pong"}')
        except WebSocketDisconnect:
            ws_manager.disconnect(websocket)
        except Exception as e:
            logger.debug("WebSocket exception on /ws/alerts: %s", e)
            ws_manager.disconnect(websocket)

    return app


# Default app instance
app = create_app()

