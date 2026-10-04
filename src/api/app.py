"""
FastAPI Application Entry Point.
Provides REST API endpoints, database initialization, and real-time WebSocket alert feeds.
Part 6 & Missing Persons Case Management: JWT authentication, role-based access control, audit logging.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from src.api.routes.alerts import router as alerts_router
from src.api.routes.audit import router as audit_router
from src.api.routes.auth import router as auth_router
from src.api.routes.cameras import router as cameras_router
from src.api.routes.demo import router as demo_router
from src.api.routes.health import router as health_router
from src.api.routes.historical import router as historical_router
from src.api.routes.media import router as media_router
from src.api.routes.persons import router as persons_router
from src.api.routes.workers import router as workers_router
from src.api.websocket import ws_manager
from src.auth.dependencies import get_ws_user
from src.db.database import get_db, get_engine, init_db
from src.services.historical_service import get_historical_service
from src.workers.worker_manager import get_worker_manager

logger = logging.getLogger(__name__)


def create_app(
    database_url: Optional[str] = None,
    init_database: bool = True,
    load_workers: bool = True,
) -> FastAPI:
    """
    Application factory for the FindU Missing Person Finder backend.
    """
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        loop = asyncio.get_running_loop()
        ws_manager.set_event_loop(loop)

        if init_database:
            engine = get_engine(database_url)
            init_db(engine=engine, seed_defaults=True)

        # Recover stale historical jobs
        try:
            get_historical_service().recover_stale_jobs()
        except Exception as e:
            logger.warning("Could not recover stale jobs on startup: %s", e)

        worker_mgr = get_worker_manager()
        if load_workers:
            worker_mgr.load_cameras_from_db(enabled_only=True)

        logger.info("FindU FastAPI application started successfully.")
        yield

        # Shutdown
        worker_mgr = get_worker_manager()
        worker_mgr.stop_all()
        logger.info("FindU FastAPI application shutdown complete.")

    configured_origins = [
        origin.strip()
        for origin in os.getenv(
            "FRONTEND_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000",
        ).split(",")
        if origin.strip()
    ]

    app = FastAPI(
        title="FindU — Missing Person Recognition & Alert System",
        description=(
            "Multi-camera face recognition, IoU tracking, PostgreSQL persistence, "
            "real-time WebSocket alerts, JWT authentication, and role-based access control."
        ),
        version="6.1.0",
        lifespan=lifespan,
    )

    # CORS Middleware — explicit origins only; wildcard is not used with credentials
    app.add_middleware(
        CORSMiddleware,
        allow_origins=configured_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Public routes (no auth required)
    app.include_router(health_router)
    app.include_router(auth_router)
    app.include_router(media_router)

    # Protected REST routes (auth required via individual route dependencies)
    app.include_router(cameras_router)
    app.include_router(alerts_router)
    app.include_router(workers_router)
    app.include_router(demo_router)
    app.include_router(audit_router)
    app.include_router(persons_router)
    app.include_router(historical_router)

    # WebSocket Real-Time Alert Feed — JWT authenticated via ?token= query param
    @app.websocket("/ws/alerts")
    async def websocket_alerts_endpoint(
        websocket: WebSocket,
        db: Session = Depends(get_db),
    ):
        """
        WebSocket stream for live alert broadcasting.

        Authentication: pass JWT as query parameter ?token=<bearer_token>
        The connection is rejected with close code 4001 for missing/invalid/expired tokens.
        """
        user = await get_ws_user(websocket, db)
        if user is None:
            return

        await ws_manager.connect(websocket)
        logger.info("WebSocket client authenticated: user='%s' role='%s'", user.username, user.role)

        try:
            # Keep connection alive; handle optional ping-pong heartbeats
            while True:
                data = await websocket.receive_text()
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
