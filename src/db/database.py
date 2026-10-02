"""
Database connection, session management, and schema initialization.
Supports PostgreSQL (with connection pooling) and graceful SQLite fallback for testing.
"""

import logging
import os
from pathlib import Path
from typing import Generator, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.config import PROJECT_ROOT
from src.db.models import Base, Camera, Person

logger = logging.getLogger(__name__)

# Default Database Connection URL
DEFAULT_DB_URL = os.getenv("DATABASE_URL", f"sqlite:///{PROJECT_ROOT / 'data' / 'missing_person.db'}")

_engine: Optional[Engine] = None
_SessionFactory: Optional[sessionmaker] = None


def get_engine(database_url: Optional[str] = None) -> Engine:
    """
    Get or create SQLAlchemy Engine.
    Configures pool pre-ping and appropriate connect args based on DB dialect.
    """
    global _engine, _SessionFactory
    if database_url is None and _engine is not None:
        return _engine

    url = database_url or os.getenv("DATABASE_URL", DEFAULT_DB_URL)

    # If engine already initialized for the same URL, reuse it
    if _engine is not None and str(_engine.url) == url:
        return _engine

    connect_args = {}
    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
        _engine = create_engine(url, connect_args=connect_args, echo=False)
    else:
        # PostgreSQL / other dialects
        _engine = create_engine(url, pool_pre_ping=True, pool_size=10, max_overflow=20, echo=False)

    _SessionFactory = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
    logger.info("Initialized database engine for '%s'", _engine.url.render_as_string(hide_password=True))
    return _engine


def get_session_factory(database_url: Optional[str] = None) -> sessionmaker:
    """Return configured sessionmaker instance."""
    global _SessionFactory
    if _SessionFactory is None or database_url is not None:
        get_engine(database_url)
    return _SessionFactory


def get_db(database_url: Optional[str] = None) -> Generator[Session, None, None]:
    """
    FastAPI dependency that yields a database session and ensures clean closure.
    """
    session_factory = get_session_factory(database_url)
    db = session_factory()
    try:
        yield db
    finally:
        db.close()


def check_db_connection(engine: Optional[Engine] = None) -> bool:
    """
    Health check query to verify database connectivity.

    Returns:
        bool: True if query executes successfully, False otherwise.
    """
    eng = engine or get_engine()
    try:
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error("Database health check connection failed: %s", e)
        return False


def init_db(engine: Optional[Engine] = None, seed_defaults: bool = True) -> None:
    """
    Create all database tables and seed default camera / person records if empty.
    """
    eng = engine or get_engine()
    Base.metadata.create_all(bind=eng)
    logger.info("Database tables verified/created successfully.")

    if seed_defaults:
        session_factory = sessionmaker(bind=eng)
        with session_factory() as session:
            # Seed default cameras (C1 to C4) if cameras table is empty
            if session.query(Camera).count() == 0:
                default_cameras = [
                    Camera(
                        camera_id="C1",
                        name="North Entrance",
                        location="Building A - Gate 1",
                        source="data/videos/camera_1.mp4",
                        enabled=True,
                        sample_fps=2.0,
                    ),
                    Camera(
                        camera_id="C2",
                        name="Parking Lot East",
                        location="East Wing Parking",
                        source="data/videos/camera_2.mp4",
                        enabled=True,
                        sample_fps=2.0,
                    ),
                    Camera(
                        camera_id="C3",
                        name="Main Lobby",
                        location="Central Reception",
                        source="data/videos/camera_3.mp4",
                        enabled=False,
                        sample_fps=2.0,
                    ),
                    Camera(
                        camera_id="C4",
                        name="South Corridor",
                        location="Ground Floor South",
                        source="data/videos/camera_4.mp4",
                        enabled=False,
                        sample_fps=2.0,
                    ),
                ]
                session.add_all(default_cameras)
                session.commit()
                logger.info("Seeded %d default cameras; C3/C4 are disabled until real sources are configured.", len(default_cameras))

