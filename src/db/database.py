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

from src.auth.security import hash_password
from src.config import PROJECT_ROOT
from src.db.models import AlertRecord, AuditLog, Base, Camera, Person, TrackRecord, User
from src.db.repositories.users import UserRepository

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
    target_url = database_url if database_url is not None else (str(_engine.url) if _engine is not None else os.getenv("DATABASE_URL", DEFAULT_DB_URL))

    if _engine is not None and str(_engine.url) == target_url:
        return _engine

    connect_args = {}
    if target_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
        _engine = create_engine(target_url, connect_args=connect_args, echo=False)
    else:
        # PostgreSQL / other dialects
        _engine = create_engine(target_url, pool_pre_ping=True, pool_size=10, max_overflow=20, echo=False)

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


def _migrate_sqlite_columns(engine: Engine) -> None:
    """Safely add any missing columns to existing SQLite tables."""
    try:
        with engine.connect() as conn:
            # Check if persons table exists
            table_check = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='persons'")).fetchone()
            if table_check:
                # Fetch existing columns
                cols = {row[1] for row in conn.execute(text("PRAGMA table_info(persons)")).fetchall()}
                expected_cols = {
                    "case_id": "VARCHAR(64)",
                    "age": "INTEGER",
                    "gender": "VARCHAR(32)",
                    "date_last_seen": "DATETIME",
                    "last_known_location": "VARCHAR(256)",
                    "notes": "TEXT",
                    "status": "VARCHAR(32) DEFAULT 'ACTIVE'",
                    "photo_paths": "TEXT DEFAULT '[]'",
                    "embedding_path": "VARCHAR(512)",
                    "is_active": "BOOLEAN DEFAULT 1",
                    "updated_at": "DATETIME",
                }
                for col, col_type in expected_cols.items():
                    if col not in cols:
                        logger.info("Migrating SQLite schema: adding column %s to persons table", col)
                        conn.execute(text(f"ALTER TABLE persons ADD COLUMN {col} {col_type}"))

            # Check if cameras table exists
            cam_check = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='cameras'")).fetchone()
            if cam_check:
                cam_cols = {row[1] for row in conn.execute(text("PRAGMA table_info(cameras)")).fetchall()}
                expected_cam_cols = {
                    "source_type": "VARCHAR(32) DEFAULT 'file'",
                    "status": "VARCHAR(32) DEFAULT 'OFFLINE'",
                    "stream_fps": "FLOAT DEFAULT 30.0",
                    "ai_fps": "FLOAT DEFAULT 2.0",
                    "latency_ms": "FLOAT DEFAULT 0.0",
                    "reconnect_count": "INTEGER DEFAULT 0",
                    "last_connected_at": "DATETIME",
                    "last_frame_at": "DATETIME",
                    "error_message": "TEXT",
                    "updated_at": "DATETIME",
                }
                for col, col_type in expected_cam_cols.items():
                    if col not in cam_cols:
                        logger.info("Migrating SQLite schema: adding column %s to cameras table", col)
                        conn.execute(text(f"ALTER TABLE cameras ADD COLUMN {col} {col_type}"))

            conn.commit()
    except Exception as e:
        logger.warning("SQLite column migration notice: %s", e)


def init_db(engine: Optional[Engine] = None, seed_defaults: bool = True) -> None:
    """
    Create all database tables and seed default camera / person / user records if empty.
    """
    eng = engine or get_engine()
    Base.metadata.create_all(bind=eng)
    _migrate_sqlite_columns(eng)
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
                        source="",
                        enabled=False,
                        sample_fps=2.0,
                    ),
                    Camera(
                        camera_id="C4",
                        name="South Corridor",
                        location="Ground Floor South",
                        source="",
                        enabled=False,
                        sample_fps=2.0,
                    ),
                ]
                session.add_all(default_cameras)
                session.commit()
                logger.info("Seeded %d default cameras (C1-C4); C3/C4 are disabled until real sources are configured.", len(default_cameras))

            # Seed demo operator accounts
            admin_password = os.getenv("ADMIN_PASSWORD", "admin-demo-CHANGE-ME")
            police_password = os.getenv("POLICE_PASSWORD", "police-demo-CHANGE-ME")
            hospital_password = os.getenv("HOSPITAL_PASSWORD", "hospital-demo-CHANGE-ME")
            ngo_password = os.getenv("NGO_PASSWORD", "ngo-demo-CHANGE-ME")

            user_repo = UserRepository(session)
            user_repo.ensure_user_exists("admin", hash_password(admin_password), "ADMIN")
            user_repo.ensure_user_exists("demo_admin", hash_password(admin_password), "ADMIN")
            user_repo.ensure_user_exists("police", hash_password(police_password), "POLICE")
            user_repo.ensure_user_exists("officer01", hash_password(police_password), "POLICE")
            user_repo.ensure_user_exists("demo_police", hash_password(police_password), "POLICE")
            user_repo.ensure_user_exists("hospital", hash_password(hospital_password), "HOSPITAL")
            user_repo.ensure_user_exists("hospital01", hash_password(hospital_password), "HOSPITAL")
            user_repo.ensure_user_exists("demo_hospital", hash_password(hospital_password), "HOSPITAL")
            user_repo.ensure_user_exists("ngo", hash_password(ngo_password), "NGO")
            user_repo.ensure_user_exists("ngo01", hash_password(ngo_password), "NGO")
            user_repo.ensure_user_exists("demo_ngo", hash_password(ngo_password), "NGO")
            logger.info("Seeded/verified demo users (admin, police, hospital, ngo, demo accounts).")

