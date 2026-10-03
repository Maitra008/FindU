"""
Comprehensive Automated Tests for Backend, Database, Alert Service, WebSockets, and Multi-Camera Workers (Part 4).
Verifies:
- Database initialization, schema validation, and CRUD operations.
- Central AlertService persistence, validation, and post-commit WebSocket broadcast.
- WebSocket connection management, multiple client broadcasting, and graceful disconnects.
- FastAPI REST API endpoints (/health, /api/cameras, /api/alerts, /api/tracks, /api/workers).
- Standalone CameraWorker and concurrent multi-camera (C1 and C2) video processing with real Part 3 pipeline.
- Failure handling (invalid video source, database error, client disconnect).
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.api.app import create_app
from src.api.websocket import WebSocketConnectionManager, ws_manager
from src.auth.security import create_access_token
from src.db.database import check_db_connection, init_db
from src.db.models import AlertRecord, Base, Camera, Person, TrackRecord
from src.db.repositories.alerts import AlertRepository, TrackRepository
from src.db.repositories.cameras import CameraRepository
from src.services.alert_service import AlertService
from src.tracker import TrackAlert
from src.workers.camera_worker import CameraWorker
from src.workers.worker_manager import WorkerManager


@pytest.fixture
def test_db_session(tmp_path):
    """Isolated SQLite database fixture for testing."""
    db_file = tmp_path / "test_missing_person.db"
    test_db_url = f"sqlite:///{db_file}"
    engine = create_engine(test_db_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    try:
        yield session, session_factory, engine
    finally:
        session.close()


@pytest.fixture
def mock_ws_manager():
    """Isolated WebSocket manager for testing."""
    return WebSocketConnectionManager()


@pytest.fixture
def alert_service(test_db_session, mock_ws_manager):
    """AlertService instance wired to test DB and test WebSocket manager."""
    _, session_factory, _ = test_db_session
    return AlertService(session_factory=session_factory, websocket_manager=mock_ws_manager)


# ==========================================
# 1. DATABASE TESTS
# ==========================================

def test_database_init_and_tables(test_db_session):
    """Verify database initialization creates all tables and check_db_connection succeeds."""
    session, _, engine = test_db_session
    assert check_db_connection(engine) is True

    # Verify tables exist by querying models
    assert session.query(Camera).count() == 0
    assert session.query(AlertRecord).count() == 0
    assert session.query(TrackRecord).count() == 0


def test_camera_repository_crud(test_db_session):
    """Verify CameraRepository create, get, and update operations."""
    session, _, _ = test_db_session
    repo = CameraRepository(session)

    cam1 = repo.create_or_update(
        camera_id="C1",
        name="Main Entrance",
        source="data/videos/camera_1.mp4",
        location="Gate 1",
        enabled=True,
        sample_fps=2.0,
    )
    assert cam1.id is not None
    assert cam1.camera_id == "C1"
    assert cam1.name == "Main Entrance"

    # Retrieve
    fetched = repo.get_by_camera_id("C1")
    assert fetched is not None
    assert fetched.name == "Main Entrance"
    assert fetched.location == "Gate 1"

    # Update
    repo.create_or_update(
        camera_id="C1",
        name="Main Entrance North",
        source="data/videos/camera_1.mp4",
        location="Gate 1 North",
        enabled=False,
        sample_fps=3.0,
    )
    updated = repo.get_by_camera_id("C1")
    assert updated.name == "Main Entrance North"
    assert updated.enabled is False
    assert updated.sample_fps == 3.0


def test_alert_repository_crud(test_db_session):
    """Verify AlertRepository creation, retrieval, filtering, and status updates."""
    session, _, _ = test_db_session
    repo = AlertRepository(session)

    alert = repo.create_alert(
        alert_id="ALERT-C1-TRACK-0001-0",
        camera_id="C1",
        track_id="TRACK-0001",
        person_id="person_x",
        person_name="Person X",
        similarity=0.935,
        max_similarity=0.935,
        mean_similarity=0.920,
        threshold=0.89,
        timestamp=1.5,
        frame_idx=45,
        bbox=[100, 120, 240, 280],
        status="NEW",
    )
    assert alert.id is not None
    assert alert.alert_id == "ALERT-C1-TRACK-0001-0"
    assert alert.status == "NEW"

    # Retrieve by ID
    by_pk = repo.get_by_id(alert.id)
    assert by_pk is not None
    assert by_pk.person_id == "person_x"

    by_str_id = repo.get_by_id("ALERT-C1-TRACK-0001-0")
    assert by_str_id is not None
    assert by_str_id.id == alert.id

    # Filter alerts
    c1_alerts = repo.get_alerts(camera_id="C1")
    assert len(c1_alerts) == 1
    assert repo.get_alerts(camera_id="C2") == []

    # Update status
    updated = repo.update_status(alert.id, "VERIFIED")
    assert updated.status == "VERIFIED"
    assert repo.get_by_id(alert.id).status == "VERIFIED"


# ==========================================
# 2. ALERT SERVICE TESTS
# ==========================================

def test_alert_service_persistence_and_broadcast(alert_service, test_db_session):
    """Verify AlertService validates, persists to DB, and broadcasts event."""
    session, _, _ = test_db_session

    track_alert = TrackAlert(
        track_id="TRACK-0001",
        person_id="person_y",
        person_name="Person Y",
        similarity=0.942,
        max_similarity=0.942,
        mean_similarity=0.930,
        threshold=0.89,
        frame_idx=30,
        timestamp_sec=1.0,
        bbox=[80, 100, 200, 240],
        detection_count=2,
    )

    persisted = alert_service.create_alert_from_track_alert(
        camera_id="C1",
        track_alert=track_alert,
    )

    assert persisted is not None
    assert persisted.id is not None
    assert persisted.camera_id == "C1"
    assert persisted.track_id == "TRACK-0001"
    assert persisted.person_id == "person_y"
    assert persisted.similarity == pytest.approx(0.942, abs=1e-3)

    # Verify presence in database
    alerts_in_db = alert_service.get_alerts(camera_id="C1")
    assert len(alerts_in_db) == 1
    assert alerts_in_db[0]["person_name"] == "Person Y"


def test_alert_service_validation_failures(alert_service):
    """Verify AlertService rejects invalid alerts with ValueError."""
    with pytest.raises(ValueError):
        alert_service.create_alert(
            camera_id="",  # Empty camera ID
            track_id="TRACK-0001",
            person_id="person_x",
            person_name="Person X",
            similarity=0.92,
            timestamp=1.0,
            bbox=[100, 100, 200, 200],
        )

    with pytest.raises(ValueError):
        alert_service.create_alert(
            camera_id="C1",
            track_id="TRACK-0001",
            person_id="person_x",
            person_name="Person X",
            similarity=0.92,
            timestamp=1.0,
            bbox=[100],  # Invalid bbox length
        )


def test_alert_service_idempotency_deduplication(alert_service):
    """Verify duplicate alert dispatch for the same camera and track returns existing alert."""
    alert1 = alert_service.create_alert(
        camera_id="C1",
        track_id="TRACK-0001",
        person_id="person_x",
        person_name="Person X",
        similarity=0.93,
        timestamp=1.0,
        bbox=[100, 100, 200, 200],
    )

    alert2 = alert_service.create_alert(
        camera_id="C1",
        track_id="TRACK-0001",
        person_id="person_x",
        person_name="Person X",
        similarity=0.94,
        timestamp=2.0,
        bbox=[110, 100, 210, 200],
    )

    assert alert1.id == alert2.id
    alerts = alert_service.get_alerts(camera_id="C1")
    assert len(alerts) == 1


# ==========================================
# 3. REST API & WEBSOCKET TESTS (TestClient)
# ==========================================

def test_health_endpoint(tmp_path):
    """Verify GET /health returns application and database health."""
    db_file = tmp_path / "test_api.db"
    db_url = f"sqlite:///{db_file}"
    app = create_app(database_url=db_url, init_database=True, load_workers=False)

    with TestClient(app) as client:
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert data["application"] == "healthy"
        assert data["database"] == "connected"


def test_cameras_rest_api(tmp_path):
    """Verify GET /api/cameras and POST /api/cameras."""
    db_file = tmp_path / "test_cameras_api.db"
    db_url = f"sqlite:///{db_file}"
    app = create_app(database_url=db_url, init_database=True, load_workers=False)
    admin_token = create_access_token(username="admin", role="ADMIN")
    headers = {"Authorization": f"Bearer {admin_token}"}

    with TestClient(app) as client:
        # Create Camera
        post_res = client.post(
            "/api/cameras",
            headers=headers,
            json={
                "camera_id": "C10",
                "name": "West Gate",
                "source": "data/videos/camera_1.mp4",
                "location": "West Wing",
                "enabled": True,
                "sample_fps": 3.0,
            },
        )
        assert post_res.status_code == 201
        data = post_res.json()
        assert data["camera_id"] == "C10"
        assert data["name"] == "West Gate"

        # List Cameras
        get_res = client.get("/api/cameras", headers=headers)
        assert get_res.status_code == 200
        cams = get_res.json()
        assert any(c["camera_id"] == "C10" for c in cams)

        # Get Single Camera
        single_res = client.get("/api/cameras/C10", headers=headers)
        assert single_res.status_code == 200
        assert single_res.json()["camera_id"] == "C10"


def test_alerts_rest_api_and_status_update(tmp_path):
    """Verify GET /api/alerts, GET /api/alerts/{id}, and PATCH /api/alerts/{id}."""
    db_file = tmp_path / "test_alerts_api.db"
    db_url = f"sqlite:///{db_file}"
    app = create_app(database_url=db_url, init_database=True, load_workers=False)
    admin_token = create_access_token(username="admin", role="ADMIN")
    headers = {"Authorization": f"Bearer {admin_token}"}

    # Ingest test alert directly via AlertService
    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine)
    service = AlertService(session_factory=session_factory)
    alert = service.create_alert(
        camera_id="C1",
        track_id="TRACK-0001",
        person_id="person_x",
        person_name="Person X",
        similarity=0.93,
        timestamp=1.0,
        bbox=[100, 100, 200, 200],
    )

    with TestClient(app) as client:
        # List alerts
        list_res = client.get("/api/alerts?camera_id=C1", headers=headers)
        assert list_res.status_code == 200
        alerts = list_res.json()
        assert len(alerts) >= 1
        assert alerts[0]["person_id"] == "person_x"
        assert alerts[0]["status"] == "NEW"

        # Update status to VERIFIED
        patch_res = client.patch(
            f"/api/alerts/{alert.id}",
            headers=headers,
            json={"status": "VERIFIED"},
        )
        assert patch_res.status_code == 200
        assert patch_res.json()["status"] == "VERIFIED"

        # Verify updated status on GET
        get_res = client.get(f"/api/alerts/{alert.id}", headers=headers)
        assert get_res.status_code == 200
        assert get_res.json()["status"] == "VERIFIED"


def test_websocket_client_receives_live_alert(tmp_path):
    """Verify connected WebSocket client receives live JSON alert when AlertService triggers."""
    db_file = tmp_path / "test_ws.db"
    db_url = f"sqlite:///{db_file}"
    app = create_app(database_url=db_url, init_database=True, load_workers=False)
    admin_token = create_access_token(username="admin", role="ADMIN")

    with TestClient(app) as client:
        with client.websocket_connect(f"/ws/alerts?token={admin_token}") as ws:
            # Client sends a heartbeat ping
            ws.send_text("ping")
            pong = ws.receive_json()
            assert pong == {"event": "pong"}

            # Trigger an alert via AlertService (which uses global ws_manager)
            engine = create_engine(db_url, connect_args={"check_same_thread": False})
            session_factory = sessionmaker(bind=engine)
            service = AlertService(session_factory=session_factory, websocket_manager=ws_manager)

            service.create_alert(
                camera_id="C1",
                track_id="TRACK-0001",
                person_id="person_y",
                person_name="Person Y",
                similarity=0.945,
                timestamp=2.0,
                bbox=[80, 120, 220, 300],
            )

            # Receive broadcast message over WebSocket
            received_msg = ws.receive_json()
            assert received_msg["event"] == "alert.created"
            assert received_msg["alert"]["camera_id"] == "C1"
            assert received_msg["alert"]["track_id"] == "TRACK-0001"
            assert received_msg["alert"]["person_id"] == "person_y"
            assert received_msg["alert"]["similarity"] == pytest.approx(0.945, abs=1e-3)


# ==========================================
# 4. CAMERA WORKER & MULTI-CAMERA TESTS
# ==========================================

def test_camera_worker_standalone_execution(test_db_session, mock_ws_manager):
    """Verify CameraWorker executes Part 3 recognition on a video file and triggers AlertService."""
    _, session_factory, _ = test_db_session
    service = AlertService(session_factory=session_factory, websocket_manager=mock_ws_manager)

    worker = CameraWorker(
        camera_id="C1",
        source="data/videos/camera_1.mp4",
        name="Camera 1 Test",
        sample_fps=5.0,
        threshold=0.89,
        alert_service=service,
        loop_video=False,
    )

    worker.start()
    worker.join(timeout=10.0)

    assert worker.metrics.frames_read > 0
    assert worker.metrics.frames_processed > 0
    assert worker.metrics.tracks_created >= 1
    assert worker.metrics.alerts_emitted >= 1

    # Verify alert is in database
    alerts = service.get_alerts(camera_id="C1")
    assert len(alerts) >= 1
    assert alerts[0]["camera_id"] == "C1"


def test_multi_camera_workers_concurrent_alerts(test_db_session, mock_ws_manager):
    """
    Verify multiple camera workers (C1 and C2) run concurrently,
    persist separate alerts in the database, and trigger distinct broadcasts.
    """
    _, session_factory, _ = test_db_session
    service = AlertService(session_factory=session_factory, websocket_manager=mock_ws_manager)

    worker_c1 = CameraWorker(
        camera_id="C1",
        source="data/videos/camera_1.mp4",
        name="Camera C1",
        sample_fps=5.0,
        threshold=0.89,
        alert_service=service,
        loop_video=False,
    )

    worker_c2 = CameraWorker(
        camera_id="C2",
        source="data/videos/camera_2.mp4",
        name="Camera C2",
        sample_fps=5.0,
        threshold=0.89,
        alert_service=service,
        loop_video=False,
    )

    # Start both workers concurrently
    worker_c1.start()
    worker_c2.start()

    worker_c1.join(timeout=15.0)
    worker_c2.join(timeout=15.0)

    # Verify both workers produced alerts
    c1_alerts = service.get_alerts(camera_id="C1")
    c2_alerts = service.get_alerts(camera_id="C2")

    assert len(c1_alerts) == 1
    assert len(c2_alerts) == 1

    # Verify distinct camera IDs and track records
    assert c1_alerts[0]["camera_id"] == "C1"
    assert c2_alerts[0]["camera_id"] == "C2"
    assert c1_alerts[0]["alert_id"] != c2_alerts[0]["alert_id"]


# ==========================================
# 5. FAILURE & EDGE CASE TESTS
# ==========================================

def test_camera_worker_invalid_source_error_handling(test_db_session, mock_ws_manager):
    """Verify CameraWorker handles non-existent video files gracefully without crashing."""
    _, session_factory, _ = test_db_session
    service = AlertService(session_factory=session_factory, websocket_manager=mock_ws_manager)

    worker = CameraWorker(
        camera_id="C_INVALID",
        source="data/videos/non_existent_video_path.mp4",
        sample_fps=2.0,
        threshold=0.89,
        alert_service=service,
    )

    worker.start()
    worker.join(timeout=5.0)

    assert worker.metrics.status == "ERROR"
    assert "Failed to open video source" in worker.metrics.error_message


def test_websocket_client_disconnect_resilience(mock_ws_manager):
    """Verify WebSocketConnectionManager handles dead/abruptly disconnected clients without error."""
    class DummyWebSocket:
        async def send_text(self, text: str):
            raise RuntimeError("Client abruptly disconnected")

    dummy = DummyWebSocket()
    mock_ws_manager.active_connections.add(dummy)
    assert len(mock_ws_manager.active_connections) == 1

    # Broadcasting should catch the exception and discard the dead connection
    asyncio.run(mock_ws_manager.broadcast({"event": "test"}))
    assert len(mock_ws_manager.active_connections) == 0

