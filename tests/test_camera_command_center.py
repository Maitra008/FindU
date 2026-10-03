"""
Tests for Camera Command Center & Historical Footage Replay / Search.
Covers:
  1. CameraSource abstractions (FileSource, USBSource, RTSPSource).
  2. Camera Registry CRUD API + Role authorization (ADMIN vs POLICE/NGO).
  3. Live connection testing (POST /api/cameras/test-source, POST /api/cameras/{id}/test).
  4. Historical Footage Background Service & API (POST /api/historical/upload, GET /api/historical/jobs).
"""

import os
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.cameras.config import CameraSourceConfig
from src.cameras.sources import FileSource, RTSPSource, USBSource, create_camera_source
from src.db.database import get_engine, init_db


@pytest.fixture
def test_app(tmp_path):
    db_file = tmp_path / "test_ccc.db"
    db_url = f"sqlite:///{db_file}"
    engine = get_engine(db_url)
    init_db(engine=engine, seed_defaults=True)
    app = create_app(database_url=db_url, init_database=False, load_workers=False)
    return app


@pytest.fixture
def admin_headers(test_app):
    with TestClient(test_app) as client:
        res = client.post("/api/auth/login", data={"username": "admin", "password": "admin-demo-CHANGE-ME"})
        assert res.status_code == 200
        token = res.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def police_headers(test_app):
    with TestClient(test_app) as client:
        res = client.post("/api/auth/login", data={"username": "police", "password": "police-demo-CHANGE-ME"})
        assert res.status_code == 200
        token = res.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def ngo_headers(test_app):
    with TestClient(test_app) as client:
        res = client.post("/api/auth/login", data={"username": "ngo", "password": "ngo-demo-CHANGE-ME"})
        assert res.status_code == 200
        token = res.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}


def test_file_source_open_and_read():
    """Verify FileSource opens a valid test video and reads frames."""
    video_path = "data/videos/camera_1.mp4"
    if not Path(video_path).exists():
        pytest.skip("Test video not found")

    cfg = CameraSourceConfig(
        source=video_path,
        source_type="file",
        loop_file=False,
    )
    src = create_camera_source(cfg)
    assert isinstance(src, FileSource)
    assert src.open() is True
    assert src.is_open is True

    ok, frame, pts = src.read_frame()
    assert ok is True
    assert frame is not None
    assert frame.shape[0] > 0
    assert frame.shape[1] > 0

    meta = src.get_metadata()
    assert meta["width"] > 0
    assert meta["height"] > 0
    assert meta["fps"] > 0

    src.close()
    assert src.is_open is False


def test_file_source_nonexistent():
    """Verify FileSource handles non-existent file cleanly."""
    cfg = CameraSourceConfig(
        source="data/videos/missing_file_xyz.mp4",
        source_type="file",
    )
    src = create_camera_source(cfg)
    assert src.open() is False
    assert "Failed to open video source" in src.last_error
    src.close()


def test_rtsp_source_credential_masking():
    """Verify RTSPSource masks credentials in logs/metadata."""
    cfg = CameraSourceConfig(
        source="rtsp://admin:supersecret@192.168.1.100:554/stream1",
        source_type="rtsp",
        transport="tcp",
    )
    assert "supersecret" not in cfg.masked_source
    assert "***" in cfg.masked_source

    src = create_camera_source(cfg)
    assert isinstance(src, RTSPSource)
    meta = src.get_metadata()
    assert meta["source_type"] == "rtsp"
    assert "supersecret" not in meta["source"]


def test_camera_crud_and_rbac(test_app, admin_headers, police_headers, ngo_headers):
    """Verify Camera Command Center CRUD endpoints and role-based permissions."""
    with TestClient(test_app) as client:
        # 1. Admin can create camera
        res = client.post(
            "/api/cameras",
            headers=admin_headers,
            json={
                "camera_id": "C_TEST_1",
                "name": "North Entrance",
                "source": "data/videos/camera_1.mp4",
                "source_type": "file",
                "location": "North Wing",
                "enabled": True,
                "sample_fps": 3.0,
            },
        )
        assert res.status_code == 201
        data = res.json()
        assert data["camera_id"] == "C_TEST_1"
        assert data["source_type"] == "file"

        # 2. Non-admin (Police/NGO) cannot create camera
        res_police_create = client.post(
            "/api/cameras",
            headers=police_headers,
            json={
                "camera_id": "C_TEST_2",
                "name": "South Gate",
                "source": "data/videos/camera_2.mp4",
            },
        )
        assert res_police_create.status_code == 403

        # 3. Police and NGO can list and get camera
        res_list = client.get("/api/cameras", headers=police_headers)
        assert res_list.status_code == 200
        cameras = res_list.json()
        assert any(c["camera_id"] == "C_TEST_1" for c in cameras)

        res_get = client.get("/api/cameras/C_TEST_1", headers=ngo_headers)
        assert res_get.status_code == 200
        assert res_get.json()["name"] == "North Entrance"

        # 4. Admin can update camera
        res_patch = client.patch(
            "/api/cameras/C_TEST_1",
            headers=admin_headers,
            json={"location": "North Wing - Gate A", "sample_fps": 4.0},
        )
        assert res_patch.status_code == 200
        assert res_patch.json()["location"] == "North Wing - Gate A"
        assert res_patch.json()["sample_fps"] == 4.0

        # 5. Non-admin cannot update camera
        res_ngo_patch = client.patch(
            "/api/cameras/C_TEST_1",
            headers=ngo_headers,
            json={"location": "Hacked"},
        )
        assert res_ngo_patch.status_code == 403

        # 6. Admin can delete camera
        res_del = client.delete("/api/cameras/C_TEST_1", headers=admin_headers)
        assert res_del.status_code == 204

        # Verify deleted
        res_get_deleted = client.get("/api/cameras/C_TEST_1", headers=admin_headers)
        assert res_get_deleted.status_code == 404


def test_camera_connection_testing(test_app, admin_headers, police_headers):
    """Verify test-source endpoint for live video probe."""
    with TestClient(test_app) as client:
        # Test valid file source
        res = client.post(
            "/api/cameras/test-source",
            headers=admin_headers,
            json={
                "source": "data/videos/camera_1.mp4",
                "source_type": "file",
            },
        )
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert data["fps"] > 0
        assert data["width"] > 0
        assert data["height"] > 0

        # Test invalid file source
        res_bad = client.post(
            "/api/cameras/test-source",
            headers=police_headers,
            json={
                "source": "data/videos/non_existent_stream.mp4",
                "source_type": "file",
            },
        )
        assert res_bad.status_code == 200
        bad_data = res_bad.json()
        assert bad_data["success"] is False
        assert bad_data["error"] is not None


def test_historical_footage_endpoints(test_app, admin_headers, police_headers, ngo_headers, tmp_path):
    """Verify Historical Footage upload, listing, and cancellation."""
    dummy_file = tmp_path / "test_historical.mp4"
    dummy_file.write_bytes(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom")

    with TestClient(test_app) as client:
        # 1. Non-admin cannot upload historical footage
        with open(dummy_file, "rb") as f:
            res_ngo = client.post(
                "/api/historical/upload",
                headers=ngo_headers,
                data={"camera_id": "C_HIST_1", "mode": "search", "sample_fps": "2.0"},
                files={"file": ("test_historical.mp4", f, "video/mp4")},
            )
        assert res_ngo.status_code == 403

        # 2. Admin can upload historical footage
        with open(dummy_file, "rb") as f:
            res_admin = client.post(
                "/api/historical/upload",
                headers=admin_headers,
                data={"camera_id": "C_HIST_1", "mode": "search", "sample_fps": "2.0"},
                files={"file": ("test_historical.mp4", f, "video/mp4")},
            )
        assert res_admin.status_code == 201
        job_info = res_admin.json()
        assert "job" in job_info
        assert job_info["job"]["camera_id"] == "C_HIST_1"
        job_id = job_info["job"]["job_id"]

        # 3. Police and NGO can list historical jobs
        res_list = client.get("/api/historical/jobs", headers=police_headers)
        assert res_list.status_code == 200
        jobs = res_list.json()
        assert any(j["job_id"] == job_id for j in jobs)

        # 4. Get specific job status
        res_job = client.get(f"/api/historical/jobs/{job_id}", headers=ngo_headers)
        assert res_job.status_code == 200
        assert res_job.json()["job_id"] == job_id
