import os
import pytest
from starlette.testclient import TestClient
from src.api.app import app
from src.db.database import init_db

def test_frontend_dist_files_exist():
    """Verify that frontend bundle has been built and index.html/assets exist."""
    frontend_dist = os.path.join(os.getcwd(), "frontend", "dist")
    assert os.path.isdir(frontend_dist), "frontend/dist directory must exist"
    index_html = os.path.join(frontend_dist, "index.html")
    assert os.path.isfile(index_html), "frontend/dist/index.html must exist"

def test_portal_api_and_demo_e2e():
    """Verify backend API endpoints supporting the Part 5 Portal workflow."""
    init_db()
    client = TestClient(app)

    # 1. Health check
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

    # 2. Get cameras
    resp = client.get("/api/cameras")
    assert resp.status_code == 200
    cameras = resp.json()
    assert len(cameras) >= 4
    cam_ids = {c["camera_id"] for c in cameras}
    assert {"C1", "C2", "C3", "C4"}.issubset(cam_ids)

    # 3. Seed demo movement (C2 -> C3 -> C1)
    resp = client.post("/api/demo/seed-movement")
    assert resp.status_code in (200, 201)
    alerts = resp.json()
    assert len(alerts) == 3
    
    # Verify sequence C2 -> C3 -> C1
    cams_in_alerts = [a["camera_id"] for a in alerts]
    assert cams_in_alerts == ["C2", "C3", "C1"]

    # 4. Query tracks for person_id
    target_person_id = alerts[0]["person_id"]
    resp = client.get(f"/api/tracks?person_id={target_person_id}")
    assert resp.status_code == 200
    tracks = resp.json()
    assert len(tracks) == 3
    track_cams = [t["camera_id"] for t in sorted(tracks, key=lambda x: x["first_seen"])]
    assert track_cams == ["C2", "C3", "C1"]

    # 5. Verify human operator verification: Confirm (VERIFIED)
    first_alert_id = alerts[0]["id"]
    patch_resp = client.patch(f"/api/alerts/{first_alert_id}", json={"status": "VERIFIED"})
    assert patch_resp.status_code == 200
    assert patch_resp.json()["status"] == "VERIFIED"

    # 6. Verify human operator rejection: Dismiss (DISMISSED)
    second_alert_id = alerts[1]["id"]
    patch_resp2 = client.patch(f"/api/alerts/{second_alert_id}", json={"status": "DISMISSED"})
    assert patch_resp2.status_code == 200
    assert patch_resp2.json()["status"] == "DISMISSED"

