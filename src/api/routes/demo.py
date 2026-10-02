"""
Controlled Demo Scenario Endpoints for Operator Portal Verification.
Allows triggering standard cross-camera movement sequences (e.g. C2 -> C3 -> C1)
via the real backend database, AlertService, and WebSocket broadcasting.
"""

import time
from typing import Any, Dict, List
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from src.db.database import get_db, get_session_factory
from src.db.models import AlertRecord, TrackRecord
from src.services.alert_service import AlertService, get_alert_service

router = APIRouter(prefix="/api/demo", tags=["Demo Scenarios"])


@router.post("/seed-movement", response_model=List[Dict[str, Any]], status_code=status.HTTP_201_CREATED)
def seed_cross_camera_movement(
    db: Session = Depends(get_db),
    alert_service: AlertService = Depends(get_alert_service),
):
    """
    Seed an explicit controlled cross-camera movement sequence:
    Person X: C2 (Parking Lot East) -> C3 (Main Lobby) -> C1 (North Entrance)
    Creates real database TrackRecord and AlertRecord entries and broadcasts them live.
    """
    now = time.time()
    steps = [
        {
            "camera_id": "C2",
            "track_id": "TRACK-0001",
            "person_id": "person_x",
            "person_name": "Person X",
            "similarity": 0.942,
            "timestamp": round(now - 120, 1),
            "frame_idx": 45,
            "bbox": [120, 80, 240, 220],
            "first_seen": round(now - 125, 1),
            "last_seen": round(now - 115, 1),
        },
        {
            "camera_id": "C3",
            "track_id": "TRACK-0004",
            "person_id": "person_x",
            "person_name": "Person X",
            "similarity": 0.915,
            "timestamp": round(now - 60, 1),
            "frame_idx": 110,
            "bbox": [140, 90, 260, 230],
            "first_seen": round(now - 65, 1),
            "last_seen": round(now - 55, 1),
        },
        {
            "camera_id": "C1",
            "track_id": "TRACK-0007",
            "person_id": "person_x",
            "person_name": "Person X",
            "similarity": 0.938,
            "timestamp": round(now - 10, 1),
            "frame_idx": 215,
            "bbox": [110, 85, 230, 225],
            "first_seen": round(now - 15, 1),
            "last_seen": round(now - 5, 1),
        },
    ]

    created_alerts = []
    for step in steps:
        # Create alert
        alert = alert_service.create_alert(
            camera_id=step["camera_id"],
            track_id=step["track_id"],
            person_id=step["person_id"],
            person_name=step["person_name"],
            similarity=step["similarity"],
            timestamp=step["timestamp"],
            frame_idx=step["frame_idx"],
            bbox=step["bbox"],
            status="NEW",
        )
        # Update corresponding track
        alert_service.update_track_state(
            camera_id=step["camera_id"],
            track_id=step["track_id"],
            person_id=step["person_id"],
            person_name=step["person_name"],
            first_seen=step["first_seen"],
            last_seen=step["last_seen"],
            first_frame=step["frame_idx"] - 10,
            last_frame=step["frame_idx"] + 10,
            detection_count=6,
            max_similarity=step["similarity"],
            mean_similarity=round(step["similarity"] - 0.015, 3),
            top_k_similarity=step["similarity"],
            threshold_matches=4,
            alert_triggered=True,
            status="TERMINATED",
        )
        created_alerts.append(alert.to_dict())

    return created_alerts

