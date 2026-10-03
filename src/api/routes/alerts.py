"""
Alerts and Tracks REST API routes.
Part 6: JWT authentication + role-based access control + audit logging.
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.api.websocket import ws_manager
from src.auth.dependencies import get_current_user
from src.db.database import get_db
from src.db.repositories.alerts import AlertRepository, TrackRepository
from src.services.audit_service import get_audit_service

router = APIRouter(tags=["Alerts & Tracks"])


class AlertStatusUpdateSchema(BaseModel):
    status: str = Field(..., description="Alert status: NEW, VERIFIED, or DISMISSED")


@router.get("/api/alerts", response_model=List[Dict[str, Any]])
def list_alerts(
    camera_id: Optional[str] = Query(None, description="Filter by camera ID (e.g. C1)"),
    person_id: Optional[str] = Query(None, description="Filter by person ID (e.g. person_x)"),
    status: Optional[str] = Query(None, description="Filter by status (NEW, VERIFIED, DISMISSED)"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Retrieve alerts with optional camera, person, and status filtering. Requires authentication."""
    repo = AlertRepository(db)
    alerts = repo.get_alerts(
        camera_id=camera_id,
        person_id=person_id,
        status=status,
        limit=limit,
        offset=offset,
    )
    return [a.to_dict() for a in alerts]


@router.get("/api/alerts/{alert_id}", response_model=Dict[str, Any])
def get_alert(
    alert_id: str,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Retrieve specific alert by alert_id or primary key. Requires authentication."""
    repo = AlertRepository(db)
    alert = repo.get_by_id(alert_id)
    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Alert '{alert_id}' not found",
        )
    return alert.to_dict()


@router.patch("/api/alerts/{alert_id}", response_model=Dict[str, Any])
def update_alert_status(
    alert_id: str,
    payload: AlertStatusUpdateSchema,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Update human verification status of an alert (NEW, VERIFIED, DISMISSED).

    Requires authentication. Only ADMIN and POLICE may change alert status.
    HOSPITAL and NGO users receive 403.
    """
    if current_user.role not in ("ADMIN", "POLICE"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only ADMIN and POLICE roles may update alert status.",
        )

    valid_statuses = {"NEW", "VERIFIED", "DISMISSED"}
    status_upper = payload.status.upper()
    if status_upper not in valid_statuses:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid status '{payload.status}'. Must be one of: {valid_statuses}",
        )

    repo = AlertRepository(db)
    alert = repo.update_status(alert_id, status_upper)
    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Alert '{alert_id}' not found",
        )
    alert_dict = alert.to_dict()
    ws_manager.broadcast_sync({
        "event": "alert.updated",
        "alert": alert_dict,
    })

    # Audit log
    event_name = f"alert.{status_upper.lower()}"
    audit = get_audit_service()
    audit.log(
        event=event_name,
        actor_username=current_user.username,
        resource_type="alert",
        resource_id=alert_id,
        detail={"new_status": status_upper},
        session=db,
    )
    try:
        db.commit()
    except Exception:
        pass

    return alert_dict


@router.get("/api/tracks", response_model=List[Dict[str, Any]])
def list_tracks(
    camera_id: Optional[str] = Query(None, description="Filter by camera ID"),
    status: Optional[str] = Query(None, description="Filter by status (ACTIVE, TERMINATED)"),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Retrieve historical tracks recorded across camera streams. Requires authentication."""
    repo = TrackRepository(db)
    tracks = repo.get_tracks(camera_id=camera_id, status=status, limit=limit)
    return [t.to_dict() for t in tracks]

