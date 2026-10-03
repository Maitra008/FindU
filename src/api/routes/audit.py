"""
Audit Log REST API routes.

Endpoints:
    GET /api/audit   — List audit log entries (ADMIN only)
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from src.auth.dependencies import require_role
from src.db.database import get_db
from src.db.models import AuditLog

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/audit", tags=["Audit"])


@router.get("", response_model=List[Dict[str, Any]])
def list_audit_logs(
    event: Optional[str] = Query(None, description="Filter by event type"),
    actor_username: Optional[str] = Query(None, description="Filter by actor username"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    _current_user=require_role("ADMIN"),
):
    """
    List audit log entries. ADMIN role required.

    Supports filtering by event type and actor username.
    """
    query = db.query(AuditLog)
    if event:
        query = query.filter(AuditLog.event == event)
    if actor_username:
        query = query.filter(AuditLog.actor_username == actor_username)

    entries = (
        query.order_by(AuditLog.id.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return [e.to_dict() for e in entries]
