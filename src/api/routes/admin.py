"""
Administrative & System REST API routes.
Provides controlled test environment reset capabilities strictly for ADMIN operators.
"""

import logging
from typing import Any, Dict
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.auth.dependencies import require_role
from src.db.database import get_db
from src.services.admin_service import AdminService, get_admin_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin", tags=["System Administration"])


@router.post("/reset-test-environment", response_model=Dict[str, Any], status_code=status.HTTP_200_OK)
def reset_test_environment(
    db: Session = Depends(get_db),
    admin_svc: AdminService = Depends(get_admin_service),
    current_user=require_role("ADMIN"),
):
    """
    Controlled administrative reset for the test environment.
    Strictly restricted to users with the 'ADMIN' role.

    Cleans:
    - Registered test persons and reference photos
    - Alert records and track/sighting history
    - Historical background jobs and uploaded footage
    - Dynamic embeddings while preserving baseline identities
    - Synchronizes FAISS index and restarts camera workers

    Preserves:
    - User accounts and authentication data
    - Camera configurations
    - Baseline/static demo identities and models
    """
    try:
        result = admin_svc.reset_test_environment(
            actor_username=current_user.username,
            session=db,
        )
        return result
    except RuntimeError as r_err:
        logger.warning("Reset conflict: %s", r_err)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(r_err),
        )
    except Exception as exc:
        logger.error("Failed to reset test environment: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Test environment reset failed: {str(exc)}",
        )
