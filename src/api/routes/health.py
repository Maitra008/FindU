"""
Health check endpoints.
Verifies system health and database connectivity.
"""

from fastapi import APIRouter, Response, status
from src.db.database import check_db_connection

router = APIRouter(tags=["Health"])


@router.get("/health")
@router.get("/api/health")
def health_check(response: Response):
    """
    Health check endpoint.
    Distinguishes application runtime health from database connectivity failure.
    """
    db_ok = check_db_connection()
    if not db_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {
            "status": "degraded",
            "application": "healthy",
            "database": "disconnected",
        }

    return {
        "status": "ok",
        "application": "healthy",
        "database": "connected",
    }

