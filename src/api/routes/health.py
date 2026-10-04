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


@router.get("/api/diagnostic/runtime")
def diagnostic_runtime():
    """Read-only diagnostic endpoint for runtime object identity and index state."""
    from src.face_engine import get_face_engine
    from src.index import get_face_index
    from src.workers.worker_manager import get_worker_manager

    eng = get_face_engine()
    idx = get_face_index()
    wm = get_worker_manager()

    meta_pids = [m.get("person_id") for m in idx.metadata]

    worker_info = {}
    for cid, w in wm.workers.items():
        w_pids = [m.get("person_id") for m in w.index.metadata] if w.index else []
        worker_info[cid] = {
            "is_running": w.is_running,
            "engine_id": id(w.engine),
            "index_id": id(w.index),
            "ntotal": w.index.total_identities if w.index else 0,
            "metadata_count": len(w.index.metadata) if w.index else 0,
            "partho_b55022_in_worker_index": "partho_b55022" in w_pids,
            "worker_person_ids": w_pids,
        }

    return {
        "engine_id": id(eng),
        "index_id": id(idx),
        "index_ntotal": idx.total_identities,
        "metadata_count": len(idx.metadata),
        "partho_b55022_in_canonical_index": "partho_b55022" in meta_pids,
        "canonical_person_ids": meta_pids,
        "workers": worker_info,
    }


