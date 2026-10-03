"""
Media REST API routes for serving reference photos, CCTV snapshots, and evidence crops.
Includes strict path traversal guards.
"""

import logging
import os
from pathlib import Path
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse

from src.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/media", tags=["Media & Evidence"])

REFERENCE_DIR = (PROJECT_ROOT / "data" / "reference_photos").resolve()
SNAPSHOTS_DIR = (PROJECT_ROOT / "data" / "snapshots").resolve()


def _safe_resolve(base_dir: Path, subpath: str) -> Path:
    """Resolve subpath within base_dir and verify no directory traversal."""
    # Prevent traversal
    clean_subpath = os.path.normpath(subpath).lstrip("\\/.")
    target_path = (base_dir / clean_subpath).resolve()

    if not str(target_path).startswith(str(base_dir)):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access forbidden: Invalid path.",
        )
    return target_path


@router.get("/reference/{person_id}/{filename}")
def get_reference_photo(person_id: str, filename: str):
    """Serve a registered reference photograph."""
    target_path = _safe_resolve(REFERENCE_DIR, f"{person_id}/{filename}")
    if not target_path.exists() or not target_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Reference photo not found.",
        )
    return FileResponse(target_path)


@router.get("/snapshots/{filename}")
def get_snapshot(filename: str):
    """Serve an alert snapshot or CCTV frame."""
    target_path = _safe_resolve(SNAPSHOTS_DIR, filename)
    if not target_path.exists() or not target_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Snapshot file not found.",
        )
    return FileResponse(target_path)

