"""
Missing Persons REST API routes for FindU.
Provides person registration, search, case details, status updates, and photo uploads.
Enforces role-based access control (ADMIN/POLICE for mutations; all authenticated roles for read).
"""

import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.auth.dependencies import get_current_user, require_role
from src.db.database import get_db
from src.db.repositories.persons import PersonRepository
from src.services.person_service import get_person_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/persons", tags=["Missing Persons & Cases"])


class PersonUpdateSchema(BaseModel):
    name: Optional[str] = None
    case_id: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    date_last_seen: Optional[str] = None
    last_known_location: Optional[str] = None
    notes: Optional[str] = None
    status: Optional[str] = Field(None, description="ACTIVE, FOUND, or CLOSED")


@router.get("", response_model=List[Dict[str, Any]])
def list_missing_persons(
    status: Optional[str] = Query(None, description="Filter by status (ACTIVE, FOUND, CLOSED)"),
    search: Optional[str] = Query(None, description="Search term for name, person ID, or case ID"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    List registered missing persons.
    Accessible to all authenticated users (ADMIN, POLICE, HOSPITAL, NGO).
    """
    repo = PersonRepository(db)
    persons = repo.get_all(status=status, search=search, limit=limit, offset=offset)
    return [p.to_dict() for p in persons]


@router.get("/{person_id}", response_model=Dict[str, Any])
def get_missing_person(
    person_id: str,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Get detailed case record for a single missing person including sightings and tracks.
    Accessible to all authenticated users.
    """
    service = get_person_service()
    details = service.get_person_details(person_id=person_id, session=db)
    if not details:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Missing person record '{person_id}' not found.",
        )
    return details


@router.post("", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
async def register_missing_person(
    name: str = Form(..., description="Full name of missing person"),
    case_id: Optional[str] = Form(None, description="Official Case Reference ID"),
    age: Optional[int] = Form(None, description="Estimated or exact age"),
    gender: Optional[str] = Form(None, description="Gender identity"),
    date_last_seen: Optional[str] = Form(None, description="Date/time last seen"),
    last_known_location: Optional[str] = Form(None, description="Last known location description"),
    notes: Optional[str] = Form(None, description="Identifying marks, clothing, or case notes"),
    photos: List[UploadFile] = File(..., description="1-10 reference photos for facial recognition"),
    db: Session = Depends(get_db),
    current_user=require_role("ADMIN", "POLICE"),
):
    """
    Register a new missing person identity:
    - Extracts ArcFace embeddings via SCRFD detector.
    - Updates the active FAISS index.
    - Persists record in database.
    - Dispatches real-time WebSocket event.

    Authorization: ADMIN and POLICE roles only.
    """
    if not photos or len(photos) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least 1 reference photograph is required.",
        )

    # Read uploaded file contents
    photo_payloads = []
    for upload in photos:
        content = await upload.read()
        if content:
            photo_payloads.append((upload.filename or "photo.jpg", content))

    if not photo_payloads:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded photo files are empty or unreadable.",
        )

    service = get_person_service()
    try:
        person = service.register_person(
            name=name,
            case_id=case_id,
            age=age,
            gender=gender,
            date_last_seen=date_last_seen,
            last_known_location=last_known_location,
            notes=notes,
            photos=photo_payloads,
            actor_username=current_user.username,
            session=db,
        )
        return person.to_dict()
    except ValueError as val_err:
        logger.warning("Registration validation error: %s", val_err)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(val_err),
        )
    except Exception as exc:
        logger.error("Failed to register person: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(exc)}",
        )


@router.patch("/{person_id}", response_model=Dict[str, Any])
def update_missing_person(
    person_id: str,
    payload: PersonUpdateSchema,
    db: Session = Depends(get_db),
    current_user=require_role("ADMIN", "POLICE"),
):
    """
    Update missing person status or case details.
    Authorization: ADMIN and POLICE roles only.
    """
    updates = payload.model_dump(exclude_unset=True)
    if "status" in updates and updates["status"]:
        valid_statuses = {"ACTIVE", "FOUND", "CLOSED"}
        st_upper = updates["status"].upper()
        if st_upper not in valid_statuses:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status '{updates['status']}'. Must be one of: {valid_statuses}",
            )
        updates["status"] = st_upper

    service = get_person_service()
    person = service.update_person(
        person_id=person_id,
        updates=updates,
        actor_username=current_user.username,
        session=db,
    )
    if not person:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Missing person record '{person_id}' not found.",
        )
    return person.to_dict()

