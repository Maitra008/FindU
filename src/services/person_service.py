"""
Person Registration & Case Management Service.
Handles reference photo storage, face embedding extraction via FaceRegistrar,
FAISS index synchronization, database persistence, and WebSocket event broadcasts.
"""

import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from sqlalchemy.orm import Session

from src.api.websocket import ws_manager
from src.config import EMBEDDINGS_DIR, PROJECT_ROOT
from src.db.database import get_session_factory
from src.db.models import AlertRecord, Person, TrackRecord
from src.db.repositories.persons import PersonRepository
from src.face_engine import FaceEngine
from src.registration import FaceRegistrar, RegistrationResult
from src.services.audit_service import get_audit_service
from src.workers.worker_manager import get_worker_manager

logger = logging.getLogger(__name__)

REFERENCE_PHOTOS_DIR = PROJECT_ROOT / "data" / "reference_photos"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _slugify(text: str) -> str:
    """Sanitize string for file paths and identifiers."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "_", text)
    return text.strip("_")


class PersonService:
    """
    Coordinates missing person registration, face embedding extraction,
    FAISS index propagation, and case management.
    """

    def __init__(
        self,
        face_engine: Optional[FaceEngine] = None,
        session_factory=None,
    ):
        self.engine = face_engine or FaceEngine()
        self.registrar = FaceRegistrar(face_engine=self.engine)
        self.session_factory = session_factory or get_session_factory()
        REFERENCE_PHOTOS_DIR.mkdir(parents=True, exist_ok=True)
        EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    def register_person(
        self,
        name: str,
        case_id: Optional[str] = None,
        age: Optional[int] = None,
        gender: Optional[str] = None,
        date_last_seen: Optional[str] = None,
        last_known_location: Optional[str] = None,
        notes: Optional[str] = None,
        photos: Optional[List[Tuple[str, bytes]]] = None,
        actor_username: Optional[str] = None,
        session: Optional[Session] = None,
    ) -> Person:
        """
        Register a new missing person:
        1. Validate metadata and photos.
        2. Save reference photos to disk.
        3. Extract face embeddings via FaceRegistrar (SCRFD + ArcFace).
        4. Update active FAISS search index.
        5. Persist Person record in DB.
        6. Record tamper-evident audit log.
        7. Broadcast WebSocket event.
        """
        if not name or not name.strip():
            raise ValueError("Person name is required.")

        clean_name = name.strip()
        slug = _slugify(clean_name) or "person"
        unique_suffix = uuid.uuid4().hex[:6]
        person_id = f"{slug}_{unique_suffix}"

        generated_case_id = case_id.strip() if (case_id and case_id.strip()) else f"CASE-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{unique_suffix.upper()}"

        if not photos or len(photos) == 0:
            raise ValueError("At least 1 clear reference photograph is required for facial recognition registration (3-10 recommended).")

        # 1. Save uploaded images to data/reference_photos/{person_id}/
        person_photo_dir = REFERENCE_PHOTOS_DIR / person_id
        person_photo_dir.mkdir(parents=True, exist_ok=True)

        saved_file_paths: List[Path] = []
        relative_photo_paths: List[str] = []

        for idx, (filename, file_bytes) in enumerate(photos, start=1):
            if not file_bytes:
                continue
            ext = Path(filename).suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                ext = ".jpg"

            safe_filename = f"ref_{idx:02d}{ext}"
            file_path = person_photo_dir / safe_filename

            with open(file_path, "wb") as f:
                f.write(file_bytes)

            saved_file_paths.append(file_path)
            relative_photo_paths.append(f"data/reference_photos/{person_id}/{safe_filename}")

        if not saved_file_paths:
            raise ValueError("No valid photo files could be saved.")

        # 2. Extract ArcFace Embedding via FaceRegistrar
        reg_result: Optional[RegistrationResult] = self.registrar.register_person(
            person_id=person_id,
            name=clean_name,
            image_dir_or_paths=saved_file_paths,
            output_dir=EMBEDDINGS_DIR,
        )

        if reg_result is None or reg_result.num_valid_embeddings == 0:
            raise ValueError(
                "Face registration failed: No clear facial features detected across the uploaded reference photos. "
                "Please upload high-quality, well-lit, frontal portrait photos."
            )

        # 3. Synchronize FAISS Index across all running CameraWorkers
        worker_mgr = get_worker_manager()
        worker_mgr.register_face_identity(
            embedding=reg_result.identity_embedding,
            person_id=person_id,
            name=clean_name,
            npz_path=str(reg_result.output_path),
        )

        # 4. Persist in Database
        def _persist(db: Session) -> Person:
            repo = PersonRepository(db)
            person = repo.create(
                person_id=person_id,
                name=clean_name,
                case_id=generated_case_id,
                age=age,
                gender=gender,
                date_last_seen=date_last_seen,
                last_known_location=last_known_location,
                notes=notes,
                status="ACTIVE",
                photo_paths=relative_photo_paths,
                embedding_path=str(reg_result.output_path),
            )

            # Audit Logging
            audit = get_audit_service()
            audit.log(
                event="person.registered",
                actor_username=actor_username,
                resource_type="person",
                resource_id=person_id,
                detail={
                    "name": clean_name,
                    "case_id": generated_case_id,
                    "valid_embeddings": reg_result.num_valid_embeddings,
                    "photos_count": len(saved_file_paths),
                },
                session=db,
            )
            try:
                db.commit()
            except Exception:
                pass

            return person

        if session is not None:
            created_person = _persist(session)
        else:
            with self.session_factory() as db:
                created_person = _persist(db)

        # 5. Broadcast WebSocket notification
        ws_manager.broadcast_sync({
            "event": "person.registered",
            "person": created_person.to_dict(),
        })

        logger.info("Successfully registered missing person: %s (ID: %s, Case: %s)", clean_name, person_id, generated_case_id)
        return created_person

    def get_person_details(self, person_id: str, session: Session) -> Optional[Dict[str, Any]]:
        """
        Fetch complete person details including sighting count, recent alerts, and track history.
        """
        repo = PersonRepository(session)
        person = repo.get_by_person_id(person_id)
        if not person:
            return None

        person_dict = person.to_dict()

        # Query associated sightings/alerts
        alerts = (
            session.query(AlertRecord)
            .filter(AlertRecord.person_id == person_id)
            .order_by(AlertRecord.created_at.desc())
            .all()
        )
        tracks = (
            session.query(TrackRecord)
            .filter(TrackRecord.person_id == person_id)
            .order_by(TrackRecord.created_at.desc())
            .all()
        )

        verified_count = sum(1 for a in alerts if a.status == "VERIFIED")
        dismissed_count = sum(1 for a in alerts if a.status == "DISMISSED")
        new_count = sum(1 for a in alerts if a.status == "NEW")

        person_dict["total_sightings"] = len(alerts)
        person_dict["verified_sightings"] = verified_count
        person_dict["dismissed_sightings"] = dismissed_count
        person_dict["new_sightings"] = new_count
        person_dict["alerts"] = [a.to_dict() for a in alerts]
        person_dict["tracks"] = [t.to_dict() for t in tracks]
        person_dict["has_embedding"] = bool(person.embedding_path and Path(person.embedding_path).exists())

        return person_dict

    def update_person(
        self,
        person_id: str,
        updates: Dict[str, Any],
        actor_username: Optional[str] = None,
        session: Optional[Session] = None,
    ) -> Optional[Person]:
        """Update metadata, notes, or status for an existing missing person."""
        def _do_update(db: Session) -> Optional[Person]:
            repo = PersonRepository(db)
            person = repo.update(person_id, **updates)
            if not person:
                return None

            audit = get_audit_service()
            audit.log(
                event="person.updated",
                actor_username=actor_username,
                resource_type="person",
                resource_id=person_id,
                detail=updates,
                session=db,
            )
            try:
                db.commit()
            except Exception:
                pass
            return person

        if session is not None:
            updated_person = _do_update(session)
        else:
            with self.session_factory() as db:
                updated_person = _do_update(db)

        if updated_person:
            ws_manager.broadcast_sync({
                "event": "person.updated",
                "person": updated_person.to_dict(),
            })

        return updated_person


_person_service_instance: Optional[PersonService] = None


def get_person_service() -> PersonService:
    """Get singleton PersonService instance."""
    global _person_service_instance
    if _person_service_instance is None:
        _person_service_instance = PersonService()
    return _person_service_instance

