"""
Repository for Person database queries and mutations.
"""

import json
import logging
from typing import Any, Dict, List, Optional
from sqlalchemy import or_
from sqlalchemy.orm import Session

from src.db.models import Person, get_utc_now

logger = logging.getLogger(__name__)


class PersonRepository:
    """Encapsulates CRUD operations for Person database entities."""

    def __init__(self, session: Session):
        self.session = session

    def get_all(
        self,
        status: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Person]:
        """Query persons with optional status filter and search term (matches name, person_id, or case_id)."""
        query = self.session.query(Person)

        if status:
            query = query.filter(Person.status == status.upper())

        if search and search.strip():
            term = f"%{search.strip()}%"
            query = query.filter(
                or_(
                    Person.name.ilike(term),
                    Person.person_id.ilike(term),
                    Person.case_id.ilike(term),
                    Person.last_known_location.ilike(term),
                )
            )

        return query.order_by(Person.created_at.desc()).offset(offset).limit(limit).all()

    def get_by_person_id(self, person_id: str) -> Optional[Person]:
        """Fetch single person by person_id string."""
        return self.session.query(Person).filter(Person.person_id == person_id).first()

    def get_by_case_id(self, case_id: str) -> Optional[Person]:
        """Fetch single person by case_id string."""
        return self.session.query(Person).filter(Person.case_id == case_id).first()

    def create(
        self,
        person_id: str,
        name: str,
        case_id: Optional[str] = None,
        age: Optional[int] = None,
        gender: Optional[str] = None,
        date_last_seen: Optional[str] = None,
        last_known_location: Optional[str] = None,
        notes: Optional[str] = None,
        status: str = "ACTIVE",
        photo_paths: Optional[List[str]] = None,
        embedding_path: Optional[str] = None,
    ) -> Person:
        """Create and persist a new Person record."""
        photos_json = json.dumps(photo_paths or [])
        person = Person(
            person_id=person_id,
            name=name,
            case_id=case_id,
            age=age,
            gender=gender,
            date_last_seen=date_last_seen,
            last_known_location=last_known_location,
            notes=notes,
            status=status.upper(),
            photo_paths=photos_json,
            embedding_path=embedding_path,
            is_active=(status.upper() == "ACTIVE"),
        )
        self.session.add(person)
        self.session.commit()
        self.session.refresh(person)
        logger.info("Persisted new Person: '%s' (%s, case: %s)", name, person_id, case_id)
        return person

    def update(self, person_id: str, **kwargs: Any) -> Optional[Person]:
        """Update fields of an existing person record."""
        person = self.get_by_person_id(person_id)
        if not person:
            return None

        for key, value in kwargs.items():
            if value is not None and hasattr(person, key):
                if key == "status":
                    value = str(value).upper()
                    person.is_active = (value == "ACTIVE")
                elif key == "photo_paths" and isinstance(value, list):
                    value = json.dumps(value)
                setattr(person, key, value)

        person.updated_at = get_utc_now()
        self.session.commit()
        self.session.refresh(person)
        logger.info("Updated Person: %s", person_id)
        return person

    def add_photos(self, person_id: str, new_photo_paths: List[str]) -> Optional[Person]:
        """Append new photo paths to an existing person record."""
        person = self.get_by_person_id(person_id)
        if not person:
            return None

        current_photos: List[str] = []
        if person.photo_paths:
            try:
                current_photos = json.loads(person.photo_paths) if isinstance(person.photo_paths, str) else person.photo_paths
            except Exception:
                current_photos = []

        for p in new_photo_paths:
            if p not in current_photos:
                current_photos.append(p)

        person.photo_paths = json.dumps(current_photos)
        person.updated_at = get_utc_now()
        self.session.commit()
        self.session.refresh(person)
        return person

    def count(self, status: Optional[str] = None) -> int:
        """Count total persons matching status filter."""
        query = self.session.query(Person)
        if status:
            query = query.filter(Person.status == status.upper())
        return query.count()

