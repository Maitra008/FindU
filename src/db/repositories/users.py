"""
User repository for FindU authentication.
Provides CRUD operations on the users table.
"""

import logging
from typing import List, Optional

from sqlalchemy.orm import Session

from src.db.models import User

logger = logging.getLogger(__name__)


class UserRepository:
    """Data-access layer for the User model."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def get_by_username(self, username: str) -> Optional[User]:
        """Return User by username or None."""
        return self.session.query(User).filter(User.username == username).first()

    def get_by_id(self, user_id: int) -> Optional[User]:
        """Return User by primary key or None."""
        return self.session.query(User).filter(User.id == user_id).first()

    def list_users(self) -> List[User]:
        """Return all users ordered by id."""
        return self.session.query(User).order_by(User.id).all()

    def create_user(
        self,
        username: str,
        password_hash: str,
        role: str = "POLICE",
        enabled: bool = True,
    ) -> User:
        """Persist a new user and return it."""
        user = User(
            username=username,
            password_hash=password_hash,
            role=role,
            enabled=enabled,
        )
        self.session.add(user)
        self.session.commit()
        self.session.refresh(user)
        return user

    def ensure_user_exists(
        self,
        username: str,
        password_hash: str,
        role: str,
    ) -> User:
        """Create user only if username does not already exist. Safe for init_db seeding."""
        existing = self.get_by_username(username)
        if existing:
            return existing
        return self.create_user(username=username, password_hash=password_hash, role=role)
