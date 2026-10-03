"""
AuditLog service — write-only append interface for the audit trail.

Rules:
  * NEVER log passwords, tokens, face embeddings, or raw video frames.
  * All detail fields must be JSON-safe strings.
  * Each call is a single INSERT; reads never go through this service.
"""

import json
import logging
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from src.db.database import get_session_factory
from src.db.models import AuditLog

logger = logging.getLogger(__name__)


class AuditService:
    """Append-only service for writing audit log entries."""

    def __init__(self, session_factory=None) -> None:
        self._session_factory = session_factory

    @property
    def session_factory(self):
        return self._session_factory or get_session_factory()

    def log(
        self,
        event: str,
        actor_username: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        detail: Optional[Dict[str, Any]] = None,
        session: Optional[Session] = None,
    ) -> None:
        """
        Write one audit log entry.

        Args:
            event:          Event type slug (e.g. 'login.success', 'alert.verified').
            actor_username: Username who performed the action. None for system events.
            resource_type:  Entity type affected (e.g. 'alert', 'camera', 'user').
            resource_id:    Entity identifier (e.g. alert_id, camera_id).
            detail:         Additional context as a plain dict — must be JSON-serialisable.
                            Do NOT include passwords, tokens, or embeddings.
            session:        Optional existing SQLAlchemy session. When provided,
                            the entry is added to the session but NOT committed here.
                            This allows the caller to commit in one transaction.
                            When None, a new session is opened and auto-committed.
        """
        detail_str: Optional[str] = None
        if detail:
            try:
                detail_str = json.dumps(detail)
            except (TypeError, ValueError) as exc:
                logger.warning("AuditService: detail is not JSON-serialisable (%s); omitting.", exc)

        entry = AuditLog(
            event=event,
            actor_username=actor_username,
            resource_type=resource_type,
            resource_id=str(resource_id) if resource_id is not None else None,
            detail=detail_str,
        )

        if session is not None:
            try:
                session.add(entry)
                session.flush()  # Write to DB within caller's transaction (not committed yet)
            except Exception as exc:
                logger.error("AuditService: failed to write audit log entry '%s' via session: %s", event, exc)
        else:
            try:
                with self.session_factory() as sess:
                    sess.add(entry)
                    sess.commit()
            except Exception as exc:
                # Audit failures must NEVER crash the main application flow.
                logger.error("AuditService: failed to write audit log entry '%s': %s", event, exc)


# Module-level singleton
_audit_service: Optional[AuditService] = None


def get_audit_service() -> AuditService:
    """Return the singleton AuditService instance."""
    global _audit_service
    if _audit_service is None:
        _audit_service = AuditService()
    return _audit_service
