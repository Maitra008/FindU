"""
Authentication REST API routes.

Endpoints:
    POST /api/auth/login   — OAuth2 password flow; returns JWT access token
    GET  /api/auth/me      — Returns currently authenticated user's profile
"""

import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from src.auth.dependencies import get_current_user
from src.auth.security import create_access_token, verify_password
from src.db.database import get_db
from src.db.repositories.users import UserRepository
from src.services.audit_service import get_audit_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


@router.post("/login", response_model=Dict[str, Any])
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    """
    Authenticate with username + password and receive a JWT bearer token.

    The token should be included in subsequent requests as:
        Authorization: Bearer <token>

    WebSocket connections should pass it as ?token=<token>.
    """
    audit = get_audit_service()
    repo = UserRepository(db)
    user = repo.get_by_username(form_data.username)

    if user is None or not verify_password(form_data.password, user.password_hash):
        # Write audit entry using the same session (so it works in tests with overridden DB)
        audit.log(
            event="login.failure",
            actor_username=form_data.username,
            detail={"reason": "invalid credentials"},
            session=db,
        )
        try:
            db.commit()
        except Exception:
            pass
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.enabled:
        audit.log(
            event="login.failure",
            actor_username=form_data.username,
            detail={"reason": "account disabled"},
            session=db,
        )
        try:
            db.commit()
        except Exception:
            pass
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled",
        )

    token = create_access_token(username=user.username, role=user.role)

    audit.log(
        event="login.success",
        actor_username=user.username,
        resource_type="user",
        resource_id=str(user.id),
        session=db,
    )
    try:
        db.commit()
    except Exception:
        pass

    logger.info("User '%s' (%s) authenticated successfully.", user.username, user.role)

    return {
        "access_token": token,
        "token_type": "bearer",
        "username": user.username,
        "role": user.role,
    }


@router.get("/me", response_model=Dict[str, Any])
def get_me(current_user=Depends(get_current_user)):
    """Return the authenticated user's profile (no password hash)."""
    return current_user.to_dict()
