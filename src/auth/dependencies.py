"""
FastAPI dependency factories for authentication and role-based access control.

Usage in routes:
    from src.auth.dependencies import require_authenticated, require_role

    @router.get("/api/secure")
    def secure(current_user = Depends(require_authenticated)):
        ...

    @router.post("/api/admin-only")
    def admin_only(current_user = Depends(require_role("ADMIN"))):
        ...
"""

import logging
from typing import Optional

import jwt
from fastapi import Depends, HTTPException, Query, WebSocket, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from src.auth.security import decode_access_token
from src.db.database import get_db
from src.db.repositories.users import UserRepository

logger = logging.getLogger(__name__)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


def _decode_token(token: str) -> dict:
    """Decode JWT and raise HTTP 401 on any failure."""
    try:
        return decode_access_token(token)
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
):
    """
    Decode the JWT bearer token and return the active User ORM object.
    Raises HTTP 401 if token is invalid/expired or user not found/disabled.
    """
    payload = _decode_token(token)
    username: Optional[str] = payload.get("sub")
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing subject claim")

    repo = UserRepository(db)
    user = repo.get_by_username(username)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    if not user.enabled:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User account is disabled")
    return user


# Alias for cleaner route signatures
require_authenticated = Depends(get_current_user)


def require_role(*allowed_roles: str):
    """
    Returns a FastAPI dependency that requires the current user to hold one of *allowed_roles*.

    Example:
        @router.delete("/api/cameras/{id}", dependencies=[require_role("ADMIN")])
    """
    def _check_role(current_user=Depends(get_current_user)):
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires role: {' or '.join(allowed_roles)}. Your role: {current_user.role}",
            )
        return current_user
    return Depends(_check_role)


# ---------------------------------------------------------------------------
# WebSocket token validation (token passed as query param ?token=<jwt>)
# ---------------------------------------------------------------------------

async def get_ws_user(websocket: WebSocket, db: Session = Depends(get_db)):
    """
    Authenticate a WebSocket connection using the ?token= query parameter.
    Closes the connection with code 4001 on auth failure.
    """
    token: Optional[str] = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=4001, reason="Missing authentication token")
        return None

    try:
        payload = decode_access_token(token)
    except jwt.ExpiredSignatureError:
        await websocket.close(code=4001, reason="Token expired")
        return None
    except jwt.InvalidTokenError:
        await websocket.close(code=4001, reason="Invalid token")
        return None

    username = payload.get("sub")
    if not username:
        await websocket.close(code=4001, reason="Invalid token payload")
        return None

    repo = UserRepository(db)
    user = repo.get_by_username(username)
    if user is None or not user.enabled:
        await websocket.close(code=4001, reason="User not found or disabled")
        return None

    return user
