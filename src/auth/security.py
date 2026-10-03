"""
JWT Security Utilities — token creation and validation for FindU.

Tokens are signed HS256 JWTs containing the user's username, role, and jti.
"""

import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Union

import jwt
from pwdlib import PasswordHash
from pwdlib.hashers.argon2 import Argon2Hasher
from pwdlib.hashers.bcrypt import BcryptHasher

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — pulled from environment at import time
# ---------------------------------------------------------------------------

SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "CHANGE-ME-in-production-at-least-32-chars!!!")
ALGORITHM: str = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "480"))  # 8 hours

# ---------------------------------------------------------------------------
# Password hashing context (Argon2 + Bcrypt via pwdlib)
# ---------------------------------------------------------------------------

password_hash = PasswordHash((
    Argon2Hasher(),
    BcryptHasher(),
))


def hash_password(plain: str) -> str:
    """Return secure hash (Argon2id default) of *plain* password."""
    return password_hash.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    """Return True when *plain* matches *hashed* digest."""
    return password_hash.verify(plain, hashed)


# ---------------------------------------------------------------------------
# Token creation / validation
# ---------------------------------------------------------------------------

def create_access_token(
    username: Optional[Union[dict, str]] = None,
    role: Optional[str] = None,
    expires_delta: Optional[timedelta] = None,
    data: Optional[dict] = None,
    **kwargs: Any,
) -> str:
    """
    Create a signed JWT bearer token for username / role or dictionary payload.

    Supports:
        create_access_token(username="admin", role="ADMIN")
        create_access_token("admin", "ADMIN")
        create_access_token({"sub": "admin", "role": "ADMIN"})
    """
    now = datetime.now(timezone.utc)
    expire = now + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))

    if isinstance(username, dict):
        payload = username.copy()
    elif data is not None:
        payload = data.copy()
    else:
        payload = {}
        if username is not None:
            payload["sub"] = str(username)
        if role is not None:
            payload["role"] = role

    if "sub" not in payload and "sub" in kwargs:
        payload["sub"] = str(kwargs["sub"])
    if "role" not in payload and "role" in kwargs:
        payload["role"] = str(kwargs["role"])
    if "role" not in payload:
        payload["role"] = role or "POLICE"
    if "jti" not in payload:
        payload["jti"] = str(uuid.uuid4())
    if "iat" not in payload:
        payload["iat"] = now
    if "exp" not in payload:
        payload["exp"] = expire

    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> dict:
    """
    Decode and validate a JWT bearer token.

    Raises jwt.ExpiredSignatureError or jwt.InvalidTokenError on failure.
    Returns the decoded payload dict on success.
    """
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
