"""
Security utilities for the AURA Platform.

Production hardening checklist:
  - Rotate SECRET_KEY and store in a secrets manager (AWS Secrets Manager, Vault)
  - Switch to RS256 (asymmetric) so the public key can be shared with services
  - Integrate with an OIDC provider (Auth0, Cognito, Okta) for real user auth
  - Enable AUTH_ENABLED=true and add the get_current_user dependency to routes
"""

import re
from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger("aura.security")

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Characters that signal injection attempts — log and strip
_DANGEROUS_PATTERN = re.compile(r"[<>{};\"\'\\]")


def hash_password(password: str) -> str:
    return _pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return _pwd_context.verify(plain, hashed)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    payload = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.access_token_expire_minutes)
    )
    payload.update({"exp": expire, "type": "access"})
    return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)


def create_refresh_token(data: dict) -> str:
    payload = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(days=settings.refresh_token_expire_days)
    payload.update({"exp": expire, "type": "refresh"})
    return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)


def decode_token(token: str) -> dict:
    """
    Decode and validate a JWT.  Raises ValueError on any failure so callers
    can convert it to an appropriate HTTP error without leaking internals.
    """
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
        return payload
    except JWTError as exc:
        raise ValueError(f"Invalid token: {exc}") from exc


def sanitize_input(text: str, max_length: int = 2000) -> str:
    """
    Sanitize free-text input before it enters the NLP pipeline.

    - Truncates to max_length to prevent prompt-injection via very long inputs
    - Strips leading/trailing whitespace
    - Logs if potentially dangerous characters are detected (does not block)
    """
    if not text:
        return ""

    text = text[:max_length].strip()

    if _DANGEROUS_PATTERN.search(text):
        logger.warning("Potentially unsafe characters detected in user input")

    return text
