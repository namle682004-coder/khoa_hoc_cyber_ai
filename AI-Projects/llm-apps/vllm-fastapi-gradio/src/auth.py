"""Simple demo auth dependency."""
from __future__ import annotations

from fastapi import Header, HTTPException

from .config import settings


def verify_bearer_token(authorization: str | None = Header(default=None)) -> bool:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = authorization.split(" ", 1)[1].strip()
    if token != settings.demo_token:
        raise HTTPException(status_code=401, detail="Invalid token")
    return True
