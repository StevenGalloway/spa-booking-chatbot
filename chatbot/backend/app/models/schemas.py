from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Shared envelope
# ---------------------------------------------------------------------------

class APIResponse(BaseModel):
    """Standard response wrapper used by all non-chat endpoints."""
    data: Optional[Any] = None
    meta: Dict[str, Any] = {}
    errors: List[str] = []


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: Optional[datetime] = None


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    user_id: str = Field(..., min_length=1, max_length=128)
    conversation_state: Optional[Dict[str, Any]] = Field(default_factory=dict)

    @field_validator("message")
    @classmethod
    def strip_message(cls, v: str) -> str:
        return v.strip()

    @field_validator("user_id")
    @classmethod
    def strip_user_id(cls, v: str) -> str:
        return v.strip()


class ChatResponse(BaseModel):
    response: str
    intent: str
    confidence: float = Field(ge=0.0, le=1.0)
    conversation_state: Dict[str, Any]
    timestamp: datetime
    request_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Appointments
# ---------------------------------------------------------------------------

class AppointmentCreate(BaseModel):
    service_type: str = Field(..., min_length=1, max_length=200)
    date: str
    time: str
    duration: Optional[int] = Field(default=60, ge=15, le=480)
    notes: Optional[str] = Field(default=None, max_length=500)


class AppointmentResponse(BaseModel):
    id: int
    user_id: str
    service_type: str
    date: str
    time: str
    status: str
    created_at: datetime


# ---------------------------------------------------------------------------
# Services & Practitioners
# ---------------------------------------------------------------------------

class ServiceInfo(BaseModel):
    name: str
    category: str
    price: float
    duration: int
    description: str


class PractitionerInfo(BaseModel):
    id: int
    name: str
    title: str
    specialties: List[str]
    experience_years: int
    rating: float = Field(ge=0.0, le=5.0)
    bio: str
    available: bool = True


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class GuestTokenRequest(BaseModel):
    user_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class HealthCheck(BaseModel):
    status: str
    version: str
    environment: str
    uptime_seconds: float
    checks: Dict[str, str]
    timestamp: datetime
