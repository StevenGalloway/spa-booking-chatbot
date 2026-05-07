"""
Core API routes for the AURA Platform.

Endpoints:
  POST /api/v1/chat                       — conversational AI
  GET  /api/v1/services                   — wellness service catalog
  GET  /api/v1/services/{category}        — filtered by category
  GET  /api/v1/practitioners              — practitioner profiles
  GET  /api/v1/appointments/{user_id}     — user appointment history
"""

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, Request

from app.core.logging_config import get_logger
from app.core.security import sanitize_input
from app.models.schemas import (
    AppointmentResponse,
    ChatRequest,
    ChatResponse,
    PractitionerInfo,
    ServiceInfo,
)
from app.services.chatbot_service import ChatbotService
from app.tools.appointment_tool import AppointmentTool

router = APIRouter()
logger = get_logger("aura.api.chat")

_chatbot_service = ChatbotService()
_appointment_tool = AppointmentTool()

# ---------------------------------------------------------------------------
# Practitioners (static profiles — swap for DB queries in v2)
# ---------------------------------------------------------------------------

_PRACTITIONERS: List[PractitionerInfo] = [
    PractitionerInfo(
        id=1,
        name="Sofia Marin",
        title="Lead Massage Therapist",
        specialties=["Swedish Massage", "Deep Tissue Massage", "Prenatal Massage"],
        experience_years=9,
        rating=4.9,
        bio="Sofia blends evidence-based techniques with a holistic approach to restore balance and ease chronic pain.",
        available=True,
    ),
    PractitionerInfo(
        id=2,
        name="James Okafor",
        title="Sports & Rehabilitation Specialist",
        specialties=["Sports Massage", "Trigger Point Massage", "Dry Needling"],
        experience_years=12,
        rating=4.8,
        bio="Former athletic trainer turned therapist, James specialises in injury prevention and performance recovery.",
        available=True,
    ),
    PractitionerInfo(
        id=3,
        name="Aiko Tanaka",
        title="Wellness & Mindfulness Coach",
        specialties=["Shiatsu Massage", "Reiki", "Hatha Yoga Session", "Guided Breathwork"],
        experience_years=7,
        rating=4.9,
        bio="Trained in Tokyo and Bali, Aiko weaves ancient Eastern traditions with modern mindfulness science.",
        available=True,
    ),
    PractitionerInfo(
        id=4,
        name="Lucia Ferrara",
        title="Aesthetician & Skincare Specialist",
        specialties=["Classic Facial", "Anti-Aging Facial", "Microdermabrasion", "LED Light Therapy"],
        experience_years=6,
        rating=4.7,
        bio="Lucia's personalised skin analysis protocols have earned her a loyal following across luxury wellness brands.",
        available=True,
    ),
    PractitionerInfo(
        id=5,
        name="David Chen",
        title="Licensed Acupuncturist",
        specialties=["Traditional Acupuncture", "Dry Needling", "Auricular Acupuncture", "Cupping Therapy"],
        experience_years=14,
        rating=5.0,
        bio="David holds dual certification in Traditional Chinese Medicine and Western sports medicine acupuncture.",
        available=False,
    ),
]

# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

@router.post("/chat", response_model=ChatResponse, summary="Send a message to the AURA AI")
async def chat_endpoint(request: ChatRequest, http_request: Request) -> ChatResponse:
    """
    Process a conversational message through the AURA LangGraph workflow.

    The workflow performs:
      1. Intent classification (DistilBERT + keyword fallback)
      2. Data retrieval (service catalog pricing)
      3. Appointment mutation (create / reschedule / cancel)
    """
    request_id = getattr(http_request.state, "request_id", None)
    sanitized_message = sanitize_input(request.message, max_length=2000)

    logger.info(
        "Chat request received",
        extra={
            "request_id": request_id,
            "user_id": request.user_id,
            "message_length": len(sanitized_message),
        },
    )

    try:
        response = _chatbot_service.process_message(
            message=sanitized_message,
            user_id=request.user_id,
            conversation_state=request.conversation_state or {},
        )
        response.request_id = request_id
        logger.info(
            "Chat response generated",
            extra={
                "request_id": request_id,
                "user_id": request.user_id,
                "intent": response.intent,
                "confidence": round(response.confidence, 3),
            },
        )
        return response
    except Exception as exc:
        logger.error(
            "Chat endpoint error",
            extra={"request_id": request_id, "error": str(exc)},
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Internal error processing your message")


# ---------------------------------------------------------------------------
# Service catalog
# ---------------------------------------------------------------------------

@router.get("/services", response_model=List[ServiceInfo], summary="List all wellness services")
async def get_services(
    category: Optional[str] = Query(default=None, description="Filter by category"),
) -> List[ServiceInfo]:
    """
    Returns the full AURA service catalog, optionally filtered by category.

    Categories: Massage | Yoga & Meditation | Facial & Skincare |
                Acupuncture | Hair & Beauty | Fitness & Wellness
    """
    try:
        from app.tools.data_tool import DataTool
        tool = DataTool()
        all_services = tool.get_all_services()

        if category:
            all_services = [s for s in all_services if s["category"].lower() == category.lower()]

        return [
            ServiceInfo(
                name=s["name"],
                category=s["category"],
                price=float(s["price"]),
                duration=int(s["duration"]),
                description=s["description"],
            )
            for s in all_services
        ]
    except Exception as exc:
        logger.error("Failed to load service catalog", exc_info=True)
        # Graceful degradation — return hardcoded core services
        return _FALLBACK_SERVICES


_FALLBACK_SERVICES: List[ServiceInfo] = [
    ServiceInfo(name="Swedish Massage", category="Massage", price=85, duration=60, description="Gentle full-body relaxation massage"),
    ServiceInfo(name="Deep Tissue Massage", category="Massage", price=110, duration=60, description="Targets deep muscle layers"),
    ServiceInfo(name="Hot Stone Massage", category="Massage", price=125, duration=75, description="Heated stones for deep relaxation"),
    ServiceInfo(name="Aromatherapy Massage", category="Massage", price=95, duration=60, description="Essential oils with Swedish technique"),
    ServiceInfo(name="Thai Massage", category="Massage", price=100, duration=60, description="Traditional stretching and acupressure"),
    ServiceInfo(name="Sports Massage", category="Massage", price=120, duration=60, description="Performance-focused muscle treatment"),
    ServiceInfo(name="Hatha Yoga Session", category="Yoga & Meditation", price=60, duration=60, description="Foundational yoga for strength and flexibility"),
    ServiceInfo(name="Classic Facial", category="Facial & Skincare", price=80, duration=60, description="Deep cleanse and nourishing mask"),
]


# ---------------------------------------------------------------------------
# Practitioners
# ---------------------------------------------------------------------------

@router.get("/practitioners", response_model=List[PractitionerInfo], summary="List wellness practitioners")
async def get_practitioners(
    available_only: bool = Query(default=False, description="Return only currently available practitioners"),
) -> List[PractitionerInfo]:
    """Returns AURA practitioner profiles. Filter by availability for real-time scheduling."""
    if available_only:
        return [p for p in _PRACTITIONERS if p.available]
    return _PRACTITIONERS


# ---------------------------------------------------------------------------
# Appointments
# ---------------------------------------------------------------------------

@router.get(
    "/appointments/{user_id}",
    response_model=List[AppointmentResponse],
    summary="Retrieve appointment history for a user",
)
async def get_user_appointments(user_id: str) -> List[AppointmentResponse]:
    try:
        raw_appointments = _appointment_tool.get_appointments(user_id)
        result: List[AppointmentResponse] = []

        for appt in raw_appointments:
            appt_id, appt_user_id, service_type, date_time, status, *_ = appt

            if not date_time or date_time == "Not extracted":
                date_part, time_part = "TBD", "TBD"
                created_at = datetime.now()
            elif " " in date_time and ":" in date_time:
                date_part, time_part = date_time.split(" ", 1)
                try:
                    created_at = datetime.fromisoformat(date_time.replace(" ", "T"))
                except ValueError:
                    created_at = datetime.now()
            else:
                date_part = date_time
                time_part = ""
                created_at = datetime.now()

            result.append(
                AppointmentResponse(
                    id=appt_id,
                    user_id=appt_user_id,
                    service_type=service_type,
                    date=date_part,
                    time=time_part,
                    status=status,
                    created_at=created_at,
                )
            )
        return result
    except Exception as exc:
        logger.error("Failed to fetch appointments", extra={"user_id": user_id}, exc_info=True)
        raise HTTPException(status_code=500, detail="Error fetching appointments")
