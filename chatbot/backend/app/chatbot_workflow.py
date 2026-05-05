"""
AURA LangGraph Workflow.

State machine with three sequential nodes:

  START → intent_analysis → data_retrieval → appointment_trigger → END

Intent classification uses a two-tier strategy:
  Tier 1: DistilBERT fine-tuned on 240 labeled wellness-booking examples
  Tier 2: Keyword rules (always-on override for high-confidence patterns)

The keyword layer ensures that safety-critical intents (cancel, reschedule)
are never misrouted by model uncertainty.
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.core.logging_config import get_logger
from app.tools.appointment_tool import AppointmentTool
from app.tools.data_tool import DataTool
from app.tools.inference_tool import InferenceTool

logger = get_logger("aura.workflow")


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class ChatState(TypedDict):
    query: str
    intent: str
    confidence: float
    response: str
    appointment_action: str
    datetime: str
    conversation_state: dict


# ---------------------------------------------------------------------------
# Tool singletons (module-level — initialized once per worker process)
# ---------------------------------------------------------------------------

_inference_tool = InferenceTool()
_appointment_tool = AppointmentTool()
_data_tool = DataTool()

# ---------------------------------------------------------------------------
# Keyword maps (ordered — more specific entries first)
# ---------------------------------------------------------------------------

_CANCEL_KEYWORDS = {
    "cancel", "cancellation", "cancelling", "canceling", "cancelled",
    "remove booking", "delete booking", "remove appointment", "delete appointment",
    "want to cancel", "need to cancel", "i want to cancel", "i need to cancel",
}

_RESCHEDULE_KEYWORDS = {
    "reschedule", "rescheduling", "change my booking", "move my booking",
    "shift my booking", "postpone", "change the date", "change the time",
    "want to reschedule", "need to reschedule",
}

_BOOKING_KEYWORDS = {
    "book", "schedule", "appointment", "reserve", "reservation",
    "set up", "make", "create", "arrange",
    "i want", "i need", "i'd like", "i would like", "can i get",
    "can i have", "i'm looking for", "looking to",
    "want to book", "need to book", "get me", "book me",
}

_SERVICE_KEYWORDS = {
    # Massage
    "massage", "thai", "swedish", "deep tissue", "hot stone",
    "neck", "shoulder", "aromatherapy", "sports", "prenatal", "postnatal",
    "reflexology", "full body", "relaxation", "shiatsu", "trigger point",
    "lymphatic", "craniosacral", "myofascial", "cupping", "reiki",
    "couples", "chair", "foot", "back", "head", "scalp", "watsu",
    "lomi", "balinese", "ayurvedic", "indian head", "stone", "bamboo",
    "four hands", "geriatric", "oncology", "therapeutic",
    # Yoga & Meditation
    "yoga", "hatha", "vinyasa", "yin", "meditation", "breathwork",
    "sound bath", "mindfulness",
    # Facial & Skincare
    "facial", "skincare", "hydrating", "anti-aging", "microdermabrasion",
    "led light", "chemical peel",
    # Acupuncture
    "acupuncture", "dry needling", "auricular",
    # Hair & Beauty
    "keratin", "lash", "brow", "hair restoration",
    # Fitness
    "personal training", "nutrition", "body composition",
}

_PRICING_KEYWORDS = {
    "price", "cost", "how much", "pricing", "fee", "charge", "rates", "rate",
    "expensive", "affordable", "budget",
}

_STATUS_KEYWORDS = {
    "status", "check", "view", "show", "my booking", "my appointment",
    "what do i have", "upcoming", "list",
}

_GREETING_KEYWORDS = {
    "hello", "hi", "hey", "greetings", "good morning",
    "good afternoon", "good evening", "howdy", "what's up",
}

_THANKS_KEYWORDS = {
    "thanks", "thank you", "appreciate", "thank", "grateful", "cheers",
}

# ---------------------------------------------------------------------------
# Service detection map  (keyword → canonical service name)
# ---------------------------------------------------------------------------

_SERVICE_MAP: list[tuple[str, str]] = [
    # Multi-word matches first (most specific)
    ("hot stone", "Hot Stone Massage"),
    ("deep tissue", "Deep Tissue Massage"),
    ("neck and shoulder", "Neck and Shoulder Massage"),
    ("full body", "Full Body Relaxation"),
    ("four hands", "Four Hands Massage"),
    ("lomi lomi", "Lomi Lomi Massage"),
    ("indian head", "Indian Head Massage"),
    ("warm bamboo", "Warm Bamboo Massage"),
    ("trigger point", "Trigger Point Massage"),
    ("lymphatic drainage", "Lymphatic Drainage Massage"),
    ("dry needling", "Dry Needling"),
    ("sound bath", "Sound Bath Meditation"),
    ("yin yoga", "Yin Yoga"),
    ("anti-aging", "Anti-Aging Facial"),
    ("anti aging", "Anti-Aging Facial"),
    ("hydrating facial", "Hydrating Facial"),
    ("classic facial", "Classic Facial"),
    ("led light", "LED Light Therapy"),
    ("chemical peel", "Chemical Peel"),
    ("hair restoration", "Hair Restoration Therapy"),
    ("personal training", "Personal Training Session"),
    ("body composition", "Body Composition Analysis"),
    # Single-word matches
    ("aromatherapy", "Aromatherapy Massage"),
    ("swedish", "Swedish Massage"),
    ("sports", "Sports Massage"),
    ("prenatal", "Prenatal Massage"),
    ("postnatal", "Postnatal Massage"),
    ("thai", "Thai Massage"),
    ("reflexology", "Reflexology"),
    ("shiatsu", "Shiatsu Massage"),
    ("craniosacral", "Craniosacral Therapy"),
    ("myofascial", "Myofascial Release"),
    ("cupping", "Cupping Therapy"),
    ("reiki", "Reiki"),
    ("couples", "Couples Massage"),
    ("chair", "Chair Massage"),
    ("watsu", "Watsu Massage"),
    ("balinese", "Balinese Massage"),
    ("ayurvedic", "Ayurvedic Massage"),
    ("bamboo", "Warm Bamboo Massage"),
    ("geriatric", "Geriatric Massage"),
    ("oncology", "Oncology Massage"),
    ("therapeutic", "Therapeutic Massage"),
    ("hatha", "Hatha Yoga Session"),
    ("vinyasa", "Vinyasa Flow Yoga"),
    ("breathwork", "Guided Breathwork"),
    ("mindfulness", "Mindfulness Meditation"),
    ("microdermabrasion", "Microdermabrasion"),
    ("acupuncture", "Traditional Acupuncture"),
    ("auricular", "Auricular Acupuncture"),
    ("keratin", "Keratin Treatment"),
    ("lash", "Lash Extensions"),
    ("brow", "Brow Shaping and Tint"),
    ("nutrition", "Nutrition Consultation"),
    ("meditation", "Mindfulness Meditation"),
    ("yoga", "Hatha Yoga Session"),
    ("facial", "Classic Facial"),
    # Broad fallbacks
    ("foot", "Foot Massage"),
    ("back", "Back Massage"),
    ("scalp", "Head and Scalp Massage"),
    ("neck", "Neck and Shoulder Massage"),
    ("shoulder", "Neck and Shoulder Massage"),
    ("massage", "Swedish Massage"),
]


def _detect_service(query_lower: str) -> str:
    """Return the best-matching canonical service name, or 'Swedish Massage' as default."""
    for keyword, service_name in _SERVICE_MAP:
        if keyword in query_lower:
            return service_name
    return "Swedish Massage"


def _any_keyword(query_lower: str, keywords: set) -> bool:
    return any(kw in query_lower for kw in keywords)


# ---------------------------------------------------------------------------
# Node 1: Intent Analysis
# ---------------------------------------------------------------------------

def intent_analysis(state: ChatState) -> ChatState:
    query = state["query"]
    query_lower = query.lower()
    conv_state = state.get("conversation_state", {})

    # --- Tier 1: safety overrides (cancel / reschedule must never be misrouted) ---
    has_cancel = _any_keyword(query_lower, _CANCEL_KEYWORDS)
    has_reschedule = _any_keyword(query_lower, _RESCHEDULE_KEYWORDS)

    if has_cancel:
        state["intent"] = "cancel_booking"
        state["confidence"] = 0.95
        state["response"] = "I can help you cancel your appointment. Let me look up your bookings."
        return state

    if has_reschedule:
        state["intent"] = "reschedule_booking"
        state["confidence"] = 0.95
        state["response"] = "Sure, let's reschedule. I'll find your appointment details."
        return state

    # --- Tier 2: ML model (with graceful fallback) ---
    ml_intent, ml_confidence = "unknown", 0.0
    try:
        result = _inference_tool.predict_and_respond(query)
        ml_intent = result["intent"]
        ml_confidence = float(result["confidence"])
        state["intent"] = ml_intent
        state["confidence"] = ml_confidence
        state["response"] = result["response"]
    except Exception:
        # Model not loaded — keyword fallback covers all intents below
        pass

    # --- Tier 3: keyword rules (supplement or override low-confidence ML) ---
    threshold = 0.70

    if ml_confidence < threshold or ml_intent == "unknown":
        has_booking = _any_keyword(query_lower, _BOOKING_KEYWORDS)
        has_service = _any_keyword(query_lower, _SERVICE_KEYWORDS)
        has_pricing = _any_keyword(query_lower, _PRICING_KEYWORDS)
        has_status = _any_keyword(query_lower, _STATUS_KEYWORDS)
        has_greeting = _any_keyword(query_lower, _GREETING_KEYWORDS)
        has_thanks = _any_keyword(query_lower, _THANKS_KEYWORDS)

        if (has_booking or has_service) and has_pricing:
            state["intent"] = "pricing_inquiry"
            state["confidence"] = 0.88
            state["response"] = "Let me find that pricing for you."
        elif has_booking or has_service:
            state["intent"] = "book_service"
            state["confidence"] = 0.88
            state["response"] = "I'd be happy to help you book a session!"
        elif has_pricing:
            state["intent"] = "pricing_inquiry"
            state["confidence"] = 0.88
            state["response"] = "Let me check our service pricing."
        elif has_status:
            state["intent"] = "booking_status"
            state["confidence"] = 0.88
            state["response"] = "Let me pull up your appointments."
        elif has_greeting:
            state["intent"] = "greeting"
            state["confidence"] = 0.95
            state["response"] = (
                "Welcome to AURA! I can help you book wellness sessions, "
                "check pricing, view appointments, or reschedule. How can I help?"
            )
        elif has_thanks:
            state["intent"] = "thanks"
            state["confidence"] = 0.95
            state["response"] = "You're very welcome! Is there anything else I can help you with?"

    # --- Conversation state: completing in-progress flows ---

    # Completing a booking that was waiting for date/time
    if conv_state.get("pending_service"):
        try:
            dt = _inference_tool.extract_datetime(query)
            if dt:
                state["intent"] = "book_service"
                state["confidence"] = 0.92
        except Exception:
            pass

    # Completing a reschedule waiting for date/time
    if conv_state.get("pending_reschedule_id"):
        try:
            dt = _inference_tool.extract_datetime(query)
            if dt:
                state["intent"] = "reschedule_booking"
                state["confidence"] = 0.92
        except Exception:
            pass

    # Completing a flow waiting for a booking ID
    if conv_state.get("awaiting_booking_id"):
        extracted_id = _appointment_tool.extract_booking_id_from_text(query)
        if extracted_id:
            action = conv_state["awaiting_booking_id"]
            state["intent"] = f"{action}_booking"
            state["confidence"] = 0.92

    return state


# ---------------------------------------------------------------------------
# Node 2: Data Retrieval
# ---------------------------------------------------------------------------

def data_retrieval(state: ChatState) -> ChatState:
    if state["intent"] == "pricing_inquiry":
        try:
            state["response"] = _data_tool.retrieve_and_generate(state["query"])
        except Exception as exc:
            logger.error("Data retrieval failed", exc_info=True)
            state["response"] = "I'm having trouble fetching pricing right now. Please try again shortly."
    return state


# ---------------------------------------------------------------------------
# Node 3: Appointment Trigger
# ---------------------------------------------------------------------------

def appointment_trigger(state: ChatState) -> ChatState:
    intent = state["intent"]
    user_id = state.get("conversation_state", {}).get("user_id", "anonymous")
    conv_state = state.get("conversation_state", {})

    if intent not in ("book_service", "reschedule_booking", "cancel_booking"):
        # Non-appointment intents: handle booking_status here
        if intent == "booking_status":
            appointments = _appointment_tool.get_appointments(user_id)
            if appointments:
                count = len(appointments)
                pending = [a for a in appointments if a[4] == "pending"]
                latest = appointments[-1]
                booking_id = _appointment_tool.format_booking_id(latest[0])
                state["response"] = (
                    f"You have **{count}** appointment(s) ({len(pending)} pending). "
                    f"Most recent: **{booking_id}** — {latest[2]} on {latest[3]} "
                    f"(Status: {latest[4]})"
                )
            else:
                state["response"] = (
                    "You don't have any appointments yet. "
                    "Would you like to book a wellness session?"
                )
        return state

    state["appointment_action"] = intent

    # Extract datetime
    try:
        state["datetime"] = _inference_tool.extract_datetime(state["query"]) or "Not extracted"
    except Exception:
        state["datetime"] = "Not extracted"

    extracted_dt = state["datetime"]
    appt = _appointment_tool

    # ------------------------------------------------------------------ BOOK
    if intent == "book_service":
        service = conv_state.get("pending_service") or _detect_service(state["query"].lower())

        if conv_state.get("pending_service") and extracted_dt != "Not extracted":
            service = conv_state.pop("pending_service")
            result = appt.add_appointment(user_id, service, extracted_dt)
            appts = appt.get_appointments(user_id)
            booking_id = appt.format_booking_id(max(a[0] for a in appts))
            state["response"] = (
                f"**{booking_id}** confirmed!\n"
                f"Service: {service}\nDate/Time: {extracted_dt}\n\n"
                "We'll see you then! Reply 'view my appointments' to see all bookings."
            )
            state["conversation_state"] = conv_state
        elif extracted_dt == "Not extracted":
            state["response"] = (
                f"I'd love to book a **{service}** session for you!\n"
                "Please share your preferred date and time "
                "(e.g. *'Friday at 3pm'* or *'next Tuesday morning'*)."
            )
            conv_state["pending_service"] = service
            state["conversation_state"] = conv_state
        else:
            result = appt.add_appointment(user_id, service, extracted_dt)
            appts = appt.get_appointments(user_id)
            booking_id = appt.format_booking_id(max(a[0] for a in appts))
            state["response"] = (
                f"**{booking_id}** confirmed!\n"
                f"Service: {service}\nDate/Time: {extracted_dt}\n\n"
                "We'll see you then! Reply 'view my appointments' to see all bookings."
            )

    # -------------------------------------------------------------- RESCHEDULE
    elif intent == "reschedule_booking":
        appointments = appt.get_appointments(user_id)
        pending = [a for a in appointments if a[4] == "pending"]
        extracted_id = appt.extract_booking_id_from_text(state["query"])
        pending_id = conv_state.get("pending_reschedule_id")

        if pending_id and extracted_dt != "Not extracted":
            appt.reschedule_appointment(pending_id, extracted_dt)
            booking_id = appt.format_booking_id(pending_id)
            state["response"] = f"**{booking_id}** has been rescheduled to {extracted_dt}."
            conv_state.pop("pending_reschedule_id", None)
            state["conversation_state"] = conv_state

        elif not pending:
            state["response"] = "You don't have any pending appointments to reschedule."

        elif len(pending) == 1:
            appt_id = pending[0][0]
            booking_id = appt.format_booking_id(appt_id)
            if extracted_dt != "Not extracted":
                appt.reschedule_appointment(appt_id, extracted_dt)
                state["response"] = f"**{booking_id}** rescheduled to {extracted_dt}."
            else:
                state["response"] = (
                    f"What's your new preferred date/time for **{booking_id}** — {pending[0][2]}?"
                )
                conv_state["pending_reschedule_id"] = appt_id
                state["conversation_state"] = conv_state

        else:
            if conv_state.get("awaiting_booking_id") == "reschedule" and extracted_id:
                found = next((a for a in pending if a[0] == extracted_id), None)
                if found:
                    if extracted_dt != "Not extracted":
                        appt.reschedule_appointment(extracted_id, extracted_dt)
                        state["response"] = f"**{appt.format_booking_id(extracted_id)}** rescheduled to {extracted_dt}."
                        conv_state.pop("awaiting_booking_id", None)
                    else:
                        state["response"] = f"New date/time for **{appt.format_booking_id(extracted_id)}**?"
                        conv_state["pending_reschedule_id"] = extracted_id
                    state["conversation_state"] = conv_state
                else:
                    ids = ", ".join(appt.format_booking_id(a[0]) for a in pending)
                    state["response"] = f"Booking not found. Your pending appointments: {ids}"
            elif extracted_id:
                found = next((a for a in pending if a[0] == extracted_id), None)
                if found:
                    if extracted_dt != "Not extracted":
                        appt.reschedule_appointment(extracted_id, extracted_dt)
                        state["response"] = f"**{appt.format_booking_id(extracted_id)}** rescheduled to {extracted_dt}."
                    else:
                        state["response"] = f"New date/time for **{appt.format_booking_id(extracted_id)}**?"
                        conv_state["pending_reschedule_id"] = extracted_id
                        state["conversation_state"] = conv_state
                else:
                    ids = ", ".join(appt.format_booking_id(a[0]) for a in pending)
                    state["response"] = f"Booking not found. Your pending appointments: {ids}"
            else:
                ids = ", ".join(appt.format_booking_id(a[0]) for a in pending)
                state["response"] = (
                    f"You have multiple pending appointments: {ids}.\n"
                    "Which booking ID would you like to reschedule?"
                )
                conv_state["awaiting_booking_id"] = "reschedule"
                state["conversation_state"] = conv_state

    # --------------------------------------------------------------- CANCEL
    elif intent == "cancel_booking":
        appointments = appt.get_appointments(user_id)
        pending = [a for a in appointments if a[4] == "pending"]
        extracted_id = appt.extract_booking_id_from_text(state["query"])

        if not pending:
            state["response"] = "You don't have any pending appointments to cancel."

        elif len(pending) == 1:
            appt_id = pending[0][0]
            booking_id = appt.format_booking_id(appt_id)
            appt.cancel_appointment(appt_id)
            state["response"] = (
                f"**{booking_id}** has been cancelled successfully.\n"
                "Is there anything else I can help you with?"
            )

        else:
            if conv_state.get("awaiting_booking_id") == "cancel" and extracted_id:
                found = next((a for a in pending if a[0] == extracted_id), None)
                if found:
                    appt.cancel_appointment(extracted_id)
                    booking_id = appt.format_booking_id(extracted_id)
                    state["response"] = f"**{booking_id}** cancelled successfully."
                    conv_state.pop("awaiting_booking_id", None)
                    state["conversation_state"] = conv_state
                else:
                    ids = ", ".join(appt.format_booking_id(a[0]) for a in pending)
                    state["response"] = f"Booking not found. Your pending appointments: {ids}"
            elif extracted_id:
                found = next((a for a in pending if a[0] == extracted_id), None)
                if found:
                    appt.cancel_appointment(extracted_id)
                    state["response"] = f"**{appt.format_booking_id(extracted_id)}** cancelled successfully."
                else:
                    ids = ", ".join(appt.format_booking_id(a[0]) for a in pending)
                    state["response"] = f"Booking not found. Your pending appointments: {ids}"
            else:
                ids = ", ".join(appt.format_booking_id(a[0]) for a in pending)
                state["response"] = (
                    f"You have multiple pending appointments: {ids}.\n"
                    "Which booking ID would you like to cancel?"
                )
                conv_state["awaiting_booking_id"] = "cancel"
                state["conversation_state"] = conv_state

    return state


# ---------------------------------------------------------------------------
# Build and compile the graph
# ---------------------------------------------------------------------------

_graph = StateGraph(ChatState)
_graph.add_node("intent_analysis", intent_analysis)
_graph.add_node("data_retrieval", data_retrieval)
_graph.add_node("appointment_trigger", appointment_trigger)
_graph.add_edge(START, "intent_analysis")
_graph.add_edge("intent_analysis", "data_retrieval")
_graph.add_edge("data_retrieval", "appointment_trigger")
_graph.add_edge("appointment_trigger", END)

compiled_graph = _graph.compile()
