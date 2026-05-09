"""
AURA Platform — Streamlit Frontend

AI-Powered Wellness Booking Interface
"""

import os
import uuid
from datetime import datetime

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")
BACKEND_HEALTH_URL = os.getenv("API_BASE_URL", "http://localhost:8000").replace("/api/v1", "")

# ---------------------------------------------------------------------------
# Page setup  (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AURA — Wellness Booking",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — AURA brand skin on top of the config.toml theme
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    /* ── AURA brand header ─────────────────────────────────────────────── */
    .aura-logo {
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: 0.18em;
        color: #2D6A6A;
        margin: 0;
        padding: 0;
    }
    .aura-tagline {
        font-size: 0.85rem;
        color: #6B7F7F;
        letter-spacing: 0.08em;
        margin-top: -6px;
        text-transform: uppercase;
    }
    .aura-divider {
        border: none;
        border-top: 2px solid #C9A96E;
        margin: 12px 0 18px 0;
    }

    /* ── Chat bubbles ──────────────────────────────────────────────────── */
    .stChatMessage[data-testid="stChatMessageContent"] {
        border-radius: 12px;
    }

    /* ── Service category badges ───────────────────────────────────────── */
    .category-badge {
        display: inline-block;
        background-color: #2D6A6A;
        color: #F7F4EF;
        font-size: 0.7rem;
        padding: 2px 8px;
        border-radius: 999px;
        margin-right: 4px;
        margin-bottom: 4px;
        font-weight: 600;
        letter-spacing: 0.04em;
    }

    /* ── Booking ID pill ───────────────────────────────────────────────── */
    .booking-id {
        font-family: monospace;
        background: #E8E3D8;
        color: #2D6A6A;
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: 700;
    }

    /* ── Sidebar practitioner card ─────────────────────────────────────── */
    .practitioner-name {
        font-weight: 700;
        color: #1C2B2B;
    }
    .practitioner-title {
        font-size: 0.78rem;
        color: #6B7F7F;
    }
    .rating-stars {
        color: #C9A96E;
        font-size: 0.8rem;
    }

    /* ── Status indicators ─────────────────────────────────────────────── */
    .status-pending  { color: #2D6A6A; font-weight: 600; }
    .status-cancelled { color: #C0392B; font-weight: 600; }
    .status-completed { color: #27AE60; font-weight: 600; }

    /* ── Hide Streamlit branding ───────────────────────────────────────── */
    #MainMenu, footer, header { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

if "conversation_state" not in st.session_state:
    st.session_state.conversation_state = {}

if "processing" not in st.session_state:
    st.session_state.processing = False

if "show_all_services" not in st.session_state:
    st.session_state.show_all_services = False

if "active_tab" not in st.session_state:
    st.session_state.active_tab = "chat"


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def fetch_services(category: str = "") -> list:
    try:
        url = f"{API_BASE_URL}/services"
        if category:
            url += f"?category={category}"
        r = requests.get(url, timeout=5)
        return r.json() if r.status_code == 200 else []
    except Exception:
        return []


@st.cache_data(ttl=120)
def fetch_practitioners(available_only: bool = False) -> list:
    try:
        url = f"{API_BASE_URL}/practitioners"
        if available_only:
            url += "?available_only=true"
        r = requests.get(url, timeout=5)
        return r.json() if r.status_code == 200 else []
    except Exception:
        return []


def fetch_appointments(user_id: str) -> list:
    try:
        r = requests.get(f"{API_BASE_URL}/appointments/{user_id}", timeout=5)
        return r.json() if r.status_code == 200 else []
    except Exception:
        return []


def check_backend_health() -> bool:
    try:
        r = requests.get(f"{BACKEND_HEALTH_URL}/health/live", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def send_message(message: str) -> dict | None:
    try:
        payload = {
            "message": message,
            "user_id": st.session_state.user_id,
            "conversation_state": st.session_state.conversation_state,
        }
        r = requests.post(
            f"{API_BASE_URL}/chat",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        return r.json() if r.status_code == 200 else None
    except Exception as exc:
        st.error(f"Connection error: {exc}")
        return None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

CATEGORY_ICONS = {
    "Massage": "💆",
    "Yoga & Meditation": "🧘",
    "Facial & Skincare": "✨",
    "Acupuncture": "🪡",
    "Hair & Beauty": "💇",
    "Fitness & Wellness": "🏋️",
}

with st.sidebar:
    # Logo
    st.markdown('<p class="aura-logo">AURA</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="aura-tagline">AI Wellness Booking Platform</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="aura-divider">', unsafe_allow_html=True)

    # Backend status
    if "backend_ok" not in st.session_state:
        st.session_state.backend_ok = check_backend_health()

    if st.session_state.backend_ok:
        st.success("🟢 Connected")
    else:
        st.error("🔴 Backend offline")
        if st.button("Retry connection"):
            del st.session_state.backend_ok
            st.rerun()

    st.markdown(f"**Session:** `{st.session_state.user_id[:8]}…`")

    # New session button
    if st.button("✦ New Session", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_state = {}
        st.rerun()

    st.markdown('<hr class="aura-divider">', unsafe_allow_html=True)

    # Service catalogue
    st.markdown("#### Services")

    all_services = fetch_services()
    if all_services:
        # Group by category
        categories: dict[str, list] = {}
        for svc in all_services:
            cat = svc.get("category", "Massage")
            categories.setdefault(cat, []).append(svc)

        if not st.session_state.show_all_services:
            # Show category summary
            for cat, svcs in categories.items():
                icon = CATEGORY_ICONS.get(cat, "●")
                with st.expander(f"{icon} {cat} ({len(svcs)})", expanded=False):
                    for s in svcs[:5]:
                        st.markdown(
                            f"• **{s['name']}** — ${s['price']} · {s['duration']}min"
                        )
                    if len(svcs) > 5:
                        st.caption(f"+ {len(svcs) - 5} more…")
            if st.button("View full catalogue", use_container_width=True):
                st.session_state.show_all_services = True
                st.rerun()
        else:
            for cat, svcs in categories.items():
                icon = CATEGORY_ICONS.get(cat, "●")
                st.markdown(f"**{icon} {cat}**")
                for s in svcs:
                    st.markdown(f"• {s['name']} — ${s['price']}")
            if st.button("Collapse catalogue", use_container_width=True):
                st.session_state.show_all_services = False
                st.rerun()
    else:
        # Fallback list
        for name in [
            "Swedish Massage", "Deep Tissue", "Hot Stone",
            "Hatha Yoga", "Classic Facial", "Acupuncture",
        ]:
            st.markdown(f"• {name}")

    st.markdown('<hr class="aura-divider">', unsafe_allow_html=True)

    # Practitioners
    st.markdown("#### Practitioners")
    practitioners = fetch_practitioners(available_only=True)
    if practitioners:
        for p in practitioners[:3]:
            stars = "★" * round(p.get("rating", 5))
            st.markdown(
                f'<p class="practitioner-name">{p["name"]}</p>'
                f'<p class="practitioner-title">{p["title"]}</p>'
                f'<p class="rating-stars">{stars} {p.get("rating", 5.0)}</p>',
                unsafe_allow_html=True,
            )
            st.markdown("")
    else:
        st.caption("Practitioner data unavailable")

    st.markdown('<hr class="aura-divider">', unsafe_allow_html=True)
    st.caption("© 2026 AURA Platform · v1.0.0")

# ---------------------------------------------------------------------------
# Main panel — Header
# ---------------------------------------------------------------------------

col_title, col_action = st.columns([3, 1])
with col_title:
    st.markdown("## 🌿 AURA Wellness Assistant")
    st.caption(
        "Book sessions, check pricing, manage appointments — all in one conversation."
    )
with col_action:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("📋 My Appointments", use_container_width=True):
        st.session_state.active_tab = (
            "chat" if st.session_state.active_tab == "appointments" else "appointments"
        )

st.divider()

# ---------------------------------------------------------------------------
# Appointments panel (toggle)
# ---------------------------------------------------------------------------

if st.session_state.active_tab == "appointments":
    st.markdown("### Your Appointments")
    appts = fetch_appointments(st.session_state.user_id)
    if appts:
        for appt in appts:
            status = appt.get("status", "pending")
            status_icon = {"pending": "🟦", "cancelled": "🟥", "completed": "🟩"}.get(status, "⬜")
            st.markdown(
                f"{status_icon} **{appt.get('service_type', 'Service')}** &nbsp;·&nbsp; "
                f"{appt.get('date', 'TBD')} {appt.get('time', '')} &nbsp;·&nbsp; "
                f"Status: **{status.capitalize()}**"
            )
        st.divider()
    else:
        st.info("No appointments on file yet. Start a conversation to book your first session!")
    if st.button("← Back to Chat"):
        st.session_state.active_tab = "chat"
        st.rerun()

# ---------------------------------------------------------------------------
# Chat interface
# ---------------------------------------------------------------------------

else:
    # Greeting message on first load
    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.markdown(
                "**Welcome to AURA!** 🌿\n\n"
                "I'm your AI wellness booking assistant. I can help you:\n"
                "- 📅 **Book** a massage, yoga, facial, acupuncture, or fitness session\n"
                "- 💰 **Check pricing** for any of our 55+ services\n"
                "- 🔄 **Reschedule or cancel** existing appointments\n"
                "- 📋 **View** your upcoming bookings\n\n"
                "What can I help you with today?"
            )

    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Intent metadata expander (assistant messages only)
            if msg["role"] == "assistant" and "metadata" in msg:
                meta = msg["metadata"]
                with st.expander("AI Analysis", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Intent", meta.get("intent", "—"))
                    with col2:
                        conf = meta.get("confidence", 0)
                        st.metric("Confidence", f"{conf:.0%}")

    # Chat input
    if prompt := st.chat_input("Ask about booking, pricing, or your appointments…"):
        if not st.session_state.processing:
            st.session_state.processing = True

            # Display user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner(""):
                    result = send_message(prompt)

                if result:
                    response_text = result.get("response", "I didn't catch that. Could you rephrase?")
                    intent = result.get("intent", "unknown")
                    confidence = result.get("confidence", 0.0)
                    request_id = result.get("request_id", "")

                    # Update conversation state for multi-turn flows
                    st.session_state.conversation_state = result.get("conversation_state", {})

                    st.markdown(response_text)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_text,
                        "metadata": {
                            "intent": intent,
                            "confidence": confidence,
                            "request_id": request_id,
                        },
                    })
                else:
                    err = "Unable to reach the AURA backend. Please check your connection and try again."
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err})

            st.session_state.processing = False

    # Quick-action suggestion chips (shown when chat is empty after first turn)
    if len(st.session_state.messages) <= 2:
        st.markdown("**Try asking:**")
        suggestions = [
            "Book a Swedish massage for Friday at 3pm",
            "How much does a deep tissue massage cost?",
            "What yoga sessions do you offer?",
            "Show me available practitioners",
        ]
        cols = st.columns(len(suggestions))
        for col, suggestion in zip(cols, suggestions):
            with col:
                if st.button(suggestion, use_container_width=True, key=f"sug_{suggestion[:20]}"):
                    # Inject as if the user typed it
                    st.session_state.messages.append({"role": "user", "content": suggestion})
                    with st.spinner(""):
                        result = send_message(suggestion)
                    if result:
                        st.session_state.conversation_state = result.get("conversation_state", {})
                        resp = result.get("response", "")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": resp,
                            "metadata": {
                                "intent": result.get("intent"),
                                "confidence": result.get("confidence"),
                            },
                        })
                    st.rerun()
