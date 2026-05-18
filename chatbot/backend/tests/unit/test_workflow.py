"""
Unit tests for the LangGraph ChatState workflow.

The ML model (DistilBERT) is deliberately NOT mocked — the workflow has a
keyword-rule fallback tier that operates independently of the model.
These tests exercise that fallback path, which is the primary runtime path
when the model file is absent (the common case during CI).
"""

import pytest

from app.chatbot_workflow import intent_analysis, data_retrieval, appointment_trigger


def _make_state(query: str, conv_state: dict = None) -> dict:
    return {
        "query": query,
        "intent": "",
        "confidence": 0.0,
        "response": "",
        "appointment_action": "",
        "datetime": "",
        "conversation_state": conv_state or {},
    }


class TestIntentAnalysis:
    def test_greeting_detected(self):
        state = intent_analysis(_make_state("Hello there!"))
        assert state["intent"] == "greeting"
        assert state["confidence"] >= 0.9

    def test_cancel_keyword_overrides_ml(self):
        state = intent_analysis(_make_state("I need to cancel my booking"))
        assert state["intent"] == "cancel_booking"
        assert state["confidence"] >= 0.9

    def test_reschedule_keyword_overrides_ml(self):
        state = intent_analysis(_make_state("I'd like to reschedule my appointment"))
        assert state["intent"] == "reschedule_booking"
        assert state["confidence"] >= 0.9

    def test_booking_intent_with_service(self):
        state = intent_analysis(_make_state("Book me a Swedish massage please"))
        assert state["intent"] == "book_service"

    def test_pricing_intent(self):
        state = intent_analysis(_make_state("How much does a deep tissue massage cost?"))
        assert state["intent"] == "pricing_inquiry"

    def test_thanks_detected(self):
        state = intent_analysis(_make_state("Thank you so much!"))
        assert state["intent"] == "thanks"

    def test_booking_status_intent(self):
        state = intent_analysis(_make_state("Show me my appointments"))
        assert state["intent"] == "booking_status"

    def test_cancel_takes_priority_over_pricing(self):
        # "cancel" should always win even when other keywords are present
        state = intent_analysis(_make_state("How much to cancel my booking?"))
        assert state["intent"] == "cancel_booking"

    def test_pending_service_triggers_book_service_for_datetime(self):
        # When we're waiting for a date, any date-like input should route to book_service
        conv = {"pending_service": "Hot Stone Massage", "user_id": "u1"}
        state = intent_analysis(_make_state("Friday at 3pm", conv_state=conv))
        assert state["intent"] == "book_service"

    def test_awaiting_cancel_id_routes_correctly(self):
        conv = {"awaiting_booking_id": "cancel", "user_id": "u1"}
        state = intent_analysis(_make_state("AURA-003-2026", conv_state=conv))
        assert state["intent"] == "cancel_booking"


class TestDataRetrieval:
    def test_pricing_intent_sets_response(self):
        import os, tempfile, pandas as pd
        from app.tools.data_tool import DataTool

        # Inject a minimal dataset
        csv_data = pd.DataFrame([{
            "ID": 1, "Service_Name": "Swedish Massage", "Category": "Massage",
            "Avg_Spending": 85, "Duration_Minutes": 60, "Description": "Test",
        }])
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
            csv_data.to_csv(f, index=False)
            tmp_path = f.name

        import app.chatbot_workflow as wf
        original = wf._data_tool
        wf._data_tool = DataTool(csv_path=tmp_path)

        state = _make_state("how much is swedish massage?")
        state["intent"] = "pricing_inquiry"
        result = data_retrieval(state)
        assert "Swedish Massage" in result["response"]
        assert "85" in result["response"]

        wf._data_tool = original
        os.unlink(tmp_path)

    def test_non_pricing_intent_leaves_response_unchanged(self):
        state = _make_state("Hello")
        state["intent"] = "greeting"
        state["response"] = "Hello!"
        result = data_retrieval(state)
        assert result["response"] == "Hello!"


class TestServiceDetection:
    """Verify the _detect_service helper maps queries to canonical names."""

    def test_detect_thai_massage(self):
        from app.chatbot_workflow import _detect_service
        assert _detect_service("i want a thai massage") == "Thai Massage"

    def test_detect_yoga(self):
        from app.chatbot_workflow import _detect_service
        assert _detect_service("book a hatha yoga session") == "Hatha Yoga Session"

    def test_detect_facial(self):
        from app.chatbot_workflow import _detect_service
        assert _detect_service("classic facial please") == "Classic Facial"

    def test_detect_acupuncture(self):
        from app.chatbot_workflow import _detect_service
        assert _detect_service("traditional acupuncture session") == "Traditional Acupuncture"

    def test_defaults_to_swedish(self):
        from app.chatbot_workflow import _detect_service
        # Unrecognised service → safe default
        assert _detect_service("i want to relax") == "Swedish Massage"
