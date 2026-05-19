"""
Integration tests for the chat and service catalog API endpoints.

Uses FastAPI's TestClient with a real in-process SQLite database so HTTP
routing, middleware, schema validation, and database writes are all exercised
in a single test run without spawning external processes.
"""

import pytest
from fastapi.testclient import TestClient


class TestHealthEndpoints:
    def test_liveness_returns_200(self, client: TestClient):
        r = client.get("/health/live")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_readiness_returns_status(self, client: TestClient):
        r = client.get("/health/ready")
        assert r.status_code == 200
        data = r.json()
        assert "status" in data
        assert "version" in data
        assert "checks" in data

    def test_metrics_returns_json(self, client: TestClient):
        r = client.get("/metrics")
        assert r.status_code == 200
        data = r.json()
        assert "uptime_seconds" in data
        assert "appointments_total" in data

    def test_root_returns_service_info(self, client: TestClient):
        r = client.get("/")
        assert r.status_code == 200
        assert "service" in r.json()


class TestChatEndpoint:
    def test_greeting_returns_200(self, client: TestClient, sample_user_id: str):
        r = client.post(
            "/api/v1/chat",
            json={"message": "Hello!", "user_id": sample_user_id, "conversation_state": {}},
        )
        assert r.status_code == 200
        body = r.json()
        assert "response" in body
        assert "intent" in body
        assert "confidence" in body
        assert "conversation_state" in body

    def test_chat_returns_request_id_header(self, client: TestClient, sample_user_id: str):
        r = client.post(
            "/api/v1/chat",
            json={"message": "Hi", "user_id": sample_user_id, "conversation_state": {}},
        )
        assert "X-Request-ID" in r.headers
        assert len(r.headers["X-Request-ID"]) == 36  # UUID format

    def test_empty_message_rejected(self, client: TestClient, sample_user_id: str):
        r = client.post(
            "/api/v1/chat",
            json={"message": "", "user_id": sample_user_id, "conversation_state": {}},
        )
        assert r.status_code == 422  # Pydantic validation error

    def test_message_too_long_rejected(self, client: TestClient, sample_user_id: str):
        r = client.post(
            "/api/v1/chat",
            json={"message": "x" * 2001, "user_id": sample_user_id, "conversation_state": {}},
        )
        assert r.status_code == 422

    def test_pricing_inquiry_intent(self, client: TestClient, sample_user_id: str):
        r = client.post(
            "/api/v1/chat",
            json={
                "message": "How much does a Swedish massage cost?",
                "user_id": sample_user_id,
                "conversation_state": {},
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["intent"] == "pricing_inquiry"

    def test_cancel_intent_detected(self, client: TestClient, sample_user_id: str):
        r = client.post(
            "/api/v1/chat",
            json={
                "message": "I need to cancel my appointment",
                "user_id": sample_user_id,
                "conversation_state": {},
            },
        )
        assert r.status_code == 200
        assert r.json()["intent"] == "cancel_booking"

    def test_conversation_state_is_preserved(self, client: TestClient, sample_user_id: str):
        # Turn 1: request a booking — no datetime provided
        r1 = client.post(
            "/api/v1/chat",
            json={
                "message": "Book me a hot stone massage",
                "user_id": sample_user_id,
                "conversation_state": {},
            },
        )
        assert r1.status_code == 200
        state_after_turn1 = r1.json()["conversation_state"]

        # State should now hold the pending service
        assert "pending_service" in state_after_turn1

        # Turn 2: provide date/time
        r2 = client.post(
            "/api/v1/chat",
            json={
                "message": "Friday at 3pm",
                "user_id": sample_user_id,
                "conversation_state": state_after_turn1,
            },
        )
        assert r2.status_code == 200
        # Booking should now be confirmed (pending_service cleared)
        state_after_turn2 = r2.json()["conversation_state"]
        assert "pending_service" not in state_after_turn2


class TestServicesEndpoint:
    def test_returns_service_list(self, client: TestClient):
        r = client.get("/api/v1/services")
        assert r.status_code == 200
        services = r.json()
        assert isinstance(services, list)
        assert len(services) > 0

    def test_service_has_required_fields(self, client: TestClient):
        r = client.get("/api/v1/services")
        service = r.json()[0]
        for field in ("name", "category", "price", "duration", "description"):
            assert field in service, f"Missing field: {field}"

    def test_category_filter(self, client: TestClient):
        r = client.get("/api/v1/services?category=Massage")
        assert r.status_code == 200
        services = r.json()
        assert all(s["category"] == "Massage" for s in services)

    def test_unknown_category_returns_empty(self, client: TestClient):
        r = client.get("/api/v1/services?category=NonExistentCategory")
        assert r.status_code == 200
        assert r.json() == []


class TestPractitionersEndpoint:
    def test_returns_practitioners(self, client: TestClient):
        r = client.get("/api/v1/practitioners")
        assert r.status_code == 200
        practitioners = r.json()
        assert isinstance(practitioners, list)
        assert len(practitioners) >= 1

    def test_practitioner_has_required_fields(self, client: TestClient):
        r = client.get("/api/v1/practitioners")
        p = r.json()[0]
        for field in ("id", "name", "title", "specialties", "experience_years", "rating", "bio"):
            assert field in p

    def test_available_only_filter(self, client: TestClient):
        r = client.get("/api/v1/practitioners?available_only=true")
        assert r.status_code == 200
        practitioners = r.json()
        assert all(p["available"] for p in practitioners)


class TestAppointmentsEndpoint:
    def test_empty_appointments_for_new_user(self, client: TestClient):
        r = client.get("/api/v1/appointments/brand-new-user-xyz")
        assert r.status_code == 200
        assert r.json() == []

    def test_appointment_appears_after_booking(self, client: TestClient, sample_user_id: str):
        # Book via the chat endpoint
        client.post(
            "/api/v1/chat",
            json={
                "message": "Book a Swedish massage for next Monday at 10am",
                "user_id": sample_user_id,
                "conversation_state": {},
            },
        )
        r = client.get(f"/api/v1/appointments/{sample_user_id}")
        assert r.status_code == 200
        # There should be at least one appointment
        assert len(r.json()) >= 1

    def test_appointment_response_schema(self, client: TestClient, sample_user_id: str):
        client.post(
            "/api/v1/chat",
            json={
                "message": "Book a deep tissue massage tomorrow at 2pm",
                "user_id": sample_user_id,
                "conversation_state": {},
            },
        )
        r = client.get(f"/api/v1/appointments/{sample_user_id}")
        if r.json():
            appt = r.json()[0]
            for field in ("id", "user_id", "service_type", "date", "time", "status", "created_at"):
                assert field in appt
