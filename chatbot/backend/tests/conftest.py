"""
Shared pytest fixtures for the AURA Platform test suite.

Architecture note: integration tests use a real in-process SQLite database
(not mocks) to catch schema drift and query regressions early.
See docs/architecture.md §ADR-005 for the reasoning.
"""

import os
import tempfile
from typing import Generator

import pytest
from fastapi.testclient import TestClient

from app.tools.appointment_tool import AppointmentTool


# ---------------------------------------------------------------------------
# Database fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db_path() -> Generator[str, None, None]:
    """Provide a fresh, isolated SQLite file for each test."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def appointment_tool(tmp_db_path: str) -> AppointmentTool:
    """Return an AppointmentTool backed by a fresh test database."""
    tool = AppointmentTool(db_path=tmp_db_path)
    tool._ensure_initialized()
    return tool


# ---------------------------------------------------------------------------
# API client fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def client(tmp_db_path: str, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """
    FastAPI TestClient with a fresh SQLite database injected via monkeypatch.
    This avoids touching the real aura.db during tests.
    """
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_db_path}")

    # Import app after patching env so settings picks up the test DB
    from app.main import app
    return TestClient(app, raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_user_id() -> str:
    return "test-user-fixture-001"


@pytest.fixture
def booked_appointment(appointment_tool: AppointmentTool, sample_user_id: str) -> int:
    """Create one pending appointment and return its ID."""
    appointment_tool.add_appointment(
        user_id=sample_user_id,
        service="Swedish Massage",
        date_time="2026-12-20 14:00",
    )
    rows = appointment_tool.get_appointments(sample_user_id)
    return rows[0][0]
