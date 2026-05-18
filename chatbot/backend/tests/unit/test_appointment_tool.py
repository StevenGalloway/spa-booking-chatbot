"""
Unit tests for AppointmentTool.

Tests use a real SQLite file (via the tmp_db_path fixture) so schema
integrity and SQL correctness are verified, not mocked away.
"""

import pytest

from app.tools.appointment_tool import AppointmentTool


class TestSchemaInit:
    def test_creates_table_on_first_use(self, appointment_tool: AppointmentTool):
        """The appointments table must exist after initialization."""
        import sqlite3
        conn = sqlite3.connect(appointment_tool._db_path)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='appointments'")
        assert cur.fetchone() is not None
        conn.close()

    def test_created_at_column_present(self, appointment_tool: AppointmentTool):
        import sqlite3
        conn = sqlite3.connect(appointment_tool._db_path)
        cur = conn.cursor()
        cols = {row[1] for row in cur.execute("PRAGMA table_info(appointments)")}
        conn.close()
        assert "created_at" in cols
        assert "updated_at" in cols


class TestCRUD:
    def test_add_appointment_returns_success(self, appointment_tool: AppointmentTool, sample_user_id: str):
        result = appointment_tool.add_appointment(sample_user_id, "Deep Tissue Massage", "2026-11-15 10:00")
        assert "successfully" in result.lower()

    def test_get_appointments_returns_correct_user(self, appointment_tool: AppointmentTool, sample_user_id: str):
        appointment_tool.add_appointment(sample_user_id, "Swedish Massage", "2026-11-10 09:00")
        appointment_tool.add_appointment("other-user", "Thai Massage", "2026-11-11 11:00")
        rows = appointment_tool.get_appointments(sample_user_id)
        assert len(rows) == 1
        assert rows[0][1] == sample_user_id

    def test_get_appointments_all_when_no_user_id(self, appointment_tool: AppointmentTool, sample_user_id: str):
        appointment_tool.add_appointment(sample_user_id, "Reiki", "2026-11-12 13:00")
        appointment_tool.add_appointment("user-b", "Reflexology", "2026-11-12 14:00")
        rows = appointment_tool.get_appointments()
        assert len(rows) == 2

    def test_initial_status_is_pending(self, appointment_tool: AppointmentTool, sample_user_id: str):
        appointment_tool.add_appointment(sample_user_id, "Sports Massage", "2026-11-20 16:00")
        rows = appointment_tool.get_appointments(sample_user_id)
        assert rows[0][4] == "pending"

    def test_cancel_appointment(self, appointment_tool: AppointmentTool, booked_appointment: int, sample_user_id: str):
        result = appointment_tool.cancel_appointment(booked_appointment)
        assert "cancelled" in result.lower()
        rows = appointment_tool.get_appointments(sample_user_id)
        assert rows[0][4] == "cancelled"

    def test_cancel_nonexistent_returns_not_found(self, appointment_tool: AppointmentTool):
        result = appointment_tool.cancel_appointment(9999)
        assert "not found" in result.lower()

    def test_reschedule_appointment(self, appointment_tool: AppointmentTool, booked_appointment: int, sample_user_id: str):
        new_dt = "2026-12-25 11:00"
        result = appointment_tool.reschedule_appointment(booked_appointment, new_dt)
        assert "rescheduled" in result.lower()
        rows = appointment_tool.get_appointments(sample_user_id)
        assert rows[0][3] == new_dt

    def test_reschedule_updates_updated_at(self, appointment_tool: AppointmentTool, booked_appointment: int, sample_user_id: str):
        import sqlite3, time
        time.sleep(0.05)  # Ensure timestamp changes
        appointment_tool.reschedule_appointment(booked_appointment, "2026-12-26 10:00")
        conn = sqlite3.connect(appointment_tool._db_path)
        row = conn.execute("SELECT created_at, updated_at FROM appointments WHERE id = ?", (booked_appointment,)).fetchone()
        conn.close()
        # updated_at should differ from created_at after reschedule
        assert row[0] != row[1]


class TestFormatBookingId:
    def test_format_pads_id(self):
        bid = AppointmentTool.format_booking_id(7)
        assert bid.startswith("AURA-007-")

    def test_format_large_id(self):
        bid = AppointmentTool.format_booking_id(1234)
        assert "1234" in bid

    def test_format_contains_current_year(self):
        from datetime import datetime
        bid = AppointmentTool.format_booking_id(1)
        assert str(datetime.now().year) in bid


class TestExtractBookingId:
    def test_new_aura_format(self):
        assert AppointmentTool.extract_booking_id_from_text("Cancel AURA-007-2026 please") == 7

    def test_legacy_book_format(self):
        assert AppointmentTool.extract_booking_id_from_text("I want to cancel BOOK-03-2025") == 3

    def test_hash_shorthand(self):
        assert AppointmentTool.extract_booking_id_from_text("cancel booking #5") == 5

    def test_returns_none_for_no_match(self):
        assert AppointmentTool.extract_booking_id_from_text("I want to cancel my appointment") is None

    def test_plain_number_with_appointment_keyword(self):
        result = AppointmentTool.extract_booking_id_from_text("reschedule appointment 12")
        assert result == 12
