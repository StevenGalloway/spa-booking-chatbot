"""
AppointmentTool — SQLite-backed appointment management for the AURA Platform.

Schema v2 additions (backward compatible via ALTER TABLE migration):
  - created_at  TIMESTAMP  — immutable booking timestamp
  - updated_at  TIMESTAMP  — last modification timestamp

Production upgrade path:
  Replace SQLite with PostgreSQL + asyncpg + SQLAlchemy 2.0 async engine.
  Use Alembic for schema migrations.  See docs/data-model.md.
"""

import os
import re
import sqlite3
from datetime import datetime
from typing import List, Optional, Tuple

from app.core.logging_config import get_logger

logger = get_logger("aura.tools.appointment")

# Type alias: raw DB row (id, user_id, service, date_time, status, created_at, updated_at)
AppointmentRow = Tuple


class AppointmentTool:
    def __init__(self, db_path: Optional[str] = None) -> None:
        self._db_path = db_path
        self._initialized = False

    # ------------------------------------------------------------------
    # Initialization & schema migration
    # ------------------------------------------------------------------

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return

        if self._db_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self._db_path = os.path.normpath(
                os.path.join(current_dir, "..", "aura.db")
            )

        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        self._init_schema()
        self._initialized = True

    def _connect(self) -> sqlite3.Connection:
        self._ensure_initialized()
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent read performance
        return conn

    def _init_schema(self) -> None:
        """Create tables and apply lightweight schema migrations."""
        conn = sqlite3.connect(self._db_path, timeout=10)
        try:
            cur = conn.cursor()

            # Base table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS appointments (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id     TEXT    NOT NULL,
                    service     TEXT    NOT NULL,
                    date_time   TEXT,
                    status      TEXT    NOT NULL DEFAULT 'pending',
                    created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
                    updated_at  TEXT    NOT NULL DEFAULT (datetime('now'))
                )
            """)

            # Schema migration: add columns if they were missing in v1
            existing_cols = {row[1] for row in cur.execute("PRAGMA table_info(appointments)")}
            if "created_at" not in existing_cols:
                cur.execute("ALTER TABLE appointments ADD COLUMN created_at TEXT DEFAULT (datetime('now'))")
                logger.info("Schema migration: added created_at column")
            if "updated_at" not in existing_cols:
                cur.execute("ALTER TABLE appointments ADD COLUMN updated_at TEXT DEFAULT (datetime('now'))")
                logger.info("Schema migration: added updated_at column")

            # Index for fast per-user queries
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_appointments_user_id
                ON appointments (user_id)
            """)

            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add_appointment(self, user_id: str, service: str, date_time: str) -> str:
        now = datetime.now().isoformat()
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO appointments (user_id, service, date_time, status, created_at, updated_at)
                VALUES (?, ?, ?, 'pending', ?, ?)
                """,
                (user_id, service, date_time, now, now),
            )
            conn.commit()
            logger.info(
                "Appointment created",
                extra={"user_id": user_id, "service": service, "date_time": date_time},
            )
            return "Appointment added successfully."
        finally:
            conn.close()

    def cancel_appointment(self, appointment_id: int) -> str:
        now = datetime.now().isoformat()
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                "UPDATE appointments SET status = 'cancelled', updated_at = ? WHERE id = ?",
                (now, appointment_id),
            )
            conn.commit()
            success = cur.rowcount > 0
            if success:
                logger.info("Appointment cancelled", extra={"appointment_id": appointment_id})
            return "Appointment cancelled successfully." if success else "Appointment not found."
        finally:
            conn.close()

    def reschedule_appointment(self, appointment_id: int, new_date_time: str) -> str:
        now = datetime.now().isoformat()
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                "UPDATE appointments SET date_time = ?, updated_at = ? WHERE id = ?",
                (new_date_time, now, appointment_id),
            )
            conn.commit()
            success = cur.rowcount > 0
            if success:
                logger.info(
                    "Appointment rescheduled",
                    extra={"appointment_id": appointment_id, "new_date_time": new_date_time},
                )
            return "Appointment rescheduled successfully." if success else "Appointment not found."
        finally:
            conn.close()

    def get_appointments(self, user_id: Optional[str] = None) -> List[AppointmentRow]:
        conn = self._connect()
        try:
            cur = conn.cursor()
            if user_id:
                cur.execute(
                    "SELECT id, user_id, service, date_time, status, created_at, updated_at "
                    "FROM appointments WHERE user_id = ? ORDER BY id",
                    (user_id,),
                )
            else:
                cur.execute(
                    "SELECT id, user_id, service, date_time, status, created_at, updated_at "
                    "FROM appointments ORDER BY id"
                )
            return cur.fetchall()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def format_booking_id(appointment_id: int) -> str:
        """Format as AURA-{zero-padded id}-{year}, e.g. AURA-007-2026."""
        year = datetime.now().year
        return f"AURA-{appointment_id:03d}-{year}"

    @staticmethod
    def extract_booking_id_from_text(text: str) -> Optional[int]:
        """
        Extract an appointment ID from free-form text.

        Recognises:
          AURA-007-2026  (new format)
          BOOK-01-2025   (legacy format)
          #7, booking 7  (numeric shorthand)
        """
        text_lower = text.lower()

        # New format: AURA-NNN-YYYY
        match = re.search(r"aura-(\d+)-\d+", text_lower)
        if match:
            return int(match.group(1))

        # Legacy format: BOOK-NN-YYYY
        match = re.search(r"book-(\d+)-\d+", text_lower)
        if match:
            return int(match.group(1))

        # Shorthand: #N or booking N or appointment N
        match = re.search(r"(?:#|booking\s*|appointment\s*)(\d+)", text_lower)
        if match:
            return int(match.group(1))

        # Plain number when context strongly implies booking
        if any(kw in text_lower for kw in ("booking", "appointment", "aura", "book")):
            match = re.search(r"\b(\d+)\b", text_lower)
            if match:
                return int(match.group(1))

        return None
