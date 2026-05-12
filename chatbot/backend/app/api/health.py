"""
Health and observability endpoints.

Kubernetes / Docker Compose uses these probes:
  GET /health/live   → liveness  (is the process alive?)
  GET /health/ready  → readiness (is every dependency reachable?)
  GET /metrics       → lightweight application metrics
"""

import sqlite3
import time
from datetime import datetime, timezone

from fastapi import APIRouter

from app.core.config import settings
from app.core.logging_config import get_logger

router = APIRouter(tags=["observability"])
logger = get_logger("aura.health")

_START_TIME = time.time()


@router.get("/health/live", summary="Liveness probe")
async def liveness() -> dict:
    """
    Returns 200 as long as the process is running.
    Kubernetes restarts the pod if this probe fails.
    """
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@router.get("/health/ready", summary="Readiness probe")
async def readiness() -> dict:
    """
    Validates every downstream dependency before the instance accepts traffic.
    Kubernetes stops routing traffic to the pod if this probe returns non-200.
    """
    checks: dict[str, str] = {}
    overall_ok = True

    # --- Database ---
    try:
        db_path = (
            settings.database_url
            .replace("sqlite:///./", "")
            .replace("sqlite:///", "")
        )
        conn = sqlite3.connect(db_path, timeout=2)
        conn.execute("SELECT 1")
        conn.close()
        checks["database"] = "ok"
    except Exception as exc:
        checks["database"] = f"error: {exc}"
        overall_ok = False
        logger.error("Database health check failed", exc_info=True)

    # --- ML model (non-fatal: keyword fallback covers outages) ---
    try:
        from app.tools.inference_tool import InferenceTool  # noqa: F401
        checks["ml_model"] = "ok (lazy-loaded)"
    except Exception as exc:
        checks["ml_model"] = f"degraded: {exc}"
        logger.warning("ML model health check degraded — keyword fallback active")

    status_str = "ok" if overall_ok else "degraded"

    return {
        "status": status_str,
        "version": settings.app_version,
        "environment": settings.environment,
        "uptime_seconds": round(time.time() - _START_TIME, 1),
        "checks": checks,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/metrics", summary="Application metrics")
async def metrics() -> dict:
    """
    Lightweight JSON metrics surface.

    For production Prometheus scraping, add `prometheus-fastapi-instrumentator`
    and expose a `/metrics` endpoint in Prometheus exposition format.
    """
    appointment_count = 0
    pending_count = 0
    cancelled_count = 0

    try:
        db_path = (
            settings.database_url
            .replace("sqlite:///./", "")
            .replace("sqlite:///", "")
        )
        conn = sqlite3.connect(db_path, timeout=2)
        cur = conn.cursor()
        cur.execute("SELECT status, COUNT(*) FROM appointments GROUP BY status")
        for row in cur.fetchall():
            status, count = row
            appointment_count += count
            if status == "pending":
                pending_count = count
            elif status == "cancelled":
                cancelled_count = count
        conn.close()
    except Exception:
        pass  # Metrics failure must never crash the service

    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "uptime_seconds": round(time.time() - _START_TIME, 1),
        "appointments_total": appointment_count,
        "appointments_pending": pending_count,
        "appointments_cancelled": cancelled_count,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
