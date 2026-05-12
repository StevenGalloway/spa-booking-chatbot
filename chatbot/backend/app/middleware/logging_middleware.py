"""
Request logging middleware.

Attaches a unique X-Request-ID to every inbound request so that all log
lines for a single request can be correlated across services.  Also measures
and logs end-to-end latency via X-Response-Time.
"""

import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.logging_config import get_logger

logger = get_logger("aura.middleware")

# Paths excluded from verbose access logging (liveness probes, etc.)
_SILENT_PATHS = {"/health/live", "/health/ready", "/metrics", "/"}


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Per-request structured logging middleware.

    Emits two log lines per request:
      1. request.started  — method, path, client IP
      2. request.completed / request.failed — status code, latency
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        start = time.perf_counter()
        path = request.url.path
        silent = path in _SILENT_PATHS

        if not silent:
            logger.info(
                "Request started",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": path,
                    "client": request.client.host if request.client else "unknown",
                },
            )

        try:
            response = await call_next(request)
        except Exception as exc:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.error(
                "Request failed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": path,
                    "duration_ms": duration_ms,
                    "error": str(exc),
                },
                exc_info=True,
            )
            raise

        duration_ms = round((time.perf_counter() - start) * 1000, 2)

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{duration_ms}ms"

        if not silent:
            level = "warning" if response.status_code >= 400 else "info"
            getattr(logger, level)(
                "Request completed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": path,
                    "status_code": response.status_code,
                    "duration_ms": duration_ms,
                },
            )

        return response
