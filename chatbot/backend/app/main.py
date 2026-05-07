"""
AURA Platform — FastAPI Application Entry Point

Startup sequence:
  1. Configure structured logging
  2. Validate settings from environment
  3. Register middleware (request logging, CORS)
  4. Mount API routers (chat, health, auth)
  5. Log startup summary
"""

import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import chatbot, health
from app.api import auth
from app.core.config import settings
from app.core.logging_config import get_logger, setup_logging
from app.middleware.logging_middleware import RequestLoggingMiddleware

# ---------------------------------------------------------------------------
# Logging — must be set up before any other module emits log lines
# ---------------------------------------------------------------------------
setup_logging(level=settings.log_level, log_format=settings.log_format)
logger = get_logger("aura.main")

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title=settings.app_name,
    description=settings.app_description,
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "chat", "description": "Conversational AI endpoints"},
        {"name": "auth", "description": "Authentication and token management"},
        {"name": "observability", "description": "Health probes and metrics"},
    ],
)

# ---------------------------------------------------------------------------
# Middleware (order matters — outermost middleware runs first)
# ---------------------------------------------------------------------------

# 1. Request ID + structured access logging
app.add_middleware(RequestLoggingMiddleware)

# 2. CORS — tighten CORS_ORIGINS in production via environment variable
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.effective_cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID", "X-API-Key"],
    expose_headers=["X-Request-ID", "X-Response-Time"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(chatbot.router, prefix="/api/v1", tags=["chat"])
app.include_router(health.router, tags=["observability"])
app.include_router(auth.router, prefix="/api/v1", tags=["auth"])

# ---------------------------------------------------------------------------
# Lifecycle events
# ---------------------------------------------------------------------------

_startup_time = time.time()


@app.on_event("startup")
async def on_startup() -> None:
    logger.info(
        "AURA Platform starting",
        extra={
            "app_name": settings.app_name,
            "version": settings.app_version,
            "environment": settings.environment,
            "auth_enabled": settings.auth_enabled,
            "log_level": settings.log_level,
        },
    )


@app.on_event("shutdown")
async def on_shutdown() -> None:
    uptime = round(time.time() - _startup_time, 1)
    logger.info("AURA Platform shutting down", extra={"uptime_seconds": uptime})


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root() -> dict:
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health/ready",
    }
