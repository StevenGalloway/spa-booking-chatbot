import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict


class JSONFormatter(logging.Formatter):
    """
    Emits every log record as a single-line JSON object for ingestion by
    log aggregation platforms (Datadog, Loki, CloudWatch, etc.).

    Standard fields:
        timestamp   ISO-8601 UTC
        level       DEBUG | INFO | WARNING | ERROR | CRITICAL
        service     aura-backend
        logger      dotted module path
        message     human-readable log text
        module      Python module name
        function    function name
        line        source line number

    Optional context fields (added via extra={}):
        request_id  UUID for request correlation
        user_id     session user identifier
        intent      detected NLP intent
        duration_ms request or operation latency
    """

    SERVICE_NAME = "aura-backend"

    def format(self, record: logging.LogRecord) -> str:
        entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "service": self.SERVICE_NAME,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Propagate optional request-scoped context
        for field in ("request_id", "user_id", "intent", "duration_ms", "path", "method", "status_code", "client"):
            if hasattr(record, field):
                entry[field] = getattr(record, field)

        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(entry, default=str)


class TextFormatter(logging.Formatter):
    """Human-readable format for local development."""

    FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    def __init__(self) -> None:
        super().__init__(fmt=self.FMT, datefmt="%Y-%m-%d %H:%M:%S")


def setup_logging(level: str = "INFO", log_format: str = "json") -> None:
    """
    Configure the root logger.  Call once at application startup in main.py.

    Args:
        level:      Log level string (DEBUG | INFO | WARNING | ERROR | CRITICAL)
        log_format: "json" for structured output, "text" for human-readable
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level, logging.INFO))

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter() if log_format == "json" else TextFormatter())

    root.handlers.clear()
    root.addHandler(handler)

    # Suppress high-volume, low-signal library loggers
    for noisy_logger in ("uvicorn.access", "httpx", "httpcore", "multipart"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger. Use dotted module paths, e.g. 'aura.api.chat'."""
    return logging.getLogger(name)
