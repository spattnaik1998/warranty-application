"""Structured logging.

Emits one JSON object per log line with the workflow fields the project
mandates (stage, agent, tool, query, video_url, arxiv_id, status,
duration_ms) plus Warrant-specific fields (delegation_class, posterior_shift,
distortion). Use :func:`log_event` for structured events and
:func:`get_logger` to obtain a configured logger.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any

# Canonical field order for readability when scanning logs.
_FIELD_ORDER = (
    "stage",
    "agent",
    "tool",
    "query",
    "video_url",
    "arxiv_id",
    "delegation_class",
    "status",
    "duration_ms",
    "posterior_shift",
    "distortion",
)

_CONFIGURED = False


class _JsonFormatter(logging.Formatter):
    """Render log records as compact JSON, merging structured `extra` fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            # Preserve the canonical order, then append any extras.
            for key in _FIELD_ORDER:
                if key in fields:
                    payload[key] = fields[key]
            for key, value in fields.items():
                if key not in payload:
                    payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(level: str | None = None) -> None:
    """Install the JSON formatter on the root Warrant logger once."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    lvl = (level or os.getenv("WARRANT_LOG_LEVEL", "INFO")).upper()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_JsonFormatter())
    root = logging.getLogger("warrant")
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(lvl)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the `warrant` namespace."""
    configure_logging()
    return logging.getLogger(f"warrant.{name}")


def log_event(logger: logging.Logger, msg: str, *, level: int = logging.INFO, **fields: Any) -> None:
    """Emit a structured event.

    Example::

        log_event(log, "delegation admitted", stage="orchestrate",
                  agent="orchestrator", delegation_class="INJECTOR",
                  tool="arxiv", status="ok", duration_ms=812)
    """
    logger.log(level, msg, extra={"fields": fields})
