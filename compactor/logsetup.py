"""
compactor.logsetup — V2.3 Theme 4: structured logging.

Centralizes log configuration for the compactor and its sidecars (selftest,
backup) so they all honor one switch:

    COMPACTOR_LOG_FORMAT = text   (default — human-readable, what the web
                                   terminal has always shown)
                         = json   (one JSON object per line, for grepping /
                                   shipping to a log aggregator)

Text is the default so existing operator habits (tail -f, eyeballing
compactor.log) are unchanged. Set json when you're forwarding logs somewhere
that wants structured fields.

No third-party dependency — the JSON formatter is ~15 lines of stdlib,
keeping the compactor venv lean.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os

_TEXT_FORMAT = "%(asctime)s %(name)s %(levelname)s %(message)s"


class JsonFormatter(logging.Formatter):
    """One compact JSON object per log line. Includes exception text when
    present so tracebacks stay attached to their record."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": _dt.datetime.fromtimestamp(
                record.created, _dt.timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def _log_format() -> str:
    return os.environ.get("COMPACTOR_LOG_FORMAT", "text").strip().lower()


def configure(level: int = logging.INFO) -> None:
    """Install the chosen formatter on the root logger. Idempotent — clears
    existing handlers first so calling it from multiple entry points (main,
    selftest, backup) doesn't stack duplicate handlers."""
    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler()
    if _log_format() == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(_TEXT_FORMAT))
    root.addHandler(handler)
