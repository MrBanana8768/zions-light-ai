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

Also home to `log_once` (v3.1 P0-2b): the gate that lets a handler on the
request path report a failure without becoming its own denial of service.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import sys

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
    # stdout, NOT the StreamHandler() default of stderr. Under supervisord
    # stderr is compactor-error.log, so every `conv_id=… source=… msgs=…`
    # line lived there while compactor.log held uvicorn access noise — and
    # OPERATIONS.md:44 points the operator at compactor.log. The 2026-08-24
    # investigation turned on two adjacent lines in the file the runbook
    # says not to read. (v3.1 P0-2 / F17a.)
    handler = logging.StreamHandler(sys.stdout)
    if _log_format() == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(_TEXT_FORMAT))
    root.addHandler(handler)


# ---------------------------------------------------------------------------
# Once-per-process reporting gate (v3.1 P0-2b)
# ---------------------------------------------------------------------------

# Call sites that have already reported. Deliberately never cleared: the
# point is one line per process, not one line per failure.
_logged_once: set[str] = set()


def log_once(key: str) -> bool:
    """True the first time `key` is seen in this process, False after.

    The P0-2b sweep gave a voice to handlers that fire on the request path
    or on the 30s healthcheck — `gather_memory_stats` alone runs once per
    conversation per probe. Logging every one of those turns the fix into
    its own outage, so the noisy sites read:

        except Exception as e:
            if logsetup.log_once("health.stats.facts"):
                logger.warning(...)

    `key` is the call site, not the error, so a permanently broken probe
    says so once and then stops. Anything that needs a per-occurrence count
    surfaces it in its own output instead — see the `unreadable` block in
    health.gather_memory_stats. The log is for noticing, not for counting.
    """
    if key in _logged_once:
        return False
    _logged_once.add(key)
    return True


def _reset_log_once_for_tests() -> None:
    """Test helper — forget which call sites have already reported."""
    _logged_once.clear()
