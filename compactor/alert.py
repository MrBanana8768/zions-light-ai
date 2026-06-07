"""
compactor.alert — V2.3 Theme 4: optional failure alerting.

A single configurable webhook that the boot self-test and the backup daemon
POST to when they FAIL, so the owner finds out before the user does. Off by
default — set COMPACTOR_ALERT_WEBHOOK to a URL to turn it on.

Design rules (an alerter must never break the thing it watches):
  - No-op when the webhook is unset.
  - Short timeout; all errors swallowed and logged at warning, never raised.
  - Generic JSON payload that also carries `text` (Slack) and `content`
    (Discord) so the same URL works with the common chat webhooks or any
    generic receiver.

Both a sync `notify()` (for the selftest/backup CLI scripts) and an async
`notify_async()` (for the compactor's event loop) are provided.
"""

from __future__ import annotations

import logging
import os
import socket
import time

import httpx

logger = logging.getLogger("compactor.alert")

WEBHOOK = os.environ.get("COMPACTOR_ALERT_WEBHOOK", "").strip()
TIMEOUT_S = float(os.environ.get("COMPACTOR_ALERT_TIMEOUT_S", "10") or 10)
_HOST = socket.gethostname()


def enabled() -> bool:
    return bool(WEBHOOK)


def _payload(service: str, status: str, detail: str, extra: dict | None) -> dict:
    summary = f"[zions-light-ai/{service}] {status.upper()}: {detail}"
    body = {
        "service": service,
        "status": status,
        "detail": detail,
        "host": _HOST,
        "ts": int(time.time()),
        # For Slack ("text") and Discord ("content"); generic receivers can
        # read the structured fields above and ignore these.
        "text": summary,
        "content": summary,
    }
    if extra:
        body["extra"] = extra
    return body


def notify(service: str, status: str, detail: str, *, extra: dict | None = None) -> bool:
    """Synchronous fire-and-forget alert (for the selftest/backup scripts).
    Returns True if a request was sent, False if disabled or it failed.
    Never raises."""
    if not WEBHOOK:
        return False
    try:
        r = httpx.post(WEBHOOK, json=_payload(service, status, detail, extra),
                       timeout=TIMEOUT_S)
        if r.status_code >= 400:
            logger.warning(f"alert webhook returned HTTP {r.status_code}")
            return False
        return True
    except Exception as e:
        logger.warning(f"alert webhook failed (non-fatal): {type(e).__name__}: {e}")
        return False


async def notify_async(service: str, status: str, detail: str,
                       *, extra: dict | None = None) -> bool:
    """Async variant for the compactor event loop. Never raises."""
    if not WEBHOOK:
        return False
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_S) as c:
            r = await c.post(WEBHOOK, json=_payload(service, status, detail, extra))
        if r.status_code >= 400:
            logger.warning(f"alert webhook returned HTTP {r.status_code}")
            return False
        return True
    except Exception as e:
        logger.warning(f"alert webhook failed (non-fatal): {type(e).__name__}: {e}")
        return False
