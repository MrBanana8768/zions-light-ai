"""
CPU-only Tier-1 tests for compactor.alert (V2.3 Theme 4).

An alerter must never break what it monitors: no-op when unset, errors
swallowed, payload usable by Slack/Discord/generic webhooks. Both sync and
async paths.

Run: python test_alert.py
"""

import asyncio
import importlib
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import alert


def assert_eq(a, b, label):
    if a != b:
        print(f"FAIL {label}: expected {b!r}, got {a!r}")
        sys.exit(1)
    print(f"  ok   {label}")


def assert_true(c, label):
    if not c:
        print(f"FAIL {label}")
        sys.exit(1)
    print(f"  ok   {label}")


def _reload_with_webhook(url: str | None):
    if url is None:
        os.environ.pop("COMPACTOR_ALERT_WEBHOOK", None)
    else:
        os.environ["COMPACTOR_ALERT_WEBHOOK"] = url
    importlib.reload(alert)


# ---------------------------------------------------------------------------
# Payload shape
# ---------------------------------------------------------------------------

def test_payload_has_structured_and_chat_fields():
    print("\n[test] payload: structured fields + text/content for chat webhooks")
    p = alert._payload("backup", "fail", "disk full", {"k": "v"})
    assert_eq(p["service"], "backup", "service")
    assert_eq(p["status"], "fail", "status")
    assert_eq(p["detail"], "disk full", "detail")
    assert_true("host" in p and "ts" in p, "host + ts present")
    assert_true("backup" in p["text"] and "FAIL" in p["text"], "slack text summary")
    assert_eq(p["text"], p["content"], "discord content mirrors text")
    assert_eq(p["extra"], {"k": "v"}, "extra passed through")


# ---------------------------------------------------------------------------
# No-op when unset
# ---------------------------------------------------------------------------

def test_notify_noop_when_unset():
    print("\n[test] notify: no webhook → no-op, returns False, no HTTP call")
    _reload_with_webhook(None)
    assert_eq(alert.enabled(), False, "disabled")
    with patch.object(alert.httpx, "post",
                      MagicMock(side_effect=AssertionError("must not POST"))):
        assert_eq(alert.notify("backup", "fail", "x"), False, "returns False")


def test_notify_async_noop_when_unset():
    print("\n[test] notify_async: no webhook → no-op False")
    _reload_with_webhook(None)
    out = asyncio.run(alert.notify_async("selftest", "fail", "x"))
    assert_eq(out, False, "async returns False when unset")


# ---------------------------------------------------------------------------
# Sends when set
# ---------------------------------------------------------------------------

def test_notify_posts_when_set():
    print("\n[test] notify: webhook set → POSTs payload, returns True")
    _reload_with_webhook("http://hook.example/x")
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return MagicMock(status_code=200)

    with patch.object(alert.httpx, "post", fake_post):
        ok = alert.notify("backup", "fail", "verify failed")
    assert_eq(ok, True, "returns True on 2xx")
    assert_eq(captured["url"], "http://hook.example/x", "posted to webhook")
    assert_eq(captured["json"]["service"], "backup", "payload carried")


def test_notify_async_posts_when_set():
    print("\n[test] notify_async: webhook set → POSTs, returns True")
    _reload_with_webhook("http://hook.example/y")

    async def go():
        mock_resp = MagicMock(status_code=204)
        client = MagicMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        client.post = AsyncMock(return_value=mock_resp)
        with patch.object(alert.httpx, "AsyncClient", lambda *a, **k: client):
            return await alert.notify_async("selftest", "fail", "boot failed")

    assert_eq(asyncio.run(go()), True, "async returns True on 2xx")


# ---------------------------------------------------------------------------
# Never raises
# ---------------------------------------------------------------------------

def test_notify_swallows_errors():
    print("\n[test] notify: network error swallowed, returns False (never raises)")
    _reload_with_webhook("http://hook.example/z")
    with patch.object(alert.httpx, "post",
                      MagicMock(side_effect=ConnectionError("boom"))):
        out = alert.notify("backup", "fail", "x")
    assert_eq(out, False, "returns False on error, no raise")


def test_notify_treats_4xx_5xx_as_failure():
    print("\n[test] notify: webhook 500 → returns False (not a successful alert)")
    _reload_with_webhook("http://hook.example/z")
    with patch.object(alert.httpx, "post",
                      MagicMock(return_value=MagicMock(status_code=500))):
        out = alert.notify("backup", "fail", "x")
    assert_eq(out, False, "5xx → False")


def _all():
    return [
        test_payload_has_structured_and_chat_fields,
        test_notify_noop_when_unset,
        test_notify_async_noop_when_unset,
        test_notify_posts_when_set,
        test_notify_async_posts_when_set,
        test_notify_swallows_errors,
        test_notify_treats_4xx_5xx_as_failure,
    ]


if __name__ == "__main__":
    try:
        for t in _all():
            t()
        print("\nAll alert smoke tests passed.")
    finally:
        _reload_with_webhook(None)
