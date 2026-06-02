"""
Shared helpers for the Tier-3 integration validation suite.

Lives outside the Docker image (this directory is excluded by .dockerignore
and never COPYed by the Dockerfile). Hits the deployed compactor's HTTP API
via httpx.

Environment variables (read at import time):
    ZIONS_TEST_BASE_URL   REQUIRED. The compactor's public URL.
                          e.g. https://abc123-8080.proxy.runpod.net
                          (or http://localhost:8080 for an SSH-tunneled
                          local dev run).
    ZIONS_TEST_ADMIN_URL  OPTIONAL. If set, tests that need /admin/*
                          endpoints (state inspection, /forget verification)
                          will use this URL. If unset, those tests SKIP
                          gracefully rather than fail — so a "basic"
                          validation run works without admin access.
                          Usually the same as BASE_URL if you've
                          SSH-tunneled to localhost:8080, OR an internal
                          URL if you've set COMPACTOR_ADMIN_BIND=0.0.0.0
                          on the pod (NOT recommended for production).
    ZIONS_TEST_MODEL      OPTIONAL. The model name to send in chat
                          requests. Auto-detected from /v1/models if unset.
    ZIONS_TEST_TIMEOUT    OPTIONAL, seconds. Per-request HTTP timeout.
                          Default 120 (chat can be slow with rollups firing).
    ZIONS_TEST_TAIL_WAIT  OPTIONAL, seconds. How long to wait for the
                          async post-response tail (fact extraction +
                          rollup) to finish before assertions. Default 8.

The harness deliberately does NOT import anything from compactor/ — these
tests must run with nothing but pytest + httpx installed, against a black-
box deployment. That matches how a real CI gate would invoke them.
"""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass
from typing import Any

import httpx
import pytest

# ---------------------------------------------------------------------------
# Configuration (read once at import)
# ---------------------------------------------------------------------------

BASE_URL = os.environ.get("ZIONS_TEST_BASE_URL", "").rstrip("/")
ADMIN_URL = (os.environ.get("ZIONS_TEST_ADMIN_URL") or BASE_URL).rstrip("/") or None
MODEL = os.environ.get("ZIONS_TEST_MODEL", "")
TIMEOUT = float(os.environ.get("ZIONS_TEST_TIMEOUT", "120") or 120)
TAIL_WAIT = float(os.environ.get("ZIONS_TEST_TAIL_WAIT", "8") or 8)

# Only "true" if user explicitly opted in to admin tests. Bare BASE_URL
# fallback for ADMIN_URL doesn't count — admin tests require an explicit
# decision because they may need extra pod-side config.
ADMIN_ENABLED = bool(os.environ.get("ZIONS_TEST_ADMIN_URL"))


def require_base_url() -> None:
    """Pytest fixtures call this so missing config produces a clear error."""
    if not BASE_URL:
        pytest.exit(
            "ZIONS_TEST_BASE_URL is not set. Run with e.g.:\n"
            "  ZIONS_TEST_BASE_URL=https://<pod>-8080.proxy.runpod.net "
            "pytest tests/integration/"
        )


def skip_if_no_admin(reason: str = "admin endpoint required") -> None:
    """Tests that need /admin/* call this in setup. Cleanly skips when
    ZIONS_TEST_ADMIN_URL is unset, rather than failing with a 403/404."""
    if not ADMIN_ENABLED:
        pytest.skip(f"{reason} (set ZIONS_TEST_ADMIN_URL to enable)")


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _client(url_base: str) -> httpx.Client:
    return httpx.Client(base_url=url_base, timeout=TIMEOUT)


def list_models() -> list[str]:
    """GET /v1/models → list of model ids the compactor proxies. Useful
    both as a liveness check and for auto-detecting MODEL."""
    require_base_url()
    with _client(BASE_URL) as c:
        r = c.get("/v1/models")
        r.raise_for_status()
        data = r.json()
    return [m.get("id") for m in (data.get("data") or []) if m.get("id")]


def resolve_model() -> str:
    """If ZIONS_TEST_MODEL is set, use it; otherwise pick the first
    advertised by /v1/models. Cached after first call."""
    global MODEL
    if MODEL:
        return MODEL
    models = list_models()
    if not models:
        pytest.fail("/v1/models returned no models — vLLM not ready?")
    MODEL = models[0]
    return MODEL


def health_ok() -> bool:
    """GET /health → True if compactor is up + admits to being ok."""
    require_base_url()
    try:
        with _client(BASE_URL) as c:
            r = c.get("/health")
        return r.status_code == 200 and r.json().get("status") == "ok"
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Conversation helpers
# ---------------------------------------------------------------------------


def fresh_conv_id() -> str:
    """Generate a sentinel conv_id with a recognizable prefix. Easy to
    spot in `/admin/conversations` listings and easy to clean up later."""
    return f"itest-{uuid.uuid4().hex[:12]}"


@dataclass
class ChatResult:
    response_text: str
    raw: dict
    status_code: int
    conv_id: str
    turns_sent: int


def chat(
    user_msg: str,
    *,
    conv_id: str,
    system: str | None = None,
    prior_turns: list[dict] | None = None,
    max_tokens: int = 128,
    temperature: float = 0.0,
) -> ChatResult:
    """Send one non-streaming chat completion to the compactor.

    `conv_id` is propagated via the body.metadata.chat_id field that the
    OpenWebUI Function uses — the compactor's `resolve_conv_id` reads it
    AND the X-Conversation-Id header. We set both for belt-and-suspenders.

    `prior_turns` is the conversation context that came BEFORE this user
    message (alternating user/assistant pairs). The new user message gets
    appended automatically. Pass [] for a fresh conversation.
    """
    require_base_url()
    model = resolve_model()
    msgs: list[dict] = []
    if system:
        msgs.append({"role": "system", "content": system})
    if prior_turns:
        msgs.extend(prior_turns)
    msgs.append({"role": "user", "content": user_msg})

    body = {
        "model": model,
        "messages": msgs,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
        "metadata": {"chat_id": conv_id},
    }
    headers = {"X-Conversation-Id": conv_id}

    with _client(BASE_URL) as c:
        r = c.post("/v1/chat/completions", json=body, headers=headers)

    text = ""
    try:
        text = (r.json().get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
    except Exception:
        pass

    return ChatResult(
        response_text=text,
        raw=r.json() if r.headers.get("content-type", "").startswith("application/json") else {},
        status_code=r.status_code,
        conv_id=conv_id,
        turns_sent=len(msgs),
    )


def extend_history(
    prior_turns: list[dict], user_msg: str, assistant_msg: str
) -> list[dict]:
    """Append a completed exchange to a conversation history list. Used to
    build up multi-turn contexts test-by-test without losing state."""
    return prior_turns + [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant_msg},
    ]


def wait_for_async_tail(seconds: float | None = None) -> None:
    """The compactor's post-response work (fact extraction, episodic
    indexing, summary rollup) runs as fire-and-forget tasks AFTER the
    chat response is sent. Tests that assert "X was saved" need to wait.

    Polling the admin endpoint would be more elegant, but requires
    ADMIN_ENABLED — this simple sleep works in both modes.
    """
    time.sleep(seconds if seconds is not None else TAIL_WAIT)


# ---------------------------------------------------------------------------
# Admin helpers — gracefully skip when ADMIN_URL is unset
# ---------------------------------------------------------------------------


def admin_conv_summary(conv_id: str) -> dict:
    """GET /admin/conversations/<id> → JSON with facts/episodic/summary
    counts. Test caller must skip_if_no_admin() first."""
    assert ADMIN_URL, "admin_conv_summary requires ZIONS_TEST_ADMIN_URL"
    with _client(ADMIN_URL) as c:
        r = c.get(f"/admin/conversations/{conv_id}")
        r.raise_for_status()
        return r.json()


def admin_get_facts(conv_id: str) -> list[dict]:
    """GET /admin/conversations/<id>/facts → list of fact dicts."""
    assert ADMIN_URL, "admin_get_facts requires ZIONS_TEST_ADMIN_URL"
    with _client(ADMIN_URL) as c:
        r = c.get(f"/admin/conversations/{conv_id}/facts")
        r.raise_for_status()
        return r.json().get("facts", [])


def admin_get_summary(conv_id: str) -> dict:
    """GET /admin/conversations/<id>/summary → raw summarizer state."""
    assert ADMIN_URL, "admin_get_summary requires ZIONS_TEST_ADMIN_URL"
    with _client(ADMIN_URL) as c:
        r = c.get(f"/admin/conversations/{conv_id}/summary")
        r.raise_for_status()
        return r.json()


def admin_forget(conv_id: str) -> dict:
    """DELETE /admin/conversations/<id>/facts → clears all three memory
    layers (facts, episodic, summary)."""
    assert ADMIN_URL, "admin_forget requires ZIONS_TEST_ADMIN_URL"
    with _client(ADMIN_URL) as c:
        r = c.delete(f"/admin/conversations/{conv_id}/facts")
        r.raise_for_status()
        return r.json()


def admin_safe_forget(conv_id: str) -> None:
    """Best-effort cleanup — used in test teardown. Swallows errors so
    a teardown failure doesn't mask the real test failure. No-op when
    admin URL is unset."""
    if not ADMIN_ENABLED:
        return
    try:
        admin_forget(conv_id)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Assertion helpers (semantic, not just textual)
# ---------------------------------------------------------------------------


def response_mentions(text: str, *needles: str) -> bool:
    """Case-insensitive substring check across multiple needles. The LLM
    won't reproduce text verbatim, so individual tests should be lenient
    — pass several plausible phrasings as needles."""
    if not text:
        return False
    low = text.lower()
    return any(n.lower() in low for n in needles)


def assert_response_mentions(text: str, *needles: str, hint: str = "") -> None:
    """Fail with a useful message that includes the actual response. The
    hint helps a human reader understand what was expected semantically."""
    if response_mentions(text, *needles):
        return
    msg = (
        f"Expected response to mention one of {list(needles)}"
        + (f" ({hint})" if hint else "")
        + f"\nActual response was:\n---\n{text[:500]}\n---"
    )
    pytest.fail(msg)
