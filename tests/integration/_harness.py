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
    """Coarse fallback: just sleep for `seconds` (or TAIL_WAIT default).

    Use this for tests that can't observe completion through admin
    endpoints (i.e. when ADMIN_ENABLED is False). For admin-mode tests,
    prefer the deterministic polling helpers below — they return as soon
    as the expected state appears AND give the work up to `max_wait`
    seconds rather than a fixed sleep.
    """
    time.sleep(seconds if seconds is not None else TAIL_WAIT)


def wait_for_facts(
    conv_id: str,
    *,
    min_count: int = 1,
    max_wait: float = 30.0,
    poll_interval: float = 2.0,
) -> list[dict]:
    """Poll /admin/conversations/<id>/facts until at least `min_count`
    facts are persisted, or `max_wait` seconds elapse. Returns the final
    fact list (which may still be shorter than min_count if the timeout
    fired — the caller's assertion then fails with a clear message).

    Why polling beats a fixed sleep: fact extraction is a real LLM call
    against the model. On Magnum 12B / A40 it's typically ~5-15s but can
    spike under load. A fixed 8s sleep was sometimes catching the admin
    GET BEFORE extraction finished writing — false negative. Polling
    returns as soon as the expected state arrives (so fast paths stay
    fast) AND gives slow paths a generous ceiling.

    Falls back to a coarse sleep when admin endpoints aren't reachable
    (otherwise we'd be looping uselessly against 403s).
    """
    if not ADMIN_ENABLED:
        time.sleep(max_wait if max_wait > TAIL_WAIT else TAIL_WAIT)
        return []
    deadline = time.monotonic() + max_wait
    facts: list[dict] = []
    while True:
        try:
            facts = admin_get_facts(conv_id)
        except Exception:
            facts = []
        if len(facts) >= min_count or time.monotonic() >= deadline:
            return facts
        time.sleep(poll_interval)


def wait_for_indexed_exchanges(
    conv_id: str,
    *,
    min_count: int = 1,
    max_wait: float = 30.0,
    poll_interval: float = 2.0,
) -> int:
    """Poll /admin/conversations/<id> until episodic.indexed_exchanges
    reaches `min_count`, or `max_wait` elapses. Returns the final count.

    Episodic indexing (one embedding generated, one ChromaDB upsert) is
    typically faster than fact extraction (no LLM call) — usually <1s —
    but on a cold pod the first embedding can take longer because
    fastembed lazy-loads the ONNX model. Polling handles both ends.

    Falls back to a coarse sleep when admin endpoints aren't reachable.
    """
    if not ADMIN_ENABLED:
        time.sleep(TAIL_WAIT)
        return 0
    deadline = time.monotonic() + max_wait
    count = 0
    while True:
        try:
            summary = admin_conv_summary(conv_id)
            count = int((summary.get("episodic") or {}).get("indexed_exchanges") or 0)
        except Exception:
            count = 0
        if count >= min_count or time.monotonic() >= deadline:
            return count
        time.sleep(poll_interval)


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


# V2.1 Phase 6 — health / selftest / portability helpers.

def health_full() -> tuple[int, dict]:
    """GET /health/full → (status_code, json_body). Status code is part
    of the contract (200 ok/degraded, 503 down), so we return it not
    just the body. No admin URL required — /health/full is public.
    """
    with _client(BASE_URL) as c:
        r = c.get("/health/full")
        # Don't raise_for_status — 503 is a valid value here.
        try:
            body = r.json()
        except Exception:
            body = {}
        return r.status_code, body


def admin_selftest(round_trip: bool = True) -> tuple[int, dict]:
    """GET /admin/selftest → (status_code, report). 200=pass, 503=fail.
    Skip-friendly: returns (0, {}) when admin not configured."""
    if not ADMIN_ENABLED:
        return 0, {}
    with _client(ADMIN_URL) as c:
        r = c.get(
            "/admin/selftest",
            params={"round_trip": "true" if round_trip else "false"},
            timeout=300.0,
        )
        try:
            body = r.json()
        except Exception:
            body = {}
        return r.status_code, body


def admin_export(conv_id: str) -> dict:
    """GET /admin/conversations/<id>/export → full bundle."""
    assert ADMIN_URL, "admin_export requires ZIONS_TEST_ADMIN_URL"
    with _client(ADMIN_URL) as c:
        r = c.get(f"/admin/conversations/{conv_id}/export")
        r.raise_for_status()
        return r.json()


def admin_import(bundle: dict, *, target_conv_id: str | None = None,
                 overwrite: bool = False) -> tuple[int, dict]:
    """POST /admin/conversations/import → (status_code, json_body).
    Status is part of contract (400 = malformed bundle / refused overwrite,
    200 = success)."""
    assert ADMIN_URL, "admin_import requires ZIONS_TEST_ADMIN_URL"
    body: dict = {"bundle": bundle, "overwrite": overwrite}
    if target_conv_id is not None:
        body["target_conv_id"] = target_conv_id
    with _client(ADMIN_URL) as c:
        r = c.post("/admin/conversations/import", json=body)
        try:
            data = r.json()
        except Exception:
            data = {}
        return r.status_code, data


def admin_fork(conv_id: str, *, new_conv_id: str | None = None) -> dict:
    """POST /admin/conversations/<id>/fork → result dict."""
    assert ADMIN_URL, "admin_fork requires ZIONS_TEST_ADMIN_URL"
    body: dict = {}
    if new_conv_id is not None:
        body["new_conv_id"] = new_conv_id
    with _client(ADMIN_URL) as c:
        r = c.post(f"/admin/conversations/{conv_id}/fork", json=body)
        r.raise_for_status()
        return r.json()


# V2.1 Phase 7 — dedup + archival helpers.

def admin_dedup(conv_id: str) -> dict:
    """POST /admin/conversations/<id>/dedup → {conv_id, before, after, removed}."""
    assert ADMIN_URL, "admin_dedup requires ZIONS_TEST_ADMIN_URL"
    with _client(ADMIN_URL) as c:
        # Dedup can call the LLM N times; generous timeout.
        r = c.post(f"/admin/conversations/{conv_id}/dedup", timeout=300.0)
        r.raise_for_status()
        return r.json()


def admin_archive_stale(conv_id: str, older_than_days: int | None = None) -> dict:
    """POST /admin/conversations/<id>/archive → {kept, archived, ...}."""
    assert ADMIN_URL, "admin_archive_stale requires ZIONS_TEST_ADMIN_URL"
    params = {}
    if older_than_days is not None:
        params["older_than_days"] = older_than_days
    with _client(ADMIN_URL) as c:
        r = c.post(f"/admin/conversations/{conv_id}/archive", params=params)
        r.raise_for_status()
        return r.json()


def admin_get_archive(conv_id: str) -> list[dict]:
    """GET /admin/conversations/<id>/archive → list of archived facts."""
    assert ADMIN_URL, "admin_get_archive requires ZIONS_TEST_ADMIN_URL"
    with _client(ADMIN_URL) as c:
        r = c.get(f"/admin/conversations/{conv_id}/archive")
        r.raise_for_status()
        return r.json().get("archived", [])


def admin_restore_from_archive(
    conv_id: str, text_substring: str | None = None
) -> dict:
    """POST /admin/conversations/<id>/restore → {restored, filter, ...}."""
    assert ADMIN_URL, "admin_restore_from_archive requires ZIONS_TEST_ADMIN_URL"
    body: dict = {}
    if text_substring is not None:
        body["text_substring"] = text_substring
    with _client(ADMIN_URL) as c:
        r = c.post(f"/admin/conversations/{conv_id}/restore", json=body)
        r.raise_for_status()
        return r.json()


# V2.1 Phase 8 — persona helpers.

def admin_get_persona(conv_id: str) -> dict | None:
    """GET /admin/conversations/<id>/persona → record or None on 404."""
    assert ADMIN_URL, "admin_get_persona requires ZIONS_TEST_ADMIN_URL"
    with _client(ADMIN_URL) as c:
        r = c.get(f"/admin/conversations/{conv_id}/persona")
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()


def admin_set_persona(conv_id: str, text: str) -> tuple[int, dict]:
    """POST /admin/conversations/<id>/persona body: {text} → (status, body)."""
    assert ADMIN_URL, "admin_set_persona requires ZIONS_TEST_ADMIN_URL"
    with _client(ADMIN_URL) as c:
        r = c.post(
            f"/admin/conversations/{conv_id}/persona", json={"text": text},
        )
        try:
            body = r.json()
        except Exception:
            body = {}
        return r.status_code, body


def admin_delete_persona(conv_id: str) -> dict:
    """DELETE /admin/conversations/<id>/persona → {deleted, ...}."""
    assert ADMIN_URL, "admin_delete_persona requires ZIONS_TEST_ADMIN_URL"
    with _client(ADMIN_URL) as c:
        r = c.delete(f"/admin/conversations/{conv_id}/persona")
        r.raise_for_status()
        return r.json()


def admin_list_personas() -> list[dict]:
    """GET /admin/personas → library."""
    assert ADMIN_URL, "admin_list_personas requires ZIONS_TEST_ADMIN_URL"
    with _client(ADMIN_URL) as c:
        r = c.get("/admin/personas")
        r.raise_for_status()
        return r.json().get("personas", [])


def admin_inherit_persona(target_conv_id: str, source_conv_id: str) -> tuple[int, dict]:
    """POST /admin/conversations/<id>/inherit-persona → (status, body)."""
    assert ADMIN_URL, "admin_inherit_persona requires ZIONS_TEST_ADMIN_URL"
    with _client(ADMIN_URL) as c:
        r = c.post(
            f"/admin/conversations/{target_conv_id}/inherit-persona",
            json={"source_conv_id": source_conv_id},
        )
        try:
            body = r.json()
        except Exception:
            body = {}
        return r.status_code, body


# V2.3 Theme 1 — backup helpers.

def admin_list_backups() -> dict:
    """GET /admin/backups → {backups: [...], info: {...}}."""
    assert ADMIN_URL, "admin_list_backups requires ZIONS_TEST_ADMIN_URL"
    with _client(ADMIN_URL) as c:
        r = c.get("/admin/backups")
        r.raise_for_status()
        return r.json()


def admin_run_backup() -> tuple[int, dict]:
    """POST /admin/backups → (status_code, report). 200 ok / 503 fail.
    Generous timeout — a real backup snapshots the db + tars + verifies."""
    assert ADMIN_URL, "admin_run_backup requires ZIONS_TEST_ADMIN_URL"
    with _client(ADMIN_URL) as c:
        r = c.post("/admin/backups", timeout=300.0)
        try:
            body = r.json()
        except Exception:
            body = {}
        return r.status_code, body


def admin_verify_backup(name: str | None = None) -> tuple[int, dict]:
    """GET /admin/backups/verify[?name=] → (status_code, body)."""
    assert ADMIN_URL, "admin_verify_backup requires ZIONS_TEST_ADMIN_URL"
    params = {"name": name} if name else {}
    with _client(ADMIN_URL) as c:
        r = c.get("/admin/backups/verify", params=params, timeout=120.0)
        try:
            body = r.json()
        except Exception:
            body = {}
        return r.status_code, body


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
