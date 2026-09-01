"""
v3.1.5 — main.py must pass `query_text` to facts.select_for_injection.

THE GAP THIS CLOSES. test_facts_injection.py covers select_for_injection
itself thoroughly, including the query_text=None path. Nothing covered
whether main.py ever PASSES query_text. Deleting `query_text=last_user_text`
from the hot path left the whole suite green — verified, and recorded as an
open gate finding on 2026-08-31.

WHY IT NOW MATTERS. That was tolerable while the injection budget was 800:
ranking off meant LRU order, and LRU order over a budget that fit most of the
store was close enough to "everything". v3.1.5 lowers the default to 400
because ~91 injected bullets per turn were making replies formulaic. At 400
the budget genuinely binds, and the two behaviours diverge:

  with query_text  — the ~26 facts THIS turn is about, rotating by topic.
  without          — a fixed most-recently-used prefix, the degenerate FIFO
                     order F1 exists to replace, frozen in place forever.

So the silent regression stops being a missed optimisation and becomes
strictly worse than the 800 it replaced, while every existing test still
passes and the logs still read `injected memory [26fact(s) ...]` either way.
There is no signal anywhere that distinguishes the two. Hence this file.

BOTH CALL SITES. This branch has been bitten sixteen times by a fix landing
on one call site and not its identical twin, so both are asserted here:
the request hot path (main.py, "--- Facts (Phase 2) ---") and the fallback
default inside _async_tail.

Only synthetic conversation content appears below (project rule: this repo
is public).

    python test_facts_wiring.py
"""

import asyncio
import json
import os
import shutil
import sys
import tempfile
from unittest.mock import patch

os.environ.setdefault("MODEL_REPO", "test-model")
os.environ.setdefault("VLLM_URL", "http://stub:8000")
os.environ["COMPACTOR_RAG_ENABLED"] = "false"  # no ChromaDB/fastembed here

_TMP_ROOT = tempfile.mkdtemp(prefix="compactor-test-facts-wiring-")
os.environ["COMPACTOR_STORAGE_ROOT"] = _TMP_ROOT

import facts  # noqa: E402
import main  # noqa: E402
import memory  # noqa: E402
import retrieval  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

# Mirrors test_degenerate_skip / test_budget_guard: latch retrieval
# unavailable so a stray import order cannot reach for an embedder.
retrieval._available = False
retrieval._embedder = None
retrieval._chroma_collection = None

memory.ensure_storage_layout()

client = TestClient(main.app, client=("127.0.0.1", 12347), raise_server_exceptions=False)

USER_TURN = "tell me about the greenhouse on the north side"


def assert_eq(actual, expected, label):
    if actual != expected:
        print(f"FAIL {label}: expected {expected!r}, got {actual!r}")
        sys.exit(1)
    print(f"  ok   {label}")


def assert_true(cond, label):
    if not cond:
        print(f"FAIL {label}")
        sys.exit(1)
    print(f"  ok   {label}")


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

_calls: list = []


def _spy(real):
    """Record every select_for_injection call, then delegate to the real one
    so the request completes normally and this stays an observation rather
    than a substitute implementation."""

    def wrapper(facts_list, *args, **kwargs):
        _calls.append({"args": args, "kwargs": kwargs, "n_in": len(facts_list)})
        return real(facts_list, *args, **kwargs)

    return wrapper


class _StubResponse:
    status_code = 200
    text = ""

    def json(self):
        return {
            "id": "stub",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "an ordinary reply"},
                    "finish_reason": "stop",
                }
            ],
        }


class _StubVLLM:
    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None, **kwargs):
        return _StubResponse()

    async def aclose(self):
        pass


def _seed_facts(conv_id, n=40):
    items = [
        {"text": f"[MISC] synthetic fact number {i}", "added_turn": i, "last_used": i}
        for i in range(n)
    ]
    facts.save_facts(conv_id, items)
    return items


def _post(conv_id, user_text):
    _calls.clear()
    with patch.object(main.httpx, "AsyncClient", _StubVLLM), \
         patch.object(facts, "select_for_injection", _spy(facts.select_for_injection)), \
         patch.object(main, "_fire_and_forget", lambda coro, label=None: coro.close()):
        return client.post(
            "/v1/chat/completions",
            json={
                "model": "stub-model",
                "messages": [{"role": "user", "content": user_text}],
                "stream": False,
            },
            headers={"X-Conversation-Id": conv_id},
        )


# ---------------------------------------------------------------------------
# [1] The request hot path
# ---------------------------------------------------------------------------


def test_hot_path_passes_the_current_user_turn_as_query_text():
    print("\n[test] hot path passes query_text = the current user turn")
    conv = "wiring-hot-path"
    _seed_facts(conv)
    r = _post(conv, USER_TURN)
    assert_eq(r.status_code, 200, "request succeeded")

    # Vacuity guard first: if the endpoint stopped injecting facts entirely,
    # every assertion below would pass by never running.
    assert_true(_calls, "select_for_injection was actually called")

    call = _calls[0]
    assert_true(
        "query_text" in call["kwargs"],
        "query_text passed as a keyword (positional would be max_tokens)",
    )
    assert_eq(call["kwargs"]["query_text"], USER_TURN, "query_text is the user's turn")
    assert_true(
        call["kwargs"]["query_text"] is not None,
        "query_text is not None — None is the unranked LRU path",
    )


def test_hot_path_ranking_is_not_silently_disabled_by_an_empty_query():
    print("\n[test] an empty/whitespace query_text would disable ranking too")
    # select_for_injection branches on `not query_text`, so "" and None are
    # the same unranked path. A refactor that passed "" instead of the turn
    # text would satisfy a naive "was it passed?" check and still be the
    # regression. Assert on truthiness, not presence.
    conv = "wiring-nonempty"
    _seed_facts(conv)
    _post(conv, USER_TURN)
    assert_true(_calls, "select_for_injection was actually called")
    assert_true(
        bool(_calls[0]["kwargs"].get("query_text")),
        "query_text is truthy, so the ranked branch is the one taken",
    )


# ---------------------------------------------------------------------------
# [2] The twin: _async_tail's fallback default
# ---------------------------------------------------------------------------


def test_async_tail_fallback_also_ranks():
    print("\n[test] _async_tail's own injected_facts default also passes query_text")
    # The request path hands _async_tail an already-computed injected_facts
    # list. Its `injected_facts is None` fallback recomputes, and must rank
    # too — otherwise a caller that omits the argument silently pushes an
    # unranked prefix into the extraction prompt, which is a request to vLLM
    # and therefore has a window. Identical twin of the hot-path call site
    # above; this branch has been bitten sixteen times by exactly that.
    conv = "wiring-tail"
    seeded = _seed_facts(conv)
    _calls.clear()

    async def _stub_extract(*a, **k):
        return []

    async def _run():
        with patch.object(
            facts, "select_for_injection", _spy(facts.select_for_injection)
        ), patch.object(
            facts, "extract_facts_from_exchange", _stub_extract
        ):
            await main._async_tail(
                conv,
                seeded,
                USER_TURN,
                "an ordinary reply",
                1,
                [{"role": "user", "content": USER_TURN}],
                # injected_facts deliberately omitted — this is the fallback
                # under test, not the request path's precomputed list.
            )

    asyncio.run(_run())

    assert_true(_calls, "the fallback actually called select_for_injection")
    assert_eq(
        _calls[0]["kwargs"].get("query_text"),
        USER_TURN,
        "fallback ranks against the same user turn",
    )


if __name__ == "__main__":
    try:
        for t in (
            test_hot_path_passes_the_current_user_turn_as_query_text,
            test_hot_path_ranking_is_not_silently_disabled_by_an_empty_query,
            test_async_tail_fallback_also_ranks,
        ):
            t()
        print("\nAll facts-wiring tests passed.")
    finally:
        shutil.rmtree(_TMP_ROOT, ignore_errors=True)
