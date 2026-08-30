"""
Residual 1 (v3.1.3) — the degeneracy skip must actually keep a degenerate
reply out of memory, not just claim to.

`main.reply_is_degenerate` exists so a repetition-loop reply cannot be
extracted as facts, indexed for retrieval, or rolled into a hierarchical
summary and injected back later as though it were worth remembering. Both
call sites (streaming, ~line 3616; non-streaming, ~line 3753) gate the async
memory tail on `not _degen` — verified in section [1] below, end to end,
through the real endpoint, for BOTH paths (this branch has been bitten eight
times by a fix landing on one call site and not its twin).

That gate is provably sufficient for jobs 1-2 of `_async_tail` (episodic
indexing, fact extraction): both only ever see THIS exchange's own
last_user_text/assistant_text, so skipping the call for a degenerate turn
keeps it out of both forever — no later call revisits that turn.

It is NOT sufficient for job 3 (hierarchical summary rollup), and section [2]
proves why empirically: `summarizer.maybe_rollup` slices its input out of the
conversation's message history by TURN POSITION on every subsequent call.
That history is the client's, not the compactor's — the user already read
the degenerate reply, so the client resends it as part of history on the
very next request. Skipping _async_tail for turn N does nothing to turn N's
text once it shows up inside `original_messages` on turn N+1's tail call;
`maybe_rollup` has never heard of "degenerate" and will fold that turn's raw
text into an L1 chunk exactly as it would any other, once the turn falls
inside a chunk's range. That chunk is then held in `summaries/<conv>.json`
and injected back into future prompts as memory — precisely the outcome the
detector exists to prevent, achieved one turn later than the skip is looking.

The fix lives at three call sites (this file covers the live one) (`_redact_degenerate_turns`,
main.py): before `original_messages` is handed to `summarizer.maybe_rollup`,
every assistant turn that is itself degenerate is replaced with a neutral
placeholder. Section [2] would fail without it — this file was run against
the pre-fix code and the raw repeated text reached the rollup input verbatim
(see the commit this test ships with for that evidence).

Only synthetic conversation content appears below (project rule: this repo
is public).

    python test_degenerate_skip.py
"""

import asyncio
import json
import logging
import os
import sys
import tempfile
from unittest.mock import patch

os.environ.setdefault("MODEL_REPO", "test-model")
os.environ.setdefault("VLLM_URL", "http://stub:8000")
os.environ["COMPACTOR_RAG_ENABLED"] = "false"  # no ChromaDB/fastembed here

_TMP_ROOT = tempfile.mkdtemp(prefix="compactor-test-degenerate-skip-")
os.environ["COMPACTOR_STORAGE_ROOT"] = _TMP_ROOT

import facts  # noqa: E402
import main  # noqa: E402
import memory  # noqa: E402
import retrieval  # noqa: E402
import summarizer  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

# Belt and braces alongside COMPACTOR_RAG_ENABLED=false (mirrors
# test_budget_guard.py / test_retrieval._force_unavailable): latch retrieval
# unavailable outright so a stray import order can't reach for an embedder.
retrieval._available = False
retrieval._embedder = None
retrieval._chroma_collection = None

memory.ensure_storage_layout()

client = TestClient(main.app, client=("127.0.0.1", 12346), raise_server_exceptions=False)

RULE = "━"
DEGENERATE_TEXT = RULE * 400  # same shape as the 2026-08-29 production incident


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
# [1] Both call sites actually skip the async tail for a degenerate reply —
#     and both actually FIRE it for an ordinary one, so this harness would
#     catch a regression rather than passing vacuously.
# ---------------------------------------------------------------------------

_tail_labels: list = []


def _spy_fire_and_forget(coro, label=None):
    _tail_labels.append(label)
    try:
        coro.close()
    except Exception:
        pass


class _StubResponse:
    status_code = 200
    text = ""

    def __init__(self, content):
        self._content = content

    def json(self):
        return {
            "id": "stub",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": self._content},
                    "finish_reason": "stop",
                }
            ],
        }


class _StubVLLM:
    """Non-streaming stub. `reply` is class-level so each test sets it just
    before posting (mirrors test_budget_guard._StubVLLM)."""

    reply = "an ordinary reply"

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None, **kwargs):
        return _StubResponse(_StubVLLM.reply)

    async def aclose(self):
        pass


def _sse(obj) -> bytes:
    return f"data: {json.dumps(obj)}\n\n".encode()


def _stream_chunks(text: str) -> list:
    return [
        _sse({"choices": [{"delta": {"content": text}}]}),
        _sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        b"data: [DONE]\n\n",
    ]


class _StubStreamResp:
    status_code = 200

    def __init__(self, chunks):
        self._chunks = chunks

    async def aread(self):
        return b""

    async def aiter_raw(self):
        for c in self._chunks:
            yield c


class _StubStreamCM:
    def __init__(self, chunks):
        self._resp = _StubStreamResp(chunks)

    async def __aenter__(self):
        return self._resp

    async def __aexit__(self, *exc):
        return False


class _StubStreamVLLM:
    """Streaming stub. `reply` is class-level, same convention as above."""

    reply = "an ordinary reply"

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def stream(self, method, url, json=None, **kwargs):
        return _StubStreamCM(_stream_chunks(_StubStreamVLLM.reply))

    async def aclose(self):
        pass


def _post_nonstream(conv_id, reply_text):
    _tail_labels.clear()
    _StubVLLM.reply = reply_text
    with patch.object(main.httpx, "AsyncClient", _StubVLLM), \
         patch.object(main, "_fire_and_forget", _spy_fire_and_forget):
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "stub-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
            headers={"X-Conversation-Id": conv_id},
        )
    return r


def _post_stream(conv_id, reply_text):
    _tail_labels.clear()
    _StubStreamVLLM.reply = reply_text
    with patch.object(main.httpx, "AsyncClient", _StubStreamVLLM), \
         patch.object(main, "_fire_and_forget", _spy_fire_and_forget):
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "stub-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
            headers={"X-Conversation-Id": conv_id},
        )
        # TestClient's post() (unlike client.stream()) fully drains the ASGI
        # response before returning, so event_stream()'s `finally` — where
        # the tail is fired — has already run (mirrors test_resilience.py,
        # which asserts on r.text after an identical plain post()).
    return r


print("[1] both call sites gate the memory tail on reply_is_degenerate")

r = _post_nonstream("degen-skip-nonstream", DEGENERATE_TEXT)
assert_eq(r.status_code, 200, "non-stream: request completed")
assert_eq(len(_tail_labels), 0,
           "non-stream: the memory tail was NOT fired for a degenerate reply")

r = _post_nonstream("degen-skip-nonstream-control", "a perfectly normal reply")
assert_eq(r.status_code, 200, "non-stream control: request completed")
assert_eq(len(_tail_labels), 1,
           "non-stream control: the memory tail IS fired for an ordinary reply "
           "(so this harness would catch a regression, not pass vacuously)")

r = _post_stream("degen-skip-stream", DEGENERATE_TEXT)
assert_eq(r.status_code, 200, "stream: request completed")
assert_eq(len(_tail_labels), 0,
           "stream: the memory tail was NOT fired for a degenerate reply")

r = _post_stream("degen-skip-stream-control", "a perfectly normal reply")
assert_eq(r.status_code, 200, "stream control: request completed")
assert_eq(len(_tail_labels), 1,
           "stream control: the memory tail IS fired for an ordinary reply")


# ---------------------------------------------------------------------------
# [2] The actual residual: a PAST degenerate reply sitting in conversation
#     history must not reach the summarizer's rollup input, even though the
#     skip above never re-examines that turn.
# ---------------------------------------------------------------------------

print()
print("[2] a past degenerate reply in history is kept out of rollup input")


def _run_tail_capture_rollup_input(conv_id, original_messages, assistant_text):
    """Run the real _async_tail with facts extraction and dedup stubbed (no
    vLLM calls), summarizer forced on, and maybe_rollup replaced by a spy
    that records exactly the `messages` argument it was handed."""
    captured = {}

    async def spy_maybe_rollup(cid, messages, vllm_url, model):
        captured["messages"] = messages
        return {"l1": [], "l2": [], "l3": None, "last_summarized_turn": 0}

    async def spy_extract(*_a, **_k):
        return []

    with patch.object(summarizer, "enabled", lambda: True), \
         patch.object(summarizer, "maybe_rollup", spy_maybe_rollup), \
         patch.object(facts, "extract_facts_from_exchange", spy_extract):
        asyncio.run(main._async_tail(
            conv_id, [], "what's next?", assistant_text, 3, original_messages,
        ))
    return captured.get("messages")


original_messages = [
    {"role": "user", "content": "Tell me something about the ocean."},
    {"role": "assistant", "content": DEGENERATE_TEXT},
    {"role": "user", "content": "what's next?"},
]
seen = _run_tail_capture_rollup_input(
    "conv-degen-history", original_messages, "Here is something ordinary."
)
assert_true(seen is not None, "fixture: summarizer.maybe_rollup was invoked")
texts = [main._message_text(m) for m in seen]
assert_true(
    not any(DEGENERATE_TEXT in t for t in texts),
    "the degenerate reply's raw repeated text did not reach the rollup input",
)
assert_true(
    main._DEGENERATE_HISTORY_PLACEHOLDER in texts,
    "a neutral placeholder took its place, rather than the turn vanishing "
    "outright (that would just be a different unrecorded loss)",
)
assert_eq(len(texts), 4, "and no turn was dropped from the array outright")

print()
print("[3] an ordinary past reply is passed through to rollup unchanged")

original_messages_2 = [
    {"role": "user", "content": "Tell me something about the ocean."},
    {"role": "assistant", "content": "It covers most of the planet's surface."},
    {"role": "user", "content": "what's next?"},
]
seen2 = _run_tail_capture_rollup_input(
    "conv-degen-history-2", original_messages_2, "Here is more detail."
)
texts2 = [main._message_text(m) for m in seen2]
assert_true(
    "It covers most of the planet's surface." in texts2,
    "an ordinary historical reply is not touched by the redaction",
)

print()
print("[4] the newly-completed (non-degenerate) turn is untouched either way")
assert_true(
    "Here is something ordinary." in texts,
    "this turn's own assistant_text — guaranteed non-degenerate by the call "
    "sites — passes through unredacted",
)

print()
print("All degenerate-skip tests passed.")
