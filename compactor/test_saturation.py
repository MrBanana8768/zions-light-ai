"""
SATURATION — a conversation driven to hundreds of turns, locally, no docker.

    python test_saturation.py                    # default 200 exchanges
    COMPACTOR_SATURATION_TURNS=600 python test_saturation.py

## Why this exists, and why it is not test_soak_conversation.py

`test_soak_conversation.py` is the adversarial soak and it is better than this
file in one way: it charges REAL tokens against a vLLM-shaped fixture, so it
can catch a budget error this cannot. It also needs docker on localhost:18000,
so on a developer machine it does not run, and until 2026-09-02 it reported
exit 0 while not running — which is how "all suites pass" came to exclude it.

This file trades that fidelity for the ability to always run. Everything is
stubbed; nothing leaves the process. What it buys is SCALE: every other Tier-1
suite drives one to five exchanges, and every production failure this project
has had appeared only after a conversation got long:

  - 2026-08-28  the counter under-read, the summarizer built over-budget
                batches, compaction 400'd, the guard shed 80 turns per
                request. No component was broken; the composition was.
  - 2026-09-01  63 exchanges never reached memory. Each individual skip was
                correct-looking; the AGGREGATE was the defect.
  - 2026-09-02  the position anchor stalls when the tail repeats, so the
                rollup gate never reopens. Invisible at five turns.

So the assertions here are about accumulation, not logic: counters that must
reconcile over hundreds of turns, quantities that must not grow without
bound, and work that must keep pace rather than fall behind.

## What this does NOT cover, stated so a green run is not over-read

  * Real token accounting. The stub does not tokenize; only the docker soak
    charges what vLLM would. A budget regression can pass here.
  * The summarizer's position arithmetic under a CAPPED window (backlog R23,
    R12). Those are open, reproduced defects; asserting the correct behaviour
    here would make this suite red for a reason that is already tracked, and
    asserting the current behaviour would pin a bug. The cap case is
    deliberately absent until those land — see the note at section [4].
  * Anything about the real model. Replies are generated fixtures.

Only synthetic content appears below; this repo is public.
"""

import contextlib
import json
import logging
import os
import sys
import tempfile
from unittest.mock import patch

os.environ.setdefault("MODEL_REPO", "test-model")
os.environ.setdefault("VLLM_URL", "http://stub:8000")
os.environ["COMPACTOR_RAG_ENABLED"] = "false"

_TMP_ROOT = tempfile.mkdtemp(prefix="compactor-test-saturation-")
os.environ["COMPACTOR_STORAGE_ROOT"] = _TMP_ROOT

import main  # noqa: E402
import memory  # noqa: E402
import retrieval  # noqa: E402
import tailhealth  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

retrieval._available = False
retrieval._embedder = None
retrieval._chroma_collection = None
memory.ensure_storage_layout()

client = TestClient(main.app, client=("127.0.0.1", 12351),
                    raise_server_exceptions=False)

TURNS = int(os.environ.get("COMPACTOR_SATURATION_TURNS", "200") or 200)

FAILED: list[str] = []


def check(cond, label):
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}")
        FAILED.append(label)


def check_eq(actual, expected, label):
    check(actual == expected, f"{label} (expected {expected!r}, got {actual!r})"
          if actual != expected else label)


# ---------------------------------------------------------------------------
# Stub backend. Replies are ordinary prose well over the memory floor, with a
# turn number woven in so a mislabelled or misattributed reply is visible.
# ---------------------------------------------------------------------------

def _reply(n: int) -> str:
    return (
        f"Reply number {n}. " + " ".join(
            f"This is sentence {i} of turn {n}, written plainly so the "
            f"degeneracy rules have nothing to catch." for i in range(1, 6)
        )
    )


class _Resp:
    status_code = 200

    def __init__(self, text):
        self._text = text

    def json(self):
        return {
            "id": "stub", "object": "chat.completion", "model": "stub-model",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": self._text}}],
        }

    @property
    def text(self):
        return json.dumps(self.json())

    async def aread(self):
        return b""

    def raise_for_status(self):
        # Real httpx responses have this and some callers use it. A stub that
        # omits it raises AttributeError deep inside the code under test,
        # which surfaces as 110 tracebacks at 200 turns and none at 25 —
        # a stub gap that only scale reveals.
        return None


class _StubVLLM:
    text = ""

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None, **kw):
        # /tokenize is asked for a count; the chat endpoint for a reply.
        if url.endswith("/tokenize"):
            body = json or {}
            text = str(body.get("prompt") or body.get("messages") or "")
            return _Resp("")  # count supplied below via .json override
        return _Resp(_StubVLLM.text)

    async def aclose(self):
        pass


class _TokenizingStub(_StubVLLM):
    """Answers /tokenize with a real-ish count so the budget path exercises
    its arithmetic rather than falling to the local estimator every turn.

    char/4 is not what vLLM charges — that is the whole point of the docker
    soak — but a stub that REFUSES would put every request on the fallback
    path and this suite would silently stop covering the guard at all.
    """

    async def post(self, url, json=None, **kw):
        if url.endswith("/tokenize"):
            body = json or {}
            if "messages" in body:
                n = sum(len(str(m.get("content", ""))) for m in body["messages"])
            else:
                n = len(str(body.get("prompt", "")))
            r = _Resp("")
            r.json = lambda: {"count": max(1, n // 4)}   # type: ignore[method-assign]
            return r
        return _Resp(_StubVLLM.text)


def _stub_tokenize_post(url, json=None, timeout=None, **kw):
    """count_tokens_exact / count_text_tokens_exact call httpx.post DIRECTLY
    and synchronously, not through the AsyncClient the chat path uses.
    Patching only the client leaves every budget decision on the local
    estimator, and this suite stops covering the guard at all — which is
    exactly what the first run of this file did, visibly, via "token scale
    unavailable (/tokenize refused)".

    char/4 is not what vLLM charges; the docker soak owns that fidelity. What
    this buys is that the guard's arithmetic runs at all.
    """
    body = json or {}
    if "messages" in body:
        n = sum(len(str(m.get("content", ""))) for m in body["messages"])
    else:
        n = len(str(body.get("prompt", "")))
    r = _Resp("")
    r.json = lambda: {"count": max(1, n // 4)}
    return r


_tail_calls: list[dict] = []


def _spy_tail(conv_id, touched_facts, last_user_text, assistant_text,
              turn_index, original_messages, *, injected_facts=None):
    _tail_calls.append({"conv_id": conv_id, "assistant_text": assistant_text,
                        "turn_index": turn_index})
    return {"spy": True}


def _spy_fire(obj, label=None):
    return None


class _Collector(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)


@contextlib.contextmanager
def capture():
    lg = logging.getLogger("compactor")
    h = _Collector()
    lg.addHandler(h)
    try:
        yield h
    finally:
        lg.removeHandler(h)


def _drive(conv_id: str, turns: int, history: list[dict]) -> list[logging.LogRecord]:
    """Run `turns` exchanges, growing `history` the way a client does."""
    recs: list[logging.LogRecord] = []
    with patch.object(main.httpx, "AsyncClient", _TokenizingStub), \
         patch.object(main.httpx, "post", _stub_tokenize_post), \
         patch.object(main, "_async_tail", _spy_tail), \
         patch.object(main, "_fire_and_forget", _spy_fire), \
         capture() as cap:
        for n in range(1, turns + 1):
            user = f"Question number {n}, asked plainly."
            history.append({"role": "user", "content": user})
            _StubVLLM.text = _reply(n)
            r = client.post(
                "/v1/chat/completions",
                json={"model": "stub-model", "messages": list(history),
                      "stream": False},
                headers={"X-Conversation-Id": conv_id},
            )
            if r.status_code != 200:
                FAILED.append(f"turn {n} returned HTTP {r.status_code}")
                break
            history.append({"role": "assistant", "content": _reply(n)})
        recs = list(cap.records)
    return recs


# ---------------------------------------------------------------------------

print(f"\n[1] {TURNS} exchanges through the real endpoint, full history each turn")
tailhealth._reset_for_tests()
_tail_calls.clear()
history: list[dict] = [{"role": "system", "content": "persona"}]
records = _drive("sat-main", TURNS, history)

check_eq(len(_tail_calls), TURNS, "every exchange reached the memory tail")
check(all(c["conv_id"] == "sat-main" for c in _tail_calls),
      "every tail carries the same conv_id — no fork across the run")

print("\n[2] the memory-tail ledger reconciles")
# The counter is what /health/full reports. Over hundreds of turns it must
# account for every exchange exactly once: 63 lost exchanges were invisible
# precisely because nothing added them up.
snap = tailhealth.snapshot()
check_eq(snap["stored"] + snap["skipped"], TURNS,
         "stored + skipped equals the exchanges driven")
check_eq(sum(snap["outcomes"].values()), TURNS,
         "the per-outcome breakdown sums to the same number")
check_eq(snap["skipped"], 0, "clean finishes are never skipped")
check(snap["kept_chars"] > 0 and snap["raw_chars"] >= snap["kept_chars"],
      "character totals are consistent")

print("\n[3] turn_index advances monotonically and never repeats")
idxs = [c["turn_index"] for c in _tail_calls]
check(all(b > a for a, b in zip(idxs, idxs[1:])),
      "turn_index strictly increases across the whole run")
check_eq(len(set(idxs)), len(idxs), "no two exchanges share a turn_index")

print("\n[4] the summary hierarchy keeps pace with the conversation")
# Driven DIRECTLY, not through the endpoint. Section [1] spies out
# _async_tail, so nothing there ever reaches maybe_rollup — asserting on
# summarizer state after [1] read 0 and proved only that the spy worked.
# This drives the real position tracking over the same number of turns with
# the LLM call faked, which is the part that has to survive scale.
#
# UNCAPPED window only. The capped case is backlog R23/R12 — reproduced,
# open, and deliberately not asserted: a green suite must not imply coverage
# of a bug we already know about.
import asyncio  # noqa: E402
import summarizer  # noqa: E402


async def _fake_pieces(*a, **k):
    return "a summary of the span"


_rollup_conv = "sat-rollup"
_msgs: list[dict] = [{"role": "system", "content": "persona"}]
with patch.object(summarizer, "_summarize_pieces", _fake_pieces):
    for _n in range(1, TURNS + 1):
        _msgs.append({"role": "user", "content": f"Question number {_n}."})
        _msgs.append({"role": "assistant", "content": _reply(_n)})
        asyncio.run(summarizer.maybe_rollup(
            _rollup_conv, list(_msgs), "http://stub:8000", "stub-model"))

_state = summarizer.load_state(_rollup_conv)
check_eq(_state.get("turns_seen", 0), TURNS * 2,
         "turns_seen equals the non-system turns sent")
# The WATERMARK is what "keeping pace" means, not the L1 count. Past ~400
# turns L1 chunks are FOLDED into an L2 chapter and removed from state["l1"],
# so a count assertion reads 0 at 200 exchanges and 2 at 25 — the opposite of
# the truth. That is the shape of bug this suite exists to surface, and the
# first version of this very assertion had it.
_watermark = _state.get("last_summarized_turn", 0)
check(_watermark >= TURNS * 2 - summarizer.L1_CHUNK_SIZE,
      f"the watermark kept pace ({_watermark} of {TURNS * 2} turns, at most "
      f"one unrolled chunk of {summarizer.L1_CHUNK_SIZE} behind)")

# Coverage across every tier: no turn below the watermark may fall between
# two spans. L2/L3 fold L1 away, so the union has to be checked across all of
# them, not within one.
_spans = [(c["first_turn"], c["last_turn"])
          for tier in ("l1", "l2") for c in (_state.get(tier) or [])]
_l3 = _state.get("l3")
if isinstance(_l3, dict) and "first_turn" in _l3:
    _spans.append((_l3["first_turn"], _l3["last_turn"]))
_spans.sort()
check(bool(_spans), f"the hierarchy produced spans ({len(_spans)} across tiers)")
check(all(a <= b for a, b in _spans), "every span covers a forward range")
_covered: set[int] = set()
for _a, _b in _spans:
    _covered.update(range(_a, _b + 1))
_gaps = [t for t in range(1, _watermark + 1) if t not in _covered]
check(not _gaps,
      f"no turn below the watermark is uncovered by any tier"
      + (f" — {len(_gaps)} gaps, first at {_gaps[:5]}" if _gaps else ""))

print("\n[5] nothing grew without bound")
# The accumulator's buffer and the fact store are the two places this project
# has had unbounded growth proposed and rejected; assert the shape rather
# than a magic number.
import facts  # noqa: E402

stored = facts.load_facts("sat-main")
check(len(stored) <= facts._MAX_FACTS_TOKENS,
      f"the fact store stayed bounded ({len(stored)} entries)")
tracebacks = [r for r in records if r.exc_info or "Traceback" in r.getMessage()]
check_eq(len(tracebacks), 0, "no exception reached the log across the run")

print("\n[6] the request path stayed quiet")
# A saturation run that emits a warning per turn is telling you something even
# when every assertion passes. Anything at this rate is a finding.
noisy: dict[str, int] = {}
for r in records:
    key = r.getMessage().split(":")[0][:60]
    noisy[key] = noisy.get(key, 0) + 1
# transformers is deliberately absent here (the Dockerfile guard fails the
# build if it is missing from the IMAGE), so the tokenizer warning is an
# artifact of the harness rather than a signal about the code. Listed
# explicitly, not filtered by a wildcard, so a NEW per-turn warning still
# fails this check.
_ENV_ARTIFACTS = ("could not load tokenizer",)
per_turn = {k: v for k, v in noisy.items()
            if v >= TURNS and not any(a in k for a in _ENV_ARTIFACTS)}
for k, v in sorted(per_turn.items(), key=lambda kv: -kv[1]):
    print(f"       once per turn or more: {v:5d}x  {k}")
check(not per_turn, "no warning fires on every single turn")

print()
if FAILED:
    for f in FAILED:
        print("FAIL " + f)
    sys.exit(1)
print(f"All saturation tests passed ({TURNS} exchanges).")
