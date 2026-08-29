"""
End-to-end soak: a conversation that GROWS, against a real token-counting server.

## Why this exists

Every test in this repo before it checks a component. Not one drives a
conversation from turn 1 to turn N and asks whether the thing still works — and
that is exactly the shape of every failure this project has had:

  - 2026-08-28: the counter under-read, so the summarizer built over-budget
    batches, so compaction 400'd, so the guard shed 80 turns per request. No
    single component was broken. The COMPOSITION was, and only after a
    conversation got long enough.
  - 2026-08-29: /tokenize refused an assistant-final list. Same cascade, new
    trigger, found in production rather than in CI.

A component test cannot see either. Both need a conversation with a history
long enough to compact, replies expensive enough to matter, and a server that
charges real tokens and refuses what does not fit.

## What makes it adversarial rather than merely long

The fixture generates replies containing the four things that actually broke
production, not lorem alone:

  - box-drawing rules (U+2501) — where local and server tokenizers diverge most
  - emoji — the same, by a different route
  - markdown headings and code fences — what fact extraction stored as memory
  - fabricated numeric status lines — the loop where the model reads its own
    invented figures back as fact

Prose is where tokenizers agree. A soak written on prose alone would pass while
production burned, which is the mistake the contract harness documents.

## Running it

    docker compose -f docker-compose.tokenizer-contract.yml up -d tokenize-fixture
    VLLM_URL=http://localhost:18000 MODEL_REPO=fixture-model \\
      python test_soak_conversation.py

With no fixture reachable it SKIPS and exits 0, so the CPU-only suite is
unaffected.

## What it does NOT prove

The fixture's tokenizer is not Cydonia's, so absolute token numbers mean
nothing for production. What it proves is that the SYSTEM holds together over a
growing conversation against a server that charges real tokens — the property
no other test in this repo asserts.
"""

import os
import sys
import tempfile

# --- Environment must be set before main is imported ------------------------
os.environ.setdefault("VLLM_URL", "http://localhost:18000")
os.environ.setdefault("MODEL_REPO", "fixture-model")
# A small window so compaction fires within a soak-sized run instead of after
# hundreds of turns. The RATIOS are what this test asserts, not the absolutes.
# Forced, not setdefault: the production image BAKES MAX_MODEL_LEN=32768 as an
# ENV, so setdefault silently loses and the soak would run against a window
# four times too large — compaction would never fire and the run would assert
# nothing while reporting success. Override with SOAK_MAX_MODEL_LEN.
os.environ["MAX_MODEL_LEN"] = os.environ.get("SOAK_MAX_MODEL_LEN", "8192")
os.environ["COMPACTOR_GENERATION_RESERVE"] = os.environ.get(
    "SOAK_GENERATION_RESERVE", "2048")
# No network on the budgeting path.
#
# The first real soak run blocked for 231 SECONDS inside get_tokenizer, which
# tried to resolve MODEL_REPO against huggingface.co and hit a 429. The fixture
# connection dropped while it was stalled and the run failed at turn 10 for an
# entirely environmental reason.
#
# That is REMEDIATION F33 (the tokenizer load is retried per request and its
# failure is indistinguishable from not-yet-loaded) composing with F25 (nothing
# sets HF_HUB_OFFLINE even when every weight is cached). Latent in production
# only because the model IS cached there. Setting these keeps the soak
# measuring the compactor rather than huggingface.co's rate limiter.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_STORE = tempfile.mkdtemp(prefix="soak-store-")
os.environ["COMPACTOR_STORAGE_ROOT"] = _STORE

import asyncio  # noqa: E402
import logging  # noqa: E402
import time  # noqa: E402
from unittest.mock import patch  # noqa: E402

import httpx  # noqa: E402

FIXTURE_URL = os.environ.get("FIXTURE_URL", os.environ["VLLM_URL"]).rstrip("/")
TURNS = int(os.environ.get("SOAK_TURNS", "40"))
REPLY_CHARS = int(os.environ.get("SOAK_REPLY_CHARS", "1400"))


def _skip(reason: str) -> None:
    print("SKIPPED: test_soak_conversation.py")
    print(f"  {reason}")
    print("  Start the fixture with:")
    print("    docker compose -f docker-compose.tokenizer-contract.yml "
          "up -d tokenize-fixture")
    sys.exit(0)


try:
    _mode = httpx.get(f"{FIXTURE_URL}/_fixture/mode", timeout=3.0)
    _mode.raise_for_status()
    _mode = _mode.json()
except Exception as e:
    _skip(f"{FIXTURE_URL} unreachable ({type(e).__name__}: {e})")

# A STALE FIXTURE IS NOT A SKIP — same doctrine as test_tokenizer_contract.
# Silently soaking against a fixture that cannot generate adversarial replies
# would report coverage this run does not have.
if "reply_chars" not in _mode:
    print("FAIL the fixture image is stale: /_fixture/mode has no 'reply_chars'.")
    print("     Rebuild it:  docker compose -f docker-compose.tokenizer-contract.yml "
          "build tokenize-fixture")
    sys.exit(1)

from fastapi.testclient import TestClient  # noqa: E402

import memory  # noqa: E402

memory.ensure_storage_layout()

import facts as facts_mod  # noqa: E402
import main  # noqa: E402
import summarizer  # noqa: E402

client = TestClient(main.app, client=("127.0.0.1", 12345),
                    raise_server_exceptions=False)

CONV = "soak_conversation"


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)

    def text(self) -> str:
        return "\n".join(r.getMessage() for r in self.records)


_pending: list = []


def _defer_tail(coro, label=None):
    """Collect the memory tail instead of firing it into the background.

    The tail is what writes facts, indexes episodic and rolls up summaries. In
    production it is fire-and-forget; in a soak it has to COMPLETE before the
    next turn, or the run measures a system whose memory never caught up and
    every assertion about accumulation is meaningless.
    """
    _pending.append(coro)


def _drain_tails() -> None:
    while _pending:
        coro = _pending.pop(0)
        try:
            asyncio.run(coro)
        except Exception as e:  # a tail failure is a finding, not a crash
            print(f"  NOTE tail raised: {type(e).__name__}: {e}")


def _reset_fixture() -> None:
    """Put the fixture back in an honest mode before soaking.

    The contract tests deliberately leave it lying — a 0.5 count factor, a 400
    status, an artificial delay. Soaking against that measures the fixture's
    sabotage rather than the compactor, and would either fail for the wrong
    reason or, worse, pass while asserting nothing.
    """
    # /_fixture/reset clears STATS, not the mode — the sabotage settings
    # survive it, so they have to be set back by name.
    httpx.post(f'{FIXTURE_URL}/_fixture/reset', timeout=5.0)
    httpx.post(
        f'{FIXTURE_URL}/_fixture/mode',
        json={'tokenize_mode': 'ok', 'factor': 1.0, 'status': 200,
              'delay': 0.0, 'assistant_final_400': False},
        timeout=5.0,
    )
    m = httpx.get(f'{FIXTURE_URL}/_fixture/mode', timeout=5.0).json()
    if m.get('tokenize_mode') != 'ok' or float(m.get('factor', 1)) != 1.0:
        fail('the fixture would not return to an honest mode', repr(m))
    if m.get('assistant_final_400'):
        fail('the fixture is still refusing assistant-final lists',
             'that is the D1 sabotage mode; a soak cannot run against it')


def _fixture_stats() -> dict:
    try:
        return httpx.get(f"{FIXTURE_URL}/_fixture/stats", timeout=5.0).json()
    except Exception:
        return {}


def _set_reply(seq: int) -> None:
    httpx.post(f"{FIXTURE_URL}/_fixture/mode",
               json={"reply_chars": REPLY_CHARS, "reply_seq": seq}, timeout=5.0)


def _turn(n: int, history: list[dict]) -> tuple[int, str, list[dict], str, int]:
    """One real POST through the compactor to the fixture.

    Returns (status, log text, forwarded payload, reply text, LLM calls made
    ON THE REQUEST PATH) — the last one measured before the memory tail runs,
    because the tail does not block the user."""
    _set_reply(n)
    _calls_at_start = _fixture_stats().get("chat_completions", 0)
    cap = _Capture()
    lg = logging.getLogger("compactor")
    lg.addHandler(cap)
    forwarded: list[dict] = []

    _real_post = main.httpx.AsyncClient.post

    async def _spy(self, url, **kw):
        if url.endswith("/v1/chat/completions"):
            forwarded.append(kw.get("json") or {})
        return await _real_post(self, url, **kw)

    try:
        with patch.object(main, "_fire_and_forget", _defer_tail), \
             patch.object(main.httpx.AsyncClient, "post", _spy):
            r = client.post(
                "/v1/chat/completions",
                json={"model": "fixture-model", "messages": history,
                      "stream": False},
                headers={"X-Conversation-Id": CONV},
            )
    finally:
        lg.removeHandler(cap)
    # Counted HERE, before the tail is drained. The tail (fact extraction,
    # dedup, rollup) is fire-and-forget in production and does NOT block the
    # reply; this soak runs it synchronously only so memory accumulates
    # deterministically. Counting it against the request-path budget measures
    # the wrong thing — and the first version of this assertion did exactly
    # that, then blamed the code for it.
    request_calls = _fixture_stats().get("chat_completions", 0) - _calls_at_start
    _drain_tails()
    sent = forwarded[-1].get("messages", []) if forwarded else []
    reply = ""
    try:
        reply = (r.json()["choices"][0]["message"]["content"]) or ""
    except Exception:
        pass
    return r.status_code, cap.text(), sent, reply, request_calls


def fail(label: str, detail: str = "") -> None:
    print(f"FAIL {label}")
    if detail:
        print(f"     {detail}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# The soak
# ---------------------------------------------------------------------------

print(f"[soak] {TURNS} turns, ~{REPLY_CHARS}-char adversarial replies, "
      f"window {os.environ['MAX_MODEL_LEN']}, store {_STORE}")
print(f"[soak] fixture: {FIXTURE_URL}")
_reset_fixture()

history: list[dict] = []
rows: list[dict] = []
calls_seen: list[tuple] = []
compaction_fired = False

for n in range(1, TURNS + 1):
    history.append({"role": "user",
                    "content": f"Turn {n}. Tell me about item {n} in detail."})
    _t0 = time.monotonic()
    status, log, sent, reply, _calls = _turn(n, history)
    _elapsed = time.monotonic() - _t0

    # ---- invariants that must hold on EVERY turn --------------------------
    # Each one is a failure this project has actually shipped.

    if status != 200:
        fail(f"turn {n}: backend rejected the request (HTTP {status})",
             "A conversation that grows must never become unanswerable. "
             "This is the 2026-08-29 shape.")

    for needle, why in (
        ("/tokenize degraded",
         "the counter fell back to an estimator that reads up to 51% low "
         "(D1: an assistant-final list refused by the chat template)"),
        ("compaction failed",
         "compaction fell through and forwarded the original messages "
         "(the head of the 2026-08-28 cascade)"),
        ("hard budget FAILED to fit",
         "the guard could not fit the payload and forwarded anyway"),
        ("REQUEST REJECTED",
         "the turn produced no reply, no facts and no episodic write"),
        (", margin ",
         "the calibration margin latched and is now narrowing the window "
         "for every conversation in the process (D4)"),
    ):
        if needle in log:
            fail(f"turn {n}: {needle!r} appeared", why)

    # THE ASSERTION THAT WAS MISSING, and its absence is why 2026-08-29
    # happened. Compaction runs on the REQUEST PATH. A conversation with a
    # summarization backlog produced 33 LLM calls on one request — eight
    # minutes with a dead composer, and the user got no reply at all. Nothing
    # here measured how much work a single turn did, so a green soak said
    # everything was fine.
    #
    # The budget is the cap plus one: the cap bounds summarization calls, and
    # the user's own reply is the +1. Fact extraction and dedup run on the
    # background tail, which the soak drains separately after the turn.
    _budget = main.MAX_SUMMARY_CALLS_PER_REQUEST + 1
    if _calls > _budget:
        fail(f"turn {n}: one request made {_calls} LLM calls (budget {_budget})",
             f"compaction is unbounded on the request path. At ~1024 output "
             f"tokens per call on a 24B model this is minutes of latency for a "
             f"user who is watching a blank composer. See "
             f"MAX_SUMMARY_CALLS_PER_REQUEST.")
    calls_seen.append((n, _calls, round(_elapsed, 1)))

    if "hard budget enforced" in log:
        compaction_fired = True  # shedding implies we got past the target

    if "summarize:" in log:
        compaction_fired = True

    nonsys = [m for m in sent if m.get("role") != "system"]
    rows.append({"turn": n, "sent_msgs": len(sent), "sent_nonsys": len(nonsys),
                 "history": len(history),
                 "watermark": summarizer.load_state(CONV).get(
                     "last_summarized_turn", 0)})

    # The REAL reply, not a placeholder. A soak that appends "(reply)" grows a
    # 47-message conversation weighing 1,211 tokens, never reaches the
    # compaction trigger, and asserts nothing while looking busy — which is
    # precisely the class of test this file exists to replace.
    if not reply:
        fail(f"turn {n}: the backend returned no assistant text",
             "the soak cannot grow a conversation it never receives")
    history.append({"role": "assistant", "content": reply})

    if n % 10 == 0:
        st = summarizer.load_state(CONV)
        print(f"  turn {n:>3}: history={len(history):>3} forwarded={len(sent):>3} "
              f"facts={len(facts_mod.load_facts(CONV)):>3} "
              f"L1={len(st.get('l1') or [])}")

# ---------------------------------------------------------------------------
# End-of-run invariants
# ---------------------------------------------------------------------------

print()
print("[soak] end-of-run checks")
if calls_seen:
    _worst = max(calls_seen, key=lambda x: x[1])
    _slow = max(calls_seen, key=lambda x: x[2])
    print(f"  ok   LLM calls per request stayed bounded "
          f"(worst {_worst[1]} at turn {_worst[0]}, budget "
          f"{main.MAX_SUMMARY_CALLS_PER_REQUEST + 1}; slowest turn "
          f"{_slow[2]}s at turn {_slow[0]})")

if not compaction_fired:
    fail("the run never exceeded the compaction target",
         f"{TURNS} turns of ~{REPLY_CHARS} chars did not fill a "
         f"{os.environ['MAX_MODEL_LEN']}-token window. Raise SOAK_TURNS or "
         f"SOAK_REPLY_CHARS — a soak that never compacts proves nothing.")
print("  ok   the conversation actually got big enough to exercise compaction")

# Context delivery — and the right property is NOT "many raw turns arrived".
#
# A guard that sheds 36 of 40 messages is behaving CORRECTLY if compaction
# summarised them first: the older material is still represented, just densely.
# That is the whole design. The 2026-08-28 failure was not shedding — it was
# shedding turns that nothing had summarised, because compaction had died. The
# turns were simply gone, and the log said "enforced" either way.
#
# So the invariant is COVERAGE: every turn is either forwarded verbatim or sits
# below the summarizer's watermark. A turn that is neither has left the system.
substantive = [r for r in rows if r["history"] >= 6]
if not substantive:
    fail("the run never built a conversation worth measuring", "raise SOAK_TURNS")

# The property is that the SUMMARIZER KEEPS PACE — not that every message is
# covered at every instant.
#
# My first version asserted `watermark + forwarded >= history` and fired at
# turn 19 with history=37, watermark=20. That is not loss: L1_CHUNK_SIZE is 20
# message-units, so a lag of up to one unfilled chunk is the design working.
# Asserting otherwise would have made this test red on a healthy system, which
# is how a suite teaches people to ignore it.
#
# The incident looked different in a way this DOES catch: the watermark froze
# (compaction was failing) while the conversation grew, so the lag climbed
# without bound — production reached history=116 against a watermark that had
# stopped, and the guard shed everything above it. So: bound the lag, and
# require the watermark to actually move.
_LAG_BUDGET = summarizer.L1_CHUNK_SIZE * 2 + main.KEEP_RECENT_TURNS * 2

lagging = [r for r in substantive if (r["history"] - r["watermark"]) > _LAG_BUDGET]
if lagging:
    w = max(lagging, key=lambda r: r["history"] - r["watermark"])
    fail(f"turn {w['turn']}: the summarizer fell {w['history'] - w['watermark']} "
         f"message-units behind (budget {_LAG_BUDGET})",
         f"history={w['history']} watermark={w['watermark']} "
         f"forwarded={w['sent_nonsys']}. A watermark that stops advancing while "
         f"the conversation grows means compaction is failing, and everything "
         f"above it is being DROPPED by the guard rather than summarised. That "
         f"is 2026-08-28, and it logs as 'hard budget enforced' either way.")

if substantive[-1]["watermark"] <= 0:
    fail("the summarizer watermark never advanced",
         "the conversation grew past the compaction target and nothing was "
         "ever summarised — the hierarchy is dead, which is what A1 fixed")

worst = max(substantive, key=lambda r: r["history"] - r["watermark"])
print(f"  ok   the summarizer kept pace (worst lag {worst['history'] - worst['watermark']} "
      f"of {_LAG_BUDGET} at turn {worst['turn']}; final watermark "
      f"{substantive[-1]['watermark']})")

# The memory layers must actually contain something.
stored = facts_mod.load_facts(CONV)
if not stored:
    fail("no facts were stored across the whole run",
         "the memory tail ran but nothing accumulated")
print(f"  ok   facts accumulated ({len(stored)})")

# ...and must not contain what the model decorates with. This is the loop where
# the system reads its own scaffolding back as established truth.
bad = [f for f in stored if not facts_mod.is_storable_fact(f.get("text", ""))]
if bad:
    fail(f"{len(bad)} stored fact(s) are markup, not facts",
         "every reply in this run contained a code fence, a heading and a "
         "62-character box-drawing rule; none of them may be memory")
print("  ok   no markup reached the fact store")

state = summarizer.load_state(CONV)
l1 = state.get("l1") or []
if compaction_fired and not l1:
    print("  NOTE no L1 chunk was produced. Not fatal — the rollup threshold "
          "may simply not have been reached — but if the run compacted and the "
          "hierarchy never advanced, that is the 2026-08-28 dead-hierarchy "
          "shape and worth a look.")
else:
    print(f"  ok   the summary hierarchy advanced (L1={len(l1)})")

print()
print(f"All soak checks passed over {TURNS} turns.")
print("REMINDER: the fixture's tokenizer is not Cydonia's. This proves the "
      "system holds together over a growing conversation, NOT that the "
      "production token numbers are right.")
