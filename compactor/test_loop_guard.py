"""
v3.1.9.2 — repetition-loop hardening.

Production facts (logs only; no real data used or opened here):

  - The model (Cydonia-24B, vLLM 0.19.0) sometimes degenerates into one
    token/phrase repeated, or an unbroken line of short fragments.
    `reply_is_degenerate` + `_redact_degenerate_turns` already keep such a
    reply out of what gets MEMORIZED (facts, episodic index, L1/L2/L3
    rollup). They do NOT keep it out of what is FORWARDED to vLLM: OpenWebUI
    resends the loop reply in chat history on every later request, and right
    after a loop the hard-budget guard's ~5-turn window makes it a large
    fraction of everything the model sees — plausibly why the reply AFTER a
    loop has come back empty in production. Section [2] covers the fix:
    `_redact_forwarded_loop_replies`, called in chat_completions after
    compaction/injection and before `_enforce_hard_budget`.

  - The owner's OpenWebUI model had Ollama's `repeat_penalty` set.
    `apply_model_params_to_body_openai` passes unknown keys through
    verbatim, vLLM 0.19 ignores `repeat_penalty` (lands in `model_extra`),
    so `SamplingParams.repetition_penalty` silently stayed at 1.0. Section
    [1] covers the fix: `_translate_ollama_sampling_params`, called right
    after conv_id resolution in chat_completions.

Only synthetic conversation content appears below (project rule: this repo
is public).

Run inside the compactor image or any container with the requirements:
    python test_loop_guard.py
"""

import json
import logging
import os
import sys
import tempfile
import time
from unittest.mock import patch

os.environ.setdefault("MODEL_REPO", "test-model")
os.environ.setdefault("VLLM_URL", "http://stub:8000")
os.environ["COMPACTOR_RAG_ENABLED"] = "false"  # no ChromaDB/fastembed here

_TMP_ROOT = tempfile.mkdtemp(prefix="compactor-test-loop-guard-")
os.environ["COMPACTOR_STORAGE_ROOT"] = _TMP_ROOT

import main  # noqa: E402
import memory  # noqa: E402
import retrieval  # noqa: E402
import summarizer  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

# Mirrors test_degenerate_skip.py / test_budget_guard.py: RAG stays off and
# latched unavailable so no test can accidentally reach for a real embedder.
retrieval._available = False
retrieval._embedder = None
retrieval._chroma_collection = None

# Same fast-fail as test_budget_guard.py R22: nothing here points VLLM_URL
# at a live server, so an unstubbed /tokenize call would just burn ~4s per
# hit on two stacked dead-TCP timeouts. Fail it immediately instead.
_real_httpx_post = main.httpx.post


def _fail_tokenize_fast(url, *args, **kwargs):
    if "/tokenize" in url:
        raise main.httpx.ConnectError("connection refused (stubbed for test speed)")
    return _real_httpx_post(url, *args, **kwargs)  # pragma: no cover


main.httpx.post = _fail_tokenize_fast

memory.ensure_storage_layout()

client = TestClient(main.app, client=("127.0.0.1", 12347), raise_server_exceptions=False)


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


def user(text):
    return {"role": "user", "content": text}


def asst(text):
    return {"role": "assistant", "content": text}


# Same shape as the 2026-08-29 production incident and test_degenerate_skip's
# fixture: a long run of one repeated character has no clean sentence head,
# so it is guaranteed to hit the placeholder branch, not the head-kept one.
RULE = "━"
DEGENERATE_TEXT = RULE * 400


class _CaptureLogs(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


def _find(records, needle):
    return any(needle in r.getMessage() for r in records)


class _StubResponse:
    status_code = 200
    text = ""

    def json(self):
        return {
            "id": "stub",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
        }


class _StubVLLM:
    """Same shape as test_budget_guard._StubVLLM: stands in for
    httpx.AsyncClient and records the JSON body actually POSTed upstream —
    the assertion surface for both sections below."""

    sent: list = []

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None, **kwargs):
        _StubVLLM.sent.append(json)
        return _StubResponse()

    def stream(self, *args, **kwargs):
        raise AssertionError("these tests drive the non-streaming path only")

    async def aclose(self):
        pass


def _swallow_tail(coro, label=None):
    try:
        coro.close()
    except Exception:
        pass


def _post_chat(messages, conv_id, extra_body=None):
    """One real POST /v1/chat/completions with vLLM stubbed.

    -> (response, forwarded_body, log_records)."""
    _StubVLLM.sent.clear()
    handler = _CaptureLogs()
    lg = logging.getLogger("compactor")
    lg.addHandler(handler)
    body = {"model": "stub-model", "messages": messages, "stream": False}
    if extra_body:
        body.update(extra_body)
    try:
        with patch.object(main.httpx, "AsyncClient", _StubVLLM), \
             patch.object(main, "_fire_and_forget", _swallow_tail):
            r = client.post(
                "/v1/chat/completions",
                json=body,
                headers={"X-Conversation-Id": conv_id},
            )
    finally:
        lg.removeHandler(handler)
    return r, (_StubVLLM.sent[-1] if _StubVLLM.sent else None), handler.records


print("=" * 70)
print("[1] R1 — Ollama sampling-name translation")
print("=" * 70)

# [1a] translate: repeat_penalty alone -> repetition_penalty, repeat_penalty
# removed from the forwarded body.
msgs = [user("hello there, how are you today")]
r, sent, records = _post_chat(msgs, "loop-r1-translate", {"repeat_penalty": 1.2})
assert_eq(r.status_code, 200, "[1a] request accepted")
assert_true(sent is not None, "[1a] request reached vLLM")
assert_eq(sent.get("repetition_penalty"), 1.2, "[1a] repetition_penalty set from repeat_penalty")
assert_true("repeat_penalty" not in sent, "[1a] repeat_penalty removed from forwarded body")
assert_true(_find(records, "translated Ollama repeat_penalty"), "[1a] INFO line logged")

# [1b] both present: repetition_penalty wins, repeat_penalty just dropped.
r, sent, records = _post_chat(
    msgs, "loop-r1-both",
    {"repeat_penalty": 1.9, "repetition_penalty": 1.05},
)
assert_eq(sent.get("repetition_penalty"), 1.05, "[1b] repetition_penalty (caller's) wins")
assert_true("repeat_penalty" not in sent, "[1b] repeat_penalty still removed")

# [1c] numeric-string repetition_penalty coerced to float (hand-built client).
r, sent, records = _post_chat(msgs, "loop-r1-string", {"repetition_penalty": "1.1"})
assert_eq(sent.get("repetition_penalty"), 1.1, "[1c] string repetition_penalty coerced to float")
assert_true(isinstance(sent.get("repetition_penalty"), float), "[1c] result is a float, not str")

# [1d] bad value (non-numeric / non-positive) dropped with a WARNING, never
# forwarded as repetition_penalty.
r, sent, records = _post_chat(msgs, "loop-r1-bad", {"repeat_penalty": "not-a-number"})
assert_true("repetition_penalty" not in sent, "[1d] bad repeat_penalty not forwarded")
assert_true("repeat_penalty" not in sent, "[1d] bad repeat_penalty removed too")
warn = [rec for rec in records if rec.levelno == logging.WARNING and "dropping repeat_penalty" in rec.getMessage()]
assert_true(bool(warn), "[1d] WARNING logged for the dropped bad value")

r2, sent2, records2 = _post_chat(msgs, "loop-r1-bad-zero", {"repeat_penalty": 0})
assert_true("repetition_penalty" not in sent2, "[1d] zero repeat_penalty (not > 0) dropped")

# [1e] repeat_last_n has no vLLM equivalent: removed, logged, never forwarded.
r, sent, records = _post_chat(msgs, "loop-r1-lastn", {"repeat_last_n": 64})
assert_true("repeat_last_n" not in sent, "[1e] repeat_last_n removed")
assert_true(_find(records, "dropping repeat_last_n"), "[1e] dropped-parameter line logged")

# [1f] CONTROL: a body with neither key is forwarded with sampling keys
# untouched (no repeat_penalty/repetition_penalty ever appear).
r, sent, records = _post_chat(msgs, "loop-r1-control", {"temperature": 1.1})
assert_eq(sent.get("temperature"), 1.1, "[1f] CONTROL: unrelated sampling key untouched")
assert_true("repetition_penalty" not in sent, "[1f] CONTROL: no key materialized from nothing")
assert_true("repeat_penalty" not in sent, "[1f] CONTROL: no key materialized from nothing")

print()
print("=" * 70)
print("[2] R2 — degenerate assistant turns redacted from the FORWARDED window")
print("=" * 70)

# [2a] a degenerate assistant turn in recent history is replaced by the
# placeholder in what reaches vLLM; alternation (role sequence) is intact.
history = [
    user("what does the sunrise look like"),
    asst(DEGENERATE_TEXT),
    user("are you still there"),
]
r, sent, records = _post_chat(history, "loop-r2-basic")
assert_eq(r.status_code, 200, "[2a] request accepted")
sent_msgs = sent["messages"]
roles_before = [m["role"] for m in history]
roles_after = [m["role"] for m in sent_msgs if m["role"] != "system"]
assert_eq(roles_after, roles_before, "[2a] role alternation preserved")
degenerate_forwarded = [m for m in sent_msgs if m.get("content") == DEGENERATE_TEXT]
assert_true(not degenerate_forwarded, "[2a] raw degenerate text did not reach vLLM")
placeholder_forwarded = [m for m in sent_msgs if m.get("content") == main._DEGENERATE_FORWARD_PLACEHOLDER]
assert_true(bool(placeholder_forwarded), "[2a] placeholder present in forwarded body")
assert_true(_find(records, "replaced 1 degenerate assistant turn"), "[2a] INFO count-only line logged")

# [2b] CONTROL: a healthy turn that is NOT the newest message is
# byte-identical to what the client sent. (The newest user message gets the
# current-time line prepended by a later, unrelated stage — main.py's
# _time_line_for_request — so only a non-newest turn is a clean probe for
# "did OUR redaction touch anything it wasn't supposed to".)
first_user_sent = next(m["content"] for m in sent_msgs if m["role"] == "user")
assert_eq(first_user_sent, "what does the sunrise look like", "[2b] CONTROL: earlier user turn byte-identical")

# [2c] CONTROL: a fully clean conversation is forwarded unchanged. The
# newest message gets a current-time line prepended by a later, unrelated
# stage (_time_line_for_request), so only the non-newest turns are a clean
# probe here.
clean = [
    user("what's the weather like"),
    asst("It's sunny and about 70 degrees where you are."),
    user("nice, thanks"),
]
r, sent, records = _post_chat(clean, "loop-r2-clean")
non_newest_clean = {m["content"] for m in clean[:-1]}
sent_contents = {m["content"] for m in sent["messages"] if m["role"] != "system"}
assert_true(non_newest_clean.issubset(sent_contents), "[2c] CONTROL: clean earlier turns forwarded verbatim")
assert_true(not _find(records, "replaced"), "[2c] CONTROL: no replacement logged for a clean conversation")

# [2d] the newest message is never touched even if the client asked to
# continue a degenerate final assistant turn (continue_final_message=True).
# Guards the "never the newest message" rule directly against the detector,
# not just against role — an assistant-final-turn request is the one shape
# where role alone would not have protected it.
continued = [user("go on"), asst(DEGENERATE_TEXT)]
r, sent, records = _post_chat(
    continued, "loop-r2-newest", {"continue_final_message": True, "add_generation_prompt": False}
)
last_sent = sent["messages"][-1]
assert_eq(last_sent["content"], DEGENERATE_TEXT, "[2d] newest (final) assistant turn left untouched")

# [2e] user turns are never redacted even if their text alone would trip the
# detector — the detector is only ever run on assistant turns (the shared
# helper is called from a site gated on role == "assistant").
user_degenerate = [user(DEGENERATE_TEXT), user("hello")]
# reply_is_degenerate is calibrated for assistant text; call it directly to
# confirm this fixture WOULD trip it if role weren't the gate, so [2e] is
# actually testing the gate and not an accidentally-clean fixture.
assert_true(main.reply_is_degenerate(DEGENERATE_TEXT) is not None, "[2e] fixture sanity: DEGENERATE_TEXT trips the detector")
r, sent, records = _post_chat(user_degenerate, "loop-r2-user-untouched")
first_user = sent["messages"][0]
assert_eq(first_user["role"], "user", "[2e] role preserved")
assert_true(
    first_user["content"] == DEGENERATE_TEXT or main._DEGENERATE_FORWARD_PLACEHOLDER not in first_user["content"],
    "[2e] a degenerate-shaped USER turn is not redacted",
)

# [2f] a degenerate reply with a CLEAN SENTENCE HEAD (hostile pass #4,
# reviewer A F7) keeps that head in the forwarded window instead of being
# replaced whole — same rule _redact_degenerate_turns already applies to the
# rollup input, via the SHARED helper (_degenerate_replacement_content). A
# site that duplicated this rule instead of sharing it could silently use a
# different cut (or none at all); this is the check that must catch that.
_PROSE = ("We went over the plan for the week. The clinic opens at nine and "
          "the pharmacy closes early on Fridays. Bring the blue folder with "
          "the insurance letter, the list of questions we wrote together, and "
          "the notebook with last month's readings. Ask about the new dose "
          "before you leave, and write the answer down. ")
_loop_reply_with_head = "answer here. " + _PROSE + _PROSE.replace("week", "month") + "\n" + (RULE * 600)
_cut = main.decide_memory_tail(_loop_reply_with_head, finished=False, truncated=True, holed=False)
assert_true(
    bool(main.reply_is_degenerate(_loop_reply_with_head)) and _cut.outcome == "stored_trimmed",
    "[2f] fixture: the full reply is a loop, and memory's own rule keeps a clean head",
)
history_with_head = [
    user("what should I remember"),
    asst(_loop_reply_with_head),
    user("got it, anything else"),
]
r, sent, records = _post_chat(history_with_head, "loop-r2-cleanhead")
sent_asst = next(m["content"] for m in sent["messages"] if m.get("role") == "assistant")
assert_eq(sent_asst, _cut.text, "[2f] forwarded content is exactly memory's clean head, not the placeholder")
assert_true(RULE not in sent_asst, "[2f] the runaway tail did not reach vLLM")

print()
print("[3] pairing: fingerprinting happens on ORIGINAL text, before this redaction runs")
print("-" * 70)
# compact_if_needed's reuse machinery pairs recent turns against the stored
# covered-turn record by fingerprinting the turn's CONTENT
# (summarizer._covered_turn_fingerprint). This call site runs AFTER
# compaction (see the comment at the call site in chat_completions), so that
# fingerprinting always sees the untouched original text — proven here by
# showing the fingerprint of the original differs from the fingerprint of
# what this redaction would produce, so an accidental reordering (redact
# BEFORE compaction) would visibly break pairing rather than silently no-op.
_fp = summarizer._covered_turn_fingerprint
_orig_fp = _fp(asst(DEGENERATE_TEXT))
_redacted_msgs, _n = main._redact_forwarded_loop_replies(
    [user("x"), asst(DEGENERATE_TEXT), user("y")]
)
_redacted_fp = _fp(_redacted_msgs[1])
assert_eq(_n, 1, "[3] fixture: exactly one turn redacted")
assert_true(_orig_fp != _redacted_fp, "[3] redacted content fingerprints differently than the original — "
            "confirms redaction must stay AFTER compaction's fingerprinting, not before")

print()
print("[3b] pairing END-TO-END: redacting BEFORE compaction breaks reuse, "
      "redacting AFTER (the real call site) does not")
print("-" * 70)
# Builds a real covered-turn record the way _record_chunk_fps writes it (same
# writer test_compaction_reuse.py uses via its _seed helper), covering a
# 20-turn span that has a degenerate assistant turn inside it, then computes
# summarizer._coverage_plan on (a) the ORIGINAL messages — what compaction
# itself sees, since this redaction runs AFTER it — and (b) messages already
# run through _redact_forwarded_loop_replies — what compaction would see if
# this call were moved ahead of it, which is exactly the ordering mistake R2
# warns against.
_PAIR_CONV = "loop-r2-pairing"


def _pair_turn(i):
    if i % 2 == 1:
        return user(f"question {i}")
    if i == 10:
        return asst(DEGENERATE_TEXT)
    return asst(f"answer {i}, a plain reply with a clean sentence.")


_pair_msgs = [_pair_turn(i) for i in range(1, 21)]  # turns 1..20
_pair_state = summarizer.load_state(_PAIR_CONV)
# Same shape test_compaction_reuse.py's _seed() writes: an unbroken L1 chain
# from turn 1 (so _covered_prefix sees it) plus the per-turn fingerprints
# _record_chunk_fps writes from what the chunk actually read.
_pair_state["l1"] = [{"text": "STORED-CHUNK", "first_turn": 1, "last_turn": 20}]
_pair_state["last_summarized_turn"] = 20
summarizer._record_chunk_fps(_pair_state, 1, 20, _pair_msgs)
summarizer.save_state(_PAIR_CONV, _pair_state)

_covered_orig, _changed_orig = summarizer._coverage_plan(_pair_state, _pair_msgs)
assert_eq(_covered_orig, 20, "[3b] ORIGINAL messages: the whole seeded span reuses (compaction's real input)")
assert_true(9 not in _changed_orig, "[3b] ORIGINAL: turn 10 (0-based index 9, the degenerate one) is NOT flagged changed")

_pair_msgs_redacted, _pair_n = main._redact_forwarded_loop_replies(_pair_msgs + [user("21st, keeps turn 10 non-newest")])
_pair_msgs_redacted = _pair_msgs_redacted[:20]  # drop the trailing probe turn again
assert_eq(_pair_n, 1, "[3b] fixture: exactly the one degenerate turn was redacted")
_covered_bad, _changed_bad = summarizer._coverage_plan(_pair_state, _pair_msgs_redacted)
assert_true(
    _covered_bad < 20 or 9 in _changed_bad,
    "[3b] if this redaction ran BEFORE compaction (fed to _coverage_plan "
    "instead of the original), the degenerate turn's fingerprint no longer "
    "matches the stored record and reuse breaks — this is the check that "
    "must go red for mutation 4 (R2 moved before compaction)",
)

print()
print("[4] performance: detector cost on a large forwarded window stays bounded")
print("-" * 70)
# The detector now runs on EVERY request (it did not before). Build a large
# but realistic recent window (the hard-budget guard already caps what
# reaches this point to a handful of turns in production, but this measures
# the redaction function in isolation against a much larger window than it
# will ever actually see, as a safety margin) and time it.
_big_window = []
for i in range(200):
    _big_window.append(user(f"question number {i} about something ordinary"))
    _big_window.append(asst(f"answer number {i}, a perfectly normal reply with several sentences. "
                             f"It has more than one clause. It ends cleanly."))
_t0 = time.monotonic()
_out, _replaced = main._redact_forwarded_loop_replies(_big_window)
_elapsed_ms = (time.monotonic() - _t0) * 1000
print(f"  {len(_big_window)} messages, {_elapsed_ms:.1f} ms, {_replaced} replaced (expect 0)")
assert_eq(_replaced, 0, "[4] no false positives on ordinary prose")
assert_true(_elapsed_ms < 2000, f"[4] stayed under 2000ms for a 400-message window (got {_elapsed_ms:.1f}ms)")

print()
print("ALL TESTS PASSED")
