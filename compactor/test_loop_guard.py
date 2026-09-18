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

# [1g] p7 hostile pass #7, F4: "inf" / "Infinity" / "1e999" (strings OpenWebUI
# would otherwise json-decode to a real float, but "inf" itself is not valid
# JSON, so a typed value like this reaches us as a string) used to satisfy
# `f > 0` and get forwarded as a real `inf`, which httpx 0.28.1's
# `allow_nan=False` encoder then REFUSES to serialize (ValueError, raised
# from inside the proxy, after compaction/injection already spent the turn —
# see _translate_ollama_sampling_params's docstring). Each must be dropped
# with a WARNING instead, exactly like any other invalid value.
for _bad_val, _label in (
    ("inf", "the string 'inf'"),
    ("Infinity", "the string 'Infinity'"),
    ("1e999", "the string '1e999' (parses to inf)"),
):
    r, sent, records = _post_chat(msgs, f"loop-r1-nonfinite-{_label[:6]}", {"repeat_penalty": _bad_val})
    assert_true(
        "repetition_penalty" not in sent,
        f"[1g] *** {_label} as repeat_penalty is dropped, not forwarded as inf",
    )
    assert_true(
        _find(records, "dropping repeat_penalty"),
        f"[1g] {_label}: WARNING logged for the dropped non-finite value",
    )

# [1g2] the same for repetition_penalty sent directly (numeric OR string —
# a numeric 1e999 parses to inf at json.loads time too, closing that half of
# F4's pre-existing hole as well).
r, sent, records = _post_chat(msgs, "loop-r1-nonfinite-direct-str", {"repetition_penalty": "inf"})
assert_true("repetition_penalty" not in sent, "[1g2] string repetition_penalty='inf' dropped")
# Build the JSON body TEXT by hand: `1e999` must appear as the literal
# digits of a JSON NUMBER (not the bare `Infinity` CONSTANT chat_completions
# already rejects at parse time via `parse_constant`). Python's own float()
# happily turns the digit string "1e999" into inf during json.loads's number
# parsing, which `parse_constant` never saw before P8-6 (hostile pass #8) —
# that path was a hole distinct from the string-to-inf coercion this lane
# closed first (F4). json.dumps(1e999) cannot be used to build this fixture:
# Python's own `1e999` LITERAL already evaluates to inf before json.dumps
# ever sees it, so it would round-trip as the `Infinity` constant, not as
# digits.
_body_1e999 = (
    '{"model": "stub-model", "messages": ' + json.dumps(msgs) +
    ', "stream": false, "repetition_penalty": 1e999}'
)
assert_true("1e999" in _body_1e999, "[1g2] fixture sanity: the raw body text carries the digits 1e999")
_StubVLLM.sent.clear()
_h = _CaptureLogs()
_lg = logging.getLogger("compactor")
_lg.addHandler(_h)
try:
    with patch.object(main.httpx, "AsyncClient", _StubVLLM), \
         patch.object(main, "_fire_and_forget", _swallow_tail):
        _r_direct = client.post(
            "/v1/chat/completions",
            content=_body_1e999,
            headers={"X-Conversation-Id": "loop-r1-nonfinite-numeric", "Content-Type": "application/json"},
        )
finally:
    _lg.removeHandler(_h)
_sent_direct = _StubVLLM.sent[-1] if _StubVLLM.sent else None
# P8-6 (hostile pass #8): this numeral used to reach
# _translate_ollama_sampling_params, get coerced, and be dropped there
# (the assertion this replaces). Now chat_completions's json.loads itself
# (parse_float=_finite_json_float) rejects ANY numeral that overflows to
# inf, for every numeric field at once — this request never reaches
# translation, compaction or vLLM at all; it 400s at the door instead.
assert_eq(
    _r_direct.status_code, 400,
    "[1g2] *** a numeric 1e999 repetition_penalty (any numeral that "
    "overflows to inf, in ANY numeric field) is now rejected at JSON-parse "
    "time with a 400, not coerced and forwarded",
)
assert_true(_sent_direct is None, "[1g2] the request never reached vLLM")

# [1h] CONTROL: an ordinary finite repeat_penalty still translates normally —
# the non-finite check does not accidentally reject real values.
r, sent, records = _post_chat(msgs, "loop-r1-finite-control", {"repeat_penalty": 1.15})
assert_eq(sent.get("repetition_penalty"), 1.15, "[1h] CONTROL: an ordinary finite value still translates")

# [1i] p7 hostile pass #7, F4: when repetition_penalty ITSELF is invalid but
# repeat_penalty is valid, the valid one is used instead of losing both.
r, sent, records = _post_chat(
    msgs, "loop-r1-fallback",
    {"repeat_penalty": 1.1, "repetition_penalty": "abc"},
)
assert_eq(
    sent.get("repetition_penalty"), 1.1,
    "[1i] *** repetition_penalty invalid + repeat_penalty valid -> the valid "
    "repeat_penalty value is forwarded, not neither",
)

# [1j] CONTROL: when repetition_penalty IS valid, it still wins over
# repeat_penalty exactly as [1b] already covers — repeated here so a mutant
# that always prefers repeat_penalty cannot hide behind [1i] alone.
r, sent, records = _post_chat(
    msgs, "loop-r1-both-still-wins",
    {"repeat_penalty": 1.9, "repetition_penalty": 1.05},
)
assert_eq(sent.get("repetition_penalty"), 1.05, "[1j] CONTROL: a VALID repetition_penalty still wins")

# [1k] P8-6 (hostile pass #8): a numeric max_tokens that overflows to inf
# ("1e999", as literal JSON digits — see [1g2]'s comment on why this must
# be built by hand, not via json.dumps) used to reach
# int(body.get("max_tokens") or 0) and raise OverflowError, which the
# surrounding `except (TypeError, ValueError)` did NOT catch — a 500 from
# inside the proxy. It is now rejected at JSON-PARSE time with a 400, for
# ANY numeral that overflows, not only this one field (chat_completions's
# json.loads now takes parse_float=_finite_json_float).
_body_max_tokens_1e999 = (
    '{"model": "stub-model", "messages": ' + json.dumps(msgs) +
    ', "stream": false, "max_tokens": 1e999}'
)
_StubVLLM.sent.clear()
_h6 = _CaptureLogs()
_lg6 = logging.getLogger("compactor")
_lg6.addHandler(_h6)
try:
    with patch.object(main.httpx, "AsyncClient", _StubVLLM), \
         patch.object(main, "_fire_and_forget", _swallow_tail):
        _r6 = client.post(
            "/v1/chat/completions",
            content=_body_max_tokens_1e999,
            headers={"X-Conversation-Id": "loop-r1-maxtokens-inf", "Content-Type": "application/json"},
        )
finally:
    _lg6.removeHandler(_h6)
assert_eq(
    _r6.status_code, 400,
    "[1k] *** P8-6: a max_tokens numeral that overflows to inf is REJECTED "
    "with a 400, not a 500",
)
assert_true(not _StubVLLM.sent, "[1k] the request never reached vLLM")

# [1l] CONTROL: an ordinary, finite max_tokens is untouched — the new
# parse-time check does not reject real values, and a valid value is never
# rewritten (only ever dropped when invalid, per P8-6).
r, sent, records = _post_chat(msgs, "loop-r1-maxtokens-control", {"max_tokens": 500})
assert_eq(r.status_code, 200, "[1l] CONTROL: an ordinary max_tokens request is accepted")
assert_eq(sent.get("max_tokens"), 500, "[1l] CONTROL: a valid max_tokens is forwarded UNCHANGED, never rewritten")

# [1m] P8-6: an UNPARSEABLE (but JSON-valid) max_tokens — a string — is
# dropped from the forwarded body with a WARNING instead of riding along
# unexamined. Before this fix, int(body.get("max_tokens") or 0) decided
# the LOCAL budget math would treat it as absent (0), but left the
# client's own bad value sitting untouched in `body`, still headed for
# vLLM as-is.
r, sent, records = _post_chat(msgs, "loop-r1-maxtokens-bad", {"max_tokens": "not-a-number"})
assert_eq(r.status_code, 200, "[1m] a bad max_tokens does not fail the whole request")
assert_true("max_tokens" not in sent, "[1m] *** P8-6: the invalid max_tokens is dropped, not forwarded as-is")
assert_true(
    _find(records, "dropping invalid max_tokens"),
    "[1m] WARNING logged for the dropped bad value",
)

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
assert_true(_find(records, "touched 1 degenerate assistant turn"), "[2a] INFO count-only line logged")
assert_true(_find(records, "whole=1 cut=0"), "[2a] *** P8-8: whole/cut split logged (DEGENERATE_TEXT has no clean head, so it is a WHOLE replacement)")

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
assert_true(not _find(records, "touched"), "[2c] CONTROL: no replacement logged for a clean conversation")

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

# [2g] p7 hostile pass #7, F2: a PHRASE loop ("X. X. X. ...") ends on
# sentence boundaries, so the OLD rule (trim_to_last_sentence on the WHOLE
# text, then re-judge) could not cut it at all -- the "clean head" it found
# was still the whole degenerate reply, so it fell back to the placeholder
# and threw away a real, long, clean answer. The fix cuts at the position
# the detector itself located (start of the phrase-loop span), which cannot
# be inside the loop, so the head it re-judges is guaranteed clean.
_phrase_unit = "Absolutely. With Desperation. With Humility. "
_phrase_loop_reply = "answer here. " + _PROSE + _PROSE.replace("week", "month") + "\n" + (_phrase_unit * 30)
assert_true(
    bool(main.reply_is_degenerate(_phrase_loop_reply)),
    "[2g] fixture: the phrase loop trips the detector",
)
assert_true(
    main.decide_memory_tail(_phrase_loop_reply, finished=False, truncated=True, holed=False).outcome
    != "stored_trimmed",
    "[2g] fixture sanity: the OLD whole-text trim rule cannot cut this clean "
    "(it lands back inside the loop's own sentence boundaries, so memory's "
    "rule — unchanged by this lane — does not call it a clean trim either)",
)
r, sent, records = _post_chat(
    [user("what should I remember"), asst(_phrase_loop_reply), user("got it, anything else")],
    "loop-r2-phrase",
)
_g_sent_asst = next(m["content"] for m in sent["messages"] if m.get("role") == "assistant")
assert_true(
    _g_sent_asst != main._DEGENERATE_FORWARD_PLACEHOLDER,
    "[2g] *** the clean prose head survived — NOT replaced whole by the "
    "placeholder just because the loop itself ends on sentence boundaries",
)
assert_true("clinic opens at nine" in _g_sent_asst, "[2g] the real prose head reached vLLM")
# Not necessarily ZERO copies of the phrase: _tail_loop_span (unmodified by
# this lane) measures its span in whole repeated UNITS from the end of the
# text, and can leave up to about one unit's worth of the loop's own start
# uncollapsed when that leftover fragment happens to end on a sentence
# boundary itself (as this phrase does, being itself period-terminated) —
# trim_to_last_sentence then legitimately keeps it as "the last sentence".
# The fix is judged the way F2's own real-data proof judges it: the loop is
# overwhelmingly cut, not that a detector span imprecision inherited from
# unmodified code is papered over here.
_g_phrase_copies = _g_sent_asst.count(_phrase_unit.strip())
assert_true(
    _g_phrase_copies <= 2,
    f"[2g] the looping phrase reached vLLM at most a couple of times, not "
    f"all 30 repeats (got {_g_phrase_copies})",
)

# [2h] p7 hostile pass #7, F3: a healthy reply with ONE mid-reply elongation
# (a scream) is not thrown away whole -- only the flagged span is collapsed,
# and both the clean text BEFORE and AFTER it reach the model.
_scream = "A" * 130
_mid_span_reply = (
    _PROSE + "Then she screamed: " + _scream + ". After that, everyone sat "
    "back down and the meeting continued as planned, calmly and clearly, "
    "for several more minutes without incident."
)
assert_true(bool(main.reply_is_degenerate(_mid_span_reply)), "[2h] fixture: the scream trips the detector")
r, sent, records = _post_chat(
    [user("what happened"), asst(_mid_span_reply), user("go on")],
    "loop-r2-midspan",
)
_h_sent_asst = next(m["content"] for m in sent["messages"] if m.get("role") == "assistant")
assert_true(
    _h_sent_asst != main._DEGENERATE_FORWARD_PLACEHOLDER,
    "[2h] *** not replaced whole — a mid-reply span costs at most itself",
)
assert_true("clinic opens at nine" in _h_sent_asst, "[2h] clean text BEFORE the scream reached vLLM")
assert_true("meeting continued as planned" in _h_sent_asst, "[2h] clean text AFTER the scream reached vLLM")
assert_true(_scream not in _h_sent_asst, "[2h] the scream itself did not reach vLLM")

# [2i] p7 hostile pass #7, F3 + P8-2 (hostile pass #8) + P9-3 (hostile pass
# #9): a repeated IDENTIFIER-shaped token trips the token rule outside a
# fence, and — as of P9-3 — trips it EVERY TIME, fence or no fence. F3
# added an exemption for text inside a CLOSED fence; P8-2 narrowed it after
# an UNCLOSED opener was found to exempt everything after it forever; P9-3
# found that P8-2's own narrowing ("closed AND does not reach the end of
# the reply") is unsatisfiable — a fence can only be judged "closed" by
# finding a LATER toggle, and that later toggle is necessarily after the
# run, so "reaches the end" and "is inside a closed fence" can never both
# be true, and the "reaches the end" clause never fired (0 of 20,000
# mutation-tested verdicts depended on it). A loop inside an ordinary
# closed decorative box — the common case, not the exotic one — was
# silently exempted regardless of position. REMOVED ENTIRELY, not narrowed
# again: this rule now judges text exactly as v3.1.9 did, with no fence
# reading at all. See [2i4] below for the five-row table that pins this
# down case by case. P8-7 (hostile pass #8): the OLD [2i] fixture here
# ("[0.00, 0.00, 0.00, 0.00] " * 20) never matched _TOKEN_RUN_RE at all —
# the brackets and commas break each row into four DIFFERENT space-
# separated tokens ("[0.00,", "0.00,", "0.00,", "0.00]"), none of which
# repeats three times in a row the way the regex requires — so a CONTROL
# built on it would pass whether or not fenced code was exempt from the
# token rule. Two mutants (nofence/closed, SP\p8\fencemut.py) left every
# suite including this one at rc=0.
_ID_UNIT = "identifier_run_9f3k2"  # 20 chars, alnum — a real _TOKEN_RUN_RE unit
_ID_LOOP = (_ID_UNIT + " ") * 10   # well past DEGENERATE_TOKEN_RUN_CHARS (120)
assert_true(
    len(_ID_LOOP) >= main.DEGENERATE_TOKEN_RUN_CHARS,
    "[2i] fixture sanity: the identifier run clears the token-run threshold",
)
assert_true(
    main.reply_is_degenerate(_PROSE + _ID_LOOP) is not None,
    "[2i] *** CONTROL: the identifier run trips the token rule when it is "
    "NOT fenced at all — this fixture actually exercises _TOKEN_RUN_RE, "
    "unlike the old '[0.00, ...]' array, which matched no rule at all",
)

# Inside a fence that CLOSES (p7's F3 shape): P9-3 removed the exemption,
# so this is now flagged too, exactly as v3.1.9 judged it. The loop itself
# is still cut from what reaches vLLM like any other mid-reply span (full
# pre/post/balance assertions on this exact "prose, boxed run, more prose"
# shape live at [2k] below); the cost of losing the exemption is that a
# genuinely decorative closed box is skipped from MEMORY, not that
# anything leaks forward.
_codeblock_reply = (
    _PROSE + "Here is the matrix:\n```\n" + _ID_LOOP +
    "\n```\nLet me know if that helps, and I can explain any row you like."
)
assert_true(
    main.reply_is_degenerate(_codeblock_reply) is not None,
    "[2i] *** P9-3: the SAME identifier run inside a fence that CLOSES is "
    "now flagged — the fence exemption is gone, not narrowed",
)
r, sent, records = _post_chat(
    [user("show me"), asst(_codeblock_reply), user("thanks")],
    "loop-r2-codeblock",
)
_i_sent_asst = next(m["content"] for m in sent["messages"] if m.get("role") == "assistant")
assert_true(
    _ID_UNIT * 3 not in _i_sent_asst,
    "[2i] *** P9-3: the identifier loop inside the closed box did not "
    "reach vLLM verbatim in the forwarded window",
)
assert_true(
    "Let me know if that helps" in _i_sent_asst,
    "[2i] the clean prose AFTER the closed box still reached vLLM",
)

# [2i2] P8-2 (hostile pass #8), still true after P9-3 removed the fence
# exemption entirely: the SAME identifier run after an UNCLOSED ``` line (a
# lone opener — this model's own decorative-box style; 128 of 1,709 unique
# real replies in the 2026-09-16 backup have an odd ``` count) is flagged,
# and is kept out of the forwarded window. Kept as a regression check: with
# no fence reading left in the token rule at all, this passes for a
# simpler reason now (there is nothing to exempt it), but the case must
# stay caught.
_unclosed_fence_reply = _PROSE + "Here is a note:\n```\n" + _ID_LOOP
assert_true(
    main.reply_is_degenerate(_unclosed_fence_reply) is not None,
    "[2i2] *** P8-2: the identifier run after an UNCLOSED ``` opener is "
    "flagged — an unmatched opener must not exempt everything after it "
    "forever",
)
r, sent, records = _post_chat(
    [user("note this"), asst(_unclosed_fence_reply), user("thanks")],
    "loop-r2-unclosed-fence",
)
_i2_sent_asst = next(m["content"] for m in sent["messages"] if m.get("role") == "assistant")
assert_true(
    _ID_UNIT * 3 not in _i2_sent_asst,
    "[2i2] *** P8-2: the identifier loop after the unclosed fence did not "
    "reach vLLM verbatim in the forwarded window",
)
assert_true(
    "clinic opens at nine" in _i2_sent_asst,
    "[2i2] the clean prose BEFORE the unclosed fence still reached vLLM",
)

# [2i3] Originally P8-2's isolation of the "must actually CLOSE" half from
# the "never if it reaches the end" half (that second half was P9-3's dead
# clause — see [2i]'s comment). Kept as a regression check now that there
# is no fence reading at all: a MID-reply identifier run inside a fence
# that NEVER closes anywhere in the whole reply must stay flagged.
_mid_unclosed = (
    _PROSE + "\n```\n" + _ID_LOOP +
    "\nmore prose after the loop, and the fence never closes anywhere in "
    "this whole reply, so by toggle count it stays open all the way to "
    "the end even though the run itself is nowhere near that end."
)
assert_true(
    main.reply_is_degenerate(_mid_unclosed) is not None,
    "[2i3] *** P8-2: a MID-reply identifier run inside a fence that NEVER "
    "closes is still flagged — exemption requires the fence to actually "
    "CLOSE, not merely to still read as 'open' by toggle parity",
)

# [2i4] P9-3 (hostile pass #9) regression table — the exact five rows the
# lane brief verified against the v3.1.9 tag before assigning this fix:
#
#   case                                    v3.1.9   de3c376 (unfixed)
#   loop, no fence                          flagged  flagged
#   loop after an UNCLOSED fence            flagged  flagged
#   loop inside a fence that CLOSES         flagged  NOT flagged  <- hole
#   closed fence, then trailing text        flagged  NOT flagged  <- hole
#   repeated-value array in a closed fence  flagged  NOT flagged  <- hole
#
# Every row must now read as flagged, matching the v3.1.9 column: the
# token rule has no fence exemption of any kind any more, so a fence
# closing or not closing, or being followed by more text or not, cannot
# change the verdict.
_i4_cases = {
    "no fence at all": _PROSE + _ID_LOOP,
    "after an unclosed fence opener": _PROSE + "```\n" + _ID_LOOP,
    "inside a fence that closes, mid-reply": (
        _PROSE + "```\n" + _ID_LOOP + "\n```\nmore prose follows the box."
    ),
    "inside a fence that closes, ending the reply": (
        _PROSE + "```\n" + _ID_LOOP + "\n```"
    ),
    "inside a fence that closes, ending the reply, trailing newline": (
        _PROSE + "```\n" + _ID_LOOP + "\n```\n"
    ),
    "repeated-value array shape, closed fence, ending the reply": (
        _PROSE + "```\n" + ((_ID_UNIT + "_row ") * 10) + "\n```"
    ),
}
for _i4_name, _i4_text in _i4_cases.items():
    assert_true(
        main.reply_is_degenerate(_i4_text) is not None,
        f"[2i4] *** P9-3: '{_i4_name}' is flagged — matches v3.1.9, no "
        "fence exemption of any shape survives",
    )

# [2i5] P9-6 (hostile pass #9): a 4-space-indented ``` line is CommonMark
# indented CODE CONTENT, not a fence delimiter — `_fence_toggle_offsets`
# used to strip all leading whitespace before checking, so such a line was
# wrongly counted as a toggle. A real opener (unindented) followed by a
# 4-space-indented ``` line (literal content, not a closer) must still
# read as an OPEN fence at a position after the indented line — if the
# indented line were wrongly treated as a toggle, that position would
# wrongly read as CLOSED.
_indent_toggles = main._fence_toggle_offsets(
    "prose\n```\ncode line one\n    ```\nmore code\n```\nprose after"
)
assert_eq(
    len(_indent_toggles), 2,
    "[2i5] *** P9-6: only the two UNINDENTED ``` lines are toggles — the "
    "4-space-indented one in the middle is literal content",
)
assert_true(
    main._in_open_fence(_indent_toggles, len("prose\n```\ncode line one\n    ```\nmore c")),
    "[2i5] *** P9-6: a position after the (correctly ignored) indented "
    "``` line still reads as inside the still-open real fence",
)


# [2j] P8-3 (hostile pass #8): the last real sentence of the reply sits
# INSIDE a ``` box that never closes again (this model's own "box, then
# runs to the end" shape — see P8-2's comment on how common the unclosed
# form is), and a tail loop runs from right after it to the end. The OLD
# memory-side rule (trim_to_last_sentence: no boundary inside ANY fence,
# closed or not) would find NOTHING before this loop — the one real
# sentence is excluded for being fenced, and nothing else in the prefix
# ends in punctuation at all — so the whole reply would have gone out as
# the placeholder. The forwarded path must do better: nothing is
# extracted from this text, so a boundary inside the box is fine, and the
# kept prefix must come back self-balanced (P8-4) even though the box it
# was cut from never closes in the original.
_p3_prefix_no_period = (
    "Preliminary remark with no terminal punctuation anywhere in it at all "
    "and it keeps going for a while without a single period question mark "
    "or exclamation point anywhere in this whole stretch of ordinary words "
    "so that the kept prefix clears the memorable-trim floor on its own "
    "even before the box and its one real sentence are added after it here"
)
_p3_boxed_reply = (
    _p3_prefix_no_period + "\n```\n"
    "Final thought inside an unclosed box ends here.\n" + (RULE * 600)
)
assert_true(bool(main.reply_is_degenerate(_p3_boxed_reply)), "[2j] fixture: the tail run trips the detector")
assert_eq(
    main.decide_memory_tail(_p3_boxed_reply, finished=False, truncated=True, holed=False).outcome,
    main.tailhealth.SKIPPED_NO_BOUNDARY,
    "[2j] fixture sanity: the OLD memory-side rule (fence-restricted) finds "
    "NO sentence boundary at all before the loop — the one real sentence "
    "is excluded for sitting inside the (never-closing) box — so memory's "
    "own rule (unchanged by this lane) falls back to nothing kept",
)
r, sent, records = _post_chat(
    [user("note this down"), asst(_p3_boxed_reply), user("thanks, go on")],
    "loop-r2-boxed-tail",
)
_j_sent_asst = next(m["content"] for m in sent["messages"] if m.get("role") == "assistant")
assert_true(
    _j_sent_asst != main._DEGENERATE_FORWARD_PLACEHOLDER,
    "[2j] *** P8-3: NOT replaced whole — the sentence inside the unclosed "
    "box is kept even though the memory-side rule cannot use it",
)
assert_true(
    "Final thought inside" in _j_sent_asst,
    "[2j] *** P8-3: the sentence that sat INSIDE the box reached vLLM",
)
assert_true(RULE not in _j_sent_asst, "[2j] the tail loop itself did not reach vLLM")
assert_true(
    sum(1 for _ln in _j_sent_asst.splitlines() if _ln.strip().startswith("```")) % 2 == 0,
    "[2j] *** P8-4: the forwarded text has a BALANCED ``` count, even "
    "though it was cut from a box that never closes in the original",
)
assert_true(
    main.reply_is_degenerate(_j_sent_asst) is None,
    "[2j] the forwarded text is not itself re-flagged as degenerate",
)

# [2k] P8-4 (hostile pass #8): a MID-reply span sitting INSIDE a fence that
# DOES close (prose, box with a status line and the flagged run, more
# prose after). The kept prefix and suffix are cut from two different
# HALVES of the same original box; joined naively, that leaves ONE stray
# ``` marker and everything after it misreads as code. Both the prose
# BEFORE and the prose AFTER the box, plus the box's own non-degenerate
# status line, must all still reach vLLM, and the result must stay
# balanced.
_p4_reply = (
    _PROSE + "Here is a diagnostic dump:\n```\n" + (RULE * 300) +
    "\nstatus: nominal\n```\n"
    "Everything looks fine now, thanks for checking. And we can continue "
    "with the next part of the plan."
)
assert_true(bool(main.reply_is_degenerate(_p4_reply)), "[2k] fixture: the boxed run trips the detector")
r, sent, records = _post_chat(
    [user("run the check"), asst(_p4_reply), user("and then?")],
    "loop-r2-boxed-midspan",
)
_k_sent_asst = next(m["content"] for m in sent["messages"] if m.get("role") == "assistant")
assert_true(
    _k_sent_asst != main._DEGENERATE_FORWARD_PLACEHOLDER,
    "[2k] *** P8-4: NOT replaced whole — a boxed mid-reply span costs at "
    "most itself",
)
assert_true("clinic opens at nine" in _k_sent_asst, "[2k] clean prose BEFORE the box reached vLLM")
assert_true(
    "Everything looks fine now" in _k_sent_asst and "next part of the plan" in _k_sent_asst,
    "[2k] clean prose AFTER the box reached vLLM",
)
assert_true(RULE not in _k_sent_asst, "[2k] the run itself did not reach vLLM")
assert_true(
    sum(1 for _ln in _k_sent_asst.splitlines() if _ln.strip().startswith("```")) % 2 == 0,
    "[2k] *** P8-4: the forwarded text has a BALANCED ``` count — the box "
    "split across the kept prefix and suffix did not leave a stray marker",
)
assert_true(
    main.reply_is_degenerate(_k_sent_asst) is None,
    "[2k] the forwarded text is not itself re-flagged as degenerate",
)

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
_redacted_msgs, _n, _n_whole = main._redact_forwarded_loop_replies(
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

_pair_msgs_redacted, _pair_n, _pair_n_whole = main._redact_forwarded_loop_replies(_pair_msgs + [user("21st, keeps turn 10 non-newest")])
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
print("[3c] pairing END-TO-END via chat_completions itself (p7 hostile pass #7, "
      "F5): [3b] above never calls chat_completions — it feeds _coverage_plan "
      "a list IT redacted itself, so it cannot see where the real call site "
      "is. This drives the actual endpoint with a conversation large enough "
      "to reuse for real and confirms reuse still fires with the redaction "
      "in its real position.")
print("-" * 70)
_E2E_CONV = "loop-r2-e2e-reuse"


def _e2e_turn(i):
    # Padded well past TARGET_TOKENS in aggregate (main.py's char/4 local
    # estimator is what's live here — /tokenize is stubbed to fail fast, same
    # as every other section in this file) so compact_if_needed actually
    # takes the compaction path instead of its early under-TARGET return.
    if i % 2 == 1:
        return user(f"question {i} " + ("word " * 60))
    if i == 10:
        return asst(DEGENERATE_TEXT)
    return asst(f"answer {i}, a plain reply with a clean sentence. " + ("word " * 60))


_e2e_msgs = [_e2e_turn(i) for i in range(1, 201)]  # turns 1..200
_e2e_state = summarizer.load_state(_E2E_CONV)
_e2e_state["l1"] = [{"text": "STORED-CHUNK", "first_turn": 1, "last_turn": 200}]
_e2e_state["last_summarized_turn"] = 200
summarizer._record_chunk_fps(_e2e_state, 1, 200, _e2e_msgs)
summarizer.save_state(_E2E_CONV, _e2e_state)

_e2e_summarize_calls = []
_real_summarize = main.summarize


async def _spy_summarize(client, to_summarize):
    _e2e_summarize_calls.append(list(to_summarize))
    return await _real_summarize(client, to_summarize)


with patch.object(main, "summarize", _spy_summarize):
    _e2e_r, _e2e_sent, _e2e_records = _post_chat(
        _e2e_msgs + [user("201st, keeps turn 10 well out of the newest slot")],
        _E2E_CONV,
    )

assert_eq(_e2e_r.status_code, 200, "[3c] request accepted")
assert_true(
    not _e2e_summarize_calls,
    "[3c] *** reuse actually fired through the real endpoint: summarize() "
    "was never called, so the seeded hierarchy covered everything — this is "
    "the check that would go red if the redaction's real call-site position "
    "(after compaction, before the guard) regressed back to running before "
    "compaction",
)
_e2e_sent_texts = [m.get("content") for m in _e2e_sent.get("messages", [])]
assert_true(
    DEGENERATE_TEXT not in _e2e_sent_texts,
    "[3c] the degenerate turn's raw text did not reach vLLM even on the "
    "reuse path",
)

print()
print("[4] performance: detector cost on a large forwarded window stays bounded")
print("-" * 70)
# p7 hostile pass #7, F5: the old fixture here (400 messages, ~100 chars
# each) measured nothing realistic and its own docstring's claim was false —
# this redaction runs BEFORE _enforce_hard_budget (main.py's chat_completions
# call order), not after, so on the DECLINED path it sees the client's WHOLE
# array, not "a handful of turns". Real-data measurement (p7's real_fp.out,
# her main chat's current branch: 811 messages, 4.66M chars) cost 0.78-0.83s
# of GIL-bound CPU PER REQUEST that declines reuse. This fixture is sized to
# the same order of magnitude (800 turns, several KB each) instead of a
# fixture too small to show the cost at all.
#
# _reply_degenerate_verdict is now cached per content digest (see its
# comment): a real conversation resends the SAME turn text on every later
# request, so only genuinely NEW turns pay full detection cost — this run
# below is a cold-cache worst case (every turn here is unique), which is
# also why it is timed as a single pass rather than "first call vs repeat".
_big_window = []
for i in range(800):
    # Varied per-sentence (the counter `j` changes every repeat) so this is
    # genuinely non-repeating prose, not an accidental phrase loop the
    # detector is SUPPOSED to catch — an earlier draft of this fixture
    # repeated the identical sentence 40x per message and tripped [4]'s own
    # "no false positives" assertion, which was the tail-loop rule correctly
    # firing on a fixture bug, not a detector bug.
    body = " ".join(
        f"Sentence {j} of turn {i} says something ordinary and specific."
        for j in range(40)
    )  # ~2.9KB per message, comparable order of magnitude to her real data
    _big_window.append(user(f"question {i}: " + body))
    _big_window.append(asst(f"answer {i}: " + body))
_t0 = time.monotonic()
_out, _replaced, _replaced_whole = main._redact_forwarded_loop_replies(_big_window)
_elapsed_ms = (time.monotonic() - _t0) * 1000
_total_chars = sum(len(main._message_text(m)) for m in _big_window)
print(f"  {len(_big_window)} messages, {_total_chars} chars, {_elapsed_ms:.1f} ms, "
      f"{_replaced} replaced (expect 0)")
assert_eq(_replaced, 0, "[4] no false positives on ordinary prose")
assert_true(
    _elapsed_ms < 5000,
    f"[4] stayed under a generous 5000ms bound for a {len(_big_window)}-message, "
    f"{_total_chars}-char window (got {_elapsed_ms:.1f}ms) — this is a safety "
    f"ceiling, not the production expectation; see the real-data number quoted "
    f"above for what she actually pays per request",
)

print()
print("[4b] the cache bound: a resent (unchanged) window is cheap, and still "
      "catches every turn it caught cold")
print("-" * 70)
# OpenWebUI resends the SAME turn text on every later request. This is
# exactly the shape the digest cache on _reply_degenerate_verdict exists for (see
# its comment): the second call over the IDENTICAL window should cost a
# small fraction of the first, and — the correctness half, not just the
# speed half — replace exactly the same turns, proving the cache cannot
# skip a turn that reached vLLM uncached.
_t1 = time.monotonic()
_out2, _replaced2, _replaced2_whole = main._redact_forwarded_loop_replies(_big_window)
_elapsed2_ms = (time.monotonic() - _t1) * 1000
print(f"  repeat pass: {_elapsed2_ms:.1f} ms, {_replaced2} replaced")
assert_eq(_replaced2, _replaced, "[4b] identical replacement count on the resent window")
assert_true(
    _elapsed2_ms < _elapsed_ms / 2,
    f"[4b] *** the cache bound actually bounds something: repeat pass "
    f"({_elapsed2_ms:.1f}ms) is well under half the cold pass "
    f"({_elapsed_ms:.1f}ms)",
)
# And with one NEW degenerate turn appended (never seen by the cache before),
# it is still caught — the cache narrows to "already-judged text", it never
# widens to "any text that looks similar".
_fresh_window = _big_window + [user("one more"), asst(DEGENERATE_TEXT), user("still there?")]
_out3, _replaced3, _replaced3_whole = main._redact_forwarded_loop_replies(_fresh_window)
assert_eq(_replaced3, _replaced + 1,
          "[4b] CONTROL: a brand-new degenerate turn appended after the cached "
          "window is still caught — the cache does not paper over a real turn "
          "that reaches vLLM")

print()
print("[5] a phrase loop longer than the tail-loop window is cut until clean "
      "(coordinator review)")
print("-" * 70)
# The tail-loop rule measures at most _TAIL_LOOP_WINDOW characters, so one
# cut removes at most that much of a long phrase loop and the kept head is
# still a loop. Real data had 10 of 67 flagged replies in that state after a
# single cut. The redaction must keep cutting until the text is clean, and
# still keep the clean head.
_head5 = " ".join(
    f"Sentence {j} is ordinary and specific about the garden." for j in range(40)
)
for _reps5 in (300, 600, 1500):
    _loop5 = _head5 + " " + ("And I will stay right here with you. " * _reps5)
    assert_true(
        len(_loop5) - len(_head5) > 2 * main._TAIL_LOOP_WINDOW,
        f"[5] fixture: the loop ({len(_loop5) - len(_head5)} chars) is longer "
        f"than two tail-loop windows",
    )
    _r5, _sent5, _ = _post_chat(
        [user("tell me about the garden"), asst(_loop5), user("and then?")],
        f"loopguard-long-{_reps5}",
    )
    assert_eq(_r5.status_code, 200, f"[5] request with a {_reps5}x loop succeeded")
    _fwd5 = [m for m in _sent5["messages"] if m.get("role") == "assistant"][0]["content"]
    assert_true(
        main.reply_is_degenerate(_fwd5) is None,
        f"[5] *** {_reps5}x: what reached vLLM is no longer flagged as a loop "
        f"({len(_fwd5)} chars forwarded)",
    )
    assert_true(
        _head5[:200] in _fwd5 and len(_fwd5) >= len(_head5) - 80,
        f"[5] *** {_reps5}x: the clean head survived "
        f"({len(_fwd5)} of {len(_head5)} head chars)",
    )

print()
print("[6] P8-5 (hostile pass #8): the cut-pass budget is bounded by "
      "CHARACTERS scanned, not a flat pass count, and giving up to the "
      "placeholder because that budget ran out is LOGGED, not silent")
print("-" * 70)
# [5] above is the CONTROL that a long loop converges to a clean kept head
# under the real (large) budget. This section proves the budget itself is
# real: starved down to where a loop this size cannot possibly converge,
# the redaction still ends in a well-defined placeholder (never half-cut
# loop text) and says so in the log — the old version fell back to the
# placeholder silently whenever the fixed 64-pass cap was exhausted.
_head6 = " ".join(
    f"Note {j} is ordinary and specific about the weather." for j in range(40)
)
_loop6 = _head6 + " " + ("Stay with me a little longer, please. " * 4000)
assert_true(
    main.reply_is_degenerate(_loop6) is not None,
    "[6] fixture: the loop trips the detector",
)
_saved_budget = main._DEGENERATE_CUT_BUDGET_CHARS
main._DEGENERATE_CUT_BUDGET_CHARS = len(_loop6)  # ~1 pass allowed, nowhere near enough
_h6b = _CaptureLogs()
_lg6b = logging.getLogger("compactor")
_lg6b.addHandler(_h6b)
try:
    _content6, _kept6 = main._degenerate_replacement_content(
        _loop6, main._DEGENERATE_FORWARD_PLACEHOLDER, keep_middle=True
    )
finally:
    _lg6b.removeHandler(_h6b)
    main._DEGENERATE_CUT_BUDGET_CHARS = _saved_budget
assert_true(
    not _kept6 and _content6 == main._DEGENERATE_FORWARD_PLACEHOLDER,
    "[6] *** P8-5: with the pass budget starved to ~1 pass, a loop that "
    "needs many falls back cleanly to the placeholder rather than leaving "
    "a half-cut loop fragment",
)
assert_true(
    _find(_h6b.records, "cut pass"),
    "[6] *** P8-5: giving up to the placeholder because of the pass "
    "budget is LOGGED (used to be silent)",
)

print()
print("ALL TESTS PASSED")
