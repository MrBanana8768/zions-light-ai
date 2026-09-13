"""
The memory-tail gate for a CUT reply (v3.1.4) — decide_memory_tail and
_run_memory_tail, end to end through the real endpoint, on BOTH paths.

The defect: every reply she stopped by hand, and every reply vLLM cut at the
generation ceiling, was discarded from memory whole. 63 exchanges in one
2026-09-01 log window — more than half her recent conversation — never
reached facts, the episodic index or a summary rollup. The old gate tested
COMPLETION; a 27,000-character reply cut at the end is 99% complete prose.
The fix trims a cut reply to its last complete sentence and judges THAT.

Every endpoint case below is asserted on both the streaming and the
non-streaming path. That symmetry is the mechanical guard against a fix
landing at one call site and not its twin, which has happened eighteen
times on this branch. The one asymmetry is explained where it occurs: only
the streaming path can be holed or unfinished.

Mutations this file exists to kill (plan §1.7), and where:

    floor -> 0                              [U] too-short case; [E3]
    floor -> 100000                         [U] trimmed-store case; [E2]
    trims on the clean-finish path too      [E1] byte-identical control
    gates on truncated() only               [E5] the cancelled stream
    applied to one call site only           every [E] case on the other path
    degeneracy judged on raw text           [U] prose-head-plus-decoration
    holed ignored                           [U]; [E6] the dropped chunk
    skip not counted                        [E7]

v3.1.7, R11 — a 43-mutation sweep of decide_memory_tail, _tail_store_blocked,
_run_memory_tail, _async_tail and tailhealth found eight survivors. Six were
real gaps and are closed here:

    floor `<` -> `<=`                       [U] the at-the-floor prefix
    raw_chars measured on stripped text     [U] the whitespace-padded reply
    the user-text rule without .strip()     [E9] the whitespace-only user turn
    _async_tail's episodic gate weakened    [E10b]
    _async_tail's rollup gate weakened      [E10b]
    _async_tail's own degrade.guard removed [E10b]

v3.1.9, M7 — two more of the same, at the third copy of the rule. R8 had
already moved job 2 out into _facts_tail, so the R11 sweep above did not
reach it and [E10b] watched jobs 1 and 3 while running job 2's inputs:

    job 2's user/reply rule without .strip()  [E10c]
    job 2's own degrade.guard absent          [E10c]

Two survived and are recorded rather than chased; see the block at the foot
of this file for both, with the reasons.

Conventions from test_degenerate_skip.py (canonical). Only synthetic
content appears below — the repo is public.

    python test_truncated_tail.py
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
os.environ["COMPACTOR_RAG_ENABLED"] = "false"  # no ChromaDB/fastembed here

_TMP_ROOT = tempfile.mkdtemp(prefix="compactor-test-truncated-tail-")
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

client = TestClient(main.app, client=("127.0.0.1", 12349), raise_server_exceptions=False)


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
# Fixtures. PROSE is 12 ordinary sentences (well over the 300-char floor);
# CUT is PROSE followed by a fragment, the shape of a reply stopped mid-word.
# ---------------------------------------------------------------------------

PROSE = " ".join(
    f"Sentence number {i} of a perfectly ordinary reply about nothing." for i in range(1, 13)
)
FRAGMENT = " And then the reply was cut right in the middle of a wo"
CUT = PROSE + FRAGMENT
assert len(PROSE) > main.MIN_MEMORABLE_TRIMMED_CHARS, "fixture must clear the floor"
assert main.trim_to_last_sentence(CUT) == PROSE, "fixture: the trim lands on PROSE"

RULE = "━"
DEGENERATE = RULE * 400  # the 2026-08-29 incident shape


# ---------------------------------------------------------------------------
# Log capture on main's logger ("compactor"), so the skip line can be
# asserted on. Same shape as test_log_levels.py.
# ---------------------------------------------------------------------------

class _Collector(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)


@contextlib.contextmanager
def capture(logger_name: str = "compactor"):
    lg = logging.getLogger(logger_name)
    handler = _Collector()
    prev_level = lg.level
    lg.addHandler(handler)
    lg.setLevel(logging.DEBUG)
    try:
        yield handler
    finally:
        lg.removeHandler(handler)
        lg.setLevel(prev_level)


def _find(records, needle):
    return next((r for r in records if needle in r.getMessage()), None)


# ---------------------------------------------------------------------------
# [U] decide_memory_tail as a pure function
# ---------------------------------------------------------------------------

print("[U] decide_memory_tail — the policy, in isolation")
D = main.decide_memory_tail

d = D(CUT, finished=True, truncated=False, holed=True)
assert_eq((d.store, d.outcome), (False, "skipped_holed"),
          "holed -> skipped, EVEN on a clean finish with long prose")
d = D("", finished=True, truncated=False, holed=False)
assert_eq((d.store, d.outcome), (False, "skipped_empty"), "empty -> skipped")
d = D("   \n", finished=True, truncated=False, holed=False)
assert_eq(d.outcome, "skipped_empty", "whitespace-only -> skipped as empty")

d = D(CUT, finished=True, truncated=False, holed=False)
assert_eq((d.store, d.outcome), (True, "stored"), "finished -> stored")
assert_eq(d.text, CUT, "...VERBATIM, fragment included: a finished reply is not trimmed")
assert_eq(d.reason, None, "...with nothing to explain")
d = D(DEGENERATE, finished=True, truncated=False, holed=False)
assert_eq((d.store, d.outcome), (False, "skipped_degenerate"),
          "finished but degenerate -> skipped (today's rule, unchanged)")

d = D(CUT, finished=False, truncated=False, holed=False)
assert_eq((d.store, d.outcome), (True, "stored_trimmed"),
          "she hit Stop -> the trimmed prefix is stored")
assert_eq(d.text, PROSE, "...and it is exactly the prose up to the last boundary")
assert_true(d.reason and "stream ended without completion" in d.reason,
            "...the note says what was cut, in the phrase the log filter matches")
assert_eq(d.raw_chars, len(CUT), "...raw length recorded for the retention counter")

d = D(CUT, finished=True, truncated=True, holed=False)
assert_eq((d.store, d.outcome, d.text), (True, "stored_trimmed", PROSE),
          "the generation ceiling takes the SAME path as a manual stop")
assert_true("stream truncated at the generation ceiling" in (d.reason or ""),
            "...with its own phrase for the log")

d = D("just a fragment with no boundary at all", finished=False, truncated=False, holed=False)
assert_eq((d.store, d.outcome), (False, "skipped_no_boundary"),
          "no sentence boundary -> skipped")
d = D("Short. " + "then a long fragment " * 20, finished=False, truncated=False, holed=False)
assert_eq((d.store, d.outcome), (False, "skipped_too_short"),
          "a 6-char surviving prefix is under the floor -> skipped")
assert_true(str(main.MIN_MEMORABLE_TRIMMED_CHARS) in (d.reason or ""),
            "...and the reason names the floor")

# Degeneracy is judged on the TRIMMED text. Raw, this is 60% box-drawing and
# reply_is_degenerate rejects it; trimmed, the decoration tail (unterminated)
# is gone and a clean prose head remains.
head_plus_decoration = PROSE + "\n" + RULE * 600
assert_true(main.reply_is_degenerate(head_plus_decoration),
            "fixture: the RAW text is degenerate")
d = D(head_plus_decoration, finished=False, truncated=False, holed=False)
assert_eq((d.store, d.outcome, d.text), (True, "stored_trimmed", PROSE),
          "prose head + decoration tail -> the head is kept (judged trimmed, not raw)")

# ...and when what SURVIVES the trim is itself the loop, it is skipped.
looped_then_cut = "Intro sentence here. " + RULE * 400 + ". and then a frag"
d = D(looped_then_cut, finished=False, truncated=False, holed=False)
assert_eq((d.store, d.outcome), (False, "skipped_degenerate_partial"),
          "a trimmed prefix that is itself a loop -> skipped")

assert_eq(main.MIN_MEMORABLE_TRIMMED_CHARS, main.DEGENERATE_MIN_CHARS,
          "one floor, not two: the trim floor IS the structural-judgement floor")

# --- the floor's own boundary (v3.1.7, R11 sweep, mutation M08) -------------
#
# The cases above straddle the floor by a wide margin, so `<` and `<=` are
# indistinguishable to them. The floor is a KEEP/DISCARD line on a reply she
# actually read, and the sweep found nothing pinning which side of it the
# equal case falls on. `<` is right: MIN_MEMORABLE_TRIMMED_CHARS is the
# minimum that IS memorable, not the first length that is not.
#
# Built to land exactly on the floor: whole sentences, then the last one
# padded so the trimmed prefix measures the floor to the character.
_floor = main.MIN_MEMORABLE_TRIMMED_CHARS
_unit = "Sentence of an ordinary reply. "
_body = _unit * (_floor // len(_unit))
_pad = _floor - len(_body) - len("A.")
assert _pad >= 0, "fixture: the padding cannot be negative"
_at_floor = _body + "A" + ("a" * _pad) + "."
assert_eq(len(_at_floor), _floor, "fixture: the prefix measures the floor exactly")
d = D(_at_floor + " and then it was cu", finished=False, truncated=False, holed=False)
assert_eq(len(main.trim_to_last_sentence(_at_floor + " and then it was cu")), _floor,
          "fixture: ...and the trim lands on it")
assert_eq((d.store, d.outcome), (True, "stored_trimmed"),
          "a trimmed prefix of EXACTLY the floor is memorable — the floor is a "
          "minimum, not the first rejected length")
d = D(_at_floor[:-2] + "." + " and then it was cu",
      finished=False, truncated=False, holed=False)
assert_eq((d.store, d.outcome), (False, "skipped_too_short"),
          "...and one character under it is not")

# --- raw_chars is what ARRIVED (v3.1.7, R11 sweep, mutation M11) ------------
#
# raw_chars is the denominator of tailhealth's trim_retention and the "N chars
# accumulated" in the skip line an operator greps. It must measure the reply
# as it arrived, not some normalized form of it — every fixture above happens
# to have no surrounding whitespace, so a raw that quietly stripped read the
# same as one that did not. A reply cut mid-word after a newline is ordinary.
_padded = "\n\n" + CUT + "  \n"
d = D(_padded, finished=False, truncated=False, holed=False)
assert_eq(d.raw_chars, len(_padded),
          "raw_chars counts the text as it arrived, whitespace included — it "
          "is the denominator the retention ratio is measured against")
assert_eq(d.text, "\n\n" + PROSE,
          "...while the text STORED is the trimmed prefix with its leading "
          "whitespace intact — the trim cuts at a sentence boundary, it does "
          "not normalize what she was shown")


# ---------------------------------------------------------------------------
# Endpoint harness — both paths, one shape.
# ---------------------------------------------------------------------------

_fired: list = []


def _spy_tail(conv_id, touched_facts, last_user_text, assistant_text, turn_index,
              original_messages, *, injected_facts=None):
    # Stands in for _async_tail. Returns a plain record instead of a
    # coroutine; _spy_fire records it. So `_fired` holds exactly what the
    # memory tail would have been given, byte for byte.
    return {"conv_id": conv_id, "assistant_text": assistant_text}


def _spy_fire(obj, label=None):
    # Returns True: it stands in for a pool that ACCEPTED the work.
    # v3.1.8 gave _fire_and_forget a bool contract (a shed tail is now
    # counted as skipped_shed rather than as a store, F-07), so a double
    # returning None reads as 'the pool shed it' and every clean finish
    # comes back skipped. A test double that does not honour the contract
    # tests the double.
    # TAILS ONLY (v3.1.8): a skipped reply also fires the hierarchy rollup
    # under a `rollup conv=` label, and _fired_text() below reads _fired[0]
    # expecting the tail's own arguments. See test_rollup_on_skip.py.
    if (label or "").startswith("tail"):
        _fired.append(obj)
    else:
        # CLOSED, not merely ignored. Dropping the rollup coroutine on the
        # floor emitted 19 'coroutine was never awaited' RuntimeWarnings,
        # and nothing failed on them: scripts/run-tests.py judges by return
        # code and discards stderr on a pass, so the warning was invisible.
        try:
            obj.close()
        except Exception:
            pass
    return True


class _StubResponse:
    status_code = 200
    text = ""

    def __init__(self, content, finish_reason):
        self._content = content
        self._fr = finish_reason

    def json(self):
        return {
            "id": "stub",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": self._content},
                "finish_reason": self._fr,
            }],
        }


class _StubVLLM:
    reply = ""
    finish_reason = "stop"

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None, **kwargs):
        return _StubResponse(_StubVLLM.reply, _StubVLLM.finish_reason)

    async def aclose(self):
        pass


def _sse(obj) -> bytes:
    return f"data: {json.dumps(obj)}\n\n".encode()


def _content(text) -> bytes:
    return _sse({"choices": [{"delta": {"content": text}}]})


def _finish(reason) -> bytes:
    return _sse({"choices": [{"delta": {}, "finish_reason": reason}]})


DONE = b"data: [DONE]\n\n"


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
    chunks: list = []

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def stream(self, method, url, json=None, **kwargs):
        return _StubStreamCM(list(_StubStreamVLLM.chunks))

    async def aclose(self):
        pass


def _post_nonstream(conv_id, reply, finish_reason="stop"):
    _fired.clear()
    _StubVLLM.reply = reply
    _StubVLLM.finish_reason = finish_reason
    with patch.object(main.httpx, "AsyncClient", _StubVLLM), \
         patch.object(main, "_async_tail", _spy_tail), \
         patch.object(main, "_fire_and_forget", _spy_fire), \
         capture() as cap:
        r = client.post(
            "/v1/chat/completions",
            json={"model": "stub-model",
                  "messages": [{"role": "user", "content": "hi"}],
                  "stream": False},
            headers={"X-Conversation-Id": conv_id},
        )
    return r, cap.records


def _post_stream(conv_id, chunks):
    _fired.clear()
    _StubStreamVLLM.chunks = chunks
    with patch.object(main.httpx, "AsyncClient", _StubStreamVLLM), \
         patch.object(main, "_async_tail", _spy_tail), \
         patch.object(main, "_fire_and_forget", _spy_fire), \
         capture() as cap:
        # TestClient.post() drains the whole stream before returning, so
        # event_stream()'s `finally` — where the tail is decided — has run.
        r = client.post(
            "/v1/chat/completions",
            json={"model": "stub-model",
                  "messages": [{"role": "user", "content": "hi"}],
                  "stream": True},
            headers={"X-Conversation-Id": conv_id},
        )
    return r, cap.records


def _stream_of(text, finish_reason="stop"):
    """A stream that FINISHES: content, finish_reason, [DONE]."""
    return [_content(text), _finish(finish_reason), DONE]


def _fired_text():
    assert_eq(len(_fired), 1, "the memory tail was fired exactly once")
    return _fired[0]["assistant_text"]


# ---------------------------------------------------------------------------
# [E1] control: a FINISHED reply is stored verbatim — fragment and all.
# Kills "trims on the clean-finish path too": CUT ends mid-word, and it must
# arrive at the tail byte-identical.
# ---------------------------------------------------------------------------

print()
print("[E1] control — a finished reply reaches the tail byte-identical, untrimmed")
r, _ = _post_nonstream("tt-control-ns", CUT)
assert_eq(r.status_code, 200, "non-stream: 200")
assert_eq(_fired_text(), CUT, "non-stream: the tail got the whole reply, fragment included")

r, _ = _post_stream("tt-control-s", _stream_of(CUT))
assert_eq(r.status_code, 200, "stream: 200")
assert_eq(_fired_text(), CUT, "stream: the tail got the whole reply, fragment included")

# ---------------------------------------------------------------------------
# [E2] cut at the generation ceiling: the trimmed prefix is stored.
# ---------------------------------------------------------------------------

print()
print("[E2] finish_reason=length — the prose up to the last boundary is memorized")
tailhealth._reset_for_tests()
r, recs = _post_nonstream("tt-length-ns", CUT, finish_reason="length")
assert_eq(r.status_code, 200, "non-stream: 200")
assert_eq(_fired_text(), PROSE, "non-stream: the tail got the TRIMMED reply")
line = _find(recs, "memorizing the")
assert_true(line is not None and line.levelno == logging.INFO,
            "non-stream: an INFO line says what was kept")
assert_true("stream truncated at the generation ceiling" in line.getMessage(),
            "non-stream: ...in the phrase the log filter matches")
assert_eq(tailhealth.snapshot()["outcomes"]["stored_trimmed"], 1,
          "non-stream: counted as a trimmed store")

r, recs = _post_stream("tt-length-s", _stream_of(CUT, finish_reason="length"))
assert_eq(r.status_code, 200, "stream: 200")
assert_eq(_fired_text(), PROSE, "stream: the tail got the TRIMMED reply")
line = _find(recs, "memorizing the")
assert_true(line is not None and "stream truncated at the generation ceiling" in line.getMessage(),
            "stream: the INFO line says what was kept and why")
assert_eq(tailhealth.snapshot()["outcomes"]["stored_trimmed"], 2,
          "stream: counted as a trimmed store")

# ---------------------------------------------------------------------------
# [E3] cut with nothing memorable: no boundary, or under the floor -> skip.
# Kills "floor -> 0" (the too-short case would store).
# ---------------------------------------------------------------------------

print()
print("[E3] a cut reply with no memorable prefix is skipped, and says so")
NO_BOUNDARY = "a reply that was stopped before any sentence could end so there is nothing"
TOO_SHORT = "Short. " + "and then a long unterminated fragment " * 12

for label, reply, outcome in (
    ("no boundary", NO_BOUNDARY, "skipped_no_boundary"),
    ("too short", TOO_SHORT, "skipped_too_short"),
):
    tailhealth._reset_for_tests()
    r, recs = _post_nonstream(f"tt-{outcome}-ns", reply, finish_reason="length")
    assert_eq(r.status_code, 200, f"non-stream/{label}: 200")
    assert_eq(len(_fired), 0, f"non-stream/{label}: the tail was NOT fired")
    line = _find(recs, "skipping memory tail")
    assert_true(line is not None and line.levelno == logging.WARNING,
                f"non-stream/{label}: a WARNING says the tail was skipped")
    snap = tailhealth.snapshot()
    assert_eq(snap["last_skip_outcome"], outcome, f"non-stream/{label}: counted under {outcome}")

    tailhealth._reset_for_tests()
    r, recs = _post_stream(f"tt-{outcome}-s", _stream_of(reply, finish_reason="length"))
    assert_eq(r.status_code, 200, f"stream/{label}: 200")
    assert_eq(len(_fired), 0, f"stream/{label}: the tail was NOT fired")
    line = _find(recs, "skipping memory tail")
    assert_true(line is not None and line.levelno == logging.WARNING,
                f"stream/{label}: a WARNING says the tail was skipped")
    assert_eq(tailhealth.snapshot()["last_skip_outcome"], outcome,
              f"stream/{label}: counted under {outcome}")

# ---------------------------------------------------------------------------
# [E4] a finished degenerate reply is still skipped (test_degenerate_skip.py
# owns this; repeated here so the symmetry table is complete on both paths).
# ---------------------------------------------------------------------------

print()
print("[E4] a finished repetition loop is still skipped on both paths")
r, recs = _post_nonstream("tt-degen-ns", DEGENERATE)
assert_eq(len(_fired), 0, "non-stream: not fired")
assert_true(_find(recs, "repetition loop") is not None, "non-stream: the line names the loop")
r, recs = _post_stream("tt-degen-s", _stream_of(DEGENERATE))
assert_eq(len(_fired), 0, "stream: not fired")
assert_true(_find(recs, "repetition loop") is not None, "stream: the line names the loop")

# ---------------------------------------------------------------------------
# [E5] streaming only: the CANCELLED stream (she hit Stop) — content and then
# nothing, no finish_reason, no [DONE]. Kills "gates on truncated() only".
# The non-streaming path cannot be unfinished: its response is whole by
# construction, which is why its call passes finished=True.
# ---------------------------------------------------------------------------

print()
print("[E5] stream only — she hit Stop: the prose up to the last boundary is memorized")
tailhealth._reset_for_tests()
r, recs = _post_stream("tt-stop-s", [_content(PROSE), _content(FRAGMENT)])
assert_eq(r.status_code, 200, "stream: 200")
assert_eq(_fired_text(), PROSE, "stream: the tail got the TRIMMED reply after a cancel")
line = _find(recs, "memorizing the")
assert_true(line is not None and "stream ended without completion" in line.getMessage(),
            "stream: the INFO line uses the cancel phrase")
r, recs = _post_stream("tt-stop-frag-s", [_content(NO_BOUNDARY)])
assert_eq(len(_fired), 0, "stream: a cancel with no boundary is skipped")
assert_eq(tailhealth.snapshot()["last_skip_outcome"], "skipped_no_boundary",
          "stream: ...and counted")

# ---------------------------------------------------------------------------
# [E6] streaming only: a dropped chunk. The relay yields the chunk to the
# client BEFORE feeding the accumulator, and Starlette encodes a str chunk
# for the wire, so a str here reaches the user fine and is dropped only by
# OUR copy (str has no .decode) — the real shape of the hazard. The stream
# then finishes cleanly; until v3.1.4 that was memorized. Kills "holed
# ignored". The non-streaming path has no accumulator and passes holed=False.
# ---------------------------------------------------------------------------

print()
print("[E6] stream only — a dropped chunk on a clean finish is NOT memorized")
tailhealth._reset_for_tests()
dropped = _content(" the middle of the reply. ").decode()  # a str: our copy drops it
r, recs = _post_stream("tt-holed-s", [_content(PROSE), dropped, _content(PROSE), _finish("stop"), DONE])
assert_eq(r.status_code, 200, "stream: 200 — the user saw the whole reply")
assert_eq(len(_fired), 0, "stream: the tail was NOT fired for text with a hole in it")
assert_eq(tailhealth.snapshot()["last_skip_outcome"], "skipped_holed", "stream: counted as holed")
line = _find(recs, "skipping memory tail")
assert_true(line is not None and "hole" in line.getMessage(), "stream: the WARNING says why")

# ---------------------------------------------------------------------------
# [E7] the counter moved on both paths, with the streak and the char totals.
# ---------------------------------------------------------------------------

print()
print("[E7] tailhealth sees every decision from both paths")
tailhealth._reset_for_tests()
_post_nonstream("tt-count-ns-1", CUT)                        # stored
_post_nonstream("tt-count-ns-2", CUT, finish_reason="length")  # stored_trimmed
_post_nonstream("tt-count-ns-3", NO_BOUNDARY, finish_reason="length")  # skip
_post_stream("tt-count-s-1", _stream_of(CUT))                # stored
_post_stream("tt-count-s-2", [_content(CUT)])                # stored_trimmed (cancel)
_post_stream("tt-count-s-3", [_content(NO_BOUNDARY)])        # skip
snap = tailhealth.snapshot()
assert_eq(snap["stored"], 4, "4 stored across both paths")
assert_eq(snap["skipped"], 2, "2 skipped across both paths")
assert_eq(snap["outcomes"]["stored_trimmed"], 2, "one trimmed store per path")
assert_eq(snap["consecutive_skips"], 1, "the streak counts the latest run only")
assert_eq(snap["trimmed_raw_chars"], 2 * len(CUT), "raw chars of the trimmed stores")
assert_eq(snap["trimmed_kept_chars"], 2 * len(PROSE), "kept chars of the trimmed stores")
assert_eq(snap["trim_retention"], round(len(PROSE) / len(CUT), 3), "retention is measured")
assert_eq(snap["skipped_recently"], True, "and health would see it")

# The skip WARNING carries the streak, so the log line itself says how bad.
tailhealth._reset_for_tests()
for i in range(3):
    _, recs = _post_stream(f"tt-streak-{i}", [_content(NO_BOUNDARY)])
line = _find(recs, "skipping memory tail")
assert_true(line is not None and "3 consecutive memory-tail skip(s)" in line.getMessage(),
            "the third skip's WARNING reports a streak of 3")

# ---------------------------------------------------------------------------
# [E8] The endpoint must FINALIZE the accumulator before it reads it.
#
# v3.1.7 gave SseAccumulator an incremental UTF-8 decoder so a character split
# across two socket reads is reassembled instead of becoming two U+FFFD. That
# fix has a second half: the decoder can be holding the first bytes of a
# character when the stream ends, and only finalize() turns that into the
# `holed` flag. Deleting the finalize() CALL SITE left every accumulator unit
# test green — the fix is in the class, the bug would be in the endpoint that
# forgets to use it. This drives the real endpoint with a stream cut inside a
# multi-byte character and asserts the tail saw a hole.
# ---------------------------------------------------------------------------
print()
print("[E8] a stream cut inside a multi-byte character is holed, end to end")
tailhealth._reset_for_tests()
# "Peace be with you 平安." — split so the last event ends one byte into the
# 3-byte encoding of 平, with no finish_reason and no [DONE]: the shape of a
# backend that died mid-character.
# A COMPLETE event carrying real prose, then a second event whose bytes stop
# one byte into a 3-byte CJK character. The first event is parsed normally, so
# text() holds prose well over the floor; the decoder is left holding a partial
# character, which only finalize() can turn into the `holed` flag. Cutting mid
# JSON instead would end as skipped_empty and prove nothing about the decoder.
# ensure_ascii=False so the CJK character is real UTF-8 on the wire. The
# module's own _sse() escapes non-ASCII to \uXXXX, which would leave nothing
# multi-byte to split and make this test pass against the broken code.
def _content_utf8(text) -> bytes:
    body = json.dumps({"choices": [{"delta": {"content": text}}]},
                      ensure_ascii=False)
    return f"data: {body}\n\n".encode()


_prose = _content_utf8(
    "Peace be with you. This reply is long enough to clear the three hundred "
    "character floor, so that the decision under test is the hole and not the "
    "length. It keeps going for another sentence to be sure of that, because a "
    "reply that trips the floor would be skipped for a reason unrelated to the "
    "decoder and the assertion below would prove nothing at all.")
_tail_bytes = _content_utf8("and then 平安")
_cut_at = _tail_bytes.rfind("平".encode()) + 1   # one byte into the CJK char
assert _cut_at > 0, "fixture: the CJK character must be real UTF-8 on the wire"
_, _recs = _post_stream("tt-split-char", [_prose, _tail_bytes[:_cut_at]])
assert_true(not _fired, "a holed stream fires no tail")
_snap = tailhealth.snapshot()
assert_eq(_snap["outcomes"].get("skipped_holed"), 1,
          "the endpoint finalized the accumulator, so the hole was seen")
assert_true(_find(_recs, "hole in it") is not None
            or _find(_recs, "skipping memory tail") is not None,
            "and it said so in the log")

# Control: the SAME character delivered whole must NOT be holed, or the
# assertion above would pass on any stream at all.
tailhealth._reset_for_tests()
_, _ = _post_stream("tt-split-ok", [_prose, _tail_bytes, _finish("stop"), DONE])
assert_eq(tailhealth.snapshot()["outcomes"].get("skipped_holed"), 0,
          "an intact multi-byte character is not a hole")
assert_true(_fired and "平安" in _fired[-1]["assistant_text"],
            "and the reassembled text reaches the tail intact")

# ---------------------------------------------------------------------------
# [E9] R8 — the counter must not report memory that was never written.
#
# decide_memory_tail says "store", and then _async_tail declines to store
# anything. Until v3.1.7 that decision was taken AFTER tailhealth.note had
# already published `stored`, so /health/full reported a healthy tail for an
# exchange that never reached memory — the silent-skip class this counter was
# built to end, one layer up from where it was closed.
#
# Two conditions, both asserted on both paths:
#   disk pressure       degrade.guard("async memory tail") is False
#   no user text        a parts list with no text and no recognised image
#
# Mutations this section kills:
#   the hoist deleted entirely (count as `stored`, fire the tail) -> all of it
#   hoisted but not counted (return without note())               -> the counts
#   counted under SKIPPED_EMPTY instead of a new label            -> lossy check
# ---------------------------------------------------------------------------

print()
print("[E9] R8 — a tail that will store nothing is counted as a SKIP, not a store")

import degrade  # noqa: E402


def _post_nonstream_no_disk(conv_id, reply):
    with patch.object(degrade, "guard", lambda op: False):
        return _post_nonstream(conv_id, reply)


def _post_stream_no_disk(conv_id, chunks):
    with patch.object(degrade, "guard", lambda op: False):
        return _post_stream(conv_id, chunks)


for label, poster, arg in (
    ("non-stream", _post_nonstream_no_disk, CUT),
    ("stream", _post_stream_no_disk, _stream_of(CUT)),
):
    tailhealth._reset_for_tests()
    r, recs = poster(f"tt-disk-{label}", arg)
    assert_eq(r.status_code, 200, f"{label}/disk: 200 — chat is never gated on this")
    assert_eq(len(_fired), 0, f"{label}/disk: the tail was NOT fired")
    snap = tailhealth.snapshot()
    assert_eq(snap["stored"], 0, f"{label}/disk: NOTHING was counted as stored")
    assert_eq(snap["outcomes"]["skipped_disk_pressure"], 1,
              f"{label}/disk: counted under its own label")
    assert_eq(snap["skipped_recently"], True,
              f"{label}/disk: and it is LOSSY — a reply she read did not reach memory")
    line = _find(recs, "skipping memory tail")
    assert_true(line is not None and line.levelno == logging.WARNING,
                f"{label}/disk: the greppable WARNING names the conversation")
    assert_true("disk pressure" in line.getMessage(),
                f"{label}/disk: ...and says which of the two it was")

# A parts list with neither text nor a recognised image part: it falls through
# _extract_last_user_text AND _memorable_user_text, and _async_tail's
# `if not assistant_text or not last_user_text: return` was a bare return with
# no log line of any kind.
NO_TEXT_PARTS = [{"type": "video_url", "video_url": {"url": "x"}}]
assert_eq(main._extract_last_user_text([{"role": "user", "content": NO_TEXT_PARTS}]), "",
          "fixture: the parts list yields no user text")
assert_eq(main._memorable_user_text([{"role": "user", "content": NO_TEXT_PARTS}], ""), "",
          "fixture: ...and no image marker rescues it")
# The control that keeps this from being an argument against image uploads:
# an image-only turn DOES get a marker, so it never takes this branch.
assert_true(main._memorable_user_text(
    [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "u"}}]}], ""
).startswith("[shared 1 image"),
    "fixture: an image-only upload is rescued by the marker, NOT skipped")


def _post_nonstream_content(conv_id, reply, content):
    """_post_nonstream with an arbitrary user content payload."""
    _fired.clear()
    _StubVLLM.reply = reply
    _StubVLLM.finish_reason = "stop"
    with patch.object(main.httpx, "AsyncClient", _StubVLLM), \
         patch.object(main, "_async_tail", _spy_tail), \
         patch.object(main, "_fire_and_forget", _spy_fire), \
         capture() as cap:
        r = client.post(
            "/v1/chat/completions",
            json={"model": "stub-model",
                  "messages": [{"role": "user", "content": content}],
                  "stream": False},
            headers={"X-Conversation-Id": conv_id},
        )
    return r, cap.records


# v3.1.7 (R11 sweep, mutation M15). The guard is `.strip()`, not truthiness,
# and nothing pinned the difference: a user turn of nothing but whitespace is
# TRUE, so a guard that tested the bare string would let the tail fire on it.
# _async_tail would then index "[user]:   \n[assistant]: ..." as a real
# exchange — retrievable, injectable, and rebuilt as a real turn by
# /admin/compact — while tailhealth published `stored`. That is the silent
# skip R8 exists to end, reached through one missing method call.
tailhealth._reset_for_tests()
r, recs = _post_nonstream_content("tt-whitespace-user", CUT, "   \n\t  ")
assert_eq(r.status_code, 200, "whitespace-only user text: 200")
assert_eq(len(_fired), 0,
          "whitespace-only user text: the tail was NOT fired — a turn of "
          "spaces is nothing to pair a reply with")
snap = tailhealth.snapshot()
assert_eq(snap["stored"], 0,
          "whitespace-only user text: NOTHING was counted as stored")
assert_eq(snap["outcomes"]["skipped_no_user_text"], 1,
          "whitespace-only user text: counted under the same label as no text "
          "at all, because it is the same thing to the store")
assert_eq(snap["skipped_recently"], True,
          "whitespace-only user text: and it is LOSSY — she read the reply")

tailhealth._reset_for_tests()
r, recs = _post_nonstream_content("tt-nousertext", CUT, NO_TEXT_PARTS)
assert_eq(r.status_code, 200, "no-user-text: 200")
assert_eq(len(_fired), 0, "no-user-text: the tail was NOT fired")
snap = tailhealth.snapshot()
assert_eq(snap["stored"], 0, "no-user-text: NOTHING was counted as stored")
assert_eq(snap["outcomes"]["skipped_no_user_text"], 1,
          "no-user-text: counted under its own label")
assert_eq(snap["skipped_recently"], True, "no-user-text: and it is LOSSY")
assert_true(_find(recs, "skipping memory tail") is not None,
            "no-user-text: a line naming the conversation exists AT ALL")

# The ledger test_saturation.py asserts must still reconcile with the two new
# labels in play: every decision is exactly one of stored or skipped.
snap = tailhealth.snapshot()
assert_eq(sum(snap["outcomes"].values()), snap["stored"] + snap["skipped"],
          "the outcome tally still reconciles with stored + skipped")

# And the control: with the disk healthy and real user text, the same reply
# still stores. Without this, [E9] passes for an endpoint that never stores.
tailhealth._reset_for_tests()
r, _ = _post_nonstream("tt-disk-control", CUT)
assert_eq(_fired_text(), CUT, "control: a healthy disk and real user text still store")
assert_eq(tailhealth.snapshot()["stored"], 1, "control: counted as a store")

# ---------------------------------------------------------------------------
# [E10] R8, second half — job 2 must not be able to cancel job 3.
#
# _async_tail's docstring calls its three jobs independent, and job 2 (facts)
# owned two early `return`s that returned from the WHOLE tail. So a deployment
# running with COMPACTOR_FACTS_EXTRACTION=false had the hierarchical summary
# rollup silently off for its entire life, with nothing in the log and nothing
# in the docstring claiming the dependency existed.
#
# Mutation: put the `return` back in _facts_tail's caller (i.e. inline the
# function again) and this dies.
# ---------------------------------------------------------------------------

print()
print("[E10] R8 — extraction disabled must not switch off the summary rollup")

import asyncio      # noqa: E402
import facts        # noqa: E402
import summarizer   # noqa: E402

_ROLLUPS: list = []
_INDEXED: list = []


async def _spy_rollup(conv_id, messages, vllm_url, model, **_kw):
    _ROLLUPS.append(conv_id)
    return {"l1": [], "l2": [], "l3": None, "last_summarized_turn": 0}


def _run_tail(conv_id, *, extraction, user_text="a real question"):
    _ROLLUPS.clear()
    _INDEXED.clear()
    with patch.object(facts, "extraction_enabled", lambda: extraction), \
         patch.object(facts, "load_facts", lambda c: []), \
         patch.object(summarizer, "enabled", lambda: True), \
         patch.object(summarizer, "maybe_rollup", _spy_rollup), \
         patch.object(summarizer, "load_state",
                      lambda c: {"l1": [], "l2": [], "l3": None,
                                 "last_summarized_turn": 0}), \
         patch.object(retrieval, "index_exchange",
                      lambda *a, **k: (_INDEXED.append(a), True)[1]):
        asyncio.run(main._async_tail(
            conv_id, [], user_text, PROSE, 4,
            [{"role": "user", "content": user_text}],
        ))


_run_tail("tt-noextract", extraction=False)
assert_eq(len(_INDEXED), 1, "extraction off: job 1 (episodic) still ran")
assert_eq(len(_ROLLUPS), 1,
          "extraction off: job 3 (summary rollup) ran too — it is not job 2's "
          "to cancel")

_run_tail("tt-extract-on", extraction=True)
assert_eq(len(_ROLLUPS), 1, "control: extraction on, the rollup still runs")

# ---------------------------------------------------------------------------
# [E10b] R11 sweep — _async_tail's OWN guards, called directly.
#
# Since R8 the request path refuses to fire the tail for an empty reply, a
# blank user turn, or a full disk, so _async_tail's three inner guards are
# unreachable through the endpoint — which is why the sweep's M21, M22 and M24
# all survived every endpoint test in this file. Unreachable from the endpoint
# is not unreachable: five other suites call _async_tail directly, and R8's
# own docstring gives the reason the inner half must stay, in the disk case
# exactly — "a tail can sit in the pool's queue while the disk fills under
# it". The outer check is made on the request path; the disk can be full by
# the time the coroutine runs. The same is true of the pool re-ordering work
# behind a /forget.
#
# Mutations this section kills:
#   `if assistant_text and last_user_text:` -> `if assistant_text:`      (M21)
#   `if summarizer.enabled() and assistant_text:` -> drop the conjunct   (M22)
#   the `degrade.guard` at the top of _async_tail -> `if False:`         (M24)
# ---------------------------------------------------------------------------

print()
print("[E10b] R11 — _async_tail's own guards hold when it is entered directly")

_run_tail("tt-inner-blank-user", extraction=True, user_text="   ")
assert_eq(len(_INDEXED), 0,
          "blank user text: no episodic row — an exchange with nothing on the "
          "user side is not an exchange, however it got here")

_ROLLUPS.clear()
_INDEXED.clear()
with patch.object(facts, "extraction_enabled", lambda: True), \
     patch.object(facts, "load_facts", lambda c: []), \
     patch.object(summarizer, "enabled", lambda: True), \
     patch.object(summarizer, "maybe_rollup", _spy_rollup), \
     patch.object(summarizer, "load_state",
                  lambda c: {"l1": [], "l2": [], "l3": None,
                             "last_summarized_turn": 0}), \
     patch.object(retrieval, "index_exchange",
                  lambda *a, **k: (_INDEXED.append(a), True)[1]):
    asyncio.run(main._async_tail(
        "tt-inner-empty-reply", [], "a real question", "", 4,
        [{"role": "user", "content": "a real question"}],
    ))
assert_eq(len(_INDEXED), 0, "empty reply: no episodic row")
assert_eq(len(_ROLLUPS), 0,
          "empty reply: and NO rollup — rolling up a turn with no assistant "
          "text advances the watermark over a turn that says nothing")

# The same for a reply of nothing but whitespace, which is TRUE and therefore
# passed both inner gates until the shared rule landed. decide_memory_tail
# calls this SKIPPED_EMPTY on the request path; the two gates here must agree
# with it rather than with `bool("   ")`.
_ROLLUPS.clear()
_INDEXED.clear()
with patch.object(facts, "extraction_enabled", lambda: True), \
     patch.object(facts, "load_facts", lambda c: []), \
     patch.object(summarizer, "enabled", lambda: True), \
     patch.object(summarizer, "maybe_rollup", _spy_rollup), \
     patch.object(summarizer, "load_state",
                  lambda c: {"l1": [], "l2": [], "l3": None,
                             "last_summarized_turn": 0}), \
     patch.object(retrieval, "index_exchange",
                  lambda *a, **k: (_INDEXED.append(a), True)[1]):
    asyncio.run(main._async_tail(
        "tt-inner-blank-reply", [], "a real question", "  \n\t ", 4,
        [{"role": "user", "content": "a real question"}],
    ))
assert_eq(main.decide_memory_tail("  \n\t ", finished=True, truncated=False,
                                  holed=False).outcome, "skipped_empty",
          "fixture: the request path calls a whitespace reply empty")
assert_eq(len(_INDEXED), 0,
          "whitespace reply: no episodic row — the inner gate agrees with "
          "decide_memory_tail, not with bool('   ')")
assert_eq(len(_ROLLUPS), 0, "whitespace reply: and no rollup")

_ROLLUPS.clear()
_INDEXED.clear()
with patch.object(degrade, "guard", lambda op: False), \
     patch.object(facts, "extraction_enabled", lambda: True), \
     patch.object(facts, "load_facts", lambda c: []), \
     patch.object(summarizer, "enabled", lambda: True), \
     patch.object(summarizer, "maybe_rollup", _spy_rollup), \
     patch.object(summarizer, "load_state",
                  lambda c: {"l1": [], "l2": [], "l3": None,
                             "last_summarized_turn": 0}), \
     patch.object(retrieval, "index_exchange",
                  lambda *a, **k: (_INDEXED.append(a), True)[1]), \
     capture() as cap:
    asyncio.run(main._async_tail(
        "tt-inner-disk", [], "a real question", PROSE, 4,
        [{"role": "user", "content": "a real question"}],
    ))
assert_eq(len(_INDEXED), 0,
          "disk pressure at RUN time: no episodic row — the request path's "
          "check was made before this coroutine was queued")
assert_eq(len(_ROLLUPS), 0, "disk pressure at run time: and no rollup")
assert_true(_find(cap.records, "disk pressure") is not None,
            "...and it says why, rather than returning in silence")

# The control, without which the three assertions above pass for an
# _async_tail that does nothing at all.
_run_tail("tt-inner-control", extraction=True)
assert_eq(len(_INDEXED), 1, "control: a healthy tail still indexes")
assert_eq(len(_ROLLUPS), 1, "control: ...and still rolls up")

# ---------------------------------------------------------------------------
# [E10c] M7 — job 2's OWN guards, which [E10b] structurally could not see.
#
# [E10b] above runs a blank user turn, a whitespace reply and a full disk
# through _async_tail and asserts on _INDEXED and _ROLLUPS: jobs 1 and 3. It
# has no spy on job 2 at all. So every one of those three cases passed while
# _facts_tail accepted all three of them — the shapes were exercised, the
# conclusions were only ever read off the other two jobs. A case that runs
# the input and looks somewhere else is not coverage, and this is the fourth
# time on this branch that it has read as coverage.
#
# What job 2 was carrying:
#
#   * bare truthiness, `if not assistant_text or not last_user_text`. The
#     R11 sweep unified that rule into _has_pairable_user_text and reached
#     _tail_store_blocked and the episodic gate; it did not reach this copy,
#     because R8 had moved the condition into a function of its own two
#     releases earlier and a moved condition is not what a sweep greps for.
#     A user turn of nothing but spaces is TRUE, so job 1 refused it and job
#     2 spent a vLLM extraction call on a blank question — against a prompt
#     that says "when in doubt, extract" — and stored the answer.
#
#   * no degrade.guard of its own. Job 3 took one when it was extracted in
#     v3.1.8, with the reason written next to it: the outer check happens
#     once at the top of _async_tail, and job 1 indexes and job 2 makes an
#     LLM round trip before job 2's writes land, so the disk can fill in
#     between. Job 2 was extracted a release earlier and kept nothing.
#     degrade.py's module docstring lists "fact extraction (async tail)"
#     FIRST under what gets gated, which is exactly what made it read as
#     covered.
#
# The guard patches below are LABEL-SELECTIVE on purpose. A blanket
# `guard -> False` returns at the top of _async_tail and job 2 never runs,
# so the assertion would hold for a _facts_tail with no guard whatsoever —
# passing for the wrong reason. Blocking only job 2's own label is the disk-
# fills-mid-tail case, and it fails if the second call site is gone.
#
# Mutations this section kills:
#   drop `if not degrade.guard("fact extraction tail")`                 (M7a)
#   `not (assistant_text or "").strip()` -> `not assistant_text`        (M7b)
#   `not _has_pairable_user_text(...)` -> `not last_user_text`          (M7c)
# ---------------------------------------------------------------------------

print()
print("[E10c] M7 — _facts_tail's own guards, watched at job 2 rather than 1")

_EXTRACTED: list = []
_SAVED: list = []
_STORED_FACT = {"text": "an existing fact", "last_used": 1}


async def _spy_extract(*a, **k):
    _EXTRACTED.append(k.get("conv_id"))
    return []


def _run_job2(conv_id, *, extraction=True, user_text="a real question",
              reply=PROSE, blocked_label=None):
    """Enter _async_tail directly and watch what JOB 2 did.

    `blocked_label` is the single degrade.guard label to refuse; everything
    else is allowed. See the header for why a blanket refusal would make
    these assertions vacuous.
    """
    _EXTRACTED.clear()
    _SAVED.clear()
    _ROLLUPS.clear()
    _INDEXED.clear()
    touched = [{**_STORED_FACT, "last_used": 99}]
    with patch.object(degrade, "guard", lambda op: op != blocked_label), \
         patch.object(facts, "extraction_enabled", lambda: extraction), \
         patch.object(facts, "load_facts", lambda c: [dict(_STORED_FACT)]), \
         patch.object(facts, "save_facts",
                      lambda c, f: _SAVED.append((c, list(f)))), \
         patch.object(facts, "extract_facts_from_exchange", _spy_extract), \
         patch.object(summarizer, "enabled", lambda: True), \
         patch.object(summarizer, "maybe_rollup", _spy_rollup), \
         patch.object(summarizer, "load_state",
                      lambda c: {"l1": [], "l2": [], "l3": None,
                                 "last_summarized_turn": 0}), \
         patch.object(retrieval, "index_exchange",
                      lambda *a, **k: (_INDEXED.append(a), True)[1]):
        asyncio.run(main._async_tail(
            conv_id, touched, user_text, reply, 4,
            [{"role": "user", "content": user_text}],
        ))


# CONTROL FIRST, so every refusal below is measured against a job 2 that is
# demonstrably alive in this harness. Without it the four cases that follow
# pass just as well for a _facts_tail that never extracts anything.
_run_job2("tt-job2-control")
assert_eq(len(_EXTRACTED), 1, "control: a healthy job 2 makes its extraction call")
assert_eq(len(_SAVED), 1, "control: ...and writes the store")
assert_eq(len(_INDEXED), 1, "control: job 1 ran too")
assert_eq(len(_ROLLUPS), 1, "control: and job 3")

_run_job2("tt-job2-blank-user", user_text="   ")
assert_eq(len(_EXTRACTED), 0,
          "blank user turn: NO extraction call — job 1 already refuses this "
          "shape, and job 2 must not pay an LLM round trip to invent facts "
          "out of a question of three spaces")
assert_eq(len(_SAVED), 0, "blank user turn: and nothing written")
assert_eq(len(_INDEXED), 0, "blank user turn: job 1 still refuses it (E10b)")

_run_job2("tt-job2-blank-reply", reply="  \n\t ")
assert_eq(len(_EXTRACTED), 0,
          "whitespace reply: NO extraction call — decide_memory_tail calls "
          "this SKIPPED_EMPTY on the request path and job 2 must agree with "
          "it rather than with bool('   ')")
assert_eq(len(_SAVED), 0, "whitespace reply: and nothing written")

_run_job2("tt-job2-disk", blocked_label="fact extraction tail")
assert_eq(len(_EXTRACTED), 0,
          "disk fills mid-tail: NO extraction call — job 2 re-checks the "
          "pause for itself, exactly as job 3 does, because the outer check "
          "was made before job 1 and an LLM round trip")
assert_eq(len(_SAVED), 0, "disk fills mid-tail: and no fact write")
assert_eq(len(_INDEXED), 1,
          "disk fills mid-tail: job 1 still ran — this is the selective case, "
          "not _async_tail's outer guard firing and hiding the point")
assert_eq(len(_ROLLUPS), 1, "disk fills mid-tail: and job 3 still ran")

# The extraction-OFF branch writes too — the touched-save that keeps LRU
# tracking across restarts is a save_facts like any other, and it sits ABOVE
# the text check, so it is reached on turns nothing else is.
_run_job2("tt-job2-off-disk", extraction=False,
          blocked_label="fact extraction tail")
assert_eq(len(_SAVED), 0,
          "extraction off + disk pressure: not even the touched-save runs — "
          "it is a write, and the guard is above it")

_run_job2("tt-job2-off-control", extraction=False)
assert_eq(len(_SAVED), 1,
          "control: extraction off on a healthy disk still persists the LRU "
          "touch, so the line above is the guard and not the branch")
assert_eq(len(_EXTRACTED), 0,
          "control: ...and still makes no extraction call, obviously")

# The label-selective case above proves job 2 ASKS — that a second
# degrade.guard call exists in this coroutine with its own label. It does not
# prove a state the system can be in: the label is a log string, so the real
# guard() cannot answer differently for two labels at one instant. A hostile
# review made that point the day [E10c] was written and it is correct.
#
# This is the reachable version. degrade.guard() calls writes_allowed(), whose
# reading is cached for COMPACTOR_DEGRADE_CHECK_TTL_S (10 s) — so the two
# checks differ only when more than the TTL passes between them, which is a
# tail re-queued behind a bgwork.pool backlog or a slow extraction round trip.
# Patch writes_allowed, NOT guard, and let the shipped guard() run: first call
# allowed (the outer check), every call after it blocked (the disk filled and
# the TTL expired).
_WA_CALLS: list = []


def _flip_writes_allowed():
    _WA_CALLS.append(1)
    return (len(_WA_CALLS) == 1, 10.0)


def _run_job2_flip(conv_id, *, always=False):
    _EXTRACTED.clear()
    _SAVED.clear()
    _ROLLUPS.clear()
    _INDEXED.clear()
    _WA_CALLS.clear()
    degrade._reset_cache_for_tests()
    _wa = (lambda: (True, 10.0)) if always else _flip_writes_allowed
    with patch.object(degrade, "writes_allowed", _wa), \
         patch.object(facts, "extraction_enabled", lambda: True), \
         patch.object(facts, "load_facts", lambda c: [dict(_STORED_FACT)]), \
         patch.object(facts, "save_facts",
                      lambda c, f: _SAVED.append((c, list(f)))), \
         patch.object(facts, "extract_facts_from_exchange", _spy_extract), \
         patch.object(summarizer, "enabled", lambda: True), \
         patch.object(summarizer, "maybe_rollup", _spy_rollup), \
         patch.object(summarizer, "load_state",
                      lambda c: {"l1": [], "l2": [], "l3": None,
                                 "last_summarized_turn": 0}), \
         patch.object(retrieval, "index_exchange",
                      lambda *a, **k: (_INDEXED.append(a), True)[1]):
        asyncio.run(main._async_tail(
            conv_id, [{**_STORED_FACT, "last_used": 99}],
            "a real question", PROSE, 4,
            [{"role": "user", "content": "a real question"}],
        ))
    degrade._reset_cache_for_tests()


# CONTROL FIRST again: the same harness with the disk healthy throughout.
_run_job2_flip("tt-job2-flip-control", always=True)
assert_eq(len(_EXTRACTED), 1,
          "control: through the REAL degrade.guard, a healthy disk still "
          "extracts")
assert_eq(len(_ROLLUPS), 1, "control: and job 3 still rolls up")

_run_job2_flip("tt-job2-flip")
assert_eq(len(_INDEXED), 1,
          "the TTL expired and the disk filled: job 1 ran, so the OUTER guard "
          "passed and we are genuinely past it")
assert_true(len(_WA_CALLS) >= 2,
            "...and something asked writes_allowed() a second time after it "
            "(got %d call(s)) — measured through the shipped guard(), not a "
            "patched one" % len(_WA_CALLS))
assert_eq(len(_EXTRACTED), 0,
          "...and job 2 refused: no extraction call once the second reading "
          "came back blocked")
assert_eq(len(_SAVED), 0, "...and wrote nothing")
assert_eq(len(_ROLLUPS), 0,
          "...and job 3 refused too, which is correct — by then the disk "
          "really is full, and this is what makes the case reachable rather "
          "than a per-label fiction")

# ---------------------------------------------------------------------------
# [E11] R26 — vLLM dying mid-stream must not skip the tail silently.
#
# `vllm_failed` is set when the backend drops the connection PART WAY THROUGH a
# reply she has already read. The accumulator holds real prose, and the old
# guard `if conv_id and not vllm_failed` meant decide_memory_tail was never
# called, tailhealth.note was never called, and no line containing "memory
# tail" was emitted. Measured before the fix: 0 decisions, an unchanged
# snapshot, an empty grep, while the client got the prose.
#
# A connection that dies mid-reply IS a cut reply, so this takes the same trim
# path as a manual Stop.
#
# Mutations this section kills:
#   `if conv_id:` -> `if conv_id and not vllm_failed:`   -> the whole section
#   the tail fired but on the error text                 -> [E11b]
# ---------------------------------------------------------------------------

print()
print("[E11] R26 — a stream vLLM killed mid-reply is memorized, not silently dropped")


class _DyingStreamResp:
    status_code = 200

    def __init__(self, chunks, status=200):
        self._chunks = chunks
        self.status_code = status

    async def aread(self):
        return b'{"error": {"message": "context length exceeded"}}'

    async def aiter_raw(self):
        for c in self._chunks:
            yield c
        raise main.httpx.ReadError("connection reset by peer")


class _DyingCM:
    def __init__(self, chunks, status=200):
        self._resp = _DyingStreamResp(chunks, status)

    async def __aenter__(self):
        return self._resp

    async def __aexit__(self, *exc):
        return False


class _DyingVLLM:
    chunks: list = []
    status: int = 200

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def stream(self, method, url, json=None, **kwargs):
        return _DyingCM(list(_DyingVLLM.chunks), _DyingVLLM.status)

    async def aclose(self):
        pass


def _post_dying(conv_id, chunks, status=200):
    _fired.clear()
    _DyingVLLM.chunks = chunks
    _DyingVLLM.status = status
    with patch.object(main.httpx, "AsyncClient", _DyingVLLM), \
         patch.object(main, "_async_tail", _spy_tail), \
         patch.object(main, "_fire_and_forget", _spy_fire), \
         capture() as cap:
        r = client.post(
            "/v1/chat/completions",
            json={"model": "stub-model",
                  "messages": [{"role": "user", "content": "hi"}],
                  "stream": True},
            headers={"X-Conversation-Id": conv_id},
        )
    return r, cap.records


tailhealth._reset_for_tests()
r, recs = _post_dying("tt-midstream-death", [_content(CUT)])
assert_eq(r.status_code, 200, "the client still gets its 200")
assert_true("Sentence number 12" in r.content.decode("utf-8", "replace"),
            "...and the prose it had already been sent")
assert_eq(_fired_text(), PROSE,
          "the tail was fired with the prose up to the last sentence boundary")
snap = tailhealth.snapshot()
assert_eq(snap["outcomes"]["stored_trimmed"], 1,
          "and the decision was COUNTED — the accumulator held a cut reply")
assert_true(_find(recs, "the backend failed during this stream") is not None,
            "a WARNING names the conversation whose backend died")

# The same death with nothing memorable must still be COUNTED and greppable —
# the half of R26 that a store alone would not prove.
tailhealth._reset_for_tests()
r, recs = _post_dying("tt-midstream-nothing", [_content(NO_BOUNDARY)])
assert_eq(len(_fired), 0, "nothing memorable: no tail")
assert_eq(tailhealth.snapshot()["last_skip_outcome"], "skipped_no_boundary",
          "nothing memorable: but the skip is counted")
assert_true(_find(recs, "skipping memory tail") is not None,
            "nothing memorable: and greppable, naming the conversation")

print()
print("[E11b] R26 — the compactor's own apology must never become a memory")
# vLLM REJECTS the request (4xx). The friendly error chunks are yielded
# straight to the client and never fed to the accumulator, so the tail sees ""
# and decides SKIPPED_EMPTY — the one outcome tailhealth treats as lossless,
# because nothing was ever generated to lose. If the error chunks were ever
# accumulated, this would come back as a stored reply made of an apology.
tailhealth._reset_for_tests()
r, recs = _post_dying("tt-rejected", [], status=400)
assert_eq(len(_fired), 0, "a rejected request stores nothing")
snap = tailhealth.snapshot()
assert_eq(snap["stored"], 0, "...and counts no store")
assert_eq(snap["outcomes"]["skipped_empty"], 1,
          "...counted as empty: there was never any reply")
assert_eq(snap["skipped_recently"], False,
          "...and NOT as a loss: a request the backend refused lost nothing")

print()
print("All truncated-tail tests passed.")

# ---------------------------------------------------------------------------
# Mutation record — v3.1.7 R11 sweep.
#
# 43 mutations across compactor/main.py (decide_memory_tail,
# _has_pairable_user_text, _tail_store_blocked, _run_memory_tail, _async_tail)
# and compactor/tailhealth.py (_window_s, _safe_int, note, snapshot,
# LOSSY_SKIP_OUTCOMES, STORING_OUTCOMES), each applied alone and run against
# test_truncated_tail.py, test_tailhealth.py, test_degenerate_skip.py and
# test_memory.py. 41 killed. The six that were not, before this pass, are the
# six now listed in the header docstring.
#
# TWO SURVIVED. Both are recorded here rather than chased, because the reason
# is the finding:
#
#   `kept_chars=len(decision.text) if decision.store else 0`
#     -> `kept_chars=len(decision.text)`
#   Every skip is built by decide_memory_tail._skip, which is
#   TailDecision(False, "", ...) — `text` is "" on every skip path there is,
#   so len() is already 0 and the conditional cannot change a published
#   number. It guards a decision shape that does not yet exist (a skip that
#   carries text). Pinning a value that is structurally zero would assert
#   the constructor, not the behaviour.
#
#   `since_lossy <= window` -> `since_lossy < window`
#   The boundary is unreachable through the public API. `since_lossy` is
#   round(monotonic - last_lossy_skip_at, 1) and `window` is refused by both
#   _window_s and snapshot's own guard unless it is finite and > 0; landing
#   exactly on it needs a window equal to a rounded monotonic delta the
#   caller does not control, and freezing the clock to get one gives
#   since_lossy == 0.0, which needs the one window value both guards reject.
#   The error direction is also the safe one: `<` clears the degrade signal
#   an instant early rather than pinning it on, which is the failure the
#   window exists to prevent.
#
# A test whose assertions cannot be made to fail is a test that asserts
# nothing — and a mutation that changes no observable behaviour is not one
# either. Both are worth writing down; only the first is worth fixing.
# ---------------------------------------------------------------------------
