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
    _fired.append(obj)


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

print()
print("All truncated-tail tests passed.")
