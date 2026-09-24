"""
v3.1.9.4 B4 (P15-1): vLLM 0.19 streams a reply, then on a request-level
internal failure yields ONE `data: {"error": {...}}` event and `data:
[DONE]` — no "choices" key on the error event, so it was silently
ignored, and [DONE] then read as "the model actually FINISHED the reply"
(SseAccumulator.complete()'s own docstring). A cut reply was memorized
VERBATIM, mid-word fragment included, counted `stored`, with no log line.

This file mirrors the reviewer's wire shape (SP\\p15\\p15_c2_stream_error.py)
at the SseAccumulator/decide_memory_tail layer (the two pieces this lane
owns), plus the alternate finish_reason="error" shape the fix also
covers, plus a CONTROL proving an ordinary finished stream is unaffected.
Synthetic text only.

Run: python test_v3194_bg_b4.py
"""
import json
import logging
import os
import sys

os.environ.setdefault("MODEL_REPO", "")

import main  # noqa: E402

FAILED = []


def check(cond, label):
    print(("  ok   " if cond else "FAIL ") + label, flush=True)
    if not cond:
        FAILED.append(label)


def _sse(obj) -> bytes:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8")


def _content(text: str) -> bytes:
    return _sse({"choices": [{"delta": {"content": text}}]})


def _finish(reason: str) -> bytes:
    return _sse({"choices": [{"delta": {}, "finish_reason": reason}]})


DONE = b"data: [DONE]\n\n"

# vLLM 0.19's own shape (read from the shipped image, per SP\p15-findings.md
# P15-1): a top-level "error" object, no "choices" key at all.
VLLM_ERROR_EVENT = _sse({
    "error": {"message": "Internal server error", "type": "InternalServerError",
              "param": None, "code": 500}
})

PROSE = " ".join(f"Sentence number {i} of a perfectly ordinary reply about nothing." for i in range(1, 13))
CUT = PROSE + " And then the engine failed right in the middle of a wo"


def _fresh(*chunks) -> "main.SseAccumulator":
    acc = main.SseAccumulator()
    for c in chunks:
        acc.feed(c)
    acc.finalize()
    return acc


class _Collector(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.recs = []

    def emit(self, record):
        self.recs.append(record)


def _with_log_capture(fn):
    lg = logging.getLogger("compactor")
    h = _Collector()
    lg.addHandler(h)
    lg.setLevel(logging.DEBUG)
    try:
        result = fn()
    finally:
        lg.removeHandler(h)
    return result, [r.getMessage() for r in h.recs]


# ---------------------------------------------------------------------------
# 1. The reviewer's exact wire shape: content, in-band error, [DONE]
# ---------------------------------------------------------------------------

def test_defect_shape_content_error_done():
    print("\n[test] content + in-band error + [DONE]: THE DEFECT SHAPE")
    acc, msgs = _with_log_capture(
        lambda: _fresh(_content(CUT), VLLM_ERROR_EVENT, DONE)
    )
    check(acc.text() == CUT, "the full partial text is still retrievable from text()")
    check(acc.complete() is False,
          "THE FIX: complete() is False — the [DONE] after the error does not mean finished")
    check(acc.errored() is True, "errored() reports the in-band failure")
    check(acc.truncated() is False, "not the finish_reason=length path — a different failure, own flag")
    check(acc.holed() is False, "not a decode/parse hole — the text is intact, just not FINISHED")
    warn_lines = [m for m in msgs if "error" in m.lower() and "stream" in m.lower()]
    check(bool(warn_lines),
          f"THE FIX: a WARNING line names the in-band stream error (got: {warn_lines})")

    decision = main.decide_memory_tail(
        acc.text(), finished=acc.complete(), truncated=acc.truncated(), holed=acc.holed()
    )
    check(decision.outcome == main.tailhealth.STORED_TRIMMED,
          f"decide_memory_tail takes the CUT path (outcome={decision.outcome}), "
          f"not verbatim STORED")
    check(decision.text == main.trim_to_last_sentence(CUT),
          "stored text is trimmed to the last complete sentence, mid-word fragment dropped")
    check(decision.text != CUT, "critically: NOT the raw mid-word fragment verbatim")


def test_error_finish_reason_shape_is_also_logged():
    print("\n[test] the finish_reason=='error' shape ALSO produces a WARNING line")
    _acc, msgs = _with_log_capture(lambda: _fresh(_content("partial "), _finish("error"), DONE))
    warn_lines = [m for m in msgs if "error" in m.lower() and "stream" in m.lower()]
    check(bool(warn_lines), f"a WARNING line names this shape too (got: {warn_lines})")


def test_control_no_done_is_trimmed_as_before():
    print("\n[test] CONTROL: the SAME partial reply, connection simply ends "
          "(no [DONE], no error) — unaffected by this fix, still the "
          "existing cut-and-trim path")
    acc = _fresh(_content(CUT))  # connection just stops; no [DONE], no error
    check(acc.complete() is False, "CONTROL: not complete, exactly as before this fix")
    check(acc.errored() is False, "CONTROL: not errored — this is the OTHER cut reason")
    decision = main.decide_memory_tail(
        acc.text(), finished=acc.complete(), truncated=acc.truncated(), holed=acc.holed()
    )
    check(decision.outcome == main.tailhealth.STORED_TRIMMED, "CONTROL: same CUT path as the error case")
    check(decision.text == main.trim_to_last_sentence(CUT), "CONTROL: same trimmed result")


def test_control_a_clean_finish_is_still_stored_verbatim():
    print("\n[test] CONTROL: an ordinary, error-free finish is completely "
          "unaffected — still complete, still stored verbatim")
    acc = _fresh(_content(PROSE), _finish("stop"), DONE)
    check(acc.complete() is True, "CONTROL: a real finish is still complete()")
    check(acc.errored() is False, "CONTROL: no error seen")
    check(acc.truncated() is False, "CONTROL: finish_reason=stop, not length")
    check(acc.usable() is True, "CONTROL: usable() still true for a genuine clean finish")
    decision = main.decide_memory_tail(
        acc.text(), finished=acc.complete(), truncated=acc.truncated(), holed=acc.holed()
    )
    check(decision.outcome == main.tailhealth.STORED, "CONTROL: verbatim STORED, not trimmed")
    check(decision.text == PROSE, "CONTROL: exact text, untrimmed")


# ---------------------------------------------------------------------------
# 2. The alternate shape: finish_reason == "error" on a choice
# ---------------------------------------------------------------------------

def test_finish_reason_error_shape():
    print("\n[test] the OTHER error shape: a choice with finish_reason=='error'")
    acc = _fresh(_content(CUT), _finish("error"), DONE)
    check(acc.errored() is True, "errored() catches the finish_reason=='error' shape too")
    check(acc.complete() is False, "complete() is False for this shape as well")
    check(acc.truncated() is False, "not the length/truncated path — a different flag entirely")
    decision = main.decide_memory_tail(
        acc.text(), finished=acc.complete(), truncated=acc.truncated(), holed=acc.holed()
    )
    check(decision.outcome == main.tailhealth.STORED_TRIMMED,
          "same CUT path for the alternate error shape")


# ---------------------------------------------------------------------------
# 3. Sticky: error before OR interleaved with other events; order-independent
# ---------------------------------------------------------------------------

def test_errored_is_sticky_regardless_of_order():
    print("\n[test] errored() is sticky: a LATER good-looking event cannot "
          "un-error an accumulator that already saw the failure")
    acc = _fresh(_content("start "), VLLM_ERROR_EVENT, _content("more after the error"), DONE)
    check(acc.errored() is True, "still errored after later content arrives")
    check(acc.complete() is False, "still not complete")
    check(acc.text() == "start more after the error",
          "text() still accumulates everything fed to it — the ERROR flag "
          "is what changes, not what feed() collects")


def test_error_then_finish_reason_stop_still_not_complete():
    print("\n[test] an error followed by a normal finish_reason=stop event "
          "(unrealistic, but must still not read as finished)")
    acc = _fresh(_content(CUT), VLLM_ERROR_EVENT, _finish("stop"), DONE)
    check(acc.errored() is True, "errored")
    check(acc.complete() is False,
          "complete() stays False even though a finish_reason=stop event "
          "arrived afterward — the sticky flag wins")


# ---------------------------------------------------------------------------
# 4. usable() reflects the same fix
# ---------------------------------------------------------------------------

def test_usable_is_false_on_an_errored_stream():
    print("\n[test] usable(): an errored stream is not usable, and not "
          "because it looks truncated (it is a different flag)")
    acc = _fresh(_content(CUT), VLLM_ERROR_EVENT, DONE)
    check(acc.usable() is False, "usable() is False")
    check(acc.truncated() is False, "...but not because of _truncated — errored() is the reason")


# ---------------------------------------------------------------------------
# 5. Malformed error payload does not crash the accumulator
# ---------------------------------------------------------------------------

def test_error_with_non_dict_value_does_not_crash():
    print("\n[test] an {\"error\": \"just a string\"} shape does not raise "
          "(defensive: only vLLM's own shape is documented, not guaranteed)")
    weird = _sse({"error": "just a string, not a dict"})
    acc = _fresh(_content("hello "), weird, DONE)
    check(acc.errored() is True, "still detected as an error (top-level 'error' key alone is enough)")
    check(acc.complete() is False, "still not complete")


def _all_tests():
    return [
        test_defect_shape_content_error_done,
        test_error_finish_reason_shape_is_also_logged,
        test_control_no_done_is_trimmed_as_before,
        test_control_a_clean_finish_is_still_stored_verbatim,
        test_finish_reason_error_shape,
        test_errored_is_sticky_regardless_of_order,
        test_error_then_finish_reason_stop_still_not_complete,
        test_usable_is_false_on_an_errored_stream,
        test_error_with_non_dict_value_does_not_crash,
    ]


if __name__ == "__main__":
    for t in _all_tests():
        # log_once is process-scoped (once per key, ever): reset before
        # each test so a log-line assertion never depends on run order —
        # each test proves the WARNING fires from a cold state, matching
        # what a freshly-started compactor process would actually do the
        # first time it sees this shape.
        main.logsetup._reset_log_once_for_tests()
        t()
    print("\nRESULT:", "all v3.1.9.4 B4 checks passed" if not FAILED else f"{len(FAILED)} check(s) FAILED: {FAILED}")
    sys.exit(0 if not FAILED else 1)
