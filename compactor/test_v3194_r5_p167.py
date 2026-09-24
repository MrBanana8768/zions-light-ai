"""
v3.1.9.4 R5 (hostile pass #16 follow-up): P16-7.

Both `summarizer._llm_summarize` and `main._summarize_once`'s cut-summary
retry let a retry that finished CLEANLY win automatically, the instant it
happened. `_summary_retry_suffix`/`_retry_suffix` ask for roughly HALF the
tier's ordinary word target, so a clean retry can be much SHORTER than a
first attempt that was cut mid-sentence but still trims (via
`_trim_best_effort[_summary]`) to a complete, much longer result. The
clean-but-short retry silently replaced the longer, perfectly usable
trimmed first attempt, and nothing counted that a retry had even run.

Fix: a clean retry wins only if it is AT LEAST AS LONG as the first
attempt trimmed — the same "keep the longer trimmed candidate" rule the
both-cut case already used. A new counter
(`retried_summary_count`/`retried_compaction_summary_count`) counts every
retry actually made, next to the existing `truncated_*` counters (which
count only the outcome "ended up trimmed").

Direct calls to `_llm_summarize` and `_summarize_once` with a stub vLLM
client. Synthetic content only.

Run: python test_v3194_r5_p167.py
"""

import asyncio
import os
import sys
import tempfile

os.environ.setdefault("MODEL_REPO", "test-model")
os.environ.setdefault("VLLM_URL", "http://stub:8000")
os.environ["COMPACTOR_RAG_ENABLED"] = "false"
os.environ["COMPACTOR_STORAGE_ROOT"] = tempfile.mkdtemp(prefix="compactor-test-v3194-r5-p167-")
import memory  # noqa: E402
memory.ensure_storage_layout()
import main  # noqa: E402
import summarizer  # noqa: E402

FAILED: list[str] = []


def check(cond, label):
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"FAIL {label}")
        FAILED.append(label)


# A first attempt that is CUT (no terminal punctuation) but trims, via a
# real sentence boundary, to a long and clearly usable result (217 chars —
# see this file's own dev check).
_LONG_CUT_FIRST = (
    "Sentence one is here with padding padding padding padding padding "
    "padding padding padding. Sentence two continues on and on with much "
    "more content padding it out significantly with extra words extra "
    "words extra words. Sentence three is cut of"
)
_LONG_CUT_TRIMMED_LEN = len(main._trim_best_effort_summary(_LONG_CUT_FIRST))

# A retry that finishes CLEANLY (no cut) but is much shorter than the
# trimmed first attempt above — the exact shape that used to win
# unconditionally.
_SHORT_CLEAN_RETRY = "Short summary."
assert len(_SHORT_CLEAN_RETRY) < _LONG_CUT_TRIMMED_LEN

# A retry that finishes cleanly AND is at least as long as the trimmed
# first attempt — this one SHOULD win.
_LONG_CLEAN_RETRY = (
    "A complete retry that finished cleanly and is deliberately padded "
    "with enough words words words words words words words words words "
    "words words words words to be at least as long as the trimmed "
    "first attempt above, so it wins the length comparison outright."
)
assert len(_LONG_CLEAN_RETRY) >= _LONG_CUT_TRIMMED_LEN


class _Resp:
    def __init__(self, content, fr):
        self._d = {"choices": [{"message": {"content": content}, "finish_reason": fr}]}

    def raise_for_status(self):
        pass

    def json(self):
        return self._d


class _Client:
    """First call always cut (_LONG_CUT_FIRST/"length"); retry returns
    whatever `retry_content`/`retry_finish` say."""

    def __init__(self, retry_content, retry_finish):
        self.n = 0
        self.retry_content = retry_content
        self.retry_finish = retry_finish

    async def post(self, url, json=None, timeout=None, **kw):
        self.n += 1
        if self.n == 1:
            return _Resp(_LONG_CUT_FIRST, "length")
        return _Resp(self.retry_content, self.retry_finish)


# ---------------------------------------------------------------------------
# main._summarize_once
# ---------------------------------------------------------------------------

def test_main_short_clean_retry_loses_to_the_longer_trimmed_first_attempt():
    print("\n[test] main._summarize_once: a clean-but-SHORT retry must not "
          "beat the longer trimmed first attempt")
    before = main.retried_compaction_summary_count()
    before_trunc = main.truncated_compaction_summary_count()
    c = _Client(_SHORT_CLEAN_RETRY, "stop")
    result = asyncio.run(main._summarize_once(c, [{"role": "user", "content": "x"}]))
    check(c.n == 2, f"exactly 2 calls made (first + retry): {c.n}")
    check(result == main._trim_best_effort_summary(_LONG_CUT_FIRST),
          f"kept the longer trimmed first attempt, not the short clean retry: {result!r}")
    check(main.retried_compaction_summary_count() == before + 1,
          f"retried counter moved by 1: {main.retried_compaction_summary_count()}")
    check(main.truncated_compaction_summary_count() == before_trunc + 1,
          f"truncated counter ALSO moved (final output is trimmed, not clean): "
          f"{main.truncated_compaction_summary_count()}")


def test_main_long_clean_retry_wins():
    print("\n[test] main._summarize_once: CONTROL — a clean retry that is "
          "at least as long as the trimmed first attempt still wins")
    before = main.retried_compaction_summary_count()
    before_trunc = main.truncated_compaction_summary_count()
    c = _Client(_LONG_CLEAN_RETRY, "stop")
    result = asyncio.run(main._summarize_once(c, [{"role": "user", "content": "x"}]))
    check(c.n == 2, f"exactly 2 calls made: {c.n}")
    check(result == _LONG_CLEAN_RETRY, f"the clean retry won: {result!r}")
    check(main.retried_compaction_summary_count() == before + 1,
          f"retried counter moved by 1: {main.retried_compaction_summary_count()}")
    check(main.truncated_compaction_summary_count() == before_trunc,
          f"truncated counter did NOT move (final output is the clean retry, "
          f"not trimmed): {main.truncated_compaction_summary_count()}")


def test_main_control_both_cut_keeps_the_longer_trimmed_candidate():
    print("\n[test] main._summarize_once: CONTROL — both cut, unchanged "
          "'keep the longer trimmed' behaviour")
    c = _Client("a retry that is also cut mid", "length")
    result = asyncio.run(main._summarize_once(c, [{"role": "user", "content": "x"}]))
    check(c.n == 2, f"exactly 2 calls made: {c.n}")
    trimmed_first = main._trim_best_effort_summary(_LONG_CUT_FIRST)
    trimmed_retry = main._trim_best_effort_summary("a retry that is also cut mid")
    expected = trimmed_first if len(trimmed_first) >= len(trimmed_retry) else trimmed_retry
    check(result == expected, f"kept the longer of the two trimmed candidates: {result!r}")


# ---------------------------------------------------------------------------
# summarizer._llm_summarize
# ---------------------------------------------------------------------------

def test_summarizer_short_clean_retry_loses_to_the_longer_trimmed_first_attempt():
    print("\n[test] summarizer._llm_summarize: a clean-but-SHORT retry "
          "must not beat the longer trimmed first attempt")
    before = summarizer.retried_summary_count()
    before_trunc = summarizer.truncated_summary_count()
    c = _Client(_SHORT_CLEAN_RETRY, "stop")
    result = asyncio.run(summarizer._llm_summarize(
        c, "http://stub", "m", "system prompt", "body", 500,
    ))
    check(c.n == 2, f"exactly 2 calls made: {c.n}")
    check(result == summarizer._trim_best_effort(_LONG_CUT_FIRST),
          f"kept the longer trimmed first attempt: {result!r}")
    check(summarizer.retried_summary_count() == before + 1,
          f"retried counter moved by 1: {summarizer.retried_summary_count()}")
    check(summarizer.truncated_summary_count() == before_trunc + 1,
          f"truncated counter also moved: {summarizer.truncated_summary_count()}")


def test_summarizer_long_clean_retry_wins():
    print("\n[test] summarizer._llm_summarize: CONTROL — a long-enough "
          "clean retry still wins")
    before = summarizer.retried_summary_count()
    before_trunc = summarizer.truncated_summary_count()
    c = _Client(_LONG_CLEAN_RETRY, "stop")
    result = asyncio.run(summarizer._llm_summarize(
        c, "http://stub", "m", "system prompt", "body", 500,
    ))
    check(c.n == 2, f"exactly 2 calls made: {c.n}")
    check(result == _LONG_CLEAN_RETRY, f"the clean retry won: {result!r}")
    check(summarizer.retried_summary_count() == before + 1,
          f"retried counter moved by 1: {summarizer.retried_summary_count()}")
    check(summarizer.truncated_summary_count() == before_trunc,
          f"truncated counter did NOT move: {summarizer.truncated_summary_count()}")


def test_summarizer_control_both_cut_keeps_the_longer_trimmed_candidate():
    print("\n[test] summarizer._llm_summarize: CONTROL — both cut, "
          "unchanged 'keep the longer trimmed' behaviour")
    c = _Client("a retry that is also cut mid", "length")
    result = asyncio.run(summarizer._llm_summarize(
        c, "http://stub", "m", "system prompt", "body", 500,
    ))
    check(c.n == 2, f"exactly 2 calls made: {c.n}")
    trimmed_first = summarizer._trim_best_effort(_LONG_CUT_FIRST)
    trimmed_retry = summarizer._trim_best_effort("a retry that is also cut mid")
    expected = trimmed_first if len(trimmed_first) >= len(trimmed_retry) else trimmed_retry
    check(result == expected, f"kept the longer of the two trimmed candidates: {result!r}")


def test_health_surfaces_both_retried_counters():
    print("\n[test] /health/full's checks.truncated_summaries surfaces "
          "the new *_retried counts next to the existing *_truncated ones")
    import health
    state = health._truncated_summary_state()
    check(state.get("available") is True, f"available: {state}")
    check("hierarchy_retried" in state, f"hierarchy_retried present: {state}")
    check("compaction_retried" in state, f"compaction_retried present: {state}")
    check(isinstance(state.get("hierarchy_retried"), int), f"hierarchy_retried is a count: {state}")
    check(isinstance(state.get("compaction_retried"), int), f"compaction_retried is a count: {state}")


def _all_tests():
    return [
        test_main_short_clean_retry_loses_to_the_longer_trimmed_first_attempt,
        test_main_long_clean_retry_wins,
        test_main_control_both_cut_keeps_the_longer_trimmed_candidate,
        test_summarizer_short_clean_retry_loses_to_the_longer_trimmed_first_attempt,
        test_summarizer_long_clean_retry_wins,
        test_summarizer_control_both_cut_keeps_the_longer_trimmed_candidate,
        test_health_surfaces_both_retried_counters,
    ]


if __name__ == "__main__":
    for t in _all_tests():
        t()
    print("\nRESULT:", "all v3.1.9.4 R5 P16-7 checks passed" if not FAILED else f"{len(FAILED)} check(s) FAILED: {FAILED}")
    sys.exit(0 if not FAILED else 1)
