"""
v3.1.9.4 R6 (hostile pass #17): P17-1.

Both `summarizer._llm_summarize` and `main._summarize_once`'s cut-summary
retry logic (R5/P16-7's "keep the longer trimmed candidate" rule) has two
residual problems:

  (a) The retry is spent even when the first attempt's trim ALREADY meets
      or beats the retry's own (half) word target — an obedient retry
      cannot win the "longer candidate" comparison in that case, so the
      second vLLM call is pure waste.
  (b) A first attempt cut mid REPETITION LOOP (a model stuck repeating one
      line until max_tokens) trims to a long but worthless candidate that
      beat a short, clean retry purely on raw length, because nothing
      checked the trimmed first attempt for degeneracy.

Fix: trim the first attempt and compute the retry's word target BEFORE
deciding to spend the retry call; skip the retry when the trim already
reaches that target AND is not a repetition loop. The "clean retry wins"
comparison also now wins whenever the first attempt trimmed to a loop and
the retry did not, regardless of relative length.

`summarizer.py` cannot import `main.reply_is_degenerate` (an import cycle:
main.py imports summarizer.py). It gets a minimal, LOCAL loop detector,
`_is_repetition_loop`, built from the sentence-boundary regex
`_trim_to_last_sentence` already uses. `main.py` reuses
`reply_is_degenerate` directly (no cycle — this IS main.py).

Direct calls to `_llm_summarize`/`_summarize_once` with a stub vLLM client,
mirroring test_v3194_r5_p167.py's own shape. Synthetic content only.

Run: python test_v3194_r6_p171.py
"""

import asyncio
import os
import sys
import tempfile

os.environ.setdefault("MODEL_REPO", "test-model")
os.environ.setdefault("VLLM_URL", "http://stub:8000")
os.environ["COMPACTOR_RAG_ENABLED"] = "false"
os.environ["COMPACTOR_STORAGE_ROOT"] = tempfile.mkdtemp(prefix="compactor-test-v3194-r6-p171-")
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


def _distinct(n_words, tag):
    """`n_words` (or a little more) of DISTINCT sentences — never a loop."""
    out, w, k = [], 0, 0
    while w < n_words:
        s = f"{tag} detail number {k} mentions Place{k} and Person{k} deciding item {k}."
        out.append(s)
        w += len(s.split())
        k += 1
    return " ".join(out)


# main._summarize_once uses SUMMARY_MAX_TOKENS=1024 -> _target_words=614 ->
# retry target=307 words. summarizer._llm_summarize is called below with
# max_tokens=500 -> _target_words=300 -> retry target=150 words. Both first
# attempts below are well past their own retry target once trimmed, and the
# retry (150 real words) is well short of it, matching the reviewer's own
# probe shape.
_LONG_DISTINCT_CUT_FIRST = _distinct(400, "Scene") + " and then the group went to the"
_LOOP_CUT_FIRST = ("She said she would think about it. " * 60) + "She said she would"
_SHORT_CLEAN_RETRY = _distinct(150, "Retry")


class _Resp:
    def __init__(self, content, fr):
        self._d = {"choices": [{"message": {"content": content}, "finish_reason": fr}]}

    def raise_for_status(self):
        pass

    def json(self):
        return self._d


class _Client:
    """First call always cut (`first`/"length"); retry returns whatever
    `retry_content`/`retry_finish` say. Raises if a third call is made,
    so a skipped retry that still tries to call again fails loudly."""

    def __init__(self, first, retry_content, retry_finish):
        self.n = 0
        self.first = first
        self.retry_content = retry_content
        self.retry_finish = retry_finish

    async def post(self, url, json=None, timeout=None, **kw):
        self.n += 1
        if self.n == 1:
            return _Resp(self.first, "length")
        if self.n == 2:
            return _Resp(self.retry_content, self.retry_finish)
        raise AssertionError(f"unexpected 3rd call to vLLM (n={self.n})")


# ---------------------------------------------------------------------------
# main._summarize_once
# ---------------------------------------------------------------------------

def test_main_skip_retry_when_first_attempt_trim_already_long_enough():
    print("\n[test] main._summarize_once: (P17-1 part a) the retry is "
          "SKIPPED when the trimmed first attempt already meets the "
          "retry's own word target — the retry could not win the length "
          "comparison, so spending the call is pure waste")
    before_retried = main.retried_compaction_summary_count()
    before_trunc = main.truncated_compaction_summary_count()
    c = _Client(_LONG_DISTINCT_CUT_FIRST, _SHORT_CLEAN_RETRY, "stop")
    result = asyncio.run(main._summarize_once(c, [{"role": "user", "content": "x"}]))
    check(c.n == 1, f"exactly 1 call made (retry skipped): {c.n}")
    check(result == main._trim_best_effort_summary(_LONG_DISTINCT_CUT_FIRST),
          f"kept the trimmed first attempt untouched: {result!r}")
    check(main.retried_compaction_summary_count() == before_retried,
          f"retried counter did NOT move (no retry call was made): "
          f"{main.retried_compaction_summary_count()}")
    check(main.truncated_compaction_summary_count() == before_trunc + 1,
          f"truncated counter moved (result IS a trim): "
          f"{main.truncated_compaction_summary_count()}")


def test_main_loop_first_attempt_loses_to_clean_retry_regardless_of_length():
    print("\n[test] main._summarize_once: (P17-1 part b) a first attempt "
          "trimmed to a REPETITION LOOP must not win the length comparison "
          "over a short, clean retry — the retry always runs for a looping "
          "first attempt, and the clean retry always wins")
    c = _Client(_LOOP_CUT_FIRST, _SHORT_CLEAN_RETRY, "stop")
    result = asyncio.run(main._summarize_once(c, [{"role": "user", "content": "x"}]))
    check(c.n == 2, f"exactly 2 calls made (loop does not skip the retry): {c.n}")
    trimmed_loop = main._trim_best_effort_summary(_LOOP_CUT_FIRST)
    check(len(trimmed_loop) > len(_SHORT_CLEAN_RETRY),
          f"sanity: the trimmed loop really is longer than the clean retry "
          f"({len(trimmed_loop)} vs {len(_SHORT_CLEAN_RETRY)})")
    check(result == _SHORT_CLEAN_RETRY,
          f"the CLEAN retry won despite being shorter: {result!r}")


def test_main_control_long_clean_retry_still_wins():
    print("\n[test] main._summarize_once: CONTROL — a clean retry at "
          "least as long as the trimmed first attempt still wins outright")
    long_retry = _distinct(700, "Retry")
    c = _Client(_LOOP_CUT_FIRST, long_retry, "stop")
    result = asyncio.run(main._summarize_once(c, [{"role": "user", "content": "x"}]))
    check(c.n == 2, f"exactly 2 calls made: {c.n}")
    check(result == long_retry, f"the long clean retry won: {result!r}")


def test_main_control_both_cut_unaffected():
    print("\n[test] main._summarize_once: CONTROL — both attempts cut, "
          "unchanged 'keep the longer trimmed' behaviour (retry still runs "
          "because the fixture's first-attempt trim never meets the retry "
          "target)")
    short_first = "Sentence one is short and cut of"
    c = _Client(short_first, "a retry that is also cut mid", "length")
    result = asyncio.run(main._summarize_once(c, [{"role": "user", "content": "x"}]))
    check(c.n == 2, f"exactly 2 calls made: {c.n}")
    trimmed_first = main._trim_best_effort_summary(short_first)
    trimmed_retry = main._trim_best_effort_summary("a retry that is also cut mid")
    expected = trimmed_first if len(trimmed_first) >= len(trimmed_retry) else trimmed_retry
    check(result == expected, f"kept the longer of the two trimmed candidates: {result!r}")


# ---------------------------------------------------------------------------
# summarizer._llm_summarize
# ---------------------------------------------------------------------------

def test_summarizer_skip_retry_when_first_attempt_trim_already_long_enough():
    print("\n[test] summarizer._llm_summarize: (P17-1 part a) retry SKIPPED "
          "when the trim already meets the retry's own word target")
    before_retried = summarizer.retried_summary_count()
    before_trunc = summarizer.truncated_summary_count()
    c = _Client(_LONG_DISTINCT_CUT_FIRST, _SHORT_CLEAN_RETRY, "stop")
    result = asyncio.run(summarizer._llm_summarize(
        c, "http://stub", "m", "system prompt", "body", 500,
    ))
    check(c.n == 1, f"exactly 1 call made (retry skipped): {c.n}")
    check(result == summarizer._trim_best_effort(_LONG_DISTINCT_CUT_FIRST),
          f"kept the trimmed first attempt untouched: {result!r}")
    check(summarizer.retried_summary_count() == before_retried,
          f"retried counter did NOT move: {summarizer.retried_summary_count()}")
    check(summarizer.truncated_summary_count() == before_trunc + 1,
          f"truncated counter moved: {summarizer.truncated_summary_count()}")


def test_summarizer_loop_first_attempt_loses_to_clean_retry_regardless_of_length():
    print("\n[test] summarizer._llm_summarize: (P17-1 part b) a looping "
          "trimmed first attempt must not beat a shorter clean retry")
    c = _Client(_LOOP_CUT_FIRST, _SHORT_CLEAN_RETRY, "stop")
    result = asyncio.run(summarizer._llm_summarize(
        c, "http://stub", "m", "system prompt", "body", 500,
    ))
    check(c.n == 2, f"exactly 2 calls made (loop does not skip the retry): {c.n}")
    trimmed_loop = summarizer._trim_best_effort(_LOOP_CUT_FIRST)
    check(len(trimmed_loop) > len(_SHORT_CLEAN_RETRY),
          f"sanity: trimmed loop is longer than the clean retry "
          f"({len(trimmed_loop)} vs {len(_SHORT_CLEAN_RETRY)})")
    check(result == _SHORT_CLEAN_RETRY,
          f"the CLEAN retry won despite being shorter: {result!r}")


def test_summarizer_control_long_clean_retry_still_wins():
    print("\n[test] summarizer._llm_summarize: CONTROL — a clean retry at "
          "least as long as the trimmed first attempt still wins outright")
    long_retry = _distinct(700, "Retry")
    c = _Client(_LOOP_CUT_FIRST, long_retry, "stop")
    result = asyncio.run(summarizer._llm_summarize(
        c, "http://stub", "m", "system prompt", "body", 500,
    ))
    check(c.n == 2, f"exactly 2 calls made: {c.n}")
    check(result == long_retry, f"the long clean retry won: {result!r}")


def test_summarizer_control_both_cut_unaffected():
    print("\n[test] summarizer._llm_summarize: CONTROL — both cut, "
          "unchanged 'keep the longer trimmed' behaviour")
    short_first = "Sentence one is short and cut of"
    c = _Client(short_first, "a retry that is also cut mid", "length")
    result = asyncio.run(summarizer._llm_summarize(
        c, "http://stub", "m", "system prompt", "body", 500,
    ))
    check(c.n == 2, f"exactly 2 calls made: {c.n}")
    trimmed_first = summarizer._trim_best_effort(short_first)
    trimmed_retry = summarizer._trim_best_effort("a retry that is also cut mid")
    expected = trimmed_first if len(trimmed_first) >= len(trimmed_retry) else trimmed_retry
    check(result == expected, f"kept the longer of the two trimmed candidates: {result!r}")


def test_is_repetition_loop_unit():
    print("\n[test] summarizer._is_repetition_loop: direct unit checks")
    check(summarizer._is_repetition_loop(_LOOP_CUT_FIRST) is True,
          "60x-repeated sentence flagged as a loop")
    check(summarizer._is_repetition_loop(_LONG_DISTINCT_CUT_FIRST) is False,
          "distinct sentences NOT flagged")
    check(summarizer._is_repetition_loop("Short.") is False,
          "too few sentences (<4) never flagged")
    check(summarizer._is_repetition_loop("") is False, "empty text never flagged")


def test_both_cut_loop_never_wins_on_length():
    print("\n[test] both paths: (P17-1 part b, coordinator follow-up) when "
          "the retry is ALSO cut, a first attempt trimmed to a repetition "
          "loop still does not beat the retry's clean trim on length")
    cut_retry = _SHORT_CLEAN_RETRY + " and then the group went to the"
    c = _Client(_LOOP_CUT_FIRST, cut_retry, "length")
    result = asyncio.run(main._summarize_once(c, [{"role": "user", "content": "x"}]))
    check(c.n == 2, f"main: exactly 2 calls made: {c.n}")
    check(result == main._trim_best_effort_summary(cut_retry),
          f"main: kept the retry's clean trim, not the longer loop: {result[:60]!r}")
    c = _Client(_LOOP_CUT_FIRST, cut_retry, "length")
    result = asyncio.run(summarizer._llm_summarize(
        c, "http://stub", "m", "system prompt", "body", 500,
    ))
    check(c.n == 2, f"summarizer: exactly 2 calls made: {c.n}")
    check(result == summarizer._trim_best_effort(cut_retry),
          f"summarizer: kept the retry's clean trim, not the longer loop: {result[:60]!r}")


def _all_tests():
    return [
        test_main_skip_retry_when_first_attempt_trim_already_long_enough,
        test_main_loop_first_attempt_loses_to_clean_retry_regardless_of_length,
        test_main_control_long_clean_retry_still_wins,
        test_main_control_both_cut_unaffected,
        test_summarizer_skip_retry_when_first_attempt_trim_already_long_enough,
        test_summarizer_loop_first_attempt_loses_to_clean_retry_regardless_of_length,
        test_summarizer_control_long_clean_retry_still_wins,
        test_summarizer_control_both_cut_unaffected,
        test_is_repetition_loop_unit,
        test_both_cut_loop_never_wins_on_length,
    ]


if __name__ == "__main__":
    for t in _all_tests():
        t()
    print("\nRESULT:", "all v3.1.9.4 R6 P17-1 checks passed" if not FAILED else f"{len(FAILED)} check(s) FAILED: {FAILED}")
    sys.exit(0 if not FAILED else 1)
