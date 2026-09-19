"""
CPU-only tests for v3.1.9.4 (P15-6): a summary vLLM cuts at max_tokens is no
longer stored as if finished.

Reproduces the defect this fixes (a `finish_reason=length` reply stored
byte-for-byte, indistinguishable from a `"stop"` reply, watermark advanced
past turns the summary never reached) against the REAL `summarizer.
maybe_rollup` / `_llm_summarize` — only the HTTP client is stubbed, the same
shape test_summarizer.py's `_mock_client_returning` uses, now carrying
`finish_reason`. Synthetic conversational text only (no real user data).

Covers:
  - CONTROL: a `"stop"` reply is stored and advances the watermark exactly as
    before (nothing about the untruncated path changes).
  - A cut reply whose retry ALSO cuts is trimmed to its last complete
    sentence, never stored mid-sentence, still advances the unit, is logged
    at WARNING with the conversation and tier, and counted. The retry uses
    the SAME max_tokens as the first attempt (lowering it guarantees
    nothing — the prompt's own word target is what changes).
  - A cut reply whose retry finishes cleanly is stored as the retry's (full,
    shorter) text — no trimming, not counted as a truncation.
  - A cut reply with no sentence boundary anywhere (a bullet list, or any
    reply with no terminal punctuation) falls back to a line or word
    boundary rather than being refused — the hierarchy must never stall on
    this, since it is routine, not degenerate.
  - Only a truly empty/whitespace-only reply is refused (returns "") — the
    one case the fallback chain has nothing to trim to.
  - When the retry is ALSO cut, the LONGER of the two trimmed candidates is
    kept, not automatically the retry's (a tighter retry target can produce
    LESS usable content than the first attempt trimmed would have).
  - The retry is skipped, not attempted, when the shared vLLM-call budget
    has nothing left for a second real call — and that second call IS
    counted against the budget when it does run.
  - M1's hard constraint: EVERY summarize call across a whole L1->L2->L3
    cascade is cut and stays cut after its retry, and the hierarchy still
    advances through all three tiers in one maybe_rollup call — it does not
    stall.

Run: python test_v3194_mem_cut.py
"""

import asyncio
import contextlib
import logging
import os
import shutil
import sys
import tempfile

_TMP = tempfile.mkdtemp(prefix="compactor-test-v3194-mem-cut-")
os.environ["COMPACTOR_STORAGE_ROOT"] = _TMP
os.environ["COMPACTOR_L1_CHUNK_SIZE"] = "4"
os.environ["COMPACTOR_L2_CHUNK_SIZE"] = "3"
os.environ["COMPACTOR_L3_CHUNK_SIZE"] = "2"

import httpx  # noqa: E402
import memory  # noqa: E402
import summarizer  # noqa: E402


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


def _wipe():
    if os.path.exists(_TMP):
        shutil.rmtree(_TMP)
    memory.ensure_storage_layout()


# ---------------------------------------------------------------------------
# Log capture
# ---------------------------------------------------------------------------

class _Collector(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)


@contextlib.contextmanager
def capture(logger_name: str = "compactor.summarizer"):
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


def find(records, needle: str):
    for r in records:
        if needle in r.getMessage():
            return r
    return None


# ---------------------------------------------------------------------------
# Stub HTTP client — records max_tokens per call, returns a scripted
# (content, finish_reason) queue, or a fixed one forever.
# ---------------------------------------------------------------------------

class _Resp:
    def __init__(self, payload, status=200):
        self._p = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("x", request=None, response=None)

    def json(self):
        return self._p


def _install(handler):
    """handler(kwargs) -> (content, finish_reason). Patches
    summarizer.httpx.AsyncClient; returns the restore callable."""
    calls: list[int] = []

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, **kw):
            if str(url).endswith("/tokenize"):
                raise RuntimeError("no /tokenize in this stub — pessimistic pricing only")
            body = kw.get("json") or {}
            calls.append(body.get("max_tokens"))
            content, finish_reason = handler(body)
            return _Resp({
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": finish_reason,
                }],
            })

        async def aclose(self):
            pass

    orig = summarizer.httpx.AsyncClient
    summarizer.httpx.AsyncClient = lambda *a, **k: _Client()
    return calls, orig


def _restore(orig):
    summarizer.httpx.AsyncClient = orig


def _msgs(n_turns: int) -> list[dict]:
    out = [{"role": "system", "content": "sys"}]
    for i in range(n_turns):
        out.append({"role": "user", "content": f"user turn {i} about the garden plan, item {i}."})
        out.append({"role": "assistant", "content": f"assistant turn {i} answering about item {i}."})
    return out


CUT = (
    "Scene summary: the user and the assistant planned the garden, chose "
    "basil and tomatoes for the east bed, and agreed the irrigation timer "
    "would run at six. Then the user asked about the"
)
FINISHED = "Scene summary: the garden plan settled on basil and tomatoes, irrigation at six."


# ---------------------------------------------------------------------------
# CONTROL: a "stop" reply is unaffected
# ---------------------------------------------------------------------------

def test_control_stop_reply_unchanged():
    print("\n[test] CONTROL: a finish_reason=stop reply is stored and advances the watermark exactly as before")
    _wipe()
    calls, orig = _install(lambda body: (FINISHED, "stop"))
    try:
        # _msgs(2) -> 4 turns -> exactly ONE L1 chunk (COMPACTOR_L1_CHUNK_SIZE=4
        # from this module's own env, set above import), so exactly one
        # real _llm_summarize call — no L2 fold to conflate with L1's own
        # count.
        st = asyncio.run(summarizer.maybe_rollup("cut-control", _msgs(2), "http://stub:8000", "m"))
    finally:
        _restore(orig)
    assert_true(bool(st.get("l1")), "an L1 chunk was produced")
    assert_eq(st["l1"][0]["text"], FINISHED, "the chunk holds the model's text verbatim")
    assert_eq(st["last_summarized_turn"], 4, "the watermark reached the chunk's last turn")
    assert_eq(summarizer.truncated_summary_count(), 0, "nothing was counted as truncated")


# ---------------------------------------------------------------------------
# The defect this fix targets: a length-cut reply, retry also cut
# ---------------------------------------------------------------------------

def test_cut_reply_retried_and_still_cut_is_trimmed_not_stored_verbatim():
    print("\n[test] a finish_reason=length reply, still cut on retry, is trimmed to its last complete sentence")
    _wipe()
    before = summarizer.truncated_summary_count()
    calls, orig = _install(lambda body: (CUT, "length"))
    try:
        with capture() as log:
            # _msgs(2) -> exactly ONE L1 unit (see test_control's comment).
            st = asyncio.run(summarizer.maybe_rollup("cut-defect", _msgs(2), "http://stub:8000", "m"))
    finally:
        _restore(orig)
    assert_true(bool(st.get("l1")), "the unit still advanced despite every call being cut")
    text = st["l1"][0]["text"]
    assert_true(text != CUT, "the stored text is NOT the raw cut reply verbatim (the defect this fixes)")
    assert_true(CUT.startswith(text) or text in CUT, "the stored text is a PREFIX of what the model wrote")
    assert_true(text.rstrip().endswith((".", "!", "?")), "the stored text ends on real sentence punctuation, never mid-sentence")
    assert_true("Then the user asked about the" not in text, "the trailing mid-sentence fragment was cut off")
    assert_eq(st["last_summarized_turn"], 4, "the watermark still advances — the hierarchy does not stall on a cut reply")
    assert_eq(summarizer.truncated_summary_count(), before + 1, "the truncation was counted")
    rec = find(log.records, "rollup summary was cut at max_tokens")
    assert_true(rec is not None, "a WARNING was logged for the still-cut reply")
    assert_eq(rec.levelno, logging.WARNING, "...at WARNING")
    assert_true("conv=cut-defect" in rec.getMessage(), "...naming the conversation")
    assert_true("L1-tier" in rec.getMessage(), "...naming the tier")
    # Two calls: the first attempt, then the one retry. Coordinator review:
    # the retry keeps the SAME max_tokens as the first attempt — lowering it
    # guarantees nothing, since the system prompt's own word target
    # (_target_words) is what the retry's instruction tightens instead.
    assert_eq(len(calls), 2, "exactly one retry was attempted, not a loop")
    assert_eq(calls[1], calls[0], "the retry uses the SAME max_tokens as the first attempt — only the instruction changed")


def test_cut_reply_whose_retry_finishes_is_stored_untrimmed():
    print("\n[test] a cut first attempt whose retry finishes cleanly stores the retry's full text, uncounted")
    _wipe()
    before = summarizer.truncated_summary_count()
    queue = [(CUT, "length"), (FINISHED, "stop")]

    def handler(body):
        return queue.pop(0)

    calls, orig = _install(handler)
    try:
        with capture() as log:
            st = asyncio.run(summarizer.maybe_rollup("cut-retry-ok", _msgs(2), "http://stub:8000", "m"))
    finally:
        _restore(orig)
    assert_eq(st["l1"][0]["text"], FINISHED, "the retry's own (complete) text is stored, not a trim of the first attempt")
    assert_eq(summarizer.truncated_summary_count(), before, "a reply that finished on retry is not counted as truncated")
    assert_true(find(log.records, "rollup summary was cut at max_tokens") is None, "no truncation WARNING when the retry finished cleanly")
    assert_eq(len(calls), 2, "first attempt + one retry")


def test_cut_reply_with_no_sentence_boundary_falls_back_to_word_boundary():
    print("\n[test] a cut reply with no terminal punctuation anywhere falls back to a word boundary, marked, rather than being refused")
    _wipe()
    NO_PUNCT = "no punctuation anywhere in this reply at all"

    async def _direct():
        class _Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, **kw):
                return _Resp({
                    "choices": [{
                        "message": {"role": "assistant", "content": NO_PUNCT},
                        "finish_reason": "length",
                    }],
                })
        return await summarizer._llm_summarize(
            _Client(), "http://stub:8000", "m", summarizer._PROMPT_L1, "body", 500,
        )

    result = asyncio.run(_direct())
    # Coordinator review (problem 3): the hierarchy must never stall just
    # because a cut reply has no sentence punctuation — routine for a
    # bullet/numbered-list summary. _trim_best_effort falls back to the last
    # complete WORD, marked with " …" so nothing downstream mistakes this for
    # a naturally short, complete summary.
    assert_true(result != "", "NOT refused — a word boundary exists, so there is something usable")
    assert_eq(result, "no punctuation anywhere in this reply at …", "trimmed to the last complete word, with the marker appended")
    assert_true(NO_PUNCT.startswith(result[:-2]), "the kept text is a real prefix of what the model wrote, marker aside")


def test_only_a_genuinely_empty_reply_is_refused():
    print("\n[test] CONTROL: only a reply that is itself empty/whitespace is refused — not merely unpunctuated")
    _wipe()

    async def _direct(content):
        class _Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, **kw):
                return _Resp({
                    "choices": [{
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "length",
                    }],
                })
        return await summarizer._llm_summarize(
            _Client(), "http://stub:8000", "m", summarizer._PROMPT_L1, "body", 500,
        )

    assert_eq(asyncio.run(_direct("")), "", "an empty reply is still refused")
    assert_eq(asyncio.run(_direct("   \n  ")), "", "a whitespace-only reply is still refused")


# ---------------------------------------------------------------------------
# Coordinator review, problem 3: a bullet-list reply with no terminal
# punctuation, cut on BOTH attempts, still advances the hierarchy and lands
# on a LINE boundary, not a sentence one (there is no sentence to find) and
# not mid-word.
# ---------------------------------------------------------------------------

CUT_BULLETS = (
    "Scene summary:\n"
    "- Chose basil and tomatoes for the east bed\n"
    "- Agreed the irrigation timer runs at six\n"
    "- Started listing what still needed digging up before the first fro"
)


def test_bullet_list_cut_on_both_attempts_falls_back_to_a_line_boundary():
    print("\n[test] a bullet-list summary with no terminal punctuation, cut on both attempts, advances and ends on a line boundary")
    _wipe()
    before = summarizer.truncated_summary_count()
    calls, orig = _install(lambda body: (CUT_BULLETS, "length"))
    try:
        with capture() as log:
            st = asyncio.run(summarizer.maybe_rollup("cut-bullets", _msgs(2), "http://stub:8000", "m"))
    finally:
        _restore(orig)
    assert_true(bool(st.get("l1")), "the unit still advanced — a bullet list with no punctuation is not a stall")
    assert_eq(st["last_summarized_turn"], 4, "the watermark advanced")
    text = st["l1"][0]["text"]
    assert_true("fro" not in text, "the mid-word cut fragment of the last, incomplete bullet is gone")
    assert_eq(
        text, "\n".join(CUT_BULLETS.split("\n")[:-1]),
        "the stored text is exactly the complete lines before the cut one — a real LINE boundary, not a sentence (there is none) and not a word fallback either",
    )
    assert_eq(summarizer.truncated_summary_count(), before + 1, "still counted as a truncation")
    rec = find(log.records, "rollup summary was cut at max_tokens")
    assert_true(rec is not None, "still logged at WARNING")


# ---------------------------------------------------------------------------
# Coordinator review, problem 2: when the retry is ALSO cut, keep whichever
# TRIMMED candidate is longer, not automatically the retry's.
# ---------------------------------------------------------------------------

LONG_FIRST_ATTEMPT = (
    "Scene summary: the garden plan settled on basil and tomatoes for the east bed, with irrigation "
    "set for six each morning. Then the user began asking about whether the compost bin should move "
    "before the first fro"
)
SHORT_RETRY = "Scene summary: talked about the gar"


def test_when_both_cut_the_longer_trimmed_candidate_is_kept_not_automatically_the_retry():
    print("\n[test] when the retry is also cut, the LONGER trimmed candidate (first attempt or retry) is kept")
    _wipe()
    queue = [(LONG_FIRST_ATTEMPT, "length"), (SHORT_RETRY, "length")]

    def handler(body):
        return queue.pop(0)

    calls, orig = _install(handler)
    try:
        with capture() as log:
            st = asyncio.run(summarizer.maybe_rollup("cut-keep-longer", _msgs(2), "http://stub:8000", "m"))
    finally:
        _restore(orig)
    text = st["l1"][0]["text"]
    expected_long_trim = summarizer._trim_to_last_sentence(LONG_FIRST_ATTEMPT)
    expected_short_trim = summarizer._trim_best_effort(SHORT_RETRY)
    assert_true(len(expected_long_trim) > len(expected_short_trim), "test setup: the first attempt's trim really is the longer of the two")
    assert_eq(text, expected_long_trim, "the LONGER trim (the first attempt's) is stored")
    assert_true(text != expected_short_trim, "...not the shorter retry's trim, even though the retry is the more recent attempt")
    assert_eq(len(calls), 2, "the retry was still attempted (it might have finished cleanly and been shorter but complete — it did not here)")


# ---------------------------------------------------------------------------
# Coordinator review, problem 4: the retry is a real extra vLLM call and
# must be counted against, and skipped when exhausted by, the shared
# call budget.
# ---------------------------------------------------------------------------

def test_exhausted_budget_skips_the_retry_and_trims_the_first_attempt():
    print("\n[test] a shared vLLM-call budget with nothing left for a second call skips the retry entirely")
    _wipe()
    before = summarizer.truncated_summary_count()
    calls, orig = _install(lambda body: (CUT, "length"))
    try:
        with capture() as log, summarizer.vllm_call_budget_ctx(1) as budget:
            # max_calls=1: the unit boundary check allows this one L1 chunk
            # to START (remaining=1 > 0); _call's own per-call decrement
            # then spends it on the guaranteed first call, leaving nothing
            # for _llm_summarize's retry.
            st = asyncio.run(summarizer.maybe_rollup("cut-budget-exhausted", _msgs(2), "http://stub:8000", "m"))
    finally:
        _restore(orig)
    assert_eq(len(calls), 1, "only the ONE guaranteed call was made — no retry")
    assert_true(bool(st.get("l1")), "the unit still advanced on the trimmed first attempt alone")
    text = st["l1"][0]["text"]
    assert_eq(text, summarizer._trim_to_last_sentence(CUT), "the first attempt was trimmed and stored, exactly as if no retry had ever been possible")
    assert_eq(summarizer.truncated_summary_count(), before + 1, "still counted")
    rec = find(log.records, "rollup summary was cut at max_tokens")
    assert_true(rec is not None and "no vLLM-call budget left for a retry" in rec.getMessage(),
                f"the log says WHY there was no retry: {rec.getMessage() if rec else None!r}")


def test_budget_accounting_includes_the_retry_call():
    print("\n[test] the retry's real HTTP call is counted against the shared vLLM-call budget too")
    _wipe()
    calls, orig = _install(lambda body: (CUT, "length"))
    try:
        with summarizer.vllm_call_budget_ctx(5) as budget:
            asyncio.run(summarizer.maybe_rollup("cut-budget-accounting", _msgs(2), "http://stub:8000", "m"))
            spent_inside = 5 - budget["remaining"]
    finally:
        _restore(orig)
    assert_eq(len(calls), 2, "the first attempt AND the retry both made real HTTP calls")
    assert_eq(spent_inside, 2, "the budget reflects BOTH calls, not just the one _call itself decrements for")


# ---------------------------------------------------------------------------
# M1's hard constraint: the hierarchy must never stall
# ---------------------------------------------------------------------------

def test_hierarchy_advances_through_every_tier_when_every_call_is_cut():
    print("\n[test] EVERY summarize call (L1, L2, L3, and every retry) is cut — the hierarchy still advances end to end")
    _wipe()
    before = summarizer.truncated_summary_count()
    # L1=4, L2=3, L3=2 (module env above): _msgs(14) -> 28 turns -> 7 L1
    # chunks -> 2 L2 folds (chapters) -> L3_CHUNK_SIZE=2 reached -> L3 fires,
    # all inside ONE maybe_rollup call, every single one of those calls (and
    # every retry) reporting finish_reason=length.
    calls, orig = _install(lambda body: (CUT, "length"))
    try:
        st = asyncio.run(summarizer.maybe_rollup("cut-never-stalls", _msgs(14), "http://stub:8000", "m"))
    finally:
        _restore(orig)
    assert_eq(st["last_summarized_turn"], 28, "L1 advanced through all 7 due chunks, not just the first")
    assert_true(len(st.get("l2") or []) >= 1 or st.get("l3") is not None,
                "L2 rolled up at least one chapter (or L3 already consumed it)")
    assert_true(st.get("l3") is not None, "L3 refreshed — the cascade reached the top tier")
    assert_eq(st["l2"], [], "L3 consumed and cleared l2, exactly as an untruncated cascade would")
    assert_true(
        st["l3"]["text"].rstrip().endswith((".", "!", "?")),
        "the stored L3 text ends on real punctuation too — trimming applies at every tier, not just L1",
    )
    assert_true(
        summarizer.truncated_summary_count() > before + 1,
        "more than one unit (spanning L1, L2 and L3) was cut-and-trimmed, proving no tier silently refused and stalled",
    )


def main():
    tests = [
        test_control_stop_reply_unchanged,
        test_cut_reply_retried_and_still_cut_is_trimmed_not_stored_verbatim,
        test_cut_reply_whose_retry_finishes_is_stored_untrimmed,
        test_cut_reply_with_no_sentence_boundary_falls_back_to_word_boundary,
        test_only_a_genuinely_empty_reply_is_refused,
        test_bullet_list_cut_on_both_attempts_falls_back_to_a_line_boundary,
        test_when_both_cut_the_longer_trimmed_candidate_is_kept_not_automatically_the_retry,
        test_exhausted_budget_skips_the_retry_and_trims_the_first_attempt,
        test_budget_accounting_includes_the_retry_call,
        test_hierarchy_advances_through_every_tier_when_every_call_is_cut,
    ]
    for t in tests:
        t()
    print("\nAll v3194-mem cut-summary tests passed.")


if __name__ == "__main__":
    main()
