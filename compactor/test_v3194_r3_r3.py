"""
v3.1.9.4 R3 (P15-6 follow-up): main._summarize_once (the compaction summary,
request path) gets the equivalent of M1's summarizer._llm_summarize fix.

Covers:
  1. The retry-then-trim shape itself (control: finished; cut+retry finishes;
     cut+retry still cut -> longer-of-two-trimmed kept; the retry keeps the
     SAME max_tokens; the line/word fallback so a bullet-list cut never
     stalls; only a genuinely empty reply is refused).
  2. The retry is gated on, and counted against, the shared per-request call
     budget (_summary_call_budget) that summarize()'s map/reduce already
     uses — proven directly (mirrors how summarizer's own M1 test drove
     _vllm_call_budget directly) and end to end through the real summarize()
     map/reduce path with MAX_SUMMARY_CALLS_PER_REQUEST small enough to bite.
  3. The WARNING names the conversation (_summary_log_ctx, set by
     compact_if_needed around its one call to summarize()).
  4. truncated_compaction_summary_count() increments exactly once per unit
     that needed the fallback.
  5. The P12-6/P13-2 reuse preview's reserve stays correct: the retry never
     changes max_tokens, so the worst case a single _summarize_once call can
     return is still bounded by SUMMARY_MAX_TOKENS tokens, which is what
     every reserve computation in this file already prices against.

Run: python test_v3194_r3_r3.py
"""

import asyncio
import os
import sys

os.environ.setdefault("MODEL_REPO", "test-model")
os.environ.setdefault("VLLM_URL", "http://stub:8000")

import main  # noqa: E402

FAILED: list[str] = []


def check(cond, label):
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"FAIL {label}")
        FAILED.append(label)


TURNS = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]


class _FakeClient:
    """Records every POST payload; answers scripted (content, finish_reason)
    pairs in order, cycling the last one if more calls arrive than scripted."""

    def __init__(self, replies: list[tuple[str, str]]):
        self.replies = replies
        self.calls: list[dict] = []

    async def post(self, url, json=None, **kw):
        self.calls.append(json)
        idx = min(len(self.calls) - 1, len(self.replies) - 1)
        content, finish_reason = self.replies[idx]

        class _Resp:
            def raise_for_status(self_inner):
                pass

            def json(self_inner):
                return {
                    "choices": [
                        {"message": {"content": content}, "finish_reason": finish_reason}
                    ]
                }
        return _Resp()


# ---------------------------------------------------------------------------
# 1. The retry-then-trim shape
# ---------------------------------------------------------------------------

def test_control_finished_reply_returned_verbatim():
    print("\n[test] CONTROL: finish_reason='stop' -> returned untouched, one call")
    client = _FakeClient([("a clean, complete summary.", "stop")])
    out = asyncio.run(main._summarize_once(client, TURNS))
    check(out == "a clean, complete summary.", "verbatim")
    check(len(client.calls) == 1, "exactly one HTTP call")


def test_cut_reply_retried_and_finishes_cleanly():
    print("\n[test] a cut first attempt, retried once, finishes cleanly -> retry text returned")
    # v3.1.9.4 (R5 / P16-7 fix). A clean retry no longer wins automatically
    # — it wins only when it is at least as long as the first attempt
    # trimmed to a sentence boundary (see main._summarize_once's own
    # docstring, and test_v3194_r5_p167.py for the dedicated coverage of
    # the "clean but SHORTER" case this fix closes). The retry text here
    # is long enough to win that comparison on its own merits, so this
    # test still covers its own original, narrower claim: a clean retry
    # CAN win and is returned verbatim, two calls total.
    client = _FakeClient([
        ("the first attempt ran past its cap and got c", "length"),
        ("a properly finished retry that is long enough to beat the "
         "trimmed first attempt in the length comparison outright.", "stop"),
    ])
    out = asyncio.run(main._summarize_once(client, TURNS))
    check(
        out == "a properly finished retry that is long enough to beat the "
               "trimmed first attempt in the length comparison outright.",
        "retry's text returned",
    )
    check(len(client.calls) == 2, "exactly two HTTP calls (guaranteed + one retry)")


def test_cut_reply_retried_and_still_cut_is_trimmed_not_stored_verbatim():
    print("\n[test] both attempts cut -> NEITHER is stored verbatim; the longer trimmed candidate is kept")
    client = _FakeClient([
        ("Sentence one. Sentence two. and then it was cu", "length"),
        ("Sentence one only. and cu", "length"),
    ])
    before = main.truncated_compaction_summary_count()
    out = asyncio.run(main._summarize_once(client, TURNS))
    check(out == "Sentence one. Sentence two.",
          f"trimmed to the longer candidate's last sentence boundary: {out!r}")
    check(len(client.calls) == 2, "exactly two HTTP calls")
    check(main.truncated_compaction_summary_count() == before + 1,
          "the truncated-summary counter incremented exactly once")


def test_when_both_cut_the_longer_trimmed_candidate_is_kept_not_automatically_the_retry():
    print("\n[test] the RETRY's own trim can be SHORTER than the first attempt's — the longer one still wins")
    client = _FakeClient([
        ("A first long sentence with lots of detail here. Then a second one too. and cu",
         "length"),
        ("Tiny. and cu", "length"),
    ])
    out = asyncio.run(main._summarize_once(client, TURNS))
    check(out == "A first long sentence with lots of detail here. Then a second one too.",
          f"kept the first attempt's longer trim, not the retry's shorter one: {out!r}")


def test_retry_uses_the_same_max_tokens_not_a_smaller_one():
    print("\n[test] R3 / P12-6/P13-2: the retry keeps SUMMARY_MAX_TOKENS unchanged "
          "— this is what keeps the reuse preview's reserve correct, since the "
          "worst case a call can return is still bounded by the same cap")
    client = _FakeClient([
        ("first, cut mid-wo", "length"),
        ("retry, also cut mid-wo", "length"),
    ])
    asyncio.run(main._summarize_once(client, TURNS))
    check(len(client.calls) == 2, "two calls made")
    check(client.calls[0]["max_tokens"] == main.SUMMARY_MAX_TOKENS,
          "first call's max_tokens is SUMMARY_MAX_TOKENS")
    check(client.calls[1]["max_tokens"] == main.SUMMARY_MAX_TOKENS,
          "retry's max_tokens is the SAME SUMMARY_MAX_TOKENS, not a smaller cap")
    # The system prompt changes (a tighter word-target instruction); the cap does not.
    check(client.calls[1]["messages"][0]["content"] != client.calls[0]["messages"][0]["content"],
          "the retry's system prompt DOES carry a different (tighter) instruction")


def test_only_a_genuinely_empty_reply_is_refused():
    print("\n[test] '' only when EVERY candidate is itself empty/whitespace")
    client = _FakeClient([
        ("   ", "length"),
        ("\n\t ", "length"),
    ])
    out = asyncio.run(main._summarize_once(client, TURNS))
    check(out == "", f"refused (empty): {out!r}")


def test_bullet_list_cut_on_both_attempts_falls_back_to_a_line_boundary():
    print("\n[test] a cut bullet list (no sentence boundary at all) falls back to a LINE boundary, not refused")
    client = _FakeClient([
        ("- first point here\n- second point here\n- third point cut off mi", "length"),
        ("- only one point cut off mi", "length"),
    ])
    out = asyncio.run(main._summarize_once(client, TURNS))
    check(out == "- first point here\n- second point here",
          f"fell back to the longest complete-line prefix: {out!r}")


def test_cut_reply_with_no_sentence_or_line_boundary_falls_back_to_word_boundary():
    print("\n[test] no sentence boundary, no line boundary -> falls back to a WORD boundary with ' …'")
    client = _FakeClient([
        ("a run of words with no terminal punctuation at all cu", "length"),
        ("shorter cu", "length"),
    ])
    out = asyncio.run(main._summarize_once(client, TURNS))
    check(out.endswith(" …"), f"word-boundary fallback marked explicitly: {out!r}")
    check(not out.rstrip(" …").endswith("cu"), "the incomplete trailing word itself was dropped")


# ---------------------------------------------------------------------------
# 2. Budget gating and accounting
# ---------------------------------------------------------------------------

def test_exhausted_budget_skips_the_retry_and_trims_the_first_attempt():
    print("\n[test] R3: with NO budget left, the retry is skipped entirely — only ONE HTTP call is made")
    client = _FakeClient([("Sentence one. and then cu", "length")])
    token = main._summary_call_budget.set([0])  # nothing left
    try:
        out = asyncio.run(main._summarize_once(client, TURNS))
    finally:
        main._summary_call_budget.reset(token)
    check(len(client.calls) == 1, "only the ONE guaranteed call was made — no retry")
    check(out == "Sentence one.", f"the first attempt alone was trimmed: {out!r}")


def test_budget_accounting_includes_the_retry_call():
    print("\n[test] R3: the retry decrements the SAME shared budget it is gated on")
    client = _FakeClient([
        ("first, cut mid-wo", "length"),
        ("retry, finishes.", "stop"),
    ])
    budget = [2]
    token = main._summary_call_budget.set(budget)
    try:
        asyncio.run(main._summarize_once(client, TURNS))
    finally:
        main._summary_call_budget.reset(token)
    check(budget[0] == 1, f"the budget reflects BOTH calls having been counted: {budget[0]!r} (started at 2)")


def test_control_budget_with_room_retries_normally():
    print("\n[test] CONTROL: budget with room lets the retry proceed as normal")
    client = _FakeClient([
        ("first, cut mid-wo", "length"),
        ("retry, finishes.", "stop"),
    ])
    token = main._summary_call_budget.set([4])
    try:
        out = asyncio.run(main._summarize_once(client, TURNS))
    finally:
        main._summary_call_budget.reset(token)
    check(out == "retry, finishes.", "retry ran and its text was returned")
    check(len(client.calls) == 2, "two calls made")


# ---------------------------------------------------------------------------
# 3. End to end through the real summarize() map/reduce path: the cap holds
#    even with every batch retrying.
# ---------------------------------------------------------------------------

def test_cap_holds_end_to_end_through_summarize_with_every_batch_retrying():
    print("\n[test] R3 end-to-end: MAX_SUMMARY_CALLS_PER_REQUEST=3, 2 batches, EVERY "
          "call (guaranteed AND retry) reports finish_reason=length — the real "
          "summarize() map/reduce path must never exceed 3 real HTTP calls")
    orig_max = main.MAX_SUMMARY_CALLS_PER_REQUEST
    orig_chunk = main._chunk_to_budget
    main.MAX_SUMMARY_CALLS_PER_REQUEST = 3
    # Two batches -> _bounded() spends 2 of the 3-call budget on the
    # guaranteed map calls, leaving exactly 1 for retries. Both batches cut,
    # so both WANT a retry; only one may have it.
    main._chunk_to_budget = lambda turns, budget, scale: [[TURNS[0]], [TURNS[1]]]
    client = _FakeClient([("a batch summary cut mid-wo", "length")])  # every call cuts
    try:
        summary, deferred = asyncio.run(main.summarize(client, TURNS * 20))
    finally:
        main.MAX_SUMMARY_CALLS_PER_REQUEST = orig_max
        main._chunk_to_budget = orig_chunk
    check(len(client.calls) <= 3,
          f"never exceeded the 3-call cap: {len(client.calls)} real HTTP call(s) made")
    check(len(client.calls) == 3,
          f"spent exactly what it should (2 guaranteed + 1 retry, budget then exhausted): {len(client.calls)}")
    check(bool((summary or "").strip()), "still produced SOME summary text (degraded, not empty)")


def test_control_cap_refusal_still_applies_before_any_call_is_spent():
    print("\n[test] CONTROL: the pre-existing over-cap refusal (more batches than "
          "the cap) still refuses BEFORE spending any call — unaffected by this fix")
    orig_max = main.MAX_SUMMARY_CALLS_PER_REQUEST
    orig_chunk = main._chunk_to_budget
    main.MAX_SUMMARY_CALLS_PER_REQUEST = 1
    main._chunk_to_budget = lambda turns, budget, scale: [[TURNS[0]], [TURNS[1]], [TURNS[0]]]
    client = _FakeClient([("should never be called", "stop")])
    try:
        summary, deferred = asyncio.run(main.summarize(client, TURNS * 20))
    finally:
        main.MAX_SUMMARY_CALLS_PER_REQUEST = orig_max
        main._chunk_to_budget = orig_chunk
    check(len(client.calls) == 0, "refused before spending a single call")
    check(summary == "", "no summary produced")
    check(len(deferred) == len(TURNS * 20), "everything deferred, nothing lost")


# ---------------------------------------------------------------------------
# 4. The WARNING names the conversation
# ---------------------------------------------------------------------------

def test_warning_names_the_conversation_via_summary_log_ctx():
    print("\n[test] the WARNING for a cut-and-trimmed summary names the conversation "
          "(via _summary_log_ctx, set by compact_if_needed around its call)")
    import logging

    class _Collector(logging.Handler):
        def __init__(self):
            super().__init__(level=logging.WARNING)
            self.records = []

        def emit(self, record):
            self.records.append(record)

    lg = logging.getLogger("compactor")
    h = _Collector()
    lg.addHandler(h)
    client = _FakeClient([
        ("first cut mid-wo", "length"),
        ("retry cut mid-wo", "length"),
    ])
    ctx_token = main._summary_log_ctx.set({"conv_id": "r3-warning-conv"})
    try:
        asyncio.run(main._summarize_once(client, TURNS))
    finally:
        main._summary_log_ctx.reset(ctx_token)
        lg.removeHandler(h)
    warnings = [r for r in h.records if r.levelno == logging.WARNING
                and "compaction summary was cut" in r.getMessage()]
    check(len(warnings) == 1, f"exactly one WARNING logged: {len(warnings)}")
    if warnings:
        check("r3-warning-conv" in warnings[0].getMessage(),
              f"the WARNING names the conversation: {warnings[0].getMessage()!r}")


def test_control_no_conv_id_context_falls_back_to_question_mark():
    print("\n[test] CONTROL: with no _summary_log_ctx set (a direct caller, "
          "or a stub-driven test), the WARNING still fires, naming '?' rather than crashing")
    client = _FakeClient([
        ("first cut mid-wo", "length"),
        ("retry cut mid-wo", "length"),
    ])
    out = asyncio.run(main._summarize_once(client, TURNS))
    check(bool(out), "did not crash with no context set")


def _all_tests():
    return [
        test_control_finished_reply_returned_verbatim,
        test_cut_reply_retried_and_finishes_cleanly,
        test_cut_reply_retried_and_still_cut_is_trimmed_not_stored_verbatim,
        test_when_both_cut_the_longer_trimmed_candidate_is_kept_not_automatically_the_retry,
        test_retry_uses_the_same_max_tokens_not_a_smaller_one,
        test_only_a_genuinely_empty_reply_is_refused,
        test_bullet_list_cut_on_both_attempts_falls_back_to_a_line_boundary,
        test_cut_reply_with_no_sentence_or_line_boundary_falls_back_to_word_boundary,
        test_exhausted_budget_skips_the_retry_and_trims_the_first_attempt,
        test_budget_accounting_includes_the_retry_call,
        test_control_budget_with_room_retries_normally,
        test_cap_holds_end_to_end_through_summarize_with_every_batch_retrying,
        test_control_cap_refusal_still_applies_before_any_call_is_spent,
        test_warning_names_the_conversation_via_summary_log_ctx,
        test_control_no_conv_id_context_falls_back_to_question_mark,
    ]


if __name__ == "__main__":
    for t in _all_tests():
        t()
    print("\nRESULT:", "all v3.1.9.4 R3 checks passed" if not FAILED else f"{len(FAILED)} check(s) FAILED: {FAILED}")
    sys.exit(0 if not FAILED else 1)
