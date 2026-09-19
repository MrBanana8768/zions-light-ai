"""
v3.1.9.4 R5 (hostile pass #16 follow-up): P16-4.

`_summarize_once`'s cut-summary retry (inside the map phase's `_bounded`)
drew on the SAME shared per-request call budget (`_summary_call_budget`)
the map phase's own guaranteed-one-call-per-batch promise relies on. With
`COMPACTOR_MAX_SUMMARY_CALLS` set above the map semaphore's 4 and that
many batches, one batch's retry could spend a call a LATER, not-yet-
started batch needed for its OWN first call — that batch then found the
shared budget empty, returned "" for a reason that had nothing to do with
the model's output, and "ANY empty map batch fails the whole summarize"
forwarded every turn in the compaction verbatim, burning every call in
the budget for nothing. The default cap (4) was never exposed: the
semaphore already caps concurrent batches at 4, so 4 batches consume
exactly the default budget with nothing left for even one retry.

Fix: `_bounded` tracks how many batches have not yet taken their own
guaranteed call (`_batches_pending`) and passes it to `_summarize_once` as
`pending_others`. The retry may only spend a budget call when doing so
still leaves at least that many calls in the shared budget.

This is the reviewer's own reproduction (`SP\\p16\\test_p16_map_retry_budget.py`),
formalized with assertions against `main.summarize` directly (real map/
reduce code, `_chunk_to_budget`/`count_tokens_exact` stubbed so N turns
become exactly N one-turn batches, and a stub vLLM client that cuts batch
0's first call only). Synthetic content only.

Run: python test_v3194_r5_p164.py
"""

import asyncio
import os
import sys
import tempfile

os.environ.setdefault("MODEL_REPO", "test-model")
os.environ.setdefault("VLLM_URL", "http://stub:8000")
os.environ["COMPACTOR_RAG_ENABLED"] = "false"
os.environ["COMPACTOR_STORAGE_ROOT"] = tempfile.mkdtemp(prefix="compactor-test-v3194-r5-p164-")
import memory  # noqa: E402
memory.ensure_storage_layout()
import main  # noqa: E402

FAILED: list[str] = []


def check(cond, label):
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"FAIL {label}")
        FAILED.append(label)


main.count_tokens_exact = lambda msgs: None
main._chunk_to_budget = lambda turns, budget, scale=1.0: [[t] for t in turns]


class _Resp:
    def __init__(self, content, fr):
        self._d = {"choices": [{"message": {"content": content}, "finish_reason": fr}]}

    def raise_for_status(self):
        pass

    def json(self):
        return self._d


class _Client:
    """batch 0's FIRST call is cut (finish_reason='length') when `cut` is
    True; every other call, including batch 0's own retry, finishes clean.
    """

    def __init__(self, cut: bool):
        self.n = 0
        self.cut = cut

    async def post(self, url, json=None, timeout=None):
        self.n += 1
        k = self.n
        user = json["messages"][-1]["content"]
        is_retry = "previous attempt" in json["messages"][0]["content"]
        if self.cut and "turn-0 " in user and not is_retry:
            await asyncio.sleep(0.01)
            return _Resp("A partial summary that was cut mid", "length")
        await asyncio.sleep(0.02)
        return _Resp(
            f"Summary part {k} finished cleanly with reasonably long "
            f"content describing what happened in this part of the chat.",
            "stop",
        )


def _turns(n):
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": f"turn-{i} synthetic"} for i in range(n)]


def _run(n_batches, max_calls, cut):
    # MAX_SUMMARY_CALLS_PER_REQUEST is read as a plain module attribute at
    # call time (inside summarize()/`_summarize_body`), not only at import
    # — so it can be overridden directly per run without reloading `main`
    # (a reload would re-run its whole module body, including FastAPI
    # route registration, for no benefit here).
    main.MAX_SUMMARY_CALLS_PER_REQUEST = max_calls
    c = _Client(cut)
    summary, deferred = asyncio.run(main.summarize(c, _turns(n_batches)))
    # getattr fallback: this counter is itself new in this fix (P16-7); a
    # reproduction run against the pre-fix tree does not have it, and the
    # bug this file's other assertions demonstrate (empty summary, verbatim
    # forwarding) does not depend on it existing.
    _retried_fn = getattr(main, "retried_compaction_summary_count", None)
    return {
        "calls": c.n,
        "summary_empty": not (summary or "").strip(),
        "deferred": len(deferred),
        "retried": _retried_fn() if _retried_fn is not None else None,
    }


def test_control_default_cap_no_cut():
    print("\n[test] CONTROL: MAX=4 (default), 4 batches, nothing cut — "
          "unaffected by this fix")
    r = _run(4, 4, cut=False)
    check(r["calls"] == 4, f"4 real calls: {r}")
    check(not r["summary_empty"], f"summary produced: {r}")
    check(r["deferred"] == 0, f"nothing deferred: {r}")


def test_control_default_cap_at_the_semaphore_boundary():
    print("\n[test] CONTROL: MAX=4 (default) with a cut batch — the "
          "semaphore and the cap are equal, so there was never slack for "
          "a retry; this is ALREADY-FIXED-AT-HEAD behaviour, unaffected "
          "by this change")
    r = _run(4, 4, cut=True)
    check(r["calls"] == 4, f"4 real calls, no retry possible: {r}")
    check(not r["summary_empty"], f"summary produced (trimmed, not empty): {r}")
    check(r["deferred"] == 0, f"nothing deferred verbatim: {r}")


def test_max_five_cut_batch_no_longer_starves_a_later_batch():
    print("\n[test] MAX=5, 5 batches, batch 0 cut — the retry must not "
          "spend the 5th batch's own guaranteed call")
    r = _run(5, 5, cut=True)
    check(r["calls"] == 5, f"exactly 5 real calls (no wasted extra): {r}")
    check(not r["summary_empty"], f"summary produced, not empty (was empty pre-fix): {r}")
    check(r["deferred"] == 0, f"nothing forwarded verbatim (was all 5 turns pre-fix): {r}")


def test_max_six_cut_batch_no_longer_starves_a_later_batch():
    print("\n[test] MAX=6, 6 batches, batch 0 cut — same shape as the "
          "reviewer's own MAX=6 reproduction")
    r = _run(6, 6, cut=True)
    check(r["calls"] == 6, f"exactly 6 real calls: {r}")
    check(not r["summary_empty"], f"summary produced, not empty (was empty pre-fix): {r}")
    check(r["deferred"] == 0, f"nothing forwarded verbatim (was all 6 turns pre-fix): {r}")


def test_slack_budget_still_allows_a_real_retry():
    print("\n[test] CONTROL: MAX=6 with only 4 batches — genuine slack — "
          "the retry is still allowed to spend a call when no later batch "
          "needs it")
    r = _run(4, 6, cut=True)
    check(r["calls"] == 5, f"4 guaranteed + 1 retry = 5 calls: {r}")
    check(r["retried"] is not None and r["retried"] >= 1, f"the retry counter moved: {r}")
    check(not r["summary_empty"], f"summary produced: {r}")
    check(r["deferred"] == 0, f"nothing deferred: {r}")


# ---------------------------------------------------------------------------
# The identical starvation shape, one level up: the REDUCE phase reuses the
# SAME `_bounded` closure the map phase does (asyncio.gather over groups),
# so a reduce-round group's retry can just as easily starve a SIBLING
# group in that same round — the "fixed at one site, missed the identical
# sibling" defect this codebase keeps paying for. Needs > 4 concurrent
# groups in one reduce round (the semaphore's own capacity) so that at
# least one group is still WAITING on the semaphore — not yet counted as
# having taken its guaranteed call — while an earlier group's retry runs.
# ---------------------------------------------------------------------------

def _phase_aware_chunk(turns, budget, scale=1.0):
    """Map phase (content starts with 'turn-'): one turn per batch, same as
    every other test in this file. Reduce phase (folding partial
    summaries): two parts per group, so a 9-partial reduce round produces
    5 groups — one more than the map semaphore's concurrency cap of 4."""
    if turns and str(turns[0].get("content", "")).startswith("turn-"):
        return [[t] for t in turns]
    groups = []
    i = 0
    while i < len(turns):
        groups.append(turns[i:i + 2])
        i += 2
    return groups


class _ReduceClient:
    """9 map-phase batches, none cut (all finish 'stop'). The reduce
    phase's FIRST group (parts 1-2, i.e. whichever call's transcript
    contains "Summary part 1.") is cut on its first attempt; every other
    call (map or reduce) finishes cleanly."""

    def __init__(self):
        self.n = 0
        self.group0_calls = 0

    async def post(self, url, json=None, timeout=None):
        self.n += 1
        user = json["messages"][-1]["content"]
        is_retry = "previous attempt" in json["messages"][0]["content"]
        if "Summary part 1." in user:
            self.group0_calls += 1
            if self.group0_calls == 1 and not is_retry:
                return _Resp("group 0's first attempt, cut mid-fold", "length")
        if "[user]: turn-" in user:
            k = self.n
            return _Resp(f"Summary part {k}.", "stop")
        return _Resp(
            f"a clean fold of this group, call {self.n}, padded so it "
            f"reads as real content rather than a placeholder string.",
            "stop",
        )


def test_reduce_phase_sibling_group_is_not_starved_by_an_earlier_retry():
    print("\n[test] the REDUCE phase gets the same pending_others "
          "protection as the map phase: a retry in reduce-round group 0 "
          "must not starve group 4's own guaranteed call")
    main.MAX_SUMMARY_CALLS_PER_REQUEST = 14  # 9 (map) + 5 (reduce round 1)
    orig_chunk = main._chunk_to_budget
    main._chunk_to_budget = _phase_aware_chunk
    try:
        c = _ReduceClient()
        summary, deferred = asyncio.run(main.summarize(c, _turns(9)))
    finally:
        main._chunk_to_budget = orig_chunk
    check(bool((summary or "").strip()), f"a non-empty summary was produced: {summary!r}")
    check(deferred == [], f"nothing deferred verbatim: {len(deferred)} turn(s)")
    # The concrete failure mode pre-fix: group 4 (the odd one out, waiting
    # on the semaphore while groups 0-3 run) finds calls_left already at 0
    # by the time it gets a turn, returns "", and the reduce round's own
    # "any empty group -> concatenate instead of folding" guard fires —
    # summary is still non-empty (concatenation, not a total failure like
    # the map phase's), but "Summary part 9." (group 4's own partial) would
    # then appear VERBATIM, unfolded, in the final text, rather than folded
    # together with "Summary part 8." the way every other pair was.
    check("Summary part 9." not in (summary or ""),
          f"group 4 was actually folded (not left unfolded/verbatim, the "
          f"pre-fix starvation signature): {summary!r}")


def _all_tests():
    return [
        test_control_default_cap_no_cut,
        test_control_default_cap_at_the_semaphore_boundary,
        test_max_five_cut_batch_no_longer_starves_a_later_batch,
        test_max_six_cut_batch_no_longer_starves_a_later_batch,
        test_slack_budget_still_allows_a_real_retry,
        test_reduce_phase_sibling_group_is_not_starved_by_an_earlier_retry,
    ]


if __name__ == "__main__":
    for t in _all_tests():
        t()
    print("\nRESULT:", "all v3.1.9.4 R5 P16-4 checks passed" if not FAILED else f"{len(FAILED)} check(s) FAILED: {FAILED}")
    sys.exit(0 if not FAILED else 1)
