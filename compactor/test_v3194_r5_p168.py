"""
v3.1.9.4 R5 (hostile pass #16 follow-up): P16-8.

Two capture points read `current_wipe_generation` too late.

1. `main._run_memory_tail` used to read `current_wipe_generation(conv_id)`
   itself, INSIDE its own body, at the moment it is CALLED — which for the
   streaming call site is the generator's `finally:`, after the whole
   reply has finished streaming (up to a minute or more later). A /forget
   that lands and finishes during that window is invisible to the check
   this value feeds (`_wipe_generation_stale`): the "baseline" itself
   already reflects the wipe by the time it is read, so the later
   comparison inside the tail/rollup finds no mismatch and proceeds to
   extract/roll up from the pre-wipe exchange anyway.

   Fix: `chat_completions` now captures the generation once, right after
   conv_id resolution (`_tail_wipe_generation_snapshot`), and both call
   sites of `_run_memory_tail` pass it through explicitly. This file
   proves the MECHANISM: `_run_memory_tail` threads whatever
   `wipe_generation` it is given straight through to
   `_rollup_hierarchy`'s own `wipe_generation=` keyword (spied via
   `_fire_and_forget`, the same technique test_v3194_r4_w2.py uses for
   backfill) — an early (pre-wipe) snapshot makes the downstream check see
   a mismatch after a wipe; a value read fresh at call time (what
   `wipe_generation=None`'s fallback does, kept only for this file's own
   test doubles) does not.

2. `backfill.start_backfill_if_needed` captured
   `current_wipe_generation(conv_id)` AFTER the `redact` await rather than
   before it (up to ~0.7s at 1,000 turns) — a /forget landing during that
   await is invisible for the identical reason: the capture already
   reflects it. Fix: capture right after `raw_snapshot` is taken, before
   the `redact` await. This file drives the REAL
   `start_backfill_if_needed` -> `_run_backfill` path end to end with a
   `redact` stub that bumps the wipe generation partway through (standing
   in for a concurrent /forget), and shows the backfill discards itself at
   the FIRST locked check inside `_run_backfill` (before spending a single
   extraction call) rather than proceeding as if nothing happened.

Synthetic fixtures only. Run: python test_v3194_r5_p168.py
"""

import asyncio
import os
import shutil
import sys
import tempfile

_TMP = tempfile.mkdtemp(prefix="compactor-test-v3194-r5-p168-")
os.environ["COMPACTOR_STORAGE_ROOT"] = _TMP
os.environ.setdefault("MODEL_REPO", "test-model")
os.environ.setdefault("VLLM_URL", "http://stub:8000")
os.environ["COMPACTOR_RAG_ENABLED"] = "false"
os.environ["COMPACTOR_FACTS_EXTRACTION"] = "true"
os.environ["COMPACTOR_HIERARCHICAL_SUMMARY"] = "false"

import backfill  # noqa: E402
import facts  # noqa: E402
import main  # noqa: E402
import memory  # noqa: E402

FAILED: list[str] = []


def check(cond, label):
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"FAIL {label}")
        FAILED.append(label)


def _wipe_storage():
    if os.path.exists(_TMP):
        shutil.rmtree(_TMP)
    memory.ensure_storage_layout()


def _msgs(n_pairs: int) -> list[dict]:
    out = [{"role": "system", "content": "sys"}]
    for i in range(n_pairs):
        out.append({"role": "user", "content": f"user turn {i} with enough real text to extract from"})
        out.append({"role": "assistant", "content": f"assistant reply {i} with enough real text to extract from"})
    return out


# ---------------------------------------------------------------------------
# 1. _run_memory_tail threads its `wipe_generation` argument straight
#    through to _rollup_hierarchy, rather than reading a fresh value at its
#    own call time — the mechanism the request-input-time capture depends on.
# ---------------------------------------------------------------------------

def test_run_memory_tail_uses_the_generation_it_is_given_not_a_fresh_read():
    print("\n[test] _run_memory_tail threads an EXPLICIT wipe_generation "
          "through to _rollup_hierarchy rather than recomputing it at its "
          "own (possibly much later) call time")
    _wipe_storage()
    cid = "p168-tail-explicit"

    captured = {}

    def _spy_fire_and_forget(coro, label=None):
        # A normal, finished, non-empty reply goes through _async_tail
        # (label "tail conv=..."), which takes wipe_generation via the
        # _tail_wipe_generation CONTEXTVAR (set right before this call,
        # reset in a `finally` right after — see main.py's own comment on
        # why _async_tail cannot take it as a plain keyword: three test
        # doubles in this suite replace that function wholesale). Read the
        # contextvar here, synchronously, before returning — this spy runs
        # inside the `try:` that surrounds the real `.set()`/`.reset()`.
        if label and str(label).startswith("tail"):
            captured["wipe_generation"] = main._tail_wipe_generation.get()
        coro.close()
        return True

    early_snapshot = memory.current_wipe_generation(cid)  # 0, "at request-input time"
    # Simulate a /forget landing AFTER the request's inputs were captured
    # but BEFORE _run_memory_tail is finally called (the streaming case:
    # the reply is still streaming when the wipe completes).
    memory.bump_wipe_generation(cid)
    check(memory.current_wipe_generation(cid) != early_snapshot,
          "sanity: the generation moved after the early snapshot was taken")

    orig_ff = main._fire_and_forget
    main._fire_and_forget = _spy_fire_and_forget
    try:
        main._run_memory_tail(
            cid, "a real assistant reply with plenty of content in it",
            finished=True, truncated=False, holed=False,
            touched_facts=[], last_user_text="a real user turn",
            turn_index=4, messages=_msgs(3), injected_facts=None,
            wipe_generation=early_snapshot,
        )
    finally:
        main._fire_and_forget = orig_ff

    check(captured.get("wipe_generation") == early_snapshot,
          f"the EARLY (pre-wipe) snapshot reached _rollup_hierarchy, not a "
          f"fresh read: got {captured.get('wipe_generation')!r}, "
          f"expected {early_snapshot!r}")
    # And the downstream staleness check (the same one _async_tail/
    # _rollup_hierarchy actually use) agrees this is stale — a wipe DID
    # land after this snapshot.
    check(main._wipe_generation_stale(cid, captured.get("wipe_generation")) is True,
          "the threaded-through value is correctly seen as stale")


def test_control_no_wipe_generation_threads_through_as_current():
    print("\n[test] CONTROL: no wipe in between — the snapshot IS the "
          "current generation, and the downstream check agrees nothing "
          "is stale")
    _wipe_storage()
    cid = "p168-tail-control"
    snapshot = memory.current_wipe_generation(cid)

    captured = {}

    def _spy_fire_and_forget(coro, label=None):
        if label and str(label).startswith("tail"):
            captured["wipe_generation"] = main._tail_wipe_generation.get()
        coro.close()
        return True

    orig_ff = main._fire_and_forget
    main._fire_and_forget = _spy_fire_and_forget
    try:
        main._run_memory_tail(
            cid, "a real assistant reply with plenty of content in it",
            finished=True, truncated=False, holed=False,
            touched_facts=[], last_user_text="a real user turn",
            turn_index=4, messages=_msgs(3), injected_facts=None,
            wipe_generation=snapshot,
        )
    finally:
        main._fire_and_forget = orig_ff

    check(captured.get("wipe_generation") == snapshot, f"snapshot threaded through: {captured.get('wipe_generation')!r}")
    check(main._wipe_generation_stale(cid, captured.get("wipe_generation")) is False,
          "not stale — no wipe happened")


def test_a_late_fresh_read_would_have_hidden_the_wipe():
    print("\n[test] contrast: reading the generation FRESH at "
          "_run_memory_tail's own call time (wipe_generation=None, the "
          "pre-fix shape for a caller that captured nothing earlier) "
          "misses a wipe that already completed by then — this is "
          "exactly why the capture had to move earlier, not why None is "
          "unsafe by itself: real callers always pass a snapshot now")
    _wipe_storage()
    cid = "p168-tail-late-read"
    memory.bump_wipe_generation(cid)  # the wipe already "happened"

    captured = {}

    def _spy_fire_and_forget(coro, label=None):
        if label and str(label).startswith("tail"):
            captured["wipe_generation"] = main._tail_wipe_generation.get()
        coro.close()
        return True

    orig_ff = main._fire_and_forget
    main._fire_and_forget = _spy_fire_and_forget
    try:
        main._run_memory_tail(
            cid, "a real assistant reply with plenty of content in it",
            finished=True, truncated=False, holed=False,
            touched_facts=[], last_user_text="a real user turn",
            turn_index=4, messages=_msgs(3), injected_facts=None,
            wipe_generation=None,  # no earlier snapshot at all
        )
    finally:
        main._fire_and_forget = orig_ff

    # A None caller reads current_wipe_generation right now, which by now
    # already reflects the wipe — so it reads as NOT stale, even though a
    # wipe genuinely happened before this call. This is the trap the fix
    # closes by never letting real request traffic reach this fallback.
    check(main._wipe_generation_stale(cid, captured.get("wipe_generation")) is False,
          f"a fresh-at-call-time read cannot see a wipe that already "
          f"completed by call time: {captured.get('wipe_generation')!r}")


# ---------------------------------------------------------------------------
# 2. backfill.start_backfill_if_needed captures before the redact await,
#    not after.
# ---------------------------------------------------------------------------

class _CountingExtraction:
    def __init__(self):
        self.calls = 0

    async def __call__(self, client, vllm_url, model, u, a, existing, **kw):
        self.calls += 1
        return [f"fact #{self.calls}"]


async def _wipe_during_redact(cid, messages):
    """Stand-in for the real redactor: bumps the wipe generation (a
    concurrent /forget finishing) DURING what production spends up to
    ~0.7s on, then returns messages unchanged. start_backfill_if_needed is
    `async def` and awaits `run_in_threadpool(redact, messages)` — redact
    itself is a plain sync callable there, so the bump has to happen
    before this coroutine is even scheduled; done here directly since the
    ordering under test is "capture then redact", not the thread hop
    itself."""
    memory.bump_wipe_generation(cid)
    return messages


def test_backfill_capture_before_redact_catches_a_wipe_during_redact():
    print("\n[test] a wipe landing DURING the redact step is caught: the "
          "capture must happen before the redact await, not after")
    _wipe_storage()
    cid = "p168-backfill-redact-wipe"
    msgs = _msgs(6)

    tasks = []

    def ff(coro, label=None):
        tasks.append(asyncio.ensure_future(coro))
        return True

    def _sync_redact(messages):
        # start_backfill_if_needed calls `redact` via run_in_threadpool,
        # which requires a plain (non-async) callable. Do the bump here,
        # synchronously, standing in for the wipe finishing while the real
        # (threaded) redaction work is in flight.
        memory.bump_wipe_generation(cid)
        return messages

    ex = _CountingExtraction()
    orig_ex = facts.extract_facts_from_exchange
    facts.extract_facts_from_exchange = ex
    try:
        started = asyncio.run(backfill.start_backfill_if_needed(
            cid, msgs, "http://stub", "m", fire_and_forget=ff, redact=_sync_redact,
        ))
        check(started is True, "a backfill was started (the decision to run predates the wipe)")

        async def _drain():
            for t in tasks:
                await t
        asyncio.run(_drain())
    finally:
        facts.extract_facts_from_exchange = orig_ex

    check(ex.calls == 0,
          f"zero extraction calls — the FIRST locked check inside "
          f"_run_backfill caught the mismatch before spending any "
          f"(got {ex.calls})")
    state = backfill.read_state(cid)
    check(state is not None and state.get("state") == "wiped",
          f"record reads 'wiped': {state!r}")


def test_control_backfill_no_wipe_completes_normally():
    print("\n[test] CONTROL: redact runs with no wipe in between — the "
          "backfill completes normally")
    _wipe_storage()
    cid = "p168-backfill-control"
    msgs = _msgs(6)

    tasks = []

    def ff(coro, label=None):
        tasks.append(asyncio.ensure_future(coro))
        return True

    def _noop_redact(messages):
        return messages

    ex = _CountingExtraction()
    orig_ex = facts.extract_facts_from_exchange
    facts.extract_facts_from_exchange = ex
    try:
        started = asyncio.run(backfill.start_backfill_if_needed(
            cid, msgs, "http://stub", "m", fire_and_forget=ff, redact=_noop_redact,
        ))
        check(started is True, "a backfill was started")

        async def _drain():
            for t in tasks:
                await t
        asyncio.run(_drain())
    finally:
        facts.extract_facts_from_exchange = orig_ex

    check(ex.calls == 6, f"all 6 exchanges extracted: {ex.calls}")
    state = backfill.read_state(cid)
    check(state is not None and state.get("state") == "complete", f"record reads 'complete': {state!r}")


def _all_tests():
    return [
        test_run_memory_tail_uses_the_generation_it_is_given_not_a_fresh_read,
        test_control_no_wipe_generation_threads_through_as_current,
        test_a_late_fresh_read_would_have_hidden_the_wipe,
        test_backfill_capture_before_redact_catches_a_wipe_during_redact,
        test_control_backfill_no_wipe_completes_normally,
    ]


if __name__ == "__main__":
    for t in _all_tests():
        t()
    print("\nRESULT:", "all v3.1.9.4 R5 P16-8 checks passed" if not FAILED else f"{len(FAILED)} check(s) FAILED: {FAILED}")
    sys.exit(0 if not FAILED else 1)
