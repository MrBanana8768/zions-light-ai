"""
compactor.test_merge_concurrency — v3.1.9.3 (P11 / race-merge.md).

Reproduces, and then guards against, the concurrent-merge data-loss defect
recorded in tests/adversarial/test_adv_race.py::
test_two_merges_into_one_conversation_lose_facts:

    portability.merge_conversation runs on a threadpool worker (main.py:
    `await run_in_threadpool(portability.merge_conversation, ...)`) and its
    only mutual exclusion on the destination was `memory.conv_lock(dst)
    .locked()` — an asyncio.Lock PROBE. A thread can neither take nor wait
    on that lock, so two merges into the same destination both pass the
    probe and both do an unsynchronised load_facts -> _merge_fact_lists ->
    save_facts of the SAME file. Whichever save lands second overwrites the
    first, discarding facts BOTH sides were told (HTTP 200, `facts_added`
    counted) had been added.

These cases drive `portability.merge_conversation` through the SAME
dispatch mechanism production uses — `anyio.to_thread.run_sync` under a
running event loop, which is exactly what `starlette.concurrency.
run_in_threadpool` calls — rather than raw `threading.Thread`. That
matters for two reasons: (1) it is the actual code path being fixed, and
(2) the fix itself (see portability.merge_conversation's dst-lock comment)
bridges back onto that SAME event loop via `anyio.from_thread.run`, which
only works from a genuine AnyIO worker thread. A raw `threading.Thread`
would not reproduce the fix's own synchronisation and would give a false
read on whether it works.

To make the race deterministic (not a rate over many attempts, which is
what the HTTP-level adversarial suite already covers), `facts.save_facts`
is patched with a two-party rendezvous: on UNFIXED code both threads reach
their own independent read+fold and arrive at the save call at roughly the
same time, so the rendezvous is a no-op synchronisation point that simply
confirms they are about to race, and then both write, one clobbering the
other. On FIXED code only one merge is ever inside the critical section at
a time (the other is blocked waiting for conv_lock), so the second thread
cannot reach the save call concurrently — the rendezvous times out
(harmlessly; see `_racy_save_facts`) and the two writes are correctly
serialised instead of racing.

Isolated tmpdir as COMPACTOR_STORAGE_ROOT; retrieval stubbed (no ChromaDB)
exactly as test_portability.py does. No real or conversation-like text
anywhere — every fact string here is a synthetic, clearly-labelled fixture.

Round 2 (coordinator review) adds two more cases for two more defects the
first version of this fix introduced:

  * `test_merge_does_not_block_the_event_loop_while_writing` — the first
    version ran the WHOLE critical section (load_facts, _merge_fact_lists,
    save_facts, the exchanges loop) inside the coroutine bridged onto the
    event loop, not just the lock operations. That runs the merge's file
    I/O ON THE LOOP, stalling every other request in the process for as
    long as it takes — defeating the reason this function is dispatched
    through run_in_threadpool at all.
  * `test_short_hold_serializes_and_both_merges_land` /
    `test_long_hold_times_out_with_the_original_refusal` — the first
    version awaited conv_lock(dst) with NO bound. A summary rebuild can
    hold it for the whole drain (10-30 minutes, the identity runbook's own
    estimate), which would park an admin request that long instead of
    refusing fast the way the original D18 code did.

Run: python test_merge_concurrency.py
"""

from __future__ import annotations

import asyncio
import functools
import os
import shutil
import sys
import tempfile
import threading
import time

# Isolate storage to a tmpdir BEFORE importing memory (same pattern as
# test_portability.py).
_TMP_ROOT = tempfile.mkdtemp(prefix="zions_merge_race_test_")
os.environ["COMPACTOR_STORAGE_ROOT"] = _TMP_ROOT
os.environ["COMPACTOR_RAG_ENABLED"] = "false"  # stub retrieval below

import facts  # noqa: E402
import memory  # noqa: E402
import retrieval  # noqa: E402

# Stub retrieval's ChromaDB integration with a pure-Python dict, same as
# test_portability.py — this suite is about the FACTS race, not embeddings.
_STUB_STORE: dict[str, list[dict]] = {}


def _stub_export(conv_id):
    return list(_STUB_STORE.get(conv_id, []))


def _stub_import(conv_id, turn_index, document):
    _STUB_STORE.setdefault(conv_id, []).append(
        {"turn_index": turn_index, "document": document}
    )
    return True


retrieval.export_indexed_exchanges = _stub_export
retrieval.import_indexed_exchange = _stub_import

import portability  # noqa: E402 — must import after stubs are wired

import anyio  # noqa: E402
import anyio.to_thread  # noqa: E402


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

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


def reset_state(conv_id):
    facts.save_facts(conv_id, [])
    _STUB_STORE.pop(conv_id, None)


def distinct_facts(n: int, tag: str) -> list[str]:
    """N facts that are distinct from each other AND from any other tag's
    set — synthetic, no conversation-like content, safe for a public repo.
    """
    return [f"synthetic fixture fact #{i} for {tag}" for i in range(n)]


def seed_facts(conv_id: str, texts: list[str]) -> None:
    now = int(time.time())
    facts.save_facts(conv_id, [
        {"text": t, "added_turn": 1, "last_used": now, "pin": False}
        for t in texts
    ])


def fact_texts(conv_id: str) -> set[str]:
    return {f["text"] for f in facts.load_facts(conv_id)}


def dispatch_two_merges(src_a: str, src_b: str, dst: str) -> tuple[dict, dict]:
    """Run merge_conversation(src_a -> dst) and merge_conversation(src_b ->
    dst) CONCURRENTLY, through the exact mechanism production uses:
    `anyio.to_thread.run_sync` under a live event loop (what `starlette.
    concurrency.run_in_threadpool` — main.py's actual call — wraps). Real
    OS threads, real `anyio.from_thread` portal, same as production.
    """
    async def _race():
        return await asyncio.gather(
            anyio.to_thread.run_sync(
                functools.partial(portability.merge_conversation, src_a, dst, dry_run=False)
            ),
            anyio.to_thread.run_sync(
                functools.partial(portability.merge_conversation, src_b, dst, dry_run=False)
            ),
        )
    return asyncio.run(_race())


class _RendezvousSave:
    """Wraps facts.save_facts so two concurrent writers to the SAME conv_id
    rendezvous right before writing — see module docstring for what this
    proves on unfixed vs. fixed code. `timeout` is generous enough to
    absorb scheduling jitter but short enough that a properly-serialised
    fixed run (where the second party never shows up concurrently) does
    not make the test slow.
    """

    def __init__(self, conv_id: str, timeout: float = 0.75):
        self._conv_id = conv_id
        self._barrier = threading.Barrier(2, timeout=timeout)
        self._orig = facts.save_facts
        self.calls: list[float] = []  # monotonic timestamps of each actual write

    def __enter__(self):
        outer = self

        def _patched(conv_id, facts_list):
            if conv_id == outer._conv_id:
                try:
                    outer._barrier.wait()
                except threading.BrokenBarrierError:
                    # The other side never arrived concurrently - on fixed
                    # code this is the EXPECTED outcome (the second merge
                    # is still blocked on conv_lock, not racing us).
                    pass
            outer.calls.append(time.monotonic())
            return outer._orig(conv_id, facts_list)

        facts.save_facts = _patched
        return self

    def __exit__(self, *exc):
        facts.save_facts = self._orig


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_two_concurrent_merges_do_not_lose_facts():
    print("\n[test] two merges into the SAME destination, forced to race on "
          "the write, lose none of the facts either side was told it added")
    memory.ensure_storage_layout()
    dst, src_a, src_b = "race-dst-1", "race-src-a-1", "race-src-b-1"
    reset_state(dst)
    reset_state(src_a)
    reset_state(src_b)

    n = 20
    a_texts = distinct_facts(n, "race-a-1")
    b_texts = distinct_facts(n, "race-b-1")
    seed_facts(src_a, a_texts)
    seed_facts(src_b, b_texts)
    seed_facts(dst, ["the destination's own pre-existing seed fact"])

    with _RendezvousSave(dst) as rv:
        res_a, res_b = dispatch_two_merges(src_a, src_b, dst)

    assert_true(len(rv.calls) >= 1, "at least one write actually happened")

    got = fact_texts(dst)
    missing = [t for t in a_texts + b_texts if t not in got]
    assert_eq(
        missing, [],
        f"no acknowledged fact missing after both merges settle "
        f"(facts_added: a={res_a.get('facts_added')}, b={res_b.get('facts_added')}; "
        f"missing={len(missing)} of {2 * n}; stored={len(got)})",
    )
    assert_true(
        "the destination's own pre-existing seed fact" in got,
        "the destination's own pre-existing fact also survived",
    )
    # Both sides must have reported adding their own facts - a fix that
    # silently drops the SECOND merge's contribution but still answers 200
    # would pass the "nothing missing from disk" check above by luck if it
    # also lied about facts_added; this closes that gap.
    assert_eq(res_a.get("facts_added"), n, "merge A reported adding all of its facts")
    assert_eq(res_b.get("facts_added"), n, "merge B reported adding all of its facts")


def test_single_merge_via_threadpool_still_commits():
    print("\n[test] CONTROL: an UNCONTENDED merge dispatched through the "
          "real anyio/threadpool path still commits normally - this is a "
          "serialisation fix, not a new blanket refusal")
    memory.ensure_storage_layout()
    dst, src = "race-dst-ctrl", "race-src-ctrl"
    reset_state(dst)
    reset_state(src)
    texts = distinct_facts(5, "race-control")
    seed_facts(src, texts)
    reset_state(dst)

    async def _one():
        return await anyio.to_thread.run_sync(
            functools.partial(portability.merge_conversation, src, dst, dry_run=False)
        )

    result = asyncio.run(_one())
    assert_eq(result.get("facts_added"), 5, "CONTROL: uncontended merge via the threadpool still commits")
    assert_eq(fact_texts(dst), set(texts), "CONTROL: all facts landed")


def test_merge_serializes_behind_an_async_writer_on_the_destination():
    print("\n[test] a merge dispatched through the threadpool while an "
          "ASYNC writer (simulating /remember, import, or the extraction "
          "tail) holds conv_lock(dst) does not clobber that writer's fact - "
          "closes the gap a merge-vs-merge-only fix would leave (see "
          "test_merge_and_import_into_one_destination / "
          "test_merge_against_a_remember_holding_the_lock in the "
          "adversarial suite, which target exactly this)")
    memory.ensure_storage_layout()
    dst, src = "race-dst-asyncwriter", "race-src-asyncwriter"
    reset_state(dst)
    reset_state(src)
    seed_facts(dst, ["the destination's pre-existing fact"])
    merge_texts = distinct_facts(10, "race-vs-async")
    seed_facts(src, merge_texts)

    order: list[tuple[str, float]] = []
    order_guard = threading.Lock()

    def record(tag):
        with order_guard:
            order.append((tag, time.monotonic()))

    async def _async_writer_holds_lock():
        """Simulates /remember's own documented fix: hold conv_lock(dst)
        across a load-modify-write, exactly like main.py's command handler
        does."""
        async with memory.conv_lock(dst):
            record("async-writer-acquire")
            current = facts.load_facts(dst)
            await asyncio.sleep(0.3)  # simulate the work /remember does under the lock
            current.append({
                "text": "the async writer's own remembered fact",
                "added_turn": 1, "last_used": int(time.time()), "pin": False,
            })
            facts.save_facts(dst, current)
            record("async-writer-release")

    async def _race():
        writer = asyncio.create_task(_async_writer_holds_lock())
        await asyncio.sleep(0.05)  # let the writer actually acquire first
        record("merge-dispatch")
        try:
            merge_result = await anyio.to_thread.run_sync(
                functools.partial(portability.merge_conversation, src, dst, dry_run=False)
            )
            merge_error = None
        except ValueError as e:
            # A REFUSAL loses nothing and is not a hit — same standard the
            # HTTP-level adversarial case uses (test_merge_against_a_
            # remember_holding_the_lock): "a merge that refused with
            # 400/409 lost nothing and is not a hit." Both the old probe
            # (when its timing happens to land inside the hold) and the
            # fixed code's own D18-shaped fallback for a loop-bound caller
            # can legitimately refuse here; only a claimed SUCCESS that then
            # lost the writer's fact is the defect.
            merge_result, merge_error = None, e
        record("merge-return")
        await writer
        return merge_result, merge_error

    result, error = asyncio.run(_race())

    got = fact_texts(dst)
    assert_true(
        "the async writer's own remembered fact" in got,
        f"the async writer's fact was not overwritten by the merge or its "
        f"refusal (got: {sorted(got)[:5]}...)",
    )
    assert_true(
        "the destination's pre-existing fact" in got,
        "the destination's original fact also survived",
    )
    release_t = next(ts for tag, ts in order if tag == "async-writer-release")
    if error is not None:
        print(f"  ok   merge refused rather than racing ({error}) — safe, not a hit")
        return
    missing = [t for t in merge_texts if t not in got]
    assert_eq(missing, [], f"the merge's own facts are all present too (missing {len(missing)})")

    dispatch_t = next(ts for tag, ts in order if tag == "merge-dispatch")
    return_t = next(ts for tag, ts in order if tag == "merge-return")
    assert_true(
        dispatch_t < release_t < return_t,
        f"the merge was dispatched WHILE the async writer held the lock and "
        f"only returned AFTER it released - order={order}",
    )


# ---------------------------------------------------------------------------
# Round 2 (coordinator review): the lock must not carry the WORK onto the
# event loop, and the wait for it must be BOUNDED.
# ---------------------------------------------------------------------------

def test_merge_does_not_block_the_event_loop_while_writing():
    print("\n[test] coordinator review round 2, finding 1: while a merge's "
          "OWN save is slow, a concurrent coroutine on the SAME event loop "
          "(standing in for e.g. GET /health) still gets scheduled promptly "
          "- the file I/O has to run on the worker thread, not the loop")
    memory.ensure_storage_layout()
    dst, src = "loop-block-dst", "loop-block-src"
    reset_state(dst)
    reset_state(src)
    seed_facts(src, distinct_facts(3, "loop-block"))

    slow_s = 1.0
    orig_save = facts.save_facts

    def _slow_save(conv_id, facts_list):
        if conv_id == dst:
            # A real, synchronous, GIL-releasing sleep - stands in for a
            # slow disk write. If this ends up running ON THE EVENT LOOP
            # (the bug), the loop cannot service anything else for its
            # whole duration; if it runs on the worker thread (the fix),
            # the loop is untouched by it.
            time.sleep(slow_s)
        return orig_save(conv_id, facts_list)

    facts.save_facts = _slow_save
    try:
        async def _health_loop(ticks: list[float]) -> None:
            """Stands in for a concurrent request handler that only needs
            the loop to keep turning - e.g. GET /health. Samples
            CONTINUOUSLY across the whole merge (not one measurement after
            a fixed pre-delay - an earlier version of this test did that
            and the pre-delay itself silently absorbed the stall, so the
            actual measurement ran only after the block had already ended
            and always looked fast; see merge_probe4.py / merge_probe5.py
            in this lane's scratchpad report for the isolated repro of that
            mistake). Each tick's OWN wall-clock duration is what matters:
            if the loop is ever stalled, whichever tick is in flight at
            that moment records it, regardless of timing.
            """
            for _ in range(25):
                t0 = time.monotonic()
                await asyncio.sleep(0.03)
                ticks.append(time.monotonic() - t0)

        async def _race():
            ticks: list[float] = []
            merge_task = asyncio.create_task(
                anyio.to_thread.run_sync(
                    functools.partial(portability.merge_conversation, src, dst, dry_run=False)
                )
            )
            health_task = asyncio.create_task(_health_loop(ticks))
            await merge_task
            await health_task
            return ticks

        ticks = asyncio.run(_race())
    finally:
        facts.save_facts = orig_save

    worst = max(ticks)
    assert_true(
        worst < slow_s * 0.5,
        f"the worst of {len(ticks)} concurrent 0.03s health-loop ticks took "
        f"{worst:.3f}s while a merge's save (patched to take {slow_s}s) was "
        f"in flight - the event loop was stalled by the merge's own file "
        f"I/O instead of that work running on the worker thread "
        f"(ticks: {[round(t, 3) for t in ticks]})",
    )


def test_short_hold_serializes_and_both_merges_land():
    print("\n[test] coordinator review round 2, finding 2 (the short side): "
          "a hold on dst well UNDER the acquire timeout serializes - the "
          "merge waits its turn and lands, rather than refusing")
    memory.ensure_storage_layout()
    dst, src = "short-hold-dst", "short-hold-src"
    reset_state(dst)
    reset_state(src)
    seed_facts(dst, ["the short-hold destination's own pre-existing fact"])
    merge_texts = distinct_facts(5, "short-hold")
    seed_facts(src, merge_texts)

    hold_s = 0.15
    test_timeout_s = 1.0  # generous vs. hold_s; tiny vs. the real 10.0s default

    async def _holder():
        async with memory.conv_lock(dst):
            current = facts.load_facts(dst)
            await asyncio.sleep(hold_s)
            current.append({
                "text": "the short-hold writer's own fact",
                "added_turn": 1, "last_used": int(time.time()), "pin": False,
            })
            facts.save_facts(dst, current)

    async def _race():
        holder = asyncio.create_task(_holder())
        await asyncio.sleep(0.02)  # let the holder actually acquire first
        merge_result = await anyio.to_thread.run_sync(
            functools.partial(portability.merge_conversation, src, dst, dry_run=False)
        )
        await holder
        return merge_result

    orig_timeout = portability._MERGE_DST_LOCK_TIMEOUT_S
    portability._MERGE_DST_LOCK_TIMEOUT_S = test_timeout_s
    try:
        result = asyncio.run(_race())
    finally:
        portability._MERGE_DST_LOCK_TIMEOUT_S = orig_timeout

    got = fact_texts(dst)
    assert_eq(
        result.get("facts_added"), len(merge_texts),
        "the merge landed all of its facts (it waited, not refused)",
    )
    missing = [t for t in merge_texts if t not in got]
    assert_eq(missing, [], "none of the merge's facts are missing")
    assert_true("the short-hold writer's own fact" in got, "the holder's own fact also landed")
    assert_true(
        "the short-hold destination's own pre-existing fact" in got,
        "the destination's original fact survived",
    )


def test_long_hold_times_out_with_the_original_refusal():
    print("\n[test] coordinator review round 2, finding 2 (the bound "
          "itself): a hold on dst well OVER the acquire timeout is refused "
          "with the ORIGINAL D18 message, promptly - not parked until the "
          "holder eventually releases (the identity runbook's summary-"
          "rebuild shape: 10-30 minutes for real, simulated short here)")
    memory.ensure_storage_layout()
    dst, src = "long-hold-dst", "long-hold-src"
    reset_state(dst)
    reset_state(src)
    seed_facts(dst, ["the long-hold destination's own pre-existing fact"])
    merge_texts = distinct_facts(5, "long-hold")
    seed_facts(src, merge_texts)

    hold_s = 0.8
    test_timeout_s = 0.15  # << hold_s - the merge must give up long before the holder is done

    async def _holder():
        async with memory.conv_lock(dst):
            await asyncio.sleep(hold_s)

    async def _race():
        holder = asyncio.create_task(_holder())
        await asyncio.sleep(0.02)  # let the holder actually acquire first
        t0 = time.monotonic()
        try:
            await anyio.to_thread.run_sync(
                functools.partial(portability.merge_conversation, src, dst, dry_run=False)
            )
            merge_error = None
        except ValueError as e:
            merge_error = e
        elapsed = time.monotonic() - t0
        await holder  # let it finish cleanly before the loop shuts down
        return merge_error, elapsed

    orig_timeout = portability._MERGE_DST_LOCK_TIMEOUT_S
    portability._MERGE_DST_LOCK_TIMEOUT_S = test_timeout_s
    try:
        error, elapsed = asyncio.run(_race())
    finally:
        portability._MERGE_DST_LOCK_TIMEOUT_S = orig_timeout

    assert_true(error is not None, "the merge was refused (ValueError), not left to hang")
    assert_true(
        "has a memory write in flight" in str(error) and "Retry in a moment" in str(error),
        f"the ORIGINAL D18-style refusal message is unchanged (got: {error})",
    )
    assert_true(
        elapsed < hold_s * 0.5,
        f"the refusal arrived close to the {test_timeout_s}s bound "
        f"({elapsed:.3f}s), not after waiting out the whole {hold_s}s hold",
    )
    got = fact_texts(dst)
    landed = [t for t in merge_texts if t in got]
    assert_eq(landed, [], "NOTHING from the refused merge was written to dst")


def test_merge_failure_releases_the_dst_lock():
    print("\n[test] P12-3 (hostile pass #12): a merge whose COMMIT raises "
          "(save_facts failing partway through — disk full, an unreadable "
          "store) must still release conv_lock(dst). The normal dispatch "
          "path (merge_conversation's `else` branch, portal-acquired) runs "
          "`try: _merge_commit() finally: anyio.from_thread.run_sync("
          "_release_dst_lock)` — the release is IN a `finally`, so this is "
          "a test-gap fix, not a code fix: every one of the six cases "
          "above only ever drives a merge that succeeds, is refused before "
          "acquiring, or times out waiting to acquire — none of them makes "
          "_merge_commit() itself raise AFTER the lock is held, so a "
          "regression here (the release falling out of the `finally`, or "
          "acquiring on a code path that forgets to release) would ship "
          "green. If it ever did regress: every later writer to dst — the "
          "extraction tail, /remember, /forget, archive/restore/dedup, and "
          "the next merge — takes this SAME lock with an unbounded "
          "`async with` (see the module comment on _acquire_dst_lock_"
          "bounded/_release_dst_lock), so a stranded lock wedges every one "
          "of them until the process restarts.")
    memory.ensure_storage_layout()
    dst, src = "mergefail-dst", "mergefail-src"
    reset_state(dst)
    reset_state(src)
    seed_facts(dst, ["the mergefail destination's own pre-existing fact"])
    merge_texts = distinct_facts(5, "mergefail")
    seed_facts(src, merge_texts)

    orig_save = facts.save_facts

    def _boom(conv_id, facts_list):
        if conv_id == dst:
            # Synthetic failure standing in for a real one (disk full, a
            # corrupt store) — not a network or lock-layer error, so it
            # exercises exactly the "the commit itself raised" shape this
            # test is named for, not a refusal before the lock was ever
            # held.
            raise OSError(28, "No space left on device (injected, P12-3 test)")
        return orig_save(conv_id, facts_list)

    async def _raise_then_check():
        facts.save_facts = _boom
        try:
            await anyio.to_thread.run_sync(
                functools.partial(portability.merge_conversation, src, dst, dry_run=False)
            )
            return "no_raise", None
        except OSError as e:
            return "raised", e
        finally:
            facts.save_facts = orig_save

    outcome, err = asyncio.run(_raise_then_check())
    assert_eq(
        outcome, "raised",
        f"the injected failure actually raised through the bridge (got {err})",
    )
    assert_true(
        not memory.conv_lock(dst).locked(),
        "*** P12-3: conv_lock(dst) is FREE after the failed commit — the "
        "`finally` around _merge_commit() released it even though the "
        "commit itself raised",
    )

    async def _retry():
        return await anyio.to_thread.run_sync(
            functools.partial(portability.merge_conversation, src, dst, dry_run=False)
        )

    t0 = time.monotonic()
    asyncio.run(_retry())
    elapsed = time.monotonic() - t0
    assert_true(
        elapsed < portability._MERGE_DST_LOCK_TIMEOUT_S * 0.5,
        f"*** P12-3: the retry merge committed promptly ({elapsed:.2f}s), "
        f"not after waiting out the {portability._MERGE_DST_LOCK_TIMEOUT_S}s "
        f"acquire bound — a stranded lock would force every later writer "
        f"to wait the FULL bound and then be refused, forever",
    )
    got = fact_texts(dst)
    landed = [t for t in merge_texts if t in got]
    assert_eq(
        sorted(landed), sorted(merge_texts),
        "every fact from the retried merge landed on dst",
    )

    async def _async_writer_can_acquire():
        try:
            await asyncio.wait_for(memory.conv_lock(dst).acquire(), timeout=2.0)
            memory.conv_lock(dst).release()
            return True
        except asyncio.TimeoutError:
            return False

    acquired = asyncio.run(_async_writer_can_acquire())
    assert_true(
        acquired,
        "*** P12-3: a tail-shaped async writer (holding conv_lock the same "
        "way _facts_tail/_async_tail/`/remember`/`/forget`/archive/"
        "restore/dedup do, via a plain `async with`) can still acquire "
        "conv_lock(dst) after the failed merge — the lock was never "
        "stranded",
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _all_tests():
    return [
        test_two_concurrent_merges_do_not_lose_facts,
        test_single_merge_via_threadpool_still_commits,
        test_merge_serializes_behind_an_async_writer_on_the_destination,
        test_merge_does_not_block_the_event_loop_while_writing,
        test_short_hold_serializes_and_both_merges_land,
        test_long_hold_times_out_with_the_original_refusal,
        test_merge_failure_releases_the_dst_lock,
    ]


if __name__ == "__main__":
    try:
        for t in _all_tests():
            t()
        print("\nAll merge-concurrency tests passed.")
    finally:
        shutil.rmtree(_TMP_ROOT, ignore_errors=True)
