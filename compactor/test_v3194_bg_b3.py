"""
v3.1.9.4 B3 (P15-5): bgwork.BackgroundPool.drain used to CANCEL every
outstanding task on timeout (asyncio.wait_for cancels the awaitable it
times out on, and a cancelled gather cancels every task it wraps). A
/forget in ONE conversation drains the whole (process-wide) pool, so this
silently killed every OTHER conversation's in-flight memory tail, and a
cancelled task is a DONE task, so commands._settle_background_work read
"outstanding == 0" as "everything settled" when it had actually just been
murdered.

Part A mirrors the reviewer's shape (SP\\p15\\p15_e2_drain_cancel.py Part
A): the pool alone, a slow task and a fast task, a short drain timeout.
Part B proves the real _dedup_pass path is unaffected (embedding call
patterns are not confused with cancellation). Part C proves
cancel_on_timeout=True (the opt-in for shutdown) restores the OLD
behaviour exactly. Synthetic text only.

Run: python test_v3194_bg_b3.py
"""
import asyncio
import sys

import bgwork

FAILED = []


def check(cond, label):
    print(("  ok   " if cond else "FAIL ") + label, flush=True)
    if not cond:
        FAILED.append(label)


# ---------------------------------------------------------------------------
# Part A: the pool alone (the reviewer's shape)
# ---------------------------------------------------------------------------

async def _part_a(drain_timeout, cancel_on_timeout=False):
    pool = bgwork.BackgroundPool(max_concurrent=4, max_outstanding=64)
    done = {}

    async def work(name, secs):
        try:
            await asyncio.sleep(secs)
            done[name] = "finished"
        except asyncio.CancelledError:
            done[name] = "cancelled"
            raise

    pool.submit(work("slow", 3.0), "slow")
    pool.submit(work("fast", 0.05), "fast")
    pending = await pool.drain(timeout=drain_timeout, cancel_on_timeout=cancel_on_timeout)
    # Snapshot `done` HERE, inside the coroutine, before returning: a task
    # left in `pending` (cancel_on_timeout=False) is still genuinely
    # running at this point, but asyncio.run()'s OWN cleanup cancels every
    # task still outstanding when the top-level coroutine it was given
    # returns — which happens a moment after this function returns to it.
    # Returning the live `done` dict would let that cleanup-triggered
    # cancellation mutate it out from under the caller's check, which
    # would test asyncio.run()'s teardown, not drain(). A plain dict copy
    # freezes exactly what drain() itself did.
    return dict(done), pending, pool


def test_timeout_does_not_cancel_the_slow_task():
    print("\n[test] drain(timeout=0.3): the slow task is NOT cancelled, "
          "just still running when drain returns")
    # Wide margin (0.3s drain vs a 3.0s task) deliberately — this is a
    # real-wall-clock test, and a narrow margin is exactly what makes a
    # timing test flaky under system load, not a correctness signal.
    done, pending, pool = asyncio.run(_part_a(0.3))
    check("slow" not in done, "the slow task had not finished yet (correct — it only had 0.3s of its 3.0s)")
    check("fast" in done and done["fast"] == "finished", "the fast task finished within the timeout")
    check(len(pending) == 1, "drain reports exactly one task still pending")
    # (The next test confirms the slow task actually finishes naturally
    # afterwards, rather than merely "not yet observed as cancelled".)


def test_slow_task_actually_completes_after_the_timed_out_drain():
    print("\n[test] CONFIRM: the slow task genuinely keeps running and "
          "completes on its own after a timed-out drain — not cancelled, "
          "not silently dropped")

    async def go():
        pool = bgwork.BackgroundPool(max_concurrent=4, max_outstanding=64)
        done = {}

        async def work(name, secs):
            await asyncio.sleep(secs)
            done[name] = True

        pool.submit(work("slow", 0.5), "slow")
        pending = await pool.drain(timeout=0.1)
        mid_outstanding = pool.stats()["outstanding"]
        # Give the still-running task time to finish for real (wide
        # margin — the task needs 0.4s more; this waits 2s).
        await asyncio.sleep(2.0)
        return done, pending, mid_outstanding, pool.stats()["outstanding"]

    done, pending, mid_outstanding, final_outstanding = asyncio.run(go())
    check(len(pending) == 1, "drain reported the task as still pending")
    check(mid_outstanding == 1, "stats() agrees: 1 outstanding right after the timed-out drain")
    check(done.get("slow") is True, "the task completed on its own, later — it was never cancelled")
    check(final_outstanding == 0, "and stats() reflects it finishing (via the normal _on_done callback)")


def test_control_a_long_enough_timeout_both_finish():
    print("\n[test] CONTROL: drain(timeout=5) — long enough — both tasks finish normally")
    done, pending, pool = asyncio.run(_part_a(5.0))
    check("slow" in done and done["slow"] == "finished", "slow task finished")
    check("fast" in done and done["fast"] == "finished", "fast task finished")
    check(len(pending) == 0, "nothing left pending")
    check(pool.stats()["outstanding"] == 0, "stats agree: fully settled")


def test_outstanding_after_timeout_reflects_reality_not_a_lie():
    print("\n[test] the EXACT P15-5 shape: stats().outstanding must NOT read "
          "0 immediately after a timed-out drain (that used to mean "
          "'cancelled to death', not 'settled')")

    async def go():
        pool = bgwork.BackgroundPool(max_concurrent=4, max_outstanding=64)

        async def work(secs):
            await asyncio.sleep(secs)

        pool.submit(work(1.0), "conv-B")
        await pool.drain(timeout=0.2)
        return pool.stats()["outstanding"]

    outstanding = asyncio.run(go())
    check(outstanding == 1,
          f"outstanding={outstanding}: honestly non-zero — commands.py's "
          f"_settle_background_work now sees this and the /forget reply's "
          f"'still finishing' caveat can actually fire")


# ---------------------------------------------------------------------------
# Part B: the real dedup path through the pool is unaffected
# ---------------------------------------------------------------------------

def test_a_fast_real_tail_still_drains_cleanly():
    print("\n[test] CONTROL: an ordinary (fast) submission still drains "
          "to zero outstanding, same as before this fix")

    async def go():
        pool = bgwork.BackgroundPool(max_concurrent=2, max_outstanding=8)
        ran = []

        async def tail():
            await asyncio.sleep(0.01)
            ran.append(1)

        for _ in range(3):
            pool.submit(tail())
        pending = await pool.drain(timeout=5.0)
        return ran, pending, pool.stats()

    ran, pending, stats = asyncio.run(go())
    check(len(ran) == 3, "all three tails ran")
    check(len(pending) == 0, "nothing pending")
    check(stats["outstanding"] == 0, "stats: fully drained")


# ---------------------------------------------------------------------------
# Part C: cancel_on_timeout=True restores the OLD (shutdown) behaviour
# ---------------------------------------------------------------------------

def test_cancel_on_timeout_true_restores_old_behaviour():
    print("\n[test] cancel_on_timeout=True (the shutdown opt-in): the slow "
          "task IS cancelled, and drain waits for the cancellation to land")
    done, pending, pool = asyncio.run(_part_a(0.2, cancel_on_timeout=True))
    check(len(pending) == 0,
          "drain reports NOTHING pending — cancel_on_timeout waits for the "
          "cancellation to finish before returning, same guarantee the old "
          "wait_for(gather(...)) gave")
    check(done.get("slow") == "cancelled",
          "the slow task's own CancelledError handler ran — cleanup code "
          "gets a chance to run, unlike an abrupt interpreter exit")
    check(pool.stats()["outstanding"] == 0, "stats: pool is empty after a cancelling drain")


def test_cancel_on_timeout_default_is_false():
    print("\n[test] cancel_on_timeout defaults to False — a caller that "
          "does not opt in gets the SAFE (non-destructive) behaviour")
    import inspect
    sig = inspect.signature(bgwork.BackgroundPool.drain)
    assert sig.parameters["cancel_on_timeout"].default is False


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _all_tests():
    return [
        test_timeout_does_not_cancel_the_slow_task,
        test_slow_task_actually_completes_after_the_timed_out_drain,
        test_control_a_long_enough_timeout_both_finish,
        test_outstanding_after_timeout_reflects_reality_not_a_lie,
        test_a_fast_real_tail_still_drains_cleanly,
        test_cancel_on_timeout_true_restores_old_behaviour,
        test_cancel_on_timeout_default_is_false,
    ]


if __name__ == "__main__":
    for t in _all_tests():
        t()
    print("\nRESULT:", "all v3.1.9.4 B3 checks passed" if not FAILED else f"{len(FAILED)} check(s) FAILED: {FAILED}")
    sys.exit(0 if not FAILED else 1)
