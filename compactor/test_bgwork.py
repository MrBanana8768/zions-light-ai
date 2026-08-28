"""
CPU-only Tier-1 tests for compactor.bgwork (V2.3 Theme 3).

Bounded background pool: concurrency cap, hard outstanding ceiling with
shedding (no coroutine leak), drain, stats.

Run: python test_bgwork.py
"""

import asyncio
import logging
import os
import sys

os.environ["COMPACTOR_MAX_CONCURRENT_TAILS"] = "2"
os.environ["COMPACTOR_MAX_OUTSTANDING_TAILS"] = "4"

import bgwork  # noqa: E402


def assert_eq(a, b, label):
    if a != b:
        print(f"FAIL {label}: expected {b!r}, got {a!r}")
        sys.exit(1)
    print(f"  ok   {label}")


def assert_true(c, label):
    if not c:
        print(f"FAIL {label}")
        sys.exit(1)
    print(f"  ok   {label}")


def test_accepts_and_runs_within_caps():
    print("\n[test] submit: a coro under the caps runs to completion")

    async def go():
        p = bgwork.BackgroundPool(max_concurrent=2, max_outstanding=4)
        ran = []

        async def work():
            ran.append(1)

        accepted = p.submit(work())
        await p.drain()
        return accepted, ran

    accepted, ran = asyncio.run(go())
    assert_eq(accepted, True, "submission accepted")
    assert_eq(len(ran), 1, "coro ran")


def test_concurrency_cap_respected():
    print("\n[test] at most max_concurrent run simultaneously")

    async def go():
        p = bgwork.BackgroundPool(max_concurrent=2, max_outstanding=10)
        active = 0
        peak = 0
        gate = asyncio.Event()

        async def work():
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await gate.wait()
            active -= 1

        for _ in range(6):
            p.submit(work())
        # Let tasks start and hit the gate
        await asyncio.sleep(0.05)
        peak_while_gated = peak
        gate.set()
        await p.drain()
        return peak_while_gated

    peak = asyncio.run(go())
    assert_eq(peak, 2, "never more than 2 running at once")


def test_sheds_beyond_outstanding_ceiling():
    print("\n[test] submissions beyond outstanding ceiling are shed (not queued)")

    async def go():
        p = bgwork.BackgroundPool(max_concurrent=2, max_outstanding=4)
        gate = asyncio.Event()

        async def work():
            await gate.wait()

        results = [p.submit(work()) for _ in range(8)]
        await asyncio.sleep(0.05)
        stats_mid = p.stats()
        gate.set()
        await p.drain()
        return results, stats_mid

    results, stats_mid = asyncio.run(go())
    accepted = sum(1 for r in results if r)
    shed = sum(1 for r in results if not r)
    assert_eq(accepted, 4, "exactly max_outstanding (4) accepted")
    assert_eq(shed, 4, "the other 4 shed")
    assert_eq(stats_mid["shed"], 4, "stats reflect 4 shed")
    assert_eq(stats_mid["outstanding"], 4, "4 outstanding while gated")


def test_shed_coroutine_does_not_leak():
    print("\n[test] a shed coroutine is closed (no 'never awaited' leak)")
    import inspect

    async def go():
        p = bgwork.BackgroundPool(max_concurrent=1, max_outstanding=1)
        gate = asyncio.Event()

        async def blocker():
            await gate.wait()

        async def victim():
            await asyncio.sleep(1)

        p.submit(blocker())            # fills the single outstanding slot
        victim_coro = victim()
        accepted = p.submit(victim_coro)  # must be shed + closed
        state = inspect.getcoroutinestate(victim_coro)
        gate.set()
        await p.drain()
        return accepted, state

    accepted, state = asyncio.run(go())
    assert_eq(accepted, False, "victim shed")
    # A closed coroutine reports CORO_CLOSED — no "never awaited" warning.
    assert_eq(state, "CORO_CLOSED", "shed coroutine was closed (not leaked)")


def test_exception_in_task_is_logged_not_raised():
    print("\n[test] an exception inside a task doesn't propagate to the loop")

    async def go():
        p = bgwork.BackgroundPool(max_concurrent=2, max_outstanding=4)

        async def boom():
            raise RuntimeError("kaboom")

        accepted = p.submit(boom())
        await p.drain()  # must not raise
        return accepted, p.stats()

    accepted, stats = asyncio.run(go())
    assert_eq(accepted, True, "accepted")
    assert_eq(stats["completed"], 1, "counted as completed despite exception")


def test_stats_shape_and_counters():
    print("\n[test] stats: submitted/completed/shed/outstanding/caps present")

    async def go():
        p = bgwork.BackgroundPool(max_concurrent=2, max_outstanding=3)

        async def work():
            return

        for _ in range(2):
            p.submit(work())
        await p.drain()
        return p.stats()

    s = asyncio.run(go())
    for k in ("outstanding", "max_concurrent", "max_outstanding",
              "submitted", "completed", "shed"):
        assert_true(k in s, f"stats has {k}")
    assert_eq(s["submitted"], 2, "submitted=2")
    assert_eq(s["completed"], 2, "completed=2")
    assert_eq(s["outstanding"], 0, "drained to 0 outstanding")
    assert_eq(s["max_concurrent"], 2, "cap echoed")


def test_drain_with_nothing_is_noop():
    print("\n[test] drain with no tasks is a clean no-op")

    async def go():
        p = bgwork.BackgroundPool()
        await p.drain()
        return True

    assert_eq(asyncio.run(go()), True, "drain no-op ok")


def test_outstanding_floor_at_least_concurrency():
    print("\n[test] max_outstanding is floored to >= max_concurrent")
    p = bgwork.BackgroundPool(max_concurrent=8, max_outstanding=2)
    assert_true(p._max_outstanding >= p._max_concurrent, "ceiling >= concurrency")


# ---------------------------------------------------------------------------
# Shed recency — the field /health/full's status reads (v3.1 A11)
#
# The defect: stats() reported a cumulative `shed` count that nothing
# consulted, so shedding was invisible to `status`. `shed` alone cannot drive
# a status either — it never goes down, so it would pin the endpoint to
# "degraded" until restart. These tests pin both halves: shedding is visible,
# and it stops being visible on its own.
# ---------------------------------------------------------------------------

def _shed_once(pool):
    """Fill `pool` to its ceiling and force one shed. Returns the gate the
    caller must set before draining."""
    gate = asyncio.Event()

    async def blocker():
        await gate.wait()

    async def victim():
        await asyncio.sleep(1)

    for _ in range(pool._max_outstanding):
        pool.submit(blocker())
    pool.submit(victim(), label="conv=shed-me")
    return gate


def test_fresh_pool_reports_no_shed_recency():
    print("\n[test] a pool that has never shed reports no recency at all")
    s = bgwork.BackgroundPool(max_concurrent=1, max_outstanding=1).stats()
    assert_eq(s["seconds_since_last_shed"], None, "None, not 0 — never happened")
    assert_eq(s["shed_recently"], False, "not shedding")
    assert_eq(s["at_capacity"], False, "empty pool is not at capacity")


def test_shed_is_visible_in_stats():
    print("\n[test] a shed sets shed_recently and the age clock")

    async def go():
        p = bgwork.BackgroundPool(max_concurrent=1, max_outstanding=1,
                                  shed_window_s=300)
        gate = _shed_once(p)
        s = p.stats()
        gate.set()
        await p.drain()
        return s

    s = asyncio.run(go())
    assert_eq(s["shed"], 1, "one shed counted")
    assert_eq(s["shed_recently"], True, "shedding is VISIBLE, not just counted")
    assert_true(s["seconds_since_last_shed"] is not None, "age reported")
    assert_true(s["seconds_since_last_shed"] < 5.0, "age is recent")
    assert_eq(s["at_capacity"], True, "ceiling full while the blocker holds it")


def test_shed_recency_expires_but_the_count_does_not():
    """The status must clear itself once the burst is over — nobody restarts a
    box to silence a stale health warning. The cumulative counter stays, as the
    historical record; only `shed_recently` ages out."""
    print("\n[test] shed_recently ages out of the window; shed count persists")

    async def go():
        p = bgwork.BackgroundPool(max_concurrent=1, max_outstanding=1,
                                  shed_window_s=0.05)
        gate = _shed_once(p)
        during = p.stats()
        await asyncio.sleep(0.12)   # past the 0.05s window
        after = p.stats()
        gate.set()
        await p.drain()
        return during, after

    during, after = asyncio.run(go())
    assert_eq(during["shed_recently"], True, "degraded during the burst")
    assert_eq(after["shed_recently"], False, "clears itself once the burst ages out")
    assert_eq(after["shed"], 1, "the cumulative count is NOT reset")
    assert_eq(after["shed_window_s"], 0.05, "window echoed for the reader")


def test_shed_warning_names_the_caller_that_lost_its_tail():
    """`submit` gets an opaque coroutine, so without a label the warning says
    memory growth stopped for somebody and gives no way to find out who."""
    print("\n[test] the shed warning names the label the caller passed")

    records = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    async def go():
        p = bgwork.BackgroundPool(max_concurrent=1, max_outstanding=1)
        gate = _shed_once(p)   # the victim carries label="conv=shed-me"
        gate.set()
        await p.drain()

    h = _Capture()
    bgwork.logger.addHandler(h)
    try:
        asyncio.run(go())
    finally:
        bgwork.logger.removeHandler(h)

    shed_lines = [m for m in records if "background work shed" in m]
    assert_eq(len(shed_lines), 1, "the first shed logs exactly once")
    assert_true("conv=shed-me" in shed_lines[0],
                "the shed line names WHICH submission was dropped")


def test_submit_still_accepts_a_bare_coroutine():
    """`label` is optional — main.py's _fire_and_forget does not pass one yet,
    and this must not become a TypeError on the shedding path."""
    print("\n[test] submit(coro) with no label still works")

    async def go():
        p = bgwork.BackgroundPool(max_concurrent=1, max_outstanding=1)
        ran = []

        async def work():
            ran.append(1)

        accepted = p.submit(work())
        await p.drain()
        return accepted, ran

    accepted, ran = asyncio.run(go())
    assert_eq(accepted, True, "accepted without a label")
    assert_eq(len(ran), 1, "and ran")


def _all():
    return [
        test_accepts_and_runs_within_caps,
        test_concurrency_cap_respected,
        test_sheds_beyond_outstanding_ceiling,
        test_shed_coroutine_does_not_leak,
        test_exception_in_task_is_logged_not_raised,
        test_stats_shape_and_counters,
        test_drain_with_nothing_is_noop,
        test_outstanding_floor_at_least_concurrency,
        # v3.1 A11 — shed recency, the field /health/full's status reads.
        test_fresh_pool_reports_no_shed_recency,
        test_shed_is_visible_in_stats,
        test_shed_recency_expires_but_the_count_does_not,
        test_shed_warning_names_the_caller_that_lost_its_tail,
        test_submit_still_accepts_a_bare_coroutine,
    ]


if __name__ == "__main__":
    for t in _all():
        t()
    print("\nAll bgwork smoke tests passed.")
