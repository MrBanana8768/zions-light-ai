"""
CPU-only Tier-1 tests for compactor.bgwork (V2.3 Theme 3).

Bounded background pool: concurrency cap, hard outstanding ceiling with
shedding (no coroutine leak), drain, stats.

Run: python test_bgwork.py
"""

import asyncio
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
    ]


if __name__ == "__main__":
    for t in _all():
        t()
    print("\nAll bgwork smoke tests passed.")
