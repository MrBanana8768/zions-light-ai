"""
compactor.bgwork — V2.3 Theme 3: bounded background work.

The compactor fires post-response work (fact extraction, episodic indexing,
summary rollup, lazy backfill) as fire-and-forget asyncio tasks. The naive
version (`asyncio.create_task` per request, tracked in a set) is unbounded:
under a burst of concurrent chats, it spawns one extraction task per request,
each holding an httpx client and making LLM calls. Enough of them and the
process thrashes — exactly the resource-stability failure this theme guards.

BackgroundPool bounds it two ways:
  1. **Concurrency cap** (semaphore) — at most `max_concurrent` tails run at
     once. Excess submissions wait their turn.
  2. **Outstanding ceiling** — a hard cap on total tracked tasks (running +
     waiting). Beyond it, new submissions are **shed** (dropped, counted,
     and the coroutine closed so it doesn't leak) rather than queued without
     limit. Shedding a fact-extraction tail is acceptable degradation — the
     chat response already went out; we just skip *growing* memory for that
     turn under overload, same spirit as the disk-pressure write-gate.

Stats are surfaced in /health/full so sustained shedding is visible.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time

from envcfg import env_int, env_window_s
import logsetup

logger = logging.getLogger("compactor.bgwork")

MAX_CONCURRENT = env_int("COMPACTOR_MAX_CONCURRENT_TAILS", 4)
MAX_OUTSTANDING = env_int("COMPACTOR_MAX_OUTSTANDING_TAILS", 64)

# How long after a shed /health/full keeps calling the system "degraded"
# (v3.1 A11).
#
# `shed` is cumulative for the life of the process, so degrading on `shed > 0`
# would pin the endpoint to "degraded" from the first burst until the next
# restart — and a warning that is always on is a warning nobody reads, the same
# habit that let the token-counter fallback run unnoticed for months. A window
# instead: shedding degrades while it is happening and for a while after, then
# clears itself with no operator action. The cumulative counter stays in the
# payload as the historical record; the window is only what drives `status`.
#
# 300 s spans ten consecutive 30 s HEALTHCHECK probes, so a burst that starts
# and ends between two looks still shows up on the next one.
def _window_s(name: str, default: float) -> float:
    """Read a degrade-window seconds value from the environment, safely.

    A thin alias for envcfg.env_window_s (v3.1.7 R30 rest): this module's own
    copy used to duplicate the parsing logic byte-for-byte, because it cannot
    import main (main imports it) and main.py's helpers are not shared code.
    envcfg.py fixes exactly that — a module with no dependency on anything
    else in the package, so both bgwork and main (and tailhealth, which
    keeps its own identical copy rather than importing this one, since it
    is not owned by this change) can read through it without a cycle.

    Kept as a named wrapper, not inlined at the two call sites below, so
    every existing caller (`bgwork._window_s(...)` — see test_bgwork.py) and
    every existing docstring reference to `bgwork._window_s` keeps working
    unchanged. Behaviour is unchanged: unparseable/blank -> default,
    non-positive or non-finite (`0`, a negative, `inf`, `nan`) -> default.
    See envcfg.env_window_s's docstring for the full reasoning, including the
    incident (an always-on `shed_recently` from an `inf` window) this guards
    against. tailhealth._window_s is a separate, unowned copy of the same
    logic; test_envcfg.py asserts the two still agree.
    """
    return env_window_s(name, default)


SHED_DEGRADE_WINDOW_S = _window_s("COMPACTOR_SHED_DEGRADE_WINDOW_S", 300.0)


class BackgroundPool:
    """Bounded fire-and-forget task pool. Construct once at module load;
    asyncio primitives created here bind to the running loop lazily on first
    use (Python 3.10+), so construction outside a running loop is fine."""

    def __init__(self, max_concurrent: int = MAX_CONCURRENT,
                 max_outstanding: int = MAX_OUTSTANDING,
                 shed_window_s: float = SHED_DEGRADE_WINDOW_S):
        self._max_concurrent = max(1, max_concurrent)
        self._max_outstanding = max(self._max_concurrent, max_outstanding)
        self._shed_window_s = shed_window_s
        self._sem = asyncio.Semaphore(self._max_concurrent)
        self._tasks: set[asyncio.Task] = set()
        self._shed = 0
        self._submitted = 0
        self._completed = 0
        # monotonic timestamp of the most recent shed, or None if we have
        # never shed. Monotonic, not wall clock: this feeds a "how long ago"
        # that must not jump when the clock is stepped.
        self._last_shed_at: float | None = None

    def submit(self, coro, label: str | None = None) -> bool:
        """Schedule `coro` to run under the concurrency cap. Returns True if
        accepted, False if shed (outstanding ceiling hit). Must be called
        from within the event loop.

        `label` names what is being dropped — a conv_id, in practice. `submit`
        receives an opaque coroutine and cannot work out whose tail it is, so
        the caller has to say; without it the shed line reports that memory
        growth stopped for *somebody* and leaves the operator no way to find
        out who. Log only, deliberately: /health/full is not localhost-gated
        the way the /admin endpoints are, and conv_ids do not belong in an
        ungated payload. (v3.1 A11.)
        """
        self._submitted += 1
        if len(self._tasks) >= self._max_outstanding:
            self._shed += 1
            self._last_shed_at = time.monotonic()
            # Close the coroutine so Python doesn't warn "never awaited"
            # and so it releases anything it captured.
            try:
                coro.close()
            except Exception as e:
                # A close() that fails leaks whatever the coroutine
                # captured — an httpx client, a message list — and it did so
                # with no trace. Once per process: this is on the shedding
                # path, which by definition fires in bursts.
                # (v3.1 P0-2b / F61.)
                if logsetup.log_once("bgwork.submit.coro_close"):
                    logger.warning(
                        f"shed coroutine would not close "
                        f"({type(e).__name__}: {e}); it may be holding "
                        f"resources for the life of the process"
                    )
            if self._shed == 1 or self._shed % 25 == 0:
                # Only every 25th shed is logged (bursts, by definition), so
                # this line is a sample rather than a census — it names the one
                # submission that tripped it, not all of them.
                whose = f" most recent: {label};" if label else ""
                logger.warning(
                    f"background work shed (outstanding >= {self._max_outstanding}); "
                    f"total shed={self._shed}.{whose} New-memory growth is pausing "
                    f"under load; chat is unaffected."
                )
            return False
        task = asyncio.create_task(self._run(coro))
        self._tasks.add(task)
        task.add_done_callback(self._on_done)
        return True

    async def _run(self, coro) -> None:
        async with self._sem:
            await coro

    def _on_done(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        self._completed += 1
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.exception(f"background task raised: {exc!r}")

    async def drain(
        self, timeout: float = 10.0, *, cancel_on_timeout: bool = False
    ) -> set[asyncio.Task]:
        """Wait up to `timeout` for outstanding tasks. Returns the tasks
        still not done when the wait ended (empty if everything finished,
        or if `cancel_on_timeout` forced everything to a done state).

        v3.1.9.4 B3 (P15-5). Used to be `asyncio.wait_for(asyncio.gather(
        *tasks, return_exceptions=True), timeout)`. `wait_for` CANCELS the
        awaitable it times out on, and cancelling a gather cancels every
        task it wraps — so a caller that only meant "stop waiting" was
        actually killing every outstanding tail. This module's own
        docstring says the pool is process-wide, and `commands.py`'s
        `_settle_background_work` (the /forget wipe's drain) confirms it:
        "a /forget on one conversation waits on another's tail". A full
        wipe anywhere therefore CANCELLED every other conversation's
        in-flight fact extraction, dedup and rollup, and any running lazy
        backfill (P15-2), the instant `FORGET_SETTLE_TIMEOUT` (10s
        default) passed — routinely, since P15-4 alone measured one
        tail's dedup embed at ~12.9s uncached, before this release's B1
        fix. Worse, a cancelled task is a DONE task: `_on_done` fires for
        it exactly as for a normal completion and discards it from
        `self._tasks`, so `_settle_background_work`'s `stats().outstanding
        == 0` read as "everything settled" when the pool had actually just
        been murdered — the /forget reply's own "background work was
        still finishing" caveat (commands.py) existed but could never
        fire, because after a cancelling drain the pool was always empty.

        `asyncio.wait(tasks, timeout=timeout)` (no `wait_for`, no
        `gather`) is the fix: it simply returns the (done, pending) split
        when the timeout elapses, touching nothing still running. A task
        in `pending` keeps running exactly as if drain had never been
        called — the caller sees it in `stats().outstanding` afterwards
        (and now honestly), and `_on_done` still fires for it, normally,
        whenever it actually finishes.

        Exceptions: unchanged. Every task submitted through `submit()`
        already has `_on_done` as a done-callback, which logs
        `task.exception()` regardless of whether this function is the one
        that observes completion — `return_exceptions=True` on the old
        `gather` was never load-bearing for that; nothing here needs to
        replicate it.

        `cancel_on_timeout=True` restores the OLD behaviour for a caller
        that genuinely wants it — process shutdown is the one such caller
        in this codebase (main.py's `lifespan`, outside this lane's file
        ownership; see fix-3194-bg.md B3 for the exact line it should
        pass). There, the loop is about to close regardless, and letting
        each task's cancellation actually land — so its own `finally`/
        cleanup code runs (an httpx client closing, a lock releasing) —
        is preferable to the interpreter tearing them down mid-work with
        no chance to clean up at all.
        """
        if not self._tasks:
            return set()
        tasks = list(self._tasks)
        logger.info(f"draining {len(tasks)} background task(s)")
        done, pending = await asyncio.wait(tasks, timeout=timeout)
        if not pending:
            return set()
        if cancel_on_timeout:
            logger.warning(
                f"{len(pending)} background task(s) didn't finish in "
                f"{timeout}s; cancelling (cancel_on_timeout=True)"
            )
            for t in pending:
                t.cancel()
            # Wait for the cancellation to actually land — same guarantee
            # the old wait_for(gather(...)) gave: a caller that asked to
            # cancel sees every task done, not merely requested-to-stop.
            await asyncio.wait(pending)
            return set()
        logger.warning(
            f"{len(pending)} background task(s) still running after "
            f"{timeout}s; left running rather than cancelled — a drain "
            f"timing out must not destroy someone else's in-flight work"
        )
        return pending

    def stats(self) -> dict:
        """For /health/full — outstanding/shed/throughput + caps.

        `shed_recently` is the field the health status actually reads. Until
        v3.1 this whole dict was computed, placed in the payload and never
        consulted, so sustained shedding — the pool dropping fact extraction,
        episodic indexing and summary rollups on the floor — reported as
        `"status": "ok"` and passed the Docker HEALTHCHECK. A health check that
        cannot report degradation is decoration. (v3.1 A11 / incident C2.)
        """
        since = (
            None if self._last_shed_at is None
            else round(time.monotonic() - self._last_shed_at, 1)
        )
        return {
            "outstanding": len(self._tasks),
            "max_concurrent": self._max_concurrent,
            "max_outstanding": self._max_outstanding,
            "submitted": self._submitted,
            "completed": self._completed,
            "shed": self._shed,
            "seconds_since_last_shed": since,
            "shed_recently": since is not None and since <= self._shed_window_s,
            "shed_window_s": self._shed_window_s,
            # The instant before shedding: the ceiling is full, so the next
            # submission is dropped. Reported, but deliberately NOT a degrade
            # condition on its own — a pool that touches its ceiling and drains
            # again lost nothing, and degrading on it would flap on every
            # burst. If it stays full, the shed follows within one request and
            # `shed_recently` picks it up then.
            "at_capacity": len(self._tasks) >= self._max_outstanding,
        }


# Process-wide singleton.
pool = BackgroundPool()
