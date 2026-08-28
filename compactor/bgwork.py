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

import logsetup

logger = logging.getLogger("compactor.bgwork")

MAX_CONCURRENT = int(os.environ.get("COMPACTOR_MAX_CONCURRENT_TAILS", "4") or 4)
MAX_OUTSTANDING = int(os.environ.get("COMPACTOR_MAX_OUTSTANDING_TAILS", "64") or 64)

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
SHED_DEGRADE_WINDOW_S = float(
    os.environ.get("COMPACTOR_SHED_DEGRADE_WINDOW_S", "300") or 300
)


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

    async def drain(self, timeout: float = 10.0) -> None:
        """Await outstanding tasks (used at shutdown)."""
        if not self._tasks:
            return
        logger.info(f"draining {len(self._tasks)} background task(s)")
        try:
            await asyncio.wait_for(
                asyncio.gather(*list(self._tasks), return_exceptions=True),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(f"background tasks didn't finish in {timeout}s; abandoning")

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
