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

logger = logging.getLogger("compactor.bgwork")

MAX_CONCURRENT = int(os.environ.get("COMPACTOR_MAX_CONCURRENT_TAILS", "4") or 4)
MAX_OUTSTANDING = int(os.environ.get("COMPACTOR_MAX_OUTSTANDING_TAILS", "64") or 64)


class BackgroundPool:
    """Bounded fire-and-forget task pool. Construct once at module load;
    asyncio primitives created here bind to the running loop lazily on first
    use (Python 3.10+), so construction outside a running loop is fine."""

    def __init__(self, max_concurrent: int = MAX_CONCURRENT,
                 max_outstanding: int = MAX_OUTSTANDING):
        self._max_concurrent = max(1, max_concurrent)
        self._max_outstanding = max(self._max_concurrent, max_outstanding)
        self._sem = asyncio.Semaphore(self._max_concurrent)
        self._tasks: set[asyncio.Task] = set()
        self._shed = 0
        self._submitted = 0
        self._completed = 0

    def submit(self, coro) -> bool:
        """Schedule `coro` to run under the concurrency cap. Returns True if
        accepted, False if shed (outstanding ceiling hit). Must be called
        from within the event loop."""
        self._submitted += 1
        if len(self._tasks) >= self._max_outstanding:
            self._shed += 1
            # Close the coroutine so Python doesn't warn "never awaited"
            # and so it releases anything it captured.
            try:
                coro.close()
            except Exception:
                pass
            if self._shed == 1 or self._shed % 25 == 0:
                logger.warning(
                    f"background work shed (outstanding >= {self._max_outstanding}); "
                    f"total shed={self._shed}. New-memory growth is pausing "
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
        """For /health/full — outstanding/shed/throughput + caps."""
        return {
            "outstanding": len(self._tasks),
            "max_concurrent": self._max_concurrent,
            "max_outstanding": self._max_outstanding,
            "submitted": self._submitted,
            "completed": self._completed,
            "shed": self._shed,
        }


# Process-wide singleton.
pool = BackgroundPool()
