"""
compactor.health — V2.1 Phase 6 Step 1: deep health probe.

Why a new module instead of expanding /health: /health is a liveness
probe that needs to be fast and dependency-free (called every 30s by
the Docker HEALTHCHECK). /health/full is a *readiness/diagnostics*
probe that actually walks the stack:

  - Can the compactor reach vLLM?
  - Is /data writable?
  - How many conversations / facts / indexed exchanges exist?

The output is the single source of truth used by:
  1. /health/full HTTP endpoint (Docker HEALTHCHECK target after this
     phase — replaces the current `curl :3000` check which can't tell
     whether vLLM is up)
  2. /admin/selftest (Step 2) — folds these checks into its report
  3. Future V2.1 Theme 3 UI elements (memory growth metrics)

All probes degrade to a structured error rather than raising — a
single broken probe should never make /health/full itself 500.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any

import httpx

import facts
import logsetup
import memory
import retrieval
import summarizer

logger = logging.getLogger("compactor.health")

# Probe timeout — short, because /health/full is hit by HEALTHCHECK
# every 30s and an unresponsive vLLM shouldn't make the probe hang.
_VLLM_PROBE_TIMEOUT_S = float(os.environ.get("COMPACTOR_HEALTH_PROBE_TIMEOUT_S", "3.0"))


# ---------------------------------------------------------------------------
# Individual probes
# ---------------------------------------------------------------------------

async def probe_vllm(vllm_url: str) -> dict:
    """Hit vLLM's /v1/models. ok=True iff 2xx with a model list."""
    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=_VLLM_PROBE_TIMEOUT_S) as c:
            r = await c.get(f"{vllm_url.rstrip('/')}/v1/models")
        latency_ms = (time.monotonic() - t0) * 1000.0
        if r.status_code >= 400:
            return {
                "ok": False,
                "latency_ms": round(latency_ms, 1),
                "error": f"HTTP {r.status_code}",
                "models": [],
            }
        data = r.json()
        model_ids = [m.get("id") for m in (data.get("data") or []) if m.get("id")]
        return {
            "ok": bool(model_ids),
            "latency_ms": round(latency_ms, 1),
            "models": model_ids,
            "error": None if model_ids else "no models listed",
        }
    except Exception as e:
        latency_ms = (time.monotonic() - t0) * 1000.0
        return {
            "ok": False,
            "latency_ms": round(latency_ms, 1),
            "error": f"{type(e).__name__}: {e}",
            "models": [],
        }


def probe_storage() -> dict:
    """Verify the persistent volume is mounted and writable. We touch a
    sentinel file rather than just checking st_mode — read-only mounts
    can still report rwx perms but fail on write.
    """
    root = memory.storage_root()
    try:
        root.mkdir(parents=True, exist_ok=True)
        sentinel = root / ".health_probe"
        sentinel.write_text("ok", encoding="utf-8")
        sentinel.unlink()
        # Free-space report is best-effort — st_size on a directory isn't
        # portable. shutil.disk_usage works on POSIX and Windows.
        try:
            import shutil
            usage = shutil.disk_usage(str(root))
            free_gb = round(usage.free / (1024 ** 3), 2)
            total_gb = round(usage.total / (1024 ** 3), 2)
        except Exception:
            free_gb = None
            total_gb = None
        return {
            "ok": True,
            "writable": True,
            "root": str(root),
            "free_gb": free_gb,
            "total_gb": total_gb,
            "error": None,
        }
    except Exception as e:
        return {
            "ok": False,
            "writable": False,
            "root": str(root),
            "free_gb": None,
            "total_gb": None,
            "error": f"{type(e).__name__}: {e}",
        }


def gather_memory_stats() -> dict:
    """Aggregate counters across every known conversation. Best-effort
    per-conv: a single corrupted file doesn't poison the totals — but it is
    COUNTED, in `unreadable`, so the caller can see that the totals are
    incomplete.

    Before v3.1 each of these handlers was a bare `pass`. An unreadable
    facts file — the exact corruption that destroys memory — was silently
    skipped, so `facts_total` simply read lower while `status` stayed "ok".
    The one endpoint whose purpose is to notice could not see the thing it
    exists to catch. A silently smaller number is the defect, so a nonzero
    `unreadable` count is the signal; the totals alone are not.
    (v3.1 P0-2b / F61.)
    """
    conv_ids = memory.list_known_conv_ids()
    facts_total = 0
    indexed_total = 0
    summaries_with_l1 = 0
    summaries_with_l3 = 0
    # Conversations whose layer could not be read at all. For episodic this
    # also covers conversation_doc_count returning None (store unavailable),
    # which is not an exception but is equally "we could not count this".
    unreadable = {"facts": 0, "episodic": 0, "summaries": 0}
    for cid in conv_ids:
        try:
            facts_total += len(facts.load_facts(cid))
        except Exception as e:
            unreadable["facts"] += 1
            if logsetup.log_once("health.stats.facts"):
                logger.warning(
                    f"conv={cid}: facts unreadable during health scan "
                    f"({type(e).__name__}: {e}); facts_total is incomplete — "
                    f"see stats.unreadable.facts for the running count"
                )
        try:
            n_indexed = retrieval.conversation_doc_count(cid)
            if n_indexed is None:
                unreadable["episodic"] += 1
            else:
                indexed_total += n_indexed
        except Exception as e:
            unreadable["episodic"] += 1
            if logsetup.log_once("health.stats.episodic"):
                logger.warning(
                    f"conv={cid}: episodic count unreadable during health "
                    f"scan ({type(e).__name__}: {e})"
                )
        try:
            state = summarizer.load_state(cid)
            if state.get("l1"):
                summaries_with_l1 += 1
            if state.get("l3"):
                summaries_with_l3 += 1
        except Exception as e:
            unreadable["summaries"] += 1
            if logsetup.log_once("health.stats.summaries"):
                logger.warning(
                    f"conv={cid}: summary state unreadable during health scan "
                    f"({type(e).__name__}: {e}); this conversation has stopped "
                    f"being counted in summaries_with_l1/l3"
                )
    return {
        "conversations": len(conv_ids),
        "facts_total": facts_total,
        # None, not 0, when nothing could be counted — a dead vector store
        # must not report as an empty one (retrieval.conversation_doc_count).
        "indexed_exchanges_total": (
            None
            if conv_ids and unreadable["episodic"] == len(conv_ids)
            else indexed_total
        ),
        "summaries_with_l1": summaries_with_l1,
        "summaries_with_l3": summaries_with_l3,
        "unreadable": unreadable,
    }


# ---------------------------------------------------------------------------
# Aggregated report
# ---------------------------------------------------------------------------

def _gather_blocking() -> dict:
    """Every filesystem-touching probe, in one call, meant to be run OFF the
    event loop (v3.1 A12).

    All four of these blocked the loop directly until v3.1:

      - `probe_storage` writes and unlinks a sentinel on the data volume —
        and it blocks longest in exactly the situation it exists to detect,
        a volume that has stopped answering.
      - `gather_memory_stats` reads three layers for every conversation
        `memory.list_known_conv_ids()` returns, which is uncapped.
      - `degrade.write_state` statvfs's the watch path (TTL-cached, cheap).
      - `backup.latest_backup_info` lists and stats the backup directory.

    Measured on the v3.0.5-cu12 image with RAG disabled, so only the
    facts+summaries reads are counted and the Chroma leg inside
    `conversation_doc_count` is *on top of* these figures. Median of 5 runs of
    `gather_memory_stats` against a seeded store, 8 facts and an L1 stack per
    conversation:

        convs=   10  median=   1.0 ms
        convs=  100  median=  13.5 ms
        convs=  500  median=  49.3 ms
        convs= 1000  median= 100.0 ms

    Linear and unbounded — `memory.list_known_conv_ids()` has no cap — and
    until v3.1 it ran on the one event loop this process has (`supervisord.conf`
    starts uvicorn with no `--workers`) every 30 s, on the Docker HEALTHCHECK.
    Every concurrent chat stalled for the length of the scan. ~100 ms of dead
    loop twice a minute is not an outage, which is why this is S3 and not
    higher; it is also free to fix.

    Each probe keeps its own error handling: a thread hop must not turn one
    broken probe into a 500 from the whole endpoint, which is the promise the
    module docstring makes.
    """
    storage = probe_storage()
    stats = gather_memory_stats()

    # V2.3 Theme 2: disk-pressure write state. "paused" means we're still
    # serving but no longer persisting new memory — a degraded condition the
    # operator needs to see.
    try:
        import degrade
        writes = degrade.write_state()
    except Exception as e:
        writes = {"new_memory_writes": "unknown", "error": f"{type(e).__name__}: {e}"}

    # V2.3 Theme 1: surface backup durability status (best-effort).
    backups: dict[str, Any]
    try:
        import backup as backup_module
        backups = backup_module.latest_backup_info()
    except Exception as e:
        backups = {"count": None, "latest": None, "error": f"{type(e).__name__}: {e}"}

    return {"storage": storage, "stats": stats, "writes": writes, "backups": backups}


async def gather_health_full(
    vllm_url: str, target_tokens: int, tokenize: dict | None = None
) -> dict:
    """The single source of truth used by /health/full and /admin/selftest.

    Status semantics:
      - "ok"       — all checks pass; serve traffic normally
      - "degraded" — storage OK but something the operator needs to see:
                     vLLM unreachable, new-memory writes paused under disk
                     pressure, or background work shedding. Compactor can
                     still serve admin/export endpoints. Container stays
                     alive so supervisord can restart vLLM independently.
      - "down"     — storage broken. Nothing useful possible. Container
                     should be replaced.

    `status_reasons` carries WHY, because "degraded" on its own tells the
    operator to go read three sub-dicts and diff them against a healthy run.

    Returned 200 for ok+degraded, 503 for down (caller maps).
    """
    # The vLLM probe is async with its own timeout; everything else is
    # blocking filesystem work and goes to a thread (see _gather_blocking).
    # Run them concurrently — serialized, the probe's up-to-3 s timeout sat in
    # front of the store scan for no reason. They share no state: the probe is
    # an httpx call, the scan reads files under a different directory.
    vllm, blocking = await asyncio.gather(
        probe_vllm(vllm_url),
        asyncio.to_thread(_gather_blocking),
    )
    storage = blocking["storage"]
    stats = blocking["stats"]
    writes = blocking["writes"]
    backup_info = blocking["backups"]

    # V2.3 Theme 3: bounded-background-work pool stats (outstanding/shed).
    # Pure in-memory counters, so this one stays on the loop thread.
    try:
        import bgwork
        bg = bgwork.pool.stats()
    except Exception as e:
        bg = {"error": f"{type(e).__name__}: {e}"}

    # Why a reason list and not a bare string: `bg` used to be computed here,
    # placed in the payload, and never read. Sustained shedding — the pool
    # dropping fact extraction, episodic indexing and summary rollups because
    # it was over its outstanding ceiling — reported "ok" and passed the
    # HEALTHCHECK. That is the C2 finding from the incident write-up: the user
    # has been the monitoring for this system twice, and this endpoint said ok
    # both times. Anything that means "the system is not doing its job right
    # now" has to reach `status`, and has to say which thing it was.
    # (v3.1 A11.)
    reasons: list[str] = []
    if not storage["ok"]:
        # Storage is the one condition that makes everything else moot, so it
        # short-circuits rather than joining the list.
        reasons.append(f"storage not writable ({storage.get('error')})")
        status = "down"
    else:
        if not vllm["ok"]:
            reasons.append(f"vLLM unreachable ({vllm.get('error')})")
        if writes.get("new_memory_writes") == "paused":
            reasons.append(
                f"new-memory writes paused under disk pressure "
                f"(free_mb={writes.get('free_mb')})"
            )
        if bg.get("error"):
            # We could not read the pool at all. Same doctrine as
            # indexed_exchanges_total: unknown is not the same as fine, and a
            # layer we cannot see must not be reported as healthy.
            reasons.append(f"background pool unobservable ({bg['error']})")
        elif bg.get("shed_recently"):
            reasons.append(
                f"background work shedding: {bg.get('shed')} task(s) dropped, "
                f"most recent {bg.get('seconds_since_last_shed')}s ago "
                f"(outstanding {bg.get('outstanding')}/"
                f"{bg.get('max_outstanding')}). New memory is not being "
                f"written for the turns that were dropped."
            )
        # The counter the budget is computed from. When /tokenize is
        # unreachable the compactor keeps serving on a local estimate that has
        # measured up to 51% low on assistant content — it is degraded, not
        # broken, and it is precisely the state both 2026-08-28 outages ran in
        # while every health surface said ok. Optional so an older caller that
        # passes two arguments still works.
        if tokenize and not tokenize.get("ok", True):
            reasons.append(
                f"/tokenize unavailable: {tokenize.get('consecutive_failures')} "
                f"consecutive failure(s). Token budgets are running on the "
                f"local estimate, which reads low on assistant content."
            )
        status = "degraded" if reasons else "ok"

    return {
        "status": status,
        "status_reasons": reasons,
        "checks": {
            "vllm": vllm,
            "storage": storage,
            # None when the caller did not supply it, so "we did not ask" stays
            # distinguishable from "we asked and it is fine" — the same
            # doctrine as indexed_exchanges_total above.
            "tokenize": tokenize,
        },
        "stats": stats,
        "backups": backup_info,
        "memory_writes": writes,
        "background_work": bg,
        "config": {
            "vllm_url": vllm_url,
            "target_tokens": target_tokens,
        },
    }


def status_to_http_code(status: str) -> int:
    """Map a status string to an HTTP code for the /health/full endpoint.
    Used as the Docker HEALTHCHECK target — 200 keeps the container
    healthy, 503 trips the restart policy.

    Deliberately unchanged by v3.1 A11: shedding now degrades `status`, but
    "degraded" still answers 200. Shedding is backpressure — the pool is over
    its ceiling because the box is busy — and restarting the container in the
    middle of that would kill every in-flight chat AND guarantee the loss of
    every tail still outstanding, which is the harm we were trying to report.
    The fix for a health check that could not report degradation is to make it
    report degradation, not to make it restart things. The signal belongs in
    the body, where `status` and `status_reasons` now carry it.
    """
    return 503 if status == "down" else 200
