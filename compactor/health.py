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

import logging
import os
import time
from pathlib import Path
from typing import Any

import httpx

import facts
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
    per-conv: a single corrupted file doesn't poison the totals.
    """
    conv_ids = memory.list_known_conv_ids()
    facts_total = 0
    indexed_total = 0
    summaries_with_l1 = 0
    summaries_with_l3 = 0
    for cid in conv_ids:
        try:
            facts_total += len(facts.load_facts(cid))
        except Exception:
            pass
        try:
            indexed_total += retrieval.conversation_doc_count(cid)
        except Exception:
            pass
        try:
            state = summarizer.load_state(cid)
            if state.get("l1"):
                summaries_with_l1 += 1
            if state.get("l3"):
                summaries_with_l3 += 1
        except Exception:
            pass
    return {
        "conversations": len(conv_ids),
        "facts_total": facts_total,
        "indexed_exchanges_total": indexed_total,
        "summaries_with_l1": summaries_with_l1,
        "summaries_with_l3": summaries_with_l3,
    }


# ---------------------------------------------------------------------------
# Aggregated report
# ---------------------------------------------------------------------------

async def gather_health_full(vllm_url: str, target_tokens: int) -> dict:
    """The single source of truth used by /health/full and /admin/selftest.

    Status semantics:
      - "ok"       — all checks pass; serve traffic normally
      - "degraded" — storage OK but vLLM unreachable. Compactor can still
                     serve admin/export endpoints. Container stays alive
                     so supervisord can restart vLLM independently.
      - "down"     — storage broken. Nothing useful possible. Container
                     should be replaced.

    Returned 200 for ok+degraded, 503 for down (caller maps).
    """
    vllm = await probe_vllm(vllm_url)
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

    if not storage["ok"]:
        status = "down"
    elif not vllm["ok"] or writes.get("new_memory_writes") == "paused":
        status = "degraded"
    else:
        status = "ok"

    # V2.3 Theme 1: surface backup durability status (best-effort).
    backup_info: dict[str, Any]
    try:
        import backup as backup_module
        backup_info = backup_module.latest_backup_info()
    except Exception as e:
        backup_info = {"count": None, "latest": None, "error": f"{type(e).__name__}: {e}"}

    return {
        "status": status,
        "checks": {
            "vllm": vllm,
            "storage": storage,
        },
        "stats": stats,
        "backups": backup_info,
        "memory_writes": writes,
        "config": {
            "vllm_url": vllm_url,
            "target_tokens": target_tokens,
        },
    }


def status_to_http_code(status: str) -> int:
    """Map a status string to an HTTP code for the /health/full endpoint.
    Used as the Docker HEALTHCHECK target — 200 keeps the container
    healthy, 503 trips the restart policy.
    """
    return 503 if status == "down" else 200
