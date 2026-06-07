"""
compactor.degrade — V2.3 Theme 2: graceful degradation under disk pressure.

The system already degrades memory *reads* to no-ops on failure (a corrupt
facts file → empty list, ChromaDB unavailable → no retrieval). This module
adds the *write* side: when `/data` is nearly full, stop persisting NEW
memory rather than crashing on a failed write — keep serving chat.

What gets gated (the memory-GROWTH paths):
  - fact extraction (async tail)
  - episodic indexing (async tail)
  - hierarchical summary rollup (async tail)
  - persona auto-capture (request path)

What is NOT gated (deliberate):
  - chat itself (reads only — must always work)
  - explicit user writes: `/remember`, admin set-persona, import. These are
    deliberate, rare, and small; a user choosing to write should not be
    silently dropped. (`/forget` and deletes free space — also never gated.)

The free-space check is cached with a short TTL so a burst of requests
doesn't hammer `statvfs`. Surfaced in `/health/full` so the operator sees
"serving but not persisting" before it becomes a mystery.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from pathlib import Path

logger = logging.getLogger("compactor.degrade")

# Block new-memory writes when free space on the storage volume drops below
# this. Distinct from (and higher than) the backup min-free guard — we want
# to stop *growing* memory well before the disk is truly full, leaving
# headroom for the backup daemon and for in-flight writes to complete.
MIN_FREE_MB_WRITES = int(
    os.environ.get("COMPACTOR_MIN_FREE_MB_WRITES", "200") or 200
)

# How long a free-space reading is trusted before re-checking. Keeps a
# request burst from calling statvfs hundreds of times a second.
_CHECK_TTL_S = float(os.environ.get("COMPACTOR_DEGRADE_CHECK_TTL_S", "10") or 10)

# What volume to watch. Defaults to the storage root's filesystem.
_WATCH_PATH = os.environ.get(
    "COMPACTOR_STORAGE_ROOT", "/data/openwebui/compactor"
)

# Cache: (monotonic_expiry, allowed, free_mb)
_cache: tuple[float, bool, float] | None = None
# So we only log the transition into/out of blocked state, not every check.
_last_logged_blocked: bool = False


def _free_mb(path: str) -> float:
    """Free MB on the filesystem holding `path` (or nearest existing
    ancestor, since the dir may not exist yet). inf if undeterminable —
    we fail OPEN (allow writes) rather than block on a bad reading."""
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    try:
        return shutil.disk_usage(str(p)).free / (1024 * 1024)
    except Exception:
        return float("inf")


def writes_allowed() -> tuple[bool, float]:
    """Return (allowed, free_mb). Cached for _CHECK_TTL_S.

    Fails OPEN: if free space can't be read, writes are allowed (inf MB) —
    a broken probe must not silently stop all persistence.
    """
    global _cache, _last_logged_blocked
    now = time.monotonic()
    if _cache is not None and now < _cache[0]:
        return _cache[1], _cache[2]

    free = _free_mb(_WATCH_PATH)
    allowed = free >= MIN_FREE_MB_WRITES
    _cache = (now + _CHECK_TTL_S, allowed, free)

    # Log only on state transition, so the log shows the moment it tripped
    # and the moment it recovered — not a line per check.
    if (not allowed) and (not _last_logged_blocked):
        logger.warning(
            f"DISK PRESSURE: {free:.0f} MB free at {_WATCH_PATH} "
            f"(< {MIN_FREE_MB_WRITES} MB) — pausing new-memory writes; "
            f"chat continues, explicit user writes still allowed"
        )
        _last_logged_blocked = True
    elif allowed and _last_logged_blocked:
        logger.info(
            f"disk pressure cleared: {free:.0f} MB free — resuming "
            f"new-memory writes"
        )
        _last_logged_blocked = False

    return allowed, free


def guard(operation: str) -> bool:
    """Convenience for the call sites: returns True if `operation` may
    proceed. When blocked, logs at debug (the transition was already warned
    in writes_allowed) and returns False so the caller skips the write."""
    allowed, free = writes_allowed()
    if not allowed:
        logger.debug(
            f"skipping {operation}: disk pressure ({free:.0f} MB free)"
        )
    return allowed


def write_state() -> dict:
    """For /health/full. Reports whether new-memory writes are currently
    being persisted, and the free-space reading behind that decision."""
    allowed, free = writes_allowed()
    return {
        "new_memory_writes": "allowed" if allowed else "paused",
        "free_mb": round(free, 1) if free != float("inf") else None,
        "min_free_mb": MIN_FREE_MB_WRITES,
        "watch_path": _WATCH_PATH,
    }


def _reset_cache_for_tests() -> None:
    """Test helper — clear the TTL cache + log-transition state."""
    global _cache, _last_logged_blocked
    _cache = None
    _last_logged_blocked = False
