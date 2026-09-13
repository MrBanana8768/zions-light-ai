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
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx

import facts
import logsetup
import memory
import retrieval
import summarizer
from envcfg import env_float

logger = logging.getLogger("compactor.health")

# Probe timeout — short, because /health/full is hit by HEALTHCHECK
# every 30s and an unresponsive vLLM shouldn't make the probe hang.
#
# READ THROUGH envcfg, NOT `float(os.environ.get(...))` (v3.1.9). The v3.1.7
# R30 sweep converted ~47 sites and missed this one, and it was the worst of
# the seven it missed: it had no `or default` softening at all, so unlike its
# siblings in dedup.py and persona.py it raised on an EMPTY value as well as
# a mistyped one — and runpod.env.template:72 says in its own words that
# RunPod "handles empty values inconsistently", so the empty case is not
# hypothetical. main.py imports this module at module scope, so either raise
# is a container that will not boot, with a ValueError traceback nobody
# connects to a config line. 3.0 is unchanged: `float("3.0")` and the literal
# `3.0` are the same double.
_VLLM_PROBE_TIMEOUT_S = env_float("COMPACTOR_HEALTH_PROBE_TIMEOUT_S", 3.0)


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


# SQLite's rollback-journal magic, from the file format spec. A journal
# whose header still carries it has work in it that was never committed.
_SQLITE_JOURNAL_MAGIC = bytes.fromhex("d9d505f920a163d7")


def probe_sqlite_journal() -> dict:
    """Is OpenWebUI's database sitting next to a HOT rollback journal?

    v3.1.8, and this is the second time it has mattered. On 2026-08-31 the
    MooseFS volume dropped I/O mid-transaction, SQLite tried to roll the
    journal back on every subsequent open, rolling back needs to WRITE, the
    write failed, and OpenWebUI reported 'readonly database' for 24 minutes
    while 1,819 queries failed. On 2026-09-07 an orphaned hot journal sat
    beside a database that was otherwise being written to normally - so
    nothing looked wrong, and nothing in this endpoint said anything.

    WHY THE FILE'S EXISTENCE IS NOT THE SIGNAL. In `delete` journal mode a
    journal is created and removed around every transaction, so a file
    caught mid-write is ordinary. In `persist` mode one is left behind
    deliberately with its header zeroed. Only the 8-byte magic distinguishes
    'this contains an uncommitted transaction' from 'this is debris', which
    is why this reads the header rather than calling os.path.exists.

    AND WHY NOT JUST OPEN THE DATABASE. A second connection cannot tell a
    hot journal from one belonging to a live writer: it has to take a write
    lock to find out, OpenWebUI holds that lock, so the probe fails with the
    same 'readonly database' text whether or not anything is wrong. That
    ambiguity cost real time on 2026-09-07. Reading 8 bytes takes no lock,
    blocks nothing, and cannot be confused by a healthy writer.

    Best-effort by the module's own contract: a probe that cannot answer
    reports that it could not, and never raises into the endpoint.
    """
    db = os.environ.get("WEBUI_SNAPSHOT_DB", "/data/openwebui/webui.db")
    journal = db + "-journal"
    try:
        if not os.path.exists(journal):
            return {"ok": True, "hot": False, "path": journal}
        with open(journal, "rb") as fh:
            head = fh.read(8)
    except OSError as e:
        return {"ok": None, "hot": None, "path": journal,
                "error": f"{type(e).__name__}: {e}"}
    hot = head == _SQLITE_JOURNAL_MAGIC
    return {"ok": not hot, "hot": hot, "path": journal,
            "header": head.hex()}


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
    # THE ROLLUP HAS NO OTHER OBSERVABLE. v3.1.8 gave the skip path a
    # hierarchy rollup and left six ways for it to do nothing, five of
    # them completely silent, with /health/full reporting "ok" either way
    # - the exact failure it was built to fix (a 22-turn soak,
    # last_summarized_turn still 0) reads as a healthy pod. This scan
    # already opens every state file, so the number costs nothing extra.
    worst_lag = 0
    worst_lag_conv = None
    # v3.1.9 (hostile pass 2, H-1). What every readable state file says about
    # how far its hierarchy has got - the same fields maybe_rollup compares to
    # decide whether to write at all. hierarchy_lag cannot see a rollup that
    # never runs (both of its numbers are written BY the rollup), so
    # gather_health_full compares THIS, poll to poll, against the memory
    # tail's decision count, which moves on every turn whether or not the
    # rollup ran. See _hierarchy_progress.
    progress: list[tuple] = []
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
            _seen = state.get("turns_seen")
            _done = state.get("last_summarized_turn")
            if isinstance(_seen, int) and isinstance(_done, int):
                _lag = _seen - _done
                if _lag > worst_lag:
                    worst_lag, worst_lag_conv = _lag, cid
            progress.append((
                cid, _seen, _done, tuple(state.get("tail_fp") or ()),
                state.get("head_fp"), state.get("window_turns"),
                len(state.get("l1") or ()), len(state.get("l2") or ()),
                state.get("l3") is not None,
            ))
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
        "hierarchy_lag": worst_lag,
        "hierarchy_lag_conv": worst_lag_conv,
        "unreadable": unreadable,
        # Private (leading underscore): an opaque in-process hash, compared
        # for equality only and popped by gather_health_full before the
        # payload is returned. It is a hash rather than the tuples because
        # tail_fp is up to _ANCHOR_TURNS digests per conversation. Sorted by
        # conv_id so a directory listing that comes back in a different order
        # is not mistaken for progress.
        "_hierarchy_fingerprint": hash(tuple(sorted(progress, key=lambda t: t[0]))),
    }


# ---------------------------------------------------------------------------
# Aggregated report
# ---------------------------------------------------------------------------

_CLOCK_TOLERANCE_S = 1.0


def probe_snapshot() -> dict:
    """Is the durable copy of webui.db still being written?

    Only meaningful while the sync daemon owns that file. With
    WEBUI_DB_LOCAL=false the snapshot IS the live database, its mtime moves on
    every message, and "stale" would be meaningless - so this reports
    watched=False rather than a number nobody should read.

    With the gate ON, /data/openwebui/webui.db is the ONLY copy that survives
    a pod recreate, and sync_loop is the only thing writing it. When publishing
    fails repeatedly that loop shouts three times and goes quiet, /health/full
    said "ok", chat worked perfectly, and the durability gap was unbounded.
    This is the number that was missing.

    Three intervals, not one: a single missed cycle is ordinary (a publish that
    overran, a refusal that cleared). Three in a row is a daemon that is not
    coming back on its own.
    """
    enabled = os.environ.get("WEBUIDB_SYNC_ENABLED", "").strip().lower() == "true"
    snap = os.environ.get("WEBUI_SNAPSHOT_DB", "/data/openwebui/webui.db")
    # RANGED HERE, because envcfg does not range anything (its own docstring
    # says so) and this value is arithmetic, not a label (v3.1.9, hostile pass
    # 2 H-4). Measured before this: `inf` - reachable from a typo, `1e400`
    # parses to it - made `round(interval)` raise OverflowError, `nan` raised
    # ValueError, neither is an OSError, so the whole probe died into
    # _gather_blocking's catch-all as {"stale": False} and /health/full said
    # "ok" with the durability probe dead. `0` and `-1` failed the other way:
    # a threshold of 0 or -3 made a snapshot written one second ago stale,
    # permanently.
    #
    # The default rather than a refusal, with the bad value reported: the
    # daemon reads the SAME variable, and what it does with it is what the
    # age will show. `inf` never republishes, so the snapshot ages and the
    # stale reason below fires on the default threshold, which is the true
    # report; `0` republishes continuously, so the snapshot is fresh and
    # nothing here should claim otherwise. Clamping to the default is what
    # tailhealth._window_s and bgwork._window_s already do, for the same
    # reason. Not a status reason on its own: the harm, when there is any, is
    # a stale snapshot, and that already is one.
    interval = env_float("WEBUI_DB_SYNC_INTERVAL_S", 300.0)
    interval_error = None
    if not (math.isfinite(interval) and interval > 0):
        interval_error = (
            f"WEBUI_DB_SYNC_INTERVAL_S={os.environ.get('WEBUI_DB_SYNC_INTERVAL_S')!r} "
            f"is not a positive finite number of seconds; judging staleness "
            f"against the 300 s default"
        )
        interval = 300.0
    out: dict[str, Any] = {
        "watched": enabled, "path": snap, "interval_s": round(interval),
        "interval_error": interval_error,
        "age_s": None, "stale": False, "error": None,
    }
    if not enabled:
        return out
    try:
        age = time.time() - os.path.getmtime(snap)
    except OSError as e:
        # The snapshot is supposed to exist whenever the daemon runs. Missing
        # is worse than stale, not better, so it reports as stale with the
        # reason attached rather than as a quiet False.
        out["error"] = f"{type(e).__name__}: {e}"
        out["stale"] = True
        return out
    out["age_s"] = round(age)
    # A SNAPSHOT FROM THE FUTURE IS NOT FRESH (v3.1.9, webuidb hostile pass 2).
    # webuidb stamps the snapshot with the LOCAL database's mtime and skips
    # every cycle while the snapshot's mtime is >= the local one. So a
    # snapshot stamped ahead of real time - a fast pod clock stepped back by
    # NTP, and copy2 carries the poisoned stamp to every later pod - freezes
    # the durable copy with no error and no log line, until wall-clock time
    # catches up. `age > 3 * interval` read the negative age as the freshest
    # snapshot possible: measured age_s=-3600, stale=False, status "ok".
    #
    # _CLOCK_TOLERANCE_S absorbs timestamp rounding only (a filesystem that
    # stores whole seconds can round a just-written stamp up by under one);
    # it is not a window for a clock that is actually wrong.
    if age < -_CLOCK_TOLERANCE_S:
        out["stale"] = True
        out["future_s"] = round(-age)
        out["error"] = (
            f"clock: the snapshot's mtime is {round(-age)}s in the future"
        )
        return out
    out["stale"] = age > 3 * interval
    return out


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

    sqlite_journal = probe_sqlite_journal()

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

    try:
        snapshot = probe_snapshot()
    except Exception as e:
        snapshot = {"watched": None, "stale": False,
                    "error": f"{type(e).__name__}: {e}"}

    return {"storage": storage, "stats": stats, "writes": writes,
            "backups": backups, "sqlite_journal": sqlite_journal,
            "snapshot": snapshot}


# ---------------------------------------------------------------------------
# Is the summary hierarchy advancing at all? (v3.1.9, hostile pass 2 H-1)
# ---------------------------------------------------------------------------

# Rollup-eligible memory-tail decisions that may pass with NO summary state
# changing anywhere before the pod reads degraded.
#
# WHY hierarchy_lag COULD NOT DO THIS. It is turns_seen - last_summarized_turn,
# and both numbers are written by summarizer.maybe_rollup. Every way the
# rollup can do nothing BEFORE maybe_rollup - the master switch, the degrade
# guard, the gate's raw_chars and history conjuncts, a pool that shed it - and
# every way its state write can fail after it (a full disk, a permission
# change, the lone-surrogate C2 case), freezes both numbers, and a frozen pair
# reads as lag 0: caught up. Measured: 25 real turns with every state write
# failing, hierarchy_lag 0, status "ok".
#
# WHAT MOVES REGARDLESS. tailhealth counts every decision on the request path,
# rollup or no rollup. On a healthy pod every accepted turn changes its
# conversation's state file within one tail (maybe_rollup writes whenever the
# position, the anchor or the window signature moved, which a new turn and a
# regenerated reply both do), so "N decisions and nothing on disk changed" is
# a hierarchy that has stopped - whatever stopped it. No main.py change is
# needed: both halves are already observable from here.
#
# WHY 20. It is the existing lag reason's own tolerance, two L1 chunks of 20
# turns = 40 turns = 20 exchanges, restated in exchanges. It is fixed rather
# than derived from COMPACTOR_L1_CHUNK_SIZE because this signal is not about
# chunks - a healthy pod writes state every turn - and because deriving it
# from that unranged knob is H-7's defect. The decisions that legitimately
# write nothing (a regenerate of a byte-identical window, a lossy skip on a
# conversation with no history, a tail still in the queue when the poll
# lands) do not come twenty in a row with no ordinary turn between them.
#
# HARMLESS_SKIP_OUTCOMES are not counted, for the same reason the gate
# declines them: task traffic and empty replies never roll up, on a healthy
# pod or a broken one. A run of empties is tailhealth.EMPTY_RUN_DEGRADE's
# job, and counting them here too would report one outage twice.
HIERARCHY_STALL_DECISIONS = 20

# In-process, compared poll to poll. Written only from gather_health_full on
# the event loop thread, with no await inside _hierarchy_progress, so two
# concurrent /health/full calls cannot interleave inside it.
_progress: dict[str, Any] = {"fingerprint": None, "eligible": 0}


def _reset_hierarchy_progress_for_tests() -> None:
    _progress["fingerprint"] = None
    _progress["eligible"] = 0


def _hierarchy_progress(fingerprint: Any, mt: dict) -> dict:
    """checks.hierarchy: is the switch on, and how many rollup-eligible
    decisions have passed since the summary state last changed.

    POLL-RELATIVE, which is its one limit and is stated here rather than
    discovered: the count starts at the first poll this process answers, and
    progress is attributed to the poll that first sees it. The Docker
    HEALTHCHECK polls every 30 s (Dockerfile), so the error is at most one
    poll's worth of turns, and it errs toward quiet. A pod nobody polls
    reports nothing until it has been polled twice.

    SCOPE, also stated: this is a store-wide heartbeat. A rollup frozen on
    ONE conversation while another conversation advances is masked - that
    is what hierarchy_lag's per-conversation scan is for when maybe_rollup
    runs. The failures this exists for (the switch, the gate, a write path
    that is failing) are not per-conversation.
    """
    out: dict[str, Any] = {
        "enabled": summarizer.enabled(),
        "decisions_since_progress": None,
        "limit": HIERARCHY_STALL_DECISIONS,
        "stalled": False,
    }
    outcomes = mt.get("outcomes") if not mt.get("error") else None
    if fingerprint is None or not isinstance(outcomes, dict):
        # Either half unreadable: None ("not measured"), never 0 ("measured,
        # fine"). A tail counter that raised has no outcomes at all, and the
        # sum below would raise out of the endpoint. Both halves already carry
        # their own "unobservable"/"unreadable" reasons.
        return out
    import tailhealth  # mt came from it, so it is importable
    eligible = sum(
        v for k, v in outcomes.items()
        if k not in tailhealth.HARMLESS_SKIP_OUTCOMES and isinstance(v, int)
    )
    # A changed fingerprint is progress. A count that went DOWN is a counter
    # that was reset (a restart cannot reach here, a test reset can), and
    # measuring against the old base would read as negative.
    if fingerprint != _progress["fingerprint"] or eligible < _progress["eligible"]:
        _progress["fingerprint"] = fingerprint
        _progress["eligible"] = eligible
    since = eligible - _progress["eligible"]
    out["decisions_since_progress"] = since
    # Only while the switch is ON. A hierarchy switched off on purpose is not
    # stalled, and reporting it as one is the always-on warning this module
    # has already had to remove twice (see the COMPACTOR_HIERARCHICAL_SUMMARY
    # note in gather_health_full).
    out["stalled"] = bool(out["enabled"]) and since >= HIERARCHY_STALL_DECISIONS
    return out


def _tokenizer_state() -> dict:
    """checks.tokenizer: main.tokenizer_state(), or why it cannot be read.

    CONTRACT (the v3.1.9 config lane): main.tokenizer_state() -> {"loaded":
    bool, "last_error": str | None, "failed_at": float | None,
    "next_retry_at": float | None}, read-only and cheap.

    HOW main IS REACHED. `import health` runs at main.py module scope, so a
    module-scope `import main` here is a circular import. A call-time
    `import main` would be safe INSIDE the app, where uvicorn has already
    finished importing it - but in any other process that imports health
    without the app, it would boot the whole compactor (logging config,
    FastAPI app, every module main imports) as a side effect of a health
    probe. sys.modules is the same object in the app and nothing at all
    elsewhere, so it is read from there: this function never causes an
    import.
    """
    main_mod = sys.modules.get("main")
    if main_mod is None:
        return {"available": False,
                "reason": "main is not loaded in this process"}
    fn = getattr(main_mod, "tokenizer_state", None)
    if not callable(fn):
        return {"available": False,
                "reason": "main.tokenizer_state() is not present in this build"}
    try:
        st = fn()
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        return {"available": False, "reason": err, "error": err}
    if not isinstance(st, dict):
        err = f"main.tokenizer_state() returned {type(st).__name__}, not a dict"
        return {"available": False, "reason": err, "error": err}
    return {"available": True, **st}


async def gather_health_full(
    vllm_url: str, target_tokens: int, tokenize: dict | None = None
) -> dict:
    """The single source of truth used by /health/full and /admin/selftest.

    Status semantics:
      - "ok"       — all checks pass; serve traffic normally
      - "degraded" — storage OK but something the operator needs to see:
                     vLLM unreachable, new-memory writes paused under disk
                     pressure, background work shedding, or the memory tail
                     skipping replies (v3.1.4). Compactor can still serve
                     admin/export endpoints. Container stays alive so
                     supervisord can restart vLLM independently.
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

    # v3.1.4: memory-tail decisions — replies that did NOT enter memory
    # (facts, episodic index, rollup) and why, as counts. Same shape and same
    # reason as the pool above: until v3.1.4 a skipped tail was a WARNING
    # line and nothing else, so 63 skipped exchanges in one 2026-09-01 log
    # window — more than half her recent conversation — ran for weeks with
    # this endpoint saying ok. Read here directly, like bgwork, because
    # main.py cannot be imported from this module (main imports health).
    try:
        import tailhealth
        mt = tailhealth.snapshot()
    except Exception as e:
        mt = {"error": f"{type(e).__name__}: {e}"}

    # v3.1.9 (H-1). After the tail snapshot, never before it: the decisions
    # are then counted no earlier than the scan that looked at the disk, so a
    # write still in flight errs toward an undercount (quiet) and never
    # toward a stall that did not happen. Popped so the opaque hash never
    # reaches the payload.
    hierarchy = _hierarchy_progress(stats.pop("_hierarchy_fingerprint", None), mt)
    # Pure in-memory read of main's own state; see _tokenizer_state.
    tokenizer = _tokenizer_state()

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
        if mt.get("error"):
            # Unknown is not fine (same doctrine as the pool above).
            reasons.append(f"memory tail unobservable ({mt['error']})")
        elif mt.get("skipped_recently"):
            # Worded on the shed reason above, because it is the same harm
            # seen from the other side: there, the pool dropped the tail;
            # here, the gate declined to run it. `skipped_recently` is the
            # windowed field (tailhealth.SKIP_DEGRADE_WINDOW_S), so a skip
            # degrades while it is happening and for a while after, then
            # clears itself — the cumulative `skipped` is history, not status.
            reasons.append(
                f"memory tail skipping: {mt.get('skipped')} reply(ies) not "
                f"memorized ({mt.get('consecutive_skips')} consecutive, most "
                f"recent {mt.get('seconds_since_last_skip')}s ago, last "
                f"outcome {mt.get('last_skip_outcome')}). New memory is not "
                f"being written for the turns that were skipped."
            )
        # v3.1.9 (H-2). Not an elif: a run of empty replies and a recent lossy
        # skip are different faults and both can be true. skipped_recently is
        # blind to this by design (SKIPPED_EMPTY is harmless ONE at a time,
        # R27), /v1/models keeps answering while the backend generates
        # nothing, and the rollup gate declines raw_chars == 0 - so before this
        # a backend returning empty content read "ok" on every surface.
        if mt.get("empty_replies_degraded"):
            reasons.append(
                f"the backend has returned no text for "
                f"{mt.get('consecutive_empty_replies')} consecutive replies "
                f"(limit {mt.get('empty_run_limit')}; most recent "
                f"{mt.get('seconds_since_last_skip')}s ago). An empty 200, a "
                f"rejected request and a Stop before the first token all look "
                f"like this; that many in a row, with nothing in between that "
                f"carried text, is not a person pressing Stop. Nothing from "
                f"those turns reached memory. vLLM answering /v1/models (the "
                f"vllm check) does not mean it is generating - send it one "
                f"completion by hand."
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
        # v3.1.8 (adversarial state sweep). stats.unreadable was COUNTED and
        # then never consulted: three corrupted conversations present and
        # /health/full still answered status "ok", status_reasons [], with a
        # green Docker HEALTHCHECK.
        #
        # This module's own docstring says a layer we cannot see must not be
        # reported as healthy, and the scan above goes to the trouble of
        # counting each unreadable layer for exactly this purpose. The number
        # was in the payload; nothing read it. An operator would have had to
        # notice a nested count inside a body whose top line said everything
        # was fine.
        _unreadable = stats.get("unreadable") or {}
        _damaged = {k: v for k, v in _unreadable.items() if isinstance(v, int) and v > 0}
        if _damaged:
            reasons.append(
                "unreadable memory on disk: "
                + ", ".join(f"{n} {layer}" for layer, n in sorted(_damaged.items()))
                + ". Those conversations are not being read and must not be "
                "written over; see stats.unreadable."
            )
        # A hot journal is not a maybe: it is an uncommitted transaction
        # that the next process to open the database will try to roll back,
        # and on a network filesystem that rollback is what wedged the pod
        # for 24 minutes on 2026-08-31. Say so while it is still cheap to
        # fix - stop the writers, open it once read-write, and SQLite
        # finishes the job itself.
        _sj = blocking.get("sqlite_journal") or {}
        if _sj.get("hot"):
            reasons.append(
                f"a HOT SQLite rollback journal is sitting beside "
                f"{_sj.get('path')}: an uncommitted transaction that the "
                f"next open will roll back. Stop openwebui and open the "
                f"database once read-write to let SQLite finish it. Do NOT "
                f"delete the journal - it and the database are a matched "
                f"pair, and separating them turns a recoverable file into "
                f"a corrupt one."
            )
        # THE ROLLUP IS OTHERWISE UNOBSERVABLE. v3.1.8's skip-path rollup has
        # six ways to do nothing and five are silent: raw_chars == 0, no
        # conversational history, task traffic, the summarizer disabled, an
        # empty chunk, and a degrade guard that logs at DEBUG while production
        # runs at INFO. The only positive evidence is a "rollup ->" line, at
        # best once per 20 turns. So the failure this feature exists to fix -
        # a 22-turn soak with last_summarized_turn still 0 - reported "ok",
        # empty reasons, green HEALTHCHECK.
        #
        # Two chunks of lag, because one chunk of drift is ordinary: the
        # rollup fires on the tail, so the newest turns are always ahead of
        # the watermark. Twice that is a hierarchy that has stopped keeping up.
        _lag = stats.get("hierarchy_lag")
        _lag_limit = 2 * summarizer.L1_CHUNK_SIZE
        if isinstance(_lag, int) and _lag > _lag_limit:
            reasons.append(
                f"the summary hierarchy is {_lag} turns behind on "
                f"conv={stats.get('hierarchy_lag_conv')} (limit {_lag_limit}). "
                f"Turns past the watermark are carried by the raw window "
                f"alone, so the oldest of them fall out of the request as it "
                f"grows. POST /admin/conversations/<id>/compact drains the "
                f"backlog off the request path."
            )
        # v3.1.9 (H-1). The half hierarchy_lag cannot see: the rollup is not
        # running, or its writes are not landing, so both of the numbers the
        # lag is made of are frozen. See HIERARCHY_STALL_DECISIONS.
        if hierarchy.get("stalled"):
            reasons.append(
                f"the summary hierarchy has not advanced in "
                f"{hierarchy.get('decisions_since_progress')} memory-tail "
                f"decisions (limit {hierarchy.get('limit')}): replies are "
                f"arriving and no conversation's summary state has changed. "
                f"The rollup is being declined before it runs or its state "
                f"write is failing, and hierarchy_lag cannot show either - it "
                f"is written by the rollup itself. Grep the log for "
                f"'rollup state write failed' and 'async rollup failed'."
            )
        # COMPACTOR_HIERARCHICAL_SUMMARY=false is reported in config and in
        # checks.hierarchy, and is deliberately NOT a status reason (v3.1.9,
        # H-3). It is a documented operator switch (RUNPOD_DEPLOY.md), and
        # MEMORY_REVIEW.md §8.1 prescribes a session with it off as the
        # isolation experiment for the double-summary hypothesis. A reason
        # would pin every such pod "degraded" for as long as the experiment
        # runs, and a status that is degraded for a known, chosen reason hides
        # the next reason that is not - the always-on warning this module has
        # removed before. What H-3 found was that the switch had NO observable;
        # it has one now, next to vllm_url, where a leftover `false` in
        # runpod.env is read on every look at the payload.
        #
        # v3.1.9. The local tokenizer, from main.tokenizer_state(). count_tokens
        # is what the hard budget sheds turns by - per message, scaled - and
        # it is the whole budget whenever /tokenize is down. A tokenizer that
        # FAILED to load leaves that on the char/4 estimate, a different
        # instrument from the one the budget was measured against; that is the
        # 2026-08-28 class of degraded, so it is a reason. Only a FAILURE:
        # "not loaded, no error" is the moment before the first load (lifespan
        # warms it at boot), and an absent contract is a build that predates
        # it, not a fault on the pod. Since the load now retries with backoff,
        # the reason clears on the first retry that succeeds, which is what
        # keeps it from being always-on.
        if tokenizer.get("error"):
            reasons.append(f"tokenizer state unobservable ({tokenizer['error']})")
        elif tokenizer.get("loaded") is False and tokenizer.get("last_error"):
            _retry = tokenizer.get("next_retry_at")
            _retry_in = (
                f"next retry in {max(0, round(_retry - time.time()))}s"
                if isinstance(_retry, (int, float)) and math.isfinite(_retry)
                else "no retry scheduled"
            )
            reasons.append(
                f"the local tokenizer failed to load "
                f"({tokenizer.get('last_error')}; {_retry_in}). Token counts "
                f"that do not come from /tokenize are running on the char/4 "
                f"estimate, and the request budget sheds turns by them."
            )
        # A SNAPSHOT THAT HAS STOPPED REFRESHING is the one condition where
        # /data no longer holds a copy that survives a pod recreate, and
        # nothing anywhere reported it: sync_loop shouted three times and went
        # quiet, and this endpoint never looked. Only meaningful while the sync
        # daemon owns that file - with WEBUI_DB_LOCAL=false the snapshot IS the
        # live database and its mtime moves on every message.
        _snap = blocking.get("snapshot") or {}
        if _snap.get("future_s") is not None:
            reasons.append(
                f"the webui.db snapshot on /data has a modification time "
                f"{_snap.get('future_s')}s in the FUTURE. The clock has "
                f"stepped backwards since it was stamped, or it was stamped "
                f"from a clock that ran fast. The sync daemon skips every "
                f"cycle while the snapshot looks newer than the live database, "
                f"so the durable copy is frozen, with no error, until the "
                f"clock catches up - and a pod recreate in that window loses "
                f"everything since. Check `date` against real time."
            )
        elif _snap.get("stale") and _snap.get("age_s") is None:
            reasons.append(
                f"the webui.db snapshot on /data cannot be read "
                f"({_snap.get('error')}). The live database is on local disk, "
                f"which a pod recreate destroys, and this is supposed to be "
                f"its durable copy. Check `supervisorctl status webuidb-sync`."
            )
        elif _snap.get("stale"):
            reasons.append(
                f"the webui.db snapshot on /data is {_snap.get('age_s')}s old "
                f"against a {_snap.get('interval_s')}s sync interval. The live "
                f"database is on local disk, which a pod recreate destroys, so "
                f"this is the durable copy and it is not being written. Check "
                f"`supervisorctl status webuidb-sync`."
            )
        elif _snap.get("error"):
            # v3.1.9 (H-4). The branch every sibling probe had and this one did
            # not (compare bg and mt above). _gather_blocking turns a probe
            # that RAISED into {"watched": None, "stale": False, "error": ...},
            # and reading only `stale` reported that as a healthy snapshot -
            # which is how a non-finite interval killed the durability probe
            # while the endpoint said "ok". Unknown is not fine.
            reasons.append(f"snapshot probe unobservable ({_snap['error']})")
        status = "degraded" if reasons else "ok"

    return {
        "status": status,
        "status_reasons": reasons,
        "checks": {
            "vllm": vllm,
            "storage": storage,
            "sqlite_journal": blocking.get("sqlite_journal"),
            "snapshot": blocking.get("snapshot"),
            # None when the caller did not supply it, so "we did not ask" stays
            # distinguishable from "we asked and it is fine" — the same
            # doctrine as indexed_exchanges_total above.
            "tokenize": tokenize,
            # v3.1.9 (H-1/H-3): the switch, and the heartbeat.
            "hierarchy": hierarchy,
            # v3.1.9: main.tokenizer_state(), or {"available": false, ...}.
            "tokenizer": tokenizer,
        },
        "stats": stats,
        "backups": backup_info,
        "memory_writes": writes,
        "background_work": bg,
        "memory_tail": mt,
        "config": {
            "vllm_url": vllm_url,
            "target_tokens": target_tokens,
            # v3.1.9 (H-3). ENABLED is read once at import, so a leftover
            # COMPACTOR_HIERARCHICAL_SUMMARY=false survives every redeploy;
            # before this it had no observable anywhere.
            "hierarchical_summary": summarizer.enabled(),
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
