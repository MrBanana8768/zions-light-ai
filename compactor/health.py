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

# v3.1.9 (hostile317-c F4, HIGH). The best proxy this module has for "how
# long has this pod been up", read at IMPORT time. health.py is imported by
# main.py near the top of boot (module scope), and the backup daemon is
# started by the same entrypoint.sh within moments of it — there is no IPC
# between the two processes, so this is an approximation, stated here rather
# than hidden, the same way _hierarchy_progress states that it is
# poll-relative. Used only for the backup-freshness grace window below: a
# pod that has not been up for one whole COMPACTOR_BACKUP_INTERVAL_HOURS yet
# has not necessarily had time for its first cycle to run and publish.
_PROCESS_STARTED_AT = time.time()


# v3.1.9 (hostile pass 3, F7). Test-only override for _container_started_at,
# so existing tests that drive the grace window via
# _reset_process_started_at_for_tests keep working unchanged — see that
# function and _container_started_at's docstring. None (the default) means
# "read the real /proc files"; production never sets this.
_CONTAINER_STARTED_AT_OVERRIDE: float | None = None
_CONTAINER_STARTED_AT_OVERRIDE_SET = False


def _reset_process_started_at_for_tests(t: float | None = None) -> None:
    """Reset BOTH uptime signals health.py can consult to the same value:
    `_PROCESS_STARTED_AT` (the pre-F7 clock, still the fallback) and the
    `_container_started_at()` override (the new primary clock). One lever,
    same as before this fix — a test that wants the two signals to DISAGREE
    (proving F7's actual point: a compactor-only respawn must not reset the
    grace) uses `_set_container_started_at_for_tests` instead, independently.
    """
    global _PROCESS_STARTED_AT, _CONTAINER_STARTED_AT_OVERRIDE, _CONTAINER_STARTED_AT_OVERRIDE_SET
    v = time.time() if t is None else t
    _PROCESS_STARTED_AT = v
    _CONTAINER_STARTED_AT_OVERRIDE = v
    _CONTAINER_STARTED_AT_OVERRIDE_SET = True


def _set_container_started_at_for_tests(t: float | None) -> None:
    """Independently override `_container_started_at()`'s return value.
    `t=None` clears the override (falls back to the real /proc read, or to
    `_PROCESS_STARTED_AT` if that also fails) — distinct from
    `_reset_process_started_at_for_tests`, which moves BOTH signals
    together. This one exists so a test can hold the container's start
    fixed while `_PROCESS_STARTED_AT` moves (simulating a compactor-only
    respawn) — the exact scenario F7 part 2 is about.
    """
    global _CONTAINER_STARTED_AT_OVERRIDE, _CONTAINER_STARTED_AT_OVERRIDE_SET
    _CONTAINER_STARTED_AT_OVERRIDE = t
    _CONTAINER_STARTED_AT_OVERRIDE_SET = t is not None


def _clear_container_started_at_override_for_tests() -> None:
    """Fully clear the override (distinct from passing t=None to
    _set_container_started_at_for_tests, which also clears it — this name
    exists for readability at call sites that just want 'stop overriding,
    go back to the real /proc read')."""
    global _CONTAINER_STARTED_AT_OVERRIDE, _CONTAINER_STARTED_AT_OVERRIDE_SET
    _CONTAINER_STARTED_AT_OVERRIDE = None
    _CONTAINER_STARTED_AT_OVERRIDE_SET = False


def _container_started_at() -> float | None:
    """Best-effort wall-clock epoch of PID 1's start — i.e. this CONTAINER's
    own start, not this process's. None if it cannot be read (non-Linux, no
    /proc, permission denied, or a malformed stat line).

    v3.1.9 (hostile pass 3, F7). `_PROCESS_STARTED_AT` above is read at
    health.py's OWN import time, inside the COMPACTOR process specifically —
    and the compactor is not the backup daemon. Production supervisord logs
    show compactor-only respawns with no backup-daemon restart alongside
    them (2026-08-31, 2026-09-01) and deploy windows with 6 pod boots inside
    25 minutes; on a genuinely FRESH pod either clock is fine, but on a
    compactor crash-loop or a burst of hot-patch restarts,
    `_PROCESS_STARTED_AT` keeps sliding forward to "now" on every respawn —
    so a pod whose backups have NEVER succeeded, ever, can sit at
    `status: ok` indefinitely, because the grace window's clock never
    accumulates real elapsed time. The daemon itself would be the honest
    clock (a `.daemon_started` stamp, or a last-cycle timestamp it writes)
    but backup.py is not a file this lane may edit — see the report for
    exactly what such a stamp would need.

    PID 1 is the closest available proxy that does NOT reset on a
    compactor-only respawn: under supervisord (this project's entrypoint)
    PID 1 is supervisord itself, alive for the container's whole life
    regardless of how many times any one of its children — compactor OR
    backup — gets restarted. It DOES reset on what actually matters here: a
    pod recreate, a full container restart. That is exactly the survives-a-
    child-restart, resets-on-a-real-reboot signal this grace window needs.

    Mechanism: /proc/uptime's first field is seconds-since-boot (of the
    kernel the container's namespace shares with the host, monotonic and
    independent of wall-clock changes), and /proc/1/stat's 22nd
    whitespace-separated field (after skipping the parenthesized comm name,
    which can itself contain spaces or parens) is PID 1's start time in
    clock ticks since boot. `os.sysconf("SC_CLK_TCK")` converts ticks to
    seconds (typically 100 on Linux). PID1_start_epoch = now - uptime_s +
    (start_ticks / clk_tck) is then an absolute wall-clock estimate of when
    PID 1 (the container) started.

    Best-effort per this module's own doctrine: ANY failure (file missing,
    permission denied, unexpected format) returns None, and callers fall
    back to `_PROCESS_STARTED_AT` — the exact pre-fix behavior — rather than
    raising into a health poll or guessing a value that looks precise and
    is not.
    """
    if _CONTAINER_STARTED_AT_OVERRIDE_SET:
        return _CONTAINER_STARTED_AT_OVERRIDE
    try:
        with open("/proc/uptime", "r", encoding="ascii") as fh:
            uptime_s = float(fh.read().split()[0])
        with open("/proc/1/stat", "r", encoding="ascii") as fh:
            stat_line = fh.read()
        # comm (field 2) is parenthesized and may itself contain ")" or
        # spaces (a process renamed via prctl); fields are unambiguous only
        # after the LAST ")", per proc(5).
        after_comm = stat_line.rsplit(")", 1)[1].split()
        # after_comm[0] is field 3 (state); starttime is field 22, i.e.
        # index 22 - 3 = 19 into this remainder.
        start_ticks = float(after_comm[19])
        clk_tck = os.sysconf("SC_CLK_TCK")
        if clk_tck <= 0:
            return None
        now = time.time()
        return now - uptime_s + (start_ticks / clk_tck)
    except Exception:
        return None


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


def _resolve_live_webui_db() -> tuple[Any, str | None]:
    """The SQLite file OpenWebUI actually has open, or (None, why-not).

    v3.1.9 (hostile2-backup A3-8). Calls backup.live_webui_db() rather than
    re-deriving the same rule a third time: webuidb.py's own module docstring
    is where the LOCAL_DB / SNAPSHOT_DB split is explained, and
    backup.live_webui_db() (1ecd4b6) is the one place that already resolves
    "which of those OpenWebUI is reading right now" the way entrypoint.sh
    decides it — explicit override, then DATABASE_URL, then the gate compared
    EXACTLY as entrypoint.sh compares it. A second, independent reading of
    WEBUI_DB_LOCAL here could disagree with that one on a typo or a future
    edit, which is precisely the reader-disagrees-with-writer defect
    live_webui_db()'s own docstring was written to close. backup.py is
    already imported in this same request (_gather_blocking imports it for
    latest_backup_info()), so this costs nothing extra.

    CAVEAT, stated rather than hidden: backup.live_webui_db() also honors
    COMPACTOR_BACKUP_WEBUI_DB, an override meant for "where backup.py reads
    from", not "where OpenWebUI writes to". An operator who sets it to
    something other than the live database would make this probe watch that
    same, deliberately different, file. That is an intentional operator
    override read the way its own module documents it, not a defect this
    probe introduces.
    """
    try:
        import backup as backup_module
        return backup_module.live_webui_db(), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _check_one_journal(db_path: str) -> dict:
    """The magic-header check, isolated to one candidate path so
    probe_sqlite_journal can run it against more than one file without
    duplicating the read/parse logic.

    v3.1.9 (hostile pass 3, F4). The magic header alone is NOT "an
    uncommitted transaction, never a healthy writer" the way
    probe_sqlite_journal's own docstring used to claim. SQLite writes the
    magic into the journal header at commit AND at every mid-transaction
    page-cache spill (the default `synchronous` syncs the journal before it
    touches the database), and OpenWebUI stores a whole conversation in one
    `chat` row (27-33MB for the production one), so a single UPDATE of that
    row spills and carries the magic header for the ENTIRE write, seconds
    long, while the writer holds SQLite's RESERVED lock throughout. Measured
    (reviewer, local disk): 72% of the time spent writing that row read as a
    hot journal. That is a live transaction, not the debris this reason
    exists to catch, and reporting it "degraded... stop openwebui" trains
    the operator to ignore the one reason that exists for the 2026-08-31
    outage class.
    backup._probe_reserved_lock (added in 6171faf for restore, same
    underlying SQLite fact: RESERVED means a write transaction is active
    right now) is the second half of SQLite's own definition of "hot" — no
    process holds RESERVED, AND the journal's first byte is non-zero — the
    magic-header check alone only ever implemented the second half. Imported
    rather than re-implemented: one raw fcntl() byte-range probe, used by
    both the restore guard and this one, so they cannot drift apart on what
    "hot" means the way the magic-only check and pager.c's own definition
    already had (reviewer B, "The open discrepancy").
    """
    journal = db_path + "-journal"
    try:
        if not os.path.exists(journal):
            return {"ok": True, "hot": False, "path": journal}
        with open(journal, "rb") as fh:
            head = fh.read(8)
    except OSError as e:
        return {"ok": None, "hot": None, "path": journal,
                "error": f"{type(e).__name__}: {e}"}
    magic_present = head == _SQLITE_JOURNAL_MAGIC
    # v3.1.9 (F4): a live writer holding RESERVED while the magic header is
    # present is a transaction IN PROGRESS, not a stalled recovery — see the
    # docstring. `_probe_reserved_lock` fails OPEN (returns False, "no lock
    # detected") on any platform/mount without POSIX fcntl byte-range locks
    # (Windows; a network filesystem whose client doesn't implement them) —
    # the SAFE direction here, deliberately: on such a mount `writer_active`
    # is always False and `hot` collapses back to the magic-only check this
    # probe has always run, so a real hot journal is never masked by an
    # inability to prove a writer is active. The cost is the false-positive
    # this finding is about staying possible on THAT class of mount — a
    # known, pre-existing gap `_probe_reserved_lock`'s own docstring already
    # names — never a false negative on a genuinely hot journal.
    writer_active = magic_present and _probe_reserved_lock_from_backup(db_path)
    hot = magic_present and not writer_active
    result = {
        "ok": not hot, "hot": hot, "path": journal, "header": head.hex(),
    }
    if magic_present:
        # Surfaced only when it is actually informative (the magic header is
        # present) — a quiet False on every other probe would just be noise.
        result["writer_active"] = writer_active
    return result


def _probe_reserved_lock_from_backup(db_path: str) -> bool:
    """Thin wrapper around backup._probe_reserved_lock so _check_one_journal
    has one call site and this module never copies the fcntl logic itself
    (the brief for F4: "import it; do not copy it"). A lazy import, matching
    _resolve_live_webui_db's own pattern just above — backup.py is already
    imported in this same request by _gather_blocking for latest_backup_info,
    so this costs nothing extra in the request path that matters (the health
    poll), and keeps health.py free of a module-level dependency on backup.py
    for the (rarer) callers that only want the cheap probes.
    """
    try:
        import backup as backup_module
        return bool(backup_module._probe_reserved_lock(Path(db_path)))
    except Exception:
        # Best-effort by this module's own contract (see probe_sqlite_
        # journal's docstring): if the probe itself cannot run (backup.py
        # failed to import, the path is bogus), that is "could not prove a
        # writer is active", not "one is" — same fail-open direction
        # _probe_reserved_lock takes internally on its own missing-fcntl case.
        return False


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
    ambiguity cost real time on 2026-09-07. Reading 8 bytes takes no lock and
    blocks nothing.

    v3.1.9 (hostile pass 3, F4) CORRECTS THE CLAIM ABOVE. The magic header
    by itself CAN be confused by a healthy writer, and routinely is: SQLite
    writes it at every mid-transaction page-cache spill, not only at
    interrupted commit, and OpenWebUI's one-row-per-conversation schema
    spills on every save of a large chat. _check_one_journal now also checks
    whether a RESERVED lock is held (backup._probe_reserved_lock) before
    calling a magic header "hot" — see that function's docstring for the
    mechanism and the measured false-positive rate. The header read is still
    what makes this lock-free and non-blocking; the lock PROBE (a
    non-blocking fcntl try-lock, not a SQLite connection) is what tells a
    live transaction from actual debris.

    Best-effort by the module's own contract: a probe that cannot answer
    reports that it could not, and never raises into the endpoint.

    v3.1.9 (hostile2-backup A3-8, MEDIUM). THIS PROBE CHECKED THE WRONG FILE
    UNDER THE SHIPPED DEFAULT. It read WEBUI_SNAPSHOT_DB unconditionally, but
    that is only where SQLite writes when WEBUI_DB_LOCAL=false. Under the
    shipped default (WEBUI_DB_LOCAL=true, entrypoint.sh + backup.py both
    default it there), OpenWebUI's hot writer is WEBUI_LOCAL_DB on local
    disk, and WEBUI_SNAPSHOT_DB is a periodically-refreshed COPY that
    webuidb.sync_once() writes with its own sqlite3 backup() connection - a
    second, independent place a hot journal can appear, and the one place
    this probe was NOT looking under the default gate. It happened to look
    at the right file in THIS pod's actual production configuration only
    because WEBUI_DB_LOCAL is forced to false here (see MEMORY.md) - which
    made the miss invisible without making it correct.

    Now checks BOTH candidates that can matter: whichever file
    backup.live_webui_db() resolves as the one OpenWebUI has open (see
    _resolve_live_webui_db - this is the file the 2026-08-31 and 2026-09-07
    incidents were both about), and WEBUI_SNAPSHOT_DB when it names a
    DIFFERENT file (webuidb-sync's own writer, "if that still matters" per
    the brief - it is a real SQLite writer whenever WEBUI_DB_LOCAL=true).
    When the two resolve to the SAME path (WEBUI_DB_LOCAL=false: the
    snapshot IS the live db) it is checked once, not twice.

    The merged, top-level `ok`/`hot`/`path` stay the SAME SHAPE this probe
    has always returned (existing callers - the status reason below, and
    test_sqlite_journal.py, which is not this lane's file to edit - read
    only those three keys): hot wins if either candidate is hot; otherwise
    an unreadable candidate wins over 'clean', so a probe that could not
    read ONE of the two files is never reported as if both were fine. Every
    candidate's own result is also kept, in full, under `checked`, so an
    operator (or a future caller) can tell which file the health reason is
    actually about.
    """
    live_db, live_err = _resolve_live_webui_db()
    live_path = str(live_db) if live_db is not None else None
    snap = os.environ.get("WEBUI_SNAPSHOT_DB", "/data/openwebui/webui.db")

    checked: dict[str, dict] = {}
    if live_path is not None:
        checked["live"] = _check_one_journal(live_path)
    else:
        # live_webui_db() could not be resolved at all (backup.py itself
        # failed to import, or raised) - unreadable, not "clean", same
        # doctrine as every other probe in this module.
        checked["live"] = {
            "ok": None, "hot": None, "path": None,
            "error": live_err or "the live webui.db path could not be resolved",
        }
    # normcase + abspath: the two env vars are free-form operator strings
    # (a trailing slash, a relative path, a different case on a
    # case-insensitive test filesystem) and a false "different file" would
    # silently double-check the same journal under two names, which is
    # harmless, while a false "same file" would silently DROP the snapshot
    # check the brief asked for. Comparing loosely favors checking twice.
    same_file = (
        live_path is not None
        and os.path.normcase(os.path.abspath(snap))
        == os.path.normcase(os.path.abspath(live_path))
    )
    if not same_file:
        checked["snapshot"] = _check_one_journal(snap)

    hot_entry = next((v for v in checked.values() if v.get("hot")), None)
    if hot_entry is not None:
        result = dict(hot_entry)
    else:
        unread_entry = next(
            (v for v in checked.values() if v.get("hot") is None), None
        )
        result = dict(unread_entry) if unread_entry is not None else dict(
            checked.get("live") or checked.get("snapshot")
        )
    result["checked"] = checked
    return result


# v3.1.9 (health H-6). One day. A healthy pod writes its summary state on
# every accepted turn (the H-1 comment's own claim, which this reuses), so a
# conversation genuinely lagging RIGHT NOW has had its state file touched
# within the current session - minutes to a few hours, not days. A full day
# is generous headroom for someone who chats once in the morning and once at
# night, and is still nowhere near the "400-day-old junk namespace" shape
# hostile pass 2 measured. A conversation that goes quiet for longer than
# this simply drops out of contention for the reason's conv= name until she
# writes to it again - which is correct: it is not "live" in the sense this
# reason exists to flag, and picking it back up brings it back into scope on
# the next write.
_WORST_LAG_RECENCY_S = 24 * 3600.0


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
    # v3.1.9 (hostile pass 2 MEDIUM, health H-6). worst_lag ABOVE is kept as
    # the true, unfiltered maximum over every readable conversation - a
    # diagnostic number, and shrinking it silently would make stats.hierarchy_lag
    # lie by omission. But it is not what the status REASON below should
    # name: turns_seen and last_summarized_turn are BOTH written only by
    # maybe_rollup (see the H-1 comment above _hierarchy_progress), so a
    # conversation whose summary STATE FILE has not been touched in a long
    # time has a FROZEN lag number, not a growing one - it stopped changing
    # the day its rollup last ran, whatever that number says today. On a
    # store that is 62% abandoned test/junk namespaces (hostile pass 2's own
    # figure), one such conversation can sit at a huge, permanently-frozen
    # lag and win the `_lag > worst_lag` comparison over every conversation
    # actually being chatted in right now - so the reason always named the
    # ancient one and an operator chasing "the summary hierarchy is behind
    # on conv=X" investigated a namespace nobody has touched in over a year
    # while a live one, genuinely falling behind today, was never named.
    #
    # worst_lag_recent tracks the same maximum, but only among conversations
    # whose summary state file's own mtime is within _WORST_LAG_RECENCY_S -
    # the file mtime is the only "when was this touched" signal available
    # without editing summarizer.py (out of this lane's file list; see the
    # report). This is what the status reason uses.
    worst_lag_recent = 0
    worst_lag_recent_conv = None
    # v3.1.9 (hostile pass 2, H-1). What every readable state file says about
    # how far its hierarchy has got - the same fields maybe_rollup compares to
    # decide whether to write at all. hierarchy_lag cannot see a rollup that
    # never runs (both of its numbers are written BY the rollup), so
    # gather_health_full compares THIS, poll to poll, against the memory
    # tail's decision count, which moves on every turn whether or not the
    # rollup ran. See _hierarchy_progress.
    progress: list[tuple] = []
    _lag_scan_now = time.time()
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
                if _lag > worst_lag_recent:
                    try:
                        _state_mtime = summarizer.summary_path(cid).stat().st_mtime
                        _recent = (
                            (_lag_scan_now - _state_mtime) <= _WORST_LAG_RECENCY_S
                        )
                    except OSError:
                        # A lag was computed FROM this state, so the file
                        # exists; an OSError here is a race (deleted between
                        # load_state and this stat) or a permissions change.
                        # Excluded, the same safe direction as "could not
                        # tell" everywhere else in this module.
                        _recent = False
                    if _recent:
                        worst_lag_recent, worst_lag_recent_conv = _lag, cid
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
        # v3.1.9 (H-6). The recency-filtered pair the status reason actually
        # uses; see the comment above worst_lag_recent's initialization.
        "hierarchy_lag_recent": worst_lag_recent,
        "hierarchy_lag_recent_conv": worst_lag_recent_conv,
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
        snap_mtime = os.path.getmtime(snap)
        age = time.time() - snap_mtime
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
    # v3.1.9 (OPEN_ISSUES2 webuidb MEDIUM, "the one durability alarm
    # measures her IDLE TIME"). webuidb.sync_once() stamps the published
    # snapshot with `min(local_mtime, time.time())` - the LOCAL database's
    # own last-WRITE time, not the moment of publish (see its comment at the
    # os.replace/os.utime call: the stamp exists so "has anything changed
    # since the last publish" is answerable next cycle, not to record when
    # the publish happened). So `age = now - snapshot_mtime`, above, measures
    # how long it has been since she last sent a message, not how long it
    # has been since the daemon last ran: a publish that succeeded THIS
    # SECOND, on a pod that has been quiet for eight hours, reports
    # age_s=28800 and (with the un-fixed `age > 3 * interval` below) stale
    # True — measured exactly that shape. And sync_once()'s "unchanged since
    # last sync" branch returns BEFORE os.replace/os.utime, so on a quiet
    # night the snapshot's own ctime goes just as stale as its mtime: no
    # timestamp ON THIS FILE distinguishes "healthy, nothing new to publish"
    # from "the daemon died N hours ago" - which is exactly why this was
    # crying wolf every quiet night.
    #
    # WHAT DOES DISTINGUISH THEM, without webuidb.py writing anything new:
    # WEBUI_LOCAL_DB's own mtime is the live source sync_once() reads FROM,
    # and its own skip condition is `snapshot_mtime >= local_mtime` - so
    # "is the durable copy caught up" is the SAME comparison the daemon
    # itself makes, not "how old is the snapshot in wall-clock time". A
    # quiet night moves neither file, so local_lag_s stays ~0 and nothing
    # fires; real unpublished activity moves the local file and not the
    # snapshot, and THAT growing lag is the unbounded-durability-gap this
    # probe was built to catch in the first place (see the module
    # docstring: "sync_loop shouted three times and went quiet").
    #
    # FALLS BACK SAFELY when WEBUI_LOCAL_DB cannot be read (the daemon is
    # watching but the local overlay is gone, or WEBUI_DB_LOCAL disagrees
    # with what actually exists on disk): local_lag_s stays None and
    # staleness reverts to the exact wall-clock comparison this probe has
    # always made — the known idle-time limitation, not a crash and not
    # silently "fixed" for a case this process cannot see into.
    #
    # SPECIFIED, NOT BUILT HERE (per the brief: do not edit webuidb.py). A
    # signal that survives even the fallback case needs webuidb.py to stamp
    # something OTHER than content mtime - e.g. a sibling file such as
    # `f"{SNAPSHOT_DB}.synced_at"`, written with the WALL-CLOCK time of every
    # completed sync_once() call (published OR skipped-because-unchanged, so
    # a live daemon still "checks in" on a quiet night), containing just the
    # epoch float. If that file starts existing, prefer it outright over
    # both age_s and local_lag_s; nothing here reads it because it does not
    # exist yet.
    local_db = os.environ.get("WEBUI_LOCAL_DB", "/var/lib/openwebui/webui.db")
    local_lag_s: float | None
    try:
        local_lag_s = os.path.getmtime(local_db) - snap_mtime
    except OSError:
        local_lag_s = None
    out["local_lag_s"] = None if local_lag_s is None else round(local_lag_s)
    if local_lag_s is not None:
        out["stale"] = local_lag_s > 3 * interval
    else:
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

    # v3.1.9 (hostile2-health, sibling of H-4). This call was the one probe
    # in this function with no try/except around it: every other probe here
    # (writes, backups, snapshot) catches its own exception because a thread
    # hop must not turn one broken probe into a 500 from the whole endpoint —
    # the module docstring's own promise. probe_sqlite_journal() now also
    # imports backup (_resolve_live_webui_db), which is one more way it can
    # raise something that is not the OSError its own internals already
    # catch. Same shape as the others: an exception here degrades this one
    # check, never the endpoint.
    try:
        sqlite_journal = probe_sqlite_journal()
    except Exception as e:
        sqlite_journal = {"ok": None, "hot": None,
                           "error": f"{type(e).__name__}: {e}"}

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

    # p4-b G3. An in-flight/stale backup.py::restore_backup() marker,
    # checked the SAME placement-independent way entrypoint.sh's own boot
    # gate does (webuidb.py --check-restore-marker) and restore_backup's own
    # refusal to start a second restore over an earlier one's marker: this
    # is the third of the three places G3 asks the marker be read, so a
    # marker left by a kill is visible here even on a pod that has already
    # booted past entrypoint.sh's own refusal (an older image, or the
    # marker appearing from an out-of-band `--restore` run AFTER boot).
    # find_interrupted_restore() only globs a small forensics directory and
    # reads one small JSON file — it never opens webui.db, so this costs
    # nothing on the same volume the sqlite_journal probe above already
    # worries about the read reliability of.
    try:
        import webuidb
        _marker = webuidb.find_interrupted_restore()
        restore_marker = {"present": _marker is not None, "marker": _marker}
    except Exception as e:
        restore_marker = {"present": None, "marker": None,
                           "error": f"{type(e).__name__}: {e}"}

    return {"storage": storage, "stats": stats, "writes": writes,
            "backups": backups, "sqlite_journal": sqlite_journal,
            "snapshot": snapshot, "restore_marker": restore_marker}


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
        elif writes.get("new_memory_writes") == "unknown":
            # v3.1.9 (hostile2-health, sibling of H-4). _gather_blocking's
            # except turns a raising degrade.write_state() into
            # {"new_memory_writes": "unknown", "error": ...} and only the
            # "paused" branch above ever read this field. A degrade probe
            # that cannot answer is not the same as one that answered "we are
            # writing fine" - same doctrine as bg/mt below.
            reasons.append(
                f"new-memory write state unobservable "
                f"({writes.get('error')})"
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
            # v3.1.9 (hostile pass 3, F10, LOW — wording only). All FOUR
            # numbers below are keyed on ANY skip, harmless ones (task
            # traffic, a deliberate /forget, a duplicate) included, not just
            # lossy ones — only the TRIGGER (skipped_recently) uses
            # HARMLESS_SKIP_OUTCOMES-aware windowing. OPERATIONS.md already
            # warns the `last outcome` alone can read harmless; it does not
            # say the COUNT and the "most recent" age can be inflated by the
            # same harmless traffic — under the connection-header identity
            # route, OpenWebUI's own follow-up call lands seconds after every
            # real turn as skipped_task_traffic, so by the time anyone reads
            # this reason all four fields typically describe THAT, not the
            # one lossy skip that actually triggered it (measured: 1,500
            # task-traffic skips against 1 real one gave "1503 reply(ies)
            # not memorized... last outcome skipped_task_traffic"). A
            # genuinely lossy-only count/clock/outcome needs tailhealth.py
            # (not this lane's file) to track them separately — see the
            # report for exactly what that needs. This wording caveat is the
            # health-side half: honest about what the numbers already are,
            # without fabricating precision this module does not have.
            reasons.append(
                f"memory tail skipping: {mt.get('skipped')} reply(ies) not "
                f"memorized since start ({mt.get('consecutive_skips')} "
                f"consecutive skips of ANY kind, most recent "
                f"{mt.get('seconds_since_last_skip')}s ago, that skip's "
                f"outcome {mt.get('last_skip_outcome')}) — these counts "
                f"include harmless skips (task traffic, /forget, duplicates), "
                f"not only lossy ones; see OPERATIONS.md. New memory is not "
                f"being written for the turns that were actually skipped."
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
        # v3.1.9 (hostile317-c F4, HIGH — live in production; re-checked at
        # this HEAD, where the stats.unreadable half above was already fixed
        # in v3.1.8, and only THIS half was still open). `backups` has been
        # in the payload since V2.3 Theme 1 and nothing ever turned it into a
        # reason: `.stats.unreadable` above is the OTHER half of the same
        # finding and it was fixed; `.backups` was not. Demonstrated on her
        # real store both with unreadable memory files present AND with
        # `backups: {count: 0, latest: null}` — status "ok" either way.
        #
        # ENV NAMES AND DEFAULTS ARE THE DAEMON'S OWN (backup.py, read via
        # envcfg the same way the rest of this module reads config; and
        # entrypoint.sh, which normalizes COMPACTOR_BACKUP_ENABLED through
        # _bool before this process ever sees it, default true, so a bare
        # `== "true"` compare here agrees with the writer exactly — the same
        # reasoning WEBUIDB_SYNC_ENABLED's compare above already uses, and
        # unlike it COMPACTOR_BACKUP_ENABLED IS operator-facing, which is
        # exactly why entrypoint.sh runs it through _bool first). A pod
        # deployed with backups turned off on purpose is not a fault, the
        # same doctrine as COMPACTOR_HIERARCHICAL_SUMMARY=false (H-3): the
        # switch is documented, and a reason that fires on a chosen
        # configuration trains the operator to ignore this endpoint.
        _backup_enabled = (
            os.environ.get("COMPACTOR_BACKUP_ENABLED", "true").strip().lower()
            == "true"
        )
        if _backup_enabled:
            # Same clamp probe_snapshot already applies to
            # WEBUI_DB_SYNC_INTERVAL_S, for the same reason: this value is
            # arithmetic, not a label, and envcfg does not range it.
            #
            # v3.1.9 (hostile pass 3, F8, LOW). THIS CLAMP IS FOR HEALTH'S OWN
            # ARITHMETIC ONLY — it does NOT mean the backup DAEMON (backup.py,
            # not a file this lane may edit) is protected the same way. The
            # daemon reads COMPACTOR_BACKUP_INTERVAL_HOURS unranged and uses
            # it AS GIVEN: 0 or negative makes `time.sleep(min(interval,
            # RETRY_BACKOFF_S))` / `time.sleep(interval)` publish back-to-back
            # with no real pause (measured: 21 archives in 20s), filling
            # /data until COMPACTOR_BACKUP_MIN_FREE_MB trips a failure loop;
            # a non-finite value crashes `time.sleep` outright under
            # supervisord. Clamping SILENTLY here means that misconfiguration
            # produced no reason at all — freshness looked "ok" (archives a
            # few seconds old) throughout. This is the health-only half of
            # the fix: name the misconfiguration itself as a reason, since
            # this module cannot range the DAEMON's own reading without
            # editing backup.py. What backup.py would need instead: range
            # INTERVAL_HOURS once, at its own definition (`h if
            # math.isfinite(h) and h > 0 else 24`, floored around 0.1h so an
            # operator CAN legitimately configure sub-hourly backups without
            # tripping this), and have health read `backup.INTERVAL_HOURS`
            # directly instead of re-reading the environment, so the two
            # cannot disagree again.
            _bk_interval_h_raw = env_float("COMPACTOR_BACKUP_INTERVAL_HOURS", 24.0)
            if not (math.isfinite(_bk_interval_h_raw) and _bk_interval_h_raw > 0):
                reasons.append(
                    f"COMPACTOR_BACKUP_INTERVAL_HOURS={_bk_interval_h_raw!r} "
                    f"is not a positive, finite number of hours. The backup "
                    f"daemon uses this value exactly as given, unlike this "
                    f"health check (which clamps to 24h for its own "
                    f"freshness arithmetic below): zero or negative makes it "
                    f"publish archives back-to-back with no pause until the "
                    f"volume fills; a non-finite value crashes the daemon "
                    f"outright. Fix the environment variable; do not rely on "
                    f"this check's own clamp to mean the daemon is safe."
                )
            _bk_interval_h = (
                _bk_interval_h_raw
                if math.isfinite(_bk_interval_h_raw) and _bk_interval_h_raw > 0
                else 24.0
            )
            _bk_interval_s = _bk_interval_h * 3600.0
            _bk_now = time.time()
            if backup_info.get("error"):
                # v3.1.9 (sibling of H-4). latest_backup_info() raising is
                # caught in _gather_blocking as {"count": None, "latest":
                # None, "error": ...} and, like sqlite_journal/writes above,
                # nothing here ever read the error half — a probe that could
                # not answer read exactly like an empty-but-fine backup dir.
                reasons.append(
                    f"backup status unobservable ({backup_info['error']})"
                )
            elif backup_info.get("count") == 0:
                # v3.1.9 (hostile pass 3, F7, part 3). list_backups (backup.py)
                # uses Path.glob, which swallows PermissionError and returns
                # an empty result — so an unreadable backup directory reads
                # here as "count: 0" with no error, exactly like a genuinely
                # empty one. backup.py is not a file this lane may edit (its
                # own fix is a `os.scandir` rewrite so the OSError propagates
                # — noted below), so this checks readability INDEPENDENTLY,
                # from health.py's side, before trusting a zero count: a
                # directory this process cannot even list is unobservable,
                # not "zero archives, if a restore were needed right now
                # there is nothing to restore" — a materially different and
                # more alarming claim than "I could not check."
                _bk_dir = backup_info.get("dir")
                _bk_dir_err: str | None = None
                if _bk_dir:
                    try:
                        with os.scandir(_bk_dir):
                            pass
                    except FileNotFoundError:
                        # The directory does not exist YET — a genuinely
                        # fresh pod that has not run its first backup cycle
                        # at all (run_daemon creates BACKUP_DIR on its first
                        # write). This is the SAME "zero archives" case the
                        # grace below already handles, not an unreadable
                        # one — Path.glob (backup.py's own list_backups)
                        # treats a missing directory as empty too, so
                        # disagreeing here would make this process refuse to
                        # boot cleanly on turn one for no reason.
                        pass
                    except OSError as e:
                        # PermissionError and everything else genuinely
                        # means "this process could not read a directory
                        # that DOES exist" — the actual F7 part-3 case.
                        _bk_dir_err = f"{type(e).__name__}: {e}"
                if _bk_dir_err is not None:
                    reasons.append(
                        f"backup status unobservable: {_bk_dir} could not be "
                        f"listed ({_bk_dir_err}). latest_backup_info() reports "
                        f"this as zero archives, which is not the same claim — "
                        f"an unreadable directory means this pod cannot tell "
                        f"whether backups exist, not that they don't."
                    )
                else:
                    # GRACE, NOT A REFUSAL. A pod that just booted has not had
                    # a full cycle yet — run_daemon fires almost immediately
                    # on a truly empty backup dir, but "almost immediately"
                    # still means staging, verifying and publishing an
                    # archive, which takes real time (measured, cross-mount,
                    # up to ~1,790s for her real store). Reporting zero
                    # backups as a fault on turn one of a pod's life would
                    # make every fresh deploy read degraded for no reason.
                    #
                    # v3.1.9 (hostile pass 3, F7, parts 1-2). TWO separate
                    # fixes to the grace itself:
                    #
                    # (1) THE WINDOW WAS A FULL BACKUP INTERVAL (24h default).
                    # run_daemon retries a failed cycle every
                    # COMPACTOR_BACKUP_RETRY_BACKOFF_S (900s = 15min), and a
                    # cycle on her real store takes 13-40s — so "still zero
                    # archives" after even two hours already means roughly
                    # eight consecutive failed cycles, not a pod still
                    # waiting on its first one. Waiting a full 24h to say so
                    # is the worst durability state there is (nothing at all
                    # to restore) staying silent for a day. The window is now
                    # `min(interval, 2h)` — a cycle plus several retries, per
                    # the finding's own framing — capped at 2h regardless of
                    # how long COMPACTOR_BACKUP_INTERVAL_HOURS is configured,
                    # since the retry cadence that bounds "how long is one
                    # legitimate attempt" does not get slower just because the
                    # SUCCESS cadence is configured slower.
                    #
                    # (2) THE CLOCK WAS THE COMPACTOR'S OWN IMPORT TIME, not
                    # the backup daemon's, so a compactor-only respawn (crash
                    # loop, a burst of hot-patch restarts) reset the grace
                    # window to full every time even though the backup daemon
                    # itself never restarted and never got closer to
                    # succeeding. `_container_started_at()` (PID 1 / the
                    # container's own start) survives that respawn and only
                    # resets on an actual pod recreate — see its docstring.
                    # Falls back to `_PROCESS_STARTED_AT` (the pre-fix clock)
                    # when unreadable (non-Linux, no /proc), so this is never
                    # WORSE than before, only more honest where it can be.
                    _uptime_anchor = _container_started_at()
                    if _uptime_anchor is None:
                        _uptime_anchor = _PROCESS_STARTED_AT
                    _uptime = _bk_now - _uptime_anchor
                    _bk_grace_s = min(_bk_interval_s, 2 * 3600.0)
                    if _uptime >= _bk_grace_s:
                        reasons.append(
                            f"COMPACTOR_BACKUP_ENABLED is true, this pod has "
                            f"been up {round(_uptime)}s (past the "
                            f"{round(_bk_grace_s)}s grace for a first "
                            f"archive), and {backup_info.get('dir')} holds "
                            f"zero archives. If a restore were needed right "
                            f"now there is nothing to restore. Run "
                            f"`backup.py --once` by hand and read its output."
                        )
            else:
                _latest_mtime = backup_info.get("latest_mtime")
                if isinstance(_latest_mtime, (int, float)):
                    _bk_age = _bk_now - _latest_mtime
                    # 1.5x, not 1x: a single cycle that overran (a slow
                    # volume, a large store) is ordinary and must not fire on
                    # its own — hostile317-c F4's own suggested threshold.
                    if _bk_age > 1.5 * _bk_interval_s:
                        reasons.append(
                            f"the newest backup ({backup_info.get('latest')}) "
                            f"is {round(_bk_age)}s old against a "
                            f"{round(_bk_interval_s)}s backup interval "
                            f"(limit {round(1.5 * _bk_interval_s)}s). The "
                            f"daemon has not published a new archive in over "
                            f"one and a half cycles. Check "
                            f"`supervisorctl status backup` and the tail of "
                            f"its log."
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
        elif _sj.get("error"):
            # v3.1.9 (hostile2-health, sibling of H-4). probe_sqlite_journal
            # returns {"ok": None, "hot": None, "error": ...} on an OSError
            # reading a candidate's journal (a permission change, an
            # unmounted volume) and only `hot` was ever read here - so a
            # probe that could not answer read exactly like a probe that
            # answered "clean". Unknown is not fine, same doctrine as bg,
            # mt, backups and writes below.
            reasons.append(f"sqlite journal probe unobservable ({_sj['error']})")
        # p4-b G3: the third of the three places this finding asks the
        # in-flight/stale restore marker be checked (the other two:
        # entrypoint.sh before OpenWebUI starts, and restore_backup itself
        # before starting a new restore over an earlier one's marker). A
        # marker here means backup.py::restore_backup was interrupted, or
        # finished (successfully or via a full rollback) without reaching
        # its own removal call — either way the live webui.db and/or
        # compactor store may be missing or at a mixed generation, and
        # nothing else on this endpoint would say so: the census/payload
        # checks above compare archives to each other, not to what is
        # actually live right now.
        _rm = blocking.get("restore_marker") or {}
        if _rm.get("present"):
            _rm_marker = _rm.get("marker") or {}
            reasons.append(
                f"an in-flight/stale restore marker is present at "
                f"{_rm_marker.get('marker_path')}: backup.py::restore_backup "
                f"was interrupted, or finished without removing its own "
                f"marker. The live webui.db and/or compactor store MAY be "
                f"missing or at a mixed generation. A boot refuses on this "
                f"same marker (entrypoint.sh, before OpenWebUI starts); if "
                f"the pod is already up, the marker is stale from before "
                f"this boot or was left by a restore run after it. See "
                f"OPERATIONS.md's restore section for what the marker's "
                f"plan means and how to confirm the live state and clear it "
                f"safely."
            )
        elif _rm.get("error"):
            reasons.append(f"restore-marker probe unobservable ({_rm['error']})")
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
        #
        # v3.1.9 (health H-6). Named from hierarchy_lag_RECENT, not the raw
        # hierarchy_lag: see gather_memory_stats' worst_lag_recent comment.
        # The raw max stays in stats for anyone reading the payload directly;
        # the reason names only a conversation whose state file was touched
        # in the last day, so a permanently-frozen ancient conversation
        # cannot mask a live one from the operator investigating this line.
        _lag = stats.get("hierarchy_lag_recent")
        # v3.1.9 (health H-7). env_int does not range anything (its own
        # docstring says so - callers disagree about what a legal range is),
        # and L1_CHUNK_SIZE is read through it unranged. A `0` here made
        # `2 * 0 == 0` a limit that fires on lag 1 - one turn of ordinary
        # drift degrading the pod - and a NEGATIVE L1_CHUNK_SIZE made the
        # limit negative, which degrades a pod with ZERO conversations: with
        # no conv_ids the scan above never runs, hierarchy_lag stays its
        # init value 0, and `0 > -N` is true, printing "conv=None" in the
        # reason below - a store with nothing in it, reported as behind.
        # Floored at the computation this project's own default
        # (COMPACTOR_L1_CHUNK_SIZE=20) produces, the same shape as
        # backup.MIN_KEEP and pgarchive.MIN_KEEP: a bad env value degrades to
        # a sane number, never to a value that cannot decide anything.
        _lag_limit = 2 * summarizer.L1_CHUNK_SIZE
        if _lag_limit <= 0:
            _lag_limit = 2 * 20
        if isinstance(_lag, int) and _lag > _lag_limit:
            reasons.append(
                f"the summary hierarchy is {_lag} turns behind on "
                f"conv={stats.get('hierarchy_lag_recent_conv')} "
                f"(limit {_lag_limit}). Turns past the watermark are carried "
                f"by the raw window alone, so the oldest of them fall out of "
                f"the request as it grows. "
                f"POST /admin/conversations/<id>/compact drains the "
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
            # v3.1.9 (webuidb MEDIUM, "measures her IDLE TIME"). When
            # local_lag_s decided this (the normal case - see probe_snapshot),
            # say THAT number: it is "how far behind the live database this
            # copy is", which is the actual durability gap. age_s alone ("how
            # long since anything changed") is what cried wolf every quiet
            # night, so it is named only as extra context here, not as the
            # headline number.
            _lag = _snap.get("local_lag_s")
            if _lag is not None:
                reasons.append(
                    f"the webui.db snapshot on /data is {_lag}s behind the "
                    f"live database on local disk (unchanged for {_snap.get('age_s')}s, "
                    f"sync interval {_snap.get('interval_s')}s). The live "
                    f"database is on local disk, which a pod recreate "
                    f"destroys, so this is the durable copy and it is not "
                    f"catching up. Check `supervisorctl status webuidb-sync`."
                )
            else:
                reasons.append(
                    f"the webui.db snapshot on /data is {_snap.get('age_s')}s "
                    f"old against a {_snap.get('interval_s')}s sync interval "
                    f"(WEBUI_LOCAL_DB could not be read, so this is judged by "
                    f"age alone and may simply mean nobody has chatted "
                    f"recently). The live database is on local disk, which a "
                    f"pod recreate destroys, so this is supposed to be its "
                    f"durable copy. Check `supervisorctl status webuidb-sync`."
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
