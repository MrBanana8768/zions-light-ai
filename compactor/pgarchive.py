"""
compactor.pgarchive — durability for the Postgres state home.

THE SHAPE (ARCHITECTURE.md Decision 4, forced by the 2026-08-31 webui.db
incident and applied here *before* Postgres ever has a chance to repeat it):

    Postgres  ->  PGDATA on LOCAL disk (the pod overlay), unix socket only
                          |
                          |  archive_once(), every PGARCHIVE_INTERVAL_S
                          v
                   /data/openwebui/pg/*.sql.gz   (MooseFS, durable)

Postgres itself never writes to `/data` — its live files, its WAL, its
sockets all live on local disk, where writes do not fail. `/data` holds
periodic `pg_dump` archives, gzipped and timestamped, which is what survives
the pod being recreated or local disk being lost. Restoring one of those
archives into a freshly-initialised, empty database is `restore_if_needed()`,
called from entrypoint.sh before any service that depends on Postgres starts.

THE TRADE, stated plainly, same as webuidb.py's: local disk is not
persistent across pod recreation, so PGARCHIVE_INTERVAL_S is the RPO for
whatever lives in Postgres. That is a property of the archive cadence, not
a flaw to be engineered away here — a shorter interval buys a smaller RPO
at the cost of a `pg_dump` more often.

THIS MODULE DOES NOT MIGRATE DATA. It is infrastructure: it dumps whatever
is in the configured Postgres database and restores whatever the newest
good archive holds. The SQLite -> Postgres data migration is separate,
deliberately out of scope here.

CONNECTIONS ARE UNIX-SOCKET ONLY. Every psql/pg_dump/pg_isready call below
passes `-h <socket dir>`, never a hostname, so there is no TCP path for any
of this to depend on and nothing to expose.

THREE GUARDS, all earned the same way webuidb.py's were — by a failure mode
that is real once you think through the composition, not a hypothetical:

  1. **Verify the archive file BEFORE it can be selected as "the newest
     good one."** A `pg_dump` that got truncated by a killed container, or
     landed on a stalling `/data` mid-write, must never be gzip-corrupt AND
     sitting there as the file `restore_if_needed()` would reach for next
     boot. Verified via a full gzip integrity read, and via re-deriving the
     exact row count straight from the dump's own `COPY ... FROM stdin`
     blocks — not trusted metadata, the dump content itself.
  2. **Refuse to archive an empty/near-empty database over a healthy
     previous archive.** The exact composition that destroyed 400 chats in
     testing for webui.db: some step degrades silently (Postgres restarts
     into a schema-only state, a connection races the app's own migration,
     a role/permissions problem hides most tables from this dump's user),
     the dump "succeeds" because pg_dump exits 0 on an almost-empty result,
     and publishing it overwrites the only durable copy with next to
     nothing. SHRINK_REFUSE_BELOW below is that guard, same shape and same
     override (PGARCHIVE_ALLOW_SHRINK=1) as webuidb.py's.
  3. **Watch the disk PGDATA itself lives on, not just /data.** The
     compactor's free-space guard watches /data (backup.py); nothing
     watched the 20 GB ephemeral overlay PGDATA is actually on, and a full
     PGDATA is a Postgres PANIC, not a degraded database — see the sizing
     fact above. `_pgdata_disk_status()` reports free space and PGDATA's
     footprint every archive cycle and escalates from a warning to a loud
     error as it gets worse (DISK_WARN_FREE_MB / DISK_CRITICAL_FREE_MB
     below); it only ever observes and logs — see its docstring.

Run inside the compactor image or any container with postgresql-client-16
and the compactor venv installed:
    python pgarchive.py --archive-once
    python pgarchive.py --restore-if-needed
    python pgarchive.py --status
    python pgarchive.py                    # archive_loop() daemon (default)
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

logger = logging.getLogger("compactor.pgarchive")

# ---------------------------------------------------------------------------
# Configuration — mirrors entrypoint.sh's exports so both agree on where
# Postgres actually lives without either hardcoding the other's defaults.
# ---------------------------------------------------------------------------

PGDATA = Path(os.environ.get("PGDATA", "/var/lib/postgresql/data"))
POSTGRES_SOCKET_DIR = os.environ.get("POSTGRES_SOCKET_DIR", "/var/run/postgresql")
PGPORT = os.environ.get("PGPORT", "5432")
PGUSER = os.environ.get("POSTGRES_USER", "openwebui")
PGDATABASE = os.environ.get("POSTGRES_DB", "openwebui")

# Where archives land. A sibling layout to /data/backups (compactor/backup.py)
# but its own directory — this module owns its own retention and naming and
# must not be confused with the JSON/webui.db archive set backup.py manages.
ARCHIVE_DIR = Path(os.environ.get("PGARCHIVE_DIR", "/data/openwebui/pg"))

ARCHIVE_INTERVAL_S = float(os.environ.get("PGARCHIVE_INTERVAL_S", "300") or 300)

# Retention: keep the newest RETAIN archives, never fewer than MIN_KEEP
# regardless of RETAIN's value — the same "floor in code, not just config"
# rule backup.py uses, so a typo'd env var can't empty the archive directory.
RETAIN = int(os.environ.get("PGARCHIVE_RETAIN", "10") or 10)
MIN_KEEP = max(3, int(os.environ.get("PGARCHIVE_MIN_KEEP", "3") or 3))

# REGRESSION GUARD — see the module docstring. Refuse to publish an archive
# holding less than this fraction of the previous archive's row count.
SHRINK_REFUSE_BELOW = float(
    os.environ.get("PGARCHIVE_SHRINK_REFUSE_BELOW", "0.5") or 0.5
)
# Below this many rows in the PREVIOUS archive the ratio is meaningless
# (2 -> 1 is a 50% drop and means nothing), so the guard stands down —
# same reasoning as webuidb.SHRINK_GUARD_MIN_CHATS.
SHRINK_GUARD_MIN_ROWS = int(
    os.environ.get("PGARCHIVE_SHRINK_GUARD_MIN_ROWS", "50") or 50
)
# The deliberate override for a real, intended shrink (bulk delete, a
# deliberate `/forget`-equivalent). Refusing forever would be its own
# failure mode.
ALLOW_SHRINK = (
    os.environ.get("PGARCHIVE_ALLOW_SHRINK", "").strip().lower()
    in ("1", "true", "yes")
)

# PGDATA DISK-HEADROOM GUARD — see module docstring's sizing fact: the
# overlay PGDATA lives on is 20 GB total, EPHEMERAL, shared with
# /var/lib/openwebui and every container layer. Nothing else in this image
# watches that filesystem (the compactor's own free-space guard watches
# /data, a completely different mount). A full PGDATA is not a degraded
# database, it is a Postgres PANIC and a wedged pod, so this warns loudly
# well before that.
#
# WARN: 4096 MB (~20% of the 20 GB overlay). At the last measured baseline
# (388 MB used, 2026-08-31) that is still a wide margin, but it is the point
# where "plenty of runway" stops being true and someone should look within
# the week rather than after the next incident.
DISK_WARN_FREE_MB = int(os.environ.get("PGARCHIVE_DISK_WARN_FREE_MB", "4096") or 4096)
# CRITICAL: 1024 MB. Below this, zionslight.conf's own settings (entrypoint.sh)
# no longer have room to fail safely — max_wal_size=512MB plus a single
# query hitting temp_file_limit=1GB is 1.5 GB by itself, before
# shared_buffers or anything else sharing the overlay. Under 1 GB free, the
# next checkpoint or one large query can be the PANIC.
DISK_CRITICAL_FREE_MB = int(os.environ.get("PGARCHIVE_DISK_CRITICAL_FREE_MB", "1024") or 1024)

_PREFIX = "pg-"
_SUFFIX = ".sql.gz"


def _reload_env() -> None:
    """Re-read the env-driven knobs. For tests, and for anyone who changes
    them without restarting the process."""
    global PGDATA, POSTGRES_SOCKET_DIR, PGPORT, PGUSER, PGDATABASE
    global ARCHIVE_DIR, ARCHIVE_INTERVAL_S, RETAIN, MIN_KEEP
    global SHRINK_REFUSE_BELOW, SHRINK_GUARD_MIN_ROWS, ALLOW_SHRINK
    PGDATA = Path(os.environ.get("PGDATA", "/var/lib/postgresql/data"))
    POSTGRES_SOCKET_DIR = os.environ.get("POSTGRES_SOCKET_DIR", "/var/run/postgresql")
    PGPORT = os.environ.get("PGPORT", "5432")
    PGUSER = os.environ.get("POSTGRES_USER", "openwebui")
    PGDATABASE = os.environ.get("POSTGRES_DB", "openwebui")
    ARCHIVE_DIR = Path(os.environ.get("PGARCHIVE_DIR", "/data/openwebui/pg"))
    ARCHIVE_INTERVAL_S = float(os.environ.get("PGARCHIVE_INTERVAL_S", "300") or 300)
    RETAIN = int(os.environ.get("PGARCHIVE_RETAIN", "10") or 10)
    MIN_KEEP = max(3, int(os.environ.get("PGARCHIVE_MIN_KEEP", "3") or 3))
    SHRINK_REFUSE_BELOW = float(
        os.environ.get("PGARCHIVE_SHRINK_REFUSE_BELOW", "0.5") or 0.5
    )
    SHRINK_GUARD_MIN_ROWS = int(
        os.environ.get("PGARCHIVE_SHRINK_GUARD_MIN_ROWS", "50") or 50
    )
    ALLOW_SHRINK = (
        os.environ.get("PGARCHIVE_ALLOW_SHRINK", "").strip().lower()
        in ("1", "true", "yes")
    )
    global DISK_WARN_FREE_MB, DISK_CRITICAL_FREE_MB
    DISK_WARN_FREE_MB = int(os.environ.get("PGARCHIVE_DISK_WARN_FREE_MB", "4096") or 4096)
    DISK_CRITICAL_FREE_MB = int(
        os.environ.get("PGARCHIVE_DISK_CRITICAL_FREE_MB", "1024") or 1024
    )


def _stamp() -> str:
    # Millisecond precision — see webuidb._stamp for why: two archives inside
    # the same second must not collide on the filename.
    return time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"


# ---------------------------------------------------------------------------
# Small process/query helpers. Shelling out to the real psql/pg_dump client
# binaries (postgresql-client-16) rather than adding a Python driver
# dependency to the compactor venv — this module's whole job is talking to
# Postgres over a unix socket, which the client tools already do correctly.
# ---------------------------------------------------------------------------


def _pg_reachable() -> bool:
    try:
        r = subprocess.run(
            [
                "pg_isready",
                "-h", POSTGRES_SOCKET_DIR,
                "-p", str(PGPORT),
                "-d", PGDATABASE,
                "-U", PGUSER,
                "-q",
            ],
            capture_output=True,
            timeout=10,
        )
        return r.returncode == 0
    except Exception:
        return False


def _pgdata_initialized() -> bool:
    return (PGDATA / "PG_VERSION").exists()


def _dir_size_bytes(path: Path) -> int | None:
    """Sum of regular file sizes under path. None if path itself can't be
    walked at all (e.g. PGDATA not yet initdb'd). A file that disappears or
    errors mid-walk (normal under a live Postgres) is just skipped — this is
    a headroom estimate, not an accounting record."""
    if not path.exists():
        return None
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                continue
    return total


def _pgdata_disk_status() -> dict:
    """Free space on the filesystem PGDATA actually lives on, plus PGDATA's
    own footprint. Deliberately independent of `/data`, of ARCHIVE_DIR, and
    of Postgres reachability — PGDATA and /data are different filesystems
    (that is the entire point of THE SHAPE above), so a stalled or
    unreadable /data must never make this check fail, and this check must
    never touch Postgres. It only reads filesystem metadata.

    Never raises — every failure mode collapses to level="unknown", which
    the caller treats as "couldn't determine", the same discipline
    _psql_scalar uses for a failed query.
    """
    status: dict = {
        "free_mb": None,
        "total_mb": None,
        "pgdata_mb": None,
        "level": "unknown",
    }

    # PGDATA may not exist yet (pre-initdb) — walk up to the nearest
    # existing ancestor so the check still reports the overlay's real
    # free space instead of failing outright.
    probe = PGDATA
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent

    try:
        usage = shutil.disk_usage(probe)
        status["free_mb"] = usage.free / 1e6
        status["total_mb"] = usage.total / 1e6
    except Exception as e:
        logger.debug(f"pgdata disk guard: disk_usage({probe}) failed: {type(e).__name__}: {e}")
        return status

    size = _dir_size_bytes(PGDATA)
    if size is not None:
        status["pgdata_mb"] = size / 1e6

    free_mb = status["free_mb"]
    if free_mb < DISK_CRITICAL_FREE_MB:
        status["level"] = "critical"
    elif free_mb < DISK_WARN_FREE_MB:
        status["level"] = "warn"
    else:
        status["level"] = "ok"

    pgdata_mb = status["pgdata_mb"]
    pgdata_str = f"{pgdata_mb:.0f} MB" if pgdata_mb is not None else "unknown"
    if status["level"] == "critical":
        logger.error(
            f"PGDATA DISK CRITICAL: {free_mb:.0f} MB free on the filesystem "
            f"holding {PGDATA} (PGDATA itself is {pgdata_str}, "
            f"PGARCHIVE_DISK_CRITICAL_FREE_MB={DISK_CRITICAL_FREE_MB}). This is "
            f"close enough to full that the next checkpoint or one large query "
            f"can PANIC Postgres and wedge the pod — see module docstring's "
            f"sizing fact. Free space on the overlay now."
        )
    elif status["level"] == "warn":
        logger.warning(
            f"PGDATA disk low: {free_mb:.0f} MB free on the filesystem "
            f"holding {PGDATA} (PGDATA itself is {pgdata_str}, "
            f"PGARCHIVE_DISK_WARN_FREE_MB={DISK_WARN_FREE_MB}). Still safe, but "
            f"plan to free space before it reaches "
            f"PGARCHIVE_DISK_CRITICAL_FREE_MB={DISK_CRITICAL_FREE_MB} MB."
        )

    return status


def _psql_scalar(sql: str, timeout: float = 20) -> str | None:
    """Run a single scalar query. None on ANY failure — connection refused,
    timeout, bad SQL — the caller treats that as "couldn't determine", never
    as "zero"."""
    cmd = [
        "psql",
        "-h", POSTGRES_SOCKET_DIR,
        "-p", str(PGPORT),
        "-U", PGUSER,
        "-d", PGDATABASE,
        "-tAc", sql,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            logger.debug(f"psql scalar failed: {r.stderr.strip()[:300]}")
            return None
        return r.stdout.strip()
    except Exception as e:
        logger.debug(f"psql scalar error: {type(e).__name__}: {e}")
        return None


def _table_count() -> int | None:
    """How many user tables exist. None if unreachable/unqueryable — a
    schema-agnostic analog of webuidb._has_rows: the cheapest honest proxy
    for "does this database hold anything at all"."""
    out = _psql_scalar(
        "SELECT count(*) FROM information_schema.tables "
        "WHERE table_schema NOT IN ('pg_catalog', 'information_schema')"
    )
    if out is None:
        return None
    try:
        return int(out)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Archive file helpers
# ---------------------------------------------------------------------------


def _list_archives() -> list[Path]:
    """Newest last. Scoped to the CURRENT PGDATABASE name — if that env var
    ever changes, archives from a previous database name are left alone
    rather than silently candidates for restore into a differently-named
    database."""
    if not ARCHIVE_DIR.exists():
        return []
    return sorted(ARCHIVE_DIR.glob(f"{_PREFIX}{PGDATABASE}-*{_SUFFIX}"))


def _meta_path(archive: Path) -> Path:
    return Path(str(archive) + ".meta.json")


def _write_meta(archive: Path, rows: int, size_bytes: int) -> None:
    meta = {
        "rows": rows,
        "bytes": size_bytes,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "database": PGDATABASE,
    }
    mp = _meta_path(archive)
    tmp = mp.with_name(mp.name + f".tmp-{os.getpid()}")
    try:
        tmp.write_text(json.dumps(meta), encoding="utf-8")
        os.replace(tmp, mp)
    except Exception as e:
        logger.warning(f"could not write meta for {archive.name}: {type(e).__name__}: {e}")
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def _read_meta(archive: Path) -> dict | None:
    mp = _meta_path(archive)
    if not mp.exists():
        return None
    try:
        return json.loads(mp.read_text(encoding="utf-8"))
    except Exception:
        return None


def _gzip_integrity(path: Path) -> bool:
    """Read the WHOLE stream, not just the header — a truncated gzip (the
    exact shape of a container killed mid-dump) can open cleanly and still
    fail partway through."""
    try:
        with gzip.open(path, "rb") as fh:
            while fh.read(1024 * 1024):
                pass
        return True
    except Exception as e:
        logger.debug(f"gzip integrity check failed for {path}: {type(e).__name__}: {e}")
        return False


def _count_rows_in_dump(path: Path) -> int:
    """Exact row count, derived from the dump's own `COPY ... FROM stdin`
    data blocks — not trusted metadata, the archive content itself. A
    schema-only dump (no data at all) correctly counts as 0."""
    rows = 0
    in_copy = False
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if in_copy:
                if line.rstrip("\n") == "\\.":
                    in_copy = False
                else:
                    rows += 1
            elif line.startswith("COPY ") and line.rstrip("\n").endswith("FROM stdin;"):
                in_copy = True
    return rows


def _quarantine(path: Path, why: str) -> None:
    """Set a bad/refused archive aside — never delete. Same rule as
    webuidb._set_aside: anything removing state must be reversible."""
    try:
        qdir = ARCHIVE_DIR / "quarantine"
        qdir.mkdir(parents=True, exist_ok=True)
        dest = qdir / f"{path.name}.{why}-{_stamp()}"
        shutil.move(str(path), str(dest))
        logger.warning(f"quarantined {path.name} ({why}) -> {dest}")
    except Exception as e:
        logger.error(f"could not quarantine {path}: {type(e).__name__}: {e}")


def _prune() -> None:
    """Delete the oldest archives beyond RETAIN, never below MIN_KEEP.
    Deleting a superseded, already-verified-good archive is not the
    dangerous case — publishing a bad one over the last good one is, and
    that is guarded in archive_once(), not here."""
    archives = _list_archives()
    keep = max(RETAIN, MIN_KEEP)
    if len(archives) <= keep:
        return
    for a in archives[: len(archives) - keep]:
        try:
            a.unlink(missing_ok=True)
            _meta_path(a).unlink(missing_ok=True)
            logger.info(f"pruned {a.name} (retention)")
        except Exception as e:
            logger.warning(f"could not prune {a.name}: {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def archive_once() -> dict:
    """Dump PGDATABASE to a gzipped, timestamped archive on /data, safely.

    Same three-part discipline as webuidb.sync_once:
      * Written to a temp name and only os.replace'd into place after it
        verifies, so a crash mid-write can never leave a partial file where
        restore_if_needed() would find it.
      * Verified (gzip integrity + a real row count derived from the dump
        content) BEFORE it can be treated as "the newest good archive".
      * The shrink guard — see module docstring — refuses to let a
        near-empty dump become the newest archive over a healthy one.
    """
    out = {"archived": False, "skipped": None, "error": None, "bytes": 0, "rows": 0, "path": None}

    try:
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        out["error"] = f"cannot create {ARCHIVE_DIR}: {type(e).__name__}: {e}"
        logger.warning(
            f"{out['error']}. Postgres itself is unaffected; only the durable "
            f"copy on /data is at risk while this persists."
        )
        return out

    if not _pg_reachable():
        out["skipped"] = "postgres not reachable over the unix socket"
        return out

    final = ARCHIVE_DIR / f"{_PREFIX}{PGDATABASE}-{_stamp()}{_SUFFIX}"
    tmp = ARCHIVE_DIR / f"{final.name}.partial-{os.getpid()}"

    cmd = [
        "pg_dump",
        "-h", POSTGRES_SOCKET_DIR,
        "-p", str(PGPORT),
        "-U", PGUSER,
        "-d", PGDATABASE,
        "--no-owner",
        "--no-privileges",
        "--format=plain",
    ]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert proc.stdout is not None
        with gzip.open(tmp, "wb") as gz:
            shutil.copyfileobj(proc.stdout, gz)
        stderr = proc.stderr.read() if proc.stderr else b""
        code = proc.wait(timeout=120)
        if code != 0:
            raise RuntimeError(f"pg_dump exited {code}: {stderr.decode('utf-8', 'replace')[:500]}")
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        logger.warning(
            f"pg_dump failed ({out['error']}). Postgres is unaffected; the "
            f"previous good archive on /data is untouched; retrying next cycle."
        )
        return out

    if not _gzip_integrity(tmp):
        out["error"] = "produced archive failed gzip integrity check"
        _quarantine(tmp, "bad-gzip")
        logger.warning(out["error"])
        return out

    rows = _count_rows_in_dump(tmp)
    size_bytes = tmp.stat().st_size

    previous = _list_archives()
    prev_meta = _read_meta(previous[-1]) if previous else None
    prev_rows = prev_meta.get("rows") if prev_meta else None

    if (
        isinstance(prev_rows, int)
        and prev_rows >= SHRINK_GUARD_MIN_ROWS
        and rows < prev_rows * SHRINK_REFUSE_BELOW
        and not ALLOW_SHRINK
    ):
        _quarantine(tmp, f"refused-shrink-{rows}-of-{prev_rows}")
        out["error"] = (
            f"REFUSING to publish: this dump has {rows} row(s) across all "
            f"tables but the previous archive ({previous[-1].name}) has "
            f"{prev_rows}. That is not ordinary use — it is what a broken "
            f"connection, a permissions problem hiding tables from this dump's "
            f"role, or a database mid-migration looks like, and publishing it "
            f"would make an almost-empty dump the newest 'good' archive that "
            f"restore_if_needed() would restore from next. The dump was kept "
            f"at {ARCHIVE_DIR / 'quarantine'} for inspection. If this shrink is "
            f"real and intended, set PGARCHIVE_ALLOW_SHRINK=1."
        )
        logger.warning(out["error"])
        return out

    os.replace(tmp, final)
    _write_meta(final, rows=rows, size_bytes=size_bytes)
    out["archived"] = True
    out["bytes"] = size_bytes
    out["rows"] = rows
    out["path"] = str(final)
    logger.info(
        f"archived {PGDATABASE} -> {final.name} ({size_bytes / 1e6:.2f} MB, {rows} rows)"
    )

    _prune()
    return out


def restore_if_needed() -> dict:
    """On boot: if the database is empty (a fresh initdb, or local disk lost
    and recreated), restore the newest archive that verifies. Never restores
    over a database that already holds tables — this is a fresh-start path,
    not a sync, and a live database always wins, same rule as
    webuidb.restore_on_boot's "a healthy local database wins outright".

    Called from entrypoint.sh AFTER Postgres is up (temporarily, for setup)
    and the role/database exist, and BEFORE any dependent service starts.
    """
    result: dict = {"action": None, "pgdata_initialized": _pgdata_initialized()}

    if not _pg_reachable():
        result["action"] = "error"
        result["detail"] = "postgres not reachable over the unix socket"
        logger.error(f"restore_if_needed: {result['detail']}")
        return result

    n_tables = _table_count()
    if n_tables is None:
        result["action"] = "error"
        result["detail"] = "could not query information_schema.tables"
        logger.error(f"restore_if_needed: {result['detail']}")
        return result

    if n_tables > 0:
        result["action"] = "kept_existing"
        result["tables"] = n_tables
        logger.info(
            f"database already has {n_tables} table(s) — not restoring over it "
            f"(a live database always wins over an archive)"
        )
        return result

    archives = _list_archives()
    if not archives:
        result["action"] = "fresh"
        logger.info(
            "no archive on /data and the database is empty — a new deployment; "
            "Postgres/OpenWebUI will create their own schema and the first "
            "archive cycle will publish it"
        )
        return result

    # Newest first. A candidate that fails integrity or fails to restore is
    # skipped (never deleted — see _quarantine's discipline, though a
    # skipped-but-not-corrupt candidate here is just left where it is) and we
    # fall back to an older one, same shape as webuidb trying local-then-
    # snapshot rather than giving up on the first bad option.
    for candidate in reversed(archives):
        if not _gzip_integrity(candidate):
            logger.error(
                f"archive {candidate.name} failed gzip integrity check — "
                f"trying an older archive instead of restoring a broken one"
            )
            continue

        meta = _read_meta(candidate)
        logger.info(
            f"restoring {candidate.name} "
            f"({meta.get('rows') if meta else '?'} rows) into the empty database"
        )
        try:
            with gzip.open(candidate, "rb") as gz:
                proc = subprocess.Popen(
                    [
                        "psql",
                        "-h", POSTGRES_SOCKET_DIR,
                        "-p", str(PGPORT),
                        "-U", PGUSER,
                        "-d", PGDATABASE,
                        "-v", "ON_ERROR_STOP=1",
                        # Single transaction: any error rolls back the WHOLE
                        # restore, leaving the database exactly as empty as it
                        # started. Without this, a restore that fails halfway
                        # leaves a partially-populated database that the NEXT
                        # boot's table_count()>0 check would treat as "already
                        # has data" and never retry — the exact partial-state
                        # trap this guards against.
                        "--single-transaction",
                    ],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                assert proc.stdin is not None
                shutil.copyfileobj(gz, proc.stdin)
                proc.stdin.close()
                stderr = proc.stderr.read() if proc.stderr else b""
                code = proc.wait(timeout=300)
            if code != 0:
                raise RuntimeError(f"psql restore exited {code}: {stderr.decode('utf-8', 'replace')[:500]}")
        except Exception as e:
            logger.error(
                f"restore from {candidate.name} failed ({type(e).__name__}: {e}) — "
                f"the transaction rolled back, database still empty; trying an "
                f"older archive"
            )
            continue

        after = _table_count()
        result["action"] = "restored"
        result["path"] = str(candidate)
        result["tables"] = after
        result["rows"] = meta.get("rows") if meta else None
        logger.info(f"restored from {candidate.name} — database now has {after} table(s)")
        return result

    result["action"] = "restore_failed"
    result["detail"] = "every archive on /data failed integrity or failed to restore"
    logger.error(
        f"restore_if_needed: {result['detail']}. The database stays empty; "
        f"Postgres/OpenWebUI will create their own schema. Recover archives "
        f"from {ARCHIVE_DIR} manually if possible."
    )
    return result


def archive_loop() -> None:
    """Daemon entry point (supervisord program `pgarchive`)."""
    logger.info(
        f"pgarchive: {PGDATABASE} -> {ARCHIVE_DIR} every {ARCHIVE_INTERVAL_S:.0f}s"
    )
    consecutive_failures = 0
    while True:
        time.sleep(ARCHIVE_INTERVAL_S)
        try:
            # Observe-and-report only, on the same timer as archiving — see
            # _pgdata_disk_status's docstring. Belt-and-braces try/except
            # around an already-defensive function: this guard must NEVER
            # be the reason an archive cycle stops running.
            _pgdata_disk_status()
        except Exception as e:
            logger.debug(f"pgdata disk guard raised (ignored): {type(e).__name__}: {e}")
        r = archive_once()
        if r["error"]:
            consecutive_failures += 1
            if consecutive_failures in (3, 12, 48):
                logger.error(
                    f"pg_dump archive has failed {consecutive_failures} times in a "
                    f"row ({consecutive_failures * ARCHIVE_INTERVAL_S / 60:.0f} "
                    f"minutes without a durable copy of the database). Postgres "
                    f"itself is unaffected — only the durability window widens."
                )
        elif r["archived"]:
            consecutive_failures = 0


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    if "--restore-if-needed" in sys.argv:
        print(restore_if_needed())
    elif "--archive-once" in sys.argv:
        print(archive_once())
    elif "--status" in sys.argv:
        archives = _list_archives()
        print(
            f"pgdata_initialized={_pgdata_initialized()} "
            f"reachable={_pg_reachable()} tables={_table_count()}"
        )
        disk = _pgdata_disk_status()
        if disk["free_mb"] is not None:
            pgdata_str = f"{disk['pgdata_mb']:.0f}MB" if disk["pgdata_mb"] is not None else "unknown"
            print(
                f"pgdata_disk: free={disk['free_mb']:.0f}MB "
                f"total={disk['total_mb']:.0f}MB pgdata_size={pgdata_str} "
                f"level={disk['level']} "
                f"(warn<{DISK_WARN_FREE_MB}MB critical<{DISK_CRITICAL_FREE_MB}MB)"
            )
        else:
            print("pgdata_disk: unknown (could not stat the filesystem PGDATA lives on)")
        print(f"archive_dir={ARCHIVE_DIR} archives={len(archives)}")
        for a in archives[-5:]:
            meta = _read_meta(a)
            size_mb = a.stat().st_size / 1e6
            rows = meta.get("rows") if meta else "?"
            print(f"  {a.name}  {size_mb:.2f}MB  rows={rows}")
    else:
        archive_loop()
