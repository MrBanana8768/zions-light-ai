"""
compactor/pgarchive.py — Postgres archive/restore on the /data volume.

Companion to test_webuidb.py, same shape: two separate directories stand in
for the two volumes (PGARCHIVE_DIR for /data, PGDATA for local disk), so the
volume separation this design depends on is real in the test too, not
assumed.

pg_dump/psql are NOT invoked for real here — this machine (and the baseline
image this suite is also run against, per the release checklist) may not
have PostgreSQL installed at all, and pgarchive.py's whole job is talking to
a database this test has no business standing up just to check a guard
condition. Instead, subprocess.Popen is replaced with an in-memory fake that
speaks the exact same protocol (stdout for pg_dump's plain-SQL output, stdin
for psql's restore input) that archive_once()/restore_if_needed() actually
use, so every byte of the real code path — gzip write, atomic rename, gzip
verify, row counting from the dump's own COPY blocks, the shrink-refusal
guard, retention pruning, single-transaction restore, corrupt-archive
fallback — runs for real. Only "is postgres up" (_pg_reachable) and "how
many tables does it have" (_table_count) are stubbed, because those really
do require a live server and are exactly the two functions this module
keeps deliberately small and easy to trust by inspection.

Every table/row value below is synthetic ("synthetic-lorem-ipsum-row") —
never real conversation content (repo is public).

    python test_pgarchive.py
"""

import gzip
import io
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Two volumes. LOCAL stands in for the pod's overlay (PGDATA, barely used by
# this module beyond a PG_VERSION existence check), ARCHIVE for MooseFS.
_LOCAL_VOL = Path(tempfile.mkdtemp(prefix="vol-local-"))
_ARCHIVE_VOL = Path(tempfile.mkdtemp(prefix="vol-moosefs-"))

os.environ["PGDATA"] = str(_LOCAL_VOL / "pgdata")
os.environ["PGARCHIVE_DIR"] = str(_ARCHIVE_VOL / "openwebui" / "pg")
os.environ["POSTGRES_USER"] = "openwebui"
os.environ["POSTGRES_DB"] = "openwebui"
os.environ["POSTGRES_SOCKET_DIR"] = str(_LOCAL_VOL / "run-postgresql")
os.environ["PGARCHIVE_SHRINK_REFUSE_BELOW"] = "0.5"
os.environ["PGARCHIVE_SHRINK_GUARD_MIN_ROWS"] = "10"
os.environ["PGARCHIVE_RETAIN"] = "3"
os.environ["PGARCHIVE_MIN_KEEP"] = "2"
os.environ["PGARCHIVE_ALLOW_SHRINK"] = ""

import pgarchive  # noqa: E402

FAILED = []
ARCHIVE_DIR = pgarchive.ARCHIVE_DIR


def check(cond, label):
    if cond:
        print(f"  ok   {label}")
    else:
        FAILED.append(label)
        print(f"  FAIL {label}")


print("two volumes:")
print(f"  local (overlay)  {_LOCAL_VOL}")
print(f"  archive (mfs)    {_ARCHIVE_VOL}")


# ---------------------------------------------------------------------------
# Fake subprocess.Popen — stands in for pg_dump (produces stdout) and psql
# (consumes stdin), entirely in-memory. Real subprocess.Popen is restored at
# the end of the file, in a finally block.
# ---------------------------------------------------------------------------

_real_popen = subprocess.Popen

_dump_plan = {"text": "", "returncode": 0, "raise_exc": None}
_restore_plan = {"returncode": 0, "raise_exc": None, "captured": None}


class _FakeStdin(io.BytesIO):
    def __init__(self, sink_holder):
        super().__init__()
        self._sink_holder = sink_holder

    def close(self):
        self._sink_holder["bytes"] = self.getvalue()
        super().close()


class _FakeProc:
    def __init__(self, stdout_bytes=None, stdin_sink=None, returncode=0, stderr_bytes=b""):
        self.stdout = io.BytesIO(stdout_bytes) if stdout_bytes is not None else None
        self.stdin = _FakeStdin(stdin_sink) if stdin_sink is not None else None
        self.stderr = io.BytesIO(stderr_bytes)
        self._returncode = returncode

    def wait(self, timeout=None):
        return self._returncode


def _fake_popen(cmd, **kw):
    prog = cmd[0]
    if prog == "pg_dump":
        if _dump_plan["raise_exc"] is not None:
            raise _dump_plan["raise_exc"]
        rc = _dump_plan["returncode"]
        return _FakeProc(
            stdout_bytes=_dump_plan["text"].encode("utf-8"),
            returncode=rc,
            stderr_bytes=b"" if rc == 0 else b"simulated pg_dump failure",
        )
    if prog == "psql":
        if _restore_plan["raise_exc"] is not None:
            raise _restore_plan["raise_exc"]
        rc = _restore_plan["returncode"]
        return _FakeProc(
            stdin_sink=_restore_plan["captured"],
            returncode=rc,
            stderr_bytes=b"" if rc == 0 else b"simulated psql failure",
        )
    raise AssertionError(f"unexpected command in fake Popen: {cmd!r}")


subprocess.Popen = _fake_popen


def _dump_text(table_rows: dict) -> str:
    """A synthetic plain-format pg_dump body with real COPY...FROM stdin
    blocks, so _count_rows_in_dump exercises its actual parser."""
    lines = ["-- PostgreSQL database dump", "", "SET statement_timeout = 0;", ""]
    for table, n in table_rows.items():
        lines.append(f"CREATE TABLE public.{table} (id integer, data text);")
        lines.append("")
        lines.append(f"COPY public.{table} (id, data) FROM stdin;")
        for r in range(n):
            lines.append(f"{r}\tsynthetic-lorem-ipsum-row")
        lines.append("\\.")
        lines.append("")
    lines.append("-- PostgreSQL database dump complete")
    return "\n".join(lines) + "\n"


def _set_dump(table_rows: dict, returncode: int = 0, raise_exc=None):
    _dump_plan["text"] = _dump_text(table_rows) if table_rows is not None else ""
    _dump_plan["returncode"] = returncode
    _dump_plan["raise_exc"] = raise_exc


def _set_restore(returncode: int = 0, raise_exc=None):
    _restore_plan["returncode"] = returncode
    _restore_plan["raise_exc"] = raise_exc
    _restore_plan["captured"] = {}


def _wipe_archive_dir():
    import shutil

    if ARCHIVE_DIR.exists():
        shutil.rmtree(ARCHIVE_DIR)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)


def _make_archive(rows: int, tables: int = 1, age_offset_s: float = 0.0) -> Path:
    """Write a real, valid archive + meta pair directly (bypassing
    archive_once), for tests that need a pre-existing 'previous good
    archive' to check against."""
    stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
    path = ARCHIVE_DIR / f"pg-{pgarchive.PGDATABASE}-{stamp}.sql.gz"
    table_rows = {f"t{i}": rows // max(tables, 1) for i in range(tables)}
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(_dump_text(table_rows))
    actual_rows = pgarchive._count_rows_in_dump(path)
    pgarchive._write_meta(path, rows=actual_rows, size_bytes=path.stat().st_size)
    time.sleep(0.01)  # keep millisecond stamps ordered across calls in a tight loop
    return path


try:
    # -------------------------------------------------------------------
    print()
    print("[1] gzip integrity: a valid archive passes, a truncated one fails")
    _wipe_archive_dir()
    good = ARCHIVE_DIR / "good.sql.gz"
    with gzip.open(good, "wt", encoding="utf-8") as fh:
        fh.write(_dump_text({"chat": 5}))
    check(pgarchive._gzip_integrity(good), "a well-formed gzip archive passes integrity")

    bad = ARCHIVE_DIR / "bad.sql.gz"
    with open(good, "rb") as src, open(bad, "wb") as dst:
        dst.write(src.read()[: -10])  # truncate — the exact shape of a killed-mid-write dump
    check(not pgarchive._gzip_integrity(bad), "a truncated gzip archive fails integrity")

    # -------------------------------------------------------------------
    print()
    print("[2] row counting reads the dump's own COPY blocks, exactly")
    _wipe_archive_dir()
    multi = ARCHIVE_DIR / "multi.sql.gz"
    with gzip.open(multi, "wt", encoding="utf-8") as fh:
        fh.write(_dump_text({"chat": 7, "message": 13}))
    check(pgarchive._count_rows_in_dump(multi) == 20, "counted 7+13=20 rows across two tables")

    empty = ARCHIVE_DIR / "empty.sql.gz"
    with gzip.open(empty, "wt", encoding="utf-8") as fh:
        fh.write(_dump_text({}))  # schema-only-less dump, no COPY blocks at all
    check(pgarchive._count_rows_in_dump(empty) == 0, "a dump with no COPY blocks counts as 0 rows")

    # -------------------------------------------------------------------
    print()
    print("[3] meta read/write round-trips")
    _wipe_archive_dir()
    a = ARCHIVE_DIR / "meta-test.sql.gz"
    a.write_bytes(b"")
    pgarchive._write_meta(a, rows=42, size_bytes=1234)
    meta = pgarchive._read_meta(a)
    check(meta is not None and meta["rows"] == 42, f"meta round-trips (got {meta})")
    check(pgarchive._read_meta(ARCHIVE_DIR / "nonexistent.sql.gz") is None, "missing meta reads as None, not an error")

    # -------------------------------------------------------------------
    print()
    print("[4] archive listing is scoped to the CURRENT database name")
    _wipe_archive_dir()
    (ARCHIVE_DIR / f"pg-{pgarchive.PGDATABASE}-20260101-000000-000.sql.gz").write_bytes(b"")
    (ARCHIVE_DIR / "pg-someotherdb-20260101-000000-000.sql.gz").write_bytes(b"")
    listed = pgarchive._list_archives()
    check(
        len(listed) == 1 and pgarchive.PGDATABASE in listed[0].name,
        f"only the current database's archives are listed (got {[p.name for p in listed]})",
    )

    # -------------------------------------------------------------------
    print()
    print("[5] archive_once(): a healthy dump is published and verified")
    _wipe_archive_dir()
    pgarchive._pg_reachable = lambda: True
    _set_dump({"chat": 25, "message": 60})
    r = pgarchive.archive_once()
    check(r["archived"] is True, f"archived successfully ({r['error']})")
    check(r["rows"] == 85, f"row count matches the dump content (got {r['rows']})")
    archives_after = pgarchive._list_archives()
    check(len(archives_after) == 1, "exactly one archive published")
    check(
        not list(ARCHIVE_DIR.glob("*.partial-*")),
        "no partial temp file left where restore_if_needed() would find it",
    )
    on_disk_rows = pgarchive._count_rows_in_dump(archives_after[0])
    check(on_disk_rows == 85, "the published archive itself actually contains 85 rows")

    # -------------------------------------------------------------------
    print()
    print("[6] archive_once(): pg_dump failing does not touch the previous archive")
    _wipe_archive_dir()
    good_prev = _make_archive(rows=30, tables=1)
    prev_bytes = good_prev.read_bytes()
    _set_dump(None, returncode=1)
    r = pgarchive.archive_once()
    check(r["error"] is not None, f"the failure is reported ({r['error'][:60]})")
    check(good_prev.exists() and good_prev.read_bytes() == prev_bytes, "the previous good archive is byte-for-byte untouched")
    check(len(pgarchive._list_archives()) == 1, "no new (broken) archive was published")

    # -------------------------------------------------------------------
    print()
    print("[7] archive_once(): a Popen exception (e.g. pg_dump missing) is never fatal")
    _wipe_archive_dir()
    _set_dump(None, raise_exc=FileNotFoundError("pg_dump: no such file"))
    r = pgarchive.archive_once()
    check(r["error"] is not None, f"reported as an error, not a crash ({r['error'][:60]})")
    check(not list(ARCHIVE_DIR.glob("*.partial-*")), "no partial file left behind")

    # -------------------------------------------------------------------
    print()
    print("[8] THE GUARD: refuses to publish a near-empty dump over a healthy archive")
    _wipe_archive_dir()
    prev = _make_archive(rows=100, tables=1)  # well above SHRINK_GUARD_MIN_ROWS=10
    prev_bytes = prev.read_bytes()
    _set_dump({"chat": 3})  # 3 rows: far below 100 * 0.5
    r = pgarchive.archive_once()
    check(r["archived"] is False and r["error"] is not None, f"refused to publish ({r})")
    check(prev.exists() and prev.read_bytes() == prev_bytes, "the previous archive (100 rows) survives untouched")
    check(len(pgarchive._list_archives()) == 1, "still exactly one (the original) archive in the main directory")
    quarantined = list((ARCHIVE_DIR / "quarantine").glob("*refused-shrink*"))
    check(len(quarantined) == 1, "the refused dump was quarantined, not deleted")

    # And the override actually overrides:
    os.environ["PGARCHIVE_ALLOW_SHRINK"] = "1"
    pgarchive._reload_env()
    _set_dump({"chat": 3})
    r = pgarchive.archive_once()
    check(r["archived"] is True, f"PGARCHIVE_ALLOW_SHRINK=1 lets a real shrink through ({r.get('error')})")
    os.environ["PGARCHIVE_ALLOW_SHRINK"] = ""
    pgarchive._reload_env()

    # -------------------------------------------------------------------
    print()
    print("[9] the guard stands down below SHRINK_GUARD_MIN_ROWS (small numbers are noisy)")
    _wipe_archive_dir()
    _make_archive(rows=4, tables=1)  # under MIN_ROWS=10 -> ratio is meaningless
    _set_dump({"chat": 1})  # a "75% drop" that means nothing at this scale
    r = pgarchive.archive_once()
    check(r["archived"] is True, f"small previous counts don't trigger the guard ({r.get('error')})")

    # -------------------------------------------------------------------
    print()
    print("[10] retention: prunes down to RETAIN=3, never below MIN_KEEP=2")
    _wipe_archive_dir()
    for i in range(5):
        _make_archive(rows=20 + i, tables=1)
    check(len(pgarchive._list_archives()) == 5, "seed: 5 archives on disk before pruning")
    pgarchive._prune()
    remaining = pgarchive._list_archives()
    check(len(remaining) == 3, f"pruned down to RETAIN=3 (got {len(remaining)})")
    for a in remaining:
        check(pgarchive._meta_path(a).exists(), f"meta for {a.name} pruned alongside its archive, not orphaned")

    # -------------------------------------------------------------------
    print()
    print("[11] restore_if_needed(): a genuinely new deployment starts fresh")
    _wipe_archive_dir()
    pgarchive._table_count = lambda: 0
    r = pgarchive.restore_if_needed()
    check(r["action"] == "fresh", f"fresh start, no archive to restore (action={r['action']})")

    # -------------------------------------------------------------------
    print()
    print("[12] restore_if_needed(): an existing populated database is NEVER touched")
    _wipe_archive_dir()
    _make_archive(rows=999, tables=1)  # a tempting-looking archive that must be ignored
    pgarchive._table_count = lambda: 12
    _set_restore(returncode=0)
    r = pgarchive.restore_if_needed()
    check(r["action"] == "kept_existing", f"kept the existing database (action={r['action']})")
    check(_restore_plan["captured"] == {}, "psql was never even invoked — a live database always wins")

    # -------------------------------------------------------------------
    print()
    print("[13] restore_if_needed(): empty database restores the newest good archive")
    _wipe_archive_dir()
    older = _make_archive(rows=50, tables=1)
    newer = _make_archive(rows=77, tables=1)
    pgarchive._table_count = lambda: 0
    _set_restore(returncode=0)
    r = pgarchive.restore_if_needed()
    check(r["action"] == "restored", f"restored (action={r['action']})")
    check(r["path"] == str(newer), f"restored from the NEWEST archive, not {older.name}")
    # psql receives DECOMPRESSED SQL on stdin (gzip.open(candidate, "rb")
    # yields plain bytes) — not the gzip container itself.
    restored_payload = _restore_plan["captured"]["bytes"].decode("utf-8")
    check("synthetic-lorem-ipsum-row" in restored_payload, "the exact archive content was streamed into psql's stdin")

    # -------------------------------------------------------------------
    print()
    print("[14] restore_if_needed(): a corrupted newest archive falls back to an older good one")
    _wipe_archive_dir()
    older_good = _make_archive(rows=50, tables=1)
    newest_bad = _make_archive(rows=99, tables=1)
    with open(newest_bad, "r+b") as fh:
        fh.seek(0)
        fh.write(b"\xff" * 20)  # corrupt the gzip header itself
    pgarchive._table_count = lambda: 0
    _set_restore(returncode=0)
    r = pgarchive.restore_if_needed()
    check(r["action"] == "restored", f"fell back and restored anyway (action={r['action']})")
    check(r["path"] == str(older_good), f"restored from the older GOOD archive, not the corrupt newest one (got {r['path']})")
    check(newest_bad.exists(), "the corrupt archive was left in place, not deleted")

    # -------------------------------------------------------------------
    print()
    print("[15] restore_if_needed(): psql failing rolls back and tries an older archive")
    _wipe_archive_dir()
    older_ok = _make_archive(rows=50, tables=1)
    newer_ok = _make_archive(rows=77, tables=1)
    pgarchive._table_count = lambda: 0

    _attempt = {"n": 0}
    _real_fake_popen = _fake_popen

    def _flaky_popen(cmd, **kw):
        if cmd[0] == "psql":
            _attempt["n"] += 1
            if _attempt["n"] == 1:
                return _FakeProc(stdin_sink={}, returncode=1, stderr_bytes=b"simulated mid-restore failure")
        return _real_fake_popen(cmd, **kw)

    subprocess.Popen = _flaky_popen
    try:
        r = pgarchive.restore_if_needed()
    finally:
        subprocess.Popen = _fake_popen
    check(r["action"] == "restored", f"succeeded on the second (older) archive (action={r['action']})")
    check(r["path"] == str(older_ok), f"fell back to the older archive after the newest one's restore failed (got {r['path']})")

    # -------------------------------------------------------------------
    print()
    print("[16] restore_if_needed(): every archive failing is reported, not silently swallowed")
    _wipe_archive_dir()
    _make_archive(rows=50, tables=1)
    pgarchive._table_count = lambda: 0
    _set_restore(returncode=1)
    r = pgarchive.restore_if_needed()
    check(r["action"] == "restore_failed", f"reported failure honestly (action={r['action']})")
    check("detail" in r and r["detail"], "a human-readable detail is present")

finally:
    subprocess.Popen = _real_popen
    if hasattr(pgarchive, "_pg_reachable"):
        # Restore the real function in case anything imports this module
        # again in-process (not expected for a __main__ run, but cheap).
        import importlib
        importlib.reload(pgarchive)

print()
if FAILED:
    for f in FAILED:
        print("FAIL " + f)
    sys.exit(1)
print("All pgarchive tests passed.")
