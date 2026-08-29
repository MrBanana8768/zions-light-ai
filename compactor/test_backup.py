"""
CPU-only Tier-1 tests for compactor.backup (V2.3 Theme 1).

The point of this release is the FAILURE paths, so they get the most
coverage: unverifiable archives are rejected and deleted, the disk-full
guard trips, restore refuses bad/unconfirmed input.

v3.1 F2/F7 adds the paths that were destroying history rather than merely
failing to protect it: a missing store must raise instead of publishing an
empty archive, verification must be able to contradict the manifest that
run wrote, a collapsed payload must not publish, a census that went
backwards must not prune, and no sequence of restarts may take the archive
count below the floor.

Sets DATA_DIR / STORAGE_ROOT / BACKUP_DIR to a tmp tree BEFORE importing
backup so module-level config points at the sandbox. Uses a real SQLite
db so the live-snapshot path is actually exercised.

Run: python test_backup.py
"""

import json
import os
import shutil
import sqlite3
import sys
import tarfile
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="zions-backup-test-"))
_DATA = _TMP / "data" / "openwebui"
_STORE = _DATA / "compactor"
_BACKUPS = _TMP / "data" / "backups"
_DB = _DATA / "webui.db"

os.environ["DATA_DIR"] = str(_DATA)
os.environ["COMPACTOR_STORAGE_ROOT"] = str(_STORE)
os.environ["COMPACTOR_BACKUP_DIR"] = str(_BACKUPS)
os.environ["COMPACTOR_BACKUP_WEBUI_DB"] = str(_DB)
os.environ["COMPACTOR_BACKUP_RETAIN"] = "3"

import backup  # noqa: E402


def assert_eq(a, b, label):
    if a != b:
        print(f"FAIL {label}: expected {b!r}, got {a!r}")
        sys.exit(1)
    print(f"  ok   {label}")


def assert_true(c, label):
    if not c:
        print(f"FAIL {label}")
        sys.exit(1)
    print(f"  ok   {label}")


def assert_raises(fn, exc, label):
    try:
        fn()
    except exc:
        print(f"  ok   {label}")
        return
    except Exception as e:
        print(f"FAIL {label}: expected {exc.__name__}, got {type(e).__name__}: {e}")
        sys.exit(1)
    print(f"FAIL {label}: nothing raised")
    sys.exit(1)


def _seed_sources(*, with_db=True, facts_text="seed fact", n_facts=1,
                  pad=0, n_summaries=0, episodic=None):
    """Create a realistic source tree: a live-ish sqlite db + memory files.

    n_facts / pad / n_summaries / episodic exist so a test can make the store
    grow or shrink in a controlled way — the v3.1 payload and census guards
    are entirely about the delta between two archives.
    """
    if _STORE.exists():
        shutil.rmtree(_STORE)
    (_STORE / "facts").mkdir(parents=True, exist_ok=True)
    (_STORE / "summaries").mkdir(parents=True, exist_ok=True)
    (_STORE / "facts" / "conv1.json").write_text(
        json.dumps({"conv_id": "conv1", "facts": [
            {"text": (facts_text if n_facts == 1 else f"{facts_text} {i}")
                     + ("x" * pad)}
            for i in range(n_facts)
        ]}),
        encoding="utf-8",
    )
    if n_summaries:
        (_STORE / "summaries" / "conv1.json").write_text(
            json.dumps({"conv_id": "conv1", "l1": [
                {"text": f"chunk {i}", "first_turn": i, "last_turn": i + 1}
                for i in range(n_summaries)
            ], "l2": [], "l3": None}),
            encoding="utf-8",
        )
    if episodic is not None:
        _seed_chroma(episodic)
    if _DB.exists():
        _DB.unlink()
    if with_db:
        _DATA.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(str(_DB))
        con.execute("CREATE TABLE chat (id INTEGER PRIMARY KEY, body TEXT)")
        con.execute("INSERT INTO chat (body) VALUES ('hello')")
        con.commit()
        con.close()


def _seed_chroma(counts: dict):
    """Minimal stand-in for ChromaDB's persistent SQLite: the one table
    _episodic_counts reads, with the same `conv_id` metadata key
    retrieval.py:172 writes."""
    cdir = _STORE / "chromadb"
    cdir.mkdir(parents=True, exist_ok=True)
    db = cdir / "chroma.sqlite3"
    if db.exists():
        db.unlink()
    con = sqlite3.connect(str(db))
    con.execute(
        "CREATE TABLE embedding_metadata ("
        "id INTEGER, key TEXT, string_value TEXT, int_value INTEGER)"
    )
    for conv_id, n in counts.items():
        for i in range(n):
            con.execute(
                "INSERT INTO embedding_metadata (id, key, string_value) "
                "VALUES (?, 'conv_id', ?)", (i, conv_id),
            )
    con.commit()
    con.close()
    return db


def _clean_backups():
    if _BACKUPS.exists():
        shutil.rmtree(_BACKUPS)


def _make_archive(name: str, manifest: dict, files: dict | None = None):
    """Hand-build an archive with an arbitrary manifest and payload, for the
    verifier tests that need a manifest which lies."""
    _BACKUPS.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp())
    try:
        for rel, body in (files or {}).items():
            p = staging / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(body, encoding="utf-8")
        (staging / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        arch = _BACKUPS / name
        with tarfile.open(arch, "w:gz") as tar:
            tar.add(staging, arcname=".")
        return arch
    finally:
        shutil.rmtree(staging, ignore_errors=True)


class _CapturedAlerts:
    """Swap in for backup._alert_failure. The whole point of these paths is
    that they are loud, so 'was an alert fired' is an assertion, not a
    detail."""

    def __init__(self):
        self.sent = []
        self._orig = None

    def __enter__(self):
        self._orig = backup._alert_failure
        backup._alert_failure = self.sent.append
        return self

    def __exit__(self, *exc):
        backup._alert_failure = self._orig
        return False


# ---------------------------------------------------------------------------
# Happy path + round-trip
# ---------------------------------------------------------------------------

def test_create_verify_publish_round_trip():
    print("\n[test] run_once: create → verify → publish, no .partial left behind")
    _seed_sources()
    _clean_backups()
    rep = backup.run_once()
    assert_true(rep["ok"], "run_once ok")
    assert_true(rep["verified"], "verified flag set")
    assert_true(rep["archive"].endswith(".tar.gz"), "archive is .tar.gz")
    # No leftover .partial
    partials = list(_BACKUPS.glob("*.partial"))
    assert_eq(partials, [], "no .partial files remain")
    # Archive actually exists and verifies on its own
    arch = _BACKUPS / rep["archive"]
    ok, _ = backup.verify_backup(arch)
    assert_true(ok, "published archive independently verifies")


def test_restore_round_trip_recovers_data():
    print("\n[test] restore: wipe sources, restore from archive, data comes back")
    _seed_sources(facts_text="precious memory")
    _clean_backups()
    rep = backup.run_once()
    arch = _BACKUPS / rep["archive"]

    # Simulate disaster: wipe the live store + db
    shutil.rmtree(_STORE)
    _DB.unlink()
    assert_true(not _STORE.exists(), "store wiped")

    res = backup.restore_backup(arch, confirm=True)
    assert_true(res["ok"], "restore ok")
    assert_true("compactor" in res["restored"], "compactor restored")
    assert_true("webui.db" in res["restored"], "webui.db restored")
    # The precious fact is back
    data = json.loads((_STORE / "facts" / "conv1.json").read_text())
    assert_eq(data["facts"][0]["text"], "precious memory", "fact content recovered")
    # The db is back and openable
    con = sqlite3.connect(str(_DB))
    n = con.execute("SELECT COUNT(*) FROM chat").fetchone()[0]
    con.close()
    assert_eq(n, 1, "db row recovered")


def test_backup_without_db_succeeds():
    print("\n[test] missing webui.db → backup still succeeds (memory-only)")
    _seed_sources(with_db=False)
    _clean_backups()
    rep = backup.run_once()
    assert_true(rep["ok"], "memory-only backup ok")
    ok, detail = backup.verify_backup(_BACKUPS / rep["archive"])
    assert_true(ok, "verifies")
    assert_true("absent" in detail, "detail notes db absent")


# ---------------------------------------------------------------------------
# Failure paths — the heart of this release
# ---------------------------------------------------------------------------

def test_verify_rejects_truncated_archive():
    print("\n[test] verify: a truncated/garbage archive fails cleanly (no raise)")
    _clean_backups()
    _BACKUPS.mkdir(parents=True, exist_ok=True)
    bad = _BACKUPS / "zions-backup-garbage.tar.gz"
    bad.write_bytes(b"this is not a gzip tar at all")
    ok, detail = backup.verify_backup(bad)
    assert_eq(ok, False, "garbage archive → not ok")
    assert_true("extract failed" in detail, "detail explains extract failure")


def test_verify_rejects_corrupt_memory_json():
    print("\n[test] verify: a corrupt memory JSON inside the archive fails")
    # Hand-build an archive with a broken facts file
    _clean_backups()
    _BACKUPS.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp())
    try:
        (staging / "compactor" / "facts").mkdir(parents=True)
        (staging / "compactor" / "facts" / "bad.json").write_text("{not valid json")
        (staging / "manifest.json").write_text(json.dumps({
            "schema": "v1", "sources": {"webui.db": {"present": False},
                                         "compactor": {"present": True}},
        }))
        arch = _BACKUPS / "zions-backup-corruptjson.tar.gz"
        with tarfile.open(arch, "w:gz") as tar:
            tar.add(staging, arcname=".")
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    ok, detail = backup.verify_backup(arch)
    assert_eq(ok, False, "corrupt json → not ok")
    assert_true("corrupt memory file" in detail, "detail names the failure")


def test_verify_rejects_bad_sqlite():
    print("\n[test] verify: manifest claims db present but it's not a real db")
    _clean_backups()
    _BACKUPS.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp())
    try:
        (staging / "webui.db").write_text("definitely not sqlite")
        # The compactor half has to be present and sound, or verification
        # stops at "no compactor store" before it ever opens the db.
        (staging / "compactor" / "facts").mkdir(parents=True)
        (staging / "compactor" / "facts" / "conv1.json").write_text(
            json.dumps({"conv_id": "conv1", "facts": []}))
        (staging / "manifest.json").write_text(json.dumps({
            "schema": "v1", "sources": {"webui.db": {"present": True},
                                        "compactor": {"present": True}},
        }))
        arch = _BACKUPS / "zions-backup-baddb.tar.gz"
        with tarfile.open(arch, "w:gz") as tar:
            tar.add(staging, arcname=".")
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    ok, detail = backup.verify_backup(arch)
    assert_eq(ok, False, "bad db → not ok")
    assert_true("sqlite" in detail.lower() or "integrity" in detail.lower(),
                "detail mentions sqlite/integrity")


def test_verify_rejects_missing_manifest():
    print("\n[test] verify: archive with no manifest fails")
    _clean_backups()
    _BACKUPS.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp())
    try:
        (staging / "compactor").mkdir()
        arch = _BACKUPS / "zions-backup-nomanifest.tar.gz"
        with tarfile.open(arch, "w:gz") as tar:
            tar.add(staging, arcname=".")
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    ok, detail = backup.verify_backup(arch)
    assert_eq(ok, False, "no manifest → not ok")
    assert_true("manifest" in detail, "detail mentions manifest")


def test_run_once_discards_unverifiable_archive():
    print("\n[test] run_once: if verify fails, archive is deleted + FAIL reported")
    _seed_sources()
    _clean_backups()
    # Force verification to fail
    orig = backup.verify_backup
    backup.verify_backup = lambda p: (False, "forced failure")
    try:
        rep = backup.run_once()
    finally:
        backup.verify_backup = orig
    assert_eq(rep["ok"], False, "run reports failure")
    assert_true("VERIFICATION FAILED" in rep["detail"], "detail flags verification failure")
    # No archive (and no .partial) left — no false confidence
    leftovers = list(_BACKUPS.glob("*.tar.gz*"))
    assert_eq(leftovers, [], "unverifiable archive discarded, nothing left")


def test_min_free_guard_blocks_backup():
    print("\n[test] create_backup: disk-full guard raises rather than filling /data")
    _seed_sources()
    _clean_backups()
    orig = backup._free_mb
    backup._free_mb = lambda p: 1.0  # pretend nearly full
    try:
        assert_raises(lambda: backup.create_backup(), RuntimeError, "min-free guard trips")
    finally:
        backup._free_mb = orig


# ---------------------------------------------------------------------------
# v3.1 F2 — the empty backup that verified, published and pruned
# ---------------------------------------------------------------------------

def test_missing_storage_root_fails_alerts_and_does_not_prune():
    print("\n[test] F2: a missing store fails the cycle, alerts, and prunes nothing")
    _seed_sources()
    _clean_backups()
    # A real archive to lose.
    first = backup.run_once()
    assert_true(first["ok"], "baseline backup ok")

    orig = backup.STORAGE_ROOT
    backup.STORAGE_ROOT = _TMP / "not" / "mounted" / "anywhere"
    try:
        with _CapturedAlerts() as alerts:
            rep = backup.run_once()
    finally:
        backup.STORAGE_ROOT = orig

    assert_eq(rep["ok"], False, "cycle fails rather than publishing an empty archive")
    assert_true("not a directory" in rep["detail"], "detail names the missing store")
    assert_eq(len(alerts.sent), 1, "exactly one alert fired")
    assert_true("not a directory" in alerts.sent[0], "the alert carries the reason")
    names = {r["name"] for r in backup.list_backups()}
    assert_eq(names, {first["archive"]}, "the real archive survives; nothing new published")
    assert_eq(list(_BACKUPS.glob("*.partial")), [], "no .partial left behind")


def test_verify_rejects_archive_with_no_compactor_store():
    print("\n[test] F2: an archive whose manifest records no store fails verification")
    _clean_backups()
    # The exact shape the bug produced: nothing but manifest.json. Before
    # v3.1 this returned (True, "db=absent, 0 json file(s) parsed").
    arch = _make_archive("zions-backup-empty.tar.gz", {
        "schema": "v1",
        "sources": {"webui.db": {"present": False}, "compactor": {"present": False}},
    })
    ok, detail = backup.verify_backup(arch)
    assert_eq(ok, False, "empty archive → not ok")
    assert_true("not a recovery point" in detail, "detail says why")

    # And a perfectly good webui.db does not redeem it — the memory store is
    # the half that cannot be regenerated.
    staging = Path(tempfile.mkdtemp())
    try:
        con = sqlite3.connect(str(staging / "webui.db"))
        con.execute("CREATE TABLE chat (id INTEGER PRIMARY KEY)")
        con.commit()
        con.close()
        (staging / "manifest.json").write_text(json.dumps({
            "schema": "v1", "sources": {"webui.db": {"present": True},
                                        "compactor": {"present": False}},
        }), encoding="utf-8")
        arch2 = _BACKUPS / "zions-backup-dbonly.tar.gz"
        with tarfile.open(arch2, "w:gz") as tar:
            tar.add(staging, arcname=".")
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    ok, detail = backup.verify_backup(arch2)
    assert_eq(ok, False, "db-only archive → not ok")
    assert_true("no compactor store" in detail, "detail names the missing store")


def test_verify_rejects_missing_store_the_manifest_claims():
    print("\n[test] F2: manifest claims a compactor store, archive has none")
    _clean_backups()
    arch = _make_archive("zions-backup-nostore.tar.gz", {
        "schema": "v2",
        "sources": {"webui.db": {"present": False},
                    "compactor": {"present": True, "files": 3, "json_files": 3}},
    })
    ok, detail = backup.verify_backup(arch)
    assert_eq(ok, False, "missing store → not ok")
    assert_true("compactor/ is missing" in detail, "detail names the missing directory")


def test_verify_rejects_json_count_shortfall():
    print("\n[test] F2: fewer memory JSON files than the manifest counted")
    _clean_backups()
    arch = _make_archive("zions-backup-short.tar.gz", {
        "schema": "v2",
        "sources": {"webui.db": {"present": False},
                    "compactor": {"present": True, "files": 5, "json_files": 5}},
    }, files={"compactor/facts/conv1.json": json.dumps({"facts": []})})
    ok, detail = backup.verify_backup(arch)
    assert_eq(ok, False, "short archive → not ok")
    assert_true("only 1 are in the archive" in detail, "detail gives both counts")


def test_verify_rejects_census_shortfall():
    print("\n[test] F2: the archive holds fewer facts than its own census claims")
    _clean_backups()
    arch = _make_archive("zions-backup-census.tar.gz", {
        "schema": "v2",
        "sources": {"webui.db": {"present": False}, "compactor": {
            "present": True, "files": 1, "json_files": 1,
            "conversations": {"conv1": {"facts": 9, "summaries": 0, "episodic": 0}},
        }},
    }, files={"compactor/facts/conv1.json": json.dumps(
        {"conv_id": "conv1", "facts": [{"text": "only one"}]})})
    ok, detail = backup.verify_backup(arch)
    assert_eq(ok, False, "census shortfall → not ok")
    assert_true("conv1.facts 9->1" in detail, "detail names the conversation and layer")


def test_manifest_records_the_per_conversation_census():
    print("\n[test] F2: manifest carries fact/summary/episodic counts per conversation")
    _seed_sources(n_facts=4, n_summaries=3, episodic={"conv1": 6, "conv2": 2})
    _clean_backups()
    rep = backup.run_once()
    assert_true(rep["ok"], "backup ok")
    man = backup.read_manifest(_BACKUPS / rep["archive"])
    src = man["sources"]["compactor"]
    assert_eq(src["conversations"]["conv1"]["facts"], 4, "fact count recorded")
    assert_eq(src["conversations"]["conv1"]["summaries"], 3, "summary count recorded")
    assert_eq(src["conversations"]["conv1"]["episodic"], 6, "episodic count recorded")
    assert_eq(src["conversations"]["conv2"]["episodic"], 2, "second conversation too")
    assert_eq(src["chroma_sqlite"], True, "chroma.sqlite3 snapshotted")


def test_chroma_is_snapshotted_and_integrity_checked():
    print("\n[test] F2: chroma.sqlite3 goes through the backup API and is checked")
    _seed_sources(episodic={"conv1": 3})
    _clean_backups()
    rep = backup.run_once()
    assert_true(rep["ok"], "backup ok")
    arch = _BACKUPS / rep["archive"]
    with tarfile.open(arch, "r:gz") as tar:
        members = set(tar.getnames())
    assert_true("./compactor/chromadb/chroma.sqlite3" in members,
                "the episodic db is in the archive")
    assert_true("chroma=ok" in rep["detail"], "verification reports the chroma check")

    # Corrupt the snapshot inside a rebuilt archive: integrity_check must fail.
    # (It checks SQLite pages only — see the note in verify_backup. A green
    # result is not a statement about the memory being coherent.)
    scratch = Path(tempfile.mkdtemp())
    try:
        with tarfile.open(arch, "r:gz") as tar:
            tar.extractall(scratch, filter="data")
        cdb = scratch / "compactor" / "chromadb" / "chroma.sqlite3"
        with open(cdb, "r+b") as fh:
            fh.seek(200)
            fh.write(b"\xff" * 4096)
        bad = _BACKUPS / "zions-backup-badchroma.tar.gz"
        with tarfile.open(bad, "w:gz") as tar:
            tar.add(scratch, arcname=".")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    ok, detail = backup.verify_backup(bad)
    assert_eq(ok, False, "corrupted chroma snapshot → not ok")
    assert_true("chroma.sqlite3" in detail, "detail names chroma.sqlite3")


def test_payload_collapse_is_refused_and_does_not_prune():
    print("\n[test] F2: an archive under 50% of the previous one is not published")
    _seed_sources(n_facts=40, pad=4000)
    _clean_backups()
    first = backup.run_once()
    assert_true(first["ok"], "fat baseline backup ok")

    # The store shrinks to almost nothing — the shape a half-mounted volume
    # produces, and the shape a legitimate edit does not.
    _seed_sources(n_facts=1, pad=0, with_db=False)
    with _CapturedAlerts() as alerts:
        rep = backup.run_once()

    assert_eq(rep["ok"], False, "collapsed payload → cycle fails")
    assert_true("PAYLOAD COLLAPSED" in rep["detail"], "detail flags the collapse")
    assert_eq(len(alerts.sent), 1, "an alert fired")
    names = {r["name"] for r in backup.list_backups()}
    assert_eq(names, {first["archive"]}, "the fat archive survives, the thin one is gone")


def test_census_regression_publishes_but_refuses_to_prune():
    print("\n[test] F2: memory shrank → archive published, prune skipped, alert sent")
    _seed_sources(n_facts=6, pad=2000)
    _clean_backups()
    first = backup.run_once()
    assert_true(first["ok"], "baseline ok")

    import time
    time.sleep(1.05)
    # One fact fewer, but more bytes overall — the payload guard cannot see
    # this, which is exactly why the census exists.
    _seed_sources(n_facts=5, pad=4000)
    with _CapturedAlerts() as alerts:
        rep = backup.run_once()

    assert_true(rep["ok"], "the archive still publishes — it is real data")
    assert_true(rep["archive"] is not None, "published")
    assert_eq(rep["pruned"], [], "nothing pruned on a cycle that saw a loss")
    assert_true("conv1.facts 6->5" in rep["detail"], "detail names the loss")
    assert_eq(len(alerts.sent), 1, "an alert fired")
    assert_eq(len(backup.list_backups()), 2, "both archives on disk")


# ---------------------------------------------------------------------------
# Prune / restore gating / info
# ---------------------------------------------------------------------------

def test_prune_keeps_everything_inside_the_age_window():
    print("\n[test] prune: five backups minutes apart, none pruned (v3.1 F7)")
    _seed_sources()
    _clean_backups()
    names = []
    for _ in range(5):
        import time
        time.sleep(1.05)  # ensure distinct YYYYmmdd-HHMMSS stamps
        rep = backup.run_once()
        assert_true(rep["ok"], "backup ok")
        names.append(rep["archive"])
    kept = {r["name"] for r in backup.list_backups()}
    # Under the old "keep the newest RETAIN=3" cap two of these were deleted
    # minutes after being written. All five are inside RETAIN_DAYS.
    assert_eq(kept, set(names), "all five survive — RETAIN is a floor, not a cap")


def _fake_archives(ages_days):
    """Create empty archive files with backdated mtimes. Retention only ever
    looks at names and mtimes, so the contents are irrelevant here."""
    _clean_backups()
    _BACKUPS.mkdir(parents=True, exist_ok=True)
    import time as _t
    now = _t.time()
    made = []
    for age in ages_days:
        stamp = _t.strftime("%Y%m%d-%H%M%S", _t.gmtime(now - age * 86400))
        p = _BACKUPS / f"zions-backup-{stamp}-{age}d.tar.gz"
        p.write_bytes(b"x")
        os.utime(p, (now - age * 86400, now - age * 86400))
        made.append(p.name)
    return now, made


def test_prune_age_and_gfs_tiers():
    print("\n[test] prune: age tier keeps 14d, GFS keeps one per week for 8w")
    # 1d and 10d are inside the age window; 20d/34d are in distinct ISO weeks
    # inside the 8-week GFS window; 200d is claimed by no tier.
    now, names = _fake_archives([1, 2, 10, 20, 34, 200])
    removed = backup.prune_old_backups(now=now)
    kept = {r["name"] for r in backup.list_backups()}
    assert_true(names[0] in kept and names[1] in kept and names[2] in kept,
                "everything younger than 14d is kept")
    assert_true(names[3] in kept and names[4] in kept,
                "20d and 34d kept by the weekly GFS tier")
    assert_true(names[5] not in kept, "the 200d archive is pruned")
    assert_eq(removed, [names[5]], "only the unclaimed archive was removed")


def test_prune_never_goes_below_the_floor():
    print("\n[test] prune: three ancient archives all survive (hard floor)")
    now, names = _fake_archives([400, 500, 600])
    removed = backup.prune_old_backups(now=now)
    assert_eq(removed, [], "nothing pruned")
    assert_eq(len(backup.list_backups()), 3, "all three survive despite age")
    # And the floor cannot be configured away from the call site either.
    removed = backup.prune_old_backups(retain=0, now=now)
    assert_eq(removed, [], "retain=0 cannot empty the backup directory")


def test_ten_restarts_do_not_erase_the_oldest_archive():
    print("\n[test] F7: ten restart cycles in a row, the oldest archive survives")
    _seed_sources()
    now, _ = _fake_archives([13.9])   # just inside the age window
    oldest = backup.list_backups()[0]["name"]
    for _ in range(10):
        import time
        time.sleep(1.05)
        rep = backup.run_once()
        assert_true(rep["ok"], "cycle ok")
    kept = {r["name"] for r in backup.list_backups()}
    # Under the old scheme this loop left RETAIN archives, all of them
    # younger than the restart that started it.
    assert_true(oldest in kept, "the pre-restart archive is still there")
    assert_eq(len(kept), 11, "ten new archives plus the original")


def test_boot_run_suppressed_when_a_recent_archive_exists():
    print("\n[test] F7: run_daemon's boot cycle is suppressed after a recent backup")
    _seed_sources()
    _clean_backups()
    assert_true(backup._newest_archive_age_s() is None,
                "no archives → nothing to suppress, the boot run must happen")
    backup.run_once()
    age = backup._newest_archive_age_s()
    assert_true(age is not None and age < (6 * 3600) / 2,
                "fresh archive is younger than half a 6h interval → boot run skipped")


def test_restore_requires_confirm():
    print("\n[test] restore: refuses without confirm=True")
    _seed_sources()
    _clean_backups()
    rep = backup.run_once()
    arch = _BACKUPS / rep["archive"]
    assert_raises(lambda: backup.restore_backup(arch, confirm=False),
                  RuntimeError, "unconfirmed restore raises")


def test_restore_refuses_unverifiable_archive():
    print("\n[test] restore: refuses an archive that doesn't verify")
    _clean_backups()
    _BACKUPS.mkdir(parents=True, exist_ok=True)
    bad = _BACKUPS / "zions-backup-bad.tar.gz"
    bad.write_bytes(b"garbage")
    assert_raises(lambda: backup.restore_backup(bad, confirm=True),
                  RuntimeError, "unverifiable restore raises")


def test_latest_backup_info_shape():
    print("\n[test] latest_backup_info: count + latest after a backup")
    _seed_sources()
    _clean_backups()
    info0 = backup.latest_backup_info()
    assert_eq(info0["count"], 0, "zero before any backup")
    assert_eq(info0["latest"], None, "no latest")
    backup.run_once()
    info1 = backup.latest_backup_info()
    assert_eq(info1["count"], 1, "one after backup")
    assert_true(info1["latest"] is not None, "latest set")
    assert_true(info1["latest_mtime"] is not None, "mtime set")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _all():
    return [
        test_create_verify_publish_round_trip,
        test_restore_round_trip_recovers_data,
        test_backup_without_db_succeeds,
        test_verify_rejects_truncated_archive,
        test_verify_rejects_corrupt_memory_json,
        test_verify_rejects_bad_sqlite,
        test_verify_rejects_missing_manifest,
        test_run_once_discards_unverifiable_archive,
        test_min_free_guard_blocks_backup,
        test_missing_storage_root_fails_alerts_and_does_not_prune,
        test_verify_rejects_archive_with_no_compactor_store,
        test_verify_rejects_missing_store_the_manifest_claims,
        test_verify_rejects_json_count_shortfall,
        test_verify_rejects_census_shortfall,
        test_manifest_records_the_per_conversation_census,
        test_chroma_is_snapshotted_and_integrity_checked,
        test_payload_collapse_is_refused_and_does_not_prune,
        test_census_regression_publishes_but_refuses_to_prune,
        test_prune_keeps_everything_inside_the_age_window,
        test_prune_age_and_gfs_tiers,
        test_prune_never_goes_below_the_floor,
        test_ten_restarts_do_not_erase_the_oldest_archive,
        test_boot_run_suppressed_when_a_recent_archive_exists,
        test_restore_requires_confirm,
        test_restore_refuses_unverifiable_archive,
        test_latest_backup_info_shape,
    ]


if __name__ == "__main__":
    try:
        for t in _all():
            t()
        print("\nAll backup smoke tests passed.")
    finally:
        shutil.rmtree(_TMP, ignore_errors=True)
