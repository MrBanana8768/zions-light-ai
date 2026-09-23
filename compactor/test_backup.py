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
import threading
import time
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="zions-backup-test-"))
_DATA = _TMP / "data" / "openwebui"
_STORE = _DATA / "compactor"
_BACKUPS = _TMP / "data" / "backups"
_DB = _DATA / "webui.db"
# v3.1.9 (A3-10): restore set-asides now land in webuidb.QUARANTINE, not
# beside their targets. Set BEFORE backup.py can lazily `import webuidb`
# (the first restore_backup call), or the module default (/data/forensics)
# would apply and tests would write outside the sandbox.
_QUARANTINE = _TMP / "quarantine"
# D2: a sandboxed local-disk staging dir for _snapshot_sqlite_to_data, set
# BEFORE import so the module default (/var/lib/openwebui/.backup-staging)
# never applies and no test writes outside the sandbox.
_LOCAL_STAGING = _TMP / "local-staging"

os.environ["DATA_DIR"] = str(_DATA)
os.environ["COMPACTOR_STORAGE_ROOT"] = str(_STORE)
os.environ["COMPACTOR_BACKUP_DIR"] = str(_BACKUPS)
os.environ["COMPACTOR_BACKUP_WEBUI_DB"] = str(_DB)
os.environ["COMPACTOR_BACKUP_RETAIN"] = "3"
os.environ["WEBUI_DB_QUARANTINE"] = str(_QUARANTINE)
os.environ["COMPACTOR_BACKUP_LOCAL_STAGING_DIR"] = str(_LOCAL_STAGING)

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
            ], "l2": [], "l3": None,
                # v3.1.9 (F1): the census reads this watermark now, not chunk
                # count — see backup._census. n_summaries chunks of one turn
                # each cover turns 1..n_summaries.
                "last_summarized_turn": n_summaries}),
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


def test_backup_without_db_refuses_by_default():
    print("\n[test] p3-b F12: missing webui.db refuses the cycle by default (used to succeed silently)")
    # v3.1 F2's own doctrine ("an archive that holds nothing is a
    # failure") was applied to the compactor store only, not its sibling
    # webui.db — a missing db used to log one WARNING and publish a
    # memory-only archive anyway. p3-b F12: with the store the SMALLER
    # half of a real payload, that archive passed the payload-ratio guard
    # and pruned the real archives behind it. Now it refuses, matching
    # the store's own behavior a few lines above it in create_backup.
    _seed_sources(with_db=False)
    _clean_backups()
    rep = backup.run_once()
    assert_true(not rep["ok"], "F12 fix: refuses instead of publishing memory-only")
    assert_true("webui.db" in rep["detail"], "detail names webui.db")


def test_backup_without_db_still_succeeds_with_the_escape_hatch():
    print("\n[test] p3-b F12 CONTROL: COMPACTOR_BACKUP_ALLOW_NO_WEBUI_DB=1 restores the old behavior")
    _seed_sources(with_db=False)
    _clean_backups()
    os.environ["COMPACTOR_BACKUP_ALLOW_NO_WEBUI_DB"] = "1"
    try:
        rep = backup.run_once()
        assert_true(rep["ok"], "memory-only backup ok with the escape hatch set")
        ok, detail = backup.verify_backup(_BACKUPS / rep["archive"])
        assert_true(ok, "verifies")
        assert_true("absent" in detail, "detail notes db absent")
    finally:
        os.environ.pop("COMPACTOR_BACKUP_ALLOW_NO_WEBUI_DB", None)


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
    # v3.1.9 (F1): "summaries" (chunk count) was replaced by "summary_turn"
    # (coverage) — see backup._census's docstring.
    assert_eq(src["conversations"]["conv1"]["summary_turn"], 3, "summary coverage recorded")
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


# ---------------------------------------------------------------------------
# D2: the live DB's read lock must never span the write to /data.
# ---------------------------------------------------------------------------

def test_snapshot_to_data_releases_the_live_lock_before_touching_data():
    print("\n[test] D2: the live db's read lock is released before the /data copy, "
          "not held across it")
    _seed_sources()
    _clean_backups()
    dest = _TMP / "d2-dest.sqlite3"
    dest.unlink(missing_ok=True)

    # Stand in for a slow/stalling /data: block inside shutil.copy2 (stage 2,
    # the plain file copy) until the test has had a chance to write to the
    # LIVE source. If stage 1's backup-API lock were still held at that
    # point, this commit would raise "database is locked" (busy_timeout is
    # 0 by default on a plain sqlite3.connect, so it would fail immediately
    # rather than wait).
    copy_started = threading.Event()
    release_copy = threading.Event()
    orig_copy2 = shutil.copy2

    def _slow_copy2(src, dst, *a, **kw):
        copy_started.set()
        release_copy.wait(timeout=10)
        return orig_copy2(src, dst, *a, **kw)

    backup.shutil.copy2 = _slow_copy2
    try:
        t = threading.Thread(
            target=lambda: backup._snapshot_sqlite_to_data(_DB, dest)
        )
        t.start()
        assert_true(copy_started.wait(timeout=10),
                    "the /data-bound copy started (stage 2 reached)")
        # The live database's own lock must already be free here.
        con = sqlite3.connect(str(_DB), timeout=0)
        con.execute("insert into chat (body) values ('written mid-copy')")
        con.commit()
        con.close()
        release_copy.set()
        t.join(timeout=10)
        assert_true(not t.is_alive(), "the snapshot call finished")
    finally:
        backup.shutil.copy2 = orig_copy2
        release_copy.set()

    assert_true(dest.exists(), "the /data-side destination was written")
    con = sqlite3.connect(str(dest))
    n = con.execute("select count(*) from chat").fetchone()[0]
    con.close()
    assert_true(n >= 1, "the published snapshot is a real, readable copy")


def test_snapshot_to_data_stages_locally_first():
    print("\n[test] D2: the backup API's destination is a LOCAL file, never /data "
          "directly")
    _seed_sources()
    dest = _TMP / "d2-dest2.sqlite3"
    dest.unlink(missing_ok=True)
    seen_dests = []
    orig_snapshot_sqlite = backup._snapshot_sqlite

    def _spy(src, d):
        seen_dests.append(d)
        return orig_snapshot_sqlite(src, d)

    backup._snapshot_sqlite = _spy
    try:
        ok = backup._snapshot_sqlite_to_data(_DB, dest)
    finally:
        backup._snapshot_sqlite = orig_snapshot_sqlite
    assert_true(ok, "the snapshot reported success")
    assert_eq(len(seen_dests), 1, "the backup API ran exactly once")
    assert_true(
        _LOCAL_STAGING in seen_dests[0].parents,
        f"the backup API's own destination ({seen_dests[0]}) is under the "
        f"LOCAL staging dir, not {dest}",
    )
    assert_true(dest.exists(), "the final /data-side copy still landed")


def test_snapshot_to_data_falls_back_when_local_disk_is_full():
    print("\n[test] D2: falls back to the old direct-to-/data path when local "
          "disk has no room, rather than failing the backup")
    _seed_sources()
    dest = _TMP / "d2-dest3.sqlite3"
    dest.unlink(missing_ok=True)
    orig_free_mb = backup._free_mb
    backup._free_mb = lambda p: 0.0  # "no room anywhere"
    direct_calls = []
    orig_snapshot_sqlite = backup._snapshot_sqlite

    def _spy(src, d):
        direct_calls.append(d)
        return orig_snapshot_sqlite(src, d)

    backup._snapshot_sqlite = _spy
    try:
        ok = backup._snapshot_sqlite_to_data(_DB, dest)
    finally:
        backup._free_mb = orig_free_mb
        backup._snapshot_sqlite = orig_snapshot_sqlite
    assert_true(ok, "still succeeds — a full local disk falls back, it does not fail")
    assert_eq(direct_calls, [dest],
              "fell back to the old direct call, straight onto dest")
    assert_true(dest.exists(), "and the backup still landed on /data")


def test_backup_once_survives_a_stalling_data_with_a_live_writer():
    print("\n[test] D2: backup.run_once (webui.db path) does not fail while a "
          "writer is committing to the live db, even with a slow /data")
    _seed_sources()
    _clean_backups()
    orig_copy2 = shutil.copy2

    def _stalling_copy2(src, dst, *a, **kw):
        if str(_BACKUPS) in str(dst):
            time.sleep(0.4)
        return orig_copy2(src, dst, *a, **kw)

    backup.shutil.copy2 = _stalling_copy2
    stop = threading.Event()
    errors = []

    def _writer():
        # timeout=10 mirrors OpenWebUI's own DATABASE_SQLITE_PRAGMA_BUSY_TIMEOUT
        # (10s, see _snapshot_sqlite_to_data's docstring) - a real writer waits
        # out a brief lock rather than failing on it instantly. The assertion
        # below is about "database is locked" errors that OUTLAST that
        # tolerance, which is exactly what D2 used to produce.
        con = sqlite3.connect(str(_DB), timeout=10)
        i = 0
        while not stop.is_set():
            try:
                con.execute("insert into chat (body) values (?)", (f"row{i}",))
                con.commit()
            except sqlite3.OperationalError as e:
                errors.append(str(e))
            i += 1
            time.sleep(0.05)
        con.close()

    t = threading.Thread(target=_writer)
    t.start()
    try:
        time.sleep(0.1)
        rep = backup.run_once()
    finally:
        stop.set()
        t.join(timeout=10)
        backup.shutil.copy2 = orig_copy2
    assert_true(rep["ok"], "run_once still succeeds with /data stalling")
    assert_eq(errors, [], "no 'database is locked' errors on the live writer "
              "while the /data copy stalled (D2)")


def test_payload_collapse_is_refused_and_does_not_prune():
    print("\n[test] F2: an archive under 50% of the previous one is not published")
    _seed_sources(n_facts=40, pad=4000)
    _clean_backups()
    first = backup.run_once()
    assert_true(first["ok"], "fat baseline backup ok")

    # The store shrinks to almost nothing — the shape a half-mounted volume
    # produces, and the shape a legitimate edit does not. This test's
    # subject is the STORE collapsing, not the db, so webui.db is left
    # PRESENT (with_db defaults to True, same tiny one-row db as the
    # baseline) — p3-b F12 added its own refusal for a MISSING db, and
    # p4-b G6 made the escape hatch that used to work around F12 here
    # INERT once any archive exists (as one already does, from the "fat
    # baseline" a few lines up) — so this fixture no longer touches the
    # hatch at all, and only ever exercises the payload-collapse check it
    # is actually about. (A previous version of this test used
    # with_db=False plus COMPACTOR_BACKUP_ALLOW_NO_WEBUI_DB=1 to dodge
    # F12's refusal; G6's expiry made that combination refuse for a
    # DIFFERENT reason — "hatch no longer applies" — instead of ever
    # reaching the payload-collapse check, a wrong-reason failure caught by
    # this lane's own whole-suite run.)
    _seed_sources(n_facts=1, pad=0)
    with _CapturedAlerts() as alerts:
        rep = backup.run_once()

    assert_eq(rep["ok"], False, "collapsed payload → cycle fails")
    assert_true("PAYLOAD COLLAPSED" in rep["detail"], "detail flags the collapse")
    assert_eq(len(alerts.sent), 1, "an alert fired")
    names = {r["name"] for r in backup.list_backups()}
    assert_eq(names, {first["archive"]}, "the fat archive survives, the thin one is gone")


def test_census_regression_publishes_but_refuses_to_prune():
    """v3.1.9 (F1): this fixture used to be 6 facts -> 5, with no archive
    sidecar touched — the exact false-positive shape hostile317-c F1 found
    live in production (facts.prune_facts moving evictions to the archive,
    or dedup merging duplicates, both shrink the ACTIVE count the same way
    and neither is a loss). That shape is now covered — without regressing
    — by test_backup_v319.py's real-writer tests. This test keeps its own
    name and structure (publish-but-don't-prune-and-alert) but drives it
    with a fixture that IS unrecoverable loss under the new rule: the
    facts union going to zero, which backup._census_regressions still
    flags (see its docstring)."""
    print("\n[test] F2: memory shrank → archive published, prune skipped, alert sent")
    # No padding on the facts themselves: webui.db's own page allocation
    # dominates the archive's total payload either way, so zeroing 6 tiny
    # fact entries stays comfortably above the payload-collapse guard's 50%
    # floor — this test is about the CENSUS catching a loss the payload
    # guard cannot see, so the fixture must not also trip THAT guard first.
    _seed_sources(n_facts=6, pad=0)
    _clean_backups()
    first = backup.run_once()
    assert_true(first["ok"], "baseline ok")

    import time
    time.sleep(1.05)
    # Every fact gone — not "one fewer", which normal eviction/dedup can
    # produce with no loss at all (v3.1.9 F1) — which the census exists to
    # catch even though the payload guard (checked above) cannot.
    _seed_sources(n_facts=0, pad=0)
    with _CapturedAlerts() as alerts:
        rep = backup.run_once()

    assert_true(rep["ok"], "the archive still publishes — it is real data")
    assert_true(rep["archive"] is not None, "published")
    assert_eq(rep["pruned"], [], "nothing pruned on a cycle that saw a loss")
    assert_true("conv1.facts 6->0" in rep["detail"], "detail names the loss")
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
        test_backup_without_db_refuses_by_default,
        test_backup_without_db_still_succeeds_with_the_escape_hatch,
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
        test_snapshot_to_data_releases_the_live_lock_before_touching_data,
        test_snapshot_to_data_stages_locally_first,
        test_snapshot_to_data_falls_back_when_local_disk_is_full,
        test_backup_once_survives_a_stalling_data_with_a_live_writer,
        test_payload_collapse_is_refused_and_does_not_prune,
        test_census_regression_publishes_but_refuses_to_prune,
        test_prune_keeps_everything_inside_the_age_window,
        test_prune_age_and_gfs_tiers,
        test_prune_never_goes_below_the_floor,
        test_ten_restarts_do_not_erase_the_oldest_archive,
        test_boot_run_suppressed_when_a_recent_archive_exists,
        test_restore_requires_confirm,
        test_restore_refuses_unverifiable_archive,
        test_restore_moves_sqlite_sidecars_aside,
        test_restore_targets_the_live_database_not_data_dir,
        # v3.1.9, hostile pass #2 on restore_backup
        test_failed_db_copy_keeps_the_original_journal,
        test_failed_db_replace_puts_the_journal_back,
        test_failed_store_copy_leaves_the_live_store_whole,
        test_successful_store_restore_keeps_the_old_store,
        test_back_to_back_set_asides_do_not_overwrite_each_other,
        test_restore_refuses_up_front_when_there_is_no_room,
        test_restore_fsyncs_the_staged_database_before_the_rename,
        test_live_database_follows_the_gate_not_existence,
        test_latest_backup_info_shape,
    ]


def test_restore_moves_sqlite_sidecars_aside():
    """A stale -journal beside the target must not survive the restore.

    SQLite derives the journal path from the database path it is GIVEN, so a
    journal left next to the restored file belongs to the database that was
    just replaced - and SQLite applies it on the next open. That is exactly
    what scripts/recover-webui-db.py and OPERATIONS.md warn about, and
    webuidb._set_aside already sweeps these three suffixes. restore_backup was
    the sibling that did not: it did a bare copy2 over the live path.

    RENAMED, NEVER DELETED - a hot journal may hold the only copy of anything
    written since the last commit, and this path runs when an operator is
    already recovering from something.
    """
    print("")
    print("[test] restore: stale sqlite sidecars are moved aside, not left")
    _seed_sources(facts_text="sidecar case")
    _clean_backups()
    rep = backup.run_once()
    arch = _BACKUPS / rep["archive"]

    # A journal from the database about to be REPLACED, carrying the real
    # magic so nothing can pass by treating it as an ordinary file.
    hot = bytes.fromhex("d9d505f920a163d7")
    journal = _DB.with_name(_DB.name + "-journal")
    journal.write_bytes(hot + b"stale rollback")
    wal = _DB.with_name(_DB.name + "-wal")
    wal.write_bytes(b"stale wal")

    res = backup.restore_backup(arch, confirm=True)
    assert_true(res["ok"], "restore ok")
    assert_true(not journal.exists(),
                "the -journal is gone from beside the restored database")
    assert_true(not wal.exists(), "and so is the -wal")

    assert_eq(sorted(_DATA.glob("webui.db-journal.pre-restore-*")), [],
              "v3.1.9 (A3-10): not left beside the target any more")
    aside = sorted(_QUARANTINE.glob("webui.db-journal.pre-restore-*"))
    assert_true(len(aside) == 1,
                "it was RENAMED rather than deleted (found %d)" % len(aside))
    assert_eq(aside[0].read_bytes()[:8], hot,
              "and its contents are intact - a hot journal may be the only "
              "copy of what was written since the last commit")

    # CONTROL: the restore still did its job. Without this the test passes
    # perfectly if restore_backup simply stopped restoring anything.
    con = sqlite3.connect(str(_DB))
    n = con.execute("SELECT COUNT(*) FROM chat").fetchone()[0]
    con.close()
    assert_true(n >= 1, "and the database itself was restored and is readable")
    assert_eq(list(_DATA.glob("webui.db.restore-*")), [],
              "no temp file left behind by the atomic replace")


def test_restore_targets_the_live_database_not_data_dir():
    """restore_backup honours an explicit `webui_db=`.

    This docstring used to say "WEBUI_DB follows the local-disk gate". It did
    not — it followed Path.exists(), at import — and that sentence is what let
    the default branch go untested: this test passes `webui_db=` explicitly, so
    it never reaches the resolution it is named after, and a mutation replacing
    that resolution with a bare path passed it (MX1, hostile pass #2). It
    proves the PARAMETER works and nothing more. The default is proved by
    test_live_database_follows_the_gate_not_existence, which unsets the
    suite-wide COMPACTOR_BACKUP_WEBUI_DB pin so it can actually fail.
    """
    print("")
    print("[test] restore: lands on the live database, not DATA_DIR/webui.db")
    _seed_sources(facts_text="target case")
    _clean_backups()
    rep = backup.run_once()
    arch = _BACKUPS / rep["archive"]

    elsewhere = _TMP / "live" / "webui.db"
    elsewhere.parent.mkdir(parents=True, exist_ok=True)
    _DB.unlink()

    res = backup.restore_backup(arch, confirm=True, webui_db=elsewhere)
    assert_true(res["ok"], "restore ok")
    assert_true(elsewhere.is_file(),
                "the archive landed on the path it was told was live")
    assert_true(not _DB.exists(),
                "and NOT on DATA_DIR/webui.db, which nothing is reading")
    con = sqlite3.connect(str(elsewhere))
    n = con.execute("SELECT COUNT(*) FROM chat").fetchone()[0]
    con.close()
    assert_true(n >= 1, "and it is a real database, not an empty file")


# ---------------------------------------------------------------------------
# v3.1.9 — the second hostile pass on restore_backup. Every defect below was
# DEMONSTRATED against the code 0123135 shipped, and 0123135 was itself the
# fix for this function. Each test has a control, and _clear_asides runs first
# in each, because set-asides now accumulate by design and a test that finds
# the previous test's file passes for that test's reason.
# ---------------------------------------------------------------------------

_HOT = bytes.fromhex("d9d505f920a163d7")


def _clear_asides():
    for root in (_DATA, _DATA.parent, _TMP / "live", _TMP / "local", _TMP / "snap",
                 _QUARANTINE):
        if not root.exists():
            continue
        for p in list(root.iterdir()):
            if any(t in p.name for t in (".pre-restore-", ".incoming-", ".restore-")):
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    p.unlink(missing_ok=True)
    for suffix in backup.SIDECARS:
        _DB.with_name(_DB.name + suffix).unlink(missing_ok=True)


def _fresh_archive(facts_text):
    _clear_asides()
    _seed_sources(facts_text=facts_text)
    _clean_backups()
    rep = backup.run_once()
    return _BACKUPS / rep["archive"]


def _store_text():
    p = _STORE / "facts" / "conv1.json"
    return p.read_text(encoding="utf-8") if p.exists() else None


def test_failed_db_copy_keeps_the_original_journal():
    """A3-2. A restore that FAILS must leave the original database's hot
    journal where SQLite will find it.

    The sidecars were renamed aside BEFORE the copy that could fail, so an
    ENOSPC left the database byte-for-byte intact and its journal gone: the one
    file that can roll back a half-applied transaction, moved away at the
    moment an operator is already recovering. SQLite then opens the database
    silently, because it can see nothing to replay.
    """
    from unittest.mock import patch
    print("")
    print("[test] restore: a FAILED database copy leaves the hot journal in place")
    arch = _fresh_archive("failed-copy case")
    journal = _DB.with_name(_DB.name + "-journal")
    journal.write_bytes(_HOT + b"the only copy of an uncommitted write")
    before = _DB.read_bytes()

    real_copy2 = shutil.copy2

    def _enospc(src, dst, *a, **k):
        if ".restore-" in str(dst):
            raise OSError(28, "No space left on device")
        return real_copy2(src, dst, *a, **k)

    with patch.object(backup.shutil, "copy2", _enospc):
        assert_raises(lambda: backup.restore_backup(arch, confirm=True), OSError,
                      "the restore failed and said so")
    assert_true(journal.exists() and journal.read_bytes()[:8] == _HOT,
                "the hot journal is STILL beside the database, under the name "
                "SQLite looks for — nothing was set aside for a copy that "
                "never happened")
    assert_eq(sorted(_DATA.glob("webui.db-journal.pre-restore-*")), [],
              "and no set-aside was made")
    assert_eq(_DB.read_bytes(), before, "the database is byte-for-byte untouched")
    assert_eq(sorted(_DATA.glob("webui.db.restore-*")), [],
              "and the staging copy was cleaned up")


def test_failed_db_replace_puts_the_journal_back():
    """A3-2, the narrower window. Once staging succeeds the sidecars are moved,
    then os.replace runs. If THAT fails, the sidecars must go back — there is
    no path through restore that leaves the original without its journal."""
    from unittest.mock import patch
    print("")
    print("[test] restore: a failed replace puts the moved journal BACK")
    arch = _fresh_archive("failed-replace case")
    journal = _DB.with_name(_DB.name + "-journal")
    journal.write_bytes(_HOT + b"uncommitted")
    real_replace = os.replace

    def _refuse_db(src, dst):
        if Path(dst) == _DB:
            raise OSError(5, "Input/output error")
        return real_replace(src, dst)

    with patch.object(backup.os, "replace", _refuse_db):
        assert_raises(lambda: backup.restore_backup(arch, confirm=True), OSError,
                      "the replace failed and the restore said so")
    assert_true(journal.exists() and journal.read_bytes()[:8] == _HOT,
                "the journal was moved aside and then PUT BACK")
    assert_eq(sorted(_DATA.glob("webui.db-journal.pre-restore-*")), [],
              "so no set-aside remains")
    # And the store was never reached: its staged copy is gone, the live one
    # untouched.
    assert_eq(sorted(_DATA.glob("compactor.incoming-*")), [],
              "the store's staged copy was cleaned up too")
    assert_true("failed-replace case" in (_store_text() or ""),
                "and the live store was never touched")


def test_failed_store_copy_leaves_the_live_store_whole():
    """A3-3. The store half was rmtree(sroot) then copytree: every fact, every
    summary tier and ChromaDB deleted before the first byte came back, with no
    set-aside — ten lines below the database half 0123135 had just made
    atomic."""
    from unittest.mock import patch
    print("")
    print("[test] restore: a FAILED store copy leaves the live store whole")
    arch = _fresh_archive("the ARCHIVED store")
    _seed_sources(facts_text="the LIVE store, written after the archive")

    def _enospc_tree(src, dst, *a, **k):
        Path(dst).mkdir(parents=True, exist_ok=True)
        (Path(dst) / "half-written.json").write_text("{", encoding="utf-8")
        raise OSError(28, "No space left on device")

    with patch.object(backup.shutil, "copytree", _enospc_tree):
        assert_raises(lambda: backup.restore_backup(arch, confirm=True), OSError,
                      "the restore failed and said so")
    assert_true("the LIVE store" in (_store_text() or ""),
                "the live store is WHOLE — rmtree did not run ahead of a copy "
                "that failed")
    assert_eq(sorted(_DATA.glob("compactor.incoming-*")), [],
              "the half-written staging copy was removed")
    assert_eq(sorted(_DATA.glob("compactor.pre-restore-*")), [],
              "and nothing was set aside, because nothing was replaced")


def test_successful_store_restore_keeps_the_old_store():
    """A3-3's CONTROL, and a property of its own: the store in place before a
    restore is SET ASIDE, never deleted. It is the only copy of everything
    written since the archive, and an operator who restored the wrong archive
    finds that out after the restore, not before."""
    print("")
    print("[test] restore: the replaced store is kept, not deleted")
    arch = _fresh_archive("the ARCHIVED store")
    _seed_sources(facts_text="the LIVE store, written after the archive")
    res = backup.restore_backup(arch, confirm=True)
    assert_true(res["ok"] and "compactor" in res["restored"], "restore ok")
    assert_true("the ARCHIVED store" in (_store_text() or ""),
                "CONTROL: the archive's store is what is live now")
    assert_eq(sorted(_DATA.glob("compactor.pre-restore-*")), [],
              "v3.1.9 (A3-10): not left beside the target any more — the "
              "overlay a pod recreate destroys")
    asides = sorted(_QUARANTINE.glob("compactor.pre-restore-*"))
    assert_eq(len(asides), 1, "the previous store was set aside in quarantine")
    kept = (asides[0] / "facts" / "conv1.json").read_text(encoding="utf-8")
    assert_true("the LIVE store" in kept,
                "and it holds what was live before the restore")


def test_back_to_back_set_asides_do_not_overwrite_each_other():
    """A3-5. The stamp was whole seconds, and on POSIX a rename onto an
    existing name silently replaces it — so two restores in one second left
    ONE journal set-aside, the second, and the first (older, more likely to
    hold what the operator is chasing) was gone without a log line.
    webuidb._stamp() carries a comment saying this already happened there.

    The stamp is pinned to a constant here on purpose: that FORCES the
    collision, so this tests the no-overwrite rule itself rather than whether
    two calls happened to land in different milliseconds.
    """
    from unittest.mock import patch
    print("")
    print("[test] restore: two set-asides with the SAME stamp are both kept")
    arch = _fresh_archive("collision case")
    journal = _DB.with_name(_DB.name + "-journal")
    with patch.object(backup, "_restore_stamp", lambda: "SAME-STAMP"):
        journal.write_bytes(_HOT + b"FIRST journal")
        backup.restore_backup(arch, confirm=True)
        # Remove the first restore's STORE set-aside, and only that. Without
        # this the second restore's store set-aside collides on a non-empty
        # DIRECTORY, which os.replace refuses loudly (ENOTEMPTY) — so a mutant
        # that stopped avoiding collisions went red on that crash and never
        # reached the journal assertion below. The journal is a FILE, and a
        # rename onto an existing file is the SILENT replace this test exists
        # for. The store half being loud is fine; this isolates the quiet one.
        # v3.1.9 (A3-10): set-asides land in quarantine now, not beside _DATA.
        for _d in _QUARANTINE.glob("compactor.pre-restore-*"):
            shutil.rmtree(_d)
        journal.write_bytes(_HOT + b"SECOND journal")
        backup.restore_backup(arch, confirm=True)
    kept = sorted(_QUARANTINE.glob("webui.db-journal.pre-restore-*"))
    bodies = {p.read_bytes()[8:] for p in kept}
    assert_eq(len(kept), 2, "two restores, two set-aside files")
    assert_true(b"FIRST journal" in bodies and b"SECOND journal" in bodies,
                "and BOTH journals survive — the first was not silently "
                "replaced by the second")
    stamp = backup._restore_stamp()
    assert_true(len(stamp) == 19 and stamp[15] == "-" and stamp[16:].isdigit(),
                "the unpatched stamp carries milliseconds (got %r)" % stamp)


def test_restore_refuses_up_front_when_there_is_no_room():
    """A3-12. create_backup has always had a free-space guard; restore, the
    destructive half, had none, and ENOSPC mid-copy was the named trigger for
    both losses above. Staging no longer touches anything live, so running out
    of space is merely a failed restore now — this makes it a clear one,
    before a large copy, with nothing moved."""
    from collections import namedtuple
    from unittest.mock import patch
    print("")
    print("[test] restore: refuses before staging when the volume is full")
    arch = _fresh_archive("no-room case")
    journal = _DB.with_name(_DB.name + "-journal")
    journal.write_bytes(_HOT + b"uncommitted")
    _Usage = namedtuple("_Usage", "total used free")
    with patch.object(backup.shutil, "disk_usage", lambda p: _Usage(10, 10, 1)):
        try:
            backup.restore_backup(arch, confirm=True)
            raised = None
        except RuntimeError as e:
            raised = e
    assert_true(raised is not None and "refusing to restore" in str(raised),
                "it refused, by name (got %r)" % (raised,))
    assert_true(journal.exists(), "the journal was not touched")
    assert_true("no-room case" in (_store_text() or ""),
                "the store was not touched")
    assert_eq(sorted(_DATA.glob("*.incoming-*")) + sorted(_DATA.glob("*.restore-*")),
              [], "and nothing was even staged")


def test_restore_fsyncs_the_staged_database_before_the_rename():
    """A3-7. The atomic replace was never fsynced, while
    memory.atomic_write_json next door fsyncs the file AND its directory and
    names /data being MooseFS as the reason. A rename is durable only if the
    bytes it points at are."""
    from unittest.mock import patch
    print("")
    print("[test] restore: the staged database is fsynced BEFORE it is renamed")
    arch = _fresh_archive("fsync case")
    events: list[tuple[str, str]] = []
    real_fsync_file, real_replace = backup._fsync_file, os.replace

    def _rec_fsync(p):
        events.append(("fsync", Path(p).name))
        return real_fsync_file(p)

    def _rec_replace(src, dst):
        events.append(("replace", Path(src).name))
        return real_replace(src, dst)

    with patch.object(backup, "_fsync_file", _rec_fsync), \
         patch.object(backup.os, "replace", _rec_replace):
        backup.restore_backup(arch, confirm=True)
    staged = [n for k, n in events if k == "replace" and n.startswith("webui.db.restore-")]
    assert_eq(len(staged), 1, "the staged database was renamed into place once")
    i_sync = events.index(("fsync", staged[0])) if ("fsync", staged[0]) in events else -1
    i_repl = events.index(("replace", staged[0]))
    assert_true(0 <= i_sync < i_repl,
                "and it was fsynced BEFORE that rename (events: %r)" % events)


def test_live_database_follows_the_gate_not_existence():
    """A3-4 and A3-9. The live database was chosen by Path.exists(), at import.

    entrypoint.sh's documented rollback, WEBUI_DB_LOCAL=false, points OpenWebUI
    at the snapshot and leaves the local file exactly where it was — so it
    still exists, every nightly archive became a copy of the abandoned file,
    and every CLI restore landed on it. The string WEBUI_DB_LOCAL did not
    appear in backup.py. And a gate flipped under the running daemon was
    invisible until restart, because the answer was computed once at import.

    This suite pins COMPACTOR_BACKUP_WEBUI_DB for every other test, which
    makes the default branch unreachable — the reason MX1 (replace the whole
    resolution with a bare path) passed it. It is UNSET here, or this test
    cannot fail.
    """
    print("")
    print("[test] live database: the gate decides, at call time, not exists()")
    local = _TMP / "local" / "webui.db"
    snap = _TMP / "snap" / "webui.db"
    for p, body in ((local, "STALE - abandoned when the flag flipped"),
                    (snap, "LIVE - every conversation since")):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.unlink(missing_ok=True)
        con = sqlite3.connect(str(p))
        con.execute("CREATE TABLE chat (id INTEGER PRIMARY KEY, body TEXT)")
        con.execute("INSERT INTO chat (body) VALUES (?)", (body,))
        con.commit()
        con.close()
    keys = ("COMPACTOR_BACKUP_WEBUI_DB", "DATABASE_URL", "WEBUI_DB_LOCAL",
            "WEBUI_LOCAL_DB", "WEBUI_SNAPSHOT_DB")
    saved = {k: os.environ.get(k) for k in keys}
    # v3.1.9 round 2: backup.live_webui_db() now DELEGATES its gate-fallback
    # branch to webuidb.live_webui_db() (finding 3, "two live-db
    # resolvers"), which returns webuidb.LOCAL_DB / webuidb.SNAPSHOT_DB —
    # MODULE CONSTANTS read once at webuidb.py's own import, not
    # WEBUI_LOCAL_DB/WEBUI_SNAPSHOT_DB re-read fresh from the environment
    # the way this test used to assume (and the way the OLD, pre-fix
    # backup.live_webui_db() genuinely did). Setting those two env vars
    # below no longer has any effect on an ALREADY-imported webuidb module
    # (almost certainly already imported by an earlier test in this file via
    # backup.py's own lazy `import webuidb`) — so this test now patches
    # webuidb.LOCAL_DB/.SNAPSHOT_DB directly for its duration instead,
    # exactly the same way test_backup_v319.py's own agreement test does.
    # The env vars are still set too (harmless, and correct for anything
    # that reads them directly rather than through webuidb's constants).
    import webuidb
    saved_webuidb = (webuidb.LOCAL_DB, webuidb.SNAPSHOT_DB)
    webuidb.LOCAL_DB, webuidb.SNAPSHOT_DB = local, snap
    try:
        os.environ.pop("COMPACTOR_BACKUP_WEBUI_DB", None)
        os.environ.pop("DATABASE_URL", None)
        os.environ["WEBUI_LOCAL_DB"] = str(local)
        os.environ["WEBUI_SNAPSHOT_DB"] = str(snap)

        os.environ["WEBUI_DB_LOCAL"] = "false"
        assert_true(local.exists(), "fixture: the abandoned local file EXISTS")
        assert_eq(backup.live_webui_db(), snap,
                  "gate false -> the snapshot, even with a local file present")

        # A3-9: same process, gate flipped, answer follows.
        os.environ["WEBUI_DB_LOCAL"] = "true"
        assert_eq(backup.live_webui_db(), local,
                  "gate flipped to true in the SAME process -> local; the "
                  "answer is not frozen at import")

        # v3.1.9 round 2: this USED to pin the byte-exact shell comparison
        # deliberately ("`True` is not `true` to entrypoint.sh"). That
        # justification no longer holds: entrypoint.sh's own WEBUI_DB_LOCAL
        # NORMALIZATION block (fix-webuidb.md finding 3) now folds case and
        # whitespace before any SUPERVISED child ever sees the variable, and
        # this function now delegates its own gate reading to
        # webuidb.live_webui_db() (see this function's docstring) instead of
        # re-deriving the rule a second, narrower way — so `True` now folds
        # to true here too, exactly as entrypoint.sh's own normaliser and
        # webuidb.live_webui_db() already agree it should.
        os.environ["WEBUI_DB_LOCAL"] = "True"
        assert_eq(backup.live_webui_db(), local,
                  "`True` folds to true now - no longer a second, "
                  "disagreeing reader of the same rule (v3.1.9 round 2)")

        # DATABASE_URL is what OpenWebUI actually opens, and outranks the gate.
        os.environ["WEBUI_DB_LOCAL"] = "true"
        os.environ["DATABASE_URL"] = f"sqlite:///{snap}"
        assert_eq(backup.live_webui_db(), snap,
                  "a sqlite DATABASE_URL outranks the gate")
        os.environ["DATABASE_URL"] = "postgresql://u@h/db"
        assert_eq(backup.live_webui_db(), local,
                  "a non-sqlite DATABASE_URL is not a file, and falls to the gate")
        os.environ.pop("DATABASE_URL", None)

        # End to end: gate false -> the archive holds the LIVE database, and a
        # restore lands on it.
        os.environ["WEBUI_DB_LOCAL"] = "false"
        _clear_asides()
        _seed_sources(with_db=False, facts_text="gate case")
        _clean_backups()
        arch = _BACKUPS / backup.run_once()["archive"]
        with tarfile.open(arch, "r:gz") as tar:
            # By basename: archives are written with arcname=".", so the
            # member is "./webui.db". The first draft asked for "webui.db"
            # and died on a KeyError that said nothing about the database.
            dbs = [m for m in tar.getmembers()
                   if Path(m.name).name == "webui.db" and m.isfile()]
            assert_eq(len(dbs), 1,
                      "the archive CONTAINS a webui.db — not a memory-only "
                      "backup of a path that did not exist")
            got = _TMP / "extracted.db"
            got.write_bytes(tar.extractfile(dbs[0]).read())
        con = sqlite3.connect(str(got))
        archived = con.execute("SELECT body FROM chat").fetchone()[0]
        con.close()
        assert_true(archived.startswith("LIVE"),
                    "the archive holds the database OpenWebUI reads, not the "
                    "abandoned one (got %r)" % archived)

        con = sqlite3.connect(str(snap))
        con.execute("UPDATE chat SET body = 'CHANGED AFTER THE BACKUP'")
        con.commit()
        con.close()
        backup.restore_backup(arch, confirm=True)
        con = sqlite3.connect(str(snap))
        back = con.execute("SELECT body FROM chat").fetchone()[0]
        con.close()
        assert_true(back.startswith("LIVE"),
                    "and the restore landed on that same file (got %r)" % back)
        con = sqlite3.connect(str(local))
        untouched = con.execute("SELECT body FROM chat").fetchone()[0]
        con.close()
        assert_true(untouched.startswith("STALE"),
                    "CONTROL: the abandoned local file was never written")
    finally:
        webuidb.LOCAL_DB, webuidb.SNAPSHOT_DB = saved_webuidb
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        _clear_asides()


if __name__ == "__main__":
    try:
        for t in _all():
            t()
        print("\nAll backup smoke tests passed.")
    finally:
        shutil.rmtree(_TMP, ignore_errors=True)
