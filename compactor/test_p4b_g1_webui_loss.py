"""
p4-b (hostile pass #4, reviewer B) G1 — backups kept pruning when webui.db was
PRESENT but her chat history was gone. Neither the prune gate (payload_ratio,
dominated by the compactor store) nor the census (store-only) had any signal
for webui.db's own CONTENT.

Two production-reachable shapes read `ok` and pruned behind them before this
fix:
  1. the database replaced by a fresh, empty OpenWebUI-shaped schema (what
     OpenWebUI builds when webui.db goes missing on a WEBUI_DB_LOCAL=false
     pod, e.g. a kill mid-restore, G3's window);
  2. her one big conversation row gutted to `{"messages": []}` in place, no
     VACUUM — the file size does not move and Connection.backup() copies the
     same free pages either way, so payload_ratio read 1.0.

Ported from SP\\p4-b\\emptydb.py, with normal-operation CONTROLs added (a
growing chat, a real delete, a VACUUM) — a rule that fires on normal
operation is hostile317-c F1 again.

Run inside the compactor image or any container with the requirements
installed:
    python test_p4b_g1_webui_loss.py
"""

import json
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

_ROOT = Path(tempfile.mkdtemp(prefix="zions-p4b-g1-test-"))
_DATA = _ROOT / "openwebui"
_STORE = _DATA / "compactor"
_BK = _ROOT / "backups"

os.environ["DATA_DIR"] = str(_DATA)
os.environ["COMPACTOR_STORAGE_ROOT"] = str(_STORE)
os.environ["COMPACTOR_BACKUP_DIR"] = str(_BK)
os.environ["COMPACTOR_BACKUP_WEBUI_DB"] = str(_DATA / "webui.db")
os.environ["COMPACTOR_BACKUP_MIN_FREE_MB"] = "1"
os.environ["COMPACTOR_BACKUP_RETAIN"] = "3"
os.environ.pop("COMPACTOR_BACKUP_ALLOW_NO_WEBUI_DB", None)

import backup  # noqa: E402

CID = "0123456789abcdef0123456789abcdef"
DB = _DATA / "webui.db"


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


def _owui_schema(p, chats):
    c = sqlite3.connect(str(p))
    c.execute(
        "create table chat (id text primary key, user_id text, title text, "
        "chat text, updated_at integer)"
    )
    c.executemany("insert into chat values (?,?,?,?,?)", chats)
    c.commit()
    c.close()


def _build():
    shutil.rmtree(_ROOT, ignore_errors=True)
    _DATA.mkdir(parents=True)
    big = json.dumps({"messages": [{"role": "user", "content": "m" * 1500}] * 2000})
    chats = [("conv", "u", "her chat", big, 2000)] + [
        (f"c{i}", "u", "t", "{}", 1000) for i in range(50)
    ]
    _owui_schema(DB, chats)
    (_STORE / "facts").mkdir(parents=True)
    (_STORE / "facts" / f"{CID}.json").write_bytes(
        json.dumps({"conv_id": CID, "facts": [{"text": f"fact {i}"} for i in range(40)]}).encode()
    )
    # A real pod's store is the LARGER half of the payload once the episodic
    # index is counted (pod manifests: ~436 MB payload vs ~138 MB db) — this
    # padding keeps payload_ratio comfortably above MIN_PAYLOAD_RATIO even
    # after webui.db's content collapses, so these tests actually exercise
    # G1's NEW webui.db-content check rather than the pre-existing (and
    # already-tested) payload_ratio guard catching the same cycle first.
    (_STORE / "chromadb").mkdir()
    with open(_STORE / "chromadb" / "index.bin", "wb") as f:
        f.write(os.urandom(6_500_000))
    _BK.mkdir()


def _age_out_older_archives(n=4):
    for k in range(n):
        p = _BK / f"zions-backup-2026050{k}-000000.tar.gz"
        shutil.copy2(backup.list_backups(_BK)[0]["path"], p)
        t = time.time() - (100 + k) * 86400
        os.utime(p, (t, t))


def test_db_replaced_by_empty_schema_holds_the_prune():
    print("\n[test] G1 case 1: webui.db replaced by an empty OpenWebUI-shaped schema")
    _build()
    time.sleep(1.1)
    baseline = backup.run_once(_BK)
    assert_true(baseline["ok"], f"PRECONDITION: baseline cycle ok (got {baseline})")
    _age_out_older_archives()
    before_count = len(backup.list_backups(_BK))

    DB.unlink()
    _owui_schema(DB, [])  # OpenWebUI's own first-boot schema: present, empty
    time.sleep(1.1)
    r = backup.run_once(_BK)

    assert_true(r["ok"], f"G1 fix: still publishes (memory is real data) (got {r})")
    assert_eq(r.get("pruned"), [], f"G1 fix: the prune is HELD (got {r.get('pruned')})")
    assert_true(bool(r.get("census_regressions")),
                f"G1 fix: census_regressions names the loss (got {r.get('census_regressions')})")
    assert_true(any("webui.db.chats" in x for x in r["census_regressions"]),
                f"and specifically names webui.db.chats (got {r['census_regressions']})")
    assert_eq(len(backup.list_backups(_BK)), before_count + 1,
              "G1 fix: the new (empty) archive published, but nothing pruned behind it")


def test_row_gutted_in_place_holds_the_prune():
    print("\n[test] G1 case 2: her 3 MB conversation row gutted in place, no VACUUM")
    _build()
    time.sleep(1.1)
    baseline = backup.run_once(_BK)
    assert_true(baseline["ok"], f"PRECONDITION: baseline cycle ok (got {baseline})")
    _age_out_older_archives()
    before_count = len(backup.list_backups(_BK))
    before_size = DB.stat().st_size

    c = sqlite3.connect(str(DB))
    c.execute("update chat set chat=? where id='conv'", (json.dumps({"messages": []}),))
    c.commit()
    c.close()
    assert_eq(DB.stat().st_size, before_size,
              "PRECONDITION: gutting in place did not change the file size (no VACUUM)")

    time.sleep(1.1)
    r = backup.run_once(_BK)

    assert_true(r["ok"], f"G1 fix: still publishes (got {r})")
    assert_eq(r.get("pruned"), [], f"G1 fix: the prune is HELD (got {r.get('pruned')})")
    assert_true(bool(r.get("census_regressions")),
                f"G1 fix: census_regressions names the loss (got {r.get('census_regressions')})")
    assert_true(
        any("webui.db.content_bytes" in x or "webui.db.row[" in x for x in r["census_regressions"]),
        f"and specifically names the content/row loss, not just the count (got {r['census_regressions']})",
    )
    assert_eq(len(backup.list_backups(_BK)), before_count + 1,
              "G1 fix: the new (gutted) archive published, but nothing pruned behind it")


def test_control_old_manifest_without_new_keys_does_not_false_alarm():
    print("\n[test] G1 CONTROL: an OLD manifest (no chats/content_bytes keys) is not a false baseline")
    _build()
    time.sleep(1.1)
    r0 = backup.run_once(_BK)
    assert_true(r0["ok"], f"PRECONDITION: baseline ok (got {r0})")

    # Simulate a pre-fix archive: strip the new keys from the newest
    # manifest.json, the way a real pre-fix archive would already look.
    import tarfile
    newest_path = Path(backup.list_backups(_BK)[0]["path"])
    scratch = Path(tempfile.mkdtemp())
    with tarfile.open(newest_path, "r:gz") as t:
        t.extractall(scratch, filter="data")
    man_path = scratch / "manifest.json"
    man = json.loads(man_path.read_text())
    for k in ("chats", "content_bytes", "newest_updated_at", "row_bytes"):
        man["sources"]["webui.db"].pop(k, None)
    man_path.write_text(json.dumps(man, indent=2))
    old_shaped = _BK / "zions-backup-20260401-000000.tar.gz"
    with tarfile.open(old_shaped, "w:gz") as t:
        t.add(scratch, arcname=".")
    t_old = time.time() - 200 * 86400
    os.utime(old_shaped, (t_old, t_old))
    newest_path.unlink()  # only the old-shaped archive remains as the baseline

    time.sleep(1.1)
    r = backup.run_once(_BK)
    assert_true(r["ok"], f"CONTROL: cycle over an old-shaped baseline is ok (got {r})")
    assert_eq([x for x in r.get("census_regressions", []) if x.startswith("webui.db.")], [],
              f"CONTROL: no webui.db.* false alarm against a baseline with no new keys (got {r.get('census_regressions')})")


def test_control_chat_growth_does_not_alarm():
    print("\n[test] G1 CONTROL: her chat growing (normal operation) does not alarm")
    _build()
    time.sleep(1.1)
    r0 = backup.run_once(_BK)
    assert_true(r0["ok"], f"PRECONDITION: baseline ok (got {r0})")

    c = sqlite3.connect(str(DB))
    bigger = json.dumps({"messages": [{"role": "user", "content": "m" * 1500}] * 2500})
    c.execute("update chat set chat=?, updated_at=? where id='conv'", (bigger, 3000))
    c.execute("insert into chat values (?,?,?,?,?)", ("c-new", "u", "t", "{}", 3001))
    c.commit()
    c.close()

    time.sleep(1.1)
    r = backup.run_once(_BK)
    assert_true(r["ok"], f"CONTROL: ok (got {r})")
    assert_eq([x for x in r.get("census_regressions", []) if x.startswith("webui.db.")], [],
              f"CONTROL: chat growth does not false-alarm (got {r.get('census_regressions')})")
    assert_true(bool(r.get("pruned") is not None), "CONTROL: prune ran normally (not held)")


def test_control_a_real_delete_does_not_alarm():
    print("\n[test] G1 CONTROL: deleting ONE chat in OpenWebUI (normal operation) does not alarm")
    _build()
    time.sleep(1.1)
    r0 = backup.run_once(_BK)
    assert_true(r0["ok"], f"PRECONDITION: baseline ok (got {r0})")

    c = sqlite3.connect(str(DB))
    c.execute("delete from chat where id='c0'")
    c.commit()
    c.close()

    time.sleep(1.1)
    r = backup.run_once(_BK)
    assert_true(r["ok"], f"CONTROL: ok (got {r})")
    assert_eq([x for x in r.get("census_regressions", []) if x.startswith("webui.db.")], [],
              f"CONTROL: deleting one of 51 chats does not false-alarm (got {r.get('census_regressions')})")


def test_control_vacuum_does_not_alarm():
    print("\n[test] G1 CONTROL: a VACUUM (file shrinks, content does not) does not alarm")
    _build()
    time.sleep(1.1)
    r0 = backup.run_once(_BK)
    assert_true(r0["ok"], f"PRECONDITION: baseline ok (got {r0})")

    c = sqlite3.connect(str(DB))
    c.execute("delete from chat where id='c1'")  # make room for VACUUM to reclaim
    c.commit()
    c.execute("VACUUM")
    c.close()

    time.sleep(1.1)
    r = backup.run_once(_BK)
    assert_true(r["ok"], f"CONTROL: ok (got {r})")
    # One real chat was deleted above (c1) alongside the VACUUM, so allow
    # that single, expected drop but nothing more.
    webui_losses = [x for x in r.get("census_regressions", []) if x.startswith("webui.db.")]
    assert_eq(webui_losses, [],
              f"CONTROL: a VACUUM plus one ordinary delete does not false-alarm (got {webui_losses})")


if __name__ == "__main__":
    tests = [
        test_db_replaced_by_empty_schema_holds_the_prune,
        test_row_gutted_in_place_holds_the_prune,
        test_control_old_manifest_without_new_keys_does_not_false_alarm,
        test_control_chat_growth_does_not_alarm,
        test_control_a_real_delete_does_not_alarm,
        test_control_vacuum_does_not_alarm,
    ]
    for t in tests:
        t()
    print("\nAll p4-b G1 (webui.db content loss) tests passed.")
