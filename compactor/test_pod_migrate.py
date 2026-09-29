"""
Tests for scripts/pod-migrate.py (v3.1.9.10): the old-pod -> new-pod migration tool.

SYNTHETIC DATA ONLY. Every database row, upload and fact here is written by this
file; nothing reads a real path. Assertions report counts, never names, and one test
asserts that no synthetic upload name ever reaches the tool's output.

Covers: a dry run changes nothing; seed; verify; a change in the source (new message
rows, new facts, a new upload, a deleted file) followed by cutover and a passing
verify; the LOCAL-DB REPLACEMENT case (a stale healthy local database must not
survive, and webuidb.py's own restore/sync must then agree with the fresh copy);
the refusal cases (source active, target services running, target not a seed, swapped
arguments); the source tree is byte-and-mtime identical before and after every
apply; the upload-path rewrite; a reverse cutover.

Run: python test_pod_migrate.py
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SCRIPT = _HERE.parent / "scripts" / "pod-migrate.py"
_spec = importlib.util.spec_from_file_location("pod_migrate", _SCRIPT)
pm = importlib.util.module_from_spec(_spec)
sys.modules["pod_migrate"] = pm
_spec.loader.exec_module(pm)

_TMP = Path(tempfile.mkdtemp(prefix="zions-pod-migrate-test-"))
os.environ["WEBUI_DB_LOCAL"] = "true"
pm.GUARDED_PORTS = ()  # a dev machine may have something on :3000; one test below opens its own port

FAILED = 0
HER = pm.HER_CONV_ID
PRIVATE = "zzPRIVATEphoto"  # every synthetic upload name contains this; it must never be printed


def assert_eq(a, b, label):
    global FAILED
    if a != b:
        print(f"FAIL {label}: expected {b!r}, got {a!r}")
        FAILED += 1
        return
    print(f"  ok   {label}")


def assert_true(c, label):
    global FAILED
    if not c:
        print(f"FAIL {label}")
        FAILED += 1
        return
    print(f"  ok   {label}")


# ---------------------------------------------------------------------------
# synthetic world
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE user (id TEXT PRIMARY KEY, name TEXT);
CREATE TABLE chat (id TEXT PRIMARY KEY, user_id TEXT, title TEXT, chat TEXT, updated_at INTEGER);
CREATE TABLE chat_message (id TEXT PRIMARY KEY, chat_id TEXT, role TEXT, content TEXT);
CREATE TABLE file (id TEXT PRIMARY KEY, user_id TEXT, path TEXT, meta TEXT);
CREATE TABLE config (id INTEGER PRIMARY KEY, data TEXT);
"""


def add_messages(db: Path, conv: str, first: int, count: int) -> None:
    c = sqlite3.connect(str(db))
    try:
        row = c.execute("SELECT chat FROM chat WHERE id=?", (conv,)).fetchone()
        doc = json.loads(row[0]) if row else {"history": {"messages": {}}}
        for i in range(first, first + count):
            mid = f"m{i}"
            doc["history"]["messages"][mid] = {"id": mid, "role": "user" if i % 2 == 0 else "assistant",
                                               "content": f"synthetic message {i}"}
            c.execute("INSERT OR REPLACE INTO chat_message VALUES (?,?,?,?)", (f"{conv}-{mid}", conv, "user", f"synthetic message {i}"))
        if row:
            c.execute("UPDATE chat SET chat=?, updated_at=updated_at+1 WHERE id=?", (json.dumps(doc), conv))
        else:
            c.execute("INSERT INTO chat VALUES (?,?,?,?,?)", (conv, "u1", "synthetic", json.dumps(doc), 1))
        c.commit()
    finally:
        c.close()


def add_upload(old: Path, name: str, data: bytes, db_row: bool = True) -> None:
    (old / "uploads").mkdir(parents=True, exist_ok=True)
    p = old / "uploads" / name
    p.write_bytes(data)
    if db_row:
        c = sqlite3.connect(str(old / "webui.db"))
        try:
            c.execute("INSERT INTO file VALUES (?,?,?,?)", (name.split("_")[0], "u1", str(p), "{}"))
            c.commit()
        finally:
            c.close()


def make_chroma(path: Path, docs: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(path))
    try:
        c.execute("CREATE TABLE IF NOT EXISTS embeddings (id INTEGER PRIMARY KEY, doc TEXT)")
        c.execute("CREATE TABLE IF NOT EXISTS collections (id TEXT)")
        c.execute("CREATE TABLE IF NOT EXISTS segments (id TEXT)")
        for i in range(docs):
            c.execute("INSERT INTO embeddings (doc) VALUES (?)", (f"d{i}",))
        c.commit()
    finally:
        c.close()


def build_world(tag: str) -> dict:
    """An 'old pod' tree and an (unborn) 'new pod' target, plus a local disk dir."""
    root = _TMP / tag
    old = root / "openwebui"
    new = root / "v10" / "openwebui"
    local = root / "local"
    local.mkdir(parents=True)
    old.mkdir(parents=True)
    db = old / "webui.db"
    c = sqlite3.connect(str(db))
    c.executescript(SCHEMA)
    c.execute("INSERT INTO user VALUES ('u1','synthetic user')")
    c.commit()
    c.close()
    add_messages(db, HER, 0, 20)
    add_messages(db, "other-chat", 0, 3)
    for i in range(5):
        add_upload(old, f"id{i}_{PRIVATE}{i}.png", (b"png%d" % i) * (100 + i))
    (old / "compactor" / "facts").mkdir(parents=True)
    (old / "compactor" / "summaries").mkdir(parents=True)
    (old / "compactor" / "personas").mkdir(parents=True)
    for i in range(4):
        (old / "compactor" / "facts" / f"conv{i}.json").write_text(json.dumps({"facts": [{"text": f"f{j}"} for j in range(i + 1)]}))
        (old / "compactor" / "summaries" / f"conv{i}.json").write_text(json.dumps({"L1": "s"}))
    (old / "compactor" / "personas" / "p.json").write_text("{}")
    make_chroma(old / "compactor" / "chromadb" / "chroma.sqlite3", 7)
    (old / "compactor" / "chromadb" / "seg1").mkdir()
    (old / "compactor" / "chromadb" / "seg1" / "data_level0.bin").write_bytes(b"\x01" * 512)
    (old / "compactor-control" / "wipes").mkdir(parents=True)
    (old / "compactor-control" / "wipes" / "conv9.json").write_text("{}")
    (old / "compactor-pending" / "ledger-pending").mkdir(parents=True)
    make_chroma(old / "vector_db" / "chroma.sqlite3", 3)
    (old / "cache").mkdir()
    (old / "cache" / "x").write_text("regenerable")
    (old / ".webui_secret_key").write_text("synthetic-key")
    zl = root / "zl-ops"
    (zl / "word-ban").mkdir(parents=True)
    (zl / "word-ban" / "active.json").write_text('{"rev": 5}')
    # age everything so "activity" heuristics see a quiet tree
    old_t = time.time() - 3600
    for dp, dns, fns in os.walk(old):
        for n in fns:
            os.utime(os.path.join(dp, n), (old_t, old_t))
    fake = root / "supervisorctl"
    state = root / "sup.state"
    state.write_text("openwebui STOPPED\ncompactor STOPPED\nwebuidb-sync STOPPED\nbackup STOPPED\n")
    fake.write_text(
        "#!/bin/bash\n"
        "shift; shift 2>/dev/null; [ \"$1\" = status ] && shift\n"
        f"for n in \"$@\"; do l=$(grep \"^$n \" {state}); "
        "if [ -n \"$l\" ]; then set -- $l; printf '%-32s %s   pid 1, uptime 0:00:01\\n' \"$1\" \"$2\"; "
        "else echo \"$n: ERROR (no such process)\"; fi; done\n"
    )
    fake.chmod(0o755)
    return {"root": root, "old": old, "new": new, "local": local, "zl": zl, "sup": fake, "state": state}


def tree_fingerprint(root: Path) -> dict:
    """Every entry under root: files by (size, mtime_ns, sha), dirs by (mtime_ns, entry count)."""
    out = {}
    for dp, dns, fns in os.walk(root):
        st = os.lstat(dp)
        out[os.path.relpath(dp, root) + "/"] = (st.st_mtime_ns, len(dns) + len(fns))
        for n in fns:
            p = os.path.join(dp, n)
            s = os.lstat(p)
            out[os.path.relpath(p, root)] = (s.st_size, s.st_mtime_ns, hashlib.sha256(open(p, "rb").read()).hexdigest())
    return out


def run(w: dict, cmd: str, *extra: str, supervisor: bool = True, local_db: bool = True, quiet: bool = True) -> tuple[int, str]:
    argv = [cmd, "--from", str(w["old"]), "--to", str(w["new"]), "--local-data-dir", str(w["new"]),
            "--zl-from", str(w["zl"]), "--zl-to", str(w["root"] / "v10" / "zl-ops")]
    if cmd != "verify":
        argv += ["--tmp-dir", str(w["local"] / "scratch"), "--retries", "2", "--retry-wait", "0.1"]
        argv += ["--db-mode", "local" if local_db else "snapshot", "--local-db", str(w["local"] / "webui.db")]
        if supervisor:
            argv += ["--supervisorctl", str(w["sup"]), "--supervisor-conf", "x.conf"]
        else:
            argv += ["--no-supervisor-check"]
        if cmd == "cutover" and quiet:
            argv += ["--quiet-seconds", "0"]
    else:
        argv += ["--db-mode", "local" if local_db else "snapshot", "--local-db", str(w["local"] / "webui.db")]
    argv += list(extra)
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        try:
            rc = pm.main(argv)
        except SystemExit as e:  # argparse
            rc = int(e.code or 0)
    return rc, out.getvalue() + err.getvalue()


def scalar(db: Path, sql: str, *a):
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return c.execute(sql, a).fetchone()[0]
    finally:
        c.close()


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_dry_run_changes_nothing():
    w = build_world("dry")
    before_old = tree_fingerprint(w["old"])
    rc, out = run(w, "seed")
    assert_eq(rc, 0, "seed dry run exits 0")
    assert_true("DRY RUN" in out, "the dry run says so")
    assert_true(not w["new"].exists(), "the target was not even created")
    assert_true(not (w["local"] / "webui.db").exists(), "no local database was placed")
    assert_eq(tree_fingerprint(w["old"]), before_old, "the source is byte-and-mtime identical")
    rc, out = run(w, "seed", "--apply")
    assert_eq(rc, 0, "seed --apply exits 0")
    before_new = tree_fingerprint(w["new"])
    before_local = tree_fingerprint(w["local"])
    (w["old"] / "compactor" / "facts" / "conv0.json").write_text('{"facts":[]}')
    rc, out = run(w, "cutover", "--old-pod-stopped")
    assert_eq(rc, 0, "cutover dry run exits 0")
    assert_eq(tree_fingerprint(w["new"]), before_new, "a cutover dry run changes nothing in the target")
    assert_eq(tree_fingerprint(w["local"]), before_local, "a cutover dry run changes nothing on the local disk")
    assert_true("changed=1" in out, "the plan reports the one changed file")


def test_seed_copies_everything_and_never_writes_the_source():
    w = build_world("seed")
    before_old = tree_fingerprint(w["old"])
    rc, out = run(w, "seed", "--apply")
    assert_eq(rc, 0, "seed exits 0")
    assert_eq(tree_fingerprint(w["old"]), before_old, "the source tree is unchanged (hash, mtime, directory mtimes)")
    assert_true(not (w["old"] / ".migrate").exists(), "no .migrate directory was written into the source")
    m = json.loads((w["new"] / ".migrate" / "seed.json").read_text())
    assert_true(m["complete"] and m["kind"] == "seed", "the manifest is complete")
    assert_true(len(m["files"]) >= 10, "the manifest lists the copied files")
    for rel in ("compactor/facts/conv1.json", "compactor-control/wipes/conv9.json", f"uploads/id2_{PRIVATE}2.png"):
        assert_eq(sha(w["new"] / rel), sha(w["old"] / rel), f"copied byte-for-byte: {rel.split('/')[0]}")
    assert_eq(scalar(w["new"] / "webui.db", "select count(*) from chat_message"), scalar(w["old"] / "webui.db", "select count(*) from chat_message"),
              "chat_message rows equal")
    assert_eq(scalar(w["new"] / "compactor/chromadb/chroma.sqlite3", "select count(*) from embeddings"), 7, "compactor chroma copied via the backup API")
    assert_eq(scalar(w["new"] / "vector_db/chroma.sqlite3", "select count(*) from embeddings"), 3, "openwebui vector_db copied")
    assert_eq(sha(w["local"] / "webui.db"), sha(w["new"] / "webui.db"), "the local DB equals the snapshot")
    assert_eq(scalar(w["new"] / "webui.db", "select count(*) from file where path like ?", str(w["old"]) + "/uploads/%"), 0,
              "no file.path still points at the OLD pod's uploads")
    assert_eq(scalar(w["new"] / "webui.db", "select count(*) from file where path like ?", str(w["new"]) + "/uploads/%"), 5,
              "every file.path was rewritten to the new pod's uploads")
    assert_eq(scalar(w["old"] / "webui.db", "select count(*) from file where path like ?", str(w["old"]) + "/uploads/%"), 5,
              "the SOURCE database still has its own paths")
    assert_true(not (w["new"] / ".webui_secret_key").exists(), "the secret key file is NOT copied by default")
    assert_true((w["root"] / "v10" / "zl-ops" / "word-ban" / "active.json").is_file(), "the word-ban file was copied")
    assert_true(not (w["new"] / "cache").exists(), "regenerable top-level entries are not copied")
    assert_true(PRIVATE not in out and "conv0" not in out, "no upload or conversation name reaches the output")
    rc, out = run(w, "verify")
    assert_eq(rc, 0, "verify passes right after a seed")
    assert_true(PRIVATE not in out, "verify output carries no upload names")
    assert_true("PASS" in out and "FAIL" not in out.replace("FAILED", ""), "verify prints PASS lines only")


def test_with_cache_flag():
    w = build_world("cache")
    rc, _ = run(w, "seed", "--apply")
    assert_eq(rc, 0, "seed")
    assert_true(not (w["new"] / "cache").exists(), "cache is not copied by default")
    rc, _ = run(w, "seed", "--apply", "--with-cache")
    assert_eq(rc, 0, "seed --with-cache")
    assert_eq((w["new"] / "cache" / "x").read_text(), "regenerable", "cache was copied on request")
    assert_eq(run(w, "verify", "--with-cache")[0], 0, "verify --with-cache passes")


def test_secret_key_flag():
    w = build_world("secret")
    rc, _ = run(w, "seed", "--apply", "--with-secret-key")
    assert_eq(rc, 0, "seed --with-secret-key exits 0")
    assert_eq((w["new"] / ".webui_secret_key").read_text(), "synthetic-key", "the key file was copied")
    assert_eq(oct((w["new"] / ".webui_secret_key").stat().st_mode & 0o777), "0o600", "and is 0600")


def test_source_change_then_cutover_then_verify():
    w = build_world("cut")
    assert_eq(run(w, "seed", "--apply")[0], 0, "seed")
    assert_eq(run(w, "verify")[0], 0, "verify passes after the seed")
    # the old pod keeps living: new rows, new facts, a new upload, a deleted file, a changed file
    add_messages(w["old"] / "webui.db", HER, 20, 6)
    (w["old"] / "compactor" / "facts" / "conv-new.json").write_text(json.dumps({"facts": [{"text": "n1"}, {"text": "n2"}]}))
    (w["old"] / "compactor" / "facts" / "conv1.json").write_text(json.dumps({"facts": [{"text": "changed"}]}))
    add_upload(w["old"], f"id9_{PRIVATE}9.png", b"new-upload" * 50)
    gone = w["old"] / "uploads" / f"id0_{PRIVATE}0.png"
    gone_sha = sha(gone)
    gone.unlink()
    c = sqlite3.connect(str(w["old"] / "webui.db"))
    c.execute("DELETE FROM file WHERE id='id0'")
    c.commit()
    c.close()
    (w["old"] / "compactor" / "summaries" / "conv3.json").unlink()
    make_chroma(w["old"] / "vector_db" / "chroma.sqlite3", 5)
    rc, out = run(w, "verify")
    assert_eq(rc, 1, "verify FAILS while the source has moved on")
    before_old = tree_fingerprint(w["old"])
    rc, out = run(w, "cutover", "--old-pod-stopped", "--apply")
    assert_eq(rc, 0, "cutover exits 0 (it runs verify itself)")
    assert_eq(tree_fingerprint(w["old"]), before_old, "cutover left the source byte-and-mtime identical")
    assert_true(PRIVATE not in out, "cutover output carries no upload names")
    assert_eq(run(w, "verify")[0], 0, "verify passes after the cutover")
    assert_eq(scalar(w["new"] / "webui.db", "select count(*) from chat_message where chat_id=?", HER), 26, "her new messages arrived")
    assert_true((w["new"] / "compactor/facts/conv-new.json").is_file(), "a new facts file arrived")
    assert_eq(json.loads((w["new"] / "compactor/facts/conv1.json").read_text())["facts"][0]["text"], "changed", "a changed file was replaced")
    assert_true((w["new"] / f"uploads/id9_{PRIVATE}9.png").is_file(), "the new upload arrived")
    assert_true(not (w["new"] / f"uploads/id0_{PRIVATE}0.png").exists(), "the file deleted in the source is gone from the target ...")
    removed = list((w["new"] / ".migrate").glob("removed-*"))
    assert_eq(len(removed), 1, "... into a removed-<stamp> directory")
    moved = [p for p in removed[0].rglob("*") if p.is_file() and sha(p) == gone_sha]
    assert_eq(len(moved), 1, "the moved file is intact (moved, not deleted)")
    assert_eq(scalar(w["new"] / "vector_db/chroma.sqlite3", "select count(*) from embeddings"), 8, "the changed vector_db was refreshed")
    m = json.loads((w["new"] / ".migrate" / "seed.json").read_text())
    assert_eq([r["cmd"] for r in m["runs"]], ["seed", "cutover"], "the manifest records both runs")
    rc, out = run(w, "seed", "--apply", "--json")
    assert_eq(rc, 0, "a re-seed after a cutover is a no-op refresh")
    plan = json.loads(out.strip().splitlines()[-1])["plan"]
    assert_eq((plan["new"], plan["changed"], plan["deleted_in_source"]), (0, 0, 0), "nothing left to copy (stat-trusted)")


def test_local_db_replacement():
    """THE dangerous step: a stale but healthy local DB on the new pod must not win."""
    w = build_world("localdb")
    assert_eq(run(w, "seed", "--apply")[0], 0, "seed")
    local = w["local"] / "webui.db"
    # the new pod was started for testing and wrote something: the local file is now
    # NEWER than the snapshot and holds a chat the source does not have
    add_messages(local, "test-only-chat", 0, 2)
    (w["local"] / "webui.db-journal").write_bytes(b"")  # a leftover journal beside it
    (w["local"] / ".empty-start").write_text("x")
    time.sleep(0.05)
    os.utime(local, None)
    add_messages(w["old"] / "webui.db", HER, 20, 4)
    # control: what webuidb.py would do with the stale file, untouched
    env = {**os.environ, "WEBUI_LOCAL_DB": str(local), "WEBUI_SNAPSHOT_DB": str(w["new"] / "webui.db"),
           "WEBUI_DB_QUARANTINE": str(w["root"] / "forensics"), "PYTHONPATH": str(_HERE), "DATA_DIR": str(w["new"])}
    probe = ("import json, webuidb; print(json.dumps(webuidb.restore_on_boot()))")
    r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, env=env, cwd=str(_HERE))
    assert_true('"kept_local"' in r.stdout, "CONTROL: the boot restore keeps a healthy stale local file (this is the hazard)")
    assert_eq(scalar(local, "select count(*) from chat where id='test-only-chat'"), 1, "CONTROL: the stale local DB is still the live one")
    # the fix under test
    rc, out = run(w, "cutover", "--old-pod-stopped", "--apply")
    assert_eq(rc, 0, "cutover exits 0")
    assert_eq(scalar(local, "select count(*) from chat where id='test-only-chat'"), 0, "the stale local database is GONE from the live path")
    assert_eq(sha(local), sha(w["new"] / "webui.db"), "the local DB is byte-identical to the fresh snapshot")
    assert_eq(scalar(local, "select count(*) from chat_message where chat_id=?", HER), 24, "and it holds the source's newest messages")
    aside = list(w["local"].glob(".migrate-aside-*"))
    assert_eq(len(aside), 1, "the stale local file was moved aside, not deleted")
    kept = sorted(p.name for p in aside[0].iterdir())
    assert_true("webui.db" in kept and "webui.db-journal" in kept and ".empty-start" in kept, "with its journal and empty-start marker")
    assert_true(not (w["local"] / "webui.db-journal").exists() and not (w["local"] / ".empty-start").exists(), "no stale sidecar remains")
    assert_eq(scalar(aside[0] / "webui.db", "select count(*) from chat where id='test-only-chat'"), 1, "the aside copy still has the test chat")
    st_l, st_s = local.stat().st_mtime_ns, (w["new"] / "webui.db").stat().st_mtime_ns
    assert_eq(st_l, st_s, "local and snapshot carry the same mtime (the daemon's 'snapshot is as new' skip holds)")
    # webuidb.py's own view: boot keeps local (now the fresh one), and a sync publishes nothing stale
    r = subprocess.run([sys.executable, "-c", probe + "; print(json.dumps(webuidb.sync_once()))"], capture_output=True, text=True, env=env, cwd=str(_HERE))
    assert_true('"kept_local"' in r.stdout, "the boot restore keeps the (fresh) local file")
    assert_true('"skipped": "unchanged since last sync"' in r.stdout, "the sync daemon skips: nothing to publish over the snapshot")
    assert_eq(scalar(w["new"] / "webui.db", "select count(*) from chat where id='test-only-chat'"), 0, "the snapshot never saw the test chat")
    r = subprocess.run([sys.executable, "-c", "import json, webuidb; print(json.dumps(webuidb.sync_once(force=True)))"], capture_output=True,
                       text=True, env=env, cwd=str(_HERE))
    assert_true('"error": null' in r.stdout, "even a forced publish is accepted (same content), not refused")
    assert_eq(scalar(w["new"] / "webui.db", "select count(*) from chat_message where chat_id=?", HER), 24, "and the snapshot still holds the fresh data")
    assert_eq(run(w, "verify", "--strict-local")[0], 0, "verify --strict-local passes: local equals the snapshot")


def test_first_boot_leftovers_are_accepted_as_pristine():
    w = build_world("pristine")
    # what a fresh new pod leaves behind: an empty schema, empty dirs, selftest probe files
    (w["new"] / "compactor" / "facts").mkdir(parents=True)
    (w["new"] / "compactor" / "summaries").mkdir(parents=True)
    (w["new"] / "compactor-control" / "wipes").mkdir(parents=True)
    (w["new"] / "compactor-control" / "wipes" / "__selftest_oneshot_x__.json").write_text("{}")
    (w["new"] / "compactor" / "summaries" / "__selftest_oneshot_x__.wipe.json").write_text("{}")
    make_chroma(w["new"] / "compactor" / "chromadb" / "chroma.sqlite3", 0)
    c = sqlite3.connect(str(w["new"] / "webui.db"))
    c.executescript(SCHEMA)
    c.commit()
    c.close()
    c = sqlite3.connect(str(w["local"] / "webui.db"))
    c.executescript(SCHEMA)
    c.commit()
    c.close()
    assert_eq(pm.target_state(w["new"]), "pristine", "a first-boot target classifies as pristine")
    rc, out = run(w, "seed", "--apply")
    assert_eq(rc, 0, "seeding over a first boot works without --force")
    assert_eq(run(w, "verify")[0], 0, "verify passes")
    assert_eq(scalar(w["local"] / "webui.db", "select count(*) from chat"), 2, "the empty local schema was replaced by the seeded DB")


def test_refusals():
    w = build_world("refuse")
    # target has real data and is not a seed
    (w["new"] / "compactor" / "facts").mkdir(parents=True)
    (w["new"] / "compactor" / "facts" / "real.json").write_text('{"facts":[{"text":"hers"}]}')
    before = tree_fingerprint(w["new"])
    rc, out = run(w, "seed", "--apply")
    assert_eq(rc, 2, "seed refuses a --to that holds data and is not a seed")
    assert_eq(tree_fingerprint(w["new"]), before, "and changed nothing")
    rc, out = run(w, "seed", "--apply", "--force")
    assert_eq(rc, 0, "--force allows it")
    assert_true(any((w["new"] / ".migrate").glob("removed-*/compactor/facts/real.json")), "and the unknown file was kept aside, not deleted")
    w = build_world("refuse2")
    rc, out = run(w, "cutover", "--old-pod-stopped", "--apply")
    assert_eq(rc, 2, "cutover refuses a target that was never seeded")
    assert_eq(run(w, "seed", "--apply")[0], 0, "seed")
    rc, out = run(w, "cutover", "--apply")
    assert_eq(rc, 2, "cutover refuses without --old-pod-stopped")
    assert_true("--old-pod-stopped" in out, "and says why")
    # the new pod's own services running
    w["state"].write_text("openwebui RUNNING\ncompactor STOPPED\nwebuidb-sync STOPPED\nbackup STOPPED\n")
    before = tree_fingerprint(w["new"])
    for cmd, extra in (("seed", ()), ("cutover", ("--old-pod-stopped",))):
        rc, out = run(w, cmd, "--apply", *extra)
        assert_eq(rc, 2, f"{cmd} refuses while openwebui is RUNNING on the new pod")
    assert_eq(tree_fingerprint(w["new"]), before, "and changed nothing")
    w["state"].write_text("openwebui STOPPED\ncompactor STARTING\nwebuidb-sync STOPPED\nbackup STOPPED\n")
    assert_eq(run(w, "seed", "--apply")[0], 2, "STARTING counts as running")
    w["state"].write_text("openwebui STOPPED\ncompactor STOPPED\nwebuidb-sync BACKOFF\nbackup STOPPED\n")
    assert_eq(run(w, "seed", "--apply")[0], 2, "BACKOFF counts as running")
    w["state"].write_text("openwebui STOPPED\ncompactor STOPPED\nbackup STOPPED\n")
    assert_eq(run(w, "seed", "--apply")[0], 2, "a program supervisorctl cannot report on is a refusal, not a pass")
    w["state"].write_text("openwebui STOPPED\ncompactor EXITED\nwebuidb-sync FATAL\nbackup STOPPED\n")
    assert_eq(run(w, "seed", "--apply")[0], 0, "STOPPED/EXITED/FATAL pass")
    # source active: a hot journal beside the source database
    (w["old"] / "webui.db-journal").write_bytes(b"x")
    before_new = tree_fingerprint(w["new"])
    rc, out = run(w, "cutover", "--old-pod-stopped", "--apply")
    assert_eq(rc, 2, "cutover refuses while the source has a -journal")
    assert_eq(tree_fingerprint(w["new"]), before_new, "and changed nothing")
    (w["old"] / "webui.db-journal").unlink()
    # source active: files change while the tool watches
    stop = threading.Event()

    def churn():
        i = 0
        while not stop.is_set():
            (w["old"] / "compactor" / "facts" / "churn.json").write_text(str(i))
            i += 1
            time.sleep(0.1)

    t = threading.Thread(target=churn)
    t.start()
    try:
        rc, out = run(w, "cutover", "--old-pod-stopped", "--apply", "--quiet-seconds", "1.5", "--poll-interval", "0.3", quiet=False)
    finally:
        stop.set()
        t.join()
    assert_eq(rc, 2, "cutover refuses while the source tree is changing")
    assert_true("still writing" in out, "and says the old pod is still writing")
    (w["old"] / "compactor" / "facts" / "churn.json").unlink()
    rc, out = run(w, "cutover", "--old-pod-stopped", "--apply", "--quiet-seconds", "1", "--poll-interval", "0.3", quiet=False)
    assert_eq(rc, 0, "a quiet source passes the poll")
    # wrong / swapped trees
    argv = ["seed", "--from", str(w["new"]), "--to", str(w["old"]), "--local-data-dir", str(w["new"]), "--no-supervisor-check"]
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = pm.main(argv)
    assert_eq(rc, 2, "swapped --from/--to is refused (--to is not this pod's DATA_DIR)")
    before_old = tree_fingerprint(w["old"])
    argv = ["cutover", "--from", str(w["new"]), "--to", str(w["old"]), "--old-pod-stopped", "--apply", "--no-supervisor-check",
            "--local-data-dir", str(w["new"]), "--quiet-seconds", "0"]
    with redirect_stdout(out), redirect_stderr(err):
        rc = pm.main(argv)
    assert_eq(rc, 2, "a forward cutover INTO the old tree is refused")
    assert_eq(tree_fingerprint(w["old"]), before_old, "and the old tree is untouched")
    argv = ["seed", "--from", str(w["old"]), "--to", str(w["old"] / "sub"), "--local-data-dir", str(w["old"] / "sub"), "--no-supervisor-check"]
    with redirect_stdout(out), redirect_stderr(err):
        rc = pm.main(argv)
    assert_eq(rc, 2, "a --to nested inside --from is refused")
    argv = ["seed", "--from", str(w["old"]), "--to", str(w["new"]), "--no-supervisor-check"]
    env_backup = dict(os.environ)
    for k in ("DATA_DIR",):
        os.environ.pop(k, None)
    with redirect_stdout(out), redirect_stderr(err):
        rc = pm.main(argv + ["--apply"])
    os.environ.update(env_backup)
    assert_eq(rc, 2, "with no detectable DATA_DIR on this pod, --to is refused rather than guessed")


def test_corrupt_source_installs_nothing():
    w = build_world("corrupt")
    assert_eq(run(w, "seed", "--apply")[0], 0, "seed")
    local_before = sha(w["local"] / "webui.db")
    snap_before = sha(w["new"] / "webui.db")
    (w["old"] / "webui.db").write_bytes(b"this is not a database" * 100)
    rc, out = run(w, "cutover", "--old-pod-stopped", "--apply")
    assert_eq(rc, 1, "a corrupt source database fails the cutover (exit 1)")
    assert_eq(sha(w["local"] / "webui.db"), local_before, "the local database was not replaced")
    assert_eq(sha(w["new"] / "webui.db"), snap_before, "the snapshot was not replaced")
    assert_true(not list(w["local"].glob(".migrate-aside-*")), "nothing was moved aside")
    assert_true(not (w["new"] / ".migrate" / "lock").exists(), "the lock was released")


def test_backup_api_copy_is_consistent_under_a_writer():
    w = build_world("writer")
    stop = threading.Event()
    n = [0]

    def writer():
        c = sqlite3.connect(str(w["old"] / "webui.db"), timeout=30)
        while not stop.is_set():
            c.execute("INSERT INTO chat_message VALUES (?,?,?,?)", (f"w{n[0]}", "writer-chat", "user", "x" * 200))
            c.commit()
            n[0] += 1
            time.sleep(0.005)
        c.close()

    t = threading.Thread(target=writer)
    t.start()
    try:
        dst = w["local"] / "copy.db"
        pm.sqlite_backup_copy(w["old"] / "webui.db", dst, busy_s=20, retries=3, retry_wait_s=0.1, step_pages=1)
    finally:
        stop.set()
        t.join()
    assert_eq(scalar(dst, "pragma integrity_check"), "ok", "a copy taken under a live writer passes integrity_check")
    assert_true(scalar(dst, "select count(*) from chat_message") >= 23, "and holds at least the rows committed before it started")
    assert_eq(scalar(w["old"] / "webui.db", "pragma integrity_check"), "ok", "the source is undamaged")


def test_path_map_and_other_prefixes():
    w = build_world("pathmap")
    c = sqlite3.connect(str(w["old"] / "webui.db"))
    c.execute("INSERT INTO file VALUES ('legacy','u1','/app/backend/data/uploads/legacy_x.png','{}')")
    c.execute("INSERT INTO file VALUES ('nopath','u1',NULL,'{}')")
    c.commit()
    c.close()
    (w["old"] / "uploads" / "legacy_x.png").write_bytes(b"legacy")
    rc, out = run(w, "seed", "--apply", "--path-map", "/app/backend/data/uploads/=" + str(w["new"]) + "/uploads/")
    assert_eq(rc, 0, "seed with an extra --path-map")
    assert_eq(scalar(w["new"] / "webui.db", "select path from file where id='legacy'"), str(w["new"]) + "/uploads/legacy_x.png", "the extra map applied")
    assert_true(scalar(w["new"] / "webui.db", "select path from file where id='nopath'") is None, "a NULL path is left alone")
    rc, out = run(w, "verify")
    assert_eq(rc, 0, "verify passes")
    w2 = build_world("pathmap2")
    c = sqlite3.connect(str(w2["old"] / "webui.db"))
    c.execute("INSERT INTO file VALUES ('legacy','u1','/app/backend/data/uploads/legacy_x.png','{}')")
    c.commit()
    c.close()
    assert_eq(run(w2, "seed", "--apply")[0], 0, "seed without the extra map")
    rc, out = run(w2, "verify")
    assert_true("other=1" in out, "verify reports the unmapped legacy path as a count")


def test_reverse_cutover_rollback():
    w = build_world("reverse")
    assert_eq(run(w, "seed", "--apply")[0], 0, "seed")
    # the new pod went live and wrote
    add_messages(w["new"] / "webui.db", HER, 20, 5)
    (w["new"] / "compactor" / "facts" / "written-on-new-pod.json").write_text('{"facts":[{"text":"x"}]}')
    (w["new"] / "uploads" / f"idN_{PRIVATE}N.png").write_bytes(b"made on the new pod")
    c = sqlite3.connect(str(w["new"] / "webui.db"))
    c.execute("INSERT INTO file VALUES ('idN','u1',?,'{}')", (str(w["new"]) + f"/uploads/idN_{PRIVATE}N.png",))
    c.commit()
    c.close()
    old_t = time.time() - 3600
    for dp, dns, fns in os.walk(w["new"]):
        for n in fns:
            os.utime(os.path.join(dp, n), (old_t, old_t))
    # rollback: this pod's tree (--from) back into the OLD tree (--to); the old pod stays stopped
    argv = ["cutover", "--reverse", "--from", str(w["new"]), "--to", str(w["old"]), "--local-data-dir", str(w["new"]),
            "--zl-from", str(w["root"] / "v10" / "zl-ops"), "--zl-to", str(w["zl"]), "--no-supervisor-check",
            "--old-pod-stopped", "--quiet-seconds", "0", "--tmp-dir", str(w["local"] / "scratch"), "--retry-wait", "0.1"]
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = pm.main([a for a in argv if a != "--old-pod-stopped"] + ["--apply"])
    assert_eq(rc, 2, "a reverse cutover also needs --old-pod-stopped")
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = pm.main(argv + ["--apply"])
    if rc != 0:
        print(out.getvalue()[-1500:], err.getvalue()[-500:])
    assert_eq(rc, 0, "reverse cutover exits 0")
    assert_eq(scalar(w["old"] / "webui.db", "select count(*) from chat_message where chat_id=?", HER), 25, "the old tree has what was written on the new pod")
    assert_eq(scalar(w["old"] / "webui.db", "select count(*) from file where path like ?", str(w["old"]) + "/uploads/%"), 6,
              "its file.path rows point at the OLD tree again")
    assert_eq(scalar(w["old"] / "webui.db", "select count(*) from file where path like ?", str(w["new"]) + "/%"), 0, "none point at the new tree")
    assert_true((w["old"] / "compactor/facts/written-on-new-pod.json").is_file(), "the new pod's facts came back")
    assert_true((w["old"] / f"uploads/idN_{PRIVATE}N.png").is_file(), "and its upload")
    assert_true(not (w["local"] / "webui.db.migrate-tmp").exists(), "no temp file left on the local disk")


def test_interrupted_seed_is_resumable():
    w = build_world("resume")
    assert_eq(run(w, "seed", "--apply")[0], 0, "seed")
    m = json.loads((w["new"] / ".migrate" / "seed.json").read_text())
    m["complete"] = False
    m["files"] = {k: v for k, v in list(m["files"].items())[:3]}
    (w["new"] / ".migrate" / "seed.json").write_text(json.dumps(m))
    (w["new"] / "uploads" / f"id1_{PRIVATE}1.png.migrate-tmp").write_bytes(b"partial")
    rc, out = run(w, "seed", "--apply")
    assert_eq(rc, 0, "a seed can resume after a partial manifest")
    assert_eq(run(w, "verify")[0], 0, "and verifies")
    assert_true(json.loads((w["new"] / ".migrate" / "seed.json").read_text())["complete"], "the manifest is complete again")


def test_lock_prevents_concurrent_runs():
    w = build_world("lock")
    assert_eq(run(w, "seed", "--apply")[0], 0, "seed")
    (w["new"] / ".migrate" / "lock").write_text("1 now")
    rc, out = run(w, "seed", "--apply")
    assert_eq(rc, 2, "a held lock refuses a second run")


def test_supervisor_status_parser():
    st = pm.parse_supervisor_status("openwebui                        RUNNING   pid 90, uptime 0:00:45\n"
                                    "webuidb-sync                     STOPPED   Not started\n"
                                    "compactor: ERROR (no such process)\n"
                                    "unix:///var/run/supervisor.sock no such file\n")
    assert_eq(st, {"openwebui": "RUNNING", "webuidb-sync": "STOPPED"}, "supervisorctl status is parsed; errors are not states")
    probs = pm.local_services_stopped("/nonexistent/supervisorctl", None, ports=())
    assert_true(len(probs) == 1, "a missing supervisorctl is a refusal, not a pass")


def test_a_listener_on_a_guarded_port_refuses():
    import socket
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    try:
        w = build_world("port")
        pm.GUARDED_PORTS = (port,)
        rc, out = run(w, "seed", "--apply")
        assert_eq(rc, 2, "seed refuses while something listens on a guarded port, even if supervisorctl says STOPPED")
        assert_true("listening" in out, "and says so")
    finally:
        pm.GUARDED_PORTS = ()
        srv.close()


def test_exit_codes_and_usage():
    rc, _ = run(build_world("usage"), "seed", "--bogus-flag")
    assert_eq(rc, 2, "an unknown flag exits 2")
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        try:
            pm.main([])
            rc = 0
        except SystemExit as e:
            rc = e.code
    assert_eq(rc, 2, "no subcommand exits 2")
    try:
        pm.main(["seed", "--from", "/x", "--to", "/y", "--appl"])
        rc = 0
    except SystemExit as e:
        rc = e.code
    assert_eq(rc, 2, "an abbreviated flag is not accepted")


def _all():
    return [
        test_dry_run_changes_nothing,
        test_seed_copies_everything_and_never_writes_the_source,
        test_with_cache_flag,
        test_secret_key_flag,
        test_source_change_then_cutover_then_verify,
        test_local_db_replacement,
        test_first_boot_leftovers_are_accepted_as_pristine,
        test_refusals,
        test_corrupt_source_installs_nothing,
        test_backup_api_copy_is_consistent_under_a_writer,
        test_path_map_and_other_prefixes,
        test_reverse_cutover_rollback,
        test_interrupted_seed_is_resumable,
        test_lock_prevents_concurrent_runs,
        test_supervisor_status_parser,
        test_a_listener_on_a_guarded_port_refuses,
        test_exit_codes_and_usage,
    ]


if __name__ == "__main__":
    try:
        for t in _all():
            print(f"\n[{t.__name__}]")
            t()
        print()
        if FAILED:
            print(f"{FAILED} FAILED")
            sys.exit(1)
        print("All pod-migrate tests passed.")
    finally:
        shutil.rmtree(_TMP, ignore_errors=True)
