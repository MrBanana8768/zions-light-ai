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
    assert_eq(rc, 2, "a re-seed after a cutover is REFUSED (the target is live; fix round 2)")
    assert_true("ALREADY COMPLETED" in out, "and says a cutover already completed")


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
    # the fix under test. The stale local DB is a WRITTEN one now, so the tool insists on being told
    rc, out = run(w, "cutover", "--old-pod-stopped", "--apply")
    assert_eq(rc, 2, "cutover REFUSES while the new pod's local DB has writes the tool did not install")
    assert_eq(scalar(local, "select count(*) from chat where id='test-only-chat'"), 1, "... and changed nothing")
    rc, out = run(w, "cutover", "--old-pod-stopped", "--apply", "--discard-new-pod-writes")
    assert_eq(rc, 0, "cutover exits 0 with --discard-new-pod-writes")
    assert_true("MOVED ASIDE" in out, "and it printed what it moves aside")
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


def reverse(w: dict, *extra: str, frm=None, to=None, defaults_zl: bool = False, keep_stopped: bool = True) -> tuple[int, str]:
    """`cutover --reverse` as the rollback runbook runs it, on the new pod (--from the new tree)."""
    argv = ["cutover", "--reverse", "--from", str(frm or w["new"]), "--to", str(to or w["old"]),
            "--local-data-dir", str(w["new"]), "--no-supervisor-check", "--quiet-seconds", "0",
            "--tmp-dir", str(w["local"] / "scratch"), "--retry-wait", "0.1", "--local-db", str(w["local"] / "webui.db")]
    if keep_stopped:
        argv.append("--old-pod-stopped")
    if not defaults_zl:
        argv += ["--zl-from", str(w["root"] / "v10" / "zl-ops"), "--zl-to", str(w["zl"])]
    argv += list(extra)
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        try:
            rc = pm.main(argv)
        except SystemExit as e:
            rc = int(e.code or 0)
    return rc, out.getvalue() + err.getvalue()


def new_pod_writes(w: dict) -> None:
    """She (or a test) wrote on the new pod: the LIVE LOCAL database, the store and an upload.
    The snapshot on the volume is NOT updated (the sync has not run): that is the H2 hazard."""
    local = w["local"] / "webui.db"
    add_messages(local, HER, 20, 5)
    (w["new"] / "compactor" / "facts" / "written-on-new-pod.json").write_text('{"facts":[{"text":"x"}]}')
    (w["new"] / "uploads" / f"idN_{PRIVATE}N.png").write_bytes(b"made on the new pod")
    c = sqlite3.connect(str(local))
    c.execute("INSERT INTO file VALUES ('idN','u1',?,'{}')", (str(w["new"]) + f"/uploads/idN_{PRIVATE}N.png",))
    c.commit()
    c.close()
    old_t = time.time() - 3600
    for dp, dns, fns in os.walk(w["new"]):
        for n in fns:
            os.utime(os.path.join(dp, n), (old_t, old_t))


def test_reverse_cutover_rollback():
    """R2 H2: the rollback copies the new pod's LIVE LOCAL database, not the snapshot."""
    w = build_world("reverse")
    assert_eq(run(w, "seed", "--apply")[0], 0, "seed")
    new_pod_writes(w)
    snap = w["new"] / "webui.db"
    assert_eq(scalar(snap, "select count(*) from chat_message where chat_id=?", HER), 20, "the snapshot on the volume is STALE (20)")
    assert_eq(scalar(w["local"] / "webui.db", "select count(*) from chat_message where chat_id=?", HER), 25, "the local live DB has 25")
    rc, out = reverse(w, "--apply", keep_stopped=False)
    assert_eq(rc, 2, "a reverse cutover also needs --old-pod-stopped")
    rc, out = reverse(w, "--apply")
    if rc != 0:
        print(out[-1500:])
    assert_eq(rc, 0, "reverse cutover exits 0 (its verify compares the LOCAL database with the old tree)")
    assert_true("DIFFERS from the snapshot" in out, "it says the local DB differs from the snapshot and that LOCAL is copied")
    assert_eq(scalar(w["old"] / "webui.db", "select count(*) from chat_message where chat_id=?", HER), 25,
              "the old tree has ALL 25 messages, including the ones only the local DB had")
    assert_eq(scalar(w["old"] / "webui.db", "select count(*) from file where path like ?", str(w["old"]) + "/uploads/%"), 6,
              "its file.path rows point at the OLD tree again")
    assert_eq(scalar(w["old"] / "webui.db", "select count(*) from file where path like ?", str(w["new"]) + "/%"), 0, "none point at the new tree")
    assert_true((w["old"] / "compactor/facts/written-on-new-pod.json").is_file(), "the new pod's facts came back")
    assert_true((w["old"] / f"uploads/idN_{PRIVATE}N.png").is_file(), "and its upload")
    assert_true(not (w["local"] / "webui.db.migrate-tmp").exists(), "no temp file left on the local disk")
    assert_eq(sha(w["local"] / "webui.db") != sha(snap), True, "the new pod's own files were only READ")
    assert_eq(scalar(w["local"] / "webui.db", "select count(*) from chat_message where chat_id=?", HER), 25, "the local DB is untouched")
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = pm.main(["status", "--to", str(w["old"]), "--require", "cutover"])
    assert_eq(rc, 0, "status --to <old tree> --require cutover: COMPLETE after the reverse cutover (the rollback gate)")


def test_reverse_refuses_without_the_local_db_and_reads_the_snapshot_only_in_snapshot_mode():
    w = build_world("revlocal")
    assert_eq(run(w, "seed", "--apply")[0], 0, "seed")
    new_pod_writes(w)
    (w["local"] / "webui.db").rename(w["local"] / "webui.db.gone")
    before = tree_fingerprint(w["old"])
    rc, out = reverse(w, "--apply")
    assert_eq(rc, 2, "no local DB: the rollback REFUSES instead of quietly using the (older) snapshot")
    assert_true("local live database does not exist" in out, "and says so")
    assert_eq(tree_fingerprint(w["old"]), before, "the old tree is untouched")
    rc, out = reverse(w, "--apply", "--db-mode", "snapshot")
    assert_eq(rc, 0, "--db-mode snapshot is the explicit way to use the snapshot")
    assert_eq(scalar(w["old"] / "webui.db", "select count(*) from chat_message where chat_id=?", HER), 20, "and it holds the snapshot's 20")
    (w["local"] / "webui.db.gone").rename(w["local"] / "webui.db")
    (w["local"] / "webui.db-journal").write_bytes(b"x")
    rc, out = reverse(w, "--apply")
    assert_eq(rc, 2, "a hot journal beside the local DB refuses the rollback")
    (w["local"] / "webui.db-journal").unlink()


def test_reverse_from_must_be_this_pods_own_tree():
    """R2 M7: `--reverse --from` that is not THIS pod's DATA_DIR (the arguments swapped) is refused."""
    w = build_world("revfrom")
    assert_eq(run(w, "seed", "--apply")[0], 0, "seed")
    before_old, before_new = tree_fingerprint(w["old"]), tree_fingerprint(w["new"])
    rc, out = reverse(w, "--apply", frm=w["old"], to=w["new"])
    assert_eq(rc, 2, "--reverse --from <the OLD tree> --to <this pod's tree> is refused")
    assert_true("--reverse: --from must be THIS pod's DATA_DIR" in out, "with the reason")
    assert_eq((tree_fingerprint(w["old"]), tree_fingerprint(w["new"])), (before_old, before_new), "and nothing changed on either side")


def test_reverse_does_not_copy_the_wordban_dir_onto_itself():
    """R2 L9: a reverse with default ZL paths maps this pod's word-ban to the OLD pod's, and never onto itself."""
    w = build_world("revzl")
    assert_eq(run(w, "seed", "--apply")[0], 0, "seed")
    new_pod_writes(w)
    zl_new = w["root"] / "v10" / "zl-ops"
    (zl_new / "word-ban").mkdir(parents=True, exist_ok=True)
    (zl_new / "word-ban" / "active.json").write_text('{"rev": 8}')
    old_env = os.environ.get("ZL_HOME")
    os.environ["ZL_HOME"] = str(zl_new)
    try:
        rc, out = reverse(w, "--apply", defaults_zl=True)
        assert_eq(rc, 0, "reverse with default zl paths exits 0")
        assert_eq((w["zl"] / "word-ban" / "active.json").read_text(), '{"rev": 8}', "the new pod's word ban replaced the old pod's (mapped correctly)")
        assert_true(not any(zl_new.glob("**/*.migrate-tmp")) and not (zl_new / ".migrate").exists(), "and this pod's own dir got nothing")
        rc, out = reverse(w, "--apply", "--zl-from", str(zl_new), "--zl-to", str(zl_new))
        assert_true("same on both sides" in out, "identical source and target are recognised and skipped")
        assert_eq((zl_new / "word-ban" / "active.json").read_text(), '{"rev": 8}', "and this pod's word-ban file is unchanged")
    finally:
        if old_env is None:
            os.environ.pop("ZL_HOME", None)
        else:
            os.environ["ZL_HOME"] = old_env


def test_new_pod_writes_block_a_second_seed_before_go_live():
    """R2 H3: before go-live, re-running seed must not silently roll back a new pod that was only tested."""
    w = build_world("recut")
    assert_eq(run(w, "seed", "--apply")[0], 0, "seed")
    m = json.loads((w["new"] / ".migrate" / "seed.json").read_text())
    assert_eq(m["format"], 2, "the manifest is format 2")
    assert_eq(m["db"]["local_path"], str(w["local"] / "webui.db"), "the manifest records the local path it installed")
    assert_eq(m["db"]["local_sha256"], sha(w["local"] / "webui.db"), "and the installed database's sha256")
    assert_true("pending_install" not in m, "and no pending install remains once the install is committed")
    assert_eq(run(w, "seed", "--apply")[0], 0, "a re-seed with NO writes on the new pod is fine")
    local = w["local"] / "webui.db"
    add_messages(local, HER, 20, 5)            # a test login on the new pod
    before = (sha(local), tree_fingerprint(w["new"]))
    for cmd, extra in (("cutover", ("--old-pod-stopped",)), ("seed", ())):
        rc, out = run(w, cmd, "--apply", *extra)
        assert_eq(rc, 2, f"a {cmd} after the new pod wrote is REFUSED")
        assert_true("has written" in out and "chat_message" in out and "section 7" in out, "with the row counts and the rollback section (7)")
        rc, out = run(w, cmd, *extra)
        assert_eq(rc, 2, f"even the {cmd} dry run says so")
    assert_eq((sha(local), tree_fingerprint(w["new"])), before, "nothing changed")
    rc, out = run(w, "cutover", "--old-pod-stopped", "--apply", "--discard-new-pod-writes")
    assert_eq(rc, 0, "--discard-new-pod-writes lets a PRE-go-live cutover through")
    assert_true("MOVED ASIDE" in out and "chat_message 23 -> 28" in out, "after printing what it moves aside")
    aside = [d for d in w["local"].glob(".migrate-aside-*") if (d / "webui.db").is_file()
             and scalar(d / "webui.db", "select count(*) from chat_message where chat_id=?", HER) == 25]
    assert_eq(len(aside), 1, "the written database (all 25 of her messages) was moved aside, not deleted")
    assert_eq(scalar(local, "select count(*) from chat_message where chat_id=?", HER), 20, "the live one is the old pod's again")


def _live_target(tag: str):
    """A world with a completed cutover, and the new pod having taken 5 more of her messages."""
    w = build_world(tag)
    assert_eq(run(w, "seed", "--apply")[0], 0, f"{tag}: seed")
    assert_eq(run(w, "cutover", "--old-pod-stopped", "--apply")[0], 0, f"{tag}: cutover")
    return w


def test_after_go_live_no_flag_reaches_past_the_guard():
    """Fix round 2 (3): a completed cutover makes seed/cutover refuse, whatever the flags."""
    w = _live_target("golive")
    local = w["local"] / "webui.db"
    add_messages(local, HER, 20, 5)
    before = (sha(local), sha(w["new"] / "webui.db"), tree_fingerprint(w["new"]), tree_fingerprint(w["old"]))
    combos = [("no flag", (), True), ("--discard-new-pod-writes", ("--discard-new-pod-writes",), True),
              ("--force", ("--force",), True), ("--force --discard", ("--force", "--discard-new-pod-writes"), True),
              ("--db-mode snapshot", (), False), ("--db-mode snapshot --discard --force", ("--discard-new-pod-writes", "--force"), False)]
    for label, extra, local_db in combos:
        for cmd, more in (("cutover", ("--old-pod-stopped",)), ("seed", ())):
            for apply in (("--apply",), ()):
                rc, out = run(w, cmd, *apply, *more, *extra, local_db=local_db)
                assert_true(rc == 2 and "ALREADY COMPLETED" in out, f"{cmd}{' --apply' if apply else ''} with {label} is REFUSED after go-live")
    assert_eq((sha(local), sha(w["new"] / "webui.db"), tree_fingerprint(w["new"]), tree_fingerprint(w["old"])), before,
              "and nothing changed anywhere (her 25 messages are the live ones)")
    # with NO writes on the new pod it is refused too: the target is live either way
    w2 = _live_target("golive2")
    rc, out = run(w2, "cutover", "--old-pod-stopped", "--apply")
    assert_eq(rc, 2, "a re-cutover into a live target is refused even when the new pod has not written")
    # the way forward: --reverse works, and so does a fresh --to
    rc, out = reverse(w, "--apply")
    assert_eq(rc, 0, "--reverse (the rollback) is still allowed after go-live")
    assert_eq(scalar(w["old"] / "webui.db", "select count(*) from chat_message where chat_id=?", HER), 25, "and brings her 25 messages back")
    w3 = build_world("golive3")
    shutil.rmtree(w3["new"], ignore_errors=True)
    assert_eq(run(w3, "seed", "--apply")[0], 0, "a fresh --to directory seeds fine")


def test_manual_recovery_after_a_cutover_whose_verify_failed():
    """Runbook 6.4: the hand step (move .migrate aside, seed --force, cutover) works, and the tool alone never lets it happen."""
    w = _live_target("recov")
    mp = w["new"] / ".migrate" / "seed.json"
    m = json.loads(mp.read_text())
    m["verified"], m["step"] = False, "verify FAILED"
    mp.write_text(json.dumps(m))
    rc, out = status(w, "--require", "cutover")
    assert_true(rc == 1 and "A cutover completed here" in out and "NEW" in out, "status: INCOMPLETE, and it says not to re-run into this directory")
    assert_eq(run(w, "cutover", "--old-pod-stopped", "--apply")[0], 2, "the tool refuses the re-run")
    shutil.move(str(w["new"] / ".migrate"), str(w["new"] / ".migrate-failed-x"))
    assert_eq(run(w, "seed", "--apply", "--force")[0], 0, "after the hand step, seed --force works")
    assert_eq(run(w, "cutover", "--old-pod-stopped", "--apply")[0], 0, "and so does a fresh cutover")
    assert_eq(run_status_rc(w, "--require", "cutover"), 0, "status: COMPLETE")


def test_an_older_or_damaged_manifest_is_refused_whatever_the_flags():
    """Fix round 2 (3): MANIFEST_FORMAT 2. A format-1 (or unreadable) manifest may hide a completed cutover."""
    for label, mutate in (("format 1", lambda m: {**m, "format": 1}), ("no format", lambda m: {k: v for k, v in m.items() if k != "format"}),
                          ("unreadable", None)):
        w = _live_target("oldman-" + label.replace(" ", ""))
        mp = w["new"] / ".migrate" / "seed.json"
        m = json.loads(mp.read_text())
        mp.write_text(json.dumps(mutate(m)) if mutate else "{not json")
        before = (sha(w["local"] / "webui.db"), tree_fingerprint(w["new"]))
        for extra, local_db in (((), True), (("--force",), True), (("--discard-new-pod-writes", "--force"), True), ((), False)):
            for cmd, more in (("cutover", ("--old-pod-stopped",)), ("seed", ())):
                rc, out = run(w, cmd, "--apply", *more, *extra, local_db=local_db)
                assert_true(rc == 2 and "cannot use it" in out, f"{label}: {cmd} {' '.join(extra) or '(no flag)'}{' snapshot' if not local_db else ''} is refused")
        assert_eq((sha(w["local"] / "webui.db"), tree_fingerprint(w["new"])), before, f"{label}: nothing changed")
        rc, out = status(w)
        assert_true(rc == 1 and "cannot be used" in out, f"{label}: status says INCOMPLETE and why (exit 1)")
    w = build_world("oldman-fresh")
    assert_eq(run(w, "seed", "--apply")[0], 0, "a fresh --to directory works")


def test_a_kill_during_the_db_install_does_not_cause_a_false_refusal():
    """Fix round 2 (2): a REAL SIGKILL between placing the new local DB and the manifest commit."""
    w = build_world("kill9")
    assert_eq(run(w, "seed", "--apply")[0], 0, "seed")
    add_messages(w["old"] / "webui.db", HER, 20, 4)      # so the cutover's database differs from the seed's
    local = w["local"] / "webui.db"
    seed_sha = sha(local)
    child = (
        "import importlib.util, os, signal, sys, json\n"
        "spec = importlib.util.spec_from_file_location('pod_migrate', sys.argv[1]); pm = importlib.util.module_from_spec(spec)\n"
        "sys.modules['pod_migrate'] = pm; spec.loader.exec_module(pm); pm.GUARDED_PORTS = ()\n"
        "real = os.replace\n"
        "def replace(a, b, *x, **k):\n"
        "    r = real(a, b, *x, **k)\n"
        "    if str(b) == sys.argv[2]: os.kill(os.getpid(), signal.SIGKILL)\n"
        "    return r\n"
        "pm.os.replace = replace\n"
        "sys.exit(pm.main(json.loads(sys.argv[3])))\n")
    argv = ["cutover", "--from", str(w["old"]), "--to", str(w["new"]), "--local-data-dir", str(w["new"]),
            "--zl-from", str(w["zl"]), "--zl-to", str(w["root"] / "v10" / "zl-ops"), "--tmp-dir", str(w["local"] / "scratch"),
            "--retries", "2", "--retry-wait", "0.1", "--db-mode", "local", "--local-db", str(local),
            "--supervisorctl", str(w["sup"]), "--supervisor-conf", "x.conf", "--quiet-seconds", "0", "--old-pod-stopped", "--apply"]
    r = subprocess.run([sys.executable, "-c", child, str(_SCRIPT), str(local), json.dumps(argv)], capture_output=True, text=True)
    assert_eq(r.returncode, -9, "the child really died of SIGKILL")
    assert_true(sha(local) != seed_sha, "the new local database had been placed (the manifest still records the seed's hash)")
    m = json.loads((w["new"] / ".migrate" / "seed.json").read_text())
    assert_eq(m["db"]["local_sha256"], seed_sha, "the manifest's committed hash is the OLD one")
    assert_eq(m["pending_install"]["sha256"], sha(local), "but the pending install hash is the one on disk")
    assert_true((w["new"] / ".migrate" / "lock").exists(), "the killed run left its lock (the runbook says to remove it by hand)")
    (w["new"] / ".migrate" / "lock").unlink()
    assert_eq(run_status_rc(w), 1, "status: INCOMPLETE")
    rc, out = run(w, "cutover", "--old-pod-stopped", "--apply")
    assert_eq(rc, 0, "the re-run SUCCEEDS (before this fix: REFUSED 'the new pod has written')")
    assert_true("has written" not in out, "with no false refusal")
    assert_eq(run_status_rc(w, "--require", "cutover"), 0, "status: COMPLETE")
    assert_eq(sha(local), sha(w["new"] / "webui.db"), "the local DB equals the snapshot")
    # and a REAL write by the new pod after the kill is still caught
    w2 = build_world("second")
    assert_eq(run(w2, "seed", "--apply")[0], 0, "seed (second world)")
    add_messages(w2["old"] / "webui.db", HER, 20, 4)
    local2 = w2["local"] / "webui.db"
    argv2 = [a.replace(str(w["old"]), str(w2["old"])).replace(str(w["new"]), str(w2["new"])).replace(str(w["root"]), str(w2["root"]))
             .replace(str(local), str(local2)) for a in argv]
    r = subprocess.run([sys.executable, "-c", child, str(_SCRIPT), str(local2), json.dumps(argv2)], capture_output=True, text=True)
    assert_eq(r.returncode, -9, "second child killed too")
    (w2["new"] / ".migrate" / "lock").unlink()
    add_messages(local2, HER, 30, 3)
    rc, out = run(w2, "cutover", "--old-pod-stopped", "--apply")
    assert_eq(rc, 2, "a write by the new pod after the kill is still refused (pending is not a blanket pass)")


def test_live_source_written_under_the_copy_still_finishes():
    """Fix round 2 (1): every write restarts the backup API. Gentle -> fast -> single ladder."""
    real_connect = sqlite3.connect
    state = {"writes": 0, "limit": 10 ** 9}

    class WriteOnStep(sqlite3.Connection):
        def backup(self, target, *, pages=-1, progress=None, name="main", sleep=0.250):
            def wrapped(status, remaining, total):
                state["calls"] = state.get("calls", 0) + 1
                if state["writes"] < state["limit"] and state["calls"] % state.get("every", 1) == 0:
                    state["writes"] += 1
                    c = real_connect(state["src"], timeout=5)
                    c.execute("UPDATE chat SET updated_at=updated_at+1 WHERE id=?", (HER,))
                    c.commit()
                    c.close()
                if progress:
                    progress(status, remaining, total)
            return super().backup(target, pages=pages, progress=wrapped if progress else None, name=name, sleep=sleep)

    def ro_only(*a, **k):
        # only the SOURCE connection (mode=ro URI) gets the writing subclass
        if a and isinstance(a[0], str) and a[0].startswith("file:") and "mode=ro" in a[0]:
            return real_connect(*a, factory=WriteOnStep, **k)
        return real_connect(*a, **k)

    w = build_world("ladder")
    src = w["old"] / "webui.db"
    state["src"] = str(src)
    dst = w["local"] / "copy.db"
    real_fast = pm.FAST_PAGES
    slept = []
    real_sleep = pm.time.sleep
    pm.time.sleep = lambda x: slept.append(x) or real_sleep(0)
    sqlite3.connect = ro_only
    try:
        # (a) the source is written at every pause of the gentle attempt; the fast attempt has no pause and no write
        state["writes"], state["limit"] = 0, 1
        st = pm.sqlite_backup_copy(src, dst, retries=5, retry_wait_s=0.0, step_pages=1, step_sleep_s=0.01)
        assert_eq((st["mode"], st["attempts"]), ("fast", 2), "one write under a paced copy: the second attempt (fast, no pause) finishes")
        assert_eq(scalar(dst, "pragma integrity_check"), "ok", "and the copy is consistent")
        # (b) written under EVERY step, gentle and fast (fast steps forced to 1 page): only the single copy can finish
        pm.FAST_PAGES = 1
        state["writes"], state["limit"] = 0, 10 ** 9
        st = pm.sqlite_backup_copy(src, dst, retries=3, retry_wait_s=0.0, step_pages=1, step_sleep_s=0.01)
        assert_eq((st["mode"], st["attempts"]), ("single", 3), "written under every step: gentle and fast both restart, the last resort (one full copy) finishes")
        assert_true(st["restarts"] >= 2, f"and the restarts were counted ({st['restarts']})")
        assert_eq(scalar(dst, "pragma integrity_check"), "ok", "the single copy is consistent")
        # (b2) if the single copy is projected to hold the old pod's lock too long it is NOT tried
        state["writes"], state["every"], state["calls"] = 0, 3, 0     # steps 1 and 2 run clean (throughput is measured), step 3 is written under
        notes = io.StringIO()
        try:
            with redirect_stderr(notes), redirect_stdout(notes):
                pm.sqlite_backup_copy(src, dst, retries=3, retry_wait_s=0.0, step_pages=1, step_sleep_s=0.01, single_max_s=1e-9)
            skipped = False
        except pm.Failed:
            skipped = True
        state["every"] = 1
        assert_true(skipped and "not trying the single full copy" in notes.getvalue(), "the single copy is skipped when its projected lock hold exceeds the limit")
        # (c) the same but retries=2 (no single rung): fails with the actionable message, bounded
        state["writes"] = 0
        try:
            pm.sqlite_backup_copy(src, dst, retries=2, retry_wait_s=0.0, step_pages=1, step_sleep_s=0.01)
            failed = False
        except pm.Failed as e:
            failed = "close the app on every device" in str(e)
        assert_true(failed, "without the last rung it fails, telling her to close the app")
        assert_true(not dst.exists(), "and leaves no partial copy")
        # (d) the total pause of a gentle attempt is bounded
        pm.FAST_PAGES = real_fast
        slept.clear()
        state["writes"], state["limit"] = 0, 0
        pm.sqlite_backup_copy(src, dst, retries=2, retry_wait_s=0.0, step_pages=1, step_sleep_s=1.0, pause_budget_s=3.0)
        assert_true(abs(sum(slept) - 3.0) < 1e-6 and len(slept) == 3, f"the pause budget caps the total pause ({sum(slept):.1f} s of 1 s pauses)")
    finally:
        sqlite3.connect = real_connect
        pm.time.sleep = real_sleep
        pm.FAST_PAGES = real_fast
    assert_eq(pm._mode_for(1, 5), "gentle", "ladder: attempt 1 is gentle")
    assert_eq([pm._mode_for(i, 5) for i in range(2, 6)], ["fast", "fast", "fast", "single"], "then fast, and the last is the single copy")


def test_status_reports_and_clock_skew():
    w = build_world("skew")
    # the age of the source DB is measured against a file stamped ON THE VOLUME, not this host's clock
    real_time = pm.time.time
    old_t = time.time()
    os.utime(w["old"] / "webui.db", (old_t - 7, old_t - 7))
    pm.time.time = lambda: real_time() + 3600          # this host's clock is an hour AHEAD of the volume's
    try:
        age = pm.volume_age_s(w["old"] / "webui.db")
        rc, out = run(w, "seed")
    finally:
        pm.time.time = real_time
    assert_true(6 < age < 12, f"the volume's clock says the DB was written {age:.0f} s ago, not 3600 s")
    assert_true("last written 7 s ago" in out or "last written 8 s ago" in out or "last written 6 s ago" in out, "and the plan prints that")
    assert_true("she may be chatting" in out, "so the recent-write warning fires despite the skewed host clock")
    assert_true(not list(w["root"].glob(".pod-migrate-clock-*")), "the clock probe file is removed")


def test_same_size_same_second_change_is_caught_at_cutover():
    """R2 M4: MooseFS has 1 s mtimes. A same-size rewrite with an unchanged mtime must still be copied."""
    w = build_world("racy")
    assert_eq(run(w, "seed", "--apply")[0], 0, "seed")
    cdb = w["old"] / "compactor" / "chromadb" / "chroma.sqlite3"
    st2 = cdb.stat()
    c = sqlite3.connect(str(cdb))
    c.execute("UPDATE embeddings SET doc='dX' WHERE id=1")
    c.commit()
    c.close()
    os.utime(cdb, ns=(st2.st_atime_ns, st2.st_mtime_ns))
    assert_eq(cdb.stat().st_size, st2.st_size, "the chroma change is the same size too")
    rc_only, out_only = run(w, "verify")
    assert_true(rc_only == 1 and "tree 'compactor'" in out_only and "DIFFERENT" in out_only,
                "verify catches a same-size, same-mtime change to a SQLite file ALONE (the tree digest covers sqlite content)")
    f = w["old"] / "compactor" / "facts" / "conv1.json"
    st = f.stat()
    data = f.read_bytes()
    new = data.replace(b'"f0"', b'"F0"')
    assert_true(new != data and len(new) == len(data), "the change is the same size")
    f.write_bytes(new)
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns))
    seg = w["old"] / "compactor" / "chromadb" / "seg1" / "data_level0.bin"
    st3 = seg.stat()
    seg.write_bytes(b"\x02" * 512)
    os.utime(seg, ns=(st3.st_atime_ns, st3.st_mtime_ns))
    assert_true(run(w, "verify")[0] == 1, "verify already sees it (content hashes, including the sqlite logical hash)")
    rc, out = run(w, "cutover", "--old-pod-stopped", "--apply")
    assert_eq(rc, 0, "cutover exits 0, its own verify included")
    assert_eq((w["new"] / "compactor/facts/conv1.json").read_bytes(), new, "the same-size JSON rewrite was copied")
    assert_eq(scalar(w["new"] / "compactor/chromadb/chroma.sqlite3", "select doc from embeddings where id=1"), "dX",
              "the same-size chroma rewrite was re-copied through the backup API")
    assert_eq((w["new"] / "compactor/chromadb/seg1/data_level0.bin").read_bytes(), b"\x02" * 512, "and a same-size binary")
    assert_eq(run(w, "verify")[0], 0, "verify passes")
    a, b = pm.sqlite_logical_hash(w["old"] / "compactor/chromadb/chroma.sqlite3"), pm.sqlite_logical_hash(w["new"] / "compactor/chromadb/chroma.sqlite3")
    assert_eq(a, b, "the logical hash of a backup-API copy equals its source")
    c = sqlite3.connect(str(w["old"] / "compactor/chromadb/chroma.sqlite3"))
    c.execute("UPDATE embeddings SET doc='dY' WHERE id=1")
    c.commit()
    c.close()
    assert_true(pm.sqlite_logical_hash(w["old"] / "compactor/chromadb/chroma.sqlite3") != b, "and any content change changes it")


def test_snapshot_sidecars_are_moved_aside():
    """R2 M3: a stale -journal / .sync-tmp beside the snapshot must not survive next to the new snapshot."""
    w = build_world("sidecar")
    assert_eq(run(w, "seed", "--apply")[0], 0, "seed")
    for n in ("webui.db-journal", "webui.db-wal", "webui.db.sync-tmp", "webui.db.sync.lock"):
        (w["new"] / n).write_bytes(b"stale " + n.encode())
    (w["local"] / "webui.db-journal").write_bytes(b"stale local journal")
    rc, out = run(w, "cutover", "--old-pod-stopped", "--apply", "--discard-new-pod-writes")
    assert_eq(rc, 0, "cutover")
    for n in ("webui.db-journal", "webui.db-wal", "webui.db.sync-tmp", "webui.db.sync.lock"):
        assert_true(not (w["new"] / n).exists(), f"no stale {n} beside the new snapshot")
    kept = sorted(p.name for p in (w["new"] / ".migrate").glob("removed-*/snapshot-db/*"))
    assert_eq(kept, ["webui.db", "webui.db-journal", "webui.db-wal", "webui.db.sync-tmp", "webui.db.sync.lock"],
              "they were moved aside with the old snapshot (never deleted)")
    assert_true(not (w["local"] / "webui.db-journal").exists(), "and the local one too")
    assert_eq(run(w, "verify", "--strict-local")[0], 0, "verify")


def test_source_changing_during_the_cutover_fails_it():
    """R2 M1: the source is checked again after the copy; a change means it was not frozen."""
    w = build_world("m1")
    assert_eq(run(w, "seed", "--apply")[0], 0, "seed")
    orig = pm.apply_tree_plan

    def moving(*a, **k):
        r = orig(*a, **k)
        (w["old"] / "compactor" / "facts" / "written-mid-cutover.json").write_text('{"facts":[]}')
        return r

    pm.apply_tree_plan = moving
    try:
        rc, out = run(w, "cutover", "--old-pod-stopped", "--apply")
    finally:
        pm.apply_tree_plan = orig
    assert_eq(rc, 1, "a source that changed while the cutover ran fails it (exit 1)")
    assert_true("not frozen" in out, "and says the old pod is not frozen")
    m = json.loads((w["new"] / ".migrate" / "seed.json").read_text())
    assert_eq(m["complete"], False, "the manifest is NOT complete")
    assert_eq(pm.run_status(argparse_ns(dst=str(w["new"]), require=None)), 1, "status says INCOMPLETE")
    assert_true(not (w["new"] / ".migrate" / "lock").exists(), "the lock was released")


def argparse_ns(**kw):
    import argparse
    return argparse.Namespace(**kw)


def test_status_subcommand():
    w = build_world("status")
    assert_eq(run_status_rc(w), 2, "no manifest: exit 2")
    assert_eq(run(w, "seed", "--apply")[0], 0, "seed")
    assert_eq(run_status_rc(w), 0, "after a seed: COMPLETE (exit 0)")
    assert_eq(run_status_rc(w, "--require", "cutover"), 1, "--require cutover: a seed is not enough")
    m = json.loads((w["new"] / ".migrate" / "seed.json").read_text())
    m["complete"] = False
    m["step"] = "syncing the trees"
    (w["new"] / ".migrate" / "seed.json").write_text(json.dumps(m))
    rc, out = status(w)
    assert_eq(rc, 1, "complete=false: INCOMPLETE (exit 1)")
    assert_true("syncing the trees" in out and "Do NOT start services" in out, "it shows the last step and says not to start services")
    assert_eq(run(w, "seed", "--apply")[0], 0, "a re-run repairs it")
    assert_eq(run(w, "cutover", "--old-pod-stopped", "--apply")[0], 0, "cutover")
    assert_eq(run_status_rc(w, "--require", "cutover"), 0, "after a cutover with a passing verify: COMPLETE")
    m = json.loads((w["new"] / ".migrate" / "seed.json").read_text())
    m["verified"] = False
    (w["new"] / ".migrate" / "seed.json").write_text(json.dumps(m))
    rc, out = status(w)
    assert_eq(rc, 1, "a cutover whose verify did not pass is INCOMPLETE")
    m["verified"] = True
    (w["new"] / ".migrate" / "seed.json").write_text(json.dumps(m))
    (w["new"] / ".migrate" / "lock").write_text("1 now")
    assert_eq(run_status_rc(w), 1, "a held lock: INCOMPLETE")
    (w["new"] / ".migrate" / "lock").unlink()
    assert_eq(run_status_rc(w), 0, "and COMPLETE again")


def status(w, *extra):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = pm.main(["status", "--to", str(w["new"]), *extra])
    return rc, out.getvalue()


def run_status_rc(w, *extra):
    return status(w, *extra)[0]


def test_signals():
    """R2 M6: SIGHUP is ignored; SIGINT/SIGTERM stop cleanly and leave complete=false."""
    import signal
    before_handlers = {s: signal.getsignal(s) for s in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM)}
    w = build_world("sig")
    orig = pm.apply_tree_plan

    def hup(*a, **k):
        os.kill(os.getpid(), signal.SIGHUP)
        return orig(*a, **k)

    pm.apply_tree_plan = hup
    try:
        rc, out = run(w, "seed", "--apply")
    finally:
        pm.apply_tree_plan = orig
    assert_eq(rc, 0, "SIGHUP mid-run does not kill the run")
    assert_eq(json.loads((w["new"] / ".migrate" / "seed.json").read_text())["complete"], True, "and it completed")
    assert_eq({s: signal.getsignal(s) for s in before_handlers}, before_handlers, "the signal handlers are restored afterwards")

    for signame in ("SIGTERM", "SIGINT"):
        w = build_world("sig" + signame)
        assert_eq(run(w, "seed", "--apply")[0], 0, "seed")
        add_messages(w["old"] / "webui.db", HER, 20, 2)
        local = w["local"] / "webui.db"
        local_before = sha(local)
        signum = getattr(signal, signame)

        def kill_in_trees(*a, **k):
            os.kill(os.getpid(), signum)
            return orig(*a, **k)

        pm.apply_tree_plan = kill_in_trees
        try:
            rc, out = run(w, "cutover", "--old-pod-stopped", "--apply")
        finally:
            pm.apply_tree_plan = orig
        assert_eq(rc, 1, f"{signame} during the tree sync: exit 1")
        assert_true("INTERRUPTED" in out, "it says it was interrupted")
        m = json.loads((w["new"] / ".migrate" / "seed.json").read_text())
        assert_eq((m["complete"], m.get("interrupted")), (False, signame), "manifest complete=false, with the signal recorded")
        assert_eq(sha(local), local_before, "the live database was not touched")
        assert_true(not (w["new"] / ".migrate" / "lock").exists(), "the lock was released")
        assert_eq(run_status_rc(w), 1, "status says INCOMPLETE")
        assert_eq(run(w, "cutover", "--old-pod-stopped", "--apply")[0], 0, "and the same command, re-run, completes it")
        assert_eq(run_status_rc(w, "--require", "cutover"), 0, "status: COMPLETE")

    # a SIGTERM inside the database install is deferred until the install has finished: never a half-installed pair
    w = build_world("sigcrit")
    assert_eq(run(w, "seed", "--apply")[0], 0, "seed")
    add_messages(w["old"] / "webui.db", HER, 20, 2)
    orig_fsync = pm.fsync_dir
    fired = []

    def fsync_and_kill(path):
        if not fired and Path(path) == w["local"]:
            fired.append(1)
            os.kill(os.getpid(), signal.SIGTERM)
        return orig_fsync(path)

    pm.fsync_dir = fsync_and_kill
    try:
        rc, out = run(w, "cutover", "--old-pod-stopped", "--apply")
    finally:
        pm.fsync_dir = orig_fsync
    assert_true(bool(fired), "the signal really arrived inside install_db")
    assert_eq(rc, 1, "the run still stops (exit 1)")
    assert_eq(sha(w["local"] / "webui.db"), sha(w["new"] / "webui.db"), "but the local DB and the snapshot were BOTH fully installed first")
    assert_eq(scalar(w["local"] / "webui.db", "select count(*) from chat_message where chat_id=?", HER), 22, "with the newest data")
    assert_eq(json.loads((w["new"] / ".migrate" / "seed.json").read_text())["complete"], False, "and the manifest still says complete=false")

    # a SIGTERM that lands while the manifest is being marked complete: the run reports INTERRUPTED, so the manifest must not claim complete
    w = build_world("sigsave")
    orig_save = pm.save_manifest
    hit = []

    def save_and_kill(to, m):
        orig_save(to, m)
        if m.get("complete") and not hit:
            hit.append(1)
            os.kill(os.getpid(), signal.SIGTERM)

    pm.save_manifest = save_and_kill
    try:
        rc, out = run(w, "seed", "--apply")
    finally:
        pm.save_manifest = orig_save
    assert_true(bool(hit), "the signal arrived while the manifest was marked complete")
    assert_eq(rc, 1, "the run reports the interruption (exit 1)")
    assert_eq(json.loads((w["new"] / ".migrate" / "seed.json").read_text())["complete"], False,
              "and the manifest is rewritten complete=false: an interrupted run is never COMPLETE")
    assert_eq(run_status_rc(w), 1, "status: INCOMPLETE")


def test_seed_on_a_live_source_is_gentle_and_retries_bad_copies():
    """R2 L8: small steps with pauses, bounded retry on an integrity_check failure, total-time report."""
    w = build_world("live")
    ns = argparse_ns(busy_timeout=1.0, retries=3, retry_wait=0.0, step_pages=None, step_sleep=None)
    kw = pm.backup_kwargs(ns, "seed")
    assert_eq((kw["step_pages"], kw["step_sleep_s"]), (4096, 0.1), "a seed reads a live source in 4096-page steps with a 0.1 s pause (total pause capped at 20 s)")
    kw = pm.backup_kwargs(ns, "cutover")
    assert_eq((kw["step_pages"], kw["step_sleep_s"]), (4096, 0.0), "a cutover (source frozen) is fast")
    # the pause really happens between steps
    slept = []
    real_sleep = pm.time.sleep
    pm.time.sleep = lambda x: slept.append(x) or real_sleep(0)
    try:
        rc, out = run(w, "seed", "--apply", "--step-pages", "1", "--step-sleep", "0.123")
    finally:
        pm.time.sleep = real_sleep
    assert_eq(rc, 0, "seed with 1-page steps")
    assert_true(slept.count(0.123) >= 3, f"it paused between steps ({slept.count(0.123)} pauses)")
    assert_true("database copy: 1 attempt(s)" in out and " s in total" in out, "and reports the attempts and the total time")
    assert_true("she may be chatting" in out or "only while she is idle" in out, "and reminds that a seed needs her idle")
    # integrity_check failing twice, then passing: retried
    bad = {"n": 2}
    real_connect = sqlite3.connect

    class Flaky(sqlite3.Connection):
        def execute(self, sql, *a):
            if sql == "PRAGMA integrity_check" and bad["n"] > 0:
                bad["n"] -= 1
                return super().execute("SELECT 'corrupt'")
            return super().execute(sql, *a)

    sqlite3.connect = lambda *a, **k: real_connect(*a, factory=Flaky, **k)
    try:
        w2 = build_world("live2")
        rc, out = run(w2, "seed", "--apply", "--retries", "3")
    finally:
        sqlite3.connect = real_connect
    assert_eq(rc, 0, "a copy that fails integrity_check twice is retried and the third attempt is used")
    assert_true(out.count("failed (the copy failed integrity_check)") == 2 and "3 attempt(s)" in out, "two retries reported, three attempts in total")
    assert_eq(run(w2, "verify")[0], 0, "and the result verifies")
    # it never passes: bounded, and nothing is installed
    bad["n"] = 999
    sqlite3.connect = lambda *a, **k: real_connect(*a, factory=Flaky, **k)
    try:
        w3 = build_world("live3")
        rc, out = run(w3, "seed", "--apply")
    finally:
        sqlite3.connect = real_connect
    assert_eq(rc, 1, "an integrity_check that never passes fails the seed after the bounded retries (exit 1)")
    assert_true("failed integrity_check on all 2 attempts" in out, "and says so")
    assert_true(not (w3["local"] / "webui.db").exists() and not (w3["new"] / "webui.db").exists(), "nothing was installed")


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
        test_reverse_refuses_without_the_local_db_and_reads_the_snapshot_only_in_snapshot_mode,
        test_reverse_from_must_be_this_pods_own_tree,
        test_reverse_does_not_copy_the_wordban_dir_onto_itself,
        test_new_pod_writes_block_a_second_seed_before_go_live,
        test_after_go_live_no_flag_reaches_past_the_guard,
        test_manual_recovery_after_a_cutover_whose_verify_failed,
        test_an_older_or_damaged_manifest_is_refused_whatever_the_flags,
        test_a_kill_during_the_db_install_does_not_cause_a_false_refusal,
        test_live_source_written_under_the_copy_still_finishes,
        test_status_reports_and_clock_skew,
        test_same_size_same_second_change_is_caught_at_cutover,
        test_snapshot_sidecars_are_moved_aside,
        test_source_changing_during_the_cutover_fails_it,
        test_status_subcommand,
        test_signals,
        test_seed_on_a_live_source_is_gentle_and_retries_bad_copies,
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
