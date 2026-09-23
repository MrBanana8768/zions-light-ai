"""Tests for scripts/fix-stale-unfinished.py — the upstream OpenWebUI
#14806 stale-spinner fix (see that script's own module docstring for the
full story: storage facts, the two target categories, the tip/in-flight
exclusion, and the forensics-directory backup this script uses instead of
the plain-copy convention its siblings use).

The `chat`/`chat_message` CREATE TABLE strings in `_SCHEMA` below are
copied verbatim (`sqlite_master.sql`) from a READ-ONLY copy of the real
2026-09-23 pod backup (`/home/drew/pod-exports/2026-09-23/backup/webui.db`,
opened `?mode=ro&immutable=1`, never written to) — not hand-written, so a
column this script depends on (`chat_message.done`, `chat.
current_message_id`) is exercised against the actual production shape,
not a simplified stand-in.

Run directly:
    python compactor/test_fix_stale_unfinished_script.py
"""

import importlib.util
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SCRIPT = _HERE.parent / "scripts" / "fix-stale-unfinished.py"
_spec = importlib.util.spec_from_file_location("fix_stale_unfinished_script", _SCRIPT)
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)

_FAILS = []


def assert_eq(actual, expected, label):
    if actual != expected:
        print(f"FAIL {label}: expected {expected!r}, got {actual!r}")
        _FAILS.append(label)
        return
    print(f"  ok   {label}")


def assert_true(cond, label, detail=""):
    if not cond:
        print(f"FAIL {label}" + (f" ({detail})" if detail else ""))
        _FAILS.append(label)
        return
    print(f"  ok   {label}")


# ===========================================================================
# real 0.11 schema (verbatim from the real 2026-09-23 backup's sqlite_master)
# ===========================================================================

_CHAT_SCHEMA = """
CREATE TABLE "chat" (
	id VARCHAR(255) NOT NULL,
	user_id VARCHAR(255) NOT NULL,
	title TEXT NOT NULL,
	share_id VARCHAR(255),
	archived INTEGER NOT NULL,
	created_at DATETIME NOT NULL,
	updated_at DATETIME NOT NULL,
	chat JSON,
	pinned BOOLEAN,
	meta JSON DEFAULT '{}' NOT NULL,
	folder_id TEXT,
	tasks JSON,
	summary TEXT,
	last_read_at BIGINT, current_message_id TEXT, variables JSON,
	CONSTRAINT pk_chat PRIMARY KEY (id)
)
"""

_CHAT_MESSAGE_SCHEMA = """
CREATE TABLE chat_message (
	id TEXT NOT NULL,
	chat_id TEXT NOT NULL,
	user_id TEXT,
	role TEXT NOT NULL,
	parent_id TEXT,
	content JSON,
	output JSON,
	model_id TEXT,
	files JSON,
	sources JSON,
	embeds JSON,
	done BOOLEAN,
	status_history JSON,
	error JSON,
	usage JSON,
	created_at BIGINT,
	updated_at BIGINT, context_summary TEXT, meta JSON,
	PRIMARY KEY (id),
	FOREIGN KEY(chat_id) REFERENCES chat (id) ON DELETE CASCADE
)
"""

CHAT_ID = "ea1494ea-e9d7-46fb-8b7c-3a50d685d00e"

# Timestamps are anchored to the REAL wall clock at fixture-build time, not a
# fixed constant: the end-to-end tests run the actual script as a subprocess,
# which reads real time.time() with no override, so a fixed epoch baked in
# once would silently drift stale (and did, during manual debugging of this
# suite -- a "recent" message aged past the 10-minute guard between when the
# constant was written and when a later test actually ran). The direct
# build_targets() unit tests pass this same `now` back in explicitly instead,
# so they stay deterministic regardless of wall-clock drift.


def _hist_msg(mid, role, ts, parent=None, content="", done=None, done_present=True):
    m = {"id": mid, "role": role, "parentId": parent, "childrenIds": [],
         "timestamp": ts, "content": content}
    if done_present:
        m["done"] = done
    return m


def _link(msgs):
    for m in msgs.values():
        m["childrenIds"] = []
    for mid, m in msgs.items():
        p = m.get("parentId")
        if p in msgs and mid not in msgs[p]["childrenIds"]:
            msgs[p]["childrenIds"].append(mid)
    return msgs


def _scenario_messages(now):
    """The tree described in the module docstring's test-plan: two
    fixable targets on the branch (category a and category b), a
    same-branch candidate too young to touch, the tip and its in-flight
    reply (both category-a shaped but never to be touched), and two
    off-branch candidates (one of each category)."""
    OLD = now - 100_000    # ~27.8h old -- comfortably past any --min-age-minutes
    RECENT = now - 60      # 1 minute old -- well inside the default 10-minute guard
    msgs = {}
    msgs.update(dict([
        ("root", _hist_msg("root", "user", OLD)),
        ("q1", _hist_msg("q1", "user", OLD, parent="root")),
        ("a1_stale", _hist_msg("a1_stale", "assistant", OLD, parent="q1", content="", done=False)),
        ("q2", _hist_msg("q2", "user", OLD, parent="a1_stale")),
        ("a2_done", _hist_msg("a2_done", "assistant", OLD, parent="q2", content="hi there", done=True)),
        ("q3", _hist_msg("q3", "user", OLD, parent="a2_done")),
        ("a3_gapb", _hist_msg("a3_gapb", "assistant", OLD, parent="q3", content="world", done_present=False)),
        ("q4", _hist_msg("q4", "user", OLD, parent="a3_gapb")),
        ("young_on_branch", _hist_msg("young_on_branch", "assistant", RECENT, parent="q4", content="", done=False)),
        ("tip", _hist_msg("tip", "assistant", OLD, parent="young_on_branch", content="", done=False)),
        ("inflight", _hist_msg("inflight", "assistant", OLD, parent="tip", content="", done=False)),
        ("off_a_stale", _hist_msg("off_a_stale", "assistant", OLD, parent="q1", content="", done=False)),
        ("off_b_gap", _hist_msg("off_b_gap", "assistant", OLD, parent="q1", content="offb", done_present=False)),
    ]))
    return _link(msgs)


# table-side `done`: 0/1/None per message, matching the JSON side above
# except a3_gapb/off_b_gap (json missing, table already 1 -- category b)
_TABLE_DONE = {
    "root": None, "q1": None,
    "a1_stale": 0, "q2": None, "a2_done": 1, "q3": None,
    "a3_gapb": 1, "q4": None, "young_on_branch": 0, "tip": 0, "inflight": 0,
    "off_a_stale": 0, "off_b_gap": 1,
}


def _make_db(path, current_message_id="tip", now=None):
    now = int(now if now is not None else time.time())
    con = sqlite3.connect(str(path))
    con.executescript(_CHAT_SCHEMA)
    con.executescript(_CHAT_MESSAGE_SCHEMA)

    msgs = _scenario_messages(now)
    blob = json.dumps({"history": {"messages": msgs, "currentId": current_message_id}, "messages": []})
    con.execute(
        "insert into chat (id, user_id, title, archived, created_at, updated_at, chat, meta, current_message_id) "
        "values (?,?,?,?,?,?,?,?,?)",
        (CHAT_ID, "u1", "her chat", 0, now, now, blob, "{}", current_message_id),
    )
    prefix = CHAT_ID + "-"
    for mid, m in msgs.items():
        ts = m["timestamp"]
        con.execute(
            "insert into chat_message (id, chat_id, role, parent_id, content, done, created_at, updated_at) "
            "values (?,?,?,?,?,?,?,?)",
            (prefix + mid, CHAT_ID, m["role"], m.get("parentId"), json.dumps(m["content"]),
             _TABLE_DONE[mid], ts, ts),
        )
    con.commit()
    con.close()


def _read_chat(db):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    blob, cur_col, updated = con.execute(
        "select chat, current_message_id, updated_at from chat where id=?", (CHAT_ID,)
    ).fetchone()
    chat = json.loads(blob)
    prefix = CHAT_ID + "-"
    table = {}
    for mid, parent_id, role, content, done, created, upd in con.execute(
        "select id, parent_id, role, content, done, created_at, updated_at from chat_message where chat_id=?",
        (CHAT_ID,),
    ):
        table[mid[len(prefix):]] = {
            "row": mid, "parentId": parent_id, "role": role, "content": content,
            "done": done, "created": created, "updated": upd,
        }
    con.close()
    return chat, table, cur_col, updated


def run_script(args, env=None):
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    proc = subprocess.run([sys.executable, str(_SCRIPT), *args], capture_output=True, text=True, env=full_env)
    return proc.returncode, proc.stdout, proc.stderr


# ===========================================================================
# build_targets — categorization, tip/in-flight protection, branch, age
# ===========================================================================


def test_build_targets_categorizes_and_scopes_correctly():
    tmp = Path(tempfile.mkdtemp(prefix="fsu-targets-"))
    try:
        db = tmp / "webui.db"
        now = time.time()
        _make_db(db, now=now)
        chat, table, cur_col, _ = _read_chat(db)
        targets = S.build_targets(chat, table, cur_col, min_age_minutes=10, branch_only=True, now=now)
        by_id = {t.mid: t for t in targets}

        assert_true("a1_stale" in by_id and by_id["a1_stale"].category == "a", "a1_stale found as category a")
        assert_true(by_id["a1_stale"].in_scope, "a1_stale in scope (on-branch, old, unprotected)")
        assert_true("a3_gapb" in by_id and by_id["a3_gapb"].category == "b", "a3_gapb found as category b")
        assert_true(by_id["a3_gapb"].in_scope, "a3_gapb in scope (on-branch, old, unprotected)")
        assert_true("a2_done" not in by_id, "a2_done (already done both copies) is not a candidate at all")

        assert_true(not by_id["young_on_branch"].in_scope, "young_on_branch excluded")
        assert_eq(by_id["young_on_branch"].skip_reason, "younger than --min-age-minutes (10)",
                  "young_on_branch excluded specifically by the age guard")

        assert_true(not by_id["tip"].in_scope, "the current tip itself is never in scope")
        assert_true("tip" in by_id["tip"].skip_reason, "tip's skip reason names the tip/in-flight rule")
        assert_true(not by_id["inflight"].in_scope, "the tip's direct in-flight child is never in scope")

        assert_true(not by_id["off_a_stale"].in_scope, "off-branch category-a excluded under --branch-only")
        assert_eq(by_id["off_a_stale"].skip_reason, "off-branch (use --all-branches)", "off-branch reason is explicit")
        assert_true(not by_id["off_b_gap"].in_scope, "off-branch category-b excluded under --branch-only")

        in_scope_ids = {t.mid for t in targets if t.in_scope}
        assert_eq(in_scope_ids, {"a1_stale", "a3_gapb"}, "exactly the two intended on-branch targets are in scope")

        assert_eq(by_id["a1_stale"].branch_pos, 3, "a1_stale's branch position (root=1)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_all_branches_widens_scope_but_never_touches_tip_or_inflight():
    tmp = Path(tempfile.mkdtemp(prefix="fsu-allbranches-"))
    try:
        db = tmp / "webui.db"
        now = time.time()
        _make_db(db, now=now)
        chat, table, cur_col, _ = _read_chat(db)
        targets = S.build_targets(chat, table, cur_col, min_age_minutes=10, branch_only=False, now=now)
        in_scope_ids = {t.mid for t in targets if t.in_scope}
        assert_eq(in_scope_ids, {"a1_stale", "a3_gapb", "off_a_stale", "off_b_gap"},
                  "--all-branches adds the two off-branch targets")
        by_id = {t.mid: t for t in targets}
        assert_true(not by_id["tip"].in_scope, "tip still excluded under --all-branches")
        assert_true(not by_id["inflight"].in_scope, "in-flight reply still excluded under --all-branches")
        assert_true(not by_id["young_on_branch"].in_scope, "the age guard still applies under --all-branches")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ===========================================================================
# end-to-end: --apply changes only the intended flags, in both copies,
# leaves every byte of content alone, and is idempotent
# ===========================================================================


def test_apply_changes_only_targets_in_both_copies_and_preserves_content():
    tmp = Path(tempfile.mkdtemp(prefix="fsu-apply-"))
    forensics = tmp / "forensics"
    try:
        db = tmp / "webui.db"
        _make_db(db)
        before_chat, before_table, _, before_updated = _read_chat(db)

        rc, out, err = run_script([str(db), "--apply", "--json"], env={"FSU_FORENSICS_DIR": str(forensics)})
        assert_eq(rc, 0, f"apply exits 0 on full success (stderr={err!r})")
        result = json.loads(out)
        assert_eq(result["written"], 2, "exactly the 2 in-scope targets were written")
        assert_true(result["verify_ok"], "script's own post-write verification passed")
        assert_eq(result["integrity"], "ok", "integrity_check ok")

        after_chat, after_table, after_cur, after_updated = _read_chat(db)
        after_hist = after_chat["history"]["messages"]

        for mid in before_chat["history"]["messages"]:
            expect_json_done = mid in ("a1_stale", "a3_gapb") or before_chat["history"]["messages"][mid].get("done") is True
            assert_eq(after_hist[mid].get("done"), True if expect_json_done else before_chat["history"]["messages"][mid].get("done"),
                      f"{mid}: JSON done flag matches expectation")

        # both copies updated for the category-a target
        assert_eq(after_hist["a1_stale"]["done"], True, "a1_stale JSON done -> True")
        assert_eq(after_table["a1_stale"]["done"], 1, "a1_stale table done -> 1")
        # category-b target: JSON backfilled, table already correct and untouched
        assert_eq(after_hist["a3_gapb"]["done"], True, "a3_gapb JSON done -> True (was missing)")
        assert_eq(after_table["a3_gapb"]["done"], 1, "a3_gapb table done stays 1")

        # nothing else changed: every other message's done flag, and EVERY
        # message's content in both copies, byte for byte
        untouched = set(before_chat["history"]["messages"]) - {"a1_stale", "a3_gapb"}
        for mid in untouched:
            assert_eq(after_hist[mid].get("done"), before_chat["history"]["messages"][mid].get("done"),
                       f"{mid}: JSON done flag unchanged")
        for mid in before_chat["history"]["messages"]:
            assert_eq(after_hist[mid].get("content"), before_chat["history"]["messages"][mid].get("content"),
                       f"{mid}: JSON content byte-identical")
        for mid in before_table:
            assert_eq(after_table[mid]["content"], before_table[mid]["content"], f"{mid}: table content byte-identical")

        # nothing else on the chat row moved: current_message_id, updated_at
        assert_eq(after_cur, "tip", "current_message_id untouched")
        assert_eq(after_updated, before_updated, "chat.updated_at deliberately left untouched")

        # idempotency: a second run finds nothing to do
        rc2, out2, err2 = run_script([str(db), "--json"], env={"FSU_FORENSICS_DIR": str(forensics)})
        assert_eq(rc2, 0, f"second (dry) run exits 0 -- nothing left to do (stderr={err2!r})")
        result2 = json.loads(out2)
        assert_eq(result2["in_scope_total"], 0, "second run finds zero in-scope targets")

        rc3, out3, err3 = run_script([str(db), "--apply", "--json"], env={"FSU_FORENSICS_DIR": str(forensics)})
        assert_eq(rc3, 0, f"a second --apply is also a clean no-op (stderr={err3!r})")
        result3 = json.loads(out3)
        assert_eq(result3.get("written", 0), 0, "second --apply writes nothing")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_apply_never_touches_tip_or_inflight_even_with_all_branches():
    tmp = Path(tempfile.mkdtemp(prefix="fsu-tip-"))
    forensics = tmp / "forensics"
    try:
        db = tmp / "webui.db"
        _make_db(db)
        rc, out, err = run_script([str(db), "--apply", "--all-branches", "--json"],
                                   env={"FSU_FORENSICS_DIR": str(forensics)})
        assert_eq(rc, 0, f"apply --all-branches succeeds (stderr={err!r})")
        result = json.loads(out)
        assert_eq(result["written"], 4, "the 2 on-branch + 2 off-branch targets were written, no more")
        _, table, _, _ = _read_chat(db)
        assert_eq(table["tip"]["done"], 0, "tip's table row is still 0 -- never touched")
        assert_eq(table["inflight"]["done"], 0, "in-flight reply's table row is still 0 -- never touched")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ===========================================================================
# backup / restore (sqlite3 online backup API into the forensics dir)
# ===========================================================================


def test_backup_uses_forensics_dir_and_restore_is_md5_identical():
    tmp = Path(tempfile.mkdtemp(prefix="fsu-backup-"))
    forensics = tmp / "forensics"
    try:
        db = tmp / "webui.db"
        _make_db(db)
        old_env = os.environ.get("FSU_FORENSICS_DIR")
        os.environ["FSU_FORENSICS_DIR"] = str(forensics)
        try:
            stamp = S.utc_stamp()
            bak = S.backup_db(str(db), stamp)
            assert_true(bak.exists(), "backup file created under the forensics dir")
            assert_eq(bak.parent.parent, forensics, "backup lives directly under $FSU_FORENSICS_DIR")
            assert_true(bak.parent.name.startswith("fix-stale-unfinished-"), "backup dir is named for this script")

            before_bak_md5 = S.md5_of(bak)
            db.write_bytes(b"deliberately corrupted")
            got, msg = S.restore_db(str(db), stamp)
            assert_true(got is not None, "restore reports success", msg)
            after_md5 = S.md5_of(db)
            assert_eq(after_md5, before_bak_md5, "restored live db is md5-identical to the forensics backup file")
        finally:
            if old_env is None:
                os.environ.pop("FSU_FORENSICS_DIR", None)
            else:
                os.environ["FSU_FORENSICS_DIR"] = old_env
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_restore_refuses_cleanly_when_no_backup_exists():
    tmp = Path(tempfile.mkdtemp(prefix="fsu-norestore-"))
    forensics = tmp / "forensics"
    try:
        db = tmp / "webui.db"
        _make_db(db)
        old_env = os.environ.get("FSU_FORENSICS_DIR")
        os.environ["FSU_FORENSICS_DIR"] = str(forensics)
        try:
            got, msg = S.restore_db(str(db), "20000101T000000Z")
            assert_true(got is None, "restore with a nonexistent stamp refuses rather than crashing")
            assert_true("REFUSED" in msg, "refusal message says so", msg)
        finally:
            if old_env is None:
                os.environ.pop("FSU_FORENSICS_DIR", None)
            else:
                os.environ["FSU_FORENSICS_DIR"] = old_env
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ===========================================================================
# refusals
# ===========================================================================


def test_refuses_apply_when_another_process_holds_the_db_open():
    tmp = Path(tempfile.mkdtemp(prefix="fsu-openers-"))
    forensics = tmp / "forensics"
    try:
        db = tmp / "webui.db"
        _make_db(db)
        proc = subprocess.Popen([sys.executable, "-c", f"import time; f=open(r'{db}','rb'); time.sleep(6)"])
        try:
            deadline = time.time() + 3
            found = []
            while time.time() < deadline and not found:
                found = S.openers(str(db))
            assert_true(str(proc.pid) in found, "openers() sees the other process", str(found))

            rc, out, err = run_script([str(db), "--apply"], env={"FSU_FORENSICS_DIR": str(forensics)})
            assert_eq(rc, 1, f"--apply refuses (exit 1) while the db is held open (stderr={err!r})")
            assert_true("REFUSING" in out, "refusal message printed", out)
            assert_true(not forensics.exists() or not any(forensics.iterdir()),
                        "no backup was taken -- refusal happens before any write")
        finally:
            proc.kill()
            proc.wait()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_refuses_apply_when_a_journal_sidecar_exists():
    tmp = Path(tempfile.mkdtemp(prefix="fsu-journal-"))
    forensics = tmp / "forensics"
    try:
        db = tmp / "webui.db"
        _make_db(db)
        (tmp / "webui.db-journal").write_bytes(b"")
        rc, out, err = run_script([str(db), "--apply"], env={"FSU_FORENSICS_DIR": str(forensics)})
        assert_eq(rc, 1, f"--apply refuses (exit 1) when a -journal sidecar exists (stderr={err!r})")
        assert_true("REFUSING" in out, "refusal message printed", out)
        assert_true("RUNBOOK_DB_JOURNAL.md" in out, "refusal points at the journal runbook, not a guess", out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_wal_and_journal_sidecars_are_detected():
    tmp = Path(tempfile.mkdtemp(prefix="fsu-sidecars-"))
    try:
        db = tmp / "webui.db"
        _make_db(db)
        assert_eq(S.live_sidecars(str(db)), [], "no sidecars: none reported")
        (tmp / "webui.db-wal").write_bytes(b"")
        assert_true(len(S.live_sidecars(str(db))) == 1, "a -wal sidecar is detected")
        (tmp / "webui.db-wal").unlink()
        (tmp / "webui.db-journal").write_bytes(b"")
        assert_true(len(S.live_sidecars(str(db))) == 1, "a -journal sidecar is detected")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ===========================================================================
# dry run never writes anything
# ===========================================================================


def test_dry_run_writes_nothing():
    tmp = Path(tempfile.mkdtemp(prefix="fsu-dryrun-"))
    forensics = tmp / "forensics"
    try:
        db = tmp / "webui.db"
        _make_db(db)
        before = db.read_bytes()
        rc, out, err = run_script([str(db), "--json"], env={"FSU_FORENSICS_DIR": str(forensics)})
        assert_eq(rc, 3, f"dry run with pending work exits 3 (stderr={err!r})")
        after = db.read_bytes()
        assert_eq(after, before, "dry run wrote zero bytes to the db")
        assert_true(not forensics.exists(), "dry run took no backup")
        result = json.loads(out)
        assert_eq(result["in_scope_total"], 2, "dry run finds the 2 on-branch targets by default")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_chat_default_is_her_id():
    assert_eq(S.DEFAULT_CHAT_ID, "ea1494ea-e9d7-46fb-8b7c-3a50d685d00e", "--chat defaults to her conversation id")


if __name__ == "__main__":
    test_build_targets_categorizes_and_scopes_correctly()
    test_all_branches_widens_scope_but_never_touches_tip_or_inflight()
    test_apply_changes_only_targets_in_both_copies_and_preserves_content()
    test_apply_never_touches_tip_or_inflight_even_with_all_branches()
    test_backup_uses_forensics_dir_and_restore_is_md5_identical()
    test_restore_refuses_cleanly_when_no_backup_exists()
    test_refuses_apply_when_another_process_holds_the_db_open()
    test_refuses_apply_when_a_journal_sidecar_exists()
    test_wal_and_journal_sidecars_are_detected()
    test_dry_run_writes_nothing()
    test_chat_default_is_her_id()

    if _FAILS:
        print(f"\n{len(_FAILS)} FAILURE(S): {_FAILS}")
        sys.exit(1)
    print("\nAll fix-stale-unfinished.py script tests passed.")
