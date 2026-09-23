"""Tests for scripts/fix-encoded-messages.py — undoing the encoding damage
the retired v1 chat-tree repair left behind. See that script's own module
docstring for what changed from Andrew's original (origin/fix/ops-chat-tree
193813b, imported verbatim, then hardened in separate commits).

Run directly:
    python compactor/test_fix_encoded_messages_script.py
"""

import importlib.util
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SCRIPT = _HERE.parent / "scripts" / "fix-encoded-messages.py"
_spec = importlib.util.spec_from_file_location("fix_encoded_messages_script", _SCRIPT)
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)

_FAILS = []
CHAT_ID = "ea1494ea-e9d7-46fb-8b7c-3a50d685d00e"


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


def _mk_db(path, chat_id, history_messages, table_rows):
    """table_rows: {msg_id: raw_content_string_as_stored}"""
    con = sqlite3.connect(str(path))
    con.execute("create table chat (id text primary key, chat text, current_message_id text, updated_at int)")
    con.execute("create table chat_message (id text primary key, chat_id text, parent_id text, role text, "
                "content text, created_at int)")
    blob = json.dumps({"history": {"messages": history_messages, "currentId": None}, "messages": []})
    con.execute("insert into chat values (?,?,?,?)", (chat_id, blob, None, int(time.time())))
    prefix = chat_id + "-"
    for mid, content in table_rows.items():
        con.execute("insert into chat_message values (?,?,?,?,?,?)",
                    (prefix + mid, chat_id, None, "assistant", content, 0))
    con.commit()
    con.close()


def test_decode_once_only_accepts_str_deliberately():
    # Unlike repair-chat-tree.py's decode(), this script's decode_once
    # stays str-only ON PURPOSE (module docstring) -- a list-of-blocks
    # value was never damaged by v1's specific bug, so treating it as a
    # "target" here would be wrong.
    assert_eq(S.decode_once(json.dumps("hello")), "hello", "decode_once unwraps a plain string")
    assert_eq(S.decode_once(json.dumps([1, 2, 3])), None, "decode_once returns None for non-str JSON (deliberate)")
    assert_eq(S.decode_once("not json"), None, "decode_once returns None for non-JSON text")


def test_find_chat_row_is_exact_or_unambiguous_prefix_never_biggest():
    tmp = Path(tempfile.mkdtemp(prefix="fem-findchat-"))
    try:
        db = tmp / "webui.db"
        con = sqlite3.connect(str(db))
        con.execute("create table chat_message (id text primary key, chat_id text)")
        # a bigger clone with MORE rows than her real chat -- must not win
        for i in range(50):
            con.execute("insert into chat_message values (?,?)", (f"big-clone-{i}", "big-clone"))
        con.execute("insert into chat_message values ('her-1', ?)", (CHAT_ID,))
        con.commit()
        found = S.find_chat_row(con, CHAT_ID)
        assert_eq(found, CHAT_ID, "an exact --chat id wins even though another chat has far more rows")
        found_none = S.find_chat_row(con, "does-not-exist")
        assert_true(found_none is None, "no match -> None, not a crash")
        con.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_restores_a_message_v1_left_double_encoded_in_the_history():
    tmp = Path(tempfile.mkdtemp(prefix="fem-restore-"))
    try:
        pre = tmp / "pre.db"
        live = tmp / "live.db"
        good_text = "hello world"
        good_raw = json.dumps(good_text)   # what the table correctly stores
        # PRE-repair: message exists ONLY in the table (this is what v1 saw)
        _mk_db(pre, CHAT_ID, history_messages={}, table_rows={"m1": good_raw})
        # LIVE (post-v1-damage): v1 copied it into history, encoded, and a
        # browser save later encoded the table copy AGAIN too.
        double_encoded = json.dumps(good_raw)
        live_hist = {"m1": {"id": "m1", "role": "assistant", "content": good_raw, "parentId": None, "childrenIds": []}}
        _mk_db(live, CHAT_ID, history_messages=live_hist, table_rows={"m1": double_encoded})

        con = sqlite3.connect(str(live))
        plan, err = S.plan_fix(con, str(pre), CHAT_ID)
        assert_true(err is None, "plan_fix succeeds", err)
        assert_true("m1" in plan["fix_hist"], "the history copy is recognized as needing a restore")
        assert_true("m1" in plan["fix_table"], "the double-encoded table copy is recognized too")
        assert_true(plan["ok"], "verification against the pre-repair copy passes")
        con.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_not_yet_synced_table_only_message_is_not_a_false_positive():
    # H-VERIFY (bug found testing against the real 09-22/09-23 backups,
    # neither of which was ever actually touched by v1): a message that
    # is STILL table-only on the live side too (never copied into history
    # by anything) is not v1 damage -- it is repair-chat-tree.py's job,
    # not this script's, and must be classified "already correct", not a
    # verification failure.
    tmp = Path(tempfile.mkdtemp(prefix="fem-notsynced-"))
    try:
        pre = tmp / "pre.db"
        live = tmp / "live.db"
        good_raw = json.dumps("some text")
        _mk_db(pre, CHAT_ID, history_messages={}, table_rows={"m1": good_raw})
        # live: m1 is STILL not in history, and the table copy is unchanged
        _mk_db(live, CHAT_ID, history_messages={}, table_rows={"m1": good_raw})

        con = sqlite3.connect(str(live))
        plan, err = S.plan_fix(con, str(pre), CHAT_ID)
        assert_true(err is None, "plan_fix succeeds", err)
        assert_eq(plan["fix_hist"], [], "nothing to fix in history for a message v1 never touched")
        assert_eq(plan["fix_table"], [], "nothing to fix in the table either")
        assert_eq(plan["already_ok"], 1, "classified as already-correct, not a mismatch")
        assert_true(plan["ok"], "verification passes -- this is NOT a v1-damage case")
        con.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_message_changed_since_is_skipped_not_clobbered():
    tmp = Path(tempfile.mkdtemp(prefix="fem-skip-"))
    try:
        pre = tmp / "pre.db"
        live = tmp / "live.db"
        good_raw = json.dumps("original")
        _mk_db(pre, CHAT_ID, history_messages={}, table_rows={"m1": good_raw})
        # live: she (or something else) legitimately edited this message
        # to a THIRD value after the repair -- must be left alone, not
        # forced back to the pre-repair text.
        edited_hist = {"m1": {"id": "m1", "role": "assistant", "content": "a completely different edit",
                               "parentId": None, "childrenIds": []}}
        _mk_db(live, CHAT_ID, history_messages=edited_hist, table_rows={"m1": good_raw})

        con = sqlite3.connect(str(live))
        plan, err = S.plan_fix(con, str(pre), CHAT_ID)
        assert_true(err is None, "plan_fix succeeds", err)
        assert_true("m1" in plan["skipped"], "a message changed since the repair is skipped, not overwritten")
        assert_eq(plan["fix_hist"], [], "skipped message is not queued for a fix")
        con.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_chat_default_is_her_id():
    assert_eq(S.DEFAULT_CHAT_ID, CHAT_ID, "--chat defaults to her conversation, not the biggest chat_message count")


def test_backup_then_restore_round_trip():
    tmp = Path(tempfile.mkdtemp(prefix="fem-backup-"))
    try:
        db = tmp / "webui.db"
        _mk_db(db, CHAT_ID, {}, {})
        import hashlib
        before = hashlib.md5(db.read_bytes()).hexdigest()
        stamp = S.utc_stamp()
        bak = S.backup_db(str(db), stamp)
        db.write_bytes(b"corrupted")
        got, msg = S.restore_db(str(db), stamp)
        assert_true(got is not None, "restore succeeds", msg)
        after = hashlib.md5(db.read_bytes()).hexdigest()
        assert_eq(after, before, "restored file is byte-identical (md5)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_wal_and_journal_sidecars_detected():
    tmp = Path(tempfile.mkdtemp(prefix="fem-wal-"))
    try:
        db = tmp / "webui.db"
        _mk_db(db, CHAT_ID, {}, {})
        assert_eq(S.live_sidecars(str(db)), [], "no sidecars: none reported")
        (tmp / "webui.db-wal").write_bytes(b"")
        assert_true(len(S.live_sidecars(str(db))) == 1, "a -wal sidecar is detected")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ===========================================================================
# D3 -- the database-move rehearsal's defect: this script used to accept
# /data/openwebui/webui.db unconditionally, even once WEBUI_DB_LOCAL=true
# moves OpenWebUI's live database to local disk and leaves /data holding
# only a periodically-published snapshot. See scripts/_webui_live_path.py
# and dbmove/findings.md section 4. Only the LIVE positional argument is
# ever subject to this check -- `pre` is a read-only reference copy, never
# a write target.
# ===========================================================================


def _run(args, env):
    import subprocess
    full_env = dict(os.environ)
    full_env.update(env)
    r = subprocess.run([sys.executable, str(_SCRIPT), *args], capture_output=True, text=True, env=full_env)
    return r.returncode, r.stdout, r.stderr


def test_d3_apply_against_snapshot_in_local_mode_refuses():
    tmp = Path(tempfile.mkdtemp(prefix="fem-d3-refuse-"))
    try:
        pre = tmp / "pre.db"
        snapshot = tmp / "live.db"
        _mk_db(pre, CHAT_ID, {}, {})
        _mk_db(snapshot, CHAT_ID, {}, {})
        local = tmp / "local.db"
        rc, out, err = _run(
            [str(snapshot), str(pre), "--apply"],
            {"WEBUI_DB_LOCAL": "true", "WEBUI_SNAPSHOT_DB": str(snapshot), "WEBUI_LOCAL_DB": str(local)},
        )
        assert_eq(rc, 1, "D3: --apply against the snapshot in local mode refuses (exit 1)")
        assert_true("REFUSING" in err, "D3: refusal is printed", err)
        assert_true(str(local) in err, "D3: refusal names the live (local) path", err)
        assert_true("webuidb-sync" in err and "openwebui" in err,
                    "D3: refusal names both services to stop", err)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_d3_dry_run_against_snapshot_in_local_mode_warns_but_runs():
    tmp = Path(tempfile.mkdtemp(prefix="fem-d3-warn-"))
    try:
        pre = tmp / "pre.db"
        snapshot = tmp / "live.db"
        # find_chat_row derives the chat id from chat_message rows, so
        # each fixture needs at least one -- an already-healthy one
        # (present, decoded, matching on both sides) so this is a clean
        # no-op dry run, not a v1-damage case.
        good_raw = json.dumps("hi")
        _mk_db(pre, CHAT_ID, history_messages={}, table_rows={"m1": good_raw})
        _mk_db(snapshot, CHAT_ID, history_messages={"m1": {"id": "m1", "role": "assistant", "content": "hi",
                                                            "parentId": None, "childrenIds": []}},
               table_rows={"m1": good_raw})
        local = tmp / "local.db"
        rc, out, err = _run(
            [str(snapshot), str(pre)],
            {"WEBUI_DB_LOCAL": "true", "WEBUI_SNAPSHOT_DB": str(snapshot), "WEBUI_LOCAL_DB": str(local)},
        )
        assert_eq(rc, 0, "D3: a read-only dry run against the snapshot in local mode still runs")
        assert_true("WARNING" in err, "D3: dry run warns rather than refusing", err)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_d3_local_mode_off_snapshot_path_behaves_as_before():
    tmp = Path(tempfile.mkdtemp(prefix="fem-d3-off-"))
    try:
        pre = tmp / "pre.db"
        snapshot = tmp / "live.db"
        good_raw = json.dumps("hi")
        _mk_db(pre, CHAT_ID, history_messages={}, table_rows={"m1": good_raw})
        _mk_db(snapshot, CHAT_ID, history_messages={"m1": {"id": "m1", "role": "assistant", "content": "hi",
                                                            "parentId": None, "childrenIds": []}},
               table_rows={"m1": good_raw})
        rc, out, err = _run(
            [str(snapshot), str(pre)],
            {"WEBUI_DB_LOCAL": "false", "WEBUI_SNAPSHOT_DB": str(snapshot)},
        )
        assert_eq(rc, 0, "D3: WEBUI_DB_LOCAL=false: a dry run against /data works exactly as before")
        assert_true("REFUSING" not in err, "D3: no D3 refusal when local mode is off", err)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_d3_apply_against_live_local_path_succeeds_and_prints_sync_hint():
    tmp = Path(tempfile.mkdtemp(prefix="fem-d3-hint-"))
    try:
        pre = tmp / "pre.db"
        local = tmp / "live.db"
        good_text = "hello world"
        good_raw = json.dumps(good_text)
        _mk_db(pre, CHAT_ID, history_messages={}, table_rows={"m1": good_raw})
        live_hist = {"m1": {"id": "m1", "role": "assistant", "content": good_raw,
                             "parentId": None, "childrenIds": []}}
        double_encoded = json.dumps(good_raw)
        _mk_db(local, CHAT_ID, history_messages=live_hist, table_rows={"m1": double_encoded})
        rc, out, err = _run(
            [str(local), str(pre), "--apply"],
            {"WEBUI_DB_LOCAL": "true", "WEBUI_LOCAL_DB": str(local),
             "WEBUI_SNAPSHOT_DB": str(tmp / "snapshot-not-used.db"),
             "WEBUI_DB_FORENSICS": str(tmp / "forensics")},
        )
        assert_eq(rc, 0, "D3: --apply against the LIVE local path succeeds")
        assert_true("supervisorctl stop openwebui webuidb-sync" in out,
                    "D3: success prints the final-sync stop command", out)
        assert_true("--sync-once --force" in out, "D3: success prints the final-sync command itself", out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_decode_once_only_accepts_str_deliberately()
    test_find_chat_row_is_exact_or_unambiguous_prefix_never_biggest()
    test_restores_a_message_v1_left_double_encoded_in_the_history()
    test_not_yet_synced_table_only_message_is_not_a_false_positive()
    test_message_changed_since_is_skipped_not_clobbered()
    test_chat_default_is_her_id()
    test_backup_then_restore_round_trip()
    test_wal_and_journal_sidecars_detected()

    test_d3_apply_against_snapshot_in_local_mode_refuses()
    test_d3_dry_run_against_snapshot_in_local_mode_warns_but_runs()
    test_d3_local_mode_off_snapshot_path_behaves_as_before()
    test_d3_apply_against_live_local_path_succeeds_and_prints_sync_hint()

    if _FAILS:
        print(f"\n{len(_FAILS)} FAILURE(S): {_FAILS}")
        sys.exit(1)
    print("\nAll fix-encoded-messages.py script tests passed.")
