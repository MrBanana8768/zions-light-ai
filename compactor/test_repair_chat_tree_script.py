"""Tests for scripts/repair-chat-tree.py — hostile-testing the relinking,
the pointer rule (H-PTR), the decode fix (H-DEC), and the house-style CLI
(liveness refusal, backup/restore, exit codes) brought to this script in
this pass. See that script's own module docstring for what was verified
against the pinned image's real source and what changed from Andrew's v2
(origin/fix/ops-chat-tree 193813b).

Run directly:
    python compactor/test_repair_chat_tree_script.py
"""

import importlib.util
import json
import os
import shutil
import sqlite3
import stat
import sys
import tempfile
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SCRIPT = _HERE.parent / "scripts" / "repair-chat-tree.py"
_spec = importlib.util.spec_from_file_location("repair_chat_tree_script", _SCRIPT)
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
# helpers to build synthetic message trees
# ===========================================================================


def msg(mid, role, ts, parent=None, children=None, content="x"):
    return mid, {"id": mid, "role": role, "parentId": parent,
                 "childrenIds": list(children or []), "timestamp": ts, "content": content}


def tree(*entries):
    return dict(entries)


def link(msgs):
    """Recompute childrenIds from parentId, like OpenWebUI's own
    ChatMessageTable.get_messages_map_by_chat_id does for the table."""
    for m in msgs.values():
        m["childrenIds"] = []
    for mid, m in msgs.items():
        p = m.get("parentId")
        if p in msgs and mid not in msgs[p]["childrenIds"]:
            msgs[p]["childrenIds"].append(mid)
    return msgs


# ===========================================================================
# H-DEC — decode() handles list/dict content, not just str
# ===========================================================================


def test_decode_plain_text():
    assert_eq(S.decode(json.dumps("hello")), "hello", "decode: plain text unwraps once")


def test_decode_list_of_blocks():
    blocks = [{"type": "text", "text": "hi"}, {"type": "image_url", "image_url": {"url": "x"}}]
    assert_eq(S.decode(json.dumps(blocks)), blocks, "decode: list-of-blocks content unwraps (H-DEC, v2 left this encoded)")


def test_decode_non_json_text_is_returned_unchanged():
    assert_eq(S.decode("not json { at all"), "not json { at all", "decode: non-JSON text passes through untouched")


def test_decode_non_string_input_passed_through():
    assert_eq(S.decode(["already", "a", "list"]), ["already", "a", "list"], "decode: non-str input untouched")


# ===========================================================================
# relinking — table-link preference, ties, missing-grandparent, forks, cycles
# ===========================================================================


def test_relink_prefers_table_link_when_it_resolves():
    msgs = link(tree(
        msg("root", "user", 0),
        msg("a1", "assistant", 10, parent="root"),
        # "orphan" has a broken parentId in the JSON, but the table still
        # remembers the real link (root/a1's era) -- OpenWebUI's own
        # `chat_message` fast path is the AUTHORITATIVE copy (module
        # docstring, point 1), so this must win over the time heuristic.
        msg("orphan", "user", 20, parent="missing-parent"),
    ))
    table = {"orphan": {"parentId": "a1"}}
    plan, msgs2 = S.build_plan({"history": {"messages": msgs}, "messages": []}, table, None)
    assert_eq(plan.relinks.get("orphan"), "a1", "table link used when it resolves cleanly")
    assert_eq(plan.relink_src.get("orphan"), "table", "table link recorded as the source")


def test_relink_table_link_into_own_subtree_is_rejected():
    # orphan's table-recorded parent is its OWN descendant -- using it
    # would create a cycle. Must fall back to the time heuristic instead.
    msgs = link(tree(
        msg("root", "user", 0),
        msg("u1", "user", 30),           # candidate real parent by time
        msg("orphan", "assistant", 40, parent="gone"),
        msg("child", "user", 50, parent="orphan"),
    ))
    table = {"orphan": {"parentId": "child"}}  # points into its own subtree
    plan, msgs2 = S.build_plan({"history": {"messages": msgs}, "messages": []}, table, None)
    assert_eq(plan.relink_src.get("orphan"), "time",
              "a table link into the orphan's own subtree is rejected, not used")
    assert_true(plan.relinks.get("orphan") != "child", "did not attach orphan under its own descendant")


def test_relink_ties_at_the_same_second_prefer_larger_subtree():
    # Two opposite-role candidates tied at the exact same timestamp as
    # each other; the one with the larger existing subtree (more likely
    # the real thread than a stray duplicate) should win.
    msgs = link(tree(
        msg("root", "user", 0),
        msg("big", "assistant", 100),
        msg("big_child", "user", 110, parent="big"),
        msg("small", "assistant", 100),          # tied with "big"
        msg("orphan", "user", 150, parent="gone"),
    ))
    table = {}
    plan, msgs2 = S.build_plan({"history": {"messages": msgs}, "messages": []}, table, None)
    assert_eq(plan.relinks.get("orphan"), "big", "tie at the same second broken toward the larger subtree")


def test_relink_orphan_whose_true_parent_is_also_missing():
    # Real-data case (both 09-22 and 09-23 pod-export backups): TWO
    # orphans declare the identical missing parent id, which does not
    # exist anywhere. Each must fall back to the time heuristic
    # independently, and must not accidentally point at each other in a
    # way that creates a cycle.
    msgs = link(tree(
        msg("root", "user", 0),
        msg("u_before", "user", 100),
        msg("a_before", "assistant", 110, parent="u_before"),
        msg("orphan_u", "user", 200, parent="ghost-parent"),
        msg("orphan_a", "assistant", 250, parent="ghost-parent"),
    ))
    table = {"orphan_u": {"parentId": "ghost-parent"}, "orphan_a": {"parentId": "ghost-parent"}}
    plan, msgs2 = S.build_plan({"history": {"messages": msgs}, "messages": []}, table, None)
    assert_true("orphan_u" in plan.relinks, "orphan_u relinked despite its true (missing) parent")
    assert_true("orphan_a" in plan.relinks, "orphan_a relinked despite its true (missing) parent")
    assert_eq(S.find_any_cycle(msgs2), None, "no cycle from two orphans sharing one missing parent")


def test_relink_orphan_subtree_with_a_fork_moves_as_one_unit():
    # The orphan subtree ROOT has two children (a fork) below it. Only the
    # root needs relinking; the fork must survive intact underneath it.
    msgs = link(tree(
        msg("root", "user", 0),
        msg("u1", "user", 50),
        msg("orphan_root", "assistant", 60, parent="gone"),
        msg("fork_a", "user", 70, parent="orphan_root"),
        msg("fork_b", "user", 71, parent="orphan_root"),
    ))
    table = {}
    plan, msgs2 = S.build_plan({"history": {"messages": msgs}, "messages": []}, table, None)
    assert_true("orphan_root" in plan.relinks, "the fork's subtree root got relinked")
    assert_eq(set(msgs2["orphan_root"]["childrenIds"]), {"fork_a", "fork_b"},
              "the fork's own two children survive under the reattached root")
    assert_eq(msgs2["fork_a"]["parentId"], "orphan_root", "fork_a still a child of orphan_root")
    assert_eq(msgs2["fork_b"]["parentId"], "orphan_root", "fork_b still a child of orphan_root")


def test_relink_never_produces_a_cycle_even_with_adversarial_table_links():
    # Three orphans whose TABLE links point in a ring (A->B->C->A). None
    # of them can be used as recorded; the algorithm must fall back to
    # the time heuristic for all three without ever creating the ring.
    msgs = link(tree(
        msg("root", "user", 0),
        msg("anchor", "assistant", 10),
        msg("a", "user", 20, parent="gone"),
        msg("b", "assistant", 21, parent="gone"),
        msg("c", "user", 22, parent="gone"),
    ))
    table = {"a": {"parentId": "b"}, "b": {"parentId": "c"}, "c": {"parentId": "a"}}
    plan, msgs2 = S.build_plan({"history": {"messages": msgs}, "messages": []}, table, None)
    assert_eq(S.find_any_cycle(msgs2), None, "a ring of table links never becomes a real cycle")


def test_find_any_cycle_detects_a_pre_existing_cycle_untouched_by_relinking():
    # A cycle among ALREADY-linked messages (no orphans at all -- every
    # node has a parentId present in msgs) is invisible to the orphan
    # scan, but find_any_cycle must still catch it directly.
    msgs = tree(
        msg("x", "user", 0, parent="y"),
        msg("y", "assistant", 1, parent="x"),
    )
    assert_true(S.find_any_cycle(msgs) is not None, "a 2-node mutual-parent cycle is detected globally")


# ===========================================================================
# current_message_id column vs history.currentId disagreement
# ===========================================================================


def test_pick_stored_pointer_prefers_column_when_it_resolves():
    msgs = {"a": {}, "b": {}}
    assert_eq(S.pick_stored_pointer("a", "b", msgs), "a",
              "chat.current_message_id column wins over history.currentId (routers/chats.py:1274)")


def test_pick_stored_pointer_falls_back_when_column_points_nowhere():
    msgs = {"b": {}}
    assert_eq(S.pick_stored_pointer("does-not-exist", "b", msgs), "b",
              "falls back to history.currentId when the column value does not resolve (H-PTR2, v2 did not)")


def test_pick_stored_pointer_none_when_neither_resolves():
    # Neither value resolves to a real message. pick_stored_pointer still
    # returns SOMETHING (the raw column value, for reporting) rather than
    # None -- callers (build_plan) already treat "orig not in msgs" as
    # broken regardless of which non-resolving value comes back, and only
    # a genuinely empty pair (None, None) reports None itself.
    msgs = {"other": {}}
    assert_eq(S.pick_stored_pointer("gone", "also-gone", msgs), "gone",
              "neither resolves: the raw column value is still reported, not silently discarded")
    assert_eq(S.pick_stored_pointer(None, None, msgs), None,
              "truly nothing stored: reported as None, not a crash")


# ===========================================================================
# H-PTR — the pointer rule itself
# ===========================================================================


def test_pointer_kept_when_she_deliberately_navigated_back_and_stopped():
    # The exact scenario from the task brief: she gets A1, regenerates a
    # NEWER sibling A1' she doesn't like, clicks back to A1, and does not
    # continue. v2 always recomputed "newest leaf" and would have jumped
    # her onto A1' (newer, but abandoned). Nothing here is broken -- no
    # orphans -- so the pointer must not move at all.
    msgs = link(tree(
        msg("root", "user", 0),
        msg("q1", "user", 10, parent="root"),
        msg("a1", "assistant", 20, parent="q1"),     # her real, deliberate position
        msg("a1p", "assistant", 50, parent="q1"),    # abandoned regenerate, NEWER
    ))
    plan, msgs2 = S.build_plan({"history": {"messages": msgs}, "messages": []}, {}, "a1")
    assert_eq(plan.new_pointer, "a1", "pointer stays on her deliberate position, not the newer abandoned sibling")
    assert_true(not plan.pointer_moved, "pointer_moved is False — this is the whole point of H-PTR")


def test_pointer_walks_forward_when_the_live_tab_kept_going_stale_tab_case():
    # The genuine stale-tab bug: the stored pointer is old but fully
    # intact, and the conversation for real continued past it (children
    # exist). Must walk FORWARD along that lineage to the true tip --
    # never sideways to an unrelated branch.
    msgs = link(tree(
        msg("root", "user", 0),
        msg("old", "assistant", 20),          # the stale pointer
        msg("q2", "user", 30, parent="old"),
        msg("a2", "assistant", 40, parent="q2"),   # the true current tip
    ))
    plan, msgs2 = S.build_plan({"history": {"messages": msgs}, "messages": []}, {}, "old")
    assert_eq(plan.new_pointer, "a2", "stale-but-intact pointer walks forward to the true tip")
    assert_true(plan.pointer_moved, "pointer_moved is True for the genuine stale-tab case")


def test_pointer_never_regresses_below_a_broken_pointers_own_timestamp():
    # The stored pointer itself is missing entirely (genuinely broken,
    # not just stale). The newest-leaf fallback must not pick something
    # OLDER than where the broken pointer itself was.
    msgs = link(tree(
        msg("root", "user", 0),
        msg("mid", "assistant", 500),          # roughly where the broken pointer was
        msg("old_leaf", "assistant", 100),     # older leaf elsewhere -- must not be picked
        msg("new_leaf", "assistant", 900, parent="mid"),
    ))
    plan, msgs2 = S.build_plan({"history": {"messages": msgs}, "messages": []}, {}, "totally-gone")
    # the broken pointer had no resolvable timestamp of its own here, so
    # this exercises the "no anchor at all" branch -- newest leaf wins.
    assert_eq(plan.new_pointer, "new_leaf", "with no anchor at all, the newest leaf overall is chosen")


def test_forward_tip_prefers_assistant_on_an_exact_timestamp_tie():
    # a_child is inserted (and so linked as a child) BEFORE u_child, so a
    # stable sort on timestamp alone (dropping the role tiebreak) would
    # pick u_child (last in list order) -- only the real role preference
    # picks a_child here, which is what actually distinguishes this from
    # the mutant that drops it.
    msgs = link(tree(
        msg("root", "user", 0),
        msg("a_child", "assistant", 100, parent="root"),   # tied with u_child
        msg("u_child", "user", 100, parent="root"),        # tied with a_child
    ))
    assert_eq(S.forward_tip(msgs, "root"), "a_child",
              "forward_tip prefers the assistant reply on an exact timestamp tie (mutation M4)")


def test_pointer_not_regressed_when_a_newer_candidate_exists():
    # The common, verification-passing case: a newer, properly-reachable
    # candidate exists, so pointer_regressed must stay False.
    msgs = link(tree(
        msg("root", "user", 0),
        msg("broken_ptr", "user", 400, parent="ghost"),
        msg("new_leaf", "assistant", 900, parent="root"),
    ))
    plan, msgs2 = S.build_plan({"history": {"messages": msgs}, "messages": []}, {}, "broken_ptr")
    assert_true(not plan.pointer_regressed, "not flagged when the chosen leaf is newer than the broken pointer")


def test_dangling_child_ids_are_actually_removed():
    msgs = tree(
        msg("root", "user", 0, children=["real_child", "ghost_child"]),
        msg("real_child", "assistant", 10, parent="root"),
        # "ghost_child" is listed in root's childrenIds but does not exist
    )
    plan, msgs2 = S.build_plan({"history": {"messages": msgs}, "messages": []}, {}, "real_child")
    assert_true(plan.dangling_removed >= 1, "a dangling child id is counted as removed (mutation M9)")
    assert_true("ghost_child" not in msgs2["root"]["childrenIds"],
                "the dangling id is actually gone from childrenIds, not just counted")


def test_true_root_parent_id_is_forced_to_none():
    # true_root is chosen as the EARLIEST orphan by timestamp, but it may
    # itself carry a stale non-None parentId (e.g. pointing at something
    # that no longer exists) -- it must be cleared, not left dangling.
    msgs = tree(
        msg("earliest", "user", 0, parent="something-that-does-not-exist"),
        msg("later", "assistant", 100, parent="earliest"),
    )
    plan, msgs2 = S.build_plan({"history": {"messages": msgs}, "messages": []}, {}, "later")
    assert_eq(plan.true_root, "earliest", "the earliest orphan is chosen as true_root (sanity check)")
    assert_eq(msgs2["earliest"]["parentId"], None, "true_root's parentId is forced to None (mutation M10)")


def test_pointer_regression_guard_with_a_resolvable_but_broken_pointer():
    # broken_ptr EXISTS and is a genuine orphan (role "user", so the
    # general relink pass -- which would otherwise happily reattach it to
    # "root", the only earlier opposite-role message -- has no assistant
    # candidate before it and leaves it unlinkable: its walk stays broken
    # even after the relink pass runs, unlike a would-be orphan the
    # general pass fixes for free (that case is exercised by
    # test_pointer_kept_when_she_deliberately... via H-PTR's "keep the
    # SAME id once its walk is fixed" path, not this one).
    msgs = link(tree(
        msg("root", "user", 0),
        msg("broken_ptr", "user", 400, parent="ghost"),   # unlinkable: no assistant before it
        msg("new_leaf", "assistant", 900, parent="root"),
    ))
    plan, msgs2 = S.build_plan({"history": {"messages": msgs}, "messages": []}, {}, "broken_ptr")
    assert_true("broken_ptr" in plan.unlinkable, "broken_ptr genuinely could not be relinked (sanity check)")
    assert_true(plan.new_pointer != None, "a broken pointer with no earlier candidate still resolves to something")
    assert_eq(plan.new_pointer, "new_leaf", "newest-at-or-after-the-broken-pointer's-time wins")


# ===========================================================================
# idempotency
# ===========================================================================


def test_second_pass_over_an_already_repaired_tree_is_a_clean_noop():
    msgs = link(tree(
        msg("root", "user", 0),
        msg("old", "assistant", 20),
        msg("q2", "user", 30, parent="old"),
        msg("a2", "assistant", 40, parent="q2"),
    ))
    chat1 = {"history": {"messages": msgs}, "messages": []}
    plan1, msgs_after = S.build_plan(chat1, {}, "old")
    assert_true(plan1.pointer_moved, "first pass moves the pointer forward")

    chat2 = {"history": {"messages": msgs_after}, "messages": []}
    plan2, msgs_after2 = S.build_plan(chat2, {}, plan1.new_pointer)
    assert_true(not plan2.relinks and not plan2.unlinkable, "second pass finds nothing left to relink")
    assert_true(not plan2.pointer_moved, "second pass does not move an already-correct pointer")
    assert_true(plan2.ok, "second pass verifies clean")


# ===========================================================================
# CLI-level: liveness refusal, backup/restore, exit codes, --chat default
# ===========================================================================


def _make_db(path, chat_id="ea1494ea-e9d7-46fb-8b7c-3a50d685d00e", extra_chat=None):
    con = sqlite3.connect(str(path))
    con.execute("create table chat (id text primary key, chat text, current_message_id text, updated_at int)")
    con.execute("create table chat_message (id text primary key, chat_id text, parent_id text, role text, "
                "content text, created_at int)")
    msgs = link(tree(
        msg("root", "user", 0),
        msg("old", "assistant", 20),
        msg("q2", "user", 30, parent="old"),
        msg("a2", "assistant", 40, parent="q2"),
    ))
    blob = json.dumps({"history": {"messages": msgs, "currentId": "old"}, "messages": []})
    con.execute("insert into chat values (?,?,?,?)", (chat_id, blob, "old", int(time.time())))
    if extra_chat:
        con.execute("insert into chat values (?,?,?,?)", (extra_chat, json.dumps({"history": {"messages": {}}}), None, int(time.time())))
    con.commit()
    con.close()


def test_chat_default_is_her_id_not_largest():
    assert_eq(S.DEFAULT_CHAT_ID, "ea1494ea-e9d7-46fb-8b7c-3a50d685d00e",
              "--chat defaults to her conversation id, not a size-based guess")


def test_liveness_refusal_when_another_process_holds_the_file_open():
    # openers() deliberately excludes the CALLER's own pid (so a script
    # inspecting the db under its own read doesn't refuse itself) --
    # so proving it catches an opener needs an ACTUAL other process.
    import subprocess
    tmp = Path(tempfile.mkdtemp(prefix="rct-openers-"))
    try:
        db = tmp / "webui.db"
        _make_db(db)
        proc = subprocess.Popen([sys.executable, "-c",
                                  f"import time; f=open(r'{db}','rb'); time.sleep(5)"])
        try:
            deadline = time.time() + 3
            found = []
            while time.time() < deadline and not found:
                found = S.openers(str(db))
            assert_true(str(proc.pid) in found, "openers() finds the OTHER process holding the file open", str(found))
            assert_true(str(os.getpid()) not in found, "openers() excludes the caller's own pid")
        finally:
            proc.kill()
            proc.wait()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_wal_and_journal_sidecars_are_detected():
    tmp = Path(tempfile.mkdtemp(prefix="rct-wal-"))
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


def test_backup_then_restore_is_byte_identical():
    tmp = Path(tempfile.mkdtemp(prefix="rct-backup-"))
    try:
        db = tmp / "webui.db"
        _make_db(db)
        import hashlib
        before = hashlib.md5(db.read_bytes()).hexdigest()
        stamp = S.utc_stamp()
        bak = S.backup_db(str(db), stamp)
        assert_true(bak.exists(), "backup file created")
        db.write_bytes(b"corrupted!!")  # simulate a bad write
        got_bak, msg = S.restore_db(str(db), stamp)
        assert_true(got_bak is not None, "restore reports success", msg)
        after = hashlib.md5(db.read_bytes()).hexdigest()
        assert_eq(after, before, "restored file is byte-identical (md5) to the pre-backup original")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_restore_refuses_cleanly_when_no_backup_exists():
    tmp = Path(tempfile.mkdtemp(prefix="rct-norestore-"))
    try:
        db = tmp / "webui.db"
        _make_db(db)
        bak, msg = S.restore_db(str(db), "20000101T000000Z")
        assert_true(bak is None, "restore with a nonexistent stamp refuses rather than crashing")
        assert_true("REFUSED" in msg, "refusal message says so", msg)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_find_chat_requires_a_single_unambiguous_match():
    tmp = Path(tempfile.mkdtemp(prefix="rct-findchat-"))
    try:
        db = tmp / "webui.db"
        _make_db(db, extra_chat="ea14other-not-a-clash")
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        found = S.find_chat(con, "ea1494ea-e9d7-46fb-8b7c-3a50d685d00e")
        assert_true(found is not None, "exact id matches")
        found_bad = S.find_chat(con, "does-not-exist")
        assert_true(found_bad is None, "no match -> None, not a crash")
        con.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ===========================================================================
# --sync-json-only (PART 3B) -- the conservative mode
# ===========================================================================


def trow(parent, role, created, output=None, done=None, model_id=None, content="x"):
    """A synthetic chat_message table row shaped like load_table()'s output."""
    return {"parentId": parent, "role": role, "content": json.dumps(content), "created": created,
            "output": json.dumps(output) if output is not None else None,
            "done": done, "model_id": model_id}


def test_sync_fills_a_mid_branch_gap():
    # root -> [gap: missing from JSON, present in table] -> tip (in JSON,
    # parentId points at the gap). Before sync, "tip" is an orphan (its
    # parentId isn't in msgs at all); after, the gap message is copied in
    # verbatim from the table and the walk reconnects.
    msgs = link(tree(
        msg("root", "user", 0),
        msg("tip", "assistant", 20, parent="gap"),  # "gap" not in msgs yet
    ))
    table = {
        "root": trow(None, "user", 0),
        "gap": trow("root", "user", 10),
        "tip": trow("gap", "assistant", 20),
    }
    plan, msgs2, pre = S.build_sync_plan({"history": {"messages": msgs}, "messages": []}, table, "tip")
    assert_true("gap" in plan.added, "the mid-branch gap message was copied into the JSON")
    assert_eq(msgs2["gap"]["parentId"], "root", "gap's parentId copied verbatim from the table")
    assert_eq(msgs2["tip"]["parentId"], "gap", "tip's own parentId is untouched (never relinked)")
    assert_true("gap" in msgs2["root"]["childrenIds"], "root's childrenIds gained the gap id")
    assert_true("tip" in msgs2["gap"]["childrenIds"], "gap's own childrenIds computed from the table")
    assert_true(plan.ok, "verification passes once the gap is filled")
    assert_eq(plan.json_path_len, 3, "JSON walk from the pointer now reaches all 3 messages")


def test_sync_table_only_chain_of_two_or_more():
    # A whole continuation -- TWO consecutive messages -- exists only in
    # the table, past the JSON's current tip.
    msgs = link(tree(
        msg("root", "user", 0),
        msg("known_tip", "assistant", 10, parent="root"),
    ))
    table = {
        "root": trow(None, "user", 0),
        "known_tip": trow("root", "assistant", 10),
        "cont_q": trow("known_tip", "user", 20),
        "cont_a": trow("cont_q", "assistant", 30),
    }
    plan, msgs2, pre = S.build_sync_plan({"history": {"messages": msgs}, "messages": []}, table, "cont_a")
    assert_eq(set(plan.added), {"cont_q", "cont_a"}, "both table-only continuation messages copied")
    assert_eq(msgs2["cont_a"]["parentId"], "cont_q", "chain order preserved")
    assert_eq(msgs2["cont_q"]["parentId"], "known_tip", "chain reattaches to the known tip verbatim")
    assert_true("cont_q" in msgs2["known_tip"]["childrenIds"], "known_tip's childrenIds appended with cont_q")
    assert_true(plan.ok, "verification passes for a 2-message table-only chain")


def test_sync_output_field_is_copied():
    output_blocks = [{"type": "message", "role": "assistant",
                       "content": [{"type": "output_text", "text": "hi there"}]}]
    msgs = link(tree(msg("root", "user", 0)))
    table = {
        "root": trow(None, "user", 0),
        "a1": trow("root", "assistant", 10, output=output_blocks, done=True, model_id="some/model"),
    }
    plan, msgs2, pre = S.build_sync_plan({"history": {"messages": msgs}, "messages": []}, table, "a1")
    assert_eq(msgs2["a1"]["output"], output_blocks,
               "output column copied and decoded (rendering reads output[].content[].text)")
    assert_eq(msgs2["a1"]["done"], True, "done copied as a bool")
    assert_eq(msgs2["a1"]["model"], "some/model", "model set from model_id")


def test_sync_output_absent_when_column_is_null():
    msgs = link(tree(msg("root", "user", 0)))
    table = {"root": trow(None, "user", 0), "a1": trow("root", "assistant", 10)}  # output=None
    plan, msgs2, pre = S.build_sync_plan({"history": {"messages": msgs}, "messages": []}, table, "a1")
    assert_true("output" not in msgs2["a1"], "no output field fabricated when the column is null")
    assert_true("model" not in msgs2["a1"], "no model field fabricated when model_id is null")


def test_sync_never_relinks_an_unrelated_orphan():
    # "orphan" has a broken parentId that the table does NOT explain (its
    # own table row, if it existed, would show a different, resolvable
    # parent) -- this mode must leave it exactly as broken as it found it,
    # unlike build_plan's relinking pass.
    msgs = link(tree(
        msg("root", "user", 0),
        msg("orphan", "user", 50, parent="missing-parent-not-in-table-either"),
    ))
    table = {"root": trow(None, "user", 0)}  # orphan has no table row at all here
    plan, msgs2, pre = S.build_sync_plan({"history": {"messages": msgs}, "messages": []}, table, "root")
    assert_eq(msgs2["orphan"]["parentId"], "missing-parent-not-in-table-either",
              "the orphan's parentId is byte-for-byte untouched -- never relinked")
    assert_eq(plan.added, [], "nothing was added (orphan already existed in the JSON)")


def test_sync_pointer_unchanged_when_column_and_json_already_agree():
    msgs = link(tree(msg("root", "user", 0), msg("a1", "assistant", 10, parent="root")))
    chat = {"history": {"messages": msgs, "currentId": "a1"}, "messages": []}
    plan, msgs2, pre = S.build_sync_plan(chat, {"root": trow(None, "user", 0), "a1": trow("root", "assistant", 10)}, "a1")
    assert_true(not plan.pointer_aligned, "no alignment needed -- column and history.currentId already agree")
    assert_eq(plan.pointer_after, "a1", "history.currentId left exactly as it was")


def test_sync_pointer_aligned_to_column_when_they_disagree_and_column_resolves():
    # history.currentId is stale ("old"); chat.current_message_id ("new")
    # is the authoritative column (module docstring #3) and DOES resolve in
    # the table -- the one narrow exception fires, and ONLY history.currentId
    # moves (chat.current_message_id is never written by this mode at all).
    msgs = link(tree(msg("root", "user", 0), msg("old", "user", 5, parent="root"),
                      msg("new", "assistant", 10, parent="root")))
    chat = {"history": {"messages": msgs, "currentId": "old"}, "messages": []}
    table = {"root": trow(None, "user", 0), "old": trow("root", "user", 5), "new": trow("root", "assistant", 10)}
    plan, msgs2, pre = S.build_sync_plan(chat, table, "new")
    assert_true(plan.pointer_aligned, "history.currentId aligned to the authoritative column")
    assert_eq(plan.pointer_after, "new", "history.currentId now equals chat.current_message_id")
    assert_eq(plan.pointer_col, "new", "the column value itself is only ever read, reported back unchanged")


def test_sync_pointer_not_aligned_when_column_does_not_resolve_in_table():
    # The exception's own guard: chat.current_message_id disagrees with
    # history.currentId, but the column's value isn't a real table row --
    # must NOT align to a value that doesn't resolve.
    msgs = link(tree(msg("root", "user", 0), msg("old", "user", 5, parent="root")))
    chat = {"history": {"messages": msgs, "currentId": "old"}, "messages": []}
    table = {"root": trow(None, "user", 0), "old": trow("root", "user", 5)}
    plan, msgs2, pre = S.build_sync_plan(chat, table, "does-not-exist-in-table")
    assert_true(not plan.pointer_aligned, "guard holds: column value must resolve in the table to be used")
    assert_eq(plan.pointer_after, "old", "history.currentId left alone when the column doesn't resolve")


def test_sync_idempotent_second_run_reports_nothing_to_do():
    msgs = link(tree(msg("root", "user", 0)))
    table = {"root": trow(None, "user", 0), "a1": trow("root", "assistant", 10)}
    chat1 = {"history": {"messages": msgs, "currentId": "a1"}, "messages": []}
    plan1, msgs_after, pre1 = S.build_sync_plan(chat1, table, "a1")
    assert_true(plan1.added, "first run copies the table-only message")

    chat2 = {"history": {"messages": msgs_after, "currentId": plan1.pointer_after}, "messages": []}
    plan2, msgs_after2, pre2 = S.build_sync_plan(chat2, table, "a1")
    assert_eq(plan2.added, [], "second run finds nothing left to copy")
    assert_true(not plan2.pointer_aligned, "second run does not touch an already-aligned pointer")
    assert_true(plan2.ok, "second run still verifies clean")


def test_sync_refuses_when_result_would_not_match_table_branch():
    # The pointer's table ancestor chain includes an id that genuinely does
    # not exist ANYWHERE (not in the table either) -- copying what the
    # table CAN supply still leaves the JSON walk short of the table walk,
    # so verification must report not-ok rather than pretend success.
    msgs = link(tree(msg("root", "user", 0)))
    table = {
        # "root"'s own chain is fine, but "tip"'s table parent is a ghost
        # that the table itself never records -- nothing this mode does
        # can bridge that, so the two walks cannot end up equal.
        "root": trow(None, "user", 0),
        "tip": trow("ghost-nowhere", "assistant", 10),
    }
    plan, msgs2, pre = S.build_sync_plan({"history": {"messages": msgs}, "messages": []}, table, "tip")
    assert_true(not plan.ok, "verification correctly refuses: JSON and table walks cannot be made to match")
    assert_true(not plan.matches_table or not plan.reaches_root, "at least one of (a)/(b) fails as expected")


def test_sync_field_audit_reports_what_the_table_cannot_supply():
    msgs = link(tree(
        msg("root", "user", 0),
        msg("a1", "assistant", 10, parent="root"),
    ))
    msgs["root"]["models"] = ["some/model"]           # real field the table can't reliably reconstruct
    msgs["a1"]["followUps"] = ["ok?"]                  # real field with no chat_message equivalent
    msgs["a1"]["modelIdx"] = 0                          # ditto
    table = {"root": trow(None, "user", 0), "a1": trow("root", "assistant", 10)}
    plan, msgs2, pre = S.build_sync_plan({"history": {"messages": msgs}, "messages": []}, table, "a1")
    assert_eq(set(plan.field_audit["assistant"]), {"followUps", "modelIdx"},
              "assistant-only fields the table cannot supply are reported, not fabricated")
    assert_eq(set(plan.field_audit["user"]), {"models"},
              "the user message's 'models' list is reported as unsuppliable, not guessed from model_id")


def test_sync_existing_parent_childrenids_is_a_pure_append():
    # An existing parent already has a childrenIds entry recorded in some
    # order this script must not disturb; the new id must be appended, not
    # used to rebuild/reorder the list.
    msgs = link(tree(msg("root", "user", 0), msg("z_existing", "assistant", 5, parent="root")))
    msgs["root"]["childrenIds"] = ["z_existing"]  # deliberately not alphabetical/timestamp order
    table = {"root": trow(None, "user", 0), "z_existing": trow("root", "assistant", 5),
             "new_sibling": trow("root", "assistant", 1)}  # earlier timestamp than z_existing
    plan, msgs2, pre = S.build_sync_plan({"history": {"messages": msgs}, "messages": []}, table, "z_existing")
    assert_eq(msgs2["root"]["childrenIds"], ["z_existing", "new_sibling"],
              "append only -- the pre-existing order/entry is preserved, not rebuilt")
    problems = S.diff_preexisting_messages(pre, msgs2, plan.parents_appended)
    assert_eq(problems, [], "diff_preexisting_messages finds the append clean")


def test_diff_preexisting_messages_catches_an_unexpected_field_change():
    before = {"m1": {"id": "m1", "role": "user", "content": "x", "childrenIds": []}}
    after = {"m1": {"id": "m1", "role": "user", "content": "CHANGED", "childrenIds": []}}
    problems = S.diff_preexisting_messages(before, after, [])
    assert_true(any("content" in p for p in problems), "a changed field on an existing message is caught")


# ===========================================================================
# --sync-json-only, CLI level: table byte-identical, idempotent exit codes
# ===========================================================================


def _make_sync_db(path, chat_id="ea1494ea-e9d7-46fb-8b7c-3a50d685d00e"):
    con = sqlite3.connect(str(path))
    con.execute("create table chat (id text primary key, chat text, current_message_id text, updated_at int)")
    con.execute("create table chat_message (id text primary key, chat_id text, parent_id text, role text, "
                "content text, output text, model_id text, done boolean, created_at int)")
    msgs = link(tree(msg("root", "user", 0)))
    blob = json.dumps({"history": {"messages": msgs, "currentId": "a1"}, "messages": []})
    con.execute("insert into chat values (?,?,?,?)", (chat_id, blob, "a1", int(time.time())))
    con.execute("insert into chat_message values (?,?,?,?,?,?,?,?,?)",
                (f"{chat_id}-root", chat_id, None, "user", json.dumps("x"), None, None, None, 0))
    con.execute("insert into chat_message values (?,?,?,?,?,?,?,?,?)",
                (f"{chat_id}-a1", chat_id, "root", "assistant", json.dumps("hi"),
                 json.dumps([{"type": "message", "content": [{"type": "output_text", "text": "hi"}]}]),
                 "some/model", True, 10))
    con.commit()
    con.close()


def _chat_message_rows(db):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = list(con.execute("select * from chat_message order by id"))
    con.close()
    return rows


def test_sync_apply_is_byte_identical_on_chat_message_and_idempotent():
    import subprocess
    tmp = Path(tempfile.mkdtemp(prefix="rct-sync-cli-"))
    try:
        db = tmp / "webui.db"
        _make_sync_db(db)
        before_rows = _chat_message_rows(db)

        r1 = subprocess.run([sys.executable, str(_SCRIPT), str(db), "--sync-json-only", "--apply", "--json"],
                             capture_output=True, text=True)
        assert_true(r1.returncode == 0, "first --sync-json-only --apply run exits 0", r1.stdout + r1.stderr)
        after_rows = _chat_message_rows(db)
        assert_eq(after_rows, before_rows, "chat_message table is byte-for-byte unchanged after --apply")

        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        chat = json.loads(con.execute("select chat from chat where id=?",
                                       ("ea1494ea-e9d7-46fb-8b7c-3a50d685d00e",)).fetchone()[0])
        con.close()
        hmsgs = chat["history"]["messages"]
        assert_true("a1" in hmsgs, "the table-only assistant message was copied into the JSON")
        assert_eq(hmsgs["a1"].get("output"),
                  [{"type": "message", "content": [{"type": "output_text", "text": "hi"}]}],
                  "output field present and decoded in the written JSON")

        r2 = subprocess.run([sys.executable, str(_SCRIPT), str(db), "--sync-json-only", "--apply"],
                             capture_output=True, text=True)
        assert_true(r2.returncode == 0, "second run exits 0", r2.stdout + r2.stderr)
        assert_true("nothing to do" in r2.stdout, "second run reports nothing to do", r2.stdout)
        assert_eq(_chat_message_rows(db), before_rows, "chat_message still byte-identical after the second run")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_sync_dry_run_reports_pending_work_exit_3():
    import subprocess
    tmp = Path(tempfile.mkdtemp(prefix="rct-sync-dry-"))
    try:
        db = tmp / "webui.db"
        _make_sync_db(db)
        r = subprocess.run([sys.executable, str(_SCRIPT), str(db), "--sync-json-only"],
                            capture_output=True, text=True)
        assert_true(r.returncode == 3, "dry run with pending work exits 3", r.stdout + r.stderr)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_decode_plain_text()
    test_decode_list_of_blocks()
    test_decode_non_json_text_is_returned_unchanged()
    test_decode_non_string_input_passed_through()

    test_relink_prefers_table_link_when_it_resolves()
    test_relink_table_link_into_own_subtree_is_rejected()
    test_relink_ties_at_the_same_second_prefer_larger_subtree()
    test_relink_orphan_whose_true_parent_is_also_missing()
    test_relink_orphan_subtree_with_a_fork_moves_as_one_unit()
    test_relink_never_produces_a_cycle_even_with_adversarial_table_links()
    test_find_any_cycle_detects_a_pre_existing_cycle_untouched_by_relinking()

    test_pick_stored_pointer_prefers_column_when_it_resolves()
    test_pick_stored_pointer_falls_back_when_column_points_nowhere()
    test_pick_stored_pointer_none_when_neither_resolves()

    test_pointer_kept_when_she_deliberately_navigated_back_and_stopped()
    test_pointer_walks_forward_when_the_live_tab_kept_going_stale_tab_case()
    test_pointer_never_regresses_below_a_broken_pointers_own_timestamp()
    test_pointer_regression_guard_with_a_resolvable_but_broken_pointer()
    test_forward_tip_prefers_assistant_on_an_exact_timestamp_tie()
    test_pointer_not_regressed_when_a_newer_candidate_exists()
    test_dangling_child_ids_are_actually_removed()
    test_true_root_parent_id_is_forced_to_none()

    test_second_pass_over_an_already_repaired_tree_is_a_clean_noop()

    test_chat_default_is_her_id_not_largest()
    test_liveness_refusal_when_another_process_holds_the_file_open()
    test_wal_and_journal_sidecars_are_detected()
    test_backup_then_restore_is_byte_identical()
    test_restore_refuses_cleanly_when_no_backup_exists()
    test_find_chat_requires_a_single_unambiguous_match()

    test_sync_fills_a_mid_branch_gap()
    test_sync_table_only_chain_of_two_or_more()
    test_sync_output_field_is_copied()
    test_sync_output_absent_when_column_is_null()
    test_sync_never_relinks_an_unrelated_orphan()
    test_sync_pointer_unchanged_when_column_and_json_already_agree()
    test_sync_pointer_aligned_to_column_when_they_disagree_and_column_resolves()
    test_sync_pointer_not_aligned_when_column_does_not_resolve_in_table()
    test_sync_idempotent_second_run_reports_nothing_to_do()
    test_sync_refuses_when_result_would_not_match_table_branch()
    test_sync_field_audit_reports_what_the_table_cannot_supply()
    test_sync_existing_parent_childrenids_is_a_pure_append()
    test_diff_preexisting_messages_catches_an_unexpected_field_change()
    test_sync_apply_is_byte_identical_on_chat_message_and_idempotent()
    test_sync_dry_run_reports_pending_work_exit_3()

    if _FAILS:
        print(f"\n{len(_FAILS)} FAILURE(S): {_FAILS}")
        sys.exit(1)
    print("\nAll repair-chat-tree.py script tests passed.")
