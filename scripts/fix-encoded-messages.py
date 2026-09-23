#!/usr/bin/env python3
"""Undo the encoding damage repair-chat-tree.py did to her chat.

repair-chat-tree.py copied messages that existed only in the chat_message
table into the chat's JSON history. That table stores content JSON-encoded,
and the script copied the encoded form, so those messages render with a
literal leading quote and literal "\\n". A browser save afterwards writes the
history back into the table, encoding it again, so the damage can also be in
the table, which is what the model reads.

This script restores each affected message from the pre-repair copy of the
database, where its table content is the original, correct value.

Usage:
  fix-encoded-messages.py <live webui.db> <pre-repair webui.db> [--apply]

Dry run by default. Refuses to write while any other process has the live
database open. Writes in one BEGIN IMMEDIATE transaction after re-reading the
row. Touches only the affected messages, in both copies. Never deletes.
Output is counts and short ids only.
"""
import json, os, sqlite3, sys


def openers(path):
    real = os.path.realpath(path)
    me = str(os.getpid())
    found = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or pid == me:
            continue
        try:
            for fd in os.listdir(f"/proc/{pid}/fd"):
                try:
                    if os.path.realpath(f"/proc/{pid}/fd/{fd}").startswith(real):
                        found.append(pid)
                        break
                except OSError:
                    pass
        except OSError:
            pass
    return found


def decode_once(s):
    try:
        v = json.loads(s)
        return v if isinstance(v, str) else None
    except Exception:
        return None


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    live, pre, apply = sys.argv[1], sys.argv[2], "--apply" in sys.argv

    src = sqlite3.connect(f"file:{pre}?mode=ro", uri=True, timeout=60)
    cid = src.execute("select chat_id from chat_message group by chat_id order by count(*) desc limit 1").fetchone()[0]
    pre_hist = json.loads(src.execute("select chat from chat where id=?", (cid,)).fetchone()[0])["history"]["messages"]
    prefix = cid + "-"
    pre_table = {r[0][len(prefix):]: r[1] for r in src.execute(
        "select id, content from chat_message where chat_id=?", (cid,))}
    src.close()
    # the messages the repair copied: in the table, not in the history, before the repair
    targets = {m: raw for m, raw in pre_table.items() if m not in pre_hist and decode_once(raw) is not None}
    print(f"chat {cid[:8]}: {len(targets)} message(s) the repair copied into the history")

    if apply:
        who = openers(live)
        if who:
            sys.exit(f"REFUSING: the live database is open by process(es) {who}. "
                     f"Stop openwebui and backup, close every browser tab on the chat, and retry.")

    con = sqlite3.connect(live, timeout=60, isolation_level=None)
    if apply:
        con.execute("BEGIN IMMEDIATE")
    blob, updated = con.execute("select chat, updated_at from chat where id=?", (cid,)).fetchone()
    chat = json.loads(blob)
    hist = chat["history"]["messages"]
    live_table = {r[0][len(prefix):]: r[1] for r in con.execute(
        "select id, content from chat_message where chat_id=?", (cid,))}

    fix_hist, fix_table, already_ok, skipped = [], [], 0, []
    for mid, good_raw in targets.items():
        good = decode_once(good_raw)
        h = (hist.get(mid) or {}).get("content")
        t = live_table.get(mid)
        hist_bad = h is not None and h != good and h in (good_raw, decode_once(t) if t else None)
        # table is double-encoded if decoding it once yields the bad (encoded) form
        table_bad = t is not None and t != good_raw and decode_once(t) == good_raw
        if h is not None and h != good and not hist_bad:
            skipped.append(mid)      # changed some other way since; leave it alone
            continue
        if hist_bad:
            fix_hist.append(mid)
        if table_bad:
            fix_table.append(mid)
        if not hist_bad and not table_bad:
            already_ok += 1
    print(f"history copies to restore: {len(fix_hist)} | table copies double-encoded, to restore: {len(fix_table)} | "
          f"already correct: {already_ok} | changed since, left alone: {len(skipped)} {[m[:8] for m in skipped]}")

    for mid in fix_hist:
        hist[mid]["content"] = decode_once(targets[mid])
    new_blob = json.dumps(chat)

    # independent verification: every target, in the NEW state, equals the pre-repair truth
    bad_after = []
    for mid, good_raw in targets.items():
        if mid in skipped:
            continue
        good = decode_once(good_raw)
        h = hist.get(mid, {}).get("content")
        t = good_raw if mid in fix_table else live_table.get(mid)
        if h != good or decode_once(t) != good:
            bad_after.append(mid)
    ok = not bad_after
    print(f"verification against the pre-repair copy: {'OK' if ok else 'FAILED ' + str([m[:8] for m in bad_after])}")
    if not apply:
        print("DRY RUN — nothing written")
        return
    if not ok:
        con.execute("ROLLBACK")
        sys.exit("verification failed; nothing written")
    now_updated = con.execute("select updated_at from chat where id=?", (cid,)).fetchone()[0]
    if now_updated != updated:
        con.execute("ROLLBACK")
        sys.exit("the chat changed while this ran; nothing written")
    if fix_hist:
        con.execute("update chat set chat=? where id=?", (new_blob, cid))
    for mid in fix_table:
        con.execute("update chat_message set content=? where id=?", (targets[mid], prefix + mid))
    con.execute("COMMIT")
    back = json.loads(con.execute("select chat from chat where id=?", (cid,)).fetchone()[0])["history"]["messages"]
    tb = {r[0][len(prefix):]: r[1] for r in con.execute("select id, content from chat_message where chat_id=?", (cid,))}
    good_n = sum(1 for m, g in targets.items() if m not in skipped
                 and back.get(m, {}).get("content") == decode_once(g) and decode_once(tb.get(m)) == decode_once(g))
    print(f"written. re-read: {good_n} of {len(targets) - len(skipped)} messages correct in both copies")
    print("integrity:", con.execute("PRAGMA integrity_check").fetchone()[0])


if __name__ == "__main__":
    main()
