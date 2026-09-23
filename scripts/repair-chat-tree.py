#!/usr/bin/env python3
"""Repair an OpenWebUI 0.11 chat whose message tree has snapped links.

v2 — supersedes scripts/repair-chat-tree.py (this file, v1 is retired), which had four defects found in review:
  * it copied message text in its stored (JSON-encoded) form, so copied
    messages rendered with a literal quote and literal "\\n";
  * it required a parent to be STRICTLY earlier, but OpenWebUI stamps a
    question and its reply in the same second, so a snapped reply was linked
    to the PREVIOUS turn's question;
  * it rebuilt the legacy flat `messages` list with the whole branch, which
    nearly doubled the row OpenWebUI rewrites on every send;
  * it read and wrote outside a transaction, and its verification compared
    the result with itself.

What v2 does:
  1. Re-links every message whose parent is missing, preferring the parent
     OpenWebUI itself recorded in the chat_message table, and otherwise the
     nearest earlier-or-equal message of the opposite role that is not inside
     the orphan's own subtree.
  2. Copies table-only messages into the history with their text DECODED.
  3. Points the chat at the newest message that has no children (a reply
     rather than the question it answers).
  4. Leaves the flat `messages` list exactly as it found it.
  5. Writes both copies in one BEGIN IMMEDIATE transaction, only after
     re-reading the row, and refuses to run while the database is open.

Usage:
  repair-chat-tree.py <webui.db> [--chat <id-prefix|largest>] [--apply]
Dry run by default. Output is counts, short ids and timestamps only.
"""
import argparse, datetime, json, os, sqlite3, sys, time

fmt = lambda t: datetime.datetime.fromtimestamp(t if t < 1e11 else t / 1e9, datetime.UTC).strftime("%m-%d %H:%M:%SZ") if t else "?"


def walk_up(msgs, start):
    path, n = [], start
    while n in msgs and n not in path:
        path.append(n)
        n = msgs[n].get("parentId")
    return path


def decode(s):
    """chat_message.content is stored JSON-encoded; give back the real text."""
    if not isinstance(s, str):
        return s
    try:
        v = json.loads(s)
        return v if isinstance(v, str) else s
    except Exception:
        return s


def openers(path):
    real, me, found = os.path.realpath(path), str(os.getpid()), []
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("--chat", default="largest")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    if a.apply:
        who = openers(a.db)
        if who:
            sys.exit(f"REFUSING: the database is open by process(es) {who}. Stop openwebui "
                     f"and backup, close every browser tab on the chat, and retry.")

    con = sqlite3.connect(a.db, timeout=60, isolation_level=None)
    if a.apply:
        con.execute("BEGIN IMMEDIATE")
    target = None
    for cid, blob, cur in con.execute("select id, chat, current_message_id from chat"):
        if a.chat != "largest" and not cid.startswith(a.chat):
            continue
        chat = json.loads(blob)
        m = chat.get("history", {}).get("messages", {})
        if isinstance(m, dict) and (target is None or len(m) > len(target[1]["history"]["messages"])):
            target = (cid, chat, cur)
    if target is None:
        sys.exit("no matching chat")
    cid, chat, cur_col = target
    updated_at = con.execute("select updated_at from chat where id=?", (cid,)).fetchone()[0]
    prefix = cid + "-"
    table = {r[0][len(prefix):] if r[0].startswith(prefix) else r[0]:
             {"row": r[0], "parentId": r[1], "role": r[2], "content": r[3], "created": r[4]}
             for r in con.execute(
                 "select id, parent_id, role, content, created_at from chat_message where chat_id=?", (cid,))}

    hist = chat.setdefault("history", {})
    msgs = hist.setdefault("messages", {})
    flat_before = len(chat.get("messages") or [])
    print(f"chat {cid[:8]}: history {len(msgs)} | table {len(table)} | pointer {(cur_col or '-')[:8]} "
          f"| flat list {flat_before} (left untouched)")
    before_branch = walk_up(msgs, cur_col or hist.get("currentId"))
    print(f"before: the branch the UI would show is {len(before_branch)} message(s)")

    added = 0
    for mid, r in table.items():
        if mid not in msgs:
            msgs[mid] = {"id": mid, "parentId": r["parentId"], "childrenIds": [], "role": r["role"],
                         "content": decode(r["content"]),
                         "timestamp": int(r["created"] if r["created"] < 1e11 else r["created"] / 1e9)}
            added += 1
    print(f"table-only messages copied into the history (text decoded): {added}")

    ts = lambda m: msgs[m].get("timestamp") or 0
    dangling = 0
    for m in msgs.values():
        kids = m.get("childrenIds") or []
        clean = [k for k in kids if k in msgs]
        dangling += len(kids) - len(clean)
        m["childrenIds"] = clean

    orphans = sorted([m for m, v in msgs.items()
                      if v.get("parentId") is None or v.get("parentId") not in msgs], key=ts)
    true_root = orphans[0] if orphans else None
    if true_root:
        msgs[true_root]["parentId"] = None
    relinks, from_table = {}, 0
    for mid in orphans[1:]:
        role = msgs[mid].get("role")
        parent = None
        t_parent = (table.get(mid) or {}).get("parentId")
        if t_parent in msgs and mid not in walk_up(msgs, t_parent):
            parent, src = t_parent, "table"
            from_table += 1
        else:
            cands = sorted(((ts(x), 0 if msgs[x].get("role") != role else 1, x)
                            for x in msgs
                            if x != mid and ts(x) <= ts(mid)), reverse=True)
            for _, _, cand in cands:
                if msgs[cand].get("role") == role or mid in walk_up(msgs, cand):
                    continue
                parent, src = cand, "time"
                break
        if parent is None:
            print(f"  {mid[:8]} at {fmt(ts(mid))}: no usable parent, left as a root")
            continue
        msgs[mid]["parentId"] = parent
        if mid not in msgs[parent]["childrenIds"]:
            msgs[parent]["childrenIds"].append(mid)
        relinks[mid] = parent
        print(f"  {mid[:8]} ({role}, {fmt(ts(mid))}) -> {parent[:8]} "
              f"({msgs[parent].get('role')}, {fmt(ts(parent))}) [{src}]")
    print(f"orphans re-linked: {len(relinks)} of {max(0, len(orphans) - 1)} "
          f"({from_table} from the table's own links) | dangling child ids removed: {dangling} "
          f"| true root {true_root[:8] if true_root else '-'} at {fmt(ts(true_root)) if true_root else '-'}")

    fixed_kids = 0
    for mid, m in msgs.items():
        p = m.get("parentId")
        if p in msgs and mid not in msgs[p]["childrenIds"]:
            msgs[p]["childrenIds"].append(mid)
            fixed_kids += 1

    # the newest LEAF: a reply, not the question it answers
    leaves = [m for m in msgs if not msgs[m].get("childrenIds")]
    pool = leaves or list(msgs)
    newest = sorted(pool, key=lambda m: (ts(m), 1 if msgs[m].get("role") == "assistant" else 0))[-1]
    hist["currentId"] = newest
    print(f"pointer: {(cur_col or '-')[:8]} -> {newest[:8]} ({msgs[newest].get('role')}, {fmt(ts(newest))})")

    # verification, against the way OpenWebUI itself loads the chat
    ui = {m: {"id": m, "parentId": relinks.get(m, (table[m]["parentId"] if m in table else msgs[m].get("parentId"))),
              "role": (table.get(m) or msgs[m]).get("role")} for m in set(table) | set(msgs)}
    for m in ui:                      # OpenWebUI fills gaps from the history
        if ui[m]["parentId"] and ui[m]["parentId"] not in ui and m in msgs:
            ui[m]["parentId"] = msgs[m].get("parentId")
    ui_path = walk_up(ui, newest)
    json_path = walk_up(msgs, newest)
    roots = [m for m, v in msgs.items() if v.get("parentId") is None or v.get("parentId") not in msgs]
    bad_time = [m for m, p in relinks.items() if ts(p) > ts(m)]
    bad_role = [m for m, p in relinks.items() if msgs[p].get("role") == msgs[m].get("role")
                and p != (table.get(m) or {}).get("parentId")]
    encoded = [m for m in msgs if isinstance(msgs[m].get("content"), str)
               and msgs[m]["content"][:1] == '"' and isinstance(decode(msgs[m]["content"]), str)
               and decode(msgs[m]["content"]) != msgs[m]["content"]]
    ok = (len(roots) == 1 and json_path and json_path[-1] == true_root
          and ui_path and ui_path[-1] == true_root and not bad_time and not bad_role and not encoded)
    print(f"after: roots {len(roots)} | branch {len(json_path)} | OpenWebUI's own walk {len(ui_path)} | "
          f"both reach the true root: {bool(json_path) and json_path[-1] == true_root and bool(ui_path) and ui_path[-1] == true_root}")
    print(f"checks: parents not newer than their child: {not bad_time} | roles alternate or came from the table: "
          f"{not bad_role} | no encoded text left in the history: {not encoded}")
    print("verification:", "OK" if ok else "FAILED")
    if not a.apply:
        print("DRY RUN — nothing written")
        return
    if not ok:
        con.execute("ROLLBACK")
        sys.exit(2)
    if con.execute("select updated_at from chat where id=?", (cid,)).fetchone()[0] != updated_at:
        con.execute("ROLLBACK")
        sys.exit("the chat changed while this ran; nothing written")
    con.execute("update chat set chat=?, current_message_id=?, updated_at=? where id=?",
                (json.dumps(chat), newest, int(time.time()), cid))
    for mid, parent in relinks.items():
        if mid in table:
            con.execute("update chat_message set parent_id=? where id=?", (parent, table[mid]["row"]))
    con.execute("COMMIT")
    back = json.loads(con.execute("select chat from chat where id=?", (cid,)).fetchone()[0])
    cur2 = con.execute("select current_message_id from chat where id=?", (cid,)).fetchone()[0]
    bm = back["history"]["messages"]
    print(f"written. re-read: branch {len(walk_up(bm, cur2))} of {len(bm)} message(s), "
          f"flat list {len(back.get('messages') or [])} (was {flat_before}), "
          f"row {len(json.dumps(back).encode())/1e6:.1f} MB")
    print("integrity:", con.execute("PRAGMA integrity_check").fetchone()[0])


if __name__ == "__main__":
    main()
