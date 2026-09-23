#!/usr/bin/env python3
"""Mark stale "unfinished" assistant messages as finished, in one chat.

WHY. Upstream OpenWebUI issue #14806: an assistant message stored with
`done: false` renders as a permanent loading spinner. On chat load,
OpenWebUI's own code marks only the NEWEST unfinished response done —
every OLDER `done: false` message is left exactly as it was, forever. In
her chat (`ea1494ea-e9d7-46fb-8b7c-3a50d685d00e`, this script's `--chat`
default), the 2026-09-23 backup has 6 assistant messages on the current
branch that are EMPTY and `done: false` in BOTH copies (branch positions
508, 516, 604, 1371, 1818, 2672 — verified against that backup with
`scripts/repair-chat-tree.py`'s own `walk_up`), plus 13 more with the same
shape sitting off the current branch, plus 9 assistant messages where the
JSON's `done` key is simply MISSING (not false) while the table already
says `done=1` — see CATEGORIES below for why those get handled, and
reported, separately.

STORAGE FACTS (OpenWebUI 0.11, matches repair-chat-tree.py's own module
docstring point 1-2, re-confirmed here for `done` specifically against
`backend/open_webui/models/chat_messages.py` at tag v0.11.4 and against
this pod's own real 2026-09-23 backup — schema unchanged between the two):
each message is stored TWICE. `chat.chat` (a JSON column on the `chat`
table) holds `history.messages[<id>]`, a dict with a `done` key that can
be `True`, `False`, or simply ABSENT. `chat_message` holds one row per
message, keyed `"<chat_id>-<message_id>"`, with an ordinary `done` column
(SQLAlchemy `Boolean`, stored as SQLite `INTEGER` 0/1 — never NULL was
observed on the real backup, but this script treats NULL the same as 0
defensively). `get_messages_map_by_chat_id` reads the TABLE first and only
uses the JSON to fill gaps the table's own graph cannot resolve — the
table is the copy OpenWebUI itself already treats as authoritative.

CATEGORIES (both fixed by `--apply`; reported as separate counts):
  (a) STALE UNFINISHED — `done` is not `True` in the JSON (missing or
      `False`) AND the table's `done` column is not `1` (0 or missing).
      Both copies agree the message never finished — this is the actual
      spinner bug. Fixed by setting `done=True` in the JSON and `done=1`
      in the table.
  (b) JSON-ONLY GAP — the JSON `done` key is MISSING (not `False`) while
      the table already says `done=1`. The table is authoritative, so
      this was never rendering as a live spinner the same way (a) does;
      it is a plain hygiene backfill, not the bug fix. Fixed by setting
      `done=True` in the JSON only (the table needs no write). Counted
      and printed separately from (a) so a dry run cannot make (b) look
      like additional instances of the actual bug.
  Measured on the real 2026-09-23 backup: (a) 19 total (6 on the current
  branch, 13 off it), (b) 9 total (0 on the current branch, all off it).

TARGETING RULES.
  - `--chat` selects the conversation (exact id or unique prefix), default
    her conversation.
  - `--branch-only` (default) restricts BOTH categories to messages that
    are ancestors of the stored pointer (`chat.current_message_id`, or
    `history.currentId` if the column is unset — same fallback order as
    `repair-chat-tree.py`'s `pick_stored_pointer`, and the same `walk_up`
    definition of "branch"). `--all-branches` lifts that restriction.
  - The current tip itself, and any of its direct children (an in-flight
    reply actually generating right now would be exactly this: a fresh
    assistant child of the pointer), are NEVER touched, regardless of the
    branch/age flags below — this is the one exclusion this script will
    not override.
  - `--min-age-minutes` (default 10): a candidate is only a target if it
    is older than this, using `max(chat_message.created_at,
    chat_message.updated_at)` (falling back to the JSON message's own
    `timestamp` field for a message that somehow has no table row at
    all). This is a second, independent guard against touching a reply
    that is genuinely mid-generation — belt-and-suspenders alongside the
    tip exclusion above, not a replacement for it.

THE WRITE. Sets `done=True`/`done=1` for exactly the in-scope targets.
Nothing else changes: not `content`, not `created_at`/`updated_at` on
either copy, not `chat.updated_at`, not `current_message_id`, not the flat
`chat.chat["messages"]` list. `chat.updated_at` is read once before the
write (to detect a concurrent change, the same guard
`repair-chat-tree.py` uses) and deliberately never written back — this
fix is plumbing, not a user-visible edit, and must not appear as one.

REFUSALS (checked before any backup or write): another process holding
`webui.db` open (`/proc/<pid>/fd` scan — same check
`scripts/clean-decoration.py` and `scripts/repair-chat-tree.py` use), or a
`-wal`/`-journal` sidecar beside it (that failure mode belongs to
RUNBOOK_DB_JOURNAL.md, not this script).

BACKUP. Unlike `repair-chat-tree.py`/`fix-encoded-messages.py` (which use
a plain `shutil.copy2` beside the live db specifically so `--restore` can
give back the exact original bytes), this script's own backup uses
`sqlite3`'s online backup API (`Connection.backup()`) and writes into a
dedicated forensics directory: `<forensics dir>/fix-stale-unfinished-
<UTC stamp>/webui.db` (`<forensics dir>` is `/data/forensics` by default,
overridable with `$FSU_FORENSICS_DIR` for testing off the pod). Preceded
by the same free-space check (20% headroom) the other two scripts use,
against that directory's own filesystem. `--restore <stamp>` copies that
exact forensics file back over the live path with `shutil.copy2` — so the
restored file is byte-identical to what the backup produced (verified by
md5 in the test suite), which is what "reversible" means for this
script's own backup, whatever internal page layout `Connection.backup()`
happened to choose when it made that file.

VERIFICATION. After the write commits, this script opens a fresh
connection and re-reads: (1) exactly the intended N flags are now `True`/1
and were not before, in both copies, for exactly the intended ids; (2)
every message's `content` — table and JSON, EVERY message, not just the
targets — is byte-identical to what a pre-write snapshot recorded; (3)
`PRAGMA integrity_check`. A verification failure is reported and exits 1;
by that point the write has already committed (SQLite has no ROLLBACK
after COMMIT), so the way back is `--restore <the stamp this run printed>`,
not this script re-trying anything on its own.

OUTPUT. Counts, short (8-char) ids, branch positions, and content LENGTHS
only. Never message text — see `_report`.

OPERATOR PROCEDURE (on the pod):
    1. Ask her to close every browser tab on this chat.
    2. supervisorctl stop openwebui backup
    3. /app/venv/bin/python /opt/zl-repo/scripts/fix-stale-unfinished.py \\
           /data/openwebui/webui.db          # dry run — writes nothing
    4. /app/venv/bin/python /opt/zl-repo/scripts/fix-stale-unfinished.py \\
           /data/openwebui/webui.db --apply
    5. supervisorctl start openwebui backup
    6. Have her hard-refresh the chat.

Usage:
  fix-stale-unfinished.py <webui.db> [--chat <id-or-prefix>]
      [--branch-only | --all-branches] [--min-age-minutes N]
      [--apply] [--json]
  fix-stale-unfinished.py <webui.db> --restore <stamp>

Dry run by default.

EXIT CODES — the shared convention (OPERATIONS.md "Exit codes — the
shared convention across the operator scripts"): 0 nothing in scope to
fix (dry run) or `--apply` fixed everything in scope and verification
passed, including a no-op `--apply`, or `--restore` succeeded; 1 a
refusal, an error, or a verification failure after `--apply` (already
committed — use `--restore`); 2 argparse's own usage errors; 3 a dry run
found in-scope work, informational.
"""

import argparse
import datetime
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import time
from pathlib import Path

DEFAULT_CHAT_ID = "ea1494ea-e9d7-46fb-8b7c-3a50d685d00e"
JOURNAL_RUNBOOK = "RUNBOOK_DB_JOURNAL.md"
DEFAULT_MIN_AGE_MINUTES = 10
FORENSICS_DIRNAME = "fix-stale-unfinished"


def forensics_base():
    return Path(os.environ.get("FSU_FORENSICS_DIR", "/data/forensics"))


def utc_stamp():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def fmt_ts(t):
    if not t:
        return "?"
    if t > 1e11:
        t = t / 1e9
    return datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime("%m-%d %H:%M:%SZ")


# ===========================================================================
# PART 1 — liveness refusals (same checks as repair-chat-tree.py /
# fix-encoded-messages.py / clean-decoration.py)
# ===========================================================================


def openers(path):
    real = os.path.realpath(str(path))
    me = str(os.getpid())
    found = []
    try:
        pids = os.listdir("/proc")
    except OSError:
        return []
    for pid in pids:
        if not pid.isdigit() or pid == me:
            continue
        try:
            fds = os.listdir(f"/proc/{pid}/fd")
        except OSError:
            continue
        for fd in fds:
            try:
                if os.path.realpath(f"/proc/{pid}/fd/{fd}").startswith(real):
                    found.append(pid)
                    break
            except OSError:
                pass
    return found


def live_sidecars(db_path):
    p = Path(db_path)
    return [str(s) for s in (p.with_name(p.name + "-wal"), p.with_name(p.name + "-journal")) if s.exists()]


# ===========================================================================
# PART 2 — backup / restore (sqlite3 online backup API, into a dedicated
# forensics directory — see module docstring's BACKUP section for why this
# script, unlike its siblings, does not use a plain byte copy for this)
# ===========================================================================


def check_free_space(dest_dir, needed_bytes):
    free = shutil.disk_usage(dest_dir).free
    required = int(needed_bytes * 1.2) + 10 * 1024 * 1024
    if free < required:
        return (f"ERROR: only {free / 1e6:.1f} MB free at {dest_dir}, need "
                f"~{required / 1e6:.1f} MB to back up {needed_bytes / 1e6:.1f} MB. "
                f"Refusing to proceed without a successful backup.")
    return None


def verify_backup_openable(bak):
    try:
        con = sqlite3.connect(f"file:{Path(bak).resolve().as_posix()}?mode=ro", uri=True)
        try:
            row = con.execute("PRAGMA integrity_check").fetchone()
        finally:
            con.close()
    except Exception as e:
        return f"{type(e).__name__}: {e}"
    if row is None or row[0] != "ok":
        return str(row)
    return None


def backup_db(db_path, stamp):
    """The sqlite3 online backup API, into
    `<forensics dir>/fix-stale-unfinished-<stamp>/webui.db` — see the
    module docstring's BACKUP section. Verifies the fresh copy opens
    cleanly under PRAGMA integrity_check before returning."""
    src = Path(db_path)
    dest_dir = forensics_base() / f"{FORENSICS_DIRNAME}-{stamp}"
    if dest_dir.exists():
        raise FileExistsError(f"backup directory already exists: {dest_dir}")
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    err = check_free_space(dest_dir.parent, src.stat().st_size)
    if err:
        raise OSError(err)
    dest_dir.mkdir()
    dest = dest_dir / "webui.db"
    src_con = sqlite3.connect(f"file:{src.resolve().as_posix()}?mode=ro", uri=True)
    dst_con = sqlite3.connect(str(dest))
    try:
        src_con.backup(dst_con)
    finally:
        dst_con.close()
        src_con.close()
    err = verify_backup_openable(dest)
    if err:
        shutil.rmtree(dest_dir, ignore_errors=True)
        raise OSError(f"backup of {src} failed integrity_check: {err}")
    return dest


def restore_db(db_path, stamp):
    dest = forensics_base() / f"{FORENSICS_DIRNAME}-{stamp}" / "webui.db"
    if not dest.is_file():
        return None, f"REFUSED: no backup found at {dest}"
    src = Path(db_path)
    for sidecar in (src.with_name(src.name + "-wal"), src.with_name(src.name + "-journal")):
        sidecar.unlink(missing_ok=True)
    shutil.copy2(dest, src)
    err = verify_backup_openable(src)
    if err:
        return None, f"REFUSED: restored file failed integrity_check: {err}"
    return dest, f"restored {src} from {dest}"


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ===========================================================================
# PART 3 — tree walk + targeting
# ===========================================================================


def walk_up(msgs, start):
    """Ancestors of `start`, including itself, root-most last — same
    definition as repair-chat-tree.py's own `walk_up` (duplicated here
    rather than imported, matching this project's convention of each
    operator script standing alone)."""
    path, n = [], start
    while n in msgs and n not in path:
        path.append(n)
        n = msgs[n].get("parentId")
    return path


def pick_stored_pointer(cur_col, hist_current, msgs):
    if cur_col and cur_col in msgs:
        return cur_col
    if hist_current and hist_current in msgs:
        return hist_current
    return cur_col or hist_current or None


def load_table(con, chat_id):
    prefix = chat_id + "-"
    table = {}
    for mid, parent_id, role, content, done, created, updated in con.execute(
        "select id, parent_id, role, content, done, created_at, updated_at from chat_message where chat_id=?",
        (chat_id,),
    ):
        key = mid[len(prefix):] if mid.startswith(prefix) else mid
        table[key] = {
            "row": mid, "parentId": parent_id, "role": role, "content": content,
            "done": done, "created": created, "updated": updated,
        }
    return table


def find_chat(con, chat_arg):
    rows = list(con.execute("select id, chat, current_message_id, updated_at from chat"))
    matches = [r for r in rows if r[0] == chat_arg or r[0].startswith(chat_arg)]
    if len(matches) == 1:
        return matches[0]
    return None


def content_len(v):
    if v is None:
        return 0
    if isinstance(v, str):
        return len(v)
    try:
        return len(json.dumps(v))
    except TypeError:
        return len(str(v))


def msg_age_minutes(mid, hist, table, now):
    row = table.get(mid)
    if row is not None and (row.get("created") or row.get("updated")):
        ts = max(row.get("created") or 0, row.get("updated") or 0)
        if ts > 1e11:  # ms -> s, defensive; not observed in the real schema
            ts = ts / 1000
        return (now - ts) / 60.0
    ts = (hist.get(mid) or {}).get("timestamp")
    if ts:
        if ts > 1e11:
            ts = ts / 1000
        return (now - ts) / 60.0
    return None  # unknown age -> caller treats as "too new to touch"


class Target:
    __slots__ = ("mid", "category", "on_branch", "branch_pos", "age_minutes",
                 "json_len", "table_len", "in_scope", "skip_reason")

    def __init__(self, mid, category, on_branch, branch_pos, age_minutes, json_len, table_len):
        self.mid = mid
        self.category = category
        self.on_branch = on_branch
        self.branch_pos = branch_pos
        self.age_minutes = age_minutes
        self.json_len = json_len
        self.table_len = table_len
        self.in_scope = True
        self.skip_reason = None


def build_targets(chat, table, cur_col, min_age_minutes, branch_only, now=None):
    now = now if now is not None else time.time()
    hist = chat.get("history", {}).get("messages", {})
    hist_current = chat.get("history", {}).get("currentId")
    pointer = pick_stored_pointer(cur_col, hist_current, hist)

    branch = set(walk_up(hist, pointer)) if pointer else set()
    branch_order = list(reversed(walk_up(hist, pointer))) if pointer else []  # root..pointer
    branch_pos = {mid: i + 1 for i, mid in enumerate(branch_order)}

    protected = set()
    if pointer:
        protected.add(pointer)
        for child in (hist.get(pointer) or {}).get("childrenIds") or []:
            if (hist.get(child) or {}).get("role") == "assistant":
                protected.add(child)

    targets = []
    for mid, m in hist.items():
        if m.get("role") != "assistant":
            continue
        row = table.get(mid)
        json_done = m.get("done") is True
        json_missing = "done" not in m
        table_done = bool(row and row.get("done") == 1)

        if not json_done and not table_done:
            category = "a"
        elif json_missing and table_done:
            category = "b"
        else:
            continue  # already done in both, or the (unmentioned) reverse case — not this script's job

        on_b = mid in branch
        t = Target(
            mid, category, on_b, branch_pos.get(mid),
            msg_age_minutes(mid, hist, table, now),
            content_len(m.get("content")), content_len(row.get("content")) if row else None,
        )
        if mid in protected:
            t.in_scope = False
            t.skip_reason = "current tip / its in-flight reply"
        elif branch_only and not on_b:
            t.in_scope = False
            t.skip_reason = "off-branch (use --all-branches)"
        elif t.age_minutes is None or t.age_minutes < min_age_minutes:
            t.in_scope = False
            t.skip_reason = f"younger than --min-age-minutes ({min_age_minutes})"
        targets.append(t)

    targets.sort(key=lambda t: (t.branch_pos is None, t.branch_pos or 0, t.mid))
    return targets


# ===========================================================================
# PART 4 — report + main
# ===========================================================================


def summarize(targets):
    in_scope = [t for t in targets if t.in_scope]
    return {
        "found_total": len(targets),
        "found_category_a": sum(1 for t in targets if t.category == "a"),
        "found_category_b": sum(1 for t in targets if t.category == "b"),
        "in_scope_total": len(in_scope),
        "in_scope_category_a": sum(1 for t in in_scope if t.category == "a"),
        "in_scope_category_b": sum(1 for t in in_scope if t.category == "b"),
        "targets": [
            {
                "id": t.mid[:8], "category": t.category, "on_branch": t.on_branch,
                "branch_pos": t.branch_pos, "age_minutes": round(t.age_minutes, 1) if t.age_minutes is not None else None,
                "json_content_len": t.json_len, "table_content_len": t.table_len,
                "in_scope": t.in_scope, "skip_reason": t.skip_reason,
            }
            for t in targets
        ],
    }


def emit(result, as_json):
    if as_json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    print(f"chat {result.get('chat_id', '?')[:8]}: found {result['found_total']} not-fully-done assistant "
          f"message(s) — category (a) stale-unfinished: {result['found_category_a']}, "
          f"category (b) json-only-gap: {result['found_category_b']}")
    print(f"in scope for this run: {result['in_scope_total']} "
          f"(a: {result['in_scope_category_a']}, b: {result['in_scope_category_b']})")
    for t in result["targets"]:
        pos = f"pos {t['branch_pos']}" if t["branch_pos"] else "off-branch"
        scope = "TARGET" if t["in_scope"] else f"skip ({t['skip_reason']})"
        print(f"  {t['id']} [{t['category']}] {pos} age={t['age_minutes']}m "
              f"json_len={t['json_content_len']} table_len={t['table_content_len']} — {scope}")
    if "written" in result:
        print(f"written: {result['written']} | verification: {'OK' if result['verify_ok'] else 'FAILED'}")
        print(f"integrity: {result.get('integrity')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("--chat", default=DEFAULT_CHAT_ID,
                     help="chat id or id-prefix (defaults to her conversation)")
    branch_group = ap.add_mutually_exclusive_group()
    branch_group.add_argument("--branch-only", action="store_true", default=True,
                               help="only touch messages on the current branch (default)")
    branch_group.add_argument("--all-branches", action="store_true",
                               help="also touch off-branch messages")
    ap.add_argument("--min-age-minutes", type=float, default=DEFAULT_MIN_AGE_MINUTES)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--restore", metavar="STAMP", default=None)
    a = ap.parse_args()
    branch_only = not a.all_branches

    if a.restore:
        dest, msg = restore_db(a.db, a.restore)
        result = {"action": "restore", "stamp": a.restore, "ok": dest is not None, "detail": msg}
        if a.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print(msg)
        sys.exit(0 if dest is not None else 1)

    if a.apply:
        who = openers(a.db)
        if who:
            print(f"REFUSING: the database is open by process(es) {who}. Stop openwebui "
                  f"and backup, close every browser tab on the chat, and retry.")
            sys.exit(1)
        sidecars = live_sidecars(a.db)
        if sidecars:
            print(f"REFUSING: {sidecars} present — see {JOURNAL_RUNBOOK}, not this script, "
                  f"for a live WAL/journal.")
            sys.exit(1)

    con = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True) if not a.apply else sqlite3.connect(a.db, timeout=60, isolation_level=None)

    found = find_chat(con, a.chat)
    if found is None:
        print(f"REFUSING: no single chat matches --chat {a.chat!r}.")
        sys.exit(1)
    cid, blob, cur_col, updated_before = found
    chat = json.loads(blob)
    table = load_table(con, cid)

    targets = build_targets(chat, table, cur_col, a.min_age_minutes, branch_only)
    result = summarize(targets)
    result["chat_id"] = cid
    in_scope = [t for t in targets if t.in_scope]

    if not a.apply:
        emit(result, a.json)
        if not in_scope:
            if not a.json:
                print("nothing to do")
            sys.exit(0)
        if not a.json:
            print("DRY RUN — nothing written")
        sys.exit(3)

    # --apply from here
    if not in_scope:
        emit(result, a.json)
        if not a.json:
            print("nothing to do — nothing written")
        con.close()
        sys.exit(0)

    # pre-write snapshot of EVERY message's content, for the post-write
    # byte-identical check (module docstring, VERIFICATION)
    pre_json_content = {mid: m.get("content") for mid, m in chat.get("history", {}).get("messages", {}).items()}
    pre_table_content = {mid: r["content"] for mid, r in table.items()}
    pre_json_done = {mid: m.get("done") for mid, m in chat.get("history", {}).get("messages", {}).items()}
    pre_table_done = {mid: r["done"] for mid, r in table.items()}

    stamp = utc_stamp()
    try:
        bak = backup_db(a.db, stamp)
    except (OSError, FileExistsError) as e:
        print(f"REFUSING: backup failed: {e}")
        con.close()
        sys.exit(1)
    if not a.json:
        print(f"backed up to {bak}")

    con.execute("BEGIN IMMEDIATE")
    now_updated = con.execute("select updated_at from chat where id=?", (cid,)).fetchone()[0]
    if now_updated != updated_before:
        con.execute("ROLLBACK")
        print("the chat changed while this ran; nothing written")
        sys.exit(1)

    hist = chat["history"]["messages"]
    for t in in_scope:
        hist[t.mid]["done"] = True
    new_blob = json.dumps(chat)
    con.execute("update chat set chat=? where id=?", (new_blob, cid))
    for t in in_scope:
        if t.category == "a":  # category b's table row is already done=1
            row = table[t.mid]
            con.execute("update chat_message set done=1 where id=?", (row["row"],))
    con.execute("COMMIT")

    # post-commit re-read and verification (see module docstring)
    back_blob, back_updated = con.execute("select chat, updated_at from chat where id=?", (cid,)).fetchone()
    back_chat = json.loads(back_blob)
    back_hist = back_chat["history"]["messages"]
    back_table = load_table(con, cid)

    in_scope_ids = {t.mid for t in in_scope}
    unexpected_json_change = [mid for mid in pre_json_done
                               if back_hist.get(mid, {}).get("done") != pre_json_done[mid]
                               and mid not in in_scope_ids]
    unexpected_table_change = [mid for mid in pre_table_done
                                if back_table.get(mid, {}).get("done") != pre_table_done[mid]
                                and mid not in in_scope_ids]
    flags_ok = all(back_hist.get(t.mid, {}).get("done") is True for t in in_scope) and \
        all(back_table.get(t.mid, {}).get("done") == 1 for t in in_scope) and \
        not unexpected_json_change and not unexpected_table_change

    content_ok = all(back_hist.get(mid, {}).get("content") == c for mid, c in pre_json_content.items()) and \
        all((back_table.get(mid, {}).get("content")) == c for mid, c in pre_table_content.items())

    updated_at_ok = back_updated == updated_before
    integrity = con.execute("PRAGMA integrity_check").fetchone()[0]
    verify_ok = flags_ok and content_ok and updated_at_ok and integrity == "ok"

    result["written"] = len(in_scope)
    result["verify_ok"] = verify_ok
    result["integrity"] = integrity
    result["backup"] = str(bak)
    result["restore_stamp"] = stamp
    emit(result, a.json)
    con.close()

    if not verify_ok:
        if not a.json:
            print(f"VERIFICATION FAILED after an already-committed write. Restore with: "
                  f"--restore {stamp}")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
