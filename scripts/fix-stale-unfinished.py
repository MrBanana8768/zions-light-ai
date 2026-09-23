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

BRANCH CONSTRUCTION MUST BE TABLE-FIRST (fixed after a live-pod bug,
2026-09-23). Once the pod moved to OpenWebUI 0.11.4, the JSON copy started
lagging the table — some messages exist ONLY in `chat_message`, not yet
folded into `chat.chat["history"]["messages"]` at all. A first version of
this script walked ancestors through the JSON dict alone; on the real
chat that walk died after 9 hops (the first table-only gap), reported the
tip at branch position 9 instead of ~3,889, and reclassified every real
target past that gap — including `dd9292d9` at the true position ~2672 —
as "off-branch", finding 0 in scope. Nothing was written (the guards did
their job), but the report was silently wrong. Fixed by walking a
TABLE-FIRST merge (`merged_for_walk`): every `chat_message` row counts as
a real node for parent/child purposes even when the JSON has never heard
of it, exactly the shape `get_messages_map_by_chat_id` and
`repair-chat-tree.py`'s own `build_plan` merge already use. Unlike that
script, this merge is READ-ONLY and never written back — see CATEGORIES
and THE WRITE below for why a table-only message stays out of the JSON
forever rather than being folded in.

CATEGORIES (all fixed by `--apply`; reported as separate counts):
  (a) STALE UNFINISHED — `done` is not `True` in the JSON (missing,
      `False`, or the message doesn't exist in the JSON at all) AND the
      table's `done` column is not `1` (0 or missing). Both copies agree
      the message never finished — this is the actual spinner bug. Fixed
      by setting `done=True` in the JSON (if a JSON entry exists — see
      the table-only note below) and `done=1` in the table.
      TABLE-ONLY variant: when the message has no JSON entry at all
      (a table-only row not yet folded into the JSON — see BRANCH
      CONSTRUCTION above), there is nothing to set `done=True` ON in the
      JSON, so only `chat_message.done` is written; reported with
      `json_present: false` so this is visible, not silent. A table-only
      row whose `done` is already `1` is not a candidate at all — its
      only copy already agrees, nothing to fix and nothing to report.
  (b) JSON-ONLY GAP — the message DOES have a JSON entry, its `done` key
      is MISSING there (not `False`), while the table already says
      `done=1`. The table is authoritative, so this was never rendering
      as a live spinner the same way (a) does; it is a plain hygiene
      backfill, not the bug fix. Fixed by setting `done=True` in the
      JSON only (the table needs no write). Counted and printed
      separately from (a) so a dry run cannot make (b) look like
      additional instances of the actual bug.
  Measured on the real 2026-09-23 backup with the ORIGINAL (JSON-only)
  branch walk, now known wrong: (a) 19 total (6 on the current branch,
  13 off it), (b) 9 total (0 on the current branch, all off it). Rerun
  after this fix — see the module's own test suite and CHANGELOG-style
  note in the commit for the corrected, table-first numbers.

TARGETING RULES.
  - `--chat` selects the conversation (exact id or unique prefix), default
    her conversation.
  - `--branch-only` (default) restricts BOTH categories to messages that
    are ancestors of the stored pointer (`chat.current_message_id`, or
    `history.currentId` if the column is unset — same fallback order as
    `repair-chat-tree.py`'s `pick_stored_pointer`), walked table-first per
    BRANCH CONSTRUCTION above. `--all-branches` lifts that restriction.
    If the pointer's own walk does not reach a genuine root (a broken
    link or a cycle — a tree-structure problem, not this script's job),
    this REFUSES outright (`BranchWalkIncomplete`) rather than silently
    treating the unreached remainder as off-branch, the same silent-wrong
    shape the live-pod bug above took.
  - The current tip itself, and any of its direct children (an in-flight
    reply actually generating right now would be exactly this: a fresh
    assistant child of the pointer, table-only if it is new enough that
    OpenWebUI has not yet folded it into the JSON), are NEVER touched,
    regardless of the branch/age flags below — this is the one exclusion
    this script will not override.
  - `--min-age-minutes` (default 10): a candidate is only a target if it
    is older than this, using `max(chat_message.created_at,
    chat_message.updated_at)` (falling back to the JSON message's own
    `timestamp` field for a message that somehow has no table row at
    all). This is a second, independent guard against touching a reply
    that is genuinely mid-generation — belt-and-suspenders alongside the
    tip exclusion above, not a replacement for it.

THE WRITE. Sets `done=True`/`done=1` for exactly the in-scope targets, in
whichever copy actually has an entry for that message (a table-only
target gets only `chat_message.done=1` — see CATEGORIES). Nothing else
changes: not `content`, not `created_at`/`updated_at` on either copy, not
`chat.updated_at`, not `current_message_id`, not the flat
`chat.chat["messages"]` list, and a table-only message is never folded
into the JSON (that permanent merge is `repair-chat-tree.py`'s job, not
this script's). `chat.updated_at` is read once before the write (to
detect a concurrent change, the same guard `repair-chat-tree.py` uses)
and deliberately never written back — this fix is plumbing, not a
user-visible edit, and must not appear as one.

REFUSALS (checked before any backup or write): another process holding
`webui.db` open (`/proc/<pid>/fd` scan — same check
`scripts/clean-decoration.py` and `scripts/repair-chat-tree.py` use), a
`-wal`/`-journal` sidecar beside it (that failure mode belongs to
RUNBOOK_DB_JOURNAL.md, not this script), or the pointer's branch walk not
reaching a genuine root (see TARGETING RULES above).

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


class BranchWalkIncomplete(Exception):
    """The pointer's ancestor walk did not reach a genuine root (a node
    whose own `parentId` is `None`) — a broken link or a cycle. Raised
    instead of silently truncating the branch and mis-classifying
    everything past the break as "off-branch" (the real bug this pass
    fixes: `dd9292d9` at the real branch position ~2672 read as
    off-branch and 0-in-scope on a live pod purely because the walk had
    already died at position 9)."""

    def __init__(self, pointer, stopped_at, parent_id):
        self.pointer, self.stopped_at, self.parent_id = pointer, stopped_at, parent_id
        super().__init__(
            f"branch walk from pointer {pointer!r} stopped at {stopped_at!r} "
            f"(parentId={parent_id!r}) without reaching a root"
        )


def merged_for_walk(hist, table):
    """A read-only merge of the JSON history and the `chat_message` table,
    for WALKING the branch only — table rows first, filling in only the
    ids the JSON does not have at all, mirroring `get_messages_map_by_chat_id`
    (table primary, JSON fills gaps) and `repair-chat-tree.py`'s own
    `build_plan` merge step. UNLIKE that script, this merge is never
    written back: this script's whole mandate is to touch only `done`
    flags, never to fold a table-only message permanently into the JSON
    blob, so a table-only node stays out of `chat.chat` forever — this
    dict exists only so a branch walk does not die at the first gap
    (the real pod bug this pass fixes: OpenWebUI 0.11.4's JSON lagged the
    table by several messages, several of them mid-branch, and the old
    JSON-only walk stopped after 9 hops instead of the true 3,889)."""
    msgs = {}
    for mid, m in hist.items():
        msgs[mid] = {"parentId": m.get("parentId"), "childrenIds": [], "role": m.get("role")}
    for mid, r in table.items():
        if mid not in msgs:
            msgs[mid] = {"parentId": r.get("parentId"), "childrenIds": [], "role": r.get("role")}
    for mid, m in msgs.items():
        p = m.get("parentId")
        if p in msgs and mid not in msgs[p]["childrenIds"]:
            msgs[p]["childrenIds"].append(mid)
    return msgs


def branch_from_pointer(merged, pointer):
    """Ancestors of `pointer` (root-most last), or raises
    `BranchWalkIncomplete` if the walk does not end at a genuine root —
    see that exception's own docstring for why this refuses rather than
    treating the unreached remainder as simply off-branch."""
    path = walk_up(merged, pointer)
    last = path[-1] if path else None
    if last is None or merged.get(last, {}).get("parentId") is not None:
        raise BranchWalkIncomplete(pointer, last, merged.get(last, {}).get("parentId") if last else None)
    return path


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
                 "json_len", "table_len", "json_present", "in_scope", "skip_reason")

    def __init__(self, mid, category, on_branch, branch_pos, age_minutes, json_len, table_len, json_present):
        self.mid = mid
        self.category = category
        self.on_branch = on_branch
        self.branch_pos = branch_pos
        self.age_minutes = age_minutes
        self.json_len = json_len
        self.table_len = table_len
        self.json_present = json_present
        self.in_scope = True
        self.skip_reason = None


def build_targets(chat, table, cur_col, min_age_minutes, branch_only, now=None):
    """Raises `BranchWalkIncomplete` if a stored pointer exists but its
    ancestor walk (table-first, see `merged_for_walk`) does not reach a
    genuine root — see that exception's docstring."""
    now = now if now is not None else time.time()
    hist = chat.get("history", {}).get("messages", {})
    hist_current = chat.get("history", {}).get("currentId")
    merged = merged_for_walk(hist, table)
    pointer = pick_stored_pointer(cur_col, hist_current, merged)

    branch = set()
    branch_pos = {}
    protected = set()
    if pointer:
        path = branch_from_pointer(merged, pointer)  # raises BranchWalkIncomplete on a broken/cyclic chain
        branch = set(path)
        branch_pos = {mid: i + 1 for i, mid in enumerate(reversed(path))}  # root..pointer
        protected.add(pointer)
        for child in merged.get(pointer, {}).get("childrenIds") or []:
            if merged.get(child, {}).get("role") == "assistant":
                protected.add(child)

    targets = []
    for mid in set(hist) | set(table):
        hist_m = hist.get(mid)
        row = table.get(mid)
        role = (hist_m or {}).get("role") or (row or {}).get("role")
        if role != "assistant":
            continue

        json_present = hist_m is not None
        json_done = json_present and hist_m.get("done") is True
        json_missing_key = json_present and "done" not in hist_m
        table_done = bool(row and row.get("done") == 1)

        if not json_present:
            if table_done:
                continue  # its only copy already says done -- nothing to fix, nothing to report
            category = "a"  # table-only AND not done: set chat_message.done only (see THE WRITE)
        elif not json_done and not table_done:
            category = "a"
        elif json_missing_key and table_done:
            category = "b"
        else:
            continue  # already done in both, or the (unmentioned) reverse case — not this script's job

        on_b = mid in branch
        t = Target(
            mid, category, on_b, branch_pos.get(mid),
            msg_age_minutes(mid, hist, table, now),
            content_len(hist_m.get("content")) if hist_m else None,
            content_len(row.get("content")) if row else None,
            json_present,
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
    table_only = [t for t in targets if not t.json_present]
    return {
        "found_total": len(targets),
        "found_category_a": sum(1 for t in targets if t.category == "a"),
        "found_category_b": sum(1 for t in targets if t.category == "b"),
        "found_table_only": len(table_only),
        "in_scope_total": len(in_scope),
        "in_scope_category_a": sum(1 for t in in_scope if t.category == "a"),
        "in_scope_category_b": sum(1 for t in in_scope if t.category == "b"),
        "in_scope_table_only": sum(1 for t in in_scope if not t.json_present),
        "targets": [
            {
                "id": t.mid[:8], "category": t.category, "on_branch": t.on_branch,
                "branch_pos": t.branch_pos, "age_minutes": round(t.age_minutes, 1) if t.age_minutes is not None else None,
                "json_content_len": t.json_len, "table_content_len": t.table_len,
                "json_present": t.json_present, "in_scope": t.in_scope, "skip_reason": t.skip_reason,
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
          f"category (b) json-only-gap: {result['found_category_b']}, "
          f"of which {result['found_table_only']} exist ONLY in the table (JSON copy lacks them entirely)")
    print(f"in scope for this run: {result['in_scope_total']} "
          f"(a: {result['in_scope_category_a']}, b: {result['in_scope_category_b']}, "
          f"table-only: {result['in_scope_table_only']})")
    for t in result["targets"]:
        pos = f"pos {t['branch_pos']}" if t["branch_pos"] else "off-branch"
        scope = "TARGET" if t["in_scope"] else f"skip ({t['skip_reason']})"
        where = "table-only, JSON copy lacks it" if not t["json_present"] else "both copies"
        print(f"  {t['id']} [{t['category']}] {pos} age={t['age_minutes']}m "
              f"json_len={t['json_content_len']} table_len={t['table_content_len']} ({where}) — {scope}")
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

    try:
        targets = build_targets(chat, table, cur_col, a.min_age_minutes, branch_only)
    except BranchWalkIncomplete as e:
        print(f"REFUSING: {e}. This is a tree-structure problem (a broken link or a cycle), "
              f"not something this script repairs — run scripts/repair-chat-tree.py first.")
        con.close()
        sys.exit(1)
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
        if t.json_present:  # a table-only target has no JSON entry to set -- and none is created (THE WRITE)
            hist[t.mid]["done"] = True
    new_blob = json.dumps(chat)
    con.execute("update chat set chat=? where id=?", (new_blob, cid))
    for t in in_scope:
        if t.category == "a" and t.mid in table:  # category b's table row is already done=1
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
    flags_ok = all(back_hist.get(t.mid, {}).get("done") is True for t in in_scope if t.json_present) and \
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
