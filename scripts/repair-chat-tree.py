#!/usr/bin/env python3
"""Repair an OpenWebUI chat whose message tree has snapped parent links or a
stale display pointer, without losing or duplicating a single message.

v3 — supersedes Andrew's v2 (origin/fix/ops-chat-tree 193813b, imported
verbatim in the previous commit), which itself superseded a v1 retired for
the reasons RUNBOOK_CHAT_TREE.md records. v2 fixed all three of v1's
correctness defects and was run once, successfully, on the production pod
against her live chat before this hardening pass landed (see "ALREADY RUN
ON PRODUCTION" below for what that means for this script's own tests).
This pass keeps v2's relinking and verification design and changes four
things: a genuine pointer-selection bug (H-PTR below), a decoding bug for
non-text message content (H-DEC below), and brings the script to the same
operational rigour as scripts/backfill-records.py and
scripts/import-history.py — its own backup before any write, `--restore`,
`--json`, the shared 5-value exit code convention, and an explicit `--chat`
that defaults to HER conversation rather than guessing which one is hers.

WHAT WAS VERIFIED, AND HOW (all against the pinned image's own installed
`open_webui` package — `docker create --entrypoint sh
angreg/zions-light-ai@sha256:c1295894dd585784611c6833b1d4c396880ac8723e6b9b46531a5aa846cb8a65`,
then `docker cp` the files below out and read them):

  1. THE CHAT IS STORED TWICE — TRUE, and more specifically than "OpenWebUI
     0.11" describes: this image's `open_webui` is a fork with a second
     store on top of the upstream JSON-history design. `models/chats.py`
     keeps the tree as `chat.chat["history"]["messages"]`, a dict keyed by
     message id with `parentId`/`childrenIds` on each node (unchanged from
     upstream). `models/chat_messages.py` ALSO keeps one `chat_message` SQL
     row per message (`id`, composite as `"{chat_id}-{message_id}"`,
     `parent_id`, `role`, `content`, `created_at`, ...), written by
     `ChatTable.backfill_messages_by_chat_id` on every save
     (`upsert_message_to_chat_by_id_and_message_id`, chats.py:969-1016) and
     read back by `ChatMessageTable.get_messages_map_by_chat_id`
     (chat_messages.py:331), which is what `ChatTable.get_messages_map_by_chat_id`
     (chats.py:909) calls FIRST — the table, not the JSON, is this fork's
     PRIMARY copy; the JSON is read only to enrich rows the table's own
     `parent_id` graph cannot resolve (chats.py:918-947), and whatever gets
     enriched that way is written back into the table
     (`backfill_messages_by_chat_id`, chats.py:945). This is why v2's
     "prefer the table's own link" rule (kept here) is correct, not just a
     convenient heuristic: it is preferring the copy OpenWebUI itself
     already treats as authoritative.
  2. `chat_message.content` IS JSON-ENCODED — true for exactly the reader
     this script is (raw `sqlite3`, bypassing the ORM), and worth being
     precise about why. `chat_messages.py:141` declares
     `content = Column(JSON, nullable=True)  # Can be str or list of blocks`.
     SQLAlchemy's `JSON` type serializes the Python value with
     `json.dumps` before it reaches SQLite, which stores it as ordinary
     TEXT, and deserializes it with `json.loads` on the way back out
     THROUGH THE ORM. A script that opens the `.db` file directly with
     `sqlite3` (as this one must, on a stopped pod) never goes through the
     ORM and sees the raw encoded text — a plain string content of `hi`
     reads back as the four characters `"hi"` (with the quotes), not `hi`.
     v2's own `decode()` handled this for TEXT content, but returned the
     ENCODED string unchanged whenever the decoded value was not itself a
     `str` (H-DEC below) — silently missing the "or list of blocks" half
     of that same column comment, i.e. multimodal/tool-output content.
  3. `current_message_id` IS USED FOR RENDERING — true, and the fork adds
     a fallback chain worth naming precisely.
     `routers/chats.py:1274`: `current_message_id = chat.current_message_id
     or history.get('currentId')` — the `chat` TABLE COLUMN wins over the
     JSON blob's own `currentId` whenever the column is set. v2 already
     read it in that order (`cur_col or hist.get("currentId")`); this
     version keeps that order and additionally falls back to whichever of
     the two actually resolves to a message that still exists (H-PTR2).
  4. `merge_history`'S STALE-TAB BEHAVIOUR — matches the runbook's
     description, with one precision worth stating so nobody reads it as
     data loss. `models/chats.py:774`: `merged = {**existing, **incoming}`
     — a UNION of both tabs' message dicts, incoming winning only on an id
     both tabs share; nothing present only in `existing` is dropped. The
     pointer is different: `current_id = incoming.currentId if
     incoming.currentId in merged else existing.currentId if [...] else
     None` (chats.py:787-791) — a stale tab's OWN old `currentId` wins
     outright as long as that old message is still in the merged set,
     which it always is (nothing was deleted). That is exactly "every
     message still in the database, the pointer just landed on an old
     one" — the runbook's "neither loses data" line is correct; "lets it
     override stored messages" (RUNBOOK_CHAT_TREE.md's old wording, fixed
     in this pass) overstated it into sounding like content loss, when
     only the pointer field is actually clobbered.
  5. WHICH COPY IS SENT TO THE MODEL — this needed tracing through
     `utils/middleware.py`, not just chats.py/chat_messages.py.
     `process_chat_payload` (middleware.py:2248), for a SAVED chat with a
     `user_message_id` (middleware.py:2300), calls
     `load_messages_from_db(chat_id, user_message_id)` (middleware.py:2040)
     and, if it returns anything, REPLACES `form_data["messages"]` —
     the client's own submitted array — with it entirely
     (middleware.py:2317-2318), specifically because "DB preserves
     structured 'output' items which the frontend strips" (its own
     comment). `load_messages_from_db` calls
     `Chats.get_messages_map_by_chat_id` (the same table-preferring,
     JSON-enriching lookup from #1) and then `get_message_list(messages_map,
     message_id)` — the SAME parent-link walk that decides what she SEES.
     So for a saved chat, a broken link does not just mis-render her
     screen: the model itself receives a truncated context, walked the
     same way, from the same (primarily table-backed) copy. This is the
     concrete reason this repair is not merely cosmetic.

H-PTR (the pointer-selection bug, task-confirmed real). v2 always recomputed
"the newest LEAF, assistant preferred" and OVERWROTE `history["currentId"]`
with it — unconditionally, even when NOTHING was orphaned and her stored
pointer was already exactly where she left it. Concretely: she gets a
reply A1, regenerates it into a NEWER sibling A1' she does not like, clicks
back to A1 (her real, deliberate, fully-intact pointer), and does not
continue from there. v2's rule picks A1' — newer, unrelated, abandoned —
over her real position, EVEN THOUGH nothing about her actual branch was
broken. Fixed here (`_choose_pointer`): if the stored pointer resolves and
its own walk reaches the true root, it is kept and only walked FORWARD
along its OWN existing children (never sideways to an unrelated sibling
branch, and never backward) to whatever it has actually grown into since —
which is also how the genuine stale-tab case gets fixed: that pointer's
message DOES have real children by the time this runs (the live tab kept
going), so walking forward from it reaches the true tip; a message she
deliberately backed into and left alone has no children, so "walk forward"
is a no-op and it is not moved. Only when the stored pointer is missing or
its walk does not reach the true root (genuinely broken, not just old)
does this fall back to a global newest-leaf search — and even then, never
to a leaf older than the broken pointer's own timestamp (never regress).
Verified against both real backups (compactor/test_real_image_chat_tree.py):
on 2026-09-22 her stored pointer (8b2b6547) is intact and already IS the
newest leaf — unchanged either way. On 2026-09-23 it is NOT a no-op: her
stored pointer (fb6dc3bc) turns out to have real children once the 14
table-only messages this pass merges into the history are added — a
continuation that existed in `chat_message` but never made it into the
JSON blob — so walking forward from it lands on 107f5f0a
(2026-09-23 00:08:49Z). v2's own global newest-leaf search, on the SAME
merged data, lands on the identical message (107f5f0a happens to also be
the single newest leaf in the whole tree once those 14 rows are in), so
v2 and this version still agree on her actual data: production, which
already carries v2's applied choice (see "ALREADY RUN ON PRODUCTION"
above), is not sitting on a different pointer than this version would
have chosen. The bug this pass fixes is real (built as a synthetic test
below, `test_pointer_regenerate_then_back` in
compactor/test_repair_chat_tree_script.py) but the two backups available
for testing do not happen to exercise the divergence — both her pointer
and the abandoned-branch case would need to coexist in the SAME chat, and
neither backup has an abandoned regenerate branch newer than her real
position.

H-DEC (the decode bug). `decode()` returned the ENCODED string unchanged
whenever `json.loads` produced a non-`str` (fixed here to also decode
list/dict content — "list of blocks" per chat_messages.py:141 — while
still returning non-JSON text unchanged so a plain, un-encoded value is
never touched twice).

ALREADY RUN ON PRODUCTION. Andrew ran v2 with `--apply` on the pod before
this hardening pass was written; the live db is in a post-v2-repair state
that neither backup below reflects. Two things follow: (1) this script
must be a clean, harmless no-op against an ALREADY-repaired tree — dry run
finds nothing pending, exits 0, moves no pointer — and
compactor/test_repair_chat_tree_script.py has a case that applies the
ORIGINAL v2 to a copy and then runs this version against the result to
prove it; (2) on her real pre-repair data (both backups below), v2's
pointer choice and this version's agree (see H-PTR above) — production is
not carrying a different choice than this version would have made.

Real data checked (COPIES ONLY — see "REFUSALS" below and OPERATIONS.md):
/home/drew/pod-exports/2026-09-22 and .../2026-09-23, chat
ea1494ea-e9d7-46fb-8b7c-3a50d685d00e (this script's `--chat` default).
Measured directly (not the runbook's paraphrase, which used different
counts from an earlier pass): BOTH backups have one true root
(6c64570c..., 2026-08-24) plus THREE disconnected fragments — two orphans
(a79c97c7... a user message, and 139ca02c... an assistant message, both
declaring the identical missing parent id `f56f394a-070b-...`, which does
not exist anywhere in `chat_message` for either backup — a parent lost
outright, not merely orphaned, exercising exactly the "orphan whose true
parent is also missing" case) and one fully isolated singleton
(a3e2670d..., `parentId` genuinely `None`, not a dangling reference). The
stored pointer is intact in both backups, but only turns out to already
be the global newest leaf on 2026-09-22 — on 2026-09-23 it gains real
children once this pass's table-merge step runs (see H-PTR above) and
this script does move it forward, to the same message v2 also picks.

REFUSALS. Refuses `--apply` if any OTHER process holds the database file
open (`/proc/<pid>/fd` scan, same check scripts/clean-decoration.py
documents borrowing from this script) OR if a `-wal`/`-journal` sidecar
exists beside it (that failure mode has its own runbook,
RUNBOOK_DB_JOURNAL.md, and this script must not be the thing that decides
what to do about a live WAL). Both checks run before the backup, so a
refusal never leaves a partial backup behind.

BACKUP. A plain byte-for-byte copy (`shutil.copy2`), not the sqlite3
`Connection.backup()` API, for the same reason scripts/clean-decoration.py
gives for the identical choice: `Connection.backup()` produces a fresh page
layout, not a byte-identical file, and `--restore` here is required to
give back the EXACT original bytes (verified by md5 in
compactor/test_real_image_chat_tree.py), which only a plain copy
guarantees. The sqlite3 API is still used, read-only, to run
`PRAGMA integrity_check` against the fresh backup immediately (catching a
copy that silently failed) and again after `--restore` and after
`--apply`. Preceded by a free-space check (20% headroom, matching
clean-decoration.py's own margin for the same MooseFS bookkeeping reason).
Backup path: `<db>.bak-<UTC timestamp>`; `--restore <that timestamp>`
copies it back over the live path and refuses if no such backup exists.

Usage:
  repair-chat-tree.py <webui.db> [--chat <id-or-prefix>] [--apply] [--json]
  repair-chat-tree.py <webui.db> --restore <stamp>

Dry run by default. Output is counts, short ids and timestamps only — never
message text (see PART 6, `_redact`).

SUPPORTED INVOCATION on a pod: from a clone, like every other operator
script here (git ships in the image; there is no ssh/scp/rsync in it) —

    git clone --depth 1 --branch <tag-or-branch> \\
        https://github.com/MrBanana8768/zions-light-ai.git /opt/zl-repo
    /app/venv/bin/python /opt/zl-repo/scripts/repair-chat-tree.py \\
        /data/openwebui/webui.db

This script needs nothing from the compactor package (stdlib `sqlite3`
and `json` only), so `/app/venv/bin/python` (OpenWebUI's own venv) and
`/opt/compactor-venv/bin/python` both work; RUNBOOK_CHAT_TREE.md uses the
former since this runs on the `webui.db` side, not the compactor's.

EXIT CODES — the shared convention, OPERATIONS.md "Exit codes — the shared
convention across the operator scripts" (see that section for full
rationale; this script's own mapping):
    0   dry run found nothing to relink and the stored pointer already
        needs no correction (the desired end state is already true), OR
        `--apply` fixed everything it found (including a no-op `--apply`
        because there was nothing to fix) and verification passed, OR
        `--restore` succeeded.
    1   a refusal or an error to look at: bad `--db`/`--chat` (no matching
        chat), the liveness/WAL refusal, a backup that already exists at
        that timestamp, a free-space refusal, verification FAILING after
        `--apply` (rolled back — nothing written), or `--apply` leaving
        every orphan it targeted unlinked (indistinguishable from doing
        nothing, H1 in backfill-records.py's own docstring, same rule
        here), or `--restore` finding no backup at that stamp.
    2   argparse's own usage errors (missing/unknown flags) — the
        standard library's convention, unrelated to the four below.
    3   a DRY RUN found something to do (an orphan to relink, a pointer
        that would move) and nothing else needs attention — informational,
        "re-run with --apply once ready".
    4   `--apply` relinked at least one orphan but left at least one other
        one still unlinked (partial progress; look at what remains).
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


def fmt(t):
    if not t:
        return "?"
    if t > 1e11:
        t = t / 1e9
    return datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime("%m-%d %H:%M:%SZ")


def utc_stamp():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ===========================================================================
# PART 1 — liveness refusals
# ===========================================================================


def openers(path):
    """Every OTHER process on this host with `path` open, via /proc/<pid>/fd.
    Best-effort: a /proc read that fails (permissions, a pid that exited
    mid-scan) is skipped, never raised. See scripts/clean-decoration.py's
    `_openers`, which borrows this exact check from this script."""
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
    """`-wal`/`-journal` files beside the db. Their presence is a SEPARATE
    failure mode with its own runbook (RUNBOOK_DB_JOURNAL.md) — this
    script must refuse and point there, not guess what to do about a live
    WAL itself."""
    p = Path(db_path)
    return [str(s) for s in (p.with_name(p.name + "-wal"), p.with_name(p.name + "-journal")) if s.exists()]


# ===========================================================================
# PART 2 — backup / restore
# ===========================================================================


def check_free_space(path, needed_bytes):
    free = shutil.disk_usage(Path(path).parent).free
    required = int(needed_bytes * 1.2) + 10 * 1024 * 1024
    if free < required:
        return (f"ERROR: only {free / 1e6:.1f} MB free at {Path(path).parent}, need "
                f"~{required / 1e6:.1f} MB to back up {path} ({needed_bytes / 1e6:.1f} MB). "
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
    """A plain byte-for-byte copy — see the module docstring's BACKUP
    section for why this, not sqlite3's own backup API, is the right tool
    for the RESTORABLE copy. Verifies the fresh copy opens cleanly under
    PRAGMA integrity_check before returning, so a backup that failed to
    copy correctly is caught immediately, not at --restore time."""
    src = Path(db_path)
    err = check_free_space(src, src.stat().st_size)
    if err:
        raise OSError(err)
    bak = src.with_name(src.name + f".bak-{stamp}")
    if bak.exists():
        raise FileExistsError(f"backup already exists: {bak}")
    shutil.copy2(src, bak)
    err = verify_backup_openable(bak)
    if err:
        bak.unlink(missing_ok=True)
        raise OSError(f"backup of {src} failed integrity_check: {err}")
    return bak


def restore_db(db_path, stamp):
    src = Path(db_path)
    bak = src.with_name(src.name + f".bak-{stamp}")
    if not bak.is_file():
        return None, f"REFUSED: no backup found at {bak}"
    shutil.copy2(bak, src)
    return bak, f"restored {src} from {bak}"


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ===========================================================================
# PART 3 — tree mechanics
# ===========================================================================


def decode(s):
    """chat_message.content is stored JSON-encoded (H-DEC, see module
    docstring #2): give back the real value, str OR list/dict of blocks —
    v2 only ever returned a decoded str, silently leaving encoded text
    behind for any multimodal/tool-output message. A value that is not
    valid JSON, or that round-trips to something JSON-legal but was never
    actually encoded (rare, but decoding it AGAIN would corrupt it) is
    returned unchanged: this only "unwraps" a value once, and only when
    doing so does not need a second guess about whether it was encoded at
    all — the caller (copying a table-only message into the history) only
    ever calls this on content this script already knows came from the
    JSON column, so one unwrap is always correct there."""
    if not isinstance(s, str):
        return s
    try:
        v = json.loads(s)
    except Exception:
        return s
    return v


def walk_up(msgs, start):
    path, n = [], start
    while n in msgs and n not in path:
        path.append(n)
        n = msgs[n].get("parentId")
    return path


def forward_tip(msgs, start):
    """Walk DOWN from `start` through its OWN children only — never
    sideways to a sibling branch, never to an unrelated newer leaf
    elsewhere in the tree (H-PTR). Picks the child with the greatest
    timestamp at each fork, assistant preferred on an exact tie, matching
    the "newest reply, not the question it answers" preference v2 already
    applied globally; here it is applied one hop at a time, along the
    pointer's real lineage only. A childless `start` returns itself
    unchanged."""
    ts = lambda m: msgs[m].get("timestamp") or 0
    node, seen = start, {start}
    while True:
        kids = [c for c in (msgs[node].get("childrenIds") or []) if c in msgs and c not in seen]
        if not kids:
            return node
        node = sorted(kids, key=lambda c: (ts(c), msgs[c].get("role") == "assistant"))[-1]
        seen.add(node)


def pick_stored_pointer(cur_col, hist_current, msgs):
    """`chat.current_message_id` (the table column) wins over
    `history["currentId"]` whenever it resolves to a real message
    (routers/chats.py:1274's own order, module docstring #3) — H-PTR2
    over v2, which used `cur_col or hist.get("currentId")` and so fell
    through to `hist_current` only when `cur_col` was falsy, not when it
    was set but pointed at nothing (a case built in
    compactor/test_repair_chat_tree_script.py: the two disagree, and only
    one of them still exists)."""
    if cur_col and cur_col in msgs:
        return cur_col
    if hist_current and hist_current in msgs:
        return hist_current
    return cur_col or hist_current or None


def find_any_cycle(msgs):
    """Whole-graph cycle check, independent of the current pointer's own
    branch — v2's only cycle protection was implicit (walk_up checks made
    during relinking against a graph that is monotonically improving, see
    H-PTR/H-DEC companion reasoning in compactor/test_repair_chat_tree_script.py's
    docstring for why that protects the RELINKING pass itself), and never
    checked a cycle that might already exist among messages this script
    never touches (already-linked, not orphaned) — corruption from some
    other cause entirely. Returns the first cycle found as a list of ids,
    or None. Bounded: no node is visited via more than one origin, and
    walking never exceeds len(msgs) steps, so this always terminates even
    if the corruption is exactly a cycle."""
    state = {}  # id -> 0 in progress, 1 done
    for start in msgs:
        if state.get(start) == 1:
            continue
        path = []
        n = start
        while n in msgs and state.get(n) != 1:
            if n in path:
                return path[path.index(n):] + [n]
            path.append(n)
            n = msgs[n].get("parentId")
        for p in path:
            state[p] = 1
    return None


# ===========================================================================
# PART 4 — relink + repair plan
# ===========================================================================


class Plan:
    def __init__(self):
        self.chat_id = None
        self.n_history = 0
        self.n_table = 0
        self.flat_before = 0
        self.added_from_table = 0
        self.dangling_removed = 0
        self.relinks = {}         # orphan id -> new parent id
        self.relink_src = {}      # orphan id -> "table" | "time"
        self.unlinkable = []      # orphan ids left as their own root
        self.true_root = None
        self.old_pointer = None
        self.new_pointer = None
        self.pointer_moved = False
        self.pointer_reason = ""
        self.pointer_regressed = False
        self.roots_after = 0
        self.branch_after = 0
        self.ok = False
        self.detail = ""
        self.refused = None


def build_plan(chat, table, cur_col):
    plan = Plan()
    hist = chat.setdefault("history", {})
    msgs = hist.setdefault("messages", {})
    plan.n_history = len(msgs)
    plan.n_table = len(table)
    plan.flat_before = len(chat.get("messages") or [])
    ts = lambda m: msgs[m].get("timestamp") or 0

    for mid, r in table.items():
        if mid not in msgs:
            msgs[mid] = {
                "id": mid, "parentId": r["parentId"], "childrenIds": [], "role": r["role"],
                "content": decode(r["content"]),
                "timestamp": int(r["created"] if r["created"] < 1e11 else r["created"] / 1e9),
            }
            plan.added_from_table += 1

    dangling = 0
    for m in msgs.values():
        clean = [k for k in (m.get("childrenIds") or []) if k in msgs]
        dangling += len(m.get("childrenIds") or []) - len(clean)
        m["childrenIds"] = clean
    plan.dangling_removed = dangling

    orphans = sorted([m for m, v in msgs.items()
                      if v.get("parentId") is None or v.get("parentId") not in msgs], key=ts)
    true_root = orphans[0] if orphans else None
    if true_root:
        msgs[true_root]["parentId"] = None
    plan.true_root = true_root

    def subtree_size(node):
        stack, seen = [node], set()
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            stack.extend(c for c in (msgs[n].get("childrenIds") or []) if c in msgs)
        return len(seen)

    for mid in orphans[1:]:
        role = msgs[mid].get("role")
        parent, src = None, None
        t_parent = (table.get(mid) or {}).get("parentId")
        if t_parent in msgs and mid not in walk_up(msgs, t_parent):
            parent, src = t_parent, "table"
        else:
            # nearest earlier-or-equal message of the OPPOSITE role, not in
            # mid's own subtree; ties at the same timestamp broken by the
            # larger existing subtree (more likely the real thread than a
            # stray duplicate), then by id for a deterministic result.
            #
            # PERFORMANCE (found while smoke-testing against her real
            # 4,000+-message chat: this loop hung). walk_up() and
            # subtree_size() are each O(depth-of-branch), which on a real
            # chat is O(thousands). Computing either for every one of
            # thousands of same-role-eligible candidates before sorting is
            # O(n * depth) per orphan — tens of millions of Python-level
            # steps. Sorting on the CHEAP key (timestamp) first and only
            # paying for walk_up/subtree_size on candidates that actually
            # tie at the timestamp nearest `mid` (in practice a handful,
            # never all of `msgs`) keeps this to a handful of those calls
            # per orphan, same as v2's own early-break loop achieved by a
            # different means (breaking at the first accepted candidate).
            cheap = sorted(
                (x for x in msgs if x != mid and ts(x) <= ts(mid) and msgs[x].get("role") != role),
                key=ts, reverse=True,
            )
            parent = None
            i = 0
            while i < len(cheap):
                tie_ts = ts(cheap[i])
                j = i
                while j < len(cheap) and ts(cheap[j]) == tie_ts:
                    j += 1
                group = sorted(
                    (x for x in cheap[i:j] if mid not in walk_up(msgs, x)),
                    key=lambda x: (subtree_size(x), x), reverse=True,
                )
                if group:
                    parent = group[0]
                    break
                i = j
            if parent is not None:
                src = "time"
        if parent is None:
            plan.unlinkable.append(mid)
            continue
        msgs[mid]["parentId"] = parent
        if mid not in msgs[parent]["childrenIds"]:
            msgs[parent]["childrenIds"].append(mid)
        plan.relinks[mid] = parent
        plan.relink_src[mid] = src

    for mid, m in msgs.items():
        p = m.get("parentId")
        if p in msgs and mid not in msgs[p]["childrenIds"]:
            msgs[p]["childrenIds"].append(mid)

    # --- pointer (H-PTR: keep-and-walk-forward, never sideways/backward) ---
    hist_current = hist.get("currentId")
    orig = pick_stored_pointer(cur_col, hist_current, msgs)
    plan.old_pointer = orig
    orig_path = walk_up(msgs, orig) if orig in msgs else []
    intact = bool(orig_path) and orig_path[-1] == true_root

    ts2 = lambda m: msgs[m].get("timestamp") or 0
    if intact:
        newest = forward_tip(msgs, orig)
        plan.pointer_reason = "kept (branch intact); walked forward to its own tip" if newest != orig else "kept (branch intact, already its own tip)"
    else:
        # Broken: no reliable anchor to walk forward from, so fall back to
        # the newest leaf overall (assistant preferred on a tie).
        #
        # NON-REGRESSION (fixed here after mutation-testing this file
        # caught it: an earlier draft filtered the leaf pool to
        # `ts >= floor` before taking the max, which is a no-op by
        # construction -- adding OLDER candidates to a pool never changes
        # which one has the LARGEST timestamp, so the filter could never
        # once have changed the answer, and a fallback to the unfiltered
        # pool when the filter emptied it made it doubly inert). What
        # "never regress" can actually mean here: when even the newest
        # leaf available is older than the broken pointer's own
        # timestamp -- everything reachable is from BEFORE it, e.g. its
        # only children were themselves cut off by the same failure --
        # there is no candidate that does not regress, so the newest
        # leaf is still the least-bad choice, but that fact is recorded
        # (`plan.pointer_regressed`) rather than silently presented as an
        # ordinary repair.
        leaves = [m for m in msgs if not (msgs[m].get("childrenIds") or [])]
        pool = leaves or list(msgs)
        newest = sorted(pool, key=lambda m: (ts2(m), msgs[m].get("role") == "assistant"))[-1]
        plan.pointer_reason = "recomputed (stored pointer missing or its walk does not reach the true root)"
        if orig in msgs and ts2(newest) < ts2(orig):
            plan.pointer_regressed = True
            plan.pointer_reason += (
                f" — WARNING: the best available message ({fmt(ts2(newest))}) is still older "
                f"than the broken pointer's own timestamp ({fmt(ts2(orig))}); nothing reachable "
                f"is newer"
            )
    hist["currentId"] = newest
    plan.new_pointer = newest
    plan.pointer_moved = newest != orig

    roots_after = [m for m, v in msgs.items() if v.get("parentId") is None or v.get("parentId") not in msgs]
    plan.roots_after = len(roots_after)
    json_path = walk_up(msgs, newest)
    plan.branch_after = len(json_path)

    cycle = find_any_cycle(msgs)
    encoded_left = [m for m in msgs if isinstance(msgs[m].get("content"), str)
                    and msgs[m]["content"][:1] == '"' and decode(msgs[m]["content"]) != msgs[m]["content"]]
    plan.ok = (len(roots_after) == 1 and json_path and json_path[-1] == true_root
               and not cycle and not encoded_left)
    plan.detail = (f"roots {plan.roots_after} | branch from pointer {plan.branch_after}/{len(msgs)} "
                   f"| cycle: {'FOUND ' + str([c[:8] for c in cycle]) if cycle else 'none'} "
                   f"| encoded text left: {len(encoded_left)}")
    return plan, msgs


# ===========================================================================
# PART 5 — main
# ===========================================================================


def load_table(con, chat_id):
    prefix = chat_id + "-"
    table = {}
    for mid, parent_id, role, content, created in con.execute(
        "select id, parent_id, role, content, created_at from chat_message where chat_id=?", (chat_id,)
    ):
        key = mid[len(prefix):] if mid.startswith(prefix) else mid
        table[key] = {"row": mid, "parentId": parent_id, "role": role, "content": content, "created": created}
    return table


def find_chat(con, chat_arg):
    rows = list(con.execute("select id, chat, current_message_id from chat"))
    matches = [r for r in rows if r[0] == chat_arg or r[0].startswith(chat_arg)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        return None
    return None


def emit(result, as_json):
    if as_json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    p = result
    print(f"chat {p['chat_id'][:8]}: history {p['n_history']} | table {p['n_table']} | "
          f"flat list {p['flat_before']} (left untouched)")
    print(f"table-only messages copied into the history (decoded): {p['added_from_table']} | "
          f"dangling child ids removed: {p['dangling_removed']}")
    print(f"orphans re-linked: {len(p['relinks'])} ({sum(1 for v in p['relink_src'].values() if v == 'table')} "
          f"from the table's own links) | left unlinkable: {len(p['unlinkable'])}")
    for mid, parent in p["relinks"].items():
        print(f"  {mid[:8]} -> {parent[:8]} [{p['relink_src'][mid]}]")
    for mid in p["unlinkable"]:
        print(f"  {mid[:8]}: no usable parent, left as its own root")
    print(f"pointer: {(p['old_pointer'] or '-')[:8]} -> {(p['new_pointer'] or '-')[:8]} "
          f"({'moved' if p['pointer_moved'] else 'unchanged'}) — {p['pointer_reason']}")
    print(p["detail"])
    print("verification:", "OK" if p["ok"] else "FAILED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("--chat", default=DEFAULT_CHAT_ID,
                    help="chat id or id-prefix to repair (defaults to her conversation, NOT the largest chat)")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--restore", metavar="STAMP", default=None)
    a = ap.parse_args()

    if a.restore:
        bak, msg = restore_db(a.db, a.restore)
        result = {"action": "restore", "stamp": a.restore, "ok": bak is not None, "detail": msg}
        emit(result, a.json) if a.json else print(msg)
        sys.exit(0 if bak is not None else 1)

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
    if a.apply:
        con.execute("BEGIN IMMEDIATE")

    found = find_chat(con, a.chat)
    if found is None:
        print(f"REFUSING: no single chat matches --chat {a.chat!r}.")
        if a.apply:
            con.execute("ROLLBACK")
        sys.exit(1)
    cid, blob, cur_col = found
    updated_before = con.execute("select updated_at from chat where id=?", (cid,)).fetchone()[0]
    chat = json.loads(blob)
    table = load_table(con, cid)

    plan, msgs = build_plan(chat, table, cur_col)
    plan.chat_id = cid

    result = {
        "chat_id": cid, "n_history": plan.n_history, "n_table": plan.n_table,
        "flat_before": plan.flat_before, "added_from_table": plan.added_from_table,
        "dangling_removed": plan.dangling_removed,
        "relinks": {k[:8]: v[:8] for k, v in plan.relinks.items()},
        "relink_src": {k[:8]: v for k, v in plan.relink_src.items()},
        "unlinkable": [m[:8] for m in plan.unlinkable],
        "old_pointer": plan.old_pointer, "new_pointer": plan.new_pointer,
        "pointer_moved": plan.pointer_moved, "pointer_reason": plan.pointer_reason,
        "pointer_regressed": plan.pointer_regressed,
        "detail": plan.detail, "ok": plan.ok,
    }

    nothing_pending = not plan.relinks and not plan.unlinkable and not plan.pointer_moved and plan.ok

    if not a.apply:
        emit(result, a.json)
        if nothing_pending:
            if not a.json:
                print("nothing to do")
            sys.exit(0)
        if not a.json:
            print("DRY RUN — nothing written")
        sys.exit(3)

    # --apply from here
    emit(result, a.json)
    if not plan.ok:
        con.execute("ROLLBACK")
        print("verification FAILED — nothing written")
        sys.exit(1)
    if nothing_pending:
        con.execute("ROLLBACK")
        print("nothing to do — nothing written")
        sys.exit(0)

    stamp = utc_stamp()
    con.execute("ROLLBACK")  # release the write lock before the backup copy
    try:
        bak = backup_db(a.db, stamp)
    except (OSError, FileExistsError) as e:
        print(f"REFUSING: backup failed: {e}")
        sys.exit(1)
    print(f"backed up to {bak}")

    con = sqlite3.connect(a.db, timeout=60, isolation_level=None)
    con.execute("BEGIN IMMEDIATE")
    if con.execute("select updated_at from chat where id=?", (cid,)).fetchone()[0] != updated_before:
        con.execute("ROLLBACK")
        print("the chat changed while this ran; nothing written")
        sys.exit(1)
    con.execute("update chat set chat=?, current_message_id=?, updated_at=? where id=?",
                (json.dumps(chat), plan.new_pointer, int(time.time()), cid))
    for mid, parent in plan.relinks.items():
        if mid in table:
            con.execute("update chat_message set parent_id=? where id=?", (parent, table[mid]["row"]))
    con.execute("COMMIT")

    back = json.loads(con.execute("select chat from chat where id=?", (cid,)).fetchone()[0])
    bm = back["history"]["messages"]
    cur2 = con.execute("select current_message_id from chat where id=?", (cid,)).fetchone()[0]
    print(f"written. re-read: branch {len(walk_up(bm, cur2))} of {len(bm)} message(s), "
          f"flat list {len(back.get('messages') or [])} (was {plan.flat_before}), "
          f"row {len(json.dumps(back).encode())/1e6:.1f} MB")
    integrity = con.execute("PRAGMA integrity_check").fetchone()[0]
    print("integrity:", integrity)
    con.close()

    if integrity != "ok":
        sys.exit(1)
    if plan.unlinkable:
        sys.exit(4)
    sys.exit(0)


if __name__ == "__main__":
    main()
