#!/usr/bin/env python3
"""Undo the encoding damage the retired v1 chat-tree repair did to a chat.

v1 (retired; see RUNBOOK_CHAT_TREE.md's "Never do these") copied messages
that existed only in the `chat_message` table into the chat's JSON history
using the table's RAW, still-JSON-encoded content (chat_messages.py:141's
`Column(JSON, ...)` means `sqlite3`, read directly and bypassing the ORM,
sees the encoded text — see scripts/repair-chat-tree.py's module docstring,
point 2, for the full trace through the pinned image's own source). Those
messages render with a literal leading quote and literal `\\n`. A browser
save afterwards writes the history back into the table, encoding it AGAIN,
so the damage can also be in the table — which is what the model actually
reads (repair-chat-tree.py's module docstring, point 5).

This script restores each affected message from a PRE-REPAIR copy of the
database, where its table content is still the original, correct value.

WHAT CHANGED FROM ANDREW'S ORIGINAL (origin/fix/ops-chat-tree 193813b,
imported verbatim two commits back). Brought to the same rigour as
scripts/backfill-records.py, scripts/import-history.py and the hardened
scripts/repair-chat-tree.py:
  * `--chat <id-or-prefix>`, defaulting to her conversation
    (ea1494ea-e9d7-46fb-8b7c-3a50d685d00e) — the original picked
    "whichever chat has the most chat_message rows", which silently
    targets the wrong chat the moment any clone or test fixture in the
    same database is bigger than hers (she has clones of 48.8 MB against
    her own 55.6 MB chat — close enough that a slightly larger throwaway
    clone would have won outright).
  * dry-run by default (unchanged), `--apply`, `--json`, the shared
    5-value exit code convention (OPERATIONS.md), its own backup of the
    LIVE db before any write (free-space check, then a plain byte copy,
    verified with PRAGMA integrity_check — see repair-chat-tree.py's
    BACKUP section for why a plain copy, not sqlite3's own backup() API,
    is the right tool here), and `--restore <stamp>`.
  * the liveness refusal now also checks for a `-wal`/`-journal` sidecar
    beside the LIVE db (that case belongs to RUNBOOK_DB_JOURNAL.md, not
    here), alongside the original's `/proc` openers scan.
  * `decode_once` is unchanged in what it accepts back from `json.loads`
    (str only) — DELIBERATELY, unlike repair-chat-tree.py's more general
    `decode()`. This script's whole logic (`hist_bad`/`table_bad` below)
    assumes the GOOD value is the TEXT content a v1 run would have
    mangled; a message whose real content is a list of multimodal blocks
    was never affected by v1's specific bug in the first place (v1 only
    ever touched the `content` string it copied verbatim; a list value
    copied verbatim is still the correct list, byte for byte, needing no
    restoration).  Broadening this one would risk treating an intact
    list-content message as a mismatch.

Usage:
  fix-encoded-messages.py <live webui.db> <pre-repair webui.db> \\
      [--chat <id-or-prefix>] [--apply] [--json] [--restore <stamp>]

Dry run by default. Output is counts and short ids only — never message
text.
"""

import argparse
import datetime
import hashlib
import json
import os
import shutil
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _webui_live_path as _live  # noqa: E402

DEFAULT_CHAT_ID = "ea1494ea-e9d7-46fb-8b7c-3a50d685d00e"
JOURNAL_RUNBOOK = "RUNBOOK_DB_JOURNAL.md"
TOOL_NAME = "fix-encoded-messages.py"


def utc_stamp():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


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


def check_free_space(path, needed_bytes):
    free = shutil.disk_usage(Path(path).parent).free
    required = int(needed_bytes * 1.2) + 10 * 1024 * 1024
    if free < required:
        return (f"ERROR: only {free / 1e6:.1f} MB free at {Path(path).parent}, need "
                f"~{required / 1e6:.1f} MB to back up {path}. Refusing without a successful backup.")
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


def _backup_target(db_path, stamp):
    """Where this run's own safety backup goes. D10: when `db_path` IS the
    local, ephemeral disk file, a backup left beside it rides the same
    container overlay and is lost on a pod stop -- so it goes to the
    durable `/data/forensics/fix-encoded-messages-<stamp>/` instead.
    Unchanged (beside `db_path`) in every other case."""
    src = Path(db_path)
    forensics_dir = _live.forensics_backup_dir("fix-encoded-messages", stamp, src)
    if forensics_dir is not None:
        return forensics_dir / (src.name + f".bak-{stamp}")
    return src.with_name(src.name + f".bak-{stamp}")


def backup_db(db_path, stamp):
    src = Path(db_path)
    bak = _backup_target(db_path, stamp)
    bak.parent.mkdir(parents=True, exist_ok=True)
    err = check_free_space(bak, src.stat().st_size)
    if err:
        raise OSError(err)
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
    bak = _backup_target(db_path, stamp)
    if not bak.is_file():
        return None, f"REFUSED: no backup found at {bak}"
    shutil.copy2(bak, src)
    return bak, f"restored {src} from {bak}"


def decode_once(s):
    try:
        v = json.loads(s)
        return v if isinstance(v, str) else None
    except Exception:
        return None


def find_chat_row(con, chat_id_arg):
    """Exact id or id-prefix match — NEVER "the chat with the most rows"
    (see module docstring). Requires exactly one match."""
    rows = list(con.execute("select distinct chat_id from chat_message"))
    matches = [r[0] for r in rows if r[0] == chat_id_arg or r[0].startswith(chat_id_arg)]
    return matches[0] if len(matches) == 1 else None


def plan_fix(live_con, pre_path, cid):
    src = sqlite3.connect(f"file:{pre_path}?mode=ro", uri=True, timeout=60)
    pre_row = src.execute("select chat from chat where id=?", (cid,)).fetchone()
    if pre_row is None:
        src.close()
        return None, f"chat {cid!r} not found in the pre-repair copy {pre_path}"
    pre_hist = json.loads(pre_row[0])["history"]["messages"]
    prefix = cid + "-"
    pre_table = {r[0][len(prefix):]: r[1] for r in src.execute(
        "select id, content from chat_message where chat_id=?", (cid,))}
    src.close()

    # the messages the v1 repair copied: in the table, not in the history,
    # before that repair — i.e. table-only messages that got a decoded
    # copy pushed into the JSON in ENCODED form.
    targets = {m: raw for m, raw in pre_table.items() if m not in pre_hist and decode_once(raw) is not None}

    blob, updated = live_con.execute("select chat, updated_at from chat where id=?", (cid,)).fetchone()
    chat = json.loads(blob)
    hist = chat["history"]["messages"]
    live_table = {r[0][len(prefix):]: r[1] for r in live_con.execute(
        "select id, content from chat_message where chat_id=?", (cid,))}

    fix_hist, fix_table, already_ok, skipped = [], [], 0, []
    for mid, good_raw in targets.items():
        good = decode_once(good_raw)
        h = (hist.get(mid) or {}).get("content")
        t = live_table.get(mid)
        hist_bad = h is not None and h != good and h in (good_raw, decode_once(t) if t else None)
        table_bad = t is not None and t != good_raw and decode_once(t) == good_raw
        if h is not None and h != good and not hist_bad:
            skipped.append(mid)
            continue
        if hist_bad:
            fix_hist.append(mid)
        if table_bad:
            fix_table.append(mid)
        if not hist_bad and not table_bad:
            already_ok += 1

    for mid in fix_hist:
        hist[mid]["content"] = decode_once(targets[mid])
    new_blob = json.dumps(chat)

    # H-VERIFY (bug found while testing against the real 09-22/09-23
    # backups: neither has ever actually been through the retired v1
    # repair, so most "targets" here are ordinary not-yet-synced table
    # rows, never copied into the history at all — h is None, by design,
    # not by damage). The original script's verification compared such a
    # message's (nonexistent) history content against `good` unconditionally
    # and always lost, failing verification on data v1 never touched. Only
    # check a copy that either WAS a fix target, or already existed before
    # this ran — a copy this script never expected to exist yet (still
    # table-only, exactly what repair-chat-tree.py, not this script, syncs)
    # is not this script's concern and is not evidence of anything wrong.
    bad_after = []
    for mid, good_raw in targets.items():
        if mid in skipped:
            continue
        good = decode_once(good_raw)
        if mid in fix_hist or (hist.get(mid) or {}).get("content") is not None:
            if hist.get(mid, {}).get("content") != good:
                bad_after.append(mid)
                continue
        t = good_raw if mid in fix_table else live_table.get(mid)
        if t is not None and decode_once(t) != good:
            bad_after.append(mid)

    plan = {
        "chat_id": cid, "targets": len(targets), "fix_hist": fix_hist, "fix_table": fix_table,
        "already_ok": already_ok, "skipped": skipped, "ok": not bad_after,
        "new_blob": new_blob, "updated_before": updated, "raw_targets": targets,
    }
    return plan, None


def emit(plan, as_json):
    if as_json:
        out = {k: v for k, v in plan.items() if k not in ("new_blob", "raw_targets")}
        out["fix_hist"] = [m[:8] for m in plan["fix_hist"]]
        out["fix_table"] = [m[:8] for m in plan["fix_table"]]
        out["skipped"] = [m[:8] for m in plan["skipped"]]
        print(json.dumps(out, indent=2, sort_keys=True))
        return
    print(f"chat {plan['chat_id'][:8]}: {plan['targets']} message(s) the v1 repair copied into the history")
    print(f"history copies to restore: {len(plan['fix_hist'])} | table copies double-encoded, to restore: "
          f"{len(plan['fix_table'])} | already correct: {plan['already_ok']} | changed since, left alone: "
          f"{len(plan['skipped'])} {[m[:8] for m in plan['skipped']]}")
    print("verification against the pre-repair copy:", "OK" if plan["ok"] else "FAILED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("live", nargs="?")
    ap.add_argument("pre", nargs="?")
    ap.add_argument("--chat", default=DEFAULT_CHAT_ID,
                    help="chat id or id-prefix to fix (defaults to her conversation, "
                         "NOT the chat with the most chat_message rows)")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--restore", metavar="STAMP", default=None)
    a = ap.parse_args()

    if a.restore:
        if not a.live:
            print("usage: fix-encoded-messages.py <live webui.db> --restore <stamp>")
            sys.exit(2)
        _live.refuse_if_snapshot_in_local_mode(a.live, tool_name=TOOL_NAME, dry_run=False)
        bak, msg = restore_db(a.live, a.restore)
        print(msg)
        if bak is not None:
            _live.maybe_print_final_sync_hint(TOOL_NAME, a.live)
        sys.exit(0 if bak is not None else 1)

    if not a.live or not a.pre:
        print(__doc__)
        sys.exit(2)

    _live.refuse_if_snapshot_in_local_mode(a.live, tool_name=TOOL_NAME, dry_run=not a.apply)

    if a.apply:
        who = openers(a.live)
        if who:
            print(f"REFUSING: the live database is open by process(es) {who}. "
                  f"Stop openwebui and backup, close every browser tab on the chat, and retry.")
            sys.exit(1)
        sidecars = live_sidecars(a.live)
        if sidecars:
            print(f"REFUSING: {sidecars} present — see {JOURNAL_RUNBOOK}, not this script, "
                  f"for a live WAL/journal.")
            sys.exit(1)

    con = sqlite3.connect(a.live, timeout=60, isolation_level=None)
    con.execute("BEGIN IMMEDIATE")

    cid = find_chat_row(con, a.chat)
    if cid is None:
        con.execute("ROLLBACK")
        print(f"REFUSING: no single chat matches --chat {a.chat!r} in the live database.")
        sys.exit(1)

    plan, err = plan_fix(con, a.pre, cid)
    if err:
        con.execute("ROLLBACK")
        print(f"REFUSING: {err}")
        sys.exit(1)

    emit(plan, a.json)
    nothing_pending = not plan["fix_hist"] and not plan["fix_table"]

    if not a.apply:
        con.execute("ROLLBACK")
        if nothing_pending and plan["ok"]:
            if not a.json:
                print("nothing to do")
            sys.exit(0)
        if not a.json:
            print("DRY RUN — nothing written")
        sys.exit(3)

    if not plan["ok"]:
        con.execute("ROLLBACK")
        print("verification FAILED; nothing written")
        sys.exit(1)
    if nothing_pending:
        con.execute("ROLLBACK")
        print("nothing to do — nothing written")
        sys.exit(0)

    stamp = utc_stamp()
    con.execute("ROLLBACK")
    try:
        bak = backup_db(a.live, stamp)
    except (OSError, FileExistsError) as e:
        print(f"REFUSING: backup failed: {e}")
        sys.exit(1)
    print(f"backed up to {bak}")

    con = sqlite3.connect(a.live, timeout=60, isolation_level=None)
    con.execute("BEGIN IMMEDIATE")
    now_updated = con.execute("select updated_at from chat where id=?", (cid,)).fetchone()[0]
    if now_updated != plan["updated_before"]:
        con.execute("ROLLBACK")
        print("the chat changed while this ran; nothing written")
        sys.exit(1)
    prefix = cid + "-"
    if plan["fix_hist"]:
        con.execute("update chat set chat=? where id=?", (plan["new_blob"], cid))
    for mid in plan["fix_table"]:
        con.execute("update chat_message set content=? where id=?", (plan["raw_targets"][mid], prefix + mid))
    con.execute("COMMIT")

    back = json.loads(con.execute("select chat from chat where id=?", (cid,)).fetchone()[0])["history"]["messages"]
    tb = {r[0][len(prefix):]: r[1] for r in con.execute("select id, content from chat_message where chat_id=?", (cid,))}
    good_n = sum(1 for m, g in plan["raw_targets"].items() if m not in plan["skipped"]
                 and back.get(m, {}).get("content") == decode_once(g) and decode_once(tb.get(m)) == decode_once(g))
    total = plan["targets"] - len(plan["skipped"])
    print(f"written. re-read: {good_n} of {total} messages correct in both copies")
    integrity = con.execute("PRAGMA integrity_check").fetchone()[0]
    print("integrity:", integrity)
    con.close()

    if integrity != "ok" or good_n < total:
        sys.exit(4 if good_n > 0 else 1)
    _live.maybe_print_final_sync_hint(TOOL_NAME, a.live)
    sys.exit(0)


if __name__ == "__main__":
    main()
