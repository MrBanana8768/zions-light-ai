#!/usr/bin/env python3
"""Read-only inspector for the compactor's per-message chat journal.

    chat-journal.py stats   [--store DIR] [--conv ID]
    chat-journal.py verify  [--store DIR] [--conv ID]
    chat-journal.py compare --db /path/webui.db [--store DIR] [--conv ID]
                            [--chat-id ID] [--ids]

WHAT THE JOURNAL IS (compactor/chatjournal.py): one JSON line per completed
reply, `<STORE>/journal/<conv_id>.jsonl`, written and fsynced after every
message. OpenWebUI keeps a whole conversation in ONE JSON row it rewrites on
every save; the journal is the independent copy.

THIS TOOL NEVER WRITES. It opens the journal read-only and the OpenWebUI
database with `mode=ro`. There is no write-back into OpenWebUI in this
release; the rebuild design is at the foot of compactor/chatjournal.py.

NO CONTENT IS EVER PRINTED. Output is counts, timestamps and (with --ids, and
only for `compare`) OpenWebUI message ids. Never message text, never a
needle, never an image reference.

  stats    per conversation: lines, redacted tombstones, partial replies,
           first/last ts, file size.
  verify   every line must parse, carry the schema fields, and (unless it is
           a /forget tombstone) have a sha256 matching its own user/assistant
           text. Exit 0 all good, 1 otherwise. Prints line NUMBERS of bad
           lines only.
  compare  for one chat (default: her conversation), which journal exchanges
           are missing from OpenWebUI's chat JSON (`chat.chat` history) and
           from the `chat_message` table.

           MATCHING IS BY CONTENT HASH IN PRACTICE. An exchange matches by
           message id (its own id, or its parent id) when the journal has one,
           else by the exact pair hash, then a whitespace-normalised pair hash,
           taken over every user->assistant pair OpenWebUI holds. But the
           deployed OpenWebUI (0.11.4) forwards no message ids (it strips
           metadata, and ENABLE_FORWARD_USER_INFO_HEADERS is off on purpose),
           so the journal's ids are ALWAYS null there and the hash is the only
           key.

           Three kinds of exchange cannot be judged and are NOT reported as
           "missing":
             - REDACTED tombstones (`/forget <text>` emptied them; nothing is
               left to hash and there is no id): counted on their own line;
             - image-only turns are recognised: the journal holds the
               `[shared N image]` marker the compactor's memory uses, while
               OpenWebUI holds an empty user text, so the marker is compared
               as the empty text;
             - RAG / file-attachment turns can still show as unmatched: the
               text the compactor saw (retrieved context, extracted file
               content) can differ from the user message OpenWebUI stored. A
               small unmatched count is therefore expected; only a large or
               growing one means missing chat.
           Counts only; --ids lists the ids of the unmatched ones (journal
           message ids where the journal has them, which on 0.11.4 is never,
           else the first 12 hex of the pair hash).

Stdlib only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _podpaths  # noqa: E402  (v3.1.9.10: this pod's paths, not another pod's)

HER_CONV = "ea1494ea-e9d7-46fb-8b7c-3a50d685d00e"
REQUIRED = ("v", "conv_id", "ts", "epoch", "user", "assistant", "finish_reason", "partial", "sha256")


def pair_sha(user: str, assistant: str) -> str:
    """Identical to compactor/chatjournal.pair_sha (a test pins the two)."""
    blob = json.dumps([user, assistant], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def norm_sha(user: str, assistant: str) -> str:
    return pair_sha(" ".join((user or "").split()), " ".join((assistant or "").split()))


def default_store() -> Path:
    # This pod's store (DATA_DIR / COMPACTOR_STORAGE_ROOT, adopted from PID 1 in a bare
    # shell), never a hardcoded /data path: two pods can share the volume (v3.1.9.10).
    return Path(_podpaths.store())


def journal_files(store: Path, conv: str | None) -> list[Path]:
    d = store / "journal"
    if conv:
        p = d / f"{conv}.jsonl"
        return [p] if p.is_file() else []
    return sorted(d.glob("*.jsonl")) if d.is_dir() else []


def read_lines(path: Path):
    """Yield (lineno, record|None). None = unparseable."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for n, raw in enumerate(f, 1):
            raw = raw.rstrip("\n")
            if not raw:
                continue
            try:
                rec = json.loads(raw)
                yield n, rec if isinstance(rec, dict) else None
            except Exception:
                yield n, None


# ---------------------------------------------------------------------------

def cmd_stats(a) -> int:
    files = journal_files(Path(a.store), a.conv)
    if not files:
        print("no journal files")
        return 0
    tot = 0
    for p in files:
        lines = red = part = bad = 0
        first = last = None
        for _, rec in read_lines(p):
            if rec is None:
                bad += 1
                continue
            lines += 1
            red += 1 if rec.get("redacted") else 0
            part += 1 if rec.get("partial") else 0
            ts = rec.get("ts")
            if isinstance(ts, str):
                first = ts if first is None or ts < first else first
                last = ts if last is None or ts > last else last
        tot += lines
        print(f"{p.stem}: lines={lines} redacted={red} partial={part} unparseable={bad} "
              f"first={first} last={last} bytes={p.stat().st_size}")
    print(f"total: {len(files)} file(s), {tot} line(s)")
    return 0


def cmd_verify(a) -> int:
    files = journal_files(Path(a.store), a.conv)
    if not files:
        print("no journal files")
        return 0
    bad_total = 0
    for p in files:
        ok = bad = 0
        badlines: list[int] = []
        for n, rec in read_lines(p):
            good = rec is not None and all(k in rec for k in REQUIRED)
            if good and not rec.get("redacted"):
                good = (
                    isinstance(rec["user"], str) and isinstance(rec["assistant"], str)
                    and rec["sha256"] == pair_sha(rec["user"], rec["assistant"])
                )
            elif good:
                good = rec.get("sha256") is None
            if good:
                ok += 1
            else:
                bad += 1
                badlines.append(n)
        bad_total += bad
        extra = f" bad_lines={badlines[:20]}{'...' if len(badlines) > 20 else ''}" if bad else ""
        print(f"{p.stem}: ok={ok} bad={bad}{extra}")
    print("VERIFY OK" if not bad_total else f"VERIFY FAILED: {bad_total} bad line(s)")
    return 0 if not bad_total else 1


# ---------------------------------------------------------------------------
# compare

def _text(content) -> str:
    """User/assistant text out of an OpenWebUI content value (string, or a list
    of blocks with text parts). Same joining as compactor._message_text."""
    if content is None:
        return ""
    if isinstance(content, list):
        return " ".join(c.get("text", "") for c in content if isinstance(c, dict))
    return str(content)


def _decode_table_content(raw):
    """chat_message.content is JSON-encoded TEXT when read with raw sqlite3."""
    if raw is None:
        return ""
    if isinstance(raw, str):
        try:
            return _text(json.loads(raw))
        except Exception:
            return raw
    return _text(raw)


def load_openwebui(db: str, chat_id: str):
    """(json_msgs, table_msgs), each {message_id: {parent, role, text}}."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=30)
    try:
        json_msgs: dict[str, dict] = {}
        row = con.execute("select chat from chat where id=?", (chat_id,)).fetchone()
        if row and row[0]:
            try:
                hist = (json.loads(row[0]).get("history") or {}).get("messages") or {}
            except Exception:
                hist = {}
            for mid, m in hist.items():
                if isinstance(m, dict):
                    json_msgs[mid] = {"parent": m.get("parentId"), "role": m.get("role"),
                                      "text": _text(m.get("content"))}
        table_msgs: dict[str, dict] = {}
        try:
            prefix = chat_id + "-"
            for mid, parent, role, content in con.execute(
                "select id, parent_id, role, content from chat_message where chat_id=?", (chat_id,)
            ):
                key = mid[len(prefix):] if mid.startswith(prefix) else mid
                pk = parent[len(prefix):] if isinstance(parent, str) and parent.startswith(prefix) else parent
                table_msgs[key] = {"parent": pk, "role": role, "text": _decode_table_content(content)}
        except sqlite3.OperationalError:
            pass  # no chat_message table in this database
        return json_msgs, table_msgs
    finally:
        con.close()


def pair_index(msgs: dict[str, dict]) -> tuple[set[str], set[str]]:
    """Exact and whitespace-normalised pair hashes of every user->assistant pair."""
    exact: set[str] = set()
    norm: set[str] = set()
    for m in msgs.values():
        if m.get("role") != "assistant":
            continue
        u = msgs.get(m.get("parent") or "")
        if not u or u.get("role") != "user":
            continue
        exact.add(pair_sha(u["text"], m["text"]))
        norm.add(norm_sha(u["text"], m["text"]))
    return exact, norm


# The marker main._memorable_user_text gives an image-only user turn (twin of
# summarizer._image_only_marker). OpenWebUI stores such a turn with an EMPTY
# text, so the journal's marker is compared as "".
_IMAGE_ONLY_RE = re.compile(r"^\[shared \d+ images?\]$")


def _present(rec: dict, msgs: dict, exact: set, norm: set) -> bool:
    for k in ("message_id", "parent_id"):
        v = rec.get(k)
        if v and v in msgs:
            return True
    if rec.get("redacted"):
        return False
    u, a = rec.get("user") or "", rec.get("assistant") or ""
    if _IMAGE_ONLY_RE.match(u.strip()):
        u = ""
    if pair_sha(u, a) in exact or norm_sha(u, a) in norm:
        return True
    if rec.get("truncated"):
        # A field was clipped at write time, so the full-pair hash cannot
        # match; compare on the leading 512 whitespace-normalised chars.
        ku = " ".join(u.split())[:512]
        ka = " ".join(a.split())[:512]
        for m in msgs.values():
            if m.get("role") == "assistant" and " ".join(m["text"].split())[:512] == ka:
                p = msgs.get(m.get("parent") or "")
                if p and " ".join(p["text"].split())[:512] == ku:
                    return True
    return False


def cmd_compare(a) -> int:
    conv = a.conv or HER_CONV
    chat_id = a.chat_id or conv
    p = Path(a.store) / "journal" / f"{conv}.jsonl"
    if not p.is_file():
        print(f"no journal for that conversation")
        return 2
    if not Path(a.db).is_file():
        print("database not found")
        return 2
    recs = [r for _, r in read_lines(p) if r is not None]
    jm, tm = load_openwebui(a.db, chat_id)
    jx, jn = pair_index(jm)
    tx, tn = pair_index(tm)
    miss_json, miss_table, miss_both = [], [], []
    redacted = 0
    for r in recs:
        if r.get("redacted"):
            # A /forget tombstone: no words and no hash left, and no id on
            # 0.11.4. It cannot be judged either way, so it is not "missing".
            redacted += 1
            continue
        in_j = _present(r, jm, jx, jn)
        in_t = _present(r, tm, tx, tn)
        label = r.get("message_id") or r.get("parent_id") or (
            (r.get("sha256") or "")[:12] or "?")
        if not in_j:
            miss_json.append(label)
        if not in_t:
            miss_table.append(label)
        if not in_j and not in_t:
            miss_both.append(label)
    print(f"journal exchanges: {len(recs)} "
          f"({redacted} redacted, "
          f"{sum(1 for r in recs if r.get('partial'))} partial)")
    print(f"openwebui chat JSON messages: {len(jm)} | chat_message rows: {len(tm)}")
    print(f"redacted (/forget tombstones, not compared): {redacted}")
    print(f"compared: {len(recs) - redacted}")
    print(f"missing from chat JSON: {len(miss_json)}")
    print(f"missing from chat_message table: {len(miss_table)}")
    print(f"missing from both: {len(miss_both)}")
    if miss_both or miss_json or miss_table:
        print("note: matching is by content hash (0.11.4 sends no message ids); "
              "RAG / file-attachment turns can show as unmatched")
    if a.ids:
        for name, lst in (("chat JSON", miss_json), ("chat_message", miss_table), ("both", miss_both)):
            print(f"ids missing from {name}: {' '.join(lst)}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("stats", "verify", "compare"):
        sp = sub.add_parser(name)
        sp.add_argument("--store", default=str(default_store()),
                        help="compactor STORAGE_ROOT (default: this pod's store, from $COMPACTOR_STORAGE_ROOT or $DATA_DIR)")
        sp.add_argument("--conv", default=None, help="conversation id (compare defaults to hers)")
        if name == "compare":
            sp.add_argument("--db", required=True, help="OpenWebUI webui.db (opened read-only; use a copy)")
            sp.add_argument("--chat-id", default=None, help="OpenWebUI chat id if it differs from the conv id")
            sp.add_argument("--ids", action="store_true", help="list the ids of the missing exchanges")
    a = ap.parse_args(argv)
    return {"stats": cmd_stats, "verify": cmd_verify, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
