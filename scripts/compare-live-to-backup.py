#!/usr/bin/env python3
"""What is in the backup that is NOT live right now? Read-only, both sides.

    /opt/compactor-venv/bin/python /data/scripts/compare-live-to-backup.py
    /opt/compactor-venv/bin/python /data/scripts/compare-live-to-backup.py \
        --backup /data/backups/zions-backup-20260901-192406.tar.gz

WHY THIS EXISTS. "Roll back the conversation" is a decision that cannot be
un-made: restoring overwrites whatever is live with whatever the archive
held, and if the live copy was actually FINE you have just traded real
recent turns for older ones. This answers the question that should come
first — is anything actually missing, and if so, what and how much?

It NEVER WRITES. Both databases are opened read-only, the archive is
extracted to a temp directory, and nothing under /data/openwebui is touched.
Deciding to restore, and doing it, stay manual.

WHAT IT COMPARES, across the two places her memory lives:

  OpenWebUI (webui.db)   the conversation itself - per chat, how many
                         messages and when the last one was.

  Compactor storage      facts, archived facts, summary hierarchy and
                         persona, per conv_id. These are keyed on conv_id,
                         NOT on the OpenWebUI chat id, so a conversation can
                         look intact in one and be empty in the other - which
                         is exactly what a conv_id fork looks like, and is
                         worth knowing before restoring anything.

READ THE SIGN OF THE DELTA. A negative number means the BACKUP has more than
live, i.e. something was lost and a restore would bring it back. A positive
number means LIVE has more, and restoring would DESTROY that difference.
Both appear in a normal comparison — the backup is older, so live having
more recent messages is expected and healthy.

Stdlib only.
"""

import argparse
import glob
import json
import os
import shutil
import sqlite3
import sys
import tarfile
import tempfile
from pathlib import Path


def say(m=""):
    print(m, flush=True)


def newest_backup(d: str) -> str:
    hits = sorted(glob.glob(os.path.join(d, "zions-backup-*.tar.gz")))
    if not hits:
        raise SystemExit(f"no backups found in {d}")
    return hits[-1]


def chats(db: Path) -> dict:
    """{chat_id: (title_len, n_messages, last_ts)} — never the text itself."""
    if not db.exists():
        return {}
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    out = {}
    for r in con.execute("select id, chat from chat"):
        try:
            d = json.loads(r["chat"])
        except Exception:
            continue
        h = d.get("history") or {}
        msgs = (
            list(h["messages"].values())
            if isinstance(h.get("messages"), dict)
            else (d.get("messages") or [])
        )
        ts = [m.get("timestamp") for m in msgs if m.get("timestamp")]
        out[r["id"]] = (len(msgs), max(ts) if ts else 0)
    con.close()
    return out


def compactor_state(root: Path) -> dict:
    """{conv_id: {facts, archived, l1, l2, l3, persona}} from the JSON on disk."""
    out: dict = {}

    def slot(cid):
        return out.setdefault(
            cid, {"facts": 0, "archived": 0, "l1": 0, "l2": 0, "l3": 0, "persona": 0}
        )

    fdir = root / "facts"
    if fdir.exists():
        for f in fdir.glob("*.json"):
            stem = f.stem
            try:
                d = json.load(open(f, encoding="utf-8"))
            except Exception:
                continue
            n = len(d.get("facts") or [])
            if stem.endswith(".archive"):
                slot(stem[: -len(".archive")])["archived"] = n
            elif "." not in stem:
                slot(stem)["facts"] = n
    sdir = root / "summaries"
    if sdir.exists():
        for f in sdir.glob("*.json"):
            if "." in f.stem:
                continue
            try:
                d = json.load(open(f, encoding="utf-8"))
            except Exception:
                continue
            s = slot(f.stem)
            s["l1"] = len(d.get("l1") or [])
            s["l2"] = len(d.get("l2") or [])
            s["l3"] = 1 if d.get("l3") else 0
    pdir = root / "personas"
    if pdir.exists():
        for f in pdir.glob("*.json"):
            if "." not in f.stem:
                slot(f.stem)["persona"] = 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backup", help="archive path (default: newest in --backup-dir)")
    ap.add_argument("--backup-dir", default="/data/backups")
    ap.add_argument("--live-db", default="/data/openwebui/webui.db")
    ap.add_argument("--live-compactor", default="/data/openwebui/compactor")
    args = ap.parse_args()

    archive = args.backup or newest_backup(args.backup_dir)
    say("=" * 76)
    say("LIVE vs BACKUP — read-only on both sides, nothing is restored")
    say("=" * 76)
    say(f"backup: {archive}")
    say(f"live:   {args.live_db}")
    say("")

    tmp = Path(tempfile.mkdtemp(prefix="zl-compare-"))
    try:
        with tarfile.open(archive) as t:
            wanted = [
                m
                for m in t.getmembers()
                if m.name.endswith("webui.db")
                or "/compactor/facts/" in m.name
                or "/compactor/summaries/" in m.name
                or "/compactor/personas/" in m.name
                or m.name.startswith(("compactor/facts/", "compactor/summaries/",
                                      "compactor/personas/"))
            ]
            t.extractall(tmp, members=wanted)
        b_db = next(iter(sorted(tmp.rglob("webui.db"))), None)
        b_root = next(iter(sorted(p.parent for p in tmp.rglob("compactor/facts"))),
                      None) or tmp / "compactor"

        live_c, back_c = chats(Path(args.live_db)), chats(b_db) if b_db else {}
        say(f"CONVERSATIONS   live {len(live_c)}   backup {len(back_c)}")
        say("")
        say(f"{'chat id':<40}{'live':>7}{'backup':>8}{'delta':>8}")
        say("-" * 76)
        gone, shrunk = [], []
        for cid in sorted(set(live_c) | set(back_c)):
            lv = live_c.get(cid, (0, 0))[0]
            bk = back_c.get(cid, (0, 0))[0]
            if lv == bk:
                continue
            d = lv - bk
            say(f"{cid[:38]:<40}{lv:>7}{bk:>8}{d:>+8}")
            if lv == 0 and bk > 0:
                gone.append((cid, bk))
            elif d < 0:
                shrunk.append((cid, lv, bk))
        if not gone and not shrunk:
            say("  (no conversation is smaller live than in the backup)")
        say("")

        say("COMPACTOR MEMORY (facts / archived / L1 / L2 / L3 / persona)")
        say("-" * 76)
        live_m = compactor_state(Path(args.live_compactor))
        back_m = compactor_state(b_root) if b_root else {}
        lost = []
        for cid in sorted(set(live_m) | set(back_m)):
            L = live_m.get(cid, {})
            B = back_m.get(cid, {})
            keys = ("facts", "archived", "l1", "l2", "l3", "persona")
            if all(L.get(k, 0) == B.get(k, 0) for k in keys):
                continue
            lf = "/".join(str(L.get(k, 0)) for k in keys)
            bf = "/".join(str(B.get(k, 0)) for k in keys)
            flag = ""
            if any(L.get(k, 0) < B.get(k, 0) for k in keys):
                flag = "  <-- backup has more"
                lost.append(cid)
            say(f"  {cid[:22]:<24} live {lf:<18} backup {bf:<18}{flag}")
        if not lost:
            say("  (no conv_id has less memory live than in the backup)")
        say("")

        say("=" * 76)
        if not gone and not shrunk and not lost:
            say("NOTHING IS MISSING. Live holds everything the backup holds, and")
            say("more where the conversation has continued. A restore would only")
            say("DISCARD the difference. Do not roll back on this evidence.")
        else:
            if gone:
                say(f"CONVERSATIONS PRESENT IN BACKUP BUT EMPTY/ABSENT LIVE: {len(gone)}")
                for cid, n in gone[:10]:
                    say(f"    {cid}  ({n} messages in the backup)")
            if shrunk:
                say(f"CONVERSATIONS SMALLER LIVE THAN IN BACKUP: {len(shrunk)}")
                for cid, lv, bk in shrunk[:10]:
                    say(f"    {cid}  live {lv} vs backup {bk}")
            if lost:
                say(f"CONV_IDS WITH LESS COMPACTOR MEMORY LIVE: {len(lost)}")
                for cid in lost[:10]:
                    say(f"    {cid}")
            say("")
            say("Before restoring, check the OTHER direction in the table above:")
            say("any chat where live is LARGER is real conversation a restore")
            say("would throw away. Restoring is not free just because something")
            say("else is missing.")
        say("=" * 76)
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
