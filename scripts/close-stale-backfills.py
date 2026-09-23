#!/usr/bin/env python3
"""Close backfill records that v3.1.9.4 would re-run over existing facts.

v3.1.9.4 decides whether to backfill by the backfill record before it checks
for a facts file, so any conversation with an interrupted ("in_progress") or
"failed" record re-extracts its whole history on its next message, even when
it already has facts, and that re-run can evict established facts. Until the
code is fixed, close those records before deploying v3.1.9.4.

Only records whose conversation ALREADY HAS A FACTS FILE are touched; a
conversation with no facts still needs its backfill and is left alone.
Closing writes state "complete" and keeps every original field under
"closed_by_operator". Nothing is deleted. Run with the compactor stopped.

Usage:
  close-stale-backfills.py <compactor storage root> [--apply]
Output is conversation ids, states and counts only.
"""
import json, os, sys, tempfile, time


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    root, apply = sys.argv[1], "--apply" in sys.argv
    fdir = os.path.join(root, "facts")
    rows = []
    for name in sorted(os.listdir(fdir)):
        if not name.endswith(".backfill.json"):
            continue
        cid = name[: -len(".backfill.json")]
        path = os.path.join(fdir, name)
        try:
            st = json.load(open(path, encoding="utf-8"))
        except Exception as e:
            rows.append((cid, "UNREADABLE", None, None, path, str(e)))
            continue
        facts_path = os.path.join(fdir, f"{cid}.json")
        n_facts = None
        if os.path.isfile(facts_path):
            try:
                v = json.load(open(facts_path, encoding="utf-8"))
                n_facts = len(v.get("facts", [])) if isinstance(v, dict) else len(v)
            except Exception:
                n_facts = -1
        rows.append((cid, st.get("state"), st.get("attempts"), n_facts, path, st))
    to_close = [r for r in rows if r[1] in ("in_progress", "failed") and r[3] is not None]
    print(f"backfill records: {len(rows)}")
    for cid, state, att, n, _, st in rows:
        mark = "CLOSE" if any(cid == r[0] for r in to_close) else "     "
        done = st.get("exchanges_done") if isinstance(st, dict) else None
        total = st.get("exchanges_total") if isinstance(st, dict) else None
        print(f"  {mark} {cid[:36]:36s} state={state!s:11s} attempts={att!s:4s} "
              f"progress={done}/{total} facts_on_disk={'none' if n is None else n}")
    print(f"to close: {len(to_close)}")
    if not apply:
        print("DRY RUN — nothing written")
        return
    for cid, state, att, n, path, st in to_close:
        new = {"state": "complete", "closed_by_operator": {"at": int(time.time()), "previous": st,
               "reason": "v3.1.9.4 would re-run this backfill over existing facts"}}
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".close-", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(new, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        back = json.load(open(path, encoding="utf-8"))
        print(f"  closed {cid[:36]}: now {back['state']}")


if __name__ == "__main__":
    main()
