#!/usr/bin/env python3
"""The two numbers F2 asked for and never got.

F2's "Still open" says it plainly: *"A little too repetitive" is not a
number, so nothing here can be confirmed to have helped.* The drift detector
is the proof that the degeneracy approach works; this is the equivalent for
the failure the detector CANNOT see — not a repetition loop inside one reply,
but sameness ACROSS replies.

    python scripts/chat-metrics.py <path to webui.db>

Two metrics, both from F2's own description:

  * frequency of opening n-grams over the corpus — where "every reply starts
    the same way" shows up first and most sharply;
  * n-gram overlap between CONSECUTIVE replies.

Measured 2026-09-04 on a 976-reply corpus: 13.5% of replies opened with the
same two words, the top five openers were 12.7% of everything, and 10.2% of
consecutive pairs shared over 20% of their 5-grams. Re-run it after any
prompt or sampling change; the point of a number is that it moves.

Prints counts and short opening n-grams only. It reads a private
conversation store: keep it that way.
"""
import collections
import json
import re
import sqlite3
import sys

if len(sys.argv) != 2:
    sys.exit("usage: python scripts/chat-metrics.py <path to webui.db>")
DB = sys.argv[1]
c = sqlite3.connect(DB)


def messages_of(blob):
    try:
        d = json.loads(blob)
    except Exception:
        return []
    hist = d.get("history")
    if isinstance(hist, dict) and isinstance(hist.get("messages"), dict):
        return list(hist["messages"].values())
    m = d.get("messages")
    return m if isinstance(m, list) else []


rows = list(c.execute("select id,updated_at,chat from chat"))
print(f"{len(rows)} chats\n")

totals = collections.Counter()
per_chat = []
assistant_texts = []          # (chat_id, text)
for cid, ua, blob in rows:
    msgs = messages_of(blob)
    roles = collections.Counter(
        m.get("role") for m in msgs if isinstance(m, dict)
    )
    per_chat.append((cid[:4], len(msgs), roles.get("user", 0), roles.get("assistant", 0)))
    totals.update(roles)
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "assistant":
            t = m.get("content")
            if isinstance(t, str) and t.strip():
                assistant_texts.append((cid[:4], t))

per_chat.sort(key=lambda r: -r[1])
print("  msgs   user  asst  conv")
for cid, n, u, a in per_chat[:8]:
    print(f"  {n:5}  {u:5} {a:5}  {cid}")
print(f"\ntotal roles: {dict(totals)}")
print(f"assistant replies with text: {len(assistant_texts)}")

if not assistant_texts:
    sys.exit(0)

lens = sorted(len(t) for _, t in assistant_texts)
def pct(p):
    return lens[min(len(lens) - 1, int(len(lens) * p))]
print(f"\nreply length chars: p50={pct(.5)} p90={pct(.9)} p99={pct(.99)} max={lens[-1]}")

# --- F2's metric, part 1: opening-phrase frequency -------------------------
def opener(t, words=4):
    t = re.sub(r"^[\s*_#>\-]+", "", t)
    return " ".join(re.findall(r"[A-Za-z']+", t)[:words]).lower()


openers = collections.Counter(opener(t) for _, t in assistant_texts if t.strip())
print(f"\ntop opening 4-grams ({len(assistant_texts)} replies, "
      f"{len(openers)} distinct):")
for phrase, n in openers.most_common(12):
    if phrase:
        print(f"  {n:5}  ({100*n/len(assistant_texts):4.1f}%)  {phrase!r}")

top_share = sum(n for _, n in openers.most_common(5)) / len(assistant_texts)
print(f"\n  top-5 openers account for {100*top_share:.1f}% of replies")

# --- F2's metric, part 2: n-gram overlap between CONSECUTIVE replies -------
def shingles(t, k=5):
    w = re.findall(r"[a-z']+", t.lower())
    return {tuple(w[i:i + k]) for i in range(max(0, len(w) - k + 1))}


by_chat = collections.defaultdict(list)
for cid, t in assistant_texts:
    by_chat[cid].append(t)

overlaps = []
for cid, texts in by_chat.items():
    for a, b in zip(texts, texts[1:]):
        sa, sb = shingles(a), shingles(b)
        if sa and sb:
            overlaps.append(len(sa & sb) / min(len(sa), len(sb)))
if overlaps:
    overlaps.sort()
    print(f"\nconsecutive-reply 5-gram overlap over {len(overlaps)} pairs:")
    print(f"  p50={overlaps[len(overlaps)//2]:.3f} "
          f"p90={overlaps[int(len(overlaps)*.9)]:.3f} "
          f"max={overlaps[-1]:.3f}")
    print(f"  pairs over 0.20 overlap: "
          f"{sum(1 for o in overlaps if o > .20)} "
          f"({100*sum(1 for o in overlaps if o > .20)/len(overlaps):.1f}%)")

# --- sampling params actually deployed -------------------------------------
print("\nmodel rows (sampling params as SET, not as recommended):")
for mid, name, params in c.execute("select id,name,params from model"):
    try:
        p = json.loads(params) if isinstance(params, str) else (params or {})
    except Exception:
        p = {}
    keep = {k: v for k, v in (p or {}).items() if v not in (None, "", [])}
    print(f"  {name!r}: {keep if keep else '(none set)'}")
