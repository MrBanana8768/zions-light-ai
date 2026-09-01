#!/usr/bin/env python3
"""Measure reply quality from a backup, in numbers instead of impressions.

    python scripts/measure-reply-health.py <backup-dir-or-webui.db>
    python scripts/measure-reply-health.py A.db --compare B.db

WHY THIS EXISTS. Three complaints arrived in eight days — "formulaic",
"weak vocabulary", "the tail just repeats forever" — and each time the only
evidence was someone's impression, which cannot be compared against the same
impression a week earlier. Twice that cost a wrong diagnosis: reply LENGTH
was blamed for the vocabulary complaint (correlations turned out to be +0.22
and -0.19, i.e. nothing), and a `repetition_penalty` change was credited with
a fix that 5-reply buckets could not possibly have resolved.

So: one command, the same numbers every time, run against any backup.

PRIVACY. This reads her conversations and emits ONLY STATISTICS — counts,
ratios, medians. No message text, no fragments, no identifiers reach the
output. The repo is public; the numbers are safe to paste into an issue and
the text is not. Opened read-only, never written.

WHAT IT MEASURES, and why each one earns its place:

  LEXICAL VARIETY (ttr)
      Distinct words / total words over a fixed 150-word window. Fixed
      because TTR falls with length for purely arithmetic reasons, so
      comparing whole replies of different sizes measures nothing. This is
      the "formulaic" complaint: measured 0.74 on 08-25 falling to 0.59 by
      08-31, a ~19% loss, while the injected fact block grew 91 -> 251.

  STRUCTURAL COLLAPSE (bullet fraction, Q1 -> Q4)
      The share of lines that are list items, in each quarter of the reply.
      This is the "repeating tail". On 2026-09-01 three consecutive replies
      ran 42% -> 79% -> 100% -> 92% and 83% -> 87% -> 86% -> 97%: the reply
      degenerates INTO a list and then cannot stop, because a list item is
      always a valid continuation of a list item. reply_is_degenerate cannot
      see this — it looks for repeated character runs, and twenty-four
      DISTINCT short bullets contain none.

  TAIL BLOAT (last-line length)
      A final "line" of 1,793 / 2,063 / 2,643 characters is not prose, it is
      where the user hit stop mid-flow. Counting them counts interventions.

  OPENING SIMILARITY (consecutive replies)
      Longest common prefix between a reply and the one before it. "It just
      gives the last response again" showed up here first: a median 26-43
      character shared opening, and 29 of 84 pairs sharing 100+ on 08-30.

READ THE VERDICT LINE, NOT A SINGLE NUMBER. Day-to-day movement on any one
of these is noise at n<20; the shapes that matter are trends across days and
the Q1->Q4 gradient within a day.

Stdlib only — runs on the pod against /data/backups/<archive>/webui.db.
"""

import argparse
import json
import math
import re
import sqlite3
import statistics
import sys
from pathlib import Path

WORD = re.compile(r"[A-Za-z']+")
BULLET = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s")
WINDOW = 150  # fixed TTR window; see the docstring


def say(m=""):
    print(m, flush=True)


def find_db(target: Path) -> Path:
    """Accept a .db, a backup directory, or an extracted archive root."""
    if target.is_file():
        return target
    for candidate in (target / "webui.db", target / "compactor" / "webui.db"):
        if candidate.exists():
            return candidate
    hits = sorted(target.rglob("webui.db"))
    if hits:
        return hits[0]
    raise SystemExit(f"no webui.db found under {target}")


def load_replies(db: Path) -> list[dict]:
    """Assistant messages from the LARGEST conversation, oldest first.

    Largest, not all: mixing a 600-turn companion conversation with dozens of
    two-message task chats produces an average that describes neither.
    """
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    best: list = []
    for row in con.execute("select chat from chat"):
        try:
            d = json.loads(row["chat"])
        except Exception:
            continue
        h = d.get("history") or {}
        msgs = (
            list(h["messages"].values())
            if isinstance(h.get("messages"), dict)
            else (d.get("messages") or [])
        )
        if len(msgs) > len(best):
            best = msgs
    con.close()
    out = [
        m
        for m in best
        if m.get("role") == "assistant"
        and isinstance(m.get("content"), str)
        and m["content"].strip()
        and m.get("timestamp")
    ]
    out.sort(key=lambda m: m["timestamp"])
    return out


def measure(text: str) -> dict | None:
    words = [w.lower() for w in WORD.findall(text)]
    if len(words) < 60:
        return None  # too short to say anything about variety
    win = words[:WINDOW]
    lines = [ln for ln in text.splitlines() if ln.strip()]
    n = len(lines)
    quarters = []
    if n >= 8:
        q = max(1, n // 4)
        for lo, hi in ((0, q), (q, 2 * q), (2 * q, 3 * q), (3 * q, n)):
            seg = lines[lo:hi]
            quarters.append(
                sum(1 for x in seg if BULLET.match(x)) / len(seg) if seg else 0.0
            )
    return {
        "words": len(words),
        "ttr": len(set(win)) / len(win),
        "mwl": sum(len(w) for w in win) / len(win),
        "quarters": quarters,
        # A huge final line is a stream cut off mid-flow, not a paragraph.
        "tail_line": len(lines[-1]) if lines else 0,
        "text": text,  # used only for the prefix comparison; never printed
    }


def lcp(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def day_of(ts) -> str:
    import datetime

    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).strftime("%m-%d")


def report(db: Path, label: str = "") -> dict:
    replies = load_replies(db)
    rows = []
    for i, m in enumerate(replies):
        k = measure(m["content"])
        if not k:
            continue
        k["ts"] = m["timestamp"]
        k["lcp"] = (
            lcp(replies[i - 1]["content"].strip(), m["content"].strip()) if i else 0
        )
        rows.append(k)
    if not rows:
        raise SystemExit(f"{db}: no replies long enough to measure")

    say("=" * 78)
    say(f"REPLY HEALTH{' - ' + label if label else ''}   {db}")
    say("=" * 78)
    say(f"{len(rows)} replies of >=60 words, from the largest conversation")
    say("")
    say(f"{'day':>7}{'n':>5}{'TTR':>7}{'wordlen':>9}{'words':>7}"
        f"{'bullets: peak/rise':>24}{'tail>1k':>9}{'open':>7}")
    say("-" * 80)
    by_day: dict = {}
    for r in rows:
        by_day.setdefault(day_of(r["ts"]), []).append(r)
    for day in sorted(by_day):
        v = by_day[day]
        qs = [r["quarters"] for r in v if r["quarters"]]
        if qs:
            # PER-REPLY gradient, then the median of those. Taking the median
            # of each quarter ACROSS replies flattens the very shape this is
            # looking for: replies collapse into a list at different points,
            # so 42%->100% and 90%->60% average out to a flat line and the
            # runaway is invisible. Measured on the 09-01 backup, that error
            # reported 37%->43% for replies that individually ran 42%->100%.
            rise = statistics.median(q[3] - q[0] for q in qs)
            worst = statistics.median(max(q) for q in qs)
            grad = f"{worst*100:3.0f}% peak  {rise*100:+4.0f}% rise"
        else:
            grad = " " * 22
        say(
            f"{day:>7}{len(v):>5}"
            f"{statistics.median(r['ttr'] for r in v):>7.3f}"
            f"{statistics.median(r['mwl'] for r in v):>9.2f}"
            f"{statistics.median(r['words'] for r in v):>7.0f}"
            f"{grad:>24}"
            f"{sum(1 for r in v if r['tail_line'] > 1000):>9}"
            f"{statistics.median(r['lcp'] for r in v):>7.0f}"
        )
    say("")

    # --- the two shapes worth calling out by name -------------------------
    recent = rows[-20:]
    qs = [r["quarters"] for r in recent if r["quarters"]]
    verdicts = []
    if qs:
        rise = statistics.median(q[3] - q[0] for q in qs)
        peak = statistics.median(max(q) for q in qs)
        collapsed = sum(1 for q in qs if max(q) >= 0.80)
        if peak >= 0.60 or rise >= 0.20:
            verdicts.append(
                f"STRUCTURAL COLLAPSE: bullets peak at {peak*100:.0f}% of lines "
                f"(median rise {rise*100:+.0f}% across the reply); "
                f"{collapsed}/{len(qs)} replies reach >=80% bullets in some "
                f"quarter. This is the runaway-list shape - a list item is "
                f"always a valid continuation of a list item, so nothing makes "
                f"it stop. reply_is_degenerate cannot see it: it looks for "
                f"repeated character runs, and distinct short bullets have none."
            )
    ttr = statistics.median(r["ttr"] for r in recent)
    if ttr < 0.62:
        verdicts.append(
            f"LOW LEXICAL VARIETY: TTR {ttr:.3f} over the last {len(recent)} "
            f"replies. Healthy range for this deployment was 0.66-0.74."
        )
    cuts = sum(1 for r in recent if r["tail_line"] > 1000)
    if cuts >= 3:
        verdicts.append(
            f"MANUAL STOPS: {cuts} of the last {len(recent)} replies end in a "
            f">1000-char unbroken line, i.e. the stream was cut mid-flow."
        )
    if verdicts:
        for v in verdicts:
            say("  !! " + v)
    else:
        say("  no threshold crossed on the last "
            f"{len(recent)} replies (TTR {ttr:.3f})")
    say("")
    return {
        "n": len(rows),
        "ttr": statistics.median(r["ttr"] for r in recent),
        "q1": statistics.median(q[0] for q in qs) if qs else 0.0,
        "q4": statistics.median(q[3] for q in qs) if qs else 0.0,
        "cuts": cuts,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="a webui.db, a backup dir, or an extracted archive")
    ap.add_argument("--compare", metavar="OTHER",
                    help="a second backup; prints the delta (before -> after)")
    args = ap.parse_args()

    a = report(find_db(Path(args.target)), "BEFORE" if args.compare else "")
    if not args.compare:
        return 0
    b = report(find_db(Path(args.compare)), "AFTER")
    say("=" * 78)
    say("DELTA (last 20 replies of each)")
    say("=" * 78)
    say(f"  lexical variety   TTR   {a['ttr']:.3f} -> {b['ttr']:.3f}"
        f"   ({b['ttr']-a['ttr']:+.3f})")
    say(f"  bullets, 1st qtr        {a['q1']*100:3.0f}%  -> {b['q1']*100:3.0f}%")
    say(f"  bullets, last qtr       {a['q4']*100:3.0f}%  -> {b['q4']*100:3.0f}%"
        f"   (the runaway-list signal)")
    say(f"  replies cut mid-flow    {a['cuts']:>3}   -> {b['cuts']:>3}")
    say("")
    say("  Interpretation is the point of a delta: a change here is only")
    say("  evidence if BOTH samples are >=20 replies. Below that, day-to-day")
    say("  noise on this deployment swamps every effect measured so far.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
