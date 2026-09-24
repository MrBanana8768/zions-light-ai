#!/usr/bin/env python3
"""Re-derive the STRUCTURAL-COLLAPSE thresholds in reply_is_degenerate from a backup.

    python scripts/calibrate-structural-degeneracy.py <backup-dir-or-webui.db> [another ...]
    python scripts/calibrate-structural-degeneracy.py A.db --cut-line 1000

WHY THIS EXISTS. The thresholds in compactor/main.py (DEGENERATE_LINE_CHARS,
DEGENERATE_LINE_SENTENCE_CHARS, DEGENERATE_LIST_RUN, DEGENERATE_LIST_ITEM_CHARS)
were measured on 2026-09-01 against 349 real replies, not chosen. When the
model, the sampling, or her assistant's style changes, they must be measured
again, and the person doing that should not have to rediscover which
discriminators were tried and rejected. This prints all of them.

TWO POPULATIONS. Nothing in the backup says "the user hit stop", so the proxy
from measure-reply-health.py is used: a reply whose final non-empty line is
longer than --cut-line characters was stopped mid-flow (the model had stopped
emitting newlines). Everything else "completed". The proxy is imperfect in
both directions - a runaway that ran to its own end counts as completed, and a
reply that merely ends in a long paragraph counts as cut - so read TP/FP as
"agreement with the proxy", and read the per-day table alongside it: replies
from before the complaint arrived are the best available picture of what her
assistant writes when it is well.

PRIVACY. Emits ONLY STATISTICS - counts, ratios, percentiles. No message text,
no fragments, no identifiers reach the output. The repo is public; these
numbers are safe to paste into an issue and the text is not. Opened read-only.

Stdlib only, so it runs on the pod against /data/backups/<archive>/webui.db.
When compactor/main.py is importable it also cross-checks the local copy of
the rule against the real one, so the copy cannot silently drift.
"""

import argparse
import collections
import datetime
import importlib.util
import os
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "measure_reply_health", HERE / "measure-reply-health.py"
)
_mrh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mrh)
load_replies, find_db, BULLET = _mrh.load_replies, _mrh.find_db, _mrh.BULLET

# Defaults mirror compactor/main.py. Override on the command line to sweep.
LINE_CHARS = 1500
LINE_SENTENCE_CHARS = 40
LINE_MIN_SPACES = 100
LIST_RUN = 50
LIST_ITEM_CHARS = 30


def say(m=""):
    print(m, flush=True)


# --------------------------------------------------------------------------
# The rule. A stdlib COPY of the structural-collapse block in
# reply_is_degenerate (compactor/main.py), kept here so the threshold SWEEP
# below (which varies line_chars/sentence_chars/list_run/item_chars — values
# reply_is_degenerate does not take as parameters, only as module constants)
# has something to call. The cross-check at the bottom of the report proves
# this copy and the original agree, AT THE SHIPPED DEFAULTS, on every reply
# in the backup — that comparison is only meaningful if this stays a real,
# independent reimplementation and not a thin wrapper around the import, so
# resist the temptation to just call main.reply_is_degenerate here.
#
# v3.1.9 (hostile pass 3, F5) fixed FIVE drifts this finding named, so the
# CROSS-CHECK's own report of mismatches means something again (before that,
# it was comparing two rules that could legitimately disagree even when
# neither had a bug). v3.1.9 (hostile pass 5, C5-6) rewrote the
# trailing-content exemption itself (both the shipped rule and this copy,
# together, in the same change) — see compactor/main.py's block comment
# above reply_is_degenerate's exemption for the full reasoning; ported
# here unchanged so the sweep below still means what it says:
#
#   1. ANY-LINE, not last-line-or-substantial-exempt. reply_is_degenerate
#      only judges the reply's own last non-blank line UNCONDITIONALLY, and
#      a non-last candidate line only when what follows it is short or is
#      itself fragment-shaped (R25 / F5's own fix — see main.py's block
#      comment above DEGENERATE_LINE_CHARS for the full reasoning). Ported
#      below as the same two-pass shape: find last_nonblank_idx first, then
#      apply the same trailing-content test main.py's loop does.
#   2. A NAIVE ". " COUNT instead of _count_real_period_breaks, which
#      excludes "Dr. ", "Mrs. ", "9 a.m. " etc. — imported from main.py
#      directly when available (try_import_main), because unlike the four
#      threshold constants below, this one takes no sweep parameter, so
#      there is no reason to keep a second, lesser copy of it. Falls back to
#      the naive count only when main.py cannot be imported at all (the
#      genuinely stdlib-only path this script's docstring promises), with
#      the resulting inaccuracy stated rather than hidden.
#   3. best_run (the LONGEST run seen ANYWHERE) instead of the end-anchored
#      `run` R9 fixed main.py to use — a run broken by later prose no longer
#      counts. Fixed below by simply not tracking `best_run` at all.
#   4. NO DEGENERATE_MIN_CHARS FLOOR on the list-run branch (R19) — added.
#   5. AN EXTRA `breaks >= 2` CONJUNCT with no counterpart in main.py, and
#      one that cannot fire anyway once `n >= line_chars` (>=1500 by
#      default) and `sentence_chars` (40 by default) are both in play:
#      `n / (breaks + 1) <= sentence_chars` already forces
#      `breaks >= n / sentence_chars - 1`, which is >= 36 at the shipped
#      defaults — `breaks >= 2` cannot be the binding constraint at any
#      threshold combination this script's own sweep explores. Removed.
# --------------------------------------------------------------------------
def structural_collapse(
    text: str,
    line_chars: int = LINE_CHARS,
    sentence_chars: int = LINE_SENTENCE_CHARS,
    list_run: int = LIST_RUN,
    item_chars: int = LIST_ITEM_CHARS,
    min_chars: int = 300,
    period_break_counter=None,
    sentence_end_checker=None,
) -> str | None:
    def _breaks(line: str) -> int:
        if period_break_counter is not None:
            real = period_break_counter(line)
        else:
            # Fallback ONLY when main.py could not be imported at all (drift
            # #2's residual case) — abbreviation-blind, same as before.
            real = line.count(". ")
        b = real + line.count("! ") + line.count("? ") + line.count("… ")
        if b == 0:
            b = line.count(", ")
        return b

    def _is_fragment(line: str, *, floor: int, min_spaces: int = LINE_MIN_SPACES) -> bool:
        n = len(line)
        if n < floor or line.count(" ") < min_spaces:
            return False
        return n / (_breaks(line) + 1) <= sentence_chars

    def _prose_dense(line: str, ratio: int = 8) -> bool:
        return line.count(" ") * ratio >= len(line)

    def _ends_in_real_sentence(joined: str) -> bool:
        if sentence_end_checker is not None:
            return sentence_end_checker(joined)
        # Fallback ONLY when main.py could not be imported at all: no
        # abbreviation/single-initial check, same accuracy trade as the
        # naive ". " count above.
        s = joined.rstrip()
        return bool(s) and s[-1] in ".!?\u2026"

    lines = text.splitlines()
    last_nonblank_idx = -1
    for i, raw in enumerate(lines):
        if raw.strip():
            last_nonblank_idx = i

    run = 0  # drift #3: end-anchored, not a running max
    in_fence = False
    for line_idx, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("```"):
            in_fence = not in_fence
            run = 0
            continue
        if in_fence:
            run = 0
            continue
        if len(line) <= item_chars and BULLET.match(line):
            run += 1
        else:
            run = 0
        if _is_fragment(line, floor=line_chars):
            exempt = False
            if line_idx != last_nonblank_idx:  # drift #1
                # v3.1.9 (hostile pass 5, C5-6): mirrors main.py's rewritten
                # trailing-content exemption exactly (see the block comment
                # above reply_is_degenerate's exemption in compactor/main.py
                # for the full reasoning) — fenced code skipped, SHORT
                # trailing content judged by termination, SUBSTANTIAL
                # trailing content judged by list-majority / multi-line
                # join / single-longest-prose-line, in that order.
                trailing_nonblank: list[str] = []
                tc_in_fence = False
                for t in lines[line_idx + 1:]:
                    s = t.strip()
                    if not s:
                        continue
                    if s.startswith("```"):
                        tc_in_fence = not tc_in_fence
                        continue
                    if tc_in_fence:
                        continue
                    trailing_nonblank.append(s)
                if trailing_nonblank:
                    trailing_chars = sum(len(t) for t in trailing_nonblank)
                    if trailing_chars < min_chars:
                        exempt = _ends_in_real_sentence(
                            " ".join(trailing_nonblank)
                        )
                    else:
                        list_lines = [
                            t for t in trailing_nonblank if BULLET.match(t)
                        ]
                        if (
                            len(list_lines) >= 8
                            and len(list_lines) >= len(trailing_nonblank) / 2
                        ):
                            exempt = False
                        else:
                            prose_candidates = [
                                t for t in trailing_nonblank
                                if t not in list_lines and _prose_dense(t)
                            ]
                            long_lines = [
                                t for t in prose_candidates
                                if len(t) >= sentence_chars
                            ]
                            if len(long_lines) >= 2:
                                joined = " ".join(long_lines)
                                exempt = not _is_fragment(
                                    joined, floor=0,
                                    min_spaces=max(1, len(joined) // 8),
                                )
                            elif prose_candidates:
                                exempt = not _is_fragment(
                                    max(prose_candidates, key=len), floor=0,
                                )
                            else:
                                exempt = True
            if not exempt:
                return "fragment-line"
    if len(text) >= min_chars and run >= list_run:  # drift #4
        return "list-run"
    return None


# --------------------------------------------------------------------------
# Candidate discriminators - the ones proposed for the runaway-list shape,
# each measured so the reader can see why the shipped rule is not one of them.
# --------------------------------------------------------------------------
def features(text: str) -> dict:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    n = len(lines)
    is_item = [bool(BULLET.match(x)) for x in lines]
    item_len = [len(x.strip()) for x, b in zip(lines, is_item) if b]
    n_items = sum(is_item)
    quarters = []
    if n >= 8:
        q = max(1, n // 4)
        for lo, hi in ((0, q), (q, 2 * q), (2 * q, 3 * q), (3 * q, n)):
            seg = is_item[lo:hi]
            quarters.append(sum(seg) / len(seg) if seg else 0.0)
    run = short_run = best_run = best_short = 0
    for x, b in zip(lines, is_item):
        if b:
            run += 1
            short_run = short_run + 1 if len(x.strip()) <= LIST_ITEM_CHARS else 0
        else:
            run = short_run = 0
        best_run = max(best_run, run)
        best_short = max(best_short, short_run)
    longest = max((ln.strip() for ln in lines), key=len, default="")
    breaks = (
        longest.count(". ") + longest.count("! ") + longest.count("? ")
        + longest.count("… ")
    )
    return {
        "chars": len(text),
        "last_line": len(lines[-1].strip()) if lines else 0,
        "n_items": n_items,
        "item_frac": n_items / n if n else 0.0,
        "peak": max(quarters) if quarters else 0.0,
        "rise": (quarters[3] - quarters[0]) if quarters else 0.0,
        "median_item": statistics.median(item_len) if item_len else 0.0,
        "run": best_run,
        "short_run": best_short,
        "max_line": len(longest),
        "max_line_sentence": len(longest) / (breaks + 1),
    }


def pct(values, p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def rate(k: int, n: int) -> str:
    return f"{k:>4}/{n:<4} ({100 * k / n:5.1f}%)" if n else f"{k:>4}/0    (  n/a)"


def try_import_main():
    """The real detector, if this checkout (or the pod) can import it. Also
    returns the real period-break counter (main._count_real_period_breaks)
    and the real short-trailer termination check
    (main._trailing_ends_in_real_sentence) when available — see
    structural_collapse's docstring for why these pieces are imported
    rather than approximated even in the local copy, unlike the
    threshold-swept constants below."""
    os.environ.setdefault("MODEL_REPO", "")
    for cand in (HERE.parent / "compactor", Path("/app"), Path.cwd()):
        if (cand / "main.py").exists() and str(cand) not in sys.path:
            sys.path.insert(0, str(cand))
    try:
        import main  # type: ignore

        return (
            main.reply_is_degenerate,
            main._count_real_period_breaks,
            main._trailing_ends_in_real_sentence,
        )
    except Exception:
        return None, None, None


def report(db: Path, cut_line: int, real_detector, period_break_counter=None, sentence_end_checker=None) -> None:
    replies = load_replies(db)
    rows = []
    for m in replies:
        text = m["content"]
        f = features(text)
        f["day"] = _mrh.day_of(m["timestamp"])
        f["cut"] = f["last_line"] > cut_line
        f["rule"] = structural_collapse(
            text, period_break_counter=period_break_counter,
            sentence_end_checker=sentence_end_checker,
        )
        f["existing"] = None
        if real_detector is not None:
            f["real"] = real_detector(text)
            # A verdict from one of the OLDER rules (repetition, script drift,
            # decoration): those fire first, so the new rule never sees the
            # reply at all. Kept separate so "new flags" means new.
            f["existing"] = f["real"] if (
                f["real"] and not f["real"].startswith(("an unbroken line", "a run of"))
            ) else None
        rows.append(f)
    cut = [r for r in rows if r["cut"]]
    done = [r for r in rows if not r["cut"]]

    say("=" * 78)
    say(f"STRUCTURAL COLLAPSE CALIBRATION   {db}")
    say("=" * 78)
    say(f"{len(rows)} assistant replies from the largest conversation")
    say(f"  cut (final line > {cut_line} chars, i.e. stopped mid-flow): {len(cut)}")
    say(f"  completed:                                         {len(done)}")
    say("")

    if real_detector is not None:
        say("EXISTING RULES (repetition, script drift, decoration) - fire first:")
        say(f"  cut flagged       {rate(sum(1 for r in cut if r['existing']), len(cut))}")
        say(f"  completed flagged {rate(sum(1 for r in done if r['existing']), len(done))}")
        say("")

    say("THE SHIPPED RULE (structural collapse), as it would fire on its own:")
    say(f"  line >= {LINE_CHARS} chars whose sentences average <= {LINE_SENTENCE_CHARS} chars,"
        f" or >= {LIST_RUN} consecutive list items of <= {LIST_ITEM_CHARS} chars")
    tp = [r for r in cut if r["rule"]]
    fp = [r for r in done if r["rule"]]
    say(f"  true positives  (cut flagged)        {rate(len(tp), len(cut))}")
    say(f"  false positives (completed flagged)  {rate(len(fp), len(done))}")
    for branch in ("fragment-line", "list-run"):
        say(f"    by branch {branch:<14} cut {sum(1 for r in tp if r['rule'] == branch):>3}"
            f"   completed {sum(1 for r in fp if r['rule'] == branch):>3}")
    if real_detector is not None:
        new_cut = [r for r in tp if not r["existing"]]
        new_done = [r for r in fp if not r["existing"]]
        say(f"  beyond the existing rules: +{len(new_cut)} cut, +{len(new_done)} completed"
            f"   (union on cut: {rate(sum(1 for r in cut if r['rule'] or r['existing']), len(cut))})")
    say("")

    say("PER DAY  (n / cut / flagged by rule / of which completed / max short-item run /"
        " longest fragment-shaped line)")
    by_day = collections.defaultdict(list)
    for r in rows:
        by_day[r["day"]].append(r)
    for day in sorted(by_day):
        v = by_day[day]
        frag = [r["max_line"] for r in v if r["max_line_sentence"] <= LINE_SENTENCE_CHARS]
        say(f"  {day}  {len(v):>4} {sum(r['cut'] for r in v):>4}"
            f" {sum(1 for r in v if r['rule']):>4}"
            f" {sum(1 for r in v if r['rule'] and not r['cut']):>4}"
            f" {max(r['short_run'] for r in v):>6}"
            f" {max(frag, default=0):>8}")
    say("")

    say("SENSITIVITY of the shipped thresholds (FP = completed flagged, TP = cut flagged):")
    texts_cut = [m["content"] for m, r in zip(replies, rows) if r["cut"]]
    texts_done = [m["content"] for m, r in zip(replies, rows) if not r["cut"]]

    def sweep(label, **kw):
        fp_n = sum(1 for t in texts_done
                   if structural_collapse(t, period_break_counter=period_break_counter,
                                           sentence_end_checker=sentence_end_checker, **kw))
        tp_n = sum(1 for t in texts_cut
                   if structural_collapse(t, period_break_counter=period_break_counter,
                                           sentence_end_checker=sentence_end_checker, **kw))
        say(f"  {label:<34} FP {rate(fp_n, len(texts_done))}   TP {rate(tp_n, len(texts_cut))}")

    for v in (1000, 1200, 1500, 2000, 2500):
        sweep(f"line_chars={v}", line_chars=v)
    for v in (30, 35, 40, 45, 50, 60):
        sweep(f"sentence_chars={v}", sentence_chars=v)
    for v in (15, 20, 25, 30, 40, 50, 100):
        sweep(f"list_run={v}", list_run=v)
    for v in (20, 25, 30, 40):
        sweep(f"item_chars={v}", item_chars=v)
    say("")

    say("CANDIDATE DISCRIMINATORS for the runaway-list shape, each alone")
    say("  (the ones proposed before measuring; none separates the populations):")

    def cand(label, pred):
        fp_n = sum(1 for r in done if pred(r))
        tp_n = sum(1 for r in cut if pred(r))
        say(f"  {label:<34} FP {rate(fp_n, len(done))}   TP {rate(tp_n, len(cut))}")

    for v in (0.5, 0.6, 0.7):
        cand(f"bullet fraction >= {v:.0%}", lambda r, v=v: r["item_frac"] >= v)
    for v in (0.6, 0.8):
        cand(f"worst-quarter fraction >= {v:.0%}", lambda r, v=v: r["peak"] >= v)
    for v in (0.2, 0.3):
        cand(f"rise Q1->Q4 >= {v:.0%}", lambda r, v=v: r["rise"] >= v)
    for v in (50, 100, 200):
        cand(f"bullet count >= {v}", lambda r, v=v: r["n_items"] >= v)
    for v in (20, 25, 30):
        cand(f"median bullet length <= {v}", lambda r, v=v: 0 < r["median_item"] <= v)
    for v in (10, 15, 20, 25, 50):
        cand(f"consecutive bullets <= {LIST_ITEM_CHARS}ch >= {v}",
             lambda r, v=v: r["short_run"] >= v)
    for v in (20, 30, 40, 60):
        cand(f"consecutive bullets (any len) >= {v}", lambda r, v=v: r["run"] >= v)
    say("")

    say("DISTRIBUTIONS (p50 / p90 / p95 / p98 / max)")
    for key in ("n_items", "item_frac", "peak", "rise", "median_item", "short_run",
                "run", "max_line", "max_line_sentence"):
        for name, pop in (("cut", cut), ("completed", done)):
            vals = [r[key] for r in pop]
            say(f"  {key:<18}{name:<10}" + "".join(
                f"{pct(vals, p):>9.2f}" for p in (0.5, 0.9, 0.95, 0.98, 1.0)))
    say("")

    if real_detector is not None:
        # The copy above must agree with the real rule wherever the real rule
        # actually reached the new block (i.e. no older rule fired first).
        bad = 0
        for r in rows:
            if r["existing"]:
                continue
            if bool(r["real"]) != bool(r["rule"]):
                bad += 1
        say(f"CROSS-CHECK against compactor.main.reply_is_degenerate: "
            f"{bad} verdict mismatch(es) over {len(rows)} replies"
            + ("" if bad == 0 else "   <-- the copy in this script has drifted"))
    else:
        say("CROSS-CHECK skipped: compactor/main.py not importable from here")
    say("")


def main() -> int:
    global LINE_CHARS, LINE_SENTENCE_CHARS, LIST_RUN, LIST_ITEM_CHARS
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("targets", nargs="+",
                    help="webui.db files, backup dirs, or extracted archives")
    ap.add_argument("--cut-line", type=int, default=1000,
                    help="final-line length above which a reply counts as cut "
                         "(measure-reply-health.py's proxy; default 1000)")
    ap.add_argument("--line-chars", type=int, default=LINE_CHARS)
    ap.add_argument("--sentence-chars", type=int, default=LINE_SENTENCE_CHARS)
    ap.add_argument("--list-run", type=int, default=LIST_RUN)
    ap.add_argument("--item-chars", type=int, default=LIST_ITEM_CHARS)
    args = ap.parse_args()
    LINE_CHARS, LINE_SENTENCE_CHARS = args.line_chars, args.sentence_chars
    LIST_RUN, LIST_ITEM_CHARS = args.list_run, args.item_chars
    real, period_break_counter, sentence_end_checker = try_import_main()
    # v3.1.9 (hostile pass 3, F5): DEGENERATE_MIN_CHARS drift #4's fix reads
    # the real constant when main.py is importable, the local stdlib default
    # otherwise — same doctrine as period_break_counter above.
    min_chars = 300
    if real is not None:
        try:
            import main  # already on sys.path from try_import_main
            min_chars = main.DEGENERATE_MIN_CHARS
        except Exception:
            pass
    # __defaults__ reassignment must cover EVERY parameter that has a
    # default, in signature order, or it silently misaligns later ones —
    # structural_collapse's signature is
    # (text, line_chars, sentence_chars, list_run, item_chars, min_chars,
    #  period_break_counter), so all six trailing defaults are set here,
    # not just the four CLI-swept ones.
    structural_collapse.__defaults__ = (
        LINE_CHARS, LINE_SENTENCE_CHARS, LIST_RUN, LIST_ITEM_CHARS,
        min_chars, period_break_counter, sentence_end_checker,
    )
    for t in args.targets:
        report(find_db(Path(t)), args.cut_line, real, period_break_counter,
               sentence_end_checker)
    return 0


if __name__ == "__main__":
    sys.exit(main())
