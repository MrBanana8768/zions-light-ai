"""
v3195-main M1 — the two 2026-09-20 degeneracy shapes the pre-existing rules
in main.py cannot see (SP\\p18a-findings.md F1, SP\\V3195_MAIN_BRIEF.md M1),
plus shape B's SECOND pass after the coordinator's real-corpus calibration
showed the whole-reply form of that rule (this file's own first revision)
never separates and ships inert forever.

Shape A, "the symbol wall": many DISTINCT Unicode symbol characters packed
into a short span. Shape B, "the synonym cascade": a WINDOW of almost every
word used exactly once and almost no function words — not the whole reply,
which the calibration showed dilutes an embedded cascade below any usable
threshold. Every fixture below is synthetic, built from Unicode code points
and constructed nonsense syllables — no real conversation, fact or persona
text anywhere in this file.

    python test_v3195_main_m1.py
"""

import os
import random
import subprocess
import sys
import unicodedata

os.environ.setdefault("MODEL_REPO", "")

import main  # noqa: E402

_FAILED = False


def check(cond, label):
    global _FAILED
    status = "ok  " if cond else "FAIL"
    print(f"  {status} {label}")
    if not cond:
        _FAILED = True


PROSE_SENT = ("The quarterly figure moved again and the operator noted the "
              "change in the log before the next review window opened. ")


def prose(n_chars):
    out, tot, i = [], 0, 0
    while tot < n_chars:
        out.append(PROSE_SENT)
        tot += len(PROSE_SENT)
        i += 1
        if i % 4 == 0:
            out.append("\n\n")
            tot += 2
    return "".join(out)[:n_chars]


def symbol_pool(n):
    """n distinct Unicode symbol (So/Sm/Sk/Sc) characters, none of them in
    main._DECOR_CHARS -- the exact "outside the box-drawing block" shape
    the 2026-09-20 wall measured."""
    out = []
    c = 0x2200
    while len(out) < n:
        ch = chr(c)
        c += 1
        if (unicodedata.category(ch) in ("So", "Sm", "Sk", "Sc")
                and ch not in main._DECOR_CHARS):
            out.append(ch)
    return out


def wall(n_distinct, total_chars):
    pool = symbol_pool(n_distinct)
    return "".join(pool[i % len(pool)] for i in range(total_chars))


def make_words(total_words, distinct_words, seed=7):
    rng = random.Random(seed)
    syll = ["ba", "ti", "ko", "lu", "sen", "dra", "mir", "fen", "vor", "cal",
            "ith", "zan", "ru", "pel", "os", "tark", "wen", "gil", "nor", "fay"]
    words = []
    while len(words) < distinct_words:
        w = "".join(rng.choice(syll) for _ in range(rng.randint(1, 3)))
        if w not in words:
            words.append(w)
    filler = [rng.choice(words) for _ in range(total_words - distinct_words)]
    seq = words + filler
    rng.shuffle(seq)
    return " ".join(seq) + "."


def repetitive_prose(n_words, seed):
    """Ordinary conversational prose: small vocabulary, lots of function
    words -- real prose repeats constantly, which is what makes this a
    fair control for the novelty/stopword rule (a fixed single sentence
    would trip the tail-loop rule instead and make the comparison
    meaningless -- same trap SP\\p18a-findings.md Q2 records)."""
    rng = random.Random(seed)
    vocab = ["the", "reply", "moves", "on", "to", "the", "next", "point",
             "and", "does", "not", "linger", "there", "for", "long", "she",
             "said", "that", "it", "was", "fine", "with", "her", "and",
             "we", "can", "keep", "going", "from", "here", "if", "you",
             "want", "to"]
    return " ".join(rng.choice(vocab) for _ in range(n_words))


def cascade_words(n_words, seed):
    """A stretch of wide-vocabulary, function-word-light text -- nonsense
    syllables exercising exactly the two properties the rule measures
    (near-total novelty, near-zero stopword density), not real English."""
    rng = random.Random(seed)
    syll = ["ba", "ti", "ko", "lu", "sen", "dra", "mir", "fen", "vor", "cal",
            "ith", "zan", "ru", "pel", "os", "tark", "wen", "gil", "nor",
            "fay", "quen", "hesh", "obril", "sana", "delth", "korr", "amyn",
            "ost", "vael", "ruhn"]
    words, used = [], set()
    while len(words) < n_words:
        w = "".join(rng.choice(syll) for _ in range(rng.randint(2, 3)))
        if w not in used:
            used.add(w)
            words.append(w)
        else:
            words.append(w + "x" + str(len(words)))  # force near-uniqueness
    return " ".join(words)


def embedded_cascade_fixture(cascade_words_n=70):
    """~460 words: repetitive prose, a cascade embedded in the middle,
    repetitive prose again -- the shape the coordinator's calibration
    describes: the cascade is a SECTION of a normal-length reply, not the
    whole thing."""
    pre = repetitive_prose(200, seed=1)
    cascade = cascade_words(cascade_words_n, seed=2)
    post = repetitive_prose(200, seed=3)
    return pre + ". " + cascade + ". " + post + "."


def whole_reply_novelty(text):
    words = [w.lower() for w in main._WORD_RE.findall(text)]
    return len(set(words)) / len(words) if words else 0.0


def control_prose(n_sentences=140, seed=3):
    """Ordinary synthetic prose: varied enough not to trip the OTHER
    degeneracy rules (a fixed single sentence repeated trips the tail-loop
    rule and would make every comparison here meaningless -- same lesson
    SP\\p18a-findings.md Q2 records), but built from a small fixed
    vocabulary, so word novelty is LOW, same as real prose."""
    subjects = ["the operator", "the pod", "the batch job", "the reviewer",
                "the summary", "the guard", "the log line", "the request"]
    verbs = ["reported", "checked", "queued", "measured", "closed",
             "rebuilt", "flagged", "retried"]
    objects = ["the figure", "the window", "the ledger", "the counter",
               "the archive", "the token budget", "the schema", "the record"]
    tails = ["before the next cycle.", "without further comment.",
             "after the retry succeeded.", "and moved on.",
             "once the lock cleared.", "ahead of the deadline.",
             "the same way it always had.", "and logged the outcome."]
    rng = random.Random(seed)
    out = []
    for _ in range(n_sentences):
        out.append(f"{rng.choice(subjects).capitalize()} {rng.choice(verbs)} "
                    f"{rng.choice(objects)} {rng.choice(tails)}")
    return " ".join(out)


print("[1] shape A: the symbol wall fires, and reports a span")
A = prose(1200) + "\n\n" + wall(45, 260) + "\n\n" + prose(900)
reason, start, end = main._reply_degenerate_verdict_uncached(A)
check(reason is not None and "symbol characters" in reason,
      f"a 260-char wall of 45 distinct symbols in prose is flagged  [{reason}]")
check(start is not None and end is not None and 0 <= start < end <= len(A),
      f"the verdict carries a real span (start={start}, end={end}, len={len(A)})")
check(start is not None and end is not None
      and (end - start) <= main._SYMBOL_WALL_WINDOW,
      "the span is at most one window wide, not the whole reply")

print()
print("[2] shape A: the ORIGINAL small-wall fixture (7 distinct symbols) "
      "is a different, uncalibrated shape and must stay unflagged")
small_wall = wall(7, 120)
B_small = prose(4000) + "\n\n" + small_wall + "\n\n" + prose(4000)
r2, _, _ = main._reply_degenerate_verdict_uncached(B_small)
check(r2 is None,
      "7 distinct symbols in a 120-char wall does not clear the 30-distinct "
      "floor and is not this rule's target shape")

print()
print("[3] shape A: distinct-count and window-count are BOTH required")
only_window = "≈" * 150  # 1 distinct symbol, high window count, but
                                # under DEGENERATE_RUN_CHARS (250) too, so
                                # this isolates the distinct-count
                                # requirement from the pre-existing
                                # single-char-run rule rather than tripping
                                # that rule instead
r3a, _, _ = main._reply_degenerate_verdict_uncached(
    prose(2000) + only_window + prose(2000))
check(r3a is None,
      "a window full of ONE repeated symbol (1 distinct, 150-char run, "
      "under the existing run rule's own 250 floor) does not trip the "
      "symbol-wall rule -- distinct count is a real second requirement, "
      "not window density alone")
scattered = "".join(symbol_pool(45))  # 45 distinct, 1 each, spread across
                                        # ~8000 chars of prose -- never
                                        # dense enough for one 200-char
                                        # window to hold 100 of them
r3b, _, _ = main._reply_degenerate_verdict_uncached(
    prose(4000) + "".join(c + prose(80) for c in scattered))
check(r3b is None,
      "45 distinct symbols thinly scattered through 8k+ of prose (never "
      "100+ in any single 200-char window) is not flagged")

print()
print("[4] shape B, SECOND PASS: reproduces why the whole-reply form (this "
      "file's own first revision) never separates on the real corpus")
Bfix = embedded_cascade_fixture(cascade_words_n=70)
_wr_novelty = whole_reply_novelty(Bfix)
check(_wr_novelty < 0.881,
      f"a 70-word cascade embedded in a 460-word reply scores "
      f"whole-reply novelty {_wr_novelty:.3f} -- BELOW the max (0.881) the "
      f"coordinator measured across all 5,895 real replies, confirming the "
      f"whole-reply ratio drowns an embedded cascade exactly as measured, "
      f"not just as argued")
r4, s4, e4 = main._reply_degenerate_verdict_uncached(Bfix)
check(r4 is not None and "novel words" in r4,
      f"the WINDOWED rule catches the same fixture the whole-reply form "
      f"could never have caught  [{r4}]")
check(s4 is not None and e4 is not None and 0 <= s4 < e4 <= len(Bfix),
      f"the verdict carries a real span (start={s4}, end={e4}, len={len(Bfix)})")
_span_words = len(main._WORD_RE.findall(Bfix[s4:e4]))
check(_span_words == main.DEGENERATE_SYNONYM_WINDOW_WORDS,
      f"the span is exactly one window wide "
      f"({_span_words} words, window={main.DEGENERATE_SYNONYM_WINDOW_WORDS})"
      f" -- not the whole 460-word reply")

print()
print("[5] shape B: the conjunction is load-bearing -- novelty alone and "
      "low-stopword-density alone must each fail to fire on their own")
# Isolated (not embedded in surrounding prose, so the best-scoring window
# cannot drift across a boundary and mix results): a genuinely novel
# vocabulary with stopwords woven through it at ordinary density -- the
# corpus's own "24.4% of replies have a <0.05-stopword 100-word window
# somewhere" finding is why novelty alone was rejected as a signal.
rng_mix = random.Random(42)
_stopword_sample = ["the", "and", "of", "to", "in", "is", "that"]
_cascade_only = cascade_words(60, seed=5).split()
_mixed = []
for i, w in enumerate(_cascade_only):
    _mixed.append(w)
    if i % 4 == 3:
        _mixed.append(rng_mix.choice(_stopword_sample))
high_novelty_high_stopword = " ".join(_mixed)
_span5a = main._synonym_cascade_span(high_novelty_high_stopword)
check(_span5a is not None and _span5a[2] >= 0.85,
      f"fixture check: this window is genuinely high-novelty "
      f"(novelty={_span5a[2] if _span5a else None})")
check(_span5a is not None and _span5a[3] >= main.DEGENERATE_SYNONYM_STOPWORD_FRACTION,
      f"fixture check: and genuinely above the stopword floor "
      f"(stopword_frac={_span5a[3] if _span5a else None} >= "
      f"{main.DEGENERATE_SYNONYM_STOPWORD_FRACTION})")
r5a, _, _ = main._reply_degenerate_verdict_uncached(high_novelty_high_stopword)
check(r5a is None,
      "a novel-vocabulary window with stopwords woven through it at "
      "ordinary density does not fire -- novelty alone is not enough")

# The opposite failure mode: LOW novelty (a small pool of distinct
# nonsense words drawn RANDOMLY, not cycled -- a strict cyclic repeat like
# A B C A B C ... is itself a repeating PHRASE and correctly belongs to the
# pre-existing tail-loop rule, a few rules earlier in the same function;
# random draws from a small pool give the same low novelty without also
# being that different shape) but genuinely zero stopwords.
_pool = ["zanfleth", "korrsana", "obrilax", "delthar", "vaelnor", "oshtiq"]
_rng_pool = random.Random(21)
low_novelty_low_stopword = " ".join(_rng_pool.choice(_pool) for _ in range(60))
_span5b = main._synonym_cascade_span(low_novelty_low_stopword)
check(_span5b is not None and _span5b[2] < main.DEGENERATE_SYNONYM_NOVELTY_FRACTION,
      f"fixture check: this window is genuinely low-novelty "
      f"(novelty={_span5b[2] if _span5b else None} < "
      f"{main.DEGENERATE_SYNONYM_NOVELTY_FRACTION})")
check(_span5b is not None and _span5b[3] == 0.0,
      f"fixture check: and genuinely zero stopwords "
      f"(stopword_frac={_span5b[3] if _span5b else None})")
r5b, _, _ = main._reply_degenerate_verdict_uncached(low_novelty_low_stopword)
check(r5b is None,
      "a function-word-free window drawn from a 6-word pool does not fire "
      "-- zero stopwords alone is not enough")

print()
print("[6] shape B: the minimum-reply-length floor exempts short replies")
short = cascade_words(40, seed=6)  # well under the 60-word window itself
r6, _, _ = main._reply_degenerate_verdict_uncached(short)
check(r6 is None,
      "a 40-word reply has no full 60-word window to judge at all, "
      "regardless of its own novelty")
check(main._synonym_cascade_span(short) is None,
      "_synonym_cascade_span itself returns None below the window floor, "
      "not just a verdict that happens not to fire")

print()
print("[7] shape B: the English-only stopword list is overridable by "
      "environment, replacing the default wholesale (a fresh process, "
      "since _STOPWORDS is resolved at import time)")
_env = dict(os.environ)
_env["MODEL_REPO"] = ""
_env["COMPACTOR_DEGENERATE_STOPWORDS"] = "zanfleth,korrsana"
_probe = (
    "import sys; sys.path.insert(0, r'" +
    os.path.dirname(os.path.abspath(__file__)) + "'); import main; "
    "print(sorted(main._STOPWORDS))"
)
_proc = subprocess.run(
    [sys.executable, "-c", _probe], env=_env, capture_output=True, text=True,
    timeout=60,
)
check(_proc.returncode == 0, f"the override subprocess ran cleanly "
      f"(stderr: {_proc.stderr[-300:] if _proc.returncode else ''!r})")
check("['korrsana', 'zanfleth']" == _proc.stdout.strip(),
      f"COMPACTOR_DEGENERATE_STOPWORDS REPLACES the default list wholesale "
      f"(got {_proc.stdout.strip()!r})")

print()
print("[8] control: ordinary synthetic prose is unaffected by either new "
      "rule under the SHIPPED defaults")
ctrl_default = control_prose()
r8, _, _ = main._reply_degenerate_verdict_uncached(ctrl_default)
check(r8 is None, "ordinary prose stays healthy under shipped defaults")
# The repetitive-vocabulary fixture used as shape B's own control above,
# at realistic length, also stays healthy.
r8b, _, _ = main._reply_degenerate_verdict_uncached(repetitive_prose(460, seed=9))
check(r8b is None,
      "an all-repetitive, no-cascade reply of realistic length stays "
      "healthy under shipped defaults")

print()
print("[9] M2: the new rules' fire counts are visible and additive")
before = main.degeneracy_rule_counters()
main._reply_degenerate_verdict_uncached(A)  # same text as [1]; uncached
                                             # call always re-runs and
                                             # re-increments
after = main.degeneracy_rule_counters()
check(after["symbol_wall"] == before["symbol_wall"] + 1,
      f"symbol_wall counter incremented on a fresh uncached call "
      f"({before['symbol_wall']} -> {after['symbol_wall']})")
before2 = main.degeneracy_rule_counters()
main._reply_degenerate_verdict_uncached(Bfix)  # same text as [4]
after2 = main.degeneracy_rule_counters()
check(after2["word_novelty"] == before2["word_novelty"] + 1,
      f"word_novelty counter incremented on a fresh uncached call "
      f"({before2['word_novelty']} -> {after2['word_novelty']}) -- ACTIVE "
      f"now, not inert")

print()
if _FAILED:
    print("SOME v3195-main M1 CHECKS FAILED")
    sys.exit(1)
print("All v3195-main M1 checks passed.")
