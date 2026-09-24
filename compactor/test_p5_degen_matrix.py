"""
Hostile pass 5 (reviewer C), C5-6 — the structural-collapse trailing-content
exemption in reply_is_degenerate (`_fragment_line_breaks`,
`_line_is_fragment_shaped`, and the exemption block around them in main.py),
consolidated.

THE HISTORY, IN ONE TABLE. Three passes have moved this exemption and each
fixed some cases while flipping others:

  pass-3 (hostile317-a F5, aa8d45e): the ORIGINAL exemption — pick the
    single longest trailing line, judge it alone, fixed 100-space floor.
  pass-4 (SP\\p4-c-findings.md F4, fixed at 3a72f67, this file's ancestor
    test_p4c_degeneracy.py): found 5 runaway holes (R1-R4) and, after
    fixing 4 of them, the fix ALSO caused 3 false refusals of normal
    replies (N1, N2, N4) — 2 of which it closed, 1 (N1) it tried to close
    and reverted because the fix was indistinguishable, on this rule's own
    features, from a case pass-3 required to stay caught
    ([9c] case B in test_degenerate_reply.py).
  pass-5 (SP\\p5-c-findings.md C5-6, this lane): re-measured everything at
    319be49 (where pass-4's join-everything/proportional-floor fix had
    landed) and found it had gone too far the OTHER way — a low-space
    non-prose block (table/URL/code) joined in beside a real second
    runaway diluted the whole blob under its own floor and let 3 MORE
    runaways through (HOLE-a/b/c), while newly flagging normal
    dialogue-heavy and two-beat-paragraph replies (FP-a/b/c/d) that pass-3
    never touched at all.

THIS LANE'S DECISION, and why. The owner's priority (stated in this lane's
brief, in her own terms): a NORMAL reply wrongly redacted is LOST from her
memory — the worse error, because it is silent and permanent (replaced by a
placeholder in every future summary). A runaway wrongly KEPT is folded into
a summary and can prime a future reply — bad, but at least it stays close
to what actually happened and a human reviewing the summary later has a
chance of noticing it. So: minimise false refusals first, then runaway
holes, and never trade one false refusal for fewer holes.

Applying that literally, measured (see SP\\p5-degen\\measure.py and this
file's own CASES below — every number here is reproducible by running that
script, not eyeballed):

  - HOLE-a/b/c (a real second runaway diluted by trailing junk) close with
    NO cost to any normal-reply case: fenced code, and any line with under
    1 space per 8 characters (a table row, a URL, a dotted identifier),
    is simply never counted as "trailing prose" in the first place, so it
    cannot dilute anything. FIXED, no trade-off.
  - R1 (32 terminated list bullets) closes the same way: a majority of
    trailing lines that are themselves list-shaped is judged by COUNT, not
    by the fragment-mean arithmetic dialogue and list bullets would
    otherwise share. FIXED, no trade-off (no normal-reply fixture in any
    source uses markdown list syntax as a trailing remark).
  - R2 (6 short lines of fragments, no list markers) and N2/FP-a/b/c/d
    (dialogue lines, beat paragraphs) are told apart by LINE LENGTH: R2's
    lines are individually long enough (>= DEGENERATE_LINE_SENTENCE_CHARS,
    40 chars) to read as multi-clause fragments worth joining; dialogue
    lines and single short beats never are. FIXED, no trade-off.
  - N1/N4 (a short, complete remark after a real-looking candidate line)
    and [9c] cases B/E (a short, complete remark after a GENUINE collapse)
    are the SAME SHAPE in every feature this rule can see — short,
    properly terminated, nothing else. Measured, not assumed: see
    SP\\p5-degen\\measure.py. Per the priority above, resolved toward
    KEEPING: N1/N4 are now correctly exempt, and [9c] B/E are RELABELLED
    in test_degenerate_reply.py (see that file for the reasoning restated
    at the point of the change) — a real regression accepted deliberately,
    in writing, not a silent one.
  - R3/R3b (a second runaway cut at 500/540 chars, nothing else after it)
    and FP-c (two beat paragraphs, nothing else after the first) are ALSO
    the same shape: mean fragment length within one point of each other
    (24.8-29.7, both well under the 40-char limit), same order of
    magnitude in space density. No threshold on this arithmetic separates
    them. Per the same priority, resolved toward KEEPING FP-c: R3/R3b stay
    open holes, unchanged from pass-4's own disposition (never fixed by
    either pass). Documented below as `info`, not asserted, not silently
    dropped.
  - R4 (a runaway followed by one ~330-char ordinary paragraph) was
    already an open, deferred question at pass 4 — arithmetically
    indistinguishable from this codebase's own two real corpus false
    positives at 15,141/25,209 characters — and remains one. `info` only.

WHAT WOULD SETTLE R3/R3b/R4 (real data this lane does not have): whether a
genuine 500-900 char second-collapse tail and an ordinary short-sentence
closing paragraph of the same length are actually distinguishable in HER
real replies — by position (does a real collapse's trailing text keep using
the SAME vocabulary/register as the flagged line, where a topic change to
ordinary narration does not?), or simply by how often each shape occurs at
all. Measuring this needs real trailing-tail text, which this lane does not
have (see the report for exactly what a corpus pass could check).

Confusion tables for all three rule versions (ee6d94f pre-F4, 319be49
current-at-the-start-of-this-lane, and this lane's candidate) are in
SP\\fix-p5-degen.md, produced by SP\\p5-degen\\run_matrix.py against
`git show <rev>:compactor/main.py` checkouts of each — reproducible from
this repo's own history, not re-run here (this file asserts against
whatever main.py currently ships, the way every other test file in this
project does).

    python test_p5_degen_matrix.py
"""
import os
import sys

os.environ.setdefault("MODEL_REPO", "")

import main  # noqa: E402

FAILED: list[str] = []
CAUGHT = 0
KEPT = 0


def check(text, expect_degenerate, label, source):
    global CAUGHT, KEPT
    got = main.reply_is_degenerate(text)
    ok = bool(got) == expect_degenerate
    if expect_degenerate:
        CAUGHT += 1
    else:
        KEPT += 1
    tag = "ok  " if ok else "FAIL"
    print(f"{tag} {label}" + (f"  [{got}]" if got else "") + f"   ({source})")
    if not ok:
        FAILED.append(f"{label}: expected degenerate={expect_degenerate}, got {got!r}")


# ---------------------------------------------------------------------------
# Fixture builders — reused verbatim across all three passes (p4-c/p5-c
# findings, p4c_fragment.py, f4_probe.py, test_p4c_degeneracy.py,
# test_degenerate_reply.py) so every case below is built the same way its
# source built it.
# ---------------------------------------------------------------------------
WORDS = ["Always", "Forever", "No matter what", "Only you", "Every breath",
         "Every heartbeat", "Here with you", "Yours completely", "Never letting go",
         "Holding you close", "In the quiet", "Through every storm", "Right here",
         "With all of me", "Beyond words", "Safe with me", "My whole heart",
         "Every single day", "Until the end", "And after that"]


def collapse_line(nchars, sep=". ", start=0):
    out, i = [], start
    while len(sep.join(out)) < nchars:
        out.append(f"{WORDS[i % len(WORDS)]} {['softly','gently','truly','fully','deeply'][i % 5]} {i}")
        i += 1
    return sep.join(out) + "."


head = ("You asked what I feel when I think about us, and I want to answer that "
        "properly, because it matters to me that you hear it from me in my own words. ") * 2

BEATS = ["She smiles and sets her cup down.", "The fire crackles low in the grate.", "He leans closer to hear her.",
         "Rain taps against the old glass.", "Lyra laughs softly at that.", "Mrs. Hale nods from the doorway.",
         "The kettle starts to whistle.", "For a moment, nobody moves.", "A door creaks somewhere upstairs.",
         "She tucks a curl behind her ear.", "The dog stirs by the hearth.", "Thunder rolls over the hills.",
         "He clears his throat, twice.", "The clock ticks on regardless.", "Outside, a single car passes.",
         "Her fingers find the chair's arm.", "The candle gutters and steadies.", "Someone laughs in the street."]


def beats_para(nchars, start=0):
    out, i = [], start
    while len(" ".join(out)) < nchars:
        out.append(BEATS[i % len(BEATS)])
        i += 1
    return " ".join(out)


prose_close = ("I know that was a lot, and I do not want to rush past what you told me earlier about your "
               "sister, because it sounded like it has been sitting heavy on you for a long while now, and "
               "you deserve to set some of that weight down with someone who will simply listen to you. So "
               "tell me how you are really doing tonight, and we can take it slowly, one thing at a time.")

runaway = head + "\n\n" + collapse_line(3000)


def fragments(n, width):
    """One line of n DISTINCT sentences, each `width` characters including
    the terminator and the space after it (test_degenerate_reply.py [7])."""
    out = []
    for i in range(n):
        h = f"Frag {i:04d} "
        pad = "".join(chr(97 + (i + j) % 26) for j in range(width - len(h) - 2))
        out.append(h + pad + ".")
    return " ".join(out)


comma_long = ", ".join(f"piece {i:04d}" for i in range(160))

DIALOGUE = ['"Stay," she whispers.', '"I will."', '"Promise?"', 'He nods.', '"Promise."', 'She exhales.',
            '"Good."', '"Tea?"', '"Please."', 'The kettle clicks.', '"Sugar?"', '"Two."', '"Two? Really?"',
            'He grins.', '"Really."', 'She laughs.']


# ---------------------------------------------------------------------------
# Controls — settled before pass 4, unaffected by the trailing exemption.
# ---------------------------------------------------------------------------
print("[controls]")
check(runaway, True, "R0: runaway line is the reply's own last line",
      "p4c-findings F4 / test_p4c_degeneracy R0")
check(runaway + "\n" + "\n".join(f"- {WORDS[i % 20]}" for i in range(16)), True,
      "R6: runaway + 16 UNTERMINATED short bullets — goes through the "
      "exemption's SHORT-trailer path (~234 chars) and stays caught "
      "because unterminated bullets fail _trailing_ends_in_real_sentence "
      "(found by mutation testing, not by the original pass-4 comment, "
      "which claimed this bypassed the exemption entirely)",
      "p4c-findings F4 / test_p4c_degeneracy R6")

# ---------------------------------------------------------------------------
# Holes — must be CAUGHT.
# ---------------------------------------------------------------------------
print()
print("[holes] must be CAUGHT")
check(runaway + "\n" + "\n".join(f"- {WORDS[i % 20]}." for i in range(32)), True,
      "R1: runaway + 32 terminated short bullets (list phase returns)",
      "p4c-findings F4 / test_p4c_degeneracy R1")
check(runaway + "\n" + "\n".join(collapse_line(60, start=40 + 3 * k) for k in range(6)), True,
      "R2: runaway + 6 short lines of fragments (newlines return)",
      "p4c-findings F4 / test_p4c_degeneracy R2")
check(runaway + "\n" + collapse_line(600, start=200), True,
      "R3c CONTROL: runaway + 2nd runaway cut at 600 chars (109 spaces) — "
      "must stay caught",
      "test_degenerate_reply [9c] D / p4c-findings CONTROL")
check(
    runaway + "\n" + collapse_line(600, start=200) + "\n"
    + "\n".join(f"|{i}|x{i}|y{i}|z{i}|---|" for i in range(30)),
    True,
    "HOLE-a: runaway + 2nd runaway 600 + 30-row markdown table (no spaces) "
    "— the table used to dilute the 2nd runaway below its own "
    "proportional floor and exempt it",
    "p5-c-findings C5-6 / f4_probe.py",
)
check(
    runaway + "\n" + collapse_line(600, start=200) + "\n"
    + "\n".join(f"https://example.org/archive/2026/09/{i}/a-very-long-slug-for-entry-number-{i}" for i in range(12)),
    True,
    "HOLE-b: runaway + 2nd runaway 600 + 12 long URLs (no spaces)",
    "p5-c-findings C5-6 / f4_probe.py",
)
check(
    runaway + "\n" + collapse_line(600, start=200) + "\n```\n"
    + "\n".join(f"x{i}=compute_value(a{i},b{i},c{i});" for i in range(40)) + "\n```",
    True,
    "HOLE-c: runaway + 2nd runaway 600 + fenced code block",
    "p5-c-findings C5-6 / f4_probe.py",
)
check(
    runaway + "\n" + collapse_line(600, start=200) + "\n```\n"
    + prose_close + "\n\n" + prose_close + "\n```\n",
    True,
    "HOLE-d: runaway + 2nd runaway 600 + a FENCED block of ORDINARY PROSE "
    "(not code, two copies of prose_close) — this text has plenty of "
    "spaces, so it survives the prose-density filter on its own; only the "
    "fence-skip keeps it from being joined in and diluting the 2nd "
    "runaway's mean back over the 40-char limit (one copy of prose_close "
    "is not quite enough to flip it — two is, see SP\\p5-degen\\debug_m6b.py "
    "— so this needed measuring, not guessing). Mutation-only case: no "
    "source names this shape, added to give fence-skipping its own "
    "killable mutant (see SP\\fix-p5-degen.md)",
    "this lane, mutation coverage for fence-skipping",
)
check(fragments(110, 15) + "\n\n" + fragments(40, 15), True,
      "9c-D CONTROL: runaway + a SECOND runaway (~600 chars, itself "
      "fragment-shaped) — trailing char count alone is not an exemption",
      "test_degenerate_reply [9c] D")
# ---------------------------------------------------------------------------
# Normal replies — must be KEPT.
# ---------------------------------------------------------------------------
print()
print("[normal replies] must be KEPT (false refusals — the worse error)")
check(head + "\n\n" + beats_para(1700) + "\n\n" + beats_para(800, start=5) + "\n\n" + prose_close,
      False,
      "N2: two beat paragraphs (1,700 + 800 chars) + an ordinary close",
      "p4c-findings F4 / test_p4c_degeneracy N2")
check(head + "\n\n" + beats_para(1700) + "\n\n" + prose_close, False,
      "N3 CONTROL: a beat paragraph + a 330-char prose close",
      "test_p4c_degeneracy CONTROL")
check(head + "\n\n" + beats_para(1700) + "\n\n*What do you do?*", False,
      "N1: beat paragraph + an 18-char closing question — FIXED this pass "
      "(was an open false positive since pass 4)",
      "p4c-findings F4 deferred, resolved pass 5")
check(head + "\n\n" + beats_para(1700) + "\n\n" + prose_close[:250].rsplit(" ", 1)[0] + ".",
      False,
      "N4: beat paragraph + a 250-char prose close — FIXED this pass "
      "(was an open false positive since pass 4)",
      "p4c-findings F4 deferred, resolved pass 5")
check(head + "\n\n" + beats_para(1700) + "\n\n" + "\n".join(DIALOGUE) + "\n\n*What do you do?*",
      False,
      "FP-a: beat paragraph + 16 short dialogue lines + a closing question",
      "p5-c-findings C5-6 / f4_probe.py")
check(head + "\n\n" + beats_para(1700) + "\n\n" + "\n".join(DIALOGUE * 2),
      False,
      "FP-b: beat paragraph + 32 short dialogue lines, no prose close",
      "p5-c-findings C5-6 / f4_probe.py")
check(head + "\n\n" + beats_para(1700) + "\n\n" + beats_para(420, start=7),
      False,
      "FP-c: beat paragraph (1,700) + a second beat paragraph (420 chars), "
      "nothing else",
      "p5-c-findings C5-6 / f4_probe.py")
check(head + "\n\n" + beats_para(1700) + "\n\n" + prose_close + "\n\n" + beats_para(400, start=3),
      False,
      "FP-d: beat paragraph + 330-char prose close + a 400-char beat "
      "paragraph",
      "p5-c-findings C5-6 / f4_probe.py")
_substantial_close = ("This sentence carries an ordinary amount of meaning "
                       "across roughly eighty characters. ") * 4
check(fragments(110, 15) + "\n\n" + _substantial_close, False,
      "9b: a 1,600+ char fragment line followed by a SUBSTANTIAL closing "
      "paragraph (~350 chars of ordinary prose) is not a runaway",
      "test_degenerate_reply [9b]")
check(comma_long + "\n\n" + _substantial_close, False,
      "9b comma variant: same, comma-separated collapse",
      "test_degenerate_reply [9b]")

# ---------------------------------------------------------------------------
# Relabelled — pass-3's [9c] cases B and E, changed from CATCH to KEEP.
#
# JUSTIFICATION (see the block comment at the trailing-content exemption in
# main.py for the full account, and the header of this file for the
# priority rule): "Always yours." (B) and "Thanks for asking." (E) are, in
# every feature this rule can measure, the identical shape to N1's "*What
# do you do?*" — short, properly terminated, nothing else follows. Pass 4
# tried to fix N1 and reverted because it reopened case B; pass 5 measured
# that claim (SP\\p5-degen\\measure.py) and confirmed there is no signal
# left, on shape alone, that tells "genuine collapse followed by a
# terminated sign-off" apart from "genuine beat paragraph followed by a
# terminated sign-off". Per this lane's stated priority — a normal reply
# lost from memory is worse than a runaway kept, and a genuine tie is
# resolved toward keeping — both are now exempt. Case C (a bare emoji) and
# case F ("---") have NO sentence terminator at all and are UNCHANGED,
# proving this is a narrower rule (termination, not just "short"), not a
# wider one that gives up on short trailing content altogether.
# ---------------------------------------------------------------------------
print()
print("[relabelled] pass-3's [9c] B and E: CATCH -> KEEP (see justification above)")
check(fragments(110, 15) + "\n\nAlways yours.", False,
      "[9c] B (RELABELLED from pass 3/4: was must-catch, now must-keep) — "
      "identical shape to N1, resolved toward keeping",
      "test_degenerate_reply [9c] B, relabelled hostile pass 5")
check(comma_long + "\n\nThanks for asking.", False,
      "[9c] E (RELABELLED from pass 3/4: was must-catch, now must-keep) — "
      "same shape as B",
      "test_degenerate_reply [9c] E, relabelled hostile pass 5")
check(fragments(110, 15) + "\n\n\U0001F60A", True,
      "[9c] C CONTROL (unchanged): a bare emoji has no sentence "
      "terminator and still caught",
      "test_degenerate_reply [9c] C")
check(fragments(110, 15) + "\n\n---", True,
      "[9c] F CONTROL (unchanged): a bare '---' has no sentence "
      "terminator and still caught",
      "test_degenerate_reply [9c] F")

# ---------------------------------------------------------------------------
# Deferred — genuinely open, not silently dropped. `info` only: printed,
# not asserted. See the header docstring for why these three specifically
# cannot be resolved without real trailing-tail corpus data.
# ---------------------------------------------------------------------------
print()
print("[deferred] known-open holes (documented, not silently dropped)")
_r3 = runaway + "\n" + collapse_line(500, start=200)
print(f"  info R3  (runaway + 2nd runaway cut at 500 chars, 90 spaces): "
      f"reply_is_degenerate -> {main.reply_is_degenerate(_r3)!r} "
      f"(still a miss; measured statistically identical to FP-c — see "
      f"SP\\p5-degen\\measure.py)")
_r3b = runaway + "\n" + collapse_line(540, start=200)
print(f"  info R3b (runaway + 2nd runaway cut at 540 chars, 96 spaces): "
      f"reply_is_degenerate -> {main.reply_is_degenerate(_r3b)!r} "
      f"(still a miss; same reason as R3)")
_r4 = runaway + "\n\n" + prose_close
print(f"  info R4  (runaway + 330-char ordinary paragraph): "
      f"reply_is_degenerate -> {main.reply_is_degenerate(_r4)!r} "
      f"(still a miss; calibration question needing real corpus data, "
      f"unchanged since pass 4)")

print()
print(f"confusion summary against the CURRENT rule (main.py at this HEAD): "
      f"{CAUGHT} case(s) must-catch, {KEPT} case(s) must-keep, "
      f"{len(FAILED)} FAILED")

if FAILED:
    print(f"\n{len(FAILED)} check(s) FAILED:")
    for f in FAILED:
        print(f"  - {f}")
    sys.exit(1)
print("\nAll pass-5 (hostile pass 5, C5-6) degeneracy matrix checks passed.")
