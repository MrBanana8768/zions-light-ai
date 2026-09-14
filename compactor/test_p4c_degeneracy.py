"""
Hostile pass 4 (reviewer C), F4: the trailing-content degeneracy exemption
(reply_is_degenerate's structural-collapse rule) let real runaways through
AND wrongly redacted normal replies from memory.

Findings source: SP\\p4-c-findings.md F4. Fixture shapes reused from the
reviewer's own reproduction (SP\\p4-c\\p4c_fragment.py) — same word lists,
same shape builders, rebuilt here as assertion-based tests per the fix-lane
brief.

STATUS AS OF HOSTILE PASS 5 (C5-6, SP\\p5-c-findings.md): this file's own
pass-4 fix (join-everything, proportional floor) was found to have reopened
3 MORE runaway holes (a low-space non-prose block joined in beside a real
second runaway diluted it below its own floor) while newly flagging normal
dialogue-heavy and two-beat-paragraph replies pass-4 never touched. Pass 5
replaced the exemption again — see the block comment above it in main.py,
and the consolidated compactor/test_p5_degen_matrix.py, which is now the
authoritative record of every shape from every pass and the reasoning
behind each label. This file is kept as a real regression test (it is not
redundant with the matrix file: it pins the exact pass-4 shapes against
their own history) with the specific checks pass 5 changed updated in
place, named below.

Two independent problems from pass 4, both revisited at pass 5:

  HOLES (a runaway that should be caught, was not): a second runaway cut
  short (500/540 chars, 90/96 spaces) never cleared the OLD exemption's
  fixed 100-space-per-line floor; trailing content spread over many short
  lines (terminated bullets, several short fragment lines) had no single
  line long enough to trip the old per-line check. Pass 4 "fixed" all of
  these by judging the trailing content AS A WHOLE (joined), with a spaces
  floor PROPORTIONAL to its own length — which pass 5 found reopened 3
  DIFFERENT holes (HOLE-a/b/c in test_p5_degen_matrix.py) the same way.
  Pass 5's replacement closes R1/R2 (list-majority / multi-line-join, see
  main.py) with no new cost, but found R3/R3b measured statistically
  identical to a normal two-beat-paragraph reply (FP-c) and, per this
  lane's stated priority, left them open rather than risk that false
  refusal — see where they are now asserted (moved to `info`, below).

  REGRESSION (a normal reply wrongly redacted): a "beat" paragraph (short
  scene-setting sentences, e.g. "She smiles. The fire crackles.") has a low
  apparent mean-fragment-length by the SAME arithmetic that flags a real
  collapse, purely from being short-sentenced, not degenerate. When it was
  the single longest trailing line, the old rule flagged it directly.
  Pass 4 fixed N2 (a second such paragraph followed by ordinary prose) by
  the same join; N1 (a runaway followed by only a SHORT beat-style remark,
  an 18-character closing question) was tried, shown to fix N1, and
  REVERTED after it reopened `test_degenerate_reply.py` [9c] case B (a
  short, terminated SIGN-OFF after a REAL collapse, which that pass-3
  fixture required to stay caught and which is the identical shape to N1
  with the opposite correct answer) — see SP\\fix-p4c.md F4 for the full
  account of that attempt. Pass 5 measured the "identical shape" claim
  directly (SP\\p5-degen\\measure.py) instead of taking it on faith,
  confirmed [9c] case E is the same shape too, found no signal that
  separates any of them, and — per this lane's stated priority, a normal
  reply lost from memory is worse than a runaway kept — resolved the tie
  toward KEEPING: N1/N4 are now fixed, and [9c] cases B and E are
  RELABELLED (both here and in test_degenerate_reply.py) rather than left
  as an unresolved conflict.

    python test_p4c_degeneracy.py
"""
import os
import sys

os.environ.setdefault("MODEL_REPO", "")

import main  # noqa: E402

FAILED: list[str] = []


def check(text, expect_degenerate, label):
    got = main.reply_is_degenerate(text)
    ok = bool(got) == expect_degenerate
    if ok:
        print(f"  ok   {label}" + (f"  [{got}]" if got else ""))
    else:
        print(f"FAIL {label}: expected degenerate={expect_degenerate}, got {got!r}")
        FAILED.append(label)


# ---------------------------------------------------------------------------
# Fixture builders — reused verbatim from SP\p4-c\p4c_fragment.py.
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


# ---------------------------------------------------------------------------
# Holes — FIXED: these must now be caught.
# ---------------------------------------------------------------------------
print("[F4 holes] must now be CAUGHT (were missed before this fix)")

check(runaway, True, "R0 control: runaway line is the reply's own last line")

check(runaway + "\n" + "\n".join(f"- {WORDS[i % 20]}." for i in range(32)),
      True,
      "R1: runaway + 32 terminated short bullets (list phase returns) — "
      "no single bullet line was ever long enough to trip the OLD "
      "single-longest-line check")

check(runaway + "\n" + "\n".join(collapse_line(60, start=40 + 3 * k) for k in range(6)),
      True,
      "R2: runaway + 6 short lines of fragments (newlines return) — same "
      "shape, spread over several lines instead of one")

# R3/R3b: v3.1.9 (hostile pass 5, C5-6) MOVED these from asserted-True to
# `info` below (with R4) — see that section for why: measured
# (SP\\p5-degen\\measure.py), a 500/540-char second runaway is
# statistically indistinguishable from FP-c (test_p5_degen_matrix.py), an
# ordinary two-beat-paragraph reply of the same length, on every feature
# this rule can see. Per this lane's priority (a normal reply lost from
# memory is worse than a runaway kept), resolved toward keeping FP-c, which
# leaves R3/R3b open — unchanged from their pass-4 disposition, just no
# longer asserted as if pass 4 had closed them.

check(runaway + "\n" + collapse_line(600, start=200), True,
      "R3c CONTROL: runaway + 2nd runaway cut at 600 chars (109 spaces) — "
      "this one already cleared the OLD fixed floor and must stay caught")

check(runaway + "\n" + "\n".join(f"- {WORDS[i % 20]}" for i in range(16)),
      True,
      "R6 CONTROL: runaway + 16 UNTERMINATED short bullets — stays caught. "
      "CORRECTED (hostile pass 5, mutation testing): the trailing content "
      "here is under DEGENERATE_MIN_CHARS (~234 chars), so this DOES go "
      "through the exemption's SHORT-trailer path — it stays caught "
      "because unterminated bullets fail _trailing_ends_in_real_sentence, "
      "not because it bypasses the exemption. (The original pass-4 comment "
      "claimed the opposite; a mutation on the short-trailer check turned "
      "this case red, which is how the mistake was found.)")


# ---------------------------------------------------------------------------
# Regression — PARTIALLY FIXED: N2 must now pass; N1/N4 remain open
# (documented, not silently dropped).
# ---------------------------------------------------------------------------
print()
print("[F4 regression] normal replies must NOT be flagged")

check(head + "\n\n" + beats_para(1700) + "\n\n" + prose_close, False,
      "N3 CONTROL: a beat paragraph + a 330-char prose close was already "
      "exempt before this fix and must stay exempt")

check(head + "\n\n" + beats_para(1700) + "\n\n" + beats_para(800, start=5) + "\n\n" + prose_close,
      False,
      "N2: two beat paragraphs (1,700 + 800 chars) + an ordinary close — "
      "FIXED by judging the trailing content as a whole: the old rule "
      "picked the SECOND beat paragraph as the single longest trailing "
      "line and flagged it on its own fragment-shape; the aggregate mean "
      "(including the prose close) reads as ordinary prose")


# ---------------------------------------------------------------------------
# v3.1.9 (hostile pass 5, C5-6): N1/N4 FIXED (moved from `info` to asserted
# checks below — they were false positives at pass 4, and are not any
# more). R3/R4 remain genuinely open, still `info` only — see the block
# comment above the trailing-content exemption in main.py, and
# test_p5_degen_matrix.py, for the measured reasoning behind each.
# ---------------------------------------------------------------------------
print()
print("[F4 regression, continued] N1/N4 are FIXED this pass (hostile pass "
      "5, C5-6) — see main.py's block comment for the short-trailer "
      "termination check that fixed them")

check(head + "\n\n" + beats_para(1700) + "\n\n*What do you do?*", False,
      "N1: beat paragraph + an 18-char closing question — FIXED (hostile "
      "pass 5): a short, properly TERMINATED remark is now exempt the same "
      "way [9c] B/E are (test_degenerate_reply.py, relabelled)")

check(head + "\n\n" + beats_para(1700) + "\n\n"
      + prose_close[:250].rsplit(" ", 1)[0] + ".", False,
      "N4: beat paragraph + a 250-char prose close — FIXED (hostile pass "
      "5): terminated, same mechanism as N1")


# ---------------------------------------------------------------------------
# Known, documented, still NOT fixed (deferred) — recorded here as an
# honest "still open" pin, not silently dropped. R3/R3b joined this section
# in hostile pass 5 (see the comment where they used to be asserted,
# above): measured statistically identical to FP-c
# (test_p5_degen_matrix.py), a normal two-beat-paragraph reply, on every
# feature this rule can see — resolved toward keeping FP-c, which leaves
# R3/R3b open. R4 was already here at pass 4 and is unchanged.
# ---------------------------------------------------------------------------
print()
print("[F4/C5-6 deferred] known-open holes (documented trade-off, not "
      "silently dropped — see SP\\fix-p5-degen.md)")

_r3 = runaway + "\n" + collapse_line(500, start=200)
_r3_result = main.reply_is_degenerate(_r3)
print(f"  info R3 (runaway + 2nd runaway cut at 500 chars, 90 spaces): "
      f"reply_is_degenerate -> {_r3_result!r} (still a miss; measured "
      f"statistically identical to FP-c — see SP\\p5-degen\\measure.py)")

_r3b = runaway + "\n" + collapse_line(540, start=200)
_r3b_result = main.reply_is_degenerate(_r3b)
print(f"  info R3b (runaway + 2nd runaway cut at 540 chars, 96 spaces): "
      f"reply_is_degenerate -> {_r3b_result!r} (still a miss; same reason "
      f"as R3)")

_r4 = runaway + "\n\n" + prose_close
_r4_result = main.reply_is_degenerate(_r4)
print(f"  info R4 (runaway + 330-char ordinary paragraph): "
      f"reply_is_degenerate -> {_r4_result!r} (still a miss; explicitly a "
      f"calibration question the finding itself says needs real corpus "
      f"data this lane was not granted)")


# ---------------------------------------------------------------------------
# Sibling sweep: the pass-3 [9c] fixtures this fix must not disturb (also
# re-run directly via test_degenerate_reply.py; restated briefly here so
# this file's own pass/fail is a complete signal on its own).
#
# v3.1.9 (hostile pass 5, C5-6) RELABELS B from must-catch to must-keep —
# see test_degenerate_reply.py's [9c] section for the full justification
# (measured identical in shape to N1; resolved toward keeping per this
# lane's stated priority). F is unaffected: no sentence terminator at all.
# ---------------------------------------------------------------------------
print()
print("[sibling] pass-3 [9c] short-trailer-after-a-real-collapse cases: B "
      "is RELABELLED (hostile pass 5, same reasoning as "
      "test_degenerate_reply.py), F stays caught")


check(runaway + "\n\nAlways yours.", False,
      "B (RELABELLED, hostile pass 5 C5-6): runaway + a short, TERMINATED "
      "sign-off ('Always yours.') is now kept, not caught — identical "
      "shape to N1")
check(runaway + "\n\n---", True,
      "F: runaway + a bare '---' is still caught — no sentence terminator")


if FAILED:
    print(f"\n{len(FAILED)} check(s) FAILED:")
    for f in FAILED:
        print(f"  - {f}")
    sys.exit(1)
print("\nAll F4 (hostile pass 4) degeneracy checks passed.")
