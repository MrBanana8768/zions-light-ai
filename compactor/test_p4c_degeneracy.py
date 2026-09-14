"""
Hostile pass 4 (reviewer C), F4: the trailing-content degeneracy exemption
(reply_is_degenerate's structural-collapse rule) let real runaways through
AND wrongly redacted normal replies from memory.

Findings source: SP\\p4-c-findings.md F4. Fixture shapes reused from the
reviewer's own reproduction (SP\\p4-c\\p4c_fragment.py) — same word lists,
same shape builders, rebuilt here as assertion-based tests per the fix-lane
brief.

Two independent problems, two independent (partial) fixes:

  HOLES (a runaway that should be caught, was not): a second runaway cut
  short (500/540 chars, 90/96 spaces) never cleared the OLD exemption's
  fixed 100-space-per-line floor; trailing content spread over many short
  lines (terminated bullets, several short fragment lines) had no single
  line long enough to trip the old per-line check. FIXED: the exemption now
  judges the trailing content AS A WHOLE (joined), with a spaces floor
  PROPORTIONAL to its own length instead of a fixed 100.

  REGRESSION (a normal reply wrongly redacted): a "beat" paragraph (short
  scene-setting sentences, e.g. "She smiles. The fire crackles.") has a low
  apparent mean-fragment-length by the SAME arithmetic that flags a real
  collapse, purely from being short-sentenced, not degenerate. When it was
  the single longest trailing line, the old rule flagged it directly.
  PARTIALLY FIXED: judging the trailing content as a whole (above) also
  fixes the case where a second such paragraph is followed by ordinary
  prose (N2) — the aggregate mean is no longer dominated by one short-
  sentence line. NOT fixed: a runaway followed by only a SHORT beat-style
  remark (the finding's N1, an 18-character closing question) is still a
  false positive — a targeted fix for exactly that case was written, shown
  to fix N1, and then REVERTED after it reopened
  `test_degenerate_reply.py` [9c] case B (a short, terminated SIGN-OFF
  after a REAL collapse, which that pass-3 fixture requires to stay caught
  and which is the identical shape to N1 with the opposite correct answer)
  — see the reverted block comment in main.py (search "TRIED AND REVERTED")
  and SP\\fix-p4c.md F4 for the full account.

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

check(runaway + "\n" + collapse_line(500, start=200), True,
      "R3: runaway + 2nd runaway cut at 500 chars (90 spaces) — used to "
      "read as 'too sparse to judge' under the OLD fixed 100-space floor")

check(runaway + "\n" + collapse_line(540, start=200), True,
      "R3b: runaway + 2nd runaway cut at 540 chars (96 spaces) — same "
      "shape, still under the old fixed floor")

check(runaway + "\n" + collapse_line(600, start=200), True,
      "R3c CONTROL: runaway + 2nd runaway cut at 600 chars (109 spaces) — "
      "this one already cleared the OLD fixed floor and must stay caught")

check(runaway + "\n" + "\n".join(f"- {WORDS[i % 20]}" for i in range(16)),
      True,
      "R6 CONTROL: runaway + 16 UNTERMINATED short bullets — already "
      "caught before this fix (the runaway line itself, not the trailing "
      "exemption) and must stay caught")


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
# Known, documented, NOT fixed (deferred) — recorded here as an honest
# "still broken" pin, not silently dropped. If either of these ever starts
# passing on its own, that is progress worth noting, not a reason to relax
# this test — but they are not expected to with the current, reverted
# design (see main.py's "TRIED AND REVERTED" comment).
# ---------------------------------------------------------------------------
print()
print("[F4 deferred] known-open false positives/negatives (documented "
      "trade-off, not silently dropped — see SP\\fix-p4c.md F4)")

_n1 = head + "\n\n" + beats_para(1700) + "\n\n*What do you do?*"
_n1_result = main.reply_is_degenerate(_n1)
print(f"  info N1 (beat paragraph + 18-char closing question): "
      f"reply_is_degenerate -> {_n1_result!r} (still a false positive; "
      f"deferred, see main.py 'TRIED AND REVERTED')")

_n4 = head + "\n\n" + beats_para(1700) + "\n\n" + prose_close[:250].rsplit(" ", 1)[0] + "."
_n4_result = main.reply_is_degenerate(_n4)
print(f"  info N4 (beat paragraph + 250-char prose close): "
      f"reply_is_degenerate -> {_n4_result!r} (still a false positive; "
      f"falls in the 150-300 char gap between the short-trailer path that "
      f"was reverted and the substantial/DEGENERATE_MIN_CHARS floor)")

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
# ---------------------------------------------------------------------------
print()
print("[sibling] pass-3 [9c] short-trailer-after-a-real-collapse cases "
      "must still be caught (these are what the reverted short-trailer "
      "exemption would have broken)")


check(runaway + "\n\nAlways yours.", True,
      "B: runaway + a short, terminated sign-off ('Always yours.') is "
      "still caught")
check(runaway + "\n\n---", True,
      "F: runaway + a bare '---' is still caught")


if FAILED:
    print(f"\n{len(FAILED)} check(s) FAILED:")
    for f in FAILED:
        print(f"  - {f}")
    sys.exit(1)
print("\nAll F4 (hostile pass 4) degeneracy checks passed.")
