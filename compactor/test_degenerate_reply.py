"""
Degenerate-reply detection (v3.1.2).

The thresholds in main were MEASURED against 504 real assistant replies from a
production backup, not chosen, so these tests are calibrated against that
corpus rather than against invented numbers:

    501 healthy    max decoration 37.9%   longest single-char run 146 (p99 75)
      3 degenerate min decoration 52.8%   shortest run 386

Both boundary cases below come from those extremes, and that is the point: a
detector exercised only on obvious cases tells you nothing about where it will
misfire on real content. The healthy maximum is the case that matters — if
this ever starts refusing to memorise ordinary replies, it is worse than the
loop it was built for, because the loop is visible and a silently unmemorised
conversation is not.

    python test_degenerate_reply.py
"""

import os
import sys

os.environ.setdefault("MODEL_REPO", "")

import main  # noqa: E402

RULE = "━"


def check(text, expect_degenerate, label):
    got = main.reply_is_degenerate(text)
    if bool(got) != expect_degenerate:
        print(f"FAIL {label}: expected degenerate={expect_degenerate}, got {got!r}")
        sys.exit(1)
    print(f"  ok   {label}" + (f"  [{got}]" if got else ""))


print("[1] the production incident, reproduced")
# The three real replies of 2026-08-29 ran 386, 425 and 569 characters of one
# repeated glyph and ended mid-run.
for run in (386, 425, 569):
    check("# Status\n\n```\n" + RULE * run + "\n", True,
          f"a {run}-char unbroken run is degenerate")

print()
print("[2] the healthy extremes from the same corpus must NOT trip it")
# The longest single-char run across 501 healthy replies was 146.
check("Here is a thought.\n\n" + RULE * 146 + "\n\nAnd the reply continues "
      "afterwards with ordinary prose for a good while longer.", False,
      "the longest run in 501 healthy replies (146) is allowed")
check(RULE * 70 + "\n" + ("Some prose. " * 12) + "\n" + RULE * 70, False,
      "two normal rules with prose between them")
check("A short answer.", False, "short prose")
check("", False, "empty is not degenerate, it is empty")

print()
print("[3] the fraction rule and its length floor")
check(RULE * 50, False, "a bare 50-char rule is under the length floor")
check(RULE * 260, True, "a bare 260-char rule trips the run rule")
# A loop that VARIES the glyph defeats a run-only check, so the fraction rule
# is the backstop. No single run here exceeds three characters.
check("━─═" * 200, True,
      "alternating decoration glyphs still trip the fraction rule")

print()
print("[4] it must not judge content it has no business judging")
# A run of 400 identical letters is a loop too — the 2026-08-29 incident
# happened to use U+2501, but the defect is REPETITION, not that particular
# glyph. Catching this is correct; my first version of this test asserted the
# opposite and was wrong.
check("x" * 400, True, "a 400-char run of an ordinary letter is also a loop")
# 200 identical letters is caught by the TOKEN rule at 120, not the character
# rule at 250 — and that is correct: it is a loop. The two rules are disjoint
# by content class (decoration -> character rule, word-like -> token rule), so
# the effective limit for an alphanumeric run is the lower of the two.
check("x" * 200, True, "200 identical letters is a loop under the token rule")
check("x" * 100, False, "100 identical letters is under both limits")
check("-" * 30 + "\n" + ("Real content describing something at length. " * 8),
      False, "a markdown horizontal rule followed by prose")
# Code is full of punctuation the decoration set contains. A reply that is
# mostly a code block must survive, or the assistant stops being able to
# remember anything technical it said.
check("Here is the fix:\n\n```python\n" + "x = a - b * c  # __init__\n" * 20 +
      "```\n\nThat should do it.", False, "a reply that is mostly code")

print()
print("[5] repeated TOKENS, the 2026-08-29 tail collapse")
# Long replies degenerated into training-data identifiers at the tail:
#     _batch_handler_shared _batch_handler_shared _batch_handler_shared ...
#     config_config_config_config_config ...
# Neither is one repeated CHARACTER nor decoration-heavy, so the v3.1.2 rules
# saw nothing — 3 of 48 caught. Threshold measured against 512 real replies:
# longest repeated-token run sits at p90=56, p97=72, p98=80, then jumps to
# p99=384. 120 is 1.5x over the normal ceiling and 3x under the pathological
# floor. Against the full corpus the rule now scores 9 caught / 0 missed /
# 0 false positives.
check("Here is the answer." + " _batch_handler_shared" * 12, True,
      "a repeated identifier trips the token rule")
check("config_" * 40, True, "underscore-joined repetition trips it")

# The longest repeated-token run in 464 healthy replies was 56 characters.
check("Really " + "yes " * 12 + "that is what I meant, and here is more prose "
      "to make it a normal length reply.", False,
      "emphasis repetition well under the measured healthy ceiling")
check("The value is 3. The value is 3. The value is 3.", False,
      "a phrase repeated three times is writing, not a loop")

# LONGEST match, not first. A brief repetition early must not mask a runaway
# later — that cost 4 of 9 detections before it was fixed.
check("aaa aaa aaa aaa. Then ordinary prose for a while. " + "stuck " * 40,
      True, "a late runaway is caught even after an early short repetition")

print()
print("All degenerate-reply tests passed.")
