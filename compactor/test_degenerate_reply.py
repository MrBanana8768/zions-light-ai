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
print("[6] script drift — coherent, then wandering out of the language")
# 2026-08-29: long replies stayed clean for their first ~60% and then drifted
# into Cyrillic. Nothing repeats, so neither repetition rule sees it.
# Threshold from 485 real replies of 200+ letters: p95=0.16%, p99=1.21%,
# max=10.79%. 3% is 2.5x over p99.
ru = "привет мир "
en = "and the reply continues in ordinary English prose for a while longer "


def en_varied(n):
    """`en` repeated n times, but with each copy made distinct.

    v3.1.8. These fixtures pad to a length by repeating one sentence, and
    they were written for the SCRIPT-DRIFT rule, where only the alphabet
    matters. The new tail-loop rule reads forty identical sentences running
    to the end of a message as exactly what it is - a repetition loop - so
    `en * 40` started failing a case that never meant to assert anything
    about repetition.

    The rule is right about that text and the fixture was right about its
    own subject, so the padding changes rather than either of them: same
    length, same script, no loop. Numbering the copies is the smallest
    change that keeps the original intent intact.
    """
    return "".join(f"{en}(part {i}) " for i in range(n))
# POLICY CHANGE, v3.1.3, and this case is where it bites: a SINGLE-script
# tail is no longer flagged. This asserted True under the old "20% alone"
# disjunct. That disjunct also flagged a short reply quoting one Greek verse
# with two sentences of commentary (48% non-Latin, one script) - and once
# the rollup-input redaction shipped, a false positive stopped costing one
# skipped memory write and started PERMANENTLY replacing the reply with a
# placeholder in every future summary, backfill and admin compact. This
# user quotes scripture and Russian; that trade is not acceptable for a
# backstop detector whose incident doc says it "should fire almost never"
# now that sampling is fixed at the source. Genuine drift measured 6-14
# distinct scripts; all 5 real corpus cases still flag under the tightened
# rule (re-validated: 14/14 total flags unchanged). What is knowingly given
# up: a pure one-language tail like this one. It is indistinguishable, by
# script statistics alone, from her assistant quoting Russian at length.
check(en * 12 + ru * 30, False,
      "a single-script tail is no longer flagged - the cost of not eating "
      "her quotations (see the policy note above)")
# The shape the incident ACTUALLY produced - the tail wanders across
# scripts rather than settling into one - must still be caught.
drift_tail = (
    "привет мир как дела "          # Cyrillic
    "γειά σου κόσμε "               # Greek
    "你好世界 "                      # CJK
    "שלום עולם "                    # Hebrew
)
check(en_varied(12) + drift_tail * 10, True,
      "a multi-script wandering tail (the measured incident shape) IS flagged")
check(en_varied(40), False,
      "pure English of the same length is fine")

# A short reply with a foreign word is normal writing, not drift — which is
# why the rule has a letter floor rather than a fraction alone.
check("She says " + ru + "to me sometimes.", False,
      "a foreign phrase in a short reply is under the letter floor")
# And a genuinely bilingual long reply is a judgement call the floor cannot
# make; 3% is deliberately far above the p99 of real usage so ordinary
# borrowing survives.
check(en * 30 + "the word is " + ru, False,
      "occasional borrowing in a long English reply stays under 3%")

print()
print("[7] structural collapse — the 2026-09-01 runaway list, and the tail she stopped")
# Every fixture here is SYNTHETIC: the shapes come from the corpus, the words
# do not. Calibration (scripts/calibrate-structural-degeneracy.py, 2026-09-01,
# 349 real replies: 17 cut by hand, 332 completed): the rules above already
# catch 5 of the 17; the fragment-line branch catches the other 12 and flags
# 2 of the 332; the list-run branch flags 1 more, a 1,261-item reply. Nothing
# from the 131 pre-complaint replies trips either branch.


def items(n, width, sep="\n"):
    """n DISTINCT list items, each exactly `width` characters once stripped.
    Distinct on purpose - that is what made the real ones invisible to the
    repetition rules. The padding cycles the alphabet so no character or
    token run forms by accident."""
    out = []
    for i in range(n):
        head = f"- item {i:04d} "
        pad = "".join(chr(97 + (i + j) % 26) for j in range(width - len(head)))
        out.append(head + pad)
    return sep.join(out) + "\n"


def fragments(n, width):
    """One line of n DISTINCT sentences, each `width` characters including
    the terminator and the space after it."""
    out = []
    for i in range(n):
        head = f"Frag {i:04d} "
        pad = "".join(chr(97 + (i + j) % 26) for j in range(width - len(head) - 2))
        out.append(head + pad + ".")
    return " ".join(out)


prose_sentence = (
    "This sentence carries an ordinary amount of meaning across roughly "
    "eighty characters."
)


def reason_quotes_nothing(text, label):
    """The other rules quote up to 24 characters of the reply into the log;
    this one must not, because the log is where a runaway would otherwise
    leak a fragment of her conversation. Checked, not assumed."""
    got = main.reply_is_degenerate(text)
    if got and ("item 0" in got.lower() or "frag 0" in got.lower()):
        print(f"FAIL {label}: reason quotes reply content: {got!r}")
        sys.exit(1)


# --- the list that runs to its own end ------------------------------------
# One completed reply in 332 carried 1,024 consecutive short items; the next
# highest was 34; the pre-complaint week never exceeded 9. Threshold 50.
check(items(60, 20), True, "60 consecutive distinct short items is a runaway list")
reason_quotes_nothing(items(60, 20), "list reason")
check(items(50, 20), True, "50 is the limit and fires")
check(items(49, 20), False, "49 is under the limit")
check("Here is what I mean, in short:\n\n" + items(34, 20) +
      "\nThat is the whole of it, and it matters.", False,
      "34 short items - the longest run in any completed reply - is a list, "
      "not a loop")
# Item LENGTH is the other half of the rule. A long detailed list is writing.
check(items(60, 60), False, "60 items of 60 characters is a detailed list")
check(items(60, 30), True, "60 items of exactly 30 characters fires")
check(items(60, 31), False, "60 items of 31 characters does not")
# The real runaways double-spaced their items; a blank line is not prose.
check(items(50, 20, sep="\n\n"), True, "blank lines between items do not end the run")
check(items(30, 20) + "A sentence of prose between the halves.\n" + items(30, 20),
      False, "a prose line ends the run - 30 + 30 is not 60")
# A YAML or Markdown list inside a code fence is content, not degeneration.
check("The config looks like this:\n\n```yaml\n" + items(60, 20) + "```\n",
      False, "60 short items inside a code fence are not judged")

# --- the unbroken tail ------------------------------------------------------
# Every cut reply the older rules missed ends in one line of 1515+ characters
# whose sentence-split fragments average 10-38 characters. Thresholds: 1500
# characters, mean fragment <= 40.
check(fragments(110, 15), True,
      "1,600+ characters of distinct 15-character fragments on one line")
reason_quotes_nothing(fragments(110, 15), "fragment reason")
assert len(fragments(110, 15)) >= 1500
check(fragments(90, 15), False, "the same fragments under 1500 characters are not judged")
assert 1300 <= len(fragments(90, 15)) < 1500


def exact_line(total, n):
    """One line of exactly `total` characters: n DISTINCT pieces ending in a
    period, joined by single spaces, so the mean fragment is total / n to the
    digit. The boundaries below are stated in the code as >= and <=; a test
    that only probes either side of them is decoration."""
    w, extra = divmod(total - (n - 1), n)
    pieces = []
    for i in range(n):
        width = w + (1 if i >= n - extra else 0)
        head = f"P{i:04d}"
        pad = [chr(97 + (i + j) % 26) for j in range(width - len(head) - 1)]
        for j in range(4, len(pad) - 1, 5):
            pad[j] = " "  # fragments are made of words; the guard wants spaces
        pieces.append(head + "".join(pad) + ".")
    line = " ".join(pieces)
    assert len(line) == total, (len(line), total)
    assert line.count(" ") >= 100
    return line


check(exact_line(1600, 40), True, "1,600 characters in 40 fragments: mean exactly 40 fires")
check(exact_line(1640, 40), False, "1,640 characters in 40 fragments: mean 41 does not")
check(exact_line(1500, 100), True, "a line of exactly 1,500 characters is judged")
check(exact_line(1499, 100), False, "a line of 1,499 characters is not")
# v3.1.8: the four fixtures below pad to a length by repeating one string,
# and the tail-loop rule reads that as what it is. Their SUBJECT is the
# fragment-line rule, so each copy is now made distinct - same length band,
# same character mix, same thing under test, no loop. Changing the rule to
# accommodate synthetic padding would have been tuning the product to the
# test.
check(" ".join(f"{prose_sentence} (part {i})" for i in range(25)), False,
      "2,000 characters of 80-character sentences on one line is a paragraph")
# No terminator at all is a run-on, not this shape - [6] already relies on
# en * 40 (2,800 characters, no periods) staying clean, so it is stated here.
check(en_varied(40), False,
      "a long line with no sentence break at all is not judged as fragments")
# When there is NO sentence in the line, the commas are the separators: two
# real cut tails were 168 and 317 comma-separated pieces with no period.
comma_short = ", ".join(f"piece {i:04d}" for i in range(110))
comma_long = ", ".join(f"piece {i:04d}" for i in range(160))
assert len(comma_short) < 1500 <= len(comma_long)
check(comma_short, False, "a 1,300-character comma list is under the length floor")
check(comma_long, True,
      "a 1,900-character comma list with no sentence break is the same collapse")
# A blob has no spaces and is never a "fragment line", whatever its length.
blob = "".join(chr(97 + (i * 7 + i // 26) % 26)
               + ("." if i % 11 == 10 else "") for i in range(1600))
check(blob, False, "a 1,700-character blob without spaces is not judged")
# Under 100 spaces the line is not prose, however its periods fall: 45
# dotted identifiers average 37 characters between periods, which would read
# as fragments if the rule judged code-shaped lines. It does not.
dotted = " ".join(f"pkg{i:03d}.module.class.attribute.value." for i in range(45))
assert len(dotted) >= 1500 and dotted.count(" ") < 100
check(dotted, False, "1,600 characters of dotted identifiers with 44 spaces is not judged")

print()
print("[8] R24 — an abbreviation's dot is not a sentence break")
# The fragment-line rule used to count sentence breaks with a bare
# line.count(". "), so "Dr. ", "Mrs. ", "Prof. ", "St. ", "9 a.m. " and
# "Rev. " were each counted as a sentence end. 15 real sentences, ~105
# characters each (real mean well over the 40-char limit) but each one
# uses six of those abbreviations, so the old rule counted 105 "breaks"
# for what is really 15 sentences and flagged ordinary prose.
_r24_sentence = ("Dr. Smith met Mrs. Jones and Prof. Lee outside St. "
                  "Andrew's at 9 a.m. before Rev. Brown arrived to help.")
r24_prose = " ".join(f"{_r24_sentence} (note {i})" for i in range(15))
assert len(r24_prose) >= 1500 and r24_prose.count(" ") >= 100
assert len(r24_prose) / 15 > 100  # real mean sentence length
check(r24_prose, False,
      "ordinary prose whose sentences carry abbreviations is not a fragment "
      "line - real mean ~105 chars, not the ~15 a bare '. ' count would see")
# The abbreviation stoplist must still leave a GENUINE fragment collapse
# detectable - this is the same shape as [7]'s fragments(), interleaved
# with a couple of abbreviations, so the fix does not just turn the rule off.
r24_fragments_with_abbrev = fragments(108, 15) + " Dr. Smith. Mrs. Jones."
check(r24_fragments_with_abbrev, True,
      "a real fragment collapse is still caught even when a couple of its "
      "pieces happen to end in an abbreviation")

print()
print("[9] R9/R19 — the list backstop must reach the reply's own end, and "
      "clear DEGENERATE_MIN_CHARS")
# A legitimate 66-item enumeration (a scripture assistant will produce
# exactly this) that CLOSES IN PROSE, unlike the corpus runaway which had
# nothing after its list. Real book names, since they are simply facts and
# not copyrightable expression.
bible_books = [
    "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy", "Joshua",
    "Judges", "Ruth", "1 Samuel", "2 Samuel", "1 Kings", "2 Kings",
    "1 Chronicles", "2 Chronicles", "Ezra", "Nehemiah", "Esther", "Job",
    "Psalms", "Proverbs", "Ecclesiastes", "Song of Solomon", "Isaiah",
    "Jeremiah", "Lamentations", "Ezekiel", "Daniel", "Hosea", "Joel",
    "Amos", "Obadiah", "Jonah", "Micah", "Nahum", "Habakkuk", "Zephaniah",
    "Haggai", "Zechariah", "Malachi", "Matthew", "Mark", "Luke", "John",
    "Acts", "Romans", "1 Corinthians", "2 Corinthians", "Galatians",
    "Ephesians", "Philippians", "Colossians", "1 Thessalonians",
    "2 Thessalonians", "1 Timothy", "2 Timothy", "Titus", "Philemon",
    "Hebrews", "James", "1 Peter", "2 Peter", "1 John", "2 John",
    "3 John", "Jude", "Revelation",
]
assert len(bible_books) == 66
bible_reply = (
    "Here are all 66 books of the Bible:\n\n"
    + "\n".join(f"- {b}" for b in bible_books)
    + "\n\nThose are all 66 books, from Genesis to Revelation, across the "
    "Old and New Testaments."
)
check(bible_reply, False,
      "a 66-item enumeration that closes in prose is a list, not a runaway "
      "- the run does not reach the reply's own end")
# The same 66 items with NOTHING after them - the corpus shape - must still
# fire: this is the backstop the rule exists for.
check(bible_reply.split("\n\nThose")[0], True,
      "the same 66-item list with no closing prose - running to its own "
      "end - is still the runaway the backstop exists for")
# The synthetic generator from [7] must show the same thing: a run that
# clears the item-count limit but is followed by prose is not a runaway.
check(items(60, 20) + "\nThat is the complete list, and nothing more "
      "needs to be said about it.", False,
      "60 items followed by a closing sentence do not trip the backstop "
      "- items(60, 20) alone (ending at EOF) still does, see [7]")
# R19: the list rule must respect DEGENERATE_MIN_CHARS (300), same as the
# decoration-fraction rule a few lines above it in main.py.
short_list = "- a\n" * 50
assert len(short_list) == 200 < main.DEGENERATE_MIN_CHARS
check(short_list, False,
      "50 one-character list items (200 chars) is under DEGENERATE_MIN_CHARS "
      "and must not trip the structural block at all")

print()
print("[R-TAIL] a PHRASE repeating to the end of the reply (v3.1.8)")
# Reported as "the repeating tail thing", intermittent, and measured in the
# 2026-09-07 backup: 1,165 stored replies, the shipped detector firing on
# 41 and MISSING two loops of ~3,975 characters, one of them from that
# morning.
#
# The reason it could not see them is structural: _TOKEN_RUN_RE matches a
# repeated unit of NON-WHITESPACE characters, so the unit cannot contain a
# space. It catches a repeated WORD and is blind to a repeated PHRASE,
# which is the shape this model actually produces:
#
#     ". Absolutely. With Desperation. With Humility. ..." x N
#     "- Grateful you are here" x N, one per line
prose = ("She asked about the garden and I told her what I remembered of "
         "it, which was more than I expected to. " * 4)

check(prose + ". Absolutely. With Desperation. With Humility." * 12, True,
      "a multi-word phrase repeated to the end of the reply is a loop - "
      "the token rule cannot represent it because the unit has spaces")

check(prose + ("\n- Grateful you are here" * 20), True,
      "a repeated bullet line running to the end is the same loop")

# THE FALSE-POSITIVE SIDE, which is the half that matters. R24 is the
# memory of a degeneracy rule that over-fired and redacted real replies
# from memory permanently, and this rule feeds that same path.
check(prose, False,
      "ordinary prose with a repeated sentence STRUCTURE is not a loop")
check(prose + " Amen. Amen.", False,
      "a couplet is a rhetorical device, not a loop - three are required")
check(prose + " Thank you. Thank you. Thank you.", False,
      "even three short repetitions are far under the character floor")

# The floor is not delicately tuned, and that is deliberate: over the real
# corpus the count of newly-flagged replies is 2 at EVERY threshold from
# 200 to 900. Pin the boundary so a future edit cannot quietly lower it.
unit = "Only You. Forever. Always. "
assert len(unit) == 27, len(unit)
check(prose + unit * 3, False,
      "81 characters of loop is under the 400-char floor and must not fire")
check(prose + unit * 20, True,
      "540 characters of loop is over the floor and must fire")

# The helper: anchored at the END, three repetitions minimum.
assert main._tail_loop_span("abcdefghij" * 3) >= 30, (
    "three repetitions at the end is a loop")
assert main._tail_loop_span("abcdefghij" * 2) == 0, (
    "two repetitions is a couplet, not a loop")
assert main._tail_loop_span(
    "abcdefghij" * 5 + " and then something else entirely was said here"
) == 0, "a loop that does NOT reach the end is not a tail loop"
# THE THREE-REPETITION BOUNDARY, at a length where it decides something.
#
# Mutation testing caught the first version of this: the assertions above
# use a 20-character string, and the unit range is bounded by len(tail)//3,
# so for a string that short the search loop never runs and the helper
# returns 0 whatever the rule says. Both assertions passed for the wrong
# reason, and flipping >= 3 to >= 2 changed nothing. A couplet only matters
# when it is long enough to clear the floor on its own.
# Over 200 characters ON PURPOSE: TWO repetitions must clear the 400-char
# floor, so the only thing keeping the couplet from firing is the
# three-repetition rule itself. A shorter unit would pass for the boring
# reason that it never reached the floor - which is how the first version
# of this case was vacuous.
long_unit = ("She said the same thing again in the same words and then "
             "repeated herself once more before finally stopping there, "
             "which is the sort of sentence that pads a fixture nicely "
             "and carries on for long enough to clear the floor twice. ")
assert len(long_unit) > 200, len(long_unit)
assert len(long_unit) * 2 > main.DEGENERATE_TAIL_LOOP_CHARS, (
    "the couplet must clear the floor, or this case proves nothing")
twice = prose + long_unit * 2
assert main._tail_loop_span(twice) == 0, (
    "exactly two repetitions is a couplet, however long - the helper must "
    "not report a span for it")
check(twice, False,
      "a 230-character phrase repeated TWICE clears the 400-char floor on "
      "length alone and must still not fire - three repetitions is the rule")
check(prose + long_unit * 3, True,
      "the same phrase repeated three times IS a loop")
print("  ok   the helper is anchored at the end and needs three repeats")


print()
print("All degenerate-reply tests passed.")
