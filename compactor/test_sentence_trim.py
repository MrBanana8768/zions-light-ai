"""
trim_to_last_sentence (v3.1.4) — the primitive the memory-tail gate trims a
cut reply with before deciding whether to remember it.

What this pins, and which mutation each case exists to kill:

    returns the input unchanged          -> [1] the mid-word cut
    appends a marker                     -> [2] the prefix property
    treats "\\n" as a boundary            -> [3] unterminated bullets
    drops the whitespace-after clause    -> [4] decimals and versions
    drops the stoplist / initial rule    -> [5] abbreviations, [6] initials
    treats an ellipsis as an end         -> [7]
    ignores fences                       -> [8]
    ignores closing marks                -> [9]
    takes the FIRST boundary, not last   -> [11] longest prefix
    an unbounded pattern                 -> [13] cost, measured

Every fixture below is synthetic. The repo is public and nothing from a real
conversation may appear here.

    python test_sentence_trim.py
"""

import os
import sys
import time

os.environ.setdefault("MODEL_REPO", "")

import main  # noqa: E402

trim = main.trim_to_last_sentence


def assert_eq(actual, expected, label):
    if actual != expected:
        print(f"FAIL {label}: expected {expected!r}, got {actual!r}")
        sys.exit(1)
    print(f"  ok   {label}")


def assert_true(cond, label):
    if not cond:
        print(f"FAIL {label}")
        sys.exit(1)
    print(f"  ok   {label}")


print("[1] a mid-word cut loses the fragment and keeps the sentence before it")
assert_eq(trim("First sentence. Second sen"), "First sentence.",
          "cut mid-word: the finished sentence survives, the fragment does not")
assert_eq(trim("Only a fragment with no end"), "",
          "no boundary anywhere -> empty, not the fragment")
assert_eq(trim(""), "", "empty in, empty out")
assert_eq(trim("Done."), "Done.", "a text that already ends on a boundary is unchanged")
assert_eq(trim("Done.\n\n"), "Done.",
          "trailing whitespace after the last boundary is not part of the sentence")

print()
print("[2] the result is a plain prefix — NO marker, ever")
# The docstring explains why this trimmer is the one that must not mark its
# cut: the text goes into the store, and a marker would be extracted as a
# fact, embedded and summarized. This property is what a well-meaning
# "add a marker for consistency" change breaks.
for text in (
    "First sentence. Second sen",
    "A. B? C! D",
    "Here is prose.\n- Always\n- Forever\n- No matter wh",
    "She said \"hello.\" Then she le",
    "Only a fragment",
):
    out = trim(text)
    assert_true(text.startswith(out), f"prefix property holds for {text[:20]!r}")
    assert_true(len(out) <= len(text), "never longer than the input")

print()
print("[3] newlines are NOT boundaries: an unterminated bullet run is discarded")
# This is the corrected design decision (plan §1.1). The runaway list IS the
# thing being discarded; `- eg` cut mid-word is indistinguishable from
# `- eggs`, and punctuation is the only positive evidence a unit finished.
assert_eq(trim("Here is prose.\n- Always\n- Forever\n- No matter wh"),
          "Here is prose.",
          "an unterminated bullet run trims back to the prose above it")
assert_eq(trim("line one\nline two\nline thr"), "",
          "lines without punctuation are not sentences")
assert_eq(trim("Intro.\n- One thing.\n- Two thin"), "Intro.\n- One thing.",
          "a TERMINATED bullet is a sentence and is kept")
assert_eq(trim("## Title\nSome text. More te"), "## Title\nSome text.",
          "a heading line above prose rides along inside the kept prefix")

print()
print("[4] decimals and version numbers are not boundaries — no special case needed")
assert_eq(trim("Pi is 3.14 and v3.1.6 shipped. Then it br"),
          "Pi is 3.14 and v3.1.6 shipped.",
          "3.14 and v3.1.6 pass through; the real terminator is found")
assert_eq(trim("About 1.500 units were"), "", "1.500 is not a boundary")
assert_eq(trim("It cost 3.50."), "It cost 3.50.",
          "a number's final period followed by end-of-text IS a boundary")
assert_eq(trim("end.Next"), "", "no whitespace after the terminator -> not a boundary")
assert_eq(trim("Version 2.0alpha is out"), "", "2.0alpha is not a boundary")

print()
print("[5] the abbreviation stoplist")
assert_eq(trim("Ask Dr. Smith about it tomor"), "",
          "'Dr.' is not a sentence end")
assert_eq(trim("Ask Dr. Smith. Then go ho"), "Ask Dr. Smith.",
          "...but the sentence after 'Dr.' ends normally")
assert_eq(trim("Bring fruit, e.g. apples and"), "", "'e.g.' is rejected")
assert_eq(trim("It is in Gen. 1:1 and Rev. 21:4 but"), "",
          "scripture book abbreviations are rejected")
assert_eq(trim("Go to St. Louis and"), "", "'St.' is rejected as a whole word")
assert_eq(trim("It was the 1st. Then we"), "It was the 1st.",
          "'1st.' is NOT the abbreviation 'st.' — '1' precedes it, so it ends its sentence")
assert_eq(trim("Call os.path.join. Then"), "Call os.path.join.",
          "a dotted identifier longer than any entry is accepted")

print()
print("[6] single initials")
assert_eq(trim("Written by J. R. R. Tolkien and oth"), "",
          "'J. R. R.' contributes no boundary")
assert_eq(trim("Written by J. R. R. Tolkien. And oth"),
          "Written by J. R. R. Tolkien.",
          "...the sentence after the initials ends normally")
assert_eq(trim("So am I. Then we"), "",
          "'I.' is treated as an initial too — the deliberate cheaper error")
assert_eq(trim("Take plan b. Then"), "Take plan b.",
          "a lowercase single letter is a word, not an initial")

print()
print("[7] ellipsis is a pause, not an end")
assert_eq(trim("Well... I suppose"), "", "'...' is not a boundary")
assert_eq(trim("Well... I suppose so. And th"), "Well... I suppose so.",
          "...the sentence containing it ends normally")
assert_eq(trim("Wait… and then"), "", "U+2026 is not a terminator")
assert_eq(trim("Really?! Yes it"), "Really?!", "'?!' is a boundary")
assert_eq(trim("Go!!! Now we"), "Go!!!", "'!!!' is a boundary")

print()
print("[8] fences")
assert_eq(trim("Here it is.\n```python\nx = 1.\nprint(x)"), "Here it is.",
          "a '.' inside an OPEN fence is not a boundary; the cut lands before the opener")
assert_eq(trim("Look.\n```\nfoo.\n```\nDone. And mo"),
          "Look.\n```\nfoo.\n```\nDone.",
          "a fence that closes again is fine; the cut can land after it")
assert_eq(trim("```\ncode.\n```\n"), "", "a closed fence with no prose has no boundary")

print()
print("[9] closing marks ride along with the terminator")
assert_eq(trim('She said "hello." Then she le'), 'She said "hello."',
          "a closing double quote")
assert_eq(trim("(See above.) Next up"), "(See above.)", "a closing paren")
assert_eq(trim("**Bold.** Then"), "**Bold.**", "markdown emphasis closers")
assert_eq(trim("He said “go.” And"), "He said “go.”",
          "a curly closing quote")

print()
print("[10] numbered-list markers are not sentences")
assert_eq(trim("Steps:\n1. Do this\n2. Do th"), "", "'1.' at line start is a marker")
assert_eq(trim("Steps:\n1. Do this.\n2. Do th"), "Steps:\n1. Do this.",
          "a terminated numbered item is kept")
assert_eq(trim("Chapter 3. Then it"), "Chapter 3.",
          "a number NOT at line start ends its sentence")
assert_eq(trim("3. Fragment"), "", "a marker at the very start of the text")

print()
print("[11] the LONGEST prefix, not the first boundary")
body = " ".join(f"Sentence number {i}." for i in range(1, 41))
assert_eq(trim(body + " And then a fragm"), body,
          "forty sentences plus a fragment keeps all forty")
assert_eq(trim(body), body, "forty sentences alone are unchanged")

print()
print("[12] fullwidth terminators need no trailing whitespace")
assert_eq(trim("今日は良い天気です。明日は"), "今日は良い天気です。",
          "'。' is a boundary without a following space")
assert_eq(trim("本当ですか？そう"), "本当ですか？", "'？' likewise")

print()
print("[13] cost — measured, because this runs on the event loop")
prose = ("It was a long day and the work went well, all things considered. "
         * 460)[:30000]
t0 = time.perf_counter()
out = trim(prose)
prose_ms = (time.perf_counter() - t0) * 1000
assert_true(out and prose.startswith(out), "30k of prose trims to a prefix")
pathological = ". " * 15000  # a candidate every other character
t0 = time.perf_counter()
trim(pathological)
patho_ms = (time.perf_counter() - t0) * 1000
print(f"       30,000 chars of prose: {prose_ms:.1f} ms; "
      f"30,000 chars of '. ': {patho_ms:.1f} ms")
# A generous bound: the point is to catch a quadratic regression (seconds),
# not to flake on a slow box.
assert_true(prose_ms < 1000 and patho_ms < 1000,
            "both well under a second (a quadratic pattern would be seconds)")
fence_heavy = ("Text.\n```\ncode.\n```\n" * 1500)[:30000]
t0 = time.perf_counter()
trim(fence_heavy)
fence_ms = (time.perf_counter() - t0) * 1000
print(f"       30,000 chars with 3,000 fence toggles: {fence_ms:.1f} ms")
assert_true(fence_ms < 1000, "fence bookkeeping is a bisect, not a rescan")

print()
print("[14] R25 — a dash introduces a word exactly as a space does")
# _WORD_LEAD_CHARS used to have no dash of any kind in it, so the
# abbreviation stoplist and the single-initial rule were both skipped right
# after an em dash, en dash or hyphen - the opposite of every other lead
# character tested above ([5], [6]), which all led with a space or
# start-of-text. That made the set too PERMISSIVE-never-narrow in those
# tests; these lead with a dash instead, so a mutation that makes
# `whole_word` always True is caught elsewhere, and one that makes it
# always False (or that never recognises a dash as a lead character) is
# caught here.
assert_eq(trim("Something ended properly. Then—i.e. a fragment that never fin"),
          "Something ended properly.",
          "an em dash before 'i.e.' does not defeat the abbreviation "
          "stoplist - the trim lands on the clean sentence before it, not "
          "on the abbreviation")
assert_eq(trim("Written by the author—J. R. R. Tolkien and oth"), "",
          "an em dash before a single initial does not defeat the initial "
          "rule either - 'J.' is still not a boundary")
assert_eq(trim("Written by the author—J. R. R. Tolkien. And oth"),
          "Written by the author—J. R. R. Tolkien.",
          "...the sentence after the initials still ends normally")
assert_eq(trim("Ask the doctor-Dr. Smith-about it tomor"), "",
          "a hyphen before 'Dr.' does not defeat the stoplist either")
assert_eq(trim("Read Rev-Rev. 21:4 says so, and it clo"), "",
          "an en dash reads the same way")

print()
print("All sentence-trim tests passed.")
