"""Rule/box decoration: measuring it, and taking it off text bound for memory.

WHY THIS EXISTS. Measured 2026-09-04 over 976 stored replies: **277 of them
(28.4%) contain box-drawing characters**, median 438 per affected reply, worst
4,758 in a 16,256-character reply. `reply_is_degenerate` fires on 18 of the
277, and it is right not to fire on the rest — those replies are prose, they
are just heavily decorated, and the degeneracy rule exists to catch a
repetition LOOP rather than a style.

The part that is a defect rather than a taste is what happens next.
Decoration reaches MEMORY: 8 of 156 fact files carry box characters (3,119 of
them) and 3 of 14 summary files do (1,173), including the live summary of the
conversation in daily use. Injected memory is re-read by the model on every
single turn, so the compactor has been showing her her own decoration and
asking her to continue the conversation. Whatever the prompt says, the
strongest style signal in the window is the one the compactor put there.

`facts._reject_reason` already refuses a line with no alphanumeric character
in it — a pure `━━━━━━` rule — and its comment names U+2501 and U+2500
explicitly. The hole is the MIXED line: `━━━ Current Status ━━━` has
alphanumerics, is not a markdown heading, is not a fence, does not end on a
bracket, and so is stored verbatim. That is the recurring defect on this
project in its usual form — the pure case handled at one site, the mixed case
at none — and it is exactly the shape the report described: "a lot of boxing
characters, less so than words".

Kept dependency-free, and in its own module for the same reason envcfg.py is:
facts.py and summarizer.py both need it, summarizer imports almost nothing on
purpose, and a rule that lives in one of them and is copied into the other is
the defect this module is about.

NOTE ON SCOPE. Nothing here touches what is SENT to the user. Her replies are
forwarded byte for byte, decoration included; this is only about what the
compactor writes down and reads back. Reducing the decoration in the replies
themselves is a prompt change, and the prompt is the owner's.
"""

from __future__ import annotations

import re

# U+2500-U+257F box drawing and U+2580-U+259F block elements, plus the
# typographic rules and bullets models reach for when they build separators.
# Ranges rather than a hand-listed set: the corpus alone contained ten
# distinct characters from this block, and an enumeration would have to be
# extended every time a model picks a new one.
_RULE_CHARS = frozenset(
    [chr(c) for c in range(0x2500, 0x25A0)]
    + list("─━│┃═║▪▫◆◇○●▬※‾⎯⏤﹏＿")
)
# ASCII rules only count when they RUN: "---" is a separator, but a hyphen in
# "well-known" is not, and an em dash is ordinary prose punctuation.
_ASCII_RULE_CHARS = frozenset("-=_*~#")
_MIN_ASCII_RUN = 3

# The same run rule as rule_char_count, as a pattern, so measuring and
# stripping cannot disagree about what an ASCII rule is. The first draft had
# them disagree — the counter understood runs and the stripper did not, so
# `rule_char_count("-" * 30)` said 30 while `strip_rule_decoration("-" * 30)`
# returned all thirty characters unchanged. A module whose two functions
# describe different worlds is the fix-one-site-miss-the-sibling defect
# without even the excuse of distance.
_ASCII_RUN_RE = re.compile(
    "|".join(re.escape(ch) + "{" + str(_MIN_ASCII_RUN) + ",}"
             for ch in sorted(_ASCII_RULE_CHARS))
)

# v3.1.9.5 D2 (p18a F2): the 2026-09-04 corpus that calibrated _RULE_CHARS
# above is not the corpus that matters now. Measured on 5,895 replies from
# 2026-09-20: a symbol wall built from SEVEN distinct code points, six of
# them outside every set this module knows (only U+2501 — already in
# _RULE_CHARS — was covered). Widening the list is not the fix: the operator
# named the three characters in the OLD wall in the system prompt and the
# model moved to seven new ones. A hand-listed set only ever covers symbols
# already seen.
#
# So this is code-point agnostic: ANY character that is neither alphanumeric
# nor whitespace, in a run long enough to be a wall rather than ordinary
# repeated punctuation, is decoration — whatever its code point. Calibration
# (share of the 5,895 replies containing a run of at least this length):
#
#     run >=  6   33.0%
#     run >= 10   32.0%
#     run >= 20   28.9%
#     run >= 40   20.6%
#
# A third of her ordinary replies already contain a run of 6+ — for example
# an em-dash-built divider, or "??" doubled into "????" for emphasis — which
# is exactly why this COLLAPSES rather than rejects: reducing a run to 3
# keeps a genuine rule or emphasis run readable (an em-dash divider is still
# a divider; "????" reads the same as "???") and costs nothing when the run
# turns out to be decoration, while a run under the floor is never touched at
# all. 6 is chosen over a higher floor (10/20/40) because those leave short,
# real dividers ("──────", six characters) unstripped, which is the exact
# defect this rule exists to close; 6 also matches the run floor the module's
# proposed fix (v3.1.9.5 review, Q4) was calibrated against.
#
# Deliberately NOT applied to characters _RULE_CHARS or _ASCII_RULE_CHARS
# already recognise: those are removed outright by the passes below
# regardless of run length (a lone box character mid-sentence is still
# decoration), and collapsing them first would only add a redundant step.
# This reaches exactly the characters neither existing pass would have
# touched — the new, unlisted ones.
_GENERIC_RUN_MIN = 6
_GENERIC_COLLAPSE_TO = 3


def is_rule_char(ch: str) -> bool:
    return ch in _RULE_CHARS


def _is_unlisted_decor_char(ch: str) -> bool:
    """Neither alphanumeric nor whitespace, and not already covered by
    _RULE_CHARS or the ASCII rule set — i.e. exactly the kind of character a
    hand-listed set cannot anticipate. Numeric runs ("000000") are
    alphanumeric and excluded; whitespace is excluded so indentation inside a
    fenced code block is never touched."""
    return (
        ch not in _RULE_CHARS
        and ch not in _ASCII_RULE_CHARS
        and not ch.isalnum()
        and not ch.isspace()
    )


def _collapse_generic_runs(text: str) -> str:
    """Collapse a run of >= _GENERIC_RUN_MIN identical characters flagged by
    `_is_unlisted_decor_char` down to _GENERIC_COLLAPSE_TO copies. See the
    calibration comment above _GENERIC_RUN_MIN for why 6/3 and why collapse
    rather than remove."""
    if not text:
        return text
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if _is_unlisted_decor_char(ch):
            j = i + 1
            while j < n and text[j] == ch:
                j += 1
            run_len = j - i
            out.append(
                ch * (_GENERIC_COLLAPSE_TO if run_len >= _GENERIC_RUN_MIN else run_len)
            )
            i = j
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def rule_char_count(text: str) -> int:
    """Unicode rule characters, plus ASCII characters that appear in a run of
    at least three, plus any other non-alphanumeric non-whitespace character
    that appears in a run of at least `_GENERIC_RUN_MIN` (v3.1.9.5 D2). The
    run requirements are what keep ordinary hyphenation and emphasis out of
    the count. The third pass counts the FULL run — this function measures
    how decorated the text is, not how much of it `strip_rule_decoration`
    leaves behind; the two are allowed to differ, since the strip
    deliberately keeps a 3-character remainder for readability (see
    `_collapse_generic_runs`)."""
    if not text:
        return 0
    n = sum(1 for ch in text if ch in _RULE_CHARS)
    run_ch, run_len = "", 0
    for ch in text + "\0":
        if ch in _ASCII_RULE_CHARS and ch == run_ch:
            run_len += 1
            continue
        if run_len >= _MIN_ASCII_RUN:
            n += run_len
        run_ch, run_len = (ch, 1) if ch in _ASCII_RULE_CHARS else ("", 0)
    run_ch, run_len = "", 0
    for ch in text + "\0":
        if _is_unlisted_decor_char(ch) and ch == run_ch:
            run_len += 1
            continue
        if run_len >= _GENERIC_RUN_MIN:
            n += run_len
        run_ch, run_len = (ch, 1) if _is_unlisted_decor_char(ch) else ("", 0)
    return n


def rule_char_ratio(text: str) -> float:
    """Share of the text that is decoration. 0.0 for empty text — an absent
    reply is not a decorated one, and callers reading this as "how bad is it"
    must not get a division by zero instead of an answer."""
    if not text:
        return 0.0
    return rule_char_count(text) / len(text)


def strip_rule_decoration(text: str) -> str:
    """Remove rule/box decoration, keeping the words.

    Line by line, because that is how decoration is written: a line that is
    ONLY decoration disappears, and a line that is decoration wrapped around
    prose keeps the prose. `━━━ Current Status ━━━` becomes `Current Status`;
    `She likes gardening` is returned unchanged; `━━━━━━━━` becomes nothing.

    Whitespace inside a kept line is preserved apart from the edges, so a
    table row does not lose its columns and become one run-on sentence.
    """
    if not text:
        return text
    out: list[str] = []
    for line in text.split("\n"):
        # v3.1.9.5 D2: collapse runs of unlisted decoration FIRST, so a wall
        # built from code points this module has never seen is already down
        # to 3 characters by the time the curated passes below decide what
        # to do with it — see _collapse_generic_runs.
        line = _collapse_generic_runs(line)
        # ASCII runs first, and as a substitution rather than an edge trim:
        # "--- Status ---" needs both ends gone, and "### Heading" needs a
        # prefix gone, and both are the same rule.
        stripped = _ASCII_RUN_RE.sub(" ", line).strip()
        if not stripped:
            out.append("")
            continue
        # Drop the decoration from both ends, then check what survived. A line
        # that was ALL decoration collapses to "" and is dropped entirely
        # rather than left as a blank, so a block of six rules does not become
        # six blank lines in a summary.
        kept = stripped
        while kept and (kept[0] in _RULE_CHARS or kept[0].isspace()):
            kept = kept[1:]
        while kept and (kept[-1] in _RULE_CHARS or kept[-1].isspace()):
            kept = kept[:-1]
        # Interior decoration too: "Status ━━━ Active" is one line with a
        # separator in the middle of it, and the separator is not a word.
        kept = "".join(" " if ch in _RULE_CHARS else ch for ch in kept)
        kept = " ".join(kept.split())
        if kept:
            out.append(kept)
    # Collapse the runs of blank lines the drops leave behind, but keep single
    # paragraph breaks: a summary is prose and prose has paragraphs.
    cleaned: list[str] = []
    for line in out:
        if not line and cleaned and not cleaned[-1]:
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()
