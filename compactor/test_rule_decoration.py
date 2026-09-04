"""
Box/rule decoration must not reach memory (v3.1.8).

THE REPORT: "a lot of boxing characters again, less so than words". Measured
over the 976 stored replies in the 2026-09-04 backup, that is exactly right:

    277 of 976 replies (28.4%) contain box-drawing characters
    median 438 per affected reply, p90 1,642, worst 4,758 in 16,256 chars
    reply_is_degenerate() fires on 18 of the 277

And it is right not to fire on the other 259. Those replies are prose with
heavy decoration; the degeneracy rule exists to catch a repetition LOOP, and
widening it to catch a style would start redacting real replies — the R24
mistake, which cost this project a working fix and a day.

THE DEFECT IS WHERE THE DECORATION GOES. It reached MEMORY: 8 of 156 fact
files carried box characters (3,119 of them), 3 of 14 summary files (1,173),
including the live summary of the conversation in daily use. Injected memory
is re-read on every turn, so the compactor was showing the model its own
decoration and asking it to keep going. Whatever the system prompt says, that
was the strongest style signal in the window.

`facts._reject_reason` already refused a line with NO alphanumeric character,
and its comment names U+2501 and U+2500. The hole is the MIXED line —
`---- Current Status ----` has alphanumerics, is not a heading, is not a
fence, does not end on a bracket — so it stored verbatim. The pure case
handled at one site, the mixed case at none.

Mutations this file exists to kill:

    strip_rule_decoration returns text unchanged      [1], [2]
    the interior-separator pass is dropped            [2c]
    _reject_reason loses the decoration-only check    [3]
    _parse_extraction_output stores the raw line      [4]
    the summarizer wrapper is bypassed                [5]
    ASCII runs counted without the run requirement    [6] hyphenated words

    python test_rule_decoration.py
"""

import os
import sys
import tempfile

os.environ.setdefault("MODEL_REPO", "test-model")
os.environ.setdefault("VLLM_URL", "http://stub:8000")
os.environ["COMPACTOR_RAG_ENABLED"] = "false"
os.environ["COMPACTOR_STORAGE_ROOT"] = tempfile.mkdtemp(prefix="compactor-test-rules-")

import facts  # noqa: E402
import summarizer  # noqa: E402
import textclean  # noqa: E402

FAILED: list[str] = []


def check(cond, label):
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"FAIL {label}")
        FAILED.append(label)


HEAVY = "━"   # BOX DRAWINGS HEAVY HORIZONTAL - 85,696 in the corpus
LIGHT = "─"
DOUBLE = "═"
TEE = "├"     # LIGHT VERTICAL AND RIGHT - 19,578 in the corpus

print("[1] a line that is ONLY decoration keeps nothing")
check(textclean.strip_rule_decoration(HEAVY * 40) == "", "a heavy rule")
check(textclean.strip_rule_decoration("-" * 30) == "", "an ASCII rule")
check(textclean.strip_rule_decoration(DOUBLE * 20) == "", "a double rule")

print("[2] a decorated line keeps its WORDS")
check(
    textclean.strip_rule_decoration(f"{HEAVY * 3} Current Status {HEAVY * 3}")
    == "Current Status",
    "decoration wrapped around prose is removed, prose survives",
)
check(
    textclean.strip_rule_decoration(f"{TEE}{LIGHT * 2} She prefers tea")
    == "She prefers tea",
    "a tree-branch prefix is decoration too",
)

print("[2b] ordinary prose is returned UNCHANGED")
for prose in (
    "She prefers tea to coffee.",
    "The well-known author lives in Portland.",   # a hyphen is not a rule
    "It cost 3-4 dollars.",
    "Use the em dash — like this — in ordinary prose.",
):
    check(textclean.strip_rule_decoration(prose) == prose,
          f"unchanged: {prose[:38]!r}")

print("[2c] an INTERIOR separator is decoration as well")
check(
    textclean.strip_rule_decoration(f"Status {HEAVY * 3} Active")
    == "Status Active",
    "a separator in the middle of a line is not a word",
)

print("[3] facts refuses a line that is decoration all the way down")
# These two pass on the PRE-EXISTING "no alphanumeric content" rule, and that
# is worth saying out loud. A "rule decoration only" check was added to
# _reject_reason during this fix and mutation testing showed it dead — removing
# it changed nothing, because strip_rule_decoration only removes rule
# characters, so a line with any alphanumeric always survives it. These
# assertions were "covering" the older rule the whole time. They stay, because
# the behaviour is still required; the coverage claim is what was wrong.
check(facts.is_storable_fact(HEAVY * 20) is False, "a pure rule is not a fact")
check(facts.is_storable_fact(f"{HEAVY * 3} {LIGHT * 3}") is False,
      "mixed rule characters are still only rules")
check(facts.is_storable_fact(f"{HEAVY * 3} She has two cats {HEAVY * 3}") is True,
      "but a DECORATED fact is a fact - the words are what matter, and the "
      "cleaning happens where it is STORED, not by refusing it here")

print("[4] and what gets STORED is the prose, not the border")
parsed = facts._parse_extraction_output(
    f"- {HEAVY * 3} She has two cats {HEAVY * 3}\n"
    f"- {HEAVY * 20}\n"
    f"- She lives in Portland\n"
)
check(parsed == ["She has two cats", "She lives in Portland"],
      f"decoration stripped, pure-rule line dropped (got {parsed!r})")
check(not any(HEAVY in p for p in parsed),
      "nothing that reaches the store carries a box character")

print("[5] every summary tier goes through the same seam")
check(hasattr(summarizer, "_summarize_pieces_raw"),
      "the raw producer still exists")
import inspect  # noqa: E402
src = inspect.getsource(summarizer._summarize_pieces)
check("strip_rule_decoration" in src,
      "and the wrapper every tier calls strips decoration")
check("_summarize_pieces_raw" in src,
      "by delegating, so all five of the raw function's return paths are "
      "covered by one seam rather than five edits")

print("[6] the ASCII run requirement — a hyphen is not a rule")
check(textclean.rule_char_count("well-known co-op ice-cream") == 0,
      "single hyphens inside words do not count")
check(textclean.rule_char_count("---") == 3, "a run of three does")
check(textclean.rule_char_ratio("") == 0.0,
      "empty text is not decorated, and must not divide by zero")

print("[7] the ratio measure reports what the corpus showed")
sample = f"{HEAVY * 10}\nShe likes gardening.\n{HEAVY * 10}"
r = textclean.rule_char_ratio(sample)
check(0.4 < r < 0.6, f"a half-decorated block reads about half ({r:.2f})")

if FAILED:
    print(f"\n{len(FAILED)} check(s) FAILED")
    sys.exit(1)
print("\nAll rule-decoration checks passed.")
