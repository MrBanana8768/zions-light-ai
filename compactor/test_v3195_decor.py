"""
v3.1.9.5 lane fix/v3195-decor — p18a F2 and F3, the two findings in this
lane's files (textclean.py, dedup.py). F4 (retrieval budget pricing) needs
retrieval.py, which belongs to the other lane; not fixed here.

D2 (F2): strip_rule_decoration only recognised a hand-listed character set
and passed 102 of 120 characters of the 2026-09-20 wall shape through into
stored memory. Fixed with a code-point-agnostic run collapse: any run of
>= 6 identical characters that are neither alphanumeric nor whitespace, and
that are not already one of the characters this module curates, collapses
to 3 — enough to keep a genuine rule or emphasis run readable, never enough
to matter as a share of a summary or fact.

D1 (F3): dedup's merged canonical fact never got the decoration strip the
other two non-extraction write paths (commands._handle_remember,
facts._parse_extraction_output) already had. One line, at the site, in the
same order the siblings use — strip, then judge — placed before BOTH the
is_storable_fact check and the MIN_MERGE_LENGTH_RATIO check, since
stripping changes the length the ratio check has to compare.

D3: re-derivation of the "must not touch" list turned up nothing to fix in
this lane's files, but pins the one separation the brief called out by
name: the covered-turn fingerprint (summarizer._covered_turn_fingerprint)
must keep reading the client's raw array, never a normalized version, or
v3.1.9 hierarchy reuse breaks silently on the next turn.

Mutations this file exists to kill:

    D2 [1]  the generic run-collapse in strip_rule_decoration is skipped
    D2 [2]  the collapse threshold stops being >= 6
    D2 [3]  the collapse target stops being 3
    D2 [4]  a character already in _RULE_CHARS/_ASCII_RULE_CHARS is
            double-collapsed or double-counted by the new pass
    D2 [5]  rule_char_count stops counting unlisted decorative runs
    D2 [6]  whitespace runs get swept into the generic collapse
    D2 [7]  numeric (alphanumeric) runs get swept into the generic collapse
    D1 [8]  dedup.llm_merge_candidate stops stripping before storing
    D1 [9]  the strip moves to AFTER the MIN_MERGE_LENGTH_RATIO check
    D3 [10] the covered-turn fingerprint path starts reading normalized
            text, or the rollup body stops sourcing it from raw_messages

    python test_v3195_decor.py
"""

import asyncio
import inspect
import os
import sys
import tempfile

os.environ.setdefault("MODEL_REPO", "test-model")
os.environ.setdefault("VLLM_URL", "http://stub:8000")
os.environ["COMPACTOR_RAG_ENABLED"] = "false"
os.environ["COMPACTOR_STORAGE_ROOT"] = tempfile.mkdtemp(prefix="compactor-test-v3195-decor-")

import dedup  # noqa: E402
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


# The 2026-09-20 wall shape: seven distinct code points, six of them outside
# every character set textclean.py recognised before this fix (only U+2501
# was already in _RULE_CHARS). Entirely synthetic — code points and run
# lengths only, no real reply text.
HEAVY = "━"          # already in _RULE_CHARS
SYMS = [HEAVY, "≈", "∞", "↑", "⁓", "¨", "¡"]
UNLISTED = SYMS[1:]        # the six D2 has to newly catch
RUNS = (18, 28, 10, 16, 12, 20, 16)  # sum = 120, each unlisted run >= 6


def wall():
    return "".join(s * n for s, n in zip(SYMS, RUNS))


print("=== D2 [1]/[2]/[3]: strip_rule_decoration is code-point agnostic ===")
W = wall()
out = textclean.strip_rule_decoration(W)
check(len(out) == 18,  # 6 unlisted symbols x 3 survivors each, HEAVY -> 0
      f"a 120-char/7-symbol wall collapses to 18 residual chars (got {len(out)})")
check(all(out.count(s) in (0, 3) for s in SYMS),
      "each surviving symbol appears exactly 3 times, or not at all")
check(HEAVY not in out, "the one already-listed symbol is still fully removed")

print()
print("=== D2 [2]: the collapse threshold is a run of >= 6, not shorter ===")
for n, expect_collapsed in ((5, False), (6, True), (7, True)):
    s = "≈" * n
    got = textclean.strip_rule_decoration(s)
    if expect_collapsed:
        check(got == "≈" * 3, f"run of {n} collapses to 3 (got {got!r})")
    else:
        check(got == s, f"run of {n}, below the floor, is left alone (got {got!r})")

print()
print("=== D2: a MIXED line keeps the words, the wall collapses per symbol ===")
expected_mid = "".join(s * 3 for s in UNLISTED)
check(
    textclean.strip_rule_decoration(f"Status {W} Active")
    == f"Status {expected_mid} Active",
    "interior wall collapses per unlisted symbol; HEAVY removed; prose survives",
)

print()
print("=== D2: a PURE unlisted-decoration line no longer survives whole ===")
pure = "≈" * 40
after = textclean.strip_rule_decoration(pure)
check(len(after) == 3 and after == "≈" * 3,
      f"40 chars of one unlisted symbol collapse to 3, never pass through "
      f"whole (got {len(after)} chars)")

print()
print("=== D2 [4]/[6]/[7]: known rule chars, whitespace and numerals (CONTROL) ===")
check(textclean.strip_rule_decoration(HEAVY * 40) == "", "a heavy rule -> nothing, unchanged")
check(textclean.strip_rule_decoration("-" * 30) == "", "an ASCII rule -> nothing, unchanged")
check(
    textclean.strip_rule_decoration(f"{HEAVY * 3} Current Status {HEAVY * 3}")
    == "Current Status",
    "decoration wrapped around prose is still fully removed, prose survives",
)
check(textclean._collapse_generic_runs(" " * 12) == " " * 12,
      "a run of whitespace is excluded from the generic collapse")
check(textclean.strip_rule_decoration("Code: 000000 entered") == "Code: 000000 entered",
      "a numeric run is alphanumeric and is left alone")

print()
print("=== D2: ordinary prose and short punctuation runs untouched (CONTROL) ===")
for prose in (
    "She prefers tea to coffee.",
    "The well-known author lives in Portland.",
    "It cost 3-4 dollars.",
    "Use the em dash — like this — in ordinary prose.",
    "Wait... really?!",  # a 3-dot ellipsis and two marks, all below the floor
):
    check(textclean.strip_rule_decoration(prose) == prose,
          f"unchanged: {prose[:38]!r}")

print()
print("=== D2 [5]: rule_char_count agrees with what strip recognises as decoration ===")
check(textclean.rule_char_count(W) == 120,
      f"every wall character now counts as decoration (got "
      f"{textclean.rule_char_count(W)}/120, was 18/120 before this fix)")
check(textclean.rule_char_count("≈" * 5) == 0,
      "a run below the floor does not count")
check(textclean.rule_char_count("≈" * 6) == 6,
      "a run at the floor counts in FULL — a measurement of how decorated "
      "the text is, not of what survives the strip's 3-char residual")
check(textclean.rule_char_count("well-known co-op ice-cream") == 0,
      "single hyphens inside words still do not count (pre-existing rule, "
      "unmoved by this fix)")

print()
print("=== D2 propagates through the pre-existing facts.py seam (site 4) ===")
unknown_wall = "".join(s * 20 for s in UNLISTED)  # 120 chars, all unrecognised before D2
parsed = facts._parse_extraction_output(
    f"- {unknown_wall} She has two cats {unknown_wall}\n"
    f"- She lives in Portland\n"
)
check(len(parsed) == 2, f"both lines survive as facts (got {parsed!r})")
check("She has two cats" in parsed[0], "the prose is preserved")
check(len(parsed[0]) < len(f"{unknown_wall} She has two cats {unknown_wall}"),
      "and the wall is drastically shorter than the 120+120 chars it was")
check(textclean.rule_char_count(parsed[0]) <= 6 * 3,
      "residual decoration is bounded to the 3-per-symbol collapse, not "
      "the full wall — this seam needed no edit of its own, only the fix "
      "inside strip_rule_decoration")

print()
print("=== D1 [8]: dedup's merged canonical fact gets the same strip ===")
BOX = HEAVY * 3  # in _RULE_CHARS -- live on the pod for this shape since v3.1.8
cluster = [
    {"text": "She prefers tea in the morning rather than coffee.",
     "added_turn": 3, "last_used": 100},
    {"text": "In the mornings she reaches for tea, not coffee, every day.",
     "added_turn": 9, "last_used": 120},
]


class _Resp:
    def __init__(self, t):
        self._t = t

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": self._t}, "finish_reason": "stop"}]}


class _Client:
    def __init__(self, t):
        self._t = t

    async def post(self, *a, **k):
        return _Resp(self._t)


merge_text = f"{BOX} She prefers tea in the morning {BOX}"
text, reason = asyncio.run(
    dedup.llm_merge_candidate(_Client(merge_text), "http://x", "m", cluster)
)
check(reason == "merged", f"a decorated merge still merges (reason={reason!r})")
check(text == "She prefers tea in the morning",
      f"and the STORED text carries no box characters (got {text!r})")
check(text is not None and textclean.rule_char_count(text) == 0,
      "rule_char_count on the stored canonical fact is zero")
rec = dedup._merge_metadata(cluster, text)
check(HEAVY not in rec["text"], "the record dedup would actually write is clean too")

print()
print("=== D1: clean merges, KEEP, short and truncated verdicts unaffected (CONTROL) ===")
clean_merge = "She drinks tea most mornings instead of coffee."
t, r = asyncio.run(dedup.llm_merge_candidate(_Client(clean_merge), "http://x", "m", cluster))
check(r == "merged" and t == clean_merge, "an undecorated merge passes through byte for byte")

t, r = asyncio.run(
    dedup.llm_merge_candidate(_Client("KEEP - these are different facts"), "http://x", "m", cluster)
)
check(r == "keep" and t is None, "KEEP is still detected before any stripping happens")

t, r = asyncio.run(dedup.llm_merge_candidate(_Client("hi"), "http://x", "m", cluster))
check(r == "short" and t is None, "a too-short reply is still rejected as short")

print()
print("=== D1 [9]: MIN_MERGE_LENGTH_RATIO is measured AFTER stripping ===")
# Raw text is 35 chars (clears the pre-existing "short" floor); after
# stripping, only "Tea" (3 chars) remains -- well under
# shortest(50) * MIN_MERGE_LENGTH_RATIO(0.5) = 25. Before this fix, this
# decoration-padded reply cleared the length gate on its RAW length and was
# stored, replacing two real facts with three words of content.
tiny_decorated = (BOX + " ") * 8 + "Tea"
check(len(tiny_decorated) >= 6, "sanity: clears the short-reply floor pre-strip")
t, r = asyncio.run(
    dedup.llm_merge_candidate(_Client(tiny_decorated), "http://x", "m", cluster)
)
check(r == "collapsed" and t is None,
      f"a decoration-padded merge is now caught as collapsed and the "
      f"cluster is preserved, not replaced (got reason={r!r} text={t!r})")

print()
print("=== D3 [10]: the covered-turn fingerprint stays separate from memory-write "
      "normalization ===")
# The hazard this pins: if _covered_turn_fingerprint (or its caller) ever
# normalized text before hashing, a client re-sending the SAME raw turn on
# the next request would fingerprint differently from what was recorded,
# and every reusing chunk would read as edited -- silently breaking v3.1.9
# hierarchy reuse, the feature this whole release line exists for. Pinned
# two ways: (a) the fingerprint functions do not reference textclean at
# all, and (b) they are demonstrably sensitive to decoration, so a future
# change that fed them normalized text instead of raw text would visibly
# change what gets recorded.
for name in ("_covered_turn_fingerprint", "_covered_turn_fingerprints", "_record_chunk_fps"):
    src = inspect.getsource(getattr(summarizer, name))
    check("textclean" not in src, f"{name} does not import or call textclean")

decorated_msg = {"role": "assistant", "content": f"{BOX} She has two cats {BOX}"}
stripped_msg = {
    "role": "assistant",
    "content": textclean.strip_rule_decoration(decorated_msg["content"]),
}
fp_raw = summarizer._covered_turn_fingerprint(decorated_msg)
fp_stripped = summarizer._covered_turn_fingerprint(stripped_msg)
check(fp_raw != fp_stripped,
      "the fingerprint is sensitive to decoration -- feeding it normalized "
      "text instead of raw text would silently change what gets recorded")

body_src = inspect.getsource(summarizer._maybe_rollup_body)
check(
    "if raw_messages is not None" in body_src and "raw_turns = _rt" in body_src,
    "the covered-turn record is still built from raw_messages, not from "
    "`messages`, whenever the caller supplies it",
)

if FAILED:
    print(f"\n{len(FAILED)} check(s) FAILED")
    sys.exit(1)
print("\nAll v3.1.9.5 decor checks passed.")
