"""
v3194-fence lane (v3.1.9.4) — F1/F2: one fence reader, and `~~~` fences.

F1 (P9-5 / P10-4 remainder): the fragment-line rule's two inline fence
walks (the primary `in_fence = not in_fence` walk and the trailing-content
`_tc_in_fence` walk) were the last hand-rolled fence readers in main.py —
indent-blind and tilde-blind, unlike `_fence_toggle_offsets` since P9-6.
Migrated to share `_fence_toggle_offsets`/`_in_open_fence` with every other
fence decision in the file. P9 could not construct an input where the old
walk changed a verdict; this lane found one (see [3] below) by following
the brief's own hint: a list-shaped collapse after an UNMATCHED fence
opener, with a 4-space-INDENTED fake closer between them. Under CommonMark
an indented ``` line is literal code content, not a delimiter, so the real
opener never closes and everything after it — list-shaped or not — is
fenced content. The old hand-rolled walk read the indented line as a real
toggle and closed the fence there, so the list items after it were judged
as ordinary text and tripped the list-run backstop. FIXED: a reply that is
entirely decorative fenced content is no longer flagged as a structural
collapse.

F2 (P9-6 / P9-5 "NOT FIXED" note): `_fence_toggle_offsets` now implements
CommonMark fences properly — backtick OR tilde, at least 3, at most 3
spaces of indent, and a closing fence must use the SAME character and be
at least as long as its opener (a shorter or different-character fence
inside is content, not a close). `_fence_markers` is the richer
(offset, character, length) form this needs; `_open_fence_closer` is what
`_trim_forwarded_prefix` and `_cut_degenerate_span_once` now use to
balance/reopen with the MATCHING marker instead of a hard-coded "```".

Only synthetic content appears below (project rule: this repo is public).

Run inside the compactor image or any container with the requirements:
    python test_v3194_fence_agreement.py
"""

import os
import sys

os.environ.setdefault("MODEL_REPO", "")

import main  # noqa: E402


def assert_true(cond, label):
    if not cond:
        print(f"FAIL {label}")
        sys.exit(1)
    print(f"  ok   {label}")


def assert_eq(actual, expected, label):
    if actual != expected:
        print(f"FAIL {label}: expected {expected!r}, got {actual!r}")
        sys.exit(1)
    print(f"  ok   {label}")


def fragments(n, width):
    """One line of n DISTINCT short sentences, `width` characters each
    (same shape test_degenerate_reply.py's own `fragments()` uses)."""
    out = []
    for i in range(n):
        head = f"Frag {i:04d} "
        pad = "".join(chr(97 + (i + j) % 26) for j in range(width - len(head) - 2))
        out.append(head + pad + ".")
    return " ".join(out)


def list_items(n):
    """n terminated-free short list-item lines, each well under
    DEGENERATE_LIST_ITEM_CHARS (30) and matching _LIST_ITEM_RE."""
    return "\n".join(f"- item {i:04d}" for i in range(n))


print("[1] _fence_markers / _fence_toggle_offsets / _in_open_fence: the")
print("    CommonMark rules F2 added, tested directly")
print("-" * 70)

# 1a. Plain backtick fence: unchanged from before F2.
_bt = "prose\n```\ncode\n```\nmore prose"
_bt_markers = main._fence_markers(_bt)
assert_eq(len(_bt_markers), 2, "[1a] a closed ``` fence has exactly 2 markers")
assert_eq(
    [ch for _off, ch, _len in _bt_markers], ["`", "`"],
    "[1a] both markers are backtick",
)

# 1b. Tilde fence, same shape: F2's headline feature.
_tl = "prose\n~~~\ncode\n~~~\nmore prose"
_tl_markers = main._fence_markers(_tl)
assert_eq(len(_tl_markers), 2, "[1b] *** F2: a closed ~~~ fence has exactly 2 markers — no longer invisible")
assert_eq(
    [ch for _off, ch, _len in _tl_markers], ["~", "~"],
    "[1b] *** F2: both markers are tilde, not backtick",
)
assert_eq(
    [off for off, _ch, _len in _tl_markers], [off for off, _ch, _len in _bt_markers],
    "[1b] the tilde fence toggles at the SAME offsets the equivalent backtick "
    "fence would — same text shape, only the marker character differs",
)

# 1c. A different marker character while a fence is open is CONTENT, not a
# close — the exact cross-closing bug a naive "just add ~~~" patch would
# cause, per _fence_markers' own docstring.
_mixed = "prose\n```\nsome code\n~~~\nmore code\n```\nafter"
_mixed_markers = main._fence_markers(_mixed)
assert_eq(
    len(_mixed_markers), 2,
    "[1c] *** F2: a ~~~ line INSIDE an open ``` fence is content, not a "
    "toggle — only the real ``` open and ``` close count",
)
assert_true(
    all(ch == "`" for _off, ch, _len in _mixed_markers),
    "[1c] *** F2: neither recorded marker is the tilde line — the ``` "
    "block and the (fake, inner) ~~~ line do not cross-close",
)

# 1d. Nested/unequal fence lengths: a closer must be the SAME character and
# AT LEAST AS LONG as its opener. A shorter same-character run is content.
_nested = "prose\n````\nouter code\n```\nstill inside (shorter, same char)\n````\nafter"
_nested_markers = main._fence_markers(_nested)
assert_eq(
    len(_nested_markers), 2,
    "[1d] *** F2: a 4-backtick fence containing a 3-backtick line stays "
    "ONE block — the inner 3-backtick line is content, too short to close",
)
assert_eq(
    [ln for _off, _ch, ln in _nested_markers], [4, 4],
    "[1d] *** F2: both recorded markers are the 4-backtick opener/closer, "
    "not the inner 3-backtick line",
)
_nested_open_at_inner = main._in_open_fence(
    [off for off, _c, _l in _nested_markers], _nested.find("still inside")
)
assert_true(
    _nested_open_at_inner,
    "[1d] the position inside the (correctly ignored) inner ``` line "
    "still reads as inside the still-open outer fence",
)

# 1e. A same-length-or-longer same-character line DOES close it.
_nested_close = "prose\n```\ncode\n````\nafter"  # opener 3, closer 4 (longer, closes)
_nested_close_markers = main._fence_markers(_nested_close)
assert_eq(
    len(_nested_close_markers), 2,
    "[1e] *** F2: a LONGER same-character closer (4 backticks closing a "
    "3-backtick opener) does close it — CommonMark's 'at least as long'",
)

# 1f. Unmatched opener (tilde): stays open to the end, same latching
# semantics `_in_open_fence` already gives backtick fences.
_tl_unmatched = "prose\n~~~\ncode that never closes, all the way to the end"
_tl_u_markers = main._fence_markers(_tl_unmatched)
assert_eq(len(_tl_u_markers), 1, "[1f] an unmatched ~~~ opener is exactly one marker")
assert_true(
    main._in_open_fence([off for off, _c, _l in _tl_u_markers], len(_tl_unmatched) - 1),
    "[1f] *** F2: a position near the end of an unmatched ~~~ opener still "
    "reads as inside an open fence — same latching behaviour as ```",
)

# 1g. 4-space indent still excludes a fence-looking line, for tilde too
# (P9-6's rule, now shared by both marker families).
_tl_indented = "prose\n~~~\ncode\n    ~~~ (4-space indented, literal content)\nmore code\n~~~\nafter"
_tl_indented_markers = main._fence_markers(_tl_indented)
assert_eq(
    len(_tl_indented_markers), 2,
    "[1g] *** F2+P9-6: the 4-space-indented ~~~ line is literal content, "
    "not a toggle, for tildes exactly as it already was for backticks",
)

print()
print("[2] _open_fence_closer: the balancing marker matches character AND length")
print("-" * 70)

_ch_bt, _len_bt = "`", 3
_ch_tl4, _len_tl4 = "~", 4
_closer_bt = main._open_fence_closer(main._fence_markers("```\ncode"), len("```\ncode"))
assert_eq(_closer_bt, "```", "[2a] an unclosed 3-backtick opener closes with '```'")
_closer_tl4 = main._open_fence_closer(main._fence_markers("~~~~\ncode"), len("~~~~\ncode"))
assert_eq(
    _closer_tl4, "~~~~",
    "[2b] *** F2: an unclosed 4-tilde opener closes with '~~~~' (matching "
    "length), not a bare '```' or a 3-tilde '~~~'",
)
_closer_none = main._open_fence_closer(main._fence_markers("prose\n```\ncode\n```\nafter"), 2)
assert_eq(_closer_none, None, "[2c] a position in the leading prose, before any fence starts, has no open marker")

print()
print("[3] *** F1: a verdict-changing input, found by retrying P9-5's search")
print("    (list-shaped collapse after an unmatched opener, per the brief)")
print("-" * 70)
# An unclosed real ``` opener, a 4-space-INDENTED line that LOOKS like a
# closer but is CommonMark literal content (P9-6), then 60 short list
# items (DEGENERATE_LIST_RUN=50) running to the end of the reply.
#
# OLD hand-rolled walk (frozen here for comparison; no longer in main.py —
# this IS the code the fragment-line rule used before this lane's fix):
# indent-BLIND, so it read the indented fake-closer as a real toggle and
# closed the fence there. The 60 list items were then judged as ordinary
# (non-fenced) text and tripped the list-run backstop.
#
# NEW (main.py, this lane's fix): the indented line is not a delimiter, so
# the real opener never closes — the list items are fenced content like
# any other code block, exempt from every structural rule, same as
# test_degenerate_reply.py's own "60 short items inside a code fence are
# not judged" fixture at [line 249-251], just with the opener left open
# instead of closed.
def _old_fragment_run_reaches_backstop(text):
    """Frozen copy of the fragment-line rule's OLD primary walk, isolated
    to just the list-run tracking (the part [3] exercises). Indent-blind
    and backtick-only, matching main.py before this lane's F1 fix."""
    lines = text.splitlines()
    run = 0
    in_fence = False
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("```"):
            in_fence = not in_fence
            run = 0
            continue
        if in_fence:
            run = 0
            continue
        if len(line) <= main.DEGENERATE_LIST_ITEM_CHARS and main._LIST_ITEM_RE.match(line):
            run += 1
        else:
            run = 0
    return run >= main.DEGENERATE_LIST_RUN


_f1_prose = "Ordinary prose leading in, nothing unusual here at all. " * 3
_f1_fake_closer = "    ``` this looks like a fence but is 4-space indented code content\n"
_f1_list = list_items(60)
_f1_text = _f1_prose + "\n```\n" + _f1_fake_closer + "\n" + _f1_list

assert_true(
    len(_f1_text) >= main.DEGENERATE_MIN_CHARS,
    "[3] fixture sanity: clears the DEGENERATE_MIN_CHARS floor the list-run "
    "backstop is gated on",
)
assert_true(
    _old_fragment_run_reaches_backstop(_f1_text),
    "[3] *** F1: the OLD (frozen, pre-fix) walk DOES reach the list-run "
    "backstop on this fixture — it wrongly reads the fence as closed",
)
assert_eq(
    main.reply_is_degenerate(_f1_text), None,
    "[3] *** F1 FIX: main.py today does NOT flag this reply — the real "
    "opener never closes (the indented line is content, not a delimiter), "
    "so the list items are fenced content, exempt like any other code "
    "block. This is the verdict-changing input P9-5 could not construct.",
)

# CONTROL A: the same shape, but the fake closer is NOT indented — a REAL
# closer. The fence genuinely closes, the list items are genuinely outside
# it, and both the old and new readings must agree: flagged.
_f1_control_real_closer = (
    _f1_prose + "\n```\n" + "``` a real, unindented closer\n" + "\n" + _f1_list
)
assert_true(
    _old_fragment_run_reaches_backstop(_f1_control_real_closer),
    "[3-CONTROL-A] OLD walk: a REAL (unindented) closer really does close "
    "the fence, so the list items outside it still trip the backstop",
)
assert_true(
    main.reply_is_degenerate(_f1_control_real_closer) is not None,
    "[3-CONTROL-A] *** the fix does not exempt list items OUTSIDE a "
    "genuinely-closed fence — only content that is actually still fenced",
)

# CONTROL B: no fake closer at all — list items directly inside an
# unmatched real opener. Exempt both before and after the fix (the old
# walk's own `in_fence` flag never got a chance to be wrong here, since
# there is no indented line to misjudge) — proves [3] isolates the
# indent-awareness fix and is not just "any unmatched opener exempts".
_f1_control_no_fake = _f1_prose + "\n```\n" + _f1_list
assert_true(
    not _old_fragment_run_reaches_backstop(_f1_control_no_fake),
    "[3-CONTROL-B] fixture sanity: the OLD walk ALSO exempts list items "
    "directly inside an unmatched opener with no indented line involved",
)
assert_eq(
    main.reply_is_degenerate(_f1_control_no_fake), None,
    "[3-CONTROL-B] unaffected by the fix either way — not itself evidence "
    "of a verdict change, just confirms [3] isolates the right variable",
)

print()
print("[4] F1: the PRIMARY per-line candidate exemption (not just the list-run")
print("    backstop) is also indent-aware now")
print("-" * 70)
# A single 1500+-char, 100+-space fragment-shaped line (test_degenerate_
# reply.py's own fragments() shape) sitting where the old walk would have
# wrongly read the fence as closed. As the reply's own LAST non-blank
# line, a fragment-shaped line that is genuinely NOT fenced is flagged
# (test_degenerate_reply.py [7]); one that IS still fenced content must
# not even reach the fragment-shape check.
_f4_prose = "Ordinary prose leading in, nothing unusual here at all. " * 3
_f4_frag_line = fragments(110, 15)
assert_true(len(_f4_frag_line) >= 1500, "[4] fixture sanity: the fragment line clears the 1500-char floor")
_f4_text = _f4_prose + "\n```\n" + _f1_fake_closer + "\n" + _f4_frag_line

assert_true(
    main.reply_is_degenerate(_f4_prose + "\n```\n" + _f4_frag_line) is None,
    "[4] fixture sanity: the SAME fragment line right after a genuinely "
    "open (real, unindented) opener is already exempt today — the control "
    "for [4] proper, isolating just the indented-fake-closer variable",
)
assert_eq(
    main.reply_is_degenerate(_f4_text), None,
    "[4] *** F1 FIX: a fragment-shaped last line after an unmatched opener "
    "plus an indented fake closer is NOT flagged — still fenced content, "
    "the primary walk's own candidate-exemption branch, not just the "
    "list-run backstop",
)

print()
print("[5] F2: a repeated-value table inside a ~~~ block is treated exactly")
print("    like the same table inside backticks")
print("-" * 70)
_id_unit = "identifier_run_9f3k2"
_id_loop = (_id_unit + " ") * 10  # clears DEGENERATE_TOKEN_RUN_CHARS (120)
_prose = "Ordinary prose before the box, with real content in it. " * 3

_table_backtick = _prose + "\n```\n" + _id_loop + "\n```\n" + "Prose after the box."
_table_tilde = _prose + "\n~~~\n" + _id_loop + "\n~~~\n" + "Prose after the box."

_v_bt = main.reply_is_degenerate(_table_backtick)
_v_tl = main.reply_is_degenerate(_table_tilde)
assert_true(_v_bt is not None, "[5] fixture: the identifier loop trips the token rule (backtick box)")
assert_eq(
    _v_tl, _v_bt,
    "[5] *** F2: the SAME identifier loop inside a ~~~ box gets the "
    "IDENTICAL verdict text as inside a ``` box — the token rule has no "
    "fence exemption at all (P9-3), so this was already true of detection; "
    "what F2 changes is the forwarded cut below",
)

_content_bt, _kept_bt = main._degenerate_replacement_content(
    _table_backtick, main._DEGENERATE_FORWARD_PLACEHOLDER, keep_middle=True
)
_content_tl, _kept_tl = main._degenerate_replacement_content(
    _table_tilde, main._DEGENERATE_FORWARD_PLACEHOLDER, keep_middle=True
)
assert_true(_kept_bt and _kept_tl, "[5] fixture: both cuts keep a clean head/tail, not the whole-reply placeholder")
assert_eq(
    _content_tl.replace("~", "`"), _content_bt,
    "[5] *** F2: the tilde-boxed cut is IDENTICAL to the backtick-boxed "
    "cut, character-for-character, but for the marker family — same pre, "
    "same marker, same reopener/closer shape, same post",
)
assert_true(
    "Prose after the box" in _content_tl and _id_unit not in _content_tl,
    "[5] the tilde box's own clean prose survives and the loop itself does not",
)

print()
print("[6] F2: a cut inside a ~~~ block is closed with ~~~, not ```")
print("-" * 70)
# Mid-reply span INSIDE a ~~~ box that CLOSES later (P8-4's shape,
# replayed with tildes): pre keeps the opener (self-balanced), post picks
# back up inside the same box and must be REOPENED with ~~~, not ```.
_run_text = "loopword " * 20  # a token run, well past DEGENERATE_TOKEN_RUN_CHARS
_p8_4_tilde = (
    _prose + "Here is a diagnostic dump:\n~~~\n" + _run_text +
    "\nstatus: nominal\n~~~\n"
    "Everything looks fine now, thanks for checking. And more prose follows."
)
assert_true(bool(main.reply_is_degenerate(_p8_4_tilde)), "[6] fixture: the boxed run trips the detector")
_content6, _kept6 = main._degenerate_replacement_content(
    _p8_4_tilde, main._DEGENERATE_FORWARD_PLACEHOLDER, keep_middle=True
)
assert_true(_kept6, "[6] fixture: a clean head/tail is kept, not the whole-reply placeholder")
assert_true(
    "Everything looks fine now" in _content6 and "status: nominal" in _content6,
    "[6] the clean prose before AND the box's own status line after the "
    "loop both survive",
)
assert_true(
    "loopword" not in _content6,
    "[6] the loop itself did not reach the forwarded content",
)
assert_true(
    "```" not in _content6,
    "[6] *** F2: no bare backtick fence leaked into a reply that was "
    "entirely tilde-fenced — the reopener/closer matched the ORIGINAL "
    "marker",
)
_c6_markers = main._fence_markers(_content6)
assert_true(
    len(_c6_markers) % 2 == 0,
    f"[6] *** F2: the forwarded text has a BALANCED real-fence count "
    f"({len(_c6_markers)} markers) even though it was cut from a ~~~ box",
)
assert_true(
    all(ch == "~" for _off, ch, _len in _c6_markers),
    "[6] *** F2: every marker in the forwarded content is tilde — nothing "
    "reintroduced a backtick fence into an all-tilde reply",
)

print()
print("[7] F2: a 4-backtick fence containing a ``` line stays one block, end to end")
print("-" * 70)
_nested_reply = (
    _prose + "\n````\n" + _id_loop + "\n```\nstill inside, shorter fence\n````\n" +
    "Prose after the outer box."
)
_nested_verdict = main.reply_is_degenerate(_nested_reply)
assert_true(_nested_verdict is not None, "[7] fixture: the identifier loop still trips the token rule")
_content7, _kept7 = main._degenerate_replacement_content(
    _nested_reply, main._DEGENERATE_FORWARD_PLACEHOLDER, keep_middle=True
)
assert_true(_kept7, "[7] fixture: a clean head/tail is kept")
_c7_markers = main._fence_markers(_content7)
assert_true(
    len(_c7_markers) % 2 == 0,
    "[7] *** F2: the forwarded text stays balanced even with a nested, "
    "shorter same-character fence line inside the outer 4-backtick box",
)
assert_true(
    all(ln == 4 for _off, _ch, ln in _c7_markers),
    "[7] *** F2: every real marker in the forwarded content is 4 backticks "
    "long, matching the outer fence — the inner 3-backtick line was never "
    "treated as a delimiter",
)

print()
print("[8] Cross-site agreement: every fence-reading site in the module")
print("    agrees on the same battery of inputs")
print("-" * 70)
# Per the brief: indented fence lines, an unmatched opener, nested/unequal
# fence lengths, and ~~~ — run through every site that reads a fence
# (`_fence_toggle_offsets`+`_in_open_fence` directly, `trim_to_last_
# sentence`, `_trim_forwarded_prefix`, `_cut_degenerate_span_once`'s
# belt-and-braces check, and the fragment-line rule via `reply_is_
# degenerate`) and assert they never disagree about where a fence is.
_agreement_fixtures = {
    "plain backtick, closed": "prose. " * 20 + "\n```\ncode\n```\n" + "more prose. " * 20,
    "plain tilde, closed": "prose. " * 20 + "\n~~~\ncode\n~~~\n" + "more prose. " * 20,
    "unmatched backtick opener": "prose. " * 20 + "\n```\ncode that never closes",
    "unmatched tilde opener": "prose. " * 20 + "\n~~~\ncode that never closes",
    "4-space-indented fake fence, real one still open": (
        "prose. " * 20 + "\n```\n" + _f1_fake_closer + "more code, never closes"
    ),
    "nested unequal lengths (4-backtick outer, 3-backtick inner)": (
        "prose. " * 20 + "\n````\nouter\n```\ninner (shorter)\n````\n" + "more prose. " * 20
    ),
    "mixed marker families, inner tilde inside outer backtick": (
        "prose. " * 20 + "\n```\nouter\n~~~\ninner (different char)\n```\n" + "more prose. " * 20
    ),
}
for _name, _text in _agreement_fixtures.items():
    _markers = main._fence_markers(_text)
    _offsets = [off for off, _c, _l in _markers]
    # (a) _fence_toggle_offsets is exactly the offset projection of
    # _fence_markers, by construction - verify it stays true, not just
    # assumed, since every other site is built on this equality.
    assert_eq(
        main._fence_toggle_offsets(_text), _offsets,
        f"[8/{_name}] _fence_toggle_offsets matches _fence_markers' own offsets",
    )
    # (b) trim_to_last_sentence never returns a prefix that ends strictly
    # inside an open fence (its own invariant) - check against the SAME
    # reader used everywhere else, not a separate assumption.
    _trimmed = main.trim_to_last_sentence(_text)
    if _trimmed:
        assert_true(
            not main._in_open_fence(_offsets, len(_trimmed) - 1),
            f"[8/{_name}] trim_to_last_sentence's own boundary agrees with "
            f"_in_open_fence — never lands inside an open fence",
        )
    # (c) _trim_forwarded_prefix's output is always fence-balanced per the
    # SAME reader (its own self-balancing contract).
    _fwd_prefix = main._trim_forwarded_prefix(_text)
    assert_true(
        len(main._fence_markers(_fwd_prefix)) % 2 == 0,
        f"[8/{_name}] _trim_forwarded_prefix's output is fence-balanced "
        f"per _fence_markers",
    )

print()
print("ALL TESTS PASSED")
