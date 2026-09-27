"""
Tests for ops/zlops/privacy.py — the ONE log scrubber `zl bundle` and `zl
dash`'s log panel both import.

All test lines are SYNTHETIC, built in the real line SHAPES this module's
own docstring documents (compactor/dedup.py's fact-merge INFO line,
compactor/main.py's repetition-loop WARNING, and OpenWebUI/SQLAlchemy's
`[SQL: ...]`/`[parameters: ...]` blocks) — never content copied from a real
bundle. The real-bundle proof (counts only, never content) is a separate,
manual step documented in the hand-off report, not a test here (a test
that depends on `/home/drew/pod-exports` existing would not run in CI or
inside the docker unit-tests image, which does not mount that path).

Run: python test_ops_privacy.py
"""

from __future__ import annotations

import sys
# LOW (round 3): never write ops/**/__pycache__ from a test run --
# this file's own `from zlops import ...` below would otherwise create
# it even when the harness invoking this file forgot PYTHONDONTWRITEBYTECODE.
sys.dont_write_bytecode = True
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_OPS = _HERE.parent / "ops"
if str(_OPS) not in sys.path:
    sys.path.insert(0, str(_OPS))

from zlops import privacy  # noqa: E402


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


def assert_in(needle, haystack, label):
    if needle not in haystack:
        print(f"FAIL {label}: {needle!r} not in {haystack!r}")
        sys.exit(1)
    print(f"  ok   {label}")


def assert_not_in(needle, haystack, label):
    if needle in haystack:
        print(f"FAIL {label}: {needle!r} unexpectedly in {haystack!r}")
        sys.exit(1)
    print(f"  ok   {label}")


# =============================================================================
# dedup.py shape: "dedup: merging N fact(s) into '<text>'; removed: [...]"
# =============================================================================


def test_dedup_merge_line_is_scrubbed():
    secret = "she prefers oat milk in her coffee every morning"
    removed = "['likes oat milk', 'drinks coffee with oat milk daily']"
    line = (
        "2026-09-22 09:19:33,554 compactor.dedup INFO "
        f"dedup: merging 2 fact(s) into {secret!r}; removed: {removed}"
    )
    out = privacy.scrub_log_line(line, "compactor.log")
    assert_true(out is not None, "dedup merge line survives (scrubbed, not dropped)")
    assert_not_in(secret, out, "the merged fact text itself is gone")
    assert_not_in("likes oat milk", out, "a removed fact text is gone")
    assert_in("<scrubbed:", out, "a scrub marker is present")
    assert_in("dedup: merging 2 fact(s) into", out, "the diagnostic prefix survives")
    assert_in("compactor.dedup INFO", out, "the timestamp/level/logger prefix survives")


def test_dedup_merge_line_scrub_count_matches_removed_length():
    removed_repr = "['a', 'b']"
    line = f"dedup: merging 2 fact(s) into 'x'; removed: {removed_repr}"
    out = privacy.scrub_log_line(line, "compactor.log")
    assert_in(f"<scrubbed:{len(removed_repr)} chars>", out, "removed-span length is reported")


# =============================================================================
# main.py shape: repetition-loop WARNING with up to 24 chars of her reply
# =============================================================================


def test_repetition_loop_token_is_scrubbed():
    leaked = "her actual secret plan h"  # 24 chars, as main.py's [:24] caps it
    assert len(leaked) == 24
    line = (
        "2026-09-22 10:00:00,000 compactor WARNING conv=abc123: reply looks "
        f"like a repetition loop (the token {leaked!r} repeated for 480 "
        "characters (limit 400)) — skipping memory tail (900 chars accumulated; streak=3)"
    )
    out = privacy.scrub_log_line(line, "compactor.log")
    assert_not_in(leaked, out, "the leaked reply token is gone")
    assert_in("repeated for 480 characters (limit 400)", out, "the diagnostic numbers survive")
    assert_in("skipping memory tail", out, "the rest of the line survives")
    assert_in("conv=abc123", out, "the conversation id survives (not content)")


def test_repetition_token_with_both_quote_types_is_fully_scrubbed():
    """F2 (review follow-up): a token containing BOTH a single quote and a
    double quote -- Python's `repr()` then picks single quotes for the
    outer delimiter and escapes only the embedded single quote
    (`repr("Mom's \"garden\"")` -> `'Mom\\'s "garden"'`) -- must not let
    the embedded escaped quote (or the unescaped double quote) end the
    scrub span early. The old `_scrub_repetition_token` used a plain
    `line.find(quote, ...)`, which is not escape-aware and would close
    the span at the FIRST embedded `'`, whatever precedes it, leaking the
    rest of the token (here, "garden") verbatim.
    """
    tok = "Mom's \"garden\""
    r = repr(tok)
    assert r == "'Mom\\'s \"garden\"'", f"unexpected repr shape: {r!r}"
    line = (
        "conv=abc123: reply looks like a repetition loop "
        f"(the token {r} repeated for 42 characters (limit 40))"
    )
    out = privacy.scrub_log_line(line, "compactor.log")
    assert_not_in("Mom", out, "the first part of the token is scrubbed")
    assert_not_in("garden", out, "the second part of the token (past the escaped quote) is also scrubbed")
    assert_in("repeated for 42 characters (limit 40)", out, "the diagnostic numbers survive")
    assert_in("conv=abc123", out, "the conversation id survives")


def test_mutation_reverting_repetition_token_escape_handling_leaks_the_rest():
    """Mutation guard for F2: simulates the OLD, non-escape-aware
    `_scrub_repetition_token` (a plain `line.find(quote, ...)`) and proves
    it actually leaks -- i.e. this test would FAIL against the buggy
    implementation, so it bites if that regression is ever reintroduced.
    """

    def _old_scrub_repetition_token(line: str) -> str:
        m = privacy._REPETITION_TOKEN_RE.search(line)
        if not m:
            return line
        quote = m.group(2)
        prefix_end = m.end(1)
        close_quote_at = line.find(quote, prefix_end + 1)
        if close_quote_at < 0:
            return line
        token_span = line[prefix_end : close_quote_at + 1]
        return line[:prefix_end] + f"<scrubbed:{len(token_span)} chars>" + line[close_quote_at + 1 :]

    tok = "Mom's \"garden\""
    r = repr(tok)
    line = (
        "conv=abc123: reply looks like a repetition loop "
        f"(the token {r} repeated for 42 characters (limit 40))"
    )
    old_out = _old_scrub_repetition_token(line)
    assert_in("garden", old_out, "sanity: the OLD implementation really does leak 'garden' here")
    new_out = privacy._scrub_repetition_token(line)
    assert_not_in("garden", new_out, "the FIXED implementation does not leak it")


# =============================================================================
# OpenWebUI / SQLAlchemy: [SQL: ...] and [parameters: (...)]
# =============================================================================


def test_sql_parameters_single_line_scrubbed():
    line = "[parameters: ('her actual message text here', 'abc-123')]"
    out = privacy.scrub_log_line(line, "openwebui.log")
    assert_not_in("her actual message text here", out, "the bound parameter value is gone")
    assert_in("[parameters: <scrubbed:", out, "the bracket marker and scrub tag survive")


def test_sql_block_single_line_scrubbed():
    line = "[SQL: UPDATE chat SET title=? WHERE id=?]"
    out = privacy.scrub_log_line(line, "openwebui.log")
    assert_not_in("UPDATE chat", out, "the SQL text itself is gone (defense in depth)")
    assert_in("[SQL: <scrubbed:", out, "the bracket marker survives")


def test_sql_block_spanning_multiple_lines_scrubbed_as_one_record():
    secret = "her actual chat message content goes here"
    text = (
        "sqlalchemy.exc.OperationalError: (sqlite3.OperationalError) database is locked\n"
        "[SQL: INSERT INTO chat (id, title, chat)\n"
        "VALUES (?, ?, ?)]\n"
        f"[parameters: ('abc-123', 'untitled', {secret!r})]\n"
    )
    out = privacy.scrub_log_text(text, "openwebui.log")
    assert_not_in(secret, out, "the multi-line SQL block's payload is gone")
    assert_not_in("INSERT INTO chat", out, "the multi-line SQL statement text is gone")
    assert_in("database is locked", out, "the diagnostic error line survives")
    assert_eq(out.count("[SQL: <scrubbed:"), 1, "the 2-line SQL block collapses to ONE scrubbed line")
    assert_in("[parameters: <scrubbed:", out, "the parameters line is scrubbed too")


def test_scrub_log_text_preserves_trailing_newline():
    text = "a\nb\n"
    out = privacy.scrub_log_text(text, "generic")
    assert_true(out.endswith("\n"), "trailing newline is preserved")
    text_no_nl = "a\nb"
    out2 = privacy.scrub_log_text(text_no_nl, "generic")
    assert_true(not out2.endswith("\n"), "no trailing newline is added when the input has none")


# =============================================================================
# Benign lines pass through unchanged — the whole point of scrubbing SPANS
# instead of dropping LINES (the previous approach dropped 265/876, 30%, of
# real compactor.log WARNING/ERROR lines because they contained the bare
# word "content" with no user text anywhere near it).
# =============================================================================


def test_benign_content_word_line_survives_unscrubbed():
    line = "2026-09-22 09:19:29,845 compactor WARNING could not apply the chat template; content window trimmed"
    out = privacy.scrub_log_line(line, "compactor.log")
    assert_eq(out, line, "a diagnostic line that merely contains the word 'content' is untouched")


def test_benign_vllm_unreachable_line_survives():
    line = "2026-09-22 09:10:32,103 compactor WARNING vLLM unreachable (/v1/models): Connection refused"
    out = privacy.scrub_log_line(line, "compactor.log")
    assert_eq(out, line, "an ordinary connectivity warning is untouched")


def test_uvicorn_style_line_survives():
    line = "INFO:     Application startup complete."
    out = privacy.scrub_log_line(line, "compactor-error.log")
    assert_eq(out, line, "a plain uvicorn lifecycle line is untouched")


def test_none_line_returns_none():
    assert_eq(privacy.scrub_log_line(None, "generic"), None, "None in, None out")


# =============================================================================
# Generic defense-in-depth (any future call site nobody has audited yet)
# =============================================================================


def test_generic_content_colon_quoted_is_scrubbed():
    line = "logger.info: content: 'this is a message body nobody reviewed'"
    out = privacy.scrub_log_line(line, "generic")
    assert_not_in("this is a message body nobody reviewed", out, "a generic content: '...' span is scrubbed")


def test_generic_reply_equals_quoted_is_scrubbed():
    line = 'debug dump reply="the assistant said something private"'
    out = privacy.scrub_log_line(line, "generic")
    assert_not_in("the assistant said something private", out, "a generic reply=\"...\" span is scrubbed")


# =============================================================================
# Mutation proof: a test only counts if you watched it fail.
# =============================================================================


def test_mutation_disabling_dedup_pattern_is_caught():
    """Prove `test_dedup_merge_line_is_scrubbed` actually bites: break the
    dedup-merge regex so it never matches, watch the real test fail, then
    restore from an in-memory copy (never git).

    Round 3 (M2) widened `_scrub_long_quoted_spans` to scrub ANY quoted
    span with whitespace on compactor*.log, regardless of length -- so
    "leaked fact text" (it has spaces) is now ALSO caught by that generic
    allowlist-shaped fallback even with `_DEDUP_MERGE_RE` disabled. That
    is real defense in depth, not a test bug, but it means isolating
    `_DEDUP_MERGE_RE`'s OWN contribution needs the fallback disabled too
    for the duration of this one mutation, or this test would no longer
    prove anything about the specific pattern it names.
    """
    original = privacy._DEDUP_MERGE_RE
    original_fallback = privacy._scrub_long_quoted_spans
    try:
        import re

        privacy._DEDUP_MERGE_RE = re.compile(r"$a^never matches^a$")  # unsatisfiable
        privacy._scrub_long_quoted_spans = lambda line: line  # isolate: no fallback either
        secret = "leaked fact text"
        line = f"dedup: merging 1 fact(s) into {secret!r}; removed: ['x']"
        out = privacy.scrub_log_line(line, "compactor.log")
        caught = secret not in out
        assert_true(not caught, "mutated (disabled) pattern no longer scrubs -- confirms the test is not tautological")
    finally:
        privacy._DEDUP_MERGE_RE = original
        privacy._scrub_long_quoted_spans = original_fallback
    # Now prove the restored version is green again.
    secret = "leaked fact text"
    line = f"dedup: merging 1 fact(s) into {secret!r}; removed: ['x']"
    out = privacy.scrub_log_line(line, "compactor.log")
    assert_not_in(secret, out, "restored pattern scrubs again")


def test_mutation_disabling_sql_bracket_scrub_is_caught():
    original = privacy._scrub_bracket_span
    try:
        privacy._scrub_bracket_span = lambda line, marker: line  # no-op mutation
        secret = "her chat text"
        line = f"[parameters: ('{secret}',)]"
        out = privacy.scrub_log_line(line, "openwebui.log")
        caught = secret not in out
        assert_true(not caught, "mutated (no-op) bracket scrub no longer removes the parameter value")
    finally:
        privacy._scrub_bracket_span = original
    secret = "her chat text"
    line = f"[parameters: ('{secret}',)]"
    out = privacy.scrub_log_line(line, "openwebui.log")
    assert_not_in(secret, out, "restored bracket scrub removes it again")


# =============================================================================
# Real-bundle proof (counts only, never content) — best-effort, skipped
# honestly if the bundles aren't present on this machine (they never will
# be inside the docker unit-tests image or CI).
# =============================================================================


def test_real_bundles_scrub_cleanly_if_present():
    import glob

    roots = sorted(glob.glob("/home/drew/pod-exports/*/*/zla-bundle-*/logs/zla"))
    if not roots:
        print("  skip real_bundles_scrub_cleanly_if_present: no /home/drew/pod-exports bundles on this machine")
        return
    total_spans = 0
    for root in roots:
        for name in ("compactor.log", "compactor-error.log", "openwebui.log", "openwebui-error.log"):
            p = Path(root) / name
            if not p.is_file():
                continue
            text = p.read_text(encoding="utf-8", errors="replace")
            scrubbed = privacy.scrub_log_text(text, name)
            total_spans += scrubbed.count("<scrubbed:")
            assert_true(
                "[parameters: <scrubbed" in scrubbed or "[parameters:" not in scrubbed,
                f"{root}/{name}: no unscrubbed [parameters: block survives",
            )
            assert_true(
                "[SQL: <scrubbed" in scrubbed or "[SQL:" not in scrubbed,
                f"{root}/{name}: no unscrubbed [SQL: block survives",
            )
    print(f"  ok   real bundles: {total_spans} spans scrubbed across {len(roots)} bundle(s), zero unscrubbed blocks")


# =============================================================================
# R2-B4: the allowlist-shaped long-quoted-span pass for compactor*.log
# (synthetic lines built in the exact SHAPE of v3.1.9 facts.py:1543 and
# :1706 -- never real model output).
# =============================================================================


def test_facts_extraction_rejected_markup_shape_is_scrubbed():
    # facts.py:1543 (v3.1.9) / :1571 (worktree) shape.
    secret = "## Heading that looks like a fake markdown reply from the model"
    line = f"conv=abc123: extraction returned 1 line(s) of markup rather than facts; not stored — parse error: {secret[:60]!r}"
    out = privacy.scrub_log_line(line, "compactor.log")
    assert_not_in(secret[:60], out, "the rejected-markup snippet is gone")
    assert_true("<scrubbed:" in out, "a scrub marker is present")
    assert_true("conv=abc123" in out, "the conv id survives (diagnostic, not content)")
    assert_true("extraction returned 1 line(s) of markup" in out, "the diagnostic prose survives")


def test_facts_zero_fact_snippet_shape_is_scrubbed():
    # facts.py:1706 (v3.1.9) / :1744 (worktree) shape.
    secret = "sure, here is a whole paragraph the model actually said back to her"
    line = f"conv=abc123: extracted 0 fact(s) — model returned: {secret!r}"
    out = privacy.scrub_log_line(line, "compactor.log")
    assert_not_in(secret, out, "the model-returned snippet is gone")
    assert_true("extracted 0 fact(s)" in out, "the diagnostic prose survives")


def test_short_quoted_span_under_threshold_survives():
    # A short, genuinely diagnostic repr (a state name, <=16 chars) must
    # not be eaten -- the brief says "keep diagnostic value".
    line = "conv=abc123: resuming a backfill whose earlier attempt left state='would-resume'"
    out = privacy.scrub_log_line(line, "compactor.log")
    assert_true("'would-resume'" in out, "a short state repr is not scrubbed (only 13 chars)")


def test_identifier_shaped_long_span_survives():
    # A long but identifier-shaped repr (a path, no whitespace) must
    # survive even past the 16-char threshold -- it cannot be her text.
    line = "COMPACTOR_BACKUP_REMOTE='s3://some-bucket/long/path/prefix' is set but off-volume upload is not implemented"
    out = privacy.scrub_log_line(line, "compactor.log")
    assert_true("s3://some-bucket/long/path/prefix" in out, "a long path-shaped repr (no spaces) is not scrubbed")


def test_non_compactor_source_not_subject_to_long_span_pass():
    # Gated by source: an openwebui.log line with a long quoted span that
    # is NOT one of the specific known-risky shapes is left alone by this
    # pass (openwebui has its own [SQL:]/[parameters:] coverage instead).
    line = "uvicorn.access: GET /api/config \"some long quoted header value here\" 200"
    out = privacy.scrub_log_line(line, "openwebui.log")
    assert_true("some long quoted header value here" in out, "long-span pass does not apply to non-compactor sources")


def test_mutation_disabling_long_quoted_span_scrub_is_caught():
    """Prove the two tests above actually bite: disable
    `_scrub_long_quoted_spans`, watch the facts.py-shaped secret survive,
    then restore from an in-memory copy (never git).
    """
    original = privacy._scrub_long_quoted_spans
    try:
        privacy._scrub_long_quoted_spans = lambda line: line  # no-op mutation
        secret = "sure, here is a whole paragraph the model actually said back to her"
        line = f"conv=abc123: extracted 0 fact(s) — model returned: {secret!r}"
        out = privacy.scrub_log_line(line, "compactor.log")
        caught = secret not in out
        assert_true(not caught, "mutated (no-op) long-span scrub no longer removes the facts.py snippet")
    finally:
        privacy._scrub_long_quoted_spans = original
    secret = "sure, here is a whole paragraph the model actually said back to her"
    line = f"conv=abc123: extracted 0 fact(s) — model returned: {secret!r}"
    out = privacy.scrub_log_line(line, "compactor.log")
    assert_not_in(secret, out, "restored long-span scrub removes it again")


# =============================================================================
# Round-3 review M2: short whitespace-bearing spans, and repr escape
# handling in _QUOTED_SPAN_RE.
# =============================================================================


def test_short_whitespace_span_is_scrubbed_even_under_16_chars():
    """M2: "Mom died in May" (16 chars, real chat-shaped text) used to
    survive because the old pass only scrubbed spans OVER 16 chars. Any
    quoted span with whitespace is now scrubbed whatever its length.
    """
    secret = "Mom died in May"
    line = f"conv=abc123: extracted 0 fact(s) — model returned: {secret!r}"
    out = privacy.scrub_log_line(line, "compactor.log")
    assert_not_in(secret, out, "a 16-char whitespace-bearing span is scrubbed")
    assert_true("<scrubbed:" in out, "a scrub marker is present")


def test_apostrophe_in_prose_before_a_quoted_span_does_not_shift_pairing():
    """Round-4 blocker: `couldn't store fact 'Mom died in May'` -- the
    apostrophe inside "couldn't" (a contraction, not a quote) used to be
    scanned as the OPENING quote, with the very next `'` (the one right
    before "Mom") closing that bogus span immediately. That left the real
    quoted fact entirely OUTSIDE any matched span, so it survived the
    scrub untouched. A `'` flanked by a letter on both sides must never
    open or close a span.
    """
    secret = "Mom died in May"
    line = f"couldn't store fact '{secret}'"
    out = privacy.scrub_log_line(line, "compactor.log")
    assert_not_in(secret, out, "the real quoted fact is scrubbed despite the contraction before it")
    assert_true("couldn't store fact" in out, "the surrounding prose (with its own apostrophe) survives")
    assert_true("<scrubbed:" in out, "a scrub marker is present")


def test_mutation_reverting_apostrophe_handling_leaks_the_fact():
    """Prove the test above bites: restore a version of `_QUOTED_SPAN_RE`
    that treats every `'` as an equally valid delimiter (the OLD, round-4
    bug), confirm the fact leaks, then restore.
    """
    import re

    original = privacy._QUOTED_SPAN_RE
    try:
        privacy._QUOTED_SPAN_RE = re.compile(r"(['\"])((?:\\.|(?!\1).)*)\1")  # pre-round-4: any ' is a delimiter
        secret = "Mom died in May"
        line = f"couldn't store fact '{secret}'"
        out = privacy.scrub_log_line(line, "compactor.log")
        assert_true(secret in out, "mutated (apostrophe-blind) regex leaks the fact -- confirms the bite")
    finally:
        privacy._QUOTED_SPAN_RE = original
    secret = "Mom died in May"
    line = f"couldn't store fact '{secret}'"
    out = privacy.scrub_log_line(line, "compactor.log")
    assert_not_in(secret, out, "the real (restored) regex scrubs the fact again")


def test_plural_possessive_apostrophe_does_not_shift_pairing():
    """Round-6 LOW: `users' fact 'x y'` -- the trailing apostrophe of a
    plural possessive (a LETTER then `'` then a NON-letter, no letter on
    the closing side at all) is a different shape from the "letter on
    BOTH sides" contraction case round-4 already fixed, and used to be
    scanned as the opening quote the same way, closing at the very next
    `'` and leaving the real fact outside any matched span.
    """
    secret = "x y"
    line = f"users' fact '{secret}'"
    out = privacy.scrub_log_line(line, "compactor.log")
    assert_not_in(secret, out, "the real quoted fact is scrubbed despite the plural possessive before it")
    assert_true("users'" in out, "the plural possessive itself survives unscrubbed")
    assert_true("<scrubbed:" in out, "a scrub marker is present")


def test_non_ascii_letter_flanked_apostrophe_does_not_shift_pairing():
    """Round-6 LOW: `Jose's fact '...'` -- an accented letter next to the
    apostrophe (`[A-Za-z]` alone never matches it) reintroduces the exact
    round-4 bug for any name/word that has one.
    """
    secret = "a secret note"
    line = f"José's fact '{secret}'"
    out = privacy.scrub_log_line(line, "compactor.log")
    assert_not_in(secret, out, "the real quoted fact is scrubbed despite the accented-letter contraction before it")
    assert_true("José's fact" in out, "the surrounding prose (with its own accented apostrophe) survives")
    assert_true("<scrubbed:" in out, "a scrub marker is present")


def test_mutation_reverting_to_ascii_only_letter_flank_leaks_the_fact():
    """Prove the test above bites: restore the ASCII-only (`[A-Za-z]`)
    version of the letter-flank guard, confirm the accented-letter case
    leaks, then restore.
    """
    import re

    original = privacy._QUOTED_SPAN_RE
    try:
        privacy._QUOTED_SPAN_RE = re.compile(
            r'(")((?:\\.|(?!").)*)"'
            r"|"
            r"(?:(?!(?<=[A-Za-z])'(?=[A-Za-z]))(')((?:\\.|[^']|(?<=[A-Za-z])'(?=[A-Za-z]))*)')"
        )  # round-4's own ASCII-only guard -- misses non-ASCII letters and plural possessives alike
        secret = "a secret note"
        line = f"José's fact '{secret}'"
        out = privacy.scrub_log_line(line, "compactor.log")
        assert_true(secret in out, "mutated (ASCII-only) regex leaks the fact -- confirms the bite")
    finally:
        privacy._QUOTED_SPAN_RE = original
    secret = "a secret note"
    line = f"José's fact '{secret}'"
    out = privacy.scrub_log_line(line, "compactor.log")
    assert_not_in(secret, out, "the real (restored) regex scrubs the fact again")


def test_prune_facts_eviction_line_is_scrubbed():
    """M2: facts.py:1053 (v3.1.9) prune_facts' own no-conv_id fallback --
    `logger.warning(f"prune_facts called without conv_id — {n} fact(s)
    evicted with no archive to land in: {shown}{more}")` where `shown`
    joins `repr(f["text"])` for up to 5 evicted facts. Added to this
    module's own audit list this round.
    """
    secret1 = "Mom died in May"
    secret2 = "the dog's name is Biscuit"
    shown = "; ".join(repr(t) for t in (secret1, secret2))
    line = f"prune_facts called without conv_id — 2 fact(s) evicted with no archive to land in: {shown}"
    out = privacy.scrub_log_line(line, "compactor.log")
    assert_not_in(secret1, out, "the first evicted fact's text is gone")
    assert_not_in(secret2, out, "the second evicted fact's text is gone")
    assert_true("prune_facts called without conv_id" in out, "the diagnostic prose survives")
    assert_true("2 fact(s) evicted" in out, "the count survives")


def test_escaped_quote_does_not_shift_the_pairing():
    """M2: `repr("Mom's \\"garden\\"")` picks single quotes and escapes
    only the embedded single quote (`'Mom\\'s "garden"'`) -- the OLD
    `_QUOTED_SPAN_RE`, not knowing about the escape, closed the span at
    that `\\'` and treated everything after it as ordinary text, leaking
    whatever quoted span came next on the line verbatim. Both spans must
    be scrubbed independently now.
    """
    secret1 = 'Mom\'s "garden"'
    secret2 = "a completely different fifty-plus character heading that must not leak either"
    line = f"conv=abc123: heading was {secret1!r} and the next one was {secret2!r}"
    out = privacy.scrub_log_line(line, "compactor.log")
    assert_not_in(secret1, out, "the escaped-quote span itself is gone")
    assert_not_in("garden", out, "no fragment of the escaped-quote span survives")
    assert_not_in(secret2, out, "the FOLLOWING span (what used to leak) is also gone")
    assert_true(out.count("<scrubbed:") == 2, "both spans were scrubbed as two separate spans, not one merged blob")


def test_mutation_reverting_quoted_span_regex_leaks_the_next_span():
    """Prove the test above bites: restore the OLD `_QUOTED_SPAN_RE` (no
    escape handling), confirm the second span leaks, then restore.
    """
    import re

    original = privacy._QUOTED_SPAN_RE
    try:
        privacy._QUOTED_SPAN_RE = re.compile(r"(['\"])((?:(?!\1).)*)\1")  # the old, escape-blind version
        secret1 = 'Mom\'s "garden"'
        secret2 = "a completely different fifty-plus character heading that must not leak either"
        line = f"conv=abc123: heading was {secret1!r} and the next one was {secret2!r}"
        out = privacy.scrub_log_line(line, "compactor.log")
        assert_true(secret2 in out, "mutated (old) regex leaks the span AFTER the escaped quote -- confirms the bite")
    finally:
        privacy._QUOTED_SPAN_RE = original
    secret1 = 'Mom\'s "garden"'
    secret2 = "a completely different fifty-plus character heading that must not leak either"
    line = f"conv=abc123: heading was {secret1!r} and the next one was {secret2!r}"
    out = privacy.scrub_log_line(line, "compactor.log")
    assert_not_in(secret2, out, "restored regex scrubs both spans again")


# =============================================================================
# Round-3 review M3: uvicorn access-log request line allowlist.
# =============================================================================


def test_uvicorn_access_log_method_and_path_survive():
    line = 'INFO:     127.0.0.1:54321 - "POST /v1/chat/completions HTTP/1.1" 200 OK'
    out = privacy.scrub_log_line(line, "compactor.log")
    assert_eq(out, line, "a plain (no-query) access-log request line is untouched")


def test_uvicorn_access_log_id_only_query_survives():
    line = 'INFO:     127.0.0.1:54321 - "GET /v1/chat?conv_id=abc123&limit=5 HTTP/1.1" 200 OK'
    out = privacy.scrub_log_line(line, "compactor.log")
    assert_eq(out, line, "a query string holding only ids/numbers survives untouched")


def test_uvicorn_access_log_non_id_query_value_is_scrubbed():
    line = 'INFO:     127.0.0.1:54321 - "GET /v1/models?token=abcdef123456 HTTP/1.1" 200 OK'
    out = privacy.scrub_log_line(line, "compactor.log")
    assert_true("GET /v1/models?token=<scrubbed:" in out, "a non-id query value is scrubbed")
    assert_true("abcdef123456" not in out, "the non-id query value itself is gone")
    assert_true('HTTP/1.1" 200 OK' in out, "the rest of the line survives")


def test_mutation_disabling_access_log_allowlist_eats_method_and_path():
    """Prove the two tests above bite: disable the access-log allowlist,
    watch a plain request line get replaced wholesale by the generic
    long-quoted-span scrub (losing method/endpoint -- exactly M3's
    complaint about ~6,800 lines per bundle), then restore.
    """
    original = privacy._scrub_access_log_request
    try:
        privacy._scrub_access_log_request = lambda inner: None  # never matches -- falls through
        line = 'INFO:     127.0.0.1:54321 - "POST /v1/chat/completions HTTP/1.1" 200 OK'
        out = privacy.scrub_log_line(line, "compactor.log")
        assert_true("POST /v1/chat/completions HTTP/1.1" not in out,
                    "mutated (disabled) allowlist loses method+endpoint -- confirms the bite")
    finally:
        privacy._scrub_access_log_request = original
    line = 'INFO:     127.0.0.1:54321 - "POST /v1/chat/completions HTTP/1.1" 200 OK'
    out = privacy.scrub_log_line(line, "compactor.log")
    assert_eq(out, line, "restored allowlist keeps method+endpoint again")


# =============================================================================
# Round-3 coordinator addendum: v3.1.9.8 request tracing (req=<hex>, child
# ids req=<hex>.sum1/.facts/.dedup, and a per-request summary line holding
# only ids and numbers). None of it is user content -- it must never be
# scrubbed.
# =============================================================================


def test_request_tracing_line_survives_scrub_log_line():
    line = "req=a1b2c3d4e5f6a1b2 conv=abc123: dedup pass start (child req=a1b2c3d4e5f6a1b2.dedup)"
    out = privacy.scrub_log_line(line, "compactor.log")
    assert_eq(out, line, "an ordinary req=/child-id tracing line is untouched")


def test_request_summary_line_survives_scrub_log_line():
    # "holding only ids and numbers": conv, msgs, phase timings, TTFT,
    # total, tokens, finish, status -- no natural-language content at all.
    line = ("req=a1b2c3d4e5f6a1b2 conv=abc123 msgs=5 sum_ms=12 facts_ms=8 dedup_ms=3 "
            "ttft_ms=120 total_ms=450 tokens=230 finish=stop status=200")
    out = privacy.scrub_log_line(line, "compactor.log")
    assert_eq(out, line, "the per-request summary line is untouched")


def test_request_tracing_lines_survive_scrub_log_text():
    text = (
        "req=a1b2c3d4e5f6a1b2 conv=abc123: dedup pass start\n"
        "req=a1b2c3d4e5f6a1b2.facts conv=abc123: extracted 3 fact(s)\n"
        "req=a1b2c3d4e5f6a1b2 conv=abc123 msgs=5 ttft_ms=120 total_ms=450 tokens=230 "
        "finish=stop status=200\n"
    )
    out = privacy.scrub_log_text(text, "compactor.log")
    assert_eq(out, text, "a whole request's tracing lines survive scrub_log_text byte for byte")


def test_mutation_a_naive_req_scrubber_would_be_caught():
    """Prove the tests above actually bite: simulate a plausible FUTURE
    regression (someone adds an over-broad rule that also eats bare
    `req=...` tokens, mistaking them for a secret-shaped value), confirm
    the request-tracing tests above would fail against it, then restore.
    No current code does this -- this proves the tests are not
    tautological, the same way every other mutation test in this file
    proves its neighbor bites.
    """
    import re as _re

    original = privacy.scrub_log_line
    naive_req_scrub_re = _re.compile(r"req=\S+")

    def _mutated_scrub_log_line(line, source="generic"):
        out = original(line, source)
        return naive_req_scrub_re.sub("req=<scrubbed>", out) if out is not None else out

    try:
        privacy.scrub_log_line = _mutated_scrub_log_line
        line = "req=a1b2c3d4e5f6a1b2 conv=abc123: dedup pass start"
        out = privacy.scrub_log_line(line, "compactor.log")
        assert_true(out != line, "mutated (naive req scrubber) changes the tracing line -- confirms the bite")
    finally:
        privacy.scrub_log_line = original
    line = "req=a1b2c3d4e5f6a1b2 conv=abc123: dedup pass start"
    out = privacy.scrub_log_line(line, "compactor.log")
    assert_eq(out, line, "restored: the tracing line is untouched again")


def test_long_no_space_non_identifier_span_over_16_chars_is_scrubbed():
    """N13 (round-3 mutation harness): a span with NO whitespace but also
    not identifier/path-shaped (contains punctuation `_SAFE_REPR_RE`
    disallows, e.g. commas) -- a plausible shape for a header value or a
    `prior_state` dict-repr (module docstring) -- must still be scrubbed
    once it is over `_LONG_REPR_MIN` (16) chars. Unlike a whitespace-
    bearing span (scrubbed at ANY length, M2), this is the one case that
    still depends on `_LONG_REPR_MIN` itself, so it is the one case a
    length-threshold regression (`_LONG_REPR_MIN` raised from 16 to 60)
    would NOT be caught by the whitespace-first tests above.
    """
    secret = "abc,def,ghi,jkl,mnop"  # 20 chars, no whitespace, has commas (not _SAFE_REPR_RE-shaped)
    assert_true(len(secret) > privacy._LONG_REPR_MIN, "sanity: the secret is over the length floor")
    line = f"conv=abc123: header value was {secret!r}"
    out = privacy.scrub_log_line(line, "compactor.log")
    assert_not_in(secret, out, "a long, non-identifier-shaped, no-space span is scrubbed")


def test_mutation_raising_long_repr_min_lets_the_no_space_span_through():
    """Prove the test above bites: raise `_LONG_REPR_MIN` from 16 to 60
    (N13), watch the 20-char span above survive unscrubbed, then restore.
    """
    original = privacy._LONG_REPR_MIN
    try:
        privacy._LONG_REPR_MIN = 60
        secret = "abc,def,ghi,jkl,mnop"
        line = f"conv=abc123: header value was {secret!r}"
        out = privacy.scrub_log_line(line, "compactor.log")
        assert_true(secret in out, "mutated (_LONG_REPR_MIN=60) lets the 20-char span through -- confirms the bite")
    finally:
        privacy._LONG_REPR_MIN = original
    secret = "abc,def,ghi,jkl,mnop"
    line = f"conv=abc123: header value was {secret!r}"
    out = privacy.scrub_log_line(line, "compactor.log")
    assert_not_in(secret, out, "restored _LONG_REPR_MIN scrubs the span again")


# =============================================================================
# Round-7 BLOCKER: the round-6 "s'/S' guard" was applied a SECOND time
# inside `_QUOTED_SPAN_RE`'s repeated inner content group (not just at the
# opening lookahead). On a line with TWO OR MORE single-quoted items, the
# apostrophe closing the FIRST item (when its content happens to end in a
# word ending "s", e.g. "...cats'") was made invisible there too, so the
# match kept consuming through the separator and swallowed the SECOND
# item's opening quote as ordinary content -- merging both facts into one
# over-wide "quoted span" whose OWN closing quote then landed on the
# second item's actual closer, leaving whatever came after the true end
# of the first item (the separator prose and the second item's leading
# text) sitting INSIDE what render_facts/prune_facts records as content
# that will get spliced back in unscrubbed once the match's boundaries no
# longer line up with the real spans. Fixed by keeping the guard ONLY in
# the opening lookahead (see `_QUOTED_SPAN_RE`'s own comment above).
# Templates below are rendered from the real functions this module's
# docstring already documents auditing (v3.1.9.8's
# `compactor/facts.py:prune_facts`'s no-conv_id fallback and
# `_parse_extraction_output`'s markup-rejection log line), with invented
# synthetic fact text -- never real content.
# =============================================================================


def test_prune_facts_eviction_line_two_single_quoted_items_all_scrubbed():
    """facts.py `prune_facts`'s no-conv_id fallback: `f"prune_facts called
    without conv_id — {n} fact(s) evicted with no archive to land in:
    {shown}"` where `shown = "; ".join(repr(f["text"]) for f in
    evicted[:5])`. With >=2 evicted facts this renders as TWO OR MORE
    single-quoted spans separated by "; " on one line -- exactly the
    round-6 regression shape.
    """
    facts_text = [
        "she has two pet cats",
        "her mother lives in Ogden",
        "she is afraid of hospitals",
    ]
    shown = "; ".join(repr(t) for t in facts_text)
    line = (
        f"prune_facts called without conv_id — {len(facts_text)} fact(s) "
        f"evicted with no archive to land in: {shown}"
    )
    out = privacy.scrub_log_line(line, "compactor.log")
    for secret in facts_text:
        assert_not_in(secret, out, f"evicted fact {secret!r} is scrubbed")
    assert_eq(out.count("<scrubbed:"), len(facts_text), "every evicted item got its own scrub marker")
    assert_true("prune_facts called without conv_id" in out, "the diagnostic prose survives")
    assert_true(f"{len(facts_text)} fact(s) evicted" in out, "the count survives")


def test_markup_rejection_line_two_single_quoted_items_all_scrubbed():
    """facts.py `_parse_extraction_output`'s markup-rejection log line:
    `f"{prefix}extraction returned {n} line(s) of markup rather than
    facts; not stored — " + "; ".join(f"{reason}: {line[:60]!r}" for
    reason, line in rejected[:3])`. With >=2 rejected lines this is again
    TWO OR MORE single-quoted spans separated by "; " on one line.
    """
    rejected = [
        ("heading", "## Things she loves"),
        ("bullet-only", "- she walks the dogs every morning at the park"),
        ("table", "| she misses her father a great deal |"),
    ]
    shown = "; ".join(f"{reason}: {text[:60]!r}" for reason, text in rejected)
    line = (
        f"conv=abc123: extraction returned {len(rejected)} line(s) of markup "
        f"rather than facts; not stored — {shown}"
    )
    out = privacy.scrub_log_line(line, "compactor.log")
    for _, text in rejected:
        assert_not_in(text[:60], out, f"rejected item {text[:60]!r} is scrubbed")
    assert_eq(out.count("<scrubbed:"), len(rejected), "every rejected item got its own scrub marker")
    assert_true("extraction returned 3 line(s) of markup" in out, "the diagnostic prose survives")
    assert_true("conv=abc123" in out, "the conv id survives (diagnostic, not content)")


def test_mutation_reverting_to_round6_inner_guard_leaks_the_second_item():
    """Prove the two tests above bite: restore the round-6 `_QUOTED_SPAN_RE`
    (the s'/S' guard applied a second time inside the repeated inner
    content group, not just at the opening), confirm the second item on a
    two-single-quoted-item line leaks, then restore the round-7 fix.
    """
    import re

    original = privacy._QUOTED_SPAN_RE
    round6_re = re.compile(
        r'(")((?:\\.|(?!").)*)"'
        r"|"
        r"(?:(?!(?<=[^\W\d_])'(?=[^\W\d_])|(?<=[sS])'(?![^\W\d_]))"
        r"(')((?:\\.|[^']|(?<=[^\W\d_])'(?=[^\W\d_])|(?<=[sS])'(?![^\W\d_]))*)')"
    )
    facts_text = [
        "she has two pet cats",
        "her mother lives in Ogden",
        "she is afraid of hospitals",
    ]
    shown = "; ".join(repr(t) for t in facts_text)
    line = (
        f"prune_facts called without conv_id — {len(facts_text)} fact(s) "
        f"evicted with no archive to land in: {shown}"
    )
    try:
        privacy._QUOTED_SPAN_RE = round6_re
        out = privacy.scrub_log_line(line, "compactor.log")
        leaked = any(secret in out for secret in facts_text[1:])
        assert_true(leaked, "mutated (round-6) regex leaks a later item on a two-single-quoted-item line -- confirms the bite")
    finally:
        privacy._QUOTED_SPAN_RE = original
    out = privacy.scrub_log_line(line, "compactor.log")
    for secret in facts_text:
        assert_not_in(secret, out, "the real (round-7 fixed) regex scrubs every item again")


def test_real_bundles_facts_shapes_zero_leaks_if_present():
    """Counts only, never content (privacy hard rule) -- proves the two
    REVIEW2 R2-B4 leaking shapes survive scrubbed=0 on whatever real
    bundles are present on this machine. Skipped honestly if none are.
    """
    import glob
    import re

    paths = sorted(glob.glob("/home/drew/pod-exports/*/*/zla-bundle-*/logs/zla/compactor.log"))
    if not paths:
        print("  skip real_bundles_facts_shapes_zero_leaks_if_present: no bundles on this machine")
        return
    shape1 = re.compile(r"extraction returned \d+ line\(s\) of markup")
    shape2 = re.compile(r"extracted 0 fact\(s\) — model returned:")
    for path in paths:
        raw = Path(path).read_text(encoding="utf-8", errors="replace")
        scrubbed = privacy.scrub_log_text(raw, source="compactor.log")
        lines = scrubbed.splitlines()
        shape1_lines = [l for l in lines if shape1.search(l)]
        shape2_lines = [l for l in lines if shape2.search(l)]
        # A line matching either shape must either carry a scrub marker or
        # have had nothing >16 chars to scrub in the first place (proven
        # separately by the synthetic tests above) -- here we only assert
        # no long (>16 char) quoted span survives unscrubbed.
        # round-4: `_QUOTED_SPAN_RE` now has two independent group pairs
        # (1,2 for `"`, 3,4 for `'` -- see its own docstring), so a match's
        # inner text is whichever pair actually populated.
        def _inner(m):
            return m.group(2) if m.group(1) is not None else m.group(4)

        leak1 = sum(1 for l in shape1_lines if privacy._QUOTED_SPAN_RE.search(l) and "<scrubbed:" not in l
                    and any(len(_inner(m)) > 16 and not privacy._SAFE_REPR_RE.match(_inner(m))
                            for m in privacy._QUOTED_SPAN_RE.finditer(l)))
        leak2 = sum(1 for l in shape2_lines if privacy._QUOTED_SPAN_RE.search(l) and "<scrubbed:" not in l
                    and any(len(_inner(m)) > 16 and not privacy._SAFE_REPR_RE.match(_inner(m))
                            for m in privacy._QUOTED_SPAN_RE.finditer(l)))
        print(f"  ok   {path}: shape1 lines={len(shape1_lines)} leaked={leak1}; "
              f"shape2 lines={len(shape2_lines)} leaked={leak2}")
        assert_eq(leak1, 0, f"{path}: zero shape1 (extraction-rejected-markup) leaks")
        assert_eq(leak2, 0, f"{path}: zero shape2 (zero-fact-snippet) leaks")


# =============================================================================
# Round-7 coordinator addendum: v3.1.9.8 line shapes that hold only ids
# and numbers -- never her text -- from feature/v3.1.9.8 (read from
# `/home/drew/code/zl-v3198`, read-only): request tracing's background
# form and summary line (reqctx.py), the per-request conv_id/source line
# (main.py), and the word-ban warm-up loop's start/end lines
# (wordban_runtime.py). None of these must ever be scrubbed, including
# when a field on the line is itself quoted (`vllm_identity={...!r}`).
# =============================================================================


def test_v3198_req_summary_line_status_ok_survives():
    line = ("request done req=a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4 conv=abc123 msgs=12 status=ok "
            "t_images=0.001 t_compact=0.200 t_retrieval=0.050 t_inject=0.010 t_guard=0.002 "
            "t_ttft=1.300 t_total=9.800 tokens_out=412 finish=stop")
    out = privacy.scrub_log_line(line, "compactor.log")
    assert_eq(out, line, "the request-done summary line (status=ok) is untouched")


def test_v3198_req_summary_line_status_error_survives():
    line = ("request done req=a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4 conv=abc123 msgs=3 status=error "
            "t_images=0.001 t_total=0.400")
    out = privacy.scrub_log_line(line, "compactor.log")
    assert_eq(out, line, "the request-done summary line (status=error) is untouched")


def test_v3198_req_summary_line_status_cancelled_survives():
    line = ("request done req=a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4 conv=abc123 msgs=7 status=cancelled "
            "t_total=2.100")
    out = privacy.scrub_log_line(line, "compactor.log")
    assert_eq(out, line, "the request-done summary line (status=cancelled) is untouched")


def test_v3198_req_summary_line_with_wb_field_survives():
    # reqctx.RequestTrace.finish appends `wb=<revision>:<sha8>` last, only
    # when the reply was made under the deployed word-ban file's grammar.
    line = ("request done req=a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4 conv=abc123 msgs=5 status=ok "
            "t_total=3.500 tokens_out=88 finish=stop wb=7:9f8e7d6c")
    out = privacy.scrub_log_line(line, "compactor.log")
    assert_eq(out, line, "the request-done summary line with a wb= field is untouched")


def test_v3198_req_background_form_survives():
    line = "req=a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4/bg:tail conv=abc123: background tail running"
    out = privacy.scrub_log_line(line, "compactor.log")
    assert_eq(out, line, "the background req= form (req=<id>/bg:<tag>) is untouched")


def test_v3198_conv_id_source_header_line_survives():
    line = "conv_id=abc123 source=header msgs=5 ow_chat=chat-xyz-789 ow_msg=msg-uvw-456"
    out = privacy.scrub_log_line(line, "compactor.log")
    assert_eq(out, line, "the conv_id/source/ow_chat/ow_msg line is untouched")


def test_v3198_conv_id_source_header_line_without_ow_ids_survives():
    line = "conv_id=abc123 source=header msgs=5"
    out = privacy.scrub_log_line(line, "compactor.log")
    assert_eq(out, line, "the conv_id/source line with no ow_chat/ow_msg fields is untouched")


def test_v3198_wordban_warmup_starting_line_survives():
    # wordban_runtime._FileWarmer._warm: `vllm_identity={key[1]!r}` is an
    # opaque, identifier-shaped engine-process id -- quoted (repr'd), but
    # still not her text.
    line = ("word-ban warm-up starting: revision=7 value_sha256=9f8e7d6c… "
            "vllm_identity='vllm-0:pid-48213' (n=3)")
    out = privacy.scrub_log_line(line, "compactor.log")
    assert_eq(out, line, "the warm-up starting line (with a quoted, id-shaped vllm_identity) is untouched")


def test_v3198_wordban_warmup_done_line_survives():
    line = "word-ban warm-up done: revision=7 value_sha256=9f8e7d6c… compile_s=4.2"
    out = privacy.scrub_log_line(line, "compactor.log")
    assert_eq(out, line, "the warm-up done line is untouched")


def test_v3198_wordban_request_id_header_shape_survives():
    # The X-Request-Id header wordban_runtime sends for the warm-up call
    # itself (`warmup-<sha8>-<n>`) can show up echoed back in an access
    # log style line too.
    line = 'INFO:     127.0.0.1:54321 - "POST /v1/chat/completions HTTP/1.1" 200 OK  # X-Request-Id: warmup-9f8e7d6c-3'
    out = privacy.scrub_log_line(line, "compactor.log")
    assert_eq(out, line, "a warmup-<sha8>-<n> id embedded in an otherwise-benign line is untouched")


def test_v3198_lines_survive_scrub_log_text_byte_for_byte():
    text = (
        "req=a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4/bg:tail conv=abc123: background tail running\n"
        "conv_id=abc123 source=header msgs=5 ow_chat=chat-xyz-789 ow_msg=msg-uvw-456\n"
        "word-ban warm-up starting: revision=7 value_sha256=9f8e7d6c… vllm_identity='vllm-0:pid-48213' (n=3)\n"
        "word-ban warm-up done: revision=7 value_sha256=9f8e7d6c… compile_s=4.2\n"
        "request done req=a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4 conv=abc123 msgs=12 status=ok "
        "t_total=9.800 tokens_out=412 finish=stop wb=7:9f8e7d6c\n"
    )
    out = privacy.scrub_log_text(text, "compactor.log")
    assert_eq(out, text, "a whole v3.1.9.8 request's lines survive scrub_log_text byte for byte")


def test_invalid_image_demoted_shape_line_survives_for_every_shape_value():
    """.8 thread: a new compactor log line, `N invalid image(s) demoted …
    shape=<uuid36|owui-files-path|http|data-malformed|other> len=N` --
    carries no URL content (the brief's own description), only a small,
    fixed vocabulary of shape labels and integers. Every declared shape
    value must survive both `scrub_log_line` and `scrub_log_text` byte
    for byte -- this is diagnostic telemetry, not something scrubbed.
    """
    for shape in ("uuid36", "owui-files-path", "http", "data-malformed", "other"):
        line = (
            f"2026-09-27 00:00:00,000 compactor INFO 3 invalid image(s) demoted "
            f"… shape={shape} len=128"
        )
        out = privacy.scrub_log_line(line, "compactor.log")
        assert_eq(out, line, f"scrub_log_line leaves the {shape!r}-shaped demotion line untouched")
        text = line + "\n"
        out_text = privacy.scrub_log_text(text, "compactor.log")
        assert_eq(out_text, text, f"scrub_log_text leaves the {shape!r}-shaped demotion line untouched")


def test_v3198_wordban_warmup_does_not_mask_a_real_quoted_secret_on_the_same_line():
    """Defense in depth, the other direction: the warm-up lines' quoted
    `vllm_identity` field must not become a blanket excuse to skip
    scrubbing -- if a line happens to ALSO carry a genuinely risky quoted
    span (not one of the audited shapes), that span is still scrubbed.
    This does not happen in the real warm-up code path (vllm_identity is
    the only quoted field on that line), so this is a synthetic
    composition proving the scrubber has no special-case allowlist keyed
    on the words "warm-up" or "vllm_identity" that would blind it to
    something else quoted on the same line.
    """
    secret = "she told the assistant her home address"
    line = f"word-ban warm-up starting: vllm_identity='vllm-0:pid-48213' note='{secret}' (n=3)"
    out = privacy.scrub_log_line(line, "compactor.log")
    assert_not_in(secret, out, "an unrelated risky quoted span on the same line is still scrubbed")
    assert_true("vllm_identity='vllm-0:pid-48213'" in out, "the id-shaped quoted field still survives")


# =============================================================================
# Runner
# =============================================================================

_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]

if __name__ == "__main__":
    for t in _TESTS:
        print(f"-- {t.__name__}")
        t()
    print(f"\nAll {len(_TESTS)} privacy tests passed.")
