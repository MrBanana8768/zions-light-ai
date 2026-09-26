"""privacy.py — the ONE log-content scrubber. `zl bundle` and `zl dash`'s
log panel both import THIS module; neither re-implements a filter of its
own (two copies of one privacy rule drifting apart is this codebase's most
expensive recurring defect — see memory.md).

Why a scrubber, not a line-level allow/deny filter: the previous approach
(`telemetry.privacy_filter_line`, now replaced) dropped a whole line if it
contained any of a few substrings like "content". Checked against a real
bundle's `compactor.log`, that dropped **30% of every WARNING/ERROR line**
(265 of 876 in one real bundle) because ordinary diagnostic text ("content
window", "content budget") contains the word "content" with no user text
anywhere near it — exactly the failure the hostile review's "only ever
shows openwebui-error.log Traceback headers" finding describes. A
substring-drop filter cannot tell "the word content" from "her content";
this module scrubs the SPAN that is actually data-bearing and keeps the
diagnostic prose around it, which is both safer (a scrubbed span can never
leak) and more useful (the line survives).

What actually carries her text, grepped from `compactor/*.py` (worktree)
and `git show v3.1.9:compactor/*.py` (prod), and from real bundles under
`/home/drew/pod-exports/*/bundle*/zla-bundle-*/logs/zla/` (line SHAPES
only — never content, see the module's own tests for the synthetic
stand-ins):

  - OpenWebUI/SQLAlchemy: a failed write (e.g. "database is locked", the
    known MooseFS failure mode) logs `[SQL: ...]` (often wrapped across
    several physical lines, ending in a lone `]`) immediately followed by
    `[parameters: (...)]` — the literal bound values, i.e. her message
    text when the statement is a chat insert/update. Present in 2 of 4
    real bundles checked (openwebui.log AND openwebui-error.log both, at
    material volume — hundreds of lines per bundle).
  - compactor/dedup.py's `_dedup_pass` (~line 776): `logger.info(f"dedup:
    merging {n} fact(s) into {merged_text!r}; removed: {removed_texts!r}")`
    — both the merged fact text and every fact it replaced, verbatim,
    repr'd, at INFO.
  - compactor/main.py's `decide_memory_tail` (~line 4759 in the review's
    numbering; the log call itself is the `logger.warning(f"conv={id}:
    {decision.reason} — skipping memory tail (...)")` a few hundred lines
    below `reply_is_degenerate`): `decision.reason` can be `f"reply looks
    like a repetition loop (the token {tok[:24]!r} repeated for {n}
    characters (limit {limit}))"` — up to 24 characters of her reply,
    repr'd, at WARNING.

Two entry points:

  - `scrub_log_line(line, source) -> str | None` — one already-split
    line. Returns the line with every risky span replaced by
    `<scrubbed:N chars>` (N = the length of what was removed), or the
    line unchanged if nothing risky was found, or None if the whole line
    is not worth keeping once scrubbed (nothing diagnostic survives —
    e.g. a bare continuation line that is nothing but part of a
    multi-line SQL statement with no marker of its own; `scrub_log_text`
    is what actually assembles those into one scrubbed record, so
    `scrub_log_line` alone drops rarely).
  - `scrub_log_text(text, source) -> str` — a whole file/tail. Walks
    `[SQL: ...]` blocks (which can span several physical lines before the
    closing `]`) and Python tracebacks as UNITS, not line-by-line, so a
    SQL statement broken across three lines is scrubbed as the one record
    it is rather than three separately-inspected fragments (the
    hostile-review requirement this module exists to satisfy: "must
    handle multi-line records ... that span lines").

`source` is a free-form label (a filename like "compactor.log" or
"openwebui.log", or "generic") — the patterns here are not source-specific
today (the risky shapes above are distinctive enough on their own), but the
parameter is kept so a future source-specific rule has somewhere to live
without changing the call signature every caller already uses.

Style: stdlib only, no bare `except:`.
"""

from __future__ import annotations

import re

# =============================================================================
# Bracketed SQL/parameters blocks (OpenWebUI / SQLAlchemy)
# =============================================================================

_SQL_OPEN_RE = re.compile(r"\[SQL:")
_PARAMS_OPEN_RE = re.compile(r"\[parameters:")
# A block is "closed" once a line ends with a `]` that isn't immediately
# preceded by an escaped/quoted context we'd need a real parser for — good
# enough here because SQLAlchemy always emits the block's own closing `]`
# at the end of a physical line.
_CLOSES_RE = re.compile(r"\]\s*$")


def _scrub_bracket_span(line: str, marker: str) -> str:
    """Replace `marker ... ]` (single physical line, closed on this line)
    with `marker <scrubbed:N chars>]`. Caller guarantees `marker` and a
    closing `]` are both present on `line`.
    """
    start = line.index(marker)
    end = line.rindex("]")
    if end <= start:
        return line
    inner_start = start + len(marker)
    inner = line[inner_start:end]
    return f"{line[:inner_start]} <scrubbed:{len(inner)} chars>{line[end:]}"


# =============================================================================
# Specific known content-bearing shapes (compactor)
# =============================================================================

# dedup.py: `dedup: merging N fact(s) into '<text>'; removed: ['<text>', ...]`
# `!r` on a str uses single quotes unless the text itself contains a single
# quote and no double quote, in which case repr switches to double quotes —
# handle both.
_DEDUP_MERGE_RE = re.compile(
    r"(dedup: merging \d+ fact\(s\) into )(['\"]).*?\2(; removed: )(\[.*\])(\s*)$"
)

# main.py: `... reply looks like a repetition loop (the token '<=24 chars>'`
# `repeated for N characters (limit N))`
_REPETITION_TOKEN_RE = re.compile(
    r"(the token )(['\"]).*?\2( repeated for \d+ characters)"
)


def _scrub_dedup_merge(line: str) -> str:
    m = _DEDUP_MERGE_RE.search(line)
    if not m:
        return line
    quote = m.group(2)
    into_start = m.end(1)
    # Find the matching closing quote for the `into '...'` span: search from
    # just after the opening quote up to the start of `; removed: `.
    removed_marker_at = line.index(m.group(3))
    close_quote_at = line.rfind(quote, into_start, removed_marker_at)
    if close_quote_at <= into_start:
        return line
    into_span = line[into_start : close_quote_at + 1]
    removed_start = removed_marker_at + len(m.group(3))
    removed_span = m.group(4)
    return (
        line[:into_start]
        + f"<scrubbed:{len(into_span)} chars>"
        + m.group(3)
        + f"<scrubbed:{len(removed_span)} chars>"
        + line[removed_start + len(removed_span) :]
    )


def _scrub_repetition_token(line: str) -> str:
    m = _REPETITION_TOKEN_RE.search(line)
    if not m:
        return line
    quote = m.group(2)
    prefix_end = m.end(1)
    close_quote_at = line.find(quote, prefix_end + 1)
    if close_quote_at < 0:
        return line
    token_span = line[prefix_end : close_quote_at + 1]
    return line[:prefix_end] + f"<scrubbed:{len(token_span)} chars>" + line[close_quote_at + 1 :]


# =============================================================================
# Generic defense-in-depth: a quoted span right after a risky verb, for any
# future call site nobody has audited yet. Deliberately broad (see the
# module docstring: a false-positive scrub is free, a leaked span is not).
# =============================================================================

_RISKY_VERB_RE = re.compile(
    r"\b(into|removed|content|text|reply|message|fact|facts|summary|prompt)\s*[:=]\s*(['\"])",
    re.IGNORECASE,
)


def _scrub_generic_quoted(line: str) -> str:
    out = []
    i = 0
    for m in _RISKY_VERB_RE.finditer(line):
        if m.start() < i:
            continue  # inside a span we already emitted
        out.append(line[i : m.end()])
        quote = m.group(2)
        close_at = line.find(quote, m.end())
        if close_at < 0:
            i = m.end()
            continue
        span = line[m.end() : close_at]
        out.append(f"<scrubbed:{len(span)} chars>{quote}")
        i = close_at + 1
    out.append(line[i:])
    return "".join(out)


# =============================================================================
# R2-B4 (round-2 review): allowlist-shaped scrub for compactor*.log. Two
# known-bad shapes slipped past every specific pattern above:
#   - facts.py:1543 (v3.1.9) / :1571 (worktree): logger.info(f"{prefix}
#     extraction returned {len(rejected)} line(s) of markup rather than
#     facts; not stored — {shown}{more}") where `shown` embeds up to 3 ×
#     60 chars of the model's actual output via `{line[:60]!r}`.
#   - facts.py:1706 (v3.1.9) / :1744 (worktree): logger.info(f"{where}:
#     extracted 0 fact(s) — model returned: {snippet!r}") -- up to 120
#     chars of what she sent.
# Both are INFO; neither matched `_RISKY_VERB_RE` ("markup" / "returned:"
# are not in its verb list).
#
# Rather than adding these two shapes to the specific list above (this
# review: "keeping a list of known-bad shapes has already missed two call
# sites"), compactor*.log gets a THIRD, allowlist-shaped pass: any quoted
# span (single OR double quote -- Python's own `!r` repr uses single
# unless the text itself has one) longer than `_LONG_REPR_MIN` chars is
# scrubbed UNLESS its content is "safe shaped" (`_SAFE_REPR_RE`) -- an
# identifier/path/state/mime-type/count with NO whitespace. Real chat
# text (a reply, a fact, a markup fragment, a model snippet) always
# contains at least one space well within 16 chars; a conv_id, state
# name, header value, or type name never does. Default-DENY, same
# posture as `common.redact_env`: anything not provably an identifier is
# scrubbed, not the other way around.
#
# Swept every `logger.<level>(...)` call in `git show v3.1.9:compactor/
# *.py` and the worktree's `compactor/*.py` that interpolates `!r` or a
# `key: '...'`/`key="..."` quoted span (136 call sites; see the
# implementer's audit script/output for round 3). Coverage:
#
#   COVERED by this pass (arbitrary-length content, `!r` or quoted, now
#   scrubbed whenever it has whitespace, whatever its length, or is
#   >16 chars and not identifier-shaped -- see M2 below for the
#   whitespace widening):
#     facts.py:1053 (v3.1.9)/:1090 (worktree) prune_facts' own
#     no-conv_id fallback: `logger.warning(f"prune_facts called without
#     conv_id — {len(evicted)} fact(s) evicted with no archive to land
#     in: {shown}{more}")` where `shown = "; ".join(repr(f["text"]) for f
#     in evicted[:5])` -- up to 5 evicted facts' text, repr'd, at
#     WARNING. Round-3 review M2: a short fact text ("Mom died in May",
#     16 chars) used to survive here because the old pass only scrubbed
#     spans OVER 16 chars; the whitespace rule below catches it now
#     regardless of length.
#     facts.py (the two shapes above); dedup.py:776/748 (`merged_text!r`,
#     `removed_texts!r` -- also already caught by `_scrub_dedup_merge`
#     above; this pass is defense-in-depth for it); main.py's repetition-
#     loop token (also already caught by `_scrub_repetition_token`
#     above); memory.py:114 (`raw!r` -- a sanitized-empty HEADER VALUE,
#     which can carry arbitrary client-supplied text); bgwork.py:160
#     (`exc!r` -- an exception's own args can embed arbitrary text
#     depending on what raised it); backfill.py's `prior_state.get(...)`
#     dict-repr lines when the state value is long and non-identifier.
#
#   NOT covered by this pass (checked, and safe by shape or genuinely out
#   of scope -- not silently missed):
#     `conv_id` -- used unquoted almost everywhere, and where quoted is
#     alnum/underscore/dash-only (`memory.py`'s own `UnsafeConvId` check
#     enforces `[A-Za-z0-9_-]`) -- matches `_SAFE_REPR_RE`, correctly left
#     visible: the brief requires conv ids stay for diagnosis.
#     `type(e).__name__` / bare `{e}` exception messages with NO quotes
#     around the whole thing -- out of scope for a QUOTED-span scrub;
#     these are overwhelmingly OSError/sqlite3 system text, not user
#     content, and a no-quote-required pass here would eat nearly every
#     warning/error line in the log, repeating the 30% false-positive-
#     drop failure this module's own docstring describes for the retired
#     line-drop filter. Flagged, not fixed, this round -- see the
#     implementer's final report.
#     `backup.py`'s `REMOTE_TARGET!r` (an operator-configured env value,
#     path/URL-shaped -- matches `_SAFE_REPR_RE` whenever it has no
#     spaces, which every real path/URL value does).
#     `selftest.py`'s `content-type!r`, `webuidb.py`'s `local_chats!r`
#     (an int repr) -- always well under 16 chars, never actually
#     scrubbed, listed for completeness.
#     `persona.py`'s `source='...'` values -- a short fixed vocabulary,
#     always <16 chars.
# =============================================================================

_SAFE_REPR_RE = re.compile(r"^[A-Za-z0-9_./:+=-]*$")
_LONG_REPR_MIN = 16
# Review M2 (round 3): the OLD version was `(['\"])((?:(?!\1).)*)\1` --
# "any char that is not the opening quote". That does not know about
# Python `repr()`'s own backslash escapes: `repr("Mom's \"garden\"")`
# produces `'Mom\'s "garden"'` (repr picks single quotes here because the
# text has BOTH quote types, and escapes only the embedded SINGLE quote --
# see Python's own reprlib rule). The old regex, scanning for the next
# UNESCAPED-looking `'`, closed the span at the `\'` inside "Mom\'s" --
# the backslash is just another "not a quote" char to it, so it never
# recognized the escape -- and treated the REST of the line (starting
# from "s \"garden\"'...") as ordinary text, exposing whatever quoted span
# came after it verbatim. `(?:\\.|(?!\1).)*` fixes this the standard way:
# at each position, either consume a backslash-escape pair as ONE unit
# (so `\'`/`\"`/`\\`/`\n`/... can never be mistaken for the closing quote)
# or consume one character that is not the opening quote.
#
# Round-4 privacy bug: a single quote is also an ordinary English
# apostrophe (contraction/possessive: couldn't, it's, y'all, Mom's), and
# the original version of this pattern had no way to tell the two apart
# -- it treated ANY `'` as a candidate span delimiter. In prose like
# `couldn't store fact 'Mom died in May'`, the apostrophe in "couldn't"
# was scanned as the OPENING quote, and the very next `'` (the one right
# before "Mom") closed that (bogus) span immediately -- leaving the real
# quoted fact ("Mom died in May") entirely outside any matched span, so
# it survived this scrub untouched.
#
# Fix: a `'` with a LETTER on both sides never counts as an opening or
# closing delimiter -- it is invisible to span-finding, exactly as if it
# were not there. This only applies to `'` (an apostrophe is never
# ambiguous with `"`, which has no such double duty in English prose), so
# the pattern is now two alternatives: `"..."` unchanged, and `'...'`
# with that extra guard on both the opening quote (a leading negative
# lookahead: refuse to open right here if THIS `'` is letter-flanked) and
# every `'` encountered while consuming the inside (the third arm of the
# inner alternation: a letter-flanked `'` is consumed as an ORDINARY
# character, not treated as the closer -- so only a `'` that is NOT
# letter-flanked can ever end the span). Two separate capture-group pairs
# (1,2 for `"`, 3,4 for `'`) since the two branches need independent
# "is this the opening quote" assertions; callers use whichever pair is
# not None (`m.group(1) or m.group(3)`, `m.group(2) or m.group(4)`).
# Round-6 LOW: two more apostrophe false negatives, both cheap extensions
# of the same "invisible apostrophe" idea above, NOT a rewrite of it:
#   - non-ASCII letters ("Jose's" with an accented e): `[A-Za-z]` never
#     matched them, so the ORIGINAL bug (any letter-flanked `'` scanned as
#     a real delimiter) came right back for any name/word with an accented
#     letter next to it. `[^\W\d_]` (a "word" character that is not a
#     digit or underscore) is Python re's own idiom for "a Unicode letter"
#     with no extra flags needed, so this is a one-token swap, not a
#     rewrite.
#   - plural possessive ("users'", "James'" -- a LETTER then `'` then a
#     NON-letter, no letter on the far side at all): the existing
#     "letters on BOTH sides" guard does not cover this shape (there is no
#     letter after the apostrophe), so `users' fact 'x y'` still opened
#     its bogus span at the "users'" apostrophe, exactly the round-4
#     failure mode, just for a different apostrophe shape. Extended with a
#     SECOND guard: an apostrophe preceded by `s`/`S` and NOT followed by
#     a letter is ALSO invisible -- but ONLY at the OPENING position (the
#     leading negative lookahead). Round-6 shipped this same guard a
#     SECOND time inside the repeated inner content group, on the theory
#     (recorded in that revision's comment here, since removed) that
#     regex backtracking would make it "safely" fall back to a real
#     closing quote whenever one existed later on the line. That theory
#     was wrong in practice: round-7's hostile review (priv_variant.py)
#     found real lines with TWO OR MORE single-quoted items where the
#     inner-group copy of the guard made the apostrophe ending the FIRST
#     item's content (e.g. "...into 'She has two cats'; 'Her mother..."
#     -- the `s` before that closing `'` is the last letter of "cats")
#     invisible too, so the match kept consuming and swallowed the
#     SEPARATOR and the entire NEXT quoted item as if it were still
#     inside the first span, merging two facts into one and letting
#     everything between the true open and the eventual real close leak
#     unscrubbed once "scrubbed" text is spliced back in around it. The
#     guard is now applied ONLY when deciding whether THIS position may
#     open a span; once a span is open, an `s'`-shaped apostrophe is
#     always treated as a real closing quote, exactly like any other `'`.
#     This still fully covers the plural-possessive false-open case (the
#     guard still stops "users'" from opening a bogus span in the first
#     place) without the backtracking hazard above. The remaining true
#     edge case (a value that itself ends in a word ending "...s" with
#     nothing else quoted afterward on the same line, where that `'` is
#     BOTH the value's own apostrophe and the real closer) is unchanged
#     from round-6 and still resolved the safe way: it closes the span
#     there, which is the correct reading anyway (there is no way to
#     tell, from the line alone, that the apostrophe was "supposed" to be
#     part of the content rather than the closing delimiter).
_QUOTED_SPAN_RE = re.compile(
    r'(")((?:\\.|(?!").)*)"'
    r"|"
    r"(?:(?!(?<=[^\W\d_])'(?=[^\W\d_])|(?<=[sS])'(?![^\W\d_]))"
    r"(')((?:\\.|[^']|(?<=[^\W\d_])'(?=[^\W\d_]))*)')"
)
_HAS_WHITESPACE_RE = re.compile(r"\s")

# Review M3: uvicorn's own access-log request line, always inside quotes:
# `"METHOD /path?query HTTP/1.1"`. Allowlisted here (checked BEFORE the
# whitespace rule below, which would otherwise scrub every one of these --
# the method/path/version are separated by spaces) so method and endpoint
# survive; only a query string is scrubbed, and only the values in it that
# are not plainly an id (see `_scrub_query_string`).
_ACCESS_LOG_RE = re.compile(
    r"^(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS|TRACE|CONNECT) (\S+) (HTTP/\d(?:\.\d)?)$"
)
_QS_ID_VALUE_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_QS_ID_KEY_RE = re.compile(r"(^|_)id$", re.IGNORECASE)
_QS_NUMERIC_RE = re.compile(r"^[0-9]+$")


def _scrub_query_string(qs: str) -> str:
    """`qs` excludes the leading `?`. A value survives ONLY if it is
    plainly an id, not merely id-SHAPED: the key itself names an id
    (`conv_id`, `chat_id`, ... -- ends in `id`) and the value is
    alnum/dash/underscore, or the value is purely numeric (an offset,
    a limit, a count). Everything else -- notably `token=`/`api_key=`/
    `password=`-shaped params, whose VALUES are just as alnum-shaped as a
    real id -- is scrubbed (review M3: "scrub a query string only if it
    holds anything beyond ids"; character shape alone is not enough to
    call something an id, or every bearer token would qualify too).
    """
    parts = []
    for part in qs.split("&"):
        key, sep, value = part.partition("=")
        if not sep:
            parts.append(part if _QS_NUMERIC_RE.match(key) else f"<scrubbed:{len(part)} chars>")
            continue
        is_id = (_QS_ID_KEY_RE.search(key) and _QS_ID_VALUE_RE.match(value)) or _QS_NUMERIC_RE.match(value)
        parts.append(part if is_id else f"{key}=<scrubbed:{len(value)} chars>")
    return "&".join(parts)


def _scrub_access_log_request(inner: str) -> str | None:
    """`inner` is the text between the quotes. Returns the (possibly
    query-scrubbed) request line if `inner` is shaped like a uvicorn
    access-log request, else None (not this shape -- caller falls through
    to the generic quoted-span rule).
    """
    m = _ACCESS_LOG_RE.match(inner)
    if not m:
        return None
    method, target, version = m.group(1), m.group(2), m.group(3)
    path, qmark, qs = target.partition("?")
    if qmark:
        target = f"{path}?{_scrub_query_string(qs)}"
    return f"{method} {target} {version}"


def _scrub_long_quoted_spans(line: str) -> str:
    """Allowlist-shaped fallback for compactor*.log (review R2-B4, widened
    by M2/M3 in round 3): a quoted span is kept ONLY if it is a uvicorn
    access-log request line (M3 -- method/endpoint survive, a non-id query
    string does not), or if it has NO whitespace AND (is `_LONG_REPR_MIN`
    chars or shorter, or is identifier/path shaped). Everything else is
    scrubbed, whatever verb (if any) precedes it and whatever its length
    -- review M2: real chat text ("Mom died in May", a `prune_facts`
    eviction text) is exactly as risky at 16 chars as at 116; the OLD
    length floor only ever caught the long ones.
    """

    def _sub(m: re.Match) -> str:
        # group(1)/(2) from the `"` branch, group(3)/(4) from the `'`
        # branch -- see `_QUOTED_SPAN_RE`'s own docstring/comment.
        quote = m.group(1) or m.group(3)
        inner = m.group(2) if m.group(1) is not None else m.group(4)
        access_log = _scrub_access_log_request(inner)
        if access_log is not None:
            return f"{quote}{access_log}{quote}"
        if _HAS_WHITESPACE_RE.search(inner):
            return f"{quote}<scrubbed:{len(inner)} chars>{quote}"
        if len(inner) <= _LONG_REPR_MIN or _SAFE_REPR_RE.match(inner):
            return m.group(0)
        return f"{quote}<scrubbed:{len(inner)} chars>{quote}"

    return _QUOTED_SPAN_RE.sub(_sub, line)


def scrub_log_line(line: str, source: str = "generic") -> str | None:
    """Scrub one already-split line (no trailing newline expected, but one
    is tolerated). Returns the (possibly modified) line, or None if the
    line is a bare SQL-block continuation with nothing diagnostic of its
    own (only reachable from `scrub_log_text`, which is what actually
    tracks multi-line block state — passed to `scrub_log_line` on a bare
    single line, that case can't be detected, so a lone continuation
    fragment passed here comes back unchanged, not dropped; use
    `scrub_log_text` for real files).
    """
    if line is None:
        return None
    result = line

    if _SQL_OPEN_RE.search(result) and _CLOSES_RE.search(result):
        result = _scrub_bracket_span(result, "[SQL:")
    if _PARAMS_OPEN_RE.search(result) and _CLOSES_RE.search(result):
        result = _scrub_bracket_span(result, "[parameters:")

    result = _scrub_dedup_merge(result)
    result = _scrub_repetition_token(result)
    result = _scrub_generic_quoted(result)

    # R2-B4: compactor*.log gets the allowlist-shaped long-quoted-span pass
    # too (see the block comment above `_scrub_long_quoted_spans`). Gated
    # by source, not applied to openwebui/other logs, which are already
    # covered by the `[SQL:`/`[parameters:` bracket scrub above and have
    # their own, differently-shaped risk (bound SQL params, not `!r`
    # reprs in an f-string).
    if source.startswith("compactor"):
        result = _scrub_long_quoted_spans(result)

    return result


def scrub_log_text(text: str, source: str = "generic") -> str:
    """Scrub a whole file/tail, handling the multi-line records single-line
    `scrub_log_line` can't: a `[SQL: ...]` statement wrapped across several
    physical lines (SQLAlchemy's own pretty-printing), ending on whichever
    line first closes with `]`. Every such block collapses to one scrubbed
    line: `[SQL: <scrubbed:N chars>]`, where N counts the ORIGINAL block's
    characters (informational only). Everything else goes through
    `scrub_log_line` unchanged.

    Python tracebacks are not specially reassembled: each frame line is
    source code or a `File "...", line N, in func` header, never a value by
    itself, EXCEPT the final `ExceptionType: message` line, which — like
    any other line — still goes through `scrub_log_line` (so a
    SQLAlchemy `OperationalError` whose message happens to embed a
    `[parameters: ...]` fragment is still caught).
    """
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if _SQL_OPEN_RE.search(line) and not _CLOSES_RE.search(line):
            # Multi-line SQL block: consume until (and including) the line
            # that closes it, or end of file if it never does.
            block = [line]
            j = i + 1
            while j < n and not _CLOSES_RE.search(lines[j]):
                block.append(lines[j])
                j += 1
            if j < n:
                block.append(lines[j])
                j += 1
            total_chars = sum(len(b) for b in block)
            out.append(f"[SQL: <scrubbed:{total_chars} chars>]")
            i = j
            continue
        scrubbed = scrub_log_line(line, source)
        if scrubbed is not None:
            out.append(scrubbed)
        i += 1
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")
