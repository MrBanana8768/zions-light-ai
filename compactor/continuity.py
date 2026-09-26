"""
compactor.continuity — v3.1.9.8 `.15g`: the continuity band.

Design: V4_DESIGN.md §17.6a (r10). Evidence this ports faithfully:
`zl-ops/v4/lab/r10/strip10.py` (the two filters and the fallback entry),
`zl-ops/v4/lab/r9/band9.py` (the band format and the wipe/generation
rules), `zl-ops/v4/lab/r10/hold10.py` and the r10 design text for the
persistent wipe epoch and ledger.

WHAT THIS MODULE DOES.

r7 filled the summary gap with unsummarized turns VERBATIM, first inside
memory, and her replies are long enough that the fill ate all of memory
and still left holes. r8 compresses the gap instead: every unsummarized
exchange that the allocator does not place verbatim gets a bounded
DIGEST (<=160 tokens full, <=40 tokens headline), made in the background
after each reply, never on the request path.

Decoration never enters the band, and her words are never dropped
(r10, review r9 A1): `clean_her` keeps every line of hers except fence
markers and lines under 50% letters; `clean_reply` drops a line only if
it is not prose-like AND it is decoration. See each function's docstring.

WIPES (r9 HIGH-B, r10 A2/A3). A digest is a cache of the transcript, and
it must never outlive what she asked to forget:
  - `record_wipe` / `record_targeted_forget` / `reapply_ledger` maintain a
    ledger in COMPACTOR_CONTROL_ROOT — OUTSIDE STORAGE_ROOT and outside
    backup/restore scope (r10 A2/A3), so a restart, restore or import
    cannot resurrect what a `/forget` already removed.
  - `digests.visible` (this module's `visible()`) is the ONE reader every
    consumer (the allocator, /why, admin/diagnostic endpoints) must use.
  - `wipe_conversation_band` is what every full-wipe path calls, under
    conv_lock, after `memory.bump_wipe_generation` and before its own
    deletes — the same ordering `bump_wipe_generation`'s own docstring
    requires for every other memory layer.

NOTHING IN THIS MODULE MAY IMPORT `main`. It is imported BY main.py,
commands.py and portability.py; the reverse would be a cycle.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import unicodedata
from pathlib import Path
from typing import Any, Callable

import httpx

import alert
import logsetup
import memory
import reqctx
from envcfg import env_bool, env_int

logger = logging.getLogger("compactor.continuity")

# ---------------------------------------------------------------------------
# Config (§17.13: v3.1.9.8 adds no environment variable that MUST be set;
# every new knob below has a safe, code-level default).
# ---------------------------------------------------------------------------

DIGEST_FULL_TOKENS = env_int("COMPACTOR_DIGEST_FULL_TOKENS", 160)
DIGEST_HEADLINE_TOKENS = env_int("COMPACTOR_DIGEST_HEADLINE_TOKENS", 40)
DIGEST_FULL_UNITS = env_int("COMPACTOR_DIGEST_FULL_UNITS", 10)
# The extractive fallback's own per-side caps (lab/r10/strip10.py: 24 / 14).
_FALLBACK_HER_CAP = 24
_FALLBACK_REPLY_CAP = 14
# Approximate char cap on the LLM digest call's input, per side, standing
# in for the design's "3,000 tokens, head and tail kept" (~4 chars/token,
# the same rough ratio _enforce_hard_budget's own prescreen comment uses).
# A true head+tail-with-local-tokenizer cap is future work — see the
# v3.1.9.8 report's deviations. Digests never run on the request path, so
# this cost is background-only.
_LLM_INPUT_CHAR_CAP = 12_000


# Q38 (owner, unanswered as of r10): whether the LLM digest call should run
# at all. Default ON; OFF means fallback-only, everywhere this module is
# used, with no other behaviour change.
DIGEST_LLM_ENABLED = env_bool("COMPACTOR_DIGEST_LLM", True)


def _default_control_root() -> Path:
    return memory.STORAGE_ROOT.parent / "compactor-control"


CONTROL_ROOT = Path(os.environ.get("COMPACTOR_CONTROL_ROOT", "") or _default_control_root())


def _default_pending_root() -> Path:
    return memory.STORAGE_ROOT.parent / "compactor-pending"


# v3.1.9.8 round-4 review (item 3, MEDIUM — probe N2): the pending-intent
# durability sidecar used to live under STORAGE_ROOT
# (`STORAGE_ROOT/ledger-pending/`) — but STORAGE_ROOT is exactly what
# `restore_backup` replaces WHOLESALE (backup.py:151), same as CONTROL_ROOT
# is deliberately kept OUTSIDE it for the ledger itself (§17.6a r10 A3, see
# `assert_control_root_safe` below). A restore from a backup taken BEFORE a
# failed forget wipes the sidecar out along with everything else in
# STORAGE_ROOT, while the stale ledger in CONTROL_ROOT (never in
# backup/restore scope at all) survives untouched — so the pending needle
# that sidecar existed to carry forward is gone, and her words resurface
# after a restore + restart (probe N2).
#
# Two fixes were on the table: (a) teach `restore_backup` to carry
# `ledger-pending/` over as a union with whatever the archive holds, or
# (b) give the sidecar its own root, outside BOTH STORAGE_ROOT and
# CONTROL_ROOT. (a) would work, but it ties this module's own durability
# guarantee to `backup.py` correctly special-casing one subdirectory on
# every future restore path (hand restores, portability imports, a
# future restore tool) — exactly the "fixed at one site, missed the
# sibling" shape this codebase keeps paying for elsewhere. (b) is chosen
# instead: `PENDING_ROOT` is a THIRD sibling of STORAGE_ROOT and
# CONTROL_ROOT (not nested in either), so no restore of STORAGE_ROOT and
# no CONTROL_ROOT-specific fault can touch it, and no future restore path
# needs to know it exists. It is also independent of CONTROL_ROOT itself:
# round 3 deliberately kept the sidecar out of CONTROL_ROOT so a broken
# CONTROL_ROOT (the very thing that makes the LEDGER write fail) does not
# ALSO take down its own durability fallback — moving it INTO CONTROL_ROOT
# now would reintroduce that correlated-failure risk. `assert_pending_
# root_safe` (below) enforces the STORAGE_ROOT half of this the same way
# `assert_control_root_safe` already does for CONTROL_ROOT.
PENDING_ROOT = Path(os.environ.get("COMPACTOR_PENDING_ROOT", "") or _default_pending_root())


class ControlRootInsideStorageRoot(RuntimeError):
    """§17.6a r10: 'The compactor refuses to start if the control root is
    inside STORAGE_ROOT.' Raised by `assert_control_root_safe`, which
    main.py must call once at boot before serving any request."""


class PendingRootInsideStorageRoot(RuntimeError):
    """Round-4 review (item 3): the pending-intent sidecar root must not
    resolve inside STORAGE_ROOT either, for the same reason the ledger's
    own CONTROL_ROOT must not — see the PENDING_ROOT comment above.
    Raised by `assert_pending_root_safe`, which main.py must call once at
    boot before serving any request."""


def _assert_root_outside_storage(root: Path, *, env_name: str, exc: type) -> None:
    storage = memory.STORAGE_ROOT
    storage_abs = storage.resolve() if storage.exists() else storage.absolute()
    root_abs = root.resolve() if root.exists() else root.absolute()
    try:
        root_abs.relative_to(storage_abs)
    except ValueError:
        return
    raise exc(
        f"{env_name} ({root_abs}) resolves inside COMPACTOR_STORAGE_ROOT "
        f"({storage_abs}). This must live outside backup/restore scope "
        f"(§17.6a r10 A3) or a restore could silently bring back what a "
        f"/forget already removed. Refusing to start; set {env_name} to a "
        f"path outside the storage root."
    )


def assert_control_root_safe() -> None:
    _assert_root_outside_storage(
        CONTROL_ROOT, env_name="COMPACTOR_CONTROL_ROOT", exc=ControlRootInsideStorageRoot
    )


def assert_pending_root_safe() -> None:
    _assert_root_outside_storage(
        PENDING_ROOT, env_name="COMPACTOR_PENDING_ROOT", exc=PendingRootInsideStorageRoot
    )


# ---------------------------------------------------------------------------
# §17.6a: is_symbol, strip_symbols, letter_ratio, prose_like, VERSE, clock.
# Ported faithfully from lab/r10/strip10.py (the r10 proof harness).
# ---------------------------------------------------------------------------

VERSE = re.compile(
    r"\b(?:[1-3]\s?)?[A-Z][a-z]+\.?\s\d{1,3}:\d{1,3}(?:[-–]\d{1,3})?\b"
)
_CLOCK = re.compile(
    r"\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:[AaPp]\.?[Mm]\.?)?|"
    r"\b\d{1,2}\s*(?:AM|PM|am|pm|a\.m\.|p\.m\.)\b|"
    r"\b(?:MST|MDT|PST|PDT|EST|EDT|CST|CDT|UTC|GMT)\b"
)
_STATUS = re.compile(r"\(\s*\d{1,3}\s*%\s*\)")
_BOARD_CODE = re.compile(r"\b[A-Z]{2,}\s?\d+\b")


def is_symbol(ch: str) -> bool:
    """Unicode So/Sk/Mn (non-letter), box/block/geometric/dingbats
    (U+2500-27BF incl. ❤/✝), U+2B00-2BFF, U+1F000-1FAFF, variation
    selectors and ZWJ. Exactly §17.6a's helper list."""
    o = ord(ch)
    return (
        (unicodedata.category(ch) in ("So", "Sk", "Mn") and not ch.isalpha())
        or 0x2500 <= o <= 0x27BF
        or 0x2B00 <= o <= 0x2BFF
        or 0x1F000 <= o <= 0x1FAFF
        or 0xFE00 <= o <= 0xFE0F
        or o == 0x200D
    )


def strip_symbols(s: str) -> str:
    """Removes symbol characters, leading `# > * - • |` markers, and
    markdown emphasis characters. Never drops letters or the words that
    carry them."""
    s = "".join(ch for ch in s if not is_symbol(ch))
    s = re.sub(r"^[\s#>*\-•|]+", "", s)
    s = re.sub(r"[*_`~]+", "", s)
    return re.sub(r"\s{2,}", " ", s).strip()


def letter_ratio(s: str) -> float:
    ns = sum(not ch.isspace() for ch in s)
    return sum(ch.isalpha() for ch in s) / ns if ns else 0.0


def prose_like(s: str) -> bool:
    """At least 6 words and at least 70% letters after `strip_symbols`."""
    t = strip_symbols(s)
    return len(t.split()) >= 6 and letter_ratio(t) >= 0.7


def _clock_spans(s: str) -> list[tuple[int, int]]:
    """Clock matches that are NOT inside a verse reference — a Bible verse
    such as `John 3:16` is never a clock (§17.6a r10, A1)."""
    verse_spans = [m.span() for m in VERSE.finditer(s)]
    return [
        m.span()
        for m in _CLOCK.finditer(s)
        if not any(a <= m.start() < b for a, b in verse_spans)
    ]


def has_clock(s: str) -> bool:
    return bool(_clock_spans(s))


def remove_clocks(s: str) -> str:
    for a, b in reversed(_clock_spans(s)):
        s = s[:a] + s[b:]
    return re.sub(r"\s{2,}", " ", s).strip(" –-—,")


def clean_her(text: str) -> list[str]:
    """§17.6a r10, A1: for HER messages. Keeps every line except code-fence
    marker lines and lines under 50% letters (unless the line is a verse
    reference). Applies NO clock, ALL-CAPS, prefix or emoji rule — her
    words, including a clock time she wrote or a ❤/✝ line, are
    kept, symbol characters stripped."""
    out: list[str] = []
    for line in (text or "").splitlines():
        s = line.strip()
        if s.startswith("```"):
            continue
        if not s or (letter_ratio(s) < 0.5 and not VERSE.search(s)):
            continue
        t = strip_symbols(line)
        if t:
            out.append(t)
    return out


def _is_reply_decoration(line: str) -> bool:
    """A table row, under 50% letters, a `#` header or a line wholly
    wrapped in `**`/`__`, a clock and <=16 words, a `(NN%)` status, a
    symbol and <=8 words, or ALL-CAPS of 2-12 words. A verse-reference line
    is never decoration."""
    t = line.strip()
    if VERSE.search(t) and strip_symbols(t):
        return False
    if t.startswith("|") or letter_ratio(t) < 0.5:
        return True
    words = t.split()
    if (
        t.startswith("#")
        or (t.startswith("**") and t.endswith("**"))
        or (t.startswith("__") and t.endswith("__"))
    ):
        return True
    if has_clock(t) and len(words) <= 16:
        return True
    if _STATUS.search(t):
        return True
    if any(is_symbol(ch) for ch in t) and len(words) <= 8:
        return True
    letters = [ch for ch in t if ch.isalpha()]
    if (
        2 <= len(words) <= 12
        and letters
        and sum(ch.isupper() for ch in letters) / len(letters) >= 0.7
    ):
        return True
    return False


_LABEL = re.compile(
    r"^[\s>*\-•]*(?:\d{1,3}[.)]\s*)?(?:\*\*|__)[^*_]{1,60}(?:\*\*|__)\s*"
    r"(?:\(\s*\d{1,3}\s*%\s*\))?\s*[:—–-]\s*"
)


def _band_decoration(line: str) -> bool:
    """r10 review R10-A1 (HIGH): the STRICT band-output test — no
    `prose_like` override. `clean_reply`'s `keep = prose_like(line) or
    not decoration` let any >=6-word, >=70%-letter header, bold banner,
    clock line or ALL-CAPS line through once symbols were stripped
    (measured: 55,601 of 190,469 kept reply lines, 29.2% of kept
    characters, entered ONLY through that override). This is `clean_reply`
    minus the override, plus no 12-word cap on the ALL-CAPS rule — a
    clock LINE is still kept (with its clock tokens removed), so a
    sentence that happens to mention a time is not thrown away, only its
    decoration wrapper."""
    t = line.strip()
    if not t:
        return True
    if VERSE.search(t) and strip_symbols(t):
        return False
    if t.startswith("|") or letter_ratio(t) < 0.5:
        return True
    if (
        t.startswith("#")
        or (t.startswith("**") and t.endswith("**"))
        or (t.startswith("__") and t.endswith("__"))
    ):
        return True
    if _STATUS.search(t):
        return True
    words = t.split()
    if any(is_symbol(ch) for ch in t) and len(words) <= 8:
        return True
    letters = [ch for ch in t if ch.isalpha()]
    if len(words) >= 2 and letters and sum(ch.isupper() for ch in letters) / len(letters) >= 0.7:
        return True  # NO 12-word cap — r9/r10's clean_reply capped this at 12 words
    return False


def clean_band(text: str) -> list[str]:
    """r10 review R10-A1 (HIGH): the band's OUTPUT filter — used for the
    extractive fallback's reply part and the LLM digest's post-filter.
    `clean_reply` (above) stays the digest INPUT filter only, so the
    0.03% prose-loss metric for what the summarizer reads is unaffected.
    Fenced content never reaches band output at all (measured leak was
    through `prose_like`, not through fences, so this is stricter than
    `clean_reply` there too). A leading label such as `> 1. **LABEL**
    (99%): sentence` has its label stripped and the sentence rescued if
    what follows is itself prose-like."""
    out: list[str] = []
    fence = False
    for line in (text or "").splitlines():
        s = line.strip()
        if s.startswith("```"):
            fence = not fence
            continue
        if fence or not s:
            continue
        m = _LABEL.match(line)
        if m and prose_like(line[m.end():]):
            line = line[m.end():]
        if _band_decoration(line):
            continue
        t = remove_clocks(strip_symbols(line))
        if t:
            out.append(t)
    return out


def clean_reply(text: str) -> list[str]:
    """§17.6a r10, A1: for the model's REPLIES. Drops a line only if it is
    NOT prose-like AND it is decoration. Inside code fences, keeps
    prose-like or verse-reference lines only. Kept lines are symbol-
    stripped, and their clock tokens are removed."""
    out: list[str] = []
    fence = False
    for line in (text or "").splitlines():
        s = line.strip()
        if s.startswith("```"):
            fence = not fence
            continue
        if not s:
            continue
        if fence:
            keep = prose_like(line) or bool(VERSE.search(line))
        else:
            keep = prose_like(line) or not _is_reply_decoration(line)
        if keep:
            t = remove_clocks(strip_symbols(line))
            if t:
                out.append(t)
    return out


def first_prose_sentence(lines: list[str], *, reply: bool = False) -> str:
    """The first sentence with >=4 words and >=70% letters; for a reply,
    one that also carries no invented board code (`LAW 7`)."""
    text = " ".join(lines)
    for m in re.finditer(r"[^.!?]+[.!?]?", text):
        s = m.group(0).strip()
        if not s:
            continue
        if len(s.split()) >= 4 and letter_ratio(s) >= 0.7 and not (
            reply and _BOARD_CODE.search(s)
        ):
            return s
    return ""


def cut(tok: Any, s: str, n: int) -> str:
    """An EXACT token cap: truncates step by step so the result (with its
    "…" if truncated) re-encodes within `n` tokens, and never leaves a
    broken character. `tok` is any object with `.encode(s, bos=False,
    eos=False) -> list[int]` and `.decode(ids) -> str` (the compactor's
    local Tekken tokenizer — see main.get_tokenizer). If `tok` is None
    (no local tokenizer available, e.g. in a unit test), falls back to a
    conservative word-based cap so callers still get a bounded string
    rather than an exception."""
    if not s:
        return s
    if tok is None:
        words = s.split()
        # ~0.75 words/token is a common English rule of thumb; conservative
        # (undershoots the real cap) so the fallback never OVER-runs it.
        approx_words = max(1, int(n * 0.7))
        if len(words) <= approx_words:
            return s
        return " ".join(words[:approx_words]).rstrip() + "…"
    ids = tok.encode(s, bos=False, eos=False)
    if len(ids) <= n:
        return s
    k = n - 1
    while k > 0:
        t = tok.decode(ids[:k]).replace("�", "").rstrip() + "…"
        if len(tok.encode(t, bos=False, eos=False)) <= n:
            return t
        k -= 1
    return "…"


# ---------------------------------------------------------------------------
# The extractive fallback (§17.6a, lab/r10/strip10.py Part B).
# ---------------------------------------------------------------------------

def extractive_fallback(tok: Any, her_text: str, reply_text: str, unit_number: int) -> str:
    """Built in this order, and ALWAYS available (no network call, no
    failure mode): her first prose sentence (or her first 24 cleaned
    tokens, or the no-text marker); then, if the reply has one, its first
    prose sentence containing no board code; capped exactly to
    DIGEST_HEADLINE_TOKENS.

    r10 review R10-A1: the reply part is drawn from `clean_band`, the
    strict band-OUTPUT filter — not `clean_reply` (the digest input
    filter), which let headers/banners/ALL-CAPS lines through via its
    `prose_like` override (measured: 1,122 of 1,942 fallback reply parts
    were decoration-sourced under the un-amended r10 filter).

    `unit_number` is NOT rendered into the returned text — `format_band`
    is the one place that numbers band entries (its own `f"{i + 1}. "`),
    covering an LLM digest's output exactly the same way it covers a
    fallback's. Numbering it here too used to produce "1. 1. …" on every
    fallback-sourced entry once format_band added its own prefix.
    `unit_number` stays a parameter because the digest store still
    records it (see `generate_digest`'s `unit_number` field) for
    diagnostics."""
    her_lines = clean_her(her_text)
    reply_lines = clean_band(reply_text)
    hs = (
        first_prose_sentence(her_lines)
        or cut(tok, " ".join(her_lines), _FALLBACK_HER_CAP)
        or "[a message without text]"
    )
    rs = first_prose_sentence(reply_lines, reply=True)
    body = cut(tok, hs, _FALLBACK_HER_CAP)
    if rs:
        body += " — answered: " + cut(tok, rs, _FALLBACK_REPLY_CAP)
    entry = cut(tok, body, DIGEST_HEADLINE_TOKENS)
    return entry


# ---------------------------------------------------------------------------
# The LLM digest call — background only, never on the request path.
# ---------------------------------------------------------------------------

DIGEST_INSTRUCTION = (
    "Line 1: in at most 30 words, what she said or asked and what was "
    "answered. Then at most 3 short lines of specifics: names, numbers, "
    "promises, requests, anything she asked to be remembered word for "
    "word. Do not add anything that is not in the text. No times, dates, "
    "headers, boards, rule lists or symbols; do not quote codes such as "
    "LAW 7."
)


def _postfilter_llm_output(text: str) -> str:
    """§17.6a, amended by r10 review R10-A1: post-filtered with
    `clean_band` (the strict band-output filter), not `clean_reply` —
    the review found an LLM headline like `# **SHE ASKED ABOUT THE PLAN**`
    survives `clean_reply`'s `prose_like` override untouched, which
    falsifies "the guarantee does not depend on the model obeying" for
    exactly headers, bold-only lines and ALL-CAPS lines. Plus removal of
    any sentence containing a board code, as before."""
    lines = clean_band(text)
    kept = []
    for line in lines:
        sentences = re.split(r"(?<=[.!?])\s+", line)
        sentences = [s for s in sentences if not _BOARD_CODE.search(s)]
        s = " ".join(sentences).strip()
        if s:
            kept.append(s)
    return "\n".join(kept)


async def _call_digest_llm(
    vllm_url: str,
    model: str,
    her_clean_lines: list[str],
    reply_clean_lines: list[str],
    http_client: "httpx.AsyncClient | None",
) -> str:
    """One vLLM call. Input is `clean_her`/`clean_reply` output (decoration
    already stripped before the model ever sees it), each side capped
    (see `_LLM_INPUT_CHAR_CAP`'s docstring above for why this is a char
    cap, not the design's exact token cap). `max_tokens` = the full-digest
    cap, so a rambling reply cannot make this call unbounded."""
    her_input = "\n".join(her_clean_lines)[:_LLM_INPUT_CHAR_CAP]
    reply_input = "\n".join(reply_clean_lines)[:_LLM_INPUT_CHAR_CAP]
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": DIGEST_INSTRUCTION},
            {
                "role": "user",
                "content": f"She said:\n{her_input}\n\nThe reply:\n{reply_input}",
            },
        ],
        "max_tokens": DIGEST_FULL_TOKENS,
        "temperature": 0.2,
        "stream": False,
    }
    owns_client = http_client is None
    client = http_client or httpx.AsyncClient()
    try:
        r = await client.post(
            f"{vllm_url}/v1/chat/completions", json=payload, timeout=30,
            headers=reqctx.vllm_headers("digest"),
        )
        r.raise_for_status()
        data = r.json() or {}
        choices = data.get("choices") or []
        if not choices:
            raise ValueError("vLLM returned no choices for a digest call")
        return ((choices[0].get("message") or {}).get("content") or "").strip()
    finally:
        if owns_client:
            await client.aclose()


async def make_digest_text(
    tok: Any,
    her_text: str,
    reply_text: str,
    unit_number: int,
    *,
    vllm_url: str | None,
    model: str | None,
    http_client: "httpx.AsyncClient | None" = None,
) -> str:
    """Try the LLM digest (if COMPACTOR_DIGEST_LLM is on and a vLLM target
    is configured); fall back to the extractive summary on any failure, a
    missing configuration, or an output the post-filter empties. There is
    therefore ALWAYS a digest (§17.6a)."""
    if DIGEST_LLM_ENABLED and vllm_url and model:
        try:
            raw = await _call_digest_llm(
                vllm_url, model, clean_her(her_text), clean_reply(reply_text), http_client
            )
            filtered = _postfilter_llm_output(raw)
            if filtered.strip():
                return filtered
        except Exception as e:
            logger.warning(
                f"continuity digest LLM call failed (falling back to the "
                f"extractive summary, never on the request path): "
                f"{type(e).__name__}: {e}"
            )
    return extractive_fallback(tok, her_text, reply_text, unit_number)


# ---------------------------------------------------------------------------
# The wipe ledger (§17.6a r10, A2/A3): persistent, in COMPACTOR_CONTROL_ROOT.
# ---------------------------------------------------------------------------

class LedgerUnreadable(RuntimeError):
    """r10 review R10-B1 (HIGH): a ledger file that EXISTS but cannot be
    parsed must fail closed — never silently treated as epoch 0 (absent
    == never wiped; unreadable == unknown, and unknown must not be
    trusted with what a /forget already promised her)."""


_ABSENT_LEDGER = {"epoch": 0, "boundary": None, "targeted": []}

# v3.1.9.8 `.15p`/Task-B round-2 review (item 1, HIGH), REVISED round-3
# (the owner's "fix it fully" directive — round 2's fix was incomplete):
# a ledger WRITE failure is not merely "a future restore won't replay
# this" — with COMPACTOR_CONTINUITY_BAND on, a `/forget` whose needle (or
# epoch bump) never reached disk leaves the forgotten words fully
# recoverable from the band on her VERY NEXT request, while she has
# already been told the forget succeeded.
#
# Round 2 fixed that with a process-local "degraded" mark that failed the
# band closed — but the mark cleared on ANY later successful write for
# the SAME conv, even one built from the STALE on-disk ledger that never
# carried the failed change (round-3 review's probe expC/probe_ledger.py,
# attack A1: a failed "/forget Maren" followed by a successful, unrelated
# "/forget pizza" cleared the mark and Maren's words came back). And the
# mark was process-local only (attack A2: a restart re-opened the band
# with the needle silently gone; attack A3: same shape for a lost full
# wipe — the epoch stayed 0 and a pre-wipe digest was still visible).
#
# Round 3's fix: PENDING INTENTS. `_pending_intents[conv_id]` holds
# whatever a failed write for that conv could not persist — needles to
# add (a UNION: every failed attempt's needles accumulate, none are ever
# dropped) and the highest epoch (+ its boundary) any failed wipe/retire
# wanted to set (epoch = MAX, never a downgrade). Every LATER write for
# that conv, triggered by ANY of the three public functions below (or by
# the startup recovery path), MERGES this in before it writes
# (`_merge_pending_intent`) — so attack A1's unrelated "/forget pizza"
# now carries Maren's needle forward instead of losing it. The pending
# intent is cleared only after a write that included it actually lands
# (fsynced, same guarantee `_write_ledger` has always given the ledger
# file itself).
#
# DURABILITY ACROSS RESTART: the in-memory dict alone does not survive a
# restart (attack A2), so every time a pending intent changes it is ALSO
# best-effort persisted to a sidecar (`_pending_sidecar_path`) under
# `PENDING_ROOT` — its OWN root, a sibling of both STORAGE_ROOT and
# CONTROL_ROOT (round-4 review, item 3; see `PENDING_ROOT`'s own
# module-level comment for why neither of the other two roots would do).
# `reapply_all_ledgers` (called at startup) reads these sidecars back —
# merging BOTH the sidecar's own content AND anything already in memory
# (`_merge_pending_intent`, round-4 review item 1) — retries the
# merge+write for each, and deletes a sidecar only once ITS merged write
# actually lands; an unreadable (not merely absent) sidecar is NEVER
# deleted, on any code path, so a corrupt file always keeps the band
# failed closed rather than silently being treated as satisfied.
#
# RESIDUAL: if BOTH the ledger write and the sidecar write fail in the
# same attempt, the pending intent exists ONLY in this process's memory.
# The band still fails closed while this process stays up (`_pending_
# intents` alone is enough for that), but a restart before either write
# succeeds loses it for good — the disk refused two independent writes,
# and there is nowhere else left to keep it. This is the ONE scenario in
# which the forgotten words could still reappear, and it is the ONLY
# scenario the /forget reply now warns about (see commands.py);
# `_double_failure_current`/`_double_failure_ever` name it precisely, and
# `/health` reports it for as long as it lasts.
#
# BACKUP INTERACTION: PENDING_ROOT is OUTSIDE STORAGE_ROOT specifically so
# `restore_backup` (backup.py:151), which replaces STORAGE_ROOT WHOLESALE,
# never touches it (round-4 review, item 3 — probe N2 demonstrated the
# STORAGE_ROOT-nested version losing a pending intent to exactly this). A
# restore therefore leaves any LIVE pending sidecar untouched. The
# scenario that remains — an operator restoring PENDING_ROOT itself from
# an old, independent backup and bringing back a STALE sidecar (one
# already merged and deleted since) — is still safe by the SAME
# union/max merge rule every pending-intent merge uses: it can only ADD
# needles or RAISE the epoch, which can only hide MORE of the transcript
# than the current ledger already does, never LESS. A pending intent is
# never a request to REVEAL anything, only to suppress it. Confirmed by
# `test_continuity.py`'s "restoring an old pending sidecar only hides
# more" test.
#
# None of `_ledger_write_failed` / `_ledger_write_failed_ever` /
# `_ledger_unreadable_ever` / `_double_failure_current` /
# `_double_failure_ever` are persisted — they reset on restart, same as
# every other in-process health counter in this file (see memory.
# current_wipe_generation's own docstring for the same tradeoff).
# `_pending_intents` is the one exception with a durability story at all,
# and even that story is "best effort sidecar", not "guaranteed".
_ledger_write_failed: set[str] = set()
_ledger_write_failed_ever: set[str] = set()
_ledger_unreadable_ever: set[str] = set()
_pending_intents: dict[str, dict] = {}
_double_failure_current: set[str] = set()
_double_failure_ever: set[str] = set()


def _pending_sidecar_path(conv_id: str) -> Path:
    """`PENDING_ROOT / "ledger-pending" / f"{conv_id}.json"` — round-4
    review (item 3) moved this from `STORAGE_ROOT` to its own sibling
    root; see the `PENDING_ROOT` module-level comment for why (probe N2:
    a restore replaces STORAGE_ROOT wholesale, which used to take the
    sidecar down with it). Same conv_id validation chokepoint as
    `_ledger_path` (`memory._safe_path`); raises memory.UnsafeConvId on a
    hostile conv_id, same as that function."""
    return memory._safe_path("ledger-pending", conv_id, ".json", root=PENDING_ROOT)


class _PendingSidecarUnreadable(Exception):
    """The pending sidecar file exists but could not be parsed. Distinct
    from 'absent' (`_read_pending_sidecar` returns None for that) —
    callers must never delete an unreadable sidecar and must keep the
    band failed closed for that conv, matching the 'left for manual
    repair' promise `reapply_all_ledgers` already makes for an unreadable
    LEDGER file (round-4 review, item 1)."""


def _has_pending_intent(conv_id: str) -> bool:
    """True if conv_id has a pending intent EITHER in this process's
    memory OR only on disk (the sidecar survived a restart that has not
    yet run `reapply_all_ledgers`, or `reapply_all_ledgers` itself is
    mid-retry and has not resolved this conv yet). `_read_ledger_failing_
    closed` uses this, not a bare `_pending_intents` membership check —
    checking the sidecar file directly closes the exact gap a fresh
    process would otherwise have between boot and startup re-apply.

    Round-4 review (item 4, MEDIUM — probe N3): the previous version
    caught bare `Exception` from the stat and returned False — failing
    OPEN on exactly the kind of fault (a permission error, a transient
    I/O error) that means 'unknown', not 'absent'. Only `memory.
    UnsafeConvId` (a genuinely hostile conv_id, which cannot have a
    sidecar at all) means False; every other exception must be treated
    as 'assume pending' and fail CLOSED — the same direction every other
    fail-closed decision in this module already takes."""
    if conv_id in _pending_intents:
        return True
    try:
        return _pending_sidecar_path(conv_id).is_file()
    except memory.UnsafeConvId:
        return False
    except Exception:
        return True


def _read_pending_sidecar(conv_id: str) -> dict | None:
    """Reads the sidecar FRESH every call — same R10-B1 doctrine as
    `_read_ledger`: another process, or an earlier failed attempt in
    THIS process, may have written it since the last read. Returns None
    if the file is absent. Raises `_PendingSidecarUnreadable` if it
    exists but cannot be parsed."""
    p = _pending_sidecar_path(conv_id)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("pending ledger sidecar is not a JSON object")
        return data
    except Exception as e:
        raise _PendingSidecarUnreadable(f"{p}: {type(e).__name__}: {e}") from e


def _persist_pending_sidecar(conv_id: str, pending: dict) -> bool:
    """Best-effort durability write for a pending intent. Failure here
    (this sidecar ALSO cannot be written) is the DOUBLE FAILURE residual
    — the caller (`_record_pending_intent`) is what decides that means
    and logs it; this function just reports success or failure."""
    try:
        memory.atomic_write_json(_pending_sidecar_path(conv_id), pending)
        return True
    except Exception:
        return False


def _delete_pending_sidecar(conv_id: str) -> None:
    """Best-effort cleanup once a pending intent has actually been merged
    and written. A leftover sidecar (this fails) is not a correctness
    problem — it gets read, merged (a no-op, since its content is already
    on disk) and deleted again the next time anything writes for this
    conv, or at the next startup re-apply. Callers must NEVER call this
    when the sidecar was found unreadable this round — see
    `_write_ledger_best_effort`'s `sidecar_unreadable` handling."""
    try:
        p = _pending_sidecar_path(conv_id)
        if p.is_file():
            p.unlink()
    except Exception:
        pass


def _fold_needles(dst: list[dict], src: list[dict]) -> list[dict]:
    """Union of two needle lists, deduped by (needle, epoch)."""
    existing = {(n.get("needle"), n.get("epoch")) for n in dst}
    out = list(dst)
    for n in src:
        key = (n.get("needle"), n.get("epoch"))
        if key not in existing:
            out.append(n)
            existing.add(key)
    return out


def _fold_epoch_boundary(dst: dict, src_epoch, src_boundary) -> None:
    """Mutates dst's 'epoch'/'boundary' in place: a HIGHER `src_epoch`
    always wins (and takes its boundary with it); an EQUAL epoch keeps
    whichever boundary is BROADER — "all" (a retire/tombstone) beats any
    partial (turn-fingerprint pair) boundary or None, since "all" is the
    strictly wider claim about what is covered. Round-4 review (item 6,
    LOW): two failed writes landing on the SAME target epoch — a wipe and
    a retire both bumping to the same next epoch — must not let whichever
    happens to merge first silently narrow what the other already
    covered."""
    if src_epoch is None:
        return
    cur = dst.get("epoch")
    if cur is None or src_epoch > cur:
        dst["epoch"] = src_epoch
        dst["boundary"] = src_boundary
    elif src_epoch == cur and src_boundary == "all" and dst.get("boundary") != "all":
        dst["boundary"] = "all"


def _merge_pending_intent(conv_id: str, data: dict) -> "tuple[dict, bool]":
    """Folds the pending intent for `conv_id` into `data` BEFORE it is
    written: needles are a UNION, epoch is a MAX with the broader-
    boundary-on-a-tie rule (`_fold_epoch_boundary`) — the 'never lose
    anything' contract every later write for this conv now carries,
    regardless of which of the three public functions below (or the
    startup recovery path) triggered it.

    Round-4 review (item 1, HIGH — probes N1b/N4): folds BOTH the
    in-memory `_pending_intents[conv_id]` AND whatever is on the
    DURABILITY SIDECAR right now, not memory alone. Round 3's version
    only merged memory, so a sidecar that existed on disk but was never
    loaded into memory (a startup read that failed, or `reapply_all_
    ledgers` raising before its pending loop even ran) was invisible to
    every LATER write attempt — that write landed, and the unmerged
    sidecar was then deleted as if its content had been included, losing
    the needle it held. Reading the sidecar fresh on every write closes
    that: the two sources are merged together first, then folded into
    `data`.

    Mutates and returns `(data, sidecar_unreadable)`. `data` is always a
    dict the caller just built fresh (from `_read_ledger` or `_ABSENT_
    LEDGER`), never cached or shared, so mutating it in place is safe.
    `sidecar_unreadable` is True when the sidecar file exists but could
    not be parsed — the caller must NOT delete it in that case (nothing
    here confirms its content was actually included in what gets
    written) and must keep the band failed closed for this conv."""
    pending = dict(_pending_intents.get(conv_id) or {"epoch": None, "boundary": None, "needles": []})
    sidecar_unreadable = False
    try:
        disk_pending = _read_pending_sidecar(conv_id)
    except _PendingSidecarUnreadable:
        disk_pending = None
        sidecar_unreadable = True
    if disk_pending:
        _fold_epoch_boundary(pending, disk_pending.get("epoch"), disk_pending.get("boundary"))
        pending["needles"] = _fold_needles(pending.get("needles", []), disk_pending.get("needles", []))
    if pending.get("epoch") is not None or pending.get("needles"):
        _fold_epoch_boundary(data, pending.get("epoch"), pending.get("boundary"))
        data["targeted"] = _fold_needles(data.get("targeted", []), pending.get("needles", []))
    return data, sidecar_unreadable


def _record_pending_intent(conv_id: str, attempted: dict) -> bool:
    """A ledger write just failed. `attempted` is the dict this call
    tried to persist (already folded with whatever was pending — memory
    AND sidecar — via `_merge_pending_intent`, inside `_write_ledger_
    best_effort`) — merge it into `_pending_intents[conv_id]` (additive
    only: nothing already pending is ever removed here, even if
    `attempted` does not carry it forward — e.g. `retire_tombstone`'s
    empty `targeted` list must never erase a needle pending from an
    earlier failed forget) and try to persist the result to the
    durability sidecar. Returns whether the sidecar write itself
    succeeded — the caller uses that to tell a plain single failure
    (ledger down, sidecar fine — fully recoverable) from the DOUBLE
    FAILURE residual (both down — recoverable only until the next
    restart)."""
    pending = _pending_intents.setdefault(
        conv_id, {"epoch": None, "boundary": None, "needles": []}
    )
    _fold_epoch_boundary(pending, attempted.get("epoch"), attempted.get("boundary"))
    pending["needles"] = _fold_needles(pending.get("needles", []), attempted.get("targeted", []))
    sidecar_ok = _persist_pending_sidecar(conv_id, pending)
    _ledger_write_failed.add(conv_id)
    _ledger_write_failed_ever.add(conv_id)
    if sidecar_ok:
        _double_failure_current.discard(conv_id)
    else:
        _double_failure_current.add(conv_id)
        _double_failure_ever.add(conv_id)
    return sidecar_ok


def _pending_intent_conv_ids() -> set:
    """Every conv_id with a pending intent, in memory OR only on disk (a
    sidecar left from before a restart that `reapply_all_ledgers` has not
    yet processed this boot). Backs `ledger_health_counts`'s
    `pending_intent_current` (round-4 review, item 6, LOW: the previous
    version counted only `len(_pending_intents)`, undercounting exactly
    the sidecar-only gap `_has_pending_intent` already treats as
    pending) — the two must agree on what 'pending' means."""
    out = set(_pending_intents.keys())
    try:
        pending_dir = PENDING_ROOT / "ledger-pending"
        if pending_dir.is_dir():
            out.update(p.stem for p in pending_dir.glob("*.json"))
    except Exception:
        pass
    return out


def ledger_degraded(conv_id: str) -> bool:
    """True while conv_id has a pending intent — in memory, or only on
    disk (sidecar not yet reapplied). commands._handle_forget and
    main.py's full-wipe reply-building call this AFTER record_wipe/
    record_targeted_forget/retire_tombstone to decide whether to append a
    note to her /forget reply — see those call sites for the exact
    wording, and `ledger_double_failure` for the one case that note must
    say more than 'not yet durable, but nothing should reappear'."""
    return _has_pending_intent(conv_id)


def ledger_double_failure(conv_id: str) -> bool:
    """True only in the RESIDUAL case (see the module-level PENDING
    INTENTS comment): this conv's pending intent could not be saved to
    the durability sidecar either, so it exists solely in this process's
    memory, and a restart before the next successful write would lose
    it. This is the ONLY condition under which a /forget reply should
    say the words could come back."""
    return conv_id in _double_failure_current


def ledger_health_counts() -> dict:
    """v3.1.9.8 round-2/round-3/round-4 review: backs the /health ledger
    section. Counts are DISTINCT CONVERSATIONS since process start, not
    events — a conv that fails and later recovers still counts once in
    the *_ever tallies, and drops out of the *_current ones the moment it
    recovers. No conv ids are exposed here, matching every other /health
    aggregate in this codebase. `pending_intent_current` uses
    `_pending_intent_conv_ids()` (round-4 review, item 6), not a bare
    `len(_pending_intents)` — it must count sidecar-only pending intents
    too, or this number would understate exactly the convs `_has_pending_
    intent` is protecting."""
    return {
        "unreadable_ever": len(_ledger_unreadable_ever),
        "write_failed_ever": len(_ledger_write_failed_ever),
        "write_failed_current": len(_ledger_write_failed),
        "pending_intent_current": len(_pending_intent_conv_ids()),
        "double_failure_current": len(_double_failure_current),
        "double_failure_ever": len(_double_failure_ever),
    }


def _ledger_path(conv_id: str) -> Path:
    """`CONTROL_ROOT / "wipes" / f"{conv_id}.json"` — the convention
    `reapply_all_ledgers` below relies on (it globs `CONTROL_ROOT / "wipes"
    / "*.json"` and recovers conv_id from each file's stem). portability.py
    does NOT depend on this layout: it never reads or writes the ledger —
    see `quarantine_conversation`'s own docstring for why the digest cache
    and this ledger are deliberately excluded from what it carries.
    conv_id reaches a filesystem path here
    exactly as it does for every other per-conversation file, so this
    reuses `memory._safe_path` — the one chokepoint that validates a
    conv_id against the [A-Za-z0-9_-] charset and re-checks the resolved
    path never leaves its root (v3.1.8's arbitrary-file-write fix) —
    rather than restating that rule against CONTROL_ROOT instead of
    STORAGE_ROOT. Raises memory.UnsafeConvId on a hostile conv_id (path
    separators, "..", NUL bytes, empty, over-length); never writes
    outside CONTROL_ROOT/wipes."""
    return memory._safe_path("wipes", conv_id, ".json", root=CONTROL_ROOT)


def _read_ledger(conv_id: str) -> dict:
    """r10 review R10-B1: reads the ledger file FRESH on every call — no
    process-local cache — because the backup/restore CLI is a different
    process and can rewrite this file between two calls in THIS process.
    Raises LedgerUnreadable (rather than returning epoch 0) when the file
    exists but cannot be parsed."""
    p = _ledger_path(conv_id)
    if not p.is_file():
        return dict(_ABSENT_LEDGER)
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("wipe ledger is not a JSON object")
    except Exception as e:
        raise LedgerUnreadable(f"{p}: {type(e).__name__}: {e}") from e
    data.setdefault("epoch", 0)
    data.setdefault("boundary", None)
    data.setdefault("targeted", [])
    return data


def _read_ledger_failing_closed(conv_id: str, *, on_fail_context: str) -> dict | None:
    """Wraps `_read_ledger`: returns None (never a fabricated epoch 0) and
    logs ERROR + alerts when the file exists but is unreadable, per r10
    review R10-B1. Callers that get None must show NOTHING from the band
    older than the verbatim floor, rather than guess.

    v3.1.9.8 round-2/round-3 review (item 1, HIGH): also returns None —
    the SAME fail-closed contract — while `conv_id` has a PENDING INTENT
    (`_has_pending_intent`, checked in memory AND on disk — see the
    module-level PENDING INTENTS comment above `_pending_intents`). A
    read of the ledger FILE would succeed here (the file on disk is
    simply stale: a needle or epoch bump that was supposed to land never
    did), so without this check `visible()`/`apply_needles` would
    confidently show band content built against a ledger state that is
    known to be wrong. Checked before the read, not after, and
    unconditionally (the log line below is gated by log_once, the
    fail-closed return is not) — round 2's version of this check only
    consulted the in-process `_ledger_write_failed` set, which a fresh
    process after a restart starts empty even though a pending intent's
    sidecar is still sitting on disk; `_has_pending_intent` checks that
    sidecar directly and closes that gap."""
    if _has_pending_intent(conv_id):
        if logsetup.log_once(f"continuity.ledger_write_failed_closed.{conv_id}"):
            logger.error(
                f"conv={conv_id}: this conversation has a pending ledger "
                f"intent (a wipe/forget that could not be fully persisted "
                f"yet); failing CLOSED during {on_fail_context} exactly as "
                f"an unreadable ledger would, until it is merged into a "
                f"successful write. This is reported at /health."
            )
        return None
    try:
        return _read_ledger(conv_id)
    except LedgerUnreadable as e:
        _ledger_unreadable_ever.add(conv_id)
        if logsetup.log_once(f"continuity.ledger_unreadable.{conv_id}"):
            logger.error(
                f"conv={conv_id}: wipe ledger unreadable during "
                f"{on_fail_context} ({e}); failing CLOSED — no band "
                f"entries older than the verbatim floor will be shown "
                f"until this is repaired. This is reported at /health."
            )
            try:
                import alert as _alert

                _alert.notify(
                    "compactor", "error",
                    f"continuity wipe ledger unreadable for conv={conv_id}",
                )
            except Exception:
                pass
        return None


def _write_ledger(conv_id: str, data: dict) -> None:
    p = _ledger_path(conv_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(p.parent), prefix=p.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, p)
        try:
            dir_fd = os.open(str(p.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError as e:
            logger.debug(f"ledger directory fsync skipped for {p.parent}: {e}")
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def current_ledger_epoch(conv_id: str) -> int | None:
    """§17.6a r10 A2: the PERSISTENT epoch. Unlike memory.current_wipe_
    generation (in-process, resets on restart), this survives a restart —
    it is what digests are stamped against and what every reader compares
    to. r10 review R10-B1: returns None (not 0) when the ledger file
    exists but is unreadable — callers must treat that as 'unknown', not
    'never wiped'."""
    ledger = _read_ledger_failing_closed(conv_id, on_fail_context="current_ledger_epoch")
    return int(ledger["epoch"]) if ledger is not None else None


def _write_ledger_best_effort(conv_id: str, data: dict, *, op: str) -> bool:
    """v3.1.9.8 (Task B): an OSError out of `_write_ledger` — an unwritable
    CONTROL_ROOT, a full disk, anything — must never block the deletion the
    user just asked for. This is the SAME principle `record_wipe`'s own
    docstring already states for an unreadable EXISTING ledger ("rather
    than blocking a wipe the user asked for"); before this fix it was not
    extended to a WRITE failure, so `record_wipe` / `record_targeted_
    forget` / `retire_tombstone` would raise straight out of `_write_
    ledger`, the caller (commands._handle_forget's selective branch,
    continuity.wipe_conversation_band for the full wipe) never reached its
    own `save_facts`/delete call, and /forget reported nothing removed
    while her facts were still on disk.

    Round-3 review (the "fix it fully" directive — see the module-level
    PENDING INTENTS comment above `_pending_intents`): `data` is first
    MERGED with anything already pending for this conv
    (`_merge_pending_intent`) — so a write attempt from ANY of the three
    public functions below, or from the startup recovery path, always
    carries forward whatever an earlier failed attempt for this conv
    could not persist. On success, the pending intent (memory AND
    sidecar) is cleared — it was just included in what landed. On
    failure, `_record_pending_intent` folds THIS attempt into the pending
    state (additive, never lossy) and tries to persist it durably;
    logged loudly either way (ERROR for a single failure with the
    sidecar intact — fully recoverable — or CRITICAL for the double-
    failure residual, log_once per conv+op so a permanently broken
    CONTROL_ROOT says so once per conversation rather than once per
    request) and alerted. Returns True if the ledger write itself
    landed.

    Round-4 review (item 1, HIGH): `_merge_pending_intent` now also reads
    the durability sidecar off disk (not just memory), and can report
    `sidecar_unreadable` — the file exists but could not be parsed. When
    that happens on an otherwise-successful write, the sidecar is NEVER
    deleted (its content might not be what `data` just carried — we
    cannot confirm it either way) and the conv is kept marked degraded,
    so the band stays failed closed until the sidecar is repaired or
    replaced by a later, readable one."""
    data, sidecar_unreadable = _merge_pending_intent(conv_id, data)
    try:
        _write_ledger(conv_id, data)
        if sidecar_unreadable:
            _ledger_write_failed.add(conv_id)
            if logsetup.log_once(f"continuity.sidecar_unreadable_kept.{conv_id}"):
                logger.error(
                    f"conv={conv_id}: this write to the wipe ledger during "
                    f"{op} succeeded, but a pending-intent durability "
                    f"sidecar for this conversation exists and could not "
                    f"be read or parsed — its content is unknown and was "
                    f"NOT merged into what just landed. Leaving the "
                    f"sidecar file in place for manual repair rather than "
                    f"deleting it, and keeping the continuity band failed "
                    f"closed for this conversation until it is repaired "
                    f"or superseded by a readable one."
                )
            return True
        had_pending = conv_id in _pending_intents
        _pending_intents.pop(conv_id, None)
        _delete_pending_sidecar(conv_id)
        _ledger_write_failed.discard(conv_id)
        _double_failure_current.discard(conv_id)
        if had_pending and logsetup.log_once(f"continuity.ledger_pending_recovered.{conv_id}"):
            logger.info(
                f"conv={conv_id}: a pending ledger intent (queued after an "
                f"earlier write failure) merged and persisted successfully "
                f"during {op}; the continuity band resumes normal reads for "
                f"this conversation."
            )
        return True
    except OSError as e:
        sidecar_ok = _record_pending_intent(conv_id, data)
        if logsetup.log_once(f"continuity.ledger_write_failed.{op}.{conv_id}"):
            if sidecar_ok:
                logger.error(
                    f"conv={conv_id}: could not persist the wipe ledger "
                    f"during {op} ({type(e).__name__}: {e}); proceeding "
                    f"with the deletion she asked for anyway rather than "
                    f"blocking it. The intended change is queued as a "
                    f"pending intent and saved to a durable sidecar outside "
                    f"CONTROL_ROOT — it will be merged into the next "
                    f"successful ledger write for this conversation, "
                    f"including one retried at startup. The continuity "
                    f"band fails CLOSED for this conversation until then; "
                    f"nothing should reappear."
                )
            else:
                logger.critical(
                    f"conv={conv_id}: DOUBLE FAILURE during {op} — could "
                    f"not persist the wipe ledger ({type(e).__name__}: {e}) "
                    f"AND could not save the durability sidecar either. "
                    f"Proceeding with the deletion she asked for anyway. "
                    f"The pending intent exists ONLY in this process's "
                    f"memory now; the continuity band still fails closed "
                    f"while this process stays up, but a RESTART before "
                    f"either write succeeds would lose it, and the words "
                    f"could then reappear. Reported at /health (checks."
                    f"ledger.double_failure_current) while this lasts."
                )
            try:
                alert.notify(
                    "compactor", "error" if sidecar_ok else "critical",
                    f"wipe ledger write failed for conv={conv_id} during "
                    f"{op}; band fails closed" + (
                        "" if sidecar_ok else
                        " but ONLY in this process's memory (the durability "
                        "sidecar also failed) — a restart would lose it"
                    ),
                )
            except Exception:
                pass
        return False


def _write_wipe_marker_best_effort(conv_id: str, epoch: int, boundary, *, op: str) -> bool:
    """Round-4 review (item 2, HIGH — probe p2.py): `memory.wipe_marker_
    path`'s file is a CACHE (`wipe_conversation_band` deletes it straight
    away as one of `memory.memory_layer_paths`, on every full wipe; it
    exists only so a reader can answer 'what epoch was this wiped at'
    without opening the ledger). Before this fix, `record_wipe` and
    `retire_tombstone` called `memory.atomic_write_json` on it BARE — an
    ENOSPC (or any other OSError) there raised straight out of both
    functions, which are called from `wipe_conversation_band` BEFORE its
    own deletes run. The exception propagated through `_clear_all_memory`
    and `commands._handle_forget`'s selective branch alike, aborting the
    wipe/forget entirely — her facts, episodic memory, summary, persona
    and chapters all stayed on disk, and the log claimed 'proceeding with
    the deletion anyway' one line before the deletion never ran (the
    Task-B/round-2/round-3 principle — a ledger-recording failure must
    never block the deletion she asked for — silently did not extend to
    this SECOND write). A cache write failing is never allowed to cost
    her the wipe it is merely a convenience for: caught, logged once per
    conv+op, and reported — never raised."""
    try:
        memory.atomic_write_json(
            memory.wipe_marker_path(conv_id), {"epoch": epoch, "boundary": boundary}
        )
        return True
    except Exception as e:
        if logsetup.log_once(f"continuity.wipe_marker_write_failed.{op}.{conv_id}"):
            logger.error(
                f"conv={conv_id}: could not write the wipe-marker cache "
                f"during {op} ({type(e).__name__}: {e}); this is a cache "
                f"only (wipe_conversation_band deletes it on every full "
                f"wipe anyway) and never blocks the deletion she asked "
                f"for. A reader that consults this marker before the "
                f"ledger itself may see a stale epoch until the next "
                f"successful write for this conv."
            )
        return False


def record_wipe(conv_id: str, *, boundary: dict | None = None) -> int | None:
    """§17.6a r9/r10: called under conv_lock, by every full-wipe path,
    BEFORE that path's own deletes (see memory.bump_wipe_generation's
    docstring for why the ordering matters — the same reasoning applies
    here). Bumps and persists the ledger epoch; `boundary`, when given, is
    the covered-turn fingerprint pair of the last exchange before the
    wipe (used by a future `.15f`; None means 'everything before now').
    An unreadable EXISTING ledger is not allowed to block a wipe the user
    asked for — it is overwritten with a fresh one starting above 0, and
    logged loudly, rather than refusing /forget.

    v3.1.9.8 round-2 review (item 4, LOW): this used to always return
    `data["epoch"]` — the epoch it INTENDED to persist — even when
    `_write_ledger_best_effort` reported the write failed, so a caller
    trusting the return value would believe a number was on disk that
    never got there. Returns the epoch actually persisted, or None if the
    write failed (check `ledger_degraded(conv_id)` / the log for why).
    `wipe_conversation_band`, the only real caller, does not use the
    return value, so this is a correctness fix with no behaviour change
    downstream today — it matters the moment a future caller starts
    trusting this number."""
    try:
        data = _read_ledger(conv_id)
    except LedgerUnreadable as e:
        logger.error(
            f"conv={conv_id}: wipe ledger unreadable while recording a "
            f"wipe ({e}); starting a fresh ledger rather than blocking "
            f"the wipe she asked for. The corrupt file is overwritten."
        )
        data = dict(_ABSENT_LEDGER)
    data["epoch"] = int(data.get("epoch", 0)) + 1
    data["boundary"] = boundary
    # _write_ledger_best_effort merges any PENDING intent into `data` in
    # place (round-3 review) — a higher pending epoch (from an earlier
    # failed wipe/retire for this same conv) can raise `data["epoch"]`
    # and, with it, `data["boundary"]` past what THIS call intended. The
    # marker write below must follow that merged result, not the raw
    # `boundary` parameter — using the parameter here was the exact
    # staleness bug this whole mechanism exists to close, just moved one
    # write sideways.
    persisted = _write_ledger_best_effort(conv_id, data, op="record_wipe")
    _write_wipe_marker_best_effort(conv_id, data["epoch"], data["boundary"], op="record_wipe")
    return data["epoch"] if persisted else None


def retire_tombstone(conv_id: str) -> int | None:
    """r10 review R10-B3 (MEDIUM): `/retire` must NOT delete the ledger
    file — §17.6a's "needles ... deleted when the conversation is
    retired" was read too literally by the original implementation, which
    would have unlinked `control/wipes/<src>.json` entirely. A later
    restore of a pre-retire archive would then see an ABSENT ledger
    (epoch 0) and bring back every digest with a stamp >= 1, needles and
    all. Instead: bump the epoch (hides every existing digest), set
    boundary="all", and clear the needle list (the needles' job — hiding
    her words — is already done by the epoch bump; keeping them around
    for a retired, about-to-be-inaccessible conversation buys nothing).
    The file itself is never unlinked.

    v3.1.9.8 round-2 review (item 4, LOW): returns the epoch actually
    persisted, or None if the write failed — same reasoning as
    `record_wipe`. `commands.py`'s only caller does not use the return
    value today."""
    try:
        data = _read_ledger(conv_id)
    except LedgerUnreadable as e:
        logger.error(
            f"conv={conv_id}: wipe ledger unreadable during retire ({e}); "
            f"writing a fresh tombstone rather than blocking retire"
        )
        data = dict(_ABSENT_LEDGER)
    data["epoch"] = int(data.get("epoch", 0)) + 1
    data["boundary"] = "all"
    data["targeted"] = []
    # Same reasoning as record_wipe's own comment just above: the merge
    # inside _write_ledger_best_effort can change `data["boundary"]` (and
    # re-add pending needles to `data["targeted"]` — harmless, see that
    # function's docstring), so the marker write follows `data` post-merge
    # rather than the literal "all" this call intended.
    persisted = _write_ledger_best_effort(conv_id, data, op="retire_tombstone")
    _write_wipe_marker_best_effort(conv_id, data["epoch"], data["boundary"], op="retire_tombstone")
    return data["epoch"] if persisted else None


def record_targeted_forget(conv_id: str, needle: str) -> int:
    """§17.6a r10 A3: the needle is kept in the ledger (not just applied to
    today's store) so a restore or import made from a bundle taken BEFORE
    this /forget can re-apply it. Returns the epoch it was recorded at —
    unlike `record_wipe`/`retire_tombstone`, this never BUMPS the epoch,
    so the value returned is the epoch already current (persisted before
    this call, or about to be), never one invented for a failed write.
    Round-2 review (item 4): the honesty gap that fix closes for the other
    two functions does not apply here for that reason. What CAN fail to
    persist is the needle itself, which `ledger_degraded(conv_id)` reports
    after this call — that failure closes the band for this conv via
    `_read_ledger_failing_closed` (item 1), independent of this return
    value."""
    try:
        data = _read_ledger(conv_id)
    except LedgerUnreadable as e:
        logger.error(
            f"conv={conv_id}: wipe ledger unreadable while recording a "
            f"targeted forget ({e}); starting a fresh ledger rather than "
            f"losing the needle she just asked to forget"
        )
        data = dict(_ABSENT_LEDGER)
    epoch = int(data.get("epoch", 0))
    targeted = list(data.get("targeted", []))
    targeted.append({"needle": needle, "epoch": epoch})
    data["targeted"] = targeted
    _write_ledger_best_effort(conv_id, data, op="record_targeted_forget")
    return epoch


def _needle_hit(text: str, needles: list[dict]) -> bool:
    t = (text or "").lower()
    return any((n.get("needle") or "").strip().lower() in t for n in needles if n.get("needle"))


def reapply_ledger(conv_id: str) -> dict:
    """§17.6a r10 A3, strengthened by review R10-B1: the DESIGN calls for
    this to run at the end of every restore (backup.py, admin restore) and
    every import (portability.import_conversation) for each affected
    conversation, in addition to compactor startup. As shipped, ONLY
    `reapply_all_ledgers` (below) calls this — once per conversation with
    a ledger file, at startup (see main.py's lifespan handler) — and
    idempotently, since a no-op ledger just re-derives the same wipe.json
    cache. backup.py and portability.import_conversation do NOT call it
    after a restore/import; a restore that lands while the process stays
    up (rather than restarting it) will not see this re-apply until the
    next boot. This is a documented gap, not a design change — wiring
    those call sites is future work, not done here.

    Purges any digest stamped below the ledger epoch, re-applies every
    targeted needle against what remains — checking the FINAL rendered
    text (headline AND full), not just membership — and rewrites the
    wipe.json cache from the ledger boundary."""
    ledger = _read_ledger_failing_closed(conv_id, on_fail_context="reapply_ledger")
    if ledger is None:
        return {"purged": 0, "suppressed_reapplied": 0, "epoch": None, "failed_closed": True}
    epoch = int(ledger.get("epoch", 0))
    needles = ledger.get("targeted", [])
    store = _read_digest_store(conv_id)
    kept: dict[str, dict] = {}
    purged = 0
    for uid, entry in store.items():
        if int(entry.get("epoch", -1)) != epoch:
            purged += 1
            continue
        kept[uid] = entry
    suppressed = 0
    for entry in kept.values():
        if entry.get("suppressed"):
            continue
        if _needle_hit(entry.get("headline", ""), needles) or _needle_hit(entry.get("full", ""), needles):
            entry["headline"] = SUPPRESSED_MARKER
            entry["full"] = SUPPRESSED_MARKER
            entry["suppressed"] = True
            suppressed += 1
    if purged or suppressed or store != kept:
        _write_digest_store(conv_id, kept)
    memory.atomic_write_json(
        memory.wipe_marker_path(conv_id), {"epoch": epoch, "boundary": ledger.get("boundary")}
    )
    return {"purged": purged, "suppressed_reapplied": suppressed, "epoch": epoch, "failed_closed": False}


def reapply_all_ledgers() -> dict:
    """r10 review R10-B1: run the re-apply at compactor STARTUP, for every
    conversation that has a ledger file. This is the ONLY caller of
    `reapply_ledger` as shipped (see that function's own docstring for the
    still-open gap: backup.py and portability.import_conversation do not
    call it after a restore/import that lands without a restart). Cheap
    and idempotent (a no-op ledger just re-derives the same wipe.json
    cache), and it closes the 'hand restore' gap: an operator who restores
    a directory by hand, outside backup.py/portability.py entirely, gets
    the same protection the moment the compactor next starts.

    v3.1.9.8 round-3 review (DURABILITY ACROSS RESTART): also the ONE
    place `_pending_intents` is repopulated after a restart — the dict
    itself does not survive one, only its sidecars
    (`PENDING_ROOT/ledger-pending/*.json` — round-4 review, item 3, moved
    this off STORAGE_ROOT) do. Each sidecar found is
    loaded into `_pending_intents`, then a real write is attempted for
    that conv (reusing `_write_ledger_best_effort`, so the merge, the
    sidecar deletion on success, and the pending/degraded bookkeeping are
    all the SAME code path a live request would use — nothing here is a
    special case). If the write STILL fails (the same unwritable
    CONTROL_ROOT survived the restart too), the conv stays marked
    degraded and the sidecar stays in place for the next attempt — the
    band keeps failing closed for it, exactly as it should.

    Round-4 review (item 1, HIGH — probe N4): the wipes-dir scan and the
    pending-sidecar recovery below are INDEPENDENT concerns (a stale
    digest purge vs. an unpersisted forget/wipe), so a fault in one must
    never take the other down with it. Before this, `wipes_dir.is_dir()`
    raising (e.g. a permission fault on CONTROL_ROOT) propagated straight
    out of this function — the pending-sidecar loop never ran at all,
    and every conv with a sidecar sitting on disk from before the restart
    stayed unrecovered (though still protected in the interim by
    `_has_pending_intent`'s own file check). The wipes-dir scan is now
    wrapped so a failure there is logged and this function proceeds to
    the pending-sidecar loop regardless."""
    wipes_dir = CONTROL_ROOT / "wipes"
    n = 0
    try:
        if wipes_dir.is_dir():
            for p in wipes_dir.glob("*.json"):
                conv_id = p.stem
                try:
                    reapply_ledger(conv_id)
                    n += 1
                except Exception as e:
                    logger.warning(f"conv={conv_id}: startup ledger re-apply failed (non-fatal): {e}")
    except Exception as e:
        logger.error(
            f"continuity: the wipe-ledger directory scan itself failed at "
            f"startup ({type(e).__name__}: {e}); this only affects the "
            f"per-conv digest-purge re-apply pass — the pending-intent "
            f"sidecar recovery below still runs regardless."
        )
    n_pending = 0
    try:
        pending_dir = PENDING_ROOT / "ledger-pending"
        _pending_dir_is_dir = pending_dir.is_dir()
    except Exception as e:
        logger.critical(
            f"continuity: could not even check whether the pending-ledger "
            f"sidecar directory exists at startup ({type(e).__name__}: "
            f"{e}); any pending intents left over from before this "
            f"restart could NOT be recovered this boot. Their sidecar "
            f"files, if any, remain on disk — a per-conv "
            f"_has_pending_intent stat still protects each one "
            f"individually as long as ITS OWN path can be reached. "
            f"Investigate STORAGE_ROOT permissions."
        )
        _pending_dir_is_dir = False
    if _pending_dir_is_dir:
        for p in pending_dir.glob("*.json"):
            conv_id = p.stem
            try:
                pending = json.loads(p.read_text(encoding="utf-8"))
                if not isinstance(pending, dict):
                    raise ValueError("pending ledger sidecar is not a JSON object")
            except Exception as e:
                # Residual within a residual: the sidecar itself is now
                # unreadable. Nothing to merge from it — left in place
                # (never deleted on a read failure) in case a later manual
                # repair can recover it; the conv is NOT marked pending
                # from this file, but it may already be marked from
                # elsewhere (e.g. still in `_ledger_write_failed` this
                # same boot, which cannot happen at true startup, or a
                # concurrent live failure) — either way `_has_pending_
                # intent`'s file-existence check still fails the band
                # closed for it, since the file is still there.
                logger.error(
                    f"conv={conv_id}: pending ledger sidecar unreadable at "
                    f"startup ({type(e).__name__}: {e}); its intent cannot "
                    f"be recovered from this file. The file is left in "
                    f"place — its mere presence still fails the continuity "
                    f"band closed for this conversation via "
                    f"_has_pending_intent."
                )
                continue
            _pending_intents[conv_id] = {
                "epoch": pending.get("epoch"),
                "boundary": pending.get("boundary"),
                "needles": list(pending.get("needles") or []),
            }
            try:
                data = _read_ledger(conv_id)
            except LedgerUnreadable:
                data = dict(_ABSENT_LEDGER)
            if _write_ledger_best_effort(conv_id, data, op="startup_reapply_pending"):
                n_pending += 1
            # else: still failing — _write_ledger_best_effort already
            # re-marked the conv degraded, re-persisted the sidecar (a
            # harmless rewrite of the same content) and logged loudly.
    if n:
        logger.info(f"continuity: re-applied the wipe ledger for {n} conversation(s) at startup")
    if n_pending:
        logger.info(
            f"continuity: recovered {n_pending} pending ledger intent(s) "
            f"from durability sidecars at startup"
        )
    return {"conversations": n, "pending_recovered": n_pending}


# ---------------------------------------------------------------------------
# The digest store (`summaries/<conv>.digests.json`).
# ---------------------------------------------------------------------------

def unit_key(fp_user: str | None, fp_assistant: str | None) -> str:
    """§17.6a: 'keyed by content, not position: the pair of covered-turn
    fingerprints of her message and the reply.' Either half may be absent
    (a gap unit made of only one side, at the gap's first turn)."""
    return f"{fp_user or ''}\u0001{fp_assistant or ''}"


def _read_digest_store(conv_id: str) -> dict:
    try:
        data = memory.read_json_strict(memory.digests_path(conv_id), default={})
    except memory.StoreUnreadable as e:
        if logsetup.log_once(f"continuity.store_unreadable.{conv_id}"):
            logger.error(
                f"conv={conv_id}: digest store unreadable ({e}); the band "
                f"falls back to the extractive summary for every gap unit "
                f"this request, and the digest job will rewrite the file "
                f"next time it runs. Left in place — not overwritten from "
                f"an unknown state."
            )
        return {}
    return data if isinstance(data, dict) else {}


def _write_digest_store(conv_id: str, data: dict) -> None:
    memory.atomic_write_json(memory.digests_path(conv_id), data)


SUPPRESSED_MARKER = "[an earlier exchange; details removed at her request]"


async def generate_digest(
    conv_id: str,
    *,
    tok: Any,
    her_text: str,
    reply_text: str,
    unit_number: int,
    key: str,
    epoch_at_submit: int,
    vllm_url: str | None,
    model: str | None,
    http_client: "httpx.AsyncClient | None" = None,
) -> bool:
    """§17.6a: made in the background, after the reply, on the memory tail,
    before facts extraction, NEVER on the request path. Re-checks the
    ledger epoch under conv_lock — the same shape memory.bump_wipe_
    generation's callers already use for facts/summaries — and discards
    the write on a mismatch (a wipe landed while this call was in
    flight). Returns True if written, False if discarded or suppressed."""
    text = await make_digest_text(
        tok, her_text, reply_text, unit_number, vllm_url=vllm_url, model=model,
        http_client=http_client,
    )
    headline = cut(tok, text.splitlines()[0] if text else "", DIGEST_HEADLINE_TOKENS)
    full = cut(tok, text, DIGEST_FULL_TOKENS)
    async with memory.conv_lock(conv_id):
        ledger = _read_ledger_failing_closed(conv_id, on_fail_context="generate_digest")
        if ledger is None:
            # Fail closed: an unreadable ledger means we do not know
            # whether this unit is covered by a wipe boundary or a
            # needle, so the digest is not written at all this round.
            return False
        epoch_now = int(ledger["epoch"])
        if epoch_now != epoch_at_submit:
            logger.info(
                f"conv={conv_id}: digest for unit {unit_number} discarded — "
                f"the wipe epoch moved from {epoch_at_submit} to {epoch_now} "
                f"while this job was in flight"
            )
            return False
        store = _read_digest_store(conv_id)
        existing = store.get(key)
        if existing and existing.get("suppressed"):
            # §17.6a r9: "it is never re-digested" once a targeted forget
            # has suppressed it.
            return False
        # r10 review R10-B1: "the digest writer applies [the needle
        # check] before storing" — a targeted forget that landed for a
        # DIFFERENT unit could still name a word that also appears in
        # THIS one; check the full ledger needle list, not just this
        # unit's own prior state.
        if _needle_hit(headline, ledger.get("targeted", [])) or _needle_hit(full, ledger.get("targeted", [])):
            headline = full = SUPPRESSED_MARKER
            store[key] = {
                "headline": headline, "full": full, "epoch": epoch_now,
                "unit_number": unit_number, "suppressed": True,
            }
            _write_digest_store(conv_id, store)
            return False
        store[key] = {
            "headline": headline,
            "full": full,
            "epoch": epoch_now,
            "unit_number": unit_number,
            "suppressed": False,
        }
        _write_digest_store(conv_id, store)
        return True


def suppress_needle(conv_id: str, needle: str) -> int:
    """§17.6a r9, and r10 review R10-B2: `/forget <text>` — case-
    insensitive, the same rule facts already applies. MUST be called
    under conv_lock (the caller already holds it for the matching facts
    removal), and BEFORE any early return the command takes on "no facts
    matched" (R10-B2) — a needle can live only in a digest, never in a
    fact, and the old ordering silently skipped exactly that case.
    Records the needle in the ledger (r10 A3) so a later restore/import
    re-applies it, even to units that get no digest until after the
    restore. Returns the number of units suppressed just now (0 is not
    an error: the needle is recorded in the ledger regardless, so a unit
    digested later still comes out suppressed)."""
    needle_l = needle.strip().lower()
    if not needle_l:
        return 0
    epoch = record_targeted_forget(conv_id, needle)
    store = _read_digest_store(conv_id)
    n = 0
    for entry in store.values():
        if entry.get("suppressed"):
            continue
        if int(entry.get("epoch", -1)) != epoch:
            continue
        text = f"{entry.get('headline', '')} {entry.get('full', '')}".lower()
        if needle_l in text:
            entry["headline"] = SUPPRESSED_MARKER
            entry["full"] = SUPPRESSED_MARKER
            entry["suppressed"] = True
            n += 1
    if n:
        _write_digest_store(conv_id, store)
    return n


def visible(conv_id: str) -> dict[str, dict]:
    """§17.6a r10, strengthened by review R10-B1: the ONE reader. The
    band-injection call site, /why and every admin or diagnostic endpoint
    must call this rather than reading the digest store directly.

      - Reads the ledger FRESH (no process cache) on every call.
      - EQUALITY: a digest is visible only if its stamp == the ledger
        epoch, not >=. An entry from an OLDER epoch is exactly as stale
        as one from a newer one would be suspicious — both mean this
        entry was not produced under today's ledger state.
      - Applies every ledger needle to the FINAL rendered text
        (headline and full), not just to whatever the entry's own
        `suppressed` flag says — a needle recorded after this entry was
        written (a /forget that landed while a digest job for a
        different, unrelated unit was in flight and got written
        un-suppressed before the needle existed) is still caught here.
      - Fails CLOSED on an unreadable ledger: returns {} rather than
        guessing epoch 0, so nothing older than the verbatim floor is
        shown until the ledger is repaired.
    """
    ledger = _read_ledger_failing_closed(conv_id, on_fail_context="visible")
    if ledger is None:
        return {}
    epoch = int(ledger.get("epoch", 0))
    needles = ledger.get("targeted", [])
    store = _read_digest_store(conv_id)
    out: dict[str, dict] = {}
    for uid, entry in store.items():
        if int(entry.get("epoch", -1)) != epoch:
            continue
        rendered = dict(entry)
        if not rendered.get("suppressed") and (
            _needle_hit(rendered.get("headline", ""), needles)
            or _needle_hit(rendered.get("full", ""), needles)
        ):
            rendered["headline"] = SUPPRESSED_MARKER
            rendered["full"] = SUPPRESSED_MARKER
            rendered["suppressed"] = True
        out[uid] = rendered
    return out


def apply_needles(conv_id: str, text: str) -> str:
    """r10 review R10-B1: 'the allocator applies the same [needle] check
    to fallback entries it builds.' For an extractive-fallback entry
    built OUTSIDE the digest store (no digest exists yet for this unit),
    this is what a caller must run the rendered text through before
    injecting it. Fails closed the same way `visible()` does: if the
    ledger cannot be read, the text is suppressed rather than risked."""
    ledger = _read_ledger_failing_closed(conv_id, on_fail_context="apply_needles")
    if ledger is None:
        return SUPPRESSED_MARKER
    if _needle_hit(text, ledger.get("targeted", [])):
        return SUPPRESSED_MARKER
    return text


def wipe_conversation_band(conv_id: str, *, boundary: dict | None = None) -> None:
    """§17.6a r9/r10: call from every full-wipe path (main._clear_all_
    memory, /retire's apply step, portability.import_conversation with
    overwrite=True), under conv_lock, AFTER memory.bump_wipe_generation
    and BEFORE returning — matching the ordering memory.bump_wipe_
    generation's own docstring requires for every other layer. Bumps the
    PERSISTENT ledger epoch (survives a restart, unlike the in-process
    generation alone) and deletes every path in memory.memory_layer_
    paths."""
    record_wipe(conv_id, boundary=boundary)
    for p in memory.memory_layer_paths(conv_id):
        try:
            if p.is_file():
                p.unlink()
        except Exception as e:
            logger.warning(f"conv={conv_id}: could not delete {p.name} during wipe: {e}")


# ---------------------------------------------------------------------------
# Band formatting (§17.6a: "Format in the prompt").
# ---------------------------------------------------------------------------

BAND_HEADER = "Recent exchanges not yet summarized (oldest first):"


def excess_marker(n: int) -> str:
    """§17.6: 'the oldest units are replaced by one line ... and an alert
    fires.' Only reachable when even the headlines cannot all fit."""
    return f"[{n} earlier exchanges are not summarized yet]"


def format_band_entry(entry: dict, *, full: bool) -> str:
    if entry.get("suppressed"):
        return SUPPRESSED_MARKER
    text = entry.get("full") if full else entry.get("headline")
    return text or entry.get("headline") or SUPPRESSED_MARKER


def format_band(units_oldest_first: list[dict], *, full_n: int = DIGEST_FULL_UNITS) -> str:
    """`units_oldest_first`: resolved digest entries (already through
    `visible()` plus the extractive fallback for any unit still missing
    one), oldest first. The newest `full_n` units get the full digest;
    the rest, the headline. r9: no per-entry time prefix."""
    if not units_oldest_first:
        return ""
    n = len(units_oldest_first)
    lines = [BAND_HEADER]
    for i, u in enumerate(units_oldest_first):
        full = (n - i) <= full_n
        lines.append(f"{i + 1}. " + format_band_entry(u, full=full))
    return "\n".join(lines)
