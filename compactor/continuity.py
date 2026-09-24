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


class ControlRootInsideStorageRoot(RuntimeError):
    """§17.6a r10: 'The compactor refuses to start if the control root is
    inside STORAGE_ROOT.' Raised by `assert_control_root_safe`, which
    main.py must call once at boot before serving any request."""


def assert_control_root_safe() -> None:
    storage = memory.STORAGE_ROOT
    storage_abs = storage.resolve() if storage.exists() else storage.absolute()
    control_abs = CONTROL_ROOT.resolve() if CONTROL_ROOT.exists() else CONTROL_ROOT.absolute()
    try:
        control_abs.relative_to(storage_abs)
    except ValueError:
        return
    raise ControlRootInsideStorageRoot(
        f"COMPACTOR_CONTROL_ROOT ({control_abs}) resolves inside "
        f"COMPACTOR_STORAGE_ROOT ({storage_abs}). The wipe ledger must live "
        f"outside backup/restore scope (§17.6a r10 A3) or a restore could "
        f"silently bring back what a /forget already removed. Refusing to "
        f"start; set COMPACTOR_CONTROL_ROOT to a path outside the storage "
        f"root."
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
    were decoration-sourced under the un-amended r10 filter)."""
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
    entry = cut(tok, f"{unit_number % 100}. " + body, DIGEST_HEADLINE_TOKENS)
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
        r = await client.post(f"{vllm_url}/v1/chat/completions", json=payload, timeout=30)
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
    older than the verbatim floor, rather than guess."""
    try:
        return _read_ledger(conv_id)
    except LedgerUnreadable as e:
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


def record_wipe(conv_id: str, *, boundary: dict | None = None) -> int:
    """§17.6a r9/r10: called under conv_lock, by every full-wipe path,
    BEFORE that path's own deletes (see memory.bump_wipe_generation's
    docstring for why the ordering matters — the same reasoning applies
    here). Bumps and persists the ledger epoch; `boundary`, when given, is
    the covered-turn fingerprint pair of the last exchange before the
    wipe (used by a future `.15f`; None means 'everything before now').
    An unreadable EXISTING ledger is not allowed to block a wipe the user
    asked for — it is overwritten with a fresh one starting above 0, and
    logged loudly, rather than refusing /forget."""
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
    _write_ledger(conv_id, data)
    memory.atomic_write_json(
        memory.wipe_marker_path(conv_id), {"epoch": data["epoch"], "boundary": boundary}
    )
    return data["epoch"]


def retire_tombstone(conv_id: str) -> int:
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
    The file itself is never unlinked."""
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
    _write_ledger(conv_id, data)
    memory.atomic_write_json(
        memory.wipe_marker_path(conv_id), {"epoch": data["epoch"], "boundary": "all"}
    )
    return data["epoch"]


def record_targeted_forget(conv_id: str, needle: str) -> int:
    """§17.6a r10 A3: the needle is kept in the ledger (not just applied to
    today's store) so a restore or import made from a bundle taken BEFORE
    this /forget can re-apply it. Returns the epoch it was recorded at."""
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
    _write_ledger(conv_id, data)
    return epoch


def _needle_hit(text: str, needles: list[dict]) -> bool:
    t = (text or "").lower()
    return any((n.get("needle") or "").strip().lower() in t for n in needles if n.get("needle"))


def reapply_ledger(conv_id: str) -> dict:
    """§17.6a r10 A3, strengthened by review R10-B1: call at the END of
    every restore (backup.py, admin restore) and every import
    (portability.import_conversation) for each affected conversation —
    AND at compactor startup, for every conversation with a ledger file
    (idempotent). Purges any digest stamped below the ledger epoch,
    re-applies every targeted needle against what remains — checking the
    FINAL rendered text (headline AND full), not just membership — and
    rewrites the wipe.json cache from the ledger boundary."""
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
    """r10 review R10-B1: run the re-apply at compactor STARTUP too, for
    every conversation that has a ledger file — not only after a restore
    or import. Cheap and idempotent (a no-op ledger just re-derives the
    same wipe.json cache), and it closes the 'hand restore' gap: an
    operator who restores a directory by hand, outside backup.py/
    portability.py entirely, gets the same protection the moment the
    compactor next starts."""
    wipes_dir = CONTROL_ROOT / "wipes"
    if not wipes_dir.is_dir():
        return {"conversations": 0}
    n = 0
    for p in wipes_dir.glob("*.json"):
        conv_id = p.stem
        try:
            reapply_ledger(conv_id)
            n += 1
        except Exception as e:
            logger.warning(f"conv={conv_id}: startup ledger re-apply failed (non-fatal): {e}")
    if n:
        logger.info(f"continuity: re-applied the wipe ledger for {n} conversation(s) at startup")
    return {"conversations": n}


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
