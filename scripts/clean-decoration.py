#!/usr/bin/env python3
"""Strip emoji, rule-line "walls", and status-board debris out of the
three places the model re-reads its own decoration from — her chat's
recent stored replies, the retrieved episodic exchanges, and the facts
that happen to be status-board lines — WITHOUT running `/forget` and
without touching a single word she wrote. Deterministic, no LLM. See
/tmp/zl/degeneration-2026-09-23.md for the forensic report this closes
part of (the "clean up her last four replies" and "clean the decorated
facts" items).

SUPPORTED INVOCATION (same clone-not-copy reasoning as backfill-records.py
and import-history.py — see PACKAGE RESOLUTION below):

    git clone --depth 1 --branch <tag-or-branch> \\
        https://github.com/MrBanana8768/zions-light-ai.git /opt/zl-repo
    supervisorctl stop compactor
    supervisorctl stop openwebui
    /opt/compactor-venv/bin/python /opt/zl-repo/scripts/clean-decoration.py \\
        --webui-db /data/openwebui/webui.db --store /data/openwebui/compactor \\
        --conv ea1494ea-e9d7-46fb-8b7c-3a50d685d00e
    /opt/compactor-venv/bin/python /opt/zl-repo/scripts/clean-decoration.py \\
        --webui-db /data/openwebui/webui.db --store /data/openwebui/compactor \\
        --conv ea1494ea-e9d7-46fb-8b7c-3a50d685d00e --apply
    supervisorctl start compactor
    supervisorctl start openwebui

Dry run (the default) never writes anything, under any target, under any
circumstance. `--apply` is required to write.

WHY BOTH SERVICES STOP. `facts/`, `summaries/` and `chromadb/` under
`--store` are live files the compactor reads and writes on every request;
`webui.db` is the same live file OpenWebUI itself reads and writes on
every request. Cleaning any of them out from under a running process is
the same hazard `import-history.py` and `backfill-records.py` already
refuse — this script refuses the same way (`--force` overrides the
refusal, never the underlying risk). This is a SHORT chat outage: the
dry run makes no writes and completes in well under a minute; `--apply`
on the three targets together, on a store this size, is a few seconds of
actual I/O plus (for `--targets episodic`) one CPU embedding pass per
changed document — call it under two minutes end to end for her store.

PACKAGE RESOLUTION. Same order, same reasoning, as the other operator
scripts (`--compactor-pkg` > `HERE.parent/"compactor"` > `/opt/compactor`
> an already-importable module on sys.path) — see
`_resolve_compactor_pkg`'s docstring below. This script also needs
`scripts/import-history.py` BESIDE itself (not the compactor package) for
`_resolve_resume_offset` and the branch-reconstruction helpers — reused
rather than re-implemented, for the same "two copies of one rule drifting
apart" reason backfill-records.py's docstring gives for not re-deriving
`is_stale` itself. If that sibling script is missing or its shape has
changed underneath this one, this script refuses with an actionable
message rather than guess.

THE HAZARD THIS SCRIPT IS BUILT AROUND. `compactor/summarizer.py`
identifies where it is in a conversation by fingerprinting the WHOLE,
whitespace-normalized text of each turn (`_turn_fingerprints`,
`_covered_turn_fingerprints` — verified by reading the frozen v3.1.9
source, not assumed: neither function strips emoji, rule lines, or any
other decoration before hashing). Rewriting a turn's text changes its
fingerprint. Two different fingerprint records depend on this:

  - `tail_fp`/`head_fp`/`window_turns` — the alignment anchor
    `_observed_position` compares the client's next window against. If
    this script rewrites a turn inside that anchor and does not also
    rewrite the anchor to match, the NEXT live request (which will
    naturally re-send the now-cleaned text, because that is what is now
    stored in webui.db) fails to align against the stale anchor, falls
    back to a flat `_ASSUMED_NEW_TURNS`-per-call guess, and that is a
    permanent hole in the summary hierarchy from then on (the exact
    `window_offset` trap `import-history.py`'s B1 fix closed).
  - `covered_fps` — the per-position record of every turn already folded
    into an L1/L2/L3 summary. Rewriting a turn inside that region makes
    the record disagree with the transcript, and `import-history.py`'s
    own `_resolve_resume_offset` (which this script reuses) would then
    refuse to find a resume offset for it, or misalign one, on its own
    hostile-review-verified logic.

So this script:
  1. NEVER rewrites a turn inside the covered/summarized region. Default
     scope is the last `--last N` (default 6) assistant replies; with
     `--since-summarized` the whole region strictly after the store's own
     `last_summarized_turn`, mapped to a branch position with the exact
     same verified offset arithmetic `import-history.py` uses. Both are,
     by construction, entirely outside `covered_fps`.
  2. After cleaning, RECOMPUTES `tail_fp`/`head_fp`/`window_turns` with
     the frozen summarizer's own `_turn_fingerprints`, against the
     CLEANED branch, and writes the new anchor atomically (backed up
     first) — but only when it actually changed. The goal, and what the
     real-image test proves: the next live request aligns at exactly the
     position and `window_offset` it would have without this script ever
     having run.

TARGETS (each behind `--only`, default = all three; each independently
backed up before any write):

  webui   Her current branch in `chat.chat` (walking `currentId` via
          `parentId`, the same linear branch OpenWebUI shows — never
          insertion order). OpenWebUI 0.11 keeps THREE copies of an
          in-scope assistant turn's text and this script updates all
          three or none: `chat.chat.history.messages[id].content` (the
          tree), `chat.chat.messages[i].content` for the small tail
          array OpenWebUI also caches inline in the same JSON blob
          (verified against the real 09-23 backup: 8 entries, a subset
          of `history.messages`' ids, byte-identical content — updating
          only the tree copy would leave this one stale), and the
          `chat_message` table's own row (id `f"{chat_id}-{msg_id}"`,
          content column byte-identical to the tree copy on every row
          checked). Never touches a `role: "user"` message. Refuses if a
          `-wal`/`-journal` file sits beside the database, or if
          OpenWebUI answers on `--openwebui-port` (default 8080) or
          `supervisorctl status openwebui` reports it RUNNING, unless
          `--force`. Backed up via sqlite3's own online backup API
          (`Connection.backup`), after a free-space check (the DB is
          large — 500MB on her export — and lives on a network volume).
          One transaction for every row changed; `PRAGMA integrity_check`
          run and checked immediately after commit, before this script
          reports success.
  facts   `facts/<conv>.json`'s active facts. `retrieval._embed` and
          `facts.select_for_injection` embed fact text FRESH on every
          call (verified by reading facts.py/retrieval.py in both
          v3.1.9 and v3.1.9.4: no cached embedding or content hash lives
          on a fact record), so there is no cache to invalidate here —
          cleaning the text is the whole job. If cleaning would make a
          fact's text empty, or a duplicate of another fact's text, that
          fact is reported and left EXACTLY as it was — the owner does
          not want `/forget` semantics, and turning a fact silently
          empty is exactly that. Written through the package's own
          `memory.atomic_write_json`.
  episodic  The chromadb documents for `--conv`. A document's id is
          content-addressed (`retrieval._doc_id`, verified: sha256 of the
          exact stored text) — cleaning the text changes the id, not
          just the row, so this script computes the new id, embeds the
          NEW text with the same `retrieval._embed` (the exact call the
          live compactor makes; same model, same code path — not a
          second copy of the embedding logic) the compactor itself uses,
          upserts under the new id with the ORIGINAL row's metadata
          (`conv_id`, `turn_index`), and deletes the old id. Only the
          `[assistant]: ...` half of a stored exchange is ever cleaned;
          the `[user]: ...` half is never touched, matching the
          "never her words" rule everywhere else in this script. The
          whole `chromadb/` directory is copied for backup before any
          write (chroma keeps its own sqlite file plus per-collection
          segment directories; there is no smaller unit to back up
          safely). If the resolved package's chromadb cannot be opened
          at all (missing dependency, version mismatch) this target is
          skipped and reported, never guessed at — the owner can use
          `COMPACTOR_RAG_ENABLED=false` meanwhile.

EXIT CODES — the same 5-value convention as the other v3.1.9.6 operator
scripts (OPERATIONS.md, "Exit codes — the shared convention"):
    0   the desired end state is in place: a dry run with nothing to
        change on any requested target, or an `--apply` that changed
        every target it was asked to and refused nothing.
    1   a refusal or error the operator needs to look at (bad --store/
        --webui-db/--conv, no usable compactor package, a live
        OpenWebUI/compactor without --force, a backup path that already
        exists, PRAGMA integrity_check failing after a write), OR an
        `--apply` that changed NONE of the targets it was asked to
        (indistinguishable from doing nothing otherwise).
    2   argparse usage error (the standard library's own convention).
    3   a DRY RUN found decoration to remove on at least one requested
        target and nothing refused outright — re-run with --apply.
    4   `--apply` changed at least one target but at least one other
        requested target was refused, failed, or (for `episodic`)
        skipped for a documented reason.

`--restore <stamp>` puts every target this script backed up under that
stamp back exactly as they were, and exits 0 if every backup found was
restored, 1 if any named target's backup was missing or restoring it
failed.
"""

import argparse
import errno
import hashlib
import importlib.util
import json
import os
import re
import shutil
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
OPT_COMPACTOR = Path("/opt/compactor")

DEFAULT_STORE = "/data/openwebui/compactor"
DEFAULT_WEBUI_DB = "/data/openwebui/webui.db"
DEFAULT_HEALTH_URL = "http://127.0.0.1:8080/health"
DEFAULT_OPENWEBUI_PORT = 8080
HEALTH_PROBE_TIMEOUT_S = 3
DEFAULT_LAST_N = 6
ALL_TARGETS = ("webui", "facts", "episodic")

_PROBE_MODULE = "summarizer"

_CLONE_HINT = """\
On a pod, the supported way to run this script is from a clone of this
public repo (git ships in the image; there is no ssh/scp/rsync in it):

  git clone --depth 1 --branch <tag-or-branch> \\
      https://github.com/MrBanana8768/zions-light-ai.git /opt/zl-repo
  /opt/compactor-venv/bin/python /opt/zl-repo/scripts/clean-decoration.py \\
      --webui-db /data/openwebui/webui.db --store /data/openwebui/compactor \\
      --conv <chat id>

A clone always carries a self-consistent target-release `compactor`
package AND `scripts/import-history.py` beside this script, which is the
one thing a bare copy of just this file cannot supply for itself."""


# ===========================================================================
# PART 1 — the pure cleaning function. No I/O, no network, no LLM.
# Deterministic: clean_text(x) == clean_text(x) always, and
# clean_text(clean_text(x)) == clean_text(x) (see the idempotency test in
# compactor/test_clean_decoration_script.py).
# ===========================================================================

# Every Unicode range actually needed to catch what the forensic report
# measured (emoji bullets, ✅ ✔ ☑ ▶️ ⚡ and the like) — never Hebrew or any
# other script (nowhere near these ranges), never em/en dash, ellipsis, or
# ASCII punctuation (also nowhere near these ranges).
_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F000-\U0001FFFF"   # emoticons, symbols, pictographs, flags, …
    "☀-⛿"           # misc symbols (☑ ⚡ ☀ …)
    "✀-➿"           # dingbats (✅ ✔ ✂ …)
    "■-◿"           # geometric shapes (▶ ▲ ● …)
    "⬀-⯿"           # misc symbols and arrows (⭐ ➡ …)
    "︀-️"           # variation selectors (text/emoji presentation)
    "‍"                  # ZWJ (emoji sequences)
    "⃣"                  # combining enclosing keycap
    "]"
)

# A line consisting ONLY of these characters, length >= 4, is a decoration
# rule (the forensic report's "60-66 character rule lines"). Deliberately
# narrow: no ASCII space tolerated inside the run (real observed rule lines
# never have one), and no dash/bar character that also does ordinary
# punctuation duty (em dash U+2014, en dash U+2013, horizontal bar U+2015
# are NOT in this set — "ordinary punctuation... must not be altered").
# The hyphen is placed first inside the class so it is never misread as a
# range operator. U+FFFD (the Unicode replacement character) is included:
# found in a real episodic document from her 09-23 store — a single
# corrupted byte sitting in the MIDDLE of an otherwise-obvious 60-char
# "═" rule line ("═══════<FFFD>══════...") — which, without this,
# fails the "every character is a rule character" test over one stray
# byte and survives cleaning whole. A replacement character carries no
# meaning of its own to preserve.
_RULE_LINE_RE = re.compile(r"^[-=_~*�─-╿]{4,}$")

# "Strip trailing status tags like '→ ACTIVE (100%)'" — exact-pattern and
# conservative by design (the spec's own words). Configurable: a caller
# (the CLI, or a test) can extend this tuple.
DEFAULT_STATUS_TAG_PATTERNS = (
    re.compile(r"[ \t]*(?:→|->)\s*\bACTIVE\b\s*\(\s*100%\s*\)\s*$"),
    re.compile(r"[ \t]*[-–—]\s*\bACTIVE\b\s*\(\s*100%\s*\)\s*$"),
    # Zero-or-more leading whitespace, not one-or-more: an emoji bullet
    # right before "ACTIVE" (e.g. "✅ ACTIVE (100%)") has already had its
    # own leading space eaten by _strip_emoji's cleanup by the time this
    # runs, so requiring at least one space here would silently never
    # fire on the single most common real shape. `\b` on both sides of
    # ACTIVE stops this from ever matching inside a longer word such as
    # "INACTIVE (100%)" or "PROACTIVE (100%)".
    re.compile(r"[ \t]*\bACTIVE\b\s*\(\s*100%\s*\)\s*$"),
)

_FENCE_RE = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)

_CODE_SIGN_PATTERNS = (
    # Deliberately CASE-SENSITIVE, lowercase-only, for the general keyword
    # list: real code overwhelmingly writes these lowercase at statement
    # start (`def foo():`, `protected void bar()`), while prose capitalizes
    # the first word of a line/sentence ("Protected By: Father, Jesus...").
    # Verified the hard way: `re.IGNORECASE` here classified a genuine,
    # heavily decorated status board inside a real ``` fence as "real
    # code" purely because its line "Protected By: Father, Jesus
    # (continuous, uninterrupted)" starts with the OOP-modifier keyword
    # "protected" — and a fence judged as code is left completely
    # untouched (see the module docstring's "Must NOT alter... real code
    # fences"), so this one false positive left 13 of her real episodic
    # documents with a full, uncleaned status board inside. SQL keywords
    # get their own (already-uppercase) pattern below, unaffected.
    re.compile(
        r"^\s*(def|class|import|from|function|const|let|var|return|elif|"
        r"package|namespace|public|private|protected|async\s+def|await|"
        r"try:|except\b|finally:|#include)\b"
    ),
    re.compile(r"\b(SELECT|INSERT|UPDATE|DELETE|CREATE)\b\s+\S"),
    re.compile(r"^\s*#!/"),
    re.compile(r"[{};]\s*$"),
    re.compile(r"=>|::|</\w+>|/>"),
    re.compile(r"^\s*(if|for|while)\s*\("),
)


def _looks_like_code(content: str) -> bool:
    """True if ANY line of `content` matches a recognisable code shape.
    Deliberately permissive (one hit is enough) — the cost of a false
    negative (an unwrapped fence that really held code) is much lower than
    a false positive (a status board saved from unwrapping because one
    line happened to end in a semicolon); the forensic report's actual
    fence content is either real code or a decoration board, never a
    near-miss between the two."""
    return any(pat.search(line) for line in content.split("\n") for pat in _CODE_SIGN_PATTERNS)


def _unwrap_noncode_fences(text: str, protected: list) -> str:
    """Real code fences are replaced with an opaque placeholder (appended
    to `protected`, restored byte-for-byte at the very end of
    `clean_text`) so nothing later in the pipeline can touch them — not
    emoji-strip, not rule-line removal, not dedup. A fence whose content
    has no code-like line has its ``` delimiters removed; the inner text
    is left in place to be cleaned exactly like ordinary prose."""

    def repl(m: "re.Match") -> str:
        content = m.group(2)
        if _looks_like_code(content):
            protected.append(m.group(0))
            return f"\x00PROTECTED{len(protected) - 1}\x00"
        inner = content[:-1] if content.endswith("\n") else content
        return inner

    return _FENCE_RE.sub(repl, text)


def _strip_emoji(line: str) -> str:
    new = _EMOJI_PATTERN.sub("", line)
    if new == line:
        return line
    # Only a line that actually HAD an emoji has its leftover whitespace
    # tidied — a bullet emoji's leading space, or a doubled space where an
    # inline emoji sat between two words. Every other line is untouched.
    new = new.lstrip(" \t")
    new = re.sub(r"[ \t]{2,}", " ", new)
    return new.rstrip()


def _is_rule_line(line: str) -> bool:
    s = line.strip()
    return bool(s) and bool(_RULE_LINE_RE.match(s))


def _dedupe_consecutive_nonblank(lines: list) -> list:
    """Collapse identical consecutive duplicate lines. Blank lines are
    deliberately exempt — that is DEFAULT_STATUS_TAG_PATTERNS's sibling
    rule ("collapse 3+ blank lines to 1"), a different threshold, and
    conflating the two would fight over how many blank lines survive."""
    out: list = []
    for ln in lines:
        if ln.strip() and out and out[-1] == ln:
            continue
        out.append(ln)
    return out


def clean_text(text: str, extra_status_patterns: tuple = ()) -> str:
    """The pure decoration stripper. `text` in, cleaned `text` out — no
    I/O, nothing else. See the module docstring's TARGETS section and the
    rules enumerated in the project brief:
      - remove emoji/pictographs/variation-selectors/ZWJ/keycaps and the
        ✅ ✔ ☑ ▶️ ⚡ family;
      - remove rule lines (repeated box-drawing or =/-/_/~/* runs, >= 4);
      - unwrap a ``` fence with no code-like line to plain text;
      - collapse 3+ blank lines to 1;
      - strip configured trailing status tags;
      - collapse identical consecutive duplicate (non-blank) lines.
    Never touches ordinary punctuation (em/en dash, ellipsis), any
    non-Latin script, asterisk emphasis around real words, quoted text,
    numbers, or the content of a real code fence — none of those fall
    inside any of the patterns above, by construction, not by exception.
    """
    if not text:
        return text
    protected: list = []
    text = _unwrap_noncode_fences(text, protected)
    lines = text.split("\n")
    lines = [_strip_emoji(ln) for ln in lines]
    patterns = DEFAULT_STATUS_TAG_PATTERNS + tuple(extra_status_patterns)
    out_lines = []
    for ln in lines:
        if _is_rule_line(ln):
            continue
        for pat in patterns:
            ln = pat.sub("", ln)
        out_lines.append(ln.rstrip())
    out_lines = _dedupe_consecutive_nonblank(out_lines)
    text = "\n".join(out_lines)
    text = re.sub(r"\n{4,}", "\n\n", text)
    for i, block in enumerate(protected):
        text = text.replace(f"\x00PROTECTED{i}\x00", block)
    return text


def clean_exchange_assistant_half(document: str, extra_status_patterns: tuple = ()) -> tuple:
    """For an episodic store document shaped `"[user]: U\\n[assistant]: A"`
    (see retrieval._exchange_doc) — cleans ONLY the assistant half, never
    the user half. Returns (new_document, changed). Falls back to treating
    the WHOLE document as the assistant half (still never touching a
    literal `[user]:` prefix if present) if the expected shape is not
    found, rather than silently doing nothing."""
    marker = "\n[assistant]: "
    idx = document.find(marker)
    if idx == -1 or not document.startswith("[user]: "):
        return document, False
    user_half = document[: idx + 1]  # includes the trailing \n
    assistant_half = document[idx + len(marker):]
    cleaned_assistant = clean_text(assistant_half, extra_status_patterns)
    if cleaned_assistant == assistant_half:
        return document, False
    return user_half + "[assistant]: " + cleaned_assistant, True


# ===========================================================================
# PART 2 — package resolution (compactor package + sibling import-history.py)
# ===========================================================================


def _resolve_compactor_pkg(explicit: str) -> tuple:
    """(pkg_dir, tried, already_importable) — identical resolution order
    to backfill-records.py/import-history.py's own copies of this
    function (intentionally duplicated, not shared — see those scripts'
    module docstrings for why this SPECIFIC helper is meant to be a
    per-script copy rather than a shared import)."""
    tried = []
    if explicit:
        p = Path(explicit)
        tried.append(f"{p} (--compactor-pkg)")
        if p.is_dir():
            return p, tried, False
    p2 = HERE.parent / "compactor"
    tried.append(f"{p2} (repo/clone layout beside this script — the supported way)")
    if p2.is_dir():
        return p2, tried, False
    p3 = OPT_COMPACTOR
    tried.append(f"{p3} (image layout)")
    if p3.is_dir():
        return p3, tried, False
    tried.append(f"an already-importable {_PROBE_MODULE!r} on sys.path")
    if importlib.util.find_spec(_PROBE_MODULE) is not None:
        return None, tried, True
    return None, tried, False


_REQUIRED_SUMMARIZER_SYMBOLS = (
    "load_state",
    "save_state",
    "_turn_fingerprints",
    "_covered_turn_fingerprints",
    "_covered_fps",
    "_FP_UNKNOWN",
)
_REQUIRED_MEMORY_SYMBOLS = ("atomic_write_json", "facts_path", "summary_path", "chromadb_path")


def _check_capability(mod, required: tuple, pkg_source: str, name: str):
    missing = [n for n in required if not hasattr(mod, n)]
    if not missing:
        return None
    return (
        f"ERROR: the compactor package at {pkg_source} is missing "
        f"{', '.join(missing)} from `{name}` — this script asks the REAL "
        f"module for the fingerprint/storage primitives rather than "
        f"keeping a second copy of them. Refusing rather than guess.\n\n"
        f"{_CLONE_HINT}"
    )


def _load_import_history_module():
    """Import scripts/import-history.py (beside this script) as a plain
    module, to reuse `_resolve_resume_offset`, `_walk_branch_from_history`,
    `reconstruct_transcript`, `_flatten_content`, `ChatNotFound`,
    `TranscriptUnavailable`, `_chunk_span_ending_at`,
    `_MIN_ANCHOR_FINGERPRINTS`, `_live_sidecars`, `_open_ro` — never
    re-derived here, same reasoning as merge-conversations.py's docstring
    gives for portability.merge_conversation. Returns (module, error) —
    error is an actionable string, never a raw ImportError, if the file is
    missing or its shape has moved underneath this one."""
    path = HERE / "import-history.py"
    if not path.is_file():
        return None, (
            f"ERROR: {path} not found beside this script. This script "
            f"reuses import-history.py's own resume-offset arithmetic "
            f"rather than duplicating it — a bare copy of just this file, "
            f"without its sibling, cannot run. See PACKAGE RESOLUTION in "
            f"the module docstring."
        )
    spec = importlib.util.spec_from_file_location("_zl_import_history", path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        return None, f"ERROR importing {path}: {type(e).__name__}: {e}"
    required = (
        "_resolve_resume_offset", "_walk_branch_from_history", "reconstruct_transcript",
        "_flatten_content", "ChatNotFound", "TranscriptUnavailable",
        "_chunk_span_ending_at", "_MIN_ANCHOR_FINGERPRINTS", "_live_sidecars", "_open_ro",
    )
    missing = [n for n in required if not hasattr(mod, n)]
    if missing:
        return None, (
            f"ERROR: {path} is missing {', '.join(missing)} — its shape "
            f"has moved since this script was written against it. "
            f"Refusing rather than guess at equivalent logic."
        )
    return mod, None


# ===========================================================================
# PART 3 — liveness refusals
# ===========================================================================


def _connection_refused(e: BaseException) -> bool:
    if isinstance(e, ConnectionRefusedError):
        return True
    if isinstance(e, OSError) and getattr(e, "errno", None) == errno.ECONNREFUSED:
        return True
    if isinstance(e, urllib.error.URLError):
        return _connection_refused(e.reason) if isinstance(e.reason, BaseException) else False
    return False


def _compactor_is_alive(url: str) -> bool:
    """False ONLY for an unambiguous connection refusal — see
    backfill-records.py's own `_compactor_is_alive` docstring for the full
    "fails open" reasoning this mirrors exactly."""
    try:
        with urllib.request.urlopen(url, timeout=HEALTH_PROBE_TIMEOUT_S) as r:
            r.read()
        return True
    except urllib.error.HTTPError:
        return True
    except Exception as e:
        return not _connection_refused(e)


def _port_is_open(port: int, host: str = "127.0.0.1") -> bool:
    try:
        with socket.create_connection((host, port), timeout=HEALTH_PROBE_TIMEOUT_S):
            return True
    except OSError:
        return False


def _supervisorctl_status(program: str) -> str:
    """"RUNNING", "STOPPED", "absent" (no such program / no supervisorctl
    at all), or "ambiguous" (any other outcome — timeout, permission,
    unparsable output). "absent" and "ambiguous" both fail OPEN to the
    port check, exactly like _compactor_is_alive's own reasoning: this
    function existing at all must never be the reason a live process goes
    unnoticed."""
    try:
        r = subprocess.run(
            ["supervisorctl", "status", program],
            capture_output=True, text=True, timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "absent"
    out = (r.stdout or "") + (r.stderr or "")
    if "no such process" in out.lower():
        return "absent"
    if re.search(r"\bRUNNING\b", out):
        return "RUNNING"
    if re.search(r"\bSTOPPED\b", out):
        return "STOPPED"
    return "ambiguous"


def _openwebui_is_alive(port: int) -> str:
    """"alive", "dead", or "ambiguous" — combining supervisorctl and a raw
    port probe, neither of which alone is proof either way (supervisorctl
    may not be managing openwebui by that name; a bound port could be
    something else). "ambiguous" refuses exactly like "alive" does — see
    _compactor_is_alive's fails-open reasoning, applied the same way
    here."""
    sup = _supervisorctl_status("openwebui")
    if sup == "RUNNING":
        return "alive"
    port_open = _port_is_open(port)
    if port_open:
        return "alive"
    if sup == "STOPPED":
        return "dead"
    if sup == "absent" and not port_open:
        return "dead"
    return "ambiguous"


def _openers(path: Path) -> list:
    """Every OTHER process on this host with `path` open, by scanning
    /proc/<pid>/fd — the same check the ops "repair-chat-tree.py" script
    uses before writing to webui.db, added here as an ADDITIONAL signal
    alongside the -wal/-journal check and the OpenWebUI/compactor
    liveness probes above, never a replacement for them (a process could
    have the file open without OpenWebUI itself being the one holding
    it — a stray `sqlite3` shell, a backup job mid-copy, anything).
    Best-effort: a /proc read that fails (permissions, a pid that exited
    mid-scan) is skipped, never raised."""
    real = os.path.realpath(str(path))
    me = str(os.getpid())
    found = []
    try:
        pids = os.listdir("/proc")
    except OSError:
        return []
    for pid in pids:
        if not pid.isdigit() or pid == me:
            continue
        try:
            fds = os.listdir(f"/proc/{pid}/fd")
        except OSError:
            continue
        for fd in fds:
            try:
                if os.path.realpath(f"/proc/{pid}/fd/{fd}").startswith(real):
                    found.append(pid)
                    break
            except OSError:
                pass
    return found


# ===========================================================================
# PART 4 — backups, restore bookkeeping
# ===========================================================================


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _check_free_space(path: Path, needed_bytes: int) -> str:
    free = shutil.disk_usage(path.parent).free
    # 20% headroom beyond the raw copy — a network volume's own bookkeeping
    # (MooseFS chunk allocation) is not byte-for-byte with local disk.
    required = int(needed_bytes * 1.2) + 10 * 1024 * 1024
    if free < required:
        return (
            f"ERROR: only {free / 1e6:.1f} MB free at {path.parent}, need "
            f"~{required / 1e6:.1f} MB to back up {path} ({needed_bytes / 1e6:.1f} MB). "
            f"Refusing to proceed without a successful backup."
        )
    return None


def _sqlite_online_backup(src_path: Path, dest_path: Path) -> None:
    """The sqlite3 backup API — safe to run against a database that is
    still being concurrently written, which is exactly what makes it the
    wrong tool for THIS backup: used only for `--restore`'s own
    verification copy (see `_verify_backup_openable` below), never for the
    restorable backup itself. `Connection.backup()` produces a fresh
    on-disk page layout, not a byte-for-byte copy of the source file — a
    real, well-documented property of the API, and one that would make
    `--restore` merely CONTENT-identical, not byte/md5-identical, which
    the operator brief explicitly requires. By the time this script backs
    up `webui.db` at all, `_live_sidecars` has already refused unless the
    file is quiescent (no `-wal`/`-journal`, OpenWebUI confirmed stopped),
    so the one property the backup API buys over a plain copy — safety
    against a database still being written — is already guaranteed by an
    earlier refusal, and a plain copy is strictly better for restore.
    """
    src = sqlite3.connect(f"file:{src_path.resolve().as_posix()}?mode=ro", uri=True)
    try:
        dest = sqlite3.connect(str(dest_path))
        try:
            src.backup(dest)
        finally:
            dest.close()
    finally:
        src.close()


def _verify_backup_openable(bak: Path) -> str:
    """Belt-and-braces sanity check for a fresh sqlite backup file: opens
    it read-only and runs `PRAGMA integrity_check`, WITHOUT touching the
    real backup bytes `--restore` will later copy back (a separate
    in-memory connection to the same file, opened `mode=ro`). None if
    clean; a detail string otherwise."""
    try:
        con = sqlite3.connect(f"file:{bak.resolve().as_posix()}?mode=ro", uri=True)
        try:
            row = con.execute("PRAGMA integrity_check").fetchone()
        finally:
            con.close()
    except Exception as e:
        return f"{type(e).__name__}: {e}"
    if row is None or row[0] != "ok":
        return str(row)
    return None


def _backup_file(path: Path, stamp: str) -> Path:
    """A plain, exact byte-for-byte copy — see `_sqlite_online_backup`'s
    own docstring for why that API, despite being named in the operator
    brief, is the wrong choice for the RESTORABLE copy specifically. For a
    `.db` file this also verifies the fresh copy opens cleanly under
    `PRAGMA integrity_check` before returning, so a backup that silently
    failed to copy correctly is caught immediately, not discovered later
    at `--restore` time."""
    bak = path.with_name(path.name + f".bak-{stamp}")
    if bak.exists():
        raise FileExistsError(str(bak))
    shutil.copy2(path, bak)
    if path.suffix == ".db" or path.name.endswith("webui.db"):
        err = _verify_backup_openable(bak)
        if err:
            bak.unlink(missing_ok=True)
            raise OSError(f"backup of {path} failed integrity_check: {err}")
    return bak


def _backup_dir(path: Path, stamp: str) -> Path:
    bak = path.with_name(path.name + f".bak-{stamp}")
    if bak.exists():
        raise FileExistsError(str(bak))
    shutil.copytree(path, bak)
    return bak


def _restore_file(original: Path, stamp: str) -> str:
    bak = original.with_name(original.name + f".bak-{stamp}")
    if not bak.is_file():
        return f"REFUSED: no backup found at {bak}"
    shutil.copy2(bak, original)
    return f"restored {original} from {bak}"


def _restore_dir(original: Path, stamp: str) -> str:
    bak = original.with_name(original.name + f".bak-{stamp}")
    if not bak.is_dir():
        return f"REFUSED: no backup found at {bak}"
    if original.exists():
        shutil.rmtree(original)
    shutil.copytree(bak, original)
    return f"restored {original} from {bak}"


# ===========================================================================
# PART 5 — webui.db target
# ===========================================================================


class _WebuiPlan:
    def __init__(self):
        self.chat_id = None
        self.branch = []          # raw history.messages nodes, oldest-first
        self.in_scope_ids = []    # message ids (assistant) eligible to clean
        self.changes = {}        # id -> (old_text, new_text)
        self.detail = ""
        self.refused = None


def _build_webui_plan(
    con: sqlite3.Connection, chat_id: str, ih_mod, summarizer_mod, state: dict,
    scope: str, last_n: int, l1_chunk_size: int, extra_status_patterns: tuple,
) -> _WebuiPlan:
    plan = _WebuiPlan()
    plan.chat_id = chat_id
    row = con.execute("SELECT chat FROM chat WHERE id = ?", (chat_id,)).fetchone()
    if row is None:
        plan.refused = f"REFUSED: no chat row for id {chat_id!r}"
        return plan
    try:
        data = json.loads(row[0]) if isinstance(row[0], (str, bytes)) else None
    except (json.JSONDecodeError, TypeError, UnicodeDecodeError) as e:
        plan.refused = f"REFUSED: chat.chat is not readable JSON for {chat_id!r}: {e}"
        return plan
    history = data.get("history") if isinstance(data, dict) else None
    chain = ih_mod._walk_branch_from_history(history) if isinstance(history, dict) else None
    if chain is None:
        plan.refused = (
            f"REFUSED: could not walk history.messages/currentId for "
            f"{chat_id!r} — this script determines SCOPE (which branch, "
            f"which turns) from that tree, the same way import-history.py "
            f"does, because there is no currentId in the chat_message "
            f"table to prove which branch is live. It still cross-checks "
            f"and writes the chat_message table's own rows for every "
            f"in-scope id (see the dual-copy agreement check below) — see "
            f"the module docstring's TARGETS section."
        )
        return plan
    plan.branch = chain

    # Flattened (role, text) turns, in the SAME shape reconstruct_transcript
    # produces, for fingerprinting and for _resolve_resume_offset.
    flat_turns = [
        {"role": n.get("role") or "unknown", "content": ih_mod._flatten_content(n.get("content"))}
        for n in chain
    ]

    assistant_positions = [i for i, t in enumerate(flat_turns) if t["role"] == "assistant"]

    if scope == "since-summarized":
        offset, detail = ih_mod._resolve_resume_offset(state, flat_turns, summarizer_mod, l1_chunk_size)
        plan.detail = detail
        if offset is None:
            plan.refused = f"REFUSED: cannot verify a resume offset for {chat_id!r}: {detail}"
            return plan
        last_summarized = int(state.get("last_summarized_turn") or 0)
        boundary_branch_turn = last_summarized - offset  # 1-indexed non-system turn count
        non_system_idx = [i for i, t in enumerate(flat_turns)]
        cutoff_i = boundary_branch_turn if boundary_branch_turn > 0 else 0
        eligible = [i for i in assistant_positions if i >= cutoff_i]
    else:
        eligible = assistant_positions[-last_n:] if last_n > 0 else []
        plan.detail = f"last {last_n} assistant repl{'y' if last_n == 1 else 'ies'} on the branch"

    # OpenWebUI keeps this same conversation TWICE: chat.chat's
    # history.messages tree (walked above) and the chat_message table's
    # own rows. Its own backend (models/chats.py's
    # get_messages_map_by_chat_id) PREFERS the table as the fast path
    # when building the branch it actually sends to the model, falling
    # back to this JSON only on a broken parent-link graph — so the
    # table, not this script's own read, is the closer approximation of
    # "what she will actually see resent". In the normal case the two
    # are byte-identical (verified on the real 09-23 backup: 4034/4034
    # matched). A documented past incident (RUNBOOK_CHAT_TREE.md, the
    # retired v1 repair script) shows they CAN diverge — double-encoding
    # one copy but not the other. Refusing outright on any disagreement,
    # rather than silently trusting the JSON copy this script reads from,
    # is cheap insurance against cleaning a message whose two stored
    # copies already disagree about what its own text is.
    mismatches = []
    for i in eligible:
        node = chain[i]
        msg_id = node.get("id")
        if not isinstance(node.get("content"), str):
            continue
        json_text = node.get("content")
        cm_row = con.execute(
            "SELECT content FROM chat_message WHERE id = ? AND chat_id = ?",
            (f"{chat_id}-{msg_id}", chat_id),
        ).fetchone()
        if cm_row is None:
            continue  # no table row for this id — nothing to disagree with
        try:
            table_text = json.loads(cm_row[0]) if isinstance(cm_row[0], (str, bytes)) else cm_row[0]
        except (json.JSONDecodeError, TypeError, UnicodeDecodeError) as e:
            mismatches.append(f"{msg_id}: chat_message.content is not readable JSON ({e})")
            continue
        if not isinstance(table_text, str):
            continue  # a non-text shape there is not this check's concern
        if table_text != json_text:
            mismatches.append(
                f"{msg_id}: history.messages has {len(json_text)} char(s), "
                f"chat_message table has {len(table_text)} char(s) — they disagree"
            )
    if mismatches:
        plan.refused = (
            f"REFUSED: {len(mismatches)} in-scope message(s) disagree between "
            f"chat.chat.history.messages and the chat_message table — refusing "
            f"to guess which copy is right rather than clean over a divergence "
            f"(see RUNBOOK_CHAT_TREE.md's retired v1 repair for how this "
            f"happens): " + "; ".join(mismatches[:5])
            + (f" (and {len(mismatches) - 5} more)" if len(mismatches) > 5 else "")
        )
        return plan

    for i in eligible:
        node = chain[i]
        msg_id = node.get("id")
        old_text = ih_mod._flatten_content(node.get("content"))
        if not isinstance(node.get("content"), str):
            # Multimodal/list content: clean only if it is plain text under
            # the hood; anything else is left alone rather than risk
            # corrupting a shape this script was not built to round-trip.
            continue
        new_text = clean_text(old_text, extra_status_patterns)
        if new_text != old_text:
            plan.changes[msg_id] = (old_text, new_text)
    return plan


def _apply_webui_plan(con: sqlite3.Connection, plan: _WebuiPlan) -> None:
    """One transaction: every changed id's three copies, or none. Caller
    commits/rolls back and runs PRAGMA integrity_check."""
    row = con.execute("SELECT chat FROM chat WHERE id = ?", (plan.chat_id,)).fetchone()
    data = json.loads(row[0])
    messages_map = data["history"]["messages"]
    for msg_id, (_, new_text) in plan.changes.items():
        if msg_id in messages_map:
            messages_map[msg_id]["content"] = new_text
        for m in data.get("messages") or []:
            if m.get("id") == msg_id:
                m["content"] = new_text
    con.execute("UPDATE chat SET chat = ? WHERE id = ?", (json.dumps(data), plan.chat_id))
    for msg_id, (_, new_text) in plan.changes.items():
        cm_id = f"{plan.chat_id}-{msg_id}"
        con.execute(
            "UPDATE chat_message SET content = ? WHERE id = ? AND chat_id = ? AND role = 'assistant'",
            (json.dumps(new_text), cm_id, plan.chat_id),
        )


def _recompute_anchor(summarizer_mod, cleaned_flat_turns: list) -> tuple:
    """(tail_fp, head_fp, window_turns) as `_observed_position` would set
    them for a request whose window is exactly `cleaned_flat_turns` — see
    the module docstring's HAZARD section. Only ever called with a branch
    that has ALREADY had its in-scope turns cleaned in memory."""
    turns = [t for t in cleaned_flat_turns if t.get("role") != "system"]
    tail_n = getattr(summarizer_mod, "_FINGERPRINT_TAIL_TURNS", 64)
    anchor_n = getattr(summarizer_mod, "_ANCHOR_TURNS", 4)
    if not turns:
        return [], "", 0
    fps = summarizer_mod._turn_fingerprints(turns[-tail_n:])
    head_fp = summarizer_mod._turn_fingerprints(turns[:1])[0]
    return fps[-anchor_n:], head_fp, len(turns)


# A fixed, neutral turn used ONLY to probe alignment before deciding
# whether to write a new anchor — never sent anywhere, never stored.
_PROBE_TURN = {"role": "user", "content": "\x00CLEAN_DECORATION_ALIGNMENT_PROBE\x00"}


def _simulate_position(summarizer_mod, state: dict, flat_turns: list) -> tuple:
    """(position, window_offset) `_observed_position` would report for a
    request whose window is `flat_turns + [_PROBE_TURN]`, against a COPY
    of `state` (never mutates the caller's real state)."""
    import copy as _copy
    probe_state = _copy.deepcopy(state)
    messages = list(flat_turns) + [_PROBE_TURN]
    pos = summarizer_mod._observed_position("_probe", probe_state, messages)
    non_system = [m for m in messages if m.get("role") != "system"]
    return pos, pos - len(non_system)


def _anchor_rewrite_is_verified(
    summarizer_mod, state: dict, original_flat_turns: list, cleaned_flat_turns: list,
    new_anchor: tuple,
) -> str:
    """None if rewriting the anchor to `new_anchor` provably preserves
    the position/offset the NEXT live request would have landed at
    without this script ever having run; otherwise a detail string
    naming the mismatch, and the anchor must NOT be written (see the
    module docstring's HAZARD section — proven empirically against the
    real 09-23 backup: her real stored anchor is ALREADY one turn out of
    sync with a fresh branch reconstruction, INDEPENDENT of any cleaning
    this script does — `window_turns` stored 3880 vs a fresh
    reconstruction's 3879 — so a freshly recomputed anchor can disagree
    with the CURRENT (already slightly stale) stored anchor's own
    tolerant partial-prefix matching even when the text is unchanged).
    This check catches exactly that disagreement before it is ever
    written, rather than assume recomputing is always safe.
    """
    before_pos, before_offset = _simulate_position(summarizer_mod, state, original_flat_turns)
    after_state = dict(state)
    after_state["tail_fp"], after_state["head_fp"], after_state["window_turns"] = new_anchor
    after_pos, after_offset = _simulate_position(summarizer_mod, after_state, cleaned_flat_turns)
    if before_pos == after_pos and before_offset == after_offset:
        return None
    return (
        f"a synthetic-probe simulation shows the NEW anchor would align the next "
        f"live request at position {after_pos} (offset {after_offset}) instead of "
        f"{before_pos} (offset {before_offset}) — the CURRENT stored anchor is "
        f"already not exactly in sync with a fresh branch reconstruction "
        f"(pre-existing drift, unrelated to this run's cleaning: see "
        f"import-history.py's own resume-offset verification for the same class "
        f"of drift in the covered region). Refusing to rewrite the anchor; "
        f"webui.db/facts/episodic cleaning still proceeded. Pass --force to "
        f"rewrite it anyway and accept a small realignment on the next live "
        f"request (NOT a permanent hole — a unique alignment was still found on "
        f"both sides, just at a different position)."
    )


def _align_finds_a_match(summarizer_mod, anchor: list, flat_turns: list) -> bool:
    """True if `anchor` (a tail_fp-shaped list) finds at least one real
    `_align_candidates` match against `flat_turns + [_PROBE_TURN]` —
    i.e. the NEXT live request would be answered from real fingerprint
    evidence, never the `_ASSUMED_NEW_TURNS` fallback guess that (see the
    module docstring's HAZARD section) is what actually turns into a
    permanent hole. A non-empty result may still be a DIFFERENT position
    than before (a realignment — see `_anchor_rewrite_is_verified`), but
    that is overlap/shift, never a hole: `_observed_position` never
    double-DROPS turns on a real match, only on an empty one. Item 2 of
    the hostile review: "refuse a hole even with --force" — this is the
    proof that gate reads."""
    if not anchor:
        return True  # nothing to align against yet — not this check's concern
    turns = [t for t in flat_turns if t.get("role") != "system"] + [_PROBE_TURN]
    tail_n = getattr(summarizer_mod, "_FINGERPRINT_TAIL_TURNS", 64)
    fps = summarizer_mod._turn_fingerprints(turns[-tail_n:])
    return bool(summarizer_mod._align_candidates(anchor, fps))


def _anchor_protected_turn_ids(summarizer_mod, branch: list) -> set:
    """Message ids of the turns `tail_fp` (the last `_ANCHOR_TURNS` non-
    system turns) and `head_fp` (the first non-system turn) are computed
    FROM, on the CURRENT, uncleaned branch. Item 2 of the hostile review:
    never clean one of these without a VERIFIED anchor rewrite to match —
    cleaning a turn tail_fp/head_fp themselves cover, while leaving the
    OLD anchor in place, is exactly the red case
    (test_alignment_proof_goes_red_without_the_anchor_rewrite):
    align_candidates comes back empty and `_observed_position` falls back
    to a flat per-call guess."""
    non_system = [n for n in branch if (n.get("role") or "unknown") != "system"]
    if not non_system:
        return set()
    anchor_n = getattr(summarizer_mod, "_ANCHOR_TURNS", 4)
    protected = {n.get("id") for n in non_system[-anchor_n:]}
    protected.add(non_system[0].get("id"))
    return protected


# ===========================================================================
# PART 6 — facts target
# ===========================================================================


def _clean_facts(facts: list, extra_status_patterns: tuple) -> tuple:
    """Returns (new_facts, changes, would-be-empty, would-be-duplicate).
    `new_facts` is a NEW list (input never mutated) where every fact whose
    cleaned text is non-empty and not a duplicate of another fact's
    (possibly also cleaned) text is updated; everything else is left byte
    -for-byte as it was, and reported instead."""
    cleaned_texts = [clean_text(f.get("text", ""), extra_status_patterns) for f in facts]
    seen: dict = {}
    for t in cleaned_texts:
        if t:
            seen[t] = seen.get(t, 0) + 1

    new_facts = []
    changes = []
    empties = []
    duplicates = []
    for f, new_text in zip(facts, cleaned_texts):
        old_text = f.get("text", "")
        if new_text == old_text:
            new_facts.append(f)
            continue
        if not new_text.strip():
            empties.append(old_text)
            new_facts.append(f)
            continue
        if seen.get(new_text, 0) > 1:
            duplicates.append((old_text, new_text))
            new_facts.append(f)
            continue
        nf = dict(f)
        nf["text"] = new_text
        new_facts.append(nf)
        changes.append((old_text, new_text))
    return new_facts, changes, empties, duplicates


# ===========================================================================
# PART 7 — episodic (chromadb) target
# ===========================================================================


def _load_retrieval_module(pkg_dir_str: str):
    """Best-effort import of the resolved package's retrieval.py, and a
    working chroma client — never guessed at if either is unavailable.
    Returns (retrieval_mod, collection, error)."""
    try:
        import retrieval as retrieval_mod  # noqa: E402  (pkg_dir already on sys.path)
    except Exception as e:
        return None, None, f"retrieval module unavailable: {type(e).__name__}: {e}"
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(retrieval_mod.CHROMA_PATH))
        # The REAL collection name and metadata retrieval.py itself uses
        # (verified by reading it, not guessed — a wrong name here does
        # NOT error, it silently get-or-CREATES a new, empty phantom
        # collection, which is exactly how this script's own episodic
        # target first shipped reporting 0 documents on her real store
        # while the forensic report counted 72 decorated ones: it was
        # reading/writing a collection the real compactor never touches).
        collection = client.get_or_create_collection(
            name=getattr(retrieval_mod, "COLLECTION_NAME", "conversation_turns"),
            metadata={"hnsw:space": "cosine"},
        )
    except Exception as e:
        return retrieval_mod, None, f"chromadb unavailable/incompatible: {type(e).__name__}: {e}"
    return retrieval_mod, collection, None


def _clean_episodic(collection, retrieval_mod, conv_id: str, extra_status_patterns: tuple) -> tuple:
    """Returns (changes, error). `changes` is a list of
    {old_id, new_id, old_doc, new_doc}. Embeds every changed document with
    `retrieval_mod._embed` — the compactor's own call, not a re-derivation.
    """
    try:
        got = collection.get(where={"conv_id": conv_id}, include=["documents", "metadatas"])
    except Exception as e:
        return [], f"could not read the episodic store: {type(e).__name__}: {e}"
    ids = got.get("ids") or []
    docs = got.get("documents") or []
    metas = got.get("metadatas") or []
    changes = []
    for old_id, doc, meta in zip(ids, docs, metas):
        new_doc, changed = clean_exchange_assistant_half(doc, extra_status_patterns)
        if not changed:
            continue
        new_id = retrieval_mod._doc_id(conv_id, new_doc)
        changes.append({"old_id": old_id, "new_id": new_id, "old_doc": doc, "new_doc": new_doc, "meta": meta})
    return changes, None


def _apply_episodic_changes(collection, retrieval_mod, changes: list) -> str:
    for ch in changes:
        vecs = retrieval_mod._embed([ch["new_doc"]])
        if not vecs:
            return f"embedding failed for a changed document (old id {ch['old_id']})"
        collection.upsert(
            ids=[ch["new_id"]], embeddings=vecs, documents=[ch["new_doc"]], metadatas=[ch["meta"]],
        )
        if ch["new_id"] != ch["old_id"]:
            try:
                collection.delete(ids=[ch["old_id"]])
            except Exception:
                pass
    return None


# ===========================================================================
# PART 8 — CLI
# ===========================================================================


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Deterministically strip emoji/rule-line/status-board "
        "decoration from her stored chat history, active facts, and "
        "episodic memory — no LLM, no /forget. Dry run by default.",
    )
    ap.add_argument("--webui-db", default=DEFAULT_WEBUI_DB, metavar="PATH")
    ap.add_argument("--store", default=DEFAULT_STORE, metavar="PATH")
    ap.add_argument("--conv", required=False, metavar="CONV_ID",
                     help="the conversation id to clean (required unless --restore)")
    ap.add_argument("--compactor-pkg", default=None, metavar="PATH")
    ap.add_argument("--only", action="append", choices=list(ALL_TARGETS), metavar="TARGET",
                     help="restrict to this target; repeatable. Default: all three.")
    scope = ap.add_mutually_exclusive_group()
    scope.add_argument("--last", type=int, default=DEFAULT_LAST_N, metavar="N",
                        help=f"clean the last N assistant replies on the branch (default {DEFAULT_LAST_N})")
    scope.add_argument("--since-summarized", action="store_true",
                        help="clean every assistant reply strictly after the store's own "
                             "last_summarized_turn, mapped to a branch position with "
                             "import-history.py's verified _resolve_resume_offset")
    ap.add_argument("--l1-chunk-size", type=int, default=20, metavar="N",
                     help="only used with --since-summarized, to seed the resume-offset "
                          "search the same way import-history.py does (default 20)")
    ap.add_argument("--status-tag-pattern", action="append", default=[], metavar="REGEX",
                     help="extra trailing-status-tag regex to strip, in addition to the "
                          "built-in conservative defaults; repeatable")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--force", action="store_true",
                     help="override the OpenWebUI/compactor liveness refusals and an "
                          "existing-backup refusal. Never overrides a failed integrity_check "
                          "or a refused resume-offset.")
    ap.add_argument("--health-url", default=DEFAULT_HEALTH_URL, metavar="URL")
    ap.add_argument("--openwebui-port", type=int, default=DEFAULT_OPENWEBUI_PORT, metavar="PORT")
    ap.add_argument("--restore", default=None, metavar="STAMP",
                     help="restore every target from its backup taken under this stamp "
                          "(printed by the --apply run that made it), then exit")
    ap.add_argument("--debug-skip-anchor-rewrite", action="store_true",
                     help=argparse.SUPPRESS)
    # TESTING ONLY — never pass this operationally. It exists so
    # compactor/test_real_image_clean_decoration.py can demonstrate its own
    # alignment proof going RED when the anchor-rewrite step is disabled
    # (proving that step is load-bearing, not vacuous) without duplicating
    # the whole apply path a second time just to break it on purpose. With
    # this flag, webui.db is still cleaned normally but tail_fp/head_fp/
    # window_turns are left stale — reproducing exactly the hazard this
    # script's own module docstring warns about.
    return ap


def _fatal(args, message: str) -> int:
    if args.json:
        print(json.dumps({"error": message}, indent=2))
    else:
        print(message)
    return 1


def _print_report(args, report: dict) -> None:
    if args.json:
        print(json.dumps(report, indent=2, default=str))
        return
    for target, r in report["targets"].items():
        print(f"\n== {target} ==")
        if r.get("refused"):
            print(f"  REFUSED: {r['refused']}")
            continue
        if r.get("skipped"):
            print(f"  SKIPPED: {r['skipped']}")
            continue
        print(f"  {r.get('detail', '')}")
        print(f"  {r['count']} change(s){' (dry run)' if not args.apply else ''}")
        for old, new in r.get("preview", [])[:3]:
            print(f"  --- before ---\n{old[:300]}")
            print(f"  +++ after +++\n{new[:300]}")
        if args.apply and r.get("backup"):
            print(f"  backup: {r['backup']}")
    if report.get("anchor"):
        print(f"\n== anchor ==\n  {report['anchor']}")


def main(argv=None) -> int:
    args = _build_argparser().parse_args(argv)

    if args.restore:
        return _do_restore(args)

    if not args.conv:
        return _fatal(args, "ERROR: --conv is required (unless --restore).")

    targets = args.only or list(ALL_TARGETS)
    scope = "since-summarized" if args.since_summarized else "last-n"
    extra_status_patterns = tuple(re.compile(p) for p in args.status_tag_pattern)

    store_root = Path(args.store)
    if not store_root.is_dir():
        return _fatal(args, f"ERROR: --store {store_root} does not exist or is not a directory.")
    db_path = Path(args.webui_db)
    if "webui" in targets and not db_path.is_file():
        return _fatal(args, f"ERROR: --webui-db {db_path} does not exist or is not a file.")

    pkg_dir, tried, already_importable = _resolve_compactor_pkg(args.compactor_pkg)
    if pkg_dir is None and not already_importable:
        lines = ["ERROR: no compactor package found. Tried, in order:"]
        lines += [f"  - {t}" for t in tried]
        lines += ["", _CLONE_HINT]
        return _fatal(args, "\n".join(lines))
    if pkg_dir is not None:
        sys.path.insert(0, str(pkg_dir))
        pkg_source = str(pkg_dir)
    else:
        pkg_source = "an already-importable module (no package directory)"

    os.environ["COMPACTOR_STORAGE_ROOT"] = str(store_root.resolve())
    try:
        import summarizer as summarizer_mod  # noqa: E402
        import memory as memory_mod  # noqa: E402
    except Exception as e:
        return _fatal(args, f"ERROR importing compactor package from {pkg_source}: {type(e).__name__}: {e}")

    for mod, req, name in (
        (summarizer_mod, _REQUIRED_SUMMARIZER_SYMBOLS, "summarizer"),
        (memory_mod, _REQUIRED_MEMORY_SYMBOLS, "memory"),
    ):
        err = _check_capability(mod, req, pkg_source, name)
        if err:
            return _fatal(args, err)

    ih_mod, ih_err = _load_import_history_module()
    if ih_err:
        return _fatal(args, ih_err)

    if args.apply:
        if not args.force:
            ow = _openwebui_is_alive(args.openwebui_port)
            if ow != "dead":
                return _fatal(
                    args,
                    f"REFUSED: OpenWebUI looks {ow} (supervisorctl and/or port "
                    f"{args.openwebui_port}). --apply rewrites webui.db and the "
                    f"compactor's own store while it is a live file for a running "
                    f"OpenWebUI/compactor; pass --force only if you are certain both "
                    f"are stopped by another means.",
                )
            if _compactor_is_alive(args.health_url):
                return _fatal(
                    args,
                    f"REFUSED: something answered {args.health_url} — the compactor "
                    f"looks alive (or the check was ambiguous, which is treated the "
                    f"same way). Stop it (supervisorctl stop compactor) or pass --force.",
                )
            if "webui" in targets and db_path.is_file():
                openers_found = _openers(db_path)
                if openers_found:
                    return _fatal(
                        args,
                        f"REFUSED: {db_path} is open by process(es) {openers_found} "
                        f"(a /proc fd scan, the same additional check "
                        f"repair-chat-tree.py uses — this is on TOP of the "
                        f"OpenWebUI/compactor liveness probes above, not instead of "
                        f"them: something else, a stray shell or a backup job, can "
                        f"hold this file open without either of those answering). "
                        f"Close it, or pass --force.",
                    )

    conv_id = args.conv
    state = summarizer_mod.load_state(conv_id)
    report = {"targets": {}, "conv_id": conv_id}
    stamp = _utc_stamp()
    any_change = False
    any_refusal = False
    any_target_requested = False

    webui_plan = None
    cleaned_flat_turns = None
    original_flat_turns = None
    if "webui" in targets:
        any_target_requested = True
        sidecars = ih_mod._live_sidecars(db_path)
        if sidecars and not args.force:
            report["targets"]["webui"] = {
                "refused": f"{', '.join(str(p) for p in sidecars)} sits beside {db_path} "
                           f"— looks like a live or crashed database. Pass --force to override."
            }
            any_refusal = True
        else:
            con = ih_mod._open_ro(db_path)
            try:
                webui_plan = _build_webui_plan(
                    con, conv_id, ih_mod, summarizer_mod, state, scope, args.last,
                    args.l1_chunk_size, extra_status_patterns,
                )
            finally:
                con.close()
            if webui_plan.refused:
                report["targets"]["webui"] = {"refused": webui_plan.refused}
                any_refusal = True
            else:
                preview = list(webui_plan.changes.values())
                report["targets"]["webui"] = {
                    "detail": webui_plan.detail, "count": len(webui_plan.changes),
                    "preview": preview,
                }
                if webui_plan.changes:
                    any_change = True
                # Build the cleaned-in-memory flat turns for anchor recompute,
                # regardless of --apply, so the dry run can also report what
                # the anchor WOULD become.
                flat_turns = [
                    {"role": n.get("role") or "unknown",
                     "content": webui_plan.changes.get(n.get("id"), (None, ih_mod._flatten_content(n.get("content"))))[1]}
                    for n in webui_plan.branch
                ]
                cleaned_flat_turns = flat_turns
                original_flat_turns = [
                    {"role": n.get("role") or "unknown", "content": ih_mod._flatten_content(n.get("content"))}
                    for n in webui_plan.branch
                ]

    facts_plan = None
    if "facts" in targets:
        any_target_requested = True
        fpath = memory_mod.facts_path(conv_id)
        if not fpath.is_file():
            report["targets"]["facts"] = {"detail": "no facts file for this conversation", "count": 0, "preview": []}
        else:
            raw = json.loads(fpath.read_text(encoding="utf-8"))
            facts = raw.get("facts", [])
            new_facts, changes, empties, duplicates = _clean_facts(facts, extra_status_patterns)
            facts_plan = (raw, new_facts)
            report["targets"]["facts"] = {
                "detail": f"{len(facts)} active fact(s) examined", "count": len(changes),
                "preview": changes, "would_be_empty": empties, "would_be_duplicate": duplicates,
            }
            if changes:
                any_change = True

    episodic_changes = None
    retrieval_mod = collection = None
    if "episodic" in targets:
        any_target_requested = True
        retrieval_mod, collection, err = _load_retrieval_module(pkg_source if pkg_dir else "sys.path")
        if err:
            report["targets"]["episodic"] = {"skipped": err}
        else:
            episodic_changes, err2 = _clean_episodic(collection, retrieval_mod, conv_id, extra_status_patterns)
            if err2:
                report["targets"]["episodic"] = {"skipped": err2}
            else:
                report["targets"]["episodic"] = {
                    "detail": "documents for this conversation",
                    "count": len(episodic_changes),
                    "preview": [(c["old_doc"], c["new_doc"]) for c in episodic_changes],
                }
                if episodic_changes:
                    any_change = True

    anchor_note = None
    skipped_protected = {}
    hole_risk = False
    if webui_plan is not None and not webui_plan.refused and cleaned_flat_turns is not None:
        old = (state.get("tail_fp"), state.get("head_fp"), state.get("window_turns"))
        old_tail_fp = [x for x in (state.get("tail_fp") or []) if isinstance(x, str)]

        if not args.force:
            # Item 2 of the hostile review: NEVER leave cleaned text behind
            # an un-rewritten anchor. The right question is NOT "does a
            # freshly recomputed anchor exactly equal the stored one" —
            # verified (real-image suite) that this can fail even with
            # ZERO turns cleaned, whenever the store already has the kind
            # of pre-existing drift item 1 found, making that equality
            # test unsatisfiable by exclusion no matter how much is
            # excluded. The question that IS well-posed: "does the anchor
            # ALREADY on disk still find a real match once these turns
            # are cleaned, leaving it un-rewritten?" If yes, cleaning is
            # safe with NO anchor rewrite at all. If no, exclude
            # candidate turns (narrowest window first, widening once)
            # until it is yes again — guaranteed to succeed once enough
            # are excluded, since excluding everything reproduces
            # whatever the CURRENT live production state already is.
            safe = _align_finds_a_match(summarizer_mod, old_tail_fp, cleaned_flat_turns)
            for widen in (False, True):
                if safe:
                    break
                if widen:
                    tail_n = getattr(summarizer_mod, "_FINGERPRINT_TAIL_TURNS", 64)
                    non_sys = [n for n in webui_plan.branch if (n.get("role") or "unknown") != "system"]
                    protected_ids = {n.get("id") for n in non_sys[-tail_n:]}
                else:
                    protected_ids = _anchor_protected_turn_ids(summarizer_mod, webui_plan.branch)
                newly_skipped = {
                    mid: old_new for mid, old_new in webui_plan.changes.items() if mid in protected_ids
                }
                if newly_skipped:
                    for mid in newly_skipped:
                        del webui_plan.changes[mid]
                    skipped_protected.update(newly_skipped)
                    cleaned_flat_turns = [
                        {"role": n.get("role") or "unknown",
                         "content": webui_plan.changes.get(n.get("id"), (None, ih_mod._flatten_content(n.get("content"))))[1]}
                        for n in webui_plan.branch
                    ]
                safe = _align_finds_a_match(summarizer_mod, old_tail_fp, cleaned_flat_turns)
            if not safe:
                # Should not happen (excluding every candidate turn
                # reproduces the branch as it stands in production right
                # now, which is safe by construction) — but never guess:
                # refuse the whole webui text write rather than risk it.
                skipped_protected.update(webui_plan.changes)
                webui_plan.changes.clear()
                report["targets"]["webui"]["refused"] = (
                    "could not find any subset of the requested turns that "
                    "leaves the CURRENT anchor able to align at all, even "
                    "after excluding every candidate turn — refusing rather "
                    "than guess."
                )
                any_refusal = True
            if skipped_protected:
                if webui_plan.changes:
                    any_change = True  # never force this back to False: facts/episodic may have their own changes
                report["targets"]["webui"]["count"] = len(webui_plan.changes)
                report["targets"]["webui"]["preview"] = list(webui_plan.changes.values())
                report["targets"]["webui"]["skipped_protected_by_anchor"] = [
                    {"id": mid, "old": old_new[0], "new": old_new[1]}
                    for mid, old_new in skipped_protected.items()
                ]
                # Skipping is a SAFETY CHOICE, not an error — informational
                # (use --force or a wider/narrower scope), never a hard
                # refusal on its own. any_refusal is untouched here.

            # The remaining, now-safe change set never needs the anchor
            # rewritten at all: the old anchor still matches after it.
            new = old
            anchor_changed = False
            anchor_verify_error = None
        else:
            # --force: clean everything requested, then decide the anchor.
            new = _recompute_anchor(summarizer_mod, cleaned_flat_turns)
            anchor_changed = old != new
            anchor_verify_error = None
            if anchor_changed:
                anchor_verify_error = _anchor_rewrite_is_verified(
                    summarizer_mod, state, original_flat_turns, cleaned_flat_turns, new,
                )
                # --force accepts a REALIGNMENT (a different but genuinely
                # matched position) — never a HOLE (the _ASSUMED_NEW_TURNS
                # fallback that never repays the turns it drops). Proven
                # by checking _align_candidates itself returns a real
                # match for the NEW anchor, not by re-deriving position
                # arithmetic. This is item 2's "refuse a hole even with
                # --force", and it overrides --force unconditionally.
                hole_risk = not _align_finds_a_match(summarizer_mod, list(new[0]), cleaned_flat_turns)
                if hole_risk:
                    prior = f"{anchor_verify_error} " if anchor_verify_error else ""
                    anchor_verify_error = (
                        f"{prior}REFUSING even with --force: the new anchor finds "
                        f"NO real fingerprint match at all against a realistic "
                        f"next window (align_candidates empty) — that is the "
                        f"fallback-guess shape that causes a permanent hole, not "
                        f"a mere realignment. webui.db/facts/episodic text "
                        f"cleaning still proceeded; the anchor was left untouched."
                    )
                    any_refusal = True

        anchor_note = (
            f"anchor would {'change' if anchor_changed else 'stay the same'} "
            f"(window_turns {old[2]} -> {new[2]})"
        )
        if anchor_verify_error:
            anchor_note += f" — UNVERIFIED: {anchor_verify_error}"
        report["anchor"] = anchor_note
        report["_anchor_new"] = new
        report["_anchor_changed"] = anchor_changed
        report["_anchor_verify_error"] = anchor_verify_error
        report["_anchor_hole_risk"] = hole_risk

    if not args.apply:
        _print_report(args, report)
        if any_refusal:
            return 1
        return 3 if any_change else 0

    # --apply
    backups = {}
    written_any = False
    failed_any = False

    webui_text_write_ok = not webui_plan.changes if webui_plan is not None else False
    if webui_plan is not None and not webui_plan.refused and webui_plan.changes:
        try:
            size = db_path.stat().st_size
            fs_err = _check_free_space(db_path, size)
            if fs_err:
                report["targets"]["webui"]["refused"] = fs_err
                any_refusal = True
            else:
                bak = _backup_file(db_path, stamp)
                backups["webui"] = str(bak)
                con = sqlite3.connect(str(db_path))
                try:
                    con.execute("BEGIN")
                    _apply_webui_plan(con, webui_plan)
                    con.commit()
                    row = con.execute("PRAGMA integrity_check").fetchone()
                    if row is None or row[0] != "ok":
                        con.close()
                        shutil.copy2(bak, db_path)
                        report["targets"]["webui"]["refused"] = (
                            f"PRAGMA integrity_check failed after write ({row}); "
                            f"restored {db_path} from {bak} immediately."
                        )
                        any_refusal = True
                    else:
                        con.close()
                        report["targets"]["webui"]["backup"] = str(bak)
                        written_any = True
                        webui_text_write_ok = True
                except Exception as e:
                    con.rollback()
                    con.close()
                    report["targets"]["webui"]["refused"] = f"write failed, rolled back: {type(e).__name__}: {e}"
                    any_refusal = True
        except FileExistsError as e:
            report["targets"]["webui"]["refused"] = f"backup already exists: {e}"
            any_refusal = True

    # Anchor: independent of whether there was any TEXT left to clean —
    # e.g. a prior run may have already cleaned the text but had its
    # anchor rewrite refused (pre-existing drift, see
    # _anchor_rewrite_is_verified); a follow-up --force run must be able
    # to fix JUST the anchor without anything left to re-clean. Gated on
    # the webui.db write having succeeded (or there being nothing to
    # write in the first place) so a failed/refused text write never
    # leaves the anchor pointing at text that was never actually written.
    if webui_plan is not None and not webui_plan.refused and webui_text_write_ok:
        # A hole risk refuses the anchor write EVEN WITH --force (item 2:
        # "refuse a hole even with --force") — --force only ever overrides
        # a mere realignment, never a fallback-guess hole.
        anchor_blocked = bool(report.get("_anchor_verify_error")) and (
            report.get("_anchor_hole_risk") or not args.force
        )
        if anchor_blocked:
            report["targets"]["webui"]["anchor_refused"] = report["_anchor_verify_error"]
            any_refusal = True
        if (
            report.get("_anchor_changed")
            and not args.debug_skip_anchor_rewrite
            and not anchor_blocked
        ):
            spath = memory_mod.summary_path(conv_id)
            if spath.is_file() and "summaries" not in backups:
                sbak = _backup_file(spath, stamp)
                backups["summaries"] = str(sbak)
            new_tail_fp, new_head_fp, new_window_turns = report["_anchor_new"]
            state["tail_fp"] = new_tail_fp
            state["head_fp"] = new_head_fp
            state["window_turns"] = new_window_turns
            summarizer_mod.save_state(conv_id, state)
            written_any = True

    if facts_plan is not None and report["targets"].get("facts", {}).get("count"):
        raw, new_facts = facts_plan
        fpath = memory_mod.facts_path(conv_id)
        try:
            bak = _backup_file(fpath, stamp)
            backups["facts"] = str(bak)
            raw = dict(raw)
            raw["facts"] = new_facts
            memory_mod.atomic_write_json(fpath, raw)
            report["targets"]["facts"]["backup"] = str(bak)
            written_any = True
        except Exception as e:
            report["targets"]["facts"]["refused"] = f"write failed: {type(e).__name__}: {e}"
            any_refusal = True

    if episodic_changes:
        chroma_dir = memory_mod.chromadb_path()
        try:
            bak = _backup_dir(chroma_dir, stamp)
            backups["chromadb"] = str(bak)
            err = _apply_episodic_changes(collection, retrieval_mod, episodic_changes)
            if err:
                shutil.rmtree(chroma_dir)
                shutil.copytree(bak, chroma_dir)
                report["targets"]["episodic"]["refused"] = f"{err}; restored from {bak}"
                any_refusal = True
            else:
                report["targets"]["episodic"]["backup"] = str(bak)
                written_any = True
        except FileExistsError as e:
            report["targets"]["episodic"]["refused"] = f"backup already exists: {e}"
            any_refusal = True

    report["stamp"] = stamp
    report["backups"] = backups
    _print_report(args, report)
    if not args.json:
        print(f"\nrestore stamp: {stamp}")

    if any_refusal and written_any:
        return 4
    if any_refusal and not written_any:
        return 1
    if not any_target_requested:
        return 1
    if not any_change:
        return 0  # nothing needed changing — desired end state already true
    if not written_any:
        return 1  # there was work but --apply accomplished none of it
    if skipped_protected:
        # Real progress happened, but at least one turn was deliberately
        # left decorated for a documented safety reason (item 2: never
        # clean a turn tail_fp/head_fp cover without a verified anchor
        # rewrite to match) — the same "some done, some not" shape as any
        # other partial target, so it gets the same code.
        return 4
    return 0


def _do_restore(args) -> int:
    stamp = args.restore
    results = []
    if "webui" in (args.only or ALL_TARGETS):
        results.append(("webui", _restore_file(Path(args.webui_db), stamp)))
    if "facts" in (args.only or ALL_TARGETS) and args.conv:
        os.environ["COMPACTOR_STORAGE_ROOT"] = str(Path(args.store).resolve())
        pkg_dir, _, already = _resolve_compactor_pkg(args.compactor_pkg)
        if pkg_dir is not None:
            sys.path.insert(0, str(pkg_dir))
        try:
            import memory as memory_mod  # noqa: E402
            results.append(("facts", _restore_file(memory_mod.facts_path(args.conv), stamp)))
            results.append(("summaries", _restore_file(memory_mod.summary_path(args.conv), stamp)))
            if "episodic" in (args.only or ALL_TARGETS):
                results.append(("chromadb", _restore_dir(memory_mod.chromadb_path(), stamp)))
        except Exception as e:
            results.append(("facts/summaries/chromadb", f"REFUSED: could not import memory module: {e}"))
    ok = True
    for name, msg in results:
        print(f"{name}: {msg}")
        if msg.startswith("REFUSED"):
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
