#!/usr/bin/env python3
"""Catch a conversation's hierarchical summary up from a `webui.db` export,
for a backlog the admin endpoint cannot reach.

SUPPORTED INVOCATION — from a clone, not a copy (see "PACKAGE RESOLUTION"
below for why a copy to /data/scripts/ fails on a pre-v3.1.9.4 pod):

    git clone --depth 1 --branch <tag-or-branch> \\
        https://github.com/MrBanana8768/zions-light-ai.git /opt/zl-repo
    /opt/compactor-venv/bin/python /opt/zl-repo/scripts/import-history.py \\
        --webui-db /data/openwebui/webui.db --chat-id <id> \\
        --store /data/openwebui/compactor
    /opt/compactor-venv/bin/python /opt/zl-repo/scripts/import-history.py \\
        --webui-db /data/openwebui/webui.db --chat-id <id> \\
        --store /data/openwebui/compactor --json
    /opt/compactor-venv/bin/python /opt/zl-repo/scripts/import-history.py \\
        --webui-db /data/openwebui/webui.db --chat-id <id> \\
        --store /data/openwebui/compactor --apply

`--compactor-pkg PATH` overrides package auto-detection entirely, if you
need to point this at a package that isn't beside the script and isn't at
/opt/compactor (see "PACKAGE RESOLUTION").

PACKAGE RESOLUTION. Copying just this one file to /data/scripts/ and
running it from there used to fail with "no compactor package beside this
script (/data/compactor)" — HERE.parent/"compactor" resolves relative to
wherever THIS FILE sits, and /data/scripts/../compactor does not exist.
Resolved now, in order: `--compactor-pkg PATH` (explicit, highest
precedence) > `HERE.parent/"compactor"` (the repo/clone layout — the
SUPPORTED way, see above) > `/opt/compactor` (the image's own layout) > an
already-importable `summarizer` on sys.path. If every one of those fails,
the error lists every path tried and prints the clone command above. See
_resolve_compactor_pkg's own docstring (and scripts/backfill-records.py's
twin of it) for the full reasoning, including a layout trap that can make
a substitute package silently invisible.

WHY THIS DOES NOT ALSO WORK BY JUST COPYING NEWER FILES IN. This script
asks the real `summarizer`/`memory` modules for their own rules rather
than keeping a second copy of them (see _check_summarizer_capability); a
package too old or too different to have them refuses with an actionable
message rather than an `AttributeError` partway through a run. The only
self-consistent fix for a genuinely incompatible package is the WHOLE
target release together, which is exactly what the clone above supplies.

WHY THIS EXISTS, AND WHY `POST /admin/conversations/{conv_id}/compact`
CANNOT DO IT. That endpoint (`compactor/main.py`, `admin_compact`) drains
the same L1/L2/L3 rollup loop this script uses, but it rebuilds the
transcript to summarize from the EPISODIC store (chromadb, via
`retrieval.export_indexed_exchanges`) — the only ordered record the
compactor itself owns. That store is a rolling window, not an archive: on
the conversation this script exists for it holds 70 exchanges against a
history of roughly 3,850 messages. Handed a transcript that short,
`admin_compact` either refuses outright (its own guard: a reconstruction
shorter than the conversation's recorded position would summarize text
that is not the text the chunk labels claim) or, if it ran anyway, would
"cover" thousands of turns as gap placeholders — recording that they are
unknown, permanently, rather than what was actually said. Either way it
cannot close a gap this size. The full transcript exists in exactly one
place: OpenWebUI's own `webui.db`. This script reads it from an EXPORT
(never the live database — see the refusal below) and drives the real
`compactor/summarizer.maybe_rollup` drain over it, the same drain
`admin_compact` runs, just fed from a different source.

WHAT THIS DOES NOT DO.
  - It does not touch the episodic/chromadb index. The backlog this script
    closes is a summarization backlog, not a retrieval backlog; rebuilding
    chromadb from an old export is a different, larger job and out of
    scope here.
  - It never reads or writes anything under `facts/`, `chromadb/` or
    `personas/`. Its entire blast radius is one file:
    `summaries/<conv_id>.json` (and its own dated backup beside it).
  - It never opens `webui.db` for writing, and refuses outright to run
    against what looks like a live or crashed database (a `-journal` or
    `-wal` file beside it) unless `--force`.
  - Dry run (the default) makes NO vLLM calls at all — not even to probe
    the model. `--apply` is required to make any.

HOW THE TRANSCRIPT IS BUILT. OpenWebUI 0.11 stores a conversation as one
JSON blob in `chat.chat`, shaped `{"history": {"messages": {id: {...
"parentId": ..., "role": ..., "content": ...}}, "currentId": id}}` — a
tree of every edit and regeneration, where `currentId` names the leaf the
user is actually looking at. This script walks the parent chain back from
`currentId` and reverses it, which is the LINEAR branch she sees — not
insertion order, which would include abandoned edits and every
regenerated reply. If that JSON is missing or unreadable, it falls back to
the `chat_message` table (`chat_id`, `parent_id`, `role`, `content`,
`created_at`), reconstructing the same way from the most recently created
leaf. Either way the dry-run report says which source it used. A
multimodal `content` array becomes its text parts joined, with each image
part replaced by a short `[image]` placeholder — this script's own
flattening, upstream of anything `compactor/summarizer.py` does with the
result. Non-alternating turns (two user turns in a row, a missing
assistant reply) are reported as anomalies and handled — the reconstructed
array is still built and handed to the drain — never crashed on.

EXIT CODES. The architect's ruling for v3.1.9.6, normalized across all
three operator scripts (scripts/backfill-records.py, scripts/setup-
sshd.py, and this one) -- closing H2, where there used to be no code
meaning "ran but accomplished nothing", so a dead vLLM and an exhausted
--apply both silently read as ordinary success:

    Code  Meaning
    0     Desired end state reached (a no-op --apply is 0)
    1     Refusal or error, OR --apply accomplished nothing (no rollup
          UNIT completed and was saved -- see N5, round-3 fix pass A:
          this is no longer just "the watermark did not move", since an
          L2 fold or the L3 refresh is real progress that does not move
          it either)
    2     argparse usage error (unrelated to the rest of this table --
          the standard library's own convention)
    3     A DRY RUN found pending work -- informational, not a failure:
          re-run with --apply once ready
    4     --apply made real progress but work remains (most commonly
          the --max-calls budget ran out, or SIGINT/SIGTERM/SIGHUP asked
          it to stop after the unit in flight finished -- see N1, round-3
          fix pass A): re-run to continue

Concretely, for THIS script: 1 covers a bad `--webui-db` or `--store`
(including a `--store` that exists but has no `summaries/`, or whose
conv_id has no summaries file yet under it), an unknown `--chat-id`, a
`-journal`/`-wal` beside the database without `--force`, no usable
compactor package (or one missing what this script needs from it -- see
PACKAGE RESOLUTION above), a refused `--apply` precondition (a DETECTED
live compactor -- never overridable by --force, N7 -- an AMBIGUOUS
compactor health check without `--force`, this run's own cross-process
import lock already held by another import-history.py --apply against
the same conv (N7), an existing backup path, no `--model`/`MODEL_REPO`),
a reconstructed transcript whose content does not match what the store
already has confirmed (see _prefix_matches_store -- NOT simply "fewer
turns than recorded_position", which a normal edit or abandoned branch
can cause on its own), an unverifiable/ambiguous resume offset OR an
offset/window combination `_resolve_apply_feasibility` cannot satisfy
against the store's own covered-turn record (see N4 and
_resolve_apply_feasibility's own docstring -- the dry run reports this
identically, per N6), a post-apply re-verify that found the store
changed underneath this run (N7), AND an `--apply` that ran but never
completed a single rollup unit (vLLM unreachable from the first call, a
signal interrupting before any unit finished, or an offset-consistency
check failed after writing -- see _verify_offset_after_apply's own
docstring). 3 is the dry-run-found-work case. 4 is `--apply` making
real, verified progress but stopping with more due.

POD PROCEDURE
    1. Get this script (and the matching `compactor` package) onto the
       pod — there is no `ssh`/`scp`/`rsync` in the image, so clone the
       repo (see SUPPORTED INVOCATION above):
           git clone --depth 1 --branch <tag-or-branch> \\
               https://github.com/MrBanana8768/zions-light-ai.git /opt/zl-repo
    2. Stop the compactor so nothing else is writing the same state file:
           supervisorctl stop compactor
    3. Dry run first — reports only, makes no vLLM calls:
           /opt/compactor-venv/bin/python /opt/zl-repo/scripts/import-history.py \\
               --webui-db /data/openwebui/webui.db \\
               --chat-id <conversation id> --store /data/openwebui/compactor
    4. Read the report. It states plainly whether `--apply` is safe: the
       source used to build the transcript, turns found, the existing
       watermark against what the transcript implies, how many L1/L2/L3
       units are due, an estimated (floor) count of real vLLM calls, an
       ESTIMATED wall-clock (seconds-per-call x calls; labelled an
       estimate because a chunk needing map-reduce costs more than one
       call), and every anomaly found.
    5. Run it for real:
           /opt/compactor-venv/bin/python /opt/zl-repo/scripts/import-history.py \\
               --webui-db /data/openwebui/webui.db \\
               --chat-id <conversation id> --store /data/openwebui/compactor \\
               --apply
       Interrupting it (Ctrl-C/SIGINT, SIGTERM, SIGHUP, kill -9, or a pod
       restart) is safe: the on-disk anchor is PRE-SEATED before any vLLM
       call (never left blank), and state is saved after every rollup
       unit, never batched (N1, v3.1.9.6 round-3 fix pass A) — so
       re-running resumes from the real watermark and never redoes
       finished work, and the live compactor's next request stays
       contiguous with whatever this run actually finished, even if it
       finished zero units. A `.bak-<stamp>` beside the summary file is
       this run's own pre-apply backup; restoring it by hand is never
       necessary for a normal interrupt-and-resume, only if you want to
       throw this run's progress away entirely.
    6. Start the compactor again:
           supervisorctl start compactor
    7. Check `curl -s http://127.0.0.1:8080/health/full` — `checks.hierarchy`
       should show the lag for this conversation falling on later requests.
"""

import argparse
import asyncio
import errno
import fcntl
import importlib.util
import json
import os
import shutil
import signal
import sqlite3
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
# A module-level name (not inlined at the call site) so a test can
# monkeypatch it to a temp directory, the same reason HERE itself is a
# module global rather than computed fresh inside every function.
OPT_COMPACTOR = Path("/opt/compactor")

DEFAULT_STORE = "/data/openwebui/compactor"
DEFAULT_VLLM_URL = "http://127.0.0.1:8000"
DEFAULT_HEALTH_URL = "http://127.0.0.1:8080/health"
DEFAULT_MAX_CALLS = 200
DEFAULT_SECONDS_PER_CALL = 40.0
HEALTH_PROBE_TIMEOUT_S = 3

_IMAGE_TYPES = ("image_url", "image", "input_image")

# The one invocation verified end to end against a pre-v3.1.9.4 pod (see
# RUNPOD_DEPLOY.md "Upgrading within v3.1.9.x" and OPERATIONS.md). Printed
# whenever no compactor package could be found at all, and whenever the one
# found is too old to answer the questions this script asks it (see
# _check_summarizer_capability) — in both cases a clone is the fix.
_CLONE_HINT = """\
On a pod, the supported way to run this script is from a clone of this
public repo (git ships in the image; there is no ssh/scp/rsync in it):

  git clone --depth 1 --branch <tag-or-branch> \\
      https://github.com/MrBanana8768/zions-light-ai.git /opt/zl-repo
  /opt/compactor-venv/bin/python /opt/zl-repo/scripts/import-history.py \\
      --webui-db /path/to/webui.db.export --chat-id <conversation id> \\
      --store /data/openwebui/compactor

A clone always carries a self-consistent target-release `compactor`
package beside this script, which is the one thing an old pod's own
installed package cannot supply for itself."""


# ---------------------------------------------------------------------------
# Where the real compactor package lives — resolved fresh every run, never
# assumed (Defect 1). Copied to /data/scripts/, HERE.parent/"compactor"
# resolves to /data/compactor, which does not exist: that is the exact
# operator error this function exists to stop happening again. Identical in
# spirit to scripts/backfill-records.py's own _resolve_compactor_pkg — see
# that function's docstring for the full reasoning and the layout trap.
# ---------------------------------------------------------------------------

_PROBE_MODULE = "summarizer"


def _resolve_compactor_pkg(explicit: str | None) -> tuple[Path | None, list[str], bool]:
    """(pkg_dir, tried, already_importable) — see
    scripts/backfill-records.py's `_resolve_compactor_pkg` for the full
    docstring; this is the same resolution order, probing for `summarizer`
    (this script's own entry point into the package) instead of
    `backfill`."""
    tried: list[str] = []
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


# ---------------------------------------------------------------------------
# Capability check (Defect 2) — the same discipline scripts/backfill-
# records.py applies to compactor/backfill.py, applied here to the
# summarizer/memory symbols this script depends on, so an incompatible
# package refuses with an actionable message rather than a raw
# AttributeError partway through a run. Every one of these has in fact been
# present continuously from v3.1.9 through v3.1.9.4 (verified), so this is
# a defensive floor against a package older or stranger than that range,
# not a fix for a known gap the way _check_backfill_capability is.
# ---------------------------------------------------------------------------

_REQUIRED_SUMMARIZER_SYMBOLS = (
    "load_state", "save_state", "recorded_position", "summary_path",
    "needs_rollup", "maybe_rollup", "vllm_call_budget_ctx",
    "L1_CHUNK_SIZE", "L2_CHUNK_SIZE", "L3_CHUNK_SIZE",
    # Defect 3's content guard: the fingerprint primitives it reuses rather
    # than reinventing (see _prefix_matches_store below).
    "_turn_fingerprints", "_align_candidates", "_covered_fps",
    "_covered_turn_fingerprints", "_FP_UNKNOWN",
    # N1/N4 (v3.1.9.6 round-3 fix pass A). The pre-seat (below) and the
    # offset-bound check inside _resolve_apply_feasibility both ask the
    # frozen module for its OWN constants/helper rather than hard-coding a
    # second copy that could silently drift from the real ones.
    "_ANCHOR_TURNS", "_FINGERPRINT_TAIL_TURNS", "_highest_chunk_turn",
)
_REQUIRED_MEMORY_SYMBOLS = ("conv_lock", "summary_archive_path")


def _check_summarizer_capability(summarizer_mod, memory_mod, pkg_source: str) -> str | None:
    """None if the package has everything this script needs; otherwise the
    full actionable error message (never a bare AttributeError)."""
    missing = [n for n in _REQUIRED_SUMMARIZER_SYMBOLS if not hasattr(summarizer_mod, n)]
    missing += [f"memory.{n}" for n in _REQUIRED_MEMORY_SYMBOLS if not hasattr(memory_mod, n)]
    if not missing:
        return None
    return (
        f"ERROR: the compactor package at {pkg_source} is missing "
        f"{', '.join(missing)} — this pod's installed compactor is too old "
        f"(or too different) for this script's rollup-catch-up drain, "
        f"which asks the REAL summarizer/memory modules for their own "
        f"rules rather than keeping a second copy of them (see this "
        f"script's own module docstring). Running against a package "
        f"missing these would either crash mid-drain or silently "
        f"misjudge what is due.\n\n"
        f"{_CLONE_HINT}"
    )


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _utc_stamp() -> str:
    """Separate function, not inlined, so a test can pin it (same reason
    scripts/backfill-records.py's own `_utc_stamp` is a function)."""
    return _now_utc().strftime("%Y%m%dT%H%M%SZ")


def _format_eta(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    return f"{minutes / 60:.1f}h"


def _connection_was_refused(exc: BaseException) -> bool:
    """True only for a CLEAN refusal (ECONNREFUSED) -- the one condition
    that reliably means "nothing is listening at this address", safe to
    read as "the compactor is not running". `urllib` sometimes raises the
    OSError bare and sometimes wraps it as `URLError(reason=...)`; this
    checks both shapes."""
    if isinstance(exc, urllib.error.URLError):
        exc = exc.reason if isinstance(exc.reason, BaseException) else exc
    return isinstance(exc, ConnectionRefusedError) or (
        isinstance(exc, OSError) and exc.errno == errno.ECONNREFUSED
    )


def _compactor_is_alive(url: str = DEFAULT_HEALTH_URL) -> bool | str:
    """True for a clean 200 from `url`. False ONLY for a clean CONNECTION
    REFUSED (nothing listening there -- safe to read as "not running").
    The string "ambiguous" for everything else this cannot tell apart
    from "not running": a timeout, a DNS failure, a non-200 HTTP response
    (N14, round-3 fix pass A -- something answering, however oddly, is
    evidence AGAINST "nothing is there", not for it; scripts/backfill-
    records.py's own probe already refuses on this), or any other
    exception.

    M3 (hostile review, this release). Until now every one of those
    non-200 conditions collapsed into plain `False`, which FAILS OPEN:
    "starting up, not yet bound", "a 3s timeout under GPU load", and
    "bound on a different port" all used to read exactly like "not
    running" and let `--apply` proceed -- racing a live rollup that may
    be writing this exact state file. Only a clean refusal is now trusted
    as evidence of "not running"; everything else refuses `--apply`
    (see the caller) unless `--force` says the operator has confirmed it
    some other way."""
    try:
        with urllib.request.urlopen(url, timeout=HEALTH_PROBE_TIMEOUT_S) as r:
            return r.status == 200
    except urllib.error.HTTPError:
        # N14 (v3.1.9.6 round-3 fix pass A). Until now this read as plain
        # `False` ("not running") -- but a real, non-200 HTTP response
        # means SOMETHING is bound and answering on this exact port,
        # which is direct evidence against "nothing is listening here",
        # not for it. scripts/backfill-records.py's own probe already
        # refuses on a non-200 rather than reading it as "not running"
        # (hostile review round 2, N14: "the importer treats a non-200
        # /health as not running and proceeds; backfill refuses in the
        # same case"). Matching that: ambiguous, so the caller refuses
        # unless --force says the operator confirmed some other way.
        return "ambiguous"
    except Exception as e:
        if _connection_was_refused(e):
            return False
        return "ambiguous"


class ChatNotFound(Exception):
    pass


class TranscriptUnavailable(Exception):
    pass


# ---------------------------------------------------------------------------
# webui.db: open read-only, refuse a live/crashed database
# ---------------------------------------------------------------------------

def _live_sidecars(db_path: Path) -> list[Path]:
    out = []
    for suffix in ("-journal", "-wal"):
        p = db_path.with_name(db_path.name + suffix)
        if p.exists():
            out.append(p)
    return out


def _open_ro(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{db_path.resolve().as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True, timeout=30)


# ---------------------------------------------------------------------------
# Content flattening — multimodal arrays become text + image placeholders
# ---------------------------------------------------------------------------

def _flatten_content(content: Any) -> str:
    """Whatever `content` is on a stored OpenWebUI message, as one string.
    A part shaped like an image (`type` one of `_IMAGE_TYPES`, or carrying
    an `image_url` key) becomes a short placeholder rather than being
    dropped silently or crashing on the part's own shape."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for c in content:
            if isinstance(c, str):
                if c:
                    parts.append(c)
                continue
            if not isinstance(c, dict):
                continue
            ctype = c.get("type")
            if ctype in _IMAGE_TYPES or "image_url" in c:
                parts.append("[image]")
            elif "text" in c or ctype == "text":
                text = c.get("text", "")
                if text:
                    parts.append(str(text))
            else:
                parts.append(f"[{ctype or 'unrecognised'} content]")
        return " ".join(parts)
    return str(content)


# ---------------------------------------------------------------------------
# Transcript reconstruction — the LINEAR branch, not insertion order
# ---------------------------------------------------------------------------

def _walk_branch_from_history(history: dict) -> list[dict] | None:
    """Walk `history["messages"]` back from `history["currentId"]` via
    `parentId`, then reverse — the branch OpenWebUI actually shows, not
    every abandoned edit or regenerated reply insertion order would
    include. None if the shape does not support it (missing map,
    missing/unknown currentId, or a cycle), so the caller can fall back
    rather than return a wrong answer."""
    messages_map = history.get("messages")
    current_id = history.get("currentId")
    if not isinstance(messages_map, dict) or not current_id:
        return None
    chain: list[dict] = []
    seen: set[str] = set()
    node_id = current_id
    while node_id is not None:
        if node_id in seen:
            return None  # cycle — corrupt; let the fallback try
        seen.add(node_id)
        node = messages_map.get(node_id)
        if not isinstance(node, dict):
            return None
        chain.append(node)
        node_id = node.get("parentId")
    chain.reverse()
    return chain


def _decode_chat_message_content(raw: Any) -> Any:
    """`chat_message.content` is JSON-encoded (confirmed empirically
    against the real 2026-09-22/23 backups: a plain string turn is
    stored as e.g. `'"hello"'`, quotes included) -- decode it once here
    before handing it to `_flatten_content`, which otherwise returns the
    raw JSON text (quotes, brackets and all) as if it were the turn's
    actual content. Used by both `_reconstruct_from_chat_message` (the
    fallback source) and `_cross_check_against_chat_message_table`
    (the divergence check) -- one site so the fix cannot drift between
    the two."""
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw


def _reconstruct_from_chat_message(con: sqlite3.Connection, chat_id: str) -> list[dict]:
    """Fallback source: the `chat_message` table (`chat_id`, `parent_id`,
    `role`, `content`, `created_at`). No `currentId` exists at this layer,
    so the leaf most recently CREATED stands in for it — the same choice
    OpenWebUI's own UI makes implicitly by always showing the newest
    branch tip."""
    try:
        rows = con.execute(
            "SELECT id, parent_id, role, content, created_at "
            "FROM chat_message WHERE chat_id = ?",
            (chat_id,),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    if not rows:
        return []
    # N-round-3-pass-A follow-up (chat_message id/parent_id convention).
    # OpenWebUI 0.11's `chat_message.id` is "<chat_id>-<msg_id>", but
    # `parent_id` is the BARE `msg_id` -- the same key space
    # `history.messages` itself uses. Keying `by_id` on the raw `id`
    # column (as this function used to) means `by_id[node["parent_id"]]`
    # (a bare id) can never match a key (always chat_id-prefixed): every
    # row looks like a leaf (`r[0] not in parent_ids` is vacuously true
    # for ALL rows, since prefixed ids never appear in the bare
    # parent_ids set either), and the walk from whichever leaf sorts
    # last by created_at stops after exactly one step, every time.
    # Verified empirically against the real 2026-09-22 and 2026-09-23
    # backups: this fallback would have reconstructed a 1-turn
    # "transcript" for a 3863+ turn conversation. Stripping the
    # `<chat_id>-` prefix once here, up front, is the fix -- the rest of
    # this function's own logic (leaf selection by created_at, walking
    # via parent_id) already assumed the bare key space.
    prefix = f"{chat_id}-"

    def _bare(raw_id):
        return raw_id[len(prefix):] if raw_id.startswith(prefix) else raw_id

    by_id = {_bare(r[0]): {"id": _bare(r[0]), "parent_id": r[1], "role": r[2],
                           "content": _decode_chat_message_content(r[3]),
                           "created_at": r[4]} for r in rows}
    parent_ids = {r[1] for r in rows if r[1] is not None}
    leaves = [r for r in rows if _bare(r[0]) not in parent_ids]
    if not leaves:
        leaves = rows
    leaf = max(leaves, key=lambda r: (r[4] if r[4] is not None else 0))
    chain: list[dict] = []
    seen: set[str] = set()
    node_id = _bare(leaf[0])
    while node_id is not None and node_id in by_id and node_id not in seen:
        seen.add(node_id)
        node = by_id[node_id]
        chain.append(node)
        node_id = node["parent_id"]
    chain.reverse()
    return [
        {"role": n["role"] or "unknown", "content": _flatten_content(n["content"])}
        for n in chain
    ]


class ChatMessageTableDivergence(Exception):
    """Raised when the chat_message table's own reconstruction of this
    conversation's current branch disagrees with the history.messages
    (chat.chat JSON) walk -- see _cross_check_against_chat_message_table's
    own docstring for why this is checked at all."""


def _cross_check_against_chat_message_table(
    con: sqlite3.Connection, chat_id: str, history_messages: dict, current_id: str | None,
) -> str | None:
    """None if the chat_message table either has no rows for this chat (a
    conversation OpenWebUI has never backfilled -- nothing to cross-check)
    or its own branch reconstruction agrees with the history.messages walk;
    otherwise a detail string describing the disagreement.

    WHY THIS EXISTS (architect follow-up, round-3 fix pass A). OpenWebUI
    0.11's OWN `get_messages_map_by_chat_id` (open_webui/models/chats.py)
    prefers `chat_message` rows over the embedded JSON history -- "Fast
    path: build from normalized chat_message rows" -- falling back to (or
    merging gaps from) `history.messages` only when the row-based parent
    graph has unresolved references. That is what the LIVE compactor
    actually receives on every real request; this script's own
    `reconstruct_transcript` walks `history.messages` FIRST instead,
    falling back to the table only if the JSON is unreadable. Verified
    empirically against the real 2026-09-22 and 2026-09-23 backups for the
    conversation this script exists for: the two reconstructions are
    byte-identical (same 3863/3879-turn branch, same ids, 0 text
    differences) -- so flipping the priority order is not motivated by an
    observed divergence on the one conversation this matters for, and
    doing so is a materially larger, riskier change than this fix pass has
    budget to re-verify safely. What IS cheap and safe: detecting a
    disagreement, when one exists, and refusing rather than silently
    trusting whichever source happened to be tried first.

    `chat_message.id` is "<chat_id>-<msg_id>" (confirmed empirically);
    `parent_id` is the bare `msg_id`, matching `history.messages`' own key
    space -- see `_reconstruct_from_chat_message`'s own docstring for the
    identical fix this same convention needed there.

    This mirrors (not byte-for-byte, but in effect) OpenWebUI's own
    `get_unresolved_parent_ids` + gap-fill: any parent id a table row
    references but the table itself does not contain is filled in from
    `history_messages` (if present there), exactly as OpenWebUI's real
    algorithm does before walking the branch.
    """
    try:
        rows = con.execute(
            "SELECT id, parent_id, role, content FROM chat_message WHERE chat_id = ?",
            (chat_id,),
        ).fetchall()
    except sqlite3.OperationalError:
        return None
    if not rows:
        return None

    prefix = f"{chat_id}-"

    def _bare(raw_id):
        return raw_id[len(prefix):] if raw_id.startswith(prefix) else raw_id

    table_map = {
        _bare(r[0]): {"id": _bare(r[0]), "parentId": r[1], "role": r[2],
                      "content": _decode_chat_message_content(r[3])}
        for r in rows
    }
    unresolved = {
        n["parentId"] for n in table_map.values()
        if n.get("parentId") and n["parentId"] not in table_map
    }
    for missing_id in unresolved:
        node = history_messages.get(missing_id)
        if isinstance(node, dict):
            table_map[missing_id] = node

    table_chain = _walk_branch_from_history({"messages": table_map, "currentId": current_id})
    if table_chain is None:
        return (
            "the chat_message table has rows for this chat but its own "
            "branch walk from currentId is broken (missing node or a "
            "cycle) -- refusing rather than trust the history.messages "
            "walk without being able to cross-check it"
        )

    json_chain = _walk_branch_from_history({"messages": history_messages, "currentId": current_id})
    if json_chain is None:
        return None  # nothing to compare against; the caller already
        # refuses elsewhere if history.messages itself produced nothing

    table_turns = [
        {"role": n.get("role") or "unknown", "content": _flatten_content(n.get("content"))}
        for n in table_chain
    ]
    json_turns = [
        {"role": n.get("role") or "unknown", "content": _flatten_content(n.get("content"))}
        for n in json_chain
    ]
    if table_turns == json_turns:
        return None
    if len(table_turns) != len(json_turns):
        return (
            f"the chat_message table's own branch reconstruction has "
            f"{len(table_turns)} turn(s) against {len(json_turns)} from "
            f"history.messages -- OpenWebUI's own code prefers the table; "
            f"this export's history.messages walk disagrees with what the "
            f"live compactor actually received"
        )
    for i, (t, j) in enumerate(zip(table_turns, json_turns)):
        if t != j:
            return (
                f"the chat_message table's own branch reconstruction "
                f"disagrees with history.messages at turn {i + 1} "
                f"(role/content differ) -- OpenWebUI's own code prefers "
                f"the table; this export's history.messages walk "
                f"disagrees with what the live compactor actually received"
            )
    return None  # unreachable (table_turns == json_turns already handled)


def reconstruct_transcript(
    con: sqlite3.Connection, chat_id: str
) -> tuple[list[dict], str, list[str]]:
    """(turns, source, notes). `turns` is `[{"role", "content"}, ...]`, the
    same shape the compactor receives. Raises ChatNotFound if no `chat`
    row matches `chat_id`; TranscriptUnavailable if neither source could
    produce anything. Raises ChatMessageTableDivergence if history.messages
    was used but disagrees with the chat_message table's own reconstruction
    -- see _cross_check_against_chat_message_table's own docstring."""
    notes: list[str] = []
    row = con.execute("SELECT chat FROM chat WHERE id = ?", (chat_id,)).fetchone()
    if row is None:
        raise ChatNotFound(chat_id)
    raw = row[0]

    chain = None
    history_messages: dict | None = None
    current_id = None
    try:
        data = json.loads(raw) if isinstance(raw, (str, bytes)) else None
        history = data.get("history") if isinstance(data, dict) else None
        if isinstance(history, dict):
            chain = _walk_branch_from_history(history)
            history_messages = history.get("messages")
            current_id = history.get("currentId")
        else:
            notes.append(
                "chat.chat has no readable history.messages/currentId; "
                "falling back to the chat_message table"
            )
    except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
        notes.append(
            "chat.chat could not be parsed as JSON; falling back to the "
            "chat_message table"
        )

    if chain is not None:
        turns = [
            {"role": n.get("role") or "unknown", "content": _flatten_content(n.get("content"))}
            for n in chain
        ]
        if turns:
            if isinstance(history_messages, dict):
                divergence = _cross_check_against_chat_message_table(
                    con, chat_id, history_messages, current_id
                )
                if divergence is not None:
                    raise ChatMessageTableDivergence(divergence)
            return turns, "history.messages (chat.chat JSON)", notes
        notes.append(
            "history.messages/currentId walk produced an empty branch; "
            "falling back to the chat_message table"
        )

    turns = _reconstruct_from_chat_message(con, chat_id)
    if turns:
        return turns, "chat_message table", notes
    raise TranscriptUnavailable(chat_id)


def detect_anomalies(turns: list[dict]) -> list[str]:
    """Non-alternating turns and a missing final reply. Reported, never
    crashed on — the caller still hands the array to the drain as-is."""
    anomalies: list[str] = []
    prev_role = None
    prev_idx = 0
    for i, t in enumerate(turns, start=1):
        role = t.get("role")
        if role == "system":
            continue
        if prev_role is not None and role == prev_role:
            anomalies.append(
                f"turns {prev_idx} and {i}: consecutive {role!r} turns "
                f"(expected alternation)"
            )
        prev_role, prev_idx = role, i
    non_system = [t for t in turns if t.get("role") != "system"]
    if non_system and non_system[-1].get("role") == "user":
        anomalies.append(
            "the transcript ends on a user turn with no assistant reply"
        )
    return anomalies


# ---------------------------------------------------------------------------
# Estimating what is due, without calling anything
# ---------------------------------------------------------------------------

def estimate_due(
    state: dict, current_turns: int, l1_size: int, l2_size: int, l3_size: int
) -> tuple[int, int, int]:
    """(l1_due, l2_due, l3_due) — a FLOOR estimate of rollup units, by pure
    arithmetic over the same thresholds `summarizer.py`'s real gates use
    (`_needs_l1_rollup` / `_needs_l2_rollup` / `_needs_l3_rollup`). Not
    exact: it assumes one real vLLM call per unit, and an oversized chunk
    can cost more via map-reduce (see `_summarize_pieces`). Good enough to
    size a dry run; `--apply` runs the real drain, which is exact.

    `current_turns` must be the EFFECTIVE position --apply will actually
    resume against (`max(recorded_position(state), the reconstructed
    branch's own turn count)` -- see the caller), NOT the raw
    reconstructed branch length on its own. The two differ exactly when
    the store's `turns_seen` already exceeds the branch (the ordinary
    "real backlog" shape a piecewise-offset conversation has -- see B1);
    passing the raw branch length there under-counts how many L1 chunks
    are actually due by however many turns the offset accounts for, since
    the count of due units depends only on how far the store's own
    position has to advance, never on which offset maps a label to
    branch text (v3.1.9.6 architect follow-up: this used to under-report,
    e.g. 57 chunks on the real backup instead of the true 58)."""
    last = int(state.get("last_summarized_turn") or 0)
    l1_count = len(state.get("l1") or [])
    l2_count = len(state.get("l2") or [])

    new_turns = max(0, current_turns - last)
    l1_due = new_turns // l1_size if l1_size > 0 else 0
    l1_count += l1_due

    l2_due = 0
    while l2_size > 0 and l1_count >= l2_size:
        l1_count -= l2_size
        l2_count += 1
        l2_due += 1

    l3_due = 0
    while l3_size > 0 and l2_count >= l3_size:
        l2_count -= l3_size
        l3_due += 1

    return l1_due, l2_due, l3_due


# ---------------------------------------------------------------------------
# Defect 3: verify CONTENT, not turn COUNT, against the store.
# ---------------------------------------------------------------------------

def _prefix_matches_store(
    state: dict, turns: list[dict], summarizer_mod
) -> tuple[bool | None, str]:
    """Whether `turns` (this run's reconstructed linear branch) is
    content-consistent with what the store already has recorded for this
    conversation — using the SAME fingerprint primitives
    `summarizer._observed_position` itself uses to align a window
    (`tail_fp` plus `_turn_fingerprints`/`_align_candidates`), rather than
    a plain turn-count comparison.

    WHY A LENGTH COMPARISON IS THE WRONG GUARD. `summarizer._recorded_
    position` (invariant I2 in `_observed_position`'s own docstring) is a
    MONOTONIC lower bound: `turns_seen` only ever grows, by design,
    because "a deletion, an edit, a branch switch or a bounded client
    window all shrink len(messages)+1" — quoting that docstring directly.
    This script reconstructs the CURRENT linear branch from
    `history.messages`, which is exactly the shape that shrinks once an
    earlier edit or an abandoned regeneration takes a message off the
    branch the user actually sees. A real 2026-09-22 backup proved this is
    not hypothetical: `recorded_position` 3883 against a 3863-turn linear
    reconstruction of the SAME conversation, fully explained by ~153
    messages sitting on abandoned branches — not a wrong `--chat-id`, not
    a stale export, and not something `--apply` should ever refuse over.

    Returns (matches, detail):
      True  — the store's own anchor (the last confirmed live turns) was
              found somewhere in this reconstruction. Proceed regardless
              of how `current_turns` compares to `recorded_position`.
      False — the anchor was searched for and is NOT anywhere in this
              reconstruction. Refuse: wrong conversation, or an export
              from a different lineage than the one the store tracks.
      None  — no anchor recorded yet (`tail_fp` empty — a conversation
              that has never been through a live rollup, or mid-reset).
              Nothing to check content against; the caller falls back to
              the length comparison this replaces.
    """
    anchor = [x for x in (state.get("tail_fp") or []) if isinstance(x, str)]
    if not anchor:
        return None, "the store has no recorded anchor (tail_fp) to check content against"
    fps = summarizer_mod._turn_fingerprints(turns)
    cands = summarizer_mod._align_candidates(anchor, fps)
    if cands:
        return True, (
            f"the store's last {len(anchor)} confirmed turn(s) were found in "
            f"this reconstruction — content confirmed. A turn count below "
            f"recorded_position is expected here if edits or abandoned "
            f"branches have shrunk the linear history since those turns "
            f"were last seen live (see this function's own docstring)"
        )
    return False, (
        f"the store's last {len(anchor)} confirmed turn(s) do not appear "
        f"anywhere in this reconstruction"
    )


# ---------------------------------------------------------------------------
# B1 (v3.1.9.6 fix pass 1). _prefix_matches_store above answers "is this
# roughly the right conversation" from a 4-turn tail_fp anchor -- good
# enough to gate reporting, but too weak to trust for the number that
# actually decides where new text gets labelled. Her real store's own
# position -> branch mapping is PIECEWISE (0, -6, -16, -18, -22): the
# missing turns are scattered abandoned-branch messages, not a clean
# prefix, so the drain's single flat `window_offset = position -
# len(window)` is wrong for everything after the first scatter. This
# derives that offset from harder evidence -- the store's own per-
# position covered-turn record (`_covered_fps` / `_record_chunk_fps`) --
# and requires the match to be UNIQUE before any real work runs. This is
# also the fix for M2 (a transcript sharing exactly one turn with a
# 4-turn anchor could satisfy _prefix_matches_store's weaker check); at
# least 8 independent non-empty fingerprints must agree, in order, for
# --apply to proceed.
# ---------------------------------------------------------------------------

# Never anchor on fewer than this many independent, non-empty covered-turn
# fingerprints, even when the last L1 chunk's own span is shorter (a small
# L1_CHUNK_SIZE, or a store recorded before this release). This is the
# floor that makes a one-turn (or few-turn) coincidence unable to pass --
# see M2.
_MIN_ANCHOR_FINGERPRINTS = 8


def _chunk_span_ending_at(state: dict, last_turn: int) -> int | None:
    """The `last_turn - first_turn + 1` span of whichever stored chunk
    (l1, l2, or l3) claims to END exactly at `last_turn` -- the chunk
    that `last_summarized_turn` names, wherever an L2 fold or L3 refresh
    has since moved it to. None if no chunk claims that turn (a fresh
    conversation, or a state file predating chunk-level spans)."""
    l1 = [c for c in (state.get("l1") or []) if isinstance(c, dict)]
    l2 = [c for c in (state.get("l2") or []) if isinstance(c, dict)]
    l3 = state.get("l3") if isinstance(state.get("l3"), dict) else None
    for c in l1 + l2 + ([l3] if l3 else []):
        ft, lt = c.get("first_turn"), c.get("last_turn")
        if isinstance(ft, int) and isinstance(lt, int) and lt == last_turn:
            return lt - ft + 1
    return None


def _resolve_resume_offset(
    state: dict, turns: list[dict], summarizer_mod, l1_chunk_size: int
) -> tuple[int | None, str]:
    """(offset, detail). `offset` is `position - len(window)` VERIFIED
    against the store's own covered-turn record, or None with a refusal
    detail if it cannot be determined uniquely.

    Algorithm (the architect's B1 design, v3.1.9.6): take the store's
    recorded covered-turn fingerprints (`summarizer._covered_fps`) for
    the last K positions ending at `last_summarized_turn`. K starts at
    the span of whichever chunk claims that turn (or `l1_chunk_size` if
    none does), never fewer than `_MIN_ANCHOR_FINGERPRINTS` non-UNKNOWN
    entries. Find the branch index j such that `turns`' own
    covered-turn fingerprints for the K turns ending at j equal that
    slice exactly, position for position (an _FP_UNKNOWN slot in the
    store's record can never match a real turn -- see
    summarizer._FP_UNKNOWN -- so only the non-UNKNOWN slots are
    required to match, but there must be at least
    `_MIN_ANCHOR_FINGERPRINTS` of them). If that match is not unique,
    grow K (reach further back into the record) and retry, until it is
    unique or the record runs out. `offset = last_summarized_turn - j`.

    A fresh conversation (nothing summarized: last_summarized_turn <= 0
    and no covered-fp record at all) needs no anchor -- offset 0, the
    ordinary case, unchanged from before this fix.
    """
    last = int(state.get("last_summarized_turn") or 0)
    entries = summarizer_mod._covered_fps(state)
    if last <= 0 and not entries:
        return 0, "nothing summarized yet for this conversation -- no anchor needed"
    if last <= 0:
        return None, (
            "the store has a covered-turn record but last_summarized_turn "
            "is 0 -- an inconsistent state file; refusing rather than "
            "guessing an offset"
        )
    if last > len(entries):
        return None, (
            f"the store's covered-turn record has only {len(entries)} "
            f"entries but last_summarized_turn is {last} -- an "
            f"inconsistent state file; refusing rather than guessing an "
            f"offset"
        )

    # N1 kill-test follow-up (round-3 fix pass A). `_chunk_span_ending_at`
    # can return the span of an L2 CHAPTER (or the L3 refresh) once L1
    # has folded away entirely -- measured up to 200+ turns on the real
    # backup, once enough L1 chunks had folded into one chapter. Using
    # that whole span as the STARTING k demands an EXACT match across
    # every one of those turns before growing at all; a single edited or
    # regenerated turn anywhere in a 200-turn span (ordinary and
    # expected -- see this function's own docstring on piecewise
    # offsets) then fails the WHOLE span, with nothing smaller ever
    # tried, and refuses with "none of the store's last 200 recorded
    # ... appear anywhere" even though a clean, unique match exists at
    # the ordinary l1_chunk_size granularity. Reproduced on the real
    # 2026-09-22 backup by a real-image kill -9 landing exactly on an L2
    # chapter boundary (turn 2800): the ORIGINAL bug this whole
    # mechanism exists to prevent (a silent flat-offset mislabel) never
    # fires here -- this is a SAFE refusal, not a hole -- but it
    # needlessly blocks the documented "re-run to continue" path for a
    # resume point that per-unit persistence (N1) makes newly reachable.
    # Anchoring on the owning chunk's OWN span is still right when that
    # chunk is an ordinary (or partial) L1 chunk; only a fold/refresh
    # far larger than that falls back to the ordinary starting point,
    # which the ambiguity-driven growth below can still enlarge from.
    _owning_span = _chunk_span_ending_at(state, last)
    min_span = (
        _owning_span if _owning_span and _owning_span <= l1_chunk_size
        else l1_chunk_size
    )
    transcript_fps = summarizer_mod._covered_turn_fingerprints(turns)
    fp_unknown = getattr(summarizer_mod, "_FP_UNKNOWN", None)
    n = len(transcript_fps)

    k = max(min_span, _MIN_ANCHOR_FINGERPRINTS)
    while True:
        if k > last:
            return None, (
                f"could not find a unique, sufficiently-anchored match "
                f"against the store's covered-turn record for conv "
                f"{state.get('conv_id')!r} even using the entire "
                f"{last}-position record ending at turn {last} -- "
                f"refusing rather than guess a resume offset"
            )
        target = entries[last - k:last]
        known = [(i, fp) for i, fp in enumerate(target) if fp != fp_unknown]
        if len(known) < _MIN_ANCHOR_FINGERPRINTS:
            k += 1
            continue
        matches = [
            j for j in range(k, n + 1)
            if all(transcript_fps[j - k + i] == fp for i, fp in known)
        ]
        if len(matches) == 1:
            j = matches[0]
            return last - j, (
                f"verified against {len(known)} non-empty covered-turn "
                f"fingerprint(s) of the store's last {k} recorded "
                f"position(s) ending at turn {last}, uniquely matching "
                f"branch turn {j}"
            )
        if len(matches) == 0:
            return None, (
                f"none of the store's last {k} recorded covered-turn "
                f"fingerprint(s) ending at turn {last} ({len(known)} "
                f"non-empty) appear anywhere in this reconstruction -- "
                f"refusing rather than risk labelling a chunk over text "
                f"it did not summarize"
            )
        # Ambiguous (M2): more than one branch position is consistent
        # with the evidence so far. Reach further back for more evidence
        # rather than trust a coincidence.
        k += 1


def _apply_resume_offset(
    turns: list[dict], offset: int, position: int
) -> list[dict] | None:
    """The window to hand the drain so that, inside the frozen
    `summarizer` module, `position - len(window) == offset` exactly (B1
    step 2): trimmed at the TAIL, preserving every leading turn
    (including any leading system messages) up through the
    `position - offset`-th non-system turn.

    `position` is the STORE's own recorded position (turns_seen /
    `summarizer.recorded_position`), NOT `len(turns)` -- those two
    legitimately differ (the real 2026-09-22 backup: turns_seen=3883
    against a 3863-turn reconstruction), and it is `position`, not the
    reconstructed branch's own length, that the frozen module's
    `_observed_position` converges on whenever the window handed to it
    is at least `highest_chunk_turn(state)` long (true for any offset
    this function is ever asked to apply, by construction of
    `_resolve_resume_offset`). Trimming by the branch's own length
    instead of `position` was caught by the real-backup test: it silently
    computed the wrong window length whenever the two differ, and B1
    step 4's belt-and-braces check refused the resulting write outright.

    None if `position - offset` is negative, or LONGER than the
    reconstructed transcript actually has turns for (the "double
    coverage" case -- the store claims more coverage than this
    transcript contains at all), either of which the caller must refuse
    rather than silently truncate the wrong way."""
    non_system_idx = [i for i, t in enumerate(turns) if t.get("role") != "system"]
    keep = position - offset
    if keep < 0 or keep > len(non_system_idx):
        return None
    if keep == 0:
        return []
    cutoff = non_system_idx[keep - 1]
    return turns[:cutoff + 1]


def _verify_offset_after_apply(
    before_entries: list[str], after_entries: list[str], turns: list[dict],
    offset: int, summarizer_mod,
) -> str | None:
    """Belt-and-braces (B1 step 4): re-map every NEWLY recorded
    covered-turn position to this same transcript at `position - offset`
    and assert the fingerprint the run just wrote agrees. None if every
    newly recorded entry checks out; otherwise a detail string naming
    the first position that does not, so the caller can restore the
    backup and refuse rather than trust a write that disagrees with the
    very transcript it was supposedly built from."""
    transcript_fps = summarizer_mod._covered_turn_fingerprints(turns)
    fp_unknown = getattr(summarizer_mod, "_FP_UNKNOWN", None)
    n = len(transcript_fps)
    start = len(before_entries)
    for pos in range(start + 1, len(after_entries) + 1):
        fp = after_entries[pos - 1]
        if fp == fp_unknown:
            continue
        j = pos - offset
        expected = transcript_fps[j - 1] if 1 <= j <= n else None
        if expected is None or fp != expected:
            return (
                f"position {pos}: recorded fingerprint {fp!r} does not "
                f"match the transcript at branch turn {j} (offset "
                f"{offset}) -- expected {expected!r}"
            )
    return None


# ---------------------------------------------------------------------------
# N1/N4/N5/N6 (v3.1.9.6 round-3 fix pass A). The architect's design for a
# crash-consistent apply: never clear the anchor (pre-seat it instead,
# below), persist after every UNIT rather than once per --apply run (see
# _run_apply_loop), and make the dry run and --apply share the SAME
# feasibility decision (_resolve_apply_feasibility) so a refusal --apply
# would give is never hidden behind a hard-coded "safe_to_apply: True".
# ---------------------------------------------------------------------------

def _state_signature(state: dict) -> tuple:
    """A cheap signature of the parts of `state` that represent REAL
    rollup progress: last_summarized_turn, plus the SHAPE of l1/l2/l3.
    Used by `_run_apply_loop` to judge whether one `maybe_rollup` call
    actually accomplished a unit of work (N5).

    last_summarized_turn ALONE under-counts: an L2 fold or the L3
    refresh is genuine, useful progress -- l1 chunks fold into an l2
    chapter, or l2 chapters fold into a fresh l3 refresh -- without
    moving last_summarized_turn at all. The round-2 hostile review's N5
    finding is exactly a re-run whose only due work was folds: the old
    code judged progress by the watermark alone, mistook real fold work
    for "nothing was accomplished", and reported exit 1 over a run that
    had, in fact, done something.
    """
    l3 = state.get("l3")
    return (
        int(state.get("last_summarized_turn") or 0),
        len(state.get("l1") or []),
        len(state.get("l2") or []),
        bool(l3),
        (l3.get("last_turn") if isinstance(l3, dict) else None),
    )


def _state_fingerprint(state: dict) -> tuple:
    """A signature broad enough to catch ANY externally-caused change
    between two reads of the same conv's state file -- used only by the
    final post-loop re-verify (N7) to detect a race with a writer this
    run's own lock and health probe did not catch, never for judging
    rollup progress (see `_state_signature`, narrower and progress-
    focused, for that)."""
    return (
        _state_signature(state),
        tuple(state.get("tail_fp") or []),
        state.get("head_fp"),
        state.get("window_turns"),
        int(state.get("turns_seen") or 0),
    )


def _compute_pre_seat_anchor(window: list[dict], summarizer_mod) -> tuple[list[str], str, int]:
    """(tail_fp, head_fp, window_turns) -- exactly what the frozen
    `_observed_position` would compute and persist for THIS window on
    its own first call, computed here with the frozen module's own
    fingerprint helpers (`_turn_fingerprints`, `_ANCHOR_TURNS`,
    `_FINGERPRINT_TAIL_TURNS`) so the two are byte-identical (N1, round-3
    fix pass A -- the architect's design, item 1).

    WHY THIS REPLACES CLEARING THE ANCHOR. Before this fix,
    `_run_apply_loop` CLEARED tail_fp/head_fp/window_turns to empty
    before the drain, relying on the first successfully COMPLETED pass
    to write a fresh, self-consistent anchor derived from this same
    window. A kill (kill -9, SIGTERM, SIGHUP, or a pod restart) landing
    between that clearing write and the first pass's completion left
    tail_fp empty ON DISK with nothing yet run to give it a fresh one --
    closing the documented "re-run to continue" path (a re-run's own
    `_prefix_matches_store` check reads an empty tail_fp as "no anchor
    recorded", refusing with "no content evidence either way" against a
    store whose recorded_position has not moved), and leaving the LIVE
    compactor path's `_observed_position` in its own no-anchor branch,
    which under-counts new turns by assuming exactly
    `_ASSUMED_NEW_TURNS` (2) per call rather than the true gap -- see the
    round-2 hostile review, N1.

    Pre-seating computes and writes the CORRECT, FINAL anchor value up
    front, in the SAME atomic write as everything else this run touches
    before the drain starts. There is no longer a moment where the
    on-disk anchor is a blank placeholder: it is always either the live
    path's original anchor (if this write has not landed yet) or this
    run's own anchor for the window it is about to summarize (once it
    has) -- and the second one is no more "wrong" to leave on disk after
    a kill than it would be after this run finishes normally, because
    it is exactly the tail of the window `_apply_resume_offset` already
    verified against the store's own covered-turn record.
    """
    non_system = [t for t in window if t.get("role") != "system"]
    n = len(non_system)
    if n == 0:
        return [], "", 0
    fps = summarizer_mod._turn_fingerprints(
        non_system[-summarizer_mod._FINGERPRINT_TAIL_TURNS:]
    )
    head_fp = summarizer_mod._turn_fingerprints(non_system[:1])[0]
    tail_fp = fps[-summarizer_mod._ANCHOR_TURNS:]
    return tail_fp, head_fp, n


class ApplyFeasibility:
    """The ONE decision the dry run and --apply must share (N6): the
    verified resume offset, the effective position to trim against, and
    either a window to hand the drain or a refusal detail -- never both,
    never neither. Both callers in `main()` read this from a single call
    to `_resolve_apply_feasibility`, so a refusal `--apply` would give is
    always exactly what the dry run already reported, and the dry run
    can never again hard-code `safe_to_apply: True` over a refusal it
    never actually checked for."""

    __slots__ = ("resume_offset", "resume_offset_detail", "effective_position",
                 "window", "refusal")

    def __init__(self, resume_offset, resume_offset_detail, effective_position,
                 window, refusal):
        self.resume_offset = resume_offset
        self.resume_offset_detail = resume_offset_detail
        self.effective_position = effective_position
        self.window = window
        self.refusal = refusal

    @property
    def feasible(self) -> bool:
        return self.refusal is None


def _resolve_apply_feasibility(
    state: dict, turns: list[dict], summarizer_mod, current_turns: int,
    recorded_position: int, l1_chunk_size: int,
) -> ApplyFeasibility:
    """N4 + N6 (v3.1.9.6 round-3 fix pass A). Resolves the verified resume
    offset, THEN derives the effective position and the window
    `--apply` would actually drain -- the same computation for both the
    dry run and `--apply`.

    N4's bug, and the fix. `effective_position = max(recorded_position,
    current_turns)` is only correct when there is no piecewise offset to
    honour at all (offset 0: nothing summarized yet, or a store that has
    never diverged from the branch). Once the verified offset is
    POSITIVE, `recorded_position` already encodes, under that SAME
    offset, how far the live conversation has been confirmed to reach --
    using `max(recorded_position, current_turns)` instead lets a branch
    LONGER than recorded_position (an export with more turns than the
    store has ever observed -- ordinary once live chat keeps running
    after an export is taken) push the window past what the frozen
    module's own `_observed_position` will actually converge on. Traced
    through the frozen module's own arithmetic: with a pre-seated anchor
    that matches the window's own tail exactly, `_observed_position`
    holds at `new=0`, so `position = max(len(window), recorded_position)`.
    Handing it a window of length `current_turns - offset` (the OLD
    formula, when `current_turns > recorded_position`) can make
    `len(window) > recorded_position`, so `position` collapses to
    `len(window)` itself -- i.e. `window_offset = position - len(window)
    = 0`, not the verified offset at all. The round-2 hostile review's
    repro (`t_ahead.sh`) shows exactly this: 133 real completions burned
    before the belt-and-braces check in `_verify_offset_after_apply`
    catches the mislabelling and restores -- correct, but only after
    doing (and then discarding) the whole drain.

    THE RULE. For offset 0 (nothing summarized yet, or no divergence):
    unchanged, `effective_position = max(recorded_position, current_turns)`.
    For a POSITIVE offset: `effective_position = recorded_position`
    directly (never `max`'d with `current_turns`), which makes
    `keep = recorded_position - offset` -- the window this run drains
    stops exactly at what the store already claims to have reached under
    that offset, leaving anything past it (turns the live conversation
    has had but the store has not yet confirmed) for the live compactor's
    own next request, which is the only path that ever legitimately
    advances turns_seen. This must additionally satisfy
    `highest_chunk_turn(state) <= keep <= current_turns`: below the
    floor, the window would need to cover LESS than the store's own
    existing chunks already claim (moving backward); above the ceiling,
    the transcript does not have enough turns to build that window at
    all (double coverage, the same failure `_apply_resume_offset`'s own
    bounds check already catches for the offset-0 case). Refuse rather
    than guess whenever either bound fails -- in BOTH the dry run and
    --apply (N6).
    """
    resume_offset, resume_offset_detail = _resolve_resume_offset(
        state, turns, summarizer_mod, l1_chunk_size
    )
    if resume_offset is None:
        # No verified offset at all -- --apply cannot proceed. The
        # ESTIMATE still uses the old (best-effort, informational-only)
        # formula so the dry run's due-count is not itself blank; only
        # `refusal` governs whether --apply may run.
        effective_position = max(recorded_position, current_turns)
        return ApplyFeasibility(
            resume_offset=None, resume_offset_detail=resume_offset_detail,
            effective_position=effective_position, window=None,
            refusal=resume_offset_detail,
        )

    if resume_offset > 0:
        effective_position = recorded_position
        keep = recorded_position - resume_offset
        highest = summarizer_mod._highest_chunk_turn(state)
        if not (highest <= keep <= current_turns):
            detail = (
                f"the verified resume offset ({resume_offset}) would need a "
                f"window of {keep} turn(s) (recorded_position "
                f"{recorded_position} - offset {resume_offset}), but that is "
                + (
                    f"below the {highest} turn(s) the store's own chunks "
                    f"already claim coverage through"
                    if keep < highest else
                    f"more than this {current_turns}-turn reconstruction "
                    f"actually has"
                )
                + " -- refusing rather than risk a gap or double coverage "
                  "(N4, round-3 fix pass A). This should be unreachable; if "
                  "it fires, the offset arithmetic above has a bug."
            )
            return ApplyFeasibility(
                resume_offset=resume_offset, resume_offset_detail=resume_offset_detail,
                effective_position=effective_position, window=None, refusal=detail,
            )
    else:
        effective_position = max(recorded_position, current_turns)
        # N6 follow-up (round-3 fix pass A). offset 0 carries NO verified
        # evidence the way a positive offset does -- it is
        # `_resolve_resume_offset`'s trivial "nothing summarized yet, or
        # no divergence" default. When recorded_position (this
        # conversation's own confirmed live position) is ahead of this
        # reconstruction under that trivial offset, that is the
        # ORDINARY real-backlog shape B1's own architect follow-up
        # named (the real 2026-09-22 backup, before it had ever been
        # chunked at all: turns_seen 3883 vs a 3863-turn branch) -- not
        # "double coverage". N6 making the dry run actually call this
        # function (instead of hard-coding safe_to_apply: True) is what
        # first exposed this: `_apply_resume_offset` would trim to
        # `keep = effective_position - 0`, which is LONGER than the
        # reconstruction has turns for, and refuse. The fix is not to
        # trim at all here: hand the WHOLE reconstruction, and let the
        # frozen module's own `_observed_position` -- given the anchor
        # this run pre-seats from this exact window (N1) -- compute the
        # real, DYNAMIC bounded-window offset itself
        # (`effective_position - current_turns`), the same arithmetic
        # the module has always used for an unbounded-history resend
        # against a capped position. `resume_offset` is updated to
        # match, so the belt-and-braces verify below checks the offset
        # the drain will ACTUALLY use, not the trivial one this
        # function started with.
        if effective_position > current_turns:
            resume_offset = effective_position - current_turns
            return ApplyFeasibility(
                resume_offset=resume_offset,
                resume_offset_detail=(
                    f"{resume_offset_detail} -- but recorded_position "
                    f"({recorded_position}) exceeds this "
                    f"{current_turns}-turn reconstruction, so the WHOLE "
                    f"reconstruction is used untrimmed and the frozen "
                    f"module's own bounded-window arithmetic determines "
                    f"the real offset ({resume_offset}), not the trivial "
                    f"one above"
                ),
                effective_position=effective_position, window=turns, refusal=None,
            )

    window = _apply_resume_offset(turns, resume_offset, effective_position)
    if window is None:
        detail = (
            f"the verified resume offset ({resume_offset}) would need a "
            f"window longer than this {current_turns}-turn reconstruction "
            f"actually is -- the store may already claim more coverage "
            f"than this transcript contains at all (double coverage). "
            f"Refusing rather than risk a gap or a duplicate."
        )
        return ApplyFeasibility(
            resume_offset=resume_offset, resume_offset_detail=resume_offset_detail,
            effective_position=effective_position, window=None, refusal=detail,
        )

    return ApplyFeasibility(
        resume_offset=resume_offset, resume_offset_detail=resume_offset_detail,
        effective_position=effective_position, window=window, refusal=None,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="import-history.py",
        description=(
            "Catch a conversation's hierarchical summary up from a "
            "webui.db export. Dry run by default; --apply is required to "
            "write anything or make any vLLM call."
        ),
    )
    ap.add_argument("--webui-db", required=True, metavar="PATH",
                     help="path to a webui.db EXPORT (never the live database)")
    ap.add_argument("--chat-id", required=True, metavar="ID",
                     help="OpenWebUI chat id to reconstruct")
    ap.add_argument("--conv-id", metavar="ID",
                     help="compactor conv_id, if different from --chat-id "
                          "(default: same as --chat-id)")
    ap.add_argument("--store", default=DEFAULT_STORE, metavar="PATH",
                     help=f"compactor storage root (default: {DEFAULT_STORE})")
    ap.add_argument("--vllm-url", default=DEFAULT_VLLM_URL, metavar="URL",
                     help=f"default: {DEFAULT_VLLM_URL}")
    ap.add_argument("--health-url", default=DEFAULT_HEALTH_URL, metavar="URL",
                     help=f"the live compactor's own health endpoint, "
                          f"probed before --apply (default: "
                          f"{DEFAULT_HEALTH_URL}). A clean connection "
                          f"refusal is read as 'not running' and --apply "
                          f"proceeds; a timeout or any other ambiguous "
                          f"response REFUSES --apply (M3, v3.1.9.6) "
                          f"unless --force says you have confirmed it "
                          f"some other way")
    ap.add_argument("--model", default=None, metavar="NAME",
                     help="default: $MODEL_REPO")
    ap.add_argument("--max-calls", type=int, default=DEFAULT_MAX_CALLS,
                     metavar="N",
                     help=f"bound on REAL vLLM calls per --apply run "
                          f"(default: {DEFAULT_MAX_CALLS}; documented "
                          f"overshoot of at most one rollup unit)")
    ap.add_argument("--seconds-per-call", type=float,
                     default=DEFAULT_SECONDS_PER_CALL, metavar="S",
                     help=f"for the dry-run wall-clock ESTIMATE only "
                          f"(default: {DEFAULT_SECONDS_PER_CALL})")
    ap.add_argument("--apply", action="store_true",
                     help="actually run the drain and write. Without "
                          "this: dry run, no writes, no vLLM calls.")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--force", action="store_true",
                     help="override the live-database and AMBIGUOUS-health "
                          "refusals (see the module docstring). Does NOT "
                          "override a DETECTED live compactor, a "
                          "backup-collision, or the cross-process import "
                          "lock (N7, v3.1.9.6 round-3 fix pass A) -- those "
                          "always refuse")
    ap.add_argument(
        "--compactor-pkg", default=None, metavar="PATH",
        help="explicit path to the compactor package directory (highest "
             "precedence; overrides the repo/clone-layout and /opt/compactor "
             "auto-detection — see the module docstring's resolution order)",
    )
    return ap


def _fatal(args, message: str) -> int:
    if args.json:
        print(json.dumps({"error": message}, indent=2))
    else:
        print(message)
    return 1


def main(argv=None) -> int:
    args = _build_argparser().parse_args(argv)
    warnings: list[str] = []

    db_path = Path(args.webui_db)
    if not db_path.is_file():
        return _fatal(args, f"ERROR: --webui-db {db_path} does not exist or is not a file.")

    sidecars = _live_sidecars(db_path)
    if sidecars and not args.force:
        return _fatal(
            args,
            f"REFUSING: {', '.join(str(p) for p in sidecars)} sits beside "
            f"{db_path} — this looks like a live or crashed database, not "
            f"a clean export. Pass --force to override (only if you are "
            f"certain this is a consistent snapshot)."
        )
    if sidecars and args.force:
        warnings.append(
            f"WARNING: --force overriding {len(sidecars)} sidecar file(s) "
            f"beside {db_path} that suggest a live or crashed database."
        )

    # B5 (v3.1.9.6). Validate --store BEFORE any work at all -- checked
    # with plain pathlib, before even resolving the compactor package, so
    # a typo is caught as early as a bad --webui-db is. The hostile review
    # found a typo'd --store silently building a brand-new phantom store
    # from scratch and reporting an encouraging "re-run with --apply",
    # while the REAL store (and the operator's belief that the backlog
    # was closing) sat untouched. This script exists to catch an EXISTING
    # conversation's hierarchy up from a backlog, so the conversation's
    # own summary file is required to already be there too: a bootstrap
    # of a conv_id the store has never seen is indistinguishable, from
    # here, from exactly that --store/--conv-id typo.
    conv_id = args.conv_id or args.chat_id
    store_root = Path(args.store)
    if not store_root.is_dir():
        return _fatal(
            args,
            f"ERROR: --store {store_root} does not exist or is not a "
            f"directory. Refusing before any work -- a typo here would "
            f"otherwise silently build a brand-new store from scratch and "
            f"treat an existing backlog as fresh, burning real vLLM time "
            f"against the wrong place.",
        )
    summaries_dir = store_root / "summaries"
    if not summaries_dir.is_dir():
        return _fatal(
            args,
            f"ERROR: --store {store_root} exists but has no summaries/ "
            f"subdirectory -- this does not look like a real compactor "
            f"storage root. Refusing before any work.",
        )
    conv_summary_file = summaries_dir / f"{conv_id}.json"
    if not conv_summary_file.is_file():
        return _fatal(
            args,
            f"ERROR: {conv_summary_file} does not exist -- conv "
            f"{conv_id!r} has no summary state under --store {store_root} "
            f"yet. This script catches an EXISTING conversation's "
            f"hierarchy up from a backlog; a conv_id the store has never "
            f"seen is indistinguishable, from here, from a --store or "
            f"--conv-id/--chat-id typo pointed at the wrong place. "
            f"Refusing before any work.",
        )

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
        pkg_source = f"an already-importable {_PROBE_MODULE!r} on sys.path (no package directory)"

    os.environ["COMPACTOR_STORAGE_ROOT"] = str(Path(args.store).resolve())
    try:
        import summarizer  # noqa: E402
        import memory  # noqa: E402
    except Exception as e:
        return _fatal(
            args,
            f"ERROR importing compactor package from {pkg_source}: "
            f"{type(e).__name__}: {e}",
        )

    capability_error = _check_summarizer_capability(summarizer, memory, pkg_source)
    if capability_error is not None:
        return _fatal(args, capability_error)

    warnings.append(f"NOTE: compactor package resolved from {pkg_source}")

    # B4 (v3.1.9.6). The module docstring, the argparser's own --model
    # help text, and the error message below all say MODEL_REPO is a
    # fallback -- it never was read. Every documented invocation that
    # omits --model used to fail every time.
    model = args.model or os.environ.get("MODEL_REPO")

    try:
        con = _open_ro(db_path)
    except Exception as e:
        return _fatal(args, f"ERROR opening {db_path} read-only: {type(e).__name__}: {e}")

    try:
        turns, source, notes = reconstruct_transcript(con, args.chat_id)
    except ChatNotFound:
        return _fatal(args, f"ERROR: no chat with id {args.chat_id!r} in {db_path}.")
    except TranscriptUnavailable:
        return _fatal(
            args,
            f"ERROR: could not reconstruct a transcript for chat "
            f"{args.chat_id!r} from either history.messages or "
            f"chat_message — neither source produced any turns.",
        )
    except ChatMessageTableDivergence as e:
        return _fatal(
            args,
            f"REFUSING: {e} -- this export's history.messages "
            f"reconstruction does not match what OpenWebUI's own code "
            f"(which prefers chat_message rows) would have built for "
            f"chat {args.chat_id!r}. Running would risk summarizing text "
            f"that is not what the live compactor actually received.",
        )
    finally:
        con.close()

    anomalies = detect_anomalies(turns)
    current_turns = sum(1 for t in turns if t.get("role") != "system")

    state = summarizer.load_state(conv_id)
    recorded_position = summarizer.recorded_position(state)

    # Defect 3: a CONTENT check, not a turn-count check. See
    # _prefix_matches_store's own docstring for why "current_turns <
    # recorded_position" alone is the wrong guard — recorded_position is a
    # monotonic lower bound that a legitimate edit or abandoned
    # regeneration can leave the (shorter) current linear branch below,
    # with no wrong-export or wrong-chat-id involved at all.
    matches, match_detail = _prefix_matches_store(state, turns, summarizer)
    if matches is False:
        return _fatal(
            args,
            f"REFUSING: {match_detail} for conv {conv_id!r} "
            f"({current_turns} turn(s) reconstructed from {source}). This "
            f"usually means the wrong --chat-id/--conv-id, or an export of "
            f"a different conversation than the one already tracked under "
            f"--store.",
        )
    if matches is None and current_turns < recorded_position:
        return _fatal(
            args,
            f"REFUSING: the reconstructed transcript has {current_turns} "
            f"turn(s), but conv {conv_id!r}'s store already records "
            f"position {recorded_position}, and {match_detail} — there is "
            f"no content evidence either way. Running would risk "
            f"summarizing text that is not the text the existing chunk "
            f"labels claim — check --chat-id/--conv-id, or that this "
            f"export is not older than the store under --store.",
        )

    # N4/N6 (v3.1.9.6 round-3 fix pass A): ONE function decides
    # feasibility -- the verified resume offset, the effective position,
    # and the window --apply would drain -- shared between the dry run
    # and --apply so a refusal --apply would give is never hidden behind
    # a hard-coded "safe_to_apply: True" (see _resolve_apply_feasibility's
    # own docstring for the offset-arithmetic bug this replaces).
    feasibility = _resolve_apply_feasibility(
        state, turns, summarizer, current_turns, recorded_position,
        summarizer.L1_CHUNK_SIZE,
    )
    resume_offset = feasibility.resume_offset
    resume_offset_detail = feasibility.resume_offset_detail

    # The dry run's own due-estimate must be computed on the SAME
    # effective position --apply resumes against (B1 architect
    # follow-up; N4 corrected it further), not the raw reconstructed
    # branch length: those two differ whenever the store's turns_seen
    # already exceeds the branch (the ordinary "real backlog" shape --
    # see _apply_resume_offset's docstring), and the raw length
    # under-counts how many L1 chunks are actually due by however many
    # turns the gap accounts for.
    effective_position = feasibility.effective_position
    l1_due, l2_due, l3_due = estimate_due(
        state, effective_position,
        summarizer.L1_CHUNK_SIZE, summarizer.L2_CHUNK_SIZE, summarizer.L3_CHUNK_SIZE,
    )
    est_calls = l1_due + l2_due + l3_due
    est_seconds = est_calls * args.seconds_per_call
    work_due = est_calls > 0

    report: dict[str, Any] = {
        "conv_id": conv_id,
        "chat_id": args.chat_id,
        "webui_db": str(db_path),
        "store": str(Path(args.store).resolve()),
        "transcript_source": source,
        "transcript_notes": notes,
        "turns_found": current_turns,
        "last_summarized_turn_before": int(state.get("last_summarized_turn") or 0),
        "turns_seen_before": int(state.get("turns_seen") or 0),
        "recorded_position_before": recorded_position,
        "l1_chunks_due_estimate": l1_due,
        "l2_folds_due_estimate": l2_due,
        "l3_refreshes_due_estimate": l3_due,
        "estimated_real_vllm_calls": est_calls,
        "estimated_wall_clock": (
            f"~{_format_eta(est_seconds)} (ESTIMATE: {args.seconds_per_call:g}s "
            f"per call x {est_calls} call(s); a chunk needing map-reduce "
            f"costs more than one call)"
        ),
        "anomalies": anomalies,
        "apply": args.apply,
        "warnings": warnings,
        "resume_offset": resume_offset,
        "resume_offset_detail": resume_offset_detail,
    }

    if not args.apply:
        # N6 (v3.1.9.6 round-3 fix pass A). safe_to_apply is no longer a
        # hard-coded True: when there is work due, it is exactly
        # `feasibility.feasible` -- the SAME check --apply itself runs
        # below -- so an edit inside the last covered chunk, or a
        # net-shrinking edit in the unsummarized region, is refused by
        # the dry run too, not just discovered by --apply after the fact.
        # When nothing is due, --apply would be a no-op regardless of
        # whether a hypothetical window could be resolved, so this is
        # trivially safe.
        safe_to_apply = feasibility.feasible or not work_due
        if not work_due:
            report["note"] = "DRY RUN — nothing is due; --apply would do nothing."
            rc = 0
        elif safe_to_apply:
            report["note"] = "DRY RUN — no writes, no vLLM calls. Re-run with --apply once ready."
            rc = 3
        else:
            report["note"] = (
                f"DRY RUN — --apply would REFUSE: {feasibility.refusal}"
            )
            rc = 1
        report["safe_to_apply"] = safe_to_apply
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            _print_dry_run_report(report)
        return rc

    # --- --apply ---
    if not work_due:
        report["note"] = "nothing due — --apply made no changes."
        report["applied"] = []
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print(f"conv={conv_id}: nothing due. Not touching the store.")
        return 0

    if model is None:
        return _fatal(
            args,
            "ERROR: --apply needs a model — pass --model or set MODEL_REPO.",
        )

    # M3 (v3.1.9.6). A clean connection refusal is the only condition
    # trusted as "not running"; a timeout or anything else ambiguous now
    # REFUSES rather than failing open (see _compactor_is_alive's own
    # docstring).
    health = _compactor_is_alive(args.health_url)
    if health == "ambiguous":
        if not args.force:
            return _fatal(
                args,
                f"REFUSING --apply: {args.health_url} did not answer "
                f"clearly within {HEALTH_PROBE_TIMEOUT_S}s (a timeout, or "
                f"something short of a clean connection refusal) -- this "
                f"cannot be told apart from the compactor starting up, "
                f"overloaded, or bound on a different port, and "
                f"--apply's blast radius is a state file it may be "
                f"writing right now. Pass --force to override only if "
                f"you have confirmed some other way that it is not "
                f"running.",
            )
        warnings.append(
            f"WARNING: --force overriding an AMBIGUOUS health check at "
            f"{args.health_url} (a timeout, not a clean refusal) -- "
            f"proceeding anyway."
        )
        health = False
    if health:
        # N7 (v3.1.9.6 round-3 fix pass A). --force no longer overrides a
        # DETECTED live compactor at all. memory.conv_lock is an
        # in-process asyncio.Lock only, and there is no cross-process
        # lock the live compactor also respects (checked: it does not
        # take a file lock either) -- so this health check is the ONLY
        # thing standing between --apply and a live rollup tearing this
        # exact state file. The round-2 hostile review's t_conc.sh
        # showed exactly this with the old --force override: a live
        # exchange raced the drain, the live copy won on disk, and the
        # importer still printed "apply complete", exit 0. Forcing past
        # a health check that DID detect something live is not "the
        # operator confirmed some other way" -- it is overriding the one
        # piece of live evidence this script has. Stop it first.
        return _fatal(
            args,
            f"REFUSING --apply: something is answering {args.health_url} "
            f"— the live compactor may be writing this exact state file "
            f"right now. Stop it first (supervisorctl stop compactor). "
            f"--force cannot override this refusal (N7, round-3 fix pass "
            f"A) -- see this script's own module docstring for why.",
        )

    # N6 (v3.1.9.6 round-3 fix pass A): the SAME feasibility decision the
    # dry run already reported (offset verification, the N4-corrected
    # effective position, and the window trim) is what gates real work
    # here too -- one refusal message, never two independently-computed
    # ones that could drift apart.
    if feasibility.refusal is not None:
        return _fatal(args, f"REFUSING --apply: {feasibility.refusal}")
    window = feasibility.window
    warnings.append(
        f"NOTE: resume offset verified at {resume_offset} -- "
        f"{resume_offset_detail}"
    )

    summary_path = summarizer.summary_path(conv_id)
    archive_path = memory.summary_archive_path(conv_id)

    # N7 (v3.1.9.6 round-3 fix pass A): an exclusive lock around the
    # WHOLE apply (backup through the final re-verify below), so two
    # import-history.py --apply invocations against the SAME conv can
    # never interleave their writes. This is importer-vs-importer
    # protection only -- the live compactor does not take this lock (it
    # has no on-disk lock of its own at all; memory.conv_lock is an
    # in-process asyncio.Lock, checked above) -- so the live-compactor
    # health check (and the final re-verify below) remain the only
    # protection against THAT race. The lock file holds no data; the
    # advisory OS lock on the open fd is released automatically when the
    # holding process exits for any reason, including kill -9, so a
    # stale lock file left behind by a killed run is never a deadlock.
    lock_path = summary_path.with_name(summary_path.name + ".import.lock")
    lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(lock_fd)
        return _fatal(
            args,
            f"REFUSING --apply: {lock_path} is already locked -- another "
            f"import-history.py --apply appears to be running against "
            f"this same conv. Refusing regardless of --force: running two "
            f"applies concurrently against the same state file is exactly "
            f"the unguarded race N7 exists to stop. If no other import is "
            f"actually running, a stale lock left by a killed process is "
            f"safe to remove by hand -- the OS releases the advisory lock "
            f"itself the instant the holding process exits, so this "
            f"refusal only ever means a live holder or a leftover empty "
            f"file from one that died before it could be cleaned up.",
        )

    try:
        backup_path = None
        archive_backup_path = None
        if summary_path.exists():
            stamp = _utc_stamp()
            backup_path = summary_path.with_name(summary_path.name + f".bak-{stamp}")
            if backup_path.exists():
                return _fatal(
                    args,
                    f"REFUSING --apply: backup path already exists: "
                    f"{backup_path}. Remove or rename it before re-running.",
                )
            shutil.copy2(summary_path, backup_path)
            # H3 (v3.1.9.6). An L2 fold or L3 refresh rewrites the
            # archive sidecar too (_archive_chapters) -- "its entire
            # blast radius is one file" was not true once either ran,
            # and a rollback after B1 is exactly the moment this
            # matters: restoring only summary_path would leave the
            # archive holding chapters from the discarded run.
            if archive_path.exists():
                archive_backup_path = archive_path.with_name(
                    archive_path.name + f".bak-{stamp}"
                )
                if archive_backup_path.exists():
                    return _fatal(
                        args,
                        f"REFUSING --apply: backup path already exists: "
                        f"{archive_backup_path}. Remove or rename it "
                        f"before re-running.",
                    )
                shutil.copy2(archive_path, archive_backup_path)

        before_covered_fps = summarizer._covered_fps(state)

        interrupt_flag: dict[str, str] = {}

        def _on_signal(signum, _frame):
            try:
                interrupt_flag["signal"] = signal.Signals(signum).name
            except ValueError:
                interrupt_flag["signal"] = str(signum)

        # N1 (v3.1.9.6 round-3 fix pass A). SIGINT, SIGTERM and SIGHUP are
        # now handled IDENTICALLY: a handler that only records which
        # signal arrived and returns, never raising into the running
        # event loop. Ctrl-C used to rely on Python's default
        # KeyboardInterrupt, which the interpreter can raise at ANY
        # bytecode boundary -- including mid-await inside a real network
        # call -- and SIGTERM/SIGHUP raise nothing at all by default, so
        # they used to just kill the process outright with no chance to
        # report anything. The flag-based handler below is the shape all
        # three need anyway: the CURRENT rollup unit is already
        # guaranteed by the frozen summarizer to finish once it starts
        # (see _budget_allows_unit's own docstring), so it runs to
        # completion and is saved exactly as any other unit is; the
        # per-unit loop in `_run_apply_loop` then notices the flag and
        # stops BEFORE starting the next one, rather than being torn out
        # of the one in flight.
        _previous_handlers = {
            sig: signal.signal(sig, _on_signal)
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
        }
        try:
            try:
                passes = asyncio.run(_run_apply_loop(
                    summarizer, memory, conv_id, window,
                    args.vllm_url, model, args.max_calls,
                    interrupt_flag=interrupt_flag,
                ))
            except KeyboardInterrupt:
                # Defensive fallback only -- with the handlers above
                # installed, SIGINT no longer raises this. Kept in case
                # something else in the stack still does.
                return _fatal(
                    args,
                    "INTERRUPTED: a raw KeyboardInterrupt reached main() "
                    "(not through the flag-based signal handler -- this "
                    "should not normally happen). Whatever unit was in "
                    "flight may or may not have been saved; re-run to "
                    "check and continue.",
                )
        finally:
            for sig, handler in _previous_handlers.items():
                signal.signal(sig, handler)

        final_state = summarizer.load_state(conv_id)

        # B1 step 4 (belt-and-braces, v3.1.9.6): re-map every NEWLY
        # recorded covered-turn position back onto this same transcript
        # at `position - resume_offset` and assert the fingerprint just
        # written agrees. Never expected to fire when the arithmetic
        # above is right; if it does, trust the backup over the write
        # just made rather than leave a state file that mislabels its
        # own chunks.
        after_covered_fps = summarizer._covered_fps(final_state)
        verify_detail = _verify_offset_after_apply(
            before_covered_fps, after_covered_fps, turns, resume_offset, summarizer
        )
        if verify_detail is not None:
            if backup_path is not None:
                shutil.copy2(backup_path, summary_path)
            else:
                summary_path.unlink(missing_ok=True)
            if archive_backup_path is not None:
                shutil.copy2(archive_backup_path, archive_path)
            elif archive_path.exists():
                # N13 (v3.1.9.6 round-3 fix pass A). The condition here
                # used to also require `backup_path is None` -- but
                # B5 already guarantees summary_path (and so backup_path)
                # exists on every real run, making that clause dead: a
                # NEW archive.json this run itself created (an L2 fold or
                # L3 refresh's own _archive_chapters write, with none
                # existing before) was never cleaned up on a restore,
                # leaving discarded chapters behind under a conv this
                # exact run had just refused. The only question that
                # matters is whether an archive backup was taken at all
                # (archive_backup_path is None means it was not, because
                # none existed before this run) -- backup_path (the
                # SUMMARY file's own backup) is unrelated.
                archive_path.unlink(missing_ok=True)
            return _fatal(
                args,
                f"REFUSING: the covered-turn record this run just wrote "
                f"disagrees with the transcript it was built from "
                f"({verify_detail}) -- restored the pre-apply backup and "
                f"refusing rather than leave a state file that mislabels "
                f"its own chunks. This should be unreachable; if it "
                f"fires, the offset arithmetic above has a bug.",
            )

        # N7 (v3.1.9.6 round-3 fix pass A). Re-verify on disk: even
        # holding this run's own lock and having refused a DETECTED live
        # compactor above, nothing stops a live rollup that started
        # AFTER the health probe (or one this run's own --force on the
        # ambiguous-health path chose to override) from writing to this
        # exact file while --apply ran. Re-reading once more and
        # comparing against this run's OWN last known state catches
        # that: silently reporting "apply complete" over a state the
        # live path has since overwritten is exactly the failure the
        # round-2 hostile review's t_conc.sh demonstrated.
        racing_state = summarizer.load_state(conv_id)
        if _state_fingerprint(racing_state) != _state_fingerprint(final_state):
            return _fatal(
                args,
                f"REFUSING: the store for conv {conv_id!r} changed "
                f"underneath this run between its own last write and this "
                f"final check -- something else (most likely a live "
                f"compactor this run's health probe did not catch) wrote "
                f"to it concurrently. Refusing to report success over a "
                f"race; inspect {summary_path} by hand before re-running.",
            )

        report["backup"] = str(backup_path) if backup_path else None
        report["archive_backup"] = str(archive_backup_path) if archive_backup_path else None
        report["passes"] = passes["log"]
        report["rollup_calls"] = passes["rollup_calls"]
        report["vllm_calls_spent"] = passes["vllm_calls_spent"]
        report["budget_exhausted"] = passes["exhausted"]
        report["stopped_because"] = passes["stopped_because"]
        report["interrupted_signal"] = passes.get("interrupted_signal")
        report["last_summarized_turn_after"] = final_state.get("last_summarized_turn", 0)
        report["turns_seen_after"] = final_state.get("turns_seen", 0)
        report["still_due"] = summarizer.needs_rollup(
            final_state, summarizer.recorded_position(final_state)
        )

        # H2/N5 (v3.1.9.6; N5 round-3 fix pass A). The exit-code table in
        # the module docstring. Progress is judged by whether any UNIT
        # actually completed and was saved (`rollup_calls` -- see
        # `_state_signature` and `_run_apply_loop`), not by the watermark
        # alone: an L2 fold or the L3 refresh is real, saved progress
        # that never moves last_summarized_turn, and the old
        # watermark-only check mistook a fold-only re-run for "nothing
        # was accomplished" (N5). An --apply that never completed a
        # single unit accomplished nothing -- vLLM unreachable from the
        # first call, or an interrupt before any unit finished -- which
        # is a 1, not a silent 0. Real progress with more still due is a
        # 4 (re-run to continue, including when a signal is why it
        # stopped -- with per-unit saves, completed units really ARE
        # saved, so this is an honest "re-run", not a claim nothing was
        # lost that isn't true); real progress that reaches "nothing
        # left due" is a plain 0.
        progress_made = report["rollup_calls"] > 0
        sig_name = report["interrupted_signal"]
        if not progress_made:
            if sig_name:
                report["note"] = (
                    f"INTERRUPTED by {sig_name} before any unit completed "
                    f"-- nothing was accomplished this run. Nothing needs "
                    f"cleanup or restoring: the pre-apply anchor pre-seated "
                    f"for this window (see this script's own N1 fix) is "
                    f"exactly what a fresh pass would write anyway. "
                    f"Re-run to continue."
                )
            else:
                report["note"] = (
                    "apply ran but made no progress at all — the watermark "
                    "never advanced (see stopped_because); nothing was "
                    "accomplished"
                )
            rc = 1
        elif report["still_due"]:
            if sig_name:
                report["note"] = (
                    f"INTERRUPTED by {sig_name} after {report['rollup_calls']} "
                    f"unit(s) completed -- each was saved as it finished "
                    f"(state is saved after every unit, never batched). "
                    f"Re-run to continue; nothing needs cleanup."
                )
            else:
                report["note"] = "apply stopped with more work due — re-run to continue"
            rc = 4
        else:
            report["note"] = "apply complete"
            rc = 0

        if args.json:
            print(json.dumps(report, indent=2))
        else:
            _print_apply_report(report)
        return rc
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


async def _run_apply_loop(summarizer, memory, conv_id, messages, vllm_url, model, max_calls,
                           interrupt_flag: dict | None = None):
    """The real drain: mirrors main.admin_compact's own loop exactly (same
    module, same function, different transcript) -- but ONE UNIT (one L1
    chunk, one L2 fold, or the L3 refresh) per `maybe_rollup` call, not
    the whole backlog in one call (N1, round-3 fix pass A, architect
    design item 2). Each call is given its own fresh `vllm_call_budget=
    {"remaining": 1, ...}`, which the frozen `_budget_allows_unit` reads
    at each unit boundary (see that function's own docstring): allowed to
    start exactly one unit, guaranteed to finish it, then refuses a
    second one within the SAME call. `maybe_rollup` itself saves state
    (atomically: tempfile + fsync + rename) exactly once per call, when
    anything changed -- so persisting per unit falls out of calling it
    once per unit, with no extra bookkeeping of "was this saved yet" for
    this loop to get wrong. A kill (kill -9, SIGTERM, SIGHUP, or a pod
    restart) at ANY point during this loop therefore leaves the store at
    one of: the pre-seated anchor with zero units applied, or that anchor
    plus N completed (and saved) units, for some N -- never a partial
    unit and never a blank anchor (see `_compute_pre_seat_anchor`, called
    by the caller before this loop starts).

    `interrupt_flag`, if given, is a plain dict a SIGINT/SIGTERM/SIGHUP
    handler installed by the caller writes `{"signal": "SIGTERM"}` (or
    similar) into -- checked here between units, never inside one, so
    the unit currently running always finishes and is saved before this
    loop notices and stops. See main()'s own signal-handler installation
    for why this replaces relying on KeyboardInterrupt/BaseException
    propagation entirely.

    Progress is judged by comparing `_state_signature` before and after
    each call (N5): last_summarized_turn ALONE under-counts a fold-only
    or refresh-only unit, which moves neither but is still real, saved
    work.
    """
    interrupt_flag = interrupt_flag if interrupt_flag is not None else {}
    log: list[dict] = []

    # N1 (round-3 fix pass A): PRE-SEAT the anchor, never clear it. See
    # _compute_pre_seat_anchor's own docstring for the full reasoning —
    # this closes the exact crash window the round-2 hostile review's N1
    # finding demonstrated (a kill between "clear" and "first pass
    # completes" used to leave tail_fp empty on disk with nothing yet
    # run to give it a fresh one).
    tail_fp, head_fp, window_turns = _compute_pre_seat_anchor(messages, summarizer)
    async with memory.conv_lock(conv_id):
        _state = summarizer.load_state(conv_id)
        _state["tail_fp"] = tail_fp
        _state["head_fp"] = head_fp
        _state["window_turns"] = window_turns
        summarizer.save_state(conv_id, _state)
        last_known_state = _state

    units_completed = 0
    vllm_calls_spent = 0
    stopped_because = None
    interrupted_signal = None

    while True:
        sig_name = interrupt_flag.get("signal")
        if sig_name:
            interrupted_signal = sig_name
            stopped_because = (
                f"interrupted by {sig_name} after {units_completed} unit(s) "
                f"completed and saved"
            )
            break
        if vllm_calls_spent >= max_calls:
            stopped_because = f"hit max_calls={max_calls} (vLLM calls)"
            break

        budget = {"remaining": 1, "exhausted": False}
        prev_sig = _state_signature(last_known_state)
        try:
            new_state = await summarizer.maybe_rollup(
                conv_id, messages, vllm_url, model, vllm_call_budget=budget,
            )
        except Exception as e:
            stopped_because = f"{type(e).__name__}: {e}"
            break

        spent = max(0, 1 - budget["remaining"])
        vllm_calls_spent += spent
        new_sig = _state_signature(new_state)
        last_known_state = new_state

        if new_sig == prev_sig:
            # Nothing changed. Distinguish a real call that fired but
            # produced no visible progress (a genuine stall -- the old
            # code's own "watermark stopped advancing" case, still
            # possible for a persistently failing tier) from simply
            # nothing being due at all (spent == 0: the unit boundary
            # check never let anything start).
            stopped_because = (
                "the watermark stopped advancing" if spent > 0
                else "nothing left due"
            )
            break

        units_completed += 1
        log.append({
            "pass": units_completed,
            "last_summarized_turn": new_state.get("last_summarized_turn", 0),
        })

    return {
        "log": log,
        "rollup_calls": units_completed,
        "vllm_calls_spent": vllm_calls_spent,
        "exhausted": vllm_calls_spent >= max_calls,
        "stopped_because": stopped_because,
        "interrupted_signal": interrupted_signal,
    }


def _print_dry_run_report(report: dict) -> None:
    print("=" * 70)
    print(f"import-history.py — dry run for conv={report['conv_id']}")
    print("=" * 70)
    print(f"source:           {report['webui_db']}")
    print(f"store:            {report['store']}")
    print(f"transcript from:  {report['transcript_source']}")
    for n in report["transcript_notes"]:
        print(f"  note: {n}")
    print(f"turns found:      {report['turns_found']}")
    print(f"watermark before: last_summarized_turn={report['last_summarized_turn_before']} "
          f"turns_seen={report['turns_seen_before']} "
          f"recorded_position={report['recorded_position_before']}")
    print(f"due (estimate):   L1 chunks={report['l1_chunks_due_estimate']} "
          f"L2 folds={report['l2_folds_due_estimate']} "
          f"L3 refreshes={report['l3_refreshes_due_estimate']}")
    print(f"estimated calls:  {report['estimated_real_vllm_calls']}")
    print(f"estimated time:   {report['estimated_wall_clock']}")
    print(f"resume offset:    {report['resume_offset']} ({report['resume_offset_detail']})")
    if report["anomalies"]:
        print("anomalies:")
        for a in report["anomalies"]:
            print(f"  - {a}")
    else:
        print("anomalies:        none")
    for w in report["warnings"]:
        print(w)
    print()
    print(report["note"])


def _print_apply_report(report: dict) -> None:
    print("=" * 70)
    print(f"import-history.py — --apply for conv={report['conv_id']}")
    print("=" * 70)
    print(f"backup:           {report['backup']}")
    print(f"archive backup:   {report.get('archive_backup')}")
    for p in report["passes"]:
        print(f"  pass {p['pass']}: last_summarized_turn -> {p['last_summarized_turn']}")
    print(f"rollup passes:    {report['rollup_calls']}")
    print(f"vLLM calls spent: {report['vllm_calls_spent']}")
    print(f"budget exhausted: {report['budget_exhausted']}")
    print(f"stopped because:  {report['stopped_because']}")
    if report.get("interrupted_signal"):
        print(f"interrupted by:   {report['interrupted_signal']}")
    print(f"watermark after:  last_summarized_turn={report['last_summarized_turn_after']} "
          f"turns_seen={report['turns_seen_after']}")
    print(f"still due:        {report['still_due']}")
    for w in report["warnings"]:
        print(w)
    print()
    print(report["note"])


if __name__ == "__main__":
    sys.exit(main())
