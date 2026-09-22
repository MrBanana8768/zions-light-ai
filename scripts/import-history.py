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

EXIT CODES. The same convention as scripts/backfill-records.py and
scripts/setup-sshd.py — normalized across all three (see CHANGELOG.md and
OPERATIONS.md; setup-sshd.py used to have 0 and 3 swapped from this):
    0   success — nothing was due (dry run), or `--apply` ran to
        completion (even if it stopped early on its own call budget or an
        LLM failure — see `stopped_because` in its report; that is not a
        script error)
    1   a refusal or an error the operator needs to look at: a bad
        `--webui-db` or `--store`, an unknown `--chat-id`, a
        `-journal`/`-wal` beside the database without `--force`, no
        usable compactor package (or one missing what this script needs
        from it — see PACKAGE RESOLUTION above), a refused `--apply`
        precondition (live compactor without `--force`, an existing
        backup path, no `--model`/`MODEL_REPO`), or a reconstructed
        transcript whose content does not match what the store already
        has confirmed (see _prefix_matches_store — NOT simply "fewer
        turns than recorded_position", which a normal edit or abandoned
        branch can cause on its own)
    3   a DRY RUN found one or more rollup units due — informational, not
        a failure: re-run with `--apply` once ready
    (argparse's own usage errors — unknown flags, missing required values
    — exit 2, the standard library's own convention, unrelated to the
    three above)

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
       Interrupting it (Ctrl-C, a pod restart) is safe: state is saved
       after every rollup unit, never batched, so re-running resumes from
       the real watermark and never redoes finished work.
    6. Start the compactor again:
           supervisorctl start compactor
    7. Check `curl -s http://127.0.0.1:8080/health/full` — `checks.hierarchy`
       should show the lag for this conversation falling on later requests.
"""

import argparse
import asyncio
import importlib.util
import json
import os
import shutil
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
)
_REQUIRED_MEMORY_SYMBOLS = ("conv_lock",)


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


def _compactor_is_alive(url: str = DEFAULT_HEALTH_URL) -> bool:
    """True if anything answers `url`. Same liveness probe
    scripts/backfill-records.py uses — --apply's blast radius is a state
    file the live compactor's own rollup could be mid-write on."""
    try:
        with urllib.request.urlopen(url, timeout=HEALTH_PROBE_TIMEOUT_S) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False
    except Exception:
        return False


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
    by_id = {r[0]: {"id": r[0], "parent_id": r[1], "role": r[2], "content": r[3],
                    "created_at": r[4]} for r in rows}
    parent_ids = {r[1] for r in rows if r[1] is not None}
    leaves = [r for r in rows if r[0] not in parent_ids]
    if not leaves:
        leaves = rows
    leaf = max(leaves, key=lambda r: (r[4] if r[4] is not None else 0))
    chain: list[dict] = []
    seen: set[str] = set()
    node_id = leaf[0]
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


def reconstruct_transcript(
    con: sqlite3.Connection, chat_id: str
) -> tuple[list[dict], str, list[str]]:
    """(turns, source, notes). `turns` is `[{"role", "content"}, ...]`, the
    same shape the compactor receives. Raises ChatNotFound if no `chat`
    row matches `chat_id`; TranscriptUnavailable if neither source could
    produce anything."""
    notes: list[str] = []
    row = con.execute("SELECT chat FROM chat WHERE id = ?", (chat_id,)).fetchone()
    if row is None:
        raise ChatNotFound(chat_id)
    raw = row[0]

    chain = None
    try:
        data = json.loads(raw) if isinstance(raw, (str, bytes)) else None
        history = data.get("history") if isinstance(data, dict) else None
        if isinstance(history, dict):
            chain = _walk_branch_from_history(history)
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
    size a dry run; `--apply` runs the real drain, which is exact."""
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
                     help="override the live-database, live-compactor and "
                          "backup-collision refusals (see the module docstring)")
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

    conv_id = args.conv_id or args.chat_id
    model = args.model  # default handled below, only required for --apply

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

    l1_due, l2_due, l3_due = estimate_due(
        state, current_turns,
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
    }

    if not args.apply:
        report["note"] = (
            "DRY RUN — no writes, no vLLM calls. Re-run with --apply once "
            "ready." if work_due else
            "DRY RUN — nothing is due; --apply would do nothing."
        )
        report["safe_to_apply"] = True
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            _print_dry_run_report(report)
        return 3 if work_due else 0

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

    alive = _compactor_is_alive()
    if alive and not args.force:
        return _fatal(
            args,
            f"REFUSING --apply: something is answering {DEFAULT_HEALTH_URL} "
            f"— the live compactor may be writing this exact state file "
            f"right now. Stop it first (supervisorctl stop compactor) or "
            f"pass --force to override.",
        )
    if alive and args.force:
        warnings.append(
            f"WARNING: --force overriding a live compactor detected at "
            f"{DEFAULT_HEALTH_URL} — proceeding anyway."
        )

    summary_path = summarizer.summary_path(conv_id)
    backup_path = None
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

    passes = asyncio.run(_run_apply_loop(summarizer, memory, conv_id, turns,
                                          args.vllm_url, model, args.max_calls))

    final_state = summarizer.load_state(conv_id)
    report["backup"] = str(backup_path) if backup_path else None
    report["passes"] = passes["log"]
    report["rollup_calls"] = passes["rollup_calls"]
    report["vllm_calls_spent"] = passes["vllm_calls_spent"]
    report["budget_exhausted"] = passes["exhausted"]
    report["stopped_because"] = passes["stopped_because"]
    report["last_summarized_turn_after"] = final_state.get("last_summarized_turn", 0)
    report["turns_seen_after"] = final_state.get("turns_seen", 0)
    report["still_due"] = summarizer.needs_rollup(
        final_state, summarizer.recorded_position(final_state)
    )
    report["note"] = (
        "apply complete" if not report["still_due"]
        else "apply stopped with more work due — re-run to continue"
    )

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _print_apply_report(report)
    return 0


async def _run_apply_loop(summarizer, memory, conv_id, messages, vllm_url, model, max_calls):
    """The real drain: mirrors main.admin_compact's own loop exactly (same
    module, same function, different transcript). Progress is persisted by
    `maybe_rollup` itself after every unit — this loop holds no state of
    its own that a crash could lose."""
    log: list[dict] = []

    # v3.1.7 R10, applied here for the same reason main.admin_compact
    # applies it before ITS drain: a stale tail_fp anchor from the live
    # chat path may not match this reconstruction's own tail, which would
    # misalign the first chunk's offset. Dropped once, under conv_lock;
    # every later pass re-derives its own anchor from this same array.
    async with memory.conv_lock(conv_id):
        _state = summarizer.load_state(conv_id)
        if _state.get("tail_fp"):
            _state["tail_fp"] = []
            _state["head_fp"] = ""
            _state["window_turns"] = 0
            summarizer.save_state(conv_id, _state)

    calls = 0
    stopped_because = None
    with summarizer.vllm_call_budget_ctx(max_calls) as budget:
        while calls < max_calls:
            prev = summarizer.load_state(conv_id).get("last_summarized_turn", 0)
            try:
                await summarizer.maybe_rollup(conv_id, messages, vllm_url, model)
            except Exception as e:
                stopped_because = f"{type(e).__name__}: {e}"
                break
            calls += 1
            now = summarizer.load_state(conv_id).get("last_summarized_turn", 0)
            log.append({"pass": calls, "last_summarized_turn": now})
            if now <= prev:
                stopped_because = "the watermark stopped advancing"
                break
            if budget["remaining"] <= 0:
                stopped_because = f"hit max_calls={max_calls} (vLLM calls)"
                break
        else:
            stopped_because = f"hit max_calls={max_calls} (rollup passes)"
        vllm_calls_spent = max_calls - budget["remaining"]
        exhausted = budget["exhausted"]

    return {
        "log": log,
        "rollup_calls": calls,
        "vllm_calls_spent": vllm_calls_spent,
        "exhausted": exhausted,
        "stopped_because": stopped_because,
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
    for p in report["passes"]:
        print(f"  pass {p['pass']}: last_summarized_turn -> {p['last_summarized_turn']}")
    print(f"rollup passes:    {report['rollup_calls']}")
    print(f"vLLM calls spent: {report['vllm_calls_spent']}")
    print(f"budget exhausted: {report['budget_exhausted']}")
    print(f"stopped because:  {report['stopped_because']}")
    print(f"watermark after:  last_summarized_turn={report['last_summarized_turn_after']} "
          f"turns_seen={report['turns_seen_after']}")
    print(f"still due:        {report['still_due']}")
    for w in report["warnings"]:
        print(w)
    print()
    print(report["note"])


if __name__ == "__main__":
    sys.exit(main())
