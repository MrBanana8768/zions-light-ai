#!/usr/bin/env python3
"""Inspect compactor backfill records and, only on request, close the
stale ones before an upgrade resumes them.

SUPPORTED INVOCATION — from a clone, not a copy (see "PACKAGE RESOLUTION"
below for why a copy to /data/scripts/ fails on a pre-v3.1.9.4 pod):

    git clone --depth 1 --branch <tag-or-branch> \\
        https://github.com/MrBanana8768/zions-light-ai.git /opt/zl-repo
    /opt/compactor-venv/bin/python /opt/zl-repo/scripts/backfill-records.py \\
        --store /data/openwebui/compactor
    /opt/compactor-venv/bin/python /opt/zl-repo/scripts/backfill-records.py \\
        --store /data/openwebui/compactor --json
    /opt/compactor-venv/bin/python /opt/zl-repo/scripts/backfill-records.py \\
        --store /data/openwebui/compactor --apply
    /opt/compactor-venv/bin/python /opt/zl-repo/scripts/backfill-records.py \\
        --store /data/openwebui/compactor --conv <id> --apply

`--compactor-pkg PATH` overrides package auto-detection entirely, if you
need to point this at a package that isn't beside the script and isn't at
/opt/compactor (see "PACKAGE RESOLUTION").

PACKAGE RESOLUTION. Copying just this one file to /data/scripts/ and
running it from there used to fail with "no compactor package beside this
script (/data/compactor)" — HERE.parent/"compactor" resolves relative to
wherever THIS FILE sits, and /data/scripts/../compactor does not exist.
Resolved now, in order: `--compactor-pkg PATH` (explicit, highest
precedence — and now validated: a `--compactor-pkg` that is not a
directory is a fatal error, never a silent fall-through to the next
source, see H4) > `HERE.parent/"compactor"` (the repo/clone layout — the
SUPPORTED way, see above) > `/opt/compactor` (the image's own layout) > an
already-importable `backfill` on sys.path. If every one of those fails,
the error lists every path tried and prints the clone command above. See
_resolve_compactor_pkg's own docstring for the full reasoning, including a
layout trap that can make a substitute package silently invisible.

PROVENANCE, NOT JUST A CHOSEN DIRECTORY (H4). Resolving a directory is not
the same as importing from it — something else on `sys.path` (most often a
stray `PYTHONPATH`) can shadow it: `sys.path.insert(0, pkg_dir)` puts the
chosen directory first, but if `pkg_dir` itself has no `backfill.py` (an
empty or wrong `--compactor-pkg`), Python's import machinery keeps
searching and can silently succeed against a DIFFERENT `backfill` module
further down `sys.path` — while this script's own NOTE still names the
directory it *chose*, not the module it actually *got*. After importing,
this script asserts `Path(backfill.__file__).resolve().parent` equals the
resolved package directory and refuses, naming the real `__file__`, if
they disagree — see main()'s provenance check, right after `import
backfill`.

WHY THIS DOES NOT ALSO WORK BY JUST COPYING NEWER FILES IN. On a pod that
has never run v3.1.9.4 or later, the INSTALLED `compactor/backfill.py`
itself lacks `_MAX_BACKFILL_ATTEMPTS` and `_backoff_ready` — this script
asks the real module for its verdict rather than keeping a second copy of
the rules (see below), and that pre-v3.1.9.4 module simply does not have
them yet. Substituting a newer `backfill.py` onto that older package does
not work either (verified): v3.1.9.4's `backfill.py` imports
`current_wipe_generation` from `memory`, which the older `memory.py` does
not have. This script detects the missing capability before classifying
anything and refuses with an actionable message rather than an
`AttributeError` — see _check_backfill_capability. The only self-consistent
fix is the WHOLE target release together, which is exactly what a clone
supplies, and is why the clone above is the only invocation that works on
such a pod.

WHY THIS EXISTS. `compactor/backfill.py`'s `needs_backfill()` decides "does
this conversation need a lazy history backfill?" by reading the
`facts/<conv>.backfill.json` RECORD first, as of v3.1.9.4 — not by checking
whether a facts file already exists, which is what every release up to
v3.1.9.3 did (see that function's own docstring, and RUNPOD_DEPLOY.md's
"Upgrading within v3.1.9.x, and rolling back"). That fix is correct — a
stale `in_progress` or `failed` record on a conversation that also has
live facts used to be ignored forever, silently losing that conversation's
pre-live history for good — but it also means upgrading a pod that has
EVER run v3.1.9.3 or earlier RESUMES every stale record already sitting on
the volume, the moment each conversation's next eligible request arrives.
On a real 2026-09-22 backup this is not hypothetical: four conversations
carry a stale `in_progress` record (6/1908, 296/793, 572/732 and 589/626
exchanges respectively). Resuming all four at once means roughly 2,600
background vLLM extraction calls competing with her live chat on the pod's
one GPU.

WHAT THIS DOES. Reads every `facts/*.backfill.json` record under `--store`
(or just the ones named by `--conv`, repeatable) and classifies each one:

  - `leave`         — already terminal (`complete` / `abandoned` / `wiped`).
                       `needs_backfill()` never retries these.
  - `would-resume`  — `needs_backfill()` will restart this the next time
                       this conversation is used: a stale `in_progress`
                       record, or a `failed` one whose backoff has elapsed.
  - `needs-review`  — ambiguous: an `in_progress` record that is not yet
                       stale (may be genuinely running right now), a
                       `failed` record still inside its backoff window, a
                       record already at the attempt cap (which
                       `needs_backfill()` already refuses on its own), or
                       a record this script could not read.

The verdict comes from the REAL `compactor/backfill.py` module beside this
script (`is_stale`, `_backoff_ready`, `_MAX_BACKFILL_ATTEMPTS`) — this file
does not keep its own copy of that logic. Two copies of one rule drifting
apart is this codebase's most expensive recurring defect (see
`scripts/merge-conversations.py`'s docstring for the same reasoning, applied
there to `portability.merge_conversation`).

Dry run (the default) only reports; nothing is written under any
circumstance. `--apply` rewrites the `state` of every `would-resume` record
THAT ALSO HAS A FACTS FILE to `"abandoned"` — the exact terminal state
`_run_backfill` itself writes once a record has spent its retries — so the
upgraded compactor's own `needs_backfill()` leaves it alone from then on.
Every other field on the record is kept; `closed_by` and `closed_at` are
added. A `--apply` run prints exactly what it changed, one line per record.

WHAT THIS WILL NOT DO. It never touches `facts/<conv>.json`,
`facts/<conv>.archive.json`, `summaries/`, `chromadb/` or `personas/` — its
entire blast radius is the `facts/*.backfill.json` sidecars, and even there
it only ever changes `state`, `closed_by` and `closed_at` on records already
classified `would-resume`. It never closes a `would-resume` record whose
conversation has NO facts file at all without `--force`: that conversation
has never had a fact extracted, so closing its only open backfill attempt
would cancel its one chance at ever getting one, not close a hazard.
`--force` also overrides the refusal to run `--apply` while something
answers the compactor's `/health` — see `_compactor_is_alive` below.

EXIT CODES. The same 5-value convention as scripts/import-history.py and
scripts/setup-sshd.py — stated ONCE, canonically, in OPERATIONS.md's "Exit
codes — the shared convention across the operator scripts" (see that
section for the full rationale; setup-sshd.py used to have 0 and 3 swapped
from this). This script's own mapping onto that table:
    0   the desired end state is in place: a dry run that found nothing
        `would-resume`, or an `--apply` that closed every `would-resume`
        record it was asked to (including a no-op `--apply` because there
        was nothing to close). A `--force`-gated skip is not an error
        either: it is this script refusing to do something unsafe, not a
        failure — but see H1 below for what an unsafe *outcome* still does.
    1   a refusal or an error the operator needs to look at: a bad
        `--store`, no usable compactor package (or one too old, or one
        whose provenance this script could not verify — see PACKAGE
        RESOLUTION above), an explicit `--compactor-pkg` that is not a
        directory, a refused `--apply` precondition without `--force`, a
        backup path that already existed, EVERY `--conv` value given being
        rejected (unsafe or no record found — nothing was inspected), ANY
        record classified `needs-review` (ambiguous — a human needs to
        read why, see the per-record reason), or an `--apply` that closed
        NONE of the `would-resume` records it targeted (H1: this is
        indistinguishable from doing nothing, and doing nothing is exactly
        the upgrade-time hazard this script exists to prevent — it must
        never look the same as success)
    3   a DRY RUN found one or more `would-resume` records and nothing
        else needs attention — informational, not a failure: it means
        "re-run with --apply once you're ready"
    4   `--apply` closed at least one `would-resume` record but at least
        one other one is still open (H1: refused-no-facts,
        refused-backup-exists, or a write failure) — progress was made,
        but the upgrade hazard is not fully defused; look at what remains
        refused, then re-run (with `--force` if that is what the
        remaining refusal calls for)
    (argparse's own usage errors — unknown flags, missing required values —
    exit 2, the Python standard library's own convention, unrelated to the
    four above)

H1's rule, spelled out: after `--apply`, let TARGETED be the `would-resume`
records this run attempted to close and CLOSED be how many it actually
closed. `--apply` exits 0 if TARGETED == CLOSED (everything closed, or
there was nothing to close), 4 if 0 < CLOSED < TARGETED (some progress,
some still open), and 1 if CLOSED == 0 < TARGETED (no progress at all —
the exact "exits 0 having closed nothing" defect this replaces). The
needs-review and all-`--conv`-rejected rules above apply to a dry run too,
and take priority over exit 3: a run that needs a human's eyes must never
report the same code as one that is merely "safe to --apply now".
"""

import argparse
import errno
import importlib.util
import json
import shutil
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
# A module-level name (not inlined at the call site) so a test can
# monkeypatch it to a temp directory, the same reason HERE itself is a
# module global rather than computed fresh inside every function.
OPT_COMPACTOR = Path("/opt/compactor")

DEFAULT_STORE = "/data/openwebui/compactor"
# M3: a module-level name, like HERE/OPT_COMPACTOR above — main() reassigns
# it from --health-url before any liveness probe runs, and
# _compactor_is_alive() reads it fresh rather than via a default-parameter
# value (which would freeze it at function-definition time, before argparse
# has even run).
DEFAULT_HEALTH_URL = "http://127.0.0.1:8080/health"
HEALTH_PROBE_TIMEOUT_S = 3

TERMINAL_STATES = ("complete", "abandoned", "wiped")

# The one invocation verified end to end against a pre-v3.1.9.4 pod (see
# RUNPOD_DEPLOY.md "Upgrading within v3.1.9.x" and OPERATIONS.md). Printed
# whenever no compactor package could be found at all, and whenever the one
# found is too old to answer the questions this script asks it (see
# _check_backfill_capability) — in both cases a clone is the fix.
_CLONE_HINT = """\
On a pod, the supported way to run this script is from a clone of this
public repo (git ships in the image; there is no ssh/scp/rsync in it):

  git clone --depth 1 --branch <tag-or-branch> \\
      https://github.com/MrBanana8768/zions-light-ai.git /opt/zl-repo
  /opt/compactor-venv/bin/python /opt/zl-repo/scripts/backfill-records.py \\
      --store /data/openwebui/compactor

A clone always carries a self-consistent target-release `compactor`
package beside this script, which is the one thing a pre-v3.1.9.4 pod's
own installed package cannot supply for itself."""


# ---------------------------------------------------------------------------
# Where the real compactor/backfill.py lives — resolved fresh every run,
# never assumed (Defect 1). Copied to /data/scripts/, HERE.parent/"compactor"
# resolves to /data/compactor, which does not exist: that is the exact
# operator error this function exists to stop happening again.
# ---------------------------------------------------------------------------

# The module this script actually needs from the package, used both to
# decide whether "already importable on sys.path" (source 4) is true, and
# to import it for real once a directory is chosen.
_PROBE_MODULE = "backfill"


def _resolve_compactor_pkg(explicit: str | None) -> tuple[Path | None, list[str], bool]:
    """(pkg_dir, tried, already_importable).

    An explicit `--compactor-pkg` that is NOT a directory is validated by
    the CALLER (main()) before this function ever runs, and is a fatal
    error there (H4) — never a silent fall-through to sources 2-4. This
    function itself only ever sees an `explicit` that is either empty/None
    or already a real directory; the `p.is_dir()` check below is what
    main() itself uses to decide that.

    Tried in order, and the first hit wins:
      1. --compactor-pkg PATH — explicit always wins.
      2. HERE.parent / "compactor" — the repo/clone layout. THE SUPPORTED
         way to run this script (see the module docstring and _CLONE_HINT).
      3. /opt/compactor — the image's own layout, in case a copy of this
         script ends up sitting somewhere the image also mounts/keeps the
         package.
      4. an already-importable `backfill` on sys.path (e.g. PYTHONPATH
         already points at a package, or a test harness pre-imported one).

    `pkg_dir` is None with `already_importable=True` for source 4 — there
    is no directory to insert into sys.path, the caller just imports.
    `pkg_dir` is None with `already_importable=False` if every source
    failed; `tried` then lists every path (and the sys.path probe) in the
    order they were checked, for the caller's error message.

    THE TRAP (see the module docstring): a script living directly at
    `/opt/sim/x.py` (not nested under a `scripts/` directory) has
    HERE.parent == /opt, so source 2 resolves to /opt/compactor — the
    SAME path as source 3. A substituted package placed at
    /opt/sim/compactor (sibling of the script itself, not of its parent)
    is never tried at all, and a test built that way would "pass" against
    the image's own /opt/compactor without ever exercising source 2 or 3
    as distinct candidates. Exercising them as distinct requires nesting
    the probe script one level deeper, e.g. /opt/sim/scripts/x.py.
    """
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
# Small pure helpers
# ---------------------------------------------------------------------------

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _utc_stamp() -> str:
    """UTC timestamp used to name a record's backup file. A separate
    function (not inlined at the call site) so a test can monkeypatch it
    for a deterministic collision check on the refusal in `_close_record`.
    """
    return _now_utc().strftime("%Y%m%dT%H%M%SZ")


def _parse_iso(ts) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None


def _format_age(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.0f}m"
    hours = minutes / 60
    if hours < 48:
        return f"{hours:.1f}h"
    return f"{hours / 24:.1f}d"


def _safe_conv_id(conv_id: str) -> bool:
    """Reject a `--conv` value that could point outside facts/ before it is
    ever turned into a path. Operator input, not attacker input — this is a
    guard against a typo (a stray `/` or `..`) rather than a hardened
    boundary like memory.py's own `_safe_path`, which this script does not
    import (see the module docstring: no dependency on `memory`'s
    import-time-cached `STORAGE_ROOT` is deliberate).
    """
    return bool(conv_id) and "/" not in conv_id and "\\" not in conv_id and conv_id not in (".", "..")


def _load_record(path: Path) -> tuple[dict | None, str | None]:
    """Return (record, error). `record` is None if the file could not be
    read as a JSON object — reported by the caller, never guessed at or
    rewritten.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as e:
        return None, f"{type(e).__name__}: {e}"
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        return None, f"{type(e).__name__}: {e}"
    if not isinstance(data, dict):
        return None, f"expected a JSON object, found {type(data).__name__}"
    return data, None


def _compactor_is_alive(url: str | None = None) -> bool:
    """False ONLY for an unambiguous connection refusal at `url` — the one
    signal that unambiguously proves nothing is listening there at all.
    Reads `DEFAULT_HEALTH_URL` fresh (a module global, like `HERE` and
    `OPT_COMPACTOR` above — reassigned by main() from `--health-url`
    rather than passed as a default-parameter value, so the value is live
    at CALL time, not baked in at function-definition time) when `url` is
    not given.

    M3 (fails open, before this fix): a bare `except: return False` made a
    timeout indistinguishable from "nothing is listening" — a compactor
    that is merely slow to answer under GPU load, or not yet bound during
    boot, read exactly the same as one that had genuinely stopped, and
    `--apply` proceeded either way. Every outcome OTHER than a confirmed
    `ECONNREFUSED` is now treated as "might be alive" and refuses --apply
    the same way a confirmed-live compactor does: a real 200, a non-200
    HTTP response (something else answered — still not proof the compactor
    itself is down), a timeout, a DNS failure, or any other OSError. Only
    "nothing is listening on this port at all" is safe to read as "not
    running".
    """
    if url is None:
        url = DEFAULT_HEALTH_URL
    try:
        with urllib.request.urlopen(url, timeout=HEALTH_PROBE_TIMEOUT_S) as r:
            r.read()
        return True
    except urllib.error.HTTPError:
        # A real HTTP response with a non-200 status. Something IS bound
        # to this port and answered — just not the compactor's own
        # /health, which is always 200 when it is genuinely up. Not a
        # connection refusal, so not "not-running": ambiguous, refuse.
        return True
    except urllib.error.URLError as e:
        reason = e.reason
        if isinstance(reason, ConnectionRefusedError) or (
            isinstance(reason, OSError)
            and getattr(reason, "errno", None) == errno.ECONNREFUSED
        ):
            return False
        return True
    except ConnectionRefusedError:
        return False
    except OSError as e:
        if getattr(e, "errno", None) == errno.ECONNREFUSED:
            return False
        return True
    except Exception:
        return True


# ---------------------------------------------------------------------------
# Capability check (Defect 2) — this script is a PRE-upgrade step that
# learns its rules from the INSTALLED package, which on a pre-v3.1.9.4 pod
# is exactly the version that does not have them yet. `_classify` below
# reads `backfill_mod._MAX_BACKFILL_ATTEMPTS`, calls
# `backfill_mod._backoff_ready` AND `backfill_mod.is_stale`, and
# `_close_record` calls `backfill_mod.atomic_write_json` — EVERY symbol this
# script asks the module for, not just the two that happen to be new in
# v3.1.9.4 (M1: `is_stale`/`atomic_write_json` predate v3.1.9.4 and so were
# wrongly assumed always-present; a package missing WHOLESALE, or a
# fabricated stand-in missing just those two, would otherwise reach
# `_classify`/`_close_record` and crash with a raw `AttributeError` — see
# below for why that crash, specifically inside `_close_record`, is doubly
# dangerous). Detected BEFORE classifying or closing anything, so the
# failure is one actionable message naming the pod's version state, never a
# raw AttributeError.
#
# WHY THIS MUST RUN BEFORE ANY BACKUP (M1). `_close_record` calls
# `shutil.copy2` (the backup) BEFORE it calls `atomic_write_json` — if that
# second call were to raise (missing symbol, or any other error),
# `_close_record` would leave an ORPHAN `.bak-` file: not cleaned up, and
# permanently arming the backup-already-exists refusal on every later
# retry of that exact record. Requiring every symbol `_classify` AND
# `_close_record` use to be present before either ever runs is what makes
# that specific crash-after-backup window impossible for a missing-symbol
# failure; `_close_record` itself also removes its own backup file if
# `atomic_write_json` still somehow raises for an unrelated reason (disk
# full, permissions), as defense in depth — see its own docstring.
#
# This is NOT fixed by substituting a newer backfill.py onto an older
# package on disk — verified to fail differently: v3.1.9.4's backfill.py
# imports `current_wipe_generation` from `memory`, which v3.1.9/.1/.2/.3's
# `memory.py` does not have. The only self-consistent fix is the whole
# target release together, which is exactly what a clone supplies.
# ---------------------------------------------------------------------------

_REQUIRED_BACKFILL_SYMBOLS = (
    "_MAX_BACKFILL_ATTEMPTS",
    "_backoff_ready",
    "is_stale",
    "atomic_write_json",
)


def _check_backfill_capability(backfill_mod, pkg_source: str) -> str | None:
    """None if `backfill_mod` has everything `_classify` needs; otherwise
    the full actionable error message (never a bare AttributeError)."""
    missing = [n for n in _REQUIRED_BACKFILL_SYMBOLS if not hasattr(backfill_mod, n)]
    if not missing:
        return None
    return (
        f"ERROR: the compactor package at {pkg_source} is missing "
        f"{', '.join(missing)} — this pod's installed compactor predates "
        f"v3.1.9.4 (backfill.needs_backfill() gained the record-first "
        f"rules, and this script asks the REAL module for its verdict "
        f"rather than keeping a second copy of the logic — see this "
        f"script's own module docstring). Classifying records against a "
        f"package this old would misjudge every one of them, silently.\n\n"
        f"{_CLONE_HINT}"
    )


# ---------------------------------------------------------------------------
# Classification — asks the REAL backfill.py module, never a copy of it
# ---------------------------------------------------------------------------

def _classify(record: dict, backfill_mod) -> tuple[str, str]:
    """Classify one on-disk backfill record the way `backfill.needs_backfill`
    would react to it on the first eligible request after an upgrade.

    This skips the two gates `needs_backfill` applies FIRST against live
    message history (`len(messages) < _MIN_MESSAGES_FOR_BACKFILL`, and
    `facts_module.extraction_enabled()`) — this script has no message
    history for any of these conversations, only their records, and both
    gates can only turn a `would-resume` record into a safe one, never the
    reverse. A record this classifies `would-resume` is therefore never a
    false negative; it can be a false positive only for a conversation that
    is currently too short or has extraction disabled, in which case
    closing it anyway costs nothing (there was nothing to resume).

    This also does not reproduce `needs_backfill`'s `_facts_tombstoned`
    defense-in-depth check (a wipe whose facts file reads empty AND whose
    archive sidecar is also empty). That check exists only for a wipe that
    predates the `mark_wiped` fix (v3.1.9.4 R5) or used a wipe path that
    never called it — every current wipe path calls `mark_wiped`, which
    writes `state: "wiped"` directly, so a genuinely wiped conversation's
    record already reads `"wiped"` and is classified `leave` below without
    needing it. Reproducing it here would also mean importing `facts.py`
    and depending on `memory.STORAGE_ROOT`, which this script deliberately
    does not (see the module docstring).
    """
    state = record.get("state")
    if state in TERMINAL_STATES:
        return "leave", f"terminal state {state!r} — needs_backfill() never retries this"

    max_attempts = backfill_mod._MAX_BACKFILL_ATTEMPTS

    if state == "in_progress":
        if not backfill_mod.is_stale(record):
            return (
                "needs-review",
                "in_progress and not yet stale — may be a backfill genuinely "
                "running right now; re-check rather than close it",
            )
        attempts = int(record.get("attempts") or 0)
        if attempts >= max_attempts:
            return (
                "needs-review",
                f"stale in_progress already at the {max_attempts}-attempt "
                f"cap — needs_backfill() already refuses to resume this on "
                f"its own; closing it is optional tidiness, not required "
                f"to defuse the upgrade hazard",
            )
        return (
            "would-resume",
            f"stale in_progress ({record.get('exchanges_done', 0)}/"
            f"{record.get('exchanges_total', 0)} exchanges, attempt "
            f"{attempts or 1}) — needs_backfill() resumes this on the next "
            f"eligible request",
        )

    if state == "failed":
        attempts = int(record.get("attempts") or 0)
        if attempts >= max_attempts:
            return (
                "needs-review",
                f"failed at the {max_attempts}-attempt cap — should have "
                f"been written as 'abandoned' by _run_backfill; "
                f"needs_backfill() already refuses to resume this",
            )
        if not backfill_mod._backoff_ready(record):
            return (
                "needs-review",
                "failed, but its retry backoff has not elapsed yet — not "
                "due to retry right now; re-check later",
            )
        return (
            "would-resume",
            f"failed (attempt {attempts or 1}), backoff elapsed — "
            f"needs_backfill() resumes this on the next eligible request",
        )

    return (
        "needs-review",
        f"unrecognised or missing state {state!r} — needs_backfill() falls "
        f"through to the facts-file check for this; left alone rather than "
        f"guessed at",
    )


# ---------------------------------------------------------------------------
# --apply: closing a record
# ---------------------------------------------------------------------------

def _close_record(backfill_mod, record_path: Path, record: dict) -> tuple[bool, str, str]:
    """Rewrite `record_path`'s state to "abandoned", after backing up the
    original. Returns (ok, action, message). Never called for a record
    whose verdict is not `would-resume` — see `_apply`.

    M1: the backup (`shutil.copy2`) happens BEFORE `atomic_write_json` is
    called, which means a failure in that second call would otherwise
    leave an ORPHAN `.bak-` file — never cleaned up, and permanently
    arming the backup-already-exists refusal above on every later retry
    of this exact record. The capability check in main() (see
    `_REQUIRED_BACKFILL_SYMBOLS`) already stops a MISSING `atomic_write_json`
    from ever reaching this far; this `try`/`except` is the second line of
    defense, for anything else that could make the real call raise (disk
    full, a permissions change mid-run) — the backup this run made is
    removed before returning, so no failure downstream of the backup can
    ever leave one behind.
    """
    stamp = _utc_stamp()
    backup_path = record_path.with_name(record_path.name + f".bak-{stamp}")
    if backup_path.exists():
        return False, "refused-backup-exists", f"refused — backup path already exists: {backup_path.name}"
    shutil.copy2(record_path, backup_path)
    updated = dict(record)
    updated["state"] = "abandoned"
    updated["closed_by"] = "scripts/backfill-records.py"
    updated["closed_at"] = _now_utc().isoformat(timespec="seconds")
    try:
        # atomic_write_json is imported (via `backfill`, which imports it
        # from `memory`) rather than reimplemented: same
        # temp-file-in-same-dir + fsync + os.replace this codebase already
        # relies on everywhere else a sidecar like this one gets written.
        backfill_mod.atomic_write_json(record_path, updated)
    except Exception as e:
        try:
            backup_path.unlink()
        except OSError:
            pass
        return (
            False,
            "refused-write-failed",
            f"refused — failed writing the closed record ({type(e).__name__}: "
            f"{e}); no backup left behind",
        )
    return True, "closed", f"closed -> state=abandoned (backup: {backup_path.name})"


def _apply(args, backfill_mod, rows: list[dict], raw_records: dict[str, dict]) -> tuple[list[dict], bool]:
    """Close every `would-resume` row that has a facts file (or every one,
    with `--force`). Returns (changes, had_error).
    """
    changes = []
    had_error = False
    for row in rows:
        if row["verdict"] != "would-resume":
            continue
        conv_id = row["conv_id"]
        record_path = Path(row["path"])
        if not row["facts_exists"] and not args.force:
            changes.append({
                "conv_id": conv_id,
                "action": "refused-no-facts",
                "detail": (
                    "would-resume but no facts file exists for this "
                    "conversation — closing it would cancel its only "
                    "chance at a first backfill; rerun with --force to "
                    "override"
                ),
            })
            continue
        ok, action, detail = _close_record(backfill_mod, record_path, raw_records[conv_id])
        changes.append({"conv_id": conv_id, "action": action, "detail": detail})
        if not ok:
            had_error = True
    return changes, had_error


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_report(store_root: Path, rows: list[dict], counts: dict, changes: list[dict] | None) -> None:
    print("=" * 62)
    print(f"backfill records — {store_root}")
    print("=" * 62)
    if not rows:
        print("(no backfill records found)")
    for r in rows:
        done = r["exchanges_done"]
        total = r["exchanges_total"]
        progress = f"{done}/{total}" if done is not None else "?/?"
        attempts = r["attempts"] if r["attempts"] is not None else "?"
        facts = "yes" if r["facts_exists"] else "no"
        print(
            f"{r['conv_id']}  state={str(r['state']):<11} {progress:<10} "
            f"attempts={attempts} age={r['age']:>7} facts={facts:<3} "
            f"-> {r['verdict']}"
        )
        print(f"    {r['reason']}")
    print()
    print(
        f"{len(rows)} record(s): {counts['would-resume']} would-resume, "
        f"{counts['leave']} leave, {counts['needs-review']} needs-review"
    )

    if changes is not None:
        print()
        print("Changes:")
        if not changes:
            print("  (nothing to close)")
        for c in changes:
            print(f"  {c['conv_id']}: {c['action']} — {c['detail']}")
    elif counts["would-resume"] > 0:
        print()
        print(
            f"DRY RUN — nothing was written. {counts['would-resume']} "
            f"record(s) would be resumed by v3.1.9.4+; re-run with --apply "
            f"to close them."
        )
    else:
        print()
        print("DRY RUN — nothing was written, and nothing would resume.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="backfill-records.py",
        description=(
            "Inspect compactor backfill records and, only with --apply, "
            "close the ones v3.1.9.4+ would otherwise resume on upgrade."
        ),
    )
    ap.add_argument(
        "--store", default=DEFAULT_STORE,
        help=f"compactor storage root (default: {DEFAULT_STORE})",
    )
    ap.add_argument(
        "--conv", dest="conv_ids", action="append", metavar="ID",
        help="limit to this conv_id (repeatable; default: every backfill "
             "record found under --store)",
    )
    ap.add_argument(
        "--apply", action="store_true",
        help="close would-resume records that have a facts file. Without "
             "this it is a dry run: nothing is ever written.",
    )
    ap.add_argument(
        "--json", action="store_true",
        help="machine-readable output",
    )
    ap.add_argument(
        "--force", action="store_true",
        help="override the live-compactor refusal and the no-facts-file "
             "refusal (see the module docstring)",
    )
    ap.add_argument(
        "--compactor-pkg", default=None, metavar="PATH",
        help="explicit path to the compactor package directory (highest "
             "precedence; overrides the repo/clone-layout and /opt/compactor "
             "auto-detection — see the module docstring's resolution order). "
             "Must be a real directory: refused outright otherwise (H4), "
             "never silently skipped in favor of auto-detection.",
    )
    ap.add_argument(
        "--health-url", default=DEFAULT_HEALTH_URL, metavar="URL",
        help=f"compactor health endpoint probed before --apply (default: "
             f"{DEFAULT_HEALTH_URL}). Only an unambiguous connection "
             f"refusal at this URL is treated as 'not running'; anything "
             f"else (a real response, a timeout, a DNS failure) refuses "
             f"--apply the same way a confirmed-live compactor does — see "
             f"the module docstring, M3.",
    )
    return ap


def _fatal(args, message: str) -> int:
    if args.json:
        print(json.dumps({"error": message}, indent=2))
    else:
        print(message)
    return 1


def main(argv=None) -> int:
    global DEFAULT_HEALTH_URL
    args = _build_argparser().parse_args(argv)
    # M3: reassigned BEFORE any liveness probe can run, so
    # _compactor_is_alive()'s fresh read of the global picks up
    # --health-url (see that function's own docstring for why it is read
    # fresh rather than bound as a default-parameter value).
    DEFAULT_HEALTH_URL = args.health_url

    store_root = Path(args.store)
    facts_dir = store_root / "facts"

    # H4: an explicit --compactor-pkg that is not a real directory is a
    # fatal error here, BEFORE _resolve_compactor_pkg ever runs — never a
    # silent fall-through to the repo-layout/opt-compactor/sys.path
    # sources below it. The docstring's own claim ("explicit always wins")
    # is meaningless if "wins" can mean "is quietly ignored".
    if args.compactor_pkg is not None and not Path(args.compactor_pkg).is_dir():
        return _fatal(
            args,
            f"ERROR: --compactor-pkg {args.compactor_pkg} is not a "
            f"directory. An explicit --compactor-pkg is never skipped in "
            f"favor of auto-detection (see the module docstring) — fix the "
            f"path or drop the flag.",
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

    try:
        import backfill  # noqa: E402
    except Exception as e:
        return _fatal(
            args,
            f"ERROR importing compactor/backfill.py from {pkg_source}: "
            f"{type(e).__name__}: {e}",
        )

    # H4: pkg_source above names the directory this script CHOSE — not
    # necessarily the module it actually GOT. Something else on sys.path
    # (most often a stray PYTHONPATH) can shadow the chosen directory if
    # that directory itself has no backfill.py (e.g. an empty or wrong
    # --compactor-pkg): Python's import machinery just keeps searching and
    # can silently succeed against a DIFFERENT backfill module further
    # down sys.path. Verified against the real __file__, every time.
    real_file = getattr(backfill, "__file__", None)
    if pkg_dir is not None:
        actual_parent = Path(real_file).resolve().parent if real_file else None
        expected_parent = pkg_dir.resolve()
        if actual_parent != expected_parent:
            return _fatal(
                args,
                f"ERROR: resolved the compactor package at {pkg_source}, "
                f"but the imported 'backfill' module actually loaded from "
                f"{real_file!r} (parent {actual_parent}) — something else "
                f"on sys.path (often a stray PYTHONPATH) is shadowing the "
                f"intended package. Fix PYTHONPATH, or point "
                f"--compactor-pkg at a directory that really contains "
                f"backfill.py.",
            )

    capability_error = _check_backfill_capability(backfill, pkg_source)
    if capability_error is not None:
        return _fatal(args, capability_error)

    if not facts_dir.is_dir():
        return _fatal(args, f"ERROR: {facts_dir} does not exist or is not a directory.")

    warnings: list[str] = [
        f"NOTE: compactor package resolved from {pkg_source} "
        f"(backfill.__file__={real_file!r})"
    ]
    all_conv_rejected = False
    if args.conv_ids:
        paths = []
        for cid in args.conv_ids:
            if not _safe_conv_id(cid):
                warnings.append(f"WARNING: skipping --conv {cid!r} — not a safe conv_id")
                continue
            p = facts_dir / f"{cid}.backfill.json"
            if not p.is_file():
                warnings.append(f"WARNING: no backfill record for conv_id {cid!r} at {p}")
                continue
            paths.append(p)
        if not paths:
            # M8: every --conv value given was rejected (unsafe, or no
            # record found) — nothing was actually inspected. This must
            # read as "look at this", not as "0 record(s), all clear".
            all_conv_rejected = True
            warnings.append(
                "ERROR: none of the given --conv id(s) resolved to a "
                "backfill record — nothing was inspected."
            )
    else:
        paths = sorted(facts_dir.glob("*.backfill.json"))

    rows: list[dict] = []
    raw_records: dict[str, dict] = {}
    for path in paths:
        conv_id = path.name[: -len(".backfill.json")]
        record, err = _load_record(path)
        facts_exists = (facts_dir / f"{conv_id}.json").is_file()
        if record is None:
            rows.append({
                "conv_id": conv_id, "state": None, "exchanges_done": None,
                "exchanges_total": None, "attempts": None,
                "age": "unknown", "age_seconds": None,
                "facts_exists": facts_exists, "verdict": "needs-review",
                "reason": f"could not read this record ({err}); left untouched",
                "path": str(path),
            })
            continue
        raw_records[conv_id] = record
        verdict, reason = _classify(record, backfill)
        ts = _parse_iso(record.get("updated_at")) or _parse_iso(record.get("started_at"))
        age_seconds = (_now_utc() - ts).total_seconds() if ts else None
        rows.append({
            "conv_id": conv_id,
            "state": record.get("state"),
            "exchanges_done": record.get("exchanges_done"),
            "exchanges_total": record.get("exchanges_total"),
            "attempts": record.get("attempts"),
            "age": _format_age(age_seconds),
            "age_seconds": age_seconds,
            "facts_exists": facts_exists,
            "verdict": verdict,
            "reason": reason,
            "path": str(path),
        })

    rows.sort(key=lambda r: r["conv_id"])
    counts = {"leave": 0, "would-resume": 0, "needs-review": 0}
    for r in rows:
        counts[r["verdict"]] += 1

    changes = None
    apply_had_error = False
    if args.apply:
        alive = _compactor_is_alive()
        if alive and not args.force:
            return _fatal(
                args,
                f"REFUSING --apply: {args.health_url} did not give an "
                "unambiguous 'nothing is listening' signal (a real "
                "response, a non-200 HTTP status, or the probe itself "
                "timed out or failed all refuse the same way — see M3) — "
                "a live compactor may be writing these exact backfill "
                "records right now. Stop it first (supervisorctl stop "
                "compactor) or pass --force to override.",
            )
        if alive and args.force:
            warnings.append(
                f"WARNING: --force overriding an ambiguous or live-looking "
                f"probe of {args.health_url} — proceeding anyway."
            )
        changes, apply_had_error = _apply(args, backfill, rows, raw_records)

    if args.json:
        payload = {
            "store": str(store_root),
            "generated_at": _now_utc().isoformat(timespec="seconds"),
            "apply": args.apply,
            "warnings": warnings,
            "records": rows,
            "counts": counts,
        }
        if changes is not None:
            payload["changes"] = changes
        print(json.dumps(payload, indent=2))
    else:
        for w in warnings:
            print(w)
        _print_report(store_root, rows, counts, changes)

    # EXIT CODES — see the module docstring's table and H1's rule spelled
    # out there. all_conv_rejected and needs-review both mean "a human
    # needs to look at this", and take priority over everything else.
    if all_conv_rejected:
        return 1
    if args.apply:
        targeted = sum(1 for r in rows if r["verdict"] == "would-resume")
        closed = sum(1 for c in changes if c["action"] == "closed") if changes else 0
        remaining_open = targeted - closed
        if remaining_open == 0:
            return 0
        return 4 if closed > 0 else 1
    if counts["needs-review"] > 0:
        return 1
    return 3 if counts["would-resume"] > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
