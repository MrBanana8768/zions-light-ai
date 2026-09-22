#!/usr/bin/env python3
"""Inspect compactor backfill records and, only on request, close the
stale ones before an upgrade resumes them.

    /opt/compactor-venv/bin/python /data/scripts/backfill-records.py
    /opt/compactor-venv/bin/python /data/scripts/backfill-records.py --json
    /opt/compactor-venv/bin/python /data/scripts/backfill-records.py --apply
    /opt/compactor-venv/bin/python /data/scripts/backfill-records.py --conv <id> --apply

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

EXIT CODES
    0   nothing to do (no `would-resume` records found), or `--apply`
        completed without a hard error (a `--force`-gated skip is not an
        error: it is this script refusing to do something unsafe)
    1   an error the operator needs to look at: a bad `--store`, an
        unreadable compactor package, a refused `--apply` precondition
        without `--force`, or a backup path that already existed
    3   a DRY RUN found one or more `would-resume` records — informational,
        not a failure: it means "re-run with --apply once you're ready"
    (argparse's own usage errors — unknown flags, missing required values —
    exit 2, the Python standard library's own convention, unrelated to the
    three above)
"""

import argparse
import json
import shutil
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent / "compactor"

DEFAULT_STORE = "/data/openwebui/compactor"
DEFAULT_HEALTH_URL = "http://127.0.0.1:8080/health"
HEALTH_PROBE_TIMEOUT_S = 3

TERMINAL_STATES = ("complete", "abandoned", "wiped")


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


def _compactor_is_alive(url: str = DEFAULT_HEALTH_URL) -> bool:
    """True if anything answers `url`. The same liveness probe
    scripts/merge-conversations.py uses against its own port, pointed at
    the plain liveness endpoint here — --apply's blast radius is a sidecar
    file a live backfill run could be mid-write on, not something that
    needs the full /health/full diagnostic.
    """
    try:
        with urllib.request.urlopen(url, timeout=HEALTH_PROBE_TIMEOUT_S) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False
    except Exception:
        return False


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

def _close_record(backfill_mod, record_path: Path, record: dict) -> tuple[bool, str]:
    """Rewrite `record_path`'s state to "abandoned", after backing up the
    original. Returns (ok, message). Never called for a record whose
    verdict is not `would-resume` — see `_apply`.
    """
    stamp = _utc_stamp()
    backup_path = record_path.with_name(record_path.name + f".bak-{stamp}")
    if backup_path.exists():
        return False, f"refused — backup path already exists: {backup_path.name}"
    shutil.copy2(record_path, backup_path)
    updated = dict(record)
    updated["state"] = "abandoned"
    updated["closed_by"] = "scripts/backfill-records.py"
    updated["closed_at"] = _now_utc().isoformat(timespec="seconds")
    # atomic_write_json is imported (via `backfill`, which imports it from
    # `memory`) rather than reimplemented: same temp-file-in-same-dir +
    # fsync + os.replace this codebase already relies on everywhere else a
    # sidecar like this one gets written.
    backfill_mod.atomic_write_json(record_path, updated)
    return True, f"closed -> state=abandoned (backup: {backup_path.name})"


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
        ok, detail = _close_record(backfill_mod, record_path, raw_records[conv_id])
        changes.append({
            "conv_id": conv_id,
            "action": "closed" if ok else "refused-backup-exists",
            "detail": detail,
        })
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
    return ap


def _fatal(args, message: str) -> int:
    if args.json:
        print(json.dumps({"error": message}, indent=2))
    else:
        print(message)
    return 1


def main(argv=None) -> int:
    args = _build_argparser().parse_args(argv)

    store_root = Path(args.store)
    facts_dir = store_root / "facts"

    if not PKG.is_dir():
        return _fatal(args, f"ERROR: no compactor package beside this script ({PKG}).")

    sys.path.insert(0, str(PKG))
    try:
        import backfill  # noqa: E402
    except Exception as e:
        return _fatal(
            args,
            f"ERROR importing compactor/backfill.py from {PKG}: "
            f"{type(e).__name__}: {e}",
        )

    if not facts_dir.is_dir():
        return _fatal(args, f"ERROR: {facts_dir} does not exist or is not a directory.")

    warnings: list[str] = []
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
                f"REFUSING --apply: something is answering {DEFAULT_HEALTH_URL} "
                "— a live compactor may be writing these exact backfill "
                "records right now. Stop it first (supervisorctl stop "
                "compactor) or pass --force to override.",
            )
        if alive and args.force:
            warnings.append(
                f"WARNING: --force overriding a live compactor detected at "
                f"{DEFAULT_HEALTH_URL} — proceeding anyway."
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

    if args.apply:
        return 1 if apply_had_error else 0
    return 3 if counts["would-resume"] > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
