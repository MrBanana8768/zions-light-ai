#!/usr/bin/env python3
"""word-ban-deploy.py — deploy the word-ban grammar to the compactor's active
file (v3.1.9.8 FILE DEPLOY), replacing the old workflow of pasting
`word-ban-structured_outputs.value.txt` into OpenWebUI's Advanced Params.

Runs ON THE POD, `python3` stdlib only — no venv, no pip install, nothing but
the interpreter every RunPod image already has. See OPERATIONS.md "Word-ban
file deploy" for the full transition steps.

    word-ban-deploy.py deploy <value.txt> --revision N [--sha SHA] [--model M]
    word-ban-deploy.py status [--file PATH] [--host HOST] [--port PORT]
    word-ban-deploy.py rollback [--file PATH]
    word-ban-deploy.py clear [--file PATH] --yes

Exit codes: 0 ok, 1 refused, 2 usage.

PRIVACY: this script never prints the grammar itself — only shas, revisions,
byte counts and timings. Every `deploy`/`status`/`rollback` line is safe to
paste into a chat or a ticket.

CANONICALIZATION (must match `compactor/wordban_runtime.py`'s
`canonical_value_bytes` exactly, and `scripts/wordban/wordban.py cmd_build`'s
own `json.dumps({"grammar": ebnf}, ensure_ascii=True)`):

    json.dumps(structured_outputs, ensure_ascii=True).encode("utf-8")

`<value.txt>` is exactly that string, already on disk (`wordban.py release`'s
`word-ban-structured_outputs.value.vN.txt`). `deploy` parses it once, and
refuses (exit 1) unless RE-canonicalizing the parsed value reproduces the
file's own bytes byte-for-byte — closing the one loophole that would let a
non-canonical value.txt (a stray newline, reordered keys, a hand-edit) pass
its own sha check here and then FAIL the compactor's identical check on
load, which would silently fall back to no grammar at all on every request
until someone noticed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_FILE = Path(
    os.environ.get("COMPACTOR_WORDBAN_FILE", "") or "/data/zl-ops/word-ban/active.json"
)

# scripts/wordban/banlist.yaml's own `model:` line (v8: coder3101/
# Cydonia-24B-v4.3-vision-heretic) — hostile review round 1, F8: the file
# now applies to ONE model, recorded here at deploy time.
DEFAULT_MODEL = "coder3101/Cydonia-24B-v4.3-vision-heretic"

# Hostile review round 1, F4: revisions that must NEVER be deployed or
# rolled back to. Kept in sync BY HAND with scripts/wordban/banlist.yaml's
# own `withdrawn_revisions` (commit 2d964d0) — NOT read from that file at
# runtime, because this script runs ON THE POD, which does not have
# scripts/wordban/ checked out (only the released value.txt/release.vN.json
# artifacts are copied there). The alternative (recording the withdrawn
# list INSIDE active.json/previous.json at deploy time) was considered and
# rejected: it would mean a revision withdrawn AFTER a rollback target was
# deployed is invisible to a LATER `rollback` that swaps that stale
# previous.json back in — this hardcoded list is checked fresh, every
# time, regardless of when the file on disk was written. Update this
# whenever banlist.yaml's withdrawn_revisions changes.
WITHDRAWN_REVISIONS = {
    4: "it traps the model — after 'angel' plus an accented letter "
       "(angelĂ, Angelø, angeléñÖs), only combining marks are allowed and "
       "the reply cannot end (live 2026-09-23, 5 replies).",
}

# Hostile review round 2, N2: WITHDRAWN_REVISIONS matches by REVISION LABEL
# only — a withdrawn value re-released under a different --revision number
# (rev 4's own bytes, redeployed as "--revision 5") sails straight past it,
# because nothing here ever looks at the bytes themselves. This is the
# actual trap-carrying VALUE's sha256 (the same canonicalization
# `canonical_value_bytes` uses — confirmed by hand against
# ~/zl-ops/word-ban/word-ban-structured_outputs.value.v4.txt, whose raw
# bytes are ALREADY in canonical form, so its plain sha256sum and this
# module's canonical_value_bytes hash agree: 6b052ac1a4aa...), kept in
# sync BY HAND for the same reason WITHDRAWN_REVISIONS is (no
# scripts/wordban/ checkout on the pod). Checked in BOTH `cmd_deploy` (by
# the raw file's own sha256, before any --revision label is trusted) and
# `_validate_doc_shape` (by previous.json's own value_sha256, before a
# rollback swap) — a withdrawn value must never load, however it is
# labelled.
WITHDRAWN_VALUE_SHA256S = {
    "6b052ac1a4aa9be2e3c0428f6229a60b83c8f36fc07f06df65dac0da24d5c740":
        "this is revision 4's released value — it traps the model (see "
        "WITHDRAWN_REVISIONS[4] for the detail), regardless of what "
        "--revision label it is deployed or rolled back under.",
}


def _default_model() -> str:
    """N1: `--model` defaults to `$MODEL_REPO` when that env is set (the
    same variable the compactor itself reads for its OWN configured model
    — see compactor/main.py's `MODEL_REPO = os.environ.get("MODEL_REPO")`
    and wordban_runtime.configured_model()), else DEFAULT_MODEL. Computed
    fresh at parser-build time (not a module-level constant) so a test — or
    an operator's env — that sets MODEL_REPO after import still sees it."""
    return os.environ.get("MODEL_REPO") or DEFAULT_MODEL


def say(*a) -> None:
    print(*a, flush=True)


def canonical_value_bytes(structured_outputs) -> bytes:
    """MUST match compactor/wordban_runtime.py's function of the same name
    exactly — see this file's module docstring."""
    return json.dumps(structured_outputs, ensure_ascii=True).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_write_json(path: Path, doc: dict) -> None:
    """tmp-in-same-dir, fsync, rename, fsync-the-dir — a crash mid-write
    leaves either the OLD file (rename never happened) or the NEW one
    (rename is atomic on the same filesystem), never a half-written one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(doc, indent=1, ensure_ascii=False).encode("utf-8") + b"\n"
    tmp = path.parent / f".{path.name}.tmp{os.getpid()}"
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(str(tmp), str(path))
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    dfd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# deploy
# ---------------------------------------------------------------------------

def _sha_from_release_manifest(release_path: Path, value_path: Path, revision: int):
    """Hostile review round 1, F3: the REAL `release.vN.json` (written by
    `wordban.py cmd_release`, confirmed against
    `~/zl-ops/word-ban/release.v8.json`) has NO top-level `value_sha256`
    key at all — the old code here read one and would refuse every real
    release. Its actual top-level keys are `revision, released, files,
    grammar_sha256, banlist_sha256, verify, e2e`, and the sha this command
    wants lives at `files[value_path.name]`, as a plain sha256 hex STRING
    (`{"word-ban-structured_outputs.value.v8.txt": "a40a9de8...", ...}`).
    Also cross-checks `rel["revision"] == revision` — a value.txt copied
    next to the WRONG release.vN.json (a stale directory, a typo in
    --revision) must not silently borrow that file's sha.

    Returns (sha_or_None, error_message_or_None).
    """
    try:
        rel = _read_json(release_path)
    except (OSError, ValueError) as e:
        return None, f"could not read {release_path}: {type(e).__name__}: {e}"
    if not isinstance(rel, dict):
        return None, f"{release_path} does not contain a JSON object"
    if rel.get("revision") != revision:
        return None, (
            f"{release_path} is for revision {rel.get('revision')!r}, "
            f"not --revision {revision}"
        )
    files = rel.get("files")
    if not isinstance(files, dict):
        return None, f"{release_path} has no usable 'files' object"
    entry = files.get(value_path.name)
    if entry is None:
        return None, f"{release_path}'s files has no entry for {value_path.name!r}"
    if isinstance(entry, str):
        return entry, None
    if isinstance(entry, dict):
        # Defensive only — every real release manifest inspected so far
        # (v8) stores a plain string; accepting a dict shape too costs
        # nothing if a future release format wraps it with more fields.
        sha = entry.get("sha256") or entry.get("value_sha256")
        if isinstance(sha, str) and sha:
            return sha, None
        return None, f"{release_path}'s files[{value_path.name!r}] is a dict with no usable sha256"
    return None, f"{release_path}'s files[{value_path.name!r}] is {type(entry).__name__}, not a string"


def cmd_deploy(args) -> int:
    if args.revision in WITHDRAWN_REVISIONS:
        say(f"REFUSED: revision {args.revision} is withdrawn — "
            f"{WITHDRAWN_REVISIONS[args.revision]}")
        return 1
    value_path = Path(args.value).expanduser()
    if not value_path.is_file():
        say(f"REFUSED: {value_path} is not a file")
        return 1
    raw = value_path.read_bytes()
    computed = sha256_hex(raw)

    # N2: check the VALUE itself against the withdrawn-value set BEFORE
    # trusting whatever --revision label was passed — this is exactly the
    # loophole a mislabelled redeploy of rev 4's bytes under "--revision 5"
    # would otherwise slip through (WITHDRAWN_REVISIONS alone only catches
    # the label, never the bytes).
    if computed in WITHDRAWN_VALUE_SHA256S:
        say(
            f"REFUSED: {value_path} (sha256 {computed[:16]}…) is a withdrawn "
            f"value, regardless of --revision {args.revision} — "
            f"{WITHDRAWN_VALUE_SHA256S[computed]}"
        )
        return 1

    expected = args.sha
    release_path = None
    if expected is None:
        release_path = value_path.parent / f"release.v{args.revision}.json"
        if release_path.is_file():
            expected, err = _sha_from_release_manifest(release_path, value_path, args.revision)
            if err is not None:
                say(f"REFUSED: {err}")
                return 1
        if expected is None:
            say(
                f"REFUSED: no --sha given and no usable sha in "
                f"{release_path} — pass --sha explicitly"
            )
            return 2
    if computed != expected:
        say(
            f"REFUSED: sha256 mismatch — {value_path} hashes to "
            f"{computed[:16]}…, expected {expected[:16]}…"
            + (f" (from {release_path})" if release_path else " (--sha)")
        )
        return 1

    try:
        so = json.loads(raw)
    except ValueError as e:
        say(f"REFUSED: {value_path} is not valid JSON ({type(e).__name__}: {e})")
        return 1
    recanonicalized = canonical_value_bytes(so)
    recomputed = sha256_hex(recanonicalized)
    if recomputed != computed:
        # See the module docstring: this is the check that keeps the
        # compactor's own load-time verification from ever disagreeing
        # with this one.
        say(
            f"REFUSED: {value_path} is not in canonical form — "
            f"re-serializing the parsed value (json.dumps(ensure_ascii=True)) "
            f"produces different bytes (sha256 {recomputed[:16]}… vs the "
            f"file's {computed[:16]}…). Rebuild it with wordban.py rather "
            f"than hand-editing."
        )
        return 1

    active_path = Path(args.file).expanduser()
    previous_path = active_path.with_name("previous.json")
    if active_path.is_file():
        try:
            old_doc = _read_json(active_path)
            _atomic_write_json(previous_path, old_doc)
            say(
                f"kept the previous deploy as {previous_path} "
                f"(revision {old_doc.get('revision')}, sha256 "
                f"{str(old_doc.get('value_sha256'))[:12]}…)"
            )
        except (OSError, ValueError) as e:
            say(
                f"WARNING: could not preserve the previous active.json as "
                f"{previous_path} ({type(e).__name__}: {e}); rollback to it "
                f"will not be possible until the NEXT successful deploy"
            )

    new_doc = {
        "revision": args.revision,
        "value_sha256": computed,
        "structured_outputs": so,
        "model": args.model,
        "deployed_at": _now_iso(),
    }
    _atomic_write_json(active_path, new_doc)
    say(
        f"DEPLOYED revision {args.revision} (model={args.model}) to {active_path}: "
        f"{len(raw):,} B, sha256 {computed}"
    )
    say(
        "Next: confirm the compactor picked it up (`word-ban-deploy.py "
        "status`) and watch for the warm-up to complete (word_ban.warmed / "
        "last_warm_compile_s in the same status)."
    )
    return 0


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

def cmd_status(args) -> int:
    active_path = Path(args.file).expanduser()
    if not active_path.is_file():
        say(f"no active deploy: {active_path} does not exist")
    else:
        try:
            doc = _read_json(active_path)
            say(
                f"active: revision={doc.get('revision')} "
                f"model={doc.get('model')} "
                f"sha256={str(doc.get('value_sha256'))[:16]}… "
                f"deployed_at={doc.get('deployed_at')} "
                f"(file {active_path})"
            )
        except (OSError, ValueError) as e:
            say(f"REFUSED to read {active_path}: {type(e).__name__}: {e}")
            return 1

    url = f"http://{args.host}:{args.port}/health/full"
    try:
        with urllib.request.urlopen(url, timeout=args.timeout) as r:
            report = json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as e:
        say(f"compactor unreachable at {url} ({type(e).__name__}: {e}); "
            "file state above is the only thing this command could confirm")
        return 0
    wb = report.get("word_ban")
    if wb is None:
        say(f"compactor at {url} has no word_ban key in /health/full yet "
            "(older build?)")
        return 0
    say(f"compactor reports: {json.dumps(wb, sort_keys=True)}")
    return 0


# ---------------------------------------------------------------------------
# rollback
# ---------------------------------------------------------------------------

def _validate_doc_shape(doc, label: str):
    """Hostile review round 1, F4: `rollback` used to swap `previous.json`
    in with NO checks at all — a corrupted or hand-edited previous.json
    (a truncated write, a stray manual edit) would be trusted outright,
    the one thing `deploy` never does for a fresh value.txt. Runs the
    SAME shape/sha checks `_parse_and_validate` (compactor/
    wordban_runtime.py) enforces on load, here, BEFORE the swap, so a
    rollback that would fail to load is refused instead of silently
    disabling the ban (fail-open on the request path; fail CLOSED here,
    on an explicit operator action, is the right asymmetry — a rollback
    the operator is watching for should say so, not degrade quietly).

    Returns an error message, or None if `doc` is usable.
    """
    if not isinstance(doc, dict):
        return f"{label} does not contain a JSON object"
    for key in ("revision", "value_sha256", "structured_outputs", "model"):
        if key not in doc:
            return f"{label} is missing {key!r}"
    revision = doc["revision"]
    if not isinstance(revision, int):
        return f"{label}'s revision is {type(revision).__name__}, not an integer"
    if revision in WITHDRAWN_REVISIONS:
        return f"{label} is revision {revision}, which is withdrawn — {WITHDRAWN_REVISIONS[revision]}"
    model = doc["model"]
    if not isinstance(model, str) or not model:
        return f"{label}'s model must be a non-empty string"
    computed = value_sha256_of(doc["structured_outputs"])
    if computed != doc["value_sha256"]:
        return (
            f"{label}'s value_sha256 ({str(doc['value_sha256'])[:12]}…) does not "
            f"match its own structured_outputs ({computed[:12]}…) — this file was "
            f"corrupted or hand-edited"
        )
    # N2: the VALUE itself, not just its revision label, must not be a
    # withdrawn one — a previous.json for revision 5 whose bytes are
    # actually rev 4's (the same mislabelling this check exists to close in
    # cmd_deploy) must not be trusted just because the label passed above.
    if doc["value_sha256"] in WITHDRAWN_VALUE_SHA256S:
        return (
            f"{label}'s value (sha256 {str(doc['value_sha256'])[:16]}…) is a "
            f"withdrawn value, regardless of its revision label "
            f"({revision}) — {WITHDRAWN_VALUE_SHA256S[doc['value_sha256']]}"
        )
    return None


def value_sha256_of(structured_outputs) -> str:
    """Same function as compactor/wordban_runtime.py's — see this file's
    module docstring for why the two must agree."""
    return sha256_hex(canonical_value_bytes(structured_outputs))


def cmd_rollback(args) -> int:
    active_path = Path(args.file).expanduser()
    previous_path = active_path.with_name("previous.json")
    if not previous_path.is_file():
        say(f"REFUSED: no {previous_path} to roll back to")
        return 1
    try:
        prev_doc = _read_json(previous_path)
    except (OSError, ValueError) as e:
        say(f"REFUSED: could not read {previous_path}: {type(e).__name__}: {e}")
        return 1
    _err = _validate_doc_shape(prev_doc, str(previous_path))
    if _err is not None:
        say(f"REFUSED: {_err}")
        return 1
    # A real swap, not a one-way copy: whatever WAS active becomes the new
    # "previous" (never itself re-enabled — see the module note below),
    # so a rollback can itself be rolled back if it turns out to be wrong,
    # without needing the operator to have kept a copy by hand.
    cur_doc = None
    if active_path.is_file():
        try:
            cur_doc = _read_json(active_path)
        except (OSError, ValueError) as e:
            say(
                f"WARNING: could not read the current {active_path} to keep "
                f"it as the new previous.json ({type(e).__name__}: {e}); "
                f"rolling back anyway"
            )
    _atomic_write_json(active_path, prev_doc)
    if cur_doc is not None:
        _atomic_write_json(previous_path, cur_doc)
    say(
        f"ROLLED BACK to revision {prev_doc.get('revision')} "
        f"(sha256 {str(prev_doc.get('value_sha256'))[:16]}…)"
    )
    return 0


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------

def cmd_clear(args) -> int:
    if not args.yes:
        say("REFUSED: clear requires --yes")
        return 2
    active_path = Path(args.file).expanduser()
    if not active_path.is_file():
        say(f"nothing to clear: {active_path} does not exist")
        return 0
    active_path.unlink()
    say(f"cleared {active_path} — the compactor now falls back to the "
        "client-sent structured_outputs, if any, else none")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("deploy", help="deploy a released value file")
    d.add_argument("value", help="path to word-ban-structured_outputs.value.vN.txt")
    d.add_argument("--revision", type=int, required=True)
    d.add_argument("--sha", default=None, help="expected sha256 of the value file")
    d.add_argument("--model", default=_default_model(),
                   help=f"the model this grammar applies to (default $MODEL_REPO "
                        f"if set, else {DEFAULT_MODEL})")
    d.add_argument("--file", default=str(DEFAULT_FILE))
    d.set_defaults(func=cmd_deploy)

    s = sub.add_parser("status", help="show the active deploy and the compactor's own view")
    s.add_argument("--file", default=str(DEFAULT_FILE))
    s.add_argument("--host", default="localhost")
    s.add_argument("--port", type=int, default=8080)
    s.add_argument("--timeout", type=float, default=5.0)
    s.set_defaults(func=cmd_status)

    r = sub.add_parser("rollback", help="swap the previous deploy back in")
    r.add_argument("--file", default=str(DEFAULT_FILE))
    r.set_defaults(func=cmd_rollback)

    c = sub.add_parser("clear", help="remove the active deploy")
    c.add_argument("--file", default=str(DEFAULT_FILE))
    c.add_argument("--yes", action="store_true")
    c.set_defaults(func=cmd_clear)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
