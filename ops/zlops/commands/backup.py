"""`zl backup now|list|verify [ARCHIVE|--latest]` -- wraps
`compactor/backup.py` under podenv.COMPACTOR_PY. `--once` is live-SQLite-safe
by design (compactor/backup.py's own module docstring, principle 2), so
`now` runs with no service stop and no --apply gate.

`zl backup restore` always REFUSES (exit 1): automated restore is out of
scope for this bundle -- OPERATIONS.md's manual procedure exists precisely
because `backup.py --restore` is unsafe on some versions of this codebase
and not yet reviewed on others (OPERATIONS.md "Restore from a backup").

Review H1: `now`/`list`/`verify` gate everything after their own arguments
through `_gate_passthrough` -- ONLY `--json` is accepted; `--restore` (and
any argparse prefix of it, e.g. `--rest`), `--prune-pre-restore`,
`--daemon`, `--list`, `--verify`, `--yes` and `--list-pre-restore` are all
refused outright, whatever subcommand they follow. `zl backup` has no
--apply gate the way `zl fix`/`zl sshd` do (`--once` needs none; nothing
else here should ever run), so there is no way to opt into a wider
passthrough here at all -- unlike `refuse_passthrough_write_flags`'s
`zl_apply=` escape hatch.
"""

from __future__ import annotations

import sys

from .. import common, podenv

_BACKUP_PY = podenv.COMPACTOR_PKG / "backup.py"

# backup.py's own COMPLETE declared long-option list (its argparse, read
# from compactor/backup.py -- review H1). `now`/`list`/`verify` used to
# forward `rest` to backup.py with NO gate at all: backup.py's own
# argparse checks `--restore` (and `--prune-pre-restore`) the same way it
# checks `--once`/`--list`/`--verify`, so `zl backup now --restore ARCHIVE
# --yes` reached `restore_backup` outright, and `zl backup list
# --prune-pre-restore 0` ran the prune -- exactly the class of hole H1's
# own predecessor (review R2-B2) fixed for `zl fix` and `zl sshd`, left
# open here.
_ALL_FLAGS = (
    "--daemon", "--json", "--list", "--list-pre-restore", "--once",
    "--prune-pre-restore", "--restore", "--verify", "--yes",
)
# `zl backup` has no --apply gate to route a write through in the first
# place (`--once` is live-SQLite-safe by design -- compactor/backup.py's
# own module docstring, principle 2 -- and nothing else this wrapper calls
# is), so unlike `zl fix`/`zl sshd`'s write_flags check (refused unless
# --apply), there is no escape hatch here at all: ONLY `--json` is ever
# safe passthrough for `now`/`list`/`verify`. Everything else backup.py's
# own argparse declares -- `--restore` and any prefix of it, `--prune-
# pre-restore`, `--daemon`, `--list`, `--verify`, `--yes`,
# `--list-pre-restore` -- is refused outright, whatever subcommand it
# follows.
_SAFE_FLAGS = frozenset({"--json"})


def _gate_passthrough(rest: list[str], what: str) -> tuple[list[str], int | None]:
    """Normalize `rest` against backup.py's real argparse (`--flag=value`
    split, unambiguous-prefix expansion -- the same machinery `zl fix`
    uses, review R2-B2) and refuse anything that isn't `--json`. Returns
    `(normalized, None)` when clean, or `([], exit_code)` when refused.
    """
    try:
        checked = common.normalize_passthrough(rest, _ALL_FLAGS)
    except common.AmbiguousPassthroughFlag as e:
        print(f"zl backup {what}: refusing -- {e}", file=sys.stderr)
        return [], 2
    bad = [f for f in checked if f not in _SAFE_FLAGS]
    if bad:
        print(
            f"zl backup {what}: refusing -- {bad} is not a safe passthrough flag for this wrapper "
            f"(only --json is; `zl backup restore` always refuses, and there is no --apply gate here "
            "to route anything else through) -- see ops/README.md",
            file=sys.stderr,
        )
        return [], 2
    return checked, None

_RESTORE_REFUSAL = """\
zl backup restore: REFUSED.

Automated restore is deliberately out of scope for this bundle. Follow the
manual procedure instead: ops/README.md's restore pointer, or directly
OPERATIONS.md -> "Restore from a backup (recover lost/corrupted memory)".

That procedure stops all FOUR writers first:
  supervisorctl stop openwebui compactor backup webuidb-sync
(`webuidb-sync: ERROR (not running)` is the expected answer on a
WEBUI_DB_LOCAL=false pod.)
"""


def run(argv: list[str]) -> int:
    if not argv:
        print("usage: zl backup now|list|verify [ARCHIVE|--latest]|restore")
        return 2
    sub, rest = argv[0], argv[1:]

    if sub == "restore":
        print(_RESTORE_REFUSAL)
        return 1

    # review B3: backup.py reads WEBUI_DB_LOCAL/COMPACTOR_BACKUP_RETAIN/etc
    # straight from os.environ, no fallback of its own -- an SSH/Web
    # Terminal session lacks them (only pid 1 carries them), so every
    # invocation here gets common.merged_environ() explicitly.
    env = common.merged_environ()

    if sub == "now":
        checked, gate_code = _gate_passthrough(rest, "now")
        if gate_code is not None:
            return gate_code
        return common.run_stream([str(podenv.COMPACTOR_PY), str(_BACKUP_PY), "--once", *checked], env=env)

    if sub == "list":
        checked, gate_code = _gate_passthrough(rest, "list")
        if gate_code is not None:
            return gate_code
        return common.run_stream([str(podenv.COMPACTOR_PY), str(_BACKUP_PY), "--list", *checked], env=env)

    if sub == "verify":
        archive = None
        remaining = []
        it = iter(rest)
        for token in it:
            if token == "--latest":
                candidates = sorted(
                    (p for p in podenv.BACKUPS.glob("*.tar.gz") if p.is_file()),
                    key=lambda p: p.stat().st_mtime,
                )
                if not candidates:
                    print(f"zl backup verify --latest: no archives in {podenv.BACKUPS}")
                    return 1
                archive = str(candidates[-1])
            else:
                remaining.append(token)
        if archive is None:
            if not remaining:
                print("usage: zl backup verify ARCHIVE|--latest")
                return 2
            archive = remaining.pop(0)
        checked, gate_code = _gate_passthrough(remaining, "verify")
        if gate_code is not None:
            return gate_code
        return common.run_stream(
            [str(podenv.COMPACTOR_PY), str(_BACKUP_PY), "--verify", archive, *checked], env=env
        )

    print(f"zl backup: unknown subcommand {sub!r}; usage: zl backup now|list|verify [ARCHIVE|--latest]|restore")
    return 2
