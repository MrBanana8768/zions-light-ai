"""`zl` — the top-level dispatcher.

Every subcommand module exposes `run(argv: list[str]) -> int`. This module
just: picks the right one, times it, writes the audit log line (always, even
on a crash or Ctrl-C -- see `main`), and prints what the exit code means.

No bare except here swallows a traceback silently (memory.md /
Mistakes log, and the brief says so explicitly): a genuine bug still prints
its full traceback, after the audit line is written.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable

from . import common
from .commands import (
    backup as cmd_backup,
    bundle as cmd_bundle,
    dash as cmd_dash,
    disk as cmd_disk,
    fix as cmd_fix,
    health as cmd_health,
    history as cmd_history,
    journal as cmd_journal,
    logs as cmd_logs,
    models as cmd_models,
    selftest as cmd_selftest,
    shadow as cmd_shadow,
    sshd as cmd_sshd,
    status as cmd_status,
    tools as cmd_tools,
    update as cmd_update,
    version as cmd_version,
    wordban as cmd_wordban,
)

# (name, group, one-line description, run(argv) -> int)
# group is "read" (READ-ONLY) or "change" (CHANGES THINGS) -- the brief's own
# split, shown in `zl help` so an operator never has to guess.
COMMANDS: list[tuple[str, str, str, "callable"]] = [
    ("status", "read", "supervisorctl status + health verdict + GPU one-liner", cmd_status.run),
    ("health", "read", "/health/full, interpreted (exit 0 ok, 1 crit, 3 warn)", cmd_health.run),
    ("selftest", "read", "GET /admin/selftest (a real chat round-trip)", cmd_selftest.run),
    ("logs", "read", "tail the service logs (scripts/tail-logs.sh, passthrough)", cmd_logs.run),
    ("dash", "read", "live terminal telemetry dashboard", cmd_dash.run),
    ("disk", "read", "du of each /data/*, df /, backups, HF cache size", cmd_disk.run),
    ("bundle", "change", "build a zla-bundle-<ts>.tar.gz diagnostic bundle", cmd_bundle.run),
    ("backup", "change", "wrap compactor/backup.py: now / list / verify", cmd_backup.run),
    ("journal", "change", "check/repair a hot sqlite journal (recover-webui-db.py)", cmd_journal.run),
    ("sshd", "change", "install/harden sshd on this pod (scripts/setup-sshd.py)", cmd_sshd.run),
    ("tools", "change", "apt-get install tmux htop sqlite3 ... (not persistent)", cmd_tools.run),
    ("fix", "change", "registry of repair/maintenance scripts -- `zl fix` lists it", cmd_fix.run),
    ("shadow", "change", "run a build beside the live compactor to test it", cmd_shadow.run),
    ("wordban-check", "read", "run the pod's uploaded word-ban check", cmd_wordban.run),
    ("models", "change", "passthrough to clean-models.sh (its own dry-run default)", cmd_models.run),
    ("version", "read", "clone ref, compactor build id, package versions", cmd_version.run),
    ("history", "read", "tail of this pod's own zl audit log, human-readable", cmd_history.run),
    ("update", "change", "git fetch + checkout in the pod's clone", cmd_update.run),
]

_BY_NAME = {name: (group, desc, fn) for name, group, desc, fn in COMMANDS}

# Per-command overrides of the generic EXIT_MEANINGS text (review LOW: `zl
# health`'s own exit 3 means "warnings found", not the generic "dry run
# found pending work" -- that generic text is actively misleading for a
# read-only command that never had a dry run to begin with). `status`
# shares the same convention (see zlops/commands/status.py's docstring).
def _journal_exit_note(args: list[str], code: int) -> str | None:
    """Review M1: recover-webui-db.py's own `return 2` site (webui.db does
    not exist -- checked before anything else the script does) used to
    print the generic EXIT_MEANINGS text for exit 2, "usage error" -- a
    wrong citation for `zl journal check`'s single most likely real
    failure. `zl journal`'s OWN usage-error returns (no subcommand, an
    unknown subcommand, an ambiguous/disallowed passthrough flag -- see
    zlops/commands/journal.py's own `return 2` sites) are genuinely usage
    errors and keep the generic text; only a `check`/`repair` invocation
    that actually reached the wrapped script gets this override.
    """
    sub = args[1] if len(args) > 1 else None
    if sub in ("check", "repair") and code == 2:
        return "recover-webui-db.py: webui.db does not exist (not a usage error)"
    # MEDIUM (round 4): common.EXIT_MEANINGS' generic text for 4 is
    # "partial apply (re-run)" -- flatly wrong for `zl journal repair`.
    # RUNBOOK_DB_JOURNAL.md section 6 says the opposite of "re-run": leave
    # the writers stopped, do not touch webui.db further, ask for help.
    # Codes 6/7 share the same generic "partial apply" text and the same
    # correction; journal.py's own message (printed above this one) gives
    # openwebui's real per-code state.
    if sub == "repair" and code in (4, 6, 7):
        return (
            "recover-webui-db.py: RUNBOOK_DB_JOURNAL.md section 6 -- writers are left STOPPED on "
            "purpose; do NOT re-run, see the message above and ask for help"
        )
    return None


_EXIT_NOTES: dict[str, dict[int, str] | Callable[[list[str], int], str | None]] = {
    "health": {
        1: "crit -- see the findings above",
        3: "warnings found (not a dry run -- `zl health` has no --apply) -- see the findings above",
    },
    "status": {
        1: "a service is not RUNNING, or health is crit -- see above",
        3: "health warnings -- see above",
    },
    "journal": _journal_exit_note,
}


def _exit_note(command_name: str, args: list[str], code: int) -> str | None:
    entry = _EXIT_NOTES.get(command_name)
    if entry is None:
        return None
    if callable(entry):
        return entry(args, code)
    return entry.get(code)


def _print_help() -> int:
    print("zl -- one command per repeated pod-operator action. Full guide: ops/README.md\n")
    for label, group in (("READ-ONLY", "read"), ("CHANGES THINGS", "change")):
        print(f"{label}:")
        width = max(len(n) for n, g, _d, _f in COMMANDS if g == group)
        for name, g, desc, _fn in COMMANDS:
            if g == group:
                print(f"  zl {name.ljust(width)}  {desc}")
        print()
    print("`zl backup restore` always refuses -- see ops/README.md's manual restore pointer.")
    print("Every command that changes anything is a DRY RUN by default; add --apply to do it.")
    return 0


def _dispatch(args: list[str]) -> int:
    if not args or args[0] in ("help", "-h", "--help"):
        return _print_help()
    name, rest = args[0], args[1:]
    entry = _BY_NAME.get(name)
    if entry is None:
        print(f"zl: unknown command {name!r} -- `zl help` lists every command", file=sys.stderr)
        return 2
    _group, _desc, fn = entry
    return fn(rest)


def main(argv: list[str]) -> int:
    """Every invocation, read-only included, appends one audit-log line
    (brief: "Every invocation ... appends one JSON line to
    podenv.AUDIT_LOG") and, at the end, prints what the exit code means.
    A crash still gets an audit line (exit 1) before its traceback prints;
    it is never hidden.

    ONE exception: `zl status --brief`, bootstrap.sh's login-hook command,
    which runs on every interactive SSH login and skips the audit write
    (review MEDIUM) -- a per-login audit line would both risk a MooseFS
    stall turning into a login hang and drown `zl history` in noise that
    was never an operator action.
    """
    start = time.time()
    start_iso = common.utcnow_iso()
    args = list(argv)
    command_name = args[0] if args else ""
    code = 2
    try:
        code = _dispatch(args)
        if code < 0:
            # review LOW: a wrapped child killed by a signal comes back as
            # Python's own `returncode == -N` (subprocess's convention) --
            # `common.exit_meaning` already PRINTS that as "reported as
            # exit 128+N" (the shell's own convention), but `sys.exit(code)`
            # in __main__.py would pass the RAW negative number through,
            # and the OS truncates that to `256 + code` (e.g. -2 -> 254),
            # not 128+2=130 -- the message and the actual process exit code
            # disagreed. Convert here, once, before anything (the audit
            # line, the printed note, `sys.exit`) sees `code`, so all three
            # agree.
            code = 128 + (-code)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    except KeyboardInterrupt:
        print("zl: interrupted", file=sys.stderr)
        duration = time.time() - start
        common.write_audit(args, 130, duration, start_iso)
        common.print_exit_meaning(130)
        return 130
    except Exception:
        duration = time.time() - start
        common.write_audit(args, 1, duration, start_iso)
        common.print_exit_meaning(1)
        raise
    duration = time.time() - start
    # `zl status --brief` is bootstrap.sh's login-hook command -- it runs on
    # EVERY interactive SSH login, so it skips the audit write (review
    # MEDIUM): a MooseFS stall on /data must not add a second, unrelated way
    # for a login to hang, and a login-per-line audit log would drown out
    # the operator-initiated actions `zl history` exists to show.
    if not (command_name == "status" and "--brief" in args[1:]):
        common.write_audit(args, code, duration, start_iso)
    common.print_exit_meaning(code, note=_exit_note(command_name, args, code))
    return code
