"""Shared plumbing for every `zl` command: the audit log, the dry-run/--apply
confirmation gate, supervisorctl helpers, subprocess wrappers, and the env
redaction rule used by `zl bundle`.

Kept here, not duplicated per command, for the same reason podenv.py is one
file: two copies of a safety rule (what counts as a secret, what "apply"
means) drifting apart is this codebase's most expensive recurring defect
(memory.md, Mistakes log).

Style: stdlib only, Python 3.12, type hints, no bare excepts that swallow a
traceback silently (the one exception is the audit log write itself, which
must never abort the action it is logging -- MooseFS stalls; see
`write_audit`).
"""

from __future__ import annotations

import http.client
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import podenv

# --- exit-code convention (brief: "Operator-script exit codes") ------------
EXIT_MEANINGS = {
    0: "done / nothing to do",
    1: "refusal or error",
    2: "usage error",
    3: "dry run found pending work (informational)",
    4: "partial apply (re-run)",
    5: "the pre-apply backup failed (a distinct code from the target script's own exit -- review H5)",
    130: "interrupted (Ctrl-C)",
}

# The pre-apply backup's own refusal code (review H5: "backup failure gets
# its own exit code distinct from the target's" -- a mutation that skips
# checking the backup's exit code must not be able to hide behind a number
# the wrapped script could also plausibly return on its own).
EXIT_BACKUP_FAILED = 5


def exit_meaning(code: int) -> str:
    """A negative code is a child killed by a signal (Python's own
    subprocess convention: `returncode == -N` for signal N) -- report it
    using the shell's own 128+N convention instead of falling through to
    "unrecognized exit code" (review LOW finding).

    Round-4 cosmetic: a POSITIVE code above 128 (this process's own exit
    code, or one reported by `region.finish()`'s `128+sig` fallback -- see
    its own docstring) is the SAME 128+N convention from the other side --
    `exit_meaning(143)` used to fall through to "unrecognized exit code"
    (143 not being a key `EXIT_MEANINGS` happens to define) instead of
    saying what it plainly is, "killed by signal 15". Checked only AFTER
    `EXIT_MEANINGS` itself (130 is explicitly "interrupted (Ctrl-C)" there
    and must keep that wording, not become "killed by signal 2").
    """
    if code < 0:
        sig = -code
        return f"killed by signal {sig} (reported as exit {128 + sig})"
    if code in EXIT_MEANINGS:
        return EXIT_MEANINGS[code]
    if 128 < code <= 128 + signal.NSIG:
        return f"killed by signal {code - 128}"
    return "unrecognized exit code"


def print_exit_meaning(code: int, note: str | None = None) -> None:
    """`note`, when given, replaces the generic EXIT_MEANINGS text for this
    one invocation -- e.g. `zl health`'s exit 3 means "warnings found", not
    the generic "dry run found pending work" (review LOW finding: that
    generic text is actively misleading for a read-only command).
    """
    print(f"zl: exit {code} -- {note or exit_meaning(code)}", file=sys.stderr)


def safe_print(msg: str, *, file=None) -> None:
    """`print`, swallowing OSError. Review B1: once `ProtectedRegion`'s
    SIGHUP handler has dup2'd /dev/null onto fds 1 and 2, a write to a dead
    terminal succeeds silently (it's writing to /dev/null) rather than
    raising -- but that dup2 is itself best-effort (it can fail; see
    `ProtectedRegion._on_hangup`), and code can run before the handler has
    fired at all (a race) or reach here through some path this review
    hasn't found yet. Every print reachable from inside a `ProtectedRegion`
    (stop_for_apply, restart_after_apply, refuse_after_stop_failure,
    `_wait_for_openwebui_http_ok`'s heartbeat, and each command's own
    "stopped ..."/"WARNING: ..." lines) goes through this, not the bare
    builtin, so a write that fails anyway can never abort the stop -> run
    -> restart flow it is only narrating. Never used for anything the
    operator's NEXT action depends on synchronously -- the run log under
    `ZL_HOME/runs` (written by `run_protected`) is the durable record.
    """
    try:
        if file is None:
            print(msg)
        else:
            print(msg, file=file)
    except OSError:
        pass


# --- audit log ---------------------------------------------------------------


def utcnow_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ssh_client_ip() -> str:
    """The IP of whoever is running this, from SSH's own env var (set by
    sshd for every session it starts -- see scripts/setup-sshd.py). A Web
    Terminal or a local shell has no SSH_CONNECTION at all.
    """
    conn = os.environ.get("SSH_CONNECTION", "").strip()
    if not conn:
        return "web-terminal"
    parts = conn.split()
    return parts[0] if parts else "web-terminal"


def write_audit(argv: list[str], exit_code: int, duration_s: float, start_iso: str) -> None:
    """Append one JSON line to podenv.AUDIT_LOG. MUST NEVER raise -- a
    MooseFS stall on /data must never abort the action it is logging. On any
    failure, warn to stderr and continue.

    Review LOW: a SECOND Ctrl-C, landing here (`cli.main` calls this AFTER
    the `with ProtectedRegion()` block in `fix.py`/`journal.py` has already
    exited and restored the default SIGINT handler -- services are already
    back by this point, so there is no longer a `ProtectedRegion` to defer
    through) used to be able to interrupt the write mid-line, losing the
    one record of what just happened. SIGINT is deferred for the duration
    of the write itself (not the whole function -- `ssh_client_ip()` and
    the record's own construction are cheap and unblockable is fine there).
    A signal caught this way still takes effect -- as `SystemExit(130)`,
    raised once the write is done, the same code `cli.main`'s own
    `except KeyboardInterrupt` already uses -- just not mid-write.
    """
    record = {
        "time": start_iso,
        "argv": argv,
        "client_ip": ssh_client_ip(),
        "exit_code": exit_code,
        "duration_s": round(duration_s, 3),
    }
    deferred: list[int] = []

    def _defer(signum: int, _frame) -> None:
        deferred.append(signum)

    old_sigint = None
    try:
        old_sigint = signal.signal(signal.SIGINT, _defer)
    except (ValueError, OSError):
        old_sigint = None  # not the main thread, or no SIGINT on this platform
    try:
        podenv.AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with podenv.AUDIT_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as e:
        print(f"zl: warning: could not write the audit log ({e}); continuing", file=sys.stderr)
    finally:
        if old_sigint is not None:
            signal.signal(signal.SIGINT, old_sigint)
    if deferred:
        print("zl: interrupted (the audit line above was written first)", file=sys.stderr)
        raise SystemExit(130)


# --- dry-run / --apply confirmation gate -------------------------------------


class ApplyRefused(SystemExit):
    """Raised (as a SystemExit(1)) when --apply is refused."""

    def __init__(self, message: str) -> None:
        print(f"zl: refusing to --apply: {message}", file=sys.stderr)
        super().__init__(1)


def split_passthrough(argv: list[str]) -> tuple[list[str], list[str]]:
    """Split argv at a literal "--": (our_own_flags, passthrough_after_it).
    Used by every command that wraps another script's own flags, so those
    never get parsed or validated twice.
    """
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1 :]
    return list(argv), []


class AmbiguousPassthroughFlag(Exception):
    """Raised by `normalize_passthrough_token` when a `--` token is a
    prefix of more than one flag the wrapped script's argparse declares
    (`allow_abbrev` is on by default, and argparse itself would refuse an
    ambiguous abbreviation with a usage error -- we refuse it OURSELVES,
    before the token ever reaches the gate check below, so an ambiguous
    prefix can never accidentally normalize to the WRONG flag and slip
    past `refuse_passthrough_write_flags`)."""


def normalize_passthrough_token(token: str, all_flags: tuple[str, ...]) -> tuple[str, str | None]:
    """Review R2-B2: `argparse.ArgumentParser()` (every wrapped script uses
    it, with abbreviations allowed by default) treats `--flag=value` as
    `--flag` with a value, and treats any unambiguous PREFIX of a declared
    long flag as that flag -- `--restore=S1`, `--rest S1`, `--appl`,
    `--forc` all mean exactly what `--restore`/`--restore`/`--apply`/
    `--force` mean to the script itself. The exact-match `in` test this
    gate used to run never saw any of that: it checked the LITERAL token
    against `write_flags`, so `--restore=S1` (never equal to `--restore`)
    and `--rest S1` (never equal to `--restore`) both sailed through as if
    they carried no write flag at all.

    Splits `--flag=value` into `("--flag", "value")` and expands any `--x`
    that is a prefix of exactly one flag in `all_flags` (equality counts as
    matching itself) to that flag's own full spelling. `all_flags` must be
    the SCRIPT's complete declared long-option list (every flag its own
    argparse defines, not just the write ones) -- a prefix ambiguous
    between a write flag and a harmless one (e.g. `--f` matching both
    `--force` and `--foo`) must still be refused as ambiguous, not silently
    treated as the harmless one.

    Returns `(normalized_token, value_or_None)`. A short option (`-x`), a
    positional, or a token not starting with `--` is returned unchanged
    with `value=None`. Raises `AmbiguousPassthroughFlag` if `token` (after
    splitting `=`) is a strict prefix of more than one entry in
    `all_flags`.
    """
    if not token.startswith("--") or token == "--":
        return token, None
    if "=" in token:
        flag, _, value = token.partition("=")
    else:
        flag, value = token, None

    if flag in all_flags:
        return flag, value

    prefix_matches = [f for f in all_flags if f.startswith(flag)]
    if len(prefix_matches) == 1:
        return prefix_matches[0], value
    if len(prefix_matches) > 1:
        raise AmbiguousPassthroughFlag(
            f"{token!r} is an ambiguous prefix of {prefix_matches} in the wrapped script's own flags"
        )
    return flag, value


def normalize_passthrough(passthrough: list[str], all_flags: tuple[str, ...]) -> list[str]:
    """Apply `normalize_passthrough_token` across a whole passthrough list,
    splitting `--flag=value` into two tokens (`refuse_passthrough_write_flags`
    below does a plain membership test, so the flag name must stand alone)
    and expanding unambiguous prefixes to their full spelling. Re-raises
    `AmbiguousPassthroughFlag` from the token normalizer unchanged -- the
    caller (each command's own `run`) turns that into the usual exit-2
    refusal, the same as any other bad passthrough.
    """
    out: list[str] = []
    for token in passthrough:
        flag, value = normalize_passthrough_token(token, all_flags)
        out.append(flag)
        if value is not None:
            out.append(value)
    return out


def refuse_passthrough_write_flags(
    passthrough: list[str], write_flags: tuple[str, ...], *, zl_apply: bool, what: str,
    all_flags: tuple[str, ...] | None = None,
) -> int | None:
    """Review B2: passthrough (after `--`) must never sneak a write past
    our own dry-run/confirm/backup/stop/restore gate. `write_flags` is the
    WRAPPED SCRIPT's own real list of write flags (read from its argparse,
    not guessed). Returns a usage-error exit code (2) and prints why when
    refused; None when the passthrough is fine.

    `all_flags`, when given, is the script's COMPLETE declared long-option
    list; `passthrough` is first run through `normalize_passthrough` (split
    `--flag=value`, expand unambiguous prefixes) so `--restore=S1`,
    `--rest S1`, `--appl`, `--forc` are all recognized for what they are,
    the same as the wrapped script's own argparse would recognize them
    (review R2-B2). An ambiguous prefix (matches more than one of the
    script's own flags) is refused outright, the same exit code, rather
    than guessed. When `all_flags` is omitted (a caller that has not yet
    been updated), the old literal-token behaviour is unchanged -- but
    every wrapped-script caller in this bundle passes it now.

    - `--apply` in passthrough is ALWAYS refused, even with zl's own
      --apply given: that is THIS wrapper's flag name, not the script's,
      and typing it after `--` is either a mistake or an attempt to run
      the script's real apply path (whatever the script calls it, e.g. its
      own `--apply`/`--force`/`--restore`) without going through zl at
      all -- `--apply` alone before `--` is the only correct way.
    - Any other declared write flag in passthrough is refused UNLESS zl's
      own --apply was given, so a write always goes through confirm /
      backup / stop / restore, never disguised as a dry run.
    """
    if all_flags is not None:
        try:
            checked = normalize_passthrough(passthrough, all_flags)
        except AmbiguousPassthroughFlag as e:
            print(f"zl: refusing -- {e}", file=sys.stderr)
            return 2
    else:
        checked = passthrough

    if "--apply" in checked:
        print(
            f"zl: refusing -- '--apply' after `--` is not the wrapped script's own flag; "
            f"use zl's own --apply on {what}",
            file=sys.stderr,
        )
        return 2
    hit = [f for f in write_flags if f in checked]
    if hit and not zl_apply:
        print(
            f"zl: refusing -- passthrough contains {hit} (a write flag of the wrapped script) "
            f"without zl's own --apply on {what}; add --apply so this goes through "
            "confirm/backup/stop/restore",
            file=sys.stderr,
        )
        return 2
    return None


def require_apply_confirmation(apply: bool, yes: bool, what: str = "this change") -> None:
    """Global rule (brief): every command that changes anything is a DRY RUN
    by default; --apply does it; --apply asks the operator to type "apply"
    unless --yes; a non-TTY session without --yes refuses outright (exit 1).
    Call this only once the caller has already decided --apply was given;
    a dry run never calls this at all.
    """
    if not apply:
        return
    if yes:
        return
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise ApplyRefused(
            f"--apply on {what} needs --yes on a non-interactive session (no TTY to confirm on)"
        )
    print(f"About to apply: {what}")
    try:
        typed = input("Type 'apply' to confirm, anything else aborts: ")
    except EOFError:
        typed = ""
    if typed.strip() != "apply":
        raise ApplyRefused("confirmation not given")


# --- subprocess helpers -------------------------------------------------------


def run(argv: list[str], timeout: float | None = None, **kw) -> subprocess.CompletedProcess:
    """Capture stdout/stderr as text. Never raises on a nonzero exit.

    Does NOT default `env=` to `merged_environ()` -- deliberately. This
    helper also runs plain admin utilities (supervisorctl, git, du,
    nvidia-smi) that have no business having their operator's PATH (and
    every fake-on-PATH this test suite's own convention relies on)
    silently overridden by pid 1's. Only the actual WRAPPED SCRIPTS (the
    repair/backup/sshd/tools/models scripts -- review B3) need pid 1's
    service-config values; each of those call sites passes
    `env=merged_environ()` itself, explicitly.
    """
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, **kw)


def run_stream(argv: list[str], **kw) -> int:
    """Run with inherited stdio (the child's own output goes straight to the
    operator) and return its exit code, unchanged -- see the brief's "child
    exit codes pass through unchanged". Same env policy as `run`: no
    implicit `merged_environ()` default (see `run`'s own docstring) --
    callers that need it (the wrapped scripts) pass it explicitly.
    """
    try:
        return subprocess.run(argv, **kw).returncode
    except FileNotFoundError as e:
        print(f"zl: could not run {argv[0]!r}: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        raise


class ProtectedRegion:
    """Review R2-B1: `run_protected`'s own SIGHUP-ignore/SIGINT-forward used
    to be installed only around the CHILD's own run -- the stop phase
    before it and the restart phase after it (which can block up to
    `_OPENWEBUI_READY_TIMEOUT_S` polling `/api/config`) were unprotected.
    A hangup during either killed the wrapper outright (no child to catch
    it, no controlling terminal trick to save it), abandoning the pod
    stopped with no audit line. `ProtectedRegion` widens the SAME
    protection to span the whole stop -> run -> restart window: a caller
    opens one with `with ProtectedRegion() as region:` before the first
    `stop_for_apply`, passes `region` into `run_protected`, and calls
    `region.finish()` once `restart_after_apply` has returned -- after
    which, if a signal arrived and was deferred, `finish()` raises
    `SystemExit(128+signal)` (the shell's own convention, already what
    `exit_meaning` prints) so the exit code matches what happened, and
    `cli.main`'s existing `except SystemExit` handler still writes the
    audit line, unchanged, before the process actually exits.

    - SIGHUP: ignored for the ENTIRE region (same reasoning as the old
      `run_protected`, just widened) -- we are still attached to the
      terminal that might hang up; the child, once running, is not (its
      own `start_new_session=True`).
    - SIGINT/SIGTERM: while a child is registered (`set_child`/
      `clear_child`, called by `run_protected` around the one child it
      runs), forward to it -- exactly the old behaviour. At every OTHER
      point in the region (during `stop_for_apply`, during
      `restart_after_apply`) there is no child to forward to: the signal
      is instead RECORDED (`deferred_signal`) and a message printed once,
      so an operator who hits Ctrl-C while the wrapper is, say, still
      polling openwebui sees why nothing happened yet, instead of a
      silently dropped keystroke.
    """

    def __init__(self) -> None:
        self.child_pid: int | None = None
        self.deferred_signal: int | None = None
        self.hungup = False
        self._message: str | None = None
        self._old_sighup = None
        self._old_sigint = None
        self._old_sigterm = None

    def set_child(self, pid: int) -> None:
        self.child_pid = pid

    def clear_child(self) -> None:
        self.child_pid = None

    def set_message(self, message: str) -> None:
        """Override the printed explanation for a deferred signal while no
        child is being forwarded to -- used by `zl journal repair` (review
        HIGH: "do not forward SIGINT/SIGTERM to recover-webui-db.py at
        all") so the operator sees "repair in progress; will not
        interrupt" rather than the generic stop/restart message, even
        though the mechanism (defer, never forward) is identical.
        """
        self._message = message

    def clear_message(self) -> None:
        self._message = None

    def _on_hangup(self, _signum, _frame) -> None:
        """Review B1: the pty-close probe (`review3/pty/`) found that
        `SIG_IGN` alone is not enough -- ignoring SIGHUP stops it from
        KILLING us, but every later `print()` (ours, or the copy of the
        child's own output in `run_protected`) still raises `OSError:
        [Errno 5]` writing to the now-dead terminal, and only
        `run_protected`'s child-output copy caught that; `fix.py`'s
        "stopped ..." print and `journal.py`'s did not, and neither did
        `restart_after_apply`'s own WARNING/heartbeat prints -- an
        uncaught OSError there escaped this region's `try`/`finally`
        entirely, abandoning the pod exactly as stopped as it was when the
        connection dropped, with no restart and no audit line.

        So: dup2 /dev/null onto fds 1 and 2 -- every later write to
        "the terminal" (ours or the child's relayed copy) now succeeds
        silently instead of raising at all -- and record that this
        happened (`self.hungup`), for `finish()`'s own message. NOT
        SIG_IGN: we keep this as a real handler (not "ignore and forget")
        specifically so we can note that the connection dropped, and
        `safe_print` (used everywhere reachable from inside the region,
        as defense in depth) still swallows any OSError that gets through
        anyway -- e.g. if the dup2 itself fails.
        """
        self.hungup = True
        try:
            devnull_fd = os.open(os.devnull, os.O_RDWR)
        except OSError:
            return
        try:
            os.dup2(devnull_fd, 1)
            os.dup2(devnull_fd, 2)
        except OSError:
            pass
        finally:
            try:
                os.close(devnull_fd)
            except OSError:
                pass

    def _handle(self, signum: int, _frame) -> None:
        if self.child_pid is not None:
            try:
                os.killpg(self.child_pid, signal.SIGINT)
            except OSError:
                pass  # already gone
            return
        if self.deferred_signal is None:
            safe_print(
                self._message
                or (
                    f"zl: signal {signum} received -- services are being stopped/restored and cannot "
                    "be interrupted safely right now; this will be acted on once that finishes"
                ),
                file=sys.stderr,
            )
        self.deferred_signal = signum

    def __enter__(self) -> "ProtectedRegion":
        try:
            self._old_sighup = signal.signal(signal.SIGHUP, self._on_hangup)
        except (ValueError, OSError):
            self._old_sighup = None  # not the main thread, or no SIGHUP on this platform
        self._old_sigint = signal.signal(signal.SIGINT, self._handle)
        self._old_sigterm = signal.signal(signal.SIGTERM, self._handle)
        return self

    def __exit__(self, *_exc) -> None:
        signal.signal(signal.SIGINT, self._old_sigint)
        signal.signal(signal.SIGTERM, self._old_sigterm)
        if self._old_sighup is not None:
            signal.signal(signal.SIGHUP, self._old_sighup)

    def finish(self, child_code: int = 0, *, restored: bool = True, applied: bool = False,
               label: str = "the fix") -> None:
        """Call once the whole protected region (stop, run, restart-or-
        leave-stopped) is over. Review H2: this USED to take no arguments
        and simply raise `SystemExit(128+signal)` whenever a signal had
        been deferred -- so a Ctrl-C during `journal repair`'s restart
        wait turned the SCRIPT's own exit 6 ("re-run", audited and
        printed) into a bare 130, and a Ctrl-C during `fix`'s restart wait
        did the same to a 4. The underlying command's own exit code
        (`child_code`) was simply lost.

        Precedence, spelled out (there is no single "right" answer here,
        so this is the documented convention, not a discovered fact):
        if `child_code` is NONZERO, it WINS -- kept as the actual exit
        code (so it reaches the audit log's `exit_code` field and
        `cli.py`'s own exit-code explanation unchanged), with the
        deferred signal named in the printed message rather than acted
        on. Only when `child_code` is 0 (nothing else worth reporting)
        does the deferred signal become the exit code, as `128+signal`
        (the shell's own convention, unchanged from before this review).

        `restored` controls the WORDING only, never the code: pass
        `restored=False` on a path that deliberately left services
        stopped (`zl journal repair`'s leave-stopped path, RUNBOOK_DB_
        JOURNAL.md section 6) so the printed message never claims
        "restored" when it did the opposite.

        `applied` (round-6): pass `True` only from the call site reached
        AFTER the wrapped script/repair has actually run to completion
        (never from an early refusal -- backup failure, stop-verification
        failure, an in-flight-wait abort -- where nothing was applied at
        all). A signal deferred during the RESTART phase, after a
        completed run, used to produce a message that never said whether
        the underlying command's own work had actually happened -- with
        `child_code == 0` the exit code became `128+signal`, and an
        operator going by the exit code alone (not this message) could
        reasonably re-run something that had, in fact, already succeeded.
        `applied=True` makes the message say so explicitly regardless of
        which code wins, so "do not re-run this" is never left implicit.
        `label` names what was applied ("the fix" / "the repair") for that
        sentence.

        A no-op (returns without raising) when nothing was deferred --
        the caller's own `return code` (or its `sys.exit` via `cli.main`)
        is what actually ends the process in that case.
        """
        if self.deferred_signal is None:
            return
        sig = self.deferred_signal
        state = "restored" if restored else "left stopped (see the message above)"
        hup_note = " (the terminal connection was lost partway through this)" if self.hungup else ""
        if child_code != 0:
            # Round-7: a nonzero code is not necessarily a plain failure --
            # some wrapped scripts use a specific nonzero code to mean
            # "safe (or required) to re-run" (see EXIT_MEANINGS), so an
            # unconditional "do not re-run it" here would flatly contradict
            # the script's own documented exit-code meaning. Report only
            # the fact ("the fix ran and exited N") and let EXIT_MEANINGS'S
            # own explanation of N stand on its own, unqualified either way.
            ran_note = f" -- {label} ran and exited {child_code}" if applied else ""
            safe_print(
                f"zl: signal {sig} received earlier -- keeping the underlying command's own exit "
                f"code {child_code} rather than the signal (services are {state}); "
                f"see EXIT_MEANINGS for what {child_code} means{ran_note}{hup_note}",
                file=sys.stderr,
            )
            raise SystemExit(child_code)
        # child_code == 0: the wrapped script completed and reported
        # success, so THIS is the one case where "do not re-run it" is an
        # unambiguous, safe thing to say -- there is no script-specific
        # exit-code meaning attached to 0 that could contradict it.
        applied_note = f" -- {label} WAS APPLIED (exit 0); do not re-run it" if applied else ""
        safe_print(
            f"zl: services are {state}; exiting now on the signal {sig} received earlier{hup_note}{applied_note}",
            file=sys.stderr,
        )
        raise SystemExit(128 + sig)


def run_protected(
    argv: list[str], *, name: str, env: dict[str, str] | None = None, region: ProtectedRegion | None = None,
    forward_signals: bool = True, no_forward_message: str | None = None,
) -> int:
    """Run a child that must survive losing the operator's connection --
    used for every stop -> run -> restore flow (`zl fix --apply`,
    `zl journal repair --apply`; review B1).

    - `start_new_session=True`: the child gets its own session (setsid), so
      it has NO controlling terminal at all. A tty hangup delivers SIGHUP
      only to the foreground process group of the terminal that hung up --
      the child, having no controlling terminal any more, cannot receive
      it, no matter what happens to us.
    - `PYTHONUNBUFFERED=1` is forced into the child's env (review MEDIUM):
      the image ships no such setting, the child's stdout is a pipe (not a
      tty), and 6 of the 7 wrapped scripts never flush their own output --
      so without this, `zl fix --apply` prints nothing until the child
      exits, which invites exactly the "did it hang? let me Ctrl-C" this
      whole protected-region mechanism exists to survive.
    - `region`, when given (review R2-B1), is a `ProtectedRegion` the
      CALLER opened before the stop phase -- SIGHUP/SIGINT/SIGTERM are
      already being handled by it for the whole stop -> run -> restart
      window, so this function does not install its own handlers; it only
      registers/clears itself as the region's current child
      (`set_child`/`clear_child`) so the region's own SIGINT/SIGTERM
      handler knows to forward here while this child is running. When
      `region` is omitted (no caller currently does this, kept for any
      future standalone use), this function protects only its own
      duration, exactly as it always did.
    - `forward_signals=False` (review HIGH, `zl journal repair`'s use of
      this function): a SIGINT/SIGTERM received while THIS child runs is
      NOT forwarded to it -- `region` is told (`set_message`) to defer the
      signal and print `no_forward_message` instead of registering this
      child, so the region's own handler falls into its "no child, defer"
      path even though a child is in fact running. Used for
      recover-webui-db.py: RUNBOOK_DB_JOURNAL.md says plainly not to
      interrupt its rename+copy2 swap, and its own `except Exception`
      does not catch `KeyboardInterrupt` -- forwarding SIGINT there (the
      normal behaviour) could leave webui.db renamed to `.broken-*` with
      a half-written replacement in its place.
    - Output goes to BOTH the terminal (best-effort: a write after the tty
      hangs up raises EPIPE/OSError, which we swallow -- a dead terminal
      must never kill this function) and a run log under
      `ZL_HOME/runs/<UTC>-<name>.log`, so what happened is on record even
      if nobody was watching the terminal when it happened.
    """
    if env is None:
        env = merged_environ()
    env = dict(env)
    env.setdefault("PYTHONUNBUFFERED", "1")

    run_dir = podenv.ZL_HOME / "runs"
    log_f = None
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        stamp = utcnow_iso().replace(":", "").replace("-", "")
        log_f = (run_dir / f"{stamp}-{name}.log").open("wb")
    except OSError as e:
        safe_print(f"zl: warning: could not open a run log ({e}); continuing without one", file=sys.stderr)
        log_f = None

    try:
        proc = subprocess.Popen(
            argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True,
        )
    except OSError as e:
        safe_print(f"zl: could not run {argv[0]!r}: {e}", file=sys.stderr)
        if log_f is not None:
            log_f.close()
        return 1

    own_handlers = region is None
    old_sighup = old_sigterm = old_sigint = None
    if region is not None:
        if forward_signals:
            region.set_child(proc.pid)
        else:
            region.set_message(
                no_forward_message
                or f"zl: signal received -- {name} is in progress and will not be interrupted"
            )
    else:
        try:
            old_sighup = signal.signal(signal.SIGHUP, signal.SIG_IGN)
        except (ValueError, OSError):
            pass  # not the main thread, or a platform without SIGHUP -- nothing to restore either

        def _forward_sigint(_signum, _frame):
            try:
                os.killpg(proc.pid, signal.SIGINT)
            except OSError:
                pass  # already gone

        old_sigterm = signal.signal(signal.SIGTERM, _forward_sigint)
        old_sigint = signal.signal(signal.SIGINT, _forward_sigint)

    try:
        assert proc.stdout is not None
        for chunk in iter(lambda: proc.stdout.read(4096), b""):
            if log_f is not None:
                try:
                    log_f.write(chunk)
                    log_f.flush()
                except OSError:
                    pass  # the run log itself must never be why this function dies
            try:
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
            except (BrokenPipeError, OSError):
                pass  # tty hung up -- never die on a write; the run log still has it
        proc.wait()
        return proc.returncode
    finally:
        if log_f is not None:
            try:
                log_f.close()
            except OSError:
                pass
        if region is not None:
            region.clear_child()
            region.clear_message()
        if own_handlers:
            signal.signal(signal.SIGTERM, old_sigterm)
            signal.signal(signal.SIGINT, old_sigint)
            if old_sighup is not None:
                signal.signal(signal.SIGHUP, old_sighup)


# --- supervisorctl -------------------------------------------------------------

_STATUS_LINE = re.compile(r"^(\S+)\s+(\S+)")


def supervisorctl_status() -> dict[str, str]:
    """`{program: STATE}` from `supervisorctl status`. STATE is whatever
    supervisorctl prints (RUNNING, STOPPED, FATAL, EXITED, ERROR, ...).
    A program supervisorctl doesn't know about (not configured) is simply
    absent from the returned dict. On any failure to even run
    supervisorctl, returns {} -- callers must treat that as "unknown", not
    "everything stopped".
    """
    try:
        r = run(["supervisorctl", "status"], timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return {}
    out: dict[str, str] = {}
    for line in (r.stdout or "").splitlines():
        m = _STATUS_LINE.match(line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


_ACTIVE_STATES = frozenset({"RUNNING", "STARTING", "BACKOFF"})
_STOPPED_STATES = frozenset({"STOPPED", "EXITED", "FATAL"})


def supervisorctl_stop(names: list[str]) -> subprocess.CompletedProcess | None:
    """None means the command itself did not complete (a TimeoutExpired or
    OSError) -- callers must treat that as a stop failure, not silently
    proceed (review H2: "a TimeoutExpired from stop/start escapes outside
    the try").

    Round-6 HIGH: `start_new_session=True` -- without it, `supervisorctl`
    inherits OUR foreground process group, and a Ctrl-C at the terminal
    (SIGINT delivered to the whole foreground group, not just to us) kills
    it mid-command exactly like `ProtectedRegion`'s own signal handling was
    built to prevent for the WRAPPER. A `supervisorctl stop` killed
    partway through can leave some of `names` stopped and others not, with
    nothing here able to tell which -- `stop_for_apply`'s own post-stop
    verification loop still catches that (it re-reads status rather than
    trusting this call's exit code), but there is no reason to let the
    terminal kill this subprocess in the first place when a plain setsid
    avoids it entirely.
    """
    try:
        return run(["supervisorctl", "stop", *names], timeout=60, start_new_session=True)
    except (OSError, subprocess.TimeoutExpired) as e:
        safe_print(f"zl: `supervisorctl stop {' '.join(names)}` did not complete: {e}", file=sys.stderr)
        return None


def supervisorctl_start(names: list[str]) -> subprocess.CompletedProcess | None:
    """Same `start_new_session=True` reasoning as `supervisorctl_stop` --
    a Ctrl-C reaching THIS command mid-restart is exactly as bad (services
    left down that should be coming back up) as one reaching the stop.
    """
    try:
        return run(["supervisorctl", "start", *names], timeout=60, start_new_session=True)
    except (OSError, subprocess.TimeoutExpired) as e:
        safe_print(f"zl: `supervisorctl start {' '.join(names)}` did not complete: {e}", file=sys.stderr)
        return None


class StopVerificationFailed(Exception):
    """Raised by `stop_for_apply` when, after `supervisorctl stop`, one of
    the services cannot be verified STOPPED/EXITED/FATAL within the
    timeout. `was_running` is the pre-stop set, for the caller to hand to
    `restart_after_apply` before refusing (see `refuse_after_stop_failure`)
    -- review H2: a script's own liveness refusal must never be the only
    backstop against running with a writer still (or again) up.
    """

    def __init__(self, message: str, was_running: list[str]) -> None:
        self.was_running = was_running
        super().__init__(message)


def stop_for_apply(names: list[str], *, verify_timeout: float = 20.0) -> list[str]:
    """Record which of `names` are active right now, stop exactly those,
    and BLOCK until every one of them is verifiably stopped -- re-reading
    `supervisorctl status` (never trusting the stop command's own exit
    code alone) until each is STOPPED/EXITED/FATAL or `verify_timeout`
    elapses. An empty/unreadable status (supervisorctl itself failing) is
    always treated as "not verified", never as "nothing is running"
    (review H2). Raises `StopVerificationFailed` on failure; returns the
    ones that were running (to hand to `restart_after_apply` in a
    `finally`) on success.

    Review R2-HIGH ("H2 pre-stop half is not fixed"): the OLD code went
    straight to `running_services(names)`, which is `[n for n in names if
    status.get(n) in _ACTIVE_STATES]` -- if `supervisorctl status` itself
    failed (unreachable socket, timeout, ...) that produced `status == {}`,
    every `status.get(n)` came back `None`, and `was_running` ended up `[]`
    by COINCIDENCE, not because anything was verified stopped. The caller
    then printed "(none of them were running)" and ran `--apply` with
    every writer still live. The FIRST status read, before anything is
    stopped, gets the same "empty/unreadable is never treated as nothing
    running" rule the post-stop read already had: refuse outright
    (`StopVerificationFailed`, `was_running=[]` -- nothing was stopped, so
    there is nothing to restart) if the read came back empty, or if any
    REQUESTED name is simply absent from it (supervisorctl not knowing
    about a program we were told to stop is exactly as unverifiable as the
    whole read failing -- it is never treated as "so it must be stopped").
    """
    if names:
        pre_status = supervisorctl_status()
        missing = [n for n in names if n not in pre_status]
        if not pre_status or missing:
            raise StopVerificationFailed(
                f"could not verify service status before stopping anything -- "
                f"supervisorctl status came back "
                f"{'empty/unreadable' if not pre_status else f'without {missing}'}; "
                f"refusing to guess whether {names} are safe to leave running",
                [],
            )
        was_running = [n for n in names if pre_status.get(n) in _ACTIVE_STATES]
    else:
        was_running = []
    if not was_running:
        return []

    result = supervisorctl_stop(was_running)
    if result is not None and result.returncode != 0:
        safe_print(
            f"zl: `supervisorctl stop {' '.join(was_running)}` exited {result.returncode}: "
            f"{((result.stdout or '') + (result.stderr or '')).strip()}",
            file=sys.stderr,
        )

    deadline = time.monotonic() + verify_timeout
    status: dict[str, str] = {}
    while True:
        status = supervisorctl_status()
        if status:
            still_active = [n for n in was_running if status.get(n) not in _STOPPED_STATES]
            if not still_active:
                return was_running
        if time.monotonic() >= deadline:
            break
        time.sleep(0.3)

    raise StopVerificationFailed(
        f"{was_running} did not all reach STOPPED/EXITED/FATAL within {verify_timeout:.0f}s "
        f"(supervisorctl status: {status or '(supervisorctl status failed or returned nothing)'})",
        was_running,
    )


def refuse_after_stop_failure(exc: StopVerificationFailed) -> tuple[int, bool]:
    """Shared refusal path for `StopVerificationFailed`: restart whatever
    was running before the (failed) stop attempt, report any that did not
    come back, and return `(1, restored)` -- the refusal exit code AND
    whether every one of them actually came back RUNNING (round-6 HIGH:
    callers must pass this straight through as `region.finish()`'s own
    `restored=` argument rather than hardcoding `True`, so the message
    never claims "restored" when a service did not come back). Used by
    every stop -> run -> restore wrapper so this logic exists once.
    """
    safe_print(f"zl: refusing to apply -- {exc}", file=sys.stderr)
    came_back = restart_after_apply(exc.was_running)
    restored = report_restart_failures(came_back)
    return 1, restored


_OPENWEBUI_READY_TIMEOUT_S = 180.0  # "a large database can take a few minutes" -- RUNBOOK_DB_JOURNAL.md:219


def _openwebui_ready_url() -> str:
    return f"http://127.0.0.1:{podenv.OPENWEBUI_PORT}/api/config"


_PROGRESS_PRINT_INTERVAL_S = 10.0


def _wait_for_openwebui_http_ok(timeout: float) -> bool:
    """Review R2-B1: this can block up to `_OPENWEBUI_READY_TIMEOUT_S`
    (180s) in total silence -- the exact window the review's probe found
    an operator most likely to conclude the wrapper had hung and hit
    Ctrl-C. Print a heartbeat roughly every `_PROGRESS_PRINT_INTERVAL_S`
    seconds so a quiet minute reads as "still working", not "stuck".
    """
    deadline = time.monotonic() + timeout
    url = _openwebui_ready_url()
    next_print = time.monotonic() + _PROGRESS_PRINT_INTERVAL_S
    while True:
        status, _doc, _error = http_get_json(url, timeout=5.0)
        if status == 200:
            return True
        now = time.monotonic()
        if now >= deadline:
            return False
        if now >= next_print:
            remaining = max(0, int(deadline - now))
            safe_print(
                f"zl: still waiting for openwebui's {url} to answer 200 (about {remaining}s left) "
                "-- do not disconnect; services are being restored",
                file=sys.stderr,
            )
            next_print = now + _PROGRESS_PRINT_INTERVAL_S
        time.sleep(2.0)


# --- in-flight guard (round-4 coordinator addendum, v3.1.9.8) ----------------
#
# A restart this week killed her in-flight request mid-stream. /health/full
# (v3.1.9.8 only -- v3.1.9 does not report this at all) carries
# `requests_in_flight`; before `zl fix`/`zl journal repair` stop anything,
# wait for it to settle at 0 rather than stopping compactor/openwebui out
# from under a live reply. Must be called from INSIDE the protected region,
# before the first stop -- a hangup/Ctrl-C during this wait is exactly the
# window ProtectedRegion exists to survive, the same as the stop/restart
# phases it already covers.

_INFLIGHT_POLL_INTERVAL_S = 1.0
_INFLIGHT_QUIET_S = 5.0  # must read 0 for about this long before proceeding -- not just one lucky poll
_INFLIGHT_TIMEOUT_S = 300.0  # ~5 minutes, then refuse rather than wait forever


def _extract_inflight(doc: object) -> int | None:
    """Pulls `request_tracing.requests_in_flight` out of a `/health/full`
    payload -- the REAL v3.1.9.8 shape (nested under `request_tracing`,
    read.py from `reqctx.in_flight_stats()`; see health.py:2290). Returns
    None if the field is absent, malformed, or `request_tracing` itself is
    an error shape (`{"error": ...}`) -- all of which must degrade the
    same way as "not reported at all", never as "0 in flight".
    """
    if not isinstance(doc, dict):
        return None
    rt = doc.get("request_tracing")
    if not isinstance(rt, dict):
        return None
    n = rt.get("requests_in_flight")
    return n if isinstance(n, int) else None


def _is_connection_refused(error: str | None) -> bool:
    """True only for "the compactor process is not there to answer" --
    e.g. ECONNREFUSED. A bare timeout is NOT connection-refused: the
    compactor may well be up and simply slow/overloaded, so it must not
    be treated as "safe to proceed" the way an outright refusal is.
    """
    if not error:
        return False
    low = error.lower()
    return "connection refused" in low or "econnrefused" in low


def wait_for_no_inflight_requests(*, force: bool, health_url: str | None = None) -> tuple[bool, str]:
    """Returns `(proceed, message)`. `proceed=True` means it is safe to go
    on: `request_tracing.requests_in_flight` settled at 0, `--force-busy`
    was given (skips the wait outright), the field is simply ABSENT on
    this compactor version (v3.1.9 does not report it at all -- this must
    never block on a field that does not exist, only degrade gracefully),
    or the compactor flatly refused the connection (nothing there to have
    a request in flight). `proceed=False` means REFUSE: nothing has been
    stopped yet, the caller's own exit code convention applies (1,
    refusal).

    A timeout or a non-200 response WITHOUT a parseable `requests_in_flight`
    (5xx, malformed body) does NOT fail open: it is treated as "busy,
    status unknown" and retried until the overall wait deadline, same as
    an actual busy count -- only an explicit 200-with-absent-field or an
    explicit connection-refusal may fail open early. A 5xx response that
    DOES carry a parseable `requests_in_flight` (a degraded compactor that
    still reports it) is honored like any other reading.
    """
    url = (health_url or podenv.COMPACTOR_URL) + "/health/full"
    status, doc, error = http_get_json(url, timeout=5.0)
    n = _extract_inflight(doc)

    if status == 200 and n is None:
        return True, "requests_in_flight not reported by this compactor -- continuing without the wait"
    if status == 0 and _is_connection_refused(error):
        return True, "compactor connection refused -- assuming nothing in flight"
    if force:
        return True, "requests_in_flight present but --force-busy was given -- skipping the wait"
    if n is not None and n <= 0:
        return True, "requests_in_flight is 0 -- nothing in flight"

    safe_print("zl: waiting for her in-flight request(s) to finish before stopping anything ...", file=sys.stderr)
    deadline = time.monotonic() + _INFLIGHT_TIMEOUT_S
    quiet_since: float | None = None
    while True:
        status, doc, error = http_get_json(url, timeout=5.0)
        n = _extract_inflight(doc)
        now = time.monotonic()
        if status == 200 and n is None:
            return True, "requests_in_flight not reported by this compactor -- continuing without the wait"
        if status == 0 and _is_connection_refused(error):
            return True, "compactor connection refused -- assuming nothing in flight"
        if n is not None and n <= 0:
            if quiet_since is None:
                quiet_since = now
            if now - quiet_since >= _INFLIGHT_QUIET_S:
                return True, "requests_in_flight settled to 0 -- proceeding"
        else:
            # busy with a real count, OR a timeout/5xx/malformed read we
            # cannot trust as "nothing in flight" -- neither resets the
            # quiet window early nor fails open; both just keep polling.
            quiet_since = None
        if now >= deadline:
            status_word = n if n is not None else "unknown (compactor health check unreliable)"
            return False, (
                f"requests_in_flight is still {status_word} after "
                f"{_INFLIGHT_TIMEOUT_S:.0f}s -- refusing to stop anything with her request possibly still "
                "in flight (pass --force-busy to override). Round-6 LOW: a known pre-fix v3.1.9.8 leak can "
                "leave this stuck above 0 with nothing actually in flight -- check "
                "/health/full request_tracing.requests_in_flight by hand before assuming this refusal is real."
            )
        time.sleep(_INFLIGHT_POLL_INTERVAL_S)


def restart_after_apply(
    was_running: list[str], *, openwebui_ready_timeout: float = _OPENWEBUI_READY_TIMEOUT_S
) -> dict[str, bool]:
    """Starts `was_running` again and verifies each came back.

    If `openwebui` is one of them, it is started ALONE FIRST and its own
    `/api/config` is polled for HTTP 200 before anything else is started
    -- RUNBOOK_CHAT_TREE.md:148 and RUNBOOK_DB_JOURNAL.md:216-229 ("Start
    OpenWebUI alone first" / "Wait until this prints 200" / "Then start
    the rest"), and explicitly under "Never do these": "Start OpenWebUI,
    the compactor and the backup daemon all at once right after a
    recovery." A database that just had surgery done on it should answer
    its own health check before anything else starts putting load on it.

    Returns {name: came_back_running}; the caller must report any False
    loudly, per the brief.
    """
    if not was_running:
        return {}
    rest = [n for n in was_running if n != "openwebui"]
    if "openwebui" in was_running:
        supervisorctl_start(["openwebui"])
        if not _wait_for_openwebui_http_ok(openwebui_ready_timeout):
            safe_print(
                f"zl: WARNING: openwebui did not answer 200 on {_openwebui_ready_url()} within "
                f"{openwebui_ready_timeout:.0f}s -- starting the rest anyway, but check it by hand",
                file=sys.stderr,
            )
    if rest:
        supervisorctl_start(rest)
    time.sleep(1.0)
    status = supervisorctl_status()
    return {n: status.get(n) == "RUNNING" for n in was_running}


def report_restart_failures(came_back: dict[str, bool]) -> bool:
    """Round-6 HIGH: every caller of `restart_after_apply` used to print a
    per-service WARNING for one that did not come back, but then still
    told `region.finish()` `restored=True` unconditionally and kept
    whatever exit code the wrapped script itself returned -- so a run that
    genuinely applied cleanly (child exit 0) but left, say, `openwebui`
    down afterward still printed "services are restored" and exited 0.
    An operator relying on the exit code (that is the whole point of
    having one) had no way to notice short of reading the WARNING text.

    Returns `True` only when every service in `came_back` is back RUNNING.
    Callers MUST: (1) pass this return value as `region.finish()`'s own
    `restored=` argument -- never hardcode `restored=True` once this
    exists -- and (2) force their own exit code nonzero (e.g. `code = 1`)
    when this returns `False`, even if the wrapped script's own exit code
    was 0, so "restored" and "exit 0" never both lie at once. Prints the
    exact `supervisorctl start ...` command for whichever services are
    still down, so there is nothing to look up by hand.
    """
    down = sorted(svc for svc, ok in came_back.items() if not ok)
    if not down:
        return True
    safe_print(
        f"zl: WARNING: {down} did NOT come back RUNNING -- services are NOT fully restored. "
        f"Run this by hand: supervisorctl start {' '.join(down)}",
        file=sys.stderr,
    )
    return False


# --- HTTP (stdlib only -- no `requests`) --------------------------------------


def http_get_json(url: str, timeout: float = 5.0) -> tuple[int, dict | None, str | None]:
    """Returns (status_code, parsed_json_or_None, error_or_None). Never
    raises -- a collector must record `error` per panel, not crash.

    Round-6 LOW: `http.client.HTTPException` (its own base class, which
    covers `IncompleteRead` -- a connection that closes partway through a
    chunked body, plausible on a compactor mid-restart) used to be
    uncaught here -- `urllib.request.urlopen`/`resp.read()` can raise it
    directly, not wrapped in `urllib.error.URLError`, and every caller of
    this function (the in-flight guard chief among them) depends on it
    NEVER raising.
    """
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 (localhost admin endpoint)
            status = resp.status
            body = resp.read()
    except urllib.error.HTTPError as e:
        status = e.code
        try:
            body = e.read()
        except Exception:
            body = b""
    except (urllib.error.URLError, OSError, TimeoutError, http.client.HTTPException) as e:
        return 0, None, str(e)
    try:
        return status, json.loads(body.decode("utf-8", "replace")), None
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        return status, None, f"not valid JSON: {e}"


# --- env redaction (privacy hard rule, brief) ---------------------------------

_SECRET_KEY = re.compile(r"(TOKEN|KEY|SECRET|PASS|AUTH|COOKIE|CREDENTIAL|^HF_)", re.IGNORECASE)
_SECRET_VALUE_SHAPES = (
    re.compile(r"^hf_[A-Za-z0-9]{10,}$"),
    re.compile(r"^sk-[A-Za-z0-9_-]{10,}$"),
    re.compile(r"^[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}$"),  # JWT-shaped
    re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://[^/\s@]+:[^/\s@]+@"),  # URL with userinfo (user:pass@)
)
REDACTED = "<redacted>"

# Known-safe keys, shown (after substring scrubbing) instead of redacted
# whole. review MEDIUM: a blocklist of secret-LOOKING keys misses secrets
# under an innocuous name; an allowlist of known-safe keys is the safer
# default. Anything not here (and not caught by _SECRET_KEY/_SECRET_VALUE_SHAPES
# above) is redacted outright -- default DENY, not default allow.
_ALLOWLIST_KEYS = frozenset({
    "PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "LC_ALL", "TERM", "PWD",
    "SHLVL", "HOSTNAME", "TZ", "OLDPWD",
    "DATA_DIR", "LOG_DIR", "ZL_HOME", "ZL_REPO", "ZL_REPO_URL", "ZL_SHADOW_PORT",
    "VLLM_PORT", "COMPACTOR_PORT", "OPENWEBUI_PORT", "STT_PORT", "TTS_PORT",
    "COMPACTOR_STORAGE_ROOT", "COMPACTOR_BACKUP_DIR", "COMPACTOR_BACKUP_RETAIN",
    "WEBUI_DB_LOCAL", "DATABASE_URL", "HF_HOME", "MODEL_REPO", "VLLM_EXTRA_ARGS",
    "RUNPOD_POD_ID", "RUNPOD_PUBLIC_IP", "RUNPOD_TCP_PORT_22", "RUNPOD_GPU_COUNT",
    "RUNPOD_CPU_COUNT", "RUNPOD_MEM_GB", "NVIDIA_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
    "PYTHONPATH", "PYTHONDONTWRITEBYTECODE", "PYTHONUNBUFFERED",
})

# Substring scrub, applied to every ALLOWLISTED value: an otherwise-safe
# key (e.g. ZL_REPO_URL, VLLM_EXTRA_ARGS, DATABASE_URL) can still carry a
# secret embedded inside it. Each pattern redacts just the matched span,
# not the whole value, so the rest of the (non-secret) value stays useful
# for diagnosis.
_VALUE_SCRUB_PATTERNS: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"), REDACTED),  # GitHub PAT/OAuth/user/server/refresh token
    (re.compile(r"github_pat_[A-Za-z0-9_]{20,}"), REDACTED),  # fine-grained GitHub PAT
    (re.compile(r"AKIA[0-9A-Z]{12,}"), REDACTED),  # AWS access key id
    (re.compile(r"rpa_[A-Za-z0-9]{10,}"), REDACTED),  # RunPod API key
    (re.compile(r"sk-[A-Za-z0-9_-]{10,}"), REDACTED),  # OpenAI-shaped secret key
    (re.compile(r"hf_[A-Za-z0-9]{10,}"), REDACTED),  # Hugging Face token
    (re.compile(r"Bearer\s+\S+", re.IGNORECASE), f"Bearer {REDACTED}"),
    (re.compile(r"[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"), REDACTED),  # JWT-shaped
    # URL userinfo, INCLUDING a bare token with no colon (https://ghp_xxx@host):
    (re.compile(r"([a-zA-Z][a-zA-Z0-9+.-]*://)[^/\s@]+@"), rf"\1{REDACTED}@"),
    (re.compile(r"((?:\?|&)(?:token|api_key)=)[^&\s]+", re.IGNORECASE), rf"\1{REDACTED}"),
    (re.compile(r"(--api-key[= ])\S+"), rf"\1{REDACTED}"),
)


def _scrub_value(value: str) -> str:
    for pattern, repl in _VALUE_SCRUB_PATTERNS:
        value = pattern.sub(repl, value)
    return value


def redact_env(mapping: dict[str, str]) -> dict[str, str]:
    """Privacy hard rule (brief), tightened per review MEDIUM:
    1. Key matches the secret-name pattern -> whole value redacted, always.
    2. Otherwise the value matches a WHOLE-value secret shape (hf_..., a
       bare sk-..., a JWT, a colon-form userinfo URL) -> redacted whole,
       whatever the key is named.
    3. Otherwise: only an ALLOWLISTED key is shown at all, and even then
       its value is first scrubbed for secret-shaped SUBSTRINGS (a GitHub/
       AWS/RunPod/OpenAI/HF token, a Bearer value, a JWT, a URL with
       userinfo including a bare token with no colon, a `?token=`/
       `?api_key=` query parameter, or `--api-key X` inside an args
       string).
    4. A key that is neither secret-named nor allowlisted is redacted
       outright -- default DENY.
    """
    out = {}
    for k, v in mapping.items():
        if _SECRET_KEY.search(k) or any(p.search(v) for p in _SECRET_VALUE_SHAPES):
            out[k] = REDACTED
        elif k in _ALLOWLIST_KEYS:
            out[k] = _scrub_value(v)
        else:
            out[k] = REDACTED
    return out


def proc1_environ() -> dict[str, str]:
    """The real supervisord environment (pid 1) -- see podenv._proc1_environ,
    which is private to that module; this is the bundle's own copy of the
    same read (podenv's own helper is not exported), used only to MERGE with
    this process's environment for `zl bundle`'s env dump, per the brief:
    "Env source = this process env merged with /proc/1/environ."
    """
    try:
        raw = Path("/proc/1/environ").read_bytes()
    except OSError:
        return {}
    out: dict[str, str] = {}
    for part in raw.split(b"\0"):
        key, sep, value = part.partition(b"=")
        if sep:
            out[key.decode("utf-8", "replace")] = value.decode("utf-8", "replace")
    return out


def merged_environ() -> dict[str, str]:
    """This process's own env (a normal shell's PATH/HOME/TERM/SSH_*), with
    pid 1's -- supervisord's, the real services' -- environment OVERLAID on
    top, so pid 1 wins any key both define.

    Direction matters (review B3): an SSH or Web Terminal shell is missing
    exactly the service-config keys the wrapped scripts read directly from
    `os.environ` (WEBUI_DB_LOCAL, DATABASE_URL, COMPACTOR_BACKUP_RETAIN,
    PUBLIC_KEY, ...) -- podenv.py itself already falls back to pid 1 for
    its OWN reads, but a child process run by this bundle does not get that
    fallback unless we build its env this way. The shell's env stays the
    BASE (so PATH/HOME/TERM/SSH_CONNECTION and anything pid 1 does not set
    are still present and normal), but for every key pid 1 itself sets, pid
    1's value is authoritative -- that is what "the service values must
    win" means here.
    """
    merged = dict(os.environ)
    merged.update(proc1_environ())  # pid 1's (the real services') values win on conflict
    return merged


# --- request-id filtering (v3.1.9.8 request tracing) -------------------------

# v3.1.9.8 tags every compactor log line belonging to one request with
# `req=<hex id>`, and internal vLLM calls with child ids like
# `req=<hex>.sum1`, `.facts`, `.dedup` (coordinator note, round 3 addendum).
# Shared by `zl logs --req` and `zl bundle --req` so the two never drift on
# what counts as "this request's own lines" (memory.md: two copies of one
# rule is this codebase's most expensive recurring defect).
_REQ_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class InvalidReqId(ValueError):
    """Raised by `req_filter_pattern` when `req_id` is not a safe,
    identifier-shaped token -- refused outright rather than built into a
    regex or a filename unchecked.
    """


def req_filter_pattern(req_id: str) -> re.Pattern:
    """A line MATCHES if it carries this exact request id, OR a CHILD id
    of it (`<req_id>.sum1`, `<req_id>.facts`, ...) -- never a different
    request that merely starts with the same prefix.

    Round-4 LOW: real ids allow hyphens WITHIN a single id
    (`reqctx._ID_RE` in zl-v3198: `[A-Za-z0-9][A-Za-z0-9._-]{0,63}`), so a
    trailing `\\b` word boundary is the wrong check here -- `-` is a
    non-word character to `\\b`, so the boundary between "c" (a word
    char) and "-" (not) counts as a match, and `req=abc` used to match
    right inside `req=abc-def`, a completely DIFFERENT id that merely
    shares a prefix. Fixed with a negative lookahead for any character
    that could CONTINUE an id (letter/digit/`.`/`_`/`-`) instead: only an
    EXACT id, or `id` + a `.child` suffix, can match -- `id-` (more id
    characters immediately after) cannot.

    Log-format note: this matches the PLAIN-text log line shape
    (`req=<id>`); `COMPACTOR_LOG_FORMAT=json` structured lines are not
    matched by this pattern at all (out of scope this round -- `zl bundle
    --req` still captures a JSON-format pod's full logs even though this
    per-line filter does not narrow them).
    """
    if not req_id or not _REQ_ID_RE.match(req_id):
        raise InvalidReqId(f"{req_id!r} is not a safe request id (expected [A-Za-z0-9._-]+)")
    escaped = re.escape(req_id)
    return re.compile(rf"\breq={escaped}(?:\.[A-Za-z0-9_-]+)?(?![A-Za-z0-9._-])")


def human_bytes(n: float) -> str:
    for unit in ("B", "K", "M", "G", "T"):
        if abs(n) < 1024.0:
            return f"{n:.1f}{unit}" if unit != "B" else f"{int(n)}{unit}"
        n /= 1024.0
    return f"{n:.1f}P"
