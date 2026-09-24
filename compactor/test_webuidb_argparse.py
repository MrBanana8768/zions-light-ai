"""D6 (findings.md / review1-v3197-65ea196, H-3): webuidb.py's CLI had NO
unit test at all before this file. That gap is exactly why mutants W10
("parse_known_args" instead of "parse_args" — unknown flags, including
`--help` before the fix, fall through to the bare `else: sync_loop()`) and
W11 (the `--force` without `--sync-once` check removed) both SURVIVED the
round-1 hostile review's mutation run.

EVERY CASE HERE RUNS THE REAL `webuidb.py` AS A SUBPROCESS, under
`timeout` — not by importing `_build_arg_parser()` and calling
`.parse_args()` in-process. The actual defect this guards against is a
process that does not exit: an accepted-when-it-should-not-be argument
used to fall through to `sync_loop()`, an infinite loop with no exit
condition of its own (see test_webuidb_publish_guards.py's own
`_StopLoop` sentinel — sync_loop needs an injected exception to stop even
in-process). An in-process `parse_args()` call can raise/return without
ever proving the REAL dispatch in `if __name__ == "__main__":` still
takes the daemon branch on a mutant; a hung subprocess killed by `timeout`
is the one signal that cannot be faked by a change that only moves where
the daemon starts.

    python test_webuidb_argparse.py
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_ROOT = Path(tempfile.mkdtemp(prefix="webuidb-argparse-"))
_HERE = Path(__file__).resolve().parent
_SCRIPT = _HERE / "webuidb.py"

FAILED: list[str] = []


def check(cond, label):
    print(f"  {'ok  ' if cond else 'FAIL'} {label}")
    if not cond:
        FAILED.append(label)


def run(args, timeout_s=5):
    """Runs the real webuidb.py as a subprocess with a clean, isolated
    LOCAL/SNAPSHOT/QUARANTINE environment, under a wall-clock timeout.
    Returns (returncode_or_'TIMEOUT', combined stdout+stderr)."""
    env = dict(os.environ)
    env["WEBUI_LOCAL_DB"] = str(_ROOT / "local.db")
    env["WEBUI_SNAPSHOT_DB"] = str(_ROOT / "snap.db")
    env["WEBUI_DB_QUARANTINE"] = str(_ROOT / "q")
    try:
        p = subprocess.run(
            [sys.executable, str(_SCRIPT), *args],
            capture_output=True, text=True, timeout=timeout_s, env=env,
        )
        return p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"")
        if isinstance(out, bytes):
            out = out.decode("utf-8", "replace")
        err = (e.stderr or b"")
        if isinstance(err, bytes):
            err = err.decode("utf-8", "replace")
        return "TIMEOUT", out + err


print("[D6] --help prints usage and exits 0 - does NOT start the daemon")
rc, out = run(["--help"])
check(rc == 0, f"exit code is 0 (got {rc!r})")
check(
    "usage" in out.lower(),
    f"and it actually printed usage text, not the daemon's own startup "
    f"log line (got: {out[:200]!r})",
)
check(
    "webui.db sync:" not in out,
    "and definitely did not print sync_loop's own startup banner",
)

print("[D6/W10] an UNKNOWN flag exits 2 - does NOT fall through to the "
      "daemon (this is the mutant: parse_known_args() instead of "
      "parse_args() lets an unrecognised flag through with the daemon as "
      "the silent default)")
rc, out = run(["--this-flag-does-not-exist"])
check(
    rc == 2,
    f"exit code is 2, argparse's own usage-error code (got {rc!r}, "
    f"output: {out[-300:]!r}) - 'TIMEOUT' here means the daemon loop "
    f"started and never exited, exactly what W10 does",
)

print("[D6/W11] --force ALONE (without --sync-once) exits 2 - --force is "
      "not a mode of its own (this is the mutant: the `if _args.force and "
      "not _args.sync_once` guard removed, so --force alone silently "
      "starts the daemon instead of being rejected)")
rc, out = run(["--force"])
check(
    rc == 2,
    f"exit code is 2 (got {rc!r}, output: {out[-300:]!r}) - 'TIMEOUT' "
    f"here means --force alone fell through to sync_loop()",
)
check(
    "--force is only meaningful together with --sync-once" in out,
    f"and the error names the actual reason (got: {out[-300:]!r})",
)

print("[D6] mutually exclusive modes together exit 2 (argparse's own "
      "mutually-exclusive group, unchanged by D6 but worth pinning "
      "alongside it)")
rc, out = run(["--restore", "--status"])
check(rc == 2, f"exit code is 2 (got {rc!r})")

print("[D6] CONTROL: --sync-once (with no local database yet) is a "
      "VALID invocation that exits promptly and cleanly - the fix must "
      "not turn every argument into a rejection")
rc, out = run(["--sync-once"])
check(
    rc == 0,
    f"exit code is 0 (got {rc!r}, output: {out[-300:]!r}) - 'no local "
    f"database yet' is an ordinary skip, not a usage error",
)
check("'skipped': 'no local database yet'" in out, f"and says why (got: {out!r})")

print("[D6] CONTROL: --sync-once --force (the DB-MOVE-RUNBOOK's own "
      "documented final-sync command) is accepted, not rejected by the "
      "W11 guard")
rc, out = run(["--sync-once", "--force"])
check(rc == 0, f"exit code is 0 (got {rc!r}, output: {out[-300:]!r})")

print("[D6] CONTROL: --status (no local database, no snapshot) is a "
      "valid, prompt invocation")
rc, out = run(["--status"])
check(rc == 0, f"exit code is 0 (got {rc!r}, output: {out[-300:]!r})")

print("[D6] CONTROL: no arguments at all starts the daemon - this is the "
      "ONE case that is SUPPOSED to loop forever, proving the timeout "
      "above is actually discriminating between 'daemon started' and "
      "'usage error', not just universally timing out")
rc, out = run([], timeout_s=2)
check(
    rc == "TIMEOUT",
    f"a bare invocation with no flags DOES hang under the same timeout "
    f"that catches W10/W11 (got {rc!r}) - proving those tests' exit-code "
    f"checks are real discriminators, not just 'the process happened to "
    f"exit fast for an unrelated reason'",
)

shutil.rmtree(_ROOT, ignore_errors=True)
if FAILED:
    print(f"\n{len(FAILED)} check(s) FAILED")
    sys.exit(1)
print("All webui.db argparse checks passed.")
