"""compactor/test_cloudflared_run.py — v3.1.9.9 Cloudflare Tunnel, task 10:
unit test for docker/cloudflared-run.sh, the [program:cloudflared] command.

Runs the REAL script under `bash` (not a re-implementation), with:
  - a fake HOME/PATH sandbox so nothing here can touch a real pod or her data;
  - a stub `cloudflared` executable on PATH that, instead of dialing
    anything, dumps its own argv and environment to a file and exits 0 —
    this is how we prove the token reaches cloudflared ONLY through the
    environment and never on the command line (which would be visible in
    `ps`/supervisord's own logging) without needing a real network or a
    real Cloudflare account;
  - only OBVIOUS FAKE token strings (never a real one — there is no real
    one in this repo/session to begin with).

Covers:
  [1] no CLOUDFLARE_TUNNEL_TOKEN (unset) -> exits 0, logs the exact
      "no CLOUDFLARE_TUNNEL_TOKEN; tunnel disabled" line, never invokes the
      stub cloudflared at all.
  [2] CLOUDFLARE_TUNNEL_TOKEN set to empty string -> same as [1] (the
      script's own `-z` check treats unset and empty identically).
  [3] a fake token set -> execs the stub cloudflared; the stub's recorded
      argv contains NO occurrence of the token string anywhere in any
      argument; the stub's recorded environment DOES contain
      TUNNEL_TOKEN=<the fake token>; CLOUDFLARE_TUNNEL_TOKEN is NOT present
      in the stub's recorded environment (unset before exec, per the
      script's own comment).
  [4] the script's own stdout in the enabled case never contains the raw
      token string either (only ever "set"/"disabled", per its docstring).

    python test_cloudflared_run.py
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docker" / "cloudflared-run.sh"

FAILED = []


def check(cond, label):
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}")
        FAILED.append(label)


if shutil.which("bash") is None:
    print("SKIP: no `bash` on PATH.")
    sys.exit(3)

if not SCRIPT.is_file():
    print(f"FAIL: {SCRIPT} does not exist")
    sys.exit(1)

# A conspicuously fake token. Never a real one; there is no real one here.
FAKE_TOKEN = "FAKE-TEST-TOKEN-NOT-REAL-do-not-use-1234567890"

STUB_CLOUDFLARED = """#!/bin/bash
# Test stub standing in for the real cloudflared binary. Records its own
# argv and full environment, then exits 0 — never dials anything real.
{
    printf 'ARGV:'
    for a in "$@"; do printf ' [%s]' "$a"; done
    printf '\\n'
    echo '---ENV---'
    env
} > "$CLOUDFLARED_STUB_RECORD"
exit 0
"""


def _run(env_overrides: dict, record_path: Path) -> subprocess.CompletedProcess:
    tmp_bin = record_path.parent / "bin"
    tmp_bin.mkdir(exist_ok=True)
    stub = tmp_bin / "cloudflared"
    stub.write_text(STUB_CLOUDFLARED)
    stub.chmod(0o755)

    env = {
        "PATH": f"{tmp_bin}:/usr/bin:/bin",
        "HOME": str(record_path.parent),
        "CLOUDFLARED_STUB_RECORD": str(record_path),
    }
    env.update(env_overrides)

    return subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True, text=True, timeout=10,
        env=env,
    )


def test_no_token_unset_exits_0_and_logs_disabled():
    print("\n[test] CLOUDFLARE_TUNNEL_TOKEN unset -> exit 0, logs 'disabled', never runs cloudflared")
    with tempfile.TemporaryDirectory(prefix="zla_cfd_test_") as d:
        record = Path(d) / "record.txt"
        proc = _run({}, record)
        check(proc.returncode == 0, f"exit code 0 (got {proc.returncode})")
        check("no CLOUDFLARE_TUNNEL_TOKEN; tunnel disabled" in proc.stdout,
              "exact 'disabled' log line present")
        check(not record.exists(), "stub cloudflared was never invoked (no record file written)")


def test_no_token_empty_string_same_as_unset():
    print("\n[test] CLOUDFLARE_TUNNEL_TOKEN='' (empty) behaves identically to unset")
    with tempfile.TemporaryDirectory(prefix="zla_cfd_test_") as d:
        record = Path(d) / "record.txt"
        proc = _run({"CLOUDFLARE_TUNNEL_TOKEN": ""}, record)
        check(proc.returncode == 0, f"exit code 0 (got {proc.returncode})")
        check("no CLOUDFLARE_TUNNEL_TOKEN; tunnel disabled" in proc.stdout,
              "exact 'disabled' log line present")
        check(not record.exists(), "stub cloudflared was never invoked")


def test_fake_token_execs_cloudflared_token_only_in_env():
    print("\n[test] a fake token -> execs cloudflared; token is in ENV only, NEVER in argv")
    with tempfile.TemporaryDirectory(prefix="zla_cfd_test_") as d:
        record = Path(d) / "record.txt"
        proc = _run({"CLOUDFLARE_TUNNEL_TOKEN": FAKE_TOKEN}, record)
        check(proc.returncode == 0, f"exit code 0 from the stub (got {proc.returncode})")
        check(record.exists(), "stub cloudflared WAS invoked")

        recorded = record.read_text()
        argv_line, _, env_block = recorded.partition("---ENV---\n")

        check(FAKE_TOKEN not in argv_line,
              "the token string does not appear anywhere in cloudflared's argv")
        check("tunnel" in argv_line and "run" in argv_line and "--no-autoupdate" in argv_line,
              "argv carries the expected cloudflared subcommand/flags")
        check("--metrics" in argv_line and "127.0.0.1:8082" in argv_line,
              "argv carries a loopback-only --metrics address")

        env_lines = env_block.splitlines()
        check(f"TUNNEL_TOKEN={FAKE_TOKEN}" in env_lines,
              "TUNNEL_TOKEN=<fake token> present in cloudflared's environment")
        check(not any(line.startswith("CLOUDFLARE_TUNNEL_TOKEN=") for line in env_lines),
              "CLOUDFLARE_TUNNEL_TOKEN is NOT present in cloudflared's environment (unset before exec)")

        check(FAKE_TOKEN not in proc.stdout,
              "the script's own stdout never echoes the raw token value")
        check("CLOUDFLARE_TUNNEL_TOKEN is set; starting cloudflared" in proc.stdout,
              "logs that it is starting, without the value")


def _all_tests():
    return [
        test_no_token_unset_exits_0_and_logs_disabled,
        test_no_token_empty_string_same_as_unset,
        test_fake_token_execs_cloudflared_token_only_in_env,
    ]


if __name__ == "__main__":
    for t in _all_tests():
        t()
    if FAILED:
        print(f"\n{len(FAILED)} FAILURE(S): {FAILED}")
        sys.exit(1)
    print("\nAll cloudflared-run.sh tests passed.")
