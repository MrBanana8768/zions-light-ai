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
    one in this repo/session to begin with). A "well-shaped" fake token
    below is base64 of a small JSON object with keys a/t/s, exactly the
    shape a real tunnel token has, built from clearly-fake hex/uuid
    filler — never anything that could be mistaken for or derived from a
    real credential.
  - every timing knob (throttle window, backoff base/cap, the
    invalid-shape sleep, the stamp-file directory) is read from an
    env var the script itself supports overriding (see cloudflared-run.sh's
    own comments) precisely so this suite never has to wait out a real
    30-300s sleep.

Covers:
  [1] no CLOUDFLARE_TUNNEL_TOKEN (unset) -> exits 0, logs the exact
      "no CLOUDFLARE_TUNNEL_TOKEN; tunnel disabled" line, never invokes the
      stub cloudflared at all.
  [2] CLOUDFLARE_TUNNEL_TOKEN set to empty string -> same as [1] (the
      script's own `-z` check treats unset and empty identically).
  [3] a well-shaped fake token set -> execs the stub cloudflared; the
      stub's recorded argv contains NO occurrence of the token string
      anywhere in any argument; the stub's recorded environment DOES
      contain TUNNEL_TOKEN=<the fake token>; CLOUDFLARE_TUNNEL_TOKEN is NOT
      present in the stub's recorded environment (unset before exec, per
      the script's own comment).
  [4] the script's own stdout in the enabled case never contains the raw
      token string either (only ever "set"/"disabled", per its docstring).
  [5]-[9] rc-review MEDIUM fix, shape validation: garbage -> "(not
      base64)"; valid base64 but not a JSON object -> "(not JSON)"; valid
      JSON object missing a/t/s -> "(missing keys)"; a whole pasted
      `cloudflared service install eyJ...` line -> the eyJ... part is
      extracted and used, never refused; quotes/surrounding whitespace are
      stripped; a decoded JSON body with its own leading/trailing
      whitespace (still valid JSON) is tolerated, not misread as "not
      JSON" — this one is the exact shape a real reviewer probe used
      (faketoken-nopad.txt) and is what caught a subshell bug in the first
      draft (the validator set its result in globals from inside a
      `$(...)` substitution, which silently discarded them).
  [10] on an invalid shape, no 4+ character substring of a distinctive
      fake token appears ANYWHERE in the script's combined stdout+stderr —
      the log line must only ever name one of the fixed reason words,
      never any part of the value itself.
  [11]-[12] rc-review MEDIUM fix, restart throttle: a second spawn well
      within the throttle window sleeps for at least the base backoff
      before reaching (the stub) cloudflared; a spawn AFTER the window has
      elapsed does not sleep at all.
  [13] the invalid-shape sleep itself is honored (and overridable).

    python test_cloudflared_run.py
"""

import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
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


def _fake_token(a="a" * 32, t="00000000-0000-4000-8000-000000000000", s="s" * 44, *, pad=True):
    """A never-real, well-SHAPED fake tunnel token: base64 of
    {"a": ..., "t": ..., "s": ...}. `pad=False` strips the base64 padding
    (the script must tolerate that too)."""
    raw = json.dumps({"a": a, "t": t, "s": s}).encode()
    enc = base64.b64encode(raw).decode()
    if not pad:
        enc = enc.rstrip("=")
    return enc


# A conspicuously fake, but correctly-SHAPED, token. Never a real one;
# there is no real one here.
FAKE_TOKEN = _fake_token()

# A distinctive marker embedded in a token's own fields, used only for the
# "no substring leaks" check [10] — chosen to be extremely unlikely to
# collide with any of the four fixed reason words or ordinary log text.
_LEAK_MARKER = "zQ9xLeakMarkerNeverInLogsZ9q"

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

# Fails immediately (nonzero), the same shape as a shape-valid-but-
# Cloudflare-rejected token — used only by the throttle tests [11]/[12].
STUB_CLOUDFLARED_FAILS = """#!/bin/bash
exit 7
"""

# Fast test defaults for every timing knob the script supports overriding.
_FAST_TIMING = {
    "CLOUDFLARED_RUN_THROTTLE_WINDOW_S": "5",
    "CLOUDFLARED_RUN_BACKOFF_BASE_S": "2",
    "CLOUDFLARED_RUN_BACKOFF_CAP_S": "4",
    "CLOUDFLARED_RUN_INVALID_TOKEN_SLEEP_S": "1",
}


def _run(env_overrides: dict, record_path: Path, *, stub: str = STUB_CLOUDFLARED,
          state_dir: Path | None = None, timeout: float = 10) -> subprocess.CompletedProcess:
    tmp_bin = record_path.parent / "bin"
    tmp_bin.mkdir(exist_ok=True)
    stub_path = tmp_bin / "cloudflared"
    stub_path.write_text(stub)
    stub_path.chmod(0o755)

    if state_dir is None:
        state_dir = record_path.parent / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    env = {
        "PATH": f"{tmp_bin}:/usr/bin:/bin",
        "HOME": str(record_path.parent),
        "CLOUDFLARED_STUB_RECORD": str(record_path),
        "CLOUDFLARED_RUN_STATE_DIR": str(state_dir),
    }
    env.update(_FAST_TIMING)
    env.update(env_overrides)

    return subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True, text=True, timeout=timeout,
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
    print("\n[test] a well-shaped fake token -> execs cloudflared; token is in ENV only, NEVER in argv")
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


def test_garbage_token_refused_not_base64():
    print("\n[test] plain garbage -> refused, reason '(not base64)', never execs cloudflared")
    with tempfile.TemporaryDirectory(prefix="zla_cfd_test_") as d:
        record = Path(d) / "record.txt"
        proc = _run({"CLOUDFLARE_TUNNEL_TOKEN": "not-even-close-to-base64-json!!!"}, record)
        check(proc.returncode == 1, f"exit code 1 (got {proc.returncode})")
        check("does not look like a tunnel token (not base64); not starting" in proc.stdout,
              "exact reason '(not base64)' logged")
        check(not record.exists(), "stub cloudflared was never invoked")


def test_valid_base64_not_json_refused():
    print("\n[test] valid base64 of non-JSON text -> refused, reason '(not JSON)'")
    with tempfile.TemporaryDirectory(prefix="zla_cfd_test_") as d:
        record = Path(d) / "record.txt"
        bogus = base64.b64encode(b"just some plain text, not json at all").decode()
        proc = _run({"CLOUDFLARE_TUNNEL_TOKEN": bogus}, record)
        check(proc.returncode == 1, f"exit code 1 (got {proc.returncode})")
        check("does not look like a tunnel token (not JSON); not starting" in proc.stdout,
              "exact reason '(not JSON)' logged")
        check(not record.exists(), "stub cloudflared was never invoked")


def test_valid_json_missing_keys_refused():
    print("\n[test] valid base64 JSON object missing a/t/s -> refused, reason '(missing keys)'")
    with tempfile.TemporaryDirectory(prefix="zla_cfd_test_") as d:
        record = Path(d) / "record.txt"
        bogus = base64.b64encode(json.dumps({"x": 1, "y": 2}).encode()).decode()
        proc = _run({"CLOUDFLARE_TUNNEL_TOKEN": bogus}, record)
        check(proc.returncode == 1, f"exit code 1 (got {proc.returncode})")
        check("does not look like a tunnel token (missing keys); not starting" in proc.stdout,
              "exact reason '(missing keys)' logged")
        check(not record.exists(), "stub cloudflared was never invoked")


def test_pasted_install_line_extracts_token():
    print("\n[test] a whole pasted 'cloudflared service install eyJ...' line -> the token is extracted and used")
    with tempfile.TemporaryDirectory(prefix="zla_cfd_test_") as d:
        record = Path(d) / "record.txt"
        pasted = f"cloudflared service install {FAKE_TOKEN}"
        proc = _run({"CLOUDFLARE_TUNNEL_TOKEN": pasted}, record)
        check(proc.returncode == 0, f"exit code 0 (got {proc.returncode}); stdout={proc.stdout!r}")
        check(record.exists(), "stub cloudflared WAS invoked (extraction succeeded, not refused)")
        env_lines = record.read_text().partition("---ENV---\n")[2].splitlines()
        check(f"TUNNEL_TOKEN={FAKE_TOKEN}" in env_lines,
              "TUNNEL_TOKEN is exactly the extracted eyJ... part, not the whole pasted line")
        check(not any("install" in line for line in env_lines if line.startswith("TUNNEL_TOKEN=")),
              "the word 'install' from the pasted command never leaks into TUNNEL_TOKEN")


def test_quotes_and_surrounding_whitespace_tolerated():
    print("\n[test] surrounding quotes and whitespace around the token are stripped, not refused")
    with tempfile.TemporaryDirectory(prefix="zla_cfd_test_") as d:
        record = Path(d) / "record.txt"
        proc = _run({"CLOUDFLARE_TUNNEL_TOKEN": f'  "{FAKE_TOKEN}"  \n'}, record)
        check(proc.returncode == 0, f"exit code 0 (got {proc.returncode}); stdout={proc.stdout!r}")
        check(record.exists(), "stub cloudflared WAS invoked")
        env_lines = record.read_text().partition("---ENV---\n")[2].splitlines()
        check(f"TUNNEL_TOKEN={FAKE_TOKEN}" in env_lines,
              "TUNNEL_TOKEN has the quotes/whitespace stripped")


def test_unpadded_base64_tolerated():
    print("\n[test] base64 with padding stripped is tolerated (padding is re-derived)")
    with tempfile.TemporaryDirectory(prefix="zla_cfd_test_") as d:
        record = Path(d) / "record.txt"
        unpadded = _fake_token(pad=False)
        assert "=" not in unpadded
        proc = _run({"CLOUDFLARE_TUNNEL_TOKEN": unpadded}, record)
        check(proc.returncode == 0, f"exit code 0 (got {proc.returncode}); stdout={proc.stdout!r}")
        check(record.exists(), "stub cloudflared WAS invoked")


def test_decoded_json_with_its_own_whitespace_tolerated():
    print("\n[test] a decoded JSON body with its own leading/trailing whitespace is still valid, not '(not JSON)'")
    # This is the exact shape of the reviewer's faketoken-nopad.txt probe:
    # base64 of "  {...}" (literal leading spaces baked into the decoded
    # bytes, still syntactically valid JSON) — caught a subshell bug in the
    # first draft of the validator (globals set inside `$(...)` never
    # reached the caller).
    with tempfile.TemporaryDirectory(prefix="zla_cfd_test_") as d:
        record = Path(d) / "record.txt"
        raw = ("  " + json.dumps({"a": "a" * 32, "t": "t" * 8, "s": "s" * 44}) + "  ").encode()
        token = base64.b64encode(raw).decode()
        proc = _run({"CLOUDFLARE_TUNNEL_TOKEN": token}, record)
        check(proc.returncode == 0, f"exit code 0 (got {proc.returncode}); stdout={proc.stdout!r}")
        check(record.exists(), "stub cloudflared WAS invoked (not misread as '(not JSON)')")
        check("not JSON" not in proc.stdout, "never logged '(not JSON)' for this legitimately-whitespaced body")


def test_no_substring_of_token_leaks_on_invalid_shape():
    print("\n[test] on an invalid shape, no 4+ char substring of the value leaks into stdout/stderr")
    with tempfile.TemporaryDirectory(prefix="zla_cfd_test_") as d:
        record = Path(d) / "record.txt"
        # A garbage value that is itself invalid (contains '!' — not
        # base64), but distinctive enough that any leaked fragment would
        # be unmistakable.
        # Deliberately avoids any common English word (e.g. "token") that
        # could also appear in the script's own fixed log text and produce
        # a false-positive substring match against THIS test's own input
        # rather than an actual leak.
        distinctive = f"zzz-{_LEAK_MARKER}-qqq!!!"
        proc = _run({"CLOUDFLARE_TUNNEL_TOKEN": distinctive}, record)
        check(proc.returncode == 1, f"exit code 1 (got {proc.returncode})")
        combined = proc.stdout + proc.stderr
        leaked = [
            distinctive[i:i + 4]
            for i in range(len(distinctive) - 4 + 1)
            if distinctive[i:i + 4] in combined
        ]
        check(not leaked, f"no 4+ char substring of the fake token appears in output (found: {leaked[:5]})")
        check("(not base64); not starting" in proc.stdout, "the fixed reason word is logged instead")


def test_throttle_backs_off_within_window():
    print("\n[test] a second spawn within the throttle window sleeps at least the base backoff before starting cloudflared")
    with tempfile.TemporaryDirectory(prefix="zla_cfd_test_") as d:
        state_dir = Path(d) / "state"
        r1 = Path(d) / "rec1.txt"
        r2 = Path(d) / "rec2.txt"
        base_env = {"CLOUDFLARE_TUNNEL_TOKEN": FAKE_TOKEN}

        proc1 = _run(base_env, r1, stub=STUB_CLOUDFLARED_FAILS, state_dir=state_dir)
        check(proc1.returncode == 7, f"first spawn reaches (and exits like) the stub (got {proc1.returncode})")

        t0 = time.monotonic()
        proc2 = _run(base_env, r2, stub=STUB_CLOUDFLARED_FAILS, state_dir=state_dir)
        elapsed = time.monotonic() - t0
        check(proc2.returncode == 7, f"second spawn also reaches the stub (got {proc2.returncode})")
        base_backoff = float(_FAST_TIMING["CLOUDFLARED_RUN_BACKOFF_BASE_S"])
        check(elapsed >= base_backoff * 0.8,
              f"second spawn (within the throttle window) waited close to the base backoff ({elapsed:.2f}s >= ~{base_backoff}s)")
        check("backing off" in proc2.stdout, "logs that it is backing off")


def test_throttle_does_not_fire_after_window_elapses():
    print("\n[test] a spawn AFTER the throttle window has elapsed does not sleep")
    with tempfile.TemporaryDirectory(prefix="zla_cfd_test_") as d:
        state_dir = Path(d) / "state"
        r1 = Path(d) / "rec1.txt"
        r2 = Path(d) / "rec2.txt"
        base_env = {"CLOUDFLARE_TUNNEL_TOKEN": FAKE_TOKEN}

        proc1 = _run(base_env, r1, stub=STUB_CLOUDFLARED_FAILS, state_dir=state_dir)
        check(proc1.returncode == 7, f"first spawn reaches the stub (got {proc1.returncode})")

        window_s = float(_FAST_TIMING["CLOUDFLARED_RUN_THROTTLE_WINDOW_S"])
        time.sleep(window_s + 0.5)

        t0 = time.monotonic()
        proc2 = _run(base_env, r2, stub=STUB_CLOUDFLARED_FAILS, state_dir=state_dir)
        elapsed = time.monotonic() - t0
        check(proc2.returncode == 7, f"second spawn reaches the stub (got {proc2.returncode})")
        check(elapsed < 1.0, f"second spawn (after the window elapsed) did not sleep ({elapsed:.2f}s)")
        check("backing off" not in proc2.stdout, "does not log a backoff")


def test_invalid_shape_sleep_is_honored_and_overridable():
    print("\n[test] the invalid-shape sleep is honored (overridden short here) before exiting 1")
    with tempfile.TemporaryDirectory(prefix="zla_cfd_test_") as d:
        record = Path(d) / "record.txt"
        t0 = time.monotonic()
        proc = _run({"CLOUDFLARE_TUNNEL_TOKEN": "still-not-base64!!!"}, record)
        elapsed = time.monotonic() - t0
        sleep_s = float(_FAST_TIMING["CLOUDFLARED_RUN_INVALID_TOKEN_SLEEP_S"])
        check(proc.returncode == 1, f"exit code 1 (got {proc.returncode})")
        check(elapsed >= sleep_s * 0.8, f"slept close to the configured {sleep_s}s before exiting ({elapsed:.2f}s)")


def _all_tests():
    return [
        test_no_token_unset_exits_0_and_logs_disabled,
        test_no_token_empty_string_same_as_unset,
        test_fake_token_execs_cloudflared_token_only_in_env,
        test_garbage_token_refused_not_base64,
        test_valid_base64_not_json_refused,
        test_valid_json_missing_keys_refused,
        test_pasted_install_line_extracts_token,
        test_quotes_and_surrounding_whitespace_tolerated,
        test_unpadded_base64_tolerated,
        test_decoded_json_with_its_own_whitespace_tolerated,
        test_no_substring_of_token_leaks_on_invalid_shape,
        test_throttle_backs_off_within_window,
        test_throttle_does_not_fire_after_window_elapses,
        test_invalid_shape_sleep_is_honored_and_overridable,
    ]


if __name__ == "__main__":
    for t in _all_tests():
        t()
    if FAILED:
        print(f"\n{len(FAILED)} FAILURE(S): {FAILED}")
        sys.exit(1)
    print("\nAll cloudflared-run.sh tests passed.")
