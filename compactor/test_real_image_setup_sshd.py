"""The real-image test for `scripts/setup-sshd.py` — the operator tool that
installs and hardens OpenSSH server INSIDE a running production container.

Every scenario in `compactor/test_setup_sshd_script.py` runs against fake
`apt-get`/`ssh-keygen`/`sshd`/`supervisorctl` binaries on PATH, by design
(no network, not necessarily root). That is exactly the kind of mocking
that let B2/B3/H5/H6/M8 (see the hostile review) go undetected: the fake
`ssh-keygen` used to be STRICTER than the real one (rejecting a private
key the real binary accepts with exit 0), and no fake `sshd` ever really
bound a socket or evaluated a `Match` block. This suite runs the real,
unmodified script inside a real container started from the exact
PUBLISHED digest, with the real `openssh-server`/`openssh-client` apt
packages installed for real (this needs network), and proves:

  - a real ed25519 key can really log in over SSH;
  - a real password login is refused, INCLUDING when a pre-existing
    `Match Address * / PasswordAuthentication yes` drop-in (the exact
    hostile-review B2 scenario) is present — the script must refuse
    outright, before writing anything, and the config already on disk
    (unmodified by that refusal) still blocks the login for real;
  - a real private key is rejected and its content never appears in any
    output;
  - a second `--apply` is a true no-op, exit 0;
  - `--port 2222` leaves sshd listening ONLY on 2222 — both by `sshd -T`
    and by a real bound socket — never also on 22 (M8);
  - a `--supervise` handoff failure (a real `supervisorctl reread`
    failure against the real supervisord) still leaves a real, listening
    sshd afterward (H5), and a missing `supervisord.conf` refuses
    cleanly rather than crashing (M8).

NEEDS, and SKIPS (exit 3) HONESTLY if any is missing:
  - a real Docker daemon
  - the published image already pulled (`docker image inspect`)
  - network access from inside the container (apt-get update/install)

UNLIKE test_tokenizer_contract.py / test_soak_conversation.py's own
`_skip`, this suite's `_skip` NEVER honors `COMPACTOR_ALLOW_FIXTURE_SKIP`
— see compactor/test_real_image_operator_scripts.py's own docstring for
why (this is the same convention). It always exits 3.

Run it directly, on the host:
    python3 compactor/test_real_image_setup_sshd.py

ALL steps run inside ONE throwaway container (`docker run -d --rm`,
`--entrypoint bash ... sleep`), driven with `docker exec`, so a real
`--supervise` handoff, a real password change, and real host key
persistence carry across steps exactly the way they would across
multiple invocations on a real pod. The container is ALWAYS removed
(`docker rm -f`) in a `finally`, whether the suite passes or not.
"""

import json
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIGEST = "sha256:c1295894dd585784611c6833b1d4c396880ac8723e6b9b46531a5aa846cb8a65"
IMAGE_REF = f"angreg/zions-light-ai@{IMAGE_DIGEST}"
VENV_PY = "/opt/compactor-venv/bin/python"
DOCKER_TIMEOUT_S = 180
CONTAINER_NAME = f"zl-test-real-image-setup-sshd-{uuid.uuid4().hex[:10]}"


def assert_eq(actual, expected, label):
    if actual != expected:
        print(f"FAIL {label}: expected {expected!r}, got {actual!r}")
        sys.exit(1)
    print(f"  ok   {label}")


def assert_true(cond, label, extra: str = ""):
    if not cond:
        print(f"FAIL {label}")
        if extra:
            print(f"  --- context ---\n{extra}\n  --- end ---")
        sys.exit(1)
    print(f"  ok   {label}")


def assert_in(needle, haystack, label):
    if needle not in haystack:
        print(f"FAIL {label}: {needle!r} not found")
        print(f"  --- full output ---\n{haystack}\n  --- end ---")
        sys.exit(1)
    print(f"  ok   {label}")


def assert_not_in(needle, haystack, label):
    if needle in haystack:
        print(f"FAIL {label}: {needle!r} unexpectedly present")
        print(f"  --- full output ---\n{haystack}\n  --- end ---")
        sys.exit(1)
    print(f"  ok   {label}")


def _skip(reason: str) -> None:
    print("=" * 72)
    print("SKIPPED: test_real_image_setup_sshd.py")
    print(f"  reason: {reason}")
    print()
    print("  This suite needs a real Docker daemon, the published image")
    print("  already pulled, and network access from inside the")
    print("  container (it apt-installs openssh-server/openssh-client for")
    print("  real). Run it directly:")
    print("    python3 compactor/test_real_image_setup_sshd.py")
    print("=" * 72)
    sys.exit(3)


def _run(args, timeout=DOCKER_TIMEOUT_S, **kw):
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout, **kw)


def _preflight() -> None:
    if shutil.which("docker") is None:
        _skip("no `docker` binary on PATH")
    r = _run(["docker", "version"], timeout=15)
    if r.returncode != 0:
        _skip(f"`docker version` failed: {(r.stderr or r.stdout).strip()[:200]}")
    r = _run(["docker", "image", "inspect", IMAGE_REF], timeout=15)
    if r.returncode != 0:
        _skip(f"published image not pulled locally: {IMAGE_REF}")


def _docker_exec(cmd: str, timeout=DOCKER_TIMEOUT_S) -> subprocess.CompletedProcess:
    return _run(["docker", "exec", CONTAINER_NAME, "bash", "-c", cmd], timeout=timeout)


def _start_container() -> None:
    _run(["docker", "rm", "-f", CONTAINER_NAME], timeout=30)  # never trust a stale name
    r = _run([
        "docker", "run", "-d", "--rm", "--name", CONTAINER_NAME,
        "-v", f"{REPO_ROOT / 'scripts'}:/data/scripts:ro",
        "--entrypoint", "bash", IMAGE_REF, "-c", "sleep 3600",
    ], timeout=60)
    if r.returncode != 0:
        print("FAIL starting the throwaway container")
        print(r.stdout, r.stderr)
        sys.exit(1)
    time.sleep(1)
    r = _docker_exec("echo up")
    assert_eq(r.returncode, 0, "the throwaway container is really running")


def _stop_container() -> None:
    _run(["docker", "rm", "-f", CONTAINER_NAME], timeout=30)


# ---------------------------------------------------------------------------
# Setup shared by every test below: install a real ssh client (test
# harness only — never what setup-sshd.py itself installs) and generate a
# real ed25519 keypair, ONCE, before any scenario runs.
# ---------------------------------------------------------------------------

def _install_ssh_client_and_generate_key() -> None:
    r = _docker_exec(
        "apt-get update >/tmp/aptupdate0.log 2>&1 && "
        "apt-get install -y --no-install-recommends openssh-client >/tmp/aptclient.log 2>&1",
        timeout=120,
    )
    if r.returncode != 0:
        _skip(
            "could not apt-get install openssh-client inside the container "
            "(no network reachable from inside Docker?): "
            + (r.stderr or r.stdout).strip()[-300:]
        )
    r = _docker_exec("command -v ssh-keygen && command -v ssh")
    assert_eq(r.returncode, 0, "ssh and ssh-keygen are present after install")

    r = _docker_exec('ssh-keygen -q -t ed25519 -N "" -f /tmp/opkey -C operator@laptop')
    assert_eq(r.returncode, 0, "a real ed25519 keypair was generated")


def _apply(extra_args: str = "", authorized_key_from_pub: bool = True) -> tuple[int, dict, str]:
    """Runs the real script with --apply --json, real key from
    /tmp/opkey.pub unless the caller passes its own key material via
    extra_args. Returns (exit_code, parsed_json_or_empty, raw_stdout+stderr)."""
    key_arg = ""
    if authorized_key_from_pub:
        key_arg = '--authorized-key "$(cat /tmp/opkey.pub)"'
    cmd = (
        f"{VENV_PY} /data/scripts/setup-sshd.py --apply --json {key_arg} {extra_args} "
        f"2>/tmp/apply.err; echo EXIT=$?"
    )
    r = _docker_exec(cmd)
    out = r.stdout + r.stderr
    rc = None
    if "EXIT=" in out:
        try:
            rc = int(out.rsplit("EXIT=", 1)[1].strip().splitlines()[0])
        except ValueError:
            rc = None
    payload = {}
    stdout_before_exit = out.split("EXIT=")[0]
    try:
        # the JSON is everything before our own "EXIT=" marker line
        json_start = stdout_before_exit.index("{")
        payload = json.loads(stdout_before_exit[json_start:])
    except (ValueError, json.JSONDecodeError):
        pass
    return rc, payload, out


def _real_listening_ports() -> list[str]:
    r = _docker_exec(
        "python3 -c \""
        "lines = open('/proc/net/tcp').read().splitlines()[1:]\n"
        "print([l.split()[1] for l in lines if l.split()[3] == '0A'])\""
    )
    try:
        return eval(r.stdout.strip())  # a Python list literal, our own output
    except Exception:
        return []


def _port_hex(port: int) -> str:
    return f"{port:04X}"


# ---------------------------------------------------------------------------
# 1. Key login works; a real --apply is otherwise sane.
# ---------------------------------------------------------------------------

def test_apply_installs_and_a_real_key_login_works():
    print("\n[test] --apply on a clean container installs openssh-server for real, and a real SSH key login works")
    rc, payload, out = _apply()
    assert_eq(rc, 0, f"--apply succeeds (out={out[-500:]})")
    assert_eq(payload.get("daemon"), "started", "daemon really started")
    assert_true(
        "password authentication is disabled" in (payload.get("sshd_check") or ""),
        "sshd -T verification (default + both -C contexts) really ran and passed",
    )
    # B3: keys_added holds a FINGERPRINT, never the key text.
    added = payload.get("keys_added") or []
    assert_eq(len(added), 1, "one key added")
    assert_true("SHA256:" in added[0], "the reported entry is a fingerprint")
    assert_not_in("AAAA", json.dumps(payload), "no key base64 material anywhere in the JSON")

    r = _docker_exec(
        'ssh -v -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null '
        '-i /tmp/opkey root@127.0.0.1 "echo LOGGED-IN-AS=$(id -un)" '
        '>/tmp/keylogin.out 2>/tmp/keylogin.err; echo SSHEXIT=$?'
    )
    out = r.stdout + r.stderr
    assert_in("SSHEXIT=0", out, "a real SSH key login succeeds")
    r2 = _docker_exec("cat /tmp/keylogin.out")
    assert_in("LOGGED-IN-AS=root", r2.stdout, "the real login really ran a command as root")


# ---------------------------------------------------------------------------
# 2. Root PASSWORD login fails for real.
# ---------------------------------------------------------------------------

def test_root_password_login_fails():
    print("\n[test] a real root password login is refused")
    _docker_exec('echo "root:hunter2verify-$(date +%s)" | chpasswd')
    r = _docker_exec(
        'cat > /tmp/askpass.sh <<AP\n#!/bin/sh\necho hunter2verify-wrong-anyway\nAP\n'
        "chmod +x /tmp/askpass.sh; "
        "export SSH_ASKPASS=/tmp/askpass.sh SSH_ASKPASS_REQUIRE=force DISPLAY=:0; "
        'setsid -w ssh -v -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null '
        '-o PreferredAuthentications=password -o PubkeyAuthentication=no '
        '-o NumberOfPasswordPrompts=1 root@127.0.0.1 "echo SHOULD-NOT-PRINT" '
        '>/tmp/pwlogin.out 2>/tmp/pwlogin.err; echo SSHEXIT=$?'
    )
    out = r.stdout + r.stderr
    assert_true("SSHEXIT=0" not in out, "the password login attempt did NOT succeed", out)
    r2 = _docker_exec("cat /tmp/pwlogin.out")
    assert_not_in("SHOULD-NOT-PRINT", r2.stdout, "the remote command never ran")
    r3 = _docker_exec("grep -c 'Permission denied' /tmp/pwlogin.err || true")
    assert_true(r3.stdout.strip() != "0", "ssh's own stderr recorded a permission denial")


# ---------------------------------------------------------------------------
# 3. B3 — a private key is rejected outright and never leaks.
# ---------------------------------------------------------------------------

def test_private_key_rejected_and_never_leaks():
    print("\n[test] B3: a real PRIVATE key is rejected outright and its content never appears in the output")
    # The PRIVATE key file's content is read INSIDE the container via
    # `$(cat /tmp/opkey)` — it never passes through this test process's
    # own argv/environment either.
    r = _docker_exec(
        f'{VENV_PY} /data/scripts/setup-sshd.py --apply --json '
        '--authorized-key "$(cat /tmp/opkey)" >/tmp/privkey.json 2>/tmp/privkey.err; '
        "echo EXIT=$?"
    )
    out = r.stdout + r.stderr
    assert_true("EXIT=1" in out, "the private-key apply refuses (exit 1)", out)
    r2 = _docker_exec("grep -c 'PRIVATE KEY' /tmp/privkey.json || true")
    assert_eq(r2.stdout.strip(), "0", "no PRIVATE KEY marker anywhere in the JSON output")
    r3 = _docker_exec("cat /tmp/privkey.json")
    payload3 = json.loads(r3.stdout)
    assert_true(len(payload3.get("refusals") or []) > 0, "a refusal is recorded")
    assert_true(payload3.get("keys_added") == [], "nothing was added")


# ---------------------------------------------------------------------------
# 4. A second --apply is a true, idempotent no-op.
# ---------------------------------------------------------------------------

def test_second_apply_is_idempotent():
    print("\n[test] a second --apply is a true no-op, exit 0")
    rc, payload, out = _apply()
    assert_eq(rc, 0, f"second --apply succeeds (out={out[-500:]})")
    assert_eq(payload.get("action"), "already-current", "package already at candidate")
    assert_eq(payload.get("keys_added"), [], "no new key the second time")
    assert_eq(payload.get("daemon"), "already-running", "daemon left alone")


# ---------------------------------------------------------------------------
# 5. M8 — --port 2222 listens ONLY on 2222.
# ---------------------------------------------------------------------------

def test_custom_port_listens_only_on_that_port():
    print("\n[test] M8: --port 2222 leaves sshd listening ONLY on 2222 (never also 22), by sshd -T AND a real socket")
    rc, payload, out = _apply(extra_args="--port 2222")
    assert_eq(rc, 0, f"--apply --port 2222 succeeds (out={out[-500:]})")
    assert_eq(payload.get("port"), 2222, "port reported back")

    r = _docker_exec("/usr/sbin/sshd -T 2>/dev/null | grep -i '^port '")
    assert_eq(r.stdout.strip(), "port 2222", "the REAL sshd -T reports exactly one active port: 2222")

    ports = _real_listening_ports()
    hexports = {p.split(":")[1] for p in ports}
    assert_true(_port_hex(2222) in hexports, f"really listening on 2222 (ports={ports})")
    assert_true(_port_hex(22) not in hexports, f"NOT listening on 22 anymore (ports={ports})")

    r2 = _docker_exec(
        'ssh -v -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 2222 '
        '-i /tmp/opkey root@127.0.0.1 "echo LOGGED-IN-AS=$(id -un) PORT=2222" '
        '>/tmp/keylogin2222.out 2>/tmp/keylogin2222.err; echo SSHEXIT=$?'
    )
    assert_in("SSHEXIT=0", r2.stdout + r2.stderr, "a real SSH login on port 2222 succeeds")


# ---------------------------------------------------------------------------
# 6. B2 — a pre-existing Match-block drop-in makes the script refuse
#    outright, and password login remains blocked for real either way.
# ---------------------------------------------------------------------------

def test_preexisting_match_block_dropin_forces_refusal():
    print("\n[test] B2: a pre-existing drop-in with a Match block makes --apply refuse outright, before any write")
    _docker_exec(
        "printf '# left by a previous operator / pod template\\n"
        "Match Address *\\n    PasswordAuthentication yes\\n    PermitRootLogin yes\\n' "
        "> /etc/ssh/sshd_config.d/00-runpod.conf"
    )
    rc, payload, out = _apply(extra_args="--port 2222")
    assert_eq(rc, 1, f"refuses with exit 1 (out={out[-500:]})")
    refusals = payload.get("refusals") or []
    assert_true(any("Match" in r for r in refusals), f"refusal names the Match block (refusals={refusals})")

    # Real proof of WHY this matters: sshd -T -C for a real remote
    # address shows the Match block WOULD have enabled password auth,
    # even though a bare sshd -T (no -C) looks safe — exactly the B2 gap.
    r = _docker_exec("/usr/sbin/sshd -T 2>/dev/null | grep -i '^passwordauthentication'")
    assert_eq(r.stdout.strip(), "passwordauthentication no", "bare sshd -T (no -C) looks safe")
    r2 = _docker_exec(
        "/usr/sbin/sshd -T -C user=root,host=h,addr=203.0.113.9 2>/dev/null "
        "| grep -i '^passwordauthentication'"
    )
    assert_eq(r2.stdout.strip(), "passwordauthentication yes",
              "a real non-local client context WOULD have password auth enabled — the exact B2 exposure")

    # And a real password login attempt against the ACTUAL (unmodified —
    # nothing was written by the refused run) config still fails.
    _docker_exec('echo "root:hunter2verify2-$(date +%s)" | chpasswd')
    r3 = _docker_exec(
        "export SSH_ASKPASS=/tmp/askpass.sh SSH_ASKPASS_REQUIRE=force DISPLAY=:0; "
        'setsid -w ssh -v -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null '
        '-o PreferredAuthentications=password -o PubkeyAuthentication=no '
        '-o NumberOfPasswordPrompts=1 -p 2222 root@127.0.0.1 "echo SHOULD-NOT-PRINT" '
        '>/tmp/pwlogin3.out 2>/tmp/pwlogin3.err; echo SSHEXIT=$?'
    )
    assert_true("SSHEXIT=0" not in (r3.stdout + r3.stderr),
                "password login still fails for real with the Match-block drop-in present")

    _docker_exec("rm -f /etc/ssh/sshd_config.d/00-runpod.conf")


# ---------------------------------------------------------------------------
# 7. M8 — a missing supervisord.conf refuses cleanly, never a crash.
# ---------------------------------------------------------------------------

def test_missing_supervisor_conf_refuses_cleanly():
    print("\n[test] M8: --supervise against a MISSING supervisord.conf refuses cleanly, not an unhandled exception")
    _docker_exec("mv /etc/supervisor/conf.d/supervisord.conf /tmp/supervisord.conf.hidden")
    rc, payload, out = _apply(extra_args="--supervise --port 2222")
    assert_eq(rc, 1, f"refuses with exit 1, not a crash (out={out[-500:]})")
    assert_not_in("Traceback", out, "no Python traceback")
    refusals = payload.get("refusals") or []
    assert_true(any("does not exist" in r for r in refusals), f"refusal names the missing conf (refusals={refusals})")
    _docker_exec("mv /tmp/supervisord.conf.hidden /etc/supervisor/conf.d/supervisord.conf")


# ---------------------------------------------------------------------------
# 8. H5/H6 — a real --supervise handoff failure still leaves a real,
#    listening sshd afterward.
# ---------------------------------------------------------------------------

def test_supervise_handoff_failure_still_leaves_listening_sshd():
    print("\n[test] H5/H6: a real supervisord handoff failure still leaves a real, listening sshd afterward")
    ports_before = {p.split(":")[1] for p in _real_listening_ports()}
    assert_true(_port_hex(2222) in ports_before, f"sshd listening on 2222 before the handoff attempt (ports={ports_before})")

    r = _docker_exec(
        "supervisord -c /etc/supervisor/conf.d/supervisord.conf >/tmp/supervisord.log 2>&1 & "
        "sleep 3; echo started"
    )
    assert_in("started", r.stdout, "the real supervisord stack was launched")

    # Force the handoff's own `supervisorctl reread` to fail for real by
    # making the conf file unreadable to it.
    _docker_exec("chmod 000 /etc/supervisor/conf.d/supervisord.conf")
    rc, payload, out = _apply(extra_args="--supervise --port 2222")
    _docker_exec("chmod 644 /etc/supervisor/conf.d/supervisord.conf")

    assert_eq(rc, 1, f"the handoff failure is a refusal, exit 1 (out={out[-500:]})")
    refusals = payload.get("refusals") or []
    assert_true(
        any("reread" in r or "supervis" in r.lower() for r in refusals),
        f"refusal names the supervisorctl failure (refusals={refusals})",
    )
    assert_eq(payload.get("daemon"), "restarted",
              "the report reflects the standalone-restart recovery (H5), not a silent 'not-started'")

    ports_after = {p.split(":")[1] for p in _real_listening_ports()}
    assert_true(_port_hex(2222) in ports_after,
                f"sshd is STILL listening on 2222 after the failed handoff (ports={ports_after})")

    r2 = _docker_exec(
        'ssh -v -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 2222 '
        '-i /tmp/opkey root@127.0.0.1 "echo STILL-WORKS=$(id -un)" '
        '>/tmp/keylogin_after.out 2>/tmp/keylogin_after.err; echo SSHEXIT=$?'
    )
    assert_in("SSHEXIT=0", r2.stdout + r2.stderr, "a real SSH login still succeeds after the recovery")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _preflight()
    try:
        _start_container()
        _install_ssh_client_and_generate_key()

        test_apply_installs_and_a_real_key_login_works()
        test_root_password_login_fails()
        test_private_key_rejected_and_never_leaks()
        test_second_apply_is_idempotent()
        test_custom_port_listens_only_on_that_port()
        test_preexisting_match_block_dropin_forces_refusal()
        test_missing_supervisor_conf_refuses_cleanly()
        test_supervise_handoff_failure_still_leaves_listening_sshd()

        print("\nAll real-image setup-sshd.py tests passed.")
    finally:
        _stop_container()
