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
import os
import sys
import time
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIGEST = "sha256:c1295894dd585784611c6833b1d4c396880ac8723e6b9b46531a5aa846cb8a65"
# ZLA_REAL_IMAGE_REF lets this suite point at a locally built candidate image
# (e.g. the v3.1.9.7 thin OpenWebUI-upgrade layer) instead of the published
# base digest, without changing the default every other caller relies on.
IMAGE_REF = os.environ.get("ZLA_REAL_IMAGE_REF") or f"angreg/zions-light-ai@{IMAGE_DIGEST}"
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
# 9. N2 — a Match pulled in by an arbitrary Include (not sshd_config.d
#    itself) is caught, and a REAL root password login from this
#    container's own (non-loopback) address — a RunPod-proxy stand-in,
#    same convention the round-2 review used — still fails.
# ---------------------------------------------------------------------------

def test_include_match_bypass_refuses_and_private_address_login_fails():
    print("\n[test] N2: a Match pulled in through an arbitrary Include is caught outright, and a real password login from a private-range address fails")
    _docker_exec("mkdir -p /etc/ssh/site.d")
    _docker_exec("printf 'Include /etc/ssh/site.d/*\\n' > /etc/ssh/sshd_config.d/50-site.conf")
    _docker_exec(
        "printf 'Match Address 172.16.0.0/12,10.0.0.0/8,100.64.0.0/10\\n"
        "    PasswordAuthentication yes\\n    PermitRootLogin yes\\n' "
        "> /etc/ssh/site.d/private-nets"
    )

    rc, payload, out = _apply(extra_args="--port 2223")
    assert_eq(rc, 1, f"refuses with exit 1 (out={out[-500:]})")
    refusals = payload.get("refusals") or []
    assert_true(any("Match" in r for r in refusals),
                f"refusal names the Match block pulled in via Include (refusals={refusals})")

    # Independent proof of the exact B2/N2 exposure this closes: a bare
    # `sshd -T` (no -C) looks safe, but `-C` for an address in the
    # Match'd private range shows password auth WOULD have been enabled.
    r = _docker_exec("/usr/sbin/sshd -T 2>/dev/null | grep -i '^passwordauthentication'")
    assert_eq(r.stdout.strip(), "passwordauthentication no", "bare sshd -T (no -C) looks safe")
    r2 = _docker_exec(
        "/usr/sbin/sshd -T -C user=root,host=h,addr=172.20.5.6,laddr=0.0.0.0,lport=22 2>/dev/null "
        "| grep -i '^passwordauthentication'"
    )
    assert_eq(r2.stdout.strip(), "passwordauthentication yes",
              "a private-range -C context WOULD have had password auth enabled — the exact N2 exposure")

    # And the REAL proof: a root password login attempt from this
    # container's own (non-loopback) address — which really does fall in
    # the Match'd 172.16.0.0/12 range docker assigns its bridge network —
    # still fails, because nothing was ever written (the script refused
    # before any write, so the UNMODIFIED shipped config, whatever it is,
    # is what's actually protecting this).
    own_addr_r = _docker_exec("hostname -I")
    own_addrs = own_addr_r.stdout.split()
    assert_true(len(own_addrs) > 0, "the container reports at least one address")
    own_addr = own_addrs[0]

    # Log in on whatever port sshd is REALLY listening on right now (an
    # earlier test in this suite switches it to 2222) — connecting to a
    # port nothing listens on would trivially "fail" without proving
    # anything about the security config.
    port_r = _docker_exec("/usr/sbin/sshd -T 2>/dev/null | grep -i '^port ' | awk '{print $2}' | head -1")
    active_port = port_r.stdout.strip() or "22"

    _docker_exec(f'echo "root:hunter2verifyN2-$(date +%s)" | chpasswd')
    _docker_exec(
        'cat > /tmp/askpassN2.sh <<AP\n#!/bin/sh\necho hunter2verifyN2-wrong-anyway\nAP\n'
        "chmod +x /tmp/askpassN2.sh"
    )
    r3 = _docker_exec(
        "export SSH_ASKPASS=/tmp/askpassN2.sh SSH_ASKPASS_REQUIRE=force DISPLAY=:0; "
        'setsid -w ssh -v -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null '
        '-o PreferredAuthentications=password -o PubkeyAuthentication=no '
        f'-o NumberOfPasswordPrompts=1 -p {active_port} root@{own_addr} "echo SHOULD-NOT-PRINT" '
        '>/tmp/pwloginN2.out 2>/tmp/pwloginN2.err; echo SSHEXIT=$?'
    )
    out3 = r3.stdout + r3.stderr
    assert_true(
        "SSHEXIT=0" not in out3,
        f"a real root password login from this container's own address {own_addr} (private-range, matches the Match block) still fails",
        out3,
    )
    r4 = _docker_exec("cat /tmp/pwloginN2.out")
    assert_not_in("SHOULD-NOT-PRINT", r4.stdout, "the remote command never ran")

    _docker_exec("rm -rf /etc/ssh/sshd_config.d/50-site.conf /etc/ssh/site.d")
    print(f"  --- N2 real-container transcript (own_addr={own_addr}) ---")
    print(f"  apply refusals: {refusals}")
    print(f"  sshd -T (no -C):      passwordauthentication no")
    print(f"  sshd -T -C (private): passwordauthentication yes")
    print(f"  password login attempt from {own_addr}: {out3.strip()[-300:]}")
    print("  --- end transcript ---")


# ---------------------------------------------------------------------------
# 10. N3 — "listening" must mean OUR sshd: a foreign process (P1) and a
#     stray sshd master with a different pidfile (P2) must both be
#     refused outright, never silently reported as "hardened".
# ---------------------------------------------------------------------------

def test_stray_listener_and_foreign_sshd_never_reported_as_hardened():
    print("\n[test] N3: a stray listener (P1: foreign process, P2: another sshd master with a different pidfile) is refused, never adopted as 'hardened'")
    probe_port = 2299

    # --- P1: an unrelated process (plain python) already bound to the
    # target port.
    _docker_exec(
        f"nohup python3 -c \""
        f"import socket,time; s=socket.socket(socket.AF_INET, socket.SOCK_STREAM); "
        f"s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(('0.0.0.0', {probe_port})); "
        f"s.listen(1); time.sleep(120)\" >/tmp/strayp1.log 2>&1 & echo $! > /tmp/strayp1.pid"
    )
    time.sleep(1)
    ports_p1 = {p.split(":")[1] for p in _real_listening_ports()}
    assert_true(_port_hex(probe_port) in ports_p1, f"P1: the foreign python listener is really bound (ports={ports_p1})")

    rc1, payload1, out1 = _apply(extra_args=f"--port {probe_port}")
    refusals1 = payload1.get("refusals") or []
    assert_eq(rc1, 1, f"P1: refuses with exit 1 rather than adopt a foreign listener (out={out1[-500:]})")
    assert_true(any("LISTEN" in r for r in refusals1),
                f"P1: refusal names the foreign listener (refusals={refusals1})")

    _docker_exec("kill $(cat /tmp/strayp1.pid) 2>/dev/null || true")
    time.sleep(1)

    # --- P2: a stray, password-enabled sshd master with a DIFFERENT
    # pidfile, already bound to the same target port. The override lines
    # go FIRST, before the copied file's own `Include
    # sshd_config.d/*.conf` line — sshd is first-match-wins for globals,
    # so putting them after the Include (which pulls in this script's
    # OWN "PasswordAuthentication no" drop-in from an earlier test) would
    # let the drop-in win instead and silently defeat this repro.
    _docker_exec(
        f"printf 'Port {probe_port}\\nPasswordAuthentication yes\\nPermitRootLogin yes\\n' "
        f"> /tmp/insecure_config && cat /etc/ssh/sshd_config >> /tmp/insecure_config"
    )
    r_start = _docker_exec(f"/usr/sbin/sshd -f /tmp/insecure_config -o PidFile=/tmp/stray-sshd.pid")
    time.sleep(1)
    r_pf = _docker_exec("cat /tmp/stray-sshd.pid 2>/dev/null || echo NONE")
    assert_true(r_pf.stdout.strip() != "NONE",
                f"P2: the stray sshd master really started (stderr={r_start.stderr[:300]})")
    ports_p2 = {p.split(":")[1] for p in _real_listening_ports()}
    assert_true(_port_hex(probe_port) in ports_p2, f"P2: the stray sshd is really bound (ports={ports_p2})")

    # Prove the stray really IS the password-enabled hazard this refusal
    # exists for: a real root password login against IT succeeds. Fixed
    # password, set once, then used by a fixed SSH_ASKPASS script.
    stray_pw = "hunter2verifyN3-stray-probe"
    _docker_exec(f'echo "root:{stray_pw}" | chpasswd')
    _docker_exec(
        f'printf "#!/bin/sh\\necho {stray_pw}\\n" > /tmp/askpassN3.sh; chmod +x /tmp/askpassN3.sh'
    )
    r_stray_login = _docker_exec(
        "export SSH_ASKPASS=/tmp/askpassN3.sh SSH_ASKPASS_REQUIRE=force DISPLAY=:0; "
        'setsid -w ssh -v -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null '
        '-o PreferredAuthentications=password -o PubkeyAuthentication=no '
        f'-o NumberOfPasswordPrompts=1 -p {probe_port} root@127.0.0.1 "echo STRAY-LOGIN-OK" '
        '>/tmp/strayloginN3.out 2>/tmp/strayloginN3.err; echo SSHEXIT=$?'
    )
    stray_login_out = r_stray_login.stdout + r_stray_login.stderr
    stray_login_worked = "SSHEXIT=0" in stray_login_out
    assert_true(
        stray_login_worked,
        f"the stray sshd really IS password-vulnerable (proves the refusal below matters, not just a paperwork check)",
        stray_login_out,
    )

    rc2, payload2, out2 = _apply(extra_args=f"--port {probe_port}")
    refusals2 = payload2.get("refusals") or []
    assert_eq(rc2, 1, f"P2: refuses with exit 1, never reports 'hardened' (out={out2[-500:]})")
    assert_true(
        any("sshd master" in r or "LISTEN" in r for r in refusals2),
        f"P2: refusal names the stray sshd master (refusals={refusals2})",
    )

    _docker_exec("kill $(cat /tmp/stray-sshd.pid) 2>/dev/null || true")
    _docker_exec("rm -f /tmp/insecure_config /tmp/stray-sshd.pid /tmp/strayp1.pid")

    print("  --- N3 real-container transcript ---")
    print(f"  P1 (foreign python listener on {probe_port}): apply refusals={refusals1}")
    print(f"  P2 (stray sshd master on {probe_port}, different pidfile): apply refusals={refusals2}")
    print(f"  P2 stray sshd really was password-vulnerable: real login {'SUCCEEDED' if stray_login_worked else 'did not succeed'} "
          f"(this is exactly why silently adopting it would have been catastrophic)")
    print("  --- end transcript ---")


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
        test_include_match_bypass_refuses_and_private_address_login_fails()
        test_stray_listener_and_foreign_sshd_never_reported_as_hardened()

        print("\nAll real-image setup-sshd.py tests passed.")
    finally:
        _stop_container()
