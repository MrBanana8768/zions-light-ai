"""The real-image test for the `zl` pod-ops bundle (ops/bootstrap.sh,
ops/zl, ops/zlops/...) -- proves the bundle actually works inside a
container started from the exact PUBLISHED digest, with a real Python
(no fakes for supervisorctl/urllib/subprocess), not just against mocks.

Same convention as compactor/test_real_image_setup_sshd.py: `_skip` NEVER
honors COMPACTOR_ALLOW_FIXTURE_SKIP and always exits 3; ALL steps run
inside ONE throwaway container (`docker run -d --rm`, `--entrypoint bash
... sleep`), driven with `docker exec`; the container is ALWAYS removed
(`docker rm -f`) in a `finally`.

The worktree is mounted READ-ONLY AS A DIRECTORY at /work (never a single
file inside an already-mounted dir -- memory.md's own "root-owned empty
file" incident), and `zl` is installed from it via
`bootstrap.sh --from-dir /work` (no network clone needed).

NEEDS, and SKIPS (exit 3) HONESTLY if any is missing:
  - a real Docker daemon
  - the published image already pulled (`docker image inspect`)

Covers, against the real image:
  - `bootstrap.sh --from-dir /work` installs `zl` and exits 0
  - `zl help`, `zl version`
  - `zl health` with no services running: crit, a clean message, no
    traceback
  - `zl dash --once` and `zl dash --json` (exit 0, `--json` is valid JSON
    on stdout with nothing else mixed in)
  - `zl bundle`: the tarball's member layout matches the brief, and a
    planted `HF_TOKEN=hf_plantedsecret123` / `SOME_API_KEY=plainsecret`
    (set as real container env vars) never appear ANYWHERE in the
    extracted tree
  - two REAL registry scripts' dry runs (their real interpreter, e.g.
    /opt/compactor-venv/bin/python, actually starts and exits cleanly --
    no traceback, no "interpreter not found")
  - the `zl fix` apply flow's stop/restore mechanics against a REAL
    supervisord (started in the container) with a small config of dummy
    `sleep` programs named after the registry's own service names: a
    successful apply, a FAILING script, and a KeyboardInterrupt mid-script
    all restore exactly the services that were running before, a
    pre-stopped program (webuidb-sync) stays stopped throughout, and an
    untouched program (backup) is never even asked about. The target
    script here is a small fake (per the brief: "Use a fake target script
    via a test-only registry override ... ONLY if unavoidable") --
    unavoidable because only `--apply` exercises stop/restore at all, and
    the real registry scripts need real webui.db/compactor-store data this
    throwaway container does not have; the ORCHESTRATION code under test
    (ops/zlops/commands/fix.py, common.py) is the real, unmodified code,
    mounted read-only from the worktree, not a copy.

Run it directly, on the host:
    python3 compactor/test_real_image_ops_bundle.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
# LOW (round 3): never write ops/**/__pycache__ from a test run --
# this file's own `from zlops import ...` below would otherwise create
# it even when the harness invoking this file forgot PYTHONDONTWRITEBYTECODE.
sys.dont_write_bytecode = True
import tarfile
import tempfile
import time
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# PROD FACT (memory.md "Current state"): production actually runs
# `angreg/zions-light-ai:v3.1.9-cu12`, digest sha256:f3fafa3e1949..., NOT
# c1295894 (that digest is a validated-but-not-yet-deployed build). Default
# to prod's own image; ZL_TEST_IMAGE overrides it (e.g. to re-run this
# suite against c1295894 too -- see the brief's "run the real-image suite
# against BOTH images at the end").
_DEFAULT_IMAGE_DIGEST = "sha256:f3fafa3e1949115dfd2c1354969b03be8d852fc491b647884ccfe7a23cf998d2"
IMAGE_DIGEST = os.environ.get("ZL_TEST_IMAGE", _DEFAULT_IMAGE_DIGEST)
# ZL_TEST_IMAGE may be a bare "sha256:..." digest (assumed to be
# angreg/zions-light-ai's own), or a full reference/tag/local image id of
# its own (e.g. "c1295894dd58" or "angreg/zions-light-ai:v3.1.9-cu12") --
# used exactly as given in that case.
IMAGE_REF = f"angreg/zions-light-ai@{IMAGE_DIGEST}" if IMAGE_DIGEST.startswith("sha256:") else IMAGE_DIGEST
DOCKER_TIMEOUT_S = 180
CONTAINER_NAME = f"zl-test-real-image-ops-bundle-{uuid.uuid4().hex[:10]}"

PLANTED_HF_TOKEN = "hf_plantedsecret123"
PLANTED_API_KEY = "plainsecret"

_SUPERVISORD_CONF = """\
[unix_http_server]
file=/var/run/supervisor.sock
chmod=0700

[supervisord]
nodaemon=true
logfile=/tmp/test-supervisord.log
pidfile=/tmp/test-supervisord.pid

[rpcinterface:supervisor]
supervisor.rpcinterface_factory = supervisor.rpcinterface:make_main_rpcinterface

[supervisorctl]
serverurl=unix:///var/run/supervisor.sock

[program:openwebui]
command=sleep 99999
autostart=true
autorestart=true

[program:compactor]
command=sleep 99999
autostart=true
autorestart=true

[program:backup]
command=sleep 99999
autostart=true
autorestart=true

[program:webuidb-sync]
command=sleep 99999
autostart=false
autorestart=false
"""

_FAKE_TARGET_PY = """\
import sys
from pathlib import Path
mode = sys.argv[1] if len(sys.argv) > 1 else ""
if mode == "--fail":
    sys.exit(1)
if mode == "--apply":
    Path("/tmp/fake_target.marker").write_text("applied\\n")
sys.exit(0)
"""

# review B1: a target that runs long enough to be signalled mid-apply. The
# marker is written only if the sleep runs to completion -- a forwarded
# SIGINT (from the wrapper's own SIGTERM handling) interrupts the sleep,
# so "no marker" is direct evidence the child was actually signalled, and
# "marker present" is direct evidence a signal the wrapper ignored (SIGHUP)
# never reached/killed it.
_SLOW_TARGET_PY = """\
import sys, time
from pathlib import Path
mode = sys.argv[1] if len(sys.argv) > 1 else ""
if mode == "--apply":
    # Round-4 LOW: written the INSTANT this process starts running, so the
    # driving test can wait for the CHILD to genuinely exist (Popen
    # returned, run_protected about to register it with the protected
    # region) instead of a fixed wall-clock sleep after merely seeing
    # supervisorctl report the service STOPPED -- which races the
    # wrapper's OWN signal-handler setup under CPU throttling (reproduced
    # 4/4 at 0.2 CPU).
    Path("/tmp/slow_target.started").write_text("started\\n")
    time.sleep(8)
    Path("/tmp/slow_target.marker").write_text("applied\\n")
sys.exit(0)
"""

_DRIVE_SLOW_FIX_PY = """\
import os, sys
os.environ.setdefault("ZL_HOME", "/tmp/zl-home")
sys.path.insert(0, "/work/ops")
from pathlib import Path
Path("/tmp/wrapper.pid").write_text(str(os.getpid()))
from unittest import mock
from zlops import cli
from zlops.commands import fix
entry = fix.FixEntry(
    name="t-slow", script="slow_target.py", interpreter=Path(sys.executable),
    read_only=False, dry_run_args=(), apply_args=("--apply",),
    services=lambda: ("openwebui", "compactor"), services_source="real-image signal test",
    default_args=lambda: (), help="test",
)
with mock.patch.object(fix.podenv, "scripts_dir", return_value=Path("/tmp")), \
     mock.patch.object(fix.common, "_wait_for_openwebui_http_ok", return_value=True):
    # openwebui here is `sleep 99999` -- it will never answer /api/config,
    # so the restart-order wait must not actually block on it for real.
    fix.REGISTRY["t-slow"] = entry
    code = cli.main(["fix", "t-slow", "--apply", "--yes", "--skip-backup"])
Path("/tmp/wrapper_exit_code").write_text(str(code))
"""

_DRIVE_FIX_PY = """\
import sys, json
sys.path.insert(0, "/work/ops")
from unittest import mock
from pathlib import Path
from zlops import common
from zlops.commands import fix

def status():
    return common.supervisorctl_status()

report = {"pre": status()}

ok_entry = fix.FixEntry(
    name="t-ok", script="fake_target.py", interpreter=Path(sys.executable),
    read_only=False, dry_run_args=(), apply_args=("--apply",),
    services=lambda: ("openwebui", "compactor"), services_source="real-image test",
    default_args=lambda: (), help="test",
)
fail_entry = fix.FixEntry(
    name="t-fail", script="fake_target.py", interpreter=Path(sys.executable),
    read_only=False, dry_run_args=(), apply_args=("--fail",),
    services=lambda: ("openwebui", "compactor"), services_source="real-image test",
    default_args=lambda: (), help="test",
)
with mock.patch.object(fix.podenv, "scripts_dir", return_value=Path("/tmp")), \
     mock.patch.object(fix.common, "_wait_for_openwebui_http_ok", return_value=True):
    # openwebui here is just `sleep 99999` -- it will never answer
    # /api/config, so the restart-order wait (common.restart_after_apply)
    # must not actually block on it for real (its default parameter value
    # is bound at function-definition time, so patching the TIMEOUT
    # constant alone does nothing after the fact -- patch the wait itself).
    fix.REGISTRY["t-ok"] = ok_entry
    fix.REGISTRY["t-fail"] = fail_entry

    report["ok_code"] = fix.run(["t-ok", "--apply", "--yes", "--skip-backup"])
    report["ok_marker"] = Path("/tmp/fake_target.marker").exists()
    report["post_ok"] = status()

    report["fail_code"] = fix.run(["t-fail", "--apply", "--yes", "--skip-backup"])
    report["post_fail"] = status()

    try:
        with mock.patch.object(fix.common, "run_protected", side_effect=KeyboardInterrupt):
            fix.run(["t-ok", "--apply", "--yes", "--skip-backup"])
        report["kbdint_propagated"] = False
    except KeyboardInterrupt:
        report["kbdint_propagated"] = True
    report["post_kbdint"] = status()

print(json.dumps(report))
"""


def assert_eq(actual, expected, label, extra=""):
    if actual != expected:
        print(f"FAIL {label}: expected {expected!r}, got {actual!r}")
        if extra:
            print(f"  --- context ---\n{extra}\n  --- end ---")
        sys.exit(1)
    print(f"  ok   {label}")


def assert_true(cond, label, extra: str = "") -> None:
    if not cond:
        print(f"FAIL {label}")
        if extra:
            print(f"  --- context ---\n{extra}\n  --- end ---")
        sys.exit(1)
    print(f"  ok   {label}")


def _parse_supervisorctl_status(text: str) -> dict[str, str]:
    """`{program_name: STATE}` from a real `supervisorctl status` stdout
    -- round-4 LOW: every assertion in this file used to just check
    `"RUNNING" in block`/`"STOPPED" in mid.stdout`, which is true as long
    as ANY line in the block says so, not necessarily the SERVICE the
    assertion claims to be checking (e.g. webuidb-sync being STOPPED made
    an assertion about openwebui pass even if openwebui itself were
    RUNNING). Callers now look up `states.get(name)` and compare exactly.
    """
    states: dict[str, str] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            states[parts[0]] = parts[1]
    return states


def _skip(reason: str) -> None:
    print("=" * 72)
    print("SKIPPED: test_real_image_ops_bundle.py")
    print(f"  reason: {reason}")
    print()
    print("  This suite needs a real Docker daemon and the published image")
    print("  already pulled. Run it directly:")
    print("    python3 compactor/test_real_image_ops_bundle.py")
    print("=" * 72)
    sys.exit(3)


def _run(args, timeout=DOCKER_TIMEOUT_S, **kw) -> subprocess.CompletedProcess:
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
        "-v", f"{REPO_ROOT}:/work:ro",
        "-e", f"HF_TOKEN={PLANTED_HF_TOKEN}",
        "-e", f"SOME_API_KEY={PLANTED_API_KEY}",
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


def test_boot_from_read_only_mount_with_no_network_installs_zl_and_exits_0():
    """v3.1.9.8 `[program:zl-ops]` (NOTE-ops-bundle-boot.md): a container
    started with NO network at all (`--network none` -- there is no
    `apt-get update`/git clone this can possibly reach) still runs
    `ops/bootstrap.sh --from-dir <baked ops/> --boot` to completion,
    installs `zl`, and exits 0. A SEPARATE, throwaway container (not the
    shared one the rest of this file drives) -- `--network none` cannot be
    toggled on an already-running container, and no other test in this
    file needs network isolation.
    """
    name = f"zl-test-boot-no-network-{uuid.uuid4().hex[:10]}"
    _run(["docker", "rm", "-f", name], timeout=30)
    try:
        r = _run(
            [
                "docker", "run", "--rm", "--network", "none", "--name", name,
                "-v", f"{REPO_ROOT}:/work:ro",
                "--entrypoint", "bash", IMAGE_REF,
                "-c", "bash /work/ops/bootstrap.sh --from-dir /work --boot --no-login-hook "
                      "&& command -v zl && echo BOOT_OK",
            ],
            timeout=60,
        )
        assert_eq(r.returncode, 0, "--boot from a read-only mount with --network none exits 0",
                  r.stdout + r.stderr)
        assert_true("BOOT_OK" in r.stdout, "zl really is on PATH after --boot with no network at all",
                    r.stdout + r.stderr)
    finally:
        _run(["docker", "rm", "-f", name], timeout=30)


def test_bootstrap_from_dir_installs_zl():
    r = _docker_exec("bash /work/ops/bootstrap.sh --from-dir /work --no-login-hook")
    assert_eq(r.returncode, 0, "bootstrap.sh --from-dir /work exits 0")
    r2 = _docker_exec("command -v zl")
    assert_eq(r2.returncode, 0, "`zl` is on PATH after bootstrap")


def test_help_and_version():
    r = _docker_exec("zl help")
    assert_eq(r.returncode, 0, "`zl help` exits 0")
    assert_true("zl status" in r.stdout, "`zl help` lists zl status", extra=r.stdout)

    r2 = _docker_exec("zl version")
    assert_eq(r2.returncode, 0, "`zl version` exits 0", )
    assert_true("Traceback (most recent call last):" not in (r2.stdout + r2.stderr), "`zl version` prints no traceback",
                extra=r2.stdout + r2.stderr)


def test_health_no_services_is_crit_with_a_clean_message():
    r = _docker_exec("zl health")
    assert_eq(r.returncode, 1, "`zl health` with nothing running exits 1 (crit)")
    assert_true("status: crit" in r.stdout, "`zl health` reports crit", extra=r.stdout)
    assert_true("Traceback (most recent call last):" not in (r.stdout + r.stderr), "`zl health` prints no traceback on an unreachable compactor",
                extra=r.stdout + r.stderr)


def test_dash_once_and_json():
    r = _docker_exec("zl dash --once")
    assert_eq(r.returncode, 0, "`zl dash --once` exits 0", )
    assert_true("Traceback (most recent call last):" not in (r.stdout + r.stderr), "`zl dash --once` prints no traceback",
                extra=r.stdout + r.stderr)

    r2 = _docker_exec("zl dash --json 2>/dev/null")  # stderr (the exit-meaning line) must not pollute stdout
    assert_eq(r2.returncode, 0, "`zl dash --json` exits 0")
    try:
        doc = json.loads(r2.stdout)
    except json.JSONDecodeError as e:
        print(f"FAIL `zl dash --json` printed invalid JSON on stdout: {e}")
        print(r2.stdout[:2000])
        sys.exit(1)
    print("  ok   `zl dash --json` printed valid JSON on stdout")
    assert_true(isinstance(doc, dict) and doc, "`zl dash --json`'s JSON is a non-empty object", extra=str(doc)[:500])


def test_bundle_layout_and_no_secret_leak():
    r = _docker_exec("zl bundle --print-path 2>/dev/null")
    assert_eq(r.returncode, 0, "`zl bundle` exits 0")
    remote_path = r.stdout.strip().splitlines()[-1]
    assert_true(remote_path.endswith(".tar.gz"), "`zl bundle --print-path` prints an archive path", extra=remote_path)

    with tempfile.TemporaryDirectory() as td:
        local_tgz = Path(td) / "bundle.tar.gz"
        cp = _run(["docker", "cp", f"{CONTAINER_NAME}:{remote_path}", str(local_tgz)], timeout=30)
        assert_eq(cp.returncode, 0, "the bundle archive can be copied out of the container")

        with tarfile.open(local_tgz, "r:gz") as tf:
            names = {n.split("/", 1)[1] for n in tf.getnames() if "/" in n}
            tf.extractall(Path(td) / "extracted", filter="data")  # noqa: S202 -- our own just-built archive

        expected_meta = {
            "meta/health.json", "meta/supervisorctl.txt", "meta/gpu.txt", "meta/disk.txt",
            "meta/env-redacted.txt", "meta/conversations.json", "meta/telemetry.json", "meta/versions.txt",
        }
        assert_true(expected_meta <= names, "every meta/ file from the brief is present", extra=str(sorted(names)))
        assert_true(any(n.startswith("logs/zla") for n in names), "a logs/zla/ directory is present",
                    extra=str(sorted(names)))

        extracted_root = next((Path(td) / "extracted").iterdir())
        env_text = (extracted_root / "meta" / "env-redacted.txt").read_text(encoding="utf-8")
        assert_true("HF_TOKEN=<redacted>" in env_text, "HF_TOKEN is redacted in env-redacted.txt", extra=env_text)
        assert_true("SOME_API_KEY=<redacted>" in env_text, "SOME_API_KEY is redacted in env-redacted.txt",
                    extra=env_text)

        offenders = []
        for p in (Path(td) / "extracted").rglob("*"):
            if not p.is_file():
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for secret in (PLANTED_HF_TOKEN, PLANTED_API_KEY):
                if secret in text:
                    offenders.append((str(p), secret))
        assert_eq(offenders, [], "neither planted secret value appears anywhere in the extracted bundle")


def test_real_registry_scripts_dry_run_with_their_real_interpreter():
    for name in ("compare-live-to-backup", "backfill-records"):
        r = _docker_exec(f"zl fix {name}")
        # 126/127 are the shell's own "found but not executable" / "command
        # not found" codes -- exactly what a missing interpreter looks like.
        # The script's OWN exit code (usually 1 here: no real data in this
        # throwaway container) is fine; only those two, or a traceback, mean
        # the interpreter/script wiring itself is broken.
        assert_true(r.returncode not in (126, 127),
                    f"`zl fix {name}` (dry run) found its real interpreter (exit {r.returncode}, not 126/127)",
                    extra=r.stdout + r.stderr)
        assert_true("Traceback (most recent call last):" not in (r.stdout + r.stderr), f"`zl fix {name}` (dry run) prints no traceback",
                    extra=r.stdout + r.stderr)


def test_fix_apply_stop_restore_flow_against_real_supervisord():
    write_conf = _docker_exec(f"cat > /etc/supervisor/conf.d/supervisord.conf <<'CONFEOF'\n{_SUPERVISORD_CONF}CONFEOF")
    assert_eq(write_conf.returncode, 0, "wrote the test supervisord.conf")
    start = _run(["docker", "exec", "-d", CONTAINER_NAME, "supervisord", "-c",
                  "/etc/supervisor/conf.d/supervisord.conf"], timeout=30)
    assert_eq(start.returncode, 0, "launched a real supervisord in the container")
    time.sleep(2)

    pre = _docker_exec("supervisorctl status")
    pre_states = _parse_supervisorctl_status(pre.stdout)
    for name in ("openwebui", "compactor", "backup"):
        assert_eq(pre_states.get(name), "RUNNING", f"{name} is RUNNING (autostart)", extra=pre.stdout)
    assert_eq(pre_states.get("webuidb-sync"), "STOPPED",
              "webuidb-sync starts pre-stopped, as configured", extra=pre.stdout)

    write_target = _docker_exec(f"cat > /tmp/fake_target.py <<'PYEOF'\n{_FAKE_TARGET_PY}PYEOF")
    assert_eq(write_target.returncode, 0, "wrote the fake target script")
    write_driver = _docker_exec(f"cat > /tmp/drive_fix.py <<'PYEOF'\n{_DRIVE_FIX_PY}PYEOF")
    assert_eq(write_driver.returncode, 0, "wrote the fix-flow driver script")

    r = _docker_exec("python3 /tmp/drive_fix.py")
    assert_eq(r.returncode, 0, "the fix-flow driver script exits 0", )
    report = json.loads(r.stdout.strip().splitlines()[-1])

    assert_eq(report["ok_code"], 0, "a successful apply exits 0")
    assert_true(report["ok_marker"], "a successful apply really ran the target script with --apply")
    assert_eq(report["fail_code"], 1, "a failing target script's exit code passes through unchanged")
    assert_true(report["kbdint_propagated"], "a KeyboardInterrupt mid-script propagates rather than being swallowed")

    for phase in ("pre", "post_ok", "post_fail", "post_kbdint"):
        state = report[phase]
        assert_eq(state.get("backup"), "RUNNING", f"{phase}: backup (untouched by this entry) stays RUNNING throughout")
        assert_eq(state.get("webuidb-sync"), "STOPPED",
                  f"{phase}: the pre-stopped program (webuidb-sync) stays stopped throughout")
    for phase in ("post_ok", "post_fail", "post_kbdint"):
        state = report[phase]
        assert_eq(state.get("openwebui"), "RUNNING", f"{phase}: openwebui was restored to RUNNING")
        assert_eq(state.get("compactor"), "RUNNING", f"{phase}: compactor was restored to RUNNING")


def _run_signalled_apply(signal_name: str) -> dict:
    """Launches the slow apply in the background, signals the WRAPPER
    process (not the child) mid-run, and returns what happened. Assumes
    the real supervisord from the previous test is already running with
    openwebui/compactor RUNNING.
    """
    for f in ("/tmp/wrapper.pid", "/tmp/wrapper_exit_code", "/tmp/slow_target.marker", "/tmp/slow_target.started"):
        _docker_exec(f"rm -f {f}")
    write_slow = _docker_exec(f"cat > /tmp/slow_target.py <<'PYEOF'\n{_SLOW_TARGET_PY}PYEOF")
    assert_eq(write_slow.returncode, 0, "wrote the slow target script")
    write_driver = _docker_exec(f"cat > /tmp/drive_slow_fix.py <<'PYEOF'\n{_DRIVE_SLOW_FIX_PY}PYEOF")
    assert_eq(write_driver.returncode, 0, "wrote the signal-test driver script")

    launch = _run(
        ["docker", "exec", "-d", CONTAINER_NAME, "bash", "-c",
         "python3 /tmp/drive_slow_fix.py > /tmp/drive_slow_fix.log 2>&1"],
        timeout=15,
    )
    assert_eq(launch.returncode, 0, "launched the slow apply in the background")

    # Round-4 LOW (flaky, reproduced 4/4 at 0.2 CPU): the OLD code polled
    # for wrapper.pid alone, THEN did a ONE-SHOT status check, THEN slept
    # a FIXED 1.0s before signalling -- three separate steps with real
    # docker-exec latency between each, wide enough under load for the
    # entire 8s apply (stop -> sleep -> restore) to race past unnoticed
    # (observed here: by the time of the old one-shot status check,
    # openwebui/compactor were already showing a FRESH restart, seconds
    # after the fact). Now: ONE combined poll for BOTH wrapper.pid and
    # the target's own started-marker (written as literally its first
    # action, once stop_for_apply has already completed and the child is
    # genuinely running) together, and the STOPPED check reads status
    # taken immediately alongside seeing that marker -- not from a
    # separately-timed, later query.
    wrapper_pid = None
    started = False
    mid = None
    for _ in range(80):
        if wrapper_pid is None:
            r = _docker_exec("cat /tmp/wrapper.pid 2>/dev/null")
            if r.returncode == 0 and r.stdout.strip():
                wrapper_pid = r.stdout.strip()
        if wrapper_pid is not None and _docker_exec("test -f /tmp/slow_target.started").returncode == 0:
            started = True
            mid = _docker_exec("supervisorctl status")
            break
        time.sleep(0.2)
    assert_true(wrapper_pid, "the wrapper process wrote its own pid (it started)")
    assert_true(started, "the slow target wrote its own started-marker before we signal anything")

    # We are now inside the target's 8s sleep -- confirm the stop already
    # happened (services down) before signalling, so this genuinely tests
    # "mid-apply", not "before it started". Checked on openwebui BY NAME,
    # not "STOPPED" anywhere in the block (round-4 LOW: webuidb-sync is
    # ALSO stopped here, pre-existing, so a bare substring check passed
    # even on a run where openwebui itself never actually stopped).
    assert_eq(_parse_supervisorctl_status(mid.stdout).get("openwebui"), "STOPPED",
              "openwebui is STOPPED mid-apply, before we signal anything", extra=mid.stdout)

    signalled = _docker_exec(f"kill -{signal_name} {wrapper_pid}")
    assert_eq(signalled.returncode, 0, f"sent {signal_name} to the wrapper's process group leader")

    # The wrapper must still be running the instant after a SIGHUP (it's
    # ignored); give SIGTERM a brief moment to be handled without yet
    # having exited (it forwards, then waits for the child).
    time.sleep(0.5)
    alive_right_after = _docker_exec(f"kill -0 {wrapper_pid}").returncode == 0
    debug_log = _docker_exec("cat /tmp/drive_slow_fix.log 2>/dev/null").stdout
    assert_true(alive_right_after,
                f"the wrapper process is STILL ALIVE right after {signal_name} (never killed by it)",
                extra=f"driver log:\n{debug_log}")

    exit_code = None
    for _ in range(60):
        r = _docker_exec("cat /tmp/wrapper_exit_code 2>/dev/null")
        if r.returncode == 0 and r.stdout.strip():
            exit_code = r.stdout.strip()
            break
        time.sleep(0.5)
    assert_true(exit_code is not None, f"the wrapper finished on its own after {signal_name} "
                "(never hung, never got killed out from under the restore)")

    marker = _docker_exec("test -f /tmp/slow_target.marker").returncode == 0
    status_after = _docker_exec("supervisorctl status").stdout
    audit = _docker_exec("cat /tmp/zl-home/audit.log 2>/dev/null").stdout
    return {"exit_code": exit_code, "marker": marker, "status": status_after, "audit": audit}


def test_sighup_mid_apply_never_kills_the_wrapper_and_restores_services():
    """review B1: a tty hangup delivers SIGHUP to the wrapper (it is still
    attached to the terminal even though its CHILD, in its own session,
    is not). The wrapper must ignore it, let the child run to completion,
    then restore and audit -- exactly as if nothing happened.
    """
    result = _run_signalled_apply("HUP")
    assert_true(result["marker"], "SIGHUP never interrupted the child -- it ran to completion",
                extra=result["status"])
    states = _parse_supervisorctl_status(result["status"])
    assert_eq(states.get("openwebui"), "RUNNING", "openwebui is RUNNING again after a SIGHUP mid-apply",
              extra=result["status"])
    assert_eq(states.get("compactor"), "RUNNING", "compactor is RUNNING again after a SIGHUP mid-apply",
              extra=result["status"])
    assert_true('"argv": ["fix", "t-slow"' in result["audit"], "an audit line was written",
                extra=result["audit"])


def test_sigterm_mid_apply_forwards_sigint_and_still_restores_services():
    """review B1: SIGTERM to the wrapper forwards SIGINT to the child (the
    scripts' own docstrings say an interrupt is safe), waits for it, then
    restores and audits -- the wrapper must never just die and leave
    openwebui/compactor stopped.
    """
    result = _run_signalled_apply("TERM")
    assert_true(not result["marker"],
                "SIGTERM interrupted the child before its 8s sleep finished (proves forwarding happened)",
                extra=result["status"])
    states = _parse_supervisorctl_status(result["status"])
    assert_eq(states.get("openwebui"), "RUNNING", "openwebui is RUNNING again after a SIGTERM mid-apply",
              extra=result["status"])
    assert_eq(states.get("compactor"), "RUNNING", "compactor is RUNNING again after a SIGTERM mid-apply",
              extra=result["status"])
    assert_true('"argv": ["fix", "t-slow"' in result["audit"], "an audit line was written",
                extra=result["audit"])


_OPENWEBUI_STUB_PY = """\
import http.server, sys, time

DELAY = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
START = time.time()


class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if time.time() - START < DELAY or self.path != "/api/config":
            self.send_response(503)
            self.end_headers()
            return
        body = b'{"ok": true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


http.server.HTTPServer(("127.0.0.1", 3000), H).serve_forever()
"""

_SUPERVISORD_CONF_HTTP_STUB = """\
[unix_http_server]
file=/var/run/supervisor.sock
chmod=0700

[supervisord]
nodaemon=true
logfile=/tmp/test-supervisord-stub.log
pidfile=/tmp/test-supervisord-stub.pid

[rpcinterface:supervisor]
supervisor.rpcinterface_factory = supervisor.rpcinterface:make_main_rpcinterface

[supervisorctl]
serverurl=unix:///var/run/supervisor.sock

[program:openwebui]
# Round-4 LOW: 12s, deliberately ABOVE common._PROGRESS_PRINT_INTERVAL_S
# (10.0) -- the OLD 3s delay meant `_wait_for_openwebui_http_ok`'s own
# heartbeat print (the exact code path review3 B1's probe found certain
# to raise OSError on a dead pty -- see the docstring on
# test_pty_close_during_restart_phase_with_real_openwebui_wait_survives_and_audits
# below) was NEVER ACTUALLY REACHED by either test using this stub: the
# wait always finished before the first heartbeat was ever due.
command=python3 /tmp/openwebui_stub.py 12
autostart=true
autorestart=true

[program:compactor]
command=sleep 99999
autostart=true
autorestart=true
"""

# R2-B1: drives the REAL restart-order wait (common._wait_for_openwebui_http_ok
# is NOT patched away here -- it polls the real stub HTTP server above) while
# a signal is sent mid-wait, proving the deferral (not a forward -- there is
# no child running during the wait) and the eventual 128+signal exit once
# services are verifiably back.
_DRIVE_REAL_WAIT_FIX_PY = """\
import os, sys
os.environ.setdefault("ZL_HOME", "/tmp/zl-home")
sys.path.insert(0, "/work/ops")
from pathlib import Path
Path("/tmp/wrapper.pid").write_text(str(os.getpid()))
from zlops import cli
from zlops.commands import fix
entry = fix.FixEntry(
    name="t-realwait", script="fast_target.py", interpreter=Path(sys.executable),
    read_only=False, dry_run_args=(), apply_args=("--apply",),
    services=lambda: ("openwebui", "compactor"), services_source="real-image real-wait signal test",
    default_args=lambda: (), help="test",
)
fix.REGISTRY["t-realwait"] = entry
from unittest import mock
with mock.patch.object(fix.podenv, "scripts_dir", return_value=Path("/tmp")):
    code = cli.main(["fix", "t-realwait", "--apply", "--yes", "--skip-backup"])
Path("/tmp/wrapper_exit_code").write_text(str(code))
"""

_FAST_TARGET_PY = """\
import sys
sys.exit(0)
"""

# R2-B1: drives a real stop_for_apply, slowed down artificially so there is
# a reliable window to signal the wrapper WHILE it is still in the stop
# phase (before any child exists to forward to) -- proving the signal is
# deferred, not dropped and not fatal, and that the eventual exit is
# 128+signal once the (unmodified) restore machinery has actually restored
# services.
_DRIVE_SLOW_STOP_FIX_PY = """\
import os, sys, time
os.environ.setdefault("ZL_HOME", "/tmp/zl-home")
sys.path.insert(0, "/work/ops")
from pathlib import Path
from unittest import mock
Path("/tmp/wrapper.pid").write_text(str(os.getpid()))
from zlops import cli, common as zl_common
from zlops.commands import fix

entry = fix.FixEntry(
    name="t-slowstop", script="fast_target.py", interpreter=Path(sys.executable),
    read_only=False, dry_run_args=(), apply_args=("--apply",),
    services=lambda: ("openwebui", "compactor"), services_source="real-image slow-stop signal test",
    default_args=lambda: (), help="test",
)
fix.REGISTRY["t-slowstop"] = entry

real_stop_for_apply = zl_common.stop_for_apply
def slow_stop(names, **kw):
    Path("/tmp/stop_phase_started").write_text("1")
    time.sleep(3)
    return real_stop_for_apply(names, **kw)

with mock.patch.object(fix.podenv, "scripts_dir", return_value=Path("/tmp")), \\
     mock.patch.object(zl_common, "stop_for_apply", side_effect=slow_stop), \\
     mock.patch.object(fix.common, "_wait_for_openwebui_http_ok", return_value=True):
    # openwebui here is `sleep 99999` (this driver reuses the plain
    # test_fix_apply_stop_restore_flow_against_real_supervisord conf) --
    # it will never answer /api/config for real, so (like _DRIVE_FIX_PY
    # and _DRIVE_SLOW_FIX_PY) the restart-order wait must not actually
    # block on it. Round-4: this mock was MISSING here -- the wrapper
    # hung for real waiting on a stub that does not exist, past this
    # test's own exit-code polling budget, every single run.
    code = cli.main(["fix", "t-slowstop", "--apply", "--yes", "--skip-backup"])
Path("/tmp/wrapper_exit_code").write_text(str(code))
"""


def _start_http_stub_supervisord() -> None:
    _run(["docker", "rm", "-f", CONTAINER_NAME], timeout=30)
    r = _run([
        "docker", "run", "-d", "--rm", "--name", CONTAINER_NAME,
        "-v", f"{REPO_ROOT}:/work:ro",
        "--entrypoint", "bash", IMAGE_REF, "-c", "sleep 3600",
    ], timeout=60)
    assert_eq(r.returncode, 0, "started a fresh container for the real-openwebui-wait signal tests")
    time.sleep(1)
    boot = _docker_exec("bash /work/ops/bootstrap.sh --from-dir /work --no-login-hook")
    assert_eq(boot.returncode, 0, "bootstrap installed zl in the fresh container")

    write_stub = _docker_exec(f"cat > /tmp/openwebui_stub.py <<'PYEOF'\n{_OPENWEBUI_STUB_PY}PYEOF")
    assert_eq(write_stub.returncode, 0, "wrote the openwebui HTTP stub")
    write_target = _docker_exec(f"cat > /tmp/fast_target.py <<'PYEOF'\n{_FAST_TARGET_PY}PYEOF")
    assert_eq(write_target.returncode, 0, "wrote the fast (immediate-exit) target script")
    write_conf = _docker_exec(
        f"cat > /etc/supervisor/conf.d/supervisord.conf <<'CONFEOF'\n{_SUPERVISORD_CONF_HTTP_STUB}CONFEOF"
    )
    assert_eq(write_conf.returncode, 0, "wrote the HTTP-stub supervisord.conf")
    start = _run(["docker", "exec", "-d", CONTAINER_NAME, "supervisord", "-c",
                  "/etc/supervisor/conf.d/supervisord.conf"], timeout=30)
    assert_eq(start.returncode, 0, "launched a real supervisord (openwebui = the HTTP stub)")
    time.sleep(2)
    pre = _docker_exec("supervisorctl status")
    pre_states = _parse_supervisorctl_status(pre.stdout)
    for name in ("openwebui", "compactor"):
        assert_eq(pre_states.get(name), "RUNNING", f"{name} is RUNNING (autostart)", extra=pre.stdout)


def test_real_openwebui_wait_signal_during_restart_phase():
    """R2-B1: send TERM while the wrapper is genuinely blocked polling the
    REAL openwebui stub's /api/config (not a patched-away wait) -- the
    signal must be deferred (no child to forward to at that point in
    restart_after_apply), and once openwebui actually answers 200 and
    compactor is verified back RUNNING, the wrapper must exit 128+SIGTERM
    (never silently 0) with a fresh audit line.
    """
    _start_http_stub_supervisord()
    for f in ("/tmp/wrapper.pid", "/tmp/wrapper_exit_code"):
        _docker_exec(f"rm -f {f}")
    write_driver = _docker_exec(f"cat > /tmp/drive_real_wait.py <<'PYEOF'\n{_DRIVE_REAL_WAIT_FIX_PY}PYEOF")
    assert_eq(write_driver.returncode, 0, "wrote the real-wait signal-test driver script")

    launch = _run(
        ["docker", "exec", "-d", CONTAINER_NAME, "bash", "-c",
         "python3 /tmp/drive_real_wait.py > /tmp/drive_real_wait.log 2>&1"],
        timeout=15,
    )
    assert_eq(launch.returncode, 0, "launched the real-wait apply in the background")

    wrapper_pid = None
    for _ in range(30):
        r = _docker_exec("cat /tmp/wrapper.pid 2>/dev/null")
        if r.returncode == 0 and r.stdout.strip():
            wrapper_pid = r.stdout.strip()
            break
        time.sleep(0.3)
    assert_true(wrapper_pid, "the wrapper process wrote its own pid (it started)")

    # The target script exits immediately; openwebui is then started and
    # its 12s-delayed stub polled. Signal partway through that real wait
    # (comfortably inside the 12s delay, comfortably after the fast target
    # already finished and the poll loop has begun).
    time.sleep(1.2)
    mid = _docker_exec("supervisorctl status")
    assert_true("openwebui" in mid.stdout, "openwebui is at least known to supervisorctl mid-wait", extra=mid.stdout)

    signalled = _docker_exec(f"kill -TERM {wrapper_pid}")
    assert_eq(signalled.returncode, 0, "sent TERM to the wrapper during the real openwebui wait")

    time.sleep(0.5)
    alive_mid_wait = _docker_exec(f"kill -0 {wrapper_pid}").returncode == 0
    debug_log = _docker_exec("cat /tmp/drive_real_wait.log 2>/dev/null").stdout
    assert_true(alive_mid_wait,
                "the wrapper is STILL ALIVE and still waiting right after TERM (deferred, not fatal)",
                extra=f"driver log:\n{debug_log}")

    exit_code = None
    # Round-4: budget raised (30 -> 40 polls, 15s -> 20s) to comfortably
    # cover the now-12s openwebui delay plus its own retry cadence.
    for _ in range(40):
        r = _docker_exec("cat /tmp/wrapper_exit_code 2>/dev/null")
        if r.returncode == 0 and r.stdout.strip():
            exit_code = r.stdout.strip()
            break
        time.sleep(0.5)
    assert_true(exit_code is not None, "the wrapper finished on its own once the real wait completed")
    assert_eq(exit_code, "143", "the deferred TERM (128+15) is the wrapper's own exit code, not 0")

    status_after = _docker_exec("supervisorctl status").stdout
    status_after_states = _parse_supervisorctl_status(status_after)
    assert_eq(status_after_states.get("openwebui"), "RUNNING",
              "openwebui really answered 200 for real and is RUNNING", extra=status_after)
    assert_eq(status_after_states.get("compactor"), "RUNNING",
              "compactor (started after openwebui's real wait) is RUNNING too", extra=status_after)
    audit = _docker_exec("cat /tmp/zl-home/audit.log 2>/dev/null").stdout
    assert_true('"argv": ["fix", "t-realwait"' in audit, "an audit line was written despite the signal exit",
                extra=audit)


def test_stop_phase_signal_is_deferred_then_exits_128_plus_signal():
    """R2-B1: the protected region starts BEFORE the first status
    read/stop, not just around the child. Signal the wrapper while it is
    genuinely inside `stop_for_apply` (no child running yet) and confirm
    it survives, still stops/restores correctly, and exits 128+INT.
    """
    for f in ("/tmp/wrapper.pid", "/tmp/wrapper_exit_code", "/tmp/stop_phase_started"):
        _docker_exec(f"rm -f {f}")
    write_driver = _docker_exec(f"cat > /tmp/drive_slow_stop.py <<'PYEOF'\n{_DRIVE_SLOW_STOP_FIX_PY}PYEOF")
    assert_eq(write_driver.returncode, 0, "wrote the slow-stop signal-test driver script")

    launch = _run(
        ["docker", "exec", "-d", CONTAINER_NAME, "bash", "-c",
         "python3 /tmp/drive_slow_stop.py > /tmp/drive_slow_stop.log 2>&1"],
        timeout=15,
    )
    assert_eq(launch.returncode, 0, "launched the slow-stop apply in the background")

    wrapper_pid = None
    for _ in range(30):
        r = _docker_exec("cat /tmp/wrapper.pid 2>/dev/null")
        if r.returncode == 0 and r.stdout.strip():
            wrapper_pid = r.stdout.strip()
            break
        time.sleep(0.3)
    assert_true(wrapper_pid, "the wrapper process wrote its own pid (it started)")

    for _ in range(30):
        if _docker_exec("test -f /tmp/stop_phase_started").returncode == 0:
            break
        time.sleep(0.2)
    else:
        assert_true(False, "the artificially slowed stop phase never started")

    signalled = _docker_exec(f"kill -INT {wrapper_pid}")
    assert_eq(signalled.returncode, 0, "sent INT to the wrapper during the (slowed) stop phase")

    time.sleep(0.5)
    alive_mid_stop = _docker_exec(f"kill -0 {wrapper_pid}").returncode == 0
    debug_log = _docker_exec("cat /tmp/drive_slow_stop.log 2>/dev/null").stdout
    assert_true(alive_mid_stop, "the wrapper is STILL ALIVE right after INT during the stop phase (deferred)",
                extra=f"driver log:\n{debug_log}")

    exit_code = None
    for _ in range(30):
        r = _docker_exec("cat /tmp/wrapper_exit_code 2>/dev/null")
        if r.returncode == 0 and r.stdout.strip():
            exit_code = r.stdout.strip()
            break
        time.sleep(0.5)
    assert_true(exit_code is not None, "the wrapper finished on its own after the deferred INT",
                extra=_docker_exec("cat /tmp/drive_slow_stop.log 2>/dev/null").stdout)
    assert_eq(exit_code, "130", "the deferred INT (128+2) is the wrapper's own exit code")

    status_after = _docker_exec("supervisorctl status").stdout
    status_after_states = _parse_supervisorctl_status(status_after)
    assert_eq(status_after_states.get("openwebui"), "RUNNING",
              "openwebui is RUNNING again despite the signal during the stop phase", extra=status_after)
    assert_eq(status_after_states.get("compactor"), "RUNNING",
              "compactor is RUNNING again too", extra=status_after)


# =============================================================================
# Round-3 review B1: a REAL disconnect, not just `kill -HUP`. sshd closing
# the pty master delivers SIGHUP the same as `kill -HUP` does, but every
# print AFTER that (ours, or the copy of the child's own output in
# run_protected) used to raise `OSError: [Errno 5]` writing to the now-dead
# terminal -- only run_protected's own child-output copy caught that, so a
# hangup landing in `fix.py`'s own "stopped ..." print (between
# `stop_for_apply` succeeding and `run_protected` starting) or in
# `restart_after_apply`'s own WARNING/heartbeat prints escaped every
# try/finally with services left stopped and no audit line. The review's
# own probe (`review3/pty/ptyhup.py`) is what found this: `pty.fork()`,
# run the wrapper on the slave end, then CLOSE THE MASTER -- exactly what
# an sshd process does on a dropped connection. Reproduced here against
# the real image / real supervisord, in both windows.
# =============================================================================

_PTY_CLOSE_PROBE_PY = """\
import os, pty, select, subprocess, sys, time

phase = sys.argv[1]   # "stop" or "restart"
driver = sys.argv[2]

pid, master = pty.fork()
if pid == 0:
    os.execvp("python3", ["python3", driver])


def pump(until):
    while time.time() < until:
        r, _, _ = select.select([master], [], [], 0.2)
        if r:
            try:
                os.read(master, 4096)
            except OSError:
                return


if phase == "stop":
    # the driver's own stop_for_apply is artificially slowed (3s) and
    # marks /tmp/stop_phase_started the instant it enters that window --
    # close the pty comfortably inside it, well before any child exists.
    deadline = time.time() + 15
    while not os.path.exists("/tmp/stop_phase_started") and time.time() < deadline:
        pump(time.time() + 0.2)
    pump(time.time() + 1.0)
else:
    # restart phase: the target exits immediately; give the wrapper time
    # to start openwebui and genuinely be polling its (3s-delayed) stub.
    pump(time.time() + 1.5)

print("== closing pty master (simulated SSH drop) at phase", phase, flush=True)
os.close(master)
close_time = time.time()

# The wrapper is now "disconnected" -- give it the FULL time it needs to
# finish entirely unattended (no terminal at all from here on).
deadline = time.time() + 40
exit_code = None
while time.time() < deadline:
    if os.path.exists("/tmp/wrapper_exit_code"):
        exit_code = open("/tmp/wrapper_exit_code").read().strip()
        break
    time.sleep(0.5)
elapsed_since_close = time.time() - close_time

alive = subprocess.run(["kill", "-0", str(pid)]).returncode == 0
print("WRAPPER_ALIVE_AFTER_CLOSE", alive)
print("WRAPPER_EXIT_CODE", exit_code)
print("ELAPSED_SINCE_CLOSE", elapsed_since_close)
print("STATUS_START")
print(subprocess.run(["supervisorctl", "status"], capture_output=True, text=True).stdout)
print("STATUS_END")
print("AUDIT_START")
try:
    print(open("/tmp/zl-home/audit.log").read())
except FileNotFoundError:
    print("(no audit log)")
print("AUDIT_END")
try:
    wpid, st = os.waitpid(pid, os.WNOHANG)
    print("WAITPID", wpid, st)
except ChildProcessError:
    pass
"""


def test_pty_close_during_stop_phase_survives_and_restores_and_audits():
    """review3 B1 (the round's own blocker): close a REAL pty master while
    the wrapper is genuinely inside `stop_for_apply` (no child running
    yet). Before the fix, this reliably killed the wrapper outright
    (`OSError` from fix.py's own "stopped ..." print, escaping every
    try/finally) with openwebui/compactor left STOPPED and no audit line.
    SIGHUP does not set `ProtectedRegion.deferred_signal` (unlike SIGINT/
    SIGTERM) -- "keep going", not "defer and act later" -- so a
    successful apply finishes with its own normal exit code (0), not
    128+something.
    """
    for f in ("/tmp/wrapper.pid", "/tmp/wrapper_exit_code", "/tmp/stop_phase_started", "/tmp/zl-home/audit.log"):
        _docker_exec(f"rm -f {f}")
    write_driver = _docker_exec(f"cat > /tmp/drive_slow_stop.py <<'PYEOF'\n{_DRIVE_SLOW_STOP_FIX_PY}PYEOF")
    assert_eq(write_driver.returncode, 0, "wrote the slow-stop driver script")
    write_probe = _docker_exec(f"cat > /tmp/pty_close_probe.py <<'PYEOF'\n{_PTY_CLOSE_PROBE_PY}PYEOF")
    assert_eq(write_probe.returncode, 0, "wrote the pty-close probe")

    r = _docker_exec("python3 /tmp/pty_close_probe.py stop /tmp/drive_slow_stop.py", timeout=60)
    assert_eq(r.returncode, 0, "the pty-close probe itself ran to completion", extra=r.stdout + r.stderr)
    assert_true("WRAPPER_ALIVE_AFTER_CLOSE True" in r.stdout,
                "the wrapper survived the real pty close during the stop phase", extra=r.stdout)
    assert_true("WRAPPER_EXIT_CODE 0" in r.stdout,
                "SIGHUP alone never turns into a nonzero/128+signal exit -- the apply just finishes",
                extra=r.stdout)
    assert_true("openwebui" in r.stdout and "compactor" in r.stdout, "supervisorctl status was captured",
                extra=r.stdout)
    status_block = r.stdout.split("STATUS_START", 1)[1].split("STATUS_END", 1)[0]
    status_states = _parse_supervisorctl_status(status_block)
    for name in ("openwebui", "compactor"):  # this driver's own entry -- see _DRIVE_SLOW_STOP_FIX_PY
        assert_eq(status_states.get(name), "RUNNING",
                  f"{name} is RUNNING in supervisorctl status after the real disconnect",
                  extra=status_block)
    audit_block = r.stdout.split("AUDIT_START", 1)[1].split("AUDIT_END", 1)[0]
    assert_true('"argv": ["fix", "t-slowstop"' in audit_block,
                "the run log in ZL_HOME/runs kept receiving output and the audit line was written "
                "despite the real disconnect", extra=audit_block)


def test_pty_close_during_restart_phase_with_real_openwebui_wait_survives_and_audits():
    """review3 B1: close a REAL pty master while the wrapper is genuinely
    blocked polling the REAL openwebui stub's /api/config in
    `restart_after_apply` (the exact window the review's own probe found
    certain to fail: `_wait_for_openwebui_http_ok`'s 10s heartbeat print
    used to raise OSError on a dead terminal). Confirms all services come
    back RUNNING and the audit line survives.
    """
    for f in ("/tmp/wrapper.pid", "/tmp/wrapper_exit_code", "/tmp/zl-home/audit.log"):
        _docker_exec(f"rm -f {f}")
    write_driver = _docker_exec(f"cat > /tmp/drive_real_wait.py <<'PYEOF'\n{_DRIVE_REAL_WAIT_FIX_PY}PYEOF")
    assert_eq(write_driver.returncode, 0, "wrote the real-wait driver script")
    write_probe = _docker_exec(f"cat > /tmp/pty_close_probe.py <<'PYEOF'\n{_PTY_CLOSE_PROBE_PY}PYEOF")
    assert_eq(write_probe.returncode, 0, "wrote the pty-close probe")

    r = _docker_exec("python3 /tmp/pty_close_probe.py restart /tmp/drive_real_wait.py", timeout=60)
    assert_eq(r.returncode, 0, "the pty-close probe itself ran to completion", extra=r.stdout + r.stderr)
    assert_true("WRAPPER_ALIVE_AFTER_CLOSE True" in r.stdout,
                "the wrapper survived the real pty close during the real openwebui wait", extra=r.stdout)
    assert_true("WRAPPER_EXIT_CODE 0" in r.stdout,
                "SIGHUP alone never turns into a nonzero/128+signal exit -- the apply just finishes",
                extra=r.stdout)
    # Round-4 LOW: the pty closes ~1.5s after launch (well before
    # openwebui's now-12s-delayed stub answers), so the wrapper spends
    # the REST of the wait -- more than common._PROGRESS_PRINT_INTERVAL_S
    # (10.0) -- fully disconnected. This is the actual proof that
    # `_wait_for_openwebui_http_ok`'s heartbeat print was reached (and
    # survived) with no terminal at all, not merely inferred from timing
    # assumptions this test used to just assert in its docstring.
    elapsed_line = next(ln for ln in r.stdout.splitlines() if ln.startswith("ELAPSED_SINCE_CLOSE"))
    elapsed_since_close = float(elapsed_line.split()[1])
    assert_true(elapsed_since_close > 10.0,
                "the wrapper was disconnected for more than the 10s heartbeat interval, "
                "so the heartbeat print path was actually exercised", extra=r.stdout)
    status_block = r.stdout.split("STATUS_START", 1)[1].split("STATUS_END", 1)[0]
    status_states = _parse_supervisorctl_status(status_block)
    for name in ("openwebui", "compactor"):
        assert_eq(status_states.get(name), "RUNNING",
                  f"{name} is RUNNING after the real disconnect during the real openwebui wait",
                  extra=status_block)
    audit_block = r.stdout.split("AUDIT_START", 1)[1].split("AUDIT_END", 1)[0]
    assert_true('"argv": ["fix", "t-realwait"' in audit_block,
                "the audit line was written despite the real disconnect during the openwebui wait",
                extra=audit_block)


def test_env_i_backup_now_and_fix_dry_run_see_pid1s_service_env():
    """review B3: an SSH/Web Terminal session lacks WEBUI_DB_LOCAL etc --
    only pid 1 (supervisord in real life; here, this container's own
    entrypoint) carries it. Restart the container with WEBUI_DB_LOCAL=false
    baked into PID 1's own environment, then run `zl backup now` and a `zl
    fix ...` dry run under `env -i` (an exec session with NOTHING
    inherited).

    `zl backup now` is the REAL proof of the fix: `compactor/backup.py`'s
    `live_webui_db()` -> `webuidb.live_webui_db()` reads
    `os.environ.get("WEBUI_DB_LOCAL", ...)` directly, no fallback of its
    own -- it can only see the right value if `common.run_stream` actually
    passed `common.merged_environ()` as this CHILD PROCESS's own env.

    `zl version` (which prints `WEBUI_DB_LOCAL: <bool> (<reason>)` via
    podenv.detect_webui_db_local()) checks a related but DIFFERENT
    property: podenv.py's own detection already reads `/proc/1/environ`
    itself as a fallback (scripts/_webui_live_path.py), independent of
    subprocess env-passing -- so this assertion is not a mutation-proof of
    the merged_environ() fix, but it is still a real regression check that
    the WRAPPER's own env resolution keeps working end-to-end under env -i.
    (`zl fix repair-chat-tree`'s dry run would be a closer parallel to the
    backup check, but on an empty container with no real webui.db it
    raises before printing anything useful -- an unrelated, pre-existing
    property of that script, not exercised here.)
    """
    _run(["docker", "rm", "-f", CONTAINER_NAME], timeout=30)
    r = _run([
        "docker", "run", "-d", "--rm", "--name", CONTAINER_NAME,
        "-v", f"{REPO_ROOT}:/work:ro",
        "-e", "WEBUI_DB_LOCAL=false",
        "--entrypoint", "bash", IMAGE_REF, "-c", "sleep 600",
    ], timeout=60)
    assert_eq(r.returncode, 0, "started a container with WEBUI_DB_LOCAL=false only on PID 1")
    time.sleep(1)
    boot = _docker_exec("bash /work/ops/bootstrap.sh --from-dir /work --no-login-hook")
    assert_eq(boot.returncode, 0, "bootstrap installed zl in this fresh container")

    # `env -i`: an exec session that inherits NOTHING -- not even PATH.
    # zl itself needs PATH to find python3/bash; WEBUI_DB_LOCAL is
    # deliberately absent (proving the CHILD's view of it can only have
    # come from pid 1, never from this shell).
    backup = _run([
        "docker", "exec", CONTAINER_NAME, "env", "-i", "PATH=/usr/local/bin:/usr/bin:/bin",
        "zl", "backup", "now",
    ], timeout=30)
    assert_true("/data/openwebui/webui.db" in (backup.stdout + backup.stderr),
                "`zl backup now` under env -i refused at the WEBUI_DB_LOCAL=false path "
                "(/data/openwebui/webui.db), proving the child read pid 1's env, not this "
                "(empty) shell's", extra=backup.stdout + backup.stderr)
    assert_true("/var/lib/openwebui/webui.db" not in (backup.stdout + backup.stderr),
                "it did NOT fall back to the WEBUI_DB_LOCAL=true/default path",
                extra=backup.stdout + backup.stderr)

    version = _run([
        "docker", "exec", CONTAINER_NAME, "env", "-i", "PATH=/usr/local/bin:/usr/bin:/bin",
        "zl", "version",
    ], timeout=30)
    assert_true("WEBUI_DB_LOCAL: False" in (version.stdout + version.stderr),
                "`zl version` under env -i also reports WEBUI_DB_LOCAL: False (from pid 1, not "
                "this empty shell)", extra=version.stdout + version.stderr)


_SUPERVISORCTL_PGID_PROBE_PY = """\
import os, sys
sys.path.insert(0, "/work/ops")
from pathlib import Path
from zlops import common
Path("/tmp/wrapper_pgid").write_text(str(os.getpgrp()))
common.supervisorctl_stop(["nonexistent-probe-target"])
common.supervisorctl_start(["nonexistent-probe-target"])
"""

_FAKE_SUPERVISORCTL_RECORDS_PGID_SH = """\
#!/bin/bash
ps -o pgid= -p $$ | tr -d ' ' >> /tmp/supervisorctl_child_pgids
exit 0
"""


def test_supervisorctl_stop_and_start_run_in_their_own_session_not_the_wrappers():
    """HIGH (round-6, run_grp_start.sh's own concern): a Ctrl-C at a real
    terminal delivers SIGINT to the whole FOREGROUND PROCESS GROUP, not
    just to `zl` itself -- before `start_new_session=True`
    (`supervisorctl_stop`/`supervisorctl_start`, common.py), the
    `supervisorctl` child inherited the wrapper's own process group and
    was just as exposed to that group-wide signal, able to be killed
    mid-command with services left in whatever partial state the command
    reached. This proves it end to end in the REAL container (not a
    mocked subprocess.run kwarg, as the unit test already covers): a fake
    `supervisorctl` on PATH records its OWN pgid, and it must differ from
    the wrapper's.
    """
    for f in ("/tmp/wrapper_pgid", "/tmp/supervisorctl_child_pgids"):
        _docker_exec(f"rm -f {f}")
    write_fake = _docker_exec(
        "mkdir -p /tmp/fakebin && cat > /tmp/fakebin/supervisorctl <<'SHEOF'\n"
        f"{_FAKE_SUPERVISORCTL_RECORDS_PGID_SH}SHEOF\n"
        "chmod +x /tmp/fakebin/supervisorctl"
    )
    assert_eq(write_fake.returncode, 0, "wrote a fake supervisorctl that records its own pgid")
    write_probe = _docker_exec(f"cat > /tmp/pgid_probe.py <<'PYEOF'\n{_SUPERVISORCTL_PGID_PROBE_PY}PYEOF")
    assert_eq(write_probe.returncode, 0, "wrote the pgid probe driver")

    # setsid: the probe itself starts its own session, mimicking a real
    # interactive shell's foreground process group -- the property under
    # test is that supervisorctl's OWN pgid differs from THIS one, i.e.
    # `start_new_session=True` actually took effect for the real child,
    # not merely that the kwarg was passed (the unit test's own coverage).
    r = _docker_exec("setsid env PATH=/tmp/fakebin:$PATH python3 /tmp/pgid_probe.py")
    assert_eq(r.returncode, 0, "the probe ran to completion", extra=r.stdout + r.stderr)

    wrapper_pgid = _docker_exec("cat /tmp/wrapper_pgid").stdout.strip()
    child_pgids = _docker_exec("cat /tmp/supervisorctl_child_pgids").stdout.split()
    assert_true(bool(wrapper_pgid), "the wrapper recorded its own pgid")
    assert_eq(len(child_pgids), 2, "both supervisorctl_stop and supervisorctl_start ran the fake binary",
              extra=str(child_pgids))
    for pgid in child_pgids:
        assert_true(pgid != wrapper_pgid,
                     f"supervisorctl's child pgid ({pgid}) differs from the wrapper's ({wrapper_pgid}) -- "
                     "start_new_session=True actually detached it", extra=str(child_pgids))


if __name__ == "__main__":
    _preflight()
    test_boot_from_read_only_mount_with_no_network_installs_zl_and_exits_0()
    try:
        _start_container()
        test_bootstrap_from_dir_installs_zl()
        test_help_and_version()
        test_health_no_services_is_crit_with_a_clean_message()
        test_dash_once_and_json()
        test_bundle_layout_and_no_secret_leak()
        test_real_registry_scripts_dry_run_with_their_real_interpreter()
        test_fix_apply_stop_restore_flow_against_real_supervisord()
        test_sighup_mid_apply_never_kills_the_wrapper_and_restores_services()
        test_sigterm_mid_apply_forwards_sigint_and_still_restores_services()
        test_real_openwebui_wait_signal_during_restart_phase()
        test_stop_phase_signal_is_deferred_then_exits_128_plus_signal()
        test_pty_close_during_stop_phase_survives_and_restores_and_audits()
        test_pty_close_during_restart_phase_with_real_openwebui_wait_survives_and_audits()
        test_env_i_backup_now_and_fix_dry_run_see_pid1s_service_env()
        test_supervisorctl_stop_and_start_run_in_their_own_session_not_the_wrappers()
        print("\nAll real-image ops-bundle tests passed.")
    finally:
        _stop_container()
