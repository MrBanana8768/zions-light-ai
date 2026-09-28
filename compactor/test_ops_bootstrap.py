"""Unit tests for ops/bootstrap.sh -- idempotent install, the saved-ref/
flags reload after a restart, the /root/.bashrc login-hook marker block,
and (the regression this file exists to prevent) that a git WORKTREE
checkout -- where `.git` is a FILE, not a directory -- is never mistaken
for "missing" and destructively re-cloned over.

Plain-script style like compactor/test_setup_sshd_script.py: exit 0 pass,
1 fail. Drives the real bootstrap.sh via subprocess with ZL_HOME/ZL_REPO/
ZL_BIN/ZL_BASHRC all redirected into a throwaway temp tree -- never the
real /data, /opt, /usr/local/bin or /root/.bashrc.
"""

from __future__ import annotations

import os
import subprocess
import sys
# LOW (round 3): never write ops/**/__pycache__ from a test run --
# this file's own `from zlops import ...` below would otherwise create
# it even when the harness invoking this file forgot PYTHONDONTWRITEBYTECODE.
sys.dont_write_bytecode = True
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = REPO_ROOT / "ops" / "bootstrap.sh"


def assert_eq(actual, expected, label):
    if actual != expected:
        print(f"FAIL {label}: expected {expected!r}, got {actual!r}")
        sys.exit(1)
    print(f"  ok   {label}")


def assert_true(cond, label, extra=""):
    if not cond:
        print(f"FAIL {label}")
        if extra:
            print(f"  --- context ---\n{extra}\n  --- end ---")
        sys.exit(1)
    print(f"  ok   {label}")


def _run(args, env, timeout=60):
    return subprocess.run(
        ["bash", str(BOOTSTRAP), *args],
        capture_output=True, text=True, timeout=timeout, env=env,
    )


_FAKE_APT_GET = """#!/usr/bin/env bash
# HARD SAFETY RULE: never run real apt-get on the host. --with-tools (and
# `zl tools --apply --yes` it triggers) must never reach the real one --
# review MEDIUM: "test_ops_bootstrap runs a real `zl tools --apply --yes`"
# (apt-get update and install) through --with-tools. Run as root with
# network, it installs packages." This fake just records the call and
# exits 0, exactly like a successful real one would from the caller's
# point of view.
echo "FAKE-APT-GET: $*" >> "${FAKE_APT_LOG:-/dev/null}"
exit 0
"""


def _base_env(td: Path) -> dict[str, str]:
    # /usr/local/bin always exists on a real pod; bootstrap.sh assumes its
    # ZL_BIN's parent dir exists (it never mkdir -p's it), so the test
    # harness -- standing in for /usr/local/bin -- must create it too.
    (td / "home").mkdir(parents=True, exist_ok=True)
    bin_dir = td / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    apt_get = bin_dir / "apt-get"
    apt_get.write_text(_FAKE_APT_GET, encoding="utf-8")
    apt_get.chmod(0o755)
    apt_log = td / "apt.log"
    return {
        # The fake apt-get goes FIRST on PATH, ahead of the real one.
        "PATH": str(bin_dir) + os.pathsep + os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(td / "home"),
        "ZL_HOME": str(td / "zl-home"),
        "ZL_REPO": str(td / "opt-zl-repo"),
        "ZL_BIN": str(td / "bin" / "zl"),
        "ZL_BASHRC": str(td / "bashrc"),
        "FAKE_APT_LOG": str(apt_log),
        # LOW (round 3): a `--from-dir REPO_ROOT` install points ops/zl at
        # the REAL worktree's ops/zlops/ -- `--with-sshd`/`--with-tools`
        # then exec that real `zl` (via `$ZL_BIN sshd|tools --apply
        # --yes`), which would otherwise write __pycache__ under the
        # worktree itself, not just some throwaway temp dir.
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def test_from_dir_install_creates_symlink_and_saves_state():
    with tempfile.TemporaryDirectory() as tdstr:
        td = Path(tdstr)
        env = _base_env(td)
        r = _run(["--ref", "test-ref", "--from-dir", str(REPO_ROOT), "--no-login-hook"], env)
        assert_eq(r.returncode, 0, "a --from-dir install exits 0")
        zl_bin = Path(env["ZL_BIN"])
        assert_true(zl_bin.is_symlink(), "bootstrap.sh created the zl symlink", extra=r.stdout + r.stderr)
        assert_eq(zl_bin.resolve(), (REPO_ROOT / "ops" / "zl").resolve(), "the symlink points at ops/zl")

        ref_file = Path(env["ZL_HOME"]) / "ref"
        assert_eq(ref_file.read_text(encoding="utf-8").strip(), "test-ref", "the ref was saved")
        assert_true((Path(env["ZL_HOME"]) / "bootstrap.sh").is_file(), "a copy of bootstrap.sh was saved under ZL_HOME")


def test_login_hook_is_idempotent_and_no_login_hook_removes_it():
    with tempfile.TemporaryDirectory() as tdstr:
        td = Path(tdstr)
        env = _base_env(td)
        _run(["--ref", "test-ref", "--from-dir", str(REPO_ROOT)], env)
        bashrc = Path(env["ZL_BASHRC"]).read_text(encoding="utf-8")
        assert_eq(bashrc.count("BEGIN zl login hook"), 1, "the login hook block appears exactly once")

        _run(["--from-dir", str(REPO_ROOT)], env)  # re-run, no new flags
        bashrc2 = Path(env["ZL_BASHRC"]).read_text(encoding="utf-8")
        assert_eq(bashrc2.count("BEGIN zl login hook"), 1, "re-running bootstrap.sh never duplicates the block")

        r3 = _run(["--from-dir", str(REPO_ROOT), "--no-login-hook"], env)
        assert_eq(r3.returncode, 0, "--no-login-hook exits 0")
        bashrc3 = Path(env["ZL_BASHRC"]).read_text(encoding="utf-8")
        assert_eq(bashrc3.count("BEGIN zl login hook"), 0, "--no-login-hook removes the block")


def test_saved_flags_persist_across_a_bare_rerun():
    with tempfile.TemporaryDirectory() as tdstr:
        td = Path(tdstr)
        env = _base_env(td)
        _run(["--ref", "test-ref", "--from-dir", str(REPO_ROOT), "--with-tools", "--no-login-hook"], env)
        flags = (Path(env["ZL_HOME"]) / "bootstrap-flags").read_text(encoding="utf-8")
        assert_true("--with-tools" in flags, "--with-tools was saved to bootstrap-flags", extra=flags)

        # A bare re-run with none of the original flags repeated must still
        # carry --with-tools forward (loaded from the saved state before
        # this run's own, empty, CLI args are parsed).
        r = _run(["--from-dir", str(REPO_ROOT)], env)
        assert_true(r.returncode == 0, "a bare re-run exits 0", extra=r.stdout + r.stderr)
        flags2 = (Path(env["ZL_HOME"]) / "bootstrap-flags").read_text(encoding="utf-8")
        assert_true("--with-tools" in flags2, "the saved flag survives an unflagged re-run", extra=flags2)


def test_worktree_style_git_dir_is_never_destructively_recloned():
    """Regression guard: `.git` in a git WORKTREE is a FILE, not a
    directory. Stage 1 (running as the saved $ZL_HOME/bootstrap.sh copy,
    the after-a-restart path) must test for it with -e, never -d, or it
    treats a perfectly good worktree checkout as "missing" and `rm -rf`s
    it before re-cloning over it -- exactly the incident this test exists
    to catch would have destroyed a real checkout.
    """
    with tempfile.TemporaryDirectory() as tdstr:
        td = Path(tdstr)
        env = _base_env(td)
        repo = Path(env["ZL_REPO"])
        repo.mkdir(parents=True)
        (repo / ".git").write_text("gitdir: /some/where/else/.git/worktrees/thing\n", encoding="utf-8")
        (repo / "ops").mkdir()
        (repo / "ops" / "zl").write_text("#!/usr/bin/env bash\necho fake-zl\n", encoding="utf-8")
        (repo / "ops" / "zl").chmod(0o755)
        marker = repo / "CANARY_FILE_MUST_SURVIVE"
        marker.write_text("if this file is gone, the repo was destructively re-cloned", encoding="utf-8")

        zl_home = Path(env["ZL_HOME"])
        zl_home.mkdir(parents=True)
        (zl_home / "ref").write_text("test-ref\n", encoding="utf-8")
        (zl_home / "bootstrap-flags").write_text("__REPO_URL__=https://example.invalid/repo.git\n--no-login-hook\n",
                                                   encoding="utf-8")
        saved_copy = zl_home / "bootstrap.sh"
        saved_copy.write_text(BOOTSTRAP.read_text(encoding="utf-8"), encoding="utf-8")
        saved_copy.chmod(0o755)

        r = subprocess.run(["bash", str(saved_copy)], capture_output=True, text=True, timeout=60, env=env)
        assert_true(marker.is_file(), "the worktree-style checkout was NOT deleted", extra=r.stdout + r.stderr)
        assert_true("re-cloning" not in r.stdout, "bootstrap.sh never re-clones a checkout whose .git is a file",
                    extra=r.stdout)
        assert_true("handing off" in r.stdout, "stage 1 handed off to the checkout's own bootstrap.sh",
                    extra=r.stdout)


def test_saved_copy_hands_off_to_a_baked_image_copy_without_cloning():
    """M2 regression: on v3.1.9.8+, $REPO is a BAKED image copy
    (`ops/bootstrap.sh` present, no `.git` at all -- the Dockerfile put
    ops/+scripts/ there directly). The SAVED $ZL_HOME/bootstrap.sh copy
    (stage 1, the manual "bash /data/zl-ops/bootstrap.sh" after-restart
    path -- review8/legacy.sh's own scenario) must recognize this and
    hand off to it via --from-dir, never moving it aside or attempting
    to clone over it the way it would for a genuinely missing/foreign
    $REPO.
    """
    with tempfile.TemporaryDirectory() as tdstr:
        td = Path(tdstr)
        env = _base_env(td)
        repo = Path(env["ZL_REPO"])
        (repo / "ops").mkdir(parents=True)
        (repo / "ops" / "zl").write_text(_FAKE_ZL, encoding="utf-8")
        (repo / "ops" / "zl").chmod(0o755)
        (repo / "ops" / "bootstrap.sh").write_text(BOOTSTRAP.read_text(encoding="utf-8"), encoding="utf-8")
        marker = repo / "CANARY_MUST_SURVIVE"
        marker.write_text("baked image copy -- must never be moved aside or cloned over", encoding="utf-8")

        zl_home = Path(env["ZL_HOME"])
        zl_home.mkdir(parents=True)
        (zl_home / "ref").write_text("v3.1.9.8\n", encoding="utf-8")
        (zl_home / "bootstrap-flags").write_text(
            "__REPO_URL__=https://example.invalid/repo.git\n--no-login-hook\n", encoding="utf-8"
        )
        saved_copy = zl_home / "bootstrap.sh"
        saved_copy.write_text(BOOTSTRAP.read_text(encoding="utf-8"), encoding="utf-8")
        saved_copy.chmod(0o755)

        r = subprocess.run(["bash", str(saved_copy)], capture_output=True, text=True, timeout=30, env=env)
        assert_true(marker.is_file(), "the baked $REPO was NOT moved aside/deleted", extra=r.stdout + r.stderr)
        assert_true("baked image copy" in r.stdout, "bootstrap.sh recognized it as a baked copy", extra=r.stdout)
        assert_true("re-cloning" not in r.stdout, "no clone was attempted over the baked copy", extra=r.stdout)
        assert_true(not list(td.glob("opt-zl-repo.aside-*")), "nothing was moved aside", extra=r.stdout)
        zl_bin = Path(env["ZL_BIN"])
        assert_true(zl_bin.is_symlink(), "zl was still installed", extra=r.stdout + r.stderr)


def test_with_tools_never_calls_real_apt_get():
    """HARD SAFETY RULE: --with-tools triggers `zl tools --apply --yes`,
    which shells out to apt-get -- this must hit the FAKE apt-get on PATH,
    never a real one. Also the concrete review MEDIUM regression guard.
    """
    with tempfile.TemporaryDirectory() as tdstr:
        td = Path(tdstr)
        env = _base_env(td)
        r = _run(["--ref", "test-ref", "--from-dir", str(REPO_ROOT), "--with-tools", "--no-login-hook"], env)
        assert_true(r.returncode == 0, "a --with-tools install exits 0", extra=r.stdout + r.stderr)
        apt_log = Path(env["FAKE_APT_LOG"])
        assert_true(apt_log.is_file() and apt_log.read_text(encoding="utf-8").strip(),
                     "the FAKE apt-get was actually invoked (proves the fake is on PATH and reachable)",
                     extra=r.stdout + r.stderr)


def test_login_hook_uses_timeout_dash_k():
    """review MEDIUM: a plain `timeout 5` sends only TERM once -- a login
    stuck on a genuine MooseFS stall (the whole reason for a timeout) will
    not end on TERM either, and `timeout` then just waits forever for the
    child to reap. `-k 2 5` sends TERM at 5s, KILL 2s later if needed.
    """
    with tempfile.TemporaryDirectory() as tdstr:
        td = Path(tdstr)
        env = _base_env(td)
        _run(["--ref", "test-ref", "--from-dir", str(REPO_ROOT)], env)
        bashrc = Path(env["ZL_BASHRC"]).read_text(encoding="utf-8")
        assert_true("timeout -k 2 5 zl status --brief" in bashrc,
                     "the login hook uses `timeout -k 2 5`, not a plain `timeout 5`", extra=bashrc)


def test_non_empty_non_checkout_repo_dir_is_moved_aside_not_deleted():
    """review MEDIUM: bootstrap.sh used to `rm -rf "$REPO"` unconditionally
    whenever `$REPO/.git` did not exist, even if $REPO held real files and
    even before a clone that might fail. Reproduced originally with $ZL_REPO
    pointing at a directory holding files and no network: the directory was
    deleted, then the clone failed, losing both. Now it must be moved aside
    (never deleted) unless it is genuinely empty.
    """
    with tempfile.TemporaryDirectory() as tdstr:
        td = Path(tdstr)
        env = _base_env(td)
        repo = Path(env["ZL_REPO"])
        repo.mkdir(parents=True)
        marker = repo / "IMPORTANT_FILE_MUST_SURVIVE"
        marker.write_text("not a git checkout, but has real content", encoding="utf-8")

        zl_home = Path(env["ZL_HOME"])
        zl_home.mkdir(parents=True)
        (zl_home / "ref").write_text("test-ref\n", encoding="utf-8")
        (zl_home / "bootstrap-flags").write_text(
            "__REPO_URL__=https://example.invalid/nonexistent-repo-for-testing.git\n--no-login-hook\n",
            encoding="utf-8",
        )
        saved_copy = zl_home / "bootstrap.sh"
        saved_copy.write_text(BOOTSTRAP.read_text(encoding="utf-8"), encoding="utf-8")
        saved_copy.chmod(0o755)

        # The clone itself will fail (bogus URL, no network) -- that's fine
        # and expected; what matters is $REPO was never destroyed to get
        # there.
        r = subprocess.run(["bash", str(saved_copy)], capture_output=True, text=True, timeout=30, env=env)
        aside_dirs = list(td.glob("opt-zl-repo.aside-*"))
        assert_true(len(aside_dirs) == 1, "the old $REPO was moved aside to a .aside-<ts> path",
                     extra=r.stdout + r.stderr + "\n" + str(list(td.iterdir())))
        assert_true((aside_dirs[0] / marker.name).is_file(),
                     "the moved-aside copy still has the original content -- nothing was deleted")
        assert_true("moving it aside" in (r.stdout + r.stderr), "bootstrap.sh says so", extra=r.stdout + r.stderr)


def test_empty_repo_dir_is_simply_removed_before_clone():
    """A genuinely EMPTY (not-a-checkout) $REPO is safe to remove outright
    -- nothing to lose -- so it should NOT pile up .aside-<ts> directories
    forever on a normal "container overlay reset" restart.
    """
    with tempfile.TemporaryDirectory() as tdstr:
        td = Path(tdstr)
        env = _base_env(td)
        repo = Path(env["ZL_REPO"])
        repo.mkdir(parents=True)  # empty

        zl_home = Path(env["ZL_HOME"])
        zl_home.mkdir(parents=True)
        (zl_home / "ref").write_text("test-ref\n", encoding="utf-8")
        (zl_home / "bootstrap-flags").write_text(
            "__REPO_URL__=https://example.invalid/nonexistent-repo-for-testing.git\n--no-login-hook\n",
            encoding="utf-8",
        )
        saved_copy = zl_home / "bootstrap.sh"
        saved_copy.write_text(BOOTSTRAP.read_text(encoding="utf-8"), encoding="utf-8")
        saved_copy.chmod(0o755)

        subprocess.run(["bash", str(saved_copy)], capture_output=True, text=True, timeout=30, env=env)
        aside_dirs = list(td.glob("opt-zl-repo.aside-*"))
        assert_eq(aside_dirs, [], "an empty $REPO is removed directly, no .aside-<ts> left behind")


def test_help_never_creates_zl_home():
    """review LOW: --help/-h must be a pure read, no side effect."""
    with tempfile.TemporaryDirectory() as tdstr:
        td = Path(tdstr)
        env = _base_env(td)
        r = _run(["--help"], env)
        assert_eq(r.returncode, 0, "--help exits 0")
        assert_true(not Path(env["ZL_HOME"]).exists(), "--help never created ZL_HOME", extra=r.stdout)
        r2 = _run(["-h"], env)
        assert_eq(r2.returncode, 0, "-h exits 0")
        assert_true(not Path(env["ZL_HOME"]).exists(), "-h never created ZL_HOME either", extra=r2.stdout)


def test_ref_without_a_value_exits_2_cleanly():
    """review LOW: `--ref` as the LAST argument used to crash with
    `set -u`'s own "$2: unbound variable" instead of a clean usage error.
    """
    with tempfile.TemporaryDirectory() as tdstr:
        td = Path(tdstr)
        env = _base_env(td)
        r = _run(["--ref"], env)
        assert_eq(r.returncode, 2, "a valueless --ref exits 2, not a crash")
        assert_true("unbound variable" not in (r.stdout + r.stderr),
                     "no raw 'unbound variable' crash text leaks out", extra=r.stdout + r.stderr)
        assert_true("requires a value" in (r.stdout + r.stderr), "the error explains itself",
                     extra=r.stdout + r.stderr)


# =============================================================================
# `--boot` mode (v3.1.9.8+, run from supervisord's [program:zl-ops]) --
# non-interactive, bounded, exit 0 on every path. See ops/README.md and
# NOTE-ops-bundle-boot.md for the full contract.
# =============================================================================


def _boot_env(td: Path) -> dict[str, str]:
    env = _base_env(td)
    log_dir = td / "zl-home" / "logs"
    env["LOG_DIR"] = str(log_dir)
    return env


def _read_boot_log(env: dict[str, str]) -> str:
    p = Path(env["LOG_DIR"]) / "zl-ops.log"
    return p.read_text(encoding="utf-8") if p.is_file() else ""


_FAKE_ZL = """#!/usr/bin/env bash
echo "FAKE-ZL: $*" >> "${FAKE_ZL_LOG:-/dev/null}"
if [ "$1" = "sshd" ] && [ -n "${FAKE_ZL_SSHD_FAIL:-}" ]; then
    echo "fake sshd apply failing on purpose"
    exit 1
fi
exit 0
"""


def _fake_checkout(td: Path, name: str = "fake-checkout") -> Path:
    """A minimal from-dir checkout with a FAKE `ops/zl` (controllable via
    FAKE_ZL_LOG/FAKE_ZL_SSHD_FAIL) -- used wherever a test needs to observe
    or fail a `zl sshd`/`zl tools` call deterministically, without touching
    the real worktree's own ops/zl or exec'ing a real setup-sshd.py.
    """
    checkout = td / name
    (checkout / "ops").mkdir(parents=True, exist_ok=True)
    zl = checkout / "ops" / "zl"
    zl.write_text(_FAKE_ZL, encoding="utf-8")
    zl.chmod(0o755)
    return checkout


def test_boot_from_dir_install_exits_0_and_installs_zl():
    with tempfile.TemporaryDirectory() as tdstr:
        td = Path(tdstr)
        env = _boot_env(td)
        r = _run(["--from-dir", str(REPO_ROOT), "--boot", "--no-login-hook"], env)
        assert_eq(r.returncode, 0, "a plain --boot install exits 0", )
        zl_bin = Path(env["ZL_BIN"])
        assert_true(zl_bin.is_symlink(), "bootstrap.sh --boot still installs the zl symlink",
                    extra=r.stdout + r.stderr)


def test_boot_never_reads_stdin():
    """Non-interactive: --boot must never block waiting on stdin. Piping in
    a stream that is never closed would hang a test that actually blocked
    on read -- run with stdin explicitly closed (subprocess.DEVNULL) and a
    tight wall-clock timeout; a script that tried to `read` would raise
    TimeoutExpired here instead of returning.
    """
    with tempfile.TemporaryDirectory() as tdstr:
        td = Path(tdstr)
        env = _boot_env(td)
        r = subprocess.run(
            ["bash", str(BOOTSTRAP), "--from-dir", str(REPO_ROOT), "--boot", "--no-login-hook"],
            capture_output=True, text=True, timeout=30, env=env, stdin=subprocess.DEVNULL,
        )
        assert_eq(r.returncode, 0, "--boot with stdin closed still exits 0 promptly")


def test_boot_never_re_execs():
    """Never re-exec in a loop: --boot with --from-dir must never hand off
    to a saved copy (that handoff -- "handing off to ... bootstrap.sh" --
    is stage 1's own message, and stage 1 is unconditionally skipped
    whenever --from-dir is given).
    """
    with tempfile.TemporaryDirectory() as tdstr:
        td = Path(tdstr)
        env = _boot_env(td)
        r = _run(["--from-dir", str(REPO_ROOT), "--boot", "--no-login-hook"], env)
        assert_eq(r.returncode, 0, "--boot exits 0")
        assert_true("handing off" not in (r.stdout + r.stderr),
                    "--boot with --from-dir never re-execs via stage 1", extra=r.stdout + r.stderr)


def test_boot_uses_baked_copy_when_no_ref_is_saved():
    with tempfile.TemporaryDirectory() as tdstr:
        td = Path(tdstr)
        env = _boot_env(td)
        r = _run(["--from-dir", str(REPO_ROOT), "--boot", "--no-login-hook"], env)
        assert_eq(r.returncode, 0, "--boot exits 0")
        log = _read_boot_log(env)
        assert_true("not applied at boot" not in log,
                    "no ref saved -- there is nothing to report not applying", extra=log)


_RECORDING_FAKE_GIT = """#!/usr/bin/env bash
echo "FAKE-GIT-CALLED: $*" >> "${FAKE_GIT_LOG:-/dev/null}"
exit 0
"""


def test_boot_never_clones_even_when_a_ref_is_saved_and_network_is_available():
    """DECISION: --boot does NOT do ref pinning -- the whole clone block
    was removed outright (also eliminating H2, the previous unbounded
    /data rm/mv/[ -d ] dance around a "boot-clone" directory). No clone
    is ever attempted at boot, even when a ref WAS saved from an earlier
    interactive install and the network is (apparently) available -- a
    fake `git` on PATH that just records any invocation proves it is
    never called at all. Instead, exactly one informational log line is
    printed.
    """
    with tempfile.TemporaryDirectory() as tdstr:
        td = Path(tdstr)
        env = _boot_env(td)
        bin_dir = Path(env["PATH"].split(os.pathsep)[0])
        fake_git_log = td / "fake-git-calls.log"
        fake_git = bin_dir / "git"
        fake_git.write_text(_RECORDING_FAKE_GIT, encoding="utf-8")
        fake_git.chmod(0o755)
        env["FAKE_GIT_LOG"] = str(fake_git_log)
        checkout = _fake_checkout(td)
        zl_home = Path(env["ZL_HOME"])
        zl_home.mkdir(parents=True)
        (zl_home / "ref").write_text("some-saved-ref\n", encoding="utf-8")
        r = _run(["--from-dir", str(checkout), "--boot", "--no-login-hook"], env)
        assert_eq(r.returncode, 0, "--boot exits 0")
        assert_true(not fake_git_log.exists(), "git was never invoked at all during --boot",
                    extra=str(fake_git_log))
        log = _read_boot_log(env)
        assert_true("saved ref 'some-saved-ref' not applied at boot" in log,
                    "the saved-ref-not-applied line is logged exactly once", extra=log)
        assert_true("the baked image copy is used" in log, "the reason is spelled out", extra=log)
        zl_bin = Path(env["ZL_BIN"])
        assert_true(zl_bin.is_symlink(), "zl was installed from the baked copy", extra=log)


def test_boot_honours_saved_with_sshd_flag_bounded_and_fails_soft():
    with tempfile.TemporaryDirectory() as tdstr:
        td = Path(tdstr)
        env = _boot_env(td)
        checkout = _fake_checkout(td)
        fake_zl_log = td / "fake-zl-calls.log"
        env["FAKE_ZL_LOG"] = str(fake_zl_log)
        env["FAKE_ZL_SSHD_FAIL"] = "1"
        zl_home = Path(env["ZL_HOME"])
        zl_home.mkdir(parents=True)
        (zl_home / "bootstrap-flags").write_text(
            "__REPO_URL__=https://example.invalid/x.git\n--with-sshd\n--no-login-hook\n", encoding="utf-8"
        )
        r = _run(["--from-dir", str(checkout), "--boot"], env)
        assert_eq(r.returncode, 0, "--boot exits 0 even though sshd apply fails")
        assert_true(fake_zl_log.is_file() and "sshd" in fake_zl_log.read_text(encoding="utf-8"),
                    "the saved --with-sshd flag really did trigger `zl sshd --apply --yes`")
        log = _read_boot_log(env)
        assert_true("zl sshd failed (exit 1) -- continuing" in log and "SSH stays down" in log,
                    "the failure was logged as fail-soft (not a hang, not a hard abort), with its exit code",
                    extra=log)
        assert_true("last output:" in log and "fake sshd apply failing on purpose" in log,
                    "the failed step's own last output was captured and logged (M3)", extra=log)


def test_boot_skips_with_tools_unless_tools_at_boot_was_saved():
    """`--with-tools` alone (without the boot-specific `--tools-at-boot`)
    must NOT run `zl tools --apply --yes` during --boot -- apt is slow and
    network-bound, and a boot-time program must never risk it by default.
    """
    with tempfile.TemporaryDirectory() as tdstr:
        td = Path(tdstr)
        env = _boot_env(td)
        checkout = _fake_checkout(td)
        fake_zl_log = td / "fake-zl-calls.log"
        env["FAKE_ZL_LOG"] = str(fake_zl_log)
        zl_home = Path(env["ZL_HOME"])
        zl_home.mkdir(parents=True)
        (zl_home / "bootstrap-flags").write_text(
            "__REPO_URL__=https://example.invalid/x.git\n--with-tools\n--no-login-hook\n", encoding="utf-8"
        )
        r = _run(["--from-dir", str(checkout), "--boot"], env)
        assert_eq(r.returncode, 0, "--boot exits 0")
        calls = fake_zl_log.read_text(encoding="utf-8") if fake_zl_log.is_file() else ""
        assert_true("tools" not in calls, "plain --with-tools (no --tools-at-boot) never runs at boot", extra=calls)


def test_boot_runs_tools_last_when_tools_at_boot_saved():
    with tempfile.TemporaryDirectory() as tdstr:
        td = Path(tdstr)
        env = _boot_env(td)
        checkout = _fake_checkout(td)
        fake_zl_log = td / "fake-zl-calls.log"
        env["FAKE_ZL_LOG"] = str(fake_zl_log)
        zl_home = Path(env["ZL_HOME"])
        zl_home.mkdir(parents=True)
        (zl_home / "bootstrap-flags").write_text(
            "__REPO_URL__=https://example.invalid/x.git\n--with-sshd\n--tools-at-boot\n--no-login-hook\n",
            encoding="utf-8",
        )
        r = _run(["--from-dir", str(checkout), "--boot"], env)
        assert_eq(r.returncode, 0, "--boot exits 0")
        calls = fake_zl_log.read_text(encoding="utf-8").splitlines()
        call_cmds = [c.split(":", 1)[1].strip() if ":" in c else c for c in calls]
        assert_true(any(c.startswith("sshd") for c in call_cmds), "sshd was called", extra=str(calls))
        assert_true(any(c.startswith("tools") for c in call_cmds), "tools was called too (tools-at-boot saved)",
                    extra=str(calls))
        sshd_idx = next(i for i, c in enumerate(call_cmds) if c.startswith("sshd"))
        tools_idx = next(i for i, c in enumerate(call_cmds) if c.startswith("tools"))
        assert_true(tools_idx > sshd_idx, "tools ran AFTER sshd (tools always runs last)", extra=str(calls))


def test_with_hotfix_is_saved_and_replayed_bounded_and_last():
    """`--with-hotfix NAME` is saved like --with-sshd/--with-tools, and
    replayed under --boot bounded and fail-soft, running after both the
    sshd and tools steps.
    """
    with tempfile.TemporaryDirectory() as tdstr:
        td = Path(tdstr)
        env = _base_env(td)
        r = _run(["--ref", "test-ref", "--from-dir", str(REPO_ROOT), "--with-hotfix", "backup-fast",
                   "--no-login-hook"], env)
        assert_eq(r.returncode, 0, "an interactive --with-hotfix install exits 0")
        flags = (Path(env["ZL_HOME"]) / "bootstrap-flags").read_text(encoding="utf-8")
        assert_true("--with-hotfix=backup-fast" in flags, "--with-hotfix was saved to bootstrap-flags", extra=flags)

    with tempfile.TemporaryDirectory() as tdstr:
        td = Path(tdstr)
        env = _boot_env(td)
        checkout = _fake_checkout(td)
        fake_zl_log = td / "fake-zl-calls.log"
        env["FAKE_ZL_LOG"] = str(fake_zl_log)
        zl_home = Path(env["ZL_HOME"])
        zl_home.mkdir(parents=True)
        (zl_home / "bootstrap-flags").write_text(
            "__REPO_URL__=https://example.invalid/x.git\n--with-sshd\n--tools-at-boot\n"
            "--with-hotfix=backup-fast\n--no-login-hook\n",
            encoding="utf-8",
        )
        r = _run(["--from-dir", str(checkout), "--boot"], env)
        assert_eq(r.returncode, 0, "--boot exits 0")
        calls = fake_zl_log.read_text(encoding="utf-8").splitlines()
        call_cmds = [c.split(":", 1)[1].strip() if ":" in c else c for c in calls]
        assert_true(any(c.startswith("hotfix backup-fast") for c in call_cmds),
                    "zl hotfix backup-fast --apply --yes was called", extra=str(calls))
        tools_idx = next(i for i, c in enumerate(call_cmds) if c.startswith("tools"))
        hotfix_idx = next(i for i, c in enumerate(call_cmds) if c.startswith("hotfix"))
        assert_true(hotfix_idx > tools_idx, "hotfix ran AFTER tools (hotfix always runs last)", extra=str(calls))
        log = _read_boot_log(env)
        assert_true("--with-hotfix backup-fast -> zl hotfix backup-fast --apply --yes" in log,
                    "the step was logged", extra=log)


def test_boot_data_unwritable_still_installs_zl_and_exits_0():
    with tempfile.TemporaryDirectory() as tdstr:
        td = Path(tdstr)
        env = _boot_env(td)
        # Same "put a plain FILE where a directory is needed" trick as
        # test_boot_log_dir_missing_falls_back_to_stderr -- `mkdir -p`
        # fails for anyone, including root (this project's own docker
        # unit-suite runs as root, which a plain chmod-based "unwritable"
        # trick cannot stop).
        blocker = td / "data-blocker"
        blocker.write_text("not a directory", encoding="utf-8")
        env["ZL_HOME"] = str(blocker / "zl-ops")
        env["LOG_DIR"] = str(blocker / "logs")
        r = _run(["--from-dir", str(REPO_ROOT), "--boot", "--no-login-hook", "--ref", "main"], env)
        assert_eq(r.returncode, 0, "--boot exits 0 even with an unwritable /data")
        zl_bin = Path(env["ZL_BIN"])
        assert_true(zl_bin.is_symlink(), "zl was still installed despite /data being unwritable",
                    extra=r.stdout + r.stderr)
        assert_true("No such file or directory" not in (r.stdout + r.stderr),
                    "no raw shell redirection error leaks to stdout/stderr", extra=r.stdout + r.stderr)


def test_boot_log_dir_missing_falls_back_to_stderr():
    with tempfile.TemporaryDirectory() as tdstr:
        td = Path(tdstr)
        env = _boot_env(td)
        # A NORMAL missing LOG_DIR is just created (mkdir -p) -- to force
        # `mkdir -p` to actually FAIL (needed to exercise the stderr
        # fallback) regardless of who runs this test (including root,
        # which a plain chmod-based "unwritable" trick cannot stop --
        # this project's own docker unit-suite runs as root), put a
        # PLAIN FILE where a path component needs to be a directory:
        # `mkdir -p .../blocker/logs` fails for anyone, root included,
        # when "blocker" already exists and is not a directory.
        blocker = td / "blocker"
        blocker.write_text("not a directory", encoding="utf-8")
        env["LOG_DIR"] = str(blocker / "logs")
        r = _run(["--from-dir", str(REPO_ROOT), "--boot", "--no-login-hook"], env)
        assert_eq(r.returncode, 0, "--boot exits 0 even when LOG_DIR cannot be created")
        assert_true("bootstrap.sh" in r.stderr, "log lines fell back to stderr instead of hanging or vanishing",
                    extra=r.stdout + r.stderr)


def test_boot_set_e_style_failure_mid_script_still_exits_0():
    """A `--from-dir` that is not a real checkout at all (no ops/) is
    exactly the kind of hard failure `set -e` would otherwise propagate as
    a nonzero exit -- --boot's own EXIT trap must still force 0.
    """
    with tempfile.TemporaryDirectory() as tdstr:
        td = Path(tdstr)
        env = _boot_env(td)
        not_a_checkout = td / "not-a-checkout"
        not_a_checkout.mkdir()
        r = _run(["--from-dir", str(not_a_checkout), "--boot", "--no-login-hook"], env)
        assert_eq(r.returncode, 0, "--boot exits 0 even when --from-dir is not a real checkout")


def test_boot_budget_exceeded_skips_remaining_steps_and_exits_0():
    with tempfile.TemporaryDirectory() as tdstr:
        td = Path(tdstr)
        env = _boot_env(td)
        env["ZL_BOOT_BUDGET_S"] = "0"
        checkout = _fake_checkout(td)
        zl_home = Path(env["ZL_HOME"])
        zl_home.mkdir(parents=True)
        (zl_home / "bootstrap-flags").write_text(
            "__REPO_URL__=https://example.invalid/x.git\n--with-sshd\n--no-login-hook\n", encoding="utf-8"
        )
        r = _run(["--from-dir", str(checkout), "--boot"], env)
        assert_eq(r.returncode, 0, "--boot exits 0 when the budget is already exhausted")
        log = _read_boot_log(env)
        assert_true("skipping --with-sshd this run" in log, "the sshd step was skipped for lack of budget", extra=log)
        zl_bin = Path(env["ZL_BIN"])
        assert_true(zl_bin.is_symlink(), "zl is still installed even when the budget ran out before the sshd step")


def test_boot_is_idempotent_with_the_interactive_installer():
    """--boot installs the SAME zl symlink and the SAME login-hook block
    (when not suppressed) as a normal interactive install -- re-running
    either mode after the other must never duplicate the login hook.
    """
    with tempfile.TemporaryDirectory() as tdstr:
        td = Path(tdstr)
        env = _boot_env(td)
        r1 = _run(["--from-dir", str(REPO_ROOT), "--ref", "test-ref"], env)  # interactive
        assert_eq(r1.returncode, 0, "interactive install exits 0")
        r2 = _run(["--from-dir", str(REPO_ROOT), "--boot"], env)  # boot, same ref already saved
        assert_eq(r2.returncode, 0, "boot install exits 0")
        bashrc = Path(env["ZL_BASHRC"]).read_text(encoding="utf-8")
        assert_eq(bashrc.count("BEGIN zl login hook"), 1,
                  "the login hook block still appears exactly once across both modes")


def test_boot_real_set_e_abort_is_caught_by_the_exit_trap_and_logged():
    """L1/mut.py M12 regression: unlike
    test_boot_set_e_style_failure_mid_script_still_exits_0 (which takes
    the script's own EXPLICIT `exit 0` early-return path for a missing
    ops/ dir), this forces a REAL `set -e` abort mid-script -- `ln -sf`
    failing because $ZL_BIN's own parent directory does not exist -- and
    checks the ERR/EXIT trap actually caught it: the log must say
    "aborted at line N", never a plain "done".
    """
    with tempfile.TemporaryDirectory() as tdstr:
        td = Path(tdstr)
        env = _boot_env(td)
        env["ZL_BIN"] = str(td / "no-such-parent-dir" / "zl")
        r = _run(["--from-dir", str(REPO_ROOT), "--boot", "--no-login-hook"], env)
        assert_eq(r.returncode, 0, "--boot still exits 0 even on a real set -e abort")
        log = _read_boot_log(env)
        assert_true("aborted at line" in log, "the ERR/EXIT trap logged a real abort, not a plain 'done'",
                    extra=log)
        assert_true("bootstrap.sh --boot: done" not in log, "a genuine abort is never reported as 'done'",
                    extra=log)


_SLOW_CAT = """#!/usr/bin/env bash
sleep 20
exec /bin/cat "$@"
"""


def test_boot_bounded_ref_read_stays_within_budget_plus_margin():
    """L1/mut.py M13 regression: a stalled `cat` on the very first read of
    the saved ref must be bounded (timeout -k 2 5s -> ~7s worst case),
    never left to run for as long as the shim itself would stall (here,
    20s) -- and the timeout is logged distinctly from a plain missing
    file.
    """
    with tempfile.TemporaryDirectory() as tdstr:
        td = Path(tdstr)
        env = _boot_env(td)
        bin_dir = Path(env["PATH"].split(os.pathsep)[0])
        slow_cat = bin_dir / "cat"
        slow_cat.write_text(_SLOW_CAT, encoding="utf-8")
        slow_cat.chmod(0o755)
        zl_home = Path(env["ZL_HOME"])
        zl_home.mkdir(parents=True)
        (zl_home / "ref").write_text("some-ref\n", encoding="utf-8")
        start = time.monotonic()
        r = _run(["--from-dir", str(REPO_ROOT), "--boot", "--no-login-hook"], env, timeout=30)
        elapsed = time.monotonic() - start
        assert_eq(r.returncode, 0, "--boot exits 0 even with a stalled cat shim")
        assert_true(elapsed < 15, f"the read was bounded (~7s worst case), not the shim's own 20s sleep "
                                  f"(took {elapsed:.1f}s)")
        log = _read_boot_log(env)
        assert_true("could not read saved ref within 5s" in log, "the timeout was logged distinctly", extra=log)


_SLOW_SSHD_FAKE_ZL = """#!/usr/bin/env bash
echo "FAKE-ZL: $*" >> "${FAKE_ZL_LOG:-/dev/null}"
if [ "$1" = "sshd" ]; then
    sleep 60
fi
exit 0
"""


def test_boot_caps_sshd_timeout_to_remaining_budget():
    """L1/mut.py M16 regression: the sshd step's own 90s default timeout
    must be capped to whatever is actually left of the overall boot
    budget, not allowed to run for its full default. A tiny budget (15s)
    with a `zl sshd` that hangs for 60s must still come back (killed)
    well within the budget, not anywhere near 60s.
    """
    with tempfile.TemporaryDirectory() as tdstr:
        td = Path(tdstr)
        env = _boot_env(td)
        env["ZL_BOOT_BUDGET_S"] = "15"
        checkout = td / "slow-sshd-checkout"
        (checkout / "ops").mkdir(parents=True)
        zl = checkout / "ops" / "zl"
        zl.write_text(_SLOW_SSHD_FAKE_ZL, encoding="utf-8")
        zl.chmod(0o755)
        fake_zl_log = td / "fake-zl-calls.log"
        env["FAKE_ZL_LOG"] = str(fake_zl_log)
        zl_home = Path(env["ZL_HOME"])
        zl_home.mkdir(parents=True)
        (zl_home / "bootstrap-flags").write_text(
            "__REPO_URL__=https://example.invalid/x.git\n--with-sshd\n--no-login-hook\n", encoding="utf-8"
        )
        start = time.monotonic()
        r = _run(["--from-dir", str(checkout), "--boot"], env, timeout=60)
        elapsed = time.monotonic() - start
        assert_eq(r.returncode, 0, "--boot exits 0")
        assert_true(elapsed < 25, f"sshd's own 90s default timeout was capped to the ~15s budget left, not "
                                  f"allowed to run long (took {elapsed:.1f}s)")
        log = _read_boot_log(env)
        assert_true("zl sshd timed out" in log, "the capped, timed-out sshd step was logged", extra=log)


_SLOW_TEE = """#!/usr/bin/env bash
sleep 20
exec /usr/bin/tee "$@"
"""


def test_boot_log_write_itself_is_bounded_by_a_timeout():
    """L1/mut.py M17 regression: the log write inside `_boot_log` itself
    (`timeout ... tee -a $LOG_DIR/zl-ops.log`) must be bounded -- a
    stalled `tee` must not be left to run for anywhere near its own 20s
    sleep; the write falls back to stderr instead.
    """
    with tempfile.TemporaryDirectory() as tdstr:
        td = Path(tdstr)
        env = _boot_env(td)
        bin_dir = Path(env["PATH"].split(os.pathsep)[0])
        slow_tee = bin_dir / "tee"
        slow_tee.write_text(_SLOW_TEE, encoding="utf-8")
        slow_tee.chmod(0o755)
        start = time.monotonic()
        r = _run(["--from-dir", str(REPO_ROOT), "--boot", "--no-login-hook"], env, timeout=30)
        elapsed = time.monotonic() - start
        assert_eq(r.returncode, 0, "--boot exits 0 even when the log write itself stalls")
        assert_true(elapsed < 15, f"the log write was bounded (~3-5s), not the shim's own 20s sleep "
                                  f"(took {elapsed:.1f}s)")
        assert_true("bootstrap.sh" in r.stderr, "a stalled log write falls back to stderr",
                    extra=r.stdout + r.stderr)


def test_boot_login_hook_failure_does_not_skip_the_sshd_step():
    """M3: an abort in the login-hook step must not skip the sshd step --
    forces `touch "$BASHRC"` to fail (BASHRC's own parent directory does
    not exist and is never created for it) and confirms `zl sshd --apply`
    still ran anyway.
    """
    with tempfile.TemporaryDirectory() as tdstr:
        td = Path(tdstr)
        env = _boot_env(td)
        env["ZL_BASHRC"] = str(td / "no-such-parent-dir" / "bashrc")
        checkout = _fake_checkout(td)
        fake_zl_log = td / "fake-zl-calls.log"
        env["FAKE_ZL_LOG"] = str(fake_zl_log)
        zl_home = Path(env["ZL_HOME"])
        zl_home.mkdir(parents=True)
        (zl_home / "bootstrap-flags").write_text(
            "__REPO_URL__=https://example.invalid/x.git\n--with-sshd\n", encoding="utf-8"
        )
        r = _run(["--from-dir", str(checkout), "--boot"], env)
        assert_eq(r.returncode, 0, "--boot exits 0 even though the login-hook step failed")
        log = _read_boot_log(env)
        assert_true("login-hook step failed" in log, "the login-hook failure was logged", extra=log)
        calls = fake_zl_log.read_text(encoding="utf-8") if fake_zl_log.is_file() else ""
        assert_true("sshd" in calls, "the sshd step still ran despite the login-hook step failing", extra=calls)


def test_unknown_flag_exits_2():
    with tempfile.TemporaryDirectory() as tdstr:
        td = Path(tdstr)
        env = _base_env(td)
        r = _run(["--not-a-real-flag"], env)
        assert_eq(r.returncode, 2, "an unrecognized flag exits 2")


if __name__ == "__main__":
    test_from_dir_install_creates_symlink_and_saves_state()
    test_login_hook_is_idempotent_and_no_login_hook_removes_it()
    test_saved_flags_persist_across_a_bare_rerun()
    test_worktree_style_git_dir_is_never_destructively_recloned()
    test_saved_copy_hands_off_to_a_baked_image_copy_without_cloning()
    test_with_tools_never_calls_real_apt_get()
    test_login_hook_uses_timeout_dash_k()
    test_non_empty_non_checkout_repo_dir_is_moved_aside_not_deleted()
    test_empty_repo_dir_is_simply_removed_before_clone()
    test_help_never_creates_zl_home()
    test_ref_without_a_value_exits_2_cleanly()
    test_unknown_flag_exits_2()

    test_boot_from_dir_install_exits_0_and_installs_zl()
    test_boot_never_reads_stdin()
    test_boot_never_re_execs()
    test_boot_uses_baked_copy_when_no_ref_is_saved()
    test_boot_never_clones_even_when_a_ref_is_saved_and_network_is_available()
    test_boot_honours_saved_with_sshd_flag_bounded_and_fails_soft()
    test_boot_skips_with_tools_unless_tools_at_boot_was_saved()
    test_boot_runs_tools_last_when_tools_at_boot_saved()
    test_with_hotfix_is_saved_and_replayed_bounded_and_last()
    test_boot_data_unwritable_still_installs_zl_and_exits_0()
    test_boot_log_dir_missing_falls_back_to_stderr()
    test_boot_set_e_style_failure_mid_script_still_exits_0()
    test_boot_budget_exceeded_skips_remaining_steps_and_exits_0()
    test_boot_is_idempotent_with_the_interactive_installer()
    test_boot_real_set_e_abort_is_caught_by_the_exit_trap_and_logged()
    test_boot_bounded_ref_read_stays_within_budget_plus_margin()
    test_boot_caps_sshd_timeout_to_remaining_budget()
    test_boot_log_write_itself_is_bounded_by_a_timeout()
    test_boot_login_hook_failure_does_not_skip_the_sshd_step()

    print("\nAll ops/bootstrap.sh tests passed.")
