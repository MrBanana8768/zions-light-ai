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
    test_with_tools_never_calls_real_apt_get()
    test_login_hook_uses_timeout_dash_k()
    test_non_empty_non_checkout_repo_dir_is_moved_aside_not_deleted()
    test_empty_repo_dir_is_simply_removed_before_clone()
    test_help_never_creates_zl_home()
    test_ref_without_a_value_exits_2_cleanly()
    test_unknown_flag_exits_2()

    print("\nAll ops/bootstrap.sh tests passed.")
