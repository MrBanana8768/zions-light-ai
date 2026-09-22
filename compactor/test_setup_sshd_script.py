"""
Tests for scripts/setup-sshd.py — the operator tool that installs and
hardens OpenSSH server INSIDE the running production container (the image
itself ships no openssh-server and must not change — see that script's own
module docstring).

Runs with NO network and NOT necessarily as root: every external binary
this script shells out to (apt-get, apt-cache, ssh-keygen, sshd,
supervisorctl) is replaced with a small fake on PATH, every filesystem root
it touches (/etc/ssh, /root/.ssh, /data/ssh, the supervisor conf, /run) is
redirected to a temp tree via the same env-var injection points
switch-webui-db-to-local.py already uses for its own paths, and the
root-euid check is patched exactly the way scripts/backfill-records.py's
own tests patch `_utc_stamp` for determinism — the SCRIPT's real check
(`os.geteuid() == 0`) is untouched; only the test's view of it is.

The fake `sshd` binary is not a no-op: `-T` really resolves `Include`
(first-match-wins, exactly like the real daemon) against whatever this
script actually wrote to disk, so a test that reads "already-current" or
"passwordauthentication no" back from it is reading the consequence of the
script's own file-writing logic, not a canned string. The one exception is
`FAKE_SSHD_FORCE_PASSWORD_AUTH`, used for exactly one test: proving the
script's post-write verification refuses on ANY bad effective outcome, not
only ones its own config-writing code could itself produce.

Run inside the compactor image or any container with the requirements
installed:
    python test_setup_sshd_script.py
"""

import ctypes
import io
import json
import os
import shutil
import stat
import sys
import tempfile
import time
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

_HERE = Path(__file__).resolve().parent
_SCRIPT_PATH = _HERE.parent / "scripts" / "setup-sshd.py"

import importlib.util

_spec = importlib.util.spec_from_file_location("setup_sshd_script", _SCRIPT_PATH)
_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_script)


def assert_eq(actual, expected, label):
    if actual != expected:
        print(f"FAIL {label}: expected {expected!r}, got {actual!r}")
        sys.exit(1)
    print(f"  ok   {label}")


def assert_true(cond, label):
    if not cond:
        print(f"FAIL {label}")
        sys.exit(1)
    print(f"  ok   {label}")


def assert_in(needle, haystack, label):
    if needle not in haystack:
        print(f"FAIL {label}: {needle!r} not found")
        sys.exit(1)
    print(f"  ok   {label}")


# ---------------------------------------------------------------------------
# Fake binaries — written once to a scratch bin dir, prepended to PATH.
# ---------------------------------------------------------------------------

_FAKE_APT_GET = r"""#!/usr/bin/env python3
import os, sys

def log(args):
    p = os.environ.get("FAKE_CMD_LOG")
    if p:
        with open(p, "a") as f:
            f.write("apt-get " + " ".join(args) + "\n")

def main():
    args = sys.argv[1:]
    log(args)
    if not args:
        return 0
    if args[0] == "update":
        if os.environ.get("FAKE_APT_UPDATE_FAIL") == "1":
            sys.stderr.write("W: fake apt-get update failure (simulated CUDA channel)\n")
            return 1
        print("Fetched fake package lists")
        return 0
    if "install" in args:
        simulate = "-s" in args
        if simulate:
            print("Inst openssh-server (simulated)")
            return 0
        if os.environ.get("FAKE_APT_INSTALL_FAIL") == "1":
            sys.stderr.write("E: fake install failure\n")
            return 1
        state_file = os.environ.get("FAKE_APT_STATE_FILE")
        candidate = os.environ.get("FAKE_APT_CANDIDATE", "1:9.9p1-1ubuntu1")
        if state_file:
            with open(state_file, "w") as f:
                f.write(candidate)
        return 0
    # never expect a bare upgrade/dist-upgrade — logged above either way,
    # never crashed on, so the test can assert against the log.
    return 0

if __name__ == "__main__":
    sys.exit(main())
"""

_FAKE_APT_CACHE = r"""#!/usr/bin/env python3
import os, sys

def log(args):
    p = os.environ.get("FAKE_CMD_LOG")
    if p:
        with open(p, "a") as f:
            f.write("apt-cache " + " ".join(args) + "\n")

def main():
    args = sys.argv[1:]
    log(args)
    if len(args) >= 2 and args[0] == "policy":
        pkg = args[1]
        state_file = os.environ.get("FAKE_APT_STATE_FILE")
        installed = "(none)"
        if state_file and os.path.isfile(state_file):
            v = open(state_file).read().strip()
            if v:
                installed = v
        candidate = os.environ.get("FAKE_APT_CANDIDATE", "1:9.9p1-1ubuntu1")
        print(f"{pkg}:")
        print(f"  Installed: {installed}")
        print(f"  Candidate: {candidate}")
        print("  Version table:")
        print(f"     {candidate} 500")
        return 0
    return 0

if __name__ == "__main__":
    sys.exit(main())
"""

_FAKE_SSH_KEYGEN = r"""#!/usr/bin/env python3
import os, re, sys

KEY_RE = re.compile(r"^(ssh-(rsa|ed25519|dss)|ecdsa-sha2-\S+)\s+[A-Za-z0-9+/]+=*(\s+.*)?$")

def log(args):
    p = os.environ.get("FAKE_CMD_LOG")
    if p:
        with open(p, "a") as f:
            f.write("ssh-keygen " + " ".join(args) + "\n")

def main():
    args = sys.argv[1:]
    log(args)
    if "-l" in args and "-f" in args:
        path = args[args.index("-f") + 1]
        try:
            content = open(path, encoding="utf-8", errors="replace").read().strip()
        except OSError as e:
            sys.stderr.write(f"fake ssh-keygen: {e}\n")
            return 1
        if os.environ.get("FAKE_SSH_KEYGEN_FORCE_FAIL") == "1":
            sys.stderr.write("fake ssh-keygen: forced failure\n")
            return 1
        if not content or not KEY_RE.match(content):
            sys.stderr.write(f"{path} is not a public key file.\n")
            return 1
        print("2048 SHA256:fakefingerprintvalue comment (RSA)")
        return 0
    if "-A" in args:
        prefix = ""
        if "-f" in args:
            prefix = args[args.index("-f") + 1]
        etc_ssh = (prefix.rstrip("/") + "/etc/ssh") if prefix else "/etc/ssh"
        os.makedirs(etc_ssh, exist_ok=True)
        for t in ("rsa", "ecdsa", "ed25519"):
            priv = os.path.join(etc_ssh, f"ssh_host_{t}_key")
            pub = priv + ".pub"
            if not os.path.exists(priv):
                with open(priv, "w") as f:
                    f.write(f"FAKE-PRIVATE-HOST-KEY-{t}-DO-NOT-USE\n")
                os.chmod(priv, 0o600)
            if not os.path.exists(pub):
                kind = "rsa" if t == "rsa" else t
                with open(pub, "w") as f:
                    f.write(f"ssh-{kind} AAAAFAKEHOSTKEY{t} root@fake\n")
        print("fake ssh-keygen -A: host keys ensured")
        return 0
    return 0

if __name__ == "__main__":
    sys.exit(main())
"""

_FAKE_SSHD = r"""#!/usr/bin/env python3
import ctypes, glob, os, signal, sys, time

def log(args):
    p = os.environ.get("FAKE_CMD_LOG")
    if p:
        with open(p, "a") as f:
            f.write("sshd " + " ".join(args) + "\n")

_DEFAULTS = {
    "passwordauthentication": "yes",
    "permitemptypasswords": "no",
    "kbdinteractiveauthentication": "yes",
    "pubkeyauthentication": "yes",
    "permitrootlogin": "prohibit-password",
    "port": "22",
}

def _resolve(path, effective, seen):
    if path in seen:
        return
    seen.add(path)
    try:
        text = open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        return
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split(None, 1)
        key = parts[0]
        val = parts[1].strip() if len(parts) > 1 else ""
        keyl = key.lower()
        if keyl == "include":
            for gp in sorted(glob.glob(val)):
                _resolve(gp, effective, seen)
            continue
        if keyl not in effective:
            effective[keyl] = val

def _effective_config():
    etc_ssh = os.environ.get("SETUP_SSHD_ETC_SSH_DIR", "/etc/ssh")
    cfg = os.path.join(etc_ssh, "sshd_config")
    effective = {}
    _resolve(cfg, effective, set())
    for k, v in _DEFAULTS.items():
        effective.setdefault(k, v)
    if os.environ.get("FAKE_SSHD_FORCE_PASSWORD_AUTH") == "1":
        effective["passwordauthentication"] = "yes"
    return effective

def main():
    args = sys.argv[1:]
    log(args)

    if "-t" in args:
        if os.environ.get("FAKE_SSHD_T_FAIL") == "1":
            sys.stderr.write("fake sshd: forced -t failure\n")
            return 1
        return 0

    if "-T" in args:
        eff = _effective_config()
        for k in sorted(eff):
            print(f"{k} {eff[k]}")
        return 0

    pidfile = "/run/sshd.pid"
    it = iter(args)
    for a in it:
        if a == "-o":
            val = next(it, "")
            if val.startswith("PidFile="):
                pidfile = val.split("=", 1)[1]

    pid = os.fork()
    if pid == 0:
        os.setsid()
        try:
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            libc.prctl(15, b"sshd\0", 0, 0, 0)  # PR_SET_NAME, so /proc/pid/comm == "sshd"
        except Exception:
            pass
        devnull = os.open(os.devnull, os.O_RDWR)
        os.dup2(devnull, 0)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        with open(pidfile, "w") as f:
            f.write(str(os.getpid()))
        signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))
        while True:
            time.sleep(3600)
    else:
        return 0

if __name__ == "__main__":
    sys.exit(main())
"""

_FAKE_SUPERVISORCTL = r"""#!/usr/bin/env python3
import os, sys

def log(args):
    p = os.environ.get("FAKE_CMD_LOG")
    if p:
        with open(p, "a") as f:
            f.write("supervisorctl " + " ".join(args) + "\n")

def main():
    args = sys.argv[1:]
    log(args)
    cmd = args[-1] if args else ""
    if cmd == "reread":
        if os.environ.get("FAKE_SUPERVISORCTL_REREAD_FAIL") == "1":
            sys.stderr.write("fake supervisorctl: reread failed\n")
            return 1
        print("sshd: available")
        return 0
    if cmd == "update":
        print("sshd: added process group")
        return 0
    return 0

if __name__ == "__main__":
    sys.exit(main())
"""

_FAKE_BINS = {
    "apt-get": _FAKE_APT_GET,
    "apt-cache": _FAKE_APT_CACHE,
    "ssh-keygen": _FAKE_SSH_KEYGEN,
    "sshd": _FAKE_SSHD,
    "supervisorctl": _FAKE_SUPERVISORCTL,
}


def _write_fake_bins(bin_dir: Path) -> None:
    bin_dir.mkdir(parents=True, exist_ok=True)
    for name, src in _FAKE_BINS.items():
        p = bin_dir / name
        p.write_text(src, encoding="utf-8")
        p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


# ---------------------------------------------------------------------------
# Per-test fixture: a whole fake root, fresh PATH, real ssh-keygen check
# via the fake binary (not skipped), no network.
# ---------------------------------------------------------------------------

class _Fixture:
    """One throwaway fake root + fake bin dir + patched module globals.
    Tracks any daemonized fake-sshd pid it started so teardown can reap it
    — real background processes leaked across test runs would otherwise
    keep the suite from ever exiting cleanly."""

    def __init__(self):
        self.root = Path(tempfile.mkdtemp(prefix="setup-sshd-test-"))
        self.bin_dir = self.root / "fakebin"
        _write_fake_bins(self.bin_dir)

        self.etc_ssh = self.root / "etc" / "ssh"
        self.etc_ssh_d = self.etc_ssh / "sshd_config.d"
        self.sshd_config = self.etc_ssh / "sshd_config"
        self.root_home = self.root / "roothome"
        self.data_ssh = self.root / "data" / "ssh"
        self.supervisor_conf = self.root / "supervisor" / "supervisord.conf"
        self.run_dir = self.root / "run"
        self.marker_venv = self.root / "opt" / "compactor-venv"
        self.marker_main = self.root / "opt" / "compactor" / "main.py"
        self.apt_state_file = self.root / "apt-installed-version.txt"
        self.cmd_log = self.root / "cmd.log"

        self.etc_ssh.mkdir(parents=True, exist_ok=True)
        self.etc_ssh_d.mkdir(parents=True, exist_ok=True)
        self.marker_venv.mkdir(parents=True, exist_ok=True)
        self.marker_main.parent.mkdir(parents=True, exist_ok=True)
        self.marker_main.write_text("# fake compactor main\n", encoding="utf-8")
        self.supervisor_conf.parent.mkdir(parents=True, exist_ok=True)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Real shipped sshd_config shape, verified against the actual
        # angreg/zions-light-ai image (OPERATIONS.md): Include near the
        # top (line 12), and the only ACTIVE directive this script cares
        # about that ships active is KbdInteractiveAuthentication no, at
        # line 71 — well AFTER the Include, so the drop-in case applies.
        self.sshd_config.write_text(
            "\n"
            "# This is the sshd server system-wide configuration file.\n"
            "\n"
            f"Include {self.etc_ssh_d}/*.conf\n"
            "\n"
            "#Port 22\n"
            "#PermitRootLogin prohibit-password\n"
            "#PubkeyAuthentication yes\n"
            "#PasswordAuthentication yes\n"
            "#PermitEmptyPasswords no\n"
            "KbdInteractiveAuthentication no\n"
            "UsePAM yes\n",
            encoding="utf-8",
        )
        self.supervisor_conf.write_text(
            "[supervisord]\nnodaemon=true\n\n"
            "[program:vllm]\ncommand=/bin/true\n\n"
            "[program:compactor]\ncommand=/bin/true\n",
            encoding="utf-8",
        )

        self._env_patch = patch.dict(os.environ, {
            "SETUP_SSHD_ETC_SSH_DIR": str(self.etc_ssh),
            "SETUP_SSHD_ROOT_HOME": str(self.root_home),
            "SETUP_SSHD_HOST_KEY_STORE": str(self.data_ssh),
            "SETUP_SSHD_SUPERVISOR_CONF": str(self.supervisor_conf),
            "SETUP_SSHD_RUN_DIR": str(self.run_dir),
            "SETUP_SSHD_MARKER_VENV": str(self.marker_venv),
            "SETUP_SSHD_MARKER_MAIN": str(self.marker_main),
            "FAKE_CMD_LOG": str(self.cmd_log),
            "FAKE_APT_STATE_FILE": str(self.apt_state_file),
            "FAKE_APT_CANDIDATE": "1:9.9p1-1ubuntu1",
            "PATH": str(self.bin_dir) + os.pathsep + os.environ.get("PATH", ""),
            "PUBLIC_KEY": "",
        }, clear=False)
        # Scrub anything a previous fixture left behind for these specific
        # test-only knobs, so one test's forced-failure flag never leaks
        # into the next.
        for k in ("FAKE_APT_UPDATE_FAIL", "FAKE_APT_INSTALL_FAIL",
                  "FAKE_SSH_KEYGEN_FORCE_FAIL", "FAKE_SSHD_T_FAIL",
                  "FAKE_SSHD_FORCE_PASSWORD_AUTH", "FAKE_SUPERVISORCTL_REREAD_FAIL"):
            os.environ.pop(k, None)

        # Re-point the script's own module-level path constants — they
        # were computed once at import time from (at that point) the REAL
        # environment, so patching os.environ now is not enough on its
        # own; this is the same pattern as patching _utc_stamp, applied to
        # data instead of a function.
        self._attr_patch = patch.multiple(
            _script,
            ETC_SSH_DIR=self.etc_ssh,
            SSHD_CONFIG=self.sshd_config,
            SSHD_CONFIG_D=self.etc_ssh_d,
            DROPIN_PATH=self.etc_ssh_d / "00-zions.conf",
            ROOT_HOME_DIR=self.root_home,
            ROOT_SSH_DIR=self.root_home / ".ssh",
            AUTHORIZED_KEYS=self.root_home / ".ssh" / "authorized_keys",
            HOST_KEY_STORE=self.data_ssh,
            SUPERVISOR_CONF=self.supervisor_conf,
            RUN_DIR=self.run_dir,
            SSHD_PIDFILE=self.run_dir / "sshd.pid",
            PRIVSEP_DIR=self.run_dir / "sshd",
            MARKER_VENV=self.marker_venv,
            MARKER_MAIN=self.marker_main,
        )

        self._started_pids: list[int] = []

    def __enter__(self):
        self._env_patch.start()
        self._attr_patch.start()
        return self

    def __exit__(self, *exc):
        self._attr_patch.stop()
        self._env_patch.stop()
        pidfile = self.run_dir / "sshd.pid"
        if pidfile.is_file():
            try:
                pid = int(pidfile.read_text().strip())
                os.kill(pid, 15)
            except (ValueError, OSError):
                pass
        for pid in self._started_pids:
            try:
                os.kill(pid, 15)
            except OSError:
                pass
        shutil.rmtree(self.root, ignore_errors=True)

    def run(self, argv, euid=0):
        """Call the real script in-process (its own main(), argparse and
        all) with a patched euid, and capture what it printed — the
        in-process equivalent of the other scripts' subprocess-based
        `_run_script`, needed here because the euid check has to be
        deterministic regardless of which user actually owns this test
        process."""
        buf = io.StringIO()
        with patch.object(_script, "_geteuid", return_value=euid):
            with redirect_stdout(buf):
                rc = _script.main(argv)
        out = buf.getvalue()
        if self.run_dir.joinpath("sshd.pid").is_file():
            try:
                self._started_pids.append(int((self.run_dir / "sshd.pid").read_text().strip()))
            except (ValueError, OSError):
                pass
        return rc, out

    _IGNORED_TOP_LEVEL = ("fakebin", "cmd.log", "apt-installed-version.txt")

    def snapshot(self) -> dict:
        """Every regular file under the whole fake root, as bytes, keyed
        by relative path — used to assert a dry run changes NOTHING. Skips
        the fake-bin directory and this test's OWN instrumentation files
        (the command log, the fake apt "installed version" state file) —
        those are test scaffolding, not anything the real script touches
        or that a real dry run would be judged against."""
        out = {}
        for p in sorted(self.root.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(self.root)
            if rel.parts and rel.parts[0] in self._IGNORED_TOP_LEVEL:
                continue
            out[str(rel)] = p.read_bytes()
        return out

    def cmd_log_text(self) -> str:
        return self.cmd_log.read_text(encoding="utf-8") if self.cmd_log.is_file() else ""


_ED25519_KEY_A = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOperatorAAAAAAAAAAAAAAAAAAAAAAA operator@laptop"
_ED25519_KEY_B = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAISecondAAAAAAAAAAAAAAAAAAAAAAAAA someone@else"


# ---------------------------------------------------------------------------
# 1. Dry run writes NOTHING
# ---------------------------------------------------------------------------

def test_dry_run_writes_nothing_and_installs_nothing():
    print("\n[test] dry run leaves every temp-root file byte-identical and runs no install command")
    with _Fixture() as fx:
        before = fx.snapshot()
        rc, out = fx.run(["--authorized-key", _ED25519_KEY_A])
        after = fx.snapshot()

        assert_eq(before, after, "not one byte changed anywhere under the fake root")
        log = fx.cmd_log_text()
        assert_true("apt-get update" in log, "apt-get update really did run (informational only)")
        assert_true(
            not any(
                line.strip() in ("apt-get install --no-install-recommends -y "
                                  "-o Dpkg::Options::=--force-confold openssh-server",)
                for line in []
            ),
            "sanity: placeholder never trips",
        )
        # the REAL install/upgrade command must never appear without -s
        for line in log.splitlines():
            if line.startswith("apt-get ") and "install" in line and " -s " not in f" {line} ":
                assert_true(False, f"a non-simulated apt-get install ran in dry run: {line}")
        assert_true("DRY RUN" in out, "human output says DRY RUN")
        assert_eq(rc, 0, "dry run on a fresh container reports real work pending -> exit 0")


def test_dry_run_reports_planned_action_and_key():
    print("\n[test] dry run reports the planned install action and key source")
    with _Fixture() as fx:
        rc, out = fx.run(["--json", "--authorized-key", _ED25519_KEY_A])
        payload = json.loads(out)
        assert_eq(payload["mode"], "dry-run", "mode field")
        assert_eq(payload["action"], "install", "openssh-server not installed yet -> action=install")
        assert_eq(payload["key_source"], "literal", "key source is the --authorized-key literal")
        assert_eq(payload["keys_added"], [_ED25519_KEY_A], "the one new key would be added")
        assert_true(not fx.root_home.joinpath(".ssh", "authorized_keys").exists(),
                    "authorized_keys was NOT created in dry run")


# ---------------------------------------------------------------------------
# 2. --apply on a clean container
# ---------------------------------------------------------------------------

def test_apply_on_clean_container_installs_keys_and_starts():
    print("\n[test] --apply on a clean container installs, writes the drop-in, adds the key, starts sshd")
    with _Fixture() as fx:
        rc, out = fx.run(["--apply", "--json", "--authorized-key", _ED25519_KEY_A])
        payload = json.loads(out)

        assert_eq(rc, 0, f"--apply succeeds (refusals={payload.get('refusals')})")
        assert_eq(payload["action"], "install", "action=install")
        assert_eq(payload["installed_version"], "1:9.9p1-1ubuntu1", "installed version now matches candidate")
        assert_eq(payload["key_source"], "literal", "key source")
        assert_eq(payload["keys_added"], [_ED25519_KEY_A], "the key was added")
        assert_eq(payload["config_mode"], "dropin", "the real shipped config supports the drop-in case")
        assert_eq(payload["config_path"], str(fx.etc_ssh_d / "00-zions.conf"), "drop-in path")
        assert_eq(payload["host_keys"], "persisted", "host keys persisted by default")
        assert_eq(payload["daemon"], "started", "daemon started")
        assert_true("password authentication is disabled" in payload["sshd_check"],
                    "sshd -T verification really ran and passed")

        dropin = (fx.etc_ssh_d / "00-zions.conf").read_text()
        assert_in("PasswordAuthentication no", dropin, "drop-in sets PasswordAuthentication no")
        assert_in("PermitRootLogin prohibit-password", dropin, "drop-in sets PermitRootLogin")

        ak = fx.root_home / ".ssh" / "authorized_keys"
        assert_true(ak.is_file(), "authorized_keys created")
        assert_eq(oct(ak.stat().st_mode & 0o777), "0o600", "authorized_keys is 0600")
        assert_eq(oct((fx.root_home / ".ssh").stat().st_mode & 0o777), "0o700", ".ssh dir is 0700")
        assert_in(_ED25519_KEY_A, ak.read_text(), "the key line is actually in the file")

        for t in ("rsa", "ecdsa", "ed25519"):
            store_priv = fx.data_ssh / f"ssh_host_{t}_key"
            assert_true(store_priv.is_file(), f"{t} host key persisted to /data/ssh")
            assert_eq(oct(store_priv.stat().st_mode & 0o777), "0o600", f"{t} host key is 0600")
        assert_eq(oct(fx.data_ssh.stat().st_mode & 0o777), "0o700", "/data/ssh dir is 0700")

        pidfile = fx.run_dir / "sshd.pid"
        assert_true(pidfile.is_file(), "sshd pidfile written")
        pid = int(pidfile.read_text().strip())
        assert_true(Path(f"/proc/{pid}").is_dir(), "sshd (fake) is really running")
        assert_eq((Path(f"/proc/{pid}") / "comm").read_text().strip(), "sshd",
                  "the running process really identifies as sshd")


def test_apply_never_reads_or_prints_a_private_key():
    print("\n[test] no private key material ever appears in stdout/stderr/JSON")
    with _Fixture() as fx:
        rc, out = fx.run(["--apply", "--json", "--authorized-key", _ED25519_KEY_A])
        assert_eq(rc, 0, "--apply succeeds")
        assert_true("PRIVATE" not in out.upper() or "FAKE-PRIVATE-HOST-KEY" not in out,
                    "no marker of the fake host PRIVATE key content leaked into output")
        for t in ("rsa", "ecdsa", "ed25519"):
            priv_marker = f"FAKE-PRIVATE-HOST-KEY-{t}"
            assert_true(priv_marker not in out, f"{t} private host key content not in output")
        payload = json.loads(out)
        blob = json.dumps(payload)
        assert_true("FAKE-PRIVATE-HOST-KEY" not in blob, "private host key content not anywhere in the JSON")


# ---------------------------------------------------------------------------
# 3. Idempotency
# ---------------------------------------------------------------------------

def test_apply_twice_is_idempotent_and_second_run_exits_3():
    print("\n[test] --apply twice leaves the same state; the second is a no-op, exit 3")
    with _Fixture() as fx:
        rc1, out1 = fx.run(["--apply", "--json", "--authorized-key", _ED25519_KEY_A])
        assert_eq(rc1, 0, "first --apply succeeds")
        before = fx.snapshot()
        # the pidfile's own content (the running pid) is allowed to be
        # identical too, since the daemon should NOT be restarted the
        # second time (nothing changed) — captured for the daemon-status
        # assertion below, not diffed away.

        rc2, out2 = fx.run(["--apply", "--json", "--authorized-key", _ED25519_KEY_A])
        after = fx.snapshot()
        payload2 = json.loads(out2)

        assert_eq(rc2, 3, "second --apply reports nothing-to-do -> exit 3")
        assert_eq(payload2["action"], "already-current", "package already at candidate")
        assert_eq(payload2["keys_added"], [], "no new key the second time")
        assert_eq(payload2["daemon"], "already-running", "daemon left alone, not restarted")
        assert_eq(before, after, "not one byte changed on the second, idempotent --apply")


# ---------------------------------------------------------------------------
# 4. Existing authorized_keys: append-only, never clobbered, no duplicates
# ---------------------------------------------------------------------------

def test_existing_authorized_keys_preserved_and_appended_not_clobbered():
    print("\n[test] an existing authorized_keys with a different key is preserved and appended to")
    with _Fixture() as fx:
        fx.root_home.joinpath(".ssh").mkdir(parents=True, exist_ok=True)
        os.chmod(fx.root_home / ".ssh", 0o700)
        existing_key = _ED25519_KEY_B
        (fx.root_home / ".ssh" / "authorized_keys").write_text(existing_key + "\n", encoding="utf-8")
        os.chmod(fx.root_home / ".ssh" / "authorized_keys", 0o600)

        rc, out = fx.run(["--apply", "--json", "--authorized-key", _ED25519_KEY_A])
        assert_eq(rc, 0, "--apply succeeds")
        text = (fx.root_home / ".ssh" / "authorized_keys").read_text()
        assert_in(existing_key, text, "the pre-existing key is still present")
        assert_in(_ED25519_KEY_A, text, "the new key was appended")
        payload = json.loads(out)
        assert_eq(payload["keys_added"], [_ED25519_KEY_A], "only the new key was reported added")
        assert_true("authorized_keys_backup" in payload, "a backup of the pre-existing file was made")
        backup_path = Path(payload["authorized_keys_backup"])
        assert_true(backup_path.is_file(), "the backup file really exists")
        assert_eq(backup_path.read_text(), existing_key + "\n", "the backup holds the ORIGINAL content")


def test_duplicate_key_is_not_added_twice():
    print("\n[test] re-running with the same key does not duplicate it")
    with _Fixture() as fx:
        rc1, _ = fx.run(["--apply", "--authorized-key", _ED25519_KEY_A])
        assert_eq(rc1, 0, "first apply adds the key")
        rc2, out2 = fx.run(["--apply", "--json", "--authorized-key", _ED25519_KEY_A])
        payload2 = json.loads(out2)
        assert_eq(payload2["keys_added"], [], "the same key again is not re-added")
        text = (fx.root_home / ".ssh" / "authorized_keys").read_text()
        assert_eq(text.count(_ED25519_KEY_A.split()[1]), 1, "the key's base64 material appears exactly once")


# ---------------------------------------------------------------------------
# 5. Refusals — each exits 1 and changes nothing
# ---------------------------------------------------------------------------

def test_refuses_when_not_root():
    print("\n[test] refuses when not root, and changes nothing")
    with _Fixture() as fx:
        before = fx.snapshot()
        rc, out = fx.run(["--apply", "--json", "--authorized-key", _ED25519_KEY_A], euid=1000)
        after = fx.snapshot()
        payload = json.loads(out)
        assert_eq(rc, 1, "not-root refuses with exit 1")
        assert_true(any("root" in r for r in payload["refusals"]), "refusal mentions root")
        assert_eq(before, after, "nothing changed")


def test_refuses_when_no_key_available_anywhere():
    print("\n[test] refuses when no public key is available from any source")
    with _Fixture() as fx:
        before = fx.snapshot()
        rc, out = fx.run(["--apply", "--json"])
        after = fx.snapshot()
        payload = json.loads(out)
        assert_eq(rc, 1, "no-key refuses with exit 1")
        assert_true(any("no usable public key" in r for r in payload["refusals"]), "refusal names the cause")
        assert_eq(before, after, "nothing changed")


def test_refuses_on_malformed_public_key():
    print("\n[test] refuses on a malformed public key, and changes nothing")
    with _Fixture() as fx:
        before = fx.snapshot()
        rc, out = fx.run(["--apply", "--json", "--authorized-key", "not-a-real-key-at-all"])
        after = fx.snapshot()
        payload = json.loads(out)
        assert_eq(rc, 1, "malformed key refuses with exit 1")
        assert_true(any("malformed public key" in r for r in payload["refusals"]), "refusal names the cause")
        assert_eq(before, after, "nothing changed")
        assert_true(not fx.root_home.joinpath(".ssh", "authorized_keys").exists(),
                    "authorized_keys was never created")


def test_refuses_when_effective_config_would_allow_password_auth():
    print("\n[test] refuses when sshd -T reports password auth would be allowed, and rolls back")
    with _Fixture() as fx:
        os.environ["FAKE_SSHD_FORCE_PASSWORD_AUTH"] = "1"
        try:
            before_dropin_exists = (fx.etc_ssh_d / "00-zions.conf").exists()
            rc, out = fx.run(["--apply", "--json", "--authorized-key", _ED25519_KEY_A])
            payload = json.loads(out)
            assert_eq(rc, 1, "refuses with exit 1")
            assert_true(any("passwordauthentication" in r for r in payload["refusals"]),
                        "refusal names the real sshd -T finding")
            assert_eq((fx.etc_ssh_d / "00-zions.conf").exists(), before_dropin_exists,
                      "the drop-in was rolled back to its prior state (absent)")
            pidfile = fx.run_dir / "sshd.pid"
            assert_true(not pidfile.is_file(), "sshd was never started after the refusal")
        finally:
            os.environ.pop("FAKE_SSHD_FORCE_PASSWORD_AUTH", None)


def test_refuses_when_sshd_t_fails():
    print("\n[test] refuses when sshd -t (syntax check) fails, and rolls back")
    with _Fixture() as fx:
        os.environ["FAKE_SSHD_T_FAIL"] = "1"
        try:
            rc, out = fx.run(["--apply", "--json", "--authorized-key", _ED25519_KEY_A])
            payload = json.loads(out)
            assert_eq(rc, 1, "refuses with exit 1")
            assert_true(any("sshd -t failed" in r for r in payload["refusals"]), "refusal names sshd -t")
            assert_true(not (fx.etc_ssh_d / "00-zions.conf").exists(), "the drop-in was rolled back")
            assert_true(not (fx.run_dir / "sshd.pid").is_file(), "sshd was never started")
        finally:
            os.environ.pop("FAKE_SSHD_T_FAIL", None)


# ---------------------------------------------------------------------------
# 6. Never apt-get upgrade / dist-upgrade
# ---------------------------------------------------------------------------

def test_never_calls_apt_get_upgrade_or_dist_upgrade():
    print("\n[test] the script never invokes apt-get upgrade or dist-upgrade, in dry run or --apply")
    with _Fixture() as fx:
        fx.run(["--authorized-key", _ED25519_KEY_A])  # dry run
        fx.run(["--apply", "--authorized-key", _ED25519_KEY_A])
        fx.run(["--apply", "--authorized-key", _ED25519_KEY_A])  # idempotent re-run
        log = fx.cmd_log_text()
        for line in log.splitlines():
            if not line.startswith("apt-get "):
                continue
            tokens = line.split()[1:]
            assert_true("upgrade" not in tokens, f"'upgrade' as a bare apt-get subcommand: {line!r}")
            assert_true("dist-upgrade" not in tokens, f"'dist-upgrade' as a bare apt-get subcommand: {line!r}")


# ---------------------------------------------------------------------------
# 7. No private key material anywhere in output (dry run + refusal paths too)
# ---------------------------------------------------------------------------

def test_no_private_key_material_in_any_mode():
    print("\n[test] no private key material leaks in dry run, refusal, or --json, across every mode")
    with _Fixture() as fx:
        runs = [
            fx.run(["--authorized-key", _ED25519_KEY_A]),
            fx.run(["--json", "--authorized-key", _ED25519_KEY_A]),
            fx.run(["--apply", "--authorized-key", _ED25519_KEY_A]),
            fx.run(["--apply", "--json"]),  # second apply, no key given -> existing file source
        ]
        for _, out in runs:
            assert_true("FAKE-PRIVATE-HOST-KEY" not in out, "no private host key content in output")
            assert_true("-----BEGIN" not in out, "no PEM private-key marker in output")


# ---------------------------------------------------------------------------
# 8. Key source precedence: file > literal > PUBLIC_KEY > existing file
# ---------------------------------------------------------------------------

def test_key_source_precedence():
    print("\n[test] key source precedence is file > literal > $PUBLIC_KEY > existing authorized_keys")
    with _Fixture() as fx:
        # existing file only
        fx.root_home.joinpath(".ssh").mkdir(parents=True, exist_ok=True)
        (fx.root_home / ".ssh" / "authorized_keys").write_text(_ED25519_KEY_B + "\n", encoding="utf-8")
        rc, out = fx.run(["--json"])
        assert_eq(json.loads(out)["key_source"], "existing-authorized_keys", "falls back to the existing file")

        # $PUBLIC_KEY beats the existing file
        os.environ["PUBLIC_KEY"] = _ED25519_KEY_A
        try:
            rc, out = fx.run(["--json"])
            assert_eq(json.loads(out)["key_source"], "env:PUBLIC_KEY", "$PUBLIC_KEY takes precedence")

            # --authorized-key literal beats $PUBLIC_KEY
            rc, out = fx.run(["--json", "--authorized-key", _ED25519_KEY_B])
            assert_eq(json.loads(out)["key_source"], "literal", "--authorized-key beats $PUBLIC_KEY")

            # --authorized-key-file beats everything
            key_file = fx.root / "operator.pub"
            key_file.write_text(_ED25519_KEY_A + "\n", encoding="utf-8")
            rc, out = fx.run([
                "--json", "--authorized-key-file", str(key_file),
                "--authorized-key", _ED25519_KEY_B,
            ])
            assert_eq(json.loads(out)["key_source"], "file", "--authorized-key-file wins over everything")
        finally:
            os.environ.pop("PUBLIC_KEY", None)


# ---------------------------------------------------------------------------
# 9. --json is valid JSON with the documented keys, in every mode
# ---------------------------------------------------------------------------

_REQUIRED_JSON_KEYS = {
    "mode", "installed_version", "candidate_version", "action", "key_source",
    "keys_added", "config_path", "daemon", "host_keys", "port", "refusals",
    "exit_code",
}


def test_json_output_has_the_documented_keys_in_every_mode():
    print("\n[test] --json output is valid JSON and carries every documented key, dry-run/apply/refusal")
    with _Fixture() as fx:
        rc, out = fx.run(["--json", "--authorized-key", _ED25519_KEY_A])
        payload = json.loads(out)
        assert_true(_REQUIRED_JSON_KEYS.issubset(payload.keys()),
                    f"dry-run JSON missing keys: {_REQUIRED_JSON_KEYS - payload.keys()}")
        assert_eq(payload["exit_code"], rc, "JSON exit_code matches the real process exit code")

        rc2, out2 = fx.run(["--apply", "--json", "--authorized-key", _ED25519_KEY_A])
        payload2 = json.loads(out2)
        assert_true(_REQUIRED_JSON_KEYS.issubset(payload2.keys()),
                    f"apply JSON missing keys: {_REQUIRED_JSON_KEYS - payload2.keys()}")
        assert_eq(payload2["exit_code"], rc2, "JSON exit_code matches on --apply too")

        rc3, out3 = fx.run(["--apply", "--json"], euid=1000)  # refusal path
        payload3 = json.loads(out3)
        assert_true(_REQUIRED_JSON_KEYS.issubset(payload3.keys()),
                    f"refusal JSON missing keys: {_REQUIRED_JSON_KEYS - payload3.keys()}")
        assert_true(len(payload3["refusals"]) > 0, "refusal JSON actually lists a refusal")


# ---------------------------------------------------------------------------
# Extra coverage: --port, --no-persist-host-keys, --supervise, the
# container-check refusal and its --force override.
# ---------------------------------------------------------------------------

def test_custom_port_written_and_reported():
    print("\n[test] --port writes a Port directive and is reported back")
    with _Fixture() as fx:
        rc, out = fx.run(["--apply", "--json", "--port", "2222", "--authorized-key", _ED25519_KEY_A])
        payload = json.loads(out)
        assert_eq(rc, 0, "--apply with a custom port succeeds")
        assert_eq(payload["port"], 2222, "port reported back")
        dropin = (fx.etc_ssh_d / "00-zions.conf").read_text()
        assert_in("Port 2222", dropin, "the drop-in carries the custom port")


def test_no_persist_host_keys_leaves_data_ssh_empty():
    print("\n[test] --no-persist-host-keys never touches /data/ssh")
    with _Fixture() as fx:
        rc, out = fx.run([
            "--apply", "--json", "--no-persist-host-keys",
            "--authorized-key", _ED25519_KEY_A,
        ])
        payload = json.loads(out)
        assert_eq(rc, 0, "--apply succeeds")
        assert_eq(payload["host_keys"], "ephemeral", "host_keys reported ephemeral")
        assert_true(not fx.data_ssh.exists() or not any(fx.data_ssh.iterdir()),
                    "/data/ssh was never populated")


def test_container_check_refuses_without_force_and_overridden_with_force():
    print("\n[test] refuses on a container that doesn't look like zions, unless --force")
    with _Fixture() as fx:
        shutil.rmtree(fx.marker_venv)
        rc, out = fx.run(["--apply", "--json", "--authorized-key", _ED25519_KEY_A])
        payload = json.loads(out)
        assert_eq(rc, 1, "refuses without --force")
        assert_true(any("zions container" in r for r in payload["refusals"]), "refusal names the cause")

        rc2, out2 = fx.run(["--apply", "--json", "--force", "--authorized-key", _ED25519_KEY_A])
        payload2 = json.loads(out2)
        assert_eq(rc2, 0, "--force overrides the container-check refusal")
        assert_true(any("--force" in w for w in payload2["warnings"]), "a warning names the override")


def test_supervise_appends_program_and_never_touches_other_programs():
    print("\n[test] --supervise appends [program:sshd] and leaves the other programs' text untouched")
    with _Fixture() as fx:
        before_text = fx.supervisor_conf.read_text()
        rc, out = fx.run([
            "--apply", "--json", "--supervise", "--authorized-key", _ED25519_KEY_A,
        ])
        payload = json.loads(out)
        assert_eq(rc, 0, "--apply --supervise succeeds")
        after_text = fx.supervisor_conf.read_text()
        assert_in(before_text.strip(), after_text, "every original section's text is still present, untouched")
        assert_in("[program:sshd]", after_text, "the new program section was appended")
        log = fx.cmd_log_text()
        assert_in("supervisorctl", log, "supervisorctl was actually invoked")

        rc2, out2 = fx.run([
            "--apply", "--json", "--supervise", "--authorized-key", _ED25519_KEY_A,
        ])
        payload2 = json.loads(out2)
        assert_true(payload2["supervise"]["already_present"],
                    "a second --supervise run recognises the program is already there")


def test_supervise_reread_failure_restores_backup_and_refuses():
    print("\n[test] a failed supervisorctl reread restores the backup and refuses")
    with _Fixture() as fx:
        before_text = fx.supervisor_conf.read_text()
        os.environ["FAKE_SUPERVISORCTL_REREAD_FAIL"] = "1"
        try:
            rc, out = fx.run([
                "--apply", "--json", "--supervise", "--authorized-key", _ED25519_KEY_A,
            ])
            payload = json.loads(out)
            assert_eq(rc, 1, "refuses with exit 1")
            assert_eq(fx.supervisor_conf.read_text(), before_text,
                      "supervisord.conf was restored to its exact original text")
        finally:
            os.environ.pop("FAKE_SUPERVISORCTL_REREAD_FAIL", None)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_dry_run_writes_nothing_and_installs_nothing()
    test_dry_run_reports_planned_action_and_key()

    test_apply_on_clean_container_installs_keys_and_starts()
    test_apply_never_reads_or_prints_a_private_key()

    test_apply_twice_is_idempotent_and_second_run_exits_3()

    test_existing_authorized_keys_preserved_and_appended_not_clobbered()
    test_duplicate_key_is_not_added_twice()

    test_refuses_when_not_root()
    test_refuses_when_no_key_available_anywhere()
    test_refuses_on_malformed_public_key()
    test_refuses_when_effective_config_would_allow_password_auth()
    test_refuses_when_sshd_t_fails()

    test_never_calls_apt_get_upgrade_or_dist_upgrade()

    test_no_private_key_material_in_any_mode()

    test_key_source_precedence()

    test_json_output_has_the_documented_keys_in_every_mode()

    test_custom_port_written_and_reported()
    test_no_persist_host_keys_leaves_data_ssh_empty()
    test_container_check_refuses_without_force_and_overridden_with_force()
    test_supervise_appends_program_and_never_touches_other_programs()
    test_supervise_reread_failure_restores_backup_and_refuses()

    print("\nAll setup-sshd.py script tests passed.")
