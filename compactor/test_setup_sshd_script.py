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
script's own file-writing logic, not a canned string. It also really BINDS
a TCP socket for every configured port when it "starts", so the
production script's H6 listening check (which reads the real, unfaked
/proc/net/tcp) is exercised against a genuine kernel-level fact rather
than a second layer of mocking. The fake `ssh-keygen -l -f` is FAITHFUL to
the real binary in one specific, security-relevant way: it returns exit 0
on a PRIVATE key file too (confirmed against the real image) — that
faithfulness is exactly why B3's fix (a type-token / "PRIVATE KEY" check
that runs BEFORE ssh-keygen is ever invoked) has to exist and is tested
directly.

Every test that does not care about the exact port uses a fixed
non-privileged default (`_Fixture.run` injects `--port 2022` unless the
test's own argv already specifies one) so the suite runs the same whether
or not the process happens to have real CAP_NET_BIND_SERVICE for port 22.

Run inside the compactor image or any container with the requirements
installed:
    python test_setup_sshd_script.py
"""

import io
import json
import os
import shutil
import socket
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


def assert_not_in(needle, haystack, label):
    if needle in haystack:
        print(f"FAIL {label}: {needle!r} unexpectedly found")
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

# FAITHFUL to the real ssh-keygen (confirmed against the real image): `-l
# -f` returns exit 0 on a PRIVATE key file too, not just a public one. The
# production script's own type-token / "PRIVATE KEY" pre-check (B3) is
# what actually has to keep a private key out — this fake is deliberately
# NOT stricter than the real binary, so a test exercising it is exercising
# the production script's own defense, not a mock that happens to reject
# what it should.
_FAKE_SSH_KEYGEN = r"""#!/usr/bin/env python3
import os, re, sys

PUB_RE = re.compile(
    r"^(ssh-(rsa|ed25519|dss)|ecdsa-sha2-\S+|sk-ssh-ed25519@openssh\.com|"
    r"sk-ecdsa-sha2-nistp256@openssh\.com)\s+[A-Za-z0-9+/]+=*(\s+.*)?$"
)
PRIV_MARKERS = (
    "-----BEGIN OPENSSH PRIVATE KEY-----",
    "-----BEGIN RSA PRIVATE KEY-----",
    "-----BEGIN EC PRIVATE KEY-----",
    "-----BEGIN DSA PRIVATE KEY-----",
    "-----BEGIN PRIVATE KEY-----",
)

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
        if content.startswith(PRIV_MARKERS):
            # Faithful to the real binary: rc==0 on a PRIVATE key file.
            print("256 SHA256:fakeprivatekeyfingerprint comment (ED25519)")
            return 0
        if not content or not PUB_RE.match(content):
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
                # A UNIQUE value each time this is actually invoked (real
                # ssh-keygen -A creates a genuinely new random key) — so a
                # test comparing byte content across two runs can tell
                # "restored/left alone" apart from "silently
                # regenerated" (M8), which a fixed string could not.
                unique = os.urandom(8).hex()
                with open(priv, "w") as f:
                    f.write(f"FAKE-PRIVATE-HOST-KEY-{t}-{unique}-DO-NOT-USE\n")
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

# The fake sshd really resolves Include (first-match-wins, like the real
# daemon) and really BINDS a TCP socket for every configured port when it
# "starts" — so H6's listening check (production code reads the real,
# unfaked /proc/net/tcp) is exercised against genuine kernel state. It
# collects EVERY occurrence of EVERY directive (never just the first) so
# a scenario with two active `Port` lines is faithfully reproduced for
# M8, and `-C addr=...` support lets a test prove the multi-context check
# (B2) independently of the static Match-block scan.
_FAKE_SSHD = r"""#!/usr/bin/env python3
import ctypes, glob, os, signal, socket, sys, time

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
        # Real OpenSSH: most directives are first-match-wins, but a few
        # (Port, ListenAddress, HostKey...) ACCUMULATE across files. This
        # fake collects EVERY occurrence for EVERY key (a safe superset)
        # so a scenario with two active 'Port' lines is faithfully
        # reproduced (see M8) instead of silently losing the second one.
        effective.setdefault(keyl, []).append(val)

def _effective_config(this_addr=None):
    etc_ssh = os.environ.get("SETUP_SSHD_ETC_SSH_DIR", "/etc/ssh")
    cfg = os.path.join(etc_ssh, "sshd_config")
    effective = {}
    _resolve(cfg, effective, set())
    for k, v in _DEFAULTS.items():
        effective.setdefault(k, [v])
    if os.environ.get("FAKE_SSHD_FORCE_PASSWORD_AUTH") == "1":
        effective["passwordauthentication"] = ["yes"]
    forced_addr = os.environ.get("FAKE_SSHD_FORCE_PASSWORD_AUTH_FOR_ADDR")
    if forced_addr and this_addr == forced_addr:
        effective["passwordauthentication"] = ["yes"]
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
        this_addr = None
        if "-C" in args:
            cval = args[args.index("-C") + 1]
            for kv in cval.split(","):
                if kv.startswith("addr="):
                    this_addr = kv.split("=", 1)[1]
        eff = _effective_config(this_addr=this_addr)
        for k in sorted(eff):
            for v in eff[k]:
                print(f"{k} {v}")
        return 0

    pidfile = "/run/sshd.pid"
    it = iter(args)
    for a in it:
        if a == "-o":
            val = next(it, "")
            if val.startswith("PidFile="):
                pidfile = val.split("=", 1)[1]

    ports = _effective_config().get("port", ["22"])

    pid = os.fork()
    if pid == 0:
        os.setsid()
        try:
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            libc.prctl(15, b"sshd\0", 0, 0, 0)  # PR_SET_NAME, so /proc/pid/comm == "sshd"
        except Exception:
            pass
        socks = []
        if os.environ.get("FAKE_SSHD_SKIP_BIND") != "1":
            for p in ports:
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    s.bind(("127.0.0.1", int(p)))
                    s.listen(5)
                    socks.append(s)
                except (OSError, ValueError):
                    pass
        devnull = os.open(os.devnull, os.O_RDWR)
        os.dup2(devnull, 0)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        with open(pidfile, "w") as f:
            f.write(str(os.getpid()))

        def _on_term(*_a):
            # Faithful to real sshd, which removes its own pidfile on a
            # clean SIGTERM shutdown — so _stop_sshd() really leaves no
            # stale pidfile behind, same as production.
            try:
                os.unlink(pidfile)
            except OSError:
                pass
            sys.exit(0)

        signal.signal(signal.SIGTERM, _on_term)
        while True:
            time.sleep(3600)
    else:
        return 0

if __name__ == "__main__":
    sys.exit(main())
"""

_FAKE_SUPERVISORCTL = r"""#!/usr/bin/env python3
import os, re, shlex, signal, subprocess, sys, time

def log(args):
    p = os.environ.get("FAKE_CMD_LOG")
    if p:
        with open(p, "a") as f:
            f.write("supervisorctl " + " ".join(args) + "\n")

def _conf_path(args):
    if "-c" in args:
        return args[args.index("-c") + 1]
    return None

def _sshd_command(conf_path):
    # A real supervisord, on `update`/`restart`, actually launches the
    # [program:sshd] section's own `command=` line. This fake does the
    # same (fire-and-forget), so the production script's H6 listening
    # check is exercised against a genuinely spawned (fake) sshd, not
    # just a supervisorctl exit status.
    try:
        text = open(conf_path, encoding="utf-8", errors="replace").read()
    except OSError:
        return None
    # Anchored at the START of a line so the marker COMMENT ("; --- BEGIN
    # ... [program:sshd] ---", which also contains the literal substring
    # "[program:sshd]") is never mistaken for the real section header.
    m = re.search(r"^\[program:sshd\](.*?)(?=\n\[|\Z)", text, re.S | re.M)
    if not m:
        return None
    m2 = re.search(r"^command=(.*)$", m.group(1), re.M)
    return m2.group(1).strip() if m2 else None

def _pidfile_from_command(cmd):
    for tok in shlex.split(cmd):
        if tok.startswith("PidFile="):
            return tok.split("=", 1)[1]
    return None

def main():
    args = sys.argv[1:]
    log(args)
    cmd = args[-1] if args else ""
    conf_path = _conf_path(args)

    if cmd == "reread":
        if os.environ.get("FAKE_SUPERVISORCTL_REREAD_FAIL") == "1":
            sys.stderr.write("fake supervisorctl: reread failed\n")
            return 1
        print("sshd: available")
        return 0

    if cmd == "update":
        sshd_cmd = _sshd_command(conf_path) if conf_path else None
        if sshd_cmd:
            subprocess.Popen(shlex.split(sshd_cmd))
        print("sshd: added process group")
        return 0

    if cmd == "restart":
        sshd_cmd = _sshd_command(conf_path) if conf_path else None
        if sshd_cmd:
            pidfile = _pidfile_from_command(sshd_cmd)
            if pidfile and os.path.isfile(pidfile):
                try:
                    old_pid = int(open(pidfile).read().strip())
                    os.kill(old_pid, signal.SIGTERM)
                    for _ in range(50):
                        if not os.path.isdir(f"/proc/{old_pid}"):
                            break
                        time.sleep(0.1)
                except (ValueError, OSError):
                    pass
            subprocess.Popen(shlex.split(sshd_cmd))
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


# A fixed, non-privileged default port for every test that does not care
# about the exact value — lets the fake sshd really bind a socket
# regardless of whether the test process has real CAP_NET_BIND_SERVICE.
_DEFAULT_TEST_PORT = "2022"


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
                  "FAKE_SSHD_FORCE_PASSWORD_AUTH", "FAKE_SUPERVISORCTL_REREAD_FAIL",
                  "FAKE_SSHD_FORCE_PASSWORD_AUTH_FOR_ADDR", "FAKE_SSHD_SKIP_BIND"):
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
        process. Injects a fixed, non-privileged --port (2022) when the
        test's own argv does not specify one, so the fake sshd's real
        socket bind never depends on CAP_NET_BIND_SERVICE."""
        argv = list(argv)
        if "--port" not in argv:
            argv = argv + ["--port", _DEFAULT_TEST_PORT]
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

_FAKE_PRIVATE_KEY = (
    "-----BEGIN OPENSSH PRIVATE KEY-----\n"
    "b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAAAMwAAAAtzc2gtZW\n"
    "QyNTUxOQAAACBFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKEAAAAkAAAAAtzc\n"
    "-----END OPENSSH PRIVATE KEY-----"
)


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
        # the REAL install/upgrade command must never appear without -s
        for line in log.splitlines():
            if line.startswith("apt-get ") and "install" in line and " -s " not in f" {line} ":
                assert_true(False, f"a non-simulated apt-get install ran in dry run: {line}")
        assert_true("DRY RUN" in out, "human output says DRY RUN")
        # Defect 4: normalized to match scripts/backfill-records.py and
        # scripts/import-history.py — a DRY RUN that finds pending work is
        # exit 3, not 0 (this script briefly had 0 and 3 swapped from that
        # convention; see CHANGELOG.md v3.1.9.6 "Fixed").
        assert_eq(rc, 3, "dry run on a fresh container reports real work pending -> exit 3")


def test_dry_run_reports_planned_action_and_key():
    print("\n[test] dry run reports the planned install action and key source")
    with _Fixture() as fx:
        rc, out = fx.run(["--json", "--authorized-key", _ED25519_KEY_A])
        payload = json.loads(out)
        assert_eq(payload["mode"], "dry-run", "mode field")
        assert_eq(payload["action"], "install", "openssh-server not installed yet -> action=install")
        assert_eq(payload["key_source"], "literal", "key source is the --authorized-key literal")
        assert_eq(len(payload["keys_added"]), 1, "the one new key would be added")
        assert_true(_ED25519_KEY_A not in payload["keys_added"][0],
                    "the report never carries the raw key line, only a fingerprint (B3)")
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
        assert_eq(len(payload["keys_added"]), 1, "the key was added")
        assert_eq(payload["config_mode"], "dropin", "the real shipped config supports the drop-in case")
        assert_eq(payload["config_path"], str(fx.etc_ssh_d / "00-zions.conf"), "drop-in path")
        assert_eq(payload["host_keys"], "persisted", "host keys persisted by default")
        assert_eq(payload["daemon"], "started", "daemon started")
        assert_true("password authentication is disabled" in payload["sshd_check"],
                    "sshd -T verification really ran and passed")

        dropin = (fx.etc_ssh_d / "00-zions.conf").read_text()
        assert_in("PasswordAuthentication no", dropin, "drop-in sets PasswordAuthentication no")
        assert_in("PermitRootLogin prohibit-password", dropin, "drop-in sets PermitRootLogin")
        assert_in(f"Port {_DEFAULT_TEST_PORT}", dropin, "drop-in always states Port explicitly now (M8)")

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

        # H6: the daemon is really LISTENING (a real bound socket), not
        # merely a process whose exit status was 0.
        assert_true(_script._sshd_listening_on_port(int(_DEFAULT_TEST_PORT)),
                    "sshd is really listening on the configured port (real /proc/net/tcp)")


def test_apply_never_reads_or_prints_a_private_key():
    print("\n[test] no private key material ever appears in stdout/stderr/JSON")
    with _Fixture() as fx:
        rc, out = fx.run(["--apply", "--json", "--authorized-key", _ED25519_KEY_A])
        assert_eq(rc, 0, "--apply succeeds")
        for t in ("rsa", "ecdsa", "ed25519"):
            priv_marker = f"FAKE-PRIVATE-HOST-KEY-{t}"
            assert_true(priv_marker not in out, f"{t} private host key content not in output")
        payload = json.loads(out)
        blob = json.dumps(payload)
        assert_true("FAKE-PRIVATE-HOST-KEY" not in blob, "private host key content not anywhere in the JSON")


# ---------------------------------------------------------------------------
# 3. Idempotency
# ---------------------------------------------------------------------------

def test_apply_twice_is_idempotent_and_second_run_exits_0():
    print("\n[test] --apply twice leaves the same state; the second is a no-op, exit 0")
    with _Fixture() as fx:
        rc1, out1 = fx.run(["--apply", "--json", "--authorized-key", _ED25519_KEY_A])
        assert_eq(rc1, 0, "first --apply succeeds")
        before = fx.snapshot()

        rc2, out2 = fx.run(["--apply", "--json", "--authorized-key", _ED25519_KEY_A])
        after = fx.snapshot()
        payload2 = json.loads(out2)

        # Defect 4: a no-op --apply is SUCCESS (0), not "nothing to do" (3)
        # — 3 is reserved for a DRY RUN finding pending work.
        assert_eq(rc2, 0, "second --apply (a no-op) succeeds -> exit 0")
        assert_eq(payload2["action"], "already-current", "package already at candidate")
        assert_eq(payload2["keys_added"], [], "no new key the second time")
        assert_eq(payload2["daemon"], "already-running", "daemon left alone, not restarted")
        assert_eq(before, after, "not one byte changed on the second, idempotent --apply")


def test_dry_run_after_apply_finds_nothing_to_do_and_exits_0():
    print("\n[test] a DRY RUN after --apply finds nothing pending and exits 0")
    with _Fixture() as fx:
        rc1, _ = fx.run(["--apply", "--json", "--authorized-key", _ED25519_KEY_A])
        assert_eq(rc1, 0, "the --apply that sets everything up succeeds")
        before = fx.snapshot()

        rc2, out2 = fx.run(["--json", "--authorized-key", _ED25519_KEY_A])
        after = fx.snapshot()
        payload2 = json.loads(out2)

        assert_eq(rc2, 0, "a dry run finding nothing to do exits 0, not 3")
        assert_eq(payload2["mode"], "dry-run", "still reports itself as a dry run")
        assert_eq(before, after, "a dry run, even with nothing to do, writes nothing")


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
        assert_eq(len(payload["keys_added"]), 1, "only the new key was reported added")
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
# 5. Refusals — each exits 1 and changes nothing (or rolls back)
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
            assert_true(not fx.root_home.joinpath(".ssh", "authorized_keys").exists(),
                        "authorized_keys was rolled back too (it was freshly created this run)")
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


def test_refuses_when_public_context_would_allow_password_auth():
    print("\n[test] B2: refuses when ONLY the non-local (-C) context allows password auth")
    with _Fixture() as fx:
        os.environ["FAKE_SSHD_FORCE_PASSWORD_AUTH_FOR_ADDR"] = "203.0.113.9"
        try:
            rc, out = fx.run(["--apply", "--json", "--authorized-key", _ED25519_KEY_A])
            payload = json.loads(out)
            assert_eq(rc, 1, "refuses with exit 1 even though a bare `sshd -T` alone is fine")
            assert_true(any("203.0.113.9" in r for r in payload["refusals"]),
                        "refusal specifically names the non-local -C context")
            assert_true(not (fx.etc_ssh_d / "00-zions.conf").exists(),
                        "the drop-in was rolled back")
        finally:
            os.environ.pop("FAKE_SSHD_FORCE_PASSWORD_AUTH_FOR_ADDR", None)


# ---------------------------------------------------------------------------
# 5b. B2 — Match blocks and drop-in ordering, refused BEFORE any write
# ---------------------------------------------------------------------------

def test_refuses_outright_on_match_block_in_main_config():
    print("\n[test] B2: a Match block in sshd_config itself refuses outright, before any write")
    with _Fixture() as fx:
        text = fx.sshd_config.read_text()
        fx.sshd_config.write_text(
            text + "\nMatch Address *\n    PasswordAuthentication yes\n", encoding="utf-8"
        )
        before = fx.snapshot()
        rc, out = fx.run(["--apply", "--json", "--authorized-key", _ED25519_KEY_A])
        after = fx.snapshot()
        payload = json.loads(out)
        assert_eq(rc, 1, "refuses with exit 1")
        assert_true(any("Match" in r for r in payload["refusals"]), "refusal names the Match block")
        assert_eq(payload["key_source"], None, "never even got to resolving a key")
        assert_eq(before, after, "nothing written at all — no rollback needed since nothing was written")


def test_refuses_outright_on_match_block_in_other_dropin():
    print("\n[test] B2: a pre-existing drop-in with a Match block refuses outright (the real-world scenario)")
    with _Fixture() as fx:
        (fx.etc_ssh_d / "00-runpod.conf").write_text(
            "# left by a previous operator / pod template\n"
            "Match Address *\n"
            "    PasswordAuthentication yes\n"
            "    PermitRootLogin yes\n",
            encoding="utf-8",
        )
        before = fx.snapshot()
        rc, out = fx.run(["--apply", "--json", "--authorized-key", _ED25519_KEY_A])
        after = fx.snapshot()
        payload = json.loads(out)
        assert_eq(rc, 1, "refuses with exit 1")
        assert_true(any("Match" in r for r in payload["refusals"]), "refusal names the Match block")
        assert_eq(before, after, "nothing written — the pre-existing drop-in is untouched too")


def test_refuses_outright_on_earlier_sorting_dropin_without_match():
    print("\n[test] B2: another drop-in sorting before 00-zions.conf refuses outright, even with no Match")
    with _Fixture() as fx:
        (fx.etc_ssh_d / "00-early.conf").write_text(
            "PermitRootLogin yes\n", encoding="utf-8"
        )
        rc, out = fx.run(["--apply", "--json", "--authorized-key", _ED25519_KEY_A])
        payload = json.loads(out)
        assert_eq(rc, 1, "refuses with exit 1")
        assert_true(any("00-early.conf" in r for r in payload["refusals"]),
                    "refusal names the earlier-sorting drop-in")


def test_does_not_refuse_on_later_sorting_dropin():
    print("\n[test] B2: a drop-in sorting AFTER 00-zions.conf is fine (it never wins)")
    with _Fixture() as fx:
        (fx.etc_ssh_d / "99-late.conf").write_text(
            "# harmless, loses to our own drop-in\nBanner none\n", encoding="utf-8"
        )
        rc, out = fx.run(["--apply", "--json", "--authorized-key", _ED25519_KEY_A])
        payload = json.loads(out)
        assert_eq(rc, 0, f"--apply still succeeds (refusals={payload.get('refusals')})")


def test_refuses_outright_on_conflicting_port_in_other_dropin():
    print("\n[test] M8: a conflicting active Port directive in another drop-in refuses outright")
    with _Fixture() as fx:
        (fx.etc_ssh_d / "50-other.conf").write_text("Port 22\n", encoding="utf-8")
        rc, out = fx.run([
            "--apply", "--json", "--port", "2222", "--authorized-key", _ED25519_KEY_A,
        ])
        payload = json.loads(out)
        assert_eq(rc, 1, "refuses with exit 1")
        assert_true(any("Port" in r and "50-other.conf" in r for r in payload["refusals"]),
                    "refusal names the conflicting Port source")


# ---------------------------------------------------------------------------
# 5c. B3 — private keys, malformed type tokens, fingerprint-only reporting
# ---------------------------------------------------------------------------

def test_rejects_private_key_and_never_leaks_it():
    print("\n[test] B3: a private key is rejected outright, and its content never appears anywhere")
    with _Fixture() as fx:
        rc, out = fx.run(["--apply", "--json", "--authorized-key", _FAKE_PRIVATE_KEY])
        payload = json.loads(out)
        assert_eq(rc, 1, "refuses with exit 1")
        # A real private key pasted via `$(cat ~/.ssh/id_ed25519)` (the
        # classic .pub typo — see B3) is always multi-line, so the
        # single-line check is what actually catches THIS realistic case;
        # the single-line "PRIVATE KEY" substring check is exercised
        # separately below with a one-line variant.
        assert_true(any("one line" in r for r in payload["refusals"]),
                    "refusal explains the multi-line rejection")
        assert_not_in("FAKEFAKEFAKEFAKE", out, "the private key's own base64 payload never appears in output")
        assert_not_in("BEGIN OPENSSH PRIVATE KEY", out, "the PEM marker never appears in output")
        assert_true(not fx.root_home.joinpath(".ssh", "authorized_keys").exists(),
                    "authorized_keys was never created")


def test_rejects_multiline_key_material():
    print("\n[test] B3: multi-line key material is rejected even without a PRIVATE KEY marker")
    with _Fixture() as fx:
        two_lines = _ED25519_KEY_A + "\n" + _ED25519_KEY_B
        rc, out = fx.run(["--apply", "--json", "--authorized-key", two_lines])
        payload = json.loads(out)
        assert_eq(rc, 1, "refuses with exit 1")
        assert_true(any("one line" in r for r in payload["refusals"]),
                    "refusal explains the single-line requirement")


def test_rejects_single_line_key_containing_private_key_marker():
    print("\n[test] B3: a single-line value containing 'PRIVATE KEY' is rejected by that check specifically")
    with _Fixture() as fx:
        one_line = "-----BEGIN OPENSSH PRIVATE KEY----- FAKEFAKEFAKEFAKE -----END OPENSSH PRIVATE KEY-----"
        rc, out = fx.run(["--apply", "--json", "--authorized-key", one_line])
        payload = json.loads(out)
        assert_eq(rc, 1, "refuses with exit 1")
        assert_true(any("PRIVATE" in r.upper() for r in payload["refusals"]),
                    "refusal explains this looks like a private key")
        assert_not_in("FAKEFAKEFAKEFAKE", out, "the key's own content never appears in output")


def test_rejects_options_prefixed_key():
    print("\n[test] B3: an options-prefixed key line is rejected (options are not supported)")
    with _Fixture() as fx:
        prefixed = 'command="/bin/bash" ' + _ED25519_KEY_A
        rc, out = fx.run(["--apply", "--json", "--authorized-key", prefixed])
        payload = json.loads(out)
        assert_eq(rc, 1, "refuses with exit 1")
        assert_true(any("known public-key type token" in r for r in payload["refusals"]),
                    "refusal names the missing type token")


def test_keys_added_reports_fingerprints_not_key_text():
    print("\n[test] B3: keys_added in --json holds fingerprints, never the key line itself")
    with _Fixture() as fx:
        rc, out = fx.run(["--apply", "--json", "--authorized-key", _ED25519_KEY_A])
        payload = json.loads(out)
        assert_eq(rc, 0, "--apply succeeds")
        added = payload["keys_added"]
        assert_eq(len(added), 1, "one key added")
        assert_true("SHA256:" in added[0], "the reported entry looks like an ssh-keygen fingerprint")
        assert_not_in(_ED25519_KEY_A, out, "the raw key line is not in the JSON output")
        assert_not_in(_ED25519_KEY_A.split()[1], out,
                      "the key's own base64 material is not in the JSON output")


# ---------------------------------------------------------------------------
# 6. Never apt-get upgrade / dist-upgrade — and the upgrade path is
#    actually exercised, so a mutant that swaps --only-upgrade for a bare
#    `apt-get upgrade` cannot hide behind a test that never takes this path.
# ---------------------------------------------------------------------------

def test_never_calls_apt_get_upgrade_or_dist_upgrade():
    print("\n[test] the script never invokes apt-get upgrade or dist-upgrade, across install AND upgrade")
    with _Fixture() as fx:
        fx.run(["--authorized-key", _ED25519_KEY_A])  # dry run: install
        rc1, out1 = fx.run(["--apply", "--authorized-key", _ED25519_KEY_A])  # apply: install
        assert_eq(rc1, 0, "first --apply (install) succeeds")

        # Bump the fake apt candidate so the NEXT apply sees
        # installed < candidate -> action="upgrade", genuinely exercising
        # the --only-upgrade code path (M5: this used to be untestable,
        # which is exactly why a mutant swapping in a bare `apt-get
        # upgrade` survived).
        os.environ["FAKE_APT_CANDIDATE"] = "1:9.9p2-1ubuntu1"
        rc2, out2 = fx.run(["--apply", "--json", "--authorized-key", _ED25519_KEY_A])
        payload2 = json.loads(out2)
        assert_eq(payload2["action"], "upgrade", "the bumped candidate is actually seen as an upgrade")
        assert_eq(rc2, 0, "the upgrade apply succeeds")

        log = fx.cmd_log_text()
        saw_only_upgrade = False
        for line in log.splitlines():
            if not line.startswith("apt-get "):
                continue
            tokens = line.split()[1:]
            assert_true("upgrade" not in tokens, f"'upgrade' as a bare apt-get subcommand: {line!r}")
            assert_true("dist-upgrade" not in tokens, f"'dist-upgrade' as a bare apt-get subcommand: {line!r}")
            if "--only-upgrade" in tokens:
                saw_only_upgrade = True
        assert_true(saw_only_upgrade, "the --only-upgrade code path really ran at least once")


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

        # M8: a missing supervisord conf must also come back as clean
        # --json, never an unhandled exception.
        fx.supervisor_conf.unlink()
        rc4, out4 = fx.run(["--apply", "--json", "--supervise", "--authorized-key", _ED25519_KEY_A])
        payload4 = json.loads(out4)
        assert_true(_REQUIRED_JSON_KEYS.issubset(payload4.keys()),
                    f"missing-supervisor-conf JSON missing keys: {_REQUIRED_JSON_KEYS - payload4.keys()}")
        assert_eq(rc4, 1, "a missing supervisord.conf refuses cleanly with exit 1, not a crash")


# ---------------------------------------------------------------------------
# Extra coverage: --port, --no-persist-host-keys, --supervise, the
# container-check refusal and its --force override.
# ---------------------------------------------------------------------------

def test_custom_port_written_and_reported():
    print("\n[test] --port writes a Port directive and is reported back, and sshd listens ONLY on it")
    with _Fixture() as fx:
        rc, out = fx.run(["--apply", "--json", "--port", "2222", "--authorized-key", _ED25519_KEY_A])
        payload = json.loads(out)
        assert_eq(rc, 0, "--apply with a custom port succeeds")
        assert_eq(payload["port"], 2222, "port reported back")
        dropin = (fx.etc_ssh_d / "00-zions.conf").read_text()
        assert_in("Port 2222", dropin, "the drop-in carries the custom port")

        # M8: verify the REAL, POST-write effective config carries ONLY
        # the requested port — not both 22 and 2222.
        ok, effective, detail = _script._sshd_effective_config(2222)
        assert_true(ok, f"effective config check passes ({detail})")
        assert_eq(sorted(set(effective.get("port", []))), ["2222"],
                  "sshd -T reports exactly one active port: the requested one")
        assert_true(_script._sshd_listening_on_port(2222), "really listening on 2222")


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


def test_make_backup_resolves_a_same_second_collision_instead_of_crashing():
    print("\n[test] M8: _make_backup resolves a same-second stamp collision with a suffix, never an unhandled crash")
    with _Fixture() as fx:
        target = fx.root / "some-file.txt"
        target.write_text("original", encoding="utf-8")
        with patch.object(_script, "_utc_stamp", return_value="20260101T000000Z"):
            b1 = _script._make_backup(target)
            assert_true(b1.is_file(), "first backup created")
            b2 = _script._make_backup(target)
            assert_true(b2.is_file(), "second backup at the SAME stamp also succeeds")
            assert_true(b1 != b2, "the second backup got a distinct path (a numeric suffix), not a collision")
            assert_eq(b1.read_text(), "original", "the first backup is untouched by the second call")
            assert_eq(b2.read_text(), "original", "the second backup also holds the right content")

            # Real-world scenario this matters for: the SAME call site
            # hit twice in one script run within the same second (e.g. a
            # backup-then-rollback-then-backup-again sequence) must never
            # raise an unhandled RuntimeError with no --json output.
            b3 = _script._make_backup(target)
            assert_true(b3 not in (b1, b2), "a third same-second backup also gets its own distinct path")


def test_force_does_not_override_the_root_check():
    print("\n[test] --force overrides ONLY the container check, never the root check (module docstring: 'no --force override')")
    with _Fixture() as fx:
        rc, out = fx.run(["--apply", "--json", "--force", "--authorized-key", _ED25519_KEY_A], euid=1000)
        payload = json.loads(out)
        assert_eq(rc, 1, "still refuses with exit 1 even with --force")
        assert_true(any("root" in r for r in payload["refusals"]), "refusal still names root")


def test_append_keys_directly_sets_dir_and_file_permissions():
    print("\n[test] _append_keys itself sets 0700/0600 (not only the later, redundant _harden_ssh_dir_permissions call)")
    with _Fixture() as fx:
        ok, fp = _script._validate_key(_ED25519_KEY_A)
        assert_true(ok, "sanity: the test key validates")
        added, backup, created_fresh = _script._append_keys([(_ED25519_KEY_A, fp)])
        assert_eq(len(added), 1, "one fingerprint returned")
        assert_eq(oct(_script.ROOT_SSH_DIR.stat().st_mode & 0o777), "0o700",
                  "_append_keys itself leaves .ssh at 0700")
        assert_eq(oct(_script.AUTHORIZED_KEYS.stat().st_mode & 0o777), "0o600",
                  "_append_keys itself leaves authorized_keys at 0600")


def test_host_keys_are_not_regenerated_on_a_second_apply():
    print("\n[test] host key CONTENT is stable across two applies — never silently regenerated")
    with _Fixture() as fx:
        rc1, _ = fx.run(["--apply", "--authorized-key", _ED25519_KEY_A])
        assert_eq(rc1, 0, "first --apply succeeds")
        before = {
            t: (fx.data_ssh / f"ssh_host_{t}_key").read_bytes()
            for t in ("rsa", "ecdsa", "ed25519")
        }
        rc2, _ = fx.run(["--apply", "--authorized-key", _ED25519_KEY_A])
        assert_eq(rc2, 0, "second --apply succeeds")
        after = {
            t: (fx.data_ssh / f"ssh_host_{t}_key").read_bytes()
            for t in ("rsa", "ecdsa", "ed25519")
        }
        assert_eq(before, after, "host key bytes are byte-identical across the idempotent re-apply")


def test_host_keys_in_etc_are_not_regenerated_across_applies_without_persist():
    print("\n[test] M8: without persistence, host keys already in /etc/ssh are still not regenerated on a second apply")
    with _Fixture() as fx:
        # --no-persist-host-keys deliberately bypasses the "restore from
        # /data/ssh" shortcut, so this exercises the OTHER branch of
        # _ensure_host_keys — the one the M8 mutant (regenerate every
        # run) actually targets and which the default-persist test
        # above never reaches (it returns via the persisted-store
        # restore path before that code runs at all).
        rc1, _ = fx.run(["--apply", "--no-persist-host-keys", "--authorized-key", _ED25519_KEY_A])
        assert_eq(rc1, 0, "first --apply succeeds")
        before = {
            t: (fx.etc_ssh / f"ssh_host_{t}_key").read_bytes()
            for t in ("rsa", "ecdsa", "ed25519")
        }
        rc2, _ = fx.run(["--apply", "--no-persist-host-keys", "--authorized-key", _ED25519_KEY_A])
        assert_eq(rc2, 0, "second --apply succeeds")
        after = {
            t: (fx.etc_ssh / f"ssh_host_{t}_key").read_bytes()
            for t in ("rsa", "ecdsa", "ed25519")
        }
        assert_eq(before, after, "host key bytes in /etc/ssh are stable across two applies, even without persistence")


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
        assert_eq(payload["daemon"], "started", "daemon reported started under supervisord")
        assert_true(_script._sshd_listening_on_port(int(_DEFAULT_TEST_PORT)),
                    "really listening under supervisord (H6)")

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
            assert_true(not fx.root_home.joinpath(".ssh", "authorized_keys").exists(),
                        "authorized_keys (freshly created this run) was rolled back too")
        finally:
            os.environ.pop("FAKE_SUPERVISORCTL_REREAD_FAIL", None)


def test_supervise_missing_conf_file_refuses_cleanly_not_a_crash():
    print("\n[test] M8: --supervise against a MISSING supervisord.conf refuses cleanly, no exception")
    with _Fixture() as fx:
        fx.supervisor_conf.unlink()
        rc, out = fx.run([
            "--apply", "--json", "--supervise", "--authorized-key", _ED25519_KEY_A,
        ])
        payload = json.loads(out)
        assert_eq(rc, 1, "refuses with exit 1, not an unhandled exception")
        assert_true(any("does not exist" in r for r in payload["refusals"]),
                    "refusal names the missing conf file")

        # dry run must flag the same problem, not silently claim "ok"
        rc2, out2 = fx.run(["--json", "--supervise"])
        payload2 = json.loads(out2)
        assert_eq(rc2, 1, "a dry run also refuses when --supervise could never work")


# ---------------------------------------------------------------------------
# H5 — never leave the pod with no sshd because of this script.
# ---------------------------------------------------------------------------

def test_supervise_handoff_failure_restarts_the_standalone_daemon():
    print("\n[test] H5: a failed supervisord handoff restarts the ALREADY-RUNNING standalone daemon")
    with _Fixture() as fx:
        # 1) a normal standalone apply — a real daemon is running & listening.
        rc1, out1 = fx.run(["--apply", "--json", "--authorized-key", _ED25519_KEY_A])
        assert_eq(rc1, 0, "the initial standalone --apply succeeds")
        assert_true(_script._sshd_listening_on_port(int(_DEFAULT_TEST_PORT)),
                    "the standalone daemon is listening before the handoff attempt")
        old_pid = int((fx.run_dir / "sshd.pid").read_text().strip())

        # 2) --supervise now, but the handoff (reread) is forced to fail.
        os.environ["FAKE_SUPERVISORCTL_REREAD_FAIL"] = "1"
        try:
            rc2, out2 = fx.run([
                "--apply", "--json", "--supervise", "--authorized-key", _ED25519_KEY_A,
            ])
            payload2 = json.loads(out2)
            assert_eq(rc2, 1, "the handoff failure is a refusal, exit 1")
        finally:
            os.environ.pop("FAKE_SUPERVISORCTL_REREAD_FAIL", None)

        # 3) H5: sshd must be running and LISTENING again afterward — the
        # pod is never left with no sshd because THIS script stopped it.
        assert_true(_script._sshd_listening_on_port(int(_DEFAULT_TEST_PORT)),
                    "sshd is listening again after the failed handoff was recovered from")
        new_pid = int((fx.run_dir / "sshd.pid").read_text().strip())
        assert_true(Path(f"/proc/{new_pid}").is_dir(), "the recovered sshd process is really alive")
        assert_eq(payload2["daemon"], "restarted",
                  "the report reflects the standalone-restart recovery, not a silent 'not-started'")


# ---------------------------------------------------------------------------
# H6 — "started" is decided by real LISTENING, never sshd's own exit status.
# ---------------------------------------------------------------------------

def test_apply_refuses_when_sshd_reports_success_but_never_binds():
    print("\n[test] H6: sshd exits 0 but never actually binds the port -> --apply refuses, not exit 0")
    with _Fixture() as fx:
        os.environ["FAKE_SSHD_SKIP_BIND"] = "1"
        try:
            rc, out = fx.run(["--apply", "--json", "--authorized-key", _ED25519_KEY_A])
            payload = json.loads(out)
            assert_eq(rc, 1, "refuses with exit 1 despite sshd's own exit status being 0")
            assert_true(any("LISTENING" in r for r in payload["refusals"]),
                        "refusal explicitly names the real listening check, not sshd's exit status")
            assert_eq(payload["daemon"], "not-started", "daemon NOT reported started")
            # Config and the key append are rolled back. Host keys are the
            # one documented exception (see ROLLBACK in the module
            # docstring) — they are allowed to persist.
            assert_true(not (fx.etc_ssh_d / "00-zions.conf").exists(),
                        "the drop-in config write was rolled back")
            assert_true(not fx.root_home.joinpath(".ssh", "authorized_keys").exists(),
                        "authorized_keys (freshly created this run) was rolled back")
            # No dead-end sshd process/pidfile is left behind either — a
            # daemon that came up but never bound the port is worse than
            # none, so it is stopped, not left running uselessly.
            assert_true(not (fx.run_dir / "sshd.pid").is_file(),
                        "no orphaned pidfile from the non-listening daemon attempt")
        finally:
            os.environ.pop("FAKE_SSHD_SKIP_BIND", None)


# ---------------------------------------------------------------------------
# H7 — /root/.ssh and authorized_keys permissions enforced EVERY apply.
# ---------------------------------------------------------------------------

def test_permissions_enforced_even_with_no_new_keys_to_add():
    print("\n[test] H7: a pre-existing world-writable .ssh/authorized_keys is corrected even with 0 new keys")
    with _Fixture() as fx:
        ssh_dir = fx.root_home / ".ssh"
        ssh_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(ssh_dir, 0o777)
        ak = ssh_dir / "authorized_keys"
        ak.write_text(_ED25519_KEY_A + "\n", encoding="utf-8")
        os.chmod(ak, 0o666)

        rc, out = fx.run(["--apply", "--json"])  # no --authorized-key: falls back to the existing file
        payload = json.loads(out)
        assert_eq(rc, 0, "--apply succeeds")
        assert_eq(payload["key_source"], "existing-authorized_keys", "used the pre-existing file")
        assert_eq(payload["keys_added"], [], "nothing new was added")
        assert_eq(oct(ssh_dir.stat().st_mode & 0o777), "0o700",
                  ".ssh was corrected to 0700 despite zero new keys")
        assert_eq(oct(ak.stat().st_mode & 0o777), "0o600",
                  "authorized_keys was corrected to 0600 despite zero new keys")


# ---------------------------------------------------------------------------
# M5 — fallback / direct-edit path coverage, and the real DROPIN_PATH
# constant's "00-" sort-order prefix.
# ---------------------------------------------------------------------------

def test_apply_uses_fallback_when_dropin_not_usable():
    print("\n[test] M5: the fallback direct-edit path (_dropin_usable False) is exercised end to end")
    with _Fixture() as fx:
        # No 'Include .../sshd_config.d/*.conf' line at all -> drop-ins are
        # never loaded by real sshd, so _dropin_usable must be False.
        fx.sshd_config.write_text(
            "# This is the sshd server system-wide configuration file.\n"
            "PasswordAuthentication yes\n"
            "UsePAM yes\n",
            encoding="utf-8",
        )
        rc, out = fx.run(["--apply", "--json", "--authorized-key", _ED25519_KEY_A])
        payload = json.loads(out)
        assert_eq(rc, 0, f"--apply succeeds via the fallback path (refusals={payload.get('refusals')})")
        assert_eq(payload["config_mode"], "fallback", "fallback mode was really chosen")
        assert_eq(payload["config_path"], str(fx.sshd_config), "fallback edits sshd_config directly")

        text = fx.sshd_config.read_text()
        assert_in(_script.BEGIN_MARK, text, "this script's own block was prepended")
        assert_in("PasswordAuthentication no", text, "the hardening directive is active")
        assert_in("# (superseded by scripts/setup-sshd.py) PasswordAuthentication yes", text,
                   "the old conflicting directive was commented out, not deleted")
        assert_true("password authentication is disabled" in payload["sshd_check"],
                    "sshd -T verification really passed against the fallback-edited file")


def test_apply_uses_fallback_when_include_after_active_directive():
    print("\n[test] M5: Include present but AFTER an active managed directive still forces fallback")
    with _Fixture() as fx:
        fx.sshd_config.write_text(
            "PasswordAuthentication yes\n"
            f"Include {fx.etc_ssh_d}/*.conf\n",
            encoding="utf-8",
        )
        rc, out = fx.run(["--apply", "--json", "--authorized-key", _ED25519_KEY_A])
        payload = json.loads(out)
        assert_eq(rc, 0, f"--apply succeeds (refusals={payload.get('refusals')})")
        assert_eq(payload["config_mode"], "fallback",
                  "fallback chosen because the active directive precedes the Include")


def test_comment_out_conflicts_reaches_inside_match_blocks():
    print("\n[test] M5: _comment_out_conflicts (the fallback path's own writer) neutralizes directives even inside a Match block")
    text = (
        "PubkeyAuthentication yes\n"
        "Match Address *\n"
        "    PasswordAuthentication yes\n"
        "    PermitRootLogin yes\n"
    )
    out = _script._comment_out_conflicts(text)
    assert_not_in("\nPasswordAuthentication yes\n", "\n" + out,
                  "the directive inside the Match block is no longer active")
    assert_in("# (superseded by scripts/setup-sshd.py) PasswordAuthentication yes", out,
              "it was commented out, not deleted")
    assert_in("# (superseded by scripts/setup-sshd.py) PermitRootLogin yes", out,
              "PermitRootLogin inside the Match block was neutralized too")


def test_dropin_path_uses_00_prefix_for_sort_order():
    print("\n[test] M5: the REAL, unpatched DROPIN_PATH constant sorts first among typical drop-ins")
    # Deliberately OUTSIDE any _Fixture() — this reads the module's own,
    # never-patched default, so a change to the literal "00-zions.conf"
    # (which is what makes this script's drop-in win first-match-wins
    # against a typical operator/template drop-in) cannot go unnoticed.
    assert_eq(_script.DROPIN_PATH.name, "00-zions.conf", "module-level DROPIN_PATH filename")
    assert_eq(_script.DROPIN_PATH.parent, _script.SSHD_CONFIG_D, "DROPIN_PATH lives under SSHD_CONFIG_D")
    assert_true("00-zions.conf" < "50-anything.conf", "sanity: our prefix sorts before a typical operator drop-in")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_dry_run_writes_nothing_and_installs_nothing()
    test_dry_run_reports_planned_action_and_key()

    test_apply_on_clean_container_installs_keys_and_starts()
    test_apply_never_reads_or_prints_a_private_key()

    test_apply_twice_is_idempotent_and_second_run_exits_0()
    test_dry_run_after_apply_finds_nothing_to_do_and_exits_0()

    test_existing_authorized_keys_preserved_and_appended_not_clobbered()
    test_duplicate_key_is_not_added_twice()

    test_refuses_when_not_root()
    test_refuses_when_no_key_available_anywhere()
    test_refuses_on_malformed_public_key()
    test_refuses_when_effective_config_would_allow_password_auth()
    test_refuses_when_sshd_t_fails()
    test_refuses_when_public_context_would_allow_password_auth()

    test_refuses_outright_on_match_block_in_main_config()
    test_refuses_outright_on_match_block_in_other_dropin()
    test_refuses_outright_on_earlier_sorting_dropin_without_match()
    test_does_not_refuse_on_later_sorting_dropin()
    test_refuses_outright_on_conflicting_port_in_other_dropin()

    test_rejects_private_key_and_never_leaks_it()
    test_rejects_multiline_key_material()
    test_rejects_single_line_key_containing_private_key_marker()
    test_rejects_options_prefixed_key()
    test_keys_added_reports_fingerprints_not_key_text()

    test_never_calls_apt_get_upgrade_or_dist_upgrade()

    test_no_private_key_material_in_any_mode()

    test_key_source_precedence()

    test_json_output_has_the_documented_keys_in_every_mode()

    test_custom_port_written_and_reported()
    test_no_persist_host_keys_leaves_data_ssh_empty()
    test_container_check_refuses_without_force_and_overridden_with_force()
    test_make_backup_resolves_a_same_second_collision_instead_of_crashing()
    test_force_does_not_override_the_root_check()
    test_append_keys_directly_sets_dir_and_file_permissions()
    test_host_keys_are_not_regenerated_on_a_second_apply()
    test_host_keys_in_etc_are_not_regenerated_across_applies_without_persist()
    test_supervise_appends_program_and_never_touches_other_programs()
    test_supervise_reread_failure_restores_backup_and_refuses()
    test_supervise_missing_conf_file_refuses_cleanly_not_a_crash()

    test_supervise_handoff_failure_restarts_the_standalone_daemon()

    test_apply_refuses_when_sshd_reports_success_but_never_binds()

    test_permissions_enforced_even_with_no_new_keys_to_add()

    test_apply_uses_fallback_when_dropin_not_usable()
    test_apply_uses_fallback_when_include_after_active_directive()
    test_comment_out_conflicts_reaches_inside_match_blocks()
    test_dropin_path_uses_00_prefix_for_sort_order()

    print("\nAll setup-sshd.py script tests passed.")
