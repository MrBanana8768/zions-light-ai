#!/usr/bin/env python3
"""Install (or upgrade) an OpenSSH server inside the RUNNING production
container, and bring it up hardened, without touching the image.

    /opt/compactor-venv/bin/python /data/scripts/setup-sshd.py
    /opt/compactor-venv/bin/python /data/scripts/setup-sshd.py --json
    /opt/compactor-venv/bin/python /data/scripts/setup-sshd.py --apply
    /opt/compactor-venv/bin/python /data/scripts/setup-sshd.py --apply --port 2222

WHY THIS EXISTS. The image (Dockerfile:63-80) installs no openssh-server —
never has. `entrypoint.sh` never mentions ssh, and `supervisord.conf` has
no `[sshd]` program and no `[include]` section (so a `.conf` dropped into
`/etc/supervisor/conf.d/` on its own is silently ignored by the running
supervisord — confirmed against the shipped file, not assumed). The image
must stay exactly what v3.1.9.4 validated (v3.1.9.6 is a retag of that
digest), so this script is the ONLY way an operator gets a real shell
into a running pod: it installs and hardens `openssh-server` live, inside
the container's writable overlay.

THE CONTAINER FILESYSTEM RESETS ON EVERY RESTART. Everything this script
does to `/etc/ssh` and (with `--supervise`) `/etc/supervisor/conf.d/` lives
on the container's own overlay, not the `/data` network volume. Re-run
this after every pod restart — it is idempotent and safe to run on a pod
that already has sshd running.

WHAT --apply DOES, IN ORDER
    1. `apt-get update` (package lists are pruned from the image — Dockerfile
       runs `rm -rf /var/lib/apt/lists/*` — so this is required every time).
    2. Install `openssh-server` if missing, or `--only-upgrade` it to the
       apt candidate if already present. NEVER `apt-get upgrade` or
       `dist-upgrade` — see the module-level safety notes below.
    3. Resolve one usable Ed25519/RSA/ECDSA PUBLIC key (never a private
       one) from, in order: `--authorized-key-file`, `--authorized-key`,
       the RunPod-injected `$PUBLIC_KEY`, or an existing non-empty
       `/root/.ssh/authorized_keys`. Refuses outright if none is usable.
    4. Write the hardening directives — see "THE HARD SAFETY RULES" below
       — to a drop-in under `/etc/ssh/sshd_config.d/`, or fall back to
       editing `sshd_config` directly with a backup, WHICHEVER the real
       config on disk actually supports (checked fresh every run: see
       `_dropin_usable`, and OPERATIONS.md for what this pod's shipped
       config was found to be).
    5. Ensure host keys exist (`ssh-keygen -A`), and by default persist
       them to `/data/ssh/` so the pod keeps the same host identity across
       restarts — see "HOST KEYS" below.
    6. Validate the result for REAL with `sshd -t` then `sshd -T` (never
       just by reading back the file this script itself wrote) and REFUSE,
       rolling back, if the effective config would allow password login.
    7. Start (or, if already running and the binary/config changed,
       restart — by pid, never a blanket pkill) `/usr/sbin/sshd` as a
       plain background daemon. `--supervise` (opt-in, OFF by default)
       instead adds it as a supervisord program — see that flag's own
       section below for why it defaults off.

DRY RUN IS THE DEFAULT (no `--apply`). It runs a REAL `apt-get update`
(package lists only — nothing is installed, and the report says so) and a
REAL `apt-get install -s` (`-s` = simulate) so the reported action is
accurate, but writes NOTHING to `/etc/ssh`, `/root/.ssh`, `/data/ssh` or
`/etc/supervisor/conf.d/`, installs nothing, and starts nothing. If
openssh-server is not installed yet, `sshd -t`/`sshd -T` cannot run (the
binary does not exist), so the dry run says so honestly instead of
pretending to have verified the resulting config — that verification is
real and happens automatically during `--apply` (step 6 above), which
refuses if it does not pass.

THE HARD SAFETY RULES (see also inline comments at each site)
  - Never `apt-get upgrade`/`dist-upgrade` — only ever `apt-get install
    [--no-install-recommends|--only-upgrade] -y openssh-server -o
    Dpkg::Options::=--force-confold openssh-server`, so an operator's own
    `sshd_config` (or, once this script has run once, its own drop-in)
    never gets silently clobbered by a package-shipped default, and no
    unrelated package on the pod is ever touched while vLLM is running.
  - Never print, log, copy or otherwise handle a PRIVATE key. Only ever a
    PUBLIC key (validated with `ssh-keygen -l -f`) goes into
    `authorized_keys`. Host PRIVATE keys are created by `ssh-keygen -A`
    (never generated or read by this script's own code) and this script
    never reads their contents, only their existence, size and mtime.
  - Never enable password login. The directives this script writes are,
    at minimum: `PasswordAuthentication no`, `PermitEmptyPasswords no`,
    `KbdInteractiveAuthentication no`, `ChallengeResponseAuthentication
    no`, `PubkeyAuthentication yes`, `PermitRootLogin prohibit-password`.
    `sshd -T` on this image's own OpenSSH 9.6 build always PRINTS
    `permitrootlogin` back as `without-password` regardless of which of
    the two synonymous spellings was written — both are accepted when
    checking the real outcome; nothing else is.
  - Never touch any other supervisord program. The default start path
    (plain `/usr/sbin/sshd`) never goes near supervisord at all; even
    `--supervise`'s `supervisorctl update` only starts the ONE newly added
    program (verified against the real image — see OPERATIONS.md).
  - Never restart anything by killing broadly. The daemon is found by its
    own pidfile (`/run/sshd.pid`, confirmed to be `sshd` via
    `/proc/<pid>/comm`) or not touched at all.

HOST KEYS. Default: persist to `/data/ssh/` (0700, key files 0600) and
copy them into `/etc/ssh/` on every run, so the pod's host key fingerprint
survives a restart instead of changing every time (which trains an
operator to click through host-key warnings — worse than the storage
risk). `--no-persist-host-keys` opts out; the keys `ssh-keygen -A` creates
then live only in the container's overlay and are lost on the next
restart. `compactor/backup.py`'s archive is built from exactly two things
— OpenWebUI's `webui.db` snapshot and `COMPACTOR_STORAGE_ROOT`
(`/data/openwebui/compactor` by default) — never a walk of `/data` itself,
so `/data/ssh/` is NOT swept into it. Confirmed by reading
`compactor/backup.py`'s `create_backup`, not assumed; see OPERATIONS.md.

--SUPERVISE (opt-in, OFF by default). Appends a `[program:sshd]` block to
`/etc/supervisor/conf.d/supervisord.conf` (backed up first; the append is
validated with `supervisorctl reread`, and the backup is restored and the
run refused if that fails) and runs `supervisorctl update`. Verified
against a throwaway container running this image's real supervisord (no
GPU — vllm alone goes FATAL, everything else comes up): adding ONE new
`[program:sshd]` section and running `reread` + `update` started only
that new program and left every other program's own RUNNING state alone —
`update` only (re)starts a program whose OWN section changed or is new,
never one supervisord already has running unchanged. Kept, on that
evidence; see OPERATIONS.md for the transcript.

EXIT CODES. The same convention as scripts/backfill-records.py and
scripts/import-history.py — normalized across all three (see
CHANGELOG.md and OPERATIONS.md). Defect 4 / hostile-pass finding: THIS
SCRIPT USED TO HAVE 0 AND 3 SWAPPED FROM THIS CONVENTION (its own author
flagged it) — three sibling scripts sharing the same flags with opposite
exit meanings is exactly the kind of thing that burns an operator writing
`if script; then`.
    0   success — the desired end state is in place. Either `--apply`
        completed (installed/upgraded, added a key, (re)started the
        daemon, or was a NO-OP because everything was already correct —
        including a second `--apply` right after the first), or a DRY RUN
        found nothing to do (openssh-server already at the apt candidate
        version, no new key to add, and the daemon already running with
        this script's own config unchanged)
    1   a refusal the operator needs to look at (see `refusals` in
        `--json` output): not root, does not look like this container,
        no usable public key anywhere, a malformed key, `apt-get install`
        failed, `sshd -t`/`sshd -T` failed or would allow password login,
        `sshd` failed to (re)start, or `--supervise`'s `reread` failed
    3   a DRY RUN found something `--apply` WOULD do (install/upgrade, add
        a key, or start/restart the daemon) — informational, not a
        failure: re-run with `--apply` once ready
    (argparse's own usage errors — unknown flags, missing required values
    — exit 2, the Python standard library's own convention)
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Injectable roots. Defaults are the real pod paths; every one is
# overridable via an env var so compactor/test_setup_sshd_script.py can
# point the whole script at temp directories without touching a real
# system — the same discipline scripts/switch-webui-db-to-local.py already
# uses for SUPERVISOR_CONF/ENTRYPOINT_SH. An operator never needs to set
# any of these.
# ---------------------------------------------------------------------------

ETC_SSH_DIR = Path(os.environ.get("SETUP_SSHD_ETC_SSH_DIR", "/etc/ssh"))
SSHD_CONFIG = ETC_SSH_DIR / "sshd_config"
SSHD_CONFIG_D = ETC_SSH_DIR / "sshd_config.d"
DROPIN_PATH = SSHD_CONFIG_D / "00-zions.conf"

ROOT_HOME_DIR = Path(os.environ.get("SETUP_SSHD_ROOT_HOME", "/root"))
ROOT_SSH_DIR = ROOT_HOME_DIR / ".ssh"
AUTHORIZED_KEYS = ROOT_SSH_DIR / "authorized_keys"

HOST_KEY_STORE = Path(os.environ.get("SETUP_SSHD_HOST_KEY_STORE", "/data/ssh"))

SUPERVISOR_CONF = Path(os.environ.get(
    "SETUP_SSHD_SUPERVISOR_CONF", "/etc/supervisor/conf.d/supervisord.conf"
))

RUN_DIR = Path(os.environ.get("SETUP_SSHD_RUN_DIR", "/run"))
SSHD_PIDFILE = RUN_DIR / "sshd.pid"
PRIVSEP_DIR = RUN_DIR / "sshd"

MARKER_VENV = Path(os.environ.get("SETUP_SSHD_MARKER_VENV", "/opt/compactor-venv"))
MARKER_MAIN = Path(os.environ.get("SETUP_SSHD_MARKER_MAIN", "/opt/compactor/main.py"))

HOST_KEY_TYPES = ("rsa", "ecdsa", "ed25519")
DEFAULT_LOG_DIR = os.environ.get("LOG_DIR", "/data/logs")

CMD_TIMEOUT_S = 90

DIRECTIVE_KEYS = (
    "PasswordAuthentication",
    "PermitEmptyPasswords",
    "KbdInteractiveAuthentication",
    "ChallengeResponseAuthentication",
    "PubkeyAuthentication",
    "PermitRootLogin",
)

INCLUDE_RE = re.compile(
    r"^[ \t]*Include[ \t]+\S*sshd_config\.d\S*\*\.conf[ \t]*$", re.I | re.M
)

BEGIN_MARK = "# --- BEGIN scripts/setup-sshd.py hardening (see OPERATIONS.md) ---"
END_MARK = "# --- END scripts/setup-sshd.py hardening ---"

SUP_BEGIN_MARK = "; --- BEGIN scripts/setup-sshd.py [program:sshd] ---"
SUP_END_MARK = "; --- END scripts/setup-sshd.py [program:sshd] ---"


# ---------------------------------------------------------------------------
# Small pure / patchable helpers (patched by tests the same way
# scripts/backfill-records.py's own _utc_stamp is patched)
# ---------------------------------------------------------------------------

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _utc_stamp() -> str:
    return _now_utc().strftime("%Y%m%dT%H%M%SZ")


def _geteuid() -> int:
    return os.geteuid()


def _run(args, timeout: int = CMD_TIMEOUT_S) -> subprocess.CompletedProcess:
    """The one place every external command runs through. Never raises —
    a missing binary or a timeout comes back as a synthetic non-zero
    CompletedProcess so callers have one shape to handle."""
    try:
        return subprocess.run(
            args, capture_output=True, text=True, timeout=timeout
        )
    except FileNotFoundError as e:
        return subprocess.CompletedProcess(args, 127, "", f"{type(e).__name__}: {e}")
    except subprocess.TimeoutExpired as e:
        return subprocess.CompletedProcess(
            args, 124, e.stdout or "", f"timed out after {timeout}s"
        )


def _make_backup(path: Path) -> Path:
    """Back up `path` beside itself as `<name>.bak-<UTC stamp>`, refusing
    outright if that exact backup path already exists — same convention
    scripts/backfill-records.py and scripts/import-history.py use, so a
    backup is never silently overwritten."""
    stamp = _utc_stamp()
    backup = path.with_name(path.name + f".bak-{stamp}")
    if backup.exists():
        raise RuntimeError(f"refused — backup path already exists: {backup}")
    shutil.copy2(path, backup)
    return backup


# ---------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------

def _is_root() -> bool:
    return _geteuid() == 0


def _looks_like_zions_container() -> bool:
    return MARKER_VENV.is_dir() and MARKER_MAIN.is_file()


# ---------------------------------------------------------------------------
# apt — install/upgrade openssh-server, NEVER apt-get upgrade/dist-upgrade
# ---------------------------------------------------------------------------

def _apt_update() -> tuple[bool, str]:
    r = _run(["apt-get", "update"])
    return r.returncode == 0, (r.stdout + r.stderr)


def _apt_cache_policy(pkg: str = "openssh-server") -> tuple[str | None, str | None, str]:
    r = _run(["apt-cache", "policy", pkg])
    text = r.stdout
    installed = None
    candidate = None
    m = re.search(r"Installed:\s*(\S+)", text)
    if m and m.group(1) != "(none)":
        installed = m.group(1)
    m = re.search(r"Candidate:\s*(\S+)", text)
    if m and m.group(1) != "(none)":
        candidate = m.group(1)
    return installed, candidate, text


def _plan_package_action(installed: str | None, candidate: str | None) -> str:
    if candidate is None:
        return "unavailable"
    if installed is None:
        return "install"
    if installed == candidate:
        return "already-current"
    return "upgrade"


def _apt_install_args(action: str) -> list[str]:
    """NEVER 'upgrade' or 'dist-upgrade' — only ever a targeted install of
    this one package, --only-upgrade when it is already present. See the
    module docstring's hard safety rule."""
    if action == "upgrade":
        args = ["install", "--only-upgrade", "-y"]
    else:
        args = ["install", "--no-install-recommends", "-y"]
    args += ["-o", "Dpkg::Options::=--force-confold", "openssh-server"]
    return args


# ---------------------------------------------------------------------------
# Public key resolution — file > literal > $PUBLIC_KEY > existing file.
# Every candidate is validated with `ssh-keygen -l -f`; never a private key.
# ---------------------------------------------------------------------------

def _read_key_lines(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out


def _validate_key(line: str) -> tuple[bool, str]:
    """True + a fingerprint line if `ssh-keygen -l -f` parses this as a
    public key. The candidate is written to a throwaway temp file (never
    logged) purely so ssh-keygen has a path to read; nothing about a
    PRIVATE key ever reaches this function."""
    fd, tmp = tempfile.mkstemp(prefix="setup-sshd-keycheck-")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(line + "\n")
        r = _run(["ssh-keygen", "-l", "-f", tmp])
        return r.returncode == 0, (r.stdout or r.stderr).strip()
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _key_identity(line: str) -> str | None:
    """type+base64 only, ignoring the trailing comment — so the same key
    with a different comment is still recognised as already present."""
    parts = line.split()
    if len(parts) >= 2:
        return parts[0] + " " + parts[1]
    return None


def _resolve_key_source(args) -> tuple[str | None, list[str], list[str]]:
    """(source_label, candidate_lines_to_add, refusals). `candidate_lines`
    is the NEW key material found at this source; it is empty (not a
    refusal) when the source is an already-populated authorized_keys with
    nothing new to add."""
    if args.authorized_key_file:
        p = Path(args.authorized_key_file)
        if not p.is_file():
            return None, [], [f"--authorized-key-file {p} does not exist"]
        lines = _read_key_lines(p.read_text(encoding="utf-8", errors="replace"))
        if not lines:
            return None, [], [f"--authorized-key-file {p} has no key lines"]
        for ln in lines:
            ok, detail = _validate_key(ln)
            if not ok:
                return None, [], [
                    f"malformed public key in --authorized-key-file "
                    f"{p}: {detail or 'ssh-keygen could not parse it'}"
                ]
        return "file", lines, []

    if args.authorized_key:
        ln = args.authorized_key.strip()
        ok, detail = _validate_key(ln)
        if not ok:
            return None, [], [
                f"malformed public key from --authorized-key: "
                f"{detail or 'ssh-keygen could not parse it'}"
            ]
        return "literal", [ln], []

    pub_env = os.environ.get("PUBLIC_KEY", "")
    if pub_env.strip():
        lines = _read_key_lines(pub_env)
        if lines:
            for ln in lines:
                ok, detail = _validate_key(ln)
                if not ok:
                    return None, [], [
                        f"malformed public key in $PUBLIC_KEY: "
                        f"{detail or 'ssh-keygen could not parse it'}"
                    ]
            return "env:PUBLIC_KEY", lines, []

    if AUTHORIZED_KEYS.is_file():
        existing = AUTHORIZED_KEYS.read_text(encoding="utf-8", errors="replace")
        if _read_key_lines(existing):
            return "existing-authorized_keys", [], []

    return None, [], [
        "no usable public key found — checked --authorized-key-file, "
        "--authorized-key, $PUBLIC_KEY, and an existing "
        f"{AUTHORIZED_KEYS} — refusing"
    ]


def _append_keys(candidate_lines: list[str]) -> tuple[list[str], str | None]:
    """Append only the lines not already present (by type+base64
    identity), never clobbering the file. Backs the file up first if it
    already exists. Returns (keys_actually_added, backup_path_or_None)."""
    existing_text = (
        AUTHORIZED_KEYS.read_text(encoding="utf-8", errors="replace")
        if AUTHORIZED_KEYS.is_file() else ""
    )
    existing_ids = {_key_identity(l) for l in _read_key_lines(existing_text)}
    to_add = [l for l in candidate_lines if _key_identity(l) not in existing_ids]
    if not to_add:
        return [], None

    backup = None
    if AUTHORIZED_KEYS.is_file():
        backup = _make_backup(AUTHORIZED_KEYS)

    ROOT_SSH_DIR.mkdir(parents=True, exist_ok=True)
    os.chmod(ROOT_SSH_DIR, 0o700)
    with open(AUTHORIZED_KEYS, "a", encoding="utf-8") as f:
        if existing_text and not existing_text.endswith("\n"):
            f.write("\n")
        for l in to_add:
            f.write(l + "\n")
    os.chmod(AUTHORIZED_KEYS, 0o600)
    return to_add, (str(backup) if backup else None)


def _planned_new_keys(candidate_lines: list[str]) -> list[str]:
    """Dry-run equivalent of `_append_keys` that reads but never writes."""
    existing_text = (
        AUTHORIZED_KEYS.read_text(encoding="utf-8", errors="replace")
        if AUTHORIZED_KEYS.is_file() else ""
    )
    existing_ids = {_key_identity(l) for l in _read_key_lines(existing_text)}
    return [l for l in candidate_lines if _key_identity(l) not in existing_ids]


# ---------------------------------------------------------------------------
# sshd_config: drop-in vs. direct edit — decided fresh every run against
# what is REALLY on disk, never assumed.
# ---------------------------------------------------------------------------

def _directive_lines(port: int) -> list[str]:
    lines = [
        "PasswordAuthentication no",
        "PermitEmptyPasswords no",
        "KbdInteractiveAuthentication no",
        "ChallengeResponseAuthentication no",
        "PubkeyAuthentication yes",
        "PermitRootLogin prohibit-password",
    ]
    if port != 22:
        lines.append(f"Port {port}")
    return lines


def _dropin_content(port: int) -> str:
    header = [
        "# Managed by scripts/setup-sshd.py — see OPERATIONS.md.",
        "# Re-running the script overwrites this file; do not hand-edit it.",
    ]
    return "\n".join(header + _directive_lines(port)) + "\n"


def _earliest_active_directive_line(text: str) -> int | None:
    """0-based line number of the EARLIEST active (uncommented) occurrence
    of any directive this script manages, or None. A commented-out
    default (`#PasswordAuthentication yes`) never counts."""
    best = None
    for key in DIRECTIVE_KEYS:
        m = re.search(rf"^[ \t]*{re.escape(key)}\b", text, re.I | re.M)
        if m:
            lineno = text.count("\n", 0, m.start())
            if best is None or lineno < best:
                best = lineno
    return best


def _include_line(text: str) -> int | None:
    m = INCLUDE_RE.search(text)
    if not m:
        return None
    return text.count("\n", 0, m.start())


def _dropin_usable(sshd_config_text: str) -> tuple[bool, str]:
    """True only if sshd_config really does `Include
    .../sshd_config.d/*.conf` AND that Include appears strictly before any
    directive this script manages that is already active — sshd is
    first-match-wins, so a drop-in loaded at the Include line only wins if
    nothing this script cares about was already decided earlier in the
    file. Verified against the real image: see OPERATIONS.md."""
    inc = _include_line(sshd_config_text)
    if inc is None:
        return False, "no 'Include .../sshd_config.d/*.conf' line in sshd_config"
    conflict = _earliest_active_directive_line(sshd_config_text)
    if conflict is not None and conflict < inc:
        return False, (
            f"an active directive this script manages appears at line "
            f"{conflict + 1}, before the Include at line {inc + 1} — "
            f"sshd is first-match-wins, so a drop-in loaded there would "
            f"never override it"
        )
    return True, f"Include found at line {inc + 1}, before any directive this script manages"


def _strip_managed_block(text: str) -> str:
    pattern = re.compile(re.escape(BEGIN_MARK) + r".*?" + re.escape(END_MARK) + r"\n?", re.S)
    return pattern.sub("", text)


def _comment_out_conflicts(text: str) -> str:
    for key in DIRECTIVE_KEYS:
        text = re.sub(
            rf"^([ \t]*){re.escape(key)}\b(.*)$",
            rf"\1# (superseded by scripts/setup-sshd.py) {key}\2",
            text, flags=re.I | re.M,
        )
    return text


def _fallback_edit(sshd_config_text: str, port: int) -> str:
    """Edit sshd_config directly: comment out every active occurrence of a
    directive this script manages (wherever it is), then prepend this
    script's own block at the very top so first-match-wins always
    resolves to it. Idempotent: a prior run's block is replaced, not
    duplicated."""
    text = _strip_managed_block(sshd_config_text)
    text = _comment_out_conflicts(text)
    block = BEGIN_MARK + "\n" + "\n".join(_directive_lines(port)) + "\n" + END_MARK + "\n"
    return block + text


def _plan_config(port: int, apply: bool) -> dict:
    """Decide dropin-vs-fallback and compute the new content, but only
    WRITE it when apply=True. `changed` says whether anything on disk
    would differ from what is there now."""
    if not SSHD_CONFIG.is_file():
        # openssh-server is not installed yet (dry run only — --apply
        # installs it before this is ever called). Nothing to read yet;
        # say so honestly rather than guess which case a not-yet-created
        # file will land in.
        return {
            "mode": "unknown", "changed": False, "backup_path": None,
            "config_path": None,
            "reason": (
                "openssh-server is not installed yet, so sshd_config does "
                "not exist — --apply decides drop-in vs. direct-edit "
                "after installing it, and reports which one it used"
            ),
            "_old_content": None, "_new_content": None,
        }

    sshd_config_text = SSHD_CONFIG.read_text(encoding="utf-8", errors="replace")
    use_dropin, reason = _dropin_usable(sshd_config_text)

    if use_dropin:
        new_content = _dropin_content(port)
        old_content = (
            DROPIN_PATH.read_text(encoding="utf-8", errors="replace")
            if DROPIN_PATH.is_file() else None
        )
        plan = {
            "mode": "dropin", "reason": reason,
            "config_path": str(DROPIN_PATH),
            "changed": old_content != new_content,
            "backup_path": None,
            "_old_content": old_content, "_new_content": new_content,
        }
    else:
        new_text = _fallback_edit(sshd_config_text, port)
        plan = {
            "mode": "fallback", "reason": reason,
            "config_path": str(SSHD_CONFIG),
            "changed": new_text != sshd_config_text,
            "backup_path": None,
            "_old_content": sshd_config_text, "_new_content": new_text,
        }

    if apply and plan["changed"]:
        if plan["mode"] == "dropin":
            SSHD_CONFIG_D.mkdir(parents=True, exist_ok=True)
            if DROPIN_PATH.is_file():
                plan["backup_path"] = str(_make_backup(DROPIN_PATH))
            DROPIN_PATH.write_text(plan["_new_content"], encoding="utf-8")
            os.chmod(DROPIN_PATH, 0o644)
        else:
            plan["backup_path"] = str(_make_backup(SSHD_CONFIG))
            SSHD_CONFIG.write_text(plan["_new_content"], encoding="utf-8")
    return plan


def _rollback_config(plan: dict) -> None:
    """Undo exactly what `_plan_config` wrote, for a refusal discovered
    afterwards (sshd -t / sshd -T)."""
    if plan["mode"] == "dropin":
        if plan["_old_content"] is None:
            try:
                DROPIN_PATH.unlink()
            except FileNotFoundError:
                pass
        else:
            DROPIN_PATH.write_text(plan["_old_content"], encoding="utf-8")
    elif plan["mode"] == "fallback":
        SSHD_CONFIG.write_text(plan["_old_content"], encoding="utf-8")


# ---------------------------------------------------------------------------
# Host keys — ssh-keygen -A only ever creates them; this script never
# generates or reads PRIVATE key bytes itself.
# ---------------------------------------------------------------------------

def _host_key_files(root: Path) -> list[Path]:
    return [root / f"ssh_host_{t}_key" for t in HOST_KEY_TYPES]


def _ssh_keygen_dash_a() -> subprocess.CompletedProcess:
    """`ssh-keygen -A` writes into /etc/ssh by hardcoded default; `-f
    PREFIX` makes it write under PREFIX/etc/ssh/... instead (the same
    mechanism OpenSSH's own packaging uses for a chroot/staging build) —
    used here only so a test can point ETC_SSH_DIR at a temp tree.
    Production's ETC_SSH_DIR really is /etc/ssh, so this is a no-op
    there and the plain, real invocation runs unchanged."""
    if ETC_SSH_DIR == Path("/etc/ssh"):
        return _run(["ssh-keygen", "-A"])
    s = str(ETC_SSH_DIR)
    suffix = "/etc/ssh"
    prefix = s[: -len(suffix)] if s.endswith(suffix) else s
    return _run(["ssh-keygen", "-A", "-f", prefix])


def _ensure_host_keys(persist: bool) -> dict:
    """Restore from /data/ssh if a full set is already persisted there;
    otherwise generate anything missing directly into ETC_SSH_DIR and, if
    persist=True, copy the result out to /data/ssh. Never reads or logs a
    private key's CONTENT — only paths, existence and permissions."""
    notes: list[str] = []
    ETC_SSH_DIR.mkdir(parents=True, exist_ok=True)
    store_keys = _host_key_files(HOST_KEY_STORE)
    etc_keys = _host_key_files(ETC_SSH_DIR)

    def _copy_pair(src_priv: Path, dest_dir: Path) -> None:
        dest_priv = dest_dir / src_priv.name
        shutil.copy2(src_priv, dest_priv)
        os.chmod(dest_priv, 0o600)
        src_pub = src_priv.with_name(src_priv.name + ".pub")
        if src_pub.is_file():
            dest_pub = dest_dir / src_pub.name
            shutil.copy2(src_pub, dest_pub)
            os.chmod(dest_pub, 0o644)

    if persist and all(p.is_file() and p.stat().st_size > 0 for p in store_keys):
        for priv in store_keys:
            _copy_pair(priv, ETC_SSH_DIR)
        notes.append(f"restored host keys from {HOST_KEY_STORE} (stable fingerprint)")
        return {"host_keys": "persisted", "notes": notes}

    if not all(p.is_file() and p.stat().st_size > 0 for p in etc_keys):
        r = _ssh_keygen_dash_a()
        notes.append(
            "generated missing host key(s) with ssh-keygen -A"
            if r.returncode == 0
            else f"ssh-keygen -A reported an error: {(r.stderr or r.stdout).strip()}"
        )

    for p in etc_keys:
        if p.is_file():
            os.chmod(p, 0o600)

    if not persist:
        return {"host_keys": "ephemeral", "notes": notes}

    HOST_KEY_STORE.mkdir(parents=True, exist_ok=True)
    os.chmod(HOST_KEY_STORE, 0o700)
    for priv in etc_keys:
        if priv.is_file() and not (HOST_KEY_STORE / priv.name).exists():
            _copy_pair(priv, HOST_KEY_STORE)
    notes.append(f"persisted host keys to {HOST_KEY_STORE}")
    return {"host_keys": "persisted", "notes": notes}


# ---------------------------------------------------------------------------
# Verify for real — sshd -t then sshd -T, never just reading our own file.
# ---------------------------------------------------------------------------

def _sshd_binary() -> str:
    """The ABSOLUTE path to sshd. Real sshd re-execs itself for privilege
    separation and refuses ('sshd re-exec requires execution with an
    absolute path') if it was invoked via a bare, PATH-resolved argv[0] —
    confirmed against the real image. `shutil.which` still does the PATH
    lookup (so a test's fake bin dir is found the same way), but the
    RESULT handed to subprocess is always absolute."""
    return shutil.which("sshd") or "/usr/sbin/sshd"


def _sshd_effective_config() -> tuple[bool, dict, str]:
    PRIVSEP_DIR.mkdir(parents=True, exist_ok=True)
    sshd_bin = _sshd_binary()

    r = _run([sshd_bin, "-t"])
    if r.returncode != 0:
        return False, {}, f"sshd -t failed: {(r.stderr or r.stdout).strip()}"

    r2 = _run([sshd_bin, "-T"])
    if r2.returncode != 0:
        return False, {}, f"sshd -T failed: {(r2.stderr or r2.stdout).strip()}"

    effective: dict[str, str] = {}
    for line in r2.stdout.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2:
            effective[parts[0].lower()] = parts[1].strip()

    must_equal = {
        "passwordauthentication": "no",
        "permitemptypasswords": "no",
        "kbdinteractiveauthentication": "no",
        "pubkeyauthentication": "yes",
    }
    for key, expected in must_equal.items():
        got = effective.get(key)
        if got is not None and got.lower() != expected:
            return False, effective, (
                f"sshd -T reports {key}={got}, expected {expected} — "
                f"refusing to start sshd with password login reachable"
            )

    # "prohibit-password" and its older synonym "without-password" mean
    # the same thing; this build's sshd -T always prints the latter back
    # (confirmed against the real image) regardless of which was written.
    root_login = effective.get("permitrootlogin")
    if root_login is not None and root_login.lower() not in (
        "prohibit-password", "without-password"
    ):
        return False, effective, (
            f"sshd -T reports PermitRootLogin={root_login}, expected "
            f"prohibit-password — refusing"
        )

    return True, effective, "sshd -T confirms password authentication is disabled"


# ---------------------------------------------------------------------------
# Daemon lifecycle — found by pidfile, never a blanket pkill.
# ---------------------------------------------------------------------------

def _sshd_pid_if_alive() -> int | None:
    if not SSHD_PIDFILE.is_file():
        return None
    try:
        pid = int(SSHD_PIDFILE.read_text().strip())
    except (ValueError, OSError):
        return None
    proc_dir = Path(f"/proc/{pid}")
    if not proc_dir.is_dir():
        return None
    try:
        name = (proc_dir / "comm").read_text().strip()
    except OSError:
        return None
    return pid if name == "sshd" else None


def _start_sshd() -> subprocess.CompletedProcess:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    PRIVSEP_DIR.mkdir(parents=True, exist_ok=True)
    return _run([_sshd_binary(), "-o", f"PidFile={SSHD_PIDFILE}"])


def _stop_sshd(pid: int, timeout_s: float = 5.0) -> bool:
    """SIGTERM to exactly this pid, waited out — never anything broader."""
    try:
        os.kill(pid, 15)
    except ProcessLookupError:
        return True
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if not Path(f"/proc/{pid}").is_dir():
            return True
        time.sleep(0.1)
    return not Path(f"/proc/{pid}").is_dir()


# ---------------------------------------------------------------------------
# --supervise (opt-in, OFF by default)
# ---------------------------------------------------------------------------

def _supervise_block(port: int) -> str:
    log_dir = DEFAULT_LOG_DIR
    return (
        f"\n{SUP_BEGIN_MARK}\n"
        f"[program:sshd]\n"
        f"command=/usr/sbin/sshd -D -e -o PidFile={SSHD_PIDFILE}\n"
        f"autostart=true\n"
        f"autorestart=true\n"
        f"priority=50\n"
        f"startsecs=2\n"
        f"stdout_logfile={log_dir}/sshd.log\n"
        f"stderr_logfile={log_dir}/sshd-error.log\n"
        f"stdout_logfile_maxbytes=10MB\n"
        f"stdout_logfile_backups=2\n"
        f"stderr_logfile_maxbytes=10MB\n"
        f"stderr_logfile_backups=2\n"
        f"{SUP_END_MARK}\n"
    )


def _ensure_supervised(port: int, apply: bool) -> dict:
    text = (
        SUPERVISOR_CONF.read_text(encoding="utf-8", errors="replace")
        if SUPERVISOR_CONF.is_file() else ""
    )
    already = "[program:sshd]" in text
    if already:
        return {"already_present": True, "ok": True, "backup_path": None,
                 "detail": "[program:sshd] already present in supervisord.conf"}
    if not apply:
        return {"already_present": False, "ok": True, "backup_path": None,
                 "detail": "would append [program:sshd] to supervisord.conf"}

    backup = _make_backup(SUPERVISOR_CONF)
    new_text = text.rstrip("\n") + "\n" + _supervise_block(port)
    SUPERVISOR_CONF.write_text(new_text, encoding="utf-8")

    r1 = _run(["supervisorctl", "-c", str(SUPERVISOR_CONF), "reread"])
    if r1.returncode != 0:
        SUPERVISOR_CONF.write_text(text, encoding="utf-8")
        return {"already_present": False, "ok": False, "backup_path": str(backup),
                 "detail": f"supervisorctl reread failed: "
                           f"{(r1.stderr or r1.stdout).strip()} — restored the backup"}

    r2 = _run(["supervisorctl", "-c", str(SUPERVISOR_CONF), "update"])
    return {"already_present": False, "ok": True, "backup_path": str(backup),
             "detail": (r1.stdout + r2.stdout).strip()}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="setup-sshd.py",
        description=(
            "Install/upgrade OpenSSH server in the running container and "
            "bring it up hardened. Dry run by default; --apply is "
            "required to write, install or start anything."
        ),
    )
    ap.add_argument("--apply", action="store_true",
                     help="perform the install/config/start. Without this: "
                          "dry run, nothing written or started.")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--force", action="store_true",
                     help="override the 'does this look like the zions "
                          "container' refusal (see module docstring)")
    ap.add_argument("--authorized-key-file", metavar="PATH",
                     help="file of public key line(s) to authorize (highest precedence)")
    ap.add_argument("--authorized-key", metavar="KEY",
                     help="a single public key line, e.g. 'ssh-ed25519 AAAA... you'")
    ap.add_argument("--port", type=int, default=22, metavar="N",
                     help="sshd port (default: 22). RunPod's template must "
                          "expose this TCP port for it to be reachable.")
    ap.add_argument("--no-persist-host-keys", action="store_true",
                     help="do not copy host keys to/from /data/ssh — the "
                          "host fingerprint changes on every pod restart")
    ap.add_argument("--supervise", action="store_true",
                     help="OPT-IN: run sshd as a supervisord program instead "
                          "of a plain background daemon (see module docstring)")
    return ap


def _nothing_to_do(report: dict, args) -> bool:
    if report.get("action") != "already-current":
        return False
    if report.get("keys_added"):
        return False
    if report.get("daemon") != "already-running":
        return False
    if args.supervise:
        sup = report.get("supervise") or {}
        if not sup.get("already_present"):
            return False
    return True


def _finish(args, report: dict, refusals: list[str], warnings: list[str]) -> int:
    report["refusals"] = refusals
    report["warnings"] = warnings
    # Defect 4 (normalized against scripts/backfill-records.py and
    # scripts/import-history.py, whose exit codes were always right: 0 is
    # success — the desired end state is in place, or --apply completed,
    # INCLUDING an --apply that was a no-op because everything was already
    # correct (a second --apply right after the first). 3 means a DRY RUN
    # found pending work --apply would change; a dry run that finds
    # nothing to do is 0, not 3. This script used to have 0 and 3 swapped
    # from that convention — its own author flagged it — which is exactly
    # the trap of three sibling scripts sharing flags with opposite exit
    # meanings: `if script; then` reads backwards on two of the three.
    if refusals:
        exit_code = 1
    elif args.apply:
        exit_code = 0
    else:
        exit_code = 0 if _nothing_to_do(report, args) else 3
    report["exit_code"] = exit_code

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        _print_human(report, args)
    return exit_code


def _print_human(report: dict, args) -> None:
    print("=" * 68)
    print(f"setup-sshd.py — {report.get('mode', '?').upper()}")
    print("=" * 68)
    for w in report.get("warnings", []):
        print(w)
    print(f"openssh-server: installed={report.get('installed_version')} "
          f"candidate={report.get('candidate_version')} "
          f"action={report.get('action')}")
    print(f"key source:     {report.get('key_source')}")
    added = report.get("keys_added") or []
    print(f"keys added:     {len(added)}")
    print(f"config:         mode={report.get('config_mode')} "
          f"path={report.get('config_path')} changed={report.get('config_changed')}")
    print(f"                {report.get('config_reason')}")
    if report.get("sshd_check"):
        print(f"sshd -t/-T:     {report['sshd_check']}")
    print(f"host keys:      {report.get('host_keys')}")
    print(f"daemon:         {report.get('daemon')}")
    if args.supervise:
        sup = report.get("supervise") or {}
        print(f"supervise:      {sup.get('detail')}")
    print(f"port:           {report.get('port')}")
    print()
    print(f"RunPod's template must expose TCP {report.get('port')} for this to be "
          f"reachable from outside the pod. The Web Terminal keeps working either way.")
    if report.get("refusals"):
        print()
        print("REFUSED:")
        for r in report["refusals"]:
            print(f"  - {r}")
    print()
    if report.get("refusals"):
        pass  # already printed above
    elif args.apply:
        print("APPLY complete.")
    elif report["exit_code"] == 3:
        print("DRY RUN — nothing was written, installed, or started, and "
              "there is work to do. Re-run with --apply.")
    else:
        print("Nothing to do — already installed, keyed and running "
              "(dry run: nothing was written, installed, or started).")


def main(argv=None) -> int:
    args = _build_argparser().parse_args(argv)
    refusals: list[str] = []
    warnings: list[str] = []
    # Every documented --json key is present from the start, even down a
    # refusal path that returns before ever computing most of them — a
    # refusal is not an excuse for `--json` to come back a different shape.
    report: dict = {
        "mode": "apply" if args.apply else "dry-run",
        "port": args.port,
        "installed_version": None,
        "candidate_version": None,
        "action": None,
        "key_source": None,
        "keys_added": [],
        "config_path": None,
        "daemon": "not-started",
        "host_keys": None,
    }

    if not _is_root():
        refusals.append(
            "must run as root to install or configure sshd (no --force override)"
        )
        return _finish(args, report, refusals, warnings)

    if not _looks_like_zions_container():
        if not args.force:
            refusals.append(
                f"this does not look like the zions container "
                f"({MARKER_VENV} and {MARKER_MAIN} not both present) — "
                f"pass --force to override"
            )
            return _finish(args, report, refusals, warnings)
        warnings.append(
            "WARNING: --force overriding the zions-container check — "
            f"{MARKER_VENV} and/or {MARKER_MAIN} not both present"
        )

    # --- 1/2. apt-get update, then install/upgrade openssh-server -------
    ok, out = _apt_update()
    if not ok:
        warnings.append(
            "WARNING: apt-get update exited non-zero (one repo channel — e.g. "
            "the CUDA one — can fail without the Ubuntu ones failing); "
            f"continuing since apt-cache policy is what actually decides "
            f"this: {out.strip()[-300:]}"
        )
    if not args.apply:
        warnings.append(
            "NOTE: apt-get update was run for real even in this dry run — "
            "it only refreshes package lists; nothing was installed or changed."
        )

    installed, candidate, _policy_text = _apt_cache_policy()
    action = _plan_package_action(installed, candidate)
    report.update({
        "installed_version": installed,
        "candidate_version": candidate,
        "action": action,
    })

    if action == "unavailable":
        refusals.append(
            "apt-cache policy shows no candidate for openssh-server — "
            "check apt sources / apt-get update output above"
        )
        return _finish(args, report, refusals, warnings)

    if action in ("install", "upgrade"):
        if args.apply:
            r = _run(["apt-get"] + _apt_install_args(action))
            if r.returncode != 0:
                refusals.append(
                    f"apt-get install failed: {(r.stderr or r.stdout).strip()[-500:]}"
                )
                return _finish(args, report, refusals, warnings)
            installed, candidate, _ = _apt_cache_policy()
            report["installed_version"] = installed
            report["candidate_version"] = candidate
        else:
            r = _run(["apt-get", "-s"] + _apt_install_args(action))
            report["simulated_install"] = "\n".join(r.stdout.splitlines()[-15:])

    # --- 3. resolve a public key -----------------------------------------
    key_source, candidate_keys, key_refusals = _resolve_key_source(args)
    if key_refusals:
        refusals.extend(key_refusals)
        return _finish(args, report, refusals, warnings)
    report["key_source"] = key_source

    if args.apply:
        keys_added, key_backup = _append_keys(candidate_keys)
        if key_backup:
            report["authorized_keys_backup"] = key_backup
    else:
        keys_added = _planned_new_keys(candidate_keys)
    report["keys_added"] = keys_added

    # --- 4. sshd_config: drop-in vs. fallback ----------------------------
    plan = _plan_config(args.port, apply=args.apply)
    report["config_mode"] = plan["mode"]
    report["config_path"] = plan["config_path"]
    report["config_reason"] = plan["reason"]
    report["config_changed"] = plan["changed"]

    persist_host_keys = not args.no_persist_host_keys

    if args.apply:
        # --- 5. host keys (before verifying — sshd -t needs real ones) --
        host_info = _ensure_host_keys(persist=persist_host_keys)
        report["host_keys"] = host_info["host_keys"]
        warnings.extend(host_info["notes"])

        # --- 6. verify for real, refuse+rollback if it fails ------------
        ok, effective, detail = _sshd_effective_config()
        report["sshd_check"] = detail
        if not ok:
            if plan["changed"]:
                _rollback_config(plan)
            refusals.append(detail)
            return _finish(args, report, refusals, warnings)

        # --- 7. start / restart, by pid only -----------------------------
        did_install = action in ("install", "upgrade")
        pid = _sshd_pid_if_alive()

        if args.supervise:
            sup_text = (
                SUPERVISOR_CONF.read_text(encoding="utf-8", errors="replace")
                if SUPERVISOR_CONF.is_file() else ""
            )
            already_supervised = "[program:sshd]" in sup_text

            if not already_supervised:
                # First time handing this daemon to supervisord: stop any
                # STANDALONE instance first (left running from a prior
                # non-supervised run) — otherwise supervisord's own child
                # fails to bind the same port and goes FATAL (confirmed
                # against the real image: "Bind to port 22 ... Address
                # already in use"). Once supervisord owns it, its own
                # autorestart=true is what keeps it alive; never done again
                # below on an idempotent re-run.
                if pid is not None and not _stop_sshd(pid):
                    warnings.append(
                        f"WARNING: a standalone sshd (pid {pid}) did not "
                        f"exit within the timeout before handing off to "
                        f"supervisord"
                    )
                sup = _ensure_supervised(args.port, apply=True)
                report["supervise"] = sup
                if not sup["ok"]:
                    if plan["changed"]:
                        _rollback_config(plan)
                    refusals.append(sup["detail"])
                    return _finish(args, report, refusals, warnings)
                # supervisord's own startsecs (2s, _supervise_block) is how
                # long it waits before calling this RUNNING; give it a
                # moment, then read the REAL state back rather than assume.
                time.sleep(2.5)
                new_pid = _sshd_pid_if_alive()
                if new_pid is not None:
                    report["daemon"] = "restarted" if pid is not None else "started"
                else:
                    report["daemon"] = "not-started"
                    warnings.append(
                        "WARNING: supervisorctl update completed but sshd "
                        "is not showing as running yet — check "
                        "`supervisorctl status` and its own log for why"
                    )
            else:
                # Already supervisord's program from a prior run.
                # supervisord itself (autorestart=true) is what keeps it
                # alive; this script only intervenes when the binary or
                # config changed underneath it, via `supervisorctl
                # restart`, which touches ONLY this one program.
                report["supervise"] = {
                    "already_present": True, "ok": True, "backup_path": None,
                    "detail": "[program:sshd] already present in supervisord.conf",
                }
                if did_install or plan["changed"]:
                    r = _run(["supervisorctl", "-c", str(SUPERVISOR_CONF), "restart", "sshd"])
                    time.sleep(1.0)
                    new_pid = _sshd_pid_if_alive()
                    if r.returncode == 0 and new_pid is not None:
                        report["daemon"] = "restarted"
                    else:
                        report["daemon"] = "not-started"
                        warnings.append(
                            f"WARNING: 'supervisorctl restart sshd' may not "
                            f"have succeeded: {(r.stderr or r.stdout).strip()}"
                        )
                else:
                    report["daemon"] = "already-running" if pid is not None else "not-started"
        elif pid is None:
            r = _start_sshd()
            if r.returncode != 0:
                refusals.append(f"sshd failed to start: {(r.stderr or r.stdout).strip()}")
                return _finish(args, report, refusals, warnings)
            report["daemon"] = "started"
        elif did_install or plan["changed"]:
            if not _stop_sshd(pid):
                warnings.append(
                    f"WARNING: sshd (pid {pid}) did not exit within the timeout; "
                    f"starting a new instance anyway"
                )
            r = _start_sshd()
            if r.returncode != 0:
                refusals.append(f"sshd failed to restart: {(r.stderr or r.stdout).strip()}")
                return _finish(args, report, refusals, warnings)
            report["daemon"] = "restarted"
        else:
            report["daemon"] = "already-running"
    else:
        report["host_keys"] = "persisted" if persist_host_keys else "ephemeral"
        pid = _sshd_pid_if_alive()
        report["daemon"] = "already-running" if pid is not None else "not-started"
        if shutil.which("sshd") is None and not Path("/usr/sbin/sshd").is_file():
            warnings.append(
                "NOTE: openssh-server is not installed yet, so sshd -t/-T "
                "cannot verify the planned config in this dry run — --apply "
                "verifies it for real and refuses automatically if the "
                "effective config would allow password authentication."
            )
        if args.supervise:
            report["supervise"] = _ensure_supervised(args.port, apply=False)

    return _finish(args, report, refusals, warnings)


if __name__ == "__main__":
    sys.exit(main())
