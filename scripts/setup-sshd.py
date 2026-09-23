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
    0. Refuse outright, before writing anything, if the config already on
       disk is unsafe in a way this script cannot fix on its own: a
       `Match` block ANYWHERE in the real `Include` closure reachable
       from `sshd_config` — not just `sshd_config` and
       `sshd_config.d/*.conf` themselves; `Include` is followed
       recursively, case-insensitively, glob-expanded, with relative
       paths resolved against `/etc/ssh` the way sshd itself does, and
       an `Include` resolving outside `/etc/ssh` is its own refusal (see
       N2 — a `Match` can make password login live for some client
       contexts while a plain `sshd -T` — no `-C` — reports it disabled:
       see B2/N2); a stray listener already on the target port that
       isn't the sshd this script itself manages, e.g. an unrelated
       process or another sshd master running with a DIFFERENT pidfile
       (see N3 — never silently adopted as "already running"); another
       drop-in that sorts before this script's own `00-zions.conf`
       (first-match-wins order); or a conflicting active `Port` directive
       anywhere else (OpenSSH ACCUMULATES `Port` lines across files
       instead of first-match-wins — see M8 below). Since nothing has
       been written yet, this refusal needs no rollback.
    1. `apt-get update` (package lists are pruned from the image — Dockerfile
       runs `rm -rf /var/lib/apt/lists/*` — so this is required every time).
    2. Install `openssh-server` if missing, or `--only-upgrade` it to the
       apt candidate if already present. NEVER `apt-get upgrade` or
       `dist-upgrade` — see the module-level safety notes below.
    2b. Once sshd exists, refuse if the effective `AuthorizedKeysFile`
       (from `sshd -T`) is not the default `~root/.ssh/authorized_keys`
       — writing a key to a file sshd was never told to read would be a
       silent no-op (see N11).
    3. Resolve one or more usable Ed25519/RSA/ECDSA PUBLIC keys (never a
       private one) from, in order: `--authorized-key-file`,
       `--authorized-key`, the RunPod-injected `$PUBLIC_KEY`, or an
       existing non-empty `/root/.ssh/authorized_keys`. Refuses outright
       if none is usable. A candidate must start with a known
       public-key-type token and be a single line containing no
       `PRIVATE KEY` — `ssh-keygen -l -f` alone is NOT trusted for this
       (it returns 0 on a private key file too — see B3). `/root/.ssh`
       and `authorized_keys` are refused outright if either is a
       SYMLINK (checked with `lstat`, plus `O_NOFOLLOW` on the actual
       write as a second-layer guard — see N11/N15: a symlinked
       `authorized_keys` once got a key appended straight into
       `/etc/hostname`). Otherwise `/root/.ssh` is always forced to 0700
       and `authorized_keys` to 0600 (plus ownership) EVERY apply, even
       when there is no new key to add (H7). Only FINGERPRINTS — never
       key text — ever appear in the report or `--json` output's
       `keys_added`.
    4. Write the hardening directives — see "THE HARD SAFETY RULES" below
       — to a drop-in under `/etc/ssh/sshd_config.d/`, or fall back to
       editing `sshd_config` directly with a backup, WHICHEVER the real
       config on disk actually supports (checked fresh every run: see
       `_dropin_usable`, and OPERATIONS.md for what this pod's shipped
       config was found to be).
    5. Ensure host keys exist (`ssh-keygen -A`), and by default persist
       them to `/data/ssh/` so the pod keeps the same host identity across
       restarts — see "HOST KEYS" below.
    6. Validate the result for REAL: `sshd -t`, then `sshd -T` with NO
       `-C` AND `sshd -T -C` across a full representative address matrix
       — loopback, every RFC1918/CGNAT private range, link-local, a
       public v4/v6 address, and this container's own address(es), each
       for both `user=root` and a non-root user (see N2 — this is an
       INDEPENDENT check from step 0's static Match-scan, not a
       replacement for it; never just reading back the file this script
       itself wrote), and REFUSE — rolling back the config write and any
       `authorized_keys` append made this run (see ROLLBACK below) — if
       the effective config would allow password login for ANY of those
       contexts, or would leave sshd listening on any port other than the
       one requested.
    7. Start (or, if already running and the binary/config changed,
       restart — by pid, never a blanket pkill) `/usr/sbin/sshd` as a
       plain background daemon. `--supervise` (opt-in, OFF by default)
       instead adds it as a supervisord program — see that flag's own
       section below for why it defaults off. Either way, "started" means
       more than just "something is really listening on IPv4 at the
       configured port" (read from `/proc/net/tcp`, never inferred from
       sshd's own exit status, which returns 0 even on a partial bind
       failure — see H6): the listening socket's kernel inode is mapped
       to the pid that actually holds it (via `/proc/<pid>/fd`), and that
       pid must be the one recorded in this script's own pidfile (see
       N3) — a foreigner holding the port, however it got there, is
       never mistaken for "started". If it is not confirmed, that is a
       refusal (exit 1), with the same rollback as step 6.

ROLLBACK. A refusal discovered AFTER step 3/4 has already written
something undoes exactly what this run wrote: the config write (drop-in
or fallback edit) is restored to its prior content, and an
`authorized_keys` append is restored from this run's own backup (or the
file is removed if this run created it fresh). Host key files under
`/data/ssh` are the one deliberate exception — they are NEVER rolled
back, because they are never what causes a refusal and deleting or
regenerating them would only churn the pod's host-key fingerprint for no
security benefit (see HOST KEYS below); an operator connecting after a
refused run still sees the same host identity. When a refusal happens
after this script had to stop an already-running standalone `sshd` to
attempt a handoff (`--supervise`) or a restart, it makes one best-effort
attempt to bring some sshd back up and verify it is really listening
before returning — see H5/H6 — so the pod is not left with literally no
sshd whenever this script is the one that took it down.

DRY RUN IS THE DEFAULT (no `--apply`). It runs a REAL `apt-get update`
(package lists only — nothing is installed, and the report says so) and a
REAL `apt-get install -s` (`-s` = simulate) so the reported action is
accurate, but writes NOTHING to `/etc/ssh`, `/root/.ssh`, `/data/ssh` or
`/etc/supervisor/conf.d/`, installs nothing, and starts nothing. If
openssh-server is not installed yet, `sshd -t`/`sshd -T` cannot run (the
binary does not exist), so the dry run says so honestly instead of
pretending to have verified the resulting config — that verification is
real and happens automatically during `--apply` (step 6 above), which
refuses if it does not pass. The step-0 config-safety refusal (a Match
anywhere in the Include closure, an Include escaping /etc/ssh, a stray
non-managed listener already on the port, drop-in ordering, conflicting
Port directives) is checked — and enforced — in a dry run too, same as
the "not root" / "not this container" refusals always were.

THE HARD SAFETY RULES (see also inline comments at each site)
  - Never `apt-get upgrade`/`dist-upgrade` — only ever `apt-get install
    [--no-install-recommends|--only-upgrade] -y openssh-server -o
    Dpkg::Options::=--force-confold openssh-server`, so an operator's own
    `sshd_config` (or, once this script has run once, its own drop-in)
    never gets silently clobbered by a package-shipped default, and no
    unrelated package on the pod is ever touched while vLLM is running.
  - Never print, log, copy or otherwise handle a PRIVATE key. A candidate
    must start with a known public-key type token
    (`ssh-ed25519`/`ssh-rsa`/`ecdsa-sha2-nistp256`/`384`/`521`/
    `sk-ssh-ed25519@openssh.com`/`sk-ecdsa-sha2-nistp256@openssh.com`),
    be a single line, and contain no `PRIVATE KEY` substring — checked
    BEFORE this script ever shells out to `ssh-keygen`, because the real
    `ssh-keygen -l -f` returns exit 0 on a PRIVATE key file too (confirmed
    against the real image — see B3). Options prefixes (`command=...`,
    `environment=...`) are deliberately NOT supported; the safer choice is
    to reject them rather than parse them. Only a validated PUBLIC key
    ever goes into `authorized_keys`; only its FINGERPRINT — never the key
    text — ever appears in a report. Host PRIVATE keys are created by
    `ssh-keygen -A` (never generated or read by this script's own code)
    and this script never reads their contents, only their existence,
    size and mtime.
  - Never enable password login, for ANY client. The directives this
    script writes are, at minimum: `PasswordAuthentication no`,
    `PermitEmptyPasswords no`, `KbdInteractiveAuthentication no`,
    `ChallengeResponseAuthentication no`, `PubkeyAuthentication yes`,
    `PermitRootLogin prohibit-password`. Before ever writing anything,
    this script refuses outright if a `Match` block exists ANYWHERE in
    the real Include closure reachable from sshd_config — not just
    sshd_config and sshd_config.d/*.conf themselves (a `Match` can flip
    password auth on for some client contexts while a plain `sshd -T`
    reports it off; see B2/N2); after writing, it verifies with `sshd -T`
    AND `sshd -T -C` across a full representative address matrix
    (loopback, every RFC1918/CGNAT private range, link-local, public
    v4/v6, and this container's own address(es), each for root and a
    non-root user — see N2). `sshd -T` on this image's own OpenSSH 9.6
    build always PRINTS `permitrootlogin` back as `without-password`
    regardless of which of the two synonymous spellings was written —
    both are accepted when checking the real outcome; nothing else is.
  - Never leave sshd listening on any port but the one requested. OpenSSH
    ACCUMULATES `Port` directives across the main file and every loaded
    drop-in instead of first-match-wins — writing `Port 2222` in this
    script's own drop-in does NOT suppress an unrelated active `Port 22`
    elsewhere; both would apply. This script refuses outright (step 0) if
    it finds such a conflict anywhere it does not itself control, and
    re-verifies the REAL, POST-write set of listening ports via `sshd -T`
    as part of step 6 (see M8).
  - Never touch any other supervisord program. The default start path
    (plain `/usr/sbin/sshd`) never goes near supervisord at all; even
    `--supervise`'s `supervisorctl update` only starts the ONE newly added
    program (verified against the real image — see OPERATIONS.md).
  - Never restart anything by killing broadly. The daemon is found by its
    own pidfile (`/run/sshd.pid`, confirmed to be `sshd` via
    `/proc/<pid>/comm`) or not touched at all.
  - Never leave the pod with no sshd if this script is the one that took
    it down. `--supervise`'s handoff stops any standalone daemon first
    (supervisord's own child cannot bind the same port otherwise); if the
    handoff then fails, this script restarts the standalone daemon and
    re-verifies it is really listening before refusing (see H5). The same
    applies to a standalone restart whose replacement does not come up
    listening.

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
Host key files are the one thing a refusal on this run never rolls back
— see ROLLBACK above.

--SUPERVISE (opt-in, OFF by default). Appends a `[program:sshd]` block to
`/etc/supervisor/conf.d/supervisord.conf` (backed up first; the append is
validated with `supervisorctl reread`, and the backup is restored and the
run refused if that fails, or if `supervisorctl update` completes but
sshd does not come up really listening) and runs `supervisorctl update`.
Refuses cleanly (exit 1, full `--json`) rather than raising if
`supervisord.conf` does not exist at all — appending to a file supervisord
was never told to read would be pointless (see M8). Verified against a
throwaway container running this image's real supervisord (no GPU — vllm
alone goes FATAL, everything else comes up): adding ONE new
`[program:sshd]` section and running `reread` + `update` started only
that new program and left every other program's own RUNNING state alone —
`update` only (re)starts a program whose OWN section changed or is new,
never one supervisord already has running unchanged. Kept, on that
evidence; see OPERATIONS.md for the transcript.

EXIT CODES. The architect's ruling, normalized across all three operator
scripts (scripts/backfill-records.py, scripts/import-history.py, and this
one — see CHANGELOG.md and OPERATIONS.md):
    0   Desired end state reached. Either `--apply` completed
        (installed/upgraded, added a key, (re)started the daemon and
        confirmed it listening, or was a NO-OP because everything was
        already correct — including a second `--apply` right after the
        first, which is idempotent), or a DRY RUN found nothing to do.
    1   A refusal or error the operator needs to look at (see `refusals`
        in `--json` output): not root, does not look like this
        container, a config-safety refusal (a Match anywhere in the
        Include closure / an Include escaping /etc/ssh / a stray
        non-managed listener already on the port / drop-in ordering /
        conflicting Port — step 0 above), a non-default
        AuthorizedKeysFile or a symlinked `.ssh`/`authorized_keys` (N11),
        no usable public key anywhere, a malformed or private key,
        `apt-get install` failed, `sshd -t`/`sshd -T`(`-C`) failed or
        would allow password login or the wrong port, `sshd` failed to
        (re)start, sshd's listener not confirmed as the one THIS script
        started after a start/restart, or `--supervise`'s `reread` failed
        or its conf file does not exist.
    2   argparse's own usage errors — unknown flags, missing required
        values — the Python standard library's own convention.
    3   A DRY RUN found something `--apply` WOULD do (install/upgrade,
        add a key, or start/restart the daemon) — informational, not a
        failure: re-run with `--apply` once ready.
    4   `--apply` made progress but work remains. Unused by this script —
        every `--apply` here either fully succeeds (0) or refuses (1);
        listed for consistency with the other two operator scripts, which
        share this same exit-code convention.
"""

import argparse
import glob
import json
import os
import re
import shlex
import shutil
import socket
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

# /proc/net/tcp[6] and /proc itself are kernel-global truth, never
# something a test should redirect to a fake tree — H6's listening check,
# and N3's inode->pid ownership mapping, read the real ones. A test that
# needs a real, attributable listener uses the real fake-sshd process
# (which really forks, binds and sets its own /proc/<pid>/comm) rather
# than faking these paths.
PROC_NET_TCP = Path("/proc/net/tcp")
PROC_NET_TCP6 = Path("/proc/net/tcp6")
PROC_DIR = Path("/proc")

CMD_TIMEOUT_S = 90

# "Port" is included here even though it is NOT first-match-wins in real
# OpenSSH (it ACCUMULATES across files) — it still needs to be recognised
# as "a directive this script manages" so _dropin_usable's ordering check
# and the fallback path's _comment_out_conflicts both see and neutralize
# an active Port line in the MAIN file (see M8). A conflicting Port line
# in some OTHER file is handled separately, by outright refusal — see
# _conflicting_port_sources.
DIRECTIVE_KEYS = (
    "PasswordAuthentication",
    "PermitEmptyPasswords",
    "KbdInteractiveAuthentication",
    "ChallengeResponseAuthentication",
    "PubkeyAuthentication",
    "PermitRootLogin",
    "Port",
)

INCLUDE_RE = re.compile(
    r"^[ \t]*Include[ \t]+\S*sshd_config\.d\S*\*\.conf[ \t]*$", re.I | re.M
)

# N2: the GENERAL form, used to walk the FULL Include closure sshd itself
# would load (not just the one sshd_config.d/*.conf shape INCLUDE_RE
# looks for) — case-insensitive keyword, capturing every argument on the
# line so multiple space/quote-separated glob patterns are all followed.
GENERIC_INCLUDE_RE = re.compile(r"^[ \t]*Include[ \t]+(.+?)[ \t]*$", re.I | re.M)

MATCH_RE = re.compile(r"^[ \t]*Match\b", re.I | re.M)
PORT_LINE_RE = re.compile(r"^[ \t]*Port\b[ \t]+(\S+)", re.I | re.M)

# Known public-key type tokens (B3). Deliberately closed-world: anything
# not starting with one of these — including an options-prefixed line
# like `command="..." ssh-ed25519 ...` — is refused. The safer choice is
# to reject options rather than parse them.
_PUBKEY_TYPE_TOKENS = (
    "ssh-ed25519",
    "ssh-rsa",
    "ecdsa-sha2-nistp256",
    "ecdsa-sha2-nistp384",
    "ecdsa-sha2-nistp521",
    "sk-ssh-ed25519@openssh.com",
    "sk-ecdsa-sha2-nistp256@openssh.com",
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
    """Back up `path` beside itself as `<name>.bak-<UTC stamp>[-N]` —
    never overwriting an existing backup. A bare UTC-second stamp collides
    whenever this script runs twice in the same wall-clock second (M8:
    this used to raise an unhandled RuntimeError straight out of
    `_make_backup`, past every caller's own --json/refusal handling); a
    numeric suffix is added instead, so a same-second collision is
    resolved rather than crashing. Only if 1000 suffixes in the same
    second are ALSO taken (never observed; kept as a last-resort
    safety net, not a realistic case) does this still raise — every
    caller already runs inside `main()`'s own exception-free control
    flow, so even that would need a caller-side fix if it were ever hit."""
    stamp = _utc_stamp()
    backup = path.with_name(path.name + f".bak-{stamp}")
    if backup.exists():
        for n in range(1, 1000):
            candidate = path.with_name(path.name + f".bak-{stamp}-{n}")
            if not candidate.exists():
                backup = candidate
                break
        else:
            raise RuntimeError(
                f"refused — could not find an unused backup path beside "
                f"{path} even after 1000 suffixed attempts at stamp {stamp}"
            )
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
# B2 / M8 — config-safety preconditions, checked BEFORE any write this run
# makes. A refusal here needs no rollback: nothing has been written yet.
# ---------------------------------------------------------------------------

def _split_include_args(argstr: str) -> list[str]:
    """Split one `Include` line's argument text into its individual glob
    patterns, honouring double-quoted patterns containing spaces (sshd's
    own config tokenizer supports this). Falls back to a plain whitespace
    split if the text is unbalanced/unparseable — never raises."""
    try:
        return shlex.split(argstr)
    except ValueError:
        return argstr.split()


def _resolve_config_closure(start: Path) -> tuple[list[Path], str | None]:
    """N2: follow `Include` recursively, the way sshd itself parses it —
    not just the one 'Include .../sshd_config.d/*.conf' shape this script
    writes. The `Include` keyword is matched case-insensitively; each
    argument is glob-expanded (sorted, like sshd); a relative pattern is
    resolved against ETC_SSH_DIR regardless of which file contains the
    Include line (sshd_config(5): "Files without absolute path names are
    assumed to be in /etc/ssh" — NOT relative to the including file's own
    directory). Cycles are broken by resolved real path.

    Returns `(files_visited, refusal)`. `files_visited` always includes
    everything visited before a refusal (so a refusal message can point
    at the offending file); a refusal fires if ANY resolved Include
    target — including through a symlink — lands outside ETC_SSH_DIR:
    this script refuses rather than try to reason about config pulled in
    from somewhere it does not manage."""
    try:
        etc_root = ETC_SSH_DIR.resolve()
    except OSError:
        etc_root = ETC_SSH_DIR
    visited_real: set[Path] = set()
    files: list[Path] = []
    queue: list[Path] = [start]
    while queue:
        current = queue.pop(0)
        if not current.is_file():
            continue
        try:
            real = current.resolve()
        except OSError:
            real = current
        if real in visited_real:
            continue
        visited_real.add(real)
        files.append(current)
        text = current.read_text(encoding="utf-8", errors="replace")
        for m in GENERIC_INCLUDE_RE.finditer(text):
            for tok in _split_include_args(m.group(1)):
                pat = Path(tok)
                if not pat.is_absolute():
                    pat = ETC_SSH_DIR / pat
                for matched in sorted(glob.glob(str(pat))):
                    mp = Path(matched)
                    try:
                        mp_real = mp.resolve()
                    except OSError:
                        mp_real = mp
                    if mp_real != etc_root and etc_root not in mp_real.parents:
                        return files, (
                            f"refusing — an Include in {current} resolves "
                            f"to {mp}, which is outside {ETC_SSH_DIR} — "
                            f"this script cannot safely reason about "
                            f"config pulled in from outside the tree it "
                            f"manages (N2)"
                        )
                    queue.append(mp)
    return files, None


def _match_block_locations() -> tuple[list[str], str | None]:
    """Every ACTIVE (uncommented) `Match` line ANYWHERE in the real
    Include closure sshd would load starting from sshd_config — not just
    sshd_config itself and sshd_config.d/*.conf (see N2: a `Match` pulled
    in by an arbitrary Include used to bypass this entirely). Matching is
    case-insensitive and tolerates leading whitespace (indented Match),
    exactly like sshd's own keyword parsing (MATCH_RE). A `Match` block
    can make password login live for some client contexts while `sshd -T`
    with no `-C` (which evaluates NO Match criteria at all) reports it
    disabled — see B2. This script does not try to reason about arbitrary
    Match criteria; it refuses outright on any hit.

    Returns `(locations, refusal)` — `refusal` is set instead of
    `locations` being trustworthy if the Include closure itself could not
    be safely resolved (an Include escaping ETC_SSH_DIR)."""
    if not SSHD_CONFIG.is_file():
        return [], None
    files, refusal = _resolve_config_closure(SSHD_CONFIG)
    if refusal:
        return [], refusal
    locations = []
    for p in files:
        text = p.read_text(encoding="utf-8", errors="replace")
        for m in MATCH_RE.finditer(text):
            locations.append(f"{p}:{text.count(chr(10), 0, m.start()) + 1}")
    return locations, None


def _conflicting_dropins() -> list[str]:
    """Other *.conf files under sshd_config.d/ that sort BEFORE
    00-zions.conf — sshd applies sshd_config.d/*.conf in sorted glob
    order, first-match-wins per directive, so a file sorting earlier
    could silently win over the hardening this script writes there."""
    if not SSHD_CONFIG_D.is_dir():
        return []
    our_name = DROPIN_PATH.name
    return sorted(
        p.name for p in SSHD_CONFIG_D.glob("*.conf")
        if p.name != our_name and p.name < our_name
    )


def _active_port_values(text: str) -> list[str]:
    return [m.group(1) for m in PORT_LINE_RE.finditer(text)]


def _conflicting_port_sources(port: int) -> list[str]:
    """Any file sshd will really parse that carries an ACTIVE `Port` line
    for a value other than the target. `Port` is one of the few
    sshd_config directives that ACCUMULATES across files instead of
    first-match-wins (confirmed: `sshd -T` can print multiple `port`
    lines) — writing our own `Port N` never suppresses another file's
    `Port 22`, so the only safe response is to refuse and name the
    source, never to assume our own write wins (see M8)."""
    if not SSHD_CONFIG.is_file():
        return []
    main_text = SSHD_CONFIG.read_text(encoding="utf-8", errors="replace")
    use_dropin, _ = _dropin_usable(main_text)
    conflicts = []
    if use_dropin:
        # Fallback mode neutralizes every active directive in DIRECTIVE_KEYS
        # (Port included) IN THE MAIN FILE itself via _comment_out_conflicts;
        # drop-in mode never touches the main file, so an active Port line
        # there is a real, permanent conflict.
        for val in _active_port_values(main_text):
            if val != str(port):
                conflicts.append(f"{SSHD_CONFIG} (active 'Port {val}')")
    # N2: walk the REAL Include closure (nested, arbitrary paths), not
    # just the one sshd_config.d/*.conf shape — a conflicting Port hiding
    # behind an unrelated Include used to be invisible here too.
    files, _closure_refusal = _resolve_config_closure(SSHD_CONFIG)
    for p in files:
        if p in (SSHD_CONFIG, DROPIN_PATH):
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        for val in _active_port_values(text):
            if val != str(port):
                conflicts.append(f"{p} (active 'Port {val}')")
    return conflicts


def _config_safety_refusal(port: int) -> str | None:
    """A hard precondition checked before ANY write this run makes (apt,
    keys, config, host keys). Covers B2 (Match blocks, drop-in ordering)
    and the setup-sshd part of M8 (a conflicting Port directive
    elsewhere). Returns a refusal string, or None if there is nothing to
    refuse (including: openssh-server not installed yet, nothing to
    check)."""
    if not SSHD_CONFIG.is_file():
        return _port_ownership_refusal(port)
    match_hits, include_refusal = _match_block_locations()
    if include_refusal:
        return include_refusal
    if match_hits:
        return (
            "refusing — a `Match` block exists in sshd_config or "
            "somewhere in its Include closure, which can make password "
            "login live for some client contexts while a plain `sshd -T` "
            "(no -C) reports it disabled: " + "; ".join(match_hits)
        )
    dropin_conflicts = _conflicting_dropins()
    if dropin_conflicts:
        return (
            "refusing — other drop-in file(s) in sshd_config.d/ sort "
            f"before {DROPIN_PATH.name} and would win first-match-wins for "
            "any directive they also set: " + ", ".join(dropin_conflicts)
        )
    port_conflicts = _conflicting_port_sources(port)
    if port_conflicts:
        return (
            "refusing — a conflicting active 'Port' directive exists "
            "outside this script's own drop-in (Port accumulates across "
            "files in OpenSSH; it does not follow first-match-wins): "
            + "; ".join(port_conflicts)
        )
    # N3: "listening" must mean OUR sshd. Checked here too (step 0, before
    # any write) so a stray listener is refused outright rather than
    # silently adopted later as "already running".
    return _port_ownership_refusal(port)


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
# Every candidate is validated by _validate_key; never a private key, and
# a FINGERPRINT (never key text) is all that is carried forward for
# reporting (see B3).
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
    """True + a FINGERPRINT (never the key text) if `line` is a genuine
    single-line public key. `ssh-keygen -l -f`'s exit code is NOT trusted
    on its own: the real binary returns 0 on a PRIVATE key file too
    (confirmed against the real image) — that is exactly why B3 was
    invisible. A type-token check and a `PRIVATE KEY` / multi-line check
    run FIRST, before this ever shells out to ssh-keygen, and are what
    actually keeps a private key out. Options prefixes
    (`command="..." ssh-ed25519 ...`) are deliberately NOT supported."""
    if "\n" in line or "\r" in line:
        return False, "key material spans more than one line — refusing"
    if "PRIVATE KEY" in line.upper():
        return False, "this looks like a PRIVATE key, not a public key — refusing"
    stripped = line.strip()
    if not stripped.startswith(_PUBKEY_TYPE_TOKENS):
        return False, (
            "does not start with a known public-key type token (" +
            ", ".join(_PUBKEY_TYPE_TOKENS) + ")"
        )

    fd, tmp = tempfile.mkstemp(prefix="setup-sshd-keycheck-")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(stripped + "\n")
        r = _run(["ssh-keygen", "-l", "-f", tmp])
        if r.returncode != 0:
            return False, ((r.stdout or r.stderr).strip() or "ssh-keygen could not parse it")
        fingerprint = ""
        if r.stdout:
            first_line = r.stdout.strip().splitlines()
            fingerprint = first_line[0] if first_line else ""
        return True, (fingerprint or "(ssh-keygen returned no fingerprint text)")
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


def _resolve_key_source(args) -> tuple[str | None, list[tuple[str, str]], list[str]]:
    """(source_label, candidates, refusals). Each candidate is
    `(key_line, fingerprint)` — the fingerprint (never the key line) is
    what ends up in the report/--json's `keys_added` (see B3).
    `candidates` is the NEW key material found at this source; it is
    empty (not a refusal) when the source is an already-populated
    authorized_keys with nothing new to add."""
    if args.authorized_key_file:
        p = Path(args.authorized_key_file)
        if not p.is_file():
            return None, [], [f"--authorized-key-file {p} does not exist"]
        lines = _read_key_lines(p.read_text(encoding="utf-8", errors="replace"))
        if not lines:
            return None, [], [f"--authorized-key-file {p} has no key lines"]
        candidates = []
        for ln in lines:
            ok, detail = _validate_key(ln)
            if not ok:
                return None, [], [
                    f"malformed public key in --authorized-key-file "
                    f"{p}: {detail}"
                ]
            candidates.append((ln, detail))
        return "file", candidates, []

    if args.authorized_key:
        ln = args.authorized_key.strip()
        ok, detail = _validate_key(ln)
        if not ok:
            return None, [], [f"malformed public key from --authorized-key: {detail}"]
        return "literal", [(ln, detail)], []

    pub_env = os.environ.get("PUBLIC_KEY", "")
    if pub_env.strip():
        lines = _read_key_lines(pub_env)
        if lines:
            candidates = []
            for ln in lines:
                ok, detail = _validate_key(ln)
                if not ok:
                    return None, [], [f"malformed public key in $PUBLIC_KEY: {detail}"]
                candidates.append((ln, detail))
            return "env:PUBLIC_KEY", candidates, []

    if AUTHORIZED_KEYS.is_file():
        existing = AUTHORIZED_KEYS.read_text(encoding="utf-8", errors="replace")
        if _read_key_lines(existing):
            return "existing-authorized_keys", [], []

    return None, [], [
        "no usable public key found — checked --authorized-key-file, "
        "--authorized-key, $PUBLIC_KEY, and an existing "
        f"{AUTHORIZED_KEYS} — refusing"
    ]


def _effective_authorized_keys_file() -> tuple[str | None, str | None]:
    """N11: the effective `AuthorizedKeysFile` from a plain `sshd -T`
    (this directive is never Match-conditional in any config that
    reaches this point — N2 already refused on any Match anywhere in the
    Include closure before this is ever called). Returns
    `(raw_value_or_None, error_or_None)`."""
    # Same requirement `_sshd_effective_config` already has: sshd -T
    # refuses to run at all ("Missing privilege separation directory")
    # until /run/sshd exists — on a just-installed sshd, nothing has
    # created it yet.
    PRIVSEP_DIR.mkdir(parents=True, exist_ok=True)
    r = _run([_sshd_binary(), "-T"])
    if r.returncode != 0:
        return None, f"sshd -T failed while checking AuthorizedKeysFile: {(r.stderr or r.stdout).strip()}"
    eff = _parse_effective(r.stdout)
    vals = eff.get("authorizedkeysfile") or [".ssh/authorized_keys"]
    return vals[0], None


def _authorized_keys_file_is_default(raw: str) -> bool:
    """True only if the FIRST path in the effective `AuthorizedKeysFile`
    (it may list several, space-separated; sshd tries each in order)
    resolves to exactly AUTHORIZED_KEYS, with `%u`/`%h` expanded for
    root. If it does not, sshd would never read the key this script
    writes — see N11."""
    expanded = raw.replace("%u", "root").replace("%h", str(ROOT_HOME_DIR))
    parts = expanded.split()
    first = parts[0] if parts else expanded
    p = Path(first)
    if not p.is_absolute():
        p = ROOT_HOME_DIR / p
    return os.path.normpath(str(p)) == os.path.normpath(str(AUTHORIZED_KEYS))


def _symlink_refusal() -> str | None:
    """N11/N15: refuse outright — before ever writing or chmodding
    anything — if `/root/.ssh` or `authorized_keys` is a symlink. This is
    exactly the mechanism behind the N15 incident (an `authorized_keys`
    symlinked to `/etc/hostname` got the key appended and mode 600):
    `Path.is_symlink()` uses `lstat`, so it reports the truth about the
    path itself rather than following it."""
    if ROOT_SSH_DIR.is_symlink():
        return f"refusing — {ROOT_SSH_DIR} is a symlink; this script never follows it (N11/N15)"
    if AUTHORIZED_KEYS.is_symlink():
        return f"refusing — {AUTHORIZED_KEYS} is a symlink; this script never writes or chmods through it (N11/N15)"
    return None


def _append_keys(candidates: list[tuple[str, str]]) -> tuple[list[str], str | None, bool]:
    """Append only the lines not already present (by type+base64
    identity), never clobbering the file. Returns
    `(fingerprints_actually_added, backup_path_or_None, created_fresh)` —
    `created_fresh` is True only when authorized_keys did not exist
    before this call, so a refusal discovered later can undo exactly this
    (see `_rollback_authorized_keys`). NEVER returns key text — only
    fingerprints (see B3).

    N11/N15: `main()` already refuses outright (via `_symlink_refusal`)
    before ever reaching here if either `ROOT_SSH_DIR` or
    `AUTHORIZED_KEYS` is a symlink. The checks repeated here, plus the
    `O_NOFOLLOW` open, are a defense-in-depth net for the TOCTOU window
    between that check and this call — a race that has never been
    observed, same tier as `_make_backup`'s own 1000-suffix safety net,
    and handled the same way: raise, rather than silently follow a
    symlink into a file this script does not own (the /etc/hostname
    incident)."""
    if ROOT_SSH_DIR.is_symlink():
        raise RuntimeError(f"refusing to use {ROOT_SSH_DIR} — it is a symlink")
    if AUTHORIZED_KEYS.is_symlink():
        raise RuntimeError(f"refusing to write {AUTHORIZED_KEYS} — it is a symlink")

    existed_before = AUTHORIZED_KEYS.is_file()
    existing_text = (
        AUTHORIZED_KEYS.read_text(encoding="utf-8", errors="replace")
        if existed_before else ""
    )
    existing_ids = {_key_identity(l) for l in _read_key_lines(existing_text)}
    to_add = [(l, fp) for l, fp in candidates if _key_identity(l) not in existing_ids]
    if not to_add:
        return [], None, False

    backup = None
    if existed_before:
        backup = _make_backup(AUTHORIZED_KEYS)

    ROOT_SSH_DIR.mkdir(parents=True, exist_ok=True)
    os.chmod(ROOT_SSH_DIR, 0o700)
    try:
        fd = os.open(
            str(AUTHORIZED_KEYS),
            os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW,
            0o600,
        )
    except OSError as e:
        raise RuntimeError(f"refusing to write {AUTHORIZED_KEYS} — {e}") from e
    with os.fdopen(fd, "a", encoding="utf-8") as f:
        if existing_text and not existing_text.endswith("\n"):
            f.write("\n")
        for l, _fp in to_add:
            f.write(l + "\n")
        f.flush()
        os.fchmod(f.fileno(), 0o600)
    return [fp for _l, fp in to_add], (str(backup) if backup else None), not existed_before


def _rollback_authorized_keys(key_backup: str | None, created_fresh: bool) -> None:
    """Undo exactly what `_append_keys` wrote, for a refusal discovered
    afterwards (sshd -t/-T(-C), daemon failed to start/listen,
    --supervise's reread). A no-op if `_append_keys` added nothing this
    run."""
    if created_fresh:
        try:
            AUTHORIZED_KEYS.unlink()
        except FileNotFoundError:
            pass
    elif key_backup:
        shutil.copy2(key_backup, AUTHORIZED_KEYS)
        os.chmod(AUTHORIZED_KEYS, 0o600)


def _planned_new_keys(candidates: list[tuple[str, str]]) -> list[str]:
    """Dry-run equivalent of `_append_keys` that reads but never writes.
    Returns fingerprints, matching --apply's report shape."""
    existing_text = (
        AUTHORIZED_KEYS.read_text(encoding="utf-8", errors="replace")
        if AUTHORIZED_KEYS.is_file() else ""
    )
    existing_ids = {_key_identity(l) for l in _read_key_lines(existing_text)}
    return [fp for l, fp in candidates if _key_identity(l) not in existing_ids]


def _harden_ssh_dir_permissions() -> list[str]:
    """H7: /root/.ssh must be 0700 and authorized_keys 0600, with root
    ownership, REGARDLESS of whether any key was actually added this run
    — a pre-existing 0777 .ssh / 0666 authorized_keys (or wrong owner) is
    silently exploitable and must be corrected every --apply, not only
    when there happens to be new key material to append. Ownership is
    best-effort: production is always real root (enforced by
    `_is_root()`), but a test environment that only patches the script's
    OWN root check may not be real root, so a chown failure there is
    noted, not fatal.

    N11/N15: `is_symlink()` (lstat) is checked FIRST for each path — a
    symlinked `.ssh` or `authorized_keys` is skipped (noted), never
    chmod/chown'd through. `main()` already refuses the whole run
    earlier via `_symlink_refusal` if either is a symlink at that point;
    this is the defense-in-depth net for the same TOCTOU window
    `_append_keys` guards against — exactly the mechanism behind the
    N15 incident (a symlinked `authorized_keys` pointing at
    `/etc/hostname` got chmodded 600)."""
    notes: list[str] = []
    if ROOT_SSH_DIR.is_symlink():
        notes.append(f"NOTE: {ROOT_SSH_DIR} is a symlink — refusing to chmod/chown through it")
    elif ROOT_SSH_DIR.is_dir():
        os.chmod(ROOT_SSH_DIR, 0o700)
        try:
            os.chown(ROOT_SSH_DIR, 0, 0)
        except OSError:
            notes.append(f"NOTE: could not chown {ROOT_SSH_DIR} to root:root")
    if AUTHORIZED_KEYS.is_symlink():
        notes.append(f"NOTE: {AUTHORIZED_KEYS} is a symlink — refusing to chmod/chown through it")
    elif AUTHORIZED_KEYS.is_file():
        os.chmod(AUTHORIZED_KEYS, 0o600)
        try:
            os.chown(AUTHORIZED_KEYS, 0, 0)
        except OSError:
            notes.append(f"NOTE: could not chown {AUTHORIZED_KEYS} to root:root")
    return notes


# ---------------------------------------------------------------------------
# sshd_config: drop-in vs. direct edit — decided fresh every run against
# what is REALLY on disk, never assumed.
# ---------------------------------------------------------------------------

def _directive_lines(port: int) -> list[str]:
    # Port is now ALWAYS written explicitly (never implicit-22), so the
    # written config never depends on OpenSSH's own compiled-in default —
    # see M8.
    return [
        "PasswordAuthentication no",
        "PermitEmptyPasswords no",
        "KbdInteractiveAuthentication no",
        "ChallengeResponseAuthentication no",
        "PubkeyAuthentication yes",
        "PermitRootLogin prohibit-password",
        f"Port {port}",
    ]


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
    directive this script manages (wherever it is — INCLUDING inside a
    `Match` block, since the regex matches by line, not by block context),
    then prepend this script's own block at the very top so first-match-
    wins always resolves to it. Idempotent: a prior run's block is
    replaced, not duplicated."""
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
    afterwards (sshd -t / sshd -T(-C))."""
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
# Verify for real — sshd -t then sshd -T (with and without -C), never just
# reading our own file (B2).
# ---------------------------------------------------------------------------

def _sshd_binary() -> str:
    """The ABSOLUTE path to sshd. Real sshd re-execs itself for privilege
    separation and refuses ('sshd re-exec requires execution with an
    absolute path') if it was invoked via a bare, PATH-resolved argv[0] —
    confirmed against the real image. `shutil.which` still does the PATH
    lookup (so a test's fake bin dir is found the same way), but the
    RESULT handed to subprocess is always absolute."""
    return shutil.which("sshd") or "/usr/sbin/sshd"


def _parse_effective(text: str) -> dict[str, list[str]]:
    """Every value for every key, IN ORDER — never overwriting a prior
    occurrence. A directive like `Port` can legitimately appear more than
    once in `sshd -T`'s own output (it ACCUMULATES rather than
    first-match-wins); collapsing to one value per key is exactly what
    made the port-accumulation bug (M8) invisible to this script before."""
    eff: dict[str, list[str]] = {}
    for line in text.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2:
            eff.setdefault(parts[0].lower(), []).append(parts[1].strip())
    return eff


def _check_effective(effective: dict[str, list[str]], port: int, context: str) -> str | None:
    """Returns a refusal string, or None if this ONE `sshd -T`(-C)
    invocation's effective config is fully safe."""
    must_equal = {
        "passwordauthentication": "no",
        "permitemptypasswords": "no",
        "kbdinteractiveauthentication": "no",
        "pubkeyauthentication": "yes",
    }
    for key, expected in must_equal.items():
        for got in effective.get(key, []):
            if got.lower() != expected:
                return (
                    f"sshd -T{context} reports {key}={got}, expected "
                    f"{expected} — refusing to start sshd with password "
                    f"login reachable"
                )

    # "prohibit-password" and its older synonym "without-password" mean
    # the same thing; this build's sshd -T always prints the latter back
    # (confirmed against the real image) regardless of which was written.
    for got in effective.get("permitrootlogin", []):
        if got.lower() not in ("prohibit-password", "without-password"):
            return (
                f"sshd -T{context} reports PermitRootLogin={got}, expected "
                f"prohibit-password — refusing"
            )

    port_vals = sorted(set(effective.get("port", [])))
    if port_vals and port_vals != [str(port)]:
        return (
            f"sshd -T{context} reports sshd would listen on port(s) "
            f"{', '.join(port_vals)}, expected ONLY {port} — refusing "
            f"(OpenSSH accumulates Port directives across the main file "
            f"and drop-ins)"
        )
    return None


# N2: representative addresses spanning every RunPod-plausible source —
# loopback, each private/CGNAT range a proxy or sidecar could sit in,
# link-local, and a public v4/v6 — PLUS whatever this container's own
# addresses turn out to be (so "connecting to myself" is covered too).
# Labels are informational only; the literal `addr=` value is what a
# refusal message names.
_CHECK_ADDRESSES = (
    ("loopback-v4", "127.0.0.1"),
    ("loopback-v6", "::1"),
    ("rfc1918-10/8", "10.1.2.3"),
    ("rfc1918-172.16/12", "172.20.5.6"),
    ("rfc1918-192.168/16", "192.168.1.50"),
    ("cgnat-100.64/10", "100.64.1.2"),
    ("link-local", "169.254.1.1"),
    ("public-v4", "203.0.113.9"),
    ("public-v6", "2001:db8::9"),
)

_CHECK_USERS = ("root", "nobody")


def _container_addresses() -> list[str]:
    """Best-effort: this container's own bound addresses (v4 and v6), so
    the -C matrix also covers "a client connecting to one of my own
    addresses" (e.g. a RunPod proxy address bound locally). Never fatal
    if it cannot be determined — an empty result just means the matrix
    runs without this extra set."""
    r = _run(["hostname", "-I"])
    if r.returncode == 0 and r.stdout.strip():
        return [a for a in r.stdout.split() if a]
    return []


def _laddr_for(addr: str) -> str:
    return "::" if ":" in addr else "0.0.0.0"


def _check_context_specs(container_addrs: list[str]) -> list[tuple[str, str]]:
    """[(label, addr)] — the fixed representative set plus this
    container's own addresses, deduplicated."""
    seen = set()
    specs = []
    for label, addr in _CHECK_ADDRESSES:
        if addr not in seen:
            seen.add(addr)
            specs.append((label, addr))
    for i, addr in enumerate(container_addrs):
        if addr not in seen:
            seen.add(addr)
            specs.append((f"container-own-{i}", addr))
    return specs


def _sshd_effective_config(port: int) -> tuple[bool, dict, str]:
    """`sshd -t`, then `sshd -T` with NO `-C`, PLUS `sshd -T -C` across a
    full representative address matrix (loopback, every private/CGNAT
    range, link-local, public v4/v6, and this container's own address(es)
    — see `_CHECK_ADDRESSES`), each checked for BOTH `user=root` and a
    non-root user (N2). A plain `sshd -T` evaluates NO `Match` criteria,
    so it can report password auth disabled while a real connection from
    some address would not be (see B2). Each `-C` spec also carries
    `laddr`/`lport` so a `Match LocalPort`/`LocalAddress` criterion is
    exercised too, not just `Match Address`. Refuses if ANY context is
    unsafe, or if sshd would listen on anything other than the requested
    port (see M8). This is an INDEPENDENT check from the static Match-scan
    in `_config_safety_refusal` — it does not assume that scan was
    complete."""
    PRIVSEP_DIR.mkdir(parents=True, exist_ok=True)
    sshd_bin = _sshd_binary()

    r = _run([sshd_bin, "-t"])
    if r.returncode != 0:
        return False, {}, f"sshd -t failed: {(r.stderr or r.stdout).strip()}"

    contexts = [("", [sshd_bin, "-T"])]
    container_addrs = _container_addresses()
    for _label, addr in _check_context_specs(container_addrs):
        laddr = _laddr_for(addr)
        for user in _CHECK_USERS:
            spec = f"user={user},host=x,addr={addr},laddr={laddr},lport={port}"
            contexts.append((f" -C(addr={addr},user={user})", [sshd_bin, "-T", "-C", spec]))

    last_effective: dict[str, list[str]] = {}
    checked = 0
    for context, cmd in contexts:
        r2 = _run(cmd)
        if r2.returncode != 0:
            return False, last_effective, (
                f"sshd -T{context} failed: {(r2.stderr or r2.stdout).strip()}"
            )
        effective = _parse_effective(r2.stdout)
        last_effective = effective
        checked += 1
        problem = _check_effective(effective, port, context)
        if problem:
            return False, effective, problem

    return True, last_effective, (
        f"sshd -T confirms password authentication is disabled (and "
        f"PermitRootLogin is prohibit-password) across {checked} "
        f"contexts (the default context, plus -C for loopback, every "
        f"RFC1918/CGNAT private range, link-local, public v4/v6, and "
        f"this container's own address(es), each for root and a "
        f"non-root user), and sshd would listen only on the configured "
        f"port"
    )


# ---------------------------------------------------------------------------
# Daemon lifecycle — found by pidfile, never a blanket pkill. "Started" is
# decided by what is REALLY listening on IPv4, never sshd's own exit
# status (H6).
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


def _sshd_listening_on_port(port: int) -> bool:
    """True only if something is really LISTENING (TCP state 0A) on this
    port over IPv4, read from the kernel's own /proc/net/tcp — never
    inferred from sshd's own exit status, which returns 0 even on a
    partial bind failure (confirmed: IPv6 up, IPv4 — the interface RunPod
    actually proxies — down). /proc/net/tcp always exists on every Linux
    kernel and needs no extra package; `ss` was checked against the real
    image and is NOT guaranteed present, so it is not relied on here."""
    try:
        lines = PROC_NET_TCP.read_text().splitlines()[1:]
    except OSError:
        return False
    target = f"{port:04X}"
    for line in lines:
        parts = line.split()
        if len(parts) < 4:
            continue
        local_addr = parts[1]
        state = parts[3]
        if ":" not in local_addr:
            continue
        port_hex = local_addr.rsplit(":", 1)[1]
        if port_hex.upper() == target and state.upper() == "0A":
            return True
    return False


def _wait_until_listening(port: int, timeout_s: float = 5.0) -> bool:
    deadline = time.time() + timeout_s
    while True:
        if _sshd_listening_on_port(port):
            return True
        if time.time() >= deadline:
            return False
        time.sleep(0.1)


# ---------------------------------------------------------------------------
# N3 — "listening" must mean OUR sshd, not any process on the port. Maps a
# real LISTEN socket's kernel inode (from /proc/net/tcp[6]) to the pid that
# actually holds it open (by scanning /proc/<pid>/fd for a `socket:[inode]`
# symlink), so "started"/"already running" can be tied to the daemon this
# script itself manages (by pidfile) rather than to whatever happens to be
# bound to the port.
# ---------------------------------------------------------------------------

def _listening_sockets_on_port(port: int) -> list[tuple[str, str]]:
    """[(family, inode)] for every row in LISTEN state (`0A`) on this port
    across both /proc/net/tcp (IPv4) and /proc/net/tcp6 (IPv6) — the same
    kernel-global truth _sshd_listening_on_port reads, but keeping the
    inode (column 10, `parts[9]`) instead of throwing it away, so the
    caller can attribute the socket to a pid."""
    out: list[tuple[str, str]] = []
    target = f"{port:04X}"
    for family, path in (("tcp4", PROC_NET_TCP), ("tcp6", PROC_NET_TCP6)):
        try:
            lines = path.read_text().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            parts = line.split()
            if len(parts) < 10:
                continue
            local_addr, state, inode = parts[1], parts[3], parts[9]
            if ":" not in local_addr:
                continue
            port_hex = local_addr.rsplit(":", 1)[1]
            if port_hex.upper() == target and state.upper() == "0A":
                out.append((family, inode))
    return out


def _proc_comm(pid: int) -> str | None:
    try:
        return (PROC_DIR / str(pid) / "comm").read_text().strip()
    except OSError:
        return None


def _proc_exe(pid: int) -> str | None:
    try:
        return os.readlink(str(PROC_DIR / str(pid) / "exe"))
    except OSError:
        return None


def _proc_pidfile_arg(pid: int) -> str | None:
    """Best-effort: the `-o PidFile=...` value from this pid's own argv,
    read from /proc/<pid>/cmdline — used only to explain, in a refusal
    message, that a stray sshd master is using a DIFFERENT pidfile than
    the one this script manages (N3)."""
    try:
        raw = (PROC_DIR / str(pid) / "cmdline").read_bytes()
    except OSError:
        return None
    toks = [t.decode("utf-8", errors="replace") for t in raw.split(b"\0") if t]
    for tok in toks:
        if tok.startswith("PidFile="):
            return tok.split("=", 1)[1]
    return None


def _pid_owning_inode(inode: str) -> int | None:
    """Scan every /proc/<pid>/fd for a `socket:[<inode>]` symlink and
    return the owning pid, or None if it cannot be attributed (a pid this
    process cannot read into, or a fd-table race — never fatal, just
    treated as "unattributable" by the caller, which refuses on that too:
    the safe default)."""
    target = f"socket:[{inode}]"
    try:
        pid_names = [p.name for p in PROC_DIR.iterdir() if p.name.isdigit()]
    except OSError:
        return None
    for pid_s in pid_names:
        fd_dir = PROC_DIR / pid_s / "fd"
        try:
            fd_entries = list(fd_dir.iterdir())
        except OSError:
            continue
        for fd in fd_entries:
            try:
                link = os.readlink(str(fd))
            except OSError:
                continue
            if link == target:
                return int(pid_s)
    return None


def _listener_owner_pids(port: int) -> list[int]:
    """Deduped pids that really own a LISTEN socket on this port right
    now, per the kernel's own inode->fd mapping. `-1` stands in for a
    listener whose inode could not be attributed to any pid — refusing on
    that is the safe default, never silently ignored."""
    owners = set()
    for _family, inode in _listening_sockets_on_port(port):
        pid = _pid_owning_inode(inode)
        owners.add(pid if pid is not None else -1)
    return sorted(owners)


def _describe_foreign_listener(pid: int) -> str:
    if pid == -1:
        return "an unattributable listener (socket inode could not be mapped to any /proc/<pid>/fd)"
    comm = _proc_comm(pid)
    exe = _proc_exe(pid)
    if comm == "sshd":
        other_pidfile = _proc_pidfile_arg(pid)
        return (
            f"another sshd master (pid {pid}, exe={exe}) using pidfile "
            f"{other_pidfile or 'unknown'} != {SSHD_PIDFILE} — a stray, "
            f"possibly password-enabled sshd must never be reported as "
            f"hardened"
        )
    return f"pid {pid} (comm={comm!r}, exe={exe!r})"


def _port_ownership_refusal(port: int) -> str | None:
    """N3: 'listening' must mean OUR sshd. Checked before this script
    ever writes anything (step 0) — the OLD check
    (`_sshd_listening_on_port`) only asked whether ANY process was bound
    to the port; it never asked WHICH one, so a foreign listener (a
    completely unrelated process, or another sshd master with a
    different pidfile — e.g. a stray, password-enabled one) was silently
    treated as this script's own daemon once it appeared to be
    listening. Returns a refusal naming every foreign owner, or None if
    the port is unoccupied or occupied only by the sshd this script
    itself already manages (its own pidfile's live pid)."""
    owners = _listener_owner_pids(port)
    if not owners:
        return None
    ours = _sshd_pid_if_alive()
    foreign = [p for p in owners if p != ours]
    if not foreign:
        return None
    return (
        f"refusing — port {port} already has a real LISTEN socket owned "
        f"by something other than the sshd this script manages "
        f"({SSHD_PIDFILE}): " + "; ".join(_describe_foreign_listener(p) for p in foreign)
    )


def _confirm_our_listener(port: int) -> tuple[bool, str]:
    """After a start/restart: not just 'is port <port> LISTEN' (H6) but
    'is THIS pidfile's sshd the one that owns it' (N3) — map the
    listening socket's inode to a pid and require that pid to be the one
    recorded in SSHD_PIDFILE, with /proc/<pid>/exe really naming sshd."""
    owners = _listener_owner_pids(port)
    if not owners:
        return False, f"nothing is really LISTENING on port {port}"
    pid = _sshd_pid_if_alive()
    if pid is None:
        return False, (
            f"port {port} has a real LISTEN socket, but this script's own "
            f"pidfile ({SSHD_PIDFILE}) names no live sshd — refusing "
            f"rather than trust a listener it cannot attribute to the "
            f"daemon it started"
        )
    if pid not in owners:
        return False, (
            f"port {port} is listening, but not by pid {pid} from "
            f"{SSHD_PIDFILE} — real owner pid(s): {owners} — refusing "
            f"rather than trust a listener this script cannot attribute "
            f"to the daemon it started"
        )
    # No separate /proc/<pid>/exe name check here: `_sshd_pid_if_alive`
    # already required /proc/<pid>/comm == "sshd" to return this pid at
    # all (the same convention the rest of this script uses — including
    # in tests, where the fake sshd is a Python script that renames
    # itself via prctl(PR_SET_NAME); its own /proc/<pid>/exe legitimately
    # names the python3 interpreter, not a file called "sshd").
    return True, f"confirmed pid {pid} (from {SSHD_PIDFILE}) owns the real LISTEN socket on port {port}"


def _wait_until_confirmed(port: int, timeout_s: float = 5.0) -> tuple[bool, str]:
    """H6+N3: 'started' means a REAL LISTEN socket on this port whose
    owning pid — mapped via /proc/net/tcp[6]'s inode -> /proc/<pid>/fd,
    never sshd's own exit status — is the daemon THIS script started,
    read fresh from SSHD_PIDFILE on every poll."""
    deadline = time.time() + timeout_s
    detail = f"nothing is really LISTENING on port {port}"
    while True:
        ok, detail = _confirm_our_listener(port)
        if ok:
            return True, detail
        if time.time() >= deadline:
            return False, detail
        time.sleep(0.1)


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
    # _sshd_binary() (not a hardcoded "/usr/sbin/sshd") so a test's fake
    # bin dir is the one supervisord's own conf ends up pointing at too —
    # in production shutil.which("sshd") resolves to the same absolute
    # path anyway, so this changes nothing there.
    return (
        f"\n{SUP_BEGIN_MARK}\n"
        f"[program:sshd]\n"
        f"command={_sshd_binary()} -D -e -o PidFile={SSHD_PIDFILE}\n"
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
    if not SUPERVISOR_CONF.is_file():
        # M8: this used to reach shutil.copy2 inside _make_backup with a
        # file that does not exist at all, raising an unhandled
        # FileNotFoundError (no --json output at all). Refuse cleanly
        # instead — there is nothing sensible to append a program to.
        return {
            "already_present": False, "ok": False, "backup_path": None,
            "detail": (
                f"--supervise refuses: {SUPERVISOR_CONF} does not exist — "
                f"supervisord is not configured on this container, so "
                f"appending a [program:sshd] section would create a file "
                f"supervisord was never told to read; run without "
                f"--supervise (the plain background daemon) instead"
            ),
        }

    text = SUPERVISOR_CONF.read_text(encoding="utf-8", errors="replace")
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
        "authorized_keys_file": None,
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

    # --- 0. B2 / M8 / N2 / N3: config-safety refusal, BEFORE any write
    # this run makes. Nothing has been written yet, so no rollback is
    # needed here.
    cfg_refusal = _config_safety_refusal(args.port)
    if cfg_refusal:
        refusals.append(cfg_refusal)
        return _finish(args, report, refusals, warnings)

    # --- 0b. N11/N15: refuse outright on a symlinked .ssh/authorized_keys,
    # before ever writing or chmodding anything.
    symlink_refusal = _symlink_refusal()
    if symlink_refusal:
        refusals.append(symlink_refusal)
        return _finish(args, report, refusals, warnings)

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

    # --- 2b. N11: refuse if sshd would never even READ the file this
    # script writes keys to (a non-default AuthorizedKeysFile). Checked
    # whenever the sshd binary already exists (always true in --apply by
    # this point; in a dry run only if it was already installed before
    # this run — otherwise there is nothing to ask sshd -T yet, same as
    # the config-verification note below).
    sshd_present = shutil.which("sshd") is not None or Path("/usr/sbin/sshd").is_file()
    if sshd_present:
        raw_akf, akf_err = _effective_authorized_keys_file()
        if akf_err:
            refusals.append(akf_err)
            return _finish(args, report, refusals, warnings)
        report["authorized_keys_file"] = raw_akf
        if not _authorized_keys_file_is_default(raw_akf):
            refusals.append(
                f"refusing — the effective AuthorizedKeysFile is "
                f"{raw_akf!r}, not the default this script writes to "
                f"({AUTHORIZED_KEYS}) — sshd would never read the key "
                f"this script adds (N11)"
            )
            return _finish(args, report, refusals, warnings)

    # --- 3. resolve a public key -----------------------------------------
    key_source, candidates, key_refusals = _resolve_key_source(args)
    if key_refusals:
        refusals.extend(key_refusals)
        return _finish(args, report, refusals, warnings)
    report["key_source"] = key_source

    key_backup: str | None = None
    key_created_fresh = False
    if args.apply:
        # N11/N15: _append_keys raises RuntimeError only in the
        # never-observed TOCTOU race its own docstring describes (a
        # symlink appearing between _symlink_refusal()'s check, above,
        # and this call) — caught here so that race still produces a
        # clean --json refusal instead of an unhandled traceback.
        try:
            keys_added, key_backup, key_created_fresh = _append_keys(candidates)
        except RuntimeError as e:
            refusals.append(str(e))
            return _finish(args, report, refusals, warnings)
        if key_backup:
            report["authorized_keys_backup"] = key_backup
        # H7: enforce .ssh / authorized_keys permissions & ownership every
        # apply, independent of whether anything was actually added.
        warnings.extend(_harden_ssh_dir_permissions())
    else:
        keys_added = _planned_new_keys(candidates)
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
        # NOTE: host keys under HOST_KEY_STORE are never rolled back below
        # — see ROLLBACK in the module docstring.
        host_info = _ensure_host_keys(persist=persist_host_keys)
        report["host_keys"] = host_info["host_keys"]
        warnings.extend(host_info["notes"])

        # --- 6. verify for real, refuse+rollback if it fails ------------
        ok, effective, detail = _sshd_effective_config(args.port)
        report["sshd_check"] = detail
        if not ok:
            if plan["changed"]:
                _rollback_config(plan)
            _rollback_authorized_keys(key_backup, key_created_fresh)
            refusals.append(detail)
            return _finish(args, report, refusals, warnings)

        # --- 7. start / restart, by pid only, verified by REAL listening
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
                # already in use"). H5: if the handoff then fails, this
                # standalone daemon is restarted before refusing, so the
                # pod is never left with no sshd because of this script.
                stopped_pid = None
                if pid is not None:
                    if _stop_sshd(pid):
                        stopped_pid = pid
                    else:
                        warnings.append(
                            f"WARNING: a standalone sshd (pid {pid}) did not "
                            f"exit within the timeout before handing off to "
                            f"supervisord"
                        )
                        stopped_pid = pid

                sup = _ensure_supervised(args.port, apply=True)
                report["supervise"] = sup

                def _recover_standalone(reason: str) -> None:
                    if plan["changed"]:
                        _rollback_config(plan)
                    _rollback_authorized_keys(key_backup, key_created_fresh)
                    if stopped_pid is not None:
                        rr = _start_sshd()
                        _confirmed, _ = (
                            (False, "") if rr.returncode != 0
                            else _wait_until_confirmed(args.port)
                        )
                        if _confirmed:
                            report["daemon"] = "restarted"
                            warnings.append(
                                f"{reason}; the standalone sshd was "
                                f"restarted and is listening again"
                            )
                        else:
                            report["daemon"] = "not-started"
                            warnings.append(
                                f"CRITICAL: {reason} AND the standalone sshd "
                                f"could not be restarted — the pod may have "
                                f"NO sshd listening; use the RunPod Web "
                                f"Terminal to investigate"
                            )
                    else:
                        report["daemon"] = "not-started"

                if not sup["ok"]:
                    _recover_standalone("the supervisord handoff failed")
                    refusals.append(sup["detail"])
                    return _finish(args, report, refusals, warnings)

                confirmed, confirm_detail = _wait_until_confirmed(args.port)
                if not confirmed:
                    _recover_standalone(
                        "supervisorctl update completed but sshd never "
                        "came up listening"
                    )
                    refusals.append(
                        f"supervisorctl update completed but sshd is not "
                        f"confirmed as the owner of a real LISTEN socket on "
                        f"IPv4 port {args.port} ({confirm_detail}) — refusing"
                    )
                    return _finish(args, report, refusals, warnings)

                report["daemon"] = "restarted" if pid is not None else "started"
            else:
                report["supervise"] = {
                    "already_present": True, "ok": True, "backup_path": None,
                    "detail": "[program:sshd] already present in supervisord.conf",
                }
                restarted = False
                if did_install or plan["changed"]:
                    r = _run(["supervisorctl", "-c", str(SUPERVISOR_CONF), "restart", "sshd"])
                    time.sleep(1.0)
                    restarted = True
                    confirmed, confirm_detail = (
                        (False, "") if r.returncode != 0
                        else _wait_until_confirmed(args.port)
                    )
                    if not confirmed:
                        if plan["changed"]:
                            _rollback_config(plan)
                        _rollback_authorized_keys(key_backup, key_created_fresh)
                        report["daemon"] = "not-started"
                        refusals.append(
                            f"'supervisorctl restart sshd' did not leave "
                            f"sshd confirmed as the owner of a real LISTEN "
                            f"socket on IPv4 port {args.port} "
                            f"({confirm_detail}): {(r.stderr or r.stdout).strip()}"
                        )
                        return _finish(args, report, refusals, warnings)
                    report["daemon"] = "restarted"

                if not restarted:
                    if pid is not None:
                        confirmed, confirm_detail = _wait_until_confirmed(args.port)
                        if not confirmed:
                            refusals.append(
                                f"sshd is supposed to already be running "
                                f"under supervisord but is not confirmed as "
                                f"the owner of a real LISTEN socket on IPv4 "
                                f"port {args.port} ({confirm_detail}) — "
                                f"refusing"
                            )
                            report["daemon"] = "not-started"
                            return _finish(args, report, refusals, warnings)
                    if pid is None:
                        refusals.append(
                            "supervisord believes [program:sshd] is already "
                            "present but no sshd pid/listener was found — "
                            "refusing (check `supervisorctl status`)"
                        )
                        report["daemon"] = "not-started"
                        return _finish(args, report, refusals, warnings)
                    report["daemon"] = "already-running"
        elif pid is None:
            r = _start_sshd()
            if r.returncode != 0:
                if plan["changed"]:
                    _rollback_config(plan)
                _rollback_authorized_keys(key_backup, key_created_fresh)
                refusals.append(f"sshd failed to start: {(r.stderr or r.stdout).strip()}")
                return _finish(args, report, refusals, warnings)
            confirmed, confirm_detail = _wait_until_confirmed(args.port)
            if not confirmed:
                # It came up (pidfile written, process alive) but never
                # bound the port, or the port is held by something else
                # entirely (N3) — a dead-end/misattributed daemon is worse
                # than none: stop OUR pid too, so no orphaned
                # pidfile/process persists.
                bogus_pid = _sshd_pid_if_alive()
                if bogus_pid is not None:
                    _stop_sshd(bogus_pid)
                if plan["changed"]:
                    _rollback_config(plan)
                _rollback_authorized_keys(key_backup, key_created_fresh)
                report["daemon"] = "not-started"
                refusals.append(
                    f"sshd started (process alive) but is not confirmed as "
                    f"the owner of a real LISTEN socket on IPv4 port "
                    f"{args.port} (checked /proc/net/tcp[6] -> "
                    f"/proc/<pid>/fd, not just sshd's own exit status: "
                    f"{confirm_detail}) — refusing"
                )
                return _finish(args, report, refusals, warnings)
            report["daemon"] = "started"
        elif did_install or plan["changed"]:
            if not _stop_sshd(pid):
                warnings.append(
                    f"WARNING: sshd (pid {pid}) did not exit within the timeout; "
                    f"starting a new instance anyway"
                )
            r = _start_sshd()
            confirmed, confirm_detail = (
                (False, "") if r.returncode != 0 else _wait_until_confirmed(args.port)
            )
            if not confirmed:
                # H5/H6/N3: the OLD daemon is already down. One
                # best-effort attempt to bring SOME sshd back before
                # refusing, even though it will be using the (about to be
                # rolled back) old config.
                if plan["changed"]:
                    _rollback_config(plan)
                _rollback_authorized_keys(key_backup, key_created_fresh)
                r2 = _start_sshd()
                confirmed2, _ = (
                    (False, "") if r2.returncode != 0 else _wait_until_confirmed(args.port)
                )
                if confirmed2:
                    report["daemon"] = "restarted"
                    warnings.append(
                        "the restart with the new config did not come up "
                        "listening; restarted again with the rolled-back "
                        "config instead — sshd is still listening"
                    )
                else:
                    report["daemon"] = "not-started"
                    warnings.append(
                        "CRITICAL: could not get sshd listening again after "
                        "a failed restart — the pod may have NO sshd "
                        "listening; use the RunPod Web Terminal to investigate"
                    )
                refusals.append(
                    f"sshd failed to restart or is not confirmed as the "
                    f"owner of a real LISTEN socket on port {args.port} "
                    f"({confirm_detail}): {(r.stderr or r.stdout).strip()}"
                )
                return _finish(args, report, refusals, warnings)
            report["daemon"] = "restarted"
        else:
            confirmed, confirm_detail = _wait_until_confirmed(args.port)
            if not confirmed:
                refusals.append(
                    f"sshd is believed already-running (pid {pid}) but is "
                    f"not confirmed as the owner of a real LISTEN socket on "
                    f"IPv4 port {args.port} ({confirm_detail}) — refusing"
                )
                report["daemon"] = "not-started"
                return _finish(args, report, refusals, warnings)
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
                "effective config would allow password authentication or "
                "listen on the wrong port."
            )
        if args.supervise:
            sup = _ensure_supervised(args.port, apply=False)
            report["supervise"] = sup
            if not sup["ok"]:
                refusals.append(sup["detail"])

    return _finish(args, report, refusals, warnings)


if __name__ == "__main__":
    sys.exit(main())
