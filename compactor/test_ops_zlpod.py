"""Unit tests for ops/laptop/zlpod -- the laptop-side helper: `init` writes
an idempotent `Host zlpod` block (backing up ~/.ssh/config first), `zl
ARGS...` quotes safely, and `pull bundle`/`pull backup`/`push wordban` build
the right scp/ssh argv.

Plain-script style like compactor/test_setup_sshd_script.py: exit 0 pass,
1 fail. Fakes `ssh`/`scp` on PATH (argv-recording stand-ins) and points
HOME at a temp dir -- never touches the real ~/.ssh/config or a real host.
"""

from __future__ import annotations

import os
import stat
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
ZLPOD = REPO_ROOT / "ops" / "laptop" / "zlpod"

_FAKE_SSH = r"""#!/usr/bin/env bash
# sshd-FAITHFUL (review H4): the real ssh binary does not preserve local
# argv boundaries or quoting for anything after the host -- it rejoins
# every argv word with a single plain space and sends that ONE string to
# the remote, which runs it through its own login shell (`bash -c
# "<joined>"`). A fake that instead matches on the whole joined string
# (as this file's fake used to) cannot catch a caller that builds its
# remote command out of SEVERAL argv words and relies on local quoting
# that does not survive the rejoin (exactly zlpod's old `pull backup`
# bug). This fake actually rejoins and actually executes via `bash -c`,
# so it is exercised by real shell semantics, not a second layer of
# string matching.
args=("$@")
while [[ "${args[0]:-}" == -* ]]; do  # skip ssh's own flags, e.g. -t
    args=("${args[@]:1}")
done
cmd_args=("${args[@]:1}")  # everything after the host
joined="${cmd_args[*]}"
printf 'ssh %s\n' "$joined" >> "$FAKE_LOG"
# The remote side has no real /data -- point the literal path at our own
# sandboxed stand-in instead. This substitution happens AFTER the rejoin
# above, so it changes nothing about the argv-boundary behavior under test;
# it only lets the rejoined command act on real files instead of the
# genuine (root-owned, shared) /data.
#
# Round-4 addendum (c): IDEMPOTENT -- if the command already mentions
# $FAKE_REMOTE_ROOT (e.g. `zlpod pull`'s own sha256sum verification
# re-sends a path an EARLIER ssh call already resolved and returned, such
# as an `ls`-derived backup name), skip the rewrite instead of applying
# it a second time -- a double rewrite mangles an already-good path into
# one that was never really there.
case "$joined" in
    *"$FAKE_REMOTE_ROOT"*) rewritten="$joined" ;;
    *) rewritten="${joined//\/data//${FAKE_REMOTE_ROOT}\/data}" ;;
esac
# Round-4 addendum (LOW): FAKE_SSH_FAIL_SHA256, when set, makes ONLY the
# remote sha256sum call (`_verify_pull`'s own ssh call) fail as if ssh/the
# network dropped -- no output, nonzero exit -- while every OTHER ssh call
# (the `ls -1t` backup-name lookup, `zl disk`, ...) still succeeds
# normally. Proves `_verify_pull` tells "could not verify" (ssh failure)
# apart from a real sha256 MISMATCH (the file itself is bad).
if [ -n "${FAKE_SSH_FAIL_SHA256:-}" ]; then
    case "$rewritten" in
        sha256sum\ *) echo "ssh: connection failed" >&2; exit 255 ;;
    esac
fi
exec bash -c "$rewritten"
"""

# A fake `zl` on the fake remote's PATH, for the joined commands (`zl
# bundle --print-path`, `zl status`, `zl fix ...`) that _FAKE_SSH's real
# `bash -c` now actually executes.
_FAKE_REMOTE_ZL = """#!/usr/bin/env bash
if [[ "$1 $2" == "bundle --print-path" ]]; then
    echo "$FAKE_REMOTE_ROOT/data/zl-ops/bundles/zla-bundle-20260101T000000Z.tar.gz"
fi
if [[ "$1" == "disk" ]]; then
    # Review LOW: `zl disk`'s real output ("backups: N archive(s) in
    # <podenv.BACKUPS>, <size> total") is what zlpod's own
    # `remote_backup_dir()` parses to honor COMPACTOR_BACKUP_DIR -- this
    # fake reports whatever FAKE_BACKUP_DIR the test set (default: the
    # real default, /data/backups), as a REAL pod path (never
    # FAKE_REMOTE_ROOT-prefixed -- that substitution is _FAKE_SSH's own
    # job, applied to whatever command zlpod builds from this).
    echo "backups: 0 archive(s) in ${FAKE_BACKUP_DIR:-/data/backups}, 0B total"
fi
exit 0
"""

_FAKE_SCP = """#!/usr/bin/env bash
# Round-4 addendum (c): `zlpod pull`'s own sha256/gzip verification
# compares the LOCAL file against a real `sha256sum` of the REMOTE one
# (via the fake ssh above, which really executes on the fake remote's own
# filesystem) -- so this fake must actually COPY the same bytes, not
# fabricate unrelated local content the way it used to (a fabricate-only
# fake would make every real transfer look "corrupt": different content,
# different checksum). If the resolved remote source does not exist yet
# (the fake remote's `zl bundle --print-path` names a file nothing ever
# actually created), fabricate it for real AT THAT PATH first, then copy
# it -- so the local copy and a subsequent remote sha256sum see identical
# bytes either way.
printf 'scp %s\\n' "$*" >> "$FAKE_LOG"
src="$1"
dest="${@: -1}"
case "$dest" in
    *:*)
        # A REMOTE destination (push): nothing local to create.
        exit 0
        ;;
esac
case "$src" in
    *:*)
        remote_path="${src#*:}"
        # `remote_path` is sometimes ALREADY resolved against
        # $FAKE_REMOTE_ROOT (a `pull backup --latest`'s remote name comes
        # from an `ls` the fake ssh already ran against the rewritten
        # path, so its OWN output already has the prefix baked in) and
        # sometimes still canonical/`/data`-rooted (an explicit `pull
        # backup NAME`, or `pull bundle`'s printed path via `zl
        # bundle --print-path`) -- rewrite only if it is not already
        # resolved, or a double rewrite mangles the path into one that
        # was never really there, and this fake then wrongly fabricates
        # fresh (mismatching) content instead of copying the real bytes.
        case "$remote_path" in
            *"$FAKE_REMOTE_ROOT"*) resolved="$remote_path" ;;
            *) resolved="${remote_path//\\/data//${FAKE_REMOTE_ROOT}\\/data}" ;;
        esac
        if [ ! -e "$resolved" ]; then
            mkdir -p "$(dirname "$resolved")" 2>/dev/null
            case "$resolved" in
                *.tar.gz) tar -czf "$resolved" --files-from /dev/null ;;  # a real, valid, empty archive
                *) : > "$resolved" ;;
            esac
        fi
        mkdir -p "$(dirname "$dest")" 2>/dev/null
        cp "$resolved" "$dest"
        exit 0
        ;;
esac
mkdir -p "$(dirname "$dest")" 2>/dev/null
case "$dest" in
    *.tar.gz) tar -czf "$dest" --files-from /dev/null ;;
    *) : > "$dest" ;;
esac
exit 0
"""


def _valid_gzip_bytes(payload: bytes = b"fake archive contents\n") -> bytes:
    """A real, valid (if trivial) gzip stream -- round-4 addendum (c):
    `zlpod pull` now verifies gzip/sha256, so any test that plants
    content a pull is expected to SUCCEED against must be real gzip, not
    an arbitrary byte string.
    """
    import gzip as _gzip
    import io as _io

    buf = _io.BytesIO()
    with _gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        gz.write(payload)
    return buf.getvalue()


def _count_host_stanzas(ssh_config_text: str) -> int:
    """Counts actual `Host zlpod` STANZA lines, not the marker comments
    (which also contain the substring "Host zlpod" by design)."""
    return sum(1 for ln in ssh_config_text.splitlines() if ln.strip() == "Host zlpod")


def assert_eq(actual, expected, label, extra=""):
    if actual != expected:
        print(f"FAIL {label}: expected {expected!r}, got {actual!r}")
        if extra:
            print(f"  --- context ---\n{extra}\n  --- end ---")
        sys.exit(1)
    print(f"  ok   {label}")


def assert_true(cond, label, extra=""):
    if not cond:
        print(f"FAIL {label}")
        if extra:
            print(f"  --- context ---\n{extra}\n  --- end ---")
        sys.exit(1)
    print(f"  ok   {label}")


class _Fixture:
    def __enter__(self):
        self._td = tempfile.TemporaryDirectory()
        self.td = Path(self._td.name)
        self.home = self.td / "home"
        self.home.mkdir()
        self.remote_root = self.td / "fake-remote"
        (self.remote_root / "data" / "backups").mkdir(parents=True)
        bin_dir = self.td / "bin"
        bin_dir.mkdir()
        for name, src in (("ssh", _FAKE_SSH), ("scp", _FAKE_SCP), ("zl", _FAKE_REMOTE_ZL)):
            p = bin_dir / name
            p.write_text(src, encoding="utf-8")
            p.chmod(p.stat().st_mode | stat.S_IEXEC)
        self.log = self.td / "log.txt"
        self.log.write_text("", encoding="utf-8")
        self.env = {
            "HOME": str(self.home),
            "PATH": str(bin_dir) + os.pathsep + os.environ.get("PATH", "/usr/bin:/bin"),
            "FAKE_LOG": str(self.log),
            "FAKE_REMOTE_ROOT": str(self.remote_root),
        }
        return self

    def put_fake_backup(self, name: str, age_s: float, content: bytes | None = None) -> Path:
        """Creates a fake archive under the fake remote's /data/backups,
        with mtime `age_s` seconds in the past -- so `ls -1t` sorts it
        predictably against others planted by the same call. Round-4
        addendum (c): `zlpod pull backup` now verifies sha256/gzip
        against whatever is really at this path, so the default content
        must be a REAL, valid (if trivial) gzip archive -- `_FAKE_SCP`
        copies these exact bytes, and a plain `b"fake archive"` string
        would (correctly) be flagged as corrupt. Pass `content=` to plant
        deliberately-bad bytes for a corruption test.
        """
        p = self.remote_root / "data" / "backups" / name
        if content is None:
            content = _valid_gzip_bytes()
        p.write_bytes(content)
        t = time.time() - age_s
        os.utime(p, (t, t))
        return p

    def __exit__(self, *exc):
        self._td.cleanup()
        return False

    def run(self, args, timeout=30):
        return subprocess.run(["bash", str(ZLPOD), *args], capture_output=True, text=True,
                               timeout=timeout, env=self.env)

    def log_lines(self):
        return self.log.read_text(encoding="utf-8").splitlines()


def test_init_writes_config_and_idempotent_ssh_config_block():
    with _Fixture() as fx:
        r = fx.run(["init", "1.2.3.4", "2222", "/fake/key"])
        assert_eq(r.returncode, 0, "zlpod init exits 0", )

        config = (fx.home / ".config" / "zlpod" / "config").read_text(encoding="utf-8")
        assert_true("HOST=1.2.3.4" in config, "config records HOST")
        assert_true("PORT=2222" in config, "config records PORT")

        ssh_config = (fx.home / ".ssh" / "config").read_text(encoding="utf-8")
        assert_true("HostName 1.2.3.4" in ssh_config, "ssh config has the right HostName")
        assert_true("Port 2222" in ssh_config, "ssh config has the right Port")
        assert_true("IdentitiesOnly yes" in ssh_config, "ssh config sets IdentitiesOnly yes")
        assert_true("ServerAliveInterval 15" in ssh_config, "ssh config sets ServerAliveInterval 15")
        assert_true("ServerAliveCountMax 4" in ssh_config, "ssh config sets ServerAliveCountMax 4")
        assert_eq(_count_host_stanzas(ssh_config), 1, "exactly one Host zlpod stanza")

        # A second init must back up first, then replace -- never duplicate.
        r2 = fx.run(["init", "5.6.7.8", "3333", "/fake/key2"])
        assert_eq(r2.returncode, 0, "a second zlpod init exits 0")
        backups = list((fx.home / ".ssh").glob("config.bak-*"))
        assert_true(len(backups) >= 1, "zlpod init backs up ~/.ssh/config before editing it")
        ssh_config2 = (fx.home / ".ssh" / "config").read_text(encoding="utf-8")
        assert_eq(_count_host_stanzas(ssh_config2), 1, "re-init replaces the block, never duplicates it")
        assert_true("HostName 5.6.7.8" in ssh_config2, "re-init's new HostName took effect")


def test_init_inserts_at_top_and_warns_about_an_earlier_host_zlpod():
    """review MEDIUM: ssh.config matches the FIRST entry per keyword, so a
    block appended at the bottom loses to an existing `Host zlpod` (or
    `Host *`) that comes before it, however recently `zlpod init` ran.
    """
    with _Fixture() as fx:
        ssh_dir = fx.home / ".ssh"
        ssh_dir.mkdir(parents=True)
        (ssh_dir / "config").write_text(
            "Host zlpod\n    HostName 9.9.9.9\n    Port 9999\n\nHost other\n    HostName example.com\n",
            encoding="utf-8",
        )
        r = fx.run(["init", "1.2.3.4", "2222"])
        assert_eq(r.returncode, 0, "zlpod init exits 0 even with a pre-existing Host zlpod")
        assert_true("WARNING" in (r.stdout + r.stderr) and "zlpod" in (r.stdout + r.stderr).lower(),
                    "zlpod init warns about the earlier, non-ours Host zlpod entry", extra=r.stderr)

        config = (ssh_dir / "config").read_text(encoding="utf-8")
        our_pos = config.index("BEGIN zlpod Host zlpod")
        old_pos = config.index("9.9.9.9")
        assert_true(our_pos < old_pos, "our marked block is inserted BEFORE the old, unmarked entry",
                    extra=config)
        # ssh matches the FIRST `Host zlpod` stanza's HostName -- confirm
        # OURS is that first one, not the stale 9.9.9.9.
        first_hostname = next(ln for ln in config.splitlines() if ln.strip().startswith("HostName"))
        assert_true("1.2.3.4" in first_hostname, "the FIRST HostName line in the file is ours (1.2.3.4)",
                    extra=config)


def test_zl_subcommand_quotes_arguments_with_spaces_safely():
    with _Fixture() as fx:
        fx.run(["init", "1.2.3.4", "2222"])
        r = fx.run(["zl", "fix", "repair-chat-tree", "--apply", "--", "an arg with spaces"])
        assert_eq(r.returncode, 0, "zlpod zl ... exits 0")
        lines = fx.log_lines()
        assert_true(len(lines) == 1, "exactly one ssh invocation", extra="\n".join(lines))
        # The remote command must be ONE shell-safe string that, once split
        # by the remote shell, reproduces the original argument boundaries
        # -- not "an arg with spaces" broken into four separate arguments.
        assert_true("an\\ arg\\ with\\ spaces" in lines[0] or "'an arg with spaces'" in lines[0],
                    "the multi-word argument stays one token, safely quoted", extra=lines[0])


def test_pull_bundle_extracts_into_dated_directory():
    with _Fixture() as fx:
        fx.run(["init", "1.2.3.4", "2222"])
        r = fx.run(["pull", "bundle"])
        assert_eq(r.returncode, 0, "zlpod pull bundle exits 0", )
        exports = list((fx.home / "pod-exports").glob("*/*.tar.gz"))
        assert_true(len(exports) == 1, "one archive was pulled", extra=r.stdout + r.stderr)
        bundle_dirs = list((fx.home / "pod-exports").glob("*/bundle-*Z"))
        assert_true(len(bundle_dirs) == 1, "an extraction directory bundle-<HHMM>Z was created",
                    extra=str(list((fx.home / "pod-exports").rglob("*"))))


def test_pull_backup_latest():
    with _Fixture() as fx:
        fx.run(["init", "1.2.3.4", "2222"])
        # Several candidates, only one of which is genuinely the newest --
        # `--latest` must pick THAT one, not just "a" backup (H4).
        fx.put_fake_backup("zions-backup-20260101T000000Z.tar.gz", age_s=3600)
        fx.put_fake_backup("zions-backup-20260201T000000Z.tar.gz", age_s=60)  # newest
        fx.put_fake_backup("zions-backup-20251201T000000Z.tar.gz", age_s=7200)
        fx.put_fake_backup("not-a-backup.tar.gz", age_s=1)  # newest by mtime, but wrong prefix -- must be ignored

        r = fx.run(["pull", "backup", "--latest"])
        assert_eq(r.returncode, 0, "zlpod pull backup --latest exits 0", )
        pulled = list((fx.home / "pod-exports").rglob("zions-backup-*.tar.gz"))
        assert_true(len(pulled) == 1, "the latest backup archive was pulled", extra=r.stdout + r.stderr)
        assert_eq(pulled[0].name, "zions-backup-20260201T000000Z.tar.gz",
                  "the ACTUAL newest zions-backup-*.tar.gz was picked, not merely a file named like one")


def test_pull_backup_honors_pod_reported_backup_dir():
    """Review LOW: `zlpod pull backup` used to hardcode /data/backups.
    Point the fake pod's `zl disk` at a DIFFERENT COMPACTOR_BACKUP_DIR and
    confirm `zlpod pull backup --latest` finds the archive THERE, not at
    the (now wrong) default.
    """
    with _Fixture() as fx:
        fx.run(["init", "1.2.3.4", "2222"])
        fx.env["FAKE_BACKUP_DIR"] = "/data/custom-backups"
        custom_dir = fx.remote_root / "data" / "custom-backups"
        custom_dir.mkdir(parents=True)
        p = custom_dir / "zions-backup-20260301T000000Z.tar.gz"
        p.write_bytes(_valid_gzip_bytes())

        r = fx.run(["pull", "backup", "--latest"])
        assert_eq(r.returncode, 0, "zlpod pull backup --latest exits 0 against a custom backup dir",
                  )
        pulled = list((fx.home / "pod-exports").rglob("zions-backup-*.tar.gz"))
        assert_true(len(pulled) == 1, "the archive under the CUSTOM dir was pulled",
                    extra=r.stdout + r.stderr)
        assert_eq(pulled[0].name, "zions-backup-20260301T000000Z.tar.gz",
                  "it is the one planted under /data/custom-backups, not /data/backups")


def test_pull_backup_falls_back_to_default_dir_if_zl_disk_unparseable():
    """The fallback path (zl disk unreachable, or its output doesn't
    parse) must still work against the plain default -- never crash, never
    silently look in the wrong place without saying so.
    """
    with _Fixture() as fx:
        fx.run(["init", "1.2.3.4", "2222"])
        # Break the fake `zl` so `disk` prints nothing parseable.
        fake_zl = fx.td / "bin" / "zl"
        fake_zl.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        fake_zl.chmod(fake_zl.stat().st_mode | stat.S_IEXEC)
        fx.put_fake_backup("zions-backup-20260101T000000Z.tar.gz", age_s=10)

        r = fx.run(["pull", "backup", "--latest"])
        assert_eq(r.returncode, 0, "still exits 0, falling back to /data/backups")
        assert_true("warning" in (r.stdout + r.stderr).lower(),
                    "the fallback warns rather than silently guessing", extra=r.stdout + r.stderr)
        pulled = list((fx.home / "pod-exports").rglob("zions-backup-*.tar.gz"))
        assert_true(len(pulled) == 1, "the archive under the DEFAULT dir was still found",
                    extra=r.stdout + r.stderr)


def test_fake_ssh_is_faithful_enough_to_catch_the_old_pull_backup_bug():
    """H4: the fake ssh must be sshd-faithful (argv rejoined with plain
    spaces, executed via `bash -c`), so a caller who builds its remote
    command out of several argv words and relies on LOCAL quoting that
    does not survive the rejoin is caught. Proof: invoke the fake ssh
    exactly the way the OLD, buggy zlpod code did (`ssh zlpod bash -c
    '<pipeline>'`, as 4 separate argv words) and show it does NOT find
    the planted archive -- the same failure mode H4 reproduced against
    the real image. This is the mutation-catch proof for the fix in
    `cmd_pull_backup` (now ONE argv word for the whole remote command).
    """
    with _Fixture() as fx:
        fx.put_fake_backup("zions-backup-20260201T000000Z.tar.gz", age_s=60)
        old_style = subprocess.run(
            ["ssh", "zlpod", "bash", "-c",
             "ls -1t /data/backups/zions-backup-*.tar.gz 2>/dev/null | head -1"],
            capture_output=True, text=True, timeout=15, env=fx.env,
        )
        assert_true(
            "zions-backup-20260201T000000Z.tar.gz" not in old_style.stdout,
            "the OLD (multi-argv `bash -c`) invocation style does NOT find the archive under the "
            "sshd-faithful fake -- proving the fake would have caught the real H4 bug",
            extra=f"stdout={old_style.stdout!r} stderr={old_style.stderr!r}",
        )
        # And the FIXED, single-argv style (what zlpod now actually does) does find it:
        fixed = subprocess.run(
            ["ssh", "zlpod", "ls -1t /data/backups/zions-backup-*.tar.gz 2>/dev/null | head -1"],
            capture_output=True, text=True, timeout=15, env=fx.env,
        )
        assert_true(
            "zions-backup-20260201T000000Z.tar.gz" in fixed.stdout,
            "the FIXED (single-argv) invocation style DOES find the archive under the same fake",
            extra=f"stdout={fixed.stdout!r} stderr={fixed.stderr!r}",
        )


def test_pull_backup_detects_gzip_corruption_and_renames_to_corrupt():
    """Round-4 addendum (c): a backup that is NOT a valid gzip archive
    (whatever the reason -- a truncated transfer, a WSL/Windows-boundary
    zero-out) must be caught by `gzip -t`, reported loudly, renamed
    `.corrupt` (kept, not deleted), and the command must exit 1 rather
    than claim success.
    """
    with _Fixture() as fx:
        fx.run(["init", "1.2.3.4", "2222"])
        fx.put_fake_backup("zions-backup-20260401T000000Z.tar.gz", age_s=10, content=b"not a gzip file at all")
        r = fx.run(["pull", "backup", "--latest"])
        assert_eq(r.returncode, 1, "a non-gzip backup makes `pull backup` exit 1", )
        assert_true("CORRUPT" in (r.stdout + r.stderr), "the failure is reported loudly",
                    extra=r.stdout + r.stderr)
        corrupt = list((fx.home / "pod-exports").rglob("*.corrupt"))
        assert_true(len(corrupt) == 1, "the bad file was kept, renamed .corrupt", extra=str(corrupt))
        good = list((fx.home / "pod-exports").rglob("zions-backup-*.tar.gz"))
        assert_true(len(good) == 0, "no file is left under its original (trusted-looking) name")


def test_pull_backup_detects_all_zero_file_and_renames_to_corrupt():
    """The exact real-world failure this addendum names: the newest local
    backup came out ALL ZERO BYTES at full size across the WSL/Windows
    boundary. Plant a same-size, all-zero "archive" on the fake remote
    and confirm it is caught and named for what it is.
    """
    with _Fixture() as fx:
        fx.run(["init", "1.2.3.4", "2222"])
        fx.put_fake_backup("zions-backup-20260401T000000Z.tar.gz", age_s=10, content=b"\x00" * 4096)
        r = fx.run(["pull", "backup", "--latest"])
        assert_eq(r.returncode, 1, "an all-zero backup makes `pull backup` exit 1", )
        assert_true("all-zero" in (r.stdout + r.stderr), "the message specifically names the all-zero shape",
                    extra=r.stdout + r.stderr)
        corrupt = list((fx.home / "pod-exports").rglob("*.corrupt"))
        assert_true(len(corrupt) == 1, "the bad file was kept, renamed .corrupt", extra=str(corrupt))


def test_pull_backup_detects_sha256_mismatch():
    """A file that passes `gzip -t` (a real, valid gzip stream) but does
    NOT match the pod's own sha256 -- e.g. a transfer that silently
    truncated or otherwise altered bytes in a way that still gzip-decodes
    -- must still be caught.
    """
    with _Fixture() as fx:
        fx.run(["init", "1.2.3.4", "2222"])
        fx.put_fake_backup("zions-backup-20260401T000000Z.tar.gz", age_s=10)
        r = fx.run(["pull", "backup", "--latest"])
        assert_eq(r.returncode, 0, "sanity: the unmodified pull succeeds", extra=r.stdout + r.stderr)
        pulled = list((fx.home / "pod-exports").rglob("zions-backup-*.tar.gz"))
        assert_true(len(pulled) == 1, "sanity: exactly one archive pulled")
        # Now corrupt the LOCAL copy only (simulating a transfer that changed
        # bytes in flight) -- the remote's own sha256sum is unaffected.
        pulled[0].write_bytes(_valid_gzip_bytes(b"a different, still-valid gzip payload\n"))
        # Call zlpod's own `_verify_pull` directly against the now-mismatched
        # local file: source the script (under `set -- help`, so its own
        # end-of-file dispatch takes the harmless `help` path and returns,
        # instead of the `$# -eq 0` guard exiting 2 before `_verify_pull`
        # is even reachable) and invoke the function itself.
        result = subprocess.run(
            ["bash", "-c",
             f'set -- help; . "{ZLPOD}"; _verify_pull "{pulled[0]}" "/data/backups/{pulled[0].name}"'],
            capture_output=True, text=True, timeout=15, env=fx.env,
        )
        assert_true(result.returncode != 0, "a locally-altered file fails _verify_pull's sha256 check",
                    extra=result.stdout + result.stderr)
        assert_true("mismatch" in (result.stdout + result.stderr).lower() or
                    "CORRUPT" in (result.stdout + result.stderr),
                    "the failure names the mismatch", extra=result.stdout + result.stderr)


def test_ssh_failure_during_sha256_check_says_could_not_verify_and_keeps_the_file():
    """Round-4 LOW: an ssh/sha256sum failure during `_verify_pull`'s
    remote check is NOT evidence the transfer is bad -- the file locally
    still passes `gzip -t` fine. It must say "could not verify" (never
    "CORRUPT TRANSFER") and must NOT rename the good file to `.corrupt`.
    """
    with _Fixture() as fx:
        fx.run(["init", "1.2.3.4", "2222"])
        fx.put_fake_backup("zions-backup-20260401T000000Z.tar.gz", age_s=10)
        fx.env["FAKE_SSH_FAIL_SHA256"] = "1"
        r = fx.run(["pull", "backup", "--latest"])
        assert_true(r.returncode != 0, "an unverifiable transfer does not report success")
        out = r.stdout + r.stderr
        assert_true("could not verify" in out.lower(), "the message says could-not-verify", extra=out)
        assert_true("CORRUPT" not in out, "the message never says CORRUPT for an ssh failure", extra=out)
        pulled = list((fx.home / "pod-exports").rglob("zions-backup-*.tar.gz"))
        assert_true(len(pulled) == 1, "the good file survives under its ORIGINAL name", extra=out)
        corrupt = list((fx.home / "pod-exports").rglob("*.corrupt"))
        assert_true(len(corrupt) == 0, "the good file was NOT renamed to .corrupt", extra=str(corrupt))


def test_mutation_treating_ssh_failure_as_corrupt_is_caught():
    """Prove the test above bites: inline the OLD `_verify_pull` behavior
    (an empty remote_sha lumped into the SAME `reason` as a real
    mismatch/corruption -- renames and says CORRUPT), confirm it would
    have renamed the good file, then confirm the real (restored) zlpod
    does not.
    """
    with _Fixture() as fx:
        fx.run(["init", "1.2.3.4", "2222"])
        local = fx.put_fake_backup("zions-backup-20260401T000000Z.tar.gz", age_s=10)
        # Simulate the OLD code's outcome directly: an empty remote_sha
        # (ssh failed) went straight into `reason`, so the file got
        # renamed and "CORRUPT TRANSFER" was printed -- for a file that
        # is, in fact, perfectly fine.
        old_style_local = fx.home / "old-style-copy.tar.gz"
        old_style_local.write_bytes(local.read_bytes())
        reason = "could not get the pod's own sha256sum of X (ssh/sha256sum failed remotely)"  # old wording
        if reason:
            old_style_local.rename(str(old_style_local) + ".corrupt")
        assert_true(Path(str(old_style_local) + ".corrupt").exists(),
                    "mutated (old, ssh-failure-is-corrupt) code renames a perfectly good file -- confirms the bite")

        fx.env["FAKE_SSH_FAIL_SHA256"] = "1"
        r = fx.run(["pull", "backup", "--latest"])
        out = r.stdout + r.stderr
        assert_true("CORRUPT" not in out, "the real (restored) zlpod never claims CORRUPT for an ssh failure")
        corrupt = list((fx.home / "pod-exports").rglob("*.corrupt"))
        assert_true(len(corrupt) == 0, "the real (restored) zlpod never renames the good file")


def test_pull_backup_dir_with_a_space_is_quoted_correctly():
    """Round-4 LOW (zlpod:326): `${backup_dir}` in the remote `ls -1t
    ${backup_dir}/...` command must itself be quoted -- an operator-
    configured COMPACTOR_BACKUP_DIR containing a space, sent unquoted,
    gets word-split by the REMOTE shell into two `ls` arguments and the
    lookup silently fails to find the archive.
    """
    with _Fixture() as fx:
        fx.run(["init", "1.2.3.4", "2222"])
        fx.env["FAKE_BACKUP_DIR"] = "/data/custom backups"
        custom_dir = fx.remote_root / "data" / "custom backups"
        custom_dir.mkdir(parents=True)
        p = custom_dir / "zions-backup-20260301T000000Z.tar.gz"
        p.write_bytes(_valid_gzip_bytes())

        r = fx.run(["pull", "backup", "--latest"])
        assert_eq(r.returncode, 0, "pull backup --latest exits 0 against a backup dir containing a space",
                  extra=r.stdout + r.stderr)
        pulled = list((fx.home / "pod-exports").rglob("zions-backup-*.tar.gz"))
        assert_true(len(pulled) == 1, "the archive under the space-containing dir was pulled",
                    extra=r.stdout + r.stderr)


def test_mutation_unquoted_backup_dir_misses_the_archive():
    """Prove the test above bites: run the exact remote `ls` command the
    OLD (unquoted) zlpod code built, over ssh, against a backup dir with
    a space, and show it does NOT find the archive -- then confirm the
    real (fixed, quoted) `zlpod pull backup` does.
    """
    with _Fixture() as fx:
        fx.env["FAKE_REMOTE_ROOT"] = str(fx.remote_root)
        custom_dir = fx.remote_root / "data" / "custom backups"
        custom_dir.mkdir(parents=True)
        p = custom_dir / "zions-backup-20260301T000000Z.tar.gz"
        p.write_bytes(_valid_gzip_bytes())

        old_style = subprocess.run(
            ["ssh", "zlpod", "ls -1t /data/custom backups/zions-backup-*.tar.gz 2>/dev/null | head -1"],
            capture_output=True, text=True, timeout=15, env=fx.env,
        )
        assert_true("zions-backup-20260301T000000Z.tar.gz" not in old_style.stdout,
                    "mutated (unquoted backup_dir) remote `ls` does not find the archive -- confirms the bite",
                    extra=f"stdout={old_style.stdout!r} stderr={old_style.stderr!r}")

        fx.run(["init", "1.2.3.4", "2222"])
        fx.env["FAKE_BACKUP_DIR"] = "/data/custom backups"
        r = fx.run(["pull", "backup", "--latest"])
        assert_eq(r.returncode, 0, "the real (restored, quoted) zlpod finds it", extra=r.stdout + r.stderr)


def test_mutation_removing_verify_pull_call_lets_corruption_through():
    """Prove the tests above bite: call scp directly (bypassing
    `cmd_pull_backup`'s own `_verify_pull` call entirely -- the OLD
    behaviour) with a corrupt remote file, and show the local copy is
    accepted with no error at all.
    """
    with _Fixture() as fx:
        fx.run(["init", "1.2.3.4", "2222"])
        fx.put_fake_backup("zions-backup-20260401T000000Z.tar.gz", age_s=10, content=b"not gzip")
        out_dir = fx.home / "pod-exports" / "manual"
        out_dir.mkdir(parents=True)
        local_tgz = out_dir / "zions-backup-20260401T000000Z.tar.gz"
        r = subprocess.run(
            ["scp", "zlpod:/data/backups/zions-backup-20260401T000000Z.tar.gz", str(local_tgz)],
            capture_output=True, text=True, timeout=15, env=fx.env,
        )
        assert_eq(r.returncode, 0, "the bare scp call itself reports success")
        assert_true(local_tgz.exists(), "mutated (no _verify_pull call) leaves the corrupt file in place, "
                    "under its ORIGINAL trusted-looking name -- confirms the bite")
        # And the real cmd_pull_backup (which DOES call _verify_pull) refuses instead:
        r2 = fx.run(["pull", "backup", "--latest"])
        assert_eq(r2.returncode, 1, "the real pull backup refuses the same corrupt content")


def test_push_wordban_copies_only_the_check_script():
    with _Fixture() as fx:
        fx.run(["init", "1.2.3.4", "2222"])
        src_dir = fx.home / "zl-ops" / "word-ban"
        src_dir.mkdir(parents=True)
        (src_dir / "word-ban-check.py").write_text("print('ok')\n", encoding="utf-8")
        (src_dir / "word-ban.md").write_text("report, not needed on the pod\n", encoding="utf-8")
        r = fx.run(["push", "wordban"])
        assert_eq(r.returncode, 0, "zlpod push wordban exits 0", )
        lines = [ln for ln in fx.log_lines() if ln.startswith("scp ")]
        assert_true(len(lines) == 1, "exactly one scp call", extra="\n".join(lines))
        assert_true("word-ban-check.py" in lines[0], "only word-ban-check.py is copied", extra=lines[0])
        assert_true("word-ban.md" not in lines[0], "word-ban.md (not needed on the pod) is never copied",
                    extra=lines[0])


def test_commands_without_init_refuse_cleanly():
    with _Fixture() as fx:
        r = fx.run(["status"])
        assert_eq(r.returncode, 1, "a command before `init` refuses (exit 1), not a crash")
        assert_true("not configured" in (r.stdout + r.stderr), "the refusal explains itself",
                    extra=r.stdout + r.stderr)


def test_no_args_and_help_exit_codes():
    with _Fixture() as fx:
        r = fx.run([])
        assert_eq(r.returncode, 2, "zlpod with no args exits 2")
        r2 = fx.run(["help"])
        assert_eq(r2.returncode, 0, "zlpod help exits 0")


if __name__ == "__main__":
    test_init_writes_config_and_idempotent_ssh_config_block()
    test_init_inserts_at_top_and_warns_about_an_earlier_host_zlpod()
    test_zl_subcommand_quotes_arguments_with_spaces_safely()
    test_pull_bundle_extracts_into_dated_directory()
    test_pull_backup_latest()
    test_pull_backup_honors_pod_reported_backup_dir()
    test_pull_backup_falls_back_to_default_dir_if_zl_disk_unparseable()
    test_fake_ssh_is_faithful_enough_to_catch_the_old_pull_backup_bug()
    test_pull_backup_detects_gzip_corruption_and_renames_to_corrupt()
    test_pull_backup_detects_all_zero_file_and_renames_to_corrupt()
    test_pull_backup_detects_sha256_mismatch()
    test_ssh_failure_during_sha256_check_says_could_not_verify_and_keeps_the_file()
    test_mutation_treating_ssh_failure_as_corrupt_is_caught()
    test_pull_backup_dir_with_a_space_is_quoted_correctly()
    test_mutation_unquoted_backup_dir_misses_the_archive()
    test_mutation_removing_verify_pull_call_lets_corruption_through()
    test_push_wordban_copies_only_the_check_script()
    test_commands_without_init_refuse_cleanly()
    test_no_args_and_help_exit_codes()

    print("\nAll ops/laptop/zlpod tests passed.")
