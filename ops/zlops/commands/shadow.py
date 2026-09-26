"""`zl shadow up --ref REF | status | logs | down` -- COMMANDS.md "Shadow
deploy": run a build beside the live compactor, sharing the running vLLM,
writing to its OWN storage root so it never touches her live memory.

Safety: refuses `up` outright if podenv.SHADOW_STORE would resolve to
podenv.STORE (that would mean the shadow writes into her live memory).
`down` kills by the PID this command itself recorded, NEVER `pkill -f`
(memory.md: a name-pattern kill can hit an unrelated process); it removes
the shadow storage root afterward only when that path is EXACTLY
podenv.SHADOW_STORE, never a wildcard/glob.

Review H1: `_PIDFILE` lives on `/data`, which SURVIVES a pod restart,
while PIDs on a freshly restarted container start counting from low
numbers again -- a bare pid number is not enough to trust "this is still
our shadow process" after a restart (it could now be vLLM or the
compactor). The pidfile therefore records pid + boot_id
(`/proc/sys/kernel/random/boot_id`) + the process's own start time
(`/proc/<pid>/stat` field 22 -- changes if the pid was reused by an
unrelated later process even within the SAME boot) + its cmdline (must
still look like our own shadow uvicorn).

Review LOW (round 2): `boot_id`, inside a CONTAINER, is the HOST kernel's
random boot id, not the container's -- it does NOT change across a pod
restart the way the name suggests (a restarted container is a fresh
`docker run`/RunPod restart, but the underlying host kernel, and so
`/proc/sys/kernel/random/boot_id`, is unchanged). It is checked anyway as
a cheap, free extra signal (a genuinely different HOST, e.g. after
RunPod reschedules the pod onto different hardware, is still caught), but
the actual safety against pid reuse WITHIN the same host/boot is carried
entirely by `start_time` (a reused pid gets a different start tick almost
certainly) and `cmdline` (round-2 review's own mutation M8, dropping the
`start_time` check, was caught by the existing tests -- confirming those
two, not `boot_id`, are load-bearing here). `status` and `down` both
verify all three before trusting the record; `down` also repeats `up`'s
STORE-equality guard immediately before `rmtree`, not just at `up` time.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from .. import common, podenv

_CHECKOUT = podenv.REPO.parent / "compactor-shadow"  # /opt/compactor-shadow, container overlay -- ephemeral by design
_PIDFILE = podenv.ZL_HOME / "shadow.pid"
_LOGFILE = podenv.ZL_HOME / "shadow.log"
_URL = f"http://127.0.0.1:{podenv.SHADOW_PORT}"
_BOOT_ID_FILE = Path("/proc/sys/kernel/random/boot_id")


def _refuse_if_storage_unsafe() -> int | None:
    """Shared by `up` (would a new shadow write into her live memory?) and
    `down` (would deleting SHADOW_STORE also delete her live memory?) --
    review H1 wants the SAME guard repeated right before `down`'s rmtree,
    not trusted from whenever `up` last checked it.
    """
    if podenv.SHADOW_STORE.resolve() == podenv.STORE.resolve():
        print(
            f"zl shadow: refusing -- SHADOW_STORE ({podenv.SHADOW_STORE}) resolves to the live STORE "
            f"({podenv.STORE}). Starting or removing a shadow here would touch her live memory."
        )
        return 1
    return None


def _boot_id() -> str:
    try:
        return _BOOT_ID_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _proc_start_time(pid: int) -> str | None:
    """Field 22 (starttime, ticks since boot) of /proc/<pid>/stat -- read
    past the `(comm)` field the safe way (comm itself can contain spaces
    or parens), by splitting on the LAST ')' rather than the first.
    """
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return None
    rparen = raw.rfind(")")
    if rparen == -1:
        return None
    fields = raw[rparen + 2:].split()
    try:
        return fields[19]  # state(0) ppid(1) ... starttime is index 19 after comm
    except IndexError:
        return None


def _cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\0", b" ").decode("utf-8", "replace")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _write_pid_record(pid: int) -> None:
    record = {
        "pid": pid,
        "boot_id": _boot_id(),
        "start_time": _proc_start_time(pid),
        "cmdline": _cmdline(pid),
    }
    _PIDFILE.write_text(json.dumps(record), encoding="utf-8")


def _read_pid_record() -> dict | None:
    try:
        record = json.loads(_PIDFILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def _record_is_our_live_process(record: dict) -> bool:
    """True only if EVERY identity check still agrees: the pid is alive,
    the boot id matches (a pod restart invalidates every old record
    outright), the process's own start time matches (catches the pid
    having been reused by an unrelated process, restart or not), and its
    cmdline still looks like our own shadow uvicorn (review H1).
    """
    pid = record.get("pid")
    if not isinstance(pid, int) or not _pid_alive(pid):
        return False
    if not record.get("boot_id") or record.get("boot_id") != _boot_id():
        return False
    if not record.get("start_time") or _proc_start_time(pid) != record.get("start_time"):
        return False
    cmdline = _cmdline(pid)
    if not cmdline or str(podenv.SHADOW_PORT) not in cmdline or str(_CHECKOUT) not in cmdline:
        return False
    return True


def _up(argv: list[str]) -> int:
    unsafe = _refuse_if_storage_unsafe()
    if unsafe is not None:
        return unsafe

    ap = argparse.ArgumentParser(prog="zl shadow up")
    ap.add_argument("--ref", required=True)
    try:
        args = ap.parse_args(argv)
    except SystemExit as e:
        return int(e.code) if e.code is not None else 2

    existing = _read_pid_record()
    if existing is not None and _record_is_our_live_process(existing):
        print(f"zl shadow: already up (pid {existing['pid']}) -- `zl shadow down` first")
        return 1

    if _CHECKOUT.exists():
        shutil.rmtree(_CHECKOUT)
    clone = common.run(["git", "clone", "-b", args.ref, podenv.GITHUB_REPO_URL, str(_CHECKOUT)], timeout=300)
    if clone.returncode != 0:
        print(f"zl shadow up: git clone failed:\n{clone.stdout}\n{clone.stderr}")
        return 1

    env = dict(common.merged_environ())
    env["COMPACTOR_STORAGE_ROOT"] = str(podenv.SHADOW_STORE)
    env["VLLM_URL"] = podenv.VLLM_URL

    _PIDFILE.parent.mkdir(parents=True, exist_ok=True)
    logf = open(_LOGFILE, "ab")  # noqa: SIM115 -- handed to Popen, closed by the OS on exit
    proc = subprocess.Popen(
        [
            str(podenv.COMPACTOR_PY), "-m", "uvicorn", "main:app",
            "--app-dir", str(_CHECKOUT / "compactor"),
            "--host", "127.0.0.1", "--port", str(podenv.SHADOW_PORT), "--log-level", "info",
        ],
        env=env,
        stdout=logf,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    _write_pid_record(proc.pid)
    print(f"zl shadow up: started pid {proc.pid}, ref {args.ref}, storage {podenv.SHADOW_STORE}")
    print(f"log: {_LOGFILE}  ({_URL}/health/full once it is up)")
    return 0


def _status(argv: list[str]) -> int:
    record = _read_pid_record()
    if record is None or not _record_is_our_live_process(record):
        print("zl shadow: not up (no live, verified pid recorded)")
        return 1
    pid = record["pid"]
    print(f"zl shadow: up, pid {pid}")
    _status_code, doc, error = common.http_get_json(f"{_URL}/health/full", timeout=5.0)
    if doc is None:
        print(f"  {_URL}/health/full: not answering yet ({error})")
    else:
        print(f"  {_URL}/health/full: status={doc.get('status')}")
    return 0


def _logs(argv: list[str]) -> int:
    if not _LOGFILE.exists():
        print(f"zl shadow: no log file yet ({_LOGFILE})")
        return 1
    return common.run_stream(["tail", "-n", "200", "-f", str(_LOGFILE)])


def _down(argv: list[str]) -> int:
    record = _read_pid_record()
    refused_to_signal = False
    if record is not None:
        if _record_is_our_live_process(record):
            pid = record["pid"]
            try:
                os.kill(pid, signal.SIGTERM)  # NEVER pkill -f -- kill exactly the pid we recorded AND verified
            except OSError as e:
                print(f"zl shadow down: could not signal pid {pid}: {e}")
            for _ in range(20):
                if not _pid_alive(pid):
                    break
                time.sleep(0.5)
            if _pid_alive(pid):
                print(f"zl shadow down: pid {pid} did not exit; leaving it -- check it by hand")
        else:
            # Review LOW (round 2): this pid is UNVERIFIABLE, not
            # verifiably dead -- it could still be alive and actively
            # writing under SHADOW_STORE (a pid-reuse false negative here
            # means "some OTHER process now has this pid", not "our
            # shadow is gone"). The OLD code fell through to `rmtree`
            # regardless, which could delete storage a live, unrelated-
            # looking process still has open. Refuse the whole `down`
            # instead: leave the pidfile (so the operator can see what
            # was recorded) and the storage untouched, and say so.
            print("zl shadow down: the recorded pid is not verifiably our own shadow process any more "
                  "(pid reused, or a restart changed the boot id) -- not signalling anything, and "
                  "NOT removing SHADOW_STORE (it could still be in use by whatever this pid now is). "
                  "Check `ps` by hand, then remove the pidfile and SHADOW_STORE yourself if it's safe.")
            refused_to_signal = True
    else:
        print("zl shadow down: no pidfile -- nothing recorded as up")

    if refused_to_signal:
        return 1

    if record is not None:
        try:
            _PIDFILE.unlink()
        except OSError:
            pass

    # Always exactly podenv.SHADOW_STORE, a fixed constant -- never a glob or
    # a path built from user input (see module docstring). Repeat the SAME
    # guard `up` uses, immediately before rmtree, not just at `up` time
    # (review H1) -- podenv.STORE/SHADOW_STORE are module-level constants,
    # but re-checking here costs nothing and never trusts an earlier check.
    unsafe = _refuse_if_storage_unsafe()
    if unsafe is not None:
        return unsafe
    if podenv.SHADOW_STORE.is_dir():
        shutil.rmtree(podenv.SHADOW_STORE)
        print(f"zl shadow down: removed {podenv.SHADOW_STORE}")
    return 0


def run(argv: list[str]) -> int:
    if not argv:
        print("usage: zl shadow up --ref REF | status | logs | down")
        return 2
    sub, rest = argv[0], argv[1:]
    if sub == "up":
        return _up(rest)
    if sub == "status":
        return _status(rest)
    if sub == "logs":
        return _logs(rest)
    if sub == "down":
        return _down(rest)
    print(f"zl shadow: unknown subcommand {sub!r}; usage: zl shadow up --ref REF | status | logs | down")
    return 2
