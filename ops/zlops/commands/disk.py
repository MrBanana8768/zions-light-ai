"""`zl disk` -- du of each /data/* entry (with a per-entry timeout: MooseFS
can stall), the CONTAINER OVERLAY's df /, a backups count+size, and the HF
cache size. `df /data` itself is never run here: it reports the whole
shared MooseFS cluster, not this volume's quota (OPERATIONS.md "Disk is
filling up") -- printing it would be actively misleading, so we say so
instead of printing a number that looks like an answer.
"""

from __future__ import annotations

import subprocess

from .. import common, podenv

_DU_TIMEOUT_S = 20.0


def du(path) -> str:
    try:
        r = subprocess.run(["du", "-sh", str(path)], capture_output=True, text=True, timeout=_DU_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return f"(du timed out after {_DU_TIMEOUT_S:.0f}s -- MooseFS stall)"
    except OSError as e:
        return f"(du failed: {e})"
    if r.returncode != 0:
        return f"(du error: {(r.stderr or r.stdout).strip()[:160]})"
    size = (r.stdout.split("\t", 1) or [""])[0].strip()
    return size or "?"


def dir_count_and_size(path) -> tuple[int, int]:
    count = 0
    total = 0
    try:
        for p in path.iterdir():
            if p.is_file():
                count += 1
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    return count, total


def run(argv: list[str]) -> int:
    print(f"du -sh of each entry under {podenv.DATA} (MooseFS -- each entry has its own {_DU_TIMEOUT_S:.0f}s timeout):")
    if podenv.DATA.is_dir():
        for entry in sorted(podenv.DATA.iterdir()):
            print(f"  {du(entry):>10}  {entry}")
    else:
        print(f"  ({podenv.DATA} is not a directory here)")
    print()

    print("df / (the CONTAINER OVERLAY, reset on every restart -- NOT /data):")
    try:
        r = subprocess.run(["df", "-h", "/"], capture_output=True, text=True, timeout=10)
        for ln in r.stdout.strip().splitlines():
            print(f"  {ln}")
    except (OSError, subprocess.TimeoutExpired) as e:
        print(f"  (df failed: {e})")
    print()
    print(f"NOTE: `df {podenv.DATA}` is deliberately never run above -- it reports the whole")
    print("shared MooseFS cluster's free space (hundreds of TB), not this volume's quota.")
    print("Compare the `du` total above with the size you gave the volume in the RunPod console.")
    print()

    count, total = dir_count_and_size(podenv.BACKUPS)
    print(f"backups: {count} archive(s) in {podenv.BACKUPS}, {common.human_bytes(total)} total")

    # compactor's own control/pending roots (podenv.CONTROL_ROOT/PENDING_ROOT
    # -- siblings of STORAGE_ROOT, never nested inside it, see podenv.py's
    # own comment) -- reported only if present, same `du` (time-bounded).
    for label, root in (("control root", podenv.CONTROL_ROOT), ("pending root", podenv.PENDING_ROOT)):
        if root.is_dir():
            print(f"{label} ({root}): {du(root)}")

    if podenv.HF_HOME.is_dir():
        print(f"HF cache ({podenv.HF_HOME}): {du(podenv.HF_HOME)}")
    else:
        print(f"HF cache: {podenv.HF_HOME} does not exist")

    return 0
