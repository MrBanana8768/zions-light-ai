"""`zl status [--brief]` -- supervisorctl status + the health verdict +
a GPU one-liner. `--brief` is what bootstrap.sh's login hook runs on every
interactive SSH login, so it must finish in ~3s total even over a stalled
MooseFS mount: short HTTP/subprocess timeouts, no bundle/backup-sized work.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time

from .. import common, podenv


def _gpu_line(timeout: float) -> str:
    if shutil.which("nvidia-smi") is None:
        return "GPU: nvidia-smi not present (normal off a real pod; the NVIDIA runtime injects it)"
    try:
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"GPU: nvidia-smi did not answer in time ({e})"
    if r.returncode != 0 or not r.stdout.strip():
        return f"GPU: nvidia-smi failed: {(r.stderr or r.stdout).strip()[:160]}"
    lines = [ln.strip() for ln in r.stdout.strip().splitlines()]
    parts = []
    for i, ln in enumerate(lines):
        name, util, used, total = (x.strip() for x in ln.split(","))
        parts.append(f"GPU{i}: {name} {util}% util {used}/{total} MiB")
    return " | ".join(parts)


def _health_verdict(timeout: float) -> dict:
    from .. import telemetry  # lazy: B's module, not needed by other commands

    _status_code, doc, error = common.http_get_json(f"{podenv.COMPACTOR_URL}/health/full", timeout=timeout)
    if error is not None:
        doc = None
    return telemetry.interpret_health(doc, time.time())


def _service_table(status: dict[str, str]) -> tuple[list[str], bool]:
    lines = []
    any_unexpected = False
    width = max((len(s) for s in podenv.SERVICES), default=8)
    for name in podenv.SERVICES:
        state = status.get(name, "UNKNOWN")
        expected_running = name not in podenv.ONE_SHOT_OR_OPTIONAL
        unexpected = state != "RUNNING" and expected_running
        if unexpected:
            any_unexpected = True
        flag = " <-- unexpected" if unexpected else ""
        lines.append(f"  {name.ljust(width)}  {state}{flag}")
    return lines, any_unexpected


def run(argv: list[str]) -> int:
    brief = "--brief" in argv
    timeout = 1.0 if brief else 3.0

    status = common.supervisorctl_status()
    lines, any_unexpected = _service_table(status)
    verdict = _health_verdict(timeout)
    gpu = _gpu_line(timeout)

    if brief:
        n_running = sum(1 for s in status.values() if s == "RUNNING")
        bad = ", ".join(n for n in podenv.SERVICES if status.get(n) != "RUNNING" and n not in podenv.ONE_SHOT_OR_OPTIONAL)
        bad_note = f" ({bad} NOT running)" if bad else ""
        print(f"zl status: {n_running}/{len(podenv.SERVICES)} services RUNNING{bad_note}")
        print(f"health: {verdict['status']}" + (f" -- {'; '.join(verdict['reasons'])}" if verdict["reasons"] else ""))
        print(gpu)
        print("zl dash for live telemetry", file=sys.stderr)
    else:
        print("supervisorctl status:")
        for ln in lines:
            print(ln)
        print()
        print(f"health: {verdict['status']}")
        for f in verdict["findings"]:
            print(f"  [{f['level']}] {f['text']}")
        print()
        print(gpu)

    if verdict["status"] == "crit" or any_unexpected:
        return 1
    if verdict["status"] == "warn":
        return 3
    return 0
