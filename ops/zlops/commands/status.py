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


def _fetch_health_doc(timeout: float) -> dict | None:
    _status_code, doc, error = common.http_get_json(f"{podenv.COMPACTOR_URL}/health/full", timeout=timeout)
    return None if error is not None else doc


def _service_table(status: dict[str, str], doc: dict | None) -> tuple[list[str], bool, str | None, frozenset[str]]:
    from .. import telemetry  # lazy: B's module, not needed by other commands

    lines = []
    any_unexpected = False
    tunnel_warn: str | None = None
    hidden: set[str] = set()
    width = max((len(s) for s in podenv.SERVICES), default=8)
    for name in podenv.SERVICES:
        state = status.get(name, "UNKNOWN")
        # v3.1.9.9: `cloudflared` is in ONE_SHOT_OR_OPTIONAL (EXITED with no
        # token set is correct), but "a token IS set and cloudflared isn't
        # RUNNING anyway" is a real fault that membership alone would hide
        # -- cross-checked against /health/full's own `tunnel.configured`.
        warn_text = telemetry.cloudflared_fault(doc, state) if name == "cloudflared" else None
        # Pre-v3.1.9.9 pod: supervisord has no [program:cloudflared] at all
        # (state defaults to "UNKNOWN") AND the compactor's /health/full
        # predates the `tunnel` key entirely (`tunnel_status_word` is None).
        # There is nothing real to report -- showing "cloudflared UNKNOWN"
        # is pure noise, and it must not count against the "N/10" total
        # either. The warning above still fires (and this branch is not
        # taken) whenever `tunnel.configured` is true but the program is
        # missing -- that IS a real fault, never hidden.
        if name == "cloudflared" and state == "UNKNOWN" and warn_text is None and telemetry.tunnel_status_word(doc) is None:
            hidden.add(name)
            continue
        expected_running = name not in podenv.ONE_SHOT_OR_OPTIONAL
        unexpected = state != "RUNNING" and expected_running
        if unexpected:
            any_unexpected = True
        if warn_text:
            tunnel_warn = warn_text
            flag = f" <-- warn: {warn_text}"
        elif unexpected:
            flag = " <-- unexpected"
        else:
            flag = ""
        lines.append(f"  {name.ljust(width)}  {state}{flag}")
    return lines, any_unexpected, tunnel_warn, frozenset(hidden)


def run(argv: list[str]) -> int:
    brief = "--brief" in argv
    timeout = 1.0 if brief else 3.0

    from .. import telemetry  # lazy: B's module, not needed by other commands

    status = common.supervisorctl_status()
    doc = _fetch_health_doc(timeout)
    verdict = telemetry.interpret_health(doc, time.time())
    lines, any_unexpected, tunnel_warn, hidden = _service_table(status, doc)
    gpu = _gpu_line(timeout)

    if brief:
        counted = [n for n in podenv.SERVICES if n not in hidden]
        n_running = sum(1 for n in counted if status.get(n) == "RUNNING")
        bad = ", ".join(n for n in counted if status.get(n) != "RUNNING" and n not in podenv.ONE_SHOT_OR_OPTIONAL)
        bad_note = f" ({bad} NOT running)" if bad else ""
        if tunnel_warn:
            bad_note += f"; {tunnel_warn}"
        print(f"zl status: {n_running}/{len(counted)} services RUNNING{bad_note}")
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
    if verdict["status"] == "warn" or tunnel_warn:
        return 3
    return 0
