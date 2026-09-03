#!/usr/bin/env python3
"""Run the whole local test suite, and be honest about what actually ran.

    python scripts/run-tests.py                # everything
    python scripts/run-tests.py --fast         # skip the suites over 60s
    python scripts/run-tests.py --only tail    # substring filter
    python scripts/run-tests.py --saturation   # add the long soak
    python scripts/run-tests.py --list         # names and known timings

WHY THIS EXISTS. Until 2026-09-02 "all 59 suites pass" was the release signal
on this project, and it was not true. Three suites exited 0 WITHOUT running
their checks when a docker fixture on localhost:18000 was absent, and one of
them says so in its own skip text: "Skipping it means the budget code is
currently only covered by char/4 assertions - the estimator that took
production down." Every green run in the v3.1.7 branch, including several the
author reported with confidence, excluded the tokenizer contract entirely.

A runner cannot fix a test that lies, so the suites now exit 3 to mean SKIP
(see COMPACTOR_ALLOW_FIXTURE_SKIP). This runner is the other half: it counts
SKIP separately, never folds it into PASS, and by default treats a skipped
suite as a failed run, because on this project an unrun test has twice been
the thing that shipped the bug.

EXIT CODES
    0   every selected suite ran and passed
    1   at least one suite failed
    2   the runner itself could not proceed (bad interpreter, no suites)
    3   everything that ran passed, but something was SKIPPED
        (--allow-skips downgrades this to 0)

WHY NOT pytest. These suites are plain scripts that own their own process:
each sets environment variables BEFORE importing `main`, and several patch
module globals that only apply at import time. Collecting them into one
pytest process would make them interfere. One subprocess per file is not a
workaround here, it is the design.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMPACTOR = ROOT / "compactor"
PIPELINES = ROOT / "pipelines"

# The real interpreter, not the Windows Store stub that reports itself absent.
DEFAULT_PY = Path(
    r"C:\Users\rngge\AppData\Local\Programs\Python\Python314\python.exe"
)

# Every suite runs with these. Offline flags keep a test run from reaching the
# network for a tokenizer or an embedding model: a suite that silently
# downloads is a suite whose result depends on the network.
BASE_ENV = {
    "PYTHONIOENCODING": "utf-8",
    "PYTHONPATH": ".",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "COMPACTOR_FORCE_OFFLINE": "true",
}

# Measured on this machine, 2026-09-02 (test_budget_guard.py re-measured
# 2026-09-03 after R22). Only used for --fast and for the "still running"
# note; a suite that drifts far from its entry is worth a look.
SLOW_S = {
    "test_budget_guard.py": 5,
    "test_dedup_churn_gate.py": 147,
    "test_summarize_invariant.py": 71,
    "test_backup.py": 26,
    "test_saturation.py": 45,
}
FAST_CUTOFF_S = 60

# Not run unless --saturation. It is the only suite whose job is scale rather
# than logic, and it is slow enough that gating it keeps the default run
# usable during development.
SATURATION = {"test_saturation.py", "test_soak_conversation.py"}

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


def discover(only: str | None) -> list[tuple[Path, Path]]:
    """(cwd, script) pairs. cwd matters: each suite imports its siblings."""
    out = [(COMPACTOR, p) for p in sorted(COMPACTOR.glob("test_*.py"))]
    out += [(PIPELINES, p) for p in sorted(PIPELINES.glob("test_*.py"))]
    if only:
        out = [(d, p) for d, p in out if only in p.name]
    return out


def run_one(py: Path, cwd: Path, script: Path, timeout: int) -> tuple[str, float, str]:
    env = {**os.environ, **BASE_ENV}
    t0 = time.monotonic()
    try:
        r = subprocess.run(
            [str(py), script.name],
            cwd=cwd, env=env, timeout=timeout,
            capture_output=True, text=True, errors="replace",
        )
    except subprocess.TimeoutExpired:
        return FAIL, time.monotonic() - t0, f"timed out after {timeout}s"
    dt = time.monotonic() - t0
    if r.returncode == 0:
        return PASS, dt, ""
    if r.returncode == 3:
        
        # The reason line, not the "SKIPPED: <file>" banner above it.
        reason = next(
            (ln.split("reason:", 1)[-1].strip()
             for ln in (r.stdout or "").splitlines() if "reason:" in ln),
            "",
        )
        if not reason:
            body = [ln.strip() for ln in (r.stdout or "").splitlines() if ln.strip()]
            reason = next((ln for ln in body if not ln.startswith("SKIPPED")), "fixture unavailable")
        return SKIP, dt, reason[:120]
    body = (r.stdout or "") + (r.stderr or "")
    hit = next((ln for ln in body.splitlines() if ln.startswith("FAIL")), "")
    if not hit:
        hit = next((ln for ln in reversed(body.splitlines()) if ln.strip()), "")
    return FAIL, dt, hit.strip()[:160]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=str(DEFAULT_PY))
    ap.add_argument("--only", help="substring filter on the file name")
    ap.add_argument("--fast", action="store_true",
                    help=f"skip suites known to exceed {FAST_CUTOFF_S}s")
    ap.add_argument("--saturation", action="store_true",
                    help="include the scale suites (slow, off by default)")
    ap.add_argument("--allow-skips", action="store_true",
                    help="a SKIP does not fail the run")
    ap.add_argument("--timeout", type=int, default=600,
                    help="per-suite ceiling in seconds (default 600)")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    py = Path(args.python)
    if not py.exists():
        print(f"interpreter not found: {py}", file=sys.stderr)
        print("bare `python` on this machine is a Store stub; pass --python",
              file=sys.stderr)
        return 2

    suites = discover(args.only)
    if not args.saturation:
        suites = [(d, p) for d, p in suites if p.name not in SATURATION]
    if args.fast:
        suites = [(d, p) for d, p in suites if SLOW_S.get(p.name, 0) < FAST_CUTOFF_S]
    if not suites:
        print("no suites selected", file=sys.stderr)
        return 2

    if args.list:
        for _, p in suites:
            known = SLOW_S.get(p.name)
            print(f"  {p.name:<44}{f'~{known}s' if known else ''}")
        return 0

    print(f"{len(suites)} suite(s), interpreter {py.name}")
    print("-" * 72)
    results, t_all = [], time.monotonic()
    for cwd, script in suites:
        # The per-suite ceiling must clear the slowest known suite with room.
        # R22: a 240s ceiling once reported test_budget_guard (239s, real —
        # confirmed by running it twice) as a HANG, because every other
        # suite finished in <=21s and the runner had no way to tell "slow"
        # from "stuck". The 239s was two unstubbed synchronous httpx.post
        # calls per guard test (count_tokens_exact / count_text_tokens_exact
        # hitting VLLM_URL/tokenize for real, with nobody listening) times
        # dozens of tests — each one paying a ~4.2s dual-stack (IPv6-then-
        # IPv4 loopback) connect timeout on this machine. test_budget_guard.py
        # now stubs that call at the module level so the failure it was
        # always going to hit arrives immediately instead of after a real OS
        # timeout; the suite runs in ~1.6s and asserts exactly what it did
        # before (see the R22 comment near the top of that file). The ceiling
        # logic itself stays — this class of "cannot tell slow from hung" bug
        # is exactly why a fixed per-suite ceiling isn't enough on its own,
        # and SLOW_S is what lets a FUTURE regression be caught the same way
        # this one was: by comparing a fresh measurement against history.
        ceiling = max(args.timeout, SLOW_S.get(script.name, 0) * 2)
        status, dt, note = run_one(py, cwd, script, ceiling)
        results.append((status, script.name, dt, note))
        mark = {PASS: "ok  ", FAIL: "FAIL", SKIP: "SKIP"}[status]
        line = f"  {mark} {script.name:<44}{dt:6.1f}s"
        print(line + (f"  {note}" if note else ""), flush=True)

    n = {s: sum(1 for r in results if r[0] == s) for s in (PASS, FAIL, SKIP)}
    print("-" * 72)
    print(f"{n[PASS]} passed, {n[FAIL]} failed, {n[SKIP]} skipped "
          f"in {time.monotonic() - t_all:.0f}s")

    if n[FAIL]:
        print("\nFAILED:")
        for s, name, _, note in results:
            if s == FAIL:
                print(f"  {name}: {note}")
    if n[SKIP]:
        # ASCII only: the Windows console this runs on renders an em dash as a
        # replacement character, and a summary line nobody can read is a
        # summary line nobody reads.
        print("\nSKIPPED - these did NOT run, and are not evidence of anything:")
        for s, name, _, note in results:
            if s == SKIP:
                print(f"  {name}: {note}")
        print("  Start the fixture:  docker compose -f "
              "docker-compose.tokenizer-contract.yml up -d tokenize-fixture")

    if n[FAIL]:
        return 1
    if n[SKIP] and not args.allow_skips:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
