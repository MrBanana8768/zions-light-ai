#!/usr/bin/env python3
"""Run the whole local test suite, and be honest about what actually ran.

PREFER THE CONTAINER. This project tests on Linux, because Linux is what
ships — the production image is Ubuntu 24.04 userland:

    docker compose -f docker-compose.tests.yml run --rm unit-tests
    docker compose -f docker-compose.tests.yml run --rm unit-tests --saturation

Everything below still works when invoked directly on the host, and the flags
are identical either way. But a host run is a development convenience, not
evidence: the asyncio loop differs (Proactor vs epoll/uvloop, which is the
whole of the open client-disconnect question), and the timings differ enough
to have cost a full investigation once already (see SLOW_S). The two
fixture-backed suites need their own stack and will honestly SKIP in both:

    docker compose -f docker-compose.tokenizer-contract.yml up --build --exit-code-from contract-tests
    docker compose -f docker-compose.tokenizer-contract.yml run --rm --build soak-tests

    python scripts/run-tests.py                # everything
    python scripts/run-tests.py --fast         # skip the suites over 60s
    python scripts/run-tests.py --only tail    # substring filter
    python scripts/run-tests.py --saturation   # add the long soak
    python scripts/run-tests.py --real-image   # add the real-image operator-script suite
    python scripts/run-tests.py --list         # names and known timings

A THIRD suite needs its own thing this runner cannot give it either: a real
Docker daemon, the published image already pulled, `git`, and (for the
version-gating and content-guard cases specifically) the real 2026-09-22
pod-export backup on this host. `compactor/test_real_image_operator_scripts.py`
drives `scripts/backfill-records.py` and `scripts/import-history.py` against
containers started from the exact published digest — the only way to catch
what a 143-test all-mocked gate already missed once (see CHANGELOG.md
v3.1.9.6 "Fixed"): these scripts are only ever run ON A POD, against an OLDER
installed compactor, from a path that is not the repo. It is excluded from
the default run, the same way SATURATION is, and SKIPS honestly (exit 3) if
Docker or the backup path is not reachable, the same way the tokenizer
contract suite does:

    python scripts/run-tests.py --real-image

THESE ARE TWO DIFFERENT SERVICES IN THE SAME COMPOSE FILE, AND RUNNING ONE
DOES NOT RUN THE OTHER (P9-6, re-confirmed P11, v3.1.9.2 -> v3.1.9.3: the
tokenizer-contract suite was SKIPPED IN EVERY GATE run for three releases in
a row). `contract-tests` drives `compactor/test_tokenizer_contract.py` — the
ONLY suite that exercises a real /tokenize contract, and the only suite that
would have caught the fixture-handler change v3.1.9.2 shipped.
`soak-tests` drives `compactor/test_soak_conversation.py` against its OWN,
separate fixture on a different port — a scale/lag check, not a contract
check. Every "gate" script assembled for a release so far (see the scratch
gate3192/gate.sh's own history) ran `soak-tests` and never ran
`contract-tests` at all, so "the gate is green" has meant "the suite that
would have caught D1, D7 and the v3.1.9.2 fixture-handler change never ran"
— exactly the failure mode this file's WHY THIS EXISTS section already
describes for the plain `docker-compose.tests.yml` case, recurring one layer
up. A release gate is not the four/five docker-compose files that happen to
exist; it is EVERY line above this paragraph, run and its exit code read —
in particular `contract-tests`, on its own, with `--exit-code-from
contract-tests` so a fixture crash before the test process starts still
fails the run. There is no single command that runs everything in this repo;
assembling one from a subset and calling the result "the gate" is exactly
how this suite kept getting left out.

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
    1   at least one suite failed, OR at least one suite was INCONCLUSIVE
    2   the runner itself could not proceed (bad interpreter, no suites)
    3   everything that ran passed, but something was SKIPPED
        (--allow-skips downgrades this to 0)

A suite is INCONCLUSIVE when its subprocess exits with anything other than
0 (pass), 1 (fail) or 3 (skip) — most often 137 (128+9, SIGKILL). Every
compose test stack used to pin a fixed container_name, and the documented
cleanup for a stuck run used to be a wildcard `docker rm -f` filter that matches ANY
project's in-flight container sharing that substring — under concurrent
runs that kills a NEIGHBOUR's suite mid-run, not the stuck one, and an
unexplained exit code is not evidence either way (see
hostile2-apparatus.md, "every stack pins container_name"). Folding that into
FAIL used to let a killed run be read as a real defect; run-tests.py now
prints it under its own INCONCLUSIVE heading and counts it separately, but
still returns exit 1 overall — this runner cannot claim a clean pass (0)
when it could not tell whether a suite passed.

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

# Measured 2026-09-03 in the Ubuntu 24.04 test image
# (docker-compose.tests.yml), which is the platform this project tests on.
# Only used for --fast and for the "still running" note; a suite that drifts
# far from its entry is worth a look.
#
# THESE WERE WINDOWS NUMBERS UNTIL 2026-09-03, AND THAT MADE THE DRIFT SIGNAL
# USELESS. Windows resolves "localhost" to the IPv6 loopback and THEN the IPv4
# one, each against its own 2s connect timeout, so every suite that talks to a
# port nothing listens on paid ~4.2s per call. That is not a small skew:
# test_summarize_invariant measured 71s there and 3.6s here, and
# test_dedup_churn_gate 147s against 58.4s. R22 chased that cost out of
# test_budget_guard specifically (239s -> 1.1s here) before the platform was
# the thing that changed; the entries below are what the suites actually cost
# on the userland production runs.
#
# Windows numbers are kept in the second column for exactly one reason: if
# someone runs the suite on the host and sees 60s where this table says 3.6s,
# the table should tell them why rather than look broken.
SLOW_S = {                            # linux (docker)   windows (host)
    "test_dedup_churn_gate.py": 58,   #      58.4              147
    "test_saturation.py": 19,         #      19.4               45
    "test_backup.py": 18,             #      17.9               26
    "test_summarize_invariant.py": 4,  #       3.6               71
    "test_budget_guard.py": 1,        #       1.1                5
}
# Lowered from 60 with the platform change. At 60 nothing on Linux qualified
# as slow, which made --fast a synonym for a full run — a flag that silently
# does nothing is worse than no flag. At 15 it drops the three suites that
# account for most of the wall clock and leaves the run around 35s.
FAST_CUTOFF_S = 15

# Not run unless --saturation. It is the only suite whose job is scale rather
# than logic, and it is slow enough that gating it keeps the default run
# usable during development.
SATURATION = {"test_saturation.py", "test_soak_conversation.py"}

# Not run unless --real-image. Needs a real docker daemon, the published
# image, git, and the real backup path on this host — none of which the
# sandboxed unit-tests container has. Kept OUT of the default selection so
# `docker compose -f docker-compose.tests.yml run --rm --build unit-tests`
# keeps its documented baseline (145 passed / 1 skipped): were this suite
# discovered there too it would report a SECOND skip, honestly, but a
# baseline that drifts every time a new docker-dependent suite is added is
# not a baseline anyone can gate on. Run it explicitly, on the host.
NEEDS_DOCKER = {"test_real_image_operator_scripts.py"}

PASS, FAIL, SKIP, INCONCLUSIVE = "PASS", "FAIL", "SKIP", "INCONCLUSIVE"


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
    if r.returncode != 1:
        # This suite's own contract is 0/3/1 (pass/skip/fail — see the module
        # docstring). Anything else is not this suite telling us it failed;
        # it is the PROCESS ending some other way — most often 137 (128+9,
        # SIGKILL), which the compose stacks' fixed container_names (since
        # removed) made a real hazard: a second concurrent run's cleanup filter could kill a
        # first run's still-in-flight container. Folding that into FAIL used
        # to let an operator read "the suite failed" off a kill it never
        # asked for; a negative Python returncode (POSIX: killed by signal
        # -N) is the same situation from this side of the subprocess call.
        # Reporting it as its own status means a 137 can no longer be
        # silently scored as either a pass or a fail — see
        # hostile2-apparatus.md, "every stack pins container_name".
        sig = f" (signal {-r.returncode})" if r.returncode < 0 else ""
        # ASCII only, same reason as the SKIP banner below: an em dash here
        # renders as a replacement character on the Windows console.
        return (INCONCLUSIVE, dt,
                f"exited {r.returncode}{sig}, not 0/1/3 - the process ended "
                f"some other way (killed? crashed before the harness could "
                f"report?), not this suite reporting pass/fail/skip")
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
    ap.add_argument("--real-image", action="store_true",
                    help="include the real-image operator-script suite "
                         "(needs docker + the published image + the real "
                         "backup path on this host; off by default)")
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
    if not args.real_image:
        suites = [(d, p) for d, p in suites if p.name not in NEEDS_DOCKER]
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
        mark = {PASS: "ok  ", FAIL: "FAIL", SKIP: "SKIP",
                INCONCLUSIVE: "????"}[status]
        line = f"  {mark} {script.name:<44}{dt:6.1f}s"
        print(line + (f"  {note}" if note else ""), flush=True)

    n = {s: sum(1 for r in results if r[0] == s)
         for s in (PASS, FAIL, SKIP, INCONCLUSIVE)}
    print("-" * 72)
    print(f"{n[PASS]} passed, {n[FAIL]} failed, {n[SKIP]} skipped, "
          f"{n[INCONCLUSIVE]} inconclusive in {time.monotonic() - t_all:.0f}s")

    if n[FAIL]:
        print("\nFAILED:")
        for s, name, _, note in results:
            if s == FAIL:
                print(f"  {name}: {note}")
    if n[INCONCLUSIVE]:
        print("\nINCONCLUSIVE - these did not report pass, fail or skip; "
              "the process ended some other way (see notes) and this is "
              "NOT evidence of anything, including a failure:")
        for s, name, _, note in results:
            if s == INCONCLUSIVE:
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

    if n[FAIL] or n[INCONCLUSIVE]:
        # INCONCLUSIVE folds into the same exit code as FAIL rather than
        # getting a fifth code of its own: this runner's own exit-code
        # contract (0/1/2/3, documented above) is read by other scripts, and
        # a run that could not tell whether a suite passed must not be able
        # to report 0. It is kept OUT of n[FAIL]'s own count and printed
        # under its own heading above precisely so a reader does not mistake
        # it for a suite that actually failed its checks.
        return 1
    if n[SKIP] and not args.allow_skips:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
