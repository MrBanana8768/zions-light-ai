"""MUTATION HARNESS for the guards 6630134 added (hostile pass 2, AREA 4).

Runs INSIDE the unit container against /work/compactor (the copy made by
`cp -r /src /work`), so nothing in the worktree is ever edited:

  docker compose -f docker-compose.tests.yml run --rm --entrypoint /bin/bash \
    unit-tests -c 'cp -r /src /work && cd /work/compactor && \
    /opt/compactor-venv/bin/python /work/tests/adversarial/mutate_hostile2_health.py'

Every needle is asserted to occur EXACTLY ONCE in the pristine file before
anything is written. A needle that matches 0 or 2+ lines is reported as
ABORTED and no result is claimed for it.
"""

import subprocess
import sys

TARGET = "/work/compactor/health.py"
SUITE = ["/opt/compactor-venv/bin/python", "test_health.py"]
CWD = "/work/compactor"

with open(TARGET, "rb") as fh:
    PRISTINE = fh.read()


def run_suite():
    p = subprocess.run(SUITE, cwd=CWD, capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr)


def restore():
    with open(TARGET, "wb") as fh:
        fh.write(PRISTINE)


# (id, needle, replacement, what it proves)
MUTATIONS = [
    ("MH1  (commit's own) probe_snapshot never reports stale",
     b'    out["stale"] = age > 3 * interval\n',
     b'    out["stale"] = False\n',
     "the staleness computation is asserted"),

    ("MH4  (commit's own) the lag limit becomes 0",
     b"        _lag_limit = 2 * summarizer.L1_CHUNK_SIZE\n",
     b"        _lag_limit = 0\n",
     "the control at one chunk of drift is real"),

    ("N1   the lag limit becomes ONE chunk instead of two",
     b"        _lag_limit = 2 * summarizer.L1_CHUNK_SIZE\n",
     b"        _lag_limit = 1 * summarizer.L1_CHUNK_SIZE\n",
     "whether the 'TWO CHUNKS, NOT ZERO' constant is actually pinned"),

    ("N2   the lag limit becomes THREE chunks",
     b"        _lag_limit = 2 * summarizer.L1_CHUNK_SIZE\n",
     b"        _lag_limit = 3 * summarizer.L1_CHUNK_SIZE\n",
     "whether the limit is pinned from above"),

    ("N3   the STALE SNAPSHOT never reaches status",
     b'        _snap = blocking.get("snapshot") or {}\n        if _snap.get("stale"):\n',
     b'        _snap = blocking.get("snapshot") or {}\n        if False and _snap.get("stale"):\n',
     "whether any test asserts the snapshot degrades the pod"),

    ("N4   the snapshot probe is dropped from checks{} entirely",
     b'            "snapshot": blocking.get("snapshot"),\n',
     b'            "snapshot": None,\n',
     "whether any test reads checks.snapshot"),

    ("N5   the staleness threshold becomes 3.5 intervals",
     b'    out["stale"] = age > 3 * interval\n',
     b'    out["stale"] = age > 3.5 * interval\n',
     "whether 'three intervals, not one' is pinned from above"),

    ("N6   WEBUIDB_SYNC_ENABLED becomes a != 'false' test",
     b'    enabled = os.environ.get("WEBUIDB_SYNC_ENABLED", "").strip().lower() == "true"\n',
     b'    enabled = os.environ.get("WEBUIDB_SYNC_ENABLED", "").strip().lower() != "false"\n',
     "whether the exact truthiness spelling is pinned"),

    ("N7   probe_snapshot reports watched but never looks at the file",
     b'    if not enabled:\n        return out\n',
     b'    if not enabled or True:\n        return out\n',
     "whether the enabled path is exercised end to end"),

    ("N8   hierarchy_lag_conv is never recorded",
     b"                    worst_lag, worst_lag_conv = _lag, cid\n",
     b"                    worst_lag = _lag\n",
     "whether the conversation name is asserted"),
]

print("=" * 74)
print("BASELINE (pristine health.py)")
print("=" * 74)
rc, out = run_suite()
print(f"  test_health.py exit={rc}")
if rc != 0:
    print(out[-3000:])
    print("BASELINE IS NOT GREEN -- no mutation result below means anything.")
    sys.exit(1)

RESULTS = []
for name, needle, repl, proves in MUTATIONS:
    n = PRISTINE.count(needle)
    print("")
    print("-" * 74)
    print(name)
    print(f"  needle occurrences in pristine file: {n}")
    if n != 1:
        print("  ABORTED -- occurrence count is not exactly 1. NO RESULT CLAIMED.")
        RESULTS.append((name, "ABORTED", proves))
        continue
    with open(TARGET, "wb") as fh:
        fh.write(PRISTINE.replace(needle, repl))
    rc, out = run_suite()
    restore()
    verdict = "RED (caught)" if rc != 0 else "GREEN (NOT COVERED)"
    print(f"  test_health.py exit={rc}  ->  {verdict}")
    if rc != 0:
        fails = [ln for ln in out.splitlines() if ln.startswith("FAIL")]
        for ln in fails[:3]:
            print(f"    {ln}")
    print(f"  proves: {proves}")
    RESULTS.append((name, verdict, proves))

print("")
print("=" * 74)
print("BASELINE AFTER RESTORE")
print("=" * 74)
rc, out = run_suite()
print(f"  test_health.py exit={rc}")

print("")
print("=" * 74)
print("SUMMARY")
print("=" * 74)
for name, verdict, proves in RESULTS:
    print(f"  {verdict:<20} {name}")
print("")
green = [r for r in RESULTS if r[1].startswith("GREEN")]
print(f"  mutations that survived: {len(green)}")
for name, _, proves in green:
    print(f"    - {name}  ({proves})")
print("=" * 74)
