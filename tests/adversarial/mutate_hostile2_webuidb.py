"""Mutation harness for AREA 2 (webuidb). Runs INSIDE the container against
/work (a copy of /src), so nothing in the worktree is ever modified.

Every needle's occurrence count is asserted == 1 before the mutation is
applied; a count of 0 or >1 is reported as ABORTED, never as a result.
"""
import shutil
import subprocess
import sys
from pathlib import Path

SRC = Path("/src")
WORK = Path("/work")
MOD = "compactor/webuidb.py"
SUITES = [
    "test_webuidb_restore_guard.py",
    "test_webuidb.py",
    "test_webuidb_migration.py",
    "test_webuidb_gate.py",
]

MUTATIONS = [
    # (id, needle, replacement, what it removes)
    ("M1-chat-floor",
     "                previous >= SHRINK_GUARD_MIN_CHATS",
     "                previous >= 10",
     "the chat floor goes back to the off-switch value 10"),
    ("M2-content-measure",
     "                    lost_content = new_content < prev_content * SHRINK_REFUSE_BELOW",
     "                    lost_content = False",
     "the stored-content half of the shrink guard never fires"),
    ("M3-content-floor",
     "                elif prev_content >= SHRINK_GUARD_MIN_BYTES:",
     "                elif False:",
     "the content measure is never reached"),
    ("M4-four-state",
     "            if previous is None:",
     "            if False:",
     "the unreadable/empty-snapshot split disappears"),
    ("M5-unreadable-refusal",
     "                elif not ALLOW_PUBLISH_OVER_UNREADABLE:",
     "                elif False:",
     "publishing over an unreadable snapshot is allowed"),
    ("M6-empty-start",
     "        if EMPTY_START_MARKER.exists() and SNAPSHOT_DB.exists():",
     "        if False:",
     "the .empty-start refusal is gone"),
    ("M7-schema-blindspot",
     "            if not cols:\n                return None",
     "            if not cols:\n                return 0",
     "no `chat` table measures as zero content instead of unknown"),
    ("M8-exit-default",
     "        sys.exit(RESTORE_EXIT_CODES.get(r.get(\"action\"), 6))",
     "        sys.exit(RESTORE_EXIT_CODES.get(r.get(\"action\"), 0))",
     "an unknown restore action exits 0"),
    ("M9-set-aside-sidecars",
     "        for suffix in (\"\",) + SIDECARS:",
     "        for suffix in (\"\",):",
     "_set_aside stops moving -journal/-wal/-shm"),
    ("M10-set-aside-failure",
     "            if not _set_aside(LOCAL_DB, \"failed-quickcheck\"):",
     "            if False:",
     "a failed set-aside no longer stops the restore overwriting it"),
    ("M11-snapshot-utime",
     "            os.utime(SNAPSHOT_DB, (mtime, mtime))",
     "            pass",
     "the snapshot keeps the temp file's own mtime"),
    ("M12-zero-chat-refusal",
     "        if not chats:",
     "        if chats is None:",
     "a 0-chat new image is publishable"),
    ("M13-empty-local",
     "        local_chats = _has_rows(LOCAL_DB) if ok else None",
     "        local_chats = 1 if ok else None",
     "an empty local shell always looks populated"),
    ("M14-shrink-ratio",
     "SHRINK_REFUSE_BELOW = env_float(\"WEBUI_DB_SHRINK_REFUSE_BELOW\", 0.5)\n# The floor",
     "SHRINK_REFUSE_BELOW = env_float(\"WEBUI_DB_SHRINK_REFUSE_BELOW\", 0.0)\n# The floor",
     "the shrink ratio floor is 0 (nothing is ever a shrink)"),
    ("M15-mtime-skip",
     "            if SNAPSHOT_DB.stat().st_mtime >= mtime:",
     "            if False:",
     "the unchanged-since-last-sync skip is gone"),
]


def prepare() -> None:
    if WORK.exists():
        shutil.rmtree(WORK)
    shutil.copytree(SRC, WORK, symlinks=True)


def run_suites() -> dict:
    res = {}
    for s in SUITES:
        p = subprocess.run(
            ["/opt/compactor-venv/bin/python", s],
            cwd=str(WORK / "compactor"), capture_output=True, text=True, timeout=900,
        )
        res[s] = p.returncode
    return res


print("=" * 78)
print("BASELINE (unmutated copy)")
print("=" * 78)
prepare()
base = run_suites()
for s, rc in base.items():
    print(f"    {s:40} rc={rc}")
if any(rc != 0 for rc in base.values()):
    print("BASELINE IS NOT GREEN - every mutation result below is meaningless.")
    sys.exit(2)

survivors = []
for mid, needle, repl, what in MUTATIONS:
    prepare()
    target = WORK / MOD
    src = target.read_text(encoding="utf-8")
    n = src.count(needle)
    if n != 1:
        print()
        print(f"### {mid}: ABORTED - needle occurs {n} times, not 1. No result.")
        continue
    target.write_bytes(src.replace(needle, repl, 1).encode("utf-8"))
    res = run_suites()
    red = [s for s, rc in res.items() if rc != 0]
    verdict = "RED" if red else "GREEN - NOT COVERED"
    print()
    print(f"### {mid}  ({what})")
    print(f"    needle occurrences: {n}")
    for s, rc in res.items():
        print(f"    {s:40} rc={rc}")
    print(f"    => {verdict}" + (f" (red: {', '.join(red)})" if red else ""))
    if not red:
        survivors.append((mid, needle, what))

print()
print("=" * 78)
print(f"SURVIVING MUTANTS: {len(survivors)} of {len(MUTATIONS)}")
for mid, needle, what in survivors:
    print(f"  - {mid}: {what}")
    print(f"      needle (1 occurrence): {needle.strip()!r}")
print("=" * 78)
