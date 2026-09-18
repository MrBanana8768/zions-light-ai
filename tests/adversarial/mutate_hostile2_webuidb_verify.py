"""Re-run ONLY the four surviving mutants and confirm from the LOG BODY (not
the exit code) that each suite actually ran and reported a pass line.

Under concurrent docker use an exit code alone is not evidence: 137 is SIGKILL
from another agent's cleanup, and the suite never ran. Each row below prints
the returncode, whether the suite's own final success sentence appeared on
stdout, whether any 'FAIL ' line appeared, and how many 'ok ' checks ran.
"""
import shutil
import subprocess
import sys
from pathlib import Path

SRC, WORK, MOD = Path("/src"), Path("/work"), "compactor/webuidb.py"
SUITES = ["test_webuidb_restore_guard.py", "test_webuidb.py",
          "test_webuidb_migration.py", "test_webuidb_gate.py"]
SURVIVORS = [
    ("M7-schema-blindspot",
     "            if not cols:\n                return None",
     "            if not cols:\n                return 0"),
    ("M9-set-aside-sidecars",
     "        for suffix in (\"\",) + SIDECARS:",
     "        for suffix in (\"\",):"),
    ("M11-snapshot-utime",
     "            os.utime(SNAPSHOT_DB, (mtime, mtime))",
     "            pass"),
    ("M12-zero-chat-refusal",
     "        if not chats:",
     "        if chats is None:"),
]


def prepare():
    if WORK.exists():
        shutil.rmtree(WORK)
    shutil.copytree(SRC, WORK, symlinks=True)


def run(label):
    for s in SUITES:
        p = subprocess.run(["/opt/compactor-venv/bin/python", s],
                           cwd=str(WORK / "compactor"),
                           capture_output=True, text=True, timeout=900)
        lines = [ln for ln in p.stdout.splitlines() if ln.strip()]
        last = lines[-1] if lines else "<NO STDOUT AT ALL>"
        fails = [ln for ln in lines if ln.startswith("FAIL ")]
        oks = sum(1 for ln in lines if ln.strip().startswith("ok "))
        print(f"  {label:24} {s:34} rc={p.returncode:<4} "
              f"ok_checks={oks:<4} FAIL_lines={len(fails)}")
        print(f"      last stdout line: {last[:100]!r}")
        if p.returncode == 137:
            print("      *** rc=137 is SIGKILL, NOT a result. Re-run needed.")
        if fails:
            print(f"      first FAIL: {fails[0][:140]}")


print("=" * 78)
print("BASELINE, log body checked")
print("=" * 78)
prepare()
run("baseline")

for mid, needle, repl in SURVIVORS:
    print()
    print("=" * 78)
    print(f"{mid}, log body checked")
    print("=" * 78)
    prepare()
    t = WORK / MOD
    src = t.read_text(encoding="utf-8")
    n = src.count(needle)
    print(f"  needle occurrences: {n}")
    if n != 1:
        print("  ABORTED - not a mutation. No result.")
        continue
    t.write_bytes(src.replace(needle, repl, 1).encode("utf-8"))
    run(mid)
print()
print("done")
