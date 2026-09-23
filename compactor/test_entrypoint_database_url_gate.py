"""compactor/test_entrypoint_database_url_gate.py — D12 (findings.md, the
2026-09-23 WEBUI_DB_LOCAL rehearsal).

entrypoint.sh's WEBUI_DB_LOCAL=true branch used to derive DATABASE_URL with
`export DATABASE_URL="${DATABASE_URL:-sqlite:///${WEBUI_LOCAL_DB}}"` — which
ONLY sets it when unset. A DATABASE_URL row left on the RunPod template from
before the move (or set by hand) silently wins: OpenWebUI keeps writing
wherever THAT points — almost always still /data — while webuidb.py's sync
daemon believes the live database is WEBUI_LOCAL_DB and publishes whatever
stale or empty file sits there OVER the real one on every cycle.
DB-MOVE-RUNBOOK.md's own template table says this row "must NOT exist" once
WEBUI_DB_LOCAL is true.

The fix is a refusal, added between "# BEGIN/END DATABASE_URL GATE CHECK"
markers in entrypoint.sh's WEBUI_DB_LOCAL=true branch: boot stops (exit 1,
with a banner) whenever DATABASE_URL is set to anything other than exactly
`sqlite:///${WEBUI_LOCAL_DB}`.

This file extracts and runs that REAL block under `sh` (same technique as
test_config_supervisord_bool.py, and for the same reason: this must be the
code that actually boots the pod, not a re-implementation of what it is
supposed to do) with every combination of DATABASE_URL / WEBUI_LOCAL_DB this
finding names.

    python test_entrypoint_database_url_gate.py
"""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / "entrypoint.sh"

FAILED = []


def check(cond, label):
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}")
        FAILED.append(label)


def extract_block(text: str, begin: str, end: str) -> str:
    # Same technique as test_config_supervisord_bool.py's extract_block:
    # skip to the end of the BEGIN marker's own line (the rest of that line
    # is more comment prose, not something dash can parse as code), then
    # slice up to (not including) the END marker's line.
    i = text.index(begin)
    line_end = text.index("\n", i)
    j = text.index(end, line_end)
    return text[line_end + 1:j]


ENTRYPOINT_SRC = ENTRYPOINT.read_text(encoding="utf-8")
GATE_BLOCK = extract_block(
    ENTRYPOINT_SRC,
    "# BEGIN DATABASE_URL GATE CHECK",
    "# END DATABASE_URL GATE CHECK",
)


def _sh_quote(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


def run_gate(webui_local_db: str, database_url: str | None) -> tuple[int, str]:
    """Run the REAL extracted block under `sh`, with WEBUI_LOCAL_DB set and
    DATABASE_URL either set to `database_url` or left unset entirely (when
    None). Returns (exit code, combined output). Exit 1 with output
    containing "REFUSING TO START" is the refusal; exit 0 with no such text
    is "let the boot continue"."""
    lines = [f"WEBUI_LOCAL_DB={_sh_quote(webui_local_db)}"]
    if database_url is not None:
        lines.append(f"DATABASE_URL={_sh_quote(database_url)}")
    lines.append(GATE_BLOCK)
    lines.append('echo "GATE DID NOT REFUSE"')
    script = "\n".join(lines) + "\n"
    proc = subprocess.run(
        ["sh", "-c", script],
        capture_output=True, text=True, timeout=10,
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


if shutil.which("sh") is None:
    print("SKIP: no `sh` on PATH — this file needs the unit-tests container "
          "(or any POSIX shell) to exercise the real entrypoint.sh block.")
    sys.exit(3)  # 3 = skipped, per this project's runner convention — never a pass


LOCAL = "/var/lib/openwebui/webui.db"
EXPECTED = f"sqlite:///{LOCAL}"

print("[D12] the real entrypoint.sh gate, driven for every case in the finding")
print()

rc, out = run_gate(LOCAL, None)
check(rc == 0 and "REFUSING" not in out and "GATE DID NOT REFUSE" in out,
      "CONTROL: DATABASE_URL unset — the boot is not refused, this is the "
      "ordinary shipped case")

rc, out = run_gate(LOCAL, EXPECTED)
check(rc == 0 and "REFUSING" not in out and "GATE DID NOT REFUSE" in out,
      f"CONTROL: DATABASE_URL already set to EXACTLY the expected value "
      f"({EXPECTED!r}) — not refused, the two agree")

rc, out = run_gate(LOCAL, "")
check(rc == 0 and "REFUSING" not in out and "GATE DID NOT REFUSE" in out,
      "CONTROL: DATABASE_URL set to the EMPTY string reads as 'unset' here, "
      "consistent with the `${DATABASE_URL:-...}` default-assignment this "
      "gate stands in front of — both treat empty the same way")

rc, out = run_gate(LOCAL, "sqlite:////data/openwebui/webui.db")
check(rc == 1, f"the stale-/data-path case (D12's own example) REFUSES "
      f"(rc={rc})")
check("REFUSING TO START" in out, "and says so, loudly")
check("/data/openwebui/webui.db" in out, "naming the bad value")
check(EXPECTED in out, "and naming what it expected instead")
check("DB-MOVE-RUNBOOK" in out, "pointing at the runbook that documents the rule")

rc, out = run_gate(LOCAL, "postgresql://user:pass@host/db")
check(rc == 1, f"a totally different (non-sqlite) DATABASE_URL still "
      f"refuses, not just the /data-sqlite case (rc={rc})")
check("REFUSING TO START" in out, "and says so")

rc, out = run_gate(LOCAL, "sqlite:///var/lib/openwebui/webui.db")
check(rc == 1, "a value that is CLOSE but not byte-identical (missing one "
      "of the three slashes) still refuses rather than guessing it was "
      "probably meant the same way (rc=%d)" % rc)

print()
print("    CONTROL: a DIFFERENT WEBUI_LOCAL_DB changes what counts as "
      "'agrees', not whether the gate can fire at all")
OTHER_LOCAL = "/mnt/fast/webui.db"
rc, out = run_gate(OTHER_LOCAL, f"sqlite:///{OTHER_LOCAL}")
check(rc == 0 and "REFUSING" not in out,
      "matching the (non-default) WEBUI_LOCAL_DB — not refused")
rc, out = run_gate(OTHER_LOCAL, EXPECTED)  # the OTHER path's expected value
check(rc == 1, "the DEFAULT path's URL, now stale under a non-default "
      "WEBUI_LOCAL_DB, still refuses (rc=%d)" % rc)

print()
if FAILED:
    for f in FAILED:
        print("FAIL " + f)
    sys.exit(1)
print("All entrypoint.sh DATABASE_URL-gate checks passed.")
