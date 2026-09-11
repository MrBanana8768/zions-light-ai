"""WEBUI_DB_LOCAL gates the one change that moves her chat history.

WHY THE FLAG EXISTS. The v3.1.4.x rollout deploys v3.1.5 -> v3.1.6 -> v3.1.7 ->
v3.1.8 in steps, so a problem found at one step can be fixed before the next.
But the local-disk move is INSIDE v3.1.6 and was unconditional, so "deploy
v3.1.6" and "move the live database" were one action that could not be
separated by version ordering.

They needed separating because they are not alike. Every other step in the
series rolls back by redeploying the previous image. This one has MOVED the
live database: /data then holds a snapshot as old as the last sync, and
rolling back means choosing between a stale copy and a hand repair. It is the
only irreversible step in the line, so it goes last, alone, with everything
else already proven.

THE DANGEROUS CASE, which is what this file exists for. The sync daemon
publishes the LOCAL file over the snapshot path. With the gate off, the live
database IS at that snapshot path — so a daemon left running while the gate is
off would overwrite her real chat history with whatever stale copy happens to
be on local disk. "Off" has to mean the daemon does not start at all, not that
it starts and is pointed somewhere else.

This reads the shipped entrypoint.sh and supervisord.conf as text rather than
executing them: the branching is four lines of shell, and what must be true is
a property of what those files SAY.

    python test_webuidb_gate.py
"""

import pathlib
import re
import sys

FAILED: list[str] = []


def check(cond, label):
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"FAIL {label}")
        FAILED.append(label)


ROOT = pathlib.Path(__file__).resolve().parent.parent
ENTRY = (ROOT / "entrypoint.sh").read_text(encoding="utf-8", errors="replace")
SUPER = (ROOT / "supervisord.conf").read_text(encoding="utf-8", errors="replace")


def _gate_block(branch: str) -> str:
    """The text of one side of the `if [ "${WEBUI_DB_LOCAL}" = "true" ]` branch."""
    start = ENTRY.index('if [ "${WEBUI_DB_LOCAL}" = "true" ]; then')
    else_at = ENTRY.index("\nelse", start)
    end = ENTRY.index("\nfi", else_at)
    return ENTRY[start:else_at] if branch == "on" else ENTRY[else_at:end]


print("[1] the flag exists and defaults to the fix being ON")
check('WEBUI_DB_LOCAL="${WEBUI_DB_LOCAL:-true}"' in ENTRY,
      "WEBUI_DB_LOCAL defaults to true — the local-disk move is the right "
      "default and stays it; the rollout opts OUT for a few steps rather than "
      "the image shipping the bug by default")
check('if [ "${WEBUI_DB_LOCAL}" = "true" ]; then' in ENTRY,
      "and the boot block branches on it")

print("[2] gate ON does what v3.1.6 always did")
on = _gate_block("on")
check("webuidb.py --restore" in on,
      "the snapshot is restored to local disk on a fresh container")
check("WEBUIDB_SYNC_ENABLED=true" in on, "and the sync daemon is enabled")
check("sqlite:///${WEBUI_LOCAL_DB}" in on,
      "and OpenWebUI is pointed at the LOCAL file")

print("[3] gate OFF leaves the database exactly where v3.1.5 had it")
off = _gate_block("off")
check("sqlite:///${WEBUI_SNAPSHOT_DB}" in off,
      "OpenWebUI is pointed at the /data path, unmoved")
check("--restore" not in off,
      "nothing is restored — with the live database already at that path, a "
      "restore would write over it")

print("[4] THE SAFETY PROPERTY: off means the daemon does not start")
check("WEBUIDB_SYNC_ENABLED=false" in off,
      "the gate-off branch disables the sync daemon explicitly")
check("WEBUIDB_SYNC_ENABLED=true" not in off,
      "and does NOT enable it — a daemon publishing the local file over the "
      "live database is the one way this flag could destroy data")

print("[5] supervisord honours the flag rather than a constant")
m = re.search(r"\[program:webuidb-sync\](.*?)(?=\n\[program:|\Z)", SUPER, re.S)
check(m is not None, "the webuidb-sync program is still defined")
if m:
    block = m.group(1)
    check("autostart=%(ENV_WEBUIDB_SYNC_ENABLED)s" in block,
          "its autostart reads the environment variable")
    check("autostart=true" not in block,
          "and no literal autostart=true survives beside it — two autostart "
          "lines in one program is a silent win for whichever comes last")

print("[6] the variable is always exported, on both paths")
# supervisord dies at startup on an %(ENV_x)s it cannot resolve, so an
# unexported variable is not a disabled daemon, it is a container that will
# not boot.
check(len(re.findall(r"export WEBUIDB_SYNC_ENABLED=", ENTRY)) == 2,
      "exported on BOTH branches — supervisord refuses to start at all on an "
      "%(ENV_x)s it cannot resolve, so a missing export is a dead pod rather "
      "than a stopped daemon")

if FAILED:
    print(f"\n{len(FAILED)} check(s) FAILED")
    sys.exit(1)
print("\nAll webuidb-gate checks passed.")
