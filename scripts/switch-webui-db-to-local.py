#!/usr/bin/env python3
"""Move OpenWebUI's live database onto LOCAL disk, on a running pod.

    /opt/compactor-venv/bin/python /data/scripts/switch-webui-db-to-local.py --check
    /opt/compactor-venv/bin/python /data/scripts/switch-webui-db-to-local.py --apply

This is the hot-patch form of the v3.1.6 change, for a pod that cannot wait
for a rebuild. It does exactly what the built image will do, by editing the
running container's supervisord config instead of the one baked into the
image:

  * the LIVE database moves to /var/lib/openwebui/webui.db (local disk)
  * OpenWebUI is pointed at it via DATABASE_URL
  * a webuidb-sync daemon publishes snapshots back to /data on a timer

Why: RunPod's MooseFS mount drops I/O. SQLite responds by leaving a hot
rollback journal it then cannot roll back, and the front end dies with
"attempt to write a readonly database". Local disk removes the network from
SQLite's write path entirely.

IT DOES NOT SURVIVE THE CONTAINER BEING RECREATED. Everything it changes
lives in the container's filesystem. After a pod recreate, either re-run it
or deploy the built image. It says so again at the end.

IDEMPOTENT: running it twice changes nothing the second time.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

SUPERVISOR_CONF = Path(
    os.environ.get("SUPERVISOR_CONF", "/etc/supervisor/conf.d/supervisord.conf")
)
LOCAL_DB = Path(os.environ.get("WEBUI_LOCAL_DB", "/var/lib/openwebui/webui.db"))
SNAPSHOT_DB = Path(os.environ.get("WEBUI_SNAPSHOT_DB", "/data/openwebui/webui.db"))
COMPACTOR_DIR = Path(os.environ.get("COMPACTOR_DIR", "/opt/compactor"))
VENV_PY = os.environ.get("VENV_PY", "/opt/compactor-venv/bin/python")
LOG_DIR = os.environ.get("LOG_DIR", "/data/logs")
SYNC_INTERVAL = os.environ.get("WEBUI_DB_SYNC_INTERVAL_S", "300")

MARKER = "; --- webuidb local-disk hot-patch ---"


def say(m=""):
    print(m, flush=True)


def sh(*args, **kw):
    return subprocess.run(args, capture_output=True, text=True, timeout=180, **kw)


def build_sync_program() -> str:
    return f"""
{MARKER}
; Publishes the LOCAL webui.db back to /data periodically, so a pod recreate
; costs at most WEBUI_DB_SYNC_INTERVAL_S of chat history. A stalled /data
; makes this log a warning; it can no longer take the front end down.
[program:webuidb-sync]
command={VENV_PY} {COMPACTOR_DIR}/webuidb.py
directory={COMPACTOR_DIR}
autostart=true
autorestart=true
priority=25
startsecs=5
environment=WEBUI_LOCAL_DB="{LOCAL_DB}",WEBUI_SNAPSHOT_DB="{SNAPSHOT_DB}",WEBUI_DB_SYNC_INTERVAL_S="{SYNC_INTERVAL}"
stdout_logfile={LOG_DIR}/webuidb-sync.log
stderr_logfile={LOG_DIR}/webuidb-sync-error.log
stdout_logfile_maxbytes=10MB
stdout_logfile_backups=2
stderr_logfile_maxbytes=10MB
stderr_logfile_backups=2
"""


def patch_supervisor(text: str) -> tuple[str, list[str]]:
    """Return (new_text, notes). Idempotent."""
    notes = []
    dburl = f'sqlite:///{LOCAL_DB}'

    # 1. point OpenWebUI at the local database
    if f'DATABASE_URL="{dburl}"' in text:
        notes.append("openwebui already points at the local database")
    else:
        m = re.search(r"^\[program:openwebui\]\s*$", text, re.M)
        if not m:
            raise RuntimeError("no [program:openwebui] section in the supervisor config")
        section_start = m.end()
        nxt = re.search(r"^\[program:", text[section_start:], re.M)
        section_end = section_start + (nxt.start() if nxt else len(text) - section_start)
        section = text[section_start:section_end]

        # environment= in supervisord is an INI value that CONTINUES onto
        # indented following lines, and the real config uses that form:
        #
        #     environment=
        #         HF_HOME="...",
        #         MODEL_REPO="...",
        #         VLLM_URL="..."
        #
        # Matching only the FIRST line produced `environment=,DATABASE_URL=`
        # and orphaned the three real variables below it - caught in testing,
        # and it would have started OpenWebUI without HF_HOME or VLLM_URL.
        # So consume the whole block: the key line plus every indented
        # continuation after it.
        env_re = "^environment=(.*(?:" + chr(92) + "n[ \t]+.*)*)$"
        env_line = re.search(env_re, section, re.M)
        if env_line:
            existing = env_line.group(1).strip().rstrip(",")
            if existing:
                merged = (
                    "environment=" + existing + "," + chr(10)
                    + '    DATABASE_URL="' + dburl + '"'
                )
            else:
                merged = 'environment=DATABASE_URL="' + dburl + '"'
            section = section[: env_line.start()] + merged + section[env_line.end():]
            kept = len([x for x in existing.split(",") if x.strip()])
            notes.append(
                "appended DATABASE_URL to openwebui's existing environment= "
                "(" + str(kept) + " var(s) preserved)"
            )
        else:
            section = "\n" + f'environment=DATABASE_URL="{dburl}"' + section
            notes.append("added environment=DATABASE_URL to [program:openwebui]")
        text = text[:section_start] + section + text[section_end:]

    # 2. the sync daemon
    if "[program:webuidb-sync]" in text:
        notes.append("webuidb-sync program already present")
    else:
        text = text.rstrip() + "\n" + build_sync_program()
        notes.append("added [program:webuidb-sync]")
    return text, notes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if not (args.apply or args.check):
        args.check = True

    say("=" * 66)
    say(f"{'APPLY' if args.apply else 'CHECK'}: move webui.db to local disk")
    say("=" * 66)

    # --- preconditions ----------------------------------------------------
    module = COMPACTOR_DIR / "webuidb.py"
    if not module.exists():
        say(f"FAIL: {module} is missing. Install it first:")
        say("  curl -fsSL -o /opt/compactor/webuidb.py \\")
        say("    https://raw.githubusercontent.com/MrBanana8768/zions-light-ai/"
            "<sha>/compactor/webuidb.py")
        return 2
    if not SUPERVISOR_CONF.exists():
        say(f"FAIL: {SUPERVISOR_CONF} not found (set SUPERVISOR_CONF)")
        return 2

    sys.path.insert(0, str(COMPACTOR_DIR))
    os.environ.setdefault("WEBUI_LOCAL_DB", str(LOCAL_DB))
    os.environ.setdefault("WEBUI_SNAPSHOT_DB", str(SNAPSHOT_DB))
    import webuidb  # noqa: E402

    say("")
    say("current state:")
    for label, p in (("local   ", LOCAL_DB), ("snapshot", SNAPSHOT_DB)):
        if p.exists():
            ok, detail = webuidb.integrity(p)
            say(f"  {label} {p}  {p.stat().st_size/1e6:.1f} MB  "
                f"quick_check={detail}  chats={webuidb._has_rows(p)}")
        else:
            say(f"  {label} {p}  (absent)")

    text = SUPERVISOR_CONF.read_text()
    new_text, notes = patch_supervisor(text)
    say("")
    say("supervisor config changes:")
    for n in notes:
        say(f"  - {n}")

    if not args.apply:
        say("")
        say("CHECK ONLY — nothing changed. Re-run with --apply.")
        return 0

    # --- 1. stop the writer ----------------------------------------------
    say("")
    say("[1/5] stopping OpenWebUI")
    say("      " + (sh("supervisorctl", "stop", "openwebui").stdout.strip() or "?"))
    time.sleep(2)

    # --- 2. place the live database on local disk -------------------------
    say("[2/5] placing the live database on local disk")
    r = webuidb.restore_on_boot()
    say(f"      {r['action']}")
    if r["action"] in ("snapshot_unhealthy", "restore_failed", "error"):
        say("")
        say("FAIL: could not establish a local database. NOTHING has been")
        say("      switched over; OpenWebUI still points at /data. Recover the")
        say("      snapshot first:")
        say("        /opt/compactor-venv/bin/python /data/scripts/recover-webui-db.py --check")
        sh("supervisorctl", "start", "openwebui")
        return 3
    if LOCAL_DB.exists():
        say(f"      local: {LOCAL_DB.stat().st_size/1e6:.1f} MB, "
            f"{webuidb._has_rows(LOCAL_DB)} chats")

    # --- 3. patch supervisor ---------------------------------------------
    say("[3/5] updating the supervisor config")
    backup = SUPERVISOR_CONF.with_suffix(
        SUPERVISOR_CONF.suffix + f".pre-webuidb-{time.strftime('%Y%m%d-%H%M%S')}"
    )
    shutil.copy2(SUPERVISOR_CONF, backup)
    SUPERVISOR_CONF.write_text(new_text)
    say(f"      previous config saved to {backup}")

    # --- 4. reload --------------------------------------------------------
    say("[4/5] reloading supervisor")
    say("      " + (sh("supervisorctl", "reread").stdout.strip() or "?"))
    say("      " + (sh("supervisorctl", "update").stdout.strip() or "?"))
    sh("supervisorctl", "start", "openwebui")
    time.sleep(8)

    # --- 5. verify --------------------------------------------------------
    say("[5/5] verifying")
    status = sh("supervisorctl", "status").stdout
    for line in status.splitlines():
        if line.split()[:1] and line.split()[0] in ("openwebui", "webuidb-sync"):
            say(f"      {line.strip()}")

    # Did OpenWebUI actually open the LOCAL file? If DATABASE_URL had not
    # taken, it would quietly be back on /data and this whole exercise would
    # have achieved nothing while looking fine.
    time.sleep(3)
    fresh = LOCAL_DB.exists() and (time.time() - LOCAL_DB.stat().st_mtime) < 120
    say(f"      local database touched since start: {fresh}"
        f"{'' if fresh else '  <-- check DATABASE_URL took effect'}")

    say("")
    say("=" * 66)
    say("SWITCHED. The live database is on local disk; /data now receives")
    say(f"snapshots every {SYNC_INTERVAL}s.")
    say("")
    say("THIS DOES NOT SURVIVE THE CONTAINER BEING RECREATED — everything")
    say("changed here lives in the container filesystem. Deploy the built")
    say("image, or re-run this script after a recreate.")
    say("")
    say("Roll back:  supervisorctl stop webuidb-sync openwebui")
    say(f"            cp {backup} {SUPERVISOR_CONF}")
    say("            supervisorctl reread && supervisorctl update")
    say("            supervisorctl start openwebui")
    say("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
