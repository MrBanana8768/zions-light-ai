"""compactor/test_supervisord_cloudflared.py — v3.1.9.9 Cloudflare Tunnel,
task 10: a static parse test on the shipped supervisord.conf, proving
[program:cloudflared] has the shape the design calls for. Same technique as
test_supervisord_stop_order.py (regex section extraction on the real file,
not a real supervisord run — that needs a booted container, see
scripts/rc-boot-test-v3199.sh).

Checks:
  - the section exists, with a command= line pointing at cloudflared-run.sh;
  - autostart=true, autorestart=unexpected, exitcodes=0 (an exit 0 — no
    token — is expected and must not be restarted; anything else is not);
  - priority is strictly after openwebui's;
  - startsecs=0, exactly (proven necessary by an actual boot — see
    supervisord.conf's own comment and this file's own check below);
  - startretries gives real headroom for "the network may be briefly down
    at boot" without being unbounded;
  - stdout/stderr logs live under %(ENV_LOG_DIR)s with maxbytes/backups
    set, matching the convention every other program in this file uses;
  - stopwaitsecs is set (not left at supervisord's low default);
  - CRITICALLY: no `environment=` line in this section references
    CLOUDFLARE_TUNNEL_TOKEN via %(ENV_...)s — that would crash supervisord's
    config load the instant the var is unset (unlike the four
    autostart=%(ENV_x)s booleans, which entrypoint.sh guarantees are always
    concrete before supervisord starts, an operator secret has no safe
    default to normalise to, so this section must not reference it at all
    and instead rely on plain environment inheritance).

    python test_supervisord_cloudflared.py
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUPER = (ROOT / "supervisord.conf").read_text(encoding="utf-8", errors="replace")

FAILED: list[str] = []


def check(cond, label):
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}")
        FAILED.append(label)


def program_block(name: str) -> str:
    m = re.search(
        rf"\[program:{re.escape(name)}\](.*?)(?=\n\[program:|\n\[eventlistener:|\Z)",
        SUPER, re.S,
    )
    assert m is not None, f"[program:{name}] not found in supervisord.conf"
    return m.group(1)


def priority_of(name: str) -> int:
    block = program_block(name)
    m = re.search(r"^priority=(\d+)", block, re.M)
    assert m is not None, f"no priority= line in [program:{name}]"
    return int(m.group(1))


print("[cloudflared] section exists with a real command=")
block = program_block("cloudflared")
check(
    re.search(r"^command=.*cloudflared-run\.sh", block, re.M) is not None,
    "command= runs docker/cloudflared-run.sh (baked to /opt/cloudflared-run.sh)",
)

print()
print("[cloudflared] autostart / restart policy matches the 'EXITED means "
      "disabled' design")
check("autostart=true" in block, "autostart=true (the run script itself, "
      "not supervisord, decides enabled vs disabled)")
m = re.search(r"^autorestart=(\S+)", block, re.M)
check(m is not None and m.group(1) == "unexpected",
      f"autorestart=unexpected (got: {m.group(1) if m else None!r}) - an "
      f"exit 0 (disabled) is expected and is NOT restarted; anything else "
      f"IS restarted")
m = re.search(r"^exitcodes=(\S+)", block, re.M)
check(m is not None and m.group(1) == "0",
      f"exitcodes=0 (got: {m.group(1) if m else None!r}) - only a clean "
      f"exit is 'expected'")

print()
print("[cloudflared] never references CLOUDFLARE_TUNNEL_TOKEN via "
      "%(ENV_x)s in a LIVE config line (explanatory comments naming it are "
      "fine and expected) - that would crash supervisord's config load if unset")
_live_lines = "\n".join(
    line for line in block.splitlines()
    if line.strip() and not line.strip().startswith((";", "#"))
)
check(
    "%(ENV_CLOUDFLARE_TUNNEL_TOKEN)s" not in _live_lines,
    "no %(ENV_CLOUDFLARE_TUNNEL_TOKEN)s expansion in any non-comment line",
)

print()
print("[cloudflared] priority puts it after openwebui (the tunnel serves it)")
cf_priority = priority_of("cloudflared")
webui_priority = priority_of("openwebui")
check(
    cf_priority > webui_priority,
    f"cloudflared priority ({cf_priority}) is higher (= starts later, "
    f"shuts down earlier) than openwebui's ({webui_priority})",
)

print()
print("[cloudflared] startsecs=0 - MUST be 0, not merely small. Proven by "
      "an actual boot (rc-boot-test-v3199.sh): at startsecs=1 the disabled "
      "(exit 0) path went FATAL, not EXITED, because supervisord scores an "
      "exit against exitcodes/autorestart only AFTER startsecs is survived "
      "- any positive value re-opens that trap for a script whose disabled "
      "path exits near-instantly. See [program:zl-ops]/[program:selftest] "
      "for the same pattern already in this file.")
m = re.search(r"^startsecs=(\d+)", block, re.M)
check(m is not None, "startsecs= is set")
if m:
    startsecs = int(m.group(1))
    check(startsecs == 0,
          f"startsecs is exactly 0 (got {startsecs}) - the EXITED-when-"
          f"disabled path must not read as FATAL")

print()
print("[cloudflared] startretries gives headroom for a briefly-down "
      "network at boot, without being unbounded")
m = re.search(r"^startretries=(\d+)", block, re.M)
check(m is not None, "startretries= is set")
if m:
    startretries = int(m.group(1))
    check(startretries >= 3,
          f"startretries >= 3 (got {startretries}) - at least as much "
          f"headroom as vllm/compactor/openwebui's own startretries=3")

print()
print("[cloudflared] stopwaitsecs is set (not left at supervisord's low default)")
check(re.search(r"^stopwaitsecs=\d+", block, re.M) is not None,
      "stopwaitsecs= present")

print()
print("[cloudflared] logs go under %(ENV_LOG_DIR)s with maxbytes/backups, "
      "same convention as every other program in this file")
for stream in ("stdout", "stderr"):
    check(
        re.search(rf"^{stream}_logfile=%\(ENV_LOG_DIR\)s/cloudflared", block, re.M) is not None,
        f"{stream}_logfile is under %(ENV_LOG_DIR)s",
    )
    check(
        re.search(rf"^{stream}_logfile_maxbytes=\d+MB", block, re.M) is not None,
        f"{stream}_logfile_maxbytes is set",
    )
    check(
        re.search(rf"^{stream}_logfile_backups=\d+", block, re.M) is not None,
        f"{stream}_logfile_backups is set",
    )

print()
if FAILED:
    for f in FAILED:
        print("FAIL " + f)
    sys.exit(1)
print("All [program:cloudflared] supervisord config checks passed.")
