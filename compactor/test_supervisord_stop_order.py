"""compactor/test_supervisord_stop_order.py — D1 (findings.md, the
2026-09-23 WEBUI_DB_LOCAL rehearsal): webuidb-sync must stop AFTER
openwebui, and get enough time to publish before supervisord gives up.

Supervisord's own documented rule (verbatim, `[program:x]` -> `priority`):
"Lower priorities indicate programs that start first and shut down last."
webuidb-sync used to be priority 25, openwebui priority 20 - 25 > 20, so
webuidb-sync shut down BEFORE openwebui on every stop, the opposite of what
a final sync needs (nothing has been written yet for it to publish, or
worse, it races openwebui's own last write). The fix lowers webuidb-sync's
priority below openwebui's, and gives it stopwaitsecs=120 so its SIGTERM
handler's final sync_once(force=True) (webuidb.py, D1) has room to finish
before SIGKILL.

This is a STATIC check on the shipped config, not a real supervisord run
(that needs the rehearsal harness's containers - see the D1 section of
DB-MOVE-RUNBOOK.md / the report this fix's commit describes). It exists so
the ordering invariant cannot silently regress in a future edit to
supervisord.conf without a real rehearsal catching it.

    python test_supervisord_stop_order.py
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


def all_sections() -> dict[str, str]:
    """{'program:name' or 'eventlistener:name': block text} for every
    section in the file, found generically rather than by a hardcoded
    list — so a section added later is covered automatically."""
    out = {}
    for m in re.finditer(
        r"\[((?:program|eventlistener):[^\]]+)\](.*?)"
        r"(?=\n\[(?:program|eventlistener):|\Z)",
        SUPER, re.S,
    ):
        out[m.group(1)] = m.group(2)
    return out


print("[GENERAL] every [program:*]/[eventlistener:*] section has a command=")
# Caught for real, the hard way: an earlier version of the D1 edit below
# accidentally dropped `command=` (and `directory=`) from
# [program:webuidb-sync] while rewriting the surrounding comment block.
# supervisord refused to boot the WHOLE container on it ("does not specify
# a command in section 'program:webuidb-sync'") - found only by actually
# booting the real image under the rehearsal harness, not by any unit
# test that existed at the time. This is a general guard against that
# whole CLASS of edit, not just this one instance: every section, found
# generically rather than by name, must have exactly what supervisord's
# own config loader requires before it will start ANY of them.
_sections = all_sections()
check(len(_sections) >= 9, f"found every expected section "
      f"({len(_sections)}): {sorted(_sections)}")
for _name, _block in sorted(_sections.items()):
    check(
        re.search(r"^command=\S", _block, re.M) is not None,
        f"[{_name}] has a non-empty `command=` line",
    )

print()
print("[D1] CONTROL: webuidb-sync still has every key a [program:] section "
      "needs, unchanged by the priority/stopwaitsecs edit")
# Caught for real, the hard way: an earlier version of this same edit
# accidentally dropped `command=` and `directory=` while rewriting the
# surrounding comment block, and supervisord refused to boot the WHOLE
# container on it ("does not specify a command in section
# 'program:webuidb-sync'") - found only by actually booting the real image
# under the rehearsal harness, not by any unit test, because nothing here
# was asserting these two lines even exist. It is now.
_sync_block_for_keys = program_block("webuidb-sync")
for _key in ("command", "directory"):
    check(
        re.search(rf"^{_key}=", _sync_block_for_keys, re.M) is not None,
        f"[program:webuidb-sync] still has a `{_key}=` line",
    )

print()
print("[D1] webuidb-sync's priority puts it AFTER openwebui at shutdown")
sync_priority = priority_of("webuidb-sync")
webui_priority = priority_of("openwebui")
check(
    sync_priority < webui_priority,
    f"webuidb-sync priority ({sync_priority}) is LOWER than openwebui's "
    f"({webui_priority}) - per supervisord's own rule ('lower priorities "
    f"... shut down last'), webuidb-sync now stops AFTER openwebui",
)

compactor_priority = priority_of("compactor")
check(
    compactor_priority <= sync_priority,
    f"CONTROL: webuidb-sync (priority {sync_priority}) still starts no "
    f"earlier than compactor ({compactor_priority}) - lowering it below "
    f"openwebui must not accidentally race it ahead of dependencies it "
    f"never needed to race before",
)

print()
print("[D1] webuidb-sync gets enough stopwaitsecs for a real publish to finish")
sync_block = program_block("webuidb-sync")
m = re.search(r"^stopwaitsecs=(\d+)", sync_block, re.M)
check(m is not None, "stopwaitsecs= is set on [program:webuidb-sync] "
      "(supervisord's default is 10s, well under a warm publish's 13-23s "
      "or a cold one's ~71s)")
if m:
    stopwaitsecs = int(m.group(1))
    check(stopwaitsecs >= 120,
          f"and it is at least 120s (got {stopwaitsecs}) - comfortably "
          f"longer than the rehearsal's measured warm (13-23s) and cold "
          f"(~71s) publish times")

print()
print("[D1/S1] webuidb-sync is stopped with the DEFAULT stopsignal (TERM), "
      "not something webuidb.py's own handler cannot catch")
# review1-v3197-65ea196 mutant S1: an added `stopsignal=INT` line makes
# supervisord send SIGINT instead of SIGTERM on stop - webuidb.py's
# SIGTERM handler (D1) never fires, Python's own default SIGINT action
# (KeyboardInterrupt) unwinds the process with NO final sync, and nothing
# about this is visible from the D1 priority/stopwaitsecs checks above,
# which do not look at stopsignal at all.
_stopsignal_m = re.search(r"^stopsignal=(\S+)", sync_block, re.M)
check(
    _stopsignal_m is None or _stopsignal_m.group(1) == "TERM",
    f"[program:webuidb-sync] either has no stopsignal= override (supervisord's "
    f"default is TERM, which webuidb.py's sync_loop() installs a handler "
    f"for) or explicitly says TERM (got: "
    f"{_stopsignal_m.group(1) if _stopsignal_m else None!r})",
)

print()
print("    CONTROL: openwebui itself was not touched by this fix")
webui_block = program_block("openwebui")
check(
    "priority=20" in webui_block,
    "openwebui's own priority is unchanged (still 20) - this fix moves "
    "webuidb-sync relative to it, not the other way around",
)

print()
if FAILED:
    for f in FAILED:
        print("FAIL " + f)
    sys.exit(1)
print("All supervisord stop-order checks passed.")
