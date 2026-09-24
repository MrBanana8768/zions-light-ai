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
import shutil
import subprocess
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


print("[1] WEBUI_DB_LOCAL is normalised before the boot branch ever reads it")
# v3.1.9 (hostile pass #2, LOW). This used to assert the literal source text
# `WEBUI_DB_LOCAL="${WEBUI_DB_LOCAL:-true}"` and `[ "${WEBUI_DB_LOCAL}" =
# "true" ]` — i.e. it PINNED the strict byte comparison that WAS the finding
# (`True`/`TRUE`/`1`/`yes`/`on`/`" true"` all silently meant false, while
# every sibling flag in this subsystem folds case and trims whitespace).
# Asserting the exact string that is the bug in place is not a regression
# test for it; it is the opposite. Updated deliberately: this now runs the
# REAL entrypoint.sh block under `sh`, the same technique
# test_config_supervisord_bool.py uses for its four booleans, and checks the
# PROPERTY that matters — every spelling the subsystem's other flags accept
# resolves to the same true/false, unset/empty still defaults to true
# unchanged, and an unrecognised value is refused rather than guessed.
check('if [ "${WEBUI_DB_LOCAL}" = "true" ]; then' in ENTRY,
      "the boot block still branches on a strict, canonical comparison — "
      "correct AFTER normalisation, which is what makes a strict comparison "
      "safe here")


def extract_block(text: str, begin: str, end: str) -> str | None:
    # None, not a raised exception, when the markers are absent — the
    # REVERT of this whole fix (the mutation-testing baseline: "the code AS
    # IT SHIPPED" before this finding was fixed) has no such block at all,
    # and a test that can only fail by crashing has not demonstrated
    # anything about the property it claims to check (brief rule: "a
    # traceback ... is a wrong-reason red — fix the test").
    if begin not in text:
        return None
    i = text.index(begin)
    line_end = text.index("\n", i)
    j = text.index(end, line_end)
    return text[line_end + 1:j]


WDB_BLOCK = extract_block(
    ENTRY,
    "# BEGIN WEBUI_DB_LOCAL NORMALIZATION",
    "# END WEBUI_DB_LOCAL NORMALIZATION",
)
check(WDB_BLOCK is not None,
      "the WEBUI_DB_LOCAL normaliser block (BEGIN/END markers) is present "
      "in entrypoint.sh")


def _sh_quote(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


_MARK = "===WEBUI_DB_LOCAL_RESULT==="


def run_wdb_normalizer(raw_value: str | None) -> tuple[str, str, int]:
    """Run the REAL block under `sh` with WEBUI_DB_LOCAL=raw_value (unset
    entirely if raw_value is None — a distinct case from the empty string),
    and return (full stdout — the block's own banner/echo, if any; the
    exported value after the marker, '' if the script never reached it;
    exit code). `env -i` is not used here (unlike
    test_config_supervisord_bool.py's run_normalizer) because the block's
    own unset-vs-empty branch needs to observe a truly ABSENT variable,
    which a `VAR=''` assignment cannot represent.

    A MARKER, not "take the last line": unlike the supervisord `_bool`
    block (silent on success, WARNING to stderr only on a fallback), this
    block echoes a confirmation line to STDOUT on every path that does not
    exit — so the raw exported value and the block's own prose share one
    stream and a plain trailing-printf would run the two together."""
    setup = "" if raw_value is None else f"WEBUI_DB_LOCAL={_sh_quote(raw_value)}\n"
    script = f'{setup}{WDB_BLOCK}\nprintf "{_MARK}%s" "${{WEBUI_DB_LOCAL}}"\n'
    proc = subprocess.run(
        ["sh", "-c", script], capture_output=True, text=True, timeout=10,
    )
    if _MARK in proc.stdout:
        banner, _, value = proc.stdout.partition(_MARK)
    else:
        banner, value = proc.stdout, ""
    return banner, value, proc.returncode


if WDB_BLOCK is None:
    print("  FAIL cannot run the normaliser checks below — the block itself "
          "is missing (see the check just above)")
    FAILED.append("WEBUI_DB_LOCAL NORMALIZATION block is missing from entrypoint.sh")
elif shutil.which("sh") is None:
    print("  SKIP (no `sh` on PATH — needs the unit-tests container or any "
          "POSIX shell): the WEBUI_DB_LOCAL normaliser checks below did not "
          "run. The text checks in this file still did.")
else:
    print("    unset/empty keeps meaning true, unchanged")
    for raw, label in ((None, "unset"), ("", "empty string")):
        _banner, value, rc = run_wdb_normalizer(raw)
        check(rc == 0 and value == "true",
              f"WEBUI_DB_LOCAL {label}: resolves to true, exit 0 "
              f"(got {value!r}, rc={rc})")

    print("    explicit true/false are UNCHANGED — the one thing this fix "
          "must not touch")
    for raw in ("true", "false"):
        _banner, value, rc = run_wdb_normalizer(raw)
        check(rc == 0 and value == raw,
              f"WEBUI_DB_LOCAL={raw!r}: resolves to {raw!r} exactly, exit 0 "
              f"(got {value!r}, rc={rc})")

    print("    folded the same way this subsystem's other flags fold "
          "(trimmed, case-insensitive, 1/yes/on and 0/no/off)")
    FOLD_CASES = [
        ("True", "true"), ("TRUE", "true"), ("1", "true"),
        ("yes", "true"), ("YES", "true"), ("on", "true"),
        (" true", "true"), ("true ", "true"), ("true\n", "true"),
        ("False", "false"), ("0", "false"), ("no", "false"),
        ("off", "false"), (" false ", "false"),
    ]
    for raw, want in FOLD_CASES:
        _banner, value, rc = run_wdb_normalizer(raw)
        check(rc == 0 and value == want,
              f"WEBUI_DB_LOCAL={raw!r}: folds to {want!r} — the finding's own "
              f"table named this spelling as silently meaning false before "
              f"this fix (got {value!r}, rc={rc})")

    print("    an UNRECOGNISED, non-empty value refuses to boot rather than "
          "guessing which placement is safe")
    for raw in ("enabled", "2", "y", "t", "maybe", "TrueFalse"):
        banner, value, rc = run_wdb_normalizer(raw)
        check(rc == 1,
              f"WEBUI_DB_LOCAL={raw!r}: exits 1 (got rc={rc}, "
              f"stdout={banner[:60]!r})")
        check(value == "",
              f"WEBUI_DB_LOCAL={raw!r}: the script exits before ever "
              f"exporting a value — does NOT silently resolve to either "
              f"placement (got {value!r}) — a guess in either direction is "
              f"worse than refusing, per the comment beside the block")

    print("    CONTROL: the refusal names the raw value and both wrong "
          "guesses, so an operator reading the boot log knows what to fix")
    banner, _value, _rc = run_wdb_normalizer("enabled")
    check(
        "WEBUI_DB_LOCAL=[enabled]" in banner and "REFUSING" in banner
        and "false" in banner and "true" in banner,
        f"the banner quotes the raw value and explains both directions "
        f"(got {banner[:200]!r})",
    )

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
