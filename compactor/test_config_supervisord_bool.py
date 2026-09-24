"""compactor/test_config_supervisord_bool.py — the shell half of the
supervisord `autostart=%(ENV_x)s` boolean fix (v3.1.9 HIGH #4, hostile pass 2).

supervisord.conf gates five programs on `autostart=%(ENV_x)s`. supervisord's
own `boolean()` (supervisor/datatypes.py) accepts only
`{yes,true,on,1}` / `{no,false,off,0}`, case-insensitively, and does NOT
strip whitespace — anything else raises `ValueError` DURING CONFIG LOAD,
before any program starts. `exec supervisord` is this container's PID 1, so
that raise takes vLLM, OpenWebUI, the compactor, STT, TTS and the backup
daemon down together, from one typo'd RunPod template field — most
dangerously `true ` (a trailing space), which is exactly what a copy-paste
into a web form leaves behind.

entrypoint.sh now normalises every one of those variables (a `_bool` helper,
between the `# BEGIN/END SUPERVISORD BOOL NORMALIZATION` markers) before
`exec supervisord` ever reads them. This file proves two separate things:

  [A] the normaliser itself, run for real under `sh` (not bash — supervisord
      boots from a POSIX shell in the shipped image, and bash-only syntax in
      this block would silently not be what runs there) with every fatal
      spelling from the finding's own table, plus the ones supervisord
      itself still accepts.
  [B] that entrypoint.sh's block covers EVERY `autostart=%(ENV_x)s` boolean
      supervisord.conf actually declares — enumerated by reading
      supervisord.conf, not by trusting a hardcoded count of five, so this
      goes red the day a sixth one is added without a matching normalise
      line.

No server, no model, no network — but it DOES need a `sh` on PATH, which the
unit-tests container has and a bare Windows checkout does not:
    python test_config_supervisord_bool.py
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / "entrypoint.sh"
SUPERVISORD_CONF = ROOT / "supervisord.conf"

FAILED = []


def check(cond, label):
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}")
        FAILED.append(label)


def extract_block(text: str, begin: str, end: str) -> str:
    # Skip to the END of the BEGIN marker's own line: the marker comment
    # continues past the literal marker text on that same source line (more
    # comment prose after it), and slicing right after the marker string
    # would hand the shell a line fragment with no leading "#" - dash reads
    # that as code, not a comment, and the whole block fails to parse.
    i = text.index(begin)
    line_end = text.index("\n", i)
    j = text.index(end, line_end)
    return text[line_end + 1:j]


ENTRYPOINT_SRC = ENTRYPOINT.read_text(encoding="utf-8")
NORM_BLOCK = extract_block(
    ENTRYPOINT_SRC,
    "# BEGIN SUPERVISORD BOOL NORMALIZATION",
    "# END SUPERVISORD BOOL NORMALIZATION",
)


def run_normalizer(var: str, raw_value: str) -> tuple[str, int]:
    """Run the REAL extracted block under `sh`, with `var` set to
    `raw_value`, and return (exported value, exit code). Uses `env -i` plus
    an explicit PATH so this is not accidentally exercising the caller's
    shell environment — only what the block itself sets."""
    script = (
        f"{var}={_sh_quote(raw_value)}\n"
        f"{NORM_BLOCK}\n"
        f'eval "printf \'%s\' \\"\\${var}\\""\n'
    )
    proc = subprocess.run(
        ["sh", "-c", script],
        capture_output=True, text=True, timeout=10,
    )
    return proc.stdout, proc.returncode


def _sh_quote(s: str) -> str:
    # Single-quote, doubling embedded single quotes the POSIX way. Good
    # enough for the spellings this file feeds it (no embedded quotes).
    return "'" + s.replace("'", "'\\''") + "'"


if shutil.which("sh") is None:
    print("SKIP: no `sh` on PATH — this file needs the unit-tests container "
          "(or any POSIX shell) to exercise the real entrypoint.sh block.")
    sys.exit(3)  # 3 = skipped, per this project's runner convention — never a pass


# ---------------------------------------------------------------------------
# [A] The normaliser, run for real, against the finding's own table plus the
# CONTROL that a real "true" still starts the service.
# ---------------------------------------------------------------------------

print("[A] _bool() over the finding's fatal-spelling table")
print()

# (raw value,      expected normalized output,  label)
CASES = [
    ("true",        "true",  "CONTROL: the ordinary value still says yes"),
    ("false",       "false", "CONTROL: and the ordinary negative still says no"),
    ("True",        "true",  "case-insensitive"),
    ("TRUE",        "true",  None),
    ("1",           "true",  None),
    ("yes",         "true",  None),
    ("on",          "true",  None),
    ("0",           "false", None),
    ("off",         "false", None),
    ("no",          "false", None),
    # Whitespace cases use "false"-family values against STT_ENABLED, whose
    # SAFE DEFAULT is "true" — so if whitespace-stripping breaks, the value
    # falls through to the default and reads "true" instead of "false",
    # which is a DIFFERENT answer from the correct one and therefore an
    # assertion that can actually fail. A "true "-family value here would
    # be indistinguishable from a broken normalizer, because both the
    # correct match AND the broken fallback produce "true" for THIS
    # variable — exactly the "check that cannot fire" shape the project's
    # own playbook warns about, and it shipped once before it was caught.
    ("false ",      "false", "F: trailing space — the RunPod-template-field case"),
    (" false",      "false", "F: leading space"),
    ("false\n",     "false", "F: trailing newline"),
    ("enabled",     None,    "F: unrecognised word — falls to the safe default, not raises"),
    ("",             None,   "F: empty value — falls to the safe default"),
    ("2",            None,   "F: unrecognised digit"),
    ("y",            None,   "F: not in supervisord's own vocabulary either"),
    ("t",             None,  "F: same"),
]

for raw, want, tag in CASES:
    label = f"STT_ENABLED={raw!r}"
    if tag:
        label += f"  [{tag}]"
    out, rc = run_normalizer("STT_ENABLED", raw)
    check(rc == 0, f"{label}: the block itself does not exit non-zero (rc={rc})")
    expect = want if want is not None else "true"  # STT_ENABLED's safe default is true
    check(out == expect, f"{label}: normalizes to {expect!r} (got {out!r})")

# Each of the four live variables gets its OWN safe default — verified
# explicitly rather than assumed from STT_ENABLED's table above.
print()
print("[A2] each variable's SAFE default, on an unrecognised value")
SAFE_DEFAULTS = {
    "STT_ENABLED": "true",
    "TTS_ENABLED": "true",
    "COMPACTOR_SELFTEST_ON_BOOT": "true",
    "COMPACTOR_BACKUP_ENABLED": "true",
    # WEBUIDB_SYNC_ENABLED is deliberately NOT here - see [B] below.
}
for var, safe in SAFE_DEFAULTS.items():
    out, rc = run_normalizer(var, "enabled")  # a value in neither vocabulary
    check(rc == 0, f"{var}=enabled: the block does not exit non-zero (rc={rc})")
    check(out == safe, f"{var}=enabled falls to its documented safe default "
                        f"{safe!r} (got {out!r})")
    # CONTROL: the same variable still honours an explicit override in
    # BOTH directions, so this is not "the safe default no matter what".
    other = "false" if safe == "true" else "true"
    out2, _ = run_normalizer(var, other)
    check(out2 == other,
          f"{var}={other} is honoured explicitly, not overridden by the "
          f"safe default (got {out2!r})")


# ---------------------------------------------------------------------------
# [B] Coverage: every autostart=%(ENV_x)s in supervisord.conf has EITHER a
# matching `_bool` normalise line in entrypoint.sh, OR is on the short,
# explicit, documented exemption list (currently just WEBUIDB_SYNC_ENABLED -
# see entrypoint.sh's own comment beside it: both branches of the
# WEBUI_DB_LOCAL if/else set it themselves to an exact "true"/"false"
# literal, never from a raw operator-facing env var, and
# test_webuidb_gate.py [6] already pins it being exported on both paths).
# Enumerated from supervisord.conf itself, not a hardcoded count — this is
# what goes red the day a sixth boolean is added without EITHER a `_bool`
# line or a deliberate addition to _ALREADY_SAFE below.
# ---------------------------------------------------------------------------

print()
print("[B] every supervisord.conf autostart env-boolean is normalized or exempt")
print()

_ALREADY_SAFE = {"WEBUIDB_SYNC_ENABLED"}

conf_src = SUPERVISORD_CONF.read_text(encoding="utf-8")
declared = sorted(set(re.findall(r"autostart=%\(ENV_([A-Za-z0-9_]+)\)s", conf_src)))
check(len(declared) >= 4,
      f"sanity: supervisord.conf declares at least the four live "
      f"autostart env-booleans this fix targets (found {declared})")

for name in declared:
    if name in _ALREADY_SAFE:
        check(f"_bool {name} " not in NORM_BLOCK,
              f"{name}: on the documented exemption list AND not routed "
              f"through _bool — if this ever changes, take it off the list "
              f"instead of letting both be true")
        continue
    check(f"_bool {name} " in NORM_BLOCK,
          f"F: supervisord.conf's ENV_{name} autostart boolean has a matching "
          f"`_bool {name} ...` normalise line in entrypoint.sh (declared: {declared})")

# And the reverse spot-check: the four variables the finding named plus the
# one exemption are exactly what supervisord.conf declares (a sixth one
# appearing here without showing up above would mean the regex stopped
# matching, not that there is nothing to normalize).
check(set(declared) == {"STT_ENABLED", "TTS_ENABLED", "COMPACTOR_SELFTEST_ON_BOOT",
                         "COMPACTOR_BACKUP_ENABLED", "WEBUIDB_SYNC_ENABLED"},
      f"declared set matches the finding's inventory exactly (got {declared})")


# ---------------------------------------------------------------------------
print()
print("[C] MEDIUM (hostile2-config): STT_ENABLED/TTS_ENABLED=1 used to start "
      "the service while dropping its probe from the boot self-test")
print()
# The break, as filed: supervisord's own boolean() accepts {yes,true,on,1}
# case-insensitively, so STT_ENABLED=1 (or yes, or on) starts `stt` - but
# compactor/selftest.py read STT_ENABLED with a plain
# `.strip().lower() == "true"`, which "1"/"yes"/"on" all fail. So the
# service started, unprobed, and the self-test's own summary total is
# `len(checks)` - a denominator that shrinks with the check list, so
# "N/N passed" never signals the missing check either.
#
# ALREADY FIXED, as a side effect of [A]/[B] above, not by a change to
# selftest.py (which is not this lane's file): entrypoint.sh's `_bool`
# normalizes STT_ENABLED/TTS_ENABLED to a canonical "true"/"false" BEFORE
# `exec supervisord`, and every child supervisord starts - selftest
# included; see supervisord.conf's `environment=` under [program:selftest],
# which sets COMPACTOR_URL/VLLM_URL/MODEL_REPO but does NOT override
# STT_ENABLED or TTS_ENABLED, so those two are inherited from the
# already-normalized parent environment. Both consumers therefore see the
# SAME canonical string, never the raw one - proven end-to-end below: the
# REAL entrypoint.sh normalizer, then selftest.py's OWN comparison line
# (extracted from its source, not reimplemented, so a change to that line
# is caught here too), fed the normalized value.
SELFTEST = ROOT / "compactor" / "selftest.py"
SELFTEST_SRC = SELFTEST.read_text(encoding="utf-8")


def extract_selftest_bool_expr(var: str) -> str | None:
    m = re.search(
        rf'{var}\s*=\s*os\.environ\.get\("{var}",\s*"false"\)\.strip\(\)\.lower\(\)\s*==\s*"true"',
        SELFTEST_SRC,
    )
    check(
        m is not None,
        f"selftest.py still reads {var} as "
        f'os.environ.get("{var}", "false").strip().lower() == "true" '
        f"(if this line changed, re-verify this check by hand rather than "
        f"trusting it)",
    )
    return m.group(0) if m else None


for _var in ("STT_ENABLED", "TTS_ENABLED"):
    _expr = extract_selftest_bool_expr(_var)
    if _expr is None:
        continue
    # The finding's own unprobed spellings ([A5-3]'s table): supervisord
    # starts the service on every one of these, and selftest's RAW
    # `== "true"` read would not - NOT "TRUE" or "true " (with whitespace),
    # which selftest's own `.strip().lower()` already handled before this
    # fix existed; the CONTROL below would be vacuously true for those and
    # prove nothing about the mismatch this section is about.
    for _raw in ("1", "yes", "on"):
        _normalized, _rc = run_normalizer(_var, _raw)
        check(_rc == 0 and _normalized == "true",
              f"{_var}={_raw!r}: entrypoint.sh normalizes to true "
              f"(got {_normalized!r}, rc={_rc})")
        # Execute selftest.py's OWN line against the NORMALIZED value - what
        # selftest.py actually sees once supervisord has started it, because
        # entrypoint.sh exported the normalized string before `exec
        # supervisord`, never the raw one.
        _ns = {"os": os}
        os.environ[_var] = _normalized
        try:
            exec(compile(_expr, "<selftest.py extract>", "exec"), _ns)
        finally:
            del os.environ[_var]
        check(_ns[_var] is True,
              f"{_var}={_raw!r} -> normalized {_normalized!r} -> selftest.py's "
              f"own comparison now reads True (the probe runs): got "
              f"{_ns[_var]!r}")

        # CONTROL, and the point of the whole check: the SAME raw value fed
        # DIRECTLY to selftest's line (bypassing entrypoint.sh) still reads
        # False - proving the fix is the normalization step sitting in
        # front of both consumers, not some change to selftest.py's own
        # comparison.
        _ns2 = {"os": os}
        os.environ[_var] = _raw
        try:
            exec(compile(_expr, "<selftest.py extract>", "exec"), _ns2)
        finally:
            del os.environ[_var]
        check(_ns2[_var] is False,
              f"CONTROL: {_var}={_raw!r} fed directly to selftest.py's own "
              f"comparison, bypassing entrypoint.sh, still reads False - "
              f"confirming the mismatch the finding described exists at "
              f"that line, and that the fix is normalizing BEFORE that "
              f"line runs, not changing the line itself")


if FAILED:
    print()
    print(f"{len(FAILED)} assertion(s) failed:")
    for label in FAILED:
        print(f"  - {label}")
    sys.exit(1)
print()
print("All supervisord bool-normalization tests passed.")
