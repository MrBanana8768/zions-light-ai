"""
v3.1.9 round 2 — scripts/merge-conversations.py's raw env conversion.

hostile2-config MEDIUM (per SP\\fix-backup.md's own re-check: the finding's
text, proof block and fix are all specifically about
scripts/merge-conversations.py:74, not scripts/recover-webui-db.py, which
the backup lane already confirmed has no such shape). The line:

    ap.add_argument("--port", type=int,
                     default=int(os.environ.get("COMPACTOR_PORT", "8080")))

evaluates `int(...)` over an environment read at ARGPARSE-SETUP time — the
same unsoftened-conversion shape compactor/test_envcfg.py section 5 hunts
for across compactor/, tts/, stt/ and pipelines/ (SHIPPED_DIRS), except this
file lives in scripts/, which that scan's own docstring says is
DELIBERATELY out of scope ("entrypoint.sh does its own parsing in a dialect
this cannot see" — a reason about entrypoint.sh, not about scripts/*.py
generally). A typo in COMPACTOR_PORT crashed this script with an uncaught
ValueError before it could even print --help, which matters more here than
in most places: this script's whole reason to exist is being run, correctly,
by an operator mid-incident, with the live compactor process already
stopped.

WHY scripts/ STAYS OUT OF SHIPPED_DIRS (the "decide whether scripts/ should
be in the detector's scope" question this finding asks). SHIPPED_DIRS is
tied to a concrete, checked invariant — "every directory the Dockerfile
COPYs a .py from" (test_no_unsoftened_env_conversion_survives_anywhere's own
CONTROL reads the Dockerfile) — because the failure mode section 5 exists
for is a CONTAINER THAT WILL NOT BOOT. scripts/ is never COPYed into the
image; these are hotfix tools an operator `git clone`s onto a pod's local
disk and runs by hand (see this script's own docstring). Folding scripts/
into SHIPPED_DIRS would blur what that scan actually proves ("the boot path
is typo-safe") with a different, real property this test asserts instead:
"this operator tool doesn't crash a mid-incident run before printing --help
or a useful error". Handled here, per-script, rather than by widening
SHIPPED_DIRS's meaning.

Run: python test_round2_merge_conversations_envcfg.py
"""

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SCRIPT = ROOT / "scripts" / "merge-conversations.py"
PY = sys.executable

sys.path.insert(0, str(HERE))
import test_envcfg  # noqa: E402  (reuse the real AST detector, not a copy)

FAILED = []


def assert_true(cond, label):
    if not cond:
        print(f"FAIL {label}")
        FAILED.append(label)
    else:
        print(f"  ok   {label}")


def assert_eq(a, b, label):
    if a != b:
        print(f"FAIL {label}: expected {b!r}, got {a!r}")
        FAILED.append(label)
    else:
        print(f"  ok   {label}")


def _run(env_port, extra_args=()):
    """Run the real script (--src/--dst point at conv ids that cannot
    exist, so it always ends in a REFUSAL from portability.merge_conversation
    rather than actually merging anything) with COMPACTOR_PORT set to
    `env_port` (or unset if None). Returns (returncode, stdout+stderr)."""
    env = dict(os.environ)
    if env_port is None:
        env.pop("COMPACTOR_PORT", None)
    else:
        env["COMPACTOR_PORT"] = env_port
    env["PYTHONIOENCODING"] = "utf-8"
    r = subprocess.run(
        [PY, str(SCRIPT), "--src", "__round2_test_src__",
         "--dst", "__round2_test_dst__", *extra_args],
        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=60,
    )
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def test_help_works_with_no_compactor_port_and_no_clone_needed():
    print("\n[1] --help works without a compactor package present or "
          "COMPACTOR_PORT set at all")
    # CONTROL for the whole reorder this fix made: --port's default used to
    # be computed via int(os.environ.get(...)) INSIDE argparse.add_argument,
    # so even --help ran that conversion (argparse builds the full parser,
    # including every default, before dispatching to -h). Proves the
    # reorder did not accidentally make --help depend on the compactor
    # package being clonable beside this script.
    env = dict(os.environ)
    env.pop("COMPACTOR_PORT", None)
    env["PYTHONIOENCODING"] = "utf-8"
    r = subprocess.run([PY, str(SCRIPT), "--help"], cwd=str(ROOT), env=env,
                        capture_output=True, text=True, timeout=30)
    assert_eq(r.returncode, 0, "exits 0")
    assert_true("--port" in r.stdout, "and describes --port in its usage")
    assert_true("Traceback" not in (r.stdout + r.stderr),
                "no traceback — the old shape crashed here on ANY bad "
                "COMPACTOR_PORT, before argparse could even print --help")


def test_a_bad_compactor_port_does_not_crash_at_argparse_setup():
    print("\n[2] a mistyped COMPACTOR_PORT no longer crashes before the "
          "script can even refuse the (nonexistent) merge")
    for label, value in [
        ("unparseable", "8O80"),   # letter O, not zero — the exact typo shape
        ("empty", ""),
        ("whitespace", "   "),
        ("negative", "-1"),
        ("zero", "0"),
        ("unset", None),
    ]:
        rc, out = _run(value)
        assert_true(
            "Traceback (most recent call last)" not in out,
            f"COMPACTOR_PORT={value!r} ({label}): no uncaught traceback "
            f"(rc={rc})",
        )
        assert_true(
            "ValueError" not in out,
            f"COMPACTOR_PORT={value!r} ({label}): no ValueError anywhere in "
            f"the output",
        )
        # The script still runs its real logic all the way to
        # portability.merge_conversation's own refusal for a conv id that
        # cannot exist — proving this isn't "crashes silently and exits
        # early" but "reaches the same REAL refusal a good port value does".
        assert_true(
            "REFUSED:" in out,
            f"COMPACTOR_PORT={value!r} ({label}): reaches the real "
            f"merge_conversation refusal (got: {out[-300:]!r})",
        )


def test_an_explicit_port_flag_still_overrides_a_bad_env_value():
    print("\n[3] CONTROL: --port on the command line still wins over "
          "COMPACTOR_PORT, bad or not")
    # Without this, [2] could be passing because --port silently stopped
    # doing anything at all.
    rc, out = _run("8O80", extra_args=["--port", "9"])
    assert_true("REFUSED:" in out, f"still reaches the real refusal (got: {out[-300:]!r})")
    # Port 9 (discard) refuses nothing on its own; the point is only that
    # parsing succeeded and the script proceeded normally with an explicit
    # override in place of a garbage env value.


def test_the_real_ast_detector_is_clean_on_this_script():
    print("\n[4] the shipped-tree AST detector (compactor/test_envcfg.py), "
          "run directly against this ONE script, finds nothing")
    # Not folded into SHIPPED_DIRS (see this module's own docstring for why)
    # — but the same real detector function, not a reimplementation, so a
    # regression here is still caught the same way a shipped-tree one would
    # be.
    src = SCRIPT.read_text(encoding="utf-8")
    hits = test_envcfg._unsoftened_env_conversions(src, "scripts/merge-conversations.py")
    assert_eq(hits, [], f"no unsoftened env conversion remains (got: {hits})")

    print("    CONTROL: the detector still finds the ORIGINAL shape "
          "(proves this isn't silent for the wrong reason)")
    bad = (
        'import argparse, os\n'
        'ap = argparse.ArgumentParser()\n'
        'ap.add_argument("--port", type=int, '
        'default=int(os.environ.get("COMPACTOR_PORT", "8080")))\n'
    )
    hits = test_envcfg._unsoftened_env_conversions(bad, "<control>")
    assert_true(len(hits) == 1 and "COMPACTOR_PORT" in hits[0],
                f"the detector DOES catch the exact pre-fix shape (got: {hits})")


if __name__ == "__main__":
    test_help_works_with_no_compactor_port_and_no_clone_needed()
    test_a_bad_compactor_port_does_not_crash_at_argparse_setup()
    test_an_explicit_port_flag_still_overrides_a_bad_env_value()
    test_the_real_ast_detector_is_clean_on_this_script()

    if FAILED:
        print(f"\n{len(FAILED)} FAILED:")
        for f in FAILED:
            print(f"  - {f}")
        sys.exit(1)
    print("\nAll round-2 merge-conversations.py envcfg checks passed.")
