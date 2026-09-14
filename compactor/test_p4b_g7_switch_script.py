"""
p4-b (hostile pass #4, reviewer B) G7 — scripts/switch-webui-db-to-local.py
on a v3.1.6+ image left OpenWebUI on local disk with webuidb-sync never
actually publishing (it inherits WEBUI_DB_LOCAL=false and F8's placement
gate refuses it, even after a manual `supervisorctl start`), yet printed
"SWITCHED ... /data now receives snapshots".

Fix: the script now refuses outright when the image's entrypoint.sh
normalizes WEBUI_DB_LOCAL (the "BEGIN WEBUI_DB_LOCAL NORMALIZATION" marker),
pointing the operator at the flag instead.

Run inside the compactor image or any container with the requirements
installed:
    python test_p4b_g7_switch_script.py
"""

import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = Path(tempfile.mkdtemp(prefix="zions-p4b-g7-test-"))


def assert_eq(a, b, label):
    if a != b:
        print(f"FAIL {label}: expected {b!r}, got {a!r}")
        sys.exit(1)
    print(f"  ok   {label}")


def assert_true(c, label):
    if not c:
        print(f"FAIL {label}")
        sys.exit(1)
    print(f"  ok   {label}")


def _real_entrypoint_sh_text():
    p = _HERE.parent / "entrypoint.sh"
    assert_true(p.is_file(), f"fixture: {p} exists")
    text = p.read_text(encoding="utf-8")
    assert_true("BEGIN WEBUI_DB_LOCAL NORMALIZATION" in text,
                "fixture: the real entrypoint.sh really does carry the marker this fix reads")
    return text


def _run_switch_script(env_extra: dict) -> subprocess.CompletedProcess:
    """Run the real script as a subprocess (not imported) so its argparse /
    __main__ / exit-code path is exercised exactly as an operator's shell
    invocation would see it."""
    script = _HERE.parent / "scripts" / "switch-webui-db-to-local.py"
    env = dict(os.environ, **env_extra)
    return subprocess.run(
        [sys.executable, str(script), "--check"], env=env,
        capture_output=True, text=True, timeout=30,
    )


def test_g7_refuses_on_an_image_whose_entrypoint_normalizes_the_flag():
    print("\n[test] p4-b G7: the hot-patch script refuses when entrypoint.sh normalizes WEBUI_DB_LOCAL")
    ep = _ROOT / "entrypoint.sh"
    ep.write_bytes(_real_entrypoint_sh_text().encode("utf-8"))
    conf = _ROOT / "supervisord.conf"
    conf.write_text("[program:openwebui]\nenvironment=X=1\n")
    # A real webuidb.py stub, so the ORIGINAL "module is missing" precondition
    # (a few lines below this fix's own check, also exit 2) cannot pass for
    # the wrong reason if this fix's own check is ever disabled — the two
    # refusals share an exit code, so the TEXT is the only reliable signal.
    compactor_dir = _ROOT / "opt-compactor-g7a"
    compactor_dir.mkdir()
    (compactor_dir / "webuidb.py").write_text("# stub\n")

    r = _run_switch_script({
        "ENTRYPOINT_SH": str(ep), "SUPERVISOR_CONF": str(conf),
        "COMPACTOR_DIR": str(compactor_dir),
    })
    assert_true("WEBUI_DB_LOCAL=true in the template" in r.stdout,
                f"G7 fix: names the actual recovery (got rc={r.returncode}, stdout={r.stdout!r})")
    assert_true("FAIL" in r.stdout, "and says FAIL, not SWITCHED")
    assert_eq(r.returncode, 2, f"and exits non-zero (got rc={r.returncode})")


def test_g7_control_proceeds_on_an_image_with_no_normalization_marker():
    print("\n[test] p4-b G7 CONTROL: an image with no WEBUI_DB_LOCAL normalization block still runs the hot-patch form")
    ep = _ROOT / "entrypoint-old.sh"
    ep.write_text("#!/bin/bash\necho legacy entrypoint, no normalization here\n")
    conf = _ROOT / "supervisord-2.conf"
    conf.write_text("[program:openwebui]\nenvironment=X=1\n")
    compactor_dir = _ROOT / "opt-compactor"
    compactor_dir.mkdir()
    (compactor_dir / "webuidb.py").write_text("# stub so the script's own precondition passes\n")

    r = _run_switch_script({
        "ENTRYPOINT_SH": str(ep), "SUPERVISOR_CONF": str(conf),
        "COMPACTOR_DIR": str(compactor_dir),
    })
    # It will fail LATER (importing the stub webuidb.py has none of the real
    # module's functions), but it must get PAST the G7 precondition check —
    # this CONTROL proves the refusal is scoped to the marker, not blanket.
    assert_true("set WEBUI_DB_LOCAL=true in the template" not in r.stdout,
                f"CONTROL: no normalization marker, so G7's refusal text is absent (got {r.stdout!r})")


def test_g7_control_missing_entrypoint_sh_does_not_block_either():
    print("\n[test] p4-b G7 CONTROL: no entrypoint.sh at all (a stripped-down container) does not block the script")
    missing = _ROOT / "does-not-exist.sh"
    conf = _ROOT / "supervisord-3.conf"
    conf.write_text("[program:openwebui]\nenvironment=X=1\n")
    compactor_dir = _ROOT / "opt-compactor-2"
    compactor_dir.mkdir()
    (compactor_dir / "webuidb.py").write_text("# stub\n")

    r = _run_switch_script({
        "ENTRYPOINT_SH": str(missing), "SUPERVISOR_CONF": str(conf),
        "COMPACTOR_DIR": str(compactor_dir),
    })
    assert_true("set WEBUI_DB_LOCAL=true in the template" not in r.stdout,
                f"CONTROL: a missing entrypoint.sh is not treated as 'normalized' (got {r.stdout!r})")


if __name__ == "__main__":
    tests = [
        test_g7_refuses_on_an_image_whose_entrypoint_normalizes_the_flag,
        test_g7_control_proceeds_on_an_image_with_no_normalization_marker,
        test_g7_control_missing_entrypoint_sh_does_not_block_either,
    ]
    for t in tests:
        t()
    print("\nAll p4-b G7 (switch-webui-db-to-local.py) tests passed.")
