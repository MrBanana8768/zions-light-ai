"""Unit tests for ops/zlops/commands/status.py -- `zl status`'s exit-code
contract (0 ok / 3 warn / 1 crit) and, v3.1.9.9, the Cloudflare Tunnel
"configured but cloudflared is not RUNNING" cross-check that
`podenv.ONE_SHOT_OR_OPTIONAL` membership alone cannot express (an operator
must never have a real, configured tunnel outage read as "cloudflared is
just its normal one-shot-or-optional self").

Plain-script style like compactor/test_setup_sshd_script.py: exit 0 pass, 1
fail. No real supervisorctl/compactor/network -- `common.supervisorctl_status`
and `common.http_get_json` are both monkeypatched.

Run: python test_ops_status.py
"""

from __future__ import annotations

import io
import sys
# LOW (round 3): never write ops/**/__pycache__ from a test run --
# this file's own `from zlops import ...` below would otherwise create
# it even when the harness invoking this file forgot PYTHONDONTWRITEBYTECODE.
sys.dont_write_bytecode = True
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ops"))

from zlops import common  # noqa: E402
from zlops.commands import status  # noqa: E402


def assert_eq(actual, expected, label):
    if actual != expected:
        print(f"FAIL {label}: expected {expected!r}, got {actual!r}")
        sys.exit(1)
    print(f"  ok   {label}")


def assert_true(cond, label, extra=""):
    if not cond:
        print(f"FAIL {label}")
        if extra:
            print(f"  --- context ---\n{extra}\n  --- end ---")
        sys.exit(1)
    print(f"  ok   {label}")


_ALL_RUNNING = {
    "vllm": "RUNNING", "compactor": "RUNNING", "webuidb-sync": "STOPPED",
    "openwebui": "RUNNING", "stt": "RUNNING", "tts": "RUNNING",
    "selftest": "EXITED", "backup": "RUNNING", "zl-ops": "EXITED",
    "cloudflared": "EXITED",
}


def _run_status(status_map: dict, doc, argv: list[str] = ()) -> tuple[int, str]:
    buf = io.StringIO()
    with (
        mock.patch.object(common, "supervisorctl_status", return_value=dict(status_map)),
        mock.patch.object(common, "http_get_json", return_value=(200, doc, None)),
        mock.patch.object(sys, "stdout", buf),
    ):
        code = status.run(list(argv))
    return code, buf.getvalue()


def test_cloudflared_exited_with_no_tunnel_configured_is_fine():
    doc = {"status": "ok", "tunnel": {"configured": False, "ready": None}}
    code, out = _run_status(_ALL_RUNNING, doc)
    assert_eq(code, 0, "no token configured -> cloudflared EXITED is fine, exit 0")
    assert_true("tunnel configured but cloudflared" not in out, "no warning text printed", extra=out)


def test_cloudflared_exited_with_tunnel_configured_warns_and_exits_3():
    doc = {"status": "ok", "tunnel": {"configured": True, "ready": None}}
    code, out = _run_status(_ALL_RUNNING, doc)
    assert_eq(code, 3, "configured but cloudflared EXITED -> warn exit code (3), not crit (1)")
    assert_true("tunnel configured but cloudflared is EXITED" in out, "the warning names the actual state", extra=out)


def test_cloudflared_running_with_tunnel_configured_is_fine():
    status_map = dict(_ALL_RUNNING, cloudflared="RUNNING")
    doc = {"status": "ok", "tunnel": {"configured": True, "ready": True}}
    code, out = _run_status(status_map, doc)
    assert_eq(code, 0, "configured and RUNNING -> exit 0")
    assert_true("tunnel configured but cloudflared" not in out, "no warning text printed", extra=out)


def test_cloudflared_absent_tunnel_key_is_tolerated():
    """An older compactor (pre-v3.1.9.9) never sends a `tunnel` key at
    all -- must never be misread as "configured".
    """
    doc = {"status": "ok"}
    code, out = _run_status(_ALL_RUNNING, doc)
    assert_eq(code, 0, "no tunnel key at all -> cloudflared EXITED stays tolerated, exit 0")
    assert_true("tunnel configured but cloudflared" not in out, "no warning text printed", extra=out)


def test_cloudflared_warning_appears_in_brief_mode_too():
    doc = {"status": "ok", "tunnel": {"configured": True, "ready": None}}
    code, out = _run_status(_ALL_RUNNING, doc, argv=["--brief"])
    assert_eq(code, 3, "--brief mode still exits 3 on the configured-but-down tunnel")
    assert_true("tunnel configured but cloudflared is EXITED" in out,
                "--brief mode's one-line summary still surfaces the warning", extra=out)


def test_a_real_unexpected_service_still_exits_1_even_with_tunnel_warning():
    """crit/unexpected must still win over a mere tunnel warning -- a
    genuinely crashed vllm is not something the tunnel's own warn level
    should ever downgrade.
    """
    status_map = dict(_ALL_RUNNING, vllm="FATAL")
    doc = {"status": "ok", "tunnel": {"configured": True, "ready": None}}
    code, out = _run_status(status_map, doc)
    assert_eq(code, 1, "a genuinely unexpected service still exits 1 (crit-equivalent), tunnel warning notwithstanding")
    assert_true("<-- unexpected" in out, "vllm is still flagged unexpected", extra=out)


def test_mutation_service_table_ignoring_doc_would_hide_the_configured_but_down_case():
    """Prove the tests above bite: a version of `_service_table` that never
    looks at `doc` at all (the pre-v3.1.9.9 shape) would let a configured-
    but-down cloudflared through as ordinary ONE_SHOT_OR_OPTIONAL EXITED,
    with no warning and exit 0 -- exactly what an operator with a real,
    silently-broken tunnel must NOT see.
    """
    from zlops import podenv

    def _mutated_service_table(status_map, doc):
        lines = []
        any_unexpected = False
        width = max((len(s) for s in podenv.SERVICES), default=8)
        for name in podenv.SERVICES:
            state = status_map.get(name, "UNKNOWN")
            expected_running = name not in podenv.ONE_SHOT_OR_OPTIONAL
            unexpected = state != "RUNNING" and expected_running
            if unexpected:
                any_unexpected = True
            flag = " <-- unexpected" if unexpected else ""
            lines.append(f"  {name.ljust(width)}  {state}{flag}")
        return lines, any_unexpected, None, frozenset()  # never computes a tunnel warning -- the bug

    doc = {"status": "ok", "tunnel": {"configured": True, "ready": None}}
    original = status._service_table
    try:
        status._service_table = _mutated_service_table
        code, out = _run_status(_ALL_RUNNING, doc)
        assert_eq(code, 0, "mutated (doc-blind) service table exits 0 -- confirms the bite: a real outage goes silent")
        assert_true("tunnel configured but cloudflared" not in out, "mutated version prints no warning at all", extra=out)
    finally:
        status._service_table = original

    code, out = _run_status(_ALL_RUNNING, doc)
    assert_eq(code, 3, "the real (restored) service table warns and exits 3 again")
    assert_true("tunnel configured but cloudflared is EXITED" in out, "the real code's warning is back", extra=out)


# =============================================================================
# Round-11 review (LOW): a pre-v3.1.9.9 pod has no [program:cloudflared] in
# supervisord at all (state defaults to "UNKNOWN") and its compactor never
# sends a `tunnel` key (`tunnel_status_word(doc)` is None) -- there is
# nothing real to say about cloudflared there, so it must be hidden from
# the table AND excluded from the "N/10" brief count, not shown as
# "cloudflared UNKNOWN" noise on every single such pod.
# =============================================================================

_PRE_V3199 = dict(_ALL_RUNNING)
del _PRE_V3199["cloudflared"]  # supervisord simply doesn't know this program


def test_cloudflared_hidden_when_unknown_and_no_tunnel_key():
    doc = {"status": "ok"}  # no `tunnel` key at all -- pre-v3.1.9.9 compactor
    code, out = _run_status(_PRE_V3199, doc)
    assert_eq(code, 0, "nothing real to report -> exit 0")
    assert_true("cloudflared" not in out, "cloudflared is hidden from the table entirely", extra=out)


def test_cloudflared_hidden_from_brief_count_when_unknown_and_no_tunnel_key():
    doc = {"status": "ok"}
    code, out = _run_status(_PRE_V3199, doc, argv=["--brief"])
    assert_eq(code, 0, "exit 0")
    first_line = out.splitlines()[0]
    n_expected_total = len(podenv_SERVICES()) - 1  # cloudflared excluded from the denominator
    assert_true(f"/{n_expected_total} services RUNNING" in first_line,
                f"the N/{n_expected_total} count excludes the hidden cloudflared row", extra=out)
    assert_true("cloudflared" not in first_line, "no cloudflared mention in the brief summary at all", extra=out)


def podenv_SERVICES():
    from zlops import podenv

    return podenv.SERVICES


def test_cloudflared_warning_still_shown_when_configured_but_program_missing():
    """The one case the hide-rule must NOT swallow: `tunnel.configured` is
    true (a token IS set) but supervisord doesn't even have the program --
    that is a real fault (cloudflared should exist and doesn't), not the
    boring pre-v3.1.9.9 case, and must keep warning.
    """
    doc = {"status": "ok", "tunnel": {"configured": True, "ready": None}}
    code, out = _run_status(_PRE_V3199, doc)
    assert_eq(code, 3, "configured but the program isn't even present -> still warns, exit 3")
    assert_true("cloudflared" in out and "UNKNOWN" in out, "cloudflared UNKNOWN is shown, not hidden", extra=out)
    assert_true("tunnel configured but cloudflared is UNKNOWN" in out, "the warning names the actual state", extra=out)


def test_cloudflared_shown_normally_when_tunnel_key_present_but_disabled():
    """A v3.1.9.9+ pod with no token set DOES send a `tunnel` key
    (`configured: False`) -- that must show cloudflared EXITED as usual
    (ONE_SHOT_OR_OPTIONAL, tolerated), not get caught by the pre-v3.1.9.9
    hide-rule, which requires the `tunnel` key to be absent entirely.
    """
    doc = {"status": "ok", "tunnel": {"configured": False, "ready": None}}
    code, out = _run_status(_ALL_RUNNING, doc)
    assert_eq(code, 0, "exit 0")
    assert_true("cloudflared" in out and "EXITED" in out, "cloudflared EXITED is shown normally", extra=out)


def test_mutation_hiding_cloudflared_unconditionally_would_swallow_a_real_fault():
    """Prove the hide-rule's extra conditions (warn_text is None AND no
    tunnel key) actually matter: a version that hides cloudflared on
    state == "UNKNOWN" alone, with no cross-check, would swallow the
    configured-but-missing fault too.
    """
    from zlops import podenv, telemetry

    def _mutated_service_table(status_map, doc):
        lines = []
        any_unexpected = False
        tunnel_warn = None
        hidden = set()
        width = max((len(s) for s in podenv.SERVICES), default=8)
        for name in podenv.SERVICES:
            state = status_map.get(name, "UNKNOWN")
            if name == "cloudflared" and state == "UNKNOWN":  # the bug: no warn_text/tunnel-key cross-check
                hidden.add(name)
                continue
            warn_text = telemetry.cloudflared_fault(doc, state) if name == "cloudflared" else None
            expected_running = name not in podenv.ONE_SHOT_OR_OPTIONAL
            unexpected = state != "RUNNING" and expected_running
            if unexpected:
                any_unexpected = True
            if warn_text:
                tunnel_warn = warn_text
                flag = f" <-- warn: {warn_text}"
            elif unexpected:
                flag = " <-- unexpected"
            else:
                flag = ""
            lines.append(f"  {name.ljust(width)}  {state}{flag}")
        return lines, any_unexpected, tunnel_warn, frozenset(hidden)

    doc = {"status": "ok", "tunnel": {"configured": True, "ready": None}}
    original = status._service_table
    try:
        status._service_table = _mutated_service_table
        code, out = _run_status(_PRE_V3199, doc)
        assert_eq(code, 0, "mutated (unconditional hide) version exits 0 -- confirms the bite")
        assert_true("cloudflared" not in out, "mutated version hides the real fault too", extra=out)
    finally:
        status._service_table = original

    code, out = _run_status(_PRE_V3199, doc)
    assert_eq(code, 3, "the real (restored) code warns and exits 3 again")
    assert_true("tunnel configured but cloudflared is UNKNOWN" in out, "the real code's warning is back", extra=out)


# =============================================================================
# Runner
# =============================================================================

_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]

if __name__ == "__main__":
    for t in _TESTS:
        print(f"-- {t.__name__}")
        t()
    print(f"\nAll {len(_TESTS)} status tests passed.")
