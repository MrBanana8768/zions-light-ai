"""
Tier-3 validation of /admin/selftest (V2.1 Phase 6 Step 2).

The on-demand mirror of the supervisord boot one-shot. Localhost-only —
requires admin URL. The quick (?round_trip=false) mode also exercised
here since it's the cheap "is the deploy healthy right now?" path.
"""

import _harness as H


def test_selftest_no_round_trip_is_quick_pass():
    """The cheap path — five checks, no LLM call. Should be ~ms latency
    and always pass on a healthy deploy.
    """
    H.skip_if_no_admin("selftest endpoint is localhost-only")
    status, report = H.admin_selftest(round_trip=False)
    assert status == 200, (
        f"selftest returned HTTP {status}\n"
        f"report: {report!r}"
    )
    assert report.get("status") == "pass", (
        f"some checks failed:\n"
        + "\n".join(f"  {c['name']}: ok={c['ok']} detail={c['detail']!r}"
                    for c in report.get("checks", []))
    )
    assert report["summary"]["total"] == 5, (
        f"expected 5 checks without round-trip, got {report['summary']['total']}"
    )


def test_selftest_with_round_trip_includes_chat():
    """Full battery — six checks including a real LLM round-trip. Slower
    (LLM inference) but the only end-to-end "the model is actually
    responding" gate.
    """
    H.skip_if_no_admin("selftest endpoint is localhost-only")
    status, report = H.admin_selftest(round_trip=True)
    assert status == 200, (
        f"selftest returned HTTP {status}\n"
        f"report: {report!r}"
    )
    assert report.get("status") == "pass", (
        f"some checks failed:\n"
        + "\n".join(f"  {c['name']}: ok={c['ok']} detail={c['detail']!r}"
                    for c in report.get("checks", []))
    )
    names = [c["name"] for c in report["checks"]]
    assert "chat_round_trip" in names, f"chat check missing: {names}"


def test_selftest_report_has_all_required_check_names():
    """Locks in the documented check inventory — if a check gets renamed
    or removed, this test trips so the change is intentional.
    """
    H.skip_if_no_admin()
    _, report = H.admin_selftest(round_trip=True)
    names = {c["name"] for c in report["checks"]}
    expected = {
        "storage",
        "facts_round_trip",
        "vllm_models",
        "compactor_health",
        "admin_localhost",
        "chat_round_trip",
    }
    assert expected.issubset(names), (
        f"missing checks: {expected - names}; got: {names}"
    )


def test_selftest_each_check_has_required_fields():
    """Every check entry: name (str), ok (bool), latency_ms (number),
    detail (str). UI consumers depend on this shape."""
    H.skip_if_no_admin()
    _, report = H.admin_selftest(round_trip=False)
    for c in report["checks"]:
        assert isinstance(c.get("name"), str) and c["name"], f"bad name: {c!r}"
        assert isinstance(c.get("ok"), bool), f"ok not bool: {c!r}"
        assert isinstance(c.get("latency_ms"), (int, float)), f"latency not number: {c!r}"
        assert isinstance(c.get("detail"), str), f"detail not str: {c!r}"
