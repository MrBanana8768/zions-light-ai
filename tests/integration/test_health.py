"""
Tier-3 validation of /health/full (V2.1 Phase 6).

The deep health probe. Replaces the Dockerfile HEALTHCHECK target so
the container goes unhealthy when vLLM is down (the old `curl :3000`
check stayed green even when vLLM was FATAL).

Public endpoint — no admin URL required.
"""

import _harness as H


def test_health_full_returns_structured_report():
    """Hits /health/full and validates the documented schema."""
    status, body = H.health_full()
    # On a healthy deploy: status=ok, HTTP 200. On an unhealthy deploy
    # we still want a structured body — so we don't bail on 503 here,
    # we just check the shape, then assert ok separately.
    assert status in (200, 503), f"unexpected HTTP {status}: {body!r}"
    assert isinstance(body, dict) and body, f"empty/non-dict body: {body!r}"
    for top in ("status", "checks", "stats", "config"):
        assert top in body, f"missing top-level key {top!r}: {body!r}"
    assert body["status"] in ("ok", "degraded", "down"), body["status"]
    for sub in ("vllm", "storage"):
        assert sub in body["checks"], f"missing checks.{sub}"
        assert "ok" in body["checks"][sub], f"missing checks.{sub}.ok"
    for stat in ("conversations", "facts_total", "indexed_exchanges_total"):
        assert stat in body["stats"], f"missing stats.{stat}"


def test_health_full_is_ok_on_healthy_deploy():
    """On a working pod, /health/full should be HTTP 200 + status='ok'.

    If this fails, the deploy itself is unhealthy — not the test. Useful
    canary at the start of any Tier-3 run.
    """
    status, body = H.health_full()
    assert status == 200, (
        f"/health/full returned HTTP {status} — deploy is unhealthy.\n"
        f"body: {body!r}"
    )
    assert body["status"] == "ok", (
        f"status={body['status']!r}, not 'ok'.\n"
        f"vllm: {body['checks']['vllm']!r}\n"
        f"storage: {body['checks']['storage']!r}"
    )


def test_health_full_lists_at_least_one_model():
    """vLLM probe should report at least one loaded model."""
    _, body = H.health_full()
    models = body.get("checks", {}).get("vllm", {}).get("models", [])
    assert len(models) >= 1, f"no models reported by vllm probe: {body!r}"


def test_health_full_storage_writable():
    """Storage probe must be ok=True — if not, no V2 features work at all."""
    _, body = H.health_full()
    storage = body.get("checks", {}).get("storage", {})
    assert storage.get("ok") is True, f"storage probe failed: {storage!r}"
    assert storage.get("writable") is True, f"storage not writable: {storage!r}"
