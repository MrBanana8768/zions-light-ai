"""
pytest configuration for the Tier-3 integration suite.

Provides a session-scoped fixture that verifies the deployment is
reachable BEFORE any individual test runs — if the compactor isn't up,
fail loudly once instead of producing N confusing per-test failures.
"""

from __future__ import annotations

import pytest

import _harness as H


def pytest_addoption(parser):
    """Allow --base-url / --admin-url on the command line as an
    alternative to the ZIONS_TEST_* environment variables. Env vars
    still take precedence when both are set."""
    parser.addoption(
        "--base-url",
        action="store",
        default=None,
        help="Compactor public URL (overrides ZIONS_TEST_BASE_URL)",
    )
    parser.addoption(
        "--admin-url",
        action="store",
        default=None,
        help="Admin endpoint URL (enables admin-requiring tests)",
    )


def pytest_configure(config):
    """Apply CLI overrides into the harness module before tests run.
    Env vars win if already set — CLI is a convenience for one-off runs."""
    base = config.getoption("--base-url")
    admin = config.getoption("--admin-url")
    if base and not H.BASE_URL:
        H.BASE_URL = base.rstrip("/")
    if admin and not H.ADMIN_ENABLED:
        H.ADMIN_URL = admin.rstrip("/")
        H.ADMIN_ENABLED = True


@pytest.fixture(scope="session", autouse=True)
def _deployment_reachable():
    """Session-scope sanity check: fail fast with one clear error if the
    compactor isn't reachable, instead of N per-test failures with
    cryptic connection errors."""
    H.require_base_url()
    if not H.health_ok():
        pytest.exit(
            f"Compactor not reachable at {H.BASE_URL}/health. "
            "Check pod is running and the URL is correct."
        )
    try:
        models = H.list_models()
    except Exception as e:
        pytest.exit(f"GET {H.BASE_URL}/v1/models failed: {e}")
    if not models:
        pytest.exit("No models advertised — vLLM not loaded yet?")
    print(f"\n  Deployment OK: {H.BASE_URL}  model={H.resolve_model()}  "
          f"admin={'enabled' if H.ADMIN_ENABLED else 'skipped'}")


@pytest.fixture
def conv_id():
    """Per-test sentinel conv_id with automatic cleanup. Tests that need
    a fresh conversation should request this fixture rather than calling
    fresh_conv_id() directly — that way cleanup is guaranteed even on
    failure."""
    cid = H.fresh_conv_id()
    yield cid
    H.admin_safe_forget(cid)
