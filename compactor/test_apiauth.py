"""
CPU-only Tier-1 tests for compactor.apiauth (V4 API-key gate).

Pure: just the module — no FastAPI, no network. Verifies the backward-compatible
disabled default, which paths are gated, and the bearer/bare-key checks.
Run: python test_apiauth.py
"""
import sys

import apiauth


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


def _set(enabled, key="s3cret"):
    """Patch the module's config to a known state (mirrors how the real env
    would be read at import)."""
    apiauth.API_KEY = key if enabled else ""
    apiauth.AUTH_ENABLED = bool(enabled)


# ---------------------------------------------------------------------------
# Disabled (default / current single-container deploy) — everything passes
# ---------------------------------------------------------------------------

def test_disabled_is_backward_compatible():
    print("\n[test] auth disabled (no key) -> nothing gated, all keys ok")
    _set(False)
    assert_eq(apiauth.path_requires_auth("/v1/chat/completions"), False, "/v1 not gated when off")
    assert_eq(apiauth.key_ok(None), True, "missing key allowed when off")
    assert_eq(apiauth.key_ok("Bearer anything"), True, "any key allowed when off")


# ---------------------------------------------------------------------------
# Enabled — path gating
# ---------------------------------------------------------------------------

def test_enabled_path_gating():
    print("\n[test] auth enabled -> only /v1/* gated; /health + /admin not")
    _set(True)
    assert_eq(apiauth.path_requires_auth("/v1/chat/completions"), True, "/v1/chat gated")
    assert_eq(apiauth.path_requires_auth("/v1/models"), True, "/v1/models gated")
    assert_eq(apiauth.path_requires_auth("/health"), False, "/health exempt")
    assert_eq(apiauth.path_requires_auth("/health/full"), False, "/health/full exempt")
    assert_eq(apiauth.path_requires_auth("/admin/conversations"), False,
              "/admin not gated here (localhost-gated in main.py)")


# ---------------------------------------------------------------------------
# Enabled — key checking
# ---------------------------------------------------------------------------

def test_enabled_key_checks():
    print("\n[test] auth enabled -> bearer/bare-key checks")
    _set(True, "s3cret")
    assert_eq(apiauth.key_ok("Bearer s3cret"), True, "correct bearer key ok")
    assert_eq(apiauth.key_ok("bearer s3cret"), True, "scheme is case-insensitive")
    assert_eq(apiauth.key_ok("s3cret"), True, "bare key tolerated")
    assert_eq(apiauth.key_ok("Bearer wrong"), False, "wrong key rejected")
    assert_eq(apiauth.key_ok(None), False, "missing header rejected")
    assert_eq(apiauth.key_ok(""), False, "empty header rejected")
    assert_eq(apiauth.key_ok("Bearer "), False, "empty bearer token rejected")


def test_extract_key():
    print("\n[test] _extract_key parsing")
    assert_eq(apiauth._extract_key("Bearer abc"), "abc", "strips Bearer")
    assert_eq(apiauth._extract_key("  Bearer   abc  "), "abc", "trims whitespace")
    assert_eq(apiauth._extract_key("abc"), "abc", "bare key passes through")
    assert_eq(apiauth._extract_key(None), None, "None -> None")
    assert_eq(apiauth._extract_key(""), None, "empty -> None")


def _all():
    return [
        test_disabled_is_backward_compatible,
        test_enabled_path_gating,
        test_enabled_key_checks,
        test_extract_key,
    ]


if __name__ == "__main__":
    for t in _all():
        t()
    print("\nAll apiauth tests passed.")
