"""
compactor.apiauth — V4 foundation: optional API-key auth for the public API.

When the front end is split off the GPU pod (see ARCHITECTURE.md), the
front-end → compactor hop crosses a network and can no longer rely on the
"everything is localhost" trust the single-container deploy assumes. This adds
an opt-in bearer-token gate on the public surface (the OpenAI-compatible
`/v1/*` routes).

**Backward compatible:** if `COMPACTOR_API_KEY` is unset/empty, auth is
DISABLED and behavior is unchanged — the current single-container deploy keeps
working untouched. Set `COMPACTOR_API_KEY` (and OpenWebUI's `OPENAI_API_KEY` to
match) when you split.

Scope:
  - PROTECTED: `/v1/*` (chat completions, models) — the public client surface.
  - EXEMPT:    `/health`, `/health/full` — liveness probes stay open.
  - NOT here:  `/admin/*` — already gated to localhost via `_require_localhost`
               in main.py; the network split keeps admin on the private side.

The config + path/key logic lives here so it's unit-testable without importing
the full compactor (chromadb / httpx / etc.). main.py wires the HTTP middleware
around these functions.
"""
from __future__ import annotations

import hmac
import os

# The shared secret. Empty => auth disabled (backward compatible).
API_KEY = os.environ.get("COMPACTOR_API_KEY", "").strip()
AUTH_ENABLED = bool(API_KEY)

# Only the public client surface is key-gated.
PROTECTED_PREFIXES = ("/v1/",)
# Liveness probes must stay reachable even when auth is on.
EXEMPT_PATHS = frozenset({"/health", "/health/full"})


def path_requires_auth(path: str) -> bool:
    """True if this request path must carry a valid key under current config."""
    if not AUTH_ENABLED:
        return False
    if path in EXEMPT_PATHS:
        return False
    return any(path.startswith(p) for p in PROTECTED_PREFIXES)


def _extract_key(authorization: str | None) -> str | None:
    """Pull the token from an Authorization header. Accepts 'Bearer <key>'
    (what OpenAI clients and OpenWebUI send) and tolerates a bare key."""
    if not authorization:
        return None
    value = authorization.strip()
    if value.lower().startswith("bearer "):
        token = value[7:].strip()
        return token or None
    return value or None


def key_ok(authorization: str | None) -> bool:
    """Constant-time comparison of the presented Authorization header against
    the configured key. Always True when auth is disabled."""
    if not AUTH_ENABLED:
        return True
    provided = _extract_key(authorization)
    if not provided:
        return False
    return hmac.compare_digest(provided, API_KEY)
