"""
compactor.apiauth — optional API-key auth for the public API and the admin API.

Carried forward from `v4/foundation-compactor-auth` (PR #30), which gated
`/v1/*` only and left `/admin/*` on its localhost check. FRONTEND_SPEC §3.1
names that as a hard prerequisite: every `/admin/*` route is gated by
`_require_admin_access` in main.py, so a front end in a SEPARATE CONTAINER has
a non-local source IP and receives 403 on every memory endpoint. The client
described by that spec cannot ship its memory surfaces until admin is reachable
with a credential instead of only from localhost.

**Backward compatible where it can be, and deliberately not where it cannot.**
If `COMPACTOR_API_KEY` is unset, auth is DISABLED, `/v1/*` is ungated, and
`/admin/*` keeps exactly the localhost-only behaviour it has today — the
single-container deploy is untouched.

The one behaviour that CHANGES is a configuration that was never safe. Today
`COMPACTOR_ADMIN_BIND=0.0.0.0` opens all twenty-five admin routes — forget,
import, fork, prune, restore — to any source on the network with no credential
at all. That is not an opt-in to external admin, it is an opt-in to
unauthenticated external admin, and nothing in the code said so. Under this
module an external bind REQUIRES a key, and refuses with a message naming the
variable when there is none. Nothing on the current pod is affected:
`COMPACTOR_ADMIN_BIND` is unset there, so the localhost path is the only one
in use.

Scope:
  - `/v1/*`     key-gated when a key is set (the public client surface).
  - `/admin/*`  localhost OR a valid key. See `admin_access` for the full rule.
  - `/health`, `/health/full`  always open — liveness probes must not need a
    credential, and `/health/full` is what an operator reaches for first when
    something is wrong.

Config, path rules and the key comparison live HERE rather than in main.py so
they are unit-testable without importing the full compactor (chromadb, httpx,
FastAPI). main.py wires the HTTP middleware and the route dependency around
these functions and holds no copy of the logic — one rule, two callers, which
on this codebase is the difference between a fix and a fix at one of two
sites.
"""
from __future__ import annotations

import hmac
import os

# The shared secret. Empty => auth disabled (backward compatible).
API_KEY = os.environ.get("COMPACTOR_API_KEY", "").strip()
AUTH_ENABLED = bool(API_KEY)

# The public client surface, gated by the middleware.
PROTECTED_PREFIXES = ("/v1/",)
# The admin surface, gated by the route dependency (it needs the client host,
# which a prefix test cannot see).
ADMIN_PREFIXES = ("/admin/",)
# Liveness probes must stay reachable even when auth is on.
#
# REDUNDANT TODAY, and deliberately kept: `/health*` does not match
# PROTECTED_PREFIXES either, so removing this set changes no behaviour -
# a mutation proving exactly that stayed green. It earns its place the day
# PROTECTED_PREFIXES widens (a bare `/` prefix, or a versioned health
# route), which is precisely when nobody will be thinking about liveness.
EXEMPT_PATHS = frozenset({"/health", "/health/full"})

LOCAL_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})
DEFAULT_ADMIN_BIND = "127.0.0.1"


def path_requires_auth(path: str) -> bool:
    """True if this request path must carry a valid key under current config.

    `/admin/*` is deliberately NOT here. Admin needs the client host as well as
    the header, so it is decided by `admin_access` at the route rather than by
    a prefix test in the middleware — a middleware that returned 401 for a
    localhost admin call would break the pod's own selftest, backup and
    `curl localhost:8080/admin/...`, none of which carry a key.
    """
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
    the configured key. Always True when auth is disabled.

    hmac.compare_digest, not `==`: the comparison is against a secret, and
    `==` returns as soon as two bytes differ, which leaks the shared prefix
    length by timing. It costs nothing to be right about this.
    """
    if not AUTH_ENABLED:
        return True
    provided = _extract_key(authorization)
    if not provided:
        return False
    return hmac.compare_digest(provided, API_KEY)


def admin_access(
    client_host: str | None,
    authorization: str | None,
    admin_bind: str = DEFAULT_ADMIN_BIND,
) -> tuple[bool, str]:
    """May this request reach `/admin/*`? -> (allowed, reason-if-not).

    A pure function of three inputs so the whole rule can be tested without a
    request object, and so there is exactly one place where it is written.
    main.py's dependency turns a False into the 403.

    THE FOUR CASES, and why each is what it is:

      1. LOCALHOST -> allow, always, key or no key. The compactor's own
         selftest, the backup daemon and every runbook command in
         OPERATIONS.md reach admin over localhost without a credential.
         Requiring one there would break the pod's own operation to protect it
         from itself.

      2. Remote WITH a valid key -> allow. This is the whole point: it is what
         lets the front end in a separate container use the memory API, and it
         is what FRONTEND_SPEC §3.1 is blocked on.

      3. Remote, auth DISABLED, default bind -> refuse. Unchanged from the
         behaviour this replaces, and the message still names
         COMPACTOR_ADMIN_BIND so an operator reading it learns the same thing
         they learned before — plus the key, which is now the better answer.

      4. Remote, auth DISABLED, EXTERNAL bind -> refuse, and this is the
         change. The old code returned early on `ADMIN_BIND != "127.0.0.1"`
         and allowed any source with no credential. Twenty-five routes
         including forget, import, fork and restore. Opting into external
         admin is a reasonable thing to want; opting into UNAUTHENTICATED
         external admin is not a thing anyone should be able to do by setting
         one variable, and the old name gave no hint that it did.
    """
    if client_host in LOCAL_HOSTS:
        return True, ""
    if AUTH_ENABLED:
        if key_ok(authorization):
            return True, ""
        return False, (
            "admin endpoints require an API key from a non-local address; "
            "send 'Authorization: Bearer <COMPACTOR_API_KEY>'"
        )
    if admin_bind != DEFAULT_ADMIN_BIND:
        return False, (
            "COMPACTOR_ADMIN_BIND exposes the admin API off-host but "
            "COMPACTOR_API_KEY is not set, so there is no credential to check. "
            "Refusing rather than serving forget/import/fork/restore to any "
            "caller on the network. Set COMPACTOR_API_KEY."
        )
    return False, (
        "admin endpoints are localhost-only unless COMPACTOR_API_KEY is set; "
        "set it and send 'Authorization: Bearer <key>' to reach them remotely"
    )
