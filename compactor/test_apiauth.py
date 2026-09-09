"""API-key auth for the public surface, and key-or-localhost for admin.

B1 of FRONTEND_HANDOFF: `/admin/*` was localhost-only, so the front end
described by FRONTEND_SPEC — which runs in a SEPARATE CONTAINER — got 403 on
every memory endpoint. Carried forward from PR #30, which gated `/v1/*` only.

WHAT THIS FILE HAS TO PIN, beyond "the key works":

  * the localhost path still works with NO key, because the pod's own
    selftest, backup daemon and every runbook command in OPERATIONS.md reach
    admin that way;
  * `/health` and `/health/full` stay open even with auth on — a liveness
    probe that needs a credential is not a liveness probe, and /health/full is
    what an operator reaches for first;
  * `/admin/*` is NOT gated by the middleware. It is decided at the route,
    because the rule needs the client host as well as the header. A middleware
    that 401'd on path alone would break every localhost admin call;
  * and the one DELIBERATE behaviour change: an external COMPACTOR_ADMIN_BIND
    with no key set is now refused rather than served. It used to return early
    and hand twenty-five routes — forget, import, fork, restore — to any
    caller on the network with no credential at all.

    python test_apiauth.py
"""

import os
import sys
import tempfile

# Set BEFORE importing apiauth: the key is read at module scope, which is what
# lets the disabled case be the zero-config default in production.
os.environ["COMPACTOR_API_KEY"] = "test-key-abc123"
os.environ.setdefault("MODEL_REPO", "test-model")
os.environ.setdefault("VLLM_URL", "http://stub:8000")
os.environ["COMPACTOR_RAG_ENABLED"] = "false"
os.environ["COMPACTOR_STORAGE_ROOT"] = tempfile.mkdtemp(prefix="apiauth-")

import apiauth  # noqa: E402

FAILED: list[str] = []


def check(cond, label):
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"FAIL {label}")
        FAILED.append(label)


KEY = "test-key-abc123"

print("[1] path gating on the public surface")
check(apiauth.AUTH_ENABLED, "auth is on for this file (key set before import)")
check(apiauth.path_requires_auth("/v1/chat/completions"),
      "/v1/chat/completions is gated")
check(apiauth.path_requires_auth("/v1/models"), "/v1/models is gated")
check(not apiauth.path_requires_auth("/health"), "/health stays open")
check(not apiauth.path_requires_auth("/health/full"),
      "/health/full stays open — it is the first thing an operator reaches for")
check(not apiauth.path_requires_auth("/admin/conversations"),
      "/admin/* is NOT middleware-gated: the rule needs the client host, and a "
      "path-only 401 would break every localhost admin call")

print("[2] the key comparison")
check(apiauth.key_ok(f"Bearer {KEY}"), "a bearer token is accepted")
check(apiauth.key_ok(KEY), "a bare key is accepted (tolerated, not required)")
check(apiauth.key_ok(f"bearer {KEY}"), "the scheme is case-insensitive")
check(not apiauth.key_ok("Bearer wrong"), "a wrong key is refused")
check(not apiauth.key_ok("Bearer "), "an empty bearer token is refused")
check(not apiauth.key_ok(None), "a missing header is refused")
check(apiauth.key_ok(f"Bearer {KEY} "),
      "surrounding whitespace is tolerated")

print("[3] admin access — the four cases, as a pure function")
for host in ("127.0.0.1", "::1", "localhost"):
    ok, _ = apiauth.admin_access(host, None)
    check(ok, f"localhost ({host}) reaches admin with NO key — the pod's own "
              f"selftest and backup depend on this")

ok, _ = apiauth.admin_access("10.0.0.5", f"Bearer {KEY}")
check(ok, "a remote caller WITH a valid key reaches admin — this is B1, and "
          "what FRONTEND_SPEC §3.1 is blocked on")

ok, why = apiauth.admin_access("10.0.0.5", "Bearer wrong")
check(not ok and "API key" in why,
      f"a remote caller with a WRONG key is refused, and told why ({why[:40]}…)")

ok, why = apiauth.admin_access("10.0.0.5", None)
check(not ok, "a remote caller with no key at all is refused")

print("[4] the deliberate behaviour change: external bind needs a key")
# Simulate the zero-config deploy: no key anywhere.
_saved_key, _saved_enabled = apiauth.API_KEY, apiauth.AUTH_ENABLED
apiauth.API_KEY, apiauth.AUTH_ENABLED = "", False
try:
    ok, _ = apiauth.admin_access("127.0.0.1", None)
    check(ok, "with auth OFF, localhost is unchanged — the single-container "
              "deploy is untouched")

    ok, why = apiauth.admin_access("10.0.0.5", None, admin_bind="127.0.0.1")
    check(not ok and "localhost-only" in why,
          "with auth OFF and the default bind, a remote caller is refused "
          "exactly as before")

    # The WORDING, not just the refusal: both refusal paths mention
    # COMPACTOR_API_KEY, so asserting that could not tell which branch
    # fired - and a mutation that disabled this branch entirely stayed
    # green until this assertion was sharpened.
    ok, why = apiauth.admin_access("10.0.0.5", None, admin_bind="0.0.0.0")
    check(not ok and "COMPACTOR_ADMIN_BIND exposes" in why,
          "COMPACTOR_ADMIN_BIND=0.0.0.0 with NO key is now REFUSED and names "
          "the variable to set. It used to serve forget/import/fork/restore to "
          "any caller on the network — an opt-in to unauthenticated external "
          "admin that nothing in the name disclosed")

    check(not apiauth.path_requires_auth("/v1/chat/completions"),
          "and with auth off /v1 is ungated, so the current deploy keeps "
          "working with no configuration at all")
finally:
    apiauth.API_KEY, apiauth.AUTH_ENABLED = _saved_key, _saved_enabled

print("[5] end to end through the real app, not a reconstruction of it")
import memory  # noqa: E402

memory.ensure_storage_layout()

from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402

# A non-local source address is the whole point: this is what a front end in a
# separate container looks like to the compactor.
remote = TestClient(main.app, client=("10.0.0.5", 51000),
                    raise_server_exceptions=False)
local = TestClient(main.app, client=("127.0.0.1", 51000),
                   raise_server_exceptions=False)

r = remote.get("/admin/conversations")
check(r.status_code == 403,
      f"a remote admin call with no key is 403 (got {r.status_code})")

r = remote.get("/admin/conversations", headers={"Authorization": f"Bearer {KEY}"})
check(r.status_code == 200,
      f"the SAME call with the key is served (got {r.status_code}) — the 403 "
      f"above is the credential, not the route being broken")

r = local.get("/admin/conversations")
check(r.status_code == 200,
      f"localhost still needs no key (got {r.status_code})")

r = remote.get("/health/full")
check(r.status_code == 200,
      f"/health/full is reachable from anywhere, keyless (got {r.status_code})")

r = remote.post("/v1/chat/completions", json={"model": "m", "messages": []})
check(r.status_code == 401,
      f"the public surface refuses a keyless call with 401, not 403 (got "
      f"{r.status_code}) — 401 is 'authenticate', which is what an OpenAI "
      f"client knows how to act on")

if FAILED:
    print(f"\n{len(FAILED)} check(s) FAILED")
    sys.exit(1)
print("\nAll apiauth checks passed.")
