# RUNBOOK: Cloudflare Tunnel for OpenWebUI (v3.1.9.9)

Gives her OpenWebUI a stable URL, `https://ai.revelationsaints.com`, instead of the RunPod proxy URL —
served through a Cloudflare Tunnel (`cloudflared`) that dials OUT from the pod to Cloudflare's edge, so no
inbound port needs to be opened on the pod itself. The RunPod proxy URL **keeps working** alongside this
during the transition — nothing about it is disabled by adding the tunnel.

This is the only place the tunnel design is documented. OPERATIONS.md has a one-paragraph pointer to this
file and nothing else; CHANGELOG.md's `[3.1.9.9]` entry lists what shipped and where.

---

## 1. One-time Cloudflare dashboard setup (owner does this once)

1. Log in to the Cloudflare dashboard for the `revelationsaints.com` zone.
2. **Zero Trust → Networks → Tunnels → Create a tunnel.**
3. Choose **Cloudflared** as the connector type (not WARP Connector).
4. Name it (e.g. `zions-light-ai`). Cloudflare generates a tunnel and shows a **token** (a long string,
   `eyJ...`) — this is the only secret this whole setup needs. Copy it now; it is shown again later under the
   tunnel's own "Configure" page if you need it, but treat it as a credential (don't paste it into chat,
   email, or a screenshot you'll share).
5. Under that tunnel's **Public Hostname** tab, add one:
   - **Subdomain**: `ai`
   - **Domain**: `revelationsaints.com` (so the public hostname is `ai.revelationsaints.com`)
   - **Service Type**: `HTTP`
   - **URL**: `localhost:3000` (OpenWebUI's port inside the pod — `OPENWEBUI_PORT`, unchanged from today)
6. Save. Cloudflare now has a DNS record + edge route for `ai.revelationsaints.com` pointing at whatever
   `cloudflared` process on the pod authenticates with that tunnel's token — nothing else to configure on
   the Cloudflare side for a basic HTTP proxy.

**Optional — Cloudflare Access (an email one-time-PIN gate in front of the tunnel):**

Zero Trust → Access → Applications → Add an application → Self-hosted, with the same public hostname
(`ai.revelationsaints.com`). Add a policy allowing her specific email address, action "Allow", with the
"One-time PIN" login method (no account/password to manage — Cloudflare emails her a 6-digit code each time
a browser session expires). What this changes for her: before she ever reaches OpenWebUI's own login page,
she gets a Cloudflare-hosted page asking for her email, then a PIN emailed to her; after that, Access issues
its own short-lived cookie and she is not asked again until it expires (configurable session length in the
Access policy, e.g. 24h or 30 days). This is a SEPARATE login from OpenWebUI's own (`WEBUI_AUTH`) — both
stay in place if you add Access; it is a gate in front of, not a replacement for, OpenWebUI's own auth. Not
required for the tunnel itself to work; it is defense-in-depth if you want to keep the URL from being usable
by anyone who guesses it before she signs in.

## 2. The RunPod secret

Add an environment variable to the RunPod pod template (the same place `WEBUI_AUTH`, `HF_HOME`, etc. already
live):

```
CLOUDFLARE_TUNNEL_TOKEN=<the token from step 1.4>
```

Use RunPod's "Secret" field type for it if the template UI offers one (masks it in the RunPod dashboard);
either way it reaches the container as a normal environment variable. **Never** paste the real token into
this repo, a commit, a chat log, or this runbook — there is no real token in any file here, only obvious
fakes in the test suite (`compactor/test_cloudflared_run.py`).

How it reaches `cloudflared`: `entrypoint.sh` execs `supervisord` as the container's PID 1
(`entrypoint.sh`, the final line, `exec /usr/bin/supervisord ...`), so supervisord — and every program it
spawns, including `cloudflared` via `docker/cloudflared-run.sh` — inherits the full container environment
automatically. `[program:cloudflared]` in `supervisord.conf` deliberately has **no** `environment=` line
naming `CLOUDFLARE_TUNNEL_TOKEN` (see that section's own comment): unlike `STT_ENABLED`/`TTS_ENABLED`/etc.,
which entrypoint.sh normalizes to a concrete `true`/`false` before supervisord starts (an
`autostart=%(ENV_x)s` reference to an unset var crashes supervisord's config load outright), a secret token
has no safe default to normalize to — so this program is never gated that way, and instead always
autostarts, with `docker/cloudflared-run.sh` itself deciding enabled vs. disabled at runtime by checking
whether the variable is set. No `ops/zlops` `/proc/1/environ` trick is needed for `cloudflared` to see the
token — it's a direct supervisord child, not a separate shell that would miss what entrypoint.sh exported.
(That `/proc/1/environ` pattern, used by `ops/zlops/podenv.py` and `scripts/_webui_live_path.py`, exists for
tooling running OUTSIDE the supervisord tree — an SSH shell, `zl status` — see the "ops asks" section below
for what that tooling still needs to know about this program.)

## 3. Verification, on the pod

```bash
supervisorctl status cloudflared
```

- `EXITED` (expected, not FATAL) — the token is not set, or is empty. The tunnel is disabled by design; this
  is not an error state. Check the log to confirm why:
  ```bash
  tail -5 ${LOG_DIR:-/data/logs}/cloudflared.log
  # should end with: ... [cloudflared-run] no CLOUDFLARE_TUNNEL_TOKEN; tunnel disabled
  ```
- `RUNNING` — the tunnel is up and dialed out to Cloudflare's edge.
- `FATAL` — should not happen by design (see the CHANGELOG's `[3.1.9.9]` entry on why `startsecs=0` is
  required for this). If you ever see it, `cloudflared` itself failed to even spawn (e.g. the binary is
  missing/corrupted) — check `${LOG_DIR}/cloudflared-error.log`.

Then, from anywhere (not just the pod):

```bash
curl -sI https://ai.revelationsaints.com
```

Expect an HTTP response with OpenWebUI's own headers (not a Cloudflare error page). A `524` here means
Cloudflare's edge gave up waiting on the origin — see the timeout finding below for when that can happen and
why it does not mean her chat failed.

`/health/full`'s `tunnel` block (compactor, not cloudflared itself):

```bash
curl -s http://127.0.0.1:${COMPACTOR_PORT:-8080}/health/full | python3 -c "import json,sys; print(json.load(sys.stdin)['tunnel'])"
# {"configured": true, "ready": true}
```

`configured` reflects only whether `CLOUDFLARE_TUNNEL_TOKEN` is set (never its value); `ready` is
cloudflared's own `/ready` endpoint on its loopback-only metrics port (127.0.0.1:8082), true once it has an
active connection to Cloudflare's edge. Visibility only — this never changes `/health/full`'s overall
`status`, same as the existing `word_ban`/`tokenizer`/`reuse` blocks.

## 4. Her switch

1. Once `curl -sI https://ai.revelationsaints.com` looks right, have her open that URL in her normal browser.
2. She'll need to **sign in once** on the new origin — OpenWebUI's session cookie is scoped per-origin, so a
   session from the RunPod proxy URL does not carry over to the new one.
3. **Re-add the home-screen icon** (if she uses one on a phone/tablet) pointing at
   `https://ai.revelationsaints.com` instead of the old RunPod proxy URL — the old shortcut still works (the
   proxy URL is untouched), but a new one on the tunnel URL is what you actually want her using day to day.
4. The RunPod proxy URL is not removed or disabled by any of this — it is fine to leave it as a fallback
   during the transition, and to decide later (separately) whether to keep it around long-term.

## 5. Rollback / disabling

- **Disable the tunnel entirely:** remove the `CLOUDFLARE_TUNNEL_TOKEN` secret from the RunPod template and
  restart the `cloudflared` program (`supervisorctl restart cloudflared`) or redeploy — `docker/
  cloudflared-run.sh` sees no token, logs "tunnel disabled", and exits 0 (`EXITED`, not a crash). Nothing
  else on the pod is affected; OpenWebUI keeps serving the RunPod proxy URL exactly as before.
- **Roll back the image itself:** the `rollback` build target (`local/zla:v3199-rollback` /
  `Dockerfile.v3199 --target rollback`) has **no** `cloudflared` binary and **no** `[program:cloudflared]`
  stanza at all — it is defined as "what a pod running today would fall back to", and today's pod has no
  tunnel. Booting the rollback target is itself the full rollback; there is no separate tunnel-specific
  rollback step needed on that path.
- Deleting the Cloudflare-side tunnel/hostname (Zero Trust dashboard) is independent of both of the above and
  only needed if you want to stop paying attention to it there too — not required for the pod to keep
  working correctly either way.

---

## Findings: OpenWebUI 0.11.4 behind a new origin

Extracted and read directly from the image's own `/app/venv/.../open_webui` (OpenWebUI 0.11.4, matching what
prod runs and what `Dockerfile.v3199`'s `rollback`/`rc` stages ship) — not from OpenWebUI's public docs,
which can drift from what an installed version actually does. Every line number below is from that source
tree, `docker cp`'d out of `local/zla:v3198-rc` for reading (a throwaway container, never a run).

### Env vars worth setting for the new origin

| Var | File:line (default) | Recommendation for the RunPod template |
|---|---|---|
| `WEBUI_URL` | `config.py:1661` (`os.getenv('WEBUI_URL', '')`, stored as `webui.url`, `config.py:3082`) | `https://ai.revelationsaints.com` |
| `CORS_ALLOW_ORIGIN` | `config.py:2134` (`os.getenv('CORS_ALLOW_ORIGIN', '*')`) | Leave at the default `*` unless you have a reason to lock it down — the tunnel does not require narrowing this. |
| `WEBUI_SESSION_COOKIE_SECURE` | `env.py:770` (default `false`) | Leave as-is or set `true` — `https://ai.revelationsaints.com` is HTTPS end-to-end (Cloudflare terminates TLS at its edge; the tunnel and the loopback hop to `localhost:3000` are both trusted paths), so a `Secure` cookie works fine over it. Not required to change for the tunnel to function. |
| `WEBUI_SESSION_COOKIE_SAME_SITE` | `env.py:769` (default `lax`) | No change needed — `lax` is fine for a single-origin app like this. |
| `FORWARDED_ALLOW_IPS` | N/A — hardcoded, not an env var here | **No action needed.** `open_webui/__init__.py:88` and `:108` already pass `forwarded_allow_ips='*'` to `uvicorn.run(...)` unconditionally, so `X-Forwarded-*` handling is already permissive regardless of what proxies in front of it. |

**Why `WEBUI_URL` is worth setting** (not just cosmetic): it's read wherever the app needs to build an
absolute link back to itself rather than a relative path — `utils/middleware.py:214,227` and `:4392,4404`
(the URL embedded in a push/webhook notification payload, `f'{webui_url}/c/{chat_id}'`, falls back to a bare
relative path if unset), `utils/oauth.py:514-515,664-665,1318-1319,2175-2176` (OAuth redirect base URL),
`routers/channels.py:942,949` (channel links), and `main.py:2922,2931-2933` (the OpenSearch descriptor
OpenWebUI serves for browser "add as search engine" integration). None of these BLOCK basic chat use if
`WEBUI_URL` is left empty — they just fall back to a relative path or `request.base_url` — but leaving it
unset means any of those generated links point at whatever host happened to serve that particular request,
which is exactly the kind of thing that gets confusing once there are two live origins (tunnel + RunPod
proxy) during the transition.

**Do NOT change the image's own defaults for any of these** — this table is RunPod-template guidance, not a
`Dockerfile.v3199` change. Nothing above required touching the image.

### Cloudflare limits checked against her actual usage pattern

1. **Proxied origin response timeout (~100s to first byte on Free) vs. her cold-start reply time (~74s,
   up to ~175s with a cold grammar compile).**

   Traced the actual delivery path for a chat reply: `POST /api/chat/completions` (`main.py:1108-1110`)
   calls `process_chat_response()` (`utils/middleware.py:6720`), which for a streaming backend response
   dispatches to `streaming_chat_response_handler()` (`utils/middleware.py:4431`). Inside it, when a
   real browser client is talking (an `event_emitter` present — the normal case), **every** content delta
   is sent as a `chat:completion` event through `event_emitter(...)`, which resolves to
   `sio.emit('events', ..., room=f'user:{user_id}')` — real Socket.IO, `socket/main.py:1152`
   (`get_event_emitter`, `socket/main.py:1134`) — i.e., **the actual token-by-token content the user sees
   arrives over the WebSocket, not the HTTP response body.**

   BUT the original HTTP request is not returned early either: `utils/middleware.py:6642` is
   `return await response_handler(response, events)` — an `await`, not a fire-and-forget background task.
   The coroutine handling `POST /api/chat/completions` stays suspended, holding that HTTP connection open,
   for the ENTIRE generation, and only returns (a small completion-status JSON, not the reply text) once
   generation finishes. Nothing in this path sends periodic keep-alive bytes on that HTTP response while
   waiting.

   **Conclusion: a `524` on the initiating `POST /api/chat/completions` CAN happen** for any generation
   that runs past Cloudflare's ~100s idle-origin timeout (Free plan) — which her cold-compile case (up to
   ~175s) exceeds. It does **not** mean her reply is lost: the content itself is already flowing to her
   browser over the separate, independent WebSocket connection as it's generated, and that connection is
   not subject to the same per-request HTTP timeout (Cloudflare Tunnel proxies WebSocket traffic as a single
   long-lived stream, not as a request/response pair with its own 100s clock — see the websockets finding
   below). **What is NOT fully verified from source alone:** whether OpenWebUI's frontend treats that
   specific POST's failure (a `524`, which the browser sees as the fetch erroring or timing out) as a
   user-visible error/retry prompt even though the message is actually still arriving via the socket. No
   compiled frontend source shipped in this package to check that directly (`open_webui/frontend/static`
   in the image is the built app, not readable TypeScript source). **Recommendation:** treat this as a
   residual risk to watch for after cutover, not a blocker — worth one deliberate test of a real cold-start
   reply through the tunnel, watching the browser console/network tab for how it reacts to the POST outcome.
   If it does surface a spurious error toast while the reply arrives fine underneath, the practical
   workaround is upgrading past Cloudflare's Free-plan proxied timeout (a paid plan setting) or keeping the
   compactor's own idle warm-up (`wordban_runtime.py`'s warm-up path, OPERATIONS.md) doing its job so a cold
   compile is rare in practice, not eliminating the timeout risk at the protocol level.

2. **Upload body limit (100 MB on Cloudflare Free) vs. her photos.** Nothing in OpenWebUI's own source
   imposes a smaller limit ahead of Cloudflare's own (`env.py:671` `AIOHTTP_FILE_STREAM_CHUNK_SIZE` is a
   streaming chunk size, not a cap; no `MAX_UPLOAD_SIZE`/`client_max_body_size`-style setting exists in this
   version). A single photo from a phone camera is very unlikely to approach 100 MB. If she ever uploads
   something larger (e.g. a long voice memo or a video), Cloudflare's Free-plan 100 MB body limit would
   reject it with its own error page before it reaches the pod at all — no change made for this; flagging
   it here so it's a known limit rather than a surprise.

3. **WebSockets through the tunnel.** On by default for both sides: Cloudflare Tunnel proxies WebSocket
   upgrades transparently with no extra configuration, and OpenWebUI's own Socket.IO server defaults to
   **websocket-only** transport (`socket/main.py:102`/`:117`,
   `transports=(['websocket'] if ENABLE_WEBSOCKET_SUPPORT else ['polling'])`, and
   `env.py:473` shows `ENABLE_WEBSOCKET_SUPPORT` defaults to `True`) — i.e., there is no automatic
   long-polling fallback if a websocket upgrade were ever blocked somewhere in the path. Nothing to
   configure for this setup (both defaults already line up), but worth knowing if real-time updates (typing
   indicators, live deltas) ever stop working through the tunnel while plain HTTP still does — that would
   point at a websocket-upgrade problem specifically, not a general connectivity one.

---

## What `ops`/`zl status`/`podenv` would need (listed here, not implemented — `ops/` is owned by a separate
session; this list is the ask, to be relayed)

- `podenv.py`'s `SERVICES`/`ONE_SHOT_OR_OPTIONAL` tuples (`ops/zlops/podenv.py:97,101`) don't yet include
  `cloudflared`. Without an entry, `zl status`/the dash likely has no opinion on it either way today — but
  once it does inspect every `[program:...]` generically, `cloudflared`'s `EXITED` (disabled, by design) can
  read as a problem unless it's added to `ONE_SHOT_OR_OPTIONAL` alongside `selftest`/`webuidb-sync`/`zl-ops`
  (all of which are "EXITED can be fine" programs for their own reasons).
- If `zl status`/the dash ever wants to say something more specific than "EXITED (ok)" for `cloudflared` —
  e.g. distinguish "disabled by config" from "briefly down between retries" — the `/health/full` `tunnel`
  block (`configured`/`ready`) is the source to read for that; the ops session can decide whether that's
  worth wiring in or whether the generic one-shot/optional handling is enough.
- Nothing else needed. This program does not need a port, a data path, or anything else `podenv.py`
  currently centralizes for other services.
