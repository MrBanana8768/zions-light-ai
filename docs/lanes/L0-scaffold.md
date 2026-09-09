# L0 — Scaffold (F1 + F2)

Branch `feat/frontend`. Scope was F1 (repo scaffold) and F2 (co-location) only,
per `FRONTEND_PLAN.md` §4 and the delegation table at §5. No git commands were
run by this lane; everything below is unstaged in the working tree for the
owner to review and commit.

Read first: `FRONTEND_PLAN.md`, `FRONTEND_HANDOFF.md` §6/§7. This report
assumes both.

---

## 1. What was built, and where

### `frontend/` — new SvelteKit app

Scaffolded with `npx sv create --template minimal --types ts`, then hand-built
on top. TypeScript end-to-end (`strict: true` in `tsconfig.json`, inherited
from the generated `.svelte-kit/tsconfig.json`); `svelte-check` reports **0
errors, 0 warnings** across 172 files as of the last run in this lane.

```
frontend/
├── README.md                          how to run/build; what's L0 vs. later lanes
├── package.json                       engines.node >=22; devDependencies only
├── package-lock.json
├── .npmrc                             engine-strict=true (from the scaffolder)
├── .gitignore                         (from the scaffolder — excludes node_modules,
│                                        /build, /.svelte-kit, .env*)
├── tsconfig.json
├── vite.config.ts                     SvelteKit config now lives HERE (see §2
│                                        below) — adapter-node, not adapter-auto
├── static/
│   └── robots.txt                     (scaffold default, untouched)
└── src/
    ├── app.html                       pre-paint theme script (§13's "before
    │                                   first paint" requirement) — see §1.2
    ├── app.d.ts                       (scaffold default, untouched)
    ├── lib/
    │   ├── index.ts                   (scaffold default placeholder, untouched)
    │   ├── assets/favicon.svg         (scaffold default)
    │   ├── theme.svelte.ts            theme preference store (Svelte 5 runes)
    │   ├── styles/
    │   │   ├── tokens.css             every design token — see §1.1
    │   │   ├── reset.css              a small, deliberate reset
    │   │   └── global.css             base styles, focus-visible, imports
    │   ├── i18n/
    │   │   ├── index.ts               t() lookup — the i18n seam
    │   │   └── locales/en.json        every user-facing string in this lane
    │   └── components/shell/
    │       ├── AppShell.svelte        header + rail + main frame, theme sync
    │       ├── ConversationRail.svelte   empty, "No conversations yet."
    │       ├── MessagePane.svelte     empty, placeholder text
    │       └── ThemeToggle.svelte     System / Light / Dark button group
    └── routes/
        ├── +layout.svelte             wires global.css + AppShell
        └── +page.svelte               renders MessagePane
```

**No chat logic, no store, no fetch to `:8080` or anywhere else exists
anywhere in this tree.** `grep -rn "fetch(" frontend/src` returns nothing.

### `supervisord.conf` — one new stanza

`[program:client]` added directly after `[program:openwebui]` (see the diff in
§4). Nothing else in this file was touched.

### `Dockerfile` — one new hunk + one shared line

A Node.js install + client build block added after the OpenWebUI venv block,
plus `3001` added to the existing shared `EXPOSE` line and its comment (see
§4). Nothing else in this file was touched.

### `docs/lanes/L0-scaffold.md`

This file.

---

## 1.1 Design tokens (`frontend/src/lib/styles/tokens.css`)

Complete, hand-authored CSS custom property set: color (a neutral scale +
accent/danger/warning/success + semantic aliases), space (0–4rem, 4px base),
type (system font stack, a 6-step size scale, 3 line-heights, 3 weights),
radii (4 steps), motion (3 durations + 2 easings). **No `@telos-llc/*`
dependency, and no attempt to reproduce Doulos's real values** — see decision
D-1 below; this is a from-scratch set the plan's naming convention allowed for,
not a copy of anything.

Theming: light values on bare `:root`; `@media (prefers-color-scheme: dark)`
guarded by `:not([data-theme="light"])` for the system default; `[data-theme]`
on `<html>` wins over both. `prefers-reduced-motion: reduce` collapses all
three duration tokens to near-zero in one place.

## 1.2 Theme before first paint

`src/app.html` carries an inline `<script>` in `<head>`, before the SvelteKit
head placeholder, that reads `localStorage['zl-theme']` synchronously and sets
`data-theme`/`data-theme-pref` on `<html>` before any stylesheet is linked.
`src/lib/theme.svelte.ts` (Svelte 5 runes state, shared across components)
takes over after hydration — same storage key, same fallback logic, applied
via an `$effect` in `AppShell.svelte`.

**Verified interactively** (Browser pane, built `adapter-node` server, not
`vite dev`): dark renders correctly on first load under a dark system
preference; the toggle switches to light instantly; a hard reload after
picking "Light" reloads light with **no dark flash**; Tab reveals a
skip-link with a visible focus ring; the 640px breakpoint stacks the rail
above the message pane on a 375px viewport. No console errors, no hydration
mismatches.

## 1.3 A bug this lane found and fixed in itself

The first draft of `app.html`'s pre-paint-script comment explained the script
in prose that included the literal string `%sveltekit.head%` — "this script
must run before `%sveltekit.head%` below." SvelteKit's `app.html` templating
is a **raw string substitution**, not HTML-comment-aware: it replaced that
occurrence too, and the substituted markup's own `<!---->` marker closed the
outer comment early, leaking the rest of the sentence as visible page text.
Caught by an actual browser screenshot, not by `svelte-check` or the build
(both were silent). Fixed by rephrasing to avoid the literal token, with a
comment warning the next editor not to reintroduce it. **Lesson for later
lanes:** never write a literal `%sveltekit.*%` token as prose inside
`app.html`, comments included.

## 1.4 i18n seam

`src/lib/i18n/index.ts` exports `t(key)` against `locales/en.json`. Every
string in this lane's components goes through it — `grep -rn '>[A-Z][a-z]' 
frontend/src/lib/components` (a rough hardcoded-string smell test) turns up
nothing outside the JSON file. No library (paraglide, svelte-i18n, ...) —
see decision D-3.

---

## 2. Decisions this lane made that the plan did not settle

The plan (§4, F1/F2 rows) specifies outcomes, not implementation. Everything
below is a call this lane had to make. Hard rule 5 applied: pick the
reversible option, write it down.

**D-1. Token values are authored from scratch, not copied from Doulos.**
FRONTEND_PLAN.md §2.3 says "Doulos design tokens copied into the repo, not
the packages." That phrasing presumes access to Doulos's actual values to
copy from. This thread has none — `@telos-llc/doulos` is a private registry
package, and nothing in this repo or its history contains Doulos's real
color/space/type scale. So `tokens.css` is a **complete, coherent,
hand-authored** token set using names a real Doulos import could plausibly
slot into later (`--zl-bg`, `--zl-space-4`, etc. — generic enough to not
collide with a future real import), not an attempt to reproduce Doulos's
actual numbers. **Reversible:** swapping in the real Doulos scale is an edit
to one file's values, not a restructuring — no component references a raw
color; everything goes through the semantic tokens.

**D-2. Fonts: system stack, not vendored files.** F1 allows either. No brand
typeface is named anywhere in the plan, spec, or handoff. Committing a guessed
font (and its license) seemed worse than a well-chosen system stack.
**Reversible:** swap `--zl-font-sans`/`--zl-font-mono` in `tokens.css` for
`@font-face` declarations pointing at files under `frontend/static/fonts/` —
no component change needed.

**D-3. Hand-rolled i18n seam, not a library.** Only one locale exists today
and F1 only requires "no hard-coded strings," not a real i18n runtime. A
library (paraglide is fully local/offline-capable, so it wouldn't have
violated the offline requirement) would have been defensible too, but adds a
build step and a dependency for a currently-single-locale app.
**Reversible:** `t()`'s signature (`key -> string`) is exactly what most
i18n libraries also expose; swapping the implementation inside
`lib/i18n/index.ts` doesn't need call sites to change.

**D-4. No PWA manifest / service worker in F1.** FRONTEND_SPEC.md §13 lists
"installable PWA" as a quality bar, but FRONTEND_PLAN.md §4 assigns it to
**F20** (Phase 2), not F1, and this lane's own task brief's F1 bullet list
doesn't mention it either. Built nothing toward it to avoid scope creep past
F1/F2. **Reversible:** additive — a manifest.json + service worker registration
later, no restructuring.

**D-5. Node.js installed as a pinned direct download, not apt/NodeSource.**
The Dockerfile's existing pattern for "add a new runtime to the image" is
apt (Postgres) or a Python venv (everything else) — neither fits Node
directly. Chose a pinned tarball from `nodejs.org/dist` into `/opt/node`,
mirroring the TTS voice-file download pattern already in the Dockerfile,
over adding NodeSource's third-party apt repo + signing key.
**Reversible:** swapping to an apt-based install is a self-contained
Dockerfile edit; nothing downstream references how Node got onto the image,
only that `/opt/node/bin/node` exists (which BUILD GUARD 4 checks).

**D-6. `.tar.gz`, not `.tar.xz`, for the Node download.** `.tar.xz` is
smaller but needs `xz-utils`, which this image's apt install list does not
include and whose presence is otherwise unverified. `.tar.gz` needs only
`gzip`, present in effectively every base image. Confirmed both are
published for the pinned version before choosing.

**D-7. `startsecs=5` / `stopwaitsecs=35` for `[program:client]`, evidence-based.**
F2 mandates `stopwaitsecs`; the exact numbers were mine to pick.
`startsecs=5` matches the file's own "fast-starting, no import-time-heavy-work"
group (postgres, backup, webuidb-sync) rather than the ~10s group
(compactor/openwebui/stt/tts, which do real Python import work). Measured
(Windows dev host, **not** Linux/Docker evidence per the project's own
testing doctrine): the built server reaches "Listening" in ~0.2s cold.
`stopwaitsecs=35` = adapter-node's own default `SHUTDOWN_TIMEOUT` (30s,
pinned explicitly in the stanza's `environment=` rather than left implicit)
plus 5s headroom for `closeAllConnections()` and process exit — read directly
out of `@sveltejs/adapter-node`'s generated `build/index.js`, not guessed.
Confirmed by reading that generated file: `SIGTERM` and `SIGINT` get an
**identical** graceful-shutdown handler (unlike Postgres, which needed
`stopsignal=INT` — the client needs no such override).
**Reversible:** two numbers in one stanza.

**D-8. `environment=` in the client stanza carries only `PORT`/`HOST`/
`NODE_ENV`/`SHUTDOWN_TIMEOUT`.** Mirrors how `[program:compactor]`'s own
comment explains inheriting most config from the Dockerfile's `ENV`
defaults rather than re-listing it. **Did not** add `COMPACTOR_URL` or an
`Authorization` bearer var — that belongs to L2 (transport), which doesn't
exist yet; inventing the env var name now without the code that reads it
would be exactly the "invent architecture" hard rule 5 forbids. Left a
comment pointing at where L2 should add it.

**D-9. No `docker-compose.tests.yml` / `docker-compose.integration.yml`
service additions.** FRONTEND_PLAN.md §6 describes future `client-unit` and
`client` services, but F1/F2 (this lane's actual brief) don't ask for tests
to be wired yet, and there is no test content yet for a stub service to run.
Adding empty service stanzas now seemed like exactly the kind of
premature-scaffolding hard rule 5 warns against. Left for L1/L2/L5, whichever
lands the first real test.

**D-10. `.dockerignore` left untouched.** A stray local `frontend/node_modules`
or `frontend/build` on a dev machine will bloat every future `docker build`'s
context-transfer step (the Dockerfile's explicit `COPY` list means it can
never leak into the *image*, only slow down sending the *context*). Adding
`frontend/node_modules`, `frontend/.svelte-kit`, `frontend/build` to
`.dockerignore` would fix that and is low-risk, but `.dockerignore` is not
`frontend/`, the supervisord stanza, or the Dockerfile hunk — outside the
three paths this lane owns. **Not done; recommended as a one-line follow-up**
for whoever touches `.dockerignore` next, or for the owner to wave through
directly.

**D-11. Added BUILD GUARD 4.** Not asked for explicitly, but the Dockerfile
already has an established, explicitly-documented doctrine ("BUILD GUARD" /
"BUILD GUARD 2") for exactly this failure class — a file referenced at
runtime that a build step silently failed to produce — with two cited
production incidents (`tokenhealth.py`, `dbselect.py`). `[program:client]`
introduces a third runtime reference (`/opt/node/bin/node` and
`/opt/client/build/index.js`) that neither existing guard's regex covers
(both are scoped to `/opt/compactor/*.py`). Added a fourth guard rather than
leaving that class of failure uncovered for the one non-Python program in
the file. Named "BUILD GUARD 4" (not "3" — "BUILD GUARD 3" already exists,
at the `mistral_common` version-agreement check).

---

## 3. Verification performed (and what it does and doesn't prove)

All of the following ran on **this Windows machine**, which per
`FRONTEND_HANDOFF.md` §6 / this project's standing doctrine is **a
development convenience, not evidence**. None of it substitutes for a Linux/
Docker CI run; there is no CI wiring for `frontend/` yet (see D-9).

- `npm run check` (`svelte-kit sync && svelte-check`) — **0 errors, 0
  warnings, 172 files** — after every edit in this lane, most recently after
  the `app.html` fix in §1.3.
- `npm run build` — succeeds; adapter-node output at `frontend/build/`.
- **Confirmed the built server needs zero `node_modules` at runtime**: ran
  `node build/index.js` with `node_modules/` renamed away entirely; it still
  bound its port and served a correct `200` with full page content. This is
  why the Dockerfile hunk deletes `node_modules` after `npm run build` rather
  than pruning dev dependencies — nothing survives that the runtime needs.
- Cold-start timing measured directly (spawn timestamp to first successful
  HTTP response): **~0.27s** — informs D-7's `startsecs`, explicitly labeled
  in the stanza's comment as Windows-measured, not Linux evidence.
- **Interactive browser verification** (Claude Browser pane against the real
  built `adapter-node` server, not `vite dev`): dark/light/system theme
  switching, no-flash reload persistence, keyboard-only skip-link + focus
  ring, 375px mobile layout, zero console errors. Screenshots taken at each
  step; this is where the §1.3 bug was actually caught.
- `supervisord.conf` parsed with Python's `configparser` after the edit —
  16 sections (15 before + 1), the new `[program:client]` section's keys all
  present and structurally sane. **This is not a substitute for actually
  running supervisord** (Windows has no unix-socket/fork semantics to test
  against) — it only rules out gross INI-syntax breakage.
- The Dockerfile hunk's shell fragments (the `curl`/`tar` install, BUILD
  GUARD 4's `test ... || { ...; exit 1; }` construct) were syntax-checked
  against a local POSIX-ish `sh`, and the Node download URL was confirmed to
  resolve with a `HEAD` request (both `.tar.gz` and the pinned version
  exist). **The Dockerfile was NOT actually built** — no GPU host with the
  full CUDA base image and ~15+ minutes of vLLM/Postgres/Whisper layer builds
  was available in this session. This is the largest unverified surface; see
  §5.

---

## 4. Exact diffs

### `supervisord.conf`

```diff
--- a/supervisord.conf
+++ b/supervisord.conf
@@ -234,6 +234,72 @@
     ENABLE_OLLAMA_API="%(ENV_ENABLE_OLLAMA_API)s",
     ENABLE_OPENAI_API="%(ENV_ENABLE_OPENAI_API)s",
     WEBUI_AUTH="%(ENV_WEBUI_AUTH)s"
 
+# FRONTEND_PLAN.md F1/F2 — the SvelteKit replacement client (frontend/).
+# Co-located in this same pod container per D1 (§2.1): it reaches the
+# compactor on 127.0.0.1:8080, so it passes _require_localhost unchanged and
+# the split into its own container is a later deploy change, not a code one.
+# Built by adapter-node to a plain Node HTTP server (no framework runtime
+# beyond Node itself) — see the Dockerfile hunk that installs /opt/node and
+# builds /opt/client.
+#
+# priority=21: directly after openwebui(20), before stt(25) — this is
+# openwebui's eventual replacement, so it sits right beside it in the
+# startup order rather than off with the sidecars. Nothing here depends on
+# openwebui or is depended on by it; the ordering is for a human reading
+# `supervisorctl status` top to bottom, not a real dependency.
+#
+# HOST=0.0.0.0: like vllm/compactor/openwebui, this must be reachable from
+# outside the container over RunPod's port mapping, not just from
+# localhost. NOTE (handed back, not this lane's to fix): RunPod's pod
+# template must add 3001 to its exposed HTTP ports, the same way 3000/8080
+# are exposed today — this file and the Dockerfile's EXPOSE line cannot do
+# that themselves.
+[program:client]
+command=/opt/node/bin/node build/index.js
+directory=/opt/client
+autostart=true
+autorestart=true
+priority=21
+; Measured (this lane, Windows dev host, not Linux/Docker evidence — see
+; TESTING.md doctrine): adapter-node's built server reaches "Listening on
+; ..." in ~0.2s cold. 5s matches the other fast-starting, non-model-loading
+; programs here (postgres, backup, webuidb-sync) rather than the ~10s given
+; to compactor/openwebui/stt/tts, whose startup does real import-time work
+; (module loads, in the case of the Python services). Re-measure once this
+; runs on the actual container — a Node process competing with vLLM's model
+; load for CPU/disk at boot may be slower than this measurement.
+startsecs=5
+startretries=3
+; stopwaitsecs — of the ten [program:] blocks in this file, only postgres
+; had one before this (see its comment above); supervisord's 10s default
+; SIGTERM-then-SIGKILL races every graceful shutdown, this program
+; included. Unlike postgres this one does NOT need stopsignal=INT: adapter-
+; node's generated build/index.js (see node_modules/@sveltejs/adapter-node/
+; files/index.js, or this project's own build/index.js after `npm run
+; build`) installs an IDENTICAL handler for both SIGTERM and SIGINT —
+; httpServer.close() (stop accepting new connections, let in-flight ones
+; finish) followed by a hard closeAllConnections() after SHUTDOWN_TIMEOUT
+; seconds. SHUTDOWN_TIMEOUT is pinned to 30 below (adapter-node's own
+; default, made explicit here rather than left implicit) instead of left to
+; drift if a future adapter-node release changes its default without this
+; file changing to match. stopwaitsecs=35 is that 30s plus 5s headroom for
+; closeAllConnections() and process exit to actually complete — the same
+; "give it enough room to finish, not just enough room most days" reasoning
+; as postgres's 60. A streaming reply cut off by SIGKILL mid-response is
+; exactly the kind of silent, mid-stream failure this whole project exists
+; to stop causing.
+stopwaitsecs=35
+stdout_logfile=%(ENV_LOG_DIR)s/client.log
+stderr_logfile=%(ENV_LOG_DIR)s/client-error.log
+stdout_logfile_maxbytes=50MB
+stdout_logfile_backups=3
+stderr_logfile_maxbytes=50MB
+stderr_logfile_backups=3
+; Only what this program needs that nothing else already sets. Once L2
+; (frontend/src/lib/server/compactor/**) exists it will need COMPACTOR_URL
+; and, if a key is configured, an Authorization bearer value (D1, §2.1) —
+; add those here alongside PORT/HOST rather than inventing a second
+; environment= block, and prefer inheriting from a Dockerfile ENV default
+; the same way compactor's own stanza inherits everything but HF_HOME/
+; MODEL_REPO/VLLM_URL from the container environment.
+environment=
+    PORT="3001",
+    HOST="0.0.0.0",
+    NODE_ENV="production",
+    SHUTDOWN_TIMEOUT="30"
+
 # V3.2: Whisper speech-to-text service.
```

*(Full mechanical diff also saved this session at the scratchpad path used
during this run — regenerable at any time with `git diff -- supervisord.conf`
since nothing has been staged or committed.)*

### `Dockerfile`

```diff
--- a/Dockerfile
+++ b/Dockerfile
@@ -256,6 +256,74 @@
 # Note: OpenWebUI data lives at DATA_DIR=/data/openwebui (on the persistent
 # volume), created by entrypoint.sh at boot. No /app/data dirs needed —
 # that was the pre-single-volume layout (removed in V2.2 cleanup).
 
+# =============================================================================
+# Node.js — FRONTEND_PLAN.md F1/F2. The one non-Python runtime in this
+# image, for the SvelteKit client (frontend/) only; vLLM/compactor/STT/
+# TTS/OpenWebUI stay Python, each in its own venv per the pattern above.
+#
+# Installed as a pinned, direct download of the official prebuilt tarball
+# into /opt/node — not an apt/NodeSource install. [... full comment in file]
+# =============================================================================
+ARG NODE_VERSION=24.21.0
+# .tar.gz, not the smaller .tar.xz nodejs.org also publishes: `tar -z`
+# needs only gzip ... [full comment in file]
+RUN curl -fsSL -o /tmp/node.tar.gz \
+        "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.gz" && \
+    mkdir -p /opt/node && \
+    tar -xzf /tmp/node.tar.gz -C /opt/node --strip-components=1 && \
+    rm -f /tmp/node.tar.gz
+ENV PATH="/opt/node/bin:${PATH}"
+
+# =============================================================================
+# Client (frontend/) — SvelteKit + adapter-node (FRONTEND_PLAN.md F1/F2).
+# [... full comment in file — cache-layering rationale, network call-out,
+#  zero-runtime-node_modules verification]
+# =============================================================================
+COPY frontend/package.json frontend/package-lock.json frontend/.npmrc /opt/client/
+RUN cd /opt/client && npm ci
+COPY frontend/vite.config.ts frontend/tsconfig.json /opt/client/
+COPY frontend/src /opt/client/src
+COPY frontend/static /opt/client/static
+RUN cd /opt/client && \
+    npm run build && \
+    rm -rf node_modules package-lock.json src static .svelte-kit \
+        vite.config.ts tsconfig.json && \
+    npm cache clean --force && \
+    rm -rf /root/.npm /root/.cache /tmp/* /var/tmp/*
+
+# BUILD GUARD 4: the client's built server must exist exactly where
+# supervisord.conf's [program:client] stanza invokes it. [... full comment]
+RUN test -x /opt/node/bin/node || \
+      { echo "BUILD GUARD 4 FAILED: /opt/node/bin/node missing — supervisord.conf's [program:client] command= references this path directly."; exit 1; }; \
+    test -f /opt/client/build/index.js || \
+      { echo "BUILD GUARD 4 FAILED: /opt/client/build/index.js missing — the client build did not produce adapter-node's entrypoint where [program:client] expects it."; exit 1; }; \
+    echo "build guard: the client binary and its built server both exist"
+
 # Compactor sources copied AFTER the expensive install layer so editing
 # the Python files doesn't invalidate the vllm install cache. List each
@@ -549,10 +617,15 @@
 ENV LOG_DIR="/data/logs"
 
 # 3000 — OpenWebUI (user-facing)
+# 3001 — client (user-facing; FRONTEND_PLAN.md F1/F2 — the SvelteKit
+#        replacement for OpenWebUI, [program:client] in supervisord.conf).
+#        This EXPOSE is necessary but not sufficient: RunPod's pod template
+#        exposes ports independently of the image, and adding 3001 there is
+#        NOT done by this Dockerfile change — hand that back (see below).
 # 8080 — context-compactor (OpenAI-compatible, what OpenWebUI talks to)
 # 8000 — vLLM (internal; can also be exposed for direct API access)
 # 9000 — STT / Whisper (OpenAI audio API; OpenWebUI talks here for voice input)
 # 9001 — TTS / Piper (OpenAI audio API; OpenWebUI talks here for voice output)
-EXPOSE 8000 8080 3000 9000 9001
+EXPOSE 8000 8080 3000 3001 9000 9001
```

*(Comments are elided with `[...]` above for length; the actual file has the
full text — see `Dockerfile` lines ~261–335 and ~617–633. Full diff
regenerable with `git diff -- Dockerfile`.)*

**Note on the header comment block at the top of `supervisord.conf`
(lines 1–13):** it states "ten streams at 50MB x4 (vllm, compactor,
openwebui, stt, tts)" — five programs × 2 streams (stdout+stderr) = ten.
With `[program:client]` added at the same 50MB/3-backups tier, that becomes
six programs × 2 = **twelve** streams, so the header's arithmetic and its
"~2.2 GB across the seven programs and the event listener" total are now
both stale. This lane did **not** edit that header comment: it's outside the
single stanza this lane owns per hard rule 2. Flagged here for whoever
touches that comment next, or for the owner to fix directly.

---

## 5. Could not verify from this machine

1. **The Dockerfile was not actually built.** No CUDA/GPU host was available
   in this session, and the full build (vLLM + Postgres + three more Python
   venvs + Whisper model + TTS voice download) takes real time even without
   the new client hunk. The client-specific hunk was validated piecemeal
   (URL resolves, shell syntax checks out, the exact same `npm ci && npm run
   build` sequence was run for real on this host against `frontend/`) but
   never inside the actual image. **This is the single largest risk in this
   handoff** — recommend the owner or the next lane runs a real
   `docker build` before merging, specifically watching BUILD GUARD 4's
   output and the final image size delta.
2. **`supervisord.conf` was never actually run under supervisord.** Validated
   with Python's `configparser` only (structural INI sanity), not with the
   real supervisor package's config loader, which additionally validates
   things like log directory existence, numeric ranges, and the unix-socket
   plumbing — none of which is testable on Windows. Recommend a
   `supervisord -c supervisord.conf -t`-equivalent smoke check (or just
   booting the real container) before merge.
3. **RunPod's exposed-HTTP-ports template setting** — cannot be checked or
   changed from this machine at all; this lane has no RunPod access. Per the
   task brief, explicitly **not** attempted. **Someone with RunPod template
   access must add 3001 alongside the existing 3000/8080.**
4. **Actual container startup timing for `startsecs=5`.** Measured at ~0.27s
   cold on a Windows dev laptop with nothing else running. The real
   container starts Node while vLLM is loading a 24B model and Postgres is
   initializing — CPU/disk contention could plausibly push this past 5s.
   `startretries=3` gives some slack (three chances before FATAL), but the
   number itself is worth re-measuring against a real boot log once this
   runs on the target host.
5. **Interaction with `entrypoint.sh`'s boot-summary banner** (around line
   616, `echo "      - OpenWebUI        on port ..."`) — it does not mention
   the client or port 3001. Not fixed here: `entrypoint.sh` is outside this
   lane's three owned paths. Flagged as a cosmetic gap for whoever next
   touches that file.
6. **npm audit: 3 low-severity advisories**, all the same transitive
   `cookie` package pulled in by `@sveltejs/kit` (GHSA-pxg6-pf52-xh8x, "out
   of bounds characters" in cookie name/path/domain parsing). `npm audit`'s
   suggested fix is a major downgrade of `@sveltejs/kit` to `0.0.30`, which
   is clearly wrong for a transitive pin — this will resolve on a normal
   `@sveltejs/kit` patch bump. Not a runtime risk today (nothing in this
   scaffold sets cookies), but worth a `npm update` pass once L2/L3 add
   session handling (F16).

---

## 6. What I believe is wrong (or at least under-specified) in `FRONTEND_PLAN.md`

Asked to be blunt about this, and told two errors were already caught by
review — a third would be normal. Here is what this lane found, scoped to
what F1/F2 work actually touches:

**6.1 — §4 and §5 disagree about who owns `routes/**` and
`components/**`.** §5's delegation table assigns `frontend/src/routes/**`
and `frontend/src/lib/components/**` to **L3**, startable only "after the
interface freeze." But §4's own F1 row — the row assigned to **L0** — reads
*"A minimal app shell only: layout, theme switching, an empty
conversation-list rail and an empty message pane with a placeholder."* That
sentence cannot be built without creating files under exactly the two paths
§5 reserves for L3. There is no interface freeze yet (it depends on F3's
store, which does not exist), so a strict reading of §5 would mean F1's own
shell requirement is unbuildable on day one.

This lane resolved it pragmatically, per the direct task brief (which
restates F1's shell requirement explicitly and is the actual authority
here): built the shell under `frontend/src/lib/components/shell/` — a
namespace distinct from `components/system/**` (§5: L4) and
`components/memory/**` (§5: L6), so there's no path collision with either of
those — plus the two trivial route files SvelteKit requires
(`+layout.svelte`, `+page.svelte`). **L3 should expect to extend, not
recreate, these files** when it picks up branch navigation, the composer,
etc. If L3's actual plan is to replace this shell wholesale rather than
build on it, that's a fine outcome too, but §4/§5 should be reconciled
explicitly so the next lane isn't guessing which authority governs.

**6.2 — "Doulos tokens copied into the repo" presumes access this thread
doesn't have.** Covered in decision D-1 above; repeating here because it's a
plan-level (not just implementation-level) gap. §2.3 and §14 both phrase the
Doulos decision as "copy the tokens, skip the packages" — correct as an
architecture call, but neither document acknowledges that *copying* requires
someone with access to the private `@telos-llc/doulos` package to actually
extract the real values first. No such access existed in this thread. The
plan should either (a) note that the real Doulos values need to be supplied
by someone with registry access and swapped in later, or (b) explicitly bless
a from-scratch token set as this lane did. Right now a future reader could
mistakenly assume `tokens.css`'s values ARE Doulos's real scale.

**6.3 — Minor: the supervisord header-comment staleness noted in §4 above**
is really a plan-adjacent finding (F2's own row correctly predicts this
exact problem — "of ten `[program:]` blocks only postgres has one" — but
doesn't flag that F2 landing makes the file's own top-of-file arithmetic
comment go stale). Not a plan *error* exactly, more a thing F2's own
description could have called out given it clearly anticipated the file's
state.

Nothing else in F1/F2's own text was found to be wrong against the current
`feat/frontend` HEAD — the port (3001), priority (21, beside openwebui's 20),
and the `stopwaitsecs` ask (the only prior one is postgres,
`supervisord.conf:112-134` — actually landed at 84–144 after other edits in
this branch's history, but the postgres block itself and its content are
exactly as described) all checked out.

---

## 7. Summary for the next lane

- `frontend/` builds, type-checks clean, and renders correctly (verified
  interactively, light/dark/system, keyboard nav, mobile breakpoint) — on
  Windows. **Not yet built or run in Docker/Linux.**
- Every design decision left open by the plan is in §2 above with its
  reversibility argument. None of them touch architecture L1/L2/L3 will
  depend on (the store, transport, or send-set) — this lane built
  presentation only.
- `[program:client]` is ready for `supervisorctl status` once the image
  builds; `stopwaitsecs`/`startsecs` are evidence-based but Windows-measured.
- The Dockerfile hunk is untested end-to-end (§5.1) — **build it for real
  before merging.**
- RunPod's exposed-ports template still needs a human with RunPod access to
  add 3001 (§5.3) — this lane could not do it and did not try.

---

## Reviewer addendum — Linux/Docker evidence (2026-09-09)

The lane's own verification was Windows-side, which this project does not count
as evidence (`FRONTEND_HANDOFF.md` §6, `COMMANDS.md`). The client hunk was
therefore rebuilt on Linux, in Docker, from the Dockerfile's own lines —
`debian:bookworm-slim`, the pinned Node tarball, `npm ci`, `npm run build`, the
`node_modules` strip, and BUILD GUARD 4 verbatim.

| Check | Result |
|---|---|
| `node-v24.21.0-linux-x64.tar.gz` published | **yes** — HTTP 200, and v24.21.0 is the current LTS (Krypton). *Note: Docker Hub has no `node:24.21.0-slim` tag, so a reviewer reaching for one will be misled — the tarball pin is correct.* |
| `npm ci` against the committed lockfile | clean |
| `npm run build` on Linux | clean |
| `node_modules` deleted, then guard | **BUILD GUARD 4 passed**; `node_modules` confirmed absent |
| Server boots with no `node_modules` | **yes** — `Listening on http://0.0.0.0:3001` |
| `GET /` | **200** |
| Pre-paint theme | `data-theme`, `prefers-color-scheme`, `localStorage` all present in the served HTML |
| External asset hosts in served HTML | **none** — the only absolute URL is the `www.w3.org` SVG namespace, which is not a fetch |
| SIGTERM graceful shutdown | **exit 0 in 2 s**, so `stopwaitsecs=35` has ample headroom |

The lane's central claim — that a SvelteKit `adapter-node` build is
self-contained and `node_modules` can be deleted — holds, and holds for a
structural reason worth recording: `package.json` has **zero** `dependencies`;
everything is a `devDependency`. `build/index.js` imports only `node:*` builtins
and relative paths. (A naive grep finds `polka`/`sirv`/`@sveltejs/kit` inside
`build/server/chunks/*`, but those are bundled string content, not import
statements — check before concluding otherwise.)

**Still not verified, and still blocking a merge to master:** the *full* image
has never been built. This test used a Debian base, not the project's CUDA base,
and it exercised only the client hunk. The interaction with the rest of the image
— layer ordering, the shared `EXPOSE`, and whether `curl`/`tar` behave the same
on that base — is unproven.
