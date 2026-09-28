#!/bin/bash
# docker/cloudflared-run.sh — baked to /opt/cloudflared-run.sh (Dockerfile.v3199).
# The command supervisord's [program:cloudflared] runs. v3.1.9.9, task: give
# her OpenWebUI a stable https://ai.revelationsaints.com URL via a Cloudflare
# Tunnel, instead of the RunPod proxy URL.
#
# TOKEN SOURCE. CLOUDFLARE_TUNNEL_TOKEN is a RunPod secret/env var on the pod
# template (see RUNBOOK_CLOUDFLARE_TUNNEL.md). entrypoint.sh execs supervisord
# as PID 1 (`exec /usr/bin/supervisord ...`), and supervisord's own
# [program:cloudflared] stanza sets no `environment=` line for this var, so
# this script — a supervisord child — inherits it straight from the
# container's environment, the SAME way [program:compactor] inherits every
# COMPACTOR_*/RAG/FASTEMBED var (supervisord.conf's own comment on that
# stanza: "supervisord children inherit the parent env"). No /proc/1/environ
# read is needed HERE — that pattern (ops/zlops/podenv.py, scripts/
# _webui_live_path.py) exists for tooling that runs OUTSIDE the supervisord
# tree (an SSH shell, `zl status`) and so does not automatically see what
# entrypoint.sh exported; this script IS in that tree.
#
# DISABLED PATH (no token) — exit 0, not `sleep infinity`. Chosen over the
# alternative because of what `supervisorctl status` needs to mean:
#   - autorestart=unexpected + exitcodes=0 (supervisord.conf) means an exit 0
#     is EXPECTED and supervisord does NOT restart it — the program settles
#     into EXITED and stays there. `supervisorctl status` then reads
#     "cloudflared EXITED (expected)", which an operator (or `zl status`,
#     see the ops asks list) can read directly as "tunnel disabled by
#     config", with no log dive.
#   - `exec sleep infinity` would instead sit at RUNNING forever — green,
#     healthy-looking, and lying: nothing is actually listening on the
#     tunnel's behalf, exactly the "green but broken" trap this repo's own
#     mistakes log warns about elsewhere (bg/mt in health.py: "a layer we
#     cannot see must not be reported as healthy"). RUNNING is the wrong
#     word for "intentionally does nothing".
#   - It also means a genuinely missing token can never look like a crash
#     loop: there is exactly one clean exit, no BACKOFF/FATAL noise to sift
#     through.
#
# ENABLED PATH — the token goes to cloudflared via environment ONLY, never on
# argv. `ps`, `ps aux` (visible to anyone in the container's PID namespace)
# and supervisord's own child-process logging both show argv; neither shows
# a child's environment by default. `TUNNEL_TOKEN` is cloudflared's own
# documented env var for `tunnel run` (equivalent to --token on the command
# line, which this script deliberately never uses).
#
# LOGGING — info level, and this script never echoes the token itself, only
# ever "set" / "unset" / "disabled".

set -u

log() {
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [cloudflared-run] $*"
}

if [ -z "${CLOUDFLARE_TUNNEL_TOKEN:-}" ]; then
    log "no CLOUDFLARE_TUNNEL_TOKEN; tunnel disabled"
    exit 0
fi

log "CLOUDFLARE_TUNNEL_TOKEN is set; starting cloudflared"

# Move the token from CLOUDFLARE_TUNNEL_TOKEN (this repo's/RunPod's name for
# it, chosen to be unambiguous among every other env var on the pod) to
# TUNNEL_TOKEN (cloudflared's own expected name), then remove the former so
# it does not sit twice in this process's environment for anything else that
# introspects it (e.g. `cat /proc/<pid>/environ`) — one copy, under the name
# the thing that actually reads it expects.
export TUNNEL_TOKEN="${CLOUDFLARE_TUNNEL_TOKEN}"
unset CLOUDFLARE_TUNNEL_TOKEN

# --no-autoupdate: this image pins cloudflared by version+sha256 at build
# time (Dockerfile.v3199); an in-place self-update would silently diverge
# from that pin and from what was tested.
# --metrics 127.0.0.1:8082: loopback-only (never exposed past the tunnel or
# the pod's network), so /health/full (task 7) can optionally probe
# cloudflared's own /ready without opening anything new to the outside.
# `exec`: cloudflared replaces this script as supervisord's actual child, so
# SIGTERM/SIGKILL from supervisord (stopwaitsecs) reaches cloudflared
# directly rather than a shell that has to relay it.
exec cloudflared tunnel --no-autoupdate --metrics 127.0.0.1:8082 run
