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
# ever "set" / "unset" / "disabled". The shape-validation failure log line
# below is likewise restricted to a fixed, closed set of reason words — see
# validate_tunnel_token_shape's own comment — never a length, prefix,
# suffix, character class or any other substring of the actual value.
#
# RC-REVIEW FIX (v3.1.9.9 rc-fix, MEDIUM): a bad-but-well-formed token used
# to make cloudflared spawn, get rejected at Cloudflare's edge in well under
# a second, and exit — over and over, as fast as supervisord could respawn
# it (measured: 61 spawns in 40s, ~10% of a core plus heavy log churn on
# MooseFS, while `supervisorctl status` still read RUNNING because
# startsecs=0 counts every one of those instant spawns as a successful
# start). Two independent layers fix this, both BEFORE `exec cloudflared`,
# so startsecs=0 keeps meaning what it already means for the disabled path:
#   (1) validate_tunnel_token_shape — a cheap, local, no-network check that
#       catches garbage before we ever spawn the real binary. An invalid
#       shape sleeps 300s and exits 1 itself: one spawn per five minutes,
#       no loop.
#   (2) throttle_if_hot — a small on-disk stamp (last spawn time + current
#       backoff) that makes any repeat spawn within
#       CLOUDFLARED_RUN_THROTTLE_WINDOW_S of the last one sleep first, for
#       an exponentially growing (and capped) delay — this is what actually
#       protects against a shape-VALID but Cloudflare-rejected token, which
#       (1) cannot see coming since it has no way to ask Cloudflare's edge
#       anything.
# Neither layer touches supervisord.conf: startsecs=0/autorestart=unexpected/
# exitcodes=0 stay exactly as documented there (a disabled/no-token exit
# still needs to look instant and EXPECTED).

set -u

log() {
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [cloudflared-run] $*"
}

# --- restart throttle state -------------------------------------------
# All three knobs are env-overridable purely so compactor/
# test_cloudflared_run.py can exercise real timing on a short clock instead
# of waiting on real 30-300s sleeps; production never sets them, so the
# documented 60/30/300 defaults are what actually ships.
THROTTLE_WINDOW_S="${CLOUDFLARED_RUN_THROTTLE_WINDOW_S:-60}"
BACKOFF_BASE_S="${CLOUDFLARED_RUN_BACKOFF_BASE_S:-30}"
BACKOFF_CAP_S="${CLOUDFLARED_RUN_BACKOFF_CAP_S:-300}"
# Same reasoning as the three knobs above: overridable purely for the test
# suite (a 300s real sleep per invalid-shape case would make the unit tests
# unusable); production sets none of these, so the documented defaults are
# what ships.
INVALID_TOKEN_SLEEP_S="${CLOUDFLARED_RUN_INVALID_TOKEN_SLEEP_S:-300}"
STATE_DIR="${CLOUDFLARED_RUN_STATE_DIR:-/run}"
if [ ! -d "$STATE_DIR" ] || [ ! -w "$STATE_DIR" ]; then
    STATE_DIR=/tmp
fi
STAMP_FILE="$STATE_DIR/cloudflared-run.stamp"

throttle_if_hot() {
    # Reads "<last_start_epoch> <next_backoff_s>" from STAMP_FILE (missing
    # or garbage counts as "never started, base backoff"). If the previous
    # start was under THROTTLE_WINDOW_S ago, sleeps for the recorded
    # backoff (doubling it, capped, for whoever reads the stamp next);
    # otherwise this spawn isn't part of a hot loop, so the backoff resets
    # to the base. Always rewrites the stamp with the CURRENT time (taken
    # AFTER any sleep, so it reflects when this attempt is actually about
    # to exec) before returning.
    local now last_start backoff elapsed next
    now=$(date +%s)
    last_start=0
    backoff=$BACKOFF_BASE_S
    if [ -f "$STAMP_FILE" ]; then
        read -r last_start backoff < "$STAMP_FILE" 2>/dev/null || true
    fi
    case "$last_start" in (*[!0-9]*|'') last_start=0 ;; esac
    case "$backoff" in (*[!0-9]*|'') backoff=$BACKOFF_BASE_S ;; esac

    if [ "$last_start" -gt 0 ]; then
        elapsed=$(( now - last_start ))
    else
        elapsed=$THROTTLE_WINDOW_S  # never started before -> never "hot"
    fi

    if [ "$elapsed" -lt "$THROTTLE_WINDOW_S" ]; then
        log "restarted ${elapsed}s after the last start (< ${THROTTLE_WINDOW_S}s window); backing off ${backoff}s before starting cloudflared"
        sleep "$backoff"
        next=$(( backoff * 2 ))
        [ "$next" -gt "$BACKOFF_CAP_S" ] && next=$BACKOFF_CAP_S
    else
        next=$BACKOFF_BASE_S
    fi

    now=$(date +%s)
    { printf '%s %s\n' "$now" "$next" > "$STAMP_FILE"; } 2>/dev/null || true
}

# --- token shape validation --------------------------------------------
# hostile-review round (rc-review), MEDIUM fix, second half: catches
# obviously-wrong tokens (empty, plain garbage, or the whole
# `cloudflared service install eyJ...` line pasted instead of just the
# token) locally and for free, with no network round-trip, before we ever
# spawn the real binary. A tunnel token is base64 of a small JSON object
# with keys "a" (account), "t" (tunnel id) and "s" (secret) — this checks
# exactly that shape, nothing about whether Cloudflare would actually
# accept it (only Cloudflare's own edge can tell us that; see
# throttle_if_hot for the case where the shape is fine but the value
# itself is rejected).
_TOKEN_INVALID_REASON=""

_looks_like_install_line() {
    case "$1" in
        *cloudflared*service*install*) return 0 ;;
        *tunnel*run*--token*) return 0 ;;
        *) return 1 ;;
    esac
}

_extract_eyj() {
    # Pulls the first eyJ...-shaped run (base64/base64url JWT-style header)
    # out of a longer pasted line. Never echoes anything BUT that
    # candidate substring — the surrounding command text is discarded.
    printf '%s' "$1" | grep -oE 'eyJ[A-Za-z0-9+/_=-]+' | head -n1
}

validate_tunnel_token_shape() {
    # $1 = the raw candidate straight from CLOUDFLARE_TUNNEL_TOKEN. Must be
    # called DIRECTLY (never inside a `$(...)` command substitution) —
    # both of its results are communicated through the globals below, not
    # stdout, precisely so they survive: a `$(...)` subshell would silently
    # discard any variable this function sets, which is what actually
    # bit the first draft of this function (caught by a probe with a
    # deliberately whitespace-padded decoded JSON body).
    #
    # On a valid shape: sets _SHAPE_TOKEN to the CLEANED token
    # (quotes/whitespace stripped, padding fixed, and — if a whole install
    # line was pasted — only the extracted eyJ... part) and returns 0.
    # On an invalid shape: sets _TOKEN_INVALID_REASON to exactly one of a
    # fixed, closed set of words:
    #   "not base64" | "not JSON" | "missing keys" | "looks like a pasted install command"
    # and returns 1. This function and its log-line caller must NEVER
    # include the token's length, a prefix/suffix, a character class, or
    # any other substring of the value in a log line or in _SHAPE_TOKEN on
    # the failure path — only ever one of those four fixed words.
    local raw="$1" cleaned candidate decoded decoded_trimmed decode_input k pad i padded
    _SHAPE_TOKEN=""
    _TOKEN_INVALID_REASON=""

    cleaned=$(printf '%s' "$raw" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
    case "$cleaned" in
        \"*\") cleaned=${cleaned#\"}; cleaned=${cleaned%\"} ;;
        \'*\') cleaned=${cleaned#\'}; cleaned=${cleaned%\'} ;;
    esac
    cleaned=$(printf '%s' "$cleaned" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')

    if _looks_like_install_line "$cleaned"; then
        candidate=$(_extract_eyj "$cleaned")
        if [ -z "$candidate" ]; then
            _TOKEN_INVALID_REASON="looks like a pasted install command"
            return 1
        fi
    else
        candidate="$cleaned"
    fi

    # Tolerate whitespace pasted INSIDE the value (line-wrapped paste),
    # then tolerate missing/extra padding.
    candidate=$(printf '%s' "$candidate" | tr -d '[:space:]')
    if [ -z "$candidate" ]; then
        _TOKEN_INVALID_REASON="not base64"
        return 1
    fi
    case "$candidate" in
        *[!A-Za-z0-9+/=_-]*)
            _TOKEN_INVALID_REASON="not base64"
            return 1
            ;;
    esac
    # Decode-only normalization (base64url -> base64, padding re-derived)
    # happens on a SEPARATE copy — `candidate` itself (what ends up in
    # _SHAPE_TOKEN, and so in TUNNEL_TOKEN) must stay exactly what the
    # operator gave us, minus only the whitespace/quote/install-line
    # cleanup already applied above. Rewriting -_/+/ or padding into the
    # exported value would risk handing cloudflared a token that no longer
    # matches what was actually issued.
    decode_input=$(printf '%s' "$candidate" | tr -- '-_' '+/')
    decode_input=${decode_input%%=*}
    pad=$(( (4 - ${#decode_input} % 4) % 4 ))
    padded=$decode_input
    i=0
    while [ "$i" -lt "$pad" ]; do padded="${padded}="; i=$((i + 1)); done

    decoded=$(printf '%s' "$padded" | base64 -d 2>/dev/null)
    if [ -z "$decoded" ]; then
        _TOKEN_INVALID_REASON="not base64"
        return 1
    fi
    # A tunnel token's decoded JSON may legitimately carry leading/
    # trailing whitespace around the object itself (still valid JSON) —
    # trim before checking for the opening/closing braces, same tolerance
    # spirit as the outer whitespace-around-the-token handling above.
    decoded_trimmed=$(printf '%s' "$decoded" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
    case "$decoded_trimmed" in
        \{*\}) : ;;
        *)
            _TOKEN_INVALID_REASON="not JSON"
            return 1
            ;;
    esac
    for k in a t s; do
        case "$decoded" in
            *\"$k\"*) : ;;
            *)
                _TOKEN_INVALID_REASON="missing keys"
                return 1
                ;;
        esac
    done

    _SHAPE_TOKEN="$candidate"
    return 0
}

if [ -z "${CLOUDFLARE_TUNNEL_TOKEN:-}" ]; then
    log "no CLOUDFLARE_TUNNEL_TOKEN; tunnel disabled"
    exit 0
fi

# Read and immediately clear CLOUDFLARE_TUNNEL_TOKEN from this shell's own
# named variable as soon as we're done with it, same doctrine as the
# enabled path below: one copy of the value alive at a time, never longer
# than needed.
_RAW_TOKEN="${CLOUDFLARE_TUNNEL_TOKEN}"
unset CLOUDFLARE_TUNNEL_TOKEN

if ! validate_tunnel_token_shape "${_RAW_TOKEN}"; then
    log "CLOUDFLARE_TUNNEL_TOKEN does not look like a tunnel token (${_TOKEN_INVALID_REASON}); not starting"
    unset _RAW_TOKEN _TOKEN_INVALID_REASON
    # 300s sleep (INVALID_TOKEN_SLEEP_S) before exiting 1:
    # exitcodes=0/autorestart=unexpected (supervisord.conf) mean a nonzero
    # exit IS "unexpected" and DOES get respawned — this sleep is what
    # keeps that respawn to at most once per five minutes instead of a
    # tight loop, without touching startsecs=0 or the disabled (no-token)
    # path above.
    sleep "$INVALID_TOKEN_SLEEP_S"
    exit 1
fi
_CLEAN_TOKEN="$_SHAPE_TOKEN"
unset _RAW_TOKEN _SHAPE_TOKEN

log "CLOUDFLARE_TUNNEL_TOKEN is set; starting cloudflared"

# Shape is fine; this could still be a well-formed token Cloudflare itself
# rejects in well under a second (wrong account/tunnel/secret) — that
# failure mode can only be caught after the fact, by how fast we're being
# asked to restart, which is exactly what this checks/waits on.
throttle_if_hot

# Move the token from CLOUDFLARE_TUNNEL_TOKEN's cleaned form to
# TUNNEL_TOKEN (cloudflared's own expected name); nothing keeps the raw
# CLOUDFLARE_TUNNEL_TOKEN value (already unset above) or a second lingering
# copy of the cleaned one (_CLEAN_TOKEN is unset right after use) sitting
# in this process's environment for anything else that introspects it
# (e.g. `cat /proc/<pid>/environ`) any longer than exec requires.
export TUNNEL_TOKEN="${_CLEAN_TOKEN}"
unset _CLEAN_TOKEN

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
