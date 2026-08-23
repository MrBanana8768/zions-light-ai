#!/bin/bash
# =============================================================================
# clean-models.sh — operator tool for reclaiming space on the /data volume.
#
# When you swap MODEL_REPO (e.g. anthracite-org/magnum-v4-12b -> a Cydonia-24B),
# the OLD weights stay cached on the Network Volume and keep eating space
# (models are 10-50 GB each). This script lists what's cached, marks the ACTIVE
# model, and lets you delete the stale ones — WITHOUT ever touching the model
# that's currently in use.
#
# HuggingFace stores weights in hub format under $HF_HOME/hub:
#   $HF_HOME/hub/models--<org>--<name>/snapshots/<hash>/...  (+ blobs/, refs/)
# The ACTIVE model is derived from $MODEL_REPO ("org/name" -> "models--org--name").
#
# Safe by default: with no destructive flag it only LISTS (dry run). Deletion
# needs an explicit --yes or an interactive y/N confirmation, and the active
# model is refused even if you name it directly.
#
# Runs inside the pod:  docker exec <container> /opt/clean-models.sh
#                or:    /opt/clean-models.sh --prune-others
# =============================================================================
set -euo pipefail

# --- Config (env with sane defaults, matching entrypoint.sh) -----------------
HF_HOME="${HF_HOME:-/data/models}"                 # cache root
MODEL_REPO="${MODEL_REPO:-}"                        # active model (org/name)
COMPILE_CACHE="${VLLM_COMPILE_CACHE:-/data/vllm-compile-cache}"

HUB="${HF_HOME%/}/hub"                              # where models--* dirs live

# --- Runtime state -----------------------------------------------------------
MODE="list"            # list | delete | prune
DELETE_ARG=""          # value for --delete
DO_COMPILE=0           # also clear the torch.compile cache?
ASSUME_YES=0           # --yes skips the confirmation prompt

usage() {
    cat <<'EOF'
clean-models.sh — reclaim volume space by removing stale HuggingFace model caches.

USAGE:
  clean-models.sh [--list]                 List cached models, mark the ACTIVE one (default).
  clean-models.sh --delete <repo-or-dir>   Remove ONE cached model.
                                             Accepts "org/name" or "models--org--name".
  clean-models.sh --prune-others           Remove ALL cached models EXCEPT the active one.
  clean-models.sh --compile-cache          Also clear the vLLM torch.compile cache
                                             (regenerable perf cache; first boot after is slower).
                                             Combine with a mode, or use alone.
  clean-models.sh --yes                     Skip the interactive confirmation (for scripts).
  clean-models.sh --help                   Show this help.

SAFETY:
  * Default is a DRY RUN — nothing is deleted without --delete/--prune-others/--compile-cache.
  * The ACTIVE model (from $MODEL_REPO) is ALWAYS protected, even if you name it.
  * Deletion needs --yes OR a "y" at the confirmation prompt; it prints the space freed.
  * Only real directories directly under $HF_HOME/hub named "models--*" are ever removed.

ENV:
  HF_HOME               Cache root (default: /data/models). Models live under $HF_HOME/hub.
  MODEL_REPO            Active model, "org/name" (protected). Required for --prune-others.
  VLLM_COMPILE_CACHE    torch.compile cache dir (default: /data/vllm-compile-cache).

EXAMPLES:
  clean-models.sh                                       # see what's cached
  clean-models.sh --delete anthracite-org/magnum-v4-12b # drop one old model
  clean-models.sh --prune-others --yes                  # keep only the active model
  clean-models.sh --compile-cache                       # clear the compile cache only
EOF
}

# --- Arg parsing -------------------------------------------------------------
while [ "$#" -gt 0 ]; do
    case "$1" in
        --list)          MODE="list" ;;
        --prune-others)  MODE="prune" ;;
        --delete)
            if [ "$#" -lt 2 ] || [ -z "${2:-}" ]; then
                echo "ERROR: --delete needs a model argument (org/name or models--org--name)." >&2
                exit 2
            fi
            MODE="delete"; DELETE_ARG="$2"; shift ;;
        --delete=*)      MODE="delete"; DELETE_ARG="${1#--delete=}" ;;
        --compile-cache) DO_COMPILE=1 ;;
        --yes|-y)        ASSUME_YES=1 ;;
        --help|-h)       usage; exit 0 ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            echo "Try: clean-models.sh --help" >&2
            exit 2 ;;
    esac
    shift
done

# --- Helpers -----------------------------------------------------------------

# "org/name" (or "models--org--name") -> hub dir name "models--org--name".
hubdir_for() {
    local x="$1"
    case "$x" in
        models--*) printf '%s' "$x" ;;
        *)         printf 'models--%s' "${x//\//--}" ;;
    esac
}

# Bytes -> human readable (avoids depending on numfmt).
human() {
    awk -v b="${1:-0}" 'BEGIN{
        split("B KB MB GB TB PB", u, " ");
        i = 1;
        while (b >= 1024 && i < 6) { b /= 1024; i++ }
        if (i == 1) printf "%d %s", b, u[i]; else printf "%.1f %s", b, u[i];
    }'
}

# Guard: the target must be a real directory directly under $HUB and named
# "models--*". Refuses "/", the cache root, symlink escapes, and anything else.
# Returns 0 if safe to rm -rf, non-zero otherwise.
assert_removable() {
    local target="$1" base parent hub_real
    [ -n "$target" ] || return 1
    [ -d "$target" ] || return 1

    # Canonicalize both sides when we can, to defeat .. and symlink tricks.
    if command -v realpath >/dev/null 2>&1; then
        target="$(realpath "$target" 2>/dev/null)" || return 1
        hub_real="$(realpath "$HUB" 2>/dev/null)"   || return 1
    else
        hub_real="$HUB"
    fi

    base="$(basename "$target")"
    parent="$(dirname "$target")"

    case "$base" in models--*) : ;; *) return 1 ;; esac  # right name
    [ "$parent" = "$hub_real" ] || return 1              # directly under hub
    [ "$target" != "/" ]         || return 1             # never /
    [ "$target" != "$hub_real" ] || return 1             # never the hub root
    [ "$target" != "${HF_HOME%/}" ] || return 1          # never the cache root
    return 0
}

# du -sb wrapper -> bytes only (0 on failure).
dir_bytes() {
    du -sb "$1" 2>/dev/null | awk '{print $1}' || echo 0
}

ACTIVE_HUBDIR=""
[ -n "$MODEL_REPO" ] && ACTIVE_HUBDIR="$(hubdir_for "$MODEL_REPO")"

# --- Banner ------------------------------------------------------------------
echo "=============================================="
echo "  Zion's Light AI - model cache cleanup"
echo "  HF_HOME:     ${HF_HOME}"
echo "  Hub cache:   ${HUB}"
if [ -n "$MODEL_REPO" ]; then
    echo "  Active:      ${MODEL_REPO}  (${ACTIVE_HUBDIR})"
else
    echo "  Active:      <MODEL_REPO not set — nothing is protected>"
fi
echo "=============================================="

# --- Sanity: cache root present? --------------------------------------------
if [ ! -d "$HUB" ]; then
    echo ""
    echo "No HuggingFace hub cache found at: ${HUB}"
    if [ ! -d "${HF_HOME%/}" ]; then
        echo "(HF_HOME '${HF_HOME}' does not exist — is the /data volume mounted?)"
    else
        echo "(nothing has been downloaded yet — cache is empty)"
    fi
    # Listing an empty cache is a no-op success; deleting from it is an error.
    if [ "$MODE" != "list" ] || [ "$DO_COMPILE" -eq 1 ]; then
        if [ "$DO_COMPILE" -eq 0 ]; then
            echo "Nothing to delete." >&2
            exit 1
        fi
    else
        exit 0
    fi
fi

# --- Inventory (always shown) ------------------------------------------------
declare -a ALL_DIRS=()
if [ -d "$HUB" ]; then
    for d in "$HUB"/models--*; do
        [ -d "$d" ] || continue
        ALL_DIRS+=("$d")
    done
fi

echo ""
echo "Cached models under ${HUB}:"
if [ "${#ALL_DIRS[@]}" -eq 0 ]; then
    echo "  (none)"
else
    for d in "${ALL_DIRS[@]}"; do
        name="$(basename "$d")"
        size="$(du -sh "$d" 2>/dev/null | cut -f1)"
        if [ -n "$ACTIVE_HUBDIR" ] && [ "$name" = "$ACTIVE_HUBDIR" ]; then
            printf '  %-9s %s   <== ACTIVE (protected)\n' "$size" "$name"
        else
            printf '  %-9s %s\n' "$size" "$name"
        fi
    done
fi
echo ""

# --- Build the removal plan --------------------------------------------------
declare -a TARGETS=()

case "$MODE" in
    delete)
        want="$(hubdir_for "$DELETE_ARG")"
        target="${HUB}/${want}"
        if [ -n "$ACTIVE_HUBDIR" ] && [ "$want" = "$ACTIVE_HUBDIR" ]; then
            echo "REFUSED: '${DELETE_ARG}' is the ACTIVE model (${want}) and is protected." >&2
            echo "         Swap MODEL_REPO to something else first if you really mean to remove it." >&2
            exit 1
        fi
        if [ ! -d "$target" ]; then
            echo "ERROR: no cached model named '${want}' under ${HUB}." >&2
            echo "       Run with no arguments to see what's actually cached." >&2
            exit 1
        fi
        if ! assert_removable "$target"; then
            echo "ERROR: refusing to remove '${target}' — failed the safety check." >&2
            exit 1
        fi
        TARGETS+=("$target")
        ;;
    prune)
        if [ -z "$ACTIVE_HUBDIR" ]; then
            echo "REFUSED: --prune-others needs MODEL_REPO set so it knows what to KEEP." >&2
            echo "         Without it, every cached model would be a deletion target." >&2
            exit 1
        fi
        for d in "${ALL_DIRS[@]:-}"; do
            [ -n "$d" ] || continue
            [ -d "$d" ] || continue
            [ "$(basename "$d")" = "$ACTIVE_HUBDIR" ] && continue   # keep active
            assert_removable "$d" && TARGETS+=("$d")
        done
        ;;
    list)
        : # nothing to remove from the model cache
        ;;
esac

# --- Nothing destructive requested -> we're done (pure dry run) --------------
if [ "${#TARGETS[@]}" -eq 0 ] && [ "$DO_COMPILE" -eq 0 ]; then
    if [ "$MODE" = "prune" ]; then
        echo "Nothing to prune — the only cached model is the active one."
    fi
    echo "Dry run: nothing was deleted. Re-run with --delete/--prune-others/--compile-cache to reclaim space."
    exit 0
fi

# --- Present the plan + compute space to free --------------------------------
total_bytes=0
echo "The following will be REMOVED:"
for t in "${TARGETS[@]:-}"; do
    [ -n "$t" ] || continue
    b="$(dir_bytes "$t")"
    total_bytes=$(( total_bytes + b ))
    printf '  - %s   (%s)\n' "$t" "$(human "$b")"
done

compile_bytes=0
if [ "$DO_COMPILE" -eq 1 ]; then
    if [ -d "$COMPILE_CACHE" ]; then
        # Guard the compile-cache path too: must live under /data and not be a root.
        case "$COMPILE_CACHE" in
            /data/?*)
                compile_bytes="$(dir_bytes "$COMPILE_CACHE")"
                total_bytes=$(( total_bytes + compile_bytes ))
                printf '  - %s   (%s)  [torch.compile cache — regenerable; first boot after is slower]\n' \
                    "$COMPILE_CACHE" "$(human "$compile_bytes")"
                ;;
            *)
                echo "  ! Skipping compile cache: '${COMPILE_CACHE}' is not under /data — refusing." >&2
                DO_COMPILE=0
                ;;
        esac
    else
        echo "  (compile cache ${COMPILE_CACHE} does not exist — nothing to clear)"
        DO_COMPILE=0
    fi
fi

# Re-check: maybe the compile cache was skipped and there's nothing left.
if [ "${#TARGETS[@]}" -eq 0 ] && [ "$DO_COMPILE" -eq 0 ]; then
    echo ""
    echo "Nothing to remove. Dry run only — nothing was deleted."
    exit 0
fi

echo ""
echo "Space to reclaim: $(human "$total_bytes")"
echo ""

# --- Confirm -----------------------------------------------------------------
if [ "$ASSUME_YES" -ne 1 ]; then
    printf 'Proceed with deletion? [y/N] '
    reply=""
    read -r reply || reply=""     # EOF / non-interactive -> empty -> abort
    case "$reply" in
        y|Y|yes|YES) : ;;
        *)
            echo "Aborted. Nothing was deleted."
            exit 0 ;;
    esac
fi

# --- Execute -----------------------------------------------------------------
echo ""
for t in "${TARGETS[@]:-}"; do
    [ -n "$t" ] || continue
    if assert_removable "$t"; then          # belt-and-suspenders: check again
        echo "Removing $t ..."
        rm -rf -- "$t"
    else
        echo "SKIPPED (failed safety recheck): $t" >&2
    fi
done

if [ "$DO_COMPILE" -eq 1 ]; then
    case "$COMPILE_CACHE" in
        /data/?*)
            echo "Clearing torch.compile cache $COMPILE_CACHE ..."
            rm -rf -- "$COMPILE_CACHE"
            mkdir -p "$COMPILE_CACHE"        # entrypoint symlinks to it; keep it present
            ;;
    esac
fi

echo ""
echo "Done. Freed approximately $(human "$total_bytes")."
if [ "$DO_COMPILE" -eq 1 ]; then
    echo "Note: the next vLLM cold start will re-capture the CUDA graphs (60-120s slower once)."
fi
