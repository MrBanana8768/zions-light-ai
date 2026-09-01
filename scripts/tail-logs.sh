#!/usr/bin/env bash
# Follow the compactor and vLLM together, signal first.
#
#   /data/scripts/tail-logs.sh            # the lines that mean something
#   /data/scripts/tail-logs.sh --all      # everything except pure noise
#   /data/scripts/tail-logs.sh --grep pat # add your own pattern to the signal set
#
# WHY NOT JUST `tail -f`. Two reasons, both measured on a real bundle.
#
# `-f` goes deaf on rotation. supervisord rotates these at 50MB
# (stdout_logfile_maxbytes in supervisord.conf) and plain -f keeps following
# the old inode, so the tail silently stops updating at exactly the moment a
# busy incident produces enough output to rotate. `-F` re-opens by name.
#
# And volume: 19,051 compactor lines in one window, of which the health-check
# poll alone is ~47%. Dropping the obvious noise still leaves 53% - mostly
# routine dedup clustering, retrieval-block sizing and token-scale INFO. That
# is not a tail anyone can watch. So the default shows only lines that change
# a decision, and --all is there when you need the surrounding context.
#
# Each line is prefixed with its source, because "which process said that"
# is the first question and interleaved output loses it.

set -uo pipefail
LOG_DIR="${LOG_DIR:-/data/logs}"

MODE=signal
EXTRA=""
while [ $# -gt 0 ]; do
    case "$1" in
        --all)  MODE=all ;;
        --grep) shift; EXTRA="${1:-}" ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

# SIGNAL: a request starting, what memory it got, and anything that went
# wrong. Deliberately NOT the routine INFO - dedup clustering, retrieval
# sizing, per-call token scale - which is 60%+ of the volume and tells you
# nothing while you are watching a problem happen.
SIGNAL='conv_id=|injected memory|WARNING|ERROR|Traceback|hard budget|compaction skipped'
SIGNAL="$SIGNAL"'|rollup|repetition loop|stream ended|stream truncated|REJECTED|FAILED'
SIGNAL="$SIGNAL"'|degraded|forked|merged|image retention|space-filled|modality'
[ -n "$EXTRA" ] && SIGNAL="$SIGNAL|$EXTRA"

# NOISE: health polling, model-list probes, CUDA graph capture and weight
# loading progress bars. None of it is ever the answer, and the progress
# bars are single lines thousands of characters wide.
NOISE='GET /health|GET /v1/models|GET / HTTP|Capturing CUDA|it/s\]|Loading safetensors'
NOISE="$NOISE"'|Non special vocabulary|Cutting non special'

pids=()
follow() {  # follow <tag> <file>
    [ -f "$2" ] || { echo "[$1] (no such file: $2)"; return; }
    if [ "$MODE" = all ]; then
        tail -F -n 20 "$2" 2>/dev/null \
            | grep --line-buffered -avE "$NOISE" \
            | sed -u "s/^/[$1] /" &
    else
        tail -F -n 20 "$2" 2>/dev/null \
            | grep --line-buffered -aE "$SIGNAL" \
            | grep --line-buffered -avE "$NOISE" \
            | sed -u "s/^/[$1] /" &
    fi
    pids+=($!)
}

cleanup() { kill "${pids[@]}" 2>/dev/null; exit 0; }
trap cleanup INT TERM

echo "following ${LOG_DIR} — mode=${MODE}${EXTRA:+ (+${EXTRA})} — ctrl-c to stop"
echo "  [cmp] compactor   [cmp!] compactor stderr   [llm] vLLM   [web] OpenWebUI"
echo

follow "cmp"  "${LOG_DIR}/compactor.log"
follow "cmp!" "${LOG_DIR}/compactor-error.log"
follow "llm"  "${LOG_DIR}/vllm-error.log"
follow "web"  "${LOG_DIR}/openwebui.log"

wait
