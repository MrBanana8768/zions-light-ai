#!/bin/bash
set -e

echo "=============================================="
echo "  Zion's Light AI - Startup"
echo "  Backend: vLLM"
echo "  Model:   ${MODEL_REPO}"
echo "  Cache:   ${HF_HOME}"
echo "  Ctx:    ${MAX_MODEL_LEN} tokens (compactor target: ${COMPACTOR_TARGET_TOKENS:-auto})"
echo "=============================================="

# =============================================================================
# Preflight checks — fail loud and fast with actionable messages instead of
# letting vLLM crash 2-3 minutes into its startup with a cryptic stack trace.
# =============================================================================
echo "[1/3] Preflight checks..."

# Check 1: /data volume is writable. If not, the pod has no persistence and
# both model cache + OpenWebUI state will be lost on every restart.
if ! touch /data/.write-test 2>/dev/null; then
    echo "      ERROR: /data is not writable. Did you attach a Network Volume?"
    echo "             Expected mount: /data (single shared volume per RUNPOD_DEPLOY.md)"
    exit 1
fi
rm -f /data/.write-test
echo "      /data is writable"

# Create persistent subdirs on the volume (empty on first attach).
mkdir -p "${HF_HOME}" "${DATA_DIR}" /data/vllm-compile-cache

# Logs live on the volume, not in the container. A container-local log dies
# with the container, so every redeploy, OOM-kill or pod recreate destroyed
# the evidence for whatever prompted the redeploy. The 2026-08-27 context
# overflows are unreconstructable for exactly this reason: the failing
# container's logs went with it, and the investigation had to stop at
# "probable". LOG_DIR is expanded by supervisord.conf; /data is already a
# hard precondition above, so there is no fallback path to get wrong.
export LOG_DIR="${LOG_DIR:-/data/logs}"
mkdir -p "${LOG_DIR}"
# A boot marker, because these files now span deployments and a reader needs
# to know where one container's history ends and the next begins.
printf '\n===== boot %s | container %s | image %s =====\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(cat /etc/hostname)" "${IMAGE_TAG:-unknown}" \
    >> "${LOG_DIR}/boot.log"
echo "      logs -> ${LOG_DIR} (persists across redeploys)"

# Check 2: GPU is visible. nvidia-smi runs cleanly = host driver passthrough
# is working. If this fails, the container was started without --gpus all
# (or RunPod's equivalent).
if ! nvidia-smi >/dev/null 2>&1; then
    echo "      ERROR: nvidia-smi failed. No GPU passthrough?"
    echo "             For RunPod: confirm pod has a GPU attached."
    echo "             For local docker: use 'docker compose up' (compose file requests GPU)."
    exit 1
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)
echo "      GPU: ${GPU_NAME} (${GPU_MEM} MiB), driver ${DRIVER}"

# Check 3: driver version satisfies what our torch/vLLM wheels need. The CUDA
# channel this image was built against is baked in as ${TORCH_CUDA}; CUDA 13
# (cu130) needs a much newer driver than CUDA 12 (cu128/cu126). Bail with an
# actionable message rather than letting torch/vLLM crash later with "NVIDIA
# driver too old" or "libcudart.so.NN: cannot open shared object file".
DRIVER_MAJOR=$(echo "${DRIVER}" | cut -d. -f1)
# Fail CLOSED on a missing/unknown channel: default to the strictest floor
# (cu130 -> driver 580) so a mis-built image can't silently pass an under-spec host.
case "${TORCH_CUDA:-cu130}" in
    cu130) MIN_DRIVER=580; CUDA_LABEL="CUDA 13 (cu130)" ;;
    cu128) MIN_DRIVER=525; CUDA_LABEL="CUDA 12.8 (cu128)" ;;
    cu126) MIN_DRIVER=525; CUDA_LABEL="CUDA 12.6 (cu126)" ;;
    *)     MIN_DRIVER=580; CUDA_LABEL="${TORCH_CUDA:-unset} (unrecognized — requiring newest)" ;;
esac
if [ "${DRIVER_MAJOR}" -lt "${MIN_DRIVER}" ] 2>/dev/null; then
    echo "      ERROR: Driver ${DRIVER} is too old for this image (${CUDA_LABEL})."
    echo "             Need driver >= ${MIN_DRIVER}."
    if [ "${MIN_DRIVER}" -ge 580 ]; then
        echo "             This is the CUDA-13 build; it needs a driver-580+ host."
        echo "             Deploy on a newer host, or build the CUDA-12 fallback:"
        echo "               --build-arg CUDA_BASE_IMAGE=nvidia/cuda:12.6.3-runtime-ubuntu24.04 \\"
        echo "               --build-arg TORCH_CUDA=cu128 --build-arg VLLM_VERSION=0.19.0"
    else
        echo "             Pick a RunPod GPU/host with a newer driver."
    fi
    exit 1
fi
echo "      driver ${DRIVER} OK for ${CUDA_LABEL} (needs >= ${MIN_DRIVER})"

# Symlink vLLM's torch.compile cache onto the persistent volume. Without
# this, every cold start re-runs the 60-120s CUDA graph capture even
# though the cache key would have hit. Symlink is idempotent — re-runs
# are no-ops.
if [ ! -L /root/.cache/vllm ]; then
    mkdir -p /root/.cache
    rm -rf /root/.cache/vllm
    ln -s /data/vllm-compile-cache /root/.cache/vllm
    echo "      torch.compile cache linked to /data/vllm-compile-cache"
fi

echo ""

# =============================================================================
# Network / HuggingFace reachability check.
# =============================================================================
echo "[2/3] Checking HuggingFace connectivity..."

# OFFLINE IS A SUPPORTED MODE, NOT A FAILURE.
#
# This block used to `exit 1` after 60 seconds without huggingface.co — on a
# pod whose every model byte is already cached on /data. A boot-time network
# dependency that the running system does not actually need is the worst kind:
# it fires during a redeploy, which is exactly when nobody wants a puzzle.
# (REMEDIATION F25.)
#
# The rule now: if the cache is populated, unreachable HuggingFace is a
# supported state and we say so, loudly, once. We only refuse to boot when we
# have NEITHER weights nor a way to fetch them, because that is the only case
# where continuing produces a container that cannot serve.
HF_READY=false
for i in $(seq 1 10); do
    if curl -sf --max-time 3 "https://huggingface.co" > /dev/null 2>&1; then
        echo "      HuggingFace is reachable."
        HF_READY=true
        break
    fi
    sleep 1
done

# Is there a usable model cache? A snapshots/ directory under the HF hub
# layout is the cheapest honest proxy for "weights are already here".
HAVE_WEIGHTS=false
if [ -n "$(find "${HF_HOME}/hub" -maxdepth 3 -type d -name snapshots 2>/dev/null | head -1)" ]; then
    HAVE_WEIGHTS=true
fi

if [ "$HF_READY" = false ] && [ "$HAVE_WEIGHTS" = true ]; then
    echo "      HuggingFace is UNREACHABLE — running OFFLINE from ${HF_HOME}."
    echo "      This is supported. Model downloads and tokenizer resolution are"
    echo "      disabled for this boot; a model change will need connectivity."
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
elif [ "$HF_READY" = false ]; then
    echo "ERROR: HuggingFace is unreachable AND ${HF_HOME} holds no cached model."
    echo "       There is nothing to serve and no way to fetch it."
    echo "       Attach the volume holding the model cache, or restore"
    echo "       connectivity, then start the pod again."
    exit 1
fi

# Belt and braces: honour an operator-set offline flag even when the network
# IS reachable, so a deployment can be pinned offline deliberately.
if [ "${COMPACTOR_FORCE_OFFLINE:-false}" = "true" ]; then
    echo "      COMPACTOR_FORCE_OFFLINE=true — pinning HF offline."
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
fi

# Generate OpenWebUI secret key if not set
if [ -z "${WEBUI_SECRET_KEY}" ]; then
    export WEBUI_SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")
    echo "      Generated WebUI secret key"
fi

# =============================================================================
# Hand off to supervisord. vLLM, context-compactor, and OpenWebUI all run
# as supervised child processes from here.
# =============================================================================
# =============================================================================
# webui.db lives on LOCAL disk, not on /data.
#
# RunPod's MooseFS mount drops I/O occasionally. When that lands while
# OpenWebUI is mid-transaction, SQLite leaves a hot rollback journal, every
# later open tries to roll it back, rolling back needs to WRITE, the write
# fails, and the whole front end is down with "attempt to write a readonly
# database". That happened twice on 2026-08-31.
#
# So the live database and its journals sit on local disk where writes work,
# and compactor/webuidb.py publishes a snapshot back to /data on a timer.
# This step restores that snapshot on a fresh container - which is also the
# first-run migration of the existing database, needing no special case.
#
# DATABASE_URL is what actually points OpenWebUI at it. Deliberately NOT a
# symlink: SQLite derives the journal path from the path it was given, so a
# symlinked database can still put its journal back on MooseFS and the bug
# survives the fix.
# =============================================================================
# GATED since the v3.1.4.x rollout. WEBUI_DB_LOCAL=false keeps the live
# database exactly where v3.1.5 had it, so this release can ship its code
# without moving anything, and the move becomes its own deployment step.
#
# It is the last step of that series for a reason: it is the only change in
# the line whose rollback is not clean. Every other step is "redeploy the
# previous image"; this one has moved the live database, and /data holds a
# snapshot that is as old as the last sync.
#
# The flag is not rollout scaffolding to be deleted afterwards - it is the
# kill switch for the one subsystem here that owns where her chat history
# physically lives.
# v3.1.9 (hostile pass #2, LOW): this used to be `[ "${WEBUI_DB_LOCAL}" =
# "true" ]` — a byte comparison, no folding, no trimming — while every
# sibling boolean in this subsystem (WEBUI_DB_ALLOW_SHRINK,
# ALLOW_PUBLISH_OVER_UNREADABLE, ALLOW_ROW_LOSS, ALLOW_OLDER_GENERATION in
# webuidb.py; WEBUIDB_SYNC_ENABLED in health.py) folds case and trims
# whitespace. So `True`, `TRUE`, `1`, `yes`, `on`, or `" true"` — the exact
# shape a copy-paste into a RunPod template field leaves — all meant FALSE:
# keep the live database on MooseFS, sync daemon off, the 2026-08-31
# placement this whole module exists to retire, with the boot banner
# printing `WEBUI_DB_LOCAL=false` as if the operator had asked for it.
#
# Folded the same way below, but NOT through the `_bool` helper defined
# later in this file: `_bool` falls back to a SAFE DEFAULT on anything it
# does not recognise, and that is right for STT/TTS/selftest/backup, where
# the worst a wrong guess costs is one supervised program starting or not —
# visible in `supervisorctl status`, correctable by a redeploy. There is no
# equivalent safe guess here. This flag decides which PHYSICAL DISK holds
# her live SQLite file — webuidb.py's own header calls it "the only
# irreversible step in the line" — and the two wrong guesses are not
# equally bad but both are bad: silently resolving an unrecognised value to
# false re-admits the MooseFS rollback-journal corruption this release
# exists to remove, for the life of the pod, with nothing in any log to say
# so; silently resolving it to true moves the live database and starts the
# sync daemon publishing to /data on a placement nobody chose. So an
# unrecognised, NON-EMPTY value REFUSES TO BOOT instead of guessing between
# them — the same principle as the /data-writable and driver checks in
# [1/3] above: a boot that cannot end in an honest system does not get to
# proceed halfway.
#
# UNSET OR EMPTY IS NOT "UNRECOGNISED" and keeps meaning true, unchanged.
# Production sets this flag EXPLICITLY — WEBUI_DB_LOCAL=false is a required
# RunPod template field there for the pods still mid-rollout — so empty only
# happens on a fresh boot with no template at all, where true is the
# architecture every pod in this series is migrating towards; refusing to
# boot on nothing set would break that default path for no reason. Explicit
# `true` and explicit `false` resolve exactly as they always did — this
# folds spelling and whitespace, it does not change either literal's
# meaning.
#
# test_webuidb_gate.py reads this exact block by its BEGIN/END markers and
# runs it under `sh` with the finding's own spellings — keep it POSIX sh.
# BEGIN WEBUI_DB_LOCAL NORMALIZATION
_webui_db_local_raw="${WEBUI_DB_LOCAL:-}"
if [ -z "${_webui_db_local_raw}" ]; then
    WEBUI_DB_LOCAL=true
    _webui_db_local_why="unset/empty -> true (the local-disk move stays the default)"
else
    _webui_db_local_norm=$(printf '%s' "${_webui_db_local_raw}" | tr -d '[:space:]' | tr '[:upper:]' '[:lower:]')
    case "${_webui_db_local_norm}" in
        true|1|yes|on)
            WEBUI_DB_LOCAL=true
            _webui_db_local_why="explicitly set (${_webui_db_local_raw})"
            ;;
        false|0|no|off)
            WEBUI_DB_LOCAL=false
            _webui_db_local_why="explicitly set (${_webui_db_local_raw})"
            ;;
        *)
            echo ""
            echo "      ============================================================"
            echo "      WEBUI_DB_LOCAL=[${_webui_db_local_raw}] is not true/1/yes/on"
            echo "      or false/0/no/off - REFUSING TO START."
            echo ""
            echo "      This flag decides which disk holds her live chat database."
            echo "      Guessing is worse than stopping: reading it as false would"
            echo "      silently re-expose the MooseFS rollback-journal corruption"
            echo "      this release exists to remove; reading it as true would move"
            echo "      the live database and start publishing to /data on a"
            echo "      placement nobody chose. Set WEBUI_DB_LOCAL to exactly true"
            echo "      or false on the RunPod template and redeploy."
            echo "      ============================================================"
            echo ""
            exit 1
            ;;
    esac
fi
export WEBUI_DB_LOCAL
echo "      WEBUI_DB_LOCAL=${WEBUI_DB_LOCAL} (${_webui_db_local_why})"
# END WEBUI_DB_LOCAL NORMALIZATION
export WEBUI_LOCAL_DB="${WEBUI_LOCAL_DB:-/var/lib/openwebui/webui.db}"
export WEBUI_SNAPSHOT_DB="${WEBUI_SNAPSHOT_DB:-${DATA_DIR:-/data/openwebui}/webui.db}"
# The escape hatch for the boot refusal below. Default false: a restore that
# fails stops the boot. It exists because "refuse forever" is its own failure
# mode - if the snapshot is genuinely unrecoverable, an operator has to be
# able to bring the pod up and start again from nothing, deliberately, having
# read what that costs.
export WEBUI_DB_ALLOW_EMPTY_START="${WEBUI_DB_ALLOW_EMPTY_START:-false}"

if [ "${WEBUI_DB_LOCAL}" = "true" ]; then
    export DATABASE_URL="${DATABASE_URL:-sqlite:///${WEBUI_LOCAL_DB}}"
    export WEBUIDB_SYNC_ENABLED=true
    mkdir -p "$(dirname "${WEBUI_LOCAL_DB}")"
    echo "[2b/3] Placing webui.db on local disk (${WEBUI_LOCAL_DB})"

    # THE EXIT CODE, NOT THE PIPELINE'S. This step was
    #
    #     webuidb.py --restore 2>&1 | tail -3 || { echo WARNING...; }
    #
    # and that warning could not print, three ways over at once:
    # restore_on_boot() never raises (every failure RETURNS a dict with
    # action="error"/"snapshot_unhealthy"/"restore_failed"), __main__ printed
    # that dict with no sys.exit, and the status of `cmd | tail` is TAIL's,
    # which is 0 whatever ran to its left. So a restore that failed outright
    # reported success: OpenWebUI started with DATABASE_URL pointing at a file
    # that does not exist, alembic built a fresh empty schema, and one message
    # later the sync daemon published that 1-chat database over the snapshot
    # holding every conversation she has. webuidb.py now exits per failure
    # (RESTORE_EXIT_CODES in that module) and this reads that status.
    #
    # A command substitution, NOT `set -o pipefail`. pipefail is a GLOBAL
    # flag, and the obvious way to reach for it -
    #     set -o pipefail; cmd | tail || { set +o pipefail; ...; }
    # - puts the restore inside the branch that only runs on FAILURE, so a
    # successful restore leaves pipefail set for every later pipeline in this
    # script. `x="$(cmd)" || rc=$?` needs no flag and leaks nothing. (The
    # `|| rc=$?` is not optional either: a bare assignment from a command
    # substitution is a simple command, and under `set -e` a failing one
    # aborts the script before the next line can look at it.)
    restore_rc=0
    restore_out="$(/opt/compactor-venv/bin/python /opt/compactor/webuidb.py --restore 2>&1)" \
        || restore_rc=$?
    printf '%s\n' "${restore_out}" | tail -3

    if [ "${restore_rc}" -ne 0 ]; then
        # WHY THIS REFUSES TO BOOT RATHER THAN WARNING AND CARRYING ON. Both
        # outcomes were written out before choosing between them.
        #
        # Carrying on: OpenWebUI starts, finds nothing at DATABASE_URL, builds
        # an empty schema, and she opens the app to a companion that has
        # forgotten her. Every service is green, nothing says a word, and the
        # pod runs that way until a person happens to look. The snapshot on
        # /data is still intact in that moment - but the live database is now
        # an empty one, and something eventually publishes it.
        #
        # Refusing: the pod does not come up. Loud, immediate, and it leaves
        # /data EXACTLY as it was, with the snapshot still there for
        # scripts/recover-webui-db.py and nothing written over it.
        #
        # The asymmetry that decides it: this deployment works fail-forward
        # and does not roll back, so data loss is the one failure class with
        # no recovery path. A pod that will not boot can be recovered by hand
        # at any later hour. A pod that boots empty and then publishes cannot
        # be, and it is the COMPOSITION rather than either half that loses
        # everything - so it gets broken here, before anything writes. This
        # file already refuses to boot for an unwritable /data and for a
        # driver too old to serve, on the same principle: a boot that cannot
        # end in an honest system does not get to proceed halfway.
        echo ""
        echo "      ============================================================"
        echo "      RESTORE FAILED (exit ${restore_rc}) - REFUSING TO START."
        echo ""
        case "${restore_rc}" in
            3)  echo "      The snapshot on /data failed quick_check - a hot rollback"
                echo "      journal, a corrupt header, a stalled mount - and local disk"
                echo "      holds no history either. It was NOT copied down: restoring a"
                echo "      half-rolled-back database and calling it live is how a"
                echo "      recoverable problem becomes a permanent one." ;;
            4)  echo "      The snapshot is healthy but could not be copied to local disk."
                echo "      A full overlay is the usual cause. Nothing on /data was"
                echo "      touched." ;;
            5)  echo "      Local disk could not be prepared: either ${WEBUI_LOCAL_DB%/*}"
                echo "      could not be created, or an existing local database could not"
                echo "      be moved aside and so must not be overwritten - OR the snapshot"
                echo "      on /data could not even be checked (a stalled or erroring mount"
                echo "      answering neither 'found' nor 'not found'), which webuidb.py"
                echo "      now refuses rather than treating as 'no snapshot'. The log line"
                echo "      just above this banner says which." ;;
            *)  echo "      webuidb.py reported a failure this script has no text for."
                echo "      Treated as fatal deliberately - an unrecognised failure on"
                echo "      this path is not a pass." ;;
        esac
        echo ""
        echo "      Look, with someone watching:"
        echo "        /opt/compactor-venv/bin/python /opt/compactor/webuidb.py --status"
        echo "        /opt/compactor-venv/bin/python /data/scripts/recover-webui-db.py --check"
        echo "      Then redeploy. Nothing has been lost and nothing was changed."
        echo ""
        echo "      To start anyway, having read the above and accepting that"
        echo "      OpenWebUI will build an EMPTY schema and show her no history:"
        echo "        WEBUI_DB_ALLOW_EMPTY_START=true"
        echo "      ============================================================"
        echo ""
        if [ "${WEBUI_DB_ALLOW_EMPTY_START}" != "true" ]; then
            exit 1
        fi
        echo "      WEBUI_DB_ALLOW_EMPTY_START=true - starting anyway."

        # THE CLAIM THAT USED TO BE PRINTED HERE WAS FALSE, AND THE BEHAVIOUR
        # HAS BEEN CHANGED RATHER THAN THE WORDING. It said: "The sync daemon
        # stays ON, and that is safe rather than an oversight: every route out
        # of this state is refused by webuidb.sync_once." It is not safe. The
        # shrink guard is a RATIO, so an empty-started database is refused only
        # until it grows past half the snapshot - one was demonstrated
        # publishing over the snapshot at 51% of its size. At ~2 MB/day against
        # a 41 MB snapshot that is roughly ten days, and this is a RunPod
        # template variable, so once it is set to get a pod up it stays set
        # through every later redeploy and reprints this banner into a boot log
        # nobody re-reads. The guard was buying time and calling it safety.
        #
        # So leave a marker the sync daemon refuses on unconditionally. On
        # LOCAL disk, not /data: a marker on the volume would outlive the
        # incident and become a permanent block on publishing, which is the
        # defect this whole series is repairing. On the overlay it dies with
        # the pod, so it can only ever describe THIS boot.
        #
        # FAIL CLOSED IF IT CANNOT BE WRITTEN. A guard that silently no-ops
        # when the disk says no is worse than no guard, because the banner
        # above has just told the operator they are protected. If we cannot
        # write one file to local disk, OpenWebUI is not going to be able to
        # write a database there either.
        empty_start_marker="$(dirname "${WEBUI_LOCAL_DB}")/.empty-start"
        if ! printf '%s\n' \
            "written by entrypoint.sh at $(date -u +%Y-%m-%dT%H:%M:%SZ)" \
            "restore exited ${restore_rc} and WEBUI_DB_ALLOW_EMPTY_START=true" \
            "the only durable copy of her history is ${WEBUI_SNAPSHOT_DB}" \
            "while this file exists the sync daemon refuses to publish" \
            "delete it only to accept starting over from nothing" \
            > "${empty_start_marker}"
        then
            echo ""
            echo "      ============================================================"
            echo "      Could not write ${empty_start_marker} - REFUSING TO START."
            echo "      That file is what stops the empty database OpenWebUI is"
            echo "      about to build from being published over the snapshot on"
            echo "      /data once it grows past half its size. Without it this"
            echo "      pod would come up looking protected and would not be."
            echo "      ============================================================"
            exit 1
        fi
        echo "      The sync daemon stays ON so the log keeps saying what is"
        echo "      wrong, but it will NOT publish: ${empty_start_marker}"
        echo "      refuses every cycle while it exists. What protects /data is"
        echo "      that file, not the shrink ratio - the ratio only delays an"
        echo "      empty database, it does not stop one."
        echo ""
        echo "      To get back to normal: repair the snapshot, then run"
        echo "        /opt/compactor-venv/bin/python /opt/compactor/webuidb.py --restore"
        echo "      which restores it to local disk and clears the marker."
        echo "      To accept starting over from nothing instead, delete the"
        echo "      marker by hand and the next sync overwrites the snapshot."
    fi
else
    # THE SYNC DAEMON MUST NOT RUN HERE, and this is the whole safety of the
    # flag. It publishes the LOCAL file over the snapshot path; with the
    # live database sitting at that same snapshot path, a single publish
    # would overwrite her real chat history with whatever stale copy is on
    # local disk. Off means off, not "on but pointed elsewhere".
    export DATABASE_URL="${DATABASE_URL:-sqlite:///${WEBUI_SNAPSHOT_DB}}"
    export WEBUIDB_SYNC_ENABLED=false
    echo "[2b/3] WEBUI_DB_LOCAL=false - webui.db stays on ${WEBUI_SNAPSHOT_DB}"
    echo "      The local-disk move and its sync daemon are BOTH off. This is"
    echo "      the pre-v3.1.6 placement, so MooseFS is still in the SQLite write"
    echo "      path and a hot rollback journal remains possible."
fi

# =============================================================================
# v3.1.9 HIGH #4 (hostile pass 2). supervisord.conf gates five programs on
# `autostart=%(ENV_x)s`. supervisord's own boolean() parser
# (supervisor/datatypes.py) accepts only {yes,true,on,1}/{no,false,off,0},
# case-insensitively, and does NOT strip whitespace - anything else raises
# ValueError DURING CONFIG LOAD, before any program starts. `exec supervisord`
# below is this container's PID 1, so that raise takes vLLM, OpenWebUI, the
# compactor, STT, TTS and the backup daemon down together - not just the one
# program whose flag was wrong. `true ` (a trailing space, which is exactly
# what a copy-paste into a RunPod template field leaves behind), `enabled`,
# `y`, `t`, `2` and an empty string are all fatal. Compare: the ten sites
# 91d8463 fixed the same way for Python int()/float() reads each crash ONE
# supervised program under autorestart=true while the rest of the pod keeps
# serving; these five crash everything, which is why this is HIGH rather than
# the same MEDIUM as that commit.
#
# BEGIN SUPERVISORD BOOL NORMALIZATION (test_config_supervisord_bool.py reads
# this exact block by its BEGIN/END markers and runs it under `sh` with bad
# spellings - keep it POSIX sh, no bash-only syntax, if you touch it).
#
# _bool NAME DEFAULT: read env var NAME, trim whitespace (leading, trailing,
# and an embedded trailing newline), lowercase it, and echo "true" or "false"
# for any of supervisord's own accepted spellings (plus a couple of common
# near-misses that are unambiguous); anything else - a typo, a stray word, an
# empty value - echoes DEFAULT and PRINTS the raw value it saw to stderr, so
# a wrong template field is loud in the boot log instead of a silent
# never-boots. DEFAULT is chosen per variable below, not once for all four:
# every one of them defaults to keeping the pod's current documented
# behaviour (the service runs) rather than guessing "off is always safer" -
# an operator who typo'd a value meant to CHANGE the default, and the two
# services this gates (voice, backups, the self-test net, the compactor
# itself) are all things the pod already runs by default in Dockerfile ENV.
_bool() {
    name="$1"; default="$2"
    raw=$(eval "printf '%s' \"\${$name:-}\"")
    norm=$(printf '%s' "$raw" | tr -d '[:space:]' | tr '[:upper:]' '[:lower:]')
    case "$norm" in
        true|yes|on|1|y|t) echo "true" ;;
        false|no|off|0|n|f) echo "false" ;;
        "") echo "$default" ;;
        *)
            # Brackets, not quotes, and no bash-only ${var@Q}: this block is
            # sourced and run under plain `sh` by the unit test, and quoting
            # operators like @Q are bash-only. Brackets still make a leading/
            # trailing space or an empty value visible in the log.
            echo "      WARNING: ${name}=[${raw}] is not a recognised boolean" \
                 "(supervisord.conf autostarts a program on it) - using the" \
                 "safe default ${name}=${default}" >&2
            echo "$default"
            ;;
    esac
}
export STT_ENABLED="$(_bool STT_ENABLED true)"
export TTS_ENABLED="$(_bool TTS_ENABLED true)"
export COMPACTOR_SELFTEST_ON_BOOT="$(_bool COMPACTOR_SELFTEST_ON_BOOT true)"
export COMPACTOR_BACKUP_ENABLED="$(_bool COMPACTOR_BACKUP_ENABLED true)"
# WEBUIDB_SYNC_ENABLED is the fifth autostart=%(ENV_x)s boolean in
# supervisord.conf but is DELIBERATELY NOT routed through _bool here: unlike
# the four above, it is never a raw operator-facing env var - both branches
# of the `if [ "${WEBUI_DB_LOCAL}" = "true" ]` block above set it themselves,
# a few lines up, to the exact literal "true" or "false", never from
# something an operator typed. Normalising it again would be a harmless
# no-op today but test_webuidb_gate.py [6] pins the `export` of this
# variable occurring exactly TWICE in this file (once per branch) as the
# property that actually matters here - always exported, on both paths, so
# supervisord never sees an unresolved %(ENV_x)s - and a third occurrence
# from routing it through _bool as well would only add a place for the two
# to drift apart.
# END SUPERVISORD BOOL NORMALIZATION

echo ""
echo "[3/3] Starting services..."
echo "      - vLLM             on port ${VLLM_PORT}      (internal)"
echo "      - context-compactor on port ${COMPACTOR_PORT}  (OpenWebUI talks here)"
echo "      - OpenWebUI        on port ${OPENWEBUI_PORT}  (user-facing)"
echo ""
echo "      Note: vLLM downloads model weights on first run; first startup"
echo "      may take 5-15 minutes depending on model size and network speed."
echo "      Weights are cached to ${HF_HOME} (persist via volume mount)."
echo "      torch.compile cache lives at /data/vllm-compile-cache — second"
echo "      and later cold starts skip the 60-120s CUDA graph capture."
echo "=============================================="

exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf
