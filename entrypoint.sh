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
# Hand off to supervisord. vLLM, context-compactor, PostgreSQL, and
# OpenWebUI all run as supervised child processes from here.
# =============================================================================

# =============================================================================
# PostgreSQL — the state home (ARCHITECTURE.md Decision 4), built BEFORE the
# failure class it targets ever got the chance to repeat: `/data` is a
# RunPod MooseFS network mount that drops I/O occasionally, and Postgres's
# WAL is at least as intolerant of that as SQLite's rollback journal was —
# see the webui.db comment block just below for exactly what that failure
# looks like when it lands mid-write.
#
# So PGDATA lives on LOCAL disk (the pod overlay), Postgres listens on a
# UNIX SOCKET ONLY (no listen_addresses, no TCP port to fail or expose —
# unix_socket_directories is set below at initdb time), and `/data` is used
# ONLY as a periodic `pg_dump` archive target
# (compactor/pgarchive.py, supervisord program `pgarchive`). A stalled
# `/data` degrades that archive daemon to a loud warning; Postgres itself
# never touches the volume and is never in the hot path.
#
# This whole block is IDEMPOTENT: `[ -f "$PGDATA/PG_VERSION" ]` is the same
# "does real state already exist here" gate webuidb.py uses, so a plain
# container restart with an existing PGDATA does nothing destructive — it
# skips straight to "make sure the role/db exist" (itself idempotent) and
# restore_if_needed() (which refuses to restore over a database that
# already has tables — see pgarchive.py).
# =============================================================================
export PGDATA="${PGDATA:-/var/lib/postgresql/data}"
export PG_BIN="${PG_BIN:-/usr/lib/postgresql/16/bin}"
export POSTGRES_USER="${POSTGRES_USER:-openwebui}"
export POSTGRES_DB="${POSTGRES_DB:-openwebui}"
export POSTGRES_SOCKET_DIR="${POSTGRES_SOCKET_DIR:-/var/run/postgresql}"
export DATABASE_URL="${DATABASE_URL:-postgresql://${POSTGRES_USER}@/${POSTGRES_DB}?host=${POSTGRES_SOCKET_DIR}}"

echo "[2b/3] PostgreSQL: preparing ${PGDATA} (local disk, unix socket only)"

# Socket directory. root:postgres/2775 matches Debian's own convention for
# /var/run/postgresql: the postgres server (running as user postgres, see
# supervisord.conf) can create the socket, and any root-run client in this
# container (compactor, pgarchive, psql run by hand for debugging) can reach
# it — there is no TCP listener for anything outside the container to reach
# in the first place, so this permission only has to make sense INSIDE it.
mkdir -p "${POSTGRES_SOCKET_DIR}"
chown postgres:postgres "${POSTGRES_SOCKET_DIR}"
chmod 2775 "${POSTGRES_SOCKET_DIR}"

mkdir -p "${PGDATA}"
chown -R postgres:postgres "${PGDATA}"
# initdb refuses a world/group-readable PGDATA outright ("data directory
# has group or world access") — set this explicitly rather than relying on
# mkdir's umask-dependent default.
chmod 700 "${PGDATA}"

if [ ! -f "${PGDATA}/PG_VERSION" ]; then
    echo "      first boot: running initdb"
    # --auth-local=trust, --auth-host=reject: no password needed over the
    # unix socket (the container boundary IS the security boundary here,
    # same reasoning already accepted for the supervisorctl socket above),
    # and TCP auth is rejected outright as belt-and-braces — redundant with
    # listen_addresses='' below, but a config that can't even parse into
    # "allow TCP" is a stronger guarantee than one that merely doesn't ask
    # for it.
    runuser -u postgres -- "${PG_BIN}/initdb" -D "${PGDATA}" \
        --auth-local=trust --auth-host=reject --encoding=UTF8 \
        >>"${LOG_DIR}/postgres-initdb.log" 2>&1 || {
        echo "      ERROR: initdb failed. See ${LOG_DIR}/postgres-initdb.log"
        exit 1
    }
    {
        echo ""
        echo "# --- appended by entrypoint.sh, once, at initdb ---"
        echo "include_if_exists = 'zionslight.conf'"
    } >> "${PGDATA}/postgresql.conf"
else
    echo "      ${PGDATA}/PG_VERSION present — reusing existing cluster (idempotent)"
fi

# Settings live in an INCLUDE, rewritten on every boot, rather than appended
# to postgresql.conf at initdb. Appending only-on-first-boot means a cluster
# created by an older image keeps that image's settings forever, and the only
# way to change them is to destroy her database — which is precisely the
# thing this deployment must never require. postgresql.conf reads the include
# last, so these win over the defaults above them, and a redeploy re-applies
# whatever this file says.
#
# Sized for the disk this actually runs on. `df` on the pod, 2026-08-31:
#
#     overlay   20G  388M   20G   2%  /
#
# 20 GB, ephemeral, shared with /var/lib/openwebui and every container
# layer. Postgres's defaults assume a dedicated server volume and are wrong
# here in the direction that matters: a full PGDATA is not a degraded
# database, it is a PANIC and a wedged pod.
cat > "${PGDATA}/zionslight.conf" <<PGCONF
# Rewritten by entrypoint.sh on every boot. Do not edit in place.

# --- unix socket only, no TCP listener to fail or expose ---------------
listen_addresses = ''
unix_socket_directories = '${POSTGRES_SOCKET_DIR}'

# --- sized for a 20 GB EPHEMERAL overlay -------------------------------
# max_wal_size caps how much WAL accumulates between checkpoints. The 1 GB
# default is built for write volumes this will never see (two users, one of
# them daily) and just reserves disk that the overlay does not have spare.
# 512 MB checkpoints more often, which costs nothing at this scale and
# halves the worst-case WAL footprint.
max_wal_size = 512MB
min_wal_size = 80MB

# wal_compression matters here specifically because of IMAGES. Full-page
# writes after a checkpoint copy whole 8 KB pages into WAL, and pages full
# of base64 image data are the biggest ones this database will hold. lz4
# compresses them cheaply and keeps the WAL bounded.
wal_compression = lz4

# 128 MB (the default) is small for anything, and this box has RAM to spare
# next to a 24B model on the GPU. 256 MB comfortably holds the whole working
# set of a database this size.
shared_buffers = 256MB

# A runaway query must not be able to fill the disk out from under Postgres
# and PANIC it. Nothing OpenWebUI does should come close to 1 GB of temp
# files; if something does, failing that one query is the correct outcome.
temp_file_limit = 1GB

# --- durability. This is her conversation history ----------------------
# Both ON, deliberately and not to be "optimised" later: local disk is fast
# enough that the cost is irrelevant at this write volume, and the whole
# point of this migration was to stop losing data to a storage layer.
synchronous_commit = on
full_page_writes = on

# --- so an incident has evidence ---------------------------------------
log_min_duration_statement = 2000
log_checkpoints = on
log_line_prefix = '%m [%p] %q%u@%d '
PGCONF
chown postgres:postgres "${PGDATA}/zionslight.conf"
chmod 600 "${PGDATA}/zionslight.conf"

# Start Postgres TEMPORARILY to do setup (role/db/restore) as this script,
# not as the supervised process — supervisord starts the real, long-running
# instance later (program `postgres`, priority 12, before openwebui's 20).
# Two Postgres processes must never point at the same PGDATA at once (the
# postmaster.pid lock prevents it anyway), so this instance is stopped again
# before supervisord ever launches.
echo "      starting Postgres temporarily for role/db setup and restore"
# The log file must exist and be writable BY POSTGRES before pg_ctl -l is used.
# pg_ctl does not hand the postmaster an inherited fd: it builds
# `exec postgres ... >> "<logfile>"` and runs that through /bin/sh AS THE
# POSTGRES USER. LOG_DIR is /data/logs, created by root at 0755, so that
# append is EACCES, the postmaster never starts, -w times out at 60s, and the
# boot dies at [2b/3] before anything serves. Every other postgres call in
# this file is safe because the ROOT shell opens the fd and permission is
# checked at open, not at write. -l is the one outlier, and postgres is the
# only non-root process in this container.
touch "${LOG_DIR}/postgres-initdb.log"
chown postgres:postgres "${LOG_DIR}/postgres-initdb.log"
runuser -u postgres -- "${PG_BIN}/pg_ctl" -D "${PGDATA}" \
    -l "${LOG_DIR}/postgres-initdb.log" -w -t 60 start \
    >>"${LOG_DIR}/postgres-initdb.log" 2>&1 || {
    echo "      ERROR: Postgres would not start for setup. See ${LOG_DIR}/postgres-initdb.log"
    exit 1
}

# Role + database: idempotent (check-then-create; Postgres has no
# CREATE ROLE ... IF NOT EXISTS). Connects as OS user postgres, which trust
# auth maps to the postgres superuser role created by initdb.
runuser -u postgres -- psql -h "${POSTGRES_SOCKET_DIR}" -U postgres -d postgres -tAc \
    "SELECT 1 FROM pg_roles WHERE rolname='${POSTGRES_USER}'" 2>>"${LOG_DIR}/postgres-initdb.log" \
    | grep -q 1 || \
runuser -u postgres -- psql -h "${POSTGRES_SOCKET_DIR}" -U postgres -d postgres -c \
    "CREATE ROLE \"${POSTGRES_USER}\" LOGIN" >>"${LOG_DIR}/postgres-initdb.log" 2>&1

runuser -u postgres -- psql -h "${POSTGRES_SOCKET_DIR}" -U postgres -d postgres -tAc \
    "SELECT 1 FROM pg_database WHERE datname='${POSTGRES_DB}'" 2>>"${LOG_DIR}/postgres-initdb.log" \
    | grep -q 1 || \
runuser -u postgres -- psql -h "${POSTGRES_SOCKET_DIR}" -U postgres -d postgres -c \
    "CREATE DATABASE \"${POSTGRES_DB}\" OWNER \"${POSTGRES_USER}\"" >>"${LOG_DIR}/postgres-initdb.log" 2>&1
echo "      role '${POSTGRES_USER}' and database '${POSTGRES_DB}' present"

# Restore from the newest good /data archive if (and only if) the database
# is empty — see pgarchive.restore_if_needed(). A database that already has
# tables (a plain restart with PGDATA intact) is never touched.
# pipefail around these two: `cmd 2>&1 | tail -5 || { WARNING }` reports
# TAIL's exit status, which is always 0, so neither warning handler could
# ever run - a restore could fail silently and the boot would say nothing.
# Both sites had it; the pgarchive one was written by copying the webuidb
# one, which is how a one-site bug becomes a two-site bug.
set -o pipefail
/opt/compactor-venv/bin/python /opt/compactor/pgarchive.py --restore-if-needed 2>&1 | tail -5 || {
set +o pipefail
    echo "      WARNING: restore-if-needed step failed; Postgres starts with"
    echo "      whatever schema is already there (empty, on a truly fresh"
    echo "      deployment). Check ${LOG_DIR}/pgarchive.log and run:"
    echo "        /opt/compactor-venv/bin/python /opt/compactor/pgarchive.py --status"
}


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
# STATUS: dormant by default. DATABASE_URL is now set above, pointing
# OpenWebUI at Postgres — OpenWebUI never opens webui.db at all unless
# DATABASE_URL is explicitly overridden back to a sqlite:/// URL, which is
# this project's documented rollback path (a connection-string change, not
# a code change; see the Dockerfile's SQLite-hardening comment). The restore
# step below still runs unconditionally: it is cheap, and it means the
# rollback path stays live (a warm local copy ready to go) rather than
# something that has to be re-derived under pressure during an incident.
#
# Deliberately NOT a symlink: SQLite derives the journal path from the path
# it was given, so a symlinked database can still put its journal back on
# MooseFS and the bug survives the fix.
# =============================================================================
export WEBUI_LOCAL_DB="${WEBUI_LOCAL_DB:-/var/lib/openwebui/webui.db}"
export WEBUI_SNAPSHOT_DB="${WEBUI_SNAPSHOT_DB:-${DATA_DIR:-/data/openwebui}/webui.db}"
mkdir -p "$(dirname "${WEBUI_LOCAL_DB}")"
echo "[2c/3] Placing webui.db on local disk (${WEBUI_LOCAL_DB}) — rollback path, dormant while DATABASE_URL points at Postgres"
set -o pipefail
/opt/compactor-venv/bin/python /opt/compactor/webuidb.py --restore 2>&1 | tail -3 || {
set +o pipefail
    echo "      WARNING: restore step failed; OpenWebUI will still start."
    echo "      Check ${LOG_DIR}/webuidb-sync.log and run:"
    echo "        /opt/compactor-venv/bin/python /opt/compactor/webuidb.py --status"
}

# =============================================================================
# WHICH DATABASE DOES OPENWEBUI ACTUALLY OPEN?
#
# The step that was missing, and it is the one that loses her history. The
# Postgres block above builds a cluster and, on a truly fresh deployment,
# leaves it EMPTY — pgarchive restores only from a /data archive, and on the
# first deploy of this image no archive has ever been written. Nothing in
# the boot path copies webui.db's contents across; that is
# scripts/migrate-webui-sqlite-to-pg.py, which is deliberately manual
# because it must run AFTER OpenWebUI has built the schema and because it
# is the step that can go wrong irreversibly.
#
# So on the upgrade boot the sequence would have been: empty Postgres,
# DATABASE_URL pointing at it, OpenWebUI starts, alembic creates an empty
# schema, and she opens the app to find every conversation gone. The data
# still exists in webui.db and is recoverable, which is the only reason
# this is not catastrophic — but "your companion has forgotten you, ask
# an engineer" is not a deploy outcome worth risking on a Sunday.
#
# The fix is not to automate the migration here. It is to NOT SILENTLY
# SWITCH: if Postgres is empty and SQLite is not, keep serving SQLite —
# the path that already works, on local disk, that this release was
# already going to ship — and say loudly what to run. Nobody loses
# anything, there is no downtime, and the migration happens when a human
# is watching, which is the only way it should ever happen.
#
# Once migrated, Postgres has tables and this block hands over to it
# permanently with no further intervention.
# =============================================================================
# The DECISION itself lives in compactor/dbselect.py, not here — see that
# module's docstring. entrypoint.sh's job is only to gather the three
# inputs honestly (below) and print what the decision means for her boot;
# duplicating the if/elif branching in a test as well as here is exactly
# the kind of drift this project keeps re-discovering the hard way.
_pg_public_tables() {
    runuser -u postgres -- psql -h "${POSTGRES_SOCKET_DIR}" -U postgres \
        -d "${POSTGRES_DB}" -tAc \
        "SELECT count(*) FROM information_schema.tables WHERE table_schema='public'" \
        2>/dev/null | tr -d '[:space:]'
}
# Prints the chat count, or the literal string "unknown" if _has_rows()
# returned None (unreadable/locked/corrupt) or the probe crashed before it
# could print anything at all. NEVER prints "0" for a failure — dbselect.py
# treats "0" as a KNOWN empty database and "unknown" as "do not guess", and
# collapsing the two here would silently reintroduce the exact bug this
# split exists to prevent (see dbselect.py's module docstring).
_sqlite_chats() {
    /opt/compactor-venv/bin/python - "$1" <<'PYEOF' 2>/dev/null
import sys, pathlib
sys.path.insert(0, "/opt/compactor")
import webuidb
n = webuidb._has_rows(pathlib.Path(sys.argv[1]))
print(n if n is not None else "unknown")
PYEOF
}

# `X="$(f)"` is a BARE assignment from a command substitution, so under
# `set -e` a non-zero exit from f aborts the whole script - the ${X:-...}
# fallback on the next line never runs. Both probes are therefore guarded
# explicitly with `|| X=""`, which lets the fail-safe defaults below
# actually be reached.
#
# _pg_public_tables happened to be safe already: its body ends in a `| tr`
# pipeline, so it always returns 0. That is an accident of how it was
# written, not a property anyone chose, and it would evaporate the moment
# someone dropped the pipe. Guarded anyway: one construct, two call sites,
# and this branch has been bitten seventeen times by protecting one of a
# pair and not its twin.
# --- BEGIN db-decision -------------------------------------------------
# test_entrypoint_dbselect_wiring.py slices the bash BETWEEN these two
# markers and runs it for real. They are explicit because they used to be
# positional ("from the PG_TABLES line to the [3/3] banner"), and the
# moment the pg_ctl stop moved down here the test swallowed it, ran a
# postgres shutdown it had no business running, and failed every scenario
# at once for a reason that had nothing to do with the decision logic.
# Move code across these lines deliberately, or not at all.
PG_TABLES="$(_pg_public_tables)" || PG_TABLES=""
PG_TABLES="${PG_TABLES:-0}"
SQLITE_PRESENT="false"
SQLITE_CHATS="0"  # sqlite absent -> a KNOWN zero, not the unreadable/unknown case
if [ -f "${WEBUI_LOCAL_DB}" ]; then
    SQLITE_PRESENT="true"
    SQLITE_CHATS="$(_sqlite_chats "${WEBUI_LOCAL_DB}")" || SQLITE_CHATS=""
    # Empty output means the python probe itself crashed before printing —
    # same "couldn't determine" shape as the heredoc's own "unknown" case,
    # so it gets the same fail-safe treatment, never a silent "0".
    SQLITE_CHATS="${SQLITE_CHATS:-unknown}"
fi

eval "$(/opt/compactor-venv/bin/python /opt/compactor/dbselect.py \
    --pg-tables "${PG_TABLES}" \
    --sqlite-present "${SQLITE_PRESENT}" \
    --sqlite-chats "${SQLITE_CHATS}")"
# ^ sets DBSELECT_DATABASE, DBSELECT_SYNC_ENABLED, DBSELECT_MIGRATION_PENDING,
# DBSELECT_UNKNOWN_CHAT_COUNT. dbselect.py only ever emits those four fixed
# KEY=VALUE lines (see its CLI docstring) — nothing here is attacker- or
# runtime-controlled input to eval.

# eval() DISCARDS the exit status of its command substitution, so `set -e`
# cannot fire here: a missing dbselect.py, a broken venv or a traceback all
# leave DBSELECT_DATABASE unset, and the if/else below would then fall
# through to the POSTGRES branch - silently switching her onto the empty
# database, which is the single outcome this whole module exists to
# prevent. It fails OPEN, in the one direction that must fail closed.
#
# Refuse to boot instead. A container that will not start is a bad
# morning; a container that starts and shows her an empty history is
# worse, and harder to notice.
if [ -z "${DBSELECT_DATABASE:-}" ]; then
    echo "      ERROR: dbselect.py produced no decision. Refusing to guess"
    echo "             which database holds the chat history."
    echo "             Check: /opt/compactor-venv/bin/python /opt/compactor/dbselect.py --help"
    exit 1
fi

if [ "${DBSELECT_DATABASE}" = "sqlite" ]; then
    export DATABASE_URL="sqlite:///${WEBUI_LOCAL_DB}"
else
    export DATABASE_URL="postgresql://${POSTGRES_USER}@/${POSTGRES_DB}?host=${POSTGRES_SOCKET_DIR}"
fi
export WEBUIDB_SYNC_ENABLED="${DBSELECT_SYNC_ENABLED}"

if [ "${DBSELECT_UNKNOWN_CHAT_COUNT}" = "true" ]; then
    echo ""
    echo "      ============================================================"
    echo "      WARNING: could not determine webui.db's chat count."
    echo ""
    echo "      Postgres is EMPTY (0 public tables) and webui.db exists but"
    echo "      its chat count could not be read (locked, corrupt header, or"
    echo "      mid-write). Refusing to treat 'unknown' as 'empty' — still"
    echo "      serving SQLite rather than risk silently switching her to an"
    echo "      empty Postgres. Investigate ${WEBUI_LOCAL_DB} and check"
    echo "      ${LOG_DIR}/webuidb-sync.log before the next boot."
    echo "      ============================================================"
    echo ""
elif [ "${DBSELECT_MIGRATION_PENDING}" = "true" ]; then
    echo ""
    echo "      ============================================================"
    echo "      MIGRATION PENDING — still serving SQLite, on purpose."
    echo ""
    echo "      Postgres is up but EMPTY (0 public tables), while webui.db"
    echo "      holds ${SQLITE_CHATS} chat(s). Switching now would show her an"
    echo "      empty app. Nothing has been lost and nothing was changed."
    echo ""
    echo "      To complete the move, with someone watching:"
    echo "        1. Let OpenWebUI start once against Postgres so alembic"
    echo "           builds the schema:"
    echo "             supervisorctl stop openwebui"
    echo "             DATABASE_URL='postgresql://${POSTGRES_USER}@/${POSTGRES_DB}?host=${POSTGRES_SOCKET_DIR}' \\"
    echo "               /app/venv/bin/open-webui serve --port 3999   # ctrl-c once it is listening"
    echo "        2. Copy the rows (dry run first — it defaults to one):"
    echo "             /app/venv/bin/python /opt/compactor/migrate-webui-sqlite-to-pg.py"
    echo "             /app/venv/bin/python /opt/compactor/migrate-webui-sqlite-to-pg.py --apply"
    echo "        3. Restart the pod. This block will see the tables and hand"
    echo "           over to Postgres by itself."
    echo ""
    echo "      The SQLite source is never written by the migration, so a"
    echo "      failed attempt costs nothing but the time."
    echo "      ============================================================"
    echo ""
elif [ "${DBSELECT_DATABASE}" = "postgres" ] && [ "${PG_TABLES}" -gt 0 ]; then
    echo "      database: PostgreSQL (${PG_TABLES} public tables)"
    echo "      webuidb-sync: off — nothing writes webui.db, so snapshotting it"
    echo "                    would only keep writing to /data for no reason"
else
    echo "      database: PostgreSQL (fresh, empty — no SQLite history to carry over)"
fi

# =============================================================================
# --- END db-decision ---------------------------------------------------

# The temporary instance has done its work - supervisord owns Postgres now.
#
# THIS RUNS LAST, and the ordering IS the fix for a defect this block shipped
# with. The stop used to sit ~100 lines earlier, right after the restore,
# which put it BEFORE the "which database" probe below. That probe asks psql
# over the unix socket how many public tables exist; with the postmaster
# already stopped there is no socket, psql fails, stderr is discarded, stdout
# is empty, and ${PG_TABLES:-0} turns "could not ask" into a confident "0".
#
# It did not crash, which is what made it dangerous. The handover to Postgres
# could NEVER happen while webui.db held chats, so the migration instructions
# this script prints - "restart the pod, this block will see the tables and
# hand over by itself" - were false. An operator following them would migrate
# her rows into Postgres, come back up on SQLite, and keep writing to a
# database now diverging from the one holding the migration. Whoever later
# fixed the probe would flip her onto a Postgres frozen at migration time,
# stranding everything since.
#
# So: everything that needs to ASK Postgres a question goes above this line.
# =============================================================================
echo "      stopping the temporary Postgres instance (supervisord owns it from here)"
runuser -u postgres -- "${PG_BIN}/pg_ctl" -D "${PGDATA}" -m fast -w -t 60 stop \
    >>"${LOG_DIR}/postgres-initdb.log" 2>&1 || {
    echo "      temporary Postgres did not stop within 60s - escalating to -m immediate"
    runuser -u postgres -- "${PG_BIN}/pg_ctl" -D "${PGDATA}" -m immediate -w -t 30 stop \
        >>"${LOG_DIR}/postgres-initdb.log" 2>&1 || {
        echo "      ERROR: it is still holding ${PGDATA}."
        echo "             supervisord's postgres program cannot take the"
        echo "             postmaster.pid lock while that process lives, so it"
        echo "             would retry 3x, go FATAL, and OpenWebUI would come up"
        echo "             with NO DATABASE. Refusing to continue into that."
        echo "             Check ${LOG_DIR}/postgres-initdb.log and: ps aux | grep postgres"
        exit 1
    }
}

echo ""
echo "[3/3] Starting services..."
echo "      - vLLM             on port ${VLLM_PORT}      (internal)"
echo "      - PostgreSQL       on ${POSTGRES_SOCKET_DIR} (unix socket only, no TCP)"
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
