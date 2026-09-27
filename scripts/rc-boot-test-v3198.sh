#!/bin/bash
# rc-boot-test-v3198.sh — build + boot-test the v3.1.9.8 release candidate.
#
# WHAT THIS DOES, in order:
#   1. Builds local/zla:v3198-rc from Dockerfile.v3198 (must be run from a
#      `cp -a` copy of the repo, e.g. ~/zl-ops/v3198/rc-prep/build — NEVER
#      from the git worktree itself: a docker bind mount into an
#      already-checked-out tree has left root-owned files behind before).
#   2. Boots it TWICE on a SYNTHETIC, empty /data (a fresh tempdir each
#      time — never a real backup, never her real data): once with
#      WEBUI_DB_LOCAL=false, once with WEBUI_DB_LOCAL=true. No GPU is
#      required or used; a stub `nvidia-smi` is installed inside each
#      throwaway container so entrypoint.sh's preflight checks pass, but
#      real GPU work (vLLM) is untouched and WILL fail — that is EXPECTED
#      and is not treated as a failure below (see the vllm handling).
#   3. Asserts, on each boot: supervisord itself is up; compactor,
#      openwebui and backup reach RUNNING; zl-ops reaches EXITED (never
#      FATAL/BACKOFF) with its log's last line reading
#      "bootstrap.sh --boot: done" (ops/bootstrap.sh --boot is merged as of
#      0190911, so this is now a real assertion, not just "didn't crash");
#      the compactor tree under /opt/compactor is byte-identical (sha256)
#      to the CURRENT git tree (not just to what this one build happened to
#      embed — this is the test-time check Dockerfile.v3198 explicitly
#      defers to this script, see its own header comment); /opt/zl-repo/ops/zl
#      is executable; `import main` succeeds in the compactor venv.
#   4. Rollback: the WEBUI_DB_LOCAL=false /data directory (the one shaped
#      like what prod actually runs today) is then booted on BOTH the REAL
#      prod image, angreg/zions-light-ai:v3.1.9-cu12 (must already be
#      pulled locally — `docker images`), AND local/zla:v3198-rollback (the
#      `rollback` stage of this same Dockerfile: prod's image + the 0.11.4
#      OpenWebUI upgrade, nothing else) — each checked for fatal errors.
#      The previously-seen alembic "downgrade" warning is expected and
#      confirmed non-fatal on the plain prod image; its exact message is
#      reported (no content). local/zla:v3198-rollback already carries
#      0.11.4's own schema and should show NO alembic revision error.
#
# Logs everything under ~/zl-ops/v3198/rc-prep/ (per the owner's rule:
# build/test artifacts live there, never inside the worktree).
#
# Needs: a real Docker daemon, and either the RC build materials in the
# current directory (a `cp -a` of the repo) or ZLA_SKIP_BUILD=1 plus
# already-built local/zla:v3198-rc AND local/zla:v3198-rollback (the
# Dockerfile is multi-stage now; both targets are built by this script).
# SKIPS (exit 3) honestly if Docker is unavailable — never silently
# "passes" a suite that could not run.

set -u
REPO_COPY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_ROOT="${ZLA_RC_LOG_ROOT:-$HOME/zl-ops/v3198/rc-prep}"
RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="${LOG_ROOT}/boot-test-${RUN_TS}"
mkdir -p "${LOG_DIR}"
MAIN_LOG="${LOG_DIR}/rc-boot-test.log"

RC_IMAGE="local/zla:v3198-rc"
ROLLBACK_IMAGE="local/zla:v3198-rollback"
PROD_IMAGE="angreg/zions-light-ai:v3.1.9-cu12"
FAIL=0

log() { echo "$(date -u +%H:%M:%S) $*" | tee -a "${MAIN_LOG}"; }
fail() { log "FAIL: $*"; FAIL=1; }
ok() { log "  ok   $*"; }

_skip() {
    log "SKIPPED: $*"
    exit 3
}

command -v docker >/dev/null 2>&1 || _skip "no docker on PATH"
docker version >/dev/null 2>&1 || _skip "docker daemon unreachable"

# -----------------------------------------------------------------------
# 1. Build
# -----------------------------------------------------------------------
if [ "${ZLA_SKIP_BUILD:-0}" != "1" ]; then
    if [ "${REPO_COPY_DIR}" = "$(cd /home/drew/code/zl-v3198 2>/dev/null && pwd || echo __nope__)" ]; then
        fail "refusing to build: this looks like the git worktree itself, not a cp -a copy. Run from ~/zl-ops/v3198/rc-prep/build."
        exit 1
    fi
    # Dockerfile.v3198 is multi-stage now (rollback -> rc); build BOTH
    # targets, since task 5 boots local/zla:v3198-rollback directly, not
    # just as an intermediate layer of the rc build.
    log "Building ${ROLLBACK_IMAGE} (--target rollback) from Dockerfile.v3198 in ${REPO_COPY_DIR}"
    ( cd "${REPO_COPY_DIR}" && docker build --target rollback -f Dockerfile.v3198 -t "${ROLLBACK_IMAGE}" . ) \
        > "${LOG_DIR}/build-rollback.log" 2>&1
    if [ $? -ne 0 ]; then
        fail "docker build --target rollback failed — see ${LOG_DIR}/build-rollback.log"
        tail -60 "${LOG_DIR}/build-rollback.log" | tee -a "${MAIN_LOG}"
        exit 1
    fi
    ok "image built: ${ROLLBACK_IMAGE}"

    log "Building ${RC_IMAGE} (--target rc) from Dockerfile.v3198 in ${REPO_COPY_DIR}"
    ( cd "${REPO_COPY_DIR}" && docker build --target rc -f Dockerfile.v3198 -t "${RC_IMAGE}" . ) \
        > "${LOG_DIR}/build.log" 2>&1
    if [ $? -ne 0 ]; then
        fail "docker build --target rc failed — see ${LOG_DIR}/build.log"
        tail -60 "${LOG_DIR}/build.log" | tee -a "${MAIN_LOG}"
        exit 1
    fi
    ok "image built: ${RC_IMAGE}"
else
    docker image inspect "${RC_IMAGE}" >/dev/null 2>&1 || _skip "ZLA_SKIP_BUILD=1 but ${RC_IMAGE} is not built"
    docker image inspect "${ROLLBACK_IMAGE}" >/dev/null 2>&1 || _skip "ZLA_SKIP_BUILD=1 but ${ROLLBACK_IMAGE} is not built"
fi
IMAGE_ID="$(docker inspect --format='{{.Id}}' "${RC_IMAGE}")"
ROLLBACK_IMAGE_ID="$(docker inspect --format='{{.Id}}' "${ROLLBACK_IMAGE}")"
log "rc image id: ${IMAGE_ID}"
log "rollback image id: ${ROLLBACK_IMAGE_ID}"

FAKE_NVIDIA_SMI='#!/bin/bash
case "$*" in
  *"--query-gpu=name"*) echo "NVIDIA Fake-GPU (rc-boot-test, no real GPU)" ;;
  *"--query-gpu=memory.total"*) echo "24576" ;;
  *"--query-gpu=driver_version"*) echo "550.54.15" ;;
  *) exit 0 ;;
esac
'

# -----------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------
_dexec() { docker exec "$1" bash -c "$2"; }

_stop() { docker rm -f "$1" >/dev/null 2>&1 || true; }

_boot_container() {
    # $1 = image, $2 = data dir (host path), $3 = container name,
    # $4 = extra -e args (single string, word-split), $5 = log tag
    local image="$1" datadir="$2" name="$3" extra_env="$4" tag="$5"
    _stop "${name}"
    # shellcheck disable=SC2086
    docker run -d --name "${name}" \
        -v "${datadir}:/data" \
        --entrypoint bash \
        ${extra_env} \
        "${image}" -c "sleep 3600" > "${LOG_DIR}/${tag}-run.log" 2>&1
    if [ $? -ne 0 ]; then
        fail "docker run failed for ${tag} — see ${LOG_DIR}/${tag}-run.log"
        return 1
    fi
    sleep 1
    _dexec "${name}" "echo up" > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        fail "throwaway container ${name} is not actually running"
        return 1
    fi
    # Install the GPU-less stub so entrypoint.sh's preflight passes; real
    # vLLM GPU work is unaffected and still fails, as expected.
    docker exec -i "${name}" bash -c 'cat > /usr/local/bin/nvidia-smi && chmod +x /usr/local/bin/nvidia-smi' <<< "${FAKE_NVIDIA_SMI}"
    return 0
}

_run_entrypoint() {
    local name="$1" webui_db_local="$2" tag="$3"
    docker exec -d "${name}" bash -c "
        export WEBUI_DB_LOCAL=${webui_db_local}
        export STT_ENABLED=false TTS_ENABLED=false
        export MAX_MODEL_LEN=2048
        exec /entrypoint.sh > /tmp/${tag}-entrypoint.log 2>&1
    "
}

_wait_for_supervisord() {
    local name="$1" deadline=$((SECONDS + 90))
    while [ ${SECONDS} -lt ${deadline} ]; do
        if _dexec "${name}" "test -S /var/run/supervisor.sock" >/dev/null 2>&1; then
            return 0
        fi
        sleep 2
    done
    return 1
}

_supervisorctl() {
    _dexec "$1" "supervisorctl -c /etc/supervisor/conf.d/supervisord.conf status 2>&1 || true"
}

_wait_for_state() {
    # $1=name $2=program $3=space-separated acceptable states $4=timeout_s
    local name="$1" prog="$2" states="$3" timeout="${4:-120}" deadline
    deadline=$((SECONDS + timeout))
    while [ ${SECONDS} -lt ${deadline} ]; do
        local line status
        line="$(_supervisorctl "${name}" | grep "^${prog}[[:space:]]")"
        status="$(echo "${line}" | awk '{print $2}')"
        for s in ${states}; do
            [ "${status}" = "${s}" ] && { echo "${status}"; return 0; }
        done
        # FATAL/BACKOFF from a program we're waiting to reach RUNNING is a
        # real failure worth stopping early for, not just timing out.
        if [ "${status}" = "FATAL" ] && ! echo "${states}" | grep -qw FATAL; then
            echo "${status}"; return 1
        fi
        sleep 2
    done
    echo "${status:-MISSING}"
    return 1
}

_sha256_tree_check() {
    # Compares the built image's /opt/compactor/*.py (non-test) against
    # the CURRENT git tree at HEAD's worktree (the real repo, not this
    # cp -a copy, so a stale copy can't rubber-stamp itself) — this is the
    # test-time identity check Dockerfile.v3198 defers to here.
    local name="$1" src_dir="$2" mismatches=0
    for f in "${src_dir}"/compactor/*.py; do
        base="$(basename "$f")"
        case "$base" in test_*) continue ;; esac
        want="$(sha256sum "$f" | awk '{print $1}')"
        got="$(_dexec "${name}" "sha256sum /opt/compactor/${base} 2>/dev/null | awk '{print \$1}'")"
        if [ "${want}" != "${got}" ]; then
            fail "sha256 mismatch: compactor/${base} (tree=${want:0:12} image=${got:0:12})"
            mismatches=$((mismatches + 1))
        fi
    done
    if [ "${mismatches}" -eq 0 ]; then
        ok "all non-test compactor/*.py files are sha256-identical to the current tree"
    fi
}

# -----------------------------------------------------------------------
# 2/3. Boot + assert, for each WEBUI_DB_LOCAL setting
# -----------------------------------------------------------------------
DATA_FALSE="$(mktemp -d /tmp/zla-rc-data-false.XXXXXX)"
DATA_TRUE="$(mktemp -d /tmp/zla-rc-data-true.XXXXXX)"
chmod 777 "${DATA_FALSE}" "${DATA_TRUE}"

boot_and_check() {
    local webui_db_local="$1" datadir="$2" tag="$3"
    local name="zla-rc-boot-${tag}-$$"
    log "=== boot: WEBUI_DB_LOCAL=${webui_db_local} (tag=${tag}) ==="
    _boot_container "${RC_IMAGE}" "${datadir}" "${name}" "" "${tag}" || { _stop "${name}"; return 1; }
    _run_entrypoint "${name}" "${webui_db_local}" "${tag}"

    if ! _wait_for_supervisord "${name}"; then
        fail "supervisord never came up (${tag})"
        docker exec "${name}" bash -c "tail -100 /tmp/${tag}-entrypoint.log" | tee -a "${MAIN_LOG}"
        _stop "${name}"
        return 1
    fi
    ok "supervisord is running (${tag})"

    local st
    st="$(_wait_for_state "${name}" compactor "RUNNING" 90)"
    [ "${st}" = "RUNNING" ] && ok "compactor RUNNING (${tag})" || fail "compactor did not reach RUNNING (${tag}), saw ${st}"

    st="$(_wait_for_state "${name}" openwebui "RUNNING" 90)"
    [ "${st}" = "RUNNING" ] && ok "openwebui RUNNING (${tag})" || fail "openwebui did not reach RUNNING (${tag}), saw ${st}"

    st="$(_wait_for_state "${name}" backup "RUNNING" 60)"
    [ "${st}" = "RUNNING" ] && ok "backup RUNNING (${tag})" || fail "backup did not reach RUNNING (${tag}), saw ${st}"

    st="$(_wait_for_state "${name}" zl-ops "EXITED" 200)"
    if [ "${st}" = "EXITED" ]; then
        ok "zl-ops EXITED, not FATAL/BACKOFF (${tag})"
    else
        fail "zl-ops in unexpected state ${st} (${tag}); expected EXITED"
    fi
    # --boot always exits 0; whether it did its job is on the log's final line.
    last="$(docker exec "${name}" bash -c 'tail -1 "${LOG_DIR:-/data/logs}/zl-ops.log" 2>/dev/null' || true)"
    case "${last}" in
        *"bootstrap.sh --boot: done"*) ok "zl-ops.log ends with 'bootstrap.sh --boot: done' (${tag})" ;;
        *) fail "zl-ops.log's final line is not the --boot 'done' line (${tag}): ${last:0:200}" ;;
    esac
    if docker exec "${name}" bash -c '[ -L /usr/local/bin/zl ] && [ "$(readlink /usr/local/bin/zl)" = /opt/zl-repo/ops/zl ] && zl --help >/dev/null 2>&1'; then
        ok "/usr/local/bin/zl -> /opt/zl-repo/ops/zl and runs (${tag})"
    else
        fail "the zl symlink is missing or zl does not run (${tag})"
    fi
    docker exec "${name}" bash -c "tail -20 \${LOG_DIR:-/data/logs}/zl-ops.log 2>/dev/null || true; tail -20 \${LOG_DIR:-/data/logs}/zl-ops-supervisord.log 2>/dev/null || true" \
        > "${LOG_DIR}/${tag}-zl-ops.log" 2>&1
    log "  zl-ops log tail saved to ${LOG_DIR}/${tag}-zl-ops.log"

    log "vllm status (informational only, GPU-less failure is expected): $(_supervisorctl "${name}" | grep '^vllm')"

    _sha256_tree_check "${name}" "${REPO_COPY_DIR}"

    local ops_zl_mode
    ops_zl_mode="$(_dexec "${name}" "stat -c '%a' /opt/zl-repo/ops/zl 2>/dev/null")"
    [ "${ops_zl_mode}" = "755" ] && ok "ops/zl is executable in the image (mode ${ops_zl_mode}) (${tag})" \
        || fail "ops/zl mode is ${ops_zl_mode:-MISSING}, expected 755 (${tag})"

    local import_ok
    import_ok="$(_dexec "${name}" "cd /opt/compactor && /opt/compactor-venv/bin/python -c 'import main; print(\"OK\")' 2>&1")"
    case "${import_ok}" in
        *OK*) ok "/opt/compactor-venv/bin/python -c 'import main' succeeds (${tag})" ;;
        *) fail "'import main' failed (${tag}): ${import_ok}" ;;
    esac

    _stop "${name}"
}

boot_and_check false "${DATA_FALSE}" "db-false"
boot_and_check true "${DATA_TRUE}" "db-true"

# -----------------------------------------------------------------------
# 4. Rollback: the WEBUI_DB_LOCAL=false /data booted on BOTH the real prod
#    image and local/zla:v3198-rollback (same base, +0.11.4 OpenWebUI only).
# -----------------------------------------------------------------------
_rollback_boot_check() {
    # $1=image $2=tag $3=human label $4=datadir (both rollback boots reuse the
    # SAME touched /data, sequentially, per task 5 — the point is that each
    # image boots on the state the OTHER left behind, not on a private copy).
    # Because both boots share one persistent /data/logs/openwebui-error.log,
    # a check that just greps "the whole log file" after the SECOND boot
    # would still see the FIRST boot's traceback sitting there from before —
    # a false positive, not a new error from this boot. So every log check
    # below is scoped to the NEW bytes/lines this boot appended, measured
    # from a line-count snapshot taken (from the HOST side — the bind mount
    # target is world-readable, confirmed 2026-09-27) immediately before
    # this boot's entrypoint runs, never from the file's total content.
    local image="$1" tag="$2" label="$3" datadir="$4"
    log "=== rollback: booting ${label} (${image}) on the WEBUI_DB_LOCAL=false /data ==="
    if ! docker image inspect "${image}" >/dev/null 2>&1; then
        log "SKIPPED rollback (${tag}): ${image} is not pulled/built locally"
        return
    fi
    local err_log="${datadir}/logs/openwebui-error.log"
    local before_lines=0
    [ -f "${err_log}" ] && before_lines="$(wc -l < "${err_log}" 2>/dev/null || echo 0)"

    local name="zla-rc-${tag}-$$"
    _boot_container "${image}" "${datadir}" "${name}" "" "${tag}"
    docker exec -d "${name}" bash -c "
        export WEBUI_DB_LOCAL=false
        export STT_ENABLED=false TTS_ENABLED=false
        export MAX_MODEL_LEN=2048
        exec /entrypoint.sh > /tmp/${tag}-entrypoint.log 2>&1
    "
    if _wait_for_supervisord "${name}"; then
        ok "rollback (${tag}): supervisord came up on ${label} against the touched /data"
        st="$(_wait_for_state "${name}" openwebui "RUNNING" 90)"
        [ "${st}" = "RUNNING" ] && ok "rollback (${tag}): openwebui RUNNING" || fail "rollback (${tag}): openwebui did not reach RUNNING, saw ${st}"

        # NEW lines only (see the function comment above for why).
        tail -n "+$((before_lines + 1))" "${err_log}" 2>/dev/null > "${LOG_DIR}/${tag}-new-error-lines.log"
        grep -i -E 'alembic|revision|downgrade' "${LOG_DIR}/${tag}-new-error-lines.log" 2>/dev/null | tail -20 \
            > "${LOG_DIR}/${tag}-alembic.log"
        log "  alembic-related log lines (this boot's NEW lines only) saved to ${LOG_DIR}/${tag}-alembic.log"
        # Split each openwebui-error.log traceback into its own paragraph
        # (blank-line separated by python's own logging, one traceback per
        # exception) and flag only a BLOCK that does not itself mention
        # alembic/revision anywhere in it — a bare per-line grep -v alembic
        # wrongly flags the "Traceback (most recent call last):" header
        # line of the KNOWN alembic downgrade traceback, since that one
        # line alone never contains the word "alembic". Scoped to this
        # boot's new lines, same reason as above.
        awk 'BEGIN{RS="\n\n"} /Traceback/ && !/[Aa]lembic/ && !/revision/ {print; print "---"}' \
            "${LOG_DIR}/${tag}-new-error-lines.log" 2>/dev/null \
            > "${LOG_DIR}/${tag}-fatal-scan.log"
        if [ -s "${LOG_DIR}/${tag}-fatal-scan.log" ]; then
            fail "rollback (${tag}): possible non-alembic fatal errors — see ${LOG_DIR}/${tag}-fatal-scan.log"
        else
            ok "rollback (${tag}): no non-alembic fatal tracebacks found in the openwebui log"
        fi
        # local/zla:v3198-rollback already ships OpenWebUI 0.11.4, same as
        # whatever wrote this /data on the RC boot above — its schema
        # should already match, so unlike the plain (0.11.0) prod image,
        # NO alembic revision error is expected here (task 5).
        if [ "${tag}" = "onto-v3198-rollback" ]; then
            if grep -qi -E "Can't locate revision|CommandError" "${LOG_DIR}/${tag}-new-error-lines.log" 2>/dev/null; then
                fail "rollback (${tag}): expected NO alembic revision error onto local/zla:v3198-rollback, but found one in this boot's own new log lines — see ${LOG_DIR}/${tag}-new-error-lines.log"
            else
                ok "rollback (${tag}): no alembic revision error onto local/zla:v3198-rollback, as expected (same 0.11.4 schema)"
            fi
        fi
    else
        fail "rollback (${tag}): supervisord never came up on ${label}"
    fi
    _stop "${name}"
}

_rollback_boot_check "${PROD_IMAGE}" "onto-prod-image" "the REAL prod image" "${DATA_FALSE}"
_rollback_boot_check "${ROLLBACK_IMAGE}" "onto-v3198-rollback" "local/zla:v3198-rollback" "${DATA_FALSE}"

# The containers wrote as root inside these bind-mounted tempdirs, so a
# plain host `rm -rf` fails on their files; clean up via a throwaway
# root container against the SAME image instead of leaving them behind
# or reaching for sudo.
docker run --rm -v "${DATA_FALSE}:/d" --entrypoint bash "${RC_IMAGE}" -c 'rm -rf /d/*' >/dev/null 2>&1
docker run --rm -v "${DATA_TRUE}:/d" --entrypoint bash "${RC_IMAGE}" -c 'rm -rf /d/*' >/dev/null 2>&1
rm -rf "${DATA_FALSE}" "${DATA_TRUE}" 2>/dev/null || true

log "=== summary ==="
if [ "${FAIL}" -eq 0 ]; then
    log "ALL CHECKS PASSED. Logs: ${LOG_DIR}"
    exit 0
else
    log "ONE OR MORE CHECKS FAILED. Logs: ${LOG_DIR}"
    exit 1
fi
