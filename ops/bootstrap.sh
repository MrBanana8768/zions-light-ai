#!/usr/bin/env bash
# Installs (or re-installs, idempotently) the `zl` pod-ops bundle. See
# ops/README.md for the two one-liners this script is meant to be run with:
#
#   First time (RunPod Web Terminal):
#     git clone --depth 1 --branch REF <url> /opt/zl-repo && \
#         bash /opt/zl-repo/ops/bootstrap.sh --ref REF --with-sshd
#
#   After any restart (the container overlay -- and /opt -- reset every time):
#     bash /data/zl-ops/bootstrap.sh
#
# The second form works with NO arguments: this script saves its own ref and
# flags under /data/zl-ops (podenv.ZL_HOME) the first time, and a copy of
# itself there too, so a bare re-run after a restart replays the same setup.
# That saved copy's only job is to relocate (re-clone if needed) the real
# checkout and hand off to THAT clone's own ops/bootstrap.sh -- so a newer
# ref's bootstrap.sh always wins over a stale saved copy. ZL_BOOTSTRAP_REEXEC
# guards against re-exec'ing more than once even in a pathological setup
# where the saved copy and the checkout's copy resolve to the same path.
set -euo pipefail

ZL_HOME="${ZL_HOME:-/data/zl-ops}"
REPO="${ZL_REPO:-/opt/zl-repo}"
DEFAULT_REPO_URL="https://github.com/MrBanana8768/zions-light-ai.git"

usage() {
    cat <<'EOF'
Usage: bootstrap.sh [--ref REF] [--repo-url URL] [--from-dir DIR]
                     [--with-sshd] [--with-tools] [--tools-at-boot]
                     [--with-hotfix NAME] [--no-login-hook] [--boot]

  --ref REF          branch or TAG to install (required the first time;
                      after that, saved under $ZL_HOME/ref and reused). A
                      bare commit SHA does NOT work here: `git clone
                      --branch` (what this does, and what the first-time
                      one-liner in ops/README.md does by hand) only
                      accepts a branch/tag name. To land on an exact
                      commit, bootstrap with its branch/tag first, then
                      run `zl update --ref <sha>` (that command uses
                      `git fetch`, which does accept one).
  --repo-url URL     git remote to clone from (default: the project's own).
  --from-dir DIR      install from an existing checkout instead of cloning
                      (used by the real-image test suite on an uncommitted
                      worktree; never saved as a ref).
  --with-sshd        also run `zl sshd --apply --yes` once installed.
  --with-tools       also run `zl tools --apply --yes` once installed.
  --tools-at-boot    saved like --with-sshd/--with-tools; unlike
                      --with-tools (which is SKIPPED during `--boot` --
                      apt is slow and network-bound, not something a
                      one-shot boot-time program should risk by default),
                      this says --boot itself should also run `zl tools
                      --apply --yes`, bounded and LAST. No effect outside
                      `--boot`.
  --with-hotfix NAME also run `zl hotfix NAME --apply --yes` once
                      installed (currently just `backup-fast` -- see `zl
                      hotfix backup-fast --help`). Saved like --with-sshd;
                      runs LAST (after --with-tools/--tools-at-boot).
                      Under --boot this is bounded and fail-soft exactly
                      like --with-sshd -- `zl hotfix` itself already
                      detects "already applied" and returns fast, so a
                      .8+ image that ships the fix is simply a no-op here.
  --no-login-hook    remove (or skip adding) the /root/.bashrc login hook
                      that runs `zl status --brief` on every interactive
                      SSH login.
  --boot             non-interactive boot mode (v3.1.9.8+, run from
                      supervisord's own `[program:zl-ops]` -- see
                      ops/README.md): never prompts, never reads stdin,
                      does NO ref pinning at all -- always uses the baked
                      --from-dir copy as-is, no clone, ever. If an earlier
                      interactive bootstrap saved a ref, that is reported
                      as one informational log line only ("saved ref ...
                      not applied at boot"); to actually move to that ref,
                      update the image or run `zl update` by hand. --boot
                      only READS its saved flags (--with-sshd bounded/
                      fail-soft; --with-tools only if --tools-at-boot was
                      also saved, bounded, last) -- it never rewrites
                      $ZL_HOME/ref, $ZL_HOME/bootstrap-flags or its own
                      saved copy; those are only ever written by an
                      INTERACTIVE (non-boot) run. A timed-out read of
                      either saved file is logged distinctly from a
                      plain missing file, and the file itself is left
                      untouched either way. Logs to $LOG_DIR/zl-ops.log
                      (falls back to stderr if /data is unwritable/
                      stalled), and ALWAYS exits 0 within a fixed
                      wall-clock budget (ZL_BOOT_BUDGET_S, default 180s;
                      soft-bounded -- a few trailing bounded log writes
                      can push the real wall-clock time slightly past it)
                      -- see the module comment above _boot_trap below for
                      how.
EOF
}

BOOT=0
# --help/-h must never have a side effect -- review LOW found it creating
# $ZL_HOME (the mkdir below used to run unconditionally, before argument
# parsing ever looked at -h/--help). Scan for it FIRST, before anything else.
# --boot is scanned here too (not just in the main parsing loop below) so
# BOOT is already known before the very first $ZL_HOME touch a few lines
# down -- everything from there on can be bounded/best-effort for boot.
for _arg in "$@"; do
    if [ "$_arg" = "-h" ] || [ "$_arg" = "--help" ]; then
        usage
        exit 0
    fi
    if [ "$_arg" = "--boot" ]; then
        BOOT=1
    fi
done
unset _arg

# --- --boot: a fixed wall-clock budget and an unconditional exit-0 trap.
# `set -euo pipefail` above means ANY failing command (a stalled/failed
# /data write, a failed clone, an explicit `exit 1/2` elsewhere in this
# script) ends the script early -- this EXIT trap fires on every one of
# those paths (implicit via set -e, or an explicit `exit N`) and forces
# the actual exit code to 0, exactly once, only when --boot was given.
# This is the backstop; the individual boot-mode blocks below additionally
# use `timeout` (so a stall can't hang forever -- a trap cannot save you
# from a command that never returns) and explicit `|| true`/`if !` guards
# (so the script keeps going past an expected failure -- e.g. sshd apply
# failing must not skip the tools step after it -- a trap only decides
# the FINAL exit code, it does not resume execution past set -e).
BOOT_BUDGET_S="${ZL_BOOT_BUDGET_S:-180}"
BOOT_START_SECONDS=$SECONDS

_boot_budget_left() {
    local left=$(( BOOT_BUDGET_S - (SECONDS - BOOT_START_SECONDS) ))
    [ "$left" -lt 0 ] && left=0
    echo "$left"
}

_boot_log() {
    # BOOT=0: unchanged behaviour (plain stdout, as every existing caller
    # already expects -- the interactive tests below assert on this).
    # BOOT=1: timestamped, appended to $LOG_DIR/zl-ops.log, bounded so a
    # MooseFS stall can't hang the boot program forever, falling back to
    # stderr if that write fails or times out for any reason (missing
    # LOG_DIR, unwritable /data, ...).
    local msg="$1"
    if [ "$BOOT" != 1 ]; then
        echo "$msg"
        return 0
    fi
    local line dir
    line="$(date -u +'%Y-%m-%dT%H:%M:%SZ') $msg"
    dir="${LOG_DIR:-/data/logs}"
    if timeout -k 2 3 mkdir -p "$dir" >/dev/null 2>&1 && \
       timeout -k 2 3 tee -a "$dir/zl-ops.log" >/dev/null 2>&1 <<<"$line"; then
        return 0
    fi
    echo "$line" >&2
}

_BOOT_ABORT_LINE=0
_BOOT_ABORT_CMD=""

_boot_trap() {
    local ec=$?
    if [ "$BOOT" = 1 ]; then
        if [ "$ec" -ne 0 ]; then
            _boot_log "bootstrap.sh --boot: aborted at line ${_BOOT_ABORT_LINE:-0} (exit $ec)${_BOOT_ABORT_CMD:+ -- \$BASH_COMMAND: $_BOOT_ABORT_CMD}"
        else
            _boot_log "bootstrap.sh --boot: done -- zl: ${ZL_BIN:-?} -> ${SRC_DIR:-?}/ops/zl"
        fi
    fi
    exit 0
}
if [ "$BOOT" = 1 ]; then
    # An ERR trap only fires for a command whose failure is NOT already
    # being tested (part of `if`/`while`/`||`/`&&`/`!`) -- exactly the
    # commands that would otherwise abort the script under `set -e` with
    # no record of where. Capture $LINENO/$BASH_COMMAND here so the EXIT
    # trap above can log "aborted at line N" instead of a bare exit code.
    trap '_BOOT_ABORT_LINE=$LINENO; _BOOT_ABORT_CMD=$BASH_COMMAND' ERR
    trap _boot_trap EXIT
fi

# --- defaults, then saved state, then this run's own CLI flags (in that
# precedence order -- CLI always wins) ------------------------------------
REF=""
REPO_URL="$DEFAULT_REPO_URL"
FROM_DIR=""
WITH_SSHD=0
WITH_TOOLS=0
TOOLS_AT_BOOT=0
NO_LOGIN_HOOK=0
WITH_HOTFIX=""

REF_FILE="$ZL_HOME/ref"
FLAGS_FILE="$ZL_HOME/bootstrap-flags"

if [ "$BOOT" = 1 ]; then
    # H1: --boot only READS saved state; it never rewrites $ZL_HOME/ref,
    # $ZL_HOME/bootstrap-flags or its own saved copy (that only ever
    # happens on an interactive, non-boot run -- see the `else` branch
    # below, and the complete absence of any write to these paths anywhere
    # else in the BOOT=1 path of this script). Every /data access here is
    # also time-bounded (`timeout -k 2 Ns`: TERM first, KILL 2s later if
    # the child ignores TERM -- a plain `timeout N` alone can hang forever
    # waiting to reap a child stuck on a genuine MooseFS stall) -- /data
    # may be missing or a stalled MooseFS mount at boot, and this must
    # never hang the boot program forever. A failure here is NOT fatal to
    # the rest of boot (the zl symlink and login hook below live on the
    # container overlay, not /data) -- best-effort only.
    #
    # A TIMED-OUT read (exit 124 from TERM, 137 if KILL was needed) is
    # distinguished from a plain ENOENT/missing-file read (`cat`'s own
    # exit 1, already discarded by 2>/dev/null): only the timeout case is
    # logged -- a merely-missing file (the ordinary "never installed
    # interactively yet" case) is not noteworthy.
    timeout -k 2 5 mkdir -p "$ZL_HOME" >/dev/null 2>&1 || true
    _ref_rc=0
    REF_READ="$(timeout -k 2 5 cat "$REF_FILE" 2>/dev/null)" || _ref_rc=$?
    if [ "$_ref_rc" -eq 0 ]; then
        REF="$REF_READ"
    elif [ "$_ref_rc" -eq 124 ] || [ "$_ref_rc" -eq 137 ]; then
        _boot_log "bootstrap.sh --boot: could not read saved ref within 5s; using none this boot; the file is left untouched"
    fi
    _flags_rc=0
    FLAGS_READ="$(timeout -k 2 5 cat "$FLAGS_FILE" 2>/dev/null)" || _flags_rc=$?
    if [ "$_flags_rc" -eq 0 ]; then
        while IFS= read -r tok; do
            case "$tok" in
                __REPO_URL__=*) REPO_URL="${tok#__REPO_URL__=}" ;;
                --with-sshd) WITH_SSHD=1 ;;
                --with-tools) WITH_TOOLS=1 ;;
                --tools-at-boot) TOOLS_AT_BOOT=1 ;;
                --no-login-hook) NO_LOGIN_HOOK=1 ;;
                --with-hotfix=*) WITH_HOTFIX="${tok#--with-hotfix=}" ;;
            esac
        done <<<"$FLAGS_READ"
    elif [ "$_flags_rc" -eq 124 ] || [ "$_flags_rc" -eq 137 ]; then
        _boot_log "bootstrap.sh --boot: could not read saved flags within 5s; using none this boot; the file is left untouched"
    fi
    unset _ref_rc _flags_rc
else
    mkdir -p "$ZL_HOME"
    if [ -f "$REF_FILE" ]; then
        REF="$(cat "$REF_FILE")"
    fi
    if [ -f "$FLAGS_FILE" ]; then
        while IFS= read -r tok; do
            case "$tok" in
                __REPO_URL__=*) REPO_URL="${tok#__REPO_URL__=}" ;;
                --with-sshd) WITH_SSHD=1 ;;
                --with-tools) WITH_TOOLS=1 ;;
                --tools-at-boot) TOOLS_AT_BOOT=1 ;;
                --no-login-hook) NO_LOGIN_HOOK=1 ;;
                --with-hotfix=*) WITH_HOTFIX="${tok#--with-hotfix=}" ;;
            esac
        done < "$FLAGS_FILE"
    fi
fi

# A flag that takes a value but got none (the last argv token) must exit 2
# with a clear message -- not `set -u`'s own "$2: unbound variable" crash
# (review LOW).
_require_value() {
    if [ $# -lt 2 ]; then
        echo "bootstrap.sh: $1 requires a value" >&2
        usage >&2
        exit 2
    fi
}

while [ $# -gt 0 ]; do
    case "$1" in
        --ref) _require_value "$@"; REF="$2"; shift 2 ;;
        --ref=*) REF="${1#--ref=}"; shift ;;
        --repo-url) _require_value "$@"; REPO_URL="$2"; shift 2 ;;
        --repo-url=*) REPO_URL="${1#--repo-url=}"; shift ;;
        --from-dir) _require_value "$@"; FROM_DIR="$2"; shift 2 ;;
        --from-dir=*) FROM_DIR="${1#--from-dir=}"; shift ;;
        --with-sshd) WITH_SSHD=1; shift ;;
        --with-tools) WITH_TOOLS=1; shift ;;
        --tools-at-boot) TOOLS_AT_BOOT=1; shift ;;
        --no-login-hook) NO_LOGIN_HOOK=1; shift ;;
        --with-hotfix) _require_value "$@"; WITH_HOTFIX="$2"; shift 2 ;;
        --with-hotfix=*) WITH_HOTFIX="${1#--with-hotfix=}"; shift ;;
        --boot) BOOT=1; shift ;;  # already scanned for above; accepted here too so it is not "unknown"
        -h|--help) usage; exit 0 ;;
        *) echo "bootstrap.sh: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

# --- stage 1: are we the saved copy under $ZL_HOME, running after a
# restart? Relocate the checkout, then hand off to ITS bootstrap.sh. -------
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
ZL_HOME_COPY="$ZL_HOME/bootstrap.sh"

if [ "$SELF" = "$ZL_HOME_COPY" ] && [ "${ZL_BOOTSTRAP_REEXEC:-0}" != "1" ] && [ -z "$FROM_DIR" ]; then
    # M2: on v3.1.9.8+, $REPO is a BAKED image copy (the Dockerfile puts
    # ops/+scripts/ at /opt/zl-repo directly -- no .git at all, not a git
    # checkout). That is not "missing" -- never move it aside, never clone
    # over it. Hand off to it as-is with --from-dir (which also means REF
    # is moot here and deliberately dropped; only the saved flags carry
    # over) before the "no ref saved" / "$REPO/.git missing" checks below
    # (which are for a genuinely OLDER, pre-.8 image where $REPO really is
    # a plain git clone) ever run.
    if [ -e "$REPO/ops/bootstrap.sh" ] && [ ! -e "$REPO/.git" ]; then
        ARGS=(--from-dir "$REPO" --repo-url "$REPO_URL")
        [ "$WITH_SSHD" = 1 ] && ARGS+=(--with-sshd)
        [ "$WITH_TOOLS" = 1 ] && ARGS+=(--with-tools)
        [ "$TOOLS_AT_BOOT" = 1 ] && ARGS+=(--tools-at-boot)
        [ "$NO_LOGIN_HOOK" = 1 ] && ARGS+=(--no-login-hook)
        [ -n "$WITH_HOTFIX" ] && ARGS+=(--with-hotfix "$WITH_HOTFIX")
        echo "bootstrap.sh: $REPO is a baked image copy (no .git) -- using it as-is, no clone"
        echo "bootstrap.sh: handing off to $REPO/ops/bootstrap.sh (the baked copy)"
        export ZL_BOOTSTRAP_REEXEC=1
        exec bash "$REPO/ops/bootstrap.sh" "${ARGS[@]}"
    fi
    if [ -z "$REF" ]; then
        echo "bootstrap.sh: no --ref given and none saved under $REF_FILE" >&2
        exit 1
    fi
    # NOTE: in a git WORKTREE (not a plain clone), ".git" is a FILE, not a
    # directory -- test with -e, never -d, or a worktree checkout looks
    # "missing" every time and gets destructively re-cloned over itself.
    if [ ! -e "$REPO/.git" ]; then
        if [ -e "$REPO" ]; then
            # $REPO exists but is not a git checkout -- never `rm -rf` it
            # outright (review MEDIUM: reproduced with $ZL_REPO pointing at
            # a directory holding real files and no network -- the
            # directory was deleted, then the clone failed, losing both).
            # An EMPTY directory is safe to remove (mkdir -p leftovers,
            # e.g.); anything else is moved aside instead, never destroyed.
            if [ -d "$REPO" ] && [ -z "$(ls -A "$REPO" 2>/dev/null)" ]; then
                rmdir "$REPO"
            else
                ASIDE="${REPO}.aside-$(date -u +%Y%m%dT%H%M%SZ)"
                echo "bootstrap.sh: $REPO exists and is not empty/a git checkout -- moving it aside to $ASIDE" >&2
                mv "$REPO" "$ASIDE"
            fi
        fi
        echo "bootstrap.sh: $REPO missing (container overlay reset) -- re-cloning $REF"
        git clone --depth 1 --branch "$REF" "$REPO_URL" "$REPO"
    fi
    ARGS=(--ref "$REF" --repo-url "$REPO_URL")
    [ "$WITH_SSHD" = 1 ] && ARGS+=(--with-sshd)
    [ "$WITH_TOOLS" = 1 ] && ARGS+=(--with-tools)
    [ "$TOOLS_AT_BOOT" = 1 ] && ARGS+=(--tools-at-boot)
    [ "$NO_LOGIN_HOOK" = 1 ] && ARGS+=(--no-login-hook)
    [ -n "$WITH_HOTFIX" ] && ARGS+=(--with-hotfix "$WITH_HOTFIX")
    echo "bootstrap.sh: handing off to $REPO/ops/bootstrap.sh (the ref's own, possibly newer, copy)"
    export ZL_BOOTSTRAP_REEXEC=1
    exec bash "$REPO/ops/bootstrap.sh" "${ARGS[@]}"
fi

# --- stage 2: the real install, run from inside a real checkout -----------
SRC_DIR="${FROM_DIR:-$REPO}"

# DECISION: --boot does NOT do ref pinning. There is no clone at boot,
# ever -- SRC_DIR is always the baked --from-dir copy supervisord already
# pointed us at. This also removes H2 (the previous unbounded /data
# rm/mv/[ -d ] dance around a "boot-clone" directory) outright: there is
# nothing left here that touches $ZL_HOME/boot-clone at all. If an earlier
# INTERACTIVE bootstrap saved a ref, that is surfaced as one informational
# log line only -- never acted on during --boot.
if [ "$BOOT" = 1 ] && [ -n "$REF" ]; then
    _boot_log "saved ref '$REF' not applied at boot: the baked image copy is used (update the image, or run \`zl update\` by hand)"
fi

if [ ! -d "$SRC_DIR/ops" ]; then
    if [ "$BOOT" = 1 ]; then
        _boot_log "bootstrap.sh --boot: $SRC_DIR/ops not found -- nothing more this boot can do"
        exit 0
    fi
    echo "bootstrap.sh: $SRC_DIR/ops not found -- is this a real checkout?" >&2
    exit 1
fi

ZL_BIN="${ZL_BIN:-/usr/local/bin/zl}"  # overridable only for this script's own test suite
ln -sf "$SRC_DIR/ops/zl" "$ZL_BIN"
# Best-effort only: SRC_DIR can be a read-only mount (the real-image test
# suite mounts the worktree read-only on purpose), and ops/zl is already
# committed with its exec bit set, so a failed chmod here is not fatal.
chmod +x "$SRC_DIR/ops/zl" 2>/dev/null || true
_boot_log "bootstrap.sh: $ZL_BIN -> $SRC_DIR/ops/zl"

# H1: --boot NEVER writes $ZL_HOME/ref, $ZL_HOME/bootstrap-flags or its own
# saved copy -- boot only READS saved state (above); rewriting any of it
# here used to happen unconditionally (wrapped in `timeout`, but still a
# write, every single boot) even though --boot takes no new flags to save
# in the first place -- a stalled write there could also corrupt a
# perfectly good saved file with a truncated/partial one. Saving ref/
# flags/self only ever happens on an INTERACTIVE (non-boot) run, below.
if [ "$BOOT" = 1 ]; then
    _boot_log "bootstrap.sh --boot: read-only boot -- ref/flags/self under $ZL_HOME are left untouched"
else
    if [ "$SELF" != "$ZL_HOME_COPY" ]; then
        cp "$SELF" "$ZL_HOME_COPY"
    fi
    chmod +x "$ZL_HOME_COPY"
    if [ -n "$REF" ]; then
        printf '%s\n' "$REF" > "$REF_FILE"
    fi
    {
        printf '__REPO_URL__=%s\n' "$REPO_URL"
        [ "$WITH_SSHD" = 1 ] && printf '%s\n' "--with-sshd"
        [ "$WITH_TOOLS" = 1 ] && printf '%s\n' "--with-tools"
        [ "$TOOLS_AT_BOOT" = 1 ] && printf '%s\n' "--tools-at-boot"
        [ "$NO_LOGIN_HOOK" = 1 ] && printf '%s\n' "--no-login-hook"
        [ -n "$WITH_HOTFIX" ] && printf -- "--with-hotfix=%s\n" "$WITH_HOTFIX"
    } > "$FLAGS_FILE"
    echo "bootstrap.sh: saved ref/flags/self under $ZL_HOME"
fi

BASHRC="${ZL_BASHRC:-/root/.bashrc}"  # overridable only for this script's own test suite
MARK_BEGIN="# BEGIN zl login hook (bootstrap.sh, idempotent) >>>"
MARK_END="# <<< END zl login hook"

# M3: isolated in its own function, at boot called through `||` so a
# failure here (e.g. an unwritable/missing $BASHRC) does not skip the sshd
# step below. NOTE (bash gotcha): calling a function as the tested command
# of `||`/`if`/`&&`/`!` disables `set -e`'s automatic abort-on-failure for
# EVERY command inside that call, not just the top-level one -- a failing
# `touch`/`sed`/redirection inside would otherwise be silently skipped past
# without ever making the function itself return nonzero. So every
# fallible step below is explicitly checked with `|| return 1` instead of
# relying on `set -e` to do it -- that is what makes the failure actually
# reach and log at the call site below.
_do_login_hook_block() {
    touch "$BASHRC" || return 1
    if grep -qF "$MARK_BEGIN" "$BASHRC" 2>/dev/null; then
        sed -i "/^${MARK_BEGIN//\//\\/}\$/,/^${MARK_END//\//\\/}\$/d" "$BASHRC" || return 1
    fi
    if [ "$NO_LOGIN_HOOK" = 1 ]; then
        _boot_log "bootstrap.sh: removed the /root/.bashrc login hook (--no-login-hook)"
    else
        {
            echo "$MARK_BEGIN"
            echo 'case "$-" in *i*)'
            echo '    if command -v timeout >/dev/null 2>&1; then'
            # -k 2 5: send TERM at 5s, then KILL 2s later if TERM alone did not
            # work. A plain `timeout 5` sends only TERM once -- a login stuck on
            # a MooseFS stall (the whole reason this needs a timeout) will not
            # end on TERM either, and `timeout` itself then just waits forever
            # for the child to reap (review MEDIUM: "the login hook can hang an
            # interactive login").
            echo '        timeout -k 2 5 zl status --brief || true'
            echo '    else'
            echo '        zl status --brief || true'
            echo '    fi'
            echo '    ;;'
            echo 'esac'
            echo "$MARK_END"
        } >> "$BASHRC" || return 1
        _boot_log "bootstrap.sh: added the /root/.bashrc login hook (zl status --brief on interactive SSH login)"
    fi
}

if [ "$BOOT" = 1 ]; then
    _do_login_hook_block || _boot_log "bootstrap.sh --boot: login-hook step failed -- continuing (the sshd step below still runs)"
else
    _do_login_hook_block
fi

_SSHD_BOOT_TIMEOUT_S=90
_TOOLS_BOOT_TIMEOUT_S=90
_BOOT_STEP_MIN_S=10  # skip a step outright if less than this is left

# M3: the boot-mode --with-sshd/--with-tools calls below used to discard
# `zl sshd`/`zl tools`'s own output entirely (>/dev/null 2>&1). Capture it
# to a bounded temp file on the CONTAINER OVERLAY (never /data -- a stalled
# /data must not block writing this) and log the exit code plus its last
# ~5 lines on failure, so a boot-time sshd/tools failure is actually
# diagnosable from $LOG_DIR/zl-ops.log alone. `timeout -k 2` throughout:
# TERM first, KILL 2s later if the child ignores it.
_boot_log_tail() {
    local f="$1" ln
    while IFS= read -r ln; do
        _boot_log "    $ln"
    done < <(tail -n 5 "$f" 2>/dev/null)
}

_boot_run_bounded_step() {
    # $1=label (for messages) $2=timeout-seconds; the command itself is
    # "$ZL_BIN" "$@" (from $3 on).
    local label="$1" step_t="$2" out rc
    shift 2
    out="$(mktemp "${TMPDIR:-/tmp}/zl-boot-${label}.XXXXXX" 2>/dev/null || echo "${TMPDIR:-/tmp}/zl-boot-${label}.$$")"
    rc=0
    timeout -k 2 "$step_t" "$ZL_BIN" "$@" </dev/null >"$out" 2>&1 || rc=$?
    if [ "$rc" -ne 0 ]; then
        if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
            _boot_log "bootstrap.sh --boot: zl $label timed out (exit $rc) -- continuing"
        else
            _boot_log "bootstrap.sh --boot: zl $label failed (exit $rc) -- continuing"
        fi
        _boot_log "bootstrap.sh --boot: zl $label -- last output:"
        _boot_log_tail "$out"
    fi
    rm -f "$out" 2>/dev/null || true
    return "$rc"
}

if [ "$WITH_SSHD" = 1 ]; then
    if [ "$BOOT" = 1 ]; then
        # setup-sshd.py needs `apt-get update` and the network -- neither
        # is guaranteed at boot. Bounded (never blocks vllm/compactor/
        # openwebui coming up) and fail-soft (logged, never retried in a
        # loop -- SSH just stays down until the next restart/attempt).
        _left=$(_boot_budget_left)
        if [ "$_left" -gt "$_BOOT_STEP_MIN_S" ]; then
            _step_t=$_SSHD_BOOT_TIMEOUT_S
            [ "$_left" -lt "$_step_t" ] && _step_t=$_left
            _boot_log "bootstrap.sh --boot: --with-sshd -> zl sshd --apply --yes (bounded ${_step_t}s)"
            _boot_run_bounded_step "sshd" "$_step_t" sshd --apply --yes || \
                _boot_log "bootstrap.sh --boot: SSH stays down until the next attempt"
        else
            _boot_log "bootstrap.sh --boot: only ${_left}s left of the boot budget -- skipping --with-sshd this run"
        fi
    else
        echo "bootstrap.sh: --with-sshd -> zl sshd --apply --yes"
        "$ZL_BIN" sshd --apply --yes || echo "bootstrap.sh: zl sshd --apply failed (see above) -- continuing"
    fi
fi

# --with-tools is apt-bound like --with-sshd, but MUCH less urgent (tmux/
# htop/... are conveniences, not "get SSH back") -- skipped at boot
# entirely unless --tools-at-boot was ALSO explicitly saved, and even then
# it runs LAST (this block is physically after the sshd block above).
_RUN_TOOLS_NOW=0
if [ "$BOOT" = 1 ]; then
    [ "$TOOLS_AT_BOOT" = 1 ] && _RUN_TOOLS_NOW=1
else
    [ "$WITH_TOOLS" = 1 ] && _RUN_TOOLS_NOW=1
fi
if [ "$_RUN_TOOLS_NOW" = 1 ]; then
    if [ "$BOOT" = 1 ]; then
        _left=$(_boot_budget_left)
        if [ "$_left" -gt "$_BOOT_STEP_MIN_S" ]; then
            _step_t=$_TOOLS_BOOT_TIMEOUT_S
            [ "$_left" -lt "$_step_t" ] && _step_t=$_left
            _boot_log "bootstrap.sh --boot: --tools-at-boot -> zl tools --apply --yes (bounded ${_step_t}s, last)"
            _boot_run_bounded_step "tools" "$_step_t" tools --apply --yes || true
        else
            _boot_log "bootstrap.sh --boot: only ${_left}s left of the boot budget -- skipping --tools-at-boot this run"
        fi
    else
        echo "bootstrap.sh: --with-tools -> zl tools --apply --yes"
        "$ZL_BIN" tools --apply --yes || echo "bootstrap.sh: zl tools --apply failed (see above) -- continuing"
    fi
fi

# --with-hotfix NAME: a pod-only, no-image-change patch (`zl hotfix NAME
# --apply --yes` -- see ops/zlops/commands/hotfix.py). Runs LAST, same as
# --tools-at-boot, and is just as bounded/fail-soft at boot: `zl hotfix`
# itself already detects "already applied" and returns fast (a genuine
# .8+ image that ships the fix already is a no-op here), so no special
# case is needed for that -- only the generic bound/fail-soft wrapping is.
_HOTFIX_BOOT_TIMEOUT_S=60
if [ -n "$WITH_HOTFIX" ]; then
    if [ "$BOOT" = 1 ]; then
        _left=$(_boot_budget_left)
        if [ "$_left" -gt "$_BOOT_STEP_MIN_S" ]; then
            _step_t=$_HOTFIX_BOOT_TIMEOUT_S
            [ "$_left" -lt "$_step_t" ] && _step_t=$_left
            _boot_log "bootstrap.sh --boot: --with-hotfix $WITH_HOTFIX -> zl hotfix $WITH_HOTFIX --apply --yes (bounded ${_step_t}s, last)"
            _boot_run_bounded_step "hotfix" "$_step_t" hotfix "$WITH_HOTFIX" --apply --yes || true
        else
            _boot_log "bootstrap.sh --boot: only ${_left}s left of the boot budget -- skipping --with-hotfix this run"
        fi
    else
        echo "bootstrap.sh: --with-hotfix $WITH_HOTFIX -> zl hotfix $WITH_HOTFIX --apply --yes"
        "$ZL_BIN" hotfix "$WITH_HOTFIX" --apply --yes || \
            echo "bootstrap.sh: zl hotfix $WITH_HOTFIX --apply failed (see above) -- continuing"
    fi
fi

if [ "$BOOT" = 1 ]; then
    # L2: the budget above is soft-bounded -- it governs the sshd/tools
    # steps only. A handful of trailing bounded log writes (this one, plus
    # whatever _boot_trap logs on the way out) can each still take up to
    # ~7s (timeout -k 2 5s + the 2s kill grace) beyond BOOT_BUDGET_S in the
    # worst case (a genuinely stalled /data at the very end). The real
    # wall-clock bound documented in ops/README.md is therefore
    # BOOT_BUDGET_S plus a small constant slop for these trailing writes,
    # not a hard ceiling.
    exit 0
fi

cat <<EOF

bootstrap.sh done.
  ref:      ${REF:-"(none -- --from-dir install)"}
  repo:     $SRC_DIR
  zl:       $ZL_BIN -> $SRC_DIR/ops/zl
  ZL_HOME:  $ZL_HOME

Next:
  zl help          -- every command, one line each
  zl status        -- is it healthy right now
  zl dash          -- live terminal telemetry
After a restart:
  bash $ZL_HOME_COPY
EOF
