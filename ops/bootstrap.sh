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
                     [--with-sshd] [--with-tools] [--no-login-hook]

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
  --no-login-hook    remove (or skip adding) the /root/.bashrc login hook
                      that runs `zl status --brief` on every interactive
                      SSH login.
EOF
}

# --help/-h must never have a side effect -- review LOW found it creating
# $ZL_HOME (the mkdir below used to run unconditionally, before argument
# parsing ever looked at -h/--help). Scan for it FIRST, before anything else.
for _arg in "$@"; do
    if [ "$_arg" = "-h" ] || [ "$_arg" = "--help" ]; then
        usage
        exit 0
    fi
done
unset _arg

# --- defaults, then saved state, then this run's own CLI flags (in that
# precedence order -- CLI always wins) ------------------------------------
REF=""
REPO_URL="$DEFAULT_REPO_URL"
FROM_DIR=""
WITH_SSHD=0
WITH_TOOLS=0
NO_LOGIN_HOOK=0

mkdir -p "$ZL_HOME"

REF_FILE="$ZL_HOME/ref"
FLAGS_FILE="$ZL_HOME/bootstrap-flags"

if [ -f "$REF_FILE" ]; then
    REF="$(cat "$REF_FILE")"
fi
if [ -f "$FLAGS_FILE" ]; then
    while IFS= read -r tok; do
        case "$tok" in
            __REPO_URL__=*) REPO_URL="${tok#__REPO_URL__=}" ;;
            --with-sshd) WITH_SSHD=1 ;;
            --with-tools) WITH_TOOLS=1 ;;
            --no-login-hook) NO_LOGIN_HOOK=1 ;;
        esac
    done < "$FLAGS_FILE"
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
        --no-login-hook) NO_LOGIN_HOOK=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "bootstrap.sh: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

# --- stage 1: are we the saved copy under $ZL_HOME, running after a
# restart? Relocate the checkout, then hand off to ITS bootstrap.sh. -------
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
ZL_HOME_COPY="$ZL_HOME/bootstrap.sh"

if [ "$SELF" = "$ZL_HOME_COPY" ] && [ "${ZL_BOOTSTRAP_REEXEC:-0}" != "1" ] && [ -z "$FROM_DIR" ]; then
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
    [ "$NO_LOGIN_HOOK" = 1 ] && ARGS+=(--no-login-hook)
    echo "bootstrap.sh: handing off to $REPO/ops/bootstrap.sh (the ref's own, possibly newer, copy)"
    export ZL_BOOTSTRAP_REEXEC=1
    exec bash "$REPO/ops/bootstrap.sh" "${ARGS[@]}"
fi

# --- stage 2: the real install, run from inside a real checkout -----------
SRC_DIR="${FROM_DIR:-$REPO}"
if [ ! -d "$SRC_DIR/ops" ]; then
    echo "bootstrap.sh: $SRC_DIR/ops not found -- is this a real checkout?" >&2
    exit 1
fi

ZL_BIN="${ZL_BIN:-/usr/local/bin/zl}"  # overridable only for this script's own test suite
ln -sf "$SRC_DIR/ops/zl" "$ZL_BIN"
# Best-effort only: SRC_DIR can be a read-only mount (the real-image test
# suite mounts the worktree read-only on purpose), and ops/zl is already
# committed with its exec bit set, so a failed chmod here is not fatal.
chmod +x "$SRC_DIR/ops/zl" 2>/dev/null || true
echo "bootstrap.sh: $ZL_BIN -> $SRC_DIR/ops/zl"

# review LOW: running the ALREADY-saved copy directly with --from-dir
# (which skips stage 1 outright, see its own `[ -z "$FROM_DIR" ]` guard)
# means $SELF and $ZL_HOME_COPY can be the exact same path -- `cp` refuses
# a same-file copy (exit 1: "are the same file"). Skip the copy in that
# case; there is nothing to update.
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
    [ "$NO_LOGIN_HOOK" = 1 ] && printf '%s\n' "--no-login-hook"
} > "$FLAGS_FILE"
echo "bootstrap.sh: saved ref/flags/self under $ZL_HOME"

BASHRC="${ZL_BASHRC:-/root/.bashrc}"  # overridable only for this script's own test suite
MARK_BEGIN="# BEGIN zl login hook (bootstrap.sh, idempotent) >>>"
MARK_END="# <<< END zl login hook"
touch "$BASHRC"
if grep -qF "$MARK_BEGIN" "$BASHRC" 2>/dev/null; then
    sed -i "/^${MARK_BEGIN//\//\\/}\$/,/^${MARK_END//\//\\/}\$/d" "$BASHRC"
fi
if [ "$NO_LOGIN_HOOK" = 1 ]; then
    echo "bootstrap.sh: removed the /root/.bashrc login hook (--no-login-hook)"
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
    } >> "$BASHRC"
    echo "bootstrap.sh: added the /root/.bashrc login hook (zl status --brief on interactive SSH login)"
fi

if [ "$WITH_SSHD" = 1 ]; then
    echo "bootstrap.sh: --with-sshd -> zl sshd --apply --yes"
    "$ZL_BIN" sshd --apply --yes || echo "bootstrap.sh: zl sshd --apply failed (see above) -- continuing"
fi
if [ "$WITH_TOOLS" = 1 ]; then
    echo "bootstrap.sh: --with-tools -> zl tools --apply --yes"
    "$ZL_BIN" tools --apply --yes || echo "bootstrap.sh: zl tools --apply failed (see above) -- continuing"
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
