#!/bin/sh
# Runs the L1 store's unit suite (frontend/tests/store/**) against the
# Postgres 16 sidecar. Mirrors testfixtures/unit-suite/entrypoint.sh's
# shape: copy the read-only bind mount to a writable scratch tree first —
# not ceremony. dist-store-tests/ (compiled output) and npm's own caches
# get written next to the code; doing that on the Windows bind mount
# directly is slow across that filesystem boundary and leaves build
# artefacts in the developer's working tree, where the next `git status`
# reads them as changes. /src stays read-only.
set -e

if [ ! -d /src ]; then
    echo "no /src mount: run this through docker-compose.tests.yml" >&2
    exit 2
fi

mkdir -p /repo
tar -C /src \
    --exclude=./.git \
    --exclude=./data \
    --exclude=./frontend/node_modules \
    --exclude=./frontend/.svelte-kit \
    --exclude=./frontend/build \
    --exclude=./frontend/dist-store-tests \
    -cf - . | tar -C /repo -xf -

# node_modules was installed at IMAGE BUILD time (frontend/package*.json
# only, so it caches across source edits) and lives at /opt/client — the
# copy above deliberately excluded frontend/node_modules, so link it back
# in rather than reinstalling (which would need network at container run
# time, defeating the point of installing it at build time at all).
ln -s /opt/client/node_modules /repo/frontend/node_modules

cd /repo/frontend

# Bounded wait for the Postgres sidecar — never an unbounded one. See
# wait-for-pg.cjs for what "ready" means here and why an unreachable
# sidecar is SKIP (exit 3), not FAIL.
node /opt/client/wait-for-pg.cjs
WAIT_STATUS=$?
if [ "$WAIT_STATUS" -ne 0 ]; then
    exit "$WAIT_STATUS"
fi

# --only <substring> maps to node's own --test-name-pattern (a regex),
# mirroring scripts/run-tests.py's --only (a substring filter) as closely
# as node's test runner allows. Everything else is passed through
# unrecognised and rejected loudly rather than silently ignored.
ONLY_PATTERN=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --only)
            shift
            ONLY_PATTERN="$1"
            ;;
        *)
            echo "unrecognised argument: $1 (only --only <substring> is supported)" >&2
            exit 2
            ;;
    esac
    shift
done

npm run build:store-tests

TEST_FILES=$(find dist-store-tests/tests/store -name '*.test.js')
if [ -z "$TEST_FILES" ]; then
    echo "SKIPPED: no compiled test files found under dist-store-tests/tests/store" >&2
    exit 3
fi

if [ -n "$ONLY_PATTERN" ]; then
    # Escape regex metacharacters so a literal substring behaves like
    # run-tests.py's --only rather than being interpreted as a pattern.
    ESCAPED=$(printf '%s' "$ONLY_PATTERN" | sed 's/[.[\*^$()+?{|\\]/\\&/g')

    # A KNOWN, WATCHED-TO-FAIL gap this file's first version had: when
    # --test-name-pattern matches ZERO tests inside a file, node's test
    # runner substitutes the FILE ITSELF as a trivially-passing top-level
    # test — so `--only <typo>` reported "N passed" with N == the file
    # count and ZERO actual assertions run. Never fold a skip into a pass
    # (this project's own rule): grep the COMPILED test files for the
    # literal substring first — every test() title is a plain string
    # literal there — and treat "the pattern appears in no test title" as
    # SKIP (exit 3), before node ever gets a chance to report a false PASS.
    if ! grep -qFl -- "$ONLY_PATTERN" $TEST_FILES; then
        echo "SKIPPED: --only '$ONLY_PATTERN' matches no test title in $TEST_FILES" >&2
        exit 3
    fi

    exec node --test --test-concurrency=1 --test-name-pattern="$ESCAPED" $TEST_FILES
else
    # --test-concurrency=1: node's test runner otherwise runs separate
    # FILES concurrently by default. F7's cost measurements read Postgres's
    # GLOBAL WAL insert pointer (pg_current_wal_lsn()) around one call —
    # accurate only if no OTHER backend's writes land inside that window.
    # Every OTHER file in this suite uses the SAME Postgres sidecar (each
    # in its own schema, but WAL is cluster-wide, not per-schema), so
    # concurrent files reliably polluted F7's measurement: the very first
    # version of this suite passed 6/6 isolated and then failed exactly
    # F7's ratio assertion (5.04x against a 5x bound) the first time it ran
    # alongside the other four files. Forcing strict sequential execution
    # is the fix, at a negligible cost given this suite's size (a few
    # seconds either way).
    exec node --test --test-concurrency=1 $TEST_FILES
fi
