#!/bin/sh
# Runs the L2 transport lane's unit suite (frontend/tests/compactor/**).
# Mirrors testfixtures/client-unit/entrypoint.sh's shape: copy the
# read-only bind mount to a writable scratch tree first (dist-compactor-tests/
# and npm's caches get written next to the code, and writing those on the
# Windows bind mount directly is slow across that filesystem boundary and
# leaves build artefacts in the working tree). /src stays read-only.
set -e

if [ ! -d /src ]; then
    echo "no /src mount: run this through docker-compose.compactor-tests.yml" >&2
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
    --exclude=./frontend/dist-compactor-tests \
    -cf - . | tar -C /repo -xf -

# node_modules was installed at IMAGE BUILD time (frontend/package*.json
# only) and lives at /opt/client — link it back in rather than
# reinstalling (which would need network at container run time).
ln -s /opt/client/node_modules /repo/frontend/node_modules

cd /repo/frontend

# --only <substring> maps to node's own --test-name-pattern (a regex),
# mirroring scripts/run-tests.py's --only (a substring filter) as closely
# as node's test runner allows.
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

npm run build:compactor-tests

TEST_FILES=$(find dist-compactor-tests/tests/compactor -name '*.test.js')
if [ -z "$TEST_FILES" ]; then
    echo "SKIPPED: no compiled test files found under dist-compactor-tests/tests/compactor" >&2
    exit 3
fi

if [ -n "$ONLY_PATTERN" ]; then
    ESCAPED=$(printf '%s' "$ONLY_PATTERN" | sed 's/[.[\*^$()+?{|\\]/\\&/g')

    # Matches client-unit's own fix for node's test runner substituting the
    # FILE ITSELF as a trivially-passing test when --test-name-pattern
    # matches zero actual test titles — see that entrypoint's comment for
    # the full story (this project's "never fold a skip into a pass" rule).
    #
    # TIGHTER than client-unit's own version of this check (found and
    # fixed within THIS lane, not inherited silently): client-unit's
    # equivalent line greps the WHOLE compiled file for the substring,
    # which also matches a doc comment that happens to mention the word —
    # tsc's default `removeComments: false` means every comment in this
    # lane's heavily-commented source survives into the compiled .js.
    # Verified live: `--only fingerprint` (lowercase) matches no test
    # TITLE in this suite (every title is `turnFingerprint`/
    # `imageOnlyMarker`/etc — capital-letter camelCase, and grep is
    # case-sensitive) but DOES appear in header-comment prose across
    # every file in this suite, so the untightened check let all 6 files
    # through as "matching," and node then substituted each whole file as
    # one trivially-passing test — 6 "passed," zero real assertions run,
    # silently. Requiring the pattern to co-occur on a line that also
    # calls `test(` restricts the match to what could plausibly be an
    # actual test title (every title in this suite is a single-line
    # `test('...', ...)` call), closing that gap.
    #
    # `cat`, not a bare multi-file `grep`, feeds the pipeline: with more
    # than one FILE argument grep prefixes every matched line with
    # "<filename>:" by default, and this suite's own filenames
    # (fingerprint.test.js, receipt.test.js, ...) recreate the EXACT same
    # false-positive this whole check exists to close — `--only
    # fingerprint` matched via the "fingerprint.test.js:" PREFIX grep
    # itself adds, on every file, independent of any title. Caught live
    # while verifying this fix, not assumed safe.
    if ! (cat $TEST_FILES | grep -F 'test(' | grep -qF -- "$ONLY_PATTERN"); then
        echo "SKIPPED: --only '$ONLY_PATTERN' matches no test title in $TEST_FILES" >&2
        exit 3
    fi

    exec node --test --test-name-pattern="$ESCAPED" $TEST_FILES
else
    exec node --test $TEST_FILES
fi
