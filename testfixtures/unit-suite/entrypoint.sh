#!/bin/sh
# Copy the read-only mount to a writable scratch tree, then run the suite.
#
# The copy is not ceremony. Suites create state directories, sqlite files and
# chromadb stores next to the code; writing those straight onto the bind mount
# is slow across the Windows filesystem boundary and leaves artefacts in the
# developer's working tree, where the next `git status` reads them as changes.
# /src stays read-only so a test cannot mutate the repo even by accident.
set -e

if [ ! -d /src ]; then
    echo "no /src mount: run this through docker-compose.tests.yml" >&2
    exit 2
fi

# -a preserves times so nothing rebuilds spuriously; .git is skipped because
# it is large, irrelevant to the suite, and the one thing a test has no
# business reading.
mkdir -p /repo
tar -C /src --exclude=./.git --exclude=./data -cf - . | tar -C /repo -xf -

cd /repo
exec /opt/compactor-venv/bin/python scripts/run-tests.py \
    --python /opt/compactor-venv/bin/python "$@"
