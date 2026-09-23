#!/usr/bin/env bash
# Copies her data from the READ-ONLY pod-export into the lab's own scratch
# area. Never writes to the source. Idempotent (re-run to refresh the copy;
# use --force to overwrite an existing copy).
#
# Source (read-only, never written to):
#   /home/drew/pod-exports/2026-09-23/backup-1355Z/{webui.db,compactor/}
# Destination (gitignored, outside the repo entirely):
#   /home/drew/scratch/v4lab/data/{webui.db,compactor/}
set -euo pipefail

SRC="${LAB_SOURCE_DIR:-/home/drew/pod-exports/2026-09-23/backup-1355Z}"
DEST="${LAB_DATA_DIR:-/home/drew/scratch/v4lab/data}"
FORCE="${1:-}"

if [ ! -f "${SRC}/webui.db" ]; then
    echo "REFUSING: source webui.db not found at ${SRC}/webui.db" >&2
    exit 1
fi
if [ ! -d "${SRC}/compactor" ]; then
    echo "REFUSING: source compactor/ store not found at ${SRC}/compactor" >&2
    exit 1
fi

mkdir -p "${DEST}"

if [ -e "${DEST}/webui.db" ] && [ "${FORCE}" != "--force" ]; then
    echo "==> ${DEST}/webui.db already exists; skipping (pass --force to refresh)"
else
    echo "==> copying webui.db ($(du -h "${SRC}/webui.db" | cut -f1)) ..."
    cp --no-preserve=mode,ownership "${SRC}/webui.db" "${DEST}/webui.db.tmp"
    mv "${DEST}/webui.db.tmp" "${DEST}/webui.db"
fi

if [ -d "${DEST}/compactor" ] && [ "${FORCE}" != "--force" ]; then
    echo "==> ${DEST}/compactor already exists; skipping (pass --force to refresh)"
else
    echo "==> copying compactor/ store ..."
    rm -rf "${DEST}/compactor.tmp"
    cp -a --no-preserve=mode,ownership "${SRC}/compactor" "${DEST}/compactor.tmp"
    rm -rf "${DEST}/compactor"
    mv "${DEST}/compactor.tmp" "${DEST}/compactor"
fi

chmod -R u+rwX "${DEST}"

# Report counts only -- never her message text or account emails.
echo "==> done. Destination: ${DEST}"
echo "    webui.db: $(du -h "${DEST}/webui.db" | cut -f1)"
echo "    compactor/ files: $(find "${DEST}/compactor" -type f | wc -l)"
echo "    (source at ${SRC} was only ever read from, never written to)"
