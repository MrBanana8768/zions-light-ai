# The L1 store's unit suite (frontend/tests/store/**), on Linux, against a
# real Postgres 16 — the same major version production runs
# (entrypoint.sh:194 pins PG_BIN to /usr/lib/postgresql/16/bin). Beside
# testfixtures/unit-suite/Dockerfile, which this mirrors in shape: manifest
# files copied and installed FIRST so the dependency layer caches across
# unrelated source edits, and the actual repo content supplied at container
# START via the read-only bind mount in docker-compose.tests.yml — not
# baked into the image — so a `--build` re-run always tests the working
# tree, never a copy frozen at whatever moment the image last built.
#
# Unlike unit-suite's Python venv, this needs no CUDA/model-weight
# avoidance story: Node 24 slim is small regardless. NOT pinned to the
# exact patch Dockerfile installs for production (ARG NODE_VERSION=
# 24.21.0, fetched directly from nodejs.org/dist): as of this writing
# Docker Hub's official `node` image only publishes up to 24.20.0-slim —
# nodejs.org's own raw releases run ahead of the Docker Hub image build.
# Reproducing production's tarball-download install here would need
# touching Dockerfile's own pattern for a test-only image, which is out of
# this lane's scope (Dockerfile is explicitly not-to-touch). Same MAJOR
# version either way, and this image never runs adapter-node's server —
# only `pg`, `tsc` and node's own test runner, none of which are sensitive
# to the exact patch. Floats to the newest 24.x on each build; if that ever
# causes a real problem, pin the specific tag Docker Hub has at the time.
FROM node:24-slim

WORKDIR /opt/client

# package.json/package-lock.json only, on their own layer — editing
# frontend/src or frontend/tests must not reinstall node_modules.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund

# NODE_PATH lets any node invocation in this image (including the tiny
# connectivity probe below, which lives outside frontend/'s own module
# resolution tree once copied at runtime) resolve `pg` without a symlink
# dance per container start.
ENV NODE_PATH=/opt/client/node_modules

COPY testfixtures/client-unit/entrypoint.sh /usr/local/bin/run-store-tests
COPY testfixtures/client-unit/wait-for-pg.cjs /opt/client/wait-for-pg.cjs
# The CRLF strip is load-bearing, not tidiness — see
# testfixtures/unit-suite/Dockerfile's identical comment: this repo is
# authored on Windows, and a checked-out CRLF shebang line makes the kernel
# read "/bin/sh\r" and fail with a bare "no such file or directory" that
# names a path that plainly exists.
RUN tr -d '\r' < /usr/local/bin/run-store-tests > /tmp/rst && mv /tmp/rst /usr/local/bin/run-store-tests \
    && chmod +x /usr/local/bin/run-store-tests

WORKDIR /repo
ENTRYPOINT ["/usr/local/bin/run-store-tests"]
