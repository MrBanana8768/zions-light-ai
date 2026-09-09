# The L2 transport lane's unit suite (frontend/tests/compactor/**), on
# Linux — FRONTEND_HANDOFF.md §6: "a Windows host run is a development
# convenience, not evidence." Mirrors testfixtures/client-unit/Dockerfile's
# shape (itself mirroring testfixtures/unit-suite/Dockerfile): manifest
# files copied and installed FIRST so the dependency layer caches across
# unrelated source edits, and the actual repo content supplied at
# container START via the read-only bind mount in
# docker-compose.compactor-tests.yml — not baked into the image — so a
# --build re-run always tests the working tree.
#
# UNLIKE client-unit, this suite needs no Postgres sidecar at all: F8's
# send-set/gate tests run against an in-memory fake ChainReader
# (tests/compactor/helpers/fakeChain.ts — the real Store's own constraints
# are L1's suite to prove, not this lane's to re-derive), and every HTTP
# call to the compactor is mocked via an injected `fetchImpl`
# (tests/compactor/transport.test.ts) per this lane's brief: "Mock the
# compactor's HTTP surface — do not require a live backend." So this image
# is plain node:24-slim with no other service to wait for.
FROM node:24-slim

WORKDIR /opt/client

# package.json/package-lock.json only, on their own layer.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund

ENV NODE_PATH=/opt/client/node_modules

COPY testfixtures/compactor-unit/entrypoint.sh /usr/local/bin/run-compactor-tests
# The CRLF strip is load-bearing, not tidiness — see
# testfixtures/unit-suite/Dockerfile's identical comment: this repo is
# authored on Windows, and a checked-out CRLF shebang line makes the kernel
# read "/bin/sh\r" and fail with a bare "no such file or directory" that
# names a path that plainly exists.
RUN tr -d '\r' < /usr/local/bin/run-compactor-tests > /tmp/rct && mv /tmp/rct /usr/local/bin/run-compactor-tests \
    && chmod +x /usr/local/bin/run-compactor-tests

WORKDIR /repo
ENTRYPOINT ["/usr/local/bin/run-compactor-tests"]
