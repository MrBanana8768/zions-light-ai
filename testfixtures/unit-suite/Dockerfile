# The full local test suite, on the userland production actually runs.
#
# WHY THIS EXISTS. Until 2026-09-03 every "suite green" on this project was a
# Windows result, and Windows is not what ships. Two of the defects this
# release fixed turn on that difference:
#
#   * R22's 239-second suite was 40 tests x ~4.2s of dead TCP handshake,
#     because Windows resolves "localhost" to the IPv6 loopback and THEN the
#     IPv4 one, each against its own 2s connect timeout. On Linux the same
#     call fails in microseconds. A timing defect that only exists on the
#     development machine still cost a real afternoon to find.
#   * The unresolved client-disconnect question is about which asyncio branch
#     a cancelled request takes. Windows runs the Proactor loop; production
#     runs epoll, and likely uvloop. Two reviewers disagreed about this and
#     BOTH were arguing from Proactor evidence.
#
# So this image is the Ubuntu 24.04 the production container is built on
# (nvidia/cuda:13.0.0-runtime-ubuntu24.04 is Ubuntu 24.04 userland), with the
# compactor venv built the same way Dockerfile builds it, from the same
# requirements.txt. It carries NO CUDA, no vLLM and no model weights: the unit
# suites do not need a GPU, and an image that took tens of minutes to build
# would not get run.
#
# It is deliberately NOT testfixtures/tokenizer-contract/Dockerfile.testrunner.
# That one is python:3.12-slim with five packages, sized to prove the
# /tokenize contract and nothing else. The full suite imports chromadb,
# fastembed and mistral_common, and it must import the versions production
# pins — a suite that passes against different pins is evidence about a
# system nobody runs.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    HF_HOME=/hf \
    HF_HUB_DISABLE_TELEMETRY=1

# ca-certificates is not optional even though the suite runs offline: httpx
# imports ssl at module scope and a missing trust store fails the IMPORT, not
# the request, which reads as a broken test rather than a missing package.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        build-essential \
        ca-certificates \
        git \
    && rm -rf /var/lib/apt/lists/*

# Same venv path and same construction as the production image, so an import
# that resolves here resolves there. requirements.txt is copied on its own
# layer: editing a test must not reinstall chromadb.
COPY compactor/requirements.txt /opt/compactor/requirements.txt
RUN python3 -m venv /opt/compactor-venv && \
    /opt/compactor-venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel && \
    /opt/compactor-venv/bin/pip install --no-cache-dir -r /opt/compactor/requirements.txt

# The same two build-time guards the production Dockerfile carries, for the
# same reason: both of these dependencies fail SOFT at runtime — jinja2's
# absence silently drops ~22 tokens/message of chat framing, and
# mistral_common's silently drops the budget onto an estimator that reads up
# to 51% low. A test image that lacked either would go green while measuring
# something production does not measure. Cheap here; a production incident
# there.
RUN /opt/compactor-venv/bin/python -c \
    "import jinja2; print('chat-template rendering available: jinja2', jinja2.__version__)"
RUN /opt/compactor-venv/bin/python -c \
    "import mistral_common; print('mistral_common', mistral_common.__version__)"

# uvloop, so the disconnect and streaming paths can be exercised on the loop
# production runs rather than the one the laptop happens to have. Not in
# requirements.txt because the compactor does not install it itself — uvicorn
# [standard] pulls it on Linux, and this line makes that explicit rather than
# incidental.
RUN /opt/compactor-venv/bin/pip install --no-cache-dir uvloop==0.22.1 && \
    /opt/compactor-venv/bin/python -c "import uvloop; print('uvloop', uvloop.__version__)"

# The repo is COPIED from a read-only mount at run time, not bind-mounted for
# writing. Suites create state directories, sqlite files and chromadb stores;
# doing that straight onto the Windows bind mount is both slow and a way to
# leave artefacts in the developer's working tree. /src is the mount, /repo is
# the scratch copy, and nothing the suite does can reach back out.
COPY testfixtures/unit-suite/entrypoint.sh /usr/local/bin/run-suite
# The CRLF strip is load-bearing, not tidiness: this repo is authored on
# Windows and git may check the file out with CRLF endings, which makes the
# kernel read the shebang as "/bin/sh\r" (with a trailing carriage return)
# and fail the container with a bare
# "no such file or directory" that names a path that plainly exists.
RUN tr -d '\r' < /usr/local/bin/run-suite > /tmp/run-suite && mv /tmp/run-suite /usr/local/bin/run-suite && chmod +x /usr/local/bin/run-suite

WORKDIR /repo
ENTRYPOINT ["/usr/local/bin/run-suite"]
