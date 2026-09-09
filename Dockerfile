# Dockerfile for vLLM + context-compactor + OpenWebUI on Runpod
# Optimized for NVIDIA A40 (48GB VRAM) and larger.
#
# CUDA base + torch channel + vLLM version are parametric build args, but
# they are NOT independent — they must move together as ONE of two coherent
# profiles. vLLM's compiled kernels are linked against a specific CUDA major,
# so a mismatched base/torch fails at RUNTIME with
# "ImportError: libcudart.so.NN: cannot open shared object file" (this bit
# v3.0-rc1: CUDA-13 vLLM on a CUDA-12 base). The two supported profiles:
#
#   # DEFAULT — CUDA 13 / driver 580+ / clean vLLM 0.24.0 (recommended):
#   docker build .
#
#   # FALLBACK — CUDA 12 (cu128) / driver >=525 / legacy vLLM 0.19.0 (less secure):
#   docker build \
#     --build-arg CUDA_BASE_IMAGE=nvidia/cuda:12.6.3-runtime-ubuntu24.04 \
#     --build-arg TORCH_CUDA=cu128 \
#     --build-arg VLLM_VERSION=0.19.0 .
#
# Never mix a CUDA-12 base/torch with a CUDA-13 vLLM (or vice versa). Ampere
# cards (e.g. the A40) run CUDA 13 fine — the gate is the HOST driver (>=580),
# not the GPU model.

ARG CUDA_BASE_IMAGE=nvidia/cuda:13.0.0-runtime-ubuntu24.04
FROM ${CUDA_BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Runtime dependencies.
# - binutils: for strip during the install/cleanup layers (~10 MB).
# - build-essential + python3-dev: required at runtime by Triton's JIT,
#   which compiles per-kernel C source during CUDA graph capture.
# - postgresql-16 + postgresql-client-16: the state home (ARCHITECTURE.md
#   Decision 4). Ubuntu 24.04's own repos carry 16+257build1.1 — no
#   PostgreSQL APT repo needed, so this tracks Ubuntu's security patches on
#   the same cadence as everything else in this layer.
#   createcluster.conf is written FIRST: postgresql-common's postinst
#   otherwise auto-creates a default cluster at /var/lib/postgresql/16/main
#   as a side effect of installing the package, using Debian's
#   pg_ctlcluster machinery. We manage PGDATA ourselves (entrypoint.sh runs
#   initdb directly, on a path picked to live on local disk, not /data) —
#   two clusters existing would be pure confusion, so the auto-create is
#   switched off before apt ever runs.
# - apt-get upgrade pulls in CVE patches for the base image's installed
#   packages (gnupg2 etc.). One layer, picks up any patched versions
#   released since the base image was published.
# - apt-get clean + autoremove + lists prune keeps the layer slim.
# ~250 MB total — necessary tax for vLLM + Postgres on a slim base.
RUN mkdir -p /etc/postgresql-common && \
    printf 'create_main_cluster = false\n' > /etc/postgresql-common/createcluster.conf && \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        curl \
        wget \
        git \
        libgomp1 \
        supervisor \
        binutils \
        build-essential \
        postgresql-16 \
        postgresql-client-16 && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Persistent data root — mounted as a single network volume in production.
# Holds both the model cache (/data/models -> HF_HOME) and OpenWebUI state
# (/data/openwebui -> DATA_DIR). entrypoint.sh creates the subdirs on first run.
RUN mkdir -p /data

ENV VLLM_VENV=/opt/vllm-venv

# vLLM + torch CUDA target — see the profile note at the top of this file:
# CUDA_BASE_IMAGE, TORCH_CUDA and VLLM_VERSION must form ONE coherent CUDA
# generation, because vLLM's compiled kernels are CUDA-major-specific.
# Security floor (V3.0, audited 2026-06-30 via PyPI/OSV): 0.14.1 had
# accumulated ~18 CVEs + ~17 GHSAs since it was set; 0.24.0 is the first
# release with NO known advisories, so it is the default. BUT 0.24.0 ships
# CUDA-13 kernels (needs libcudart.so.13) and pins torch==2.11.0 — hence the
# default base is CUDA 13 + the cu130 torch channel, and it requires a host
# driver >=580 (Ampere/A40 is supported on 580; the gate is the host, not
# the card). CUDA-12 / driver-570 hosts must use the fallback profile: vLLM
# 0.19.0 is the last CUDA-12 release (torch 2.10) and still carries ~32
# advisories — an accepted trade for legacy-driver compatibility. Re-audit
# and bump before each release.
ARG VLLM_VERSION=0.24.0
ARG TORCH_CUDA=cu130

# Bake the CUDA channel into the image so entrypoint.sh's driver preflight
# knows which minimum driver to require (cu130 -> 580, cu128/cu126 -> 525).
ENV TORCH_CUDA=${TORCH_CUDA}

# =============================================================================
# vLLM venv — ONLY vLLM. As of V2.0 Phase 3 the compactor has its OWN venv
# (below), so the compactor's deps (chromadb, fastembed, etc.) can NEVER
# disturb vLLM's torch/transformers pins. This permanently closes the
# dependency-coupling bug class that caused the V1.9.x fire drills.
# Installed + stripped + cache-cleared in one layer so unstripped libs and
# .pyc caches never get committed. --extra-index-url pins the torch CUDA
# channel. pip/setuptools/wheel bumped here too (Scout-flagged Highs).
# =============================================================================
RUN python3 -m venv /opt/vllm-venv && \
    /opt/vllm-venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel && \
    /opt/vllm-venv/bin/pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/${TORCH_CUDA} \
        vllm==${VLLM_VERSION} && \
    find /opt/vllm-venv -type f \( -name "*.so" -o -name "*.so.*" \) \
        -exec strip --strip-unneeded {} + 2>/dev/null || true && \
    find /opt/vllm-venv -name "*.pyc" -delete && \
    find /opt/vllm-venv -name "__pycache__" -type d -exec rm -rf {} + && \
    rm -rf /root/.cache /tmp/* /var/tmp/*

# =============================================================================
# Compactor venv — fully decoupled from vLLM. Holds fastapi/uvicorn/httpx
# (proxy), transformers (tokenizer-only, no torch), chromadb (vector store)
# and fastembed (bge-small embeddings via ONNX runtime — no torch, keeps
# this venv lean). COPY requirements separately so editing the Python
# sources later doesn't bust this layer.
# =============================================================================
COPY compactor/requirements.txt /opt/compactor/requirements.txt
RUN python3 -m venv /opt/compactor-venv && \
    /opt/compactor-venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel && \
    /opt/compactor-venv/bin/pip install --no-cache-dir -r /opt/compactor/requirements.txt && \
    find /opt/compactor-venv -type f \( -name "*.so" -o -name "*.so.*" \) \
        -exec strip --strip-unneeded {} + 2>/dev/null || true && \
    find /opt/compactor-venv -name "*.pyc" -delete && \
    find /opt/compactor-venv -name "__pycache__" -type d -exec rm -rf {} + && \
    rm -rf /root/.cache /tmp/* /var/tmp/*

# Fail the build if the chat-template path is unavailable. jinja2 is an
# OPTIONAL transformers dependency that only apply_chat_template needs, so a
# "tokenizer-only" install omits it and count_tokens silently degrades to
# `encode(text) + 4` per message — losing ~22 tokens/message of Mistral
# framing, undetectable until a long conversation overflows the window.
# That shipped for months and surfaced as vLLM 400s on 2026-08-27.
# This guard is cheap; the failure it prevents cost a production incident.
RUN /opt/compactor-venv/bin/python -c \
    "import jinja2; print('chat-template rendering available: jinja2', jinja2.__version__)"

# Same doctrine as the jinja2 guard above, for the same reason: a token counter
# that degrades silently is the failure mode this project keeps paying for.
# compactor/tokens.py needs mistral_common to give an accurate count when
# vLLM's /tokenize cannot answer; without it the fallback is an estimator that
# reads up to 51% low. The module no-ops safely if the import fails, so nothing
# breaks — which is exactly why the absence has to fail the BUILD instead.
#
# AND THE TWO VENVS MUST AGREE ON THE VERSION (v3.1.7). vLLM declares
# mistral_common[image]>=1.10.0 — an open lower bound — so the version IT
# resolves moves on its own while the compactor's requirements.txt stays
# pinned, and nothing enforces agreement. tokens.check_divergence exists to
# catch the disagreement at runtime; this catches it at build time, which is
# where it is cheap. A local tokenizer that silently disagrees with the
# server doing the charging is the 2026-08-28 incident with a better
# disguise: the wrong number is HARDER to disbelieve because it comes from
# the right library.
#
# This is not hypothetical drift. Production runs vLLM 0.19.0, which resolves
# mistral_common 1.11.7 — exactly what requirements.txt pins, so they agree
# TODAY. But the default here is VLLM_VERSION=0.24.0 and the build command in
# the header passes --build-arg VLLM_VERSION=0.19.0, so a rebuild that simply
# forgets the flag gets a different vLLM, a different mistral_common, and no
# warning at all. The guard fails that build instead of shipping it.
RUN set -e; \
    C=$(/opt/compactor-venv/bin/python -c "import mistral_common; print(mistral_common.__version__)"); \
    V=$(/opt/vllm-venv/bin/python -c "import mistral_common; print(mistral_common.__version__)"); \
    echo "local exact tokenization available: mistral_common ${C} (compactor) / ${V} (vllm ${VLLM_VERSION})"; \
    if [ "${C}" != "${V}" ]; then \
        echo "BUILD GUARD 3 FAILED: mistral_common differs between the venvs —" >&2; \
        echo "  compactor-venv ${C} (pinned in compactor/requirements.txt)" >&2; \
        echo "  vllm-venv      ${V} (resolved by vllm==${VLLM_VERSION})" >&2; \
        echo "The compactor's local token count would disagree with the server" >&2; \
        echo "that does the charging. Re-pin mistral_common in" >&2; \
        echo "compactor/requirements.txt to ${V}, or build the vLLM version" >&2; \
        echo "this image is pinned against." >&2; \
        exit 1; \
    fi

# Pre-download the bge-small ONNX embedding model into the image so the
# first request pays no download. Static weights belong in the image, not
# on the /data volume. FASTEMBED_CACHE_PATH (ENV section below) points here.
RUN /opt/compactor-venv/bin/python -c \
    "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5', cache_dir='/opt/embeddings')" && \
    rm -rf /root/.cache /tmp/* /var/tmp/*

# =============================================================================
# Whisper (STT) venv — V3.2. Fully decoupled from vLLM AND the compactor:
# faster-whisper pulls ctranslate2 + av + onnxruntime into its OWN venv, so its
# deps can never disturb vLLM's torch pins or the compactor's. av ships ffmpeg
# in its wheel, so no apt ffmpeg is needed. Same install+strip+clean atomic
# pattern as the other venvs.
# =============================================================================
COPY stt/requirements.txt /opt/stt/requirements.txt
RUN python3 -m venv /opt/whisper-venv && \
    /opt/whisper-venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel && \
    /opt/whisper-venv/bin/pip install --no-cache-dir -r /opt/stt/requirements.txt && \
    find /opt/whisper-venv -type f \( -name "*.so" -o -name "*.so.*" \) \
        -exec strip --strip-unneeded {} + 2>/dev/null || true && \
    find /opt/whisper-venv -name "*.pyc" -delete && \
    find /opt/whisper-venv -name "__pycache__" -type d -exec rm -rf {} + && \
    rm -rf /root/.cache /tmp/* /var/tmp/*

# Pre-download the default Whisper model ("base") into the image so the first
# transcription pays no download — static small models belong in the image, not
# on /data (same principle as the bge embeddings). Bigger models (small/medium/
# large-v3) and a persistent /data root are configurable via WHISPER_MODEL /
# WHISPER_DOWNLOAD_ROOT. Built on CPU (no GPU at build time), int8 weights.
RUN /opt/whisper-venv/bin/python -c \
    "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8', download_root='/opt/whisper-models')" && \
    rm -rf /root/.cache /tmp/* /var/tmp/*

# =============================================================================
# TTS (Piper) venv — V3.3. Own venv, torch-free (Piper is onnxruntime-based),
# CPU only — never competes with vLLM for VRAM and keeps the image lean. Same
# install+strip+clean atomic pattern as the other venvs.
# =============================================================================
COPY tts/requirements.txt /opt/tts/requirements.txt
RUN python3 -m venv /opt/tts-venv && \
    /opt/tts-venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel && \
    /opt/tts-venv/bin/pip install --no-cache-dir -r /opt/tts/requirements.txt && \
    find /opt/tts-venv -type f \( -name "*.so" -o -name "*.so.*" \) \
        -exec strip --strip-unneeded {} + 2>/dev/null || true && \
    find /opt/tts-venv -name "*.pyc" -delete && \
    find /opt/tts-venv -name "__pycache__" -type d -exec rm -rf {} + && \
    rm -rf /root/.cache /tmp/* /var/tmp/*

# Pre-download the default Piper voice (en_US-lessac-medium, ~63 MB) into the
# image so the first speech pays no download — a static model belongs in the
# image (same principle as the bge + whisper models). Swap via TTS_VOICE (+ a
# /data TTS_VOICE_DIR for other voices). Voices: huggingface.co/rhasspy/piper-voices.
RUN mkdir -p /opt/tts-voices && \
    wget -q -O /opt/tts-voices/en_US-lessac-medium.onnx \
      https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx && \
    wget -q -O /opt/tts-voices/en_US-lessac-medium.onnx.json \
      https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json

# =============================================================================
# OpenWebUI venv — kept isolated from vLLM's pytorch pin. Same install+strip
# atomic pattern. Also bumps pip/setuptools/wheel here (Scout-flagged Highs
# live in both venvs since each has its own copy).
# =============================================================================
WORKDIR /app
RUN python3 -m venv /app/venv && \
    /app/venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel && \
    /app/venv/bin/pip install --no-cache-dir open-webui==0.11.0 && \
    find /app/venv -type f \( -name "*.so" -o -name "*.so.*" \) \
        -exec strip --strip-unneeded {} + 2>/dev/null || true && \
    find /app/venv -name "*.pyc" -delete && \
    find /app/venv -name "__pycache__" -type d -exec rm -rf {} + && \
    rm -rf /root/.cache /tmp/* /var/tmp/*
# Note: OpenWebUI data lives at DATA_DIR=/data/openwebui (on the persistent
# volume), created by entrypoint.sh at boot. No /app/data dirs needed —
# that was the pre-single-volume layout (removed in V2.2 cleanup).

# =============================================================================
# Node.js — FRONTEND_PLAN.md F1/F2. The one non-Python runtime in this
# image, for the SvelteKit client (frontend/) only; vLLM/compactor/STT/
# TTS/OpenWebUI stay Python, each in its own venv per the pattern above.
#
# Installed as a pinned, direct download of the official prebuilt tarball
# into /opt/node — not an apt/NodeSource install. That keeps this layer to
# ONE new trust boundary (nodejs.org's TLS cert), not two (that, plus a
# NodeSource apt repo + signing key), and it is the same "download and
# extract into /opt" shape already used below for the TTS voice files.
# NODE_VERSION is an ARG for the same reason VLLM_VERSION is: pin the exact
# version rather than "whatever apt/NodeSource has today", and make bumping
# it a one-line change. v24.x ("Krypton") is Active LTS as of this build.
# =============================================================================
ARG NODE_VERSION=24.21.0
# .tar.gz, not the smaller .tar.xz nodejs.org also publishes: `tar -z`
# needs only gzip, present in essentially every base image by construction,
# where `tar -J` needs xz-utils, which this image never installs and whose
# presence here is unverified. One dependency fewer to be wrong about.
RUN curl -fsSL -o /tmp/node.tar.gz \
        "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.gz" && \
    mkdir -p /opt/node && \
    tar -xzf /tmp/node.tar.gz -C /opt/node --strip-components=1 && \
    rm -f /tmp/node.tar.gz
ENV PATH="/opt/node/bin:${PATH}"

# =============================================================================
# Client (frontend/) — SvelteKit + adapter-node (FRONTEND_PLAN.md F1/F2).
#
# COPY package*.json before the rest of the source, same cache-layering
# reason as `COPY compactor/requirements.txt` above: editing a .svelte file
# later must not bust the (slow) npm-install layer. `npm ci`, not
# `npm install` — it installs exactly what package-lock.json records and
# fails the build if the lockfile and package.json have drifted, rather
# than silently re-resolving.
#
# NETWORK CALLED OUT EXPLICITLY (FRONTEND_PLAN.md F2 asks for this to be
# flagged so it can be challenged): `npm ci` resolves packages from the
# public npm registry. This is BUILD-time only, the same category as `pip
# install` from PyPI or `apt-get install` from the Ubuntu archive above —
# the pod still builds and boots and serves with NO network reachable at
# runtime. No package here is a private/`@telos-llc/*` package (see
# frontend/src/lib/styles/tokens.css's header and docs/lanes/L0-scaffold.md)
# so there is no private-registry credential to fail to have, either.
#
# `npm prune --omit=dev`, NOT `rm -rf node_modules`. An earlier revision of
# this hunk deleted node_modules outright, on the then-true premise that
# package.json had ZERO `dependencies` - adapter-node bundles the SvelteKit
# server into build/, so nothing was left to resolve at runtime. That premise
# died the moment the store lane added `pg`, and the failure it would have
# caused is the shape this project keeps repeating: the image still BUILDS,
# BUILD GUARD 4 still passes because build/index.js exists, and the client
# dies at import time on the pod. Pruning is correct whether or not a given
# dependency ends up bundled, costs a few MB, and cannot rot when the next
# dependency is added. It is the faithful analogue of the venvs'
# install+strip+clean above: keep what runtime needs, drop what only the
# build needed.
# =============================================================================
COPY frontend/package.json frontend/package-lock.json frontend/.npmrc /opt/client/
RUN cd /opt/client && npm ci
COPY frontend/vite.config.ts frontend/tsconfig.json /opt/client/
COPY frontend/src /opt/client/src
COPY frontend/static /opt/client/static
RUN cd /opt/client && \
    npm run build && \
    npm prune --omit=dev && \
    rm -rf package-lock.json src static .svelte-kit \
        vite.config.ts tsconfig.json && \
    npm cache clean --force && \
    rm -rf /root/.npm /root/.cache /tmp/* /var/tmp/*

# BUILD GUARD 4: the client's built server must exist exactly where
# supervisord.conf's [program:client] stanza invokes it. Same failure
# class as BUILD GUARD / BUILD GUARD 2 above (a runtime reference to a
# path this image never actually produced, e.g. because `npm run build`
# failed non-fatally, output moved, or the adapter changed its layout) —
# applied to the one program in this file that is not a Python import
# BUILD GUARD 1/2 already cover.
RUN test -x /opt/node/bin/node || \
      { echo "BUILD GUARD 4 FAILED: /opt/node/bin/node missing — supervisord.conf's [program:client] command= references this path directly."; exit 1; }; \
    test -f /opt/client/build/index.js || \
      { echo "BUILD GUARD 4 FAILED: /opt/client/build/index.js missing — the client build did not produce adapter-node's entrypoint where [program:client] expects it."; exit 1; }; \
    /opt/node/bin/node -e 'const fs=require("fs"),p=require("/opt/client/package.json"),d=Object.keys(p.dependencies||{});const miss=d.filter(n=>!fs.existsSync("/opt/client/node_modules/"+n));if(miss.length){console.error("BUILD GUARD 4 FAILED: runtime dependencies missing from node_modules after prune: "+miss.join(", ")+". The client dies at import time on the pod even though this image built cleanly.");process.exit(1)}console.log("build guard: "+d.length+" runtime dependency(ies) survived the prune")' && \
    echo "build guard: the client binary and its built server both exist"

# Compactor sources copied AFTER the expensive install layer so editing
# the Python files doesn't invalidate the vllm install cache. List each
# runtime module explicitly to avoid pulling test_*.py and V2_PLAN.md
# into the production image.
COPY compactor/main.py /opt/compactor/main.py
COPY compactor/envcfg.py /opt/compactor/envcfg.py
COPY compactor/memory.py /opt/compactor/memory.py
COPY compactor/facts.py /opt/compactor/facts.py
COPY compactor/backfill.py /opt/compactor/backfill.py
COPY compactor/retrieval.py /opt/compactor/retrieval.py
COPY compactor/summarizer.py /opt/compactor/summarizer.py
COPY compactor/health.py /opt/compactor/health.py
COPY compactor/selftest.py /opt/compactor/selftest.py
COPY compactor/portability.py /opt/compactor/portability.py
COPY compactor/dedup.py /opt/compactor/dedup.py
COPY compactor/commands.py /opt/compactor/commands.py
COPY compactor/persona.py /opt/compactor/persona.py
COPY compactor/backup.py /opt/compactor/backup.py
COPY compactor/degrade.py /opt/compactor/degrade.py
COPY compactor/bgwork.py /opt/compactor/bgwork.py
COPY compactor/textclean.py /opt/compactor/textclean.py
COPY compactor/tokens.py /opt/compactor/tokens.py
COPY compactor/tokenhealth.py /opt/compactor/tokenhealth.py
COPY compactor/tailhealth.py /opt/compactor/tailhealth.py
COPY compactor/webuidb.py /opt/compactor/webuidb.py
COPY compactor/logsetup.py /opt/compactor/logsetup.py
COPY compactor/alert.py /opt/compactor/alert.py
COPY compactor/pgarchive.py /opt/compactor/pgarchive.py
COPY compactor/dbselect.py /opt/compactor/dbselect.py
# The migration script is not a compactor module - it is the recovery step
# entrypoint.sh PRINTS when it refuses to hand over to Postgres. It shipped
# referenced but absent: the boot log said to run
# /data/scripts/migrate-webui-sqlite-to-pg.py, nothing ever put a file
# there, and BUILD GUARD 2 could not see it because its regex only covers
# /opt/compactor/. So it lives here, where the guard DOES cover it, and the
# printed instructions point at this path. The one moment anyone reads that
# line is while trying to rescue her history under pressure.
COPY scripts/migrate-webui-sqlite-to-pg.py /opt/compactor/migrate-webui-sqlite-to-pg.py

# BUILD GUARD: every compactor module must actually be in the image.
#
# v3.1.4 shipped with tokenhealth.py missing from the COPY list above and the
# compactor went FATAL on boot in production - "ModuleNotFoundError: No module
# named 'tokenhealth'" - with the chat path down until an operator hot-copied
# the file in. The module was new in v3.1.3; the COPY line was not added with
# it.
#
# Nothing could have caught that. The test suite mounts the SOURCE directory
# over the container's, so 39/39 unit tests, the contract suite, the soak and
# three adversarial review gates all ran against a file set the image does not
# have. This is a packaging defect class, not a code one, and the enumerated
# COPY list above is what arms it every time a module is added.
#
# So: import every entrypoint the same way supervisord does, at build time,
# from the image's own /opt/compactor. A missing module, a syntax error or a
# broken import fails the BUILD instead of production. Offline and
# network-free by construction (verified) so it cannot flake:
#   - uvicorn main:app          -> main
#   - selftest.py --on-boot     -> selftest
#   - backup.py --daemon        -> backup
#   - pgarchive.py --archive-loop (default) -> pgarchive
#   - dbselect.py (invoked by entrypoint.sh)  -> dbselect
# plus the modules those pull in transitively, which is the whole package.
RUN cd /opt/compactor && \
    MODEL_REPO=buildguard VLLM_URL=http://127.0.0.1:1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    COMPACTOR_STORAGE_ROOT=/tmp/buildguard LOG_DIR=/tmp/buildguard \
    /opt/compactor-venv/bin/python -c \
      "import main, selftest, backup, health, commands, portability, webuidb, pgarchive, dbselect; \
       print('build guard: every compactor entrypoint imports from the image')" \
    && rm -rf /tmp/buildguard /opt/compactor/__pycache__

# V3.2 — STT service source (copied late, after its venv, for cache efficiency).
COPY stt/server.py /opt/stt/server.py

# V3.3 — TTS service source (copied late, after its venv, for cache efficiency).
COPY tts/server.py /opt/tts/server.py

# =============================================================================
# Supervisor
# =============================================================================
RUN mkdir -p /var/log/supervisor
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY entrypoint.sh /entrypoint.sh

# BUILD GUARD 2: every /opt/compactor script the RUNTIME references must be in
# the image.
#
# The guard above catches a module nobody imports; this catches a script
# nobody imports EITHER, because it is invoked as a subprocess. v3.1.4 went
# FATAL in production on exactly that gap (tokenhealth.py, added to
# summarizer.py's imports and never to the COPY list). It recurred on
# 2026-08-31 with dbselect.py, which entrypoint.sh execs to decide WHICH
# DATABASE holds her chat history: the image would have booted, hit
# `eval "$(python /opt/compactor/dbselect.py ...)"`, found nothing there, and
# failed at the one step that must not fail.
#
# Both times the fix was "remember to add a line", and both times the line
# was forgotten. So this guard does not have a list to forget: it DERIVES
# what to check from the runtime files themselves, and any new script picked
# up by entrypoint.sh or supervisord.conf is covered the moment it is
# referenced. Runs after both are COPY'd, which is why it is here and not
# with the import guard above.
RUN missing=""; \
    for f in $(grep -ohE "/opt/compactor/[A-Za-z0-9_-]+\.py" \
                 /entrypoint.sh /etc/supervisor/conf.d/supervisord.conf \
               | sort -u); do \
        [ -f "$f" ] || missing="$missing $f"; \
    done; \
    if [ -n "$missing" ]; then \
        echo "BUILD GUARD FAILED - referenced at runtime, not in the image:$missing"; \
        echo "Add a COPY line for each. This is the v3.1.4 FATAL, again."; \
        exit 1; \
    fi; \
    echo "build guard: every /opt/compactor script referenced at runtime exists"
RUN chmod +x /entrypoint.sh

# Operator tool: reclaim volume space by removing stale HuggingFace model
# caches (old MODEL_REPO weights linger on the Network Volume). Not on the
# startup path — run on demand: docker exec <container> /opt/clean-models.sh
COPY clean-models.sh /opt/clean-models.sh
RUN chmod +x /opt/clean-models.sh

# =============================================================================
# Model configuration — override via .env or Runpod template
# Default: Cydonia 24B (heretic) + runtime fp8 — the PRODUCTION-VALIDATED A40
# config (V3.0 rc line): boots and serves at ~43GB incl. KV on a 48GB card.
# The two defaults move TOGETHER: a 24B without the fp8 flag will not fit an
# A40. (The previous default, magnum-v4-22b with no quant flag, could not boot
# on the A40 at all — the out-of-the-box trap PR #16 flagged; changed rc8.)
# On 80GB-class cards, clear VLLM_EXTRA_ARGS for full FP16 quality.
# Any HuggingFace causal-LM repo that vLLM supports works here.
# =============================================================================
ENV MODEL_REPO="coder3101/Cydonia-24B-v4.3-vision-heretic"
ENV HF_HOME="/data/models"
# Note: TRANSFORMERS_CACHE was removed in v1.9.1 — deprecated in transformers
# v5, HF_HOME is the modern equivalent and is read by both transformers
# and huggingface_hub.

# vLLM server settings
ENV VLLM_HOST="0.0.0.0"
ENV VLLM_PORT="8000"
ENV MAX_MODEL_LEN="32768"
ENV GPU_MEMORY_UTILIZATION="0.90"
# Paired with the Cydonia-24B default above — a 24B on a 48GB card requires
# runtime fp8. Clear this when swapping to a 12B-class model or an 80GB card.
ENV VLLM_EXTRA_ARGS="--quantization fp8"

# context-compactor settings (port 8080 — what OpenWebUI talks to)
ENV COMPACTOR_HOST="0.0.0.0"
ENV COMPACTOR_PORT="8080"
# V2.1 Phase 6 Step 2: post-boot self-test auto-runs as a supervisord
# one-shot. Disable per-pod (e.g. for CI containers) by setting to "false".
ENV COMPACTOR_SELFTEST_ON_BOOT="true"
# V2.3 Theme 1: periodic data-durability backup daemon. Disable per-pod
# (e.g. CI containers) by setting to "false".
ENV COMPACTOR_BACKUP_ENABLED="true"
ENV COMPACTOR_TARGET_TOKENS=""
ENV COMPACTOR_KEEP_RECENT_TURNS="4"
ENV COMPACTOR_SUMMARY_MAX_TOKENS="1024"
ENV VLLM_URL="http://localhost:8000"

# V2.0 Phase 2 — facts memory
ENV COMPACTOR_FACTS_EXTRACTION="true"
ENV COMPACTOR_MAX_FACTS_TOKENS="1500"
ENV COMPACTOR_ADMIN_BIND="127.0.0.1"

# V2.0 Phase 3 — episodic memory (RAG). Embedding model baked into the
# image at /opt/embeddings; FASTEMBED_CACHE_PATH points there so no
# runtime download. RAG can be disabled with COMPACTOR_RAG_ENABLED=false.
ENV COMPACTOR_RAG_ENABLED="true"
ENV COMPACTOR_RAG_TOP_K="5"
ENV COMPACTOR_EMBEDDING_MODEL="BAAI/bge-small-en-v1.5"
ENV FASTEMBED_CACHE_PATH="/opt/embeddings"

# V3.2 — Speech-to-text (Whisper) service. Runs in its own venv on STT_PORT.
# CPU by default so it never competes with vLLM for VRAM. Default model "base"
# is prebaked at /opt/whisper-models; swap via WHISPER_MODEL (+ WHISPER_DEVICE=
# cuda and/or a /data WHISPER_DOWNLOAD_ROOT for larger persistent models).
# Disable the whole service per-pod with STT_ENABLED=false.
ENV STT_ENABLED="true"
ENV WHISPER_MODEL="base"
ENV WHISPER_DEVICE="cpu"
ENV WHISPER_DOWNLOAD_ROOT="/opt/whisper-models"
ENV WHISPER_MODEL_ID="whisper-1"
ENV STT_HOST="0.0.0.0"
ENV STT_PORT="9000"

# V3.3 — Text-to-speech (Piper) service. Own venv on TTS_PORT, CPU + torch-free.
# Default voice en_US-lessac-medium is prebaked at /opt/tts-voices; swap via
# TTS_VOICE (+ a /data TTS_VOICE_DIR for other voices). Disable per-pod with
# TTS_ENABLED=false.
ENV TTS_ENABLED="true"
ENV TTS_VOICE="en_US-lessac-medium"
ENV TTS_VOICE_DIR="/opt/tts-voices"
ENV TTS_MODEL_ID="tts-1"
ENV TTS_HOST="0.0.0.0"
ENV TTS_PORT="9001"

# =============================================================================
# PostgreSQL — the state home (ARCHITECTURE.md Decision 4). PGDATA is on
# LOCAL disk (the pod overlay) — the exact fix already proven for webui.db,
# applied here BEFORE the failure ever gets a chance to happen instead of
# after. Postgres listens on a unix socket only (unix_socket_directories,
# set by entrypoint.sh at initdb time); there is no listen_addresses, so
# there is no TCP port to fail. /data is used ONLY as a pg_dump archive
# target (compactor/pgarchive.py) — never a place Postgres itself writes.
#
# PG_BIN: postgresql-common installs client wrappers (psql, pg_dump, ...)
# onto PATH via /usr/bin, but NOT initdb/pg_ctl/postgres itself — those stay
# under the versioned lib dir. entrypoint.sh needs the explicit path.
#
# DATABASE_URL is a plain libpq URI with an empty host component before
# `?host=` — that is what selects the unix-socket code path over TCP.
# Overridable: point it at a managed external Postgres later as a pure
# connection-string change, no code change.
# =============================================================================
ENV PGDATA="/var/lib/postgresql/data"
ENV PG_BIN="/usr/lib/postgresql/16/bin"
ENV POSTGRES_USER="openwebui"
ENV POSTGRES_DB="openwebui"
ENV POSTGRES_SOCKET_DIR="/var/run/postgresql"
ENV DATABASE_URL="postgresql://openwebui@/openwebui?host=/var/run/postgresql"

# compactor/pgarchive.py — periodic pg_dump to /data (never a place Postgres
# itself writes) plus restore-on-boot from the newest good archive. Disable
# per-pod (e.g. CI containers) via COMPACTOR_PGARCHIVE_ENABLED=false, same
# convention as COMPACTOR_BACKUP_ENABLED.
# entrypoint.sh overrides this on every boot from which database it actually
# selected. The default only has to keep supervisord's config load from
# failing on an unset %(ENV_...)s, and "false" is the right default because
# DATABASE_URL above defaults to Postgres.
ENV WEBUIDB_SYNC_ENABLED="false"

ENV COMPACTOR_PGARCHIVE_ENABLED="true"
ENV PGARCHIVE_DIR="/data/openwebui/pg"
ENV PGARCHIVE_INTERVAL_S="300"

# OpenWebUI settings — points at the compactor, not vLLM directly
ENV OPENWEBUI_PORT="3000"
ENV WEBUI_SECRET_KEY=""
ENV OLLAMA_BASE_URL=""
ENV OPENAI_API_BASE_URL="http://localhost:8080/v1"
ENV OPENAI_API_KEY="not-needed"
ENV ENABLE_OLLAMA_API="false"
ENV ENABLE_OPENAI_API="true"
ENV DATA_DIR="/data/openwebui"
ENV WEBUI_AUTH="true"

# SQLite hardening — now dormant by default. DATABASE_URL above points
# OpenWebUI at Postgres, so it never opens webui.db at all; these PRAGMAs
# only take effect if DATABASE_URL is overridden back to a sqlite:/// URL
# (the documented rollback path — see entrypoint.sh). Left in place rather
# than removed, since that rollback is exactly what this project's release
# workflow calls "no hot-patches on prod": a connection-string revert, not a
# code change. Original rationale, for whichever DB is actually in use over
# /data (the RunPod NETWORK volume): OpenWebUI 0.11.0 defaults to WAL journal mode, but
# SQLite's WAL needs an mmap'd shared-memory (-shm) index that network
# filesystems don't support reliably — on a network volume WAL *causes* the
# "database is locked" errors it's meant to avoid. So: fall back to rollback
# (DELETE) journal, which needs no -shm/mmap; raise the lock-wait modestly (10s,
# not 30s+ which would just hide a real deadlock); and drop the DB mmap (also
# unreliable over network FS). synchronous stays at OWUI's NORMAL default on
# purpose — FULL would lengthen writes on slow network storage and worsen
# contention. Inherited by the openwebui program like the AUDIO_* vars above.
ENV DATABASE_ENABLE_SQLITE_WAL="false"
ENV DATABASE_SQLITE_PRAGMA_BUSY_TIMEOUT="10000"
ENV DATABASE_SQLITE_PRAGMA_MMAP_SIZE="0"

# V3.2 — wire OpenWebUI's STT to the local Whisper service (OpenAI engine).
# Disable voice input per-pod by setting AUDIO_STT_ENGINE="" (empty).
ENV AUDIO_STT_ENGINE="openai"
ENV AUDIO_STT_OPENAI_API_BASE_URL="http://localhost:9000/v1"
ENV AUDIO_STT_OPENAI_API_KEY="not-needed"
ENV AUDIO_STT_MODEL="whisper-1"

# V3.3 — wire OpenWebUI's TTS to the local Piper service (OpenAI engine).
# Disable voice output per-pod by setting AUDIO_TTS_ENGINE="" (empty). The voice
# field is sent by OpenWebUI but ignored by the service (it uses TTS_VOICE).
ENV AUDIO_TTS_ENGINE="openai"
ENV AUDIO_TTS_OPENAI_API_BASE_URL="http://localhost:9001/v1"
ENV AUDIO_TTS_OPENAI_API_KEY="not-needed"
ENV AUDIO_TTS_MODEL="tts-1"
ENV AUDIO_TTS_VOICE="alloy"

# Log destination. supervisord.conf expands %(ENV_LOG_DIR)s at parse time and
# refuses to start if it is unset, so the default lives here rather than only
# in entrypoint.sh — a `docker run` that bypasses the entrypoint still boots.
# /data is the network volume: logs must survive the container that wrote them.
ENV LOG_DIR="/data/logs"

# 3000 — OpenWebUI (user-facing)
# 3001 — client (user-facing; FRONTEND_PLAN.md F1/F2 — the SvelteKit
#        replacement for OpenWebUI, [program:client] in supervisord.conf).
#        This EXPOSE is necessary but not sufficient: RunPod's pod template
#        exposes ports independently of the image, and adding 3001 there is
#        NOT done by this Dockerfile change — hand that back (see
#        docs/lanes/L0-scaffold.md).
# 8080 — context-compactor (OpenAI-compatible, what OpenWebUI talks to)
# 8000 — vLLM (internal; can also be exposed for direct API access)
# 9000 — STT / Whisper (OpenAI audio API; OpenWebUI talks here for voice input)
# 9001 — TTS / Piper (OpenAI audio API; OpenWebUI talks here for voice output)
EXPOSE 8000 8080 3000 3001 9000 9001

# V2.1 Phase 6: switch from `curl :3000` (OpenWebUI login page) to the
# compactor's /health/full deep probe. The old check stayed "healthy"
# even when vLLM was FATAL because OpenWebUI's login page kept serving;
# /health/full returns 503 when storage breaks and reports degraded
# status when vLLM is unreachable. start-period=300s covers model load.
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8080/health/full || exit 1

ENTRYPOINT ["/entrypoint.sh"]
