# Changelog

All notable changes to Zion's Light AI. Format inspired by
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

Image tags published at
[`angreg/zions-light-ai`](https://hub.docker.com/r/angreg/zions-light-ai)
on Docker Hub.

---

## [2.2] — Testing & Observability

**Goal:** make "is this deploy actually working?" answerable automatically,
and codify a testing standard every future feature must follow. No separate
image — all V2.2 code shipped inside the V2.1 image line; this release is
the standard + its tooling reaching completeness.

Image: folded into `angreg/zions-light-ai:v2.1` (and the `:v2.1-phase6`/
`6.1` tags specifically).

### Added
- **Tier-2 boot self-test** (`compactor/selftest.py`) — post-boot validation
  battery run as a non-blocking one-shot supervisord program
  (`COMPACTOR_SELFTEST_ON_BOOT=true`, logs to
  `/var/log/supervisor/selftest.log`). Checks: `/data` writable, vLLM lists
  the model, compactor `/health`, a real 1-token chat round-trip, a facts
  write/read/delete against a `__selftest__` sentinel, admin localhost
  gating. Also on-demand via `GET /admin/selftest`.
- **Two-phase vLLM readiness probe** — `--wait-for-ready` waits for an
  actual completion (`/v1/chat/completions` 200), not just an open
  `/v1/models` port, so the boot self-test can't false-fail during the
  1-5 minute cold model load.
- **`GET /health/full`** — deep probe (vLLM reachability + storage
  writability + memory-store stats). Now the Docker `HEALTHCHECK` target,
  replacing `curl :3000` which stayed green even when vLLM was FATAL.
- **`TESTING.md`** — the three-tier testing standard (Tier-1 unit / Tier-2
  boot self-test / Tier-3 integration), the per-PR requirements, and the
  exact run commands for each tier.

### Changed
- Docker `HEALTHCHECK` target switched from `http://localhost:3000/`
  (OpenWebUI login page) to `http://localhost:8080/health/full`.
- Removed dead `/app/data` mkdir cruft from the Dockerfile (pre-single-
  volume layout leftover).

---

## [2.1] — User control, portability, observability, quality

**Goal:** give the *user* agency over memory and make the system operable.
V2.0 gave the model memory; V2.1 lets the user inspect, edit, export,
deduplicate, and shape it — plus the observability surface to run it.

Images: `angreg/zions-light-ai:v2.1-phase6.1` (observability),
`:v2.1-phase7` (quality), `:v2.1-phase8` / `:v2.1-complete` (commands +
personas). Rolling tag: `:v2.1`.

### Added — Phase 5: chat commands
- In-chat slash commands intercepted by the compactor (zero LLM cost,
  instant, model never sees them): `/help`, `/list-facts`, `/list-archive`,
  `/remember <text>`, `/forget [substring]`, `/why`. Streaming and
  non-streaming response paths both synthesize an OpenAI-shaped completion.
  Conservative detection — non-command slash messages pass through to vLLM.

### Added — Phase 6: observability + portability
- `GET /health/full`, `GET /admin/selftest`, boot self-test (documented
  under [2.2] — they pair).
- **Conversation portability** — `GET /admin/conversations/<id>/export`,
  `POST /admin/conversations/import`, `POST /admin/conversations/<id>/fork`.
  Single JSON bundle per conv (facts + summary state + episodic exchanges);
  embeddings re-derived on import so bundles survive embedding-model swaps.

### Added — Phase 7: quality maintenance
- **Hybrid semantic deduplication** (`compactor/dedup.py`) — embedding
  clustering filters candidates, an LLM verification call (KEEP-on-doubt,
  temp 0.0) confirms merges. Runs inline after every fact extraction
  (0 LLM calls when no candidate clusters) and on-demand via
  `POST /admin/conversations/<id>/dedup`.
- **Stale-fact archival** — facts unused for N days (default 90) move to a
  cold-storage sidecar; recoverable via restore. `GET`/`POST
  …/archive` + `POST …/restore`.

### Added — Phase 8: personas as first-class memory
- Persona (long durable system prompt) recognized as its own memory layer:
  auto-detected from a long first system message, stored separately, exempt
  from summarizer rollup and LRU fact eviction, injected as a labeled block
  (with a double-injection guard). `GET /admin/personas` library, full
  GET/POST/DELETE per conv, and `POST …/inherit-persona` to clone across
  conversations.

### Changed
- `/admin/forget` (and the `/forget` chat command) now clear the persona
  layer too — a full memory wipe is truly full.
- `/admin/conversations/<id>` summary now reports persona presence.

---

## [2.0] — Three-layer persistent memory

**Goal:** give long creative-writing conversations memory that survives the
context window and pod restarts. A FastAPI "compactor" middleware sits
between OpenWebUI and vLLM and maintains per-conversation memory on the
network volume.

Image: `angreg/zions-light-ai:v2.0` (final: `:v2.0-phase4.3`,
`sha256:d142bf0a`).

### Added
- **Conversation identity** (`compactor/memory.py`) — resolved from an
  `X-Conversation-Id` header (set by a bundled OpenWebUI Pipeline filter),
  falling back to `body.metadata.chat_id`, then a SHA-256 fingerprint.
  Atomic JSON writes (temp + fsync + rename) and a per-conv `asyncio.Lock`
  manager serialize concurrent writers.
- **Layer 1 — facts** (`compactor/facts.py`, Phase 2) — a side LLM call
  after each turn distills durable facts; LRU-pruned to a token budget;
  injected on subsequent turns. Lazy backfill (`compactor/backfill.py`)
  extracts facts from pre-existing V1 conversations on first sight.
- **Layer 2 — RAG** (`compactor/retrieval.py`, Phase 3) — every exchange
  embedded (bge-small ONNX) into ChromaDB and retrieved by semantic
  similarity for later turns. Runs in a dedicated torch-free
  `compactor-venv` isolated from vLLM's torch stack; bge-small prebaked
  into the image.
- **Layer 3 — hierarchical summaries** (`compactor/summarizer.py`,
  Phase 4) — rolling L1→L2→L3 cascade so even very long conversations get
  a coherent context block.
- **Admin/observability endpoints** — `GET /admin/conversations`,
  `GET /admin/conversations/<id>`, `GET`/`DELETE
  /admin/conversations/<id>/facts`, all localhost-gated.
- **Tier-3 integration suite** (`tests/integration/`) — black-box
  pytest+httpx scenarios against a live pod.

### Fixed
- **Mistral chat-template rejection** (Phase 4.1) — the three memory layers
  were injected as three separate system messages, which Mistral-family
  templates (Magnum v4 12B/22B) reject once ≥2 layers populate. Combined
  into a single system message.
- **Fact extraction NONE-bias** (Phase 4.3) — Magnum-12B returned the
  literal `NONE` for ~65% of fact-rich prompts at temp 0.2. Rewrote the
  extraction prompt to bias toward extraction and dropped temperature to
  0.0; extraction is now reliable.
- Test reliability: replaced fixed async-tail sleeps with polling helpers
  (`wait_for_facts`, `wait_for_indexed_exchanges`).

---

## [1.9.6] — Final V1 release

**Goal:** close V1 cleanly. CVE remediation, parametric build foundation,
operational quality improvements. After this, the 1.9.x line is frozen
except for security patches — new feature work moves to V2.

### Security
- **Bumped vllm `0.11.0` → `0.14.1`** — resolves CVE-2026-22778 (Critical 9.8)
  and 7 other High-severity CVEs in vllm itself. Includes auto-bumps of
  torch and xgrammar transitive deps that resolve their respective Highs.
- **Added `apt-get upgrade -y` to the Dockerfile** — picks up Ubuntu CVE
  patches for installed packages (catches gnupg2 High and any future
  ones released after the base image was published).
- **Bumped `pip` / `setuptools` / `wheel`** in both venvs as part of the
  install layer — resolves 4 Highs (setuptools×2, wheel×2).
- **Bumped OpenWebUI to its latest release** — resolves Highs in pillow,
  ecdsa, pyjwt, python-multipart, nltk, pyarrow, langchain-classic,
  jaraco.context (all OpenWebUI's transitive deps).
- Bumped transformers ceiling within the `<5` range to pick up its
  flagged High.

### Foundation
- **Parametric CUDA build args** (`CUDA_BASE_IMAGE`, `TORCH_CUDA`,
  `VLLM_VERSION`). Same Dockerfile now builds cu128 (default) and cu130
  variants without source changes. Foundation for the eventual cu130
  variant once RunPod's GPU fleet broadly rolls out driver 580+.
- **Preflight checks in entrypoint.sh**: verify `/data` is writable,
  GPU is visible via nvidia-smi, driver version meets torch's minimum.
  Fails loud and fast with actionable messages instead of letting vLLM
  crash 2-3 minutes in with a cryptic stack trace.
- **Persistent torch.compile cache** at `/data/vllm-compile-cache`.
  Cold starts after the first one skip the 60-120s CUDA graph capture
  step. Symlinked from `/root/.cache/vllm` in entrypoint.sh.

### Documentation
- New top-level **README.md** with architecture diagram, quick start, and
  project structure.
- New top-level **CHANGELOG.md** (this file) documenting the entire 1.9.x
  line.
- New top-level **ROADMAP.md** with V1 → V2.0 → V2.1 → V3 → beyond plan.
- **V2_PLAN.md** updated to split V2 into V2.0 (memory architecture) and
  V2.1 (user control / portability / observability). Conv_id strategy
  upgraded from "hash with header fallback" to "header from day one,
  hash as fallback" — eliminates the collision risk class entirely.

---

## [1.9.5] — Triton JIT toolchain

### Fixed
- Added **`build-essential` + `python3-dev`** to the apt install layer.
  vLLM (via torch.compile) uses Triton to JIT-compile per-kernel C
  source at runtime during CUDA graph capture; without a compiler and
  Python headers, vLLM crashed at startup with either "Failed to find
  C compiler" or "Python.h: No such file or directory". ~200 MB image
  growth — necessary tax for vLLM on a slim base.

---

## [1.9.4] — Transformers compat for vLLM 0.11

### Fixed
- **Pinned `transformers>=4.50,<5`** in `compactor/requirements.txt`.
  vLLM 0.11 calls `tokenizer.all_special_tokens_extended`, which was
  removed in transformers 5.x. Unpinned `transformers` in 1.9.1/1.9.2
  let pip resolve to 5.9.0, causing `AttributeError` at vLLM startup.
- The compat range now keeps Gemma3Config available (the v1.9 → v1.9.1
  fix) AND keeps the tokenizer API stable for vLLM 0.11.

---

## [1.9.3] — supervisord rpcinterface syntax

### Fixed
- Corrected `supervisor.rpcinterface_factory` value to use the colon
  module:attr form (`supervisor.rpcinterface:make_main_rpcinterface`)
  instead of dotted Python path. With the wrong separator, supervisord
  crashed at config-parse time before spawning any subprocess.
- Since supervisord runs as PID 1 via entrypoint exec, that parse error
  killed the container and RunPod respawned it into a crash loop with no
  in-pod recovery short of a new image.

---

## [1.9.2] — CUDA 12 vLLM pin + compactor env handling

### Security / runtime
- **Pinned vllm `==0.11.0`** with `--extra-index-url cu128` to keep
  PyTorch on cu128 wheels. Modern vLLM (0.21+) ships cu130 wheels which
  require NVIDIA driver 580+; most RunPod hosts (including the A40 fleet)
  are still on driver 570 (CUDA 12.8 max). Without this pin, vLLM crashed
  at startup with "NVIDIA driver too old (found version 12080)".

### Fixed
- Compactor `_env_int()` helper handles empty-string env vars. `.env`
  files set keys to `""` for opt-in blanks; `os.environ.get(name, default)`
  returns `""` not the default, and `int("")` crashed at compactor module
  import time. Added regression test.

---

## [1.9.1] — Dep pin conflict + supervisorctl socket

### Fixed
- **Unpinned `fastapi` / `uvicorn` / `httpx` / `transformers`** in
  `compactor/requirements.txt`. The 1.9 pins (transformers==4.47.1
  specifically) caused pip to *downgrade* what vLLM had installed,
  leaving transformers below the 4.50 minimum that modern vLLM
  unconditionally imports (`Gemma3Config`). Result: vLLM crashed at
  import.
- Added the `[unix_http_server]` / `[supervisorctl]` /
  `[rpcinterface:supervisor]` sections to `supervisord.conf` so
  `supervisorctl` actually has a control socket to connect to. The 1.9
  image's supervisord ran fine but had no IPC surface — restart-from-pod
  required `kill`-ing PIDs manually.
- Dropped deprecated `TRANSFORMERS_CACHE` env var (transformers v5
  removes it; `HF_HOME` alone is the modern equivalent and covers both
  transformers and huggingface_hub).

---

## [1.9] — Migration from llama.cpp to vLLM

**The big rewrite.** Replaced the entire inference engine and added
the context-compactor middleware that prevents long-conversation context
loss.

### Added
- **vLLM** as the inference engine, replacing llama.cpp. Native
  HuggingFace safetensors support — any vllm-compatible HF causal-LM
  loads by repo ID with no GGUF gymnastics.
- **context-compactor** FastAPI middleware (`compactor/main.py`). Counts
  tokens with the target model's own tokenizer; when a request exceeds
  `COMPACTOR_TARGET_TOKENS` (default 75% of `MAX_MODEL_LEN`), older
  turns get summarized into a single system block via an extra LLM call
  and the original messages are replaced. Streaming responses are
  proxied verbatim. Backend-agnostic (works against any OpenAI-compatible
  endpoint).
- **Single `/data` Network Volume** layout — both model cache
  (`HF_HOME`) and OpenWebUI state (`DATA_DIR`) live on one volume.
  Simpler RunPod deploys, fewer moving parts.
- **`VLLM_EXTRA_ARGS` env var** for passing arbitrary flags to vLLM
  (`--quantization fp8`, `--tensor-parallel-size N`, etc.) without
  rebuilding.
- **Default model changed to** `anthracite-org/magnum-v4-22b` — creative
  writing fine-tune of Mistral-Small, lightly aligned. Several
  alternative presets commented in `.env.example`.

### Changed
- Dockerfile rewritten end-to-end: dropped the llama.cpp builder stage,
  switched to single-stage with atomic install+strip+cleanup per venv to
  keep the image at ~16 GB.
- Default `MAX_MODEL_LEN=32768` (vs llama.cpp's 32K cap which was a hard
  wall; with the compactor, this is the engine ceiling but conversations
  can effectively run longer via summarization).

### Documentation
- `RUNPOD_DEPLOY.md` rewritten for the vLLM + Network Volume pattern,
  including the pre-warm-on-CPU-pod cost optimization.
- `compactor/V2_PLAN.md` design spec for the next memory iteration.

---

## [1.8] — Previous llama.cpp release

Last release in the llama.cpp era. See git history for details — superseded
by 1.9's rewrite.

---

## [1.7] — Modular AI model config

Modular AI model configuration. Removed ability to run two AI models at
once in the same container due to operational complexity.

---

## Earlier versions

See git history for releases prior to 1.7.
