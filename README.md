# Zion's Light AI

A self-hosted creative-writing assistant that **remembers**. An
OpenAI-compatible chat backend with a custom memory middleware that gives
long conversations persistent, structured recall — facts, semantic
retrieval over past turns, hierarchical summaries, and durable personas —
so the model doesn't lose the thread as a story or project grows past the
context window.

Packaged as a single Docker image for one-click deploy to
[RunPod](https://www.runpod.io/), but runs locally on any NVIDIA GPU host.

```
┌────────────┐    ┌────────────────────┐    ┌─────────┐
│ OpenWebUI  │ →  │  context-compactor │ →  │  vLLM   │
│  :3000     │    │       :8080        │    │  :8000  │
│ (user UI)  │    │  OpenAI-compatible │    │ (model) │
│            │    │  memory + summary  │    │         │
└────────────┘    └────────────────────┘    └─────────┘
                            │                     │
                            ▼                     ▼
                  /data/openwebui/compactor   /data/models
                  (facts, RAG, summaries,     (HF weights cache)
                   personas — per conv)
                            │
                            ▼
                  single RunPod Network Volume at /data
                  (survives pod terminations)
```

## Features

**Conversation**
- OpenAI-compatible API (`/v1/chat/completions`, `/v1/models`) — works with any OpenAI client, streaming or not
- Any vLLM-supported HuggingFace causal-LM, swappable via one env var

**Memory** (the point of the project — see [USER_GUIDE.md](USER_GUIDE.md))
- **Facts** — durable claims the model extracts and re-injects each turn
- **RAG** — every exchange embedded into ChromaDB and semantically retrieved later
- **Hierarchical summaries** — rolling L1→L2→L3 compression so 1000-turn chats still fit
- **Personas** — long role/voice system prompts stored as a first-class layer, exempt from summarization and eviction
- **Semantic dedup** — near-duplicate facts merged automatically (embedding + LLM verify) so you never see the same thing three ways
- All memory is **per-conversation**, persisted to the Network Volume, and survives restarts

**User control** (chat slash-commands — [full reference](USER_GUIDE.md#slash-commands))
- `/list-facts`, `/remember <text>`, `/forget [substring]`, `/why`, `/help` — inspect and steer what the model knows, with zero LLM cost

**Portability & ops**
- Export / import / fork a conversation's entire memory as one JSON bundle
- `GET /health/full` deep healthcheck + post-boot self-test that proves the deploy actually works
- Single Network Volume holds weights *and* memory; pre-warm once, attach forever

## Quick start

### RunPod (production)

Full walkthrough in [RUNPOD_DEPLOY.md](RUNPOD_DEPLOY.md). TL;DR:

1. Create a ~200 GB Network Volume named `zions-data`
2. Pre-warm the model on a cheap CPU pod (one-time, optional)
3. Deploy a GPU pod from the [Docker Hub image](https://hub.docker.com/r/angreg/zions-light-ai) with the volume attached at `/data`, ports `3000, 8080` exposed
4. **Pick a model that fits your GPU** — see the warning below

> **⚠️ A40 users: set `MODEL_REPO=anthracite-org/magnum-v4-12b`.** The image's
> built-in default is the 22B model, which **does not fit an A40** — runtime
> FP8 quantization OOMs during the marlin repack step. The 12B model runs
> comfortably in FP16 at 32K context on an A40. Reserve the 22B for A100-class
> cards in FP16. See [GPU sizing](RUNPOD_DEPLOY.md#gpu-sizing).

### Local (dev / testing)

Requires NVIDIA GPU + Docker Desktop with WSL2 (Windows) or Docker +
nvidia-container-toolkit (Linux).

```bash
cp .env.example .env
# For consumer GPUs, edit .env to a small model:
#   MODEL_REPO=Qwen/Qwen2.5-1.5B-Instruct   (fits 8 GB)
docker compose up --build
# OpenWebUI → http://localhost:3000
```

## Using the assistant

Once it's running, **[USER_GUIDE.md](USER_GUIDE.md)** is the place to start —
it explains the memory model in plain language, documents every slash
command, shows how to set up a persona, and lists the admin endpoints for
inspecting or backing up a conversation's memory.

## Image tags

Published at [`angreg/zions-light-ai`](https://hub.docker.com/r/angreg/zions-light-ai).
Pin a specific version for reproducible deploys.

| Tag | Contents |
|---|---|
| `:v2.1` | Rolling V2.1 — full memory + user control + observability |
| `:v2.1-phase8` / `:v2.1-complete` | V2.1 final: chat commands + personas |
| `:v2.1-phase7` | + semantic dedup + stale-fact archival |
| `:v2.1-phase6.1` | + observability (`/health/full`, boot self-test) |
| `:v2.0` | Three-layer memory (facts + RAG + hierarchical summaries) |
| `:1.9.6` | Final V1 — auto-summarization only, no persistent memory |
| `:latest` | Promoted to the newest validated release |

See [CHANGELOG.md](CHANGELOG.md) for full version history.

## Documentation

| Doc | For | Covers |
|---|---|---|
| [USER_GUIDE.md](USER_GUIDE.md) | **Users** | Memory model, slash commands, personas, admin endpoints, FAQ |
| [RUNPOD_DEPLOY.md](RUNPOD_DEPLOY.md) | **Operators** | RunPod deploy, GPU sizing, env vars, troubleshooting |
| [OPERATIONS.md](OPERATIONS.md) | **Operators** | Runbook: health, log reference, failure recovery, backups/restore, rollback |
| [CHANGELOG.md](CHANGELOG.md) | Everyone | Per-version history |
| [ROADMAP.md](ROADMAP.md) | Contributors | V1 → V4 forward plan |
| [TESTING.md](TESTING.md) | Contributors | Three-tier testing standard + run commands |
| [compactor/V2_PLAN.md](compactor/V2_PLAN.md) | Contributors | Memory architecture design spec |
| [compactor/V4_PLAN.md](compactor/V4_PLAN.md) | Contributors | Agentic / tool-use design spec |

## Project structure

```
.
├── Dockerfile              # Multi-process image (parametric CUDA build args)
├── docker-compose.yml      # Local dev / single-host orchestration
├── supervisord.conf        # Runs vllm + compactor + openwebui + boot self-test
├── entrypoint.sh           # Preflight checks, then hands off to supervisord
├── .env.example            # Every configurable knob, documented
├── compactor/              # The memory + summarization middleware (FastAPI)
│   ├── main.py             # Request flow: commands → compaction → memory inject → proxy → async tail
│   ├── memory.py           # conv_id resolution, storage layout, atomic I/O, locks
│   ├── facts.py            # Facts: extract / prune / inject / archive (Phase 2 + 7)
│   ├── retrieval.py        # Episodic RAG: embeddings + ChromaDB (Phase 3)
│   ├── summarizer.py       # Hierarchical L1→L2→L3 summaries (Phase 4)
│   ├── backfill.py         # Lazy backfill of pre-V2 conversations
│   ├── dedup.py            # Hybrid embedding+LLM fact deduplication (Phase 7)
│   ├── commands.py         # Chat slash-command surface (Phase 5)
│   ├── persona.py          # Personas as first-class memory (Phase 8)
│   ├── health.py           # /health/full deep probe (Phase 6)
│   ├── selftest.py         # Post-boot live-stack self-test (Phase 6)
│   ├── portability.py      # Export / import / fork bundles (Phase 6)
│   ├── test_*.py           # 12 Tier-1 unit suites (CPU-only)
│   └── V2_PLAN.md          # Memory architecture spec
├── pipelines/              # OpenWebUI Functions
│   └── conversation_id_header.py  # Propagates chat_id → compactor conv_id
├── tests/integration/      # Tier-3 black-box suite (run against a live pod)
├── README.md               # This file
├── USER_GUIDE.md           # End-user guide
├── RUNPOD_DEPLOY.md        # RunPod deploy walkthrough
├── TESTING.md              # Testing standard
├── CHANGELOG.md            # Per-version history
└── ROADMAP.md              # Forward plan
```

## Roadmap

See [ROADMAP.md](ROADMAP.md). High-level:

- **V1.9.6** ✅ — final V1: vLLM 0.14.1 (CVE fix), parametric CUDA, persistent compile cache, preflight checks
- **V2.0** ✅ — memory architecture: persistent facts, RAG (ChromaDB), hierarchical summarization
- **V2.1** ✅ — user control: chat commands, personas, export/import, dedup, archival, observability
- **V2.2** ✅ — testing & observability: boot self-test, `/health/full`, three-tier standard ([TESTING.md](TESTING.md))
- **V2.3** — resilience & stability: durable backups + verified restore, chaos tests, operational runbook *(quality over speed)*
- **V3** — multimodal: vision (VLM swap), speech-to-text (Whisper), text-to-speech (Kokoro/XTTS)
- **V4** — agentic: model tool-use via a compactor tool-loop, sandboxed command execution, eventual agent-run harness ([compactor/V4_PLAN.md](compactor/V4_PLAN.md))

## Tech stack

| Layer | Component |
|---|---|
| Inference engine | [vLLM](https://github.com/vllm-project/vllm) 0.14.1 (cu128 wheels for RunPod A40 compat) |
| Chat frontend | [OpenWebUI](https://github.com/open-webui/open-webui) |
| Memory middleware | Custom FastAPI compactor (`compactor/`, torch-free venv) |
| Embeddings | BAAI/bge-small-en-v1.5 (ONNX, prebaked) via fastembed + ChromaDB |
| Process supervision | supervisord |
| Container base | `nvidia/cuda` runtime (parametric build args — cu128 default) |
| Recommended model | Magnum v4 **12B** on A40 / 22B on A100 (or any vLLM HF causal-LM) |

## License

See LICENSE file (if present). Bundled software keeps its own licenses:
vLLM (Apache 2.0), OpenWebUI (MIT), Magnum v4 and base models (each repo's
HF license).

## Contributing

Currently a personal/single-user project. Every code change follows the
[testing standard](TESTING.md) (Tier-1 in the same PR; green before merge).
Found it useful? Open an issue or PR.
