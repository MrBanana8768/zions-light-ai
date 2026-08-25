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

> **A40 users: the defaults just work (rc8+).** The image's built-in default
> is the production-validated A40 config — Cydonia-24B with runtime fp8 —
> so a bare deploy boots out of the box. If a 24B + fp8 boot ever OOMs on
> your host, the always-fits fallback is `anthracite-org/magnum-v4-12b` in
> FP16 (empty `VLLM_EXTRA_ARGS`). On images **older than rc8** the default
> was a 22B that did *not* fit an A40 — override per
> [runpod.env.template](runpod.env.template). See
> [GPU sizing](RUNPOD_DEPLOY.md#gpu-sizing).

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
| `:v3.0-cu12` = `:v3.0` | **Current release** (= the validated rc8 build) — V3.0 consolidation: audited dep pins, OpenWebUI 0.11, SQLite network-volume hardening, chat-proxy guards; CUDA-12 profile (any A40 host) |
| `:v3.0-rc5-cu12` / `:v3.0-rc6-cu12` / `:v3.0-rc7-cu12` | Superseded rcs — rc5 lacks the overflow fix; rc6 lacks the rc7 review fixes (compaction alternation blocker); rc7 is code-identical to rc8 but ships the old unbootable-on-A40 22B default |
| `:v3-snapshot` | Frozen last-known-good V3.3 image (= `:v3.3-tts`) — rollback target |
| `:v3.3-tts` / `:v3.2-stt` / `:v3.1-vision` | The V3.x feature line as shipped incrementally |
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
| [runpod.env.template](runpod.env.template) | **Operators** | Paste-ready RunPod pod template — every env var, the pod settings, and which ones actually matter |
| [OPERATIONS.md](OPERATIONS.md) | **Operators** | Runbook: health, log reference, failure recovery, backups/restore, rollback |
| [CHANGELOG.md](CHANGELOG.md) | Everyone | Per-version history |
| [ROADMAP.md](ROADMAP.md) | Contributors | V1 → V4 forward plan |
| [TESTING.md](TESTING.md) | Contributors | Three-tier testing standard + run commands |
| [compactor/V2_PLAN.md](compactor/V2_PLAN.md) | Contributors | Memory architecture design spec |
| [compactor/V4_PLAN.md](compactor/V4_PLAN.md) | Contributors | Agentic / tool-use design spec |
| [COGNITIVE_ARCHITECTURE.md](COGNITIVE_ARCHITECTURE.md) | Contributors | North star — the stateless→stateful arc, faculties, and the honesty/reverence principles |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Contributors | System shape — layering tiers, networking/trust boundaries, scaling, and the V4-era split decisions |
| [FRONTEND_SPEC.md](FRONTEND_SPEC.md) | Contributors | Specification for the purpose-built replacement client — compactor-native data flow, memory as a UI surface, the north-star obligations on the interface |
| [REMEDIATION.md](REMEDIATION.md) | Contributors | **The v3.1 action list.** Findings from two independent codebase reviews, phased, with ordering constraints and verification steps. Start here to do the work |
| [INCIDENT_2026-08-24.md](INCIDENT_2026-08-24.md) | Contributors | Incident report — the conversation-chain truncation, its fault tree, and what the diagnosis got wrong. Background for REMEDIATION.md |
| [FINETUNE_PLAN.md](FINETUNE_PLAN.md) | Contributors | Fine-tune track — crafting a custom voice (QLoRA), a parallel track to the V-line |
| [SEED_MODEL_PLAN.md](SEED_MODEL_PLAN.md) | Contributors | Far-future foundation — values-in-pretraining (the tabula-rasa / self-modification frontiers). **Partly superseded** — the MoE/BTX pipeline was dropped; see the notice at the top of the file |

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
- **V3** — multimodal: vision (VLM swap), speech-to-text (Whisper), text-to-speech (Piper; Kokoro optional swap)
- **V4** — agentic: model tool-use via a compactor tool-loop, sandboxed command execution, eventual agent-run harness ([compactor/V4_PLAN.md](compactor/V4_PLAN.md))

## Tech stack

| Layer | Component |
|---|---|
| Inference engine | [vLLM](https://github.com/vllm-project/vllm) 0.24.0 — cu130/CUDA 13 default (driver ≥580); cu128 + vLLM 0.19.0 CUDA-12 fallback (driver-570 hosts) |
| Chat frontend | [OpenWebUI](https://github.com/open-webui/open-webui) |
| Memory middleware | Custom FastAPI compactor (`compactor/`, torch-free venv) |
| Embeddings | BAAI/bge-small-en-v1.5 (ONNX, prebaked) via fastembed + ChromaDB |
| Process supervision | supervisord |
| Container base | `nvidia/cuda` runtime (parametric — cu130/CUDA 13 default, cu128/CUDA 12 fallback) |
| Recommended model | Magnum v4 **12B** on A40 / 22B on A100 (or any vLLM HF causal-LM) |

## License

This project's **own code** (the compactor, `stt/`, Dockerfile, supervisord
config, entrypoint, and docs) is licensed under the **PolyForm Noncommercial
License 1.0.0** — see [LICENSE.md](LICENSE.md) (SPDX:
`PolyForm-Noncommercial-1.0.0`).

- ✅ **Free for any noncommercial use** — personal, research, nonprofit,
  educational, hobby, religious.
- 💼 **Commercial use requires the author's written permission.** The author
  holds the copyright and can grant commercial licenses on request — just ask.
- 💵 **Personal financial gain counts as commercial.** Using this to make
  money — a paid service, business/freelance/consulting work, or any
  revenue-generating activity, individual *or* company — requires a commercial
  license.
- 🔄 **Commercialized outputs cross the line too.** If a work made during
  personal use is *later* sold or monetized (e.g. a story written with the app
  and then sold), that use becomes commercial and requires a license; the
  author reserves the right to a reasonable share of the proceeds, agreed in
  that license.

This is **source-available**, not OSI "open source" (open-source licenses
can't restrict commercial use). That's deliberate.

Bundled third-party software keeps its own licenses and terms — notably
**vLLM** (Apache-2.0), **OpenWebUI** (its own license, including branding
terms), and the **model weights** (each HuggingFace repo's license). Those are
unaffected by this project's license; anyone redistributing the integrated
whole must comply with all of them as well.

## Contributing

Currently a personal/single-user project. Every code change follows the
[testing standard](TESTING.md) (Tier-1 in the same PR; green before merge).
Found it useful? Open an issue or PR.
