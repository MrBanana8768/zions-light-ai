# Zion's Light AI

Self-hosted creative writing assistant. OpenAI-compatible chat backend with
**auto-summarizing context preservation** — long conversations don't hit
the model's context wall, they get intelligently compressed by an
LLM-summarization middleware that preserves narrative continuity.

Packaged as a single Docker image for one-click deploy to
[RunPod](https://www.runpod.io/), but runs locally on any NVIDIA GPU host.

```
┌────────────┐    ┌──────────────────┐    ┌─────────┐
│ OpenWebUI  │ →  │ context-compactor│ →  │  vLLM   │
│  :3000     │    │      :8080       │    │  :8000  │
│ (user UI)  │    │  (OpenAI-compat, │    │ (model) │
│            │    │   summarizes     │    │         │
│            │    │   when over      │    │         │
│            │    │   token budget)  │    │         │
└────────────┘    └──────────────────┘    └─────────┘
                                                ↓
                                          /data/models
                                          (HF cache,
                                           persistent
                                           Network Volume)
```

## What it does

- **Conversational chat** with any vLLM-supported HuggingFace causal-LM
- **Default model:** [`anthracite-org/magnum-v4-22b`](https://huggingface.co/anthracite-org/magnum-v4-22b) — creative writing fine-tune of Mistral-Small, lightly aligned, trained to mimic Claude's prose style
- **Auto-summarization** when conversations approach the model's max length, so the model doesn't "forget" early context
- **OpenAI-compatible API** (`/v1/chat/completions`, `/v1/models`) — works with any OpenAI client
- **Single Network Volume** holds both model weights and chat history; survives pod terminations
- **Configurable model & inference flags** via environment variables — swap models without rebuilding

## Quick start

### RunPod (production)

See [RUNPOD_DEPLOY.md](RUNPOD_DEPLOY.md) for the 5-step deploy. TL;DR:

1. Create a 200 GB Network Volume named `zions-data`
2. Pre-warm the model on a cheap CPU pod (one-time, optional)
3. Deploy a GPU pod from the [Docker Hub image](https://hub.docker.com/r/angreg/zions-light-ai) with the volume attached at `/data`
4. Set `VLLM_EXTRA_ARGS=--quantization fp8` in template env vars (required for A40-class GPUs)

### Local (dev / testing)

Requires NVIDIA GPU + Docker Desktop with WSL2 backend (Windows) or
native Docker + nvidia-container-toolkit (Linux).

```bash
cp .env.example .env
# Edit .env if you want a smaller model for local testing:
#   MODEL_REPO=Qwen/Qwen2.5-1.5B-Instruct  (fits 8GB consumer GPUs)
docker compose up --build
# OpenWebUI at http://localhost:3000
```

## Image tags

Published at [`angreg/zions-light-ai`](https://hub.docker.com/r/angreg/zions-light-ai).
Pin to a specific version for reproducible deploys:

| Tag | Status |
|---|---|
| `:1.9.6` | Latest — V1 final release, CVE-clean, parametric CUDA |
| `:1.9.5` | Last 1.9.x with the broken vllm 0.11 pin (do not use) |
| `:latest` | Currently points at `:1.9.6` |

See [CHANGELOG.md](CHANGELOG.md) for full version history.

## Project structure

```
.
├── Dockerfile              # Multi-process image (parametric CUDA build args)
├── docker-compose.yml      # Local dev / single-host orchestration
├── supervisord.conf        # Runs vllm + compactor + openwebui inside one container
├── entrypoint.sh           # Preflight checks, then hands off to supervisord
├── .env.example            # Configurable knobs (model, context, quantization, compactor budgets)
├── compactor/              # The summarization middleware (FastAPI + httpx + tiktoken)
│   ├── main.py             # v1 implementation
│   ├── requirements.txt    # pip deps
│   ├── test_smoke.py       # CPU-only unit tests (no GPU required)
│   └── V2_PLAN.md          # V2 architecture spec (memory: RAG + facts + tiered summary)
├── README.md               # This file
├── CHANGELOG.md            # Per-version history
├── ROADMAP.md              # V1 → V2 → V3 forward plan
└── RUNPOD_DEPLOY.md        # RunPod-specific deploy walkthrough
```

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the full plan. High-level:

- **V1.9.6** *(current)* — final V1 release: vLLM bumped to 0.14.1 (CVE fix), parametric CUDA build args, persistent torch.compile cache, preflight checks
- **V2.0** *(next major)* — memory architecture: RAG over conversation history, extracted persistent facts, hierarchical summarization. See [compactor/V2_PLAN.md](compactor/V2_PLAN.md)
- **V2.1** — user control of memory: chat commands (`/list-facts`, `/forget`, `/remember`), conversation export/import, observability endpoints
- **V3** — multimodal: vision (VLM swap), speech-to-text (Whisper), text-to-speech (Kokoro/XTTS)
- **Beyond V3** — agentic tools, fine-tuning pipeline, multi-user

## Tech stack

| Layer | Component |
|---|---|
| Inference engine | [vLLM](https://github.com/vllm-project/vllm) 0.14.1 (cu128 wheels for RunPod A40 compat) |
| Chat frontend | [OpenWebUI](https://github.com/open-webui/open-webui) |
| Context compactor | Custom FastAPI middleware, single file (`compactor/main.py`) |
| Process supervision | supervisord |
| Container base | `nvidia/cuda:12.6.3-runtime-ubuntu24.04` (parametric — can swap to 12.8/13.0) |
| Default model | Magnum v4 22B (or any vLLM-supported HF causal-LM via env var) |

## License

Inherits project license — see LICENSE file (if present).

Bundled software has its own licenses: vLLM (Apache 2.0), OpenWebUI (MIT),
Magnum v4 22B and base models (each repo's HF license).

## Contributing

This is currently a personal/single-user project. If you've found it
useful and want to contribute, open an issue or PR.
