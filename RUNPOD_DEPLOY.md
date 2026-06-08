# Runpod Deployment Guide

Deploy any HuggingFace causal-LM with vLLM, automatic Claude-style context
compression, and OpenWebUI on Runpod.

## Stack

- **vLLM** — OpenAI-compatible inference server (HuggingFace-native, paged attention, prefix caching)
- **context-compactor** — memory middleware: persistent facts, RAG over past turns, hierarchical summaries, personas, chat commands; auto-summarizes older turns near the context limit
- **OpenWebUI** — chat frontend

Request flow: `OpenWebUI :3000` → `compactor :8080` → `vLLM :8000`

> For *using* the deployed assistant (memory, slash commands, admin
> endpoints), see [USER_GUIDE.md](USER_GUIDE.md). This document is about
> standing it up.

## Quick Start

The recommended deployment uses a **single Network Volume** mounted at `/data`
that holds *both* the model cache and OpenWebUI's chat history. Network Volumes
persist across pod lifecycles, can be attached to any pod in the same
datacenter region, and let you pre-download weights on a cheap CPU pod before
spinning up the expensive GPU. Storage is roughly $0.05/GB/month — trivial
compared to the cost of re-downloading 30-100 GB of weights every cold start
on a $2/hr GPU.

**Volume layout:**
```
/data/
├── models/                  # vLLM cache (HF_HOME) — 30-150 GB depending on model
└── openwebui/
    ├── (OpenWebUI SQLite, uploads, settings — usually <1 GB)
    └── compactor/           # V2 memory — facts, summaries, chromadb, personas
        ├── facts/           #   per-conv facts + archive sidecars
        ├── summaries/       #   per-conv L1/L2/L3 summary state
        ├── chromadb/        #   episodic RAG vector store
        └── personas/        #   per-conv persona text
```

### Step 1: Create the Network Volume

1. Go to [Runpod Storage → Network Volumes](https://www.runpod.io/console/user/storage)
2. Click **New Network Volume**
3. Configure:
   - **Name:** `zions-data`
   - **Datacenter:** pick one (pods must be in the same DC to attach)
   - **Size:** `200 GB` (room for 1-2 large models + OpenWebUI state; resize later if needed)

### Step 2: Pre-warm the volume (one-time, optional but recommended)

Spin up a cheap CPU-only pod with the volume attached to populate the model
cache before paying GPU prices:

1. Deploy a CPU pod (any cheap template — `runpod/cpu-base:latest` works)
2. Attach the `zions-data` Network Volume at `/data`
3. SSH in and run:
   ```bash
   pip install huggingface_hub
   mkdir -p /data/models /data/openwebui
   export HF_HOME=/data/models
   # For gated models (Llama, Mistral):
   # export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   # 12B is the recommended A40 model (see GPU sizing):
   huggingface-cli download anthracite-org/magnum-v4-12b
   # On A100-class cards you can use the 22B instead:
   # huggingface-cli download anthracite-org/magnum-v4-22b
   ```
4. When the download completes, terminate the CPU pod. Your weights and any
   OpenWebUI state stay on the volume.

You can repeat the download step to pre-cache additional models on the same
volume — vLLM picks whichever one matches `MODEL_REPO` at runtime.

### Step 3: Build and push the image

Pre-built images are published at `angreg/zions-light-ai` on Docker Hub.
Pin a version for reproducibility (e.g. `angreg/zions-light-ai:v2.1`) or use
`:latest` for the newest validated release. See the
[image-tags table in the README](README.md#image-tags) for what each tag
contains.

To build and publish your own:
```bash
docker build -t angreg/zions-light-ai:v2.1 -t angreg/zions-light-ai:latest .
docker push angreg/zions-light-ai:v2.1
docker push angreg/zions-light-ai:latest
```

### Step 4: Create the Runpod Template

Go to [Runpod Templates](https://www.runpod.io/console/user/templates) → New Template:

- **Template Name:** `zions-light-ai`
- **Container Image:** `angreg/zions-light-ai:v2.1` *(or `:latest`)*
- **Container Disk:** `60 GB` (room for the image, supervisor logs, scratch)
- **Volume Mount Path:** `/data` (← this is where the Network Volume attaches)
- **Expose HTTP Ports:** `3000, 8080`
- **Docker Command:** (leave empty)
- **Environment Variables:** (see table below) — **on an A40, set
  `MODEL_REPO=anthracite-org/magnum-v4-12b`.** The image default is the 22B
  model, which does not fit an A40 (see GPU sizing).

### Step 5: Deploy the Pod

1. Go to [GPU Cloud](https://www.runpod.io/console/gpu-cloud)
2. Select your template
3. **Attach the `zions-data` Network Volume** (this is the key step — find the toggle in the pod-config UI)
4. Choose GPU sized for your model (see table below)
5. Deploy

Cold starts after the first one will skip the model download entirely — vLLM
finds the cached weights under `/data/models/hub/` and loads them straight from
disk. OpenWebUI also picks up its existing SQLite from `/data/openwebui` so
chat history survives pod terminations.

### Alternative: Deploy via Runpod CLI

```bash
pip install runpod
runpod config

runpod pod create \
  --gpu-type "NVIDIA A40" \
  --image "angreg/zions-light-ai:v2.1" \
  --disk-size 60 \
  --network-volume-id "<your-volume-id>" \
  --env MODEL_REPO=anthracite-org/magnum-v4-12b \
  --ports "3000/http,8080/http"
```

## GPU sizing

| Model | Quant | VRAM | Suggested Runpod GPU |
|---|---|---|---|
| Qwen2.5-1.5B-Instruct | FP16 | ~6 GB | RTX 3090 / 4090 |
| **anthracite-org/magnum-v4-12b** *(recommended on A40)* | **FP16** | **~24 GB** | **A40** |
| anthracite-org/magnum-v4-22b | FP16 | ~44 GB | A100 (40/80 GB) |
| Qwen2.5-32B-Instruct | FP16 | ~64 GB | A100 80GB |
| Llama-3.3-70B-Instruct | FP16 | ~140 GB | 2× A100 80GB |
| **Vision (V3.1) — Qwen2-VL-7B-Instruct** | FP16 | ~16 GB | A40 |
| Vision — Pixtral-12B-2409 | FP16 | ~24 GB | A40 (tight) / A100 |
| Vision — Llama-3.2-11B-Vision-Instruct *(gated)* | FP16 | ~24 GB | A40 (tight) / A100 |

> **⚠️ Do not run 22B with `--quantization fp8` on an A40.** Runtime FP8
> quantization needs the *full FP16 weights resident in VRAM first* to do
> the marlin repack, then frees them — so peak memory exceeds 44 GB and the
> A40's 48 GB doesn't leave enough headroom; it OOMs during startup. FP8
> only helps if you have an offline-quantized FP8 checkpoint, which removes
> the repack step. On an A40, **run the 12B in FP16** (the default
> `VLLM_EXTRA_ARGS` is empty — no quantization flag needed). Reserve the
> 22B for A100-class cards in FP16.

The image's built-in `MODEL_REPO` default is the 22B for historical
reasons; **override it to `anthracite-org/magnum-v4-12b` on A40-class
hardware.**

### Vision (V3.1) — enabling image understanding

Set `MODEL_REPO` to a vision-language model (see presets in `.env.example`)
and image upload in OpenWebUI works with no other changes — it sends images
in OpenAI's standard multimodal format, vLLM serves them on the same API, and
the compactor handles them correctly:

- **Image turns survive compaction** — they're kept verbatim rather than
  summarized to text (which would lose the image), so the model can still see
  an image many turns later.
- **Image tokens are budgeted** — `COMPACTOR_IMAGE_TOKENS` (default 768) is
  added per image so long, image-heavy threads don't overflow the context
  window. Raise it if the model errors on big image threads.

Most VLMs want `--limit-mm-per-prompt image=N` in `VLLM_EXTRA_ARGS`; some
(Pixtral) want `--tokenizer-mode mistral`. The creative-writing models and the
best vision models are not the same model today, so this is an opt-in swap.

## Access Your Deployment

Once deployed, access via Runpod's proxy URLs:

- **OpenWebUI:** `https://{POD_ID}-3000.proxy.runpod.net`
- **API (with memory + compaction):** `https://{POD_ID}-8080.proxy.runpod.net`
- **Deep health:** `https://{POD_ID}-8080.proxy.runpod.net/health/full`

Admin endpoints (`/admin/*` — facts, personas, export/import, dedup) are
**localhost-only by default** and intentionally *not* reachable over the
proxy. Use them from the RunPod Web Terminal (`curl localhost:8080/...`).
See [USER_GUIDE.md](USER_GUIDE.md#power-user-admin-endpoints).

## Environment Variables

Override these in your Runpod template if needed:

**Core (model + inference):**

| Variable | Default | Description |
|---|---|---|
| `MODEL_REPO` | `anthracite-org/magnum-v4-22b` | Any vLLM-compatible HF repo. **Set to `…-12b` on A40.** |
| `MAX_MODEL_LEN` | `32768` | vLLM context window (tokens) |
| `GPU_MEMORY_UTILIZATION` | `0.90` | Fraction of VRAM vLLM may use |
| `VLLM_EXTRA_ARGS` | *(empty)* | Extra flags appended to the vLLM command line (tensor-parallel, offline-FP8 checkpoint, etc.). **Leave empty for FP16** — do not add `--quantization fp8` on A40 (see GPU sizing). |
| `WEBUI_AUTH` | `true` | Require OpenWebUI login |
| `HF_TOKEN` | *(unset)* | Needed for gated models (Llama, Mistral, etc.) |

**Compaction (V1 summarization):**

| Variable | Default | Description |
|---|---|---|
| `COMPACTOR_TARGET_TOKENS` | *75% of `MAX_MODEL_LEN`* | When a request exceeds this, older turns get summarized |
| `COMPACTOR_KEEP_RECENT_TURNS` | `4` | Recent turns preserved verbatim during compaction |
| `COMPACTOR_SUMMARY_MAX_TOKENS` | `1024` | Max length of a generated summary |

**Memory (V2 — all enabled by default):**

| Variable | Default | Description |
|---|---|---|
| `COMPACTOR_FACTS_EXTRACTION` | `true` | Extract durable facts after each turn. Set `false` to disable. |
| `COMPACTOR_MAX_FACTS_TOKENS` | `1500` | Token budget for the facts block (LRU-evicted past this) |
| `COMPACTOR_RAG_ENABLED` | `true` | Episodic RAG over past turns (ChromaDB). Set `false` to disable. |
| `COMPACTOR_RAG_TOP_K` | `5` | How many past exchanges to retrieve per turn |
| `COMPACTOR_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model (prebaked ONNX in the image) |
| `COMPACTOR_HIERARCHICAL_SUMMARY` | `true` | L1→L2→L3 rolling summaries. Set `false` to disable. |
| `COMPACTOR_DEDUP_SIMILARITY` | `0.75` | Cosine threshold for fact-dedup candidate clustering |
| `COMPACTOR_DEDUP_MAX_LLM_CALLS` | `10` | Cap on LLM merge calls per dedup pass |
| `COMPACTOR_ARCHIVE_DEFAULT_DAYS` | `90` | Default staleness cutoff for fact archival |
| `COMPACTOR_PERSONA_ENABLED` | `true` | Persona detection + injection. Set `false` for V2.0 behavior. |
| `COMPACTOR_PERSONA_AUTO_DETECT_MIN_CHARS` | `200` | Min first-system-message length to auto-capture as a persona |
| `COMPACTOR_STORAGE_ROOT` | `/data/openwebui/compactor` | Where memory state is written |

**Ops (observability + safety):**

| Variable | Default | Description |
|---|---|---|
| `COMPACTOR_SELFTEST_ON_BOOT` | `true` | Run the live-stack self-test after boot, logging to `/var/log/supervisor/selftest.log` |
| `COMPACTOR_ADMIN_BIND` | `127.0.0.1` | Admin-endpoint bind address. **Keep localhost** unless you have auth/firewall in front — admin endpoints are unauthenticated. |
| `COMPACTOR_BACKUP_ENABLED` | `true` | Run the periodic verified-backup daemon (V2.3) |
| `COMPACTOR_MIN_FREE_MB_WRITES` | `200` | Pause new-memory writes (keep serving) below this free space on `/data` (V2.3) |
| `COMPACTOR_LOG_FORMAT` | `text` | `text` (human) or `json` (one object/line for aggregation) |
| `COMPACTOR_ALERT_WEBHOOK` | *(unset)* | If set, self-test + backup POST a failure alert here (Slack/Discord/generic) |

## API Usage

The compactor exposes an OpenAI-compatible API at port 8080:

```bash
curl https://{POD_ID}-8080.proxy.runpod.net/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthracite-org/magnum-v4-12b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

curl https://{POD_ID}-8080.proxy.runpod.net/v1/models
```

Long conversations are automatically compacted and memory is maintained
per-conversation — no client changes needed. To get stable per-conversation
memory through OpenWebUI, the bundled `pipelines/conversation_id_header.py`
filter propagates the chat ID; direct API callers can set an
`X-Conversation-Id` header (otherwise the compactor falls back to a content
hash). See [USER_GUIDE.md](USER_GUIDE.md).

## Troubleshooting

### Is the deploy healthy?
```bash
# Deep health probe (200 = ok/degraded, 503 = storage down)
curl -s http://localhost:8080/health/full | jq

# Post-boot self-test result — runs automatically on every start
cat /var/log/supervisor/selftest.log
# Expect: "=== N/N passed, 0 failed ==="

# On-demand self-test (real chat round-trip + facts read/write)
curl -s http://localhost:8080/admin/selftest | jq
```

### Check Logs
```bash
# Via Runpod web terminal
cat /var/log/supervisor/vllm.log         # inference engine
cat /var/log/supervisor/compactor.log    # memory + compaction events
cat /var/log/supervisor/openwebui.log    # frontend
cat /var/log/supervisor/selftest.log     # boot self-test
```

### Watch memory in real time
```bash
tail -f /var/log/supervisor/compactor.log
# Look for, per conversation:
#   "injected memory [persona(...) Nfact(s) Mretr sum(L1=.../L2=.../L3=...)]"
#   "extracted N new fact(s)"  /  "extracted 0 fact(s) — model returned: ..."
#   "indexed exchange (turn ~N)"
#   "rollup → L1=.. L2=.. L3=.."
#   "dedup merged N duplicate fact(s)"
```

### Model download is slow / failed
vLLM downloads safetensors weights on first start. If your pod has a slow
network, increase `startsecs` in `supervisord.conf` or pre-warm the model
into the cache from the pod itself:

```bash
# Inside the running pod — uses the same HF_HOME=/data/models layout vLLM expects
HF_HOME=/data/models /opt/vllm-venv/bin/huggingface-cli download "${MODEL_REPO}"
```

For gated models, set `HF_TOKEN` in your env vars first. The cleaner pattern
is to pre-warm on a cheap CPU pod with the Network Volume attached — see
**Step 2** above.

### Out of memory
Reduce `MAX_MODEL_LEN`, lower `GPU_MEMORY_UTILIZATION`, or switch to a
smaller model. Examples in `.env.example`.

### Context still feels too short
Lower `COMPACTOR_KEEP_RECENT_TURNS` (more aggressive summarization) or
raise `MAX_MODEL_LEN` if you have VRAM headroom.

## Cost Optimization

- **Network Volume for models** *(biggest single win — see Quick Start)*: model weights are 30-150 GB. Pre-warming the volume on a CPU pod ($0.05/hr) and then attaching it to your GPU pod ($0.40-$2+/hr) means cold starts skip the download entirely. At A100 prices, one avoided 60 GB re-download pays for ~3 months of volume storage.
- **Spot instances:** Use community cloud for ~70% savings on GPU time.
- **Auto-shutdown:** Configure idle timeout in Runpod settings so the pod releases the GPU when nobody's using it. The Network Volume keeps your weights ready for the next spin-up.
- **One volume, many models:** A 150 GB volume holds 2-3 mid-size models. Swap which one is active by changing `MODEL_REPO` in the pod template — no re-download.

## Local Testing

```bash
# Use the small-model preset in .env.example
docker compose build
docker compose up

# OpenWebUI
open http://localhost:3000
```
