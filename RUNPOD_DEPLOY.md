# Runpod Deployment Guide

Deploy any HuggingFace causal-LM with vLLM, automatic Claude-style context
compression, and OpenWebUI on Runpod.

## Stack

- **vLLM** — OpenAI-compatible inference server (HuggingFace-native, paged attention, prefix caching)
- **context-compactor** — middleware proxy that auto-summarizes older turns when the conversation approaches the context limit
- **OpenWebUI** — chat frontend

Request flow: `OpenWebUI :3000` → `compactor :8080` → `vLLM :8000`

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
├── models/       # vLLM cache (HF_HOME) — 30-150 GB depending on model
└── openwebui/    # OpenWebUI SQLite, uploads, settings — usually <1 GB
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
   huggingface-cli download anthracite-org/magnum-v4-22b
   ```
4. When the download completes, terminate the CPU pod. Your weights and any
   OpenWebUI state stay on the volume.

You can repeat the download step to pre-cache additional models on the same
volume — vLLM picks whichever one matches `MODEL_REPO` at runtime.

### Step 3: Build and push the image

Pre-built images are published at `angreg/zions-light-ai` on Docker Hub. To
pull a pinned version use `angreg/zions-light-ai:1.9` (recommended for
reproducibility); to always grab the newest use `angreg/zions-light-ai:latest`.

To build and publish your own:
```bash
docker build -t angreg/zions-light-ai:1.9 -t angreg/zions-light-ai:latest .
docker push angreg/zions-light-ai:1.9
docker push angreg/zions-light-ai:latest
```

### Step 4: Create the Runpod Template

Go to [Runpod Templates](https://www.runpod.io/console/user/templates) → New Template:

- **Template Name:** `zions-light-ai`
- **Container Image:** `angreg/zions-light-ai:1.9` *(or `:latest`)*
- **Container Disk:** `60 GB` (room for the image, supervisor logs, scratch)
- **Volume Mount Path:** `/data` (← this is where the Network Volume attaches)
- **Expose HTTP Ports:** `3000, 8080`
- **Docker Command:** (leave empty)
- **Environment Variables:** (see table below — at minimum set `MODEL_REPO` if different from default)

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
  --image "angreg/zions-light-ai:1.9" \
  --disk-size 60 \
  --network-volume-id "<your-volume-id>" \
  --ports "3000/http,8080/http"
```

## GPU sizing

| Model | Quant | VRAM | Suggested Runpod GPU |
|---|---|---|---|
| Qwen2.5-1.5B-Instruct | FP16 | ~6 GB | RTX 3090 / 4090 |
| anthracite-org/magnum-v4-12b | FP16 | ~24 GB | A40 |
| **anthracite-org/magnum-v4-22b** *(default)* | **FP8** | **~24 GB** | **A40** |
| anthracite-org/magnum-v4-22b | FP16 | ~44 GB | A40 (tight) / A100 |
| Qwen2.5-32B-Instruct | FP16 | ~64 GB | A100 80GB |
| Llama-3.3-70B-Instruct | FP16 | ~140 GB | 2× A100 80GB |

The default uses FP8 weight quantization via `VLLM_EXTRA_ARGS=--quantization fp8` —
halves VRAM usage with ~99% quality retention. Drop the flag if you have VRAM
headroom and want pure FP16.

## Access Your Deployment

Once deployed, access via Runpod's proxy URLs:

- **OpenWebUI:** `https://{POD_ID}-3000.proxy.runpod.net`
- **API (with compaction):** `https://{POD_ID}-8080.proxy.runpod.net`

## Environment Variables

Override these in your Runpod template if needed:

| Variable | Default | Description |
|---|---|---|
| `MODEL_REPO` | `anthracite-org/magnum-v4-22b` | Any vLLM-compatible HuggingFace repo |
| `MAX_MODEL_LEN` | `32768` | vLLM context window (tokens) |
| `GPU_MEMORY_UTILIZATION` | `0.90` | Fraction of VRAM vLLM may use |
| `VLLM_EXTRA_ARGS` | `--quantization fp8` | Extra flags appended to the vLLM command line (quantization, tensor-parallel, etc.) |
| `COMPACTOR_TARGET_TOKENS` | *75% of `MAX_MODEL_LEN`* | When request exceeds this, older turns get summarized |
| `COMPACTOR_KEEP_RECENT_TURNS` | `4` | Recent turns preserved verbatim during compaction |
| `COMPACTOR_SUMMARY_MAX_TOKENS` | `1024` | Max length of the summary the compactor generates |
| `WEBUI_AUTH` | `true` | Require OpenWebUI login |
| `HF_TOKEN` | *(unset)* | Needed for gated models (Llama, Mistral, etc.) |

## API Usage

The compactor exposes an OpenAI-compatible API at port 8080:

```bash
curl https://{POD_ID}-8080.proxy.runpod.net/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthracite-org/magnum-v4-22b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

curl https://{POD_ID}-8080.proxy.runpod.net/v1/models
```

Long conversations are automatically compacted — no client changes needed.

## Troubleshooting

### Check Logs
```bash
# Via Runpod web terminal
cat /var/log/supervisor/vllm.log         # inference engine
cat /var/log/supervisor/compactor.log    # compaction events
cat /var/log/supervisor/openwebui.log    # frontend
```

### Watch compaction in real time
```bash
tail -f /var/log/supervisor/compactor.log
# Look for: "compacted: summarized N messages, X -> Y tokens"
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
