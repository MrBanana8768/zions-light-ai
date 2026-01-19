# Runpod Deployment Guide

Deploy Qwen3-24B-A4B with llama.cpp and OpenWebUI on Runpod.

## Quick Start

### Option 1: Deploy Pre-built Image (Recommended)

1. **Build and push to Docker Hub:**
   ```bash
   docker build -t yourusername/zions-light-ai:latest .
   docker push yourusername/zions-light-ai:latest
   ```

2. **Create Runpod Template:**
   - Go to [Runpod Templates](https://www.runpod.io/console/user/templates)
   - Click "New Template"
   - Configure:
     - **Template Name:** `zions-light-ai`
     - **Container Image:** `yourusername/zions-light-ai:latest`
     - **Container Disk:** `50 GB` (for model storage)
     - **Volume Disk:** `20 GB` (for persistent data)
     - **Volume Mount Path:** `/app/data`
     - **Expose HTTP Ports:** `3000, 8080`
     - **Docker Command:** (leave empty)

3. **Deploy Pod:**
   - Go to [GPU Cloud](https://www.runpod.io/console/gpu-cloud)
   - Select your template
   - Choose **A40** GPU
   - Click Deploy

### Option 2: Deploy via Runpod CLI

```bash
# Install Runpod CLI
pip install runpod

# Configure API key
runpod config

# Deploy
runpod pod create \
  --gpu-type "NVIDIA A40" \
  --image "yourusername/zions-light-ai:latest" \
  --disk-size 50 \
  --volume-size 20 \
  --ports "3000/http,8080/http"
```

## Access Your Deployment

Once deployed, access via Runpod's proxy URLs:

- **OpenWebUI:** `https://{POD_ID}-3000.proxy.runpod.net`
- **llama.cpp API:** `https://{POD_ID}-8080.proxy.runpod.net`

## Environment Variables

Override these in your Runpod template if needed:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_FILE` | `...-Q6_K-imat.gguf` | GGUF file to download |
| `LLAMA_CTX_SIZE` | `32768` | Context window size |
| `LLAMA_N_GPU_LAYERS` | `999` | GPU layers (999 = all) |
| `LLAMA_PARALLEL` | `4` | Concurrent requests |
| `WEBUI_AUTH` | `false` | Require login |

### Alternative Quantizations

For different VRAM requirements, change `MODEL_FILE`:

| Quantization | VRAM Required | Quality |
|--------------|---------------|---------|
| `...-IQ2_M-imat.gguf` | ~8 GB | Lower |
| `...-IQ3_M-imat.gguf` | ~10 GB | Good |
| `...-IQ4_XS-imat.gguf` | ~12 GB | Good |
| `...-Q4_K_M-imat.gguf` | ~14 GB | Better |
| `...-Q5_K_M-imat.gguf` | ~16 GB | Great |
| `...-Q6_K-imat.gguf` | ~18 GB | Excellent |
| `...-Q8_0.gguf` | ~24 GB | Best |

## API Usage

The llama.cpp server exposes an OpenAI-compatible API:

```bash
# Chat completion
curl https://{POD_ID}-8080.proxy.runpod.net/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-24b-a4b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# List models
curl https://{POD_ID}-8080.proxy.runpod.net/v1/models
```

## Troubleshooting

### Check Logs
```bash
# Via Runpod web terminal
cat /var/log/supervisor/llama-server.log
cat /var/log/supervisor/openwebui.log
```

### Model Download Failed
The model (~18GB) downloads on first start. If it fails:
```bash
# Manual download
wget "https://huggingface.co/DavidAU/Qwen3-24B-A4B-Freedom-Thinking-Abliterated-Heretic-NEO-Imatrix-GGUF/resolve/main/Qwen3-24B-A4B-Freedom-Think-Ablit-Heretic-Neo-D_AU-Q6_K-imat.gguf" \
  -O /models/Qwen3-24B-A4B-Freedom-Think-Ablit-Heretic-Neo-D_AU-Q6_K-imat.gguf
```

### Out of Memory
Reduce context size or use smaller quantization:
```bash
# In template environment variables
LLAMA_CTX_SIZE=16384
MODEL_FILE=Qwen3-24B-A4B-Freedom-Think-Ablit-Heretic-Neo-D_AU-Q4_K_M-imat.gguf
```

## Cost Optimization

- **Spot instances:** Use community cloud for ~70% savings
- **Auto-shutdown:** Configure idle timeout in Runpod settings
- **Persistent volume:** Keep models on volume to avoid re-download

## Local Testing

Before deploying to Runpod, test locally:

```bash
# Build
docker compose build

# Run (requires NVIDIA GPU + nvidia-docker)
docker compose up

# Access
open http://localhost:3000
```
