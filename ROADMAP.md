# Zion's Light AI — Roadmap

Forward-looking project plan. For implementation details of each version,
see the relevant design docs ([compactor/V2_PLAN.md](compactor/V2_PLAN.md))
and the release notes in CHANGELOG.md (when added).

---

## Current capabilities (V1.9.5, becoming V1.9.6)

**One sentence:** OpenAI-compatible chat backend with auto-summarizing
context preservation, packaged for one-click deploy to RunPod.

### What the app can do today

| Capability | Implementation |
|---|---|
| **Conversational chat** with any HuggingFace causal-LM | vLLM 0.11 + Magnum v4 22B default |
| **OpenAI-compatible API** at `/v1/chat/completions` and `/v1/models` | vLLM's native endpoints + compactor passthrough |
| **Web UI** for end users | OpenWebUI at port 3000 |
| **Auto-summarizing context** when conversations approach the model's max length | context-compactor middleware: counts tokens with the model's own tokenizer, summarizes older turns into a system block when over 75% budget |
| **Configurable model swap** via single env var | `MODEL_REPO=<hf-repo>` — no rebuild |
| **Configurable inference flags** via env var | `VLLM_EXTRA_ARGS=--quantization fp8`, `--tensor-parallel-size N`, etc. |
| **Persistent model cache + chat history** | Single `/data` Network Volume on RunPod; survives pod terminations |
| **FP8 weight quantization** to fit 22B models on a 48GB A40 | vLLM runtime quantization, Marlin kernel decompression on Ampere |
| **Streaming responses** through the compactor | SSE proxied verbatim from vLLM |
| **Authenticated UI** with admin accounts | OpenWebUI's WEBUI_AUTH=true |

### What the app **cannot** do today (in V1 scope)

- ❌ Remember anything beyond the current conversation's truncated/summarized window — once summarized, exact prior text is gone
- ❌ Cross-conversation memory (each chat is isolated)
- ❌ Image understanding (text-only models)
- ❌ Voice input or output
- ❌ Tool use / function calling (architecturally possible, not wired up)
- ❌ Multi-user isolation (single-tenant assumption)
- ❌ Real-time streaming search / RAG over external documents
- ❌ Fine-tuning / personalization

---

## V1.9.6 — Final V1 release (immediate)

**Goal:** Close V1 with all known bugs and CVEs resolved. After this, the
1.9.x line is frozen except for security patches; new feature work moves to V2.

### Scope

| Item | Type | Source |
|---|---|---|
| Bump `vllm==0.11.0` → `0.14.1` | CVE fix | CVE-2026-22778 (Critical 9.8) |
| Add `apt-get upgrade -y` to baseline-patch Ubuntu packages | CVE fix | Catches ~50-70% of remaining CVEs |
| Bump base image `nvidia/cuda:12.6.3` → latest 12.6.x point release | CVE fix | Newer CUDA/driver libs |
| Verify `transformers<5` pin still holds (or lift it if vllm 0.14.1 supports 5.x) | Compat | Required by vllm bump |
| Verify CUDA wheel target (must remain cu128) | Compat | Required for RunPod A40 driver 570 |
| Add parametric CUDA build args (`CUDA_BASE_IMAGE`, `TORCH_CUDA`, already have `VLLM_VERSION`) | Foundation | Enables future cu130 variant without source change |
| Persist torch.compile cache to `/data/vllm-compile-cache` | Cold-start win | Tier-1 from V2_PLAN.md |
| Add preflight check to entrypoint.sh (GPU visible, driver version) | Diagnostics | Tier-1 from V2_PLAN.md |
| Add README.md and CHANGELOG.md at repo root | Project hygiene | Tier-1 from V2_PLAN.md |

### Effort: ~3-4 hours total

Most of it is the vllm version bump test — needs verification that
0.14.1 stays on cu128 wheels and works with our transformers pin.
Other items are small additions to the Dockerfile / entrypoint.

### Definition of "done"

- Docker Scout shows ≤5 Critical/High remaining (and they're documented as
  upstream-unfixable or accepted)
- README on GitHub gives a complete picture in <2 min of reading
- Fresh pod deploy on RunPod boots to "Application startup complete"
  without any in-pod intervention
- CHANGELOG entry covers the entire 1.9.x line

---

## V2 — Memory (split into V2.0 + V2.1)

**Principle:** memory is foundational — everything else (V3 multimodal,
tools, multi-user, personas-as-data) ultimately reads from or writes to
this layer. Build the substrate solid before bolting controls onto it.

The split is deliberate sequencing, not arbitrary versioning. V2.1
features are *additive* to V2.0 — same storage layout, same per-request
flow, just more endpoints and a chat command parser on top. Doing them
in one sprint would couple substrate changes with surface changes, and
mistakes in the substrate would force throwaway rework of the surface.

### V2.0 — Memory architecture (foundation, ship first)

**Goal:** True long-conversation continuity with minimal drift. Replace
v1's single-level summary with the three-layer memory architecture
production assistants use.

**Full design:** [compactor/V2_PLAN.md](compactor/V2_PLAN.md)

| Layer | Mechanism | Solves |
|---|---|---|
| **Episodic (RAG)** | Every turn embedded → ChromaDB → top-K retrieval | Exact text recall of relevant past moments |
| **Semantic (facts)** | LLM-extracted bullets, persisted JSON, injected on every request | Drift on stable facts (names, preferences, decisions) |
| **Working (hierarchical summary)** | Tiered summaries: 20 turns → chunk → chapter → conversation theme | Smooth narrative degradation vs catastrophic compression |

**Phased rollout (each phase shippable independently):**
1. **Conversation persistence scaffolding** (~1 day) — establish conv_id and storage
2. **Facts memory** (~2-3 days) — highest value-per-LOC; fixes "model forgot the character's name"
3. **RAG over history** (~3-4 days) — adds ChromaDB + embedding model, ~280 MB image growth
4. **Hierarchical summarization** (~2-3 days) — replaces v1's flat summary

**V2.0 total effort: ~10-12 dev-days.** Open questions (embedding model
choice, fact-extraction frequency, etc.) listed in V2_PLAN.md.

### V2.1 — User control + portability + observability (after V2.0 is stable)

**Goal:** Give the *user* a memory — the ability to inspect, edit, reset,
and back up what the model is remembering. V2.0 gives the model agency;
V2.1 gives the user agency.

**Will not start until V2.0 has shipped and run in production long enough
to be considered stable.** This is the explicit "don't despise the small
things" principle — memory correctness gets battle-tested before we paint
the user-facing layer on top.

**Themes:**
- **Theme 1 — Chat commands:** `/list-facts`, `/forget`, `/remember`, `/why-did-you-say-that` (user agency)
- **Theme 2 — Conversation portability:** export/import bundles, conversation forking, cross-pod backup
- **Theme 3 — Observability:** `/health/full`, inline UI compaction hints, retrieval highlighting, metrics
- **Theme 4 — Quality maintenance:** periodic fact deduplication, conflict resolution, stale-fact archival, memory budgets
- **Theme 5 — Personas as first-class:** persona-aware compaction, persona library, persona inheritance

**Phased rollout:**
5. Chat command surface (Theme 1) — ~1-2 days
6. Export/import + observability endpoints (Themes 2 + 3) — ~2-3 days
7. Quality maintenance background jobs (Theme 4) — ~2-3 days
8. Personas as first-class memory (Theme 5) — ~1-2 days

**V2.1 total effort: ~6-10 dev-days.** Full spec in
[compactor/V2_PLAN.md § V2.1](compactor/V2_PLAN.md#v21--user-control-portability-observability).

### Combined V2.x effort: ~16-22 dev-days across both releases

---

## V3 — Multimodal (vision + voice)

**Goal:** Move from "text-only chatbot" to "AI assistant that can see and
hear." Three independent capabilities that can ship as separate sub-versions.

### V3.1 — Vision (image understanding)

**What it adds:** User can upload images in chat; model sees them; can
describe, OCR, answer questions about them, etc. ("What's in this photo?",
"Read this receipt", "Critique this UI mockup".)

**How it works:**
- Swap `MODEL_REPO` from text-only Magnum to a vision-language model (VLM)
- vLLM supports many VLMs natively: Llama-3.2-Vision, Qwen2-VL, Pixtral,
  InternVL, etc.
- OpenWebUI already has image upload UI built in — it just sends them in
  OpenAI's standard multimodal `content` array format
- The compactor's `_message_text` helper already handles multimodal content
  arrays (we got that right by accident in V1 — only counts text portions
  for token budgeting; images are counted separately by vLLM)

**Effort: ~2-3 days**
- Model swap: env var change (1 hour)
- VRAM budget check: most VLMs are bigger than text-only; may need bigger GPU
- Compactor: verify image content survives compaction (may need to keep
  image-containing turns verbatim, not summarize-and-discard)
- Documentation: which VLMs work, GPU requirements

**Recommended VLMs for the A40 class:**
- `Qwen/Qwen2-VL-7B-Instruct` — solid generalist, ~16GB
- `meta-llama/Llama-3.2-11B-Vision-Instruct` — gated, strong reasoning
- `mistralai/Pixtral-12B-2409` — Mistral's VLM, ~24GB
- For creative writing + vision: no perfect equivalent of Magnum exists yet
  in VLM form; this is a Pareto trade-off between writing quality and
  vision capability

### V3.2 — Speech-to-text (voice input)

**What it adds:** User talks into their microphone; OpenWebUI sends audio
to a `/v1/audio/transcriptions` endpoint (OpenAI-compatible); transcribed
text becomes the prompt.

**How it works:**
- New service: Whisper or distil-whisper running as a small Python server
  exposing OpenAI's audio API contract
- Add a `[program:whisper]` block in supervisord, running on its own port
- Compactor doesn't need changes (it never sees audio — only the transcribed text)
- OpenWebUI config: point its STT URL at the local whisper service

**Model choices:**
- `distil-whisper/distil-large-v3` — ~1.5 GB, CPU-friendly, 6x faster than full Whisper
- `openai/whisper-large-v3-turbo` — best accuracy, needs ~3 GB GPU
- `Systran/faster-whisper-large-v3` — optimized C++ inference, smallest VRAM footprint

**Effort: ~4-5 days**
- New compactor-style wrapper around faster-whisper (~150 lines Python)
- Dockerfile: add whisper venv (~2 GB additional)
- supervisord: new service block
- OpenWebUI config: enable STT, point at local URL
- Test recording → transcription → chat flow end-to-end

**Resource considerations:**
- Whisper-distil runs fine on CPU (slower, ~1x realtime)
- Whisper-large on GPU shares VRAM with the LLM (modest 3 GB extra)
- For an A40 with Magnum 22B FP8: ~6 GB headroom after LLM, fits Whisper-large easily

### V3.3 — Text-to-speech (voice output)

**What it adds:** Model's text responses get spoken aloud by OpenWebUI's
audio player.

**How it works:**
- New service exposing `/v1/audio/speech` (OpenAI TTS API contract)
- OpenWebUI streams generated audio chunks as the model produces text

**Model choices:**
- `hexgrad/Kokoro-82M` — tiny, runs anywhere, surprisingly good quality
- `coqui/XTTS-v2` — voice cloning, multilingual, larger (~2 GB)
- `rhasspy/piper-tts` — extremely fast CPU inference, less natural

**Effort: ~3-4 days**
- Similar pattern to STT: wrapper service + supervisord block + OpenWebUI config
- TTS is generally simpler than STT (no audio decoding, just synthesis)
- Streaming is critical for UX — user shouldn't wait for full response
  before audio starts

### V3 total estimated effort: ~10-12 dev-days

Comparable to V2. Could be done in parallel with V2 since they touch
mostly separate code paths.

---

## Beyond V3 — speculative roadmap

These are directions worth keeping in mind but not committing to until V2
and V3 prove the architecture handles them. **Order does not imply priority.**

### Agentic capabilities (tool use / function calling)
- vLLM 0.11+ supports OpenAI-format tool calling natively
- Need: a tool registry, secure execution sandbox, OpenWebUI tool UI
- Effort: substantial; this is a whole sub-project

### Code execution
- "Run this snippet" inside chat — requires sandbox (Jupyter kernel, e2b, etc.)
- Pairs naturally with agentic tools

### Multi-user with per-user memory
- V2's conv_id hash would need user-identity scoping
- Auth surface expands (OpenWebUI handles UI auth, but compactor needs to
  enforce user isolation in memory store)
- Multi-tenant Network Volume layout

### Fine-tuning pipeline
- Personalize Magnum (or any model) on the user's own writing samples
- Use vLLM's LoRA adapter support to swap personality without reloading
  base weights
- Requires a training pipeline (out of vLLM's scope), but inference becomes
  cheap once adapters exist

### Real-time web search / external RAG
- Combine V2's vector memory with external doc ingestion
- Crawl + chunk + embed pattern, queryable through the compactor

### Browser/desktop integration
- Native client wrapping the OpenWebUI API
- System tray / always-available chat
- Screen-context awareness (model can see what you're looking at)

---

## How to navigate this roadmap

When ready to start a version's work:
- **V1.9.6:** dive in — concrete and short
- **V2:** start with [compactor/V2_PLAN.md](compactor/V2_PLAN.md) Phase 1
- **V3:** pick V3.1 (vision) first — it's the smallest lift and most
  visible capability bump

For optimizations that don't belong to any specific version (image size,
build speed, observability), see the
[Build & runtime optimization roadmap](compactor/V2_PLAN.md#build--runtime-optimization-roadmap)
section in V2_PLAN.md.
