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
   # Production A40 model (the image default since rc8):
   huggingface-cli download coder3101/Cydonia-24B-v4.3-heretic-v4
   # Always-fits A40 fallback:
   # huggingface-cli download anthracite-org/magnum-v4-12b
   ```
4. When the download completes, terminate the CPU pod. Your weights and any
   OpenWebUI state stay on the volume.

You can repeat the download step to pre-cache additional models on the same
volume — vLLM picks whichever one matches `MODEL_REPO` at runtime.

### Step 3: Build and push the image

Pre-built images are published at `angreg/zions-light-ai` on Docker Hub.
**The current deploy target is named in [runpod.env.template](runpod.env.template)'s
header — that file is the single source of truth for the image tag and every
env var.** Pin a version for reproducibility (e.g.
`angreg/zions-light-ai:v3.0-cu12`); `:latest` is only ever promoted to a
*validated* release, so during an rc cycle it lags behind. See the
[image-tags table in the README](README.md#image-tags) for what each tag
contains.

To build and publish your own (CUDA-12 profile — runs on any A40 host; see
the Dockerfile header for the CUDA-13 default profile):
```bash
docker build \
  --build-arg CUDA_BASE_IMAGE=nvidia/cuda:12.6.3-runtime-ubuntu24.04 \
  --build-arg TORCH_CUDA=cu128 \
  --build-arg VLLM_VERSION=0.19.0 \
  -t angreg/zions-light-ai:v3.0-cu12 .
docker push angreg/zions-light-ai:v3.0-cu12
# :latest is promoted ONLY after the on-pod validation gate (see CHANGELOG).
```

### Step 4: Create the Runpod Template

Go to [Runpod Templates](https://www.runpod.io/console/user/templates) → New Template:

- **Template Name:** `zions-light-ai`
- **Container Image:** the tag named in [runpod.env.template](runpod.env.template)
  (currently `angreg/zions-light-ai:v3.0-cu12`)
- **Container Disk:** `60 GB` (room for the image, supervisor logs, scratch)
- **Volume Mount Path:** `/data` (← this is where the Network Volume attaches)
- **Expose HTTP Ports:** `3000, 8080`
- **Docker Command:** (leave empty)
- **Environment Variables:** paste the block from
  [runpod.env.template](runpod.env.template) — it carries all 42 vars with
  the ones that matter marked. As of rc8 the image's built-in default IS the
  production A40 config (Cydonia-24B + runtime fp8), so the vars that
  strictly MUST be set are `WEBUI_SECRET_KEY` and **`WEBUI_DB_LOCAL=false`**
  (see the next section — runpod.env.template does not carry that row, so add
  it by hand); the template pins the model
  vars explicitly anyway so a future default change can never surprise a
  deploy (see GPU sizing for alternatives).

### WEBUI_DB_LOCAL — a hard deploy precondition

**Production runs with `WEBUI_DB_LOCAL=false`, and every deploy must keep it
that way.** It decides where her chat history physically lives. `false` keeps
`webui.db` on the `/data` volume, where it has always been. `true` moves the
live database to the pod's local disk at boot and starts a sync daemon that
copies it back to `/data` every few minutes. That move is the one change in the
v3.1.x line whose rollback is not clean, and it has not been scheduled.

**The trap on every release: a MISSING or EMPTY value means `true`.** A RunPod
template with no `WEBUI_DB_LOCAL` row, or a row that is present but blank
(nothing after the `=`), boots with the database moved.

**How other spellings are read changed in v3.1.9:**

| template value | v3.1.6.1 – v3.1.8 (the pod today) | from v3.1.9 |
|---|---|---|
| missing, or empty | `true` — database moved | `true` — database moved |
| `false` | `false` | `false` |
| `False`, `0`, `no`, `off` (any case, spaces around) | `false` (anything that is not exactly `true`) | `false` |
| `True`, `TRUE`, `1`, `yes`, `on` | **`false`** (only the exact word `true` counted) | **`true` — database moved** |
| anything else (a typo such as `flase`) | `false` | **the pod refuses to boot** |

A refused boot prints a banner in the RunPod **Logs** tab starting
`WEBUI_DB_LOCAL=[<your value>] is not true/1/yes/on or false/0/no/off -
REFUSING TO START.` and the container exits before anything starts, so nothing
is touched. Fix the row and redeploy. The one spelling that means the same on
both sides of an upgrade or a rollback is the exact lowercase word `false`.

**Before every deploy**, in the RunPod template's environment variables, check
there is a row reading exactly:

```
WEBUI_DB_LOCAL=false
```

lowercase, no spaces, no quotes, not blank. The deploy tag's own VERIFY step
may not repeat this; it applies to every release anyway.

**After every boot**, first in the RunPod **Logs** tab (the container's boot
output). From v3.1.9 it names the value it resolved and why:

```
      WEBUI_DB_LOCAL=false (explicitly set (false))
[2b/3] WEBUI_DB_LOCAL=false - webui.db stays on /data/openwebui/webui.db
```

On v3.1.6.1 only the `[2b/3]` line is printed. If v3.1.9 prints
`WEBUI_DB_LOCAL=true (unset/empty -> true …)`, the row is missing or blank.
Then, in the Web Terminal (works on every release):

```bash
tr '\0' '\n' < /proc/1/environ | grep -E '^(WEBUI_DB_LOCAL|WEBUIDB_SYNC_ENABLED|DATABASE_URL)='; supervisorctl status webuidb-sync
```

**Success** — all four of these:

```
WEBUI_DB_LOCAL=false
WEBUIDB_SYNC_ENABLED=false
DATABASE_URL=sqlite:////data/openwebui/webui.db
webuidb-sync                     STOPPED   Not started
```

(the first three may print in a different order).

**If you see `WEBUI_DB_LOCAL=true`, a `/var/lib/openwebui/webui.db` in
`DATABASE_URL`, or `webuidb-sync RUNNING`:** the database was moved to local
disk on this boot.

- **If she has not sent a message since the pod booted:** nothing was written
  to the moved copy. Fix the template row to `WEBUI_DB_LOCAL=false` and
  redeploy. Run the check again after the boot.
- **If she has:** her newest messages are on local disk and reach `/data` only
  when the sync daemon publishes. Ask her to stop chatting, do NOT redeploy
  yet, and run
  `/opt/compactor-venv/bin/python /opt/compactor/webuidb.py --status`; ask for
  help with its output before changing anything. A redeploy at this point can
  lose everything written since the last sync.

### Memory budgets — raised defaults in v3.1.9

*(Two of these six rows were raised again, and a sixth added, in v3.1.9.2
— see below; the anchor name is kept as-is so existing links into this
section do not break.)*

Six environment variables control how much of her own facts/retrieval/
summary memory is stored and injected per turn. The owner raised the first
five by hand on the running pod (2026-09-15, a `supervisorctl` `environment=`
edit on the `compactor` program — lost on every container restart, so it
had to be reapplied after any redeploy). **v3.1.9 baked the same five
values into the image and this template, so that live edit is no longer
needed; v3.1.9.2 (hostile pass #9, P9-1/P9-2) raised the fraction and
summary-block cap again and added the sixth row, for a DIFFERENT reason —
see below:**

| Variable | Code default | v3.1.9 shipped default | v3.1.9.2 shipped default |
|---|---|---|---|
| `COMPACTOR_MAX_FACTS_TOKENS` | 1500 | 3500 | 3500 (unchanged) |
| `COMPACTOR_INJECT_FACTS_TOKENS` | 400 | 600 | 600 (unchanged) |
| `COMPACTOR_MAX_RETRIEVAL_TOKENS` | 1500 | 3500 | 3500 (unchanged) |
| `COMPACTOR_INJECTION_BUDGET_FRACTION` | 0.5 | 0.6 | **0.75** |
| `COMPACTOR_SUMMARY_BLOCK_MAX_TOKENS` | 12000 | 6230 | **12000** |
| `COMPACTOR_STANDIN_BUDGET_FRACTION` | 1.0 (no v3.1.9 equivalent) | — | **1.0** |

**Why the fraction and summary-block rows moved together with the two
raised caps in v3.1.9, not independently:** `inject_budget = effective_limit
× COMPACTOR_INJECTION_BUDGET_FRACTION` is shared by persona + summary +
facts + retrieval. Retrieval is the lowest-priority block and is dropped
WHOLE (not trimmed) by `_bound_injected_blocks` when it does not fit. At the
raised facts/retrieval caps (3500/3500) under the OLD fraction (0.5, about
10,384 tokens of her 20,768-token window), retrieval would have been
silently dropped from every request. At 0.6 (about 12,460 tokens) retrieval
had room, with the summary block pinned at what it measured itself needing
at the time (6,230). **Do not change one of the first five without the
others.**

**Why the fraction and summary-block cap moved AGAIN in v3.1.9.2, and why a
sixth variable was added:** these two rows do double duty. Besides the
facts/retrieval room above, they also set the ceiling for the REUSE
STAND-IN — the array-embedded substitute `compact_if_needed` returns in
place of older turns a stored summary hierarchy already covers, main.py
`_standin_reuse_ceiling`. At the v3.1.9 pair (0.6/6230) that ceiling was a
flat 6,230 tokens, below her hierarchy within a day of the fix that
introduced it (~9,050 tokens, up from ~5.1k when v3.1.9.1 shipped) — reuse
silently declined on every request again, exactly the 2026-09-16 failure
v3.1.9.1 was written to remove, with `/health/full` and the CHANGELOG both
saying it worked. Raising `COMPACTOR_SUMMARY_BLOCK_MAX_TOKENS` alone does
not fix this: the stand-in's OLD formula multiplied the injection budget by
a hard-coded 0.6 before ever reaching the SUMMARY_BLOCK_MAX_TOKENS cap, so
even `0.75`/`20000` only reached a ~9,345-token ceiling — 1,955 tokens
short of the hierarchy's own documented construction capacity (9 L1 scenes
+ 4 L2 chapters + 1 L3 at their max sizes = 11,300 tokens). `COMPACTOR_
STANDIN_BUDGET_FRACTION` (new) is the stand-in's OWN fraction of the
injection budget, separate from the 0.6 the separately-injected summary
block still uses (that block, unlike the stand-in, has to leave room for
facts/retrieval in the SAME inject_budget — the stand-in does not, because
on a reusing turn that separate injection is skipped entirely). At 1.0, the
ceiling is `min(SUMMARY_BLOCK_MAX_TOKENS, inject_budget)` = `min(12000,
15576)` = 12,000, which clears the 11,300-token capacity with measured
margin (a hierarchy built to exactly that capacity renders 11,400-11,500
tokens in practice, header and per-item overhead included). **The
separately-injected block's own share is unaffected by this row**: it still
computes `min(SUMMARY_BLOCK_MAX_TOKENS, int(inject_budget × 0.6))` ≈ 9,345
tokens at the new fraction, comfortably under `inject_budget` (15,576) with
facts (600) and retrieval (3,500) still fitting. `/health/full`'s
`checks.reuse` now reports `attempted`/`declined_budget`/
`declined_recently` and the two numbers behind the most recent decline —
watch that field after any future change to these three rows; a growing
hierarchy can outgrow even 12,000 within one L1 rollup chunk, and this is
how the operator would see it happen instead of reading request logs.

Evidence behind the v3.1.9 numbers (2026-09-15 pod measurement, before that
raise): roughly 16 new facts extracted per exchange with roughly 16 evicted
(the 1500-token store churning), only 6-12 of about 160 stored facts
actually reaching injection, retrieval keeping only 1 of 5 candidate hits,
and the summary block dropping 1 of 2 chapters — against a 20,768-token
input budget that typically used only 15.1-18.7k of it.

**This changes only the environment defaults baked into the image and this
template — it does NOT change the Python code defaults** in `facts.py`,
`retrieval.py`, `main.py` or `summarizer.py`. If you have already overridden
any of these five in your own template, your value is unaffected — these
rows only change what an UNSET row now resolves to.

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

# Image tag + full env set: see runpod.env.template (the source of truth).
runpod pod create \
  --gpu-type "NVIDIA A40" \
  --image "angreg/zions-light-ai:v3.0-cu12" \
  --disk-size 60 \
  --network-volume-id "<your-volume-id>" \
  --env MODEL_REPO=coder3101/Cydonia-24B-v4.3-heretic-v4 \
  --env VLLM_EXTRA_ARGS="--quantization fp8" \
  --env WEBUI_SECRET_KEY="<openssl rand -hex 32>" \
  --env COMPACTOR_BACKUP_INTERVAL_HOURS=6 \
  --ports "3000/http,8080/http"
```

## GPU sizing

| Model | Quant | VRAM | Suggested Runpod GPU |
|---|---|---|---|
| Qwen2.5-1.5B-Instruct | FP16 | ~6 GB | RTX 3090 / 4090 |
| **coder3101/Cydonia-24B-v4.3-heretic-v4** *(production config)* | **FP8 (runtime)** | **~43 GB incl. KV** | **A40** |
| anthracite-org/magnum-v4-12b *(A40 fallback, no quant flag)* | FP16 | ~24 GB | A40 |
| anthracite-org/magnum-v4-22b | FP16 | ~44 GB | A100 (40/80 GB) |
| Qwen2.5-32B-Instruct | FP16 | ~64 GB | A100 80GB |
| Llama-3.3-70B-Instruct | FP16 | ~140 GB | 2× A100 80GB |
| **Vision (V3.1) — Qwen2-VL-7B-Instruct** | FP16 | ~16 GB | A40 |
| Vision — Pixtral-12B-2409 | FP16 | ~24 GB | A40 (tight) / A100 |
| Vision — Llama-3.2-11B-Vision-Instruct *(gated)* | FP16 | ~24 GB | A40 (tight) / A100 |

> **⚠️ Runtime FP8 on an A40 — what's actually validated.** The
> **production-validated A40 config is Cydonia-24B + `--quantization fp8`**
> on the V3.0 image (vLLM 0.19): it boots, repacks, and serves at ~43 GB
> including KV cache — this is what the rc5/rc6 pods ran through the
> 2026-08-13 incident testing. An earlier version of this warning said
> runtime FP8 of a ~22B could OOM during the marlin repack (peak needs the
> FP16 weights resident before freeing them); that was observed on the
> vLLM 0.14-era image and is kept here as a caution: **if a 22–24B + fp8
> boot OOMs on your host, fall back to `anthracite-org/magnum-v4-12b` in
> FP16** (empty `VLLM_EXTRA_ARGS`) — it always fits — or use an
> offline-quantized FP8 checkpoint, which removes the repack peak
> entirely. Watch the first boot's vLLM log either way.

As of **rc8** the image's built-in defaults are the production A40 config
(Cydonia-24B + runtime fp8) — a bare deploy boots correctly out of the box.
On images older than rc8 the built-in default was the 22B (unbootable on an
A40): always override per [runpod.env.template](runpod.env.template).

### The current date and time

From v3.1.9 the model is told the real date and time on every message, in
**her browser's time zone**.

**1. Prerequisite: her chat must already log `source=header`.** Adding a line
to the system prompt changes the system prompt. A conversation whose id is
still derived by hash (`source=hash`) gets a NEW id when the system prompt
changes, and its memory is left behind under the old one. Check first:
`grep -aE "conv_id=[^ ]+ source=[^ ]+ msgs=" /data/logs/compactor.log | grep -v __selftest | tail -3`.
(The boot selftest always logs `source=header`; ignore it, which is what the
`grep -v` does.) If her chat says `source=hash`, either do
[RUNBOOK_MEMORY_IDENTITY.md](RUNBOOK_MEMORY_IDENTITY.md) first, or skip step 2
entirely and use step 3 on its own: the model is still told the right time, in
the zone you set, without touching the system prompt.

**2. Add one line to her model's system prompt.** OpenWebUI → Admin Panel →
Settings → Models → her model → System Prompt. Add this line on its own, at
the start of a line, exactly as written:

```
User timezone: {{CURRENT_TIMEZONE}}
```

OpenWebUI replaces `{{CURRENT_TIMEZONE}}` with the time zone her browser
reports (for example `America/Phoenix`) on every message. It stays the same
unless her device's time zone changes, so it does not slow the model down.
Do **not** add `{{CURRENT_DATETIME}}` there: it changes every minute and would
make the model reprocess her whole conversation on every message. A model's
system prompt is shared by every user of that model; that is fine here,
because there is one user.

**3. Optional fallback.** For requests that do not come from her browser (a
direct API call, a client that does not fill in `{{CURRENT_TIMEZONE}}`), set
`COMPACTOR_TIMEZONE=America/Phoenix` in the RunPod template. The order is: her
browser's zone, then `COMPACTOR_TIMEZONE`, then UTC. `TZ` is not used for this.

**What the model sees.** One line at the start of her newest message, in the
request sent to the model only:

```
[Current date and time: Monday, September 14, 2026, 9:41 AM MST (UTC-07:00)]
```

She never sees it, OpenWebUI never stores it, and it never enters memory
(facts, summaries or the episodic index). `/remember`-style commands are
unaffected. The `User timezone:` line itself stays in the system prompt the
model reads.

**Title, tag, follow-up and query-generation calls ("task traffic").** Under
header identity (`{{CHAT_ID}}{{TASK}}`, see RUNBOOK_MEMORY_IDENTITY.md),
these are never given the current-time line, from their first call — the
compactor recognizes them by the `{{TASK}}` suffix. Under today's **hash**
identity (no identity header configured), there is no such marker, so the
same calls are recognized only once a conversation has genuinely reached
`COMPACTOR_TASK_TRAFFIC_MIN_POSITION` (default 4) turns of real history —
the first one or two task calls on a brand-new template may be dated
(harmless: OpenWebUI discards what it does not use the line for). Either
way, a real first message from her that happens to hash-collide with an
older, already-deep conversation's opener is still dated correctly, but is
**not memorized** under hash identity — a known limitation of deriving the
conversation id from a content hash, not of the time feature. Configuring
the `{{CHAT_ID}}` header removes both limitations.

**Check it took.** After her next message, `curl -s localhost:8080/health/full`
→ `config.time_injection`: `current_line` is the line the model is being
shown. **What `last_source` should read depends on which route you are on —
check the one that matches your setup, not always `browser`:**
- On `source=header` with the system-prompt line added (step 2): `last_source`
  must be `browser`, `last_timezone` her zone. `utc` with a
  `last_browser_error` means the system-prompt line was not filled in or
  names no real zone (the compactor log says so once).
- On `source=hash` with `COMPACTOR_TIMEZONE` set (step 3, today's setup):
  `last_source` is `env`, **not** `browser` — that is correct for this route,
  not a fault. Confirm the zone itself in `last_timezone`.
`fallback_error` (either route) names a misspelled `COMPACTOR_TIMEZONE`; the
pod still boots, on UTC, with one `TIME ZONE NOT APPLIED` ERROR in
`compactor.log` — and, as of this writing, `/health/full` `status` stays
`ok` with no `status_reasons` entry for it, so check `fallback_error`
explicitly rather than trusting `status` alone. A pre-merge zone spelling
(`US/Arizona`, `Asia/Calcutta`, ...) resolves correctly from v3.1.9
(`tzdata-legacy` is now in the image); prefer the zone's current canonical
name regardless.

**Turn it off.** `COMPACTOR_TIME_INJECTION=false` (also `0`, `no`, `off`) and
redeploy.

### Sampling parameters

OpenWebUI's **Advanced Params** (per-model or per-chat) only map a fixed set
of names onto the OpenAI-shaped request it sends: `temperature`, `top_p`,
`min_p`, `max_tokens`, `frequency_penalty`, `presence_penalty`,
`reasoning_effort`, `seed`, `stop`, `logit_bias`, `response_format`. Anything
else — including any name from an Ollama-style setup — has to go under
**Custom Parameters** instead, where OpenWebUI passes it through to the
request body verbatim, under whatever key you typed.

vLLM's name for what Ollama calls `repeat_penalty` is **`repetition_penalty`**.
Before v3.1.9.2, setting `repeat_penalty` as a Custom Parameter did nothing —
vLLM does not recognise the key, silently ignores it, and
`repetition_penalty` stayed at its default of 1.0. **From v3.1.9.2 on**, the
compactor translates `repeat_penalty` to `repetition_penalty` before
forwarding (and drops `repeat_last_n`, which has no vLLM equivalent at all).
It is still better to set `repetition_penalty` directly under Custom
Parameters, by its real name, so nothing depends on the translation.

**Recommended starting values for this model** (Cydonia-24B):

| Setting | Where in OpenWebUI | Value |
|---|---|---|
| `repetition_penalty` | Custom Parameters (name typed exactly) | `1.05` |
| Frequency Penalty | Advanced Params | `0.3` |
| Max Tokens | Advanced Params | `12000` |

Leaving Max Tokens unset means the request carries no ceiling at all: a
runaway reply continues until it fills the context window or someone presses
Stop. **7000 tokens is NOT "roughly 28,000 characters" on this model** — that
assumes 4 characters/token, and this model's own measured pairs (see
`count_tokens_exact`'s docstring in `compactor/main.py`, production data,
2026-08-28) run 2.0-2.4 characters/token on assistant replies, because this
model's heavy use of box-drawing and other decoration characters prices high.
At that rate 7000 tokens is roughly 14,000-17,000 characters — below her
normal p90 reply length (measured ~17,000 characters on her main chat,
hostile pass #7), so a real, non-runaway reply would routinely hit the
ceiling and come back cut mid-sentence, and get stored to memory trimmed the
same way (`stream truncated at the generation ceiling`). **12000 tokens**
(roughly 24,000-29,000 characters at the same measured rate) covers ordinary
long replies with headroom and does not change memory's budgets: the
compactor already reserves the larger of `COMPACTOR_GENERATION_RESERVE`
(12000) and Max Tokens, so 12000 is the value it already plans around. To
check the real rate on your own pod rather than trust this range, POST a
sample of her actual replies to vLLM's `/tokenize` endpoint and compare the
returned token count against the character count directly, rather than
estimating.

vLLM 0.19 applies `repetition_penalty` to **prompt tokens as well as output**
(verified by reading `model_executor/layers/utils.py::apply_penalties` and
the V2 GPU sampler kernel in the served image) — it is not output-only the
way Ollama's `repeat_penalty` behaves. Against a prompt that can run to
~20,000 tokens of her own conversation and injected memory, a HIGH
`repetition_penalty` discourages the model from using words that are already
sitting in that history, not only words it has already said in this reply —
which can flatten normal vocabulary, not just break loops. The values above
lean on Frequency Penalty (output-only) to do most of the anti-loop work and
keep `repetition_penalty` closer to its default; if loops return, raise
Frequency Penalty before raising `repetition_penalty` further.

**Confirming from the log that loops are being caught.** A repetition-loop
reply produces a WARNING when it is detected. The wording split is **FINISHED
vs CUT (Stop or the generation ceiling), not streamed vs non-streamed** —
both paths run through the same `decide_memory_tail`, and either shape can
happen on a streamed or non-streamed request: `reply looks like a repetition
loop (...)` for a reply the model finished on its own, `... look like a
repetition loop (...)` for one that was cut. One grep catches both:

```bash
grep -a 'like a repetition loop' /data/logs/compactor.log | tail
```

and, once that reply is later replayed back as history, an INFO line at the
point it is kept out of what is forwarded:

```
conv=<id>: touched <N> degenerate assistant turn(s) in the forwarded window (whole=<K> cut=<N-K>)
```

**`whole` vs `cut` (from v3.1.9.2 hostile pass #8, P8-8):** this line used to
read `replaced <N> ... with a placeholder` unconditionally. That is only true
for the `whole` count — most flagged replies keep a clean head (and, for a
mid-reply span, a clean tail too) and only lose the flagged span itself; on
the 2026-09-16 backup that was 10 of 66 touched replies replaced whole, not
all of them. Read `whole` as "the model lost the whole answer for that turn"
and `cut` as "one span was removed from an otherwise-intact reply".

**These counts do not have to match the WARNING count above, and a mismatch
is not a bug.** A CUT loop reply whose trimmed sentence head reads clean is
stored TRIMMED in memory (`stored_trimmed`, no loop WARNING at all —
memory's own judgement only sees the kept head) even though the detector
flagged the FULL text, and that full text is still touched in the forwarded
window on every later request (counted in the `touched <N>` INFO line). So it
is normal to see a `touched` count with no matching `like a repetition loop`
WARNING for the same turn; do not read that as the detector missing
something. Neither line names the reply's own text. If `repeat_penalty` was
translated because `repetition_penalty` was absent, that is a separate INFO
line at request time: `conv=<id>: translated Ollama repeat_penalty=... to
vLLM repetition_penalty=...` (logged once per conversation, not on every
turn).

**`max_tokens` and other numeric sampling fields (v3.1.9.2, hostile pass
#8, P8-6).** A request body whose JSON carries a numeral that overflows to
`inf` (for example `"max_tokens": 1e999`, in any numeric field, not only
the sampling penalties) is now rejected at parse time with an HTTP 400,
the same way a bare `NaN`/`Infinity` constant already was — it used to 500
from inside the proxy instead, after compaction and memory injection had
already run. An unparseable `max_tokens` (a string, a list, ...) is
dropped from the forwarded body with a WARNING rather than left in place
unexamined; a VALID `max_tokens` is never rewritten, only capped against
the model's context window as before.

**Preserved images and the recent-turn floor (v3.1.9.1 hostile pass #7 F1,
corrected in v3.1.9.2 hostile pass #8 P8-1).** An uploaded image always
arrives as a part of a USER turn (see "Vision" below) — never an assistant
one. The hard-budget guard's recent-turn floor accounts for that: an old,
preserved image sitting in front of the real recent window no longer
counts as part of "recent" merely because both it and the turn after it
are user turns; it is recognised by the role-alternation break instead, so
it is available to be shed ahead of injected memory (facts/retrieval) the
same as any other old turn, whether it happens to be a USER-role image or
(the pre-P8-1 test shape) an orphaned ASSISTANT turn.

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

### Speech-to-text (V3.2) — voice input

A bundled **Whisper** service (faster-whisper) exposes the OpenAI audio API on
port `9000`, and OpenWebUI's STT engine is pre-wired to it — so the microphone
button in the UI works out of the box, with nothing leaving the pod. The
compactor isn't involved (audio → text only; the transcript then flows through
memory like any typed message).

- **Runs on CPU by default** (`WHISPER_DEVICE=cpu`, int8) so it never competes
  with vLLM for VRAM — the A40 has almost none to spare at
  `GPU_MEMORY_UTILIZATION=0.90`. The default `base` model is prebaked into the
  image.
- **Swap the model** with `WHISPER_MODEL` (`base` → `small` / `medium` /
  `large-v3`): bigger = more accurate + slower. For `large-v3` you'll usually
  want `WHISPER_DEVICE=cuda` **and** real VRAM headroom (a bigger card, or a
  lowered `GPU_MEMORY_UTILIZATION`), plus
  `WHISPER_DOWNLOAD_ROOT=/data/whisper-models` so a non-default model persists
  across pod recreation instead of re-downloading on each cold start.
- **Turn it off** per-pod with `STT_ENABLED=false` (and/or `AUDIO_STT_ENGINE=""`
  to hide voice in the UI while leaving the service running).
- Port `9000` does **not** need external exposure for the UI mic to work
  (OpenWebUI reaches it over localhost). Expose it only to hit the audio API
  directly — e.g. to run the `tests/eval/` WER quality eval against the pod.

The boot self-test confirms the service actually transcribes (not just that the
port is open); transcription *accuracy* is measured by the `tests/eval/` WER
harness with your own audio clips.

### Text-to-speech (V3.3) — voice output

A bundled **Piper** service (onnxruntime, CPU) exposes the OpenAI audio API on
port `9001`, and OpenWebUI's TTS engine is pre-wired to it — so the "read aloud"
control speaks replies out of the box, with nothing leaving the pod. The
compactor isn't involved (text → audio only).

- **CPU + torch-free** — Piper is fast on CPU, so like the STT service it never
  competes with vLLM for VRAM and adds almost nothing to the image. The default
  voice **`en_US-lessac-medium`** is prebaked.
- **Swap the voice** with `TTS_VOICE` (any Piper voice from
  [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices)); set
  `TTS_VOICE_DIR=/data/tts-voices` so a non-default voice persists across pod
  recreation.
- **Output format:** the service produces **WAV** natively, and OpenWebUI
  converts it to MP3 before playing it. That conversion needs the `ffmpeg`
  and `ffprobe` binaries. **Before v3.1.9 the image did not include them, so
  the read-aloud button never worked**: OpenWebUI answered HTTP 200 with an
  error body instead of audio. From v3.1.9 `ffmpeg` is installed, and the same
  binaries let OpenWebUI transcribe recordings over 20 MB.
- **Turn it off** per-pod with `TTS_ENABLED=false` (and/or `AUDIO_TTS_ENGINE=""`
  to hide the read-aloud control while leaving the service running).
- Port `9001` does **not** need external exposure for the UI to work (OpenWebUI
  reaches it over localhost).

The boot self-test confirms the service actually synthesizes audio, not just
that the port is open.

### Audio and video FILES attached to a chat

Attaching an audio file (WAV, MP3, M4A, OGG, WEBM) to a chat works like the
microphone: OpenWebUI transcribes it and the model receives the transcript.
**Video is different.** Out of the box OpenWebUI transcribes only `.webm`
video. An `.mp4` or an iPhone `.mov` is stored with no text at all, so the
model receives nothing from it, and nothing tells the user.

To have the soundtrack of `.mp4` and `.mov` files transcribed (v3.1.9 or
later, which has `ffmpeg`), change the setting **in OpenWebUI's Admin Panel,
under the Audio settings, in the speech-to-text supported content types
field**, to exactly:

```text
audio/*,video/webm,video/mp4,video/quicktime
```

Two traps, both verified on a test copy of this stack:
- **Setting `AUDIO_STT_SUPPORTED_CONTENT_TYPES` as a RunPod template variable
  does nothing on an existing pod.** OpenWebUI reads that variable only on its
  very first boot and keeps the value in its database after that, so the
  Admin Panel is the only place the change takes effect. It applies
  immediately, with no restart.
- **Keep `audio/*` in the list.** An empty field silently means
  `audio/*,video/webm`. Typing only the video types removes that default, and
  every ordinary voice recording then fails with "It seems like the file
  format is not supported".

Only the soundtrack is transcribed; the pictures are not described (that is a
V4 roadmap item, see [V4_ROADMAP.md](V4_ROADMAP.md) §1.3). A long video takes
a while: a 26 MB MP4 took about 48 s just to extract its audio, before
transcription.

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
| `COMPACTOR_MAX_FACTS_TOKENS` | code default `1500`, **image/template default `3500`** (v3.1.9) | Token budget for the facts STORE (LRU-evicted past this). See [Memory budgets](#memory-budgets--raised-defaults-in-v319). |
| `COMPACTOR_INJECT_FACTS_TOKENS` | code default `400`, **image/template default `600`** (v3.1.9) | Token budget for facts injected into ONE turn's prompt, ranked by relevance. See [Memory budgets](#memory-budgets--raised-defaults-in-v319). |
| `COMPACTOR_RAG_ENABLED` | `true` | Episodic RAG over past turns (ChromaDB). Set `false` to disable. |
| `COMPACTOR_RAG_TOP_K` | `5` | How many past exchanges to retrieve per turn |
| `COMPACTOR_MAX_RETRIEVAL_TOKENS` | code default `1500`, **image/template default `3500`** (v3.1.9) | Token budget for the whole retrieved-exchange block. See [Memory budgets](#memory-budgets--raised-defaults-in-v319). |
| `COMPACTOR_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model (prebaked ONNX in the image) |
| `COMPACTOR_HIERARCHICAL_SUMMARY` | `true` | L1→L2→L3 rolling summaries. Set `false` to disable. |
| `COMPACTOR_INJECTION_BUDGET_FRACTION` | code default `0.5`, **image/template default `0.75`** (v3.1.9.2; was `0.6` in v3.1.9) | Fraction of the effective input limit shared by persona + summary + facts + retrieval, AND the input to the reuse stand-in's own ceiling. Must move together with the facts/retrieval caps and the summary-block cap above — see [Memory budgets](#memory-budgets--raised-defaults-in-v319). |
| `COMPACTOR_SUMMARY_BLOCK_MAX_TOKENS` | code default `12000`, **image/template default `12000`** (v3.1.9.2; was lowered to `6230` in v3.1.9) | Outer cap on the rendered summary block AND the reuse stand-in. See [Memory budgets](#memory-budgets--raised-defaults-in-v319). |
| `COMPACTOR_STANDIN_BUDGET_FRACTION` | code default `1.0` | **New in v3.1.9.2.** Fraction of the injection budget the reuse stand-in's own ceiling may claim — separate from the 60% the separately-injected summary block still uses. See [Memory budgets](#memory-budgets--raised-defaults-in-v319). |
| `COMPACTOR_TAIL_ROLLUP_MAX_CALLS` | `4` | Per-turn budget for the background tail (and the one-shot backfill rollup) catching up a summary hierarchy that has fallen far behind (a vLLM outage, days of rollup failures). Bounds where a rollup unit is allowed to **start**, not a hard per-turn ceiling: a unit that starts always finishes, so one turn can spend up to `(budget − 1)` plus that unit's own real cost — normally a few calls, but measured at 6-16 calls for one unit when `/tokenize` is down. Converges over successive turns either way; see CHANGELOG.md "Summary hierarchy catch-up" (v3.1.9). |
| `COMPACTOR_DEDUP_SIMILARITY` | `0.75` | Cosine threshold for fact-dedup candidate clustering |
| `COMPACTOR_DEDUP_MAX_LLM_CALLS` | `10` | Cap on LLM merge calls per dedup pass |
| `COMPACTOR_ARCHIVE_DEFAULT_DAYS` | `90` | Default staleness cutoff for fact archival |
| `COMPACTOR_PERSONA_ENABLED` | `true` | Persona detection + injection. Set `false` for V2.0 behavior. |
| `COMPACTOR_PERSONA_AUTO_DETECT_MIN_CHARS` | `200` | Min first-system-message length to auto-capture as a persona |
| `COMPACTOR_STORAGE_ROOT` | `/data/openwebui/compactor` | Where memory state is written |

**Ops (observability + safety):**

| Variable | Default | Description |
|---|---|---|
| `COMPACTOR_SELFTEST_ON_BOOT` | `true` | Run the live-stack self-test after boot, logging to `/data/logs/selftest.log` |
| `COMPACTOR_ADMIN_BIND` | `127.0.0.1` | Admin-endpoint bind address. **Keep localhost** unless you have auth/firewall in front — admin endpoints are unauthenticated. |
| `COMPACTOR_BACKUP_ENABLED` | `true` | Run the periodic verified-backup daemon (V2.3) |
| `COMPACTOR_MIN_FREE_MB_WRITES` | `200` | Pause new-memory writes (keep serving) below this free space on `/data` (V2.3) |
| `COMPACTOR_LOG_FORMAT` | `text` | `text` (human) or `json` (one object/line for aggregation) |
| `COMPACTOR_ALERT_WEBHOOK` | *(unset)* | If set, self-test + backup POST a failure alert here (Slack/Discord/generic) |
| `WEBUI_DB_LOCAL` | `true` (also when set but EMPTY) | **Production: exactly `false`, and it is a hard precondition** — see [WEBUI_DB_LOCAL](#webui_db_local--a-hard-deploy-precondition). `false` keeps `webui.db` on `/data`; `true` moves it to local disk with a sync daemon. From v3.1.9 an unrecognised value refuses to boot, and `1`/`yes`/`True` mean true. |
| `LOG_DIR` | `/data/logs` | Where every service log is written (on the volume, so logs survive a redeploy) |

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
memory through OpenWebUI, add a connection header in OpenWebUI (Admin Panel →
Settings → Connections → the `http://localhost:8080/v1` connection → Headers):
`{"X-Conversation-Id": "{{CHAT_ID}}{{TASK}}"}`. Follow
[RUNBOOK_MEMORY_IDENTITY.md](RUNBOOK_MEMORY_IDENTITY.md) to do it on a pod that
already has conversations: the order matters. The bundled
`pipelines/conversation_id_header.py` filter does **not** deliver the chat ID
on OpenWebUI 0.11.0 (OpenWebUI discards the metadata it writes); it is only a
history cap now. Direct API callers set `X-Conversation-Id` themselves
(otherwise the compactor falls back to a content hash). See
[USER_GUIDE.md](USER_GUIDE.md).

## Upgrading an existing pod from v3.1.8 to v3.1.9

This is the exact sequence for the production pod: today it runs v3.1.8
with `WEBUI_DB_LOCAL=false`, and every chat currently logs `source=hash`
(no `X-Conversation-Id` header configured — see
[RUNBOOK_MEMORY_IDENTITY.md](RUNBOOK_MEMORY_IDENTITY.md) if that changes
before you upgrade). **v3.1.9 does NOT move the database to local disk —
that is v3.1.9.1, a separate later release. Keep `WEBUI_DB_LOCAL=false`
through this whole procedure**, on both images.

### 1. Pre-checks (on the running v3.1.8 pod)

```bash
curl -s http://localhost:8080/health/full | python3 -m json.tool
```
`status` should read `ok`, or `degraded` only for a reason you already
recognize (see [OPERATIONS.md → Reading /health/full](OPERATIONS.md#reading-healthfull--do-not-trust-status-alone)).
Do not upgrade on top of an unexplained `degraded`/`down` — resolve it
first.

```bash
tr '\0' '\n' < /proc/1/environ | grep -E '^WEBUI_DB_LOCAL='
```
Confirm it reads `WEBUI_DB_LOCAL=false` before you touch anything.

### 2. Stop the writers, then back up and verify (on v3.1.8, before redeploying)

Do this while she is not chatting. Stop OpenWebUI, the compactor and the
backup daemon first, so the backup is taken with no save in flight and the
redeploy cannot kill a write to `webui.db` half-way (on the network volume
that is how a hot rollback journal is made — see RUNBOOK_DB_JOURNAL.md).

```bash
supervisorctl stop openwebui compactor backup
```
```bash
supervisorctl status
```
All three must read `STOPPED`.

```bash
/opt/compactor-venv/bin/python /opt/compactor/backup.py --once; echo "EXIT=$?"
```
Expect `[OK] <archive-name>  ...` and `EXIT=0`.

```bash
/opt/compactor-venv/bin/python /opt/compactor/backup.py --list
```
Confirm the archive you just made is at the top.

```bash
/opt/compactor-venv/bin/python /opt/compactor/backup.py --verify /data/backups/<archive-you-just-made>.tar.gz
```
Expect `[OK] <detail>`. **Do not proceed past a `[FAIL]` on either command** —
fix the backup first; this is the copy you would roll back to.

If the volume is tight, prune old backups by hand now — see
[OPERATIONS.md → Nightly "memory shrank" alert](OPERATIONS.md#nightly-memory-shrank-alert--noise-on-v3161-to-v318-a-real-signal-from-v319-except-one-item)
for the manual-prune command; on v3.1.8 the nightly "memory shrank" alert
holds the automatic prune, so old archives accumulate.

If a step here fails and you are not redeploying after all, restart the
services you stopped: `supervisorctl start openwebui` (wait for `200` from
`curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:3000/api/config`),
then `supervisorctl start compactor backup`.

### 3. Template changes (RunPod template, before redeploying)

- **Container Image:** update to the v3.1.9 tag.
- **`WEBUI_DB_LOCAL=false`** — confirm the row is present and spelled exactly
  that way (a missing or blank row means `true` on v3.1.7 and later). See
  [WEBUI_DB_LOCAL — a hard deploy precondition](#webui_db_local--a-hard-deploy-precondition).
- **`COMPACTOR_TIMEZONE=<her IANA zone>`** (for example `America/Phoenix`) —
  new in v3.1.9. While her chat is on `source=hash` (today), this is how the
  model is told the real date/time; do NOT edit her model's system prompt to
  add the `User timezone:` line yet — that forks a hash-identity chat's
  memory. See [The current date and time](#the-current-date-and-time).
- **The six memory-budget rows** — `COMPACTOR_MAX_FACTS_TOKENS=3500`,
  `COMPACTOR_INJECT_FACTS_TOKENS=600`, `COMPACTOR_MAX_RETRIEVAL_TOKENS=3500`,
  `COMPACTOR_INJECTION_BUDGET_FRACTION=0.75`,
  `COMPACTOR_SUMMARY_BLOCK_MAX_TOKENS=12000`,
  `COMPACTOR_STANDIN_BUDGET_FRACTION=1.0` (the last three raised/added in
  v3.1.9.2 — see [Memory budgets](#memory-budgets--raised-defaults-in-v319)
  for why). These are now the image's own defaults, so adding the rows is
  optional and self-documenting, not required — but if your template
  already has a hand-added `COMPACTOR_MAX_FACTS_TOKENS` or similar row at a
  DIFFERENT value (the pre-v3.1.9 live-pod workaround, or the v3.1.9
  `0.6`/`6230` pair), either remove it or update it to match, or it will
  silently override the new image default — **this specific shape (a
  leftover `0.6`/`6230` override) is exactly what put the reuse feature
  back to declining silently in hostile pass #9**, so check for it if
  upgrading a pod that has ever had these rows added by hand.

### 4. Deploy

Terminate the v3.1.8 pod (or use RunPod's redeploy-on-new-image flow if your
plan supports it) and start a new pod from the updated template, with the
SAME `zions-data` Network Volume attached. Nothing on `/data` is touched by
the image change alone.

### 5. Post-checks (on the new v3.1.9 pod)

```bash
tr '\0' '\n' < /proc/1/environ | grep -E '^(WEBUI_DB_LOCAL|WEBUIDB_SYNC_ENABLED|DATABASE_URL)='
supervisorctl status webuidb-sync
```
Expect the four-line "Success" block in
[WEBUI_DB_LOCAL](#webui_db_local--a-hard-deploy-precondition) above —
`WEBUI_DB_LOCAL=false`, sync disabled/stopped.

```bash
cat /data/logs/selftest.log
```
Expect it to end `=== N/N passed, 0 failed ===`.

```bash
curl -s http://localhost:8080/health/full | python3 -m json.tool
```
Expect `status: ok`, or `degraded` only for `memory tail skipping` or a
hierarchy-lag reason with `verdict: unknown` (expected right after a
restart — see [Summary hierarchy catch-up](CHANGELOG.md#summary-hierarchy-catch-up)).

Send her one real message, then:
- The compactor log should show `injected memory [...]` for her conversation,
  and `/health/full`'s `memory_tail.stored` (or `stored_trimmed`) should have
  gone up.
- Her **first** message may still log `compaction skipped` / `hard budget
  enforced` once (adopting the v3.1.7/v3.1.8-era summary record) — expected.
  From her **second** message on, requests should reuse the stored hierarchy
  with no refusals. If refusals continue past the second message, stop and
  investigate before she sends more.
- `config.time_injection.last_timezone` in `/health/full` matches the zone
  you set, and `last_source` reads `env` (she is still on `source=hash`
  today — see [The current date and time](#the-current-date-and-time) for
  what each route should show).

### 6. Rollback to v3.1.8

Full procedure and the reason for the cap step:
[CHANGELOG.md → Rolling back to an older image](CHANGELOG.md#rolling-back-to-an-older-image).
In short: set the History cap's `max_turns` to `0` BEFORE redeploying the
v3.1.8 image (rolling back with the cap on leaves a permanent hole in her
summary hierarchy), keep `WEBUI_DB_LOCAL=false` spelled exactly that way on
both images, redeploy the v3.1.8 tag against the SAME Network Volume (no
restore needed — v3.1.9 did not move or reformat anything on `/data`), then
run the post-checks above against the v3.1.8 pod. Restore from the backup
taken in step 2 only if you have independent evidence data was actually
lost — a rollback alone does not require it.

## Troubleshooting

### Is the deploy healthy?
```bash
# Deep health probe (200 = ok/degraded, 503 = storage down)
curl -s http://localhost:8080/health/full | python3 -m json.tool
# "ok" does not cover stopped backups: see OPERATIONS.md "Reading /health/full"

# Post-boot self-test result — runs automatically on every start
cat /data/logs/selftest.log
# Expect: "=== N/N passed, 0 failed ==="

# On-demand self-test (real chat round-trip + facts read/write)
curl -s http://localhost:8080/admin/selftest | python3 -m json.tool
```

### Check Logs
```bash
# Via Runpod web terminal
tail -100 /data/logs/vllm.log         # inference engine
tail -100 /data/logs/compactor.log    # memory + compaction events
tail -100 /data/logs/openwebui.log    # frontend
cat /data/logs/selftest.log           # boot self-test
cat /data/logs/boot.log               # one line per container boot
```

### Watch memory in real time
```bash
tail -f /data/logs/compactor.log
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
