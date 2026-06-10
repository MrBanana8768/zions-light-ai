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

## V2.2 — Testing & Observability  ✅ Complete

**Goal:** Make "is this deploy actually working?" answerable automatically
and repeatably, and codify a testing standard every future feature must
follow. Pairs naturally with V2.1's observability theme.

**Status: ✅ Complete.** All deliverables shipped *inside the V2.1 image
line* — there is no separate V2.2 image. Part A (the TESTING.md standard,
this ROADMAP entry, the `/app/data` Dockerfile cleanup) landed before
V2.0 Phase 3. Part B (the boot self-test harness) shipped as V2.1 Phase 6
Step 2, with the two-phase vLLM readiness probe added in Phase 6.1.
`/health/full` shipped as Phase 6 Step 1; the Tier-3 integration suite
began in V2.0 Phase 4 and grew across every V2.1 phase. See the
[1.9.6 → 2.2] CHANGELOG entries.

**Prerequisite (met):** shipped after V2.0 (memory) was complete through
Phase 4 — we didn't interrupt the in-flight memory work. The *standard*
(TESTING.md) was in force from Part A and governed the code written in
V2.0 Phases 3/4 and all of V2.1.

**Full standard:** [TESTING.md](TESTING.md). Three test tiers:

| Tier | Scope | Where | GPU |
|---|---|---|---|
| 1 — Unit/logic | pure functions, I/O, parsers, state machines | `compactor/test_*.py` (CI on every PR) | No |
| 2 — Boot self-test | live-stack health: round-trip + facts I/O, runs post-boot | `compactor/selftest.py` | Yes (pod) |
| 3 — Integration | end-to-end scenarios (e.g. facts persistence over 300 turns) | `tests/integration/` | Yes |

**What V2.2 built (all shipped):**
- ✅ `compactor/selftest.py` — boot-time validation battery (vLLM model
  loaded, compactor health, real 1-token chat round-trip, facts
  write/read/delete against a `__selftest__` sentinel conv, admin
  localhost gating). Auto-runs post-boot as a non-blocking one-shot
  supervisord program (`COMPACTOR_SELFTEST_ON_BOOT=true`), logs to
  `/var/log/supervisor/selftest.log`. Also on-demand. Two-phase vLLM
  readiness probe (Phase 6.1) waits for actual completion capability,
  not just an open port.
- ✅ `GET /admin/selftest` — on-demand self-test, JSON report, localhost-only.
- ✅ `GET /health/full` — deeper than `/health`: probes vLLM reachability +
  storage writability + memory-store stats. Is now the Docker `HEALTHCHECK`
  target so the pod reports unhealthy when vLLM is down (the old
  `curl :3000` check passed even when vLLM was FATAL). Satisfies the V2.1
  observability `/health/full` item.
- ✅ `tests/integration/` — Tier-3 suite, 58 fast + 2 slow tests, run
  against a live pod via `ZIONS_TEST_BASE_URL`. On-pod mode documented in
  `tests/integration/README.md`.
- ✅ `compactor/test_selftest.py` — Tier-1 coverage for the harness itself
  (mocked HTTP), including the two-phase readiness probe.

**V2.2 effort: ~3-4 dev-days (actual: folded into V2.1 Phase 6 + 6.1).**

---

## V2.3 — Resilience & Stability  ✅ Code-complete

**Goal:** Make the system *survive failure gracefully* and *protect
irreplaceable data*, so it can run unattended in a production-shaped
environment without the owner babysitting logs. This is the
"quality-and-stability-over-speed" release: every item is something whose
absence only hurts when it's 2am and something has already gone wrong.

**Philosophy (explicit, by owner's direction):** ship slowly and
deliberately. Each item below is "done" only when it has been *deliberately
failure-tested* — not when the happy path works. Better to spend weeks
locking one item down than to ship five half-proven ones. The whole
point of V2.3 is that the things it adds must themselves be trustworthy,
because they're the safety net the rest of the system leans on.

### Theme 1 — Data durability (the only *unrecoverable* failure class)

The `/data` volume holds two things that cannot be regenerated if lost:
OpenWebUI's `webui.db` (chat history) and `compactor/` (facts JSON +
ChromaDB vectors). Everything else (models, torch.compile cache) is
re-downloadable. Today a corrupted volume = total memory loss, no recovery.

- ✅ **Scheduled backups** of `webui.db` + `compactor/` (tar + timestamp,
  retain N). Shipped as `compactor/backup.py` + the `[program:backup]`
  supervised daemon (pairs with V2.2's selftest process model).
- ✅ **Backup verification** — a backup that can't be restored is not a
  backup. `run_once` restores its own archive into a scratch dir and asserts
  SQLite `PRAGMA integrity_check` + every memory JSON parses *before*
  publishing; an unverifiable archive is discarded and the cycle reports
  FAILURE.
- ✅ **Documented restore runbook** — [OPERATIONS.md](OPERATIONS.md) has the
  exact recover-from-wiped-volume commands.
- ✅ **Atomic-write audit** — confirmed every durable writer (facts,
  facts-archive, **Phase 4 summaries**, personas, backfill) routes through
  `memory.atomic_write_json`. No gap.
- ⏳ **Off-volume disaster recovery (future work — migration needed).** The
  shipped backups are **local to the same volume**: they protect against
  corruption / accidental delete / torn writes, but NOT total volume loss.
  True DR requires an off-volume target (S3-compatible object store) — a
  provider choice, credentials, and a real uploader (boto3 or rclone). The
  `upload_hook` seam + `COMPACTOR_BACKUP_REMOTE` env are already wired as
  the integration point; turning them on is a follow-up that will involve a
  **migration** (moving/duplicating existing local archives off-volume and
  validating restore-from-remote). Until then, the operational mitigation
  (documented in OPERATIONS.md) is to periodically copy the newest
  `/data/backups/` archive off the pod.

### Theme 2 — Graceful degradation under partial failure

The system is already designed so memory failures never break chat
(retrieval/facts degrade to no-ops). V2.3 makes that a *verified guarantee*
and extends it:

- ✅ **Chaos checks** — `tests/chaos/run_chaos.py` deliberately breaks each
  dependency (kill vLLM mid-request, corrupt a facts file, make ChromaDB
  unwritable, fill the disk) and asserts "degraded but functional," never a
  hard 500 or crash loop. Guarded + pod-local (needs fs + supervisorctl),
  refuses without `ZIONS_CHAOS_CONFIRM`, each scenario self-restores. See
  [tests/chaos/README.md](tests/chaos/README.md). *(Run on the pod to fully
  exercise — the safe shape is built + unit-verified.)*
- ✅ **Disk-pressure handling** — `compactor/degrade.py`: when free space on
  `/data` drops below `COMPACTOR_MIN_FREE_MB_WRITES` (200), new-memory
  growth (extraction/indexing/rollup/persona-autocapture) pauses while chat
  + explicit user writes keep working. Surfaced in `/health/full`
  (`memory_writes`) and reflected in `status=degraded`. Fails OPEN.
- ✅ **vLLM restart resilience** — the compactor rides out a vLLM restart:
  unreachable-vLLM now returns a clean **503** (`model_unavailable`,
  retryable) on the non-stream + `/v1/models` paths, and a visible
  "model is starting/restarting" assistant message on the stream path,
  instead of an opaque 500 / dead stream.

### Theme 3 — Process & resource stability

- ✅ **Bounded background work** — `compactor/bgwork.py`: a pool caps
  concurrent async tails (`COMPACTOR_MAX_CONCURRENT_TAILS`, 4) and **sheds**
  beyond a hard outstanding ceiling (`COMPACTOR_MAX_OUTSTANDING_TAILS`, 64)
  rather than spawning unboundedly under load. Shed coroutines are closed
  (no leak); stats in `/health/full` (`background_work`).
- ✅ **supervisord restart policy review** — confirmed (and documented) that
  autorestart + startretries + startsecs send a genuinely-broken service to
  FATAL (visible) instead of fast-restart-looping. Policy intent commented
  in `supervisord.conf`; FATAL spot-and-recover runbook in OPERATIONS.md.
- ✅ **Memory/FD leak watch** — `tests/soak/soak_monitor.py`: samples the
  compactor's RSS + FD count over time (optional self-driven load) and flags
  monotonic growth. *(The instrument is built + verified; a real multi-day
  soak is the run-on-pod gate.)*

### Theme 4 — Operational confidence

- ✅ **Runbook** (`OPERATIONS.md`) — log-line reference, how to read
  `/health/full`, recovery for each known failure mode (incl. FATAL
  services), backups/restore, and image rollback. Landed in Theme 1, grew
  across the later themes.
- ✅ **Structured logging** — `compactor/logsetup.py`: `COMPACTOR_LOG_FORMAT`
  switches the compactor + selftest + backup between human `text` (default)
  and `json` (one object/line, greppable/aggregatable). Tiny stdlib JSON
  formatter — no python-json-logger dependency. (vLLM's own log format is
  left to `VLLM_EXTRA_ARGS` for operators who want it.)
- ✅ **Alerting hook (optional)** — `compactor/alert.py`: a single
  `COMPACTOR_ALERT_WEBHOOK` the boot self-test + backup daemon POST to on
  failure (Slack/Discord/generic). Off by default; best-effort (never
  blocks the job it watches).

**Prerequisite (met):** shipped after V2.2 — builds on V2.2's selftest
process model, `/health/full`, and Tier-3 harness. Theme 1 (data
durability) was promoted first since its failure mode is the only
*unrecoverable* one.

**V2.3 status: ✅ code-complete.** All four themes shipped (PRs #15, #17,
#18, + this one). The explicit standard was *failure-tested confidence over
speed* — Tier-1 covers every new module's failure paths, and three
"prove-it-on-the-pod-for-real" gates remain the operator's to run: a live
**restore** rehearsal (Theme 1), a **chaos** run (Theme 2), and a multi-day
**soak** (Theme 3). The instruments for all three are built + documented.

---

## Cross-cutting infrastructure (no version — ongoing)

Not tied to a specific feature version. These support every release and
should be picked up whenever they unblock something or address pain.

### CI/CD automation

**Current state (V1.9.x):** every release was hand-rolled — local `docker
compose build`, `docker tag`, `docker push`, manual scout scan, manual
git tag + branch + PR. This worked at the bug-fix cadence we were running
but it's not sustainable, and "did I remember to update `:latest`?" is
the kind of footgun that bites at the worst time.

**Phased rollout:**

| Phase | What ships | Trigger |
|---|---|---|
| **CI Phase 1 — PR validation** | `.github/workflows/pr-validate.yml` runs CPU smoke tests (`test_smoke.py`, `test_memory.py`, etc.) and a Dockerfile syntax lint (`hadolint`) on every PR | After V2.0 Phase 1 lands (small surface to test against first) |
| **CI Phase 2 — Tag-triggered build + push + scan gate** | `.github/workflows/build-and-push.yml` triggered on `v*.*.*` tag push: builds image, runs `docker scout cves --exit-on-vuln critical`, on success tags + pushes `:vX.Y.Z` and `:latest` to Docker Hub | After V2.0 ships, so the build-push pattern is stable |
| **CI Phase 3 — Scheduled re-scan** | `.github/workflows/scout-recheck.yml` cron-weekly: re-scan the current `:latest` for new CVEs that have been published since release. Opens a GitHub issue if Critical found. | Optional polish, after CI Phase 2 |
| **CI Phase 4 — Multi-variant matrix** | Build matrix expands to cu128 + cu130 (and future variants) using the parametric Dockerfile args we already shipped in 1.9.6 | When RunPod broadly rolls out driver 580+ |

**Vulnerability gating policy (CI Phase 2):**

| Severity | Action |
|---|---|
| Critical | **Block push.** Build fails, no tag pushed, no `:latest` update. |
| High | Warn in build summary; don't block. Manual review during release. |
| Medium / Low | No action — noise for our deployment shape (single-user, behind RunPod proxy). |

Override: if a Critical is in an upstream dep with no fix yet, add
`[scout-allow-critical: CVE-ID]` to the tag's commit message. Workflow
scans the message and exempts that specific CVE. Logged in release notes.

**Required GitHub Secrets** (one-time setup):
- `DOCKERHUB_USERNAME=angreg`
- `DOCKERHUB_TOKEN=<personal access token from Hub>`

No HF_TOKEN needed in CI — we don't pre-warm models at build time.

**Workflow file layout:**
```
.github/workflows/
├── pr-validate.yml      # On PR: smoke tests + hadolint
├── build-and-push.yml   # On tag v*.*.*: build → scout gate → push
└── scout-recheck.yml    # On weekly cron: re-scan :latest for new CVEs
```

**Effort: ~1 day per phase.** No new tools to introduce — Docker Scout
CLI is already what we use, just runs in Actions instead of locally.

### Orchestration evolution

You're correct that today's "everything in one container" shape has a
ceiling. The path forward is **trigger-based**, not version-based — we
move when a trigger fires, not on a schedule. Keep things integrated as
long as the integration costs less than the split would.

**Tier 0 — Current shape (V1.x, V2.x)**
```
[ single pod, single container ]
  ├── vLLM (GPU)
  ├── compactor (CPU, JSON files on /data)
  └── OpenWebUI (CPU, SQLite on /data)
[ one Network Volume holds all persistence ]
```
**Holds until:** ~10k conversations, single user, no need for multi-device
sync via direct DB access, single model serving.

**Tier 1 — Multi-container, single pod (V3 likely)**
```
[ one pod, docker-compose orchestrates ]
  ├── vllm (GPU)
  ├── compactor (CPU)
  ├── whisper-stt (CPU or shared GPU) ← V3.2 brings this
  ├── kokoro-tts (CPU)                 ← V3.3 brings this
  └── openwebui (CPU)
```
**Trigger:** V3 multimodal naturally splits the image — STT and TTS want
to be separate processes (different deps, different scaling characteristics).
Once we have 2+ supplementary services, splitting the main image into
multiple compose services is a no-cost organizational improvement.

**Effort:** moderate. RunPod's default pod is single-container; need a
custom entrypoint that runs `docker-compose up` inside the pod, OR use
RunPod's "Pods with docker-compose" template if/when available.

**Tier 2 — External memory store (V3+, only if a trigger fires)**
```
[ inference pod (GPU) ]     [ memory pod or managed DB ]
  ├── vLLM                    ├── PostgreSQL + pgvector
  └── compactor (stateless) ──►   (or Redis + ChromaDB server)
                              └── persistent volume / managed backups
[ frontend (anywhere) ]
  └── OpenWebUI or replacement
```
**Triggers (any one):**
- Cross-device direct DB sync required (laptop AND phone reading same
  memory in real time — OpenWebUI's web sync covers most of this today
  via shared backend, see "Frontend replacement triggers" below)
- Multi-user (each user's memory isolated but shared infrastructure)
- Need memory durability stronger than Network Volume backups (managed
  Postgres on Neon/Supabase has HA + point-in-time recovery built in)
- >10k conversations or need SQL queries over memory contents
- Multiple compactor instances need shared memory state (horizontal scale)

**Effort:** significant. Compactor becomes stateless; JSON files become
DB rows; embedded ChromaDB → ChromaDB server or pgvector. **~1-2 weeks**
of work plus DB provisioning. Major architectural milestone.

**Tier 3 — Full microservices on Kubernetes (almost certainly never)**
```
Kubernetes cluster (RunPod K8s or elsewhere):
  ├── inference Deployment (GPU node pool, vertical autoscale)
  ├── compactor Deployment (CPU, horizontal autoscale)
  ├── memory StatefulSet (Postgres + pgvector)
  ├── frontend Deployment (could be Vercel/Cloudflare Pages)
  └── ingress + auth provider + observability stack
```
**Trigger:** Enterprise SaaS shape — >100 active users, SLAs, multi-region.
**Almost certainly never relevant for this project.** Captured for
completeness, not because it's on the path.

### Repository structure: monorepo until a real reason to split

**Decision: stay monorepo.** Multi-repo overhead (separate CI/CD pipelines,
coordinated releases, harder cross-service refactors) only pays off when:
- Different teams own different services
- Different release cadences (compactor ships twice a week, vLLM image
  ships once a month)
- Different security/compliance boundaries (frontend public, backend not)

None apply for this project. As the project grows, the repo evolves
in-place:

```
zions-light-ai/
├── compactor/                  (will become its own service eventually)
├── pipelines/                  (OpenWebUI Functions)
├── stt/                        (V3.2 — speech-to-text service, faster-whisper)
├── tts/                        (V3.3 — new service directory)
└── deploy/
    ├── docker-compose.yml          (local dev — current single-container)
    ├── docker-compose.split.yml    (multi-container, V3 era)
    ├── runpod/
    │   ├── single-pod.template     (current)
    │   └── multi-pod/              (V3+ era, when we split)
    └── kubernetes/                 (if we ever get to Tier 3)
```

When a split eventually IS required, `git filter-repo` extracts a
directory with its full history into a new repo trivially. The cost of
NOT splitting prematurely is zero. The cost of splitting prematurely is
weeks of toolchain rework you didn't need to do yet.

---

## V3 — Multimodal (vision + voice)

**Goal:** Move from "text-only chatbot" to "AI assistant that can see and
hear." Three independent capabilities that can ship as separate sub-versions.

### V3.1 — Vision (image understanding)  ✅ Compactor-ready

**Status:** the compactor is vision-ready — `count_tokens` budgets for image
tokens (`COMPACTOR_IMAGE_TOKENS`) and `compact_if_needed` preserves
image-bearing turns verbatim instead of summarizing them away
(`compactor/test_vision.py`). Enabling vision is now an **opt-in `MODEL_REPO`
swap to a VLM** (presets + GPU sizing in `.env.example` / RUNPOD_DEPLOY.md);
image upload via OpenWebUI works with no further changes. Actual VLM
inference is the operator's on-pod validation gate.


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

### V3.2 — Speech-to-text (voice input)  🔨 In progress

**Status:** the STT service is **built and Tier-1-tested**; wiring +
on-pod validation in progress on branch `v3.2-stt`.

**What it adds:** User talks into their microphone; OpenWebUI sends audio
to an OpenAI-compatible `/v1/audio/transcriptions` endpoint; the transcribed
text becomes an ordinary prompt that flows through the compactor like any
typed message.

**How it works (as built):**
- New service `stt/server.py` — a thin FastAPI wrapper around **faster-whisper**
  (CTranslate2) exposing `/v1/audio/transcriptions` + `/v1/audio/translations`
  + `/health` + `/v1/models`. Renders every OpenAI response format
  (json / text / verbose_json / srt / vtt).
- Runs in its **own venv** (`/opt/whisper-venv`) as its own supervisord program
  (`[program:stt]`, port 9000, toggle `STT_ENABLED`) — faster-whisper's deps
  (ctranslate2 / av / onnxruntime) can never touch the vLLM or compactor venvs.
- **CPU by default** (`WHISPER_DEVICE=cpu`, int8) so it never competes with
  vLLM for VRAM (the A40 has ~no headroom at `GPU_MEMORY_UTILIZATION=0.90`).
  Flip to `cuda` only with real headroom.
- Default model **`base`** is prebaked into the image; swap via `WHISPER_MODEL`
  (+ a `/data` `WHISPER_DOWNLOAD_ROOT` for larger models that should persist).
- OpenWebUI's STT engine is pre-wired to the local service via `AUDIO_STT_*`
  env; the compactor is untouched (it never sees audio).

**Testing (beyond "it turns on"):**
- ✅ Tier-1 `stt/test_stt.py` — response-format rendering, subtitle/timestamp
  formatting, param pass-through, and the 400 / 503 / 500 error paths, all with
  a fake model (no GPU, no faster-whisper).
- ✅ Boot self-test gained a **functional** audio assertion (`selftest.py`
  `_check_stt`): on boot it POSTs a tiny generated WAV to the STT service and
  asserts a well-formed OpenAI response — proving the service decodes audio and
  runs, not just that the port is open. Gated on `STT_ENABLED` so it's only run
  where STT is deployed.
- ✅ A **quality eval** harness (`tests/eval/`): `wer.py` (word-error-rate
  metric, Tier-1-tested in `test_wer.py`) + `stt_eval.py`, which transcribes
  operator-supplied speech clips through the live service and scores WER against
  known transcripts — the "is it actually accurate?" layer the three standard
  tiers don't cover. Real clips are operator-supplied (synthetic silence can't
  measure accuracy).

**Model choices:**
- `base` (default, prebaked) — fast on CPU, good for clear speech
- `small` / `medium` — better accuracy, still CPU-viable
- `large-v3` — best accuracy; pair with `WHISPER_DEVICE=cuda` + VRAM headroom

**Effort: ~4-5 days.**

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

## V4 — Agentic tool use

**Goal:** give the model *agency* — the ability to call tools and
(sandboxed) run commands, not just produce text. Sequenced after V3.
**Full design spec: [compactor/V4_PLAN.md](compactor/V4_PLAN.md).**

**Core architecture:** vLLM parses tool calls but never executes them, so
the **compactor** hosts the tool-execution loop (ReAct-style: model emits
`tool_calls` → compactor runs the tool → appends results → re-calls vLLM →
loops to a `MAX_TOOL_STEPS` cap). No new service — it extends the existing
request-interception point. The V2 memory layer is the agent's persistent
substrate.

**Three tool tiers, strictly increasing risk:**
- **A — pure-Python tools** (memory query, math, allowlisted fetch) — no
  sandbox needed, ship first.
- **B — OpenWebUI Tools/Functions** — UI-triggered, user-in-the-loop.
- **C — command-line execution** — *non-negotiably gated on a sandbox*
  (no `/data` mount, no network unless allowlisted, resource caps,
  ephemeral FS, command allowlist) plus human-in-the-loop approval for
  mutations. A sandbox escape here is catastrophic — exactly where the V2.3
  "failure-tested before shipped" stance is load-bearing.

**The harness question — when do we build a purpose-built agent-run engine?**
*Not for the first step, unavoidably for the second.* Phase 1 (a bounded
single-request tool loop) needs no harness. A harness is **forced** the
moment any of these become requirements: (1) runs outlive a single HTTP
request, (2) human-in-the-loop pause/resume, (3) shell execution at scale
needs a sandbox pool, (4) multi-agent, (5) cross-run observability/cost.
The harness is then: a durable run/event store + sandbox-runner pool +
approval state machine + agent console (the point where OpenWebUI stops
sufficing). Build it around the trigger that actually bites — don't
speculate. Adopt off-the-shelf sandboxing/queues; keep custom only the
run-store + approval + memory glue.

**Proposed phasing:** V4.0 tool loop + Tier-A (no harness) → V4.1 read-only
sandboxed commands (sandbox only) → V4.2 mutating commands + approval
(harness begins) → V4.3+ durable run store, sandbox pool, agent console.

---

## Alternative inference backend (V1.10 candidate or V2-parallel)

**Trigger for considering:** sustained pain with vLLM's VRAM footprint on
A40-class hardware, or wanting a meaningfully smaller image.

**Switch to TabbyAPI + ExLlamaV2 (EXL2 quants)** rather than vLLM:

| | vLLM (current) | TabbyAPI + EXL2 |
|---|---|---|
| Backend image footprint | ~17 GB | ~3-5 GB |
| OpenAI-compat API | Native | Yes (TabbyAPI) |
| Magnum 22B quality at ~16 GB | AWQ 4-bit (Marlin) | **EXL2 5.0bpw** (usually higher quality) |
| Single-stream throughput on Ampere | Good | **Often faster** |
| Multi-user batching | Excellent | Weaker (n/a for single-user deploys) |
| Model architecture coverage | Very broad incl. VLMs | Llama-family solid, some lag |

For Zion's Light AI's profile (single user, creative writing, Magnum-family
models, A40), TabbyAPI + EXL2 is a credible better fit than vLLM. Reason
we're on vLLM is path-dependent from the original migration.

**Cost of the swap (when we eventually do it):**
- Dockerfile rewrite (replace vLLM install with TabbyAPI + exllamav2 wheels)
- `supervisord.conf` `[program:vllm]` becomes `[program:tabby]`
- `entrypoint.sh` preflight stays useful (same CUDA driver checks)
- compactor: NO changes — it's backend-agnostic, talks pure OpenAI-compat
- OpenWebUI: NO changes
- Default model: switch to an EXL2 variant (e.g. `LoneStriker/magnum-v4-22b-5.0bpw-h6-exl2`)
- Effort: ~1 day for Dockerfile + supervisord + entrypoint adjustments, ~1
  day for testing both single-user chat flow and compactor compaction
  against the new backend

**This is a worthwhile evaluation but not urgent.** V2 memory architecture
is more user-visibly valuable than a backend swap. Revisit after V2.0
ships, or sooner if vLLM keeps causing VRAM-shape problems.

---

## Frontend replacement triggers — when to consider replacing OpenWebUI

**Default position: don't.** OpenWebUI already covers what matters for
this project:
- ✅ Cross-device sync (web-based — phone, desktop, tablet all hit the
  same backend)
- ✅ Auth + user accounts
- ✅ Image upload (V3.1-ready)
- ✅ Voice input/output configurable hooks (V3.2/V3.3-ready)
- ✅ Function/tool calling UI
- ✅ Pipelines/Functions plugin system for custom behavior
- ✅ Active development, large community

Replacement is **only** worth considering when a specific trigger fires.
Listed by likelihood of relevance to this project:

| Trigger | Why it would force a replacement | Replacement effort |
|---|---|---|
| **Offline-first mode** | OpenWebUI is web — needs network. A native client could queue messages locally and sync when online. *(Realistic for "I'm on a plane with no wifi" use cases.)* | Large — full native app (Electron/Tauri/SwiftUI) |
| **OS-level integration** (system tray, global shortcuts, screen-context awareness) | Web app can't read your screen or own a tray icon. Wraps in a native shell. | Moderate — wrap OpenWebUI in Electron/Tauri, add OS hooks |
| **Push notifications** | "Model finished long generation, here's the result on your phone." Web push exists but OpenWebUI doesn't implement it. | Small-moderate — could be added as an OpenWebUI Function rather than replacement |
| **Branded product** (selling to others, distinct visual identity, app store presence) | OpenWebUI's branding shows through; deep customization is fork-territory. | Large — full UI rewrite |
| **Features OpenWebUI rejects** (you submit a PR, it's declined, you can't live without it) | Fork point. | Moderate — fork + maintain |
| **Bypassing OpenWebUI's auth model** (different identity provider, custom RBAC) | OpenWebUI's auth is opinionated. | Moderate |
| **OpenWebUI gets abandoned** | Maintained fork would be the natural answer first; only replace if no maintained fork emerges. | Variable |

**Cross-device sync is NOT a trigger** — that's the most common
misconception. OpenWebUI's web-based shared backend already gives you
"open a chat on desktop, continue it on phone" out of the box. You're
already getting that.

The realistic future trigger for this project is probably **offline-first
mode** if you ever travel with no connectivity, or **OS-level
integration** if you want screen-context awareness (model can see what
you're looking at). Both are V3+ at the earliest, and both could
plausibly be addressed by writing a thin native shell around OpenWebUI
rather than replacing it wholesale.

---

## Beyond V3 — speculative roadmap

These are directions worth keeping in mind but not committing to until V2
and V3 prove the architecture handles them. **Order does not imply priority.**

### Agentic capabilities (tool use / function calling)
- **Promoted to a committed line — see [V4 — Agentic tool use](#v4--agentic-tool-use)
  above and [compactor/V4_PLAN.md](compactor/V4_PLAN.md).** No longer
  speculative; the design (compactor-hosted tool loop, three tool tiers,
  sandbox boundary, the harness-trigger analysis) is captured.

### Code execution
- "Run this snippet" inside chat — the Tier-C sandboxed-command path of V4
  (see V4_PLAN.md). Requires the sandbox runner; never ships unsandboxed.

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
