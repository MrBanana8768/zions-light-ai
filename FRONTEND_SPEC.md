# Front-End Specification — a client built for this system

> **Status:** specification only. No implementation. This document defines
> *what* the replacement client must do and *why*; the *how* belongs to the
> implementing work.
>
> **Governing documents:** [COGNITIVE_ARCHITECTURE.md](COGNITIVE_ARCHITECTURE.md)
> governs on every question of honesty, claims, and what the system may
> represent itself to be — where this spec appears to conflict with it, that
> document wins. [ARCHITECTURE.md](ARCHITECTURE.md) governs layering,
> networking, and trust boundaries.

---

## 1. Purpose & non-goals

OpenWebUI has carried the project since V1 and did so well. It is a *generic*
chat client. Not every V3.0 incident originated with it — the 2026-08-24 chain
began with an image-token undercount of ours in v3.0.4, and the 2026-08-13
budget failure was entirely compactor-side. What is true, and is the reason for
this document, is that a generic client *amplified* our faults into user-visible
data loss, by making reasonable-for-a-generic-app decisions that are wrong here:

| Incident | What OpenWebUI did | Why it's wrong here |
|---|---|---|
| Binary file read as prose (2026-08-24) | Routed `lol.paint` to the document/text-extraction path by extension, fed raw bytes to the model as text | A client for a *vision-capable* assistant must classify by content, and refuse honestly rather than let the model improvise on noise |
| Chat crash on a discovered model | `capabilities: None` on an OpenAI-connection model crashed `process_chat` | The client must not assume its own metadata is populated |
| Empty-`messages` task calls | Background title/tag/follow-up calls sent `messages: []` | Task traffic must be distinguishable from conversation traffic |
| Phantom memory conversation | Task-call prompts fingerprinted into a conv_id and accumulated **105 "facts" — still accruing as of 2026-08-24** | Non-conversation traffic must never write to the memory store |
| 292k-token replay | Re-sends the entire transcript every message | The compactor exists precisely so this is unnecessary |
| Orphaned message tree (2026-08-24) | On a failed request, created the turn as a **new root** and moved `currentId` to it — 208 messages became invisible; the conversation was measured at `roots=5`, and the client neither detected nor reported that | A failed request must never damage the client's own history — and the client must be able to detect that it has |
| Silent context truncation (2026-08-24) | With `currentId` on an 8-message side branch, walked `parentId` back from it and sent **7 messages** on a 241-message conversation — no error, no signal, `msgs=3 → 5 → 7` across three turns with a stable `conv_id`, proving a fixed head rather than a sliding window | The set of messages sent is the most consequential thing a chat client does; it must be computed from a recorded intent, verified before sending, and shown to the user — never left as whatever a pointer happens to reach |
| Un-diagnosable history (2026-08-24) | Dual-writes the chain to a JSON blob *and* a normalized table that can diverge, with a composite `{chat_id}-{uuid}` primary key against bare-uuid `parent_id`; external repairs were overwritten twice from stale browser state, and a naive join produced a false "every message is orphaned" diagnosis mid-incident | A client's own history must be inspectable and repairable from outside the app, in one representation and one key space |

Truncation is listed separately from the orphaned tree deliberately, on four
grounds: root creation is a *write*-path defect and truncation is a *read*-path
defect; one affects what the user sees and the other affects what the model
sees; explicit `parent_id` discipline fixes the first and does nothing for the
second; and the tree damage stopped when the 400s stopped, while the truncation
persisted afterwards and was reproducible.

Those two are the sharpest, and the honest statement of what they prove is
narrower than "a client that owns its chain is immune." Ownership does not
confer immunity: a bounded window is still a decision about which stored
messages reach the model, and the client that owns its chain also removes the
only independent cross-check — the compactor's own count of what it has
indexed. What actually prevents this class of failure is an **enforced invariant
plus a visible count**: the message list posted is derived from the rendered
chain by a checked construction, and the client shows how many of how many it
sent. That is testable, it is what §4.1 and §12 now require, and it is the
argument for this project — not that a bespoke client cannot fail, but that we
can make it fail loudly.

**In scope:** a single-purpose client for the compactor stack — chat, memory
governance, voice, vision, and honest status.

**Explicit non-goals:** multi-tenant SaaS; model management/download UI; a
plugin or "function" marketplace; document-RAG features (the compactor owns
memory); arbitrary provider support (this client speaks to *this* backend).

---

## 2. Obligations the north star places on the interface

These are requirements, not aspirations. Each traces to a principle in
[COGNITIVE_ARCHITECTURE.md](COGNITIVE_ARCHITECTURE.md).

1. **Never present the system as a someone.** The UI refers to the assistant as
   "it" and by name, never with personhood language. No fabricated emotional
   signals: no "thinking…" anthropomorphism, no simulated typing hesitation, no
   mood indicators. Streaming progress is a *mechanical* indicator only.
   *(Principle 1: model functions, not souls; "the boundary we will not cross
   in our claims.")*
2. **Claim nothing unearned — including about itself.** The UI must not display
   confidence it cannot substantiate, invent provenance for a memory it cannot
   trace, or present a stale value as live. "Unknown" and "not available" are
   first-class UI states, designed for rather than hidden.
   *(Principle 2.)*
3. **Degrade honestly, fail loud.** Every backend failure surfaces as a *typed
   system notice*, visually distinct from assistant speech, that says what
   actually happened. The client must never render an error as though the
   assistant said it, and must never silently retry in a way that hides a
   failure. **Automatic repair of the client's own state is itself a failure
   event: any code path that moves the current leaf, drops a message, or
   reconstructs a chain must surface a typed notice, never a log line only.**
   See §12 for the catalogue.
   *(Principle 6, and the v3.0.5 honesty fix: a healthy backend rejecting our
   request must not be reported as "the model is restarting.")*
4. **Memory is the user's to see and govern.** V2.1 established user agency
   over memory through slash commands; this client makes it graphical. Every
   fact the system holds must be viewable, correctable, and deletable by the
   user without special knowledge. *(Faculty D + the V2.1 user-agency theme.)*
5. **Transparency of memory, not theater of interiority.** The UI shows *what is
   stored and what was injected* — data the system genuinely has. It must not
   present speculative "inner state," simulated reasoning it did not perform,
   or a dashboard implying self-awareness. *(The "don't build the window"
   refinement.)*
6. **The human stays the moral locus.** Any future agentic action (V4) requires
   explicit human approval in the UI, with the action legible *before* it runs.
   No silent execution. *(Principle 3.)*
7. **Never reduce the model's context silently.** The assistant can answer only
   from the context it is handed. A layer that truncates, drops, reorders, or
   strips that context without saying so is not rendering badly — it is changing
   what the system is able to know while presenting the conversation as whole.
   The binding invariant: **the conversation the user sees and the messages
   actually sent must agree, or the difference must be visible on screen** —
   every turn withheld is either supplied by the compactor's memory layers or
   explicitly marked. A divergence is a correctness defect ranked with data loss,
   never cosmetic: it blocks release, it fails the integrity check of §4.1 rather
   than an adjacent one, and it surfaces at the moment of detection. On demand
   the client owes an exact account of its own outbound payload — never an
   interpretation of the model's reasoning (see 5).
   **This obligation is not the client's alone.** Every layer that composes the
   model's context is bound by it: the compactor's summarization, image
   retention, and hard-budget shedding all remove content the user can still see
   in the thread, and today none of that reaches her. The client-side
   instantiation is §4.1 and §12; the compactor-side instantiation is the
   budget-shed signal in §15, which is **required**, not optional.
   *(Principle 6: never paper over a failure. Principle 2: claim nothing
   unearned. Written after 2026-08-24, when a 241-message conversation sent 7
   messages with no signal of any kind.)*

> **Excluded on review.** An earlier draft of obligation 7 argued that a
> truncating client makes the model "bear false witness." That does not hold.
> All memory in the compactor is `conv_id`-scoped
> ([memory.py:137](compactor/memory.py:137)); the only cross-conversation path is
> an admin-triggered persona copy. When the assistant said it has no persistent
> memory across conversations, it was describing the system accurately, and a
> healthy 241-message context would have produced the same answer. The obligation
> stands on context fidelity, which is mechanically demonstrable, and not on
> induced dishonesty, which is not.

---

## 3. Architecture position

Per [ARCHITECTURE.md](ARCHITECTURE.md), the front end splits out **before** V4
and eventually moves to its own repository (Decision 7).

```
[ browser ]
     │  HTTPS
[ front end ]  ← its own container, own lifecycle
     │  private network
[ compactor :8080 ]  ← THE SINGLE FRONT DOOR
     ├── vLLM :8000   (never contacted directly by the client)
     ├── STT  :9000
     └── TTS  :9001
```

**Rules:**
- The client talks to the **compactor** for everything conversational. It must
  never hold a vLLM URL.
- STT/TTS may be reached directly on the private network *or* relayed through
  the front end's own server; the client must not expose those ports publicly.
- The client is **stateless with respect to memory** — the compactor owns
  facts, summaries, embeddings, personas.

### 3.1 Hard prerequisite — the admin API is localhost-only today

Every `/admin/*` route is gated by `_require_localhost` in
[compactor/main.py](compactor/main.py). A front end in a *separate container*
has a non-local source IP and **will receive 403 on every memory endpoint.**

Therefore: **the memory features in §7 cannot ship until the compactor exposes
an authenticated admin surface.** The mechanism already exists on branch
`v4/foundation-compactor-auth` ([PR #30](https://github.com/MrBanana8768/zions-light-ai/pull/30)):

- `COMPACTOR_API_KEY` env; when empty, auth is disabled (backward compatible)
- Protected prefixes: `/v1/*`; exempt: `/health`, `/health/full`
- `Authorization: Bearer <key>` (a bare key is also accepted)
- Constant-time comparison (`hmac.compare_digest`)

**Required extension (a "server ask", §15):** protect `/admin/*` with the same
key and drop `_require_localhost` when a key is configured. Until then the
client ships chat/voice/vision only, with memory features dark.

---

## 4. Data flow — compactor-native (the core design decision)

OpenWebUI re-sends the entire transcript on every message. This client does
not.

**Ownership:**

| Thing | Owner | Notes |
|---|---|---|
| Transcript (what was said) | **client** | local store; the client renders from it |
| Memory (facts, summaries, embeddings, persona) | **compactor** | client reads/edits via the admin API |
| conv_id (the key joining them) | **client** | generated once per conversation |

**Rules:**

1. **The client generates `conv_id`** (UUIDv4) at conversation creation and
   sends it on every request as `X-Conversation-Id`. This is the compactor's
   *preferred* path (`resolve_conv_id` in [compactor/memory.py](compactor/memory.py)),
   and it is stable across edits to the system prompt.
   *The hash-fallback path must never be relied on:* it fingerprints
   `sha256(system|||first_user[:512])`, so it silently changes when the opening
   turn changes — the exact defect that orphaned a conversation's memory in
   v3.0.1. **The client must also be able to confirm which path the compactor
   actually used.** On 2026-08-24 the compactor logged
   `conv_id=6aca8bcdf603d584 source=hash msgs=7`: OpenWebUI was on the forbidden
   fallback path and had no way to know. A client that believes it is sending
   `X-Conversation-Id` but is being resolved by hash has a bug it cannot
   otherwise observe. See the received-context echo in §15.
2. **Send a bounded window, derived from the chain and verified against it.**
   Default: the system/persona message plus the last *N* turns (N configurable;
   align with `COMPACTOR_KEEP_RECENT_TURNS`). The compactor's memory layers
   supply everything older. Sending full history is a **spec violation**.
   A bounded window is only legitimate when it is *intended*: before each request
   the client computes and records
   `window_intent = min(N, turns_on_the_current_chain)`, and the realised send
   set must equal it. A send shorter than intent is an error condition, never
   compliance. Without this, a starved context and a correct window are the same
   observation — which is exactly why 7 messages left a 241-message conversation
   on 2026-08-24 without anyone being able to call it wrong.
3. **Preserve template invariants.** The window must begin with a `user` turn
   after any system message and strictly alternate `user`/`assistant`.
   Mistral-family templates reject anything else with a 400. The client must
   never send two consecutive same-role turns.
4. **Regenerate/edit reuse the same `conv_id`.** Branching is a *client-side*
   concern; the compactor sees one conversation.
5. **Forking a conversation** calls `POST /admin/conversations/{conv_id}/fork`
   so the new conversation inherits memory, and the client stores the returned
   id.
6. **The client must not mutate the window mid-stream.**
7. **The send set is a checked construction, not a side effect.** The message
   list posted to the compactor is produced by an explicit selection over the
   current chain and then verified: every element present, contiguous, in order,
   alternation intact (rule 3), and count equal to `window_intent`. On mismatch
   the client does not send — it raises `context_truncated` (§12). The count sent
   and the range covered are recorded against the turn.
8. **Task traffic is marked at the protocol level.** Background title, tag, and
   follow-up calls carry an explicit request-kind marker so the compactor can
   refuse memory writes for them (§15). A one-message request has no
   user/assistant exchange to remember; today such calls hash to a stable
   conv_id and accumulate facts into a conversation that does not exist.

### 4.1 Message-chain and context-fidelity integrity (written 2026-08-24; revised the same day after the truncation finding)

**Requirement, in two parts:**
**(a)** a failed request must never damage stored history; **and**
**(b)** the client must never send the model a context that does not match the
conversation the user is looking at.

Part (a) is the requirement this section was first written for. It was already
satisfied by the live system on 2026-08-24 — blob and table both measured
`total=241, deepest=208` — while the user still lost her conversation. Part (b)
is the property that failed. A requirement a system can fully satisfy while
producing the exact harm it was written to prevent is the wrong requirement.

**Chain writes**

- A new turn is appended to the chain **only after** the request is accepted and
  a response (or a typed failure) is recorded against it.
- Every message carries an explicit `parent_id`; the client must **never** write
  a message whose parent is missing.
- On failure, the user turn is retained **in place** on the existing chain,
  marked `failed`, with retry available. No new root, no pointer move. A failed
  request has no structural effect at all.
- Branch variants (regenerations/edits) are siblings under a shared parent, with
  UI to move between them. The current leaf may only advance to a child of where
  it already stands; any other move is an explicit, validated branch selection.

**Integrity check — the exact properties, named so no adjacent property can be
substituted for them.** On load, and again after any request that returned a
typed failure, the client evaluates all of:

1. `roots == 1` for the conversation
2. `current_leaf` is reachable from that root
3. the chain from `current_leaf` contains every message the thread renders
4. every `parent_id` resolves in the same key space as the `id` it names
5. no message is unreachable from the root

The following must **not** be used as an integrity check, alone or combined:
`max_depth(tree) ≥ threshold` ("a long chain exists"); `count(messages) ==
expected` ("nothing was deleted"); `chain_from_current ≥ threshold`; or any check
that does not name the current leaf. On 2026-08-24 a check that measured the
deepest chain reported the conversation healthy at `deepest=208` while
`chain_from_current=8`. The number was true and irrelevant. Depth, chain length,
and total count are diagnostics; they are never the criterion. **Standing test
case: 241 messages, 5 roots, current leaf at depth 8 — the check must fail.**

**Repair policy.** On any failed check the client **quarantines and reports**. It
states what it found — "this conversation has 5 separate branches; the one you
are viewing has 8 of 241 messages" — offers the candidate chains with their
lengths and dates, and **does not move the pointer on its own**. Automatic repair
is permitted only where it is provably lossless *and* surfaced as a typed notice
(§12). The previous wording of this section — "re-derive the leaf from the
longest valid chain and log it — self-healing" — authorized a silent pointer
move, which is the mechanism of the incident and is prohibited by §2.3.

**Pre-send gate.** Before every request, recompute the send set from the chain
and compare it to `window_intent` (§4 rule 2). On mismatch, do not send: raise
`context_truncated`. A degraded answer the user cannot detect is worse than a
visible refusal.

**Post-send receipt.** Record per turn: messages sent, the range they covered,
images still visible to the model, and memory injected. Expose it (§12).

---

## 5. API contract (verified against the code)

### 5.1 Chat

`POST /v1/chat/completions` — OpenAI-compatible.

- Headers: `Content-Type: application/json`, `X-Conversation-Id: <uuid>`,
  `Authorization: Bearer <key>` when configured. The client must read the
  response-side context echo (§15): resolved `conv_id`, resolution `source`, and
  the message count the compactor actually received.
- Body: `model`, `messages`, `stream`, optional `max_tokens`
  (the compactor clamps `max_tokens` above `MAX_MODEL_LEN/2` and reserves room
  for the reply; the client should keep requests modest).
- `stream: true` → SSE, `data: {chunk}` … `data: [DONE]`.
- Multimodal content uses OpenAI content-parts:
  `[{"type":"text","text":…},{"type":"image_url","image_url":{"url":"data:image/png;base64,…"}}]`

**Failure bodies the client must handle by type** (all OpenAI-error-shaped):

| Condition | Marker | UI treatment |
|---|---|---|
| Empty/invalid messages | `code: empty_messages`, HTTP 400 | client bug — log, do not show raw |
| Backend down/restarting | `MODEL_RESTART` text, HTTP 503 | "the model backend is starting up" notice, retry affordance |
| Backend rejected the request | `REQUEST_REJECTED` text (4xx) | "couldn't be processed — a problem on our side" notice; **not** a restart claim |
| Non-JSON upstream body | HTTP 502 | generic backend-fault notice |
| Context length exceeded | vLLM context-length 400 surfaced by the compactor | "this request exceeded the model's context limit — **nothing was lost**"; offer retry with fewer images or a shorter window |

In streaming mode these arrive as synthesized assistant-role chunks; **the
client must detect and re-render them as system notices**, not as speech.

### 5.2 Models & health

- `GET /v1/models` — OpenAI list shape; `data[].id` is the model name.
- `GET /health` — cheap liveness.
- `GET /health/full` — deep probe. Shape (from [compactor/health.py](compactor/health.py)):
  `vllm: {ok, latency_ms, models[], error}`,
  `storage: {ok, writable, root, free_gb, total_gb, error}`, plus memory stats
  and background-pool stats. Drives the status surface (§12).

### 5.3 Slash commands

Handled inside the compactor with **zero token cost** — vLLM never sees them.
From [compactor/commands.py](compactor/commands.py):

| Command | Aliases | Purpose |
|---|---|---|
| `/help` | — | command list |
| `/list-facts` | `/facts` | what it remembers here |
| `/list-archive` | `/archive` | archived (stale) facts |
| `/remember <text>` | — | add a fact manually |
| `/forget [substring]` | — | remove facts (all, or matching) |
| `/why` | `/why-did-you-say-that` | what was injected into the last turn |

They return synthetic completions. **The client should render these as system
output, not assistant speech**, and should offer graphical equivalents (§7) —
keeping the commands as the power-user path.

### 5.4 Admin / memory API (all currently localhost-gated — see §3.1)

| Method & path | Purpose |
|---|---|
| `GET /admin/conversations` | list known conversations |
| `GET /admin/conversations/{id}` | summary of one conversation's memory |
| `GET /admin/conversations/{id}/facts` | facts list |
| `DELETE /admin/conversations/{id}/facts` | forget facts (all/matching) |
| `GET /admin/conversations/{id}/summary` | L1/L2/L3 summary state |
| `GET /admin/personas` | persona library |
| `GET/POST/DELETE /admin/conversations/{id}/persona` | persona CRUD |
| `POST /admin/conversations/{id}/inherit-persona` | copy a persona in |
| `GET/POST /admin/conversations/{id}/archive` | view / archive stale facts |
| `POST /admin/conversations/{id}/restore` | restore from archive |
| `POST /admin/conversations/{id}/dedup` | run fact deduplication |
| `GET /admin/conversations/{id}/export` | export bundle |
| `POST /admin/conversations/import` | import bundle |
| `POST /admin/conversations/{id}/fork` | fork with memory |
| `GET /admin/selftest` | boot self-test result |
| `GET/POST /admin/backups`, `GET /admin/backups/verify` | backup status/run/verify |

**Data shapes:**
- Fact: `{"text": str, "added_turn": int, "last_used": int}`
- Export bundle: `{"version": "v2.1", "exported_at": int, "source_conv_id": str,
  "facts": [...], "summary_state": {...}, "episodic": [{"turn_index", "document"}]}`

---

## 6. Parity matrix vs OpenWebUI

| Capability | OpenWebUI | This client | Verdict |
|---|---|---|---|
| Streaming chat | SSE | SSE, same | parity |
| Stop generation | yes | yes — **and never memorizes a partial reply** (compactor skips incomplete streams) | improved |
| Edit / regenerate / branch | yes | yes, with explicit parent links **and validated branch selection** (§4.1) | improved |
| Context actually sent | derived by walking `parentId` back from `currentId`; sent 7 messages on a 241-message chat with no signal | computed from a recorded intent, verified pre-send, disclosed per turn (§4 rule 7, §4.1, §12) | **improved — this is the differentiator** |
| History integrity visibility | none — a 5-root tree renders as a short thread and reports nothing | roots and branch state inspectable; corruption surfaced, never auto-repaired (§4.1) | improved |
| Markdown, code blocks, copy | yes | yes; syntax highlighting, copy-to-clipboard | parity |
| Conversation list, rename, delete, search | yes | yes; search local, instant | parity |
| Auth | account system | single-user first, multi-user-ready schema (§10) | simplified |
| Voice in/out | via `AUDIO_*` wiring | direct STT/TTS integration (§9) | parity |
| Image upload | extension-based | **content-sniffed, modality-aware** (§8) | improved |
| Memory visibility | slash commands only | **first-class UI** (§7) | improved |
| Model selection | full manager | read-only display of the served model | reduced (deliberate) |
| Documents / RAG | built-in | out of scope | dropped |
| Plugins / functions | marketplace | out of scope | dropped |
| Mobile / PWA | responsive | responsive, installable | parity |
| Themes, a11y, i18n | partial | required (§13) | improved |

---

## 7. Memory as a first-class surface (the differentiator)

This is what justifies building a client at all. A per-conversation **Memory
panel**:

- **Facts** — list with `added_turn` and `last_used`; inline edit; delete one or
  many; manual add. Mirrors `/list-facts`, `/remember`, `/forget`. Every fact
  must be traceable to the turn that produced it.
- **Summaries** — read-only L1/L2/L3 stack, showing what has been consolidated
  and what remains verbatim. Honest about compression: the user should be able
  to see *that* older turns were summarized.
- **Persona** — editor plus library; inherit into a new conversation. Persona is
  churn-exempt in the backend; the UI must make clear it is not evicted like a
  fact.
- **Archive** — browse archived (stale) facts; restore. Reinforces that
  forgetting here is *graceful*, not destruction.
- **"What it was given"** — per-message affordance mapping to `/why`, showing
  what was injected for that turn, and the transcript window sent with it. The
  older label, "Why did it say that?", claimed an interiority the feature does
  not have — it returns what was *injected*, a fact about our conduct, not a
  reason. §2.5 prohibits exactly that claim; the honest name is also the
  compliant one.
- **Retrieval transparency** — which facts / retrieved exchanges / summaries were
  injected into the current request. *Needs a small server addition (§15).*
  This must cover the **transcript window** as well as memory injection. On
  2026-08-24 the memory layer was intact and the transcript was starved; a panel
  showing only facts, retrieval and summaries would have shown green while the
  user was talking to a 7-message stub. A conversation whose memory holds
  hundreds of turns while the client sends 7 is self-evidently broken, and the
  panel should make that comparison visible.
- **Export / import / fork** — bundle download, restore, and fork-with-memory.

Design constraint from §2.5: this panel shows **stored data and injection
decisions** — never speculative inner state.

The dividing line is authorship. Showing the user the payload *we composed and
sent* is the client disclosing its own conduct to the party it acted on — that is
accountability, and it runs client → user. Read-access into state the system
generated runs observer → system and is the window §2.5 refuses. The test: **the
view must be diffable, never interpretive.** It shows the rendered chain against
the sent window and marks the delta; it never narrates, and it never shows
attention, logits, hidden states, or simulated deliberation. Keep it pull, not a
continuously running dashboard.

---

## 8. Modality handling (the `.paint` lesson)

1. **Classify by content, never extension.** Sniff magic bytes (PNG `89 50 4E
   47`, JPEG `FF D8 FF`, GIF, WebP). A `.paint` file containing PNG bytes is an
   image; a `.png` containing garbage is not.
2. **Refuse honestly.** A file that is not a supported image is rejected *in the
   UI* with a plain explanation. Never send unclassifiable bytes to the model
   as text.
3. **Modality awareness.** The client learns whether the backend can see (§15
   asks for an endpoint; otherwise infer from a rejection) and, on a text-only
   model, **disables upload with a reason shown** rather than letting the user
   discover it through failure.
4. **Retention is visible.** The backend keeps only the N most recent images
   (`COMPACTOR_MAX_RETAINED_IMAGES`, default 1). The UI should mark which images
   the model can still see versus those now represented by a text note —
   otherwise the user reasonably assumes it still sees everything.
5. **Cost honesty.** Images are expensive (thousands of tokens each). The
   composer should indicate that an attached image consumes significant context,
   **and must label the figure as an estimate.** Since v3.0.5 the compactor
   learns the real token count from vLLM instead of guessing;
   `COMPACTOR_IMAGE_TOKENS` is now a fallback, not the source of truth. An
   authoritative-looking wrong number is what v3.0.4 shipped, and it is the head
   of the 2026-08-24 chain.

---

## 9. Voice

- **Input:** mic capture → `POST :9000/v1/audio/transcriptions`
  (OpenAI-compatible, multipart). `/v1/audio/translations` also exists.
  Push-to-talk and tap-to-toggle; show interim state; never auto-send without
  the user confirming the transcript (transcription is fallible — showing it
  before sending is an honesty requirement).
- **Output:** `POST :9001/v1/audio/speech` → WAV. Per-message "read aloud" plus
  an auto-speak option. Playback controls; audio is not a modal state trap.
- Both services expose `/health` and `/v1/models` for capability detection.
- Voice must be **optional and clearly toggleable** — `STT_ENABLED` /
  `TTS_ENABLED` may be false on the backend, and the UI must reflect that.

---

## 10. Auth & sessions

- **Single-user first:** one account, password login, long-lived session
  (replacing `WEBUI_AUTH` / `WEBUI_SECRET_KEY`).
- **Multi-user-ready:** `user_id` present in the local store schema from day
  one, so multi-user is a feature addition and not a migration.
- **Session-key stability is a first-class operational concern.** A rotated
  signing secret invalidates sessions and *looks like data loss* to the user.
  The client must (a) distinguish "logged out" from "no data" with explicit
  copy, and (b) never render an empty state that implies deletion.
- Secrets are server-side only; the compactor API key is never exposed to the
  browser (the front end's server holds it and proxies).

---

## 11. Client storage

Requirements: durable across restarts, fast local search, exportable, and — the
hard constraint — **it must not repeat the SQLite-on-network-volume failure**
that corrupted `webui.db` twice in two weeks.

Options, with trade-offs:

| Option | Pros | Cons |
|---|---|---|
| **Browser IndexedDB** | zero server storage, instant, survives pod redeploys | per-browser; no cross-device; export becomes essential |
| **Server-side, local disk + periodic archive to `/data`** | cross-device; matches the Postgres-sidecar decision | needs backup discipline |
| **Postgres sidecar** (ARCHITECTURE Decision 4) | the decided state home; crash-safe | infrastructure work first |

**Recommendation:** align with ARCHITECTURE Decision 4 — server-side store on
**local disk** with archives to the volume, converging on the Postgres sidecar.
Under no circumstances put the primary write-hot store directly on the network
volume.

Beyond placement, two structural rules follow from 2026-08-24. Both are
requirements, not preferences, and both must be enforced by the store rather than
by convention.

### 11.1 One representation of the chain

The chain is represented exactly once, as message records each carrying its own
`parent_id`. No record anywhere may hold a serialized copy of the chain: no
`messages` array on the conversation, no `history` object, no id-keyed message
map, no column that must be rewritten when a message is appended. A schema test
must assert this.

The storage API must expose **no operation that accepts a whole chain**. Writes
are per-message deltas: `create_conversation` (which also creates the single
root), `append_message(conv_id, parent_id, …)`, `update_message_state`,
`append_stream_delta`, `select_leaf`. An endpoint of the shape `PUT /chats/{id}`
carrying a chain — or any path where a client's in-memory view can overwrite
stored structure — is a spec violation. OpenWebUI's dual write of a `chat.chat`
blob and a `chat_message` table is what let a stale browser tab revert an
external repair twice on 2026-08-24.

`parent_id` and `conv_id` are immutable after insert; reparenting is not an
operation this system has. Messages are tombstoned, never hard-deleted — a hard
delete is the remaining way to manufacture an orphaned subtree. **A tombstone
applies to the whole subtree beneath it**, so that hiding a message can never
leave its children unreachable and fail the §4.1 check.

A denormalized read model (materialized active path, search index, render cache)
may exist for performance under three rules: it is computed by a pure function of
the base tables in the same transaction as the write that invalidates it; no
write, validation, or integrity check may read from it; and a
`rebuild_read_models(conv_id)` routine exists with a test that drops and rebuilds
it and asserts equality. If dropping it entirely at any moment would cause
user-visible loss, it is not a read model — it is a second truth, and it is
forbidden.

Concurrent writers (second tab, second device) are handled by optimistic
concurrency on a `rev` column, never last-write-wins on a blob. A stale writer
refetches and retries; it must never overwrite. It loses its own write, loudly.

### 11.2 One key space

A message id is a bare UUID (v7 recommended). Ids are never composite, never
embed the conversation id, and are never built by string concatenation anywhere
in the codebase — a grep for message-id-forming concatenation is a
review-blocking finding. `parent_id` is a real foreign key to `message(id)` in
the same key space, enforced by the store; and containment is enforced by key as
well, with the parent FK composite on `(conv_id, parent_id)` so a message can
never be parented into another conversation.

OpenWebUI's `chat_message` uses a composite `{chat_id}-{uuid}` primary key
against a bare-uuid `parent_id`. Two key spaces in one table produced a false
"every message is orphaned" reading and a wrong diagnosis during a live incident.

```sql
CREATE TABLE message (
    id          uuid        PRIMARY KEY,
    conv_id     uuid        NOT NULL REFERENCES conversation(id) ON DELETE RESTRICT,
    parent_id   uuid        NULL,          -- NULL == the conversation root, and only it
    role        text        NOT NULL CHECK (role IN ('system','user','assistant')),
    content     jsonb       NOT NULL,      -- the wire form (§5.1 content-parts)
    state       text        NOT NULL DEFAULT 'complete'
                            CHECK (state IN ('pending','streaming','complete','failed')),
    error       jsonb,
    created_at  timestamptz NOT NULL DEFAULT now(),
    deleted_at  timestamptz,
    UNIQUE (conv_id, id),
    CONSTRAINT message_parent_fk
        FOREIGN KEY (conv_id, parent_id) REFERENCES message (conv_id, id)
        ON DELETE RESTRICT,
    CONSTRAINT message_not_self_parent CHECK (parent_id IS DISTINCT FROM id)
);

-- exactly one root per conversation
CREATE UNIQUE INDEX message_one_root_per_conv
    ON message (conv_id) WHERE parent_id IS NULL;

-- the leaf pointer can only ever name a message of this conversation
ALTER TABLE conversation
    ADD CONSTRAINT conversation_leaf_fk
    FOREIGN KEY (id, current_leaf_id) REFERENCES message (conv_id, id)
    DEFERRABLE INITIALLY DEFERRED;
```

**The root is synthetic.** `create_conversation` writes a single `role='system'`
root node holding the system/persona message, and the first user turn is its
child. This is what makes "exactly one root" enforceable without losing
legitimate behaviour: editing the first user message produces a *sibling under
the synthetic root*, not a second root. In OpenWebUI, editing the first message
creates a parentless node — which is one plausible source of the five roots
measured on 2026-08-24, and the reason the spec must state the invariant rather
than the story.

Structural immutability is enforced by a `BEFORE UPDATE` trigger that raises if
`id`, `conv_id`, or `parent_id` changes. On IndexedDB, which has no foreign keys,
all writes go through a single writer module performing each write inside one
`readwrite` transaction spanning both stores and aborting on any violation; no
other module may open a `readwrite` transaction on them, enforced by lint and
review.

Note what these constraints buy without any application code: a message cannot
exist without a live parent; a second root cannot be created; the leaf pointer
cannot name a foreign or nonexistent message; nothing can be reparented into a
cycle. **Every mechanical step of the 2026-08-24 failure is rejected by a
constraint** — the failed turn could not have become a new root, and the pointer
could not have moved to it.

### 11.3 The current-leaf pointer

The pointer lives in exactly one place, `conversation.current_leaf_id`. It is
never duplicated into localStorage, a URL fragment used as a write source, a read
model, or an in-memory store later written back. UI state may cache it for
rendering; the cache is never a write source.

Two operations may move it, both validating before commit. **Append** is a
compare-and-swap in the same transaction as the insert, conditioned on the new
message's parent being the current leaf (`… WHERE rev = $expected AND
current_leaf_id = $parent`, asserting rowcount 1 or rolling back). Because the
pointer can only advance to a child of where it stood, and a child must have a
live parent, **an append can never orphan history**. **Explicit branch selection**
runs the reachability predicate inside the transaction and rolls back if it
fails. No other code path writes the pointer, and the pointer is never repaired
silently on read (§4.1).

### 11.4 Audit

One function, `audit_conversation(conv_id)`, one recursive query, shared by the
load path, the test suite, the importer, and a CLI. It reports `roots`, `total`,
`reachable_n`, `missing_parent`, `leaf`, and `leaf_on_tree`, and
`PASS ⟺ roots = 1 ∧ missing_parent = 0 ∧ reachable_n = total ∧ leaf_on_tree`.
It runs before first render, after any typed failure, in CI after every test that
mutates a chain, and as `zl-chain audit --all`, which exits non-zero on any FAIL.
Any FAIL produces a typed notice (§12) and is never auto-resolved by a rule that
discards nodes. Every integrity check added in future must state, in a comment on
the check itself, which of these it verifies; a check that cannot name one is not
a check.

CI must include tests asserting the store *rejects*: a second root; a message
with an unknown parent; a message parented into another conversation; a pointer
move to an unreachable node; an update that changes `parent_id`.

### 11.5 Export

Full export/import of the transcript, pairing with the compactor's memory bundle
so a conversation is portable in both halves. A human with a database client must
be able to inspect and repair the chain from outside the app, and their repair
must survive the next send — §11.1's no-whole-chain-write rule is what makes that
true.

---

## 12. Error & status surfaces

**Typed system notices** (visually distinct from assistant messages):

| Type | Trigger | Message intent |
|---|---|---|
| `backend_starting` | 503 / `MODEL_RESTART` | it's coming up; retry shortly |
| `request_rejected` | 4xx / `REQUEST_REJECTED` | our side couldn't process it |
| `backend_fault` | 502 non-JSON | upstream returned something unusable |
| `context_trimmed` | budget shedding fired | older turns were dropped to fit — *the user deserves to know when history stopped reaching the model* |
| `offline` | network failure | the client cannot reach the backend |
| `unsupported_file` | modality/sniffing refusal | this file type can't be read |
| `context_truncated` | realised send set ≠ `window_intent` (§4 rule 7) — **client-side, no server ask needed** | only N of M messages would have reached the model; the request was not sent |
| `chain_corrupt` | `roots > 1`, unreachable leaf, missing parent, or the rendered thread not contained in the chain from the current leaf | state what was found, offer the branches, change nothing automatically |
| `context_length_exceeded` | vLLM context-length 400 surfaced by the compactor | the request was too large — nothing was lost |
| `conv_id_fallback` | the context echo (§15) reports `source=hash` while the client sent a header | memory for this conversation may be keyed somewhere else |

**Status surface** from `/health/full`: model reachability + latency, storage
writability and free space, memory counts, background-pool depth. Plus the boot
self-test result (`/admin/selftest`).

**The context receipt.** Notices fire on anomaly; the receipt makes the normal
case legible so the anomaly is noticeable. Always available for the active
conversation, without a dashboard: messages on the active path, messages in the
conversation, branch count, and **how many messages the last request actually
carried**. No user of this client should ever be in the position of being unable
to tell that 7 of 241 messages were sent.

`context_trimmed` reports *server-side* shedding and depends on the budget-shed
signal in §15, now a required ask. `context_truncated` is the client's own and
depends on nothing — the 2026-08-24 truncation happened entirely client-side,
with no budget event and no error, and any implementer who reads this section as
"blocked on the compactor" has misread it. Both matter for the same reason: on
2026-08-24 a user could not tell that her conversation had stopped reaching the
model.

---

## 13. Quality bars

- **Streaming:** first token rendered < 100 ms after arrival; no layout thrash
  during streaming; smooth at 60 fps on a mid-range laptop.
- **Accessibility:** WCAG 2.1 AA; fully keyboard-operable; screen-reader-correct
  roles for the message list and streaming updates (`aria-live` used carefully —
  streaming text must not spam announcements); visible focus; respects
  `prefers-reduced-motion`.
- **Responsive:** mobile-first; the composer and message list must work
  one-handed on a phone; installable PWA.
- **Theming:** light / dark / system, with the theme applied before first paint.
- **i18n-ready:** no hard-coded user-facing strings.
- **Performance budget:** initial JS < 200 KB gzipped; interactive < 2 s on a
  mid-range device; a 500-message conversation scrolls without virtualization
  jank.

Every bar above is performance, accessibility, or presentation. §4.1 is the
document's central argument, so correctness gets bars too:

- **Context fidelity (zero tolerance):** the set of messages sent equals the set
  the receipt reports, on every request. A test suite asserts this against
  adversarial chains: multi-root, missing parent, deep-versus-current divergence,
  mixed key spaces, mid-chain failure, and the standing 241/5/8 case.
- **No silent state mutation:** an automated test asserting that no code path
  moves the current leaf or alters chain structure without emitting a typed
  notice.
- **Integrity check cost:** the §4.1 checklist runs on load and pre-send within a
  stated budget on a 500-message, multi-branch conversation — the same shape as
  the 500-message scroll bar above.
- **Diagnosability:** one command reports `total / current_leaf /
  chain_from_current / deepest / roots` for any conversation. That five-tuple is
  what finally settled the 2026-08-24 diagnosis after hours; this client should
  emit it by construction rather than requiring a hand-written probe.

---

## 14. Stack — requirements, and a reasoned recommendation

**Normative requirements** (the stack must satisfy these; the choice is not
itself normative):

1. Native SSE/streaming handling with incremental DOM updates
2. Real client-side state (branching message tree, optimistic updates)
3. Small runtime — the budget in §13 is the constraint
4. Server-side component for secret handling and API proxying
5. Typed end-to-end (TypeScript or equivalent)

**Options considered:**

- **SvelteKit** — OpenWebUI's own lineage, so the quality bar and interaction
  patterns transfer directly; excellent streaming ergonomics; small bundles;
  SSR for the secret-holding layer. Strongest fit for the chat core.
- **Astro + the Doulos libraries** (`telos-llc/doulos`) — the owner's existing
  design system: `@telos-llc/components` (zero-JS presentational + design
  tokens) and `@telos-llc/integrations` (adapters + islands), one-way dependency
  discipline, already Astro-oriented. Real advantages: reuses owned IP,
  consistent design language across projects, and a genuinely fast static
  shell. **Honest caveat:** a chat client is not a content site — the chat view
  is one large stateful island, so Astro's islands model contributes shell,
  routing, and tokens rather than the interactive core. That is still
  worthwhile, but it should be chosen with eyes open rather than expecting the
  islands architecture to carry the chat itself.
- **React/Next** — deepest ecosystem, heaviest runtime; no specific advantage
  here.

**Recommendation:** adopt the **Doulos design tokens and component language
regardless of the core framework** — that is where the reuse genuinely pays,
and it keeps this client visually of a piece with the owner's other work. For
the chat core, SvelteKit is the closest fit to the requirements; Astro + Doulos
is a legitimate choice if design-system cohesion outweighs interaction
ergonomics. The requirements above are the contract; the implementing work owns
the decision.

---

## 15. Server asks (compactor changes this spec depends on)

| Ask | Priority | Why |
|---|---|---|
| **Auth on `/admin/*`** (extend PR #30's `COMPACTOR_API_KEY` beyond `/v1/*`) | **required** | otherwise every memory feature 403s from a separate container |
| **CORS configuration** | **required** | split origin |
| **Server-side turn sequence.** Persist `turn_seq` per `conv_id` (natural home: the summaries state file, beside `last_summarized_turn`), increment it in `_async_tail`, and replace `turn_index = len(messages) + 1` at [main.py:1202](compactor/main.py:1202) with `turn_seq + 1` | **required** | the compactor's only notion of conversational position is the client's message-array length. Under §4 rule 2 that is a constant, so for a spec-compliant client every exchange overwrites the same ChromaDB document ([retrieval.py:144](compactor/retrieval.py:144)), `recent_cutoff` stays ~0 and retrieval returns nothing ([main.py:1346](compactor/main.py:1346)), and `_needs_l1_rollup` is never true ([summarizer.py:187](compactor/summarizer.py:187)). **The memory architecture is inert against the client this spec describes until this lands.** |
| **Guard the destructive write.** When `turn_index` would regress below the stored high-water mark, allocate `turn_seq + 1` rather than upserting over the existing row | **required** | independent of any heuristic; this is what prevents a short window from overwriting an existing episodic row |
| **Received-context echo.** Return, as a response header or SSE preamble, the triple the compactor already logs at [main.py:1192](compactor/main.py:1192): resolved `conv_id`, resolution `source` (`header`\|`body_metadata`\|`hash`), and messages received | **required** | gives the client independent confirmation of what actually arrived, and powers `context_truncated` and `conv_id_fallback` from the server's view rather than the client's self-report. The compactor currently sets **no custom response headers anywhere**, so this is new surface — build it as the shared mechanism the budget-shed signal also uses |
| **Budget-shed signal** (response header or SSE event when hard-budget shedding, image stripping, or compaction removes content) | **required** *(was nice-to-have)* | `_enforce_hard_budget` sheds oldest turns, `_strip_image_parts` / `_apply_image_retention` remove content the user can still see in the thread, and none of it reaches her. Obligation §2.7 binds every layer that composes the context; leaving this optional exempts the compactor from the rule and leaves a live silent-truncation defect in production |
| **Task-traffic marker.** Accept and require an explicit request-kind marker, and skip the memory tail for requests with no prior assistant turn | **required** | `_async_tail` fires unconditionally on `if conv_id:`; a `msgs=1` background call therefore hashes to a stable conv_id and writes facts. `31365d633335bbd0` holds 105 facts and is still accruing as of 2026-08-24 |
| **Injected-memory endpoint** (what facts/RAG/summary went into a given turn) | **recommended** *(was nice-to-have)* | powers "What it was given" (§7) and the memory half of the context receipt (§12). The transcript half is the client's own payload and needs no server help |
| **Context-starvation warning.** Compare the arriving message count against the conversation's own high-water mark — `last_summarized_turn` and `max(added_turn)` over facts are already loaded in the same request — and emit a warning event when the client is far below it. If the client declares `X-History-Total`, the check becomes arithmetic rather than a heuristic | recommended | catches *any* client's truncation bug, including this one's. Two branches with different epistemic status: a declared total below the server high-water mark is a fact; an undeclared short window is a suspicion. Do not collapse them, and **never refuse on either** — a 4xx breaks first requests after a browser reset, restored backups, window migrations, and fork targets, and turns "the AI lost my history" into "the AI refuses to talk to me" |
| **Modality endpoint** (does the served model accept images?) | nice-to-have | lets the client disable upload *before* failure (§8) |
| **Pre-flight reject** for oversized requests instead of a vLLM 400 round-trip | recommended | *deliberately not raised to required.* The 400 headed the 2026-08-24 chain but damaged nothing by itself — the client's handling of it did. A client satisfying §4.1 takes a 400 with no structural effect. A real UX and efficiency ask, not a correctness dependency |

**Compactor logging changes** — not client-dependent, so not strictly asks, but
the difference between hours and minutes of diagnosis:

1. Move the per-request log line to *after* state load and join the fields
   already in the process: `conv_id=… source=hash msgs=7 nonsys=7 turn_index=8
   high_water=208 facts=… l1=… l2=… l3=…`. On 2026-08-24 the diagnosis was on
   screen, split across two adjacent lines nobody joined.
2. Add `lastturn=` to `log_parts`. `msgs=7` beside `lastturn=200` is
   self-evidently wrong; L1/L2/L3 counts alone are not.
3. Log the negative-delta case in `_needs_l1_rollup`. A negative delta is a
   different condition from "not enough new material" and currently looks
   identical to healthy quiet — the summarizer went dormant during the incident
   and said nothing.
4. Log conv_id resolution *inputs* on the hash path. There is no log at all when
   the header is simply absent, so a conv_id silently changing between turns is
   invisible.
5. Log the roles and first 40 characters of each message at the request log line.
   The incident narrative rests on a scalar; this turns the send set from an
   inference into a measurement.
6. Correct the `resolve_conv_id` docstring at [memory.py:8](compactor/memory.py:8):
   it documents two resolution paths; there are three, since `body_metadata` was
   added later. It is the first thing anyone reads when debugging conv_id.

---

## 16. Migration & rollout

1. **Run beside OpenWebUI**, on a separate port, against the same compactor —
   but **not on the same `conv_id`.** While OpenWebUI is sending a truncated
   window, the facts and summaries it writes are extracted from a starved context
   and land in the memory the new client will read; and its dual-write storage
   overwrites external state from stale browser state. Either fork per §4 rule 5
   for the new client, or treat any OpenWebUI-driven conversation as read-mostly.
2. **Import path.** OpenWebUI stores each chat **twice** — as a `chat.chat` JSON
   blob (`history.messages`, an id-keyed tree with `parentId`, plus a flattened
   `messages` array) and in a normalized `chat_message` table. "Tolerate broken
   trees" is the wrong verb; an importer that tolerates a 5-root tree by picking
   one chain reproduces the incident inside the migration. The importer:
   - reads from a **snapshot**, never the live database, with OpenWebUI stopped
     and every browser tab holding the chat closed. An open tab rewrites the blob
     from stale client state; a repair was applied and reverted twice on
     2026-08-24 for exactly this reason.
   - reads **both** representations and diffs them, reporting nodes present in
     one and not the other and nodes differing in parent, role, or content hash.
     On 2026-08-24 they agreed at `total=241 / deepest=208 / roots=5`; that
     agreement is a fact to verify, not to assume. Where they disagree, preserve
     both readings as sibling variants and record the conflict.
   - normalizes the **two key spaces**: `chat_message.id` is `{chat_id}-{uuid}`
     while `parent_id` is a bare uuid. Strip exactly the `chat_id + '-'` prefix
     and assert the remainder is a UUID. **Splitting on `-` is forbidden** —
     UUIDs contain hyphens, and a naive split is what produced the false "every
     message is orphaned" reading during the incident. Record
     `(source_table, source_id_raw, imported_id)` for every row; never join on
     the raw key.
   - derives imported ids as `uuidv5(ns, source_chat_id + ':' + bare_uuid)` so
     re-running is idempotent, and supports `--dry-run`.
   - **preserves every root.** Silently selecting the longest chain is a
     data-loss decision and must not be taken. Choose explicitly, with no
     default: `--split` makes each source root its own conversation with a
     recorded sibling-group id, or `--graft` parents the source roots under a
     synthetic `import_boundary` node that the UI renders as a visible import
     artifact — never as a message, never with a role, never attributed to the
     user or the assistant. Grafted branches must not be presented as one
     continuous conversation, and the model must not be handed a window that
     implies one.
   - **does not trust `currentId`.** On 2026-08-24 it pointed at an 8-message
     side branch. Carry it to the imported counterpart of that same node, do not
     relocate it, and present the choice on first open: "You are on a branch of 8
     messages. This conversation has 241 messages across 5 branches."
   - **imports unreachable nodes rather than dropping them**, attached to the
     boundary node with `parent_missing`. Guessing a parent fabricates history.
   - **emits a per-conversation manifest** — source counts from both
     representations, diff classes, roots, unparseable ids, parent-missing nodes,
     imported count, and the resulting audit numbers. Import succeeds only if
     every source row lands in exactly one outcome class and the §11.4 audit
     returns PASS; otherwise it exits non-zero leaving the target unmodified, and
     the source database stays read-only until cutover.
   - is covered by a fixture test built from a sanitized copy of the real
     241-message / 5-root / leaf-at-depth-8 conversation. **That data is the
     regression test for §4.1 and §11; it must not be discarded after migration.**
3. **Pre-migration audit.** Run the five-tuple (`total / current_leaf /
   chain_from_current / deepest / roots`) over every OpenWebUI conversation
   before migrating any of them, so the number of silently-forked conversations
   is known in advance rather than discovered afterwards. Five roots accumulated
   in one chat; there is no reason to assume it is the only one.
4. **Cutover criteria:** feature parity on §6; a real conversation driven
   end-to-end including voice and an image; memory panel verified against the
   admin API; and the tester's long-running conversation migrated **verifiably**
   — all 241 messages present, all branches preserved, the manifest matching the
   source, the §11.4 audit PASS, and the chain the user is shown confirmed by her
   rather than chosen by the importer.
5. **Own repo** (ARCHITECTURE Decision 7) once the client is standing on its
   own; this repo keeps the compactor and the contract.
6. **Retire OpenWebUI** only after a period of parallel running.

---

## 17. Open questions for the implementing work

1. **Client store**: IndexedDB vs server-side — decide against the Postgres
   sidecar timeline (§11).
2. **Window size**: what N of recent turns balances continuity against cost?
   Should it adapt to the model's context size?
3. **Branch UX**: how much regeneration-tree navigation is worth exposing?
4. **Offline behavior**: read-only access to local transcripts when the backend
   is unreachable — worth it?
5. **Memory panel depth**: how much is genuinely useful before it becomes the
   "window" §2.5 warns against?
6. **Multi-user timing**: schema-ready from day one, but when does it ship?
7. **Design-system fit**: how much of Doulos transfers to an interaction-dense
   client — worth a spike before committing.
8. **Repair or quarantine?** When the chain fails an integrity check, does the
   client fix it and tell the user, or refuse to guess and ask? §4.1 now says
   quarantine, on the grounds that a silent pointer move caused the incident. The
   cost is a user facing a branch-selection dialog she did not ask for.
9. **Refuse or warn?** On a pre-send fidelity mismatch, does the client block the
   request or send and warn? Refusing is honest but blocks the user; warning
   risks the warning being ignored, which is how 2026-08-24 played out in every
   other respect.
10. **Are multiple roots ever legitimate?** The synthetic-root design in §11.2
    makes `roots == 1` enforceable, but only if every legitimate branch has a
    parent. First-message edits, imports, and forks each need checking against
    that assumption before the unique index is committed to.
11. **How much receipt is right?** A persistent counter, an expandable panel, or
    on-anomaly only? Bounded by §2.5 and §2.7's own limit — the surface must stay
    diffable and never become interiority theatre.
12. **Window size (amends Q2).** Whatever N is, the client must be able to
    *prove* it sent N, so N must be recorded per request rather than inferred —
    and N interacts with the compactor's turn accounting (§15), which currently
    derives position from the client's message count.
13. **Tombstone semantics.** If a message is hidden rather than deleted, does the
    integrity check count it? §11.1 requires subtree-wide tombstoning to keep the
    answer consistent; whether users will accept "you cannot hide one message
    mid-thread" is unresolved.
14. **Shared conversations during parallel running.** §16.1 now forbids sharing a
    `conv_id` with OpenWebUI. Is a read-mostly mode good enough for the tester's
    live conversation during cutover, or does she need to fork and lose
    continuity with the OpenWebUI view?
15. **Unsettled measurement.** It is not yet established whether the 8-message
    branch ran in the conversation's real memory namespace or a fresh empty one.
    Hashing the chat's true original first user message and comparing it to
    `6aca8bcdf603d584` settles it, and it changes what the incident is called:
    context starvation with intact memory, or memory-namespace orphaning. Until
    it is run, the spec asserts the count (`msgs=7`) and the invariant, not the
    identity of the seven messages.

---

## Appendix: environment contract

| Variable | Purpose |
|---|---|
| `COMPACTOR_URL` | compactor base URL (front door) |
| `COMPACTOR_API_KEY` | bearer key when the compactor has auth enabled |
| `STT_URL` / `TTS_URL` | voice services (or relayed server-side) |
| `SESSION_SECRET` | session signing — **rotating it logs everyone out; treat as durable config** |

Backend-side variables this client's behavior depends on:
`COMPACTOR_KEEP_RECENT_TURNS`, `COMPACTOR_MAX_RETAINED_IMAGES`,
`COMPACTOR_IMAGE_TOKENS` (a **fallback** since v3.0.5, not the source of truth —
the client must not mirror it as authoritative), `MAX_MODEL_LEN`, `STT_ENABLED`,
`TTS_ENABLED`.
