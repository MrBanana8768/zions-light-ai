# Compactor v2 — Memory architecture for sustained conversations

> Status: design proposal. Not yet implemented. v1 (the current `main.py`)
> stays in production until v2 is ready to swap in.

## Why v2

v1 does one thing: when a request exceeds `COMPACTOR_TARGET_TOKENS`, it asks
the LLM to summarize older turns into a single system block and replaces them.
That works — it prevents the hard context wall — but has three drift modes:

1. **Summary-of-summary degradation.** Long conversations compact many times.
   Each compaction summarizes an already-summarized summary. Specific facts
   bleed into vague paraphrases. By turn 200 the model can't reliably recall
   that "the protagonist's name is Lyra" if it was mentioned in turn 5.
2. **No exact recall.** Even at a single compaction, the original text is
   gone. If the user asks "what was the exact phrasing you used for the
   tavern scene?" — the summary contains a paraphrase at best.
3. **No stable identity.** User preferences stated in turn 10 ("write in
   third-person past tense, never use the word 'suddenly'") get summarized,
   then re-summarized, until they're either lost or distorted.

v2 addresses these with the architecture production systems (Claude, ChatGPT,
MemGPT, Letta, Mem0) converged on: **retrieval + extracted facts + tiered
summarization**, layered together.

## Architecture overview

```
┌────────────────────────────────────────────────────────────┐
│  Per-request context construction:                         │
│                                                            │
│  1. [original system prompt]                               │
│  2. [persistent facts memory] ───── extracted, never lost  │
│  3. [retrieved relevant past turns] ── RAG, top-K by sim   │
│  4. [hierarchical summary]      ───── tiered, older=denser │
│  5. [last N raw turns from request]                        │
│  6. [latest user message]                                  │
└────────────────────────────────────────────────────────────┘
```

Three independent memory subsystems, each addressing a different drift mode:

| Layer | Mechanism | What it preserves |
|---|---|---|
| **Episodic** (RAG over history) | Every turn embedded → ChromaDB → top-K retrieval per query | Exact text of relevant past moments |
| **Semantic** (facts memory) | LLM-extracted bullets, appended to a JSON file | Stable facts, names, preferences, decisions |
| **Working** (hierarchical summary) | Multi-level summaries: 20 turns → chunk summary → chapter summary → conversation theme | Smooth narrative continuity at varying resolution |

## Storage layout

All v2 state lives on the existing `/data` Network Volume so it survives pod
restarts automatically. No new volume needed.

```
/data/
├── models/                         # vLLM weights cache (unchanged from v1)
└── openwebui/
    ├── webui.db                    # OpenWebUI SQLite (unchanged)
    ├── uploads/                    # OpenWebUI uploads (unchanged)
    └── compactor/                  # NEW — v2 state
        ├── chromadb/               # ChromaDB persistent store
        │   └── <collection files>
        ├── facts/
        │   └── <conv_id>.json      # one file per conversation
        └── summaries/
            └── <conv_id>.json      # hierarchical summary state per conversation
```

Estimated growth: ~1 MB per 100-turn conversation (embeddings dominate).
A 200 GB volume holds tens of thousands of conversations.

## Conversation identification

**The fundamental constraint:** the OpenAI-compatible `/v1/chat/completions`
spec doesn't include a conversation ID. OpenWebUI manages conversations
client-side and just resends the full message history each turn.

**Strategy:** derive a stable conversation ID by hashing the conversation's
*opening fingerprint*:

```python
def conv_id(messages: list[dict]) -> str:
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    first_user = next((m["content"] for m in messages if m["role"] == "user"), "")
    fingerprint = f"{system}|||{first_user[:512]}"
    return hashlib.sha256(fingerprint.encode()).hexdigest()[:16]
```

The opening doesn't change as the conversation grows, so the ID is stable for
the whole life of one conversation. Two different conversations that happen to
start identically would collide, but the 512-char window over the first user
message makes that extremely unlikely in practice.

**Future improvement:** OpenWebUI supports custom request headers via its
function/filter system. A small OpenWebUI filter could inject
`X-Conversation-Id` from its internal conversation primary key, eliminating
the hash heuristic. Worth doing if we hit collision issues.

## Per-request flow

```
                      ┌──────────────┐
        request ─────▶│  compactor   │
                      │              │
                      │  1. compute conv_id from opening fingerprint
                      │  2. count tokens
                      │  3. if under budget: forward as-is, EXIT
                      │
                      │  4. load facts[conv_id], summary_state[conv_id]
                      │  5. embed last user message
                      │  6. query ChromaDB → top-K relevant prior turns
                      │  7. build new message list (the 6 layers above)
                      │  8. forward to vLLM, stream response back
                      │
                      │  9. (async, after response complete)
                      │     a. embed new user+assistant exchange → ChromaDB
                      │     b. call LLM with fact-extraction prompt → update facts[conv_id]
                      │     c. check summary thresholds, roll up if needed
                      └──────────────┘
```

The async tail (step 9) is critical — it must not block the user-visible
response. Implemented as `asyncio.create_task()` after the stream closes.

## Layer details

### Layer 2: Persistent facts memory

**Trigger:** after every completed assistant response.

**Mechanism:** one extra LLM call with a fact-extraction prompt:

```
Given the latest exchange, identify any persistent facts worth remembering
about the user's preferences, the world/setting being discussed, named
entities (people, places, projects), or decisions made.

Output as terse bullets. If nothing memorable, output exactly: NONE.

Latest exchange:
[user]: <text>
[assistant]: <text>

Existing facts (for de-dup):
- ...
```

**Storage:** `facts/<conv_id>.json`:
```json
{
  "conv_id": "abc123...",
  "updated_at": "2026-05-27T14:30:00Z",
  "facts": [
    { "fact": "Protagonist is named Lyra, age 23, half-elf ranger.", "added_turn": 5 },
    { "fact": "User prefers third-person past tense.", "added_turn": 12 },
    { "fact": "Setting: low-magic medieval kingdom called Aethermere.", "added_turn": 3 }
  ]
}
```

**Injection:** facts are concatenated into a system message inserted right
after the original system prompt. Token budget: cap at ~1500 tokens
(approximately 100-150 bullets). Older facts get pruned via LRU + the LLM is
asked periodically to consolidate redundant facts.

**Cost:** ~300ms + ~200 output tokens per turn. Async, so user-invisible.

### Layer 3: Retrieved relevant past turns (RAG)

**On indexing (after each exchange):**
1. Take the user message and assistant response as one "turn document"
2. Embed with sentence-transformers (default: `BAAI/bge-small-en-v1.5`)
3. Store in ChromaDB collection scoped to this conv_id, with metadata:
   `{ "turn_index": N, "role_pair": "user_assistant", "timestamp": ... }`

**On query:**
1. Embed the latest user message
2. Query ChromaDB: `collection.query(query_embeddings=[q], n_results=K)`
   where `K` is `COMPACTOR_RAG_TOP_K` (default 5)
3. Filter results: drop any turn already present in the recent N turns from
   the request (those are already in context verbatim — don't double-include)
4. Format as a system block:
   ```
   [Relevant earlier exchanges, retrieved by topical similarity]
   --- Turn 47 ---
   [user]: <text>
   [assistant]: <text>
   --- Turn 89 ---
   ...
   ```

**Embedding model choice:**
- **Default:** `BAAI/bge-small-en-v1.5` (33M params, ~130 MB, top of MTEB
  retrieval benchmarks for size class, runs at ~10ms/embedding on CPU)
- **Multilingual alternative:** `BAAI/bge-m3` (568M params, ~1.2 GB,
  100+ languages — only if needed)
- **Tiny alternative:** `sentence-transformers/all-MiniLM-L6-v2` (22M params,
  ~80 MB, older but still solid)

Loaded once at compactor startup, lives in process memory (~500 MB RAM
including PyTorch overhead). CPU inference is plenty fast for chat-rate.

**Why not run the embedding model in vLLM?** vLLM supports embedding models,
but loading one alongside the chat model competes for VRAM and adds
serialization overhead on every query. CPU embedding in the compactor process
is simpler, faster (no network hop), and frees VRAM for the chat model.

### Layer 4: Hierarchical summarization

**Replace v1's single-level flat summary with tiered summaries.**

Three levels, each with a token budget and a "roll up" threshold:

| Level | Holds | Token budget | Roll-up trigger |
|---|---|---|---|
| **L0 — Recent turns** | Last 20 raw turns | unbounded | After 20 turns, oldest gets promoted to L1 |
| **L1 — Chunk summaries** | Summaries of 20-turn chunks | 500 tokens each | After 10 chunks (~200 turns), oldest L1s merge into L2 |
| **L2 — Chapter summaries** | Summaries of L1 batches | 1200 tokens each | After 5 chapters (~1000 turns), merge into L3 |
| **L3 — Conversation theme** | High-level summary of the whole conversation | 2000 tokens total | Always present once L2 fills |

**On each request when over budget**, inject:
- L3 (if exists) — "the conversation overall"
- Latest L2 — "recent chapters"
- L1s not yet rolled into L2 — "recent narrative chunks"
- L0 — last 20 turns from request (verbatim)

**Storage:** `summaries/<conv_id>.json`:
```json
{
  "conv_id": "abc123",
  "l3": { "text": "...", "turns_covered": [1, 850] },
  "l2": [
    { "text": "...", "turns_covered": [600, 850] }
  ],
  "l1": [
    { "text": "...", "turns_covered": [820, 840] },
    { "text": "...", "turns_covered": [840, 860] }
  ],
  "last_l0_turn": 870
}
```

**Total context cost** of the summary stack: ~5000 tokens worst case (L3 +
1 L2 + 2-5 L1 + 20 raw turns). Predictable, doesn't grow with conversation
length.

## Request flow — concrete pseudocode

```python
async def chat_completions(body):
    messages = body["messages"]
    conv_id = compute_conv_id(messages)
    current_tokens = count_tokens(messages)

    if current_tokens <= TARGET_TOKENS:
        # Under budget — forward unmodified. Still index async for later.
        response = await forward_to_vllm(body)
        asyncio.create_task(post_exchange_indexing(conv_id, messages, response))
        return response

    # Over budget — build enriched context
    facts = await load_facts(conv_id)
    summary_state = await load_summary_state(conv_id)
    last_user_msg = next(m["content"] for m in reversed(messages) if m["role"] == "user")
    retrieved = await chroma_query(conv_id, last_user_msg, k=RAG_TOP_K)

    # Drop retrieved turns that are already in the recent N
    recent_turn_texts = set(m["content"] for m in messages[-KEEP_RECENT_TURNS*2:])
    retrieved = [r for r in retrieved if r.text not in recent_turn_texts]

    new_messages = (
        [m for m in messages if m["role"] == "system"]                         # 1. original system
        + [{"role": "system", "content": format_facts(facts)}]                 # 2. facts
        + [{"role": "system", "content": format_retrieved(retrieved)}]         # 3. RAG
        + [{"role": "system", "content": format_summary_stack(summary_state)}] # 4. summary
        + messages[-KEEP_RECENT_TURNS*2:]                                      # 5+6. recent + new
    )

    body["messages"] = new_messages
    response = await forward_to_vllm(body)
    asyncio.create_task(post_exchange_indexing(conv_id, messages, response))
    return response


async def post_exchange_indexing(conv_id, messages, response):
    # Extract the new exchange
    new_user = messages[-1]["content"]
    new_assistant = extract_assistant_text(response)

    # Embed and store in ChromaDB
    await chroma_add(conv_id, turn_text=f"[user]: {new_user}\n[assistant]: {new_assistant}",
                     turn_index=summary_state.last_l0_turn + 1)

    # Fact extraction
    new_facts = await extract_facts_via_llm(new_user, new_assistant, existing_facts)
    if new_facts:
        await append_facts(conv_id, new_facts)

    # Roll up hierarchy if thresholds crossed
    await maybe_rollup_summaries(conv_id)
```

## New dependencies

| Package | Why | Size |
|---|---|---|
| `chromadb` | Persistent vector store, in-process | ~50 MB |
| `sentence-transformers` | Embedding model loader | ~100 MB |
| `BAAI/bge-small-en-v1.5` weights | Default embedding model | ~130 MB (pre-downloaded at build) |

Total image growth: **~280 MB** added to current ~14 GB image (~2% increase).

## File changes

### New files
- `compactor/memory.py` — facts and ChromaDB management
- `compactor/summarizer.py` — hierarchical summary roll-up logic
- `compactor/test_memory.py` — smoke tests for the new subsystems

### Modified files
- `compactor/main.py` — request flow becomes the pseudocode above; existing
  v1 functions (`split_messages`, `summarize`) are kept and reused by
  `summarizer.py`
- `compactor/requirements.txt` — add `chromadb`, `sentence-transformers`
- `Dockerfile` — pre-download embedding model weights at build time so first
  run is fast:
  ```
  RUN /opt/vllm-venv/bin/python -c \
      "from sentence_transformers import SentenceTransformer; \
       SentenceTransformer('BAAI/bge-small-en-v1.5').save('/opt/embeddings/bge-small-en-v1.5')"
  ```
- `.env.example` — new vars:
  ```
  COMPACTOR_RAG_TOP_K=5
  COMPACTOR_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
  COMPACTOR_FACTS_EXTRACTION=true   # Set false to skip fact extraction
  COMPACTOR_MAX_FACTS_TOKENS=1500
  COMPACTOR_L1_CHUNK_SIZE=20        # Turns per L1 summary
  COMPACTOR_L2_CHUNK_SIZE=10        # L1s per L2 summary
  ```
- `RUNPOD_DEPLOY.md` — note the additional ~1 MB/100-turn growth on the
  data volume and the new admin endpoints

### Unchanged
- `supervisord.conf` — no new processes; everything runs in the compactor
- vLLM, OpenWebUI, entrypoint.sh

## Phased rollout

Build incrementally — each phase is independently shippable and useful:

**Phase 1 — Conversation persistence scaffolding (1 day)**
- Compute `conv_id`, create `compactor/` storage layout, write per-conversation
  state file (no behavior change yet, just observability)
- Add `/admin/conversations` endpoint listing seen conversations
- Verifies the conv_id heuristic before building real memory on top

**Phase 2 — Facts memory (2-3 days)**
- Implement fact extraction + storage + injection
- Highest value-per-LOC of the three layers
- User-visible: model stops forgetting names, settings, preferences

**Phase 3 — RAG over history (3-4 days)**
- Add ChromaDB + embedding model
- Image grows ~280 MB
- User-visible: exact prior text becomes retrievable

**Phase 4 — Hierarchical summarization (2-3 days)**
- Replace v1's flat summary with tiered
- User-visible: narrative consistency across hundreds of turns

Each phase ships as its own image tag (`zions-light-ai:v2-phase2`, etc.) so
production usage can validate before the next layer adds risk.

## Performance budget

Per request (when over `TARGET_TOKENS`):
- Embedding the query: ~10 ms
- ChromaDB query: ~5 ms
- Loading facts + summaries from disk: ~5 ms
- Building enriched message list: <1 ms
- **Total compactor overhead: ~20 ms** (vs ~3000 ms for the actual LLM
  inference — negligible)

Per request (async tail, runs after response):
- Embedding new exchange: ~10 ms
- ChromaDB insert: ~5 ms
- Fact extraction LLM call: ~300 ms
- Summary roll-up (when triggered): ~3000 ms for one summarization call,
  triggered at most once per 20 turns

The async tail uses one vLLM slot. At single-user chat rate this is
invisible. At multi-user load, fact extraction batching becomes a worthwhile
optimization (not in scope for v2).

## Open questions to confirm before implementation

1. **Embedding model:** BGE-small (default, English-tuned) vs BGE-M3
   (multilingual, 10× bigger). Default assumes English creative writing.
2. **Fact extraction frequency:** every turn, or only when over budget?
   Every-turn is more thorough; only-when-over-budget cuts inference cost
   roughly in half during early conversation.
3. **L1 chunk size:** 20 turns is the default. Lower = more granular memory,
   more storage. Higher = less granular, faster roll-ups.
4. **Admin endpoints:** expose `/admin/conversations/<id>/facts` for
   inspection? Useful for debugging, slight security concern if pod is
   public-facing.
5. **Memory clearing:** add a `/admin/conversations/<id>/forget` endpoint?
   Useful for "the model is stuck on a wrong fact" scenarios.
6. **Migration path:** v1 conversations have no v2 state. Should v2 build
   facts/summary retroactively from existing v1 chat history on first
   encounter? Adds startup latency for old conversations; pure forward-only
   is simpler.

## Verification

Each phase ships with:
- Unit tests in `compactor/test_*.py` exercising the new modules in CPU-only
  containers (same pattern as v1's `test_smoke.py`)
- A "long conversation simulation" integration test: scripted 300-turn
  conversation against a small model, asserting that facts injected at turn
  5 are still respected at turn 250 (would fail under v1 today)
- Manual verification on RunPod with the Magnum 22B production setup

## What this does NOT solve

- **Cross-conversation memory.** Each conv_id is isolated. The model won't
  remember the user across separate conversations. Adding cross-conversation
  user memory (à la ChatGPT's "memory" feature) is a v3 feature.
- **Multi-user isolation.** v2 assumes single-tenant or that users share
  trust. If OpenWebUI ever runs multi-user with separate accounts, the
  conv_id hash needs to incorporate user identity to prevent leakage.
- **Voice / vision modalities.** Pure-text v2. Extending to multimodal would
  require multimodal embeddings and adjusted fact extraction.
