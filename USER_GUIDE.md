# User Guide

How to actually *use* Zion's Light AI — what it remembers, how to steer
that memory, and how to inspect or back it up. If you're deploying it, see
[RUNPOD_DEPLOY.md](RUNPOD_DEPLOY.md) instead; this guide assumes it's
already running and you're chatting with it.

---

## What makes it different: it remembers

A plain LLM chat forgets everything older than its context window. Hold a
long conversation — a novel, a campaign, a research thread — and eventually
early details fall off the edge: a character's name, a rule you set, a
decision you made twenty turns ago.

Zion's Light AI puts a **memory middleware** between you and the model. As
you chat, it quietly maintains a per-conversation memory and feeds the
relevant parts back to the model on every turn. You don't have to do
anything — it works by default. But you *can* inspect and steer it, which
is what most of this guide is about.

### The four memory layers

Each conversation accumulates four kinds of memory, all stored on disk and
surviving restarts:

| Layer | What it captures | How it's used |
|---|---|---|
| **Facts** | Durable claims — names, preferences, world details, decisions | Re-injected every turn so the model stays consistent |
| **Episodic (RAG)** | Every exchange, embedded for semantic search | When your new message resembles an old turn, that old turn is retrieved and shown to the model |
| **Summaries** | Rolling compression of older turns (L1→L2→L3) | Keeps very long conversations coherent without blowing the context budget |
| **Persona** | A long role/voice system prompt | Kept verbatim and never summarized away or evicted |

Two background processes keep memory clean:
- **Deduplication** — if the model captures the same fact three different
  ways ("Lyra is half-elf" / "Lyra is half elven" / "the protagonist is
  half-elven"), they get merged into one. You never see duplicates.
- **Archival** — facts you haven't touched in a long time (default 90
  days) move to cold storage. They're not deleted — you can restore them —
  they just stop taking up the active context budget.

### Conversations are kept separate

Memory is **per-conversation**. Facts from your sci-fi novel never leak
into your fantasy campaign. Each chat in OpenWebUI is its own memory space,
identified automatically.

---

## Slash commands

Type these as a normal chat message. The assistant handles them instantly —
they never go to the model, so they cost nothing and reply immediately.

| Command | Aliases | What it does |
|---|---|---|
| `/help` | `/?` | List the available commands |
| `/list-facts` | `/facts` | Show everything currently remembered for this conversation |
| `/list-archive` | `/archive` | Show facts that have been moved to cold storage |
| `/remember <text>` | | Manually add a fact (max 500 characters) |
| `/forget` | | **Wipe all memory** for this conversation — facts, episodic, summaries, persona |
| `/forget <substring>` | | Remove only facts whose text contains the substring (case-insensitive) |
| `/why` | `/why-did-you-say-that` | Show what's in memory right now — your window into what the model is being told |

### Examples

```
/remember The protagonist is left-handed and afraid of deep water
```
> Remembered: 'The protagonist is left-handed and afraid of deep water'
> Facts now: 7

```
/list-facts
```
> Current facts (7):
>   - Lyra Threadweaver is a half-elf ranger from Aethermere
>   - The protagonist is left-handed and afraid of deep water
>   - ...

```
/forget deep water
```
> Forgot 1 fact(s) matching 'deep water'. 6 remaining.

```
/why
```
> Memory state for this conversation:
>   Facts (6): ...
>   Summary stack: L1=2 L2=0 L3=no
>   Indexed exchanges (episodic): 34
>   Persona (412 chars): You are a hardboiled noir detective...

> **Tip:** a message that just *starts* with a slash but isn't a real
> command — like `/usr/local/bin is my path` — is passed through to the
> model normally. Only the commands above are intercepted.

---

## Personas

A **persona** is a long system prompt that defines the model's role, voice,
or character — "You are a hardboiled noir detective named Sam Cole who
speaks in terse first-person past tense…", a character bible, a world
setting. Personas get special treatment: they're stored as their own memory
layer, injected on every turn, and **never** summarized away or evicted on
budget pressure.

### Setting a persona

**The easy way — just use it.** In OpenWebUI, set a system prompt for the
chat (via the model's system-prompt field or a Model preset). If it's long
enough (≥ 200 characters by default), it's automatically detected and stored
as that conversation's persona on the first message.

**The explicit way — admin endpoint** (see below):
```bash
curl -X POST http://localhost:8080/admin/conversations/<conv_id>/persona \
  -H 'Content-Type: application/json' \
  -d '{"text": "You are Captain Salt, a pirate of the high seas..."}'
```

### Reusing a persona across conversations

Built up a great persona in one conversation and want a fresh chat that
starts with the same character?

```bash
curl -X POST http://localhost:8080/admin/conversations/<new_conv_id>/inherit-persona \
  -H 'Content-Type: application/json' \
  -d '{"source_conv_id": "<conv_with_the_persona>"}'
```

Browse all stored personas with `GET /admin/personas`.

---

## Power-user: admin endpoints

Beyond slash commands, the compactor exposes admin endpoints for deeper
inspection, backup, and maintenance.

> **Security:** admin endpoints are bound to **localhost only** by default
> (`COMPACTOR_ADMIN_BIND=127.0.0.1`) and have no authentication. Reach them
> from inside the pod (RunPod Web Terminal, or `curl localhost:8080/...`),
> **not** over the public proxy URL. Only expose them externally if you've
> put your own auth/firewall in front — see [RUNPOD_DEPLOY.md](RUNPOD_DEPLOY.md).

First, find your conversation's ID:
```bash
curl -s http://localhost:8080/admin/conversations | jq
```

### Inspect

| Endpoint | What you get |
|---|---|
| `GET /admin/conversations` | List every conversation with stored memory |
| `GET /admin/conversations/<id>` | Counts: facts, indexed exchanges, summary stack, persona presence |
| `GET /admin/conversations/<id>/facts` | The full facts list |
| `GET /admin/conversations/<id>/summary` | The raw L1/L2/L3 summary state |
| `GET /admin/conversations/<id>/archive` | Cold-storage (archived) facts |
| `GET /admin/personas` | The persona library across all conversations |
| `GET /admin/conversations/<id>/persona` | One conversation's full persona text |

### Maintain

```bash
# Merge near-duplicate facts (embedding + LLM verification)
curl -X POST http://localhost:8080/admin/conversations/<id>/dedup

# Archive facts unused for >30 days
curl -X POST "http://localhost:8080/admin/conversations/<id>/archive?older_than_days=30"

# Restore archived facts (all, or matching a substring)
curl -X POST http://localhost:8080/admin/conversations/<id>/restore \
  -H 'Content-Type: application/json' -d '{}'                       # all
curl -X POST http://localhost:8080/admin/conversations/<id>/restore \
  -H 'Content-Type: application/json' -d '{"text_substring": "Lyra"}'
```

### Back up, move, or fork a conversation

```bash
# Export everything (facts + summaries + episodic) to one JSON bundle
curl -s http://localhost:8080/admin/conversations/<id>/export > my-novel.json

# Restore it (on this pod or another) — refuses to overwrite unless told
curl -X POST http://localhost:8080/admin/conversations/import \
  -H 'Content-Type: application/json' \
  -d "{\"bundle\": $(cat my-novel.json), \"target_conv_id\": \"<id>\", \"overwrite\": true}"

# Fork — clone a conversation's memory into a new one to explore an
# alternate direction without disturbing the original
curl -X POST http://localhost:8080/admin/conversations/<id>/fork
```

> **Note:** export bundles do **not** currently include the persona layer.
> After importing into a fresh conversation, re-set the persona with
> `POST .../persona` if you need it.

### Reset memory

```bash
# Full wipe for one conversation (same as the /forget chat command)
curl -X DELETE http://localhost:8080/admin/conversations/<id>/facts
```

---

## Health & "is it working?"

```bash
# Deep health probe — vLLM reachable + storage writable + memory stats
curl -s http://localhost:8080/health/full | jq

# On-demand self-test — runs a real chat round-trip + facts read/write
curl -s http://localhost:8080/admin/selftest | jq

# Boot self-test result (runs automatically on pod start)
cat /var/log/supervisor/selftest.log
```

A healthy deploy reports `"status": "ok"` from `/health/full` and
`"status": "pass"` from `/admin/selftest`.

---

## FAQ

**The model forgot something I told it earlier. Why?**
Run `/why` to see what's actually in memory. A few possibilities: the fact
was never extracted (try `/remember <it>` to add it manually); it was
archived as stale (`/list-archive`, then restore); or it's in episodic
memory but your recent message wasn't similar enough to retrieve it.

**How do I make it remember something important for sure?**
`/remember <the fact>`. Manual facts are treated exactly like extracted
ones but you control the wording.

**How do I start fresh without losing my other conversations?**
`/forget` wipes only the current conversation. Other chats are untouched.

**It's repeating the same fact in slightly different words.**
That's what dedup is for — it runs automatically, but you can force a pass
with `POST /admin/conversations/<id>/dedup`.

**Can I move a conversation to a different pod?**
Yes — `export` it to a JSON bundle, then `import` it on the other pod. The
weights live on the Network Volume; the memory bundle is portable text.

**Does memory survive a pod restart / termination?**
Yes. All memory lives on the Network Volume at
`/data/openwebui/compactor/`, which persists independently of the pod.

**Is any of this sent to a third party?**
No. Everything — inference, embeddings, memory — runs inside your own pod.
Embeddings are computed locally with a prebaked ONNX model.

**Can it see images? (V3.1)**
Only if it's running a *vision* model. When the operator has set a
vision-language model (see [RUNPOD_DEPLOY.md](RUNPOD_DEPLOY.md#vision-v31--enabling-image-understanding)),
you can upload an image in OpenWebUI and ask about it — "what's in this
photo?", "read this receipt", "critique this mockup." The assistant also
*remembers* images sensibly: an image you shared earlier is kept intact as
the conversation grows (it isn't summarized away), so you can refer back to
it many turns later. With a text-only model, image upload won't work —
that's a deployment choice, not a bug.

**Can it hear me? (V3.2)**
Yes — if the deployment includes the speech service (it's on by default). Click
the microphone in OpenWebUI, speak, and your words are transcribed and sent as
your message — handy for long dictation or hands-free use. Transcription runs
locally on a bundled Whisper model (nothing leaves your pod); the text then
flows through memory exactly like anything you type. Accuracy depends on the
Whisper model the operator chose — the default is fast and good for clear
speech, and larger models are more accurate (see
[RUNPOD_DEPLOY.md](RUNPOD_DEPLOY.md#speech-to-text-v32--voice-input)). Voice can
be turned off per-deployment.

**Can it talk back? (V3.3)**
Yes — if the deployment includes the speech-output service (on by default). Use
OpenWebUI's "read aloud" control on a reply and the assistant speaks it, in a
local neural voice (Piper) with nothing leaving your pod. It's independent of
the memory pipeline — just the spoken rendering of the reply it already wrote.
The voice, and whether it's on, are deployment choices (see
[RUNPOD_DEPLOY.md](RUNPOD_DEPLOY.md#text-to-speech-v33--voice-output)).
