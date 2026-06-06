# V4 — Agentic tool use (design spec)

Status: **design / not started.** This captures the architecture decisions
for giving the model tool-use and eventual command-line capability, and the
question of when a purpose-built agent harness becomes necessary. Sequenced
after V3 (multimodal); the memory work in V2 is its foundation.

> Companion to [V2_PLAN.md](V2_PLAN.md). Where V2 gave the *model* memory,
> V4 gives it *agency* — the ability to take actions, not just produce text.

---

## The core insight

**vLLM parses tool calls; it does not execute them.** With a tool-capable
chat template (Mistral/Magnum has one) and the flags
`--enable-auto-tool-choice --tool-call-parser mistral`, vLLM will emit
OpenAI-style `tool_calls` in its response when the model decides to call a
tool. But it stops there — *something downstream must run the tool and feed
the result back*.

In our architecture there is exactly one correct home for that orchestration:
**the compactor.** It already intercepts every `/v1/chat/completions`,
buffers and replays responses, and runs post-response async work. A tool
loop is a natural extension of the same interception point — no new service,
no new network hop.

```
client → compactor → vLLM
                ↓ vLLM returns tool_calls instead of a final answer
        compactor executes the requested tool(s)
                ↓ appends {role:"tool", tool_call_id, content} for each
        compactor re-calls vLLM with the augmented message list
                ↓ loop until vLLM returns a normal answer OR a step cap hits
client ← compactor (final answer; optionally a step transcript)
```

This is the standard ReAct / OpenAI tool-calling loop. It lives in a new
`compactor/tools.py` (a registry + executors) plus the loop body in
`main.py`, bounded by a hard `MAX_TOOL_STEPS` cap so a misbehaving model
can't loop forever.

---

## Three tiers of tool

Build in this order — each tier is independently shippable and strictly
riskier than the last.

### Tier A — pure-Python tools (safe, ship first)
In-process Python callables with no shell, no filesystem-mutation, no
untrusted I/O:
- query this conversation's own memory (facts / RAG / summaries)
- arithmetic / unit conversion / date math
- an **allowlisted** outbound HTTP fetch (specific domains only)

No sandbox needed — the blast radius is the compactor process, and the
tools don't do anything dangerous. This tier proves the loop end-to-end and
is genuinely useful on its own (a model that can search its own memory on
demand is more capable than one that only gets a fixed injection).

### Tier B — OpenWebUI Tools/Functions (frontend-registered)
OpenWebUI already has a Tools/Functions system. Useful for UI-triggered
actions and user-authored helpers. Downside: coupled to the UI, doesn't
give *API* callers tool use, and doesn't give the model autonomous access —
it's user-in-the-loop by design. Worth supporting for the UI path but not
the core of V4.

### Tier C — command-line execution (the dangerous one)
Letting the model run shell commands. This is what "command-line level
tooling" really means, and it is **non-negotiably gated on a sandbox** (next
section). Never ship Tier C without it.

---

## The sandbox boundary (the hard part of Tier C)

The compactor runs in a pod that mounts the Network Volume — model weights
**and every conversation's memory**. An unsandboxed shell tool turns prompt
injection into remote code execution against all of that. So Tier C must
never `subprocess` directly. It shells into an **isolated runner** with:

- **No mount of `/data`** (or any host path that matters)
- **No network** unless explicitly allowlisted
- **CPU / memory / wall-clock caps**
- **A command allowlist** (start strict; widen deliberately)
- **Ephemeral filesystem** — fresh per call, destroyed after

Implementation options, cheapest→strongest:
1. A throwaway Docker container per call (`docker run --rm --network none
   --read-only …`) — simple, decent isolation, needs docker-in-pod.
2. `nsjail` / `bubblewrap` namespaced sandbox — lighter than a container,
   strong isolation, no docker dependency.
3. `gVisor` / Firecracker microVM — strongest, heaviest. Overkill until
   scale demands it.

Plus, regardless of mechanism:
- **Human-in-the-loop approval** for any mutating command, at least
  initially. The run pauses, surfaces the proposed command, waits for a
  yes/no. (This is a *harness* capability — see below.)
- **Every tool call logged** to a durable run record for audit + cost.

This is precisely where the V2.3 "quality over speed — failure-tested before
shipped" stance is load-bearing. A sandbox escape here is catastrophic, not
cosmetic.

---

## Memory is the agent's substrate

We didn't build V2 for agents, but it turns out to be exactly what an agent
needs: facts, RAG, summaries, and personas give the agent **persistent
context across tool-using sessions** without re-deriving everything each
run. An agent that remembers what it learned and did last time is the whole
point. V4 inherits this for free — a tool-using run reads and writes the
same per-conversation memory the chat path uses.

---

## The harness question: when do we build our own?

A "harness" here means a purpose-built **agent-run engine** — more than the
current OpenWebUI + compactor + vLLM + supervisord glue. The honest answer:
**not for the first agentic step, unavoidably for the second.**

### What the current architecture can and can't do

| Current piece | Good at | Breaks for agents when… |
|---|---|---|
| OpenWebUI | request→response chat UI | you need an agent-run transcript, tool-call cards, approval gates, step replay |
| compactor (FastAPI) | per-request interception + per-conv memory | a run must outlive one HTTP request — background loops, scheduling, cancellation, retries |
| supervisord | keep 3 services alive | you need to spin ephemeral sandboxes per call and track concurrent runs |
| JSON-on-volume memory | per-conversation facts/RAG/persona | you need a queryable, replayable *run log* — every step, tool call, observation, cost, timing |

### Phase 1 needs no harness
The bounded, single-request tool loop (Tier A, maybe a read-only Tier C)
returns within one request/response. OpenWebUI and the compactor handle it
as-is. **Ship this first** on the existing stack.

### Phase 2 forces the harness
You are forced to build a run engine the moment *any* of these become
requirements:
1. **Runs outlive a request** — autonomous loops, scheduled agents, "go work
   on this for ten minutes." FastAPI handlers can't own that lifecycle.
2. **Human-in-the-loop pause/resume** — approval mid-run needs a state
   machine that can suspend and be resumed; request/response can't express it.
3. **Real shell execution at scale** — a sandbox *pool* is itself
   orchestration supervisord won't do.
4. **Multi-agent / sub-agents** — message passing, a run graph.
5. **Cross-run observability & cost control** — a run/event store + a console.

When that day comes, the harness is roughly: a **durable run/event store**,
a **sandbox-runner pool**, an **approval/state machine**, and an **agent
console** (the point where OpenWebUI stops sufficing and a dedicated UI is
warranted).

### Build vs adopt
Don't hand-roll everything. Use off-the-shelf pieces where they fit —
`nsjail`/`gVisor` for sandboxing, an existing job queue for run scheduling —
and keep the *custom* part to what's genuinely ours: the run-store, the
approval gates, and the integration with our memory layer. The project's
self-hosted ethos argues for owning the orchestration glue, not for
reinventing a container sandbox.

### Recommendation
Build Phase 1 on the current architecture. Let real usage reveal which of
the five triggers bites first, and build the harness around *that* trigger
rather than speculatively. Same discipline as the rest of the project:
prove the need, then build it failure-tested.

---

## Phased rollout (proposed)

| Phase | Scope | Harness? |
|---|---|---|
| **V4.0** | Tool loop in compactor + Tier-A pure-Python tools (memory query, math, allowlisted fetch). `MAX_TOOL_STEPS` cap. Tier-1 + Tier-3 tests. | No |
| **V4.1** | Read-only Tier-C commands behind a sandbox runner (Docker/nsjail), no mutation, no `/data` mount. | Sandbox only |
| **V4.2** | Mutating commands + human-in-the-loop approval → first real run-state machine. | **Yes — harness begins** |
| **V4.3+** | Durable run store, sandbox pool, agent console, multi-agent. | Full harness |

---

## Testing implications

Per [TESTING.md](../TESTING.md):
- **Tier-1:** tool registry, the loop's step-cap + tool_call parsing, the
  KEEP/exec decision logic — all mockable, CPU-only.
- **Tier-2:** boot self-test gains a "tool loop executes a trivial tool"
  assertion so a broken agent path is caught at boot.
- **Tier-3:** end-to-end "model calls tool → gets result → answers"
  against a live pod; sandbox-escape negative tests (a tool that *tries*
  to read `/data` must fail).

---

## Open questions to settle before V4.0

1. Does the chosen production model's tool-call template parse reliably
   under vLLM's `mistral` parser at our temperatures? (Validate empirically
   — same lesson as the V2.0 extraction NONE-bias.)
2. Streaming + tool loops: how do we present intermediate tool steps to a
   streaming client? (Likely: stream the final answer only in V4.0; add
   step events when the harness/console exists.)
3. Sandbox mechanism: docker-in-pod vs nsjail — decide based on the RunPod
   container's capabilities (privileged vs unprivileged).
4. Where do agent runs store state — extend the compactor's per-conv JSON,
   or stand up the run store immediately? (Lean: extend JSON for V4.0–4.1,
   stand up the store at V4.2 when runs go durable.)
