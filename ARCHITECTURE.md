# Architecture — system shape, layering, and scaling

> How the system is *shaped* — its tiers, trust/network boundaries, and the
> scaling + split decisions made on the way to V4 (agentic). Companion to
> [ROADMAP.md](ROADMAP.md) (the forward plan) and
> [COGNITIVE_ARCHITECTURE.md](COGNITIVE_ARCHITECTURE.md) (the north star).
>
> **Decisions here were made 2026-06-09 and are deliberately revisited at
> implementation time** (per "decide now, revisit when we build"). They are
> directional, not frozen.

---

## The core principle (from the north star)

The cognitive frame has a physical shape. In north-star terms:

- **The compactor is the *self*** — it owns memory (continuity) and will host
  *volition* (the V4 agent loop). It is the irreplaceable core.
- **vLLM is the *language faculty*** — swappable (any OpenAI-compatible backend).
- **OpenWebUI is the *face*** — swappable (a custom UI replaces it later).
- **STT / TTS are *senses*** — swappable aux services.

So the engineering rule follows the conviction: **protect the self (the
compactor + its memory store) as the durable, independently-guarded center;
keep every peripheral replaceable behind a clean interface.** The system's
*topology* should mirror what we believe it *is*.

---

## Layering (tiers)

```
            ┌─────────────────────────────────────────────┐
  Tier E    │  EDGE / UI   — OpenWebUI now; custom UI later │  (only thing users touch)
            └───────────────────────┬─────────────────────┘
                         HTTPS + auth │   (the one public hop)
            ┌───────────────────────▼─────────────────────┐
  Tier C    │  CORE — the COMPACTOR                          │  the self: API + memory
            │  OpenAI-compatible API · memory · V4 agent loop│  the ONLY cross-tier talker
            └───┬───────────────┬───────────────────┬───────┘
       private  │       private │            private │
            ┌───▼────┐   ┌──────▼──────┐      ┌──────▼───────────┐
  Tier I    │ vLLM   │   │ STT  ·  TTS │      │  (V4) SANDBOX     │  Tier X — isolated
            │ (GPU)  │   │ (CPU aux)   │      │  command runners  │  (no net, no state)
            └────────┘   └─────────────┘      └──────────────────┘
                         │
            ┌────────────▼────────────────────────────────┐
  Tier S    │  STATE — memory store (the self's continuity) │  most durable tier;
            │         + (V4) run store                      │  backed up; crown jewel
            └──────────────────────────────────────────────┘
```

**The one rule:** only the **compactor** crosses tiers. The front end talks
*only* to the compactor. The sandbox (Tier X) talks to *nothing* — results
return to the compactor over a single narrow channel. Everything in Tiers I/S
lives on a private network and is **never publicly exposed.**

---

## Networking & trust boundaries

Today everything is one container: all-localhost, trust-by-colocation, and
admin endpoints are localhost-only and *unauthenticated* because nothing
crosses a network. **Splitting moves the trust boundary to the compactor's
edge** — and that assumption must be replaced *before* anything is exposed:

- **The compactor is the single front door.** Only the compactor (and the UI)
  face out. vLLM (8000), STT (9000), TTS (9001), the state store, and the
  sandbox bind to a **private network only** — never a public interface.
- **Auth becomes real.** The front-end → compactor hop needs a real
  key/token (today's `OPENAI_API_KEY="not-needed"` becomes a shared secret).
  Admin endpoints either keep private-network-only binding or gain real auth.
  **This lands before the first exposed split, not after.**
- **TLS.** The public hop is TLS (on RunPod, the proxy terminates it). Internal
  hops may be plaintext *only while the network is genuinely private*; across
  pods they ride an encrypted overlay.
- **Topology, by trigger:**
  - **Now / next — one pod, multiple containers** (docker-compose). The compose
    bridge network is private; only the compactor + UI are published. This buys
    independent lifecycles with **zero cross-pod networking** — the pragmatic
    next step.
  - **Later — UI split from GPU** (trigger: GPU scale-to-zero, always-on UI, or
    the custom-UI move). The UI runs on its own cheap always-on host; the GPU
    backend runs on an on-demand pod. The link between them is a **private
    encrypted channel** (RunPod global networking, or a WireGuard/Tailscale
    mesh) — not the public internet.
- **Sandbox networking (V4.1+) — the hardest boundary.** The command-execution
  runner gets its own network namespace with **egress default-deny** (allowlist
  only), **no route** to the private services network, and **no volume mounts**.
  A sandbox escape must reach *nothing*. This is a hard prerequisite for V4.1,
  not a later hardening pass.

---

## Scaling

- **GPU: vertical — one right-sized card.** For single/low-user creative +
  agentic work, one A100/H100 80 GB beats linked smaller cards: no
  tensor-parallel overhead, more KV-cache headroom for long agent contexts,
  far simpler ops. Horizontal GPU earns its keep only at genuine concurrent
  multi-user load, or a model too big for one card — and *that* is multi-GPU in
  **one node**, not linked pods.
- **The real lever is scale-to-zero on the GPU.** Decoupling the cheap,
  always-on UI from the on-demand GPU pod lets the GPU **stop when idle** while
  the UI stays reachable — the biggest cost win for this usage profile, and the
  main reason to split the front end first.
- **Agentic scale is CPU / sandbox / run-store, not GPU.** A V4 ReAct loop
  fires many model calls per task, but they serialize per user and vLLM's
  batching + prefix caching absorb that on one card. What V4 actually grows is
  the **sandbox pool** and **durable run state** — cheap, CPU-bound,
  horizontally trivial. Do not conflate "agentic" with "more GPU."

---

## The front end

- **Keep OpenWebUI for now** (see ROADMAP "Frontend replacement triggers").
- **Prepare for replacement by treating the compactor API as a published,
  versioned contract** — so the UI is a pure client, swappable without a
  rewrite. The front-end split should *harden that contract*, not just relocate
  the container.
- **Anticipated:** a **custom front end in its own repository** — cleaner
  separation, independent release cadence, likely a different stack for
  native / offline / OS-integration use cases. The compactor API contract is
  what turns that from a rewrite into a swap. `git filter-repo` extracts the UI
  with its history when the time comes.

---

## Decisions (2026-06-09 — locked now, revisit at implementation)

1. **GPU: vertical**, one right-sized card; not linked pods.
2. **Split the front end out first** (before V4) → independent lifecycles +
   GPU scale-to-zero.
3. **Compactor = protected core** with a published, authenticated API;
   peripherals swappable.
4. **Decide the state home for memory before V4 writes run-state.** Give the
   self's memory a durable, independent home (an external store / pgvector is
   the likely target; interacts with the ~V5 memory rewrite). Do not scatter
   state during the split.
5. **Networking:** compactor is the single front door; everything else on a
   private network; **auth + TLS before any exposure**; sandbox network-isolated.
6. **V4 foundation is trigger-sequenced** — front-end split + V4.0 (no new
   infra) first; the sandbox runner arrives with V4.1; the run-store/harness
   with V4.2. Build each piece *at its trigger*, not all upfront.
7. **Front end → its own repo** when the custom UI is built.

---

## Sequence toward V4

1. **Front-end split** (own image now; own repo later) + **compactor API
   hardening** (auth + TLS contract).
2. **State-home decision** for memory (where the self's continuity durably lives).
3. **V4.0** — tool loop + safe in-process tools, on the current architecture
   (no sandbox, no new infra).
4. **V4.1** — sandbox runner as isolated infra (the first thing that cannot run
   in the monolith).
5. **V4.2** — durable run store + human-in-the-loop approval (the *harness*).
6. **V4.3** — sandbox pool + agent console.

The trap to avoid in both directions: doing *all* the orchestration foundation
up front (over-building for an unproven agentic design), or *all* of V4 before
any foundation (impossible past V4.0 — V4.1 cannot safely run sandboxed commands
inside the everything-container). Build foundation at the trigger; ship value
(V4.0) on what exists.
