# V4 — roadmap, and what the v3.1 week changed about it

**Status:** planning. Supersedes nothing — [`compactor/V4_PLAN.md`](compactor/V4_PLAN.md)
remains the tool-execution design spec and this document does not restate it.
**Written:** 2026-08-28, the day v3.1 reached production.

This exists because the week that produced v3.1 changed what V4 has to be. The
tool-loop architecture in `V4_PLAN.md` is still right. What changed is the set
of things that must be true *before* it, and the constraints it inherits.

---

## 1. Two user reports, and where they land

The user of this system reported two things after a day on v3.1. Both are real,
both are partly fixable now, and neither needs V4 in full.

### 1.1 "It doesn't keep track of the actual time"

**Root cause: nothing in the injected context carries wall-clock time. Not one
layer.**

| Layer | What it carries | What the model can infer about time |
|---|---|---|
| Facts | `added_turn`, `last_used` (unix) | nothing — neither reaches the prompt |
| Summaries | `first_turn`, `last_turn` | nothing — turn ordinals, not dates |
| Episodic | `turn_index` | nothing — same |
| The request itself | messages | nothing — no timestamp anywhere |

So the model has no clock and no dating on its own memories. Asked what time it
is, it invents one; asked when something happened, it cannot know. The invented
timestamps visible in the live fact store (`4:01 AM`, `6:47 AM`,
`9:19 AM Friday`) are not a reasoning failure — they are the only thing a model
with no clock can do.

**Fix — v3.1.x, not V4. Four changes, in value order:**

1. **Inject the current time.** One line, ~15 tokens:
   `Current time: 2026-08-28 20:41 MST (Friday)`.

   **Put it LAST, immediately before the newest user turn — not in the system
   preamble.** vLLM's prefix cache keys on the leading prompt; a value that
   changes every request, placed early, invalidates the cache for the entire
   conversation on every turn. Placed late, the long prefix stays cacheable and
   the cost is one uncached line. This is the difference between a free feature
   and a latency regression on every message.

2. **Date the retrieved exchanges.** Highest value of the four. Episodic recall
   is the layer that answers "remember when we talked about X" — and it
   currently injects those exchanges with no indication they are from last week
   rather than this morning. Requires a wall-clock field on the episodic
   metadata at index time.

3. **Date the summary chunks.** L1 chunks already carry `first_turn`/`last_turn`;
   add the date range. One line per chunk, and it lets the model say "earlier
   this month" instead of guessing.

4. **Age the facts.** Facts have no creation timestamp at all — `last_used` is
   when the fact was last *injected*, which is not the same thing and must not
   be reused for this. Add `added_at` at write time.

   Do **not** render a date per fact; at ~90 facts that is real budget for
   little gain. Annotate the block instead ("facts learned 2026-06-01 to
   2026-08-28"), and date individual facts only where the fact is old and the
   block is small.

**Effort:** roughly a day for all four. Item 1 alone is under an hour and
resolves the literal complaint.

### 1.2 "It's not good with numbers"

Three distinct causes wearing one symptom. Separate them before fixing any.

**a. It may be the context starvation we just fixed.** On 2026-08-29 the model
was receiving 6,231–11,972 tokens of a 90-message conversation. A model that
cannot see what was said twenty turns ago cannot track a number stated there,
and the failure looks exactly like bad arithmetic. **Re-measure after the D1/D2
fixes land before building anything.** This is the cheapest possible
explanation and it is currently unexcluded.

**b. Fabricated precision is a prompting problem, not a system one.** The live
fact store contains model-generated dashboards with invented percentages
(`ENERGY LEVEL: 88% → 92%`, `PROTECTION LEVEL: 1,000,000,000%`). Those numbers
are decoration, not computation — the same output habit as the box-drawing
rules that caused `INCIDENT_2026-08-28`, and the same fix: the system prompt.
Note the compounding: those fabricated figures were then *extracted as facts*
and re-injected, so the model has been reading its own invented numbers back as
established truth. Fixing extraction (v3.1.1 D5) removes the feedback loop; the
prompt removes the source.

**c. The genuine part is real, and it is a V4.0 tool.** Mistral-class models
are weak at arithmetic and no amount of context fixes that. A calculator is the
canonical Tier-A tool: pure Python, no sandbox, no network, deterministic,
trivially testable. It is the single clearest justification for V4.0 that this
deployment has produced.

**So: (a) verify, (b) prompt, (c) V4.0.** Do them in that order and each step
may make the next unnecessary.

---

## 2. What must be true before V4 starts

V4 gives the model *agency*. Everything below is a prerequisite because agency
multiplies the cost of every defect underneath it.

**The system must be able to tell the truth about itself.** This week produced
two incidents in which the failure presented as *"she forgot me"* rather than as
an error, and both were invisible to every log, test and health check. A tool
loop built on a substrate that reports success while silently dropping context
will produce actions taken on context the operator cannot reconstruct. The
honesty work in v3.1 — measured budgets, ERROR-level shedding, `/health/full`
reporting degradation — is not polish preceding V4. It is the floor V4 stands on.

**The concrete gates:**

| Gate | Why V4 cannot start without it | Status |
|---|---|---|
| Budget measured against the enforcer | A tool loop makes N model calls per turn; an undercount multiplies by N | ✅ v3.1 (P0-0c) |
| Failure signal that reaches a human | `COMPACTOR_ALERT_WEBHOOK` is still unset; the user is the monitoring | ❌ open |
| Injection bounded as a whole | Layer caps that individually pass and jointly overflow (v3.1.1 D3) | 🔨 v3.1.1 |
| A client that can show what was sent | Approving an action means seeing the context it was based on | ❌ FRONTEND_SPEC |
| Temporal grounding | An agent that acts on "yesterday" must know when yesterday was | ❌ §1.1 |

That fourth row is the one most likely to be underestimated. **V4's
human-in-the-loop approval is not implementable on OpenWebUI.** Approving a
mutating action requires seeing the exact context the model acted on, and
`FRONTEND_SPEC` §12's receipt — *what the server actually admitted* — is the
mechanism. There is no approval UI without it.

---

## 3. The phasing, updated

`V4_PLAN.md`'s tiers stand. What follows is sequencing against what v3.1 taught.

### V4.0 — the tool loop, Tier A only

A bounded ReAct loop in the compactor: model emits `tool_calls` → compactor
executes → appends results → re-calls vLLM → stops at `MAX_TOOL_STEPS`. No new
service, no sandbox, no harness.

**First tools, chosen because this deployment asked for them:**

- **`now()`** — the clock. Makes §1.1's injected timestamp queryable rather than
  merely present, and lets the model reason about elapsed time.
- **`calculate()`** — §1.2c. Pure arithmetic, no `eval`.
- **`memory_query()`** — ask the memory store a question instead of having
  everything injected pre-emptively. This is the seed of the V5
  retrieval-on-demand idea in `ROADMAP.md`, and it is the tool with the largest
  effect on the budget: today every layer is injected on every request whether
  it is relevant or not.

**The budget consequence, which is the part to design carefully.** A tool loop
makes multiple vLLM calls per user turn, each carrying the growing tool
transcript. The guard currently budgets one request; it will need to budget a
*loop*, with the tool transcript as a new injected layer competing with facts,
retrieval and summaries. v3.1.1 D3 (bounding total injection as a fraction of
the window rather than by per-layer caps that can sum past it) is the
foundation for that, which is why it is a gate and not a nicety.

**Testing bar, inherited:** the tokenizer-contract harness proved that a test
written against an imagined contract passes while production burns. Tool tests
must exercise the shape the code actually sends — including the failure and
fallback paths, since both of the defects that reached production this week
lived in fallbacks a green harness never entered.

### V4.1 — read-only sandboxed commands

Sandbox only, no mutations, no approval flow yet. The sandbox boundary is
specified in `V4_PLAN.md` §"The sandbox boundary" and nothing this week changed
it.

### V4.2 — mutating commands + human approval

**Gated on the new client.** See §2. This is where the harness becomes forced,
per `V4_PLAN.md` §"Phase 2 forces the harness".

### V4.3+ — durable run store, sandbox pool, agent console

Build against the trigger that actually bites. Do not speculate.

---

## 4. Constraints V4 inherits from this week

These are not suggestions. Each is written from a defect that reached
production, and each would be more expensive with an agent than it was without
one.

1. **A component that budgets against a limit must verify against the authority
   that enforces it.** (`FRONTEND_SPEC` §13.) A tool loop budgets repeatedly;
   an unverified counter compounds per step.

2. **A fallback that cannot signal that it fired is not a fallback.** Three
   defects this week were graceful degradations composing into a system that
   lied. An agent degrading silently takes *actions* on degraded context.

3. **A fix applied at one call site and missed at a sibling is the recurring
   failure of this codebase** — four instances on the v3.1 branch, one of which
   reached production. Tool implementations will have parallel paths
   (sync/async, streaming/non-streaming, success/error). Enumerate every site,
   grep, do not assume.

4. **Test the contract the code sends, not the one you imagine.** D1 shipped
   through a harness built specifically to catch it.

5. **Never let the model's own output become authoritative input without a
   filter.** The fabricated dashboards were extracted as facts and re-injected;
   the box-drawing rules were stored in memory and re-fed. An agent whose tool
   *results* enter memory unfiltered has the same loop with a much shorter
   fuse.

6. **The user's data is not a test fixture.** The fact store holds real, deeply
   personal material. Any V4 operation that can write, delete, or transform
   memory must be dry-run by default, archive before removing, and keep
   anything ambiguous.

---

## 5. Sequencing — what to do, in order

**Now (v3.1.x), before any V4 work:**

1. Verify D1/D2 in production — re-pull logs, confirm compaction succeeds and
   the guard stops shedding 80+ turns
2. `COMPACTOR_ALERT_WEBHOOK` — five minutes, and it is the one gate that is
   pure configuration
3. §1.1 item 1 — inject the current time (under an hour, resolves the literal
   complaint)
4. v3.1.1 D3–D7 — the workflow already running
5. §1.2a — re-measure the numbers complaint once context is restored
6. §1.1 items 2–4 — temporal grounding for the memory layers

**Then, and this is the real fork:**

7. **The new client** (`FRONTEND_SPEC`). It is a V4 gate, it is the largest
   single piece of work outstanding, and it is on the critical path for V4.2.
   Starting it in parallel with v3.1.x is reasonable; starting V4.0 before it
   is also reasonable, since V4.0 needs no approval UI.

8. **V4.0** — tool loop + `now()` / `calculate()` / `memory_query()`.

**Open questions to settle before V4.0 starts:** `V4_PLAN.md` §"Open questions"
holds the original list. Add one from this week — **how does the guard budget a
loop rather than a request?** Every existing budget assumes one model call per
user turn. That assumption is load-bearing in `compact_if_needed`,
`_enforce_hard_budget` and the calibration path, and V4.0 breaks it on day one.

---

## 6. What this document does not cover

- **Tool-execution design** — `compactor/V4_PLAN.md`, unchanged
- **The client** — `FRONTEND_SPEC.md`
- **Memory architecture future state** — `MEMORY_REVIEW.md`, whose S-1 (the
  untrimmed L2 injection) becomes more urgent under a tool loop
- **Voice / QLoRA tracks** — `FINETUNE_PLAN.md`
- **Auth** — PR #30, which is a V4 prerequisite for a different reason: a tool
  loop reachable without authentication is a materially worse exposure than a
  chat proxy without it
