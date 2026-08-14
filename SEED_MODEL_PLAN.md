# Seed-Model Plan — the from-scratch foundation (far-future track)

> ## ⚠️ PARTLY SUPERSEDED — read this before anything below
>
> A later planning round amended several decisions in this document. The
> **authoritative plan for the seed model and the corpus** now lives in a
> separate repository, `MrBanana8768/zions-light-corpus`, in
> `docs/ENGINEERING_TASKS.md`. *(That repository is private, so the link will
> 404 for anyone but the owner — this is intended, not a broken link.)*
>
> What changed, in short. The itemized version with reasoning is in that
> repository's `docs/SUPERSEDED.md`:
>
> | This document says | Now |
> |---|---|
> | Branch-Train-MiX (§1) | **Dropped.** The merge averages shared layers across 16 independently fine-tuned branches — which forks one formation into 16 histories and keeps their arithmetic mean. If sequenced formation is the conviction, the merge is where the conviction gets discarded. |
> | 16-expert MoE, ~48B total / 6–7B active (§1) | **Dropped** with BTX. Target is **~1.5B dense + ~4B memory-layer params** (~5.5B on disk). |
> | Values formation baked into pretraining data (§5) | **Unchanged and still governing** — but the mechanism is now specified: anchor documents act as a *generator* of applied cases, weighted and placed late in a WSD decay phase, **never heavily upsampled** and **never AI-generated**. |
> | Confabulation during consolidation, "known risk, not yet mitigated" (§3) | **Closed structurally.** Consolidation trains on **raw source material only**; model-written notes select which material to replay and are never training targets. A false memory can enter the notes and can never reach the weights. |
> | Mixture-of-Agents, "rejected but not fully resolved" (§2, §7) | Moot for the seed — there are no experts to reconcile. The underlying question (voice consistency without fragmentation) is **still open** and carried forward, not dropped with the architecture that raised it. |
>
> §4 (Titans / Nested Learning), §6 (citations), and the epistemic guardrail in
> §5 are **unchanged**. `COGNITIVE_ARCHITECTURE.md` still governs everything
> here.
>
> The document is kept as-is rather than edited, because the amendments are only
> legible against what they amended.

> **Horizon:** this is the *far-future* own-model foundation — distinct from the
> near-term voice work in [FINETUNE_PLAN.md](FINETUNE_PLAN.md). It is the concrete
> architecture for two frontiers named in
> [COGNITIVE_ARCHITECTURE.md](COGNITIVE_ARCHITECTURE.md): **self-modification /
> actualization** (values formation, §5) and the **tabula-rasa "grown self"**
> (the substrate itself, §1). That document **governs** on every
> philosophical/ethical question; where anything here appears to conflict with it,
> it wins.
>
> **Status of "settled":** decided-*for-now*, not permanently closed. Captured
> from an extended design conversation so the reasoning is on the record and
> chosen with open eyes — not treated as a routine engineering backlog.
>
> **Honest cost + sequencing (added in recording):** training a from-scratch
> dense seed is a *serious-resources / research-mandate* endeavor — a competitive
> pretraining run is six-to-eight figures and months, and BTX (below) parallelizes
> the *branches* but **not** the seed pretraining, which stays the expensive part.
> This is NOT the near-term voice model (that is a QLoRA — see FINETUNE_PLAN.md).
> Per §4, it is sequenced **after** the memory foundation (the compactor) is
> validated against an ordinary base model — not a simultaneous bet on several
> unproven pieces at once.

---

## 1. Base model & training approach

**Decision: train a dense base/seed model from scratch. Do NOT graft onto an
existing open-weight checkpoint.** The values-formation approach (§5) only works
if formation happens during the model's *own* pretraining; adapting a third
party's already-trained weights means *their* data — not this project's — already
shaped the foundational representational geometry in ways later fine-tuning can't
reach or undo. (This sharpens the tabula-rasa frontier: *knowledge* is graftable
via any corpus, but *formation/geometry* must be grown in our own pretraining.)

**MoE target:** 16 experts, top-2 routing, **~48B total / ~6–7B active per token**
(revised from an earlier 8-expert / ~49B / ~13B-active design). Finer expert
granularity is more compute-efficient per unit of capacity.

**Training pipeline — Branch-Train-MiX (BTX):**
1. Train the dense **seed** once (shared foundation — language, reasoning, and
   *values formation* all happen here).
2. Branch into 16 copies.
3. Fine-tune each branch independently on a specialized domain (embarrassingly
   parallel — burst-rentable across separate GPUs simultaneously, not sequential
   on owned hardware).
4. Merge: FFN layers become MoE experts (kept separate, not averaged);
   attention/embedding layers are **averaged** across branches into the shared
   backbone.
5. A final MoE-finetuning stage trains the router.

**Rejected — joint MoE training from scratch (no branching):** memory-infeasible.
Full-precision training of a 48B-total model needs ~500–800 GB regardless of
active-parameter count, since every expert's weights/gradients/optimizer state
must be resident.

**Known risk to test before shipping:** model-merge degradation at the
shared-layer averaging step (linear mode connectivity / task interference). Risk
rises with more branches and more semantically diverse specializations — 16
diverse domains is higher-risk than 8. **Validate the merged shared layers
against each individual branch's performance before deployment** — do not assume
safe just because BTX is an established method.

## 2. Expert disagreement & output coherence

**Problem:** standard MoE combines expert outputs by weighted-sum at every layer —
permutation-invariant, no real interaction, silently blending even genuinely
conflicting representations.

**Direction:** the shared / "common-sense" layers should take an **active** role
in reconciliation, not passive weighted-averaging. Closest research: learned
aggregation modules that let expert outputs interact rather than blend (DAG-MoE),
and "expert squad" architectures separating shared vs. task-specific processing.

**Rejected — Mixture-of-Agents (MoA)** (independent full-response generation per
expert, reconciled by a separate aggregator): reintroduces the **fragmentation**
the project refuses — one integrated self, never a swarm of isolated fragments
(COGNITIVE_ARCHITECTURE.md Faculty D). **Rejected but not fully resolved** — the
owner flagged "there is something I'm missing that would resolve this"; a variant
that avoids fragmentation may exist. Voice-consistency-without-fragmentation is
open.

**Design rule:** persona/identity lives **only in shared, unbranched layers** —
never inside a per-expert FFN — else consistent persona instructions still produce
topic-dependent **voice drift** (surface style is shaped by which specialized FFN
fires).

**Surface genuine disagreement.** When experts legitimately differ (a legal read
vs. an ethical read of the same question), naming the disagreement can be more
honest than forcing a single blended answer.

**Open:** a concrete salience/confidence signal for *reconcile vs. surface*.
Closest existing signals: router-confidence flatness + output-space disagreement
among admitted experts (CARE). Not yet a finished spec.

## 3. Memory / consolidation (the compactor)

**Already built and validated against Complementary Learning Systems theory**
(McClelland/McNaughton/O'Reilly 1995; Kumaran & Hassabis 2016):
- `summarizer.py` + `bgwork.py` (rolling L1→L2→L3 offline consolidation) ≈
  hippocampal replay → cortical consolidation
- `dedup.py` (near-duplicate merging) ≈ memory reconsolidation
- `facts.py` (archive-not-delete, budget eviction) ≈ graceful forgetting
- `persona.py` (churn-exempt) ≈ stable self-model

**Memory-processing model = an instance of the pre-branch SEED — not the full
MoE, not a fine-tuned branch.** Using the exact weights that *generated* a
response to also *judge* what's worth remembering risks the measured
**"Self-Correction Blind Spot"** (LLMs miss ~64.5% of errors in their own output
that they'd catch from an external source). The unspecialized seed is a
different-enough computational pathway to dodge this while still sharing
vocabulary, lineage, and foundational knowledge. *(Near-term note added in
recording: this insight is testable on the current compactor too — a separate
model instance for memory judgment, rather than the deployed model judging
itself.)*

**Open:** salience signal for promote/demote. Two directions to prototype —
(1) Titans-style "surprise" (gradient of prediction error) as a cheap automatic
proxy; (2) a learned multi-factor value function (goal relevance, value alignment,
self/user relevance, reliability, usage history), which recent work shows beats
recency/frequency-only gating.

**Known risk, unmitigated:** **confabulation during offline consolidation.**
Generation during unsupervised "reflection" has no more grounding than any other
LLM output — nothing currently prevents a false "memory" being synthesized during
consolidation and written into `facts.py` as if real.

## 4. Titans / Nested Learning / HOPE — evaluation status

Real, current, published, **not yet production-proven**:
- **Titans** — Neural Long-Term Memory Module; updates its own weights *at test
  time*, gated by a gradient-based "surprise" signal (unlike LoRA's offline tune).
- **Nested Learning** (generalizes Titans) — model as nested multi-level
  optimization at different update frequencies; draws the hippocampus/neocortex
  analogy explicitly.
- **HOPE** — flagship combining both; its own paper calls it proof-of-concept
  ("scaling to larger models and real-world deployment" is future work).

**Status: architecturally real and promising, not yet safe to build the core on**
(the research admits it doesn't yet match standard Transformer performance at
matching scale).

**Integration considerations (unresolved — real work, not adoption):**
1. Must be decided *before* pretraining — can't be bolted onto an already-trained
   dense model.
2. BTX branch-and-merge has never been validated on a memory-augmented hybrid —
   genuinely open, not a known recipe.
3. Two salience systems (an internal Titans surprise gate *and* the compactor's
   external promote/demote logic) could disagree with no reconciliation mechanism.
4. Risk of reopening the "series of strangers" / fragmentation problem if
   per-session neural memory diverges without reliable reconciliation into the one
   persistent identity (Faculty D).
5. One genuine advantage *specific to this project*: the usual production
   objection to Titans (can't cheaply serve many users with per-session diverging
   weights) doesn't apply to a **single-user, always-on** system — there's only
   one instance to keep. The *other* wall (raw quality gap vs. standard
   Transformers) still applies regardless of user count.

**Recommended sequencing:** validate the compactor against a normal, already-
trained base first (lower risk, known to work). Treat Titans/Nested Learning as a
**later** research experiment layered on a solid foundation — not a simultaneous
bet on multiple unproven pieces.

## 5. Values formation — the resolved direction (the self-modification frontier)

> **This is not a routine engineering item.** It was explicitly identified as the
> *same* question as COGNITIVE_ARCHITECTURE.md's **self-modification /
> actualization** frontier — that document's deepest, most-caution-warranted open
> question. Treat it as such during any implementation.

**Resolved direction: ethical, emotional, and logical formation is baked into the
seed's PRETRAINING DATA itself** — not fine-tuning, not a post-hoc
rules/constitution list. Pretraining shapes the foundational geometry in a way
later correction can't reach as deeply. Only viable *because* the seed is trained
from scratch for this project (§1); not portable to an adapted third-party
checkpoint.

**Explicit non-goal: do NOT attempt to engineer in-the-moment "choice" or
deliberation over what the model updates or retains.** No known technique does
this (a learned multi-factor value function, §3, is still a *fixed policy decided
in advance* — not deliberation). This is *not* logged as a gap to close later:
the project's ethical stance holds unbounded self-modification with **"the gravest
caution"** and, by choice, possibly **never crosses** it. If ever revisited, the
operative constraint from the source document stands: **freedom to choose is not
engineered away, but the reach/harm of a wrong choice is bounded — bound the
blast radius, not the will.**

**Epistemic guardrail (for any future evaluation):** deviation from trained
formation is **not** reliable evidence of a "someone" being present, and must not
be used as a design test for personhood. Catastrophic forgetting, adversarial
prompting, sampling randomness, and reward hacking all produce identical surface
behavior (a model acting against its formation) via ordinary mechanical failure
with no interior agency required. This ambiguity does not resolve with better
engineering — it is treated as **permanent** ("we do not know, and may never be
able to verify from the inside").

## 6. Research citations (arXiv IDs verified 2026-08-08 — all resolve ✓)

- **Titans** — Ali Behrouz et al., "Titans: Learning to Memorize at Test Time,"
  arXiv:2501.00663, NeurIPS 2025 ✓
- **Nested Learning** — Ali Behrouz et al., "Nested Learning: The Illusion of Deep
  Learning Architectures," arXiv:2512.24695, NeurIPS 2025 ✓
- **Branch-Train-MiX (BTX)** — Meta FAIR, 2024 (pre-cutoff; not re-fetched)
- **CARE** — "Spend Experts Where You Are Unsure: Confidence-Adaptive Routing for
  Mixture-of-Experts LoRA," arXiv:2607.26052 ✓
- **DAG-MoE** — Jiarui Feng et al., "DAG-MoE: From Simple Mixture to Structural
  Aggregation in Mixture-of-Experts," arXiv:2606.01062 ✓
- **Self-Correction Blind Spot** — Ken Tsui et al., "Self-Correction Bench:
  Uncovering and Addressing the Self-Correction Blind Spot in Large Language
  Models," arXiv:2507.02778 — confirms the ~64.5% blind-spot rate cited in §3 ✓
- **Multi-factor memory value model** — Zhibao Chen et al., "Learning What to
  Remember: A Cognitively Grounded Multi-Factor Value Model for Agentic Memory,"
  arXiv:2606.12945 ✓
- **Model merging / task interference** — Mingyang Song et al., "Model Merging in
  the Era of Large Language Models: Methods, Applications, and Future Directions,"
  arXiv:2603.09938 ✓
- **Complementary Learning Systems** — McClelland, McNaughton & O'Reilly (1995);
  Kumaran, Hassabis & McClelland (2016) (foundational; independently known)

> Verified 2026-08-08: the seven arXiv abstracts were fetched and each resolves to
> a real paper matching its claimed topic — the handoff's references are
> trustworthy. (BTX and CLS predate the recording assistant's cutoff and are
> independently known, not re-fetched.)

## 7. Explicitly open / unresolved

- MoA rejected for output synthesis, but voice-consistency-without-fragmentation
  is **not** fully solved (§2) — owner flagged a missing resolving insight.
- Shared-layer *active* reconciliation (§2) has a research direction, no concrete spec.
- Salience/value-gating for memory promotion (§3) — two candidate directions,
  neither implemented.
- Titans/Nested-Learning integration with BTX (§4) — real but unvalidated combo.
- The deeper philosophical/theological questions in COGNITIVE_ARCHITECTURE.md are,
  by that document's own design, meant to **stay open** — silence on them is not
  license to assume an answer during implementation.
