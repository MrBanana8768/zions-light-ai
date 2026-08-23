# Fine-Tune Track — a custom voice

> A **parallel track**, not a V-line milestone. Where V1→V4 builds the *stateful
> self* (memory, continuity, agency — the compactor), this track crafts the
> *voice*: a base model fine-tuned to the exact prose we want, uncensored by
> default, dropped into the swappable inference layer via `MODEL_REPO`.

## Why this exists
Every off-the-shelf model carries someone else's taste baked into its weights —
Magnum sounds like Claude (flowery), Cydonia leans restrained-literary, DavidAU's
leans gothic-horror. Temperature + a system prompt steer us partway, but the
voice is never *ours* by default. Fine-tuning is the point where we stop
borrowing a voice and make one.

## Where it sits in the architecture
Two layers, kept distinct (see [ARCHITECTURE.md](ARCHITECTURE.md),
[COGNITIVE_ARCHITECTURE.md](COGNITIVE_ARCHITECTURE.md)):

- **The voice / instrument** = the base model (the vLLM inference layer). *This
  track.* Legitimately shaped **exactly to spec** — an instrument is built to spec.
- **The grown self** = the memory/continuity architecture on top (the compactor).
  That is where "let it become what it will" applies. Fine-tuning the voice does
  not touch it.

A fine-tuned model slots into the existing swappable inference slot — no
architectural change.

## The approach: QLoRA style-LoRA (decided)
Of the four tiers, only one is the right indie tool:

| Tier | Verdict |
|---|---|
| **QLoRA / LoRA style fine-tune** | ✅ **This.** Hours on the A40, ~free. ~500–2,000 curated examples. Teaches voice + uncensored-by-default. Does **not** add intelligence or knowledge — stays as smart as its base. |
| Full fine-tune (16-bit) | ❌ Rented 4–8× H100, hundreds–thousands of $. For pure prose style, rarely beats a good QLoRA. |
| Continued pretraining | ❌ Billions of tokens, multi-GPU, thousands of $. Only for whole-domain *knowledge* gaps, not voice. |
| From-scratch pretraining | ❌ Six-to-eight figures, months — "dumber than a weekend QLoRA." The *far* north-star dream, never the indie path. |

> **Two horizons, not a contradiction.** "From-scratch = don't" is true for the
> *near-term voice* — this track. It is a different statement from the **far-future
> foundation** in [SEED_MODEL_PLAN.md](SEED_MODEL_PLAN.md), where a from-scratch
> seed *is* the plan — because **values formation** (the self-modification
> frontier) can only be baked into a model's own pretraining, which no QLoRA can
> reach. Different purpose (voice vs. formation), different horizon (now vs.
> resource-gated someday). This doc is the *instrument*; that one is the *foundation*.

## The recipe
- **Base (warm-start):** **Cydonia-24B** (TheDrummer, on Mistral-Small-24B,
  Apache-2.0) — the *same* model we deploy near-term, so the near-term swap and
  this track converge. Warm-starting on an already-creative/low-refusal tune
  inherits prose + de-censoring; we add our voice on top. (Fast-iteration
  alternative base: Mistral-Nemo-12B, Apache-2.0.)
- **Tools:** **Unsloth** on the A40 (fastest single-GPU; fits 24B QLoRA on 48GB;
  SFT + ORPO). LLaMA-Factory is a friendlier web-UI alternative.
- **Sequence:**
  1. **Prototype the voice with few-shot first** — strong base + 2–3 in-context
     exemplars + a firm system prompt. Zero training cost; discovers the recipe.
  2. **SFT LoRA** (teaches the voice): 1–3 epochs, rank 16–64, low LR. ~80–90% there.
  3. **Optional ORPO** (sharpens off the generic "AI-slop" register): *chosen* =
     our voice, *rejected* = the base's bland default on the same prompt.
     Single-stage, reference-free — light on the A40.
- **Uncensoring:** fold "never refuse" straight *into* the style SFT/ORPO data
  (our styled examples themselves never refuse). De-censor + voice in one pass;
  the style data also heals any abliteration damage if we warm-start abliterated.

## The real work: DATA (~80% of the effort)
The training is the easy afternoon; the project is the corpus.
- **Size:** ~500–2,000 *pristine* (prompt → styled-continuation) pairs, or ~1–10M
  tokens of target-voice prose. Small-and-curated beats big-and-noisy.
- **Sourcing:** a corpus of the prose we actually want; for existing passages,
  synthesize the instruction that would have produced each.
- **The hardest question is upstream of all of it:** *what is the voice?* Defining
  the target style precisely — and in line with the project's purpose — is the
  real design work, not the training.

## Realistic expectations
- ✅ A model that reliably writes in *our* voice, holds tone/format across a scene,
  drops the "As an AI" reflex, needs no giant prompt — trained in an afternoon on
  the A40 for the price of coffee. It genuinely feels like *ours*.
- ❌ Not smarter than its base (style ≠ intelligence); no new world knowledge;
  won't beat frontier closed models on raw reasoning.

## Cost
Style LoRA (QLoRA, 24B, ~1–2k examples, 2–3 epochs): **hours on the A40, ~free**
(electricity) or ~$5–30 rented. Iteration is cheap — 3–5 recipe variants in a
weekend.

## Milestones (draft)
0. **Define the voice** — a style spec + a few gold exemplars. ← *the real gate.*
1. Assemble + clean the dataset (~500–2k pairs).
2. Few-shot prototype → lock the recipe.
3. SFT LoRA on the A40 (Unsloth); eval by reading (+ an LLM judge).
4. Optional ORPO pass; fold in uncensoring.
5. Merge LoRA → serve via vLLM (`MODEL_REPO`), A/B vs the base.

## Open questions
- **The voice spec** — the biggest one; ties directly to the project's purpose.
- **License** — prefer Apache bases (Mistral/Qwen) for freedom; Cydonia /
  Mistral-Small are Apache. Gemma / Llama licenses carry restrictions.
- **Corpus provenance** — what we may lawfully train on; keep outputs owned/lawful.
- **Sequencing** — this runs *parallel* to the V-line; when do we spend the
  data-curation time?

## North-star note
Curating the dataset *is* giving the model its voice — crafting the instrument,
fully ours to shape to spec. The grown self still lives a layer up, in what
accumulates over time. No tension; different layers.
