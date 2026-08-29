# Incident 2026-08-29 — the model degenerating, and the parameter nobody set

**Status:** root cause identified and mitigated by configuration. No code change
was required, and none of the code changes made while chasing it were the fix.
**Severity:** S1 by user impact — the assistant produced walls of box-drawing,
repeated identifiers, and text in languages the user does not speak.
**Scope:** distinct from [INCIDENT_2026-08-28.md](INCIDENT_2026-08-28.md), which
was a context-budget cascade. This one is about what the model GENERATES.

**Privacy.** Structure and statistics only. The character counts and script
fractions below are measurements of machine output.

---

## The one-paragraph version

vLLM was running with **no sampling truncation at all** — `temperature 1.0`,
`top_p 1.0`, `top_k 0`, `min_p 0.0`, `repetition_penalty 1.0`. Every one of the
~131,000 tokens in the Tekken vocabulary, including roughly 100,000 non-Latin
ones, was reachable at every decode step. That is the whole cause. It produced
a 1.6% baseline drift rate over three months, and `frequency_penalty 0.3` —
added on 2026-08-29 to fix repetition — amplified it to **83%** by stripping
probability mass off the head tokens and, with nothing clipping the tail,
redistributing it across the entire vocabulary.

The fix is `min_p 0.05` on the model in OpenWebUI. Sixty seconds, no restart,
no rebuild, no hardware.

---

## How it was verified

Not inferred — executed. vLLM 0.19.0's own `to_sampling_params()` was run inside
the production image against a request shaped exactly as OpenWebUI sends one:

```
temperature 1.0   top_p 1.0   top_k 0   min_p 0.0   repetition_penalty 1.0
```

Confirmed independently that nothing else supplies defaults:

- `grep` for vLLM's `"Default vLLM sampling parameters have been overridden by
  the model's generation_config"` across the entire log bundle: **zero hits**,
  so the model contributes none.
- `--generation-config` never passed.
- No sampling parameter appears anywhere in `compactor/*.py`; the chat handler
  forwards the body unchanged apart from `messages` and `max_tokens`.

**The control that settles it.** The sibling model row in the same OpenWebUI
database, `anthracite-org/magnum-v4-12b`, carries `min_p 0.1`, `temperature
0.7`, and `frequency_penalty -0.9` / `presence_penalty -0.9` — penalties that
actively *reward* repetition. It has **zero drift and zero repetition across 46
replies**. Penalties are not the protective ingredient. `min_p` is.

The sampling was tuned for the previous model and never carried across when the
deployment moved to Cydonia.

---

## The rate change, measured

| window | n | drift | repetition |
|---|---|---|---|
| before 2026-08-29 | 193 | 3 (1.6%) | 0 |
| 08-29 to 17:28 (no penalty) | 33 | 0 | 7 (21.2%) |
| after `frequency_penalty 0.3` at 17:28 | 6 | **5 (83.3%)** | 0 |

The penalty genuinely fixed repetition and genuinely bought drift. Fisher exact:
drift p=0.048, repetition p=0.012 — small n, but the direction is unambiguous
and the mechanism explains it.

---

## Code fences are a strong contributing factor

**Found from the owner's own hunch, not from the investigation**, and it holds
up. Across 519 replies:

```
replies containing a code fence : 288 (55%)
degenerate runs                 : 10
  INSIDE a fence                : 8
  outside                       : 2
```

The severity split is sharper than the count. **Every severe run is fenced** —
386, 425, 569, 812, 203, 2771, 1110 and 580 characters. The only two outside a
fence are the shortest, at 144 characters each.

The mechanism is plausible and fits the content: prose carries grammatical
pressure toward completion, and a fenced block carries none. The model's
training data is full of long repetitive material inside fences — ASCII art,
logs, configuration dumps — so once inside one, repetition is locally plausible.
The fenced degenerations are correspondingly code-flavoured: `_batch_handler_
shared`, `config`, `er_callback_error_handle`.

This also explains why `min_p` should be especially effective here. `min_p`
clips tokens below a fraction of the top token's probability, so it bites
hardest where the distribution is FLATTEST — which is exactly the state the
model is in inside a fence, where many continuations look equally plausible.
The fence correlation and the sampling fix describe the same failure from two
directions.

**Prompt guidance, if fenced output persists after `min_p`.** Do not phrase it
as a prohibition. `DO NOT USE BOX-DRAWING CHARACTERS EVEN THOUGH THEY LOOK
GOOD!` sat in the system prompt from 2026-08-28 and the model emitted 1,710 of
them anyway; naming the unwanted thing puts it in context at high attention
weight. Describe the register instead, and say what fences are FOR:

> Write in flowing prose. Use ordinary paragraphs and sentences. Reserve code
> blocks for actual code.

**Style is not degeneration, and the two must not be confused.** This model
likes decorative rules; a 60–146 character rule is within its normal range
(146 was the longest in 501 healthy replies). Degeneration is a run past ~250.
Judge by the number, not by irritation — `/data/lastreply.py` prints both the
longest repeated run and the worst 200-letter non-Latin window.

---

## Explicitly NOT the cause

Recorded so these stop being re-investigated. Each cost real time.

- **fp8 quantization via Marlin on Ampere.** The A40 has no native FP8 and vLLM
  logs a warning about it at every boot, which is why this looked compelling.
  It is ruled out by a single observation: **one unrestarted vLLM process
  produced 35 clean replies, then 10 failures in 18, with identical weights and
  kernels either side.** A constant cannot produce a discontinuity.
- **CUDA graphs, chunked prefill, KV cache, preemption.** Same argument.
- **`compactor/tokens.py`.** It has no importer in the application.
- **The compactor's tokenizer.** It only COUNTS, for budgeting. vLLM tokenizes
  independently for inference. Nothing the compactor measures can change which
  tokens the model emits.
- **`max_tokens` and truncation.** Degeneration onset across 9 affected replies
  was 389, 389, 838, 1134, 1867, 2380, 3835, 6473, 6857 tokens — median 1,867.
  A 10,000-token cap would cut before onset in **0 of 9** cases. Length caps
  cannot separate degenerate replies from healthy ones: the longest healthy
  reply was 15,858 tokens and the longest degenerate one 7,015.
- **The compactor changes in v3.1.x.** Repetition degeneration first appears on
  2026-08-22, six days before the v3.1 image was built.

---

## Mitigation shipped in code

The degeneracy detector (`main.reply_is_degenerate`, v3.1.2/v3.1.3) does not
prevent any of this — by the time it can measure a reply, the user has read it,
and silently rewriting model output is not something this system does. It stops
a degenerate reply being MEMORISED, so a loop cannot write itself into facts,
episodic and summaries and be injected back as though it were worth remembering.

Three rules, every threshold measured against the real corpus rather than
chosen:

| rule | threshold | corpus basis |
|---|---|---|
| repeated character | 250 | longest healthy run 146 |
| repeated token | 120 | p98 = 80, then a cliff to p99 = 384 |
| decoration fraction | 45% over 300 chars | healthy max 37.9% |
| non-Latin letters | 3% over 200 letters | p99 = 1.21%, max 10.79% |

**A false positive here costs the user a memory of her own life**, which is
worse than the loop it guards against, because the loop is visible to her and
this would not be. If drift is fixed at source the detector should fire almost
never; if it starts firing often, that is a signal to investigate sampling
again, not to widen the thresholds.

---

## What this cost, and the lesson

Roughly a day, across three confident wrong diagnoses — the local tokenizer,
reply length, and fp8 on Ampere. Each was mechanistically plausible and each
survived until someone measured instead of reasoning.

The thing that actually found it was running **vLLM's own
`to_sampling_params()` inside the production image** and reading what came back.
Not inspecting defaults in documentation, not reasoning about what they
probably were — executing the code the server executes and printing the result.

The general form, and it is the same lesson as `count_tokens_exact`:
**when a component's behaviour depends on a value, ask the component what the
value is.** Do not derive it. This project has now been hurt twice by a number
that was never wrong on paper and never checked in the running system.
