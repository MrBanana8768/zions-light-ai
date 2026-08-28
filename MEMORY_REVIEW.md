# MEMORY_REVIEW.md — the four memory layers, reviewed as one system

**Date:** 2026-08-27 · **Branch:** `fix/v3.1-remediation` · **HEAD:** `0b9fbaf`
**Scope:** `facts.py`, `retrieval.py`, `summarizer.py`, `persona.py`, and the
injection block, budget guard and async tail in `main.py`.

---

## 1. What this is

This is a **design review**, not an action list. `REMEDIATION.md` remains the
action list and nothing here changes its ordering or its release gates. Where a
finding below is already in `REMEDIATION.md`'s defect tables it is marked
**TRACKED** with its defect id, and it appears here only for the sake of a
complete picture — not as a new discovery.

**Why now.** Every previous review of this system looked at one layer at a time,
or at one incident. `INCIDENT_2026-08-24.md` established that the memory layers
each knew *content* and none knew *completeness*. `REMEDIATION.md` then triaged
~50 defects, layer by layer. What has never been written down is what the four
layers do to each other: how they share a turn counter that means four different
things, how they compete for one system message, and what the model actually
receives when they are all rendered at once. That is what this document covers.

**What it supersedes.** Nothing. It complements:

| Document | Owns |
|---|---|
| `COGNITIVE_ARCHITECTURE.md` | what the system may claim to be; the honesty constraints |
| `REMEDIATION.md` | the defect list, severities, release gates, ordering constraints |
| `ROADMAP.md` §"~V5 — Retrieval-on-demand memory" | the committed direction |
| `INCIDENT_2026-08-24.md` | the production evidence base |
| **this document** | the four layers as one system: interactions, cost, injected quality, and what the target state has to be sized against |

**Method.** Every module above was read in full at HEAD. Behavioural claims were
executed inside the production image (`angreg/zions-light-ai:v3.0.5-cu12`,
`--entrypoint /opt/compactor-venv/bin/python`, `--network none`, scratchpad
storage root, `_llm_summarize` stubbed) against scratchpad copies. Nothing in
the repo tree was modified. Five independent analyses were produced and then put
through an adversarial pass; where the adversarial pass overturned a finding,
the finding is reported as **KILLED** below rather than quietly dropped.

**Line numbers.** All `file:line` references were re-verified against HEAD for
this document. `REMEDIATION.md` cites pre-v3.1 positions in places (e.g. it
gives `main.py:1202` for `turn_index`, which is now `main.py:1394`, and
`retrieval.py:144/166` for `_doc_id`/`upsert`, now `retrieval.py:160/187`). The
defects are the same; only the lines moved.

**Honesty about this document's own evidence.** Several central quantities —
above all tokens-per-message on the production conversation — are **not
measured**. Two of the five analyses inferred values that differ by a factor of
ten. Section 8 lists what would settle each one, and each is minutes of work on
the pod. Numbers derived from unmeasured inputs are marked **[D]**; measured
ones **[M]**; assumptions **[A]**.

---

## 2. How the hierarchy is supposed to work

Four layers, all keyed by `conv_id`, all written after the response in
`_async_tail` (`main.py:1017`) and all read on the request path as pure local
reads with no LLM call.

| Layer | Holds | Written | Read | Push budget |
|---|---|---|---|---|
| `persona.py` | the captured system prompt | on capture / admin | `main.py:~1540` | client-sized |
| `facts.py` | LLM-extracted bullets | `main.py:1103` extract → `:1130` dedup → prune | `main.py:1564` | 1,500 tok (`facts.py:65`) |
| `retrieval.py` | one ChromaDB doc per exchange | `main.py:1066` | `main.py:1586` | 1,500 tok (`retrieval.py:64`) |
| `summarizer.py` | L1 (20 msgs) → L2 (10 L1s) → L3 (5 L2s) | `main.py:1199` | `main.py:1602` | **none** |

The intended read path, once per request:

```
persona.load          → block 1
facts.select_for_injection → touch → format_facts_block   → block 2
retrieval.retrieve(exclude recent 8) → format_retrieval_block → block 3
summarizer.load_state → format_summary_block              → block 4
    ↓
"\n\n".join(blocks) → inject_system_block (main.py:523)
    ↓ inserted AFTER the leading system run
_merge_adjacent_system_messages (main.py:544) → one system message
    ↓
_enforce_hard_budget (main.py:710, in a threadpool at main.py:1683)
    ↓
vLLM
```

The intended design has four load-bearing ideas, and all four are sound:

1. **Tiered rollup beats flat re-summarization.** `summarizer.py`'s docstring
   says the hierarchy "replaces v1's single-shot flat summary" and exists to
   avoid "summary-of-summary degradation". L1 chunks are dropped when they roll
   into L2 (`summarizer.py:385`), so the stack compresses rather than accretes.
2. **All LLM-backed memory work happens offline**, in the bounded pool
   (`bgwork.py`), never on the request path.
3. **Eviction is isolation, not deletion.** `prune_facts` moves evicted facts to
   the archive sidecar, and if the archive write fails, *nothing* is evicted
   (`facts.py:465-473`). `COGNITIVE_ARCHITECTURE.md` ratifies this as principle,
   not as a stopgap.
4. **One injection point.** Mistral-family templates 400 on consecutive system
   messages, so everything goes through `inject_system_block`.

A reader can now judge the gap.

---

## 3. The gap

Severity: **S1** destroys data · **S2** degrades what the model receives ·
**S3** wastes resources.

### 3.0 The cross-cutting root: turn identity

Almost everything below is downstream of one line.

```
main.py:1394    turn_index = len(messages) + 1
```

`turn_index` is derived from the length of the array the client happened to
send. It is then used as (a) the ChromaDB primary key (`retrieval.py:160`), (b)
the RAG recency cutoff (`main.py:1585`), and (c) the `added_turn` stamp on every
fact written by the tail (`main.py:1114`, `facts.py:678`).

`facts.py:95-124` already documents, in the source, that `added_turn` carries
incompatible units and "is a stable tie-breaker and an ordering hint, not a
recency signal". The four writers:

| Writer | Value | Unit |
|---|---|---|
| `main.py:1114` / `facts.py:678` | `len(messages) + 1` | messages in the client's array, **including** system messages, +1 |
| `backfill.py:300` | `i * 2` | pair index × 2 over the full history, **excluding** system messages |
| `commands.py:159` | `ctx.get("turn_index", 0)` | same as the first; the `0` default is latent, not live (`main.py:1445` always supplies it) |
| `dedup.py:344` | `min(added_turn)` over the merged cluster | whichever of the above was smallest |

The summarizer keeps a *fifth* counter: `last_summarized_turn`
(`summarizer.py:343`) is an absolute watermark compared against
`current_turns = sum(1 for m in messages if m.get("role") != "system")`
(`summarizer.py:436`) — non-system messages in the client's array. So the two
turn numbers the model is shown side by side, `--- scene (turns 401-420) ---`
(`summarizer.py:219`) and `--- (turn ~62) ---` (`retrieval.py:~400`), are in
different units and neither is anchored to anything durable.

**TRACKED** as `REMEDIATION.md` **D1** (content-addressed episodic ids; "the
root of F14, F26, D21, D22 and the full F15 fix"), **F14** and **F9**.
`REMEDIATION.md:644` also carries the ordering constraint that matters most:
*nothing that makes `conv_id` more stable may land before D1.*

### 3.1 Facts

| # | Finding | Sev | Where | User-visible | Status |
|---|---|---|---|---|---|
| F-1 | **F9 is half-shipped: eviction is by `added_turn`, and `added_turn` is not a time** | S1 | `facts.py:375-379`, `:397-415`, `main.py:1564-1565` | foundational facts leave the store; "she forgets who she is" | **TRACKED** (F9) — the unshipped half |
| F-2 | Injection order is unsorted below the cap and `added_turn`-sorted above it | S3 | `facts.py:376` vs `:390` | order under a header saying "established earlier" flips at saturation | new, minor |
| F-3 | Extraction is not idempotent; a regenerate appends a second, differently-worded set | S2 | `main.py:1103`, `facts.py:551` | duplicate near-facts, cleaned only probabilistically by dedup | new |
| F-4 | `/forget` does not clear the archive; the archive grows forever | S3 | `main.py:~1990` `_clear_all_memory` vs `memory.py:141` | a "full wipe" leaves every evicted fact on disk, restorable | new |
| F-5 | Nothing distinguishes a `/remember` fact from an extracted guess | S2 | `commands.py:159` — no `source` field in the schema | `"Remembered: ..."` promises durability the store does not honour | new |

**F-1 in detail, because it is the most consequential finding in this document.**

The v3.1 F9 fix separated hot from cold: `select_for_injection`
(`facts.py:397`) returns only what fits the budget, and `touch_facts`
(`facts.py:504`) is called on that subset alone (`main.py:1564-1565`). Its
docstring states the premise: *"the facts left out stop being refreshed and
become the eviction candidates."*

That premise does not hold in steady state, and the reason is visible in
`_lru_split` itself:

```python
facts.py:374-376
    total = sum(_estimate_tokens(f["text"]) for f in facts)
    if total <= max_tokens:
        return list(facts), []
```

`prune_facts` in the *previous* turn's tail already trimmed the store to ≤ the
cap. So on the request path `total <= max_tokens` is always true, `_lru_split`
returns everything, nothing is withheld, and `touch_facts` stamps the **entire
store with one identical timestamp on every single turn** — precisely the
condition F9 was written to eliminate. `last_used` therefore carries no signal,
and the sort at `facts.py:379`

```python
    sorted_facts = sorted(facts, key=lambda f: (f["last_used"], f["added_turn"]))
```

falls through to `added_turn` as the *sole* eviction key.

Measured in the image over a saturated store (`probe_facts.py`, `p_facts.py` —
one store, all four writers, real `select_for_injection`/`touch_facts`/
`prune_facts`):

```
distinct last_used values in the whole store after 20 steady-state turns: 2

kept   added_turns: [32, 34, 36, 38, 40, 241, 243, 245]
evicted            : [0, 2, 4, 6, 8, 8, 10, 12, 14, 16, 18, 20, ...]

TURN 4:  evicted -> (added_turn=8, 'new fact at real turn 241 ...')
TURN 21: evicted -> (added_turn=8, 'new fact at real turn 259 ...')
backfill facts (added_turn 60-118): all 31 still present
```

Read that carefully: it is not FIFO and it is not LRU. It is **"lowest arbitrary
integer wins"**, where the integer's meaning depends on which of four writers
produced it. A fact learned at real turn 241 during a truncated-window request
carries `added_turn = 8` and is archived within one or two turns *of being
learned*. A 60-turn-old backfill fact with `added_turn = 118` survives
indefinitely.

A separate 500-turn simulation against the same real functions (`sim_evict.py`)
shows the well-behaved-client case, which is FIFO:

```
COMPACTOR_MAX_FACTS_TOKENS = 1500
saturated at turn 116; final store = 115 facts
surviving added_turn range at turn 500: 386 .. 500
turn-1 fact still in the store? False
```

Under a full-history client the facts layer is a ~115-entry sliding window over
the most recent ~23% of the conversation. Under a mixed client it is worse than
that and non-monotonic. Both are already true today: `INCIDENT_2026-08-24.md`
records 105 facts on the production conversation, i.e. saturated.

**Dedup pushes the other way.** `dedup.py:344` sets a merged fact's
`added_turn = min(...)` over its cluster, so consolidating a turn-5 fact with a
turn-400 restatement moves the consolidated knowledge to the **front** of the
eviction queue. `COGNITIVE_ARCHITECTURE`'s "schema merging / reconsolidation"
and "adaptive forgetting" rows are wired against each other.

### 3.2 Episodic (retrieval)

| # | Finding | Sev | Where | User-visible | Status |
|---|---|---|---|---|---|
| R-1 | **The ChromaDB doc id is not unique per exchange, and `upsert` makes a collision silent** | S1 | `retrieval.py:160-165` → `:187`; `main.py:1394` | a stored exchange is destroyed with no log line and no recovery | **TRACKED** (D1) |
| R-2 | The recency filter is applied *after* the top-K query, so excluded hits consume K slots | S2 | `retrieval.py:224-226` vs `:240` | the layer routinely returns 1 of 5; `0retr`/`1retr`/`3retr` in production logs | **TRACKED** (F26 + D21) |
| R-3 | `exclude_turns_from = 0` disables retrieval entirely | S2 | `main.py:1585` `max(0, ...)` → `retrieval.py:240` `turn_index >= 0` | episodic recall silently off under a short window; `log_parts` just omits the entry | **TRACKED** (D21) — the one-conditional fix has **not** shipped at HEAD |
| R-4 | No relevance floor | S2 | `retrieval.py:224` `n_results=max(1,k)`, no distance threshold | weak hits injected under a header asserting relevance | **TRACKED** (F26) |
| R-5 | Episodic memory has permanent, unrepairable holes | S2 | `main.py:1066` is the only live caller; `backfill.py:51` `import retrieval` is **dead** | a V1 conversation adopted by V2 gets facts and summaries but **zero** episodic index | new |
| R-6 | `format_retrieval_block` sheds by chronology, discarding the most similar hits | S2 | `retrieval.py:389-404`; `test_retrieval.py:475` asserts "the newest is the one shed" | similarity rank is discarded before budgeting | new |
| R-7 | `conversation_doc_count` materializes every document to produce a count | S3 | `retrieval.py:292` — no `include=` argument; `health.py:158`; `Dockerfile:385` (30 s) | blocking I/O on a health-check cadence | **TRACKED** (F12) |

**R-1 in detail.** `_doc_id`'s own docstring calls it a "stable, unique id per
(conv, turn)". It is neither:

```python
retrieval.py:160-165   _doc_id = f"{conv_id}::{turn_index}"
retrieval.py:187       _chroma_collection.upsert(ids=[_doc_id(...)], ...)
main.py:1394           turn_index = len(messages) + 1
```

Verified with real fastembed + real ChromaDB (`probe_ids.py`):

```
ids after 6 full-history exchanges: [3, 5, 7, 9, 11, 13]

CASE 1 — truncated window of 6 msgs -> turn_index = 7
  id previously held: [user]: user 2: my mother is Sera
  id now holds:       [user]: user NEW: a stranger arrives
  total docs: 6 (was 6)          <- no error, no duplicate, no counter change

CASE 2 — same exchange sent once with a system message, once without
  ids: [3, 4]  docs: 2           <- one query now returns the same text twice
```

The parity observation is new and worth keeping: full history with one system
message yields **odd** ids, so any request whose message count is **even** (a
truncated window, a background/task call, a regenerate after an edit) lands on
an occupied odd id and destroys what was there. The 2026-08-24 shape (7 of 241)
landed on an even id and got away with it; 6 of 241 would not have.

`REMEDIATION.md:622` already records the production instance this analysis did
not have: phantom conversation `31365d633335bbd0` has **105 facts and one
episodic row** — the collision running to completion.

This is the only path in the system that destroys a stored memory with no log
line, no counter change and no recovery. Facts eviction lands in the archive;
this does not.

### 3.3 Summarizer

| # | Finding | Sev | Where | User-visible | Status |
|---|---|---|---|---|---|
| S-1 | **`state["l2"]` is never trimmed and `format_summary_block` has no budget** | S2 | `summarizer.py:192-222`; `:380` appends, `:389-413` keeps | the only uncapped push layer; grows into the window forever | **new** |
| S-2 | **`_needs_l3_rollup` is a standing condition, so L3 regenerates on every turn forever** | S3+S2 | `summarizer.py:241-243`, `:246-252`, `:459-460` | one `max_tokens=2000` LLM call per exchange, forever; the most identity-bearing text is re-paraphrased every message | **new** |
| S-3 | **A failed tier discards the tiers that succeeded, permanently** | S1 | `summarizer.py:470` `save_state` inside the same `try` as `:459` L3 | the summarizer stops advancing, silently and permanently, while burning 6+ LLM calls per turn | **new** |
| S-4 | `_llm_summarize` has no input budget; the L3 body is an unbounded join | S1 (cause of S-3) | `summarizer.py:303` vs `main.py:396-411`; `:400-403` | vLLM 400s once the L2 stack outgrows the window | **new** |
| S-5 | `last_summarized_turn` is an absolute watermark compared against a client-relative count; a branch or edit strands it | S2 | `summarizer.py:343` vs `:436` | rollups stop; L1 chunks describing a deleted timeline are still injected as authoritative continuity | **TRACKED** (F14); the gate half is **D22, dropped (subsumed)** |
| S-6 | The docstring's stated bound is false | S3 | `summarizer.py:18-19` | a written specification of a property the code does not have | new |

**S-2 verified** (`p_sum.py`, real functions, stubbed LLM):

```
after catch-up to 1000 msgs: L1=0 L2=5 L3=True last_summarized=1000  (56 llm calls)
  quiet turn 1: llm calls=['L3']  L2=5
  quiet turn 2: llm calls=['L3']  L2=5   ... indefinitely
```

`_needs_l3_rollup` is `len(l2) >= 5` with no "already refreshed" condition, and
`_do_l3_rollup`'s own docstring says *"this keeps the L2 list"*. So from the
fifth chapter onward `needs_rollup()` is permanently True, defeating the early
exit at `summarizer.py:441`. L1→L2 was given event semantics
(`summarizer.py:385` drops the rolled chunks so the condition clears); L3 never
was.

S-1 and S-6 are the same omission seen from two sides. The module docstring
states the intended shape — *"Total injected size stays bounded (~5K tokens
worst case: L3 + **latest** L2 + a handful of unrolled L1 chunks)"*
(`summarizer.py:18-19`) — and "latest L2" is the trim that was never
implemented. `format_summary_block` renders **every** chapter
(`summarizer.py:215-217`), and nothing anywhere removes one.

**S-3 verified** (same probe, L3 forced to fail):

```
before: L1=0 L2=5 last_summarized=1000
llm calls this pass: ['L1','L1','L1','L1','L1','L3']
returned state (in memory): L1=5 last_summarized=1100
PERSISTED state on disk:   L1=0 last_summarized=1000   <- 5 successful L1 rollups LOST
next turn, same input: identical 6 calls, identical loss, forever
```

The comment on that `try` block (`summarizer.py:462-468`) states: *"Losing a
partial rollup costs one cycle; the rollup re-triggers on the next turn."* It
re-triggers and re-fails, deterministically. This is latent — it needs the L3
body to exceed the window, i.e. tens of L2 chapters — but S-1 guarantees the
stack gets there, the fix is moving one line out of the `try`, and the same
class of error (a comment specifying behaviour the code does not have) shows up
twice more below.

### 3.4 Injection, budget guard, and the tail

| # | Finding | Sev | Where | User-visible | Status |
|---|---|---|---|---|---|
| I-1 | **The trim loop has no `protect_system` exclusion, so it can halve the caller's persona** | S1 | `main.py:805-825` `big = [...]` has no `i >= protect_system`; the drop loop 20 lines below has it | a long character bible is amputated mid-word; on this product that is the worst possible half to lose | **new** — appears in neither `REMEDIATION.md` nor the incident report |
| I-2 | Under budget pressure the memory block is sacrificed to preserve the v1 flat summary | S2 | `main.py:509` puts the flat summary in the leading system run; `:523` injects after it; `:829+` drops `sys_idxs[-1]` | the hierarchical summary dies first, then retrieval, then facts | new (consequence of a tracked defect) |
| I-3 | The halving keeps the wrong half | S2 | `main.py:819` `c[: len(c)//2]`; block join order `main.py:~1600` | headers survive their own content; the model reads "[Hierarchical summary … use them for continuity]" followed by `[...trimmed to fit the context budget]` | new |
| I-4 | The tail fires on background/task calls | S3 | `main.py:1766`, `:1830` — bare `if conv_id:` | phantom conversations accrete facts; a title-generation prompt gets the full roleplay injection | **TRACKED** (D2, scheduled V9) |
| I-5 | Compaction's tokenization runs on the event loop | S3 | `main.py:477` and `:384` vs `:1683` (budget guard *is* threadpooled) | a full `apply_chat_template` + `encode` over the whole history, plus one per message, blocking, on every over-budget request | **new** |
| I-6 | Dedup does not memoise a refused cluster | S3 | `dedup.py:382-392` | a cluster the LLM answered `KEEP` on is re-formed and re-litigated on every subsequent turn that adds a fact, at one LLM call each | **new** |
| I-7 | Dedup's Stage 1 blocks the event loop and its docstring is off by ~70× | S3 | `dedup.py:111-138`, `:198-201`; no `run_in_threadpool` | ~1.9 s of blocked loop per turn at 102 facts | new |

**I-5 is newly live.** `count_tokens` (`main.py:302`) has two real-tokenizer
tiers, and until commit `f458cf9` ("jinja2 missing from the compactor venv — the
accurate token counter has never run") tier 1 had never executed. It executes
now. `compact_if_needed` awaits it directly at `main.py:1490` — not through
`run_in_threadpool`, which the budget guard *does* use at `main.py:1683`. The
asymmetry is unintentional and it sits on the hot path.

### 3.5 What is clean, and should be protected in any rewrite

Verified, not assumed:

- **No layer receives its own literal output.** `summarizer._format_turns`
  (`summarizer.py:272-273`) skips `role == "system"`, so injected blocks never
  reach the summarizer; `last_user_text` (`main.py:1377`) is read from the
  pristine client array before image stripping and before injection; the tail
  receives `messages`, not `body["messages"]` (`main.py:1766`, `:1830`). Three
  independent guards, all holding. Protect them.
- **The archive discipline.** `prune_facts` (`facts.py:465-475`): if the archive
  write fails, nothing is evicted and the store stays over budget. That is the
  right trade and it is principled, not accidental.
- **Persona is the one layer whose storage matches its claims.**
  `auto_capture_persona` (`persona.py:273-314`) refuses to demote an admin or
  inherited record; `text_to_inject` (`persona.py:250-270`) hash-matches to
  avoid double injection.
- **Idempotence where it was worked on.** `_merge_touched` (`main.py:969-999`)
  and `_merge_backfilled` (`backfill.py:199-227`) treat disk as authoritative
  for membership and only move `last_used` forward; `archive_facts`
  (`facts.py:255-267`) de-dupes by text; `/remember` and `/forget` hold
  `conv_lock` across load-modify-write (`commands.py:155`, `:182`); the episodic
  upsert takes `conv_lock` (`main.py:~1064`) so it cannot land after a
  `/forget`.
- **The map-reduce input budget in `main.summarize`** (`main.py:396-411`) — the
  fix for the 2026-08-13 overflow. It is exactly the discipline
  `summarizer._llm_summarize` still lacks (S-4).

### 3.6 Killed by the adversarial pass

Reported as killed, per the review protocol.

| Claim | Verdict |
|---|---|
| "A1/R-1 is the most severe finding in the system" | **Down-ranked.** R-1 needs an even-length client array to collide. F-1 fires on **every** turn of a saturated store. Frequency beats severity-per-event; F-1 leads. |
| "R-1, R-2, F-1's root and the phantom tail are new findings" | **Killed.** All four are in `REMEDIATION.md` (D1, F26+D21, F9, D2). The parity case, the duplicate case and the "F9 is half-shipped" measurement survive as additions. |
| "S-5 (the rollup gate) should be fixed as its own item" | **Killed as an item.** `REMEDIATION.md:685` records D22 as *"Dropped (subsumed)"*: once the server owns the transcript and the model receives a bounded window, "count the client's array" is a bug in a code path that no longer exists. What survives is the *content* argument, which the subsumption does not cover: L1 chunks labelled "turns 21-40" that describe a branch the user abandoned are still injected verbatim as authoritative continuity, and that holds after the front end lands. |
| "`/remember` hits the `ctx.get('turn_index', 0)` default and stamps 0" | **Killed.** `main.py:1445` always supplies `turn_index`, and `commands.handle_command` has exactly one caller. Latent hazard, not live behaviour. The probe line showing an evicted `[/remember, ctx had no turn_index]` fact was an artifact. |
| "`backfill.py:300`'s `i * 2` is always small (2, 4, 6…)" | **Killed.** `main.py:1640` hands backfill the full `original_messages`, so `i * 2` is a position in the whole message stream — near-enough the same unit as `len(messages)+1`. Those two writers broadly agree; the live incompatibility is one writer (`len(messages)+1` under a short window) plus `dedup`'s `min()` propagating whatever is smallest. |
| "Injection order flips discontinuously at the cap" (adversarial said `_lru_split` sorts on both branches) | **Overturned by code evidence — the original claim stands.** `facts.py:375-376` returns `list(facts)` **unsorted** under budget; `facts.py:390` returns `sorted(..., key=added_turn)` over budget. The practical consequence is small, because the on-disk list is already close to `added_turn` order, which is why this is filed as F-2 (S3) and not higher. |
| "Retrieval ⊂ summary, always" | **Killed as stated.** The argument compares `last_summarized_turn` (non-system positional count) against `turn_index - 8` (`len(messages)+1`). Those are different units and neither is anchored. The honest statement — "usually overlapping, by an amount nobody can compute" — is a worse problem than the one claimed. |
| "$X per 100 turns" / "$43,230 to reach E=10,000" | **Killed.** `RUNPOD_DEPLOY.md` describes a pod rented by the wall-clock hour. Marginal dollars per turn on this deployment are **zero**. See §4. |
| "T ≈ 1,422 tokens/message" | **Suspended, not killed.** See §4.1 and §8. |
| "The summary block reaches 13,700 / 18,500 / 125,396 tokens" | **Relabelled.** These are ceilings computed from `L1/L2/L3_MAX_TOKENS`, which are *output caps*, not observed output sizes, at rollup states production has never reached. The unboundedness is real; the numbers are worst case. |
| "Dedup runs 9-10 LLM calls every turn" | **Killed as stated.** Dedup is gated on `if new_entries and len(combined) >= 2` (`main.py:1130`), so it does not run on turns with no extraction, and the 9-10 figure has no cited source. The real amplification mechanism is I-6. |
| "`archive_facts`'s O(N²) write amplification is urgent" | **Down-ranked.** Right mechanism (`facts.py:262-266` load → filter → save on every eviction), wrong urgency: the measured 410 ms at 20,000 archived entries is decades of this conversation away. |

---

## 4. The costed model

### 4.1 Units, and the one number that is missing

**Dollars are the wrong currency here.** The deployment is a rented pod billed
by the wall-clock hour (`RUNPOD_DEPLOY.md`). The GPU costs the same whether it
is prefilling 700k tokens or idle. The two currencies that actually bind are:

1. **wall-clock latency the single user waits through**, and
2. **window tokens** — of 32,768, how many are spent on memory rather than
   conversation.

Ratios stated in GPU-seconds remain meaningful (they say where the time goes);
absolute dollar figures do not, and have been removed.

**Tokens per message, `T`, is unmeasured and the estimates disagree by 10×.**

- `main.py:749-751` records a production measurement of **4.10 chars/token** on
  a 168-message, 979,685-char corpus. Read as tokens-per-message that gives
  **T ≈ 1,422** — but only if that corpus *is* this conversation, which is not
  established.
- Reconstructing the injected block from the production fact count gives a
  retrieved-exchange size of ~590 chars/message, i.e. **T ≈ 145**.
- An independent check argues against the high value: `format_retrieval_block`
  budgets `MAX_RETRIEVAL_TOKENS * 4` = **6,000 characters**
  (`retrieval.py:404`). At T = 1,422 a single two-message exchange is ~11,600
  chars, so the retrieval layer could never emit more than one truncated
  exchange, ever. Either T is far smaller than 1,422, or a layer everyone
  treats as live is effectively a no-op.

**This is one grep on the pod** (§8). Until it runs, every absolute figure below
is parametric in T. The *call counts* do not depend on T's exact value; the
*seconds* do.

### 4.2 The call ledger, one user message

Constants, verified at HEAD:

| Constant | Value | Source |
|---|---|---|
| `MAX_MODEL_LEN` | 32,768 | `main.py:66` |
| `TARGET_TOKENS` (compaction trigger) | 24,576 | `main.py:67` |
| `HARD_INPUT_LIMIT` | 30,720 | `main.py:79` |
| Summarize input budget | 32768 − 1024 − 2048 = **29,696** | `main.py:406-410` |
| `KEEP_RECENT_TURNS` | 4 | `main.py:68` |
| Map concurrency | `asyncio.Semaphore(4)` | `main.py:423` |
| Facts cap / retrieval cap | 1,500 / 1,500 | `facts.py:65`, `retrieval.py:64` |
| L1/L2/L3 chunk sizes | 20 msgs / 10 L1 / 5 L2 | `summarizer.py:58-60` |
| L1/L2/L3 output caps | 500 / 1200 / 2000 | `summarizer.py:65-67` |

With `M` = messages in the request and `B` = `floor(29696 / T)` messages per
compaction batch:

| Work | Calls per user message | Class |
|---|---|---|
| Compaction map (`main.py:415`) | `ceil((M − 4) / B)` | **O(N)** |
| Compaction reduce (`main.py:434-446`) | 1 while partials fit one group | O(N/B) |
| Chat completion | 1 | O(1) |
| Fact extraction (`facts.py:614`) | 1 | O(1) |
| Dedup (`dedup.py:382`, gated at `main.py:1130`) | 0 … `MAX_LLM_CALLS_PER_PASS` = 10 | O(1), capped, and see I-6 |
| L1 rollup | 0.1 amortized (every 10 exchanges) | O(1) |
| L2 rollup | 0.01 amortized | O(1) |
| **L3 rollup** | **0 below 5 chapters, then 1.0 every turn, forever** (S-2) | **standing** |
| Phantom tail (I-4) | +1 extraction, +0…10 dedup per background call | O(1) each |

**The dominant term is compaction, and it is O(N) per turn, therefore O(N²)
cumulative.** `compact_if_needed` (`main.py:476-515`) re-summarizes everything
but the last four messages, from scratch, on every request over 24,576 tokens —
on this conversation, every request. Measured in production: **2m27s and 4 LLM
calls to re-summarize 50 turns that were already summarized.** The model of the
call count reproduces that observation exactly (3 map batches + 1 reduce).

The arithmetic, stated as a ratio rather than a currency: at any conversation
length where compaction fires, compaction's prefill is `(M − 4) × T` tokens and
the chat completion the user actually asked for is one prefill of at most 24,576
plus a few hundred decoded tokens. **Compaction is roughly 90-95% of the GPU
work in a turn, and it produces a summary of material the L1/L2/L3 stack has
already summarized.** That single ratio is the strongest argument in this
document for the Stage 0 fix in §6.

### 4.3 Storage growth

Measured in the image with 2,000 synthetic exchanges [M]; `dbstat` on the
resulting 39 MB sqlite file:

```
  8.94 MB  embedding_metadata_string_value      <- the document, copy 1
  7.84 MB  embedding_metadata (index)
  7.84 MB  embedding_fulltext_search_content    <- the document, copy 2
  6.98 MB  embedding_fulltext_search_data       <- FTS5 inverted index
  0.15 MB  embeddings
```

Plus 3.35 MB of HNSW files. So ChromaDB stores the text ~3× and full-text
indexes it: **≈4.7× raw text plus ~1.9 KB fixed per document.** `retrieval.py`
only ever calls `collection.query(query_embeddings=...)` — it never runs a text
search — so **the FTS5 tables (38% of the file) are pure waste for this
workload.**

| Layer | Bytes per turn | Bounded? |
|---|---|---|
| ChromaDB | ≈4.7 × exchange text + 1.9 KB | **no** — no age or size pruning anywhere in `retrieval.py` |
| Summary state JSON | ~250 B | **no** — `l2` never trimmed (S-1) |
| Facts archive sidecar | ~170 B once saturated | **no** — `/forget` does not clear it (F-4) |
| Active facts JSON | 0 (capped at 1,500 tok ≈ 102 facts) | yes |
| `.backfill.json` sidecars | one per conversation | no |
| `_conv_locks` (`memory.py:333`) | one asyncio.Lock per conv_id | bounded by conversation count — fine |

Bytes are not the constraint on this deployment. **Write amplification is**:
`archive_facts` (`facts.py:262-266`) does a whole-file load → filter → save on
every eviction, and every write is `atomic_write_json` (temp + fsync + rename)
on MooseFS. Cumulative bytes written is O(N²). Real, not urgent (§3.6).

### 4.4 Where each layer stops working

Not "slows down" — stops.

| Layer | Breaks at | Mechanism |
|---|---|---|
| **Facts** | **already broken** | 1,500-token cap ≈ 102 facts; production is at 105. Every new fact now evicts one, chosen by `added_turn` (F-1). Nothing restores from the archive on a schedule — `restore_from_archive` is admin / `/list-archive` only. |
| **Retrieval** | **already degraded** | R-2 returns a fraction of K; R-3 can disable the layer entirely; R-6 sheds the most similar hits. |
| **Summary stack** | when `l2` grows into the window (S-1), and permanently the first time L3's oversized body 400s (S-3/S-4) | uncapped render + uncapped `_llm_summarize` input + `save_state` inside the failing `try` |
| **Compaction** | when `(M − 4) × T` makes the map-reduce wall time exceed a client or proxy idle timeout | `compact_if_needed` runs *before* the vLLM call (`main.py:1490`); the forward client sets `read=None`, so nothing on the compactor side times out first |
| **bgwork pool** | when tail duration exceeds the user's typing interval | `_async_tail` takes a pool slot then `conv_lock`, so one conversation's tails serialize; past `MAX_OUTSTANDING=64` (`bgwork.py:35`) `submit()` returns False, `_fire_and_forget` discards the return value, and those turns' facts, embeddings and rollups are **never extracted and never retried** |
| **Shutdown** | any length | `pool.drain(timeout=10.0)` (`main.py:1259`) against a tail that takes tens of seconds: every `supervisorctl stop compactor` abandons in-flight tails |

**The first thing to actually break was the facts layer, and it broke around
turn 120.**

---

## 5. Quality — what the model actually receives

### 5.1 The literal block

`inject_system_block` (`main.py:523`) inserts one system message after the
leading system run; `_merge_adjacent_system_messages` (`main.py:544`) then
collapses the run into a single `role=system` string. On an over-budget request
— which is every request on this conversation — the assembled order is:

```
role=system   <<CLIENT SYSTEM PROMPT / PERSONA>>

              [Summary of earlier conversation]
              <<V1 FLAT SUMMARY of turns 1..N-4>>          (main.py:505-509)

              <<persona · facts · retrieved · summary>>    (main.py:~1600, joined "\n\n")
role=user     ...
```

Rendered with the real formatters against a production-shaped store
(`reconstruct.py`; ~102 facts, block = 10,807 chars ≈ 2,701 tok):

| Section | chars | ~tok |
|---|---|---|
| persona | 867 | 216 |
| facts | 4,172 | 1,043 |
| retrieved | 2,491 | 622 |
| summary | 3,271 | 817 |

**Three observations that are about the prompt, not the code.**

1. **Register collapse.** The persona is second-person literary instruction. The
   102 facts are flat third-person declaratives *about* the same character. The
   retrieved block is the model's own prose in first person. The summary block
   is clinical past-tense synopsis. In one system message the model is asked to
   *be* the character, to be consistent with a dossier about the character, to
   read the character's dialogue as archive, and to absorb a synopsis of the
   character. The dossier is 39% of the block by characters. **The model is
   shown far more prose about the character than prose in the character, every
   turn.** This is a plausible mechanism for the "too flowery" complaint and it
   is *unfalsified, not established* (§5.4).
2. **Two numbering systems, presented as one.** `--- scene (turns 401-420) ---`
   (`summarizer.py:219`) and `--- (turn ~62) ---` (`retrieval.py:~400`) are in
   different units (§3.0). Any answer the model gives to "what happened around
   turn 200?" is fabricated with the system's help. This is not only a coherence
   defect; it is an honesty defect (§5.5).
3. **Two competing accounts of the same past.** The v1 flat summary and the
   L1/L2/L3 stack cover overlapping spans in different voices, ~2,000 chars
   apart, both labelled as summaries, with no instruction on how to reconcile
   them. This is the *quality* cost of the already-diagnosed `compact_if_needed`
   defect, and it has not previously been named.

### 5.2 Redundancy

Measured with the image's own `bge-small-en-v1.5` against the reconstructed
block: 7 of 102 facts score ≥0.80 against a summary sentence, 1 of 102 ≥0.90;
**30 of 102 (29%) have ≥60% of their content words covered by a single summary
or retrieved sentence.** Highest matches are exact restatements
(`0.983` fact↔summary; `0.848`; `0.819`).

The low cosine average is *worse* news than a high one: facts and summaries
describe the same events at different granularity, so the model receives the
same beat three times in three incompatible compressions and must decide which
is authoritative — with no signal about which to trust.

The structural claim "retrieval ⊂ summary, always" was **killed** (§3.6): the
counters are in different units. What survives is that the overlap is
substantial and uncomputable.

### 5.3 The foundational-memory-loss mechanism

The full chain, for a preference or world-fact established at turn 5, at turn
500:

1. **Archived out of the facts store around turn 120** (F-1, measured).
2. **Dropped at the L1→L2 hop, or never reaching it.** Compare the three rollup
   prompts (`summarizer.py:291`, `:298`, `:300`): L1 is told to preserve *"The
   user's stated preferences and goals"*; L3 is told to preserve *"the user's
   overarching goals, persistent constraints"*; **L2 — the only path from L1 to
   L3 — is told to keep "characters, settings, decisions, ongoing threads" and
   to "drop scene-by-scene minutiae"**, and a one-line style instruction inside
   a 20-turn scene summary is exactly what reads as minutiae.
   `_do_l2_rollup` then deletes its ten source L1 chunks (`summarizer.py:385`),
   so there is no recovery.
3. **Not reachable by retrieval.** `format_retrieval_block` sorts hits
   chronologically (`retrieval.py:389`) and fills the budget from the front, so
   similarity rank is discarded before budgeting (R-6); and R-2/R-3 may have
   returned nothing at all.
4. **Deleted outright the first time the budget guard bites.** The join order
   puts the summary last, and `main.py:819` cuts `c[:len(c)//2]`, so the first
   halving removes the hierarchical summary and half the retrieved exchanges
   (I-2, I-3). The code's own quoted production line records five halvings in
   one request (`main.py:830`).

**And the summary layer is not where it is assumed to be.**
`INCIDENT_2026-08-24.md:55` records the production state: **105 facts, 98
indexed episodic exchanges, 5 L1 chunks, `last_summarized_turn = 100`, on a
208-message conversation.** That is **no L2 and no L3 at all**, and turns
100→present exist in *no* summary. Every analysis in this review that reasoned
about a mature L2/L3 stack was reasoning about a state this system has never
reached. The layer carrying the middle of the conversation today is a ~115-fact
window over the most recent material, plus top-K episodic retrieval — the two
layers that F-1, R-2 and R-3 damage.

### 5.4 Established versus inferred — read this before quoting the section above

**Established** (measured against real code):

- Eviction in a saturated store is by `added_turn`, an integer that is not a
  time, every turn, deterministically (F-1).
- The episodic id collides and `upsert` makes it silent (R-1), with a production
  instance in `REMEDIATION.md:622`.
- The recency filter runs after the query (R-2); `cutoff = 0` disables retrieval
  (R-3).
- L2 is never trimmed and L3 refires every turn once five chapters exist (S-1,
  S-2); a failed L3 discards successful L1s (S-3).
- The trim loop can halve the caller's persona (I-1).
- The production summary state is 5 L1, no L2, no L3, `last_summarized_turn=100`
  on 208 messages.

**Inferred, and not established:**

- That any of the above *causes* the user's "forgetting who she is" complaint.
  At least four mechanisms predict the same symptom and nobody has separated
  them: (a) fact eviction; (b) **Cydonia-24B degrading over a 32k context on its
  own — this is the null hypothesis and no analysis stated it**; (c) the persona
  sitting atop a multi-thousand-token dossier with nothing after `messages[0]`
  surviving the prefix cache (lost-in-the-middle); (d) `compact_if_needed`
  regenerating the flat summary *from scratch every single turn*, so the
  character's own past is literally rewritten in a slightly different voice each
  message, alongside a hierarchical stack telling a different version. On
  priors, (d) is at least as strong a candidate as (a).
- That the dossier register causes "too flowery". Plausible mechanism,
  unmeasured.
- Every absolute token/latency figure that depends on `T` (§4.1).

### 5.5 There is no measurement of whether memory helps

Plainly: **no eval, no regression test, and no metric for injection quality
exists anywhere in this repo.**

- `tests/eval/` covers **speech-to-text WER only** (`wer.py`, `stt_eval.py`,
  `fixtures/*.wav`). Its README is candid that the test tiers "prove the code is
  correct… none of them answer *'is the output actually good?'*"
- `tests/integration/test_facts.py`, `test_retrieval.py`, `test_summarizer.py`
  exercise 5-20 turns — below every threshold that matters. Nothing has been
  evicted, no L2 has rolled, the budget guard has never fired.
  `REMEDIATION.md:316` already notes that `test_prune_facts_lru_eviction`
  constructs a `last_used` distribution the live pipeline cannot produce: *"It
  passes; the feature it tests does not exist."*
- `test_retrieval.py:475` asserts *"the newest is the one shed"* — locking in
  R-6 without ever asking whether shedding by chronology is right. No test
  asserts the top-similarity hit survives.
- `selftest.py`'s `_check_facts_round_trip` is liveness, not quality.

**Both user complaints are currently unfalsifiable against this codebase**, and
so is every fix proposed in this document. That is the single most important
thing in §8.

### 5.6 Honesty

Judged against `COGNITIVE_ARCHITECTURE.md` Principle 2 ("claim nothing
unearned — refuse false witness, **including about itself**") and Principle 6
("degrade honestly, fail loud"). These are live violations in shipped strings
the model reads as fact:

1. **`facts.py:487`** — *"[Persistent facts about this conversation —
   **established earlier**, maintain consistency with these]"*. "Established" is
   a provenance claim about unverified LLM extraction, produced by the same
   model that wrote the reply, possibly merged by a second LLM call whose only
   record is a log line. A confabulated bullet becomes a standing instruction to
   be consistent with a falsehood, and the header forecloses doubting it. **One
   line of text; the clearest Principle-2 violation in the codebase.**
2. **The retrieval header** — *"retrieved by similarity — use them for
   continuity and **exact recall**"* — is emitted unchanged when the budget has
   just discarded the most similar hit (R-6), when four of five hits were
   filtered after the query (R-2), and it is simply *absent*, with no line
   saying so, when the cutoff collapsed to 0 (R-3).
3. **The model has no representation of its own memory state.** Nothing tells it
   which turns are covered, that `last_summarized_turn` is 100 on a 208-message
   conversation, that 16 facts were archived, or that the budget guard halved
   the block. So it cannot give the honest answer — *"I have facts from the
   recent window and a summary of turns 1-100; if you told me that in between, I
   may have lost it"* — because the information required to say it is never
   given to it. This is the 2026-08-24 shape: the report records that the
   model's categorical answer about cross-conversation memory *was correct*, and
   the "false witness" claim was **withdrawn on review**. The failure was that
   the system can only give categorical answers about itself where the user
   needed a specific one.
4. **`/remember` overstates durability.** `"Remembered: {arg!r}"` is a
   first-person promise; the fact carries no `source` field (F-5), is evicted by
   `added_turn` like any other, and the user is never told when it goes.
5. **Fail-loud terminates in a log file.** `degrade.guard`, `StoreUnreadable`,
   the tri-state work — all of it writes to a pod log the single user never
   opens. For a one-user companion system, honesty that reaches only the
   operator is not honesty. `INCIDENT` C3 (disclosure and pre-send comparison)
   is the control that would have fired on the incident with no other symptom,
   and it is still PLANNED.

One caveat on the persona layer, which is otherwise the cleanest in the system:
an admin/inherited persona is correctly *not* overwritten from the request
payload, so if the client's system prompt differs, the stored persona is
injected **in addition to** it — two identity blocks, potentially
contradictory, both under "treat this as the primary identity and voice".

---

## 6. Target state

The committed direction is `ROADMAP.md:808-846`: memory behind a `recall(query)`
tool, only persona plus a compact digest pushed, Postgres + pgvector replacing
per-conversation JSON and embedded ChromaDB. This section sizes and sequences
it, and separates what needs it from what does not.

### 6.0 The finding that should shape the sequencing

**Retrieval-on-demand, exactly as scoped, addresses a small fraction of the
cost.** `ROADMAP.md:812-817` attributes the cost to *"every memory layer is
pushed into the context on every single request."* That is real but small:
facts (≤1,500) + retrieval (≤1,500) + summary (~2,500 today) ≈ 5,500 tokens of
push. The dominant cost is **compaction re-summarizing the client's resent
history** (§4.2) — and compaction is triggered by what OpenWebUI *sends*, not by
what the compactor *injects*. Moving memory behind a lookup interface does not
touch it.

What V5 genuinely fixes: the facts layer's *pushing* problem (100 facts is the
most you can afford to paste; there is no number you cannot afford to index),
the ~4.7× ChromaDB text duplication and its unused FTS5 tables, the whole-file
JSON rewrite pattern, and `conversation_doc_count`. Those are worth having. They
are not worth sequencing the cheap fixes behind.

### 6.1 Stage 0 — available today, needs nothing

No Postgres, no tool loop, no schema migration, no new dependency. In rough
order of value per hour:

| | Change | Fixes |
|---|---|---|
| 0.1 | **`compact_if_needed` consults `summarizer.load_state` and skips raw turns already covered by the stack**, keeping the rest verbatim; fall back to `summarize()` only for the residual tail (at most `L1_CHUNK_SIZE - 1` = 19 turns, never 50) | ~90% of the GPU work in a turn; the summary-of-summary degradation the summarizer's own docstring warns about; one of the two competing accounts (§5.1) |
| 0.2 | **Cap `format_summary_block`** — `COMPACTOR_MAX_SUMMARY_TOKENS`, newest-first, same char-budget approach `retrieval.py:404` already uses | S-1, the only uncapped push layer |
| 0.3 | **Give `_needs_l3_rollup` an "already refreshed" condition** | S-2 — one standing LLM call per turn, forever, plus the per-turn re-paraphrasing of the most identity-bearing text in the system |
| 0.4 | **Move the recency filter into Chroma's `where` clause**: `{"$and":[{"conv_id":…},{"turn_index":{"$lt":cutoff}}]}`, and treat `cutoff <= 0` as "no exclusion" | R-2 and R-3 |
| 0.5 | **`if i >= protect_system` in the trim loop's `big` list** (`main.py:806`) | I-1 — five characters, protects the persona |
| 0.6 | **Move `save_state` out of the shared `try`** (`summarizer.py:470`) and **budget `_llm_summarize`'s input** the way `main.summarize` budgets its own | S-3, S-4 |
| 0.7 | **Normalise the `added_turn` writers to message-units** and tie-break eviction on insertion order rather than `added_turn` | F-1 — F9's unshipped half |
| 0.8 | **Memoise refused dedup clusters** by text-hash | I-6 |
| 0.9 | **`include=[]` on `conversation_doc_count`** (`retrieval.py:292`) | R-7 / F12 — one keyword on a path that runs every 30 s |
| 0.10 | **Reorder the injected blocks: persona → summary → facts → retrieved** | the blocks are joined into one string, and the most volatile layer (retrieval, query-dependent) currently sits *before* the most stable (summary, changes every 20 turns), invalidating it on every request. Free prefix-cache win; also the better reading order, general → specific |
| 0.11 | **Gate the tail on background/task calls** | I-4 / D2, already scheduled as V9 |

**Caveat that must ship with 0.1:** it compares a client-array position against
`last_summarized_turn`, which is itself an array-position counter (F14). Within
one request both are non-system counts of the same array, so it is
self-consistent for today's full-history client — but it is not free of the
identity problem. Pair it with the interim mitigation `REMEDIATION.md:624`
already names as cheap: **reset `last_summarized_turn` when the observed history
is shorter than it.** Without that reset, one short request latches compaction
off as well as rollup.

Stage 0 is pull-*shaped* in effect — less pushed, cheaper, better organized —
with none of the risk of pull.

### 6.2 Stage 1 — turn identity (hard prerequisite for everything after)

`turn_seq`, server-side, monotonic, persisted per `conv_id`, allocated in
`_async_tail`; D1 content-addressed episodic ids; a write-fence so an allocated
index that would regress below the stored high-water mark allocates
`turn_seq + 1` rather than upserting; the `added_turn` unit audit, where rows
whose era cannot be proven are marked `unit=legacy` and **excluded from
span-scoped queries** rather than converted by guesswork.

**Why this gates the target design and not merely improves it.** Today, if
`turn_index` is wrong, the model gets slightly wrong material under a header
saying "possibly relevant". Under the target design the digest publishes a
*contract* — "turns 41-60: her mother's illness" — and `recall(span=[41,60])`
must honour it. A span derived from the client's array length is a fiction, and
the model then answers a continuity question from the wrong material **with a
citation**. That is Principle 2's false witness arriving through an engineering
defect.

**And the ordering constraint is not optional.** V5 requires a stable
`conv_id` — a lookup key you cannot trust is not a lookup key — and
`REMEDIATION.md:644-645` states that nothing which makes `conv_id` more stable
may land before D1, because `_doc_id` is derived from `len(messages)+1` and
written with `upsert`. **V5 therefore forces the exact ordering hazard the
incident named, and this appears nowhere in the ROADMAP's V5 section.**

### 6.3 Stage 2 — the orientation digest, still pushed

Build the digest and the recall index as offline artifacts maintained by the
consolidation tail, and ship them **pushed**, replacing the facts + retrieval +
summary blocks. Four parts, stable → volatile:

1. **Pinned identity (~400 tok, 20-25 facts).** A `pinned` flag; never evicted,
   never recalled, always present. Selection is a `/pin` command and an admin
   endpoint, not automatic. *A model that must call a tool to learn the user's
   name has been made worse, not better.*
2. **The L3 theme if one exists (~150 tok).**
3. **"Where we are now" (~350 tok)** — the newest L1 scene, verbatim. This is
   the highest-value block in the whole system and today it is rendered last in
   an unbounded stack.
4. **The recall index (~300 tok)** — a list of spans with one-line labels:

```
Things I can look up in more detail (ask me with recall):
  turns 1-20    · first meeting; the terms she set
  turns 21-40   · the trip north; the argument about the letter
  ...
```

The index is nearly free: `_do_l1_rollup` already computes `first_turn`/
`last_turn`; add one label line to the L1 prompt's output schema, ~30 tokens
generated once per 20 turns, offline. **It is also the answer to the ROADMAP's
own open question, "how does the model know what to ask for?"** Without it,
retrieval-on-demand is strictly worse than push for a model with weak tool-use
instincts.

Also at Stage 2: a **relevance floor**, and a **confidence-gated automatic
pre-pass**. The embedding query already fires every turn (`main.py:1586`); keep
it, and change only what it does with the result — inject only if the top hit
clears the floor. Zero additional compute, and it means the common case ("the
user just referenced turn 47") is served without the model deciding anything.

**Stage 2 is the whole context change with zero dependence on the model's
tool-use ability.** If the project stopped here it would have captured most of
the win. That is deliberate — see §6.6.

### 6.4 Stage 3 — Postgres + pgvector

Migrate the store behind the interface Stage 2 established; nothing
user-visible changes. Add a `scope` column (`conv` today, room for `self` later)
and a `source_principal` column now, unused. Both are trivially populated for a
single user and cost nothing; without them, cross-conversation scope later
requires re-deriving provenance that was never captured.
`COGNITIVE_ARCHITECTURE.md` (Open Questions, 2026-06-09) already decided the
topology — *"unify, don't fragment"*, per-conversation memory is *"scaffolding,
not the destination"* — **and it decided the ordering: "healing before
accumulation", with the cross-conversation self last.** V5 is not the release
that ships that; V5 is the release that makes it possible without a second
rewrite.

### 6.5 Stage 4 — the V4.0 tool loop and `recall()`

One tool, not a family:

```
recall(query: str,
       scope: "any" | "episodic" | "facts" | "chapters" = "any",
       span:  [int, int] | null = null,
       k:     int = 3) -> { status: "ok"|"empty"|"unavailable", results: [...], note }
```

Contract obligations, each traceable to a principle or an existing defect:

- **Three statuses, never two.** `empty` (searched, nothing cleared the floor)
  and `unavailable` (the store is down) must be distinguishable — the same
  discrimination `REMEDIATION` P0-2b already requires of
  `conversation_doc_count`, which today returns `0` for both. If a dead
  ChromaDB returns `empty`, the model tells the user "we never discussed that":
  a false statement produced by an outage, in the voice of a companion.
- **A relevance floor.** Under push a weak hit wastes tokens; under pull it is a
  confident wrong answer with a span citation attached.
- **Provenance rendered to the model** — `[recalled — turns 41-60, similarity
  0.71]` — so it can hedge.
- **Read-only, in-process, no LLM call**, obeying the discipline in §2.
- **Bounded** — reuse `MAX_RETRIEVAL_TOKENS` per call, plus a per-turn
  cumulative cap across the tool loop. `MAX_TOOL_STEPS` alone is not a token
  bound.
- **One deliberate side effect:** `recall` touches `last_used`. Under pull,
  "what was actually reached for" becomes a real measurement and LRU becomes
  meaningful for the first time.

Push is **added to**, not removed.

### 6.6 The risk this design must be built around

Cydonia-24B is a roleplay finetune of a Mistral-Small-24B base. That class of
model trades structured-output and instruction-following reliability for
in-character prose, and a companion roleplay runs at high temperature, which
degrades format adherence further. `V4_PLAN.md` open question 1 flags exactly
this and has never been answered empirically.

**Worse, the failure correlates with need.** Breaking character to emit a JSON
tool call is what an in-character finetune is trained *not* to do — and the
turns where continuity matters most are the deepest in-character ones. The
honest expectation is that this model will under-call `recall()`, and will
under-call it hardest exactly where recall matters most. A model that does not
call it, whose push has been narrowed, gets *less* context than today: a product
regression delivered by an architecture improvement.

There is a second, quieter cost. **Pull failures are silent and look like good
writing.** Today's push failures are legible in the log — `0retr`,
`archived 16 least-recently-used`, `sum(L1=5/L2=1/L3=n)`. A model that fails to
call `recall()` does not error; it confabulates fluently, which is what it was
optimised to do. Moving to pull trades legible failure for illegible failure on
a system whose entire remediation programme is about legibility.

Mitigations, in priority order:

1. **Never remove the floor.** The digest stays pushed, permanently. A model
   that never calls `recall()` still gets pinned identity, the current scene, the
   theme and an index — better organized than today's blob. The failure mode is
   bounded to "no episodic detail", never "no memory".
2. **The confidence-gated pre-pass** (§6.3). Most recalls should happen without
   the model deciding anything. If the design *depends* on the model choosing to
   call the tool, it is fragile by construction.
3. **`/recall <query>` as a slash command.** Works on any model, needs no
   tool-call parsing. `commands.py` already has the surface and the user's habit.
   If the model cannot reach for its memory, the person can.
4. **Log `recall_calls`, `recall_status`, `prepass_hit`, `digest_only` per
   turn** — and gate any narrowing of the push on a real number. Without this
   the system loses the ability to know whether it remembered, and so does the
   person maintaining it.
5. **A small tool-calling model on CPU is a legitimate answer.** If Cydonia's
   tool calling proves unreliable at roleplay temperature, route the *recall
   decision* to a small instruct model while Cydonia keeps the prose — the exact
   pattern the ROADMAP already endorses for the memory-judgment experiment.

Two things that would look like mitigations and are not: **do not** prompt
"you MUST call recall before answering" into the persona (it fights the
finetune's training and pollutes the one layer the project treats as exempt from
churn), and **do not** force `tool_choice: required` (it converts under-calling
into over-calling, spending a step and up to 1,500 tokens on a query the model
did not want to make).

---

## 7. What NOT to do

This section is for the maintainer with a day job. Everything here is defensible
work that would cost more than it returns *right now*.

1. **Do not sequence the Stage 0 fixes behind the V5 programme.** Postgres +
   pgvector, then the V4 tool loop, then a live-store migration, then a `recall`
   contract, then `pinned` with a `/pin` command and an admin endpoint, then a
   recall index requiring an L1 prompt-schema change — gated behind two version
   lines, multi-month. The summary block is uncapped **today** and the L3 call
   fires every turn **today**. §6.1 is hours and needs none of it.

2. **Do not treat content-addressed episodic ids (D1 / R-1) as a near-term
   item.** It is correct and it is not cheap: a schema change to a live store
   needing a migration. `REMEDIATION.md` already put it in its own release for
   exactly that reason. The parity-collision detail added here does not change
   the cost.

3. **Do not re-litigate D22 (the rollup gate counting the client's array).**
   `REMEDIATION.md:685` closed it with a rationale — it is subsumed by D1 plus
   `FRONTEND_SPEC §4/§11`, and becomes a bug in a code path that will not exist.
   Reopening it costs review time for a fix that gets deleted. (The *content*
   argument — stranded L1 chunks describing an abandoned branch, still injected
   as authoritative — is separate and does survive; it belongs with the digest
   work at Stage 2, not as a gate fix.)

4. **Do not design the `recall` error contract before the tool is scheduled.**
   The tri-state is the right principle and the best single idea in the target
   design; specifying it in detail now is work with no consumer.

5. **Do not build the E=1,000 / E=10,000 cost projections into a plan.** The
   production conversation is around 120 exchanges. The near-term row and the
   falsification test in §8 are the whole value of the cost model; the
   extrapolations are decoration, and they are parametric in an unmeasured
   constant.

6. **Do not spend more effort on redundancy measurement.** The sentence-cosine
   work in §5.2 is careful and its actionable output is one already-known line:
   the flat summary and the hierarchical stack both ship. Fix that (0.1) and
   re-measure if it still matters.

7. **Do not chase `archive_facts`'s write amplification yet.** Real mechanism,
   decades of this conversation away (§3.6).

8. **Do not narrow the push before the instrumentation in §6.6(4) shows recall
   actually firing.** That stage may never be reached, and that is an acceptable
   outcome.

9. **Do not fix `added_turn` and then write down that it caused "forgetting who
   she is."** Fix it — it is cheap and it is wrong — but see §8.1.

---

## 8. Open questions

Each of these is unresolved *and* cheap to resolve. Every disputed number in
this document is downstream of a measurement nobody has taken: four independent
analyses ran containers against invented data, and none looked at production.

### 8.1 Which mechanism actually causes "forgetting who she is"?

At least four candidates predict the same symptom (§5.4). Tests that
discriminate, cheapest first:

1. **Capture one real forwarded request.** Log the exact array sent to vLLM for
   one production turn and read it. This alone settles `T`, the real L1/L2/L3
   state, the real block sizes, and whether the two summaries visibly
   contradict each other. Cost: one log line, one turn.
2. **`GET /admin/conversations/<conv_id>`** — returns `state_summary` (l1/l2/l3
   counts, `last_summarized_turn`) and the facts count. Settles whether the
   rollup gate has stalled. Cost: one curl.
3. **The archive A/B.** `/list-archive`; take 10 archived facts and 10 active
   ones; ask 10+10 questions whose answers are those facts. If archived scores
   materially worse, eviction is implicated. **If they score the same, eviction
   is not the mechanism** — this is the only test that can falsify F-1 as the
   *cause*.
4. **The short-context control.** Ask the same 20 questions in a fresh
   conversation whose only system message is the full fact store. If she holds
   up there and not in the live conversation, it is context length, not memory
   content — the null hypothesis nobody stated.
5. **The double-summary isolation.** Run a session with
   `COMPACTOR_HIERARCHICAL_SUMMARY=false`. If coherence improves, the two
   competing accounts are the mechanism.

### 8.2 What is `T`, tokens per message, on this conversation?

Estimates differ by 10× (§4.1) and every absolute figure in §4 depends on it.
**Falsification, one grep on the pod:** compare `msgs=` on a live turn against
the `compacted:` line's batch count. If `msgs≈500` and compaction reports ~25
batches, `T` is large and the wall-time projections hold. If it reports ~7, `T`
is small and compaction is much cheaper than modelled — in which case the
`format_retrieval_block` arithmetic (`retrieval.py:404`, 6,000 chars) also
becomes consistent, and the retrieval layer is doing real work rather than
emitting one truncated exchange.

### 8.3 Is `compact_if_needed` even firing on every request?

Assumed throughout, from the conversation being over `TARGET_TOKENS`. Settled by
the same log line as 8.2.

### 8.4 Can Cydonia-24B emit well-formed tool calls at roleplay temperature?

`V4_PLAN.md` open question 1, never answered empirically. It gates Stage 4
entirely (§6.6). A single afternoon of prompting the production model at
production temperature would answer it, and the answer determines whether
`recall()` is the load-bearing path or a supplement to a permanently pushed
digest.

### 8.5 What would a memory eval look like, and is it worth building?

Nothing in this repo can currently falsify either user complaint or any fix
proposed here (§5.5). The cheapest thing that changes that: **freeze the
production conversation's export as a fixture, define ~30 probe questions whose
answers are known to lie in turns 1-100, and score recall under the current
injection versus any candidate.** The injection half needs no GPU — the block is
deterministic given the store, which is how the reconstruction in §5.1 was
produced. Until something like this exists, every change in §6 ships on argument
alone, including the ones this document argues for.

### 8.6 What does the user get told?

`degrade.guard`, `StoreUnreadable` and the tri-state work all terminate in a pod
log (§5.6). `INCIDENT` C3 — disclosure and pre-send comparison — is the control
that would have fired on 2026-08-24 with no other symptom, and it is still
PLANNED. For a one-user companion system this is the open question with the
shortest path between "known defect" and "the person actually finds out", and it
is a design question, not an engineering one: what does the model get to say
about the state of its own memory, and where does the person see it?

---

*Probe and benchmark scripts used for this review live in the session scratchpad
and are disposable:*
`C:\Users\rngge\AppData\Local\Temp\claude\D--Projects-zions-light-ai\4093b5b3-15e5-4911-b9b1-44038fc21dcb\scratchpad\`
*(`probe/`, `work/`, `q/`, `adv/`). Nothing in the repository tree was modified
in the course of producing this document.*
