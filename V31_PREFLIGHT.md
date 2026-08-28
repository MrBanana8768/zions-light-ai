# v3.1 preflight — ship call and bugfix plan

**Branch** `fix/v3.1-remediation` · **HEAD** `f241e6b` · **Written** 2026-08-28
**Inputs** three independent reviews, two adversarial attacker/defender rounds, one
adversarial verification pass, one docker test-harness investigation. Every line
number below was re-read against the working tree while writing this document.

---

## 1. The call

**Ship v3.1 — after four changes that total about 1 hour 40 minutes.**

v3.1 is a real improvement and it should not be held. The guard now budgets against
vLLM's own `/tokenize` (`main.py:902`, `main.py:1018`) instead of a local estimate that
read 23–51% low. That is the actual cause of both outages and it is fixed. D1 stopped
the episodic store overwriting its own rows. The five destructive memory paths are gone.

But four things must land first, because each one is a *first-day* harm on exactly this
user's conversations, and each fix is small enough to apply and re-read before coffee:

| # | What | Where | Est |
|---|------|-------|-----|
| **P1** | `GENERATION_RESERVE` ships **2048** — every long reply truncates. Raising it alone opens a compaction dead band, so `TARGET_TOKENS` moves in the same commit. | `runpod.env.template:59` | 15 min |
| **P2** | The trim loop recounts a trimmed block **without** the scale factor it applies everywhere else. This branch introduced it. | `main.py:979` | 20 min |
| **P3** | A reply truncated by the token limit is memorised as a **finished** one, into facts, RAG and summaries. | `main.py:1115`, `main.py:2251` | 45 min |
| **P4** | The episodic layer can return nothing and log nothing. | `main.py:1971`, `retrieval.py:373` | 20 min |

Everything else — including several genuine S1s — goes to v3.1.1. The reason is stated
per item in §3. The short version: the deferred items either need design work (a
supersession model, a token budget wired into a second module) or are already loud in
the log. **Shipping a rushed design at 2am is how this project got here.** Do not.

One caveat the owner should carry into the morning: the deployed pod's actual
`COMPACTOR_GENERATION_RESERVE` cannot be read from this repo. See §7 Q1.

---

## 2. Counts

| | S1 | S2 | S3 | total |
|---|---|---|---|---|
| **Before push** | 2 | 2 | 0 | **4** |
| **After push** | 2 | 8 | 5 | **15** |
| | **4** | **10** | **5** | **19** |

19 plan items consolidated from 20 adversarially-confirmed findings (several were the
same defect reached from different directions — `Y6` is `V-01` plus `V-02`; `B3` is
`V-01`; `X7` is one third of `Y7`; `B5` is `Y1`).

Twelve claims were **killed** during verification. They are in §5, because on this
project a recorded kill is worth as much as a finding.

---

## 3. The bugfix plan

Ordered within each group by what would hurt the user most.

### 3.1 · BEFORE v3.1

---

#### P1 · `GENERATION_RESERVE` ships 2048, and raising it opens a compaction dead band
**S1.** `runpod.env.template:59`, `.env.example:77`, `main.py:76`, `main.py:67`, `main.py:594`

Three surfaces set this and all three say 2048:

```
runpod.env.template:59   COMPACTOR_GENERATION_RESERVE=2048
.env.example:77          COMPACTOR_GENERATION_RESERVE=2048
main.py:76               GENERATION_RESERVE = _env_int("COMPACTOR_GENERATION_RESERVE", 2048)
main.py:79               HARD_INPUT_LIMIT = min(MAX_MODEL_LEN, max(256, MAX_MODEL_LEN - GENERATION_RESERVE))
```

against `REMEDIATION.md:229` — *"Keep **16,384**, or 12,288 if headroom is genuinely
needed"* — and `REMEDIATION.md:165`, which claims `8192` is *"in force."*
`git log -S "COMPACTOR_GENERATION_RESERVE" -- runpod.env.template` returns exactly one
commit: the file's creation. **The line has never been touched.** No Dockerfile,
compose file or entrypoint overrides it. The only `16384` anywhere in the tree is
`docker-compose.tokenizer-contract.yml:52`, in a fixture whose own README says
*"GENERATION_RESERVE is not validated here."* The correct value survives only where it
has no effect.

**Failure scenario.** `HARD_INPUT_LIMIT` = 32768 − 2048 = **30,720**. The shed loop at
`main.py:931` stops the instant `running <= limit`, so it deliberately hands vLLM a
payload just under 30,720. The chat path sends no `max_tokens` (`facts.py:86` and
`facts.py:664` both say so in comments; `FRONTEND_SPEC.md:344` makes it optional), so
`req_max_tokens = 0` and the `max(GENERATION_RESERVE, req_max_tokens)` escape at
`main.py:2051` never fires. vLLM then has ~2,048 tokens left to generate. This user's
replies measure **7,513–11,347 tokens**. Every long reply stops mid-sentence, at
`finish_reason: "length"`, at HTTP 200, with no log line anywhere — and P3 then
memorises the fragment.

**The coupling, which I verified myself and which is the reason this is one commit and
not two.** `TARGET_TOKENS = int(MAX_MODEL_LEN * 0.75)` = **24,576**
(`main.py:67`; `COMPACTOR_TARGET_TOKENS` is deliberately unset at
`runpod.env.template:54`). Compaction fires at `main.py:593-594` on
`count_tokens(messages)` — a **local, unscaled** count. Set the reserve to 16,384 and
`HARD_INPUT_LIMIT` becomes 16,384 *true* tokens, which at the measured 1.23x live scale
is ~13,320 local. Compaction does not fire until 24,576 local. **In the entire band
between them the guard drops whole turns while compaction, which would have preserved
them as a summary, never runs.** That is the mechanism behind "shed 60 of 65 turns."
Raising the reserve without moving `TARGET_TOKENS` trades a truncation bug for a
context-destruction bug.

**Fix.** One commit, two lines:

```
runpod.env.template:59   COMPACTOR_GENERATION_RESERVE=16384
runpod.env.template      COMPACTOR_TARGET_TOKENS=12288      # uncomment line 54's slot
```

plus the same pair in `.env.example` and the `main.py:76` default. Add the
`MAX_RETAINED_IMAGES`-style all-caps "NOT COMMENTED OUT, DELIBERATELY" note that
`runpod.env.template:143-148` already uses for exactly this class of live mitigation.
Correct `REMEDIATION.md:165`, which asserts a mitigation that was never committed — and
which the 2026-08-27 margin telemetry in that same document (`REMEDIATION.md:176-180`,
the 2628 → 2755 → 2882 climb) arithmetically excludes from ever having been in force on
the pod either.

**Est 15 min.** Config only, no code path changes.

---

#### P2 · The trim recount drops the scale factor
**S2 by consequence, blocking by provenance.** `main.py:979`

Every entry of `per` is built scaled:

```python
main.py:923    per = [int(count_tokens([m]) * scale) for m in msgs]
```

and the trim block subtracts that scaled value, then adds back an **unscaled** recount:

```python
main.py:977-980
            running -= per[i]
            per[i] = count_tokens([msgs[i]])  # recount ONLY the trimmed block
            running += per[i]
            trimmed += 1
```

`count_tokens` (`main.py:399-430`) is purely local; `scale` never enters it. The
summarizer does the same operation correctly at `main.py:491`.

**Failure scenario, measured not argued.** A verifying agent imported the real
`compactor.main`, stubbed only `count_tokens_exact` to return exactly 1.23x the local
count, and diffed the shipped guard against a copy with the single edit at 979, over
4,000 randomised payloads:

```
outputs differ                                            271 / 4000  (6.8%)
shipped forwards LESS content than the corrected version  267
injected-memory chars destroyed for no budget reason      median 1,189, max 3,606
payload still over limit at the end   shipped 122, fixed 122  (IDENTICAL)
```

Attribution over those 271, read from the guard's own log line: **265 were extra trims**
(line 978 re-reading its own corruption on a *second* halving of the same block), 2 were
extra system drops at line 1005, 0 were extra turn drops. So the dominant mechanism is
line 978, not 1005: on the first trim the error is harmless because `main.py:1018` resets
`running` to ground truth each round, but `per[i]` stays unscaled *permanently*, and the
next `running -= per[i]` under-subtracts by `(scale−1) × per[i]`, overstating `running`
and buying a halving the budget did not need.

**Why it blocks.** `git log -S` puts line 979's text at `f133519` (rc7, pre-scale, where
it was correct) and the `* scale` at 923 at `3d3e732` — the P0-0c fix scaled 923 and did
not update 979. **The remediation whose stated purpose is to stop silently degrading the
user's context shipped a new path that silently degrades the user's context.** It cannot
cause a 400 and cannot lose a turn (`main.py:1018` re-measures every round), so it is S2
on harm. It blocks on provenance.

**Fix.** `per[i] = int(count_tokens([msgs[i]]) * scale)`. The agent verified this
removes the divergence entirely.

The regression test needs care: `test_tokenizer_contract.py:481` is
`check(after <= limit, "the shed payload genuinely fits")`, and **over-shedding satisfies
"fits."** The bug is unasserted, not unreached. The new assertion must be an *upper*
bound — no injected block trimmed while the payload already measured under the limit on
the server — or the next one-sided fits-check will pass over this class again.

**Est 20 min** including that test.

---

#### P3 · A truncated reply is memorised as a finished one
**S1.** `main.py:1115-1116` (stream), `main.py:2251-2276` (non-stream)

```python
main.py:1113-1116
        obj = json.loads(payload)
        choice = obj.get("choices", [{}])[0]
        if choice.get("finish_reason"):
            self._complete = True
```

No value discrimination. `"length"` sets `_complete` exactly as `"stop"` does. The
docstring at `main.py:1129-1133` says the opposite: *"True once a finish_reason or
[DONE] was observed — i.e. the model actually FINISHED the reply."*

`main.py:2180` `if conv_id and not vllm_failed and accumulator.complete():` is the only
gate, and `_async_tail` never re-inspects: the fragment is embedded at `main.py:1248`,
fact-extracted at `main.py:1295`, and appended as a completed assistant turn to the
rollup input at `main.py:1397-1402`. The comment above the call site
(`main.py:2169-2173`) names the harm it fails to prevent: *"memorizing a half-sentence as
though the model said it plants false 'memories' in facts/RAG/summaries."*
The non-stream path at `main.py:2251-2276` has no completeness check at all.

The codebase already knows this signal matters. `dedup.py:445-450`:

```python
        if finish_reason == "length":
            logger.warning(...)
            return None, "truncated"
```

`dedup.py` is the only module in `compactor/` that discriminates on the *value*.

**This is REMEDIATION.md:673 (F20), half-applied.** F20's second half shipped — the
status check now returns before the tail, `main.py:2224-2248` carries the comment
*"v3.1 F20: the status check used to sit after the tail was fired."* The
`finish_reason` half did not. File it as *F20 partially applied*, not as a new find;
the gate that let half of an accepted 30-minute item through is worth a line in the
retro.

**Fix — and the obvious one-liner is a no-op.** Deleting or narrowing
`main.py:1115-1116` does nothing, because `main.py:1109-1111` runs first and
unconditionally on the `[DONE]` sentinel, which vLLM sends after a length-truncated
stream just as it does after a clean one:

```python
main.py:1109-1111
                if payload == "[DONE]":
                    self._complete = True
                    continue
```

So: record the terminal reason on a separate `self._truncated` flag set when
`finish_reason == "length"`, and make `complete()` return False regardless of `[DONE]`.
Apply the same check on the non-stream path before `_fire_and_forget`. Log every length
stop at WARNING with the reserve value — today there is no line for it anywhere.

**Est 45 min.** Note there is no existing coverage to break: `grep -i "SseAccum"` over
`compactor/test_*.py` returns zero hits, and `tests/integration/README.md:192`'s
reference to *"compactor/test_smoke.py::SseAccumulator tests"* is stale.

---

#### P4 · The episodic layer can return nothing and log nothing
**S2.** `main.py:1971`, `retrieval.py:373`, `main.py:1964`

Two independent one-liners, taken together because they share a failure surface.

**(a) The silence.** `format_retrieval_block` returns `None` on an empty list
(`retrieval.py:591-592`), so:

```python
main.py:1968-1971
            rblock = retrieval.format_retrieval_block(hits)
            if rblock:
                injected_blocks.append(rblock)
                log_parts.append(f"{len(hits)}retr")
```

Zero surviving hits emits **no token at all** — the injection line simply gets shorter.
`REMEDIATION.md:886` names *"an `Nretr` field present in the line"* as the verification
for exactly this condition, and it cannot be run. Meanwhile `retrieval.py:425`
(`conversation_doc_count`) counts rows regardless of retrievability, so `/health/full`
and `/admin` both report the memory as present.

**(b) P0-3, the one Phase-0 code item with no work on it at all.**

```python
retrieval.py:372-374
        turn_index = int(meta.get("turn_index", -1)) if meta else -1
        if exclude_turns_from is not None and turn_index >= exclude_turns_from:
            continue
```

`git log -L 370,376:compactor/retrieval.py` returns exactly one commit — `374d455`,
V2.0. Nothing on this branch has touched it. On a fresh or short conversation
`turn_index = len(messages) + 1` (`main.py:1767`) gives
`recent_cutoff = max(0, turn_index - 8) = 0` (`main.py:1964`), `0 is not None` is True,
and **every row with valid metadata is excluded** — only rows whose metadata is missing
survive, because `retrieval.py:372` defaults those to −1. The filter is inverted for
precisely the rows it should distrust most.

Reproduced against the real `retrieval` module with the repo's own mocks:

```
msgs= 2 cutoff= 0 hits=0/3
msgs= 7 cutoff= 0 hits=0/3
msgs=10 cutoff= 3 hits=1/3
msgs=14 cutoff= 7 hits=3/3
```

`REMEDIATION.md:400` already scoped the fix: *"Treat `exclude_turns_from <= 0` as 'no
exclusion'. One conditional."* It is 45 minutes on the estimate sheet and it has been
sitting in the phase headed **"Now."**

**Fix.** `if exclude_turns_from and turn_index >= exclude_turns_from:` at
`retrieval.py:373`; move `log_parts.append` outside the `if rblock:` at `main.py:1971`
so an empty block prints an explicit `0retr`.

**Est 20 min.** The observability half is zero-risk and is what makes every other
retrieval finding diagnosable from a log the owner can read in the morning. Take it even
if nothing else on this list is taken.

Note the conditional repairs the short-array case only. The windowed/drifted case is
**A7** and is deferred; do not record P0-3 as closed.

---

### 3.2 · AFTER v3.1

Ordered by user harm. Two of these carry S1 severity and are deferred anyway; the
justification is stated on each.

---

#### A1 · `_do_l1_rollup` has no input budget — the hierarchy is permanently dead for this user
**S1.** `summarizer.py:404-409`

```python
summarizer.py:403-409
    last_turn = last + L1_CHUNK_SIZE
    body = _format_turns(messages, first_turn, last_turn)
    if not body.strip():
        return False
    text = await _llm_summarize(
        client, vllm_url, model, _PROMPT_L1, body, L1_MAX_TOKENS
    )
```

`summarizer.py` contains **no token accounting of any kind** — its complete import list
(`summarizer.py:45-53`) is `logging, os, datetime, typing, httpx, logsetup, memory`.
Input is bounded by turn *count* only (`L1_CHUNK_SIZE = 20`, `summarizer.py:62`;
`COMPACTOR_L1_CHUNK_SIZE=20` in the template) and never by tokens. `_format_turns`
(`summarizer.py:324-342`) is a bare `"\n\n".join(parts)` with no cap. `main.py:1396-1403`
hands it the **raw, un-trimmed client array** — every trimming stage in the request path
assigns to `body["messages"]`, never to `messages`.

Twenty turns is ~10 assistant replies. At 7,513–11,347 tokens each, the floor is ~75,000
tokens into a 32,768 window; the flat average from the live payload gives ~51,700 for a
20-turn slice. **Deterministic overflow by 1.6–3.5x.**

**And it never recovers.** `summarizer.py:382` `r.raise_for_status()` fires *before* the
watermark write at `summarizer.py:415`, so `last_summarized_turn` never advances,
`_needs_l1_rollup` stays true forever, and the identical doomed POST is re-issued on the
tail of every subsequent turn for the life of the conversation. L1 never grows, so L2 and
L3 never fire either. `_do_l3_rollup` (`summarizer.py:470-473`) joins **all** L2 chapters
with nothing trimming `l2`, carrying the same shape more slowly.

This is the identical class of defect `main.summarize` was rewritten for — compare
`main.py:513-527`, which measures with `count_tokens_exact`, computes a scale, and chunks
with `_chunk_to_budget` before every call. None of that exists in the second module.

**Why it is deferred despite S1.** The exception is caught and logged with a full stack
trace every single turn at `summarizer.py:556-557`. Unlike everything the incidents were
about, **this one is loud.** It is not a silent degradation; it is a traceback per turn
in a log the owner can grep tomorrow. And the fix is not a morning change: `summarizer`
cannot import `count_tokens` from `main` (main imports summarizer), so a real budget
means extracting the counter into a shared module or duplicating it — a refactor, at 6am,
in the module that owns the memory hierarchy. No.

**Fix.** Extract `count_tokens` / `count_tokens_exact` into a `tokens.py` both modules
import, then chunk `body` against `MAX_MODEL_LEN - L1_MAX_TOKENS - reserve` the way
`main.py:513-527` does. **Est 2–3 h.** Add the same bound to `_do_l3_rollup`. Interim, if
the log is not enough: refuse the rollup with a per-conversation ERROR naming the body's
size rather than issuing an unbounded call — 20 min, and it converts a traceback into a
statement.

---

#### A2 · The test suite cannot measure the quantity under review; commit the harness
**S1 (process).** `test_budget_guard.py:47-48`, `REMEDIATION.md:227`

```python
test_budget_guard.py:47-48
# Small, predictable window. No MODEL_REPO -> char/4 token estimator.
os.environ.pop("MODEL_REPO", None)
```

executed before `import main` at line 69. `main.py:65` reads `MODEL_REPO` at import;
`get_tokenizer()` short-circuits at `main.py:123-125`; `count_tokens` falls to
`main.py:430`, the char/4 estimator. Verified by running it: 100 × U+2501 counted **29**
tokens, exactly `len//4 + 4`.

**`REMEDIATION.md:227` is therefore wrong twice over.** It says these tests *"exercised
the `encode()+4` fallback"* and that adding jinja2 gives *"a different token counter than
anything yet tested against."* They never ran `encode()+4` — that branch
(`main.py:426-429`) requires a **loaded** tokenizer whose `apply_chat_template` raises,
which requires `MODEL_REPO` set. `transformers` is never imported. jinja2 is unreachable.
**The prescribed gate action — "re-run every budget test on the v3.1 image" — produces a
byte-identical run and would be recorded as satisfied while proving nothing.** On a
project whose stated failure mode is a green board over a broken budget, a release gate
that cannot fail is itself the defect.

Corroborating, from the mutation pass: **8 of 8 budget-arithmetic mutations survived the
full 26-suite gate**, including `main.py:902 exact = count_tokens_exact(messages)` →
`exact = None` and `main.py:910 scale = ...` → `scale = 1.0`. `count_tokens_exact`
returns `None` on **100%** of its calls in the entire suite; the only behaviour any test
has ever observed from the v3.1 fix is its failure path.

**Fix.** The harness in §6 already exists, uncommitted, and closes most of this. Commit
it, wire it into the release gate, and rewrite `REMEDIATION.md:227` to say what the gate
actually tests: *re-run the budget tests against the served `MODEL_REPO` and record which
tier `count_tokens` took.* **Est 1 h** (commit + gate + doc), plus the fixture-mock fix
at `test_retrieval.py:85`, which ignores `n_results` and so cannot host A8's test.

Deferred only because it changes no shipped code path and the harness has already been
run end-to-end (exit 0, 29 s).

---

#### A3 · `/forget` reports a clean wipe, then memory comes back
**S2.** `main.py:1318`, `main.py:1361-1362`, `main.py:1246`

```python
main.py:1318   combined = _merge_touched(facts.load_facts(conv_id), touched_facts) + new_entries
main.py:1361   if combined:
main.py:1362       facts.save_facts(conv_id, kept)
```

`_merge_touched` (`main.py:1146-1176`) loops over `fresh`, so a post-wipe read of `[]`
correctly yields `[]` — membership is authoritative. But `+ new_entries` is appended
*outside* that logic and `save_facts` writes unconditionally. There is no wipe
generation, no tombstone, nothing between the wipe and the write.

**The window is not sub-second.** `bgwork.py:88-90` is:

```python
    async def _run(self, coro) -> None:
        async with self._sem:
            await coro
```

The semaphore is awaited **before** the coroutine, so a tail past `MAX_CONCURRENT=4` has
executed zero lines of `_async_tail`. A second tail on the same conversation parks at
`main.py:1246` for as long as the first holds `conv_lock` — the extraction at
`main.py:1282` (120 s timeout) plus every LLM call inside `maybe_rollup`.
`asyncio.Lock` is FIFO, so a `/forget` arriving anywhere in that stretch queues behind
the parked tail's 1246 and in front of its 1282: **the losing order, for seconds to
minutes.** `REMEDIATION.md:689` already documents four quick messages saturating the pool
on one lock.

**Reproduced three times** against the real modules:

```
A. /forget replied "Forgot: 3 fact(s)."  -> facts []  -> after drain: ['fact extracted by tail #2']
B. /forget replied "Forgot: 2 fact(s), summary state."  -> after drain: facts back
   AND the summary file back with 2 regenerated L1 chunks
C. bgwork max_concurrent=1: /forget arrives -> vector store wiped -> index_exchange
```

All three layers return from one `/forget` that reported success. This reopens **F10**
(*"/forget reports success when the wipe did not happen"*, `REMEDIATION.md:525`), fixed
on this very branch, through a different door — with a 200 and a counted success body.
The comment at `main.py:1237-1240` asserts a guarantee that taking the lock does not
provide.

**Why the test suite misses it.** `test_concurrency_guards.py:268` covers
`/forget <substring>`, which the merge logic genuinely protects because the deleted fact
lives in `touched`, not `new_entries`. There is no test for the no-arg full wipe. Worse,
the harness `_race_against_parked_tail` (`test_concurrency_guards.py:179-224`) parks the
tail *after* it has taken the lock and re-read — **it constructs the order in which
`/forget` wins and asserts that.**

**Fix.** A per-conversation wipe generation bumped inside `_clear_all_memory` and
compared before `main.py:1362` and `main.py:1246`; or re-read and re-clear after
`pool.drain`. ~10 lines. **Est 1 h** including the losing-order test.

Deferred because it requires the user to issue `/forget` concurrently with in-flight
tails — real, reproducible, but not the first-day path — and because a wipe-generation
edit made in a hurry is exactly the kind of concurrency change that goes wrong.

---

#### A4 · The retrieval cap is denominated in characters, so it does not cap tokens
**S2.** `retrieval.py:559`, `retrieval.py:586-590`

```python
retrieval.py:559    budget = MAX_RETRIEVAL_TOKENS * 4
```

with `MAX_RETRIEVAL_TOKENS` = 1500 (`facts.py:67` mirrors it; template line 76). The loop
sums `len(sep) + len(doc) + 2` and then logs, at `retrieval.py:586-590`, *"within the
{MAX_RETRIEVAL_TOKENS}-token budget"* — **having measured only characters.**

The magnitude is an order larger than the review pair estimated, and it is computable
from the repo's own fixture rather than from the disputed 4.10 chars/token figure:
`test_tokenizer_contract.py:230-235` `DECORATIVE_REPLY` is **2,209 characters** and the
established measurement is **~4,275 vLLM tokens** — a **7.74x** undercount, 0.517
chars/token, not the ~3.3 a prose-tuned multiplier assumes. So the 6,000-character cap
admits **up to ~11,600 vLLM tokens** against a nominal 1,500. That is roughly a third of
the entire input budget.

This is the live regime, not a synthetic worst case. The comment that motivated the cap
(`retrieval.py:60-65`) records: *"Observed in production 2026-08-27 … 3retr → vLLM 400,
33,127 tokens. Compaction had already reduced that request to 9,915 tokens; injection put
it over the window on its own."* **The cap written to stop that recurrence is denominated
in the one unit that cannot see decoration.**

And the guard does not absorb it — it decides who gets evicted, and the oversized block
wins. `main.py:931-946` sheds oldest non-system **turns** first; injected system blocks
are only halved at `main.py:962-981` and only dropped at `main.py:990-1005`. So the guard
deletes real conversation to make room for a block 7x larger than it believes. That
breaks the cap's own stated contract (`retrieval.py:66-67`: *"no injected memory layer
should be able to outweigh the conversation it is meant to support"*).

**Fix.** Count the assembled block through `/tokenize`, not a raised character
multiplier — a multiplier tuned for prose is wrong for decoration by the same 7.74x and
would only move the failure. Independently: `retrieval.py:586-590` must stop asserting a
token figure it measured in characters. **Est 1 h.**

Deferred because it produces silent context loss, not a 400, and because the correct fix
(measure through `/tokenize` from inside `retrieval.py`) touches a module boundary — the
same one A1 needs. Do A1 and A4 in one sitting.

*Adjacent, same unit error, bounded and not independently fatal:* `main.py:979` — that is
P2, already before the push.

---

#### A5 · Episodic `turn_index` drifts from the client's array, silently and cumulatively
**S2.** `retrieval.py:285` vs `main.py:1964`; introduced by `59a638e` (D1)

```python
retrieval.py:285   return max(highest + _TURN_INDEX_STEP, int(seed))    # _TURN_INDEX_STEP = 2
main.py:1767       turn_index = len(messages) + 1
main.py:1964       recent_cutoff = max(0, turn_index - (KEEP_RECENT_TURNS * 2))
retrieval.py:372   turn_index = int(meta.get("turn_index", -1)) if meta else -1
```

`retrieval.py:372-373` compares a **stored** ordinal, allocated monotonically from the
store's own maximum, against a cutoff built from the **client's current array length**.
Two authorities, one comparison. `git log -S"_next_turn_index"` returns exactly one
commit — `59a638e`, this branch. Pre-D1 the code wrote the raw client value, same units,
no drift.

Two triggers, both ordinary:

- **Deletion or truncating edit.** Executed against the real module: 10 exchanges, no
  deletion → drift 0. Delete 8 client messages → drift 8 *immediately*, and at every
  subsequent turn. Delete 6 more → drift 14, cumulative. Measured suppression: at drift 8
  the filter excluded 8 of 15 rows where 4 was intended; at drift 14, 11 of 19.
- **Regeneration** (`Y4`), which is one click. The regenerated reply is a new content
  hash, `_id_exists` (`retrieval.py:311`) misses, `_next_turn_index` returns seed+2, and
  the client's array did not grow. Measured: one regenerate at exchange 1 → drift +2,
  still present nine exchanges later. Four regenerations over 30 exchanges → drift 8, and
  exchanges 23–26 — **not** in the verbatim window — become silently unreachable.
  Worse: the **discarded** reply gets the lower ordinal and the **kept** one the higher,
  so the recency filter preferentially preserves the reply the user rejected.

The offset is sticky, not mathematically permanent — an un-indexed turn (a slash
command, `main.py:1853-1859`; a stopped stream, `main.py:2176-2180`; an identical
regeneration) hands the seed two units of catch-up. But under normal chatting every
deletion raises it and almost nothing lowers it.

Invisible: `main.py:1252` logs the *tail's* local `turn_index`, never the ordinal
actually written. Nothing in the log stream can distinguish a drifted store from a clean
one.

**Fix.** Store the client's `turn_index` as a second metadata field beside the monotonic
ordinal and filter on that; or derive `recent_cutoff` from the store's own maximum. The
regeneration half additionally needs a supersession model — an `exchange_key` over the
user half plus a `superseded` flag filtered in `retrieve` — because `forget_conversation`
(`retrieval.py:388-400`) is the only delete path and is all-or-nothing, so rejected
replies persist as permanently injectable rows. **Est 3 h** for the ordinal fix, and
supersession is a separate design item (**A15**).

Deferred because it degrades recall rather than destroying data, and because D1 remains
a large net improvement over the pre-D1 overwrite. It is the root cause under A7 and A8
and should be scheduled first among the three.

---

#### A6 · P0-3's deferred half: the cutoff is still client-derived
**S2.** `main.py:1964`, `retrieval.py:373`

P4's one conditional repairs `cutoff == 0` only. Under a bounded client window the cutoff
is a positive number the conditional does not touch. Reproduced: window of 20 messages,
stored ordinals `[21,23,…,43]`, cutoff 13 → **0 of 12 hits survive, on every request,
indefinitely.**

`REMEDIATION.md:397` deferred this explicitly: *"The full fix — deriving the cutoff from
stored turn identity rather than the client's array length — lands with D1 and is
deferred."* **D1 landed in `59a638e`. The deferred half did not.** It is not tracked
anywhere: `_next_turn_index`'s docstring (`retrieval.py:245-254`) and its test
(`test_retrieval.py:475-494`) reason only about the store's max falling *below* the
cutoff — over-inclusion, a budget cost. Neither considers the inverse, which is the one
that fires under a bounded window and costs recall entirely.

**Second, independent unit defect found here.** `KEEP_RECENT_TURNS = 4` (`main.py:68`) is
used as a count of *messages* at `main.py:573` (`non_system[-4:]`, i.e. 2 exchanges) and
as message-units needing a `*2` at `main.py:1964` (8 units, 4 exchanges). **The exclusion
window is twice the verbatim-preserved window**, so exchanges 5–8 message-units back are
dropped from retrieval even though compaction has already summarised them away and they
are *not* in the request verbatim. The justification comment at `main.py:1961-1962` is
false for half the range it excludes.

**Fix.** Reconcile the unit, then derive the cutoff from stored identity (same change as
A5). **Est 1 h** on top of A5.

Note `FRONTEND_SPEC.md:219` commits to *"Send a bounded window"* as the direction, so
this is the committed future client shape. It is not yet the current one — see §7 Q4.

---

#### A7 · Retrieval never over-fetches, so a filtered slot is a deleted one
**S2.** `retrieval.py:355-359`, `retrieval.py:370-380`

```python
retrieval.py:355-359    res = _chroma_collection.query(
                            query_embeddings=vecs,
                            n_results=max(1, k),
                            where={"conv_id": conv_id},
                        )
```

with `RAG_TOP_K = 5` (`retrieval.py:51`). Exactly five candidates return; the loop at
`:370-380` `continue`s past excluded rows and falls straight to `return out`. There is
one query call in the body and no re-query, so **every filtered row is a lost slot, not a
replaced one.** Under A5's drift or A6's window divergence the block empties completely.

The test suite provably cannot catch it: the fake collection at `test_retrieval.py:85`
takes `n_results` and never uses it, returning every conv-matching row, so
`test_retrieve_excludes_recent_turns` passes identically whether over-fetch exists or
not. Same blind-spot class as the char/4 tokenizer — **the mock is more generous than
production.**

**Fix.** Query `n_results = max(1, k + EXPECTED_EXCLUDED)` (bounded), filter, then
`out = out[:k]`. **Est 1 h.** Do **not** ship this before fixing the mock at
`test_retrieval.py:85`, or the fix is as unverifiable as the bug. P4(a)'s `0retr` log is
the sufficient, zero-risk half and is already before the push.

---

#### A8 · The calibration loop measures overshoot against the wrong limit
**S2.** `main.py:257`, `main.py:2132`, `main.py:2248`

```python
main.py:257-258     effective_limit = max(256, HARD_INPUT_LIMIT - _BUDGET_MARGIN)
                    overshoot = actual - effective_limit
```

but the limit the guard actually enforced is computed per-request:

```python
main.py:2049-2052   effective_limit = min(
                        MAX_MODEL_LEN,
                        max(256, MAX_MODEL_LEN - max(GENERATION_RESERVE, req_max_tokens)),
                    )
main.py:2075        enforced_limit = max(256, effective_limit - _BUDGET_MARGIN)
```

and `main.py:2132` passes only `err_body`. The correct number is computed **two lines
away** and not passed. Since `max(GENERATION_RESERVE, req_max_tokens) >= GENERATION_RESERVE`,
`effective_limit <= HARD_INPUT_LIMIT` always, so the error is one-directional: it **only
ever understates** the overshoot. A sufficiently large understatement yields
`overshoot <= 0`, no advance, `tightened=False`, and
`_rejection_user_message(err_body, False)` tells the user, falsely, that retrying will
not help.

Worked: `max_tokens=8192` → `effective_limit` 24,576. Prompt 25,000 rejected.
`overshoot = 25000 − 30720 = −5720`, not > 0 → margin never moves. Correct overshoot was
424, i.e. a 936-token margin that would have healed it next message. **Dead band =
`HARD_INPUT_LIMIT − effective_limit`, up to 14,336 tokens at the shipped reserve.**

`main.py:2248` is a second entry point that discards the return value entirely.

**Reachability is a config question, and P1 closes it.** At `GENERATION_RESERVE=16384`,
`req_max_tokens` is clamped to `MAX_MODEL_LEN//2 = 16384` at `main.py:2045-2047`, so
`max(GENERATION_RESERVE, req_max_tokens)` can never exceed the reserve and **the defect
becomes unreachable.** That is why it is here and not before the push — P1 mitigates it
as a side effect. Fix it anyway, because the mitigation is incidental.

`REMEDIATION.md:169-186` prescribes line 257 verbatim, including the wrong referent, even
though its own text notes the guard's limit is a *parameter*. Correct the doc too.

**Fix.** Pass `enforced_limit` into `_note_backend_rejection` at `main.py:2132` and
`2248`; use it at `257`. **Est 30 min.**

---

#### A9 · No log line anywhere names which counter made the decision
**S2.** `main.py:903`, `main.py:911`, `main.py:1046`, `main.py:522-532`

```python
main.py:903    total = exact if exact is not None else local_total
main.py:910    scale = (total / local_total) if (exact is not None and local_total > 0) else 1.0
main.py:911    if exact is not None and abs(scale - 1.0) > 0.05:
```

**The only counter-naming INFO line is structurally unreachable in precisely the state it
would diagnose.** When `/tokenize` refuses, `scale` is forced to exactly 1.0, the shed
runs on the tokenizer that reads 34–51% low, `main.py:1017-1018` falls back again
unmarked, and `main.py:1046 logger.warning(f"hard budget enforced: {detail}")` emits a
line textually indistinguishable in shape from the healthy case. HTTP 200.

The two sites that *do* name the counter — `_sent_token_size` at `main.py:1641-1642`,
rendered at `main.py:1676-1678` — are called from exactly two places, `main.py:2124` and
`main.py:2242`, **both gated behind `if r.status_code >= 400`.** A shed at HTTP 200
reaches neither. The 2026-08-28 signature *was* a shed at HTTP 200.

There is one discriminator: `main.py:381` / `main.py:391` warn when `/tokenize` misbehaves
— but both sit behind `logsetup.log_once`, whose `_logged_once` set is *"deliberately
never cleared"* (`logsetup.py:117-137`). Accurate framing is **"one process-lifetime
line, then silence,"** which `REMEDIATION.md:225` already concedes.

**Missed and more load-bearing: the summarizer has the identical defect.**
`main.py:522-527` computes `_scale` and **never logs it on either branch**, while
`main.py:532`'s INFO reports the batch count without saying which counter sized the
batches. Silent fallback to `scale=1.0` there is verbatim the 2026-08-28 mechanism
(batches believed 29,696, really ~46,000), and `REMEDIATION.md:213` calls the summarizer
half *"the decisive one."*

**Fix.** Thread `src = 'vLLM' if exact is not None else 'local'` through `detail` at
`main.py:1026-1046` and through the verify step at `main.py:1017-1018`; log the scale
line unconditionally, including `token scale unavailable (/tokenize refused)`; cover
`main.py:522-532` too. Add the boot-time `/tokenize` assertion to `selftest.py` and a
field to `/health/full` (`health.py:52` probes only `/v1/models`; `grep tokenize
compactor/health.py` is empty) so the endpoint's state is a startup fact, not a log line
from three days ago. **Est 1 h.**

`FRONTEND_SPEC.md:15`'s Received-context echo is marked REQUIRED and already states token
figures *"must come from vLLM's `/tokenize`, not a local estimate."* Whoever builds it
must carry the counter's identity into the echo, or this defect is reproduced on the
client surface.

---

#### A10 · `_BUDGET_MARGIN` is process-global, monotonic, and never released
**S2.** `main.py:195`, `main.py:260-262`, `main.py:860-861`

The complete reference set is lines 195, 233, 248, 257, 261, 262, 860, 861, 1443, 1453,
2071, 2075. Only two are assignments — 195 (init) and 262 (ratchet up). There is no
reset, no decay, no per-conversation scoping anywhere in the package. `supervisord.conf:87`
is `uvicorn main:app` with **no `--workers`**: one process, one global, all conversations.

```python
main.py:260-262     new_margin = min(overshoot + 512, MAX_MODEL_LEN // 4)
                    if new_margin > _BUDGET_MARGIN:
                        _BUDGET_MARGIN = new_margin
main.py:860-861     if _BUDGET_MARGIN:
                        limit = max(256, limit - _BUDGET_MARGIN)
```

Trigger is unconditional and does not depend on any unverified premise: `main.py:929-931`
`if len(idxs) <= 1: break  # always keep the most recent turn`, then `main.py:1028-1035`
forwards anyway. That path fires, 400s, and ratchets.

**Three corrections to how this has been described**, because the severity arithmetic
matters: (1) At today's deployed config `HARD_INPUT_LIMIT` is 30,720, so a fully
ratcheted margin leaves 22,528 — a **27% cut, not a halving**. The halving only
materialises once P1's 16,384 reserve lands. (2) The harm is **band-limited**:
`main.py:916-917` returns untouched when `total <= limit`, so only conversations whose
true count falls in the top 8,192-token band pay. (3) Post-P0-0b it **latches in one
event**, it does not crawl — a single oversized turn with overshoot ≥ 7,680 jumps
straight to the 8,192 ceiling on its first 400. That makes it sharper than the "gradual
climb" framing, not milder.

The scope contradiction is real: `main.py:1453` says *"for this process"*; `main.py:263-269`
says *"the next message in this conversation should succeed."*

**Fix.** Scope per `conv_id` in a bounded LRU, or decay (halve on N consecutive
successes); clamp to a fraction of the *current* effective limit; report it in
`/health/full` and in the shed line; correct the `main.py:263-269` wording. **Est 1.5 h.**
Not blocking: a degraded-mode backstop, capped, cleared by restart.

---

#### A11 · `/health/full` reports `ok` while background work is shedding
**S3.** `health.py:233-242`

```python
health.py:233        bg = bgwork.pool.stats()
health.py:237-242    if not storage["ok"]:            status = "down"
                     elif not vllm["ok"] or writes.get("new_memory_writes") == "paused":
                                                      status = "degraded"
                     else:                            status = "ok"
```

`bg` is computed, placed in the payload at `health.py:261`, and **never consulted for
`status`.** `degrade.write_state()` (`degrade.py:132-141`) reports disk space only.
`status_to_http_code` (`health.py:274`) returns 200 for `degraded` anyway. Sustained
shedding reports `ok` and the Docker HEALTHCHECK passes.

Its real significance is as an amplifier of A1: every tail carries a doomed multi-second
50k-token POST, so each of four `MAX_CONCURRENT` slots is held far longer than it should
be, pushing `outstanding` toward the ceiling — and the shedding that follows is hidden.

**Fix.** Degrade `status` when `pool.stats()` shows shedding; log `conv_id` in
`bgwork.py:76-81`'s shed warning. **Est 30 min.** (Note the shed warning cannot name
`conv_id` at `bgwork.py:70-75` — `submit` receives an opaque coroutine — so the caller
must pass it.)

---

#### A12 · Four blocking calls on the event loop, and an O(N) health scan that reads full documents
**S3.** `main.py:1909/1931/1965/1979`, `main.py:593`, `retrieval.py:425`

`persona.auto_capture_persona`, `facts.load_facts`, `retrieval.retrieve` and
`summarizer.load_state` all sit directly in `async def chat_completions` with no offload
(the complete `run_in_threadpool` inventory in `main.py` is lines 524, 526, 1440, 2062,
2123, 2241 — none of these). `main.py:1863 → compact_if_needed → main.py:593
count_tokens → main.py:129 AutoTokenizer.from_pretrained` is a fifth, and nothing between
`main.py:1719` and `1863` warms the tokenizer. `supervisord.conf:87` has no `--workers`,
so this is one loop shared with the health probe.

```python
retrieval.py:425    existing = _chroma_collection.get(where={"conv_id": conv_id})
```

No `include=[]`, called once per conversation from `health.py:158`, over an uncapped
`memory.list_known_conv_ids()`, from a sync `gather_memory_stats()` inside an async
handler, on a 30 s HEALTHCHECK. That default really does pull full document text out of
SQLite to answer a `len()` — proven from inside the repo rather than from a comment:
`retrieval.py:455-458` issues the identical no-include call and then consumes
`.get("documents")` and `.get("metadatas")`, so export would be broken otherwise.
`include=[]` is already proven valid at `retrieval.py:222`.

**One correction that changes the fix.** The Docker HEALTHCHECK (`Dockerfile:385`) hits
`/health/full` every 30 s from container start, *through* the 300 s start-period, and
`conversation_doc_count` calls `_try_init()` at `retrieval.py:422` — so the health probe
almost certainly pays the fastembed/Chroma cold init long before the user's first
message. `health.py` never imports `count_tokens`. **The first-message stall is
`AutoTokenizer.from_pretrained`, not fastembed.** So the warm-up worth adding to
`lifespan` (`main.py:1424-1457`, which today warms only `backend_is_multimodal`) is a
**tokenizer** pre-warm.

**Also: `REMEDIATION.md:723` is affirmatively wrong and should be reopened.** It kills
F33's loop-blocking claim on the grounds that *"`_enforce_hard_budget` has been in
`run_in_threadpool` since `main.py:1436`."* F33 is about `get_tokenizer` retries, which
block the loop via the **compaction** entry point at `main.py:1863`, not the guard entry
point at `main.py:2062`. `REMEDIATION.md:997` repeats it in the residuals table.

**Fix.** `include=[]` at `retrieval.py:425`; `run_in_threadpool` the four sync calls;
tokenizer pre-warm in `lifespan`; reopen the two REMEDIATION lines. **Est 45 min.**
Latency only, no incorrect output, single-user system.

---

#### A13 · The `/tokenize` failure warning is spent once per process
**S3.** `logsetup.py:114-137`, `main.py:380`, `main.py:390`

`main.py:380 if logsetup.log_once("count_tokens_exact.http"):` — the key encodes neither
the caller nor the status code, and `_logged_once` (`logsetup.py:114`) is module-level and
never cleared outside tests (`_reset_log_once_for_tests` is referenced only from
`test_*.py`; `test_logsetup.py:121` asserts the set survives `configure()`).
`count_tokens_exact` has four callers — `main.py:524` (summarize), `:902` (guard ground
truth), `:1017` (per-round verify), `:1639` (`_sent_token_size`) — and one token covers
all of them for the process lifetime.

**The aggravator.** `main.py:377-379`'s own comment says a 400 here *"is usually the
template refusing the message shape (an assistant-final list, most often) rather than a
fault."* The summarizer at `:524` passes older turns, which routinely end on an assistant
turn. **So the most likely first spender is a benign structural 400, on any conversation
long enough to compact, and it permanently silences the report of a genuinely broken
endpoint.** The silencing is close to structural, not incidental.

This is a missing signal, not a harm — `log_once` adds to a set and returns a bool that
guards only `logger.warning`; there is no behavioural coupling.

**Fix.** Give this call site a rate limit with a recovery line (*"/tokenize is answering
again"*) rather than a one-shot; use a distinct `count_tokens_exact.http.400` key so a
structural refusal does not consume the transport-failure signal; surface a
consecutive-degraded-request counter in `/health/full` so it is visible without a log
line at all. **Est 30 min.** Pairs naturally with A9.

---

#### A14 · The guard's prescreen skips all measurement, on a margin justified by the discredited oracle
**S3.** `main.py:894`, `main.py:865-883`

```python
main.py:894    if _fast_token_estimate(messages) <= limit // 2:
                   return messages
```

`count_tokens_exact` is first called eight lines later. Below the prescreen threshold,
**no component on the request path ever asks what the payload costs** — `compact_if_needed`
cannot catch it either, since it triggers at 24,576 local and a skipping payload is ~8k.

And the justification is provably a pre-P0-0c artifact: `git log -L 860,900:compactor/main.py`
shows the comment block at `main.py:865-883` was introduced by `9490303`, and the *very
next* commit `3d3e732` changed the statement immediately following it — `total =
count_tokens(messages)` became the `count_tokens_exact` chain — leaving the prescreen and
its comment untouched. The 2x margin's stated safety argument rests on the oracle P0-0c
discredited.

**But the claimed exhaustion does not hold, and the corrected arithmetic is why this is
S3 and not S2.** `_fast_token_estimate` (`main.py:701-704`) is
`sum(len(...) // 4 + 4 for m in messages)` — there is a **+4 per message**, which the
original claim dropped, so its own worked example fails the skip test. Against the four
direct chars-to-vLLM pairs at `main.py:343-346`:

```
assistant  17,930 / 8,988  = 1.995   <- worst of three, below break-even by 0.25%
assistant  16,971 / 7,513  = 2.259
assistant  27,570 / 11,347 = 2.430
user        6,865 / 1,585  = 4.331
aggregate  69,336 / 29,433 = 2.356
```

The real safety condition is roughly `chars/vLLM-token >= 2.0` at short conversations,
rising to ~2.1–2.5 at long ones once the unmodelled per-message framing (~22 tokens on
Mistral, `main.py:412`, vs the 4 charged) is included. **The margin is 1.18x–1.67x on
measured content, not zero.** No measured payload crosses. A crossing is constructible in
the medium band (M≈10–40, ~800–3,000 chars/turn, a few decorative rules per reply) but has
not been observed.

And the failure is **not silent**: the resulting 400 is logged at ERROR
(`_log_request_rejected`, `main.py:1051+`), surfaced as a typed `context_length_exceeded`,
and `_note_backend_rejection` widens `_BUDGET_MARGIN`, which is subtracted at
`main.py:860-861` *before* the prescreen reads `limit // 2` — **the prescreen self-tightens
after one failure.**

**Fix.** `limit // 8` makes the safety condition 0.5 chars/vLLM-token, covering even pure
decoration, and still skips `/tokenize` for the small payloads the prescreen exists to
protect. Restate the comment in chars-per-**vLLM**-token. **Est 20 min.** Take it in this
remediation on principle — it is the last place on the request path where an irreversible
forward/measure decision is made from the discredited estimate.

---

#### A15 · Regeneration supersession (design)
**S3 (design item).** `retrieval.py:159-161`, `:208-209`, `:311`, `:388-400`

The `Y4` half of A5, split out because it is not an edit. Identity is
`f"{conv_id}::{sha256(document)[:16]}"` over `[user]:…\n[assistant]:…`, so a regenerated
reply is a different row, `_id_exists` misses, and the rejected text persists forever —
`forget_conversation` is the only delete path and is all-or-nothing (`grep '.delete('`
over `compactor/` returns exactly one hit). Needs an `exchange_key` over the user half
plus a `superseded` flag filtered in `retrieve`. **Est: design first, half a day.**

Verification cheap enough to run today:
`retrieval.export_indexed_exchanges(conv_id)` on a real conversation — if the stored
ordinals are not `seed, seed+2, seed+4, …` in step with the exchange count, the drift is
there. `conversation_doc_count` exceeding the exchange count is the same signal, cheaper.

---

## 4. Investigated and found clean

Reported explicitly, because a review that lists only problems says nothing about
coverage. Each of these was a live hypothesis that a verifier tried to confirm and could
not.

| Area | Verdict | Evidence |
|---|---|---|
| **The two post-guard merge passes** | Clean. Both strictly *reduce* token cost. `main.py:694` joins string contents, removing one message's ~22-token framing for two characters. The image branch at `main.py:806-808` is neutral because image accounting is per-image, not per-message (`main.py:414-415`, `main.py:701-703`). The ordering at `main.py:2061-2067` is arithmetically safe. | re-verified twice |
| **Fact-extraction budget** | Clean. `facts.py:672 if _assembled_tokens(...) <= budget: return …` against 32768−256−2048 = 30,464. This user's exchanges top out near 11k real tokens; a 2x counter error cannot overflow it. | `facts.py:108-110`, `:672` |
| **The summarizer's state load** | Clean. `summarizer.py:515-516` — `async with conv_lock(conv_id):` **then** `state = load_state(conv_id)`. Load is inside the lock. The lost-update mechanism claimed here does not exist. | `summarizer.py:515-516` |
| **`_reconcile_watermark` cannot trigger a rollup** | Clean. It sets `last_summarized_turn = current_turn_count`, after which `_needs_l1_rollup` evaluates `0 >= 20` = False. The repo's own test asserts it: `test_summarizer.py:566 assert_eq(calls, [], "repairing the counter does not re-summarize anything")`. A `k`-message deletion drains `ceil(k/20)`, i.e. **nothing for k < 20**. | `summarizer.py:236-237, 270-272` |
| **The summarize drain loop terminates** | Clean. Each success advances the watermark by exactly `L1_CHUNK_SIZE`; when `first_turn` passes the array end, `_format_turns` returns `""` and `summarizer.py:406` breaks. Bounding it is cosmetic. | `summarizer.py:406, 415` |
| **The reduce round cannot overflow** | Clean, and this is the sharpest kill in the set. `_summarize_once` posts `"max_tokens": SUMMARY_MAX_TOKENS` = 1024 (`main.py:459`), so every element of `parts` is **truly** capped at ~1024 tokens by vLLM's own generation limit. Budget is 29,696. Overflow needs ≥29 batches ≈ 860,000 true tokens of history, ~6x the payload that took production down. `scale` distorts the estimate, never the true size. | `main.py:459, 515-518, 545` |
| **Extraction-disabled branch of the tail** | Clean. `main.py:1261-1269` — `merged = _merge_touched([], touched)` is `[]` and `if merged:` gates the write. | `main.py:1261-1269` |
| **`_merge_touched` membership logic** | Clean. It loops `for f in fresh`, so a post-wipe read of `[]` correctly yields `[]`. A3's defect is the `+ new_entries` outside it, not this. | `main.py:1146-1176` |
| **Docker healthcheck restart loop** | Clean. `Dockerfile:385-386 HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3`. Failures during start-period do not count, and three *consecutive* 10 s timeouts are needed after. No restart loop from a slow health scan. | `Dockerfile:385-386` |
| **`--tokenizer-mode mistral`** | Clean — not set. `supervisord.conf:68` and `runpod.env.template:39` carry no such flag, so the hypothesised tokenizer-mode divergence does not exist on this deployment. | `supervisord.conf:68` |
| **Image retention interaction** | Clean, and dormant. `runpod.env.template:148` sets retention 0, so `main.py:740`'s keep-set is empty and `text_only == to_summarize` always. (Separately, retention 0 stripping the *current* turn's image is `V11`/D56 and remains a live, user-visible defect — it is just not a budget defect.) | `main.py:738-740` |
| **All 17 compactor modules import cleanly at HEAD** | Clean. Verified by hand under Python 3.14.7 from inside `compactor/`. No live breakage of the class that has cost five deployments — only no guard to keep it that way (P0-0e, unimplemented). Note the command prescribed at `REMEDIATION.md:272` cannot work: there is no `compactor/__init__.py` and modules import each other flat, so it must run from inside `compactor/`. | manual |
| **`F10`'s unreadable contract** | Clean and correctly applied. `main.py:2464-2478` and `commands.py:215-227` do distinguish a read failure from nothing-stored. A3 reopens the *lie* through a different door; it does not undo this. | `main.py:2464-2478` |
| **The status-check reorder (F20 second half)** | Clean and shipped. `main.py:2224-2248` returns before the tail on `r.status_code >= 400`, with an in-place comment citing v3.1 F20. | `main.py:2224-2248` |

---

## 5. Killed during verification

Twelve claims, each of which would have sent someone at the wrong file. Recorded because
this project has repeatedly lost days to confident wrong claims.

1. **"The reduce round is the surviving half of P0-0c."** Refuted — see §4. `main.py:556`
   is a one-line hygiene inconsistency, not a live incident mechanism. Passing `_scale`
   is still worth doing; it is not a blocker and must not be described as one.
2. **"`_reconcile_watermark` triggers the permanent 400."** Refuted. It cannot start a
   rollup at all. Acting on the claim as written sends a fixer to the wrong function.
   The real defect (A1) is **unconditional** — a fresh conversation reaching turn 20 with
   no reset ever having occurred sends the same oversized body.
3. **"The drain loop drains without bound."** Refuted — it terminates. Bounding it is
   cosmetic.
4. **"`/forget` has only a few hundred milliseconds of exposure."** Refuted, and this was
   the whole basis of a non-blocker verdict. `bgwork.py:88-90` awaits the semaphore before
   the coroutine, and FIFO lock ordering makes the window seconds to minutes. The episodic
   sub-case is the **widest** of the three, not the narrowest.
5. **"Line 1005 dominates the trim corruption."** Refuted by measurement: 265 of 271
   divergences came from line 978 re-reading its own corruption; line 1005 scored 2.
   Within a single round the two errors **cancel exactly**; 1005 only contributes across
   a round boundary. Also refuted: *"unreachable in the suite"* — `count_tokens_exact` is
   stubbed at `test_budget_guard.py:1267`, and `test_tokenizer_contract.py` runs the guard
   against a real `/tokenize` where scale ≠ 1.0. The bug is **unasserted, not unreached**,
   which changes the required test.
6. **"The ratcheted margin halves the budget, on every request, crawling upward."** Three
   errors. At today's config it is a 27% cut; the harm is band-limited to the top 8,192
   tokens; and post-P0-0b it latches in one event rather than crawling. The 2628 → 2755 →
   2882 crawl was the *pre-fix* defect.
7. **"Ranking correlation is what empties the retrieval block."** Demoted to secondary.
   The dominant driver is the unit ratchet at `retrieval.py:285` vs the client-derived
   cutoff, which needs no correlation at all — it fires even when the top-5 are genuinely
   old exchanges, because "recent" is a bare ordinal comparison.
8. **"The turn_index offset is mathematically permanent."** Refuted. `retrieval.py:285` is
   a `max`, so any client-counted-but-un-indexed turn hands the seed 2 units of catch-up
   (slash commands `main.py:1853-1859`; stopped streams `main.py:2176-2180`; identical
   text `retrieval.py:311`). Confirmed by execution: after drift 14, seven un-indexed
   turns closed it to exactly 0. Sticky and cumulative, not irreversible.
9. **"Background work starves; new memory stops process-wide, indefinitely."** Refuted.
   `bgwork` head-of-line blocking is real, but the lock holder is itself a pool task,
   `asyncio.Lock` is FIFO, and nothing waits on a holder outside the pool. Forward progress
   is guaranteed; the cost is latency. The composed "permanent stop" needs sustained
   traffic to 64 outstanding and is unproven.
10. **"D1 made the windowed-client exclusion permanent rather than intermittent."**
    Refuted by running the pre-D1 control: a windowed client was already fully excluded
    before D1 — it just had one surviving row to lose, because the id collapse had destroyed
    the rest. **D1 did not create the permanence.** The accurate statement is sharper: D1
    stopped the store destroying rows, and A6 is now the *sole* remaining reason a windowed
    client sees zero recall.
11. **"The prescreen has no margin left and fails silently."** Both wrong. The margin is
    1.18x–1.67x on every measured aggregate (the claim dropped the `+4`-per-message term,
    so its own example would have been measured, not skipped), and the outcome is a logged
    ERROR, a typed client error, and a prescreen that self-tightens. S3, not S2 — see A14.
12. **"The tokenizer-mode / image-retention / extraction-budget / summarizer-lock /
    healthcheck-restart hypotheses."** All five refuted; see §4. The withdrawal record was
    accepted in full and needs no third round.

Two smaller corrections worth carrying so they are not re-derived:

- `_reported_prompt_tokens` (`main.py:196`) parses vLLM's rejection with
  `r"prompt contains (?:at least )?(\d+) input tokens"`. **vLLM 0.10.0 does not emit that
  wording** — `serving_engine.py:625-631` emits *"you requested M tokens (K in the
  messages, J in the completion)."* Measured against the fixture: legacy wording → parses
  41,934; v0.10 wording → returns `None`. `_is_context_overflow` still classifies both
  correctly, so the user-facing message is unaffected, but **A8's calibration would learn
  nothing from a rejection** under the newer wording. Which wording `vllm==0.24.0`
  (`Dockerfile:78`) emits is **not verified** — see §7 Q2. Do not treat this as a
  confirmed defect.
- `REMEDIATION.md` carries at least five stale or wrong statements that will misdirect a
  future implementer: `:165` (a mitigation never in force), `:227` (the release gate
  premise, see A2), `:467` (instructs moving `save_state` *inside* the rollup try, the exact
  opposite of what shipped and re-introducing MEMORY_REVIEW S-3 — `summarizer.py:558-577`
  has an in-place comment explaining why it is outside), `:723`/`:997` (F33 killed on the
  wrong entry point, see A12), `:748`/`:770` (describes D1 as unshipped; it landed in
  `59a638e`), `:465` (names a `(status, data)` / `PRESENT|ABSENT|UNREADABLE` contract that
  does not exist — `memory.py:285-310` returns the value and raises `StoreUnreadable`).
  **Est 1 h** to reconcile the document. Do it before anyone works from it again.

---

## 6. The test harness

### What was built

A **vLLM-shaped tokenizer fixture**, not vLLM. Uncommitted at HEAD:

```
?? compactor/test_tokenizer_contract.py
?? docker-compose.tokenizer-contract.yml
?? testfixtures/tokenizer-contract/{fixture_server.py,Dockerfile,Dockerfile.testrunner,requirements.txt,README.md}
```

vLLM's own CPU build was tried and **measured out**, not assumed away:
`public.ecr.aws/q9t5s3a7/vllm-cpu-release-repo:v0.10.0` pulls (4.04 GB on disk) and
`import vllm` works, but `import vllm._C` exits **132 = SIGILL** — the prebuilt wheel is
AVX-512 and this host reports only `avx2`/`sse4_2`. A real `vllm serve` exited 132 after
42 s. `vllm/vllm-openai:latest` is 8.63 GB and needs a GPU. Building from source with
`VLLM_CPU_DISABLE_AVX512=true` is a multi-hour compile per image refresh.

So: a stub, but one whose request/response shapes were **dumped from vLLM 0.10.0's own
`protocol.py` inside that image** rather than guessed, including the two behaviours that
matter — `/tokenize` does no context validation (so the guard can measure a payload too
big to send), and `/v1/chat/completions` rejects on `token_num + max_tokens > max_model_len`
with vLLM's own wording. The bug class is reproduced by making the two sides genuinely
different tokenizers, as in production: `gpt2` locally (which also ships **no chat
template**, forcing `count_tokens` into the tier-2 `encode()+4` branch) against
SmolLM2-135M-Instruct on the server. A second regime runs char/4, i.e. exactly
`test_smoke.py`'s conditions.

Fault injection: `/_fixture/mode` = `ok | wrong | http_error | garbage | hang`, plus
`/_fixture/stats` so a test can assert the guard makes a **bounded** number of `/tokenize`
calls rather than one per message.

### How to run it

```
docker compose -f docker-compose.tokenizer-contract.yml up --build --exit-code-from contract-tests
```

**29.2 s** warm, exit 0 (~66 s of image build cold). ~760 MB of images. No GPU, well under
1 GB RAM. Tokenizers are baked in at build time, so containers need no network at run
time. Pointed at a missing fixture it **skips cleanly with a loud message** that says the
skip is not neutral.

### What it proves

Both outages, at the line where each went wrong.

- **The 2026-08-28 summarizer.** Test [7] packs the same turns with `scale=1.0` — the
  literal pre-v3.1 call — and measures each batch on the server: largest **49,223 against
  a 29,696 budget, +19,527 over.** Test [6] asserts the fixed path: largest 28,144.
- **The 2026-08-28 guard.** Test [9] stubs `count_tokens_exact` to `None`, runs the guard,
  and measures its **output** on the server: **14,081 against a limit of 8,000** in the
  gpt2 regime, 42,185 in the char/4 regime. The guard certified a payload it had just shed,
  and it did not fit. Test [8] asserts the fixed path lands at 7,055; test [17] posts that
  exact payload and gets a 200.
- **The 2026-08-24 root cause.** Test [4] takes the real production reply shape — 1,710 ×
  U+2501 plus 441 × U+2500 — and prints both counts: **local 1,786 vs server 3,523 (1.97x)**
  with gpt2; **local 556 vs 3,523 (6.34x)** with char/4. Test [3] is the control that makes
  the point land: on prose the two agree within 2%. *A budget test written on prose passes
  and proves nothing.*
- **Degradation.** Tests [11]–[13] cover `/tokenize` as a refused connection, a 500, a 400,
  a 200 with no `count` key, and a hang past the 10 s read timeout — each asserting it
  returns `None` rather than raising, that the WARNING names `/tokenize` *and what the
  fallback costs*, and that the guard still serves.
- **A lying `/tokenize`.** Tests [14]–[15] assert the honest thing: at 0.5x the guard is
  deceived **and reports success**, because it has no better source; at 3.0x it over-sheds
  but the turn the user just typed and the caller's protected system message both survive.
- **A cost regression nothing enforced.** Test [10] reads the fixture's call counter: a
  62-message payload produced **2** `/tokenize` calls, not 62.

### What it does NOT prove

Say this plainly wherever a green run gets quoted.

1. **The tokenizer is not Cydonia-24B's.** Every absolute number the suite prints is
   meaningless for the production budget. It validates the **contract and the wiring** —
   that the compactor asks the server rather than trusting itself, reads `.count`, that its
   scale-corrected arithmetic lands under the limit when a real tokenizer judges it, and
   that it degrades honestly. Not the production numbers.
2. **Version gap — the largest fidelity risk.** Shapes were verified against
   `vllm==0.10.0`, the only build that would pull and import. The deployed stack pins
   `vllm==0.24.0` (`Dockerfile:78`). Shapes are *believed* stable across that range; that
   was **not verified**.
3. **It is not vLLM.** It cannot catch a divergence introduced by vLLM's own tokenization
   path (mistral_common vs transformers, multimodal placeholder expansion, tool-schema
   framing). On AVX-512 hardware the same module points at a real vLLM CPU server unchanged
   — only the URL moves. That is the upgrade worth taking.
4. **`GENERATION_RESERVE` is not validated here** (`testfixtures/.../README.md:162`).
   Nothing in this harness endorses or challenges P1's 16,384.
5. **No vision.** `/tokenize` sees text parts only. `IMAGE_TOKEN_ESTIMATE` (4096) is
   untested — and it is exactly the input A14's 2x margin is absorbing.
6. **Emoji did not diverge** with the default pair (1.00x: local 5,604, server 5,631),
   because both are byte-level BPEs. Box-drawing (1.97x) and CJK (1.47x) did. The suite now
   prints all four categories and fails only if *none* of the hostile ones diverge — a
   per-category ratio is a property of the fixture **pair**, not of the compactor. Emoji
   cost is attested by the incident, not by this suite.
7. **The summarizer tests exercise `_chunk_to_budget`, not `summarize()`.** The map-reduce
   fold and its degradation are still covered only by the CPU-only suite. **And nothing
   here touches `summarizer.py` at all** — A1 is invisible to it.
8. **It is not wired into CI.** There is no CI: `git ls-files '.github/*'` returns 0 files,
   and the only tracked YAML is `docker-compose.yml`. `TESTING.md` asserts three separate
   times that Tier-1 gating is automated (`:31`, `:185`, `:194`) and once that the tooling
   is *"fully built"* (`:8`). Separately, the documented run command
   (`TESTING.md:84-89`) **exits 0 even when a suite fails** — the `for` loop's status is
   `tail`'s. Reproduced: one failing test, `EXIT CODE OF THE DOCUMENTED LOOP: 0`. Fix is
   `... || rc=1; done; exit ${rc:-0}`, 5 minutes, and it should go in the same commit that
   commits the harness (A2).

---

## 7. Open questions nothing settled

Each with the cheapest thing that would settle it. **None of these were guessed at above;
where a number is unknown it is marked unknown.**

**Q1 · What `COMPACTOR_GENERATION_RESERVE` is the live pod actually running?**
Everything in §3 P1 is about the committed config. `REMEDIATION.md:165` claims 8192 is in
force; the 2026-08-27 margin telemetry in the same document reconstructs to 2048 exactly
(2628 / 2755 / 2882 all match the 2048 arithmetic; 8192 would have saturated the 8,192 cap
on the first failure). A console-set override could still exist.
**Settles it:** `echo $COMPACTOR_GENERATION_RESERVE` on the pod. 10 seconds.

**Q2 · Which context-overflow wording does `vllm==0.24.0` emit?**
Determines whether A8's calibration can learn anything at all from a rejection. Both
wordings are reproducible in the fixture; the answer is one env var away once known.
**Settles it:** one oversized `POST {VLLM_URL}/v1/chat/completions` against the pod, and
read the error string. 1 minute.

**Q3 · Is `/tokenize` currently answering on the pod, and for the shape the guard sends?**
A9's whole point is that neither the log nor `/health/full` can tell you. Note the guard
measures an **unmerged** list — `_enforce_hard_budget` runs at `main.py:2062` *before*
`_merge_adjacent_system_messages` at `main.py:2064` — and `inject_system_block`
(`main.py:639-657`) deliberately creates adjacent system messages, the exact shape
`main.py:663-666` records as a production 400 on this model family. `REMEDIATION.md:206-210`
records a successful live measurement of 150,050, so `/tokenize` answered for *some* real
payload; whether that payload was the guard's shape is **not recorded**.
**Settles it:** `curl -s $VLLM_URL/tokenize -d '{"model":"…","messages":[{"role":"system",…},{"role":"system",…},{"role":"user",…}]}'`.
2 minutes. If it 400s, move the two merge passes ahead of the guard.

**Q4 · Does the production client send a bounded window today?**
Decides whether A6 is a live outage or a committed-future one. `FRONTEND_SPEC.md:219`
commits to *"Send a bounded window"* as the direction; it is not established as the current
behaviour. A5/P4's short-array case needs no windowed client and fires now regardless.
**Settles it:** one live request, then
`grep 'conv_id=<CONV>' $LOG_DIR/compactor.log | tail -1` and read `msgs=`.

**Q5 · What is a *realistic* central size for the assembled retrieval block?**
A4 establishes the ceiling (~11,600 vLLM tokens, pure decoration) and the floor (~1,460,
pure prose, nominal). The central value is **unknown** and is what decides whether A4 is
routine or urgent.
**Settles it:** run `format_retrieval_block` over real stored ChromaDB documents on the pod
and count the output through `/tokenize`. ~20 lines.

**Q6 · Is `4.10` chars-per-vLLM-token or chars-per-local-token?**
`main.py:865-868` labels all three figures as *"the DEPLOYED tokenizer"*, and `main.py:879`
concedes the transcript *"was measured on the pod and cannot be checked from this repo."*
A14's arithmetic deliberately routes around this using the repo's own fixture instead.
**Settles it:** a pod-side re-measure through `/tokenize`.

**Q7 · How often does the compactor process actually restart?**
Sets the real duration of A13's *"once per process."* Not answerable from the repo.
**Settles it:** pod uptime, or a log sweep for `count_tokens_exact.http` and
`count_tokens_exact.error`.

**Q8 · Did 20-turn L1 bodies ever succeed for the affected conversation?**
The production L1 chunk count suggests they may have, which would mean A1's overflow began
when the decorative replies did.
**Settles it:** the vLLM access log for summarizer `chat/completions` during the 19.8-hour
window, or a `/tokenize` measurement of a 20-turn slice.

**Q9 · What does `gather_memory_stats()` actually cost at production scale?**
A12 is a latency finding with **no measured duration anywhere** — the magnitude is unknown,
and `chromadb` is not installed on this machine so the `include` default could not be
confirmed empirically (it was proven functionally from `retrieval.py:455-458` instead).
**Settles it:** time `gather_memory_stats()` against a production-sized store, and time a
cold `_try_init()` and `AutoTokenizer.from_pretrained`.

---

## 8. Sequenced

**Tonight / first thing (≈1 h 40 m, then push):** P1 → P2 → P4 → P3.
Re-read the diff. `docker compose -f docker-compose.tokenizer-contract.yml up --build
--exit-code-from contract-tests` must be green. Tag, build from tag, PR.

**Day 1 after push (≈2 h, no code):** Q1, Q2, Q3, Q4 — four commands against the pod that
between them resolve four open questions and confirm or refute the reachability of A8, A6
and Q3's merge-order concern. Then A2: commit the harness and fix `TESTING.md:84-89`'s
exit code.

**Week 1:** A1 + A4 together (they share the module-boundary refactor) → A5 → A6 + A7 →
A3 → A8 → A9.

**Week 2:** A10 → A12 → A13 → A14 → A11 → the REMEDIATION.md reconciliation in §5 →
A15 as a design item.
