# v3.1.7 — the plan for what the review found

Thirty findings, from four independent reviewers plus a log sweep. The full
evidence lives in `V314_BACKLOG.md` (sections R1–R30 and L1–L6); this file is
the ordering, and the reasoning behind it.

**Read this first if you are deciding what to do next.** The backlog says what
is wrong. This says what to do, in what order, and — more importantly — what
must NOT happen before what.

---

## The one hard ordering constraint

**Do not enable the `max_turns` cap until R23 and R12 are fixed and green.**

Two independent, reproduced defects make the cap unsafe:

* **R23** — a client turn is silently lost the moment the valve engages. The
  position arithmetic assumes the array length is the position, which holds
  only for an unbounded window. Measured: at cap 100 over 64 exchanges, client
  turn 101 is summarized by no tier, and the chunk labelled 101-120 actually
  contains 102-121.
* **R12** — if a state file's watermark was pulled down to the cap by the old
  code, the duplicate-label guard discards every new L1 chunk, silently, for
  roughly 280 exchanges.

Both are in `summarizer.py`'s position tracking. Neither is visible in the
log: R23 emits an INFO line describing the healthy shape, and R12 emits a
WARNING that reads like housekeeping.

The cap is what fixes the live starvation (L3: 807 turns needing 119
summarization calls against a 4-call ceiling, so inline compaction is off and
every turn sheds ~724 turns). It is the most valuable single change
available — which is exactly why it must not go on before the two defects
that make it lose memory.

---

## Order of work

### Stage 0 — ship what is already fixed

Nothing here is outstanding; it is listed so the deploy is not re-litigated.

R1 (Dockerfile `tailhealth.py`), R2 (env boot crash), R3 (empty-assistant
repair), R5 (merge/retire un-pinning), R6 (skip-as-pass suites), R7/R14
(multibyte corruption), and the main.py half of R30. All have tests, all
mutation-checked, suite green.

**This stage alone stops the live incident.** The pod runs v3.1.4, whose
repair covers what we FORWARD and not what we MEASURE, so `/tokenize` 400s
about 145 times a day and every budget decision falls to a counter reading up
to 51% low (L1). Shipping v3.1.6+ ends that chain regardless of everything
below.

### Stage 1 — unblock the cap — **DONE**

| | finding | why now |
|---|---|---|
| 1 | **R23** | silently loses a turn at the moment the cap engages |
| 2 | **R12** | silently stops summarizing for ~280 exchanges after upgrade |

Fixed. The two position regimes collapse to `position = max(n, prev + new)`:
`n` is a lower bound (every window turn is real, but a capped window is a
suffix), `prev + aligned` is the other, and the position is the larger — exact
whenever either source is exact, never over-counting. For R12 the seed now
reads `max(turns_seen, last_summarized_turn, highest chunk label)`, because
**the chunk list is the record and the watermark is a pointer derived from
it** — the only one of the three the old `_reconcile_watermark` could not
erase. A watermark below its own chunks is repaired before anything reads it.

`admin_compact` now calls `summarizer._recorded_position` rather than maxing
the two counters locally, so the endpoint and the rollup cannot disagree about
how far a conversation has got. Without that it would PERMIT a rebuild it
should refuse, on exactly the pulled-down state files R12 is about.

Two trades, taken deliberately and documented in the code:
* **R16 is left standing.** With no anchor and no chunks to corroborate, the
  hold is still never repaid; repaying it needs evidence that call does not
  have.
* R23 can under-count by one turn in one narrow case (no anchor, `n >= prev`,
  no chunks) — once per conversation, erring toward duplicating rather than
  dropping, which is the safe direction.

Six mutations were watched to fail, four of them re-run independently.

**Then, and only then:** install the OpenWebUI filter at `max_turns=60`,
following the runbook in `pipelines/conversation_id_header.py` — after R4
below, because step 8 of that runbook does not currently work.

### Stage 2 — the runbook the operator is told to follow

| | finding | why now |
|---|---|---|
| 3 | **R4** | `?dry_run=false` is read from the JSON body only, so the documented "commit the merge" command is a second dry run: HTTP 200, plausible counts, nothing changed |
| 4 | **R13** | `/admin/compact` 409s for any conversation where even one exchange missed the episodic index — i.e. every real one — and it is the endpoint R12's own ERROR line points at |

R4 is small and blocks Stage 1's install sequence. R13 is a design change
(rebuild the transcript by slot from `turn_index`, filling gaps with explicit
placeholders) and is the larger of the two; it can follow the cap going on,
but not by much, because it is the only rebuild-from-store recovery path.

### Stage 3 — memory correctness she would notice

| | finding | effect |
|---|---|---|
| 5 | **R24** | ordinary prose containing `Dr.` / `9 a.m.` is judged degenerate and then permanently redacted from every future summary |
| 6 | **R9 / R19** | a 50+ item enumeration — the 66 books of the Bible, the 50 states — is unmemorable, permanently, for a scripture assistant |
| 7 | **R25** | an em dash before an abbreviation stores a fragment as a memory |
| 8 | **R8** | `tailhealth` counts `stored` before `_async_tail` runs, so `/health/full` can report a healthy tail for an exchange that never reached memory |

R24 and R9/R19 share a root cause worth stating plainly: **the same commit
that added `_SENTENCE_ABBREVIATIONS` for the trimmer left the degeneracy rule
counting `line.count(". ")`.** Two pieces of new code in one delta disagreeing
about what a sentence end is. Fix by making them share one definition, not by
adding a second stoplist — the duplication IS the bug.

Partial work for R24/R25 exists and currently breaks a pinned boundary
assertion (`mean 41` now reads as 40). Resolve whether the fix or the fixture
is wrong before merging.

### Stage 4 — observability that would have caught all of this sooner

| | finding | effect |
|---|---|---|
| 9 | **R26** | vLLM dying mid-stream skips the tail with no counter and no greppable line, on a reply she already read |
| 10 | **R27** | an empty reply (Stop before the first token) flips `/health/full` to degraded for five minutes and claims memory was lost |
| 11 | **R29** | a partial-coverage warning says a chunk was recorded before the summarizer is called, so it lies during an outage |
| 12 | **R28** | `tailhealth.note` raises out of a `finally` on a non-numeric count, and leaves the counters inconsistent |

### Stage 5 — the remainder, and the one that is bigger than it looks

| | finding | note |
|---|---|---|
| 13 | **R30 (rest)** | ~35 bare `int(os.environ...)` / `float(os.environ...)` sites across 12 modules, every one an import-time crash on a typo. `MAX_MODEL_LEN=32K` still stops the boot via `facts.py:171` and `summarizer.py:125`. Needs one shared `envcfg` helper — nothing may import `main` — routed through everywhere. Mechanical, touches every module, deserves its own review pass. |
| 14 | R15, R16, R18, R20, R21 | remaining summarizer position edges, once R23/R12 land |
| 15 | R10, R11, R17, R22 | `/compact` equality guard, surviving mutations, event-loop hashing cost, a 240s test that trips a 240s ceiling |

---

## The unresolved contradiction — settle before building on it

A's sub-agent and B disagree about whether a client disconnect reaches the
memory tail. A's sub-agent reproduced a fourth scenario — the generator parked
at `yield` — that defers the tail to garbage collection and leaks the upstream
socket; B found the tail fires in both cancellation placements; A's own final
report filed the whole area under "checked and found sound", omitting its
sub-agent's result.

Both ran on Windows/Proactor with plain asyncio. Production is Linux and
likely uvloop, and a server advertising ASGI `spec_version 2.4` takes the
branch that lands in the deferred case every time.

**This matters because it is the mechanism behind the 51 stopped replies that
motivated the entire memory-tail change.** Settle it on a production-shaped
stack before anyone builds on either answer.

---

## How this work is verified

`python scripts/run-tests.py` — one command, and it distinguishes PASS from
SKIP. Exit 0 pass, 1 fail, **3 skipped**. Three suites used to exit 0 without
running their checks, so every "all suites pass" in this branch excluded the
tokenizer contract; that is fixed, and the runner will not let it recur.

`--saturation` adds `test_saturation.py`, 200 exchanges through the real
endpoint in ~50s, asserting accumulation rather than logic: the tail ledger
reconciles, `turn_index` never repeats, the watermark keeps pace, no turn
falls between tiers. It does NOT charge real tokens — the docker soak owns
that, and a budget regression can pass here.

**A test only counts if it has been watched to fail.** Every fix in Stage 0
was mutation-checked; keep that standard. Two mutations survived on first
attempt during this work and both were real gaps: one because the test was
never registered in the runner, one because deleting a call site left every
unit test green.

---

## What this review says about the process

Four reviewers, largely disjoint find-sets, two findings unanimous. Each of
the four — including the one told nothing but "look for issues" — found
something no other did. A single review, however well briefed, would have
shipped most of these.

Three suites **pinned the wrong behaviour**: they asserted the defect was
correct. A mutation sweep cannot find that class, because the tests agree with
the code. Only a reviewer reading intent can.

The recurring defect held: R1, R2, R5, R24 and R30 are all one rule applied at
one site and missed at its sibling. When fixing any of the above, look for the
twin — and prefer sharing one function over copying the rule.
