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

### Stage 2 — the runbook the operator is told to follow — **DONE**

| | finding | why now |
|---|---|---|
| 3 | **R4** | `?dry_run=false` is read from the JSON body only, so the documented "commit the merge" command is a second dry run: HTTP 200, plausible counts, nothing changed |
| 4 | **R13** | `/admin/compact` 409s for any conversation where even one exchange missed the episodic index — i.e. every real one — and it is the endpoint R12's own ERROR line points at |

Both fixed. R4 reads the query form as well as the body, and the runbook now
prints the body form — the one that has always been read — with a line telling
the operator to check the counts rather than the status code.

R13 rebuilds by SLOT from `turn_index`, with one decision worth stating: slots
are relative to the lowest stored index, not absolute. `turn_index` counts
system messages the summarizer skips and is reallocated on write, so it is not
an exact position — but its DIFFERENCES are exact (two message-units per
exchange in both numberings), so a jump of four is reliably one lost exchange.
A missing head makes the rebuild short and the existing 409 fires unchanged,
so the assumption checks itself. `summarizer._recorded_position` is untouched,
so a pulled-down watermark still cannot permit a short rebuild.

**Correction to `V314_BACKLOG.md`:** it lists `test_admin_compact [3d]` as
"PINNED WRONG". It is not. `[3d]` pins the refusal for a store that does not
reach the position, which survives R13 and should. The refusal R13 removes was
never covered by a test at all.

### Stage 3 — memory correctness she would notice — **R24/R9/R19/R25 DONE**

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

**Resolved: the prior attempt's counting was wrong, not the fixture.**
`_SENTENCE_END_RE` matches a terminator followed by whitespace OR
end-of-string, and that end-of-string branch is essential for
`trim_to_last_sentence` (a reply's last sentence has no trailing space) but
wrong for a break COUNT: the formula is `fragments = breaks + 1`, and the
`+1` already accounts for the line's own final fragment. Counting that last
period too yields one fragment too many and moves the calibrated 1,600/40 vs
1,640/40 line by exactly one. The fix keeps the literal two-character match
filtered through the shared predicate, so the pinned fixture needed no change
and still passes verbatim.

R24 now shares one definition of a sentence end with the trimmer
(`_is_real_sentence_end`) - what the duplication note above asked for. R25
replaced the lead-character set with "the preceding character is not
alphanumeric", covering every dash rather than enumerating them. R9 requires
the qualifying run to reach the END of the reply: a real runaway ran to its
own end, and the corpus's 1,261-item case had no closing prose, so the
66-books reply is clean while the runaway still fires. R19 gates the list
rule on `DEGENERATE_MIN_CHARS`, matching this file's own doctrine.

**R8 is now fixed** (with Stage 4's `main.py` pass). The conditions that store
nothing are evaluated BEFORE counting, rather than having `_async_tail` report
an outcome afterwards — because `bgwork.pool` sheds, and a tail dropped at the
ceiling would then never be counted at all, a new silent skip of exactly the
shape being fixed.

Extraction-disabled was deliberately NOT hoisted: episodic indexing still runs
there, so counting it as a skip would be a second lie. The real defect was that
the facts block's `return` returned from the WHOLE tail, so the summary rollup
was silently off for the life of any `COMPACTOR_FACTS_EXTRACTION=false`
deployment. Fixed at source.

### Stage 4 — observability that would have caught all of this sooner — **R27/R28/R29 DONE**

| | finding | effect |
|---|---|---|
| 9 | **R26** | vLLM dying mid-stream skips the tail with no counter and no greppable line, on a reply she already read |
| 10 | **R27** | an empty reply (Stop before the first token) flips `/health/full` to degraded for five minutes and claims memory was lost |
| 11 | **R29** | a partial-coverage warning says a chunk was recorded before the summarizer is called, so it lies during an outage |
| 12 | **R28** | `tailhealth.note` raises out of a `finally` on a non-numeric count, and leaves the counters inconsistent |

R27: `skipped_recently` is keyed off whether the outcome was LOSSY, not off
"any skip", so a run of manual stops before the first token no longer pins
`/health/full` degraded over turns that lost nothing. R28: the char counts go
through a coercion helper and the outcome tally is the LAST mutation in
`note()`, so a bookkeeping failure leaves the ledger consistent rather than
half-incremented.

R29 was **already fixed incidentally** by the R23/R12 commit, which moved the
partial-coverage warning below the summarization. Verified by reproduction
rather than assumed.

**One defect was introduced and caught during this pass**, recorded because
the comment was more convincing than the code: the lossy test was written as
membership of `LOSSY_SKIP_OUTCOMES`, and that set is built by excluding from
`OUTCOMES` - so it cannot contain a label that is not in `OUTCOMES`, making
an unrecognised outcome NON-lossy. The exact inversion of the safe default
its own comment claimed. Now tested as "not a store and not the one harmless
label", pinned by a test that fails on the set-membership form.

**R26 is now fixed.** The stronger of the two options was taken: prose she had
already read is MEMORIZED through the existing trim path, not merely counted.
`finished` still comes from the accumulator, so a reply vLLM finished before
dropping the socket is not trimmed, and the 4xx branch passes through safely
because its error chunks are never fed to the accumulator — the compactor's
own apology cannot become a memory.

### Stage 5 — the remainder, and the one that is bigger than it looks — **DONE**

| | finding | note |
|---|---|---|
| 13 | ~~**R30 (rest)**~~ **DONE** | `compactor/envcfg.py`, with no dependency on anything else in the package. Verified end to end: six typo'd variables including `MAX_MODEL_LEN=32K` now import cleanly. The positivity guard was deliberately NOT retrofitted onto the ~40 non-window sites — none rejected those values before, nothing reproduced a failure, and several knobs take 0 as a meaningful off. A range check that cannot be justified with a reproduction is a guess. |
| 14 | ~~R15, R16, R18, R20, R21~~ **DONE** | Five of these were ALREADY FIXED in the tree by `157bdc3`, beyond that commit's stated scope of R23/R12 and with no tests at all — five behaviours whose only evidence was a comment. They now have tests (`test_position_edges.py`). R18 was only half-fixed: the existing code handles a repeating TAIL but not a window that has become ALL repeats, where the two arrays are equal byte for byte and no content test of any width separates them. Measured at cap 20 the position stalled permanently at 40 while the conversation ran to 68. Fixed on `n < prev` — the window is a strict suffix, which the admin drain provably cannot be. R16's trade re-examined against the newly-persisted `head_fp` and deliberately CONFIRMED, not repaid. |
| 15 | ~~R10, R11, R17, R22~~ **DONE** | R10: the guard is RIGHT and must stay `<` (mutating it to `<=` fails 15 assertions); the symptom is real but sits downstream, in `_observed_position` reading the CHAT path's anchor against the drain's rebuild. R11 re-derived from scratch — B's four were gone — as 43 mutations, 41 killed, and one survivor that was a real defect rather than a missing test. R17: 32.4 ms -> 3.2 ms at 700 turns, by bounding the hash rather than moving it off the loop. R22: 239 s -> 1.6 s. |

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
