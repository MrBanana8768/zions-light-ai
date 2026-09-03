# What is left after v3.1.7 — the queue, and the reasoning behind the order

v3.1.7 closed all thirty review findings and all nine deferred backlog items.
This file triages what remains: the pre-review sections of `V314_BACKLOG.md`
(N/F/C/D) that were never closed, the deploy-day items carried from the
08-30 analysis, and the one question the review could not settle.

**Read `V317_PLAN.md` first if you are asking what v3.1.7 did.** This file is
only about what comes next.

Every item below was checked against the CODE, not against the backlog's own
claims about itself, because several entries described work that had since
shipped and one described work that had shipped only halfway.

---

## The one hard ordering constraint

**Ship v3.1.7 before measuring anything.**

Four of the items below are measurements, not fixes, and every one of them
would be measuring a system that no longer exists. The pod runs v3.1.4 with
hot-patches. Its `/tokenize` repair covers what is FORWARDED and not what is
MEASURED, so the token counter reads up to 51% low, inline compaction is off,
and each turn sheds ~724 turns (L1/L3). Numbers gathered against that are
numbers about a bug.

That is also why N3 and the F2 metric sit late in this queue rather than
early. They are not less important; they are unmeasurable today.

---

## Triage

Severity is about the user, not the code. "Evidence" is how well the claim is
established — the difference between a reproduction and a plausible reading
matters more on this project than usual, because three suites here have
pinned the wrong behaviour and two findings inverted on inspection.

| # | item | severity | evidence | effort | blocked by |
|---|---|---|---|---|---|
| 1 | **Deploy v3.1.7** | — | — | ~1 h + pod work | owner |
| 2 | **Client-disconnect contradiction** | high | contradictory, reproduced on the wrong OS | ~half a day | nothing (unblocked) |
| 3 | **N4b · task traffic is still fact-extracted** | high | measured in logs | ~2-3 h | nothing |
| 4 | **N4a · OpenWebUI task model** | high | measured in logs | config only | owner |
| 5 | **D1.3 · backup goes silent when the source is sick** | medium | one incident, root-caused | ~2 h | nothing |
| 6 | **D1.2 · split the backup shape** | medium | reasoned, not measured | ~3 h | nothing |
| 7 | **Cap install + drain** | high value | — | ~1 h | deploy |
| 8 | **N3 · fact churn, re-measured** | unknown | stale — pre-F1 | ~2 h analysis | deploy |
| 9 | **F2 · a repetition metric** | medium | explicitly absent | ~4 h | deploy |
| 10 | **N2b · reply latency** | high | measured, cause suspected | large | N3 |

---

## Stage A — things that do not need the deploy

### A1 · Settle the client disconnect — **do this first**

The only item here the review left explicitly unresolved, and it is now
unblocked: `testfixtures/unit-suite/Dockerfile` installs uvloop, so the
question can finally be asked on a production-shaped stack.

The contradiction: A's sub-agent reproduced a fourth scenario — the generator
parked at `yield` waiting on a slow consumer — that defers the memory tail to
garbage collection and leaks the upstream socket, and showed the `aclose()`
in the `finally` sits in front of the only bookkeeping, one `await` from
swallowing the tail entirely. B found the tail fires in both cancellation
placements. **Both ran on Windows/Proactor with plain asyncio**, and A's own
final report then filed the area under "checked and found sound", omitting
its sub-agent's result.

Why it leads the queue despite fixing nothing directly: it is the mechanism
behind the 51 stopped replies that motivated the entire memory-tail change,
and R26 has now been built on one of the two answers. A server advertising
ASGI `spec_version 2.4` takes the branch that lands in the deferred case
every time.

Deliverable is EVIDENCE, not a patch: a reproduction under uvloop on Linux
that settles which reading is true, and a test that pins it. If A's
sub-agent is right, the leak and the deferred tail are new findings and get
their own entries.

### A2 · N4b · Stop extracting memory FROM task traffic

**Correcting the backlog, which is half out of date.** N4 says the compactor
"already detects the shape" and "could stop injecting memory into and
extracting memory from requests it has already classified as tasks". The
injection half SHIPPED: `_has_conversational_history` (main.py:2710) gates
`INJECTION_NO_HISTORY_FRACTION` (0.125 against 0.5), and
`COMPACTOR_INJECTION_NO_HISTORY_FRACTION=0` turns injection off for task
traffic entirely, no code change needed.

The extraction half did not. `has_history` is computed once at main.py:4850
and reaches exactly two places: the budget fraction and a log line. Neither
`_run_memory_tail` call site (main.py:5228, 5350) is told. So every
title/tag/follow-up call is still fact-extracted, episodically indexed and
deduped — N3's "second treadmill", on a conversation that fires ~90 s after
every real turn and is not a conversation.

This is the recurring defect in its usual costume: **one classification, two
consumers, wired to one of them.** Fix by passing the value the function
already computes to the call sites that need it — not by re-deriving it
there, which is how the two would drift apart again.

Watch for the twin: `backfill.py` and `commands.py` also reach the fact
store, and neither knows what task traffic is.

### A3 · D1.3 · A backup that produces nothing is worse than a degraded one

During the 02:17 incident the backup itself failed
(`attempt to write a readonly database`) and produced NOTHING, at exactly the
moment an archive was most wanted. `_alert_failure` (backup.py:912) fires, so
the operator is told — but there is no fallback: when the SQLite-safe path
fails there is no raw copy and no degraded label.

The trigger is much less likely since D1.1 moved the live database off
MooseFS to local disk (v3.1.6, `webuidb.py`). Less likely is not handled, and
the failure mode is total rather than partial, which is what keeps it on the
list.

Fix: fall back to a byte copy of the database AND its journal — they are a
matched pair, and the incident notes are emphatic that separating them turns
a recoverable file into a corrupt one — label the archive degraded, and
alert. A byte copy of a sick database is forensically valuable; no archive
at all is not.

### A4 · D1.2 · Split the backup shape before raising the cadence

`COMPACTOR_BACKUP_INTERVAL_HOURS=1` needs no code, and that is the trap: an
hourly FULL tar rewrites chromadb and every facts file onto the volume 24x a
day. Split it into an hourly light snapshot of `webui.db` alone plus the
existing daily full archive, THEN raise the cadence.

Note this is reasoned rather than measured — no incident has been traced to
backup write amplification. It is queued below D1.3 for that reason.

---

## Stage B — the deploy, and what only it unblocks

### B1 · Ship v3.1.7

Tag, build from the tag, PR (owner merges). **This stage alone stops the
live incident** — the `/tokenize` 400 chain, the 51%-low counter, and the
shed-724-turns behaviour all end with the deploy, independent of everything
else in this file.

### B2 · Install the `max_turns=60` cap

Follows the runbook in `pipelines/conversation_id_header.py`. Unblocked: R23
and R12 (the two defects that made the cap lose memory) and R4 (the runbook
step that told the operator to commit and did not) are all fixed and green.
This is the most valuable single change available for the live starvation.

### B3 · Drain conv=4214…'s backlog, once

`POST /admin/conversations/<id>/compact`, dry-run first. ~610k tokens that
request-path compaction will correctly skip forever.

**Do not attempt this before the deploy.** That endpoint 409'd for every real
conversation until R13, and its `?dry_run=false` was a second dry run until
R4 — the command would have reported success and changed nothing.

---

## Stage C — needs production data from a healthy system

### C1 · N3 · Re-measure fact churn

The backlog records 5,341 facts extracted and 3,714 evicted (70%) with dedup
yielding 2.7%, and blames one knob doing two jobs: `MAX_FACTS_TOKENS` capping
both the store and the injection, so everything was injected every turn,
everything was touched every turn, `last_used` meant nothing, and LRU
degenerated to FIFO.

**F1 shipped and fixed that mechanism** — `select_for_injection` ranks
non-pinned facts and injects top-K plus a pinned identity tier. So the 70%
number is stale by construction: it describes the regime F1 removed.

What to capture after the deploy: eviction rate, dedup yield, and fact
survival half-life from the archive sidecar. Then decide
`COMPACTOR_MAX_FACTS_TOKENS` against the raised extraction cap — the two move
together or churn gets worse, which is the open half of deploy-day note 3.

### C2 · F2 · Build the repetition metric

F2's own "Still open" says it plainly: *"A little too repetitive" is not a
number, so nothing here can be confirmed to have helped.* The drift detector
is the proof that this approach works; the equivalent here is n-gram overlap
between consecutive replies plus opening-phrase frequency over the last N.

Also re-test `repetition_penalty 1.1` after the deploy. It was tuned against
a ~183-fact injected prompt; F2 cut that to ~26, roughly sevenfold. The knob
reaches into the PROMPT (unlike `presence_penalty`), so the surface it acts
across changed shape. **What to watch does not look like a sampling problem:
name and fact AVOIDANCE — paraphrasing around established facts rather than
stating them, which reads as vagueness.**

### C3 · N2b · Reply latency

p50 95 s, p90 138 s, max 583 s over 85 turns. She stops waiting; each
cancellation used to skip the memory tail (R26 fixed the memory half — the
prose is now memorized, not discarded). The remaining half is the driver,
and the suspected cause is N3's contention: ~8 background 24B generations per
user turn on the same GPU her replies stream from.

Sequenced last because the fix follows from C1's numbers, and because A1 may
change what is believed about the cancellation path itself.

---

## Not queued, and why

* **N1, N5, N6, F1, F2 (main), C1, D1.1** — closed. Verified in code:
  `assistant_content_is_empty`; `backup.py:1037` sleeping `interval - age`;
  `main.py:1249` `filterwarnings`; `portability.py:452-471` filtering
  `CLONE_CONV_ID_HERE`, `__selftest_oneshot_*` and `itest-*`;
  `select_for_injection`; `webuidb.py`.
* **`COMPACTOR_RAG_TOP_K` at 5** — the owner's call, recorded in F2 as
  deliberate.
* **The two mutations that survive** the memory-tail sweep — recorded with
  reasons in the test file. `kept_chars`'s conditional is structurally
  unreachable and the `<= window` boundary errs in the safe direction.
* **R16's 2-turn hold** — re-examined in v3.1.7 against a discriminator that
  did not exist when the trade was taken, and deliberately confirmed rather
  than repaid. Do not reopen without new evidence; the reasoning is in
  `summarizer.py` and `V317_PLAN.md`.

---

## How anything here gets verified

`docker compose -f docker-compose.tests.yml run --rm unit-tests` — Ubuntu
24.04, the userland production runs. Host runs are a development
convenience, not evidence: the asyncio loop differs, which is the whole of
A1, and the timings differ enough to have cost a full investigation once
(`SLOW_S` in `scripts/run-tests.py`).

The two fixture-backed suites need their own stack and will honestly SKIP
without it:

    docker compose -f docker-compose.tokenizer-contract.yml up --build --exit-code-from contract-tests
    docker compose -f docker-compose.tokenizer-contract.yml run --rm --build soak-tests

**A test only counts if it has been watched to fail.** Every fix in v3.1.7
was mutation-checked, including a merge-level pass over the two lanes whose
fixes interlocked. Keep that standard: on this project a surviving mutation
has twice been a real gap, and once been a real defect hiding behind a test
that agreed with the code.
