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
| 2 | ~~Client-disconnect contradiction~~ **DONE** | high | SETTLED on uvloop/Linux with a real socket | done | — |
| 3 | ~~N4b · task traffic is still fact-extracted~~ **DONE** | high | 99 requests in 2 days, measured | done | — |
| 4 | **N4a · OpenWebUI task model** | high | measured in logs | config only | owner |
| 5 | **D1.3 · backup goes silent when the source is sick** | medium | one incident, root-caused | ~2 h | nothing |
| 6 | **D1.2 · split the backup shape** | medium | reasoned, not measured | ~3 h | nothing |
| 7 | **Cap install + drain** | high value | — | ~1 h | deploy |
| 8 | **N3 · fact churn, re-measured** | unknown | stale — pre-F1 | ~2 h analysis | deploy |
| 9 | **F2 · a repetition metric** | medium | explicitly absent | ~4 h | deploy |
| 10 | **N2b · reply latency** | high | measured, cause suspected | large | N3 |

---

## Stage A — things that do not need the deploy

### A1 · Settle the client disconnect — **DONE, B was right**

**Reviewer B was right.** A's sub-agent's two claims — that the tail is
deferred to garbage collection and the upstream socket leaked — do not
reproduce on the loop production runs.

The reason neither reviewer could settle it is sharper than "they ran on
Windows". **Both used `TestClient`, which drives the app in-process through
the ASGI interface and never opens a socket**, so the event under test — a
peer hanging up mid-response — cannot occur in it. The disagreement was
between two readings of the code, dressed as two experiments.

`compactor/test_disconnect_uvloop.py` runs the real app under real uvicorn on
real uvloop and hangs up a real socket with `SO_LINGER 0`, sending RST rather
than FIN: a peer vanishing, which is what a closed browser tab produces.

| case | result |
|---|---|
| control, stream fully consumed | `stored` |
| RST after 397 bytes | `skipped_too_short` — counted and named |
| RST after 6,139 bytes | **`stored_trimmed`** — the prose she read reached memory |
| upstream generators still open afterwards | 0 |

The `finally` is not swallowed, the bookkeeping runs, the upstream is torn
down. R26 was built on B's answer and is standing on the correct one — and
the long-partial row above is the first time that has been demonstrated on a
production-shaped stack rather than argued.

**What this does not settle**, stated because the suite's name implies more
than it proves: A's sub-agent's scenario was a generator parked at `yield`
behind a slow consumer that never disconnects. That is backpressure, not
cancellation — it delays the tail, it does not lose it, and it is not what
the 51 stopped replies were. The suite exits 3 (SKIP) rather than passing
when uvloop is absent, so a Windows host run cannot report an answer to a
question it is unable to ask.

### A2 · N4b · Stop extracting memory FROM task traffic — **DONE**

**The backlog's own fix direction was unsafe, and implementing it as written
would have been a worse bug than the one it fixes.**

Half of N4 had already shipped: `_has_conversational_history` gates
`INJECTION_NO_HISTORY_FRACTION` (0.125 against 0.5), and
`COMPACTOR_INJECTION_NO_HISTORY_FRACTION=0` disables injection for task
traffic with no code change. The extraction half had not — `has_history` was
computed once and reached only that fraction and a log line, so neither
`_run_memory_tail` call site knew, and all 99 of the window's title/tag calls
were fact-extracted, indexed and deduped.

But that same predicate is False for a genuine FIRST TURN, so acting on it
alone silently drops the opening exchange of every new conversation. A second
attempt — "have we stored anything under this conv_id" — was killed by
`test_budget_guard` within one run, because it eats a REGENERATE of the first
reply, which sends exactly that shape.

What ships is a threshold: `_recorded_position >= 4`, two exchanges deep, is
the point past which an array with no assistant turn stops being an honest
picture of the conversation. Task traffic passes it within minutes and stays
past it; a new conversation and its regenerates sit below it. The residual
false positive — regenerating the first message of a conversation already two
exchanges deep — is rare, self-limiting, and unlike the defect it replaces it
is COUNTED, under `skipped_task_traffic`, and visible in `/health/full`.

`tailhealth` gained `HARMLESS_SKIP_OUTCOMES` on the way through: "the skips
that cost the user nothing" now has two members and was being spelled out
separately in two places, which is the fix-one-site-miss-the-sibling defect
in miniature. Both consumers read the one set now, and `test_tailhealth`
asserts the three sets partition `OUTCOMES` instead of naming a label.

**N4a remains open and remains the better fix**: a separate task model in
OpenWebUI's admin settings stops this traffic reaching the compactor at all,
where the code can only decline to remember it.

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
