# V4 — the prioritised feature set

**Status:** planning. Synthesised from three independent review lenses, 2026-08-28.
**Supersedes nothing.** [`V4_ROADMAP.md`](V4_ROADMAP.md) states the gates;
[`compactor/V4_PLAN.md`](compactor/V4_PLAN.md) is the tool-execution design;
[`FRONTEND_SPEC.md`](FRONTEND_SPEC.md) is the client specification. This document
orders the work and says what it costs.

**Verification note.** Every file:line in this document was checked against
`fix/v3.1-remediation` at `23aa348` on 2026-08-28. Claims taken from the review
documents rather than from the code are marked *(cited)*. Three lens claims were
wrong at HEAD and are corrected in §7 — read that section before planning from
`MEMORY_REVIEW.md` line numbers.

---

## 1. The V4 shape, in one paragraph

V4 is the replacement client, plus a deliberately small code lane that rides
alongside it. The client is the whole of the critical path: it is what makes the
system able to tell the user the truth about itself, it is the only thing that
mechanically prevents the 2026-08-24 chain corruption, and `V4_ROADMAP.md` §2
already establishes that V4.2's human-in-the-loop approval is unimplementable
without `FRONTEND_SPEC` §12's receipt. The code lane is scoped to what is
falsifiable on this hardware: a mechanical import-and-syntax gate that this repo
has needed five times in one week, a symbol index so a 3,994-line `main.py` can
be addressed in pieces that fit a 16,384-token input budget, and one eval — run
before the feature — that decides whether Cydonia-24B can review code at all.
Code *creation* ships as text in a chat reply, never as a write to disk, because
the compactor process mounts the volume holding every conversation's memory.
Everything that needs a network peer the owner does not run is V5 by
construction, and the standing offline requirement enforces that on its own.

**Where the lenses disagreed with the direction, stated once.** Two of three
lenses independently ranked non-V4 hardening above V4 features, and the two items
they ranked highest — the uncapped summary block and the background-work loss on
every redeploy — appear in neither `V4_ROADMAP.md` §2's gate table nor the
front-end/code direction. Both are verified live at HEAD (§2.1 P5, P6). I have
planned for the direction as given and folded those two in as V4.0 prerequisites
rather than as features, because they are cheap (0.75 days combined) and because
each one silently destroys memory, which is the failure the client exists to make
visible. That is the only place this plan departs from the stated shape, and it
departs by three-quarters of a day.

---

## 2. The prioritised feature list

Effort is in **focused engineering days** — days where the work is the only
thing happening. At a realistic part-time rate of ~2 focused days/week, multiply
by 3.5 for calendar time. These numbers are what I actually believe; §6 says what
they rest on and where they are soft.

### 2.1 V4.0 — prerequisites (server and ops; all parallel to client work)

These are not features. Each is a gate `V4_ROADMAP.md` §2 already names, or a
verified live defect that destroys memory silently.

| # | Item | Effort | Depends on | Offline |
|---|---|---|---|---|
| P1 | Import + syntax gate in CI and pre-deploy | 0.5 | — | ✅ |
| P2 | Timezone in the image and the pod env | 0.25 | — | ✅ |
| P3 | Clock + elapsed-gap line in the existing injected block | 0.5 | P2 | ✅ |
| P4 | Local alert sink, transition poller, three request-path counters | 1.5 | — | ✅ |
| P5 | Cap the summary block | 0.5 | — | ✅ |
| P6 | `stopwaitsecs` + drain timeout (config half only) | 0.25 | — | ✅ |
| P7 | Gate the memory tail on prior conversational history | 0.5 | — | ✅ |
| P8 | Teach `/tidy` the rule `facts.py` already learned | 0.5 | — | ✅ |

**Subtotal: 4.5 days.**

**P1 — Import + syntax gate.** `ast.parse` and `pyflakes` on changed files, plus
`python -c "import main, memory, commands, facts, retrieval, summarizer, ..."`
against the commit the change will land on. **Verified:** there is no `.github/`
directory in this repo, while `TESTING.md` states Tier 1 runs in GitHub Actions
on every PR — a documented gate that does not exist. `INCIDENT_2026-08-28.md`
§B.3 records this defect class recurring five times in one week, one of which
took production down during a hotfix, and concludes: every one would have been
caught by importing the module once. `REVIEW_PLAN.md` §1 is the same finding from
the other side — across four verification gates, every blocker was introduced by
a fix, and all four were "invisible to reading and obvious to running."
*Risk:* none technical. The risk is that it feels too unglamorous to do first and
the code lane gets built on an unverified substrate.

**P2 — Timezone.** Add `tzdata` to the image and `TZ` to the pod env, read
through `zoneinfo` with a UTC fallback that cannot raise. **Verified:** grep for
`TZ=|tzdata|localtime|America/|zoneinfo` across `Dockerfile`, `entrypoint.sh`,
`supervisord.conf`, `runpod.env.template` and `.env.example` returns **nothing**.
The container is UTC. `V4_ROADMAP.md` §1.1 item 1's own example line reads
`Current time: 2026-08-28 20:41 MST (Friday)` — shipping item 1 as written on
today's image injects a time 6–7 hours wrong, stated confidently. That converts
"invents a time" into "asserts a false one", which is worse for the user and
which `COGNITIVE_ARCHITECTURE.md` forbids. §1.1's "under an hour" estimate
assumes a clock the pod does not have.

**P3 — Clock and gap.** Two lines appended to the block `inject_system_block`
(`main.py:1241`) already emits, plus a `last_seen_at` written per conversation.
**Not** a trailing system message after the conversation. `V4_ROADMAP.md` §1.1
item 1 prescribes trailing placement for prefix-cache reasons and both halves of
that argument fail against the code: `inject_system_block` puts the combined
block at position 1 and that block contains the query-dependent retrieval layer,
so the cacheable prefix is already invalidated every turn and the clock costs
nothing extra there; and a non-leading system message on a Mistral template is
the exact shape `_merge_adjacent_system_messages` (`main.py:1368`) exists to
prevent. The more valuable half is the gap, not the clock — a daily companion is
asked "how long has it been", and no layer records when the conversation last
happened.

**P4 — An alert path that works with no internet.** Default sink is a local
JSONL the pod can read, exposed through `/health/full`; a webhook is optional and
additive. **Verified:** `COMPACTOR_ALERT_WEBHOOK` is still commented out
(`runpod.env.template:198`) and `alert.notify` has exactly two callers —
`backup.py:907` and `selftest.py:713`. That is `INCIDENT_2026-08-28.md` C2
verbatim: "the user is the monitoring." A public webhook cannot be the primary
path under the offline requirement. **The part most likely to be got wrong:**
alerting on the guard's ERROR would *not* have caught 2026-08-28 — that
incident's guard **succeeded**, fitting 4 messages of 65 and returning HTTP 200
(§A.2 stage 5). The signature was sustained massive shedding *at success*, so the
three counters that matter — rolling shed ratio, compaction-fallback count, and
injected-layer drop counts — do not exist anywhere today.
*Risk:* alert fatigue on a two-person system kills the channel permanently.
`OPERATIONS.md` has the owner stop/start vLLM for model swaps, which will trip
"vLLM unreachable" every time. Needs transition hysteresis or the owner learns to
ignore it within a week.

**P5 — Cap the summary block.** **Verified live at HEAD:**
`format_summary_block` (`summarizer.py:207`) renders every L3, every L2 chapter
and every L1 chunk with no token budget, and `summarizer.py:828` carries the
comment "nothing trims l2". This is the last uncapped push layer. What changed is
the consequence: v3.1.1 D3's `_bound_injected_blocks` (`main.py:1276`) now drops
whole layers lowest-priority-first, and the priorities are persona 0, summary 1,
facts 2, retrieval 3 (`main.py:~211`). So an oversized summary no longer
overflows the window — it silently evicts episodic recall first, then facts,
which presents to the user as "she forgot me" for the third time.
**Honest caveat:** it is not firing yet. `INCIDENT_2026-08-24.md` records L1=5,
L2=0, L3=none *(cited)*, and `MEMORY_REVIEW.md:395` explicitly relabels the
alarming block-size figures as ceilings computed from output caps, not observed
sizes. It fires as L1 fills toward 10.
*Risk:* newest-first is the defensible default and it is the choice that discards
the oldest chapter — the identity-bearing one. Log what was dropped.

**P6 — Stop losing background writes on every redeploy.** **Verified:**
`main.py:2447` is `await bgwork.pool.drain(timeout=10.0)`, and grep for
`stopwaitsecs` across `supervisord.conf` returns nothing against seven
`[program:]` blocks — so supervisord's 10s default SIGTERM-then-SIGKILL races the
drain with zero margin. A tail is three sequential LLM calls on a 24B; 10 seconds
is not close. Every `supervisorctl stop compactor` therefore abandons tails, and
each one permanently loses that turn's fact extraction, episodic embedding and
rollup — silently, no retry. This ranks high on likelihood alone: it fires on
every redeploy, and the owner has redeployed repeatedly this week.
**Take only the config half.** The lock-before-semaphore reordering
(`bgwork.py:126-128` is `async with self._sem: await coro`, so a tail takes a
pool slot and then blocks on `conv_lock`) is a real defect but it is a
concurrency change, and this branch has already deferred one of those for exactly
that reason. The config half is the whole redeploy-loss fix and carries no
concurrency risk.

**P7 — Gate the tail on prior conversational history.** Add
`_has_conversational_history(messages)` (`main.py:1262`) to the conditions at
**both** `_async_tail` call sites — `main.py:3297` and `main.py:3424`.
`FRONTEND_SPEC` §15 raises this as a required protocol ask needing a
request-kind marker *(cited)*; it does not need one. The function already exists,
was written for exactly this distinction, and is currently used only to size the
injection budget. Reusing it closes the phantom-conversation write path for *any*
client, including OpenWebUI during parallel running.
*Cost, which should be stated rather than discovered:* the first exchange of
every brand-new conversation loses its fact extraction, since a first turn has no
prior assistant message. The second turn re-reads the same material. That is the
correct trade.
*Risk:* two call sites — `V4_ROADMAP.md` §4 constraint 3.

**P8 — Teach `/tidy` the dashboard rule.** The extraction filter landed:
`facts._reject_reason` (`facts.py:823`) now returns `"status-dashboard line"`
(`facts.py:852`). But it is forward-only, and `facts.py:862` says so explicitly —
"Read-only callers must NOT use it to filter what is already stored." Meanwhile
`V4_ROADMAP.md` §1.2b records `ENERGY LEVEL: 88% → 92%` and
`PROTECTION LEVEL: 1,000,000,000%`, and §1.1 records invented timestamps, sitting
in the live store being re-injected as established fact. Neither
`_tidy_removal_rule` (`commands.py:822`) nor `_tidy_flag_rule`
(`commands.py:838`) matches a dashboard line. **This must land before or with P3:**
injecting a real clock beside stored facts asserting fake times gives the model
two contradictory clocks.
*Risk:* keep it a **flag**, never an auto-removal. `V4_ROADMAP.md` §4 constraint
6 — the store holds real, deeply personal material.

---

### 2.2 V4.0 — the client (the critical path)

| # | Item | Effort | Depends on | Offline | Core? |
|---|---|---|---|---|---|
| C1 | Co-locate the client as a supervisord program | 1 | — | ✅ | core |
| C2 | The message store: one representation, CAS leaf, audit | 8 | — | ✅ | core |
| C3 | Checked send-set + pre-send fidelity gate | 3 | C2 | ✅ | core |
| C4 | Chat core, with every asset vendored | 6 | C2 | ✅ | core |
| C5 | Received-context echo as response headers, both paths | 2 | — | ✅ | core |
| C6 | The context receipt, as a per-turn strip | 2 | C5 | ✅ | core |
| C7 | The typed system notice catalogue | 2 | C5, C2 | ✅ | core |
| C8 | Single-user session auth + server-side proxy | 2 | C1 | ✅ | core |
| C9 | No task traffic; deterministic titles | 0.5 | — | ✅ | core |
| C10 | Migrate by forking memory, not by importing the transcript | 1.5 | C1 | ✅ | core |
| C11 | Per-fact admin endpoints, lock-correct | 3 | C1 | ✅ | defer |
| C12 | Memory panel v1 | 4 | C11 | ✅ | defer |

**Core subtotal: 28 days. With C11–C12: 35 days.**

**C1 — Co-locate.** Add the client as `[program:client]` in `supervisord.conf`
beside `[program:openwebui]` (`supervisord.conf:113`), with its server talking to
the compactor on 127.0.0.1. Defer `ARCHITECTURE.md` Decisions 2 (own container)
and 7 (own repo) until the client stands. **Why this is first:** `FRONTEND_SPEC`
§3.1 calls the localhost gate a hard prerequisite and blocks every memory feature
on PR #30 *(cited)* — that is only true for a *separate* container.
`_require_localhost` (`main.py:2453`) gates all 20 admin routes, and OpenWebUI
already runs in the same container. A co-located client passes the gate
unchanged, which deletes the CORS ask (**verified:** no `CORSMiddleware` and no
`add_middleware` anywhere in `main.py`) and keeps PR #30 off V4.0's critical path.
*Do not* reach for `COMPACTOR_ADMIN_BIND=0.0.0.0` (`main.py:221`) — it exposes
every admin route unauthenticated.
*Risk:* couples release cadence, which Decision 2 wanted to break, and forgoes
scale-to-zero. Both reversible later at the cost of doing the auth/CORS work then.

**C2 — The message store.** `FRONTEND_SPEC` §11.1–§11.4 as written: message rows
carrying `parent_id`, no serialized chain anywhere, bare-UUID ids never built by
concatenation, per-message delta writes only, leaf moved by compare-and-swap
conditioned on `parent == current leaf`, subtree tombstones, `rev` optimistic
concurrency, and one `audit_conversation()` reporting the five-tuple. Plus the
adversarial suite including the 241-messages/5-roots/leaf-at-depth-8 case.
**This is the only part of the spec that mechanically prevents the 2026-08-24
orphaned tree**, and it is the largest single piece of the client. §11.2's payoff
is literally true: every mechanical step of that failure is rejected by a
constraint.
*Risk:* highest effort and the easiest to under-build. The failure mode is
shipping the schema without the enforcement — a store that permits
`PUT /chats/{id}` "just for the importer" reintroduces the whole defect class.

**C3 — Checked send-set.** §4 rule 7 plus §4.1's pre-send gate: compute the send
set by explicit selection over the current chain, record `window_intent`, verify
present/contiguous/in-order/alternating and count == intent, and on mismatch **do
not send** — raise `context_truncated`. This is the differentiator and it costs
nothing on the server: §12 is explicit that `context_truncated` depends on
nothing, because the 2026-08-24 truncation was entirely client-side with no
budget event and no error *(cited)*. On that day a 241-message conversation sent
7 messages across three turns with a stable conv_id and no signal of any kind.
**Send full history in V4.0** — see §3 for why the bounded window must not ship
first — so `window_intent` is the full chain.
*Risk:* §17 Q9, refuse or warn, is open. Recommend refuse: the operator is one of
the two users, a blocked send is a five-minute problem, and a silent one was a
week-long problem. But see §6.

**C4 — Chat core.** §6's parity rows: SSE streaming, markdown/code/copy, stop,
edit and regenerate as siblings under a shared parent with explicit branch
selection, conversation list, local search. **Vendor every asset** — no Google
Fonts, no CDN; a client that fetches a webfont at first paint does not boot
offline. Two things are already free from the backend: the compactor skips the
memory tail on an incomplete stream, so "never memorize a partial reply" is a
property the client must merely not undo; and regenerate reusing the same
conv_id matches how the compactor keys memory *(cited)*.
*Risk:* scope creep into §13's presentation bars — see §3.

**C5 — Received-context echo.** Custom response headers on
`/v1/chat/completions` carrying resolved conv_id, resolution source, messages
received, messages admitted, exact prompt tokens, headroom, and the guard's
shed/trim counts. **Verified:** the compactor sets no custom response headers
anywhere, so this is new surface — but the data is already computed and thrown
away. `_enforce_hard_budget` (`main.py:1572`) fills `report` with
`{limit, measured, fits, counted_by, ...}` at `main.py:~1946`. This is assembly,
not computation. **One honesty requirement:** on the prescreen path
(`main.py:1673`) `measured` is `None` and `counted_by` reads
`"the char/4 prescreen (nothing was measured)"` — the echo must emit *not
measured* there rather than a number.
*Risk:* two write sites, streaming and non-streaming. `V4_ROADMAP.md` §4
constraint 3 verbatim. Use `StreamingResponse(headers=...)`, not an SSE preamble
event, which OpenAI-shaped parsers choke on.

**C6 — The context receipt.** A one-line collapsible under each assistant
message: *"12 of 65 messages reached the model · 15,155 tokens · 1,229 headroom ·
3 turns shed"*. Pull, not a live dashboard. The field that matters is
server-**admitted**, not client-sent: on 2026-08-28 the client sent 65 and the
compactor admitted 4 (§A.2 stage 5, HTTP 200). A receipt reading "65 sent" would
have been accurate and worthless. **This is also the V4.2 gate** — `V4_ROADMAP.md`
§2 states there is no approval UI without it.
*Risk:* §2.5's forbidden "window". Counts and token figures only; never
narration, never anything about the model's reasoning.

**C7 — Typed notices.** §12's notice types as a visually distinct non-speech
surface, with `context_trimmed` and `context_shed` kept **separate**. The
compactor already synthesizes error chunks into the assistant stream, which
OpenWebUI renders as the assistant talking. The trimmed/shed split is
load-bearing: §A.2 shows the guard shedding 60 of 65 turns and logging it phrased
as a normal operation; collapsing the two would make that read as routine
housekeeping, which is precisely how it read in the logs.
*Risk:* notice fatigue for the user who is not the engineer. She should see
`context_shed`, `chain_corrupt` and `offline`; `request_rejected` and
`backend_fault` can be terser for her than for the operator. See §6.

**C8 — Single-user session auth.** One account, password login, long-lived
session, `SESSION_SECRET` as durable config, `user_id` in the schema from day one,
and copy that distinguishes *logged out* from *no data*. **Correction to one
lens:** it argued auth is a pure addition because `WEBUI_AUTH=false`. That is
`.env.example:283`; **production is `WEBUI_AUTH=true` (`runpod.env.template:156`)**.
There is a login today, supplied by OpenWebUI. Replacing OpenWebUI without auth
is therefore a **security regression**, not a neutral addition, and this item is
core rather than deferrable. The empty-state requirement is not theoretical: the
failure this whole codebase is organized against is a user concluding the system
forgot her, and an empty conversation list after a secret rotation is that same
experience with a different cause.

**C9 — No task traffic.** The client makes exactly one kind of call: a real user
turn. Titles from the first ~60 characters of the first user message, editable
inline. No auto-tagging, no suggested follow-ups. **Highest value-per-day item in
the plan.** OpenWebUI's task calls sent `messages: []` and one-message calls that
hashed to a stable conv_id and accumulated 105 facts in a phantom conversation
*(cited)*. It also spends GPU on the shared A40 that the actual conversation
needs. Combined with P7, the phantom write path closes from both ends.
*Risk:* auto-titles are pleasant and the companion user may miss them. If it
matters, generate the title on the async tail from the exchange the model already
produced — free, off the hot path, and it never creates a phantom conv_id.

**C10 — Fork, don't import.** At cutover the client generates its UUIDv4 and
calls `POST /admin/conversations/{old}/fork` (**verified at `main.py:3916`**,
handler `admin_fork_conversation` at `:3919`, and it accepts a caller-supplied
`new_conv_id` at `:3934`). Facts, summary state, episodic index and persona come
across. The OpenWebUI transcript stays where it is, read-only, as a browsable
archive. **Why:** memory continuity — the part she actually experiences as "she
remembers me" — is a single API call, against ~7 focused days for §16.2's
importer. `INCIDENT_2026-08-28.md` §A.7 confirms what fork carries is intact:
"105 facts, 98 episodic documents, L1=5 … Memory was never the problem in either
incident this week." The thing that was damaged is precisely the thing fork does
not touch.
*Risk:* she loses scrollback continuity. That is a real loss and §16.4 currently
makes verified transcript migration a cutover criterion. **Her call, not an
engineering one** — see §6. Second risk: fork carries the polluted fact store
forward, so run P8's `/tidy` sweep **before** forking, not after.

**C11 — Per-fact admin endpoints.** `POST`, `PATCH`, `DELETE` on
`/admin/conversations/{id}/facts/{ref}`, every handler taking `conv_lock` around
load-modify-write. **Verified, and worse than the spec says:** exactly two facts
routes exist — a `GET` at `main.py:3581` and a `DELETE` at `main.py:3590`. The
`DELETE` calls `_clear_all_memory` (`main.py:3601`, defined `:3607`), which wipes
facts, episodic, summary state **and** persona for the conversation.
`FRONTEND_SPEC` §5.4's table describes it as "forget facts (all/matching)"; that
is wrong against the code. There is no add route and no edit route. **So the
entire §7 facts panel — the feature that justifies building a client at all —
has no server surface today.** The locking is not optional: v3.1 F22 is recorded
at `commands.py:~222` — an unlocked `/remember` raced the tail's locked write and
"the user watched the compactor confirm *Remembered: …* and the fact was gone by
her next turn."
*Risk:* `V4_ROADMAP.md` §4 constraint 6. Delete must archive rather than unlink;
edit should archive the prior text. Facts have no id — the shape is
`{text, added_turn, last_used}` (`facts.py:125`) — so identity is the text and
duplicates are ambiguous. See §6.

**C12 — Memory panel v1.** Facts table with inline edit/delete/add, archive
browse-and-restore, read-only summary stack and persona. **Cut from v1:** persona
library, dedup trigger, export/import, fork — all of those endpoints exist
(`main.py:3680`–`3916`) and are operator tools reachable by curl. The cut is
defensible because the daily user is not the operator. What she cannot do without
a GUI is see and fix the ~90-fact store that §1.2b shows is polluted with
model-fabricated dashboards.
*Risk:* §17 Q5 — depth before it becomes the forbidden window. Read-only
summaries and persona are the right line for v1. Note `/tidy` and `/retire`
already exist for bulk cleanup; the panel should not duplicate their planning
logic.

---

### 2.3 V4.0 — the code lane (parallel; different kind of time)

| # | Item | Effort | Depends on | Offline |
|---|---|---|---|---|
| K1 | Code-review eval from this repo's own labeled defect corpus | 2 | K2 | ✅ |
| K2 | Read-only repo symbol index (stdlib `ast`), addressed by name | 2 | P1 | ✅ |
| K3 | Suppress memory injection **and** the extraction tail in code mode | 1 | C9's mode marker | ✅ |
| K4 | Per-mode generation reserve — measure first | 1 | — | ✅ |

**Subtotal: 6 days.**

**K1 — The eval, before the feature.** Freeze a fixture of known defects with
known locations, feed each region to the production model at production
temperature, and score: did it find the defect, did it invent one, did it emit
parseable output at all. **This repo has something almost no repo has: a large,
pre-labeled defect corpus.** `REMEDIATION.md` holds ~60 findings with file:line;
`MEMORY_REVIEW.md` §3 holds R-1..R-7, S-1..S-6, I-1..I-4 with severity, file,
line and user-visible consequence; `REVIEW_PLAN.md` §1 holds the four gate
blockers. The falsification bar is blunt and fair: **if the on-pod reviewer
cannot find defects that are already documented with their line numbers, it will
not find undocumented ones.** Build the fixture from git history, not from the
working tree — see §7, two of the candidate defects are already fixed.
*Risk:* it returns a bad number and two days look wasted. That is the item
working. Two days that kill this lane are cheaper than the ten that build it, and
`REVIEW_PLAN.md` §8 warns specifically: do not let an empty result read as a
clean bill.

**K2 — Symbol index.** Walk the tree with `ast`; emit
`{symbol, file, line span, byte span, exact token cost}` for every function and
class, costed with `count_text_tokens_exact` (`main.py:795`). **Selection by name
and grep, not by similarity.** **Verified:** `main.py` is 199,346 bytes / 3,994
lines — roughly 55–60k tokens, about 3.5x the entire 16,384-token input budget
and ~7x the conversational half of it. `commands.py` is 97,676 bytes / 2,175
lines. Whole-file review is arithmetically impossible; symbol-level review is
comfortable. Symbols over embeddings because `REMEDIATION.md` already carries a
warning to re-anchor by symbol name rather than line number *(cited)* — and §7
below demonstrates exactly why.
*Blocker, unresolved:* the index must live where the model can reach it, and
**the pod has no repo.** The `Dockerfile` COPYs 18 individual `.py` files to
`/opt/compactor` (lines 221–238) — no tests, no `.md`, no `.git`. See §6.

**K3 — Code-mode suppression.** A code-mode request gets no facts, no episodic
retrieval, no summary block, and — critically — **does not run the extraction
tail.** Two grounded reasons. *Budget:* `INJECTION_BUDGET_FRACTION = 0.5`
(`main.py:166`) means ~8,192 tokens of the 16,384 go to persona and ~90 personal
facts while the model is asked to read a diff. *Contamination:* a code
conversation would write facts about `main.py` into a store `V4_ROADMAP.md` §4
constraint 6 describes as real, deeply personal material — and P7 alone does not
close this, because a code conversation *does* have conversational history. It
also closes constraint 5's loop for the code lane.

**K4 — Per-mode generation reserve.** Make `COMPACTOR_GENERATION_RESERVE`
per-request-mode. **Verified:** `HARD_INPUT_LIMIT = min(MAX_MODEL_LEN, max(256,
MAX_MODEL_LEN - GENERATION_RESERVE))` (`main.py:88`) with `MAX_MODEL_LEN=32768`
and `COMPACTOR_GENERATION_RESERVE=16384` (`runpod.env.template:50, 65`) — so the
entire input budget for persona, facts, retrieval, summaries, conversation and
any code is **16,384 tokens**. If code-mode replies measure ~1.5–2k, a 4,096
reserve raises input budget to 28,672: a 75% increase from a per-request
parameter. **Measure before changing.** `INCIDENT_2026-08-28.md` §A.6.1 records
this exact advice nearly shipping a regression, with the general form: *a
headroom parameter cannot be tuned from a table that does not contain the thing
it reserves for.* One mitigating asymmetry that makes the experiment cheaper here
than it was there: a truncated patch is loud (an invalid diff that will not
apply), where a truncated companion reply was silent.

---

### 2.4 V4.1

| Item | What | Effort | Depends on | Offline |
|---|---|---|---|---|
| `/review <symbol\|diff>` — mechanical-first | Deterministic checks are the finding; the model gets one call to narrate, under a header marking it commentary | 3 | K1–K3, P1 | ✅ |
| Patch proposal as text only | Unified diff in the chat reply; compactor writes nothing | 3 | K1, `/review` | ✅ |
| Per-message token cost | Measured exactly on the async tail, not the hot path; older messages scaled and visibly labeled | 1.5 | C5, C6 | ✅ |
| Per-turn injection snapshot | Persist what was injected per turn; `GET .../turns/{n}/injected` | 3 | C5; overlaps Decision 4 | ✅ |
| Voice in and out | §9 — STT :9000, TTS :9001, transcript confirmed before send | 3 | C4 | ✅ |
| Accessibility, theming, PWA pass | §13's presentation bars, done deliberately as one pass | 4 | C4 | ✅ |
| Test generation for pure functions | Accepted only by mutation: break the code, keep the test only if it goes red | 3 | K1; mutation harness | ✅ |
| Full transcript importer | §16.2 as specified — **only if** fork proves insufficient | 7 | C2; a decision | ✅ |

**`/review`'s design inverts the usual one, and this repo's history says it
should.** `REVIEW_PLAN.md` §1: every blocker across four gates was "invisible to
reading and obvious to running." Review-by-inspection has a demonstrated hit rate
near zero here; the mechanical checks have a demonstrated hit rate of
five-for-five (`INCIDENT_2026-08-28.md` §B.3). So the machine finds and the model
explains. The **sibling-call-site grep** is the highest-value check in it:
`V4_ROADMAP.md` §4 constraint 3 names that as the recurring failure of this
codebase, and it is precisely the cross-file reasoning a 24B at a 16k budget
cannot do but `grep` does perfectly. This document found two live instances while
being written — `_async_tail` at `main.py:3297` and `:3424`, and `metadatas=` at
`retrieval.py:409` and `:653`.
*Risk:* a 24B saying "this looks correct" must never be renderable as a pass —
that is C1's fluent degradation in a new surface. If the mechanical checks did not
run, the command says so and refuses to print commentary. Second risk:
`commands.py` is already 2,175 lines and holds `/tidy` and `/retire`; a third
intricate flow argues for a new module.

**Patch-as-text, and the approval primitive already exists.** The compactor writes
nothing. If a write path is ever wanted, do not invent an approval mechanism —
reuse `/tidy`'s dry-run → content-hash → `apply <code>` pattern
(`commands.py:~1103`), which is a compare-and-swap over a hash of the plan. That
is exactly the property a patch needs: the file must not have changed since the
diff was shown. `V4_PLAN.md` is explicit that the compactor's pod mounts the
volume holding model weights and every conversation's memory, so any filesystem
write from that process is a memory-destroying operation by accident rather than
by malice.
*Risk:* diff formatting is constrained output from a roleplay finetune at high
temperature. Expect malformed diffs. `git apply` failing loudly is the good kind
of failure; do **not** have the model fix its own patch — see §3.

**Voice is the first thing to cut if V4.0 slips**, at the cost of a real
regression for one of the two users. Both services already run under supervisord
(`supervisord.conf:144, 166`). Check with her before assuming it is optional.

---

### 2.5 V4.2

| Item | What | Effort | Depends on | Offline |
|---|---|---|---|---|
| The tool loop, Tier A: `now()` + `calculate()` | Bounded ReAct per `V4_PLAN.md`, two pure-local deterministic tools | 3 | P3, P4, P5; a settled loop-budget design | ✅ |
| Human-in-the-loop approval UI | The action legible before it runs, showing the exact admitted context | 4 | C6; the tool loop | ✅ |
| Split client into its own container/repo | `ARCHITECTURE.md` Decisions 2 and 7, taken when there is a reason | 3 | PR #30, CORS, auth | ✅ |
| Semantic code search over the symbol index | Separate collection, separate root, content-hash id, distance floor, similarity-rank shedding | 2 | K2; R-4/R-6 fixed | ✅ |

**The loop budget is the thing to design before writing the loop, not after.**
`V4_ROADMAP.md` §5's open question is unanswered: every existing budget assumes
one model call per user turn, and that assumption is load-bearing in
`compact_if_needed` (`main.py:1194`), `_enforce_hard_budget` (`main.py:1572`) and
the calibration path. A loop breaks it on day one.

**Approval built before the receipt is trustworthy produces an approval screen
showing a context the server did not actually use — worse than no approval
screen.** §2 obligation 6 makes this a correctness requirement, not UX.

**Semantic code search inherits two verified-live episodic defects** — no
distance floor is applied anywhere in `retrieval.py` (R-4), and
`format_retrieval_block` sorts by `turn_index` and sheds when the budget runs
out, i.e. by chronology rather than similarity (R-6, `retrieval.py:765`,
`:773-790`). Chronological shedding is meaningless for a repo. If it cannot be
kept in a separate collection under a separate root, do not build it.

---

## 3. What I recommend against

The constraint is one engineer's time. Each of these is a real idea with a
grounded reason not to do it now.

**1. `FRONTEND_SPEC` §4 rule 2's bounded window in the first release.** It would
silently kill the summary hierarchy — the same failure class the spec exists to
prevent, caused by the spec. The compactor has no transcript of its own: the
tail feeds `summarizer.maybe_rollup` from the client's own message array, and
`_needs_l1_rollup` fires when `current_turn_count - last_summarized_turn >=
L1_CHUNK_SIZE` with `L1_CHUNK_SIZE = 20` (`summarizer.py:63`) and
`KEEP_RECENT_TURNS = 4` (`main.py:67`). Align the window to the window as rule 2
instructs and `current_turn_count` is a constant ~4, the watermark reconciles
down to 4, and the delta is permanently 0 — no L1 rollup ever fires again. Worse,
turns older than the window never reach the compactor at all, so there is nothing
to summarize even if the gate opened. §15 names the turn-accounting half of this
*(cited)*; no server change to counters fixes the content half. Keep sending full
history in V4.0 — the compactor already compacts it, so the cost is bandwidth and
latency, not context. Retain `window_intent` and the pre-send gate with intent =
the full chain; **the checked construction is what prevented 2026-08-24, not the
boundedness.**

**2. Splitting the client into its own container or repo before it works.**
Following `ARCHITECTURE.md` Decisions 2 and 7 literally converts V4.0's zero auth
work into PR #30 landing, extending `COMPACTOR_API_KEY` past `/v1/*` to
`/admin/*`, retiring `_require_localhost`, adding `CORSMiddleware` (there is none
today — verified), and holding a bearer key in a server the browser talks to.
Those decisions were made 2026-06-09 for GPU scale-to-zero and independent
release cadence, neither of which pays for anything in a two-user deployment
running one pod. It is a V4.2 exposure task, not a V4.0 prerequisite.

**3. Exact per-message token counts in the request echo, as §15 specifies.** §15
asks for a parallel array of exact per-message counts from `/tokenize`
*(cited)* — one HTTP round trip per message on the hot path, against
`count_tokens_exact`'s own stated discipline at `main.py:846`: never on the
request hot path, never per-message. On a 65-message payload that is 65 localhost
round trips before the user sees a token. Get the same finding — the ~4,275
decorative tokens in one reply — by measuring the assistant reply exactly on the
async tail, where latency is free, and labeling the rest as scaled estimates.
*Mark estimates visibly, in the UI and not only in a tooltip:* an
authoritative-looking wrong number is what v3.0.4 shipped *(cited)*.

**4. §13's performance bars as release gates.** "Initial JS < 200 KB gzipped",
"60 fps on a mid-range laptop", and "a 500-message conversation scrolls without
virtualization jank" are bars borrowed from a product with users. This one has
two, on known hardware. The last two are also in tension — 500 messages without
virtualization is exactly where a naive list drops frames. Keep §13's
**correctness** bars (context fidelity, no silent state mutation, budget
verification, diagnosability) as blocking; treat the performance numbers as
targets to measure and report.

**5. A package dependency on the Doulos libraries.** Adopt the tokens — copy the
CSS custom properties into the repo — and skip the component package. §14 already
concedes the islands model contributes shell, routing and tokens rather than the
interactive core for a chat client *(cited)*, and the offline requirement makes a
build that resolves private packages from an external registry a coupling worth
avoiding for a pod that must build and boot without internet. §17 Q7 wants a
spike; the cheapest honest answer is "take the tokens, leave the packages."

**6. A memory dashboard, live retrieval visualization, or anything resembling
inner state.** §7's test is the one to hold: the view must be diffable, never
interpretive, and it must be pull, not a continuously running dashboard. A panel
narrating *why* a fact was chosen is the window `COGNITIVE_ARCHITECTURE.md`
refuses, and it would be built for an audience of one engineer while being shown
to a user who is not one.

**7. Automatic repair of a corrupt chain.** Worth stating as an explicit non-goal
because it is the intuitive thing to build and it is the mechanism of the
incident. §4.1's earlier wording authorized a silent pointer move, and a silent
pointer move is what put a 241-message conversation on an 8-message branch.
Quarantine and report, even though it means she sometimes faces a
branch-selection dialog she did not ask for. On 2026-08-24 the alternative was
that nobody faced anything and the conversation was gone.

**8. Multi-user auth beyond the schema column.** `user_id` in the store from day
one is nearly free and worth having. Account creation, roles, and per-user admin
are not: there are two users, one of whom operates the pod. §17 Q6 asks when
multi-user ships; the honest answer is "when there is a third user", and no
document here contains that trigger.

**9. Tier-C sandboxed shell execution as part of the code lane.** `V4_PLAN.md`
correctly calls this the dangerous one and gates it on docker-in-pod or nsjail —
weeks of the highest-risk work in the plan. For code it buys nothing: the pod's
image contains 18 `.py` files, no tests, no `.md`, no `.git`, so there is nothing
on the pod to run checks against, while the owner has a machine with the full
repo and a shell. Building a sandbox so a 24B can run `pytest` on a pod with no
tests is a real cost against an imaginary benefit.

**10. An agentic multi-file refactor loop.** Three independent disqualifiers.
*Arithmetic:* `HARD_INPUT_LIMIT` is 16,384 and `main.py` alone is ~55–60k.
*Aim:* the defect class it targets is the one it is worst at — sibling call sites,
which need whole-repo visibility the window does not permit and which `grep` does
perfectly for free. *Budget:* the loop-budget question is unanswered.

**11. The model reviewing code it generated in the same session.** `ROADMAP.md`
already cites the Self-Correction Blind Spot against the compactor's own memory
processing — an LLM misses ~64.5% of errors in its own output that it would catch
from an external source *(cited)*. Generate-then-self-review is that pattern with
a shorter fuse and a more confident output format. The external source here is
mechanical and already specified: `ast`, `pyflakes`, the import gate, `git
apply`, pytest under mutation.

**12. Any write path from the model to the compactor's own source.** The obvious
first thing to try — "she can fix her own bugs" — and the one the project has
already ruled on in principle. `COGNITIVE_ARCHITECTURE.md` holds unanchored
self-modification of capability with the gravest caution, and the operative rule
is that the constraint is on the blast radius. The concrete blast radius is
specified in `V4_PLAN.md`: the compactor mounts the volume holding every
conversation's memory.

**13. Embedding-based code search as the *first* context mechanism.** "Show me
`_enforce_hard_budget`" is answered exactly by ~200 lines of stdlib `ast`,
deterministically, offline, with no model call and no vector store. Do the
deterministic thing first and let a measured shortfall justify the fuzzy one.

**14. A public-internet webhook as the alerting mechanism.** Directly contradicts
the offline requirement. `alert.py`'s design is fine and its no-op-when-unset
behaviour is fine, but "do nothing" is the wrong default for a system whose
recorded failure is that the user was the monitoring. Local file sink as
default; a LAN endpoint, if used, is strictly additive.

**15. Lowering `GENERATION_RESERVE` globally to buy injection room.** This will be
re-proposed the moment someone notices 8,192 tokens for all memory feels tight,
and the lever is right there. §A.6.1 records this exact mistake nearly shipping.
Her replies measure 7,513–11,347 tokens. Cutting the global reserve trades a
context bug for truncated replies mid-sentence. Per-*mode* (K4) is the safe
version of this idea.

**16. `memory_query()` as a first tool.** `V4_ROADMAP.md` §3 lists it beside
`now()` and `calculate()` as if the three were comparable. It is not — it is the
seed of the entire V5 pull architecture, and `MEMORY_REVIEW.md` §6.0 measures the
prize and finds it small *(cited)*: the whole push footprint is ~5,500 tokens,
while the dominant per-turn cost is compaction re-summarizing the resent history,
which a lookup interface does not touch. Ship the two cheap deterministic tools,
get the eval, then decide.

**17. Building server-side `turn_seq` now as a standalone item.** It is a genuine
root cause and §15 marks it required *(cited)*, but it is required **by** the
client, and building it first means designing a migration for a message shape
nothing yet sends, against a live store. `_cutoff_is_out_of_frame`
(`retrieval.py:330`) currently holds the failure on the over-inclusive side
rather than the silent side, which is the right place for it to wait.

**18. Dropping OpenWebUI's plugin system without noticing the one plugin in
use.** The parity matrix drops plugins to out-of-scope and that is right — but
`pipelines/conversation_id_header.py` is installed and load-bearing: it writes
OpenWebUI's chat_id into `body['metadata']['chat_id']` so the compactor resolves
conv_id by metadata instead of the hash fallback, and its own header explains
that Functions cannot inject HTTP headers in-process. The new client sends
`X-Conversation-Id` natively, which retires it properly — but **the migration
checklist must confirm the header path is actually being taken**, because on
2026-08-24 the compactor logged `source=hash` and OpenWebUI had no way to know.
C5's echo makes that checkable.

---

## 4. The V4/V5 boundary, as a rule

The owner's rule is that V5 is APIs and external services. Operationally:

> **A feature is V5 if its correct behaviour depends on a system the owner does
> not run.**

Apply it with one test:

**The unplug test.** Disconnect the pod's uplink and cold-boot it. If the feature
must now refuse, degrade, or lie, it is V5. If it works exactly as designed, it
is V4-eligible.

Three clarifications that resolve most edge cases:

1. **"External" is about ownership, not about the network.** A webhook to a
   machine on the owner's LAN is V4-eligible, because the owner runs both ends
   and the unplug test passes. A webhook to Slack or Discord is V5, because it
   fails the unplug test and because it is a phone-home. This is the distinction
   that matters, and "no network" is the wrong way to state it.

2. **Build-time and boot-time count.** A dependency resolved from a public
   registry at build time, or a font fetched at first paint, fails the unplug
   test just as surely as a runtime API call — it just fails later, on the day
   the owner rebuilds without internet. This is why assets get vendored (C4) and
   why the component package gets declined (§3 item 5).

3. **Processes the owner starts under supervisord are inside the boundary.**
   vLLM, STT, TTS, the compactor, and the new client all talk over the network
   and all pass the unplug test. A second small model on CPU in its own venv
   would too.

**The requirement is mostly self-enforcing, which is the point.** Almost every
V5-shaped idea fails the unplug test mechanically, so the boundary does not
depend on anyone's judgment about what counts as an integration. The residual
cases that need judgment are the ones where a feature works offline but is
*designed for* an external service — and those are rare enough to handle one at a
time.

---

## 5. The critical path, and what runs beside it

**Critical path (28 focused days, ~3.5 months at 2 days/week):**

```
C1 co-locate (1)
  └─ C2 message store (8)          ← the long pole; a month of calendar alone
       ├─ C3 checked send-set (3)
       └─ C4 chat core (6)
            └─ C8 auth (2) ─ C9 no task traffic (0.5) ─ C10 fork migration (1.5)
                 └─ CUTOVER
C5 echo (2) ─ C6 receipt (2) ─ C7 notices (2)     ← joins before cutover
```

C5 is the only client item with **no** dependency on C2, so it can start on day
one. C6 and C7 both hang off it. Nothing in that trio blocks C2, and all three
are compactor-side or presentation work rather than store work — genuinely
different hours.

**Runs fully in parallel, blocking nothing:**

- **All eight prerequisites (P1–P8, 4.5 days).** Server and ops only. P1 should
  be literally first — it is half a day and every later item benefits.
  Constraint: **P8 lands before or with P3** (two clocks), and **P8 runs before
  C10** (do not fork a polluted store).
- **The code lane (K1–K4, 6 days).** Independent of the client entirely. K2 is
  blocked on an unmade decision — does the pod get the repo (§6).
- **C11–C12 (7 days).** Server endpoints then panel. Deferrable within V4.0; the
  facts panel is the differentiator but nothing else waits on it.

**The single decision that most changes the schedule** is C10 vs the §16.2
importer. Fork is 1.5 days; the importer is 7. That is 5.5 days — nearly a month
of calendar — and it buys scrollback, not memory. It is not an engineering call.

**Total V4.0 as specified: 45.5 focused days.** Core-only is 28. I am not going
to present 45.5 as comfortable: it is roughly five and a half months part-time,
and C2 alone is a month of it. If that is too long, the cut line I would take is
C11–C12 (7 days) and the voice pass, in that order — not C5/C6/C7, which are
cheap and which are the entire reason the client is worth building.

---

## 6. Open questions that must be settled before V4.0 starts

Ordered by how much downstream work each one blocks.

**1. Did D1/D2 actually restore context in production?** `V4_ROADMAP.md` §5 step
1 is still open. Every judgement about the "not good with numbers" complaint,
about which mechanism causes "forgetting who she is", and about whether the code
lane's budget arithmetic is even right, is unresolvable until the logs are
re-pulled and the guard is confirmed to have stopped shedding 60+ turns per
request. *Settles it:* one log pull. **This blocks the most and costs the least.**

**2. Does she accept losing scrollback continuity at cutover?** Fork gives her a
client that remembers everything about her while showing an empty thread, with
241 messages readable in a frozen OpenWebUI. The importer buys the scrollback for
~5.5 additional days. §16.4 currently makes verified transcript migration a
cutover criterion — that criterion should be **re-decided with her**, not
inherited. Her call.

**3. Does the pod get the repo, and how?** The `Dockerfile` COPYs 18 `.py` files
and nothing else. There is nothing on the pod to review, index, or check. Options:
mount the repo read-only, ship a symbol index as a build artifact, or accept that
the client sends the code and the pod holds none. Unmade, and it gates K2, K1 and
`/review`.

**4. Can Cydonia-24B emit a well-formed unified diff or structured review output
at production temperature?** Flagged twice in the existing documents and never
answered empirically *(cited: `V4_PLAN.md` open question 1, `MEMORY_REVIEW.md`
§8.4)*. One afternoon against the production pod answers it, and the answer gates
every code-creation item and most of `/review`. Related: **should the code lane
use a second small model on CPU in its own venv?** `ROADMAP.md` already endorses
that pattern for memory judgment; it costs RAM, a second venv, and another
supervisord program.

**5. Refuse or warn on a pre-send fidelity mismatch (§17 Q9)?** My recommendation
is refuse — the operator is one of the two users, a blocked send is a five-minute
problem, a silent one was a week-long problem. But refusing means a store bug can
lock her out of talking to the system entirely. **Is there an override she can
reach without the engineer?** That sub-question is the one that actually needs an
answer.

**6. Which of the two users is the client's primary audience when they
conflict?** The receipt, per-message cost and the five-tuple diagnostic are
operator surfaces; the facts panel and archive are hers; §12's notices mix both.
This determines whether the receipt is visible by default or behind a toggle, and
§17 Q11 cannot be settled without it.

**7. What is the summary state on the live pod right now?** `INCIDENT_2026-08-24`
records L1=5, L2=0, L3=none *(cited)*, which is why P5 is not yet firing. If L1
has filled toward 10, P5 moves up sharply. *Settles it:* read `/health/full`'s
l1/l2 counts. One minute.

**8. What alert sink does the owner have that works with no internet?** Decides
whether P4's local file sink is the whole design or just the fallback. Five-minute
conversation, blocks a day and a half of work.

**9. Do facts need stable ids, or is exact-text matching with an occurrence count
enough?** Facts are `{text, added_turn, last_used}` (`facts.py:125`). Adding an id
touches the write path, dedup's merge, the archive sidecar, the export bundle
format and the importer. Exact-text avoids all of that but is ambiguous on
duplicates — which is precisely what dedup exists for. Blocks C11.

**10. What is the measured code-mode reply length?** Nothing in this repo can
answer it, and §A.6.1 is explicit that guessing here nearly shipped a regression.
One measurement pass over ~30 code-mode replies. Blocks K4.

**11. Does she use regenerate?** Regeneration supersession is unshipped, and its
effect is that the store keeps the reply she **rejected** at the lower ordinal
and the kept one higher, so the recency filter preferentially preserves the
rejected text forever *(cited)*. If she regenerates often this jumps several
places; if never, it stays a design item.

---

## 7. Corrections — three lens claims that are wrong at HEAD

Recorded because two of them would have cost real time, and because the pattern
matters more than the instances.

**1. `MEMORY_REVIEW.md` R-1 (colliding episodic `_doc_id`) is FIXED.** One lens
recommended against embedding-based code search partly on the grounds that it
would inherit a silent-destruction id collision. `_doc_id` (`retrieval.py:205`)
is now `{conv_id}::{sha256(document)[:16]}` — content-addressed as of v3.1 D1,
with the production evidence recorded in its own docstring. `_id_exists`
(`:237`) makes a re-send idempotent and `_next_turn_index` (`:293`) cannot pull
the sequence backwards. **R-4 (no distance floor) and R-6 (shedding by
chronology, `retrieval.py:765`/`:773-790`) are both still live** — the argument
survives on those two, but not on R-1.

**2. `MEMORY_REVIEW.md` I-1 (the trim loop with no `protect_system` exclusion) is
FIXED.** It is listed as an S1 appearing in neither `REMEDIATION.md` nor the
incident report, and one lens proposed it as the deliberate stretch case in the
code-review eval. At HEAD the `big` list comprehension (`main.py:1801`) filters
through `_droppable_system_indices(msgs, protect_system)`, and the comment above
it records the fix explicitly. **Build K1's fixture from git history, not from
the working tree.**

**3. `WEBUI_AUTH` is `true` in production, not `false`.** One lens read
`.env.example:283` and concluded there is no login today, making C8 a pure
addition. `runpod.env.template:156` is `WEBUI_AUTH=true`. Replacing OpenWebUI
without a login is a **security regression**, which is why C8 is core rather than
deferrable.

**The pattern is the finding.** Two of five spot-checked `MEMORY_REVIEW.md`
findings were stale, and the stale ones were stale because v3.1 shipped. Those
documents were written against a moving branch and their line numbers have
drifted — the I-1 citation `main.py:805-825` now lands in the middle of
`count_text_tokens_exact`. **Re-verify any `MEMORY_REVIEW.md` or `REMEDIATION.md`
finding against HEAD before planning a day of work around it.** This is the same
discipline `REVIEW_PLAN.md` §1 states as its governing rule — a claim is
unreviewed until something has executed — applied to the review documents
themselves. It is also, precisely, the job K2's symbol index and `/review`'s
mechanical checks exist to make cheap.
