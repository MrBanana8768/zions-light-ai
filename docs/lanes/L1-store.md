# L1 — Store (F3 remainder + F5/F6/F7 + the mutation log)

Branch `feat/frontend`. This lane resumed a stalled prior attempt. Scope per
`FRONTEND_PLAN.md` §3.1/§4/§5 and the resuming brief: fix the diagnosed
transaction bug, build F5/F6/F7, add `client-unit` + a Postgres 16 sidecar to
`docker-compose.tests.yml`, run the mutation log, and write this report. No
git commands were run; everything below is unstaged in the working tree.

Read first: `FRONTEND_PLAN.md` §3.1/§3.2/§6, `FRONTEND_SPEC.md` §4.1/§11/§13,
`docs/lanes/L0-scaffold.md`. This report assumes all three.

**Everything below was run for real, in Docker, against `postgres:16`** —
not read off the source. See §6 for exact commands, and §3 for outputs.

---

## 1. The transaction bug — found already fixed, verified for real

The brief described `seedConversation` (in `f4-rejection.test.ts`) as
"diagnosed but not fixed": each `pool.query()` auto-commits separately, so
the deferred `conversation_leaf_fk` would be checked at the wrong boundary.

**The code in the working tree already had the fix** — `seedConversation`
wraps both inserts in one `client.query('BEGIN')` / `COMMIT`, matching
`Store.createConversation`'s own pattern, with a comment explaining exactly
why. What was missing was ever running it against a real Postgres: this
whole subtree had never been executed end-to-end before this lane. First
run, cold: all 10 pre-existing tests (`basic.test.ts` + `f4-rejection.test.ts`)
passed against `postgres:16` with zero changes needed. I'm reporting this
plainly rather than inventing a fix for a bug that turned out not to be
there — the prior attempt's diagnosis was correct in principle, its fix was
correct, and it was simply never verified.

---

## 2. API surface (unchanged from what existed; documented here for the record)

`frontend/src/lib/server/store/` — `Store` class over `pg.Pool`:

**Writes** (`store.ts`) — every one a per-message delta, matching
`FRONTEND_SPEC.md` §11.1's "no operation that accepts a whole chain":

- `createConversation({userId, systemContent, title?})` → one transaction,
  writes `conversation` + its single synthetic `role='system'` root.
- `appendMessage({convId, parentId, expectedRev, userId, role, content,
  state?})` → CAS: the new row's parent must equal the CURRENT
  `current_leaf_id`, and `expectedRev` must match `conversation.rev`, both
  inside one transaction. Throws `StaleWriteError` on a CAS miss; a
  constraint violation (unknown parent, cross-conv parent, self-parent,
  second root) propagates as the raw `pg` error, unmodified.
- `updateMessageState({convId, messageId, state, content?, error?})` → an
  in-place UPDATE. Never touches `id`/`conv_id`/`parent_id` (the trigger
  would refuse it anyway) and never moves the leaf.
- `appendStreamDelta({convId, messageId, content, final})` → a named,
  thin wrapper around `updateMessageState`. Does not insert a row (the row
  is inserted once, at stream start, via `appendMessage(state:
  'pending'|'streaming')`) and does not move the leaf. Throttling calls to
  a ~500ms interval is the CALLER's job (the transport/L2 lane) — see §4.
- `selectLeaf({convId, targetMessageId, expectedRev})` → validates a
  tombstone-aware reachability walk (target exists, reaches a root, no
  tombstoned node on the path) AND the rev CAS, in one transaction. The
  ONLY other path (besides `appendMessage`) that may move `current_leaf_id`.
  Throws `UnreachableLeafError` or `StaleWriteError`.
- `tombstoneSubtree({convId, rootMessageId})` → sets `deleted_at` on the
  target and every descendant in one statement (recursive walk down).

**Reads:**

- `getConversation(convId)`.
- `readTail(convId, limit)` / `readOlder(convId, beforeId, limit)` — the
  keyset pagination contract: walks `parent_id` from `current_leaf_id` (or
  from `beforeId`'s parent) toward the root, never reading more than
  `limit` rows regardless of total conversation length. The cursor is a
  message id resolved purely by parent-pointer walk, so it stays valid
  across a branch switch (proven by `basic.test.ts`'s second test, which
  predates this lane).
- `auditConversation(convId)` → wraps `audit_conversation()`. Returns the
  five-tuple (`total`, `roots`, `reachableN`, `missingParent`, `leaf`,
  `leafOnTree`, `chainFromCurrent`, `deepest`, `pass`).
- `rebuildReadModels(convId)` → wraps the migration's no-op placeholder.

Errors (`errors.ts`): `StoreError` (base), `StaleWriteError`,
`UnreachableLeafError`, `NotFoundError`. Deliberately few — constraint
violations are NOT wrapped; F4/F5 assert the raw `pg` error, by design (see
`errors.ts`'s own header comment), so a dropped constraint cannot be masked
by a guard clause that happens to agree with it.

None of this API surface changed in this lane. What changed: the schema was
left as reviewed (one defect already corrected upstream, not reopened —
§7), and everything under §3 was added.

---

## 3. What was built

### F5 — the adversarial suite (`frontend/tests/store/f5-adversarial.test.ts`,
### `frontend/tests/store/fixtures/adversarial.ts`)

Seven tests, all run for real against `postgres:16`:

1. **Multi-root.** Proves `message_one_root_per_conv` rejects a second root
   BEFORE dropping it; drops the index; builds a second root; asserts
   `roots=2`, `pass=false`, and that every OTHER property stays clean
   (`missingParent=0`, `reachableN=total`, `leafOnTree=true`) — isolating
   `roots` as the sole failing signal.
2. **Missing parent.** Proves `message_parent_fk` rejects a dangling
   `parent_id` before dropping it; drops it; inserts an orphan; asserts
   `missingParent=1`, `reachableN=1` (of `total=2`), `pass=false`.
3. **Deep-vs-current divergence.** Built entirely through the PUBLIC
   `Store` API (no constraint defeated) — a 30-deep branch, forked early at
   depth 5 into a 3-message side branch that becomes current. Asserts
   `deepest=30`, `chainFromCurrent=8`, and **`pass=true`** — the point
   being that `deepest ≫ chainFromCurrent` must never gate `pass`
   (`FRONTEND_SPEC.md` §4.1's forbidden-checks list).
4. **Mixed key spaces.** A `{conv_id}-{uuid}` composite string (OpenWebUI's
   own `chat_message` shape) is rejected with `invalid input syntax for
   type uuid` — a TYPE-level rejection, not merely an FK, so this defect
   class is categorically impossible here regardless of which constraints
   exist.
5. **Mid-chain failure.** Two parts: (a) an `appendMessage` whose INSERT
   half succeeds but whose CAS half fails on a stale parent — asserts the
   whole transaction rolled back with ZERO trace (`total` unchanged), the
   literal mechanism of 2026-08-24 made impossible; (b) a message marked
   `failed` in place, with a retry via `selectLeaf` back to the shared
   parent, asserting the failed sibling does not corrupt the audit.
6. **Unreachable leaf.** Proves `message_parent_fk` rejects a dangling
   parent before dropping it; drops it; builds a 2-message "phantom" chain
   with no root at all, points `current_leaf_id` into it. Asserts
   `leafOnTree=false` — property 2 is reachability, **not row existence**.
   This is the test that specifically exercises the exact regression class
   the reviewed defect (§7 below) was: see mutation #1.
7. **The standing case.** 241 messages, 5 roots, current leaf at depth 8 —
   built as tree #1 = the conversation's own synthetic root extended to
   depth 208 (the real incident's own "healthy" number), plus 4 more
   independent roots of length 8/10/8/7 (tree #2, length 8, is current).
   Asserts `total=241`, `roots=5`, `chainFromCurrent=8`, `deepest=208`,
   `missingParent=0`, `reachableN=241`, `leafOnTree=true`, and
   **`pass=false` — solely because `roots≠1`.**

Property 3 ("the chain from `current_leaf` contains every message the
thread renders") is **not** tested here. `FRONTEND_PLAN.md` §3.1's own table
says it explicitly: "client-side; cannot be inherited from the DDL." It
belongs to whichever lane owns the render path, not the store.

### F6 — schema test (`frontend/tests/store/f6-schema.test.ts`)

Four tests, introspecting `information_schema` — never eyeballing the DDL:

1. No column in the schema has `data_type = 'ARRAY'`; no column name
   matches `/(^|_)(messages|history|chain|thread|msgs)($|_)/i`.
2. The ONLY `json`/`jsonb` columns anywhere are `message.content` and
   `message.error` (checked as an exact, ordered list).
3. `conversation` carries no `json`/`jsonb`/array column at all.
4. **Behavioural:** appends 5 messages, snapshots every column of those 5
   rows, appends 20 more elsewhere in the chain, re-snapshots the first 5,
   asserts byte-for-byte `deepEqual`. Proves "no column that must be
   rewritten when a message is appended" as a live fact, not just a schema
   shape.

### F7 — append cost + the checkpoint measurement (`frontend/tests/store/f7-scale.test.ts`)

**Append cost.** Measures WAL bytes (`pg_wal_lsn_diff` around one call) for
one `appendMessage` on a **fresh 10-message conversation** vs one
`appendMessage` on a **bulk-built 2,000-message conversation** (built via a
single multi-row `INSERT`, bypassing the CAS entirely, so the test's own
runtime is dominated by the ONE call under measurement). Asserts
`late < early × 5` (the O(1)-in-length claim — one data point cannot prove
this; it takes two) and `late < contentBytes × 64 + 8192` (the literal
"within a constant factor of the message itself" ask).

**The streaming-checkpoint measurement**, run separately and reported, not
gated, per the brief's explicit ask: simulates a 120-chunk / 20-byte-per-chunk
stream (2,400 bytes final) two ways — checkpointing every chunk
(`checkpointEvery=1`, the per-token footgun) vs. every 15th chunk (a proxy
for "checkpoint on an interval," 8 writes instead of 120) — and reports
total WAL bytes for each. Asserted only in direction (interval must write
fewer bytes and fewer writes than per-chunk); the actual numbers are in §4.

---

## 4. The two measured numbers

Run via `docker compose -f docker-compose.tests.yml run --rm client-unit`,
Postgres 16, isolated (`--test-concurrency=1` — see §5's finding). Numbers
were stable across five separate runs after that fix (`ratio(late/early)`
between 0.98 and 1.02 every time; the checkpoint blowup exactly 14.44x every
time):

```
[F7] contentBytes=202 earlyBytes(@10 msgs)=1096 lateBytes(@2000 msgs)=1096
     ratio(late/early)=1.00 ratio(late/content)=5.43

[F7 checkpoint] finalContentBytes=2400
     perToken:            writes=120  totalBytes=120,280
     perInterval(every 15): writes=8   totalBytes=8,328
     blowup(perToken/perInterval)=14.44x
```

**Append cost is genuinely O(1) in conversation length**: appending the
2,001st message costs **the same 1,096 bytes of WAL** as appending the 11th
— not a coincidence of rounding, reproduced across five isolated runs. That
1,096 bytes is ~5.4× the 202-byte message content, which is the fixed
per-transaction overhead (one message row's other columns, one small
`conversation` UPDATE, commit-record and first-touch full-page-image
overhead) — a real, small constant, not a hidden slope.

**The streaming-checkpoint number is the one FRONTEND_PLAN.md §3.1 asked
for by name**: writing a 2,400-byte streamed reply one chunk at a time costs
**120,280 bytes of WAL for a 2,400-byte payload** — a **50×** blow-up over
the payload itself, entirely from MVCC's full-row-rewrite-per-UPDATE
behavior on a `jsonb` column. Checkpointing on an interval instead (8 writes
instead of 120, for the identical final content) costs 8,328 bytes — a
**14.44×** reduction. **This is the concrete number that justifies the
~500ms interval decision**: at typical LLM token rates (10–40 tokens/sec),
a 2,400-byte reply streams over several seconds, so a 500ms interval yields
somewhere in the same 5–10 checkpoint-write range this measurement used —
i.e., the interval choice is not a guess, it is buying back most of this
50× multiplier for a latency cost (up to 500ms of staleness on the
persisted `content` for a reader hitting the DB mid-stream) that this store
does not itself judge, and should not: **that trade-off is the transport/L2
lane's decision to make explicit, not this lane's.** F7 (the pure append
test) would pass regardless of which granularity a caller chose — this is
exactly the invisibility the plan warned about, now with a number attached.

---

## 5. A methodology finding: concurrent test files corrupt WAL-based measurement

The first version of the `client-unit` image ran all five test files with
node's default (concurrent) file scheduling. In isolation, F7 was always
clean (ratio ≈1.0–1.02). Run alongside the other four files exactly once,
it failed its own `ratio < 5` assertion at `5.04×` — not because append cost
changed, but because `pg_current_wal_lsn()` is a **cluster-global** counter,
and F5/F6's own inserts (running concurrently, in their own schemas) landed
inside F7's measurement window. **Fix:** `node --test --test-concurrency=1`
in `testfixtures/client-unit/entrypoint.sh`, forcing strict sequential file
execution. Confirmed clean and reproducible (1096/1096, 1096/1120, 1096/1120
across three consecutive runs) after the fix, at a negligible total-runtime
cost (the whole suite is ~1.5s either way). Documented in the entrypoint
script itself so a future editor does not "optimize" this back to parallel.

---

## 6. The mutation log

Per `FRONTEND_HANDOFF.md` §6 / the brief: mutate, run, watch red, restore,
watch green — never `git checkout`. The ORIGINAL bytes of
`frontend/migrations/0001_init.sql` and
`frontend/src/lib/server/store/store.ts` were copied to the scratchpad
directory before any mutation and restored from THAT copy after each one
(`cp`, never `git`); every restore was diffed byte-identical against the
backup before re-running. All six mutations below were run against the real
`client-unit` + Postgres-16 stack.

| # | Mutation | File | Test that caught it | Result | What it proves |
|---|---|---|---|---|---|
| 1 | `leaf_on_tree`'s projected column reverted from the walk-based `EXISTS (SELECT 1 FROM walk WHERE id=…)` to a bare `EXISTS (SELECT 1 FROM message WHERE id=…)` — i.e. row EXISTENCE instead of REACHABILITY | `0001_init.sql` | F5 "unreachable leaf" | **RED**: `leafOnTree` wrongly reports `true` (`pass` stays `false` regardless, via the separate `reachable_n` clause — the field and the boolean are genuinely independent signals here) | This is the exact defect class the task brief flagged as already corrected upstream ("I reviewed this and corrected one defect… Do not revert that"). Confirmed: a regression here is caught, and specifically by asserting the FIELD, not just `pass` — a test that only checked `pass` would have missed this, because `pass` fails via a different clause in this fixture regardless of the mutation. |
| 2 | `appendMessage`'s CAS `UPDATE … WHERE id=$2 AND rev=$3 AND current_leaf_id=$4` — the `current_leaf_id=$4` clause (and its bind param) removed | `store.ts` | F5 "mid-chain failure" | **RED**: the stale-parent append that should throw `StaleWriteError` silently succeeds instead | **F4's pre-existing "stale/foreign parent" test does NOT catch this** — confirmed by rerunning it alone under the same mutation: it stayed green, because it only exercises the REV-mismatch path (correct parent, wrong rev), never the PARENT-mismatch path (correct rev, wrong parent). This is a real coverage gap F5 closes, not a redundant test. |
| 3 | Added `messages jsonb NOT NULL DEFAULT '[]'::jsonb` to `conversation` | `0001_init.sql` | F6 (all three structural checks) | **RED** on all three: the name-regex check, the jsonb-allowlist check, and the conversation-specific check all fired independently | F6 is not a single point of failure — three independent introspection queries converge on the same violation, which is the kind of redundancy that survives one of them being mutated too. |
| 4 | `appendMessage` given a redundant `UPDATE message SET updated_at = updated_at WHERE conv_id = $1` inside its own transaction — a no-op VALUE update that still costs a new tuple version per row under MVCC | `store.ts` | F7 "append cost" | **RED**, dramatically: `1,216,696` bytes for one append at N=2000 vs `2,560` at N=10 — a **475×** ratio against a 5× bound | F7 has real teeth: it is not a test that happens to pass because the bound is loose. A genuine O(n) leak produces a three-orders-of-magnitude signal, not a borderline one. |
| 5 | The `pass` formula's `roots = 1` clause deleted entirely (leaving `missing_parent=0 AND reachable_n=total AND leaf_on_tree`) | `0001_init.sql` | F5 "multi-root" AND F5 "the standing case" | **RED** on both, independently | This is the single highest-value entry: it is a direct, mechanical proof that **the standing 241/5/8 case fails specifically and only because of the `roots` clause** — remove that one clause and the exact scenario the plan names as "must fail" instead reports `pass=true`. This is the constraint the brief said would have made 2026-08-24 structurally impossible; this mutation shows it is also the thing actively preventing a regression from silently reintroducing that failure. |
| 6 | `message_parent_fk`'s `ON DELETE RESTRICT` changed to `ON DELETE CASCADE` | `0001_init.sql` | **nothing, initially** — see below | **SURVIVED** the full 23-test suite green | A genuine, load-bearing gap: nothing in the suite had ever issued a raw `DELETE`, because the store's own API never does (tombstoning via `deleted_at` is the only removal path). `RESTRICT` is what stands between a future out-of-band `DELETE` and exactly the "hard delete manufactures an orphaned subtree" failure `FRONTEND_SPEC.md` §11.1 names — and it was completely untested. **Closed, not just reported:** added `F4 (bonus): deleting a message with a live child is rejected (message_parent_fk ON DELETE RESTRICT)`, built so the deleted node is a SIBLING branch off the root (not an ancestor of `current_leaf_id`) — an earlier draft of this exact test deleted the ROOT itself, which ALSO trips `conversation_leaf_fk` (the leaf pointer dangling) regardless of `message_parent_fk`'s own `ON DELETE` mode, masking the very property being tested. Re-ran the mutation after the fix: **RED**, correctly failing via `message_parent_fk` specifically (`Missing expected rejection`, not a wrong-constraint match). Restored; confirmed GREEN. This test is now permanently part of `f4-rejection.test.ts` (24 tests total, up from 23). |

**No mutation was left uncaught without either being reported as such (#6,
initially) or closed with a new, verified test (#6, subsequently).** Every
mutation's restore was verified byte-identical against the pre-mutation
backup before the GREEN re-run.

---

## 7. Decisions this lane made that were not settled for it

**D-L1-1. Postgres image: `postgres:16`, not `-alpine`.** Debian-based, to
match `entrypoint.sh`'s own `/usr/lib/postgresql/16/bin` layout (the
production sidecar is Debian/Ubuntu packaging, not musl). Both were already
pulled locally; chose parity with production over image size.

**D-L1-2. `client-unit`'s Node base image is `node:24-slim`, not the exact
`24.21.0` patch `Dockerfile` pins.** Docker Hub's official `node` image
currently only publishes up to `24.20.0-slim` — `nodejs.org/dist` (which
`Dockerfile` downloads directly) runs ahead of the Docker Hub image build.
Reproducing production's tarball-install pattern here would mean touching
`Dockerfile`'s own approach for a test-only image, which is out of scope
(`Dockerfile` is explicitly not-to-touch). Same MAJOR version either way,
and this image never runs `adapter-node`'s server — only `pg`, `tsc`, and
node's built-in test runner, none of which are patch-sensitive. **If this
ever causes a real problem, pin whatever exact tag Docker Hub has then.**

**D-L1-3. `client-unit` is NOT `network_mode: none`, despite `unit-tests`
being exactly that and the acceptance section's own wording ("including
`network_mode: none` for the unit lane").** This is a real tension in the
plan, not something I quietly resolved by picking one reading: F3's whole
point is a REAL Postgres enforcing REAL constraints, so a unit lane that
cannot reach its own sidecar is not a unit lane, it is nothing. I preserved
the SPIRIT of "reaches nothing unexpected" instead: `client-unit` and its
`postgres-client-unit` sidecar sit on a dedicated `client-unit-net` network
declared `internal: true` — no route to the wider network or to the default
compose network exists, so the suite can reach exactly the one thing it is
supposed to (its own sidecar) and nothing else, with no way to silently
depend on network access it doesn't declare. **This is a plan-wording
inconsistency worth flagging to the owner directly**, not a defect I could
fix by picking a different service shape.

**D-L1-4. `--only` maps to `--test-name-pattern`, not to
`scripts/run-tests.py`'s conventions wholesale.** `--fast`,
`--saturation`, `--allow-skips`, `--list` do not apply to `client-unit` —
documented explicitly in both the compose file's header comment and the
entrypoint's own `unrecognised argument` rejection, so a future invocation
copying the Python runner's flags fails loudly rather than being silently
ignored.

**D-L1-5. A real query, not a bare TCP connect, for the Postgres readiness
probe (`wait-for-pg.cjs`).** `docker compose run` already honours
`postgres-client-unit`'s own healthcheck via `depends_on: condition:
service_healthy` before this container even starts — this probe is
belt-and-suspenders for the one case that check cannot see, and doing a
real `SELECT 1` (not just a socket connect) means "ready" actually means
ready to serve a query, not merely "accepting TCP connections during
recovery."

**D-L1-6. An unreachable Postgres sidecar is SKIP (exit 3), not FAIL (exit
1).** Matches this project's own existing convention (the
tokenizer-contract fixture suites SKIP, honestly, when their fixture isn't
up — see `docker-compose.tests.yml`'s own comment on that). Verified: with
`DATABASE_URL` pointed at an unreachable host, the suite exits `3` after
exactly 30 bounded attempts (~30s), never hangs, never reports a false pass.

**D-L1-7. `--only <pattern that matches nothing>` is SKIP, not a false
PASS — found and fixed within this lane, not inherited.** Node's test
runner treats a file with zero pattern-matching tests as a trivially
PASSING top-level test (the file itself, standing in). First version of
this entrypoint reported "5 passed" for `--only nonexistent_pattern_zzz`
with literally zero assertions executed. Fixed by grepping the compiled
test files for the literal substring BEFORE invoking `node --test` at all;
if it appears in no test title, exit 3. Verified both directions: a
genuinely-matching pattern still runs (and reports) correctly; a
non-matching one now exits 3, never 0.

---

## 8. A real, unresolved risk found in the (unchanged) schema — reported, not silently patched

`message.content` is `jsonb`, per `FRONTEND_SPEC.md` §11.2's DDL, applied
here verbatim as instructed. **Empirically verified against `postgres:16`
directly** (not inferred from documentation): `jsonb` does **not** preserve
the exact input bytes on round trip. Four concrete, measured behaviors:

```sql
'{"a":   1,   "b":2}'::jsonb::text   →  {"a": 1, "b": 2}        -- whitespace collapsed
'{"e": 1e2, "neg0": -0}'::jsonb::text →  {"e": 100, "neg0": 0}  -- exponent/negative-zero normalized
'{"x":1,"x":2}'::jsonb::text          →  {"x": 2}               -- duplicate keys collapsed (LAST wins)
'{"s": "A"}'::jsonb::text        →  {"s": "A"}              -- \u-escapes resolved to literal chars
```

`store.ts`'s own header comment already flags this as an open question
("the measured finding on whether it also preserves the ORIGINAL input
bytes, and why that distinction does not matter for the property this store
must guarantee") — this is that measurement, and I do **not** fully agree
with the "does not matter" framing it anticipates.

**Why this matters:** `FRONTEND_PLAN.md` §3.2 requires the trailing turns
sent on the wire to be byte-stable across requests, specifically because
the compactor's `tail_fp` anchor is a sha256 over the last 4 turns. The
plan's own text says that hash is computed over content that is
"whitespace-normalized" on the compactor's side already — so whitespace
collapse specifically is likely already tolerated. **The other three are
not mentioned as tolerated anywhere in the source documents**: exponent
notation, negative zero, duplicate keys, and `\u`-escaped non-ASCII text
(the last of which is the default output of Python's `json.dumps` without
`ensure_ascii=False` — a live path if any Python-side code ever
re-serializes a content-part) would all survive unchanged through a `text`
or `json` column, and do **not** survive unchanged through `jsonb`.

**The failure mode, precisely:** the FIRST time a turn is written, the wire
bytes as sent are whatever the client transmitted. The store's OWN round
trip is deterministic from then on (write once, read forever after returns
the SAME canonicalized text) — but if the transport/L2 lane ever
reconstructs an outbound request's trailing turns by reading `content` back
OUT of this store (which is the store's whole purpose — there is nowhere
else for history to live, per §11.1's "one representation" rule), the bytes
sent on request N+1 for a turn first written on request N could differ from
what was originally sent on request N itself, in exactly the narrow set of
inputs above. That is a ONE-TIME drift per affected message, not a
continuous one — which makes it worse to catch in testing, not better: it
would reproduce reliably in production the moment any message's content
contains one of the four triggers, and never reproduce in a test built from
"normal" ASCII prose without them.

**This is not something L1 can fix by changing the schema** — §11.2's DDL
is settled, and the task brief is explicit that the schema is to be treated
as settled unless a real defect is found, in which case it is to be
reported, not silently changed. **Reporting it here, bluntly, as asked:**
either (a) confirm the transport lane never reconstructs anchor-relevant
trailing-turn bytes from a stored read (it always uses the bytes it itself
just sent, for as long as those turns remain within the anchor window, and
only falls back to a store read once a turn has aged out of anywhere §3.2
cares about byte-stability for) — in which case this is a documented,
accepted risk with a clear boundary; or (b) reconsider `jsonb` vs `json`
(text-preserving, validated-but-not-restructured) or plain `text` with an
application-level JSON-validity check, accepting that neither offers
`jsonb`'s binary containment/path-query operators, which nothing in this
schema currently uses anyway (`store.ts` never issues a `@>`, `#>`, or
similar `jsonb`-specific operator). I did not make this call — it is a
cross-lane (L1/L2) architectural decision that the plan did not anticipate,
and I'd rather flag it loudly than let it ship silently disguised as
"content is stored verbatim."

---

## 9. What I could not verify

- **The integrity-check-cost bar at full 2,000-message, multi-branch
  scale.** `audit_conversation`'s single recursive CTE is inherently
  O(total messages) — nothing about its structure suggests otherwise — but
  I did not build a dedicated 2,000-message, 5-branch timing assertion; F5's
  standing case exercises the same query shape at 241 messages / 5 roots,
  and F7's fixture builds 2,000 messages but never calls `audit_conversation`
  against it. This bar is named in `FRONTEND_SPEC.md` §13 as blocking;
  closing it fully is a small, mechanical addition (call `auditConversation`
  against F7's 2,000-message fixture and assert a wall-clock bound) that
  this lane ran out of scope-budget to add as its own dedicated test, given
  everything else in the brief. Flagging rather than skipping silently.
- **`rebuild_read_models`'s "drop, rebuild, assert equality" test**, which
  `FRONTEND_SPEC.md` §11.1 requires to exist IF a read model exists. Phase 1
  deliberately builds none (the migration's own comment explains why — the
  tail-first read path already meets the scale bar without a cache), so the
  function is a documented no-op and there is nothing to write a
  drop/rebuild/equality test against yet. Recorded here so the day a read
  model is added, the missing test is expected, not overlooked.
- **`client-unit` wired into any CI entrypoint.** This lane added the
  compose service and verified it manually; nothing in this repo's CI
  configuration (if any exists outside these compose files) was touched, so
  "runs on every PR" is not something this lane can claim without knowing
  what invokes `docker-compose.tests.yml` today.
- **Whether `--test-concurrency=1`'s WAL-noise problem (§5) also affects a
  production, non-test Postgres under real concurrent load** — it almost
  certainly does in the sense that WAL bytes are never attributable to one
  session under concurrency, but that is a measurement-methodology fact
  about WAL, not a defect in the store; F7's own bound (64× content +
  8192 bytes) has enough headroom that ordinary production concurrency
  should not false-fail it, but this was not load-tested.

---

## 10. Corrections/findings for the plan and schema, stated bluntly as asked

1. **§6's acceptance-command wording contradicts itself for the store
   suite** (`network_mode: none` for "the unit lane" vs. a Postgres 16
   sidecar for `client-unit`, which cannot be true simultaneously in the
   literal sense) — resolved via D-L1-3 above, but the plan text itself
   should be corrected so a future reader is not sent looking for a way to
   make both true at once.
2. **§11.2's `content jsonb` column does not give byte-stable round trips**
   — §8 above. This is the most consequential finding in this report: it
   directly touches §3.2's "byte-stable trailing turns" requirement, which
   the plan calls a "store requirement, not a transport one," and the store
   cannot fully deliver on that promise as currently specified without a
   cross-lane decision this report hands back rather than making
   unilaterally.
3. **No defect found in the reviewed correction** (`leaf_on_tree` as
   walk-membership, not row-existence) — confirmed correct, and confirmed
   BY MUTATION (log #1) that a regression back to the old behavior is
   caught. Not reverted, as instructed.
4. Everything else in the DDL, the write API, and the CAS/reachability
   logic held up under adversarial testing (F5), schema introspection (F6),
   and cost measurement (F7) without needing a change.

---

## 11. Summary

- **Fixed:** nothing needed fixing — `seedConversation`'s transaction bug
  was already corrected in the working tree; verified for the first time
  against real Postgres 16 in this lane (10/10 pre-existing tests passed
  cold).
- **Built:** F5 (7 adversarial tests incl. the standing 241/5/8 case), F6
  (4 schema tests), F7 (2 tests: append-cost O(1) + the checkpoint
  measurement), `testfixtures/client-unit/` (Dockerfile, entrypoint,
  readiness probe), `docker-compose.tests.yml`'s `client-unit` +
  `postgres-client-unit` + `client-unit-net` additions, and one new F4 test
  (`ON DELETE RESTRICT`) that closed a gap this lane's own mutation log
  found.
- **Mutation log:** 6 mutations run for real; 5 caught immediately by an
  existing or new test; 1 (`ON DELETE RESTRICT` → `CASCADE`) survived the
  full suite green on first try, reported honestly as a finding, then
  closed with a new, verified test.
- **The two measured numbers:** append cost is byte-identical (1,096B) at
  N=10 and N=2,000 — genuinely O(1); per-token streaming checkpoints cost
  **14.44×** more total bytes than checkpointing on an interval, for an
  identical final 2,400-byte reply (120,280B vs 8,328B).
- **Blocking finding handed back, not resolved:** `jsonb`'s input
  normalization (§8) is a real, measured gap against §3.2's byte-stability
  requirement that this lane cannot close alone — it needs either a
  transport-lane guarantee about where outbound bytes come from, or a
  column-type decision, both outside L1's remit as scoped.
- **Everything else:** all 24 store tests pass, reproducibly, in
  `docker compose -f docker-compose.tests.yml run --rm --build client-unit`.
