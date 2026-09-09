# Zion's Light AI — the replacement front end

**Plan for the implementing thread.** Precedence: `FRONTEND_HANDOFF.md`
(2026-09-09) → §2 Decisions below → `FRONTEND_SPEC.md` → `V4_FEATURES.md` §2.2.

> **Branch base — decided and verified.** `feat/frontend` is cut from
> **`fix/v3.1.4` (`c85b385`)**, not from `master` as handoff §7 suggests. The two
> have diverged: master is 4 commits ahead of the merge base and `fix/v3.1.4` is
> 37 ahead. Master's four are **all merge commits** (PRs #37, #39, #40, #41) whose
> content came from this same lineage, and `git diff --diff-filter=D master
> fix/v3.1.4` is empty — **nothing is lost.** What `fix/v3.1.4` adds and master
> lacks is load-bearing here: `FRONTEND_HANDOFF.md`, `COMMANDS.md`, the Postgres
> sidecar, `tests/adversarial/`, and the three `docker-compose.*.yml` harnesses.
> Re-check this before merging back; master will have moved.
>
> **`V4_FEATURES.md` §7's discipline applies to this plan too:** re-verify any
> finding against HEAD before planning a day of work around it. Several claims in
> the source documents were already stale — §7.1 lists them — and two had been
> carried into an earlier draft of this plan before review caught them.

---

## 1. Context — why this is being built

OpenWebUI is a generic chat client, and every V3 incident it touched, it
amplified. Two matter:

- **2026-08-24.** A failed request created the new turn as a *new root* and moved
  `currentId` to it; 208 messages went invisible. Separately and worse, the
  client then walked `parentId` back from that pointer and sent **7 messages of a
  241-message conversation** — `msgs=3 → 5 → 7` across three turns on a stable
  `conv_id`, proving a fixed head rather than a sliding window. No error. No
  signal. For weeks.
- **2026-08-28.** The client sent a complete 65-message conversation; the
  compactor, budgeting against a token count reading 23–51% low, shed 60 turns,
  handed the model **4 messages**, and returned HTTP 200 with a fluent reply.
  Every layer logged success. *No number anywhere in the product was wrong — the
  number that would have been wrong was never computed.*

Both times the user's report was not "an error occurred." It was that the
assistant had forgotten who she was.

So this is not "a nicer chat UI." It is a client whose send set is a **recorded
intent, verified before sending and disclosed after**, and whose history cannot
be structurally damaged by a failed request because the *store rejects* the
damage rather than trusting the code not to cause it.

Three live constraints frame the work:

1. The conversation in daily use is **1,718 messages / 27.71 MB, growing ~2 MB
   per day.** OpenWebUI ships that blob to the browser on every open while the
   disk reads it in 0.45 s on a 473 MB/s volume. The volume was never the
   problem and must not be blamed for it.
2. **The compactor owns memory.** The client must never be the thing that
   starves it — §3.2.
3. **MooseFS has corrupted a write-hot SQLite store three times** (2026-08-31
   loud, 2026-09-07 silent, plus the original). `OPERATIONS.md:217`: never put a
   write-hot store directly on `/data`.

---

## 2. Decisions

### 2.1 Settled by the owner, 2026-09-08 — do not reopen

| # | Decision | Consequence |
|---|---|---|
| **D1** | **Co-locate now, split later.** The client's server runs as `[program:client]` in the existing pod container, reaching the compactor on `127.0.0.1:8080`. | Passes `_require_localhost` (`main.py:4488-4502`) unchanged, so handoff blocker **B1 does not apply to this build**. §7 memory surfaces are in scope. PR #30 + CORS become a later exposure task. Write the server transport-agnostic (`COMPACTOR_URL`; always send `Authorization: Bearer` when a key is configured) so the split out is a deploy change. **Never** set `COMPACTOR_ADMIN_BIND=0.0.0.0` — it exposes **24** admin routes unauthenticated. |
| **D2** | **Bounded window, sized to feed the summarizer.** `window_intent` recorded per request and verified pre-send; **N = 60 non-system messages**, behind a runtime switch with a full-history setting. | §3.2. N is the number production already derived under the same constraint, and `test_summarizer.py:1333` exercises it. |
| **D3** | **Fork at cutover; the transcript importer is a later, separate deliverable.** | Memory continuity is *nearly* one API call — see F-1 in §3.5: fork does **not** carry persona or the archive sidecar. OpenWebUI is frozen read-only as the scrollback archive. |
| **D4** | **Refuse on a fidelity mismatch, with an override the daily user can reach.** | §3.2's override rules are load-bearing: the override must **not** recompute its way to compliance. |

### 2.2 The rest of the handoff's escalate list — also settled, 2026-09-08

`FRONTEND_HANDOFF.md:118-121` escalates **Q8, Q9, Q10, Q14**. Q9 is D4 and Q14 is
D3. An earlier draft of this plan wrongly filed Q8 and Q10 as delegated; they
were put to the owner and answered.

| # | Decision | Consequence |
|---|---|---|
| **D5 (Q10)** | **One synthetic root; the unique index is committed.** | `message_one_root_per_conv` ships in F3's DDL. This is what makes the 2026-08-24 failure *structurally impossible* rather than merely tested against — a constraint that cannot be bypassed, not code that can. Safe because of D3: import is the only legitimate multi-root source and it is out of the phase-1 store. If the importer is funded later it grafts under an `import_boundary` node (§16.2) rather than relaxing the index. |
| **D6 (Q8)** | **Quarantine and report; never repair automatically.** | F14. State what was found, list candidate chains with lengths and dates, move nothing. `FRONTEND_SPEC.md:316-322` and `V4_FEATURES.md:577-583` reach this independently: a silent pointer move is the mechanism of the incident, and §4.1 was explicitly revised to remove the wording that authorized one. Accepted cost: she sometimes faces a branch-selection dialog she did not ask for. |
| **D7 (U-1)** | **Facts get stable ids.** | Unblocks F26. Identity is a uuid, not the text, so editing one of two identical facts is defined and §7's "every fact traceable to the turn that produced it" becomes mechanical. Cost, to be carried by the compactor lane: it touches the write path, dedup's merge, the archive sidecar, the export bundle format — **which needs a version bump from `v2.1`, and `import_conversation` enforces strict equality on that field** — and the importer. |

### 2.3 Delegated to me by handoff §5, taken and recorded

- **Stack: SvelteKit**, with **Doulos design *tokens* copied into the repo, not
  the packages** (`V4_FEATURES` §3.5 — resolving private packages from an
  external registry breaks the standing offline requirement). Meets all five of
  §14's normative requirements; `+server.ts` is the secret-holding proxy §10 needs.
- **Q2/Q12 window size:** N = 60, recorded per request, never inferred.
- **Q3 branch UX:** sibling switcher (`‹ 2 of 3 ›`) plus a branch list in the
  quarantine dialog. No tree visualiser. **See U-4 — branch switching perturbs
  the compactor's position anchor and needs an explicit decision.**
- **Q4 offline:** local transcripts readable when the backend is unreachable;
  composer disabled behind an `offline` notice.
- **Q7 design-system fit:** tokens only. The offline requirement answers it.
- **Q11 receipt depth:** one collapsible line per assistant turn, pull not push.
  *Caveat:* `V4_FEATURES.md:790-794` says Q11 cannot be settled without **Q6** —
  which of the two users is primary when they conflict. The receipt and
  per-message cost are operator surfaces; the facts panel is hers. Taken as
  "visible by default, collapsed"; **worth a one-sentence confirmation**, not a
  blocker.
- **Q13 tombstones:** subtree-wide per §11.1. In neither handoff bucket; recorded
  here so it is visible.

---

## 3. The hard parts

### 3.1 The store must reject the damage, not avoid causing it

`V4_FEATURES` C2 rates this ~8 focused days and calls it the easiest to
under-build. **The failure mode is shipping the schema without the enforcement** —
a store that permits a whole-chain write "just for the importer" reintroduces the
entire defect class.

**Engine: the existing pod-local Postgres sidecar**, already running on
`fix/v3.1.4` — `PGDATA=/var/lib/postgresql/data` on the pod's local disk, unix
socket only, `pg_dump` archives to `/data/openwebui/pg`, restore-at-boot in
`entrypoint.sh`. This *is* handoff §4's "server-side, on local disk, archiving to
`/data`, converging on the Postgres sidecar" — the convergence already happened.

**Placement: a `client` schema inside the existing `openwebui` database — not a
new database.** Verified: `pgarchive.py:95` is
`PGDATABASE = os.environ.get("POSTGRES_DB", "openwebui")`, and every dump
(`:487-491`) and archive filename (`:349`) is scoped to that **one** database.

**This is a durability decision, not a tidiness one.** `pgarchive.py:125-127`
states the overlay `PGDATA` lives on is *"20 GB total, **EPHEMERAL**, shared with
/var/lib/openwebui and every container layer."* A `zlclient` database outside
`pg_dump -d openwebui` would exist **only** on that overlay and would vanish on
pod recreation — the OpenWebUI-blob failure inverted, and precisely what §11's
hard constraint exists to prevent.

Two consequences to record rather than discover:

1. **The shrink guard becomes a mixed signal.** `_count_rows_in_dump`
   (`pgarchive.py:401-416`) counts every `COPY` row in the dump and
   `SHRINK_REFUSE_BELOW = 0.5` compares against the previous archive. Once the
   client's `message` table rides along and grows ~2 MB/day, **client growth
   masks a real OpenWebUI shrink.** And when OpenWebUI is retired and its tables
   dropped, the guard refuses every archive until `PGARCHIVE_ALLOW_SHRINK=1` is
   set once, deliberately. Both belong in the runbook.
2. **Add a Tier-2 assertion** that the client's tables survive
   `restore_if_needed()` at boot. Nothing else proves the archive carries them.

The database name becomes a misnomer once OpenWebUI is gone. Accepted, recorded,
cheaper than a parallel archive path.

#### The schema — and what §11.2 does *not* give you

§11.2's DDL (`FRONTEND_SPEC.md:635-663`) defines the `message` table and an
`ALTER` on `conversation`. **It never defines `conversation` at all.** "Verbatim"
is not an instruction that can be followed. F3 owns the full DDL:

- `message` exactly as §11.2 writes it — every constraint, no exceptions.
- `conversation`: `id`, `current_leaf_id`, **`rev`** (§11.3's CAS depends on it
  and §11.2 never declares it), `user_id`, title, `created_at`, `updated_at`.
- `user_id` on **both** tables **in Phase 1** — §10 says "from day one so
  multi-user is a feature addition and not a migration," and F3 is six days
  before F16.
- The ordering index the tail-first read path needs.

The payoff is literal — *every mechanical step of 2026-08-24 is rejected by a
constraint*, with no application code: `message_one_root_per_conv` (the failed
turn could not have become a second root); the composite FK
`(conv_id, parent_id) → (conv_id, id)` (a message cannot be parented into another
conversation, nor exist without a live parent); `conversation_leaf_fk` (the
pointer cannot name a foreign message); a `BEFORE UPDATE` trigger raising on any
change to `id`/`conv_id`/`parent_id`; `message_not_self_parent`.

**The synthetic root is load-bearing.** `create_conversation` writes one
`role='system'` root holding the system/persona message; the first user turn is
its child. That is what makes `roots == 1` enforceable without losing legitimate
behaviour — editing the first user message produces a sibling under the synthetic
root, not a parentless node.

#### The write API, and the read API an earlier draft omitted

Writes, and nothing else: `create_conversation`, `append_message`,
`update_message_state`, `append_stream_delta`, `select_leaf`. No operation
accepts a chain. There is no `PUT /chats/{id}`. Ids are bare UUIDv7, **never
built by string concatenation anywhere** — a grep for message-id-forming
concatenation is a review-blocking finding.

**Two things must be specified on day one, because three lanes consume them:**

- **The read/pagination contract.** §13 requires the client render the tail first
  and fetch earlier turns on demand, and *"never need the whole conversation in
  memory to show the newest turn"* (`FRONTEND_SPEC.md:838-839`). That is keyset
  pagination over the active path with a cursor stable across branch switches.
- **`append_stream_delta` granularity.** Per token, per SSE chunk, or once at
  stream end? Per-chunk persistence into a `jsonb content` column rewrites the
  whole row each time under MVCC — **O(reply²) per turn**. F7 measures O(1) *in
  conversation length* and will pass regardless, so this cost is invisible to the
  plan's own bar. Decide it explicitly and measure it separately.

**Leaf movement, exactly two paths, both validating inside the transaction.**
Append is a compare-and-swap conditioned on the new message's parent being the
current leaf (`… WHERE rev = $expected AND current_leaf_id = $parent`, asserting
rowcount 1 or rolling back). Explicit branch selection runs the reachability
predicate inside the transaction. Because the pointer can only advance to a child
of where it stood, **an append can never orphan history.** Optimistic concurrency
on `rev`: a stale second tab loses its own write, loudly, and never overwrites.

#### The integrity check — all five properties, not four

Spec §4.1 names **five**. Four are store properties and one is not:

| # | Property | Where |
|---|---|---|
| 1 | `roots == 1` | store audit |
| 2 | `current_leaf` reachable from the root | store audit |
| 3 | **the chain from `current_leaf` contains every message the thread renders** | **client-side; cannot be inherited from the DDL** |
| 4 | every `parent_id` resolves in the same key space as the `id` it names | store audit |
| 5 | no message unreachable from the root | store audit |

`reachable_n == total` is property **5**, not property 3 — with siblings,
"reachable from root" includes every branch and says nothing about what the
thread renders. §12's `chain_corrupt` trigger names property 3 explicitly: *"…or
the rendered thread not contained in the chain from the current leaf"*
(`FRONTEND_SPEC.md:746`). It is the property that would have caught 2026-08-24
from the render side. **Add a render-set containment check on the load path and
in the pre-send gate, distinct from the store audit, and name it in F5.**

`audit_conversation(conv_id)` — one function, one recursive query, shared by the
load path, the pre-send gate, the tests, the importer and the CLI. It must report
`chain_from_current` and `deepest` as well, because §13's diagnosability bar is
the **five-tuple** `total / current_leaf / chain_from_current / deepest / roots`,
and Phase 0.4 depends on it.

**The forbidden checks**, named so no adjacent property can be substituted:
`max_depth(tree) ≥ threshold`, `count(messages) == expected`,
`chain_from_current ≥ threshold`, or anything that does not name the current
leaf. On 2026-08-24 a depth check reported the conversation healthy at
`deepest=208` while `chain_from_current=8`. The number was true and irrelevant.
**Standing test case: 241 messages, 5 roots, current leaf at depth 8 — the check
must fail.**

**Read models** are permitted under three rules and no others: computed by a pure
function of the base tables in the same transaction as the write that invalidates
them; no write, validation or integrity check reads from them; and
`rebuild_read_models(conv_id)` exists with a test that drops, rebuilds and asserts
equality. **Plus one this plan adds:** a materialized active path is never
invalidated by an out-of-band `psql` write, so `rebuild_read_models` must run
**before first render** — otherwise §11.5's repairability guarantee is false.

### 3.2 The window is a number with a derivation, not a preference

D2 sets **N = 60 non-system messages.** Both neighbours are worse, and the
derivation is in the repo, production-measured, at
`pipelines/conversation_id_header.py:108-133`:

| N | Why not |
|---|---|
| 100 | The inline summarizer allows `COMPACTOR_MAX_SUMMARY_CALLS` = 4 batches. 96 turns of ~1000 tokens is 4 exactly — and 7 under the **pessimistic 2.0× fallback that fires whenever `/tokenize` refuses**, which on 2026-09-01 was every request. A 100-turn cap re-latches inline summarization precisely when things are already going wrong. |
| 25 | One batch (2 pessimistic), but only 1.2× margin over `L1_CHUNK_SIZE=20`. One failed rollup puts you 40 turns behind with 25 visible, and the oldest 20 are gone for good. |
| **60** | 2 batches, 4 under the pessimistic fallback, **3× the L1 chunk** — about two failed rollups of recovery room. Measured: ~1.13 M tokens → ~60 k. |

"Turns" means **non-system messages**: `KEEP_RECENT_TURNS` slices
`non_system[:-4]` at `main.py:1620-1623`.

#### Handoff blocker B2 is substantially landed — do not plan around it

Handoff §3 B2 says the compactor's only notion of position is the client's array
length, that every exchange overwrites the same ChromaDB document, and that *"the
memory architecture is inert against the client this spec describes."* **Against
`fix/v3.1.4` both named effects already hold:**

- **Rollups fire under a cap.** `turns_seen` (`summarizer.py:46`),
  `_recorded_position` (`:813-829`), `_observed_position` (`:866-1086`) with
  invariants I1–I6 written against exactly this failure, `tail_fp` alignment
  (`:642-674`), and `window_offset`-aware chunk reads in `_do_l1_rollup`
  (`:1588-1607`). The healthy-path log line says it outright: *"the client is
  sending a bounded window ({n} turns) while the conversation is at turn
  {position}; rollups are driven by the compactor's own counter from here on, and
  chunk text is read at an offset of {position - n}"* (`:1077-1082`). Built for
  the `max_turns` valve, tested at caps 100 and 60 (`test_summarizer.py:1333-1338`).
- **Episodic rows do not overwrite.** `_next_turn_index` (`retrieval.py:293-327`)
  allocates `max(stored_max + step, seed)` — *"a deletion, an edit, a branch
  switch or a bounded client window all shrink it… the sequence only ever moves
  forward."* That is §15's "guard the destructive write" ask, **already
  delivered.**

`_needs_l1_rollup`'s docstring (`summarizer.py:508-517`) is the clincher:

> `current_turn_count` is the conversation's POSITION (`_observed_position`), not
> `len(messages)`. Handed the client's array length instead, this gate latches
> shut forever the moment the client starts sending a bounded window.

**What remains open is one line**, and it is different from the original ask:
`turn_index = len(messages) + 1` (`main.py:4895`) drives `recent_cutoff`
(`main.py:5142`) and stamps `added_turn` on every extracted fact. Under N=60 the
cutoff is out of frame against a store whose maximum is in the thousands;
`_cutoff_is_out_of_frame` (`retrieval.py:330-369`) detects that and degrades
**over-inclusive rather than silent**, its docstring naming §15's `turn_seq` as
the real fix and adding *"That is not this module's to make."*

**Why this matters more than a footnote.** Handoff §3 tells the implementer
*"expect memory features to look dead until B2 ships. That is not your bug."*
That excuse is gone. **Phase 3 gets a memory acceptance test** — L1 chunk count
advances across ≥20 client turns at N=60 — because if memory looks dead now, it
*is* the client's bug.

#### The hard client obligation this creates

`turns_seen` is anchored by `tail_fp` — sha256 fingerprints of the last
`_ANCHOR_TURNS = 4` turns, whitespace-normalized, 16 hex (`summarizer.py:653`,
`_turn_fingerprints` at `:705-730`). Prefix matching makes a regeneration cost
zero advance. When the anchor cannot be found, `_ASSUMED_NEW_TURNS = 2` and the
position drifts.

**So the trailing turns must be stable across requests — but "stable" means
something narrower and more useful than "byte-identical," and the difference
matters.** An earlier draft of this section said byte-stable and prescribed a
test comparing outbound JSON bytes. That test is both too strict and wrong. Read
`_turn_fingerprints` (`summarizer.py:707-730`): the anchor hashes

```
sha256( role + "\x00" + " ".join(_message_text(m).split()) )[:16]
```

— the **extracted text of the turn, whitespace-collapsed**, not the JSON
envelope. Its own docstring says why: the anchor is compared across two separate
HTTP requests, "and a re-flowed trailing newline must not read as a different
turn."

What follows:

- **JSON key order, inter-token whitespace and escape representation are
  irrelevant.** So `content jsonb` is safe, despite Postgres `jsonb` genuinely
  not preserving input bytes — it normalizes key order, drops duplicate keys and
  re-renders `\uXXXX` escapes. None of that changes a string *value*, and values
  are all the fingerprint reads. The L1 lane verified the normalization
  empirically and correctly escalated it; this is the resolution.
- **What must be stable is the extracted text and the role.** Markdown
  normalization, smart-quote substitution, entity encoding, or trimming and
  re-wrapping that changes characters rather than only whitespace runs — those
  break the anchor, silently, and the only symptom is a summary hierarchy
  drifting behind.
- **Never re-render content for the wire from the display form.** Store the wire
  form; send that. This is a store requirement, not a transport one.
- **One real `jsonb` limitation to know:** Postgres cannot store `\u0000` inside
  a jsonb string. A message containing a literal NUL fails to insert rather than
  round-tripping wrongly — loud, not silent, which is the right failure, but it
  needs a typed notice rather than an unhandled driver error.
- **The test, corrected:** reproduce `_turn_fingerprints` client-side and assert
  the trailing-4 fingerprints are identical across two sends of the same
  conversation. That tests the invariant the server actually uses, and it keeps
  passing when a JSON serializer legitimately reorders keys. **An image-only turn
  is fingerprinted through `_image_only_marker`, not as the empty string its
  request shape reduces to** — mirror that, or every image turn reads as a new turn.

#### D4's override must not recompute its way to compliance

If the send set did not match the intent, the *recomputed* intent is by
definition the thing the check just failed on. Sending on it reproduces the
2026-08-24 shape — 7 of 241, "correct" against its own recomputed number. So:

- The override records **both** numbers.
- It **must not overwrite `window_intent`.**
- The receipt shows the delta against the **original** intent.
- The turn is marked as sent under override.

"Recompute and resend" is the defect wearing a consent dialog.

### 3.3 The receipt reports what the server ADMITTED, not what the client sent

The lesson of 2026-08-28: the client sent 65, the compactor admitted 4, and a
receipt reading "65 sent" would have been *accurate and worthless*.

The client half ships now. The server half is the **received-context echo**, a
compactor-lane ask worth handing back fully specified, because `V4_FEATURES` C5
verified the data is already computed and discarded (`_enforce_hard_budget` fills
a `report` dict with `{limit, measured, fits, counted_by, …}`):

- Response **headers** on `/v1/chat/completions`, both write sites —
  `StreamingResponse(headers=…)`, **not** an SSE preamble event, which
  OpenAI-shaped parsers choke on. Two write sites is `V4_ROADMAP` §4 constraint 3
  and this project's most-repeated defect: one shared function, not the same rule
  twice.
- Fields: resolved `conv_id`, resolution `source`, **messages received**,
  **messages admitted**, exact prompt tokens, headroom, shed/trim counts kept
  **separate**.
- **Honesty requirement:** on the prescreen path `measured` is `None` and
  `counted_by` reads `"the char/4 prescreen (nothing was measured)"`
  (`main.py:3281`). The echo emits *not measured* there, never a number.
- **Per-message token cost: not on the hot path.** §15 asks for exact per-message
  counts from `/tokenize`; `V4_FEATURES` §3.3 shows that is 65 localhost round
  trips before the user sees a token. Measure the *assistant reply* exactly on
  the async tail where latency is free, label the rest as scaled estimates, and
  **mark estimates visibly in the UI, not only in a tooltip.**

Until the echo lands the receipt renders admitted as **"not reported by the
server"** — a first-class unknown per §2 obligation 2.

### 3.4 The transport, as the code actually behaves

Spec §5 is titled "verified against the code" and is wrong in several places
against `fix/v3.1.4`.

**There is no authentication anywhere in this tree** — not on `/v1/*`, not
anywhere. `COMPACTOR_API_KEY` and `hmac.compare_digest` exist only on the
unmerged `v4/foundation-compactor-auth`. There is no CORS middleware and **no
custom response header of any kind**. The client's own server is therefore the
*only* auth boundary in the system, which is another reason F16 is core.

**Requests.** No Pydantic models; every handler hand-validates a raw `Request`.
The compactor inspects only `messages`, `stream`, `model`, `max_tokens`,
`metadata.{chat_id,conversation_id}`, `continue_final_message` and
`add_generation_prompt` — everything else is forwarded to vLLM untouched.
`max_tokens` above `MAX_MODEL_LEN // 2` is clamped *by rewriting the body*.

`conv_id` resolves in three tiers (`memory.py:87-130`): `X-Conversation-Id` →
`metadata.chat_id`/`conversation_id` → the
`sha256(system ||| first_user[:512])[:16]` fingerprint. The sanitizer strips to
`[A-Za-z0-9_-]` and truncates to 64, so a UUIDv4 passes intact. A header that
sanitizes to empty falls through to the hash **with only a log line**.

**Streaming, and the two things that will bite.** On success the compactor is a
raw byte relay (`aiter_raw()`), so the wire is byte-for-byte vLLM's own chunks.
But **HTTP 200 is committed before vLLM is contacted** (`main.py:4715-4719`), and
**`read=None`** on the httpx timeout (`main.py:5383-5385`) means there is *no*
generation timeout — a hung vLLM hangs the stream indefinitely. **The client owns
that timeout.**

**Three compactor-authored stream shapes replace the relay, and they are not
symmetrical:**

| Shape | id prefix | Terminates | Machine-readable? |
|---|---|---|---|
| Slash command | `chatcmpl-cmd-` | `finish_reason: "stop"` | **No** — id prefix only |
| vLLM 4xx rejection | `chatcmpl-rejected-` | `finish_reason: "error"` **plus a top-level `error` object** (`context_length_exceeded` / `backend_rejected`) | **Yes** |
| vLLM 5xx / unreachable | `chatcmpl-unavail-` | `finish_reason: "stop"`, **no `error` object** | **No** — id prefix only |

The asymmetry is deliberate on the rejection side and a **live gap** on the
unavailable side: an outage is machine-indistinguishable from a real reply.
**Match on the id prefix**, never the emoji, and hand back the ask.

**Two different error envelopes.** `/v1/*` failures are OpenAI-shaped
`{"error": {…}}`; the app-wide `UnsafeConvId` handler returns `{"detail": "…"}`
at 400; FastAPI query-coercion failures return the stock 422 `{"detail": [...]}`.
New 400 codes worth handling by name: `unparseable_body`, `body_not_an_object`,
`unpaired_surrogate`, `empty_messages`.

**Corrections to spec §5.3/§5.4 that cost real time if inherited:**

- A **fact has four keys**, not three: `{text, added_turn, last_used, pin}`
  (`facts.py:390-398`). Archived facts have five, adding `archived_at`. **None is
  an id** — see U-1.
- `DELETE /admin/conversations/{id}/facts` is **a five-layer wipe** — facts,
  episodic, summary state *and* persona (`main.py:5832`, `_clear_all_memory` at
  `:5876`) — not "forget facts (all/matching)". Its response carries
  `unreadable: [...]`; non-empty means the wipe was partial and **must not** be
  reported as clean. There is no single-fact delete, no add, no edit.
- Two destructive routes the spec omits: `cleanup-test-data` and `merge-into`.
  Both **default to a dry run**, and `merge-into` takes the flag in body *or*
  query where only `"false"`/`"0"`/`"no"` commit — **a typo stays a dry run**,
  returning HTTP 200 with plausible counts. Read the counts, never the status.
  `POST .../compact` is the asymmetric one: it defaults **live**.
- `POST .../fork` accepts `{"new_conv_id": …}` (`portability.py:1205`).
- The slash-command alias list is larger than §5.3's table — also `/tidy`,
  `/retire`, `/pin`, `/unpin` (`commands.py:118-153`).

**`/health/full` is richer than §5.2 says** and is the whole of F19:
`checks.{vllm,storage,sqlite_journal,tokenize}`, `stats.unreadable`,
`memory_writes`, `background_work`, `memory_tail`, and a `status_reasons` array
of **literal matchable prefixes**. `degraded` returns **HTTP 200** deliberately;
only "storage not writable" produces `down`/503. The `null`-vs-`0` distinction is
contractual — `null` means *could not tell*, `0` means *genuinely empty*. Render
`null` as unknown, never as zero.

**Voice is not proxied.** STT (`:9000`) and TTS (`:9001`) are separate processes
the client's server calls directly. TTS **ignores `voice` and `model` in the
body** and only ever speaks `TTS_VOICE`, though `GET /v1/audio/voices` lists
every voice on disk; and if `ffmpeg` is missing it **silently returns WAV with
`Content-Type: audio/wav`** whatever was asked for. Read the response content
type.

**Images: the compactor validates nothing** — no sniffing, no MIME check, no size
limit. So §8's content-sniffing is **entirely the client's job**, and it is the
only thing between a `.paint` file and the model. `COMPACTOR_MAX_RETAINED_IMAGES=0`
— the production setting — **strips the image the user just uploaded**, and
`main.py:3033-3039` notes there is no affordance saying so.
`backend_is_multimodal()` exists (`main.py:327`) but is not exposed over HTTP.

### 3.5 Six points an implementer would otherwise guess wrong

- **F-1 — a fork does not carry persona or the archive sidecar.**
  `fork_conversation` is `export → import` (`portability.py:1208-1210`), and
  `export_conversation` returns exactly three payloads: facts, summary_state,
  episodic (`:88-95`). The module says so: *"a fork still loses the persona"*
  (`:165-166`), with D19 cited at `:145-147`. Two consequences: an **admin-set**
  persona is silently dropped at cutover (`auto_capture` re-takes it from the
  client's system message, which the synthetic root supplies — recoverable only
  if the text matches, which is an assertion, not an assumption); and F27's
  archive browse **reads empty** on the forked id, which is exactly the "she
  forgot me" empty state §10 forbids presenting unexplained. **Fix:** before
  forking, `GET .../persona` and `GET .../archive`; re-apply on the new id, or
  hand back an ask to widen `export_conversation`. F25 asserts both.
- **F-2 — a stopped reply IS memorized.** `decide_memory_tail`
  (`main.py:2487-2513`) is one policy for both write sites since v3.1.4: `holed`
  → skip; empty → skip; finished-and-not-truncated → store verbatim; **otherwise
  — she hit Stop, or vLLM hit the ceiling — trim to the last complete sentence
  and store the trimmed prefix** unless nothing survives, too little survives, or
  what survives is degenerate. `FRONTEND_SPEC.md:424` and `V4_FEATURES.md:254-256`
  both say the opposite and are stale. Either surface it on the turn ("the part
  it had written was remembered") or hand back an ask for a client-declared
  discard signal. **Do not list it as a free parity win.**
- **U-1 — `{ref}` has no referent. Settled by D7: facts get stable ids.**
  `V4_FEATURES.md:805-809` flagged this as open and blocking C11. The work lands
  in the compactor lane and touches the write path, dedup's merge, the archive
  sidecar, the export bundle format and the importer. **Two things not to miss:**
  the bundle's `version` is `"v2.1"` (`portability.py:56`) and
  `import_conversation` enforces **strict equality** on it, so adding a field to
  the fact shape is a format change that needs a version bump *and* a decision
  about whether old bundles still import; and `dedup` merges facts by text today,
  so it must learn which id survives a merge rather than minting a third.
- **U-2 — when the user row is written, relative to the pre-send gate.** §4.1
  says both *"appended only after the request is accepted"* and *"on failure the
  user turn is retained in place, marked `failed`."* Resolve the ordering: it
  determines whether a D4 refusal has already mutated the chain and moved the
  leaf. Guessing wrong means re-deriving F3's CAS semantics after F8 is written.
- **U-4 — branch switching perturbs the position anchor.** `tail_fp` assumes the
  client re-sends a growing **suffix**; `_ANCHOR_TURNS = 4` exists so the prefix
  walk survives *a regeneration*, which rewrites the newest turn and nothing else
  (`summarizer.py:651-654`). A branch switch or mid-chain edit changes the window
  non-suffixically, so no candidate aligns, `new = _ASSUMED_NEW_TURNS`, a
  `log_once` warning fires (`:1026-1037`), and `window_offset` drifts so chunk
  text is read against a different branch's content. F12 and D2 ship together and
  the interaction is real. **Either** declare that a branch switch re-sends the
  full window and accept the drift with the log line as detector, **or** hand back
  an ask for a client-declared position — cheap now that the client owns the chain.
- **U-5 — `/why` is "next turn", not "last turn".** `_handle_why`
  (`commands.py:716-721`): *"Show what would be injected on the **next** turn …
  we don't keep per-turn injection snapshots in V2.1."* Labelling that "What it
  was given" for a past turn presents a stale value as live — §2 obligation 2,
  verbatim. Until the injected-memory endpoint exists, F28 renders as unknown.

---

## 4. What gets built, in order

Effort in focused engineering days. At ~2 focused days/week, multiply by 3.5 for
calendar.

### Phase 0 — before any code (1 d, the highest-value day in the plan)

| | Task |
|---|---|
| 0.1 | **Pull the logs and confirm the guard stopped shedding.** `V4_FEATURES` §6 Q1: *"This blocks the most and costs the least."* Every judgement about which mechanism causes "forgetting who she is" — and whether the receipt's admitted field shows a live defect or a healed one on day one — rests on it. |
| 0.2 | **Find out where OpenWebUI's database actually is.** One of three: Postgres (`DATABASE_URL`), local SQLite at `WEBUI_LOCAL_DB` (default `/var/lib/openwebui/webui.db`), or the `/data/openwebui/webui.db` snapshot. `entrypoint.sh` chooses at boot and falls back; `webuidb.py:54-58` holds the paths; `supervisorctl status` on `webuidb-sync` tells you which. **The snapshot is not the live database.** |
| 0.3 | **Snapshot and sanitize the 241/5-root/leaf-at-depth-8 conversation** from a **stopped** OpenWebUI with every browser tab closed — §16.2: an open tab reverted a verified repair *twice* on 2026-08-24. This fixture is the regression corpus for §4.1 and §11 and **must not be discarded after migration.** |
| 0.4 | Run the five-tuple over **every** OpenWebUI conversation. Five roots accumulated in one chat; no reason to assume it is the only one. |
| 0.5 | Confirm the conv_id path is `source=body_metadata.chat_id`, not `hash`. Forking from the wrong id forks the wrong memory. |
| 0.6 | Read `/health/full` for `l1`/`l2` counts and `status_reasons`; confirm free space on the **ephemeral** 20 GB local overlay. |
| 0.7 | ~~Get Q8 and Q10 answered~~ — **done 2026-09-08, D5/D6.** F3's DDL commits the unique index. |

### Phase 1 — handoff §8's definition of done (18 d)

| # | Item | d | Dep | Notes |
|---|---|---|---|---|
| F1 | Scaffold: SvelteKit + Doulos tokens, **every asset vendored** | 1 | — | No Google Fonts, no CDN. A client that fetches a webfont at first paint does not boot offline. |
| F2 | `[program:client]`, port 3001, priority 21 | 0.5 | F1 | Beside `[program:openwebui]` (priority 20). Add 3001 to the RunPod exposed ports. **Give it `stopwaitsecs`** — of ten `[program:]` blocks only `postgres` has one (`supervisord.conf:112-134`), and supervisord's 10 s default races every graceful shutdown. |
| F3 | **The message store** — full DDL (incl. `conversation`, `rev`, `user_id`), delta-only writes, **the read/pagination contract**, CAS leaf, subtree tombstones, `audit_conversation()` reporting the five-tuple | 6 | F2 | §3.1. The long pole. `content` holds the exact wire form. |
| F4 | **Store rejection suite** — second root; unknown parent; cross-conversation parent; pointer move to an unreachable node; an update changing `parent_id` | 1 | F3 | Asserts the *constraints*, not the code. |
| F5 | **Adversarial chain suite** — multi-root, missing parent, deep-vs-current divergence, mixed key spaces, mid-chain failure, **render-set containment (property 3)**, and the standing 241/5/8 case | 1.5 | F3 | |
| F6 | **Schema test: no serialized chain anywhere** | 0.5 | F3 | §11.1 as a test, because as prose it did not stop the failure shipping. |
| F7 | **Append-cost O(1) in conversation length** | 0.5 | F3 | Handoff §4: *phase-1 deliverable, not polish*. Measure `append_stream_delta` cost separately — F7 will not catch O(reply²). |
| F8 | **Checked send-set + pre-send gate + `window_intent`** + D4's override | 3 | F3 | §3.2, including the override's both-numbers rule. |
| F9 | Streaming chat; client half of the receipt | 3 | F8 | Client owns the generation timeout. Classify the three stream shapes by id prefix. **Stop commits a trimmed prefix to memory** (F-2) — surface it. |
| F10 | **The typed-notice surface** — the component, the catalogue module, the speech/non-speech distinction, the error-chunk detector | 1 | F8 | **Moved from Phase 2, and this is not cosmetic.** F8 raises `context_truncated`, F9 must re-render compactor error chunks as notices rather than speech, and F5 exercises `chain_corrupt` — all §12 types. §2 obligation 3 is unconditional: any path that moves the leaf, drops a message or reconstructs a chain surfaces a typed notice, **never a log line only**. Without this, Phase 1 ships the exact failure the document is organized against, for the length of Phase 2. |
| F11 | 2,000-message generated fixture + the raised scale bars | 1 | F3 | Generated, never her real conversation. |

### Phase 2 — the client is usable (11 d)

| # | Item | d | Notes |
|---|---|---|---|
| F12 | Branch UX — siblings under a shared parent, validated selection, `‹ 2 of 3 ›` | 2 | Resolve U-4 first. Also: the client knows which sibling was **rejected**, which OpenWebUI never did — `V4_FEATURES` §6 Q11 records that the store preserves the rejected reply forever via the recency filter. Free fix or free handback; take one. |
| F13 | `context_trimmed` / `context_shed` kept **separate** on the F10 surface | 1 | Collapsing them makes 2026-08-28 read as routine housekeeping — exactly how it read in the logs. Depends on the server shed signal. |
| F14 | **Quarantine dialog** for `chain_corrupt` | 1 | "This conversation has 5 separate branches; the one you are viewing has 8 of 241 messages." Candidate chains with lengths and dates. Changes nothing automatically. Ever. |
| F15 | `conv_id_fallback` notice | 0.5 | The echo reporting `source=hash` while the client sent a header. A manual check at cutover does not catch the day the header stops arriving. |
| F16 | Session auth + server-side proxy | 1.5 | Production is `WEBUI_AUTH=true`; replacing OpenWebUI without auth is a **security regression**. Copy distinguishes *logged out* from *no data*. (`user_id` already landed in F3.) |
| F17 | Conversation list, rename, local search, deterministic titles | 1 | Title = first ~60 chars of the first user message, editable. |
| F18 | **No task traffic — at all** | 0.5 | One kind of call: a real user turn. OpenWebUI's task calls sent `messages: []` and accumulated **105 facts in a conversation that does not exist**. Highest value-per-day item in the plan. |
| F19 | Status surface from `/health/full` + `/admin/selftest` | 1 | Render `status_reasons` directly. `degraded` = HTTP 200. `null` = unknown, never zero. |
| F20 | **§13 presentation bars** — WCAG 2.1 AA, full keyboard, careful `aria-live` for streaming, visible focus, `prefers-reduced-motion`, light/dark/system before first paint, no hard-coded strings, installable PWA | 2 | Handoff §2 puts these in scope explicitly. Unscheduled in an earlier draft. |
| F21 | **§12 time rules 1 and 2** — store UTC always; render in the viewer's zone from `Intl.DateTimeFormat().resolvedOptions().timeZone`; send the **IANA name**, never an offset | 0.5 | Rule 3 and "the server must not substitute its own" are the handback; these two are the client's. |

### Phase 3 — cutover (3 d)

| # | Item | d | Notes |
|---|---|---|---|
| F22 | **Fork migration** — client mints UUIDv4, `POST .../fork` with `new_conv_id` | 1 | Carries facts, summary state, episodic. **Not persona, not the archive sidecar** (F-1) — read and re-apply both. |
| F23 | Ordering interlock, in the runbook and in code | 1 | Four parts, all from `conversation_id_header.py`: (a) **fork, never merge** — `merge-into` does not move summaries, and the reason it refuses is that the destination re-derives them *"from the client's full array"*, true only while the cap is off; under D2 the cap is permanent, so a merge strands L1/L2/L3 **irrecoverably**. (b) **The identity switch is not gated by the cap** (`:59-73`) — the client mints a fresh UUID and sends it on request one, and the compactor's fork detector *cannot* warn (it returns early unless the id came from the hash). (c) Order: verify header path → fork → *then* enable the cap; capping first seeds episodic `turn_index` near N and a later merge skips those rows — *"striped, silent, and re-running the merge will not repair it"* (`:153-158`). (d) **Read the counts, not the status**, on any commit (`:96-102`). |
| F24 | Parallel running; freeze OpenWebUI read-only | 0.5 | §16.1: never share a `conv_id` with OpenWebUI while it is still writing facts from a starved context. Confirm compactor-lane P7 has landed — it closes the phantom-write path for **any** client, including OpenWebUI during parallel running. |
| F25 | Cutover checklist | 0.5 | §6 parity; a real conversation end to end; the echo confirming `source=header`; `audit_conversation` PASS; **persona and archive present on the new id**; **the memory acceptance test** (L1 advances across ≥20 turns at N=60). |

**§16.4's criteria, and which are relaxed by whose authority.** Verified
transcript migration — relaxed by **D3, owner**. Voice and image end-to-end —
relaxed against the production text-only posture. **"Memory panel verified
against the admin API" is not relaxed**: D1's whole point is that the memory
surface is unblocked, so there is no longer a reason the panel sits after
cutover. Either pull F27 ahead of F25, or record the relaxation and who
authorized it. An earlier draft dropped it silently.

### Phase 4 — the memory surface (7 d)

| # | Item | d | Lane |
|---|---|---|---|
| F26 | Per-fact `POST`/`PATCH`/`DELETE`, every handler taking `conv_lock` around load-modify-write | 3 | **compactor lane.** Blocked on U-1. Locking is not optional: v3.1 F22 records an unlocked `/remember` racing the tail's locked write — *"the user watched the compactor confirm Remembered: … and the fact was gone by her next turn."* Delete archives, never unlinks. |
| F27 | Memory panel v1 — facts table with inline edit/delete/add, archive browse-and-restore, read-only summary stack and persona | 4 | client |
| F28 | "What it was given" | — | Folds into F27 **when the injected-memory endpoint exists** — not wired to `/why` (U-5). |

**Cut from v1 deliberately:** persona library, dedup trigger, memory-bundle
export/import, fork UI. Those endpoints exist and are operator tools reachable by
curl; the daily user is not the operator. *(This is `V4_FEATURES` C12's cut of
operator tools — **not** §11.5's transcript export, which is F29.)*

### Phase 5 / interleaved — §11.5 and the diagnosability bar (2.5 d)

| # | Item | d | Notes |
|---|---|---|---|
| F29 | **§11.5 transcript export / import** | 1.5 | Missing from an earlier draft entirely. §11.5 is a §11 requirement and handoff §8 item 1 is "the store from §11". It is what makes F14's *"no automatic repair, ever"* survivable: the out-of-app SQL repair **is** the repair path. Includes a test that an out-of-band `UPDATE conversation SET current_leaf_id = …` in `psql` **is honoured by the next send** — which is what forces `rebuild_read_models` to run before first render. |
| F30 | **`zl-chain audit --all`** | 0.5 | §11.4: the audit runs before first render, after any typed failure, **in CI after every test that mutates a chain**, and as a CLI that **exits non-zero on any FAIL**. §13's diagnosability bar is this command emitting the five-tuple by construction rather than requiring a hand-written probe. |
| F31 | **No-silent-state-mutation test** | 0.5 | §13, blocking: an automated test asserting no code path moves the current leaf or alters chain structure without emitting a typed notice. |

### Deferred, with reasons

- **Vision (§8).** Production runs `COMPACTOR_MAX_RETAINED_IMAGES=0`. Build the
  sniffing, content-parts, retention-indicator and modality paths and **ship with
  upload disabled behind one switch** — text-only is a runtime posture, not an
  architectural assumption. Whatever the switch's state, the user is told *why*.
- **Voice (§9).** Gated on `STT_ENABLED`/`TTS_ENABLED`. Never auto-send a
  transcript without confirmation — transcription is fallible.
- **The transcript importer (§16.2).** ~7 d, after cutover, against the frozen
  snapshot. Strip exactly the `chat_id + '-'` prefix and assert the remainder is
  a UUID — **splitting on `-` is forbidden**, UUIDs contain hyphens and a naive
  split produced the false "every message is orphaned" reading mid-incident.
  Preserve every root via `--split` or `--graft`, no default. Do not trust
  `currentId`. Import unreachable nodes rather than dropping them.

**Totals: 41.5 focused days to end of Phase 5; ~33 to a usable cutover.** At the
stated part-time rate, roughly seven and five months respectively. *(An earlier
draft said 33.5/26 on a Phase-1 subtotal that did not sum.)*

---

## 5. Delegation

Lanes have **disjoint file ownership** — two agents never hold the same file.
Each writes its report to `docs/lanes/<lane>.md` rather than returning prose, and
**no agent runs a git mutation**: I merge, and I run the mutation check at merge
level myself.

| Lane | Owns | Model | Can start |
|---|---|---|---|
| **L0 Scaffold** | `frontend/` root config; `src/app.html`, `src/routes/+layout*`, `src/lib/components/shell/**`, `src/lib/styles/**`, `src/lib/i18n/**`; the `supervisord.conf` stanza; the Dockerfile hunk; vendored fonts | Sonnet | day 1 |
| **L1 Store** | `frontend/src/lib/server/store/**`, migration SQL | Sonnet, **Opus gate** | after L0 |
| **L2 Transport** | `frontend/src/lib/server/compactor/**` — send-set, pre-send gate, SSE parse, echo read | Sonnet, **Opus gate** | after the interface freeze |
| **L3 Chat UI** | the chat surface only — `src/routes/(chat)/**` and `src/lib/components/chat/**`. **Not** `components/shell/**` or `+layout*`, which L0 owns | Sonnet | after the interface freeze |
| **L4 Notices & receipt** | `frontend/src/lib/components/system/**`, the catalogue module | Sonnet | after L2 types |
| **L5 Fixtures & harness** | `frontend/tests/**`, generators, the 241/5/8 corpus | Sonnet | day 1 (fixtures only) |
| **L6 Memory panel** | `frontend/src/lib/components/memory/**`, `frontend/src/lib/server/admin/**` | Sonnet | after F26 |
| **Gate** | Reviews L1 and L2 against §4.1 and §11 before merge; owns nothing | **Opus** | — |

**The interface freeze is a real deliverable, not an assertion.** An earlier draft
claimed three lanes could run concurrently off a day-one typed module. Two cannot
unless the freeze includes more than write signatures:

- **L5 cannot test constraints against a mock.** F4 and F6 assert that *the
  constraints* reject writes and that no serialized chain exists in the schema —
  tests of the migration SQL, not of a typed module. F7 measures bytes written by
  the real engine. What L5 genuinely has on day one is the fixture generator and
  the 241/5/8 corpus from Phase 0.
- **L3 needs the read contract**, which is what §13's tail-first bar depends on
  and the part most likely to churn once F3 is real.

So the frozen interface must include **the read/pagination contract and the
streaming-persistence granularity** (§3.1), or L3 builds against a mock and
re-integrates — a legitimate choice, but say so rather than discover it.

**L0 and L3 both touch `routes/` and `components/` — the split is by subtree, not
by directory.** An earlier draft of this table gave L3 all of both, which
contradicted F1's own brief (F1 builds the shell). L0 owns the application
*chrome* — the HTML document, the root layout, theming, the i18n seam, and the
empty rail/pane placeholders under `components/shell/`. L3 replaces the
placeholders' contents under `components/chat/`. Neither edits the other's
subtree.

**L1 and L2 get an Opus gate** because they are where the spec's argument lives,
and because the recurring failure here is a rule applied at one call site and
missed at its identical sibling — eighteen incidents and counting, one committed
*while fixing an instance of it*. The gate's standing instruction is to grep for
the sibling, every time.

---

## 6. How this is verified

Handoff §6's standards, which a new thread will not guess:

1. **A test only counts if it has been watched to fail.** Mutate, run, watch red,
   restore, watch green. Here a surviving mutation has twice been a real gap and
   once a real defect hiding behind a test that agreed with the code.
2. **Never restore from git after a mutation.** Keep the original in memory (read
   with `newline=""`), replace one anchor, restore in a `finally`. A
   `git checkout --` has already destroyed another agent's in-flight work here.
3. **Tests run on Linux, in Docker.** A Windows host run is a development
   convenience, not evidence. **Exit code 3 means SKIPPED** and is never folded
   into a pass.
4. **Stage explicit paths.** Never `git add -A` or `git add .`. Own branch, own
   worktree: `feat/frontend`.

**Tiers, matching the project's existing shape:**

- **Tier 1 (unit)** — store constraints, send-set construction, notice mapping,
  chain audit. Offline, `network_mode: none`, as `docker-compose.tests.yml` does.
- **Tier 2 (boot)** — extend `[program:selftest]`: the client's port answers, its
  schema is reachable and writable, `audit_conversation` PASSes on every
  conversation, and **the client's tables survived `restore_if_needed()`**.
- **Tier 3 (integration)** — black box via `docker-compose.integration.yml`,
  which already builds a vLLM stand-in and the real compactor on `8081`.

**Acceptance.** The harnesses define `unit-tests` (`docker-compose.tests.yml:30`)
and `vllm-fixture`/`compactor`/`integration-tests`. Add `client-unit` beside
`unit-tests` and `client` beside `compactor` — **new services, following the
existing ones' shape**. Note `unit-tests` uses `network_mode: none`, which the
store lane **cannot** copy - it needs a Postgres sidecar, and the two are
structurally incompatible. The equivalent isolation is a dedicated
`internal: true` network, which is what `client-unit` uses; that satisfies the
intent (no egress) without the impossible letter of it. The integration lane
keeps the read-only working-tree mount.

```bash
# 1. The store rejects every mechanical step of 2026-08-24
docker compose -f docker-compose.tests.yml run --rm --build client-unit

# 2. The standing case fails the check, as it must
#    241 messages, 5 roots, leaf at depth 8 -> audit_conversation() == FAIL
docker compose -f docker-compose.tests.yml run --rm client-unit --only chain_audit

# 3. Scale + append-cost bars against a GENERATED 2,000-message fixture
docker compose -f docker-compose.tests.yml run --rm client-unit --only scale

# 4. Live stack, real streaming turn, receipt populated
#    --force-recreate is not optional: `run` will not recreate an already-running
#    dependency, and that has served code 49 minutes older than the commit under test
docker compose -f docker-compose.integration.yml up -d --force-recreate compactor client
docker compose -f docker-compose.integration.yml run --no-deps integration-tests -k client

# 5. The chain CLI, which CI runs after every chain-mutating test
zl-chain audit --all      # exits non-zero on any FAIL
```

**Manual gates before cutover** — the ones a test cannot assert:

- Send a turn; the compactor log reads `source=header` and `msgs` matches the
  receipt's sent figure. On 2026-08-24 it read `source=hash msgs=7` and OpenWebUI
  had no way to know.
- Kill vLLM mid-stream. The turn is retained **in place**, marked `failed`, retry
  available: no new root, no pointer move, and a typed notice — not speech.
- Open the 2,000-message fixture: readable < 1 s, interactive < 3 s, and the whole
  conversation never in memory to show the newest turn.
- Rotate `SESSION_SECRET` deliberately: the UI says *logged out*, not *no data*.
- Repair the chain from `psql` while the app is running; the next send honours it.

**Bars that block release.** `V4_FEATURES` §3.4 is right that §13's *generic*
performance numbers were borrowed from a product with users — but handoff §4
raised the **scale** bars deliberately against production measurement, and those
are a different set. Keep them apart:

- **Blocking:** context fidelity (zero tolerance); no silent state mutation
  (F31); budget verification; diagnosability (F30); append cost O(1) (F7); the
  integrity check running within budget on a 2,000-message multi-branch
  conversation; and handoff §4's three raised scale bars — **60 fps scroll at
  2,000 messages**, **< 1 s readable / < 3 s interactive at 2,000 messages /
  30 MB**, and O(1) append. A checklist only affordable on a conversation smaller
  than the real one is a checklist that gets disabled on the conversation that
  matters.
- **Measured and reported, not gating:** §13's generic 60 fps, < 200 KB gzipped,
  < 2 s interactive.

---

## 7. Handed back to the compactor lane

Written down rather than worked around, per handoff §2. Nothing here blocks
Phase 1.

| Ask | Priority | Note |
|---|---|---|
| **Received-context echo** as response headers, both write sites | required | §3.3. The data is already computed and discarded; the compactor sets **no custom response header anywhere**, so this is new surface. |
| **Per-fact admin endpoints**, `conv_lock`-correct, **keyed on a stable fact id** | required for Phase 4 | Unblocked by D7. Today's `DELETE` wipes all five layers; there is no add, edit or single-delete route. Adding the id changes the export bundle shape, whose `version` (`"v2.1"`, `portability.py:56`) `import_conversation` compares by strict equality — bump it deliberately and decide what happens to bundles already on disk. `dedup` must learn which id survives a merge. |
| **`error` object on the `chatcmpl-unavail-` shape** | required | A backend outage is currently machine-indistinguishable from a real reply. One-line fix at the sibling of code that already does it right. |
| **Budget-shed signal** | required | `context_shed` (F13) cannot be honest without it. `dropped_layers` is a `logger.warning` at `main.py:5250`. |
| **Stamp `added_turn` / `recent_cutoff` from the server's own sequence**, not `len(messages) + 1` | required | `main.py:4895`, `:5142`. The remaining, much smaller half of B2. Under a bounded window `added_turn` is a near-constant, which breaks §7's "every fact traceable to the turn that produced it." |
| **Client-declared IANA timezone** | required | The container runs UTC with `TZ` unset. The zone belongs to the request, not the deployment. **Interlock:** `V4_FEATURES` P8 must land before or with the clock injection — a real clock beside stored facts asserting fake times gives the model two contradictory clocks. |
| **`/tidy` does not know the dashboard rule** | required before F22 | `_tidy_removal_rule` (`commands.py:910-923`) matches only `scaffolding` and `no-content`; `_tidy_flag_rule` (`:926-950`) has no dashboard rule. `ENERGY LEVEL: 88% → 92%` passes both. The rule lives in `facts._reject_reason` (write-path only) and in **`/retire`** (`commands.py:1495-1497`). `V4_FEATURES` P8 is **unbuilt** — and even once built it **flags**: a human acts on the flags, or `/retire` is used instead. |
| **Widen `export_conversation`** to carry persona and the archive sidecar | recommended | Would make D3's "one API call" true. Until then F22 does it client-side (F-1). |
| **Regeneration supersession** | recommended | The store preserves the **rejected** reply forever via the recency filter. The new client knows which sibling was rejected; OpenWebUI never did. Free fix. |
| **A modality route** exposing `backend_is_multimodal()` | recommended | Exists at `main.py:327`, unreachable over HTTP. Cheaper than inferring from a rejection. |
| **A signal when image retention drops an image** | recommended | `main.py:3033-3039` names the gap itself. |
| **Injected-memory endpoint** | recommended | Powers F28. Until it lands, F28 renders as unknown — **never wired to `/why`** (U-5). |
| **Task-traffic marker** | see note | F18 makes *this* client send none, but `V4_FEATURES` P7 closes the path server-side for **any** client, including OpenWebUI during F24's parallel running. Confirm P7's status rather than assuming. |
| **N is coupled to `L1_CHUNK_SIZE` / `KEEP_RECENT_TURNS` / `COMPACTOR_MAX_SUMMARY_CALLS`** | finding | One place with a test, not two configs. |
| **Destructive-write guard** | **verified landed** | `_next_turn_index` (`retrieval.py:293-327`). §15 lists it as required; it is done. |
| `pipelines/conversation_id_header.py` **retires** when the client ships | note | Confirm the header path is actually taken **before** deleting it — disabling it reverts conv_id to the hash and orphans everything written under `chat_id`. |

### 7.1 Corrections owed to the source documents

Each is a place where a thread that trusted the document would have built the
wrong thing. Two were in an earlier draft of this plan.

| Document | Says | Actually |
|---|---|---|
| `FRONTEND_HANDOFF.md` §3 B2 | no server turn sequence; memory inert against a bounded-window client | `turns_seen`, `_observed_position`, `window_offset`, content-addressed episodic ids and `_next_turn_index` all landed. One line remains. §3.2. |
| `FRONTEND_SPEC.md` §6, `V4_FEATURES.md:254` | the compactor skips the tail on an incomplete stream | a stopped reply is **stored as a sentence-trimmed prefix** (`main.py:2487-2513`) |
| `V4_FEATURES.md:322` | fork carries persona | `export_conversation` carries facts, summary state, episodic only; `portability.py:165` says *"a fork still loses the persona"* |
| `FRONTEND_SPEC.md` §5.4 | `DELETE …/facts` = "forget facts (all/matching)" | a five-layer wipe including persona; no single-fact route exists |
| `FRONTEND_SPEC.md` §5.4 | fact is `{text, added_turn, last_used}` | four keys — `pin` is the fourth; archived have five; **none is an id** |
| `FRONTEND_SPEC.md` §5.4 | (omits them) | `merge-into` and `cleanup-test-data` exist and are destructive |
| `FRONTEND_SPEC.md` §5.3 | six slash commands | also `/tidy`, `/retire`, `/pin`, `/unpin` |
| `FRONTEND_SPEC.md` §3.1 | `COMPACTOR_API_KEY` covers `/v1/*` today | no auth exists anywhere in this tree |
| `FRONTEND_SPEC.md` §5.1 | failure bodies are all OpenAI-error-shaped | two envelopes plus FastAPI's 422 |
| `FRONTEND_SPEC.md` §11.2 | the DDL to use "verbatim" | never defines `conversation`, `rev`, or `user_id` |
| `V4_FEATURES.md` §3.1 | a bounded window kills the summary hierarchy | the content half stands; the counter half is fixed |
| `V4_FEATURES.md` P6 | seven `[program:]` blocks, none with `stopwaitsecs` | ten blocks; `postgres` has `stopwaitsecs=60` |
