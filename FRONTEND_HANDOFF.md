# Front-end handoff — scope for the implementing thread

**Written 2026-09-09.** This is the brief for a separate thread that builds the
client described in [FRONTEND_SPEC.md](FRONTEND_SPEC.md). It exists so that
thread starts with a settled scope instead of the spec's fifteen open
questions.

Read this file first, then the spec. Where the two disagree, **this file
wins** — it records decisions taken after the spec was written.

---

## 1. What you are building, in one paragraph

A chat client for a single companion AI, replacing OpenWebUI. It talks to the
**compactor** (an OpenAI-compatible proxy at `:8080`), not to vLLM. The
compactor owns memory — facts, episodic retrieval, a hierarchical summary — and
the client owns the conversation itself. The reason this client exists rather
than a fork of OpenWebUI is §4.1 of the spec: the set of messages sent to the
model must be a *recorded intent*, verified before sending and shown to the
user. OpenWebUI silently sent 7 messages of a 241-message conversation and said
nothing, for weeks.

## 2. Scope

### Yours

- The whole client: shell, message list, composer, streaming, branch
  navigation, settings, PWA.
- The conversation store (§11) and the chain-integrity work (§4.1, §11.1–11.5).
- The context receipt (§12) **for the half the client can compute itself** —
  what it sent, and what it intended to send.
- The quality bars in §13, including the recalibrated scale bars in §4 below.

### Not yours — the compactor lane owns these

- Everything in spec §15 ("Server asks"). Do not implement server changes.
- Auth on `/admin/*` (§3.1).
- Anything inside `compactor/`.

**If you need a server change, write it down and hand it back.** Do not work
around a missing server capability by inventing client-side state that the
server is supposed to own — that is how the position accounting broke in the
first place.

## 3. Two blockers, and exactly what they block

Both are compactor-side and already owned. Neither stops you starting.

### B1 — `/admin/*` is localhost-only (spec §3.1)

Every `/admin/*` route is gated by `_require_localhost`. A client in a separate
container gets **403 on every memory endpoint**. `COMPACTOR_API_KEY` exists on
`v4/foundation-compactor-auth` (PR #30) but covers `/v1/*` only; extending it
to `/admin/*` is the fix.

**Blocks:** §7 (memory as a surface) and the memory half of §12's receipt.
**Does not block:** chat, streaming, storage, chain integrity, the whole UI.

### B2 — the server has no turn sequence (spec §15, "required")

The compactor's only notion of conversational position is the client's message
array length. Under §4 rule 2 a spec-compliant client sends a *bounded window*,
which is a near-constant — so every exchange overwrites the same ChromaDB
document, `recent_cutoff` stays ~0, retrieval returns nothing, and
`_needs_l1_rollup` is never true.

The spec states the consequence plainly, and it is worth repeating here because
it is easy to read past: **the memory architecture is inert against the client
this spec describes until this lands.**

**Blocks:** any acceptance test that asserts memory *works* end to end.
**Does not block:** building the client, or asserting that the client sends
what it intended to send.

Practical consequence: build against the spec, and expect memory features to
look dead until B2 ships. That is not your bug. Do not "fix" it by sending a
full history.

## 4. Decisions already taken — do not reopen these

**Q1, the client store — SETTLED: server-side, on local disk, archiving to
`/data`, converging on the Postgres sidecar** (ARCHITECTURE Decision 4). This
follows §11's own recommendation. The hard constraint from §11 stands: **under
no circumstances put the primary write-hot store on the network volume.** That
is what corrupted `webui.db` twice in two weeks, and a third time on 2026-09-07.

**§13's scale bars — RAISED, and this is the reason this handoff exists.** The
owner's requirement, stated 2026-09-09: the client must never reproduce
OpenWebUI's whole-conversation load. Measured on the pod that day:

| | |
|---|---|
| live conversation | **1,718 messages, 27.71 MB in one row** |
| growth | 25.66 → 27.71 MB **in one day** (~2 MB/day) |
| disk read of that blob | **0.45 s** |
| MooseFS write throughput | **473 MB/s**, `quick_check: ok` |

**The volume was healthy. The blob was the problem.** OpenWebUI stores the
whole chain as one JSON blob and ships it to the browser on every open. The old
bar — "a 500-message conversation scrolls without virtualization jank" — was
set at under a third of the real conversation and forbade the technique that
makes it possible. §13 now requires 2,000 messages / 30 MB, first paint under
1 s, and **append cost O(1) in conversation length**, with a test that measures
bytes written.

§11.1 already forbade the blob shape in prose. It did not stop the failure
shipping in the system being replaced, which is why the rule is now also a
measurement. **Treat the append-cost test as a phase-1 deliverable, not a
polish item** — it is cheap while the schema is malleable and expensive after.

## 5. Open questions you may decide yourself

Decide and record; no need to escalate: §17 Q2/Q12 (window size N — but it
*must* be recorded per request, not inferred), Q3 (branch UX depth), Q4
(offline), Q7 (design-system fit), Q11 (receipt depth).

**Escalate rather than decide:** Q8/Q9 (repair-vs-quarantine, refuse-vs-warn) —
these are user-facing policy on the exact failure that caused the incident;
Q10 (are multiple roots legitimate) — it commits a unique index; Q14 (shared
conversations during cutover) — it touches the live conversation.

## 6. How work is verified here

This project has standards that are not negotiable and that a new thread will
not guess:

- **A test only counts if it has been watched to fail.** Apply a mutation, run
  the suite, watch it go red, restore, watch it go green. On this project a
  surviving mutation has twice been a real gap and once a real defect hiding
  behind a test that agreed with the code.
- **Never restore from git after a mutation.** Keep the original in memory and
  write it back in a `finally`. A `git checkout --` has already destroyed
  another agent's in-flight work here.
- **Tests run on Linux, in Docker.** A host run on Windows is a development
  convenience, not evidence.
- **The recurring defect on this project is a rule applied at one call site and
  missed at its identical sibling.** Prefer sharing one function over copying a
  rule. It has cost this codebase eighteen incidents and counting, including
  one committed *while fixing an instance of it*.

## 7. Git

**Use your own branch, and your own worktree** (`git worktree add`). Do not
work in `D:/Projects/zions-light-ai` on a shared branch while another lane is
active. Stage explicit paths — never `git add -A` or `git add .`. Bundling two
agents' unfinished files into a third's commit has already happened once here.

Suggested: `feat/frontend` off `master`.

## 8. Phase 1 — definition of done

1. The store from §11, with §11.1's schema test (no serialized chain anywhere)
   and the append-cost test from §4 above.
2. The §4.1 chain integrity checklist, with the adversarial suite the spec
   names: multi-root, missing parent, deep-vs-current divergence, mixed key
   spaces, mid-chain failure, and the standing 241/5/8 case.
3. Streaming chat against the compactor, with the client-side half of the
   receipt: what it intended to send, and what it sent.
4. The scale bars met against a **generated** 2,000-message fixture — not
   against the owner's real conversation, which must not be a test subject.

Memory surfaces (§7) and the server half of the receipt come after B1 and B2
land. Ask for their status rather than assuming; they are tracked in the
compactor lane.

---

## Appendix — where things are

| | |
|---|---|
| Spec | [FRONTEND_SPEC.md](FRONTEND_SPEC.md) — §4.1, §11, §13, §15 are the load-bearing ones |
| Architecture | [ARCHITECTURE.md:138](ARCHITECTURE.md) — the "Decisions" list, item 4: the state home is Postgres (+ pgvector) on local disk |
| Compactor API | spec §5, verified against the code with line references |
| Why this client exists | spec §1 and the incident table at the top of §2 |
| Operational context | [OPERATIONS.md](OPERATIONS.md), [COMMANDS.md](COMMANDS.md) |
