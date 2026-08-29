# REVIEW PLAN — the 2026-08-24 incident response

**Status:** plan only. No review has been run against it yet.
**Subject:** everything produced between 2026-08-24 and 2026-08-27 in response to
the conversation-chain incident — 2,734 lines of new documentation, 4,408 lines
of code change across 30 files, and two open pull requests.
**Baseline:** `master` … `fix/v3.1-remediation`.

---

## 1. Why this plan exists

The work under review was produced fast, under incident pressure, largely by
agents, and coordinated by an author who **could not execute code on the
authoring machine**. That combination has a specific and now well-evidenced
failure mode, and this plan is built around it rather than around a generic
checklist.

The evidence is in the work's own history. Across four verification gates on
`fix/v3.1-remediation`, **every blocker found was introduced by a fix rather
than present in the original code**:

| Gate | Blocker | Introduced by |
|---|---|---|
| 1 | `TypeError` → HTTP 500 on every import and fork | the round that made `conversation_doc_count` return `None` |
| 2 | `NameError` in two exception handlers; `/why` 500'd where it had returned 200 | the fix for gate 1's blocker |
| 3 | Every import and fork 400'd in a supported config | the fix for gate 2's blocker |
| 4 | A test fixture that no longer produced the condition it asserted | the fix for gate 3's blocker |

All four were **invisible to reading and obvious to running**. No amount of
careful review-by-inspection would have caught them; each took one execution.

The same pattern appears in the analysis, not just the code. Claims that were
asserted confidently and later refuted include: that `Path.is_file()` swallows
`EIO`; that the episodic index was destroyed during the incident; that OpenWebUI
created a new root on each 400; that MooseFS had failed twice in two weeks; that
a client owning its message chain is structurally immune to this class of
failure. Each survived until someone measured it.

**The governing rule for this review: a claim is unreviewed until something has
executed it.** Reading a diff and agreeing with it does not count.

---

## 2. What there is to review

| Artifact | Size | Nature |
|---|---|---|
| [FRONTEND_SPEC.md](FRONTEND_SPEC.md) | 1,011 lines | Design specification; no implementation |
| [INCIDENT_2026-08-24.md](INCIDENT_2026-08-24.md) | 848 lines | Incident report; mixes established fact, inference and testimony |
| [REMEDIATION.md](REMEDIATION.md) | 875 lines | Action list; ~60 findings with file:line |
| PR #37 | docs + 3 config fixes | `Dockerfile`, `.env.example`, `runpod.env.template`, `compactor/requirements.txt` |
| PR #38 | 30 files, +4,408 / −145 | Phase 0, the `read_json_strict` chain, and the destructive-path removals |
| `log-sweep.py` | 1 tool | Reproduces the silent-handler inventory |

Five distinct kinds of claim live in there, and they need different treatment:

1. **Code-verifiable claims** — "`main.py:1202` reads `turn_index = len(messages) + 1`". Cheap to check, high volume.
2. **Behavioural claims** — "a corrupt facts file causes the tail to write a truncated set". Require execution.
3. **Production claims** — "the pod runs v3.0.5 with the vision model". Require the pod.
4. **Historical claims** — "the tree damage happened between 22:30 and 23:15 MST". Partly unrecoverable; the report already labels these.
5. **Design judgements** — "the client must not hold durable state". Not falsifiable by test; reviewable only against the north star and by argument.

---

## 3. Review tracks

Tracks A–C can run concurrently. D requires pod access. E requires a human.

### Track A — Claims audit (mechanical, agent-suitable)

**Question:** does every `file:line` reference in the three documents point at
what it says it points at?

There are several hundred. They were written across a week during which the
files they cite were themselves being edited, so drift is expected rather than
hypothetical — `REMEDIATION.md` already carries a warning to re-anchor by symbol
name rather than line number.

- Extract every `file:line` and `file.py:NNN` reference from all three documents.
- For each: does the file exist, does the line exist, and does the line contain
  what the surrounding prose claims?
- Classify: CORRECT · DRIFTED (right symbol, wrong line) · WRONG (different code) ·
  STALE (the code has since changed, e.g. by PR #38).
- **A DRIFTED reference is not a defect** if the symbol is still findable; a
  WRONG one is, because it sends an implementer to the wrong place.

**Exit:** zero WRONG. DRIFTED references either corrected or the document
annotated with the commit they were accurate at.

### Track B — Implementation review (agent-suitable, execution-backed)

**Question:** does PR #38 do what REMEDIATION.md says, and only that?

- Every changed hunk maps to a numbered remediation item. Anything unmapped is
  scope creep and must be justified or reverted.
- The five destructive paths are constructively verified — not "the code looks
  right" but *construct the destroying condition and show it no longer
  destroys*. The conditions are enumerated in §5.
- `pyflakes` and `ast.parse` clean; **zero undefined names**. Three rounds
  shipped one.
- No new violation of the P0-2b rule: a handler may swallow an exception only
  if it logs it, re-raises it, or returns it to a caller that surfaces it.
  Gates 1–3 each found one; assume a fourth exists until a sweep says otherwise.

**Exit:** every item verified by execution, scope clean, sweep clean.

### Track C — Test-quality audit (the one most likely to find something)

**Question:** can these tests fail?

This is the track that matters most, because the original `TypeError` survived
four rounds of review *behind a passing test suite*. `test_portability.py`
stubbed `conversation_doc_count` with something that always returned an `int`,
so the one value the guard had to reason about never reached the code under
test. The suite was green and proved nothing.

- **Mutation testing on every test added in PR #38.** For each, break the code
  it covers and confirm it goes red. A test that stays green is deleted or
  rewritten — it is worse than no test, because it certifies.
- **Audit every mock and stub in the existing suite** for the same defect: does
  it eliminate the case the test exists to cover? `test_portability.py:56` is
  the known instance; find the rest.
- **Fixture preconditions.** `test_import_guard.py` asserts its own precondition
  (that `conversation_doc_count` really returns `None` in that fixture) because
  a fixture that silently stops producing the condition turns every test below
  it into a happy-path test that still prints `ok`. Which other fixtures need
  that self-check?
- **Find tests that assert defective behaviour as correct.**
  `test_modality.py:261-268` did, and had to be rewritten before the calibration
  fix could land. There may be others.

**Exit:** every new test mutation-proven; every stub audited; a written list of
tests that cannot fail, with each deleted or fixed.

### Track D — Production verification (requires the pod)

Claims that cannot be settled from the repository. Each needs one command.

| Claim | Command | Currently |
|---|---|---|
| Tier-1 token counting works | `apply_chat_template` on the live `MODEL_REPO` | **Known broken** — no chat template on the vision model |
| The running image digest | `docker inspect` / RunPod console | **Unrecorded** |
| `MODEL_REPO` matches the corrected template | `echo $MODEL_REPO` | Asserted from a screenshot, not confirmed |
| ChromaDB journal mode | `PRAGMA journal_mode` | **Unrun** — gates V3's snapshot design |
| Backups are restorable | Restore one into a scratch dir | **Never tested** |
| `log-sweep.py` baseline | `python3 log-sweep.py --count` | Confirmed 46 → 20 |

**Exit:** every row confirmed, and `OPERATIONS.md` updated with the running
configuration so the next incident does not start with "what is actually
deployed?"

### Track E — Design review (requires a human)

Not falsifiable by test. Reviewable only by argument, and only by the owner.

1. **The pure-window directive.** A hostile review established that "a client
   holding no data cannot disagree with the backend" is false — under a single
   owner, render and send are built from the same source, so a stalled leaf
   produces "8 of 8" with every integrity check green. Consistency is not truth.
   Does the disclosure requirement (§4.1, §12) actually carry the weight now
   placed on it?
2. **The Decision 2 collision.** GPU scale-to-zero and a server-owned transcript
   are in direct conflict: reading yesterday's conversation would wake the pod
   and load a 24B model to display text. **Unresolved, and it is a cost
   decision, not an engineering one.**
3. **Scope of v3.1.** REMEDIATION.md scopes V1–V17 as "v3.1 core." That is weeks
   of evenings for one person, during which the live S1 paths stay in
   production. The recommendation on record is to rescope.
4. **Whether the front end is the right project at all.** The incident is used
   throughout as justification. A skeptical reading — that this was a
   pre-existing preference the incident licensed — deserves a fair hearing from
   someone who is not the author.

---

## 4. Order and dependencies

```
Track A (claims)  ─┐
Track B (impl)    ─┼─→ can run concurrently, agent-driven
Track C (tests)   ─┘
                    │
Track D (pod) ──────┴─→ needs the owner; gates the v3.1 release decision
Track E (design) ─────→ needs the owner; gates the front-end work
```

**Track C before any further implementation.** If the suite cannot fail, every
subsequent green result is uninformative — including the ones this plan will
produce.

**Track D before cutting v3.1.0.** Two of its rows are known-broken and one
(`PRAGMA journal_mode`) changes the shape of work already specified.

---

## 5. The destructive paths, and how to prove they are gone

Each must be verified by constructing the condition, not by reading the fix.

| Path | Destroying condition | Must now |
|---|---|---|
| `read_json` → write-back | Corrupt facts file, then an exchange | Skip the write; the file is unchanged |
| Dedup merge | LLM replies `"These are different, KEEP"` | Not merge; cluster intact |
| Dedup truncation | Response with `finish_reason: "length"` | Not merge |
| Fact eviction | Over-fill past `COMPACTOR_MAX_FACTS_TOKENS` | Evicted facts land in the archive, not unlinked |
| Lazy backfill | Tail write concurrent with a running backfill | Both sets survive |
| Backup | `COMPACTOR_STORAGE_ROOT` points at nothing | Fail, alert, and **not** prune |
| Backup retention | Ten restarts in a row | Oldest archive survives |
| `/remember` race | `/remember` concurrent with a tail write | Both facts present |
| Import guard | Vector store unreachable, `overwrite=False` | Refuse with 400, target untouched |

**None of these is verified by a passing test suite alone.** Each needs the
mutation check from Track C: break the fix, confirm the test goes red.

---

## 6. Claims currently carrying no proof

Stated for honesty. Each is asserted somewhere in the work and has not been
verified.

- The vision model's chat template is the same as the sibling `heretic-v4`
  snapshot's. **Assumed; must be checked against vLLM's `/tokenize`** before it
  becomes the input to every budget decision.
- `~22 tokens/message` of missing chat-template framing. Derived arithmetic from
  one observation, not measured across a corpus.
- The 2026-08-24 tree damage occurred 22:30–23:15 MST. Inferred from commit
  timestamps and the user's report; the logs were never captured.
- `roots=5` was caused by failed requests. **Explicitly not established** —
  §4.2 of the incident report says do not cite it. First-message edits create
  parentless nodes by design.
- OpenWebUI 0.11 reads from `chat_message` rather than `chat.chat`. Asserted;
  the whole of detection-failure §5.4 rests on it.
- Backups have never been restored. Not a claim — an absence. It is the single
  largest unverified assumption in the system.

---

## 7. Exit criteria

The work is "reviewed" when:

1. Zero WRONG file:line references (Track A).
2. Every PR #38 hunk maps to a remediation item; nothing unmapped (Track B).
3. Every new test is mutation-proven, and every stub audited for the
   `test_portability.py` defect (Track C).
4. All nine destructive paths verified by construction (§5).
5. Every Track D row confirmed on the pod and recorded in `OPERATIONS.md`.
6. The owner has ruled on the four Track E questions.
7. §6 is empty, or each remaining entry is annotated with why it stays
   unverified and what that costs.

Anything less is a partial review, and should be described as one.

---

## 8. How this review should not be run

Three anti-patterns, each of which already happened once this week.

**Do not verify an adjacent property.** The integrity check that reported the
incident conversation "healthy" measured the *deepest chain in the tree* (208)
rather than *the chain from the current leaf* (8). Both numbers were true. Only
one was the criterion. Every check in this review must state which property it
tests and why that is the property.

**Do not accept a green suite as evidence.** It was green throughout the
incident, and green while `test_portability.py` could not see the regression it
covered. Green means "nothing that is tested is broken", which is a much smaller
claim.

**Do not let an empty result read as a clean bill.** Several agent runs died on
API errors during this work and returned nothing. An empty result and a passing
result are visually similar and mean opposite things. Every step in this plan
must record what it *ran*, not only what it found.
