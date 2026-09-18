# Adversarial review — `2f6facf` "fix: a bad reply stopped the hierarchy, not just itself"

Worktree `D:/Projects/zla-v318-nopg`, branch `rel/v3.1.8-nopg`. Read-only review; no repo file
was created, edited or deleted, and no mutating git command was run.

---

## VERDICT

**Do not ship this in v3.1.8 as written.** The diagnosis is right and the extraction of
`_rollup_hierarchy` is good work, but the gate that decides when the skip-path rollup may run is
built on the wrong signal. It tests `decision.outcome` against `ROLLUP_UNSAFE_SKIP_OUTCOMES`, and
**both** members of that frozenset are labels that `_run_memory_tail` only ever computes for a
reply that was otherwise going to be *stored* (`main.py:4360` and `main.py:4370` are both guarded by
`if decision.store`). So for exactly the class of reply this commit exists to serve — one that
`decide_memory_tail` refuses on its own merits — neither guard can fire. I demonstrated both
consequences live in the test container: (1) a degenerate reply while `degrade.guard()` returns
False now writes summary state to disk, the write the commit's own comment says it refuses; and
(2) an OpenWebUI title-generation call whose reply hits the generation ceiling now drives a rollup
against a foreign one-turn array, which inflates `turns_seen` past the client's window and makes
`_do_l1_rollup` take its `pos_last < 1` branch — **advancing `last_summarized_turn` from 0 to 21
and declaring 21 real, never-summarized turns permanently dead.** That is strictly worse than the
frozen hierarchy the commit set out to fix: a frozen watermark can be thawed, a burned one cannot.
The fix is small (check `degrade.guard(...)` inside `_rollup_hierarchy`, and call
`_is_repeat_task_traffic(conv_id, messages)` at the gate instead of reading a label that was never
computed), but it must land before this ships, and the new frozenset needs the tests it currently
does not have.

---

## FINDINGS, most severe first

### F1 — CRITICAL / CONFIRMED. The rollup gate reads outcome labels that are never computed on the path it guards; one OpenWebUI task call can burn the entire un-summarized backlog

**Files:** `compactor/main.py:4360`, `4370`, `4420-4429`; `compactor/tailhealth.py:252-255`;
`compactor/summarizer.py:1607-1626` (`_do_l1_rollup`'s `pos_last < 1` branch).

**Mechanism.** `_run_memory_tail` computes the two "unsafe" labels only for storable replies:

```python
4360:  if decision.store:
4361:      _blocked = _tail_store_blocked(last_user_text)   # sole producer of SKIPPED_DISK_PRESSURE
...
4370:  if decision.store and _is_repeat_task_traffic(conv_id, messages):  # sole producer of SKIPPED_TASK_TRAFFIC
```

and then gates the new rollup on those same labels:

```python
4422:      decision.raw_chars > 0
4423:      and decision.outcome not in tailhealth.ROLLUP_UNSAFE_SKIP_OUTCOMES
```

Any reply that `decide_memory_tail` already refused — degenerate, holed, no-boundary, too-short,
degenerate-partial, whitespace-empty — arrives at line 4422 carrying *its own* label. The request
may still be task traffic, and the disk may still be refusing writes; the code simply never asked.
`ROLLUP_UNSAFE_SKIP_OUTCOMES` is therefore dead code in precisely the scenario the commit was
written for.

**Demonstration (run in the unit container, `/tmp` only, nothing written to the repo).**
Conversation seeded with 10 real exchanges (`turns_seen=20`, `last_summarized_turn=0`), then one
OpenWebUI title call — a single user turn, no assistant turn — whose reply hits the ceiling:

```
BEFORE: {'l1_chunks': 0, 'last_summarized_turn': 0, 'turns_seen': 20}
  _is_repeat_task_traffic(conv, task array) = True      <-- it IS task traffic
  outcome = skipped_no_boundary   raw_chars = 19
  outcome in ROLLUP_UNSAFE_SKIP_OUTCOMES? False         <-- but the label says otherwise
  scheduled labels: ['rollup conv=probeB']
  WARNING  none of the 4 anchored turns appear in the 1-turn window ... advancing by 2
  INFO     the client is sending a bounded window (1 turns) while the conversation is at turn 22;
           chunk text is read at an offset of 21
  ERROR    turns 1-21 are behind the client's window and were never summarized; the watermark
           has been advanced past them so newer turns are not lost too.
AFTER : {'l1_chunks': 0, 'last_summarized_turn': 21, 'turns_seen': 22}
```

Twenty-one turns of a real conversation are now marked summarized, with zero L1 chunks holding
their text. And the corruption persists: the *next* real turn's 22-turn window reports
`position = 24` against a truth of 22, i.e. `window_offset = 2` forever — the exact R12/R23
signature ("the position inflated by 2 per rollup and stayed inflated, so `window_offset`
subtracted 2 forever").

**Reachability.** The trigger is a task-traffic request whose reply is non-storable on its own
merits. `trim_to_last_sentence` needs a sentence terminator, and **a generated chat title or a JSON
tag list has none**, so any task call vLLM reports as `finish_reason=length` lands on
`SKIPPED_NO_BOUNDARY`. A dropped SSE chunk (`SKIPPED_HOLED`) and a task prompt that makes the model
loop (`SKIPPED_DEGENERATE`) reach it too. The docstring of `_is_repeat_task_traffic` itself counts
99 such requests in two days on one conv_id. This is not a corner case; v3.1.8's own N4b protection
is punched straight through.

**Sub-finding F1a — disk pressure, also CONFIRMED.** `_rollup_hierarchy` (`main.py:4173-4243`) has
no `degrade.guard`. The normal path's guard lives in `_async_tail` at `main.py:4111` and was *not*
extracted along with the rollup — which is the fix-one-site-miss-the-sibling shape the commit
message spends a paragraph condemning. `maybe_rollup` → `save_state` → `atomic_write_json`
(`summarizer.py:248-251`) has no guard of its own; `grep -n "degrade\."` over the package shows the
only callers are `main.py:4111`, `4288`, `5139` and `health.py:299`. Demonstrated:

```
=== PROBE A: degenerate reply WHILE degrade.guard() is False ===
  degrade.guard(...) -> False
  outcome = skipped_degenerate  raw_chars = 584  store = False
  outcome in ROLLUP_UNSAFE_SKIP_OUTCOMES? False
  scheduled labels: ['rollup conv=probeA']
  summary file exists BEFORE: False
  summary file exists AFTER : True        <-- a write taken after the decision not to write
```

**Recommended fix.** Two lines, both at the level where the truth is available rather than the
level where a label happens to have been assigned:

1. Put `if not degrade.guard("hierarchy rollup"): return` at the top of `_rollup_hierarchy`, so the
   one function both paths share carries the one guard both paths need. This also removes
   `SKIPPED_DISK_PRESSURE` from the frozenset's job entirely.
2. At the gate, call `_is_repeat_task_traffic(conv_id, messages)` (or at minimum
   `_has_conversational_history(messages)`) rather than testing for the label. Cheap: it is one
   state read and only on arrays with no assistant turn.

With both in place `ROLLUP_UNSAFE_SKIP_OUTCOMES` can be deleted, which is the better outcome — the
set encodes preconditions as labels, and labels are the thing that turned out not to be computed.

---

### F2 — HIGH / CONFIRMED. The new gate has no test coverage at all, and the broken implementation of F1 passes the whole suite

**Files:** `compactor/test_rollup_on_skip.py` (whole file); `compactor/test_task_traffic.py:157`.

`grep -rn "ROLLUP_UNSAFE_SKIP_OUTCOMES\|_rollup_hierarchy" compactor/` returns **one hit outside
`main.py`/`tailhealth.py`, and it is a comment**. Neither member of the frozenset, and neither half
of the `raw_chars > 0` condition, is exercised anywhere.

`test_rollup_on_skip.py` is not vacuous about what it does test — `summarizer.enabled()` must be
True or `len(rolled) == 1` fails; the fixture's degeneracy is asserted up front; passing
`assistant_text=DEGENERATE` would leave `RULE*569` in the input and fail the exclusion check
(`_redact_degenerate_turns` does not touch the appended reply) — but its scope is one outcome.
Concretely, **this implementation passes the entire 61-suite run**:

```python
if not _fire_and_forget(_rollup_hierarchy(conv_id, messages, None), label=...):
    logger.warning(...)
```

i.e. both gate clauses deleted. That is the implementation of F1. Two further gaps: the test's
`HISTORY` contains no degenerate turn, so an implementation that never calls
`_redact_degenerate_turns` on the skip path also passes; and `maybe_rollup` is stubbed, so nothing
here pins the actual claim (that the watermark advances). `test_task_traffic.py:157` replaces
`_fire_and_forget` globally with `_no_schedule`, so it cannot see the rollup either.

Minimum additions: a task-traffic array with a `finish_reason=length` reply must schedule nothing;
a skip under `degrade.guard -> False` must schedule nothing and must leave `summaries/` empty; a
`raw_chars == 0` skip must schedule nothing; and a skip whose history *contains* a degenerate turn
must still have it redacted.

---

### F3 — MEDIUM / CONFIRMED. The spy narrowing in `test_truncated_tail.py` deletes the only assertion that covered rollup suppression under disk pressure

**File:** `compactor/test_truncated_tail.py:277-281`, and the assertions at `:693` and `:767`.

Before the narrowing, `_spy_fire` appended unconditionally, so `assert_eq(len(_fired), 0,
"{label}/disk: the tail was NOT fired")` at `:693` proved that **nothing at all** was scheduled
under disk pressure. Removing `SKIPPED_DISK_PRESSURE` from `ROLLUP_UNSAFE_SKIP_OUTCOMES` would have
made that count 1 and failed the suite. It no longer can — which is exactly the mutation F1a is.

The narrowing was also load-bearing for green rather than merely cosmetic: `:767`
(`"no-user-text: the tail was NOT fired"`) would now **fail** unnarrowed, because
`SKIPPED_NO_USER_TEXT` is not in the unsafe set and does schedule a rollup. So the diff quietly
converts a total-background-work assertion into a tail-only one in order to accommodate a behaviour
change, and that conversion is where the disk-pressure coverage went.

`test_degenerate_skip.py`'s narrowing is benign — its four assertions were always about the tail,
and the `coro.close()` stayed outside the `if`, so no coroutine leaks. `test_truncated_tail`'s was
mechanically necessary (`_fired_text()` reads `_fired[0]["assistant_text"]`), but the disk block
should have kept a separate "and no rollup either" assertion.

---

### F4 — MEDIUM / CONFIRMED. `raw_chars > 0` does not mean "the model produced something", and does not protect a failing vLLM

**File:** `compactor/main.py:2532`, `2543`, `4422`.

The commit's justification is that a backend rejection produces no text, so the rollup will not
hammer a failing engine. That holds only for a clean 4xx (the streaming site's comment at
`main.py:5598-5604` confirms the accumulator is never fed, so `raw_chars == 0`). It does **not**
hold for the outage shape this file documents elsewhere — a stream that dies mid-reply (`vllm_failed`,
"28 such streams today"). There, text has arrived, the outcome is `SKIPPED_NO_BOUNDARY` or
`SKIPPED_TOO_SHORT` with `raw_chars > 0`, and the rollup now fires summarization requests at the
process that is already failing.

Symmetrically, `raw = len(text)` is taken at `main.py:2532` *before* the `text.strip()` test at
`2543`, so a whitespace-only reply is `SKIPPED_EMPTY` with `raw_chars > 0` and answers "yes, the
model produced something".

Severity is capped by `needs_rollup` — the LLM calls only happen when a chunk is actually due — so
this is load, not corruption. But the stated rationale is weaker than the comment claims, and if
the gate is rewritten per F1 this condition should be re-derived rather than carried over.

---

### F5 — LOW / CONFIRMED. The skip-path rollup creates summary state for conversations that stored nothing, and `/admin/conversations` counts them forever

**Files:** `compactor/summarizer.py:1960-1972`, `2016-2021`; `compactor/memory.py:274-293`.

`_observed_position` unconditionally writes `turns_seen`/`tail_fp`/`head_fp`/`window_turns`, so
`changed` is True and `save_state` runs even when no tier rolled. Probe A shows the file appearing
for a conversation whose only turn was refused (`summary file exists AFTER : True`, `turns_seen: 3`,
`l1_chunks: 0`). `memory.list_known_conv_ids` unions `facts/*.json` and `summaries/*.json`, so every
such conv_id is now a permanent row in `/admin/conversations`. This is the v3.1 G2 defect class
("an empty write here creates a facts file for every background utility call the compactor ever
sees, and `list_known_conv_ids` counts them forever") arriving on the summaries side, and it
compounds F1 for task-traffic conv_ids.

---

### F6 — LOW / SUSPECTED. `_ASSUMED_NEW_TURNS`'s stated premise is now false, and no position test covers a window that ends on a user turn

**Files:** `compactor/summarizer.py:657-663`, `1005-1024`; `compactor/test_position_edges.py`.

`_ASSUMED_NEW_TURNS = 2` is justified by "main.py's tail calls maybe_rollup exactly once per
exchange, **with the user turn and the assistant turn it just produced**". The skip path is the
first caller in the codebase that hands `maybe_rollup` a window ending on a *user* turn. The comment
was not updated.

The arithmetic mostly survives, and I want to be explicit that I traced it rather than assumed it:
because `_turn_fingerprints` prefixes the role into the hash, the period-2 ambiguity of a repeating
tail can never make a user-ending window align against an assistant-ending anchor. First skip after
a store aligns at `+1`, mid-run at `+2`, the recovery turn at `+3` — all exact. So the anchored path
is clean.

The residual is the unanchored fallback (`cands` empty → `+2` per call). Across a whole skip run the
charged advance `2(k+1)` equals the true advance `2k+2`, but it is one turn high *during* the run.
Under a bounded client window that makes `window_offset` one too large, so an L1 chunk cut inside a
loop is labelled one turn ahead of its text. Narrow — it needs a cap, anchor loss, and a chunk
boundary landing mid-loop — and it self-corrects on an unbounded client where `n` dominates. I did
not reproduce it.

Worth noting regardless: every case in `test_position_edges.py` builds its windows through
`_ex(u, a)`, i.e. exchange-aligned. The shape this commit introduces has no test.

---

### F7 — NITS

* `SKIPPED_SHED` can never reach the gate (it is constructed at `main.py:4475`, after the skip
  branch returns), yet `ROLLUP_UNSAFE_SKIP_OUTCOMES`'s docstring enumerates "everything else here"
  without mentioning it, so a reader cannot tell whether it was considered. Related asymmetry: a
  shed tail still loses its rollup, which the commit's own premise says should not happen.
  Defensible under load, but undocumented.
* The warning at `main.py:4427` says "the watermark does not advance until a later turn is
  **accepted**" — under this change a later *skipped* turn advances it too.
* `main.py:4237` changed the rollup log line's arrow from `→` to `->`. Nothing greps it (verified),
  so this is cosmetic, but it is an unannounced change to a production log line inside a commit
  whose stated scope is control flow.
* The commit message reports "62 passed, 0 failed, 2 skipped". My run of the same command reports
  61 passed, 0 failed, 1 skipped.

---

## PER-AREA ANSWERS

### 1. Is the gate correct and complete?

The **membership** of `ROLLUP_UNSAFE_SKIP_OUTCOMES` is right; the **gate** is wrong, because it
reads labels that only exist on the store path (F1). Walking every outcome in `tailhealth.py`:

| Outcome | Reaches the gate? | Rollup after it? | Verdict |
|---|---|---|---|
| `STORED`, `STORED_TRIMMED` | no (`decision.store` True) | n/a | — |
| `SKIPPED_HOLED` | yes, `raw_chars > 0` | fires | Safe *as a reply judgement* — the hole is in the current reply, which is excluded via `None`; the history is real. **But** it is one of the routes by which task traffic and disk pressure reach the rollup unchecked (F1). |
| `SKIPPED_EMPTY` | only if the reply is whitespace (`raw_chars > 0`) | fires | Harmless in itself: the user turn is genuinely new, so rolling up the history is right. Inconsistent with the commit's own "did the model produce anything" framing (F4). |
| `SKIPPED_DEGENERATE` | yes | fires | The target case, and correct in principle. Primary F1/F1a carrier. |
| `SKIPPED_NO_BOUNDARY` | yes | fires | The route my probe used to burn 21 turns (F1). Also the vLLM-mid-stream-death route (F4). |
| `SKIPPED_TOO_SHORT` | yes | fires | Same as above. `MIN_MEMORABLE_TRIMMED_CHARS=300` makes this the default landing spot for any short truncated reply. |
| `SKIPPED_DEGENERATE_PARTIAL` | yes | fires | Same class as `SKIPPED_DEGENERATE`; correct in principle, same F1 exposure. |
| `SKIPPED_DISK_PRESSURE` | yes | **blocked** | In the set, and correctly so — but only ever produced when the reply was otherwise storable, so it protects a small minority of the cases that need protecting (F1a). |
| `SKIPPED_NO_USER_TEXT` | yes | fires | Genuinely safe as a *reply* judgement: the array is a real conversation and `_turn_fingerprints` handles a text-free user turn deterministically (`_image_only_marker`, then `"user\x00"`). Correctly excluded from the set. |
| `SKIPPED_TASK_TRAFFIC` | yes | **blocked** | In the set, correctly so, and comprehensively bypassed (F1). |
| `SKIPPED_SHED` | **never** — constructed after the branch returns | n/a | Absence from the set is harmless; absence from the docstring is a readability nit (F7). |

Nothing should be *added* to or *removed from* the set. What is needed is that the two conditions
be evaluated rather than inferred.

### 2. Concurrency and duplication

**No defect found.** The two paths are mutually exclusive per request — the skip branch returns
before `_async_tail` is fired — so there is exactly one rollup per exchange, as before.
`_rollup_hierarchy` holds no lock when it calls `maybe_rollup`, so the internal `conv_lock` is not
re-entered and there is no deadlock; `load_state` → mutate → `save_state` all sit inside that lock
(`summarizer.py:1949-2021`), so no lost update. No double-advance: `_do_l1_rollup` moves
`last_summarized_turn` only after its own LLM call returns, under the lock.

One thing I did chase and clear: because the skip path is a far shorter code path than `_async_tail`
(no episodic index, no `_facts_tail` vLLM round trip), a skip-turn rollup can now overtake a
still-pending tail rollup for an *earlier* turn — an interleaving that could not occur before this
commit. I traced it: the older call then sees an anchor that ends past its own window, the prefix
walk yields `cands[0] == 0`, `position = max(n, prev + 0) = prev` holds by monotonicity, and the
anchor is rewritten from the older window. The next turn re-aligns. Recoverable, so I am not
reporting it, but it is the reason F6's fallback case is worth keeping an eye on.

### 3. Cost and load

**No unbounded cost found.** The extra LLM calls are gated by `needs_rollup` inside `maybe_rollup`,
so a model that loops for *n* turns triggers no more summarization than the same *n* turns would
have if they had been stored — the claim in the commit message is right. The per-skip overhead is
one pool slot, one `run_in_threadpool(_redact_degenerate_turns, ...)` walk (446 ms of a *worker*
thread at 170 turns, per the module's own measurements — off the event loop), one `load_state`, and
usually one `save_state`.

`bgwork.pool.submit` **does** close a coroutine it refuses (`bgwork.py:113-129`, inside the shed
branch, with a `log_once` warning if `close()` itself fails) — no leaked coroutine, no "never
awaited" warning. The short-circuit `and` at `main.py:4422-4426` also means the coroutine object is
never constructed when the gate rejects.

The one real load note: a skipped turn now consumes an outstanding slot it previously did not, so at
`MAX_OUTSTANDING=64` the shed rate for genuine tails rises slightly under a sustained loop. Marginal.

### 4. Position tracking

Mostly clean, with one documentation defect and one narrow residual — see F6. The difference the
question asks about (skip path does not append the reply) is **not** in itself corrupting: the
invariant `maybe_rollup` needs is "the window's last element is turn `current_turns`", and that
holds for a user-ending window as well as an assistant-ending one, because `window_offset =
current_turns - n` is derived the same way. Role-tagged fingerprints keep the alignment exact
across the parity change. `turns_seen`, `tail_fp`, `head_fp` and `window_turns` are all written
consistently.

The R23/R12 mislabel **is** reachable — but through F1, not through the parity change: the
task-traffic array is what corrupts `window_offset`, and it corrupts it by 2 permanently, plus the
one-shot `pos_last < 1` watermark burn.

### 5. The extraction itself

**Behaviour-identical for the normal path.** I diffed the moved block line by line:

* `if summarizer.enabled() and assistant_text.strip():` became two guard clauses; for a `str`
  argument the truth table is unchanged, and the added `assistant_text is not None` arm is the only
  new case.
* The `try` scope is identical — `summarizer.enabled()` was outside it before and is outside it now.
* `list(original_messages)` → `list(messages)`; the defensive copy survived.
* The `full_messages` construction is the same list plus a conditional tail.
* `before`/`state` comparison and the INFO log are byte-identical apart from the arrow (F7).
* `except Exception` and its `logger.exception` message are unchanged.

No lost guard, no changed exception scope. The one thing that *should* have moved with it and did
not is `degrade.guard` (F1a) — but that is a gate the extraction inherited from `_async_tail`'s
preamble, not a line inside the block that was moved.

A leftover comment above the call site in `_async_tail` still describes the whitespace rule that now
lives inside `_rollup_hierarchy`; harmless duplication.

### 6. The test changes

Covered in F2 and F3. Summary: the narrowing in `test_truncated_tail.py` **does** lose a real
assertion (disk-pressure rollup suppression, which is F1a's own mutation); the narrowing in
`test_degenerate_skip.py` does not; `test_soak_conversation.py`'s `return True` is a correct fix to
a genuine double-contract bug and, because `_defer_tail` appends *every* label, the soak does now
drain the skip-path rollups. `test_rollup_on_skip.py` is honest about the one case it covers and
silent about everything else, and the F1 implementation passes it.

---

## TEST RUNS

**Unit suite.** `docker compose -f docker-compose.tests.yml run --rm unit-tests`

```
62 suite(s), interpreter python
61 passed, 0 failed, 1 skipped in 339s
  SKIP test_tokenizer_contract.py — http://localhost:18000/_fixture/info unreachable
EXIT=3        (3 == something was skipped, not a failure)
```

All three touched suites pass: `test_degenerate_skip.py` 5.3s, `test_truncated_tail.py` 6.9s,
`test_rollup_on_skip.py` 2.0s. `test_soak_conversation.py` did not appear in this stack's list (it
is fixture-backed).

**Probes.** Three read-only probes run inside the same image via
`docker compose -f docker-compose.tests.yml run --rm --entrypoint sh unit-tests -c '...'`, writing
only to the container's `/tmp` and a `tempfile.mkdtemp()` storage root. Verbatim output is quoted in
F1 and F1a above. Nothing in the repo was touched; the compose file mounts `./` read-only and the
probes copied it to `/work`.

**Soak.** Not run. The unit suite plus the three probes were decisive, and the soak exercises the
fix's happy path — which I do not dispute — rather than any of the findings above. Someone should
still run it before merge with the F1 fix applied, since F1's task-traffic path is exactly the kind
of thing a 22-turn soak against an adversarial fixture might surface.

---

# Re-review of `ee59de1`

Independent adversarial re-review of `ee59de1` ("fix: the rollup gate read a label that was never
set") on top of `2f6facf`, branch `rel/v3.1.8-nopg`, worktree `D:/Projects/zla-v318-nopg`.
Read-only: no repo file created, edited or deleted; no mutating git command; `git status --porcelain`
empty throughout. All execution in throwaway containers.

## VERDICT

**Ship v3.1.8.** F1 and F1a are genuinely closed, and closed at the right level — the gate now asks
the question instead of reading a label, and the write guard lives inside the one function both
callers share. I could not construct a route back to the watermark burn. What I found instead is
four smaller things, none of which I would hold the release for: one redundant blocking read per
task request, one test-hygiene defect that also falsifies a comment this repo relies on, a sharpened
version of F4 the author should accept knowingly, and a residue of F2's coverage gap. F4 and F5 I
would ACCEPT — F5 more comfortably than the previous report could, because `ee59de1` removed its
main amplifier.

## A. Is F1 actually closed, or did the defect move?

**Closed. CONFIRMED.**

*No reader of the frozenset remains.* `grep -rn "ROLLUP_UNSAFE_SKIP_OUTCOMES" .` over the entire
worktree returns **zero hits** — not in `main.py`, not in `tailhealth.py`, not in a test, not in a
comment. The definition is deleted and nothing dangles. `SKIPPED_DISK_PRESSURE` survives as a label
in its three legitimate places (`tailhealth.py:117`, `:147`, `main.py:4297`).

*The gate now asks.* `main.py:4428-4440`:

```python
if (
    decision.raw_chars > 0
    and not _is_repeat_task_traffic(conv_id, messages)
    and not _fire_and_forget(_rollup_hierarchy(conv_id, messages, None), label=...)
):
```

`_is_repeat_task_traffic` is a module-global lookup with no `decision.store` precondition, so on the
skip path it is evaluated for real. The short-circuit ordering is right: `raw_chars > 0` first, so a
reply that never arrived costs no state read, and the coroutine object is not constructed when
either condition rejects, so nothing leaks in production.

*The burn is unreachable, and not only because the gate blocks it.* Two independent reasons:

1. The gate blocks any history-less array once `_recorded_position >= TASK_TRAFFIC_MIN_POSITION`
   (4). That is the shape the previous probe used.
2. Even with the gate removed again, the 0 -> 21 burn needs `needs_rollup` True, which needs
   `current_turns - last_summarized_turn >= L1_CHUNK_SIZE` (`summarizer.py:94`, default **20**) —
   and `TASK_TRAFFIC_MIN_POSITION` is **4**. Below the threshold there are no chunks and no backlog
   to burn; above it the gate refuses. The two constants leave no window between them.

*Can it raise?* No new exception surface. `load_state` -> `read_json_strict` (`memory.py:384-444`)
raises exactly `StoreUnreadable` for JSONDecodeError, OSError and wrong-shape, and `path.is_file()`
re-raises non-ENOENT OSErrors — both are inside the `except (OSError, StoreUnreadable)` at
`main.py:2859`. `_recorded_position` operates on a dict already normalised by `load_state`, so its
three `int(...)` casts cannot raise. Nothing new can escape `_run_memory_tail`.

*Blocking I/O.* Yes — `_run_memory_tail` is a plain `def` called directly on the event loop from both
request sites (`main.py:5630` inside `event_stream()`, `main.py:5752` in the non-streaming handler),
so `summarizer.load_state` is a synchronous `open()`+`json.load` on the loop. But
`_has_conversational_history` short-circuits first, so an ongoing conversation never pays it: the
docstring's "never on the hot path" claim holds on the skip path too. Cost per affected turn is one
summary-file read — sub-millisecond locally, tens of ms on the MooseFS shapes this file documents.
Not a finding on its own; see N1 for the part that is.

*Wrong answer on a degenerate-reply turn?* No, and I looked hard. A degenerate reply on a real
conversation arrives with prior assistant turns in the array, so the function short-circuits False
before touching the store — the same answer the old (never-firing) guard gave. The only "wrong"
direction is over-blocking: a first-message regeneration on a conversation already past position 4
reads as task traffic and loses its rollup. That is documented N4b behaviour and it restores the
pre-`2f6facf` outcome for that turn, so it is not a regression.

*The residual weakness is inherited, not introduced:* the protection is exactly as strong as
`_is_repeat_task_traffic`, which by construction only recognises arrays with **no** assistant turn.
A foreign array carrying an assistant turn would still pass. I checked whether OpenWebUI produces
one — its title, tag, follow-up, query-generation and autocomplete templates all embed the history
inside a **single user message**, so all of them are caught. Nothing found.

## B. Are the three new cases sound?

**Sound, with one real residue and two nits.**

Case **[3]** genuinely exercises the gate. `_run_memory_tail` resolves `_is_repeat_task_traffic`
through the module global, so `setattr(main, ...)` reaches it, and the control (`lambda: False` ->
non-empty) rules out the "gate refuses everything" failure. Mutations I traced against it:

* both gate clauses deleted (the `2f6facf` implementation) -> [3] red.
* `_has_conversational_history(messages)` substituted for the call -> [3] red (the patch no longer
  reaches the predicate, and HISTORY has an assistant turn, so the True case still schedules).
* the "optimisation" a maintainer would plausibly reach for —
  `_task = decision.store and _is_repeat_task_traffic(...)` hoisted above line 4367 and reused as
  `and not _task` — restores the exact F1 defect (`decision.store` is False on this path) and turns
  [3] **red**. The most likely future regression is pinned.

Case **[4]** driving `_rollup_hierarchy` directly is **not a meaningful weakening**. Case [1]
supplies the missing half: it goes through `_run_memory_tail`, asserts the scheduled label is the
rollup, and then `asyncio.run`s the very coroutine the gate constructed — which is the composition
[4] would otherwise have to prove. And [1] passing (`len(rolled) == 1`) is what makes [4]
non-vacuous: it establishes that `summarizer.enabled()` is True in this environment, so [4]'s
`rolled == []` cannot be an artefact of the earlier `enabled()` return. Stubbing `maybe_rollup` is
adequate for "no write happens" because `_observed_position` and `save_state` both live inside it.

Case **[5]** tests the real thing after its first-pass vacuity was fixed: `decision.raw_chars == 0`
is asserted separately, so `_empty == []` cannot pass for the wrong reason.

**Broken implementations that still pass the whole file** — the answer to "try to construct one":

1. **The gate calls the right function with the wrong array** — e.g.
   `not _is_repeat_task_traffic(conv_id, messages + [{"role": "assistant", "content": ""}])`, which
   can never return True in production. Case [3]'s `lambda c, m: True` ignores its arguments, so the
   file is blind to it. Low plausibility, but a real mutation hole; one line to close (assert the
   lambda received `HISTORY`).
2. **`_redact_degenerate_turns` never called on the skip path.** `HISTORY` still contains no
   degenerate turn, so [1]/[2] pass regardless — the previous report's F2 gap, unchanged. Mitigated
   in practice by the extraction (one function, both callers) and by the soak, which logs
   `redacted 13 degenerate historical turn(s) from rollup input` on the run I did; not pinned by the
   unit file.
3. **No test anywhere drives the *real* `_is_repeat_task_traffic` through the skip path.** The F1
   reproduction itself — seed a conv to position >= 4, send a history-less array with a
   `finish_reason=length` reply, assert nothing is scheduled — does not exist.
   `test_task_traffic.py` covers the function and the store path but replaces `_fire_and_forget`
   with `_no_schedule` globally, so it cannot see the rollup at all. One `_seed_store(conv, 20)`
   case in `test_rollup_on_skip.py` would close 1 and 3 together.

## C. F3 — is the lost coverage replaced?

**Substantively yes; the old assertion cannot be restored, and should not be.** Under `ee59de1` the
skip path *does* legitimately schedule a rollup coroutine under disk pressure — the coroutine then
returns immediately at `main.py:4206`. So an unnarrowed `_spy_fire` would now count 1 at
`test_truncated_tail.py:693` and the suite would go red for a correct implementation. The property
that matters ("no write is taken after the decision not to write") moved to case [4], which pins it
at the level where it now lives. That is the better place for it.

Residual: nothing asserts the disk-pressure outcome **end to end** from `_run_memory_tail` (no state
file appears). Since `maybe_rollup` is the sole writer and [4] blocks it, that is not a real gap.

## D. F4, F5, F6 re-checked

**F4 — ACCEPT, but accept the sharpened version, not the original.** I agree `raw_chars > 0` is not
"the model produced something" and does not protect a mid-stream-dying vLLM; that reasoning is
unchanged and the code is unchanged. What the previous report under-stated is the *shape* of the
load. `_llm_summarize` (`summarizer.py:1411`) takes `timeout: float = 300.0` and passes it as the
whole `client.post` timeout, so against a vLLM that is **black-holing** rather than refusing (a pod
mid-restart, an OOM-thrashing engine — packets dropped, no RST) each rollup holds a `bgwork` slot for
up to five minutes. And it is not one turn in twenty: once the first rollup fails,
`last_summarized_turn` does not move, so `_needs_l1_rollup` stays True and **every** subsequent
skipped turn fires another one. Before `2f6facf` an outage produced no background work on this path
at all. Still not a blocker — `MAX_OUTSTANDING` is 64, one rollup per turn per conversation will not
fill it from a single user, shedding is counted and warned, and `_rollup_hierarchy`'s
`except Exception` catches the failure — but "raw_chars > 0 stops us hammering a failing engine"
should not be written down as though it were true. If anything is added later, a short timeout on
the skip-path rollup is the cheap fix.

**F5 — ACCEPT, and it is smaller than it was.** The mechanism is confirmed unchanged:
`_observed_position` always writes `turns_seen`/`tail_fp`/`head_fp`/`window_turns`, so `changed` is
True and `save_state` runs even when no tier rolled (`summarizer.py:1960-2021`). But `ee59de1`
removed the amplifier the previous report leaned on — task-traffic conv_ids past position 4 no
longer reach the rollup at all. What is left is a conversation whose *first* reply was refused
getting a `summaries/*.json` with zero chunks. That is a real conversation, so a row in
`/admin/conversations` is honest, and the early task calls that still slip through already create a
`facts/*.json` and were already counted. I would not block on it and I would not fix it now.

**F6 — still a documentation defect; the empirical worry is reduced.** The `_ASSUMED_NEW_TURNS = 2`
comment (`summarizer.py:657-663`) still says the tail calls `maybe_rollup` "with the user turn and
the assistant turn it just produced", which the skip path falsifies, and `test_position_edges.py`
still builds every window through `_ex(u, a)`. But the soak I ran drives exactly the
user-ending-window shape for 14 consecutive turns and its final assertion — *"every turn the
watermark has passed is covered by a real, non-empty L1/L2/L3 entry"* — passes, with the L1 chunk
landing exactly on turns 1-20. Direct evidence against the residual, on the shape the residual is
about. Correct the comment; not a blocker.

## E. The 62/2 vs 61/1 line — the author is right

Ran both. The existing report's F7 bullet should be corrected to say that run was made **without**
`--saturation`.

| command | suites | result |
|---|---|---|
| `run --rm unit-tests` | 62 | 61 passed, 0 failed, 1 skipped in 187s, EXIT=3 |
| `run --rm unit-tests --saturation` | **64** | **62 passed, 0 failed, 2 skipped** in 212s, EXIT=3 |

`--saturation` adds `test_saturation.py` (passes) and `test_soak_conversation.py` (SKIPs here — no
`/tokenize` fixture in this stack). The commit message's "62 passed, 0 failed, 2 skipped
(fixture-backed, run separately)" is accurate.

## F. New findings

### N1 — LOW / CONFIRMED. Task-traffic requests now read the summary state twice per turn, on the event loop

`_is_repeat_task_traffic(conv_id, messages)` is called at `main.py:4377` (behind `decision.store`)
and again at `main.py:4436` in the skip branch. For a storable task-traffic request the first call
returns True, rewrites `decision` to `SKIPPED_TASK_TRAFFIC`, and the request then falls into the skip
branch — which asks the same question again, doing a second synchronous `load_state` on the event
loop. There is no cache between them. Correct outcome, doubled cost, on precisely the request class
this release is trying to stop spending money on. Fix is one local:
`_task = _is_repeat_task_traffic(conv_id, messages)` computed once and read at both sites. Note the
*conditional* hoist — `_task = decision.store and _is_repeat_task_traffic(...)` — is the F1 defect
again; whoever does this must not gate the computation on `decision.store`.

### N2 — LOW / CONFIRMED. `test_truncated_tail.py` leaks 19 rollup coroutines, and the comment saying the runner would catch that is false

`_spy_fire` (`test_truncated_tail.py:269-281`) appends only `tail`-labelled work and **never calls
`coro.close()`** — it never had to before, because `_spy_tail` returned a dict rather than a
coroutine. Since `2f6facf` a real `_rollup_hierarchy` coroutine reaches it and is dropped. Measured,
by running the suite directly under the image's own interpreter:

```
/opt/compactor-venv/bin/python -W always test_truncated_tail.py
  RC=0 ... All truncated-tail tests passed.
  grep -c "never awaited" stderr  ->  19
  /repo/compactor/main.py:4437: RuntimeWarning: coroutine '_rollup_hierarchy' was never awaited
```

`test_degenerate_skip.py` and `test_task_traffic.py` both close correctly; this one does not.

The second half is the part worth acting on. `test_task_traffic.py:143-145` states that leaving a
coroutine unawaited "turns a passing suite into a RuntimeWarning that the runner reports as a
failure." **It does not.** `scripts/run-tests.py:143-157` judges a suite by `r.returncode` alone and
only concatenates stdout/stderr once the suite has already failed, so all 19 warnings are discarded
and the suite reports `ok`. A comment asserting a safety net that does not exist is the same class of
defect as the frozenset that was never read. Either close the coroutine in `_spy_fire` and correct
the comment, or make the runner fail on `RuntimeWarning`.

Side benefit of the measurement: 19 warnings is proof that `test_truncated_tail.py` reaches the new
skip-path gate 19 times — and asserts nothing about it.

### N3 — LOW / SUSPECTED. Below `TASK_TRAFFIC_MIN_POSITION` the gate is open, so early task traffic can still drift the position

The gate inherits N4b's threshold, so for a conversation at position 1-3 a task call whose reply is
non-storable (a title that hit the generation ceiling -> `SKIPPED_NO_BOUNDARY`) still drives a rollup
against its one-turn array. With no anchor yet, `_observed_position` takes the unanchored `+2`
fallback, so `turns_seen` and `head_fp`/`tail_fp` drift against a foreign array, and the offset
persists into the conversation's first real L1 chunk labels. Bounded three ways: at most a few calls
before the threshold latches; no chunks exist that early, so no watermark can be burned (see A2); and
a *complete* title reply is `STORED` rather than skipped, so it never takes this path. Marginal
amplification of behaviour N4b already accepts. I did not reproduce it.

### N4 — NITS

* `main.py:4444` still says the watermark "does not advance until a later turn is **accepted**";
  under this change a later *skipped* turn advances it too. Carried over from F7, uncorrected.
* `main.py:4162-4165` still carries the whitespace-rule comment for a rule that now lives inside
  `_rollup_hierarchy` (`4208-4213`). Harmless duplication, but it is a second copy of a rule in a
  file whose whole thesis is that second copies are how the sibling defect survives.
* `rollup ->` restored to `rollup →` at `main.py:4244`; I verified the commit touches no other
  grepped string.
* `degrade.guard` is now called three times per stored exchange (`_tail_store_blocked`,
  `_async_tail`, `_rollup_hierarchy`). `writes_allowed()` is TTL-cached and only `guard`'s debug line
  repeats, so this is free. Noted, not a finding. Moving it inside also *improves* the queued-tail
  case: the disk is re-checked at the moment the rollup runs, not only when the tail was submitted.
* The new guard does **not** extend to the other two `maybe_rollup` callers (`main.py:6742`
  `/admin/compact`, `backfill.py:361`). Correct by policy — explicit admin writes are gated
  separately — but stated so a future reader does not read the new guard as universal.

### Areas checked with nothing found

* **Concurrency.** Re-derived rather than inherited: the two paths remain mutually exclusive per
  request, `_rollup_hierarchy` holds no lock when it calls `maybe_rollup`, and load -> mutate ->
  save all sit inside `conv_lock`.
* **Exception surface** of `_is_repeat_task_traffic` on its new path — see A.
* **Admin endpoints and log-grep contracts** — no change beyond the restored arrow.
* **Dangling references** to the deleted frozenset — none, worktree-wide.

## TEST RUNS

All in containers; the working tree was never written (`git status --porcelain` empty throughout).

**Unit, without `--saturation`** — `docker compose -f docker-compose.tests.yml run --rm unit-tests`

```
62 suite(s), interpreter python
61 passed, 0 failed, 1 skipped in 187s
  SKIP test_tokenizer_contract.py — http://localhost:18000/_fixture/info unreachable
EXIT=3
```

**Unit, with `--saturation`** — `... run --rm unit-tests --saturation`

```
64 suite(s), interpreter python
62 passed, 0 failed, 2 skipped in 212s
  SKIP test_soak_conversation.py    — http://localhost:18000 unreachable
  SKIP test_tokenizer_contract.py   — http://localhost:18000/_fixture/info unreachable
EXIT=3
```

Touched suites green in both: `test_degenerate_skip.py` 2.6s, `test_truncated_tail.py` 3.4s,
`test_rollup_on_skip.py` pass, `test_task_traffic.py` 1.0s.

**Soak** — containers cleared first, then
`docker compose -f docker-compose.tokenizer-contract.yml run --rm soak-tests`

```
All soak checks passed over 22 turns.
EXIT=0
```

7 checks ok, zero occurrences of error/exception/traceback/"never awaited" in 1202 lines of output.
It exercises the fix directly and shows it working:

```
... reply looks like a repetition loop ... — skipping memory tail (1402 chars; 11 consecutive memory-tail skip(s))
compactor INFO  redacted 10 degenerate historical turn(s) from rollup input (21 total)
compactor.summarizer INFO conv=soak_conversation: L1 rollup — chunk 1 covers turns 1-20
compactor INFO  conv=soak_conversation: rollup → L1=1 L2=0 L3=n last_turn=20
```

The watermark reaches 20 through a run of 14 consecutive skips — the exact failure `2f6facf` was
written for — with the degenerate turns redacted out of the input and the restored arrow in the log
line. For completeness: the soak's history carries assistant turns throughout, so
`_is_repeat_task_traffic` short-circuits on every turn and the soak is **not** evidence about the new
task-traffic gate. Case [3] is the only thing covering that, which is why B3 above asks for one more.

Then `docker compose -f docker-compose.tokenizer-contract.yml down -v` and the `zla-` containers
cleared again.

**Probes.** Two, both inside the unit image with the repo copied `/src` -> `/repo` and nothing
written outside the container: the `-W always` run quoted in N2, and a direct-interpreter run to
establish the venv path. No repo file was modified to demonstrate anything; the mutations in section
B are described and reasoned through, not applied.
