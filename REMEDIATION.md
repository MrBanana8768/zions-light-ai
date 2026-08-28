# REMEDIATION — v3.1

**Status:** action list, not yet implemented.
**Baseline:** `07ede2e`, branch `docs/frontend-spec`. Every `file:line` below was checked against that commit. If you are reading this against a later commit, re-anchor by symbol name, not line number.
**Target release:** v3.1 (minor bump). Release process unchanged: tag → build from tag → PR → owner merges. **No hot patches on production.**

---

## 1. Preface

### What this document is

This is the implementation list for the v3.1 remediation release. It is written to be executed by someone who was not present for the review — including someone who has never read the incident report. Everything you need to do the work is here: the defect, its location, the user-visible consequence, the change, how to prove the change worked, and what must land before what.

### Where the findings came from

Four passes, in order:

1. **A production incident, 2026-08-23/24.** A corrupted OpenWebUI message tree caused one conversation to send 7 of 241 messages to the model for roughly ten hours. Nothing detected it. The user believed she had lost months of conversation. Written up in [`INCIDENT_2026-08-24.md`](INCIDENT_2026-08-24.md).
2. **Two follow-up review passes** on the incident report itself. Each found live data-destruction paths the report had missed. That is the reason for pass four.
3. **Two independent full-codebase reviews** — one on correctness and data integrity, one on operational resilience — run without sight of each other.
4. **A reconciliation pass** that re-checked every `file:line` claim from both reviews against the tree, adjudicated their conflicts, killed eight load-bearing claims that were wrong, and added three findings neither reviewer produced.
5. **A second production incident, 2026-08-28**, after this document was already written. The token counter read 23% low on a live payload and 34–51% low on assistant content; the hard budget guard shed 60 of 65 turns and returned HTTP 200. Written up in [`INCIDENT_2026-08-28.md`](INCIDENT_2026-08-28.md). It produced **P0-0c**, **P0-0d**, **P0-0e**, and a correction to P0-0's stated root cause.

**`INCIDENT_2026-08-24.md` and `INCIDENT_2026-08-28.md` are the background. This document is the action list.** You do not need to read either report to implement this. Read them if you want to understand why the tone here is what it is.

**A note on pass 5, because it bears on how to read the rest.** P0-0 was written confidently, three days before the outage, and was *partly wrong* — right that the counter was broken, wrong about why, wrong about how the error scaled. A correct-sounding root cause cost hours during a live incident. Treat every diagnosis in this document as a hypothesis with a `file:line` attached, and check the line.

### What this supersedes

Any earlier informal list of "things to fix after the incident", including the remediation suggestions inside the incident report itself. Two of those suggestions were based on a false premise (see §1.4). Where this document and the incident report disagree, this document is correct.

### Severity scale

| | Meaning |
|---|---|
| **S1** | Destroys or permanently loses user data. Live on production today unless marked otherwise. |
| **S2** | Silent degradation, or loss of the ability to detect and recover from S1. The system keeps answering; it just stops remembering, or stops telling you it has stopped. |
| **S3** | Correctness or robustness defect with bounded blast radius. |
| **S4** | Hygiene. Costs nothing much to leave, costs little to fix. |

S2 is not "less bad than S1" in this system. The incident ran for ten hours because there is no outbound failure signal of any kind (F13). Everything in S1 is only as dangerous as it is undetectable.

### 1.4 One thing to know before you write a single test

Both reviews independently asserted that `Path.is_file()` and `Path.is_dir()` swallow `OSError`, and built S1 triggers on that. **That is false for `EIO`.** CPython's `pathlib` ignores only `ENOENT`, `ENOTDIR`, `EBADF` and `ELOOP`; everything else re-raises. Verified empirically on 3.13, and the code path is identical in 3.12, which is what `nvidia/cuda:13.0.0-runtime-ubuntu24.04` ships.

```
is_file(EIO)    -> RAISED  [Errno 5] Input/output error
is_file(ENOENT) -> False   (swallowed — correct and intended)
is_dir(EIO)     -> RAISED  [Errno 5]
```

The real trigger set for the whole `read_json` family at `compactor/memory.py:259-270`:

| condition | behaviour | live? |
|---|---|---|
| file absent (`ENOENT`) | returns the default | yes — by design |
| **file corrupt or truncated → `JSONDecodeError`** | **returns the default** | **yes — the dominant trigger** |
| **`EIO` during `open`/`read`** (metadata fine, chunk unavailable — MooseFS's characteristic failure) | **returns the default** | **yes** |
| `EIO` during `stat` (the `is_file()` at `:263`) | **raises** — does not return the default | no |

**Consequence:** the findings below survive, but the *trigger* is a corrupt file or a read-time `OSError`, not a mocked `stat`. **A test fixture that patches `Path.stat` to raise will reproduce none of these findings and will give you false confidence.** Write the fixture against a genuinely corrupt file on disk and an `OSError` raised from `open` / `json.load`. This is F32 and it is item 4 in the order for a reason.

### 1.5 The one-sentence version

> **F1a–e and G3 are a single function** — `memory.read_json`'s "return the default on any error" contract — **applied by five callers that then atomically write back.** It presents to the user as five unrelated memory-loss mysteries. Fixing that one primitive closes more S1 surface than everything else in this document combined.

---

## 2. Start here

### 2.1 If you do only three things

| | Item | Why this one | Cost |
|---|---|---|---|
| 1 | **F13(a)** — set `COMPACTOR_ALERT_WEBHOOK` in the pod template | The system currently has **zero** outbound failure signals. Backup failures, self-test failures and vLLM FATAL all go nowhere. `alert.py:35` returns `False` and everything downstream silently no-ops. This is a config line. It converts "the user notices" into "you get a message", which is the single mechanism that would have shortened a ten-hour incident. | **5 min** |
| 2 | **F32** — the corrupt-file / `OSError`-on-read test fixture | Every S1 finding in §3 depends on a code path that **no test in the repo exercises** — `grep` finds zero occurrences of `OSError` in any unit test. Landing this first converts F1a–e, F3, F5 and F10 from "arguments in a document" into failing tests you can watch go green. | **5 h** |
| 3 | **F1 + G2 + G3** — `read_json_strict`, abort-on-unreadable at all five write-back sites | Five separate wipe paths, one primitive. Facts, summaries, persona, `/remember`, and archive/restore. F1a re-fires on **every turn** because of G2. | **5 h** |

That is roughly **10 hours** and it closes the entire `read_json` family with regression coverage while giving the system its first working failure signal.

### 2.2 The free block — config only, no code, ~10 minutes total

These are the changes with the best ratio in the document. All are edits to `runpod.env.template` / `supervisord.conf` plus one one-line Python change. They still ship through the normal tag → build → PR flow; "config only" means low risk, not "push it to the pod by hand".

| Change | File | From | To |
|---|---|---|---|
| Enable alerting | `runpod.env.template:130` | `# COMPACTOR_ALERT_WEBHOOK=` | `COMPACTOR_ALERT_WEBHOOK=<url>` |
| Widen the recovery window | `runpod.env.template:85` | `COMPACTOR_BACKUP_RETAIN=7` | `COMPACTOR_BACKUP_RETAIN=28` (interim, until F7 lands retain-by-age) |
| Make a self-test failure abnormal | `supervisord.conf:163` | `exitcodes=0,1` | `exitcodes=0` |
| Put the operational log where the runbook says it is | `compactor/logsetup.py:60` | `logging.StreamHandler()` | `logging.StreamHandler(sys.stdout)` |

**On `RETAIN=7`:** with `COMPACTOR_BACKUP_INTERVAL_HOURS=6` (`runpod.env.template:47`) that is **42 hours of total backup history**. Tightening the RPO after the 2026-08-10 incident without raising `RETAIN` shortened the recovery window from seven days to under two. The 2026-08-24 incident took days to analyse. `28` restores roughly a week; F7 replaces the whole scheme.

**On the logging one-liner:** `logging.StreamHandler()` defaults to **stderr**, so under supervisord every `conv_id=… source=… msgs=…`, `injected memory […]` and `compacted: …` line lands in `compactor-error.log`, while `compactor.log` holds uvicorn access lines. `OPERATIONS.md:44` sends the operator to the wrong file. One character of change makes the runbook true.

### 2.3 One command to run on the pod before you plan anything

```sh
sqlite3 /data/openwebui/compactor/chromadb/chroma.sqlite3 'PRAGMA journal_mode;'
```

If it returns `wal`, **F31 is live** and the episodic store is running the exact configuration identified as tearing `webui.db` on 2026-08-10, on the same MooseFS volume. That answer changes the shape of the release (see §4). It cannot be determined from the repo. Run it first.

---

## 3. Remediation items

Ids are carried over from the reconciled finding set so this document, the review output and any future notes stay cross-referenceable. `D…` ids were known before the final sweep; `F…` and `G…` came out of it. **AG** marks a finding both independent reviewers produced — the strongest signal in the exercise.

---

### Phase 0 — Now

Ships in v3.1 like everything else, but these are the first commits, they are near-zero risk, and later items assume they are in place.

---

#### P0-0 · F60 — The accurate token counter has never run in production
**S1.** `compactor/requirements.txt`, `main.py:296-302`, `main.py:131`

**Found 2026-08-27, in production, three days after the incident.** Not from review — from a user hitting repeated vLLM 400s.

`count_tokens` (`main.py:289`) has three tiers. Tier 1 applies the model's chat template and is exact. Tier 2 catches any exception and falls back to `len(tok.encode(text)) + 4` per message. Tier 3 is `char/4`. **Tier 1 has never executed.** Two independent causes, either sufficient alone, both silent:

1. **jinja2 was not installed.** `apply_chat_template` renders a Jinja template; jinja2 is an *optional* transformers dependency. `compactor/requirements.txt` pinned `transformers==5.12.1` with the comment *"transformers is used tokenizer-only… Keeps the compactor venv lean."* That decision — correct in its own terms, isolating the compactor from vLLM's torch pins — removed chat templating without anything saying so. So tier 1 could not work for **any** model, ever.
2. **The served model carries no chat template.** `coder3101/Cydonia-24B-v4.3-vision-heretic` has no `chat_template.jinja` — confirmed by a `.no_exist/` marker in the HF cache. The text-only variant it replaced *does* have one. `apply_chat_template` raises `ValueError: tokenizer.chat_template is not set` before jinja2 is reached, so cause 2 masked cause 1 during diagnosis. The tokenizer loads as `TokenizersBackend` converted from `tekken.json` — Mistral's native format — so vLLM is applying its template through mistral-common, entirely outside HF's `chat_template` attribute.

**Consequence.** Every token count the system has ever produced omits chat-template framing — roughly **22 tokens per message** on Mistral, scaling with *message count*. Observed live: a ~200-message conversation counted ~5,250 tokens light against a 32,768 window, producing repeated 400s.

> **CORRECTED 2026-08-28 — framing is the minor term.** The paragraph above was written before the outage in [INCIDENT_2026-08-28.md](INCIDENT_2026-08-28.md) and its scaling law is **backwards for the case that actually took production down**. Framing overhead is real, but it cannot produce the errors since measured:
>
> ```
> live payload:      local=122106  vLLM=150050   scale=1.23x
> one decorative rule: local=15      vLLM=809      scale=53.93x
> ```
>
> The dominant term is **vocabulary mismatch on content**. The HF-converted `tekken.json` prices box-drawing characters and emoji far below `mistral_common`, which is what vLLM actually charges. Measured per role: system reads high, user reads ~9% high, **assistant reads 34–51% low**. One assistant reply held 1,710 × U+2501 and 441 × U+2500 — about **4,275 tokens, 13% of the window, in decoration**.
>
> So the error scales with **content**, not message count, and it is worst on exactly what the model generates. An implementer optimising for per-message framing would have fixed nothing. Both terms are real; only one caused the outage.
>
> **Neither is worth estimating.** See **P0-0c** — the fix is to stop estimating.

**Why it is S1 and not S2.** It is the input to every budget decision in the system: the hard-budget guard, the compaction trigger, and the shedding arithmetic all consume this number. The v3.0.5 calibration loop exists to correct it and cannot — see P0-0b.

**Fixed in this commit:** `jinja2==3.1.6` added to `compactor/requirements.txt`, and a `RUN` guard added to the `Dockerfile` that fails the build if it is absent. **Necessary, not sufficient** — cause 2 remains open.

**Still to do.**
- Supply a chat template for the vision model. The sibling `Cydonia-24B-v4.3-heretic-v4` snapshot carries one at `/data/models/hub/models--coder3101--Cydonia-24B-v4.3-heretic-v4/snapshots/*/chat_template.jinja`; both are v4.3, so it is very likely correct — **verify against vLLM's `/tokenize` endpoint before trusting it.** Vendor it into the repo and pass it explicitly as `chat_template=`; do not depend on a sibling model staying in the cache.
- **Log the tier-2 fallback.** `main.py:298` is a bare `except Exception:` with no log statement. Tier 3 at least warns (`main.py:131`). The tier that actually ran says nothing — a total loss of accuracy presenting as normal operation, in the component whose only job is to know how big things are. Log once per process at WARNING with the exception type.
- **Assert tier 1 in the boot self-test** against the real `MODEL_REPO`, not just at build time. A build guard cannot see the served model.
- Consider `/tokenize` on the vLLM server as ground truth for calibration. It is the exact number the budget guard is trying to estimate, and the calibration loop has been groping toward it by absorbing 400s.

**Verify.** On the pod, after deploying:
```bash
/opt/compactor-venv/bin/python -c "
import os; from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained(os.environ['MODEL_REPO'])
m=[{'role':'user','content':'hi'},{'role':'assistant','content':'hello'}]
print(len(t.encode(t.apply_chat_template(m, tokenize=False, add_generation_prompt=True))))"
```
Must print a number. A `ValueError` or `ImportError` means tier 1 is still dead.

**Interim mitigation in force:** `COMPACTOR_GENERATION_RESERVE=8192` (from 2048), dropping `HARD_INPUT_LIMIT` to 24,576. Costs ~6k of usable window and buys headroom the counter cannot currently provide. **Revert to 2048 only after tier 1 is verified working on the pod.**

---

#### P0-0b · C4 — The calibration loop cannot converge, observed live
**S2.** `main.py:217-221`

`overshoot = actual - HARD_INPUT_LIMIT` measures against the *original* limit, not the already-tightened one (`main.py:711` applies `limit - _BUDGET_MARGIN`). So the computed overshoot understates the true undercount by exactly `_BUDGET_MARGIN`, and the `if new_margin > _BUDGET_MARGIN` guard then refuses to advance unless the undercount roughly doubles.

Observed on 2026-08-27 across three consecutive failures on one conversation:

| Time | vLLM counted | Margin set |
|---|---|---|
| 05:53:46 | 32,836 | 2,628 |
| 05:54:20 | 32,963 | 2,755 |
| 06:00:30 | 33,090 | 2,882 |

The margin advanced by exactly 127 each time — the conversation's own growth per turn — while the payload never shrank. It needed ~5,250 and was crawling there at 127/failure, i.e. **~19 more broken messages** for the user. It is a loop, not a retry.

**Fix.** `overshoot = actual - (HARD_INPUT_LIMIT - _BUDGET_MARGIN)`. Rewrite `test_modality.py:261-268` first — it currently asserts the defective behaviour as correct, so the code change will fail a green test until the test is fixed.

**Also.** `_BUDGET_MARGIN` is a module global (`main.py:194`), so every learned correction is lost on restart. Confirmed live: a pod recreate mid-diagnosis reset it to 0 and the climb started over. Persist it per conversation, or accept that the first long conversation after every restart eats a 400 — and say so in the log if so.

**Ordering: P0-0 before P0-0b.** Fixing the arithmetic while the counter is blind only makes a wrong number converge faster.

> **SUPERSEDED IN PART 2026-08-28.** With P0-0c shipped, the guard measures the payload exactly rather than estimating it, so there is nothing left for the loop to calibrate *away*. Fix the arithmetic anyway — the loop is still the fallback path when `/tokenize` is unreachable — but treat `_BUDGET_MARGIN` as a degraded-mode mechanism, not the primary one, and demote its severity accordingly.

---

#### P0-0c · Budget against measured tokens, not estimated ones
**S1.** `main.py` (`count_tokens_exact`, `_enforce_hard_budget`, `_chunk_to_budget`)

**Root cause of [INCIDENT_2026-08-28.md](INCIDENT_2026-08-28.md). Shipped as a production hotfix; on `fix/v3.1-remediation`.**

The compactor decided what the user would be allowed to say to the model using an *estimate* of a quantity the enforcing server computes exactly, on request, over localhost, in single-digit milliseconds. `/tokenize` was running the whole time.

The rule, stated generally because it outlives this bug:

> **A component that budgets against a limit must verify its arithmetic against the authority that enforces that limit.**

Three properties made the violation undetectable, and all three recur elsewhere in this codebase:

1. **Shared oracle.** Guard, compaction trigger, and summarizer all called `count_tokens`. Disagreement would have been a signal; agreement was worth nothing, and agreement is what monitoring saw.
2. **Content-tracking error.** Worst on long decorated assistant replies, so it looked like "long conversations are hard."
3. **Fluent degradation.** Every fallback downstream was graceful. Composed, they produced a system that lied — see **P0-0d**.

**Shipped.** `count_tokens_exact(messages)` posts to `{VLLM_URL}/tokenize`, short connect timeout, once-per-process warning, returns `None` rather than raising. Wired into the guard (real payload + derived `scale` for per-message arithmetic), the guard's verify step (re-measure, do not trust the scaled estimate), and the summarizer (measure before `_chunk_to_budget`).

Verified against the live conversation:

```
guard:       150050 -> 15155  (limit 16384)  FITS
summarizer:  NEW (measured)  5 batches, largest=29909  FITS
             OLD (scale 1.0) 4 batches, largest=36280  OVERFLOWS -> 400
```

**The summarizer half is the decisive one.** The guard was the visible symptom; it only ever ran on the full conversation because summarization was failing first.

**Still to do.**
- **A test that would have caught this.** Budget a known payload, assert the local count matches `/tokenize` within a stated tolerance — **against tokenizer-hostile content** (box-drawing, emoji, CJK), not prose. Prose is where the two tokenizers agree; a test written on prose passes and proves nothing.
- **Fail loudly when `/tokenize` is unreachable at boot.** Today it degrades to the estimate silently after one warning. The self-test should assert the endpoint answers, so "we are flying on the estimate" is a startup fact rather than a log line from three days ago.
- **Cache per-message counts.** One `/tokenize` per request is cheap; per-message is not. The `scale` factor exists to avoid that and is an approximation — bound it, and re-measure exactly at the verify step (already done).

**Do not touch `GENERATION_RESERVE` on the strength of the guard table.** That table measures *input*; the reserve exists for *output*. This user's assistant replies measure **7,513–11,347 tokens**, so the 2,048 the table appears to recommend would truncate nearly every reply mid-sentence — trading a context bug for a worse one. Keep **16,384**, or 12,288 if headroom is genuinely needed. *A headroom parameter cannot be tuned from a table that does not contain the thing it reserves for.*

---

#### P0-0d · Fluent degradation — three fallbacks that cannot say they fired
**S1.** `main.py` (`_enforce_hard_budget`, compaction failure path), `summarizer.py`

Ranked S1 because this is what converted a loud, immediately-diagnosable 400 into a silent week-long context collapse. The counter bug was the cause; **this is why nobody found out.**

Three paths, one shape:

| Path | What it does on failure | What the user sees |
|---|---|---|
| `_enforce_hard_budget` | sheds oldest turns until the payload fits | HTTP 200, fluent reply, **4 of 65 messages** |
| compaction on summarizer 400 | forwards the original messages | nothing — and the full conversation now hits the guard on *every* request |
| `count_tokens` tier fallback | silently drops to a worse estimator | one WARNING, once per process (P0-0) |

**Fix.**
- **Guard shedding is an ERROR when it discards conversational turns**, not a WARNING, and the message must state the ratio — `kept=4 of 65` — not just the token totals. A log line that reports only tokens reads like successful housekeeping. This is already partly done on `fix/v3.1-remediation`; finish it and check the phrasing.
- **Compaction failure must be visible in the response**, not only in a log. `FRONTEND_SPEC.md` §15 already lists the budget-shed signal as **required**; this is the compactor half of the same obligation (§2.7). The spec's `context_shed` notice deliberately distinguishes *discarded* from *compressed* — honour that distinction in the signal.
- **Emit the ratio in the per-request log line** unconditionally, so the healthy case is legible and the unhealthy case is a diff rather than an inference.

**The general rule:** *a fallback that cannot signal that it fired is not a fallback; it is a cover-up.*

---

#### P0-0e · Deploy hygiene — a patch must be verified against the commit it lands on
**S2.** CI, `OPERATIONS.md`

Problem B of [INCIDENT_2026-08-28.md](INCIDENT_2026-08-28.md). The hotfix failed on start:

```
ImportError: cannot import name 'StoreUnreadable' from 'memory'
```

A single `main.py` was lifted from `fix/v3.1-remediation` and dropped onto a pod running v3.0.5. The symbol exists only in the branch's `memory.py`; the dependency did not travel with the file. Production stayed degraded through the rollback.

**The same class of error recurred five times this week** — a symbol referenced but not imported, or not present in the module it landed in (`memory.StoreUnreadable`, `assert_in`, an unimported `StoreUnreadable` in `commands.py`, and two variants). Every one dies on import.

**Fix — two lines and a rule.**

1. **Import smoke test in CI and in the pre-deploy path.** `selftest.py` runs *inside* an already-working process, so by construction it cannot catch a module that fails to import at all. Nothing currently covers this.
   ```bash
   python -c "import compactor.main, compactor.memory, compactor.commands"
   ```
2. **Build hot patches from the deployed tag**, in a worktree, changing only what the fix requires — never by copying a file out of a feature branch. Record this in `OPERATIONS.md` beside the standing "production runs tagged images" rule, since the hot-patch path is exactly where that rule gets relaxed and therefore exactly where it needs written guidance.
3. **Staged files must be named `.py`.** `spec_from_file_location` returned `None` on `main.py.new` and produced `AttributeError: 'NoneType' object has no attribute 'loader'` — a confusing second failure on top of the first.

---

#### P0-1 · F13(a) — There is no outbound failure signal of any kind
**S2.** `runpod.env.template:130`, `alert.py:35`, `health.py:230`, `supervisord.conf:163,199-201`, `backup.py:336,361`

**Defect.** Six independent notification paths, all verified inert:

1. `COMPACTOR_ALERT_WEBHOOK` is commented out → `alert.py:35` returns `False` → backup failure (`backup.py:336,361`) and self-test failure go nowhere.
2. `health.py:230` — `status_to_http_code` returns 503 only for `"down"`. vLLM FATAL and disk-pressure write-pause both map to `"degraded"` → **HTTP 200**.
3. RunPod does not act on Docker `HEALTHCHECK`.
4. `supervisord.conf:163` — `exitcodes=0,1` makes a self-test **FAIL** an *expected* exit. `supervisorctl status` shows `EXITED` identically for pass and fail.
5. `supervisord.conf:199-201` — `[eventlistener:processes]` subscribes to `events=PROCESS_STATE` and its command is `while read line; do echo OK; done`. A **null sink**. It looks like monitoring and is not.
6. Nothing checks backup freshness. `latest_backup_info` is *reported* in `/health/full` but never affects `status`. A backup daemon dead for three weeks reports a three-week-old "latest" with `"status": "ok"`.

**Consequence for the user.** Every failure in this system is detected by the user noticing that the model has forgotten her. That is the direct cause of the ~10-hour detection time in the incident.

**Change.** Set `COMPACTOR_ALERT_WEBHOOK`. Set `exitcodes=0`. Make the eventlistener POST `FATAL` and `BACKOFF` transitions to the same webhook rather than swallowing them. Make backup staleness (`newest archive older than 2 × INTERVAL_HOURS`) set `status: "degraded"` **and** fire `_alert_failure`.

**Verify.** `supervisorctl stop vllm` → a message arrives. Rename the backup directory aside, wait one interval → a message arrives. Force a self-test failure → `supervisorctl status selftest` shows `FATAL`, not `EXITED`.

**Estimate.** 5 min for the webhook; **3 h** for the rest. **Depends on:** nothing.

---

#### P0-2 · F17(a) — The operational log is in the file named "error"
**S3.** `compactor/logsetup.py:60`

**Defect.** `handler = logging.StreamHandler()` → stderr. Confirmed at `:60`; the file is 65 lines total (an earlier citation of `:149` was wrong).

**Consequence.** An operator following `OPERATIONS.md:44` during an incident reads the wrong file and sees uvicorn access lines while the diagnostic trail sits in `compactor-error.log`. The 2026-08-24 investigation turned on two adjacent log lines.

**Change.** `logging.StreamHandler(sys.stdout)`.

**Verify.** `tail -f /var/log/supervisor/compactor.log` during one chat request shows `conv_id=… source=… msgs=…`.

**Estimate.** 5 min. **Depends on:** nothing.

---

#### P0-2b · F61 — Log sweep: silent exception handlers, and a health check that cannot see the corruption it exists to catch
**S2.** Whole compactor package. Inventory taken 2026-08-27.

Every failure this project has spent a week diagnosing had the same shape: **a degraded mode indistinguishable from a healthy one.** P0-0 is the extreme case — the token counter lost its accuracy for months behind a bare `except Exception:` with no log statement. It is not one bad handler, it is a pattern, so this item sweeps them all rather than fixing them one incident at a time.

**Method.** For every `except` block in `compactor/*.py` excluding tests, check whether its body contains a `logger.` call, a `raise`, an alert, or a `print()` to stderr. They fall into three classes and only the first needs code changes.

**The count is a moving target — re-run it, do not quote it.**

| Commit | Silent handlers | Note |
|---|---|---|
| `f106305` | 46 | the original inventory, 2026-08-27 |
| `0a169b8` | **23** | half of them closed as a side effect of the V2–V8 destructive-path work |
| `0a169b8` + Phase-1 working tree | **25** | two new handlers, both reviewed and defensible: one returns to a caller that surfaces, one is a per-row skip |

*A note on the original figure, so nobody re-derives it:* an ad-hoc scan reported 47 because it did not treat `print()` as reporting. The one handler separating 47 from 46 is `backup.py:488`, the CLI restore entry point, which prints to stderr and returns 1 — that is surfacing the failure, so 46 was correct at `f106305`.

*And a note on the drop:* 46 → 23 is real, not a change in method — the same script, run at both commits. It happened because fixing the destructive paths meant giving their handlers something to say. **This is the shape to expect: the sweep is a symptom counter, not a work queue.** Re-run `log-sweep.py --count` before planning against it; the remaining handlers are the ones no other item happened to touch, which makes them harder, not easier.

**Class A — the failure vanishes. Fix these.**

| File:line | Handler returns | What is lost |
|---|---|---|
| `main.py:298` | per-message `encode()+4` | **P0-0.** Total loss of token-counting accuracy. Already tracked |
| `health.py:134` | `pass` | **The most important one here.** `/health/full` sums `load_facts(cid)` across conversations. An unreadable facts file — the exact F1/D33 scenario — is silently skipped, so `facts_total` simply reads lower while `status` stays `"ok"`. **The health endpoint cannot detect the corruption that would destroy memory.** It is the monitoring blind spot that pairs with the S1 |
| `health.py:138` | `pass` | Same, for `conversation_doc_count` |
| `health.py:146` | `pass` | Same, for summary state — a conversation with an unreadable summary file just stops being counted |
| `retrieval.py:257` | `return 0` | ChromaDB unavailable is **indistinguishable from "no exchanges indexed"**. Compounds `health.py:138`: a dead vector store reports as an empty one |
| `main.py:1669,1676,1681,1691` | `None` / empty dict | `/admin/conversations/{id}` reports `facts.count = None` and `episodic = None` on any read error. Reads as "this conversation has no memory" to anyone inspecting it — including during an incident |
| `bgwork.py:62` | `pass` | A background task dying leaves no trace |
| `commands.py:196,201` | `pass` | Slash-command side effects failing silently |
| `main.py:830` | bare `return` | Async tail aborting with no record |
| `main.py:1570` | `pass` | — |
| `backup.py:370` | `pass` | `_alert_failure` swallowing its own failure. **The alert about a failure can fail silently.** Combined with P0-1 (no webhook configured at all), alerting is silent twice over |
| `degrade.py:68` | `float("inf")` | Documented fail-open, and correct — but a permanently broken `disk_usage` makes the disk-pressure guard pass forever with nothing said. Log once per process, do not change the behaviour |
| `backup.py:100` | `float("inf")` | Same shape, on the backup staleness check |

**Class B — legitimate cleanup, leave the behaviour, add a DEBUG line.** `memory.py:247` (directory `fsync` after a successful write), `memory.py:254` (orphan temp-file cleanup that deliberately does not shadow the original `raise`), `backup.py:332,357` (deleting an unverifiable archive — the real error *is* logged and alerted), `selftest.py:197,225,334,480` (post-test cleanup, though see D10: this is one source of store pollution).

**Class C — no change. The error is already propagated by return value** and surfaces to a caller: `selftest.py:123,133,189,276` return `(False, detail)`; `health.py:72,111,184,191,206` return error dicts that appear in `/health/full`; `backup.py:227,235,252,263` return `(False, msg)` from the verify path.

**The rule to adopt, and to enforce in review.** A handler may swallow an exception only if it does exactly one of: logs it, re-raises it, or returns it to a caller that surfaces it. **"Returns a plausible-looking default" is none of those three.** That single rule is what P0-0 violated for months, and what `health.py:134` violates today in the one endpoint whose purpose is to notice.

**Also worth fixing while in here.** Class-A handlers that fire on every request should log **once per process**, not per call, or the fix becomes its own denial of service. A module-level `_warned: set[str]` keyed by call site is enough.

**Concrete change.** ~13 handlers in Class A get a `logger.warning` with the exception type and enough context to identify the conversation; ~8 in Class B get `logger.debug`. No control flow changes anywhere except `retrieval.py:257`, which must distinguish "unavailable" from "zero" — return `None` and let callers render it as `unknown` rather than `0`.

**Verify.**
```bash
# 1. No Class-A handler is silent any more. Re-run the sweep; expect only Class B/C.
python3 log-sweep.py                 # full listing, triage by hand
python3 log-sweep.py --count         # re-run; 46 at f106305, 23 at 0a169b8 (see the table above)

# 2. The health endpoint reports corruption instead of hiding it:
mv /data/openwebui/compactor/facts/<some_conv>.json{,.bak}
printf 'not json' > /data/openwebui/compactor/facts/<some_conv>.json
curl -s localhost:8080/health/full | grep -iE 'status|unreadable|error'
#    expect: a non-ok status or an explicit unreadable count — NOT a silently smaller facts_total
mv /data/openwebui/compactor/facts/<some_conv>.json{.bak,}

# 3. A dead vector store is distinguishable from an empty one:
curl -s localhost:8080/admin/conversations/<conv> | grep episodic
#    expect: null/"unknown" when Chroma is down, 0 only when genuinely empty
```

**Depends on:** nothing. **Blocks:** nothing. Do it in one sitting alongside P0-1 — the alert webhook is useless if the code never says anything worth alerting about.

---

#### P0-3 · F15 — Episodic retrieval is disabled entirely for any short or windowed client array
**S2, and it was live during the incident.** `compactor/main.py:1358`, `compactor/retrieval.py:219`

**Defect.**
```python
recent_cutoff = max(0, turn_index - (KEEP_RECENT_TURNS * 2))   # turn_index = len(messages)+1
hits = retrieval.retrieve(conv_id, last_user_text, exclude_turns_from=recent_cutoff)
```
With `COMPACTOR_KEEP_RECENT_TURNS=4` (`runpod.env.template:56`), **any request carrying ≤ 7 messages yields `recent_cutoff = 0`.** `retrieval.py:219` then evaluates `if exclude_turns_from is not None and turn_index >= exclude_turns_from: continue` — `0 is not None` is `True`, and every stored `turn_index >= 0`, so **every hit is discarded**. Not narrowed. Off. The only rows that survive are ones whose metadata is missing (`turn_index` defaults to `-1`).

**Consequence.** During the incident — 7 of 241 messages — episodic recall was not merely degraded by D21, it was **completely disabled**, and no log line said so (`log_parts` simply omits the `Nretr` entry). This also fires permanently the moment the replacement front end sends a bounded window, which is the committed direction.

**Change.** Treat `exclude_turns_from <= 0` as "no exclusion". One conditional. The full fix — deriving the cutoff from stored turn identity rather than the client's array length — lands with D1 and is deferred.

**Verify.** Unit test: index three exchanges, call `retrieve(..., exclude_turns_from=0)`, assert three hits. Then a live 2-message request against a conversation with history shows `Nretr=…` in the log.

**Estimate.** 45 min. **Depends on:** nothing. **Full fix depends on:** D1.

---

#### P0-4 · F31(verify) — Run the ChromaDB journal-mode check
**Unknown severity until run.** `compactor/retrieval.py:90`

`chromadb.PersistentClient(path=CHROMA_PATH)` opens `/data/openwebui/compactor/chromadb/chroma.sqlite3` on MooseFS, and **nothing in the repo configures its journal mode**. `DATABASE_ENABLE_SQLITE_WAL=false` (`Dockerfile:337`, `runpod.env.template:105`) is an **OpenWebUI** variable — confirmed to have no effect on ChromaDB. The Dockerfile comment at `:328-336` states that WAL on a network volume is what tore `webui.db` on 2026-08-10.

Run the command in §2.3. Record the answer in this file. If `wal`, schedule F31(fix) — see §3 Deferred.

**Estimate.** 5 min. **Depends on:** pod access.

---

### Phase 1 — v3.1 core

Must ship in the release. Roughly **30 h**.

---

#### V1 · F32 — The test suite certifies the defects; the corrupt-file fixture does not exist
**S2 (meta).** `compactor/test_facts.py:62,141`, `test_summarizer.py:134`, `test_dedup.py:159`, `tests/chaos/run_chaos.py:94-113`

**Defect.** All citations verified:

- `test_facts.py:62` asserts F1a's return-empty-on-corrupt behaviour **as correct**.
- `test_summarizer.py:134` asserts F1b's lossy filter as correct, and never checks what the subsequent `save_state` persists.
- `test_facts.py:141` `test_prune_facts_lru_eviction` constructs `last_used` values of 100/500/999 — a state the live pipeline can never produce (see F9). It passes; the feature it tests does not exist.
- `test_dedup.py:159` covers `"KEEP."` and `"Keep — they are different"`, both **prefix** forms. It does not cover `"These are different, KEEP"` (D39). Its existence makes the parser look hardened.
- `tests/chaos/run_chaos.py:94-113` `scenario_corrupt_facts` writes garbage and asserts only that chat returns 200. It does **not** assert the pre-existing facts survived — and under F1a they do not. The chaos suite confirms that the system degrades gracefully *into destroying data*, and passes. `scenario_chromadb_unwritable` has the same shape.
- Coverage gaps, verified by grep: `_async_tail` has **no unit test in any `compactor/test_*.py`**. `_clear_all_memory` is mocked out entirely (`test_commands.py:247`). `_run_backfill` over a pre-existing facts file is untested. **Zero occurrences of `OSError` in any unit test.**
- Correction to an earlier claim: `test_backup.py:139` `test_backup_without_db_succeeds` seeds `with_db=False` while the **store is present**, so it certifies a missing DB, not a missing store. It is not F2 written down as a requirement. The real gap is the absence of a `test_backup_without_store_fails`.

**Change.** Add a `flaky_fs` fixture that injects, per §1.4, **(a)** a genuinely corrupt file on disk and **(b)** an `OSError` raised from `open` / `json.load` — **not** from `stat`. Run it against `load_facts`, `load_state`, `load_persona`, `load_archive`, `_run_backfill`, `create_backup` and `_clear_all_memory`. **Every one of those should fail loudly against today's code.** If a case passes today, you have written the fixture wrong — go back to §1.4. Then add post-condition assertions to every chaos scenario: after `scenario_corrupt_facts`, the pre-existing facts must still be on disk.

**Verify.** Seven new failing tests on `07ede2e`; all green after V2.

**Estimate.** 5 h. **Depends on:** nothing. **Must land before:** V2.

---

#### V2 · F1a–e + G2 + G3 — `read_json` returns the default on any error, and five callers write back
**S1, live.** `compactor/memory.py:259-270`

**Defect.** `read_json` cannot distinguish absent / corrupt / unreadable. Five callers treat all three as "empty" and then atomically overwrite.

| site | path | what is destroyed |
|---|---|---|
| **F1a** (= D33) facts | `facts.py:104-119` → `main.py:935-1000`: `combined = _merge_touched(facts.load_facts(conv_id), touched_facts) + new_entries` then `facts.save_facts(conv_id, kept)` | the entire fact store — and it **re-fires on every turn**, see G2 |
| **F1b** (= AG1) summaries | `summarizer.py:106` → `:394` → `:421` | the entire L1/L2/L3 hierarchy, replaced by a summary of the client's current window |
| **F1c** (= D8) persona | `persona.py:86` → `auto_capture_persona` → `save_persona` | the stored persona — **on the request path, before vLLM is even called** |
| **F1d** `/remember` | `commands.py:140` → `:148` | the entire fact store, replaced by the one fact the user just typed; reports `Facts now: 1`, which is accurate and completely misleading |
| **F1e** archive/restore | `facts.py:210-212` and `facts.py:255-257` | the cold-storage archive. `restore_from_archive` loses **both halves in one call** |

F1b is worse than F1a: facts at least pass through `_merge_touched`; summaries are replaced wholesale.

**G3 (verified, new).** `maybe_rollup` runs `save_state(conv_id, state)` at `summarizer.py:421` **outside** the `try` that catches rollup failure at `:416` — it has its own narrow try that guards only the write. So a corrupt load followed by an LLM failure writes the empty skeleton over the real file with **zero LLM involvement**. The "reset to a summary of 7 messages" description understates it; the file can be reduced to `_empty_state`.

**G2 (verified, new).** `main.py:~995-1000` has **no early return on empty `new_strs`**. `combined` can be `[]`, `prune_facts([])` returns `([], 0)`, and `save_facts(conv_id, [])` runs. This — not `_clear_all_memory` — is the primary generator of D10's empty-facts-file growth: every conversation the compactor has ever touched, including every `__selftest_oneshot_*`, acquires a file that `list_known_conv_ids` counts forever. It also means F1a's corrupt-read wipe re-fires on *every turn* rather than occasionally.

**Consequence.** The model abruptly forgets months of established detail, mid-conversation. The log line reads like success.

**Change.**
1. Add `read_json_strict(path) -> (status, data)` with `PRESENT | ABSENT | UNREADABLE`. Keep `read_json` for genuinely best-effort readers.
2. Every read-modify-write caller above **aborts the write** on `UNREADABLE` and calls `_alert_failure`.
3. Move `save_state` inside the rollup `try` at `summarizer.py:416`.
4. Make `load_state`'s filtering non-destructive — `summarizer.py:111-116` silently drops entries failing `_is_chunk` and the next save persists the filtered list. Round-trip unrecognised entries verbatim.
5. Drop the unconditional empty-facts write (G2): early-return from the tail when there is nothing to add.
6. `auto_capture_persona` must never overwrite a record whose `source` is `"admin"` or `"inherited"` without an explicit flag.
7. In `archive_stale_facts`, write the archive **before** trimming the active set, so a crash duplicates rather than loses.

**Verify.** V1's fixture goes green. Then, on a scratch conversation: `printf 'x' > facts/<conv>.json`, send a turn, confirm the file is **unchanged**, an alert fired, and the response still succeeded.

**Estimate.** 5 h. **Depends on:** V1.

---

#### V3 · F2 + F7 — An empty backup verifies OK, publishes, and prunes the real archives
**S1, live.** *(AG5)* `compactor/backup.py:159-164,167,177-191,212-266,275-286,344,448-452`

**Defect, F2.** `create_backup` writes the compactor store only under `if STORAGE_ROOT.is_dir():`. The else branch records `{"present": False}` and **does not even log** — contrast `:184`, which does warn for a missing `webui.db`. `verify_backup` never cross-checks the manifest: `db_expected` is read from the manifest the same run just wrote, and the compactor loop is `if store.is_dir():`, so an archive containing nothing but `manifest.json` returns `(True, "db=absent, 0 json file(s) parsed")`. `run_once` publishes it, logs `backup ok`, fires **no** alert, and calls `prune_old_backups`, which unlinks everything past `RETAIN=7` unconditionally (D9).

**Trigger, corrected.** Not a MooseFS blip — per §1.4 that raises. The real triggers are `ENOENT`: an unmounted `/data`, a lost network volume, or a `COMPACTOR_STORAGE_ROOT` typo. Those are precisely the conditions under which you will need the backup.

**Also verified:** `chroma.sqlite3` is copied by `shutil.copytree` at `:189` — not routed through `_snapshot_sqlite`, and never integrity-checked (`:259` is `rglob("*.json")` only). The episodic store is copied live, non-atomically, and unverified in **every** archive. Note that `PRAGMA integrity_check` validates SQLite pages, not OpenWebUI's parent-pointer chain — **the 2026-08-24 corruption would verify green.**

**Defect, F7.** `run_daemon` (`:448-452`) calls `run_once()` **immediately at process start**, then sleeps. Every cycle ends in an unconditional prune. Seven container restarts, pod recreates, redeploys or OOM-kills and every pre-incident archive is gone, replaced by seven copies of the damaged state. `/admin/backups` shows seven healthy recent archives with nothing indicating they are all younger than the damage.

**Consequence.** Total unrecoverable loss, discovered only at restore time.

**Change.**
- `verify_backup` asserts **against the manifest**: if `manifest["sources"]["compactor"]["present"]` is true, require the directory and require `json_checked >= manifest[...]["files"]`.
- `create_backup` **raises** rather than recording `present: False` when `STORAGE_ROOT` is configured but missing.
- Refuse to publish an archive whose payload is under 50 % of the previous one; `_alert_failure` on it.
- Route `chroma.sqlite3` through `_snapshot_sqlite` and add its `integrity_check` to verification.
- Record per-conversation fact / summary / episodic counts in the manifest, and compare against the previous archive's manifest.
- Retain by **age** (14 d) plus a GFS tier (one per day for 14 d, one per week for 8 w). **Never prune below 3 archives.**
- Suppress the boot-time run when the newest archive is younger than `INTERVAL/2`.

**Verify.** Point `COMPACTOR_STORAGE_ROOT` at a nonexistent path, run one cycle: it must fail, alert, and **not** prune. Restart the container ten times in a row: the oldest archive must survive. `tar tzf` the newest archive and confirm `compactor/` is populated and `chroma.sqlite3` present.

**Estimate.** 6 h. **Depends on:** P0-1 (alerts need somewhere to go). **Interacts with:** F31 — if the chroma store moves to the container disk, the snapshot work here changes shape. Decide F31 first.

---

#### V4 · F8 + F10 + F24 + D6 — Episodic memory latches off, `/forget` lies, the self-test cannot fail
**S2.** *(AG2)* `compactor/retrieval.py:68-106,109`, `health.py:176`, `selftest.py:369-394`, `main.py:1725-1731`, `commands.py:168-181`

**Defect, F8.** `_available` is set `False` at `retrieval.py:105` and latched by `if _available is not None: return _available` at `:74`. `_try_init()` wraps `TextEmbedding(...)` **and** `chromadb.PersistentClient(...)` in one `except Exception`. One failure at first use — a stale `.lock`, an `EIO` opening `chroma.sqlite3`, an unclean-shutdown recovery, an ONNX import failure after a dependency bump — and `retrieve()` returns `[]` and `index_exchange()` returns `False` **for the rest of the process's life**. Every exchange from that moment is never indexed, and there is no backfill for the gap: it is permanent even after a restart.

`is_available()` at `:109` documents itself as wired to `/health/full` and the self-test. It has **zero callers anywhere in the repo, including tests.** `health.py:176` reports `vllm` and `storage` only. `conversation_doc_count` returns `0` when unavailable, so `/health/full` prints `indexed_exchanges_total: 0` alongside `"status": "ok"`. `main.py:947-950` logs only on success, so the silent path emits nothing at all.

**Defect, F10.** `main.py:1725-1731` / `commands.py:168-181`: `if n_facts > 0: facts.save_facts(conv_id, [])` is skipped when the read returned empty for the wrong reason, and `forget_conversation` returns `0` when RAG is latched off. All counters falsy → the user is told *"Nothing to forget — this conversation had no stored memory."* while the facts file and every embedded exchange remain on disk and re-inject next turn. This is the exact inverse of D6.

**Defect, F24.** Verified coverage at `selftest.py:369-394`: storage, facts round-trip, vLLM `/v1/models`, compactor `/health`, admin, STT, TTS, chat. **No** chroma, summary, persona, backup-freshness, or destructive-path check. A pod with F8 latched, F1b having wiped a summary tree, and a dead backup daemon passes 8/8.

**Consequence.** "It stopped remembering anything older than the last few messages", arriving without a single error line. Then `/forget` reports success without forgetting.

**Change.**
- Wire `retrieval.is_available()` into `gather_health_full["checks"]`.
- Replace the permanent latch with retry-with-backoff — reset `_available = None` after N seconds.
- Rate-limited `WARNING` whenever `index_exchange` returns `False` for a real exchange.
- Add an episodic round-trip (upsert + read back a sentinel doc), a summary-state check, a persona check and a backup-freshness check to `run_selftest`.
- `/forget` distinguishes read-failure from nothing-stored, deletes the file unconditionally, and reports per-layer status.
- **D6:** `/forget` with no argument reaches `_clear_all_memory`, which wipes facts, episodic, summary state **and persona** with no confirmation. Require a confirmation token (`/forget --all CONFIRM`) and leave persona out of the default wipe.

**Verify.** `chmod 000` the chromadb directory, restart, send a chat: `/health/full` must show `episodic: down` and the self-test must fail. Restore permissions: within the backoff window it must recover without a restart. Then `/forget` on a conversation with known facts must report the actual per-layer counts, and the files must be gone.

**Estimate.** 5 h. **Depends on:** P0-1. **Must land before:** F5's import guard is trustworthy (it reads `conversation_doc_count`).

---

#### V5 · F9 — LRU eviction is not LRU, and it deletes permanently
**S2. The most likely non-incident explanation for "it forgets things."** `compactor/facts.py:277-306,331-339`, `main.py:1344-1345`

**Defect.** Every request does `load_facts` then `touch_facts(touched_facts)` on the **whole** set, so `last_used` is identical across the entire store at all times. The sort key at `:295` is `(last_used, added_turn)`, which therefore collapses to `added_turn` — eviction is "drop whatever was added earliest", i.e. the conversation's foundational facts.

`added_turn` carries four incompatible unit systems: `len(messages)+1` (the tail, D1), `i * 2` (`backfill.py:231`), `ctx["turn_index"]` (`commands.py:143`), and `min(...)` over a cluster (`dedup.py:231`).

Verified: `archive_stale_facts` and `restore_from_archive` are called **only** from admin endpoints (`main.py:1885,1914`) — never automatically. So eviction is **permanent deletion with no cold-storage fallback.** Past `COMPACTOR_MAX_FACTS_TOKENS=1500` (~100-150 bullets; the phantom conversation already has 105) every single turn silently deletes the oldest facts. The only signal is `pruned {dropped}` folded into an INFO line.

**Consequence.** "She remembers what I told her last week but not who she is."

**Change.** Touch only the facts actually injected or retrieved this turn. Route eviction through `archive_stale_facts` so evicted facts land in cold storage rather than being unlinked. Unify `added_turn` on a single unit — this is the D1 identity, so if D1 is deferred, at minimum normalise all four writers to message-units and document it at the field.

**Verify.** Unit test: build a store of 200 facts, inject 3, run a turn, assert the 3 injected have a newer `last_used` than the other 197. Then over-fill past the token cap and assert the dropped facts are present in the archive file.

**Estimate.** 3 h. **Depends on:** V2 (shares `facts.py` write paths).

---

#### V6 · F3 — Lazy backfill overwrites a live fact store wholesale
**S1, live.** `compactor/backfill.py:171,216,249-251`, `main.py:1398-1404`

**Defect.** `_run_backfill` builds `accumulated` from scratch and finishes with `conv_lock` → `prune_facts` → `save_facts`. The lock wraps the *write*; the read that should have informed it never happens. There is no merge with disk.

- **Trigger A (transient read error on `needs_backfill`) — KILLED.** `facts_path(conv_id).is_file()` at `:171` re-raises `EIO`; it does not return `False`. Confirmed in the tree.
- **Trigger B — CONFIRMED and unconditional.** Backfill is kicked off from the request path (`main.py:1398`) and runs for minutes. Every `_async_tail` landing in that window writes facts; the backfill's final `save_facts` erases them. Guaranteed on any conversation long enough to backfill.

Also confirmed: `main.py:1399` passes `messages,  # use original messages, not compacted` — the client's array. Under the incident's 7-of-241 condition, a triggered backfill extracts from 7 messages and writes that as the whole store.

**Change.** Read under the lock and merge (existing authoritative for membership). Refuse to run at all when `load_facts(conv_id)` returns non-empty **or** unreadable.

**Verify.** Test: start a backfill on a conversation, write facts concurrently via the tail, assert both sets survive.

**Estimate.** 2 h. **Depends on:** V2.

---

#### V7 · F4 + D39 — Dedup can merge an unbounded cluster into a truncated sentence
**S1, live, unattended.** `compactor/dedup.py:106-142,189,207-208,217,285`, called from `main.py:~1002`

**Defect.** Four compounding problems, all verified:

1. **No cluster-size cap.** Transitive union-find at `cosine >= 0.75`. A~B, B~C … Y~Z chains into one cluster with no similarity between the endpoints. `MAX_LLM_CALLS_PER_PASS=10` caps *calls*, not blast radius — one cluster is one call.
2. **`"max_tokens": 60`** at `:189` with no `finish_reason` check. A truncated merge passes the `len(cleaned) < 6` guard and is returned as canonical.
3. **D39, verified exactly as stated:** `head = raw.upper().lstrip("- *•").strip()` then `startswith("KEEP")`. Leading punctuation is tolerated; a trailing `KEEP` is not. **`"These are different, KEEP"` becomes the merged canonical fact and replaces the cluster's real facts.**
4. `to_remove.update(cluster)` at `:285`, then `kept = [f for i,f in enumerate(facts) if i not in to_remove]`. One LLM reply deletes N facts and inserts one ≤60-token line.

`_async_tail` passes `combined` — **all** facts, not just the new ones — so facts entirely unrelated to the current exchange are eligible for destruction on every turn.

**Consequence.** Thirty established facts replaced by `"Lyra is a half-elf ranger who lives in Aethermere and prefers"`.

**Change.** Cap cluster size at 4 and split larger ones. Raise `max_tokens` and reject any response with `finish_reason == "length"`. Require the merged text not be trivially shorter than the cluster's shortest member. Fix the KEEP parse to a whole-response match, not a prefix. Log removed texts at INFO so a bad merge is forensically recoverable.

**Verify.** Extend `test_dedup.py:159` with `"These are different, KEEP"` and with a `finish_reason: "length"` response. Both must be treated as "do not merge".

**Estimate.** 2 h. **Depends on:** nothing.

---

#### V8 · F22 + D18 + D49 — Missing `conv_lock` on three write paths
**S2.** *(AG4)* `compactor/commands.py` (whole module), `portability.py` (`import_conversation`), `main.py:935`

**Defect.** `commands.py` imports only `facts_module`, `time` and `logging` — **no command handler takes `conv_lock`**. `/remember` (`:140-148`) and `/forget <substring>` (`:157-163`) are unlocked load-modify-writes racing `_async_tail`'s locked write. Lost update in both directions. `import_conversation` takes no lock either (D18), unlike archive/restore/dedup. `index_exchange` runs outside `conv_lock` at `main.py:935` (D49).

**Consequence.** The user's just-remembered fact vanishes because the tail's write landed second — or the tail's extraction vanishes because `/remember` landed second.

**Change.** Pass the lock through `ctx` (avoids the import cycle) and wrap both command handlers. Add the lock to `import_conversation`. Move `index_exchange` inside the lock or document why it is safe outside.

**Verify.** Concurrency test: fire `/remember` and a tail write at the same conv simultaneously; assert both facts are present afterwards.

**Estimate.** 2 h. **Depends on:** V2 (F1d touches the same handler).

---

#### V9 · D2 + D10 + F23 — Unbounded store pollution
**S2.** `compactor/main.py` (`_async_tail` gate), `selftest.py:170-198,222`, `memory.list_known_conv_ids`

**Defect.**
- **D2:** background task calls (title/tag generation, `msgs=1`) hash to a stable `conv_id` and accumulate facts. `_async_tail` fires unconditionally on `if conv_id:`.
- **F23:** `selftest.py:170-198` posts a chat with `X-Conversation-Id: __selftest_oneshot_<hex>__` then immediately DELETEs it. The tail fires *after* the response and takes an LLM call, so the DELETE always wins and the tail then writes the file. One orphaned conversation **per boot**, forever. `_check_facts_round_trip`'s `finally` at `:222` writes `save_facts(SELFTEST_CONV_ID, [])`, leaving a permanent empty `facts/__selftest__.json`.
- **D10 + G2:** `_clear_all_memory` writes an **empty** facts file rather than removing it, and G2 writes one on every exchange regardless. `list_known_conv_ids` globs `facts/*.json`, so N only grows — which is the multiplier on F12's O(N) health scan.

**Change.** Skip the tail for background utility calls (single-message requests with no prior state, or an explicit OpenWebUI task header). Have the self-test await or suppress the tail before deleting, and remove `facts/__selftest__.json` rather than emptying it. Make `_clear_all_memory` unlink. Have `list_known_conv_ids` skip zero-fact files.

**Verify.** Boot the container five times; `ls /data/openwebui/compactor/facts/ | wc -l` must not grow. Generate a chat title; no new `facts/*.json`.

**Estimate.** 2 h. **Depends on:** V2 (G2).

---

#### V10 · F11 + F12 — The hot path and the health scan both block the event loop
**S2.** `compactor/main.py:1334-1373`, `health.py:122-154,176`, `retrieval.py:255`

**Defect, F11.** Verified that only `backend_is_multimodal` (`:1073`) and `_enforce_hard_budget` (`:1436`) were moved off the loop. Still inline on the single uvicorn event loop, per request: `degrade.guard` (`:1334`, → `shutil.disk_usage`/`statvfs`), `persona.auto_capture_persona` (`:1335`, → a read **plus** `atomic_write_json`: write + `fsync(file)` + `fsync(dir)`), `persona.text_to_inject`, `facts.load_facts` (`:1344`), `retrieval.retrieve` (`:1359`, CPU embedding + HNSW + SQLite), `summarizer.load_state` (`:1373`). On a stalled MooseFS mount these do not error — they **block**, and `/health`, `/v1/models` and every in-flight request stall together. Docker `HEALTHCHECK --timeout=10s` trips.

**Defect, F12.** `gather_memory_stats` iterates every conv_id and does three blocking operations each; `gather_health_full` is `async` but calls it synchronously at `:176`. `conversation_doc_count` (`retrieval.py:255`) calls `_chroma_collection.get(where={"conv_id": cid})` **with no `include` argument** — ChromaDB's default is `["metadatas","documents"]`, so it materialises every indexed exchange's full text just to `len()` the ids. This runs on the Docker healthcheck every 30 s, and V9 is what stops N from growing without bound.

**Consequence.** Total unresponsiveness with no error anywhere, because nothing has failed — it is waiting.

**Change.** One `run_in_threadpool` around `main.py:1334-1373`. Explicit deadlines on `degrade._free_mb` and `atomic_write_json`. `include=[]` on the count query. Threadpool + 60 s cache for `gather_memory_stats`, and make the aggregate stats opt-in so the healthcheck path is vLLM + storage only.

**Verify.** Under a `SIGSTOP`ed MooseFS mount simulation, `/health` still answers within its timeout. `time curl /health` with 300 conversations in the store returns in well under a second.

**Estimate.** 5 h. **Depends on:** V2 (same call sites), V9 (bounds N).

---

#### V11 · D56 + D13 — Every image is stripped, and the fix arms a latent budget bug
**S2 (D56, live and user-visible) + S1-adjacent (D13, currently dormant).** `compactor/main.py:580,217`

**Defect, D56.** `COMPACTOR_MAX_RETAINED_IMAGES=0` is falsy, so `keep = set()` at `main.py:580` and **every image is stripped, including the current turn's** — silently, on a vision model (`coder3101/Cydonia-24B-v4.3-vision-heretic`). The user sends a photo; the model never sees it.

**Defect, D13.** `overshoot = actual - HARD_INPUT_LIMIT` at `main.py:217`, capped at `MAX_MODEL_LEN//4 = 8192`, is **process-global, never reset, and lost on restart**. It is dormant *only* because retention=0 strips images before the token count runs. Fix D56 without fixing D13 and the first image-bearing turn latches a global budget margin that then silently truncates every subsequent conversation for the life of the process.

**Read §4 before implementing this item.** These two must land together.

**Change.** Distinguish "retain no *history* images" from "strip the current turn's image": at retention 0, keep the current turn's images and strip only prior turns'. Simultaneously make `_BUDGET_MARGIN` per-conversation (or at minimum decay it, and reset it on a model-id change — see F18).

**Verify.** Send a photo with `MAX_RETAINED_IMAGES=0`; the model must describe it. Then send twenty text turns and confirm the injected-token budget has not shrunk.

**Estimate.** 2 h. **Depends on:** nothing, but **D13 must land in the same commit as D56.**

---

#### V12 · F6 + F19 + F27 + F28 — Backup and shutdown hygiene
**S1/S3 mixed, mechanical.** `compactor/backup.py:100-101,159-189,425-427`, `supervisord.conf`

- **F6 (S1, bounded).** `backup.py:425-427`: `if sroot.exists(): shutil.rmtree(sroot)` then `shutil.copytree(...)`. Not atomic, no rollback. The archive is on the same volume by default (`/data/backups` vs `/data/openwebui/compactor`), so `ENOSPC` mid-copy is plausible. **Bounded:** verified CLI-only — `restore_backup` is not wired to any HTTP endpoint; `/admin/backups` exposes only list/run/verify at `main.py:2055-2088`. Still, this is the last-resort recovery tool and it has a window in which it causes the loss. **Fix:** copytree to `.{name}.incoming`, `os.replace`, rename the old store aside, delete last. **1 h.**
- **F19 (S2).** `backup.py:159-164,167,177-189`: the space guard compares free space to a fixed 500 MB and **never estimates what it is about to write**; `mkdtemp(..., dir=str(d))` stages a full **uncompressed** copy of `webui.db` plus the whole compactor store inside `/data/backups`. A 2 GB db with 600 MB free passes the guard and exhausts the volume. `_free_mb` returns `float("inf")` on exception (`:100-101`), disabling the guard entirely on a failed `statvfs`. The module's stated principle 3 is "Can't fill the disk." **Fix:** estimate `du(db) + du(store) + 20 %`; stage on the container disk; fail closed when `_free_mb` cannot read. **2 h.**
- **F27 (S3).** `stopwaitsecs` is unset **anywhere** in `supervisord.conf` (verified by grep) → default 10 s then `SIGKILL`. Shutdown is uvicorn drain **plus** `pool.drain(10.0)`, which already exceeds 10 s, so **`SIGKILL` is the normal shutdown**, and it can land mid-`chroma.upsert`. **Fix:** `stopwaitsecs=70` on the compactor, `drain(60)`. **30 min.**
- **F28 (S3).** `backup.py:189` `copytree` with no `ignore`; `memory.atomic_write_json` creates `<name>.json.XXXXXX.tmp` in the same directory and `os.replace`s it away. Scandir-then-stat → `shutil.Error` → the whole cycle produces nothing, reported only in `backup.log`, which nothing reads (P0-1, P0-2). Orphan `.tmp` files after `SIGKILL` are never swept. **Fix:** `ignore=shutil.ignore_patterns("*.tmp")`; sweep orphan `.tmp` files older than an hour at boot. **1 h.**

**Estimate.** 4.5 h total. **Depends on:** V3 for F19/F28 (same module).

---

#### V13 · F20 + F21 — Two small correctness bugs on the memory path
**S3.**

- **F20.** `main.py:845`: `if choice.get("finish_reason"): self._complete = True` fires for `"length"`, so the rc6 guard blocks client disconnects but **not model truncation** — a truncated reply is memorised as if complete. Separately, the non-stream path (`main.py:~1573` vs `:~1583`) fires the tail **before** `if r.status_code >= 400:`, so a rejected upstream response also reaches the tail. Harmless today only because `assistant_text` ends up empty. **Fix:** treat `finish_reason == "length"` as incomplete; move the tail after the status check. **30 min.**
- **F21.** `backfill.py:310` adds to `_in_progress_local` **before** `fire_and_forget` at `:313`. `pool.submit` may shed → `coro.close()` on a never-started coroutine → the body never runs → the `finally: _in_progress_local.discard` at `:288` never executes and no state file is written. That conversation's backfill is **blocked for the life of the process, unlogged**. **Fix:** add to the set only if `submit` returned `True`. **15 min.**

**Estimate.** 45 min. **Depends on:** nothing.

---

### Phase 1b — v3.1 if time permits

Ship if the release is not already late. Roughly **10 h**.

---

#### V14 · F16 — One long rollup starves all background memory work; shed tails vanish unlogged
**S2.** `compactor/bgwork.py:51-78,64-69`, `summarizer.py:393-415`, `main.py:~872,1080`

`_run` is `async with self._sem: await coro`, and `_async_tail` acquires `conv_lock` **after** taking a semaphore slot. `maybe_rollup` holds that lock across a `while _needs_l1_rollup(...)` drain, then L2, then L3 — each iteration a full LLM call on a 24B model. Four quick messages on one conversation occupy all `MAX_CONCURRENT=4` slots blocking on one lock, and no background work for **any** conversation proceeds. Past `MAX_OUTSTANDING=64`, `submit()` returns `False` and `_fire_and_forget` **discards the return value**; the shed log is rate-limited 1-in-25 and carries no `conv_id` and no turn index. Those turns' facts, embeddings and rollups are never extracted and never retried. `pool.drain(timeout=10.0)` at `main.py:1080` — a tail is three sequential LLM calls, so every `supervisorctl stop compactor` abandons up to 64 tails.

**Change.** Take the lock before the semaphore, or give rollups a separate pool. Bound the drain loop and release the lock between tiers. Log `conv_id` + `turn_index` on every shed. `drain(60)` with `stopwaitsecs=70` (pairs with F27).

**Estimate.** 4 h. **Depends on:** V12 (F27).

---

#### V15 · F17(b,c) — Forensics are container-local and container-disk exhaustion is invisible
**S2.** `supervisord.conf`, `compactor/degrade.py:49-51`

`/var/log/supervisor` is on the **container** filesystem; a pod recreate destroys every log, and the 2026-08-24 investigation turned on two adjacent log lines. Separately, `stdout_logfile_maxbytes=50MB` with supervisord's default `logfile_backups=10` across 7 programs × 2 streams is multiple GB on the 60 GB container disk, while `degrade.py:49-51` watches `/data` **only** — container-disk exhaustion is invisible to every guard.

**Change.** Log paths → `/data/logs` with `logfile_backups=3`. Add a second `degrade` watch on `/`.

**Estimate.** 2 h. **Depends on:** P0-2.

---

#### V16 · F18 — vLLM-derived caches are never invalidated, and the runbook creates the divergence
**S2.** `compactor/main.py:115-133,152-181,194-228`, `OPERATIONS.md:~86`

`_tokenizer`, `_backend_multimodal` and `_BUDGET_MARGIN` are all process-global and none is invalidated on a vLLM restart. `OPERATIONS.md:~86` instructs `supervisorctl stop vllm` / `start vllm`, which leaves the compactor running with the **previous** model's tokenizer, modality verdict and learned budget margin.

**Change.** Poll `/v1/models` and reset all three on a model-id change. Failing that, add `supervisorctl restart compactor` to every vLLM-restart instruction in `OPERATIONS.md`.

**Estimate.** 2 h. **Depends on:** V11 (shares `_BUDGET_MARGIN`).

---

#### V17 · F25 + F33 — Boot and tokenizer robustness
**S3.**

- **F25.** `entrypoint.sh:91-105` retries huggingface.co 30 times then `exit 1`. Weights are already at `HF_HOME=/data/models` and the embedding model is baked at `/opt/embeddings`. Verified: `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` appear **nowhere** in the repo. A pod with every byte cached locally cannot boot when huggingface.co is unreachable. **Fix:** set `HF_HUB_OFFLINE=1` when the local cache is complete, and make the reachability probe non-fatal in that case. **1 h.**
- **F33.** `main.py:132` sets `_tokenizer = None` on failure, indistinguishable from "not yet loaded", so `count_tokens([m])` (`main.py:715`) re-attempts `AutoTokenizer.from_pretrained` once per message. `get_tokenizer` is not pre-warmed in `lifespan` (only `backend_is_multimodal` is, `:1073`). *Correction: the claim that this blocks the event loop is **wrong** — `_enforce_hard_budget` has been in `run_in_threadpool` since `main.py:1436`. The retry storm is real; the loop-blocking is not.* **Fix:** cache the failure with a sentinel and a retry interval; pre-warm in `lifespan`. **1 h.**

**Estimate.** 2 h. **Depends on:** nothing.

---

#### P1b-M · Two mutations that survived the Phase-1 gate
**S3.** `retrieval.py:224`, `main.py:2073`

Recorded because they are the honest residue of a gate that otherwise came back green (26/26, 19 mutations applied, 16 killed). A surviving mutation is not a bug — it is a line no test defends, which is exactly what this project keeps getting hurt by.

1. **`_id_exists` fails open, untested.** Mutating `return False` → `return True` in the probe's exception handler **survives the entire suite.** The code is right: a failed probe should fall through to embed-and-upsert rather than skip the write. The mutant inverts that, so a broken probe makes `index_exchange` return `True` having stored nothing — silent loss presenting as success, which is this branch's whole failure class. **Fix:** one test that breaks the probe and asserts the document is still written. **30 min.**

2. **`enforced_limit` capture, untested.** Mutating away `- _BUDGET_MARGIN` at `main.py:2073` survives. Harmless in scope — the value feeds only the rejection log, while enforcement uses `effective_limit` — but the stated reason for capturing rather than recomputing has nothing holding it. **Fix:** assert the logged limit matches the enforced one. **15 min.**

**Estimate.** 45 min. **Depends on:** nothing.

---

### Phase 2 — Deferred

Not in v3.1. Each has a stated gate.

| Id | Item | Why deferred | Gate |
|---|---|---|---|
| **D1** | Content-addressed episodic document ids. `turn_index = len(messages)+1` (`main.py:1202`) is used as the ChromaDB id via `_doc_id` (`retrieval.py:144`) and written with `upsert` (`:166`). A client sending a bounded window **pins the id and every exchange overwrites the last** — confirmed: phantom conv `<phantom-conv>` has 105 facts and **one** episodic row. The accepted fix is content-addressed ids, **not** a per-request counter: `turn_index`, `added_turn` and `recent_cutoff` are message-units and a counter would be exchange-units. | It is the root of F14, F26, D21, D22 and the full F15 fix. It is a schema change to a live store and needs a migration, so it wants its own release. | Schedule immediately after v3.1 ships. **Nothing that makes `conv_id` more stable may land before it** — see §4. |
| **F5** | Import guard fails open; export hides its own failures *(AG3)*. `portability.py:58-90,116-121,147-169,224`. `pre_existing` is built from three best-effort reads that each return empty on failure — `load_facts` (F1a), `conversation_doc_count` (F8), `load_state(...).get("l1")` (F1b) — so the refusal at `:152` never fires and `save_facts`/`save_state` overwrite wholesale anyway. `_validate_bundle` checks `facts` is a *list*, nothing about elements. `export_conversation` substitutes `[]`/`{}` per layer on failure and returns **HTTP 200**; `fork_conversation:224` runs the same export. *Correction: the claim that a fully-rolled-up conversation has `l1 == []` is overstated — `_do_l2_rollup` leaves 0-9 chunks, so that holds only at exact multiples of 10. The fail-open reads are the real defect.* | V2 and V4 remove two of the three fail-open reads, which defuses most of it. The remaining work (per-layer `{ok,error}`, 5xx on partial export, element validation) is best done alongside D1's schema change. The `conv_lock` half (D18) ships in V8. | After D1. |
| **F14** | Summary turn indices are array positions, and the counter latches. `summarizer.py:187-190,223-241,300-314,391`. `last_summarized_turn` is durable state keyed on a position in whatever array the client sent; a deletion, edit, branch switch or windowed re-send shifts every subsequent index. If the history shrinks below `last_summarized_turn` (241 → 7), `_needs_l1_rollup` is false **forever** and L1 rollups stop permanently and silently. *Correction: the claim that `:314` advances the counter regardless of how many turns `_format_turns` produced is **killed** — `_needs_l1_rollup` guarantees `current_turn_count >= last + 20` and `_format_turns` walks the same non-system count.* | Shares D1's root. Interim mitigation is cheap and worth doing in V4's neighbourhood if it is free: reset `last_summarized_turn` when the observed history is shorter than it. | After D1. |
| **F26 + D21** | `retrieve()` has no relevance floor (`retrieval.py:201-226`: `n_results=max(1,k)`, no distance threshold) and filters **after** `query()` (`:178-227`), so exclusions shrink the result set instead of backfilling. Up to 5 past exchanges are injected under a header asserting they are relevant. | Compounds with F15, which P0-3 fixes. Both want the D1 turn identity. | After D1. ~45 min once there. |
| **F29** | Recovery is all-or-nothing across every conversation plus `webui.db` (`backup.py:389-432`). The realistic need is "restore this one conversation to twelve hours ago." | High value given the incident history, but V3 must land first — there is no point building selective restore on top of archives that can be empty. | After V3. ~3 h. |
| **F30** | Client transcript and server memory diverge when vLLM dies mid-generation (`main.py:1493-1512`): partial content already yielded, `vllm_failed=True` correctly skips the tail, OpenWebUI persists the partial turn. | Cosmetic today. Becomes a **correctness** problem the moment FRONTEND_SPEC's "server owns the transcript" lands. | Resolve in the same pass as FRONTEND_SPEC §4/§11. |
| **F31(fix)** | Move `chromadb/` off MooseFS to the container disk and rebuild from `webui.db` on boot. It is derived data, not primary, and this aligns with ARCHITECTURE.md Decision 4. | Only if P0-4 returns `wal`. ChromaDB 1.5.9 has no journal-mode setting, so there is no cheap version of this fix. | P0-4's answer. ~6 h. |
| **D7** | The `conv_id` sanitizer (`memory.py:41`, `[^A-Za-z0-9_\-]`) is a filename filter, not a validator — `"CLONE_CONV_ID_HERE"` passed through and created a real conversation. | Once the server owns the transcript, `conv_id` is server-generated and the class disappears. Interim: log a WARNING for any `conv_id` not matching the expected UUID shape. | FRONTEND_SPEC §4/§11. |
| **D3, D4** | No CORS middleware anywhere in the compactor; `_require_localhost` is a network-position check, not auth, and `/admin/*` has no authentication. | Both are squarely in scope for the V4 auth work (PR #30). Doing a partial version now creates a migration to undo. | PR #30. |

---

### Phase 3 — Dropped

See §5.

---

## 4. Ordering constraints

These are the traps that turn a fix into an incident. Each is stated as *X before Y, because Z*.

1. **D1 (content-addressed episodic ids) must land before any change that makes `conv_id` more stable or more widely shared.**
   Because: `_doc_id` (`retrieval.py:144`) is derived from `turn_index = len(messages)+1` and written with `upsert` (`:166`). Any change that causes more traffic to resolve to a single stable `conv_id` — canonical server-side ids, background-task consolidation, fork/import identity work — increases the number of conversations whose entire episodic history is pinned to one perpetually-overwritten row. The phantom conversation with 105 facts and **one** episodic row is what this looks like. V9's D2 fix is safe here because it *removes* traffic from the hashed background-task ids rather than routing more into them; anything that adds traffic is not.

2. **D13 (budget calibration) must land in the same commit as D56 (image stripping), and both before `COMPACTOR_MAX_RETAINED_IMAGES` is raised above 0.**
   Because: `overshoot = actual - HARD_INPUT_LIMIT` (`main.py:217`) is process-global, capped at 8192, never reset and lost on restart. It is dormant **only** because retention=0 strips images before the token count runs (`main.py:580`). Fix the stripping alone and the first image-bearing turn latches a global budget margin that silently truncates every conversation on that process from then on. This is V11; do not split it.

3. **F32 (the fixture) before F1 (the primitive).**
   Because: every S1 finding depends on a code path that no test exercises. Writing the fixture second means shipping the fix on the strength of an argument in a document. Writing it first means watching seven tests fail and then go green. And per §1.4, a fixture that patches `Path.stat` reproduces **none** of them — you would ship a green suite that proves nothing.

4. **F1 (V2) before F3 (V6), F5, F10 and F22 (V8).**
   Because: all four are callers of the same primitive, and their fixes are expressed in terms of the `PRESENT | ABSENT | UNREADABLE` tri-state. Doing them first means writing the same discrimination logic four times and then deleting it.

5. **F8 (V4) before F5's import guard is trusted.**
   Because: the guard's second clause is `retrieval.conversation_doc_count(target) > 0`, which returns `0` whenever the retrieval latch is off. Until F8 is fixed, "the guard passes" carries no information.

6. **P0-1 (the webhook) before F2/F7 (V3) and before F24 (V4).**
   Because: those items' entire value is that they raise an alarm. `alert.py:35` returns `False` with no webhook configured, so alarm-raising code added before the webhook exists is dead code you cannot test.

7. **P0-4 (the `PRAGMA journal_mode` check) before V3's chroma snapshot work.**
   Because: if the answer is `wal` and F31(fix) moves `chromadb/` to the container disk, V3's `_snapshot_sqlite` routing and integrity check target a different path with different atomicity properties. Decide first, build once.

8. **V9 (store pollution) before V10's health-scan work is meaningful.**
   Because: F12 is O(conversations), and G2 currently guarantees that N grows monotonically forever. Caching an O(N) scan whose N is unbounded postpones the problem rather than fixing it.

9. **P0-0c (measure via `/tokenize`) before P0-0b (calibration arithmetic), and before any tuning of `GENERATION_RESERVE`.**
   Because: the calibration loop exists only to grope toward a number `/tokenize` returns directly. Fix the measurement and the loop becomes a degraded-mode fallback rather than the primary path, which changes what "correct" means for its arithmetic. And per P0-0c's closing note, the guard table that appears to justify a smaller reserve measures *input* while the reserve exists for *output* — tuning it from that table truncates replies that measure 7,513–11,347 tokens.

10. **P0-0e (import smoke test) before anything is hot-patched onto a running pod, in any circumstance.**
   Because: it is two lines, it catches the class of error that has now cost a deployment five times, and the moment it matters most is exactly when nobody has time for it. `selftest.py` cannot substitute — it runs inside a process that has already imported successfully.

11. **F27 (`stopwaitsecs`) with F16's `drain(60)`, not before it.**
   Because: raising the drain timeout without raising `stopwaitsecs` means supervisord `SIGKILL`s the process mid-drain — strictly worse than today. They are one change.

---

## 5. Dropped, with reasons

Recorded so they are not re-proposed. Each of these was considered and deliberately not done.

| Item | Reason |
|---|---|
| **CORS middleware (D3)** | Production serves one non-technical user plus the owner, through OpenWebUI on the same origin, on a pod whose compactor port is not published. There is no browser origin that CORS would protect against. It is a real gap, but it is a **V4 auth** gap (PR #30), and adding a permissive-by-necessity CORS policy now would have to be undone. |
| **Authenticating `/admin/*` now (D4)** | Same reason. `_require_localhost` is a network-position check, not auth, and that is worth fixing — inside the auth work, once, with the rest of the auth model, rather than bolting a shared secret on and migrating off it three months later. |
| **`conv_id` validator (D7) as a v3.1 item** | Deferred rather than dropped, but explicitly *not* worth building a strict validator for now: once the server owns the transcript, `conv_id` is generated server-side and the whole class of "a template placeholder became a real conversation" disappears. A WARNING log line is the correct interim spend. |
| **The 64-character `conv_id` truncation (G4)** | `memory._sanitize` truncates to 64 chars (`memory.py:42,54`). OpenWebUI UUIDs are 36 chars and `fork_conversation` appends `__fork_<6hex>` (13 chars), so a **third**-generation fork exceeds 64 and its suffix is truncated away — every 3rd-gen fork of the same parent collides. Recorded so it is not re-found. **It fails loudly** (`import_conversation(overwrite=False)` raises) rather than corrupting, so it is a usability bug, not a data-loss one. Do not spend S1 effort on it. |
| **Rewriting dedup to avoid the LLM entirely** | Tempting, and out of scope. V7's four bounded changes (cluster cap, `finish_reason` check, length floor, KEEP parse) remove the destructive failure modes without touching the design. Re-architecting the merge strategy is a feature, not a remediation. |
| **Building the Postgres sidecar in v3.1** | ARCHITECTURE.md Decision 4 designates a Postgres sidecar as the state home. It is the right destination and it is not a remediation — it is the next architecture. Shipping v3.1's durability fixes into the JSON-file store is not wasted work: the tri-state read contract, the alert plumbing, the backup manifest assertions and the test fixture all survive the migration. |
| **Fixing D22 (rollup gate counts the client's message array, `summarizer.py:391`) as its own item** | Genuinely moot under the committed direction. Once the server owns the transcript and the model receives a bounded window, "count the client's array" stops being a defect and starts being a bug in a code path that no longer exists. It is subsumed by D1 + FRONTEND_SPEC §4/§11. Note that P0-3 (F15) is **not** in this category — F15 fires *harder* under a windowed client, which is why it is in Phase 0. |
| **Making `_clear_all_memory` recoverable (undo/tombstone) rather than confirmed (D6)** | V4 adds a confirmation token. A tombstone-and-restore mechanism for a destructive command that one person issues, on a system whose backup layer is being rebuilt in the same release, is more machinery than the risk justifies. Revisit if `/forget --all CONFIRM` is ever actually fired in anger. |
| **Retrying shed background tasks (part of F16)** | V14 logs `conv_id` + `turn_index` on shed so the loss is visible and manually recoverable. A durable retry queue for background memory work is the Postgres design's job. Do not build a file-backed one first. |
| **Adding auth or rate limiting to the self-test's chat probe (F23-adjacent)** | The self-test's orphaned conversations are a store-pollution problem (V9), not a security one. Fix the orphaning; do not gate the probe. |

---

## 6. Verification

How to know the whole remediation worked. Run these against a v3.1 build on a scratch pod, not production.

### 6.1 The alarm actually rings

```sh
# 1. vLLM dies
supervisorctl stop vllm
#    expect: a webhook message within one healthcheck interval
#    expect: curl -s -o /dev/null -w '%{http_code}' localhost:8000/health  -> 503, not 200

# 2. Backup goes stale
mv /data/backups /data/backups.aside && sleep "$((COMPACTOR_BACKUP_INTERVAL_HOURS*3600*2))"
#    expect: a webhook message; /health/full shows status "degraded"

# 3. Self-test failure is abnormal
#    force any check to fail, then:
supervisorctl status selftest    # expect FATAL, not EXITED
```

### 6.2 A corrupt file no longer destroys anything

```sh
CONV=<a scratch conv with >20 turns of history>
R=/data/openwebui/compactor
for f in facts/$CONV.json summaries/$CONV.json personas/$CONV.json; do
  cp "$R/$f" "/tmp/$(basename $f).good"
  printf 'not json' > "$R/$f"
  # send one chat turn through the compactor for $CONV
  cmp "$R/$f" "/tmp/$(basename $f).good"   # expect: FILES DIFFER is a FAILURE
                                            # expect: file unchanged (still 'not json'), NOT overwritten
  # expect: an alert fired naming the path
  cp "/tmp/$(basename $f).good" "$R/$f"
done
```
The pass condition is that the compactor **refused to write** and told you. A restored-from-nothing file that parses cleanly is the failure.

### 6.3 The backup is a backup

```sh
# an unmounted store must fail the cycle, not publish an empty archive
COMPACTOR_STORAGE_ROOT=/data/does-not-exist python -m compactor.backup once
#   expect: non-zero exit, an alert, and NO new file in /data/backups
ls /data/backups | wc -l    # unchanged

# a restart loop must not eat history
OLDEST=$(ls -1t /data/backups | tail -1)
for i in $(seq 1 10); do supervisorctl restart backup; sleep 5; done
ls /data/backups | grep -q "$OLDEST"    # expect: still there

# the newest archive contains user data
tar tzf /data/backups/$(ls -1t /data/backups | head -1) | grep -c 'compactor/facts/'   # expect: > 0
```

### 6.4 Memory is actually being written and read

```sh
# episodic is live and reported
curl -s localhost:8000/health/full | jq '.checks.episodic, .memory.indexed_exchanges_total'
#   expect: "ok" and a number that GROWS after each exchange

# short requests still retrieve (F15)
#   send a 2-message request to a conv with >20 indexed exchanges
grep 'conv_id=<CONV>' /var/log/supervisor/compactor.log | tail -1
#   expect: an Nretr=<n>0 field present in the line

# one episodic row per exchange, not one per conversation (post-D1 only)
```

### 6.5 The log is where the runbook says

```sh
tail -f /var/log/supervisor/compactor.log     # expect: conv_id=/injected memory/compacted lines
tail -f /var/log/supervisor/compactor-error.log   # expect: warnings and errors only
```

### 6.6 Nothing grows without bound

```sh
N1=$(ls /data/openwebui/compactor/facts | wc -l)
for i in 1 2 3 4 5; do supervisorctl restart compactor; sleep 20; done
# also: generate five chat titles through OpenWebUI
N2=$(ls /data/openwebui/compactor/facts | wc -l)
[ "$N1" = "$N2" ]    # expect: equal
```

### 6.7 Nothing blocks

```sh
# with ~300 conversations in the store
time curl -s localhost:8000/health       # expect: < 200 ms
time curl -s localhost:8000/health/full  # expect: < 1 s
```

### 6.8 The suite proves something

`pytest compactor/` must include, and pass, at least: a corrupt-file case and an `OSError`-from-`open` case for `load_facts`, `load_state`, `load_persona`, `load_archive`, `_run_backfill`, `create_backup` and `_clear_all_memory`; a `"These are different, KEEP"` dedup case; a `finish_reason: "length"` dedup case; a concurrent `/remember`-vs-tail case; a backfill-vs-tail case; and a `test_backup_without_store_fails`. Every chaos scenario must assert a post-condition about surviving data, not only an HTTP 200.

**Each of those tests must fail on `07ede2e`.** If one passes against the baseline, it is testing the wrong thing.

---

## 7. Full finding catalogue

Complete reference set, as verified against `07ede2e`. **AG** = found independently by both reviewers.

| Id | Sev | Title | Anchor | Phase |
|---|---|---|---|---|
| F1a *(D33)* | S1 | Corrupt/unreadable facts file → whole store overwritten | `facts.py:104-119`, `main.py:935-1000` | V2 |
| F1b *(AG1)* | S1 | Corrupt/unreadable summary file → whole L1/L2/L3 stack overwritten | `summarizer.py:106,394,421` | V2 |
| F1c *(D8)* | S1 | Persona wiped on the request path before vLLM is called | `persona.py:86` | V2 |
| F1d | S1 | `/remember` replaces the fact store with one fact; reports `Facts now: 1` | `commands.py:140,148` | V2 |
| F1e | S1 | Archive and restore each destroy the store they read from | `facts.py:210-212,255-257` | V2 |
| G2 | S1 | `_async_tail` writes a facts file on **every** exchange, even when extraction yields nothing | `main.py:~995-1000` | V2 |
| G3 | S1 | `save_state` sits outside the rollup try; an empty skeleton is written with zero LLM involvement | `summarizer.py:416,421` | V2 |
| F2 *(AG5)* | S1 | Empty backup verifies OK, publishes, and prunes the real archives | `backup.py:187,258,266,344` | V3 |
| F7 | S1 | Restart loop destroys backup history; nominal retention is 42 h | `backup.py:448-452`, `runpod.env.template:47,85` | V3 / §2.2 |
| F3 | S1 | Lazy backfill overwrites a live fact store wholesale | `backfill.py:249-251`, `main.py:1398-1404` | V6 |
| F4 + D39 | S1 | Dedup merges an unbounded cluster into a possibly-truncated sentence | `dedup.py:106-142,189,207-208,285` | V7 |
| F6 | S1 | Restore deletes the live store before it knows the copy will land (CLI-only) | `backup.py:425-427` | V12 |
| F5 *(AG3)* | S1 | Import guard fails open; export hides its own failures and returns 200 | `portability.py:58-90,147-169,224` | Deferred |
| D1 | S1 | Episodic doc id derived from client array length; `upsert` pins one row | `main.py:1202`, `retrieval.py:144,166` | Deferred |
| F8 *(AG2)* | S2 | Episodic memory latches off for process life; `is_available()` has zero callers | `retrieval.py:68-109`, `health.py:176` | V4 |
| F9 | S2 | "LRU" eviction is not LRU and deletes permanently | `facts.py:277-306,331-339` | V5 |
| F10 | S2 | `/forget` reports success when the wipe did not happen | `main.py:1725-1731`, `commands.py:168-181` | V4 |
| F11 | S2 | Every hot-path storage op is synchronous on the event loop, including an fsync | `main.py:1334-1373` | V10 |
| F12 | S2 | `/health/full` is O(conversations) blocking I/O every 30 s; loads the whole corpus | `health.py:122-154`, `retrieval.py:255` | V10 |
| F13 | S2 | No outbound failure signal of any kind — six inert paths | `runpod.env.template:130`, `health.py:230`, `supervisord.conf:163,199-201` | P0-1 |
| F15 | S2 | Episodic retrieval fully disabled for any ≤7-message request | `main.py:1358`, `retrieval.py:219` | P0-3 |
| F16 | S2 | One long rollup starves all background work; shed tails vanish unlogged | `bgwork.py:51-78`, `summarizer.py:393-415` | V14 |
| F17 | S2/S3 | Operational log is in the "error" file; forensics container-local; container disk unwatched | `logsetup.py:60`, `degrade.py:49-51` | P0-2 / V15 |
| F18 | S2 | vLLM-derived caches never invalidated; the runbook creates the divergence | `main.py:115-133,152-181,194-228` | V16 |
| F19 | S2 | The backup can be the thing that fills the disk; `_free_mb` fails open | `backup.py:100-101,159-189` | V12 |
| F24 | S2 | The self-test cannot fail for anything that has actually broken | `selftest.py:369-394` | V4 |
| F32 *(AG6)* | S2 | Tests certify the defects; chaos suite asserts availability, not durability | `test_facts.py:62,141`, `run_chaos.py:94-113` | V1 |
| D2 | S2 | Background task calls accumulate facts under a stable hashed conv_id | `_async_tail` gate | V9 |
| D10 | S2 | Unbounded store pollution; empty facts files counted forever | `memory.list_known_conv_ids` | V9 |
| F23 | S2 | Self-test orphans one conversation per boot | `selftest.py:170-198,222` | V9 |
| D56 | S2 | `MAX_RETAINED_IMAGES=0` is falsy → **every** image stripped on a vision model | `main.py:580` | V11 |
| D13 | S2 | Budget overshoot is process-global, never reset; dormant only because of D56 | `main.py:217` | V11 |
| F14 | S2 | Summary turn indices are array positions; the counter latches permanently | `summarizer.py:187-190,300-314` | Deferred |
| F31 | ? | ChromaDB SQLite journal mode on MooseFS — **unverified** | `retrieval.py:90` | P0-4 / Deferred |
| F20 | S3 | Truncated and rejected replies are memorised | `main.py:845,~1573` | V13 |
| F21 | S3 | A shed background task permanently blocks that conversation's backfill | `backfill.py:288,310,313` | V13 |
| F22 *(AG4)* | S3 | No `conv_lock` anywhere in `commands.py` | `commands.py:140-163` | V8 |
| D18 | S3 | `import_conversation` takes no `conv_lock` | `portability.py` | V8 |
| D49 | S3 | `index_exchange` runs outside `conv_lock` | `main.py:935` | V8 |
| F25 | S3 | A fully-cached pod cannot boot when huggingface.co is unreachable | `entrypoint.sh:91-105` | V17 |
| F27 | S3 | `stopwaitsecs` unset → `SIGKILL` is the normal shutdown | `supervisord.conf` | V12 |
| F28 | S3 | A racing `.tmp` file fails the whole backup cycle; orphans never swept | `backup.py:189`, `memory.atomic_write_json` | V12 |
| F33 | S3 | A failed tokenizer load is never cached → per-message retry storm | `main.py:132,715` | V17 |
| D6 | S3 | `/forget` reaches `_clear_all_memory` with no confirmation, and takes the persona | `main.py:1725-1731` | V4 |
| F26 + D21 | S3 | No relevance floor; filter-after-query shrinks results instead of backfilling | `retrieval.py:178-227` | Deferred |
| F29 | S3 | Recovery is all-or-nothing across every conversation plus `webui.db` | `backup.py:389-432` | Deferred |
| F30 | S3 | Client transcript and server memory diverge when vLLM dies mid-generation | `main.py:1493-1512` | Deferred |
| D22 | S3 | Rollup gate counts the client's message array | `summarizer.py:391` | Dropped (subsumed) |
| D3 | S3 | No CORS middleware anywhere | — | Dropped → PR #30 |
| D4 | S3 | `_require_localhost` is not auth; `/admin/*` unauthenticated | `main.py` | Dropped → PR #30 |
| D7 | S3 | `conv_id` sanitizer is a filename filter, not a validator | `memory.py:41` | Deferred |
| D9 | S3 | `prune_old_backups` unlinks unconditionally past `RETAIN=7` | `backup.py:275-286` | V3 |
| G4 | S4 | 64-char `conv_id` truncation collides 3rd-generation forks — **fails loudly** | `memory.py:42,54` | Dropped |

### 7.1 Claims killed or corrected during reconciliation

Recorded so nobody re-derives them from the original review text.

| Claim | Verdict |
|---|---|
| `is_file()` / `is_dir()` swallow `OSError` (asserted by both reviewers across seven findings) | **Both wrong.** Only `ENOENT`, `ENOTDIR`, `EBADF`, `ELOOP`. The findings survive via corruption and read-time `EIO`. See §1.4. |
| A MooseFS blip triggers the empty-backup path (F2) | **Wrong on mechanism, right on finding.** `ENOENT` — unmounted volume, lost network volume, config typo — reproduces it. |
| A transient read error on `needs_backfill` triggers F3 | **Killed.** `facts_path().is_file()` re-raises `EIO`. The concurrent-tail trigger carries the finding, and is unconditional. |
| `summarizer.py:314` advances the counter regardless of how many turns `_format_turns` produced | **Killed.** Unreachable given `_needs_l1_rollup`'s guarantee that `current_turn_count >= last + 20`. |
| `_enforce_hard_budget` blocks the event loop | **Killed.** In `run_in_threadpool` since `main.py:1436`. Downgraded to F33's retry storm. |
| "The Dockerfile comment claims the opposite" (re: WAL) | **Overstated.** The comment at `Dockerfile:363-368` is technically careful. The behaviour — 200 on vLLM FATAL — is real and is F13(2). |
| `retrieval.is_available()` is called only from `test_retrieval.py` | **Wrong, and the finding is stronger than stated** — zero callers anywhere, tests included. |
| A fully-rolled-up conversation has `l1 == []`, so F5's guard misreads the most valuable conversations | **Overstated.** `_do_l2_rollup` leaves 0-9 chunks, so `l1 == []` only at exact multiples of 10. The fail-open reads are the real defect. |
| `test_backup.py:139` is F2 written down as a requirement | **Corrected.** `test_backup_without_db_succeeds` seeds `with_db=False` with the store **present** — it certifies a missing DB. The gap is the absent `test_backup_without_store_fails`. |
| `logsetup.py:149` | **Citation error.** The file is 65 lines; the site is `:60`. |
| D39, D56, and D13-as-dormant | **All three independently confirmed correct as written.** |

---

*Written 2026-08-24 against `07ede2e`. If you are picking this up cold: read §1.4, then §2, then start at P0-1.*
