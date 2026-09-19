"""
compactor.backfill — Lazy V1-conversation backfill (V2.0 Phase 2).

When V2.0 first encounters a conversation that was started under V1
(no facts file on disk yet, but already many messages of history), this
module retroactively extracts facts from the full message history so
V2 memory becomes available on subsequent requests.

Design choices (per V2.0 plan):
- **Async/non-blocking.** The current request that triggered the
  backfill returns immediately without facts (graceful degrade). The
  backfill runs as a background task; facts become available on the
  next request, typically within ~30s-2min depending on conversation
  length on Magnum 12B.
- **State tracked on disk** in `backfill_state.json` per conv so a pod
  restart mid-backfill is visible and can be retried rather than the
  record sitting `in_progress` forever with nothing reading it. v3.1.9.4
  B2 (P15-2): this is a RETRY (restart from the top over the current
  message history, `_merge_backfilled` deduping against whatever is on
  disk), not an incremental resume from `exchanges_done` — a prior
  design note here claimed resume-from-progress behaviour that the code
  never actually had; `needs_backfill` is what decides a retry is due,
  and it reads the RECORD, not "does a facts file exist" (a live tail's
  own writes during an abandoned run used to look, permanently, like V2
  had already taken over).
- **Stale-detection** so a crashed backfill (process killed, OOM, etc.)
  gets retried on next encounter rather than blocking memory creation
  for that conv forever.
- **Idempotent.** Multiple concurrent calls to `maybe_start_backfill`
  for the same conv only start one task (lock + state check).
- **Additive only.** A backfill never removes or rewrites a fact that is
  already on disk. It refuses to run against a non-empty or unreadable
  store, and merges rather than replaces at the end (v3.1 F3).

Storage:
    /data/openwebui/compactor/facts/<conv_id>.backfill.json

Format:
    {
      "conv_id": "...",
      "state": "in_progress" | "complete" | "failed" | "abandoned" | "wiped",
      "started_at": "2026-05-28T...",
      "updated_at": "2026-05-28T...",
      "exchanges_done": 5,
      "exchanges_total": 47,
      "attempts": 1,
      "error": null
    }

v3.1.9.4 (R2 / P15-2 follow-up). B2's retry-by-record fix (above) turned a
PERMANENTLY abandoned backfill into one that retries — correct for a
transient failure (a redeploy, an OOM, a momentary vLLM outage), and wrong
on its own for one that fails DETERMINISTICALLY: with nothing capping it,
that conversation would spend a fresh multi-hour run of vLLM calls on every
single eligible request from then on, forever. `attempts` (new field above)
counts how many times a run has actually been started for this backfill
record's current retry generation; `needs_backfill` refuses to retry a
`failed` record until an EXPONENTIAL backoff since its last update has
elapsed (`_backoff_ready`, keyed off `attempts`), and `_run_backfill`
itself stops retrying and writes the new terminal state `"abandoned"`
instead of `"failed"` once `attempts` reaches `_MAX_BACKFILL_ATTEMPTS`,
logging one WARNING naming the conversation and its last error. A crash
(SIGKILL, OOM) that leaves a stale `in_progress` record already at the cap
is refused the same way, logged once via logsetup.log_once — that record
itself is never rewritten to `"abandoned"`, because refusing a kickoff
spends no background task to write it, so `in_progress`/stale/at-the-cap is
its own permanent (and, via the log line, discoverable) terminal shape.

v3.1.9.4 (R4 / W2, P15-5 follow-up). R2's cap/backoff above answers "how do
we stop retrying a backfill that keeps failing"; it does not answer "what
happens when she /forget-s the conversation a backfill is still running
against". A backfill is submitted the same way the live memory tail is
(`fire_and_forget`, R1's own P15-5 finding) and can run for HOURS (R2's own
docstring above, and P15-2's real-data figures: up to 2h43m for 195
exchanges) — the single longest-lived background writer in this codebase,
and until this fix the one background writer R1's wipe-generation counter
did not cover. `start_backfill_if_needed` now captures
`memory.current_wipe_generation(conv_id)` at the moment it hands the run to
`fire_and_forget` — submission, exactly where the live tail captures it, and
for the identical reason (asyncio.create_task snapshots nothing FOR us here,
since this is a plain argument rather than a monkeypatched-wholesale
coroutine — see `_run_backfill`'s own `wipe_generation` parameter). Threaded
through as `wipe_generation`, `_run_backfill` re-checks it under `conv_lock`
at TWO points: right after its first lock (before spending any vLLM calls at
all — catches a wipe that already happened by the time the backfill got its
first turn on the event loop) and again right before its final
`facts_module.save_facts` write (catches a wipe that arrived DURING the
run). It is also threaded into `summarizer.maybe_rollup` via
`summarizer.wipe_generation_ctx`, the same mechanism `main._rollup_hierarchy`
uses, so the hierarchical-summary half of a backfill is covered by the exact
same generic check `summarizer._maybe_rollup_body` already has — no new
summarizer.py logic, just this module finally passing a real value instead
of leaving the contextvar at its `None` default.

On a mismatch this module does NOT write `"failed"` (which `_backoff_ready`
would eventually retry) or `"abandoned"` (framed as "gave up after repeated
failures", the wrong story and the wrong log level for what is actually
happening): it writes the new terminal state `"wiped"` — permanent, like
`"complete"`, in `needs_backfill` — because retrying would mean
reconstructing, from the very history she asked to forget, the exact memory
`/forget` just deleted. That would not be a bug in backfill.py; it would be
`/forget` failing to be `/forget`.
"""

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from starlette.concurrency import run_in_threadpool

import envcfg
import facts as facts_module
import logsetup
import retrieval
import summarizer
from memory import (
    StoreUnreadable,
    atomic_write_json,
    conv_lock,
    current_wipe_generation,
    facts_path,
    read_json,
    storage_root,
)

logger = logging.getLogger("compactor.backfill")


# How long an "in_progress" state can sit without updates before we
# consider it crashed and retry. 10 minutes covers the longest plausible
# backfill (~2000 message conversation at 300ms/call), with margin.
_STALE_SECONDS = 600

# v3.1.9.4 (R2 / P15-2 follow-up). A backfill that fails deterministically
# (a malformed history, a store this conversation will never satisfy) must
# not re-spend a multi-hour run of vLLM calls on every future eligible
# request forever. Three tries — the first plus two retries — is enough to
# ride out a transient failure (the case B2 exists for) without turning a
# permanent one into unbounded load. Small and env-overridable rather than
# hardcoded, matching this module's other retry/budget knobs.
_MAX_BACKFILL_ATTEMPTS = envcfg.env_int("COMPACTOR_BACKFILL_MAX_ATTEMPTS", 3)

# Base backoff between a failed attempt and the next retry, DOUBLED per
# attempt already spent (attempt 1 failed -> wait this long; attempt 2
# failed -> wait 2x this long; attempt 3 failed -> abandoned, no more
# retries at all). Reuses _STALE_SECONDS' own 10-minute unit as the default
# — an operator who already knows what that number means does not need a
# second one.
_BACKFILL_RETRY_BACKOFF_S = envcfg.env_int(
    "COMPACTOR_BACKFILL_RETRY_BACKOFF_S", _STALE_SECONDS
)

# v3.1.9 (tail catch-up). YES, this gets a budget too, and reuses the tail's
# own knob (main.TAIL_ROLLUP_MAX_CALLS, same env var) rather than a
# backfill-specific one.
#
# WHY IT NEEDS ONE AT ALL. The summary rollup below (maybe_rollup at line
# ~394) is a SINGLE call, not a loop — unlike the live tail before this
# release, it was never wrapped in "drain until caught up or budget spent".
# One call still means "drain everything this state needs in its own
# internal L1/L2/L3 loops", which is exactly the all-at-once shape this
# whole feature exists to bound: a V1 conversation backfilled for the FIRST
# time can be the single largest backlog in the store (its entire history,
# discovered at once), and this module's own docstring says the backfill is
# "Async/non-blocking" — it runs as a background task WHILE she keeps
# chatting, on the same one GPU her live replies are being generated on.
# Unbounded here is the tail's original defect, reintroduced by the one
# caller that was never in scope to look at when the tail was fixed.
#
# WHY REUSING THE TAIL'S CONSTANT IS THE RIGHT CHOICE, not a new
# COMPACTOR_BACKFILL_ROLLUP_MAX_CALLS. This call is functionally the same
# kind of opportunistic, best-effort, GPU-sharing rollup attempt the live
# tail makes every turn — the module docstring already says a rollup
# failure here is non-fatal ("facts backfill is still considered
# complete"), i.e. this was ALREADY designed to leave the hierarchy for a
# later pass to finish, it just never had a "later pass" of its own. It
# gets one for free: whatever this bounded call does not finish is exactly
# what state.last_summarized_turn records as still due, and the FIRST live
# tail on this conversation (the reply that triggered the backfill, or any
# turn after) picks it up and continues under its own per-turn budget —
# the identical persisted-watermark convergence this release already
# proves for the tail, with no new resume logic needed here.
#
# `main` is not imported (nothing in this package may import it without a
# cycle — main.py imports backfill.py to serve /admin/conversations/*), so
# this reads the SAME env var independently through envcfg, the shared
# softened reader every other non-main module in this package uses.
_TAIL_ROLLUP_MAX_CALLS = envcfg.env_int("COMPACTOR_TAIL_ROLLUP_MAX_CALLS", 4)


# Module-level set of conv_ids currently being backfilled in this process.
# Avoids racing-to-start two backfills if multiple requests for the same
# stale-state conv arrive before either finishes.
_in_progress_local: set[str] = set()


# ---------------------------------------------------------------------------
# State file helpers
# ---------------------------------------------------------------------------

def _backfill_state_path(conv_id: str) -> Path:
    """Sidecar file: /data/openwebui/compactor/facts/<conv_id>.backfill.json"""
    return storage_root() / "facts" / f"{conv_id}.backfill.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _now_unix() -> int:
    return int(time.time())


def read_state(conv_id: str) -> dict | None:
    """Return current backfill state for a conv, or None if no record."""
    data = read_json(_backfill_state_path(conv_id), default=None)
    if not isinstance(data, dict):
        return None
    return data


def _write_state(conv_id: str, state: dict) -> None:
    state["conv_id"] = conv_id
    state["updated_at"] = _now_iso()
    atomic_write_json(_backfill_state_path(conv_id), state)


def _write_failed_or_abandoned(
    conv_id: str,
    attempts: int,
    *,
    started_at: str,
    exchanges_done: int,
    exchanges_total: int,
    error: str,
) -> None:
    """The one place `_run_backfill` records an attempt that did not
    succeed (v3.1.9.4, R2 / P15-2 follow-up). `attempts` is THIS run's own
    attempt number (see `_run_backfill`'s `this_attempt`) — once it reaches
    `_MAX_BACKFILL_ATTEMPTS`, the record becomes the terminal state
    `"abandoned"` instead of the retryable `"failed"`, and a WARNING names
    the conversation, the attempt count and the error, exactly once (this
    function runs once per failed run, so no log_once is needed here — see
    `needs_backfill` for the sibling case, a crash rather than a clean
    failure, which DOES need one).

    Two call sites share this rather than each deciding independently: the
    fix-one-site-miss-the-sibling defect this codebase keeps paying for is
    exactly what having only one of the two write "abandoned" would be.
    """
    if attempts >= _MAX_BACKFILL_ATTEMPTS:
        logger.warning(
            f"conv={conv_id}: backfill abandoned after {attempts} failed "
            f"attempt(s) — the {_MAX_BACKFILL_ATTEMPTS}-attempt cap is "
            f"reached, so this conversation's history will NOT be "
            f"retried again; last error: {error}"
        )
        state_value = "abandoned"
    else:
        state_value = "failed"
    _write_state(conv_id, {
        "state": state_value,
        "started_at": started_at,
        "exchanges_done": exchanges_done,
        "exchanges_total": exchanges_total,
        "attempts": attempts,
        "error": error[:500],
    })


def _write_wiped(
    conv_id: str,
    attempts: int,
    *,
    started_at: str,
    exchanges_done: int,
    exchanges_total: int,
) -> None:
    """The one place `_run_backfill` records that a wipe (`/forget`, the
    admin facts-delete endpoint, the self-test cleanup, `/retire`'s apply
    step, or an overwrite import — anything that calls
    `memory.bump_wipe_generation`) ran on this conversation while THIS run
    was in flight (v3.1.9.4, R4 / W2 / P15-5 follow-up).

    Deliberately NOT `"failed"` and NOT `"abandoned"`: both of those
    describe an extraction that could not finish and MIGHT succeed on a
    later try, which is exactly why `needs_backfill` retries a `"failed"`
    record (after `_backoff_ready`'s backoff) and only stops at
    `"abandoned"` once `_MAX_BACKFILL_ATTEMPTS` clean failures have been
    spent. A wipe is not a failure to retry past — the user asked for this
    conversation's memory to be gone, and a retry that reconstructs it from
    the very history she just asked to forget would not fix backfill.py, it
    would undo `/forget`. `"wiped"` gets its own permanent refusal in
    `needs_backfill` (the same terminal weight as `"complete"`), no backoff,
    and one INFO line rather than a WARNING — discarding here is this fix
    working as intended, not a failure worth paging anyone over.
    """
    logger.info(
        f"conv={conv_id}: backfill discarded — a wipe ran on this "
        f"conversation while this run (attempt {attempts}) was in flight; "
        f"not retrying — the user asked for that memory to be gone"
    )
    _write_state(conv_id, {
        "state": "wiped",
        "started_at": started_at,
        "exchanges_done": exchanges_done,
        "exchanges_total": exchanges_total,
        "attempts": attempts,
        "error": None,
    })


def is_stale(state: dict) -> bool:
    """A state is stale if it's marked in_progress but hasn't been touched
    in _STALE_SECONDS. Indicates a crashed backfill that should be retried.
    """
    if state.get("state") != "in_progress":
        return False
    updated_at = state.get("updated_at")
    if not updated_at:
        return True  # malformed state — treat as stale and retry
    try:
        ts = datetime.fromisoformat(updated_at)
    except (ValueError, TypeError):
        return True
    age = (datetime.now(timezone.utc) - ts).total_seconds()
    return age > _STALE_SECONDS


def _backoff_ready(state: dict) -> bool:
    """Whether enough time has passed since a `failed` backfill's last
    update to retry it again (v3.1.9.4, R2 / P15-2 follow-up).

    Round 2 (B2) made `needs_backfill` return True the instant it saw
    `state == "failed"`, with no minimum wait at all — so a backfill
    failing for a deterministic reason retried on the very next eligible
    request, spending a fresh run of vLLM calls every time. The wait
    DOUBLES per attempt already recorded (`_BACKFILL_RETRY_BACKOFF_S *
    2 ** (attempts - 1)`), so a backfill that keeps failing spreads its
    remaining tries out rather than burning through `_MAX_BACKFILL_
    ATTEMPTS` inside one busy minute — the cap in `needs_backfill`/
    `_run_backfill` is what stops it forever; this is what slows it down
    on the way there.

    Same malformed-timestamp handling as `is_stale`: a missing or
    unparseable `updated_at` does not block a retry on a guess.
    """
    updated_at = state.get("updated_at")
    if not updated_at:
        return True
    try:
        ts = datetime.fromisoformat(updated_at)
    except (ValueError, TypeError):
        return True
    age = (datetime.now(timezone.utc) - ts).total_seconds()
    # At least 1: a record with no attempts recorded (pre-fix, or a
    # malformed write) backs off by exactly the base amount, not less.
    attempts = max(1, int(state.get("attempts") or 1))
    backoff = _BACKFILL_RETRY_BACKOFF_S * (2 ** (attempts - 1))
    return age > backoff


# ---------------------------------------------------------------------------
# Message pair extraction
# ---------------------------------------------------------------------------

def _message_text(m: dict) -> str:
    content = m.get("content") or ""
    if isinstance(content, list):
        return " ".join(c.get("text", "") for c in content if isinstance(c, dict))
    return str(content)


def extract_user_assistant_pairs(messages: list[dict]) -> list[tuple[str, str]]:
    """Walk the message history and return (user, assistant) text pairs.

    Pairing rule: each user message pairs with the next assistant message,
    in order. System messages are ignored. Trailing unmatched user
    messages (the last user msg waiting for an assistant response that
    hasn't happened yet) are dropped.
    """
    pairs: list[tuple[str, str]] = []
    last_user: str | None = None
    for m in messages:
        role = m.get("role")
        text = _message_text(m).strip()
        if not text:
            continue
        if role == "user":
            last_user = text
        elif role == "assistant" and last_user is not None:
            pairs.append((last_user, text))
            last_user = None
    return pairs


# ---------------------------------------------------------------------------
# Decision: does this conv need backfill?
# ---------------------------------------------------------------------------

# Threshold: don't backfill a conversation with fewer than this many
# total messages. A fresh new conv typically has 1-3 messages on its
# first request; no point spending an LLM call to "backfill" two messages.
_MIN_MESSAGES_FOR_BACKFILL = 4


def needs_backfill(conv_id: str, messages: list[dict]) -> bool:
    """Return True iff this conv has enough history to be worth backfilling
    AND (no facts file yet, OR an earlier backfill of THIS conversation was
    abandoned or failed) AND no in-progress (non-stale) backfill is already
    running.

    v3.1.9.4 B2 (P15-2). Until this fix, `facts_path(conv_id).is_file()`
    was checked FIRST and unconditionally returned False the moment any
    facts file existed — including one the KICKOFF REQUEST'S OWN live tail
    wrote within seconds of the reply, while a lazy backfill of the same
    conversation's full history was still minutes into running in the
    background. A redeploy, OOM or crash any time after that first live
    tail write (P15-2's real-data run lengths: up to 2h43m for 195
    exchanges) left the backfill record permanently `in_progress` with no
    reader, and every later request for that conversation saw a facts file
    and never tried again — the history before that first live exchange
    was gone for good, silently.

    The retry condition is now the RECORD, not "does a facts file exist":
    a facts file only means "V2 has taken over from V1" when there is no
    unfinished backfill record for this exact conversation still open. See
    _run_backfill's own up-front refusal, which this decision has to agree
    with — a resumed run's `existing` facts (the live tail's writes made
    during the abandoned attempt) must NOT cause it to refuse itself as
    "not a V1 store"; `_merge_backfilled`'s additive merge is what makes
    running a resumed pass over a non-empty store safe.

    v3.1.9.4 (R2 / P15-2 follow-up). A `failed` record no longer retries
    unconditionally: it waits out `_backoff_ready`'s exponential backoff
    first, and once `attempts` reaches `_MAX_BACKFILL_ATTEMPTS` it does not
    retry at all — see `_run_backfill`, which writes the terminal state
    `"abandoned"` (checked here the same as `"complete"`) instead of
    `"failed"` the moment that happens, so this branch is reached at most
    `_MAX_BACKFILL_ATTEMPTS - 1` times for any one record. A stale
    `in_progress` record already AT the cap (a crash on what was already
    the Nth attempt) is refused the same way, logged once, because nothing
    will run to rewrite THAT record to `"abandoned"` — refusing a kickoff
    spends no background task.

    v3.1.9.4 (R4 / W2 / P15-5 follow-up). `"wiped"` (see `_write_wiped`) is
    the same terminal weight as `"complete"` for the identical reason: a
    wipe that ran while a backfill was in flight is not evidence the next
    attempt might succeed, it is a decision the user already made that this
    function must not second-guess by trying again.
    """
    if len(messages) < _MIN_MESSAGES_FOR_BACKFILL:
        return False  # too short to bother
    if not facts_module.extraction_enabled():
        # v3.1.9.4 B2 (P15-8). Backfill's whole job is spending vLLM calls
        # to extract facts; with extraction off there is nothing for it to
        # do; _facts_tail already honours this switch (main.py:7381) and a
        # sibling that runs the identical kind of call must too.
        return False
    state = read_state(conv_id)
    if state is not None:
        s = state.get("state")
        if s == "complete":
            return False  # already done
        if s == "abandoned":
            # v3.1.9.4 (R2). Gave up after _MAX_BACKFILL_ATTEMPTS clean
            # failures — see _run_backfill. Same terminal weight as
            # "complete": nothing about a future request changes what
            # already happened this many times.
            return False
        if s == "wiped":
            # v3.1.9.4 (R4 / W2). A wipe ran on this conversation while a
            # backfill was in flight — see _write_wiped. Refused exactly
            # like "complete"/"abandoned": retrying would reconstruct, from
            # the history she asked to forget, the exact memory /forget (or
            # /retire, or an overwrite import) just deleted.
            return False
        if s == "in_progress":
            if not is_stale(state):
                return False  # someone else is on it
            if int(state.get("attempts") or 0) >= _MAX_BACKFILL_ATTEMPTS:
                # Crashed on an attempt that was already at the cap. This
                # record stays in_progress/stale forever — see this
                # function's own docstring for why nothing rewrites it —
                # so say so exactly once rather than silently refusing
                # every request for this conversation from now on.
                if logsetup.log_once(f"backfill.cap_on_crash.{conv_id}"):
                    logger.warning(
                        f"conv={conv_id}: backfill's last attempt "
                        f"({state.get('attempts')}) crashed rather than "
                        f"failing cleanly, and it was already at the "
                        f"{_MAX_BACKFILL_ATTEMPTS}-attempt cap — not "
                        f"retrying again for this conversation"
                    )
                return False
            return True  # stale, and under the cap — retry
        if s == "failed":
            attempts = int(state.get("attempts") or 0)
            if attempts >= _MAX_BACKFILL_ATTEMPTS:
                # Reachable only for a record written before this fix
                # shipped (_run_backfill now writes "abandoned" instead of
                # letting a record reach this combination) — same decision
                # either way: defence in depth, not the primary path.
                return False
            return _backoff_ready(state)  # retry once the backoff elapses
        # Any other/unknown value on disk: fall through to the facts-file
        # check below rather than guess — same as "no record" (state=None).
    if facts_path(conv_id).is_file():
        # No backfill record for this conversation at all (state is None,
        # or an unrecognised value handled just above) — the ORIGINAL v2.0
        # gate: facts already exist, so this is not a fresh V1 conversation
        # and there is nothing to reconstruct. Advisory only: this runs at
        # kickoff and the task starts later, so _run_backfill repeats the
        # check with load_facts under the lock before it does any work
        # (v3.1 F3). Kept as is_file() here because it is the cheaper
        # answer and because an existing-but-empty facts file is still a
        # store this module has no business reconstructing.
        return False
    return True  # never attempted, and no facts file either


# ---------------------------------------------------------------------------
# The backfill itself
# ---------------------------------------------------------------------------

def _merge_backfilled(existing: list[dict], accumulated: list[dict]) -> list[dict]:
    """Reconcile a backfill's extraction with the store as it stands NOW.

    A backfill runs for minutes, off the request path, from a snapshot taken
    before it started. Every `_async_tail` that lands in that window writes
    facts the backfill has never seen, so persisting `accumulated` wholesale
    erased them — atomically, silently, on any conversation long enough to
    be worth backfilling (v3.1 F3). The per-conv lock never caught it because
    it wrapped the write and the read that should have informed it never
    happened.

    Same asymmetry as main.py's `_merge_touched`: `existing` — read under the
    lock, moments before the write — is authoritative for membership. The
    backfill only ADDS texts the store does not already carry. It never drops
    an existing fact and never rewrites one's `added_turn` or `last_used`,
    because the store's copy was recorded from a real turn while the
    backfill's is a reconstruction of one.
    """
    seen = {f.get("text") for f in existing if isinstance(f, dict)}
    merged = list(existing)
    for f in accumulated:
        if not isinstance(f, dict):
            continue
        text = f.get("text")
        if text in seen:
            continue
        seen.add(text)
        merged.append(f)
    return merged


async def _run_backfill(
    conv_id: str,
    messages: list[dict],
    vllm_url: str,
    model: str,
    raw_messages: list[dict] | None = None,
    wipe_generation: int | None = None,
) -> None:
    """The actual backfill: iterate pairs, extract facts, save state
    incrementally so a crash mid-run can resume from progress.

    Errors during individual extractions are logged but don't fail the
    whole backfill — we keep going and save whatever we got.

    `wipe_generation` (v3.1.9.4, R4 / W2 / P15-5 follow-up): the
    conversation's wipe generation as of the moment this run was SUBMITTED
    to the pool (`start_backfill_if_needed`, via `fire_and_forget`; see
    `memory.current_wipe_generation`'s own docstring). `None` (the default)
    means the caller is not participating — every direct-call test double
    of this function (`test_backfill.py`, `test_v3194_r3_r2.py`, and others
    that call `_run_backfill` positionally with today's exact signature)
    keeps working unchanged, and the check below is a no-op for them, same
    convention as `main._facts_tail`'s own `wipe_generation` keyword. When
    not None, re-checked under `conv_lock` at the two points below, and
    threaded into `summarizer.maybe_rollup` via `wipe_generation_ctx` for
    the hierarchical-summary half of this run.
    """
    started_at = _now_iso()
    pairs: list[tuple[str, str]] = []
    accumulated: list[dict] = []
    now_unix = _now_unix()
    # v3.1.9.4 (R2). Sane fallback for the outer `except` below, in the
    # (believed unreachable — read_state degrades to None rather than
    # raising) case that something fails before this is properly computed
    # a few lines into the try. Treated as attempt 1 rather than crashing
    # the exception handler itself over what to log.
    this_attempt = 1

    try:
        # Refuse before spending minutes of GPU on it. Backfill exists to give
        # a V1 conversation its FIRST facts; against a store that already has
        # any, this whole function is a write over live user memory.
        # needs_backfill checked at kickoff, but the task starts later, and
        # main.py:1509 hands us the CLIENT's message array — under the
        # 2026-08-24 7-of-241 condition a triggered backfill extracts from 7
        # messages and saves that as the entire store (v3.1 F3).
        # v3.1.9.4 B2 (P15-2). Read BEFORE this run's own _write_state below
        # overwrites it: a resumed pass (a stale in_progress or a failed
        # record for THIS conversation) is expected to find non-empty
        # `existing` facts — the live tail wrote them seconds after the
        # kickoff reply, minutes before the earlier attempt died — and
        # those are not evidence this is no longer a V1 store, just the
        # window between "backfill decided to run" and "backfill finished".
        prior_state = read_state(conv_id)
        resuming = prior_state is not None and (
            prior_state.get("state") == "failed"
            or (prior_state.get("state") == "in_progress" and is_stale(prior_state))
        )
        # v3.1.9.4 (R2 / P15-2 follow-up). This run's own attempt number —
        # carried through every _write_state call below (including the
        # per-exchange progress writes, so a crash mid-run leaves a record
        # that already shows which attempt it was), and what
        # _write_failed_or_abandoned compares against _MAX_BACKFILL_
        # ATTEMPTS. A fresh backfill (no prior record) is attempt 1.
        this_attempt = int((prior_state or {}).get("attempts") or 0) + 1
        async with conv_lock(conv_id):
            try:
                existing = facts_module.load_facts(conv_id)
            except StoreUnreadable as e:
                # The file is there and we cannot read it, so we cannot know
                # what we would be replacing. Refusing costs this conversation
                # its backfill; running would cost it its memory.
                logger.error(
                    f"conv={conv_id}: facts file unreadable ({e}); backfill "
                    f"refused rather than replacing an unknown store with a "
                    f"reconstruction"
                )
                return
            # v3.1.9.4 (R4 / W2 / P15-5 follow-up). Checked under the SAME
            # conv_lock a wipe path bumps and deletes inside (see
            # memory.bump_wipe_generation's own docstring for why that
            # ordering makes a single check here airtight), and as early as
            # possible: this catches the common case — a /forget that ran
            # any time between this run being submitted and it finally
            # getting a turn on the event loop — before a single vLLM call
            # is spent reconstructing history she just asked to forget. A
            # wipe arriving DURING the extraction loop below is caught by
            # the second check, right before this run's own write.
            # _write_wiped logs; nothing else here needs to.
            if wipe_generation is not None and wipe_generation != current_wipe_generation(conv_id):
                _write_wiped(
                    conv_id, this_attempt,
                    started_at=started_at, exchanges_done=0, exchanges_total=0,
                )
                return
        if existing and not resuming:
            logger.info(
                f"conv={conv_id}: backfill refused — {len(existing)} fact(s) "
                f"already on disk; this is not a V1 store"
            )
            return
        if existing and resuming:
            logger.info(
                f"conv={conv_id}: resuming a backfill whose earlier attempt "
                f"left state={prior_state.get('state')!r} "
                f"({prior_state.get('exchanges_done', 0)}/"
                f"{prior_state.get('exchanges_total', 0)} exchanges done) — "
                f"{len(existing)} fact(s) already on disk (from the live "
                f"tail or a prior partial pass) will be merged with, not "
                f"replaced by, this pass's extraction"
            )

        pairs = extract_user_assistant_pairs(messages)
        if not pairs:
            logger.info(f"conv={conv_id}: backfill skipped — no user/assistant pairs found")
            return

        _write_state(conv_id, {
            "state": "in_progress",
            "started_at": started_at,
            "exchanges_done": 0,
            "exchanges_total": len(pairs),
            "attempts": this_attempt,
            "error": None,
        })
        logger.info(
            f"conv={conv_id}: backfill starting over {len(pairs)} "
            f"exchange(s) (attempt {this_attempt}/{_MAX_BACKFILL_ATTEMPTS})"
        )

        async with httpx.AsyncClient() as client:
            for i, (user_text, asst_text) in enumerate(pairs, start=1):
                try:
                    new_strs = await facts_module.extract_facts_from_exchange(
                        client, vllm_url, model, user_text, asst_text, accumulated,
                        # v3.1.9.4 B2 (P15-8 fix, incidental): without this
                        # every backfill extraction call logged
                        # "conv=? (caller passed none)", so a backfill's
                        # load on vLLM could not be attributed to the
                        # conversation causing it.
                        conv_id=conv_id,
                    )
                    for s in new_strs:
                        accumulated.append({
                            "text": s,
                            # added_turn = approximate position in the original
                            # message stream (pair index * 2 for user+assistant)
                            "added_turn": i * 2,
                            "last_used": now_unix,
                        })
                except Exception as e:
                    logger.warning(
                        f"conv={conv_id}: backfill extraction failed on "
                        f"exchange {i}/{len(pairs)}: {e}"
                    )
                # Progress update every exchange (cheap atomic write)
                _write_state(conv_id, {
                    "state": "in_progress",
                    "started_at": started_at,
                    "exchanges_done": i,
                    "exchanges_total": len(pairs),
                    "attempts": this_attempt,
                    "error": None,
                })

        # Done iterating. Re-read INSIDE the lock and merge: the store may
        # have gained facts from any number of tails while we were running,
        # and `accumulated` knows about none of them. The lock alone never
        # protected this — it serializes writers, it cannot undo a read that
        # happened minutes before it was taken (v3.1 F3).
        async with conv_lock(conv_id):
            try:
                on_disk = facts_module.load_facts(conv_id)
            except StoreUnreadable as e:
                logger.error(
                    f"conv={conv_id}: facts file unreadable ({e}); skipped the "
                    f"backfill write rather than replacing the store with "
                    f"{len(accumulated)} reconstructed fact(s)"
                )
                _write_failed_or_abandoned(
                    conv_id, this_attempt,
                    started_at=started_at,
                    exchanges_done=len(pairs),
                    exchanges_total=len(pairs),
                    error=f"facts store unreadable at write time: {e}",
                )
                return
            # v3.1.9.4 (R4 / W2 / P15-5 follow-up). The second, load-bearing
            # check: a wipe can just as easily arrive DURING the (possibly
            # hours-long) extraction loop above as before it. Checked right
            # after the fresh load and before ANY of `merged`/`kept`/the
            # actual save below are computed from `accumulated` — the same
            # "checked right after the load, before anything else runs"
            # placement summarizer._maybe_rollup_body uses for its own
            # generation check, for the identical reason: nothing else may
            # hold conv_lock(conv_id) while this section does, so the wipe's
            # own bump-and-deletes are either already fully done or queued
            # immediately behind this exact lock.
            if wipe_generation is not None and wipe_generation != current_wipe_generation(conv_id):
                _write_wiped(
                    conv_id, this_attempt,
                    started_at=started_at,
                    exchanges_done=len(pairs), exchanges_total=len(pairs),
                )
                return
            merged = _merge_backfilled(on_disk, accumulated)
            added = len(merged) - len(on_disk)
            # conv_id routes eviction to the archive sidecar rather than
            # deleting (v3.1 F9). A backfill merges a whole conversation's
            # history at once, so it is the single call most likely to go over
            # budget — and the facts it would drop are the earliest ones.
            kept, dropped = facts_module.prune_facts(merged, conv_id=conv_id)
            # G2: an empty merge is nothing to say, not a store to erase. A
            # backfill that extracted nothing must not leave an empty facts
            # file behind for list_known_conv_ids to count forever.
            if merged:
                facts_module.save_facts(conv_id, kept)

        # V2.0 Phase 4: also build hierarchical summary state for this conv,
        # so the model gets continuity-of-narrative on the *next* request
        # rather than having to wait for natural rollups (which require new
        # turns to accumulate). maybe_rollup drains as many L1→L2→L3 layers
        # as the existing message history justifies. Failure is non-fatal:
        # facts backfill is still considered complete.
        try:
            if summarizer.enabled():
                # `messages` has been redacted by start_backfill_if_needed;
                # `raw_messages` is the same array snapshotted before
                # redaction, and each chunk writes its covered-turn record
                # from it (summarizer._record_chunk_fps). With None (a direct
                # caller) the chunks record the redacted text they read, so a
                # redacted reply is summarized fresh later rather than reused:
                # it costs a summarization call and never a turn.
                #
                # hostile317-b F3: `messages`/`raw_messages` were snapshotted
                # at kickoff, and this call lands MINUTES later. By then a
                # live tail has usually rolled this conversation up already —
                # in production order ALWAYS: the kickoff request's own reply
                # streams in seconds, and its tail drains the whole hierarchy
                # over the full array plus that reply (hostile pass #3,
                # reviewer A F8). `_observed_position` trusts whatever array
                # it is handed as the CURRENT window and overwrites the anchor
                # with its tail, so a stale snapshot would pin the anchor to
                # old content and put every later chunk label one turn off.
                # The live tail owns the hierarchy; this rollup runs only when
                # no tail has moved the conversation past the snapshot (the
                # kickoff reply was skipped, shed, or never produced).
                #
                # THE CHECK IS MADE UNDER THE LOCK THE ROLLUP HOLDS (hostile
                # pass #3, reviewer E F3). It was read here, outside conv_lock,
                # and then maybe_rollup took the lock: a live tail queued
                # behind this backfill's facts merge ran in between, saved its
                # position, and the stale snapshot then rolled up over it.
                # maybe_rollup now compares the recorded position with
                # `snapshot_turns` itself, after loading state under the lock,
                # and reports the skip through `skipped`.
                snapshot_turns = sum(1 for m in messages if m.get("role") != "system")
                skipped: list[int] = []
                # v3.1.9 (tail catch-up): bounded the same way the live tail
                # is now — see _TAIL_ROLLUP_MAX_CALLS' own comment for why
                # this single call needed a budget at all. A fresh dict per
                # backfill, same as the tail: this is a one-shot call, so
                # there is no "next turn" of this SAME call to carry
                # anything forward into anyway.
                _budget = {"remaining": _TAIL_ROLLUP_MAX_CALLS, "exhausted": False}
                # v3.1.9 (hostile follow-up). The SAME before/after watermark
                # record the tail makes — see summarizer.record_catchup_pass's
                # own docstring for why this has to be process-local evidence
                # rather than something health.py re-derives by polling. A
                # cheap extra read (this function already does several),
                # taken just before the call so it reflects this pass's own
                # starting point rather than an earlier snapshot from higher
                # up in this function.
                _before_state = summarizer.load_state(conv_id)
                # v3.1.9.4 (R4 / W2 / P15-5 follow-up). Same contextvar
                # mechanism main._rollup_hierarchy uses for the live tail —
                # summarizer._maybe_rollup_body already reads it back, under
                # its own conv_lock, right after loading state, and discards
                # the WHOLE rollup on a mismatch (no tier check, no LLM call,
                # no save_state). Before this fix this call always left the
                # contextvar at its default of None, i.e. opted out — this
                # was the one background writer R1 did not cover: a backfill
                # is submitted via the same fire_and_forget the live tail
                # uses and can run for hours, not "a request an operator is
                # waiting on" (which is what None correctly means for
                # main.admin_compact, wipe_generation_ctx's own docstring's
                # other example).
                with summarizer.wipe_generation_ctx(wipe_generation):
                    _rollup_state = await summarizer.maybe_rollup(
                        conv_id, messages, vllm_url, model, raw_messages=raw_messages,
                        skip_if_position_past=snapshot_turns, skipped_at=skipped,
                        vllm_call_budget=_budget,
                    )
                try:
                    summarizer.record_catchup_pass(
                        conv_id,
                        _before_state.get("last_summarized_turn", 0),
                        _rollup_state.get("last_summarized_turn", 0),
                        summarizer.needs_rollup(
                            _rollup_state, _rollup_state.get("turns_seen", 0)
                        ),
                    )
                except Exception as e:
                    logger.warning(
                        f"conv={conv_id}: could not record backfill's "
                        f"catch-up progress ({type(e).__name__}: {e}) — the "
                        f"rollup pass above already completed regardless"
                    )
                if _budget["exhausted"]:
                    logger.info(
                        f"conv={conv_id}: backfill's summary rollup spent its "
                        f"{_TAIL_ROLLUP_MAX_CALLS}-call budget with more of "
                        f"the hierarchy still behind; the live tail on this "
                        f"conversation's next turn continues the catch-up "
                        f"from the persisted watermark"
                    )
                if skipped:
                    logger.info(
                        f"conv={conv_id}: backfill's summary rollup skipped — "
                        f"a live tail already rolled this conversation up to "
                        f"turn {skipped[0]}, past this backfill's snapshot "
                        f"({snapshot_turns} turns, taken at kickoff). In "
                        f"production order that is the kickoff request's own "
                        f"reply, and it is the expected case: the live tail "
                        f"owns the hierarchy. Rolling up this snapshot would "
                        f"overwrite the position anchor with a stale tail "
                        f"(hostile317-b F3)."
                    )
        except Exception as e:
            logger.warning(f"conv={conv_id}: backfill summary rollup failed (non-fatal): {e}")

        _write_state(conv_id, {
            "state": "complete",
            "started_at": started_at,
            "exchanges_done": len(pairs),
            "exchanges_total": len(pairs),
            "facts_kept": len(kept),
            "facts_pruned": dropped,
            # Distinguishes what the backfill contributed from what tails
            # wrote underneath it while it ran — the two used to be
            # indistinguishable because the second set was gone (v3.1 F3).
            "facts_added": added,
            # v3.1.9.4 (R2): informational only once state is "complete" —
            # needs_backfill's "complete" branch never looks at it — but a
            # backfill that took two or three tries to finally succeed
            # should still SAY that, not read identically to one that
            # finished on its first attempt.
            "attempts": this_attempt,
            "error": None,
        })
        logger.info(
            f"conv={conv_id}: backfill complete — {added} fact(s) added to "
            f"{len(on_disk)} already on disk, {len(kept)} kept, {dropped} "
            f"pruned, from {len(pairs)} exchanges (attempt {this_attempt})"
        )
    except Exception as e:
        logger.exception(f"conv={conv_id}: backfill aborted: {e}")
        _write_failed_or_abandoned(
            conv_id, this_attempt,
            started_at=started_at,
            exchanges_done=0,
            exchanges_total=len(pairs),
            error=str(e),
        )
    finally:
        _in_progress_local.discard(conv_id)


async def start_backfill_if_needed(
    conv_id: str,
    messages: list[dict],
    vllm_url: str,
    model: str,
    *,
    fire_and_forget,
    redact=None,
) -> bool:
    """Public entry point. Returns True if a backfill was started,
    False if it wasn't needed.

    `fire_and_forget` is the caller's task spawner (main.py's
    _fire_and_forget) so this module doesn't need to know about the
    background-task registry. Decouples from main.py for testing.

    `redact` is applied to `messages` ONLY once we have decided to run, and
    it is a callable rather than pre-redacted messages for a measured
    reason: needs_backfill() returns False for every conversation that
    already has a facts file, which is every established conversation
    forever, while this function is called on every request. Redacting in
    the caller's argument list therefore ran a full-history degeneracy scan
    inline before the vLLM call on every single turn - 366 ms at 170 turns,
    1.6 s at 1000 - and threw the result away every time. Same
    module-boundary reason as fire_and_forget: main.py owns the detector,
    this module owns when the work happens.
    """
    if not needs_backfill(conv_id, messages):
        return False
    if conv_id in _in_progress_local:
        return False  # already started in this process
    _in_progress_local.add(conv_id)
    # The array exactly as the client sent it, snapshotted BEFORE redaction:
    # the covered-turns digest must be folded from turns the next request will
    # carry, and redaction replaces degenerate replies with placeholders that
    # no request ever does (see summarizer._record_chunk_fps).
    raw_snapshot = [dict(m) for m in messages]
    if redact is not None:
        # Off the event loop: this is an async function awaited on the
        # request path, and the redactor walks every historical assistant
        # turn through a regex detector - measured 701ms at 1,000 turns.
        # The other two redaction sites (live tail, admin compact) both got
        # run_in_threadpool; this one was the thirteenth sibling miss.
        messages = await run_in_threadpool(redact, messages)
    # Snapshot messages — caller may mutate the list before backfill runs
    snapshot = [dict(m) for m in messages]
    # v3.1.9.4 (R4 / W2 / P15-5 follow-up). Captured synchronously, right
    # before the coroutine is handed to fire_and_forget — the same
    # submission-time convention main._run_memory_tail's own capture uses,
    # and for the identical reason: this is the last instant before the run
    # becomes a background task that can outlive whatever happens next on
    # the request path, including a /forget that arrives while it is still
    # running. A capture taken earlier in this function (e.g. before the
    # `redact` await above) would only widen the window this protects,
    # never narrow it — current_wipe_generation only moves forward, so
    # comparing against an earlier snapshot just means a wipe that landed
    # during the await is caught too, not missed.
    wipe_generation = current_wipe_generation(conv_id)
    fire_and_forget(
        _run_backfill(
            conv_id, snapshot, vllm_url, model,
            raw_messages=raw_snapshot, wipe_generation=wipe_generation,
        )
    )
    return True
