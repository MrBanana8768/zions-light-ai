"""
compactor.tokenhealth — shared /tokenize failure/success accounting.

Three call sites in this process ask vLLM's /tokenize for a count:
main.count_tokens_exact (chat-shaped, request hot path),
main.count_text_tokens_exact (completion-shaped, injected-block sizing), and
summarizer._count_tokens (completion-shaped, rollup input budgeting). main.py
already carries real accounting for its own two — _tokenize_fail_streak /
_tokenize_text_fail_streak, tokenize_health(), consumed by /health/full (v3.1
A13) — because a /tokenize outage degrades every budget in the process to a
local estimate that reads up to 51% low on this model's assistant content,
and that is precisely the failure mode two production incidents ran in while
every health surface said "ok".

summarizer.py had none of that. Its /tokenize failures went through
`logsetup.log_once` — a gate that reports ONE line for the lifetime of the
process and then nothing, ever again, no matter how long the outage runs.
An outage that started after that first line was invisible everywhere: not
in the log, and not in /health/full, because summarizer's failures never fed
the counters that endpoint reads.

This module holds the counted/streak state generically, keyed by a caller-
chosen `source` name, so any /tokenize call site gets the same real
accounting main.py's request-path counters already have. summarizer.py is
wired to it (see summarizer._count_tokens and summarizer.tokenize_health()).

Deliberately NOT a logging module. `note_failure`/`note_success` return a
message string (or None) for the CALLER to hand to its own `logger.warning`,
rather than logging here directly, for two reasons:

  1. Logger identity. A record logged via `logging.getLogger(__name__)`
     inside this module would propagate up through "compactor" to the root
     logger, but NEVER through "compactor.summarizer" or "compactor.main" —
     those are siblings of "compactor.tokenhealth" in the dotted hierarchy,
     not ancestors of it. Every test in this codebase that asserts on log
     content scopes its capture to the CALLING module's own logger name, so
     a warning "from" tokenhealth would be silently invisible to exactly the
     tests meant to prove it fires.
  2. One clock, not per-caller ad-hoc ones. Rate-limiting is done here via
     logsetup.log_once with a KEY THAT ROTATES every `warn_interval_s`
     seconds, rather than this module keeping its own last-seen-at map:
     log_once's memory is normally for the life of the process — the exact
     defect being fixed here — but a key that changes every window turns
     "once ever" into "once per window": silenced within a window,
     guaranteed to speak again once the window rolls over, for as long as
     the degradation lasts. It also means the one reset hook this codebase's
     tests already call (logsetup._reset_log_once_for_tests) clears this
     rate limit too, with no second reset path for a test to forget.

main.py reads this module's state indirectly — main.py is owned by a sibling
change in this remediation pass, and this module is deliberately additive so
it can land without touching main.py. See `source_health()` below for
exactly what main.py would need to call to actually share this state.
"""

from __future__ import annotations

import time

import logsetup

# Same default as main.py's COMPACTOR_TOKENIZE_WARN_INTERVAL_S (main.py:693).
# Not re-read from the environment here: a caller that wants the operator's
# configured interval passes it in (see summarizer.TOKENIZE_WARN_INTERVAL_S),
# so there is exactly one place per process that parses the env var, and this
# module stays a pure accounting primitive.
DEFAULT_WARN_INTERVAL_S = 300.0


class _Tracker:
    """Per-source /tokenize failure state."""

    __slots__ = ("fail_streak", "last_fail_at", "degraded_since")

    def __init__(self) -> None:
        self.fail_streak = 0
        self.last_fail_at: float | None = None
        self.degraded_since: float | None = None


_trackers: dict[str, _Tracker] = {}


def _get(source: str) -> _Tracker:
    t = _trackers.get(source)
    if t is None:
        t = _Tracker()
        _trackers[source] = t
    return t


def note_failure(
    source: str,
    key: str,
    detail: str,
    *,
    warn_interval_s: float = DEFAULT_WARN_INTERVAL_S,
) -> str | None:
    """Record one /tokenize failure for `source`. Always updates the streak
    (for source_health / /health/full); returns a ready-to-log message if a
    line is due THIS call, else None.

    `key` separates failure CLASSES — a benign template refusal must not
    spend the same rate-limit budget a connection error needs, same doctrine
    as main's own _note_tokenize_failure. `detail` is used VERBATIM as the
    start of the returned message (the caller's existing wording, and any
    substring it is tested for, survives unchanged); this only appends the
    streak/degraded-for accounting after it.
    """
    t = _get(source)
    now = time.time()
    t.fail_streak += 1
    t.last_fail_at = now
    if t.degraded_since is None:
        t.degraded_since = now
    bucket = int(now // warn_interval_s) if warn_interval_s > 0 else 0
    if not logsetup.log_once(f"tokenhealth.{source}.{key}.{bucket}"):
        return None
    return (
        f"{detail} ({t.fail_streak} consecutive failure(s) on '{source}', "
        f"degraded for {now - t.degraded_since:.0f}s)"
    )


def note_success(source: str) -> str | None:
    """Clear the degraded state for `source`. Returns a recovery message if
    `source` was actually degraded, else None (mirrors main's
    _note_tokenize_success, including its no-op-when-already-healthy case).
    """
    t = _trackers.get(source)
    if t is None or t.fail_streak == 0:
        return None
    failures = t.fail_streak
    since = t.degraded_since
    t.fail_streak = 0
    t.degraded_since = None
    t.last_fail_at = None
    return (
        f"/tokenize ({source}) is answering again after {failures} "
        f"consecutive failure(s)"
        + (f" over {time.time() - since:.0f}s" if since is not None else "")
        + "."
    )


def source_health(source: str, *, stale_after_s: float | None = None) -> dict:
    """Streak/staleness snapshot for one `source`, main.tokenize_health()
    -shaped so a caller can merge it into that dict's own fields.

    `stale_after_s`: for a source that is not exercised on every request
    (summarizer's rollups; main's text form only fires when there is memory
    to inject) staleness matters — see main._text_tokenize_failing_now's own
    doctrine, reproduced here: a streak whose last failure is older than
    this is reported as recovered-but-unconfirmed rather than as an ongoing
    fault, because "we have not been asked" and "it started working again"
    are indistinguishable from outside, and asserting a fault that cannot
    currently be observed is the same error as asserting health that cannot
    be observed. Pass None (the default) for a source called on every
    request, where a standing streak IS the fact.

    WHAT main.py WOULD NEED TO DO TO ACTUALLY SHARE THIS STATE
    ------------------------------------------------------------
    Today main.py keeps its own module-level _tokenize_fail_streak /
    _tokenize_text_fail_streak and does not call into this module at all, so
    "the same counted/streak health state main uses" is realized only on
    summarizer's side of the wire. To close the loop, main.py's
    tokenize_health() would need to:

      1. Route count_tokens_exact / count_text_tokens_exact's failure and
         success calls through note_failure(source="main.chat", ...) /
         note_failure(source="main.text", ...) and the matching
         note_success(...), in place of its own _note_tokenize_failure /
         _note_tokenize_success / _note_text_tokenize_failure /
         _note_text_tokenize_success — so all three /tokenize call sites in
         the process share ONE implementation of the streak/degraded-since/
         rate-limit bookkeeping instead of three copies of it.
      2. Fold in summarizer's source — `summarizer.tokenize_health()` below,
         or equivalently `tokenhealth.source_health("summarizer",
         stale_after_s=summarizer.TOKENIZE_WARN_INTERVAL_S)` — into the dict
         tokenize_health() returns and into the `ok` / `consecutive_failures`
         aggregation, the same way it already ANDs/maxes chat-form and
         text-form today.

    DONE as of the same diff: main._summarizer_tokenize_failing_now() and
    main._summarizer_degraded_since() fold summarizer's source into
    /health/full's `ok`, `consecutive_failures`, `degraded_since` and
    `degraded_for_s`. What remains of step 1 above is the de-duplication -
    main.py still keeps its own two counters rather than routing them
    through this module - which is a tidiness item, not a visibility gap.
    """
    t = _trackers.get(source)
    if t is None:
        return {
            "ok": True,
            "consecutive_failures": 0,
            "degraded_since": None,
            "degraded_for_s": 0.0,
        }
    streak = t.fail_streak
    stale = (
        stale_after_s is not None
        and streak > 0
        and t.last_fail_at is not None
        and (time.time() - t.last_fail_at) > stale_after_s
    )
    effective = 0 if stale else streak
    return {
        "ok": effective == 0,
        "consecutive_failures": effective,
        "degraded_since": None if stale else t.degraded_since,
        "degraded_for_s": (
            round(time.time() - t.degraded_since, 1)
            if (not stale and t.degraded_since is not None)
            else 0.0
        ),
    }


def _reset_for_tests(source: str | None = None) -> None:
    """Test helper — drop tracked state so tests don't leak streaks into each
    other. `source=None` clears everything. Does NOT touch logsetup's
    log_once registry — callers that also depend on that reset it
    separately via logsetup._reset_log_once_for_tests()."""
    if source is None:
        _trackers.clear()
    else:
        _trackers.pop(source, None)
