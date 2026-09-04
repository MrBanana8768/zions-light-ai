"""
compactor.tailhealth — memory-tail decision accounting, for /health/full.

Every reply that finishes goes through decide_memory_tail (main.py), which
says whether the exchange enters memory — facts, episodic index, summary
rollup — and, for a reply that was cut, how much of it. Until v3.1.4 a skip
there was a WARNING line and nothing else: no counter, no health field, no
alert. Only four mechanisms in the whole package record a skipped operation
anywhere but the log, and this was not one of them — which is how 63
exchanges in one 2026-09-01 log window (51 stopped by hand, 12 cut at the
generation ceiling) were discarded from memory for weeks while /health/full
said ok. With this module those 63 would have shown as `degraded`, with the
count and the reason.

Why its own module: health.py cannot import main (main imports health), so
the counter has to live somewhere both can reach. bgwork is the precedent —
health.py reads `bgwork.pool.stats()` directly, and reads `snapshot()` here
the same way.

Modelled on tokenhealth, including its reason for RETURNING a string rather
than logging: a record logged under "compactor.tailhealth" propagates to the
root logger but never through "compactor.main" — siblings, not ancestors —
so every test that scopes its capture to the calling module's logger would
be blind to it. `note()` returns an accounting suffix; the caller
(main._run_memory_tail) puts it on its own line, under its own logger.

`skipped_recently` is WINDOWED, copying bgwork.SHED_DEGRADE_WINDOW_S's
argument: the cumulative count is for the life of the process, so degrading
health on `skipped > 0` would pin the endpoint to "degraded" from the first
skip until the next restart — and a warning that is always on is a warning
nobody reads. Skips degrade while they are happening and for one window
after, then clear themselves. The cumulative counters stay in the payload
as the record.

`skipped_recently` is also keyed off LOSSY_SKIP_OUTCOMES, not off "any skip"
(v3.1.7, R27). SKIPPED_EMPTY — she pressed Stop before the first token — is
counted like every other skip but carries no loss: there was no text to
memorize. Keying the degrade decision off "any skip" made that the SAME
always-on-warning failure the paragraph above already fixed once, arriving
through the other door: the release's own figure is 51 of 63 skips in one
window were manual stops, so a run of those alone used to pin /health/full
degraded for the full window on every one of them, over a turn that lost
nothing.

No conversation text and no conv_id reaches this module — only outcome
labels and character counts. /health/full is not localhost-gated the way
the /admin endpoints are (bgwork.submit's docstring makes the same point
about conv_ids), and a degeneracy reason from reply_is_degenerate can quote
a 24-character token from the reply, so the human-readable reason stays in
the log and only the machine label is counted here.
"""

from __future__ import annotations

import math
import os
import time

# Same default and the same reasoning as bgwork.SHED_DEGRADE_WINDOW_S: 300 s
# spans ten consecutive 30 s HEALTHCHECK probes, so a skip between two looks
# still shows on the next one.
def _window_s(name: str, default: float) -> float:
    """Read a degrade-window seconds value from the environment, safely.

    `float(os.environ.get(name, "300") or 300)` — copied from bgwork, where
    it has now been fixed too — is two failures in one line, and this module
    is imported at main.py module scope, so both are BOOT failures:

      * UNPARSEABLE. `or 300` rescues only the empty string, so one typo in
        runpod.env raises ValueError at import and the compactor never
        starts.
      * NON-POSITIVE. `0` or a negative parses fine and then silently
        disables this module's whole purpose: `skipped_recently` is
        `since <= window`, never true, so /health/full says ok while the
        memory tail is skipping — the regression this release exists to
        catch, disabled by a config value nobody would look at twice.

    main._env_float does NOT reject a zero or a negative (its docstring
    once said otherwise; corrected in v3.1.7), and this module cannot import
    main anyway, so the rule lives here.
    """
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return default
    # isfinite, not just > 0. `inf` parses, satisfies `v > 0`, and then pins
    # the "recently" flag True from the first event until the process
    # restarts — a warning that is always on is a warning nobody reads, which
    # is the failure this window exists to prevent, arriving through the one
    # config value whose whole job is to prevent it. `nan` fails `> 0`
    # already; this makes the rejection explicit rather than incidental.
    return v if math.isfinite(v) and v > 0 else default


SKIP_DEGRADE_WINDOW_S = _window_s("COMPACTOR_TAIL_SKIP_DEGRADE_WINDOW_S", 300.0)

# Machine outcome labels. decide_memory_tail returns one of the first eight per
# decision; the first two store, the rest skip.
STORED = "stored"                                    # finished; verbatim
STORED_TRIMMED = "stored_trimmed"                    # cut; trimmed prefix stored
SKIPPED_HOLED = "skipped_holed"                      # accumulator dropped a chunk
SKIPPED_EMPTY = "skipped_empty"                      # no assistant text at all
SKIPPED_DEGENERATE = "skipped_degenerate"            # finished; repetition loop
SKIPPED_NO_BOUNDARY = "skipped_no_boundary"          # cut; no sentence survives
SKIPPED_TOO_SHORT = "skipped_too_short"              # cut; trimmed under the floor
SKIPPED_DEGENERATE_PARTIAL = "skipped_degenerate_partial"  # cut; trimmed text loops
# v3.1.7 (R8). The two labels above this line are decisions about the REPLY;
# these two are about whether the store can be reached at all, and they are
# decided in main._run_memory_tail immediately after decide_memory_tail says
# "store". Until v3.1.7 both conditions were evaluated INSIDE the fired tail,
# after `stored` had already been counted, so /health/full reported a healthy
# tail for an exchange that never reached memory — the same class this module
# was built to end, one layer up.
SKIPPED_DISK_PRESSURE = "skipped_disk_pressure"      # degrade.guard says no
SKIPPED_NO_USER_TEXT = "skipped_no_user_text"        # nothing to pair the reply with

# v3.1.8 (N4b). OpenWebUI's title/tag/follow-up generation, arriving on a
# stable conv_id we have already stored an exchange for. Not a loss and not a
# fault: it is a request we deliberately decline to memorize, and counting it
# is what stops it looking like one of the skips above.
SKIPPED_TASK_TRAFFIC = "skipped_task_traffic"        # background task, not a conversation
STORING_OUTCOMES = frozenset({STORED, STORED_TRIMMED})
OUTCOMES = (
    STORED, STORED_TRIMMED, SKIPPED_HOLED, SKIPPED_EMPTY, SKIPPED_DEGENERATE,
    SKIPPED_NO_BOUNDARY, SKIPPED_TOO_SHORT, SKIPPED_DEGENERATE_PARTIAL,
    SKIPPED_DISK_PRESSURE, SKIPPED_NO_USER_TEXT, SKIPPED_TASK_TRAFFIC,
)

# The skips that cost the user NOTHING, named once.
#
# There are two consumers — LOSSY_SKIP_OUTCOMES just below, and the
# `skipped_recently` decision in note() — and until v3.1.8 each spelled the
# rule out for itself as "... and outcome != SKIPPED_EMPTY". One harmless
# label is a coincidence; two is a list, and a list written twice is the
# fix-one-site-miss-the-sibling defect this project keeps paying for. Add a
# harmless label HERE and both consumers follow.
HARMLESS_SKIP_OUTCOMES = frozenset({SKIPPED_EMPTY, SKIPPED_TASK_TRAFFIC})

# v3.1.7 (R27). SKIPPED_EMPTY is the one skip label that carries no loss: she
# pressed Stop before the first token, so there was never any text to
# memorize. Every OTHER skip means a reply she read did not reach memory.
# Computed by exclusion, not by naming the five lossy labels, so a skip
# outcome this module has never heard of — including one `note()`'s
# unknown-label branch below is about to count for the first time — is lossy
# BY DEFAULT. The 2026-09-01 incident this module exists to prevent was 63
# skips in one window, 51 of them manual stops (SKIPPED_EMPTY): keying the
# degrade decision off "any skip" instead of this set pinned /health/full
# degraded for the full SKIP_DEGRADE_WINDOW_S on every one of those 51, for a
# turn that lost nothing.
#
# The exclusion is why R8's two new labels needed no edit here: a reply she
# READ that did not reach memory is a loss whether the reason was the reply
# (degenerate, holed) or the store (disk pressure, no user text to pair it
# with), and adding them to OUTCOMES alone puts both on the lossy side.
LOSSY_SKIP_OUTCOMES = (
    frozenset(OUTCOMES) - STORING_OUTCOMES - HARMLESS_SKIP_OUTCOMES
)


class _State:
    __slots__ = (
        "outcomes", "stored", "skipped", "raw_chars", "kept_chars",
        "trimmed_raw_chars", "trimmed_kept_chars", "consecutive_skips",
        "last_skip_at", "last_skip_outcome", "last_lossy_skip_at",
    )

    def __init__(self) -> None:
        self.outcomes: dict[str, int] = {o: 0 for o in OUTCOMES}
        self.stored = 0
        self.skipped = 0
        # Over EVERY decision: what arrived, and what went to the store.
        self.raw_chars = 0
        self.kept_chars = 0
        # Over the trim path's STORED decisions only, so the retention ratio
        # measures what trimming costs a reply that was kept — not the skip
        # rate, which the counts above already carry. The plan for v3.1.4
        # declined a relative floor ("keep only if >= X% survived") because it
        # would fire hardest on the runaway replies that are most of the loss;
        # this pair is what makes that a measured number instead of a hunch.
        self.trimmed_raw_chars = 0
        self.trimmed_kept_chars = 0
        self.consecutive_skips = 0
        # Monotonic, like bgwork._last_shed_at: this feeds a "how long ago"
        # that must not jump when the clock is stepped.
        self.last_skip_at: float | None = None
        self.last_skip_outcome: str | None = None
        # v3.1.7 (R27). Tracked separately from last_skip_at: `skipped_recently`
        # must clear itself when the only skip in the window was SKIPPED_EMPTY,
        # even while last_skip_at (any skip, kept as the general record) is
        # still fresh.
        self.last_lossy_skip_at: float | None = None


_state = _State()


def _safe_int(v) -> int:
    """`int(v)`, clamped at 0, that never raises.

    v3.1.7 (R28). The two call sites today always pass real ints, but this
    module's whole reason for existing is a decision made in the request
    path's `finally` (see the module docstring's "must not raise" contract) —
    and until this fix only the OUTCOME label was guarded against that. A
    `None` or a non-numeric string reaching `raw_chars`/`kept_chars` went
    through a bare `int()` and raised out of the finally, which is the exact
    second failure the docstring already claimed could not happen.
    """
    try:
        return max(0, int(v))
    except (TypeError, ValueError):
        return 0


def note(outcome: str, *, raw_chars: int, kept_chars: int) -> str | None:
    """Record one memory-tail decision.

    `outcome` is one of OUTCOMES (an unknown label is counted under its own
    name rather than raising — this runs in the request path's `finally`,
    where a bookkeeping error must not become a second failure). `raw_chars`
    is the length of the text as it arrived; `kept_chars` the length of what
    was handed to the store, 0 for a skip.

    Returns an accounting suffix for the caller's OWN log line when the
    decision was a skip — "3 consecutive memory-tail skip(s)" — and None for
    a store. Never logs; see the module docstring for why.
    """
    s = _state
    raw = _safe_int(raw_chars)
    kept = _safe_int(kept_chars)
    s.raw_chars += raw
    storing = outcome in STORING_OUTCOMES
    if storing:
        s.stored += 1
        s.kept_chars += kept
        if outcome == STORED_TRIMMED:
            s.trimmed_raw_chars += raw
            s.trimmed_kept_chars += kept
        s.consecutive_skips = 0
        result = None
    else:
        s.skipped += 1
        s.consecutive_skips += 1
        s.last_skip_at = time.monotonic()
        s.last_skip_outcome = outcome
        # v3.1.7 (R27). SKIPPED_EMPTY does not touch this clock, so a skip
        # storm of manual stops cannot make `skipped_recently` true — see
        # LOSSY_SKIP_OUTCOMES above.
        #
        # Tested as "not a store and not the one harmless label", NOT as
        # membership of LOSSY_SKIP_OUTCOMES. Those read the same for every
        # label this module knows and OPPOSITELY for one it does not: a set
        # built by excluding from OUTCOMES cannot contain a name that is not
        # in OUTCOMES, so `outcome in LOSSY_SKIP_OUTCOMES` silently makes an
        # unrecognised outcome NON-lossy — the exact inversion of the safe
        # default the comment on that set claims, and of note()'s own
        # unknown-label contract two branches up. An outcome added in main.py
        # and forgotten here would then be counted and never degrade health,
        # which is the silent-skip shape this whole module exists to end.
        # v3.1.8 (N4b): the harmless labels are a SET now, named once in
        # HARMLESS_SKIP_OUTCOMES, so this test and LOSSY_SKIP_OUTCOMES cannot
        # disagree about which skips cost the user nothing. Still written as
        # exclusion from STORING_OUTCOMES, never as membership of
        # LOSSY_SKIP_OUTCOMES, for the reason above.
        if (outcome not in STORING_OUTCOMES
                and outcome not in HARMLESS_SKIP_OUTCOMES):
            s.last_lossy_skip_at = time.monotonic()
        result = f"{s.consecutive_skips} consecutive memory-tail skip(s)"
    # v3.1.7 (R28). The outcome tally is the LAST mutation in this function,
    # on purpose: everything above it is now raise-proof (_safe_int), but if
    # some future line between here and the top ever raises anyway, the
    # published block must fail with the outcome NOT yet counted rather than
    # counted without a matching stored/skipped increment — the direction
    # that keeps `sum(outcomes.values()) == stored + skipped`
    # (test_saturation.py's reconciliation) from going stale rather than
    # going wrong.
    s.outcomes[outcome] = s.outcomes.get(outcome, 0) + 1
    return result


def snapshot(*, window_s: float | None = None) -> dict:
    """For /health/full — the memory_tail block. `skipped_recently` is the
    field health's status reads; everything else is the record.

    A copy: mutating the returned dict changes nothing here.
    """
    s = _state
    # Through the SAME guard as the env value. Passing window_s=0 explicitly
    # used to bypass _window_s and make `since <= 0` true for a skip that had
    # just happened — so 0 was MORE alarming than the 300 s default while -5
    # was less, three meanings for one parameter. A caller cannot ask for a
    # window this module's own doctrine rejects.
    window = SKIP_DEGRADE_WINDOW_S
    if window_s is not None:
        try:
            v = float(window_s)
            if math.isfinite(v) and v > 0:
                window = v
        except (TypeError, ValueError):
            pass
    since = (
        None if s.last_skip_at is None
        else round(time.monotonic() - s.last_skip_at, 1)
    )
    # v3.1.7 (R27). Separate clock from `since` above: a SKIPPED_EMPTY skip
    # (Stop before the first token) advances `since` but not this one, so it
    # cannot make `skipped_recently` true on its own — see LOSSY_SKIP_OUTCOMES.
    # `since` and `last_skip_outcome`/`consecutive_skips` stay keyed off ANY
    # skip: they are the general record ("the cumulative counters stay in the
    # payload as the record" — module docstring), not the degrade signal.
    since_lossy = (
        None if s.last_lossy_skip_at is None
        else round(time.monotonic() - s.last_lossy_skip_at, 1)
    )
    return {
        "stored": s.stored,
        "skipped": s.skipped,
        "outcomes": dict(s.outcomes),
        "consecutive_skips": s.consecutive_skips,
        "last_skip_outcome": s.last_skip_outcome,
        "seconds_since_last_skip": since,
        "seconds_since_last_lossy_skip": since_lossy,
        "skipped_recently": since_lossy is not None and since_lossy <= window,
        "skip_window_s": window,
        "raw_chars": s.raw_chars,
        "kept_chars": s.kept_chars,
        "trimmed_raw_chars": s.trimmed_raw_chars,
        "trimmed_kept_chars": s.trimmed_kept_chars,
        # None, not 0.0, when nothing has been trimmed yet: "we have not
        # measured" must stay distinguishable from "everything was lost".
        "trim_retention": (
            round(s.trimmed_kept_chars / s.trimmed_raw_chars, 3)
            if s.trimmed_raw_chars else None
        ),
    }


def _reset_for_tests() -> None:
    """Drop all state so tests do not leak streaks into each other."""
    global _state
    _state = _State()
