"""
compactor.reqctx — per-request id propagation (v3.1.9.8 REQUEST TRACING).

The owner's ask: "enhance our logging with a guid to keep track of requests
through the entire process from sent to response". This module is the one
place that owns:

  - minting/validating the request id (an incoming `X-Request-Id` if it is
    sane, else a uuid4 hex);
  - a ContextVar holding it, so `logsetup`'s logging Filter can stamp
    `req=<id>` on every log line emitted anywhere in that request's context
    — including a background tail (bgwork.py sets `bg_tag_ctx` around the
    tail's execution) and a `run_in_threadpool`/anyio hop (contextvars
    propagate through both by default; test_request_tracing.py asserts it
    rather than trusting the platform silently);
  - purpose-tagged child ids for internal vLLM calls (`<id>.sum1`,
    `<id>.facts1`, ...), via `vllm_headers()`;
  - the process-wide requests-in-flight / last-done-age counters `/health/
    full` reports (a real incident: an operator restart killed a request in
    its ~100s pre-processing phase because nothing exposed it was even
    running).

NEVER carries message text, user names or emails — ids and numbers only.
This module in particular must stay that way: it is imported by every
vLLM-calling module in the package (facts, dedup, continuity, summarizer,
main), so anything unsafe put here would be unsafe everywhere at once.
"""

from __future__ import annotations

import contextvars
import re
import time
import uuid

# A sane incoming X-Request-Id: hex/uuid-ish, <=64 chars. Deliberately
# narrow (not "any header value") — this string is logged on every line for
# the life of the request and sent to vLLM as a header value, so a client
# that sent something absurd (log-injection newlines, control characters, a
# multi-KB header) gets a fresh id instead of having it echoed back into our
# own logs and outbound requests.
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

# The request id for the request currently executing in this context.  Set
# by the chat_completions wrapper (main.py) for the request path; also
# copied automatically into any asyncio.Task created from inside that
# context (asyncio.create_task snapshots the current context), which is
# what lets bgwork's background tails and run_in_threadpool/anyio hops see
# it with no extra plumbing.
request_id_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "compactor_request_id", default=None
)

# Set ONLY around a background tail's own execution (bgwork.BackgroundPool.
# _run), never on the request path itself. When set, the logging Filter
# renders `req=<id>/bg:<tag>` instead of bare `req=<id>` — see bgwork.py.
bg_tag_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "compactor_bg_tag", default=None
)

# Per-request counters for purpose-tagged vLLM child ids (`.sum1`, `.sum2`,
# ...). Reset implicitly every request: chat_completions never reuses a
# contextvar token across requests, so each request's `Context` gets its
# own fresh dict the first time vllm_headers(purpose=...) is called in it.
_purpose_counters_ctx: contextvars.ContextVar[dict[str, int] | None] = (
    contextvars.ContextVar("compactor_purpose_counters", default=None)
)


def sane_incoming_id(value: str | None) -> str | None:
    """The incoming value if it looks like a real request id, else None."""
    if not value:
        return None
    value = value.strip()
    if not value or len(value) > 64:
        return None
    return value if _ID_RE.match(value) else None


def new_request_id(incoming: str | None) -> str:
    """Honour a sane incoming id, else mint a uuid4 hex. Does not set the
    contextvar — the caller does that with `.set()` so it can hold the
    token and reset it when the request ends."""
    sane = sane_incoming_id(incoming)
    return sane if sane is not None else uuid.uuid4().hex


def current_request_id() -> str | None:
    return request_id_ctx.get()


def init_purpose_counters() -> None:
    """Create the per-request purpose-counter dict NOW, in the CALLING
    context — call this once, in chat_completions' wrapper, right after
    `request_id_ctx` is set and BEFORE any background task can be spawned
    from this request (hostile review round 2, L3).

    Without this, the dict is created LAZILY, on whichever call to
    `vllm_headers(purpose=...)` happens first. `asyncio.create_task`
    (inside bgwork.pool.submit) snapshots a COPY of the current context at
    spawn time, so the memory tail (fires `.facts*` calls) and the
    hierarchy rollup (`.sum*` calls) — two SEPARATE background tasks,
    forked from the same request — each got their OWN independent copy of
    `_purpose_counters_ctx`. If neither has been touched yet at fork time,
    each fork's first purpose call finds `None`, creates its OWN fresh
    dict, and both start counting at 1 — so the tail's first facts call
    and the rollup's first sum call both logged `.facts1`/`.sum1` as if
    each were the request's own first call of that purpose, rather than
    sharing one counter the way two calls of the SAME purpose within one
    task already correctly do.

    Creating the dict HERE, before either fork happens, fixes it: a
    `contextvars.Context` copy is shallow (only the variable BINDING is
    independent, not the object it points to), so both forked tasks
    inherit a reference to the SAME dict and their counters correctly
    interleave instead of each restarting at 1.
    """
    _purpose_counters_ctx.set({})


def vllm_headers(purpose: str | None = None) -> dict[str, str]:
    """Headers for a vLLM call in the current request's context.

    - No request id in context (a test, or a caller outside any request —
      e.g. selftest's own direct vLLM probes): `{}`, unchanged behaviour.
    - `purpose=None` (the main chat call): plain `{"X-Request-Id": <id>}`.
    - `purpose="sum"`/`"facts"`/`"dedup"`/`"tok"`/`"digest"`: a child id
      `<id>.<purpose><n>`, n incrementing per purpose within this request
      (1-based) so two dedup calls in one request get `.dedup1`/`.dedup2`
      rather than colliding.
    """
    rid = current_request_id()
    if not rid:
        return {}
    if not purpose:
        return {"X-Request-Id": rid}
    counters = _purpose_counters_ctx.get()
    if counters is None:
        counters = {}
        _purpose_counters_ctx.set(counters)
    n = counters.get(purpose, 0) + 1
    counters[purpose] = n
    return {"X-Request-Id": f"{rid}.{purpose}{n}"}


# ---------------------------------------------------------------------------
# requests_in_flight / last_request_done_age_s (/health/full)
# ---------------------------------------------------------------------------
#
# Plain module globals, not asyncio primitives: every mutation below happens
# on synchronous code with no `await` in between (chat_completions' wrapper
# increments before calling the impl, RequestTrace.finish decrements when it
# fires), so there is no interleaving to race on a single-threaded event
# loop. bgwork.py and health.py follow the same "plain counters" pattern for
# the same reason.
_in_flight = 0
_last_done_at: float | None = None


def request_started() -> None:
    global _in_flight
    _in_flight += 1


def request_finished() -> None:
    """Idempotency is the CALLER's job (RequestTrace.finish only calls this
    once) — calling it twice for one request would undercount, silently."""
    global _in_flight, _last_done_at
    _in_flight = max(0, _in_flight - 1)
    _last_done_at = time.monotonic()


def in_flight_stats() -> dict:
    age = None if _last_done_at is None else round(time.monotonic() - _last_done_at, 1)
    return {
        "requests_in_flight": _in_flight,
        "last_request_done_age_s": age,
    }


def _reset_for_tests() -> None:
    """Test helper — restore the in-flight counters to a clean process
    start. Never called from production code."""
    global _in_flight, _last_done_at
    _in_flight = 0
    _last_done_at = None


# ---------------------------------------------------------------------------
# RequestTrace — one per chat_completions call, owns the "request done" line
# ---------------------------------------------------------------------------


class RequestTrace:
    """Carries this request's id, start time, and the phase timings its
    handler records along the way, and owns emitting the ONE `request done`
    summary line (owner's ask, item 6) exactly once per request no matter
    which of success / error / client-disconnect / cancel path it takes.

    `finish()` is deliberately idempotent (a second call is a silent no-op
    — hostile review round 2 confirmed silent is the right choice here:
    every caller of `finish()` is itself a backstop for another one, by
    design, so a second call is the EXPECTED shape on most requests, not
    an anomaly worth a log line) — chat_completions' wrapper, the
    structural StreamingResponse backstop it installs, and a generator's
    own richer `finally:` can all end up trying to close out the same
    request, and "the line fires more than once" would be exactly as
    wrong an incident as "it never fires" — a grep for
    `request done req=<id>` must return exactly one line.
    """

    __slots__ = (
        "req_id", "start", "timings", "conv_id", "msgs", "streaming",
        "_finished",
    )

    def __init__(self, req_id: str) -> None:
        self.req_id = req_id
        self.start = time.monotonic()
        self.timings: dict[str, float] = {}
        self.conv_id: str | None = None
        self.msgs: int | None = None
        self.streaming = False
        self._finished = False
        request_started()

    def mark(self, name: str, seconds: float) -> None:
        """Record a phase duration (seconds, will be rendered to ms
        precision). Overwrites — a phase that runs more than once in a
        request (there is none today) would report only its last run,
        which is a reasonable default and not what any current caller
        does."""
        self.timings[name] = seconds

    @property
    def finished(self) -> bool:
        return self._finished

    def finish(
        self,
        logger,
        status: str,
        *,
        tokens_out: int | None = None,
        finish_reason: str | None = None,
    ) -> None:
        """Emit the summary line and release the in-flight slot. Safe to
        call from any of: the non-streaming return path, the streaming
        generator's `finally:`, or an exception/cancellation handler around
        either — whichever gets there first wins, and it is always exactly
        one of them because chat_completions' wrapper and event_stream()
        share this same object."""
        if self._finished:
            return
        self._finished = True
        total = time.monotonic() - self.start
        request_finished()
        parts = [
            f"request done req={self.req_id}",
            f"conv={self.conv_id or '-'}",
            f"msgs={self.msgs if self.msgs is not None else '-'}",
            f"status={status}",
        ]
        # Fixed, documented order (OPERATIONS.md "Tracing a request") so a
        # grep/awk pipeline can rely on field position, not just the name=
        # prefix. Present only when actually measured (owner's ask) —
        # `_ORDER` below is the full possible set; anything not in
        # `self.timings` is simply skipped rather than printed as 0 or "-",
        # which would claim a measurement that never happened.
        for key in (
            "t_images", "t_compact", "t_retrieval", "t_inject", "t_guard",
            "t_ttft",
        ):
            if key in self.timings:
                parts.append(f"{key}={self.timings[key]:.3f}")
        parts.append(f"t_total={total:.3f}")
        if tokens_out is not None:
            parts.append(f"tokens_out={tokens_out}")
        if finish_reason:
            parts.append(f"finish={finish_reason}")
        logger.info(" ".join(parts))
