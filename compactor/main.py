"""
context-compactor: OpenAI-compatible middleware proxy in front of vLLM.

V1 behavior (unchanged): token-counts incoming /v1/chat/completions
requests with the target model's tokenizer; when over budget, summarizes
older turns into a single system block.

V2.0 additions:
- Phase 1: conv_id resolution (header-first, hash fallback) + storage
  layout + /admin/conversations endpoints.
- Phase 2 (this file): facts memory — load facts → inject as system
  block before forwarding → after response streams back, async-extract
  new facts from the exchange + prune to budget + save atomically.
"""

import asyncio
import bisect
import codecs
import collections
import dataclasses
import hashlib
import json
import logging
import math
import warnings
import os
import re
import threading
import time
import unicodedata
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone, tzinfo
from typing import Any
from zoneinfo import ZoneInfo

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

import backfill
import backup as backup_module
import bgwork
import commands
import dedup
import degrade
import facts
import health
import logsetup
import persona
import portability
import retrieval
import selftest as selftest_module
import summarizer
import tailhealth
from envcfg import env_bool, env_float
from memory import (
    StoreUnreadable,
    UnsafeConvId,
    conv_lock,
    ensure_storage_layout,
    facts_path,
    list_known_conv_ids,
    resolve_conv_id,
    storage_summary,
    summary_path,
)


def _env_int(name: str, default: int) -> int:
    """os.environ.get returns '' (not the default) when the var is set to an
    empty string, which is what .env files do for opt-in blanks. Treat empty
    as 'use the default'.

    AN UNPARSEABLE VALUE IS THE DEFAULT, NOT A CRASH. Until v3.1.7 this was a
    bare `int(v)`, so one typo in runpod.env — `5O` for `50`, `3OO` for `300`,
    a stray quote, a trailing comment — raised ValueError at import and the
    compactor never started. Every constant on this module's critical path
    reads through here, including MAX_MODEL_LEN, the degeneracy thresholds and
    MIN_MEMORABLE_TRIMMED_CHARS, so the blast radius is the whole process and
    the symptom is a container that will not boot with a traceback nobody
    connects to a config line.

    That is the same failure bgwork._window_s and tailhealth._window_s were
    fixed for, and both cite this function's contract; it now actually holds.
    A bad value is logged nowhere because logging is not configured this early
    — the default is the safe outcome, and an operator who set a value that
    did not take will see it in /health/full's config block.
    """
    v = os.environ.get(name, "")
    if not v.strip():
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8000").rstrip("/")
MODEL_REPO = os.environ.get("MODEL_REPO")
MAX_MODEL_LEN = _env_int("MAX_MODEL_LEN", 32768)
KEEP_RECENT_TURNS = _env_int("COMPACTOR_KEEP_RECENT_TURNS", 4)
SUMMARY_MAX_TOKENS = _env_int("COMPACTOR_SUMMARY_MAX_TOKENS", 1024)
# Slack left inside MAX_MODEL_LEN when budgeting a summarization call's INPUT
# (covers the system prompt, the wrapper text, and chat-template overhead).
SUMMARY_INPUT_RESERVE = _env_int("COMPACTOR_SUMMARY_INPUT_RESERVE", 2048)
# Hard ceiling for what we will forward to vLLM. Anything above this is a
# guaranteed 400, so the guard sheds content rather than letting the request
# fail. The reserve leaves the model room to actually generate a reply.
#
# 2026-08-28: was 2048. That is not enough room to reply. This user's assistant
# turns measure 7,513-11,347 tokens, so a 2048 reserve means that whenever the
# guard actually lets a prompt grow to its ceiling, the reply is cut off
# mid-sentence. It has been masked in practice by two things: most prompts sit
# well under the ceiling, and _apply_request_budget (below) takes
# max(GENERATION_RESERVE, req_max_tokens), so a client that sends max_tokens
# gets the room it asked for. Neither is a guarantee — a client that sends no
# max_tokens and a conversation that reaches the ceiling is exactly the
# combination that truncates.
GENERATION_RESERVE = _env_int("COMPACTOR_GENERATION_RESERVE", 16384)
# Clamped to MAX_MODEL_LEN: a bare floor could sit ABOVE the model's own window
# on a small-context model, which would defeat the entire point of the guard.
HARD_INPUT_LIMIT = min(MAX_MODEL_LEN, max(256, MAX_MODEL_LEN - GENERATION_RESERVE))
# MUST be derived from HARD_INPUT_LIMIT, not from MAX_MODEL_LEN.
#
# This is the compaction trigger: exceed it and older turns get summarized.
# Deriving it from MAX_MODEL_LEN opens a dead band the moment GENERATION_RESERVE
# is non-trivial. With reserve=16384 the old formula gave a trigger of 24,576
# against a guard limit of 16,384 — so every payload between those two numbers
# skipped compaction entirely and went straight to the guard, which cannot
# summarize and can only DELETE turns. That is the 2026-08-28 failure shape:
# content that should have been compressed was discarded instead, silently.
#
# The two numbers are also in different units. `current` here is a LOCAL
# estimate; the guard's limit is measured against vLLM. Sitting at 75% of the
# hard limit leaves headroom for that discrepancy rather than pretending it is
# zero. See count_tokens_exact and REMEDIATION P0-0c.
TARGET_TOKENS = _env_int("COMPACTOR_TARGET_TOKENS", int(HARD_INPUT_LIMIT * 0.75))
# The scale assumed when /tokenize cannot be reached and the summarizer must
# size batches anyway. See the fallback in summarize() for why this is 2.0 and
# not 1.0 — a counter you cannot check must be assumed wrong in the direction
# that fails safe.
# How many summarization LLM calls compaction may make on ONE request.
#
# 2026-08-29, and this is the sharpest lesson of the v3.1 line: compaction runs
# on the REQUEST PATH (chat_completions awaits compact_if_needed). A
# conversation of 170 turns that had never successfully compacted produced 33
# batches; at a 4-wide semaphore and ~1024 output tokens per call on a 24B
# model that is 8+ minutes of the user sitting in front of a dead composer.
# She got no reply at all.
#
# The comment justifying the pessimistic scale said over-splitting "costs extra
# calls on the background tail". It does not. It costs HER LATENCY, and that
# error is why the cap did not exist from the start.
#
# Bounded, compaction makes partial progress each turn and the budget guard
# absorbs whatever is left — which is exactly what the guard is for. Unbounded
# work on a request path is not thoroughness, it is an outage.
MAX_SUMMARY_CALLS_PER_REQUEST = _env_int("COMPACTOR_MAX_SUMMARY_CALLS", 4)

# v3.1.9 (tail catch-up). How many REAL vLLM summarization calls the
# background tail (_rollup_hierarchy, below) may spend on the L1/L2/L3
# hierarchy on ONE turn. NOT the same knob as MAX_SUMMARY_CALLS_PER_REQUEST
# above, and not interchangeable with it: that one bounds request-path
# PREFIX summarization, which REFUSES outright once the backlog exceeds it
# (see summarize()'s own "Self-healing was the wrong shape for this" —
# compact_if_needed is a pure function of the client's array with nowhere to
# persist where it stopped, so a bounded partial attempt there would
# re-summarize the identical oldest batches every turn forever and never
# converge). The hierarchy is different in exactly the way that matters:
# maybe_rollup persists its watermark, its L1/L2/L3 lists and its
# covered-turn record to disk on every call, so a bounded tail genuinely
# advances and the NEXT tail resumes from where this one stopped — the same
# property that already lets /admin/conversations/<id>/compact's own
# max_calls drain a backlog over several passes instead of one.
#
# Default 4: at L1_MAX_TOKENS=500 and the ~1,650-token turns measured on her
# real replies, one 20-turn L1 chunk needs 2 map batches + 1 reduce = 3 real
# calls WHEN /tokenize IS UP — so 4 leaves headroom for one whole chunk plus
# a little, without letting one turn's tail run long enough to compete for
# the GPU with the reply she is waiting on.
#
# hostile pass #5 (C5-7/E8): "3 (more under the pessimistic /tokenize-down
# scale)" understated it badly and is corrected here with a measured number.
# With /tokenize down, pieces price at the pessimistic _WORST_TOKENS_PER_CHAR
# fallback, which over-splits every oversized piece into more, smaller
# batches — SP\p5-c\sim_cost.py measured one unit's real cost at 6-16 calls
# on her turn shape (150-700 char user turns, 5,200-8,000 char replies), not
# 3. This knob still bounds where a unit is allowed to START (see
# summarizer._budget_allows_unit): a unit that starts always finishes, so
# the actual overshoot on a turn whose due unit costs 16 calls is (this
# knob - 1) + 16, not "a little". "4 leaves headroom for one whole chunk
# plus a little" was only ever true with /tokenize answering; treat it as a
# floor on the overshoot, not a ceiling.
TAIL_ROLLUP_MAX_CALLS = _env_int("COMPACTOR_TAIL_ROLLUP_MAX_CALLS", 4)

# How many times summarize() has refused a request over the cap above, since
# the process started. hostile pass #3 (reviewer E, F2): the soak's check that
# a reusing turn is never cap-refused keyed on one phrase of that WARNING, no
# real run had ever produced the phrase, and rewording it passed the
# 2026-09-12 regression green. A counter cannot be reworded out from under
# its reader. Read with compaction_counters(); written only by summarize().
_COMPACTION_COUNTERS = {"cap_refused": 0}


def compaction_counters() -> dict:
    """A copy of the request-path compaction counters (see above)."""
    return dict(_COMPACTION_COUNTERS)

# env_float from envcfg, NOT the local _env_float: that one is defined ~30
# lines BELOW this line, so calling it here is a NameError at import — an
# unconditional boot failure in place of the conditional one being fixed.
# A typo in this variable used to stop the container from starting at all.
_PESSIMISTIC_SUMMARY_SCALE = env_float("COMPACTOR_PESSIMISTIC_SUMMARY_SCALE", 2.0)
# V3.1 (Vision): a single image in a VLM request costs far more than its
# text — hundreds to a couple thousand tokens depending on resolution and
# the model's vision encoder. The text-only token estimate misses this
# entirely, so we add a flat per-image estimate to the budget. Conservative
# default keeps us from overflowing the model's real context window; tune
# per VLM if needed.
IMAGE_TOKEN_ESTIMATE = _env_int("COMPACTOR_IMAGE_TOKENS", 4096)
# How many of the most recent image-bearing turns keep their images. Older
# image parts become a short text note.
#
# v3.0.4: this is applied on EVERY request, not just during compaction. v3.0.2
# put the cap inside compact_if_needed, which only runs when a conversation
# exceeds TARGET_TOKENS — so below that threshold images still accumulated
# without limit, and above it a cap of 3 was itself crushing: a photo tiles to
# thousands of real tokens (far more than we estimate), so three of them can
# consume a third of a 32K window before any conversation fits. Users saw their
# text truncated by their own uploads.
#
# Default 1: the image is needed for the turn that asks about it. After that
# the ASSISTANT's description carries the content forward (and v3.0.3 makes
# that description a durable fact), so the pixels are pure cost.
#   N >= 0 : keep the N most recent image turns (0 = keep none in history)
#   -1     : unlimited (pre-v3.0.2 behavior; not recommended)
MAX_RETAINED_IMAGES = _env_int("COMPACTOR_MAX_RETAINED_IMAGES", 1)


def _env_float(name: str, default: float) -> float:
    """Same contract as _env_int: an unset, blank or unparseable value is the
    default, and never a crash at import time.

    WHAT THIS DOES NOT DO, stated because the docstring used to claim it and
    two other modules cited the claim. An explicit `0`, a negative, `nan` or
    `inf` is RETURNED AS GIVEN — this function does not police the range,
    because its callers disagree about what a legal range is: a budget
    fraction of 0 is a mistake, while several knobs take 0 as a meaningful
    'off'. A caller that needs a positive value must say so itself; that is
    what bgwork._window_s and tailhealth._window_s do, and their docstrings
    used to describe this function as rejecting a silent zero, which it never
    has. Corrected in v3.1.7 rather than changing the behaviour, because
    tightening it here would silently move every knob that legitimately
    accepts 0."""
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


# v3.1 D3 — a ceiling on the SUM of injected memory, denominated in the window.
#
# Every injected layer is individually capped and their sum is not. Facts are
# bounded by COMPACTOR_MAX_FACTS_TOKENS (1500, facts.py:67), retrieval by
# COMPACTOR_MAX_RETRIEVAL_TOKENS (1500, retrieval.py:69 — "1500 mirrors the
# facts budget deliberately: no injected memory layer should be able to..."),
# and each summary chunk by its own generation ceiling. Nothing has ever
# bounded persona + facts + retrieval + L1 + L2 + L3 TOGETHER, and no layer cap
# has ever been able to see the limit the request will actually be measured
# against. Caps that cannot see the limit can sum past it, and on 2026-08-28
# they did.
#
# So the bound is a FRACTION of this request's effective limit rather than
# another token constant: it moves with GENERATION_RESERVE, with MAX_MODEL_LEN
# and with a client asking for a large completion, instead of needing a hand
# re-tune every time any of those change. 0.5 says: whatever else happens, half
# the window belongs to the conversation.
INJECTION_BUDGET_FRACTION = _env_float("COMPACTOR_INJECTION_BUDGET_FRACTION", 0.5)

# P9-1/P9-2 (hostile pass #9): a SEPARATE fraction for the reuse stand-in's
# ceiling (see `_standin_reuse_ceiling` below), deliberately NOT the same
# 0.6 constant `_standin_injected_share` uses for the separately-injected
# summary block. The two situations only look alike. The injected block
# competes for room in `inject_budget` alongside persona, facts and
# retrieval (all four bounded together by `_bound_injected_blocks`), so it
# only gets a slice. The stand-in is different: on a REUSING turn the
# summary injection site skips its own copy entirely (`sum(in-array)`,
# search `_compaction_stored_turns` in chat_completions) — nothing else in
# `inject_budget` spends this share, so there is no reason to multiply it
# down by 0.6 as though it still had to leave room for a sibling that this
# turn never renders. v3.1.9.1 and v3.1.9.2 used the SAME function for
# both (deliberately, "so the two call sites cannot drift apart") and that
# is exactly what starved the stand-in: at the shipped 0.6/6230 defaults
# the ceiling was 6,230 against her ~9,050-token hierarchy (P9-1); even at
# the planned 0.75/10000 it was 9,345, still short of what her hierarchy
# renders at once it holds an L3 (P10-2, hostile pass #10 — NOT the
# "9*L1_MAX + 4*L2_MAX + L3_MAX = 11,300" figure a previous version of this
# comment cited: that arithmetic is in OUTPUT tokens, a different unit from
# what this ceiling is actually checked against, `_estimate_block_tokens`,
# which prices non-ASCII per UTF-8 BYTE — see SUMMARY_BLOCK_MAX_TOKENS's
# own env comment in Dockerfile/runpod.env.template for the measured
# render this is sized against instead). Default 1.0: the stand-in may
# claim the WHOLE freed share, capped only by SUMMARY_BLOCK_MAX_TOKENS
# (15,000 shipped — see that constant's own comment for the arithmetic).
# The guard downstream
# (`_enforce_hard_budget`) is still free to shed OTHER injected memory if
# the whole request runs over effective_limit — this fraction only decides
# whether the stand-in is ALLOWED to render whole, not whether the request
# fits.
STANDIN_BUDGET_FRACTION = _env_float("COMPACTOR_STANDIN_BUDGET_FRACTION", 1.0)

# ...and a much tighter one for a request with no conversational history.
#
# Live, 2026-08-28: a request with msgs=2, source=hash, lastturn=0 and no prior
# assistant turn was handed 95 facts and 3 retrieval hits. It could not be
# compacted ("over budget (30437>12288) but no older turns to summarize"), it
# could not be shed (there is nothing to drop but the injected blocks and the
# one turn the user typed), and vLLM rejected it. The turn produced no reply,
# no facts, no episodic write, and nothing retried it. It repeated, unchanged,
# for four hours.
#
# A request with no prior assistant turn is one of two things, and neither
# wants a conversation's whole accumulated memory:
#
#   - Background/task traffic — OpenWebUI's title, tag and follow-up calls.
#     FRONTEND_SPEC §15 asks the client to mark these explicitly; today they
#     arrive unmarked and hash to a stable conv_id. Such a call has no exchange
#     to remember and no persona to stay in character for, so every token of
#     injected memory it receives is spent making a title worse.
#   - The first turn of a chat. There IS a case for memory here — "she
#     remembers me from the first message" is the product — but a first turn is
#     also the one shape that can neither be compacted (no older turns) nor
#     shed (one turn, and the newest turn is never dropped), so it is exactly
#     where an oversized injection stops being a degradation and becomes a lost
#     turn.
#
# Hence bound, do not refuse. An eighth of the window still carries the
# highest-ranked facts — facts.select_for_injection already orders them — and
# stops there. Set COMPACTOR_INJECTION_NO_HISTORY_FRACTION to 0 to turn
# injection off entirely for this traffic once the client marks it.
INJECTION_NO_HISTORY_FRACTION = _env_float(
    "COMPACTOR_INJECTION_NO_HISTORY_FRACTION", 0.125
)

# Drop order when the sum will not fit: highest number goes first.
#
# Retrieval is the most speculative layer — it is a guess about relevance, and
# its own log line already reports how much of its budget it kept. Facts
# degrade gracefully because they are ranked and truncating the tail loses the
# least-used ones. The summary stack is the only compressed record of the part
# of the conversation the window can no longer hold, so it outranks both.
# Persona goes last: without it the reply is wrong in KIND, not merely less
# informed, and it is the cheapest of the four.
_INJECT_PRIORITY_PERSONA = 0
_INJECT_PRIORITY_SUMMARY = 1
_INJECT_PRIORITY_FACTS = 2
_INJECT_PRIORITY_RETRIEVAL = 3

# V2.0 Phase 1: admin endpoint binding. Default "127.0.0.1" rejects any
# non-localhost client at the dependency layer (we still bind the FastAPI
# socket to 0.0.0.0 because uvicorn doesn't support dual-listen, but the
# admin paths return 403 unless the client IP is localhost). Set this to
# "0.0.0.0" to expose admin endpoints externally — only safe if you have
# auth/firewall in front.
ADMIN_BIND = os.environ.get("COMPACTOR_ADMIN_BIND", "127.0.0.1").strip()

logsetup.configure()  # V2.3 Theme 4: text (default) or JSON via COMPACTOR_LOG_FORMAT
logger = logging.getLogger("compactor")

_tokenizer = None
# Whether get_tokenizer has ALREADY tried (and possibly failed) at least
# once. See the docstring below: caching only the success made every later
# count_tokens re-enter from_pretrained, 238x slower per call — that is what
# this flag was added to stop. It no longer means "never try again"; see
# _TOKENIZER_NEXT_RETRY_AT.
_TOKENIZER_TRIED = False
# get_tokenizer takes no lock before v3.1.9 HIGH #3 (hostile pass 2 on
# 843bf9d): under uvicorn, count_tokens runs from the threadpool, so two
# requests can enter concurrently. Thread A used to latch _TOKENIZER_TRIED
# and then block inside from_pretrained; thread B would see the flag already
# set and return the char/4 estimator for a tokenizer that was about to load
# successfully — a wrong answer, not a crash, so nothing noticed. tokens.py's
# sibling singleton (_load, same file) already gets this right with a
# threading.Lock and a double-checked read; this mirrors that pattern.
# threading.Lock (not asyncio.Lock) because this is called both from the
# event loop and from run_in_threadpool workers — an asyncio.Lock only
# coordinates coroutines on one loop and would not see threadpool callers.
_TOKENIZER_LOCK = threading.Lock()
# Failure-cache bookkeeping (v3.1.9 HIGH #3). A hostile pass found that
# caching a MISS forever converts a transient fault — the MooseFS /data
# blip this repo has documented twice (2026-08-31), or an HF cache that
# is not warm yet at boot — into a PERMANENT one, because nothing in the
# tree ever clears _TOKENIZER_TRIED. count_tokens drives compact_if_needed's
# trigger and the hard budget guard, so a process pinned on char/4 for its
# whole life silently discards content that should have been compressed
# instead (the shape of the 2026-08-28 incident, worse). The fix is a
# monotonic-clock retry window instead of "forever" or "every call": doubling
# from _TOKENIZER_RETRY_FLOOR_S to _TOKENIZER_RETRY_CAP_S bounds the cost at
# one 2.9 ms attempt per window — at the floor that is 0.01% of the 6.6
# s/compaction the original fix measured — while still self-healing.
_TOKENIZER_LAST_ERROR: str | None = None
_TOKENIZER_FAILED_AT: float | None = None       # time.monotonic() of the last miss
_TOKENIZER_NEXT_RETRY_AT: float | None = None   # time.monotonic() gate; None = no gate (untried, or loaded)
_TOKENIZER_RETRY_S = 30.0        # current backoff interval; doubles on each consecutive miss
_TOKENIZER_RETRY_FLOOR_S = 30.0  # first retry ~30s after a miss
_TOKENIZER_RETRY_CAP_S = 600.0   # ...never further apart than 10 minutes


def get_tokenizer():
    """The local tokenizer, or None. THE FAILURE IS CACHED, ON A TIMER (v3.1.9).

    `if _tokenizer is not None: return` caches only a SUCCESS. The except
    below sets `_tokenizer = None`, which fails that same test, so every later
    call re-entered AutoTokenizer.from_pretrained — a filesystem walk and, when
    the HF cache is cold, a network attempt. Benchmarked: 0.012 ms cached
    against 2.859 ms per call after a miss, and `_chunk_to_budget` calls
    count_tokens once PER MESSAGE, so one 2,301-message compaction spends about
    6.6 seconds re-failing to load the same tokenizer. That is the FAST failure
    (HF_HUB_OFFLINE=1); a cold cache reaching for the network is worse. That
    cost is why a miss is cached at all.

    Caching the miss FOREVER (843bf9d's shape) traded that latency bug for a
    worse one: a five-second I/O blip at the moment of the first count_tokens
    call pins the char/4 estimator for the rest of the process, silently,
    with no field in /health/full to see it by (see tokenizer_state() below).
    So the miss is cached only until _TOKENIZER_NEXT_RETRY_AT, which backs off
    from _TOKENIZER_RETRY_FLOOR_S and doubles up to _TOKENIZER_RETRY_CAP_S on
    each consecutive miss, and resets the moment a load succeeds. Between
    misses this is a float comparison under a held lock, not a filesystem
    walk — the per-call cost the original fix was written to kill stays dead.

    _TOKENIZER_TRIED is a separate flag rather than a sentinel object because
    `None` is a legitimate return here: it means "use the char/4 estimator",
    and several callers check for it. It now means "at least one attempt has
    been made", not "never try again" — MODEL_REPO absent is the one
    exception (a static config value that cannot change without a process
    restart, which resets this module anyway), so that path still latches
    permanently rather than spinning a retry clock that can never help.

    Locking: the whole read-test-and-maybe-load body runs under
    _TOKENIZER_LOCK so a concurrent caller during an in-flight first load
    blocks and then re-reads the resolved state, instead of observing the
    latched-but-not-yet-resolved flag and returning a wrong answer (the LOW
    finding paired with this one). The fast, no-lock check below is the
    steady-state path (already loaded) and never itself the source of a wrong
    answer, because it only ever short-circuits toward re-checking, not away
    from it.
    """
    global _tokenizer, _TOKENIZER_TRIED, _TOKENIZER_LAST_ERROR
    global _TOKENIZER_FAILED_AT, _TOKENIZER_NEXT_RETRY_AT, _TOKENIZER_RETRY_S
    if _tokenizer is not None:
        return _tokenizer
    with _TOKENIZER_LOCK:
        if _tokenizer is not None:  # double-checked: another thread may have
            return _tokenizer       # finished loading while we waited for the lock
        now = time.monotonic()
        if (_TOKENIZER_TRIED and _TOKENIZER_NEXT_RETRY_AT is not None
                and now < _TOKENIZER_NEXT_RETRY_AT):
            return None  # still inside the backoff window from the last miss
        if not MODEL_REPO:
            if not _TOKENIZER_TRIED:
                logger.warning("MODEL_REPO not set; falling back to char/4 token estimator")
            _TOKENIZER_TRIED = True
            _TOKENIZER_LAST_ERROR = "MODEL_REPO not set"
            _TOKENIZER_FAILED_AT = now
            # Not a transient fault -- nothing will make MODEL_REPO appear
            # without a restart, and a restart re-imports this module anyway.
            # A real (finite) retry clock here would just re-log the same
            # warning forever for no chance of success.
            _TOKENIZER_NEXT_RETRY_AT = float("inf")
            return None
        try:
            from transformers import AutoTokenizer

            _tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
            logger.info(f"loaded tokenizer for {MODEL_REPO}")
            _TOKENIZER_LAST_ERROR = None
            _TOKENIZER_FAILED_AT = None
            _TOKENIZER_NEXT_RETRY_AT = None
            _TOKENIZER_RETRY_S = _TOKENIZER_RETRY_FLOOR_S  # a recovered process earns back the short interval
        except Exception as e:
            _TOKENIZER_LAST_ERROR = str(e)
            _TOKENIZER_FAILED_AT = now
            _TOKENIZER_NEXT_RETRY_AT = now + _TOKENIZER_RETRY_S
            logger.warning(
                f"could not load tokenizer for {MODEL_REPO}: {e}; using char/4 "
                f"estimator (retrying in {_TOKENIZER_RETRY_S:.0f}s)"
            )
            _tokenizer = None
            _TOKENIZER_RETRY_S = min(_TOKENIZER_RETRY_S * 2, _TOKENIZER_RETRY_CAP_S)
        _TOKENIZER_TRIED = True
        return _tokenizer


def tokenizer_state() -> dict:
    """Snapshot of get_tokenizer's cache for /health/full (v3.1.9 HIGH #3).

    Before this, the one degradation that could be PERMANENT (a cached
    tokenizer-load failure) was also the one with no field anywhere in
    /health/full — `tokenize` is vLLM's /tokenize HTTP endpoint and
    `tokens.is_available()` is the separate mistral_common tekken tokenizer;
    neither says anything about this cache. The health lane reads exactly
    these four keys — do not rename or add to them without updating it.

    WALL-CLOCK TIMES OUT, MONOTONIC INSIDE. The backoff gate is kept on
    time.monotonic() so a clock step cannot shorten or stretch it, but a
    monotonic reading is seconds since an arbitrary origin and means nothing
    to a caller. health.py prints `next_retry_at - time.time()`, and handed
    the raw monotonic value that was always "next retry in 0s" (monotonic is
    far smaller than an epoch timestamp, so max(0, ...) floored it). Found at
    merge, where the two lanes met: each was right against its own brief, and
    the brief had said "monotonic" for the gate and "float" for the field
    without saying which clock the field is on. `inf` (no retry will ever
    happen - MODEL_REPO unset) passes through; health renders it as "no retry
    scheduled".
    """
    def _wall(t: float | None) -> float | None:
        if t is None or t == float("inf"):
            return t
        return time.time() + (t - time.monotonic())

    return {
        "loaded": _tokenizer is not None,
        "last_error": _TOKENIZER_LAST_ERROR,
        "failed_at": _wall(_TOKENIZER_FAILED_AT),
        "next_retry_at": _wall(_TOKENIZER_NEXT_RETRY_AT),
    }


# ---------------------------------------------------------------------------
# Backend modality (v3.0.1). One uploaded image PERMANENTLY poisoned a
# conversation on a text-only backend: OpenWebUI re-sends the full history
# (image included) with every message, V3.1 compaction deliberately preserves
# image turns, and vLLM 400s each request ("is not a multimodal model") — so
# every later message in that conversation failed, forever. The compactor was
# forwarding content the backend cannot accept: an unverified modality
# boundary. When the backend is text-only, image parts are replaced with an
# honest placeholder instead of being forwarded.
#
# COMPACTOR_BACKEND_MULTIMODAL: "auto" (default — read MODEL_REPO's HF config
# and look for a vision tower), or "true"/"false" to override.
# ---------------------------------------------------------------------------

_BACKEND_MULTIMODAL_ENV = os.environ.get("COMPACTOR_BACKEND_MULTIMODAL", "auto").strip().lower()
_backend_multimodal: bool | None = (
    True if _BACKEND_MULTIMODAL_ENV == "true"
    else False if _BACKEND_MULTIMODAL_ENV == "false"
    else None
)


def backend_is_multimodal() -> bool:
    """Whether the served model can accept image input. Cached for process
    life; unknown resolves to True (no stripping — the reactive backstop in
    the 4xx handlers flips it if vLLM says otherwise)."""
    global _backend_multimodal
    if _backend_multimodal is not None:
        return _backend_multimodal
    if not MODEL_REPO:
        _backend_multimodal = True
        return True
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(MODEL_REPO)
        _backend_multimodal = getattr(cfg, "vision_config", None) is not None
        logger.info(
            f"backend modality for {MODEL_REPO}: "
            f"{'multimodal' if _backend_multimodal else 'TEXT-ONLY (image parts will be stripped)'}"
        )
    except Exception as e:
        logger.warning(
            f"could not resolve modality for {MODEL_REPO} ({e}); assuming "
            f"multimodal — the 4xx backstop will correct this if vLLM disagrees"
        )
        _backend_multimodal = True
    return _backend_multimodal


# v3.0.5: learned budget correction. Our token count is an ESTIMATE — the flat
# per-image cost especially so (a real photo tiles to 4-8k tokens on Mistral3
# encoders; production showed a 6,859-token undercount past the guard). vLLM's
# context-length 400 reports the TRUE count, so instead of guessing we learn:
# parse it, tighten the effective limit by the observed overshoot, and the next
# message heals — same self-healing pattern as the modality backstop. Capped so
# a pathological report can't crush the window.
#
# v3.1 A10: it is a MODULE GLOBAL — one process (supervisord.conf:87 runs
# uvicorn with no --workers), one margin, every conversation. It was also
# monotonic with no release path, so a single oversized turn narrowed the
# window for everything else until the next restart. It now decays on sustained
# success; see _note_backend_accepted. Since P0-0c gave the guard vLLM's own
# count, this is a degraded-mode backstop rather than the primary mechanism, and
# it is sized and released as one.
#
# v3.1 D4: decay was not enough, because the blast radius came from what the
# margin was allowed to LEARN, not from how long it held. A single conversation
# whose own messages did not fit the window latched it to the
# MAX_MODEL_LEN//4 ceiling on its first rejection and every other conversation
# in the process paid. It now only learns from a rejection the guard did not
# already predict — see _note_backend_rejection's `guard_measured_overflow`.
_BUDGET_MARGIN = 0

# Every wording vLLM has used to state the prompt size in a context-length 400,
# read out of the pinned engines rather than guessed. A regex that silently
# fails to match here is the whole calibration path going dark while the log
# still reads as though it learned something — INCIDENT_2026-08-24 D26 called
# this out and it was still one pattern until v3.1.
#
# Verified 2026-08-28 by reading the vLLM installed in the two images this
# stack actually ships:
#
#   0.24.0  (the cu13/default pin, Dockerfile:78; read from
#            angreg/zions-light-ai:v3.0-cu13,
#            vllm/renderers/params.py:429 _token_len_check)
#       "...you requested {O} output tokens and your prompt contains
#        {at least }{N} input tokens, for a total of ..."          -> (1)
#
#   0.24.0  (same file, :337 _text_len_check — a CHARACTER pre-check that
#            fires before tokenization)
#       "...your prompt contains {C} characters (more than {X} characters,
#        which is the upper bound for {N} input tokens)..."        -> (none)
#
#   0.19.0  (the CUDA-12 fallback profile, and what
#            angreg/zions-light-ai:v3.0.5-cu12 ships;
#            vllm/entrypoints/openai/engine/serving.py:752,762)
#       "...you requested {O} output tokens and your prompt contains
#        {N} input tokens, for a total of ..."                     -> (1)
#       "...However, your request has {N} input tokens. Please reduce the
#        length of the input messages."                            -> (2)
#
#   0.10.0  (not deployed, but what the contract fixture reproduces as
#            FIXTURE_ERROR_STYLE=v010)
#       "...you requested {O+N} tokens ({N} in the messages, {O} in the
#        completion)..."                                           -> (3)
#       "...you requested {N} tokens in the messages, ..."         -> (3)
#
# So (1) — the only wording the single pre-v3.1 pattern covered — is emitted by
# BOTH deployed pins, and only when the request carried a max_tokens. 0.19.0's
# no-max_tokens branch (2) was never matched, and 0.19.0 is a shipped profile.
#
# The character pre-check is deliberately NOT matched. Its number is a
# CHARACTER count, roughly 4x a token count, and feeding it to the calibration
# below as if it were tokens would saturate the margin cap off one rejection.
# `_is_context_overflow` still classifies it correctly, so the user is told the
# truth; we simply decline to learn a number that means something else.
_CTX_OVERFLOW_PATTERNS = (
    # (1) 0.19.0 and 0.24.0, request carried max_tokens; also the older
    #     "your prompt contains at least N input tokens" wording.
    r"prompt contains (?:at least )?(\d+) input tokens",
    # (2) 0.19.0, no max_tokens on the request.
    r"request has (\d+) input tokens",
    # (3) 0.10.0, both of its variants. The prompt half is the one after the
    #     parenthesis ("(N in the messages") or before "tokens in the
    #     messages"; the leading "you requested N tokens" in that wording is
    #     prompt+completion and must not be captured.
    r"(\d+)(?: tokens)? in the messages",
)
_CTX_OVERFLOW_RE = re.compile("|".join(f"(?:{p})" for p in _CTX_OVERFLOW_PATTERNS))


def _is_context_overflow(err_body: str) -> bool:
    """Whether a 4xx body is vLLM's context-length rejection specifically.

    The message the user gets turns on this: "too large for the window, send it
    again" is true here and false for every other 400 (modality, alternation,
    a malformed payload), and telling someone to resend a request that will be
    refused identically is the same class of error as telling them the backend
    is restarting when it is healthy."""
    return "maximum context length" in (err_body or "")


def _reported_prompt_tokens(err_body: str) -> int | None:
    """The TRUE prompt size vLLM reports in a context-length 400, or None.

    Read separately from the calibration below because the log line must name
    the number whether or not the calibration decided to act on it — a
    rejection that teaches us nothing (the margin is already larger, or capped)
    is exactly the one whose numbers someone will need later.

    Several alternatives, one per wording vLLM has used — see
    _CTX_OVERFLOW_PATTERNS. Exactly one group can participate in any match, so
    the first non-None group is the answer."""
    m = _CTX_OVERFLOW_RE.search(err_body or "")
    if m is None:
        return None
    for g in m.groups():
        if g is not None:
            return int(g)
    return None


# v3.1 A10: how many consecutive ACCEPTED requests release half the learned
# margin. _BUDGET_MARGIN used to be monotonic with no reset short of a process
# restart, so one pathological turn cost every conversation in the process up
# to MAX_MODEL_LEN//4 of window, forever — and post-P0-0b it latches there in a
# single event rather than crawling. Since P0-0c, count_tokens_exact does the
# real work and the margin is only the DEGRADED-mode backstop for when
# /tokenize will not answer; a margin still in force after fifty clean requests
# is describing a state the process is no longer in.
#
# This is a policy number, not a measurement, and it is named as one. Halving
# rather than clearing is the conservative half of the choice: if the margin
# was still needed, the cost of finding out is one 400 and one re-learn, and
# the re-learn lands back at the same value because the calibration measures
# the gap directly. Set to 0 to restore the pre-v3.1 monotonic behaviour.
BUDGET_MARGIN_RELEASE_AFTER = _env_int("COMPACTOR_BUDGET_MARGIN_RELEASE_AFTER", 50)
_budget_ok_streak = 0


def _note_backend_accepted() -> None:
    """One request vLLM did NOT refuse. Counts toward releasing the margin.

    Called from both response paths on any status below 400. Cheap and
    lock-free: uvicorn runs single-process here (supervisord.conf has no
    --workers) and the loop is cooperative, so the read-modify-write below
    cannot interleave."""
    global _BUDGET_MARGIN, _budget_ok_streak
    if not _BUDGET_MARGIN or BUDGET_MARGIN_RELEASE_AFTER <= 0:
        return
    _budget_ok_streak += 1
    if _budget_ok_streak < BUDGET_MARGIN_RELEASE_AFTER:
        return
    _budget_ok_streak = 0
    before = _BUDGET_MARGIN
    _BUDGET_MARGIN = 0 if before <= 512 else before // 2
    logger.info(
        f"context calibration: {BUDGET_MARGIN_RELEASE_AFTER} consecutive "
        f"accepted requests — releasing budget margin {before} -> "
        f"{_BUDGET_MARGIN}. The margin is the backstop for a /tokenize outage, "
        f"not a permanent tax on the window; if it is still needed the next "
        f"rejection measures it back in one step."
    )


def _note_backend_rejection(
    err_body: str,
    enforced_limit: int | None = None,
    guard_measured_overflow: bool = False,
) -> bool:
    """Reactive backstop for vLLM 4xx bodies. Two lessons we can learn:
    (1) the model is text-only -> strip images from subsequent requests;
    (2) our token count undercounted -> tighten the budget by the observed gap.
    Either way the conversation heals on its next message instead of staying
    poisoned.

    `enforced_limit` is the limit the guard ACTUALLY shed this payload against,
    margin already subtracted — see A8 below for why it is a parameter and not
    something this function may reconstruct.

    `guard_measured_overflow` is True when _enforce_hard_budget MEASURED this
    payload as over that limit and forwarded it anyway as a best effort. The
    margin exists to correct a SURPRISE — a payload we believed fit and vLLM
    charged more for — and there is no surprise in a rejection the guard
    predicted, at ERROR, before the request was sent. Widening the margin from
    one is learning from evidence that does not bear on the question.

    v3.1 D4, and not hypothetical. On 2026-08-28 one conversation kept sending
    two messages whose own content measured 30,437 local tokens against a
    12,288 compaction target. The guard shed everything it was permitted to
    shed and logged

        hard budget FAILED to fit: ... dropped 0 old turn(s), trimmed 6
        injected block(s), dropped 1 injected block(s) entirely - still
        16417 over

    then forwarded and took the 400 it had just predicted. The rejection
    reported 32,801 tokens; 16,384 + 16,417 = 32,801, so the number the
    calibration "learned" was the number the guard had already measured and
    logged. overshoot came to 16,417 and _BUDGET_MARGIN latched straight to its
    MAX_MODEL_LEN//4 ceiling of 8,192 — a module global, so every OTHER
    conversation in the process lost 8,192 tokens of window. Four hours later a
    real conversation was running at "limit 8192, margin 8192" and shedding on
    every request, while the conversation that imposed it was unchanged, still
    failing, and had never benefited from it.

    Note what this rule does NOT do: it does not ask whether the overflow was
    injection-driven. In the case above it was not — the residual was the
    client's own two messages — so a rule keyed on injection would not have
    fired. What the two failures have in common is not their content; it is
    that the guard already knew.

    Returns whether the budget margin actually advanced. The caller uses it to
    decide what to promise the user: "send it again" is only true when this
    rejection taught us something, and a rejection can teach us nothing (the
    margin is already wider, or it has hit the MAX_MODEL_LEN//4 cap).
    """
    global _backend_multimodal, _BUDGET_MARGIN, _budget_ok_streak
    body = err_body or ""
    tightened = False
    if "not a multimodal model" in body and _backend_multimodal is not False:
        _backend_multimodal = False
        logger.warning(
            "backend declared itself text-only via a 400; image parts will be "
            "stripped from subsequent requests (set COMPACTOR_BACKEND_MULTIMODAL "
            "to override)"
        )
    if _is_context_overflow(body) and guard_measured_overflow:
        # v3.1 D4. Reported at WARNING rather than swallowed: "the calibration
        # deliberately did not fire" is a different state from "the calibration
        # is broken again", and the whole lesson of P0-0/A9 is that a path
        # which cannot say it fired is indistinguishable from one that did.
        logger.warning(
            f"context calibration: NOT widening the budget margin from this "
            f"rejection. The hard-budget guard had already measured this "
            f"payload as over the limit it enforced and forwarded it as a best "
            f"effort, so vLLM's 400 confirms a measurement we already had — it "
            f"is not evidence that our counting is low. The margin is a module "
            f"global; learning {_reported_prompt_tokens(body)} tokens from a "
            f"predicted rejection would narrow the window for every other "
            f"conversation in this process to pay for one that is unfittable "
            f"as sent (v3.1 D4). Margin stays at {_BUDGET_MARGIN}."
        )
    elif _is_context_overflow(body):
        actual = _reported_prompt_tokens(body)
        if actual is not None:
            # v3.1 P0-0b: measure against the limit we ACTUALLY enforced, not
            # the original one. _enforce_hard_budget has already shed to
            # (limit - _BUDGET_MARGIN), so measuring against the untightened
            # limit understates the undercount by exactly the margin already in
            # force — and the monotonic guard below then refuses to advance
            # until the undercount roughly doubles. Observed live on
            # 2026-08-27: three consecutive failures on one conversation moved
            # the margin 2628 -> 2755 -> 2882, +127 each time (the
            # conversation's own growth per turn) while it needed ~5250. That is
            # a loop, not a retry: ~19 more broken messages for the user.
            #
            # v3.1 A8: and the limit is a PARAMETER, because the guard's limit
            # is per-request. chat_completions derives it from
            # MAX_MODEL_LEN - max(GENERATION_RESERVE, req_max_tokens), so a
            # client asking for a large completion is shed against something
            # well below HARD_INPUT_LIMIT. Reconstructing it from
            # HARD_INPUT_LIMIT here understated the overshoot by up to
            # HARD_INPUT_LIMIT - effective_limit, one-directionally: at
            # max_tokens=8192 on the shipped 32768 window the guard enforced
            # 24576, so a 25000-token prompt overshot by 424 and this computed
            # -5720 — no advance, tightened=False, and the user was told, in
            # those words, that retrying would not help. The correct number was
            # sitting two lines from the call site and was not passed.
            #
            # None means "no per-request limit available" — the tests, and any
            # future caller off the request path. Reconstructing is then the
            # best we can do, and the log line says which limit it used so a
            # reader is never guessing. The max(256, ...) mirrors the clamp in
            # _enforce_hard_budget.
            if enforced_limit is not None:
                measured_against = max(256, enforced_limit)
                limit_src = "the limit the guard enforced"
            else:
                measured_against = max(256, HARD_INPUT_LIMIT - _BUDGET_MARGIN)
                limit_src = (
                    "HARD_INPUT_LIMIT minus the current margin (no per-request "
                    "limit was passed — this is a reconstruction)"
                )
            overshoot = actual - measured_against
            if overshoot > 0:
                new_margin = min(overshoot + 512, MAX_MODEL_LEN // 4)
                if new_margin > _BUDGET_MARGIN:
                    _BUDGET_MARGIN = new_margin
                    # A fresh correction restarts the release clock: the
                    # successes that were accumulating were describing a
                    # process state this rejection just disproved.
                    _budget_ok_streak = 0
                    tightened = True
                    logger.warning(
                        f"context calibration: vLLM counted {actual} tokens where "
                        f"we budgeted <= {measured_against} ({limit_src}) — our "
                        f"estimate undercounts (a /tokenize outage, or images, "
                        f"are the usual causes). Tightening the hard limit by "
                        f"{new_margin} for EVERY conversation in this process "
                        f"(the margin is a module global, not per-conversation); "
                        f"the next message should succeed."
                    )
    return tightened


_IMAGE_PLACEHOLDER = (
    "[The user attached {n} here. The current model is text-only and cannot "
    "see {pron} — if the content matters, ask the user to describe {pron}.]"
)


def _strip_image_parts(messages: list[dict]) -> tuple[list[dict], int]:
    """Replace image parts with an honest text placeholder, preserving all
    text parts. Returns (new_messages, images_stripped). Honesty over
    silence: the model is TOLD an image existed and that it cannot see it,
    rather than the image quietly vanishing from the conversation."""
    out: list[dict] = []
    stripped = 0
    for m in messages:
        n = _message_image_count(m)
        if n == 0:
            out.append(m)
            continue
        text = _message_text(m).strip()
        note = _IMAGE_PLACEHOLDER.format(
            n="an image" if n == 1 else f"{n} images",
            pron="it" if n == 1 else "them",
        )
        out.append({**m, "content": f"{text}\n\n{note}" if text else note})
        stripped += n
    return out, stripped


def _message_text(m: dict) -> str:
    """Plain-text view of a message. OpenAI multimodal content is a list of
    parts; only text parts contribute (image parts have no 'text'), so this
    safely ignores images for budgeting/summarization/fact-extraction."""
    content = m.get("content") or ""
    if isinstance(content, list):
        return " ".join(c.get("text", "") for c in content if isinstance(c, dict))
    return str(content)


def _message_image_count(m: dict) -> int:
    """How many image (non-text) parts a message carries. V3.1: OpenAI
    multimodal content arrays use parts like {"type": "image_url", ...}."""
    content = m.get("content")
    if not isinstance(content, list):
        return 0
    n = 0
    for c in content:
        if not isinstance(c, dict):
            continue
        t = c.get("type")
        if t in ("image_url", "image", "input_image") or "image_url" in c:
            n += 1
    return n


def _message_has_image(m: dict) -> bool:
    return _message_image_count(m) > 0


# v3.1 A13: /tokenize outage reporting.
#
# This used to be `logsetup.log_once("count_tokens_exact.http")` — ONE line per
# process, whose `_logged_once` set is deliberately never cleared
# (logsetup.py:117-137). Four call sites share the endpoint (summarize, the
# guard's ground truth, the guard's per-round verify, _sent_token_size), so one
# token covered all of them for the process lifetime, and an outage starting
# hours after boot was completely silent.
#
# The aggravator is that the most likely first spender is BENIGN: the comment
# below is right that a 400 here is usually the chat template refusing an
# assistant-final list, which the summarizer hands it on any conversation long
# enough to compact. A structural 400 in minute two permanently silenced the
# report of a genuinely broken endpoint in hour six. That silencing is
# structural, not incidental.
#
# So: a rate limit rather than a one-shot, keyed so a structural refusal cannot
# spend the transport-failure signal, and — the part a one-shot can never have —
# a RECOVERY line, because "it started working again" is half of what the reader
# of these lines is trying to establish. The counters are also readable
# programmatically (tokenize_health) so /health/full can report the state as a
# fact rather than leaving it to a log line from three days ago.
# hostile2-config: env_float, not float(_env_int(...)). summarizer.py reads
# the SAME variable with env_float, and the comment there ("an operator
# setting this once should govern every /tokenize dependency in the
# process, not just the ones main.py happens to own") asserted the two
# already agreed. They did not: _env_int parses with int(), which rejects
# any non-integer string ("0.5", "60.5", "1e3") and silently falls back to
# the default of 300 HERE while summarizer.py applied the operator's real
# value — one process, one env var, two different rate limits, with no
# error or log line naming the disagreement.
TOKENIZE_WARN_INTERVAL_S = env_float("COMPACTOR_TOKENIZE_WARN_INTERVAL_S", 300)
_tokenize_fail_streak = 0
# Tracked separately from the chat form: see tokenize_health(). Carries its own
# timestamp because, unlike the chat form, it is NOT exercised on every request
# — count_text_tokens_exact runs only when there are injected blocks to measure
# (main.py _bound_injected_blocks). Without a staleness bound a single text-form
# failure would pin /health/full unhealthy until the next conversation that
# happens to have memory to inject, which is the _BUDGET_MARGIN latching bug
# wearing a different hat.
_tokenize_text_fail_streak = 0
_tokenize_text_last_fail_at: float | None = None
_tokenize_degraded_since: float | None = None
_tokenize_last_warn: dict[str, float] = {}


def _degraded_since_earliest() -> float | None:
    """The EARLIEST start among the degraded /tokenize sources, or None.

    `a or b` took summarizer's timestamp whenever it was non-None regardless
    of which fault started first, so a /tokenize outage an hour old was
    reported as 60 seconds old the moment the summarizer also failed. The
    field exists to tell an operator how long this has been going on;
    reporting the most RECENT start systematically under-states exactly the
    thing it is for.
    """
    starts = [t for t in (_summarizer_degraded_since(), _tokenize_degraded_since)
              if t is not None]
    return min(starts) if starts else None


def _summarizer_degraded_since() -> float | None:
    """When summarizer's /tokenize started failing, or None. Never raises."""
    try:
        return summarizer.tokenize_health().get("degraded_since")
    except Exception:
        return None


def _summarizer_tokenize_failing_now() -> int:
    """summarizer's /tokenize failure streak, or 0 if it cannot be read.

    Staleness-bounded inside summarizer.tokenize_health() for the same reason
    _text_tokenize_failing_now bounds its own: rollups do not run on every
    request, so an old streak means "not asked lately", not "still broken".

    Never raises. A health endpoint that 500s because a dependency's health
    accessor moved is worse than one that under-reports, and this is the
    endpoint an operator reaches for when everything else is already on fire.
    """
    try:
        return int(summarizer.tokenize_health().get("consecutive_failures", 0))
    except Exception:
        return 0


def tokenize_health() -> dict:
    """Current state of the /tokenize dependency, for /health/full.

    `consecutive_failures` > 0 means budgeting is running on the local
    tokenizer, which reads up to 51% low on this model's assistant content —
    i.e. the guard is in the exact degraded mode the 2026-08-28 incident ran
    in. A health endpoint that cannot say so is asking its reader to go find a
    log line instead."""
    return {
        # AND across forms. count_tokens_exact (chat) and
        # count_text_tokens_exact (completion) hit the same endpoint but ask
        # different questions, and only the chat form can be refused by the
        # model's template — which is exactly the D1 outage. A shared streak
        # let one text-count success declare the endpoint healthy while every
        # chat-form call was still 400ing, so this endpoint FLAPPED instead of
        # reporting the degraded mode it exists to report.
        "ok": (
            _tokenize_fail_streak == 0
            and _text_tokenize_failing_now() == 0
            and _summarizer_tokenize_failing_now() == 0
        ),
        "consecutive_failures": max(
            _tokenize_fail_streak,
            _text_tokenize_failing_now(),
            _summarizer_tokenize_failing_now(),
        ),
        "chat_form_failures": _tokenize_fail_streak,
        "text_form_failures": _text_tokenize_failing_now(),
        # summarizer.py POSTs /tokenize too, from its own module-level state.
        # Until this line it reported failures only through log_once, which
        # fires ONCE per process: the rollup counter could be degraded for the
        # life of the pod with /health/full still saying ok=true and not one
        # further log line. Folded into the AND for the same reason the chat
        # and text forms are - a health endpoint that can be green while a
        # counter it covers is blind is the exact shape of the two outages
        # this whole branch exists to close.
        "summarizer_form_failures": _summarizer_tokenize_failing_now(),
        # Folded in for the same reason ok/consecutive_failures are: an
        # operator reading "ok": false with "degraded_for_s": 0.0 sees an
        # endpoint that has been broken for zero seconds since never, and
        # concludes the endpoint is confused rather than the summarizer is
        # down. Reporting a fault without its duration is a sibling-site
        # miss inside the fix that added the fault.
        "degraded_since": _degraded_since_earliest(),
        "degraded_for_s": (
            round(time.time() - _degraded_since_earliest(), 1)
            if _degraded_since_earliest() is not None
            else 0.0
        ),
    }


def _note_text_tokenize_failure() -> None:
    """Count a completion-form failure. Deliberately silent: the WARNING is
    still emitted by _note_tokenize_failure under a text.* key, and duplicating
    it here would double every line during an outage."""
    global _tokenize_text_fail_streak, _tokenize_text_last_fail_at
    _tokenize_text_fail_streak += 1
    _tokenize_text_last_fail_at = time.time()


def _note_text_tokenize_success() -> None:
    global _tokenize_text_fail_streak, _tokenize_text_last_fail_at
    _tokenize_text_fail_streak = 0
    _tokenize_text_last_fail_at = None


def _text_tokenize_failing_now() -> int:
    """The completion form's streak, or 0 once it has gone stale.

    Stale means "we have not seen this form fail for a whole warn interval",
    which for a form that is only called when memory is being injected is the
    honest reading: we do not know it is broken, and asserting a fault we
    cannot currently observe is the same error as asserting health we cannot
    observe."""
    if not _tokenize_text_fail_streak or _tokenize_text_last_fail_at is None:
        return 0
    if (time.time() - _tokenize_text_last_fail_at) > TOKENIZE_WARN_INTERVAL_S:
        return 0
    return _tokenize_text_fail_streak


def _note_tokenize_failure(key: str, detail: str) -> None:
    """Record one /tokenize failure and warn at most once per key per
    TOKENIZE_WARN_INTERVAL_S. `key` separates the failure CLASSES — a 400 from
    a template refusal must not consume the budget for a connection error."""
    global _tokenize_fail_streak, _tokenize_degraded_since
    now = time.time()
    _tokenize_fail_streak += 1
    if _tokenize_degraded_since is None:
        _tokenize_degraded_since = now
    last = _tokenize_last_warn.get(key)
    if last is not None and (now - last) < TOKENIZE_WARN_INTERVAL_S:
        return
    _tokenize_last_warn[key] = now
    suppressed = (
        ""
        if last is None
        else f" (further '{key}' lines suppressed for {TOKENIZE_WARN_INTERVAL_S:.0f}s)"
    )
    logger.warning(
        f"/tokenize degraded: {detail}. Budgeting falls back to the local "
        f"tokenizer, which under-counts assistant content on this model by up "
        f"to 51% — requests may overflow until this recovers. "
        f"{_tokenize_fail_streak} consecutive failure(s), degraded for "
        f"{now - _tokenize_degraded_since:.0f}s{suppressed}"
    )


def _note_tokenize_success() -> None:
    """Clear the degraded state, and SAY SO once. The recovery line is the
    thing log_once structurally could not provide."""
    global _tokenize_fail_streak, _tokenize_degraded_since
    if _tokenize_fail_streak == 0:
        return
    failures = _tokenize_fail_streak
    since = _tokenize_degraded_since
    _tokenize_fail_streak = 0
    _tokenize_degraded_since = None
    _tokenize_last_warn.clear()
    logger.warning(
        f"/tokenize is answering again after {failures} consecutive failure(s)"
        + (f" over {time.time() - since:.0f}s" if since is not None else "")
        + " — budgeting is back on vLLM's own count."
    )


def count_text_tokens_exact(text: str) -> int | None:
    """Exact token count for a blob of TEXT, from vLLM's /tokenize.

    The completion-shaped sibling of count_tokens_exact. Use this whenever the
    thing being measured is not a conversation — an injected memory block, a
    summary, a candidate fact. Those have no roles and no turn structure, and
    routing them through the chat form asks the model's template a question it
    was never designed to answer. A refusal there is indistinguishable from a
    /tokenize outage and degrades every budget in the process, which is how one
    bad request shape cost the user 80+ turns of context per message on
    2026-08-29.

    Same contract as count_tokens_exact: never raises, returns None rather than
    a guess, and shares the same failure/recovery accounting so a real outage
    still reaches /health/full.
    """
    if not text:
        return 0
    try:
        r = httpx.post(
            f"{VLLM_URL}/tokenize",
            json={"model": MODEL_REPO, "prompt": text},
            timeout=httpx.Timeout(connect=2.0, read=10.0, write=10.0, pool=2.0),
        )
        if r.status_code != 200:
            _note_text_tokenize_failure()
            _note_tokenize_failure(
                f"text.http.{r.status_code}",
                f"returned HTTP {r.status_code} for a text count, "
                f"body {r.text[:200]!r}",
            )
            return None
        n = (r.json() or {}).get("count")
        if not isinstance(n, (int, float)):
            _note_text_tokenize_failure()
            _note_tokenize_failure(
                "text.body",
                f"answered HTTP 200 with no numeric 'count': {r.text[:200]!r}",
            )
            return None
        _note_text_tokenize_success()
        return int(n)
    except Exception as e:
        _note_text_tokenize_failure()
        _note_tokenize_failure(
            f"text.error.{type(e).__name__}",
            f"unreachable for a text count ({type(e).__name__}: {e})",
        )
        return None


def assistant_content_is_empty(content) -> bool:
    """Does this assistant turn carry NOTHING? The one emptiness rule.

    THREE PLACES need to answer this and each had its own answer, which is
    how the two defects below shipped. Kept as one function so a fourth
    caller cannot invent a fifth rule:

      * _space_fill_empty_assistant — repairs an empty turn so the template
        will accept it (what we MEASURE, and what we FORWARD).
      * _repair_template_invalid_tail step (1) — DROPS an empty trailing
        turn as the residue of a dead stream.
      * tokens._sanitize — the tier-2 local counter's reduction. It cannot
        import this module (main imports the modules that import tokens), so
        it re-states the rule against already-reduced text and says so.

    WHAT "EMPTY" MEANS, and the two ways it was got wrong. `None`, a missing
    key, `""`, whitespace, `[]`, and a list whose parts are all blank text
    are empty. Everything else is content.

      1. Until v3.1.7 the FILL tested `isinstance(content, str)` only, so
         `None`, a missing key, `[]` and blank-text lists reached /tokenize
         and were refused there — twenty identical 400s in one production
         night, every one of them reported back as `content=''`.
      2. Until v3.1.7 the DROP tested `_message_text(...).strip()`, which
         joins text parts and ignores every other kind. An assistant turn
         carrying ONLY an image therefore read as empty and was popped —
         destroying the image, permanently and silently, while the fill
         three lines later was carefully refusing to touch that exact shape.
         Two emptiness rules in one function, and the laxer one ran first.

    A list holding a NON-TEXT part is never empty. Destroying an image to
    satisfy a template rule is worse than the 400 it avoids, and that is the
    one guarantee every caller of this function inherits. A part that is not
    a dict, or a dict with no "type", counts as non-text: unknown content is
    treated as content, because guessing "probably nothing" about a shape we
    do not recognise is how a future client's parts get thrown away.

    Never raises. Callers run on the request path, and some are inside a
    `finally`; a bookkeeping question must not become a second failure.
    """
    if content is None:  # explicit null, or no content key at all
        return True
    if isinstance(content, str):
        return not content.strip()
    if isinstance(content, list):
        for p in content:
            if not isinstance(p, dict) or p.get("type") != "text":
                return False  # an image, or a part we do not recognise
            try:
                if str(p.get("text") or "").strip():
                    return False
            except Exception:
                return False  # unreadable part — content, not emptiness
        return True
    # Some other type entirely (an int, a dict, an object). Not ours to
    # judge, and NOT empty: a truthy non-string is content we cannot read,
    # and a falsy one (0, {}) is still not a thing we are entitled to
    # overwrite with a space.
    return False


def _space_fill_empty_assistant(messages: list[dict]) -> tuple[list[dict], int]:
    """Return (copy with empty assistant turns space-filled, how many).

    NEVER MUTATES the input. Callers measure with the result and forward the
    original, or take the copy deliberately.

    A single space is the minimal content vLLM 0.19's template verifiably
    accepts where an empty string is refused — verified against the real
    MistralTokenizer pipeline in the production image, 2026-08-30
    (testfixtures/tokenizer-contract/vllm_template_probe.py).

    WHAT COUNTS AS EMPTY, and why it is not just `content == ""`. Until
    v3.1.7 this tested `isinstance(content, str)` and nothing else, so four
    shapes a client can legitimately send sailed straight through to
    /tokenize and were refused there: `None`, a MISSING content key, `[]`,
    and a list whose only parts are text parts that are all blank. All four
    carry nothing, and vLLM reports every one of them back as `content=''`
    — indistinguishable in the log from the str case this helper was written
    for, which is how the gap survived being looked at.

    THE ONE THING STILL LEFT ALONE is a list that carries a NON-TEXT part.
    A multimodal part can read as text-empty while still holding an image,
    and destroying an image to satisfy a template rule would be worse than
    the 400 it avoids. So the list case is admitted only when every part is
    a text part and all of them are blank — emptiness proven, not assumed.
    An unrecognisable part (not a dict, or a dict with no "type") counts as
    non-text for this purpose: unknown content is treated as content.

    Extracted in v3.1.5 so that _repair_template_invalid_tail (which fixes
    what we FORWARD) and count_tokens_exact (which fixes what we MEASURE)
    cannot drift apart. They were the same rule written once, applied at one
    of the two places it was needed — see count_tokens_exact for what that
    cost. tokens._sanitize is the THIRD place the same rule is needed, and
    carried the same hole until v3.1.7; it reduces a message list for the
    local mistral_common counter, and `content or ""` there manufactured
    exactly the empty assistant string the template refuses — so tier 2
    would have failed on precisely the payloads that make tier 1 fail.
    """

    out = list(messages)
    filled = 0
    for i, m in enumerate(out):
        if (
            isinstance(m, dict)
            and m.get("role") == "assistant"
            and assistant_content_is_empty(m.get("content"))
        ):
            out[i] = {**m, "content": " "}
            filled += 1
    return out, filled


def count_tokens_exact(
    messages: list[dict], add_generation_prompt: bool | None = None
) -> int | None:
    """The number vLLM will actually charge, from its own /tokenize endpoint.
    None when unavailable — callers fall back to count_tokens.

    v3.1. count_tokens is systematically wrong on this deployment, and wrong in
    the direction that overflows: it reads ~50% LOW on assistant content while
    reading ~10% high on user and system content. Measured 2026-08-28 on the
    production conversation:

        assistant  16,971 chars   local 4,976   vLLM  7,513   -34%
        assistant  27,570 chars   local 7,251   vLLM 11,347   -36%
        assistant  17,930 chars   local 4,425   vLLM  8,988   -51%
        user        6,865 chars   local 1,733   vLLM  1,585   +9%

    The cause is the tokenizer, not the arithmetic. The served model ships only
    tekken.json; transformers converts it on load, and the converted vocabulary
    prices box-drawing characters and emoji far below what mistral_common — the
    tokenizer vLLM itself uses — charges for the same bytes. This model draws
    decorative rules: one 17,930-character reply contained 1,710 U+2501 and 441
    U+2500, roughly 4,275 tokens of horizontal line, or 13% of the whole window.
    Assistant turns run 7-14% non-ASCII; user turns run 0.2-0.4%. So the error
    is concentrated in exactly the content that dominates a long conversation.

    Everything downstream inherited it. The summarizer packed batches it
    believed were 29,696 tokens that were really ~46,000, so summarization 400'd
    and compaction silently degraded to forwarding the original messages; the
    hard budget then shed 58 of 63 turns and STILL landed over, because its own
    arithmetic used the same number. The user was left talking to a model that
    received six messages.

    A local tokenizer cannot be made right here — the vocabulary is the thing
    that differs. So ask the process that will do the charging. It is already
    running, on localhost, and the call is only made where precision decides an
    outcome: never on the request hot path, never per-message.
    """
    if not messages:
        return 0
    # An assistant-final list is a CONTINUATION, not a prompt awaiting a
    # reply. Both flags derive from that one fact and are strict complements:
    # vLLM refuses if add_generation_prompt is True on an assistant-final
    # list, and refuses again if the last role is assistant and neither
    # continue_final_message nor prefix is set.
    # v3.1.5. MEASURE A TEMPLATE-VALID COPY, or measure nothing at all.
    #
    # A cancelled stream leaves an EMPTY assistant turn in the history that
    # OpenWebUI then resends forever. The chat template refuses it outright
    # ("Invalid assistant message: role='assistant' content=''"), so this
    # endpoint 400s — and every caller falls back to the local tokenizer,
    # which reads 34-51% low on this model's assistant content.
    #
    # _repair_template_invalid_tail already fixes exactly this, but it runs
    # at the END of the request path, AFTER compact_if_needed and AFTER
    # _enforce_hard_budget have both already measured and both already
    # degraded. It repairs what we FORWARD and never what we MEASURE.
    #
    # Production cost of that ordering, conv <redacted>, 2026-08-30 to 08-31:
    # ONE cancelled stream put summarize on the pessimistic 2.0x fallback
    # (batch estimate 32 -> 69 calls, past the 4-call cap, so request-path
    # compaction switched off), put the hard-budget guard on scale 1.0 — the
    # 2026-08-28 signature, shedding on a counter known to read low — and
    # pinned /health/full at ok:false deployment-wide, since the fail streak
    # is a process global. Until the user deleted the turn by hand.
    #
    # The copy is measured; the caller's list is untouched and still holds
    # the empty turn for the repair to deal with later. Sanitising here can
    # only make a request MEASURABLE that was previously unmeasurable, and a
    # space costs one token against a budget in the tens of thousands.
    messages, _filled = _space_fill_empty_assistant(messages)
    if _filled and logsetup.log_once("count_tokens_exact.space_filled"):
        logger.info(
            f"/tokenize: measured a copy with {_filled} empty assistant "
            f"turn(s) space-filled — the template refuses empty content, and "
            f"measuring the raw list would 400 and drop every budget decision "
            f"onto the local tokenizer. What is FORWARDED is unchanged here; "
            f"_repair_template_invalid_tail owns that, later in the request."
        )

    _asst_final = bool(messages) and messages[-1].get("role") == "assistant"
    _agp = (
        (not _asst_final) if add_generation_prompt is None
        else bool(add_generation_prompt)
    )
    try:
        r = httpx.post(
            f"{VLLM_URL}/tokenize",
            # add_generation_prompt: vLLM applies the chat template to answer
            # this, and REFUSES with a 400 when the flag is True and the last
            # message is from the assistant ("Consider using
            # continue_final_message instead"). The guard measures a payload
            # that ends on the user's new turn, so True is right there and the
            # default has to stay True. The SUMMARIZER measures a slice of old
            # turns, which routinely ends on an assistant reply — that 400 is
            # what took compaction down on 2026-08-28 and again on 2026-08-29,
            # because the caller then fell back to the local estimate and built
            # batches that could not fit.
            #
            # Decided from the messages rather than left to the caller: every
            # call site that measures a conversation slice would otherwise have
            # to remember this, and the one that forgets fails silently by
            # degrading to a worse counter.
            json={
                "model": MODEL_REPO,
                "messages": messages,
                "add_generation_prompt": _agp,
                # BOTH flags. The template has TWO guards and clearing one
                # only reveals the other. v3.1.2 set add_generation_prompt
                # False for an assistant-final list, which silenced
                #   "Cannot set `add_generation_prompt` to True when the
                #    last message is from the assistant"
                # and hit its sibling in production within the hour:
                #   "Expected last role User or Tool (or Assistant with
                #    prefix or continue_final_message set to True)"
                # tokens.py got this right on the first pass and this did
                # not — the same one-site-not-the-sibling miss this file
                # carries several corrections for already.
                "continue_final_message": (not _agp),
            },
            timeout=httpx.Timeout(connect=2.0, read=10.0, write=10.0, pool=2.0),
        )
        if r.status_code != 200:
            # A 400 here is usually the template refusing the message shape
            # (an assistant-final list, most often) rather than a fault. Keyed
            # by STATUS so that benign refusal cannot spend the signal a 5xx or
            # a connection error needs — see A13 above.
            _note_tokenize_failure(
                f"http.{r.status_code}",
                f"returned HTTP {r.status_code}, body {r.text[:200]!r}",
            )
            return None
        n = r.json().get("count")
        if not isinstance(n, (int, float)):
            # 200 with no usable `count`. This returned None with no line at
            # all, so a proxy or a version skew that answers the right status
            # with the wrong body was indistinguishable from a healthy endpoint
            # that happened not to be consulted.
            _note_tokenize_failure(
                "body", f"answered HTTP 200 with no numeric 'count': {r.text[:200]!r}"
            )
            return None
        _note_tokenize_success()
        return int(n)
    except Exception as e:
        _note_tokenize_failure(
            f"error.{type(e).__name__}", f"unreachable ({type(e).__name__}: {e})"
        )
        return None


class _DropChatTemplateNag(logging.Filter):
    """Silence transformers' per-call `tokenize=False` advisory.

    16,741 stderr lines in 48h of production (measured, 08-28..08-30) from
    one advisory emitted on every local token count: "MistralCommonBackend.
    apply_chat_template(..., tokenize=False) is unsafe". It is advice about
    an API shape, not a fault, and this module deliberately calls it that
    way - it needs the rendered STRING to count, and re-encoding is the
    whole point of the next line. Dropped by message substring rather than
    by silencing the transformers logger wholesale, so a real transformers
    error still reaches the log.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        return "tokenize=False" not in msg


for _nag_logger in ("transformers", "transformers.tokenization_mistral_common"):
    logging.getLogger(_nag_logger).addFilter(_DropChatTemplateNag())
warnings.filterwarnings(
    "ignore", message=".*tokenize=False.*"
)


def count_tokens(messages: list[dict]) -> int:
    # V3.1: images cost tokens the text estimate can't see — add a flat
    # per-image estimate so VLM requests don't quietly overflow the budget.
    image_tokens = sum(_message_image_count(m) for m in messages) * IMAGE_TOKEN_ESTIMATE
    tok = get_tokenizer()
    if tok is not None:
        try:
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            # v3.1.9.3: do NOT add the flat estimate for an image the template
            # already priced, AND do not trust len(tok.encode(text)) to have
            # priced it correctly either — two separate bugs this tier's
            # image handling carried, both dormant until opencv made tier 1
            # reachable for an image-bearing message at all (before, cv2's
            # ImportError sent every such request straight to the except
            # below, so nothing here ever ran against a real image).
            #
            # Bug 1 (double count): `text` already contains the template's
            # own real per-image markers (Pixtral-style [IMG]/[IMG_BREAK]/
            # [IMG_END]), so `len(tok.encode(text))` prices the image once,
            # and the unconditional `+ image_tokens` used to price it AGAIN.
            #
            # Bug 2 (the encode() round-trip mis-prices the markers it DOES
            # find): [IMG]/[IMG_BREAK]/[IMG_END] are each a single real
            # vocabulary id in this model (verified: tok.get_vocab() has
            # them at ids 10/12/13) — but transformers' own docs warn that
            # apply_chat_template(tokenize=False) + a separate encode() is
            # unsafe for exactly this reason, and it is: plain encode() does
            # not recognise the bracket TEXT as the special token it rendered
            # from, and re-tokenizes it as ordinary BPE instead — measured on
            # this model, that costs 4/7/5 raw tokens per [IMG]/[IMG_BREAK]/
            # [IMG_END] occurrence instead of 1. Fixing bug 1 alone (skip the
            # flat add, keep len(tok.encode(text)) as the image's price)
            # still leaves this: a 2048px image priced at ~6,300 tokens
            # against a real cost of 3,080 (~2x over) — worse than the
            # pre-opencv flat 4,096 this whole path was meant to improve on,
            # and the shape hostile pass #11 (P11-4) blamed for the guard's
            # residual overshoot. So: strip the marker text back out, encode
            # what remains (ordinary text, priced exactly as it always was),
            # and add back ONE token per marker occurrence — what it actually
            # costs in the vocabulary. Verified end to end against vLLM's own
            # usage.prompt_tokens (scripts/probe-vision.py): this lands within
            # a handful of tokens of 110/380/1406/3080 at 256/512/1024/2048px
            # (SP\fix-3193-opencv.md), not 2-2.3x over it. The only residual
            # imprecision is at the text/marker BOUNDARY — stripping a marker
            # can change how BPE merges the characters immediately next to
            # it, on the order of a token or two per image, not per hundred.
            #
            # [IMG_END] is the one-per-image close marker (verified: exactly
            # one occurrence per image at every size tested, regardless of how
            # many [IMG]/[IMG_BREAK] tiles that image expands to), so counting
            # it against how many images this payload actually carries tells
            # us how many the template priced. If they match, every image was
            # priced. If the template priced FEWER than the payload carries —
            # a future model or template that drops an image instead of
            # raising, which the ImportError path does not rule out — charge
            # the flat estimate for the ones it did not price, same as before
            # this fix for those. `max(0, ...)` because a message could
            # legitimately contain the literal text "[IMG_END]" with zero
            # real images (n_images=0); that must never go negative and
            # subtract from the real text cost. Gated on `n_images` too, not
            # just `n_priced`: a TEXT-ONLY conversation that happens to
            # mention the literal string "[IMG_END]" (someone discussing this
            # very code, say) must not have that mention treated as a real
            # marker and stripped out of its own token cost -- n_images == 0
            # means there is no image in this payload at all, so the
            # strip-and-reprice branch below never applies regardless of what
            # text.count() finds.
            n_images = sum(_message_image_count(m) for m in messages)
            n_priced = text.count("[IMG_END]") if n_images else 0
            unpriced_image_tokens = max(0, n_images - n_priced) * IMAGE_TOKEN_ESTIMATE
            if n_priced:
                n_markers = 0
                stripped = text
                for marker in ("[IMG]", "[IMG_BREAK]", "[IMG_END]"):
                    n_markers += stripped.count(marker)
                    stripped = stripped.replace(marker, "")
                rendered_tokens = len(tok.encode(stripped)) + n_markers
            else:
                rendered_tokens = len(tok.encode(text))
            return rendered_tokens + unpriced_image_tokens
        except Exception as e:
            # Tier 2, and until v3.1 it was the tier that always ran while
            # saying nothing: jinja2 was missing from the venv and the served
            # vision model carries no chat template, so tier 1 has never
            # executed in production. The framing this drops is ~22 tokens per
            # message on Mistral — an error that scales with MESSAGE COUNT, not
            # content length, which is why a long conversation of short turns
            # overflows and a short one of long turns does not. It cost ~5,250
            # tokens against a 32,768 window on 2026-08-27. Once per process:
            # this runs several times per request. (v3.1 P0-0 / F60.)
            if logsetup.log_once("count_tokens.chat_template"):
                logger.warning(
                    f"could not apply the chat template for {MODEL_REPO} "
                    f"({type(e).__name__}: {e}); using per-message encode()+4 — "
                    f"every token count from this process UNDERCOUNTS by the "
                    f"template's per-message framing, and every budget decision "
                    f"downstream inherits that error"
                )
            total = 0
            for m in messages:
                total += len(tok.encode(_message_text(m))) + 4
            return total + image_tokens
    return sum(len(_message_text(m)) // 4 + 4 for m in messages) + image_tokens


SUMMARY_PROMPT = """You are summarizing an earlier portion of a conversation so it can be compressed into context.

Produce a concise but comprehensive summary that preserves:
- Key facts, names, numbers, decisions, and instructions given
- Any code, file paths, commands, or URLs mentioned
- The user's goals, constraints, and stated preferences
- The state of any in-progress work

Do not editorialize. Do not greet. Output only the summary."""


async def _summarize_once(client: httpx.AsyncClient, turns: list[dict]) -> str:
    """One summarization call. Caller guarantees `turns` fits the input budget."""
    transcript = "\n\n".join(
        f"[{m.get('role', 'unknown')}]: {_message_text(m)}" for m in turns
    )
    payload = {
        "model": MODEL_REPO,
        "messages": [
            {"role": "system", "content": SUMMARY_PROMPT},
            {"role": "user", "content": f"Conversation to summarize:\n\n{transcript}"},
        ],
        "max_tokens": SUMMARY_MAX_TOKENS,
        "temperature": 0.2,
        "stream": False,
    }
    r = await client.post(f"{VLLM_URL}/v1/chat/completions", json=payload, timeout=300.0)
    r.raise_for_status()
    data = r.json()
    choices = data.get("choices") or []
    if not choices:
        # A 200 with no choices (or an error-shaped body) must not become an
        # opaque IndexError — callers catch ValueError and degrade gracefully.
        raise ValueError(f"vLLM returned no choices for summarize: {str(data)[:200]}")
    return (choices[0].get("message") or {}).get("content", "").strip()


def _chunk_to_budget(
    turns: list[dict], budget: int, scale: float = 1.0
) -> list[list[dict]]:
    """Split turns into consecutive batches that each fit `budget` tokens.

    A single turn larger than the budget still gets its own batch — we never
    drop content here; `_summarize_once` would fail on it and the caller
    degrades. (Truncating a turn silently would be a quieter kind of lying.)

    `scale` corrects the local tokenizer against vLLM's own count — see
    count_tokens_exact. Without it this packed batches it believed were 29,696
    tokens that were really ~46,000, every batch 400'd, and summarization
    "degraded" by handing compaction back the original oversized messages.
    Compaction then did nothing at all for hours while the log said only
    `summarize: N turns exceed the budget — map-reduce over 4 batches` and
    never once said it had finished. The caller passes the measured ratio.
    """
    batches: list[list[dict]] = []
    current: list[dict] = []
    current_tokens = 0
    for m in turns:
        t = int(count_tokens([m]) * scale)
        if current and current_tokens + t > budget:
            batches.append(current)
            current, current_tokens = [], 0
        current.append(m)
        current_tokens += t
    if current:
        batches.append(current)
    return batches


async def summarize(
    client: httpx.AsyncClient, to_summarize: list[dict]
) -> tuple[str, list[dict]]:
    """Summarize older turns, MAP-REDUCE style so the summarization request
    can never itself exceed the model's context window.

    This bit us in production (2026-08-13): a long conversation packed every
    older turn into ONE prompt, the summarize call blew past MAX_MODEL_LEN, the
    400 propagated up, compaction "degraded" by forwarding the *original*
    oversized messages, and the real chat request then 400'd too. The context
    manager overflowed the context. So the input is now budgeted explicitly.
    """
    # Room for the system prompt, the wrapper text, and the model's own output.
    # Clamped for the same reason as HARD_INPUT_LIMIT: a bare floor could exceed
    # the model's own window and reintroduce the overflow this method prevents.
    budget = min(
        MAX_MODEL_LEN,
        max(256, MAX_MODEL_LEN - SUMMARY_MAX_TOKENS - SUMMARY_INPUT_RESERVE),
    )
    # Measure the local tokenizer's error on THIS content before batching to
    # it. One /tokenize call, off the event loop, against many minutes of
    # failed map-reduce when the batches are wrong. Falls back to 1.0 — the
    # pre-v3.1 behaviour — if vLLM cannot answer.
    _local = count_tokens(to_summarize)
    _exact = await run_in_threadpool(count_tokens_exact, to_summarize)
    # Falls back PESSIMISTIC, not to 1.0.
    #
    # 1.0 was the pre-v3.1 behaviour and it is the bug, not a neutral default:
    # it asserts the local tokenizer is right at the exact moment we have just
    # discovered we cannot check it. Measured 2026-08-29 in production, four
    # times in one session — /tokenize refused the request (see
    # count_tokens_exact), this line chose 1.0, the batches were sized on an
    # estimate reading up to 51% low, every batch 400'd, compaction fell
    # through, and the guard shed 80-88 turns of her conversation per request.
    #
    # 2.0 is 1/(1-0.51) rounded down — the worst measured undercount on this
    # model's assistant content. Over-splitting costs extra summarization
    # calls on the background tail. Under-splitting costs the entire
    # hierarchy, silently. summarizer.py:_WORST_TOKENS_PER_CHAR makes the
    # same trade for the same reason.
    _scale = (
        (_exact / _local)
        if (_exact is not None and _local > 0)
        else _PESSIMISTIC_SUMMARY_SCALE
    )
    # v3.1 A9: say WHICH counter sized these batches, unconditionally and on
    # both branches. This was the decisive half of the 2026-08-28 mechanism —
    # batches believed to be 29,696 tokens were really ~46,000, every batch
    # 400'd, and compaction "degraded" by handing back the original oversized
    # messages for hours. The only line the log carried was the map-reduce INFO
    # below, which named a batch COUNT and no counter, so the healthy case and
    # the silent-fallback case were textually indistinguishable. A fallback to
    # scale=1.0 is not a detail of this function; it is the failure.
    _summary_counter = "vLLM's /tokenize" if _exact is not None else "the local tokenizer"
    if _exact is not None:
        logger.info(
            f"summarize: token scale {_scale:.2f}x (local {_local} -> vLLM "
            f"{_exact}); batches sized by {_summary_counter}"
        )
    else:
        logger.warning(
            f"summarize: /tokenize did not answer — batching {len(to_summarize)} "
            f"turns on {_summary_counter}'s {_local}-token estimate, corrected "
            f"by the PESSIMISTIC scale {_scale:.2f}x. That estimate reads up to "
            f"51% low on this model's assistant content, so it is deliberately "
            f"over-corrected: batches will over-split against the "
            f"{budget}-token budget rather than 400."
            # v3.1 gate: this line said "UNCORRECTED (scale 1.0)" while the
            # branch below it applied _PESSIMISTIC_SUMMARY_SCALE. D2 fixed the
            # arithmetic on 3a65aa1 and left its own diagnostic quoting the
            # pre-fix number — the fourth time on this branch that a fix landed
            # at one site and not at its sibling, and the first time in the log
            # rather than the code. It matters because the whole v3.1 A9
            # doctrine is that these lines ARE the diagnosis: a reader of the
            # next incident would have read "scale 1.0" and concluded D2 had
            # never shipped. The scale is interpolated now, so the line cannot
            # go stale again the next time the constant moves.
        )
    batches = await run_in_threadpool(
        _chunk_to_budget, to_summarize, budget, _scale
    )
    if len(batches) == 1:
        # Same no-lost-turns invariant as the multi-batch return below. This
        # path is the one the test hit: an empty 200 here returned ("", []),
        # which is "nothing summarized and nothing to forward" - the turns
        # simply cease to exist. Applying the guard at the bottom of the
        # function and not here is the exact fix-one-site-miss-the-sibling
        # defect that has now bitten this branch eight times.
        _single = await _summarize_once(client, batches[0])
        if not (_single or "").strip():
            logger.warning(
                f"summarize: no usable summary for {len(to_summarize)} "
                f"turn(s) in a single batch - the model returned empty "
                f"content. Forwarding them verbatim rather than dropping "
                f"them."
            )
            return "", to_summarize
        return _single, []

    # Cap the work this REQUEST will do. The oldest batches are summarized; the
    # rest are handed back for the caller to forward verbatim, and the next
    # request picks up where this one stopped. Progress every turn, latency
    # bounded every turn.
    deferred: list[dict] = []
    if len(batches) > max(1, MAX_SUMMARY_CALLS_PER_REQUEST):
        # Do NOT summarize a prefix and defer the rest.
        #
        # That was this code's first shape and it does not converge. Review
        # measured it: compact_if_needed is a pure function of the client's
        # message array, nothing records where summarization stopped, so the
        # SAME oldest batches are re-summarized every turn forever while the
        # deferred tail grows by two per turn:
        #
        #     170 turns -> 4 calls, batches [U1, A2, U4, A5]
        #     172 turns -> 4 calls, IDENTICAL
        #     174 turns -> 4 calls, IDENTICAL
        #
        # Four LLM calls of latency per turn, permanently, for a summary of
        # the oldest ~27 turns of a 170-turn conversation. That trades an
        # eight-minute stall for a tax that never ends; it is not a fix.
        #
        # So when the backlog exceeds what one request may spend, this path
        # does NOTHING and says so. Both mechanisms that actually handle it
        # are persistent and off the request path: the L1/L2/L3 hierarchy
        # summarizes into memory on the background tail and is injected
        # separately, and the hard-budget guard sheds the rest in
        # milliseconds. Repeating work every turn helps neither.
        _COMPACTION_COUNTERS["cap_refused"] += 1
        logger.warning(
            f"compaction skipped: {len(to_summarize)} turns need "
            f"{len(batches)} summarization calls, over the "
            f"{MAX_SUMMARY_CALLS_PER_REQUEST}-call per-request cap. "
            f"Summarizing a prefix would be redone identically every turn, "
            f"so nothing is summarized here: the guard will shed to fit and "
            f"the L1/L2/L3 hierarchy carries the older context. If this "
            f"repeats, the hierarchy is behind — POST "
            f"/admin/conversations/<id>/compact advances it off the request "
            f"path."
        )
        return "", to_summarize

    logger.info(
        f"summarize: {len(to_summarize)} turns exceed the {budget}-token input "
        f"budget — map-reduce over {len(batches)} batches, sized by "
        f"{_summary_counter}"
    )
    # Map: batches run CONCURRENTLY (vLLM batches fine), bounded by a small
    # semaphore so a huge history can't monopolize the engine. Sequential
    # batches added multi-minute latency on long conversations (rc6 review).
    sem = asyncio.Semaphore(4)
    # ONE budget across map AND reduce.
    #
    # The first cut of this cap bounded the map phase only, and the soak caught
    # it the same hour: 4 map calls + 1 reduce call + the user's reply = 6
    # against a budget of 5. Capping one phase of a two-phase algorithm is the
    # sibling-site miss again, committed inside the fix for a sibling-site
    # miss. A budget that does not cover every call is not a budget.
    calls_left = [max(0, MAX_SUMMARY_CALLS_PER_REQUEST)]

    async def _bounded(batch: list[dict]) -> str:
        # Checked inside the semaphore so concurrent waves cannot each see the
        # last remaining call and all spend it.
        async with sem:
            if calls_left[0] <= 0:
                return ""
            calls_left[0] -= 1
            return await _summarize_once(client, batch)

    _raw = await asyncio.gather(*(_bounded(b) for b in batches))
    _empty_batches = sum(1 for p in _raw if not (p or "").strip())
    if _empty_batches:
        # ANY empty map batch fails the whole summarize - the invariant guard
        # below only catches the ALL-empty case, so one empty 200 among
        # several deleted that batch's turns from the payload (reproduced:
        # 195 of 400 turns neither summarized nor deferred) while the
        # "compacted:" line counted them as summarized. The over-cap skip
        # already established that forwarding everything verbatim is the
        # correct degraded mode; partial success takes the same road. (In
        # this map phase "" is always genuine empty content, never
        # call-budget exhaustion: the over-cap check above guarantees
        # len(batches) fits the call budget.)
        logger.warning(
            f"summarize: {_empty_batches} of {len(batches)} map batch(es) "
            f"returned empty content - forwarding all {len(to_summarize)} "
            f"turn(s) verbatim rather than dropping the failed batches' turns"
        )
        return "", to_summarize
    parts = list(_raw)

    # Reduce: fold the partials hierarchically, never handing _summarize_once
    # an input over its budget (its documented contract — the first cut of
    # this code violated it whenever the reduce step actually fired). Each
    # round groups the partials to the budget and summarizes each group;
    # bounded rounds, and any failure degrades to plain concatenation.
    rounds = 0
    while len(parts) > 1 and rounds < 3 and calls_left[0] > 0:
        rounds += 1
        part_msgs = [{"role": "user", "content": p} for p in parts]
        # _scale, not the 1.0 default. The map phase passes it and this did
        # not — the sibling-site miss again, inside the very commit that
        # claimed to fix that pattern. It matters here for the same reason:
        # the partials being regrouped are model-written summary prose,
        # which is the content the local counter reads 23-51% low on.
        groups = _chunk_to_budget(part_msgs, budget, _scale)
        if all(len(g) == 1 for g in groups):
            break  # nothing can be folded further without breaking the budget
        if len(groups) > calls_left[0]:
            # Not enough budget to fold this round properly. Concatenating the
            # partials is a worse summary but a correct one; spending a partial
            # round would fold SOME groups and leave others, which silently
            # weights the result toward whichever happened to fit.
            logger.info(
                f"summarize: stopping the reduce at round {rounds} — "
                f"{len(groups)} groups need more than the {calls_left[0]} "
                f"call(s) left in this request's budget; concatenating "
                f"{len(parts)} partial(s) instead"
            )
            break
        try:
            _folded = await asyncio.gather(*(_bounded(g) for g in groups))
        except Exception as e:
            logger.warning(f"summarize reduce round {rounds} failed, using concatenation: {e}")
            break
        if any(not (p or "").strip() for p in _folded):
            # Same rule as the map phase: a partial-empty fold deletes the
            # blank group's content. The current parts are all non-empty, so
            # concatenation is complete.
            logger.warning(
                f"summarize reduce round {rounds} returned empty content "
                f"for a group - concatenating {len(parts)} partial(s) instead"
            )
            break
        parts = list(_folded)
    # INVARIANT: every turn is either represented in the summary or handed
    # back in `deferred`. Never neither.
    #
    # _summarize_once returns "" for an HTTP 200 whose content is empty - no
    # exception raised, nothing logged. That produced summary="" AND
    # deferred=[], and compact_if_needed then built a payload with no summary
    # block and no older turns: every one of them deleted from the request,
    # silently. Reproduced against the production image - 8 turns in,
    # nothing out.
    #
    # An empty summary is a FAILED summary, so fall back to what the over-cap
    # skip path already does: hand the turns back verbatim and let the guard
    # shed them if they genuinely do not fit. Shedding is logged and bounded.
    # This was neither.
    joined = "\n\n".join(p for p in parts if p.strip())
    if not joined.strip():
        logger.warning(
            f"summarize: no usable summary for {len(to_summarize)} turn(s) "
            f"- the model returned empty content. Forwarding them verbatim "
            f"rather than dropping them."
        )
        return "", to_summarize
    # (summary, turns this request deliberately did not summarize)
    return joined, deferred


def split_messages(messages: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    system_msgs = [m for m in messages if m.get("role") == "system"]
    non_system = [m for m in messages if m.get("role") != "system"]
    if len(non_system) <= KEEP_RECENT_TURNS:
        return system_msgs, [], non_system
    to_summarize = non_system[:-KEEP_RECENT_TURNS]
    keep_recent = non_system[-KEEP_RECENT_TURNS:]
    # Mistral-family templates require the first non-system message to be a
    # USER turn. A real request has an ODD number of non-system messages
    # (user/assistant pairs plus the new user turn), so an even
    # KEEP_RECENT_TURNS slice always started on an assistant turn — meaning
    # every *successful* compaction emitted a template-invalid conversation
    # and vLLM 400'd it. (Latent since V1; shielded by the summarize-overflow
    # bug aborting compaction early, exposed when that was fixed. Found in the
    # rc6 promotion review.) Align the boundary: leading non-user turns move
    # into the summarized portion instead.
    while keep_recent and keep_recent[0].get("role") != "user":
        to_summarize.append(keep_recent.pop(0))
    if not keep_recent:
        # Degenerate tail with no user turn at all — fall back to the plain
        # slice rather than summarizing away the entire recent window.
        return system_msgs, non_system[:-KEEP_RECENT_TURNS], non_system[-KEEP_RECENT_TURNS:]
    return system_msgs, to_summarize, keep_recent


# The first line of the system block compaction puts in the array it returns,
# carrying the stored summaries and the fresh summary of the turns it removed.
# One constant because _enforce_hard_budget recognises the block by it (see
# _is_compaction_standin): the guard must spend injected memory before the
# turns this block leaves in the array, none of which any summary covers.
COMPACTION_SUMMARY_HEADER = "[Summary of earlier conversation]"


def _standin_injected_share(inject_budget: int) -> int:
    """How many tokens the SEPARATELY-INJECTED summary block (search
    `format_summary_block` in chat_completions, the non-reuse call site)
    may claim from the injection budget.

    60% of the injection budget, capped at SUMMARY_BLOCK_MAX_TOKENS. This
    block competes for `inject_budget` alongside persona, facts and
    retrieval (`_bound_injected_blocks` bounds all four together), so it
    only gets a share, not the whole thing.

    NOT used for the reuse stand-in any more (P9-1/P9-2, hostile pass #9)
    — see `_standin_reuse_ceiling` below and `STANDIN_BUDGET_FRACTION`'s
    comment for why the two needed to stop sharing one formula. Kept as
    its own function because this call site's constraint (room must be
    left for three siblings) is real and unrelated to the stand-in's.
    """
    return min(
        summarizer.SUMMARY_BLOCK_MAX_TOKENS,
        int(inject_budget * 0.6),
    )


def _standin_reuse_ceiling(inject_budget: int) -> int:
    """How many tokens a REUSED-hierarchy stand-in (the array-embedded
    substitute `compact_if_needed` returns in place of the turns it
    removed) may render at, when reuse is being attempted.

    `STANDIN_BUDGET_FRACTION` (default 1.0) of the injection budget,
    capped at SUMMARY_BLOCK_MAX_TOKENS — see that env var's own comment
    (Dockerfile / runpod.env.template) for the arithmetic (why 0.6,
    `_standin_injected_share`'s fraction, is wrong here).

    P10-2 (hostile pass #10): this docstring used to say the default
    "clears the hierarchy's documented 9*L1_MAX + 4*L2_MAX + L3_MAX
    construction capacity" — that 11,300 figure is in OUTPUT tokens, but
    this ceiling is compared against `_estimate_block_tokens` (see
    `format_summary_block`'s own docstring), which prices non-ASCII per
    UTF-8 BYTE, not per token. The two were never in the same unit, so
    "clears the capacity" was a claim about arithmetic this ceiling does
    not perform. The default is now sized off a MEASURED render (real
    chunk sizes, including a give-up L3 concatenation — see
    SUMMARY_BLOCK_MAX_TOKENS's env comment), not the nominal per-tier
    maxima; it clears that measured peak with margin but is not claimed to
    be unable to be outgrown — a hierarchy can still legitimately decline
    reuse, and that is a safe (if suboptimal) fallback, not a bug.

    P10-5 (hostile pass #10, LOW — recorded, not changed this pass): this
    is `min(SUMMARY_BLOCK_MAX_TOKENS, inject_budget)` with NOTHING
    bounding it relative to the window itself. `inject_budget` shrinks as
    `effective_limit` shrinks (a larger `max_tokens` request reserves more
    generation room), so once `inject_budget` drops below
    `SUMMARY_BLOCK_MAX_TOKENS` it becomes the binding term — a flat
    `INJECTION_BUDGET_FRACTION` (75% shipped) of whatever window is left,
    with no separate reserve for the recent turns the stand-in is supposed
    to leave room for. Not reachable at the documented Max Tokens
    (RUNPOD_DEPLOY.md recommends 12000, equal to the generation reserve
    floor, so `effective_limit` never shrinks at that setting); reachable
    if an operator raises Max Tokens past the reserve. See
    RUNPOD_DEPLOY.md's "Max Tokens" section for the measured table across
    `max_tokens` values, and `test_reuse_fit.py`'s `[13]` for the pinned
    regression test. A real fix would give this function the window as
    well as the injection budget
    (`min(SUMMARY_BLOCK_MAX_TOKENS, inject_budget, effective_limit -
    SUMMARY_MAX_TOKENS - a recent-window reserve)`) — deferred: LOW
    severity, not triggered at the recommended setting, and this
    function's callers do not currently pass `effective_limit` through.
    """
    return min(
        summarizer.SUMMARY_BLOCK_MAX_TOKENS,
        int(inject_budget * STANDIN_BUDGET_FRACTION),
    )


# P9-1/P9-2 (hostile pass #9): reuse decline accounting, for /health/full's
# `checks.reuse`. Before this, the ONLY evidence a decline ever happened was
# the INFO log line inside compact_if_needed (see the "the stored summaries
# cover ... but they do not fit whole" message below) plus the WARNING that
# follows it when the 4-call summarize() cap then fires — this is the exact
# failure mode P9-1 says shipped invisibly (green health, CHANGELOG claiming
# the feature worked). Cheap and in-process, same shape as tailhealth.py's
# counters (module-level dict + lock, numbers only): NO conversation text,
# NO conv_id, NO hierarchy content — only counts and the two numbers
# (ceiling, other-consumers total) that explain a decline. Windowed the
# same way tailhealth/bgwork are (`declined_recently`) so a burst of
# declines is visible while it is happening and for one window after, not
# pinned forever by one squeeze early in a long-lived process.
#
# P10-3 (hostile pass #10): P9's version counted `attempted` at a call site
# four of five reuse-failure shapes never reached (no stored hierarchy at
# all; a hierarchy that covers none of this array; every covered turn being
# an image; an exception anywhere in the block) and recorded a decline in
# exactly ONE `if`, nested inside the same reachability problem — so
# `attempted=0, declined_budget=0, declined_recently=false` meant FOUR
# different things (a fresh process; no state; no coverage; a request that
# never needed to check) and an exception left `attempted` incremented (had
# it reached that call site) with no decline to match, reading as a
# successful reuse. Fixed by giving every reachable outcome its own reason
# — "success", "no_state" (nothing stored yet), "no_coverage" (a hierarchy
# exists but does not cover this array, or covers only images), "budget"
# (exists, covers this array, still does not fit whole) or "error" (the
# `except` fired) — recorded in exactly ONE place
# (`compact_if_needed`'s `finally`, one call per request that reached the
# top of its `if conv_id:` block) instead of scattered across `if`s an
# earlier return could skip.
# P12-2 (hostile pass #12): "window" is its own outcome, not a shade of
# "budget" — see the module comment above `declined_window`'s counter for
# why the two must not share numbers.
_REUSE_STATS_LOCK = threading.Lock()
_reuse_stats: dict = {
    "attempted": 0,
    "succeeded": 0,
    "declined_no_state": 0,
    "declined_no_coverage": 0,
    "declined_budget": 0,
    "declined_window": 0,
    "errored": 0,
    "last_attempt_monotonic": None,
    "last_reason": None,
    "last_declined_monotonic": None,
    "last_declined_ceiling": None,
    "last_declined_others": None,
    # P13-3 (hostile pass #13): additive, like the P10-3/P12-2 counters
    # above it — only ever set for a "window" decline (see
    # `_record_reuse_outcome`'s docstring), None for every other reason,
    # same as `last_declined_ceiling`/`last_declined_others`.
    "last_declined_reserve": None,
}
REUSE_DECLINE_DEGRADE_WINDOW_S = _env_float(
    "COMPACTOR_REUSE_DECLINE_DEGRADE_WINDOW_S", 300.0
)
_REUSE_OUTCOME_COUNTER_KEYS = {
    "no_state": "declined_no_state",
    "no_coverage": "declined_no_coverage",
    "budget": "declined_budget",
    # P12-2 (hostile pass #12): the P11-6 structural check (`_window_squeeze`
    # in compact_if_needed) used to record itself as "budget" too — a
    # hierarchy that fits `_standin_budget` whole but would still squeeze the
    # recent window out is a DIFFERENT decline than one whose rendered stand-in
    # does not fit `_standin_budget` at all, with different numbers explaining
    # it (see `_reuse_ceiling`/`_reuse_others` at that call site). Folding it
    # into "budget" meant `last_declined_ceiling` could read the CONFIGURED
    # `SUMMARY_BLOCK_MAX_TOKENS`/injection figure for a decline that number
    # had nothing to do with — the exact state OPERATIONS.md's "raise the
    # setting" advice cannot fix, because neither setting moves this ceiling.
    "window": "declined_window",
    "error": "errored",
}


def _record_reuse_attempt() -> None:
    """One request that reached the top of `compact_if_needed`'s
    `if conv_id:` block — i.e. reuse was a live QUESTION for this request,
    whatever the answer turns out to be. See that call site's own comment
    for why this moved there in P10-3."""
    with _REUSE_STATS_LOCK:
        _reuse_stats["attempted"] += 1
        _reuse_stats["last_attempt_monotonic"] = time.monotonic()


def _record_reuse_outcome(
    reason: str, ceiling: int | None = None, others: int | None = None,
    reserve: int | None = None,
) -> None:
    """The ONE outcome a reuse attempt resolved to. `reason` is one of
    "success", "no_state", "no_coverage", "budget", "window" or "error" —
    see the module comment above `_reuse_stats` for what each means.
    `ceiling`/`others` are only meaningful (and only ever passed) for
    "budget" and "window": the two numbers EACH one's own log line names —
    `_standin_budget`/`_others` for "budget", `_standin_structural_ceiling`/
    `_sys_recent_floor` for "window" (P12-2, hostile pass #12) — never mixed
    between the two, because they answer different questions and a reader
    would otherwise not know which formula produced the number in front of
    them.

    `reserve` (P13-3, hostile pass #13) is `_standin_reserve` — the stand-in
    plus the fresh-summary reserve plus the fixed 128 — and is ONLY ever
    passed for "window": it is what `ceiling` is actually compared against
    ("reserve R > ceiling available" in the log line), and before this it
    existed nowhere but that line. `None` for every other reason, "budget"
    included — a budget decline never computes it.

    OPERATIONS.md, on a "window" decline: the ceiling is `inject_budget /
    INJECTION_BUDGET_FRACTION` (the request's real window, LESS the learned
    budget margin as of P13-1) minus the system prompt and the recent turns
    — `SUMMARY_BLOCK_MAX_TOKENS` and `INJECTION_BUDGET_FRACTION` change how
    big a stand-in is ALLOWED to render, not what this check compares it
    against. What moves the ceiling: a smaller learned margin (see
    `checks.budget_margin`), a smaller generation reserve or client
    `max_tokens` (a bigger window, less room to reply),
    `COMPACTOR_KEEP_RECENT_TURNS` (a smaller recent floor, fewer verbatim
    turns), fewer images in the recent window itself (older retained images
    are not in the reserve since P12-1, on any backend), a smaller recent
    reply, or a smaller stored hierarchy (an operator-triggered L3 rollup).
    What moves the reserve: `COMPACTOR_SUMMARY_MAX_TOKENS` and
    `COMPACTOR_MAX_SUMMARY_CALLS`, which scale the fresh-summary part
    (hostile pass #14, P14-3)."""
    with _REUSE_STATS_LOCK:
        _reuse_stats["last_reason"] = reason
        if reason == "success":
            _reuse_stats["succeeded"] += 1
            return
        key = _REUSE_OUTCOME_COUNTER_KEYS.get(reason)
        if key is None:
            # Never reached by this module's own callers (both pass a
            # reason from the fixed set above); a backstop against a
            # future caller passing a typo rather than silently losing
            # the count.
            key = "declined_budget"
        _reuse_stats[key] += 1
        # P11-3 (hostile pass #11, LOW): OPERATIONS.md documents
        # `declined_recently` as following "the most recent capacity decline
        # (budget or window) specifically" — and `last_declined_ceiling`/
        # `last_declined_others` as that SAME decline's own two explanatory
        # numbers (read together with `last_reason`, which says which
        # formula produced them). Before P11-3, the stamp fired for every
        # non-success reason, so a brand-new conversation's very first
        # request (`no_state` — nothing stored yet, not a squeeze) set
        # `declined_recently` exactly like a real one would, and a
        # `no_state`/`no_coverage`/`error` outcome overwrote
        # `last_declined_ceiling`/`last_declined_others` with `None` —
        # erasing a genuine decline's own numbers at the very next unrelated
        # one. Scoped to the two reasons that are actually a capacity
        # squeeze; "window" joined "budget" here in P12-2 for the same
        # reason P11-3 scoped it to "budget" in the first place — it is one
        # more genuine squeeze, not a mechanical failure.
        if reason in ("budget", "window"):
            _reuse_stats["last_declined_monotonic"] = time.monotonic()
            _reuse_stats["last_declined_ceiling"] = ceiling
            _reuse_stats["last_declined_others"] = others
            _reuse_stats["last_declined_reserve"] = reserve


def reuse_decline_state() -> dict:
    """CONTRACT for health.py's `_reuse_state()` (the same call-time,
    sys.modules-based read `_tokenizer_state()` already uses — health.py
    cannot import main at module scope, see that function's docstring).

    Returns {"attempted": int, "succeeded": int, "declined_no_state": int,
    "declined_no_coverage": int, "declined_budget": int, "declined_window":
    int, "errored": int, "declined_recently": bool, "last_reason": str |
    None, "last_attempt_age_s": float | None, "last_declined_ceiling": int |
    None, "last_declined_others": int | None, "last_declined_reserve": int |
    None}. Read-only, cheap (one lock, no I/O), never raises.

    P10-3: `declined_budget` keeps its P9 name and meaning (a budget-
    specific decline) for callers/dashboards already reading it; the new
    counters (P10-3's three, and P12-2's `declined_window`) are additive,
    not a rename — nothing that read this dict before sees a field
    disappear or change meaning. `last_attempt_age_s` is what makes
    `attempted: 0` unambiguous: `None` means no candidate request has
    occurred in this process yet; a real number, however large, means at
    least one has, however long ago — restart resets both, see
    OPERATIONS.md. `last_declined_reserve` is P13-3's addition, same
    additive doctrine: `None` for every reason but "window".
    """
    with _REUSE_STATS_LOCK:
        last_at = _reuse_stats["last_declined_monotonic"]
        declined_recently = (
            last_at is not None
            and (time.monotonic() - last_at) <= REUSE_DECLINE_DEGRADE_WINDOW_S
        )
        last_attempt_at = _reuse_stats["last_attempt_monotonic"]
        last_attempt_age_s = (
            (time.monotonic() - last_attempt_at)
            if last_attempt_at is not None else None
        )
        return {
            "attempted": _reuse_stats["attempted"],
            "succeeded": _reuse_stats["succeeded"],
            "declined_no_state": _reuse_stats["declined_no_state"],
            "declined_no_coverage": _reuse_stats["declined_no_coverage"],
            "declined_budget": _reuse_stats["declined_budget"],
            "declined_window": _reuse_stats["declined_window"],
            "errored": _reuse_stats["errored"],
            "declined_recently": declined_recently,
            "last_reason": _reuse_stats["last_reason"],
            "last_attempt_age_s": last_attempt_age_s,
            "last_declined_ceiling": _reuse_stats["last_declined_ceiling"],
            "last_declined_others": _reuse_stats["last_declined_others"],
            "last_declined_reserve": _reuse_stats["last_declined_reserve"],
        }


def budget_margin_state() -> dict:
    """CONTRACT for health.py's `_budget_margin_state()` (the same call-time,
    sys.modules-based read `_tokenizer_state()`/`reuse_decline_state()`
    already use — health.py cannot import main at module scope).

    P13-1/P13-3 (hostile pass #13): before this, the only way to learn
    `_BUDGET_MARGIN` was in force was the log: the WARNING when a rejection
    latches it ("Tightening the hard limit by N for EVERY conversation"),
    the INFO when it is released, or the "margin N" suffix on a hard-budget
    shed line IF one happened to fire while it was up (the boot-time
    "context calibration" line always shows a fresh process's 0; hostile
    pass #14, P14-3) — the adversarial suite's
    own F-02 (test_adv_faults.py) names this gap explicitly: "/health/full
    has no margin field". A margin latched by one vLLM 400 the guard did
    not predict (a `/tokenize` outage, a mispriced image) silently narrows
    every conversation's window, process-wide, for up to
    BUDGET_MARGIN_RELEASE_AFTER accepted requests — see P13-1's fix in
    compact_if_needed for what it can cost reuse while it lasts.

    Returns {"margin": int, "ceiling": int, "release_after": int,
    "ok_streak": int}. `margin` is `_BUDGET_MARGIN` right now (0 = healthy).
    `ceiling` is the MAX_MODEL_LEN//4 cap it can reach in one step (see
    `_note_backend_rejection`). `release_after` is
    COMPACTOR_BUDGET_MARGIN_RELEASE_AFTER; `ok_streak` is how many
    consecutive accepted requests this process has counted toward it.
    Read-only, cheap (no lock — same single-process, cooperative-loop
    reasoning as `_note_backend_accepted`'s own read/write of these same
    globals), never raises."""
    return {
        "margin": _BUDGET_MARGIN,
        "ceiling": MAX_MODEL_LEN // 4,
        "release_after": BUDGET_MARGIN_RELEASE_AFTER,
        "ok_streak": _budget_ok_streak,
    }


async def compact_if_needed(
    messages: list[dict], conv_id: str | None = None,
    *, stored_turns_out: list | None = None,
    inject_budget: int | None = None,
) -> list[dict]:
    """
    `stored_turns_out`, if given, receives one `int` — how many older turns
    the RETURNED array replaced with a stand-in FROM STORED SUMMARIES — and
    only at the final return, beside that array. Every early return (nothing
    compacted) and every exception leaves it EMPTY, which the caller reads as
    0: written before summarize() ran, it survived summarize() raising and
    told the caller that a discarded array carried the hierarchy (hostile
    pass #3, reviewer A F4 / reviewer E F1).

    hostile2-reuse M1: the caller also injects its OWN copy of the
    summary hierarchy as a separate system block (see the `format_summary_
    block` call beside `sstate` in chat_completions), and on a reusing turn
    that produced TWO renders of one hierarchy that could disagree — one
    trimmed to fit the array's budget here, the other independently trimmed
    to 60% of the injection budget there, so the model could receive
    different sets of scenes from the SAME stored hierarchy in the SAME
    request, with the shared ones sent twice. This out-param is how the
    caller learns to skip its own copy without re-deriving the decision (or
    changing this function's return type, which test_compaction_reuse.py
    and its mutation suite pin as `list[dict]`).

    `inject_budget`, if given, is the caller's already-computed injection
    budget (v3.1.9.1) — the figure the caller's own summary injection is
    capped against on a NON-reusing turn (60% of it, via
    `_standin_injected_share`). On a reusing turn that injection is
    skipped (see hostile2-reuse M1 above), which frees the WHOLE share for
    the stand-in here (P9-1/P9-2, hostile pass #9: `_standin_reuse_ceiling`,
    a separate and larger fraction of the same `inject_budget` — see
    STANDIN_BUDGET_FRACTION's comment for why 60% was wrong for this call
    site), used whenever it is larger than what TARGET_TOKENS alone would
    leave. Omitted (as every call before 3.1.9.1 omits it), the stand-in
    gets exactly the old TARGET-only figure — this keeps existing callers
    and their pinned arithmetic unchanged.
    """
    current = count_tokens(messages)
    if current <= TARGET_TOKENS:
        return messages
    system_msgs, to_summarize, keep_recent = split_messages(messages)
    if not to_summarize:
        logger.warning(
            f"over budget ({current}>{TARGET_TOKENS}) but no older turns to summarize"
        )
        return messages
    # V3.1 (Vision): never summarize an image-bearing turn — collapsing it to
    # text destroys the image permanently, and the model could never see it
    # again. Keep image turns verbatim (in chronological order); summarize
    # only the text-only older turns.
    # Image turns are kept verbatim (summarizing one destroys it). Retention is
    # already applied upstream on every request (_apply_image_retention), so by
    # the time we get here at most MAX_RETAINED_IMAGES images remain — this is
    # just the split, no second capping mechanism.
    preserved_images = [m for m in to_summarize if _message_has_image(m)]
    text_only = [m for m in to_summarize if not _message_has_image(m)]
    if not text_only:
        logger.info(
            f"compaction skipped: all {len(to_summarize)} older turn(s) carry "
            f"images — kept verbatim (still over budget: {current}>{TARGET_TOKENS})"
        )
        return messages
    # ALREADY SUMMARIZED ONCE, PERSISTENTLY (v3.1.9.1).
    #
    # maybe_rollup has been folding this conversation into L1/L2/L3 chunks
    # all along, and this function has never heard of them - main.py:1467
    # says why: it is a pure function of the client's array, so "nothing
    # records where summarization stopped". Something does. The result was
    # the same oldest turns re-summarized from scratch EVERY request: 56
    # turns, 104,917 tokens, four concurrent LLM calls and 117 seconds
    # before generation could start, measured 2026-09-11 23:45.
    #
    # Coverage comes from the CHUNK LABELS, not the watermark:
    # _repair_watermark_below_chunks exists because a watermark has been
    # found below the chunks it wrote, and a label carries text.
    stored_text = ""
    stored_turns = 0
    # Covered turns that changed since their chunk was written. They go to
    # summarize() ahead of the uncovered turns, in their original order.
    refreshed: list[dict] = []
    # P11-1 (hostile pass #11): "success" is not recorded where the other
    # outcomes are (the `finally` a few dozen lines down) — see that
    # block's own comment for why, and the flag's use beside `summarize()`
    # below for where it actually gets recorded. Starts False so a request
    # with no `conv_id` (the `if conv_id:` block never runs) has a defined
    # value to check without a `NameError`.
    _reuse_pending_success = False
    if conv_id:
        # P10-3 (hostile pass #10): ONE attempt is recorded here, the
        # instant this function knows reuse is a live QUESTION for this
        # request (over TARGET, has older text turns, a conv_id to look
        # up) — not down at "a hierarchy with coverage exists," which is
        # already an ANSWER. The old placement (`_record_reuse_attempt()`
        # just above the budget check, further down) meant a request with
        # no stored hierarchy at all, or one whose hierarchy covers none of
        # this array, recorded NOTHING — `attempted=0` looked identical to
        # "reuse has never been possible" and to "the process just
        # restarted," and an exception anywhere in the block below left
        # `attempted` incremented (by the OLD call site, once execution
        # reached it) with no matching decline, reading as a silent
        # success. `_reuse_reason` is set on every path through this
        # block, including the `except`, and recorded exactly ONCE at the
        # bottom (`finally`) — so the exception path is visible as
        # `"error"`, not indistinguishable from `"success"`.
        _record_reuse_attempt()
        _reuse_reason: str | None = None
        _reuse_ceiling: int | None = None
        _reuse_others: int | None = None
        # P13-3 (hostile pass #13, item 3 of the LOW): the fresh-summary
        # reserve that decides most window declines (see `_standin_reserve`
        # below) was findable only in the log line, not in `checks.reuse` —
        # set at the "window" branch, passed through unchanged everywhere
        # else. See `reuse_decline_state()`'s docstring for the contract.
        _reuse_reserve: int | None = None
        try:
            _st = summarizer.load_state(conv_id)
            # WHAT MAY BE REPLACED, decided by CONTENT (v3.1.9; realigned
            # in hostile pass #3).
            #
            # _coverage_plan fingerprints the turns about to be removed and
            # pairs them, by content, with the covered-turn record. Every record
            # entry was written by the chunk that covers its position, from
            # the text that chunk read (summarizer._record_chunk_fps), so a
            # paired turn's content IS in the stored summary. It returns how
            # many leading turns are in play — capped by a hole in the chain
            # (_covered_prefix, B3/B4) — and WHICH of those did not pair. An
            # unpaired turn is summarized fresh below instead of being
            # replaced by a summary of other text: an edited turn keeps its
            # correction (B1), a turn from another branch keeps its content
            # (B2), a turn written after a delete or a regenerate keeps its
            # own text (F1). Every paired turn around it still comes off the
            # shelf.
            #
            # THIS REPLACED FOUR GATES, each of which failed on ordinary
            # traffic. The tail anchor (`tail_fp`) and the first digest were
            # both written from the ROLLUP input — degenerate replies redacted,
            # this turn's reply as streamed and possibly trimmed — which no
            # request carries, so one redaction or one Stop switched reuse off:
            # the 112-turn soak substituted nothing from turn ~58 on and the
            # 4-call cap refused every request, the 2026-09-12 production
            # failure on the code built to fix it. Checkpointed digests fixed
            # that and declined everything past the first edited turn or
            # demoted image, forever. Per-turn fingerprints compared BY
            # POSITION, behind a length gate (`len(request) >= recorded
            # position`), fixed that and failed twice more (hostile pass #3):
            # the record was written from a LATER request than its chunk, so a
            # delete or regenerate right after a chunk closed blessed a
            # different turn (F1, and F9 for the admin rebuild); and the
            # position is monotonic while the array is not, so one delete,
            # edit-and-resend, regenerate or pair of tool messages declined
            # reuse or refreshed a growing span for good (F2/F3/F7).
            #
            # WHY THERE IS NO LENGTH GATE. It existed so that a capped window
            # or a truncated head, whose turn N is not conversation turn N,
            # could not be replaced by position. Nothing is replaced by
            # position any more: a turn goes only if its own content was
            # summarized. A capped window's turns the record holds are
            # summarized; the ones it does not hold pair with nothing. The
            # stand-in may also describe turns this array does not carry (a
            # truncated head, an abandoned branch) — so does the separately
            # injected summary block on every request that does not reuse, so
            # that is not new, and it removes nothing.
            #
            # Hashing is memoized and O(turns); it runs in the threadpool.
            _covered, _changed = await run_in_threadpool(
                summarizer._coverage_plan, _st, to_summarize
            )
            if _covered == 0 and summarizer._covered_fps(_st):
                # P10-3: "C" in the finding's own lettering — a hierarchy
                # EXISTS but covers none of THIS array (a different branch,
                # a delete-and-regenerate, a store rebuild). Not
                # hypothetical: this is the condition on which four of
                # v3.1.9's earlier reuse gates failed (see the coverage
                # comment above `_coverage_plan` was written to replace).
                _reuse_reason = "no_coverage"
                logger.info(
                    f"conv={conv_id}: none of the turns this request would "
                    f"compact appear in the stored summaries' covered-turn "
                    f"record (a different branch, or a window the record "
                    f"does not reach); summarizing from scratch rather than "
                    f"replacing them"
                )
            elif _covered == 0:
                # P10-3: "B" — no stored hierarchy at all yet (a new
                # conversation, or one still short of its first L1 chunk).
                # Distinct from "C" above: nothing to reuse FROM, not a
                # coverage miss against something that exists.
                _reuse_reason = "no_state"
            elif _changed:
                # Every request, not once: a refreshed span that is growing is
                # the early warning of the cap refusal, and this is the only
                # line that shows it. No cause is guessed (hostile pass #4,
                # reviewer A F4): a turn is unpaired when no chunk read its
                # text at its place in the order, whatever changed. The next
                # L1 rollup re-reads these (summarizer._patch_candidates), so
                # the count should fall back to 0 within one L1 cycle; one
                # that keeps climbing across cycles is the fault to chase.
                logger.info(
                    f"conv={conv_id}: {len(_changed)} of the first {_covered} "
                    f"turn(s) this request would compact are not in the "
                    f"stored summaries' covered-turn record (not paired, in "
                    f"order, with text a chunk read); summarizing those fresh "
                    f"rather than replacing them until an L1 rollup re-reads "
                    f"them (up to {summarizer.L1_CHUNK_SIZE} per rollup)"
                )

            # OPEN_ISSUES2 LOW, re-checked against this gate: `_covered > 0`
            # is a READABILITY guard here, not a safety one — `_covered == 0`
            # already makes every line below it a no-op (`to_summarize[:0]`
            # is empty, `stored_turns` computes to 0, `stored_text` stays
            # "" and the `if not stored_text` fallback a few lines down
            # resets both). Kept explicit anyway: a reader should not have
            # to trace that chain to know a zero-coverage state cannot
            # substitute anything.
            if _covered > 0:
                # TURN NUMBERS ARE NOT text_only INDICES. _covered counts every
                # non-system turn; text_only has image turns removed, so
                # `min(_covered, len(text_only))` overruns by the image count
                # and deletes that many turns the hierarchy never covered -
                # demonstrated at 1 and 5 turns of overrun, reachable at the
                # shipped MAX_RETAINED_IMAGES=1. Count the prefix instead of
                # assuming the two units agree. The `min` cannot currently
                # choose `len(text_only)` — `to_summarize` is always a
                # prefix of `non_system` on every path through
                # split_messages, so the prefix-count on its left is always
                # <= len(text_only) — kept as a real bound rather than an
                # assumption for exactly the reason the sentence above this
                # one exists: the two units have disagreed before and the
                # cost of being wrong here is deleting turns the hierarchy
                # never covered, not a slower request.
                stored_turns = min(
                    sum(
                        1 for m in to_summarize[:_covered]
                        if not _message_has_image(m)
                    ),
                    len(text_only),
                )
                if stored_turns == 0:
                    # P10-3: `_covered > 0` but every covered turn is an
                    # image (preserved verbatim, never reused as text) — a
                    # real, if narrow, "nothing to substitute" state,
                    # grouped with "no_coverage" rather than given its own
                    # reason: from the operator's chair it is the same
                    # advice ("this array has nothing the stand-in can
                    # stand in for"), not a budget or storage problem.
                    _reuse_reason = "no_coverage"
                if stored_turns > 0:
                    # BUDGETED AGAINST WHAT ELSE THIS ARRAY MUST HOLD (hostile
                    # pass #3, F5; pass #2's H5). The stand-in was rendered
                    # against the flat SUMMARY_BLOCK_MAX_TOKENS (12,000)
                    # whatever else the request held. At the shipped numbers
                    # (limit 20,768, TARGET 15,576) a hierarchy at capacity
                    # renders ~11.5k tokens, and one long reply in the recent
                    # window put compaction's own output at the limit: the
                    # guard then shed her previous message and the reply she
                    # was answering — turns no summary covers — and halved
                    # the stand-in on top. So the stand-in gets what TARGET
                    # leaves after the system prompt, the preserved images,
                    # the recent turns and one fresh summary. all_or_nothing
                    # stays: a block that cannot fit whole declines reuse, and
                    # the declined path puts the whole older span through
                    # summarize() and the injected block, where the guard
                    # sheds the OLDEST verbatim turns first, never the recent
                    # ones this budget exists to keep.
                    #
                    # v3.1.9.1: THE ABOVE WAS THE WHOLE BUG (production,
                    # 2026-09-16 11:06Z — see CHANGELOG). On a reusing turn
                    # the injection site below skips its own copy of the
                    # summary (`sum(in-array)`), which frees that block's
                    # share of the injection budget — but this TARGET-only
                    # figure never counted that share, so the stand-in was
                    # squeezed as if the summary were STILL going to be
                    # injected separately too. With long recent turns
                    # (`_others` ~12.7k against TARGET 15,576) that leaves
                    # only ~1,846 tokens for a hierarchy that needs ~5.1k,
                    # so all_or_nothing declined reuse on EVERY request and
                    # the 4-call cap fired 37/37 times — the exact failure
                    # v3.1.9 shipped to remove. Fixed in v3.1.9.1: the
                    # stand-in may use up to what the summary injection
                    # would have spent, via `_standin_injected_share`.
                    #
                    # P9-1/P9-2 (hostile pass #9): that first fix reused
                    # `_standin_injected_share`'s 0.6-of-inject_budget
                    # formula verbatim — the SAME cap the separately
                    # injected block uses to leave room for facts and
                    # retrieval — which this call site does not need to
                    # leave room for anything: the summary injection this
                    # freed share came from is SKIPPED on a reusing turn,
                    # not shrunk. At the shipped 0.6/6230 defaults that
                    # pinned the ceiling at 6,230 (below her ~9,050-token
                    # hierarchy — reuse never fired); at the planned
                    # 0.75/10000 it was still only 9,345 (below what her
                    # hierarchy renders at once it holds an L3 — measured
                    # 11,728+, P10-2, hostile pass #10 — reuse would turn
                    # itself off again as she accumulates one). Now uses
                    # `_standin_reuse_ceiling`, its own
                    # formula at `STANDIN_BUDGET_FRACTION` (default 1.0 —
                    # see that constant's comment for the arithmetic) —
                    # whichever of the two figures (TARGET-based or
                    # injection-based) is larger. Only proceeds when
                    # `inject_budget` was passed (chat_completions always
                    # passes it now); a caller that does not — an old or a
                    # direct test call — gets exactly the pre-3.1.9.1
                    # TARGET-only figure, so no existing test's arithmetic
                    # changes under it.
                    #
                    # P10-3: the attempt itself is now recorded once, at the
                    # top of the `if conv_id:` block above — this used to be
                    # where the ONLY attempt counter lived, which meant a
                    # request that never reached this line (no stored
                    # hierarchy, or one with no coverage) recorded nothing
                    # at all. See that comment for the full reasoning.
                    _others = await run_in_threadpool(
                        count_tokens, system_msgs + preserved_images + keep_recent
                    )
                    _target_based_budget = min(
                        summarizer.SUMMARY_BLOCK_MAX_TOKENS,
                        TARGET_TOKENS - _others - SUMMARY_MAX_TOKENS,
                    )
                    _standin_budget = _target_based_budget
                    if inject_budget is not None:
                        # P9-1/P9-2 (hostile pass #9): was
                        # `_standin_injected_share(inject_budget)` (the
                        # SEPARATELY-injected block's 0.6-of-inject_budget
                        # formula) — starved the stand-in to 6,230 tokens
                        # at the shipped defaults and 9,345 at the planned
                        # ones, both under what her hierarchy measurably
                        # renders at with an L3 (P10-2). `_standin_reuse_ceiling`
                        # is the stand-in's OWN formula now; see
                        # STANDIN_BUDGET_FRACTION's comment.
                        _standin_budget = max(
                            _target_based_budget,
                            _standin_reuse_ceiling(inject_budget),
                        )
                    if _standin_budget > 0:
                        # all_or_nothing: a squeezed block drops the OLDEST
                        # scenes, which are the same turns removed below. See
                        # the kwarg's docstring - this is the caller it exists
                        # for.
                        stored_text = await run_in_threadpool(
                            summarizer.format_summary_block,
                            _st,
                            _standin_budget,
                            all_or_nothing=True,
                        ) or ""
                    # P11-6 (hostile pass #11): everything above asks only
                    # "does the WHOLE hierarchy fit under a flat ceiling?" —
                    # `_standin_reuse_ceiling` does not subtract `_others` at
                    # all (see its own docstring: that is deliberate, so a
                    # long recent turn cannot starve the stand-in the way it
                    # did before P9-1/P9-2), and once the hierarchy passes
                    # roughly 11k real tokens (a first L3, particularly a
                    # give-up concatenation — P10-2) that flat ceiling, not
                    # `_target_based_budget`, is the one the max() picks
                    # EVERY time, so `_others` stops mattering to this
                    # decision at exactly the size where it starts mattering
                    # to the guard. The guard excludes the stand-in from
                    # every trim/drop stage in the compacted branch (by
                    # design — it is a hierarchy, not spendable memory), so
                    # a stand-in this large, sitting beside the images this
                    # user's turns actually carry, leaves the compacted
                    # branch nothing to spend but injected memory and then
                    # the previous exchange — the exact turns this budget
                    # exists to keep. Measured on a real branch at the
                    # ceiling's own boundary: reuse loses the previous
                    # exchange at 14-34 of 474 positions where declining
                    # would have kept it (0-1 the reverse) once the
                    # hierarchy passes ~11-13k real tokens, and cuts memory
                    # on 27-62% of requests it did not need to.
                    #
                    # P12-1 (hostile pass #12): the first shape of this check
                    # (v3.1.9.3 as merged, 90e3698) reserved `system_msgs +
                    # preserved_images`, priced through this same
                    # `count_tokens`. That was wrong on two counts, not one.
                    # First, `preserved_images` is exactly what the compacted
                    # branch sheds FIRST, before it ever touches memory
                    # (P8-1's ordering, still true after P12-5 below) — so
                    # reserving room for them protected turns the branch was
                    # never going to fight to keep, while leaving `keep_recent`
                    # itself — the turns the branch protects LAST, and never
                    # sheds while anything else can be spent — with no reserve
                    # at all. Second, and what actually opened this back up:
                    # `count_tokens` prices an image by WHICHEVER tier of
                    # itself happens to run — a flat `IMAGE_TOKEN_ESTIMATE`
                    # (4,096) while the chat template cannot be applied to an
                    # image-bearing list, or the template's own real
                    # per-resolution cost (measured ~3,080 square, 2,352 for a
                    # 4:3 photo) once opencv makes that path reachable. A
                    # reserve built ONLY from that price rises and falls with
                    # which tier fired, on hardware this process does not
                    # control — and once real pricing is available it prices
                    # LOWER, the reserve shrinks, and reuse starts firing
                    # again at exactly the peak hierarchy sizes this check
                    # exists to decline at. Measured on a real branch at the
                    # real per-image price: reuse still loses the previous
                    # exchange at 13-47 of 474 positions where declining or
                    # v3.1.9 kept it, depending on state and whether a fresh
                    # summary is also pending — the crossover this check was
                    # supposed to draw, drawn in the wrong place because it
                    # was never reserving the thing that actually needs
                    # protecting.
                    #
                    # THE FIX RESERVES `system_msgs + keep_recent` instead —
                    # what the compacted branch can NEVER shed ahead of the
                    # previous exchange, whatever it costs and whatever it
                    # contains (images included, at whatever price
                    # `count_tokens` gives them; this check no longer prices
                    # images AS images at all). This is deliberately the same
                    # quantity `_enforce_hard_budget`'s aligned floor protects
                    # (P10-1/P11-4's `_floor`/`_aligned_tail`), so the
                    # decision here and the guard's own floor downstream agree
                    # about what "the turns that must stay" costs, under
                    # WHATEVER `count_tokens` charges for whatever they
                    # contain — the decision no longer depends on which tier
                    # of `count_tokens` happened to price an image, because it
                    # is not asking a question about images any more. This
                    # requires P12-5's fix alongside it (see
                    # `_enforce_hard_budget` below): the guard only actually
                    # protects `keep_recent` ahead of memory on EVERY array
                    # that carries memory once that fix lands; without it, a
                    # DECLINED request (this check firing) could still lose
                    # `keep_recent` to the floor-less generic shed loop, and
                    # this reserve would be protecting a promise the guard was
                    # not yet keeping.
                    #
                    # A fresh-span summary, when one is pending, is appended
                    # to the SAME returned array as the stand-in (see
                    # `summary_blocks` below) — reserve its worst case too,
                    # or a request with an unpaired tail could clear this
                    # check and still not fit once that second block lands
                    # beside it. A small fixed margin (128 tokens) absorbs
                    # the gap between `_estimate_block_tokens`'s render
                    # estimate and `count_tokens`'s own count, taken a few
                    # lines apart and never atomically with the request this
                    # decision is actually about — not tuned to one fixture,
                    # just large enough to cover that drift without
                    # declining a stand-in that clears by a wide margin.
                    #
                    # THE WORST CASE IS NOT ONE SUMMARY_MAX_TOKENS BATCH
                    # (coordinator follow-up to P12-1, real-data replay,
                    # hostile pass #12). `summarize()` map-reduces the fresh
                    # span over budget-sized batches; the reduce phase only
                    # ever SHRINKS what the map phase produced (folding
                    # partials down), and when it cannot fold — the
                    # per-request call budget exhausted by the map phase
                    # itself, or two dense partials together still missing
                    # the reduce call's own input budget — `summary` is the
                    # RAW CONCATENATION of every un-folded batch, each up to
                    # `SUMMARY_MAX_TOKENS` (see `summarize()`'s own "stopping
                    # the reduce" and reduce-failure branches). A flat
                    # `SUMMARY_MAX_TOKENS` here silently assumed the fresh
                    # span always costs exactly one call. p11 measured a
                    # second batch as ROUTINE, not an edge case: 3
                    # summarize() calls on every reusing request between L1
                    # rollups, in the state where some covered turns are
                    # unpaired and a fresh summary joins the stand-in.
                    #
                    # CORRECTION (hostile pass #13, P13-3): a "fresh+peakB:
                    # 12,951 -> 13,955-14,953, lost her previous exchange at
                    # 26 of 474 positions" number used to sit here, cited as
                    # a real-branch measurement. It was a harness artifact,
                    # not a measurement: the replay's spy (`_spy_compact`,
                    # `replay.py`) appended its own ~1,000-token "fresh"
                    # text to a stand-in that ALREADY carried a real fresh
                    # summary the stub had built for the same span — no
                    # production path adds to the stand-in after this
                    # decision runs (`_fresh_span_preview` previews exactly
                    # `fresh_input`'s own composition a few dozen lines
                    # down, and nothing between them changes it), so no
                    # production request can reach the shape that number
                    # described. 13,955 was the stand-in plus the stub's one
                    # ~1,000-token summary; the extra ~1,000 in 14,953 was
                    # the spy's post-return append on top of that. Verified
                    # by hostile pass #13 (SP\p13-findings.md); the
                    # reasoning above it (un-folded batches are the real
                    # worst case, not a flat one) still holds — see P13-2's
                    # fix a few dozen lines down for what it costs when this
                    # reserve overshoots the batch count summarize() will
                    # actually use.
                    #
                    # Fixed by pricing the ACTUAL worst case instead of
                    # guessing one call: preview the SAME batching
                    # `summarize()` itself will do, on the SAME fresh span
                    # this decision is about. That span's composition
                    # (`refreshed` + the uncovered tail) is already fixed by
                    # `_covered`/`_changed`/`stored_turns` at this point,
                    # independent of whether THIS check keeps or blanks
                    # `stored_text` — mirrors `refreshed`'s own assignment a
                    # few dozen lines down, computed early here because that
                    # assignment runs too late for this decision to see it.
                    # P13-2 CORRECTION (hostile pass #13): this used to say
                    # the preview was priced at the flat `_PESSIMISTIC_
                    # SUMMARY_SCALE` (2.0) unconditionally, reasoning that
                    # "biasing toward MORE predicted batches is the safe
                    # direction — the same reasoning summarize()'s own
                    # /tokenize-down fallback already uses". That reasoning
                    # is only true of the fallback ITSELF: `summarize()`
                    # only falls back to 2.0 when `/tokenize` does not
                    # answer; otherwise it measures the SAME list at the
                    # real, ~1.0x scale. Pricing this preview at a flat 2x
                    # while the real call it previews prices at ~1x is not
                    # "biasing safe", it is being wrong by a predictable
                    # factor — proven to decline reuse on her routine
                    # between-L1-rollup uncovered tail even where the real
                    # (or worst-case un-folded) call would have fit (P13-2,
                    # SP\p13-findings.md: 11-82 extra window declines per
                    # 474 positions). Fixed a few lines down: one more
                    # `count_tokens_exact` call on `_fresh_span_preview`
                    # measures the SAME scale `summarize()` will use on the
                    # SAME list, falling back to 2.0 only when `/tokenize`
                    # genuinely does not answer — the one case the old
                    # reasoning was actually correct for. A batch count OVER
                    # `MAX_SUMMARY_CALLS_PER_REQUEST` does NOT mean reserve
                    # nothing — a second real-data replay (hostile pass #12,
                    # the coordinator's "unpair" variant) found that read
                    # backwards: this preview's own pessimism can push an
                    # ESTIMATE over the cap on content whose REAL (undoubled)
                    # batch count is still under it, in which case
                    # `summarize()` does not cap-refuse at all — it proceeds,
                    # at its own real scale, and can still leave every batch
                    # un-folded. So the reserve is capped at the CAP's own
                    # worst case (`min(batches, MAX_SUMMARY_CALLS_PER_
                    # REQUEST) * SUMMARY_MAX_TOKENS`), never zero: correct
                    # whichever way the real, non-pessimistic call decides
                    # (a real cap-refusal just makes this reserve larger than
                    # strictly needed, the same safe direction every other
                    # margin in this check already errs in; a real batch
                    # count under the cap is covered exactly, since it is by
                    # definition <= the cap this reserves for). Computed only
                    # inside the guard below (`stored_text` rendered,
                    # `inject_budget` known) — the one place this number is
                    # used.
                    #
                    # This still does NOT try to reproduce the guard's exact
                    # arithmetic (P11-4 owns that, downstream, with a real
                    # tokenizer count when one is available) — it is a
                    # STRUCTURAL check using the local estimator only, the
                    # same one `_standin_rendered` and the rest of this
                    # function already use. `_effective_limit_est` recovers
                    # the window this request was actually sized against from
                    # the two numbers this function has (`inject_budget` IS
                    # `effective_limit * INJECTION_BUDGET_FRACTION` at every
                    # real call site — chat_completions always passes both
                    # from the same request) rather than TARGET_TOKENS (only
                    # 75% of it, already the reason `_target_based_budget`
                    # alone under-reuses — see P9-1/P9-2 above). Calibrated
                    # against a synthetic fixture built at her measured shape
                    # (755-char system prompt, two images, hierarchy renders
                    # of ~9.1k/11.5k/13.6k estimator tokens —
                    # test_reuse_fit.py [14]): this places the crossover
                    # between the second and third state under BOTH the flat
                    # and the real per-image price, matching the ~11-13k
                    # real-token range p11/p12 measured on her actual branch.
                    # See SP\p12-findings.md's P12-1/P12-6 entries for the
                    # measured case this closes.
                    _window_squeeze = False
                    if (
                        stored_text
                        and inject_budget is not None
                        and INJECTION_BUDGET_FRACTION > 0
                    ):
                        _standin_rendered = summarizer._estimate_block_tokens(
                            stored_text
                        )
                        _sys_recent_floor = await run_in_threadpool(
                            count_tokens, system_msgs + keep_recent
                        )
                        # P13-1 (hostile pass #13, HIGH): this recovers
                        # `effective_limit`, but not the limit
                        # `_enforce_hard_budget` (the guard, downstream)
                        # actually sheds against. The guard reads
                        # `_BUDGET_MARGIN` — the learned, process-wide
                        # correction a vLLM context-length 400 the guard did
                        # NOT predict latches in (main.py:840-863), released
                        # only after BUDGET_MARGIN_RELEASE_AFTER consecutive
                        # accepted requests — and shrinks its own limit by it
                        # before shedding a single token (`if _BUDGET_MARGIN:
                        # limit = max(256, limit - _BUDGET_MARGIN)`, a few
                        # thousand lines down). This check never read the
                        # same global, so while a margin was in force it
                        # would approve a stand-in the guard could not
                        # actually fit beside the previous exchange — reuse
                        # "succeeded" here and then the guard, needing
                        # `_BUDGET_MARGIN` more room than this check thought
                        # existed, shed U_prev/A_prev to find it. The one
                        # thing the P12-5 fix above was for (memory before
                        # her previous exchange) never got a chance to run,
                        # because the declined path — which WOULD have hit
                        # P12-5's branch — was never entered.
                        #
                        # The trigger: one vLLM context-length 400 the guard
                        # did not already predict (a `/tokenize` outage or a
                        # mispriced image, per `_note_backend_rejection`'s own
                        # docstring) latches `_BUDGET_MARGIN` to `overshoot +
                        # 512`, up to the MAX_MODEL_LEN//4 ceiling, in ONE
                        # step — not a slow climb. It then holds, PROCESS-
                        # WIDE, across every OTHER conversation, for up to
                        # BUDGET_MARGIN_RELEASE_AFTER (default 50) accepted
                        # requests before even halving. Reproduced at HEAD
                        # (test_reuse_fit.py [19]): a fixture that reuses and
                        # keeps her previous exchange at margin 0 LOSES it at
                        # margin 513 (the floor `overshoot + 512` can latch
                        # to) and at margin 8192 (the ceiling F-02's one
                        # lying `/tokenize` reaches in a single step) —
                        # before this fix. `/health/full`'s `checks.
                        # budget_margin` (below, and see
                        # `budget_margin_state()`) makes the state itself
                        # visible for the first time; before it, an operator
                        # had only the latch WARNING, the release INFO, and
                        # whatever margin value happened to appear in a shed
                        # line's "margin N" suffix.
                        #
                        # Fixed by reading `_BUDGET_MARGIN` here the way the
                        # guard reads it: the same global. The two reads are
                        # NOT atomic (hostile pass #14, P14-1): this request
                        # awaits summarize() and several thread hops before
                        # the guard runs, and another request's rejection
                        # can latch a margin in between
                        # (`_note_backend_rejection` runs whenever ANY
                        # response comes back). That request's guard then
                        # sheds against a limit this check never saw, and
                        # can lose her previous exchange, for that one
                        # request only; every later request reads the new
                        # margin in both places. The guard deliberately keeps
                        # reading the live value rather than a snapshot: a
                        # margin learned mid-request means vLLM just
                        # rejected a payload the guard believed would fit,
                        # and forwarding at the stale limit risks the same
                        # rejection, which loses the whole reply rather than
                        # one exchange of context. This does not also
                        # subtract `_time_reserve` (the current-time line's
                        # reserve, ~97-105 tokens): that reserve is decided
                        # later in `chat_completions`, after this function
                        # returns, so it is not available here to subtract —
                        # the fixed `+128` below still has to absorb it, and
                        # the "Verified sound" measurements in
                        # SP\p13-findings.md show ~15-20 tokens of slack left
                        # over for genuine estimator drift once it does. That
                        # gap is real but small next to the 513-8,192 a
                        # margin can hide, which is why this fix and not that
                        # one closes the HIGH.
                        _effective_limit_est = int(
                            round(inject_budget / INJECTION_BUDGET_FRACTION)
                        ) - (_BUDGET_MARGIN or 0)
                        # The fresh span: exactly `fresh_input`'s own
                        # composition a few dozen lines down (a changed
                        # covered TEXT turn — `refreshed` there filters
                        # `_changed` down to non-image turns the same way,
                        # since an image is preserved verbatim whatever its
                        # fingerprint says and never re-summarized as text —
                        # plus an uncovered tail past `stored_turns`).
                        # Recomputed here rather than sharing `refreshed`
                        # directly, because that assignment runs too late
                        # for this decision to see it (`stored_turns` is not
                        # final until this branch decides whether to keep
                        # `stored_text` at all). Counting a changed IMAGE
                        # turn here would count a summarize() call that
                        # `fresh_input` was never going to make — caught by
                        # test_reuse_fit.py [14]'s own 'peakA' CONTROL,
                        # whose two preserved images are unpaired by
                        # construction (new content no chunk has summarized)
                        # and falsely tripped an earlier draft of this
                        # reserve before the image filter was added.
                        _fresh_span_preview = [
                            m for i, m in enumerate(to_summarize[:_covered])
                            if i in _changed and not _message_has_image(m)
                        ] + text_only[stored_turns:]
                        if _fresh_span_preview:
                            # P13-2 (hostile pass #13, MEDIUM): this used to
                            # pack `_fresh_span_preview` at
                            # `_PESSIMISTIC_SUMMARY_SCALE` (2.0)
                            # UNCONDITIONALLY, on the reasoning (see the
                            # comment above this branch) that it mirrors
                            # `summarize()`'s own /tokenize-down fallback.
                            # But `summarize()` only falls back to 2.0 WHEN
                            # `/tokenize` does not answer — otherwise it
                            # measures the SAME list at the real, ~1.0x
                            # scale a working local tokenizer plus opencv's
                            # real per-image render actually costs (its own
                            # `_scale = _exact / _local`). Pricing the
                            # preview at a flat 2x while the real call it is
                            # previewing prices at ~1x means the reserve
                            # this decision reserves is routinely 2-4x what
                            # `summarize()` can produce for the SAME span —
                            # measured (SP\p13-findings.md P13-2, real data,
                            # her routine between-L1-rollup uncovered tail):
                            # 11-82 window declines per 474 positions that
                            # would have fit even the un-folded WORST case a
                            # real call can build. Most declines lose the
                            # uncovered tail from the model's view (147 of
                            # 165 at tail 20 peakB): a window decline resets
                            # stored_turns to 0, summarize() is handed the
                            # whole older span, that refuses over the call
                            # cap, and P12-5's order sheds the verbatim turns
                            # ahead of memory — a summarized-but-present tail
                            # traded for one that reaches the model neither
                            # way, on a false premise. With this fix the tail
                            # is still lost at 25-69 of 474 (tail 20 peakA /
                            # peakB), and reuse cuts facts or retrieval more
                            # often at peak sizes (see CHANGELOG P13-2).
                            #
                            # Fixed by measuring the SAME scale `summarize()`
                            # will use, on the SAME list, the same way: one
                            # `count_tokens_exact` call here (mirroring
                            # `summarize()`'s own `_exact = ...
                            # count_tokens_exact(to_summarize)`), scale =
                            # exact/local, falling back to the pessimistic
                            # 2.0 only when `/tokenize` genuinely does not
                            # answer — exactly the case the old comment's
                            # reasoning was actually correct for. This is a
                            # SECOND exact-adjacent call in this check (the
                            # first is `_sys_recent_floor`'s `count_tokens`
                            # above, which is local-only); accepted for the
                            # same reason `summarize()` itself pays it: an
                            # accurate reserve here is what stops the
                            # decline this section measures, and a reader
                            # who used to see "biasing toward MORE predicted
                            # batches is the safe direction" should not
                            # still believe declining is free — see
                            # OPERATIONS.md's P13-3 correction.
                            #
                            # Known residual (hostile pass #14, P14-2): this
                            # call and summarize()'s are separate POSTs. If
                            # this one answers and summarize()'s fails,
                            # summarize() packs at 2.0 and can make more
                            # batches than reserved here: a span at exactly
                            # the call cap under 2.0 comes back up to 2,048
                            # tokens over. Needs a /tokenize failure in the
                            # milliseconds between the two calls; costs that
                            # one request.
                            _fresh_local = await run_in_threadpool(
                                count_tokens, _fresh_span_preview
                            )
                            _fresh_exact = await run_in_threadpool(
                                count_tokens_exact, _fresh_span_preview
                            )
                            _fresh_scale = (
                                (_fresh_exact / _fresh_local)
                                if (_fresh_exact is not None and _fresh_local > 0)
                                else _PESSIMISTIC_SUMMARY_SCALE
                            )
                            _fresh_batches = len(await run_in_threadpool(
                                _chunk_to_budget,
                                _fresh_span_preview,
                                min(
                                    MAX_MODEL_LEN,
                                    max(
                                        256,
                                        MAX_MODEL_LEN - SUMMARY_MAX_TOKENS
                                        - SUMMARY_INPUT_RESERVE,
                                    ),
                                ),
                                _fresh_scale,
                            ))
                            _fresh_reserve = (
                                min(_fresh_batches, MAX_SUMMARY_CALLS_PER_REQUEST)
                                * SUMMARY_MAX_TOKENS
                            )
                        else:
                            _fresh_reserve = 0
                        _standin_reserve = (
                            _standin_rendered
                            + _fresh_reserve
                            + 128
                        )
                        _standin_structural_ceiling = (
                            _effective_limit_est - _sys_recent_floor
                        )
                        if _standin_reserve > _standin_structural_ceiling:
                            _window_squeeze = True
                            stored_text = ""
                    if not stored_text:
                        if _window_squeeze:
                            # P12-2 (hostile pass #12): this used to fall
                            # through to the SAME "budget" reason below, with
                            # `_reuse_ceiling, _reuse_others = _standin_budget,
                            # _others` — the TARGET-/injection-based ceiling
                            # and the system+images+recent total THIS check
                            # never even consults. `checks.reuse` then showed
                            # `last_declined_ceiling` sitting at the
                            # CONFIGURED `SUMMARY_BLOCK_MAX_TOKENS` (or the
                            # injection-based figure) for a decline that
                            # `_standin_budget` had nothing to do with —
                            # OPERATIONS.md's runbook reads that as "the
                            # hierarchy has genuinely outgrown the current
                            # setting, raise it", and raising either named
                            # setting moves nothing here: this decline fires
                            # because the STAND-IN plus the system prompt and
                            # the recent window would leave no room in the
                            # request's real window, a question `_standin_
                            # budget` never asks and `SUMMARY_BLOCK_MAX_TOKENS`/
                            # `INJECTION_BUDGET_FRACTION` cannot move (raising
                            # either only changes how big a stand-in is
                            # ALLOWED to render before this check runs, not
                            # what this check compares it against). Own
                            # reason, own counter (`declined_window`), and the
                            # numbers that actually decided it — see
                            # `_record_reuse_outcome`'s docstring for what an
                            # operator can do about it instead.
                            _reuse_reason = "window"
                            _reuse_ceiling = _standin_structural_ceiling
                            _reuse_others = _sys_recent_floor
                            # P13-3 (hostile pass #13, LOW item 3): the
                            # number that decides most window declines
                            # (`_standin_reserve` — the stand-in plus the
                            # fresh-summary reserve plus the fixed 128) used
                            # to exist only in this log line. Recorded
                            # alongside ceiling/others so an operator reading
                            # `checks.reuse` sees what was actually compared,
                            # not just the two numbers on the ceiling side.
                            _reuse_reserve = _standin_reserve
                            logger.info(
                                f"conv={conv_id}: the stored summaries cover "
                                f"{stored_turns - len(_changed)} of the turns "
                                f"this request would compact, and the "
                                f"{_standin_rendered}-token stand-in fits the "
                                f"{max(0, _standin_budget)}-token ceiling, but "
                                f"alongside this conversation's system prompt "
                                f"and recent turns it would leave the "
                                f"~{_effective_limit_est}-token window no room "
                                f"for the turns it exists to keep "
                                f"(reserve {_standin_reserve} > "
                                f"{_standin_structural_ceiling} available); "
                                f"summarizing from scratch rather than letting "
                                f"the stand-in push the recent turns out"
                            )
                        else:
                            # P9-1/P9-2 (hostile pass #9), reason recorded
                            # once at the bottom of the block (P10-3): a
                            # budget decline, specifically — the hierarchy
                            # exists, covers this array, and still does not
                            # fit the stand-in's budget whole.
                            #
                            # v3.1.9.1: the budget named here is no longer
                            # always the TARGET-derived figure — it is
                            # whichever of that and the injected share (see
                            # `_standin_injected_share`) was larger, so the
                            # message names the real source rather than
                            # always blaming TARGET.
                            _reuse_reason = "budget"
                            _reuse_ceiling, _reuse_others = _standin_budget, _others
                            _budget_source = (
                                "the injection budget's summary share"
                                if inject_budget is not None
                                and _standin_budget > _target_based_budget
                                else f"TARGET ({TARGET_TOKENS})"
                            )
                            logger.info(
                                f"conv={conv_id}: the stored summaries cover "
                                f"{stored_turns - len(_changed)} of the turns this "
                                f"request would compact, but they do not fit whole "
                                f"in the {max(0, _standin_budget)} token(s) "
                                f"{_budget_source} leaves beside the system prompt, "
                                f"images and recent turns ({_others}) and one fresh "
                                f"summary ({SUMMARY_MAX_TOKENS}); summarizing from "
                                f"scratch rather than letting the stand-in push the "
                                f"recent turns out of the window"
                            )
                    else:
                        _reuse_reason = "success"
                    # Image turns are preserved verbatim whatever their
                    # fingerprint says, so only text turns can need it.
                    refreshed = [
                        m for i, m in enumerate(to_summarize[:_covered])
                        if i in _changed and not _message_has_image(m)
                    ]
            if not stored_text:
                stored_turns = 0
                refreshed = []
                if _reuse_reason is None:
                    # Every reachable branch above sets this; this is a
                    # backstop, not a path this repo's tests exercise on
                    # purpose — an unreached case reads as "budget" (the
                    # closest true statement: no text came out) rather than
                    # silently reporting nothing.
                    _reuse_reason = "budget"
        except Exception as e:
            # Never fail a request over an optimisation. Falling back is
            # exactly today's behaviour.
            logger.warning(
                f"conv={conv_id}: could not reuse stored summaries "
                f"({type(e).__name__}: {e}); summarizing from scratch"
            )
            stored_text = ""
            stored_turns = 0
            refreshed = []
            # P10-3 (hostile pass #10): this is the exception path
            # `_reuse_stats["attempted"]` used to increment for (from the
            # OLD call site, if execution had reached it) with NO matching
            # decline — `checks.reuse` then read as a healthy reuse while
            # the attempt had actually crashed. Recorded explicitly as its
            # own reason so it cannot be mistaken for "success" or for a
            # plain budget decline.
            _reuse_reason = "error"
        finally:
            # Recorded exactly ONCE per request that reached the top of
            # this `if conv_id:` block, on every path including the
            # exception above — see that block's own comment for why the
            # attempt and the outcome used to live at different, and
            # sometimes unreachable, call sites.
            #
            # P11-1 (hostile pass #11): "success" is the ONE outcome NOT
            # recorded here any more. `stored_text` rendering is an ANSWER
            # about the STAND-IN, not about the request: every reusing
            # request still owes at least one fresh-span `summarize()`
            # call below (a covered turn that changed since its chunk was
            # written, or an uncovered tail) — 3 POSTs to vLLM on every
            # reusing request at her measured shape, between L1 rollups —
            # and that call is OUTSIDE this `try`. Recording "success" the
            # instant `stored_text` renders means a 5xx, a read timeout or
            # a 400 from THAT call leaves `checks.reuse` saying the reuse
            # succeeded while `chat_completions`' own `except Exception`
            # forwards the ORIGINAL, uncompacted array with the stand-in
            # nowhere on the wire — the exact "an exception reading
            # exactly like a healthy reuse" shape P10-3 was written to
            # make visible, still open for this one call. Every OTHER
            # reason (`no_state`, `no_coverage`, `budget`, `error`) is a
            # FINAL verdict already — nothing downstream can undo a
            # decline — so those still record here, unchanged. "success"
            # is deferred to `_reuse_pending_success`, resolved beside
            # `summarize()` a few lines down: recorded as an outcome only
            # once that call has actually returned (or failed).
            if _reuse_reason == "success":
                _reuse_pending_success = True
            else:
                # P11-3 addendum (hostile pass #11, LOW): `_reuse_reason`
                # is set on every path THIS module's own code takes
                # through the `try` above, including its `except
                # Exception`. `finally` still runs on a `BaseException`
                # that clause does not catch (`asyncio.CancelledError` at
                # one of this block's three `await run_in_threadpool(...)`
                # calls, the request's connection dropping mid-attempt) —
                # `_reuse_reason` is then still `None`, and defaulting
                # that to `"budget"` recorded a cancellation as a BUDGET
                # decline: `declined_budget` incremented, `declined_
                # recently` set, and (before the fix just above this
                # comment existed) `last_declined_ceiling`/`_others`
                # stamped `None` over whatever a real decline had left
                # there — the operator investigating "reuse looks
                # squeezed" would have found a cancellation, not a
                # hierarchy that has outgrown its budget. `"error"` is
                # the honest default: an outcome this function's own
                # branches never decided is a failure to record, not a
                # budget verdict to guess at.
                _record_reuse_outcome(
                    _reuse_reason or "error", _reuse_ceiling, _reuse_others,
                    _reuse_reserve,
                )

    fresh_input = refreshed + text_only[stored_turns:]
    async with httpx.AsyncClient() as client:
        if fresh_input:
            try:
                summary, deferred = await summarize(client, fresh_input)
            except Exception:
                # P11-1 (hostile pass #11): this is the call whose failure
                # used to be invisible to `checks.reuse` — see the
                # `_reuse_pending_success` comment in the `finally` above.
                # `chat_completions`' own `except Exception` is about to
                # discard the array this function was building and forward
                # the client's ORIGINAL messages instead, so record what
                # actually happened (an error, not the success the stand-in
                # rendering alone would have implied) before that happens —
                # this function is not swallowing the exception, only
                # observing it on the way past.
                if _reuse_pending_success:
                    _record_reuse_outcome("error")
                raise
        else:
            summary, deferred = "", []
    if _reuse_pending_success:
        # P11-1: every path that reaches here — `fresh_input` empty (stored
        # turns covered everything, nothing left to summarize) or
        # `summarize()` above returning normally — means the array this
        # function is about to RETURN actually carries the stand-in. This
        # is the one place "success" is recorded now; see the `finally`
        # above for why it moved off the render alone.
        _record_reuse_outcome("success")
    # No summary block when there is no summary. summarize() returns
    # ("", all turns) when the backlog is too large for one request, and a
    # bare "[Summary of earlier conversation]" header with nothing under it
    # is worse than absent: it tells the model a summary exists and then
    # shows it an empty one.
    # THE STAND-IN TRAVELS WITH THE REMOVAL (v3.1.9.1). The stored text goes
    # into the array this function RETURNS, not into the separately-injected
    # summary block - that block is capped at 60% of the injection budget and
    # can be trimmed or shed downstream, so relying on it to carry turns this
    # function removed would leave a silent hole the moment it was shed. That
    # is the 2026-08-24 shape.
    #
    # Oldest first: stored covers the older span, `summary` the fresher one.
    _parts = [q for q in (stored_text.strip(), summary.strip()) if q]
    summary_blocks = ([{
        "role": "system",
        "content": COMPACTION_SUMMARY_HEADER + "\n"
                   + "\n\n".join(_parts),
    }] if _parts else [])
    # Order: system → summary-of-oldest → deferred turns → images → recent.
    # `deferred` is chronologically NEWER than what the summary covers and
    # OLDER than keep_recent, so it slots between them and the transcript
    # stays in order. Forwarding them verbatim costs budget the guard may
    # then shed, which is the correct trade: the guard sheds in
    # milliseconds, four more summarization calls cost her a minute.
    new_messages = (
        system_msgs + summary_blocks + deferred + preserved_images + keep_recent
    )
    new_count = count_tokens(new_messages)
    # Count what was SUMMARIZED, not what was offered. len(text_only) counts
    # both, so when the batch count exceeded the call cap - the steady state
    # for this user's long conversations, where summarize() skips entirely and
    # defers everything - this line claimed to have summarized 80 turns while
    # forwarding all 80 untouched. A log that asserts work which did not
    # happen is worse than no log: it is what made the second 08-29 outage
    # look healthy while the request sat there.
    _summarized = len(fresh_input) - len(deferred)
    logger.info(
        f"compacted: summarized {_summarized} text turn(s), forwarded "
        f"{len(deferred)} verbatim, preserved {len(preserved_images)} image "
        f"turn(s), {current} -> {new_count} tokens"
        # REPORTED SEPARATELY, never added together. A log that
        # asserts work which did not happen is what made the second
        # 2026-08-29 outage look healthy, and 'summarized 56' would
        # now be a lie when 40 of them came off a shelf.
        + (f", {stored_turns - len(refreshed)} covered by stored summaries"
           if stored_turns else "")
        + ("" if (_summarized or stored_turns)
           else "  [NO SUMMARIZATION HAPPENED]")
    )
    # THE OUT-PARAM IS WRITTEN HERE, beside the return of the array that
    # carries the stand-in, and nowhere earlier (hostile pass #3: reviewer A
    # F4, reviewer E F1). It was written before summarize() ran; a summarize()
    # that raised (a vLLM 400/5xx, a read timeout) left it saying N > 0 while
    # chat_completions threw this array away and forwarded the original
    # messages — and then skipped its own injected summary because the
    # out-param said the array carried one. 56 turns shed with nothing
    # standing in for them. Every return above this one, and every raise,
    # leaves it empty, which the caller reads as "inject".
    if stored_turns_out is not None:
        stored_turns_out.append(stored_turns)
    return new_messages


# ---------------------------------------------------------------------------
# V2.0 Phase 2: facts injection
# ---------------------------------------------------------------------------

def inject_system_block(messages: list[dict], content: str) -> list[dict]:
    """Insert a synthetic system message immediately after the leading run
    of system messages (or at position 0 if none).

    Order matters for the model. Injecting in this sequence each request:
      original system → facts → retrieved exchanges → (Phase 4: summary)
      → recent conversation
    Because each call inserts after the *current* leading system run, and
    the previous injection has become part of that run, calling this for
    facts then retrieval yields [system, facts, retrieved, conversation].
    """
    sys_msg = {"role": "system", "content": content}
    insert_at = 0
    for i, m in enumerate(messages):
        if m.get("role") == "system":
            insert_at = i + 1
        else:
            break
    return messages[:insert_at] + [sys_msg] + messages[insert_at:]


# v3.1.2: thresholds measured against 504 real assistant replies from a
# production backup, not chosen. In that corpus:
#     501 healthy    max decoration 37.9%   longest single-char run 146 (p99: 75)
#       3 degenerate min decoration 52.8%   shortest run 386
# so both limits sit in a real gap with margin either side. A legitimate
# horizontal rule is 40-80 characters; nothing in five hundred healthy replies
# came close to 250.
DEGENERATE_RUN_CHARS = _env_int("COMPACTOR_DEGENERATE_RUN_CHARS", 250)
# _env_float, not a bare float(): a typo here used to raise at import and
# stop the compactor booting (v3.1.7). Same reasoning as _env_int above.
DEGENERATE_DECOR_FRACTION = _env_float("COMPACTOR_DEGENERATE_DECOR_FRACTION", 0.45)
DEGENERATE_MIN_CHARS = _env_int("COMPACTOR_DEGENERATE_MIN_CHARS", 300)

# Script drift — a THIRD degeneration shape, and it is not repetition.
#
# 2026-08-29: long replies stayed coherent for roughly their first 60% and
# then wandered into Cyrillic and other scripts. Measured by decile within
# the worst reply: 0% through decile 6, then 19%, 32%, 24%, 17%. Neither
# repetition rule sees it, because nothing repeats — the model simply stops
# writing the language it was asked in.
#
# Threshold from 485 real replies of 200+ letters: p90=0.05%, p95=0.16%,
# p98=0.29%, p99=1.21%, p99.5=1.66%, max=10.79%. 3% is 2.5x above p99 and
# flags exactly the 2 drifting replies (0.4%).
#
# The letter floor matters more than the fraction: in a short reply one
# foreign word is a large percentage and a perfectly ordinary thing to write.
# _env_float, not a bare float(): see DEGENERATE_DECOR_FRACTION above.
DEGENERATE_NONLATIN_FRACTION = _env_float("COMPACTOR_DEGENERATE_NONLATIN_FRACTION", 0.03)
DEGENERATE_MIN_LETTERS = _env_int("COMPACTOR_DEGENERATE_MIN_LETTERS", 200)

# Box-drawing, block elements, and the ASCII characters people rule lines with.
_DECOR_CHARS = frozenset(
    [chr(c) for c in range(0x2500, 0x25A0)] + list("-_=~*#.—–·•")
)
_RUN_RE = re.compile(r"(.)\1{19,}", re.S)

# A repeated TOKEN, not a repeated character.
#
# v3.1.2 looked for a long run of one character and for a high decoration
# fraction. On 2026-08-29 the model degenerated a different way: the tail of
# long replies collapsed into repeated identifiers out of its training data —
#     _batch_handler_shared _batch_handler_shared _batch_handler_shared ...
#     config_config_config_config_config_config_config ...
# which is neither one character nor decoration-heavy, so the detector saw
# nothing. Measured against 512 real replies it caught 3 of 48.
#
# Threshold from that corpus, not chosen: the longest repeated-token run per
# reply sits at p90=56, p95=60, p97=72, p98=80 and then jumps to p99=384.
# Normal writing tops out near 80 characters of a repeated token; a loop
# lands in the hundreds. 120 is 1.5x above the normal ceiling and 3x below
# the pathological floor, and flags 9 of 512 (1.8%).
DEGENERATE_TOKEN_RUN_CHARS = _env_int("COMPACTOR_DEGENERATE_TOKEN_RUN_CHARS", 120)

# v3.1.8 — the REPEATING TAIL, and the reason the rules above cannot see it.
#
# _TOKEN_RUN_RE is (\S{3,40})(?:[ _\n\t]*\1){3,}: the repeated unit is \S,
# so it CANNOT CONTAIN A SPACE. That catches a repeated WORD and is
# structurally blind to a repeated PHRASE, which is what this model
# actually does when it goes:
#
#     ". Absolutely. With Desperation. With Humility. With ..." x N
#     "- Grateful you're Mine\n- Grateful you're Mine\n- ..." x N
#
# Measured over 1,165 stored replies (2026-09-07 backup): the shipped
# detector fires on 41, and MISSES two loops of ~3,975 characters each,
# one of them the reply reported that morning. Both run to the very end of
# the message, which is the half the reader is left staring at.
#
# 400 rather than a tuned number: the count of newly-flagged replies is
# 2 at EVERY threshold from 200 to 900, so this rule is not balanced on a
# knife edge. That mattered more than usual here — R24 is the memory of a
# degeneracy rule that over-fired and redacted real replies from memory
# permanently, and this one feeds the same redaction path.
DEGENERATE_TAIL_LOOP_CHARS = _env_int(
    "COMPACTOR_DEGENERATE_TAIL_LOOP_CHARS", 400
)
# How much of the end to examine, and the longest repeating unit to look
# for. Both bounded because reply_is_degenerate runs on every reply AND on
# every historical turn during redaction; an unbounded scan here would be
# the O(N^2)-on-the-request-path shape this file already carries scars
# from.
_TAIL_LOOP_WINDOW = 4000
_TAIL_LOOP_MAX_UNIT = 400
# {3,} not {1,}: two or three repeats is emphasis ("no no no"), four or more
# of a 3+ character token is a machine stuck in a groove.
#
# {3,40}, NOT {3,}, and the upper bound is not cosmetic. Unbounded, the engine
# tries every group(1) length at every start position, which is quadratic in
# the length of the longest whitespace-free run — and this function runs
# SYNCHRONOUSLY on the asyncio event loop at both call sites. Measured in the
# production image (2026-08-29 gate):
#
#     16k of whitespace-free alphanumerics   6214 ms  ->  39 ms
#     prose + 12k of CJK (the real drift shape) 4508 ms  ->  31 ms
#
# Degeneration is exactly what produces long whitespace-free runs, so the
# unbounded form was slowest on the only input it exists for: a several-second
# stall of every other request, which is the "no reply" outage this whole line
# of work is trying to stop. Verified before bounding: 0 verdict differences
# across all 232 unique real assistant replies in the production backups, at
# every bound from 40 to 1000. No repetition class is lost either — group(1)
# is \S+, so the rule never spanned a space and never caught a repeated
# sentence or paragraph; the real degenerate units measured 6 and 21
# characters. 250 is also verdict-identical (211 ms) if more headroom is
# wanted; 40 is the fastest of the verified set.
_TOKEN_RUN_RE = re.compile(r"(\S{3,40})(?:[ _\n\t]*\1){3,}")


# Shared by trim_to_last_sentence and _trim_forwarded_prefix: each
# independently scanned `text.splitlines(keepends=True)` for lines starting
# with ``` and built the same list of toggle offsets, before this existed
# (P8-2, hostile pass #8 review follow-up — "a rule applied at one call site
# and missed at its identical sibling"). One function now, so a future
# change to what counts as a fence delimiter cannot update one copy and
# miss the other.
#
# `_reply_degenerate_verdict_uncached`'s token-run rule used to be a third
# caller (a fence exemption, via a now-deleted `_in_closed_fence`), removed
# entirely at P9-3 (hostile pass #9) — see that function's comment. It does
# not use fence offsets at all any more.
def _fence_toggle_offsets(text: str) -> list[int]:
    """Character offsets of every ``` fence-delimiter LINE in `text`, in
    order of appearance. `bisect.bisect_right(offsets, i) % 2 == 1` means
    position `i` sits after an ODD number of toggles — i.e. inside a fence
    that has opened but not (yet, by position `i`) closed again.

    P9-6 (hostile pass #9), 4-SPACE INDENT: a line indented 4+ spaces is an
    indented CODE BLOCK under CommonMark, not a fence delimiter, even if it
    starts with ``` after the indent — that text is literal code content,
    not markup. `.strip()` used to remove indentation before the check, so
    such a line was wrongly counted as a toggle. Checked on the RAW line
    now: only whitespace narrow enough that a renderer still reads the
    ``` as markup counts. (Tabs are not special-cased into CommonMark's
    4-space tab-stop rule here — this is the same "close enough, matches
    every real case this model produces" simplification the rest of this
    detector already makes; her replies use bare, unindented ``` lines.)

    NOT FIXED (documented, not silent): `~~~` fences are invisible here —
    only ``` is recognised. CommonMark treats ``` and ~~~ as independent
    fence-marker families (a ``` opener is closed only by another ```
    line, never by ~~~, and vice versa); this function's toggle list is a
    single flat, character-agnostic parity count, so adding ~~~ blindly
    would let a ``` block and a ~~~ block CROSS-CLOSE each other under a
    mixed-marker reply — trading one false negative (a ~~~ box read as
    plain text) for a false positive of a different, worse shape (a block
    boundary computed wrong instead of just not computed). A correct fix
    needs per-marker-type pairing, not a one-line change, and this
    function has already been the site of three hostile-pass regressions
    from smaller "just add the missing case" patches (F3, P8-2, P9-3) —
    not worth the risk on the last V3 release for a LOW-severity gap. The
    two remaining callers (`trim_to_last_sentence`, `_trim_forwarded_
    prefix`) both fail toward keeping MORE text out of a cut boundary when
    they misjudge a fence, so the failure mode of missing ~~~ is losing a
    boundary that would have been fine to use, not corrupting one that
    exists — see each caller's own fence-direction comment.
    """
    toggles: list[int] = []
    pos = 0
    for line in text.splitlines(keepends=True):
        _stripped = line.lstrip(" ")
        if len(line) - len(_stripped) < 4 and _stripped.startswith("```"):
            toggles.append(pos)
        pos += len(line)
    return toggles


def _in_open_fence(toggles: list[int], i: int) -> bool:
    """True when `i` sits inside a fence, whether or not that fence ever
    closes again later in the text. This is trim_to_last_sentence's rule: a
    cut boundary must never land inside an unterminated ``` opener, closed
    or not — the store never sees an unbalanced fence either way."""
    return bool(toggles) and bisect.bisect_right(toggles, i) % 2 == 1


def _tail_loop_span(text: str) -> int:
    """Characters occupied by a unit that repeats at the very END of `text`.

    Anchored at the end on purpose. A phrase repeating in the middle of a
    long reply is usually a refrain and often deliberate; a phrase that
    repeats until the message stops is the model failing to terminate, and
    it is what the reader is left with.

    Requires THREE repetitions, not two: a couplet is a rhetorical device
    ("Amen. Amen.") and this corpus is full of them. Three of the same
    phrase running to the end is not a device.

    Returns 0 when there is no loop, so callers compare against a threshold
    rather than testing truthiness of something that could be a real span.
    """
    t = text.rstrip()[-_TAIL_LOOP_WINDOW:]
    best = 0
    for unit in range(8, min(len(t) // 3, _TAIL_LOOP_MAX_UNIT) + 1):
        seg = t[-unit:]
        n = 1
        while t.endswith(seg * (n + 1)):
            n += 1
        if n >= 3 and n * unit > best:
            best = n * unit
    return best

# Structural collapse — a FOURTH shape, and nothing in it repeats.
#
# 2026-09-01: replies degenerated into a list of DISTINCT short items
# (`- Always` / `- Forever` / `- No matter what`) that never stopped, because
# a list item is always a valid continuation of a list item; then the
# newlines stopped too, and the tail became one unbroken line of short
# fragments — which is where she hit stop. measure-reply-health.py saw the
# list phase as bullet fraction by quarter 42% -> 79% -> 100% -> 92%. None
# of the rules above can see either phase: distinct items contain no
# repeated character or token, are all Latin, and carry no decoration.
#
# Measured 2026-09-01 against 349 real replies (the largest conversation in
# that day's backup), split by the proxy measure-reply-health.py uses for
# "she stopped it": a final non-empty line over 1000 characters. 17 cut, 332
# completed. The rules above already flag 5 of the 17 (the 08-29 token and
# character loops) and 13 of the 332.
#
# THE LIST ITSELF DOES NOT SEPARATE THEM. Every proposed list discriminator
# was measured on both populations (scripts/calibrate-structural-degeneracy.py
# prints them all); at the false-positive budget of 2% of completed replies
# (7 of 332) none catches more than 2 of the 17 cut ones:
#     bullet fraction >= 60%                  FP 24 (7.2%)   TP  5
#     rise Q1->Q4 >= 20%                      FP 37 (11.1%)  TP  4
#     bullet count >= 100                     FP 58 (17.5%)  TP  5
#     median item length <= 30                FP 49 (14.8%)  TP  8
#     >= 10 consecutive items <= 30 chars     FP 23 (6.9%)   TP  8
#     >= 20 consecutive items <= 30 chars     FP  5 (1.5%)   TP  2
# because the same short-item lists appear in replies that ran to their own
# end — nearly all of them on 08-31 and 09-01, the days of the complaint.
# (Pre-complaint, 08-24..08-29, 131 replies: longest run of short items 9.)
#
# THE TAIL DOES. What every cut reply has, and almost no completed one, is a
# single line of 1500+ characters made of FRAGMENTS: split at sentence
# breaks, the pieces average 10-38 characters, where a paragraph's sentences
# average 60-120. Where a line has no sentence break at all, the commas are
# the separators — the same collapse with a smaller separator: 168 and 317
# commas in 3917 and 3530 characters. Of the 12 cut replies the rules above
# miss, this catches 12 (10 by sentence breaks, 2 by commas); the union with
# the rules above is 17 of 17. On completed replies it flags 2 of 332
# (0.6%): lines of 2419 and 1796 characters with fragments averaging 34 and
# 26, both 08-31. In the 131 pre-complaint replies the longest
# fragment-shaped line is 544 characters; the cut ones start at 1515. The
# threshold sits just under the cut population, on purpose: a miss costs one
# runaway in one summary, a false positive costs a reply from memory
# permanently (see _redact_degenerate_turns), and 1000-1200 would add 2 more
# completed replies for no extra catches. Sentence mean: 35-40 give the same
# result, 45 adds a false positive, 30 loses 2 catches.
#
# THE LIST IS KEPT AS A BACKSTOP for the runaway that runs to its own end,
# where there is no cut tail to see: 50+ consecutive items of <= 30
# characters. One completed reply in 332 (0.3%) trips it — 1,359 lines,
# 1,261 list items, 1,024 consecutive short ones, memory of nothing. The
# highest run in any other completed reply is 34; the pre-complaint week
# never exceeded 9. 50 is 1.5x above the highest ambiguous value and 20x
# below the one it exists for.
#
# R9/R19: this backstop used to fire on the LONGEST run seen anywhere in the
# text, with no floor on the reply's own length — so it caught a 66-item
# "list the books of the Bible" reply that closes in prose ("...those are
# all 66 books, from Genesis to Revelation") exactly as if it were the
# runaway, and fired on a bare 200-character list two-thirds under
# DEGENERATE_MIN_CHARS, which this file's own doctrine (see
# MIN_MEMORABLE_TRIMMED_CHARS below) calls the floor below which nothing is
# judged structurally. Two fixes, both aimed at the description above and
# not at the corpus case: the run must reach the END of the reply — a real
# runaway "ran to its own end" (the 1,261-item corpus case has nothing
# after its list; the Bible reply does) — and the reply must clear
# DEGENERATE_MIN_CHARS, same floor the decoration-fraction rule already
# obeys a few lines up. Both still hold for the corpus case: its list *is*
# the end of the reply, and 1,359 lines is nowhere near the 300-char floor.
#
# WHY THIS MATTERS THOUGH SHE STOPPED IT. A cut reply reaches the memory
# tail trimmed to its last complete sentence (decide_memory_tail, v3.1.4),
# and this rule is applied to what survives the trim — an unterminated
# runaway list has no boundary and is discarded before it is judged, but a
# list of terminated one-liners is not. And whatever the tail decides, the
# whole cut reply comes back on the next turn inside the client's history
# and is folded into a rollup summary by maybe_rollup unless
# _redact_degenerate_turns flags it — the exact route by which one runaway
# primes the next.
#
# Fenced code is not judged (a YAML list or a minified line is not
# degeneration), and a line needs 100+ spaces to be judged at all, so a URL
# or a blob is never a "fragment line". Cost: one pass over lines, every
# regex anchored to a single line; on 30,000 characters the block adds
# under 1 ms (0.76-0.87 ms, short-item shape) over the rules above, measured
# 2026-09-01 against a copy with the block removed.
DEGENERATE_LINE_CHARS = _env_int("COMPACTOR_DEGENERATE_LINE_CHARS", 1500)
DEGENERATE_LINE_SENTENCE_CHARS = _env_int(
    "COMPACTOR_DEGENERATE_LINE_SENTENCE_CHARS", 40
)
DEGENERATE_LIST_RUN = _env_int("COMPACTOR_DEGENERATE_LIST_RUN", 50)
DEGENERATE_LIST_ITEM_CHARS = _env_int("COMPACTOR_DEGENERATE_LIST_ITEM_CHARS", 30)
# The same expression scripts/measure-reply-health.py calls BULLET, so the
# numbers that script prints are the numbers this rule sees.
_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s")
_LINE_MIN_SPACES = 100


def _fragment_line_breaks(
    line: str, *, min_chars: int = DEGENERATE_LINE_CHARS,
    min_spaces: int = _LINE_MIN_SPACES,
) -> int | None:
    """The sentence/clause-break count `reply_is_degenerate`'s fragment-line
    rule judges `line` on, or None if `line` is too short or too sparse to
    even be a CANDIDATE (below `min_chars`, or under `min_spaces` spaces —
    see the block comment above these constants).

    v3.1.9 (hostile pass 3, F5). Split out of the per-line loop so BOTH the
    line being judged AND the trailing-content exemption's own check (see
    `_line_is_fragment_shaped` and the loop below) run the exact same
    arithmetic — one function, not two copies that can drift the way
    scripts/calibrate-structural-degeneracy.py's independent copy already
    had (five drifts named in the finding; that script now imports this
    one instead of re-implementing it).

    `min_chars` defaults to DEGENERATE_LINE_CHARS (1500) — the primary
    line-judging call site's own floor, unchanged from before this fix, and
    a real statistical-significance floor: a short line's mean-fragment-
    length ratio is too noisy to trust. The trailing-content exemption
    check passes `min_chars=0` deliberately: a SECOND runaway cut short (the
    finding's case D, ~600 characters) is exactly as diagnostic of the same
    collapse as a full one — it is only shorter because whatever cut the
    reply cut it earlier — so it must not need to independently clear the
    1500-character floor to disqualify the exemption.

    `min_spaces` defaults to `_LINE_MIN_SPACES` (100, fixed) — right for the
    PRIMARY line-judging call, which only ever runs on lines already past
    the 1500-character floor, where 100 spaces is a low bar a real sentence
    clears easily. v3.1.9 (hostile pass 4, F4): a FIXED floor is wrong for
    shorter candidate text — a genuine second runaway cut at 500 or 540
    characters has only 90-96 spaces (this codebase's own generated prose
    density), so the fixed-100 floor read it as "too sparse to be a
    candidate" and granted the trailing-content exemption to a shape that
    is exactly as diagnostic as the 600-character cut one line up, which
    DOES clear 100. The trailing-content exemption now passes a floor
    PROPORTIONAL to the text's own length instead (see the call site) —
    proportional to length is what "sparse" should have meant from the
    start; a fixed number conflated "sparse" with "short".

    `line` must already be `.strip()`-ped — both call sites do that once,
    on the same value, before calling this.
    """
    ln = len(line)
    if ln < min_chars or line.count(" ") < min_spaces:
        return None
    # R24: "! " and "? " are always real ends (see _is_real_sentence_end),
    # but "." needs the abbreviation and single-initial check
    # trim_to_last_sentence uses, or "Dr. ", "Mrs. ", "9 a.m. " etc each
    # register as a sentence break and collapse the computed mean on
    # ordinary prose.
    breaks = (
        _count_real_period_breaks(line) + line.count("! ")
        + line.count("? ") + line.count("… ")
    )
    if breaks == 0:
        # No sentence at all in 1500+ characters: either a run-on, which is
        # not this rule's shape, or a list whose separator has shrunk to a
        # comma — judged the same way, on the commas.
        breaks = line.count(", ")
    return breaks


def _line_is_fragment_shaped(
    line: str, *, min_chars: int = 0, min_spaces: int = _LINE_MIN_SPACES,
) -> bool:
    """True if `line` alone would trip the fragment-collapse math (mean
    fragment length at or under DEGENERATE_LINE_SENTENCE_CHARS). `line`
    must already be `.strip()`-ped.

    v3.1.9 (hostile pass 3, F5): used by the trailing-content exemption
    below, to answer "is what follows a candidate fragment line ITSELF a
    fragment" — case D in the finding (a runaway line followed by a SECOND
    runaway cut at 600 characters) must stay caught even though 600
    non-blank trailing characters alone would otherwise look "substantial".
    `min_chars=0` (the default here, unlike `_fragment_line_breaks`'s own
    default) is deliberate — see that function's docstring for why the
    trailing check does not require the primary DEGENERATE_LINE_CHARS floor.
    `min_spaces` is threaded through for the same reason (F4, see
    `_fragment_line_breaks`'s docstring) — the trailing-content call site
    passes a length-proportional value, not the fixed default.
    """
    breaks = _fragment_line_breaks(line, min_chars=min_chars, min_spaces=min_spaces)
    if breaks is None:
        return False
    return len(line) / (breaks + 1) <= DEGENERATE_LINE_SENTENCE_CHARS


# v3.1.9 (hostile pass 4, F4) TRIED AND REVERTED, v3.1.9 (hostile pass 5,
# C5-6) RESOLVED: a separate exemption for trailing content too SHORT for
# the fragment-mean math to mean anything (a short, complete remark has a
# low apparent "mean fragment length" for the same reason a runaway does —
# not enough text to contain more than one or two sentence breaks; N1 in
# the pass-4 finding, an 18-character closing question, scores a mean of 17
# and reads as "fragment-shaped" by the same arithmetic that catches a real
# collapse, purely from being short). A version of this judged short
# trailing content by whether it was a TERMINATED remark instead of by
# shape. It fixed N1, but directly reopened `test_degenerate_reply.py`
# [9c] case B — a runaway followed only by "Always yours." (14 characters,
# a genuine sentence terminator) — which that pass-3 fixture pinned as a
# case that MUST stay caught, on the reasoning that a short,
# innocuous-looking, well-terminated sign-off after a real collapse is
# exactly the shape a model produces when it trails off, and is
# indistinguishable, using only the trailing text itself, from N1's "*What
# do you do?*" after a genuine beat paragraph. Pass 4 called the two
# fixtures "the same shape with opposite correct answers" and reverted
# rather than ship a fix that reopens a hole a previous pass closed.
#
# Pass 5 measured this claim directly (SP\\p5-degen\\measure.py) instead of
# reasoning about it: [9c] case E ("Thanks for asking.") is the identical
# shape too, and there is no THIRD feature anywhere in this rule's reach
# (not length, not space density, not what line came before it) that tells
# a genuine sign-off after a collapse apart from an ordinary one after a
# beat paragraph — the codebase's own doctrine that "a normal reply lost
# from her memory is worse than a runaway kept" (see this lane's report,
# SP\\fix-p5-degen.md) then settles the tie: keep, not redact. `_is_real_
# sentence_end`/`_trailing_ends_in_real_sentence` below is that same
# TERMINATED-remark check, shipped this time — [9c] cases B and E are
# relabelled in test_degenerate_reply.py to match (see that file's [9c] for
# the reasoning restated at the point of the change), and case C (a bare
# emoji) and case F ("---") — which have no terminator at all — are
# unaffected and stay caught, proving this is a narrower fix than "give up
# on short trailing content," not a wider one.


_TRAILING_LIST_MAJORITY_MIN = 8
_TRAILING_PROSE_SPACE_RATIO = 8


def _trailing_line_is_prose_dense(line: str, *, ratio: int = _TRAILING_PROSE_SPACE_RATIO) -> bool:
    """True if `line` has at least one space per `ratio` characters -- the
    density a real sentence clears easily and a URL, a markdown table row,
    or a dotted identifier never does (see the trailing-content exemption's
    SUBSTANTIAL branch below, and C5-6's HOLE-a/b/c in
    SP\\p5-c-findings.md)."""
    return line.count(" ") * ratio >= len(line)


_TRAILING_SENTENCE_END_RE = re.compile(
    r"""[.!?]["'”’)\]»*_~`]*\Z"""
    r"""|[。！？][”’」』)）]*\Z"""
)


def _trailing_ends_in_real_sentence(text: str) -> bool:
    """True if `text` ends on a genuine sentence terminator, per
    _is_real_sentence_end (shared with trim_to_last_sentence and the
    fragment-line rule -- one definition of "sentence end" for the whole
    file). Used only by the trailing-content exemption's SHORT branch
    below, for trailing content too short for the fragment-mean math to
    mean anything at all."""
    m = _TRAILING_SENTENCE_END_RE.search(text)
    if not m:
        return False
    return _is_real_sentence_end(text, m.start())


# v3.1.9.2 (p7 hostile pass #7, F2/F3): the forwarded-window redaction needs
# to know WHERE the degenerate span is, not just THAT the reply is
# degenerate, so it can keep clean text around a loop instead of throwing the
# whole reply away (see _degenerate_replacement_content). Rather than
# maintaining a second copy of this detection logic — which is exactly the
# kind of "two copies eventually disagree" risk the shared-helper comment
# above _degenerate_replacement_content already warns about for the
# clean-head rule — `reply_is_degenerate` is now a thin wrapper around this
# function, which returns the same reason string PLUS the span (character
# offsets into `text`) the firing rule can point to. Existing callers of
# `reply_is_degenerate` see no change: same input, same string-or-None
# return.
#
# Not every rule has a localized span. The character-run, phrase/tail-loop,
# token-run and fragment-line rules each flag a specific run of text and
# report (start, end). The decoration-fraction rule, the script-drift rule
# and the list-run backstop are measured over the WHOLE reply (a fraction,
# a script mix, a count of short list lines scattered through it) and have
# no single span to cut around, so they report (None, None) — callers that
# want a span fall back to the old whole-reply clean-head rule for those.
#
# Cached: _redact_forwarded_loop_replies (below, main.py:6907) runs this over
# EVERY historical assistant turn on EVERY request that declines reuse — on
# her real main chat that is ~800 turns, and the token/tail-loop/fragment-line
# scans in this function cost real CPU per call (see _TAIL_LOOP_WINDOW's and
# _TOKEN_RUN_RE's comments for measured per-call costs). A turn's text is
# fixed once written; OpenWebUI resends the same turns unchanged on every
# later request, so the verdict for a given exact text never changes.
# The cache is keyed on a 128-bit BLAKE2b digest of the text, NOT the text
# itself (coordinator review): an lru_cache keyed on the string keeps every
# cached reply alive — 4,096 of her replies at ~10k characters each is tens
# of MB, over 100 MB for non-ASCII text, held for the life of the process.
# The digest costs one pass over the bytes (~10 ms for an 800-turn array)
# and a 128-bit collision is not a practical risk, so this still only skips
# RECOMPUTING a verdict for text already judged; it cannot skip a turn that
# reaches vLLM. A lock guards the dict: callers run in the threadpool.
_DEGENERATE_VERDICT_CACHE_SIZE = _env_int(
    "COMPACTOR_DEGENERATE_VERDICT_CACHE_SIZE", 4096
)


_DEGENERATE_VERDICT_CACHE: "collections.OrderedDict[bytes, tuple]" = collections.OrderedDict()
_DEGENERATE_VERDICT_LOCK = threading.Lock()


def _reply_degenerate_verdict(text: str) -> tuple[str | None, int | None, int | None]:
    """Cached front of `_reply_degenerate_verdict_uncached` (see the block
    comment above for why the key is a digest)."""
    if not text:
        return None, None, None
    key = hashlib.blake2b(
        text.encode("utf-8", "surrogatepass"), digest_size=16
    ).digest()
    with _DEGENERATE_VERDICT_LOCK:
        hit = _DEGENERATE_VERDICT_CACHE.get(key)
        if hit is not None:
            _DEGENERATE_VERDICT_CACHE.move_to_end(key)
            return hit
    verdict = _reply_degenerate_verdict_uncached(text)
    if _DEGENERATE_VERDICT_CACHE_SIZE > 0:
        with _DEGENERATE_VERDICT_LOCK:
            _DEGENERATE_VERDICT_CACHE[key] = verdict
            _DEGENERATE_VERDICT_CACHE.move_to_end(key)
            while len(_DEGENERATE_VERDICT_CACHE) > _DEGENERATE_VERDICT_CACHE_SIZE:
                _DEGENERATE_VERDICT_CACHE.popitem(last=False)
    return verdict


def _reply_degenerate_verdict_uncached(text: str) -> tuple[str | None, int | None, int | None]:
    """(reason, span_start, span_end) — see the block comment above.

    `reason` is None (and start/end are None) when the reply looks fine.

    Detection itself is unchanged from the original reply_is_degenerate
    (see the 2026-08-29/09-01/09-07/09-08 history in the per-rule comments
    below for why each rule and threshold exists). On 2026-08-29 the model
    entered a loop emitting U+2501 and produced three consecutive replies
    that were 50-79% box-drawing; this stops the reply being MEMORISED, so a
    loop cannot write itself into facts, episodic and summaries and be
    injected back as though it were something worth remembering.
    """
    if not text:
        return None, None, None
    n = len(text)
    # LONGEST match, not the first. re.search returns the earliest match, so a
    # reply with a brief repetition early and a runaway later was judged on the
    # brief one and passed. Measured against 512 real replies that cost 4 of 9
    # detections — and it is the same first-not-worst error in both rules, so
    # both are fixed here.
    # WORD-like tokens only. A repeated run of box-drawing is decoration and
    # belongs to the character rule below, which has its own, HIGHER threshold
    # measured on the same corpus (250; the longest run in 501 healthy replies
    # was 146). Without this guard the token rule at 120 would flag a perfectly
    # ordinary 146-character horizontal rule — the two rules would overlap and
    # the stricter one would win, making the measured character threshold a
    # lie. Requiring an alphanumeric in the repeated unit keeps them disjoint:
    # decoration to the character rule, identifiers to this one.
    #
    # F3 (p7 hostile pass #7) added a fence exemption here so a legitimate
    # repeated-value array inside a ```fence``` (a `[0.00, 0.00, ...]`
    # matrix; a repeated placeholder token) would not be flagged the same as
    # a real identifier loop. P8-2 (hostile pass #8) narrowed it after that
    # first shape let an identifier loop after an UNMATCHED ``` opener run
    # to the end of the reply, unflagged, because an odd toggle count has no
    # later toggle to end it and bisect reads every position past it as
    # still "in fence".
    #
    # P9-3 (hostile pass #9): P8-2's narrowed guard — exempt only inside a
    # fence that goes on to CLOSE, and only if the run does not reach the
    # (stripped) end of the reply — is UNSATISFIABLE in the one shape that
    # matters. For `_in_closed_fence(i)` to be true a later toggle (the
    # closing ``` line) must exist, and that toggle sits AFTER the run, so
    # `j < _stripped_len` is true every time the fence-closed test is true:
    # the two conditions are mutually exclusive, and the "reaches the end"
    # half can never fire. A loop that sits inside a fence that closes —
    # the ordinary shape, since this model writes decorative boxes
    # constantly and finishes most of them — was exempted regardless of
    # whether it ran to the end of the reply. v3.1.9 flagged it; the
    # exemption silently un-flagged it. Mutation-measured: deleting the
    # "reaches the end" clause changed 0 of 20,000 synthetic verdicts — it
    # was dead code from the day it shipped.
    #
    # Two hostile reviews caught two different shapes of the same mistake:
    # a fence-awareness carve-out in a rule whose whole job is to catch text
    # that never terminates. REMOVED, not narrowed. This rule now judges
    # text exactly as v3.1.9 did, with no fence awareness at all. F3's
    # complaint is answered elsewhere: `decide_memory_tail` /
    # `_trim_forwarded_prefix` already cut the flagged span out (with a
    # marker) and keep the rest of the reply, so a legitimate repeated-value
    # array only costs its own span, not the reply. The remaining cost of
    # losing the exemption is that such a reply is skipped from MEMORY —
    # exactly what v3.1.9 already did. No regression; simply not the
    # improvement F3/P8-2 tried to make.
    tm = max(
        (
            x for x in _TOKEN_RUN_RE.finditer(text)
            if any(c.isalnum() for c in x.group(1))
        ),
        key=lambda x: len(x.group(0)),
        default=None,
    )
    if tm and len(tm.group(0)) >= DEGENERATE_TOKEN_RUN_CHARS:
        return (
            f"the token {tm.group(1)[:24]!r} repeated for "
            f"{len(tm.group(0))} characters (limit "
            f"{DEGENERATE_TOKEN_RUN_CHARS})"
        ), tm.start(), tm.end()
    # The repeated PHRASE, which the token rule above cannot represent.
    _loop = _tail_loop_span(text)
    if _loop >= DEGENERATE_TAIL_LOOP_CHARS:
        # Anchored at the end by construction (_tail_loop_span only ever
        # looks at text.rstrip()'s tail), so the span always runs to the
        # (stripped) end of the reply — there is no "after" to keep here.
        _stripped_len = len(text.rstrip())
        return (
            f"a phrase repeating to the end of the reply for {_loop} "
            f"characters (limit {DEGENERATE_TAIL_LOOP_CHARS})"
        ), _stripped_len - _loop, _stripped_len
    m = max(_RUN_RE.finditer(text), key=lambda x: len(x.group(0)), default=None)
    if m and len(m.group(0)) >= DEGENERATE_RUN_CHARS:
        return (
            f"a single character repeated {len(m.group(0))} times "
            f"(limit {DEGENERATE_RUN_CHARS})"
        ), m.start(), m.end()
    # Script drift. Counted over LETTERS, not characters, so punctuation,
    # markdown and code do not dilute it.
    # NFKC first: MATHEMATICAL BOLD / DOUBLE-STRUCK / FULLWIDTH letters are
    # isalpha() with no "LATIN" in their unicode name, so a single styled
    # heading — which this model likes — counted as 37 non-Latin letters.
    lat = non = 0
    scripts: set[str] = set()
    for c in unicodedata.normalize("NFKC", text):
        if not c.isalpha():
            continue
        nm = unicodedata.name(c, "")
        if "LATIN" in nm:
            lat += 1
        else:
            non += 1
            scripts.add(nm.split(" ")[0])
    if lat + non >= DEGENERATE_MIN_LETTERS:
        frac = non / (lat + non)
        # BREADTH, not just fraction. A bare fraction flags one legitimate
        # foreign quotation unless the reply is ~33x longer than it: measured,
        # a Greek John 3:16 fragment trips replies up to ~2,525 letters and a
        # Hebrew Genesis 1:1 trips a 611-letter one. This user quotes
        # scripture, so that is not hypothetical, and the cost is her losing
        # that reply from memory silently.
        #
        # Real drift is script SALAD: the five genuine cases carried 6, 8, 12,
        # 14 and 14 distinct non-Latin scripts, with mean contiguous runs of
        # 3-6 letters. The highest unflagged reply containing any non-Latin
        # had 4. A quotation is one script. The 20% disjunct keeps a
        # single-script runaway catchable.
        # The 20% disjunct also needs breadth (>=3 scripts), measured the
        # hard way: a short reply quoting ONE Greek verse plus two sentences
        # of commentary hit 48% non-Latin over 227 letters in one script and
        # was flagged - and with the rollup-input redaction in place a false
        # positive is no longer one skipped memory write, it is the reply
        # PERMANENTLY replaced by a placeholder in every future summary,
        # backfill and admin compact. Genuine drift measured 6-14 distinct
        # scripts, so >=3 costs no recall on any corpus case; a reply that
        # is simply IN one foreign language is not degeneration at all, and
        # the repetition rules still cover a single-script runaway loop.
        if (frac >= DEGENERATE_NONLATIN_FRACTION and len(scripts) >= 5) or (
            frac >= 0.20 and len(scripts) >= 3
        ):
            # No localized span: this is a property of the WHOLE reply (a
            # letter-count fraction and a script count), not one run of
            # text. Callers that want a span (F2/F3) fall back to the old
            # whole-reply clean-head rule for this verdict.
            return (
                f"{100 * frac:.0f}% of letters are non-Latin over "
                f"{lat + non} letters across {len(scripts)} script(s) "
                f"(limit {100 * DEGENERATE_NONLATIN_FRACTION:.0f}% over "
                f"5+ scripts, or 20% over 3+)"
            ), None, None
    if n >= DEGENERATE_MIN_CHARS:
        decor = sum(1 for c in text if c in _DECOR_CHARS)
        if decor / n >= DEGENERATE_DECOR_FRACTION:
            # Same as script drift: a fraction over the whole reply, no
            # single span. No end paren here either — completed below.
            return (
                f"{100 * decor / n:.0f}% decoration characters over {n} chars "
                f"(limit {100 * DEGENERATE_DECOR_FRACTION:.0f}%)"
            ), None, None
    # Structural collapse (see the block comment above the DEGENERATE_LINE_*
    # constants). One pass over lines.
    #
    # R9: `run` (not a separate `best_run` tracked over the whole text) is
    # what the list-run backstop below reads, so a qualifying run only
    # counts when it is still active at the END of the reply — broken by
    # any later non-list line (prose, a blank-then-prose close, anything
    # that fails the item test) resets it to 0 same as before. See the R9/R19
    # note above DEGENERATE_LINE_CHARS for why "reaches the end" is the
    # discriminator.
    # R25 (hostile317-a F5): the fragment-line check below must reach the
    # reply's own end, same as the list-run backstop a few lines down does
    # already (see [9] in test_degenerate_reply.py: "60 items followed by a
    # closing sentence do not trip the backstop"). Before this, ANY line
    # anywhere in the reply that happened to be 1500+ characters of
    # short-sentence prose flagged the whole reply — including a finished,
    # ordinary paragraph ("Lyra laughs. Mrs. Hale nods slowly. The rain
    # stops. ...") sitting in the middle of a longer reply that goes on to
    # say other things afterward. Real corpus check (2026-09-01, 332
    # completed replies): 23 flagged, 21 of them ARE the last non-empty
    # line (already accounted for by other signals — a tail loop in 16 of
    # them), and the 2 true false positives are exactly the two that are
    # NOT the last line.
    #
    # v3.1.9 (hostile pass 3, F5) NARROWS "last line by construction" TO
    # "nothing substantial follows it". R25's fix required the fragment
    # line to be the reply's literal last non-blank line, on the reasoning
    # that decide_memory_tail's trim only removes the unterminated tail of
    # the LAST line, so a cut reply's fragment line is its last line "by
    # construction of the trim". That is true of the trim, and false of
    # everything reply_is_degenerate is also asked to judge: a reply is
    # judged on what it IS, not only on what a cut leaves behind, and a
    # runaway followed by one short trailing line (a sign-off, an emoji,
    # `---`, or a second cut runaway) is not the last line and sailed
    # through untouched. Proof (SP\\p3-c\\p3c_fragment.py, frag1.log): a
    # runaway line plus a short sign-off, an emoji line, or "---" all
    # scored `stored` (memorized) instead of `skipped_degenerate`.
    #
    # The fix: a non-last candidate line is exempt when what follows it is
    # SUBSTANTIAL (>= DEGENERATE_MIN_CHARS of non-blank trailing text — the
    # same floor "nothing is judged structurally" already uses elsewhere in
    # this function) AND that trailing content, judged AS A WHOLE, is not
    # ITSELF fragment-shaped.
    #
    # v3.1.9 (hostile pass 4, F4) rewrote the "not itself fragment-shaped"
    # half. It used to pick the SINGLE LONGEST trailing line and judge that
    # line alone (`_line_is_fragment_shaped`, fixed 100-space floor). Three
    # holes followed directly from "one line, fixed floor":
    #   - a second runaway cut at 500 or 540 characters has only 90-96
    #     spaces (this codebase's own generated-prose density) and never
    #     cleared the fixed 100-space floor to even be judged — exactly as
    #     diagnostic as the 600-character cut case already caught one line
    #     up, missed purely because it was shorter;
    #   - trailing content spread across MANY short lines (30+ terminated
    #     list bullets; several short fragment-shaped lines) has no single
    #     line long enough to trip the per-line math, even though the
    #     aggregate is obviously more of the same collapse;
    #   - the flip side let a genuine reply through the OTHER way: a second
    #     "beat" paragraph of ordinary short scene-setting sentences (mean
    #     fragment length in the same range the fragment rule flags) was
    #     picked as the single longest trailing line and judged fragment-
    #     shaped on its own, even though the paragraph AFTER it made the
    #     trailing content as a whole read as ordinary prose.
    # Fixed (v3.1.9, hostile pass 4, F4) by joining ALL trailing non-blank
    # content into one string and judging THAT as a whole, with a spaces
    # floor PROPORTIONAL to its own length (len // 8, replacing the fixed
    # 100) — proportional is what "too sparse to judge" should have meant
    # from the start. This closed the finding's holes, but hostile pass 5
    # (C5-6) found the join itself reopened three of them a different way —
    # a low-space NON-PROSE block (a markdown table, a URL list, a fenced
    # code block) joined in beside a real second runaway pulled the WHOLE
    # blob's space density under its own proportional floor, exempting the
    # runaway it was joined next to — and newly flagged ordinary
    # dialogue-heavy and two-beat-paragraph replies the same way N2 needed
    # fixing for. See the full account, the measurements, and what replaced
    # it (list-majority / multi-line-join / single-longest-prose-line, in
    # that order) at the trailing-content exemption itself, a few dozen
    # lines below — this comment only carries the F4 history forward;
    # SP\\fix-p5-degen.md has the confusion tables.
    #
    # Still deferred, unresolved by pass 5 either (see SP\\fix-p5-degen.md):
    # a genuine second runaway cut at 500 or 540 characters (R3/R3b) is
    # measured statistically indistinguishable from an ordinary ~400-600
    # char "beat" paragraph (FP-c) — mean fragment length within one point,
    # same order of magnitude in space density — so closing R3/R3b with any
    # threshold on this arithmetic also flags FP-c; resolved toward keeping
    # per this lane's priority, so R3/R3b stay open holes. A runaway
    # followed by ONE ordinary paragraph right around DEGENERATE_MIN_CHARS
    # (R4, ~330 characters) is separately undecided for the same reason:
    # indistinguishable, by this arithmetic, from the two real corpus false
    # positives at 15-25x that length — "how long a real trailing paragraph
    # needs to be" is a calibration question synthetic fixtures cannot
    # answer (real data was not granted to this lane either; see the report
    # for exactly what a corpus measurement would need to show).
    lines = text.splitlines()
    last_nonblank_idx = -1
    for _i, _raw in enumerate(lines):
        if _raw.strip():
            last_nonblank_idx = _i
    # F2/F3: character offset of each line's start, so the fragment-line
    # rule below can report a span (start, end) instead of only a reason.
    # splitlines(keepends=True) segments text identically to splitlines()
    # (same line boundaries; only the trailing separator differs), so the
    # two lists stay index-aligned.
    _line_offsets: list[int] = []
    _pos = 0
    for _kept in text.splitlines(keepends=True):
        _line_offsets.append(_pos)
        _pos += len(_kept)

    # P9-5 / P10-4 (hostile pass #10): this is the FOURTH place in this
    # module that reads ``` fences, and it is NOT migrated to
    # `_fence_toggle_offsets` in this pass — deliberately, written down
    # rather than left silent. `_fence_toggle_offsets` answers "is
    # character offset i inside an open fence"; this loop needs "is line
    # N inside an open fence" while walking `lines` (already `.strip()`ped
    # at each entry, one per iteration) to decide run-length and
    # fragment-shape, an orthogonal per-LINE question the offset-based
    # reader was not built to answer directly — bridging the two would
    # mean computing `_line_offsets[line_idx]` (already available, see
    # above) and calling `_in_open_fence` at every line, correct in
    # principle but touching the hottest, most mutation-tested loop in the
    # degenerate-reply detector (test_degenerate_reply.py's C5-6 fixtures)
    # for a fence-INDENT edge case neither inline walk below has been
    # shown to hit on real data: both still use the OLD, indent-blind
    # `.startswith("```")` test (the same class of gap `_fence_toggle_
    # offsets` fixed at P9-6), but unlike `_trim_forwarded_prefix` and the
    # belt-and-braces check above (fixed this pass), a wrong read HERE
    # fails toward re-including a fragment-shaped line the exemption would
    # otherwise have excused, or vice versa — a false-positive/negative
    # RATE question on the exemption, not an unmatched-fence-reaches-the-
    # model correctness bug like the two fixed sites. Left open rather
    # than risk this loop's calibration on the last V3 release without a
    # real-data reproduction to test against (real data was refused to
    # this lane).
    run = 0
    in_fence = False
    for line_idx, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            continue  # a blank line between items does not end a list
        if line.startswith("```"):
            in_fence = not in_fence
            run = 0
            continue
        if in_fence:
            run = 0
            continue
        if len(line) <= DEGENERATE_LIST_ITEM_CHARS and _LIST_ITEM_RE.match(line):
            run += 1
        else:
            run = 0
        breaks = _fragment_line_breaks(line)
        if breaks is not None:
            ln = len(line)
            if ln / (breaks + 1) <= DEGENERATE_LINE_SENTENCE_CHARS:
                exempt = False
                if line_idx != last_nonblank_idx:
                    # v3.1.9 (hostile pass 5, C5-6) REPLACES the F4
                    # join-everything/proportional-floor check. See the
                    # block comment above for the full history; this
                    # rewrites the "not itself fragment-shaped" half again.
                    #
                    # Fenced code after the candidate line is skipped
                    # entirely (the primary per-line loop already does this
                    # for the candidate itself; the exemption never did).
                    trailing_nonblank: list[str] = []
                    _tc_in_fence = False
                    for _tc_raw in lines[line_idx + 1:]:
                        _tc_line = _tc_raw.strip()
                        if not _tc_line:
                            continue
                        if _tc_line.startswith("```"):
                            _tc_in_fence = not _tc_in_fence
                            continue
                        if _tc_in_fence:
                            continue
                        trailing_nonblank.append(_tc_line)
                    if trailing_nonblank:
                        trailing_chars = sum(len(t) for t in trailing_nonblank)
                        if trailing_chars < DEGENERATE_MIN_CHARS:
                            # SHORT trailing content (C5-6's FP-a and the
                            # finding's own N1/N4): too little text for the
                            # fragment-mean math to mean anything -- a short
                            # remark and a short collapse have the same low
                            # apparent mean fragment length purely from being
                            # short (N1's "*What do you do?*" scores the same
                            # as a real trail-off). The only feature left
                            # that distinguishes them is whether the remark
                            # reads as a COMPLETE thought: ends on a genuine
                            # sentence terminator (_is_real_sentence_end),
                            # not an abbreviation, not nothing at all. A bare
                            # emoji or "---" has no terminator and stays
                            # caught (test_degenerate_reply.py [9c] C/F); a
                            # short, properly punctuated sign-off is exempt.
                            #
                            # v3.1.9 (hostile pass 3, F5) pinned [9c] case B
                            # ("Always yours.") and case E ("Thanks for
                            # asking.") as MUST-STAY-CAUGHT on the reasoning
                            # that a short, innocuous, well-terminated
                            # remark after a real collapse is exactly the
                            # shape a model produces trailing off. Measured
                            # (SP\p5-degen\measure.py-style check, hostile
                            # pass 5): B and E are the IDENTICAL shape to N1
                            # in every feature this rule can see (short,
                            # terminated, nothing else). There is no signal
                            # here about whether the line BEFORE the
                            # candidate was a genuine collapse or an
                            # ordinary beat paragraph -- that would need
                            # real trailing-tail corpus data (see
                            # SP\fix-p4c.md F4's own TRIED AND REVERTED
                            # account of this exact conflict). Per this
                            # lane's priority (a normal reply lost from her
                            # memory is worse than a runaway kept), resolved
                            # toward KEEPING: [9c] B and E are relabelled in
                            # test_degenerate_reply.py (see
                            # SP\fix-p5-degen.md for the write-up) and this
                            # now exempts all four equally by the one
                            # feature that is actually here -- termination,
                            # not authorship.
                            exempt = _trailing_ends_in_real_sentence(
                                " ".join(trailing_nonblank)
                            )
                        else:
                            # SUBSTANTIAL trailing content (>= 300 chars).
                            #
                            # Measured (SP\p5-degen\measure.py): a genuine
                            # second runaway cut at 500/540/600 chars and an
                            # ordinary ~400-800 char "beat" paragraph
                            # (test_p4c_degeneracy.py's beats_para) score
                            # WITHIN ONE POINT of each other on mean fragment
                            # length (24.8-29.7, all under the 40-char
                            # limit) and are the same order of magnitude in
                            # space density -- there is no arithmetic
                            # threshold on a SINGLE blob of trailing prose
                            # that catches one and keeps the other; every
                            # value tried also flags C5-6's FP-c (two beat
                            # paragraphs, nothing else). So a single
                            # substantial blob is judged the way pass-3
                            # judged it before F4: by the single longest
                            # trailing PROSE line, against the FIXED
                            # _LINE_MIN_SPACES floor (100) -- not the F4
                            # proportional one. This keeps R3c caught (600
                            # chars clears 100 spaces) and leaves R3/R3b
                            # open, exactly like FP-c (500/540 chars, 90-96
                            # spaces, never clears 100) -- documented, not
                            # silently dropped, in SP\fix-p5-degen.md.
                            #
                            # "Longest PROSE line": lines that are
                            # majority-list-shaped (C5-6's R1: 32 terminated
                            # bullets -- the list phase returning, not prose
                            # at all) are judged separately, by COUNT, not
                            # by the mean-length math the list-run backstop
                            # already owns (DEGENERATE_LIST_RUN=50; 32 is
                            # short of that independent threshold but is
                            # still "more of the collapse" for THIS
                            # exemption's purposes). And a line with too few
                            # spaces to be prose at all (a markdown table
                            # row, a URL, a dotted identifier -- none of
                            # which has 1 space per 8 characters) is
                            # excluded before anything is joined or
                            # measured: C5-6's HOLE-a/b/c is exactly a real
                            # 600-char second runaway diluted below its own
                            # proportional floor by 30 table rows or 12 URLs
                            # joined in beside it. Filtering them out first
                            # (rather than joining and hoping the floor
                            # scales) leaves the real runaway line to be
                            # judged on its own, the way it always was.
                            #
                            # Content spread over SEVERAL long prose lines
                            # (C5-6's R2: 6 lines of genuine fragment-shaped
                            # collapse, none alone clearing the old
                            # single-line check) is still joined and judged
                            # as a whole with the F4 proportional floor --
                            # that half of F4 was correct and is kept. The
                            # line-count floor (>=2) is what keeps this from
                            # reopening FP-a/b (dialogue: no individual line
                            # reaches DEGENERATE_LINE_SENTENCE_CHARS, so
                            # there is nothing to join) or FP-c (ONE long
                            # beat paragraph, correctly routed to the
                            # single-blob path above instead).
                            list_lines = [
                                t for t in trailing_nonblank
                                if _LIST_ITEM_RE.match(t)
                            ]
                            if (
                                len(list_lines) >= _TRAILING_LIST_MAJORITY_MIN
                                and len(list_lines) >= len(trailing_nonblank) / 2
                            ):
                                exempt = False
                            else:
                                prose_candidates = [
                                    t for t in trailing_nonblank
                                    if t not in list_lines
                                    and _trailing_line_is_prose_dense(t)
                                ]
                                long_lines = [
                                    t for t in prose_candidates
                                    if len(t) >= DEGENERATE_LINE_SENTENCE_CHARS
                                ]
                                if len(long_lines) >= 2:
                                    joined_trailing = " ".join(long_lines)
                                    exempt = not _line_is_fragment_shaped(
                                        joined_trailing,
                                        min_spaces=max(1, len(joined_trailing) // 8),
                                    )
                                elif prose_candidates:
                                    exempt = not _line_is_fragment_shaped(
                                        max(prose_candidates, key=len)
                                    )
                                else:
                                    # nothing prose-shaped followed at all
                                    # (pure table/URL/code) -- "too sparse
                                    # to judge" reads as not-fragment
                                    # everywhere else in this file, so it
                                    # does here too: exempt.
                                    exempt = True
                if not exempt:
                    _fl_start = _line_offsets[line_idx]
                    return (
                        f"an unbroken line of {ln} characters made of "
                        f"{breaks + 1} fragments averaging "
                        f"{ln / (breaks + 1):.0f} characters (limit "
                        f"{DEGENERATE_LINE_SENTENCE_CHARS} over "
                        f"{DEGENERATE_LINE_CHARS}+ characters)"
                    ), _fl_start, _fl_start + len(raw)
    # R19: gated on DEGENERATE_MIN_CHARS like the decoration-fraction rule
    # above — this file's own doctrine (see MIN_MEMORABLE_TRIMMED_CHARS)
    # calls that the floor below which nothing is judged structurally, and
    # this branch was the one exception.
    if n >= DEGENERATE_MIN_CHARS and run >= DEGENERATE_LIST_RUN:
        # No localized span reported (deferred): the run is a COUNT of short
        # list lines, not necessarily contiguous text free of other content
        # in between (blank lines interleave without resetting it), so
        # "first line of the run" is not as clean a boundary as the other
        # rules' regex matches. Callers needing a span fall back to the old
        # whole-reply clean-head rule for this verdict too.
        return (
            f"a run of {run} consecutive list items of "
            f"{DEGENERATE_LIST_ITEM_CHARS} characters or fewer (limit "
            f"{DEGENERATE_LIST_RUN})"
        ), None, None
    return None, None, None


def reply_is_degenerate(text: str) -> str | None:
    """Why this reply looks like a repetition loop, or None if it looks fine.

    Thin wrapper around `_reply_degenerate_verdict` (defined just above)
    that keeps the original signature every existing caller relies on. See
    that function for the detection rules themselves and for why the span
    it also computes lives there instead of in a second copy of this logic.
    """
    return _reply_degenerate_verdict(text)[0]


# v3.1.3: the skip does not do what its docstring promises without this.
#
# Both /v1/chat/completions call sites gate _async_tail on `not
# reply_is_degenerate(...)`, and that is provably enough for _async_tail's
# jobs 1-2 (episodic indexing, fact extraction): each sees only THIS
# exchange's own last_user_text/assistant_text, so skipping the call for a
# degenerate turn keeps it out of both for the life of the conversation —
# nothing ever calls back for that turn again.
#
# Job 3 (hierarchical summary rollup) does not work that way.
# `summarizer.maybe_rollup` slices its input out of the message history by
# TURN POSITION on every later call, and that history is the CLIENT's, not
# ours: the user already read the degenerate reply, so it comes back as part
# of `original_messages` on the very next turn. The skip on turn N does
# nothing to turn N's text once it is sitting in the array _async_tail is
# handed on turn N+1 — maybe_rollup has never heard of "degenerate" and folds
# it into an L1 chunk exactly like any other turn once it falls inside that
# chunk's range, and that chunk is what gets injected back as memory. That is
# the exact outcome the detector exists to prevent, landing one turn later
# than the skip is looking. Verified in test_degenerate_skip.py: run against
# the code before this function existed, the raw repeated text reached
# summarizer.maybe_rollup's input unchanged.
#
# So _async_tail (below) redacts here, immediately before building the
# message list it hands to maybe_rollup — the one place in job 3 that sees
# the full history and runs on every turn, degenerate or not.
_DEGENERATE_HISTORY_PLACEHOLDER = (
    "[a reply here looked like a repetition loop and was left out of "
    "everything memorized about this conversation]"
)


# Shared by _redact_degenerate_turns (the rollup-input redaction) and
# _redact_forwarded_loop_replies (v3.1.9.2, the forwarded-window redaction,
# below chat_completions): BOTH need the same "keep the clean sentence head,
# else fall back to the placeholder" decision on a turn reply_is_degenerate
# has flagged, and a rule this load-bearing must not exist twice — two copies
# is how the two sites would eventually disagree about what "clean" means.
_DEGENERATE_SPAN_MARKER = "[a repeated section was left out here]"


def _degenerate_replacement_content(
    text: str, placeholder: str, *, keep_middle: bool = False
) -> tuple[str, bool]:
    """-> (replacement content, whether any clean text was kept).

    `text` must already be known-degenerate (caller has checked
    `reply_is_degenerate`).

    `keep_middle=False` (the default; used by `_redact_degenerate_turns`,
    the ROLLUP-input redaction, which this lane leaves alone — see the
    module note above `_redact_degenerate_turns` for why): applies the
    original cut rule `decide_memory_tail` applies to a cut reply — keep the
    longest prefix of the WHOLE text ending on a sentence boundary if that
    prefix clears MIN_MEMORABLE_TRIMMED_CHARS, else use `placeholder` whole.

    `keep_middle=True` (used by `_redact_forwarded_loop_replies`, v3.1.9.2):
    v3.1.9.2 (p7 hostile pass #7, F2/F3). The `keep_middle=False` rule
    trims to the last sentence boundary of the WHOLE reply and re-judges
    that prefix with `reply_is_degenerate` — which does not help when the
    degenerate span itself ends on sentence boundaries (a phrase loop:
    "Absolutely. With Desperation. With Humility. ..." is fine prose by
    that measure) or sits in the MIDDLE of an otherwise clean reply (a
    120+-character scream or a zeros array in a code fence). Real corpus
    measurement (SP\\p7\\real_head.py, hostile pass #7): 33 of 68 flagged
    replies had a clean head of 416-21,569 characters the old rule threw
    away whole. Here, instead, the SAME position `_reply_degenerate_verdict`
    already located for the rule that fired is used to cut around just the
    flagged span:
      - span reaches the (stripped) end of the reply (the tail-loop rule is
        always this shape; the other three can be): keep
        `trim_to_last_sentence(text[:start])` if it clears
        MIN_MEMORABLE_TRIMMED_CHARS, else `placeholder` whole — same floor
        as before, but the sentence search is now confined to the text
        BEFORE the loop, so it can no longer land on a sentence boundary
        INSIDE the loop the way searching the whole text did.
      - span sits in the middle: keep `trim_to_last_sentence(text[:start])`
        before it AND `text[end:]` after it, with only the flagged span
        itself collapsed to `_DEGENERATE_SPAN_MARKER` — the reader (the
        model, on the next request) sees everything except the loop/scream/
        array itself, not a placeholder standing in for the whole answer.
      - no span (decoration fraction, script drift, the list-run backstop —
        see `_reply_degenerate_verdict`'s comment on which rules have one):
        falls back to the `keep_middle=False` rule above; there is nothing
        to cut AROUND.
    Only a single span is handled (the one `_reply_degenerate_verdict`'s
    "longest match wins" logic already picked as worst); a reply with two
    independently-flagged spans is not split apart further, same limit the
    detector itself already has (see its own docstring on LONGEST match).
    """
    if not keep_middle:
        head = decide_memory_tail(text, finished=False, truncated=True, holed=False)
        if head.store and head.text.strip():
            return head.text, True
        return placeholder, False

    # REPEATED UNTIL CLEAN (coordinator review, real data). One cut is not
    # always enough: the span the detector reports can start after the loop
    # really began (a phrase loop measured from its tail window; a fragment
    # line inside a longer degenerate stretch), so the kept head can itself
    # still be flagged. On the 2026-09-16 backup, 10 of 67 flagged replies
    # were still flagged after one cut. The tail-loop rule looks at most
    # _TAIL_LOOP_WINDOW characters, so each pass removes at most that much of
    # a phrase loop: a 22k-character loop needs ~6 passes. Re-judge and cut
    # again while the text keeps shrinking, up to a budget-derived pass count
    # (her longest reply, 51k characters, needs ~13, well inside it); whatever
    # is still flagged after that goes out as the placeholder, never as loop
    # text, and it is LOGGED (P8-5/P8-8, hostile pass #8 — the old version
    # fell back silently, so an operator could not tell "a real loop this
    # large happened" from any other placeholder cause).
    #
    # P8-5: a fixed PASS COUNT bounds passes, not CPU — each pass costs
    # roughly len(content) of regex scanning, so _DEGENERATE_CUT_MAX_PASSES
    # (64) over a 300k-character pathological loop measured 2.2s of
    # GIL-bound CPU on first sight; no real reply has come anywhere near
    # that (her longest is 51k). Bound total CHARACTERS scanned across all
    # passes instead of a flat pass count: for anything up to several times
    # her real maximum this is still the full 64 passes (unchanged
    # behaviour); a pathological input far beyond that gets fewer, cheaper
    # passes before giving up, instead of grinding through 64 of them.
    # Memoized per (digest, placeholder): OpenWebUI resends the same flagged
    # turn on every later request, and this is several detector passes.
    key = (
        hashlib.blake2b(text.encode("utf-8", "surrogatepass"), digest_size=16).digest(),
        placeholder,
    )
    # P8-8 (hostile pass #8): COMPACTOR_DEGENERATE_VERDICT_CACHE_SIZE=0 used
    # to disable only _reply_degenerate_verdict's cache — this cache, whose
    # VALUES are the kept replacement TEXT (not just a verdict tuple), kept
    # caching regardless, at a hard-coded 1,024 entries. An operator who set
    # that env var to 0 specifically to stop caching reply content in this
    # process (the CHANGELOG's own "the cache holds no reply text" claim was
    # false for exactly this cache, see P8-8) got no effect here at all.
    # Same env var, both caches.
    if _DEGENERATE_VERDICT_CACHE_SIZE > 0:
        with _DEGENERATE_VERDICT_LOCK:
            hit = _DEGENERATE_CUT_CACHE.get(key)
        if hit is not None:
            return hit
    _max_passes = max(
        1, min(_DEGENERATE_CUT_MAX_PASSES, _DEGENERATE_CUT_BUDGET_CHARS // max(1, len(text)))
    )
    content, kept = _cut_degenerate_span_once(text, placeholder)
    for _ in range(_max_passes):
        if not kept or not reply_is_degenerate(content):
            break
        nxt, nkept = _cut_degenerate_span_once(content, placeholder)
        if nkept and len(nxt) >= len(content):
            logger.warning(
                f"degenerate-cut: a cut pass on a {len(text)}-char reply "
                f"stopped shrinking at {len(content)} chars while still "
                f"flagged degenerate; falling back to the whole-reply "
                f"placeholder"
            )
            content, kept = placeholder, False
            break
        content, kept = nxt, nkept
    else:
        if kept and reply_is_degenerate(content):
            logger.warning(
                f"degenerate-cut: a {len(text)}-char reply was still "
                f"flagged after {_max_passes} cut pass(es); falling back "
                f"to the whole-reply placeholder instead of forwarding or "
                f"storing loop text"
            )
            content, kept = placeholder, False
    result = (content, kept)
    if _DEGENERATE_VERDICT_CACHE_SIZE > 0:
        with _DEGENERATE_VERDICT_LOCK:
            _DEGENERATE_CUT_CACHE[key] = result
            while len(_DEGENERATE_CUT_CACHE) > 1024:
                _DEGENERATE_CUT_CACHE.popitem(last=False)
    return result


_DEGENERATE_CUT_MAX_PASSES = 64
# P8-5: total characters a reply may have scanned across every cut pass
# combined (see the comment at this constant's one use, above). 60,000 is
# comfortably above her longest real reply (51k), so nothing observed in
# production loses even one pass to this; it only shortens the pathological
# tail (a loop far beyond anything her real replies reach).
_DEGENERATE_CUT_BUDGET_CHARS = _DEGENERATE_CUT_MAX_PASSES * 60_000
_DEGENERATE_CUT_CACHE: "collections.OrderedDict[tuple, tuple[str, bool]]" = collections.OrderedDict()


def _trim_forwarded_prefix(text: str) -> str:
    """The longest prefix of `text` to keep before a degenerate span, for
    the FORWARDED window only (`_cut_degenerate_span_once`'s `pre`).

    P8-3 (hostile pass #8): `trim_to_last_sentence` refuses any boundary
    inside a ``` fence — the right rule for the MEMORY-side redaction
    (`_redact_degenerate_turns`, keep_middle=False): an unterminated opener
    must never reach facts.py's line filter, and a `.` inside code is not a
    sentence end anyway. It is the wrong rule here: this text is shown to
    the model as ordinary conversation HISTORY, nothing is extracted from
    it, so a boundary INSIDE a fence costs nothing. Real-data measurement
    (P8-3): this model writes long replies as prose broken up by
    decorative ``` boxes, and the fence exclusion can throw away
    everything back to the last sentence end OUTSIDE a box — the WHOLE
    reply in one real case (the last sentence sat inside a box 4 characters
    before the loop), about 11,080 characters of boxes-and-prose in
    another.

    Longest prefix ending at a real sentence boundary (fence or no fence)
    OR a line break, whichever reaches further: a code box rarely ends in
    terminal punctuation, so the line-break fallback is what actually saves
    most of a box's own content when the true sentence boundary sits well
    before it or inside it.

    Self-balancing (P8-4): if the kept prefix has an ODD number of ```
    lines (opened, never closed within it — the cut can now legally land
    there, unlike trim_to_last_sentence), a closing ``` line is appended so
    this prefix ALONE stays a well-formed fence pair. Otherwise everything
    the model reads after it — the marker, and any `post` text
    `_cut_degenerate_span_once` appends after that — would open as code.
    """
    if not text:
        return ""
    end = 0
    for m in _SENTENCE_END_RE.finditer(text):
        i = m.start()
        if not _is_real_sentence_end(text, i):
            continue
        end = m.end()
    cut_end = max(end, text.rfind("\n") + 1)
    if cut_end <= 0:
        return ""
    cut = text[:cut_end]
    # P10-4 (hostile pass #10): this used to count fence lines with
    # `line.strip().startswith("```")`, which — unlike `_fence_toggle_
    # offsets` since P9-6 — does not know a 4-space-indented ``` line is
    # literal CommonMark code content, not a real delimiter. For a `cut`
    # that ends with such a line, the two readers disagreed by one: this
    # one saw an unbalanced fence and appended a REAL closing ``` line,
    # which then opened an unmatched fence of its own (the only genuine
    # delimiter in what this function emits), reading the rest of the
    # forwarded reply as code. `len(_fence_toggle_offsets(cut)) % 2` is the
    # same parity question asked with the same indent-aware reader every
    # other fence decision in this module now uses.
    if len(_fence_toggle_offsets(cut)) % 2 == 1:
        cut = cut.rstrip("\n") + "\n```"
    return cut


def _cut_degenerate_span_once(text: str, placeholder: str) -> tuple[str, bool]:
    """One cut around the span `_reply_degenerate_verdict` reports (see
    `_degenerate_replacement_content(keep_middle=True)` for the rules)."""
    _reason, start, end = _reply_degenerate_verdict(text)
    if _reason is None:
        return text, True
    if start is None or end is None:
        head = decide_memory_tail(text, finished=False, truncated=True, holed=False)
        if head.store and head.text.strip():
            return head.text, True
        return placeholder, False

    # P8-3: use the forwarded-only prefix rule (fence boundaries allowed,
    # self-balanced) instead of the memory-side trim_to_last_sentence — see
    # _trim_forwarded_prefix's docstring.
    pre = _trim_forwarded_prefix(text[:start]).strip()
    post = text[end:].strip()
    if end >= len(text.rstrip()):
        # Runs to the end: nothing after it worth keeping.
        if len(pre) >= MIN_MEMORABLE_TRIMMED_CHARS:
            return pre, True
        return placeholder, False
    if pre or post:
        # P8-4 (hostile pass #8): a MID-reply span inside a CLOSED fence
        # splits one box across `pre` (keeps the opener, now self-balanced
        # by _trim_forwarded_prefix above) and `post` (keeps the box's own
        # closer). Joined as `pre + marker + post`, the marker sits right
        # after pre's own synthetic close, so post's leading text — which
        # was INSIDE the box in the original — would read as plain prose
        # until its own closer, then everything AFTER that stray closer
        # opens as code with nothing left to end it. If the span started
        # inside a fence in the ORIGINAL text, post is picking back up
        # inside that same fence; prefix it with a fresh opener so its own
        # leading text (up to its own real closer) renders exactly as it
        # did originally.
        if post and _in_open_fence(_fence_toggle_offsets(text), start):
            post = "```\n" + post
        combined = "\n\n".join(p for p in (pre, _DEGENERATE_SPAN_MARKER, post) if p)
        # Belt-and-braces: regardless of what the two pieces above did
        # individually, the text actually forwarded must never itself
        # carry an odd ``` count — an unbalanced fence here is exactly what
        # leaves the REST of the conversation misread as code from this
        # point on (P8-4's failure mode).
        #
        # P10-4 (hostile pass #10): this belt-and-braces check used to
        # count with `ln.strip().startswith("```")` — the SAME wrong
        # counter that created the P10-4 defect at `_trim_forwarded_
        # prefix` above, which is exactly why it was not a backstop for
        # it: both readers agreed with each other (both indent-blind) and
        # disagreed with `_fence_toggle_offsets` (indent-aware since
        # P9-6), so an indented ``` line fooled them identically instead
        # of one catching the other's mistake. `_fence_toggle_offsets` is
        # the one reader every other fence decision in this module trusts.
        if len(_fence_toggle_offsets(combined)) % 2 == 1:
            combined = combined.rstrip() + "\n```"
        return combined, True
    return placeholder, False


def _redact_degenerate_turns(messages: list[dict]) -> list[dict]:
    """Copy of `messages` with any assistant turn that is itself a
    repetition loop replaced by a neutral placeholder, so it cannot be
    folded into a hierarchical summary chunk. See the comment above this
    function for why this is necessary in addition to (not instead of) the
    `not reply_is_degenerate(...)` gate at the call sites.

    Only assistant turns are checked. The detector is calibrated against a
    corpus of real ASSISTANT replies (see reply_is_degenerate's docstring and
    test_degenerate_reply.py); running it on user text would be an
    uncalibrated claim wearing the same thresholds, not a defensible one, and
    a user turn is not the thing this detector was ever measuring.

    The placeholder is deliberately non-blank: `summarizer._do_l1_rollup`
    skips a chunk whose every piece is blank (`if not any(p.strip() for p in
    pieces)`), and a redacted turn disappearing from a chunk that still has
    real neighbors is not the same failure as one that empties the whole
    chunk — but neither should look, to a chunk-existence check, like there
    was nothing there. Saying plainly that something was omitted is the
    difference between a gap and a silent one.

    A CLEAN HEAD IS KEPT (hostile pass #4, reviewer A F7). A reply whose
    whole text the detector calls a loop, but whose prefix up to the last
    sentence boundary is clean, is replaced by that prefix — the exact rule
    decide_memory_tail applies to a cut reply (trim to the last sentence,
    the MIN_MEMORABLE_TRIMMED_CHARS floor, judged on the kept text), not the
    placeholder. Before this, a ceiling-cut reply with a clean prose head and
    a runaway tail was stored trimmed by memory, and then its chunk read the
    placeholder while the covered-turn record held the full text the client
    re-sends: from that chunk on the reply was removed WHOLE from every
    request, head included, although memory had deliberately kept the head.
    The head is not a loop by memory's own rule. What a redacted turn now
    loses is only what follows its last sentence boundary, which is where
    the loop is. A turn with no clean head still gets the placeholder.

    The history carries no "finished" flag, so the rule is the cut rule for
    every turn: a FINISHED reply memory refused whole also keeps its clean
    head here. That is the same judgement (the object is the text being
    summarized), and the hierarchy is a summary, not the fact store.
    """
    out = []
    redacted = 0
    kept_heads = 0
    for m in messages:
        if (
            isinstance(m, dict)
            and m.get("role") == "assistant"
            and reply_is_degenerate(_message_text(m))
        ):
            content, kept_head = _degenerate_replacement_content(
                _message_text(m), _DEGENERATE_HISTORY_PLACEHOLDER
            )
            m = {**m, "content": content}
            if kept_head:
                kept_heads += 1
            redacted += 1
        out.append(m)
    if redacted:
        # Logged HERE, not only at original detection time: this
        # substitution is what actually excludes the turn from summaries,
        # and it recurs on every rollup long after the detection-time line
        # has scrolled away. A permanent exclusion nothing ever mentions
        # again is exactly the silent failure shape this branch exists to
        # kill.
        logger.info(
            f"redacted {redacted} degenerate historical turn(s) from "
            f"rollup input ({len(messages)} total); {kept_heads} of them "
            f"kept their clean sentence head"
        )
    return out


# ---------------------------------------------------------------------------
# v3.1.4: what a CUT reply contributes to memory.
#
# Measured in one log window on 2026-09-01: 51 replies skipped because she
# hit Stop, 12 because vLLM hit the generation ceiling, 0 for repetition. 63
# exchanges never reached memory — no facts, no episodic index, no rollup —
# and that is more than half of her recent conversation. The old gate
# reasoned that "memorizing a half-sentence plants false memories", which is
# true of a 200-character fragment and wrong for a 27,000-character reply
# cut at the end, which is 99% complete prose. The gate tested COMPLETION
# when it should test SUBSTANCE. So: keep the longest prefix that ends on a
# sentence boundary, and judge that.
# ---------------------------------------------------------------------------

# A sentence boundary: a terminator, optional closing marks (quotes, brackets,
# markdown emphasis), then whitespace or end-of-text. The whitespace clause is
# what keeps `3.14`, `v3.1.6` and `1.500` from being boundaries — the `.` in
# each is followed by a digit — so there is NO decimal special-case and none
# is needed. The fullwidth terminators do not need the clause: CJK prose puts
# no space after `。` and has no decimal written with it.
_SENTENCE_END_RE = re.compile(
    r"""[.!?]["'”’)\]»*_~`]*(?=\s|\Z)"""
    r"""|[。！？]["”’」』)）]*"""
)
# Not exhaustive, and it does not need to be: an abbreviation this list fails
# to reject just moves the cut to a different real terminator, which costs
# one sentence; an abbreviation it wrongly ACCEPTS stores a fragment ("...as
# it says in Rev.") as something the model said. So the list leans towards
# rejecting. The scripture books are here because this user quotes scripture
# (see reply_is_degenerate's script-drift note) and "Gen. 1:1" is how it is
# written. Matched on the dotted run immediately before the terminator, so
# "e.g" and "u.s" are entries, not "e" and "s".
_SENTENCE_ABBREVIATIONS = frozenset(
    """
    mr mrs ms dr prof sr jr st vs etc e.g i.e cf viz approx vol fig dept
    inc ltd u.s u.k a.m p.m ph.d mt ft gen rev hon capt col lt sgt
    ex lev num deut josh judg sam kgs chr neh ps prov eccl isa jer lam ezek
    dan hos mic hab zeph hag zech mal matt mk lk jn rom cor gal eph phil
    thess tim tit philem heb jas pet
    """.split()
)
# The longest entry above is 6 characters ("approx", "philem"); the dotted
# run is capped well above that so a long dotted identifier (`os.path.join`)
# is scanned, found absent, and accepted without an unbounded walk back.
_ABBREV_SCAN_CHARS = 12
_ABBREV_RUN_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.")
# What may precede a word for it to count as a WHOLE word (for the
# abbreviation and single-initial rules below): the preceding character is
# not alphanumeric. "1st." is not the abbreviation "st." because "1"
# precedes it and IS alphanumeric.
#
# R25: this used to be an explicit frozenset (space, tab, opening brackets,
# quotes, markdown emphasis marks) with no dash of any kind in it, so
# "Then—i.e. a fragment" and "author—J. R. R. Tolkien" skipped the
# abbreviation stoplist and the single-initial rule entirely — whole_word
# came back False after an em dash, en dash or hyphen, backwards, since a
# dash introduces a word exactly as a space does. The existing tests only
# ever led with a space or start-of-text, so the set was exercised for
# being too permissive and never for being too narrow. "not alphanumeric"
# is what the rule has always meant; it now covers dashes and anything
# else a hand-enumerated set could omit without another list to keep in
# sync with this one.


def _is_real_sentence_end(text: str, i: int) -> bool:
    """True when `text[i]` — a terminator `_SENTENCE_END_RE` matched at
    position `i` — is a genuine sentence end, not an ellipsis, a dotted
    abbreviation ("Dr.", "e.g."), a single initial ("J."), or a numbered-
    list marker ("1.").

    Shared by trim_to_last_sentence (deciding where to cut text going into
    the store) and reply_is_degenerate's fragment-line rule (deciding
    whether a line's dots are real sentence breaks or abbreviations) — one
    rule, one function, the same "single shared predicate" doctrine
    assistant_content_is_empty is built on. R24: before this function
    existed, reply_is_degenerate counted sentence breaks with a bare
    `line.count(". ")`, so "Dr. ", "Mrs. ", "Rev. ", "9 a.m. " were each
    counted as a sentence end, collapsing the computed mean fragment length
    on ordinary prose that happened to use an abbreviation — while this
    exact list of abbreviations already existed one function away, written
    for trim_to_last_sentence. Two pieces of code in one delta disagreed
    about what a sentence end is; now there is one definition.

    Only `.` has an abbreviation problem — `!`, `?` and the fullwidth
    terminators are always real ends.
    """
    if text[i] != ".":
        return True
    if i > 0 and text[i - 1] == ".":
        return False  # ellipsis
    # The dotted word immediately before the terminator, and what precedes
    # it, for the abbreviation, initial and list-marker rules.
    j = i
    while j > 0 and i - j < _ABBREV_SCAN_CHARS and text[j - 1] in _ABBREV_RUN_CHARS:
        j -= 1
    word = text[j:i]
    whole_word = j == 0 or not text[j - 1].isalnum()
    if whole_word and word:
        if word.lower() in _SENTENCE_ABBREVIATIONS:
            return False
        if len(word) == 1 and word.isupper():
            return False  # single initial
    if not word:
        # A bare number at the start of its line is a list marker ("1. ").
        k = i
        while k > 0 and text[k - 1].isdigit():
            k -= 1
        if k < i:
            ls = k
            while ls > 0 and text[ls - 1] in " \t":
                ls -= 1
            if ls == 0 or text[ls - 1] == "\n":
                return False
    return True


def _count_real_period_breaks(line: str) -> int:
    """Count of `. ` in `line` whose `.` is a genuine sentence end, per
    _is_real_sentence_end — shared with trim_to_last_sentence so the two
    agree on what a sentence end is (R24), instead of the bare
    `line.count(". ")` that used to count "Dr. ", "Mrs. ", "9 a.m. " as
    sentence breaks and collapsed the mean fragment length of ordinary
    prose in reply_is_degenerate's fragment-line rule.

    Deliberately `. ` (a literal period-then-space), not _SENTENCE_END_RE's
    `(?=\\s|\\Z)` — the fragment-line rule computes
    `fragments = breaks + 1`, where the "+1" already accounts for the
    line's own final fragment, whose period is never followed by a space
    (it is the last thing on the line). Matching a period at end-of-line
    too would count that last fragment's break twice and move the
    calibrated boundary test_degenerate_reply.py pins (1,600 chars / 40
    fragments fires, 1,640 / 40 does not — both exactly on ". "-separated
    fragments where every period but the line's last is followed by a
    space).
    """
    return sum(
        1 for m in re.finditer(r"\. ", line) if _is_real_sentence_end(line, m.start())
    )


def trim_to_last_sentence(text: str) -> str:
    """The longest prefix of `text` that ends on a sentence boundary, or ""
    when there is none. Pure; no logging.

    NO MARKER IS APPENDED, unlike every other trimmer in this codebase
    (facts._truncate_to_tokens, summarizer's chunk trim, the payload trim in
    _enforce_hard_budget, _DEGENERATE_HISTORY_PLACEHOLDER). Those trim text
    shown TO THE MODEL AS INPUT, where an unmarked cut reads as all there
    was, so the marker is what keeps a truncation from becoming a wrong
    fact. This text goes into the STORE: it is fact-extracted, embedded as
    episodic content and folded into an L1 chunk, so a marker here would be
    extracted as a fact, embedded and summarized — the marker would BECOME a
    memory. Do not add one for consistency with the others; the
    inconsistency is the point. The result is a plain prefix
    (`text.startswith(result)` always holds), and test_sentence_trim.py
    pins that.

    A boundary is one of `.!?` plus optional closing marks, followed by
    whitespace or end-of-text (see _SENTENCE_END_RE for why that clause
    makes decimals and version numbers a non-issue), or one of `。！？`.
    Rejected even when the regex matches:

      - a `.` that is part of `..`/`...` — an ellipsis is a pause, not an
        end, and `…` (U+2026) is not a terminator at all;
      - a `.` after a word in _SENTENCE_ABBREVIATIONS ("Dr.", "e.g.");
      - a `.` after a single capital letter ("J. R. R."; also "I." — a
        sentence that ends "...so am I." loses that one sentence, which is
        the cheaper error: an accepted initial stores "by J." as a memory);
      - a `.` after a bare number at the start of a line — a numbered list
        marker ("1. ") is not a sentence, and "Steps:\\n1." is a fragment;
      - anything inside an open ``` fence: a cut inside a fence leaves the
        unterminated opener that facts.py's line filter already refuses at
        the fact level, and a `.` in code is not a sentence end anyway. A
        fence that closes again is fine; the cut can land after it.

    NEWLINES ARE DELIBERATELY NOT BOUNDARIES. Measured over 349 of her
    replies on 2026-09-01, sentence-only discards a median of 20 characters
    but up to 25,063, where sentence-or-line would cap the worst case at
    3,917. That looked decisive and it was the wrong read: those 25,063
    characters ARE the runaway bullet list — `- Always`, `- Forever`,
    unterminated — and discarding them is the point. A reply that collapsed
    into twenty-four distinct bullets trims back to the prose above the
    list, and only the prose is remembered. There is also no way to tell
    `- eggs` from `- eg` cut mid-word: punctuation is the only positive
    evidence a unit finished. Terminated bullets ("- One thing.") are real
    sentences and are kept.

    Cost: this runs SYNCHRONOUSLY on the event loop at both memory-tail
    call sites (see _TOKEN_RUN_RE for what an unbounded pattern cost there:
    6,214 ms on 16k of input). One forward scan with a regex that has no
    nested quantifier, a bisect per candidate for the fence check, and a
    backward look of at most _ABBREV_SCAN_CHARS. Measured 2026-09-01 on
    Python 3.14 (test_sentence_trim.py [13] prints it every run): 30,000
    characters of prose in 1.3 ms; the pathological 30,000 characters of
    ". . . ." (a candidate every other character) in 10 ms; 30,000
    characters with 3,000 fence toggles in 4.6 ms.
    """
    if not text:
        return ""
    # Fence toggles as text offsets, so each candidate costs one bisect
    # rather than a re-scan of everything before it. Shares
    # `_fence_toggle_offsets`/`_in_open_fence` with `_trim_forwarded_prefix`
    # so the two agree on what a fence is — this function wants ANY open
    # fence, closed or not (a cut must never land inside an unterminated
    # opener). `reply_is_degenerate`'s token-run rule no longer has a fence
    # reading of its own at all (P9-3, hostile pass #9 — the exemption it
    # used to share this offset list with was removed, not narrowed).
    toggles = _fence_toggle_offsets(text)
    end = 0
    n = len(text)
    for m in _SENTENCE_END_RE.finditer(text):
        i = m.start()
        if _in_open_fence(toggles, i):
            continue  # inside an open fence
        if not _is_real_sentence_end(text, i):
            continue
        end = m.end()
    if end <= 0 or end > n:
        return ""
    return text[:end]


# The floor under a TRIMMED reply. Not a round number: it equals
# DEGENERATE_MIN_CHARS, the floor below which this codebase already declines
# to judge a reply structurally, and it matches the 60-word floor in
# scripts/measure-reply-health.py ("too short to say anything about") at that
# corpus's measured word length. Two floors where the corpus supports one
# would be worse. Her p1 reply is 377 characters (2026-09-01, 349 replies),
# so this excludes under 1% of real replies. It applies to the trim path
# only: a reply the model FINISHED is stored whole whatever its length, as
# it always was.
#
# No relative floor ("keep only if >= X% survived"). It would fire hardest on
# exactly the runaway replies that make up most of the 63 lost exchanges —
# trimming 27,000 characters down to a 900-character prose head is a GOOD
# outcome. tailhealth's trimmed_raw/trimmed_kept totals make the retention
# ratio a measured number instead.
MIN_MEMORABLE_TRIMMED_CHARS = _env_int("COMPACTOR_MIN_MEMORABLE_TRIMMED_CHARS", 300)


@dataclasses.dataclass(frozen=True)
class TailDecision:
    """What decide_memory_tail concluded about one reply.

    `store`   — hand `text` to the memory tail, or not.
    `text`    — exactly what to store: the reply verbatim, or its trimmed
                prefix. "" when not storing.
    `outcome` — the machine label (one of tailhealth.OUTCOMES) for the
                counter. Carries no conversation text; it goes to
                /health/full, which is not localhost-gated.
    `reason`  — the human note for the LOG only: why it was skipped, or for
                a trimmed store what was cut. None for a verbatim store. May
                quote up to 24 characters of the reply (reply_is_degenerate
                does), so it stays in the log.
    """

    store: bool
    text: str
    outcome: str
    reason: str | None
    raw_chars: int


def decide_memory_tail(
    text: str, *, finished: bool, truncated: bool, holed: bool
) -> TailDecision:
    """ONE policy for what a finished-or-cut reply contributes to memory,
    for BOTH /v1/chat/completions call sites. Pure; no logging, no
    counting — _run_memory_tail does those, once, for both.

    Order, and why:

      holed      -> skip. Text we know we could not read completely is
                    unsafe, never safe (SseAccumulator.holed). Before the
                    finished check on purpose: a hole in a cleanly finished
                    stream was memorized silently until v3.1.4.
      empty      -> skip. Nothing to remember; the old non-streaming site
                    fired the tail anyway and, with extraction disabled,
                    rewrote the facts file for a turn the model never
                    answered (v3.1 F20).
      finished and not truncated
                 -> store VERBATIM, untrimmed, unless reply_is_degenerate
                    says it is a repetition loop. This is today's behaviour
                    for a reply the model finished, byte for byte.
      otherwise  -> the reply was CUT — she hit Stop, or vLLM hit the
                    generation ceiling; one rule, one path for both. Trim to
                    the last complete sentence (trim_to_last_sentence), then
                    skip if nothing survives, skip if less than
                    MIN_MEMORABLE_TRIMMED_CHARS survives, skip if what
                    survives is degenerate; else store the trimmed prefix.

    Degeneracy is judged on the TRIMMED text, not the raw: the object of
    judgement is the thing being stored. A clean prose head followed by a
    box-drawing tail is kept once the tail is discarded — judged raw, it
    would be thrown away for the part that is not being kept. The trim is
    what makes this safe: an unterminated runaway list has no sentence
    boundary and is discarded before it is judged, and a runaway that ran
    to its own end is still caught by the structural-collapse rule on what
    remains.

    Why `truncated` is not folded into `finished` by the caller: the old
    gate was `usable()` = finished and not truncated, and the non-streaming
    site had no `finished` notion at all, so the two sites answered
    different questions. Passing all three keeps the question here.
    """
    text = text or ""
    raw = len(text)

    def _skip(outcome: str, reason: str) -> TailDecision:
        return TailDecision(False, "", outcome, reason, raw)

    if holed:
        return _skip(
            tailhealth.SKIPPED_HOLED,
            "the stream accumulator dropped a chunk, so the text has a hole "
            "in it that nothing downstream could see",
        )
    if not text.strip():
        return _skip(tailhealth.SKIPPED_EMPTY, "the reply is empty")
    if finished and not truncated:
        why = reply_is_degenerate(text)
        if why:
            return _skip(
                tailhealth.SKIPPED_DEGENERATE,
                f"reply looks like a repetition loop ({why})",
            )
        return TailDecision(True, text, tailhealth.STORED, None, raw)
    # Cut. The two phrasings are what scripts/tail-logs.sh's signal filter
    # matches on ("stream ended", "stream truncated"), and what the log has
    # said since v3.1, so a grep across the upgrade still works.
    how = (
        "stream truncated at the generation ceiling (finish_reason=length)"
        if truncated
        else "stream ended without completion"
    )
    kept = trim_to_last_sentence(text)
    if not kept:
        return _skip(
            tailhealth.SKIPPED_NO_BOUNDARY,
            f"{how} and no sentence boundary survives in {raw} chars",
        )
    if len(kept) < MIN_MEMORABLE_TRIMMED_CHARS:
        return _skip(
            tailhealth.SKIPPED_TOO_SHORT,
            f"{how}; only {len(kept)} of {raw} chars end on a sentence "
            f"boundary (floor {MIN_MEMORABLE_TRIMMED_CHARS})",
        )
    why = reply_is_degenerate(kept)
    if why:
        return _skip(
            tailhealth.SKIPPED_DEGENERATE_PARTIAL,
            f"{how}; the {len(kept)} chars that end on a sentence boundary "
            f"look like a repetition loop ({why})",
        )
    return TailDecision(True, kept, tailhealth.STORED_TRIMMED, how, raw)


def _repair_template_invalid_tail(body: dict) -> tuple[str | None, bool]:
    """Make the OUTGOING payload's tail valid for the Mistral chat template.

    Returns (description, was_actually_invalid). The second element is
    False when the payload would already have been accepted by vLLM - a
    client that sent continue_final_message itself, for instance - so the
    caller can avoid warning about a request that was never in danger.

    THE BUG THIS EXISTS FOR, observed in production 2026-08-29 22:38:46.
    Every add_generation_prompt / continue_final_message guard in this file
    lived in count_tokens_exact - the MEASURING path. Nothing guarded the
    payload actually forwarded to /v1/chat/completions, so vLLM refused the
    generation itself:

        ValueError: Cannot set `add_generation_prompt` to True when the last
        message is from the assistant. Consider using
        `continue_final_message` instead.

    and the compactor logged "this turn produced no reply, no facts and no
    episodic write, and nothing retries it". It happened 5 times today.

    HOW THE PAYLOAD GETS INTO THAT SHAPE. It is a cascade, not a client bug.
    A stream that dies mid-reply leaves an EMPTY assistant turn in the
    client's history (28 such streams today). OpenWebUI resends the whole
    array on the next turn, so that empty assistant turn comes back as the
    final message, and the template refuses to build a generation prompt from
    it. One dead stream therefore poisons the NEXT turn as well - which is
    exactly the 22:37 -> 22:38 pair in the logs.

    So there are two distinct tails to repair, and they need opposite fixes:

      1. A trailing assistant turn with NO content carries no information.
         It is the residue of a failed turn. Drop it, and the array ends on
         the user's real question again.
      2. A trailing assistant turn WITH content is a genuine "continue this
         reply" request. Dropping it would silently discard what the user
         asked to continue, so instead say so explicitly with
         continue_final_message, which is what the template's own error
         message tells you to do.

    Never removes the last remaining user turn, and never empties the array.
    """
    msgs = list(body.get("messages") or [])
    if not msgs:
        return None, False
    note = None
    # Was the payload ALREADY valid on arrival? An assistant-final list that
    # the client had already flagged with continue_final_message is exactly
    # what the template asks for; repairing it changes nothing.
    already_ok = bool(body.get("continue_final_message")) and not body.get(
        "add_generation_prompt"
    )

    # (1) Shed the residue of dead streams. Bounded by the presence of a real
    # user turn so this can never eat the conversation.
    # assistant_content_is_empty, NOT _message_text().strip(). _message_text
    # joins TEXT parts and silently ignores every other kind, so an assistant
    # turn carrying only an image read as empty here and was popped — the
    # image destroyed, permanently and silently — while step (1b) below was
    # refusing to touch that exact shape three lines later. One rule now
    # serves the drop and the fill; see assistant_content_is_empty.
    dropped = 0
    while (
        len(msgs) > 1
        and msgs[-1].get("role") == "assistant"
        and assistant_content_is_empty(msgs[-1].get("content"))
        and any(m.get("role") == "user" for m in msgs[:-1])
    ):
        msgs.pop()
        dropped += 1
    if dropped:
        note = (
            f"dropped {dropped} empty trailing assistant turn(s) left behind "
            f"by an earlier failed stream"
        )

    # (1b) INTERIOR empty assistant turns. Production, 2026-08-30 06:41:
    #
    #   06:40:42  stream cancelled at 0 chars (msgs=312)
    #             -> OpenWebUI stores an EMPTY assistant turn
    #   she types a NEW message (not a regenerate)
    #   06:41:06  payload is USER-final, the empty turn now INTERIOR
    #             (msgs=314) -> the template refuses the whole request:
    #             "Invalid assistant message: role='assistant' content=''"
    #   06:42:06  she recovered by deleting messages by hand
    #
    # Step (1) above only sheds empties while they are LAST, which covers
    # the regenerate flow its docstring describes and misses the far more
    # common type-a-new-message flow entirely. Four such rejections in the
    # 08-28..08-30 window, ~2 dead turns a day, each with HTTP 200 already
    # committed so no access log shows it.
    #
    # SPACE-FILLED, NOT DROPPED. Dropping an interior turn would splice two
    # user turns together and break the alternation the template also
    # requires - trading this 400 for a different one. A single space is
    # the minimal content vLLM 0.19's template verifiably accepts (empty
    # string is refused; see the note in (2) below and
    # testfixtures/tokenizer-contract/vllm_template_probe.py).
    #
    # str content only: a list (multimodal) part can read as text-empty
    # while still carrying an image, and destroying an image to satisfy a
    # template rule would be worse than the 400.
    # Shared with count_tokens_exact since v3.1.5 — one rule, one
    # implementation. This site fixes what we FORWARD; that one fixes what we
    # MEASURE, and having only this one was what let a single cancelled
    # stream degrade every budget decision for a conversation.
    msgs, filled = _space_fill_empty_assistant(msgs)
    if filled:
        body["messages"] = msgs
        _f = (
            f"space-filled {filled} empty assistant turn(s) left behind by "
            f"cancelled stream(s) - the template refuses empty content"
        )
        note = f"{note}; {_f}" if note else _f

    # (2) A real assistant-final list is a continuation, not an error.
    if msgs and msgs[-1].get("role") == "assistant":
        # VERIFIED against vLLM 0.19's own template stack (the full
        # MistralTokenizer -> transformers MistralCommonTokenizer pipeline,
        # driven directly in the production image, 2026-08-30): an
        # assistant-final message with EMPTY string content is refused even
        # with continue_final_message set - "Assistant message must have
        # either content or tool_calls" - while whitespace-only content is
        # accepted. The first version of this repair kept a lone empty
        # assistant turn (nothing to fall back to) and set the flag, which
        # converted one 400 into a different 400. A single space is the
        # minimal content the template verifiably accepts. Only str content
        # is touched: a list (multimodal) that reads as text-empty may
        # still carry an image, and destroying it to satisfy a template
        # rule would be worse than the 400.
        # (the fill itself now happens once, in (1b) above, which covers
        # every position including this one)
        body["continue_final_message"] = True
        body["add_generation_prompt"] = False
        cont = "asked vLLM to CONTINUE the final assistant turn rather than start a new one"
        note = f"{note}; {cont}" if note else cont
    else:
        # Both flags are refused together, so never leave a stale pair behind
        # from a client that sent one.
        body.pop("continue_final_message", None)
        body.pop("add_generation_prompt", None)

    if dropped:
        body["messages"] = msgs
    # Dropping or filling a turn is always a real repair; setting the flag on
    # a payload that already carried it is not.
    return note, bool(dropped) or bool(filled) or not already_ok


def _warn_if_conversation_forked(
    conv_id: str, source: str, messages: list[dict]
) -> None:
    """Shout when a LONG conversation resolves to a brand-new conv_id.

    The hash fallback is sha256(system|||first_user[:512]), so editing the
    system prompt gives a live conversation a new identity and forks its
    memory: facts, episodic embeddings and summaries all keep accumulating,
    just under an id nothing else references. Production, 2026-08-30: a
    prompt edit forked a 400-turn conversation and left 106 facts and ~85
    indexed exchanges stranded. It went unnoticed for hours because every
    individual signal looked healthy - the new id summarized fine, extracted
    fine, answered fine. Only the id changed, and nothing said so.

    The signature is unmistakable and costs one dict lookup: many messages,
    id derived by HASH, and no stored state under that id. A brand-new
    conversation has few messages; a resumed one has state. Long AND empty
    means the identity moved.

    WARNING, not INFO: by the time anyone reads INFO the facts have already
    been accumulating in the wrong place for a day.
    """
    if source != "hash":
        return  # header / metadata ids are stable across prompt edits
    try:
        if len([m for m in messages if m.get("role") != "system"]) < 20:
            return
        if facts.load_facts(conv_id):
            return
        if (summarizer.load_state(conv_id) or {}).get("last_summarized_turn"):
            return
    except Exception:
        return
    logger.warning(
        f"conv={conv_id}: a {len(messages)}-message conversation resolved to "
        f"a conv_id with NO stored memory, derived by HASH. That is the "
        f"signature of a FORK: the id is sha256(system|||first_user), so a "
        f"system-prompt edit gives a live conversation a new identity and "
        f"strands its facts, embeddings and summaries under the old one. "
        f"Check GET /admin/conversations for a sibling id that went quiet, "
        f"and POST /admin/conversations/<old>/merge-into/{conv_id} to fold "
        f"it back. To stop this recurring, have the client send "
        f"X-Conversation-Id or metadata.chat_id - both are stable across "
        f"prompt edits."
    )


def _has_conversational_history(messages: list[dict]) -> bool:
    """Whether the CLIENT's array contains a prior assistant turn.

    Computed on what the client sent, before compaction or injection touched
    it. "A prior assistant turn" and not "more than one message" because the
    former is what actually distinguishes a conversation from a task: a
    background title call and a brand-new chat both arrive with one or two
    messages and no reply behind them, while a real second turn always carries
    the first answer."""
    return any(
        isinstance(m, dict) and m.get("role") == "assistant" for m in messages
    )


# Turns, not exchanges: _recorded_position counts message-units, two per
# exchange. 4 means "this conversation has genuinely got two exchanges deep",
# which is the point past which an array with no assistant turn in it stops
# being an honest picture of it. Deliberately not 2: a brand-new conversation
# whose single first reply was regenerated sits at 2, and that is a real
# exchange that must still be memorized.
TASK_TRAFFIC_MIN_POSITION = _env_int("COMPACTOR_TASK_TRAFFIC_MIN_POSITION", 4)


def _is_repeat_task_traffic(conv_id: str, messages: list[dict]) -> bool:
    """Whether this is OpenWebUI background task traffic, not a conversation.

    v3.1.8 (N4b). The compactor has classified this shape since v3.1 — the
    "task traffic or a first turn" line in the over-budget warning — but only
    the INJECTION side ever acted on it (INJECTION_NO_HISTORY_FRACTION, 0.125
    against 0.5). The memory tail was never told, so OpenWebUI's title, tag
    and follow-up calls are still fact-extracted, episodically indexed and
    deduped: N3's "second treadmill", ~90 s after every real turn, on a
    conversation that is not one. Measured 2026-09-03/04: 99 such requests in
    two days on conv=026752…, against 0 real exchanges.

    THE CLASSIFICATION ALONE IS NOT ENOUGH, and this is the whole reason this
    is a function rather than a call to _has_conversational_history at the
    tail site. That predicate is False for a genuine FIRST TURN too — it says
    so itself — so skipping the tail on it would drop the opening exchange of
    every new conversation, permanently and silently. That is a worse bug
    than the one being fixed, and it is the shape the backlog's own suggested
    fix direction would have produced.

    What separates them is not the request, it is the history: OpenWebUI's
    task calls arrive on a STABLE conv_id, over and over, each time with no
    assistant turn, for as long as the deployment lives. A real conversation
    looks like that only at its very beginning.

    "AT ITS VERY BEGINNING" IS WHY THERE IS A THRESHOLD HERE RATHER THAN A
    BARE "have we stored anything". The first draft used file existence, and
    test_budget_guard caught it immediately with a fixture that is a perfectly
    ordinary shape: a conversation with a facts file receiving a history-less
    turn. That is what OpenWebUI sends when the user REGENERATES the first
    reply, and eating it would be a silent memory loss on a real exchange.
    So the bar is the recorded POSITION: a conversation that has genuinely
    got two exchanges deep cannot honestly present an array with no assistant
    turn in it, while task traffic passes that bar within its first few
    minutes and stays past it forever.

    Cost: one state read, and only on requests that already have no assistant
    turn — never on the hot path of an ongoing conversation.

    Three consequences, stated rather than left to be discovered:

      * The first few task calls for a given conv_id are memorized, because at
        that moment they are indistinguishable from a new conversation. A
        handful of polluted exchanges per task conv_id for the life of the
        store, not 99 every two days.
      * Regenerating the FIRST message of a conversation already two exchanges
        deep reads as task traffic and is not memorized. Rare, self-limiting
        (the next turn carries an assistant message and behaves normally), and
        — unlike the defect this replaces — COUNTED and named, so it is
        visible in /health/full rather than silent.
      * This is the code half of N4. The backlog's first recommendation is a
        separate task model in OpenWebUI's admin settings, and that remains
        the better fix: it stops the traffic reaching the compactor at all,
        where this can only decline to remember it.
    """
    if _has_conversational_history(messages):
        return False
    try:
        position = summarizer._recorded_position(summarizer.load_state(conv_id))
    except (OSError, StoreUnreadable):
        # These two, NOT bare Exception, and the narrowness is the point. The
        # first draft of this function caught Exception and referenced the
        # `memory` module by a name main.py does not bind — so every call
        # raised NameError, the catch-all swallowed it, and the function
        # quietly answered "not task traffic" for everything. It looked like a
        # fix and did nothing, which is the exact failure this file is full of
        # comments about. A state read can fail for real (a vanished mount, a
        # permission change, a half-written file) and that must fall back to
        # the old behaviour: memorizing task traffic is the bug being fixed,
        # dropping a real exchange is worse. Anything else is a programming
        # error and must be allowed to be loud.
        return False
    return position >= TASK_TRAFFIC_MIN_POSITION


# ---------------------------------------------------------------------------
# v3.1.9: THE CURRENT DATE AND TIME (V4_ROADMAP.md section 1.1, item 1)
# ---------------------------------------------------------------------------
#
# The user reported "it doesn't keep track of the actual time", and the cause
# was total: not one layer of the prompt carried wall-clock time. Asked the
# time, the model invented one, and the inventions ("4:01 AM", "9:19 AM
# Friday") reached the fact store as facts. So the forwarded request now
# carries ONE line, at the head of the newest user message:
#
#     [Current date and time: Monday, September 14, 2026, 9:41 AM MST (UTC-07:00)]
#
# WHERE IT GOES, and the two places it must not:
#
#   * NOT in the leading system block. vLLM's prefix cache keys on the leading
#     prompt; a value that changes every minute near the front would recompute
#     her whole ~100k-token context on every message.
#   * NOT as a separate system message near the end. Mistral-family templates
#     (Cydonia-24B is Mistral Small) accept one leading system message and then
#     strict user/assistant alternation, and 400 anything else - see "at most
#     one system message" at the memory injection in chat_completions.
#   * So it is a PREFIX of the newest user message, added to the FORWARDED
#     payload only, after the guard, the merges and the tail repair have
#     settled which message that is. OpenWebUI never sees it and never re-sends
#     it, so the model only ever sees one line (the current one), and every
#     earlier turn reaches the prefix cache exactly as it did before. What it
#     does cost: the previous request's newest user turn was sent WITH a line
#     and is re-sent without one, so the cached prefix now ends at the start
#     of that turn instead of after it - one user turn and one reply
#     re-prefilled per message, not the conversation.
#
# WHAT THE LINE IS NOT ALLOWED TO REACH. Every memory writer - fact extraction,
# the episodic index, the rollup, the covered-turn record, backfill - reads the
# REQUEST (`messages`, `last_user_text`), never `body["messages"]`, and
# _inject_time_line copies the one message it changes instead of mutating it:
# the compacted array shares message dicts with `messages`, so an in-place edit
# would put the line into the request the tail reads. A fingerprint of a user
# turn carrying a line no later request contains would read as an EDIT to the
# reuse gate and switch compaction reuse off (test_time_memory.py).
#
# THE WORDING. Square brackets and a label, no verb and no addressee: it reads
# as metadata attached to her message, the way a client stamps a message, not
# as something she typed and not as an instruction the model should act on or
# acknowledge. Nothing in it asks for a reply ("note", "remember", "the user's
# time is" would all invite one). The assistant turns the model conditions on
# never contain it, so there is no earlier reply of its own to imitate. Weekday
# and month are spelled out because models are unreliable at deriving a
# weekday from a date; minutes and no seconds, because a seconds value is
# stale before the reply is read. The zone is given as its abbreviation AND
# its UTC offset, because abbreviations collide (CST, IST) and the offset is
# what makes "is it morning where she is" unambiguous.
#
# WHICH ZONE: HER BROWSER'S, then the operator's, then UTC. OpenWebUI 0.11.0
# renders `{{CURRENT_TIMEZONE}}` in a model's system prompt with the browser's
# Intl.DateTimeFormat().resolvedOptions().timeZone (frontend
# $lib/utils getUserTimezone -> the chat request's `variables`;
# utils/payload.py resolve_system_prompt -> prompt_variables_template, a plain
# string replace; the rendered prompt becomes messages[0]). The request
# metadata itself never reaches the compactor (routers/openai.py pops it), so
# the rendered system prompt is the only carrier. The operator adds ONE line,
# `User timezone: {{CURRENT_TIMEZONE}}`, to her model's system prompt; it is
# constant for her, so the leading prompt stays cache-friendly, and it follows
# her device if she travels. The compactor reads that label from the LEADING
# system message only - never from a user or assistant turn, where anyone
# could type it - validates it with zoneinfo, and forwards the system prompt
# unchanged (the model can read the line too). A client that does not render
# variables (a direct API call) sends the literal placeholder, which falls back
# to COMPACTOR_TIMEZONE and then UTC, and says so once.

TIME_INJECTION_ENABLED = env_bool("COMPACTOR_TIME_INJECTION", True)
TIME_LINE_PREFIX = "[Current date and time: "
_TIME_LINE_SEPARATOR = "\n\n"
# Tokens beyond the line's own bytes that joining it to her message may cost:
# the separator's merge with her first word, and the template's separator
# between content parts on a list-content turn. Generous on purpose - at 16
# out of a 32k window it costs nothing, and it is the one number standing
# between an unmeasured addition and a context-length 400.
_TIME_LINE_JOIN_SLACK = 16
_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
             "Saturday", "Sunday")
_MONTHS = ("January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December")
# Names that mean UTC and must work WITHOUT a timezone database: a slim image
# or a Windows interpreter has no tzdata, and ZoneInfo("UTC") raises there.
_UTC_ZONE_NAMES = frozenset({"UTC", "ETC/UTC"})
# OpenWebUI's background task names (open_webui.constants.TASKS). With the
# connection header this project documents, {"X-Conversation-Id":
# "{{CHAT_ID}}{{TASK}}"} (RUNBOOK_MEMORY_IDENTITY.md), a task call arrives on
# "<chat uuid><task name>", so the suffix identifies it on its FIRST call -
# before _is_repeat_task_traffic has any history to judge by. Used only to
# decide whether to date a request; it changes no memory decision.
_OPENWEBUI_TASK_SUFFIXES = (
    "title_generation", "tags_generation", "follow_up_generation",
    "emoji_generation", "query_generation", "image_prompt_generation",
    "autocomplete_generation", "function_calling", "moa_response_generation",
)


# hostile pass #5 (reviewer A F1). Ubuntu 24.04 split tzdata's `backward`
# file — the aliases below — into a separate `tzdata-legacy` package the
# shipped image does not install, so ZoneInfo(raw) fails for every one of
# these even though the zone they NAME exists under its current name. Fixing
# that for real is a Dockerfile change (add tzdata-legacy, or pip install
# tzdata into the compactor venv) and belongs to whoever owns the image, not
# this file. What belongs here: when a legacy name fails, and this table
# happens to know its replacement, say so — an operator who typed the name
# she has always used should see the name that will actually resolve on this
# image, not just "not a usable IANA time zone name" and a guess. This map
# does NOT change what resolves; it only makes the error actionable, and only
# for names on it. Small and hand-picked (the ones the shipped image was
# actually probed against, SP\p5-a\tzprobe.out) rather than exhaustive,
# because a wrong canonical name would suggest a fix that PRODUCES the wrong
# offset for a zone with a genuinely different history (e.g. Asia/Calcutta
# and Asia/Kolkata are the same zone; not every backward link is this clean).
_LEGACY_ZONE_ALIASES = {
    "us/arizona": "America/Phoenix",
    "asia/calcutta": "Asia/Kolkata",
    "europe/kiev": "Europe/Kyiv",
    "asia/katmandu": "Asia/Kathmandu",
    "america/buenos_aires": "America/Argentina/Buenos_Aires",
    "asia/saigon": "Asia/Ho_Chi_Minh",
    "america/godthab": "America/Nuuk",
}


def _resolve_time_zone(environ) -> tuple[tzinfo, str, str, str | None]:
    """(zone, name, source, error) from COMPACTOR_TIMEZONE, else UTC. Also
    validates the browser-supplied name (_browser_time_zone passes it in under
    the same key).

    NEVER RAISES, for envcfg's reason: this runs at import, and a typo in a
    RunPod template field must not stop the container booting. An unusable
    name resolves to UTC and returns the error, which _announce_time_zone logs
    at ERROR once and /health/full's config block shows for as long as the
    process lives.

    TZ IS NOT READ. runpod.env.template sets TZ for the container, and it is
    tempting to use it as a second fallback; it is not one, because the
    owner's precedence is browser -> COMPACTOR_TIMEZONE -> UTC, and a second
    environment variable quietly feeding the clock is a source nobody would
    think to check when the time is wrong.
    """
    raw = str(environ.get("COMPACTOR_TIMEZONE") or "").strip()
    source = "COMPACTOR_TIMEZONE"
    if not raw:
        return timezone.utc, "UTC", "default", None
    if raw.upper() in _UTC_ZONE_NAMES:
        return timezone.utc, "UTC", source, None
    try:
        return ZoneInfo(raw), raw, source, None
    except Exception as e:  # ZoneInfoNotFoundError, ValueError (a path), OSError
        suggestion = ""
        canonical = _LEGACY_ZONE_ALIASES.get(raw.lower())
        if canonical:
            try:
                ZoneInfo(canonical)  # does THIS image actually have it?
                suggestion = f" — did you mean {canonical!r}? That name resolves here."
            except Exception:
                pass  # the alias doesn't help on this image either; say nothing extra
        return timezone.utc, "UTC", source, (
            f"{source}={raw!r} is not a usable IANA time zone name "
            f"({type(e).__name__}: {e}){suggestion}"
        )


_TIME_ZONE, TIME_ZONE_NAME, _TIME_ZONE_SOURCE, _TIME_ZONE_ERROR = _resolve_time_zone(
    os.environ
)

# The exact, documented label (RUNPOD_DEPLOY.md). At the start of a line of the
# LEADING system message; the value is the rest of that line. Case-sensitive on
# purpose: a documented string is matched as documented, not approximately.
TIME_ZONE_PROMPT_LABEL = "User timezone:"
_TIME_ZONE_LABEL_RE = re.compile(
    r"^[ \t]*" + re.escape(TIME_ZONE_PROMPT_LABEL) + r"[ \t]*(.*?)[ \t]*$",
    re.MULTILINE,
)
# What the last dated request used, for /health/full. Process-local, written on
# the request path, read by the probe; a torn read between two requests can
# only mix two valid states, and nothing decides anything on it.
_LAST_TIME_ZONE: dict = {"source": None, "timezone": None, "browser_error": None}
_last_time_zone_obj: tzinfo | None = None


def _browser_time_zone(messages: list[dict]) -> tuple[tzinfo | None, str | None, str | None]:
    """(zone, name, error) from the label in the LEADING system message.

    (None, None, None) when there is no leading system message or no label -
    an ordinary request, nothing to report. (None, None, error) when the label
    is there and unusable: an unrendered `{{CURRENT_TIMEZONE}}` (a client that
    does not render OpenWebUI's variables) or a name zoneinfo refuses. Only
    messages[0] is read, and only when it is a system message: a label in a
    user or assistant turn is text someone typed and must never move the clock.
    """
    if not messages or not isinstance(messages[0], dict) \
            or messages[0].get("role") != "system":
        return None, None, None
    found = _TIME_ZONE_LABEL_RE.search(_message_text(messages[0]))
    if found is None:
        return None, None, None
    raw = found.group(1).strip()   # .strip(): a CRLF prompt leaves a '\r'
    if not raw or "{{" in raw or "}}" in raw:
        return None, None, (
            f"the system prompt's {TIME_ZONE_PROMPT_LABEL!r} line is "
            f"{raw[:80]!r}, not a rendered time zone (a client that does not "
            f"substitute {{{{CURRENT_TIMEZONE}}}})"
        )
    zone, name, _source, err = _resolve_time_zone({"COMPACTOR_TIMEZONE": raw})
    if err:
        return None, None, (
            f"the system prompt's {TIME_ZONE_PROMPT_LABEL!r} line names "
            f"{raw[:80]!r}, which is not a usable IANA time zone"
        )
    return zone, name, None


def _request_time_zone(messages: list[dict]) -> tuple[tzinfo, str, str, str | None]:
    """(zone, name, source, browser_error) for one request. source is
    "browser", "env" (a valid COMPACTOR_TIMEZONE) or "utc"."""
    zone, name, browser_error = _browser_time_zone(messages)
    if zone is not None:
        return zone, name, "browser", None
    if browser_error and logsetup.log_once("time_injection.browser_zone_unusable"):
        logger.warning(
            f"{browser_error}; dating her messages in the fallback zone "
            f"{TIME_ZONE_NAME} instead. Said once per process; /health/full "
            f"config.time_injection shows the source in use."
        )
    if _TIME_ZONE_SOURCE == "COMPACTOR_TIMEZONE" and not _TIME_ZONE_ERROR:
        return _TIME_ZONE, TIME_ZONE_NAME, "env", browser_error
    return timezone.utc, "UTC", "utc", browser_error


def _now_utc() -> datetime:
    """The wall clock. A seam so tests can pin the minute."""
    return datetime.now(timezone.utc)


def _format_time_line(now: datetime, zone: tzinfo) -> str:
    """The line, for instant `now` in `zone`. Pure; English names come from
    fixed tables, not strftime, because %A and %B follow the process locale."""
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    local = now.astimezone(zone)
    offset = local.utcoffset() or timedelta(0)
    abbr = local.tzname() or ""
    if offset == timedelta(0) and abbr in ("UTC", ""):
        label = "UTC"
    else:
        minutes = int(offset.total_seconds()) // 60
        hh, mm = divmod(abs(minutes), 60)
        numeric = f"UTC{'+' if minutes >= 0 else '-'}{hh:02d}:{mm:02d}"
        # A zone with no letter abbreviation reports its offset AS its name
        # ("-03" for America/Sao_Paulo); printing both would say it twice.
        label = f"{abbr} ({numeric})" if abbr[:1].isalpha() else numeric
    hour12 = local.hour % 12 or 12
    return (
        f"{TIME_LINE_PREFIX}{_WEEKDAYS[local.weekday()]}, "
        f"{_MONTHS[local.month - 1]} {local.day}, {local.year}, "
        f"{hour12}:{local.minute:02d} {'AM' if local.hour < 12 else 'PM'} {label}]"
    )


def current_time_line(now: datetime | None = None) -> str:
    """The line in the FALLBACK zone (COMPACTOR_TIMEZONE, else UTC) - what a
    request without a usable browser zone is shown."""
    return _format_time_line(now if now is not None else _now_utc(), _TIME_ZONE)


def time_injection_state() -> dict:
    """For /health/full's config block (health.py cannot import main).

    `last_source` / `last_timezone`: what the most recent dated request used
    ("browser", "env", "utc"; None before the first). `current_line`: the line
    a request would get right now in that zone (the fallback zone before the
    first request). `fallback_*`: the operator's zone and why it is not in
    force if it is not."""
    zone = _last_time_zone_obj if _last_time_zone_obj is not None else _TIME_ZONE
    return {
        "enabled": TIME_INJECTION_ENABLED,
        "last_source": _LAST_TIME_ZONE["source"],
        "last_timezone": _LAST_TIME_ZONE["timezone"],
        "last_browser_error": _LAST_TIME_ZONE["browser_error"],
        "fallback_timezone": TIME_ZONE_NAME,
        "fallback_source": _TIME_ZONE_SOURCE,
        "fallback_error": _TIME_ZONE_ERROR,
        "prompt_label": TIME_ZONE_PROMPT_LABEL,
        "current_line": _format_time_line(_now_utc(), zone),
    }


def _announce_time_zone() -> None:
    """Say once per process what the model is told, and LOUDLY if the zone the
    operator set did not take. Called at startup and on the request path (a
    set lookup after the first call), so a process that never ran the
    lifespan still says it before its first dated reply."""
    if not logsetup.log_once("time_injection.announce"):
        return
    if _TIME_ZONE_ERROR:
        logger.error(
            f"TIME ZONE NOT APPLIED: {_TIME_ZONE_ERROR}. Requests without a "
            f"browser zone (the system prompt's {TIME_ZONE_PROMPT_LABEL!r} "
            f"line) are being told UTC instead, which is wrong by hours if she "
            f"is anywhere else. Fix COMPACTOR_TIMEZONE (a name such as "
            f"America/Phoenix, see RUNPOD_DEPLOY.md) and redeploy; /health/full "
            f"config.time_injection shows what is in force."
        )
    logger.info(
        f"current-time line "
        f"{'ON' if TIME_INJECTION_ENABLED else 'OFF (COMPACTOR_TIME_INJECTION)'}: "
        f"zone from the system prompt's {TIME_ZONE_PROMPT_LABEL!r} line, else "
        f"{TIME_ZONE_NAME} (from {_TIME_ZONE_SOURCE}); without a browser zone "
        f"the model sees {current_time_line()!r} at the head of her newest "
        f"message"
    )


def _is_openwebui_task_conv_id(conv_id: str | None) -> bool:
    return bool(conv_id) and str(conv_id).endswith(_OPENWEBUI_TASK_SUFFIXES)


# The literal opening of every OpenWebUI task-generation prompt (open_webui's
# get_task_model_id / task.py templates render "### Task:\n<instructions>\n
# ### Chat History:\n..." for title, tags, follow-ups, query generation and
# the rest of TASKS.*). See TASK_REQ in test_time_injection.py, which is this
# exact shape. Used only by _looks_like_openwebui_task_prompt, for the dating
# decision — never for the memory classifier, which has its own reasons
# (_is_repeat_task_traffic) that this string is not part of.
_OPENWEBUI_TASK_PROMPT_HEAD = "### Task:"


def _looks_like_openwebui_task_prompt(messages: list[dict]) -> bool:
    """Does the newest user turn look like OpenWebUI's own task template,
    rather than something she typed?

    hostile pass #5 (reviewer A F2). Under hash identity (production today —
    RUNBOOK_MEMORY_IDENTITY.md) there is no header to tell a title/tag/
    follow-up call apart from a first turn or a regenerate; only
    _is_repeat_task_traffic's history-POSITION heuristic does, and a
    conv_id built purely from content hash can put a genuine new (or
    regenerated) opener at the SAME position as an older, already-deep
    conversation that happens to hash-collide with it (same system prompt,
    same first 512 characters). That collision is real text from her, not a
    task call — dating it matters at least as much as dating a task template
    does, since an ordinary opening line is exactly where the model not
    knowing the time would show. Confirmed with a synthetic fixture, not
    production data (test_time_injection.py [6b]): an opener with no
    task-shaped content, seeded to collide by hash with an older,
    already-deep conversation, was sent undated before this fix and is
    dated after it.

    So this narrows what "_is_repeat_task_traffic says yes" is allowed to
    mean for the DATING decision only: not dated only when the request is
    ALSO shaped like the thing that classifier exists to recognise. Checked
    as an ADDITIONAL condition, never a replacement — a real first turn that
    happens to open with a markdown heading is not task traffic just because
    it looks like one; the position bar is what actually gates that, and
    still must pass first. The memory tail's classification (whether this
    gets memorized) is untouched: it has no template-text check and this
    function is not called from it. That half is a known identity
    limitation, not something a text heuristic can safely close — a real
    conversation COULD legitimately open with a message that starts
    "### Task:" (a user pasting one), and skipping the tail on it would be
    exactly the silent-memory-loss bug _is_repeat_task_traffic's docstring
    already warns about. Reported, not fixed here.
    """
    newest = next(
        (m for m in reversed(messages) if isinstance(m, dict) and m.get("role") == "user"),
        None,
    )
    if newest is None:
        return False
    return _message_text(newest).lstrip().startswith(_OPENWEBUI_TASK_PROMPT_HEAD)


def _time_line_for_request(conv_id: str | None, messages: list[dict]) -> str | None:
    """The line to date this request with, or None. Decided on the ORIGINAL
    request, before anything is forwarded, so the guard can reserve its size.

    Task traffic is not dated: a title or a tag that absorbs "Monday, 9:41 AM"
    is worse than one that does not, and a follow-up suggestion is not a turn
    she is waiting on. Both classifiers are READ here, never changed - the
    memory tail makes its own decision later, on the same `messages`.
    """
    if not TIME_INJECTION_ENABLED:
        return None
    if not any(isinstance(m, dict) and m.get("role") == "user" for m in messages):
        return None
    if _is_openwebui_task_conv_id(conv_id):
        return None
    # hostile pass #5 F2: _is_repeat_task_traffic alone over-fires on a
    # hash-identity collision (see _looks_like_openwebui_task_prompt) — add
    # the shape check so only genuine task calls skip dating here. The
    # memory classifier below this function is untouched.
    if conv_id and _is_repeat_task_traffic(conv_id, messages) \
            and _looks_like_openwebui_task_prompt(messages):
        return None
    global _last_time_zone_obj
    zone, name, source, browser_error = _request_time_zone(messages)
    _LAST_TIME_ZONE.update(source=source, timezone=name, browser_error=browser_error)
    _last_time_zone_obj = zone
    return _format_time_line(_now_utc(), zone)


def _time_line_token_reserve(line: str) -> int:
    """An upper bound on what adding `line` can cost, in tokens.

    UTF-8 bytes, not an estimate: every tokenizer this project runs (Mistral's
    tekken, the byte-level BPE fixtures) spends at least one byte per token, so
    a line cannot cost more tokens than it has bytes. Plus the separator and a
    fixed slack for the join (see _TIME_LINE_JOIN_SLACK). ~20 real tokens,
    reserved as ~100: the difference is noise in a 32k window, and the bound
    needs no /tokenize round trip on the request path.
    """
    return (len(line.encode("utf-8")) + len(_TIME_LINE_SEPARATOR.encode("utf-8"))
            + _TIME_LINE_JOIN_SLACK)


def _inject_time_line(messages: list[dict], line: str) -> tuple[list[dict], bool]:
    """Prefix the NEWEST user message of `messages` with `line`. Returns
    (messages, injected). Never mutates: a new list, and a new dict for the one
    message changed (see the section comment for why that is load-bearing).

    A string turn becomes line + blank line + her text. A content-LIST turn (an
    image, or a client that sends parts) gets a leading text part rather than
    having one of its parts rewritten, so the parts she sent reach the backend
    untouched. Any other content shape, or no user message at all, is left
    alone.
    """
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        content = m.get("content")
        if isinstance(content, str):
            new_content: Any = (
                f"{line}{_TIME_LINE_SEPARATOR}{content}" if content else line
            )
        elif isinstance(content, list):
            new_content = [{"type": "text", "text": line + _TIME_LINE_SEPARATOR},
                           *content]
        else:
            return messages, False
        out = list(messages)
        out[i] = {**m, "content": new_content}
        return out, True
    return messages, False


def _bound_injected_blocks(
    blocks: list[tuple[int, str, str]], budget: int
) -> tuple[list[str], list[str], int]:
    """Drop whole injected layers, lowest priority first, until they fit.

    `blocks` is [(priority, label, text)] in the order they must be SENT — see
    inject_system_block for why that order matters to the model. Returns
    (texts still in send order, labels dropped, the cost estimate used).

    Why whole layers rather than truncation: every layer here already has an
    internal budget and an internal ranking, so truncating one from the outside
    cuts it at a point its own ranking did not choose. Halving a retrieval
    block leaves half an exchange; dropping it leaves a conversation. The guard
    downstream still trims as a last resort, but this is the layer that can
    make the choice knowing what each block IS.

    The floor is ONE layer, and that is deliberate. A bound that can round down
    to nothing is not a bound, it is a refusal, and refusing makes "she
    remembers me from the first message" impossible — which is the product. So
    the drop loop stops before it takes the last surviving block. What that
    leaves is at most one layer's own cap (400 tokens for facts by default
    since F1 decoupled injection from the store cap - see
    COMPACTOR_INJECT_FACTS_TOKENS; 1500 for
    retrieval, a generation ceiling for a summary chunk), which is the
    strongest bound expressible here without overriding a module's own budget;
    the hole this function closes is that those caps SUM, not that any one of
    them is too large. And the guard downstream now sheds injected memory to
    nothing before it will forward a payload it knows will 400 (v3.1 D3), so
    the composite still has a floor of zero when the window genuinely demands
    one — it is just reached with the whole payload in view rather than here.

    Measurement discipline (P0-0c): per-block counts are local, because one
    /tokenize per block would be four round trips on the request path. The
    local counter reads up to 51% low on this model's content, so the cheap
    path is deliberately PESSIMISTIC — if the local sum still fits after being
    scaled by the worst measured undercount, nothing needs measuring and no
    HTTP call is made. Only a payload that might genuinely be over pays for one
    exact measurement, and that measurement supplies the scale the per-block
    arithmetic then uses. Same shape as _enforce_hard_budget: one ground truth,
    scaled per-part estimates, never a per-part round trip."""
    if not blocks:
        return [], [], 0
    local = [
        count_tokens([{"role": "system", "content": text}]) for _, _, text in blocks
    ]
    total_local = sum(local)
    if int(total_local * _PESSIMISTIC_SUMMARY_SCALE) <= budget:
        # Fits even if the local counter is as wrong as it has ever been
        # measured to be. No measurement can change the outcome, so none is
        # made.
        return [text for _, _, text in blocks], [], total_local
    # Measured as TEXT, not as a one-message conversation.
    #
    # The obvious form sends vLLM a message list whose only, and therefore
    # last, message is a system message. Nothing in this tree had ever sent
    # that shape, and /tokenize answers it by applying the served model's
    # chat template — the same machinery that refused an assistant-final
    # list and took compaction down on 2026-08-29 (D1). A refusal here
    # degrades /tokenize for the whole PROCESS and drops every budget in it
    # back onto the local estimator, so the blast radius is far wider than
    # the measurement it was serving.
    #
    # This is not a conversation and does not need a template: it is the
    # size of a blob of text. /tokenize's completion form answers that
    # without a template and so cannot refuse it for conversational-shape
    # reasons. summarizer.py's counter already uses this form.
    exact = count_text_tokens_exact("\n\n".join(t for _, _, t in blocks))
    scale = (
        (exact / total_local)
        if (exact is not None and total_local > 0)
        else _PESSIMISTIC_SUMMARY_SCALE
    )
    estimated = int(total_local * scale)
    cost = estimated
    keep = [True] * len(blocks)
    dropped: list[str] = []
    # Lowest priority first; among equals, the later-inserted block first.
    for i in sorted(range(len(blocks)), key=lambda j: (-blocks[j][0], -j)):
        if cost <= budget or sum(keep) <= 1:
            break
        keep[i] = False
        cost -= int(local[i] * scale)
        dropped.append(blocks[i][1])
    return (
        [text for k, (_, _, text) in zip(keep, blocks) if k],
        dropped,
        # The size that BLEW the budget, not the size that survived. A line
        # reading "0 tokens against 32" says nothing about how far over the
        # injection was, which is the only number that would tell an operator
        # whether the fraction is wrong or the memory is.
        estimated,
    )


def _merge_adjacent_system_messages(messages: list[dict]) -> list[dict]:
    """Collapse each run of consecutive system messages into a single one.

    Mistral-family templates (Magnum, Cydonia — anything built on
    Mistral-Small) reject multiple consecutive system messages with a 400.
    Memory injection is careful to emit ONE combined block, but V1 compaction
    prepends its own summary system message independently, so the two together
    can still produce a run. Applied just before forwarding, this makes the
    invariant hold no matter which layers fired.

    Image-bearing system messages are left alone rather than string-joined —
    collapsing them would destroy the image parts. But TEXT-ONLY list content
    (OpenAI content-parts form, no images) is flattened to a plain string
    first, so clients that send system prompts as parts still get merged —
    otherwise the adjacent-system run this exists to prevent survives intact
    (rc6 review finding).
    """
    if not isinstance(messages, list) or len(messages) < 2:
        return messages
    out: list[dict] = []
    for m in messages:
        if (
            isinstance(m, dict)
            and m.get("role") == "system"
            and isinstance(m.get("content"), list)
            and _message_image_count(m) == 0
        ):
            m = {**m, "content": _message_text(m)}
        mergeable = (
            isinstance(m, dict)
            and m.get("role") == "system"
            and isinstance(m.get("content"), str)
        )
        if mergeable and out and out[-1].get("role") == "system" and isinstance(out[-1].get("content"), str):
            prev = out[-1]
            out[-1] = {**prev, "content": f"{prev['content']}\n\n{m['content']}"}
        else:
            out.append(m)
    return out


def _fast_token_estimate(messages: list[dict]) -> int:
    """char/4 estimate + per-image cost — no tokenizer, O(total chars)."""
    image_tokens = sum(_message_image_count(m) for m in messages) * IMAGE_TOKEN_ESTIMATE
    return sum(len(_message_text(m)) // 4 + 4 for m in messages) + image_tokens


def _apply_image_retention(messages: list[dict]) -> tuple[list[dict], int]:
    """Keep images only on the most recent MAX_RETAINED_IMAGES image turns.

    Older image parts are replaced with a short text note, so the conversation
    still knows a picture was shared — it just can't be looked at again. Text
    parts are always preserved.

    Runs on EVERY request, before compaction: clients re-send full history, so
    without this an image uploaded once rides along forever, and real VLM image
    cost (thousands of tokens per photo) crowds the actual conversation out of
    the window. Returns (messages, images_demoted); the input list is never
    mutated.

    VALUES:  -1 = unlimited (no-op) · N > 0 = keep the N most recent image turns
             0  = strip EVERY image, INCLUDING the one just uploaded.

    The 0 case is easy to misread as "keep no history" — it is stronger than
    that. `MAX_RETAINED_IMAGES` is falsy at 0, so `keep` below is the empty set
    and the current turn's image is demoted with the rest: on a vision model the
    user uploads a picture, sees it in their composer, and the model never
    receives it. That is the intended production mitigation as of 2026-08-24
    (see INCIDENT_2026-08-24.md), but it is silent — the UI has no affordance
    telling the user their image was dropped. FRONTEND_SPEC.md §8 item 4 requires
    one.
    """
    if MAX_RETAINED_IMAGES < 0:
        return messages, 0
    img_idxs = [i for i, m in enumerate(messages) if _message_has_image(m)]
    if len(img_idxs) <= MAX_RETAINED_IMAGES:
        return messages, 0
    has_image = set(img_idxs)
    keep = set(img_idxs[len(img_idxs) - MAX_RETAINED_IMAGES:]) if MAX_RETAINED_IMAGES else set()
    out: list[dict] = []
    demoted = 0
    for i, m in enumerate(messages):
        if i not in has_image or i in keep:
            out.append(m)
            continue
        n = _message_image_count(m)
        txt = _message_text(m).strip()
        note = f"[{n} image{'s' if n > 1 else ''} shared earlier in this conversation]"
        out.append({**m, "content": f"{txt}\n\n{note}" if txt else note})
        demoted += n
    return out, demoted


def _memorable_user_text(messages: list[dict], last_user_text: str) -> str:
    """Give an image-only turn something the memory layers can hold onto.

    A bare upload (no caption) has NO text, so both memory layers correctly
    skip it — index_exchange and the facts tail each refuse empty input. But
    that leaves the conversation with no durable trace a picture was ever
    shared: images live only in the live window, and once they age out (or are
    demoted by the retention cap) the memory system cannot remember it was
    shown anything.

    Substituting a marker makes the exchange memorable, and the PAIRING is what
    matters: the assistant's description becomes the durable fact, so what was
    in the picture survives the picture itself.
    """
    if last_user_text.strip():
        return last_user_text
    last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
    if last_user is None:
        return last_user_text
    n = _message_image_count(last_user)
    if not n:
        return last_user_text
    return f"[shared {n} image{'s' if n > 1 else ''}]"


def _merge_consecutive_same_role(messages: list[dict]) -> list[dict]:
    """Collapse consecutive non-system messages that share a role.

    Mistral-family templates require strict user/assistant alternation, and
    several layers can independently break it. The one that bit us (v3.0.2):
    compaction hoists image-bearing turns out of chronological order to sit
    just before the recent window (they must never be summarized away), so
    an image turn landing next to the window's leading user turn produces
    user-then-user and vLLM 400s the request — every time, once an
    image-bearing conversation crosses the compaction threshold.

    Merging (rather than dropping) is lossless: text is joined, and if either
    side carries image parts both are kept as a parts list, so the picture
    still reaches the model.
    """
    if not isinstance(messages, list) or len(messages) < 2:
        return messages

    def as_parts(m: dict) -> list[dict]:
        c = m.get("content")
        if isinstance(c, list):
            return [p for p in c if isinstance(p, dict)]
        return [{"type": "text", "text": str(c or "")}]

    out: list[dict] = []
    for m in messages:
        prev = out[-1] if out else None
        if (
            prev is not None
            and isinstance(m, dict)
            and m.get("role") == prev.get("role")
            and m.get("role") in ("user", "assistant")
        ):
            if _message_image_count(m) or _message_image_count(prev):
                out[-1] = {**prev, "content": as_parts(prev) + as_parts(m)}
            else:
                a, b = _message_text(prev).strip(), _message_text(m).strip()
                out[-1] = {**prev, "content": f"{a}\n\n{b}".strip() if a or b else ""}
        else:
            out.append(m)
    if len(out) != len(messages):
        logger.info(
            f"merged {len(messages) - len(out)} consecutive same-role turn(s) "
            f"to preserve template alternation"
        )
    return out


def _droppable_system_indices(msgs: list[dict], protect_system: int) -> list[int]:
    """Indices of the system messages this guard is allowed to spend.

    `protect_system` is how many system messages the CALLER sent, counted on
    the original array before compaction or injection. Everything after that
    prefix is ours — injected memory, and compaction's own summary block.

    Extracted so the boundary is computed in ONE place. It was open-coded three
    times inside the guard (the trim loop's `sys_seen` walk, the drop loop's
    length test, and the give-up test), and a boundary restated three times is
    a boundary that drifts: the first cut of this guard applied the protection
    to the drop loop only, so the caller's prompt was safe from deletion and
    not from mutilation.

    P13-4 (hostile pass #13, LOW): this used to clamp with `sys_idxs[max(1,
    protect_system):]` — "protect AT LEAST the first system message, even
    if the caller says it sent none". That reads as conservative but is
    wrong for exactly the caller who legitimately sends none:
    `chat_completions` passes `caller_system = sum(1 for m in messages if
    role == "system")` on the ORIGINAL request, which is 0 for a
    conversation with no system prompt — and `inject_system_block` (v3.1.9)
    puts the FIRST thing this function ever injects (facts, then retrieval,
    then — Phase 4 — the summary) at index 0 in exactly that case. The
    clamp then protected THAT block as if the caller had sent it, so this
    function returned `[]`, the P12-5 branch below (gated on `if
    _droppable_system_indices(...)`) never even ran, and the floor-less
    generic shed loop it exists to preempt dropped her previous exchange
    first — with injected memory sitting right beside it, never spent, on
    every conversation with no system prompt. Proven synthetically
    (test_p5_guard.py's no-system-prompt section): facts survive whole and
    the previous exchange is lost, at every version, because this one
    clamp made "no caller system message" indistinguishable from "the
    caller's first message is sacred".

    Fixed by trusting the docstring's own contract: protect exactly the
    `protect_system` messages the caller actually sent, no floor. The real
    compaction stand-in is NOT protected here: the three loops in the
    P12-5 branch below already filter it out via `_is_compaction_standin`,
    by CONTENT, independent of position. Past that branch it is spendable
    as the very last resort, exactly as it always was behind a caller
    prompt; the clamp only ever shielded it from that when there was no
    caller prompt and it happened to sit at index 0 (test_budget_guard.py
    pins the parity). `_enforce_hard_budget`'s own `protect_system: int = 1`
    default is unchanged, so a direct caller that omits the argument still
    gets the pre-P13-4 behaviour — only a caller that explicitly PASSES 0
    (chat_completions, when it counted zero) sees anything spendable at
    index 0 now."""
    sys_idxs = [i for i, m in enumerate(msgs) if m.get("role") == "system"]
    return sys_idxs[protect_system:]


def _is_compaction_standin(m: dict) -> bool:
    """Is `m` the summary block compact_if_needed put in its returned array?

    Recognised by COMPACTION_SUMMARY_HEADER, which only compact_if_needed
    writes. A client could send a system message starting with the same
    words; the guard would then spend injected memory before that client's
    turns, which is the conservative order anyway (hostile pass #3, F5)."""
    content = m.get("content") if isinstance(m, dict) else None
    return (
        m.get("role") == "system"
        and isinstance(content, str)
        and content.startswith(COMPACTION_SUMMARY_HEADER)
    ) if isinstance(m, dict) else False


def _has_sheddable_content(msgs: list[dict], protect_system: int) -> bool:
    """Is there anything left the guard is permitted to remove?

    Exactly two things qualify: a non-system turn that is not the newest one,
    and an injected system block. Trimming is deliberately NOT a third case —
    the trim loop only ever operates on the same blocks the drop loop can
    delete outright, so if nothing is droppable then nothing is trimmable
    either, and 'we could still halve something' can never be the reason to
    keep looping.

    This replaces `len(idxs) <= 1 and trimmed >= 32 and len(sys_idxs) <= 1`,
    which was wrong in both directions. With protect_system >= 2 the last
    clause could never hold, so the guard ran its full six rounds — six exact
    /tokenize calls — over a payload it could not change. With
    protect_system == 1 and a single oversized user turn it did the same,
    because `trimmed` stays 0 when there is nothing to trim."""
    if len([m for m in msgs if m.get("role") != "system"]) > 1:
        return True
    return bool(_droppable_system_indices(msgs, protect_system))


def _enforce_hard_budget(
    messages: list[dict],
    limit: int | None = None,
    protect_system: int = 1,
    report: dict | None = None,
    reserve: int = 0,
) -> list[dict]:
    """Last line of defense: never forward a request that vLLM must reject.

    Everything upstream (compaction, facts, RAG, summary injection) is
    best-effort and can individually fail or overshoot. On 2026-08-13 they
    compounded: summarization 400'd, compaction "degraded" by forwarding the
    ORIGINAL oversized messages, and memory injection then piled 100 facts +
    retrieved exchanges on top — so the user got a hard 400 from the component
    whose whole job is to keep requests inside the window.

    Shedding order is by value: oldest turns first (already summarized, and the
    memory layers exist precisely to carry that content forward), then the
    injected memory blocks, trimmed largest-first — EXCEPT on an array that
    carries INJECTED MEMORY at all (facts, retrieval, or compaction's own
    summary block — P12-5, hostile pass #12; used to be gated on the summary
    block specifically, see that fix's comment at this branch's condition
    below for why that was too narrow): there the turns above the aligned
    recent window are the ones no summary covers, so those are shed first,
    injected memory goes next, and the recent window (and compaction's
    stand-in, if present) go last, only once nothing else is left to spend
    (hostile pass #3, F5; hostile pass #4, F5's "except the turns that go
    anyway"). The newest turn is never dropped — losing the message the user
    just typed is worse than any truncation. After shedding, role alternation
    is REPAIRED (first non-system message must be a user turn) — the first
    cut of this guard could stop mid-pair and hand the Mistral template an
    assistant-first conversation, manufacturing the very 400 it exists to
    prevent (rc6 review).

    Cost discipline (rc6 review): the first cut re-ran the full chat-template
    tokenization of the ENTIRE list once per dropped message — O(N²) blocking
    CPU exactly in the overload scenario. Now: one cheap prescreen, one full
    count, per-message counts for the shedding arithmetic, and a bounded
    number of full-count verification rounds.

    `report`, when passed, is filled in with what this guard decided:
    `limit`, `measured`, `fits`, `counted_by`, and the three shed counts. It
    exists so the request path can tell a vLLM rejection it PREDICTED from one
    that surprised it — see _note_backend_rejection and v3.1 D4. A dict rather
    than a changed return type because every existing caller passes messages
    and gets messages back, and a signature that breaks its callers to carry
    diagnostics is how the same fix gets applied at one site and missed at its
    sibling.

    `reserve`, when passed, is the number of tokens `limit` was already
    shrunk by for something this guard never sees (v3.1.9: the current-time
    line, shaved off `effective_limit` before this call so the line has room
    to be added AFTER the guard, merges and tail repair). It changes nothing
    about what is shed — the guard still tries to fit `limit`, the reduced
    number, because that is what makes room for the line at all — it only
    changes what a FAILURE to fit `limit` is allowed to mean (hostile pass
    #5, reviewer A F3). A payload that clears `limit + reserve` but not
    `limit` is not the failure this guard exists to prevent: vLLM will
    accept it, undated, exactly as sent. Logging that at ERROR ("vLLM will
    most likely reject this") was false on its face and the soak counted it
    as one. Default 0 — every other caller's `limit` already means the real
    limit, and a payload over it by any amount is a real failure.
    """
    if limit is None:
        limit = HARD_INPUT_LIMIT
    # Learned correction (v3.0.5): if vLLM previously reported a true count
    # above what we budgeted, tighten by the observed gap so the same
    # conversation succeeds on its next message.
    if _BUDGET_MARGIN:
        limit = max(256, limit - _BUDGET_MARGIN)

    # Prescreen: skip the (expensive) full tokenization when the char-based
    # estimate is far under the limit.
    #
    # v3.1, measured. All figures below are the DEPLOYED tokenizer
    # (coder3101/Cydonia-24B-v4.3-vision-heretic, per runpod.env.template) —
    # the image's ENV default is a different repo and gives different numbers,
    # which two earlier drafts of this comment mixed together.
    #
    #   this repo's root *.md, 451,819 chars   3.66 chars/tok   char/4 UNDER by 8.4%
    #                            per-file range 3.35 - 4.21
    #   this repo's *.py                       4.28             char/4 OVER by 7.0%
    #   the production chat transcript         4.10             char/4 OVER by 2.5%
    #
    # Corpus sizes are given only where a future reader can reproduce them: the
    # markdown figure is the root-level *.md files at 0b9fbaf. The .py character
    # count from an earlier draft is deleted — it matched no coherent file set,
    # and a number nobody can re-derive is worse than no number. The transcript
    # was measured on the pod and cannot be checked from this repo.
    #
    # The sign flips with content: roughly +-8% either way. Do not restate this
    # as one percentage — three drafts did, and all three were wrong.
    #
    # The margin is not idle for another reason: _fast_token_estimate adds
    # IMAGE_TOKEN_ESTIMATE per image, and that constant (4096) is roughly half
    # the true cost of a Mistral3 vision tile. Images are the one input that
    # can undercount here, and the margin is what absorbs them. Retention=0
    # currently strips every image before this runs, so it is dormant — but
    # raise COMPACTOR_MAX_RETAINED_IMAGES and this margin is what stands
    # between an underestimated photo and a 400. Do not narrow it without
    # fixing COMPACTOR_IMAGE_TOKENS first.
    #
    # v3.1 A14: the divisor was 2, and every figure above justifying it is a
    # chars-per-LOCAL-token measurement — the oracle P0-0c discredited. The
    # right way to state the safety condition is in chars per vLLM token, since
    # vLLM is what charges:
    #
    #   skip is safe when   chars/4 + 4*M  <=  limit/D   and   true <= limit
    #   i.e. roughly when   chars/vLLM-token  >=  4/D
    #
    # so D=2 was betting that no payload ever prices below 2.0 chars per vLLM
    # token. Against the four direct chars-to-/tokenize pairs measured
    # 2026-08-28 (see count_tokens_exact) the worst assistant turn came in at
    # 17,930/8,988 = 1.995 — through the break-even, by 0.25%. Nothing measured
    # crosses once the +4 per message is counted, so this is a thin margin
    # rather than an observed failure; but it is the last place on the request
    # path where an irreversible forward-without-measuring decision is made,
    # and it was making it on the discredited number.
    #
    # D=8 puts the condition at 0.5 chars per vLLM token, which no text can
    # price below (a token is at least one character), so the skip is safe on
    # content rather than on a hope about content. The cost is /tokenize calls
    # for medium payloads the old divisor waved through; the prescreen still
    # exists for the small ones it was written for, which is the overwhelming
    # majority of requests.
    if _fast_token_estimate(messages) <= limit // 8:
        if report is not None:
            report.update(
                {
                    "limit": limit,
                    "measured": None,
                    "fits": True,
                    "counted_by": "the char/4 prescreen (nothing was measured)",
                    "dropped_turns": 0,
                    "trimmed_blocks": 0,
                    "dropped_blocks": 0,
                }
            )
        return messages

    # Ground truth, once, before any decision is made on it. See
    # count_tokens_exact: the local count reads ~50% low on assistant content,
    # which is most of a long conversation, so "total <= limit" was answering a
    # question about a different request than the one about to be sent.
    local_total = count_tokens(messages)
    exact = count_tokens_exact(messages)
    total = exact if exact is not None else local_total
    # How wrong the local tokenizer is on THIS payload. The shedding loop below
    # needs a per-message cost and cannot afford an HTTP call each — that would
    # be one round trip per message on the slowest path in the system. So it
    # keeps local per-message counts and scales them by the ratio the two whole
    # counts just established. Approximate, but approximately right, and every
    # round still verifies against a real /tokenize before it stops.
    scale = (total / local_total) if (exact is not None and local_total > 0) else 1.0
    # v3.1 A9: WHICH counter answered, said out loud, on both branches.
    #
    # This line used to be gated on `exact is not None and |scale-1| > 0.05`,
    # which made the only counter-naming INFO in the package unreachable in
    # precisely the state it would diagnose: when /tokenize refuses, `scale` is
    # forced to exactly 1.0, the shed runs on a tokenizer that reads 34-51% low,
    # the verify step below falls back again unmarked, and the closing
    # "hard budget enforced" line is textually identical in shape to the healthy
    # case — at HTTP 200. That is the 2026-08-28 signature exactly. The two
    # sites that DO name the counter (_sent_token_size) are both gated behind
    # `if r.status_code >= 400` and a shed at 200 reaches neither.
    counter = "vLLM's /tokenize" if exact is not None else "the local tokenizer"
    if exact is not None:
        logger.info(
            f"token scale {scale:.2f}x (local {local_total} -> vLLM {total}); "
            f"counted by {counter}, shedding arithmetic corrected by that factor"
        )
    else:
        logger.warning(
            f"token scale unavailable (/tokenize refused) — budgeting this "
            f"payload at {local_total} tokens from {counter}, UNCORRECTED. That "
            f"counter reads up to 51% low on this model's assistant content, so "
            f"{local_total} is a floor and not a count, and every number in the "
            f"shed line below inherits it."
        )

    def _measure(ms: list[dict]) -> tuple[int, str]:
        """Ground truth for `ms`, and the name of whoever supplied it.

        Both the per-round verify step and the last-resort pass below need
        exactly this, and when they were written out twice the two drifted:
        3d3e732 scaled one recount and missed its sibling, destroying a median
        1,189 characters of memory per divergent payload for no budget
        reason."""
        v = count_tokens_exact(ms)
        if v is not None:
            return v, "vLLM's /tokenize"
        return (
            int(count_tokens(ms) * scale),
            f"the local tokenizer x{scale:.2f}"
            if abs(scale - 1.0) > 0.005
            else "the local tokenizer, UNCORRECTED",
        )

    if total <= limit:
        if report is not None:
            report.update(
                {
                    "limit": limit,
                    "measured": total,
                    "fits": True,
                    "counted_by": counter,
                    "dropped_turns": 0,
                    "trimmed_blocks": 0,
                    "dropped_blocks": 0,
                }
            )
        return messages

    msgs = list(messages)
    # Per-message costs, computed ONCE. Sum-of-parts differs from the templated
    # whole by per-message template overhead, so shedding aims below the limit
    # on arithmetic and then verifies with a real count — bounded rounds.
    per = [int(count_tokens([m]) * scale) for m in msgs]
    running = total
    dropped = 0
    trimmed = 0
    sys_dropped = 0

    for _round in range(6):
        # P11-4 (hostile pass #11): set True below once THIS round's
        # compacted branch has actually acted (shed a turn, trimmed or
        # dropped a block) on ground truth it trusts — see the comment at
        # `_skip_generic`'s assignment for why the floor-less generic
        # stages below must not ALSO run in the same round.
        _skip_generic = False
        # --- an array carrying INJECTED MEMORY: spend memory before any turn
        #     above the protected recent window ---
        #
        # hostile pass #3 (reviewer A F5; pass #2's H5). "Oldest turns first"
        # rests on the oldest turns being already summarized. That is true of
        # an array compaction did not touch, and false of one it did: its
        # stand-in block already carries every turn it removed, and the turns
        # it LEFT — her last message, the reply she is answering, whatever
        # was deferred — are exactly the ones no summary covers. Measured at
        # the shipped numbers with ~10k tokens of injected memory beside an
        # ~11.5k-token stand-in: this loop dropped her previous message and
        # the reply she was answering, then halved the stand-in (its newest
        # half, every L1 scene and the fresh summary, for turns already
        # removed). So when the stand-in is present, the injected blocks
        # around it are trimmed and then dropped FIRST, the stand-in itself
        # untouched; turns, and then the stand-in, only after that.
        #
        # P12-5 (hostile pass #12): THE SAME REASONING APPLIES WHETHER OR NOT
        # THE STAND-IN ITSELF IS PRESENT. The original condition here asked
        # `_is_compaction_standin` specifically, so a DECLINED request —
        # reuse turned itself off (P11-6/P12-1's check, or the plain
        # over-budget path), no stand-in in the array, but facts and
        # retrieval still injected the way they are on every request — fell
        # straight through to the floor-less generic shed loop a few dozen
        # lines down, which has no floor and sheds the OLDEST non-system
        # turn regardless of what it is. On a declined request that turn is
        # not always an old, already-summarized one: once the fresh span
        # itself is deferred (the 4-call cap, or simply a long backlog),
        # `text_only[stored_turns:]` verbatim turns sit ahead of `U_prev`,
        # `A_prev` and `U_new` in the SAME array injected memory sits beside
        # — and the generic loop reaches U_prev/A_prev exactly like it would
        # any other "old" turn, before ever trying the memory it is sitting
        # right next to. Measured on a real branch: every loss this caused
        # (20-23 of 474 positions per hierarchy state) had 2.5-8.5k tokens of
        # headroom left over — dropping memory instead would have kept the
        # exchange every single time. The owner's call, relayed by the
        # coordinator: her own previous exchange — the reply she is
        # answering — is kept over injected memory, on every array that
        # carries memory, not only a compacted one. So the trigger here is
        # simply "is there any injected system content to spend at all" —
        # `_droppable_system_indices` already answers exactly that (system
        # messages beyond what the caller sent), whether or not one of them
        # happens to be compaction's own stand-in. The branch body below was
        # already written generally enough for this: `_spend` (a few dozen
        # lines down) filters OUT the stand-in when computing what memory
        # can cover, so with no stand-in present `_spend` is simply "every
        # injected block", and the branch behaves exactly like the pre-P12-5
        # code did whenever a stand-in incidentally happened to be one of
        # those blocks.
        #
        # EXCEPT THE TURNS THAT GO ANYWAY (hostile pass #4, reviewer A F5).
        # In the cap-refusal state — summarize() refusing the fresh span over
        # the per-request call cap, which hands every refreshed or uncovered
        # turn back verbatim, with or without a stand-in for the OLDER span
        # beside it — the turns left are not "her last message and the reply
        # she is answering" but tens to thousands of old deferred turns, far
        # more than all injected memory together. Spending memory first then
        # bought nothing: measured at the shipped limit, the guard halved and
        # dropped persona, pinned facts and retrieval, and then dropped 100
        # old turns anyway (the pre-F5 order dropped 104 and kept persona and
        # facts). So the old turns that would have to be shed EVEN WITH EVERY
        # SPENDABLE BLOCK DROPPED are shed first, oldest first, never into
        # the last KEEP_RECENT_TURNS; memory is spent only on what is left
        # over. In steady reuse (nothing deferred) that set is empty and the
        # order above is unchanged.
        if _droppable_system_indices(msgs, protect_system):
            # hostile pass #5 (reviewer A F4): the pass-4 F5 loop this replaces
            # stopped the moment cutting ALL injected memory could cover the
            # rest ("running - _memory <= limit"), on the reasoning that
            # anything beyond that point is memory's to pay. But "memory
            # COULD pay it" is not "an old exchange isn't owed first" — the
            # oldest turns in this state are the ones no summary covers
            # (comment above), so they are worth at least as much as facts
            # and retrieval, not less. Stopping at the fractional point meant
            # this loop always left LESS THAN ONE old exchange undropped and
            # let memory absorb that remainder — every time, by construction.
            # Measured at her numbers (facts 400 + retrieval 1,500 tokens, old
            # exchanges 1,800 tokens each): memory was cut or gone on 16 of 40
            # payload sizes and whole on 0, while shedding one more old
            # exchange instead of touching memory would have fit the same
            # limit on 38 of those 16.
            #
            # So now: shed every FULL old exchange this state can spare — down
            # to the protected recent window, and NEVER a lone message — before
            # memory is touched at all; memory is spent only once that floor is
            # reached and the array still does not fit. "Never a lone message"
            # is not cosmetic: the old per-message version could stop having
            # dropped an old USER turn but not yet its reply — exactly the pair
            # the role-alternation repair below would go on to delete anyway,
            # AFTER memory had already been halved and dropped to cover the
            # tokens that orphaned reply cost. Traced at the shipped limit:
            # 1,798 tokens (9% of the window) sat unused because the space the
            # repair freed arrived after the memory spend it would have made
            # unnecessary. Shedding whole exchanges up front leaves the repair
            # nothing to do in this branch.
            # Linear, not once-per-drop (hostile pass #5 review follow-up).
            # A first cut of this loop rebuilt `idxs` (a full scan of `msgs`)
            # AND called `del msgs[i]` / `del per[i]` on every single
            # iteration — and `del` near the front of a list is itself O(n),
            # since everything after the deleted index shifts down. In this
            # state the array can hold thousands of turns with hundreds
            # needing to go (her live chat: ~480 turns today, ~1,900 in the
            # archive, ~470 shed per request in this regime), so that was
            # O(n) work repeated once per dropped exchange — quadratic on
            # the request path, which runs holding the GIL (reviewer pass-3
            # F6 measured 0.46s of GIL-bound CPU in the reuse gate on far
            # less data than this). The fix: scan `msgs` for its non-system
            # turns exactly ONCE, walk a plain integer pointer over that list
            # to decide how much to cut (no list rebuilding, no per-step
            # deletion), and — only once, after the decision is made — build
            # the shortened `msgs`/`per` in a single pass. See
            # test_p5_guard.py's timing section for the measured bound.
            #
            # P11-4 (hostile pass #11): snapshot the three counters THIS
            # branch can move, so the generic stages below can tell
            # whether it actually did anything this round — see
            # `_skip_generic`'s assignment at the end of this branch.
            _cb_before = (dropped, trimmed, sys_dropped)
            _turn_idxs = [i for i, m in enumerate(msgs) if m.get("role") != "system"]
            _floor = max(1, KEEP_RECENT_TURNS)
            # F1 (p7 hostile pass #7): split_messages ALIGNS its kept-recent
            # window to start on a USER turn (leading non-user turns move
            # into the summarized portion — required so the template stays
            # valid; see split_messages's docstring). At KEEP_RECENT_TURNS=4
            # a real request keeps 3 messages, not 4. This floor used to be
            # the raw KEEP_RECENT_TURNS message count, so with any preserved
            # OLD turn (an image, most often) sitting where the 4th-from-end
            # slot would be, this guard counted it as "recent" and protected
            # it from the shed loop above — spending injected memory (halving,
            # then dropping facts/retrieval) to keep a turn split_messages had
            # already decided was old enough to summarize away. If the array
            # still didn't fit, the plain shed loop a few lines down dropped
            # that same turn anyway, so the memory was spent for nothing.
            # Fix: align the floor the same way split_messages aligns its
            # window, so "recent" means the same thing in both places.
            #
            # P10-1 (hostile pass #10): this alignment used to run only
            # `if len(_turn_idxs) >= _floor`, i.e. only when the array held
            # MORE turns than the floor. On a reusing request the array
            # compact_if_needed returns is usually exactly AT the floor —
            # `[U_prev, A_prev, U_new]`, three turns for KEEP_RECENT_TURNS=4
            # — so the gate was false, `_floor` stayed at the raw
            # KEEP_RECENT_TURNS count, and the shed loop below never even
            # started (`_n_turns - _cut > _floor` was false immediately, 3
            # is not > 4). Every one of those turns then read as
            # "protected", the branch went straight to spending injected
            # memory (persona, facts, retrieval — halved six times, then
            # dropped), and STILL did not fit, because `U_prev`/`A_prev` can
            # legitimately run to 16k+ tokens on their own. Control then
            # fell out of this branch into the plain shed loop a few lines
            # down, which has no floor and no pairing, and it dropped
            # `U_prev` then `A_prev` anyway — spending memory for nothing
            # and finishing 6,187-9,213 tokens under the limit with persona,
            # facts and retrieval all gone (measured on her real branch,
            # 24/472 positions, 5.1%). Aligning unconditionally fixes what
            # `_floor` MEANS; the loop below fixes what happens when even
            # that aligned floor is too large to keep whole.
            _aligned_tail = _turn_idxs[-min(_floor, len(_turn_idxs)):]
            while (
                len(_aligned_tail) > 1
                and msgs[_aligned_tail[0]].get("role") != "user"
            ):
                _aligned_tail = _aligned_tail[1:]
            # P8-1 (hostile pass #8): the role check above only strips a
            # WRONG role. OpenWebUI sends an uploaded image as a USER
            # turn (RUNPOD_DEPLOY.md: "OpenAI's standard multimodal
            # format" puts an image part on the user message, never the
            # assistant's), and compact_if_needed inserts its preserved
            # OLD images (`preserved_images`) directly in front of the
            # true keep_recent window — `system + summary_blocks +
            # deferred + preserved_images + keep_recent`. So on a real
            # request the window this floor looks at is
            # [old_image(user), prev-u(user), prev-a(assistant),
            # newest(user)]: it already "starts on a user turn" whether
            # that first entry is the old image or the real first
            # recent turn, so the check above strips nothing and the
            # floor stayed at the raw KEEP_RECENT_TURNS count —
            # protecting the old image from the shed loop below at the
            # cost of injected memory (halved, then dropped) it never
            # needed to spend. split_messages's own keep_recent window
            # always ALTERNATES roles (a real exchange is never two
            # consecutive user turns); an old image sitting in front of
            # it breaks that alternation, so strip the front entry
            # whenever it shares a role with the entry right after it —
            # exactly the case the role-only check above cannot see.
            while (
                len(_aligned_tail) > 1
                and msgs[_aligned_tail[0]].get("role")
                == msgs[_aligned_tail[1]].get("role")
            ):
                _aligned_tail = _aligned_tail[1:]
            _floor = max(1, len(_aligned_tail))
            _n_turns = len(_turn_idxs)
            _cut = 0        # how many of the OLDEST entries of _turn_idxs go
            _freed = 0      # tokens that shedding them frees
            # P10-1: the most memory could ever free from here — every
            # spendable injected block gone entirely (the trim loop below
            # only ever approaches this by halving; the drop loop after it
            # is what reaches it), never the stand-in itself (protected
            # here, dropped only as a true last resort elsewhere in this
            # function). Not computed up front: in the common cap-refusal
            # regime (test_p5_guard.py's sweep and its linear-time fixture:
            # hundreds of deferred exchanges, none of them covered by any
            # summary) the turns-above-floor shed alone already satisfies
            # `limit` before the loop ever reaches the floor, so this stays
            # unpaid — an extra O(n) scan for a number the loop was never
            # going to consult would have widened the linear bound
            # test_p5_guard.py's `[4]` pins for no reason. P11-4 (hostile
            # pass #11): no longer memoized ACROSS iterations once the
            # floor region is reached either — see the comment at its
            # assignment below for why a stale value here is exactly what
            # let memory get spent for nothing on her real branch.
            _mem_ceiling = None
            while _n_turns - _cut > 1:
                if running - _freed <= limit:
                    break  # fits from turns alone already; leave memory whole
                if _n_turns - _cut <= _floor:
                    # P11-4 (hostile pass #11): P10-1's gate here decided
                    # "would spending every spendable block cover the rest?"
                    # by SCALED-ESTIMATE ARITHMETIC (`per[i]`, one flat
                    # `scale` factor applied to a whole-payload local count)
                    # — but `per[i]` prices an image at the flat
                    # IMAGE_TOKEN_ESTIMATE and the round's own exact verify
                    # a few dozen lines down can disagree with that by a
                    # wide margin (images always; her assistant turns
                    # whenever a reply carries decoration the local counter
                    # does not price the way vLLM does). When it does, this
                    # gate can say "memory covers it" and stop cutting, the
                    # trim/drop loops below only reach PART of
                    # `_mem_ceiling` before the exact verify catches the
                    # shortfall a few lines down, and round 2 re-enters this
                    # SAME branch with the memory already spent — this time
                    # correctly finding it cannot cover the (now larger,
                    # because the images were never really that cheap) gap,
                    # and shedding the exchange the spend was supposed to
                    # make unnecessary. Measured (her real branch, position
                    # 696): the arithmetic said the cut fit with 978 tokens
                    # to spare; the exact count was 978 tokens OVER; two
                    # more halvings and a dropped exchange later, the exact
                    # count fit with 10,253 tokens free — memory spent for
                    # nothing, then the exchange lost anyway, P10-1's own
                    # defect in a new shape. Fixed by deciding THIS gate on
                    # the SAME ground truth the round's own verify step uses
                    # whenever one is available: build the CANDIDATE array
                    # this cut would leave (every spendable block gone,
                    # `_cut` old turns gone) and count it exactly — one more
                    # `/tokenize` call on a path that already allows six per
                    # request (the round budget below). Falls back to the
                    # old scaled arithmetic only when `/tokenize` is down
                    # (no exact counter to ask), the same "estimate,
                    # corrected when possible" contract `_measure` already
                    # uses at the end of every round.
                    _spend = [
                        i for i in _droppable_system_indices(msgs, protect_system)
                        if not _is_compaction_standin(msgs[i])
                    ]
                    _mem_ceiling = sum(per[i] for i in _spend)
                    _gone = set(_turn_idxs[:_cut]) | set(_spend)
                    _cand_exact = count_tokens_exact(
                        [m for i, m in enumerate(msgs) if i not in _gone]
                    )
                    _covers = (
                        _cand_exact <= limit if _cand_exact is not None
                        else running - _freed - _mem_ceiling <= limit
                    )
                    if _covers:
                        # P10-1 (hostile pass #10): below this point we would
                        # be cutting into the protected recent window itself
                        # — the turns the floor above exists to keep whole.
                        # That is only ever worth doing when nothing else
                        # can make the array fit: if spending every
                        # spendable injected block down to nothing (the two
                        # loops right after this one, taken to their limit)
                        # would already cover the remaining gap, stop here
                        # and let THEM pay — exactly the "memory goes first"
                        # order the comment above this branch states. Only
                        # when even that maximum spend still cannot cover it
                        # (her 16k-token `A_prev` outweighs persona + facts +
                        # retrieval combined) does shedding the next old
                        # exchange win anything — and it wins the WHOLE
                        # exchange at once, rather than every injected block
                        # halved to nothing first for a saving that shedding
                        # the exchange would have made unnecessary. Measured:
                        # 24/24 of her affected requests fit whole, memory
                        # untouched, once the previous exchange is shed
                        # before memory is spent rather than after.
                        break
                i0 = _turn_idxs[_cut]
                # A whole exchange: the oldest surviving turn, plus its reply
                # if (and only if) that reply immediately follows it in the
                # non-system sequence. Anything else — the reply already
                # gone, or this being an assistant turn already orphaned by
                # an earlier round — is shed alone, since there is no
                # partner left to split it from.
                pair_len = (
                    2
                    if _cut + 1 < _n_turns and msgs[i0].get("role") == "user"
                    and msgs[_turn_idxs[_cut + 1]].get("role") == "assistant"
                    else 1
                )
                if _n_turns - (_cut + pair_len) < 1:
                    break  # never drop the newest turn, whole exchange or not
                _freed += sum(per[_turn_idxs[_cut + k]] for k in range(pair_len))
                _cut += pair_len
            if _cut:
                _drop = set(_turn_idxs[:_cut])
                msgs = [m for i, m in enumerate(msgs) if i not in _drop]
                per = [p for i, p in enumerate(per) if i not in _drop]
                running -= _freed
                dropped += _cut
            while running > limit and trimmed < 32:
                big = [
                    i
                    for i in _droppable_system_indices(msgs, protect_system)
                    if not _is_compaction_standin(msgs[i])
                    and isinstance(msgs[i].get("content"), str)
                    and len(msgs[i]["content"]) > 400
                ]
                if not big:
                    break
                i = max(big, key=lambda j: len(msgs[j]["content"]))
                c = msgs[i]["content"]
                msgs[i] = {
                    **msgs[i],
                    "content": c[: len(c) // 2].rstrip()
                    + "\n[...trimmed to fit the context budget]",
                }
                running -= per[i]
                per[i] = int(count_tokens([msgs[i]]) * scale)
                running += per[i]
                trimmed += 1
            while running > limit:
                spendable = [
                    i for i in _droppable_system_indices(msgs, protect_system)
                    if not _is_compaction_standin(msgs[i])
                ]
                if not spendable:
                    break
                i = spendable[-1]
                running -= per[i]
                del msgs[i]
                del per[i]
                sys_dropped += 1
            # P11-4 (hostile pass #11): this round's compacted branch acted
            # (shed a turn above, or spent memory in the two loops just
            # above) on the EXACT ground truth just established, not the
            # scaled arithmetic the generic stages below still use. Letting
            # those floor-less, exact-verify-blind stages ALSO run in the
            # SAME round is exactly how P10-1's shape survived through
            # 1069da1: round 1's arithmetic said memory covered the gap,
            # spent it, and was wrong by the round's own later exact count;
            # the plain shed loop below — which has no floor and no idea
            # memory was just spent — then dropped the previous exchange
            # anyway on the SAME stale arithmetic. Skip the generic stages
            # this round; the exact verify at the end of the round (a few
            # lines down) judges what THIS branch did first, and a round 2
            # only reaches the generic stages if the branch truly can do no
            # more (the array is already below the floor, or nothing is
            # spendable).
            _skip_generic = (dropped, trimmed, sys_dropped) != _cb_before

        # --- shed oldest non-system turns (arithmetic only) ---
        while running > limit and not _skip_generic:
            idxs = [i for i, m in enumerate(msgs) if m.get("role") != "system"]
            if len(idxs) <= 1:
                break  # always keep the most recent turn
            running -= per[idxs[0]]
            del msgs[idxs[0]]
            del per[idxs[0]]
            dropped += 1

        # --- repair the template invariant broken by mid-pair stops ---
        idxs = [i for i, m in enumerate(msgs) if m.get("role") != "system"]
        while len(idxs) > 1 and msgs[idxs[0]].get("role") != "user":
            running -= per[idxs[0]]
            del msgs[idxs[0]]
            del per[idxs[0]]
            dropped += 1
            idxs = [i for i, m in enumerate(msgs) if m.get("role") != "system"]

        # --- trim the largest injected system block if turns weren't enough ---
        #
        # Same protection as the drop stage below, and for the same reason: the
        # caller's system messages are the first `protect_system` of them, and
        # halving a persona mid-sentence is a quieter version of deleting it.
        # The first cut of this guard applied the protection to the drop loop
        # only, so the caller's prompt was safe from removal and not from
        # mutilation — which is worse, because the model still receives
        # something that looks like instructions.
        while running > limit and trimmed < 32 and not _skip_generic:
            big = [
                i
                for i in _droppable_system_indices(msgs, protect_system)
                if isinstance(msgs[i].get("content"), str)
                and len(msgs[i]["content"]) > 400
            ]
            if not big:
                break
            i = max(big, key=lambda j: len(msgs[j]["content"]))
            c = msgs[i]["content"]
            msgs[i] = {
                **msgs[i],
                "content": c[: len(c) // 2].rstrip()
                + "\n[...trimmed to fit the context budget]",
            }
            running -= per[i]
            # Recount ONLY the trimmed block — and scale it, because `per` and
            # `running` are both in vLLM units. Recounting without `scale` mixes
            # a local estimate into a scaled ledger, so `running` reads lower
            # than the truth, the loop believes it has fit, and the verify step
            # sends it round again to trim content it did not need to trim.
            # Measured over 4,000 payloads: 271 diverged, 267 of them forwarding
            # LESS content, median 1,189 characters of memory destroyed for no
            # budget reason. Introduced by 3d3e732, which scaled line ~923 and
            # missed this one.
            per[i] = int(count_tokens([msgs[i]]) * scale)
            running += per[i]
            trimmed += 1

        # --- last resort: DROP injected system blocks entirely ---
        #
        # v3.1: halving was the only thing this guard could do to a system
        # message, and injected memory IS a system message — so the one layer
        # most able to overshoot was the one it could least touch. Observed
        # 2026-08-27: "28054 -> 26565 tokens (limit 24576); dropped 0 old
        # turn(s), trimmed 5 injected block(s)" — five halvings, still 2k over,
        # forwarded anyway, 400.
        #
        # `protect_system` is how many system messages the CALLER sent, counted
        # before injection. We only drop what we added. The first cut of this
        # stage protected index 0 alone and would delete a caller's SECOND
        # system message once injected memory was exhausted — destroying content
        # the pre-v3.1 code would at worst have halved, in the one case
        # (a single oversized user turn) where dropping it does not achieve the
        # fit anyway. Memory the model cannot receive is worth less than a
        # request that succeeds; the caller's own prompt is not ours to spend.
        while running > limit and not _skip_generic:
            droppable = _droppable_system_indices(msgs, protect_system)
            if not droppable:
                break
            i = droppable[-1]
            running -= per[i]
            del msgs[i]
            del per[i]
            sys_dropped += 1

        # --- verify against GROUND TRUTH; loop only if the arithmetic left us
        #     over (each round does exactly ONE count, and it is vLLM's) ---
        #
        # This is the line that decides whether a request is forwarded, so it
        # is the one that must not be an estimate. It used to be the local
        # count, which is how a payload measured at 21,170 reached vLLM and was
        # charged 32,899 — a request the guard had just certified as fitting.
        #
        # v3.1 A9: and when it falls back it says so. This fallback was
        # unmarked, so the number the shed line reports as the FINAL size —
        # the one that decided to forward — could be either vLLM's count or a
        # local estimate scaled by a factor that is itself 1.0 when /tokenize
        # is down, with nothing in the log to tell the two apart.
        running, counter = _measure(msgs)
        if running <= limit:
            break
        if not _has_sheddable_content(msgs, protect_system):
            break  # nothing left to shed; forward best effort

    # v3.1 D3 — the last thing that happens before a payload the guard has
    # MEASURED as too large goes out the door: spend every remaining scrap of
    # injected memory.
    #
    # The rounds above can exit still over the limit with injected blocks in
    # hand. The round budget is six, and the shedding arithmetic runs on scaled
    # per-message LOCAL counts, so a round can believe it has fit, the exact
    # verify can disagree, and the sixth disagreement is simply the last one.
    # The old code then forwarded anyway, on the reasoning that the newest turn
    # is never dropped.
    #
    # That reasoning is right about TURNS and does not transfer to injected
    # memory. Dropping memory the model will never get to read costs nothing
    # that the 400 does not already cost, and the 400 additionally loses the
    # message the user just typed — no reply, no facts, no episodic write, and
    # nothing retries it. Memory the model cannot receive is worth less than a
    # request that succeeds; the drop stage already makes exactly that trade,
    # and this is the one point where the guard used to decline to make it.
    if running > limit:
        # Drop the MINIMUM that fits — then verify, and keep going if it did not.
        #
        # Two properties, and the first version of this had only one each time.
        # Dropping EVERYTHING is wasteful: measured by review, a payload over by
        # 550 tokens lost persona, facts and summary when one 1500-token block
        # covered it — discarding the priority reasoning _bound_injected_blocks
        # spends twenty lines establishing (persona last, because losing it
        # makes the reply wrong in KIND rather than merely thinner). But
        # dropping the minimum by ARITHMETIC alone is worse: `running` here is
        # scaled per-message estimate, and if the exact count still does not
        # fit, forwarding while holding memory we were allowed to spend is the
        # thing this whole path exists to prevent.
        #
        # So: cheapest-first by the estimate, then measure, then escalate on
        # the measurement until it fits or nothing droppable remains. The extra
        # /tokenize calls are bounded by the number of injected blocks and this
        # is a last-resort path that should almost never run.
        forced = _droppable_system_indices(msgs, protect_system)
        for i in reversed(forced):
            if running <= limit:
                break
            running -= per[i]
            del msgs[i]
            del per[i]
            sys_dropped += 1
        if sys_dropped:
            running, counter = _measure(msgs)
        while running > limit:
            remaining = _droppable_system_indices(msgs, protect_system)
            if not remaining:
                break
            i = remaining[-1]
            del msgs[i]
            del per[i]
            sys_dropped += 1
            running, counter = _measure(msgs)

    # v3.1 A9/A10: the shed line now names the counter behind its numbers and
    # the margin in force. Every number in this line came from one of two
    # counters that disagree by up to 51% in the direction that overflows, and
    # for a week of diagnosis the line said which one: never. `margin` is here
    # for the same reason — `limit` below is already NET of _BUDGET_MARGIN, so
    # a reader comparing it against HARD_INPUT_LIMIT could not see why they
    # differed.
    margin_note = f", margin {_BUDGET_MARGIN}" if _BUDGET_MARGIN else ""
    detail = (
        f"{total} -> {running} tokens (limit {limit}{margin_note}), counted by "
        f"{counter}; dropped {dropped} old "
        f"turn(s), trimmed {trimmed} injected block(s), dropped "
        f"{sys_dropped} injected block(s) entirely"
    )
    if report is not None:
        report.update(
            {
                "limit": limit,
                "measured": running,
                "fits": running <= limit,
                "counted_by": counter,
                "dropped_turns": dropped,
                "trimmed_blocks": trimmed,
                "dropped_blocks": sys_dropped,
            }
        )
    if running > limit:
        if reserve and running <= limit + reserve:
            # hostile pass #5 (reviewer A F3). `limit` here can already be
            # narrower than the real window: the request path shrinks it by
            # `reserve` tokens to leave room for the current-time line, which
            # is added to the payload AFTER this guard returns (see
            # _time_line_for_request / _inject_time_line). A payload that
            # clears the REAL window (limit + reserve) but not this narrowed
            # one is not the failure this guard exists to prevent — vLLM
            # will accept it exactly as sent, just undated. Before this fix,
            # this branch could not tell the two apart: every payload over
            # `limit` (reserve band or genuinely oversized) logged "hard
            # budget FAILED to fit ... vLLM will most likely reject this" at
            # ERROR, and the soak (and any operator) read that as a real
            # failure for a request that was one line short of full and
            # about to be accepted. Silent here on purpose: the caller (it
            # alone knows this is a reserve, not a real limit, and knows the
            # conversation) is the one place that can say it once per
            # conversation instead of once per process — see
            # _time_line_for_request's call site in chat_completions.
            pass
        else:
            # v3.1: this used to log at WARNING and read like a success — "hard
            # budget enforced" while forwarding a payload the guard itself has
            # just measured as too large. It is a failure of the thing whose
            # entire job is to make vLLM's 400 impossible, and the 400 is now
            # the expected outcome. Say so, at ERROR, with the shortfall, so it
            # is findable before the user reports it rather than after.
            #
            # v3.1 D3: and say WHAT is left, because the two residuals need
            # different people to act. On 2026-08-28 the line read "dropped 0
            # old turn(s), trimmed 6 injected block(s), dropped 1 injected
            # block(s) entirely - still 16417 over"; 16,384 + 16,417 = 32,801,
            # which is exactly the number vLLM went on to report, so every one
            # of those 32,801 tokens was the caller's own system prompt and the
            # single turn the user had typed. Nothing the compactor is allowed
            # to touch was still in that payload — and the line said "a
            # conversation with nothing left to shed is the usual cause"
            # without saying which case it was looking at, so it read as a
            # compactor problem for four hours.
            if not _droppable_system_indices(msgs, protect_system):
                residual = (
                    "Nothing injected remains: what is left is the caller's own "
                    "system prompt and the newest turn, and neither is this "
                    "guard's to spend. The request as SENT does not fit the "
                    "window — that is a client-side size problem, not a memory one"
                )
            else:
                # Unreachable: the pass above drops every droppable block
                # before this line can be reached. Kept as a marker, because a
                # guard that gives up holding memory it was allowed to spend is
                # the exact defect v3.1 D3 closed and it should be loud if it
                # returns.
                residual = (
                    "BUG: injected block(s) survived the last-resort drop — the "
                    "guard is holding memory it was allowed to spend"
                )
            logger.error(
                f"hard budget FAILED to fit: {detail} — still "
                f"{running - limit} token(s) over. Forwarding anyway (the newest "
                f"turn is never dropped); vLLM will most likely reject this. "
                f"{residual}."
            )
    else:
        logger.warning(f"hard budget enforced: {detail}")
    return msgs


def _extract_last_user_text(messages: list[dict]) -> str:
    """The user message that prompted the just-completed assistant response,
    for fact extraction. Walks from the end to find the most recent
    role=user message.
    """
    for m in reversed(messages):
        if m.get("role") == "user":
            return _message_text(m)
    return ""


# ---------------------------------------------------------------------------
# V2.0 Phase 2: streaming buffer-and-replay + async tail
# ---------------------------------------------------------------------------

class SseAccumulator:
    """Stateful parser that accumulates `delta.content` text from
    OpenAI-format SSE chunks. Feed it raw bytes as they arrive; call
    .text() after the stream closes to get the full assistant response.

    Robust against:
    - Chunk boundaries not aligned with SSE event boundaries (buffers
      partial events until \\n\\n delimiter)
    - Non-content events (role-only deltas, finish_reason, [DONE])
    - Malformed JSON in a single event (just drops that one event)

    Failures NEVER raise — fact extraction is best-effort downstream.
    """

    def __init__(self) -> None:
        self._buffer: str = ""
        self._parts: list[str] = []
        self._complete: bool = False
        self._truncated: bool = False
        # v3.1.7 (R7/R14): an incremental decoder held for the LIFE of the
        # accumulator, not one decode() per chunk. `r.aiter_raw()` yields
        # chunks at arbitrary TCP-read boundaries that have nothing to do
        # with UTF-8 character boundaries: decoding each chunk independently
        # with errors="replace" turned a multibyte character split across
        # two reads into U+FFFD on BOTH sides of the split — silently, with
        # the stored text differing from what the client actually received
        # (the raw bytes are forwarded unmodified by `yield chunk` above).
        # Found independently by four reviewers; measured 9 of 153 real
        # split points corrupted the text, 0 of 107 ever set holed(). The
        # incremental decoder carries a partial multibyte sequence across
        # the feed() boundary and resolves it once the rest arrives — see
        # finalize() for the case where the stream ends before it does.
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        # v3.1.4: set when this accumulator KNOWS text() has a hole in it.
        # Sticky — a later good chunk cannot un-drop an earlier one — and
        # read by the memory tail (decide_memory_tail), which skips on it
        # unconditionally. v3.1.7: before the incremental decoder above, the
        # only thing that could set this was the `except Exception` in
        # feed(), and errors="replace" made that unreachable in production —
        # the flag guarded an impossible case while the real holes (a split
        # character; a dropped event that carried real content) set
        # nothing. See feed() and finalize() for where it is actually set
        # now.
        self._holed: bool = False

    def feed(self, chunk: bytes) -> None:
        try:
            self._buffer += self._decoder.decode(chunk, final=False)
        except Exception as e:
            # Defensive only, kept for this class's "failures never raise"
            # contract: the incremental decoder above, held with
            # errors="replace", does not raise on malformed or split UTF-8
            # — that is exactly what made this branch unreachable in
            # production (see the __init__ comment) and is what the
            # R7/R14 fix relies on. What remains reachable here is a
            # caller-side contract violation (e.g. `chunk` not being
            # bytes), which is a programming error, not a network
            # condition.
            self._holed = True
            if logsetup.log_once("accumulator.feed.decode"):
                logger.warning(
                    f"stream accumulator dropped a chunk ({type(e).__name__}: "
                    f"{e}); the assistant text for this turn has a hole in "
                    f"it and will not be memorized"
                )
            return
        while "\n\n" in self._buffer:
            event, self._buffer = self._buffer.split("\n\n", 1)
            for line in event.split("\n"):
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if not payload:
                    continue
                if payload == "[DONE]":
                    self._complete = True
                    continue
                try:
                    obj = json.loads(payload)
                    choice = obj.get("choices", [{}])[0]
                    fr = choice.get("finish_reason")
                    if fr:
                        self._complete = True
                        # "length" means vLLM hit the token ceiling, not that
                        # the model finished. The text is a cut-off sentence.
                        if fr == "length":
                            self._truncated = True
                    delta = choice.get("delta", {})
                    content = delta.get("content")
                    if isinstance(content, str) and content:
                        self._parts.append(content)
                except (json.JSONDecodeError, IndexError, KeyError, TypeError):
                    # Single malformed event — drop it, keep accumulating.
                    # v3.1.7 (R7/R14, C's finding at test_sse_accumulator.py:
                    # 97): if the payload we could not parse looked like it
                    # carried reply content, its text is gone from text()
                    # the same way a decode hole is — nothing downstream can
                    # tell the difference — so it sets the same flag.
                    if '"content"' in payload:
                        self._holed = True
                        if logsetup.log_once("accumulator.feed.parse"):
                            logger.warning(
                                "stream accumulator dropped a malformed SSE "
                                "event that appears to have carried reply "
                                "content; the assistant text for this turn "
                                "has a hole in it and will not be memorized"
                            )

    def finalize(self) -> None:
        """Flush the incremental decoder. Call exactly once, after the last
        feed(), before text()/holed() are trusted.

        v3.1.7 (R7/R14): a stream that disconnects mid-character leaves
        undecoded bytes sitting in the decoder that feed() alone never
        resolves — `codecs`' buffered UTF-8 decoder exposes them via
        `.buffer` before the final flush. errors="replace" still means
        `decode(final=True)` returns U+FFFD instead of raising, so the only
        way to know a real gap happened is to check the buffer first.
        """
        incomplete = bool(self._decoder.buffer)
        self._buffer += self._decoder.decode(b"", final=True)
        if incomplete:
            self._holed = True
            if logsetup.log_once("accumulator.finalize.incomplete"):
                logger.warning(
                    "stream accumulator ended with an incomplete UTF-8 "
                    "sequence still buffered; the assistant text for this "
                    "turn has a hole in it and will not be memorized"
                )

    def text(self) -> str:
        return "".join(self._parts)

    def complete(self) -> bool:
        """True once a finish_reason or [DONE] was observed — i.e. the model
        actually FINISHED the reply. A client disconnect mid-stream leaves
        this False, and the async tail must not memorize the half-reply as if
        the model said it (rc6 review: truncated text was being fact-extracted
        and rolled into summaries as a completed assistant turn)."""
        return self._complete

    def truncated(self) -> bool:
        """True when the stream ended with finish_reason "length" — vLLM hit
        the generation ceiling and cut the reply off mid-sentence.

        Distinct from complete(). A truncated reply IS complete in the sense
        that the stream terminated normally, which is why the finish_reason
        check alone was not enough: `if choice.get("finish_reason")` treated
        "length" and "stop" identically, and [DONE] sets _complete regardless,
        so the obvious one-line fix is a no-op. It needs its own flag.

        The consequence of getting this wrong is the same as the disconnect
        case F20 already guards: a half-sentence gets fact-extracted, indexed
        into RAG and rolled into summaries as something the model actually
        said. Worse than the disconnect case, in fact — a truncated reply is
        confidently phrased right up to where it stops."""
        return self._truncated

    def holed(self) -> bool:
        """True when this accumulator KNOWS text() has a gap in it that
        nothing downstream can see: a content-bearing SSE event that failed
        to parse, a caller-side decode error (feed()), or the stream ending
        mid-character (finalize()) — see those methods and the __init__
        comment for why a split-but-eventually-complete character does NOT
        set this (v3.1.7, R7/R14): the incremental decoder resolves that
        case on its own, and flagging every chunk boundary would make this
        fire on nearly every real stream.

        Same doctrine as portability._substantial_reasons, inverted: text we
        KNOW we could not read completely counts as unsafe, never as safe.
        Trimming a holed text to its last sentence would make it worse, not
        better — the sentence boundary is real, the sentence before it may
        be missing its middle — so decide_memory_tail skips on this flag
        before it looks at anything else, including on the clean-finish
        path, where a hole was memorized silently until v3.1.4."""
        return self._holed

    def usable(self) -> bool:
        """Describes the STREAM: the model finished, and it finished because
        it was done rather than because it ran out of room.

        This is NOT the memory decision, and has not been since v3.1.4. It
        was: both call sites gated the memory tail on it, so every reply she
        stopped by hand and every reply vLLM cut at the ceiling was discarded
        from memory whole — 63 exchanges in one log window on 2026-09-01,
        more than half her recent conversation. The memory decision now lives
        in decide_memory_tail, which trims a cut reply to its last complete
        sentence and judges what is left. Gating the tail on this method
        again would reintroduce that loss; it is kept because "did the stream
        finish cleanly" is still a true thing to be able to ask."""
        return self._complete and not self._truncated


def _fire_and_forget(coro, label: str | None = None) -> bool:
    """Spawn post-response background work through the bounded pool
    (V2.3 Theme 3). The pool caps concurrency and sheds beyond a hard
    outstanding ceiling rather than spawning unboundedly under load. Task
    references are kept alive by the pool; exceptions are logged there.

    `label` is what the pool's shed WARNING names when it drops this task.
    A11 added the parameter to bgwork.pool.submit and nothing passed it, so
    the warning could say that a tail was dropped but not WHOSE — which is
    the entire reason the parameter exists. Pass the conversation.
    """
    # Returns whether the pool ACCEPTED it. The caller needs to know: a
    # shed tail is memory that will never be written, and counting it as a
    # store is the silent-loss defect wearing the fix's clothes (F-07).
    return bgwork.pool.submit(coro, label)


def _merge_touched(fresh: list[dict], touched: list[dict]) -> list[dict]:
    """Reconcile a freshly-read facts list with an older in-flight snapshot.

    The request path loads facts and bumps their `last_used` (LRU touch) long
    before the async tail runs, so the tail holds a stale snapshot. Writing
    that snapshot back would erase anything another tail persisted in between
    (a classic lost update — the per-conv lock serializes the *writes*, but the
    *read* happened before the lock was taken).

    So: `fresh` (read under the lock) is authoritative for membership, and the
    snapshot only contributes its LRU touches. Facts the snapshot has but
    `fresh` no longer does were deliberately pruned/forgotten — they stay gone.
    """
    if not touched:
        return list(fresh)
    touched_at = {
        f.get("text"): f.get("last_used")
        for f in touched
        if isinstance(f, dict) and f.get("last_used") is not None
    }
    merged = []
    for f in fresh:
        if not isinstance(f, dict):
            continue
        t = touched_at.get(f.get("text"))
        # Only ever move last_used forward — never backdate a fact that a
        # concurrent request touched more recently than our snapshot did.
        if t is not None and t > (f.get("last_used") or 0):
            f = {**f, "last_used": t}
        merged.append(f)
    return merged


async def _facts_tail(
    conv_id: str,
    touched_facts: list[dict],
    last_user_text: str,
    assistant_text: str,
    turn_index: int,
    *,
    injected_facts: list[dict] | None = None,
) -> None:
    """Job 2 of the memory tail: fact extraction, dedup, prune, save.

    v3.1.7 (R8). Lifted out of _async_tail UNCHANGED, line for line, for one
    reason: it owns two early `return`s, and inside _async_tail those returns
    were returns from the WHOLE tail — so turning fact extraction off, or
    handing the tail an exchange with no user text, also silently cancelled
    job 3, the hierarchical summary rollup. _async_tail's own docstring lists
    the three jobs as independent and says job 1 "runs regardless of facts
    settings"; nothing anywhere claimed job 3 depended on job 2, and the
    dependency was invisible because it was expressed as control flow rather
    than as a condition. A `return` here now ends only this job.

    Not merged into _async_tail as an `if/else`: the branch it would need is
    exactly the shape that let the dependency in, and a reviewer cannot see a
    missing `else` the way they can see a function boundary.
    """
    # HERE, not only at the call site, for the same reason job 3 gives at
    # _rollup_hierarchy: this job WRITES — the touched-save below and
    # save_facts further down — so it is subject to the same pause every
    # other new-memory write is. Job 3 took its own guard when it was
    # extracted in v3.1.8; job 2 was extracted in v3.1.7 (R8) and did not,
    # so the two halves of one split disagreed about whether the outer
    # check was enough. degrade.py's module docstring lists "fact
    # extraction (async tail)" FIRST under what gets gated, which is what
    # made the gap read as covered.
    #
    # WHAT THIS SECOND CHECK CAN AND CANNOT SEE. writes_allowed() caches
    # its reading for COMPACTOR_DEGRADE_CHECK_TTL_S (10 s), so a plain
    # second call here would answer from the SAME statvfs _async_tail's
    # guard took and could not see a disk that filled in between — a
    # cache read here read as covering "job 1 indexes and this job makes a
    # vLLM extraction call that can take seconds", when the TTL it also
    # relied on for cheapness made that exact window invisible. A hostile
    # review caught the contradiction the same day it was written
    # (hostile2-config): the second half nullified the first, in one
    # paragraph.
    #
    # hostile2-config's fix: fresh=True. This call is the one that exists
    # BECAUSE a vLLM extraction call can take seconds after the outer
    # check ran, so it must take a new statvfs rather than trust the
    # outer check's cached one — that is the whole reason for this call to
    # exist rather than relying on _async_tail's outer guard alone. What it
    # ALSO still buys, independent of timing:
    #   * COVERAGE. _facts_tail is a public-shaped coroutine with five
    #     suites entering the tail directly; a future caller that is not
    #     _async_tail gets the pause applied rather than the one that
    #     remembered. That is job 3's argument verbatim.
    #   * The gaps that used to exceed the 10 s TTL anyway even on a cache
    #     read: a tail re-queued behind a pool backlog (bgwork.pool caps
    #     concurrency, so a burst makes this arbitrarily long), and an
    #     extraction plus dedup round trip to a loaded vLLM. `fresh=True`
    #     subsumes both — the cache is irrelevant to a call that always
    #     re-reads.
    #
    # Silent return, matching job 3: guard() already logs at debug and
    # writes_allowed() warned on the transition.
    if not degrade.guard("fact extraction tail", fresh=True):
        return

    if not facts.extraction_enabled():
        # Even with extraction off, save the touched state so LRU
        # tracking persists across restarts. Re-read under the lock (see
        # _merge_touched) so we don't clobber a concurrent tail's writes.
        async with conv_lock(conv_id):
            try:
                merged = _merge_touched(facts.load_facts(conv_id), touched_facts)
                # Nothing to persist means nothing to write (v3.1 G2) — an
                # empty write here creates a facts file for every background
                # utility call the compactor ever sees, and list_known_conv_ids
                # counts them forever.
                if merged:
                    facts.save_facts(conv_id, merged)
            except StoreUnreadable as e:
                logger.error(
                    f"conv={conv_id}: facts file unreadable ({e}); skipped the "
                    f"touched-save rather than writing over it"
                )
            except Exception as e:
                logger.warning(f"conv={conv_id}: touched-save failed: {e}")
        return

    # .strip(), not bare truthiness — the R11 sweep's rule, which this job
    # was carrying the pre-sweep version of. A user turn of nothing but
    # spaces is TRUE, so job 1 next door refused it via
    # _has_pairable_user_text while this one accepted it: it spent a vLLM
    # extraction call on `[user]:    ` / `[assistant]: <reply>` and stored
    # whatever the extractor made of a blank question, against a prompt
    # tuned to over-extract. Same disagreement _has_pairable_user_text was
    # written to end, at the third site the sweep did not reach — R8 lifted
    # this decision out of _async_tail into a function of its own two
    # releases before the sweep unified the spelling, and a moved condition
    # is not what a sweep greps for.
    #
    # Not reachable from /v1/chat/completions today: _tail_store_blocked
    # refuses on the request path first, and counts it as
    # SKIPPED_NO_USER_TEXT. _async_tail is entered directly by six suites
    # and by anything that re-queues a tail, which is the same reachability
    # the helper's own docstring calls "not decoration".
    #
    # `(assistant_text or "")`, matching job 1's own gate a few lines up
    # (hostile2-config: it used to write a bare `assistant_text.strip()`
    # and raise AttributeError on a None reply before control ever reached
    # here, which made THIS tolerance decoration — job 1 now spells the
    # identical check the same way, so a None reply is refused here, on its
    # own terms, rather than never arriving. Both sites now say the same
    # thing for the same reason: decide_memory_tail already rejects a blank
    # reply as SKIPPED_EMPTY, and every direct caller of either job should
    # meet that same rule rather than crash on one spelling and pass on the
    # other.
    if not (assistant_text or "").strip() or not _has_pairable_user_text(
        last_user_text
    ):
        return

    async with conv_lock(conv_id):
        try:
            async with httpx.AsyncClient() as client:
                # BOUNDED, but by the STORE cap — not by the injection cap.
                #
                # This list becomes the extractor's "EXISTING FACTS", and
                # facts._EXTRACTION_SYSTEM_PROMPT says "Do NOT restate facts
                # already in the EXISTING FACTS list below" one line above
                # "When in doubt, extract." It is the ONLY duplicate
                # suppression the extraction call has, against a prompt
                # deliberately tuned to over-extract.
                #
                # v3.1.6 split one knob into two (store 1500 /
                # COMPACTOR_MAX_FACTS_TOKENS, injection 400 /
                # COMPACTOR_INJECT_FACTS_TOKENS) and this call site kept
                # taking the injection-bounded list, so the suppression list
                # shrank with it: measured on a realistic store, 114 facts
                # down to 28 — ~75% of the signal gone. What that buys is
                # byte-identical re-extractions, which cost dedup LLM calls
                # (the exact cost the F1 dedup work was cutting), churn the
                # store, and bring eviction forward. One knob doing two jobs
                # again, at a different seam.
                #
                # Still bounded, because this is a request to vLLM and vLLM
                # has a window: the store cap is what prune_facts already
                # holds the store to, so in normal operation this is the
                # whole store and nothing more, and facts._fit_extraction_input
                # narrows it again against the real extraction budget.
                # No query_text: relevance order is meaningless to a
                # "have I already stored this?" check, and asking for it
                # would spend an embedding call per turn to sort a list
                # whose ORDER nothing reads.
                extraction_facts = facts.select_for_injection(
                    touched_facts, max_tokens=facts._MAX_FACTS_TOKENS
                )
                if injected_facts:
                    # Whatever the model was actually shown is in the list too,
                    # even in the one case the two selections can disagree (a
                    # store over the store cap — which v3.1 F9 now allows to
                    # persist when an archive write fails — where a
                    # relevance-ranked injection can include an LRU-cold fact
                    # the store-cap walk left out). Cheap, and it keeps
                    # "already known" true on both sides of the exchange.
                    seen = {f.get("text") for f in extraction_facts}
                    extraction_facts = extraction_facts + [
                        f for f in injected_facts if f.get("text") not in seen
                    ]
                # conv_id is logging only, and it is what makes a lost
                # extraction attributable to the turn that lost it.
                new_strs = await facts.extract_facts_from_exchange(
                    client,
                    VLLM_URL,
                    MODEL_REPO or "",
                    last_user_text,
                    assistant_text,
                    extraction_facts,
                    conv_id=conv_id,
                )
                from facts import _now_unix
                now = _now_unix()
                new_entries = [
                    {"text": s, "added_turn": turn_index, "last_used": now}
                    for s in new_strs
                ]
                # Re-read INSIDE the lock. `touched_facts` was loaded back in
                # the request path (outside any lock), so building on it would
                # silently drop facts written by a tail that finished in the
                # meantime — the lock serializes writers but cannot prevent a
                # lost update when the read happened before it was acquired.
                combined = _merge_touched(facts.load_facts(conv_id), touched_facts) + new_entries

                # V2.1 Phase 7: hybrid dedup BEFORE pruning. Embedding
                # filter is cheap (no LLM call when no candidate clusters
                # — the common case after a single-fact extraction); LLM
                # verification only runs on actual candidates. Failures
                # degrade to no-op (returns input unchanged) so dedup
                # never affects the user chat path.
                if new_entries and len(combined) >= 2:
                    try:
                        # conv_id scopes dedup's refusal memo and labels its
                        # pass line. Without it every pass re-asks the model
                        # about clusters it has already refused to merge, and
                        # the one dedup line in the log cannot be tied to a
                        # conversation (v3.1 I-6).
                        combined, removed = await dedup.dedup_facts(
                            client, VLLM_URL, MODEL_REPO or "", combined,
                            conv_id=conv_id,
                        )
                        if removed > 0:
                            logger.info(
                                f"conv={conv_id}: dedup merged {removed} "
                                f"duplicate fact(s)"
                            )
                    except Exception as e:
                        logger.warning(
                            f"conv={conv_id}: inline dedup failed (no-op): {e}"
                        )

            # conv_id is what routes an over-budget eviction into the archive
            # sidecar instead of unlinking it (v3.1 F9). This call site is the
            # one that matters: it is on the async tail, so it fires on EVERY
            # exchange. Measured without it on a 300-fact store: dropped=263,
            # archived=0, unrecoverable. That is F9 verbatim — "past
            # COMPACTOR_MAX_FACTS_TOKENS every single turn silently deletes the
            # oldest facts" — and the oldest facts are the conversation's
            # foundational ones.
            kept, dropped = facts.prune_facts(combined, conv_id=conv_id)
            # G2: no early return on empty extraction meant save_facts(conv_id,
            # []) ran on EVERY exchange — the primary generator of the empty
            # facts files D10 counts, and what made the F1a wipe re-fire every
            # turn instead of occasionally. An empty `combined` is nothing to
            # say, not a store to erase; a real prune down to zero still writes.
            if combined:
                facts.save_facts(conv_id, kept)
            if new_entries or dropped:
                # "pruned" read as deleted, and until v3.1 F9 it was: this call
                # site did not pass conv_id, so eviction unlinked rather than
                # archived. It was visible in production as `pruned 16` every
                # turn on a store pinned at the token cap, and nobody could tell
                # from the line that the conversation's oldest facts were gone
                # for good. Say which it is.
                churn = (
                    f", archived {dropped} least-recently-used (recoverable "
                    f"with /list-archive)" if dropped else ""
                )
                logger.info(
                    f"conv={conv_id}: +{len(new_entries)} facts{churn}, "
                    f"total {len(kept)}"
                )
        except StoreUnreadable as e:
            # The re-read above says the file is there and we can't read it,
            # so `combined` is this exchange's facts and nothing else. Writing
            # it is the 2026-08-24 shape: 105 facts atomically replaced by 1,
            # logged as success. Skipping costs this one exchange's facts
            # (v3.1 F1a).
            logger.error(
                f"conv={conv_id}: facts file unreadable ({e}); skipped the "
                f"fact write to avoid overwriting the store with this "
                f"exchange alone"
            )
        except Exception as e:
            logger.exception(f"conv={conv_id}: async fact tail failed: {e}")


async def _async_tail(
    conv_id: str,
    touched_facts: list[dict],
    last_user_text: str,
    assistant_text: str,
    turn_index: int,
    original_messages: list[dict],
    *,
    injected_facts: list[dict] | None = None,
    reply_as_streamed: str | None = None,
) -> None:
    """Post-response work, fired after the assistant's reply is fully
    streamed/received. Three independent jobs:

      1. Episodic indexing (Phase 3): embed this exchange into ChromaDB so
         it's retrievable later. Runs regardless of facts settings.
      2. Facts extraction (Phase 2): pull new persistent facts from the
         exchange, merge + prune + save. Its own coroutine (_facts_tail)
         since v3.1.7 — see R8 there for why that is not cosmetic.
      3. Hierarchical rollup (Phase 4): if enough new turns have accumulated
         since the last summarization, roll L0→L1, L1→L2, L2→L3 as needed.

    INDEPENDENT means independent: none of the three may end another. Job 2
    used to, by returning out of this function, and job 3 was silently off
    for the whole of any deployment running with extraction disabled.

    All degrade to no-ops on failure — never affects the user response.
    Facts and summary writes are serialized per-conv via conv_lock.

    `original_messages` is the request's messages list (pre-compaction); we
    append the just-completed assistant turn before passing to the rollup so
    it sees the full conversation when computing turn ranges.

    `touched_facts` is the WHOLE store as the request path read it, and it is
    what gets merged and written back — the facts left out of this turn's
    working set must keep their real last_used or eviction stops meaning
    anything (v3.1 F9). `injected_facts` is the budget-bounded subset of those
    same dicts that the request path actually put in front of the model.

    What the extractor is handed is neither of those two lists verbatim: it is
    `touched_facts` bounded by the STORE cap, plus anything in
    `injected_facts` that bound left out. The extraction prompt must stay
    bounded — it is a request to vLLM and therefore has a window — but
    bounding it by the 400-token INJECTION budget threw away three quarters of
    the extractor's duplicate-suppression list, which is what the call site
    below explains at length.

    `injected_facts` is keyword-only with a default so no caller is broken by
    its omission; omitting it costs only the union above, because the store
    cap is applied either way.

    `reply_as_streamed` is the reply as the CLIENT received it, passed only
    when it differs from `assistant_text` (a stopped or ceiling-cut reply,
    which decide_memory_tail trims to its last sentence). Facts and the
    episodic index store the trimmed text; the hierarchy reads, and the
    covered-turn record describes, what OpenWebUI keeps and re-sends, which
    is what streamed (hostile pass #3 F1; hostile pass #4 reviewer A F1). See
    _rollup_hierarchy.
    """
    # V2.3 Theme 2: under disk pressure, stop GROWING memory but keep
    # serving. The chat response already went out; this tail is pure
    # persistence, so skipping it entirely is the correct degraded
    # behavior. Explicit user writes (/remember, admin) are gated
    # separately and still allowed.
    if not degrade.guard("async memory tail"):
        logger.info(f"conv={conv_id}: skipped memory tail (disk pressure)")
        return

    # --- 1. Episodic indexing (independent of facts) ---
    # v3.1 D49: this ran outside conv_lock. A prior review called it benign
    # because the upsert is idempotent for a given doc id — true of two tails
    # racing each other, and irrelevant to the case that matters. (That review
    # justified it from _doc_id being (conv_id, turn_index); D1 has since made
    # ids content-addressed, which changes the premise and not the conclusion.)
    # _clear_all_memory holds conv_lock while it calls
    # retrieval.forget_conversation; an unlocked index_exchange lands after
    # that delete and puts the exchange the user just asked to forget back in
    # the vector store, where it is retrievable and injectable again. Its own
    # acquisition rather than one lock over the whole tail: the facts block
    # below holds the lock across a vLLM call, and the summary rollup takes
    # conv_lock internally, so a single enclosing `async with` would either
    # deadlock or stall this behind an LLM round trip.
    # _has_pairable_user_text, not `and last_user_text`: the same rule as
    # _tail_store_blocked's, from the same function, so the outer check on
    # the request path and this inner one cannot disagree about a user turn
    # of nothing but whitespace. See that helper for what the disagreement
    # cost. `assistant_text` is stripped for the same reason on the same
    # line — decide_memory_tail already rejects a blank reply as
    # SKIPPED_EMPTY, and a direct caller should meet the identical rule.
    #
    # hostile2-config: `(assistant_text or "")`, not a bare `.strip()`.
    # _facts_tail (job 2, a few lines below) spells this same check
    # None-tolerantly and says so at length — but this gate runs FIRST and
    # used to raise AttributeError on a None reply before job 2's tolerance
    # could ever be reached, making it decoration rather than a real
    # defence. No production caller passes None today (decide_memory_tail's
    # `decision.text` is guaranteed non-None whenever `decision.store` is
    # true, which is what gates reaching here), so this closes a currently
    # theoretical gap rather than a live one — but it is the same gap job 2
    # was written to close, at the sibling site that actually decides
    # whether it is reachable.
    if (assistant_text or "").strip() and _has_pairable_user_text(last_user_text):
        async with conv_lock(conv_id):
            try:
                indexed = retrieval.index_exchange(
                    conv_id, turn_index, last_user_text, assistant_text
                )
                if indexed:
                    logger.info(f"conv={conv_id}: indexed exchange (turn ~{turn_index})")
            except Exception as e:
                logger.warning(f"conv={conv_id}: episodic indexing failed: {e}")

    # --- 2. Facts extraction ---
    # In its own coroutine since v3.1.7 (R8): its early returns must end
    # fact extraction and NOT the summary rollup below. See _facts_tail.
    await _facts_tail(
        conv_id,
        touched_facts,
        last_user_text,
        assistant_text,
        turn_index,
        injected_facts=injected_facts,
    )

    # --- 3. Hierarchical summary rollup (Phase 4) ---
    # Runs OUTSIDE the facts lock since maybe_rollup acquires its own
    # conv_lock internally — nesting the same lock would deadlock.
    # .strip(), matching the episodic gate above (R11 sweep). A reply of
    # whitespace is not a turn to roll up: it would advance the watermark
    # over a turn that says nothing, and the label would then cover text no
    # summary can account for.
    # ONE function, both tail paths. Extracted in v3.1.8 rather than
    # copied: the skip path below needs exactly this, and a second copy of
    # it is the fix-one-site-miss-the-sibling defect this file has paid
    # for eighteen times.
    await _rollup_hierarchy(
        conv_id, original_messages, assistant_text,
        reply_as_streamed=reply_as_streamed,
    )


async def _rollup_hierarchy(
    conv_id: str,
    messages: list[dict],
    assistant_text: str | None,
    *,
    reply_as_streamed: str | None = None,
) -> None:
    """Advance the hierarchical summary. Both tail paths call this.

    `assistant_text` is the reply to roll up WITH the history, or None to
    roll up the history alone — which is what the skipped-tail path passes.

    `reply_as_streamed` is that reply as the client received it, when it
    differs (a stopped reply trimmed for memory). The covered-turn record
    describes the streamed text, because that is what every later request
    carries (hostile pass #3, F1 — see summarizer._record_chunk_fps), and
    since hostile pass #4 (reviewer A F1) the chunk READS it too: the reply
    is appended as streamed, through the same redaction every history turn
    gets. Before, the chunk summarized `assistant_text` — the memory-trimmed
    prefix — while the record blessed the full reply, so a Stopped or
    ceiling-cut reply that CLOSED an L1 chunk lost everything after its last
    sentence boundary (a trailing list, notes after a code block) from every
    later request, in no layer at all. A cut reply that does not close a
    chunk was already read in full by a later chunk, from the next request;
    this makes the closing position read what the other nineteen do. The
    trim exists for the fact store and the episodic index, which still get
    `assistant_text`; a runaway loop still never reaches a summary, because
    the redaction judges the streamed text (its clean head, or the
    placeholder).

    WHY None IS A CASE AT ALL (v3.1.8). A reply that trips
    reply_is_degenerate must not enter memory: the fact extractor would
    store its markup and the episodic index would embed a repetition loop.
    But `_run_memory_tail` expressed that by returning before the WHOLE
    tail, and the rollup is not about this reply — it summarizes turns
    already in the history, and `_redact_degenerate_turns` below is how it
    handles degenerate ones. Coupling the two meant a model that loops for
    n turns froze the hierarchy for n turns, with no floor and no recovery:
    the soak measured 14 consecutive skips in a 22-turn run, and the
    watermark never left 0. That is the frozen hierarchy this release
    exists to fix, reached by a third route.

    So the degenerate reply is excluded from the rollup INPUT while the
    rollup itself still runs. Nothing about it reaches a summary; the turns
    around it stop being held hostage to it.
    """
    if not summarizer.enabled():
        return
    # HERE, not at the call site. A rollup WRITES state, so it is subject
    # to the same pause every other new-memory write is, and putting the
    # check inside means both callers get it rather than the one that
    # remembered.
    #
    # hostile2-config: fresh=True, not a cache read. This call exists to
    # catch disk pressure that develops WHILE the summarization LLM call
    # this function is about to make is in flight — the outer check
    # (_tail_store_blocked, on the request path) already ran, and a
    # summarization round trip is exactly the multi-second gap the
    # COMPACTOR_DEGRADE_CHECK_TTL_S cache (10s default) would otherwise
    # paper over. A fresh statvfs here costs one syscall per rollup, not
    # per request.
    if not degrade.guard("hierarchy rollup", fresh=True):
        return
    # A reply of whitespace is not a turn to roll up: it would advance the
    # watermark over a turn that says nothing, and the label would then
    # cover text no summary can account for. None is not whitespace - it is
    # 'do not append a reply at all', which is a different instruction.
    #
    # OPEN_ISSUES2 LOW re-checked: reported as unreachable from EITHER
    # PRODUCTION call site (_async_tail's job 3, which only ever receives
    # decide_memory_tail's already-non-blank decision.text; and the
    # skipped-tail path, which passes None). Both of those still hold. But
    # this function is a "public-shaped coroutine" in this file's own
    # words a few lines up about its sibling _facts_tail, entered DIRECTLY
    # by test suites (and by extension anything else that re-queues a
    # tail) with arbitrary text that never passed through decide_memory_
    # tail's own `if not text.strip()` gate — test_truncated_tail.py's
    # `tt-inner-empty-reply` case does exactly this, calling _async_tail
    # with assistant_text="" directly and asserting NO rollup happens.
    # Removing this guard as "dead" broke that real, already-shipped
    # coverage the moment it was tried — kept, and the "unreachable from
    # either call site" reading corrected to name what it was actually
    # checked against.
    if assistant_text is not None and not assistant_text.strip():
        return
    try:
        # v3.1.3: redact past degenerate turns before they can reach
        # maybe_rollup - see _redact_degenerate_turns for why the call-site
        # skip is not enough on its own for this job.
        #
        # run_in_threadpool, not a bare call: this walks EVERY historical
        # assistant turn through reply_is_degenerate, and this is a
        # coroutine, so a bare call blocks the event loop for every other
        # request. Measured against her real replies (median 5,248 chars):
        # 20 turns 4.6ms, 40 turns 9.5ms, 85 turns 65ms, 170 turns 446ms -
        # and it runs on every single turn. The detector blocking this same
        # loop is a defect this branch has already shipped once.
        #
        # hostile pass #4 (reviewer A F1): a reply that was cut is appended
        # AS STREAMED, redacted in the same pass as the history — the text
        # the record will describe, so a chunk closing on it reads its tail.
        # See this function's docstring. Should the redaction leave nothing
        # but the placeholder, the reply memory itself kept (`assistant_text`,
        # already judged clean) is what the chunk reads instead.
        _streamed_differs = (
            assistant_text is not None
            and reply_as_streamed is not None
            and reply_as_streamed != assistant_text
        )
        _to_redact = list(messages) + (
            [{"role": "assistant", "content": reply_as_streamed}]
            if _streamed_differs else []
        )
        _redacted = await run_in_threadpool(_redact_degenerate_turns, _to_redact)
        _reply_text = assistant_text
        if _streamed_differs:
            _reply_text = _message_text(_redacted.pop())
            if _reply_text == _DEGENERATE_HISTORY_PLACEHOLDER:
                _reply_text = assistant_text
        full_messages = _redacted + (
            [{"role": "assistant", "content": _reply_text}]
            if assistant_text is not None
            else []
        )
        # OFF THE EVENT LOOP (v3.1.9.2), same reasoning as the two reads
        # inside maybe_rollup. This one is purely a "did anything change"
        # snapshot for the log line below, and it runs on every turn the tail
        # runs — a blocking disk read on the loop to decide whether to print.
        before = await run_in_threadpool(summarizer.load_state, conv_id)
        # The covered-turn record is written by each chunk, in the same
        # call, from what the client SENT plus the reply as the client
        # RECEIVED it — not from full_messages, whose redaction and trimmed
        # reply no request carries. Passed only when it differs, so a caller
        # (or test double) of maybe_rollup that predates the kwarg is
        # unaffected on every reply that was not cut.
        _rollup_kwargs: dict = {"raw_messages": list(messages)}
        if _streamed_differs:
            _rollup_kwargs["reply_as_streamed"] = reply_as_streamed
        # v3.1.9 (tail catch-up). The CONTEXT-MANAGER form
        # (summarizer.vllm_call_budget_ctx), not the `vllm_call_budget=`
        # keyword — same reason admin_compact already uses it (see that
        # function's own comment on this exact point): `summarizer.
        # maybe_rollup` is monkeypatched WHOLESALE, with a fixed signature
        # that predates this feature, by test doubles this file does not
        # own (test_degenerate_skip.py's `spy_maybe_rollup` is one; there
        # are others — see the block comment above `_vllm_call_budget` in
        # summarizer.py for the enumerated list). A `vllm_call_budget=`
        # keyword on THIS call breaks every one of them with a TypeError,
        # swallowed by this function's own `except Exception` below, so the
        # spy is never entered and the test reads as "maybe_rollup was never
        # invoked" — reproduced against the unfixed shape of this line.
        # The context-manager form sets a contextvar around the call
        # instead, so the call itself keeps today's exact signature; a stub
        # that replaces `maybe_rollup` wholesale never reads the contextvar
        # either, which is exactly correct — a stub making no real vLLM
        # calls has nothing to bound.
        #
        # A FRESH budget every tail — the whole point is bounded work PER
        # TURN, so nothing here carries unspent calls forward (a light turn
        # does not bank them) or borrows against a future one (an overshot
        # turn does not shrink the next turn's budget). See
        # TAIL_ROLLUP_MAX_CALLS' own comment for what "bounded" means here
        # and why it is safe from the livelock a strict per-call bound would
        # have caused.
        with summarizer.vllm_call_budget_ctx(TAIL_ROLLUP_MAX_CALLS) as _budget:
            state = await summarizer.maybe_rollup(
                conv_id, full_messages, VLLM_URL, MODEL_REPO or "",
                **_rollup_kwargs,
            )
        if (
            len(state.get("l1") or []) != len(before.get("l1") or [])
            or len(state.get("l2") or []) != len(before.get("l2") or [])
            or (state.get("l3") is not None) != (before.get("l3") is not None)
        ):
            logger.info(
                f"conv={conv_id}: rollup → L1={len(state.get('l1') or [])} "
                f"L2={len(state.get('l2') or [])} "
                f"L3={'y' if state.get('l3') else 'n'} "
                f"last_turn={state.get('last_summarized_turn', 0)}"
            )
        # v3.1.9 (tail catch-up, hostile follow-up). Own try/except, not just
        # the one already wrapping this whole function: `state` above is
        # already SAVED by the time execution reaches here (maybe_rollup
        # persists before returning), so a failure computing or logging the
        # catch-up line must never be reported as "async rollup failed" —
        # that phrase means the rollup itself did not complete, and it did.
        # It also must never stop record_catchup_pass from running: that
        # write is what makes the NEXT poll's converging/stuck verdict
        # correct, and a tail that skips it on a formatting fluke would
        # quietly go back to the poll-cadence-dependent flapping this
        # follow-up exists to fix.
        #
        # DEFENSIVE, not decorative: maybe_rollup's contract promises
        # `turns_seen`/`last_summarized_turn` are always present ints on any
        # state it returns, but a caller relying on that promise is exactly
        # how a future change three call-levels away turns into a crashed
        # tail here — isinstance-checked rather than trusted, so a state
        # shaped unexpectedly degrades this diagnostic instead of raising it
        # into the reply she is waiting for.
        try:
            _turns_seen = state.get("turns_seen", 0)
            if not isinstance(_turns_seen, int):
                _turns_seen = 0
            _before_wm = before.get("last_summarized_turn", 0)
            if not isinstance(_before_wm, int):
                _before_wm = 0
            _after_wm = state.get("last_summarized_turn", 0)
            if not isinstance(_after_wm, int):
                _after_wm = 0
            _work_due = summarizer.needs_rollup(state, _turns_seen)
            # Recorded EVERY pass (not only while behind): this is what lets
            # health.py tell "converging" from "stuck" without depending on
            # how often it happens to poll — see summarizer.
            # record_catchup_pass's own docstring.
            summarizer.record_catchup_pass(conv_id, _before_wm, _after_wm, _work_due)
            if _work_due:
                # Visible progress while the hierarchy is behind by more
                # than one bounded pass can clear. Gated on needs_rollup
                # AFTER this call, which is true only when a tier is STILL
                # due — an ordinary turn (at most one L1 chunk due,
                # comfortably inside budget) clears it and never prints this
                # line; a hierarchy days behind a vLLM outage prints it
                # every turn until it doesn't. `_advanced` is measured
                # against `before` (loaded above, prior to this call), not
                # estimated, so a turn that spent calls without moving the
                # watermark (every batch this pass touched came back empty,
                # or the material due was skipped as blank) says so instead
                # of reporting a bogus ETA — the "does not hide a stuck
                # catch-up" half of this feature's requirement; the gate
                # above is the "does not alarm on an ordinary turn" half.
                #
                # hostile pass #5 (C5-7/E8): the OLD line described ONLY
                # L1's watermark ("N turn(s) still uncovered ... ~K more
                # turn(s)"), even on a turn where `_work_due` is True
                # because an L2 fold or the L3 refresh is left pending
                # while L1 itself is fully current — the drain's own L3 >
                # L2 > L1 priority (see the loop above) means the budget
                # can run out on exactly that boundary. The old line then
                # printed "0 turn(s) still uncovered ... ~0 more turn(s)":
                # a catch-up "in progress" with nothing to report and a
                # bogus zero ETA, on every such turn (measured 3 times in
                # SP\final-soak.log's own run). `rollup_due_tiers` names
                # EVERY tier actually pending, not just L1's turn count.
                _due = summarizer.rollup_due_tiers(state, _turns_seen)
                _turns_behind = max(0, _turns_seen - _after_wm) if _due["l1"] else 0
                _pending = []
                if _due["l1"]:
                    _pending.append(f"L1 {_turns_behind} turn(s) still uncovered")
                if _due["l2"]:
                    _pending.append("an L2 fold pending")
                if _due["l3"]:
                    _pending.append("an L3 refresh pending")
                # `_work_due` (summarizer.needs_rollup, computed a moment
                # ago from this same `state`) already means at least one of
                # the three is True — `rollup_due_tiers` delegates to the
                # identical per-tier gates needs_rollup ORs together (see
                # its own docstring), so `_pending` cannot really be empty
                # here. The fallback exists only so a future drift between
                # the two checks degrades to a vague line instead of a
                # crash on the request path.
                _pending_desc = ", ".join(_pending) if _pending else "a tier pending"
                _calls_spent = TAIL_ROLLUP_MAX_CALLS - _budget["remaining"]
                _advanced = _after_wm - _before_wm
                if _due["l1"] and _advanced > 0:
                    _eta = (
                        f"~{-(-_turns_behind // _advanced)} more turn(s) at "
                        f"this turn's rate"
                    )
                elif _advanced > 0:
                    # L1 itself is current; what is left is an L2 fold
                    # and/or the L3 refresh, neither counted in turns, so
                    # there is no turn-count ETA to give — it runs on the
                    # next pass with budget left for it.
                    _eta = (
                        "no turn-count ETA — L1 is current, waiting on the "
                        "fold/refresh above"
                    )
                else:
                    _eta = (
                        "no turns advanced this pass — see the rollup log "
                        "line above, or the absence of one, for why"
                    )
                logger.info(
                    f"conv={conv_id}: hierarchy catch-up in progress — "
                    f"{_pending_desc}, {_calls_spent} vLLM call(s) spent "
                    f"this turn (budget {TAIL_ROLLUP_MAX_CALLS}), {_eta}"
                )
        except Exception as e:
            logger.warning(
                f"conv={conv_id}: could not compute/log hierarchy catch-up "
                f"progress ({type(e).__name__}: {e}) — the rollup pass "
                f"above already completed and its state is already saved; "
                f"this is a diagnostic failure only"
            )
    except Exception as e:
        logger.exception(f"conv={conv_id}: async rollup failed: {e}")


def _has_pairable_user_text(last_user_text: str) -> bool:
    """Is there user text substantive enough to pair a reply with?

    v3.1.7 (R11 sweep). ONE rule, because it was two. `_tail_store_blocked`
    below asked `not (last_user_text or "").strip()` and refused the whole
    tail; `_async_tail`'s episodic gate asked the bare `and last_user_text`
    three hundred lines away and let it through. A user turn of nothing but
    spaces is TRUE, so the two siblings disagreed about exactly one shape —
    and the one that let it through is the one that writes to the store. It
    indexed `[user]:    \\n[assistant]: <her reply>` as a real exchange:
    retrievable, injectable, and rebuilt as a real turn by /admin/compact,
    while the request path had already published the outcome as a skip.

    Not reachable from /v1/chat/completions today — R8 hoisted the decision
    to the request path, which refuses first — which is precisely why it
    survived a mutation sweep of every endpoint test in test_truncated_tail.
    _async_tail is entered directly by five suites and by anything that
    re-queues a tail, so the inner guard is not decoration and must say the
    same thing as the outer one.
    """
    return bool((last_user_text or "").strip())


def _tail_store_blocked(last_user_text: str) -> tuple[str, str] | None:
    """Why the memory tail would store NOTHING for this exchange, or None.

    v3.1.7 (R8). These are the two conditions _async_tail evaluates that end
    in no episodic row, no fact, and no rollup — the whole tail a no-op. They
    are read here, on the request path, so the decision they force is COUNTED
    and LOGGED like any other skip instead of being taken silently after
    `stored` had already been published. Returns (outcome, reason) in the
    shape TailDecision wants.

    Order matters only in that disk pressure is the operator-visible one:
    when both apply, the operator needs to see the disk.

    `degrade.guard` is therefore called twice per exchange — once here and
    once inside _async_tail, which keeps its own guard because a tail can sit
    in the pool's queue while the disk fills under it. writes_allowed() is
    cached for COMPACTOR_DEGRADE_CHECK_TTL_S (10 s), so the second call is a
    tuple read, not a second statvfs; only its debug line repeats.
    """
    if not degrade.guard("async memory tail"):
        return (
            tailhealth.SKIPPED_DISK_PRESSURE,
            "disk pressure has paused new-memory writes, so nothing about "
            "this exchange would be persisted",
        )
    if not _has_pairable_user_text(last_user_text):
        # Reachable, and not only through a malformed request: a user turn
        # whose content is a parts LIST carrying no text field and no part
        # _message_image_count recognises falls through _extract_last_user_text
        # and then through _memorable_user_text, which only substitutes a
        # marker when it can count images. An image-only upload does NOT land
        # here — that is exactly what the marker covers. Before this, the tail
        # took a bare `return` with no log line of any kind and the counter
        # said `stored`.
        return (
            tailhealth.SKIPPED_NO_USER_TEXT,
            "the exchange has no user text to pair the reply with, so "
            "episodic indexing and fact extraction both refuse it",
        )
    return None


def _run_memory_tail(
    conv_id: str,
    text: str,
    *,
    finished: bool,
    truncated: bool,
    holed: bool,
    touched_facts: list[dict],
    last_user_text: str,
    turn_index: int,
    messages: list[dict],
    injected_facts: list[dict] | None,
) -> TailDecision:
    """Decide, count, log, and (maybe) fire the memory tail — for BOTH
    /v1/chat/completions call sites, so that no line of tail policy or
    bookkeeping exists at one site and not its twin. The sites reduce to
    argument-passing: the streaming one hands in the accumulator's three
    flags, the non-streaming one `finished=True` and finish_reason.

    Returns the decision so a caller (or a test) can see what was done.
    """
    decision = decide_memory_tail(
        text, finished=finished, truncated=truncated, holed=holed
    )
    # v3.1.7 (R8). decide_memory_tail judges the REPLY; it cannot know whether
    # the store is reachable. Two conditions inside _async_tail store nothing
    # at all, and both were reached AFTER `stored` had been counted, so
    # /health/full reported a healthy tail for an exchange that never got
    # near memory — the silent-skip class this counter exists to end, one
    # layer up from where it was closed.
    #
    # Evaluated HERE rather than returned from _async_tail and recorded by the
    # pool. Three reasons, and the first is decisive:
    #
    #   * bgwork.pool SHEDS. A tail dropped at the ceiling would then never be
    #     counted at ALL, which is a new silent skip of exactly the shape
    #     being fixed — and shedding is not hypothetical here (R13's own
    #     evidence counts "the ones the pool shed").
    #   * /health/full is read on a 30 s probe. A count that lands whenever a
    #     background coroutine happens to finish describes a different window
    #     than the one it is published in.
    #   * one note() per exchange, on the request path, at a deterministic
    #     point, keeps test_saturation.py's ledger (stored + skipped ==
    #     exchanges) exact rather than eventually-exact.
    #
    # Only conditions under which NOTHING is stored are hoisted. Extraction
    # being disabled is not one: episodic indexing still runs, and the rollup
    # it used to skip is fixed in _async_tail itself rather than counted as a
    # loss here.
    if decision.store:
        _blocked = _tail_store_blocked(last_user_text)
        if _blocked is not None:
            _outcome, _why = _blocked
            decision = TailDecision(False, "", _outcome, _why, decision.raw_chars)
    # v3.1.8 (N4b). Hoisted here for R8's reason and checked LAST among the
    # pre-count conditions: the ones above are about whether the store can be
    # reached, this is about whether this request deserves to reach it, and a
    # request that could not have been stored anyway should keep the label
    # that says why.
    # ONE read, both consumers. This reads the store, and a task-traffic
    # turn consulted it twice: once here and again at the rollup gate below,
    # because this branch REPLACES `decision` and so does not exclude it.
    #
    # Computed UNCONDITIONALLY, and that is the load-bearing part. Writing
    # `decision.store and _is_repeat_task_traffic(...)` would make this
    # False for every already-refused reply, which is precisely the class
    # the gate below has to recognise - it is F1, restored, in the shape of
    # a tidy-up. (_has_conversational_history short-circuits before the disk
    # read, so an ongoing conversation pays nothing for the extra call.)
    _task_traffic = _is_repeat_task_traffic(conv_id, messages)
    if decision.store and _task_traffic:
        decision = TailDecision(
            False,
            "",
            tailhealth.SKIPPED_TASK_TRAFFIC,
            "this is OpenWebUI background task traffic (no assistant turn, on "
            "a conv_id already in the store), not an exchange to remember",
            decision.raw_chars,
        )
    # Counted before it is logged, and before the tail is fired: the counter
    # is what /health/full reads, and a skip that is only a log line is the
    # defect this exists to close (63 exchanges in one 2026-09-01 window,
    # weeks unnoticed). tailhealth returns the streak for THIS line rather
    # than logging it under its own logger — see its module docstring.
    streak = tailhealth.note(
        decision.outcome,
        raw_chars=decision.raw_chars,
        kept_chars=len(decision.text) if decision.store else 0,
    )
    if not decision.store:
        # WARNING, not INFO. If a client sends a max_tokens below the
        # model's usual reply length, EVERY reply finishes as "length" and
        # this branch runs on all of them — that was the 2026-08-28 shape
        # exactly: correct local behaviour, no error, and the user
        # experiencing an assistant that had stopped remembering. The skip
        # is right; being quiet about it is not. "skipping memory tail" is
        # the phrase the pod is grepped for; keep it.
        logger.warning(
            f"conv={conv_id}: {decision.reason} — skipping memory tail "
            f"({decision.raw_chars} chars accumulated; {streak})"
        )
        # THE REPLY IS SKIPPED; THE HIERARCHY IS NOT (v3.1.8).
        #
        # Everything above is about not letting THIS reply into memory,
        # and that is right. The rollup is a different question: it
        # summarizes turns that are already in the history, and it redacts
        # degenerate ones itself. Returning here skipped it too, so a model
        # that loops froze the watermark for as long as the loop lasted -
        # 14 consecutive skips in a 22-turn soak, watermark still 0 - and
        # nothing recovered it afterwards, because the rollup is only ever
        # driven from the tail. The reply is passed as None so it is
        # excluded from the input rather than summarized.
        #
        # raw_chars > 0 is the discriminator, and it is not a proxy for the
        # outcome label. It asks whether the MODEL PRODUCED ANYTHING. A
        # backend rejection produces no reply, adds no turn to roll up, and
        # needs the same backend the rollup would call - so during a vLLM
        # outage every 400 would fire a summarization against the process
        # that is already failing. A repetition loop is the opposite: 1,412
        # characters arrived, the conversation moved, and only the reply is
        # unfit to store.
        if (
            decision.raw_chars > 0
            # NOTHING TO ROLL UP WITHOUT A HISTORY (v3.1.8.1). Found by the
            # R8 integration tests, which post a SINGLE user message and
            # assert a skipped tail leaves the store untouched. The rollup
            # fired anyway: it cannot build a chunk from one message, so its
            # only effect was writing turns_seen=1 for a conversation that
            # stored nothing - cost with no benefit, and adversarial finding
            # F5 (summary state for conversations that stored nothing).
            #
            # This is not the test being bent to fit the code. The feature
            # exists so a LOOPING model stops freezing the hierarchy, and
            # those arrays always carry prior assistant turns, so that case
            # is untouched. What this declines is the one where there is no
            # earlier exchange to summarize at all.
            #
            # DO NOT DELETE THIS ON THE STRENGTH OF THE DOCSTRING ON
            # _is_repeat_task_traffic. That docstring argues against calling
            # _has_conversational_history "at the tail site", and it is
            # right about the site it means: the STORE decision, where a
            # history check would silently drop the opening exchange of
            # every new conversation. This is not that site. Nothing is
            # stored here - the reply was already refused - and the only
            # question left is whether an earlier exchange exists to
            # summarize. Delete it and R8 comes straight back.
            #
            # This gate also used to carry `and not _task_traffic`, which
            # was itself the fix for a label read that could never fire
            # (both labels it tested are set only behind `if decision.store`
            # above). The history check subsumed it:
            # _is_repeat_task_traffic opens with
            # `if _has_conversational_history(messages): return False`, so
            # the conjunct was only ever reached once it was already
            # guaranteed True - dead, exactly like the label read it
            # replaced. Two of this gate's defects have now been a condition
            # that could not fire, so the dead one is removed rather than
            # left as decoration. _task_traffic is still computed
            # unconditionally above, and the store branch still uses it.
            and _has_conversational_history(messages)
            and not _fire_and_forget(
                _rollup_hierarchy(conv_id, messages, None),
                label=f"rollup conv={conv_id}",
            )
        ):
            logger.warning(
                f"conv={conv_id}: the background pool also shed the "
                f"hierarchy rollup for this turn; the watermark does not "
                f"advance until a later turn is accepted"
            )
        return decision
    if decision.reason:
        # A trimmed store. INFO: it is the fix working, not a fault — but
        # say what was cut, so `grep "stream ended"` still finds every
        # stopped reply after the upgrade and can see what became of it.
        logger.info(
            f"conv={conv_id}: {decision.reason}; memorizing the "
            f"{len(decision.text)} of {decision.raw_chars} chars that end "
            f"on a sentence boundary"
        )
    # hostile pass #3 (F1): the covered-turn record must describe the reply
    # as the client RECEIVED it — `text`, the accumulator's whole stream or
    # the non-stream body — not `decision.text`, which is trimmed for memory.
    # hostile pass #4 (reviewer A F1): and the hierarchy reads that same
    # text, so a chunk that closes on a cut reply summarizes its tail.
    # Passed only when the two differ (a trimmed store), so test doubles of
    # _async_tail that predate the kwarg keep working on every other reply.
    _tail_kwargs: dict = {"injected_facts": injected_facts}
    if text != decision.text:
        _tail_kwargs["reply_as_streamed"] = text
    accepted = _fire_and_forget(
        _async_tail(
            conv_id,
            touched_facts,
            last_user_text,
            decision.text,
            turn_index,
            messages,  # original request messages, for rollup
            **_tail_kwargs,
        ),
        label=f"tail conv={conv_id}",
    )
    if not accepted:
        # v3.1.8 (F-07). The pool shed it, so nothing will be written. Say
        # so, under its own label, and correct the store we just counted.
        #
        # Safe to amend after the fact because submit() cannot run the
        # coroutine before returning and there is no await between the
        # note above and this line — the tail cannot have completed in
        # between, so no reader can have seen the optimistic count.
        tailhealth.note_correction(
            tailhealth.STORED if not decision.reason else tailhealth.STORED_TRIMMED,
            tailhealth.SKIPPED_SHED,
        )
        logger.warning(
            f"conv={conv_id}: the background pool shed this memory tail at "
            f"its outstanding ceiling, so the {len(decision.text)} chars "
            f"this exchange would have stored are lost. Counted as "
            f"{tailhealth.SKIPPED_SHED}; see background_work.shed"
        )
        return TailDecision(
            False, "", tailhealth.SKIPPED_SHED,
            "the background pool shed this tail at its outstanding ceiling",
            decision.raw_chars,
        )
    return decision


# ---------------------------------------------------------------------------
# Lifespan + admin endpoint dependency
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: ensure /data/openwebui/compactor/{facts,summaries,chromadb}/
    exist. Idempotent. Shutdown: cancel any in-flight background tasks.
    """
    try:
        ensure_storage_layout()
        logger.info("storage layout ready")
    except Exception as e:
        logger.warning(f"could not initialize storage layout: {e}")
    # v3.1.9: an unusable COMPACTOR_TIMEZONE is an ERROR at boot, not a
    # surprise in her first reply. Never raises (see _resolve_time_zone).
    _announce_time_zone()
    # v3.1.3: warm the exact local tokenizer HERE, off the loop, for the
    # same reason as the modality probe below - lazily it loaded inside the
    # async request handler, so the FIRST request after every boot that had
    # any summary state paid ~824ms of blocked event loop. One call warms
    # the process-wide singleton that summarizer and facts both consult.
    try:
        await run_in_threadpool(summarizer._estimate_block_tokens, "warmup")
        logger.info("local exact tokenizer warmed (or confirmed unavailable)")
    except Exception as e:
        logger.warning(f"tokenizer warm failed (non-fatal): {e}")
    # v3.0.3: resolve backend modality HERE rather than lazily on the first
    # chat. AutoConfig.from_pretrained can touch the network (an HF HEAD
    # request when the config is not cached), and the lazy path ran it inside
    # the async request handler — blocking the event loop, and with a
    # slow/unreachable HF hub it would stall the very first user's message.
    # At startup a stall is invisible and the result is cached for process life.
    try:
        await run_in_threadpool(backend_is_multimodal)
    except Exception as e:
        logger.warning(f"modality probe failed at startup (will retry lazily): {e}")
    # v3.1 P0-0b: _BUDGET_MARGIN is a module global, so every correction the
    # calibration loop ever learned is gone the moment this process restarts.
    # Confirmed live on 2026-08-27 — a pod recreate mid-diagnosis reset it to 0
    # and the climb started over with nothing in the log to say why. Persisting
    # it is a larger change; until then the reset is at least announced, so the
    # 400 the next long conversation eats is explained rather than mysterious.
    # INFO, not WARNING: this fires on every clean boot, and a warning that is
    # always present is a warning nobody reads — the exact habit that let the
    # token-counter fallback run unnoticed for months.
    logger.info(
        f"context calibration starts at {_BUDGET_MARGIN} for this process — "
        f"the learned budget margin does not survive a restart. The first "
        f"conversation large enough to expose the token undercount will take "
        f"one vLLM 400 before the margin is relearned."
    )
    yield
    # Graceful: give in-flight background work (fact extraction, indexing,
    # rollup, backfill) a moment to finish via the bounded pool.
    await bgwork.pool.drain(timeout=10.0)


app = FastAPI(title="context-compactor", lifespan=lifespan)


@app.exception_handler(UnsafeConvId)
async def _unsafe_conv_id_handler(request: Request, exc: UnsafeConvId):
    """A rejected conv_id is a 400 about the request, not a 500 about us.

    APP-WIDE rather than per route, and that is the whole point. The
    traversal this guard closes reached the filesystem through TWO admin
    routes that take a conversation id from the request body, and the first
    fix caught UnsafeConvId at those two. An adversarial sweep immediately
    found a third — inherit-persona's source_conv_id — still returning 500,
    which is the fix-one-site-miss-the-sibling defect committed while
    fixing an instance of it.

    Every route that builds a store path is covered here, including ones
    not written yet. The per-route catches that remain are the ones which
    also handle ImportError_ and would otherwise need a bare re-raise.
    """
    logger.warning(
        f"rejected {request.method} {request.url.path}: unsafe conv_id ({exc})"
    )
    return JSONResponse(status_code=400, content={"detail": str(exc)})


def _require_localhost(request: Request) -> None:
    """FastAPI dependency: gate admin endpoints to localhost unless
    COMPACTOR_ADMIN_BIND is explicitly set to something other than 127.0.0.1.
    """
    if ADMIN_BIND != "127.0.0.1":
        return  # operator opted in to external admin access
    client_host = request.client.host if request.client else None
    if client_host not in ("127.0.0.1", "::1", "localhost"):
        raise HTTPException(
            status_code=403,
            detail=(
                "admin endpoints are localhost-only by default; "
                "set COMPACTOR_ADMIN_BIND=0.0.0.0 to expose externally"
            ),
        )


# ---------------------------------------------------------------------------
# V2.3 Theme 2 — vLLM-restart resilience
# ---------------------------------------------------------------------------
# When vLLM is down/restarting, a request to it raises httpx.RequestError
# (connection refused, read error mid-restart, etc.). Without handling, that
# surfaces as an opaque 500 / "Exception in ASGI application". Instead we
# return a clean 503 with a friendly, retryable message so the client (and
# the user watching OpenWebUI) sees "the model is starting" rather than a
# crash.

MODEL_RESTART_MESSAGE = (
    "⏳ The model backend is starting up or restarting. Please retry in a "
    "few moments."
)


def _vllm_unreachable_body(detail: str) -> dict:
    """OpenAI-error-shaped body for a 503 when vLLM can't be reached."""
    return {
        "error": {
            "message": MODEL_RESTART_MESSAGE,
            "type": "service_unavailable",
            "code": "model_unavailable",
            "detail": detail,
        }
    }


def _vllm_unreachable_stream_chunks(model: str, message: str | None = None) -> list[dict]:
    """SSE chunks that show a friendly message as an assistant reply, so a
    streaming client degrades visibly rather than getting a dead stream.

    `message` defaults to the backend-restarting text. Pass an explicit one for
    cases where that would be FALSE — a 4xx means the backend is healthy and
    rejected *our* request, and telling the user "the model is starting up" is
    the system bearing false witness about its own state (see
    COGNITIVE_ARCHITECTURE.md: degrade honestly, claim nothing unearned).
    """
    cid = f"chatcmpl-unavail-{int(time.time() * 1000):x}"
    created = int(time.time())
    base = {"id": cid, "object": "chat.completion.chunk", "created": created,
            "model": model or "compactor"}
    return [
        {**base, "choices": [{"index": 0,
                              "delta": {"role": "assistant",
                                        "content": message or MODEL_RESTART_MESSAGE},
                              "finish_reason": None}]},
        {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]


# What the USER reads when vLLM refuses the request. On 2026-08-24 23:49 a turn
# cost 139.9s of compaction, took a context-length 400 at 33,127 tokens, and
# then produced no reply, no indexed exchange, no "+N facts" line and no
# episodic write — while the old text below told the reader only that something
# "couldn't be processed" and that their memory was "safe". Both statements were
# true and neither answered the question the reader actually has: did my message
# go through? So every one of these leads with the outcome, and says in the same
# breath that the turn was not remembered either — because a user who believes
# the model heard them will build on it, and the model never will.
_REJECTED_PREAMBLE = "⚠️ This message did not go through."
_REJECTED_MEMORY_NOTE = (
    "There is no reply to it, and nothing about this turn was saved to memory — "
    "the model will not see it next time. Everything from before is intact."
)

REQUEST_REJECTED_MESSAGE = (
    f"{_REJECTED_PREAMBLE} The model backend rejected the request (a problem "
    f"on my side, not yours). {_REJECTED_MEMORY_NOTE} If it happens again on "
    f"the same message, the operator should check the compactor log for the "
    f"rejection reason."
)

CONTEXT_OVERFLOW_MESSAGE = (
    f"{_REJECTED_PREAMBLE} The conversation was too large for the model's "
    f"context window even after compaction. {_REJECTED_MEMORY_NOTE}"
)
# Appended to the above, and which one depends on whether the calibration
# backstop actually learned something from this rejection. "Send it again" was
# observed to be a lie on 2026-08-27: three consecutive failures moved the
# margin +127 each time while it needed ~5250, so the same advice produced the
# same failure ~19 times. Only promise the retry when the margin moved.
CONTEXT_OVERFLOW_RETRY = (
    " Send it again — the compactor has just measured how far its own size "
    "estimate was off and has tightened its budget to match."
)
CONTEXT_OVERFLOW_NO_RETRY = (
    " Sending it again will most likely fail the same way: the compactor "
    "learned nothing new from this rejection. The operator should check the "
    "compactor log and shrink the conversation or the memory budget."
)


def _request_rejected_stream_chunks(
    model: str, message: str, code: str, detail: str = ""
) -> list[dict]:
    """SSE chunks for a request vLLM REFUSED — a 4xx, not an outage.

    Deliberately not `_vllm_unreachable_stream_chunks`. That function's pair
    ends `finish_reason: "stop"`, which to OpenWebUI is an ordinary successful
    completion whose text happens to read like an apology (INCIDENT §4.2
    verified this). The user's turn is gone and the transcript records a normal
    assistant reply; nothing downstream can tell the difference. So this pair
    carries BOTH halves:

      - the visible assistant text, so a client that understands nothing still
        shows the user why their message failed, and
      - a top-level `error` object with `finish_reason: "error"`, so a client
        that does understand can mark the turn failed instead of storing it.

    Adding the error object cannot make the failure less visible; omitting it
    is what made the failure indistinguishable from a reply.
    """
    cid = f"chatcmpl-rejected-{int(time.time() * 1000):x}"
    created = int(time.time())
    base = {"id": cid, "object": "chat.completion.chunk", "created": created,
            "model": model or "compactor"}
    err = {"message": message, "type": "invalid_request_error", "code": code}
    if detail:
        err["detail"] = detail
    return [
        {**base, "choices": [{"index": 0,
                              "delta": {"role": "assistant", "content": message},
                              "finish_reason": None}]},
        {**base,
         "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
         "error": err},
    ]


def _rejection_user_message(err_body: str, tightened: bool) -> tuple[str, str]:
    """(the text the user reads, the OpenAI error code) for one 4xx body."""
    if _is_context_overflow(err_body):
        return (
            CONTEXT_OVERFLOW_MESSAGE
            + (CONTEXT_OVERFLOW_RETRY if tightened else CONTEXT_OVERFLOW_NO_RETRY),
            "context_length_exceeded",
        )
    return REQUEST_REJECTED_MESSAGE, "backend_rejected"


def _sent_token_size(messages: list[dict]) -> tuple[int | None, str]:
    """Our own size for a payload, and WHICH counter produced it.

    Resolved exactly the way _enforce_hard_budget resolves it, because this is
    reporting on the decision that guard made: /tokenize when vLLM answers,
    the local tokenizer otherwise. Naming the source is the point — the two
    disagree badly on assistant content, in the direction that overflows (see
    count_tokens_exact for the measurements), so a bare "we measured N" means
    two different things depending on which one measured it, and the reader of
    this line is trying to locate exactly that gap.

    Blocking (httpx + tokenizer); call it through run_in_threadpool.
    """
    try:
        exact = count_tokens_exact(messages)
        if exact is not None:
            return exact, "vLLM's /tokenize"
        return count_tokens(messages), "the local tokenizer"
    except Exception:
        # This runs on a path that has already failed. It may not add a second
        # failure to the first — the log line below is still worth writing
        # without a count in it.
        return None, ""


def _log_request_rejected(
    conv_id: str | None,
    status: int,
    err_body: str,
    sent_tokens: int | None,
    sent_source: str,
    limit: int,
    streaming: bool,
) -> None:
    """The turn is lost; this line is the only thing that will say so.

    2026-08-24 23:49 [M]: openwebui.log logged 200, compactor.log logged 200,
    and the entire record of a destroyed user turn was two WARNING lines in a
    file named *-error.log — carrying no conv_id and no token counts. ERROR is
    the level, because a turn that produced nothing is not a degradation. The
    two counts are the level's justification: our estimate beside vLLM's real
    one IS the undercount, and it is not recoverable after the fact from
    anything else in the log.
    """
    reported = _reported_prompt_tokens(err_body)
    if sent_tokens is None:
        counts = f"budget was {limit:,} tokens (our own count was unavailable)"
    else:
        counts = (
            f"we measured {sent_tokens:,} tokens with {sent_source} against a "
            f"{limit:,}-token budget"
        )
    if reported is not None:
        counts += f"; vLLM counted {reported:,}"
        if sent_tokens is not None:
            # Named by direction rather than always "undercount": which way the
            # gap runs is the whole diagnosis, and a line that calls an
            # overcount an undercount sends the next reader looking for a
            # cause that is not there.
            gap = reported - sent_tokens
            counts += (
                f" — we UNDERCOUNTED by {gap:,}" if gap > 0
                else f" — we OVERCOUNTED by {-gap:,}" if gap < 0
                else " — our count agreed, so the rejection is not a counting error"
            )
    # The stream path commits HTTP 200 in the response header before vLLM has
    # answered, so every access log upstream of here records a success. Saying
    # so in the line is what stops the next reader from concluding, as the
    # 2026-08-24 analysis initially did, that the turn must have succeeded.
    status_note = (
        "; the client was already sent HTTP 200 (a stream commits its status "
        "before the backend answers), so no access log will show this"
        if streaming else ""
    )
    # A 4xx is vLLM refusing a request it understood; a 5xx is vLLM failing.
    # The turn is equally gone either way — which is why both come here — but
    # calling a backend fault a rejection sends the reader to the wrong half of
    # the system.
    headline = (
        f"REQUEST REJECTED by vLLM (HTTP {status})" if status < 500
        else f"vLLM FAILED this request (HTTP {status})"
    )
    logger.error(
        f"conv={conv_id}: {headline} — this turn produced no reply, no facts "
        f"and no episodic write, and nothing retries it. {counts}{status_note}. "
        f"vLLM said: {err_body[:300]!r}"
    )


# ---------------------------------------------------------------------------
# Main request flow
# ---------------------------------------------------------------------------

def _reject_json_constant(name: str):
    """Refuse NaN / Infinity / -Infinity in a request body.

    Python's json.loads ACCEPTS these three as a non-standard extension, so
    a body carrying them parses cleanly and looks like an ordinary dict.
    Nothing downstream can take them: httpx encodes the forwarded request
    with allow_nan=False, so the failure surfaced at the FORWARD step as
    "ValueError: Out of range float values are not JSON compliant" and the
    client got a 500 — for a body the backend never saw.

    Rejecting at PARSE time rather than validating the sampling parameters
    afterwards, for two reasons. It costs nothing on a normal body: this is
    called only when one of the three literals actually appears. And it
    covers every position, including nested ones, where a hand-written list
    of numeric fields would cover the half someone thought of.
    """
    raise ValueError(f"{name} is not valid JSON for a request body")


def _finite_json_float(s: str) -> float:
    """`parse_float` for `json.loads`: like the default `float(s)`, but
    raises for a numeral that parses to a NON-FINITE value.

    P8-6 (hostile pass #8): `_reject_json_constant` above only intercepts
    the bare `NaN` / `Infinity` / `-Infinity` CONSTANT names — an ordinary-
    looking JSON NUMBER that merely overflows float range, like
    `1e999`, never calls it at all; Python's json module hands it to
    `parse_float` (or plain `float()`) which silently returns `inf`. That
    body then parsed cleanly and looked ordinary: `{"max_tokens": 1e999}`
    reached `int(body.get("max_tokens") or 0)` below, and `int(inf)` raises
    `OverflowError`, which the surrounding `except (TypeError, ValueError)`
    did not catch — a 500 from a client-supplied number the parser itself
    could reject far more cheaply, before compaction or memory injection
    ever touch the request. Any OTHER numeric sampling key (temperature,
    presence/frequency penalty, ...) sending the same digits would reach
    httpx's `allow_nan=False` encoder and 500 the same way F4 already
    documented for repeat_penalty/repetition_penalty. Rejecting at PARSE
    time, like `_reject_json_constant`, covers every numeric field at once
    — the same reasoning that function's own docstring gives for NaN and
    the bare Infinity constant applies just as well to an ordinary numeral
    that merely evaluates to one.
    """
    f = float(s)
    if not math.isfinite(f):
        raise ValueError(f"{s} is not a finite JSON number")
    return f


def _unpaired_surrogate(obj: Any) -> str | None:
    """The UnicodeEncodeError text if `obj` cannot be written as UTF-8 JSON.

    A LONE SURROGATE is valid JSON and a valid Python str, and it cannot be
    encoded: json.loads turns the escape into a one-character string that looks
    ordinary until the first write. Paired surrogates are combined by json.loads
    into an astral character and pass, so ordinary emoji are unaffected.
    """
    try:
        json.dumps(obj, ensure_ascii=False).encode("utf-8")
    except UnicodeEncodeError as e:
        return str(e)
    return None


def _refuse_unpaired_surrogate(body: Any) -> None:
    """400, before any handler does anything with `body`.

    v3.1.9 (M3, BLOCKER). chat_completions had this guard since v3.1.8 and its
    seven body-parsing siblings did not. Through /admin/conversations/import it
    was not a 500 but a DELETION: import_conversation wiped the facts file,
    wiped the episodic index, and only then tried to write the bundle, which
    raised UnicodeEncodeError — a ValueError the endpoint's handler did not name.
    Executed against the real memory.py: 105 facts in, [] on disk, HTTP 500.

    One helper, called by every handler that reads a JSON body, and
    test_surrogate_guard.py walks this file's AST to fail the build if a handler
    that calls `request.json()` does not also call this. Seven copies of a
    guard is how the first one got missed; an eighth handler is how the next
    one would be.
    """
    err = _unpaired_surrogate(body)
    if err is not None:
        logger.warning(f"rejected request carrying an unpaired surrogate: {err}")
        raise HTTPException(
            status_code=400,
            detail=(
                "body contains an unpaired surrogate, which cannot be "
                "encoded as UTF-8"
            ),
        )


# ---------------------------------------------------------------------------
# v3.1.9.2: keep detected loop replies out of what is FORWARDED to vLLM, not
# only out of what is memorized.
#
# reply_is_degenerate + _redact_degenerate_turns already keep a repetition
# loop out of the rollup/fact-extraction input. They do NOT touch what
# chat_completions sends back to vLLM: OpenWebUI keeps the loop reply in chat
# history and re-sends it on every later request, and the hard-budget guard
# leaves only ~5 turns in the forwarded window, so right after a loop the
# loop reply can be a large fraction of everything the model sees — which is
# exactly the conditioning that makes the NEXT reply degenerate too (observed
# in production: the reply after a loop came back empty).
#
# The placeholder is deliberately generic and inert: it must read as
# ordinary history (the chat template refuses empty assistant content, so it
# cannot be blank) and must not use the words "loop" or "repetition" the
# model could itself latch onto and echo — the failure this exists to stop
# is exactly the model fixating on a short phrase.
_DEGENERATE_FORWARD_PLACEHOLDER = "[a short reply was given here]"


def _redact_forwarded_loop_replies(messages: list[dict]) -> tuple[list[dict], int, int]:
    """-> (copy of `messages` with degenerate ASSISTANT turns touched, count
    touched, count of those replaced WHOLE by the placeholder).

    Mirrors _redact_degenerate_turns (same detector, same clean-head rule,
    via the shared `_degenerate_replacement_content` helper) but is a
    SEPARATE call, on the FORWARDED window, because the two redactions run at
    different times for different reasons and must not be collapsed into
    one: this one runs on every request (see the call site in
    chat_completions for why it must run AFTER compaction/injection and
    BEFORE _enforce_hard_budget), the other runs once per rollup.

    Only assistant turns are touched — never a user turn, a system message,
    the compaction stand-in, or (by construction, since only PRIOR turns are
    degenerate-checkable — the newest message is always the one this request
    is asking a reply TO) the newest message.

    P8-8 (hostile pass #8): `touched` used to be the only count returned,
    and the call site logged it as "replaced N ... with a placeholder" —
    true in v3.1.9.2's first cut, false since P8-3/P8-4's span-cut fix: on
    the 2026-09-16 backup only 10 of 66 flagged replies were replaced
    WHOLE, the other 56 kept a clean head/tail and lost only the flagged
    span. An operator could not tell "a whole answer vanished" from "one
    short span was cut out of an otherwise-intact reply" from the log
    alone. The third return value is exactly that split.
    """
    out = []
    touched = 0
    whole = 0
    last_index = len(messages) - 1
    for i, m in enumerate(messages):
        if (
            i != last_index  # never the newest message (continue_final_message
            # can make it an assistant turn; it is what THIS request is about,
            # not history to sanitize)
            and isinstance(m, dict)
            and m.get("role") == "assistant"
            and reply_is_degenerate(_message_text(m))
        ):
            content, kept_head = _degenerate_replacement_content(
                _message_text(m), _DEGENERATE_FORWARD_PLACEHOLDER, keep_middle=True
            )
            m = {**m, "content": content}
            touched += 1
            if not kept_head:
                whole += 1
        out.append(m)
    return out, touched, whole


# ---------------------------------------------------------------------------
# v3.1.9.2: Ollama sampling-name translation.
#
# The owner's OpenWebUI model had `repeat_penalty` set (Ollama's name for
# vLLM's `repetition_penalty`). OpenWebUI's `apply_model_params_to_body_openai`
# passes UNKNOWN keys through verbatim rather than dropping them, so
# `repeat_penalty` rode all the way to vLLM 0.19, which does not recognise it
# — it lands in pydantic's `model_extra` and `SamplingParams.repetition_penalty`
# stayed at its default of 1.0. Nothing rejected the request and nothing
# logged: the knob just did nothing, silently, for as long as it was set that
# way. `repeat_last_n` has no vLLM equivalent at all (vLLM's repetition
# penalty has no window) and gets the same silent-drop treatment upstream, so
# it is named here too.
#
# Bounded per-conversation-id "already logged" set, not logsetup.log_once:
# log_once's set is keyed by call site and is meant to hold a handful of
# entries for the process's lifetime; keying it by conv_id here would grow it
# by one entry per DISTINCT conversation forever. This set is capped and
# evicts the oldest entry, because the goal is "don't repeat the line on every
# turn of the SAME conversation", not "remember every conversation ever seen".
_SAMPLING_TRANSLATION_LOGGED: dict[str, None] = {}
_SAMPLING_TRANSLATION_LOGGED_CAP = 2000


def _log_sampling_translation_once(key: str) -> bool:
    """True the first time `key` is seen; False after. Bounded (see above)."""
    if key in _SAMPLING_TRANSLATION_LOGGED:
        return False
    if len(_SAMPLING_TRANSLATION_LOGGED) >= _SAMPLING_TRANSLATION_LOGGED_CAP:
        # dicts preserve insertion order; drop the oldest entry to make room
        # rather than let this grow without bound across a long-lived process.
        _SAMPLING_TRANSLATION_LOGGED.pop(next(iter(_SAMPLING_TRANSLATION_LOGGED)))
    _SAMPLING_TRANSLATION_LOGGED[key] = None
    return True


def _translate_ollama_sampling_params(body: dict, conv_id: str | None) -> None:
    """Mutate `body` in place: translate Ollama-named sampling keys vLLM does
    not understand into the vLLM name, or drop them, before forwarding.

    - `repeat_penalty` -> `repetition_penalty` when the latter is absent.
      When BOTH are present, `repetition_penalty` (the name the client meant
      for vLLM) wins and `repeat_penalty` is simply removed — a client
      sending both is not asking for two penalties, and picking the vLLM
      name is the one reading that cannot silently double-apply anything.
    - `repeat_penalty` is coerced to float; a value that is not a positive
      number (non-numeric, zero, or negative — vLLM requires > 0) is dropped
      with a WARNING rather than forwarded, since a bad value forwarded as
      `repetition_penalty` would 400 the request AFTER compaction and
      injection have already spent the turn.
    - `repetition_penalty` itself, if the client sent it as a numeric
      string (OpenWebUI json-decodes Custom Parameters, so this is normally
      a number, but a hand-built client can send "1.1"), is coerced to
      float in place so vLLM's schema does not reject it.
    - `repeat_last_n` has no vLLM equivalent and is removed either way, with
      an INFO note that it was dropped (not silently, the whole point here).

    No other sampling key is touched.
    """
    conv_key = conv_id or "?"

    def _coerce_positive_float(value):
        # F4 (p7 hostile pass #7): a string "inf" / "Infinity" / "1e999" (or
        # a numeric 1e999, which `json.loads`'s default float() already
        # parses to inf) satisfied `f > 0` and was forwarded as-is. httpx
        # 0.28.1 encodes the outgoing JSON with `allow_nan=False`, so the
        # forward raised `ValueError: Out of range float values are not
        # JSON compliant: inf` from inside the proxy, AFTER compaction and
        # injection had already spent the turn on a request that was never
        # going to reach vLLM (see the docstring above this function). Also
        # reject bool: `isinstance(True, float)` is False but `float(True)
        # == 1.0` silently accepts it as if it were a real penalty someone
        # chose, when it is almost certainly a client typo (a flag value
        # leaking into a numeric field).
        if isinstance(value, bool):
            return None
        try:
            f = float(value)
        except (TypeError, ValueError):
            return None
        if not (f > 0) or not math.isfinite(f):
            return None
        return f

    had_repeat_penalty = "repeat_penalty" in body
    had_repetition_penalty = "repetition_penalty" in body

    if had_repeat_penalty:
        raw = body.pop("repeat_penalty")
        coerced_repeat = _coerce_positive_float(raw)
        if had_repetition_penalty:
            # F4: repetition_penalty used to win unconditionally, even when
            # ITS OWN value was invalid — so
            # {"repeat_penalty": 1.1, "repetition_penalty": "abc"} forwarded
            # NEITHER penalty: the invalid repetition_penalty was dropped
            # below and the client's one valid value (repeat_penalty) had
            # already been discarded here. repetition_penalty still wins
            # when it is itself valid; otherwise fall back to the coerced
            # repeat_penalty rather than losing both.
            existing_valid = _coerce_positive_float(body.get("repetition_penalty"))
            if existing_valid is not None:
                if _log_sampling_translation_once(f"repeat_penalty.both.{conv_key}"):
                    logger.info(
                        f"conv={conv_key}: request set both repeat_penalty and "
                        f"repetition_penalty; keeping repetition_penalty="
                        f"{body['repetition_penalty']!r} and dropping repeat_penalty"
                    )
            elif coerced_repeat is not None:
                _invalid = body.get("repetition_penalty")
                body["repetition_penalty"] = coerced_repeat
                logger.warning(
                    f"conv={conv_key}: repetition_penalty="
                    f"{_invalid!r} was invalid; using "
                    f"repeat_penalty={raw!r} ({coerced_repeat!r}) instead of "
                    f"dropping both"
                )
            # else: both invalid — the general repetition_penalty validation
            # below drops whatever invalid value is still sitting in body.
        else:
            if coerced_repeat is None:
                logger.warning(
                    f"conv={conv_key}: dropping repeat_penalty={raw!r} (not a "
                    f"positive finite number) — not forwarded as "
                    f"repetition_penalty"
                )
            else:
                body["repetition_penalty"] = coerced_repeat
                if _log_sampling_translation_once(f"repeat_penalty.{conv_key}"):
                    logger.info(
                        f"conv={conv_key}: translated Ollama repeat_penalty="
                        f"{raw!r} to vLLM repetition_penalty={coerced_repeat!r} "
                        f"(vLLM 0.19 does not recognise repeat_penalty and "
                        f"ignores it silently otherwise)"
                    )

    # Any repetition_penalty (numeric or string, however it arrived) must be
    # a positive FINITE float by the time it reaches vLLM's schema — not only
    # a string one. A numeric 1e999 (parsed to inf at JSON-decode time) or a
    # string "inf"/"Infinity" both used to pass the old `isinstance(..., str)`
    # gate straight through (the numeric case) or be coerced to `inf` itself
    # (the string case), and either one 500s the forward the same way F4
    # documents for repeat_penalty.
    if "repetition_penalty" in body and not (
        isinstance(body["repetition_penalty"], (int, float))
        and not isinstance(body["repetition_penalty"], bool)
        and math.isfinite(body["repetition_penalty"])
        and body["repetition_penalty"] > 0
    ):
        coerced = _coerce_positive_float(body["repetition_penalty"])
        if coerced is None:
            logger.warning(
                f"conv={conv_key}: dropping invalid repetition_penalty="
                f"{body['repetition_penalty']!r}"
            )
            del body["repetition_penalty"]
        else:
            body["repetition_penalty"] = coerced

    if "repeat_last_n" in body:
        dropped = body.pop("repeat_last_n")
        if _log_sampling_translation_once(f"repeat_last_n.{conv_key}"):
            logger.info(
                f"conv={conv_key}: dropping repeat_last_n={dropped!r} — vLLM's "
                f"repetition penalty has no windowed equivalent"
            )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Any:
    # PARSE DEFENSIVELY. The careful empty/invalid-messages 400 below is
    # the right answer and it could never be reached by the requests that
    # needed it most: `await request.json()` raises on a body that is not
    # JSON, and `body.get(...)` raises AttributeError when the body is
    # valid JSON that is not an OBJECT. Adversarial sweep, v3.1.8: an empty
    # body, a bare string, `null`, and a NaN temperature all returned 500;
    # a JSON array, an integer, and a form Content-Type dropped the
    # connection with no HTTP envelope at all.
    #
    # A 500 from a PROXY is always its own bug. The backend never saw these
    # — they never got that far — so there is nothing to blame upstream for,
    # and a client that sent nonsense deserves to be told which nonsense.
    _raw = await request.body()
    try:
        body = json.loads(
            _raw, parse_constant=_reject_json_constant, parse_float=_finite_json_float
        )
    except Exception as e:
        logger.warning(
            f"rejected chat request with an unparseable body "
            f"({type(e).__name__}): "
            f"ua={request.headers.get('user-agent', '?')!r} "
            f"content-type={request.headers.get('content-type', '?')!r}"
        )
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": "request body must be valid JSON",
                    "type": "invalid_request_error",
                    "code": "unparseable_body",
                }
            },
        )
    if not isinstance(body, dict):
        logger.warning(
            f"rejected chat request whose body is {type(body).__name__}, "
            f"not an object: "
            f"ua={request.headers.get('user-agent', '?')!r}"
        )
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": "request body must be a JSON object",
                    "type": "invalid_request_error",
                    "code": "body_not_an_object",
                }
            },
        )
    # A LONE SURROGATE is valid JSON, valid Python str, and cannot be sent.
    #
    # json.loads happily produces '\ud83d' as a one-character string, so the
    # body parses and looks ordinary. httpx then encodes the forwarded
    # request with ensure_ascii=False and UTF-8 cannot represent an
    # unpaired surrogate, so the failure landed at the FORWARD step and the
    # client got a dropped connection with no HTTP response at all - which
    # is indistinguishable from a network fault, and so is worse than an
    # error. (Adversarial sweep, v3.1.8.)
    #
    # Gated on the escape actually appearing in the raw bytes, because the
    # check is a full re-serialisation and this must not cost anything on a
    # normal turn. A surrogate can only ARRIVE as a backslash-u escape: sent
    # as raw bytes it is invalid UTF-8 and json.loads has already refused it
    # above. Paired surrogates are legal and are combined by json.loads into
    # an astral character, which encodes fine and passes here - so ordinary
    # emoji are unaffected.
    if b"\\u" in _raw or b"\\U" in _raw:
        # The detector is shared with every admin handler; the RESPONSE is
        # not, because this endpoint answers in OpenAI's error shape.
        _surr = _unpaired_surrogate(body)
        if _surr is not None:
            logger.warning(
                f"rejected chat request carrying an unpaired surrogate: {_surr}"
            )
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": (
                            "request body contains an unpaired surrogate, "
                            "which cannot be encoded as UTF-8"
                        ),
                        "type": "invalid_request_error",
                        "code": "unpaired_surrogate",
                    }
                },
            )
    messages = body.get("messages", [])

    # Guard: never forward an empty/invalid messages list to vLLM — its chat
    # templating raises an opaque "list index out of range" (IndexError on
    # conversation[0]) that surfaces in the UI as a broken reply. Seen from
    # OpenWebUI 0.11 background/task-style calls. Log enough to identify the
    # sender, then return a clean OpenAI-shaped 400.
    if (
        not isinstance(messages, list)
        or not messages
        or not all(isinstance(m, dict) for m in messages)
    ):
        logger.warning(
            "rejected chat request with empty/invalid messages: "
            f"ua={request.headers.get('user-agent', '?')!r} "
            f"referer={request.headers.get('referer', '?')!r} "
            f"body_keys={sorted(body.keys())} model={body.get('model')!r} "
            f"stream={body.get('stream')!r} metadata={str(body.get('metadata'))[:200]!r}"
        )
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": "messages must be a non-empty list",
                    "type": "invalid_request_error",
                    "code": "empty_messages",
                }
            },
        )

    # V2.0 Phase 1: conv_id resolution
    conv_id: str | None = None
    try:
        conv_id, source = resolve_conv_id(
            dict(request.headers), messages, body=body
        )
        logger.info(f"conv_id={conv_id} source={source} msgs={len(messages)}")
        # Cheap, and it is the only thing standing between a prompt edit and
        # a silently forked memory store. Runs on the request path but only
        # touches disk for LONG hash-derived conversations, which is a rare
        # shape; never raises.
        _warn_if_conversation_forked(conv_id, source, messages)
    except Exception as e:
        logger.warning(f"conv_id resolution failed: {e}")

    # v3.1.9.2: translate/drop Ollama-named sampling keys BEFORE anything
    # else touches `body`, so every later stage (including the eventual
    # forward to vLLM) sees the vLLM-shaped body. See
    # _translate_ollama_sampling_params's docstring for why this exists.
    try:
        _translate_ollama_sampling_params(body, conv_id)
    except Exception as e:
        logger.warning(f"conv={conv_id or '?'}: sampling param translation failed (non-fatal): {e}")

    # The latest user message — used both as the RAG retrieval query and,
    # later, as the exchange's user half for the async indexing/facts tail.
    # Computed from the ORIGINAL messages (compaction preserves the last
    # user turn, but we want the pristine text here).
    last_user_text = _extract_last_user_text(messages)
    # Turn index ≈ position of the assistant reply we're about to produce.
    turn_index = len(messages) + 1

    last_user_text = _memorable_user_text(messages, last_user_text)

    # v3.0.1: a text-only backend must never receive image parts — vLLM 400s
    # the whole request, and because clients re-send full history, a single
    # uploaded image otherwise poisons its conversation permanently. Strip to
    # honest placeholders before compaction/budgeting/injection, so the V3.1
    # image-preserving paths simply never fire.
    #
    # v3.0.3: this MUST run AFTER conv_id resolution and last_user_text. The
    # hash-fallback conv_id is sha256(system|||first_user[:512]) over TEXT
    # parts only — deliberately, so it is stable across multimodal/text-only
    # *client* variants (memory.py _message_text_for_hash). Stripping appends a
    # placeholder to that text, so stripping first broke the invariant from the
    # server side: swapping between a vision model and a text-only one changed
    # the conv_id mid-conversation and orphaned every fact, summary and
    # embedding under the old id. conv_id and the RAG query now derive from what
    # the CLIENT sent; the strip only shapes what we forward.
    if not backend_is_multimodal():
        messages, _n_stripped = _strip_image_parts(messages)
        if _n_stripped:
            body["messages"] = messages
            logger.info(
                f"stripped {_n_stripped} image part(s) for the text-only backend"
            )
    else:
        # v3.0.4: on a vision backend, keep images only on the most recent
        # image turn(s). Every request, not just when compaction fires —
        # otherwise a single upload rides along forever and real per-image
        # token cost crowds the conversation out of the window.
        messages, _n_demoted = _apply_image_retention(messages)
        if _n_demoted:
            body["messages"] = messages
            logger.info(
                f"image retention: demoted {_n_demoted} older image(s) to text "
                f"(keeping {MAX_RETAINED_IMAGES} most-recent image turn(s))"
            )

    # V2.1 Phase 5: chat command short-circuit. If the user typed a
    # recognized slash command (/list-facts, /forget, /remember, etc.),
    # handle it inside the compactor and return a synthetic completion.
    # vLLM never sees the request — zero token cost, instant response.
    # Detection is permissive: messages starting with `/` whose first
    # token is NOT a recognized command pass through unchanged.
    cmd_name, cmd_arg = commands.parse_command(last_user_text)
    if cmd_name and conv_id:
        try:
            cmd_text = await commands.handle_command(
                cmd_name, cmd_arg, conv_id,
                ctx={
                    "turn_index": turn_index,
                    "clear_all_memory": lambda cid: _clear_all_memory(cid, source="chat-command"),
                    "persona_text": persona.get_persona_text(conv_id),
                },
            )
        except StoreUnreadable as e:
            # v3.1: a real person types these into a chat box. The operator
            # needs the path and the exception; she needs to know the data is
            # not gone and that she did nothing wrong. Full detail to the log,
            # plain language to the reply.
            logger.error(f"conv={conv_id}: /{cmd_name} failed — store unreadable: {e}")
            cmd_text = (
                "I couldn't read my stored memory for this conversation just "
                "now, so I've made no changes rather than risk losing anything. "
                "Nothing has been deleted. This is a problem on my side — "
                "please try again in a moment, and mention it if it keeps "
                "happening."
            )
        except Exception as e:
            logger.exception(f"command handling failed: {e}")
            cmd_text = f"Command failed: {type(e).__name__}: {e}"
        logger.info(
            f"conv={conv_id}: handled /{cmd_name} (arg_len={len(cmd_arg)})"
        )
        stream_flag = bool(body.get("stream", False))
        if stream_flag:
            chunks = commands.build_synthetic_completion_stream(
                cmd_text, body.get("model") or MODEL_REPO or "",
            )

            async def cmd_stream():
                for chunk in chunks:
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(cmd_stream(), media_type="text/event-stream")
        return JSONResponse(
            content=commands.build_synthetic_completion(
                cmd_text, body.get("model") or MODEL_REPO or "",
            ),
            status_code=200,
        )

    # The window this request will finally be measured against, computed HERE
    # rather than at the pre-flight below because the memory injection that
    # follows has to be bounded by it. vLLM enforces prompt + max_tokens <=
    # window, so a fixed reserve alone leaves a client asking for a big
    # completion still 400able; and a memory budget expressed as a token
    # constant cannot see any of that. Nothing between here and the guard
    # depends on the value, and it depends on nothing but `body`.
    #
    # v3.1.9.1: moved ahead of V1 compaction (was after it) so that
    # `inject_budget`, below, exists before `compact_if_needed` runs — see
    # that move's reason on `inject_budget` itself.
    try:
        req_max_tokens = int(body.get("max_tokens") or 0)
    except (TypeError, ValueError, OverflowError):
        # P8-6 (hostile pass #8): `int(inf)` raises OverflowError, not
        # TypeError/ValueError — not caught here before this fix, so a
        # non-finite max_tokens 500'd the proxy instead of falling into
        # this branch. `_finite_json_float` (chat_completions's
        # `json.loads`) now rejects a non-finite NUMERAL at parse time for
        # every numeric field, closing that off before this line ever
        # runs; this except is the second line of defence for any other
        # unparseable shape (a string, a list, ...). Either way, the
        # CLIENT'S OWN bad value must not silently ride along in `body` —
        # this branch decided the budget math would treat it as absent (0)
        # while leaving whatever the client actually sent untouched, so a
        # value this guard could not parse could still reach vLLM as-is
        # and fail there instead. Drop it and log, the same way
        # _translate_ollama_sampling_params drops an invalid penalty
        # rather than forwarding it unexamined — never silently REWRITE a
        # value the client chose, only ever drop an invalid one.
        if "max_tokens" in body:
            logger.warning(
                f"conv={conv_id or '?'}: dropping invalid max_tokens="
                f"{body['max_tokens']!r} (not a finite integer) — not "
                f"forwarded"
            )
            del body["max_tokens"]
        req_max_tokens = 0
    if req_max_tokens > MAX_MODEL_LEN // 2:
        # Pair with the reserve cap in effective_limit so prompt+completion
        # always fits.
        req_max_tokens = MAX_MODEL_LEN // 2
        body["max_tokens"] = req_max_tokens
    effective_limit = min(
        MAX_MODEL_LEN,
        max(256, MAX_MODEL_LEN - max(GENERATION_RESERVE, req_max_tokens)),
    )

    # v3.1.9.1: also moved ahead of V1 compaction, for the same reason —
    # `has_history` only reads `messages` (unchanged, still the client's
    # original array; compaction hasn't run yet) and `inject_budget` only
    # reads `effective_limit`, computed just above, so nothing here depends
    # on compaction having happened.
    has_history = _has_conversational_history(messages)
    inject_budget = int(
        effective_limit
        * (
            INJECTION_BUDGET_FRACTION
            if has_history
            else INJECTION_NO_HISTORY_FRACTION
        )
    )

    # V1 compaction
    # hostile2-reuse M1: `_compaction_stored_turns` is how the summary
    # injection below (search `format_summary_block`) learns whether THIS
    # call already put a stand-in for the hierarchy in the array, so it can
    # skip injecting its own, separately-trimmed copy of the same summaries
    # — see compact_if_needed's docstring for why this is an out-param
    # rather than a return-type change.
    _compaction_stored_turns: list[int] = []
    try:
        body["messages"] = await compact_if_needed(
            messages, conv_id,
            stored_turns_out=_compaction_stored_turns,
            inject_budget=inject_budget,
        )
    except Exception as e:
        logger.exception(
            f"compaction failed; falling through with the original messages — "
            f"the hard-budget guard will shed content if they don't fit: {e}"
        )

    # V2.0 memory injection. ALL three layers (facts, RAG, summary) are
    # collected into a SINGLE combined system message and injected in one
    # shot. This matters because Mistral-family chat templates (Mistral-
    # Nemo, Mistral-Small, and therefore Magnum v4 12B/22B) enforce
    # "at most one system message before strict user/assistant alternation"
    # and reject requests with multiple consecutive system messages with a
    # 400 "must alternate user/assistant" error. Combining is the
    # template-portable form: one system block holds all three sections
    # internally, separated by blank lines and labeled by each module's
    # block header (so the model still parses them as distinct contexts).
    touched_facts: list[dict] = []
    # The subset of touched_facts this turn actually put in front of the model.
    # Initialized here, beside the store, because the tail reads it whether or
    # not the facts block below ran at all — an unreadable store or a disabled
    # RAG path must leave the extractor with an empty set, not a NameError on
    # the async tail where nothing would surface it.
    injected_facts: list[dict] = []
    # (priority, label, text). Priority orders DROPPING, not sending: the list
    # is still sent in the order it is built. See _bound_injected_blocks.
    injected_blocks: list[tuple[int, str, str]] = []
    # Initialised unconditionally: it is ASSIGNED inside the facts branch and
    # READ after the injection bound, so a conversation whose facts path does
    # not run would raise NameError on the request path. Unbound names have
    # broken this deployment five times and static analysis does not see them.
    _pending_touch = None
    log_parts: list[str] = []
    # Bound before the summary load so the injection log line can name it even
    # when that load fails — an unreadable summary is exactly when you want the
    # rest of the line.
    last_turn: object = "?"
    turns_seen: object = "?"
    if conv_id:
        # --- Persona (Phase 8) ---
        # Two paths feed the persona layer:
        #   1. Auto-capture: when the request's first system message is
        #      long enough (≥ AUTO_DETECT_MIN_CHARS) we save it for
        #      portability/library/diagnostics. No injection needed —
        #      vLLM already sees the text via messages[0].
        #   2. Admin/inherited: persona stored without being in this
        #      request's messages. text_to_inject returns it so the
        #      combined system block carries it.
        # The hash-match check in text_to_inject prevents double-injection.
        try:
            # Auto-capture writes a new persona file — gate it under disk
            # pressure (it's automatic growth). Injection of an already-
            # stored persona below is a read and always proceeds.
            if degrade.guard("persona auto-capture"):
                persona.auto_capture_persona(conv_id, messages)
            ptext = persona.text_to_inject(conv_id, messages)
            pblock = persona.format_persona_block(ptext)
            if pblock:
                injected_blocks.append(
                    (_INJECT_PRIORITY_PERSONA, "persona", pblock)
                )
                log_parts.append(f"persona({len(ptext)}ch)")
        except StoreUnreadable as e:
            # auto_capture_persona reads before it writes, so an unreadable
            # record used to read as "no persona set" and get replaced by
            # whatever this request's system message happened to be — on the
            # request path, before vLLM is called (v3.1 F1c). It now raises
            # here instead: nothing is written, and this turn goes out
            # without the persona block rather than losing it.
            logger.error(
                f"conv={conv_id}: persona file unreadable ({e}); not captured, "
                f"not injected, and NOT overwritten"
            )
        except Exception as e:
            logger.warning(f"conv={conv_id}: persona handling failed (non-fatal): {e}")

        # --- Facts (Phase 2) ---
        try:
            touched_facts = facts.load_facts(conv_id)
            if touched_facts:
                # v3.1 F9: touch — and inject — only the budget-bounded subset,
                # not everything on disk. Touching the whole store stamped every
                # fact with the same second on every request, which left
                # last_used carrying no signal and eviction falling through to
                # added_turn, i.e. deleting the conversation's oldest and most
                # foundational facts first. select_for_injection returns the SAME
                # dict objects, so touched_facts still carries the touch and the
                # tail below still writes the whole store back; the facts left
                # out keep their real last_used and become the eviction
                # candidates, which is the entire point.
                # query_text activates F1's relevance ranking. It MUST ship
                # with the /pin command in commands.py: the injection budget
                # (COMPACTOR_INJECT_FACTS_TOKENS, default 400) took effect the
                # moment facts.py landed, so without ranking the block would be
                # cut from ~80 facts to ~26 by the degenerate FIFO order F1
                # exists to replace - strictly worse than before. Ranked, the
                # 26 are the ones this turn is about; pinned identity facts
                # bypass ranking entirely.
                injected_facts = facts.select_for_injection(
                    touched_facts, query_text=last_user_text
                )
                # NOT touched here. last_used is the LRU eviction key, and
                # _bound_injected_blocks (below) may drop the facts block
                # entirely — so touching now records "recently used" for facts
                # the model never saw. On a conversation where the bound fires
                # repeatedly that inverts the eviction order: the facts that
                # were never sent look freshest and survive, while facts that
                # WERE sent age out. The touch moved to after the bound.
                _pending_touch = injected_facts
                block = facts.format_facts_block(injected_facts)
                if block:
                    injected_blocks.append(
                        (_INJECT_PRIORITY_FACTS, "facts", block)
                    )
                    log_parts.append(
                        f"{len(injected_facts)}fact(s)"
                        if len(injected_facts) == len(touched_facts)
                        # Only differ when the store is over budget — which
                        # v3.1 F9 now allows to persist, because a failed
                        # archive write keeps the facts rather than deleting
                        # them. Worth seeing in the log when it happens.
                        else f"{len(injected_facts)}/{len(touched_facts)}fact(s)"
                    )
        except Exception as e:
            logger.warning(f"conv={conv_id}: facts load failed (non-fatal): {e}")

        # --- RAG retrieval (Phase 3) ---
        # exclude_turns_from drops retrieved turns that are already present
        # verbatim in the recent window (no point spending budget twice).
        try:
            recent_cutoff = max(0, turn_index - (KEEP_RECENT_TURNS * 2))
            hits = retrieval.retrieve(
                conv_id, last_user_text, exclude_turns_from=recent_cutoff
            )
            rblock = retrieval.format_retrieval_block(hits)
            if rblock:
                injected_blocks.append(
                    (_INJECT_PRIORITY_RETRIEVAL, "retrieval", rblock)
                )
            # Logged unconditionally. Inside the `if` it only ever recorded
            # SUCCESS, so a retrieval layer returning nothing on every request
            # was indistinguishable in the log from one that was never asked.
            # "0retr" is the line that makes a dead episodic layer visible.
            log_parts.append(f"{len(hits)}retr")
        except Exception as e:
            logger.warning(f"conv={conv_id}: retrieval load failed (non-fatal): {e}")

        # has_history / inject_budget: computed ABOVE, before V1 compaction
        # (search "also moved ahead of V1 compaction") — compact_if_needed
        # now needs inject_budget too, for the reused-hierarchy stand-in's
        # budget, so both moved up together rather than being computed twice
        # with two chances to drift apart.

        # --- Hierarchical summary stack (Phase 4) ---
        # State only grows via the async tail (rollups post-response), so
        # this is a purely local read - no LLM call. run_in_threadpool all
        # the same: _estimate_block_tokens consults the exact tokenizer when
        # one is available (7.7ms per request at full tier state, and the
        # FIRST call after boot pays the ~824ms tokenizer load), and this is
        # the async request handler.
        try:
            sstate = summarizer.load_state(conv_id)
            last_turn = sstate.get("last_summarized_turn", "?")
            # v3.1.4: paired with lastturn in the injection line below.
            # Under pipelines/conversation_id_header.py's max_turns cap, msgs=
            # is pinned at the cap forever, so it stopped being evidence of
            # anything about the conversation's size. seen= is the compactor's
            # own count and is the number that says whether the hierarchy is
            # keeping pace: seen climbing while lastturn stands still for more
            # than COMPACTOR_L1_CHUNK_SIZE turns is a stalled rollup.
            turns_seen = sstate.get("turns_seen", 0)
            # hostile2-reuse M1: compact_if_needed already put a stand-in
            # for the hierarchy IN THE ARRAY on this turn (_compaction_
            # stored_turns[0] > 0) — rendered against the array's own
            # all-or-nothing budget. Injecting a SECOND, independently
            # trimmed copy here (60% of a DIFFERENT budget, no all_or_
            # nothing) used to send the shared scenes twice and disagree
            # about which scenes survived — array 9, injected 6, on one
            # documented reproduction. The array copy is authoritative for
            # a reusing turn (it travels with the removal, see
            # compact_if_needed's own comment on why it cannot rely on this
            # injected block instead), so this one is skipped rather than
            # rendered a second time.
            if _compaction_stored_turns and _compaction_stored_turns[0] > 0:
                sblock = None
                log_parts.append("sum(in-array)")
            else:
                # 60% of the injection budget, capped at SUMMARY_BLOCK_MAX_
                # TOKENS: leaves the other 40% (persona, facts, retrieval)
                # room in `inject_budget`, all four bounded together by
                # `_bound_injected_blocks`. P9-1/P9-2 (hostile pass #9):
                # this is NOT what a REUSING turn's stand-in claims any
                # more — that call site (compact_if_needed) has its own
                # formula, `_standin_reuse_ceiling`, because it does not
                # share this constraint (nothing else spends its share on
                # a reusing turn — see STANDIN_BUDGET_FRACTION's comment).
                sblock = await run_in_threadpool(
                    summarizer.format_summary_block,
                    sstate,
                    _standin_injected_share(inject_budget),
                )
                if sblock:
                    injected_blocks.append(
                        (_INJECT_PRIORITY_SUMMARY, "summary", sblock)
                    )
                    log_parts.append(
                        f"sum(L1={len(sstate.get('l1') or [])}"
                        f"/L2={len(sstate.get('l2') or [])}"
                        f"/L3={'y' if sstate.get('l3') else 'n'})"
                    )
        except Exception as e:
            logger.warning(f"conv={conv_id}: summary load failed (non-fatal): {e}")

        # Single inject point — preserves Mistral template compatibility.
        if injected_blocks:
            # v3.1 D3: bound the SUM before it is injected, not after.
            #
            # Downstream of here the only remedy is the hard-budget guard, and
            # the guard's remedies are trimming (which cuts a block at a point
            # its own ranking did not choose) and dropping (which is this
            # decision made blind, without knowing which layer it is spending).
            # Making the choice here means it is made once, with the labels in
            # hand, before any of it has been merged into one opaque block.
            #
            # The no-history budget is the narrow one. See
            # INJECTION_NO_HISTORY_FRACTION for the case that forced it: a
            # request with no prior assistant turn can be neither compacted nor
            # shed, so an oversized injection there is not a degraded turn, it
            # is a lost one.
            # has_history / inject_budget are computed above the summary
            # stack section - the summary cap needed them first.
            kept, dropped_layers, inject_cost = await run_in_threadpool(
                _bound_injected_blocks, injected_blocks, inject_budget
            )
            # Mark facts used only if the facts block SURVIVED the bound.
            # last_used is the LRU eviction key; touching facts the model never
            # received makes them look freshest and pushes the facts that WERE
            # sent toward eviction instead — the eviction order inverts on
            # exactly the conversations where the bound fires most.
            if _pending_touch is not None and "facts" not in dropped_layers:
                facts.touch_facts(_pending_touch)
            if dropped_layers:
                # WARNING, not INFO. This is memory the user believes the
                # assistant has and the model is not going to see, which is the
                # 2026-08-28 lesson in one line: a fallback that cannot say it
                # fired is not a fallback.
                logger.warning(
                    f"conv={conv_id}: injected memory over budget "
                    f"({inject_cost} tokens against {inject_budget}, "
                    f"{INJECTION_BUDGET_FRACTION if has_history else INJECTION_NO_HISTORY_FRACTION:.3f}"
                    f" of the {effective_limit}-token limit"
                    f"{'' if has_history else '; this request has NO prior assistant turn, so it is task traffic or a first turn'}"
                    f") — dropped {', '.join(dropped_layers)} to keep the "
                    f"conversation itself in the window"
                )
                log_parts.append(f"dropped:{'+'.join(dropped_layers)}")
        else:
            kept = []
        if kept:
            combined = "\n\n".join(kept)
            try:
                body["messages"] = inject_system_block(body["messages"], combined)
                # msgs= is repeated here from the request line ~230 lines
                # earlier. That looks redundant and is not: on 2026-08-24 the
                # whole diagnosis was two adjacent log lines nobody joined —
                # a message count in one and the conversation's real size in
                # the other. `msgs=7` beside `105fact(s)` is self-evidently
                # wrong on sight; neither number is, alone. Also carries
                # last_summarized_turn, the only server-side record of how far
                # the conversation actually got, so a client sending a short
                # window is visible without cross-referencing anything.
                logger.info(
                    f"conv={conv_id}: injected memory [{' '.join(log_parts)}] "
                    f"msgs={len(messages)} lastturn={last_turn} "
                    f"seen={turns_seen}"
                )
            except Exception as e:
                logger.warning(f"conv={conv_id}: memory injection failed (non-fatal): {e}")

        # Lazy backfill: if this is an existing V1 conv that has no facts
        # file yet, kick off a background extraction over its full history.
        # Doesn't block this request — current request just degrades to
        # "no facts injected" and next request will see the facts.
        try:
            # Redacted, like the live tail. Backfill runs BOTH jobs the
            # detector exists to gate - it fact-extracts every historical
            # pair and calls summarizer.maybe_rollup - over the client's raw
            # array, and it has no degeneracy gate of its own (grep
            # "degenerate" backfill.py: nothing). It fires exactly when a
            # conversation has history but no facts file: a restore from
            # backup, a migration, a lost store. That is the precise moment a
            # degeneration episode already sitting in her history would be
            # replayed into facts and summaries as though it were worth
            # remembering. Redacting here rather than inside backfill.py
            # because main imports backfill, so the reverse import would be a
            # cycle.
            started = await backfill.start_backfill_if_needed(
                conv_id,
                messages,
                VLLM_URL,
                MODEL_REPO or "",
                fire_and_forget=_fire_and_forget,
                redact=_redact_degenerate_turns,
            )
            if started:
                logger.info(f"conv={conv_id}: lazy backfill started in background")
        except Exception as e:
            logger.warning(f"conv={conv_id}: backfill kickoff failed (non-fatal): {e}")

    # FINAL pre-flight, two steps in a deliberate order (rc6 review):
    #
    # 1. _enforce_hard_budget runs FIRST, while the system blocks are still
    #    separate — so a trim hits the largest individual block (usually the
    #    injected memory) instead of a pre-merged mega-block where halving
    #    would chew into the persona. It is shed against `effective_limit`,
    #    which accounts for the request's OWN max_tokens — vLLM enforces
    #    prompt + max_tokens <= window, so a fixed reserve alone leaves a
    #    client asking for a big completion still 400able. That limit is now
    #    computed further up, before memory injection, because v3.1 D3 bounds
    #    injection as a fraction of it.
    # 2. _merge_adjacent_system_messages runs LAST, collapsing every remaining
    #    run — including any adjacency the budget guard created by deleting a
    #    turn that sat between two system messages.
    # Pure CPU (tokenizer) work — off the event loop so a shedding pass on a
    # huge conversation can't stall every other request and the healthchecks.
    # How many system messages the CALLER sent, counted on the original array
    # before compaction or injection touched it. The guard may spend what we
    # added; it may not spend what the caller sent. Without this it would delete
    # a caller's second system message once injected memory ran out — and in the
    # only case that reaches (one user turn larger than the whole budget) doing
    # so does not even achieve the fit.
    caller_system = sum(1 for m in messages if m.get("role") == "system")
    # v3.1.9: the current-time line (see _inject_time_line). DECIDED here, on
    # the original request, so the guard can be handed a limit that already
    # makes room for it; ADDED after the guard, the merges and the tail
    # repair, because each of those can change which message is her newest.
    # A line the guard never counted would be the one thing in the payload
    # nobody measured, so its reserve (an upper bound, not an estimate) comes
    # out of the guard's limit. Deciding can read the store (repeat task
    # traffic) and must never cost her the reply, so a failure here sends the
    # request undated and says so once.
    _time_line: str | None = None
    _time_reserve = 0
    try:
        _announce_time_zone()
        _time_line = _time_line_for_request(conv_id, messages)
    except Exception as e:
        _time_line = None
        if logsetup.log_once("time_injection.decide"):
            logger.exception(
                f"conv={conv_id or '?'}: could not decide whether to add the "
                f"current-time line ({type(e).__name__}: {e}); sending this and "
                f"any later failing request without it"
            )
    if _time_line is not None:
        _time_reserve = _time_line_token_reserve(_time_line)
        # The guard floors its limit at 256; below that floor a reserve could
        # not be honoured, so a window that small is simply not dated.
        if effective_limit - _BUDGET_MARGIN - _time_reserve < 256:
            _time_line, _time_reserve = None, 0
    # v3.1 D4: what the guard decided, carried to the rejection path. Without
    # it a 400 the guard PREDICTED (and logged at ERROR before sending) is
    # indistinguishable from one that surprised it, and the calibration learns
    # a process-global margin from the first kind.
    guard_report: dict = {}
    # v3.1.9.2: redact detected loop replies out of the FORWARDED window here
    # — after compaction and memory injection, before _enforce_hard_budget —
    # so the guard measures what is actually sent (it must see the shorter
    # placeholder text, not the runaway original) and NOT before compaction:
    # compaction pairs recent turns against the stored covered-turn record by
    # CONTENT, and redacting first would make a degenerate turn unpaired,
    # so it would be treated as new and re-summarized on every request
    # instead of being recognised as already covered.
    body["messages"], _loop_touched, _loop_whole = await run_in_threadpool(
        _redact_forwarded_loop_replies, body["messages"]
    )
    if _loop_touched:
        # Count only — no text. The rollup-input redaction already logs a
        # near-identical line for the same underlying detector; this one is
        # the forwarded-window twin and can fire on requests that never
        # trigger a rollup at all.
        #
        # P8-8 (hostile pass #8): this used to say "replaced N ... with a
        # placeholder" unconditionally, which stopped being true once
        # P8-3/P8-4 made cutting-around-the-span the common case (10 of 66
        # flagged replies replaced whole on the 2026-09-16 backup, not all
        # 66) — an operator could not tell a vanished answer from a
        # trimmed one. whole=<k> names how many actually got the
        # placeholder; the rest (touched - whole) kept a clean head/tail
        # around the collapsed span.
        logger.info(
            f"conv={conv_id or '?'}: touched {_loop_touched} degenerate "
            f"assistant turn(s) in the forwarded window (whole={_loop_whole} "
            f"cut={_loop_touched - _loop_whole})"
        )
    body["messages"] = await run_in_threadpool(
        _enforce_hard_budget,
        body["messages"],
        effective_limit - _time_reserve,
        caller_system,
        guard_report,
        _time_reserve,
    )
    # The limit the FORWARDED payload is held to, REGARDLESS of the line: the
    # guard above shed against `effective_limit - _time_reserve` only to leave
    # the line room, and a rejection this request goes on to take is measured
    # against the real window, not that narrowed one. Computed once, here,
    # rather than recomputed later: _note_backend_rejection moves
    # _BUDGET_MARGIN, so a value read after the response comes back could name
    # a budget that was no longer in force by the time the rejection was
    # logged. Mirrors the clamp inside _enforce_hard_budget.
    enforced_limit = max(256, effective_limit - _BUDGET_MARGIN)
    # True when the guard could not fit the (possibly reserve-narrowed) limit
    # it was handed. Drives ONE decision below: whether there is room left to
    # add the current-time line at all — and there, "narrowed by the reserve"
    # is exactly the question, so this stays as-is (unrenamed) for that use.
    guard_measured_overflow = guard_report.get("fits") is False
    # hostile pass #5 (reviewer A F3). A SEPARATE question, despite starting
    # from the same report: whether a vLLM rejection on THIS payload would be
    # evidence our counting is wrong (_note_backend_rejection's calibration).
    # The guard above was handed the REDUCED limit (less `_time_reserve`), so
    # "fits is False" there can mean either "does not fit the real window" or
    # merely "does not fit with room left for the line" — and only the first
    # is evidence of anything. A payload that clears `enforced_limit` (the
    # real one) is going to be ACCEPTED by vLLM, undated, exactly as
    # measured; telling the calibration otherwise would have it distrust a
    # measurement that was never wrong. It would also, before this fix, have
    # the guard itself log "hard budget FAILED to fit ... vLLM will most
    # likely reject this" at ERROR for a request about to succeed — which the
    # soak then counted as a real failure (reserve=0 on every OTHER caller of
    # _enforce_hard_budget keeps that ERROR exactly as before; see its
    # docstring).
    _reserve_band = (
        guard_measured_overflow
        and _time_reserve > 0
        and guard_report.get("measured") is not None
        and guard_report["measured"] <= enforced_limit
    )
    calibration_overflow = guard_measured_overflow and not _reserve_band
    if _reserve_band and logsetup.log_once(f"time_injection.reserve_band.{conv_id or '?'}"):
        # Once per conversation, not once per process (logsetup.log_once's
        # usual grain): a process serves many conversations, and "no room for
        # the line" is a fact about THIS one's shape, not the process's. Said
        # here, where conv_id is available — the guard itself has none.
        logger.info(
            f"conv={conv_id or '?'}: payload fits the {enforced_limit}-token "
            f"window ({guard_report['measured']} tokens) but not with room "
            f"left for the current-time line ({_time_reserve}-token "
            f"reserve); sending it undated rather than shedding memory or "
            f"turns to make room for a line alone."
        )
    body["messages"] = _merge_adjacent_system_messages(body["messages"])
    # ...and non-system turns that ended up sharing a role (compaction hoists
    # image turns out of chronological order, which lands user next to user).
    body["messages"] = _merge_consecutive_same_role(body["messages"])
    # LAST thing before the payload goes out: make its tail template-valid.
    # After the guard and the merges, because both can change which message
    # is final. See _repair_template_invalid_tail - this is the fix for a
    # production 400 that silently cost the user whole turns.
    _tail_note, _tail_was_invalid = _repair_template_invalid_tail(body)
    if _tail_note and _tail_was_invalid:
        logger.warning(
            f"conv={conv_id or '?'}: payload tail was invalid for the chat "
            f"template - {_tail_note}. Without this the request would have "
            f"been rejected by vLLM with no reply and no memory write."
        )
    elif _tail_note:
        # A client that already sent continue_final_message is not a defect,
        # and warning about it would train the operator to ignore the line
        # that does matter.
        logger.info(f"conv={conv_id or '?'}: {_tail_note}")
    # v3.1.9: date her newest message - the very last change to the payload.
    # Not when the guard measured the payload as NOT fitting: that request is
    # already over the window with nothing left the guard may spend, and a
    # line could only make the rejection vLLM is about to send more certain.
    if _time_line is not None:
        if guard_measured_overflow:
            logger.info(
                f"conv={conv_id or '?'}: not adding the current-time line - the "
                f"payload already does not fit the window"
            )
        else:
            body["messages"], _ = _inject_time_line(body["messages"], _time_line)

    # enforced_limit (the limit the FORWARDED payload was held to, margin
    # already subtracted) was computed right after the guard call above, not
    # here — see that comment for why the timing matters.

    stream = bool(body.get("stream", False))
    # read=None keeps long generations from being cut off, but connect/write/
    # pool stay bounded: an unqualified timeout=None also removes the CONNECT
    # timeout, so a vLLM socket that accepts and then stalls (or a half-open
    # connection after a restart) would hang the request forever.
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=None, write=30.0, pool=10.0)
    )

    if stream:
        accumulator = SseAccumulator()

        async def event_stream():
            vllm_failed = False
            try:
                try:
                    stream_cm = client.stream(
                        "POST", f"{VLLM_URL}/v1/chat/completions", json=body
                    )
                    async with stream_cm as r:
                        if r.status_code >= 400:
                            # vLLM rejected the request (e.g. a 400 from chat-
                            # template validation). Relaying its JSON error body
                            # raw into a text/event-stream gives the UI a garbled
                            # reply; degrade visibly instead, like the
                            # connection-error branch below.
                            #
                            # v3.1: "visibly" used to mean visible to a HUMAN
                            # only. The pair below ended finish_reason "stop"
                            # and the response had already committed HTTP 200,
                            # so a rejection was indistinguishable from a reply
                            # to every machine in the path — INCIDENT §4.3 A5.
                            # On 2026-08-24 23:49 that is exactly what happened:
                            # a context-length 400 after 139.9s of compaction,
                            # 200 in openwebui.log, 200 in compactor.log, and
                            # the only trace two unattributed WARNINGs. So the
                            # branch now says what happened at ERROR, and hands
                            # the client an error-typed pair.
                            vllm_failed = True
                            # Truncate AFTER parsing, not before. vLLM states
                            # the true prompt size mid-sentence, so the old
                            # 300-char cut ran through the one number that
                            # explains the rejection — in the body shape seen in
                            # production it landed just inside the cut, which is
                            # luck, not a margin. The log line still shows 300.
                            err_body = (await r.aread()).decode("utf-8", "replace")[:2000]
                            sent_tokens, sent_source = await run_in_threadpool(
                                _sent_token_size, body["messages"]
                            )
                            # Before _note_backend_rejection, which is what moves
                            # the margin the line reports against.
                            _log_request_rejected(
                                conv_id, r.status_code, err_body, sent_tokens,
                                sent_source, enforced_limit, streaming=True,
                            )
                            # v3.1 A8: enforced_limit is what the guard
                            # ACTUALLY shed against. Without it the calibration
                            # reconstructed a limit from HARD_INPUT_LIMIT and
                            # only ever understated the overshoot, so a client
                            # asking for a large completion could learn nothing
                            # and still be told to retry.
                            tightened = _note_backend_rejection(
                                err_body, enforced_limit,
                                guard_measured_overflow=calibration_overflow,
                            )
                            if r.status_code < 500:
                                # A 4xx means the backend is HEALTHY and refused
                                # our request; only 5xx/unreachable justifies the
                                # "starting up or restarting" message.
                                message, code = _rejection_user_message(
                                    err_body, tightened
                                )
                                chunks = _request_rejected_stream_chunks(
                                    body.get("model") or MODEL_REPO or "",
                                    message, code, detail=err_body[:300],
                                )
                            else:
                                chunks = _vllm_unreachable_stream_chunks(
                                    body.get("model") or MODEL_REPO or ""
                                )
                            for chunk in chunks:
                                yield f"data: {json.dumps(chunk)}\n\n".encode()
                            yield b"data: [DONE]\n\n"
                        else:
                            # v3.1 A10: vLLM accepted this payload. That is the
                            # only evidence that exists for whether the learned
                            # margin is still needed, so it is counted here —
                            # at the moment the status line arrives, not after
                            # the body, because a client hanging up mid-stream
                            # says nothing about whether the prompt fitted.
                            _note_backend_accepted()
                            async for chunk in r.aiter_raw():
                                yield chunk
                                accumulator.feed(chunk)
                except httpx.RequestError as e:
                    # V2.3 Theme 2: vLLM unreachable mid-stream (down /
                    # restarting). Degrade visibly — emit the friendly
                    # message as an assistant reply rather than a dead stream.
                    vllm_failed = True
                    logger.warning(f"vLLM unreachable (stream): {type(e).__name__}: {e}")
                    for chunk in _vllm_unreachable_stream_chunks(
                        body.get("model") or MODEL_REPO or ""
                    ):
                        yield f"data: {json.dumps(chunk)}\n\n".encode()
                    yield b"data: [DONE]\n\n"
            finally:
                await client.aclose()
                # v3.1.7 (R7/R14): flush the incremental decoder before
                # text()/holed() are read below. Unconditional and cheap —
                # a no-op when nothing was ever fed(), which is the case on
                # every path that set vllm_failed without touching
                # accumulator.
                accumulator.finalize()
                # Fire-and-forget post-response work once the stream is done.
                # Everything about whether, and how much of, this reply enters
                # memory is decide_memory_tail's call, made through
                # _run_memory_tail, which the non-streaming path invokes
                # identically: no line of tail policy or bookkeeping exists at
                # one site only. Until v3.1.4 this site was three separate
                # `if`s re-evaluating usable() and its twin was an
                # if/elif/elif with no complete() analogue at all — the drift
                # that lost 63 exchanges from memory in one log window, and
                # the eighteenth "fixed at one site, missed at the other" on
                # this branch.
                #
                # v3.1.7 (R26): `and not vllm_failed` used to sit on this
                # condition, justified as "there's no real assistant turn to
                # extract/index from". That is true of the 4xx branch, where
                # nothing was ever generated — and false of the RequestError
                # branch, which fires when vLLM drops the connection PART WAY
                # THROUGH a reply she has already read. The accumulator held
                # real prose, and decide_memory_tail was never called,
                # tailhealth.note was never called, and no line containing
                # "skipping memory tail" was emitted. Measured: 0 decisions, 0
                # counter movement, an empty grep — on a reply the client
                # received in full. That is the shape of the defect this
                # release argues against in its own comment two screens up.
                #
                # A connection that dies mid-reply IS a cut reply, so the
                # existing trim path handles it exactly: keep the prose up to
                # the last sentence boundary, or skip and SAY SO with a
                # counted outcome. `finished` stays accumulator.complete() and
                # is not forced to False — if vLLM sent finish_reason and then
                # died on the trailing [DONE], the reply really is whole and
                # trimming it would throw away its last sentence.
                #
                # The 4xx branch is safe through here rather than special-
                # cased: nothing on it feeds the accumulator (the friendly
                # error chunks are yielded straight to the client, never
                # accumulator.feed'ed), so text() is "" and the decision is
                # SKIPPED_EMPTY — the one outcome tailhealth treats as
                # lossless. The compactor's own apology can never become a
                # memory.
                if vllm_failed and conv_id:
                    # `vllm_failed` no longer decides anything; it is still
                    # worth ONE line, because "the tail ran on a reply the
                    # backend cut" and "the tail ran on a whole reply" look
                    # identical in the log otherwise, and the first is the
                    # case an operator is grepping for after an outage.
                    logger.warning(
                        f"conv={conv_id}: the backend failed during this "
                        f"stream; the memory tail is deciding on the "
                        f"{len(accumulator.text())} chars that did arrive"
                    )
                if conv_id:
                    _run_memory_tail(
                        conv_id,
                        accumulator.text(),
                        finished=accumulator.complete(),
                        truncated=accumulator.truncated(),
                        holed=accumulator.holed(),
                        touched_facts=touched_facts,
                        last_user_text=last_user_text,
                        turn_index=turn_index,
                        messages=messages,  # original request messages, for rollup
                        injected_facts=injected_facts,
                    )

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # Non-streaming path
    try:
        try:
            r = await client.post(f"{VLLM_URL}/v1/chat/completions", json=body)
        except httpx.RequestError as e:
            # V2.3 Theme 2: vLLM unreachable (down / restarting). Clean 503,
            # not an opaque 500. No async tail — there's no assistant turn.
            logger.warning(f"vLLM unreachable (non-stream): {type(e).__name__}: {e}")
            return JSONResponse(
                content=_vllm_unreachable_body(f"{type(e).__name__}: {e}"),
                status_code=503,
            )
        try:
            response_json = r.json()
        except ValueError as e:
            # vLLM (or something in front of it) returned a non-JSON body — an
            # HTML 502, a truncated response, a plain-text 5xx. Without this
            # guard the JSONDecodeError escapes as an opaque 500; httpx's
            # RequestError above only covers connection-level faults.
            body_head = (r.text or "")[:200]
            logger.warning(
                f"vLLM returned non-JSON (HTTP {r.status_code}): {type(e).__name__}: {body_head!r}"
            )
            return JSONResponse(
                content=_vllm_unreachable_body(
                    f"non-JSON response (HTTP {r.status_code}): {body_head}"
                ),
                status_code=502,
            )
        if r.status_code >= 400:
            # Return BEFORE the memory tail, and say so at ERROR.
            #
            # v3.1 F20: the status check used to sit after the tail was fired,
            # so a rejected request still ran the tail — harmless only by
            # accident, because assistant_text happens to come out empty and
            # every job in the tail happens to gate on it. One shape does get
            # through even today: with extraction disabled the tail takes
            # conv_lock and rewrites the facts file for a turn the model never
            # answered. A request the backend refused has nothing to remember.
            #
            # The relay itself is unchanged — this path already hands the
            # client vLLM's real status, which is why the incident's invisible
            # failure was the STREAM path and not this one. What was missing
            # here is the same thing: a line naming the conversation and the
            # counts. (This is also the path OpenWebUI's background title/tag
            # tasks take, so conv_id is often None; the line still says which.)
            sent_tokens, sent_source = await run_in_threadpool(
                _sent_token_size, body["messages"]
            )
            _log_request_rejected(
                conv_id, r.status_code, str(response_json), sent_tokens,
                sent_source, enforced_limit, streaming=False,
            )
            # v3.1 A8: same fix as the streaming path — the limit the guard
            # enforced, not one reconstructed from HARD_INPUT_LIMIT. This path
            # still discards the return value: it relays vLLM's own body to the
            # client verbatim, so there is no compactor-authored message for
            # `tightened` to steer. The calibration still happens; only the
            # advice-to-the-user half is absent here.
            _note_backend_rejection(
                str(response_json)[:2000], enforced_limit,
                guard_measured_overflow=calibration_overflow,
            )
            return JSONResponse(content=response_json, status_code=r.status_code)

        # v3.1 A10: the counterpart to the rejection path above. Without a call
        # here the release logic in _note_backend_accepted is unreachable and
        # the margin stays monotonic exactly as it was before this branch.
        _note_backend_accepted()

        # Extract assistant text for fact extraction
        assistant_text = ""
        try:
            assistant_text = (
                response_json.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                or ""
            )
        except (IndexError, KeyError, TypeError) as e:
            # An unexpected response shape leaves assistant_text empty. Since
            # v3.1.4 decide_memory_tail skips an empty reply, so the exchange
            # is not memorized at all — which is indistinguishable from a
            # model that replied with silence unless we say so. Once per
            # process: this is the request path. (v3.1 P0-2b / F61.)
            if logsetup.log_once("nonstream.assistant_text"):
                logger.warning(
                    f"conv={conv_id}: could not read assistant text from the "
                    f"vLLM response ({type(e).__name__}: {e}); this turn "
                    f"will not be memorized"
                )
        # The same policy the streaming path applies, through the same helper
        # (_run_memory_tail / decide_memory_tail). This path has no complete()
        # analogue — a non-streaming response is whole by construction — so
        # `finished` is True here and the only cut it can see is
        # finish_reason=length. Until v3.1.4 this site had no finish_reason
        # check at all (a reply cut at the ceiling was memorized as complete),
        # then an if/elif/elif that its streaming twin did not share; the
        # shared helper is what stops the two drifting a nineteenth time.
        _finish_reason = ""
        try:
            _finish_reason = (
                response_json.get("choices", [{}])[0].get("finish_reason") or ""
            )
        except (IndexError, KeyError, TypeError):
            _finish_reason = ""
        if conv_id:
            _run_memory_tail(
                conv_id,
                assistant_text,
                finished=True,
                truncated=_finish_reason == "length",
                holed=False,
                touched_facts=touched_facts,
                last_user_text=last_user_text,
                turn_index=turn_index,
                messages=messages,  # original request messages, for rollup
                injected_facts=injected_facts,
            )
        return JSONResponse(content=response_json, status_code=r.status_code)
    finally:
        await client.aclose()


@app.get("/v1/models")
async def models():
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(f"{VLLM_URL}/v1/models", timeout=30.0)
        except httpx.RequestError as e:
            # V2.3 Theme 2: clean 503 when vLLM is down/restarting.
            logger.warning(f"vLLM unreachable (/v1/models): {type(e).__name__}: {e}")
            return JSONResponse(
                content=_vllm_unreachable_body(f"{type(e).__name__}: {e}"),
                status_code=503,
            )
        try:
            models_body = r.json()
        except ValueError as e:
            # Same non-JSON-body class as the chat path (rc6 review): an HTML
            # 502 or truncated body must not escape as an opaque 500.
            body_head = (r.text or "")[:200]
            logger.warning(
                f"vLLM returned non-JSON (/v1/models, HTTP {r.status_code}): "
                f"{type(e).__name__}: {body_head!r}"
            )
            return JSONResponse(
                content=_vllm_unreachable_body(
                    f"non-JSON response (HTTP {r.status_code}): {body_head}"
                ),
                status_code=502,
            )
        return JSONResponse(content=models_body, status_code=r.status_code)


@app.get("/health")
async def health_liveness():
    """Cheap liveness probe — no I/O, no dependencies. For load balancers
    and quick `is-this-process-up` checks. Use /health/full for the deep
    probe that actually walks vLLM + storage.
    """
    return {"status": "ok", "vllm_url": VLLM_URL, "target_tokens": TARGET_TOKENS}


@app.get("/health/full")
async def health_full(response: Response):
    """V2.1 Phase 6: deep health probe.

    Walks vLLM reachability + storage writability + memory store stats.
    Returns 200 for ok/degraded, 503 for down. After this phase, the
    Docker HEALTHCHECK targets /health/full so the container goes
    unhealthy when vLLM is FATAL (today's `curl :3000` check stays
    healthy even when vLLM is dead, because OpenWebUI keeps serving
    its login page).
    """
    # tokenize_health() existed, was tested, and had no consumer — so a
    # /tokenize outage, the exact degraded mode the 2026-08-28 incident ran
    # in, was invisible on the health endpoint. health.py cannot import main
    # (main imports health), so main hands it in.
    report = await health.gather_health_full(
        VLLM_URL, TARGET_TOKENS, tokenize=tokenize_health()
    )
    response.status_code = health.status_to_http_code(report["status"])
    return report


# ---------------------------------------------------------------------------
# V2.0 admin/observability endpoints (Phase 1 + Phase 2)
# ---------------------------------------------------------------------------

@app.get("/admin/conversations", dependencies=[Depends(_require_localhost)])
async def admin_list_conversations():
    """List every conv_id that has any V2 state on disk."""
    return {"conversations": list_known_conv_ids()}


@app.get(
    "/admin/conversations/{conv_id}",
    dependencies=[Depends(_require_localhost)],
)
async def admin_conversation_summary(conv_id: str):
    """Per-conv inventory: file presence + sizes + per-layer memory stats.
    Phase 2 adds facts count, Phase 3 adds episodic doc count, Phase 4 adds
    the hierarchical summary state shape.
    """
    # v3.1 P0-2b: every handler below reports a null/empty layer on a read
    # error, which reads as "this conversation has no memory" to whoever is
    # inspecting it — and the person inspecting it is, by definition, doing so
    # during an incident. The response shape is unchanged (that is the D1/F5
    # per-layer {ok,error} work); what changes is that the log now says the
    # difference between empty and unreadable. Once per call site: the endpoint
    # is per-request, and the JSON body carries the per-call signal.
    info = storage_summary(conv_id)
    # Facts (Phase 2)
    try:
        info["facts"]["count"] = len(facts.load_facts(conv_id))
    except Exception as e:
        if logsetup.log_once("admin.conv_summary.facts"):
            logger.warning(
                f"conv={conv_id}: facts unreadable ({type(e).__name__}: {e}); "
                f"/admin/conversations reports count=null, which is NOT the "
                f"same as zero facts"
            )
        info["facts"]["count"] = None
    # Episodic memory (Phase 3)
    try:
        info["episodic"] = {
            "indexed_exchanges": retrieval.conversation_doc_count(conv_id),
        }
    except Exception as e:
        if logsetup.log_once("admin.conv_summary.episodic"):
            logger.warning(
                f"conv={conv_id}: episodic count unreadable "
                f"({type(e).__name__}: {e}); reported as null, not zero"
            )
        info["episodic"] = {"indexed_exchanges": None}
    # Hierarchical summary (Phase 4)
    try:
        info["summary"] = summarizer.state_summary(summarizer.load_state(conv_id))
    except Exception as e:
        if logsetup.log_once("admin.conv_summary.summary"):
            logger.warning(
                f"conv={conv_id}: summary state unreadable "
                f"({type(e).__name__}: {e}); reported as null, not absent"
            )
        info["summary"] = None
    # Persona (V2.1 Phase 8)
    try:
        prec = persona.load_persona(conv_id)
        info["persona"] = {
            "present": prec is not None,
            "length": len(prec["persona_text"]) if prec else 0,
            "source": prec["source"] if prec else None,
        }
    except Exception as e:
        if logsetup.log_once("admin.conv_summary.persona"):
            logger.warning(
                f"conv={conv_id}: persona unreadable ({type(e).__name__}: "
                f"{e}); reported as present=false, which is NOT the same as "
                f"no persona stored"
            )
        info["persona"] = {"present": False, "length": 0, "source": None}
    return info


@app.get(
    "/admin/conversations/{conv_id}/facts",
    dependencies=[Depends(_require_localhost)],
)
async def admin_get_facts(conv_id: str):
    """Return the current facts list for inspection / debugging."""
    return {"conv_id": conv_id, "facts": facts.load_facts(conv_id)}


@app.delete(
    "/admin/conversations/{conv_id}/facts",
    dependencies=[Depends(_require_localhost)],
)
async def admin_forget_facts(conv_id: str, request: Request):
    """Forget ALL memory for a conversation (V2.0 granularity: all-or-
    nothing). Clears persistent facts (Phase 2), episodic embeddings
    (Phase 3), AND the hierarchical summary state (Phase 4) — a full
    three-layer memory reset for when the model is stuck on something
    wrong. Targeted forgetting (single fact by substring) is V2.1.

    Takes no body and no query key (hostile pass 5, C5-5): either is a
    400, not a silently-ignored stray.
    """
    raw_body, body = await _parse_admin_json_body(request)
    _refuse_bad_admin_request(
        request, raw_body, body, body_keys=set(), query_keys=set(),
    )
    # DRAIN FIRST, exactly as the chat /forget does (RACE-01, adversarial
    # concurrency sweep, reproduced 10/10 and 5/5 in the forced case).
    #
    # This endpoint used to call _clear_all_memory bare while its twin —
    # commands._handle_forget — settled the background pool first, verified
    # the residue afterwards, and retried once. conv_lock cannot substitute:
    # _async_tail deliberately takes that lock THREE separate times so it
    # never holds it across an LLM call, so a wipe lands BETWEEN the tail's
    # jobs and the tail then writes the conversation back.
    #
    # Measured worst case: the endpoint answered HTTP 200 with all-zero
    # counters — 'there was nothing to forget' — and the fact, the episodic
    # row carrying the verbatim user turn and reply, and the position were
    # all on disk seconds later. For an endpoint whose entire purpose is to
    # make something gone, answering 'done, nothing there' while it is being
    # written back is the worst possible way to be wrong.
    #
    # One rule, two call sites, implemented at one. The same defect this
    # codebase keeps paying for, on the delete path.
    settled = await commands._settle_background_work()
    result = await _clear_all_memory(conv_id, source="admin")
    if isinstance(result, dict):
        # Say so rather than implying a guarantee that was not made. The
        # drain is best-effort by design (commands._settle_background_work
        # refuses to block a wipe the user asked for), so the honest answer
        # is whether it succeeded.
        result["background_settled"] = settled
    return result


# V2.1 Phase 5: shared full-clear used by /admin/forget AND the /forget
# chat command. Holding conv_lock here serializes against any in-flight
# extraction tail that might otherwise re-save state we just cleared.
async def _clear_all_memory(conv_id: str, *, source: str = "admin") -> dict:
    """Wipe every memory layer for a conv. Returns counters for the
    response body. `source` is just for log labeling."""
    async with conv_lock(conv_id):
        # v3.1: an unreadable facts file must not abort the whole wipe. The
        # user asked for this data to be gone; refusing to clear the three
        # layers we CAN read would leave more behind than clearing them does,
        # and would report failure for work that partly succeeded. Clear what
        # is readable, and say plainly which layer could not be.
        unreadable: list[str] = []
        try:
            existing = facts.load_facts(conv_id)
            n_facts = len(existing)
            if n_facts > 0:
                facts.save_facts(conv_id, [])
        except StoreUnreadable as e:
            n_facts = 0
            unreadable.append("facts")
            logger.error(
                f"conv={conv_id}: {source} forget could not read the facts file "
                f"({e}); the other memory layers were still cleared. The facts "
                f"file is left in place — it cannot be safely rewritten from an "
                f"unknown state."
            )
        # Episodic memory lives in ChromaDB.
        n_episodic = retrieval.forget_conversation(conv_id)
        # Hierarchical summary state on disk.
        summary_deleted = False
        try:
            sp = summarizer.summary_path(conv_id)
            if sp.is_file():
                sp.unlink()
                summary_deleted = True
        except Exception as e:
            logger.warning(f"conv={conv_id}: summary delete failed: {e}")
        # V2.1 Phase 8: persona is a memory layer too — full forget clears it.
        persona_deleted = False
        try:
            persona_deleted = persona.clear_persona(conv_id)
        except Exception as e:
            logger.warning(f"conv={conv_id}: persona delete failed: {e}")
        if n_facts or n_episodic or summary_deleted or persona_deleted:
            logger.info(
                f"conv={conv_id}: {source} forgot {n_facts} fact(s) "
                f"+ {n_episodic} indexed exchange(s) "
                f"+ summary={'cleared' if summary_deleted else 'absent'} "
                f"+ persona={'cleared' if persona_deleted else 'absent'}"
            )
    return {
        "conv_id": conv_id,
        "forgotten_facts": n_facts,
        "forgotten_episodic": n_episodic,
        "forgotten_summary": summary_deleted,
        "forgotten_persona": persona_deleted,
        # Present only when a layer could not be read. Callers must not
        # report a clean wipe when this is non-empty.
        "unreadable": unreadable,
    }


@app.get(
    "/admin/conversations/{conv_id}/summary",
    dependencies=[Depends(_require_localhost)],
)
async def admin_get_summary(conv_id: str):
    """Return the current hierarchical summary state (L1/L2/L3) for
    debugging. Localhost-only.
    """
    return summarizer.load_state(conv_id)


# V2.1 Phase 8: persona endpoints (localhost-only).
@app.get(
    "/admin/personas",
    dependencies=[Depends(_require_localhost)],
)
async def admin_list_personas():
    """Library view: list every conv that has a persona, with length
    and metadata. Does NOT include the full text — fetch per-conv for
    that. Useful for browsing "what persona was used in which conv?".
    """
    return {"personas": persona.list_personas()}


@app.get(
    "/admin/conversations/{conv_id}/persona",
    dependencies=[Depends(_require_localhost)],
)
async def admin_get_persona(conv_id: str):
    """Return the persona record (full text + metadata) for one conv.
    404 if no persona stored."""
    rec = persona.load_persona(conv_id)
    if not rec:
        raise HTTPException(status_code=404, detail="no persona for this conversation")
    return rec


@app.post(
    "/admin/conversations/{conv_id}/persona",
    dependencies=[Depends(_require_localhost)],
)
async def admin_set_persona(conv_id: str, request: Request):
    """Set or replace the persona for a conv.

    Body: {"text": "<persona text>"}. No query key is accepted, and no
    other body key (hostile pass 5, C5-5) — either is a 400.
    """
    raw_body = await request.body()
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="body must be JSON")
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")
    _refuse_unpaired_surrogate(body)
    _refuse_bad_admin_request(
        request, raw_body, body, body_keys={"text"}, query_keys=set(),
    )
    text = body.get("text")
    if not isinstance(text, str) or not text.strip():
        raise HTTPException(status_code=400, detail="missing required field: 'text' (non-empty string)")
    try:
        return persona.save_persona(conv_id, text, source="admin")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete(
    "/admin/conversations/{conv_id}/persona",
    dependencies=[Depends(_require_localhost)],
)
async def admin_delete_persona(conv_id: str, request: Request):
    """Clear the persona for a conv. Idempotent — returns deleted=False
    if no persona was stored.

    Takes no body and no query key (hostile pass 5, C5-5): either is a
    400, not a silently-ignored stray.
    """
    raw_body, body = await _parse_admin_json_body(request)
    _refuse_bad_admin_request(
        request, raw_body, body, body_keys=set(), query_keys=set(),
    )
    deleted = persona.clear_persona(conv_id)
    return {"conv_id": conv_id, "deleted": deleted}


@app.post(
    "/admin/conversations/{conv_id}/inherit-persona",
    dependencies=[Depends(_require_localhost)],
)
async def admin_inherit_persona(conv_id: str, request: Request):
    """Copy a persona from another conv (typically a 'base persona' conv)
    into this one. Useful for spinning up new conversations that should
    start with the same role/voice context as an existing one.

    Body: {"source_conv_id": "<conv_id to copy from>"}. No query key is
    accepted, and no other body key (hostile pass 5, C5-5) — either is a
    400.
    """
    raw_body = await request.body()
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="body must be JSON")
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")
    _refuse_unpaired_surrogate(body)
    _refuse_bad_admin_request(
        request, raw_body, body, body_keys={"source_conv_id"}, query_keys=set(),
    )
    src = body.get("source_conv_id")
    if not isinstance(src, str) or not src.strip():
        raise HTTPException(status_code=400, detail="missing required field: 'source_conv_id'")
    src_rec = persona.load_persona(src)
    if not src_rec:
        raise HTTPException(status_code=404, detail=f"no persona stored for source_conv_id={src!r}")
    saved = persona.save_persona(conv_id, src_rec["persona_text"], source="inherited")
    return {"conv_id": conv_id, "inherited_from": src, "persona": saved}


# V2.1 Phase 7 Step 2: stale-fact archival endpoints.
@app.get(
    "/admin/conversations/{conv_id}/archive",
    dependencies=[Depends(_require_localhost)],
)
async def admin_get_archive(conv_id: str):
    """Return the archived (cold-storage) facts for a conv. Useful for
    auditing what got demoted and deciding whether to restore."""
    return {"conv_id": conv_id, "archived": facts.load_archive(conv_id)}


@app.post(
    "/admin/conversations/{conv_id}/archive",
    dependencies=[Depends(_require_localhost)],
)
async def admin_archive_stale(conv_id: str, request: Request):
    """Trigger a stale-fact archival pass for one conv. Moves facts whose
    last_used is older than the cutoff to the archive sidecar.

    Query: ?older_than_days=N (default 90, env-overridable). No body is
    accepted.

    v3.1.9 (hostile pass 5, C5-5). Used to be a plain FastAPI-typed query
    param, which silently ignores anything it was not told to bind: a
    misspelled `?older_than_day=365` (missing the trailing "s") fell back
    to the 90-day default with no error, and this endpoint has NO dry-run
    mode at all, so `?dry_run=true&older_than_days=0` archived every stale
    fact live — the query string's dry intent was simply never read. Both
    are now a 400: `older_than_days` is the only accepted query key
    (typo'd or not, anything else — dry_run included — is unrecognised),
    and this endpoint takes no body key at all.
    """
    raw_body, body = await _parse_admin_json_body(request)
    _refuse_bad_admin_request(
        request, raw_body, body, body_keys=set(), query_keys={"older_than_days"},
    )
    _raw_days = request.query_params.get("older_than_days")
    if _raw_days is None:
        days = facts.ARCHIVE_DEFAULT_DAYS
    else:
        try:
            days = int(_raw_days)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"older_than_days must be an integer, got {_raw_days!r}",
            )
    async with conv_lock(conv_id):
        kept, archived = facts.archive_stale_facts(conv_id, older_than_days=days)
    return {
        "conv_id": conv_id,
        "older_than_days": days,
        "kept": kept,
        "archived": archived,
    }


@app.post(
    "/admin/conversations/{conv_id}/restore",
    dependencies=[Depends(_require_localhost)],
)
async def admin_restore_from_archive(conv_id: str, request: Request):
    """Move archived facts back to active storage.

    Body JSON — exactly one of:
        {"text_substring": "<non-empty substring filter>"}
            restores only archived facts whose text contains it
            (case-insensitive).
        {"restore_all": true}
            restores EVERY archived fact. Must be explicit and parsed
            STRICTLY (see _strict_affirmative, shared with `overwrite` on
            /admin/conversations/import): only `true`, `"true"`, `"1"` or
            `"yes"` ever turn this on.

    v3.1.9 (hostile pass 4, F3). Before this fix, an ABSENT body, an EMPTY
    body, a body this endpoint could not parse at all (curl's default
    form-encoding; a trailing comma; the USER_GUIDE.md example typed into
    Windows PowerShell/cmd, where the outer quoting strips the inner
    double-quotes and the body stops being JSON), a MISSPELLED key
    (`textSubstring`), or a `text_substring` that was null / "" / 0 / false
    (facts.restore_from_archive's own `if text_substring:` treats all of
    those the same as "no filter") ALL restored EVERY archived fact — 300+
    rows in the reviewer's proof, each one stamped `last_used: now` so they
    immediately outrank her real active facts for injection and pruning.
    None of those shapes restores anything now: a malformed/non-object body
    is a 400 (same shape as /compact and /admin/conversations/import), an
    unrecognised key is a 400, and restoring everything requires the
    explicit `restore_all` flag rather than being what happens when nothing
    else was understood.
    """
    raw_body = await request.body()
    if raw_body.strip():
        try:
            body = await request.json()
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"body is present but could not be parsed as JSON "
                    f"({type(e).__name__}: {e}); omit the body entirely, or "
                    f"send {{\"text_substring\": \"...\"}} or "
                    f"{{\"restore_all\": true}}"
                ),
            )
        if not isinstance(body, dict):
            raise HTTPException(
                status_code=400,
                detail=f"body must be a JSON object, got {type(body).__name__}",
            )
    else:
        body = {}
    _refuse_unpaired_surrogate(body)

    # v3.1.9 (hostile pass 5, C5-5). This used to be an ad hoc unknown-body-
    # key check with no duplicate-key check and no query-string check at
    # all — `/restore?dry_run=true` with `{"restore_all": true}` in the
    # body restored 300 archived facts, because the query-string dry intent
    # was never even read, and a duplicated `{"restore_all": false,
    # "restore_all": true}` resolved last-wins the same way. /restore has
    # NO dry-run mode at all, so query_keys is the empty set: a `dry_run`
    # key here, correctly spelled or not, is exactly as unrecognised as any
    # other stray key (dry_run_typo_exempt defaults False).
    _refuse_bad_admin_request(
        request, raw_body, body,
        body_keys={"text_substring", "restore_all"}, query_keys=set(),
    )

    # A present text_substring must be a real, non-empty string — not None
    # (absent is the normal way to ask for "no filter"), and not "", 0,
    # false, [] or a non-string, every one of which the OLD facts.py-level
    # `if text_substring:` check treated identically to "no filter", which
    # on THIS endpoint used to mean "restore all" (the exact bug). A
    # non-string value (e.g. `text_substring: 123`) is also what used to
    # 500 inside facts.py's `.lower()` call (F8c) — caught here instead.
    _raw_substring = body.get("text_substring")
    substring: str | None = None
    if _raw_substring is not None:
        if not isinstance(_raw_substring, str) or not _raw_substring.strip():
            raise HTTPException(
                status_code=400,
                detail=(
                    f"text_substring must be a non-empty string, got "
                    f"{_raw_substring!r}"
                ),
            )
        substring = _raw_substring

    restore_all = _strict_affirmative(
        body.get("restore_all", False), commit_tokens=_OVERWRITE_COMMIT_TOKENS
    )

    if substring is not None and restore_all:
        raise HTTPException(
            status_code=400,
            detail=(
                "text_substring and restore_all=true are mutually "
                "exclusive; send exactly one"
            ),
        )
    if substring is None and not restore_all:
        raise HTTPException(
            status_code=400,
            detail=(
                'specify either {"text_substring": "<non-empty string>"} '
                "to restore matching archived facts, or "
                '{"restore_all": true} to restore every archived fact '
                "explicitly. An absent, empty, or unparseable body no "
                "longer restores everything (hostile pass 4, F3)."
            ),
        )

    async with conv_lock(conv_id):
        restored = facts.restore_from_archive(
            conv_id, text_substring=substring,
        )
    return {
        "conv_id": conv_id,
        "restored": restored,
        "filter": substring,
        "restore_all": restore_all,
    }


# V2.1 Phase 7 Step 1: on-demand semantic deduplication.
@app.post(
    "/admin/conversations/{conv_id}/dedup",
    dependencies=[Depends(_require_localhost)],
)
async def admin_dedup(conv_id: str, request: Request):
    """Run a full hybrid (embedding + LLM) dedup pass on the conv's facts.

    Returns counters for the response body:
        {"conv_id", "before": int, "after": int, "removed": int}

    Inline dedup runs automatically after every fact extraction (cheap
    when no candidate clusters); this endpoint is for manual cleanup
    of conversations that pre-date Phase 7 or accumulated dupes via
    backfill/import. Takes no body and no query key (hostile pass 5,
    C5-5): either is a 400, not a silently-ignored stray.
    """
    raw_body, body = await _parse_admin_json_body(request)
    _refuse_bad_admin_request(
        request, raw_body, body, body_keys=set(), query_keys=set(),
    )
    async with conv_lock(conv_id):
        before = facts.load_facts(conv_id)
        if len(before) < 2:
            return {
                "conv_id": conv_id, "before": len(before),
                "after": len(before), "removed": 0,
            }
        async with httpx.AsyncClient() as client:
            after, removed = await dedup.dedup_facts(
                client, VLLM_URL, MODEL_REPO or "", before, conv_id=conv_id
            )
        if removed > 0:
            facts.save_facts(conv_id, after)
        return {
            "conv_id": conv_id,
            "before": len(before),
            "after": len(after),
            "removed": removed,
        }


# V2.1 Phase 6 Step 3: portability — export / import / fork.
@app.get(
    "/admin/conversations/{conv_id}/export",
    dependencies=[Depends(_require_localhost)],
)
async def admin_export_conversation(conv_id: str):
    """Snapshot one conv's full V2 state (facts + summary + episodic) as
    a single JSON bundle. Use for backup, cross-pod migration, or
    feeding a /admin/conversations/import on a different deploy.
    """
    return portability.export_conversation(conv_id)


# v3.1.9 HIGH #1 (hostile pass 3, F1). `bool(body.get("overwrite", False))`
# reads any non-empty JSON string as truthy, so a templated client sending
# `"overwrite": "false"` (or "no", "0", "off" — everything a shell/jq
# `"$OVERWRITE"` substitution produces from an unset or literally-"false"
# variable) got `bool("false") is True` and the import OVERWROTE a live
# conversation it was explicitly told not to touch: the facts file and
# summary state replaced wholesale, then the episodic index emptied. This is
# the exact truthiness shape f0a5aba fixed for `dry_run` and that fix was
# never carried to the one admin boolean whose "yes" reading is destructive.
#
# Same rule as _dry_run_from, same direction of safety: only a value that
# POSITIVELY spells an affirmative ever flips the flag on. Every other
# shape — wrong type, empty string, unrecognised spelling, absent — reads
# as the SAFE side (False / no-overwrite) rather than being coerced. For
# `overwrite` that is the opposite polarity from `dry_run` (True is the safe
# reading there; False is the safe reading here), but it is the identical
# "ambiguous is never a license to do the dangerous thing" rule.
_OVERWRITE_COMMIT_TOKENS = ("true", "1", "yes")


def _strict_affirmative(value: Any, *, commit_tokens: tuple[str, ...]) -> bool:
    """True only if `value` is bool True or a string that spells an
    affirmative in `commit_tokens` (case/whitespace-insensitive). Every other
    JSON shape — None, 0, [], {}, a float, an unrecognised string, an empty
    string — reads as False. There is no "ambiguous -> True" branch: unlike
    `bool()`, a present-but-unrecognised value never flips this on.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in commit_tokens
    return False


# v3.1.9 (hostile pass 4, F6). /compact and /merge-into each read a
# request as a fixed, small vocabulary of keys — {dry_run, max_calls} and
# {dry_run, refresh_last_used} respectively. A KEY the caller spelled wrong
# (`dry`, `dry_runs`, `is_dry_run`, `preview`, a nested
# `{"options": {"dry_run": true}}`, `?dryrun_mode=true`) matches none of
# the existing typo rules (those only catch near-spellings OF "dry_run"
# itself), so it used to read as "no opinion" — and on /compact, whose
# default is LIVE, the caller's own dry-intent key silently did nothing.
# Enumerating every way to almost spell "dry_run" is an unbounded list;
# refusing anything outside an endpoint's small, fixed vocabulary is not,
# and it catches every misspelling in one rule instead of one typo at a
# time.
#
# v3.1.9 (hostile pass 5, C5-5). This rule landed on /compact and
# /merge-into only — /restore, /import, /cleanup-test-data and /archive
# kept the old, permissive behaviour: `/restore?dry_run=true` with
# `restore_all` restored 300 facts (restore has no dry run and just
# ignored the key); `/archive?dry_run=true` archived live for the same
# reason. `dry_run_typo_exempt` is what makes the exemption above
# CONDITIONAL rather than global: True only for an endpoint that itself
# implements dry_run (checked via `_dry_run_from`) and therefore already
# gives a near-misspelling of "dry_run" its own, stricter, forced-dry
# handling — /compact, /merge-into, /cleanup-test-data. False (the
# default) is for every endpoint with NO dry_run concept at all: there a
# dry_run key, correctly spelled or not, is exactly as unrecognised as any
# other stray key and must be refused, never silently ignored the way it
# used to be.
def _refuse_unknown_keys(
    keys, allowed: set[str], *, where: str, dry_run_typo_exempt: bool = False,
) -> None:
    # A key that is a TYPO of "dry_run" (`dryRun`, `dry-run`, `dryrun`,
    # `dry_run[]`, a trailing-space/percent-encoded variant —
    # `_looks_like_misspelled_dry_run`, shared with `_dry_run_from`) is
    # exempt from "unknown" ONLY when `dry_run_typo_exempt` is True: it
    # already has its own, stricter handling (forced dry, never a 400 —
    # test_admin_compact.py's [9b] pins this end-to-end: `?dryrun=true`
    # must still answer 200 with `dry_run: true`, not a refusal). On an
    # endpoint that never calls `dry_run_typo_exempt=True`, this exemption
    # never applies, and a dry_run typo is just one more unrecognised key.
    extra = sorted(
        k for k in set(keys)
        if k not in allowed
        and not (dry_run_typo_exempt and _looks_like_misspelled_dry_run(k))
    )
    if extra:
        raise HTTPException(
            status_code=400,
            detail=(
                f"unrecognised key(s) in {where}: {extra}; only "
                f"{sorted(allowed)} are accepted here"
            ),
        )


# v3.1.9 (hostile pass 5, C5-5). The duplicate-key + unknown-key pair above
# was called from admin_compact and admin_merge only, each re-deriving its
# own raw-body-parse-and-refuse boilerplate around them. Every OTHER
# body- or query-reading admin write endpoint now goes through this ONE
# function instead — restore, import, fork, cleanup-test-data, dedup,
# forget, the persona endpoints, and the manual backup trigger — so the
# rule cannot be applied at one call site and missed at its sibling again,
# which is exactly how it was missed the first time (F6 fixed two of what
# was, even then, already more than two admin write endpoints).
#
# Deliberately does NOT also call `_refuse_unpaired_surrogate`: that guard
# must stay a call `test_surrogate_guard.py`'s structural check can find
# textually INSIDE each handler's own function body (it walks each
# function's own AST, not functions it calls), so every handler keeps that
# one line inline even after adopting this helper for everything else.
def _refuse_bad_admin_request(
    request: Request,
    raw_body: bytes,
    body: dict,
    *,
    body_keys: set[str],
    query_keys: set[str],
    dry_run_typo_exempt: bool = False,
) -> None:
    _refuse_duplicate_json_keys(raw_body)
    _refuse_unknown_keys(
        body.keys(), body_keys, where="body",
        dry_run_typo_exempt=dry_run_typo_exempt,
    )
    _refuse_unknown_keys(
        request.query_params.keys(), query_keys, where="the query string",
        dry_run_typo_exempt=dry_run_typo_exempt,
    )


# v3.1.9 (hostile pass 5, C5-5). For an admin write endpoint that never used
# to read its body or query string AT ALL (forget, dedup, delete-persona,
# the manual backup trigger) — so a stray key there was not "ignored", it
# was never looked at. This gives every one of them the SAME strict parse
# /restore, /import and /compact already have (hostile pass 3/4, F2/F3):
# an empty/whitespace body is the common "no body was sent" case and
# becomes `{}`; anything else that fails to parse, or parses to something
# other than a JSON object, is a 400 rather than silently treated as no
# body. Unlike `_refuse_bad_admin_request`, this DOES call
# `_refuse_unpaired_surrogate` itself — it is the only place these
# endpoints call `request.json()`, so it has to be, for
# test_surrogate_guard.py's structural check (every function that reads a
# JSON body also guards it) to find both calls together. The handlers this
# lane already found calling `request.json()` inline (persona, restore,
# import, fork, cleanup-test-data, merge-into, compact) keep doing that
# themselves rather than switching to this helper — no functional change
# for them, and no risk to that check's own bookkeeping.
async def _parse_admin_json_body(request: Request) -> tuple[bytes, dict]:
    raw_body = await request.body()
    if not raw_body.strip():
        body: dict = {}
    else:
        try:
            body = await request.json()
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"body is present but could not be parsed as JSON "
                    f"({type(e).__name__}: {e})"
                ),
            )
        if not isinstance(body, dict):
            raise HTTPException(
                status_code=400,
                detail=f"body must be a JSON object, got {type(body).__name__}",
            )
    _refuse_unpaired_surrogate(body)
    return raw_body, body


def _refuse_duplicate_json_keys(raw_body: bytes) -> None:
    """400 on a JSON body whose top-level object repeats a key.

    v3.1.9 (hostile pass 4, F6). `json.loads` (what `await request.json()`
    uses under the hood) resolves a duplicate key last-wins by default, so
    `{"dry_run": true, "dry_run": false}` silently becomes
    `{"dry_run": false}` with no trace either value ever disagreed — the
    same "an ambiguous value quietly wins" shape `_dry_run_from` exists to
    end for two DIFFERENT sources (body vs. query) disagreeing, one level
    under the parse itself, where one source disagrees with ITSELF.

    Only meaningful for a body that DOES parse as an object — a
    syntactically broken body is already a 400 from whatever primary parse
    the caller already ran (this function does not replace that parse, and
    is safe to call on an absent/empty body: nothing to check).
    """
    if not raw_body or not raw_body.strip():
        return

    def _hook(pairs):
        seen: set[str] = set()
        dupes: set[str] = set()
        for k, _ in pairs:
            if k in seen:
                dupes.add(k)
            seen.add(k)
        if dupes:
            raise ValueError(f"duplicate key(s) in JSON body: {sorted(dupes)}")
        return dict(pairs)

    try:
        json.loads(raw_body, object_pairs_hook=_hook)
    except ValueError as e:
        if "duplicate key" not in str(e):
            # Not what this function checks for — a syntax error here means
            # the caller's own primary parse should already have refused
            # this body elsewhere. Nothing to add.
            return
        raise HTTPException(status_code=400, detail=str(e))


@app.post(
    "/admin/conversations/import",
    dependencies=[Depends(_require_localhost)],
)
async def admin_import_conversation(request: Request):
    """Restore a conversation from a previously-exported bundle.

    Body JSON:
        {
          "bundle":          <bundle dict>,        // required
          "target_conv_id":  "<str>" | null,       // optional override
          "overwrite":       true | false (default)
        }

    Refuses if target conv has existing state unless overwrite=true —
    prevents accidental wipe of an active conversation. `overwrite` is
    parsed STRICTLY (see _strict_affirmative): only `true`, `"true"`, `"1"`
    or `"yes"` ever overwrite. Every other spelling, including `"false"`,
    `"no"`, `"0"` and `"off"`, is read as no-overwrite — the safe side.

    No query key is accepted, and no body key outside {bundle,
    target_conv_id, overwrite} (hostile pass 5, C5-5): this endpoint has
    NO dry-run mode, so a {"dry_run": true} alongside overwrite: true used
    to be silently ignored and the overwrite ran anyway — it is now a 400,
    like any other unrecognised key. A duplicated top-level key
    ({"overwrite": false, ..., "overwrite": true}) is a 400 too, rather
    than resolving last-wins toward the destructive value.
    """
    raw_body = await request.body()
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="body must be JSON")
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")
    _refuse_unpaired_surrogate(body)
    _refuse_bad_admin_request(
        request, raw_body, body,
        body_keys={"bundle", "target_conv_id", "overwrite"}, query_keys=set(),
    )
    bundle = body.get("bundle")
    if bundle is None:
        raise HTTPException(status_code=400, detail="missing required field: 'bundle'")
    overwrite = _strict_affirmative(
        body.get("overwrite", False), commit_tokens=_OVERWRITE_COMMIT_TOKENS
    )
    # v3.1.9 (F7 fix, hostile pass 4): validate the bundle, resolve the
    # target conv_id, and check the in-flight-writer lock BEFORE the
    # pre-overwrite snapshot below, not after. Before this fix, every one of
    # those three refusals (bad bundle version, facts not a list, a target
    # held by a live extraction tail) happened INSIDE import_conversation,
    # which ran AFTER a full quarantine snapshot had already been published
    # — so a refused import (a retry loop, a scripted health check, a client
    # resending a stale bundle) left one more never-pruned copy of the
    # conversation on disk every single time, for no output a bundle
    # validator alone could not have said in microseconds. This also fixes
    # F8(b)/(c): the SAME resolved (stripped, type-checked) target is now
    # used for both the snapshot and the eventual import — see
    # portability._validate_target_ready's docstring.
    #
    # UnsafeConvId alongside ImportError_ (v3.1.8): a body-supplied
    # target_conv_id / new_conv_id is CLIENT INPUT that reaches the
    # filesystem; memory._safe_path refuses to leave STORAGE_ROOT, and that
    # refusal is a 400 about the request, not a 500 about us.
    try:
        target = portability._validate_target_ready(
            bundle, target_conv_id=body.get("target_conv_id")
        )
    except (portability.ImportError_, UnsafeConvId) as e:
        raise HTTPException(status_code=400, detail=str(e))

    # v3.1.9 (F1 fix): take a quarantine copy before an overwrite lands, the
    # same reversibility cleanup's quarantine-then-wipe already gives a test
    # conv. import_conversation itself never had this — an overwrite replaced
    # facts/summary/episodic wholesale with nothing recoverable but a backup
    # cycle.
    #
    # A FAILED SNAPSHOT REFUSES THE OVERWRITE. v3.1.9 (hostile pass 4, F1)
    # removed the one case that used to proceed anyway: a facts layer that
    # was ALREADY unreadable (StoreUnreadable) no longer skips this snapshot
    # — the lane that first wrote this comment read "the store is
    # unreadable" from ONE layer raising and let the exemption through, but
    # a torn facts file leaves the summary hierarchy and episodic index
    # fully readable, and the overwrite that followed destroyed those too.
    # quarantine_conversation itself now absorbs a StoreUnreadable facts
    # read (records the layer unverified, copies the torn file's raw bytes
    # aside) instead of raising it, so this call site no longer needs — and
    # must not have — a StoreUnreadable exemption: any exception it still
    # raises (QuarantineError, or the raw bytes themselves being
    # uncopyable) means the snapshot genuinely could not be written, and the
    # overwrite is refused.
    if overwrite:
        try:
            portability.quarantine_conversation(
                target, reason="admin import overwrite"
            )
        except Exception as e:
            logger.error(
                f"conv={target}: pre-overwrite quarantine failed "
                f"({type(e).__name__}: {e}); overwrite refused"
            )
            raise HTTPException(
                status_code=409,
                detail=(
                    f"refusing to overwrite conv_id {target!r}: "
                    f"a restorable snapshot of its current state could not be "
                    f"written first ({type(e).__name__}: {e}). Nothing was "
                    f"changed. Fix the quarantine location, or export the "
                    f"conversation yourself, and retry."
                ),
            )
    try:
        result = portability.import_conversation(
            bundle,
            target_conv_id=body.get("target_conv_id"),
            overwrite=overwrite,
        )
    # Re-checked here too (not just above): _validate_target_ready runs
    # AGAIN inside import_conversation, immediately before the write, which
    # is what closes the TOCTOU the snapshot's own I/O opens (see that
    # function's docstring) — so this exception mapping stays reachable even
    # though the common failures were already caught above.
    except (portability.ImportError_, UnsafeConvId) as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result


@app.post(
    "/admin/conversations/{conv_id}/fork",
    dependencies=[Depends(_require_localhost)],
)
async def admin_fork_conversation(conv_id: str, request: Request):
    """Clone src conv's full state into a new conv_id. Original
    untouched. Body is optional:
        {"new_conv_id": "<str>" | null}
    If omitted, the fork's id is `<src>__fork_<6hex>`.

    No query key is accepted, and no body key outside new_conv_id
    (hostile pass 5, C5-5) — either is a 400.
    """
    # Body is optional — accept empty or missing.
    raw_body = await request.body()
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    _refuse_unpaired_surrogate(body)
    _refuse_bad_admin_request(
        request, raw_body, body, body_keys={"new_conv_id"}, query_keys=set(),
    )
    try:
        return portability.fork_conversation(
            conv_id, new_conv_id=body.get("new_conv_id")
        )
    # v3.1.8: UnsafeConvId alongside ImportError_. A body-supplied
    # target_conv_id / new_conv_id is CLIENT INPUT that reaches the
    # filesystem; memory._safe_path refuses to leave STORAGE_ROOT, and
    # that refusal is a 400 about the request, not a 500 about us.
    except (portability.ImportError_, UnsafeConvId) as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post(
    "/admin/conversations/cleanup-test-data",
    dependencies=[Depends(_require_localhost)],
)
async def admin_cleanup_test_conversations(request: Request):
    """Quarantine-then-remove the test/placeholder conversations polluting
    the store: 129 "conversations" for ~26 real ones, inflating
    /admin/conversations, the health stats and every backup archive.

    Body (optional): {"dry_run": true}
    Query (optional): ?dry_run=false

    DRY RUN BY DEFAULT. Matches only ids minted by selftest.py and the
    integration harness, and refuses any match that carries substantial
    memory (or whose layers cannot be read - unreadable counts as
    substantial, never as empty). Everything is quarantined before it is
    wiped, so this is reversible; nothing is unlinked.

    v3.1.9 (hostile pass 4, F8a). `dry_run` used to be a plain FastAPI
    `bool` query param, which is Starlette's own last-wins coercion over
    repeated values (`?dry_run=true&dry_run=false` COMMITS) and had no body
    form at all — a JSON `{"dry_run": true}` was silently ignored, exactly
    gate-review holes (a) and (b) that `_dry_run_from` was built to close,
    fixed there for /compact and /merge-into but never carried here. Reuses
    that same helper now: both sources are read, and dry wins on any
    disagreement, including a source disagreeing with itself.

    v3.1.9 (hostile pass 5, C5-5). No duplicate-key check existed either —
    a body of {"dry_run": true, "dry_run": false} resolved last-wins
    (False, commit) and wiped a matched conversation, directly
    contradicting this endpoint's own docstring above ("dry wins on any
    disagreement, including a source disagreeing with itself"). dry_run is
    the only accepted key, in either the body or the query string;
    dry_run_typo_exempt=True because this endpoint DOES have a dry_run
    (via _dry_run_from immediately below), so a near-misspelling of it
    already gets forced onto the safe side there instead of being an
    unrecognised key.
    """
    raw_body = await request.body()
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    _refuse_unpaired_surrogate(body)
    _refuse_bad_admin_request(
        request, raw_body, body, body_keys={"dry_run"}, query_keys={"dry_run"},
        dry_run_typo_exempt=True,
    )
    dry_run = _dry_run_from(request, body, default=True)

    async def _wipe(conv_id: str) -> dict:
        # Through commands._wipe_all_layers rather than _clear_all_memory
        # directly, so a cleanup leaves exactly what /forget leaves - the
        # archive-sidecar clear and the empty-facts tombstone included.
        return await commands._wipe_all_layers(
            conv_id, lambda cid: _clear_all_memory(cid, source="cleanup-test-data")
        )

    return await portability.cleanup_test_conversations(
        dry_run=dry_run, wipe_layers=_wipe
    )


# v3.1.9 HIGH #1/#2 (hostile pass 2). Tokens that mean COMMIT, in both the
# query string and the JSON body — one vocabulary, not two. Everything that
# is not one of these three, in either dialect, is DRY: see _dry_run_from's
# docstring for why ambiguous now always means dry rather than "whatever
# `default` says".
_DRY_RUN_COMMIT_TOKENS = ("false", "0", "no")

# v3.1.9 HIGH (hostile pass 3, F2). One check for "this key is trying to say
# dry_run and getting it wrong", shared by the query-string side (which had
# it) and the body side (which did not until this fix). Strips every
# character that is not a letter or digit before comparing, not just "-" and
# "_": the old query-only version stripped only those two, so `dry_run[]`
# (what `?dry_run[]=true` sends — some HTTP clients array-ify a repeated-flag
# convention) and `dry_run ` (a trailing space from `?dry_run%20=true`)
# neither matched "dry_run" nor got caught as a typo, and fell through to
# "absent" — on /compact, "absent" is the live default. Comparing against the
# alnum-stripped form catches those alongside the case/hyphen/space variants
# the original handled.
def _looks_like_misspelled_dry_run(key: str) -> bool:
    return key != "dry_run" and re.sub(r"[^a-z0-9]", "", key.lower()) == "dryrun"


def _dry_run_from(request: Request, body: dict, *, default: bool) -> bool:
    """Read dry_run from the body, then the QUERY STRING, then the default.

    Both admin endpoints that take it are reachable by curl, and an operator
    reaching for one reaches for `?dry_run=true` at least as often as for a
    JSON body. admin_merge learned to read the query string in v3.1.7 (R4);
    admin_compact did not, and its default is the OPPOSITE — a live run — so
    `POST /admin/conversations/<id>/compact?dry_run=true` silently performed up
    to 200 vLLM summarization calls, rewrote the state file and advanced the
    watermark. The operator asked for a plan and got a write. admin_merge's own
    docstring already named it: "unlike the compact endpoint next door, which
    defaults to a live run and surprised an operator into one."

    One function, because the rule that was applied at one site and missed at
    its sibling is this project's most expensive recurring defect, and two
    copies of this parsing would be a third instance waiting to happen.

    v3.1.9 HIGH #1 (hostile pass 2). The rule above turned out to have a hole
    of the identical shape one level down: `?dry_run=` (what
    `curl ".../compact?dry_run=$FLAG"` sends when $FLAG is unset), a bare
    `?dry_run` (how most CLIs spell a boolean flag), and
    `?dry_run=true&dry_run=` (Starlette's QueryParams.get is last-wins, so the
    empty repeat silently discards the `true`) are all PRESENT — not absent —
    query strings, and the old code returned `default` for an empty raw value
    exactly like it did for a missing key. On /compact, default is a live
    run: an operator who typed a flag at all, however malformed, got the
    write they were explicitly trying not to get. `default` is what an ABSENT
    flag means; it is no longer what an empty or malformed PRESENT one means.
    A key that only differs from "dry_run" by case or a hyphen (?dryrun=,
    ?dry-run=) gets the same treatment as an empty value, because a typo in
    the key is not an absent flag either and must not silently commit.

    v3.1.9 HIGH #2 (hostile pass 2). The body side had its own, incompatible
    dialect: `bool(body["dry_run"])` treats every JSON value present under
    the key as Python truthiness, so `{"dry_run": null}` / `""` / `0` / `[]`
    — all of which a templated client (`{"dry_run": $FLAG}` through jq or
    envsubst) produces from an unset variable — are FALSY and therefore
    COMMIT, while `{"dry_run": "false"}` is a non-empty string, therefore
    TRUTHY, therefore DRY — the opposite of what `?dry_run=false` does on the
    same endpoint. That is the exact defect shape this function exists to
    end, one level down inside its own body. Now both dialects share
    _DRY_RUN_COMMIT_TOKENS and the same rule: only a value that positively
    spells "commit" ever writes; every other shape — wrong type, empty
    string, unrecognised spelling — is dry, never a coercion.

    v3.1.9 gate review, two more holes of the same shape:

    (a) CONFLICTING REPEATED QUERY VALUES COMMIT. The first cut of this fix
    read `request.query_params.get("dry_run")`, which is last-wins, so
    `?dry_run=true&dry_run=false` read "false" and committed — an operator
    who sent both a true and a false in the same request got the write, not
    the safer of the two answers. Fixed by reading EVERY value with
    `getlist` and requiring ALL of them to be a commit token before the
    query source says commit; any empty, unrecognised, or disagreeing value
    in the list makes the query source say dry.

    (b) BODY AND QUERY DISAGREE -> THE BODY SILENTLY WON. The first cut
    returned from the body branch immediately, so `{"dry_run": false}` with
    `?dry_run=true` on the same request committed without the query string
    ever being consulted — an explicit dry in the query was overridden by
    the body with no indication either happened. Fixed by evaluating BOTH
    sources that are actually present to a verdict (dry/commit) and
    combining them: commit only if every PRESENT source says commit; if any
    present source says dry, the whole call is dry; `default` is read only
    when NEITHER source is present at all. This is the same rule as (a) one
    level up — a single source that disagrees with itself is treated exactly
    like two sources that disagree with each other.
    """
    def _body_verdict() -> bool | None:
        """True = dry, False = commit, None = the body carries no opinion
        (the "dry_run" key is simply absent, and not even a misspelled key)."""
        # v3.1.9 (hostile pass 3, F2). The query side had a misspelled-key
        # rule from the gate review (below); the body side did not, so
        # {"dryRun": true} / {"dry-run": true} / {"DRY_RUN": true} /
        # {"dry_run ": true} all left "dry_run" absent from `body`, fell
        # through with body_verdict=None, and — on /compact, whose default is
        # live — committed a write the caller's key spelled "dry_run" wrong
        # while trying to prevent. Same rule as the query side, one level up:
        # a key that is a typo of "dry_run" is not an absent flag.
        misspelled = any(_looks_like_misspelled_dry_run(k) for k in body.keys())
        if misspelled:
            return True  # ambiguous key: dry, regardless of what "dry_run" itself says
        if "dry_run" not in body:
            return None
        v = body["dry_run"]
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            s = v.strip().lower()
            if s:
                return s not in _DRY_RUN_COMMIT_TOKENS
            return True  # empty string: ambiguous, dry
        # None, 0, [], {}, a float, ... — every other JSON shape, including
        # the ones `bool(...)` used to read as "commit". Ambiguous PRESENT
        # values are dry now, never a write.
        return True

    def _query_verdict() -> bool | None:
        """True = dry, False = commit, None = the query string carries no
        opinion (absent, and not even a misspelled key)."""
        # A key that differs from "dry_run" only by case, a hyphen, a missing
        # separator, or bracket/percent-encoded noise (?dryrun=, ?dry-run=,
        # ?DRY_RUN=, ?dry_run[]=, ?dry_run%20=) is a typo, not an absent
        # flag, and must not fall through to `default` either — see the
        # docstring and _looks_like_misspelled_dry_run.
        misspelled = any(
            _looks_like_misspelled_dry_run(k) for k in request.query_params.keys()
        )
        if misspelled:
            return True  # ambiguous key: dry, regardless of what "dry_run" itself says
        if "dry_run" not in request.query_params:
            return None  # truly absent: no opinion
        # EVERY repeated value must be a commit token for the query source
        # to say commit — a single unrecognised, empty, or disagreeing value
        # anywhere in the list makes the whole source say dry. This is what
        # makes ?dry_run= (a value of "") and ?dry_run=true&dry_run=false
        # (values ["true", "false"]) both dry: "" and "true" are not commit
        # tokens, so "every value is a commit token" is already false.
        values = [str(v).strip().lower() for v in request.query_params.getlist("dry_run")]
        return not (len(values) > 0 and all(v in _DRY_RUN_COMMIT_TOKENS for v in values))

    body_verdict = _body_verdict()
    query_verdict = _query_verdict()
    if body_verdict is None and query_verdict is None:
        return default  # neither source has an opinion: fall back to the endpoint's own default
    verdicts = [v for v in (body_verdict, query_verdict) if v is not None]
    # Commit only if EVERY present source says commit (False). If any
    # present source says dry (True), the whole call is dry — a single
    # disagreeing source, whether that's two sources disagreeing with each
    # other or one source disagreeing with itself (a), always loses to dry.
    return any(verdicts)


@app.post(
    "/admin/conversations/{src_conv_id}/merge-into/{dst_conv_id}",
    dependencies=[Depends(_require_localhost)],
)
async def admin_merge(src_conv_id: str, dst_conv_id: str, request: Request):
    """Fold a forked conversation's memory back into the live one.

    Body (all optional):  {"dry_run": true}
    Query (all optional):  ?dry_run=false

    DRY RUN BY DEFAULT - unlike the compact endpoint next door, which
    defaults to a live run and surprised an operator into one. This touches
    two conversations, so it gets the safer default; pass
    {"dry_run": false} to commit.

    THE QUERY FORM IS READ TOO, and that is not a convenience (v3.1.7, R4).
    The install runbook in pipelines/conversation_id_header.py told the
    operator to commit with `?dry_run=false`, this handler read the JSON body
    only, and the query string was silently ignored - so the step whose entire
    purpose is to un-fork her memory returned HTTP 200 with plausible counts
    and changed nothing. It is the step before the history cap goes on, and
    capping before a real merge is what makes the loss permanent.

    A POST with a query flag is what an operator reaches for under stress, and
    a flag that is accepted-looking and inert is worse than one that 400s.
    v3.1.9 (hostile pass 3, F12): when body and query DISAGREE, DRY wins, not
    the body — commit only if every source that is present says commit; see
    _dry_run_from's docstring (b). A body {"dry_run": false} next to
    `?dry_run=true` stays dry, the safer of the two answers, not a silent
    override in either direction.

    Merges FACTS and EPISODIC exchanges only. Summaries are deliberately not
    merged: the forked half re-derives its own hierarchy from the client's
    full array, so dst already covers the same history and folding src's in
    would double-count it. The source is left completely intact, so a merge
    that produces a bad result costs nothing but the re-embedding.

    `refresh_last_used` (body or query, default false — v3.1.9, hostile pass
    3, F6): opt-in only. Pass true for the id-migration/backfill recovery
    (dst already holds a fresh backfill's re-extractions, src holds her real
    older originals) — NOT for the runbook's "Older forks" step or any merge
    into a conversation that is still being chatted in, where it would
    outrank her own facts with the fork's. See portability.merge_conversation
    for the full contract and why this stopped being automatic.

    See portability.merge_conversation for the full contract.
    """
    _raw_body = await request.body()
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    _refuse_unpaired_surrogate(body)
    # v3.1.9 (hostile pass 4, F6 sibling sweep). Checked AFTER the surrogate
    # guard (which must stay the first refusal every body-reading handler
    # gives, per test_surrogate_guard.py's structural check) and only when
    # the body DID parse as an object — a malformed body already falls back
    # to {} above, which is safe here because merge's default is DRY,
    # unlike /compact's.
    if isinstance(body, dict) and body:
        _refuse_duplicate_json_keys(_raw_body)
    # dry_run_typo_exempt=True: merge-into DOES implement dry_run (below,
    # via _dry_run_from), so a near-misspelling of it already gets forced
    # onto the safe (dry) side there rather than being an unrecognised key
    # (hostile pass 5, C5-5 — see _refuse_unknown_keys' own docstring).
    _refuse_unknown_keys(
        body.keys(), {"dry_run", "refresh_last_used"}, where="body",
        dry_run_typo_exempt=True,
    )
    _refuse_unknown_keys(
        request.query_params.keys(), {"dry_run", "refresh_last_used"},
        where="the query string", dry_run_typo_exempt=True,
    )
    # Absent means DRY for merge: this endpoint rewrites two conversations
    # and an operator who meant to commit sees unchanged counts and tries
    # again, while the reverse mistake is not recoverable.
    dry_run = _dry_run_from(request, body, default=True)
    # v3.1.9 (hostile pass 3, F6). Same strict-affirmative parsing as
    # `overwrite` (_strict_affirmative): only true/"true"/"1"/"yes" from
    # EITHER source ever turns this on; every other shape is the safe
    # default (no floor). Either source asking for it is enough — unlike
    # dry_run, this is not a "which side of a disagreement wins" question,
    # because leaving it off is never the more dangerous reading.
    refresh_last_used = _strict_affirmative(
        body.get("refresh_last_used", False), commit_tokens=_OVERWRITE_COMMIT_TOKENS
    ) or _strict_affirmative(
        request.query_params.get("refresh_last_used"), commit_tokens=_OVERWRITE_COMMIT_TOKENS
    )

    try:
        return await run_in_threadpool(
            portability.merge_conversation,
            src_conv_id,
            dst_conv_id,
            dry_run=dry_run,
            refresh_last_used=refresh_last_used,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# v3.1.7 (R13). What fills a slot the episodic store has no row for.
#
# THE DISTINCTION THAT MAKES THIS ALLOWED. This string is a SUMMARIZATION
# INPUT and nothing else. It is built here, handed to summarizer.maybe_rollup,
# and dropped; it never reaches facts.extract_facts_from_exchange, never
# reaches retrieval.index_exchange, and is never written to any store. A
# marker written INTO memory would be extracted as a fact and become one of
# her memories — that is why this file has no other placeholders. The only
# thing this one can do is make a summary say that part of the transcript was
# missing, which is true.
#
# Deliberately non-blank, for _DEGENERATE_HISTORY_PLACEHOLDER's reason:
# _do_l1_rollup skips a chunk whose every piece is blank, and a gap that
# silently empties a chunk is not the same failure as one that says so.
_UNINDEXED_TURN_PLACEHOLDER = (
    "[this turn was not recorded in the episodic index and could not be "
    "rebuilt]"
)


def _rebuild_transcript_by_slot(rows: list[dict]) -> tuple[list[dict], int]:
    """The episodic rows laid out at their own turn POSITIONS, gaps filled.

    Returns (messages, gap_turns).

    v3.1.7 (R13). The old rebuild concatenated the rows it had. That is fine
    for reading them back and wrong for summarizing them: since v3.1.4 the
    summarizer locates a chunk's text at `position - len(window)` turns into
    the array it is handed, so a transcript that is SHORT by the exchanges the
    tail skipped and the pool shed does not merely stop early — every turn
    after the first gap sits at the wrong index, and a chunk labelled 652-671
    holds some other twenty turns. `admin_compact` was therefore right to 409
    on it, and 409'd for every real conversation (63 skips in one measured
    window; one gap anywhere is enough).

    So place each pair where it belongs and leave the holes visible.

    WHY THE SLOTS ARE RELATIVE TO THE FIRST ROW, not absolute. `turn_index` is
    NOT an exact position and never was: the request path sets it to
    `len(messages) + 1`, which counts system messages the summarizer's own
    numbering skips, and retrieval._next_turn_index then reallocates it as
    `max(stored_max + 2, seed)`. What IS exact is its DIFFERENCES — both the
    request seed and the store's allocator advance by exactly
    _TURN_INDEX_STEP (2) message-units per exchange, whether or not a row was
    written — so a jump of 4 is one lost exchange, reliably, in either
    numbering. Anchoring on the lowest stored index and spacing by differences
    therefore reproduces the store's own gaps without inheriting either
    scheme's offset.

    That anchoring assumes the conversation's FIRST exchange is in the store.
    The assumption is self-checking rather than trusted: if the head is
    missing, the rebuild comes up short of the recorded position and the
    caller's 409 fires — the same refusal, for the same reason, without the
    endpoint having to detect the case.

    A row whose document does not parse leaves its slot as a gap rather than
    vanishing. Vanishing is what shifted everything after it.
    """
    def _idx(ex: dict) -> int:
        try:
            return int(ex.get("turn_index") or 0)
        except (TypeError, ValueError):
            return 0

    ordered = sorted(rows, key=_idx)
    if not ordered:
        return [], 0
    base = _idx(ordered[0])
    pairs: dict[int, tuple[str, str]] = {}
    cursor = 0
    for ex in ordered:
        at = max(_idx(ex) - base, cursor)
        # Pairs sit on even offsets. An ODD difference means the two
        # numberings disagreed about a system message somewhere; rounding up
        # costs one placeholder pair and keeps user/assistant alternation,
        # which the Mistral template requires and _turn_pieces assumes.
        if at % 2:
            at += 1
        cursor = at + 2
        # _exchange_doc writes "[user]: X\n[assistant]: Y". Split it back into
        # the message pair the summarizer expects.
        doc = ex.get("document") or ""
        if "\n[assistant]: " not in doc:
            continue
        u, a = doc.split("\n[assistant]: ", 1)
        pairs[at] = (u.removeprefix("[user]: "), a)

    messages: list[dict] = []
    gap_turns = 0
    for at in range(0, cursor, 2):
        pair = pairs.get(at)
        if pair is None:
            messages.append(
                {"role": "user", "content": _UNINDEXED_TURN_PLACEHOLDER})
            messages.append(
                {"role": "assistant", "content": _UNINDEXED_TURN_PLACEHOLDER})
            gap_turns += 2
        else:
            messages.append({"role": "user", "content": pair[0]})
            messages.append({"role": "assistant", "content": pair[1]})
    return messages, gap_turns


@app.post(
    "/admin/conversations/{conv_id}/compact",
    dependencies=[Depends(_require_localhost)],
)
async def admin_compact(conv_id: str, request: Request):
    """Clear a conversation's summarization backlog OFF the request path.

    Why this exists, precisely. Compaction runs inside chat_completions, so a
    conversation whose rollups have been failing accumulates a backlog that the
    NEXT user message has to pay for. On 2026-08-29 that was 170 turns needing
    33 summarization calls: eight minutes of a dead composer, and she got no
    reply at all. MAX_SUMMARY_CALLS_PER_REQUEST now bounds that, which means a
    large backlog drains slowly over many turns instead of stalling one — and
    this endpoint is how an operator drains it deliberately instead of waiting.

    Self-healing was the wrong shape for this. Catch-up work belongs to whoever
    is willing to wait for it.

    Body (all optional):
        {"max_calls": 200, "dry_run": false}
    Query form also honoured (v3.1.9, hostile pass 4, F5):
        ?max_calls=200&dry_run=false
    No other body or query key is accepted — an unrecognised one is a 400
    rather than silently ignored.

    `max_calls` bounds REAL vLLM summarization calls (hostile pass 4, F5) —
    `{"max_calls": 1}` makes at most one vLLM HTTP call for a backlog whose
    next unit (one L1 chunk, one L2 fold, or the L3 refresh) costs one call,
    however deep the backlog is BEHIND that unit, via
    `summarizer.vllm_call_budget_ctx` wrapping the whole drain below. This
    closes the earlier hole where `max_calls` counted PASSES (calls to
    summarizer.maybe_rollup) instead: one pass drains every L1 and L2 tier
    due in its own internal loop, so a deep backlog could spend far more
    than `max_calls` real calls in a single pass. The response reports
    both: `vllm_calls` is the number this parameter now actually bounds;
    `rollup_calls` (unchanged) is the number of PASSES this loop itself
    made, kept for existing callers that read it that way.

    v3.1.9 (tail catch-up): the budget is enforced at the UNIT boundary, not
    per real call — see `summarizer._budget_allows_unit`'s docstring for
    why a per-call bound livelocks whenever a unit costs more than one call
    (a strict `{"max_calls": 1}` against a 3-call L1 chunk never advanced,
    forever). The consequence here: `max_calls` is a bound with a
    documented overshoot of AT MOST one unit's own calls, never unbounded —
    once a unit has been allowed to start it always finishes, and no
    FURTHER unit starts once the budget reads empty. `{"max_calls": 0}`
    still means exactly what hostile pass 2 fixed it to mean — "run the
    guards, make no real calls" — because 0 never has room to start even
    one unit.

    The transcript is reconstructed from the EPISODIC store, which is the only
    ordered record of the conversation the compactor owns — OpenWebUI holds the
    real one. So this can only summarize exchanges that were successfully
    indexed. It reports what it found rather than pretending that is the whole
    conversation.

    Since v3.1.7 (R13) that reconstruction is BY SLOT: each pair sits at the
    position its `turn_index` records, and an exchange the store never indexed
    becomes an explicit placeholder pair instead of a hole that shifts every
    later turn one place left. `gap_turns` / `gap_exchanges` in the plan say
    how much of the transcript is placeholder. Refusals are now only for the
    two things padding cannot fix: a store that does not reach the recorded
    position, and one with more gap than transcript.
    """
    # v3.1.9 HIGH (hostile pass 3, F2). The old code was
    # `try: body = await request.json() except Exception: body = {}` /
    # `if not isinstance(body, dict): body = {}` — so a body that could not
    # be read AT ALL (invalid JSON, a JSON value that is not an object, or
    # `request.json()` itself raising — e.g. a 5000-digit integer, which
    # trips CPython's int-string conversion limit inside json.loads) was
    # treated EXACTLY like no body being sent. On /compact, whose documented
    # default is LIVE, that means every one of these silently ran the drain:
    # form-encoded `dry_run=true` (curl's default content-type), single- or
    # un-quoted pseudo-JSON, a trailing comma, a JSON array instead of an
    # object, and the 5000-digit-int case. The operator asked this endpoint
    # for a plan and got up to `max_calls` live vLLM summarization calls that
    # rewrote the state file and advanced the watermark.
    #
    # The fix distinguishes "no body was sent" (a real, common, and safe
    # case — the documented live default applies) from "a body was sent and
    # this endpoint could not read it" (which must never be silently treated
    # as if the caller had said nothing): read the raw bytes first, and only
    # a body that is empty (or all whitespace) collapses to {}. Anything
    # else that fails to parse, or parses to something other than a JSON
    # object, is a 400 — the same shape every other admin endpoint in this
    # file already uses for a body it was actually given.
    #
    # Deliberately still `await request.json()` below, not a bare
    # `json.loads(raw_body)`: Starlette caches the body on first read, so
    # this re-reads the same bytes `request.body()` already fetched (no
    # second I/O) — and test_surrogate_guard.py's structural check (every
    # handler that calls `request.json()` must also call
    # `_refuse_unpaired_surrogate`) finds this handler by that exact call,
    # the same way it finds every sibling admin endpoint. Swapping in
    # `json.loads` directly would silently drop this handler out of that
    # audit's coverage — the AST detector has no way to know a differently-
    # spelled parse call still needs the same guard.
    raw_body = await request.body()
    if raw_body.strip():
        try:
            body = await request.json()
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"body is present but could not be parsed as JSON "
                    f"({type(e).__name__}: {e}); omit the body entirely for "
                    f"the documented live default, or send a JSON object"
                ),
            )
        if not isinstance(body, dict):
            raise HTTPException(
                status_code=400,
                detail=f"body must be a JSON object, got {type(body).__name__}",
            )
    else:
        body = {}  # truly absent body: the documented live default applies
    _refuse_unpaired_surrogate(body)
    # v3.1.9 (hostile pass 4, F6). Checked AFTER the surrogate guard (which
    # stays the first refusal every body-reading handler gives) and only on
    # a non-empty body — `{}` trivially has no duplicate keys. This is the
    # F2 (pass 3) class one level down: a body that parses fine but repeats
    # a key, or spells a dry-intent key this endpoint does not recognise
    # (`dry`, `dry_runs`, `is_dry_run`, `preview`, `{"options": {"dry_run":
    # true}}`) used to read as "no opinion" and LIVE (this endpoint's
    # default) applied — the caller's own key silently did nothing. /compact
    # accepts exactly two keys; anything else, in the body OR the query
    # string, is refused rather than enumerated as one more typo to catch.
    if body:
        _refuse_duplicate_json_keys(raw_body)
    # dry_run_typo_exempt=True: /compact DOES implement dry_run (via
    # _dry_run_from below), so a near-misspelling of it already gets forced
    # onto the safe (dry) side there rather than being an unrecognised key
    # (hostile pass 5, C5-5 — see _refuse_unknown_keys' own docstring).
    _refuse_unknown_keys(
        body.keys(), {"dry_run", "max_calls"}, where="body",
        dry_run_typo_exempt=True,
    )
    _refuse_unknown_keys(
        request.query_params.keys(), {"dry_run", "max_calls"},
        where="the query string", dry_run_typo_exempt=True,
    )
    # v3.1.9 (hostile pass 2, MEDIUM). `int(body.get("max_calls") or 200)`
    # used Python truthiness on the raw value, so an explicit
    # {"max_calls": 0} — an operator asking this endpoint to run its guards
    # and refusals with ZERO summarization calls, a legitimate "just check
    # the plan against the real store" probe distinct from dry_run — was
    # `0 or 200`, silently replaced by the 200-call LIVE default. Reproduced
    # against the unfixed code: {"max_calls": 0} on a 60-exchange backlog
    # ran 2 rollup calls and moved the watermark 0 -> 120, exactly the write
    # the caller's explicit 0 was asking this loop not to make — the same
    # "ambiguous-or-falsy value quietly becomes the write" shape as the
    # dry_run HIGH #1/#2 fixes just above this function. A non-numeric value
    # (`{"max_calls": "abc"}`) took the other failure direction: int() raised
    # ValueError uncaught, a 500 with no explanation for a caller-supplied
    # body that a 400 exists to handle everywhere else in this file.
    # v3.1.9 LOW (hostile pass 3, F3). Two more holes in the same int()
    # conversion the comment above already tightened once:
    #   - `bool` is a subclass of `int` in Python, so `int(True) == 1` ran
    #     ONE live call for {"max_calls": true} instead of being refused like
    #     every other non-integer shape. Checked explicitly, ahead of int().
    #   - `1e999` / `Infinity` / `-Infinity` are values `json` happily parses
    #     as `float`, and `int(float('inf'))` raises OverflowError, which the
    #     old `except (TypeError, ValueError)` did not catch — an uncaught
    #     500 with the fix's own comment claiming "non-integers are a 400".
    #     `NaN` was already a 400: `int(float('nan'))` raises ValueError.
    # NOT named `raw` (test_envcfg.py's file-wide, name-based env-taint scan
    # treats every `raw` in this file as descended from `_env_int`'s own
    # `raw = os.environ.get(name)`, scope or not — see that test's own
    # docstring. This value never touches the environment; `candidate`
    # sidesteps the false positive instead of fighting the detector.
    def _parse_one_max_calls(candidate: Any, *, source: str) -> int:
        if isinstance(candidate, bool):
            raise HTTPException(
                status_code=400,
                detail=f"max_calls ({source}) must be an integer, got {candidate!r}",
            )
        try:
            return int(candidate)
        except (TypeError, ValueError, OverflowError):
            raise HTTPException(
                status_code=400,
                detail=f"max_calls ({source}) must be an integer, got {candidate!r}",
            )

    # v3.1.9 (hostile pass 4, F5). max_calls used to be read from the BODY
    # ONLY: `{"dry_run": false} + ?max_calls=1` silently ran the 200-call
    # default, the same "accepted-looking and inert" shape R4 already named
    # for the flag right beside it (dry_run) — an operator probing with
    # `?max_calls=1` to see ONE rollup got the whole backlog. Both sources
    # are read now: a source that is ABSENT (the key not present at all)
    # has no opinion, exactly like `_dry_run_from`'s own present/absent
    # rule; `{"max_calls": null}` in the body is likewise "no opinion" (its
    # pre-existing meaning, unchanged) rather than a parse error. When only
    # one source is present, it wins; when BOTH are present, the SMALLER of
    # the two wins — fewer calls is the safe direction for a bound, so a
    # disagreement can never silently pick the more dangerous number.
    _max_calls_candidates: list[int] = []
    if "max_calls" in body and body["max_calls"] is not None:
        _max_calls_candidates.append(
            _parse_one_max_calls(body["max_calls"], source="body")
        )
    if "max_calls" in request.query_params:
        _max_calls_candidates.append(
            _parse_one_max_calls(request.query_params["max_calls"], source="query")
        )
    max_calls = min(_max_calls_candidates) if _max_calls_candidates else 200
    # Clamped both directions rather than trusted outright. Below zero has no
    # meaning for a count of calls (the loop already treats 0 as "run the
    # guards, make no calls" via `while calls < max_calls`, so negative would
    # be the same thing under a misleading number). Above 1000 is a real
    # operational bound, not a formality: this loop is one vLLM call plus two
    # state loads per iteration, on a conversation an operator runs WHILE she
    # is chatting (see the endpoint's own docstring), so a mistyped
    # max_calls: 9999999999 must not be able to run for hours unattended.
    # 1000 is 5x the documented 200 default and far above the worst backlog
    # on record in this codebase's incidents (33 calls).
    max_calls = max(0, min(max_calls, 1000))
    # Absent means LIVE for compact, which is the documented contract and
    # stays — but ?dry_run=true is honoured now instead of ignored.
    dry_run = _dry_run_from(request, body, default=False)

    exchanges = await run_in_threadpool(retrieval.export_indexed_exchanges, conv_id)
    if not exchanges:
        raise HTTPException(
            status_code=404,
            detail=(
                f"no indexed exchanges for conv {conv_id}. Either the id is "
                f"wrong, or episodic indexing never ran for it — check "
                f"GET /admin/conversations."
            ),
        )

    # Rebuilt BY SLOT, not by concatenation (v3.1.7, R13). See
    # _rebuild_transcript_by_slot: each pair goes to the position its
    # `turn_index` says it holds, and the exchanges the memory tail skipped
    # and the pool shed become explicit placeholder turns rather than a
    # silently shorter array.
    messages, gap_turns = _rebuild_transcript_by_slot(exchanges)

    before = summarizer.load_state(conv_id)
    # REFUSE rather than summarize the wrong text.
    #
    # The transcript here is rebuilt from the episodic store, which is lossy by
    # design — it holds only exchanges that indexed successfully. Since v3.1.4
    # the summarizer locates a chunk's text at `position - len(window)` turns
    # into the array it is handed (summarizer._do_l1_rollup), so a
    # reconstruction SHORTER than the conversation's position is not merely
    # short: its slots do not line up with the turns the chunk claims. A chunk
    # labelled 652-671 whose text is some other twenty turns is worse than no
    # chunk, because nothing downstream can tell.
    #
    # v3.1.7 (R13) narrows WHEN that is true. Until now the comparison was
    # against a concatenation, so ANY gap anywhere made the array short and
    # the endpoint refused — 63 skips in one measured window means every real
    # conversation, on the one rebuild-from-store recovery path there is, and
    # the one R12's own ERROR line sends the operator to. Filling the gaps in
    # place restores the alignment the arithmetic needs, so what is left to
    # refuse is the case the placeholders cannot fix: a store that does not
    # REACH the position at all. That is a genuinely missing tail (or head),
    # and no amount of padding invents it.
    #
    # Compared against turns_seen rather than last_summarized_turn (which it
    # can never be below): the watermark is how far the SUMMARIES got, the
    # position is how far the CONVERSATION got, and the offset arithmetic is
    # driven by the second.
    # summarizer._recorded_position, not a local max() of the two counters.
    # v3.1.7 (R12): a state file written by the pre-v3.1.4 code under a cap has
    # its watermark PULLED DOWN below the chunks it is supposed to track, and
    # turns_seen absent entirely. Both counters then read low, this guard
    # PERMITS a rebuild it should refuse, and the chunks come back labelled
    # against a position that is hundreds of turns short. The chunk labels are
    # the record; the watermark is a pointer derived from them, and it is the
    # only one of the three the old _reconcile_watermark could erase. One
    # function decides what "how far has this conversation got" means, here and
    # in the summarizer, so the endpoint and the rollup cannot disagree.
    _pos = summarizer._recorded_position(before)
    if len(messages) < _pos:
        raise HTTPException(
            status_code=409,
            detail=(
                f"refusing: the episodic store rebuilds {len(messages)} "
                f"messages for conv {conv_id} (including {gap_turns} "
                f"placeholder turns for exchanges it never indexed), but the "
                f"conversation's recorded position is already turn {_pos}. "
                f"Running would summarize text that is not the text the chunk "
                f"labels claim. Gaps INSIDE the store are filled and are not "
                f"why this refused; the store's highest turn falls short of "
                f"the position, so the end (or the beginning) of the "
                f"conversation is missing from it entirely."
            ),
        )
    # The second refusal, and the only new one: a reconstruction that is more
    # placeholder than transcript is not a transcript. Summarizing it would
    # spend a vLLM call per chunk to record that nothing is known, advance the
    # watermark past turns nothing will ever summarize, and store the result
    # as memory. It also bounds this array: one corrupt turn_index would
    # otherwise open a gap as wide as the number itself.
    _real_turns = len(messages) - gap_turns
    if gap_turns > _real_turns:
        raise HTTPException(
            status_code=409,
            detail=(
                f"refusing: rebuilding conv {conv_id} by turn position needs "
                f"{gap_turns} placeholder turns against only {_real_turns} "
                f"recorded ones. More of this transcript is missing than is "
                f"present, so summarizing it would record that it is unknown "
                f"rather than what it said. Check "
                f"GET /admin/conversations/{conv_id} and the episodic store's "
                f"turn_index values."
            ),
        )
    plan = {
        "conv_id": conv_id,
        "indexed_exchanges": len(exchanges),
        "reconstructed_messages": len(messages),
        # v3.1.7 (R13): the gap count is the honest half of the answer. A plan
        # that reports 30 rebuilt messages without saying 2 of them are
        # placeholders is the same claim the old concatenation made.
        "gap_turns": gap_turns,
        "gap_exchanges": gap_turns // 2,
        "recorded_position": _pos,
        "watermark_before": before.get("last_summarized_turn", 0),
        "l1_before": len(before.get("l1") or []),
        "dry_run": dry_run,
    }
    if gap_turns:
        logger.warning(
            f"conv={conv_id}: rebuilding from the episodic store needs "
            f"{gap_turns // 2} placeholder exchange(s) among "
            f"{len(messages) // 2} — those turns were never indexed, so the "
            f"summaries covering them will say so rather than claim text "
            f"this rebuild never had"
        )
    if dry_run or not messages:
        plan["note"] = (
            "dry run — nothing was written. Re-send with "
            '{"dry_run": false} to run it.'
        )
        return plan

    # DROP THE LIVE ANCHOR BEFORE DRAINING (v3.1.7, R10).
    #
    # The guard above proves the array is long enough to be measured against
    # the position. It does NOT make the summarizer measure it that way.
    # _observed_position aligns the window it is handed against `tail_fp` —
    # the fingerprints of the last few turns of the window the CHAT path sent
    # — and when that anchor appears nowhere in the window it falls back to
    # _ASSUMED_NEW_TURNS, i.e. "one exchange happened since last time". That
    # fallback is right for the live path, where main.py calls maybe_rollup
    # once per exchange. It is wrong here, where the array is not the next
    # exchange but the WHOLE conversation rebuilt from turn 1.
    #
    # The arithmetic, because it is the whole of R10. With a rebuild of n
    # turns against a recorded position of n — the exact case the guard
    # admits, and the healthy one — an unalignable anchor makes the position
    # max(n, n + 2) = n + 2, so window_offset becomes 2 and _do_l1_rollup
    # reads chunk 1-20's text at array slots -1..18. It clamps, labels the
    # chunk 3-20, and fills it with turns 1-18: a span it does not contain,
    # with turns 1 and 2 then covered by nothing at all, and turns_seen left
    # inflated by 2 for the rest of the conversation's life. That is verbatim
    # the outcome the refusal above calls "worse than no chunk, because
    # nothing downstream can tell" — reached past a guard that was right.
    #
    # Note WHERE the exposure is: only while n is within _ASSUMED_NEW_TURNS
    # of the position. A longer rebuild takes max(n, prev + 2) = n and lines
    # up by itself, which is why this never showed on a store that had run
    # ahead — and why equality, the case the endpoint exists to serve, was
    # the one that broke.
    #
    # Clearing the anchor is not throwing information away: the drain
    # overwrites tail_fp with the rebuild's own fingerprints on its very
    # first call regardless. All this decides is whether the FIRST call is
    # measured against an anchor that belongs to a different array. Without
    # one, _observed_position takes the no-anchor branch, and since the guard
    # has already established n >= _highest_chunk_turn it HOLDS at
    # max(n, prev) = n — window_offset 0, which is what "the array starts at
    # turn 1" means. From the second call on the anchor is the rebuild's own
    # and the drain is idempotent, which is what summarizer's
    # _ASSUMED_NEW_TURNS comment already assumed was true of the first.
    #
    # Under conv_lock, and released before the loop: maybe_rollup takes the
    # same non-reentrant lock, so this must not enclose it. Re-read inside
    # the lock rather than reusing `before`, because a live rollup may have
    # written since the guard read it.
    async with conv_lock(conv_id):
        _state = summarizer.load_state(conv_id)
        if _state.get("tail_fp"):
            _state["tail_fp"] = []
            _state["head_fp"] = ""
            _state["window_turns"] = 0
            summarizer.save_state(conv_id, _state)
            logger.info(
                f"conv={conv_id}: dropped the chat path's window anchor "
                f"before draining. This rebuild starts at turn 1 and reaches "
                f"turn {len(messages)}; measuring it against the anchor from "
                f"a bounded live window would advance the position past a "
                f"conversation this array already holds in full, and every "
                f"chunk would be labelled that far off the text inside it"
            )

    # Loop maybe_rollup until the watermark stops moving. Each call does one
    # tier's worth of work; the loop is what turns that into a catch-up. Bounded
    # by max_calls AND by lack of progress, because a rollup that cannot advance
    # must not spin.
    # Redacted ONCE, off the event loop, before the loop starts. Same reason
    # as the live tail: this walks every historical assistant turn through
    # reply_is_degenerate, and the result is identical on every pass. Called
    # bare inside the loop it blocked the loop for max_calls x the scan cost
    # - measured at 3.2 s per pass on a 2000-turn history, i.e. up to ~10
    # minutes of stalled event loop on an endpoint the operator is told to
    # run WHILE she is chatting.
    _redacted_messages = await run_in_threadpool(
        _redact_degenerate_turns, messages
    )
    calls = 0
    t0 = time.time()
    # v3.1.9 (hostile pass 4, F5). max_calls now bounds REAL vLLM
    # summarization calls (summarizer.vllm_call_budget_ctx), not just the
    # PASSES this loop makes — one pass (one maybe_rollup call) can still
    # make many real calls internally (it drains every L1/L2 chunk that is
    # due), and used to be able to spend however many the whole backlog
    # needed regardless of max_calls. `calls < max_calls` below is UNCHANGED
    # and kept as a second, independent bound on passes themselves — partly
    # a belt-and-braces safety net, partly because this exact call
    # (`summarizer.maybe_rollup(conv_id, _redacted_messages, VLLM_URL,
    # MODEL_REPO)`) is monkeypatched wholesale by a fixed-signature stub in
    # test_admin_compact.py's own max_calls coverage, so it keeps calling
    # maybe_rollup with today's EXACT signature — the budget is set via the
    # context-manager form instead of a keyword argument here for exactly
    # that reason (see vllm_call_budget_ctx's own docstring).
    with summarizer.vllm_call_budget_ctx(max_calls) as vllm_budget:
        async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
            while calls < max_calls:
                prev = summarizer.load_state(conv_id).get("last_summarized_turn", 0)
                try:
                    await summarizer.maybe_rollup(
                        conv_id, _redacted_messages, VLLM_URL, MODEL_REPO,
                    )
                except Exception as e:
                    plan["stopped_because"] = f"{type(e).__name__}: {e}"
                    break
                calls += 1
                now = summarizer.load_state(conv_id).get("last_summarized_turn", 0)
                if now <= prev:
                    plan["stopped_because"] = "the watermark stopped advancing"
                    break
                if vllm_budget["remaining"] <= 0:
                    plan["stopped_because"] = f"hit max_calls={max_calls} (vLLM calls)"
                    break
            else:
                plan["stopped_because"] = f"hit max_calls={max_calls}"
        vllm_calls_made = max_calls - vllm_budget["remaining"]

    after = summarizer.load_state(conv_id)
    plan.update({
        "rollup_calls": calls,
        # F5: the number this endpoint's own docstring now promises
        # max_calls bounds — actual vLLM HTTP calls, counted wherever in
        # the L1/L2/L3 drain (including a map-reduce split) they happened,
        # not rollup passes. rollup_calls (above) is kept unchanged for
        # existing callers that read it as "how many maybe_rollup passes".
        "vllm_calls": vllm_calls_made,
        "elapsed_s": round(time.time() - t0, 1),
        "watermark_after": after.get("last_summarized_turn", 0),
        "l1_after": len(after.get("l1") or []),
        "l2_after": len(after.get("l2") or []),
        "l3_after": bool(after.get("l3")),
    })
    logger.info(
        f"conv={conv_id}: admin compact — {calls} rollup call(s) in "
        f"{plan['elapsed_s']}s, watermark {plan['watermark_before']} -> "
        f"{plan['watermark_after']}, L1 {plan['l1_before']} -> {plan['l1_after']}"
    )
    return plan


@app.get("/admin/selftest", dependencies=[Depends(_require_localhost)])
async def admin_selftest(response: Response, round_trip: bool = True):
    """V2.1 Phase 6 Step 2: on-demand live-stack self-test.

    Runs the same check battery as the supervisord boot one-shot, but
    skips wait-for-ready (the stack is assumed up). Returns the JSON
    report. HTTP 503 if any check failed; 200 if all passed — so this
    endpoint is itself suitable as a deep healthcheck target for
    external monitoring.

    Query: ?round_trip=false to skip the real LLM call (useful for
    quick smoke checks that don't want to wait on inference).
    """
    report = await selftest_module.run_selftest(do_round_trip=round_trip)
    response.status_code = 200 if report["status"] == "pass" else 503
    return report


# V2.3 Theme 1: data-durability backup endpoints (localhost-only).
@app.get("/admin/backups", dependencies=[Depends(_require_localhost)])
async def admin_list_backups():
    """List existing backup archives (newest first) + latest-backup summary."""
    return {
        "backups": backup_module.list_backups(),
        "info": backup_module.latest_backup_info(),
    }


@app.post("/admin/backups", dependencies=[Depends(_require_localhost)])
async def admin_run_backup(request: Request, response: Response):
    """Trigger one backup cycle now (create → verify → publish → prune).
    Returns the report. HTTP 200 if the backup was created AND verified;
    503 if it failed (so this is a usable monitoring signal). Runs in a
    thread — the cycle is blocking I/O (sqlite snapshot, tar, verify).

    Takes no body and no query key (hostile pass 5, C5-5): either is a
    400, not a silently-ignored stray.
    """
    raw_body, body = await _parse_admin_json_body(request)
    _refuse_bad_admin_request(
        request, raw_body, body, body_keys=set(), query_keys=set(),
    )
    report = await asyncio.to_thread(backup_module.run_once)
    response.status_code = 200 if report.get("ok") else 503
    return report


@app.get("/admin/backups/verify", dependencies=[Depends(_require_localhost)])
async def admin_verify_backup(response: Response, name: str | None = None):
    """Verify an existing archive (default: the newest). Restores it to a
    scratch dir and runs the integrity checks. 200 ok / 503 fail / 404 none."""
    if name:
        from pathlib import Path
        target = Path(backup_module.BACKUP_DIR) / name
    else:
        archives = backup_module.list_backups()
        if not archives:
            raise HTTPException(status_code=404, detail="no backups to verify")
        from pathlib import Path
        target = Path(archives[0]["path"])
    ok, detail = await asyncio.to_thread(backup_module.verify_backup, target)
    response.status_code = 200 if ok else 503
    return {"archive": target.name, "ok": ok, "detail": detail}
