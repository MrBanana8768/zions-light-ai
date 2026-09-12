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
import dataclasses
import json
import logging
import warnings
import os
import re
import time
import unicodedata
from contextlib import asynccontextmanager
from typing import Any

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
from envcfg import env_float
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
# Whether get_tokenizer has ALREADY tried and failed. See its docstring:
# caching only the success made every later count_tokens re-enter
# from_pretrained, 238x slower per call.
_TOKENIZER_TRIED = False


def get_tokenizer():
    """The local tokenizer, or None. THE FAILURE IS CACHED TOO (v3.1.9).

    `if _tokenizer is not None: return` caches only a SUCCESS. The except
    below sets `_tokenizer = None`, which fails that same test, so every later
    call re-entered AutoTokenizer.from_pretrained — a filesystem walk and, when
    the HF cache is cold, a network attempt. Benchmarked: 0.012 ms cached
    against 2.859 ms per call after a miss, and `_chunk_to_budget` calls
    count_tokens once PER MESSAGE, so one 2,301-message compaction spends about
    6.6 seconds re-failing to load the same tokenizer. That is the FAST failure
    (HF_HUB_OFFLINE=1); a cold cache reaching for the network is worse.

    Latent rather than live on the pod today — the live logs show one
    "loaded tokenizer" per boot and zero failures — but the whole point is that
    it arms itself the first time the cache is evicted or /data hiccups, which
    is exactly when the compactor is least able to spare six seconds a turn.

    _TOKENIZER_TRIED is a separate flag rather than a sentinel object because
    `None` is a legitimate return here: it means "use the char/4 estimator",
    and several callers check for it.
    """
    global _tokenizer, _TOKENIZER_TRIED
    if _tokenizer is not None or _TOKENIZER_TRIED:
        return _tokenizer
    if not MODEL_REPO:
        logger.warning("MODEL_REPO not set; falling back to char/4 token estimator")
        _TOKENIZER_TRIED = True
        return None
    _TOKENIZER_TRIED = True
    try:
        from transformers import AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
        logger.info(f"loaded tokenizer for {MODEL_REPO}")
    except Exception as e:
        logger.warning(f"could not load tokenizer for {MODEL_REPO}: {e}; using char/4 estimator")
        _tokenizer = None
    return _tokenizer


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
TOKENIZE_WARN_INTERVAL_S = float(_env_int("COMPACTOR_TOKENIZE_WARN_INTERVAL_S", 300))
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
            return len(tok.encode(text)) + image_tokens
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


async def compact_if_needed(
    messages: list[dict], conv_id: str | None = None
) -> list[dict]:
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
    if conv_id:
        try:
            _st = summarizer.load_state(conv_id)
            # THE CONTIGUOUS PREFIX, not the highest label (v3.1.9, B3/B4).
            #
            # _highest_chunk_turn answers where coverage ENDS; the
            # substitution deletes turns from the START, so what it needs is
            # where coverage BEGINS and whether it is unbroken. Two shipped
            # rollup paths leave a hole on purpose (`pos_last < 1` advances
            # the watermark with no chunk at all; `partial` records a
            # deliberately narrower first_turn), and load_state parks an
            # unparseable chunk, which leaves one with no outage at all.
            # Across a hole this deleted 20 turns no chunk represents and
            # logged them as "covered by stored summaries". See
            # _covered_prefix for why l3 is excluded.
            _covered = summarizer._covered_prefix(_st)
            # AND THE TURNS THEMSELVES MUST MATCH (v3.1.9, B1).
            #
            # _covered_prefix proves the hierarchy CLAIMS an unbroken span
            # from turn 1, and _aligned proves the array's tail is this
            # conversation's. Neither looks at the turns being deleted.
            # OpenWebUI's edit-without-regenerate rewrites one message in the
            # middle and leaves the tail byte-identical, so both gates pass
            # and the stored summary of the PRE-EDIT text replaces the
            # corrected turn — and the hierarchy never re-reads that span, so
            # the correction is gone for good. That is the only unrecoverable
            # break in this path.
            #
            # covered_fp is folded one turn at a time at rollup over exactly
            # the turns each chunk summarized; recomputing it here over the
            # prefix about to be removed is the content relation the
            # substitution has always needed and never had.
            #
            # Absent on state written before v3.1.9, and absent is NO
            # EVIDENCE: _covered drops to 0 and this turn summarizes from
            # scratch, as it did before the feature existed. The next rollup
            # writes one.
            #
            # NO SEPARATE CHECK FOR THE ABSENT CASE, deliberately. The first
            # version read `if not _fp_want or _fp_turns <= 0 or _fp_turns >
            # _covered`, and a mutation dropping the first two conjuncts left
            # every test green — correctly, because they cannot change the
            # outcome. load_state admits the digest and its count together or
            # not at all, so no digest means _fp_turns == 0, which makes
            # `_covered = _fp_turns` zero coverage by itself; and a non-zero
            # count beside an empty digest could only reach the comparison
            # below, where _covered_fp_over of a non-empty prefix is never ""
            # and so declines. This gate has already shipped three conditions
            # that could not fire. It does not get a fourth for reassurance.
            _fp_turns = int(_st.get("covered_fp_turns") or 0)
            _fp_want = _st.get("covered_fp") or ""
            if _fp_turns > _covered:
                # Should be unreachable — the digest only extends
                # contiguously, so it cannot claim more than the contiguous
                # prefix. Unreachable is not impossible, and the two numbers
                # come from different evidence on purpose; when they disagree
                # the honest answer is to decline. [13b] builds the
                # disagreement and a mutation removing this goes RED.
                _covered = 0
            else:
                _covered = _fp_turns
                if summarizer._covered_fp_over(to_summarize, _covered) != _fp_want:
                    logger.info(
                        f"conv={conv_id}: the stored summaries do not match "
                        f"the first {_covered} turn(s) of this request — an "
                        f"edited or re-ordered turn, or a different branch. "
                        f"Summarizing from scratch rather than replacing them."
                    )
                    _covered = 0
            # ONLY WHEN THE ARRAY IS NOT A SUFFIX, and this is the whole
            # safety of it. Turn numbers are not array indices once a
            # client sends a bounded window; mapping between them needs
            # window_offset, and the function that owns that arithmetic
            # (_observed_position) MUTATES state - calling it here would
            # advance the anchor a second time per turn. Under-claiming
            # coverage costs a little speed; over-claiming drops turns the
            # hierarchy cannot actually speak for. So when the client is
            # sending everything (offset provably 0) this applies, and
            # when it is not, it declines and today's behaviour stands.
            _non_system = [m for m in messages if m.get("role") != "system"]
            _n = len(_non_system)

            # THE LENGTH TEST ALONE IS NOT ENOUGH. It proves a length
            # relation; the substitution needs a CONTENT one, and an
            # adversarial pass demonstrated the gap: OpenWebUI keeps branches
            # in ONE chat, so conv_id never changes, and a regenerate or an
            # edit produces an array that satisfies `_n >= _recorded_position`
            # while being a DIFFERENT branch. 20 of 30 branch-B exchanges were
            # deleted and replaced by branch A's summary, and it never
            # self-heals - _do_l1_rollup's duplicate-label guard then discards
            # branch B's replacement chunks.
            #
            # _align_candidates answers exactly this and is PURE - it takes the
            # anchor and the fingerprints and returns candidates, touching no
            # state. That matters because the comment above is right that
            # _observed_position must not be called here; this is the half of
            # it that is safe to borrow. Empty means "cannot be told", which is
            # the answer that must decline.
            # THE FULL ANCHOR, not any prefix of it (v3.1.9, B2).
            #
            # This read `bool(_align_candidates(...))` — non-emptiness — and
            # _align_candidates tries every prefix down to length ONE. The
            # anchor's first element is a user turn, so a single repeated
            # short turn satisfied it: an adversarial pass reproduced the
            # very break the paragraph above says was closed, 20 of 30
            # branch-B exchanges replaced by branch A's summary, with
            # candidates == [0], which is truthy. Prefixes are correct for
            # _observed_position, which must read a regenerated reply as
            # zero new turns; they are wrong for "is this the same
            # conversation". _align_new_turns already documents the rule this
            # site broke — a repeated "ok" must land on the side that
            # duplicates, not the side that loses.
            _anchor = _st.get("tail_fp") or []
            _aligned = bool(_anchor) and summarizer._aligns_fully(
                _anchor,
                summarizer._turn_fingerprints(
                    _non_system[-summarizer._FINGERPRINT_TAIL_TURNS:]
                ),
            )

            if _covered > 0 and _n >= summarizer._recorded_position(_st) and _aligned:
                # TURN NUMBERS ARE NOT text_only INDICES. _covered counts every
                # non-system turn; text_only has image turns removed, so
                # `min(_covered, len(text_only))` overruns by the image count
                # and deletes that many turns the hierarchy never covered -
                # demonstrated at 1 and 5 turns of overrun, reachable at the
                # shipped MAX_RETAINED_IMAGES=1. Count the prefix instead of
                # assuming the two units agree.
                stored_turns = min(
                    sum(
                        1 for m in to_summarize[:_covered]
                        if not _message_has_image(m)
                    ),
                    len(text_only),
                )
                if stored_turns > 0:
                    # all_or_nothing: a squeezed block drops the OLDEST scenes,
                    # which are the same turns removed below. See the kwarg's
                    # docstring - this is the caller it exists for.
                    stored_text = await run_in_threadpool(
                        summarizer.format_summary_block,
                        _st,
                        summarizer.SUMMARY_BLOCK_MAX_TOKENS,
                        all_or_nothing=True,
                    ) or ""
            if not stored_text:
                stored_turns = 0
        except Exception as e:
            # Never fail a request over an optimisation. Falling back is
            # exactly today's behaviour.
            logger.warning(
                f"conv={conv_id}: could not reuse stored summaries "
                f"({type(e).__name__}: {e}); summarizing from scratch"
            )
            stored_text = ""
            stored_turns = 0

    fresh_input = text_only[stored_turns:]
    async with httpx.AsyncClient() as client:
        if fresh_input:
            summary, deferred = await summarize(client, fresh_input)
        else:
            summary, deferred = "", []
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
        "content": "[Summary of earlier conversation]\n"
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
        + (f", {stored_turns} covered by stored summaries"
           if stored_turns else "")
        + ("" if (_summarized or stored_turns)
           else "  [NO SUMMARIZATION HAPPENED]")
    )
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


def reply_is_degenerate(text: str) -> str | None:
    """Why this reply looks like a repetition loop, or None if it looks fine.

    On 2026-08-29 the model entered a loop emitting U+2501 and produced three
    consecutive replies that were 50-79% box-drawing, each ending mid-rule after
    a single unbroken run of 386, 425 and 569 characters. Decoration fraction
    climbed 6.7% -> 50% -> 67% -> 79% across four turns, because each reply
    entered the history and the guard — shedding to the most recent handful of
    messages — made that pattern most of what the model could still see.

    This does NOT stop the reply reaching the user; by the time we can measure
    it, she has already read it, and silently rewriting model output is not
    something this system does. It stops the reply being MEMORISED, so a loop
    cannot write itself into facts, episodic and summaries and be injected back
    as though it were something worth remembering. Same doctrine as the
    finish_reason=="length" gate: a reply that is not a real answer is not a
    memory.
    """
    if not text:
        return None
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
        )
    # The repeated PHRASE, which the token rule above cannot represent.
    _loop = _tail_loop_span(text)
    if _loop >= DEGENERATE_TAIL_LOOP_CHARS:
        return (
            f"a phrase repeating to the end of the reply for {_loop} "
            f"characters (limit {DEGENERATE_TAIL_LOOP_CHARS})"
        )
    m = max(_RUN_RE.finditer(text), key=lambda x: len(x.group(0)), default=None)
    if m and len(m.group(0)) >= DEGENERATE_RUN_CHARS:
        return (
            f"a single character repeated {len(m.group(0))} times "
            f"(limit {DEGENERATE_RUN_CHARS})"
        )
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
            return (
                f"{100 * frac:.0f}% of letters are non-Latin over "
                f"{lat + non} letters across {len(scripts)} script(s) "
                f"(limit {100 * DEGENERATE_NONLATIN_FRACTION:.0f}% over "
                f"5+ scripts, or 20% over 3+)"
            )
    if n >= DEGENERATE_MIN_CHARS:
        decor = sum(1 for c in text if c in _DECOR_CHARS)
        if decor / n >= DEGENERATE_DECOR_FRACTION:
            return (
                f"{100 * decor / n:.0f}% decoration characters over {n} chars "
                f"(limit {100 * DEGENERATE_DECOR_FRACTION:.0f}%)"
            )
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
    run = 0
    in_fence = False
    for raw in text.splitlines():
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
        ln = len(line)
        if ln >= DEGENERATE_LINE_CHARS and line.count(" ") >= _LINE_MIN_SPACES:
            # R24: "! " and "? " are always real ends (see
            # _is_real_sentence_end), but "." needs the abbreviation and
            # single-initial check trim_to_last_sentence uses, or "Dr. ",
            # "Mrs. ", "9 a.m. " etc each register as a sentence break and
            # collapse the computed mean on ordinary prose.
            breaks = (
                _count_real_period_breaks(line) + line.count("! ")
                + line.count("? ") + line.count("… ")
            )
            if breaks == 0:
                # No sentence at all in 1500+ characters: either a run-on,
                # which is not this rule's shape, or a list whose separator
                # has shrunk to a comma — judged the same way, on the commas.
                breaks = line.count(", ")
            if ln / (breaks + 1) <= DEGENERATE_LINE_SENTENCE_CHARS:
                return (
                    f"an unbroken line of {ln} characters made of "
                    f"{breaks + 1} fragments averaging "
                    f"{ln / (breaks + 1):.0f} characters (limit "
                    f"{DEGENERATE_LINE_SENTENCE_CHARS} over "
                    f"{DEGENERATE_LINE_CHARS}+ characters)"
                )
    # R19: gated on DEGENERATE_MIN_CHARS like the decoration-fraction rule
    # above — this file's own doctrine (see MIN_MEMORABLE_TRIMMED_CHARS)
    # calls that the floor below which nothing is judged structurally, and
    # this branch was the one exception.
    if n >= DEGENERATE_MIN_CHARS and run >= DEGENERATE_LIST_RUN:
        return (
            f"a run of {run} consecutive list items of "
            f"{DEGENERATE_LIST_ITEM_CHARS} characters or fewer (limit "
            f"{DEGENERATE_LIST_RUN})"
        )
    return None


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
    """
    out = []
    redacted = 0
    for m in messages:
        if (
            isinstance(m, dict)
            and m.get("role") == "assistant"
            and reply_is_degenerate(_message_text(m))
        ):
            m = {**m, "content": _DEGENERATE_HISTORY_PLACEHOLDER}
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
            f"rollup input ({len(messages)} total)"
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
    # rather than a re-scan of everything before it. Same "line starts with
    # ```" test reply_is_degenerate uses, so the two agree on what a fence is.
    toggles: list[int] = []
    pos = 0
    for line in text.splitlines(keepends=True):
        if line.strip().startswith("```"):
            toggles.append(pos)
        pos += len(line)
    end = 0
    n = len(text)
    for m in _SENTENCE_END_RE.finditer(text):
        i = m.start()
        if toggles and bisect.bisect_right(toggles, i) % 2 == 1:
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
    not from mutilation."""
    sys_idxs = [i for i, m in enumerate(msgs) if m.get("role") == "system"]
    return sys_idxs[max(1, protect_system):]


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
    injected memory blocks, trimmed largest-first. The newest turn is never
    dropped — losing the message the user just typed is worse than any
    truncation. After shedding, role alternation is REPAIRED (first non-system
    message must be a user turn) — the first cut of this guard could stop
    mid-pair and hand the Mistral template an assistant-first conversation,
    manufacturing the very 400 it exists to prevent (rc6 review).

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
        # --- shed oldest non-system turns (arithmetic only) ---
        while running > limit:
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
        while running > limit and trimmed < 32:
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
        while running > limit:
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
        # v3.1: this used to log at WARNING and read like a success — "hard
        # budget enforced" while forwarding a payload the guard itself has just
        # measured as too large. It is a failure of the thing whose entire job
        # is to make vLLM's 400 impossible, and the 400 is now the expected
        # outcome. Say so, at ERROR, with the shortfall, so it is findable
        # before the user reports it rather than after.
        #
        # v3.1 D3: and say WHAT is left, because the two residuals need
        # different people to act. On 2026-08-28 the line read "dropped 0 old
        # turn(s), trimmed 6 injected block(s), dropped 1 injected block(s)
        # entirely - still 16417 over"; 16,384 + 16,417 = 32,801, which is
        # exactly the number vLLM went on to report, so every one of those
        # 32,801 tokens was the caller's own system prompt and the single turn
        # the user had typed. Nothing the compactor is allowed to touch was
        # still in that payload — and the line said "a conversation with
        # nothing left to shed is the usual cause" without saying which case it
        # was looking at, so it read as a compactor problem for four hours.
        if not _droppable_system_indices(msgs, protect_system):
            residual = (
                "Nothing injected remains: what is left is the caller's own "
                "system prompt and the newest turn, and neither is this "
                "guard's to spend. The request as SENT does not fit the "
                "window — that is a client-side size problem, not a memory one"
            )
        else:
            # Unreachable: the pass above drops every droppable block before
            # this line can be reached. Kept as a marker, because a guard that
            # gives up holding memory it was allowed to spend is the exact
            # defect v3.1 D3 closed and it should be loud if it returns.
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
    # its reading for COMPACTOR_DEGRADE_CHECK_TTL_S (10 s), so within that
    # window this call answers from the SAME statvfs _async_tail's guard
    # took and cannot see a disk that filled in between. The first version
    # of this comment claimed it covered exactly that window — "job 1
    # indexes and this job makes a vLLM extraction call that can take
    # seconds, so the disk can fill between that check and this write" —
    # and then cited the TTL two sentences later as proof the call was
    # cheap. The second half nullifies the first, in one paragraph, and a
    # hostile review caught it the same day it was written. A cache quoted
    # as a performance reassurance is still a cache.
    #
    # So the honest account of what it buys, in order of how often it
    # bites:
    #   * COVERAGE. _facts_tail is a public-shaped coroutine with five
    #     suites entering the tail directly; a future caller that is not
    #     _async_tail gets the pause applied rather than the one that
    #     remembered. That is job 3's argument verbatim and it does not
    #     depend on timing at all.
    #   * The gaps that DO exceed 10 s: a tail re-queued behind a pool
    #     backlog (bgwork.pool caps concurrency, so a burst makes this
    #     arbitrarily long), and an extraction plus dedup round trip to a
    #     loaded vLLM.
    # Inside 10 s of the outer check it is a no-op, and that is fine — the
    # outer check already refused, or the disk genuinely had room.
    #
    # Silent return, matching job 3: guard() already logs at debug and
    # writes_allowed() warned on the transition.
    if not degrade.guard("fact extraction tail"):
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
    # `(assistant_text or "")` where job 1 writes a bare
    # assistant_text.strip(). This is DEFENCE ONLY, and currently
    # unreachable: job 1's gate runs first and raises AttributeError on a
    # None reply before control ever arrives here, so no caller can
    # actually exercise the tolerance. The commit that added it justified
    # it as protecting "a direct caller's None", which was wrong — the
    # direct caller goes through job 1 too. It is kept because
    # _rollup_hierarchy types the same parameter `str | None` and means it,
    # so the tolerant spelling is what this function should have if job 1's
    # gate is ever softened, and because removing it would be a third
    # spelling of the same rule. The dialect sweep is M9's job, not this
    # commit's.
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
    if assistant_text.strip() and _has_pairable_user_text(last_user_text):
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
    await _rollup_hierarchy(conv_id, original_messages, assistant_text)


async def _rollup_hierarchy(
    conv_id: str,
    messages: list[dict],
    assistant_text: str | None,
) -> None:
    """Advance the hierarchical summary. Both tail paths call this.

    `assistant_text` is the reply to roll up WITH the history, or None to
    roll up the history alone — which is what the skipped-tail path passes.

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
    # remembered. writes_allowed() is cached for
    # COMPACTOR_DEGRADE_CHECK_TTL_S, so this is a tuple read.
    if not degrade.guard("hierarchy rollup"):
        return
    # A reply of whitespace is not a turn to roll up: it would advance the
    # watermark over a turn that says nothing, and the label would then
    # cover text no summary can account for. None is not whitespace - it is
    # 'do not append a reply at all', which is a different instruction.
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
        _redacted = await run_in_threadpool(
            _redact_degenerate_turns, list(messages)
        )
        full_messages = _redacted + (
            [{"role": "assistant", "content": assistant_text}]
            if assistant_text is not None
            else []
        )
        # OFF THE EVENT LOOP (v3.1.9.2), same reasoning as the two reads
        # inside maybe_rollup. This one is purely a "did anything change"
        # snapshot for the log line below, and it runs on every turn the tail
        # runs — a blocking disk read on the loop to decide whether to print.
        before = await run_in_threadpool(summarizer.load_state, conv_id)
        state = await summarizer.maybe_rollup(
            conv_id, full_messages, VLLM_URL, MODEL_REPO or ""
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
    accepted = _fire_and_forget(
        _async_tail(
            conv_id,
            touched_facts,
            last_user_text,
            decision.text,
            turn_index,
            messages,  # original request messages, for rollup
            injected_facts=injected_facts,
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
        body = json.loads(_raw, parse_constant=_reject_json_constant)
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

    # V1 compaction
    try:
        body["messages"] = await compact_if_needed(messages, conv_id)
    except Exception as e:
        logger.exception(
            f"compaction failed; falling through with the original messages — "
            f"the hard-budget guard will shed content if they don't fit: {e}"
        )

    # The window this request will finally be measured against, computed HERE
    # rather than at the pre-flight below because the memory injection that
    # follows has to be bounded by it. vLLM enforces prompt + max_tokens <=
    # window, so a fixed reserve alone leaves a client asking for a big
    # completion still 400able; and a memory budget expressed as a token
    # constant cannot see any of that. Nothing between here and the guard
    # depends on the value, and it depends on nothing but `body`.
    try:
        req_max_tokens = int(body.get("max_tokens") or 0)
    except (TypeError, ValueError):
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

        # Injection budget, computed HERE rather than at the inject point
        # below, because the summary block needs its share of it first: the
        # block's own 12,000-token cap exceeds this whole budget at
        # production config, and capping only inside summarizer meant
        # _bound_injected_blocks dropped whole layers (facts gone from ~50%
        # tier fill, everything but persona at ~70%).
        has_history = _has_conversational_history(messages)
        inject_budget = int(
            effective_limit
            * (
                INJECTION_BUDGET_FRACTION
                if has_history
                else INJECTION_NO_HISTORY_FRACTION
            )
        )

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
            # 60% of the injection budget: at production config that is
            # ~4,900 tokens, which reproduces the old working behaviour
            # (summary trimmed newest-kept, facts and persona still fit) and
            # leaves 40% for the other three layers.
            sblock = await run_in_threadpool(
                summarizer.format_summary_block,
                sstate,
                min(
                    summarizer.SUMMARY_BLOCK_MAX_TOKENS,
                    int(inject_budget * 0.6),
                ),
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
    # v3.1 D4: what the guard decided, carried to the rejection path. Without
    # it a 400 the guard PREDICTED (and logged at ERROR before sending) is
    # indistinguishable from one that surprised it, and the calibration learns
    # a process-global margin from the first kind.
    guard_report: dict = {}
    body["messages"] = await run_in_threadpool(
        _enforce_hard_budget,
        body["messages"],
        effective_limit,
        caller_system,
        guard_report,
    )
    guard_measured_overflow = guard_report.get("fits") is False
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

    # The limit the guard ACTUALLY shed against, captured here rather than
    # recomputed if this request is rejected: _note_backend_rejection moves
    # _BUDGET_MARGIN, so by the time a rejection is logged the margin is no
    # longer the one this payload was measured against, and the log line would
    # name a budget that was never in force. Mirrors the clamp inside
    # _enforce_hard_budget.
    enforced_limit = max(256, effective_limit - _BUDGET_MARGIN)

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
                                guard_measured_overflow=guard_measured_overflow,
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
                guard_measured_overflow=guard_measured_overflow,
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
async def admin_forget_facts(conv_id: str):
    """Forget ALL memory for a conversation (V2.0 granularity: all-or-
    nothing). Clears persistent facts (Phase 2), episodic embeddings
    (Phase 3), AND the hierarchical summary state (Phase 4) — a full
    three-layer memory reset for when the model is stuck on something
    wrong. Targeted forgetting (single fact by substring) is V2.1.
    """
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

    Body: {"text": "<persona text>"}
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="body must be JSON")
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")
    _refuse_unpaired_surrogate(body)
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
async def admin_delete_persona(conv_id: str):
    """Clear the persona for a conv. Idempotent — returns deleted=False
    if no persona was stored."""
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

    Body: {"source_conv_id": "<conv_id to copy from>"}
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="body must be JSON")
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")
    _refuse_unpaired_surrogate(body)
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
async def admin_archive_stale(conv_id: str, older_than_days: int | None = None):
    """Trigger a stale-fact archival pass for one conv. Moves facts whose
    last_used is older than the cutoff to the archive sidecar.

    Query: ?older_than_days=N (default 90, env-overridable).
    """
    days = older_than_days if older_than_days is not None else facts.ARCHIVE_DEFAULT_DAYS
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

    Body JSON (all fields optional):
        {"text_substring": "<substring filter>" | null}

    Omit body or pass {} to restore ALL archived facts.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    _refuse_unpaired_surrogate(body)
    substring = body.get("text_substring")
    async with conv_lock(conv_id):
        restored = facts.restore_from_archive(
            conv_id, text_substring=substring,
        )
    return {
        "conv_id": conv_id,
        "restored": restored,
        "filter": substring,
    }


# V2.1 Phase 7 Step 1: on-demand semantic deduplication.
@app.post(
    "/admin/conversations/{conv_id}/dedup",
    dependencies=[Depends(_require_localhost)],
)
async def admin_dedup(conv_id: str):
    """Run a full hybrid (embedding + LLM) dedup pass on the conv's facts.

    Returns counters for the response body:
        {"conv_id", "before": int, "after": int, "removed": int}

    Inline dedup runs automatically after every fact extraction (cheap
    when no candidate clusters); this endpoint is for manual cleanup
    of conversations that pre-date Phase 7 or accumulated dupes via
    backfill/import.
    """
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
    prevents accidental wipe of an active conversation.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="body must be JSON")
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")
    _refuse_unpaired_surrogate(body)
    bundle = body.get("bundle")
    if bundle is None:
        raise HTTPException(status_code=400, detail="missing required field: 'bundle'")
    try:
        result = portability.import_conversation(
            bundle,
            target_conv_id=body.get("target_conv_id"),
            overwrite=bool(body.get("overwrite", False)),
        )
    # v3.1.8: UnsafeConvId alongside ImportError_. A body-supplied
    # target_conv_id / new_conv_id is CLIENT INPUT that reaches the
    # filesystem; memory._safe_path refuses to leave STORAGE_ROOT, and
    # that refusal is a 400 about the request, not a 500 about us.
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
    """
    # Body is optional — accept empty or missing.
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    _refuse_unpaired_surrogate(body)
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
async def admin_cleanup_test_conversations(dry_run: bool = True):
    """Quarantine-then-remove the test/placeholder conversations polluting
    the store: 129 "conversations" for ~26 real ones, inflating
    /admin/conversations, the health stats and every backup archive.

    DRY RUN BY DEFAULT. Matches only ids minted by selftest.py and the
    integration harness, and refuses any match that carries substantial
    memory (or whose layers cannot be read - unreadable counts as
    substantial, never as empty). Everything is quarantined before it is
    wiped, so this is reversible; nothing is unlinked.
    """
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

    "false"/"0"/"no" mean commit; anything else, INCLUDING A TYPO, leaves the
    caller in whatever direction is safe for that endpoint. `default` is what
    an absent flag means, not what a malformed one does.
    """
    if "dry_run" in body:
        return bool(body["dry_run"])
    raw = str(request.query_params.get("dry_run", "")).strip().lower()
    if not raw:
        return default
    return raw not in ("false", "0", "no")


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
    a flag that is accepted-looking and inert is worse than one that 400s. The
    body still wins when both are present: an explicit JSON body is the more
    deliberate of the two.

    Merges FACTS and EPISODIC exchanges only. Summaries are deliberately not
    merged: the forked half re-derives its own hierarchy from the client's
    full array, so dst already covers the same history and folding src's in
    would double-count it. The source is left completely intact, so a merge
    that produces a bad result costs nothing but the re-embedding.

    See portability.merge_conversation for the full contract.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    _refuse_unpaired_surrogate(body)
    # Absent means DRY for merge: this endpoint rewrites two conversations
    # and an operator who meant to commit sees unchanged counts and tries
    # again, while the reverse mistake is not recoverable.
    dry_run = _dry_run_from(request, body, default=True)

    try:
        return await run_in_threadpool(
            portability.merge_conversation,
            src_conv_id,
            dst_conv_id,
            dry_run=dry_run,
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
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    _refuse_unpaired_surrogate(body)
    max_calls = int(body.get("max_calls") or 200)
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
        else:
            plan["stopped_because"] = f"hit max_calls={max_calls}"

    after = summarizer.load_state(conv_id)
    plan.update({
        "rollup_calls": calls,
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
async def admin_run_backup(response: Response):
    """Trigger one backup cycle now (create → verify → publish → prune).
    Returns the report. HTTP 200 if the backup was created AND verified;
    503 if it failed (so this is a usable monitoring signal). Runs in a
    thread — the cycle is blocking I/O (sqlite snapshot, tar, verify)."""
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
