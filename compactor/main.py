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
import json
import logging
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
from memory import (
    StoreUnreadable,
    conv_lock,
    ensure_storage_layout,
    facts_path,
    list_known_conv_ids,
    resolve_conv_id,
    storage_summary,
)


def _env_int(name: str, default: int) -> int:
    """os.environ.get returns '' (not the default) when the var is set to an
    empty string, which is what .env files do for opt-in blanks. Treat empty
    as 'use the default'.
    """
    v = os.environ.get(name, "")
    return int(v) if v.strip() else default


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

_PESSIMISTIC_SUMMARY_SCALE = float(
    os.environ.get("COMPACTOR_PESSIMISTIC_SUMMARY_SCALE", "2.0") or 2.0
)
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
    """Same contract as _env_int: an unset or unparseable value is the default,
    never a crash at import time and never a silent zero."""
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


def get_tokenizer():
    global _tokenizer
    if _tokenizer is not None:
        return _tokenizer
    if not MODEL_REPO:
        logger.warning("MODEL_REPO not set; falling back to char/4 token estimator")
        return None
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


async def compact_if_needed(messages: list[dict]) -> list[dict]:
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
    async with httpx.AsyncClient() as client:
        summary, deferred = await summarize(client, text_only)
    # No summary block when there is no summary. summarize() returns
    # ("", all turns) when the backlog is too large for one request, and a
    # bare "[Summary of earlier conversation]" header with nothing under it
    # is worse than absent: it tells the model a summary exists and then
    # shows it an empty one.
    summary_blocks = ([{
        "role": "system",
        "content": f"[Summary of earlier conversation]\n{summary}",
    }] if summary.strip() else [])
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
    _summarized = len(text_only) - len(deferred)
    logger.info(
        f"compacted: summarized {_summarized} text turn(s), forwarded "
        f"{len(deferred)} verbatim, preserved {len(preserved_images)} image "
        f"turn(s), {current} -> {new_count} tokens"
        + ("" if _summarized else "  [NO SUMMARIZATION HAPPENED]")
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
DEGENERATE_DECOR_FRACTION = float(
    os.environ.get("COMPACTOR_DEGENERATE_DECOR_FRACTION", "0.45") or 0.45
)
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
DEGENERATE_NONLATIN_FRACTION = float(
    os.environ.get("COMPACTOR_DEGENERATE_NONLATIN_FRACTION", "0.03") or 0.03
)
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
    dropped = 0
    while (
        len(msgs) > 1
        and msgs[-1].get("role") == "assistant"
        and not _message_text(msgs[-1]).strip()
        and any(m.get("role") == "user" for m in msgs[:-1])
    ):
        msgs.pop()
        dropped += 1
    if dropped:
        note = (
            f"dropped {dropped} empty trailing assistant turn(s) left behind "
            f"by an earlier failed stream"
        )

    # (2) A real assistant-final list is a continuation, not an error.
    if msgs and msgs[-1].get("role") == "assistant":
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
    # Dropping a turn is always a real repair; setting the flag on a payload
    # that already carried it is not.
    return note, bool(dropped) or not already_ok


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
    leaves is at most one layer's own cap (1500 tokens for facts, 1500 for
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

    def feed(self, chunk: bytes) -> None:
        try:
            self._buffer += chunk.decode("utf-8", errors="replace")
        except Exception as e:
            # Dropping a chunk here does not fail the request — the user still
            # sees the full reply, because the bytes are forwarded separately.
            # What is lost is this accumulator's copy, so the memory tail
            # extracts facts, embeds and summarizes a reply with a hole in it,
            # and .complete() may still say the stream finished cleanly. Once
            # per process: feed() runs per SSE chunk. (v3.1 P0-2b / F61.)
            if logsetup.log_once("accumulator.feed.decode"):
                logger.warning(
                    f"stream accumulator dropped a chunk ({type(e).__name__}: "
                    f"{e}); the assistant text memorized for this turn is "
                    f"incomplete"
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
                    pass

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

    def usable(self) -> bool:
        """The gate the memory tail should use: the model finished, and it
        finished because it was done rather than because it ran out of room."""
        return self._complete and not self._truncated


def _fire_and_forget(coro, label: str | None = None) -> None:
    """Spawn post-response background work through the bounded pool
    (V2.3 Theme 3). The pool caps concurrency and sheds beyond a hard
    outstanding ceiling rather than spawning unboundedly under load. Task
    references are kept alive by the pool; exceptions are logged there.

    `label` is what the pool's shed WARNING names when it drops this task.
    A11 added the parameter to bgwork.pool.submit and nothing passed it, so
    the warning could say that a tail was dropped but not WHOSE — which is
    the entire reason the parameter exists. Pass the conversation.
    """
    bgwork.pool.submit(coro, label)


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
         exchange, merge + prune + save.
      3. Hierarchical rollup (Phase 4): if enough new turns have accumulated
         since the last summarization, roll L0→L1, L1→L2, L2→L3 as needed.

    All degrade to no-ops on failure — never affects the user response.
    Facts and summary writes are serialized per-conv via conv_lock.

    `original_messages` is the request's messages list (pre-compaction); we
    append the just-completed assistant turn before passing to the rollup so
    it sees the full conversation when computing turn ranges.

    `touched_facts` is the WHOLE store as the request path read it, and it is
    what gets merged and written back — the facts left out of this turn's
    working set must keep their real last_used or eviction stops meaning
    anything (v3.1 F9). `injected_facts` is the budget-bounded subset of those
    same dicts that the request path actually put in front of the model. They
    are separate because the two jobs need different lists: only the second may
    be handed to the extractor, which is a request to vLLM and therefore has a
    window.

    `injected_facts` is keyword-only with a default so no caller is broken by
    its arrival, and the default is `select_for_injection(touched_facts)` —
    not `touched_facts` — so a caller that never learned about it still cannot
    push the whole store into an extraction prompt. The request path passes
    the real list because it already computed one; recomputing here would
    answer the question against a store that may have moved since.
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
    if assistant_text and last_user_text:
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

    if not assistant_text or not last_user_text:
        return

    async with conv_lock(conv_id):
        try:
            async with httpx.AsyncClient() as client:
                # The BOUNDED set, not the whole store. facts.py now trims its
                # own input, so passing the store no longer overflows the
                # window — but the trim it would apply is a second, later
                # opinion about which facts matter, computed from a store that
                # may have grown since. Handing it what the model was actually
                # shown means the extractor is told about the same facts the
                # assistant reply was written against, so "already known" means
                # the same thing on both sides of the exchange.
                # conv_id is logging only, and it is what makes a lost
                # extraction attributable to the turn that lost it.
                new_strs = await facts.extract_facts_from_exchange(
                    client,
                    VLLM_URL,
                    MODEL_REPO or "",
                    last_user_text,
                    assistant_text,
                    (
                        injected_facts if injected_facts is not None
                        else facts.select_for_injection(touched_facts)
                    ),
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

    # --- 3. Hierarchical summary rollup (Phase 4) ---
    # Runs OUTSIDE the facts lock since maybe_rollup acquires its own
    # conv_lock internally — nesting the same lock would deadlock.
    if summarizer.enabled() and assistant_text:
        try:
            # v3.1.3: redact past degenerate turns before they can reach
            # maybe_rollup — see _redact_degenerate_turns for why the
            # call-site skip above is not enough on its own for this job.
            # `assistant_text` itself needs no check: neither call site
            # reaches this function when it is degenerate.
            # run_in_threadpool, not a bare call: this walks EVERY historical
            # assistant turn through reply_is_degenerate, and _async_tail is a
            # coroutine, so a bare call blocks the event loop for every other
            # request. Measured against her real replies (median 5,248 chars):
            # 20 turns 4.6ms, 40 turns 9.5ms, 85 turns 65ms, 170 turns 446ms -
            # and it runs on every single turn. The detector blocking this
            # same loop is a defect this branch has already shipped once.
            _redacted = await run_in_threadpool(
                _redact_degenerate_turns, list(original_messages)
            )
            full_messages = _redacted + [
                {"role": "assistant", "content": assistant_text}
            ]
            before = summarizer.load_state(conv_id)
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

@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Any:
    body = await request.json()
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
        body["messages"] = await compact_if_needed(messages)
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
                injected_facts = facts.select_for_injection(touched_facts)
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
                    f"msgs={len(messages)} lastturn={last_turn}"
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
                # Fire-and-forget post-response work once the stream is done.
                # Skip it when vLLM failed — there's no real assistant turn to
                # extract/index from — and when the stream never COMPLETED
                # (client hit Stop / tab closed mid-reply): memorizing a
                # half-sentence as though the model said it plants false
                # "memories" in facts/RAG/summaries (rc6 review).
                if conv_id and not vllm_failed and not accumulator.usable():
                    _why = (
                        "truncated at the generation ceiling "
                        "(finish_reason=length)"
                        if accumulator.truncated()
                        else "ended without completion"
                    )
                    # WARNING, not INFO. If a client sends a max_tokens
                    # below the model's usual reply length, EVERY reply
                    # finishes as "length" and this branch silently stops all
                    # memory writing — facts, episodic and rollups — for the
                    # life of that setting. That is the 2026-08-28 shape
                    # exactly: correct local behaviour, no error, and the user
                    # experiencing an assistant that has stopped remembering.
                    # The skip is right; being quiet about it is not.
                    logger.warning(
                        f"conv={conv_id}: stream {_why} "
                        f"({len(accumulator.text())} chars accumulated) — "
                        f"skipping memory tail for the partial reply"
                    )
                _degen = (
                    reply_is_degenerate(accumulator.text())
                    if (conv_id and not vllm_failed and accumulator.usable())
                    else None
                )
                if _degen:
                    logger.warning(
                        f"conv={conv_id}: reply looks like a repetition loop "
                        f"({_degen}) — skipping memory tail so it cannot be "
                        f"extracted as facts, indexed, or rolled into a summary"
                    )
                if conv_id and not vllm_failed and accumulator.usable() and not _degen:
                    _fire_and_forget(
                        _async_tail(
                            conv_id,
                            touched_facts,
                            last_user_text,
                            accumulator.text(),
                            turn_index,
                            messages,  # original request messages, for rollup
                            injected_facts=injected_facts,
                        ),
                        label=f"tail conv={conv_id}",
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
            # An unexpected response shape leaves assistant_text empty, and the
            # tail below still fires — so the exchange is memorized as a user
            # turn answered by nothing. Indistinguishable from a model that
            # replied with silence unless we say so. Once per process: this is
            # the request path. (v3.1 P0-2b / F61.)
            if logsetup.log_once("nonstream.assistant_text"):
                logger.warning(
                    f"conv={conv_id}: could not read assistant text from the "
                    f"vLLM response ({type(e).__name__}: {e}); this turn is "
                    f"memorized without the model's reply"
                )
        # Same gate the streaming path applies via SseAccumulator.usable().
        # This path had no finish_reason check at all, so a reply vLLM cut off
        # at the generation ceiling was memorized as a completed assistant turn
        # — fact-extracted, indexed into RAG, rolled into summaries. The
        # streaming path guarded the client-disconnect case (F20) and this one
        # guarded nothing, which is the half-applied shape worth watching for.
        _finish_reason = ""
        try:
            _finish_reason = (
                response_json.get("choices", [{}])[0].get("finish_reason") or ""
            )
        except (IndexError, KeyError, TypeError):
            _finish_reason = ""
        if conv_id and _finish_reason == "length":
            # WARNING for the same reason as the streaming path above.
            logger.warning(
                f"conv={conv_id}: reply truncated at the generation ceiling "
                f"(finish_reason=length, {len(assistant_text)} chars) — "
                f"skipping memory tail for the partial reply"
            )
        elif conv_id and (_degen := reply_is_degenerate(assistant_text)):
            logger.warning(
                f"conv={conv_id}: reply looks like a repetition loop "
                f"({_degen}) — skipping memory "
                f"tail so it cannot be extracted as facts, indexed, or rolled "
                f"into a summary"
            )
        elif conv_id:
            _fire_and_forget(
                _async_tail(
                    conv_id,
                    touched_facts,
                    last_user_text,
                    assistant_text,
                    turn_index,
                    messages,  # original request messages, for rollup
                    injected_facts=injected_facts,
                ),
                label=f"tail conv={conv_id}",
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
    return await _clear_all_memory(conv_id, source="admin")


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
    bundle = body.get("bundle")
    if bundle is None:
        raise HTTPException(status_code=400, detail="missing required field: 'bundle'")
    try:
        result = portability.import_conversation(
            bundle,
            target_conv_id=body.get("target_conv_id"),
            overwrite=bool(body.get("overwrite", False)),
        )
    except portability.ImportError_ as e:
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
    try:
        return portability.fork_conversation(
            conv_id, new_conv_id=body.get("new_conv_id")
        )
    except portability.ImportError_ as e:
        raise HTTPException(status_code=400, detail=str(e))


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
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    max_calls = int(body.get("max_calls") or 200)
    dry_run = bool(body.get("dry_run", False))

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

    # _exchange_doc writes "[user]: X\n[assistant]: Y". Split it back
    # into the message pair the summarizer expects.
    messages: list[dict] = []
    for ex in sorted(exchanges, key=lambda e: e.get("turn_index", 0)):
        doc = ex.get("document") or ""
        if "\n[assistant]: " not in doc:
            continue
        u, a = doc.split("\n[assistant]: ", 1)
        messages.append({"role": "user", "content": u.removeprefix("[user]: ")})
        messages.append({"role": "assistant", "content": a})

    before = summarizer.load_state(conv_id)
    # REFUSE rather than pull the watermark backwards.
    #
    # last_summarized_turn is an absolute position in whatever array the LIVE
    # request path last saw. The transcript here is rebuilt from the episodic
    # store, which is lossy by design — it holds only exchanges that indexed
    # successfully. Feeding a shorter reconstruction into maybe_rollup lets
    # _reconcile_watermark pull the watermark back to it, and the turns in
    # between are summarized a second time on the next live turn. Duplicate
    # chunks in her memory is a worse outcome than a command declining to run.
    _wm = before.get("last_summarized_turn", 0)
    if len(messages) < _wm:
        raise HTTPException(
            status_code=409,
            detail=(
                f"refusing: the episodic store rebuilds {len(messages)} "
                f"messages for conv {conv_id}, but the summary watermark is "
                f"already at {_wm}. Running would pull it backwards and "
                f"re-summarize covered turns. The reconstruction is lossy by "
                f"design, so a shortfall means episodic indexing has gaps — "
                f"not that the summaries are behind."
            ),
        )
    plan = {
        "conv_id": conv_id,
        "indexed_exchanges": len(exchanges),
        "reconstructed_messages": len(messages),
        "watermark_before": before.get("last_summarized_turn", 0),
        "l1_before": len(before.get("l1") or []),
        "dry_run": dry_run,
    }
    if dry_run or not messages:
        plan["note"] = (
            "dry run — nothing was written. Re-send with "
            '{"dry_run": false} to run it.'
        )
        return plan

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
