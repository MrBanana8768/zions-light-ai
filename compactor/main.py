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
import time
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
TARGET_TOKENS = _env_int("COMPACTOR_TARGET_TOKENS", int(MAX_MODEL_LEN * 0.75))
KEEP_RECENT_TURNS = _env_int("COMPACTOR_KEEP_RECENT_TURNS", 4)
SUMMARY_MAX_TOKENS = _env_int("COMPACTOR_SUMMARY_MAX_TOKENS", 1024)
# Slack left inside MAX_MODEL_LEN when budgeting a summarization call's INPUT
# (covers the system prompt, the wrapper text, and chat-template overhead).
SUMMARY_INPUT_RESERVE = _env_int("COMPACTOR_SUMMARY_INPUT_RESERVE", 2048)
# Hard ceiling for what we will forward to vLLM. Anything above this is a
# guaranteed 400, so the guard sheds content rather than letting the request
# fail. The reserve leaves the model room to actually generate a reply.
GENERATION_RESERVE = _env_int("COMPACTOR_GENERATION_RESERVE", 2048)
# Clamped to MAX_MODEL_LEN: a bare floor could sit ABOVE the model's own window
# on a small-context model, which would defeat the entire point of the guard.
HARD_INPUT_LIMIT = min(MAX_MODEL_LEN, max(256, MAX_MODEL_LEN - GENERATION_RESERVE))
# V3.1 (Vision): a single image in a VLM request costs far more than its
# text — hundreds to a couple thousand tokens depending on resolution and
# the model's vision encoder. The text-only token estimate misses this
# entirely, so we add a flat per-image estimate to the budget. Conservative
# default keeps us from overflowing the model's real context window; tune
# per VLM if needed.
IMAGE_TOKEN_ESTIMATE = _env_int("COMPACTOR_IMAGE_TOKENS", 1536)
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


def _note_backend_rejection(err_body: str) -> None:
    """Reactive backstop: if vLLM rejects a request because the model is not
    multimodal, remember that — the NEXT request strips image parts and the
    conversation heals instead of staying poisoned."""
    global _backend_multimodal
    if "not a multimodal model" in (err_body or "") and _backend_multimodal is not False:
        _backend_multimodal = False
        logger.warning(
            "backend declared itself text-only via a 400; image parts will be "
            "stripped from subsequent requests (set COMPACTOR_BACKEND_MULTIMODAL "
            "to override)"
        )


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


def count_tokens(messages: list[dict]) -> int:
    # V3.1: images cost tokens the text estimate can't see — add a flat
    # per-image estimate so VLM requests don't quietly overflow the budget.
    image_tokens = sum(_message_image_count(m) for m in messages) * IMAGE_TOKEN_ESTIMATE
    tok = get_tokenizer()
    if tok is not None:
        try:
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            return len(tok.encode(text)) + image_tokens
        except Exception:
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


def _chunk_to_budget(turns: list[dict], budget: int) -> list[list[dict]]:
    """Split turns into consecutive batches that each fit `budget` tokens.

    A single turn larger than the budget still gets its own batch — we never
    drop content here; `_summarize_once` would fail on it and the caller
    degrades. (Truncating a turn silently would be a quieter kind of lying.)
    """
    batches: list[list[dict]] = []
    current: list[dict] = []
    current_tokens = 0
    for m in turns:
        t = count_tokens([m])
        if current and current_tokens + t > budget:
            batches.append(current)
            current, current_tokens = [], 0
        current.append(m)
        current_tokens += t
    if current:
        batches.append(current)
    return batches


async def summarize(client: httpx.AsyncClient, to_summarize: list[dict]) -> str:
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
    batches = _chunk_to_budget(to_summarize, budget)
    if len(batches) == 1:
        return await _summarize_once(client, batches[0])

    logger.info(
        f"summarize: {len(to_summarize)} turns exceed the {budget}-token input "
        f"budget — map-reduce over {len(batches)} batches"
    )
    # Map: batches run CONCURRENTLY (vLLM batches fine), bounded by a small
    # semaphore so a huge history can't monopolize the engine. Sequential
    # batches added multi-minute latency on long conversations (rc6 review).
    sem = asyncio.Semaphore(4)

    async def _bounded(batch: list[dict]) -> str:
        async with sem:
            return await _summarize_once(client, batch)

    parts = [p for p in await asyncio.gather(*(_bounded(b) for b in batches)) if p]

    # Reduce: fold the partials hierarchically, never handing _summarize_once
    # an input over its budget (its documented contract — the first cut of
    # this code violated it whenever the reduce step actually fired). Each
    # round groups the partials to the budget and summarizes each group;
    # bounded rounds, and any failure degrades to plain concatenation.
    rounds = 0
    while len(parts) > 1 and rounds < 3:
        rounds += 1
        part_msgs = [{"role": "user", "content": p} for p in parts]
        groups = _chunk_to_budget(part_msgs, budget)
        if all(len(g) == 1 for g in groups):
            break  # nothing can be folded further without breaking the budget
        try:
            parts = [p for p in await asyncio.gather(*(_bounded(g) for g in groups)) if p]
        except Exception as e:
            logger.warning(f"summarize reduce round {rounds} failed, using concatenation: {e}")
            break
    return "\n\n".join(parts)


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
        summary = await summarize(client, text_only)
    summary_msg = {
        "role": "system",
        "content": f"[Summary of earlier conversation]\n{summary}",
    }
    # Order: system → summary-of-old-text → preserved image turns → recent.
    new_messages = system_msgs + [summary_msg] + preserved_images + keep_recent
    new_count = count_tokens(new_messages)
    logger.info(
        f"compacted: summarized {len(text_only)} text turn(s), "
        f"preserved {len(preserved_images)} image turn(s), "
        f"{current} -> {new_count} tokens"
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


def _enforce_hard_budget(messages: list[dict], limit: int | None = None) -> list[dict]:
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
    """
    if limit is None:
        limit = HARD_INPUT_LIMIT

    # Prescreen: skip the (expensive) full tokenization when the char-based
    # estimate is far under the limit. The estimate can undercount token-dense
    # text, so the margin is 2x; a pathological miss just means the request
    # reaches vLLM and gets the 400 this guard would otherwise have prevented —
    # degraded, not corrupted.
    if _fast_token_estimate(messages) <= limit // 2:
        return messages

    total = count_tokens(messages)
    if total <= limit:
        return messages

    msgs = list(messages)
    # Per-message costs, computed ONCE. Sum-of-parts differs from the templated
    # whole by per-message template overhead, so shedding aims below the limit
    # on arithmetic and then verifies with a real count — bounded rounds.
    per = [count_tokens([m]) for m in msgs]
    running = total
    dropped = 0
    trimmed = 0

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
        while running > limit and trimmed < 32:
            big = [
                i for i, m in enumerate(msgs)
                if m.get("role") == "system"
                and isinstance(m.get("content"), str)
                and len(m["content"]) > 400
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
            per[i] = count_tokens([msgs[i]])  # recount ONLY the trimmed block
            running += per[i]
            trimmed += 1

        # --- verify with a real full count; loop only if template overhead
        #     pushed us back over (each round does exactly ONE full count) ---
        running = count_tokens(msgs)
        if running <= limit:
            break
        idxs = [i for i, m in enumerate(msgs) if m.get("role") != "system"]
        if len(idxs) <= 1 and trimmed >= 32:
            break  # nothing left to shed; forward best effort

    logger.warning(
        f"hard budget enforced: {total} -> {running} tokens "
        f"(limit {limit}); dropped {dropped} old turn(s), "
        f"trimmed {trimmed} injected block(s)"
    )
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

    def feed(self, chunk: bytes) -> None:
        try:
            self._buffer += chunk.decode("utf-8", errors="replace")
        except Exception:
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
                    if choice.get("finish_reason"):
                        self._complete = True
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


def _fire_and_forget(coro) -> None:
    """Spawn post-response background work through the bounded pool
    (V2.3 Theme 3). The pool caps concurrency and sheds beyond a hard
    outstanding ceiling rather than spawning unboundedly under load. Task
    references are kept alive by the pool; exceptions are logged there.
    """
    bgwork.pool.submit(coro)


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
    if assistant_text and last_user_text:
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
                facts.save_facts(conv_id, _merge_touched(facts.load_facts(conv_id), touched_facts))
            except Exception as e:
                logger.warning(f"conv={conv_id}: touched-save failed: {e}")
        return

    if not assistant_text or not last_user_text:
        return

    async with conv_lock(conv_id):
        try:
            async with httpx.AsyncClient() as client:
                new_strs = await facts.extract_facts_from_exchange(
                    client,
                    VLLM_URL,
                    MODEL_REPO or "",
                    last_user_text,
                    assistant_text,
                    touched_facts,
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
                        combined, removed = await dedup.dedup_facts(
                            client, VLLM_URL, MODEL_REPO or "", combined
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

            kept, dropped = facts.prune_facts(combined)
            facts.save_facts(conv_id, kept)
            if new_entries or dropped:
                logger.info(
                    f"conv={conv_id}: +{len(new_entries)} facts, pruned {dropped}, "
                    f"total {len(kept)}"
                )
        except Exception as e:
            logger.exception(f"conv={conv_id}: async fact tail failed: {e}")

    # --- 3. Hierarchical summary rollup (Phase 4) ---
    # Runs OUTSIDE the facts lock since maybe_rollup acquires its own
    # conv_lock internally — nesting the same lock would deadlock.
    if summarizer.enabled() and assistant_text:
        try:
            full_messages = list(original_messages) + [
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


REQUEST_REJECTED_MESSAGE = (
    "That request couldn't be processed — the model backend rejected it "
    "(this is a problem on my side, not yours). The conversation and its "
    "memory are safe. If this keeps happening on the same message, the "
    "operator should check the compactor logs for the rejection reason."
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
    injected_blocks: list[str] = []
    log_parts: list[str] = []
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
                injected_blocks.append(pblock)
                log_parts.append(f"persona({len(ptext)}ch)")
        except Exception as e:
            logger.warning(f"conv={conv_id}: persona handling failed (non-fatal): {e}")

        # --- Facts (Phase 2) ---
        try:
            touched_facts = facts.load_facts(conv_id)
            if touched_facts:
                facts.touch_facts(touched_facts)
                block = facts.format_facts_block(touched_facts)
                if block:
                    injected_blocks.append(block)
                    log_parts.append(f"{len(touched_facts)}fact(s)")
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
                injected_blocks.append(rblock)
                log_parts.append(f"{len(hits)}retr")
        except Exception as e:
            logger.warning(f"conv={conv_id}: retrieval load failed (non-fatal): {e}")

        # --- Hierarchical summary stack (Phase 4) ---
        # State only grows via the async tail (rollups post-response),
        # so this is a purely local read — no LLM call on the hot path.
        try:
            sstate = summarizer.load_state(conv_id)
            sblock = summarizer.format_summary_block(sstate)
            if sblock:
                injected_blocks.append(sblock)
                log_parts.append(
                    f"sum(L1={len(sstate.get('l1') or [])}"
                    f"/L2={len(sstate.get('l2') or [])}"
                    f"/L3={'y' if sstate.get('l3') else 'n'})"
                )
        except Exception as e:
            logger.warning(f"conv={conv_id}: summary load failed (non-fatal): {e}")

        # Single inject point — preserves Mistral template compatibility.
        if injected_blocks:
            combined = "\n\n".join(injected_blocks)
            try:
                body["messages"] = inject_system_block(body["messages"], combined)
                logger.info(f"conv={conv_id}: injected memory [{' '.join(log_parts)}]")
            except Exception as e:
                logger.warning(f"conv={conv_id}: memory injection failed (non-fatal): {e}")

        # Lazy backfill: if this is an existing V1 conv that has no facts
        # file yet, kick off a background extraction over its full history.
        # Doesn't block this request — current request just degrades to
        # "no facts injected" and next request will see the facts.
        try:
            started = await backfill.start_backfill_if_needed(
                conv_id,
                messages,  # use original messages, not compacted
                VLLM_URL,
                MODEL_REPO or "",
                fire_and_forget=_fire_and_forget,
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
    #    would chew into the persona. It also accounts for the request's OWN
    #    max_tokens: vLLM enforces prompt + max_tokens <= window, so a fixed
    #    reserve alone leaves a client asking for a big completion still 400able.
    # 2. _merge_adjacent_system_messages runs LAST, collapsing every remaining
    #    run — including any adjacency the budget guard created by deleting a
    #    turn that sat between two system messages.
    try:
        req_max_tokens = int(body.get("max_tokens") or 0)
    except (TypeError, ValueError):
        req_max_tokens = 0
    if req_max_tokens > MAX_MODEL_LEN // 2:
        # Pair with the reserve cap below so prompt+completion always fits.
        req_max_tokens = MAX_MODEL_LEN // 2
        body["max_tokens"] = req_max_tokens
    effective_limit = min(
        MAX_MODEL_LEN,
        max(256, MAX_MODEL_LEN - max(GENERATION_RESERVE, req_max_tokens)),
    )
    # Pure CPU (tokenizer) work — off the event loop so a shedding pass on a
    # huge conversation can't stall every other request and the healthchecks.
    body["messages"] = await run_in_threadpool(
        _enforce_hard_budget, body["messages"], effective_limit
    )
    body["messages"] = _merge_adjacent_system_messages(body["messages"])
    # ...and non-system turns that ended up sharing a role (compaction hoists
    # image turns out of chronological order, which lands user next to user).
    body["messages"] = _merge_consecutive_same_role(body["messages"])

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
                            vllm_failed = True
                            err_body = (await r.aread()).decode("utf-8", "replace")[:300]
                            _note_backend_rejection(err_body)
                            logger.warning(
                                f"vLLM HTTP {r.status_code} on stream: {err_body!r}"
                            )
                            for chunk in _vllm_unreachable_stream_chunks(
                                body.get("model") or MODEL_REPO or "",
                                # A 4xx means the backend is HEALTHY and refused
                                # our request; only 5xx/unreachable justifies the
                                # "starting up or restarting" message.
                                message=(
                                    REQUEST_REJECTED_MESSAGE
                                    if r.status_code < 500
                                    else None
                                ),
                            ):
                                yield f"data: {json.dumps(chunk)}\n\n".encode()
                            yield b"data: [DONE]\n\n"
                        else:
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
                if conv_id and not vllm_failed and not accumulator.complete():
                    logger.info(
                        f"conv={conv_id}: stream ended without completion "
                        f"({len(accumulator.text())} chars accumulated) — "
                        f"skipping memory tail for the partial reply"
                    )
                if conv_id and not vllm_failed and accumulator.complete():
                    _fire_and_forget(
                        _async_tail(
                            conv_id,
                            touched_facts,
                            last_user_text,
                            accumulator.text(),
                            turn_index,
                            messages,  # original request messages, for rollup
                        )
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
        # Extract assistant text for fact extraction
        assistant_text = ""
        try:
            assistant_text = (
                response_json.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                or ""
            )
        except (IndexError, KeyError, TypeError):
            pass
        if conv_id:
            _fire_and_forget(
                _async_tail(
                    conv_id,
                    touched_facts,
                    last_user_text,
                    assistant_text,
                    turn_index,
                    messages,  # original request messages, for rollup
                )
            )
        if r.status_code >= 400:
            _note_backend_rejection(str(response_json)[:300])
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
    report = await health.gather_health_full(VLLM_URL, TARGET_TOKENS)
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
    info = storage_summary(conv_id)
    # Facts (Phase 2)
    try:
        info["facts"]["count"] = len(facts.load_facts(conv_id))
    except Exception:
        info["facts"]["count"] = None
    # Episodic memory (Phase 3)
    try:
        info["episodic"] = {
            "indexed_exchanges": retrieval.conversation_doc_count(conv_id),
        }
    except Exception:
        info["episodic"] = {"indexed_exchanges": None}
    # Hierarchical summary (Phase 4)
    try:
        info["summary"] = summarizer.state_summary(summarizer.load_state(conv_id))
    except Exception:
        info["summary"] = None
    # Persona (V2.1 Phase 8)
    try:
        prec = persona.load_persona(conv_id)
        info["persona"] = {
            "present": prec is not None,
            "length": len(prec["persona_text"]) if prec else 0,
            "source": prec["source"] if prec else None,
        }
    except Exception:
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
        existing = facts.load_facts(conv_id)
        n_facts = len(existing)
        if n_facts > 0:
            facts.save_facts(conv_id, [])
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
                client, VLLM_URL, MODEL_REPO or "", before
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
