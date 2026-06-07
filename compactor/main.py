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
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

import backfill
import backup as backup_module
import commands
import dedup
import facts
import health
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

# V2.0 Phase 1: admin endpoint binding. Default "127.0.0.1" rejects any
# non-localhost client at the dependency layer (we still bind the FastAPI
# socket to 0.0.0.0 because uvicorn doesn't support dual-listen, but the
# admin paths return 403 unless the client IP is localhost). Set this to
# "0.0.0.0" to expose admin endpoints externally — only safe if you have
# auth/firewall in front.
ADMIN_BIND = os.environ.get("COMPACTOR_ADMIN_BIND", "127.0.0.1").strip()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
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


def _message_text(m: dict) -> str:
    content = m.get("content") or ""
    if isinstance(content, list):
        return " ".join(c.get("text", "") for c in content if isinstance(c, dict))
    return str(content)


def count_tokens(messages: list[dict]) -> int:
    tok = get_tokenizer()
    if tok is not None:
        try:
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            return len(tok.encode(text))
        except Exception:
            total = 0
            for m in messages:
                total += len(tok.encode(_message_text(m))) + 4
            return total
    return sum(len(_message_text(m)) // 4 + 4 for m in messages)


SUMMARY_PROMPT = """You are summarizing an earlier portion of a conversation so it can be compressed into context.

Produce a concise but comprehensive summary that preserves:
- Key facts, names, numbers, decisions, and instructions given
- Any code, file paths, commands, or URLs mentioned
- The user's goals, constraints, and stated preferences
- The state of any in-progress work

Do not editorialize. Do not greet. Output only the summary."""


async def summarize(client: httpx.AsyncClient, to_summarize: list[dict]) -> str:
    transcript = "\n\n".join(
        f"[{m.get('role', 'unknown')}]: {_message_text(m)}" for m in to_summarize
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
    return r.json()["choices"][0]["message"]["content"].strip()


def split_messages(messages: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    system_msgs = [m for m in messages if m.get("role") == "system"]
    non_system = [m for m in messages if m.get("role") != "system"]
    if len(non_system) <= KEEP_RECENT_TURNS:
        return system_msgs, [], non_system
    return system_msgs, non_system[:-KEEP_RECENT_TURNS], non_system[-KEEP_RECENT_TURNS:]


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
    async with httpx.AsyncClient() as client:
        summary = await summarize(client, to_summarize)
    summary_msg = {
        "role": "system",
        "content": f"[Summary of earlier conversation]\n{summary}",
    }
    new_messages = system_msgs + [summary_msg] + keep_recent
    new_count = count_tokens(new_messages)
    logger.info(
        f"compacted: summarized {len(to_summarize)} messages, {current} -> {new_count} tokens"
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
                if not payload or payload == "[DONE]":
                    continue
                try:
                    obj = json.loads(payload)
                    delta = obj.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content")
                    if isinstance(content, str) and content:
                        self._parts.append(content)
                except (json.JSONDecodeError, IndexError, KeyError, TypeError):
                    # Single malformed event — drop it, keep accumulating.
                    pass

    def text(self) -> str:
        return "".join(self._parts)


# Module-level set keeps task references alive so they don't get garbage-
# collected before completing. Standard asyncio fire-and-forget gotcha.
_background_tasks: set[asyncio.Task] = set()


def _fire_and_forget(coro) -> None:
    """Spawn an async task without awaiting it. Keeps the reference so the
    GC doesn't kill it; logs any exception that escapes.
    """
    task = asyncio.create_task(coro)
    _background_tasks.add(task)

    def _log_exception(t: asyncio.Task) -> None:
        _background_tasks.discard(t)
        if t.cancelled():
            return
        exc = t.exception()
        if exc is not None:
            logger.exception(f"background task raised: {exc!r}")

    task.add_done_callback(_log_exception)


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
        # tracking persists across restarts.
        async with conv_lock(conv_id):
            try:
                facts.save_facts(conv_id, touched_facts)
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
                combined = touched_facts + new_entries

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
    yield
    # Graceful: give in-flight fact extractions a moment to finish
    if _background_tasks:
        logger.info(f"waiting for {len(_background_tasks)} background task(s) to finish")
        try:
            await asyncio.wait_for(
                asyncio.gather(*_background_tasks, return_exceptions=True),
                timeout=10.0,
            )
        except asyncio.TimeoutError:
            logger.warning("background tasks didn't finish in 10s; abandoning")


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
# Main request flow
# ---------------------------------------------------------------------------

@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Any:
    body = await request.json()
    messages = body.get("messages", [])

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
        logger.exception(f"compaction failed; forwarding original messages: {e}")

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

    stream = bool(body.get("stream", False))
    client = httpx.AsyncClient(timeout=None)

    if stream:
        accumulator = SseAccumulator()

        async def event_stream():
            try:
                async with client.stream(
                    "POST", f"{VLLM_URL}/v1/chat/completions", json=body
                ) as r:
                    async for chunk in r.aiter_raw():
                        yield chunk
                        accumulator.feed(chunk)
            finally:
                await client.aclose()
                # Fire-and-forget post-response work once the stream is done.
                if conv_id:
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
        r = await client.post(f"{VLLM_URL}/v1/chat/completions", json=body)
        response_json = r.json()
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
        return JSONResponse(content=response_json, status_code=r.status_code)
    finally:
        await client.aclose()


@app.get("/v1/models")
async def models():
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{VLLM_URL}/v1/models", timeout=30.0)
        return JSONResponse(content=r.json(), status_code=r.status_code)


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
