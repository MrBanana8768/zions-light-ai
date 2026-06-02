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
  restart mid-backfill knows to resume rather than re-extract everything
  (or worse, mark facts file as incomplete forever).
- **Stale-detection** so a crashed backfill (process killed, OOM, etc.)
  gets retried on next encounter rather than blocking memory creation
  for that conv forever.
- **Idempotent.** Multiple concurrent calls to `maybe_start_backfill`
  for the same conv only start one task (lock + state check).

Storage:
    /data/openwebui/compactor/facts/<conv_id>.backfill.json

Format:
    {
      "conv_id": "...",
      "state": "in_progress" | "complete" | "failed",
      "started_at": "2026-05-28T...",
      "updated_at": "2026-05-28T...",
      "exchanges_done": 5,
      "exchanges_total": 47,
      "error": null
    }
"""

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

import facts as facts_module
import retrieval
import summarizer
from memory import (
    atomic_write_json,
    conv_lock,
    facts_path,
    read_json,
    storage_root,
)

logger = logging.getLogger("compactor.backfill")


# How long an "in_progress" state can sit without updates before we
# consider it crashed and retry. 10 minutes covers the longest plausible
# backfill (~2000 message conversation at 300ms/call), with margin.
_STALE_SECONDS = 600


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
    AND has no facts file yet AND no in-progress (non-stale) backfill.
    """
    if len(messages) < _MIN_MESSAGES_FOR_BACKFILL:
        return False  # too short to bother
    if facts_path(conv_id).is_file():
        return False  # facts already exist — V2 took over from new
    state = read_state(conv_id)
    if state is None:
        return True  # never attempted
    s = state.get("state")
    if s == "complete":
        return False  # already done
    if s == "in_progress" and not is_stale(state):
        return False  # someone else is on it
    # "failed" or stale "in_progress" → retry
    return True


# ---------------------------------------------------------------------------
# The backfill itself
# ---------------------------------------------------------------------------

async def _run_backfill(
    conv_id: str,
    messages: list[dict],
    vllm_url: str,
    model: str,
) -> None:
    """The actual backfill: iterate pairs, extract facts, save state
    incrementally so a crash mid-run can resume from progress.

    Errors during individual extractions are logged but don't fail the
    whole backfill — we keep going and save whatever we got.
    """
    pairs = extract_user_assistant_pairs(messages)
    if not pairs:
        logger.info(f"conv={conv_id}: backfill skipped — no user/assistant pairs found")
        return

    started_at = _now_iso()
    _write_state(conv_id, {
        "state": "in_progress",
        "started_at": started_at,
        "exchanges_done": 0,
        "exchanges_total": len(pairs),
        "error": None,
    })
    logger.info(f"conv={conv_id}: backfill starting over {len(pairs)} exchange(s)")

    accumulated: list[dict] = []
    now_unix = _now_unix()

    try:
        async with httpx.AsyncClient() as client:
            for i, (user_text, asst_text) in enumerate(pairs, start=1):
                try:
                    new_strs = await facts_module.extract_facts_from_exchange(
                        client, vllm_url, model, user_text, asst_text, accumulated
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
                    "error": None,
                })

        # Done iterating — prune to budget and persist as the facts file
        async with conv_lock(conv_id):
            kept, dropped = facts_module.prune_facts(accumulated)
            facts_module.save_facts(conv_id, kept)

        # V2.0 Phase 4: also build hierarchical summary state for this conv,
        # so the model gets continuity-of-narrative on the *next* request
        # rather than having to wait for natural rollups (which require new
        # turns to accumulate). maybe_rollup drains as many L1→L2→L3 layers
        # as the existing message history justifies. Failure is non-fatal:
        # facts backfill is still considered complete.
        try:
            if summarizer.enabled():
                await summarizer.maybe_rollup(conv_id, messages, vllm_url, model)
        except Exception as e:
            logger.warning(f"conv={conv_id}: backfill summary rollup failed (non-fatal): {e}")

        _write_state(conv_id, {
            "state": "complete",
            "started_at": started_at,
            "exchanges_done": len(pairs),
            "exchanges_total": len(pairs),
            "facts_kept": len(kept),
            "facts_pruned": dropped,
            "error": None,
        })
        logger.info(
            f"conv={conv_id}: backfill complete — {len(kept)} facts kept, "
            f"{dropped} pruned, from {len(pairs)} exchanges"
        )
    except Exception as e:
        logger.exception(f"conv={conv_id}: backfill aborted: {e}")
        _write_state(conv_id, {
            "state": "failed",
            "started_at": started_at,
            "exchanges_done": 0,
            "exchanges_total": len(pairs),
            "error": str(e)[:500],
        })
    finally:
        _in_progress_local.discard(conv_id)


async def start_backfill_if_needed(
    conv_id: str,
    messages: list[dict],
    vllm_url: str,
    model: str,
    *,
    fire_and_forget,
) -> bool:
    """Public entry point. Returns True if a backfill was started,
    False if it wasn't needed.

    `fire_and_forget` is the caller's task spawner (main.py's
    _fire_and_forget) so this module doesn't need to know about the
    background-task registry. Decouples from main.py for testing.
    """
    if not needs_backfill(conv_id, messages):
        return False
    if conv_id in _in_progress_local:
        return False  # already started in this process
    _in_progress_local.add(conv_id)
    # Snapshot messages — caller may mutate the list before backfill runs
    snapshot = [dict(m) for m in messages]
    fire_and_forget(_run_backfill(conv_id, snapshot, vllm_url, model))
    return True
