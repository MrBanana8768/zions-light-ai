"""
compactor.commands — V2.1 Phase 5: chat command surface.

The user types a slash command in their chat (e.g. "/list-facts" or
"/remember the protagonist is left-handed"). We detect commands in the
chat_completions handler BEFORE memory loading + vLLM proxy, synthesize
an OpenAI-shaped completion containing the command output, and return
it directly. Zero LLM cost, instant response, command never reaches
the model.

Commands (case-insensitive command name, args preserved as-is):

  /help                    List available commands
  /list-facts or /facts    Show current facts
  /list-archive            Show archived (cold-storage) facts
  /remember <text>         Manually add a fact
  /forget                  Clear ALL memory for this conv (facts + episodic
                           + summary + persona). Equivalent to the admin
                           /forget endpoint.
  /forget <substring>      Remove only facts whose text contains substring
                           (case-insensitive)
  /why                     Show what the next request would have injected:
                           facts that would inject, retrieval candidates for
                           recent conv tail, summary state

Detection rule: message starts with `/`, first whitespace-delimited token
(after stripping the leading `/`) matches a known command name. Anything
else (paths like "/usr/bin/...", code blocks starting with /, etc.)
passes through to vLLM untouched.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Awaitable

import facts as facts_module
from memory import StoreUnreadable

logger = logging.getLogger("compactor.commands")

# Command name → handler. Handlers take (arg_string, conv_id, ctx) and
# return the user-visible response text. ctx is a dict so handlers can
# reach the modules they need without import cycles.
HandlerResult = str
Handler = Callable[[str, str, dict], Awaitable[HandlerResult]]


# ---------------------------------------------------------------------------
# Detection / parsing
# ---------------------------------------------------------------------------

# Aliases: each alias resolves to a canonical handler name.
_ALIASES: dict[str, str] = {
    "facts": "list-facts",
    "list_facts": "list-facts",  # tolerant of underscore variant
    "list-archive": "list-archive",
    "archive": "list-archive",
    "remember": "remember",
    "forget": "forget",
    "why": "why",
    "why-did-you-say-that": "why",
    "help": "help",
    "?": "help",
    "list-facts": "list-facts",
}


def parse_command(text: str) -> tuple[str | None, str]:
    """Parse a user message for a slash command.

    Returns (canonical_command_name, arg_string) if recognized, else
    (None, ""). Recognition is permissive — surrounding whitespace and
    case ignored on the command name; arg is everything after the first
    whitespace, stripped.

    Non-command messages (no `/` prefix or unknown command) return
    (None, "") so the caller can pass them through to vLLM unmodified.
    """
    if not text:
        return None, ""
    stripped = text.lstrip()
    if not stripped.startswith("/"):
        return None, ""
    # Drop the leading slash, split on first whitespace.
    head = stripped[1:].split(None, 1)
    if not head:
        return None, ""
    name = head[0].lower()
    arg = head[1].strip() if len(head) > 1 else ""
    canonical = _ALIASES.get(name)
    if not canonical:
        return None, ""
    return canonical, arg


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

async def _handle_help(arg: str, conv_id: str, ctx: dict) -> str:
    return (
        "Available commands:\n"
        "  /list-facts          Show what I'm remembering for this conversation\n"
        "  /list-archive        Show archived (cold-storage) facts\n"
        "  /remember <text>     Manually add a fact\n"
        "  /forget              Clear ALL memory for this conversation\n"
        "  /forget <substring>  Remove only facts matching the substring\n"
        "  /why                 Show what would be injected on the next turn\n"
        "  /help                This message"
    )


async def _handle_list_facts(arg: str, conv_id: str, ctx: dict) -> str:
    facts = facts_module.load_facts(conv_id)
    if not facts:
        return "No facts stored for this conversation yet."
    lines = [f"Current facts ({len(facts)}):"]
    for f in facts:
        lines.append(f"  - {f['text']}")
    return "\n".join(lines)


async def _handle_list_archive(arg: str, conv_id: str, ctx: dict) -> str:
    archived = facts_module.load_archive(conv_id)
    if not archived:
        return "No archived facts for this conversation."
    lines = [f"Archived facts ({len(archived)}):"]
    for f in archived:
        lines.append(f"  - {f['text']}")
    return "\n".join(lines)


async def _handle_remember(arg: str, conv_id: str, ctx: dict) -> str:
    if not arg:
        return "Usage: /remember <fact text>"
    if len(arg) > 500:
        return f"Fact too long ({len(arg)} chars) — keep it under 500."
    now = int(time.time())
    existing = facts_module.load_facts(conv_id)
    new_fact = {
        "text": arg,
        "added_turn": ctx.get("turn_index", 0),
        "last_used": now,
    }
    combined = existing + [new_fact]
    kept, dropped = facts_module.prune_facts(combined)
    facts_module.save_facts(conv_id, kept)
    extra = f" (pruned {dropped} oldest to fit budget)" if dropped else ""
    return f"Remembered: {arg!r}{extra}\nFacts now: {len(kept)}"


async def _handle_forget(arg: str, conv_id: str, ctx: dict) -> str:
    if arg:
        # Selective: remove facts whose text contains the substring
        # (case-insensitive). Other layers untouched.
        existing = facts_module.load_facts(conv_id)
        needle = arg.lower()
        to_keep = [f for f in existing if needle not in f.get("text", "").lower()]
        removed = len(existing) - len(to_keep)
        if removed == 0:
            return f"No facts matched {arg!r}."
        facts_module.save_facts(conv_id, to_keep)
        return f"Forgot {removed} fact(s) matching {arg!r}. {len(to_keep)} remaining."

    # No arg: full wipe — call into the shared clear-all-memory helper
    # provided via ctx (avoids import cycles with main.py).
    clear_all = ctx.get("clear_all_memory")
    if not clear_all:
        return "ERROR: clear_all_memory helper not wired."
    result = await clear_all(conv_id)
    parts = []
    if result.get("forgotten_facts"):
        parts.append(f"{result['forgotten_facts']} fact(s)")
    if result.get("forgotten_episodic"):
        parts.append(f"{result['forgotten_episodic']} indexed exchange(s)")
    if result.get("forgotten_summary"):
        parts.append("summary state")
    # v3.1: main.py's _clear_all_memory states the contract in its return value
    # — "callers must not report a clean wipe when `unreadable` is non-empty".
    # This is that caller. A corrupt facts file is left on disk with unknown
    # contents, and telling the user the conversation "had no stored memory"
    # would be a degraded mode dressed as a healthy one, on the surface a real
    # person types into.
    unreadable = result.get("unreadable") or []
    if unreadable:
        cleared = ("Cleared " + ", ".join(parts) + ". ") if parts else ""
        return (
            f"{cleared}I could not read the stored "
            f"{' and '.join(unreadable)} for this conversation, so I left "
            f"{'it' if len(unreadable) == 1 else 'them'} untouched rather than "
            f"overwrite something I can't see. Nothing was lost. Please "
            f"mention this if it keeps happening."
        )
    if not parts:
        return "Nothing to forget — this conversation had no stored memory."
    return "Forgot: " + ", ".join(parts) + "."


async def _handle_why(arg: str, conv_id: str, ctx: dict) -> str:
    """Show what would be injected on the next turn — the user's view
    into the compactor's memory injection. Uses current state (close
    enough to 'what was just injected' for diagnostic purposes; we
    don't keep per-turn injection snapshots in V2.1).
    """
    facts = facts_module.load_facts(conv_id)
    summary_state = None
    retrieval_count = None
    try:
        import summarizer as summarizer_module
        summary_state = summarizer_module.load_state(conv_id)
    except Exception as e:
        # /why is the user's only window into her own memory. A swallowed
        # read error rendered as "(none)" tells her the layer is empty when
        # it may be corrupt — the same lie the health endpoint told.
        # (v3.1 P0-2b / F61.)
        logger.warning(
            f"conv={conv_id}: /why could not read summary state "
            f"({type(e).__name__}: {e}); it renders as '(none)', which is "
            f"indistinguishable from a conversation that has no summary yet"
        )
    try:
        import retrieval as retrieval_module
        retrieval_count = retrieval_module.conversation_doc_count(conv_id)
    except Exception as e:
        logger.warning(
            f"conv={conv_id}: /why could not read episodic count "
            f"({type(e).__name__}: {e}); reported as unavailable"
        )

    lines = ["Memory state for this conversation:"]
    if facts:
        lines.append(f"  Facts ({len(facts)}):")
        for f in facts:
            lines.append(f"    - {f['text']}")
    else:
        lines.append("  Facts: (none)")

    if summary_state:
        l1 = summary_state.get("l1") or []
        l2 = summary_state.get("l2") or []
        l3 = summary_state.get("l3")
        lines.append(
            f"  Summary stack: L1={len(l1)} L2={len(l2)} L3={'yes' if l3 else 'no'}"
        )
    else:
        lines.append("  Summary stack: (none)")

    if retrieval_count is not None:
        lines.append(f"  Indexed exchanges (episodic): {retrieval_count}")
    else:
        lines.append("  Indexed exchanges (episodic): (unavailable)")

    persona = ctx.get("persona_text")
    if persona:
        excerpt = persona[:200] + ("…" if len(persona) > 200 else "")
        lines.append(f"  Persona ({len(persona)} chars): {excerpt}")

    return "\n".join(lines)


_HANDLERS: dict[str, Handler] = {
    "help": _handle_help,
    "list-facts": _handle_list_facts,
    "list-archive": _handle_list_archive,
    "remember": _handle_remember,
    "forget": _handle_forget,
    "why": _handle_why,
}


async def handle_command(
    canonical: str, arg: str, conv_id: str, ctx: dict | None = None
) -> str:
    """Dispatch to the right handler. ctx carries injectables (helpers,
    turn_index, etc.) so handlers don't need to know main.py shape."""
    handler = _HANDLERS.get(canonical)
    if not handler:
        return f"Unknown command: {canonical!r}. Type /help for the list."
    try:
        return await handler(arg, conv_id, ctx or {})
    except StoreUnreadable as e:
        # v3.1: must be caught HERE, not only in main.py. This catch-all is a
        # superclass handler sitting in front of it, so main.py's plain-language
        # StoreUnreadable branch could never fire for anything raised inside a
        # command — only for the persona text, which main.py evaluates eagerly
        # while building ctx. Facts, archive, summary and retrieval all landed
        # here and returned the leaky string below.
        #
        # StoreUnreadable.__str__ embeds an absolute filesystem path
        # (memory.py:307). A non-technical user typed "/forget" into a chat box;
        # she gets plain language, the operator gets the path in the log.
        logger.error(
            f"command {canonical!r} failed for conv={conv_id} — store unreadable: {e}"
        )
        return (
            "I couldn't read my stored memory for this conversation just now, "
            "so I've made no changes rather than risk losing anything. Nothing "
            "has been deleted. This is a problem on my side — please try again "
            "in a moment, and mention it if it keeps happening."
        )
    except Exception as e:
        logger.exception(f"command {canonical!r} failed for conv={conv_id}: {e}")
        return f"Command failed: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Synthetic chat completion response
# ---------------------------------------------------------------------------

def build_synthetic_completion(content: str, model: str) -> dict:
    """Build an OpenAI chat-completion-shaped response with the command's
    output as the assistant's reply. OpenWebUI renders it as a normal
    assistant bubble in the conversation.

    No vLLM tokens used — usage fields are zero.
    """
    return {
        "id": f"chatcmpl-cmd-{int(time.time() * 1000):x}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model or "compactor-command",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


def build_synthetic_completion_stream(content: str, model: str) -> list[dict]:
    """SSE-shaped sequence for streaming clients. Returns the list of
    chunks that main.py joins with 'data: ' prefixes. The first chunk
    carries role + initial content; the final [DONE] marker is added
    by main.py.
    """
    cid = f"chatcmpl-cmd-{int(time.time() * 1000):x}"
    created = int(time.time())
    return [
        {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model or "compactor-command",
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": content},
                "finish_reason": None,
            }],
        },
        {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model or "compactor-command",
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }],
        },
    ]
