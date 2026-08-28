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
  /forget                  Clear ALL memory for this conv (facts + the
                           archive sidecar + episodic + summary + persona +
                           the dedup refusal memo), leave an empty facts store
                           behind so the lazy backfill cannot reconstruct the
                           history from the message list, then re-read every
                           layer and report what is actually gone rather than
                           what the wipe intended.

                           What it does NOT clear, deliberately: the periodic
                           data-durability snapshots in /data/backups. Those
                           exist so that "a corrupted file, an accidental
                           /forget, or a bad delete is recoverable"
                           (backup.py), they retain for weeks, and nothing
                           restores from them without an operator. They are
                           not an injection source, so the model cannot see a
                           forgotten fact through one — which is why the reply
                           does not raise them. The conversation transcript
                           itself is OpenWebUI's webui.db and is likewise out
                           of scope; /forget is about what the compactor
                           remembers, not about erasing the chat.
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

import bgwork
import facts as facts_module
from memory import StoreUnreadable, conv_lock

logger = logging.getLogger("compactor.commands")

# v3.1 A3: how long /forget waits for in-flight background tails to finish
# before it wipes. Bounded, and the reply says so when the wait expires —
# a /forget that hangs is worse than one that admits it cannot promise
# everything. 10s matches the shutdown drain in main.py.
FORGET_SETTLE_TIMEOUT = 10.0

# v3.1 F22: the mutating handlers take conv_lock, the same per-conv asyncio
# lock the extraction tail and the admin endpoints take. It is imported
# directly rather than injected through ctx (as the remediation plan proposed,
# to dodge an import cycle with main.py) because there is no cycle to dodge:
# conv_lock is defined in memory.py, main.py only re-exports it via
# `from memory import (...)`, and this module already imports from memory. The
# lock object is therefore identical — memory._conv_locks is the one registry —
# and a handler cannot end up unlocked because a caller forgot a ctx key.

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
    # v3.1 F22: load-modify-write, so it holds the per-conv lock for the whole
    # sequence. Unlocked, this raced _async_tail's locked write in both
    # directions: the tail re-reads facts inside its lock, so whichever write
    # landed second won outright. The user watched the compactor confirm
    # "Remembered: ..." and the fact was gone by her next turn.
    async with conv_lock(conv_id):
        existing = facts_module.load_facts(conv_id)
        new_fact = {
            "text": arg,
            "added_turn": ctx.get("turn_index", 0),
            "last_used": now,
        }
        combined = existing + [new_fact]
        # conv_id makes an over-budget eviction here land in the archive
        # sidecar instead of being deleted (v3.1 F9).
        kept, dropped = facts_module.prune_facts(combined, conv_id=conv_id)
        facts_module.save_facts(conv_id, kept)
    extra = f" (archived {dropped} least-recently-used to fit budget)" if dropped else ""
    return f"Remembered: {arg!r}{extra}\nFacts now: {len(kept)}"


async def _settle_background_work() -> bool:
    """Wait for in-flight background tails before a full wipe. True if the
    pool was empty by the time we stopped waiting.

    v3.1 A3. Taking conv_lock inside the wipe is not enough, and the comment
    that claimed it was described a guarantee the lock does not provide.
    bgwork._run awaits the concurrency semaphore BEFORE the coroutine, so a
    tail submitted past MAX_CONCURRENT has executed zero lines and is holding
    a facts snapshot taken before the user typed /forget. asyncio.Lock is
    FIFO, so a /forget arriving while one tail holds the lock queues behind
    every tail already waiting on it and in FRONT of their writes — the
    losing order, for as long as the holder's extraction runs (120s timeout)
    plus its rollup. Reproduced against the real modules: /forget replied
    "Forgot: 2 fact(s)." and the parked tail's extraction was on disk a
    moment later.

    Draining first inverts that: every tail that already exists finishes,
    and only then do we delete. It does not close the window completely — a
    tail submitted after the drain begins is not in the drained set, and the
    airtight fix is a per-conversation wipe generation compared at the write
    sites in main.py's _async_tail. What this does close is the documented
    window, the one that lasts seconds to minutes.

    The drain is process-wide, so a /forget on one conversation waits on
    another's tail. That is a few seconds on a command a user issues rarely
    and deliberately, and it is bounded; a per-conv wait would need bgwork to
    track conv ownership, which is a larger change than this defect earns.
    """
    try:
        await bgwork.pool.drain(timeout=FORGET_SETTLE_TIMEOUT)
        return int(bgwork.pool.stats().get("outstanding", 0)) == 0
    except Exception as e:
        # A failed drain must not stop the wipe — the user asked for this data
        # to be gone, and refusing to delete because we could not wait first
        # leaves strictly more behind. It only costs the guarantee, which the
        # reply then declines to make.
        logger.warning(
            f"conv-wide background drain before /forget failed "
            f"({type(e).__name__}: {e}); wiping anyway and saying so"
        )
        return False


def _memory_residue(conv_id: str) -> tuple[list[str], list[str]]:
    """Re-read every memory layer AFTER a wipe. Returns
    (still_stored_phrases, unreadable_layer_names).

    v3.1 A3. The reply used to be assembled entirely from _clear_all_memory's
    return counters, which describe what the wipe *set out to* delete — not
    what is on disk when the user reads the answer. Those are different
    whenever a background tail lands in between, and they were different
    unconditionally for the archive sidecar, which no wipe path has ever
    touched. Reporting observed state instead of intended action is the only
    version of this that cannot drift: a layer somebody adds later and forgets
    to wire into the wipe shows up here on its own.

    Reads the same four layers /why reads, and for the same reason — this is
    the user's window into her own memory, so a swallowed read error rendered
    as "clean" is the failure, not the fallback.
    """
    still: list[str] = []
    unreadable: list[str] = []

    try:
        n = len(facts_module.load_facts(conv_id))
        if n:
            still.append(f"{n} fact(s)")
    except StoreUnreadable:
        unreadable.append("facts")
    except Exception as e:
        logger.warning(
            f"conv={conv_id}: /forget could not verify the facts layer "
            f"({type(e).__name__}: {e})"
        )

    try:
        n = len(facts_module.load_archive(conv_id))
        if n:
            still.append(f"{n} archived fact(s)")
    except StoreUnreadable:
        unreadable.append("archived facts")
    except Exception as e:
        logger.warning(
            f"conv={conv_id}: /forget could not verify the archive sidecar "
            f"({type(e).__name__}: {e})"
        )

    try:
        import summarizer as summarizer_module
        state = summarizer_module.load_state(conv_id) or {}
        if state.get("l1") or state.get("l2") or state.get("l3"):
            still.append("summary state")
    except StoreUnreadable:
        unreadable.append("summary")
    except Exception as e:
        logger.warning(
            f"conv={conv_id}: /forget could not verify the summary layer "
            f"({type(e).__name__}: {e})"
        )

    try:
        import persona as persona_module
        if persona_module.load_persona(conv_id):
            still.append("persona")
    except StoreUnreadable:
        unreadable.append("persona")
    except Exception as e:
        logger.warning(
            f"conv={conv_id}: /forget could not verify the persona layer "
            f"({type(e).__name__}: {e})"
        )

    try:
        import retrieval as retrieval_module
        n = retrieval_module.conversation_doc_count(conv_id)
        if n is None:
            # Documented as "could not tell", never as zero (v3.1 F61). On a
            # wipe-verification surface those are opposite answers, so this
            # goes to the layer-I-could-not-read sentence, not to silence.
            unreadable.append("indexed exchanges")
        elif n:
            still.append(f"{n} indexed exchange(s)")
    except Exception as e:
        logger.warning(
            f"conv={conv_id}: /forget could not verify the episodic layer "
            f"({type(e).__name__}: {e})"
        )

    return still, unreadable


async def _wipe_all_layers(conv_id: str, clear_all) -> dict:
    """One full pass of the wipe: every layer, in the order that leaves the
    least behind. Returns accumulated counters plus the layers it could not
    read.

    Factored out of _handle_forget because A3's fix runs it more than once —
    see the retry there — and a wipe whose second pass does not do exactly
    what the first did is a wipe that reports a state nobody produced.
    """
    result = await clear_all(conv_id)
    unreadable = list(result.get("unreadable") or [])

    # v3.1 A3: the archive sidecar is a memory layer and until now NOTHING in
    # this codebase has ever deleted one. prune_facts and archive_stale_facts
    # MOVE evicted facts there rather than unlinking them — deliberately, so
    # eviction is recoverable (F9) — and restore_from_archive moves them back
    # into the injected set. _clear_all_memory clears the active facts file and
    # does not know the sidecar exists. Measured: a conversation whose only
    # remaining memory was archived answered "Nothing to forget — this
    # conversation had no stored memory", and /list-archive listed the fact on
    # the next line. selftest._conv_artifact_paths already keeps exactly this
    # enumeration, and says why: "a cleanup that misses one is
    # indistinguishable from a cleanup that worked."
    #
    # Cleared at this surface because this is the command that promises to
    # clear ALL memory for the conversation. The same clear belongs in
    # _clear_all_memory so that DELETE /admin/conversations/{id}/facts stops
    # leaving it behind too; until it moves there, the verification pass in
    # _handle_forget at least reports what the admin path leaves.
    n_archived = 0
    try:
        async with conv_lock(conv_id):
            archived = facts_module.load_archive(conv_id)
            n_archived = len(archived)
            if n_archived:
                facts_module.save_archive(conv_id, [])
    except StoreUnreadable as e:
        # Same rule the facts layer follows: an unknown file is not rewritten
        # from a guess, and the reply says which layer that was.
        logger.error(
            f"conv={conv_id}: /forget could not read the archive sidecar "
            f"({e}); left it in place and told the user"
        )
        unreadable.append("archived facts")
    except Exception as e:
        logger.warning(f"conv={conv_id}: archive clear failed: {e}")

    # v3.1 A3: leave an EMPTY facts store behind rather than no facts store.
    #
    # This is the wipe's tombstone and it does two jobs.
    #
    # The first is the one that matters. backfill.needs_backfill gates on
    # `facts_path(conv_id).is_file()` — its own comment says why: "an
    # existing-but-empty facts file is still a store this module has no
    # business reconstructing." With no file there, the NEXT request on a
    # conversation of four messages or more starts a background extraction
    # over the conversation's whole history and writes the result to disk.
    # Measured against the real modules: a conv whose memory was a summary and
    # a persona was wiped, /forget replied "Forgot: summary state, persona.",
    # and needs_backfill was still True immediately afterwards — so the
    # history the user had just asked to be forgotten was queued to be
    # re-extracted on her next message. Same result with a backfill state of
    # "failed", which is the documented retry path. _clear_all_memory already
    # leaves this file whenever the conv HAD facts (it writes []); the corner
    # cases that reach here without one are exactly the ones that got
    # resurrected, so this makes them behave like the ordinary case.
    #
    # The second is that it is the LAST write of the wipe, so a tail that
    # slipped a fact in behind _clear_all_memory has it removed here rather
    # than merely reported.
    #
    # Skipped when the facts file could not be read: an unknown file is never
    # overwritten from a guess (F1), and that layer is named in the reply.
    #
    # Cost, stated plainly: memory.list_known_conv_ids globs facts/*.json, so
    # a conversation that had no memory at all now appears in
    # /admin/conversations after a /forget. That is the F23 shape and it is
    # accepted here — F23 was a synthetic conv the self-test recreated every
    # 30 seconds, this is a real conversation a real person used, and the file
    # records a decision she made.
    if "facts" not in unreadable:
        try:
            async with conv_lock(conv_id):
                facts_module.save_facts(conv_id, [])
        except Exception as e:
            logger.warning(
                f"conv={conv_id}: could not write the empty-facts tombstone "
                f"({type(e).__name__}: {e}); the lazy backfill may reconstruct "
                f"this conversation's history from its message list"
            )

    return {
        "forgotten_facts": int(result.get("forgotten_facts") or 0),
        "forgotten_episodic": int(result.get("forgotten_episodic") or 0),
        "forgotten_summary": bool(result.get("forgotten_summary")),
        "forgotten_persona": bool(result.get("forgotten_persona")),
        "archived": n_archived,
        "unreadable": unreadable,
    }


async def _handle_forget(arg: str, conv_id: str, ctx: dict) -> str:
    if arg:
        # Selective: remove facts whose text contains the substring
        # (case-insensitive). Other layers untouched.
        #
        # v3.1 F22: same load-modify-write hazard as /remember, running the
        # other way — unlocked, a tail holding a snapshot taken before this
        # command wrote the just-forgotten facts straight back. _merge_touched
        # treats disk as authoritative for membership precisely so a /forget
        # is not undone, and that guarantee only holds if the /forget write
        # happens inside the lock the tail's re-read is serialized against.
        async with conv_lock(conv_id):
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
    #
    # Deliberately NOT wrapped in conv_lock: main._clear_all_memory takes
    # conv_lock itself and asyncio.Lock is not reentrant, so wrapping it
    # here would deadlock the request forever — /forget would hang instead of
    # answering, on the one command a user reaches for when something is wrong.
    clear_all = ctx.get("clear_all_memory")
    if not clear_all:
        return "ERROR: clear_all_memory helper not wired."

    # Drain first, delete second. See _settle_background_work — the order is
    # the whole point.
    settled = await _settle_background_work()

    totals = await _wipe_all_layers(conv_id, clear_all)

    # Drain a SECOND time, after the wipe. The first drain empties the pool of
    # everything that existed when the command arrived; a tail submitted while
    # the wipe itself was running (it holds conv_lock across an extraction and
    # a rollup, so this is not a narrow window) is not in that set. Draining
    # again means the state we are about to verify — and re-clear — is a
    # settled one rather than a moving one. On the ordinary /forget the pool is
    # already empty and this returns immediately.
    settled = await _settle_background_work() and settled

    # Now ask the disk what is actually there, rather than the wipe what it
    # meant to do.
    still, residue_unreadable = _memory_residue(conv_id)

    if still:
        # One bounded retry, and only when the first pass left something
        # behind. "Please run /forget again" is advice we can take ourselves,
        # and the shape it addresses — a tail landing between the wipe and the
        # verification — is precisely the shape a second pass clears. Bounded
        # at one: a layer that survives two wipes and a drain is not losing a
        # race, it is broken, and the reply should say so instead of looping.
        logger.warning(
            f"conv={conv_id}: /forget found {', '.join(still)} still stored "
            f"after the first pass; wiping again before answering"
        )
        retry = await _wipe_all_layers(conv_id, clear_all)
        for key in ("forgotten_facts", "forgotten_episodic", "archived"):
            totals[key] += retry[key]
        for key in ("forgotten_summary", "forgotten_persona"):
            totals[key] = totals[key] or retry[key]
        totals["unreadable"] = retry["unreadable"]
        still, residue_unreadable = _memory_residue(conv_id)

    unreadable = list(totals["unreadable"])
    for layer in residue_unreadable:
        if layer not in unreadable:
            unreadable.append(layer)

    # v3.1 A3: the dedup refusal memo is per-conversation state derived from
    # this conversation's facts (sha256 of the cluster's texts -> the reason
    # the model declined to merge it). Nothing can come back out of it — it
    # holds hashes and one-word reasons, and it is a cache of a deterministic
    # call — so it is cleared silently rather than counted in the reply. It is
    # cleared at all because reset_refusal_memo's own docstring names this
    # caller: "a caller that replaces a conversation wholesale can drop the
    # stale decisions rather than wait for eviction", and a wipe is that.
    try:
        import dedup as dedup_module
        dedup_module.reset_refusal_memo(conv_id)
    except Exception as e:
        logger.warning(
            f"conv={conv_id}: could not clear the dedup refusal memo "
            f"({type(e).__name__}: {e}); it is a cache, so this costs LLM "
            f"calls and not memory"
        )

    parts = []
    if totals["forgotten_facts"]:
        parts.append(f"{totals['forgotten_facts']} fact(s)")
    if totals["archived"]:
        parts.append(f"{totals['archived']} archived fact(s)")
    if totals["forgotten_episodic"]:
        parts.append(f"{totals['forgotten_episodic']} indexed exchange(s)")
    if totals["forgotten_summary"]:
        parts.append("summary state")
    # v3.1 A3: persona was cleared by _clear_all_memory and never mentioned
    # here, so a /forget on a conversation carrying only a persona deleted it
    # and replied "this conversation had no stored memory". Under-reporting a
    # deletion is the same defect as over-reporting one — either way the
    # sentence does not describe what happened.
    if totals["forgotten_persona"]:
        parts.append("persona")

    lines: list[str] = []
    if parts:
        lines.append("Forgot: " + ", ".join(parts) + ".")

    # v3.1: main.py's _clear_all_memory states the contract in its return value
    # — "callers must not report a clean wipe when `unreadable` is non-empty".
    # This is that caller. A corrupt facts file is left on disk with unknown
    # contents, and telling the user the conversation "had no stored memory"
    # would be a degraded mode dressed as a healthy one, on the surface a real
    # person types into.
    if unreadable:
        lines.append(
            f"I could not read the stored "
            f"{' and '.join(unreadable)} for this conversation, so I left "
            f"{'it' if len(unreadable) == 1 else 'them'} untouched rather than "
            f"overwrite something I can't see. Nothing was lost. Please "
            f"mention this if it keeps happening."
        )
        if parts:
            lines[0] = "Cleared: " + ", ".join(parts) + "."

    if still:
        lines.append(
            "This conversation is still storing " + ", ".join(still) + ". "
            "I could not remove that in this pass — please run /forget again, "
            "and mention it if it keeps coming back."
        )

    if not settled:
        # Honest partial success. The wipe happened; the guarantee that
        # nothing re-adds behind it did not.
        lines.append(
            "Background work from earlier turns was still finishing while I "
            "wiped, so a little of it may reappear. Run /forget again if you "
            "see something come back."
        )

    if not lines:
        return "Nothing to forget — this conversation had no stored memory."
    return " ".join(lines)


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
