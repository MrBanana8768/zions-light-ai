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
  /tidy                    Dry run: report the extraction debris in this
                           conversation's fact store, grouped by the rule that
                           matched it, and separately report the rows that look
                           odd but are being KEPT because no rule can prove they
                           are garbage. Changes nothing. Ends with a code.
  /tidy apply <code>       Apply exactly the plan that code was issued for.
                           Writes a verified whole-conversation snapshot to
                           disk first, then moves the removed rows into the
                           archive sidecar (/list-archive, restorable) rather
                           than unlinking them. Refuses if the plan has changed
                           since the dry run, or if it would take more than
                           half the store.
  /retire <conv_id>        Dry run: report how ANOTHER conversation's whole
                           memory would be emptied into this one — which facts
                           move, which are not worth moving, which the
                           destination already has, and every other layer keyed
                           to that id that would be cleared. Changes nothing.
                           Ends with a code.
  /retire <conv_id> apply <code>
                           Apply exactly that plan. Writes one verified
                           whole-conversation snapshot of the source first
                           (facts, archive, summaries, episodic AND persona),
                           refuses if any layer could not be accounted for,
                           MOVES the surviving facts into this conversation,
                           and only then clears the source. Idempotent: run it
                           again after an interruption and it finishes.
  /why                     Show what the next request would have injected:
                           facts that would inject, retrieval candidates for
                           recent conv tail, summary state

Detection rule: message starts with `/`, first whitespace-delimited token
(after stripping the leading `/`) matches a known command name. Anything
else (paths like "/usr/bin/...", code blocks starting with /, etc.)
passes through to vLLM untouched.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from typing import Any, Callable, Awaitable

import bgwork
import facts as facts_module
import portability
from memory import StoreUnreadable, conv_lock, storage_root

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
    # v3.1 D6. No single-letter or near-miss alias: /tidy is the only command
    # here that can remove a fact without naming it, and it should take a
    # deliberate keystroke to reach.
    "tidy": "tidy",
    "tidy-facts": "tidy",
    "tidy_facts": "tidy",
    "cleanup": "tidy",
    # v3.1 D8. Same rule as /tidy and more so: this one empties an entire
    # conversation's memory. One spelling, plus the underscore variant the
    # rest of this table already tolerates. No abbreviation.
    "retire": "retire",
    "retire-conversation": "retire",
    "retire_conversation": "retire",
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
        "  /pin <substring>     Always send matching facts, whatever the topic\n"
        "  /pin                 List what is pinned\n"
        "  /unpin <substring>   Stop always sending them\n"
        "  /forget              Clear ALL memory for this conversation\n"
        "  /forget <substring>  Remove only facts matching the substring\n"
        "  /tidy                Show extraction debris I could clean up "
        "(changes nothing)\n"
        "  /tidy apply <code>   Clean up exactly what that dry run listed\n"
        "  /retire <conv_id>    Show how I would empty ANOTHER conversation's "
        "memory into this one (changes nothing)\n"
        "  /retire <conv_id> apply <code>\n"
        "                       Do exactly what that dry run listed\n"
        "  /why                 Show what would be injected on the next turn\n"
        "  /help                This message"
    )


async def _handle_list_facts(arg: str, conv_id: str, ctx: dict) -> str:
    facts = facts_module.load_facts(conv_id)
    if not facts:
        return "No facts stored for this conversation yet."
    lines = [f"Current facts ({len(facts)}):"]
    for f in facts:
        # The pin marker is the only visible sign that ranking cannot drop
        # this one; without it an operator cannot tell the tiers apart.
        lines.append(f"  {'[pinned] ' if f.get('pin') else '- '}{f['text']}")
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


async def _handle_pin(arg: str, conv_id: str, ctx: dict) -> str:
    """Pin facts so relevance ranking can never drop them.

    F1 made fact injection top-K by relevance against the current message,
    which is what stops a 1,500-token block of everything from being sent
    every turn. The cost is that a fact only reaches the model when it looks
    relevant, and identity does not: "her name is X" scores near zero on a
    turn about dinner. Pinned facts bypass ranking and the budget entirely.

    Without this command the pinned tier was unreachable code - facts.py
    exposes set_pinned() and nothing called it - so the safety existed only
    on paper.
    """
    if not arg:
        pinned = [f for f in facts_module.load_facts(conv_id) if f.get("pin")]
        if not pinned:
            return (
                "Nothing is pinned yet.\n"
                "Usage: /pin <substring>   — always send facts matching it\n"
                "       /unpin <substring> — stop always sending them\n"
                "Pin the handful that must reach me every turn regardless of "
                "topic: names, who we are to each other, standing preferences."
            )
        lines = [f"Pinned facts ({len(pinned)}) — always sent, never ranked:"]
        lines += [f"  - {f['text']}" for f in pinned]
        return "\n".join(lines)

    async with conv_lock(conv_id):
        current = facts_module.load_facts(conv_id)
        n = facts_module.set_pinned(current, text_substring=arg, pinned=True)
        if n:
            facts_module.save_facts(conv_id, current)
    if not n:
        return f"No facts matched {arg!r}. /list-facts shows what I have."
    return (
        f"Pinned {n} fact(s) matching {arg!r}. They will now reach me on "
        f"every turn, whatever we are talking about."
    )


async def _handle_unpin(arg: str, conv_id: str, ctx: dict) -> str:
    if not arg:
        return "Usage: /unpin <substring>   (/pin with no argument lists them)"
    async with conv_lock(conv_id):
        current = facts_module.load_facts(conv_id)
        n = facts_module.set_pinned(current, text_substring=arg, pinned=False)
        if n:
            facts_module.save_facts(conv_id, current)
    if not n:
        return f"No pinned facts matched {arg!r}."
    return f"Unpinned {n} fact(s) matching {arg!r}."


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


# ---------------------------------------------------------------------------
# /tidy — cleaning a store that is already polluted (v3.1 D6)
# ---------------------------------------------------------------------------
#
# The store this is for holds a real person's memory of her own life mixed in
# with extraction debris. An earlier assumption that the same bucket was
# synthetic nearly produced a destructive delete. So the design premise is not
# "find the garbage"; it is:
#
#     wrongly keeping a garbage row costs tokens.
#     wrongly removing a row costs her a memory of her own life.
#
# Those are not comparable, so this does not try to balance them. Every rule
# below is written to be *provably* content-free, and the classifier's default
# — for every row no rule matched, and for every row a rule was unsure about —
# is KEEP.
#
# Three consequences worth stating plainly, because they are choices and not
# oversights:
#
#   1. **No rule judges meaning.** Nothing here decides a fact is trivial,
#      wrong, redundant, or no longer true. Every removal rule fires on
#      structure only: a row that is the extractor's own format vocabulary, a
#      row with no letters and no digits anywhere in it, or a row byte-identical
#      to one being kept. A human is the only thing that can do the other job,
#      and the AMBIGUOUS list exists to hand her a short list instead of
#      hundreds of rows.
#   2. **It will therefore under-remove.** An over-eager extraction of a real
#      but pointless detail is indistinguishable, structurally, from an
#      extraction of a real and important one. That row stays.
#   3. **The rules are the reviewable surface.** The dry run prints every
#      string it would remove, grouped under the rule that matched it, so a bad
#      rule is visible as a block of good facts under its heading BEFORE
#      anything is written.
#
# Where this code lives is a compromise: classification is fact-store logic and
# belongs in facts.py next to prune_facts. It is here because commands.py is
# the surface that owns it today and facts.py is being edited concurrently. If
# a second caller ever needs these rules, move them — do not copy them.

# Applying is refused above this share of the store. A rule that matches half
# a conversation's memory is a broken rule, not a thorough one, and the
# operator should see that as a refusal rather than as a large success
# message. The flat allowance lets a small store be cleaned at all — 5 rows of
# punctuation out of 8 is 62%, and is still 5 rows of punctuation.
TIDY_MAX_REMOVED_FRACTION = 0.5
TIDY_MIN_REMOVED_ALLOWANCE = 5

# Display caps. The confirmation code is computed over the FULL texts, never
# over the truncated display form, so a truncated line can never hide a
# difference between what was shown and what gets removed. The row cap applies
# ONLY to the flagged-and-kept list — the removal list is never elided, because
# a row the operator confirms without having seen it is the exact failure this
# command exists to prevent.
TIDY_MAX_ROWS_SHOWN = 25
TIDY_MAX_CHARS_SHOWN = 300

# Matches /remember's own cap. A stored row longer than the longest thing a
# human is allowed to type by hand is a summary or a paste, not a fact — worth
# a look, never worth an automatic delete.
TIDY_OVERSIZED_CHARS = 500
TIDY_TRIVIAL_CHARS = 15
TIDY_DECORATION_FRACTION = 0.30

# The 2026-08-28 incident's decoration: box drawing, block elements, geometric
# shapes. One assistant reply held 2,151 of these characters. They are only
# ever flagged here, never removed — a row that is *all* decoration has no
# letters and no digits and is caught by `no-content` on its own merits.
_TIDY_DECORATION_RANGES = ((0x2500, 0x257F), (0x2580, 0x259F), (0x25A0, 0x25FF))

# Emitted by facts._fit_extraction_input when the exchange is trimmed to fit
# the extraction budget. Duplicated as a literal rather than imported from a
# private name, so a rename over there turns this into a rule that stops
# matching rather than an ImportError at boot.
_TIDY_TRIM_MARK = "trimmed to fit the fact-extraction input budget"

# Whole-row matches only. Every string here is a piece of the extraction
# prompt's own format vocabulary (facts._EXTRACTION_SYSTEM_PROMPT,
# _build_extraction_messages) or a refusal token that survived
# _parse_extraction_output. A row that is exactly one of these carries no
# information in any conversation, which is the bar for automatic removal.
_TIDY_SCAFFOLD_TEXTS = frozenset({
    "none", "n/a", "nothing", "null", "empty",
    "fact", "facts", "extracted facts", "existing facts", "latest exchange",
    "user", "assistant", "system", "output", "output format",
    "no facts", "no new facts", "no facts extracted", "no new information",
    "no relevant facts", "no new facts to extract", "none of the above",
    "here are the facts", "here are the extracted facts",
})

# Flagged, never removed. Each of these CAN begin a real fact, so the rule
# earns a place on the operator's short list and nothing more.
_TIDY_SCAFFOLD_PREFIXES = (
    "[user]:", "[assistant]:", "[system]:",
    "existing facts:", "latest exchange:", "extracted facts:",
)
_TIDY_META_PREFIXES = (
    "as an ai", "i cannot ", "i can't ", "i am unable", "i'm unable",
    "i'm sorry", "i am sorry", "sorry, ",
    "here are the ", "here is the ", "the following ", "based on the ",
    "note: ", "output: ", "no new facts ", "there are no ",
)

_TIDY_WS = re.compile(r"\s+")
_TIDY_EDGE = ' \t\r\n*_`"\'“”‘’.,:;!?-–—•'


def _tidy_norm(text: str) -> str:
    """Fold a stored row to the form the whole-row rules compare against.

    Deliberately shallow: lowercase, collapse whitespace, and strip markdown
    emphasis / quotes / trailing punctuation from the ENDS only. It does not
    strip words, reorder, or transliterate — a normalizer that changes what a
    row says is a normalizer that can make two different facts look like the
    same one, and this function's output decides what gets deleted.
    """
    if not text:
        return ""
    return _TIDY_WS.sub(" ", text).strip(_TIDY_EDGE).strip().lower()


def _tidy_decoration_fraction(text: str) -> float:
    if not text:
        return 0.0
    n = sum(
        1
        for ch in text
        if any(lo <= ord(ch) <= hi for lo, hi in _TIDY_DECORATION_RANGES)
    )
    return n / len(text)


def _tidy_removal_rule(text: str) -> str | None:
    """The rule that proves this row is content-free, or None to keep it.

    None is the answer for everything this function is not certain about.
    """
    norm = _tidy_norm(text)
    if norm in _TIDY_SCAFFOLD_TEXTS:
        return "scaffolding"
    # No letter and no digit anywhere — in ANY script, since str.isalpha() is
    # true for CJK, Cyrillic, Greek and the rest. What is left is punctuation,
    # whitespace, box-drawing and emoji: "- -", "...", "**", "━━━━━━".
    if not any(ch.isalpha() or ch.isdigit() for ch in text):
        return "no-content"
    return None


def _tidy_flag_rule(text: str) -> str | None:
    """The reason a KEPT row is worth a human's attention, or None.

    Order is precedence: the first match is the one reported. Nothing here
    removes anything, so a false positive costs one line of reading.
    """
    norm = _tidy_norm(text)
    lowered = text.lower()
    if any(norm.startswith(p) for p in _TIDY_SCAFFOLD_PREFIXES):
        return "transcript-fragment"
    if _TIDY_TRIM_MARK in lowered:
        return "truncation-marker"
    if not any(ch.isalpha() for ch in text):
        # Digits but no letters. A bare date or number MIGHT be something she
        # asked to be remembered, so this is the operator's call, not ours.
        return "numeric-only"
    if _tidy_decoration_fraction(text) >= TIDY_DECORATION_FRACTION:
        return "decorative"
    if len(text) > TIDY_OVERSIZED_CHARS:
        return "oversized"
    if any(norm.startswith(p) for p in _TIDY_META_PREFIXES):
        return "meta-commentary"
    if len(norm) < TIDY_TRIVIAL_CHARS:
        return "very-short"
    return None


# Rule id -> the sentence printed under its heading. A rule with no
# explanation here is a rule the operator cannot review, so the renderer
# treats a missing entry as a bug and says so rather than printing a bare id.
_TIDY_RULE_TEXT: dict[str, str] = {
    "scaffolding": "the extractor's own format vocabulary, stored as if it were a fact",
    "no-content": "no letter and no digit anywhere in the row",
    "duplicate": "byte-identical to another row, which is kept",
    "transcript-fragment": "starts with a transcript or prompt label — may still contain something real after it",
    "truncation-marker": "contains the fact-extraction truncation marker",
    "numeric-only": "digits but no letters — could be a date or a number she asked me to keep",
    "decorative": f"at least {int(TIDY_DECORATION_FRACTION * 100)}% box-drawing or block characters",
    "oversized": f"longer than {TIDY_OVERSIZED_CHARS} characters — longer than /remember allows",
    "meta-commentary": "reads like the extractor talking about itself",
    "very-short": f"under {TIDY_TRIVIAL_CHARS} characters once normalized",
    "near-duplicate": "same text as another kept row once case and spacing are folded — which wording is the real one is a judgment this does not make",
}

_TIDY_REMOVE_ORDER = ("scaffolding", "no-content", "duplicate")
_TIDY_FLAG_ORDER = (
    "transcript-fragment", "truncation-marker", "numeric-only",
    "decorative", "oversized", "meta-commentary", "very-short",
    "near-duplicate",
)


def _tidy_survivor_index(indices: list[int], rows: list[dict]) -> int:
    """Which copy of a byte-identical group to keep.

    Highest (last_used, added_turn). Both are the eviction sort keys in
    facts._lru_split, so keeping the maximum guarantees the surviving copy is
    the one that would have outlived the others anyway: collapsing duplicates
    can never move a fact FORWARD in the eviction queue. That matters because
    dedup.py's merge already does the opposite (added_turn = min over the
    cluster, MEMORY_REVIEW F-1) and pulls consolidated facts toward eviction.
    This must not add a second mechanism doing that.
    """
    return max(
        indices,
        key=lambda i: (
            int(rows[i].get("last_used", 0) or 0),
            int(rows[i].get("added_turn", 0) or 0),
            -i,
        ),
    )


def _tidy_plan(rows: list[dict]) -> dict:
    """Classify a facts list. Pure function of the list — no I/O, no clock,
    no randomness — so the same store always yields the same plan and the
    same confirmation code.
    """
    remove: dict[str, list[int]] = {}
    kept_idx: list[int] = []

    for i, f in enumerate(rows):
        rule = _tidy_removal_rule(f.get("text", "") or "")
        if rule:
            remove.setdefault(rule, []).append(i)
        else:
            kept_idx.append(i)

    # Duplicates, over the survivors only: a row already going to the archive
    # must not also be counted as the duplicate of a row that is staying.
    by_text: dict[str, list[int]] = {}
    for i in kept_idx:
        by_text.setdefault(rows[i].get("text", "") or "", []).append(i)
    dupes: list[int] = []
    for group in by_text.values():
        if len(group) > 1:
            survivor = _tidy_survivor_index(group, rows)
            dupes.extend(i for i in group if i != survivor)
    if dupes:
        remove.setdefault("duplicate", []).extend(sorted(dupes))
        dropped = set(dupes)
        kept_idx = [i for i in kept_idx if i not in dropped]

    flags: dict[str, list[int]] = {}
    for i in kept_idx:
        rule = _tidy_flag_rule(rows[i].get("text", "") or "")
        if rule:
            flags.setdefault(rule, []).append(i)

    # Near-duplicates among survivors: same normalized form, different bytes.
    # Reported only — choosing which wording survives is a judgment about
    # meaning, and this operation does not make those.
    by_norm: dict[str, list[int]] = {}
    for i in kept_idx:
        by_norm.setdefault(_tidy_norm(rows[i].get("text", "") or ""), []).append(i)
    near = sorted(
        i
        for group in by_norm.values()
        if len(group) > 1
        for i in group
    )
    if near:
        flags.setdefault("near-duplicate", []).extend(near)

    remove_idx = sorted(i for ids in remove.values() for i in ids)
    texts = sorted((rows[i].get("text", "") or "") for i in remove_idx)
    # Over the full texts, NUL-joined — the same construction dedup._cluster_key
    # uses, and for the same reason: no fact text contains a NUL, so ["ab","c"]
    # cannot collide with ["a","bc"].
    token = hashlib.sha256(
        "\x00".join(texts).encode("utf-8", "replace")
    ).hexdigest()[:12]

    return {
        "total": len(rows),
        "remove_by_rule": remove,
        "remove_idx": remove_idx,
        "flag_by_rule": flags,
        "keep_idx": kept_idx,
        "token": token,
    }


def _tidy_allowance(total: int) -> int:
    return max(TIDY_MIN_REMOVED_ALLOWANCE, int(total * TIDY_MAX_REMOVED_FRACTION))


def _tidy_show(text: str) -> str:
    """One row, as the operator sees it.

    repr() on purpose: a `no-content` row is invisible rendered raw, and a
    review surface that renders the evidence as blank space is not a review
    surface.
    """
    if len(text) > TIDY_MAX_CHARS_SHOWN:
        return f"{text[:TIDY_MAX_CHARS_SHOWN]!r}… ({len(text)} chars total)"
    return repr(text)


def _tidy_group_lines(
    rows: list[dict],
    by_rule: dict[str, list[int]],
    order: tuple[str, ...],
    *,
    limit: int | None,
    rule_text: dict[str, str] | None = None,
    show: Callable[[Any], str] | None = None,
) -> list[str]:
    """Render one section, grouped by rule.

    `limit=None` for the removal section, always. A row the operator is asked
    to confirm and was not shown is the whole failure this command is designed
    against, so the removal list is never elided — the blast-radius cap is what
    bounds its length, not the renderer. Flagged rows are only a reading list,
    so those groups do get capped.

    `rule_text` / `show` exist so /retire (D8) can render its own rule
    vocabulary and its own row addressing through this function rather than
    growing a second copy of it. Both default to /tidy's behaviour, so the D6
    call sites are unchanged.
    """
    table = _TIDY_RULE_TEXT if rule_text is None else rule_text
    render = show or (lambda i: _tidy_show(rows[i].get("text", "") or ""))
    lines: list[str] = []
    seen = list(order) + [r for r in by_rule if r not in order]
    for rule in seen:
        idx = by_rule.get(rule)
        if not idx:
            continue
        why = table.get(
            rule, "NO DESCRIPTION FOR THIS RULE — do not confirm this plan"
        )
        lines.append(f"  [{rule}] {len(idx)} row(s) — {why}")
        shown = idx if limit is None else idx[:limit]
        for i in shown:
            lines.append(f"      {render(i)}")
        if len(shown) < len(idx):
            lines.append(f"      … and {len(idx) - len(shown)} more not shown")
    return lines


def _tidy_render_plan(rows: list[dict], plan: dict) -> str:
    total = plan["total"]
    n_remove = len(plan["remove_idx"])
    n_flag = sum(len(v) for v in plan["flag_by_rule"].values())

    lines = [
        "Fact cleanup — DRY RUN. Nothing has been changed.",
        "",
        f"This conversation has {total} stored fact(s).",
        "",
    ]

    if n_remove:
        pct = (100 * n_remove) // total if total else 0
        lines.append(f"WOULD REMOVE {n_remove} of {total} ({pct}%):")
        lines.extend(
            _tidy_group_lines(
                rows, plan["remove_by_rule"], _TIDY_REMOVE_ORDER, limit=None
            )
        )
    else:
        lines.append("WOULD REMOVE nothing — no row matched a removal rule.")
    lines.append("")

    if n_flag:
        lines.append(
            f"KEEPING {n_flag} row(s) that look odd but are not provably "
            f"garbage. I will not touch these; read them yourself:"
        )
        lines.extend(
            _tidy_group_lines(
                rows, plan["flag_by_rule"], _TIDY_FLAG_ORDER,
                limit=TIDY_MAX_ROWS_SHOWN,
            )
        )
        lines.append("")

    if n_remove:
        allowance = _tidy_allowance(total)
        if n_remove > allowance:
            lines.append(
                f"I will NOT apply this plan: {n_remove} rows is more than the "
                f"{allowance}-row limit for a store this size. A rule that "
                f"matches this much of a conversation's memory is a broken "
                f"rule, not a thorough one. Send this output to whoever "
                f"maintains the rules."
            )
        else:
            lines.append(
                "Before anything is deleted I write a full, verified snapshot "
                "of this conversation to disk, and the removed rows go to this "
                "conversation's archive — /list-archive shows them and they can "
                "be restored individually. Nothing is unlinked."
            )
            lines.append("")
            lines.append(f"To apply exactly this plan:   /tidy apply {plan['token']}")
            lines.append(
                "That code covers the exact rows listed above. If the removal "
                "set changes before you confirm, the code stops working and "
                "you get a fresh plan instead of a surprise."
            )
    return "\n".join(lines)


async def _handle_tidy(arg: str, conv_id: str, ctx: dict) -> str:
    parts = arg.split()
    mode = parts[0].lower() if parts else ""

    if mode in ("", "dry-run", "dryrun", "plan", "preview", "show"):
        rows = facts_module.load_facts(conv_id)
        if not rows:
            return "No facts stored for this conversation — nothing to tidy."
        return _tidy_render_plan(rows, _tidy_plan(rows))

    if mode != "apply":
        return (
            f"Unknown option {parts[0]!r}.\n"
            "  /tidy              show what would be removed (changes nothing)\n"
            "  /tidy apply <code> remove exactly what that dry run listed"
        )

    token = parts[1].lower() if len(parts) > 1 else ""
    if not token:
        return (
            "/tidy apply needs the code from a dry run. Run /tidy first and "
            "read what it proposes — the code exists so that nothing is "
            "removed that you have not seen."
        )

    # One lock for load → classify → snapshot → archive → save. Same shape as
    # /remember, and for the same reason: unlocked, the extraction tail's own
    # load-modify-write lands on top of this one and whichever wrote second
    # wins outright.
    async with conv_lock(conv_id):
        rows = facts_module.load_facts(conv_id)
        if not rows:
            return "No facts stored for this conversation — nothing to tidy."

        # Re-planned from disk, never carried over from the dry run. The plan
        # is a pure function of the stored rows, so this is the compare half of
        # a compare-and-swap: if the removal set still hashes to the code you
        # were given, the rows about to be archived are the rows you read.
        # A tail adding a NEW good fact in between does not change the removal
        # set and does not invalidate the code — only a change to what would be
        # removed does.
        plan = _tidy_plan(rows)
        n_remove = len(plan["remove_idx"])

        if not n_remove:
            return (
                "Nothing to remove — no row matches a removal rule right now. "
                "Run /tidy to see the rows I am flagging for you to read."
            )

        if plan["token"] != token:
            return (
                f"That code is out of date — the set of rows I would remove "
                f"has changed since the dry run, so I have removed nothing.\n\n"
                + _tidy_render_plan(rows, plan)
            )

        allowance = _tidy_allowance(plan["total"])
        if n_remove > allowance:
            return (
                f"Refusing: this plan removes {n_remove} of {plan['total']} "
                f"rows, over the {allowance}-row limit for a store this size. "
                f"Nothing has been changed. A rule that matches this much of a "
                f"conversation's memory needs fixing, not confirming."
            )

        # Archive before removing. Both halves, in this order:
        #
        #   1. A verified whole-conversation snapshot on disk. This is the
        #      backstop for a bug in this code — if the classification itself
        #      is wrong, per-row restore does not help, because the rows would
        #      go back one at a time only if someone knew which ones. Raises
        #      rather than returning a bad path, and a raise here means nothing
        #      below runs.
        #   2. facts.archive_facts — the sidecar, which is how a fact has left
        #      the active set since F9. /list-archive shows it, and
        #      restore_from_archive puts individual rows back with no operator
        #      tooling at all. This is the route the user can drive herself.
        try:
            snap = portability.quarantine_conversation(
                conv_id, reason="tidy-facts"
            )
        except portability.QuarantineError as e:
            logger.error(
                f"conv={conv_id}: /tidy apply aborted — could not write a "
                f"verified snapshot: {e}"
            )
            return (
                "I could not write a verified backup of this conversation "
                "first, so I have changed nothing. Nothing has been removed. "
                "This is a problem on my side — please mention it."
            )

        remove_rows = [rows[i] for i in plan["remove_idx"]]
        keep_rows = [rows[i] for i in plan["keep_idx"]]

        # Sidecar first, active set second — the ordering archive_facts's own
        # docstring insists on. Interrupted in between, the row is in both
        # files, which is recoverable; the other order loses it outright. On a
        # re-run the same rows match the same rules, archive_facts REPLACES
        # rather than appends, and the second pass is a no-op: idempotent.
        n_archived = facts_module.archive_facts(conv_id, remove_rows)
        facts_module.save_facts(conv_id, keep_rows)

        # Report what is on disk, not what the operation intended — the same
        # rule /forget's verification pass follows.
        after = len(facts_module.load_facts(conv_id))

    # Counts only in the log. Fact text is real personal memory and does not
    # go to an operator's terminal or a log file; it goes to the chat reply
    # the owner asked for and nowhere else.
    logger.info(
        f"conv={conv_id}: /tidy apply removed {n_remove} row(s) "
        f"({', '.join(f'{r}={len(v)}' for r, v in sorted(plan['remove_by_rule'].items()))}), "
        f"{plan['total']} -> {after}; snapshot={snap['path'].name}"
    )

    lines = [
        f"Removed {n_remove} row(s). This conversation now has {after} fact(s) "
        f"(was {plan['total']}).",
        "",
        f"All {n_archived} of them are in this conversation's archive — "
        f"/list-archive shows them, and any one of them can be put back.",
        f"A full snapshot of the conversation as it was a moment ago is at "
        f"{snap['path']} ({snap['facts']} fact(s), {snap['episodic']} indexed "
        f"exchange(s)).",
    ]
    if snap["unverified_layers"]:
        lines.append(
            "One caveat: I could not verify the "
            + " and ".join(snap["unverified_layers"])
            + " in that snapshot. The facts layer — the one I just changed — "
            "was verified."
        )
    if after != len(keep_rows):
        # A tail cannot have run inside the lock, so this means the write did
        # not do what it said. Say so rather than reporting the intent.
        lines.append(
            f"Warning: I expected {len(keep_rows)} fact(s) to remain and read "
            f"back {after}. Please run /tidy again and mention this."
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# /retire — emptying a phantom conversation WITHOUT losing what is in it
# (v3.1 D8)
# ---------------------------------------------------------------------------
#
# WHY THIS IS NOT /tidy, having read /tidy first.
#
# D6 cleans debris out of a bucket the user keeps using. It removes only rows
# it can PROVE are content-free, keeps everything else, and refuses outright if
# a plan would take more than half the store — because taking half of a live
# conversation's memory is a broken rule, not a thorough one.
#
# This operation empties a bucket the user is never going to use again, and
# every one of those properties is wrong for that job:
#
#   - It has to take 100% of the store, so /tidy's blast-radius cap would
#     refuse it by design.
#   - The rows it does NOT remove cannot simply stay, because the whole point
#     is that nothing stays. They have to go somewhere, and "somewhere" is
#     another conversation. /tidy has no concept of a destination.
#   - The facts are not the only thing keyed to the id. Retiring a conv_id and
#     leaving its episodic vectors, its summary chain and its persona behind
#     retires nothing.
#
# So it is a genuinely different operation, and it is built on D6 rather than
# beside it: the classifier calls facts.is_storable_fact and D6's own
# _tidy_removal_rule / _tidy_flag_rule / _tidy_norm / _tidy_survivor_index, the
# report goes through D6's _tidy_group_lines, and the snapshot goes through
# D6's portability.quarantine_conversation. What is new here is the
# destination, the layer sweep, and the two-conversation lock.
#
# THE ONE ASYMMETRY THAT DECIDES EVERY RULE BELOW, restated because it is even
# sharper here than in D6:
#
#     wrongly keeping a garbage row costs tokens in ANOTHER conversation.
#     wrongly dropping a row costs her a memory of her own life, and the
#     bucket it lived in no longer exists to go back to.
#
# Hence: anything not provably content-free and not provably already present
# in the destination MIGRATES. There is no "probably duplicate" branch.
#
# WHAT THIS OPERATION DOES NOT DO: it does not stop the bucket coming back.
# Nothing in this file can. See the note at the end of this section.

# The charset memory._sanitize reduces a conv_id to, and its length cap. A
# conv_id typed into a chat box is used to build filesystem paths, so this
# REFUSES anything outside the charset rather than silently stripping it the
# way the request path does: on the request path a mangled id costs a wrong
# bucket, here it would cost the wrong conversation being emptied.
_RETIRE_CONV_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# Migrated rows are not being deleted, so eliding the list costs a reader some
# scrolling and nothing else. The DROPPED list is never elided — same rule, and
# the same reason, as D6's removal list.
RETIRE_MAX_MIGRATE_ROWS_SHOWN = 40

_RETIRE_RULE_TEXT: dict[str, str] = {
    **_TIDY_RULE_TEXT,
    "markup": (
        "facts.is_storable_fact says this is markup, not a fact — a code "
        "fence, a markdown heading, a JSON key, an unclosed bracket, a "
        "fabricated status-dashboard line, or a row with no alphanumeric "
        "character in any script"
    ),
    "duplicate-in-source": (
        "byte-identical to another row of this same conversation that IS "
        "being moved"
    ),
    "already-in-destination": (
        "byte-identical to a fact the destination conversation already holds "
        "in its active set"
    ),
    "already-in-destination-archive": (
        "byte-identical to a fact the destination conversation already holds "
        "in cold storage — /list-archive there shows it and it can be "
        "restored"
    ),
    "near-duplicate-in-destination": (
        "same text as a destination fact once case and spacing are folded, "
        "but NOT byte-identical — so it is being moved, not dropped. Which "
        "wording is the real one is a judgement this does not make"
    ),
}

_RETIRE_DROP_ORDER = (
    "markup",
    "scaffolding",
    "no-content",
    "duplicate-in-source",
    "already-in-destination",
    "already-in-destination-archive",
)
_RETIRE_FLAG_ORDER = ("near-duplicate-in-destination",) + _TIDY_FLAG_ORDER


def _retire_removal_rule(text: str) -> str | None:
    """The rule that proves this row is not worth moving, or None to move it.

    facts.is_storable_fact FIRST. That is the predicate the write paths already
    share — extraction, /remember, backfill and dedup's merged canonical text
    all gate on it (facts.py:837-846) — and this operation must not invent a
    second definition of "not a fact" that could disagree with it.

    Applying it to ALREADY-STORED rows is the thing facts.py:772-777 warns
    against, and the warning is about WHERE, not whether: "WRITE PATH ONLY.
    This is deliberately NOT applied in load_facts or save_facts... a filter
    there would silently delete already-stored entries on the next unrelated
    write — an irreversible cleanup of a live store, smuggled in as a parser
    fix. Cleaning what is already stored is a separate, deliberate, reversible
    operation." This is that operation: dry-run by default, snapshot first,
    every dropped row printed verbatim, nothing unlinked.

    Then D6's own removal rules, which catch the extractor's format vocabulary
    ("NONE", "assistant", "EXISTING FACTS:") that is_storable_fact accepts
    because it is syntactically ordinary prose. Both are existing, reviewed
    predicates; neither is redefined here.
    """
    if not facts_module.is_storable_fact(text):
        return "markup"
    return _tidy_removal_rule(text)


def _retire_items(
    source_active: list[dict], source_archive: list[dict]
) -> list[tuple[str, dict]]:
    """The source conversation's whole fact memory as one addressable list.

    ACTIVE FIRST, and that ordering is load-bearing: when the same text is in
    both files, the hot copy is the one that migrates and the cold one is the
    duplicate. The other way round would move a fact the user is actively
    being reminded of into cold storage.

    The archive sidecar is in here at all because it is memory. /forget's own
    wipe path records that "NOTHING in this codebase has ever deleted one"
    before it did, and a retirement that unlinked it would destroy every fact
    that had ever been evicted from this bucket for budget — silently, since
    nothing else reads it.
    """
    return (
        [("active", f) for f in source_active]
        + [("archive", f) for f in source_archive]
    )


def _retire_plan(
    items: list[tuple[str, dict]],
    dest_active: list[dict],
    dest_archive: list[dict],
    *,
    source_id: str,
    dest_id: str,
) -> dict:
    """Classify every row of the source's fact memory. Pure function of its
    four inputs — no I/O, no clock — so the same pair of stores always yields
    the same plan and the same confirmation code.

    Positions in the returned lists index `items`.
    """
    rows = [f for _, f in items]
    drop: dict[str, list[int]] = {}
    survivors: list[int] = []

    # 1. Structure. Nothing here looks at the destination or at meaning.
    for pos, (_layer, f) in enumerate(items):
        rule = _retire_removal_rule(f.get("text", "") or "")
        if rule:
            drop.setdefault(rule, []).append(pos)
        else:
            survivors.append(pos)

    # 2. Exact duplicates WITHIN the source, across both its layers.
    #
    # Which copy survives is D6's _tidy_survivor_index — highest
    # (last_used, added_turn) — for the reason its docstring gives: those are
    # the eviction sort keys, so keeping the maximum guarantees collapsing
    # duplicates can never move a fact FORWARD in the eviction queue. An
    # active-layer copy outranks an archived one regardless, per _retire_items.
    by_text: dict[str, list[int]] = {}
    for pos in survivors:
        by_text.setdefault(rows[pos].get("text", "") or "", []).append(pos)
    dropped_dupes: set[int] = set()
    for group in by_text.values():
        if len(group) < 2:
            continue
        hot = [p for p in group if items[p][0] == "active"]
        keeper = _tidy_survivor_index(hot or group, rows)
        for p in group:
            if p != keeper:
                dropped_dupes.add(p)
    if dropped_dupes:
        drop.setdefault("duplicate-in-source", []).extend(sorted(dropped_dupes))
        survivors = [p for p in survivors if p not in dropped_dupes]

    # 3. Already in the destination.
    #
    # EXACT TEXT MATCH, and only exact. This is the one comparison in the
    # operation that can discard a real memory, so it is the one held to a
    # standard nothing can argue with: byte-identical strings carry identical
    # information, so dropping one of them loses nothing that the destination
    # does not already hold. Every looser test — normalized, token overlap,
    # embedding distance — has a threshold, and a threshold has a wrong side
    # whose cost is a memory of her own life against a saving of a few dozen
    # tokens. Near-matches are FLAGGED and MIGRATED, below; they are never a
    # reason to drop.
    dest_active_texts = {f.get("text", "") or "" for f in dest_active}
    dest_archive_texts = {f.get("text", "") or "" for f in dest_archive}
    migrate_active: list[int] = []
    migrate_archive: list[int] = []
    for pos in survivors:
        text = rows[pos].get("text", "") or ""
        if text in dest_active_texts:
            drop.setdefault("already-in-destination", []).append(pos)
        elif text in dest_archive_texts:
            drop.setdefault("already-in-destination-archive", []).append(pos)
        elif items[pos][0] == "active":
            migrate_active.append(pos)
        else:
            migrate_archive.append(pos)

    # 4. Flags — a reading list over the rows that ARE moving. Nothing here
    #    removes anything; a false positive costs one line of reading.
    dest_norms = {
        _tidy_norm(f.get("text", "") or "")
        for f in list(dest_active) + list(dest_archive)
    }
    flags: dict[str, list[int]] = {}
    moving = migrate_active + migrate_archive
    for pos in moving:
        text = rows[pos].get("text", "") or ""
        if _tidy_norm(text) in dest_norms:
            flags.setdefault("near-duplicate-in-destination", []).append(pos)
            continue
        rule = _tidy_flag_rule(text)
        if rule:
            flags.setdefault(rule, []).append(pos)
    by_norm: dict[str, list[int]] = {}
    for pos in moving:
        by_norm.setdefault(_tidy_norm(rows[pos].get("text", "") or ""), []).append(pos)
    near = sorted(p for g in by_norm.values() if len(g) > 1 for p in g)
    if near:
        flags.setdefault("near-duplicate", []).extend(near)

    # The confirmation code covers BOTH decisions, not just the destructive
    # one. D6's code hashes only the removal set because everything else stays
    # put; here a change to what MOVES is just as much a change to what the
    # operator approved, and the two conversation ids are in it because the
    # same plan pointed at a different destination is a different operation.
    parts = [f"source\x01{source_id}", f"dest\x01{dest_id}"]
    parts += sorted(
        f"migrate\x01{items[p][0]}\x01{rows[p].get('text', '') or ''}"
        for p in moving
    )
    parts += sorted(
        f"drop\x01{rule}\x01{rows[p].get('text', '') or ''}"
        for rule, ps in drop.items()
        for p in ps
    )
    token = hashlib.sha256(
        "\x00".join(parts).encode("utf-8", "replace")
    ).hexdigest()[:12]

    return {
        "items": items,
        "rows": rows,
        "n_active": sum(1 for layer, _ in items if layer == "active"),
        "n_archive": sum(1 for layer, _ in items if layer == "archive"),
        "drop_by_rule": drop,
        "drop_idx": sorted(p for ps in drop.values() for p in ps),
        "migrate_active": migrate_active,
        "migrate_archive": migrate_archive,
        "flag_by_rule": flags,
        "token": token,
    }


def _retire_other_layers(conv_id: str) -> dict:
    """Everything else keyed to this conv_id, read rather than assumed.

    The enumeration is selftest._conv_artifact_paths — kept in one place there
    "because the layers are spread across four modules and a cleanup that
    misses one is indistinguishable from a cleanup that worked" — plus the two
    that are not files: the ChromaDB episodic rows and dedup's in-process
    refusal memo.

    Full list for this conv_id, and what happens to each:

      facts/<id>.json           the active facts       — classified above
      facts/<id>.archive.json   the cold sidecar       — classified above
      summaries/<id>.json       L1/L2/L3 rollups       — in the snapshot, then deleted
      personas/<id>.json        the persona            — in the snapshot, then deleted
      facts/<id>.backfill.json  lazy-backfill state    — deleted (state, not memory)
      ChromaDB where conv_id=   indexed exchanges      — in the snapshot, then deleted
      dedup._REFUSAL_MEMO[id]   merge refusals         — dropped (a cache, process-local)

    The backfill sidecar path is built here rather than imported from
    backfill.py for the reason selftest gives: importing that module for a path
    would pull the lazy-backfill machinery in for no reason.

    Values are None for "could not tell", never 0/False — the same distinction
    /forget's verification pass and retrieval.conversation_doc_count draw, and
    for the same reason: on a surface that decides whether a wipe was complete,
    those are opposite answers.
    """
    out: dict[str, Any] = {
        "summary": None, "episodic": None, "persona": None, "backfill": None,
    }
    try:
        import summarizer as summarizer_module
        state = summarizer_module.load_state(conv_id) or {}
        out["summary"] = bool(state.get("l1") or state.get("l2") or state.get("l3"))
    except Exception as e:
        logger.warning(f"conv={conv_id}: /retire could not read the summary layer: {e}")
    try:
        import retrieval as retrieval_module
        out["episodic"] = retrieval_module.conversation_doc_count(conv_id)
    except Exception as e:
        logger.warning(f"conv={conv_id}: /retire could not read the episodic layer: {e}")
    try:
        import persona as persona_module
        out["persona"] = bool(persona_module.load_persona(conv_id))
    except Exception as e:
        logger.warning(f"conv={conv_id}: /retire could not read the persona layer: {e}")
    try:
        out["backfill"] = (
            storage_root() / "facts" / f"{conv_id}.backfill.json"
        ).is_file()
    except Exception as e:
        logger.warning(f"conv={conv_id}: /retire could not stat the backfill sidecar: {e}")
    return out


def _retire_clear_other_layers(conv_id: str) -> list[str]:
    """Delete every non-facts layer keyed to conv_id. Returns what it cleared.

    Best-effort per layer and idempotent per layer: each of these is a delete
    of something that may already be gone, so a second run is a no-op rather
    than an error. Failures are logged and reported, never swallowed — the
    caller prints an observed re-read afterwards, so a layer that refused to go
    shows up in the reply whatever this returns.
    """
    cleared: list[str] = []
    try:
        import summarizer as summarizer_module
        sp = summarizer_module.summary_path(conv_id)
        if sp.is_file():
            sp.unlink()
            cleared.append("summary state")
    except Exception as e:
        logger.warning(f"conv={conv_id}: /retire summary delete failed: {e}")
    try:
        import retrieval as retrieval_module
        n = retrieval_module.forget_conversation(conv_id)
        if n:
            cleared.append(f"{n} indexed exchange(s)")
    except Exception as e:
        logger.warning(f"conv={conv_id}: /retire episodic delete failed: {e}")
    try:
        import persona as persona_module
        if persona_module.clear_persona(conv_id):
            cleared.append("persona")
    except Exception as e:
        logger.warning(f"conv={conv_id}: /retire persona delete failed: {e}")
    try:
        bp = storage_root() / "facts" / f"{conv_id}.backfill.json"
        if bp.is_file():
            bp.unlink()
            cleared.append("lazy-backfill state")
    except Exception as e:
        logger.warning(f"conv={conv_id}: /retire backfill-state delete failed: {e}")
    try:
        import dedup as dedup_module
        dedup_module.reset_refusal_memo(conv_id)
    except Exception as e:
        logger.warning(f"conv={conv_id}: /retire memo reset failed: {e}")
    return cleared


def _retire_layer_lines(label: str, layers: dict) -> list[str]:
    def say(v, yes: str, no: str) -> str:
        if v is None:
            return "could not read it"
        if isinstance(v, int) and not isinstance(v, bool):
            return f"{v} {yes}" if v else no
        return yes if v else no

    return [
        f"{label}",
        f"  summary state         {say(layers['summary'], 'present', 'none')}",
        f"  indexed exchanges     {say(layers['episodic'], 'exchange(s)', 'none')}",
        f"  persona               {say(layers['persona'], 'present', 'none')}",
        f"  lazy-backfill state   {say(layers['backfill'], 'present', 'none')}",
    ]


def _retire_show(plan: dict) -> Callable[[int], str]:
    """Row renderer for _tidy_group_lines: prefixes the source layer, because
    'this came out of cold storage' changes what a reviewer thinks of it."""
    def render(pos: int) -> str:
        layer, f = plan["items"][pos]
        tag = "active " if layer == "active" else "archived"
        return f"({tag}) {_tidy_show(f.get('text', '') or '')}"
    return render


def _retire_render_plan(
    plan: dict, layers: dict, source_id: str, dest_id: str, dest_after: dict
) -> str:
    n_move = len(plan["migrate_active"]) + len(plan["migrate_archive"])
    n_drop = len(plan["drop_idx"])
    total = plan["n_active"] + plan["n_archive"]
    show = _retire_show(plan)

    lines = [
        f"Retire conversation {source_id} — DRY RUN. Nothing has been changed.",
        "",
        f"Source:      {source_id} — {plan['n_active']} active fact(s), "
        f"{plan['n_archive']} archived fact(s)",
        f"Destination: {dest_id} — this conversation, "
        f"{dest_after['before_active']} active fact(s), "
        f"{dest_after['before_archive']} archived fact(s)",
        "",
    ]
    lines.extend(_retire_layer_lines(
        f"Also keyed to {source_id}, and cleared by this operation:", layers))
    lines.append("")

    if n_move:
        lines.append(
            f"WOULD MOVE {n_move} of {total} fact(s) into this conversation "
            f"({len(plan['migrate_active'])} into the active set, "
            f"{len(plan['migrate_archive'])} into cold storage because that is "
            f"where they already were):"
        )
        for label, idx in (
            ("moving into the active set", plan["migrate_active"]),
            ("moving into cold storage", plan["migrate_archive"]),
        ):
            if not idx:
                continue
            lines.append(f"  [{label}] {len(idx)} row(s)")
            for pos in idx[:RETIRE_MAX_MIGRATE_ROWS_SHOWN]:
                lines.append(f"      {show(pos)}")
            if len(idx) > RETIRE_MAX_MIGRATE_ROWS_SHOWN:
                lines.append(
                    f"      … and {len(idx) - RETIRE_MAX_MIGRATE_ROWS_SHOWN} "
                    f"more not shown (nothing in this list is being deleted)"
                )
    else:
        lines.append("WOULD MOVE nothing — no fact here needs a new home.")
    lines.append("")

    if n_drop:
        pct = (100 * n_drop) // total if total else 0
        lines.append(
            f"WOULD NOT MOVE {n_drop} of {total} ({pct}%). Every one is listed "
            f"here in full, and every one is in the snapshot below:"
        )
        lines.extend(_tidy_group_lines(
            plan["rows"], plan["drop_by_rule"], _RETIRE_DROP_ORDER,
            limit=None, rule_text=_RETIRE_RULE_TEXT, show=show,
        ))
    else:
        lines.append("WOULD NOT MOVE nothing — every fact here is being kept.")
    lines.append("")

    n_flag = sum(len(v) for v in plan["flag_by_rule"].values())
    if n_flag:
        lines.append(
            f"MOVING ANYWAY, but worth your eyes — {n_flag} row(s) look odd "
            f"and are NOT provably garbage, so they are being kept:"
        )
        lines.extend(_tidy_group_lines(
            plan["rows"], plan["flag_by_rule"], _RETIRE_FLAG_ORDER,
            limit=TIDY_MAX_ROWS_SHOWN, rule_text=_RETIRE_RULE_TEXT, show=show,
        ))
        lines.append("")

    lines.append("What happens to the metadata on a moved fact:")
    lines.append(
        "  last_used  kept exactly. It is unix seconds with one writer "
        "(facts.py:126-128, \"Safe to compare across facts\"), so it means the "
        "same thing here as it did there and it is what eviction sorts on."
    )
    lines.append(
        "  added_turn kept exactly, and it is NOT meaningful here. facts.py:"
        "130-132 says do not compare two facts' added_turn unless they came "
        "from the same writer, and these came from another conversation's. It "
        "survives as an injection-order hint only. Rewriting it would be "
        "inventing a number; portability.import_conversation carries it "
        "verbatim across conversations for the same reason."
    )
    lines.append(
        "  archived_at is RE-STAMPED to now on the rows that land in cold "
        "storage, because facts.archive_facts sets it and it means \"when this "
        "row entered this sidecar\" — which, after the move, is now. The "
        "original value is in the snapshot."
    )
    lines.append(
        "  provenance is NOT written onto the row. facts.load_facts rebuilds "
        "every entry as exactly {text, added_turn, last_used, pin}, so any extra "
        "key is dropped on the next read. Where each fact came from is in the "
        "snapshot and the log instead."
    )
    lines.append("")

    if dest_after["over_budget"]:
        lines.append(
            f"Heads up: this conversation would hold "
            f"{dest_after['after_active']} active fact(s), and "
            f"facts.select_for_injection would fit "
            f"{dest_after['after_injectable']} of them into the facts budget. "
            f"The other {dest_after['after_active'] - dest_after['after_injectable']} "
            f"are not lost — the next prune moves the least-recently-used of "
            f"them into this conversation's archive, where /list-archive shows "
            f"them and they can be restored."
        )
        lines.append("")

    lines.append(
        f"Before anything moves or is deleted I write one verified snapshot of "
        f"{source_id} — every layer above, including the rows in the "
        f"WOULD NOT MOVE list — and I only proceed if it reads back complete. "
        f"That file is the complete undo and nothing ever deletes it "
        f"automatically."
    )
    lines.append("")
    lines.append(f"To do exactly this:   /retire {source_id} apply {plan['token']}")
    lines.append(
        "That code covers this exact source, this exact destination, and every "
        "row listed above on both sides. If any of it changes before you "
        "confirm, the code stops working and you get a fresh plan instead of a "
        "surprise."
    )
    return "\n".join(lines)


def _retire_dest_projection(
    dest_active: list[dict], dest_archive: list[dict], plan: dict
) -> dict:
    """What the destination looks like before and after, using only public
    fact-store API. select_for_injection is the authority on what actually
    reaches a prompt, so the answer comes from it rather than from a token
    estimate this file would have to keep in step with facts.py."""
    incoming = [plan["rows"][p] for p in plan["migrate_active"]]
    after = list(dest_active) + incoming
    # The STORE cap, explicitly. select_for_injection's own default became
    # the INJECTION cap (400) when F1 decoupled the two, so a bare call here
    # started reporting "over budget" for virtually every real store and
    # attaching an explanation that is no longer true - facts between the
    # injection cap and the store cap stay active and are simply not all
    # injected on a given turn. What this preview is actually about is what
    # survives the MOVE, so it asks the question it means.
    injectable = len(
        facts_module.select_for_injection(after, max_tokens=facts_module._MAX_FACTS_TOKENS)
    )
    return {
        "before_active": len(dest_active),
        "before_archive": len(dest_archive),
        "after_active": len(after),
        "after_injectable": injectable,
        "over_budget": injectable < len(after),
    }


_RETIRE_USAGE = (
    "Usage:\n"
    "  /retire <conv_id>              show how I would empty that "
    "conversation's memory into THIS one (changes nothing)\n"
    "  /retire <conv_id> apply <code> do exactly what that dry run listed\n"
    "\n"
    "The conversation you name is the one that gets emptied. The destination "
    "is always the conversation you are typing in, so it cannot be mistyped."
)


async def _handle_retire(arg: str, conv_id: str, ctx: dict) -> str:
    parts = arg.split()
    if not parts:
        return _RETIRE_USAGE
    source_id = parts[0]
    if not _RETIRE_CONV_ID_RE.match(source_id):
        return (
            f"{source_id!r} is not a conversation id. Ids are 1-64 characters "
            f"of letters, digits, dash and underscore. I have changed "
            f"nothing.\n\n" + _RETIRE_USAGE
        )
    if source_id == conv_id:
        return (
            "That is this conversation. /retire moves one conversation's "
            "memory into another one, so it cannot be its own destination — "
            "for emptying the conversation you are in, /forget is the command, "
            "and it does not move anything anywhere. Nothing has been changed."
        )

    mode = parts[1].lower() if len(parts) > 1 else ""
    if mode not in ("", "apply"):
        return (
            f"Unknown option {parts[1]!r}. I have changed nothing.\n\n"
            + _RETIRE_USAGE
        )

    if mode == "":
        source_active = facts_module.load_facts(source_id)
        source_archive = facts_module.load_archive(source_id)
        layers = _retire_other_layers(source_id)
        if not source_active and not source_archive and not any(
            bool(layers[k]) for k in ("summary", "episodic", "persona", "backfill")
        ):
            return (
                f"Conversation {source_id} has no stored memory of any kind — "
                f"no facts, no archive, no summary, no indexed exchanges and no "
                f"persona. There is nothing to retire."
            )
        items = _retire_items(source_active, source_archive)
        dest_active = facts_module.load_facts(conv_id)
        dest_archive = facts_module.load_archive(conv_id)
        plan = _retire_plan(
            items, dest_active, dest_archive,
            source_id=source_id, dest_id=conv_id,
        )
        return _retire_render_plan(
            plan, layers, source_id, conv_id,
            _retire_dest_projection(dest_active, dest_archive, plan),
        )

    token = parts[2].lower() if len(parts) > 2 else ""
    if not token:
        return (
            f"/retire ... apply needs the code from a dry run. Run "
            f"/retire {source_id} first and read what it proposes — the code "
            f"exists so that nothing moves and nothing is dropped that you "
            f"have not seen."
        )

    # Both conversations are live stores with their own extraction tails, so
    # both writes have to be serialized against them — the destination just as
    # much as the source, because appending to a facts file a parked tail is
    # about to rewrite from its own pre-read snapshot is the F22/F3 shape
    # exactly, and here it would lose the migrated facts outright.
    #
    # Two locks means an ordering rule. Sorted by id, unconditionally: nothing
    # else in this codebase takes two conv_locks, so self-consistency is the
    # whole requirement, and sorted() gives it without a registry. source and
    # conv_id are known distinct by the guard above, so this never
    # self-deadlocks on re-entry.
    first, second = sorted((source_id, conv_id))
    async with conv_lock(first):
        async with conv_lock(second):
            source_active = facts_module.load_facts(source_id)
            source_archive = facts_module.load_archive(source_id)
            layers = _retire_other_layers(source_id)
            dest_active = facts_module.load_facts(conv_id)
            dest_archive = facts_module.load_archive(conv_id)

            has_anything = bool(source_active) or bool(source_archive) or any(
                bool(layers[k])
                for k in ("summary", "episodic", "persona", "backfill")
            )
            if not has_anything:
                # Also the clean answer to "the apply already ran and you sent
                # it twice", which is why it is worded as a state and not as an
                # error.
                return (
                    f"Conversation {source_id} has no stored memory of any "
                    f"kind. There is nothing to retire, and nothing has been "
                    f"changed."
                )

            items = _retire_items(source_active, source_archive)
            # Re-planned from disk, never carried over from the dry run: the
            # plan is a pure function of the two stores, so this is the compare
            # half of a compare-and-swap over BOTH of them.
            plan = _retire_plan(
                items, dest_active, dest_archive,
                source_id=source_id, dest_id=conv_id,
            )
            if plan["token"] != token:
                return (
                    "That code is out of date — the plan has changed since the "
                    "dry run, so I have moved nothing and dropped nothing.\n\n"
                    + _retire_render_plan(
                        plan, layers, source_id, conv_id,
                        _retire_dest_projection(dest_active, dest_archive, plan),
                    )
                )

            # 1. ARCHIVE EVERYTHING FIRST. Raises rather than returning a bad
            #    path, and a raise here means nothing below runs.
            try:
                snap = portability.quarantine_conversation(
                    source_id, reason=f"retire-into:{conv_id}"
                )
            except portability.QuarantineError as e:
                logger.error(
                    f"conv={source_id}: /retire aborted — could not write a "
                    f"verified snapshot: {e}"
                )
                return (
                    f"I could not write a verified backup of {source_id} "
                    f"first, so I have changed nothing. Nothing has been moved "
                    f"and nothing has been removed. This is a problem on my "
                    f"side — please mention it."
                )

            # A layer the snapshot could not read is a layer this operation is
            # about to delete without a copy of it. /tidy can note an
            # unverified layer and carry on because it only touches facts;
            # retirement deletes every one of them, so an unverified layer is
            # a refusal.
            if snap["unverified_layers"]:
                logger.error(
                    f"conv={source_id}: /retire refused — snapshot could not "
                    f"verify: {'; '.join(snap['unverified_layers'])}"
                )
                return (
                    f"I have changed nothing. My backup of {source_id} could "
                    f"not account for: "
                    + "; ".join(snap["unverified_layers"])
                    + f".\n\nRetiring a conversation deletes every one of its "
                    f"layers, so I will not do it while I cannot prove I have "
                    f"a copy of one of them. The snapshot I did write is at "
                    f"{snap['path']} and nothing was removed. This is a "
                    f"problem on my side — please mention it."
                )

            # 2. MIGRATE, and only then remove. Cold storage first, on
            #    archive_facts' own ordering rule; both are additions, so an
            #    interruption here leaves rows in two places — which the next
            #    run resolves as "already in the destination" — and loses
            #    nothing, because the source has not been touched yet.
            move_archive = [plan["rows"][p] for p in plan["migrate_archive"]]
            move_active = [plan["rows"][p] for p in plan["migrate_active"]]
            n_arch = facts_module.archive_facts(conv_id, move_archive)
            if move_active:
                facts_module.save_facts(conv_id, dest_active + move_active)

            # 3. Now empty the source. Sidecar, then the active set, then the
            #    layers that are not facts.
            facts_module.save_archive(source_id, [])
            # An EMPTY facts file, not an unlinked one — /forget's tombstone,
            # for its reason: backfill.needs_backfill gates on
            # `facts_path(conv_id).is_file()`, so with no file there the next
            # request on a conversation of four messages or more starts a
            # background extraction over its whole history and writes the
            # result straight back. The documented cost is that
            # memory.list_known_conv_ids globs facts/*.json, so this conv_id
            # keeps appearing in /admin/conversations. It is the right trade
            # both times: an empty file that records a decision, against a
            # store that rebuilds itself.
            facts_module.save_facts(source_id, [])
            cleared = _retire_clear_other_layers(source_id)

            # Report what is on disk, not what the operation intended — the
            # rule /forget's verification pass and /tidy both follow.
            after_source_active = len(facts_module.load_facts(source_id))
            after_source_archive = len(facts_module.load_archive(source_id))
            after_layers = _retire_other_layers(source_id)
            after_dest_active = len(facts_module.load_facts(conv_id))
            after_dest_archive = len(facts_module.load_archive(conv_id))

    n_move = len(move_active) + len(move_archive)
    n_drop = len(plan["drop_idx"])
    # Counts only. Fact text is real personal memory: it goes to the chat reply
    # the owner asked for and nowhere else — not to a log file, not to an
    # operator's terminal.
    logger.info(
        f"conv={source_id}: /retire into {conv_id} moved {n_move} fact(s) "
        f"({len(move_active)} active, {n_arch} archived), dropped {n_drop} "
        f"({', '.join(f'{r}={len(v)}' for r, v in sorted(plan['drop_by_rule'].items())) or 'none'}), "
        f"cleared [{', '.join(cleared) or 'no other layer'}]; "
        f"snapshot={snap['path'].name}"
    )

    lines = [
        f"Retired {source_id}.",
        "",
        f"Moved {n_move} fact(s) into this conversation "
        f"({len(move_active)} active, {n_arch} into cold storage). "
        f"This conversation now has {after_dest_active} active fact(s) and "
        f"{after_dest_archive} archived.",
    ]
    if n_drop:
        lines.append(
            f"Did not move {n_drop} row(s) — every one of them is in the "
            f"snapshot below, and the dry run listed them all."
        )
    else:
        lines.append("Every fact in that conversation was worth moving.")
    if cleared:
        lines.append(f"Cleared from {source_id}: {', '.join(cleared)}.")
    lines.append(
        f"A complete snapshot of {source_id} as it was a moment ago is at "
        f"{snap['path']} ({snap['facts']} active fact(s), {snap['archive']} "
        f"archived, {snap['episodic']} indexed exchange(s), summary="
        f"{'yes' if snap['summary'] else 'no'}, persona="
        f"{'yes' if snap['persona'] else 'no'}). Nothing deletes it "
        f"automatically."
    )

    residue: list[str] = []
    if after_source_active:
        residue.append(f"{after_source_active} active fact(s)")
    if after_source_archive:
        residue.append(f"{after_source_archive} archived fact(s)")
    for name, key in (
        ("summary state", "summary"), ("indexed exchanges", "episodic"),
        ("persona", "persona"), ("lazy-backfill state", "backfill"),
    ):
        v = after_layers[key]
        if v is None:
            residue.append(f"{name} (could not re-read it)")
        elif v:
            residue.append(name)
    if residue:
        lines.append("")
        lines.append(
            "Not everything went: " + ", ".join(residue) + " is still there. "
            "Nothing has been lost — run this again and it will finish."
        )
    lines.append("")
    lines.append(
        f"One thing this does NOT do: if {source_id} was a bucket that "
        f"background traffic keeps hashing to, it will start filling again on "
        f"the next such request. Nothing in the compactor stops that yet."
    )
    return "\n".join(lines)


_HANDLERS: dict[str, Handler] = {
    "help": _handle_help,
    "list-facts": _handle_list_facts,
    "list-archive": _handle_list_archive,
    "remember": _handle_remember,
    "pin": _handle_pin,
    "unpin": _handle_unpin,
    "forget": _handle_forget,
    "why": _handle_why,
    "tidy": _handle_tidy,
    "retire": _handle_retire,
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
