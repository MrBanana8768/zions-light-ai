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
    # v3.1 D6. No single-letter or near-miss alias: /tidy is the only command
    # here that can remove a fact without naming it, and it should take a
    # deliberate keystroke to reach.
    "tidy": "tidy",
    "tidy-facts": "tidy",
    "tidy_facts": "tidy",
    "cleanup": "tidy",
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
        "  /tidy                Show extraction debris I could clean up "
        "(changes nothing)\n"
        "  /tidy apply <code>   Clean up exactly what that dry run listed\n"
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
) -> list[str]:
    """Render one section, grouped by rule.

    `limit=None` for the removal section, always. A row the operator is asked
    to confirm and was not shown is the whole failure this command is designed
    against, so the removal list is never elided — the blast-radius cap is what
    bounds its length, not the renderer. Flagged rows are only a reading list,
    so those groups do get capped.
    """
    lines: list[str] = []
    seen = list(order) + [r for r in by_rule if r not in order]
    for rule in seen:
        idx = by_rule.get(rule)
        if not idx:
            continue
        why = _TIDY_RULE_TEXT.get(
            rule, "NO DESCRIPTION FOR THIS RULE — do not confirm this plan"
        )
        lines.append(f"  [{rule}] {len(idx)} row(s) — {why}")
        shown = idx if limit is None else idx[:limit]
        for i in shown:
            lines.append(f"      {_tidy_show(rows[i].get('text', '') or '')}")
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


_HANDLERS: dict[str, Handler] = {
    "help": _handle_help,
    "list-facts": _handle_list_facts,
    "list-archive": _handle_list_archive,
    "remember": _handle_remember,
    "forget": _handle_forget,
    "why": _handle_why,
    "tidy": _handle_tidy,
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
