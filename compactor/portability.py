"""
compactor.portability — V2.1 Phase 6 Step 3: conversation export / import / fork.

Single-conversation JSON bundles that capture every layer of V2.0
memory state in one transportable blob:

  - facts          (Phase 2)
  - summary state  (Phase 4, L1/L2/L3)
  - episodic       (Phase 3, indexed exchanges from ChromaDB)

Use cases:
  - Disaster recovery: back up a critical conversation before a
    suspect operation (forget, rollback, model swap)
  - Cross-pod migration: move a long conversation off a pod that's
    being torn down to a new pod, preserving all model context
  - Forking: explore an alternative direction for a story without
    losing the original path

Embeddings are NOT in the bundle — re-embedded on import. Keeps
bundles tiny (text only) and portable across embedding-model swaps.

Bundle schema (v2.1):
    {
        "version":     "v2.1",
        "exported_at": <unix_ts>,
        "source_conv_id": <str>,
        "facts":          [<fact dict>, ...],
        "summary_state":  {<summarizer state>},
        "episodic":       [{"turn_index": int, "document": str}, ...],
    }

A future version bump may add: message history, persona pointer,
metadata. The version field lets import detect unknown schemas
without silently truncating.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

import anyio

import facts
import memory
import persona
import retrieval
import summarizer

logger = logging.getLogger("compactor.portability")

BUNDLE_VERSION = "v2.1"


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_conversation(conv_id: str, *, strict: bool = False) -> dict:
    """Snapshot one conv's full V2 state as a single JSON-serializable dict.

    Best-effort per layer BY DEFAULT: a failure in one layer doesn't poison
    the bundle — it just gets an empty value. The bundle always has every
    expected key so import logic doesn't need defensive .get() calls. This
    is the right contract for GET /admin/.../export (an operator inspecting
    or backing up a conv wants whatever CAN be read, not a hard failure).

    v3.1.9 (hostile pass 3, F9). `strict=True` is for callers that DECIDE
    something based on the fact/summary counts this returns: merge_
    conversation reports `src_facts` and folds them into dst, and
    fork_conversation writes them into a brand-new conv_id via
    import_conversation. Both used to inherit the best-effort default, so a
    SOURCE whose facts file held a torn write (StoreUnreadable) merged or
    forked as "0 facts" — silently, with no error, exactly the "unknown read
    as empty" hazard this codebase's own doctrine rejects everywhere else
    (import's pre-flight, health's `unreadable` counts, quarantine's
    `unverified_layers`). In strict mode, `facts.load_facts` and
    `summarizer.load_state` raising `memory.StoreUnreadable` PROPAGATES
    rather than being caught — callers map that to a 400, the same shape
    import_conversation already uses for "cannot verify, refusing". Episodic
    is unchanged either way: `retrieval.export_indexed_exchanges` is
    contractually never-raising (its own docstring), a retrieval.py contract
    this module does not own or override.
    """
    try:
        loaded_facts = facts.load_facts(conv_id)
    except memory.StoreUnreadable:
        if strict:
            raise
        logger.warning(f"conv={conv_id}: export facts failed, StoreUnreadable")
        loaded_facts = []
    except Exception as e:
        logger.warning(f"conv={conv_id}: export facts failed: {e}")
        loaded_facts = []

    try:
        summary_state = summarizer.load_state(conv_id)
    except memory.StoreUnreadable:
        if strict:
            raise
        logger.warning(f"conv={conv_id}: export summary failed, StoreUnreadable")
        summary_state = {}
    except Exception as e:
        logger.warning(f"conv={conv_id}: export summary failed: {e}")
        summary_state = {}

    try:
        episodic = retrieval.export_indexed_exchanges(conv_id)
    except Exception as e:
        logger.warning(f"conv={conv_id}: export episodic failed: {e}")
        episodic = []

    return {
        "version": BUNDLE_VERSION,
        "exported_at": int(time.time()),
        "source_conv_id": conv_id,
        "facts": loaded_facts,
        "summary_state": summary_state,
        "episodic": episodic,
    }


# ---------------------------------------------------------------------------
# Quarantine — the archive half of a destructive admin operation
# ---------------------------------------------------------------------------
#
# v3.1 D6. Anything in this codebase that removes stored memory has to be
# reversible, and there are already two mechanisms for that. This adds no
# third one; it wires the two together and adds the part neither had.
#
#   1. facts.archive_facts / restore_from_archive — the per-row, cold-storage
#      sidecar. This is how a fact leaves the active set today (F9), it is
#      already visible to the user as /list-archive, and it is already
#      reversible without an operator. Any cleanup that removes facts should
#      go through it rather than writing a shorter list with save_facts.
#
#   2. export_conversation / import_conversation — the whole-conversation
#      bundle. Its docstring already names this exact use case: "back up a
#      critical conversation before a suspect operation (forget, rollback,
#      model swap)". So the "archive before removing" half is half-built: the
#      export produces the snapshot, and nothing writes it anywhere.
#
# What is missing is durability and verification. export_conversation is
# best-effort per layer — every read is wrapped in `except Exception` and
# degrades to an empty value — so a bundle from a conversation whose facts
# file is unreadable is a *valid, empty, importable* bundle. Handing that to
# an operator as "your data is safe, go ahead and delete" is precisely the
# shape backup.py F2 fixed: an archive of nothing that verified green,
# published, and pruned the real archives behind it.
#
# So quarantine_conversation borrows backup.py's staging/verify/publish:
# measure what the store holds BEFORE exporting, write to a `.partial`, read
# the file back off disk and contradict the manifest from it, and only then
# publish under the real name. A crash at any point leaves either a `.partial`
# nothing reads or a published file that has been proven readable. Never a
# half-trusted snapshot.
#
# v3.1 D8 — THREE LAYERS THE BUNDLE DOES NOT CARRY, and why they are carried
# here instead. (v3.1.9.4, P15-3: the chapter archive joined this list —
# see the note at the end of this comment.)
#
# export_conversation writes exactly three payloads: facts, summary_state,
# episodic. It does not carry the archive sidecar, it does not carry the
# persona, and it does not carry the L2 chapter cold store. For an export
# that is a schema question; for a PRE-REMOVAL SNAPSHOT it is a correctness
# one, because all three of those layers are things a destructive admin
# operation deletes:
#
#   - commands._wipe_all_layers clears the archive sidecar, and its own
#     comment records that "NOTHING in this codebase has ever deleted one"
#     before it did.
#   - main._clear_all_memory calls persona.clear_persona. INCIDENT_2026-08-24
#     D19 states the consequence in one line: "/forget can destroy a persona
#     that no export can back up and no import can restore."
#   - main._clear_all_memory also deletes the chapter cold store
#     (summaries/<id>.archive.json) since v3.1.9.4 (P15-3) — the same D19
#     consequence, one layer over: it used to survive every wipe silently
#     (not deleted, so not even the thing this comment is about), and now
#     that it IS deleted the same "no export can back it up" gap would
#     apply to it too if it stopped here.
#
# A snapshot that is missing the layers the operation removes is not an
# archive-before-removing; it is a partial one that reads as complete. So all
# three are measured, carried, and verified on read-back like everything else.
#
# They go in the `quarantine` metadata block rather than into the bundle
# payload. That is deliberate and it is not laziness: adding payload keys means
# either a BUNDLE_VERSION bump — which _validate_bundle enforces by strict
# equality, so every previously written bundle and both HTTP endpoints stop
# working the moment it changes — or silently widening a documented schema.
# Under `quarantine`, _validate_bundle ignores the extra key, so a snapshot
# stays a valid v2.1 bundle that import_conversation restores with no new code,
# AND the three extra layers travel with it for an operator (or
# commands._handle_retire) to put back explicitly. Restoring them is a
# `facts.save_archive`, a `persona.save_persona`, and a
# `summarizer._archive_chapters`; the restore_hint says so.
#
# This does NOT close D19 for export/import generally — export_conversation is
# unchanged and a fork still loses the persona and the chapter archive. It
# closes it for the one path whose entire purpose is to make a removal
# reversible.
#
# WHY EXPORT ITSELF STILL DOES NOT CARRY THE CHAPTER ARCHIVE (P15-3 asks this
# question explicitly, so the answer is here rather than left implicit): the
# same BUNDLE_VERSION argument two paragraphs up applies without change — an
# ordinary GET /admin/.../export is not a removal, so there is nothing for it
# to be reversible AGAINST, and the chapter archive is the ONE copy of
# chapter-level detail an L3 refresh has already paraphrased, which is a
# reason to protect it on every REMOVAL path (done, above) but not a reason
# to widen a schema every existing bundle and both HTTP endpoints depend on
# staying fixed. A caller that wants the chapter archive alongside an export
# reads it separately: summarizer.load_chapter_archive(conv_id) — there is
# no dedicated HTTP endpoint for it today (GET /admin/conversations/<id>/
# summary returns summary_state only, not the cold chapters); that gap is
# real but is a separate, additive feature request, not something P15-3
# asked this fix to close.

# THIS COMMENT WAS FALSE, and it was the root of an arbitrary-file-write.
#
# It claimed conv_id arrives "already sanitized by memory._sanitize to
# [A-Za-z0-9_-]". _sanitize is called in exactly one place —
# resolve_conv_id, on the CHAT path. import_conversation and
# fork_conversation take their target id from the request BODY, which
# never passes through it, so a traversal went straight to the filesystem:
# target_conv_id "../../../../../../tmp/PWNED" returned HTTP 200 and wrote
# outside STORAGE_ROOT. Worse without any attacker: ".." escapes the
# per-layer subdir, so an import aimed at "../facts/<other-conv>" reported
# success while emptying a bystander's memory.
#
# Since v3.1.8 the guarantee is real but it comes from memory._safe_path,
# which refuses any path resolving outside STORAGE_ROOT, at the five path
# builders every caller goes through. Left as a comment rather than
# deleted: a confident assertion about someone else's invariant, written
# where it cannot be checked, is the thing to be suspicious of.
QUARANTINE_SUBDIR = "quarantine"


def _has_summary_content(state: Any) -> bool:
    """True when a summary_state payload holds an actual hierarchy.

    NOT `bool(state)`. summarizer.load_state "returns an empty (but
    well-formed) skeleton if no file exists" — a dict with l1/l2/l3 and
    last_summarized_turn keys — so a plain truthiness test on it is True for
    every conversation that has never been summarized at all. The quarantine
    log said `summary=yes` unconditionally, which on a snapshot surface is
    the same class of lie as reporting a wipe from the counters instead of
    from disk: the operator reads a layer that is not there. Same test
    commands._memory_residue applies for /forget's verification pass.
    """
    if not isinstance(state, dict):
        return False
    return bool(state.get("l1") or state.get("l2") or state.get("l3"))


class QuarantineError(Exception):
    """The pre-removal snapshot could not be written or could not be proven
    complete. Callers MUST abort the removal — this exception is the only
    thing standing between "reversible" and "gone"."""


def quarantine_dir() -> Path:
    """Where pre-removal snapshots live.

    Under the compactor storage root, so backup.py's `copytree(STORAGE_ROOT)`
    picks them up for free and a snapshot survives the volume it describes.
    Not under facts/, because memory.list_known_conv_ids and backup._census
    both glob that directory and a quarantine file is not a conversation.
    """
    return memory.storage_root() / QUARANTINE_SUBDIR


def _quarantine_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _quarantine_path(conv_id: str) -> Path:
    """A published name that is not already taken.

    Second-resolution stamps collide if the operation is run twice inside one
    second, and the loser of that collision would be a snapshot silently
    overwritten by the very operation it exists to make reversible.
    """
    d = quarantine_dir()
    stamp = _quarantine_stamp()
    candidate = d / f"{conv_id}.{stamp}.json"
    n = 1
    while candidate.exists():
        candidate = d / f"{conv_id}.{stamp}-{n}.json"
        n += 1
    return candidate


def list_quarantine(conv_id: str | None = None) -> list[Path]:
    """Published quarantine snapshots, newest filename last. `.partial` files
    are never returned — an interrupted write must not look like a snapshot.
    """
    d = quarantine_dir()
    if not d.is_dir():
        return []
    pattern = f"{conv_id}.*.json" if conv_id else "*.json"
    return sorted(p for p in d.glob(pattern) if not p.name.endswith(".partial"))


def quarantine_conversation(conv_id: str, *, reason: str) -> dict:
    """Write a verified, restorable snapshot of this conversation before
    something removes part of it. Returns {"path", "facts", "archive",
    "episodic", "summary", "persona", "chapters", "unverified_layers"}.

    Raises QuarantineError if the snapshot cannot be proven to hold at least
    what the store held a moment ago, OR if the facts file is unreadable and
    its raw bytes cannot be copied aside either (see F1 below) — a snapshot
    that cannot be written must abort the caller's removal/overwrite rather
    than proceed on a guess.

    The produced file is a plain export bundle plus a `quarantine` metadata
    block, so `import_conversation(json.load(open(path)),
    target_conv_id=..., overwrite=True)` restores it with no new code and no
    new format. _validate_bundle checks the version and the three payload
    keys and ignores extra ones, which is what makes that work.

    Restoring the whole bundle is the BACKSTOP, not the first move: it rolls
    the conversation back wholesale and would discard anything learned since.
    For a facts cleanup the first move is restore_from_archive, which puts
    individual rows back without touching anything else.

    Retention: nothing here deletes old snapshots. They are written only by an
    explicit operator action, they are small (text only, no embeddings), and
    this module is not going to invent an automatic delete for the one
    directory whose entire job is to survive one.

    v3.1.9 (hostile pass 4, F1). This function USED TO let StoreUnreadable
    from the very first line (measuring expected_facts) propagate straight
    past everything below — the archive, the persona, the episodic count,
    the summary state, the export, the publish. A torn FACTS file made the
    caller (admin_import_conversation's overwrite path) read "the store is
    unreadable" and skip the ENTIRE snapshot, including the summary
    hierarchy and episodic index the torn facts file does not touch and
    which were otherwise perfectly readable. The overwrite that followed
    then destroyed all of it, plus the torn file's own bytes — the one
    surviving copy of what the facts had been, gone with nothing kept
    aside. Caught here instead: an unreadable facts file marks that ONE
    layer unverified and copies its raw bytes into the quarantine (a
    sibling `<snapshot>.facts.torn` file, path returned as
    `torn_facts_path`, verified by read-back like everything else this
    function publishes) rather than aborting before anything else is even
    measured.
    """
    unverified: list[str] = []

    # v3.1.9 (F1). Measured BEFORE the export, and strictly: this is the
    # expectation the verify step tries to contradict, so it cannot come
    # from the same best-effort reads it is checking. A facts file that IS
    # unreadable no longer propagates out of this function — see the
    # docstring. `expected_facts` becomes 0 (nothing readable to count),
    # the layer is marked unverified, and the torn file's own raw bytes are
    # captured here for the staging/publish block below (search
    # "torn_partial") to carry into the quarantine. export_conversation()
    # further down independently hits the same StoreUnreadable on the same
    # file and, in its own best-effort
    # (non-strict) mode, degrades to `facts: []` — consistent with
    # expected_facts=0, so Contradiction #1 does not fire on a mismatch this
    # function itself created.
    expected_facts = 0
    _torn_facts_raw: bytes | None = None
    try:
        expected_facts = len(facts.load_facts(conv_id))
    except memory.StoreUnreadable as e:
        unverified.append("facts (unreadable)")
        logger.warning(
            f"conv={conv_id}: quarantine could not read the facts file "
            f"({e}); the OTHER layers are still measured and snapshotted, "
            f"and the torn file's raw bytes are copied aside rather than "
            f"treating the whole store as unreadable"
        )
        try:
            _torn_facts_raw = e.path.read_bytes()
        except OSError as read_err:
            # Nothing to fall back to: the one thing this branch exists to
            # preserve (the torn file's own bytes, the best evidence of
            # what the facts were) could not even be copied. Abort rather
            # than publish a snapshot that silently drops it — same
            # "refuse rather than guess" rule as every other QuarantineError
            # below.
            raise QuarantineError(
                f"conv={conv_id}: the facts file is unreadable ({e}) AND its "
                f"raw bytes could not be copied aside either "
                f"({type(read_err).__name__}: {read_err}) — refusing to "
                f"publish a snapshot that would lose the one copy of "
                f"evidence for what the facts were"
            ) from read_err

    # The two layers export_conversation does not carry — see the block
    # comment above. Read as best-effort and RECORDED when they fail, never
    # silently defaulted: a caller that is about to clear the archive sidecar
    # has to be able to tell "there was nothing there" from "I could not look",
    # and those are the same value if this swallows the exception.
    archived_rows: list[dict] = []
    try:
        archived_rows = facts.load_archive(conv_id)
    except Exception as e:
        logger.warning(f"conv={conv_id}: quarantine could not read the archive sidecar: {e}")
        unverified.append("archived facts (unreadable)")
    persona_record = None
    try:
        persona_record = persona.load_persona(conv_id)
    except Exception as e:
        logger.warning(f"conv={conv_id}: quarantine could not read the persona: {e}")
        unverified.append("persona (unreadable)")
    # v3.1.9.4 (P15-3): a THIRD layer export_conversation does not carry,
    # same reason as the two above — summaries/<id>.archive.json
    # (summarizer._archive_chapters' only writer) is the ONLY copy of
    # chapter-level detail once an L3 refresh has paraphrased it, and until
    # this fix nothing removing a conversation's state ever preserved it:
    # /forget left it on disk unreported (fixed separately, see
    # main._clear_all_memory and commands._memory_residue), an overwrite
    # import left the TARGET's old chapters beside the new state (fixed
    # below, in import_conversation), and /retire orphaned it under the
    # retired id (fixed in commands._retire_clear_other_layers). This
    # function is the one place all three of those removals already funnel
    # through for a restorable copy, so it gets the same read-record-carry
    # treatment as archived_rows/persona_record above.
    chapter_rows: list[dict] = []
    try:
        chapter_rows = summarizer.load_chapter_archive(conv_id)
    except Exception as e:
        logger.warning(f"conv={conv_id}: quarantine could not read the chapter archive: {e}")
        unverified.append("chapter archive (unreadable)")

    expected_episodic = retrieval.conversation_doc_count(conv_id)
    if expected_episodic is None:
        # None is "could not tell", never zero (F61). The episodic layer is not
        # what a facts cleanup modifies, so this is recorded rather than fatal
        # — but it is recorded, because a snapshot with an unverified layer is
        # not the same object as a snapshot with a verified one.
        unverified.append("episodic (vector store unavailable)")
    # v3.1.9 (hostile pass 5, C5-4). F1's torn-bytes rule (above) covered the
    # facts file only. A torn `summaries/<conv>.json` — the identical shape,
    # one layer over: a sync-lock/journal incident cuts the file mid-write,
    # leaving facts and episodic fully readable — used to just be recorded
    # unverified HERE with nothing kept aside. The overwrite that followed
    # (admin_import_conversation, or cleanup_test_conversations' wipe) then
    # called summarizer.save_state and replaced the file wholesale: the torn
    # bytes — the only surviving evidence of what the L1/L2/L3 hierarchy had
    # been — were gone, and the published snapshot said "summaries
    # (unreadable)" with nothing behind that sentence to restore from.
    # Reproduced (SP\p5-c\admin_probe.py case A): status 200, quarantine
    # published, `torn facts_path: None`, "UNIQUE-TORN-SUMMARY-MARKER"
    # nowhere in the quarantine dir. Same fix as F1, one layer over: capture
    # the torn file's raw bytes here, verify/publish them alongside the JSON
    # snapshot below, and refuse the whole quarantine (QuarantineError) if
    # even the raw bytes cannot be copied aside — never proceed on a "the
    # store is unreadable, nothing to lose" guess.
    _torn_summary_raw: bytes | None = None
    try:
        summarizer.load_state(conv_id)
    except memory.StoreUnreadable as e:
        unverified.append("summaries (unreadable)")
        logger.warning(
            f"conv={conv_id}: quarantine could not read the summary state "
            f"file ({e}); the OTHER layers are still measured and "
            f"snapshotted, and the torn file's raw bytes are copied aside "
            f"rather than letting an overwrite replace them with nothing "
            f"kept (hostile pass 5, C5-4 — the same rule F1 gave the facts "
            f"file)"
        )
        try:
            _torn_summary_raw = e.path.read_bytes()
        except OSError as read_err:
            raise QuarantineError(
                f"conv={conv_id}: the summary state file is unreadable "
                f"({e}) AND its raw bytes could not be copied aside either "
                f"({type(read_err).__name__}: {read_err}) — refusing to "
                f"publish a snapshot that would lose the one copy of "
                f"evidence for what the summary hierarchy was"
            ) from read_err
    except Exception:  # pragma: no cover - load_state's own best-effort paths
        unverified.append("summaries (unreadable)")

    bundle = export_conversation(conv_id)
    bundle["quarantine"] = {
        "reason": reason,
        "written_at": int(time.time()),
        "expected": {
            "facts": expected_facts,
            "episodic": expected_episodic,
            "archive": len(archived_rows),
            "chapters": len(chapter_rows),
        },
        # The three layers the v2.1 bundle payload has no key for. Carried
        # verbatim so a restore is a copy, not a reconstruction.
        "archive": list(archived_rows),
        "persona": persona_record,
        # v3.1.9.4 (P15-3): the third layer — see the read above.
        "chapters": list(chapter_rows),
        "unverified_layers": list(unverified),
        # F1: filled in below once the torn-facts sibling path is known
        # (computed after `published`, a few lines down) — None when the
        # facts layer was readable, a string path once it is.
        "torn_facts_path": None,
        # C5-4 (hostile pass 5): the same field, one layer over, for a torn
        # summary state file. Also filled in below.
        "torn_summary_path": None,
        "restore_hint": (
            "per-row: facts.restore_from_archive(conv_id) — preferred. "
            "whole-conversation: import_conversation(this file, "
            "target_conv_id=<conv>, overwrite=True) — discards anything "
            "learned since this file was written. import_conversation does "
            "NOT restore the three layers under quarantine.archive, "
            "quarantine.persona and quarantine.chapters: put those back with "
            "facts.save_archive(conv_id, bundle['quarantine']['archive']), "
            "persona.save_persona(conv_id, "
            "bundle['quarantine']['persona']['persona_text']), and "
            "summarizer._archive_chapters(conv_id, "
            "bundle['quarantine']['chapters'])."
        ),
    }

    # Contradiction #1, before anything is written: the export ran its reads
    # through `except Exception` and would have handed back [] for a facts
    # file that raised. It cannot be short of what we counted.
    if len(bundle.get("facts") or []) < expected_facts:
        raise QuarantineError(
            f"conv={conv_id}: snapshot holds "
            f"{len(bundle.get('facts') or [])} fact(s) but the store held "
            f"{expected_facts} a moment ago — refusing to publish a snapshot "
            f"that does not contain what it is supposed to protect"
        )

    # Contradiction #1b (F2, hostile pass 4). The facts check above has no
    # sibling on episodic, and `retrieval.export_indexed_exchanges` is
    # contractually "[] on ANY failure, never raises" (its own docstring) —
    # the exact "unknown read as empty" shape the facts check exists to
    # catch, one layer over. This was sized for quarantine's first caller (a
    # facts cleanup, which never touches episodic, so an unverified episodic
    # count was recorded rather than fatal). v3.1.9 reused this function as
    # the safety net for /admin/conversations/import's overwrite, which DOES
    # empty the episodic index (import_conversation -> forget_conversation)
    # — so a snapshot that is short of what conversation_doc_count just
    # measured must refuse exactly like a short facts snapshot does: a
    # transient export failure at the moment of the snapshot would otherwise
    # publish a 0-row bundle that verifies clean against itself (0 == 0) and
    # the overwrite would then empty an index this snapshot claimed to have
    # preserved. Skipped when expected_episodic is None — that means the
    # count itself could not be measured (already recorded in
    # unverified_layers above), which is a different, non-fatal claim than
    # "we measured N and got fewer back".
    if (
        expected_episodic is not None
        and len(bundle.get("episodic") or []) < expected_episodic
    ):
        raise QuarantineError(
            f"conv={conv_id}: snapshot holds "
            f"{len(bundle.get('episodic') or [])} episodic entr(ies) but the "
            f"store held {expected_episodic} a moment ago — refusing to "
            f"publish a snapshot that does not contain what it is supposed "
            f"to protect"
        )

    quarantine_dir().mkdir(parents=True, exist_ok=True)
    published = _quarantine_path(conv_id)
    partial = published.with_name(published.name + ".partial")
    # F1 / C5-4 (hostile pass 5): every layer whose own file was unreadable
    # gets its raw bytes staged and published beside the JSON snapshot under
    # the same rename-into-place discipline — a `.torn` file that only ever
    # appears once BOTH it and the JSON snapshot have been proven written.
    # One list keyed by layer name instead of one pair of variables per
    # layer, so a THIRD layer (a future caller that starts clearing persona
    # or the archive sidecar — see the C5-4 fix note above) is one more
    # entry here, not a third copy of the stage/verify/publish/cleanup block
    # below: exactly the "a rule applied at one call site and missed at its
    # sibling" shape this project keeps paying for. Empty when every layer
    # was readable (the common case; nothing to carry).
    _torn_layers: list[tuple[str, bytes]] = []
    if _torn_facts_raw is not None:
        _torn_layers.append(("facts", _torn_facts_raw))
    if _torn_summary_raw is not None:
        _torn_layers.append(("summary", _torn_summary_raw))
    torn_published: dict[str, Path] = {
        layer: published.with_name(f"{published.stem}.{layer}.torn")
        for layer, _ in _torn_layers
    }
    torn_partial: dict[str, Path] = {
        layer: path.with_name(path.name + ".partial")
        for layer, path in torn_published.items()
    }
    # Filled in now that the paths are known (the dict was built before
    # `published` existed) — written into the METADATA BLOCK, not just this
    # function's return value, so a caller reading the snapshot back off
    # disk (an operator, or a test that does not keep the live process
    # around) can still find the torn bytes.
    if "facts" in torn_published:
        bundle["quarantine"]["torn_facts_path"] = str(torn_published["facts"])
    if "summary" in torn_published:
        bundle["quarantine"]["torn_summary_path"] = str(torn_published["summary"])

    try:
        # Stage. atomic_write_json gives tmp+fsync+replace, so the `.partial`
        # itself is never torn; the `.partial` NAME is what keeps an
        # unverified snapshot from being mistaken for a usable one.
        memory.atomic_write_json(partial, bundle)

        # Contradiction #2: read it back off the disk it will have to be read
        # off later, and check the payload rather than the file size. An
        # unserializable value or a full filesystem shows up here, not in six
        # months when someone needs the file.
        back = memory.read_json_strict(partial, default=None)
        if not isinstance(back, dict):
            raise QuarantineError(
                f"conv={conv_id}: quarantine snapshot did not read back as a "
                f"JSON object"
            )
        if back.get("version") != BUNDLE_VERSION:
            raise QuarantineError(
                f"conv={conv_id}: quarantine snapshot read back with version "
                f"{back.get('version')!r}, expected {BUNDLE_VERSION!r} — "
                f"import_conversation would reject it"
            )
        if back.get("source_conv_id") != conv_id:
            raise QuarantineError(
                f"conv={conv_id}: quarantine snapshot read back for a "
                f"different conversation"
            )
        n_back = len(back.get("facts") or [])
        if n_back < expected_facts:
            raise QuarantineError(
                f"conv={conv_id}: quarantine snapshot read back with {n_back} "
                f"fact(s), expected at least {expected_facts}"
            )
        if len(back.get("episodic") or []) < len(bundle.get("episodic") or []):
            raise QuarantineError(
                f"conv={conv_id}: quarantine snapshot lost episodic entries "
                f"between write and read-back"
            )
        # Same contradiction for the three layers carried in the metadata
        # block. Verified rather than trusted for exactly the reason the
        # payload is: the caller is about to delete these, and a snapshot
        # that quietly dropped them on serialization is worse than no
        # snapshot, because the caller proceeds.
        back_q = back.get("quarantine")
        if not isinstance(back_q, dict):
            raise QuarantineError(
                f"conv={conv_id}: quarantine snapshot read back without its "
                f"metadata block"
            )
        if len(back_q.get("archive") or []) < len(archived_rows):
            raise QuarantineError(
                f"conv={conv_id}: quarantine snapshot read back with "
                f"{len(back_q.get('archive') or [])} archived fact(s), "
                f"expected at least {len(archived_rows)}"
            )
        if persona_record is not None and not (back_q.get("persona") or {}).get(
            "persona_text"
        ):
            raise QuarantineError(
                f"conv={conv_id}: quarantine snapshot read back without the "
                f"persona this conversation has stored"
            )
        # v3.1.9.4 (P15-3): the third layer, same rule.
        if len(back_q.get("chapters") or []) < len(chapter_rows):
            raise QuarantineError(
                f"conv={conv_id}: quarantine snapshot read back with "
                f"{len(back_q.get('chapters') or [])} chapter(s), expected "
                f"at least {len(chapter_rows)}"
            )

        # F1 / C5-4: stage and verify every torn layer's raw bytes exactly
        # like the JSON bundle above — write, read back, compare — BEFORE
        # any file is published under its real name. A copy that silently
        # wrote short (a full disk, a torn write of THIS write) is worse
        # than no copy, because the caller reads its presence as
        # "preserved".
        for _layer, _raw in _torn_layers:
            _tp = torn_partial[_layer]
            _tp.write_bytes(_raw)
            if _tp.read_bytes() != _raw:
                raise QuarantineError(
                    f"conv={conv_id}: the torn {_layer} file's raw-byte "
                    f"copy did not read back identical to what was written"
                )

        # Publish. Same rename-into-place backup.py uses: the file appears
        # under its real name only once it has been proven readable.
        os.replace(partial, published)
        for _layer in torn_partial:
            os.replace(torn_partial[_layer], torn_published[_layer])
    except Exception:
        try:
            partial.unlink()
        except OSError:
            pass
        for _tp in torn_partial.values():
            try:
                _tp.unlink()
            except OSError:
                pass
        raise

    # Counts only. Never fact text, never conversation content — this log goes
    # to an operator's terminal and the store holds real personal memory.
    logger.info(
        f"conv={conv_id}: quarantine snapshot published ({reason}): "
        f"{n_back} fact(s), {len(back_q.get('archive') or [])} archived, "
        f"{len(back.get('episodic') or [])} episodic, "
        f"summary={'yes' if _has_summary_content(back.get('summary_state')) else 'no'}, "
        f"chapters={len(back_q.get('chapters') or [])}, "
        f"persona={'yes' if back_q.get('persona') else 'no'}"
        + (f", unverified: {'; '.join(unverified)}" if unverified else "")
        + (
            f", torn bytes kept for: {', '.join(sorted(torn_published))}"
            if torn_published else ""
        )
    )

    return {
        "path": published,
        "facts": n_back,
        "archive": len(back_q.get("archive") or []),
        "episodic": len(back.get("episodic") or []),
        "summary": _has_summary_content(back.get("summary_state")),
        "persona": bool(back_q.get("persona")),
        # v3.1.9.4 (P15-3): the third carried layer.
        "chapters": len(back_q.get("chapters") or []),
        "unverified_layers": unverified,
        # F1: present only when the facts layer was unreadable — the raw
        # bytes of the torn file, copied aside next to this snapshot. None
        # (not absent) when there was nothing to carry, so a caller can tell
        # "facts were readable" from "this response predates the field".
        "torn_facts_path": torn_published.get("facts"),
        # C5-4 (hostile pass 5): the same field, one layer over, for a torn
        # summary state file.
        "torn_summary_path": torn_published.get("summary"),
    }


# ---------------------------------------------------------------------------
# Test/placeholder conversation cleanup — V3.1.4 N6 store pollution
# ---------------------------------------------------------------------------
#
# Production carries 129 "conversations" for ~26 real ones (2026-08-30 log
# analysis). The ~103 extras are tooling artifacts, not user activity, and
# they inflate /admin/conversations, the health stats, and every backup
# archive (backup.py's copytree(STORAGE_ROOT) copies them right alongside
# real memory). Three sources:
#
#   1. `CLONE_CONV_ID_HERE` — INCIDENT_2026-08-24 L6/D7: a runbook
#      placeholder pasted into a command unsubstituted. memory._sanitize is
#      a filename filter, not a validator ([^A-Za-z0-9_\-] stripped, nothing
#      rejected), so the literal string passed straight through and became
#      a real store key. Exact literal match only — this is one specific
#      known-bad id, not a shape.
#   2. `__selftest_oneshot_<8 lowercase hex>__` — minted at
#      selftest.py:307: `f"__selftest_oneshot_{uuid.uuid4().hex[:8]}__"`.
#      One per boot before F23 (v3.1) fixed the delete/tail race that
#      orphaned them; F23 stopped new ones, it did nothing about the ones
#      it predates. NOT `__selftest__` (selftest.py:70) — that sentinel's
#      own round trip purges its files on both sides (`_purge_conv_files`
#      before and after) and isn't named in the N6 pollution count, so
#      matching it here would be inventing a fourth pattern nothing asked
#      for.
#   3. `itest-<hex>` and its descriptive variants — minted at
#      tests/integration/_harness.py:131: `f"itest-{uuid.uuid4().hex[:12]}"`,
#      and at individual call sites with a descriptive segment before the
#      hex, e.g. tests/integration/test_dedup.py:
#      `f"itest-dedup-{uuid.uuid4().hex[:8]}"`, test_persona.py:
#      `f"itest-persona-src-{uuid.uuid4().hex[:8]}"`, test_archive.py,
#      test_portability.py similarly. Every one of them is the literal
#      `itest-`, zero or more lowercase `word-` segments, then a trailing
#      run of lowercase hex (uuid4().hex is always lowercase 0-9a-f) 6 to
#      16 characters long (the shortest seen is test_portability.py's
#      `hex[:6]`, the longest the harness default `hex[:12]`; 16 leaves
#      headroom without opening the pattern up to arbitrary trailing text).
#
# A pattern match is a HYPOTHESIS, not a verdict — two independent refusals:
#
#   - a conv_id that matches none of the three shapes above is never a
#     candidate. There is no "probably a test id" tier and no fuzzy
#     matching; a real conv_id is a UUID (header/body-metadata path) or a
#     16-hex sha256 prefix (hash-fallback path, memory._fingerprint_hash) —
#     neither shape starts with `itest-` or `__selftest_oneshot_` or equals
#     `CLONE_CONV_ID_HERE` by construction, so this signal does not degrade
#     as the store grows.
#   - a conv_id that DOES match is still refused (kept, never quarantined
#     or wiped) if it holds more than a token amount of memory. A real id
#     colliding with a test pattern is exactly the scenario the sanitizer
#     already proved possible once (CLONE_CONV_ID_HERE itself), so the
#     pattern alone is not trusted to carry the decision. See
#     _SUBSTANTIAL_* below for the threshold and its evidence.

# Exact literal only — see point 1 above.
CLONE_PLACEHOLDER_CONV_ID = "CLONE_CONV_ID_HERE"

_SELFTEST_ONESHOT_RE = re.compile(r"^__selftest_oneshot_[0-9a-f]{8}__$")

_ITEST_RE = re.compile(r"^itest-(?:[a-z]+-)*[0-9a-f]{6,16}$")

# Below this, a matched conv_id's CONTENT still looks test-shaped and is
# safe to remove; at or above it, the id is kept regardless of which
# pattern matched. Evidence for where the line goes:
#   - real conversations: INCIDENT_2026-08-24 L6 measured 105-106 facts and
#     ~85-98 indexed exchanges for a real conversation. test_conv_fork.py's
#     production case is the same order of magnitude (106 facts, ~85
#     indexed).
#   - test conversations: the richest seed in the whole integration suite
#     is 3 facts (tests/integration/test_dedup.py,
#     test_dedup_merges_seeded_duplicates_via_import) and
#     tests/integration/test_archive.py's largest fixture is also 3 facts.
#     Every other integration fixture seeds 0-2.
# 10 sits above every real integration-test fixture by more than 3x and
# below every measured real conversation by more than 8x, so a
# mismeasurement in either direction lands on the correct side of the line.
SUBSTANTIAL_FACTS = 10
SUBSTANTIAL_ARCHIVED_FACTS = 10
SUBSTANTIAL_EPISODIC = 10


def _test_conv_match_reason(conv_id: str) -> str | None:
    """Which pattern conv_id matches, or None if it matches none of them.

    Order doesn't matter — the three shapes are disjoint by construction
    (one is a fixed literal, one starts `__`, one starts `itest-`).
    """
    if conv_id == CLONE_PLACEHOLDER_CONV_ID:
        return "runbook placeholder literal (CLONE_CONV_ID_HERE)"
    if _SELFTEST_ONESHOT_RE.match(conv_id):
        return "selftest.py one-shot round-trip sentinel (__selftest_oneshot_*)"
    if _ITEST_RE.match(conv_id):
        return "tests/integration harness sentinel (itest-*)"
    return None


def _substantial_reasons(conv_id: str) -> list[str]:
    """Why a pattern-matched conv_id must be KEPT, if any. Empty means it
    is safe to remove.

    Every layer that can hold real memory is checked, and an unreadable
    layer counts as substantial rather than as empty — the failure mode of
    a cleanup tool guessing "empty" on a read error is a silent real-data
    delete, which is the one outcome this whole facility exists to prevent
    (same rule quarantine_conversation applies: StoreUnreadable must abort,
    never be read as zero).
    """
    reasons: list[str] = []

    try:
        n_facts = len(facts.load_facts(conv_id))
    except Exception as e:
        reasons.append(f"facts layer unreadable ({e}) — treated as substantial")
    else:
        if n_facts > SUBSTANTIAL_FACTS:
            reasons.append(f"{n_facts} active fact(s) (> {SUBSTANTIAL_FACTS})")

    try:
        n_archived = len(facts.load_archive(conv_id))
    except Exception as e:
        reasons.append(f"archive sidecar unreadable ({e}) — treated as substantial")
    else:
        if n_archived > SUBSTANTIAL_ARCHIVED_FACTS:
            reasons.append(f"{n_archived} archived fact(s) (> {SUBSTANTIAL_ARCHIVED_FACTS})")

    n_episodic = retrieval.conversation_doc_count(conv_id)
    if n_episodic is None:
        # None means "could not tell" (F61), never zero — treated the same
        # as an unreadable layer above.
        reasons.append("episodic layer unreadable (vector store unavailable) — treated as substantial")
    elif n_episodic > SUBSTANTIAL_EPISODIC:
        reasons.append(f"{n_episodic} indexed exchange(s) (> {SUBSTANTIAL_EPISODIC})")

    try:
        persona_record = persona.load_persona(conv_id)
    except Exception as e:
        reasons.append(f"persona layer unreadable ({e}) — treated as substantial")
    else:
        if persona_record:
            reasons.append("has a stored persona")

    try:
        summary_state = summarizer.load_state(conv_id)
    except Exception as e:
        reasons.append(f"summary layer unreadable ({e}) — treated as substantial")
    else:
        if _has_summary_content(summary_state):
            reasons.append("has summary state (L1/L2/L3)")

    return reasons


def find_test_conversations() -> list[dict]:
    """Scan every known conv_id and classify it. Read-only — never mutates
    anything, so it is always safe to call for a dry-run report.

    Returns one dict per PATTERN MATCH (conv_ids that match nothing are not
    in the list at all):
        {"conv_id", "pattern", "safe_to_remove", "reasons_kept"}
    `reasons_kept` is empty exactly when `safe_to_remove` is True.
    """
    out: list[dict] = []
    for conv_id in memory.list_known_conv_ids():
        reason = _test_conv_match_reason(conv_id)
        if reason is None:
            continue
        kept_because = _substantial_reasons(conv_id)
        out.append(
            {
                "conv_id": conv_id,
                "pattern": reason,
                "safe_to_remove": not kept_because,
                "reasons_kept": kept_because,
            }
        )
    return out


async def cleanup_test_conversations(
    *,
    dry_run: bool = True,
    wipe_layers: Callable[[str], Awaitable[dict]] | None = None,
) -> dict:
    """Find, and optionally remove, test/placeholder conversations.

    dry_run=True (the default, and the only mode that runs without
    `wipe_layers`): reports every pattern match — which pattern, and
    whether it would be removed or kept and why — and touches nothing.

    dry_run=False: for every match with safe_to_remove=True,
    quarantine_conversation() first — writes and VERIFIES a restorable
    snapshot, raising QuarantineError if it cannot prove the snapshot holds
    what the store held a moment ago — and only on success is
    `wipe_layers(conv_id)` awaited to actually clear the conversation. A
    quarantine failure for one conv_id is logged and recorded in
    "errors"; it does not touch that conv_id and does not stop the batch.
    Nothing is ever unlinked directly — quarantine-then-wipe is the same
    reversible path /forget and the admin facts-delete endpoint use.

    Matches that are NOT safe_to_remove are always listed under "kept",
    dry-run or not, and are never quarantined or wiped — matching a
    pattern is necessary, never sufficient (see the module comment above).

    `wipe_layers` is injected rather than imported, the same way
    commands.py takes a `clear_all_memory` callable through its ctx dict
    instead of importing main.py: main.py already imports this module, so
    portability importing back from main (or from commands, which itself
    imports portability) would be a cycle. Wire it in main.py to
    commands._wipe_all_layers bound to _clear_all_memory, e.g.:

        async def _wipe(conv_id: str) -> dict:
            return await commands._wipe_all_layers(
                conv_id, lambda cid: _clear_all_memory(cid, source="cleanup")
            )
        await portability.cleanup_test_conversations(
            dry_run=dry_run, wipe_layers=_wipe
        )

    That gives the cleanup the same archive-sidecar clear and empty-facts
    tombstone a normal /forget leaves — not just _clear_all_memory's three
    layers.
    """
    if not dry_run and wipe_layers is None:
        raise ValueError(
            "wipe_layers is required when dry_run=False — see this "
            "function's docstring for what to wire it to"
        )

    matches = find_test_conversations()
    removable = [m for m in matches if m["safe_to_remove"]]
    kept = [m for m in matches if not m["safe_to_remove"]]

    result: dict[str, Any] = {
        "dry_run": dry_run,
        "scanned": len(memory.list_known_conv_ids()),
        "matched": len(matches),
        "removable": len(removable),
        "kept": [
            {"conv_id": m["conv_id"], "pattern": m["pattern"], "reasons": m["reasons_kept"]}
            for m in kept
        ],
        "removed": [],
        "errors": [],
    }
    if dry_run:
        result["would_remove"] = [
            {"conv_id": m["conv_id"], "pattern": m["pattern"]} for m in removable
        ]
        return result

    for m in removable:
        conv_id = m["conv_id"]
        try:
            snapshot = quarantine_conversation(
                conv_id, reason=f"N6 cleanup: {m['pattern']}"
            )
        except Exception as e:
            logger.error(
                f"conv={conv_id}: cleanup quarantine failed, LEAVING IN PLACE: {e}"
            )
            result["errors"].append(
                {"conv_id": conv_id, "stage": "quarantine", "error": str(e)}
            )
            continue

        try:
            wipe_result = await wipe_layers(conv_id)
        except Exception as e:
            logger.error(
                f"conv={conv_id}: cleanup wipe failed AFTER a verified quarantine "
                f"snapshot was written to {snapshot['path']} — the snapshot is "
                f"safe, the conversation itself was not cleared: {e}"
            )
            result["errors"].append(
                {
                    "conv_id": conv_id,
                    "stage": "wipe",
                    "error": str(e),
                    "quarantine_path": str(snapshot["path"]),
                }
            )
            continue

        logger.info(
            f"conv={conv_id}: cleanup removed ({m['pattern']}); "
            f"quarantine={snapshot['path']}"
        )
        result["removed"].append(
            {
                "conv_id": conv_id,
                "pattern": m["pattern"],
                "quarantine_path": str(snapshot["path"]),
                "wipe": wipe_result,
            }
        )

    return result


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------

class ImportError_(Exception):
    """Bundle malformed or schema-unsupported. Endpoint maps to 400."""


def _validate_bundle(bundle: dict) -> None:
    """Cheap shape check before any I/O. Raises ImportError_ on failure."""
    if not isinstance(bundle, dict):
        raise ImportError_("bundle must be a JSON object")
    v = bundle.get("version")
    if v != BUNDLE_VERSION:
        # Strict version match for v1. When v2.2 bundles exist, this
        # gains a compatibility table — for now reject anything unknown
        # rather than silently misinterpret fields.
        raise ImportError_(
            f"unsupported bundle version {v!r} — expected {BUNDLE_VERSION!r}"
        )
    for key in ("facts", "summary_state", "episodic"):
        if key not in bundle:
            raise ImportError_(f"bundle missing required key: {key!r}")
    if not isinstance(bundle.get("facts"), list):
        raise ImportError_("bundle.facts must be a list")
    if not isinstance(bundle.get("episodic"), list):
        raise ImportError_("bundle.episodic must be a list")
    if not isinstance(bundle.get("summary_state"), dict):
        raise ImportError_("bundle.summary_state must be an object")
    # v3.1.9 (M3). Before any I/O, like everything else here: the whole
    # bundle must be writable. A lone surrogate is valid JSON and a valid
    # str, and raised UnicodeEncodeError only at the first save — after the
    # old facts and episodic rows had been cleared. The endpoint guards the
    # request body too; this is the guard that protects every other caller
    # of import_conversation, including scripts that load a bundle from disk.
    try:
        json.dumps(bundle, ensure_ascii=False).encode("utf-8")
    except UnicodeEncodeError as e:
        raise ImportError_(
            f"bundle contains text that cannot be written as UTF-8 "
            f"(an unpaired surrogate): {e}"
        )


def _validate_target_ready(bundle: dict, *, target_conv_id: str | None = None) -> str:
    """The cheap, side-effect-free part of `import_conversation`: bundle
    shape, target-id resolution, and the in-flight-writer lock check.
    Raises ImportError_ (or memory.UnsafeConvId, from an unsafe conv_id
    reaching `memory.conv_lock`) on exactly the failures import_conversation
    itself would raise for the same inputs. Returns the resolved target
    conv_id.

    v3.1.9 (hostile pass 4, F7). Split out so a caller that is about to take
    a pre-overwrite quarantine snapshot (main.admin_import_conversation) can
    run these checks FIRST and skip the snapshot entirely when the import
    was always going to be refused. Before this split, a bad bundle version,
    `bundle.facts` not a list, or a target held by an in-flight writer all
    failed INSIDE import_conversation, AFTER the caller had already
    published a full quarantine snapshot — so every refused attempt (a
    retry loop, a scripted health check, a client resending a stale bundle)
    left one more never-pruned copy of the conversation on disk, for no
    output that a bundle validator alone could not have said in
    microseconds.

    import_conversation calls this AGAIN, immediately before it writes —
    not a redundant check but the fix for the TOCTOU the snapshot's own I/O
    opens: the lock could be taken between this function returning here (in
    the caller, before the snapshot) and import_conversation's write, so
    the write must re-verify it is still clear right before it happens, not
    trust a lock state that is now stale by however long the snapshot took.

    Also fixes two F8 LOWs (hostile pass 4) that used to surface only deep
    inside quarantine_conversation or import_conversation's own writes:
      (b) a whitespace-padded target_conv_id used to reach
          quarantine_conversation UNSTRIPPED (main.py quarantined the raw
          body value, only import_conversation stripped it), so the
          snapshot step read a conv_id the import step would not — a
          harmless-looking mismatch that surfaced as a misleading 409
          ("could not write a snapshot") for what import would have handled
          as an ordinary existing-state 400. Both steps now resolve the SAME
          stripped `target` from this one function.
      (c) a non-string target_conv_id / bundle.source_conv_id (an int, a
          list) used to reach `.strip()` a few lines below and raise an
          uncaught AttributeError — a 500 for caller input every other
          malformed-body case in this file answers with 400. Checked
          explicitly, ahead of the `.strip()` call.
    """
    _validate_bundle(bundle)

    # F8(c): reject a non-string id explicitly rather than let `.strip()`
    # raise AttributeError on it a few lines down. `target_conv_id or
    # bundle.get(...)` picks whichever of the two candidates is present
    # (falsy values — None, "", 0 — fall through to the next one, matching
    # the original truthiness-based fallback exactly); only the TYPE of
    # whatever wins that fallback is checked here.
    _candidate = target_conv_id if target_conv_id else bundle.get("source_conv_id")
    if _candidate is not None and not isinstance(_candidate, str):
        raise ImportError_(
            f"conv_id must be a string, got {type(_candidate).__name__}: "
            f"{_candidate!r}"
        )
    # F8(b): stripped HERE, once, so every caller (the pre-snapshot check in
    # main.py and import_conversation's own write below) resolves the
    # identical target — no more "quarantine saw the raw id, import saw the
    # stripped one".
    target = (_candidate or "").strip()
    if not target:
        raise ImportError_("no target_conv_id provided and bundle has no source_conv_id")

    # F8(b), continued: the same class of refusal memory._safe_path raises
    # for a conv_id that would resolve outside STORAGE_ROOT (a traversal
    # attempt, or one that is merely unusable as a path component) used to
    # surface only once real I/O started — inside quarantine_conversation's
    # first read, deep enough that main.py's generic `except Exception`
    # around the snapshot read it as "the snapshot could not be written" and
    # answered 409, not as the 400-about-the-request it actually is.
    # memory.facts_path resolves a Path and does no I/O, so calling it here
    # is as cheap as everything else in this function and raises the same
    # memory.UnsafeConvId the rest of this codebase already maps to 400
    # (main.py's `except (portability.ImportError_, UnsafeConvId)`).
    memory.facts_path(target)

    # v3.1 D18: archive, restore and dedup all serialize on conv_lock; import
    # — the one operation that clears three layers and rewrites them wholesale
    # — did not. The hazard is not two concurrent importers. It is an
    # extraction tail that read facts before the import ran, is parked on its
    # vLLM call while holding conv_lock, and writes that pre-import snapshot
    # back the moment it returns. The bundle is gone, with no error anywhere
    # and `overwrote_existing: true` in the response.
    #
    # This function is a plain `def` called from an `async def` endpoint, so it
    # runs to completion on the event loop without yielding: nothing can take
    # conv_lock while it is running, and it cannot await to take the lock
    # itself. locked() is therefore the whole of the mutual exclusion — held
    # means a writer is parked mid-sequence and this import must not land
    # underneath it. A refused import loses nothing and the operator retries;
    # a clobbered one loses the bundle. Making this `async def` and awaiting
    # the lock is the better shape and needs its two call sites in main.py
    # (the import and fork endpoints) to await it.
    if memory.conv_lock(target).locked():
        raise ImportError_(
            f"conv_id {target!r} has a memory write in flight (extraction tail, "
            f"archive, restore or dedup). Refusing rather than import "
            f"underneath it — that writer would overwrite the bundle on its "
            f"next save. Retry in a moment."
        )
    return target


def import_conversation(
    bundle: dict, *, target_conv_id: str | None = None, overwrite: bool = False
) -> dict:
    """Restore a conversation from a bundle.

    `target_conv_id`: where to land the data. Default is the bundle's
    own source_conv_id (so re-importing into the same pod restores
    in place). Override to clone into a fresh conv_id without touching
    the original.

    `overwrite`: if False (default), refuses to import when target conv
    already has any state — prevents accidental wipe of an active conv.
    If True, replaces existing state wholesale.

    Returns a counters dict for the response body.
    """
    target = _validate_target_ready(bundle, target_conv_id=target_conv_id)

    # Pre-flight: detect existing state to honor overwrite=False.
    #
    # v3.1: this guard exists to stop an import silently wiping a live
    # conversation, so "I could not check" must be treated as "occupied" — the
    # opposite reading is how a safety check becomes a data-loss path. Two
    # sources of not-knowing, both introduced by making failure visible rather
    # than inventing a value:
    #   - conversation_doc_count returns None when the vector store is
    #     unavailable (0 now means genuinely empty, and only that)
    #   - load_facts / load_state raise StoreUnreadable on a corrupt or
    #     unreadable file
    # Either way we refuse unless the caller has explicitly said overwrite.
    # The message is operator-facing but travels out through an HTTP body, so
    # the layer name goes in the reason and the underlying path stays in the log.
    unverifiable: list[str] = []
    pre_existing = False
    try:
        pre_existing = len(facts.load_facts(target)) > 0
    except memory.StoreUnreadable as e:
        unverifiable.append("facts (unreadable)")
        logger.warning(f"conv={target}: import pre-flight could not read facts: {e}")
    n_indexed = retrieval.conversation_doc_count(target)
    if n_indexed is None:
        unverifiable.append("episodic (vector store unavailable)")
        logger.warning(f"conv={target}: import pre-flight could not reach the vector store")
    elif n_indexed > 0:
        pre_existing = True
    try:
        if summarizer.load_state(target).get("l1"):
            pre_existing = True
    except memory.StoreUnreadable as e:
        unverifiable.append("summaries (unreadable)")
        logger.warning(f"conv={target}: import pre-flight could not read summaries: {e}")

    if unverifiable and not overwrite:
        raise ImportError_(
            f"cannot verify whether target conv_id {target!r} is empty — "
            f"{'; '.join(unverifiable)}. Refusing rather than risk overwriting "
            f"a live conversation; pass overwrite=true to import anyway"
        )
    if pre_existing and not overwrite:
        raise ImportError_(
            f"target conv_id {target!r} has existing state; "
            f"pass overwrite=true to replace"
        )

    if unverifiable and overwrite:
        # Proceeding past a safety check that could not run is exactly the kind
        # of thing that must leave a record — the operator chose this, but in
        # six months the log is the only evidence the check was skipped rather
        # than passed.
        logger.warning(
            f"conv={target}: importing with overwrite=true while unable to "
            f"verify existing state ({'; '.join(unverifiable)}) — proceeding "
            f"on the caller's explicit instruction"
        )

    # If overwriting, nothing old may survive beside the new — a mix of old and
    # new episodic rows confuses retrieval. It used to say "clear first"; see
    # below for why that order was the bug. `unverifiable` counts as
    # "might be occupied": skipping the clear because we could not PROVE state
    # exists is how stale episodic rows survive an overwrite and how
    # overwrote_existing comes to under-report a real replacement.
    #
    # WRITE THE REPLACEMENTS FIRST, CLEAR ONLY WHAT THEY CANNOT REPLACE
    # (v3.1.9, M3). This used to run save_facts(target, []) and
    # forget_conversation BEFORE writing the bundle, so any failure in the
    # bundle write — an unpaired surrogate, ENOSPC, a stalled volume — left
    # the conversation wiped and nothing imported. Demonstrated: 105 facts
    # in, [] on disk. The facts clear bought nothing even when it worked:
    # save_facts replaces the file atomically, so writing the bundle's facts
    # IS the clear. Summary state is the same. Only the episodic index is
    # additive, so it is the one thing that must be emptied — and it is now
    # emptied only after both atomic writes above it have landed.

    # v3.1.9.4 (R4 / W1, P15-5 follow-up). An overwrite onto a target that
    # already had state (pre_existing) or could not be proven empty
    # (unverifiable) is the SAME class of wipe main._clear_all_memory and
    # commands._handle_retire's apply step already bump for: it replaces
    # this conv_id's facts, and — a few lines down — its facts archive,
    # chapter archive and persona too. Without the bump, a tail submitted
    # against `target` before this import ran, still parked on the pool's
    # concurrency semaphore or mid a vLLM extraction call, captures the
    # generation as it stood before the import, finds it UNCHANGED when it
    # finally re-checks under conv_lock, and writes its pre-import facts
    # straight into the freshly-imported store.
    #
    # Computed once and reused below (the archive/chapter/persona clear
    # shares the identical condition) rather than bumping unconditionally:
    # an overwrite=True import onto a genuinely empty, freshly-created
    # target is not wiping anything, and a bump that fires for every import
    # call would make the generation counter noisy for no protective
    # benefit.
    #
    # WHY NO conv_lock HERE, unlike every other bump site. import_conversation
    # is a plain `def` called from an `async def` endpoint (see
    # _validate_target_ready's own comment on this a few lines up) and never
    # awaits, so it runs to completion on the event loop without yielding —
    # nothing else can observe `target`'s generation or take conv_lock(target)
    # while this function is running. That makes the WHOLE function one
    # atomic section from a concurrent tail's point of view, which is why
    # _validate_target_ready's `locked()` probe was sufficient mutual
    # exclusion for the write below and is equally sufficient here: the bump
    # only has to happen somewhere before this function returns, not under an
    # explicit lock it cannot await for.
    will_replace_existing = (pre_existing or unverifiable) and overwrite
    if will_replace_existing:
        memory.bump_wipe_generation(target)

    # Restore facts wholesale (already-pruned by export, no further pruning).
    facts.save_facts(target, list(bundle.get("facts", [])))

    # Restore summary state wholesale.
    summarizer.save_state(target, dict(bundle.get("summary_state", {})))

    if will_replace_existing:
        retrieval.forget_conversation(target)
        # v3.1.9.4 (P15-3). THREE MORE layers the bundle payload does not
        # carry and this overwrite never touched: the target's OLD facts
        # archive, L2 chapter cold store and persona survived every import,
        # mixed in beside the freshly-written conversation — a "forget"
        # this is not: the SOURCE bundle's own state lands correctly, but
        # whatever the TARGET id had before (chapters an L3 refresh had
        # already archived, facts an eviction had archived, a persona)
        # stayed live under the same id. All three are already preserved —
        # main.admin_import_conversation takes a quarantine_conversation
        # snapshot of the target BEFORE calling here whenever overwrite is
        # set, and that snapshot now carries all three (see the D8 block
        # comment above quarantine_conversation) — so clearing them here is
        # the same "write the replacement, clear what it cannot replace"
        # shape this function already applies to the episodic index one
        # line up, not a second archive-before-delete of its own.
        try:
            facts.save_archive(target, [])
        except Exception as e:
            logger.warning(
                f"conv={target}: could not clear the old facts archive on "
                f"overwrite: {e}"
            )
        try:
            _chapter_path = memory.summary_archive_path(target)
            if _chapter_path.is_file():
                _chapter_path.unlink()
        except Exception as e:
            logger.warning(
                f"conv={target}: could not clear the old chapter archive on "
                f"overwrite: {e}"
            )
        try:
            persona.clear_persona(target)
        except Exception as e:
            logger.warning(
                f"conv={target}: could not clear the old persona on "
                f"overwrite: {e}"
            )

    # Re-embed and re-index each exchange.
    episodic_imported = 0
    for entry in bundle.get("episodic", []):
        try:
            ti = int(entry.get("turn_index", -1))
            doc = entry.get("document", "")
            if ti < 0 or not doc:
                continue
            if retrieval.import_indexed_exchange(target, ti, doc):
                episodic_imported += 1
        except Exception as e:
            logger.warning(f"conv={target}: skipped one episodic entry: {e}")

    logger.info(
        f"conv={target}: imported {len(bundle.get('facts', []))} fact(s), "
        f"{episodic_imported} episodic, "
        f"summary={'yes' if bundle.get('summary_state') else 'no'}"
    )

    return {
        "conv_id": target,
        "imported": {
            "facts": len(bundle.get("facts", [])),
            "episodic": episodic_imported,
            "summary": bool(bundle.get("summary_state")),
        },
        # True when the clear step actually ran. `unverifiable` is included
        # because we clear on it: reporting False there would tell the caller
        # nothing was replaced while the file on disk had just been rewritten.
        "overwrote_existing": bool((pre_existing or unverifiable) and overwrite),
        # Non-empty when a layer could not be checked before importing.
        "unverified_layers": list(unverifiable),
    }


# ---------------------------------------------------------------------------
# Fork
# ---------------------------------------------------------------------------

def _fact_key(text: str) -> str:
    """Identity for merge de-duplication: casefolded, whitespace-collapsed.

    Deliberately NOT semantic. dedup.py owns semantic merging and pays an
    LLM for it; this only has to avoid importing a byte-identical fact
    twice, which is the whole overlap when a conversation forks and both
    halves extract from the same turns.
    """
    return " ".join((text or "").split()).casefold()


def _merge_fact_pin_and_recency(dst_fact: dict, src_fact: dict) -> dict:
    """Fold src's protection into dst's copy of the "same" fact (same
    _fact_key), instead of dst silently winning outright and discarding it.

    R5 (v3.1.7): a pinned source fact merged into a conversation holding an
    unpinned copy came out unpinned, on the documented install runbook step
    (pipelines/conversation_id_header.py step 8) that has to run before the
    history cap is enabled. dedup._merge_metadata already solves this for its
    own merge path with `"pin": any(...)`; this matches it:

      - pin: UNION (any member pinned -> merged copy is pinned). A merge is a
        union of meaning, so the strongest protection between the two copies
        carries forward. dedup._merge_metadata's comment on this line applies
        verbatim here.
      - last_used: MAX. facts.py's own comment (see commands.py's /retire
        metadata note) calls this "unix seconds with one writer... safe to
        compare across facts", so unlike added_turn it is meaningful even
        across two different conversations' copies, and it is what eviction
        actually sorts on — keeping the fresher of the two is the more
        protective choice for the surviving row.
      - added_turn: LEFT ALONE (dst's own value survives). dedup._merge_metadata
        takes min() over a cluster, but every member of a dedup cluster comes
        from the SAME conversation's own turn numbering (MEMORY_REVIEW F-1) —
        that's what makes min() meaningful there. src and dst here are two
        different conversations, each numbering its own turns from 0;
        combining their added_turn values would produce a number that looks
        meaningful and isn't, exactly what facts.py warns against ("do not
        compare two facts' added_turn unless they came from the same
        writer"). dst's added_turn is left untouched rather than invented.
      - text: dst's, always — the two copies matched on _fact_key (casefolded,
        whitespace-collapsed), so they may differ in case/spacing but not in
        content; dst's wording is the one already live in the destination.
    """
    return {
        **dst_fact,
        "pin": bool(dst_fact.get("pin")) or bool(src_fact.get("pin")),
        "last_used": max(
            int(dst_fact.get("last_used", 0) or 0),
            int(src_fact.get("last_used", 0) or 0),
        ),
    }


def _dst_last_used_floor(dst_facts: list[dict]) -> int:
    """The `last_used_floor` to hand `_merge_fact_lists` for THIS dst read.

    v3.1.9 F6 (hostile pass 2, review B). See `_merge_fact_lists`'
    `last_used_floor` doc for the mechanism; this just picks the number:
    the newest `last_used` already active in dst, or 0 if dst has no facts
    yet (the recommended-order case — merge before her first message — where
    there is nothing in dst to be older than, and nothing has raced it yet
    either).

    v3.1.9 (hostile pass 3, F6): only ever called now when the caller passed
    `refresh_last_used=True` — see merge_conversation's docstring for why
    this stopped being automatic.
    """
    return max((int(f.get("last_used", 0) or 0) for f in dst_facts), default=0)


def _merge_fact_lists(
    dst_facts: list[dict],
    src_facts: list[dict],
    *,
    last_used_floor: int | None = None,
) -> tuple[list[dict], dict]:
    """Union src into dst by _fact_key. Pure function — callers decide
    whether/when to persist the result, so this is safe to call once against
    a stale read for a dry-run preview and again against a freshly re-read
    dst immediately before writing (merge_conversation does both, and
    commands._retire_plan's "already-in-destination" step uses the same
    per-pair fold via _merge_fact_pin_and_recency for the same reason).

    A brand-new key is appended verbatim. A colliding key keeps dst's row
    (dst's wording is what's live in the destination) but folds pin/last_used
    across both copies with _merge_fact_pin_and_recency — see that function
    for why pin is unioned, last_used is maxed, and added_turn is left alone.

    `last_used_floor` (v3.1.9 F6, hostile pass 2 review B): applied ONLY to
    brand-new (non-colliding) facts, stamping each one's last_used to
    max(its own last_used, last_used_floor) before it is appended.

    Why this exists, and why it is scoped to "added" and not "updated":
    a merged-in fact that does not collide with anything already in dst
    keeps whatever last_used it carried in the SOURCE conversation — often
    hours old by the time an operator runs the merge, because it is exactly
    as old as the fork that made the merge necessary in the first place. If
    the destination has already had a backfill (or any extraction) run
    under it before the merge lands, THOSE facts were minted with
    last_used at (or near) the merge's own wall-clock moment. facts.
    prune_facts's LRU eviction then treats the source's hours-old facts as
    the oldest thing in the whole store and archives them first on the
    very next write — reviewed at 115 of a real user's 136 original facts
    archived on her next exchange, in a store that merge left looking
    intact. Flooring every newly merged fact to at least the destination's
    own current newest last_used puts it in the same race the
    destination's own facts are already running, instead of a race it was
    never a real participant in — it does not fabricate an eviction
    exemption, it only stops the merge itself from handing a fact a
    last_used that makes it look OLDER than the moment it actually landed
    in this store.

    The COLLISION path (an existing dst row) is deliberately NOT touched by
    this floor. _merge_fact_pin_and_recency already computes max(dst, src)
    for that row, which is the correct answer on its own terms — forcing a
    floor on top of it would let one merge silently "refresh" an otherwise
    genuinely stale destination fact for a reason that has nothing to do
    with the merge (the row was already live in dst; the merge only
    touched its pin/last_used metadata, not its existence). The reviewed
    incident's own numbers bear this out: facts_added=135 versus 0
    collisions — the archived facts were overwhelmingly the newly-added
    ones this floor protects, not folded collisions.

    Returns (merged_list, stats): "added" is brand-new keys, "updated" is
    existing keys whose dst row actually changed (pin/last_used moved),
    "unchanged" is existing keys where the fold was a no-op (a true
    byte-for-byte-equivalent duplicate with nothing new to protect).
    """
    merged = list(dst_facts)
    key_to_index: dict[str, int] = {}
    for i, f in enumerate(merged):
        k = _fact_key(f.get("text", ""))
        if k:
            key_to_index.setdefault(k, i)

    added = 0
    updated = 0
    unchanged = 0
    for f in src_facts:
        k = _fact_key(f.get("text", ""))
        if not k:
            continue
        idx = key_to_index.get(k)
        if idx is None:
            new_fact = dict(f)
            if last_used_floor is not None:
                new_fact["last_used"] = max(
                    int(new_fact.get("last_used", 0) or 0), last_used_floor
                )
            key_to_index[k] = len(merged)
            merged.append(new_fact)
            added += 1
        else:
            folded = _merge_fact_pin_and_recency(merged[idx], f)
            if folded != merged[idx]:
                merged[idx] = folded
                updated += 1
            else:
                unchanged += 1

    return merged, {"added": added, "updated": updated, "unchanged": unchanged}


def _evicted_by_origin(
    merged: list[dict], dst_len: int, evicted: list[dict]
) -> tuple[int, int]:
    """Split an `_lru_split` eviction list by which side of the merge each
    fact came from. Returns (dst_evicted, merged_in_evicted).

    v3.1.9 (hostile pass 3, F6): `facts_over_budget_after_merge` used to be a
    bare count, so an operator reading "54 over budget" could not tell
    whether that was 54 of the FORK's old facts (expected, fine) or 54 of
    HER OWN (the failure mode this whole finding is about) — exactly the
    distinction the runbook's "Older forks" step needs before trusting its
    own "expect most of what they add to be evicted" promise.

    `_merge_fact_lists` builds `merged` as `list(dst_facts)` first (so
    indices [0, dst_len) are always dst's own rows, updated in place on a
    collision but never reordered or removed) and only APPENDS brand-new
    src-only facts after that boundary — see its docstring. So origin is a
    plain index-boundary check, not a per-fact tag: identity (`id()`) is
    used to test evicted-list membership against that same object list, the
    same pattern `facts._lru_split` itself uses for `kept_ids`.
    """
    dst_ids = {id(f) for f in merged[:dst_len]}
    dst_evicted = sum(1 for f in evicted if id(f) in dst_ids)
    return dst_evicted, len(evicted) - dst_evicted


# v3.1.9.3 (P11 / race-merge.md, coordinator review round 2). How long
# merge_conversation waits to ACQUIRE conv_lock(dst_conv_id) before giving up
# and refusing (the original D18 behaviour) rather than parking the request.
#
# A summary rebuild can hold conv_lock for the WHOLE drain — 10-30 minutes,
# the identity runbook's own estimate (see the source-side guard's comment
# below) — with her extraction tails queued behind it. Waiting unboundedly
# for that lock would park this admin request, and the threadpool thread
# under it, for up to half an hour: past any operator's patience and past
# any HTTP client or proxy timeout, indistinguishable from a hang. The
# ORIGINAL code refused instantly instead ("Retry in a moment"); trading
# that for an unbounded wait swaps a fast, clear error for a silent stall.
#
# This bounds it instead of choosing between those two: long enough that the
# shapes this fix exists for — two merges racing each other, or a merge
# landing a moment either side of /remember, import, or an ordinary
# (sub-second) extraction tail — serialize and both land, short enough that
# a merge queued behind a genuine multi-minute rebuild still fails FAST with
# the original clear error instead of hanging. 10.0s matches this codebase's
# other precedent for "a bounded wait on an admin-ish, user-triggered
# operation" — commands.FORGET_SETTLE_TIMEOUT, which /forget uses to drain
# the background pool before a wipe, justified there as "a command a user
# issues rarely and deliberately, and it is bounded". merge-into is the same
# shape (rare, deliberate, admin-triggered), so it gets the same number for
# the same reason — not a value tuned to this function specifically.
_MERGE_DST_LOCK_TIMEOUT_S = 10.0


def merge_conversation(
    src_conv_id: str, dst_conv_id: str, *, dry_run: bool = True,
    refresh_last_used: bool = False,
) -> dict:
    """Fold src's FACTS and EPISODIC memory into dst. Both survive.

    WHY THIS EXISTS. The hash-fallback conv_id is
    sha256(system|||first_user[:512]), so editing the system prompt gives a
    live conversation a NEW identity and forks its memory. Observed in
    production 2026-08-30: a prompt edit at ~19:08 left 106 facts and ~85
    indexed exchanges under the old id while the conversation carried on
    under a new one. Nothing is lost when that happens - both halves are
    intact on disk - but until this existed there was no way to put them
    back together.

    WHAT IT DOES NOT TOUCH, and this is the important part:

      * SUMMARIES. The forked half re-derives its own hierarchy from the
        client's full array (that is how the new id reached turn 411 in
        three hours), so dst's summary state already covers the same
        history src's does. Merging them would double-count the narrative
        and corrupt the very layer that survived the fork intact.
      * SRC. Read-only throughout. A merge that damages its source is not
        recoverable if the result is wrong. "Read-only" does not mean
        unguarded, though (v3.1.9, hostile pass 3, reviewer D F5): a memory
        write in flight on src (conv_lock(src_conv_id).locked()) refuses
        the merge just like one on dst does, because a merge that reads src
        while a tail is mid-drain silently omits whatever that tail was
        about to write.

    Facts are unioned on _fact_key; dst's wording wins on collision, but pin
    and last_used are folded across BOTH copies (see _merge_fact_pin_and_recency)
    rather than dst silently winning outright — a pinned src fact merging into
    an unpinned dst copy comes out pinned (R5, v3.1.7), and the fresher of the
    two last_used values survives. The active store is NOT pruned here -
    callers who want the cap enforced can prune afterwards, and leaving that
    separate means a merge never silently evicts.

    Episodic exchanges are imported by turn_index, skipping any index dst
    already holds, because re-embedding over a live index is the one part of
    this that costs GPU and cannot be undone by re-running.

    `dry_run` defaults to TRUE. The compact endpoint defaults the other way
    and that surprised an operator into a live run; this one touches two
    conversations at once and gets the safer default.

    `refresh_last_used` (v3.1.9, hostile pass 3, F6; opt-in, default FALSE).
    When True, every brand-new (non-colliding) merged-in fact has its
    last_used floored to dst's own current newest last_used before it is
    written — see `_merge_fact_lists`'s `last_used_floor` doc for the
    mechanism. This existed unconditionally in 3005438, built for ONE shape:
    an id-migration where dst already holds a BACKFILL's fresh
    re-extractions (minted at ~merge time) and src holds her real, genuinely
    older originals — there, leaving last_used alone made the originals look
    artificially ancient next to an artifact of extraction timing, and
    facts.prune_facts archived the large majority of them on the very next
    write (reviewed: 115 of 136).

    Hostile pass 3 (F6) found the SAME unconditional floor breaks the
    opposite-order, equally documented workflow: RUNBOOK_MEMORY_IDENTITY.md's
    "Older forks" step merges an abandoned fork's facts into the primary
    conversation SHE IS STILL CHATTING IN, no backfill involved. There dst's
    last_used values are GENUINE — real recency of real use, not a backfill
    artifact — and the floor stamps every merged-in (old, correctly
    lower-priority) fork fact up to "the time of her last message", which
    outranks every one of HER OWN facts not touched on that exact turn.
    Measured: floor OFF correctly archives the 14-day-old fork facts on the
    next exchange (matching what the runbook tells the operator to expect);
    floor ON archives 55 of her 85 real facts instead and keeps every one of
    the fork's.

    A boolean signal that distinguishes the two shapes without guessing
    (dst's last_used SPREAD is not reliable — an actively-chatted dst can
    have a spread as wide as a stale-backfill dst does, see the finding) does
    not exist without reading backfill.py's own completion state, which this
    module does not own. So the floor is now OPT-IN rather than automatic:
    default False leaves last_used alone (correct for "Older forks", fork
    reconcile, and any merge into a store that already reflects real usage —
    which is the more common shape, and the one where guessing wrong is
    silent data-priority loss on HER side). Pass refresh_last_used=True
    explicitly for the id-migration/backfill recovery this was built for,
    where the caller (the runbook step, or an operator who just ran a
    backfill under the new id) KNOWS dst's freshness is an extraction
    artifact rather than real use.

    Either way, `facts_over_budget_after_merge` in the result is now split by
    origin (`dst_facts_evicted_after_merge` / `merged_facts_evicted_after_merge`)
    so the operator can see WHICH side would be archived before choosing.
    """
    if not src_conv_id or not dst_conv_id:
        raise ValueError("both src_conv_id and dst_conv_id are required")
    if src_conv_id == dst_conv_id:
        raise ValueError("src and dst are the same conversation")

    # v3.1.9 (hostile pass 3, F9). strict=True: an unreadable src facts file
    # (a torn write, StoreUnreadable) must not merge as "0 facts" — silently
    # proceeding on episodic alone, or refusing with the wrong reason
    # ("nothing to merge") when facts genuinely exist but could not be read.
    # See export_conversation's own docstring for why this is scoped to
    # facts/summary and not episodic (retrieval.py's own never-raises
    # contract, not owned here).
    try:
        src = export_conversation(src_conv_id, strict=True)
    except memory.StoreUnreadable as e:
        raise ValueError(
            f"conv {src_conv_id}: could not read facts or summary state "
            f"({e}) — refusing to merge on an unknown source rather than "
            f"treating an unreadable file as empty"
        ) from e
    src_facts = src.get("facts") or []
    src_episodic = src.get("episodic") or []
    if not src_facts and not src_episodic:
        raise ValueError(
            f"conv {src_conv_id} has no facts and no indexed exchanges - "
            f"nothing to merge (check the id against /admin/conversations)"
        )

    try:
        dst_facts = facts.load_facts(dst_conv_id)
    except Exception as e:
        raise ValueError(f"could not read facts for {dst_conv_id}: {e}") from e

    # Preview only, against this (possibly stale) read — the actual write
    # below re-reads dst and re-runs this same fold so a tail that wrote
    # between here and there is not clobbered or ignored. The floor is
    # recomputed from that same fresh read down there (F6) rather than
    # reused from here, for the identical reason.
    #
    # v3.1.9 (hostile pass 3, F6): floor is OPT-IN now — see the docstring.
    # None means _merge_fact_lists leaves every merged-in fact's own
    # last_used alone, which is what makes real LRU comparison (dst's actual
    # usage vs src's actual usage) possible again for the non-backfill shape.
    _preview_merged, fact_stats = _merge_fact_lists(
        dst_facts, src_facts,
        last_used_floor=_dst_last_used_floor(dst_facts) if refresh_last_used else None,
    )

    try:
        dst_episodic = retrieval.export_indexed_exchanges(dst_conv_id)
    except Exception:
        dst_episodic = []
    dst_turns = {e.get("turn_index") for e in dst_episodic}
    new_exchanges = [
        e for e in src_episodic if e.get("turn_index") not in dst_turns
    ]

    result = {
        "src_conv_id": src_conv_id,
        "dst_conv_id": dst_conv_id,
        "dry_run": dry_run,
        "refresh_last_used": refresh_last_used,
        "src_facts": len(src_facts),
        "dst_facts_before": len(dst_facts),
        "facts_to_add": fact_stats["added"],
        "facts_pin_or_recency_to_update": fact_stats["updated"],
        "facts_skipped_duplicate": fact_stats["unchanged"],
        "src_exchanges": len(src_episodic),
        "dst_exchanges_before": len(dst_episodic),
        "exchanges_to_add": len(new_exchanges),
        "exchanges_skipped_existing": len(src_episodic) - len(new_exchanges),
        "summaries": "not merged (dst re-derived its own; see docstring)",
    }
    # v3.1.9 F6 (hostile pass 2, review B), second half of the finding's
    # "and/or": even with the last_used_floor above, dst's own store can
    # already be over facts.prune_facts's budget before this merge does
    # anything at all — the reviewed real store was ~2x COMPACTOR_MAX_
    # FACTS_TOKENS on its own — and step 9 of the runbook reads this
    # response immediately after the merge, before any tail has pruned
    # anything, so a silently-already-over-budget result would still read
    # as "the merge worked." This runs the SAME LRU split prune_facts uses
    # (facts._lru_split — read-only, no archive write, no log line: a pure
    # preview) against the set this call actually produced, so the operator
    # sees it in the same response rather than discovering it on the next
    # exchange. It cannot predict facts a not-yet-run extraction or backfill
    # will add after this call returns — nothing here can — it answers "is
    # the destination already over budget right after this merge", which is
    # exactly the number that was invisible before.
    _kept_preview, _evicted_preview = facts._lru_split(
        _preview_merged, facts._MAX_FACTS_TOKENS
    )
    result["facts_over_budget_after_merge"] = len(_evicted_preview)
    # v3.1.9 (hostile pass 3, F6): WHICH side would be archived, not just how
    # many — see _evicted_by_origin's docstring. This is the number the
    # runbook's "Older forks" step needs to see before trusting its own
    # "expect most of what they add to be evicted" line, and the number that
    # would have shown the F6 regression (55 of hers, not the fork's) in the
    # response itself instead of only on the next exchange.
    _dst_evicted, _merged_evicted = _evicted_by_origin(
        _preview_merged, len(dst_facts), _evicted_preview
    )
    result["dst_facts_evicted_after_merge"] = _dst_evicted
    result["merged_facts_evicted_after_merge"] = _merged_evicted
    if dry_run:
        return result

    # v3.1.9 (hostile pass 3, reviewer D F5). SOURCE holds the identical
    # hazard the destination guard below closes, one step earlier, and this
    # used to check only dst — merge is read-only on src, and "read-only"
    # was read as "nothing to guard". It is not, on the identity runbook's
    # own R3 (reverse merge): R2's first message under the new uuid starts a
    # summary rebuild that holds conv_lock(uuid) for the WHOLE drain (10-30
    # minutes, the runbook's own estimate; 23 summarization calls measured
    # on one branch), and her NEXT messages under the uuid queue their
    # episodic-index and fact-extraction tails behind that same lock.
    # Running R3 (src=uuid) in that window reads the uuid's store AS IT
    # STANDS mid-rebuild, commits, and reports success (`exchanges_added:
    # 1`) — then the queued tails finish and write turns 214, 216 and two
    # facts under the uuid, AFTER the header is gone and the reverse merge
    # already declared done (reviewer D, run ident3k: 17:12:52.069 merge
    # logs success, 17:12:52.775 and 17:13:09.509 the queued tail's writes
    # land). Nothing is destroyed — a second reverse merge recovers them —
    # but R3's own success check (facts_added present) passes while the
    # copy is silently incomplete. This stays a probe-and-refuse (unlike
    # the destination fix below): src is read-only here, the read already
    # happened above before any lock could be taken, and the failure mode
    # is staleness (an omitted fact), not the destination's corruption
    # (an overwritten one) — a probe close to the read narrows the window
    # without changing merge_conversation's read-only contract on src.
    if memory.conv_lock(src_conv_id).locked():
        raise ValueError(
            f"conv_id {src_conv_id!r} has a memory write in flight (extraction "
            f"tail, archive, restore or dedup). Refusing rather than merging "
            f"FROM it while incomplete - the source's own queued writes would "
            f"land after this merge already reported success, stranding them "
            f"under the source instead of copying them across. Retry in a "
            f"moment."
        )

    # v3.1.9.3 (P11 / race-merge.md, hostile pass on v3.1.9.2). This USED to
    # be the same probe-and-refuse D18 gave import_conversation: `if
    # memory.conv_lock(dst_conv_id).locked(): raise ...`. That is not the
    # same fix here, and the gap was real: import_conversation is called
    # DIRECTLY from its async endpoint and runs to completion without
    # yielding, so nothing else can interleave between its probe and its
    # write — the probe alone is sufficient mutual exclusion. merge_conversation
    # is dispatched `await run_in_threadpool(portability.merge_conversation,
    # ...)` (main.py), so it runs on a THREADPOOL WORKER — a real OS thread —
    # and NOTHING here ever called `conv_lock(dst).acquire()`, so the lock's
    # state never reflected a merge in progress. Two merges into the same
    # destination each probed `.locked()`, each saw False, and each did an
    # unsynchronised `facts.load_facts -> _merge_fact_lists -> facts.save_facts`
    # of the SAME file. Adversarial measurement (tests/adversarial/
    # test_adv_race.py::test_two_merges_into_one_conversation_lose_facts,
    # 60 reps of two merges racing into one destination): every run lost
    # roughly half of the 40 facts both sides had just been told (HTTP 200,
    # `facts_added` counted) were added — whichever merge's save landed
    # first was overwritten by the second, which had read before the first
    # wrote. The same gap reaches merge-vs-import and merge-vs-/remember on
    # one destination (test_merge_and_import_into_one_destination,
    # test_merge_against_a_remember_holding_the_lock): those DO take
    # conv_lock properly, but only as an async context manager on the event
    # loop, which a probe from a worker thread cannot see close in time.
    #
    # The fix: actually HOLD conv_lock(dst_conv_id) for the read-modify-write
    # below, the way every other writer to this file already does — not
    # probe it. A plain `def` on a worker thread cannot `await` an
    # asyncio.Lock directly, but AnyIO gives exactly this bridge: a thread
    # started by `anyio.to_thread.run_sync` (what `run_in_threadpool` calls)
    # carries a "blocking portal" back to the loop that spawned it, and
    # `anyio.from_thread.run(coro)` runs `coro` ON THAT LOOP and blocks this
    # thread until it finishes.
    #
    # ONLY THE LOCK OPERATIONS cross that bridge — not the work (coordinator
    # review round 2). An earlier version of this fix ran `_merge_commit()`
    # itself — load_facts, _merge_fact_lists, save_facts, the exchanges loop
    # — inside the bridged coroutine, i.e. ON THE EVENT LOOP, the one thread
    # every other request in the process depends on. That defeats the whole
    # reason this function is dispatched through run_in_threadpool in the
    # first place: while a merge's file I/O ran, the loop could not service
    # ANY other coroutine — a concurrent GET /health would stall for as long
    # as the merge's own save took. Proven (and now guarded against) by
    # test_merge_does_not_block_the_event_loop_while_writing, which patches
    # save_facts to sleep and measures how long a concurrent no-op coroutine
    # takes to get its next turn while a merge is in flight.
    #
    # So the bridge now carries ONLY `_acquire_dst_lock_bounded` (acquire and
    # return) and `_release_dst_lock` (release). `_merge_commit()` runs
    # between them on THIS thread — the worker thread merge_conversation was
    # already running on — exactly as if no lock existed. `release()` also
    # goes through the portal (via `anyio.from_thread.run_sync`, AnyIO's
    # sync-callable counterpart to `run`): `asyncio.Lock.release()` wakes the
    # next waiter by resolving a Future that belongs to the loop, so it is no
    # more thread-safe to call directly from a worker thread than `acquire()`
    # is — see `merge_probe3.py` in this lane's scratchpad report for a
    # standalone confirmation of both halves of this bridge before it was
    # wired in here.
    #
    # BOUNDED, not unbounded (coordinator review round 2, second finding). A
    # summary rebuild can hold conv_lock for the WHOLE drain — 10-30 minutes,
    # the identity runbook's own estimate (see the source-side guard's
    # comment above) — with her extraction tails queued behind it.
    # `_acquire_dst_lock_bounded` waits at most `_MERGE_DST_LOCK_TIMEOUT_S`
    # (see that constant's own comment for the number and why) and, on
    # timeout, raises the ORIGINAL D18-style refusal instead of parking the
    # request — a merge queued behind a genuine multi-minute rebuild fails
    # fast with a clear error rather than hanging past any caller's or
    # proxy's timeout.
    #
    # `anyio.from_thread.run` raises `anyio.NoEventLoopError` when there is
    # no portal — every existing unit test in this file calls
    # merge_conversation() directly, with no threadpool involved at all, and
    # that has to keep working. Two sub-cases, told apart by whether a loop
    # is running on THIS thread:
    #   * no loop at all (a plain script/test call, the common case here) —
    #     nothing else can be interleaving through conv_lock on this thread
    #     either, so there is no event loop to protect from blocking: a
    #     fresh `asyncio.run` doing the bounded acquire, the commit, AND the
    #     release all in one place is exactly as safe as the bridge above.
    #   * a loop IS running on this thread (a test that calls
    #     merge_conversation() synchronously from inside a coroutine that
    #     ALREADY holds conv_lock(dst) itself, simulating "a write is in
    #     flight" — see test_merge_still_refuses_while_dest_has_a_write_in_
    #     flight): awaiting the same lock here would self-deadlock the one
    #     coroutine that is both the holder and the would-be acquirer, and
    #     `asyncio.run` cannot be nested inside a running loop either. This
    #     is exactly the shape D18's probe-and-refuse was built for, and
    #     nothing on this thread can be running concurrently with us here
    #     (we own it), so the old probe (instant, no timeout needed — it is
    #     a check, not a wait) is still the correct answer for it.
    def _dst_lock_busy_error() -> ValueError:
        # The ORIGINAL D18-style message, unchanged — used both when the
        # bounded wait below times out and in the no-portal/loop-already-
        # running fallback's instant probe, so a caller sees the same error
        # regardless of which of the two ways it was refused.
        return ValueError(
            f"conv_id {dst_conv_id!r} has a memory write in flight (extraction "
            f"tail, archive, restore or dedup). Refusing rather than merging "
            f"underneath it - that writer would overwrite the merged facts on "
            f"its next save. Retry in a moment."
        )

    def _merge_commit() -> None:
        """The critical section itself — everything that reads-and-then-
        writes dst. Deliberately lock-free and synchronous: every caller
        below already holds conv_lock(dst_conv_id) (or has established, by
        construction, that nothing else can be interleaving with it) before
        calling this and releases it after — nothing in this function's own
        body touches the lock, which is what lets it run on a worker thread
        rather than the event loop.
        """
        # Re-read rather than trusting the counters computed above: the tail
        # may have added facts between the pre-flight read and here. Re-run
        # the same fold against the fresh read rather than reusing the stale
        # preview — a collision the preview never saw (because the tail
        # added that key after the preview ran) still has to have its
        # pin/last_used folded, not just its "already present" status
        # re-checked.
        current = facts.load_facts(dst_conv_id)
        merged, actual_stats = _merge_fact_lists(
            current, src_facts,
            last_used_floor=_dst_last_used_floor(current) if refresh_last_used else None,
        )
        if merged != current:
            facts.save_facts(dst_conv_id, merged)
        result["facts_added"] = actual_stats["added"]
        result["facts_pin_or_recency_updated"] = actual_stats["updated"]
        # Re-measured against the set actually written (the preview above
        # was against the pre-lock, possibly-stale read) — same reasoning as
        # re-running _merge_fact_lists itself against `current`.
        _kept_actual, _evicted_actual = facts._lru_split(merged, facts._MAX_FACTS_TOKENS)
        result["facts_over_budget_after_merge"] = len(_evicted_actual)
        _dst_evicted, _merged_evicted = _evicted_by_origin(merged, len(current), _evicted_actual)
        result["dst_facts_evicted_after_merge"] = _dst_evicted
        result["merged_facts_evicted_after_merge"] = _merged_evicted

        added = 0
        for e in new_exchanges:
            try:
                if retrieval.import_indexed_exchange(
                    dst_conv_id, e.get("turn_index"), e.get("document", "")
                ):
                    added += 1
            except Exception as ex:
                logger.warning(
                    f"merge {src_conv_id}->{dst_conv_id}: exchange "
                    f"{e.get('turn_index')} failed to import: {ex}"
                )
        result["exchanges_added"] = added

    async def _acquire_dst_lock_bounded() -> None:
        """Runs ON THE LOOP via the portal — this, and only this, is the
        part of the fix that has to. Bounded per _MERGE_DST_LOCK_TIMEOUT_S;
        a timeout raises the ORIGINAL refusal rather than continuing to
        wait.
        """
        try:
            await asyncio.wait_for(
                memory.conv_lock(dst_conv_id).acquire(),
                timeout=_MERGE_DST_LOCK_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            raise _dst_lock_busy_error() from None

    def _release_dst_lock() -> None:
        """Also runs ON THE LOOP, via anyio.from_thread.run_sync — see the
        module comment above for why release(), like acquire(), is not
        thread-safe to call directly from the worker thread.
        """
        memory.conv_lock(dst_conv_id).release()

    try:
        anyio.from_thread.run(_acquire_dst_lock_bounded)
    except anyio.NoEventLoopError:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No loop on this thread at all — see the no-portal comment
            # above. Acquire (bounded, same as the real path), commit, and
            # release all within one fresh loop; nothing else can be
            # running concurrently with us on this thread regardless.
            async def _fallback_acquire_commit_release() -> None:
                lock = memory.conv_lock(dst_conv_id)
                try:
                    await asyncio.wait_for(
                        lock.acquire(), timeout=_MERGE_DST_LOCK_TIMEOUT_S
                    )
                except asyncio.TimeoutError:
                    raise _dst_lock_busy_error() from None
                try:
                    _merge_commit()
                finally:
                    lock.release()

            asyncio.run(_fallback_acquire_commit_release())
        else:
            # A loop is already running on this thread and we cannot await
            # into it without risking a self-deadlock or a nested
            # `asyncio.run` — fall back to the original D18 probe-and-refuse
            # for this one shape (see the comment above). An instant check,
            # not a wait, so no timeout applies here.
            if memory.conv_lock(dst_conv_id).locked():
                raise _dst_lock_busy_error()
            _merge_commit()
    else:
        # The lock is ours, acquired on the loop via the portal above. The
        # actual work happens HERE, on this worker thread — not the loop —
        # which is the entire point of this fix; release back through the
        # portal when done, success or not.
        try:
            _merge_commit()
        finally:
            anyio.from_thread.run_sync(_release_dst_lock)

    logger.info(
        f"merged conv {src_conv_id} into {dst_conv_id}: "
        f"+{result.get('facts_added', 0)} fact(s), "
        f"{result.get('facts_pin_or_recency_updated', 0)} existing fact(s) "
        f"pin/last_used updated, +{result.get('exchanges_added', 0)} "
        f"exchange(s); source left intact"
    )
    return result


def fork_conversation(
    src_conv_id: str, *, new_conv_id: str | None = None
) -> dict:
    """Clone src's full state into a new conv_id. Original is untouched.

    Use case: "I want to explore an alternative direction without
    losing the path I'm currently on." Fork at right-now: the new conv
    starts with the same facts, summary state, and indexed exchanges
    — the model has the same memory the moment after the next chat
    request arrives.

    Returns the new conv_id and copy counters.
    """
    if not new_conv_id:
        # Suffix the source id with a short unique tag so the fork is
        # discoverable in /admin/conversations alongside its parent.
        suffix = uuid.uuid4().hex[:6]
        new_conv_id = f"{src_conv_id}__fork_{suffix}"

    # v3.1.9 (hostile pass 3, F9). strict=True — same reasoning as
    # merge_conversation: an unreadable src facts/summary file must not
    # silently fork as an empty-facts tombstone (which also then blocks
    # needs_backfill from ever rebuilding it — see the finding). Mapped to
    # ImportError_ rather than a bare StoreUnreadable so the existing
    # main.py catch clause on admin_fork_conversation (`except
    # (portability.ImportError_, UnsafeConvId)`) already handles this as a
    # 400 with no endpoint change needed.
    try:
        bundle = export_conversation(src_conv_id, strict=True)
    except memory.StoreUnreadable as e:
        raise ImportError_(
            f"conv {src_conv_id}: could not read facts or summary state "
            f"({e}) — refusing to fork an unknown source rather than "
            f"produce an empty-facts copy"
        ) from e
    bundle["source_conv_id"] = src_conv_id
    result = import_conversation(bundle, target_conv_id=new_conv_id, overwrite=False)
    result["forked_from"] = src_conv_id
    return result
