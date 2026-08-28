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

import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import facts
import memory
import retrieval
import summarizer

logger = logging.getLogger("compactor.portability")

BUNDLE_VERSION = "v2.1"


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_conversation(conv_id: str) -> dict:
    """Snapshot one conv's full V2 state as a single JSON-serializable dict.

    Best-effort per layer: a failure in one layer doesn't poison the
    bundle — it just gets an empty value. The bundle always has every
    expected key so import logic doesn't need defensive .get() calls.
    """
    try:
        loaded_facts = facts.load_facts(conv_id)
    except Exception as e:
        logger.warning(f"conv={conv_id}: export facts failed: {e}")
        loaded_facts = []

    try:
        summary_state = summarizer.load_state(conv_id)
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

# Filename-safe by construction: conv_id is already sanitized by
# memory._sanitize to [A-Za-z0-9_-], and the stamp adds only digits, "T" and
# "Z".
QUARANTINE_SUBDIR = "quarantine"


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
    something removes part of it. Returns {"path", "facts", "episodic",
    "summary", "unverified_layers"}.

    Raises QuarantineError if the snapshot cannot be proven to hold at least
    what the store held a moment ago. Raises memory.StoreUnreadable if the
    facts file is there and cannot be read — an operation that is about to
    rewrite that file must not proceed on a guess (F1).

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
    """
    # Measured BEFORE the export, and strictly: this is the expectation the
    # verify step tries to contradict, so it cannot come from the same
    # best-effort reads it is checking. StoreUnreadable propagates on purpose.
    expected_facts = len(facts.load_facts(conv_id))

    unverified: list[str] = []
    expected_episodic = retrieval.conversation_doc_count(conv_id)
    if expected_episodic is None:
        # None is "could not tell", never zero (F61). The episodic layer is not
        # what a facts cleanup modifies, so this is recorded rather than fatal
        # — but it is recorded, because a snapshot with an unverified layer is
        # not the same object as a snapshot with a verified one.
        unverified.append("episodic (vector store unavailable)")
    try:
        summarizer.load_state(conv_id)
    except memory.StoreUnreadable:
        unverified.append("summaries (unreadable)")
    except Exception:  # pragma: no cover - load_state's own best-effort paths
        unverified.append("summaries (unreadable)")

    bundle = export_conversation(conv_id)
    bundle["quarantine"] = {
        "reason": reason,
        "written_at": int(time.time()),
        "expected": {
            "facts": expected_facts,
            "episodic": expected_episodic,
        },
        "unverified_layers": list(unverified),
        "restore_hint": (
            "per-row: facts.restore_from_archive(conv_id) — preferred. "
            "whole-conversation: import_conversation(this file, "
            "target_conv_id=<conv>, overwrite=True) — discards anything "
            "learned since this file was written."
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

    quarantine_dir().mkdir(parents=True, exist_ok=True)
    published = _quarantine_path(conv_id)
    partial = published.with_name(published.name + ".partial")

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

        # Publish. Same rename-into-place backup.py uses: the file appears
        # under its real name only once it has been proven readable.
        os.replace(partial, published)
    except Exception:
        try:
            partial.unlink()
        except OSError:
            pass
        raise

    # Counts only. Never fact text, never conversation content — this log goes
    # to an operator's terminal and the store holds real personal memory.
    logger.info(
        f"conv={conv_id}: quarantine snapshot published ({reason}): "
        f"{n_back} fact(s), {len(back.get('episodic') or [])} episodic, "
        f"summary={'yes' if back.get('summary_state') else 'no'}"
        + (f", unverified: {'; '.join(unverified)}" if unverified else "")
    )

    return {
        "path": published,
        "facts": n_back,
        "episodic": len(back.get("episodic") or []),
        "summary": bool(back.get("summary_state")),
        "unverified_layers": unverified,
    }


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
    _validate_bundle(bundle)

    target = (target_conv_id or bundle.get("source_conv_id") or "").strip()
    if not target:
        raise ImportError_("no target_conv_id provided and bundle has no source_conv_id")

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

    # If overwriting, clear first — guarantees we don't end up with a
    # mix of old + new facts that confuses retrieval. `unverifiable` counts as
    # "might be occupied": skipping the clear because we could not PROVE state
    # exists is how stale episodic rows survive an overwrite and how
    # overwrote_existing comes to under-report a real replacement.
    if (pre_existing or unverifiable) and overwrite:
        facts.save_facts(target, [])
        retrieval.forget_conversation(target)
        # Summary state is overwritten wholesale by save_state, no clear needed.

    # Restore facts wholesale (already-pruned by export, no further pruning).
    facts.save_facts(target, list(bundle.get("facts", [])))

    # Restore summary state wholesale.
    summarizer.save_state(target, dict(bundle.get("summary_state", {})))

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

    bundle = export_conversation(src_conv_id)
    bundle["source_conv_id"] = src_conv_id
    result = import_conversation(bundle, target_conv_id=new_conv_id, overwrite=False)
    result["forked_from"] = src_conv_id
    return result
