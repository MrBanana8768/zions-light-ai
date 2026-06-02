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
import time
import uuid
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

    # Pre-flight: detect existing state to honor overwrite=False.
    pre_existing = (
        len(facts.load_facts(target)) > 0
        or retrieval.conversation_doc_count(target) > 0
        or bool(summarizer.load_state(target).get("l1"))
    )
    if pre_existing and not overwrite:
        raise ImportError_(
            f"target conv_id {target!r} has existing state; "
            f"pass overwrite=true to replace"
        )

    # If overwriting, clear first — guarantees we don't end up with a
    # mix of old + new facts that confuses retrieval.
    if pre_existing and overwrite:
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
        "overwrote_existing": bool(pre_existing and overwrite),
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
