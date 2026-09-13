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

import json
import logging
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

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
#
# v3.1 D8 — TWO LAYERS THE BUNDLE DOES NOT CARRY, and why they are carried
# here instead.
#
# export_conversation writes exactly three payloads: facts, summary_state,
# episodic. It does not carry the archive sidecar and it does not carry the
# persona. For an export that is a schema question; for a PRE-REMOVAL SNAPSHOT
# it is a correctness one, because both of those layers are things a
# destructive admin operation deletes:
#
#   - commands._wipe_all_layers clears the archive sidecar, and its own
#     comment records that "NOTHING in this codebase has ever deleted one"
#     before it did.
#   - main._clear_all_memory calls persona.clear_persona. INCIDENT_2026-08-24
#     D19 states the consequence in one line: "/forget can destroy a persona
#     that no export can back up and no import can restore."
#
# A snapshot that is missing the layers the operation removes is not an
# archive-before-removing; it is a partial one that reads as complete. So both
# are measured, carried, and verified on read-back like everything else.
#
# They go in the `quarantine` metadata block rather than into the bundle
# payload. That is deliberate and it is not laziness: adding payload keys means
# either a BUNDLE_VERSION bump — which _validate_bundle enforces by strict
# equality, so every previously written bundle and both HTTP endpoints stop
# working the moment it changes — or silently widening a documented schema.
# Under `quarantine`, _validate_bundle ignores the extra key, so a snapshot
# stays a valid v2.1 bundle that import_conversation restores with no new code,
# AND the two extra layers travel with it for an operator (or
# commands._handle_retire) to put back explicitly. Restoring them is a
# `facts.save_archive` and a `persona.save_persona`; the restore_hint says so.
#
# This does NOT close D19 for export/import generally — export_conversation is
# unchanged and a fork still loses the persona. It closes it for the one path
# whose entire purpose is to make a removal reversible.

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
    "episodic", "summary", "persona", "unverified_layers"}.

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
            "archive": len(archived_rows),
        },
        # The two layers the v2.1 bundle payload has no key for. Carried
        # verbatim so a restore is a copy, not a reconstruction.
        "archive": list(archived_rows),
        "persona": persona_record,
        "unverified_layers": list(unverified),
        "restore_hint": (
            "per-row: facts.restore_from_archive(conv_id) — preferred. "
            "whole-conversation: import_conversation(this file, "
            "target_conv_id=<conv>, overwrite=True) — discards anything "
            "learned since this file was written. import_conversation does "
            "NOT restore the two layers under quarantine.archive and "
            "quarantine.persona: put those back with "
            "facts.save_archive(conv_id, bundle['quarantine']['archive']) and "
            "persona.save_persona(conv_id, "
            "bundle['quarantine']['persona']['persona_text'])."
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
        # Same contradiction for the two layers carried in the metadata block.
        # Verified rather than trusted for exactly the reason the payload is:
        # the caller is about to delete these, and a snapshot that quietly
        # dropped them on serialization is worse than no snapshot, because the
        # caller proceeds.
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
        f"{n_back} fact(s), {len(back_q.get('archive') or [])} archived, "
        f"{len(back.get('episodic') or [])} episodic, "
        f"summary={'yes' if _has_summary_content(back.get('summary_state')) else 'no'}, "
        f"persona={'yes' if back_q.get('persona') else 'no'}"
        + (f", unverified: {'; '.join(unverified)}" if unverified else "")
    )

    return {
        "path": published,
        "facts": n_back,
        "archive": len(back_q.get("archive") or []),
        "episodic": len(back.get("episodic") or []),
        "summary": _has_summary_content(back.get("summary_state")),
        "persona": bool(back_q.get("persona")),
        "unverified_layers": unverified,
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

    # Restore facts wholesale (already-pruned by export, no further pruning).
    facts.save_facts(target, list(bundle.get("facts", [])))

    # Restore summary state wholesale.
    summarizer.save_state(target, dict(bundle.get("summary_state", {})))

    if (pre_existing or unverifiable) and overwrite:
        retrieval.forget_conversation(target)

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


def merge_conversation(
    src_conv_id: str, dst_conv_id: str, *, dry_run: bool = True
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
        recoverable if the result is wrong.

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
    """
    if not src_conv_id or not dst_conv_id:
        raise ValueError("both src_conv_id and dst_conv_id are required")
    if src_conv_id == dst_conv_id:
        raise ValueError("src and dst are the same conversation")

    src = export_conversation(src_conv_id)
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
    _preview_merged, fact_stats = _merge_fact_lists(
        dst_facts, src_facts, last_used_floor=_dst_last_used_floor(dst_facts)
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
    if dry_run:
        return result

    # Same mutual exclusion import_conversation uses, and for the same reason
    # (D18): conv_lock is an ASYNCIO lock and this is a plain def called
    # through run_in_threadpool, so it cannot await the lock - it can only
    # refuse to write underneath a holder. The hazard is concrete: the
    # extraction tail reads the fact list, parks on a vLLM call holding the
    # lock, and writes that pre-merge snapshot back when it returns. The
    # merged facts would vanish with no error anywhere.
    if memory.conv_lock(dst_conv_id).locked():
        raise ValueError(
            f"conv_id {dst_conv_id!r} has a memory write in flight (extraction "
            f"tail, archive, restore or dedup). Refusing rather than merging "
            f"underneath it - that writer would overwrite the merged facts on "
            f"its next save. Retry in a moment."
        )

    # Re-read rather than trusting the counters computed above: the tail may
    # have added facts between the pre-flight read and here. Re-run the same
    # fold against the fresh read rather than reusing the stale preview — a
    # collision the preview never saw (because the tail added that key after
    # the preview ran) still has to have its pin/last_used folded, not just
    # its "already present" status re-checked.
    current = facts.load_facts(dst_conv_id)
    merged, actual_stats = _merge_fact_lists(
        current, src_facts, last_used_floor=_dst_last_used_floor(current)
    )
    if merged != current:
        facts.save_facts(dst_conv_id, merged)
    result["facts_added"] = actual_stats["added"]
    result["facts_pin_or_recency_updated"] = actual_stats["updated"]
    # Re-measured against the set actually written (the preview above was
    # against the pre-lock, possibly-stale read) — same reasoning as
    # re-running _merge_fact_lists itself against `current`.
    _kept_actual, _evicted_actual = facts._lru_split(merged, facts._MAX_FACTS_TOKENS)
    result["facts_over_budget_after_merge"] = len(_evicted_actual)

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

    logger.info(
        f"merged conv {src_conv_id} into {dst_conv_id}: "
        f"+{result.get('facts_added', 0)} fact(s), "
        f"{result.get('facts_pin_or_recency_updated', 0)} existing fact(s) "
        f"pin/last_used updated, +{added} exchange(s); source left intact"
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

    bundle = export_conversation(src_conv_id)
    bundle["source_conv_id"] = src_conv_id
    result = import_conversation(bundle, target_conv_id=new_conv_id, overwrite=False)
    result["forked_from"] = src_conv_id
    return result
