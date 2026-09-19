"""
compactor.memory — V2.0 storage layer and conversation identification.

Phase 1 scope: just conv_id resolution + storage layout (mkdir + listing).
No reads/writes of memory contents yet — those land in Phase 2 (facts) and
Phase 3 (ChromaDB).

Resolution strategy (per V2_PLAN.md):
1. Read `X-Conversation-Id` HTTP header — set by the bundled OpenWebUI
   Pipeline filter from OpenWebUI's internal conversation primary key.
2. Fall back to sha256 of `system_prompt|||first_user_message[:512]` for
   clients that don't set the header (direct API users, third-party tools,
   or pods running without the Pipeline filter installed).

The header path is preferred because it's stable across edits to the
system prompt and immune to opening-fingerprint collisions. The hash
fallback works but is documented as the lower-quality path.
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger("compactor.memory")

# Storage root on the persistent volume. Subdirs: facts/, summaries/,
# chromadb/. Configurable for tests; default matches V2_PLAN.md.
STORAGE_ROOT = Path(
    os.environ.get("COMPACTOR_STORAGE_ROOT", "/data/openwebui/compactor")
)

# Conv-id sanitization: filesystem-safe charset, length-capped to prevent
# pathological filenames. Allows alphanumerics, dash, underscore — covers
# UUIDs from OpenWebUI and sha256 hex from the fallback path.
_CONV_ID_ALLOWED = re.compile(r"[^A-Za-z0-9_\-]")
_CONV_ID_MAX_LEN = 64

# Headers come in lowercased from Starlette/FastAPI, but we check both
# casings for robustness against direct callers.
_HEADER_NAMES = ("x-conversation-id", "X-Conversation-Id")


def _sanitize(raw: str) -> str:
    """Strip anything that isn't filename-safe and length-cap."""
    if not raw:
        return ""
    cleaned = _CONV_ID_ALLOWED.sub("", raw.strip())
    return cleaned[:_CONV_ID_MAX_LEN]


def _message_text_for_hash(m: dict) -> str:
    """Extract plain text from a message for the fingerprint hash.

    Multimodal content (list of content parts) collapses to its text
    portions only — matches main.py's _message_text behavior so the hash
    is stable across multimodal/text-only client variants.
    """
    content = m.get("content") or ""
    if isinstance(content, list):
        return " ".join(c.get("text", "") for c in content if isinstance(c, dict))
    return str(content)


def _fingerprint_hash(messages: list[dict]) -> str:
    """sha256(system|||first_user_message[:512])[:16] — stable across the
    life of one conversation, very unlikely to collide between distinct
    conversations.
    """
    system = next(
        (_message_text_for_hash(m) for m in messages if m.get("role") == "system"),
        "",
    )
    first_user = next(
        (_message_text_for_hash(m) for m in messages if m.get("role") == "user"),
        "",
    )
    fingerprint = f"{system}|||{first_user[:512]}"
    return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:16]


def resolve_conv_id(
    headers: dict,
    messages: list[dict],
    body: dict | None = None,
) -> tuple[str, str]:
    """Determine the conversation ID for an incoming request.

    Resolution order:
    1. `X-Conversation-Id` HTTP header — set by direct API callers or by an
       OpenWebUI Pipeline (the separate Pipelines server, not in-process
       Functions).
    2. `body["metadata"]["chat_id"]` or `body["metadata"]["conversation_id"]`
       — set by the bundled OpenWebUI Function filter (in-process Functions
       can't easily manipulate HTTP headers but can mutate the request
       body, so this is the path most users will exercise).
    3. SHA256 fingerprint of `system|||first_user[:512]` — fallback for
       clients that set neither.

    Returns (conv_id, source) where source describes which path resolved.
    """
    # 1. HTTP header
    for name in _HEADER_NAMES:
        raw = headers.get(name)
        if raw:
            sanitized = _sanitize(raw)
            if sanitized:
                return sanitized, "header"
            logger.warning(
                f"received {name} header but value sanitized to empty: {raw!r}"
            )

    # 2. Body metadata (OpenWebUI Function filter path)
    if body is not None:
        md = body.get("metadata") if isinstance(body.get("metadata"), dict) else None
        if md:
            for key in ("chat_id", "conversation_id"):
                raw = md.get(key)
                if raw:
                    sanitized = _sanitize(str(raw))
                    if sanitized:
                        return sanitized, f"body_metadata.{key}"

    # 3. Hash fallback
    return _fingerprint_hash(messages), "hash"


# ---------------------------------------------------------------------------
# Storage layout
# ---------------------------------------------------------------------------

class UnsafeConvId(ValueError):
    """A conv_id that would place a file outside STORAGE_ROOT."""


def _safe_path(subdir: str, conv_id: str, suffix: str) -> Path:
    """Build a storage path and REFUSE to leave STORAGE_ROOT.

    v3.1.8, and this one was a live arbitrary-file-write.

    _sanitize strips a conv_id to [A-Za-z0-9_-], but it is called in
    exactly one place: resolve_conv_id, on the CHAT path. The admin
    endpoints take a conversation id straight out of the request BODY —
    import's `target_conv_id`, fork's `new_conv_id` — and hand it to these
    builders unsanitized. portability.py carried a comment asserting
    "conv_id is already sanitized by memory._sanitize", which was simply
    not true of that route.

    Reproduced against a clean stack: POST /admin/conversations/import with
    target_conv_id "../../../../../../tmp/CLAUDE_PWNED" returned HTTP 200,
    echoed the traversal back in its own response, and wrote
    /tmp/CLAUDE_PWNED.json — outside STORAGE_ROOT, as root.

    The nastier variant needs no attacker at all. Because ".." climbs out
    of the per-layer subdirectory, facts_path and summary_path collide on
    one file, so an import targeting "../facts/<someone-else>" reports
    success while emptying a bystander conversation's memory. That is
    silent cross-conversation data loss reachable by a typo.

    THE GUARD LIVES HERE, not at the endpoints, deliberately. Sanitizing
    at each admin route would fix the two known holes and leave the sixth
    caller to reintroduce it — the fix-one-site-miss-the-sibling defect
    this codebase has paid for more than a dozen times. Every path into
    the store is built by one of the five functions below, so this is the
    chokepoint that cannot be walked past.

    Raises UnsafeConvId rather than sanitizing silently: a caller that
    passes a traversal is confused about what it is doing, and quietly
    rewriting its target would hide that. Endpoints turn this into a 400.
    """
    # TWO checks, and the first is the one that matters.
    #
    # A resolved-path check alone is NOT enough, and the first version of
    # this function proved it: "../facts/victim" passed facts_path,
    # because it resolves back INTO the facts directory. Inside the root,
    # inside the right subdirectory, and pointed at somebody else's
    # conversation — which is the silent cross-conversation destruction
    # this guard exists to stop. "Somewhere legal" is the wrong question;
    # the right one is whether the conv_id is a NAME at all.
    #
    # So the id is validated against the SAME character class _sanitize
    # enforces on the chat path, shared rather than restated: a conv_id
    # the chat path would have stripped is not one the admin path may
    # keep. That also settles ".." and "." (which produced the perfectly
    # legal, perfectly wrong "/data/compactor/facts/...json") and any id
    # long enough to hit a filesystem name limit.
    if not conv_id:
        raise UnsafeConvId("conv_id is empty")
    if chr(0) in conv_id:
        raise UnsafeConvId(f"conv_id contains a NUL byte: {conv_id!r}")
    if _CONV_ID_ALLOWED.search(conv_id):
        raise UnsafeConvId(
            f"conv_id must match [A-Za-z0-9_-] and does not: {conv_id!r}"
        )
    if len(conv_id) > _CONV_ID_MAX_LEN:
        raise UnsafeConvId(
            f"conv_id is longer than {_CONV_ID_MAX_LEN} characters: "
            f"{len(conv_id)}"
        )

    # Second line of defence. The check above already makes traversal
    # unrepresentable, so this can only fire if the character class is
    # ever widened — which is exactly when a reviewer would want it to.
    root = STORAGE_ROOT.resolve()
    candidate = root / subdir / f"{conv_id}{suffix}"
    try:
        resolved = candidate.resolve()
    except (OSError, ValueError) as e:
        raise UnsafeConvId(f"conv_id is not a usable path: {conv_id!r}") from e
    if resolved.parent != root / subdir:
        raise UnsafeConvId(
            f"conv_id would place a file outside {subdir}/: "
            f"{conv_id!r} -> {resolved}"
        )
    return candidate

def facts_path(conv_id: str) -> Path:
    return _safe_path("facts", conv_id, ".json")


def facts_archive_path(conv_id: str) -> Path:
    """V2.1 Phase 7 Step 2: cold-storage sidecar for archived facts.
    Sits next to the active facts file; list_known_conv_ids skips it
    (its stem has a dot, so the sidecar filter excludes it)."""
    return _safe_path("facts", conv_id, ".archive.json")


def persona_path(conv_id: str) -> Path:
    """V2.1 Phase 8: persona text storage. Separate subdir so conversations
    that have facts but no persona don't pollute the persona listing,
    and vice versa.
    """
    return _safe_path("personas", conv_id, ".json")


def summary_path(conv_id: str) -> Path:
    return _safe_path("summaries", conv_id, ".json")


def summary_archive_path(conv_id: str) -> Path:
    """Cold storage for L2 chapters consumed by an L3 refresh.

    Same contract as facts_archive_path, and it exists for the same reason:
    this project's stated invariant is that eviction MOVES memory to a
    sidecar and never unlinks it (see facts.py's module docstring). The L3
    rollup was the first path on this branch to delete memory outright -
    it cleared state["l2"] with no copy anywhere - which also meant the
    recursively re-paraphrased L3 had no source left to be regenerated from.
    Its stem carries a dot, so list_known_conv_ids skips it exactly as it
    skips the facts sidecar."""
    return _safe_path("summaries", conv_id, ".archive.json")


def chromadb_path() -> Path:
    """Return the ChromaDB persist directory (Phase 3 will populate it)."""
    return STORAGE_ROOT / "chromadb"


def ensure_storage_layout() -> None:
    """Create the storage subdirectories on the persistent volume.

    Idempotent — safe to call on every request, though main.py calls it
    once at startup. Required because the volume is empty on first attach.
    """
    for sub in ("facts", "summaries", "chromadb", "personas"):
        (STORAGE_ROOT / sub).mkdir(parents=True, exist_ok=True)


def list_known_conv_ids() -> list[str]:
    """For the /admin/conversations endpoint. Returns every conv_id that
    has *either* a facts file or a summary file (or eventually a ChromaDB
    collection). Sorted for stable output.

    Skips sidecar files like `<conv>.backfill.json` (Phase 2 lazy
    backfill state) so they don't show up as fake conversations.
    """
    ids: set[str] = set()
    for sub in ("facts", "summaries"):
        d = STORAGE_ROOT / sub
        if d.exists():
            for f in d.glob("*.json"):
                # Skip files whose name has a second extension (sidecars).
                # facts/<id>.json    → stem="<id>"        → keep
                # facts/<id>.backfill.json → stem="<id>.backfill" → skip
                if "." in f.stem:
                    continue
                ids.add(f.stem)
    return sorted(ids)


def storage_root() -> Path:
    """Expose STORAGE_ROOT as a function for modules that need to compute
    sidecar paths (e.g. backfill state). Stays a function so tests that
    set COMPACTOR_STORAGE_ROOT mid-test can observe the change.
    """
    return STORAGE_ROOT


# ---------------------------------------------------------------------------
# Foundational I/O primitives (used by facts.py, backfill.py, summarizer.py)
# ---------------------------------------------------------------------------

def atomic_write_json(path: Path, data: Any) -> None:
    """Crash-safe JSON write: serialize to a NamedTemporaryFile in the same
    directory, fsync, then os.replace (atomic on POSIX).

    Why this matters: a torn write (crash mid-write) leaves the destination
    file half-written. Subsequent reads see invalid JSON, the model loses
    its memory of that conversation, and we have no good way to recover.
    The temp+rename pattern guarantees readers see either the old contents
    or the new contents, never a torn state.

    Orphan *.tmp files left by a crash mid-write are ignored by
    list_known_conv_ids() since it globs for *.json specifically.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=path.name + ".",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        # fsync the PARENT DIRECTORY so the rename itself is durable, not just
        # the file contents. Without this the replace can still be lost on a
        # crash/host failure — and this state lives on a distributed network
        # volume (MooseFS), where rename/fsync guarantees are weaker than local
        # POSIX. Best-effort: a filesystem that refuses O_DIRECTORY fsync must
        # not fail the write we just completed successfully.
        try:
            dir_fd = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError as e:
            # The write itself succeeded; only the durability barrier failed.
            # DEBUG, not WARNING — on MooseFS this is expected often enough
            # that warning here would train people to ignore warnings.
            logger.debug(f"directory fsync skipped for {path.parent}: {e}")
    except Exception:
        # Best-effort cleanup of orphan temp file; don't shadow the
        # original exception if cleanup also fails.
        try:
            os.unlink(tmp_path)
        except OSError as e:
            logger.debug(f"orphan temp file left behind at {tmp_path}: {e}")
        raise


class StoreUnreadable(Exception):
    """A memory file is present on disk but its contents could not be read.

    The distinction this carries is the entire point of it existing. An
    ABSENT file is an empty store — correct for a conversation that has
    never written one. An unreadable file means the real contents are still
    on disk and we do not know what they are. Before v3.1 read_json
    answered both with the same default, and five callers read-modify-WROTE
    that default back: one transient error replaced a 105-fact store with
    the one or two facts from the current exchange, atomically, permanently,
    behind a single WARNING line (v3.1 F1).

    So every read-modify-write path raises this rather than guessing, and
    aborts its write. A skipped write loses one exchange's memory; a
    completed one loses the conversation's.
    """

    def __init__(self, path: Path, cause: Exception) -> None:
        super().__init__(f"{path}: {type(cause).__name__}: {cause}")
        self.path = Path(path)
        self.cause = cause


def read_json_strict(
    path: Path, default: Any = None, *, expect: type | tuple[type, ...] | None = None
) -> Any:
    """Read JSON, raising StoreUnreadable rather than inventing a value.

    Returns `default` for an absent file — that case is by design and must
    stay that way, or every new conversation breaks on its first turn.
    Raises StoreUnreadable for the two live failure modes: a corrupt or
    truncated file (JSONDecodeError, what a torn write leaves behind) and
    an OSError from open/read (MooseFS's characteristic
    metadata-fine/chunk-unavailable shape).

    Note that path.is_file() is NOT a third failure mode to defend against:
    pathlib swallows only ENOENT/ENOTDIR/EBADF/ELOOP and re-raises
    everything else, so an EIO on the stat already reaches the caller
    (REMEDIATION.md §1.4 — both v3.1 reviews got this wrong).

    `expect` is the SHAPE half of the same contract, and it exists because
    the shape half was missing (v3.1.8, adversarial state sweep).

    A file that parses but holds the wrong THING — a bare list where a dict
    belongs, `null`, a number — used to come back to callers that wrote
    `data.get(...) if isinstance(data, dict) else []`, i.e. as EMPTY. Not
    unreadable: empty. Nothing raised, `stats.unreadable` did not move, and
    the next turn's save wrote a fresh empty store OVER IT — measured, with
    a pinned fact, gone.

    That is the v3.1 F1a defect exactly, one branch over: F1a was that a
    corrupt file returned [] and callers wrote back over the real facts, and
    the fix taught this loader to raise on a PARSE failure while leaving the
    wrong-shape case returning empty. So the difference between 'recoverable'
    and 'destroyed' was whether the damage happened to break the JSON parser.

    The check lives HERE rather than at the eight call sites for the reason
    this codebase keeps relearning: a rule applied at some call sites and
    missed at one is how the sibling defect survives. An absent file still
    returns `default` untouched — a new conversation must not raise on its
    first turn.

    This is the loader behind facts, the archive sidecar, summary state and
    personas. Callers on the request path already treat a raising load as
    "inject nothing this turn" and keep serving.
    """
    if not path.is_file():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    # UnicodeDecodeError IS NOT A JSONDecodeError, and that is the whole gap
    # (v3.1.9). Both descend from ValueError, but json.load DECODES the bytes
    # before it parses them, so a file holding invalid UTF-8 — a write torn
    # mid-multibyte, or the volume that already dropped I/O mid-transaction on
    # 2026-08-31 — raised straight past this handler, unwrapped.
    #
    # read_json catches only StoreUnreadable, so the best-effort path missed it
    # too, and v3.1.8 added a REQUEST-path consumer (_is_repeat_task_traffic)
    # behind a deliberately narrow `except (OSError, StoreUnreadable)`. One bad
    # byte therefore escaped out of _run_memory_tail inside event_stream's
    # finally — an ASGI exception on a request the user had already seen
    # succeed — and stats.unreadable never counted it, so v3.1.8's own
    # "unreadable memory on disk" reason stayed silent about the one file that
    # actually was.
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
        raise StoreUnreadable(path, e) from e
    if expect is not None and not isinstance(data, expect):
        # A present file holding the wrong thing is UNREADABLE, not empty.
        # Returning it (or an empty stand-in) is what let the next save
        # write over real memory.
        raise StoreUnreadable(
            path,
            TypeError(
                f"expected {getattr(expect, '__name__', expect)}, "
                f"found {type(data).__name__}"
            ),
        )
    return data


def read_json(path: Path, default: Any = None) -> Any:
    """Convenience read with default-on-missing. Returns the default if
    the file doesn't exist OR if it's corrupted (logs warning on corrupt).

    Best-effort only. If you are going to write back what you read, you
    want read_json_strict — this function cannot tell you which of the two
    it just handed you (v3.1 F1).
    """
    try:
        return read_json_strict(path, default)
    except StoreUnreadable as e:
        logger.warning(f"corrupted or unreadable {e}; returning default")
        return default


# Per-conv asyncio locks for serializing concurrent writers (e.g., the
# post-response fact-extraction tail vs. an active backfill on the same
# conv). Created lazily on first request. Lives for the lifetime of the
# process — fine because each conv_id's lock is tiny and the count is
# bounded by user-visible conversation count.
_conv_locks: dict[str, asyncio.Lock] = {}


def conv_lock(conv_id: str) -> asyncio.Lock:
    """Get-or-create the asyncio.Lock for this conv_id.

    Safe under single-threaded asyncio (default uvicorn shape) — dict
    operations are atomic under CPython's GIL and we're not in a thread
    pool here. If we ever go multi-threaded, this needs a meta-lock.
    """
    if conv_id not in _conv_locks:
        _conv_locks[conv_id] = asyncio.Lock()
    return _conv_locks[conv_id]


# ---------------------------------------------------------------------------
# Per-conv wipe generation (v3.1.9.4, R1 / P15-5 follow-up)
# ---------------------------------------------------------------------------
#
# A counter, not a boolean and not a timestamp. Bumped by EVERY path that
# deletes a conversation's memory (main._clear_all_memory — the single
# chokepoint /forget, DELETE /admin/conversations/<id>/facts, and the
# self-test cleanup all call through — and commands._handle_retire's apply
# step, which empties the source conversation the same way). A background
# tail (main._async_tail / _facts_tail / _rollup_hierarchy) captures the
# CURRENT value of this counter at the moment it is SUBMITTED to the pool —
# not when it starts running, because a tail parked on the pool's
# concurrency semaphore has executed zero lines of its own body yet — and
# carries that captured value with it. Every site where the tail is about
# to WRITE re-reads this counter, under the SAME conv_lock a wipe path also
# holds while it bumps and deletes, and discards the write if the two
# disagree.
#
# WHY THIS IS THE AIRTIGHT FIX (P15-5 / R1), where draining first (v3.1.9.4
# B3) is not. bgwork.pool.drain now WAITS for outstanding tails instead of
# cancelling them on timeout (see bgwork.BackgroundPool.drain) — the right
# call for every OTHER conversation's tail, which must not be killed just
# because ONE conversation typed /forget. But it reopened a narrower hole
# for the SAME conversation: a tail submitted before /forget arrived, still
# parked on the pool semaphore (or mid an extraction call) when
# commands._settle_background_work's drain times out, is now left running
# — and it eventually acquires conv_lock and writes, with nothing left to
# stop it, AFTER the wipe already ran. No timeout, no retry count, and no
# amount of draining harder closes that: the tail could resume after the
# process is otherwise fully idle. The generation check does not depend on
# timing at all — a write is either from before the wipe (and survives,
# correctly: that memory existed and the user had not yet asked to forget
# it) or from at-or-after it (and never reaches disk), no matter how many
# seconds, minutes, or pool-scheduling quirks sit in between.
#
# WHY IN-PROCESS IS ENOUGH (does not need to survive a restart). A restart
# kills every in-flight tail along with it — bgwork.BackgroundPool holds
# live asyncio.Task objects and nothing persists a queue of pending
# background work anywhere in this codebase — so a tail that could still
# land AFTER a wipe by outliving it can only do so within the same process
# lifetime the wipe itself ran in. A generation counter that resets to 0 on
# every restart is exactly as durable as it needs to be: the only tails it
# must protect against are ones that could not already have died with the
# process. Persisting it would buy nothing and cost a disk write on every
# wipe, on the request path of a command a user types expecting an
# immediate answer.
_wipe_generations: dict[str, int] = {}


def current_wipe_generation(conv_id: str) -> int:
    """The conversation's current wipe generation. 0 if it has never been
    wiped in this process's lifetime. A tail captures this value at
    submission time; a write site compares its captured value against a
    fresh call to this function, taken under conv_lock, immediately before
    it would persist anything, and discards the write on a mismatch."""
    return _wipe_generations.get(conv_id, 0)


def bump_wipe_generation(conv_id: str) -> int:
    """Call once from inside the SAME conv_lock(conv_id) critical section a
    wipe path uses for its deletes, before any of them run. Returns the new
    generation (most callers ignore it; returned for logging/tests).

    Must be called under conv_lock, and before the deletes it guards: with
    the bump inside the same locked section as the deletes, a concurrent
    tail either fully completes before this section starts (and is then
    cleaned up by these deletes, because it ran before them — the correct,
    unsurprising case: memory written before /forget is what /forget just
    removed) or cannot acquire the lock until after both the bump and the
    deletes have finished (and then discards its own write on the
    mismatch). There is no window in which a tail can observe the bump
    without the deletes that follow it also already being underway or
    complete, because nothing else may hold conv_lock(conv_id) while this
    section does.
    """
    g = _wipe_generations.get(conv_id, 0) + 1
    _wipe_generations[conv_id] = g
    return g


# ---------------------------------------------------------------------------
# Per-conv inventory (used by /admin/conversations/<id>)
# ---------------------------------------------------------------------------

def storage_summary(conv_id: str) -> dict:
    """Per-conv inventory: which files exist + their sizes. Useful for
    /admin/conversations/<id> debugging. Phase 1 just reports presence;
    Phase 2/3/4 will add content shape (fact count, message count, etc.)
    """
    fp = facts_path(conv_id)
    sp = summary_path(conv_id)
    return {
        "conv_id": conv_id,
        "facts": {
            "exists": fp.is_file(),
            "size_bytes": fp.stat().st_size if fp.is_file() else 0,
        },
        "summary": {
            "exists": sp.is_file(),
            "size_bytes": sp.stat().st_size if sp.is_file() else 0,
        },
    }
