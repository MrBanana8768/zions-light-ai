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

def facts_path(conv_id: str) -> Path:
    return STORAGE_ROOT / "facts" / f"{conv_id}.json"


def summary_path(conv_id: str) -> Path:
    return STORAGE_ROOT / "summaries" / f"{conv_id}.json"


def chromadb_path() -> Path:
    """Return the ChromaDB persist directory (Phase 3 will populate it)."""
    return STORAGE_ROOT / "chromadb"


def ensure_storage_layout() -> None:
    """Create the storage subdirectories on the persistent volume.

    Idempotent — safe to call on every request, though main.py calls it
    once at startup. Required because the volume is empty on first attach.
    """
    for sub in ("facts", "summaries", "chromadb"):
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
    except Exception:
        # Best-effort cleanup of orphan temp file; don't shadow the
        # original exception if cleanup also fails.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def read_json(path: Path, default: Any = None) -> Any:
    """Convenience read with default-on-missing. Returns the default if
    the file doesn't exist OR if it's corrupted (logs warning on corrupt).
    """
    if not path.is_file():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"corrupted or unreadable {path}: {e}; returning default")
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
