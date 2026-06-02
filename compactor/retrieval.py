"""
compactor.retrieval — Episodic memory via embeddings + ChromaDB (V2.0 Phase 3).

The "episodic" layer of the V2.0 memory architecture: every user/assistant
exchange is embedded and stored in a vector index. On each request, the
latest user message is embedded and used to retrieve the top-K most
semantically similar past exchanges, which get injected into context. This
gives the model *exact text recall* of relevant prior moments — even ones
hundreds of turns back that summarization would have blurred away.

Design:
- Embeddings: BAAI/bge-small-en-v1.5 via fastembed (ONNX runtime, CPU).
  Deliberately torch-free so the compactor venv stays decoupled from
  vLLM's torch/transformers pins (the dependency-isolation lesson from
  the V1.9.x dependency saga).
- Vector store: ChromaDB PersistentClient at
  /data/openwebui/compactor/chromadb/. ONE collection; conversations are
  isolated via a `conv_id` metadata filter (cleaner deletes + better
  scaling than a collection-per-conversation).
- Everything degrades to a safe no-op. If fastembed/chromadb can't import
  or init (disabled, missing deps, corrupt store), retrieval returns []
  and indexing silently skips — chat is NEVER broken by a memory failure.

Heavy objects (embedding model, chroma client) are lazy singletons:
initialized on first use, reused forever.
"""

import logging
import os
import threading
from typing import Any

logger = logging.getLogger("compactor.retrieval")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RETRIEVAL_ENABLED = (
    os.environ.get("COMPACTOR_RAG_ENABLED", "true").lower() != "false"
)
EMBEDDING_MODEL = os.environ.get(
    "COMPACTOR_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"
)
RAG_TOP_K = int(os.environ.get("COMPACTOR_RAG_TOP_K", "5") or 5)

# fastembed caches the ONNX model here. Baked into the image at build time
# (NOT on /data) since the embedding model is static, not per-deployment.
FASTEMBED_CACHE = os.environ.get("FASTEMBED_CACHE_PATH", "/opt/embeddings")

# ChromaDB persistence — on the /data volume so the index survives restarts.
from memory import STORAGE_ROOT  # noqa: E402

CHROMA_PATH = str(STORAGE_ROOT / "chromadb")
COLLECTION_NAME = "conversation_turns"


# ---------------------------------------------------------------------------
# Lazy singletons (thread-safe init)
# ---------------------------------------------------------------------------

_init_lock = threading.Lock()
_embedder = None          # fastembed.TextEmbedding instance
_chroma_collection = None  # chromadb collection
_available: bool | None = None  # None=untried, True/False=resolved


def _try_init() -> bool:
    """Initialize the embedding model + chroma collection once. Returns
    True if retrieval is usable, False if it should be treated as disabled.
    Idempotent and thread-safe.
    """
    global _embedder, _chroma_collection, _available
    if _available is not None:
        return _available
    with _init_lock:
        if _available is not None:  # double-checked
            return _available
        if not RETRIEVAL_ENABLED:
            logger.info("retrieval disabled via COMPACTOR_RAG_ENABLED=false")
            _available = False
            return False
        try:
            from fastembed import TextEmbedding
            import chromadb

            _embedder = TextEmbedding(
                model_name=EMBEDDING_MODEL, cache_dir=FASTEMBED_CACHE
            )
            client = chromadb.PersistentClient(path=CHROMA_PATH)
            _chroma_collection = client.get_or_create_collection(
                name=COLLECTION_NAME,
                # cosine matches bge-small's training objective better than
                # the chroma default (l2).
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                f"retrieval ready: model={EMBEDDING_MODEL} store={CHROMA_PATH}"
            )
            _available = True
        except Exception as e:
            logger.warning(
                f"retrieval init failed ({e}); RAG disabled, chat unaffected"
            )
            _available = False
    return _available


def is_available() -> bool:
    """Public probe — used by /health/full and selftest (Phase 2.2)."""
    return _try_init()


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def _embed(texts: list[str]) -> list[list[float]] | None:
    """Embed a batch of strings → list of vectors. None on failure."""
    if not _try_init() or _embedder is None:
        return None
    try:
        # fastembed.embed returns a generator of numpy arrays.
        return [vec.tolist() for vec in _embedder.embed(texts)]
    except Exception as e:
        logger.warning(f"embedding failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Document shape
# ---------------------------------------------------------------------------

def _exchange_doc(user_text: str, assistant_text: str) -> str:
    """Canonical stored/embedded representation of one exchange."""
    return f"[user]: {user_text}\n[assistant]: {assistant_text}"


def _doc_id(conv_id: str, turn_index: int) -> str:
    """Stable, unique id per (conv, turn). Re-indexing the same turn
    overwrites rather than duplicating (chroma upserts on matching id via
    add? No — add raises on duplicate id; we use upsert()).
    """
    return f"{conv_id}::{turn_index}"


# ---------------------------------------------------------------------------
# Public operations — all degrade to no-op / [] on any failure
# ---------------------------------------------------------------------------

def index_exchange(
    conv_id: str, turn_index: int, user_text: str, assistant_text: str
) -> bool:
    """Embed one exchange and upsert it into the vector store. Returns True
    on success, False if skipped/failed. Never raises.
    """
    if not _try_init() or _chroma_collection is None:
        return False
    if not user_text or not assistant_text:
        return False
    doc = _exchange_doc(user_text, assistant_text)
    vecs = _embed([doc])
    if not vecs:
        return False
    try:
        _chroma_collection.upsert(
            ids=[_doc_id(conv_id, turn_index)],
            embeddings=vecs,
            documents=[doc],
            metadatas=[{"conv_id": conv_id, "turn_index": int(turn_index)}],
        )
        return True
    except Exception as e:
        logger.warning(f"conv={conv_id}: index_exchange failed: {e}")
        return False


def retrieve(
    conv_id: str,
    query_text: str,
    k: int = RAG_TOP_K,
    exclude_turns_from: int | None = None,
) -> list[dict]:
    """Return up to k most-similar past exchanges for this conversation.

    Each result: {"turn_index": int, "document": str, "distance": float}.
    Returns [] on any failure or if retrieval is unavailable.

    `exclude_turns_from`: if set, drop results whose turn_index >= this
    value. Used to avoid re-injecting recent turns that are already present
    verbatim in the request (waste of token budget).
    """
    if not _try_init() or _chroma_collection is None:
        return []
    if not query_text:
        return []
    vecs = _embed([query_text])
    if not vecs:
        return []
    try:
        res = _chroma_collection.query(
            query_embeddings=vecs,
            n_results=max(1, k),
            where={"conv_id": conv_id},
        )
    except Exception as e:
        logger.warning(f"conv={conv_id}: retrieve query failed: {e}")
        return []

    out: list[dict] = []
    # chroma returns parallel lists nested one level (per query).
    ids = (res.get("ids") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    for i in range(len(ids)):
        meta = metas[i] if i < len(metas) else {}
        turn_index = int(meta.get("turn_index", -1)) if meta else -1
        if exclude_turns_from is not None and turn_index >= exclude_turns_from:
            continue
        out.append({
            "turn_index": turn_index,
            "document": docs[i] if i < len(docs) else "",
            "distance": float(dists[i]) if i < len(dists) else None,
        })
    return out


def forget_conversation(conv_id: str) -> int:
    """Delete all indexed exchanges for a conversation. Returns the number
    deleted (best-effort; 0 on failure or if unavailable). Wired into the
    /admin/conversations/<id>/facts DELETE so 'forget' clears episodic
    memory too, not just facts.
    """
    if not _try_init() or _chroma_collection is None:
        return 0
    try:
        existing = _chroma_collection.get(where={"conv_id": conv_id})
        ids = existing.get("ids", []) if existing else []
        if ids:
            _chroma_collection.delete(ids=ids)
        return len(ids)
    except Exception as e:
        logger.warning(f"conv={conv_id}: forget_conversation failed: {e}")
        return 0


def conversation_doc_count(conv_id: str) -> int:
    """How many exchanges are indexed for a conv. For /admin + /health.
    0 on failure/unavailable.
    """
    if not _try_init() or _chroma_collection is None:
        return 0
    try:
        existing = _chroma_collection.get(where={"conv_id": conv_id})
        return len(existing.get("ids", []) if existing else [])
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Injection block
# ---------------------------------------------------------------------------

_RETRIEVAL_BLOCK_HEADER = (
    "[Relevant earlier exchanges from this conversation, retrieved by "
    "similarity — use them for continuity and exact recall]"
)


def format_retrieval_block(results: list[dict]) -> str | None:
    """Render retrieved exchanges as a system-message body. None if empty.
    Ordered by turn_index ascending so the model reads them chronologically.
    """
    if not results:
        return None
    ordered = sorted(results, key=lambda r: r.get("turn_index", 0))
    lines = [_RETRIEVAL_BLOCK_HEADER]
    for r in ordered:
        ti = r.get("turn_index", "?")
        lines.append(f"--- (turn ~{ti}) ---")
        lines.append(r.get("document", ""))
    return "\n".join(lines)
