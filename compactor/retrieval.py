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

import logsetup

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

# v3.1: a token budget for the retrieval block, which had none.
#
# Facts are capped by COMPACTOR_MAX_FACTS_TOKENS (1500) and each summary chunk
# by COMPACTOR_L1_MAX_TOKENS (500). Retrieved exchanges were injected verbatim,
# uncapped, however long they happened to be — and they are whole user+assistant
# pairs from a model that writes at length.
#
# Observed in production 2026-08-27 on a 32k window. Same conversation, same
# ~102 facts, same L1=5 summary stack, three requests:
#     1retr -> ok    0retr -> ok    3retr -> vLLM 400, 33,127 tokens
# Compaction had already reduced that request to 9,915 tokens; injection put it
# over the window on its own. The only variable was the retrieved-hit count.
#
# 1500 mirrors the facts budget deliberately: no injected memory layer should be
# able to outweigh the conversation it is meant to support.
MAX_RETRIEVAL_TOKENS = int(
    os.environ.get("COMPACTOR_MAX_RETRIEVAL_TOKENS", "1500") or 1500
)

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


def conversation_doc_count(conv_id: str) -> int | None:
    """How many exchanges are indexed for a conv. For /admin + /health.

    **None means "could not tell"; 0 means "genuinely nothing indexed".**
    Until v3.1 this returned 0 for both, so a dead ChromaDB was
    indistinguishable from an empty store: /health/full printed
    `indexed_exchanges_total: 0` beside `"status": "ok"` and /admin reported
    a conversation with hundreds of embedded exchanges as having none.
    Callers must render None as "unknown", never fold it into a total.
    (v3.1 P0-2b / F61.)

    `COMPACTOR_RAG_ENABLED=false` returns **0, not None** — a deliberately
    disabled store has genuinely nothing indexed, and that is knowledge, not
    ignorance. Conflating the two broke import and fork outright: the
    import pre-flight reads None as "cannot verify, refuse", so every import
    and every fork 400'd in a supported configuration, including onto a
    freshly-minted fork id that could not possibly be occupied.
    """
    if not RETRIEVAL_ENABLED:
        return 0
    if not _try_init() or _chroma_collection is None:
        return None
    try:
        existing = _chroma_collection.get(where={"conv_id": conv_id})
        return len(existing.get("ids", []) if existing else [])
    except Exception as e:
        # Once per process: /health/full calls this once per conversation
        # every 30s, so a broken store must not write a line per probe.
        if logsetup.log_once("retrieval.conversation_doc_count"):
            logger.warning(
                f"conv={conv_id}: conversation_doc_count failed "
                f"({type(e).__name__}: {e}); episodic counts report unknown "
                f"until this clears"
            )
        return None


def export_indexed_exchanges(conv_id: str) -> list[dict]:
    """V2.1 portability: dump every indexed exchange for one conv as
    [{"turn_index": int, "document": str}, ...] sorted by turn_index.

    Used by compactor/portability.py to round-trip a conversation
    into / out of a single JSON bundle. Embeddings are deliberately
    NOT exported — they'd couple the bundle to the bge-small ONNX
    model. import_indexed_exchange re-embeds on the destination side,
    which works across any embedding model swap.

    Returns [] on any failure or if retrieval is unavailable — never
    raises, so export still produces a partial bundle.
    """
    if not _try_init() or _chroma_collection is None:
        return []
    try:
        existing = _chroma_collection.get(where={"conv_id": conv_id})
        if not existing:
            return []
        ids = existing.get("ids", []) or []
        docs = existing.get("documents", []) or []
        metas = existing.get("metadatas", []) or []
        out: list[dict] = []
        for i, _doc_id in enumerate(ids):
            meta = metas[i] if i < len(metas) else {}
            ti = int((meta or {}).get("turn_index", -1))
            out.append({
                "turn_index": ti,
                "document": docs[i] if i < len(docs) else "",
            })
        return sorted(out, key=lambda d: d["turn_index"])
    except Exception as e:
        logger.warning(f"conv={conv_id}: export_indexed_exchanges failed: {e}")
        return []


def import_indexed_exchange(conv_id: str, turn_index: int, document: str) -> bool:
    """V2.1 portability: re-embed a pre-formatted exchange document and
    upsert into the vector store. Used by compactor/portability.py on
    bundle import.

    `document` is the canonical "[user]: ...\\n[assistant]: ..." string
    as produced by _exchange_doc — same format the embedding model saw
    originally, so semantic neighborhoods carry across the round-trip.

    Returns True on success, False on any failure. Never raises.
    """
    if not _try_init() or _chroma_collection is None:
        return False
    if not document:
        return False
    vecs = _embed([document])
    if not vecs:
        return False
    try:
        _chroma_collection.upsert(
            ids=[_doc_id(conv_id, turn_index)],
            embeddings=vecs,
            documents=[document],
            metadatas=[{"conv_id": conv_id, "turn_index": int(turn_index)}],
        )
        return True
    except Exception as e:
        logger.warning(f"conv={conv_id}: import_indexed_exchange failed: {e}")
        return False


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

    # Budget in characters: this module has no tokenizer (deliberately — it
    # would drag transformers into the retrieval path).
    #
    # chars/4 is an approximation whose error depends on the text. Measured on
    # the DEPLOYED tokenizer (vision-heretic; the image's ENV default is a
    # different repo with different numbers): 4.10 chars/tok on this
    # deployment's chat transcript, 4.28 on Python, 3.66 on markdown.
    #
    # Retrieved exchanges ARE chat text, so ~4.1 is the right central estimate
    # and this budget lands close to nominal in practice. A markdown-heavy
    # exchange overshoots by ~9% (~140 tokens); the densest file measured at
    # 3.35 chars/tok would overshoot ~19% (~290 tokens). Both are small against
    # a 32,768 window. An earlier draft called the 9% figure "the worst case" —
    # it is the aggregate, and the real worst case is twice it.
    #
    # Acceptable either way: this cap's job is "never let this layer dominate
    # the window", not exact accounting. _enforce_hard_budget does the exact
    # accounting downstream.
    budget = MAX_RETRIEVAL_TOKENS * 4
    lines = [_RETRIEVAL_BLOCK_HEADER]
    used = len(_RETRIEVAL_BLOCK_HEADER)
    kept = 0
    for r in ordered:
        ti = r.get("turn_index", "?")
        sep = f"--- (turn ~{ti}) ---"
        doc = r.get("document", "") or ""
        cost = len(sep) + len(doc) + 2
        if used + cost > budget:
            # Truncate rather than drop when nothing has been included yet: a
            # single oversized exchange should still contribute its opening,
            # which is where the answer to "what were we talking about" lives.
            # Once something is in, prefer whole exchanges over ragged ones.
            if kept == 0:
                room = max(0, budget - used - len(sep) - 2)
                if room > 200:
                    lines.append(sep)
                    lines.append(doc[:room].rstrip() + "\n[...truncated to fit the retrieval budget]")
                    kept = 1
            break
        lines.append(sep)
        lines.append(doc)
        used += cost
        kept += 1

    if kept < len(ordered):
        logger.info(
            f"retrieval block: kept {kept} of {len(ordered)} exchange(s) "
            f"within the {MAX_RETRIEVAL_TOKENS}-token budget "
            f"(COMPACTOR_MAX_RETRIEVAL_TOKENS)"
        )
    if kept == 0:
        return None
    return "\n".join(lines)
