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
- Document identity is CONTENT-ADDRESSED (v3.1 D1): the id is a hash of the
  exchange text. It was the request's turn index until that index was found
  running backwards in production and destroying exchanges. See _doc_id.
- Everything degrades to a safe no-op. If fastembed/chromadb can't import
  or init (disabled, missing deps, corrupt store), retrieval returns []
  and indexing silently skips — chat is NEVER broken by a memory failure.

Heavy objects (embedding model, chroma client) are lazy singletons:
initialized on first use, reused forever.
"""

import hashlib
import logging
import os
import threading
import unicodedata
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


# Length of the hex digest kept in a document id. 16 hex chars = 64 bits;
# a conversation would need billions of exchanges before a birthday collision
# is worth thinking about, and the full digest makes the id unreadable in a log
# line for no gain.
_DOC_ID_HASH_CHARS = 16

# One exchange is two messages: the user's and the assistant's. main.py:1416
# computes turn_index as len(messages)+1, so it advances by 2 per exchange —
# which is exactly the step in the production sequence below. _next_turn_index
# keeps stored ordinals in those same message-units so that main.py:1607's
# `recent_cutoff = turn_index - KEEP_RECENT_TURNS*2` still compares like with
# like. A per-exchange counter (1, 2, 3…) would be a different unit system and
# REMEDIATION rejected it for that reason.
_TURN_INDEX_STEP = 2

# main.py's own knob, read here so the two stay in step. It is the size of the
# verbatim window main.py keeps at the tail of the request, and therefore the
# number of stored rows the caller's `exclude_turns_from` filter is *expected*
# to remove. Read from the environment rather than imported because main
# imports this module, not the other way round.
_KEEP_RECENT_TURNS = int(os.environ.get("COMPACTOR_KEEP_RECENT_TURNS", "4") or 4)

# The distance, in the message-units the stored ordinals use, between the
# caller's own position and the cutoff it derives from it: main.py computes
# `max(0, turn_index - KEEP_RECENT_TURNS * 2)`. If the caller's framing and the
# store's agreed, the store's own maximum ordinal would sit within this distance
# of the cutoff. See _cutoff_is_out_of_frame.
_RECENT_WINDOW_UNITS = _KEEP_RECENT_TURNS * _TURN_INDEX_STEP

# v3.1 A7. How many candidates `retrieve` asks for BEYOND k, so that a row the
# exclusion filter removes is replaced rather than simply lost.
#
# _KEEP_RECENT_TURNS is the count above, and it is exactly the number of rows
# the filter is *expected* to remove: the caller excludes its verbatim tail, and
# that tail is _KEEP_RECENT_TURNS exchanges, which is _KEEP_RECENT_TURNS stored
# rows. So over-fetching by that much makes the common case whole — ask for five
# useful hits, get five — while staying bounded: chroma is asked for k + 4, not
# for the conversation.
_OVERFETCH = _KEEP_RECENT_TURNS


def _doc_id(conv_id: str, document: str) -> str:
    """Content-addressed id: `{conv_id}::{sha256(document)[:16]}`.

    v3.1 D1. This was `{conv_id}::{turn_index}` — the request's own
    `len(messages)+1` — which is a property of the CLIENT'S ARRAY, not of the
    exchange. Deleting messages in OpenWebUI shortens that array, so the index
    goes DOWN and the upsert lands on a row that already holds a different
    exchange. Measured in production: five DELETE /api/v1/chats/…/messages/…
    between 06:15:29 and 06:15:44 turned the indexed sequence into

        42, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58

    and the real turn ~42, written at 06:14:54, was overwritten at 14:49:22.
    The phantom conversation is the same failure at its limit: sixteen writes
    to index 2 — one document, fifteen destructions.

    **Damaged conversations do not heal.** Content-addressing stops the next
    exchange being destroyed; it recovers nothing. The overwritten text is not
    in any backup either, because the collapse was continuous rather than an
    event with a before — every archive holds the same single surviving row.

    A hash of the text cannot go backwards, so the pathological case becomes
    the harmless one: re-indexing the same exchange is a no-op (see
    index_exchange's exists probe), and any exchange whose text differs by one
    character gets its own row. Two genuinely identical exchanges in one
    conversation collapse to a single row — deliberate, since identical text
    has identical retrieval value, and the alternative is re-embedding it.
    """
    digest = hashlib.sha256(document.encode("utf-8", "surrogatepass")).hexdigest()
    return f"{conv_id}::{digest[:_DOC_ID_HASH_CHARS]}"


def _id_exists(doc_id: str) -> bool:
    """True if `doc_id` is already in the store. False on a miss AND on any
    probe failure — a failed probe must fall through to the embed+upsert, which
    is still correct (upsert on a matching id is idempotent), never skip it.
    """
    if _chroma_collection is None:
        return False
    try:
        # include=[] so the probe fetches ids only. The same query without it
        # pulls the full document text back out of SQLite to answer a yes/no.
        got = _chroma_collection.get(ids=[doc_id], include=[])
        return bool((got or {}).get("ids"))
    except Exception as e:
        if logsetup.log_once("retrieval._id_exists"):
            logger.warning(
                f"id-exists probe failed ({type(e).__name__}: {e}); indexing "
                f"will re-embed rather than skip until this clears"
            )
        return False


def _stored_max_turn_index(conv_id: str, log_key: str) -> int | None:
    """The highest `turn_index` this conversation has in the store, or None if
    it has no rows or the store could not be read.

    The store's own answer to "where is this conversation up to", as opposed to
    the request's, which is `len(messages) + 1` and therefore a measurement of
    the client's array. Both _next_turn_index (the write path) and
    _cutoff_is_out_of_frame (the read path) need exactly this number, so it is
    one query in one place. `include=["metadatas"]` keeps the document text in
    SQLite; only the ordinals are wanted.
    """
    if _chroma_collection is None:
        return None
    try:
        existing = _chroma_collection.get(
            where={"conv_id": conv_id}, include=["metadatas"]
        )
    except Exception as e:
        if logsetup.log_once(log_key):
            logger.warning(
                f"conv={conv_id}: could not read the stored turn indices "
                f"({type(e).__name__}: {e}); falling back to the request's"
            )
        return None
    highest: int | None = None
    for meta in (existing or {}).get("metadatas") or []:
        try:
            value = int((meta or {}).get("turn_index", -1))
        except (TypeError, ValueError):
            continue
        if highest is None or value > highest:
            highest = value
    return highest


def _next_turn_index(conv_id: str, seed: int) -> int:
    """Ordering metadata for a new row, allocated from the STORE's own maximum
    rather than from the request.

    v3.1 D1. The request cannot be trusted for this: `turn_index` is
    `len(messages)+1`, and a deletion, an edit, a branch switch or a bounded
    client window all shrink it. Taking the max already stored for this
    conversation and stepping past it means the sequence only ever moves
    forward, whatever the client's array is doing.

    The stored maximum is a FLOOR, not the whole answer: the result is
    `max(stored_max + step, seed)`. The request can therefore only ever push
    the sequence forward, never pull it back, which is the property that
    matters — but it is still allowed to push. That second half is for the
    conversations this fix exists for. A conversation damaged pre-D1 has one
    surviving row at whatever index the collapse pinned (the phantom's is 2)
    while the live conversation is hundreds of messages along. Ignoring the
    request there would number new rows 4, 6, 8… against a `recent_cutoff`
    (main.py:1607) computed from the client's array, so no retrieved hit would
    ever look recent and the exclusion filter would quietly stop doing
    anything. Taking the request as a floor-raiser puts the sequence back in
    the same units as the thing it is compared against.

    `seed` also stands alone when the conversation has no rows at all, where
    there is by definition nothing to collide with.

    A wrong answer here costs ordering, not data: since D1 the id is the hash,
    so two rows sharing a turn_index are still two rows.
    """
    if _chroma_collection is None:
        return int(seed)
    highest = _stored_max_turn_index(conv_id, "retrieval._next_turn_index")
    if highest is None:
        return int(seed)
    return max(highest + _TURN_INDEX_STEP, int(seed))


def _cutoff_is_out_of_frame(conv_id: str, cutoff: int) -> bool:
    """True when `exclude_turns_from` cannot be a position in THIS store's
    ordering, so honouring it would suppress the whole retrieval block.

    v3.1 A6, the deferred half of REMEDIATION P0-3. The cutoff arrives as
    `max(0, turn_index - KEEP_RECENT_TURNS * 2)` where `turn_index` is
    `len(messages) + 1` — a measurement of the CLIENT'S ARRAY. Stored ordinals
    are allocated from the store's own maximum (see _next_turn_index), so the
    two are different authorities and they drift apart: a deletion, a
    regeneration, or simply a client sending a bounded window (FRONTEND_SPEC
    §4.2 makes a bounded window the committed shape) leaves the client's number
    permanently below the store's. Reproduced in the v3.1 preflight: a window of
    20 messages against stored ordinals [21,23,…,43] gives a cutoff of 13, and
    **0 of 12 hits survive, on every request, indefinitely.**

    P4 repaired `cutoff == 0` only — "the recent window already covers
    everything" is a reason to exclude nothing, not everything. This is the same
    reading generalised: under a bounded window the cutoff is a positive number
    P4's conditional does not touch.

    The discriminator is the store's own maximum. If the two framings agreed,
    the cutoff would sit `_RECENT_WINDOW_UNITS` below it — that is the whole
    definition of the cutoff. A store maximum further above the cutoff than that
    means the client's position is behind the store's, and the number is not
    comparable with what is stored. A genuinely short conversation, where every
    stored exchange really is in the request verbatim, fails this test and keeps
    its filter: there the store's maximum IS the client's position.

    The real fix is FRONTEND_SPEC §15's server-side `turn_seq` — one authority
    for conversational position, replacing `len(messages) + 1` in main.py. That
    is not this module's to make. Until it lands, this keeps the failure mode on
    the over-inclusive side (a bounded, capped block of maybe-redundant context)
    rather than the silent side (no episodic memory at all, and nothing in the
    log that says so).
    """
    highest = _stored_max_turn_index(conv_id, "retrieval._cutoff_is_out_of_frame")
    if highest is None:
        # No rows, or the store would not answer. Neither is evidence of drift,
        # so leave the caller's filter alone.
        return False
    return highest - cutoff > _RECENT_WINDOW_UNITS


# ---------------------------------------------------------------------------
# Public operations — all degrade to no-op / [] on any failure
# ---------------------------------------------------------------------------

def index_exchange(
    conv_id: str, turn_index: int, user_text: str, assistant_text: str
) -> bool:
    """Embed one exchange and upsert it into the vector store. Returns True
    on success, False if skipped/failed. Never raises.

    v3.1 D1: `turn_index` is still accepted, so every caller is unchanged, but
    it no longer decides the row's IDENTITY — only where the row sorts, and
    even there the store's own maximum is a floor it cannot go under (see
    _next_turn_index). Identity is the hash of the exchange text, so a client
    that deletes messages and re-sends a shorter array adds an exchange
    instead of destroying one. See _doc_id for what it used to do.
    """
    if not _try_init() or _chroma_collection is None:
        return False
    if not user_text or not assistant_text:
        return False
    doc = _exchange_doc(user_text, assistant_text)
    doc_id = _doc_id(conv_id, doc)
    if _id_exists(doc_id):
        # Same text, already stored. Under content-addressing the upsert would
        # be a no-op, so this only skips the embedding — the expensive half.
        return True
    turn_index = _next_turn_index(conv_id, turn_index)
    vecs = _embed([doc])
    if not vecs:
        return False
    try:
        _chroma_collection.upsert(
            ids=[doc_id],
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
    verbatim in the request (waste of token budget). If that cutoff turns out
    not to be a position in this store's ordering it is ignored rather than
    honoured — see _cutoff_is_out_of_frame (v3.1 A6).

    v3.1 A7: the query OVER-FETCHES by _OVERFETCH rows and the result is
    trimmed to k at the end. It used to ask for exactly k and then filter, so
    every excluded row was a LOST slot rather than a replaced one: ask for five,
    have three fall in the recent window, get two — for no reason, since the
    sixth and seventh candidates were sitting right there. Under A5's ordinal
    drift or A6's window divergence that arithmetic empties the block entirely.
    """
    if not _try_init() or _chroma_collection is None:
        return []
    if not query_text:
        return []
    vecs = _embed([query_text])
    if not vecs:
        return []
    cutoff = exclude_turns_from
    # Only over-fetch when there is a filter that could spend the slots. A
    # request with no cutoff has nothing to replace, and asking chroma for rows
    # that will certainly be returned is work for nothing.
    want = max(1, k) + (_OVERFETCH if cutoff else 0)
    try:
        res = _chroma_collection.query(
            query_embeddings=vecs,
            n_results=want,
            where={"conv_id": conv_id},
        )
    except Exception as e:
        logger.warning(f"conv={conv_id}: retrieve query failed: {e}")
        return []

    rows: list[dict] = []
    dropped: list[bool] = []
    # chroma returns parallel lists nested one level (per query).
    ids = (res.get("ids") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    for i in range(len(ids)):
        meta = metas[i] if i < len(metas) else {}
        turn_index = int(meta.get("turn_index", -1)) if meta else -1
        rows.append({
            "turn_index": turn_index,
            "document": docs[i] if i < len(docs) else "",
            "distance": float(dists[i]) if i < len(dists) else None,
        })
        # `> 0`, not `is not None`. The caller passes
        # max(0, turn_index - KEEP_RECENT_TURNS * 2), which is 0 for any short
        # conversation and for any client that sends a bounded window. At 0 the
        # old test excluded every hit with turn_index >= 0 — i.e. all of them —
        # so episodic retrieval returned nothing and said nothing about it.
        # A cutoff of 0 means "the recent window already covers everything",
        # which is a reason to exclude NOTHING, not everything. (REMEDIATION P0-3.)
        dropped.append(
            cutoff is not None and cutoff > 0 and turn_index >= cutoff
        )

    kept = [r for r, drop in zip(rows, dropped) if not drop]

    # v3.1 A6 — the deferred half of REMEDIATION P0-3, now wired in.
    #
    # _cutoff_is_out_of_frame reads the store, and this runs on the request hot
    # path, so it is only paid when the filter actually COST the caller
    # something: it dropped rows AND left fewer than the k asked for. A cutoff
    # that excluded nothing cannot be suppressing anything; a cutoff that
    # excluded four rows and still returned five has already been made whole by
    # the over-fetch below, and there is nothing to rescue. What is left is
    # exactly the case worth a query — the long healthy conversation, which is
    # the one with the most metadata to scan, skips it every time.
    if any(dropped) and len(kept) < k \
            and _cutoff_is_out_of_frame(conv_id, int(cutoff or 0)):
        if logsetup.log_once("retrieval.cutoff_out_of_frame"):
            logger.warning(
                f"conv={conv_id}: exclude_turns_from={cutoff} is not a position "
                f"in this store's ordering (stored max is more than "
                f"{_RECENT_WINDOW_UNITS} ahead of it), so honouring it would "
                f"drop {sum(dropped)} of {len(rows)} hit(s) that are NOT in the "
                f"request verbatim. Ignoring the filter for this conversation "
                f"and injecting the hits; the block stays capped either way. "
                f"The real fix is a server-side turn_seq (FRONTEND_SPEC §15). "
                f"Reported once per process."
            )
        kept = rows

    # v3.1 A7 — the over-fetch is spent here. Everything above may have been
    # handed k + _OVERFETCH candidates; the caller asked for k.
    return kept[:k]


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
        # Not `_doc_id` — that is this module's id function, and rebinding it
        # here would shadow it for the rest of the loop body.
        for i, _row_id in enumerate(ids):
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

    v3.1 D1: the id is content-addressed like index_exchange's, but the
    bundle's `turn_index` is kept verbatim as the ordering metadata rather than
    re-allocated from the store — it is the source conversation's ordering, it
    is what export_indexed_exchanges sorts on, and re-numbering it here would
    make a round-trip lossy. The bundle format is unchanged and BUNDLE_VERSION
    stays "v2.1": ids were never in it.

    The exists probe means re-importing a bundle does not re-embed, and that an
    exchange already indexed live keeps the ordinal it has rather than taking
    the bundle's. import_conversation(overwrite=True) clears the conversation
    first, so a deliberate replacement still gets the bundle's ordering.
    """
    if not _try_init() or _chroma_collection is None:
        return False
    if not document:
        return False
    doc_id = _doc_id(conv_id, document)
    if _id_exists(doc_id):
        return True
    vecs = _embed([document])
    if not vecs:
        return False
    try:
        _chroma_collection.upsert(
            ids=[doc_id],
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

_TRUNCATION_NOTE = "\n[...truncated to fit the retrieval budget]"

# Below this the truncated stub is not worth emitting: a header with almost
# nothing under it tells the model relevant earlier exchanges exist and then
# shows it none of them. 50 tokens, which is the 200 CHARACTERS this threshold
# used to be, converted at the same 4 chars/token the ASCII term below uses.
_MIN_TRUNCATED_TOKENS = 50


def _estimate_tokens(text: str) -> int:
    """Approximate what vLLM will charge for `text`, in TOKENS.

    v3.1 A4. This budget used to be `MAX_RETRIEVAL_TOKENS * 4` — a CHARACTER
    count, compared against a character sum, then logged as though it were a
    token figure. Characters are the one unit that cannot see decoration, and
    decoration is what this model writes: INCIDENT_2026-08-28 measures a reply
    of 1,710 U+2501 plus 441 U+2500 that vLLM charged ~4,275 tokens, against
    2,209 characters. chars/4 called that 552. **A 7.74x undercount, so the
    6,000-character cap admitted ~11,600 real tokens against a nominal 1,500** —
    about a third of the whole input budget, in the layer whose own comment
    (above MAX_RETRIEVAL_TOKENS) says no injected memory should outweigh the
    conversation it supports. The guard does not absorb it either: it sheds
    conversation turns before it touches an injected system block, so the
    oversized block wins and real conversation is deleted to make room for it.

    A raised flat multiplier would not fix this, it would move it — a
    prose-tuned constant is wrong on decoration by that same 7.74x whatever
    value it takes. So the estimate is split by character class, because the
    divergence is entirely a property of class:

    * **ASCII → chars/4.** Unchanged, deliberately: the measured density on
      this deployment's chat transcript is 4.10 chars/token (4.28 on Python,
      3.66 on markdown), so /4 is the same slightly-conservative estimate this
      module has always used, and a pure-prose block renders byte-identically
      to what it rendered before this fix. That matters — the common case must
      not shrink.
    * **Non-ASCII letters and combining marks → 1.25 tokens per CHARACTER.**
      The per-byte rule was written for decoration and applied to everything
      non-ASCII, which over-priced natural script by its byte width: Greek at
      ~2 bytes/char measured 2.34x real cost, CJK 4.27x - so scripture-heavy
      retrieved memories were trimmed or dropped against the 1,500-token
      budget on exactly the conversations this user cares most about (the
      same mispricing v3.1.3 fixed in summarizer._estimate_block_tokens;
      this function was its unfixed sibling). 1.25/char is a measured
      ceiling for every script tested (worst: Hebrew with niqqud at 1.16).
    * **Non-ASCII everything else (decoration, emoji) → 1.05 tokens per
      UTF-8 byte.** The per-byte ceiling this rule was actually written for,
      plus 5% because emoji measured just over one token per byte.

    Deliberately no tokenizer and no HTTP: this module is torch-free on purpose,
    and `format_retrieval_block` is called from the request hot path inside an
    async handler, where a blocking /tokenize POST is the wrong trade. Callers
    that already hold a real counter can pass one — see `format_retrieval_block`
    — and the exact accounting downstream in `_enforce_hard_budget` is unchanged.
    Being wrong here now costs a retrieval block smaller than it needed to be,
    which is recoverable, instead of one 7x larger than believed, which was not.
    """
    # The pure-ASCII common case stays C-level and byte-identical to the
    # old behaviour; the per-character loop runs only over the non-ASCII
    # remainder of mixed text, which for this deployment's transcript is
    # rare outside the exact scripture/decoration cases it exists to price.
    data = text.encode("utf-8", "surrogatepass")
    ascii_chars = len(text.encode("ascii", "ignore"))
    if len(data) == ascii_chars:
        return ascii_chars // 4
    script_chars = decor_bytes = 0
    for c in text:
        if ord(c) < 128:
            continue
        if unicodedata.category(c)[0] in ("L", "M"):
            script_chars += 1
        else:
            decor_bytes += len(c.encode("utf-8", "surrogatepass"))
    return (
        ascii_chars // 4
        + int(script_chars * 1.25)
        + int(decor_bytes * 1.05)
        + 1
    )


def _longest_prefix_within(text: str, budget: int, measure) -> str:
    """The longest prefix of `text` that `measure` prices at <= `budget`.

    Bisected rather than sliced at a character offset: once the budget is in
    tokens there is no fixed chars-per-token to slice at, and the whole point of
    A4 is that assuming one is how the cap failed. `measure` is monotonic
    non-decreasing over prefixes — appending a character can never reduce a
    token count — which is what makes the bisection sound.
    """
    if budget <= 0:
        return ""
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if measure(text[:mid]) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo]


def format_retrieval_block(results: list[dict], count_tokens=None) -> str | None:
    """Render retrieved exchanges as a system-message body. None if empty.
    Ordered by turn_index ascending so the model reads them chronologically.

    `count_tokens` is an optional `str -> int` measure. It exists so the exact
    counter can be supplied from outside without this module growing a tokenizer
    or an HTTP call of its own: `main.count_tokens_exact` asks the process that
    will do the charging, but `main` imports THIS module, so the dependency can
    only run in this direction. Left unset — which is every caller today — the
    budget is measured with `_estimate_tokens`, whose ceiling on the content
    that broke this cap is documented there.
    """
    if not results:
        return None
    ordered = sorted(results, key=lambda r: r.get("turn_index", 0))
    measure = count_tokens or _estimate_tokens

    # A TOKEN budget, measured in tokens, and now honestly named. It was
    # MAX_RETRIEVAL_TOKENS * 4 characters — see _estimate_tokens for what that
    # admitted. This cap's job is still "never let this layer dominate the
    # window" rather than exact accounting; _enforce_hard_budget does the exact
    # accounting downstream.
    budget = MAX_RETRIEVAL_TOKENS
    lines = [_RETRIEVAL_BLOCK_HEADER]
    used = measure(_RETRIEVAL_BLOCK_HEADER)
    kept = 0
    for r in ordered:
        ti = r.get("turn_index", "?")
        sep = f"--- (turn ~{ti}) ---"
        doc = r.get("document", "") or ""
        # Priced as the bytes that will actually be emitted, newlines included,
        # rather than as parts plus a guessed constant for the joins.
        cost = measure(f"{sep}\n{doc}\n")
        if used + cost > budget:
            # Truncate rather than drop when nothing has been included yet: a
            # single oversized exchange should still contribute its opening,
            # which is where the answer to "what were we talking about" lives.
            # Once something is in, prefer whole exchanges over ragged ones.
            if kept == 0:
                room = budget - used - measure(f"{sep}\n\n")
                if room > _MIN_TRUNCATED_TOKENS:
                    body = _longest_prefix_within(doc, room, measure)
                    if body.strip():
                        lines.append(sep)
                        lines.append(body.rstrip() + _TRUNCATION_NOTE)
                        kept = 1
            break
        lines.append(sep)
        lines.append(doc)
        used += cost
        kept += 1

    if kept == 0:
        if ordered:
            logger.info(
                f"retrieval block: kept 0 of {len(ordered)} exchange(s) — "
                f"nothing fits the {MAX_RETRIEVAL_TOKENS}-token budget "
                f"(COMPACTOR_MAX_RETRIEVAL_TOKENS); no block injected"
            )
        return None
    block = "\n".join(lines)
    if kept < len(ordered):
        # Names the counter that made the decision, because "kept 2 of 3" with
        # no unit was how a character sum passed for a token figure for a whole
        # release (A4, and A9's general complaint).
        logger.info(
            f"retrieval block: kept {kept} of {len(ordered)} exchange(s) "
            f"within the {MAX_RETRIEVAL_TOKENS}-token budget "
            f"(COMPACTOR_MAX_RETRIEVAL_TOKENS) — block measures ~"
            f"{measure(block)} token(s) by "
            f"{'the caller-supplied counter' if count_tokens else 'the local estimate'}"
        )
    return block
