"""
compactor.dedup — V2.1 Phase 7 Step 1: hybrid semantic deduplication.

Why deduplication: over a long conversation, the fact extractor sometimes
produces paraphrases of facts already in storage. ("Lyra is half-elf" +
"Lyra is a half-elf ranger" + "User's protagonist is half-elven"). The
LRU budget in facts.py keeps the *count* bounded but does nothing about
*semantic* duplication, which clutters the injected system message and
wastes context-window budget on the same idea three ways.

Hybrid two-stage design:

  Stage 1 — embedding clustering (cheap):
    Re-use the bge-small ONNX model from retrieval. Compute pairwise
    cosine similarity. Cluster fact indices via union-find above a
    configurable threshold (default 0.75). Singletons are dropped — only
    actual candidate clusters proceed to Stage 2.

  Stage 2 — LLM verification (precise):
    For each candidate cluster, ask Magnum-12B "merge or KEEP?". The LLM
    is the false-positive guard: two facts can be embedding-similar but
    say opposite things ("user wants third-person past" vs "user wants
    first-person present" both embed near "user prose preference").
    Temperature 0.0 + a KEEP-on-doubt prompt to keep the LLM conservative.

Cost shape:
  - Stage 1: O(N²) cosine comparisons on 384-dim vectors. 50 facts ≈ 1275
    compares ≈ <1ms.
  - Stage 2: one LLM call per candidate cluster. MAX_LLM_CALLS_PER_PASS
    caps total at 10 so even pathological "everything is similar" inputs
    can't blow the time budget.

Both inline (after-extraction) and on-demand (/admin/.../dedup) paths
call the single `dedup_facts()` function. Inline path benefits from the
cheap-when-no-candidates fast exit: most extractions produce 0-1 new
facts that are distinct from everything already stored → 0 LLM calls.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os

import httpx

import retrieval as retrieval_module

logger = logging.getLogger("compactor.dedup")

# Cosine similarity threshold for Stage 1 clustering. Lower = more LLM
# calls but fewer missed dupes. 0.75 catches paraphrases ("Lyra is
# half-elf" / "Lyra is half elven") while skipping unrelated facts.
SIMILARITY_THRESHOLD = float(
    os.environ.get("COMPACTOR_DEDUP_SIMILARITY", "0.75") or 0.75
)

# Hard cap on LLM calls per dedup pass. Pathological case: 20 mutually
# similar facts → 1 cluster → 1 call. But several smaller clusters can
# multiply. 10 is generous — typical real workloads see 0-2 clusters.
MAX_LLM_CALLS_PER_PASS = int(
    os.environ.get("COMPACTOR_DEDUP_MAX_LLM_CALLS", "10") or 10
)

# Per-LLM-call timeout. Short — these are quick yes/no merges, not
# generation. Failed LLM call → cluster preserved (no false merges).
LLM_TIMEOUT_S = float(
    os.environ.get("COMPACTOR_DEDUP_LLM_TIMEOUT_S", "30.0") or 30.0
)


# ---------------------------------------------------------------------------
# Stage 1 — embedding clustering
# ---------------------------------------------------------------------------

def _embed_facts(facts: list[dict]) -> list[list[float]] | None:
    """Embed each fact's text via retrieval module's shared bge-small.
    Returns None if embedding subsystem isn't available — caller treats
    that as "no dedup possible" and returns input unchanged.
    """
    if not facts:
        return []
    texts = [f.get("text", "") or "" for f in facts]
    if not all(texts):
        # Empty/missing text entries — skip dedup rather than embed ""
        # which would cluster everything together.
        return None
    vecs = retrieval_module._embed(texts)
    if not vecs or len(vecs) != len(facts):
        return None
    return vecs


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity. Doesn't assume inputs are pre-normalized — we
    don't want to depend on retrieval's internal embedding contract."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def find_candidate_clusters(
    facts: list[dict], *, threshold: float = SIMILARITY_THRESHOLD
) -> list[list[int]]:
    """Group fact indices by transitive similarity. Returns clusters of
    size >=2 only — singletons have nothing to merge with.

    Transitive closure: if A~B and B~C but A!~C above threshold, all three
    cluster anyway. The LLM in Stage 2 decides whether the group truly
    merges.
    """
    vecs = _embed_facts(facts)
    if not vecs:
        return []
    n = len(facts)
    # Union-find for clustering
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for i in range(n):
        for j in range(i + 1, n):
            if _cosine(vecs[i], vecs[j]) >= threshold:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [g for g in groups.values() if len(g) >= 2]


# ---------------------------------------------------------------------------
# Stage 2 — LLM verification + merge
# ---------------------------------------------------------------------------

_MERGE_PROMPT = """You will see {n} facts captured from one conversation. Decide:

  MERGE: if all {n} facts say the same thing in different words, output ONE concise canonical fact under 20 words. No preamble, no explanation — just the fact text.

  KEEP: if ANY two of the facts say different things (even if related), output exactly the word KEEP (no other characters).

When in doubt, output KEEP. False merges destroy information; missed dedup chances are recoverable next pass.

Facts:
{facts_block}

Output (either ONE fact line, or KEEP):"""


async def llm_merge_candidate(
    client: httpx.AsyncClient,
    vllm_url: str,
    model: str,
    cluster_facts: list[dict],
) -> str | None:
    """Ask the LLM to merge a candidate cluster. Returns:
      - merged text (str) if LLM agreed the facts are redundant
      - None if LLM said KEEP, the call failed, or the response didn't parse

    Returning None = preserve cluster as-is. Safe default — we never lose
    information from a failed LLM call.
    """
    if len(cluster_facts) < 2:
        return None
    facts_block = "\n".join(
        f"  {i+1}. {f.get('text', '')}" for i, f in enumerate(cluster_facts)
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": _MERGE_PROMPT.format(
                n=len(cluster_facts), facts_block=facts_block,
            )}
        ],
        # 60 tokens is plenty for "KEEP" or one ~20-word merged fact.
        "max_tokens": 60,
        # Determinism: same cluster → same merge decision. Same lesson
        # we learned from V2.0 extraction NONE-bias debugging.
        "temperature": 0.0,
        "stream": False,
    }
    try:
        r = await client.post(
            f"{vllm_url}/v1/chat/completions", json=payload, timeout=LLM_TIMEOUT_S
        )
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.warning(f"dedup LLM call failed (cluster preserved): {e}")
        return None

    # KEEP detection: case-insensitive prefix, tolerates trailing punctuation
    # ("KEEP", "KEEP.", "Keep — these are different")
    head = raw.upper().lstrip("- *•").strip()
    if head.startswith("KEEP"):
        return None

    # Otherwise treat as a merged fact line. Strip leading bullets/numbers.
    cleaned = raw.lstrip("- *•").lstrip()
    # Strip leading "1. " style numbering
    if len(cleaned) >= 3 and cleaned[0].isdigit() and cleaned[1:3] in (". ", ") "):
        cleaned = cleaned[3:].lstrip()
    # Minimal sanity: too short isn't a real fact.
    if len(cleaned) < 6:
        return None
    return cleaned


def _merge_metadata(cluster_facts: list[dict], new_text: str) -> dict:
    """Build the canonical fact dict from a merged cluster. Preserve the
    most useful metadata:
      - text:       the LLM's merged version
      - added_turn: minimum (when this knowledge first appeared)
      - last_used:  maximum (most-recently-relevant)
    """
    return {
        "text": new_text,
        "added_turn": min(f.get("added_turn", 0) for f in cluster_facts),
        "last_used": max(f.get("last_used", 0) for f in cluster_facts),
    }


# ---------------------------------------------------------------------------
# Public pass — used by both inline trigger and /admin/dedup
# ---------------------------------------------------------------------------

async def dedup_facts(
    client: httpx.AsyncClient,
    vllm_url: str,
    model: str,
    facts: list[dict],
) -> tuple[list[dict], int]:
    """Run a hybrid dedup pass. Returns (deduped_facts, removed_count).

    Never raises — any failure (no embeddings, LLM down, etc.) returns
    the input unchanged with removed=0. Deduplication is hygiene, not
    correctness; it must never affect the user-facing chat path.

    Fast exit when no candidate clusters → 0 LLM calls. This is the
    common case for inline-after-extraction.
    """
    if len(facts) < 2:
        return list(facts), 0

    try:
        clusters = find_candidate_clusters(facts)
    except Exception as e:
        logger.warning(f"dedup: clustering failed (no-op): {e}")
        return list(facts), 0

    if not clusters:
        return list(facts), 0

    to_remove: set[int] = set()
    merged: list[dict] = []
    calls_used = 0

    for cluster in clusters:
        if calls_used >= MAX_LLM_CALLS_PER_PASS:
            logger.info(
                f"dedup: hit LLM call cap ({MAX_LLM_CALLS_PER_PASS}); "
                f"{len(clusters) - calls_used} cluster(s) deferred to next pass"
            )
            break
        cluster_facts = [facts[i] for i in cluster]
        merged_text = await llm_merge_candidate(
            client, vllm_url, model, cluster_facts
        )
        calls_used += 1
        if merged_text:
            merged.append(_merge_metadata(cluster_facts, merged_text))
            to_remove.update(cluster)

    if not to_remove:
        return list(facts), 0

    # Preserve the original list ordering for kept facts (callers care
    # about turn-order); merged facts append at the end and will be
    # re-sorted by callers if they want.
    kept = [f for i, f in enumerate(facts) if i not in to_remove]
    result = kept + merged
    # Sort the final result by added_turn so injection sees a stable order.
    result.sort(key=lambda f: f.get("added_turn", 0))
    removed = len(facts) - len(result)
    if removed > 0:
        logger.info(
            f"dedup: merged {removed} duplicate fact(s) into {len(merged)} "
            f"canonical entries via {calls_used} LLM call(s)"
        )
    return result, removed
