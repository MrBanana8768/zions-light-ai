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

Blast radius (v3.1 V7). A merge is the only path in this module that
deletes facts, and one LLM reply used to be able to delete a cluster of
any size and put a ≤60-token — possibly truncated — line in its place.
Four bounds now sit on that reply: clusters are capped at
MAX_CLUSTER_SIZE, a response that hit max_tokens is refused, a response
materially shorter than the facts it replaces is refused, and the word
KEEP anywhere in the response is refused. Every refusal preserves the
cluster; a missed dedup is recovered by the next pass, a bad merge is
not recovered at all.

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
import re

import httpx

import retrieval as retrieval_module

logger = logging.getLogger("compactor.dedup")

# Cosine similarity threshold for Stage 1 clustering. Lower = more LLM
# calls but fewer missed dupes. 0.75 catches paraphrases ("Lyra is
# half-elf" / "Lyra is half elven") while skipping unrelated facts.
SIMILARITY_THRESHOLD = float(
    os.environ.get("COMPACTOR_DEDUP_SIMILARITY", "0.75") or 0.75
)

# Hard cap on LLM calls per dedup pass. 10 is generous — typical real
# workloads see 0-2 clusters. Note this caps *calls*, not how many facts
# a call may delete; MAX_CLUSTER_SIZE is what bounds that.
MAX_LLM_CALLS_PER_PASS = int(
    os.environ.get("COMPACTOR_DEDUP_MAX_LLM_CALLS", "10") or 10
)

# V7: hard cap on how many facts a single LLM reply may replace.
# Stage-1 clustering is a transitive closure with no similarity floor
# between endpoints: A~B, B~C … Y~Z chains 20 unrelated facts into one
# cluster, and one cluster is one call, so MAX_LLM_CALLS_PER_PASS bounded
# the time budget and nothing else. Oversized clusters are split into
# sub-clusters of 2..4 rather than dropped — the facts stay eligible, but
# no single reply can take more than three of them with it.
# Deliberately not env-tunable: this is a safety bound, not a knob.
MAX_CLUSTER_SIZE = 4

# V7: a merged fact shorter than this fraction of the shortest fact it
# replaces has summarized the cluster away rather than reworded it. The
# mid-sentence truncation case is caught by the finish_reason check, not
# by this; this catches the degenerate collapse ("Lyra" for four facts
# about Lyra).
MIN_MERGE_LENGTH_RATIO = 0.5

# V7: was 60. A ~20-word merged fact fits in 60 tokens, but a model that
# preambles at all did not, and the truncated fragment passed the len<6
# guard and was stored as the canonical fact. Raised so a well-behaved
# reply never approaches the limit; anything that still hits it is
# refused outright by the finish_reason check.
MERGE_MAX_TOKENS = 160

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


def _split_cluster(cluster: list[int]) -> list[list[int]]:
    """Split an oversized cluster into sub-clusters of 2..MAX_CLUSTER_SIZE.

    Sizes are balanced rather than greedy so no sub-cluster is left with a
    single member: a singleton has nothing to merge with, so a greedy
    4+4+1 split would silently drop that fact from the pass. Balanced
    gives 5 → 3+2, 7 → 4+3, 9 → 3+3+3 — every member still a candidate,
    none in a group larger than the cap.
    """
    n = len(cluster)
    if n <= MAX_CLUSTER_SIZE:
        return [cluster]
    chunks = -(-n // MAX_CLUSTER_SIZE)  # ceil
    base, extra = divmod(n, chunks)
    out: list[list[int]] = []
    start = 0
    for k in range(chunks):
        size = base + (1 if k < extra else 0)
        out.append(cluster[start:start + size])
        start += size
    return out


def find_candidate_clusters(
    facts: list[dict], *, threshold: float = SIMILARITY_THRESHOLD
) -> list[list[int]]:
    """Group fact indices by transitive similarity. Returns clusters of
    size 2..MAX_CLUSTER_SIZE — singletons have nothing to merge with, and
    V7 caps the upper end.

    Transitive closure: if A~B and B~C but A!~C above threshold, all three
    cluster anyway. The LLM in Stage 2 decides whether the group truly
    merges.

    That closure is unbounded, which is the V7 defect: the chain can reach
    facts with no similarity to each other at all, and Stage 2 answers for
    the whole group in one reply. Groups over the cap are split here, at
    the point they are formed, so every consumer of this function inherits
    the bound.
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

    clusters: list[list[int]] = []
    for g in groups.values():
        if len(g) < 2:
            continue
        if len(g) > MAX_CLUSTER_SIZE:
            logger.info(
                f"dedup: transitive cluster of {len(g)} facts exceeds the "
                f"cap of {MAX_CLUSTER_SIZE}; splitting into sub-clusters so "
                f"no single merge can replace all of them"
            )
        clusters.extend(_split_cluster(g))
    return clusters


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

# D39: the old parser upper-cased the response, stripped leading bullets and
# tested startswith("KEEP"), so a leading KEEP was honoured and a trailing one
# was not — "These are different, KEEP" was read as a *merge* and became the
# canonical fact, replacing everything it was asked about. Match the word
# wherever it appears instead. A genuine merge that happens to contain "keep"
# is a missed dedup the next pass recovers; the other direction is permanent.
_KEEP_RE = re.compile(r"\bKEEP\b", re.IGNORECASE)


async def llm_merge_candidate(
    client: httpx.AsyncClient,
    vllm_url: str,
    model: str,
    cluster_facts: list[dict],
) -> str | None:
    """Ask the LLM to merge a candidate cluster. Returns:
      - merged text (str) if LLM agreed the facts are redundant
      - None if LLM said KEEP anywhere in its reply, the reply was
        truncated at max_tokens, the reply is materially shorter than the
        facts it would replace, the call failed, or it didn't parse

    Returning None = preserve cluster as-is. Safe default — we never lose
    information from a failed or half-finished LLM call.
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
        "max_tokens": MERGE_MAX_TOKENS,
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
        choice = r.json()["choices"][0]
        raw = choice["message"]["content"].strip()
        # .get: a backend that omits finish_reason is not evidence of
        # truncation, and refusing every merge on a missing optional field
        # would silently disable dedup instead of bounding it.
        finish_reason = choice.get("finish_reason")
    except Exception as e:
        logger.warning(f"dedup LLM call failed (cluster preserved): {e}")
        return None

    # V7: a reply that ran out of tokens is not a decision. Unchecked, the
    # fragment cleared the len<6 guard below and was stored as canonical —
    # "Lyra is a half-elf ranger who lives in Aethermere and prefers"
    # replacing the facts it was built from.
    if finish_reason == "length":
        logger.warning(
            f"dedup: merge response hit max_tokens ({MERGE_MAX_TOKENS}) and "
            f"is truncated; cluster of {len(cluster_facts)} preserved"
        )
        return None

    # KEEP detection (D39): anywhere in the response, not just the head.
    # Covers "KEEP", "KEEP.", "Keep — these are different" and the case
    # that used to destroy the cluster, "These are different, KEEP".
    if _KEEP_RE.search(raw):
        return None

    # Otherwise treat as a merged fact line. Strip leading bullets/numbers.
    cleaned = raw.lstrip("- *•").lstrip()
    # Strip leading "1. " style numbering
    if len(cleaned) >= 3 and cleaned[0].isdigit() and cleaned[1:3] in (". ", ") "):
        cleaned = cleaned[3:].lstrip()
    # Minimal sanity: too short isn't a real fact.
    if len(cleaned) < 6:
        return None
    # V7: a merge is a rewording of the cluster, not a summary of it. A
    # reply materially shorter than the shortest fact it would replace has
    # dropped information; the facts it came from are the better record.
    shortest = min(len(f.get("text", "") or "") for f in cluster_facts)
    if len(cleaned) < shortest * MIN_MERGE_LENGTH_RATIO:
        logger.warning(
            f"dedup: merged text ({len(cleaned)} chars) is far shorter than "
            f"the shortest of {len(cluster_facts)} clustered facts "
            f"({shortest} chars); cluster preserved"
        )
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
            # V7: log the texts, not just the count. This is the only
            # record of what a merge destroyed — the facts file is
            # overwritten by the caller and there is no tombstone. If the
            # user reports that the assistant forgot something, this line
            # is what makes it recoverable.
            removed_texts = [f.get("text", "") for f in cluster_facts]
            logger.info(
                f"dedup: merging {len(cluster_facts)} fact(s) into "
                f"{merged_text!r}; removed: {removed_texts!r}"
            )
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
