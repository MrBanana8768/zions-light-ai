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
  - Stage 2: one LLM call per candidate cluster, minus the clusters the
    model has already refused this process (see the refusal memo below).
    MAX_LLM_CALLS_PER_PASS caps total at 10 so even pathological
    "everything is similar" inputs can't blow the time budget.

Cost measured (v3.1 I-6). In a 19h47m production window, 197 of the 301
compactor->vLLM calls were dedup — the single largest consumer of vLLM
requests in the system, and nobody knew, because the module only logged
when a pass merged something. A pass that spent the full ten-call cap and
merged nothing said nothing at all: at 06:15:01-06:15:07 ten POSTs left
zero log lines. Two changes follow from that. Every pass now emits one
INFO summary whatever it did, through a wrapper that has no silent return
path. And a cluster the model has already declined to merge is memoised
by content hash for the life of the process, so the same KEEP is not
re-purchased on every subsequent turn. That re-litigation is the
mechanism MEMORY_REVIEW I-6 identifies as where most of those 197 calls
go: clustering is a pure function of the stored facts, so a refused
cluster re-forms identically every turn and nothing recorded that the
question had already been answered.

The memo stores decisions, never failures. A KEEP, a truncated reply, a
degenerate-collapse reply: those are what this model produces for this
prompt at temperature 0.0, and asking again buys the same answer. A
timeout or a connection error is not an answer, so it is not remembered
and the cluster is re-tried next pass. Forgetting a memo entry costs one
LLM call and never costs a fact, which is why eviction can be crude.

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

Cost measured again (F1 part 4, 48h production window this time): 234
passes, 2,024 LLM calls, 55 merges — 2.7% yield, 190 of 234 passes merged
nothing, and "transitive cluster exceeds the cap" fired on essentially
every pass. The refusal memo above should have absorbed most of that
re-litigation and didn't, because the piece feeding it — how an oversized
transitive cluster gets split into memo-sized sub-clusters — rebalanced
every boundary from scratch as the cluster grew, so a growing blob rarely
handed the memo the same sub-cluster shape twice even when most of its
members hadn't changed. See _split_cluster's docstring for the fix.
SIMILARITY_THRESHOLD was also re-examined and left alone; see its comment
for the measurement.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import os
import re
from collections import OrderedDict

import httpx

import facts as facts_module
import retrieval as retrieval_module

logger = logging.getLogger("compactor.dedup")

# Cosine similarity threshold for Stage 1 clustering. Lower = more LLM
# calls but fewer missed dupes. 0.75 catches paraphrases ("Lyra is
# half-elf" / "Lyra is half elven") while skipping unrelated facts.
#
# F1 part 4: measured, not assumed. 35 synthetic fact pairs embedded with
# the real bge-small model (18 true paraphrases that should MERGE, 17
# related-but-distinct pairs — same shape as the store's actual 2.7%
# yield — that should KEEP) show the two classes OVERLAP in cosine space:
# lowest true-paraphrase similarity 0.7213 ("User wants short replies" /
# "Please keep your answers brief going forward"), highest related-distinct
# similarity 0.9234 ("User prefers third-person past tense" / "User
# prefers first-person present tense" — literally the adversarial example
# this module's own docstring already names). No threshold separates the
# classes; raising it toward 0.80+ starts refusing genuine merges before
# it meaningfully thins the false-candidate rate, and 0.75 already sits
# above the lowest measured true-paraphrase score. Conclusion: leave this
# value where it is — Stage 1's job is a cheap, lossy pre-filter, and
# Stage 2 (the LLM) is the precision the module was already documented as
# depending on, not a redundant confirmation of what embeddings can do
# alone. The gate that measurably reduces calls without touching recall is
# _split_cluster's front-loaded, memo-stable splitting below, not this
# threshold.
SIMILARITY_THRESHOLD = float(
    os.environ.get("COMPACTOR_DEDUP_SIMILARITY", "0.75") or 0.75
)

# Hard cap on LLM calls per dedup pass. 10 is generous — typical real
# workloads see 0-2 clusters. Note this caps *calls*, not how many facts
# a call may delete; MAX_CLUSTER_SIZE is what bounds that.
#
# I-6: the cap is a per-pass budget, not a per-cluster one, and clusters
# are offered to it in ascending first-member order over a fact list the
# caller sorts by added_turn — oldest first. So the clusters that spend
# the budget are the oldest ones, and a store with enough long-standing
# look-alikes to saturate the cap starves every cluster involving a fact
# added since. The refusal memo is what unsticks that: a cluster the
# model already refused is skipped without spending a call, so the budget
# reaches the new material instead of re-buying old answers.
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
#
# Interaction with MAX_LLM_CALLS_PER_PASS (checked for I-6): splitting
# converts one call into ceil(n/4), so the two caps multiply rather than
# compose — a single transitive blob of 40 look-alike facts becomes ten
# sub-clusters and spends the whole ten-call budget on its own. That is
# the intended trade (bounded blast radius costs calls), it is bounded,
# and with the refusal memo it is paid once per distinct sub-cluster
# instead of once per turn.
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
# I-6 — refusal memo
# ---------------------------------------------------------------------------

# Why a memo at all: clustering is a pure function of the stored facts, so
# a cluster the model answered KEEP on re-forms identically on the next
# turn, and on every turn after that, at one LLM call each. Nothing in the
# store records that the question was already asked and answered.
#
# conv_id -> (cluster content hash -> reason the merge was refused).
# Per-conversation because facts are per-conversation and a decision about
# one conversation's facts is not evidence about another's; process-scoped
# because it is a cache of a deterministic call, not memory, and losing it
# on restart costs calls rather than data.
_REFUSAL_MEMO: OrderedDict[str, OrderedDict[str, str]] = OrderedDict()

# Bounds. Both are LRU-evicted, and eviction is safe by construction: the
# only cost of forgetting a refusal is re-asking a question we know the
# answer to. Sized so an ordinary conversation never reaches them — a
# 100-fact store yields at most ~50 clusters in a pass — while a pathological
# one cannot grow the process without limit.
MEMO_MAX_CLUSTERS_PER_CONV = 512
MEMO_MAX_CONVS = 32

# Refusal reasons that came from a completed model reply. At temperature
# 0.0 the same cluster produces the same reply, so re-sending it buys the
# same refusal — memoise these. "error" is deliberately absent: a timeout
# or a 5xx is the backend failing to answer, and remembering it would let
# one bad minute pin a mergeable cluster shut for the life of the process.
_MEMOISABLE_REFUSALS = frozenset({"keep", "truncated", "short", "collapsed"})


def _cluster_key(cluster_facts: list[dict]) -> str:
    """Content hash of a cluster's fact texts.

    Sorted before hashing: after a merge the caller re-sorts the fact list
    by added_turn, so the same set of facts can be presented in a different
    order on a later pass, and it is the same question either way. Joined on
    NUL — which no fact text contains — so ["ab", "c"] cannot collide with
    ["a", "bc"].
    """
    texts = sorted((f.get("text", "") or "") for f in cluster_facts)
    return hashlib.sha256(
        "\x00".join(texts).encode("utf-8", "replace")
    ).hexdigest()


def _memo_get(conv_id: str | None, key: str) -> str | None:
    """The reason this cluster was refused before, or None if it is new.

    No conv_id means no memo: keying every caller's clusters into one
    shared bucket is exactly the cross-conversation bleed the rest of this
    codebase spends its effort preventing, and a missed memo only costs a
    call.
    """
    if not conv_id:
        return None
    per_conv = _REFUSAL_MEMO.get(conv_id)
    if per_conv is None:
        return None
    reason = per_conv.get(key)
    if reason is not None:
        per_conv.move_to_end(key)
        _REFUSAL_MEMO.move_to_end(conv_id)
    return reason


def _memo_put(conv_id: str | None, key: str, reason: str) -> None:
    """Record that the model refused this cluster, LRU-evicting if needed."""
    if not conv_id:
        return
    per_conv = _REFUSAL_MEMO.get(conv_id)
    if per_conv is None:
        per_conv = OrderedDict()
        _REFUSAL_MEMO[conv_id] = per_conv
    per_conv[key] = reason
    per_conv.move_to_end(key)
    while len(per_conv) > MEMO_MAX_CLUSTERS_PER_CONV:
        per_conv.popitem(last=False)
    _REFUSAL_MEMO.move_to_end(conv_id)
    while len(_REFUSAL_MEMO) > MEMO_MAX_CONVS:
        _REFUSAL_MEMO.popitem(last=False)


def reset_refusal_memo(conv_id: str | None = None) -> None:
    """Forget refusals for one conversation, or all of them.

    For tests, and for the call site that edits facts out from under the
    memo: /forget and /remember change the fact set, which changes the
    cluster hashes, so they do not strictly need this — but a caller that
    replaces a conversation wholesale (import, restore) can drop the stale
    decisions rather than wait for eviction.
    """
    if conv_id is None:
        _REFUSAL_MEMO.clear()
    else:
        _REFUSAL_MEMO.pop(conv_id, None)


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

    F1 part 4: front-loaded, not balanced. Fill sub-clusters to
    MAX_CLUSTER_SIZE in cluster order, then repair a too-small trailing
    remainder by borrowing from the chunk before it — never rebalance
    every boundary from scratch. 5 → 4+1 → repaired to 3+2 (same output
    the old balanced split gave); 9 → 4+4+1 → repaired to 4+3+2; 20 →
    4+4+4+4+4, no repair needed.

    Why this matters more than it looks: the caller (find_candidate_
    clusters) re-forms the SAME transitive group turn after turn as
    extraction adds facts, and a store with enough related-but-distinct
    material clusters into one blob well past the cap almost every pass
    (V314_BACKLOG N3: "transitive cluster exceeds the cap" fired on
    essentially every pass in the measured 48h window). The refusal memo
    below keys a cluster's memo entry on its EXACT membership — so
    whether a sub-cluster is "the same question as last time" depends
    entirely on whether this function keeps handing it the same members.

    The balanced split this replaced recomputed every boundary from the
    current total size: growing a 6-member blob to 7 didn't just add a
    member, it moved every chunk boundary (6 → 3+3, 7 → 4+3 — the first
    three facts don't even share a chunk anymore). Since extraction adds
    facts roughly every turn, that meant almost no sub-cluster shape
    survived from one pass to the next, and a memo keyed on exact
    membership rarely hit despite being logically correct — the model
    kept re-answering "do these facts differ?" for group boundaries that
    were themselves the only thing that had changed.

    Front-loading fixes the boundary, not the question: once a leading
    chunk reaches MAX_CLUSTER_SIZE it is far more stable than under the
    balanced split - though not immutable: measured, n=8 gives
    [[0-3],[4-7]] and n=9 gives [[0-3],[4,5,6],[7,8]], so the SECOND full
    chunk did reshape while the leading one held. The memo win is real and
    measured; the guarantee is "the leading chunks stop churning", not
    "nothing ever changes". As the
    blob keeps growing — new members always extend the last (possibly
    under-cap) chunk instead. A leading chunk's refusal is reusable for
    as long as that chunk's own members are unchanged, same as any other
    exact-match memo hit; only the still-growing tail chunk — which
    really does contain a new member and is really a new question — pays
    a fresh call. This does not weaken the cap (MAX_CLUSTER_SIZE is still
    the largest any sub-cluster gets) and does not change which facts are
    eligible (every member of the group is still in exactly one
    sub-cluster of size >= 2, same guarantee the balanced split made).
    """
    n = len(cluster)
    if n <= MAX_CLUSTER_SIZE:
        return [cluster]
    out: list[list[int]] = []
    i = 0
    while i < n:
        out.append(cluster[i:i + MAX_CLUSTER_SIZE])
        i += MAX_CLUSTER_SIZE
    # A trailing remainder of exactly 1 (the only case MAX_CLUSTER_SIZE=4
    # chunking can leave — 0 needs no repair, 2 and 3 are already >= 2) has
    # nothing to merge with. Borrow one member from the chunk before it,
    # which is always a full MAX_CLUSTER_SIZE chunk at this point and so
    # always has one to spare. Only this last boundary moves; every
    # earlier chunk — including the one lending a member — keeps the rest
    # of its membership. Note its memo key DOES change - the memo keys on
    # exact membership - so the lender pays one fresh call; the borrower's
    # neighbours upstream are what stay stable.
    if len(out) >= 2 and len(out[-1]) < 2:
        short = 2 - len(out[-1])
        out[-2], borrowed = out[-2][:-short], out[-2][-short:]
        out[-1] = borrowed + out[-1]
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
) -> tuple[str | None, str]:
    """Ask the LLM to merge a candidate cluster. Returns (text, reason):
      - ("merged text", "merged") if the LLM agreed the facts are redundant
      - (None, reason) otherwise — "keep" (the word KEEP anywhere in the
        reply), "truncated" (stopped at max_tokens), "short" (not a fact),
        "collapsed" (materially shorter than what it would replace),
        "error" (the call failed), "singleton" (fewer than two facts)

    A None text always means "preserve the cluster as-is" — safe default,
    we never lose information from a failed or half-finished LLM call. The
    reason exists because the caller must tell a *decision* apart from a
    *failure* before memoising it (I-6): re-asking after a KEEP is waste,
    re-asking after a timeout is the retry that eventually merges.
    """
    if len(cluster_facts) < 2:
        return None, "singleton"
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
        return None, "error"

    # V7: a reply that ran out of tokens is not a decision. Unchecked, the
    # fragment cleared the len<6 guard below and was stored as canonical —
    # "Lyra is a half-elf ranger who lives in Aethermere and prefers"
    # replacing the facts it was built from.
    if finish_reason == "length":
        logger.warning(
            f"dedup: merge response hit max_tokens ({MERGE_MAX_TOKENS}) and "
            f"is truncated; cluster of {len(cluster_facts)} preserved"
        )
        return None, "truncated"

    # KEEP detection (D39): anywhere in the response, not just the head.
    # Covers "KEEP", "KEEP.", "Keep — these are different" and the case
    # that used to destroy the cluster, "These are different, KEEP".
    if _KEEP_RE.search(raw):
        return None, "keep"

    # Otherwise treat as a merged fact line. Strip leading bullets/numbers.
    cleaned = raw.lstrip("- *•").lstrip()
    # Strip leading "1. " style numbering
    if len(cleaned) >= 3 and cleaned[0].isdigit() and cleaned[1:3] in (". ", ") "):
        cleaned = cleaned[3:].lstrip()
    # Minimal sanity: too short isn't a real fact.
    if len(cleaned) < 6:
        return None, "short"
    # v3.1.1: the SAME predicate the extraction path uses. is_storable_fact's
    # own docstring names this call site — the store has more than one write
    # path and they must share one definition rather than grow three.
    #
    # This matters MORE here than at extraction, not less. Extraction storing a
    # code fence adds one junk row. A merge storing one REPLACES every fact in
    # the cluster with it: the model is asked for a canonical line, and if it
    # answers with "```json" or a heading or a rule, that line becomes the
    # record and the real facts it was built from are gone. The live store
    # already carried scaffolding from the extraction path; nothing stopped it
    # arriving by this one, which destroys rather than merely clutters.
    if not facts_module.is_storable_fact(cleaned):
        # Length and cluster size, NOT the text. This is a rewrite of a whole
        # cluster of the user's real facts, so on the false-positive shapes
        # this predicate is known to have (a quoted phrase with a prose gloss,
        # a terse ALL-CAPS metric) the 60 characters this used to print were
        # real personal memory going to an operator's log file. commands.py
        # states the rule this now follows: "Fact text is real personal memory
        # and does not go to an operator's terminal or a log file; it goes to
        # the chat reply the owner asked for and nowhere else." The two
        # neighbouring refusal branches already log counts and lengths only,
        # and the cluster is preserved, so the text remains inspectable in the
        # store rather than being lost with the diagnostic.
        logger.warning(
            f"dedup: merged text is not a storable fact "
            f"({len(cleaned)} chars); cluster of {len(cluster_facts)} preserved"
        )
        return None, "markup"
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
        return None, "collapsed"
    return cleaned, "merged"


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
        # PIN SURVIVES A MERGE, and this line is load-bearing. Inline dedup
        # runs on EVERY extraction over the whole store, pinned facts
        # included, and identity facts are the most re-extracted class -
        # exactly the ones that cluster with paraphrases of themselves. A
        # merged record built without this key silently un-pinned her
        # identity fact, after /pin had told her "they will now reach me on
        # every turn". Relevance ranking was then free to drop it: the
        # precise failure the pin tier exists to prevent.
        #
        # ANY pinned member pins the merge. A merge is a union of meaning,
        # so the strongest protection in the cluster carries forward; the
        # alternative (all-must-be-pinned) loses the pin whenever a pinned
        # fact absorbs an unpinned paraphrase, which is the common case.
        "pin": any(bool(f.get("pin")) for f in cluster_facts),
    }


# ---------------------------------------------------------------------------
# Public pass — used by both inline trigger and /admin/dedup
# ---------------------------------------------------------------------------

def _log_pass(conv_id: str | None, stats: dict) -> None:
    """The one INFO line every pass emits, productive or not (I-6).

    Before this, dedup logged only when it removed something, so most of
    the passes in the 19h47m window — the ones that spent calls and merged
    nothing — were invisible, and the cheap case was the one that logged.
    Volume is not the objection: roughly thirty passes fired in that
    window, so one line each is a few dozen a day.
    """
    extra = ""
    if stats["deferred"]:
        extra += (
            f"; {stats['deferred']} cluster(s) deferred to the next pass "
            f"(call cap {MAX_LLM_CALLS_PER_PASS})"
        )
    if not conv_id:
        # Not decoration: the memo is per-conversation and a caller that
        # passes no conv_id gets no memoisation at all, which is the
        # difference between a few calls a day and a few hundred.
        extra += "; refusal memo off (caller passed no conv_id)"
    where = f"conv={conv_id}: " if conv_id else ""
    logger.info(
        f"{where}dedup pass: {stats['facts']} fact(s) in, "
        f"{stats['clusters']} candidate cluster(s), "
        f"{stats['memo_skips']} already-refused skip(s), "
        f"{stats['calls']} LLM call(s), {stats['merges']} merge(s), "
        f"{stats['removed']} fact(s) removed{extra}"
    )


async def dedup_facts(
    client: httpx.AsyncClient,
    vllm_url: str,
    model: str,
    facts: list[dict],
    *,
    conv_id: str | None = None,
) -> tuple[list[dict], int]:
    """Run a hybrid dedup pass. Returns (deduped_facts, removed_count).

    Never raises — any failure (no embeddings, LLM down, etc.) returns
    the input unchanged with removed=0. Deduplication is hygiene, not
    correctness; it must never affect the user-facing chat path.

    Fast exit when no candidate clusters → 0 LLM calls. This is the
    common case for inline-after-extraction.

    `conv_id` is optional only so this stays a drop-in for callers that
    have not been updated; supply it. It is what scopes the refusal memo,
    and without it every pass re-asks the model questions it has already
    answered (I-6). It also labels the pass line, which is otherwise the
    one dedup line in the log that cannot be tied to a conversation.

    The work is done by _dedup_pass; this wrapper exists so that no return
    path — including an unexpected raise — can leave a pass unlogged.
    """
    stats = {
        "facts": len(facts), "clusters": 0, "memo_skips": 0,
        "calls": 0, "merges": 0, "removed": 0, "deferred": 0,
    }
    try:
        return await _dedup_pass(client, vllm_url, model, facts, conv_id, stats)
    finally:
        _log_pass(conv_id, stats)


async def _dedup_pass(
    client: httpx.AsyncClient,
    vllm_url: str,
    model: str,
    facts: list[dict],
    conv_id: str | None,
    stats: dict,
) -> tuple[list[dict], int]:
    if len(facts) < 2:
        return list(facts), 0

    try:
        clusters = find_candidate_clusters(facts)
    except Exception as e:
        logger.warning(f"dedup: clustering failed (no-op): {e}")
        return list(facts), 0

    stats["clusters"] = len(clusters)
    if not clusters:
        return list(facts), 0

    to_remove: set[int] = set()
    merged: list[dict] = []
    calls_used = 0

    for cluster in clusters:
        cluster_facts = [facts[i] for i in cluster]
        key = _cluster_key(cluster_facts)
        if _memo_get(conv_id, key) is not None:
            stats["memo_skips"] += 1
            continue
        if calls_used >= MAX_LLM_CALLS_PER_PASS:
            # Count, don't break: the remaining clusters still get their
            # memo check, which is free, so the deferred figure names the
            # clusters that will actually cost a call next pass rather
            # than every cluster we happened to stop in front of.
            stats["deferred"] += 1
            continue
        merged_text, reason = await llm_merge_candidate(
            client, vllm_url, model, cluster_facts
        )
        calls_used += 1
        stats["calls"] = calls_used
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
            stats["merges"] += 1
        elif reason in _MEMOISABLE_REFUSALS:
            # I-6: remember the model's answer, never the backend's
            # failure. _MEMOISABLE_REFUSALS is the difference between the
            # two, and it is the whole point of the reason code.
            _memo_put(conv_id, key, reason)

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
    # The "via N LLM call(s)" line that used to live here fired only when
    # removed > 0, which is why most passes in the 19h47m window left no
    # trace at all. _log_pass reports the same counters unconditionally.
    stats["removed"] = removed
    return result, removed
