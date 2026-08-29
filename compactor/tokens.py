"""
compactor.tokens — an accurate LOCAL token count, and a divergence detector.

Why this module exists, in one sentence: on 2026-08-29 vLLM's /tokenize refused
a request and the compactor fell back to a counter that reads up to 51% low,
which took compaction down and cost the user 80+ turns of context per message.

## The counter hierarchy this module completes

    1. vLLM /tokenize          ground truth — what the server will CHARGE
    2. mistral_common (here)   accurate — the same tokenizer family vLLM uses
    3. transformers/tekken     reads 23-51% low on assistant content
    4. char/4                  a guess

Before this module, tier 2 did not exist, so a /tokenize outage dropped
straight to tier 3 — an estimator whose error is largest on exactly the content
this model generates. That is not a graceful degradation; it is the incident.

## This module is NOT a replacement for /tokenize, deliberately

vLLM declares `mistral_common[image]>=1.10.0` — an open lower bound, not a pin.
So the version here and the version vLLM resolves at its next rebuild can
drift apart with nothing enforcing agreement. A local tokenizer that silently
disagrees with the enforcing server is the 2026-08-28 failure with a better
disguise: the wrong number would be *harder* to disbelieve, because it comes
from the right library.

So it has exactly two jobs, and replacing /tokenize is neither of them:

  - **Fallback.** When /tokenize cannot answer, budget on this instead of on
    the 51%-low estimator.
  - **Detector.** When /tokenize CAN answer, compare. A disagreement means the
    two versions have drifted, and finding that out is worth more than either
    number. `check_divergence` exists for that and is the mitigation for the
    unpinned dependency above.

## Failure doctrine

Everything degrades to None. A memory or budgeting component must never be able
to break chat — if mistral_common is absent, the tokenizer file is missing, or
the model is not a Mistral model at all, the caller falls through to the tier
below exactly as it did before this module existed.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

import logsetup

logger = logging.getLogger("compactor.tokens")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_REPO = os.environ.get("MODEL_REPO")
HF_HOME = Path(os.environ.get("HF_HOME", "/data/models"))

# Off switch. The module already no-ops when the library is absent; this is for
# turning it off with the library present — a bad interaction, or a model whose
# native tokenizer is not tekken.
ENABLED = os.environ.get("COMPACTOR_LOCAL_TOKENIZER", "true").lower() != "false"

# How far the local count may sit from vLLM's before it is worth saying so.
# Not a correctness threshold — the two tokenizers can legitimately differ by a
# few tokens of template framing. It is a DRIFT threshold: 5% is far larger
# than framing and far smaller than a version mismatch, which showed up as
# 23-51% in production.
DIVERGENCE_TOLERANCE = float(
    os.environ.get("COMPACTOR_TOKENIZER_DIVERGENCE_TOLERANCE", "0.05") or 0.05
)

# ---------------------------------------------------------------------------
# Lazy singleton (thread-safe, resolved once)
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_tokenizer = None
_available: bool | None = None  # None = untried, True/False = resolved


def _find_tekken() -> Path | None:
    """Locate the served model's native tokenizer inside the HF cache.

    Preferred over from_hf_hub because the pod is expected to run with its
    weights already cached and huggingface.co unreachable (REMEDIATION F25);
    a tokenizer load that needs the network is a boot dependency we do not
    want on the budgeting path.
    """
    if not MODEL_REPO:
        return None
    # models--org--name is HF's on-disk encoding of a repo id.
    cache_dir = HF_HOME / "hub" / f"models--{MODEL_REPO.replace('/', '--')}"
    if not cache_dir.is_dir():
        return None
    # snapshots/<sha>/tekken.json — take the newest snapshot if several exist.
    candidates = sorted(
        cache_dir.glob("snapshots/*/tekken.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _load():
    """Resolve the tokenizer once. Returns it, or None if unusable."""
    global _tokenizer, _available
    if _available is not None:
        return _tokenizer
    with _lock:
        if _available is not None:  # double-checked
            return _tokenizer
        if not ENABLED:
            logger.info("local tokenizer disabled via COMPACTOR_LOCAL_TOKENIZER=false")
            _available = False
            return None
        try:
            from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

            path = _find_tekken()
            if path is None:
                # Not an error worth escalating: a non-Mistral model, or a
                # cache that has not been populated yet. The caller still has
                # /tokenize and the estimator.
                logger.info(
                    f"no tekken.json under {HF_HOME} for {MODEL_REPO}; local "
                    f"exact tokenization unavailable (budgets use /tokenize)"
                )
                _available = False
                return None
            _tokenizer = MistralTokenizer.from_file(str(path))
            logger.info(f"local exact tokenizer ready: {path}")
            _available = True
        except ImportError:
            logger.info(
                "mistral_common not installed; local exact tokenization "
                "unavailable (budgets use /tokenize, then the estimator)"
            )
            _available = False
        except Exception as e:
            logger.warning(
                f"local exact tokenizer failed to load ({type(e).__name__}: {e}); "
                f"budgets fall back to /tokenize and then the estimator"
            )
            _available = False
    return _tokenizer


def is_available() -> bool:
    """Public probe — for /health/full and the boot self-test."""
    return _load() is not None


def count(messages: list[dict]) -> int | None:
    """Exact local token count for `messages`, or None if unavailable.

    Mirrors count_tokens_exact's contract: never raises, returns None rather
    than a guess, so a caller can tell "could not measure" from "measured
    zero". Returning a number it is not sure about is the one thing a token
    counter must never do.
    """
    if not messages:
        return 0
    tok = _load()
    if tok is None:
        return None
    try:
        from mistral_common.protocol.instruct.request import ChatCompletionRequest

        # continue_final_message is the same distinction that produced the D1
        # outage against vLLM's /tokenize: a slice of history ending on an
        # assistant turn is a continuation, not a prompt awaiting a reply.
        # Getting it wrong here would not 400 — it would silently add or omit
        # the generation-prompt tokens, which is worse.
        last_role = messages[-1].get("role")
        request = ChatCompletionRequest(
            model=MODEL_REPO,
            messages=_sanitize(messages),
            continue_final_message=(last_role == "assistant"),
        )
        encoded = tok.encode_chat_completion(request)
        toks = getattr(encoded, "tokens", None)
        return len(toks) if toks is not None else None
    except Exception as e:
        # Once per process. A budgeting side-channel that logs per request
        # would flood the log during exactly the incident it is meant to help
        # diagnose. (v3.1 P0-2b / F61.)
        if logsetup.log_once("tokens.count"):
            logger.warning(
                f"local exact tokenization failed ({type(e).__name__}: {e}); "
                f"budgets fall back to /tokenize and then the estimator"
            )
        return None


def _sanitize(messages: list[dict]) -> list[dict]:
    """Reduce a wire message list to what the tokenizer will accept.

    Multimodal parts, tool calls and provider-specific keys are dropped to
    text. This UNDERCOUNTS an image-bearing request, which is why the caller
    must treat this as a fallback and not as the authority — vLLM's /tokenize
    prices vision tokens and this cannot.
    """
    out: list[dict] = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            content = "".join(
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
        out.append({"role": m.get("role", "user"), "content": content or ""})
    return out


def check_divergence(local: int | None, server: int | None) -> float | None:
    """Compare the local count against vLLM's and complain if they have drifted.

    Returns the ratio (server / local), or None when either side is missing.

    This is the mitigation for the unpinned dependency described at the top of
    this module. vLLM requires mistral_common>=1.10.0 with no upper bound, so
    the version here and the version vLLM uses drift independently; nothing
    detects that except a comparison. A ratio far from 1.0 means the two are no
    longer tokenizing the same way, and every budget derived from the local
    count is suspect until someone looks.

    Rate-limited to once per process: this is a version-skew signal, not a
    per-request one, and a skew that exists at all exists on every request.
    """
    if local is None or server is None or local <= 0:
        return None
    ratio = server / local
    if abs(ratio - 1.0) > DIVERGENCE_TOLERANCE and logsetup.log_once(
        "tokens.divergence"
    ):
        logger.warning(
            f"local exact tokenizer disagrees with vLLM by {ratio:.2f}x "
            f"(local {local} -> vLLM {server}). These should agree to within "
            f"{DIVERGENCE_TOLERANCE:.0%}; a larger gap means the mistral_common "
            f"version here and the one vLLM resolved have drifted apart, or "
            f"the request carries content this cannot price (images). Budgets "
            f"still use vLLM's count — this is a warning about the FALLBACK."
        )
    return ratio


def health() -> dict:
    """Structured state for /health/full and the boot self-test."""
    return {
        "available": is_available(),
        "enabled": ENABLED,
        "model_repo": MODEL_REPO,
        "tolerance": DIVERGENCE_TOLERANCE,
    }
