"""
Word Error Rate (WER) — the metric for the STT quality eval.

Pure, dependency-free (stdlib only) so it is unit-testable on any machine with
no audio and no model. WER = (substitutions + deletions + insertions) / number
of reference words, computed as a word-level Levenshtein distance over
normalized tokens. 0.0 = perfect; 1.0 = as wrong as deleting everything.
"""
from __future__ import annotations

import re
import unicodedata


def normalize(text: str) -> str:
    """Lowercase, NFKC-fold, strip punctuation (keep intra-word apostrophes),
    and collapse whitespace — so "Hello, World!" and "hello world" score equal.
    """
    text = unicodedata.normalize("NFKC", text or "")
    text = text.lower()
    # Replace anything that isn't a word char, whitespace, or apostrophe with a space.
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    n = normalize(text)
    return n.split() if n else []


def word_edit_distance(ref: list[str], hyp: list[str]) -> int:
    """Levenshtein edit distance between two token lists (sub/del/ins = cost 1)."""
    n, m = len(ref), len(hyp)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate of `hypothesis` against `reference`.

    Edge cases: empty reference + empty hypothesis = 0.0 (nothing to get wrong);
    empty reference + non-empty hypothesis = 1.0 (all insertions).
    """
    ref = tokenize(reference)
    hyp = tokenize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return word_edit_distance(ref, hyp) / len(ref)
