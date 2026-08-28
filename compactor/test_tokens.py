"""
CPU-only tests for compactor.tokens.

Runs with or without mistral_common installed, and with or without a model
cache — the module's whole contract is that it degrades to None instead of
raising, so the tests assert that contract rather than requiring the library.
Where the library IS present the accuracy tests run for real; where it is not
they skip loudly, because a silent skip is how a test suite reports coverage it
does not have.

    python test_tokens.py
"""

import os
import sys
import tempfile

os.environ.setdefault("MODEL_REPO", "")
os.environ.setdefault("HF_HOME", tempfile.mkdtemp(prefix="tokens-test-hf-"))

import tokens  # noqa: E402


def assert_eq(actual, expected, label):
    if actual != expected:
        print(f"FAIL {label}: expected {expected!r}, got {actual!r}")
        sys.exit(1)
    print(f"  ok   {label}")


def assert_true(cond, label):
    if not cond:
        print(f"FAIL {label}")
        sys.exit(1)
    print(f"  ok   {label}")


def _reset():
    """Clear the resolved singleton so a test can re-run resolution."""
    tokens._tokenizer = None
    tokens._available = None


# ---------------------------------------------------------------------------
print("[1] the failure doctrine: never raise, always fall through")
# ---------------------------------------------------------------------------

_reset()
# HF_HOME points at an empty temp dir and MODEL_REPO is blank, so nothing can
# resolve. This is the shape of a pod whose cache has not been populated.
assert_eq(tokens.count([{"role": "user", "content": "hello"}]), None,
          "count() returns None when no tokenizer can be resolved")
assert_eq(tokens.is_available(), False, "is_available() is False, not an exception")

_reset()
assert_eq(tokens.count([]), 0, "an empty list is 0 tokens, not None")
# 0 and None mean different things and the distinction is the point: 0 is a
# measurement, None is "could not measure". A caller that cannot tell them
# apart will budget against a zero it invented.
assert_true(tokens.count([]) is not None, "0 is a measurement, not a failure")

_reset()
_saved = tokens.ENABLED
tokens.ENABLED = False
assert_eq(tokens.count([{"role": "user", "content": "x"}]), None,
          "the off switch disables it")
tokens.ENABLED = _saved
_reset()

# ---------------------------------------------------------------------------
print("[2] _sanitize: multimodal parts reduce to text without raising")
# ---------------------------------------------------------------------------

out = tokens._sanitize([
    {"role": "user", "content": [
        {"type": "text", "text": "look at this"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]},
    {"role": "assistant", "content": "I see it"},
])
assert_eq(len(out), 2, "every message survives sanitization")
assert_eq(out[0]["content"], "look at this", "text parts are kept")
assert_true("base64" not in out[0]["content"], "image parts are dropped, not stringified")
assert_eq(out[1]["content"], "I see it", "plain string content is untouched")

# A None content must not become the string "None" — that would be counted as
# four tokens of a word the user never said.
out = tokens._sanitize([{"role": "user", "content": None}])
assert_eq(out[0]["content"], "", "None content becomes empty, never 'None'")

out = tokens._sanitize([{"content": "no role"}])
assert_eq(out[0]["role"], "user", "a missing role defaults rather than raising")

# ---------------------------------------------------------------------------
print("[3] check_divergence: the detector for the unpinned dependency")
# ---------------------------------------------------------------------------

assert_eq(tokens.check_divergence(None, 100), None, "None local -> no ratio")
assert_eq(tokens.check_divergence(100, None), None, "None server -> no ratio")
assert_eq(tokens.check_divergence(0, 100), None, "zero local -> no division")
assert_eq(tokens.check_divergence(-5, 100), None, "negative local -> no ratio")

assert_eq(tokens.check_divergence(100, 100), 1.0, "agreement is 1.0")
assert_eq(tokens.check_divergence(100, 123), 1.23,
          "the 2026-08-28 production ratio is reported as measured")

# The tolerance is a DRIFT threshold, not a correctness one. Framing overhead
# of a few tokens must not trip it; a version mismatch must.
_saved_tol = tokens.DIVERGENCE_TOLERANCE
tokens.DIVERGENCE_TOLERANCE = 0.05
assert_true(abs(tokens.check_divergence(1000, 1020) - 1.02) < 1e-9,
            "2% framing drift is measured")
assert_true(abs(tokens.check_divergence(1000, 1510) - 1.51) < 1e-9,
            "51% — the production undercount — is measured")
tokens.DIVERGENCE_TOLERANCE = _saved_tol

# ---------------------------------------------------------------------------
print("[4] health(): reports absence honestly")
# ---------------------------------------------------------------------------

_reset()
h = tokens.health()
for key in ("available", "enabled", "model_repo", "tolerance"):
    assert_true(key in h, f"health() reports {key}")
assert_eq(h["available"], False, "health does not claim a tokenizer it does not have")

# ---------------------------------------------------------------------------
print("[5] real tokenization, when the library is present")
# ---------------------------------------------------------------------------

try:
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer  # noqa: F401
    _HAVE_LIB = True
except ImportError:
    _HAVE_LIB = False

if not _HAVE_LIB:
    print("  SKIP mistral_common is not installed in this environment.")
    print("       The Dockerfile guard fails the build if it is missing from")
    print("       the image, so this skip means 'not the image', not 'not shipped'.")
else:
    import glob
    # mistral_common ships real tekken vocabularies as package data, so the
    # PRESENCE path is testable without the model cache. This matters: D1
    # (/tokenize 400s on an assistant-final list) shipped because the contract
    # harness tested the shape it imagined instead of the one the code sends.
    # A skip here would have repeated that exactly.
    fixtures = sorted(glob.glob(os.path.join(
        os.path.dirname(tokens.__file__.replace("tokens.py", "")),
        "**", "mistral_common", "data", "tekken_*.json"), recursive=True))
    if not fixtures:
        import mistral_common as _mc
        fixtures = sorted(glob.glob(os.path.join(
            os.path.dirname(_mc.__file__), "data", "tekken_*.json")))
    assert_true(bool(fixtures), "a bundled tekken vocabulary is available to test against")

    _reset()
    tokens.MODEL_REPO = "fixture/tekken"
    tokens._find_tekken = lambda: __import__("pathlib").Path(fixtures[0])

    n_user = tokens.count([{"role": "user", "content": "hello there"}])
    assert_true(isinstance(n_user, int) and n_user > 0,
                f"a user-final list counts ({n_user} tokens)")

    # THE D1 SHAPE. vLLM's /tokenize 400s on this when add_generation_prompt is
    # True; the summarizer sends exactly this on every compaction, which is how
    # compaction died in production on 2026-08-29. This module must handle it.
    n_asst = tokens.count([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello back"},
    ])
    assert_true(isinstance(n_asst, int) and n_asst > 0,
                f"an ASSISTANT-final list counts, not raises ({n_asst} tokens)")

    # Multimodal content must reduce rather than explode.
    n_mm = tokens.count([{"role": "user", "content": [
        {"type": "text", "text": "look"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * 5000}},
    ]}])
    assert_true(isinstance(n_mm, int) and n_mm < 50,
                f"a base64 image is dropped, not tokenized ({n_mm} tokens)")

    # Cross-validation of a constant chosen elsewhere from production logs.
    # summarizer._WORST_TOKENS_PER_CHAR is 2.0, derived from the 2026-08-28
    # measurement of one reply. This measures the same density from an
    # independent source: the vocabulary itself.
    rule = "━" * 100
    n_rule = tokens.count([{"role": "user", "content": rule}])
    density = n_rule / len(rule)
    print(f"  ok   U+2501 density measured at {density:.2f} tokens/char "
          f"({n_rule} tokens for {len(rule)} chars)")
    assert_true(1.5 <= density <= 2.5,
                "box-drawing density is near the 2.0 the pessimistic fallback assumes")

    # And the control: prose is where the tokenizers agree, which is exactly
    # why a budget test written on prose passes while production burns.
    prose = "The quick brown fox jumps over the lazy dog. " * 10
    n_prose = tokens.count([{"role": "user", "content": prose}])
    assert_true(n_prose / len(prose) < 0.4,
                "prose is far cheaper per character than decoration")
    _reset()

print()
print("All tokens tests passed.")
