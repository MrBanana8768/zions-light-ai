"""
Tier-3 smoke tests — minimum viable "is the deployment usable?" battery.

Filename prefixed `test_00_` so pytest's default alphabetical collection
runs these FIRST. If any of these fail, downstream test failures are noise
— there's no point asserting facts memory works if the basic chat round-
trip doesn't.
"""

import _harness as H


def test_health_endpoint_responds():
    """GET /health returns ok. The simplest possible liveness check."""
    assert H.health_ok(), f"GET {H.BASE_URL}/health did not return ok"


def test_models_endpoint_lists_at_least_one_model():
    """vLLM is loaded and advertising at least one model."""
    models = H.list_models()
    assert len(models) >= 1, "vLLM advertised zero models"
    print(f"  models advertised: {models}")


def test_minimal_chat_round_trip(conv_id):
    """One chat request through the entire stack: OpenWebUI-shape body →
    compactor → vLLM → response. Asserts the response is non-empty and
    the HTTP status is 200. Deliberately uses a tiny max_tokens so it
    completes quickly even on a cold pod.
    """
    result = H.chat(
        "Say the single word 'pong' and nothing else.",
        conv_id=conv_id,
        max_tokens=8,
    )
    assert result.status_code == 200, f"chat returned {result.status_code}"
    assert result.response_text.strip(), "chat returned empty response"
    print(f"  response: {result.response_text!r}")


def test_chat_handles_system_prompt(conv_id):
    """System messages are accepted and don't crash the compactor's
    inject_system_block logic (which inserts after the leading system run)."""
    result = H.chat(
        "What animal am I supposed to be?",
        conv_id=conv_id,
        system="You are roleplaying as a cat. Always answer in one word.",
        max_tokens=8,
    )
    assert result.status_code == 200
    assert result.response_text.strip(), "empty response with system prompt"
