"""
Tier-3 degraded-mode tests.

The compactor's contract is "chat NEVER breaks because of a memory
problem." These tests deliberately exercise inputs that could trip the
memory layer (missing conv_id metadata, multimodal content, empty
messages) and assert the chat still completes successfully.

These are a precursor to V2.3's full chaos-test suite — they cover the
most common "weird but valid request" shapes a real client might send.
"""

import _harness as H


def test_chat_works_without_explicit_conv_id():
    """If the client doesn't send X-Conversation-Id or body.metadata.chat_id,
    the compactor's resolve_conv_id falls back to a sha256 hash of the
    opening fingerprint. Chat must still complete.

    Bypasses the standard chat() helper (which always sets conv_id) —
    sends a bare OpenAI-shape request with NO conv-id hints."""
    H.require_base_url()
    model = H.resolve_model()
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Say 'hi' and nothing else."}],
        "max_tokens": 6,
        "temperature": 0.0,
        "stream": False,
    }
    import httpx
    with httpx.Client(base_url=H.BASE_URL, timeout=H.TIMEOUT) as c:
        r = c.post("/v1/chat/completions", json=body)
    assert r.status_code == 200, f"hash-fallback chat returned {r.status_code}"
    text = (r.json().get("choices") or [{}])[0].get("message", {}).get("content", "")
    assert text.strip(), "hash-fallback chat returned empty response"


def test_chat_handles_empty_system_prompt(conv_id):
    """Some clients send `""` for system. Compactor must not crash on
    inject_system_block / facts injection when the leading system has
    empty content."""
    result = H.chat(
        "Reply with 'ok'.",
        conv_id=conv_id,
        system="",  # empty but present
        max_tokens=6,
    )
    assert result.status_code == 200


def test_chat_handles_multimodal_content_array(conv_id):
    """OpenAI's API allows `content` to be a list of typed parts. The
    compactor's _message_text helper must extract text portions cleanly.
    Sending the multimodal form even on a text-only model should not crash
    the compactor (vLLM may error, but the COMPACTOR should pass-through
    cleanly — we accept any well-formed response, including a vLLM error,
    as long as the COMPACTOR didn't 500 on its own logic)."""
    H.require_base_url()
    model = H.resolve_model()
    body = {
        "model": model,
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "Say 'hi' and nothing else."},
            ]},
        ],
        "max_tokens": 6,
        "temperature": 0.0,
        "stream": False,
        "metadata": {"chat_id": conv_id},
    }
    import httpx
    with httpx.Client(base_url=H.BASE_URL, timeout=H.TIMEOUT) as c:
        r = c.post("/v1/chat/completions", json=body)
    # Either it worked end-to-end (200) OR vLLM rejected the multimodal
    # input cleanly (4xx). What MUST NOT happen is the compactor 500ing
    # because its own _message_text or facts-injection logic choked on
    # the list-shaped content.
    assert r.status_code < 500, (
        f"compactor 5xx'd on multimodal text-only content: {r.status_code} "
        f"{r.text[:200]}"
    )


def test_admin_endpoints_reject_unknown_conv(conv_id):
    """GET /admin/conversations/<unknown-id> must return a well-formed
    JSON response with zero counters, not 404 or 500. (The endpoint is
    documented as 'reports whatever state exists' rather than 'errors on
    unknown conv'.)"""
    H.skip_if_no_admin()
    unknown = H.fresh_conv_id()
    summary = H.admin_conv_summary(unknown)
    assert summary.get("conv_id") == unknown
    assert (summary.get("facts", {}).get("count") or 0) == 0
    assert (summary.get("episodic", {}).get("indexed_exchanges") or 0) == 0
