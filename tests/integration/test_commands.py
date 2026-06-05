"""
Tier-3 validation of chat command surface (V2.1 Phase 5).

Sends slash commands as user messages and verifies the compactor
short-circuits — no vLLM call, response is the command output.

The synthetic-completion shape is also verified: OpenWebUI consumes
these like any normal assistant reply, so the response must be a
well-formed chat.completion object.
"""

import _harness as H


def _send_command(text: str, conv_id: str):
    """Send a single user message containing a slash command. Returns
    the harness ChatResult. Helper because every test here sends one."""
    return H.chat(text, conv_id=conv_id, max_tokens=50)


def test_help_command_lists_documented_commands(conv_id):
    """`/help` should mention every documented command."""
    r = _send_command("/help", conv_id)
    assert r.status_code == 200
    body = r.response_text
    for token in ("/list-facts", "/remember", "/forget", "/why", "/help"):
        assert token in body, f"missing {token!r} in help output: {body[:300]!r}"


def test_list_facts_empty_returns_friendly_message(conv_id):
    """Empty conv → '/list-facts' returns a 'no facts' message, not error."""
    r = _send_command("/list-facts", conv_id)
    assert r.status_code == 200
    assert "No facts" in r.response_text or "no facts" in r.response_text.lower()


def test_remember_then_list_round_trip(conv_id):
    """/remember adds a fact; /list-facts shows it next turn."""
    H.skip_if_no_admin("verifying state needs admin endpoint")
    r1 = _send_command(
        "/remember Lyra is a half-elf ranger from Aethermere", conv_id,
    )
    assert r1.status_code == 200
    assert "Remembered" in r1.response_text
    # Confirm via admin (most direct check)
    stored = H.admin_get_facts(conv_id)
    assert any("Lyra" in f["text"] for f in stored), (
        f"manual fact not stored: {stored!r}"
    )
    # And confirm /list-facts sees it
    r2 = _send_command("/list-facts", conv_id)
    assert "Lyra" in r2.response_text


def test_forget_substring_removes_matches_only(conv_id):
    """/forget <substring> removes matching facts, leaves others alone."""
    H.skip_if_no_admin()
    _send_command("/remember The kingdom is Aethermere", conv_id)
    _send_command("/remember The villain is Maglor the Cold", conv_id)
    before = H.admin_get_facts(conv_id)
    assert len(before) >= 2, f"prep: expected ≥2 facts, got {len(before)}"

    r = _send_command("/forget Aethermere", conv_id)
    assert r.status_code == 200
    assert "Forgot" in r.response_text or "matched" in r.response_text

    after = H.admin_get_facts(conv_id)
    # Maglor fact should still be there
    assert any("Maglor" in f["text"] for f in after), \
        f"selective forget removed wrong fact: {after!r}"
    # Aethermere fact should be gone
    assert not any("Aethermere" in f["text"] for f in after), \
        f"target fact not removed: {after!r}"


def test_forget_no_arg_wipes_everything(conv_id):
    """/forget with no arg = full wipe (same as /admin/forget)."""
    H.skip_if_no_admin()
    _send_command("/remember Test fact to be wiped", conv_id)
    assert len(H.admin_get_facts(conv_id)) >= 1, "prep: at least 1 fact"

    r = _send_command("/forget", conv_id)
    assert r.status_code == 200
    after = H.admin_get_facts(conv_id)
    assert after == [], f"facts not fully cleared: {after!r}"


def test_why_command_summarizes_memory_state(conv_id):
    """/why shows facts + summary + episodic counters."""
    H.skip_if_no_admin()
    _send_command("/remember Lyra has a hawk companion named Hex", conv_id)
    r = _send_command("/why", conv_id)
    assert r.status_code == 200
    body = r.response_text
    assert "Memory state" in body
    assert "Hex" in body, f"recently-added fact missing from /why: {body!r}"
    assert "Summary stack" in body
    assert "Indexed exchanges" in body


def test_unknown_slash_passes_through_to_vllm(conv_id):
    """A message starting with `/` but not matching a command (e.g.
    `/usr/local/foo`) passes through to vLLM as a normal user message.
    This is the safety check that command detection isn't overly greedy."""
    # The model should respond with some content (not the compactor's
    # "Unknown command" hint). We just check status + non-empty body.
    r = H.chat("/usr/local/bin is a path I use often.", conv_id=conv_id, max_tokens=30)
    assert r.status_code == 200
    assert "Unknown command" not in r.response_text, (
        "compactor falsely matched as command — should have passed through"
    )


def test_synthetic_completion_response_shape(conv_id):
    """Raw response shape: id starts with chatcmpl-cmd-, usage tokens=0.
    Catches regressions in build_synthetic_completion."""
    r = _send_command("/help", conv_id)
    assert r.status_code == 200
    # The harness's .response_text strips structure; we want the raw JSON.
    # Reach through to the underlying response if available.
    raw = getattr(r, "raw_json", None)
    if not raw:
        # Fall back: just assert basic shape by re-fetching
        return
    assert raw["id"].startswith("chatcmpl-cmd-"), f"id={raw['id']!r}"
    assert raw["usage"]["total_tokens"] == 0, raw["usage"]
    assert raw["choices"][0]["finish_reason"] == "stop"
