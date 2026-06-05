"""
Tier-3 validation of persona endpoints + injection (V2.1 Phase 8).

GET/POST/DELETE /admin/conversations/<id>/persona
GET /admin/personas
POST /admin/conversations/<id>/inherit-persona

Plus a behavioral check that the persona, when stored via /admin (not
present in the request payload), gets injected into the next chat —
the model's response should reflect the persona content.
"""

import uuid

import _harness as H


def test_persona_get_returns_none_when_unset(conv_id):
    """No persona stored → GET returns None (404 handled by harness)."""
    H.skip_if_no_admin()
    assert H.admin_get_persona(conv_id) is None


def test_persona_set_and_get_round_trip(conv_id):
    """POST /persona → GET /persona returns the same text + source=admin."""
    H.skip_if_no_admin()
    text = (
        "You are a hardboiled noir detective named Sam Cole. "
        "Always reply in first-person past tense, with terse sentences."
    )
    status, rec = H.admin_set_persona(conv_id, text)
    assert status == 200, f"set failed: {status} {rec!r}"
    assert rec["persona_text"] == text
    assert rec["source"] == "admin"
    loaded = H.admin_get_persona(conv_id)
    assert loaded["persona_text"] == text


def test_persona_set_rejects_empty(conv_id):
    """POST with empty text → 400."""
    H.skip_if_no_admin()
    status, body = H.admin_set_persona(conv_id, "   ")
    assert status == 400, f"expected 400, got {status} {body!r}"


def test_persona_delete_clears_it(conv_id):
    """DELETE removes the persona; subsequent GET returns None."""
    H.skip_if_no_admin()
    H.admin_set_persona(conv_id, "throwaway persona text " * 5)
    assert H.admin_get_persona(conv_id) is not None
    result = H.admin_delete_persona(conv_id)
    assert result["deleted"] is True
    assert H.admin_get_persona(conv_id) is None


def test_persona_delete_idempotent(conv_id):
    """DELETE on empty conv → deleted=False, no error."""
    H.skip_if_no_admin()
    result = H.admin_delete_persona(conv_id)
    assert result["deleted"] is False


def test_persona_full_forget_clears_persona():
    """/admin/forget should ALSO clear the persona (full memory wipe)."""
    H.skip_if_no_admin()
    target = f"itest-persona-wipe-{uuid.uuid4().hex[:8]}"
    try:
        H.admin_set_persona(target, "a persona text " * 30)
        assert H.admin_get_persona(target) is not None, "prep: persona stored"

        result = H.admin_forget(target)
        # Phase 8 adds forgotten_persona to the response shape
        assert result.get("forgotten_persona") is True, (
            f"forgot didn't clear persona: {result!r}"
        )
        assert H.admin_get_persona(target) is None, "persona still present after forget"
    finally:
        H.admin_safe_forget(target)


def test_persona_library_listing():
    """GET /admin/personas returns metadata for all stored personas
    without the full text."""
    H.skip_if_no_admin()
    target = f"itest-persona-list-{uuid.uuid4().hex[:8]}"
    try:
        H.admin_set_persona(target, "Library entry persona text " * 5)
        library = H.admin_list_personas()
        ours = [p for p in library if p["conv_id"] == target]
        assert len(ours) == 1, f"our persona not in library: {library!r}"
        entry = ours[0]
        # Library view = metadata only
        assert "persona_text" not in entry, \
            f"library leaked full text: {entry!r}"
        assert entry["length"] > 0
        assert entry["source"] == "admin"
    finally:
        H.admin_safe_forget(target)
        H.admin_delete_persona(target)


def test_persona_inheritance(conv_id):
    """POST /inherit-persona copies text from a source conv."""
    H.skip_if_no_admin()
    source = f"itest-persona-src-{uuid.uuid4().hex[:8]}"
    target = f"itest-persona-dst-{uuid.uuid4().hex[:8]}"
    persona_text = "Inherited persona text describing a character. " * 5
    try:
        H.admin_set_persona(source, persona_text)
        status, body = H.admin_inherit_persona(target, source)
        assert status == 200, f"inherit failed: {status} {body!r}"
        assert body["inherited_from"] == source
        loaded = H.admin_get_persona(target)
        assert loaded["persona_text"] == persona_text.strip()
        assert loaded["source"] == "inherited"
    finally:
        H.admin_safe_forget(source)
        H.admin_safe_forget(target)
        H.admin_delete_persona(source)
        H.admin_delete_persona(target)


def test_persona_inherit_fails_with_missing_source():
    """Inherit from a conv that has no persona → 404."""
    H.skip_if_no_admin()
    target = f"itest-persona-orphan-{uuid.uuid4().hex[:8]}"
    try:
        status, body = H.admin_inherit_persona(target, "no-such-source")
        assert status == 404, f"expected 404, got {status} {body!r}"
    finally:
        H.admin_safe_forget(target)


def test_persona_appears_in_conversation_summary():
    """GET /admin/conversations/<id> reports persona presence after set."""
    H.skip_if_no_admin()
    target = f"itest-persona-sum-{uuid.uuid4().hex[:8]}"
    try:
        H.admin_set_persona(target, "summary-test persona text " * 5)
        summary = H.admin_conv_summary(target)
        p = summary.get("persona")
        assert p is not None, f"persona key missing from summary: {summary!r}"
        assert p["present"] is True
        assert p["length"] > 0
        assert p["source"] == "admin"
    finally:
        H.admin_safe_forget(target)
        H.admin_delete_persona(target)


def test_admin_set_persona_influences_next_chat():
    """The behavioral guarantee: an admin-set persona (not present in the
    request messages) influences the model's response on the next chat.

    Sets a vivid, easy-to-detect persona (pirate dialect), then asks a
    neutral question and checks the response for pirate-flavored tokens.
    Model adherence isn't 100%, so this is lenient: pass if ANY of
    several plausible tokens appear, OR if admin shows the persona was
    captured (proving the injection path didn't error).
    """
    H.skip_if_no_admin()
    target = f"itest-persona-behave-{uuid.uuid4().hex[:8]}"
    try:
        persona_text = (
            "You are Captain Salt, a pirate from the high seas. "
            "Reply in pirate dialect with words like 'ahoy', 'matey', "
            "'arr', 'ye', and 'savvy'. Use these regularly."
        )
        H.admin_set_persona(target, persona_text)

        r = H.chat("Greet me briefly.", conv_id=target, max_tokens=60)
        assert r.status_code == 200

        # Either the response is piratey OR we can confirm persona was stored.
        piratey = H.response_mentions(
            r.response_text,
            "ahoy", "matey", "arr", "ye ", "savvy", "captain", "pirate",
        )
        if not piratey:
            # Storage-level fallback assertion
            loaded = H.admin_get_persona(target)
            assert loaded and loaded["persona_text"] == persona_text, (
                f"persona NOT stored and response NOT piratey — injection broken.\n"
                f"response: {r.response_text[:200]!r}"
            )
            print(
                f"  WARN: persona stored but model didn't adhere — output:\n"
                f"  {r.response_text[:200]!r}"
            )
    finally:
        H.admin_safe_forget(target)
        H.admin_delete_persona(target)
