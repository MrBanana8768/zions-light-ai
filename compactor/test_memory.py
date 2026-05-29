"""
CPU-only smoke tests for compactor/memory.py (V2.0 Phase 1).

Exercises conv_id resolution + storage layout helpers. No vLLM, no GPU,
no FastAPI runtime needed. Matches the test_smoke.py pattern.

Run inside the compactor image or any container with the requirements
installed:
    python test_memory.py
"""

import json
import os
import shutil
import sys
import tempfile

# Point storage at a temp dir BEFORE importing memory so the module-level
# STORAGE_ROOT picks it up.
_TMP_ROOT = tempfile.mkdtemp(prefix="compactor-test-")
os.environ["COMPACTOR_STORAGE_ROOT"] = _TMP_ROOT

import memory  # noqa: E402


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


# ---------------------------------------------------------------------------
# Conv-id resolution
# ---------------------------------------------------------------------------

def test_header_path_wins():
    print("\n[test] X-Conversation-Id header is preferred over hash + body")
    msgs = [
        {"role": "system", "content": "you are a helpful assistant"},
        {"role": "user", "content": "hello"},
    ]
    body = {"metadata": {"chat_id": "from-body"}}
    headers = {"x-conversation-id": "abc-from-header"}
    cid, src = memory.resolve_conv_id(headers, msgs, body=body)
    assert_eq(cid, "abc-from-header", "uses header value")
    assert_eq(src, "header", "source=header")


def test_body_metadata_path():
    print("\n[test] body.metadata.chat_id used when no header present")
    msgs = [{"role": "user", "content": "hi"}]
    body = {"metadata": {"chat_id": "owui-uuid-12345"}}
    cid, src = memory.resolve_conv_id({}, msgs, body=body)
    assert_eq(cid, "owui-uuid-12345", "uses body chat_id")
    assert_eq(src, "body_metadata.chat_id", "source identifies body path")


def test_body_metadata_conversation_id_alias():
    print("\n[test] body.metadata.conversation_id also accepted")
    msgs = [{"role": "user", "content": "hi"}]
    body = {"metadata": {"conversation_id": "alt-name"}}
    cid, src = memory.resolve_conv_id({}, msgs, body=body)
    assert_eq(cid, "alt-name", "uses conversation_id alias")
    assert_eq(src, "body_metadata.conversation_id", "source identifies which key")


def test_hash_fallback():
    print("\n[test] sha256 fingerprint when no header or body hint")
    msgs = [
        {"role": "system", "content": "system prompt A"},
        {"role": "user", "content": "first user message"},
    ]
    cid, src = memory.resolve_conv_id({}, msgs)
    assert_eq(src, "hash", "source=hash")
    assert_true(len(cid) == 16, "hash truncated to 16 hex chars")
    assert_true(all(c in "0123456789abcdef" for c in cid), "hash is hex")

    # Same opening should produce same id.
    msgs2 = msgs + [{"role": "assistant", "content": "..."}]
    cid2, _ = memory.resolve_conv_id({}, msgs2)
    assert_eq(cid2, cid, "stable across added turns")

    # Different opening should produce different id.
    msgs3 = [
        {"role": "system", "content": "system prompt B"},  # different
        {"role": "user", "content": "first user message"},
    ]
    cid3, _ = memory.resolve_conv_id({}, msgs3)
    assert_true(cid3 != cid, "different system prompt -> different hash")


def test_sanitization_strips_unsafe_chars():
    print("\n[test] conv_id from header is sanitized to filesystem-safe charset")
    headers = {"x-conversation-id": "../../etc/passwd"}
    msgs = [{"role": "user", "content": "x"}]
    cid, src = memory.resolve_conv_id(headers, msgs)
    assert_eq(src, "header", "still treated as header source")
    assert_true("/" not in cid and "." not in cid, "path traversal stripped")
    assert_true(cid != "", "sanitization didn't produce empty string")


def test_empty_sanitized_header_falls_through():
    print("\n[test] header with only unsafe chars falls back to hash")
    headers = {"x-conversation-id": "////"}
    msgs = [{"role": "user", "content": "x"}]
    cid, src = memory.resolve_conv_id(headers, msgs)
    assert_eq(src, "hash", "empty-after-sanitize -> hash fallback")


def test_multimodal_message_hash():
    print("\n[test] hash handles multimodal content arrays")
    # OpenWebUI may send first_user as a list of content parts.
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": [
            {"type": "text", "text": "describe this image"},
            {"type": "image_url", "image_url": {"url": "https://..."}}
        ]},
    ]
    cid, src = memory.resolve_conv_id({}, msgs)
    assert_eq(src, "hash", "multimodal -> hash works")
    assert_true(len(cid) == 16, "still produces 16-char hex")


# ---------------------------------------------------------------------------
# Storage layout
# ---------------------------------------------------------------------------

def test_ensure_storage_layout_creates_subdirs():
    print("\n[test] ensure_storage_layout creates facts/, summaries/, chromadb/")
    # Wipe and recreate to test the create path.
    if os.path.exists(_TMP_ROOT):
        shutil.rmtree(_TMP_ROOT)
    memory.ensure_storage_layout()
    for sub in ("facts", "summaries", "chromadb"):
        path = os.path.join(_TMP_ROOT, sub)
        assert_true(os.path.isdir(path), f"{sub}/ exists")


def test_ensure_storage_layout_idempotent():
    print("\n[test] ensure_storage_layout is safe to call repeatedly")
    memory.ensure_storage_layout()
    memory.ensure_storage_layout()  # second call must not error
    print("  ok   idempotent")


def test_list_known_conv_ids_empty():
    print("\n[test] list_known_conv_ids returns [] on empty volume")
    memory.ensure_storage_layout()
    ids = memory.list_known_conv_ids()
    assert_eq(ids, [], "no convs -> empty list")


def test_list_known_conv_ids_picks_up_files():
    print("\n[test] list_known_conv_ids surfaces conv_ids from filenames")
    memory.ensure_storage_layout()
    # Drop a stub facts file and a stub summary file with different conv_ids.
    facts_dir = os.path.join(_TMP_ROOT, "facts")
    summaries_dir = os.path.join(_TMP_ROOT, "summaries")
    with open(os.path.join(facts_dir, "conv-a.json"), "w") as f:
        json.dump({"facts": []}, f)
    with open(os.path.join(summaries_dir, "conv-b.json"), "w") as f:
        json.dump({"l0": []}, f)
    ids = memory.list_known_conv_ids()
    assert_eq(ids, ["conv-a", "conv-b"], "union of facts/ + summaries/ filenames")


def test_admin_routes_registered_in_main():
    """Importing main.py should register the new V2.0 admin endpoints
    alongside the existing v1 OpenAI-compatible ones.
    """
    print("\n[test] main.py registers /admin/conversations endpoints")
    # main needs MODEL_REPO unset or a tokenizer-friendly default; we
    # already cleared MODEL_REPO in test_smoke.py's import, so this is
    # safe to import here too. import is idempotent.
    import main
    routes = {r.path for r in main.app.routes if hasattr(r, "path")}
    assert_true("/admin/conversations" in routes, "/admin/conversations registered")
    assert_true(
        "/admin/conversations/{conv_id}" in routes,
        "/admin/conversations/{conv_id} registered",
    )


def test_storage_summary_reports_presence():
    print("\n[test] storage_summary reports per-file existence + size")
    memory.ensure_storage_layout()
    cid = "conv-a"
    facts_file = memory.facts_path(cid)
    facts_file.parent.mkdir(parents=True, exist_ok=True)
    facts_file.write_text(json.dumps({"facts": ["x"]}))
    info = memory.storage_summary(cid)
    assert_eq(info["conv_id"], cid, "conv_id echoed")
    assert_true(info["facts"]["exists"], "facts file detected")
    assert_true(info["facts"]["size_bytes"] > 0, "size > 0")
    assert_true(not info["summary"]["exists"], "no summary file yet")


if __name__ == "__main__":
    try:
        test_header_path_wins()
        test_body_metadata_path()
        test_body_metadata_conversation_id_alias()
        test_hash_fallback()
        test_sanitization_strips_unsafe_chars()
        test_empty_sanitized_header_falls_through()
        test_multimodal_message_hash()
        test_ensure_storage_layout_creates_subdirs()
        test_ensure_storage_layout_idempotent()
        test_list_known_conv_ids_empty()
        test_list_known_conv_ids_picks_up_files()
        test_admin_routes_registered_in_main()
        test_storage_summary_reports_presence()
        print("\nAll memory smoke tests passed.")
    finally:
        # Clean up temp storage root
        if os.path.exists(_TMP_ROOT):
            shutil.rmtree(_TMP_ROOT, ignore_errors=True)
