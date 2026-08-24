"""
CPU-only Tier-1 tests for the backend-modality guard (v3.0.1).

The production failure: one uploaded image permanently poisoned a conversation
on a text-only backend. OpenWebUI re-sends the full history (image included)
with every message, V3.1 compaction deliberately preserves image turns, and
vLLM 400s every request ("is not a multimodal model") — so every message after
the upload failed, forever.

Fix under test: image parts are replaced with an honest placeholder when the
backend is text-only, and a vLLM not-multimodal 400 flips the cached modality
so the NEXT request heals even when startup detection was wrong.

Run inside the compactor image or any container with the requirements:
    python test_modality.py
"""

import os
import sys

os.environ.pop("MODEL_REPO", None)
os.environ["MAX_MODEL_LEN"] = "1000"

import main  # noqa: E402


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


IMG = {"type": "image_url", "image_url": {"url": "data:image/png;base64,xxxx"}}


def test_strip_image_parts():
    print("\n[test] _strip_image_parts — the poisoned-conversation shape")
    msgs = [
        {"role": "system", "content": "persona"},
        {"role": "user", "content": [{"type": "text", "text": "look at this!"}, IMG]},
        {"role": "assistant", "content": "I see."},
        {"role": "user", "content": "hi again"},
    ]
    out, n = main._strip_image_parts(msgs)
    assert_eq(n, 1, "one image stripped")
    assert_true(isinstance(out[1]["content"], str), "image message flattened to string")
    assert_true("look at this!" in out[1]["content"], "user's text preserved")
    assert_true("cannot" in out[1]["content"], "honest placeholder present")
    assert_eq(out[0], msgs[0], "system untouched")
    assert_eq(out[2], msgs[2], "assistant untouched")
    assert_eq(out[3], msgs[3], "plain user turn untouched")
    assert_eq(main._message_image_count(out[1]), 0, "no image parts remain")

    print("\n[test] _strip_image_parts — image-only message still yields text")
    out, n = main._strip_image_parts([{"role": "user", "content": [IMG]}])
    assert_eq(n, 1, "stripped")
    assert_true(out[0]["content"].startswith("["), "placeholder-only content")

    print("\n[test] _strip_image_parts — multiple images, correct wording")
    out, n = main._strip_image_parts([{"role": "user", "content": [IMG, IMG, {"type": "text", "text": "t"}]}])
    assert_eq(n, 2, "both counted")
    assert_true("2 images" in out[0]["content"], "plural wording")

    print("\n[test] _strip_image_parts — text-only content passes through")
    msgs = [
        {"role": "user", "content": "plain"},
        {"role": "user", "content": [{"type": "text", "text": "parts"}]},
    ]
    out, n = main._strip_image_parts(msgs)
    assert_eq(n, 0, "nothing stripped")
    assert_eq(out, msgs, "messages unchanged")


def test_modality_cache_and_backstop():
    print("\n[test] backend_is_multimodal — explicit False strips, and is sticky")
    main._backend_multimodal = False
    assert_eq(main.backend_is_multimodal(), False, "reports text-only")

    print("\n[test] _note_backend_rejection — flips the cache on the vLLM marker")
    main._backend_multimodal = True
    main._note_backend_rejection(
        '{"error":{"message":"some/model is not a multimodal model","type":"BadRequestError"}}'
    )
    assert_eq(main._backend_multimodal, False, "400 marker flips modality to text-only")

    print("\n[test] _note_backend_rejection — unrelated 400s do NOT flip it")
    main._backend_multimodal = True
    main._note_backend_rejection('{"error":{"message":"maximum context length exceeded"}}')
    assert_eq(main._backend_multimodal, True, "context-length 400 leaves modality alone")
    main._note_backend_rejection("")
    assert_eq(main._backend_multimodal, True, "empty body leaves modality alone")

    print("\n[test] backend_is_multimodal — no MODEL_REPO resolves to True (no stripping)")
    main._backend_multimodal = None
    assert_eq(main.backend_is_multimodal(), True, "unknown model assumed multimodal")


if __name__ == "__main__":
    test_strip_image_parts()
    test_modality_cache_and_backstop()
    print("\nAll modality-guard tests passed.")
