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




IMG2 = {"type": "image_url", "image_url": {"url": "data:image/png;base64,yyyy"}}


def test_merge_consecutive_same_role():
    print("\n[test] _merge_consecutive_same_role — the compaction hoist scenario")
    # compact_if_needed emits system + summary + preserved_images + keep_recent,
    # so a hoisted image turn lands next to the recent window's leading user
    # turn: user,user -> Mistral 400. This is the v3.0.2 bug.
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "system", "content": "[Summary of earlier conversation]"},
        {"role": "user", "content": [{"type": "text", "text": "see this"}, IMG]},
        {"role": "user", "content": "u3"},
        {"role": "assistant", "content": "a3"},
        {"role": "user", "content": "u4"},
    ]
    out = main._merge_consecutive_same_role(msgs)
    roles = [m["role"] for m in out if m.get("role") != "system"]
    assert_eq(roles, ["user", "assistant", "user"], "alternation restored")
    for a, b in zip(roles, roles[1:]):
        assert_true(a != b, f"no consecutive same role ({a}->{b})")

    print("\n[test] _merge_consecutive_same_role — the image survives the merge")
    merged_user = [m for m in out if m.get("role") == "user"][0]
    assert_eq(main._message_image_count(merged_user), 1, "image part preserved")
    txt = main._message_text(merged_user)
    assert_true("see this" in txt and "u3" in txt, "both texts preserved")

    print("\n[test] _merge_consecutive_same_role — text-only merge joins content")
    out = main._merge_consecutive_same_role(
        [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]
    )
    assert_eq(len(out), 1, "two user turns merged")
    assert_true("a" in out[0]["content"] and "b" in out[0]["content"], "text joined")

    print("\n[test] _merge_consecutive_same_role — two images both kept")
    out = main._merge_consecutive_same_role([
        {"role": "user", "content": [IMG]},
        {"role": "user", "content": [IMG2]},
    ])
    assert_eq(len(out), 1, "merged")
    assert_eq(main._message_image_count(out[0]), 2, "both images retained")

    print("\n[test] _merge_consecutive_same_role — already-alternating is untouched")
    good = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "u2"},
    ]
    assert_eq(main._merge_consecutive_same_role(good), good, "no-op passthrough")

    print("\n[test] _merge_consecutive_same_role — system runs left to the other merger")
    sys2 = [{"role": "system", "content": "a"}, {"role": "system", "content": "b"}]
    assert_eq(len(main._merge_consecutive_same_role(sys2)), 2, "systems not merged here")


def test_image_retention_cap():
    print("\n[test] compact_if_needed — image retention cap bounds accumulation")
    import asyncio

    seen_turns = []

    async def fake_summarize(client, turns):
        seen_turns.extend(turns)
        return "SUMMARY"

    orig_sum, orig_keep, orig_cap = main.summarize, main.KEEP_RECENT_TURNS, main.MAX_RETAINED_IMAGES
    orig_target = main.TARGET_TOKENS
    main.summarize = fake_summarize
    main.KEEP_RECENT_TURNS = 2
    main.MAX_RETAINED_IMAGES = 2
    main.TARGET_TOKENS = 1  # force compaction
    try:
        msgs = [{"role": "system", "content": "sys"}]
        for i in range(5):  # 5 image turns + replies
            msgs.append({"role": "user", "content": [{"type": "text", "text": f"pic{i}"}, IMG]})
            msgs.append({"role": "assistant", "content": f"reply{i}"})
        msgs.append({"role": "user", "content": "final"})
        out = asyncio.run(main.compact_if_needed(msgs))
        total_imgs = sum(main._message_image_count(m) for m in out)
        assert_true(total_imgs <= 2, f"images capped at 2 (got {total_imgs})")
        # The demoted note goes INTO the summarizer input (it is summarized with
        # the rest), so the trace is verified there — not in the output, which a
        # stubbed summarizer replaces with a fixed string.
        summarized = " ".join(main._message_text(m) for m in seen_turns)
        assert_true("image shared earlier" in summarized, "demoted images reach the summarizer as text")
        assert_eq(out[-1].get("content"), "final", "newest turn preserved")
    finally:
        main.summarize, main.KEEP_RECENT_TURNS = orig_sum, orig_keep
        main.MAX_RETAINED_IMAGES, main.TARGET_TOKENS = orig_cap, orig_target




def test_memorable_user_text():
    print("\n[test] _memorable_user_text — image-only turn becomes memorable")
    # A bare upload has no text, so index_exchange and the facts tail both
    # refuse it (correctly) — leaving no durable trace a picture was shared.
    msgs = [{"role": "user", "content": [IMG]}]
    assert_eq(main._memorable_user_text(msgs, ""), "[shared 1 image]", "marker substituted")

    print("\n[test] _memorable_user_text — plural wording")
    assert_eq(
        main._memorable_user_text([{"role": "user", "content": [IMG, IMG2]}], "   "),
        "[shared 2 images]",
        "plural marker (whitespace-only counts as empty)",
    )

    print("\n[test] _memorable_user_text — real caption always wins")
    msgs = [{"role": "user", "content": [{"type": "text", "text": "my dress"}, IMG]}]
    assert_eq(main._memorable_user_text(msgs, "my dress"), "my dress", "caption preserved")

    print("\n[test] _memorable_user_text — no images: empty stays empty")
    assert_eq(main._memorable_user_text([{"role": "user", "content": ""}], ""), "", "no marker invented")
    assert_eq(main._memorable_user_text([], ""), "", "empty conversation safe")

    print("\n[test] _memorable_user_text — uses the LAST user turn")
    msgs = [
        {"role": "user", "content": [IMG]},
        {"role": "assistant", "content": "nice"},
        {"role": "user", "content": ""},
    ]
    assert_eq(main._memorable_user_text(msgs, ""), "", "latest turn has no image -> no marker")


if __name__ == "__main__":
    test_strip_image_parts()
    test_modality_cache_and_backstop()
    test_merge_consecutive_same_role()
    test_image_retention_cap()
    test_memorable_user_text()
    print("\nAll vision-path tests passed.")
