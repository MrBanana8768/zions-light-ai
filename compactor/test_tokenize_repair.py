"""
v3.1.5 — one cancelled stream must not take every budget decision with it.

THE PRODUCTION FAILURE, conv <redacted>, 2026-08-30 to 08-31. A stream was
cancelled and OpenWebUI stored an EMPTY assistant turn, then resent it with
every subsequent message. vLLM's chat template refuses empty assistant
content outright:

    /tokenize -> HTTP 400 "Invalid assistant message: role='assistant'
                 content='' tool_calls=None prefix=False"

count_tokens_exact returned None, and every caller fell back to the local
tokenizer, which reads 34-51% low on this model's assistant content. The
consequences all landed at once and none of them named the cause:

  * summarize took the PESSIMISTIC 2.0x fallback, so its batch estimate
    jumped 32 -> 69 calls, past MAX_SUMMARY_CALLS_PER_REQUEST, and
    request-path compaction switched itself off. 24 hours of
    "[NO SUMMARIZATION HAPPENED]".
  * _enforce_hard_budget fell back to scale 1.0 and sheds on a counter it
    had just failed to check — the 2026-08-28 signature exactly.
  * /health/full pinned at ok:false deployment-wide, because the tokenize
    fail streak is a process global, making a real outage indistinguishable.

`_repair_template_invalid_tail` had fixed this shape since v3.1.4 — but it
runs at the END of the request path, after compact_if_needed and after
_enforce_hard_budget have both already measured. It repaired what we
FORWARD and never what we MEASURE. The rule was written once and applied at
one of the two places that needed it: this branch's recurring defect, and
the reason the fix here is a SHARED helper rather than a second copy.

WHAT THIS ASSERTS
  [1] The stub refuses the raw history the way vLLM really does — so the
      test would fail against the pre-fix code rather than passing vacuously.
  [2] count_tokens_exact returns a real count anyway.
  [3] The caller's list is NOT mutated: the empty turn is still there for
      _repair_template_invalid_tail to handle, and what we forward to the
      model is decided by that repair, not silently by a counter.
  [4] Multimodal content is never touched — a list part can read as
      text-empty while carrying an image, and destroying an image to satisfy
      a template rule is worse than the 400 it avoids.

Only synthetic conversation content appears below (project rule: this repo
is public).

    python test_tokenize_repair.py
"""

import os
import sys
import tempfile
from unittest.mock import patch

os.environ.setdefault("MODEL_REPO", "test-model")
os.environ.setdefault("VLLM_URL", "http://stub:8000")
os.environ["COMPACTOR_RAG_ENABLED"] = "false"
os.environ["COMPACTOR_STORAGE_ROOT"] = tempfile.mkdtemp(prefix="zions-tokrepair-")

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


# ---------------------------------------------------------------------------
# A stub that refuses the way vLLM 0.19's mistral template really refuses.
# ---------------------------------------------------------------------------

_seen: list = []


class _Resp:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


def _assistant_content_is_empty_to_vllm(m: dict) -> bool:
    """Whether vLLM would report this assistant turn as `content=''`.

    FOUR SHAPES, not one. Production 2026-09-01..09-02 logged twenty
    /tokenize 400s, all with the identical body `content=''` — and the str
    case was only ever one of the ways to arrive there. `None`, a missing
    content key, `[]` and a list of blank text parts all normalise to the
    same empty content server-side and are refused the same way, and the
    log cannot tell them apart.

    An earlier version of this stub modelled only `content == ""`. That is
    the LAXER direction of stub error: the three shapes the helper did not
    yet cover were waved through here, so a test exercising them would have
    passed against code that fails in production. The stub has to refuse
    everything the server refuses, or it certifies the bug.

    A list carrying a NON-TEXT part is NOT empty — it has an image in it,
    which is real content, and that asymmetry is what the helper's carve-out
    protects."""
    content = m.get("content")
    if content is None:
        return True
    if isinstance(content, str):
        return content == ""
    if isinstance(content, list):
        for p in content:
            if not isinstance(p, dict) or p.get("type") != "text":
                return False
            if str(p.get("text") or ""):
                return False
        return True
    return False


def _stub_post(url, json=None, timeout=None, **kw):
    """400 on an assistant message vLLM would see as empty, otherwise a
    count. Records what it was sent so the test can prove the sanitised copy
    is what got measured.

    `== ""` and NOT `.strip()`. vLLM 0.19's template refuses empty content
    and ACCEPTS a single space — that asymmetry is the entire basis of the
    space-fill, verified against the real MistralTokenizer pipeline in the
    production image (testfixtures/tokenizer-contract/vllm_template_probe.py).
    This stub first used `.strip()`, which made it refuse the space too:
    stricter than the thing it models, so the fix under test looked broken
    when it was working exactly as designed. A stub that is harsher than
    production fails good code — the same class of error as one that is
    laxer letting bad code through, and harder to spot because the failure
    looks like a real finding."""
    msgs = (json or {}).get("messages", [])
    _seen.append(msgs)
    for m in msgs:
        if m.get("role") == "assistant" and _assistant_content_is_empty_to_vllm(m):
            return _Resp(
                400,
                {
                    "error": {
                        "message": "Invalid assistant message: role='assistant' "
                        "content='' tool_calls=None prefix=False",
                        "type": "BadRequestError",
                    }
                },
            )
    return _Resp(200, {"count": 4242})


def _history():
    """User-final, with an INTERIOR empty assistant turn — the shape a
    cancelled stream leaves and OpenWebUI then resends forever."""
    return [
        {"role": "system", "content": "persona"},
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": ""},          # the cancelled stream
        {"role": "user", "content": "second question"},
    ]


# ---------------------------------------------------------------------------


def test_stub_really_refuses_the_raw_history():
    print("\n[test] TEETH — the stub 400s on the unsanitised history")
    # Without this, every assertion below could pass against a stub that
    # never refuses anything, and the test would prove nothing.
    _seen.clear()
    r = _stub_post("u", json={"messages": _history()})
    assert_eq(r.status_code, 400, "raw history is refused, as vLLM refuses it")
    r2 = _stub_post("u", json={"messages": [{"role": "user", "content": "hi"}]})
    assert_eq(r2.status_code, 200, "a clean history is accepted")


def test_count_survives_an_empty_assistant_turn():
    print("\n[test] count_tokens_exact returns a count despite the empty turn")
    _seen.clear()
    msgs = _history()
    with patch.object(main.httpx, "post", _stub_post):
        got = main.count_tokens_exact(msgs)
    assert_eq(got, 4242, "a real count, not None (None is the fallback bug)")
    assert_true(_seen, "the endpoint was actually called")
    sent = _seen[-1]
    assert_eq(sent[2]["content"], " ", "the MEASURED copy was space-filled")


def test_the_callers_list_is_not_mutated():
    print("\n[test] measuring does not change what we forward")
    # The repair at the end of the request path owns what gets forwarded.
    # A counter that quietly edited the payload would be changing the
    # conversation as a side effect of measuring it.
    _seen.clear()
    msgs = _history()
    with patch.object(main.httpx, "post", _stub_post):
        main.count_tokens_exact(msgs)
    assert_eq(msgs[2]["content"], "", "the empty turn is still empty in the caller's list")
    assert_eq(len(msgs), 4, "no turn added or removed")


def test_an_image_bearing_turn_is_never_touched():
    print("\n[test] a turn carrying an image is left alone, even if its text is blank")
    # The carve-out, stated precisely: emptiness must be PROVEN, and a list
    # holding an image is not empty however blank its text reads. Destroying
    # an image to satisfy a template rule is worse than the 400 it avoids.
    img = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    msgs = [
        {"role": "user", "content": [{"type": "text", "text": ""}, img]},
        {"role": "assistant", "content": [{"type": "text", "text": ""}, img]},
        {"role": "user", "content": "and now?"},
    ]
    out, filled = main._space_fill_empty_assistant(msgs)
    assert_eq(filled, 0, "nothing space-filled — the assistant turn carries an image")
    assert_true(out[0] is msgs[0], "the user image turn is the same object, untouched")
    assert_true(out[1] is msgs[1], "the assistant image turn is untouched too")


def test_an_unrecognised_part_counts_as_content():
    print("\n[test] a part we cannot classify is treated as content, not as empty")
    # Unknown content is treated as content. Guessing "probably nothing" about
    # a part shape we do not recognise is how an image gets destroyed by a
    # future client change nobody here has seen yet.
    for label, part in (
        ("a part with no type", {"text": ""}),
        ("a non-dict part", "just a string"),
        ("an unknown type", {"type": "audio_url", "audio_url": {"url": "x"}}),
    ):
        msgs = [{"role": "assistant", "content": [part]}]
        out, filled = main._space_fill_empty_assistant(msgs)
        assert_eq(filled, 0, f"{label} -> left alone")
        assert_true(out[0] is msgs[0], f"{label} -> same object")


def test_every_empty_shape_a_client_can_send_is_repaired():
    print("\n[test] None, a missing key, [] and blank text parts all get filled")
    # v3.1.7. Production ran twenty identical 400s whose bodies all read
    # `content=''`; the str case was one of four ways to get there and the
    # only one the helper covered. Each shape is driven END TO END through
    # count_tokens_exact against a stub that refuses it, so a pass means the
    # measurement actually survived rather than that the helper looked right.
    shapes = {
        "content=None": {"role": "assistant", "content": None},
        "no content key": {"role": "assistant"},
        "content=[]": {"role": "assistant", "content": []},
        # EXACTLY empty, not whitespace. A whitespace-only text part reduces
        # to whitespace content, which the template ACCEPTS — the same
        # asymmetry the space-fill is built on. Using "   " here made this
        # case assert that the stub refuses something vLLM allows, which
        # would have been a test demanding a fix for a non-bug.
        "content=[empty text part]": {
            "role": "assistant",
            "content": [{"type": "text", "text": ""}],
        },
    }
    for label, dead_turn in shapes.items():
        msgs = [
            {"role": "system", "content": "persona"},
            {"role": "user", "content": "first question"},
            dict(dead_turn),
            {"role": "user", "content": "second question"},
        ]
        # TEETH: the stub must refuse this shape, or the assertion below is
        # vacuous and would pass against the pre-fix code.
        assert_eq(
            _stub_post("u", json={"messages": msgs}).status_code, 400,
            f"{label}: the stub refuses it, as vLLM does",
        )
        _seen.clear()
        with patch.object(main.httpx, "post", _stub_post):
            got = main.count_tokens_exact(msgs)
        assert_eq(got, 4242, f"{label}: a real count, not None")
        assert_eq(_seen[-1][2]["content"], " ", f"{label}: the measured copy was filled")
        assert_true(
            msgs[2].get("content") == dead_turn.get("content"),
            f"{label}: the caller's list is untouched",
        )


def test_the_forward_repair_does_not_destroy_a_trailing_image():
    print("\n[test] an image on the LAST assistant turn is not dropped as 'empty'")
    # THE DROP AND THE FILL ARE SIBLINGS. Step (1) of the forward repair pops
    # a trailing empty assistant turn; step (1b) space-fills empty ones. They
    # had different ideas of "empty": the pop used _message_text().strip(),
    # which joins TEXT parts and ignores images, so an image-only assistant
    # turn read as empty and was popped — the image destroyed, permanently
    # and silently — three lines before the fill refused to touch that exact
    # shape. Both now share assistant_content_is_empty.
    img = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    for label, tail in (
        ("[blank text, image]", [{"type": "text", "text": ""}, img]),
        ("[image] alone", [img]),
    ):
        body = {
            "messages": [
                {"role": "user", "content": "what is this?"},
                {"role": "assistant", "content": list(tail)},
            ]
        }
        main._repair_template_invalid_tail(body)
        kept = body["messages"]
        images = sum(main._message_image_count(m) for m in kept)
        assert_eq(len(kept), 2, f"{label}: the turn is NOT popped")
        assert_eq(images, 1, f"{label}: the image survives the repair")

    # The genuine residue of a dead stream must still be dropped, or this
    # test would pass against a repair that simply stopped working.
    body = {
        "messages": [
            {"role": "user", "content": "what is this?"},
            {"role": "assistant", "content": ""},
        ]
    }
    main._repair_template_invalid_tail(body)
    assert_eq(len(body["messages"]), 1, "a truly empty trailing turn is still dropped")


def test_one_emptiness_rule_covers_every_shape():
    print("\n[test] assistant_content_is_empty — the shared predicate")
    empty = (None, "", "   ", [], [{"type": "text", "text": ""}],
             [{"type": "text", "text": "  "}])
    for c in empty:
        assert_true(main.assistant_content_is_empty(c), f"empty: {c!r}")
    img = {"type": "image_url", "image_url": {"url": "data:x"}}
    # Content, every one of them — including the shapes we cannot read. A
    # truthy non-string raised AttributeError in tokens._sanitize before this
    # was shared; a falsy non-string is still not ours to overwrite.
    content = ("hi", [{"type": "text", "text": "hi"}], [img],
               [{"type": "text", "text": ""}, img], [{"text": "no type"}],
               ["not a dict"], 5, 0, {"a": 1}, {})
    for c in content:
        assert_true(not main.assistant_content_is_empty(c), f"content: {c!r}")


def test_the_local_tokenizer_reduction_has_the_same_rule():
    print("\n[test] tokens._sanitize does not manufacture the refused string")
    # THE TWIN. tokens.py is tier 2 — the fallback used when /tokenize cannot
    # answer. Its reduction ended in `content or ""`, which turns every one of
    # the shapes above into exactly the empty assistant string MistralTokenizer
    # refuses. That made the fallback fail on precisely the payloads that make
    # tier 1 fail: the one situation it exists for.
    import tokens

    for label, content in (
        ("empty str", ""),
        ("whitespace str", "   "),
        ("None", None),
        ("empty list", []),
        ("blank text part", [{"type": "text", "text": ""}]),
    ):
        out = tokens._sanitize([{"role": "assistant", "content": content}])
        assert_eq(out[0]["content"], " ", f"assistant {label} -> a single space")
    # And it must not invent content for a role the template does not refuse.
    out = tokens._sanitize([{"role": "user", "content": ""}])
    assert_eq(out[0]["content"], "", "an empty USER turn is left empty")
    # NON-STRING CONTENT MUST NOT RAISE. `content or ""` then `.strip()`
    # raised AttributeError on an int or a dict. tokens.count wraps this in a
    # blanket except, so the raise crashed nothing — it silently made tier 2
    # unavailable and handed the budget back to the 51%-low estimator, which
    # is the outage this module exists to prevent. A counter that degrades
    # silently is the failure mode this project keeps paying for.
    for label, c in (("int", 5), ("dict", {"a": 1}), ("falsy int", 0), ("falsy dict", {})):
        out = tokens._sanitize([{"role": "assistant", "content": c}])
        assert_true(isinstance(out[0]["content"], str), f"{label} -> a string, no raise")
        assert_true(out[0]["content"].strip(), f"{label} -> not blanked into a space")
    out = tokens._sanitize(["not a dict at all"])
    assert_eq(out, [], "a non-dict message is skipped, not raised on")
    # Real text still survives the reduction unchanged.
    out = tokens._sanitize(
        [{"role": "assistant", "content": [{"type": "text", "text": "kept"}]}]
    )
    assert_eq(out[0]["content"], "kept", "real text is reduced, not replaced")


def test_repair_and_counter_share_one_implementation():
    print("\n[test] the forward-repair uses the same helper as the counter")
    # The whole point of the fix: two sites, one rule. If _repair_template_
    # invalid_tail grows its own copy again, this conversation's failure
    # comes back the moment the two disagree.
    body = {"messages": _history()}
    note, invalid = main._repair_template_invalid_tail(body)
    assert_true(invalid, "the repair recognises this shape as invalid")
    assert_eq(body["messages"][2]["content"], " ", "and space-fills it for forwarding")
    assert_true("space-filled" in (note or ""), f"and says so: {note!r}")


if __name__ == "__main__":
    import shutil

    try:
        for t in (
            test_stub_really_refuses_the_raw_history,
            test_count_survives_an_empty_assistant_turn,
            test_the_callers_list_is_not_mutated,
            test_an_image_bearing_turn_is_never_touched,
            test_an_unrecognised_part_counts_as_content,
            test_every_empty_shape_a_client_can_send_is_repaired,
            test_the_forward_repair_does_not_destroy_a_trailing_image,
            test_one_emptiness_rule_covers_every_shape,
            test_the_local_tokenizer_reduction_has_the_same_rule,
            test_repair_and_counter_share_one_implementation,
        ):
            t()
        print("\nAll tokenize-repair tests passed.")
    finally:
        shutil.rmtree(os.environ["COMPACTOR_STORAGE_ROOT"], ignore_errors=True)
