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


def _stub_post(url, json=None, timeout=None, **kw):
    """400 on an assistant message whose string content is EXACTLY empty,
    otherwise a count. Records what it was sent so the test can prove the
    sanitised copy is what got measured.

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
        if (
            m.get("role") == "assistant"
            and isinstance(m.get("content"), str)
            and m["content"] == ""
        ):
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


def test_multimodal_content_is_never_touched():
    print("\n[test] an image-bearing turn is left alone")
    img = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    msgs = [
        {"role": "user", "content": [{"type": "text", "text": ""}, img]},
        {"role": "assistant", "content": [{"type": "text", "text": ""}]},
        {"role": "user", "content": "and now?"},
    ]
    out, filled = main._space_fill_empty_assistant(msgs)
    assert_eq(filled, 0, "nothing space-filled — no str-content empties here")
    assert_true(out[0] is msgs[0], "the image turn is the same object, untouched")
    assert_eq(out[1]["content"], [{"type": "text", "text": ""}],
              "text-empty LIST assistant content left as-is, image safety over template tidiness")


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
            test_multimodal_content_is_never_touched,
            test_repair_and_counter_share_one_implementation,
        ):
            t()
        print("\nAll tokenize-repair tests passed.")
    finally:
        shutil.rmtree(os.environ["COMPACTOR_STORAGE_ROOT"], ignore_errors=True)
