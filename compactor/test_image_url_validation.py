"""
CPU-only Tier-1 tests for invalid image_url rejection (v3.1.9.8, plus the
round-2 hostile-review fix round).

Regression cover for prod 09-26 07:33Z: vLLM rejected 4 consecutive requests
with `REQUEST REJECTED by vLLM (HTTP 400) ... The URL must be either a HTTP,
data or file URL` because a retained image carried a URL scheme vLLM 0.19
does not accept, and the compactor forwarded it unexamined. The client got
four empty replies in a row for it.

Covers:
  1. _valid_image_url's scheme allowlist: http/https/data: accepted
     case-insensitively; a well-formed `data:<mime>;base64,...` (mime
     optional -- vLLM accepts a bare `data:;base64,...` too); a trailing
     byte after the base64 payload (e.g. a stray newline) is refused, since
     vLLM's own base64.b64decode(validate=True) refuses it; file: refused
     (this deployment never sets --allowed-local-media-path).
  2. _demote_invalid_image_urls demotes only the bad part(s) and KEEPS a
     valid image (or any other surviving non-text part) in the message as
     a LIST -- round-2 review item 1 (HIGH): this used to collapse the
     WHOLE message to a plain-text string the moment any part was bad,
     silently discarding a valid image sharing the same message. Also:
     logs a closed shape class (uuid36/owui-files-path/http/data-malformed/
     other), never a raw scheme fragment or the URL itself (item 4); uses
     the vision-appropriate "could not be loaded" note, not the text-only
     one, on a multimodal backend (item 2).
  3. Ordering: invalid-URL demotion runs BEFORE image retention, so
     retention's MAX_RETAINED_IMAGES cap counts only genuinely valid
     images and never evicts a good OLDER image in favour of a broken
     NEWER one (item 3).
  4. End-to-end: a chat request with a bad-scheme image reaches the
     (stubbed) vLLM with the image already replaced by a placeholder, and
     the endpoint still returns 200.

Run inside the compactor image or any container with the requirements:
    python test_image_url_validation.py
"""

import logging
import os
import sys
import tempfile
from unittest.mock import patch

os.environ.pop("MODEL_REPO", None)  # -> backend_is_multimodal() == True
os.environ["MAX_MODEL_LEN"] = "20000"
os.environ["COMPACTOR_RAG_ENABLED"] = "false"
_TMP_ROOT = tempfile.mkdtemp(prefix="compactor-test-image-url-")
os.environ["COMPACTOR_STORAGE_ROOT"] = _TMP_ROOT

import main  # noqa: E402
import memory  # noqa: E402
import retrieval  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

retrieval._available = False
retrieval._embedder = None
retrieval._chroma_collection = None

memory.ensure_storage_layout()

client = TestClient(
    main.app, client=("127.0.0.1", 12346), raise_server_exceptions=False
)


def assert_true(cond, label):
    if not cond:
        print(f"FAIL {label}")
        sys.exit(1)
    print(f"  ok   {label}")


def assert_eq(a, b, label):
    if a != b:
        print(f"FAIL {label}: expected {b!r}, got {a!r}")
        sys.exit(1)
    print(f"  ok   {label}")


# ---------------------------------------------------------------------------
# Unit: _valid_image_url
# ---------------------------------------------------------------------------

def test_valid_image_url():
    print("\n[test] _valid_image_url scheme allowlist")
    assert_true(main._valid_image_url("http://example.com/x.png"), "http accepted")
    assert_true(main._valid_image_url("https://example.com/x.png"), "https accepted")
    assert_true(
        main._valid_image_url("HTTPS://example.com/x.png"),
        "scheme accepted case-insensitively (HTTPS://)",
    )
    assert_true(
        main._valid_image_url("Data:image/png;base64,QUJD"),
        "scheme accepted case-insensitively (Data:)",
    )
    assert_true(
        main._valid_image_url("data:image/png;base64,AAAA"),
        "well-formed data: accepted",
    )
    assert_true(
        main._valid_image_url("data:;base64,QUJD"),
        "a bare, untyped data: URL (no mime) is accepted -- vLLM itself accepts it",
    )
    assert_true(
        not main._valid_image_url("data:image/png;base64,QUJD\n"),
        "a trailing byte after the base64 payload (e.g. a stray newline) is "
        "refused -- vLLM's own base64.b64decode(validate=True) refuses it too",
    )
    assert_true(
        not main._valid_image_url("data:image/png,not-base64"),
        "data: missing base64 marker refused",
    )
    assert_true(
        not main._valid_image_url("file:///etc/passwd"),
        "file: refused -- --allowed-local-media-path is not set for this "
        "deployment's vLLM launch (VLLM_EXTRA_ARGS), so vLLM 0.19 400s on "
        "file: exactly like any other disallowed scheme",
    )
    assert_true(not main._valid_image_url("ftp://x/y.png"), "ftp refused")
    assert_true(not main._valid_image_url("blob:abcd"), "blob refused")
    assert_true(
        not main._valid_image_url("3f2a1234-5678-90ab-cdef-1234567890ab"),
        "a bare OpenWebUI-style file id (no scheme at all) refused",
    )
    assert_true(
        not main._valid_image_url("/api/v1/files/abc/content"),
        "an OpenWebUI files path (not a URL vLLM can fetch) refused",
    )
    assert_true(not main._valid_image_url(""), "empty string refused")
    assert_true(not main._valid_image_url(None), "non-string refused")
    assert_true(not main._valid_image_url({"url": "http://x"}), "dict refused")


def test_image_url_shape_class():
    print("\n[test] _image_url_shape_class: a closed, safe-to-log set of shapes")
    assert_eq(
        main._image_url_shape_class("3f2a1234-5678-90ab-cdef-1234567890ab"),
        "uuid36",
        "a bare file id classifies as uuid36",
    )
    assert_eq(
        main._image_url_shape_class("/api/v1/files/abc/content"),
        "owui-files-path",
        "an OpenWebUI files path classifies as owui-files-path",
    )
    assert_eq(main._image_url_shape_class("ftp://x/y.png"), "other", "ftp classifies as other")
    assert_eq(
        main._image_url_shape_class("data:image/png,not-base64"),
        "data-malformed",
        "a broken data: URL classifies as data-malformed",
    )
    assert_eq(main._image_url_shape_class(None), "other", "non-string classifies as other")


# ---------------------------------------------------------------------------
# Unit: _demote_invalid_image_urls
# ---------------------------------------------------------------------------

def test_demote_keeps_valid_image_alongside_invalid_one():
    print("\n[test] *** THE FIX (item 1, HIGH): a valid image survives beside "
          "a demoted invalid one, in the SAME message")
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look at both"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}},
                {"type": "image_url", "image_url": {"url": "ftp://bad/2.png"}},
            ],
        }
    ]
    out, n = main._demote_invalid_image_urls(msgs, multimodal=True)
    assert_eq(n, 1, "one image demoted")
    content = out[0]["content"]
    assert_true(isinstance(content, list), "content STAYS a list -- not collapsed to a string")
    image_parts = [c for c in content if isinstance(c, dict) and c.get("type") == "image_url"]
    assert_eq(len(image_parts), 1, "exactly one image_url part remains")
    assert_eq(
        image_parts[0]["image_url"]["url"], "data:image/png;base64,QUJD",
        "the SURVIVING part is the valid image, byte-identical",
    )
    text_parts = [c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"]
    joined = " ".join(text_parts)
    assert_true("look at both" in joined, "original text preserved")
    assert_true("ftp" not in joined, "the bad URL itself never appears in the text parts")
    assert_true(
        "could not be loaded" in joined,
        "the vision-appropriate note is present (multimodal=True)",
    )


def test_demote_collapses_to_string_only_when_nothing_but_text_remains():
    print("\n[test] a message with NO surviving image collapses to a plain string")
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this"},
                {"type": "image_url", "image_url": {"url": "ftp://bad/only.png"}},
            ],
        }
    ]
    out, n = main._demote_invalid_image_urls(msgs, multimodal=True)
    assert_eq(n, 1, "one image demoted")
    content = out[0]["content"]
    assert_true(isinstance(content, str), "content collapses to a string (nothing else to keep)")
    assert_true("what is this" in content, "original text preserved")
    assert_true("could not be loaded" in content, "vision-appropriate note present")


def test_demote_text_only_backend_uses_text_only_note():
    print("\n[test] multimodal=False uses the original text-only wording")
    msgs = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "ftp://bad.png"}},
    ]}]
    out, n = main._demote_invalid_image_urls(msgs, multimodal=False)
    assert_true("text-only" in out[0]["content"], "text-only note used when multimodal=False")


def test_demote_invalid_image_urls_leaves_all_good_alone():
    print("\n[test] _demote_invalid_image_urls: an all-good message is untouched")
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hi"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ],
        }
    ]
    out, n = main._demote_invalid_image_urls(msgs)
    assert_eq(n, 0, "nothing demoted")
    assert_true(out[0] is msgs[0], "message object passed through unchanged")


def test_demote_invalid_image_urls_never_logs_the_url_or_scheme():
    print("\n[test] a WARNING names a shape CLASS and length only -- never "
          "the URL, and never the raw scheme fragment either (item 4)")
    secret_url = "ftp://her-private-host/should-never-appear-in-logs.png"
    msgs = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": secret_url}},
    ]}]
    records = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = _Capture()
    lg = logging.getLogger("compactor")
    lg.addHandler(handler)
    try:
        main._demote_invalid_image_urls(msgs)
    finally:
        lg.removeHandler(handler)
    warnings = [r for r in records if "invalid image_url" in r]
    assert_true(len(warnings) == 1, "exactly one WARNING logged")
    assert_true("shape=other" in warnings[0], "a closed shape class is logged")
    assert_true(str(len(secret_url)) in warnings[0], "length is logged")
    assert_true(
        "ftp" not in warnings[0],
        "not even the raw scheme fragment reaches the log any more",
    )
    assert_true(
        "her-private-host" not in warnings[0] and secret_url not in warnings[0],
        "the URL content itself never reaches the log",
    )
    info_lines = [r for r in records if "image url demotion" in r]
    assert_true(len(info_lines) == 1, "a summary INFO line reports the counts")
    assert_true("1 invalid image(s) demoted" in info_lines[0], "demoted count logged")
    assert_true("0 valid image(s)" in info_lines[0], "valid-kept count logged (item 1)")


# ---------------------------------------------------------------------------
# End-to-end: bad-scheme image reaches the stub vLLM already demoted, 200 OK.
# Mirrors test_budget_guard.py's _StubVLLM / _post_chat shape.
# ---------------------------------------------------------------------------

class _StubResponse:
    status_code = 200
    text = ""

    def json(self):
        return {
            "id": "stub",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
        }


class _StubVLLM:
    sent: list = []

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None, **kwargs):
        _StubVLLM.sent.append(json)
        return _StubResponse()

    def stream(self, *args, **kwargs):
        raise AssertionError("this test drives the non-streaming path only")

    async def aclose(self):
        pass


def _swallow_tail(coro, label=None):
    try:
        coro.close()
    except Exception:
        pass


def test_bad_scheme_image_reaches_stub_vllm_demoted_and_returns_200():
    print("\n[test] e2e: bad-scheme image never reaches vLLM raw, request still 200s")
    _StubVLLM.sent.clear()
    bad_url = "ftp://evil-host/payload.png"
    body = {
        "model": "stub-model",
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this"},
                    {"type": "image_url", "image_url": {"url": bad_url}},
                ],
            }
        ],
    }
    with patch.object(main.httpx, "AsyncClient", _StubVLLM), \
         patch.object(main, "_fire_and_forget", _swallow_tail):
        r = client.post(
            "/v1/chat/completions",
            json=body,
            headers={"X-Conversation-Id": "img-url-e2e-conv"},
        )
    assert_eq(r.status_code, 200, "endpoint returns 200")
    assert_true(_StubVLLM.sent, "the request reached the stub vLLM at all")
    forwarded = _StubVLLM.sent[-1]
    forwarded_text = str(forwarded)
    assert_true(bad_url not in forwarded_text, "the bad URL was never forwarded to vLLM")
    assert_true(
        "could not be loaded" in forwarded_text,
        "the vision-appropriate placeholder was forwarded instead",
    )


def test_valid_image_reaches_stub_vllm_beside_a_demoted_invalid_one():
    print("\n[test] e2e (item 1): a valid image in the SAME message as an "
          "invalid one still reaches vLLM")
    _StubVLLM.sent.clear()
    good_b64 = "data:image/png;base64," + ("A" * 40)
    body = {
        "model": "stub-model",
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "two photos"},
                    {"type": "image_url", "image_url": {"url": good_b64}},
                    {"type": "image_url", "image_url": {"url": "ftp://bad/2.png"}},
                ],
            }
        ],
    }
    with patch.object(main.httpx, "AsyncClient", _StubVLLM), \
         patch.object(main, "_fire_and_forget", _swallow_tail):
        r = client.post(
            "/v1/chat/completions",
            json=body,
            headers={"X-Conversation-Id": "img-url-e2e-mixed-conv"},
        )
    assert_eq(r.status_code, 200, "endpoint returns 200")
    forwarded_text = str(_StubVLLM.sent[-1])
    assert_true(good_b64 in forwarded_text, "the VALID image reaches vLLM, not just the note")
    assert_true("ftp://bad" not in forwarded_text, "the invalid URL is not forwarded")


# ---------------------------------------------------------------------------
# Ordering (item 3): invalid-URL demotion must run BEFORE image retention,
# so a small MAX_RETAINED_IMAGES never evicts a good OLDER image in favour
# of a broken NEWER one that was always going to be demoted anyway.
# Default COMPACTOR_MAX_RETAINED_IMAGES is 1 (unset here), matching prod.
# ---------------------------------------------------------------------------

def test_retention_never_evicts_a_good_image_for_a_broken_newer_one():
    print("\n[test] *** THE FIX (item 3): the OLDER valid image survives "
          "retention when the NEWER image's URL is invalid")
    # A true e2e test through chat_completions -- this is what actually
    # pins the CALL-SITE order (main.py's `_demote_invalid_image_urls`
    # before `_apply_image_retention`), not just the general behaviour of
    # composing the two functions one way or the other. GENERATION_RESERVE
    # is patched down for this one request so the hard-budget guard (a
    # separate mechanism) has plenty of room and never sheds either turn --
    # otherwise a shed old turn would make this test about the guard, not
    # about which image retention kept.
    assert_eq(main.MAX_RETAINED_IMAGES, 1, "fixture assumption: default cap is 1")
    old_good = "data:image/png;base64," + ("B" * 40)
    body = {
        "model": "stub-model",
        "stream": False,
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "first photo"},
                {"type": "image_url", "image_url": {"url": old_good}},
            ]},
            {"role": "assistant", "content": "nice"},
            {"role": "user", "content": [
                {"type": "text", "text": "second photo"},
                {"type": "image_url", "image_url": {"url": "ftp://bad/newer.png"}},
            ]},
        ],
    }
    _StubVLLM.sent.clear()
    with patch.object(main, "GENERATION_RESERVE", 500), \
         patch.object(main.httpx, "AsyncClient", _StubVLLM), \
         patch.object(main, "_fire_and_forget", _swallow_tail):
        r = client.post(
            "/v1/chat/completions",
            json=body,
            headers={"X-Conversation-Id": "img-url-ordering-conv"},
        )
    assert_eq(r.status_code, 200, "endpoint returns 200")
    forwarded_text = str(_StubVLLM.sent[-1])
    assert_true(
        old_good in forwarded_text,
        "the OLDER, VALID image reaches vLLM -- retention counted only "
        "valid images once the broken newer one was already demoted (this "
        "depends on main.py's actual call-site order, not just the "
        "functions' general behaviour)",
    )


def test_verification_review_followups():
    print("\n[test] verification review: note grammar, exact ;base64, marker, image-only mime")
    msgs = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "ftp://a.png"}},
        {"type": "image_url", "image_url": {"url": "ftp://b.png"}},
    ]}]
    out, n = main._demote_invalid_image_urls(msgs, multimodal=True)
    assert_eq(n, 2, "two images demoted")
    txt = out[0]["content"]
    assert_true("2 images here, but they could not be loaded" in txt,
                "plural note uses the subject pronoun 'they', not 'them'")
    assert_true("describe them" in txt, "object pronoun stays 'them'")
    out1, _ = main._demote_invalid_image_urls(msgs[:1] and [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "ftp://a.png"}}]}], multimodal=True)
    assert_true("but it could not be loaded" in out1[0]["content"], "singular note says 'it'")
    # vLLM 0.19 connector.py compares data_type != "base64" case-sensitively.
    assert_true(not main._valid_image_url("data:image/png;BASE64,QUJD"),
                "an upper-case ;BASE64, marker is refused (vLLM would refuse it too)")
    assert_true(main._valid_image_url("DATA:IMAGE/PNG;base64,QUJD"),
                "the scheme and mime type still fold case")
    assert_true(not main._valid_image_url("data:application/pdf;base64,QUJD"),
                "a non-image mime is refused (PIL would fail on it inside vLLM)")
    assert_true(not main._valid_image_url("data:text/html;base64,QUJD"), "text/html refused")
    assert_true(main._valid_image_url("data:;base64,QUJD"), "an empty mime is still accepted")
    assert_true(main._valid_image_url("data:application/octet-stream;base64,QUJD"),
                "application/octet-stream is accepted: vLLM ignores the mime for images and PIL "
                "reads the bytes (RC review A: one of her uploads is labelled this way)")


if __name__ == "__main__":
    test_valid_image_url()
    test_image_url_shape_class()
    test_demote_keeps_valid_image_alongside_invalid_one()
    test_demote_collapses_to_string_only_when_nothing_but_text_remains()
    test_demote_text_only_backend_uses_text_only_note()
    test_demote_invalid_image_urls_leaves_all_good_alone()
    test_demote_invalid_image_urls_never_logs_the_url_or_scheme()
    test_bad_scheme_image_reaches_stub_vllm_demoted_and_returns_200()
    test_valid_image_reaches_stub_vllm_beside_a_demoted_invalid_one()
    test_retention_never_evicts_a_good_image_for_a_broken_newer_one()
    test_verification_review_followups()
    print("\nAll image-url-validation tests passed.")
