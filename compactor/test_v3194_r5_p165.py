"""
v3.1.9.4 R5 (hostile pass #16 follow-up): P16-5.

B4 taught the STREAMING accumulator (`SseAccumulator.feed`/`complete()`)
that `finish_reason: "error"` is an in-band cut, not a whole reply — a 200
whose choice carries this value mid-body, not an HTTP error status. The
non-streaming sibling call site in `chat_completions` was never updated to
match: it passed `finished=True` unconditionally, so a non-streaming 200
whose choice has `finish_reason: "error"` was stored as a complete,
finished reply — the exact misclassification B4 removed from the
streaming path, left in its sibling.

Fix: `finished=_finish_reason != "error"` at the non-streaming call site,
mirroring `SseAccumulator.complete()`'s own `self._complete and not
self._errored`.

Drives the REAL `/v1/chat/completions` non-streaming endpoint (FastAPI
TestClient, vLLM stubbed at httpx.AsyncClient — same shape
test_budget_guard.py's endpoint-level section uses) and spies on
`main._run_memory_tail` to observe exactly what `finished=` it was called
with, without letting the real background tail touch the storage volume.

Run: python test_v3194_r5_p165.py
"""

import os
import sys
import tempfile
from unittest.mock import patch

os.environ.pop("MODEL_REPO", None)
os.environ["COMPACTOR_RAG_ENABLED"] = "false"
os.environ["COMPACTOR_STORAGE_ROOT"] = tempfile.mkdtemp(prefix="compactor-test-v3194-r5-p165-")

import main  # noqa: E402
import memory  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

memory.ensure_storage_layout()

client = TestClient(main.app, client=("127.0.0.1", 12345), raise_server_exceptions=False)

FAILED: list[str] = []


def check(cond, label):
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"FAIL {label}")
        FAILED.append(label)


class _StubResponse:
    status_code = 200
    text = ""

    def __init__(self, content, finish_reason):
        self._content = content
        self._finish_reason = finish_reason

    def json(self):
        return {
            "id": "stub",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": self._content},
                    "finish_reason": self._finish_reason,
                }
            ],
        }


class _StubVLLM:
    def __init__(self, *a, **kw):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None, **kwargs):
        return _StubResponse(_NEXT["content"], _NEXT["finish_reason"])

    def stream(self, *a, **kw):
        raise AssertionError("these tests drive the non-streaming path only")

    async def aclose(self):
        pass


_NEXT = {"content": "", "finish_reason": "stop"}
_CAPTURED: dict = {}


def _spy_run_memory_tail(*args, **kwargs):
    _CAPTURED.clear()
    _CAPTURED.update(kwargs)
    return None


def _post_chat(conv_id, content, finish_reason):
    _NEXT["content"] = content
    _NEXT["finish_reason"] = finish_reason
    _CAPTURED.clear()
    with patch.object(main.httpx, "AsyncClient", _StubVLLM), \
         patch.object(main, "_run_memory_tail", _spy_run_memory_tail):
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "stub-model",
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "hello there, this is a real user turn"},
                ],
                "stream": False,
            },
            headers={"X-Conversation-Id": conv_id},
        )
    return r, dict(_CAPTURED)


def test_control_clean_stop_is_finished():
    print("\n[test] CONTROL: finish_reason='stop' — finished=True, as always")
    r, captured = _post_chat("p165-control-stop", "a normal complete reply.", "stop")
    check(r.status_code == 200, f"200 OK: {r.status_code}")
    check(captured.get("finished") is True, f"finished=True: {captured.get('finished')!r}")
    check(captured.get("truncated") is False, f"truncated=False: {captured.get('truncated')!r}")


def test_control_length_is_truncated_not_error():
    print("\n[test] CONTROL: finish_reason='length' — finished=True, "
          "truncated=True (unaffected sibling case, not what this fix "
          "touches)")
    r, captured = _post_chat("p165-control-length", "a reply that got cut at the", "length")
    check(r.status_code == 200, f"200 OK: {r.status_code}")
    check(captured.get("finished") is True, f"finished=True: {captured.get('finished')!r}")
    check(captured.get("truncated") is True, f"truncated=True: {captured.get('truncated')!r}")


def test_finish_reason_error_is_not_finished():
    print("\n[test] finish_reason='error' with partial content — must NOT "
          "be stored as a finished reply (was the bug: finished=True "
          "unconditionally)")
    r, captured = _post_chat("p165-error", "a reply that was cut off by an in-band error", "error")
    check(r.status_code == 200, f"200 OK (vLLM's own envelope, not an HTTP error): {r.status_code}")
    check(captured.get("finished") is False,
          f"finished=False for finish_reason='error' (was True pre-fix): {captured.get('finished')!r}")


def _all_tests():
    return [
        test_control_clean_stop_is_finished,
        test_control_length_is_truncated_not_error,
        test_finish_reason_error_is_not_finished,
    ]


if __name__ == "__main__":
    for t in _all_tests():
        t()
    print("\nRESULT:", "all v3.1.9.4 R5 P16-5 checks passed" if not FAILED else f"{len(FAILED)} check(s) FAILED: {FAILED}")
    sys.exit(0 if not FAILED else 1)
