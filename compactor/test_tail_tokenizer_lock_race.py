"""
compactor/test_tail_tokenizer_lock_race.py

hostile2-config LOW: "`_TOKENIZER_TRIED` latches before the load resolves,
so a concurrent caller silently gets the estimator". The shipped order used
to be `_TOKENIZER_TRIED = True` and *then* `try: ... AutoTokenizer.from_
pretrained(...)`, with no lock — under uvicorn, `count_tokens` runs from
the threadpool, so two requests can enter `get_tokenizer` concurrently:
thread A latches the flag and blocks inside `from_pretrained`; thread B
sees `_tokenizer is None and _TOKENIZER_TRIED` and returns `None` (the
char/4 estimator) for a tokenizer that was about to load successfully.

STATUS AT THIS LANE'S HEAD (604ac8d): ALREADY FIXED. `get_tokenizer` now
takes a real `threading.Lock` (`_TOKENIZER_LOCK`) around the whole
read-test-and-maybe-load body, double-checks `_tokenizer is not None`
after acquiring it, and only sets `_TOKENIZER_TRIED = True` AFTER the
try/except resolves — this file exists because the brief requires a test
proving an already-fixed finding, not a fix.

This drives the REAL `main.get_tokenizer`, across real OS threads (the
lock in question is a `threading.Lock`, not an `asyncio.Lock`, so an
asyncio-only test would not exercise it), with a scripted `AutoTokenizer.
from_pretrained` that blocks until released — the exact interleaving the
finding names.

Run:
    python test_tail_tokenizer_lock_race.py
"""

import os
import sys
import threading
import time
import types

os.environ.setdefault("MODEL_REPO", "race-test-model")
os.environ.setdefault("VLLM_URL", "http://stub:8000")

import main  # noqa: E402

FAILED: list[str] = []


def check(cond, label):
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"FAIL {label}")
        FAILED.append(label)


class _FakeTokenizer:
    def __init__(self, name):
        self.name = name


def _reset_tokenizer_globals():
    main._tokenizer = None
    main._TOKENIZER_TRIED = False
    main._TOKENIZER_LAST_ERROR = None
    main._TOKENIZER_FAILED_AT = None
    main._TOKENIZER_NEXT_RETRY_AT = None
    main._TOKENIZER_RETRY_S = main._TOKENIZER_RETRY_FLOOR_S
    main.MODEL_REPO = "race-test-model"


def _install_fake_transformers(load_started: threading.Event, release_load: threading.Event):
    fake = types.ModuleType("transformers")

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_repo):
            load_started.set()
            # The window the finding is about: a thread INSIDE this call,
            # holding (with the fix) or not holding (without it) the lock.
            if not release_load.wait(timeout=5):
                raise TimeoutError("test fixture: release_load never set")
            return _FakeTokenizer(model_repo)

    fake.AutoTokenizer = _FakeAutoTokenizer
    sys.modules["transformers"] = fake
    return fake


print("[1] a concurrent caller during an in-flight load blocks and gets "
      "the REAL tokenizer, not None")
_reset_tokenizer_globals()
load_started = threading.Event()
release_load = threading.Event()
_real_transformers = sys.modules.get("transformers")
_install_fake_transformers(load_started, release_load)

result_a: dict = {}
result_b: dict = {}
b_returned_before_release = threading.Event()


def _worker_a():
    result_a["value"] = main.get_tokenizer()


def _worker_b():
    # Wait until thread A is provably INSIDE from_pretrained (holding the
    # lock, with the fix) before entering get_tokenizer ourselves.
    load_started.wait(timeout=5)
    time.sleep(0.05)
    result_b["value"] = main.get_tokenizer()
    b_returned_before_release.set()


ta = threading.Thread(target=_worker_a)
tb = threading.Thread(target=_worker_b)
try:
    ta.start()
    load_started.wait(timeout=5)
    tb.start()
    # B must NOT have returned yet: it should be blocked on _TOKENIZER_LOCK,
    # which A is holding while inside from_pretrained. This is the
    # observable difference from the unfixed shape, where B would have
    # returned None immediately without ever touching the lock.
    time.sleep(0.2)
    check(not b_returned_before_release.is_set(),
          "*** B is still blocked waiting for A's load to resolve, not "
          "returned already")
    release_load.set()
    ta.join(timeout=5)
    tb.join(timeout=5)
finally:
    if _real_transformers is not None:
        sys.modules["transformers"] = _real_transformers
    else:
        sys.modules.pop("transformers", None)

check(ta.is_alive() is False and tb.is_alive() is False,
      "fixture: both threads finished (no deadlock, no timeout)")
check(isinstance(result_a.get("value"), _FakeTokenizer),
      f"A got the real tokenizer (got {result_a.get('value')!r})")
check(result_b.get("value") is result_a.get("value"),
      f"*** B got the SAME real tokenizer object — not None from an "
      f"unresolved latch (got {result_b.get('value')!r})")
check(main._TOKENIZER_TRIED is True,
      "and the flag is set exactly once the attempt has resolved")

_reset_tokenizer_globals()

if FAILED:
    print(f"\n{len(FAILED)} check(s) FAILED")
    sys.exit(1)
print("\nAll test_tail_tokenizer_lock_race checks passed.")
