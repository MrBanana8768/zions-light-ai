"""
compactor/test_tail_tokenize_warn_interval.py

hostile2-config: `COMPACTOR_TOKENIZE_WARN_INTERVAL_S` is read by both
main.py and summarizer.py, under a comment in summarizer.py asserting they
"deliberately" agree so an operator setting the variable once governs every
/tokenize dependency in the process. They did not: main.py parsed it with
`float(_env_int(...))` (an `int()` conversion first), so any non-integer
spelling ("0.5", "60.5", "1e3") silently reverted to the 300 default in
main.py while summarizer.py, which used `env_float` throughout, applied the
operator's real value — one process, one env var, two different rate
limits, with no error or log line naming the disagreement.

This drives the REAL module-level parsing in both files (reload, not a
stand-in), the way an operator's env var actually reaches each constant.

Run:
    python test_tail_tokenize_warn_interval.py
"""

import importlib
import os
import sys

os.environ.setdefault("MODEL_REPO", "test-model")
os.environ.setdefault("VLLM_URL", "http://stub:8000")
os.environ["COMPACTOR_RAG_ENABLED"] = "false"

import main  # noqa: E402
import summarizer  # noqa: E402

FAILED: list[str] = []


def check(cond, label):
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"FAIL {label}")
        FAILED.append(label)


NAME = "COMPACTOR_TOKENIZE_WARN_INTERVAL_S"


def _reload_both():
    # Reload summarizer first: main.py does not import summarizer's module
    # object at parse time for this constant, but reloading in a stable
    # order keeps every case deterministic regardless of what future code
    # adds between them.
    importlib.reload(summarizer)
    importlib.reload(main)


print("[1] main.py and summarizer.py must parse the SAME raw value to the "
      "SAME number")
# Cases from the finding, plus the unset (default) case. Values chosen so a
# reader can see the mechanism directly: "0.5"/"60.5"/"1e3" are exactly the
# non-integer spellings int() rejects.
CASES = [None, "300", "0.5", "60.5", "1e3", "600", "45"]
try:
    for raw in CASES:
        if raw is None:
            os.environ.pop(NAME, None)
        else:
            os.environ[NAME] = raw
        _reload_both()
        m = main.TOKENIZE_WARN_INTERVAL_S
        s = summarizer.TOKENIZE_WARN_INTERVAL_S
        check(
            m == s,
            f"*** raw={raw!r}: main.py={m!r} summarizer.py={s!r} — the "
            f"same env var must not mean two different rate limits in one "
            f"process",
        )
    # The exact false-agreement shape the finding measured: "0.5" used to
    # read as 300.0 in main.py (int() rejects it, falls back to the
    # default) and 0.5 in summarizer.py.
    os.environ[NAME] = "0.5"
    _reload_both()
    check(main.TOKENIZE_WARN_INTERVAL_S == 0.5,
          f"main.py applies a fractional interval, not silently the "
          f"default (got {main.TOKENIZE_WARN_INTERVAL_S!r})")
finally:
    os.environ.pop(NAME, None)
    _reload_both()

if FAILED:
    print(f"\n{len(FAILED)} check(s) FAILED")
    sys.exit(1)
print("\nAll test_tail_tokenize_warn_interval checks passed.")
