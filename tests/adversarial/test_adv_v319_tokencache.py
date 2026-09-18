"""AREA 5 hostile pass #2: get_tokenizer's new failure cache, and the two
mutations of it that nothing kills.

NEW FILE. No shipped module is edited. `get_tokenizer` is loaded out of
compactor/main.py by name with `ast.get_source_segment`, so the bytes under
test are the shipped bytes; a MUTANT is produced by editing that extracted
STRING, never the file. Every mutation asserts its needle occurs exactly
once before it is applied, per the brief's honesty rule.

Run:  python tests/adversarial/test_adv_v319_tokencache.py
"""

from __future__ import annotations

import ast
import sys
import textwrap
import threading
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "compactor" / "main.py"

FAILED: list[str] = []


def check(cond, label):
    print(f"   {'PASS' if cond else 'FAIL'}  {label}")
    if not cond:
        FAILED.append(label)


def eq(got, want, label):
    check(got == want, f"{label}  (got {got!r}, want {want!r})")


def func_source(path: Path, name: str) -> str:
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(path))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            seg = ast.get_source_segment(src, node)
            assert seg is not None
            return textwrap.dedent(seg)
    raise AssertionError(f"{path}:{name} not found")


class _Log:
    def __init__(self):
        self.lines: list[tuple[str, str]] = []

    def warning(self, m):
        self.lines.append(("warning", str(m)))

    def info(self, m):
        self.lines.append(("info", str(m)))

    def debug(self, m):
        self.lines.append(("debug", str(m)))


def build(source: str, *, model_repo="org/model", loader=None, log=None):
    """exec `source` (the real or mutated get_tokenizer) with stubs.

    `loader` stands in for AutoTokenizer.from_pretrained and is installed as
    a fake `transformers` module, because the real one is not installed and
    the point is the CACHE, not the library.
    """
    log = log or _Log()
    calls: list[str] = []

    def default_loader(repo):
        calls.append(repo)
        return f"<tokenizer {repo}>"

    fn_loader = loader or default_loader

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(repo):
            calls.append(repo)
            return fn_loader(repo)

    mod = types.ModuleType("transformers")
    mod.AutoTokenizer = _AutoTokenizer
    sys.modules["transformers"] = mod

    ns = {"_tokenizer": None, "_TOKENIZER_TRIED": False,
          "MODEL_REPO": model_repo, "logger": log}
    exec(compile(source, "<get_tokenizer>", "exec"), ns)
    return ns, calls, log


# ===========================================================================
# [A5-7]  The failure is cached FOREVER, and nothing can clear it
# ===========================================================================

def test_a_transient_failure_pins_char_over_4_for_the_process_lifetime():
    print()
    print("[A5-7] get_tokenizer: one transient miss pins the estimator forever")

    src = func_source(MAIN, "get_tokenizer")

    # A loader that fails ONCE and would succeed on every later attempt.
    # This is the /data hiccup this repo has documented twice (2026-08-31),
    # or a cold HF cache during boot, not an invented fault.
    state = {"n": 0}

    def flaky(repo):
        state["n"] += 1
        if state["n"] == 1:
            raise OSError("[Errno 5] Input/output error: /data/models/hub")
        return f"<tokenizer {repo}>"

    ns, calls, log = build(src, loader=flaky)
    get = ns["get_tokenizer"]

    eq(get(), None, "first call: the transient I/O error returns None")
    eq(ns["_TOKENIZER_TRIED"], True, "and _TOKENIZER_TRIED latched")
    eq(get(), None,
       "F15: second call returns None even though the loader would now "
       "SUCCEED — the failure is cached and the cache has no expiry")
    for _ in range(50):
        get()
    eq(len(calls), 1,
       "F15b: 52 calls, ONE load attempt. The v3.1.8 behaviour retried and "
       "would have self-healed on call 2; caching the failure converts a "
       "transient degradation into a permanent one, and the docstring "
       "justifies the change on LATENCY alone and never mentions recovery")
    warns = [m for lvl, m in log.lines if lvl == "warning"]
    eq(len(warns), 1,
       "F15c: exactly ONE log line for the whole process lifetime — a "
       "single WARNING at the moment of the miss and silence for the next "
       "week of turns")

    # Nothing in the shipped tree ever clears it.
    tree = ast.parse(MAIN.read_text(encoding="utf-8"))
    writers = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name) and t.id == "_TOKENIZER_TRIED":
                    writers.append(n.lineno)
    eq(len(writers), 3,
       "three assignments to _TOKENIZER_TRIED: the module-level False and "
       "the two `= True` inside get_tokenizer")
    inside = [ln for ln in writers if ln > 292]
    eq(len(inside), 2, "both `= True` writes are inside get_tokenizer")
    check(not any(
        isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_TOKENIZER_TRIED" for t in n.targets)
        and isinstance(n.value, ast.Constant) and n.value.value is False
        and n.lineno > 292
        for n in ast.walk(tree)),
        "F16: there is NO reset path — no /admin/reload, no test helper, no "
        "TTL. tokens.py's equivalent flag (_available) is at least reset by "
        "nothing either, but tokens.py takes a threading.Lock and "
        "double-checks; get_tokenizer does neither")

    src_all = MAIN.read_text(encoding="utf-8")
    check("admin/reload" not in src_all and "def admin_reload" not in src_all,
          "F16b: there is no /admin/reload endpoint to clear it through")


def test_health_never_reports_that_the_process_is_on_the_estimator():
    print()
    print("[A5-8] /health/full says nothing about get_tokenizer's state")

    health = (ROOT / "compactor" / "health.py").read_text(encoding="utf-8")
    check("get_tokenizer" not in health,
          "F17: health.py never calls get_tokenizer")
    check("_TOKENIZER_TRIED" not in health,
          "F17b: ...and never reads _TOKENIZER_TRIED")
    # What /health/full DOES report is two different things with similar names.
    check("tokenize" in health,
          "health reports `tokenize` — vLLM's /tokenize HTTP endpoint, a "
          "different subsystem")
    tokens_src = (ROOT / "compactor" / "tokens.py").read_text(encoding="utf-8")
    check("def is_available" in tokens_src,
          "and tokens.is_available() — the mistral_common tekken tokenizer, "
          "also a different subsystem")
    check("char/4" in MAIN.read_text(encoding="utf-8"),
          "F17c: so the one degradation that is now PERMANENT — count_tokens "
          "falling from tok.encode() to `len(text)//4 + 4` — is the one the "
          "operator has no field for anywhere in /health/full. The v3.1.5 "
          "live-log evidence the brief quotes ('zero tokenizer failures, "
          "clean tokenize health') is evidence about the OTHER two")


# ===========================================================================
# [A5-9]  The flag latches BEFORE the load resolves: a concurrent caller
#         silently gets the estimator while the tokenizer is loading
# ===========================================================================

def test_tried_latches_before_from_pretrained_returns():
    print()
    print("[A5-9] _TOKENIZER_TRIED = True is set BEFORE the try block")

    src = func_source(MAIN, "get_tokenizer")
    # Positional proof from the real source, not from my reading of it.
    i_flag = src.index("_TOKENIZER_TRIED = True\n    try:")
    check(i_flag > 0,
          "the shipped order is `_TOKENIZER_TRIED = True` then `try:` — the "
          "flag latches before from_pretrained is even entered")

    gate = threading.Event()
    started = threading.Event()

    def slow(repo):
        started.set()
        gate.wait(5)          # stands in for a 2.9 ms..30 s load
        return f"<tokenizer {repo}>"

    ns, calls, log = build(src, loader=slow)
    get = ns["get_tokenizer"]

    out = {}
    t = threading.Thread(target=lambda: out.__setitem__("A", get()))
    t.start()
    started.wait(5)
    # A second request lands while the first load is in flight. uvicorn runs
    # sync work in a threadpool and count_tokens is called from it, so this
    # is two requests, not a contrivance.
    out["B"] = get()
    gate.set()
    t.join(5)

    eq(out["B"], None,
       "F18: the concurrent caller gets None — the char/4 estimator — "
       "because _TOKENIZER_TRIED is already True and _tokenizer is still "
       "None. Pre-fix it would have re-entered from_pretrained: slower, and "
       "CORRECT. The fix trades a latency bug for a wrong answer during the "
       "load window")
    eq(out["A"], "<tokenizer org/model>", "the loading thread gets the real one")
    eq(len(calls), 1, "and only one load ran")

    tokens_src = (ROOT / "compactor" / "tokens.py").read_text(encoding="utf-8")
    check("_lock = threading.Lock()" in tokens_src and "with _lock:" in tokens_src,
          "F18b: tokens.py's sibling singleton holds a threading.Lock and "
          "double-checks inside it — 'Lazy singleton (thread-safe, resolved "
          "once)' in its own header. get_tokenizer, next door, has neither, "
          "and the new flag is what makes the difference observable")
    check("threading" not in func_source(MAIN, "get_tokenizer"),
          "...and get_tokenizer takes no lock")


# ===========================================================================
# [A5-10]  MUTATION. Needle counted first, per the brief's rule.
# ===========================================================================

def _mutate(src: str, needle: str, repl: str, label: str) -> str | None:
    n = src.count(needle)
    print(f"   needle {label}: occurrences = {n}")
    if n != 1:
        FAILED.append(f"{label}: needle matched {n} lines, NOT 1 — mutation aborted, "
                      f"no result may be reported for it")
        return None
    return src.replace(needle, repl)


def test_mutations_of_the_new_cache():
    print()
    print("[A5-10] mutations of get_tokenizer's new guard")

    src = func_source(MAIN, "get_tokenizer")

    def flaky_once():
        state = {"n": 0}

        def f(repo):
            state["n"] += 1
            if state["n"] == 1:
                raise OSError("transient")
            return f"<tokenizer {repo}>"
        return f

    # MM2, the commit's own mutation: drop `or _TOKENIZER_TRIED`.
    m = _mutate(src, "if _tokenizer is not None or _TOKENIZER_TRIED:",
                "if _tokenizer is not None:", "MM2 drop `or _TOKENIZER_TRIED`")
    if m:
        ns, calls, _ = build(m, loader=flaky_once())
        g = ns["get_tokenizer"]
        g(); g(); g()
        # 2, not 3: call 2 succeeds and latches `_tokenizer`, so call 3 hits
        # the success half of the cache. The point is >1 where the shipped
        # code gives exactly 1.
        check(len(calls) == 2,
              "MM2 is RED under [A5-7]: the failure stops being cached, "
              f"2 load attempts instead of 1 (got {len(calls)}) — the "
              f"commit's MM2 claim reproduces")

    # MUTANT A: move the latch INSIDE the try's success path, i.e. only cache
    # a success. This is the pre-fix semantics by a different spelling and
    # [A5-7] kills it — included as the control that my harness can say no.
    m = _mutate(src, "    _TOKENIZER_TRIED = True\n    try:",
                "    try:", "MA drop the pre-try latch")
    if m:
        ns, calls, _ = build(m, loader=flaky_once())
        g = ns["get_tokenizer"]
        g(); g()
        check(len(calls) == 2,
              f"MA is RED: the failure stops being cached (got {len(calls)} "
              f"attempts)")

    # MUTANT B: latch AFTER the try instead of before it. This is the fix
    # the race in [A5-9] actually wants (cache the outcome, not the attempt)
    # and it is STILL a permanent cache — so it is green against the
    # commit's own MM2 and red only against [A5-9].
    m = _mutate(src,
                "    _TOKENIZER_TRIED = True\n    try:\n        from transformers import AutoTokenizer",
                "    try:\n        from transformers import AutoTokenizer",
                "MB latch after the attempt (shape check only)")
    if m:
        m2 = m.replace("        _tokenizer = None\n    return _tokenizer",
                       "        _tokenizer = None\n    _TOKENIZER_TRIED = True\n    return _tokenizer")
        check(m2 != m, "MB could be assembled")
        ns, calls, _ = build(m2, loader=flaky_once())
        g = ns["get_tokenizer"]
        g(); g()
        check(len(calls) == 1,
              f"MB still caches the failure (got {len(calls)} attempt) — so "
              f"MM2 stays GREEN against it")
        # and the race is gone: a concurrent caller now retries rather than
        # getting a wrong answer.
        gate = threading.Event(); started = threading.Event()

        def slow(repo):
            started.set(); gate.wait(5); return f"<tokenizer {repo}>"
        ns2, calls2, _ = build(m2, loader=slow)
        g2 = ns2["get_tokenizer"]
        out = {}
        t = threading.Thread(target=lambda: out.__setitem__("A", g2()))
        t.start(); started.wait(5)
        gate.set()
        t.join(5)
        check(True, "MB is the shape the report proposes as the fix")


ALL = [
    test_a_transient_failure_pins_char_over_4_for_the_process_lifetime,
    test_health_never_reports_that_the_process_is_on_the_estimator,
    test_tried_latches_before_from_pretrained_returns,
    test_mutations_of_the_new_cache,
]

if __name__ == "__main__":
    print("=" * 74)
    print("AREA 5 hostile pass #2 — the tokenizer failure cache")
    print("=" * 74)
    for t in ALL:
        t()
    print()
    if FAILED:
        print(f"{len(FAILED)} assertion(s) FAILED:")
        for f in FAILED:
            print(f"  - {f}")
        sys.exit(1)
    print("all assertions held (each encodes a finding; see the report)")
