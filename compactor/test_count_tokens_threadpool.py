"""
CPU-only Tier-1 test (v3.1.9.8): whole-history count_tokens calls must run
off the event loop.

Findings (~/zl-ops/v3198/FINDINGS-pipeline-scaling-2026-09-27.md, and the
2026-09-25 RCA in memory.md): on her ~4,400-turn / ~8.3M-token chat, one
local count_tokens() pass over the WHOLE history costs ~5s of pure CPU. Three
call sites ran that pass directly on the event loop instead of through
run_in_threadpool like their sibling /tokenize calls beside them:

  - compact_if_needed's `current = count_tokens(messages)` (whole incoming
    history)
  - compact_if_needed's `new_count = count_tokens(new_messages)` (the
    post-compaction array, which on the steady state this user hits —
    reuse declines, `deferred` carries the whole backlog verbatim — is the
    same whole-history size)
  - _summarize_body's `_local = count_tokens(to_summarize)` (the whole
    backlog being considered for summarization)

Each one stalled EVERY OTHER request the process was serving (health checks
included) for ~5s while it ran.

This is an AST check, not a call-count check: it inspects main.py's own
source for these two functions and asserts every whole-list count_tokens(...)
call is the direct argument of `await run_in_threadpool(count_tokens, ...)`.
An AST check is used (rather than instrumenting count_tokens itself) because
the bug is specifically about WHICH THREAD runs the call, and the direct,
readable way to prove that from outside main.py's internals is to require the
source shape that guarantees it — the same shape every other whole-list
counter call in this file already uses (see `new_count`'s own sibling
_sent_token_size, _bound_injected_blocks, _enforce_hard_budget, all called
through run_in_threadpool at their call sites).

Run: python test_count_tokens_threadpool.py
"""

import ast
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_MAIN_PY = os.path.join(_HERE, "main.py")


def assert_true(cond, label):
    if not cond:
        print(f"FAIL {label}")
        sys.exit(1)
    print(f"  ok   {label}")


def _find_function(tree: ast.Module, name: str) -> ast.AST:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"function {name!r} not found in main.py")


def _count_tokens_refs_are_threadpooled(func_node: ast.AST) -> list[bool]:
    """For every reference to the name `count_tokens` (exact match, so
    count_tokens_exact is not caught) found anywhere inside `func_node`,
    True if that reference is passed as a bare callable argument to
    `await run_in_threadpool(count_tokens, ...)` (the correct, off-loop
    shape), False if it is called directly (`count_tokens(...)`, the bug
    this test guards against) or used any other way.

    `run_in_threadpool(fn, *args)` takes the function itself, not a call —
    so the fixed shape has `count_tokens` as a bare Name argument, never as
    a Call's own `.func`. Walks with a parent map since `ast.walk` alone
    does not expose parents.
    """
    parent = {}
    for node in ast.walk(func_node):
        for child in ast.iter_child_nodes(node):
            parent[child] = node

    results = []
    for node in ast.walk(func_node):
        if isinstance(node, ast.Name) and node.id == "count_tokens":
            p = parent.get(node)
            if isinstance(p, ast.Call) and node is p.func:
                # count_tokens(...) called directly — exactly the bug.
                results.append(False)
                continue
            wrapped = (
                isinstance(p, ast.Call)
                and isinstance(p.func, ast.Name)
                and p.func.id == "run_in_threadpool"
                and node in p.args
            )
            if wrapped:
                gp = parent.get(p)
                wrapped = isinstance(gp, ast.Await)
            results.append(wrapped)
    return results


def test_compact_if_needed_count_tokens_is_threadpooled():
    print("\n[test] compact_if_needed: every whole-list count_tokens(...) is threadpooled")
    src = open(_MAIN_PY, encoding="utf-8").read()
    tree = ast.parse(src, filename="main.py")
    func = _find_function(tree, "compact_if_needed")
    assert_true(isinstance(func, ast.AsyncFunctionDef), "compact_if_needed is async")
    results = _count_tokens_refs_are_threadpooled(func)
    assert_true(len(results) >= 2, f"found {len(results)} count_tokens(...) call(s), expected >= 2")
    assert_true(all(results), f"all threadpooled: {results}")


def test_summarize_body_count_tokens_is_threadpooled():
    print("\n[test] _summarize_body: the whole-backlog count_tokens(...) is threadpooled")
    src = open(_MAIN_PY, encoding="utf-8").read()
    tree = ast.parse(src, filename="main.py")
    func = _find_function(tree, "_summarize_body")
    assert_true(isinstance(func, ast.AsyncFunctionDef), "_summarize_body is async")
    results = _count_tokens_refs_are_threadpooled(func)
    assert_true(len(results) >= 1, f"found {len(results)} count_tokens(...) call(s), expected >= 1")
    assert_true(all(results), f"all threadpooled: {results}")


def test_behavioral_current_and_new_count_go_through_threadpool():
    """Belt and braces on top of the AST check: patch starlette's
    run_in_threadpool with a spy and drive compact_if_needed for real with a
    large synthetic history, asserting count_tokens is observed being called
    FROM A WORKER THREAD (not the event loop's own thread) at least twice —
    matching `current` and `new_count`."""
    print("\n[test] behavioral: compact_if_needed's count_tokens calls run off-thread")
    import asyncio
    import threading
    from unittest.mock import patch

    os.environ.pop("MODEL_REPO", None)
    os.environ["MAX_MODEL_LEN"] = "2000"
    os.environ["COMPACTOR_TARGET_TOKENS"] = "100"
    os.environ["COMPACTOR_KEEP_RECENT_TURNS"] = "2"

    import main  # noqa: E402  (import after env is set, matches this suite's convention)

    main_thread_id = threading.get_ident()
    observed_threads = []
    real_count_tokens = main.count_tokens

    def _spy_count_tokens(messages):
        observed_threads.append(threading.get_ident())
        return real_count_tokens(messages)

    async def _fake_summarize(client, to_summarize):
        return "a summary", []

    big_history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": "x" * 400}
        for i in range(60)
    ]

    async def _drive():
        with patch.object(main, "count_tokens", _spy_count_tokens), \
             patch.object(main, "summarize", _fake_summarize):
            await main.compact_if_needed(big_history, "test-conv")

    asyncio.run(_drive())
    assert_true(len(observed_threads) >= 1, "count_tokens was called at least once")
    assert_true(
        all(t != main_thread_id for t in observed_threads),
        f"every count_tokens call ran off the calling thread "
        f"(main={main_thread_id}, observed={observed_threads})",
    )


if __name__ == "__main__":
    test_compact_if_needed_count_tokens_is_threadpooled()
    test_summarize_body_count_tokens_is_threadpooled()
    test_behavioral_current_and_new_count_go_through_threadpool()
    print("\nAll count_tokens-threadpool tests passed.")
