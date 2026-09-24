"""
Standing guard: no `compactor/*.py` file references an undefined name.

WHY THIS EXISTS. Commit 7650668 shipped `continuity.py` calling
`_ledger_path(conv_id)` — a name defined nowhere in that module — and
`main.py` calling `tokens.raw()` with no `import tokens` anywhere in that
file. Both are plain NameError, both broke a release-blocking path
(`/forget`, `/forget-all`, `/retire`, and the continuity band's request
path), and neither was caught before landing because nothing in this
repo's Tier-1 gate checks for undefined names — every existing suite
mocks or exercises only the paths its author thought to exercise.

METHOD. Real, scope-aware analysis via `pyflakes` (F821 "undefined
name") — nothing else. Round 1 of this file also shipped a flat,
whole-file AST fallback for when pyflakes was not importable; round 2
review found it too weak to trust as the actual gate: it is file-wide
rather than per-scope (misses a name local to function A wrongly used in
sibling function B), it treats `del x` as a definition (a deleted name
is not bound), and it cannot see a missing attribute on an imported
module — `main.tokens.raw()` reads as fine to a flat scan the moment
`tokens` is bound to ANYTHING in the file, even a local variable of that
name a hundred lines away, which is exactly the kind of near-miss this
guard exists to catch. pyflakes does real per-scope binding analysis and
does not have any of those blind spots. So: this test now REQUIRES
pyflakes and FAILS outright if it cannot be imported, rather than
silently downgrading to the weaker check. `pyflakes` is installed in the
unit-suite TEST image only (testfixtures/unit-suite/requirements-test.txt,
a separate file and layer from compactor/requirements.txt — never the
production image; see that file's own comment) — it must never be added
to compactor/requirements.txt.

VERIFIED (see the v3.1.9.8 report): this file's checker fails on
`git show HEAD:compactor/continuity.py` + `main.py` (HEAD is 7650668,
the unfixed commit — the fix lives only in the working tree) at exactly
`continuity.py:543`, `continuity.py:587` and `main.py:11292`, and passes
against the working tree's fixed versions, with zero false positives
across every other compactor/*.py file.

Run: python test_no_undefined_names.py
"""

import io
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

try:
    from pyflakes.api import check as _pyflakes_check
    from pyflakes.reporter import Reporter as _PyflakesReporter
    _PYFLAKES_IMPORT_ERROR = None
except Exception as _e:  # pragma: no cover - exercised by the failure path itself
    _pyflakes_check = None
    _PyflakesReporter = None
    _PYFLAKES_IMPORT_ERROR = _e


def assert_true(cond, label):
    if not cond:
        print(f"FAIL {label}")
        sys.exit(1)
    print(f"  ok   {label}")


def _pyflakes_undefined_names(path: Path) -> list[str]:
    out, err = io.StringIO(), io.StringIO()
    _pyflakes_check(path.read_text(encoding="utf-8"), str(path), _PyflakesReporter(out, err))
    syntax_errors = err.getvalue().strip()
    lines = [ln for ln in out.getvalue().splitlines() if "undefined name" in ln]
    if syntax_errors:
        lines.append(f"SyntaxError/reporter error: {syntax_errors}")
    return lines


def _all_compactor_files() -> list[Path]:
    return sorted(HERE.glob("*.py"))


def test_pyflakes_is_actually_available():
    print("\n[test] pyflakes is importable — this guard has no weaker fallback")
    assert_true(
        _PYFLAKES_IMPORT_ERROR is None,
        "import pyflakes.api succeeds (install it: "
        "testfixtures/unit-suite/requirements-test.txt, TEST image only — "
        f"never compactor/requirements.txt); got: {_PYFLAKES_IMPORT_ERROR!r}",
    )


def test_no_undefined_names_in_any_compactor_file():
    print("\n[test] undefined-name scan over compactor/*.py (engine: pyflakes)")
    if _PYFLAKES_IMPORT_ERROR is not None:
        # test_pyflakes_is_actually_available already failed (sys.exit(1))
        # when run via __main__ below; this guards direct/partial invocation
        # from ever reporting a false pass.
        print("FAIL pyflakes is not importable — see test_pyflakes_is_actually_available")
        sys.exit(1)
    files = _all_compactor_files()
    assert_true(len(files) > 20, f"found a plausible number of .py files ({len(files)})")
    failures: dict[str, list[str]] = {}
    for path in files:
        problems = _pyflakes_undefined_names(path)
        if problems:
            failures[path.name] = problems
    if failures:
        for fname, problems in sorted(failures.items()):
            print(f"FAIL {fname}:")
            for p in problems:
                print(f"       {p}")
        sys.exit(1)
    print(f"  ok   {len(files)} files, no undefined names")


if __name__ == "__main__":
    test_pyflakes_is_actually_available()
    test_no_undefined_names_in_any_compactor_file()
    print("\nAll undefined-name guard tests passed.")
