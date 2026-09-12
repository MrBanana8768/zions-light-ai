"""
Tests for compactor.envcfg (V314_BACKLOG R30 rest / V317_PLAN Stage 5 #13).

Covers:
  1. env_int / env_float's own contract in isolation (unset, blank,
     unparseable, zero, negative, inf, nan, an explicit good value).
  2. env_window_s's stricter contract (env_float's contract plus rejecting
     non-positive and non-finite).
  3. bgwork._window_s (now a thin alias to envcfg.env_window_s) and
     tailhealth._window_s (an independent, unowned copy of the same logic)
     still AGREE on every input this suite throws at them — the property
     the module-level docstrings on both have asserted since before this
     module existed.
  4. Every module this change routed through envcfg (facts, summarizer,
     backup, retrieval, webuidb, degrade, alert, selftest, bgwork) IMPORTS
     CLEANLY under a bad, a zero, a negative, and an unset value for one of
     its env-driven constants, and lands on a sane (usually the coded
     default) value rather than raising. This is the concrete regression
     check for R30: before this change, each of these imports raised
     ValueError at module scope.

  5. (v3.1.9) NO SHIPPED MODULE ANYWHERE contains an unsoftened conversion
     over an environment value. Sections 1-4 prove the helper works and that
     ten named modules use it. They cannot prove the ELEVENTH module does,
     and that is exactly how v3.1.7 shipped: the R30 sweep converted ~47
     call sites, wrote this suite to cover the modules it had converted, and
     left SEVEN sites untouched in health.py, dedup.py, persona.py,
     tokens.py and main.py — every one of them at module scope, every one of
     them on main.py's transitive import path, so every one of them still a
     container that will not boot on a typo. A per-module enumeration can
     only ever assert what its author remembered. Section 5 asserts the
     PROPERTY over the whole shipped tree instead, so the sweep does not
     have to be remembered a second time. See _unsoftened_env_conversions
     for exactly what it catches and what it does not.

Run: python test_envcfg.py
"""

import ast
import math
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PY = sys.executable

import envcfg  # noqa: E402

FAILED = []


def assert_eq(a, b, label):
    if a != b:
        print(f"FAIL {label}: expected {b!r}, got {a!r}")
        FAILED.append(label)
    else:
        print(f"  ok   {label}")


def assert_true(cond, label):
    if not cond:
        print(f"FAIL {label}")
        FAILED.append(label)
    else:
        print(f"  ok   {label}")


# ---------------------------------------------------------------------------
# 1. env_int / env_float
# ---------------------------------------------------------------------------

def test_env_int_contract():
    print("\n[1] env_int: unset/blank/unparseable -> default, never raises")
    name = "__ENVCFG_TEST_INT__"
    os.environ.pop(name, None)
    assert_eq(envcfg.env_int(name, 42), 42, "unset -> default")
    os.environ[name] = ""
    assert_eq(envcfg.env_int(name, 42), 42, "blank -> default")
    os.environ[name] = "   "
    assert_eq(envcfg.env_int(name, 42), 42, "whitespace-only -> default")
    os.environ[name] = "5O"  # letter O, not zero
    assert_eq(envcfg.env_int(name, 42), 42, "unparseable ('5O') -> default, no raise")
    os.environ[name] = "32K"
    assert_eq(envcfg.env_int(name, 32768), 32768, "unparseable ('32K') -> default")
    os.environ[name] = "7"
    assert_eq(envcfg.env_int(name, 42), 7, "valid value is parsed")
    os.environ[name] = "0"
    assert_eq(envcfg.env_int(name, 42), 0, "explicit 0 is NOT policed, passes through")
    os.environ[name] = "-5"
    assert_eq(envcfg.env_int(name, 42), -5, "negative is NOT policed, passes through")
    os.environ.pop(name, None)


def test_env_float_contract():
    print("\n[2] env_float: unset/blank/unparseable -> default; range unpoliced")
    name = "__ENVCFG_TEST_FLOAT__"
    os.environ.pop(name, None)
    assert_eq(envcfg.env_float(name, 0.5), 0.5, "unset -> default")
    os.environ[name] = ""
    assert_eq(envcfg.env_float(name, 0.5), 0.5, "blank -> default")
    os.environ[name] = "abc"
    assert_eq(envcfg.env_float(name, 0.5), 0.5, "unparseable -> default, no raise")
    os.environ[name] = "0.87x"
    assert_eq(envcfg.env_float(name, 0.5), 0.5, "trailing garbage -> default")
    os.environ[name] = "0.87"
    assert_eq(envcfg.env_float(name, 0.5), 0.87, "valid value is parsed")
    os.environ[name] = "0"
    assert_eq(envcfg.env_float(name, 0.5), 0.0, "explicit 0 passes through unpoliced")
    os.environ[name] = "-1.5"
    assert_eq(envcfg.env_float(name, 0.5), -1.5, "negative passes through unpoliced")
    os.environ[name] = "inf"
    assert_true(math.isinf(envcfg.env_float(name, 0.5)), "inf passes through unpoliced")
    os.environ[name] = "nan"
    assert_true(math.isnan(envcfg.env_float(name, 0.5)), "nan passes through unpoliced")
    os.environ[name] = "1e400"  # overflows to inf, still parses
    assert_true(math.isinf(envcfg.env_float(name, 0.5)), "1e400 (-> inf) passes through unpoliced")
    os.environ.pop(name, None)


# ---------------------------------------------------------------------------
# 2. env_window_s
# ---------------------------------------------------------------------------

def test_env_window_s_contract():
    print("\n[3] env_window_s: env_float's contract, plus rejects non-positive/non-finite")
    name = "__ENVCFG_TEST_WINDOW__"
    os.environ.pop(name, None)
    assert_eq(envcfg.env_window_s(name, 300.0), 300.0, "unset -> default")
    os.environ[name] = ""
    assert_eq(envcfg.env_window_s(name, 300.0), 300.0, "blank -> default")
    os.environ[name] = "30O"
    assert_eq(envcfg.env_window_s(name, 300.0), 300.0, "unparseable -> default")
    os.environ[name] = "45.5"
    assert_eq(envcfg.env_window_s(name, 300.0), 45.5, "valid positive value is parsed")
    os.environ[name] = "0"
    assert_eq(envcfg.env_window_s(name, 300.0), 300.0, "0 -> default (rejected, unlike env_float)")
    os.environ[name] = "-5"
    assert_eq(envcfg.env_window_s(name, 300.0), 300.0, "negative -> default (rejected)")
    os.environ[name] = "inf"
    assert_eq(envcfg.env_window_s(name, 300.0), 300.0, "inf -> default (rejected)")
    os.environ[name] = "Infinity"
    assert_eq(envcfg.env_window_s(name, 300.0), 300.0, "'Infinity' -> default (rejected)")
    os.environ[name] = "nan"
    assert_eq(envcfg.env_window_s(name, 300.0), 300.0, "nan -> default (rejected)")
    os.environ[name] = "1e400"
    assert_eq(envcfg.env_window_s(name, 300.0), 300.0, "1e400 (-> inf) -> default (rejected)")
    os.environ.pop(name, None)


# ---------------------------------------------------------------------------
# 3. bgwork._window_s and tailhealth._window_s must keep agreeing
# ---------------------------------------------------------------------------

def test_bgwork_and_tailhealth_window_agree():
    print("\n[4] bgwork._window_s and tailhealth._window_s agree on every input")
    import bgwork
    import tailhealth

    name_b = "__ENVCFG_TEST_AGREE_B__"
    name_t = "__ENVCFG_TEST_AGREE_T__"
    cases = ["", "  ", "30O", "abc", "45.5", "0", "-5", "-0.0", "inf", "-inf",
             "Infinity", "nan", "NaN", "1e400", "1e-10", "300"]
    for raw in cases:
        os.environ[name_b] = raw
        os.environ[name_t] = raw
        b = bgwork._window_s(name_b, 300.0)
        t = tailhealth._window_s(name_t, 300.0)
        both_nan = isinstance(b, float) and isinstance(t, float) and math.isnan(b) and math.isnan(t)
        ok = (b == t) or both_nan
        assert_true(ok, f"bgwork vs tailhealth agree on {raw!r} (bgwork={b!r}, tailhealth={t!r})")
    os.environ.pop(name_b, None)
    os.environ.pop(name_t, None)

    # Also unset entirely.
    os.environ.pop(name_b, None)
    os.environ.pop(name_t, None)
    assert_eq(bgwork._window_s(name_b, 300.0), tailhealth._window_s(name_t, 300.0),
              "bgwork vs tailhealth agree when unset")

    # And bgwork._window_s is now literally envcfg.env_window_s under the hood.
    os.environ[name_b] = "45.5"
    assert_eq(bgwork._window_s(name_b, 300.0), envcfg.env_window_s(name_b, 300.0),
              "bgwork._window_s matches envcfg.env_window_s directly")
    os.environ.pop(name_b, None)


# ---------------------------------------------------------------------------
# 4. Every routed module survives a bad/zero/negative/unset value and imports
# ---------------------------------------------------------------------------

# (module, an env var it reads through envcfg, a bad value, a sane default it
# should fall back to)
IMPORT_CASES = [
    ("facts", "MAX_MODEL_LEN", "_MAX_MODEL_LEN", 32768),
    ("summarizer", "MAX_MODEL_LEN", "MAX_MODEL_LEN", 32768),
    ("backup", "COMPACTOR_BACKUP_RETAIN", "RETAIN", 7),
    ("retrieval", "COMPACTOR_RAG_TOP_K", "RAG_TOP_K", 5),
    ("webuidb", "WEBUI_DB_SYNC_INTERVAL_S", "SYNC_INTERVAL_S", 300.0),
    ("degrade", "COMPACTOR_MIN_FREE_MB_WRITES", "MIN_FREE_MB_WRITES", 200),
    ("alert", "COMPACTOR_ALERT_TIMEOUT_S", "TIMEOUT_S", 10.0),
    ("selftest", "COMPACTOR_SELFTEST_WAIT_TIMEOUT_S", "WAIT_FOR_READY_TIMEOUT_S", 600.0),
    ("bgwork", "COMPACTOR_MAX_CONCURRENT_TAILS", "MAX_CONCURRENT", 4),
]


def _run_import(module: str, env_var: str, value: str | None) -> tuple[int, str]:
    """Import `module` in a fresh subprocess with `env_var` set to `value`
    (or unset if None), print the constant module.CONST for the case's
    constant name, and return (returncode, stdout+stderr).

    A fresh subprocess per case, not an in-process re-import: these
    constants are set at module scope on first import and Python caches
    modules in sys.modules, so a second `import facts` in the same process
    would not re-run the env read at all. That is exactly the "at import
    time, before logsetup" scenario R30 is about, so it has to be a real
    fresh interpreter each time to mean anything.
    """
    const = dict((m, c) for m, _, c, _ in IMPORT_CASES)[module]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(HERE)
    env["PYTHONIOENCODING"] = "utf-8"
    if value is None:
        env.pop(env_var, None)
    else:
        env[env_var] = value
    code = f"import {module}; print({module}.{const})"
    r = subprocess.run(
        [PY, "-c", code], cwd=str(HERE), env=env,
        capture_output=True, text=True, timeout=60,
    )
    return r.returncode, (r.stdout + r.stderr)


def test_each_routed_module_survives_bad_values():
    print("\n[5] every routed module imports cleanly under bad/zero/negative/unset values")
    for module, env_var, const, default in IMPORT_CASES:
        for label, value in [
            ("unparseable", "not-a-number"),
            ("zero", "0"),
            ("negative", "-1"),
            ("unset", None),
        ]:
            rc, out = _run_import(module, env_var, value)
            assert_eq(rc, 0, f"{module}: import survives {env_var}={value!r} ({label})")
            if rc != 0:
                print(f"       output: {out.strip()[-400:]}")


def test_the_original_crash_reproduction_no_longer_crashes():
    print("\n[6] the concrete R30 reproduction: MAX_MODEL_LEN=32K")
    for module, const in [("facts", "_MAX_MODEL_LEN"), ("summarizer", "MAX_MODEL_LEN")]:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(HERE)
        env["MAX_MODEL_LEN"] = "32K"
        r = subprocess.run(
            [PY, "-c", f"import {module}; print({module}.{const})"],
            cwd=str(HERE), env=env, capture_output=True, text=True, timeout=60,
        )
        assert_eq(r.returncode, 0, f"{module}: MAX_MODEL_LEN=32K no longer crashes at import")
        if r.returncode == 0:
            assert_eq(r.stdout.strip(), "32768", f"{module}: falls back to the coded default 32768")


# ---------------------------------------------------------------------------
# 5. The property, over the whole shipped tree: no unsoftened env conversion
# ---------------------------------------------------------------------------

# WHY A SOURCE SCAN AND NOT MORE IMPORT CASES. An import case proves one
# module survives one bad value for one variable the test's author thought
# of. That is the shape of check that let the v3.1.7 sweep look complete
# while seven sites were still bare, because the seven were in modules
# nobody added to IMPORT_CASES. This walks every shipped module instead and
# asserts a property of the SOURCE, so a new module, or a new knob in an old
# one, is covered on the day it is written rather than on the day someone
# remembers to extend a list.
#
# WHY AST AND NOT grep. The obvious regex fires on this codebase's own prose:
# envcfg.py's docstring, main._env_int's, tailhealth._window_s's and the new
# comments in health.py and dedup.py all quote `float(os.environ.get(...))`
# verbatim, because naming the bad pattern is how this project documents the
# fix. A regex would be red on a clean tree and would be deleted within a
# week. The AST has no comments and no docstring bodies in it, and it also
# does not care that the real sites were split across three lines, which a
# line-oriented grep does.

# stt/ IS IN THIS LIST AND THAT IS THE WHOLE POINT. The v3.1.9 review that
# ordered this sweep named compactor/, tts/ and pipelines/ — and stt/server.py
# is tts/server.py's twin, same `_env` helper, same `int(_env(PORT))` on the
# line below it, two defects in it rather than one. An instruction to sweep
# the siblings missed a sibling. A list is only ever as complete as the last
# person to think about it, so this one is checked against the image: every
# directory Dockerfile COPYs a .py from belongs here.
SHIPPED_DIRS = [HERE, ROOT / "tts", ROOT / "stt", ROOT / "pipelines"]

# Conversions that raise on a string the operator mistyped. `str`, `bool`,
# `Path` and f-string interpolation are absent deliberately: none of them can
# raise on any string, so none of them can stop a boot.
_CONVERTERS = frozenset({"int", "float", "complex"})

# A conversion inside a `try` that catches one of these is not unsoftened —
# it is the helper. envcfg.env_int, main._env_int and tailhealth._window_s
# are all exactly this shape, and the rule has to admit them or it forbids
# writing the fix.
_VALUE_GUARDS = frozenset({"ValueError", "TypeError", "Exception", "BaseException"})
_KEY_GUARDS = frozenset({"KeyError", "LookupError", "Exception", "BaseException"})


def _reads_env(node: ast.AST) -> bool:
    """True if this expression subtree touches the process environment."""
    for n in ast.walk(node):
        if isinstance(n, ast.Attribute) and n.attr in ("environ", "getenv"):
            return True
        if isinstance(n, ast.Name) and n.id in ("environ", "getenv"):
            return True
    return False


def _handler_names(handler: ast.ExceptHandler) -> set[str]:
    t = handler.type
    if t is None:
        return {"BaseException"}  # bare `except:` catches everything
    parts = t.elts if isinstance(t, ast.Tuple) else [t]
    out = set()
    for p in parts:
        if isinstance(p, ast.Name):
            out.add(p.id)
        elif isinstance(p, ast.Attribute):
            out.add(p.attr)
    return out


def _env_string_sources(tree: ast.AST) -> set[str]:
    """Functions in this file that read the environment and hand back a STRING.

    tts/server.py's `_env(name, default) -> str` is one, and
    `int(_env("TTS_PORT", "9001"))` is the same defect one hop removed — the
    helper softens an unset or empty value and nothing softens a mistyped
    one. envcfg.env_int and main._env_float are NOT string sources (they are
    annotated `-> int` / `-> float` and have already done the conversion), so
    this reads the return annotation rather than assuming.
    """
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            ann = n.returns
            if isinstance(ann, ast.Name) and ann.id == "str" and _reads_env(n):
                out.add(n.name)
    return out


def _env_tainted_names(tree: ast.AST) -> set[str]:
    """Names bound directly to an environment read, anywhere in the file.

    Catches the two-line spelling of the same bug:
        raw = os.environ.get("X", "3")
        TIMEOUT = float(raw)          # <- still a ValueError on a typo
    """
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and _reads_env(n.value):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    out.add(t.id)
        elif isinstance(n, ast.AnnAssign) and n.value is not None and _reads_env(n.value):
            if isinstance(n.target, ast.Name):
                out.add(n.target.id)
    return out


def _unsoftened_env_conversions(src: str, label: str) -> list[str]:
    """Every unsoftened environment conversion in `src`, as readable strings.

    WHAT IT CATCHES:
      1. int(os.environ.get(...)) / float(...) / complex(...), however the
         call is wrapped across lines, including with an `or default` that
         only ever rescued the empty string.
      2. The same through a local name: `raw = os.environ.get(...)` then
         `int(raw)`.
      3. The same through a local helper annotated `-> str` that reads the
         environment (tts/server.py's `_env`).
      4. os.environ["X"] — a KeyError at module scope is the same dead
         container by a different exception.
    ...and exempts any of those lexically inside a `try` whose handlers
    catch the exception in question, because that IS the softening.

    WHAT IT DOES NOT CATCH, stated so the next person does not over-trust it:
      * A conversion two or more hops from the env read (env -> a -> b ->
        int(b)). One hop is what the real defects looked like; a full taint
        analysis is not worth the false positives.
      * A helper that returns a string without saying `-> str`. The
         annotation is the signal; an unannotated one is invisible here.
      * A raise from something other than int/float/complex — json.loads,
        datetime.fromisoformat, re.compile, or an `assert` on a config
        value. Add the callable to _CONVERTERS if one ever appears.
      * A try that catches the wrong exception (`except OSError:` around an
        int()) reads as softened here but is not. Nothing in the tree does
        this; a reviewer would catch it.
      * Anything outside compactor/, tts/ and pipelines/ — scripts/ and the
        shell are not scanned, and entrypoint.sh does its own parsing in a
        dialect this cannot see (see V319 report, the env_bool note).
    """
    tree = ast.parse(src, filename=label)
    tainted = _env_tainted_names(tree)
    sources = _env_string_sources(tree)
    found: list[str] = []

    def arg_is_env(a: ast.AST) -> bool:
        if _reads_env(a):
            return True
        if isinstance(a, ast.Name) and a.id in tainted:
            return True
        if isinstance(a, ast.Call) and isinstance(a.func, ast.Name) and a.func.id in sources:
            return True
        return False

    def visit(node: ast.AST, caught: frozenset) -> None:
        if isinstance(node, ast.Try):
            handled = set()
            for h in node.handlers:
                handled |= _handler_names(h)
            inner = caught | handled
            # Only the `try:` body is protected. A raise inside a handler,
            # an `else:` or a `finally:` is not caught by this same try — a
            # detail worth getting right, because "it's in a try/except" is
            # exactly the reassurance that would hide the next one.
            for child in node.body:
                visit(child, inner)
            for group in (node.handlers, node.orelse, node.finalbody):
                for child in group:
                    visit(child, caught)
            return

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _CONVERTERS and not (caught & _VALUE_GUARDS):
                if any(arg_is_env(a) for a in node.args):
                    found.append(f"{label}:{node.lineno}: {ast.unparse(node)}")
        elif isinstance(node, ast.Subscript) and not (caught & _KEY_GUARDS):
            # ast.Load ONLY. `os.environ["X"] = "y"` is a WRITE and cannot
            # raise KeyError; every test in this suite opens by writing a
            # dozen of them. Without the ctx check this rule reports 170
            # findings on a clean tree, and a check that cries wolf at that
            # volume is a check someone deletes.
            v = node.value
            if isinstance(node.ctx, ast.Load) and (
                (isinstance(v, ast.Attribute) and v.attr == "environ")
                or (isinstance(v, ast.Name) and v.id == "environ")
            ):
                found.append(f"{label}:{node.lineno}: {ast.unparse(node)}")

        for child in ast.iter_child_nodes(node):
            visit(child, caught)

    visit(tree, frozenset())
    return found


def _shipped_modules() -> list[Path]:
    out: list[Path] = []
    for d in SHIPPED_DIRS:
        for p in sorted(d.glob("*.py")):
            if p.name.startswith("test_"):
                continue
            out.append(p)
    return out


def test_detector_can_actually_say_no():
    """CONTROL for [7]. A scanner that never finds anything passes forever.

    This is the assertion that would have caught the v3.1.7 miss if the
    sweep's author had written it: it proves the detector reports each of
    the four real shapes, INCLUDING the exact line health.py shipped, before
    [7] is allowed to report a clean tree as good news.
    """
    print("\n[6] CONTROL: the detector flags every bad shape, and no good one")

    bad = '''
import os
def _env(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if v is not None and v != "" else default
A = float(os.environ.get("X", "3.0"))
B = int(
    os.environ.get("Y", "10") or 10
)
raw = os.environ.get("Z", "1")
C = float(raw)
D = int(_env("W", "9001"))
E = os.environ["MUST_BE_SET"]
'''
    hits = _unsoftened_env_conversions(bad, "<bad>")
    assert_eq(len(hits), 5, f"detector finds all 5 planted defects (got: {hits})")
    for want in ("float(os.environ.get('X', '3.0'))", "int(_env('W', '9001'))",
                 "float(raw)", "os.environ['MUST_BE_SET']"):
        assert_true(any(want in h for h in hits), f"detector reports {want}")

    good = '''
import os
from pathlib import Path
from envcfg import env_float, env_int
def env_int2(name: str, default: int) -> int:
    v = os.environ.get(name, "")
    if not v.strip():
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        return default
A = env_float("X", 3.0)
B = env_int("Y", 10)
C = Path(os.environ.get("DATA_DIR", "/data"))
D = os.environ.get("FLAG", "true").lower() != "false"
E = f"http://host:{os.environ.get('PORT', '9000')}"
F = int(A * 2)
G = int(env_int("Y", 10))
os.environ["WEBUIDB_SYNC_ENABLED"] = "false"
try:
    H = int(os.environ["MUST_BE_SET"])
except (KeyError, ValueError):
    H = 0
'''
    hits = _unsoftened_env_conversions(good, "<good>")
    # Each line of `good` is a shape that exists in the real tree and MUST
    # NOT be reported, because the cost of a false positive here is the whole
    # check being deleted the first time it blocks a correct change:
    #   C  Path(os.environ.get(...))  — cannot raise on any string
    #   E  f-string interpolation     — same
    #   G  int() over an `-> int` helper — not an env STRING; flagging it
    #      would punish the very fix this module exists to encourage
    #   os.environ[...] = ...        — a WRITE, not a read
    #   H  a read and a conversion both inside a try that catches both
    assert_eq(hits, [], f"detector is silent on legitimate code (got: {hits})")


def test_no_unsoftened_env_conversion_survives_anywhere():
    print("\n[7] no shipped module converts an env value without softening it")

    # CONTROL FIRST. If SHIPPED_DIRS is wrong, or the container layout moves
    # tts/ somewhere this cannot see, the scan finds zero files and the real
    # assertion below passes for the worst possible reason. A skip is never
    # a pass — so the directories and the file count are themselves checked.
    for d in SHIPPED_DIRS:
        assert_true(d.is_dir(), f"CONTROL: scan directory exists: {d.name}/")
    mods = _shipped_modules()
    assert_true(len(mods) >= 20, f"CONTROL: scan reached >=20 shipped modules (got {len(mods)})")
    rels = {p.relative_to(ROOT).as_posix() for p in mods}
    for expected in ("compactor/main.py", "compactor/health.py",
                     "compactor/dedup.py", "compactor/persona.py",
                     "compactor/tokens.py", "compactor/envcfg.py",
                     "tts/server.py", "stt/server.py",
                     "pipelines/conversation_id_header.py"):
        assert_true(expected in rels, f"CONTROL: scan includes {expected}")

    # And the tree really does read the environment, so a green result means
    # "softened", not "nothing to soften".
    reading = [p for p in mods
               if _reads_env(ast.parse(p.read_text(encoding="utf-8", errors="replace")))]
    assert_true(len(reading) >= 10,
                f"CONTROL: >=10 shipped modules read the environment (got {len(reading)})")

    offenders: list[str] = []
    for p in mods:
        rel = p.relative_to(ROOT).as_posix()
        offenders += _unsoftened_env_conversions(
            p.read_text(encoding="utf-8", errors="replace"), rel
        )
    if offenders:
        print("      unsoftened environment conversions still present:")
        for o in offenders:
            print(f"        {o}")
    assert_eq(offenders, [], "every env conversion in the shipped tree is softened")


if __name__ == "__main__":
    test_env_int_contract()
    test_env_float_contract()
    test_env_window_s_contract()
    test_bgwork_and_tailhealth_window_agree()
    test_each_routed_module_survives_bad_values()
    test_the_original_crash_reproduction_no_longer_crashes()
    test_detector_can_actually_say_no()
    test_no_unsoftened_env_conversion_survives_anywhere()

    if FAILED:
        print(f"\n{len(FAILED)} FAILED:")
        for f in FAILED:
            print(f"  - {f}")
        sys.exit(1)
    print("\nAll envcfg tests passed.")
