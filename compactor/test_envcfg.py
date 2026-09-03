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
     backup, pgarchive, retrieval, webuidb, degrade, alert, selftest,
     bgwork) IMPORTS CLEANLY under a bad, a zero, a negative, and an unset
     value for one of its env-driven constants, and lands on a sane
     (usually the coded default) value rather than raising. This is the
     concrete regression check for R30: before this change, each of these
     imports raised ValueError at module scope.

Run: python test_envcfg.py
"""

import math
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
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
    ("pgarchive", "PGARCHIVE_RETAIN", "RETAIN", 10),
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


if __name__ == "__main__":
    test_env_int_contract()
    test_env_float_contract()
    test_env_window_s_contract()
    test_bgwork_and_tailhealth_window_agree()
    test_each_routed_module_survives_bad_values()
    test_the_original_crash_reproduction_no_longer_crashes()

    if FAILED:
        print(f"\n{len(FAILED)} FAILED:")
        for f in FAILED:
            print(f"  - {f}")
        sys.exit(1)
    print("\nAll envcfg tests passed.")
