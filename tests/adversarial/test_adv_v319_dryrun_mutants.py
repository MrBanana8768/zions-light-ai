"""AREA 5 hostile pass #2: which mutations of `_dry_run_from` the shipped
suites can actually kill.

NEW FILE. No shipped module is edited. The shipped `_dry_run_from` is loaded
verbatim by name; each MUTANT is a string edit of that extracted source whose
needle is asserted to occur exactly once before it is applied (the brief's
honesty rule — a needle matching 0 or 3 lines is not a mutation and no result
is reported for it).

THE ORACLE is the set of (query string, body, default) triples the shipped
tests actually drive through this function. They are enumerated below with a
file:line citation each, taken from reading the two suites:

    compactor/test_admin_compact.py     default=False   (compact)
    compactor/test_conv_fork.py  [4]    default=True    (merge)

A mutant that agrees with the shipped function on EVERY tested triple cannot
be distinguished by those suites and is reported GREEN — survived. That is a
mechanical comparison, not a re-implementation of the assertions.

Run:  python tests/adversarial/test_adv_v319_dryrun_mutants.py
"""

from __future__ import annotations

import ast
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "compactor" / "main.py"

FAILED: list[str] = []
SURVIVED: list[str] = []


def check(cond, label):
    print(f"   {'PASS' if cond else 'FAIL'}  {label}")
    if not cond:
        FAILED.append(label)


def func_source(name: str) -> str:
    src = MAIN.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(MAIN))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return textwrap.dedent(ast.get_source_segment(src, node))
    raise AssertionError(f"{name} not found in main.py")


def compile_fn(source: str):
    from starlette.requests import Request  # noqa: F401 used via ns
    ns: dict = {"Request": Request}
    exec(compile(source, "<_dry_run_from>", "exec"), ns)
    return ns["_dry_run_from"]


def req(qs: str):
    from starlette.requests import Request
    return Request({"type": "http", "method": "POST", "path": "/admin/x",
                    "headers": [], "query_string": qs.encode()})


# ---------------------------------------------------------------------------
# The oracle: every triple the shipped suites drive through this function.
# ---------------------------------------------------------------------------
# (label, query string, body, default)
ORACLE = [
    # test_admin_compact.py — compact(), default=False
    ("[2]  body dry_run=True",           "",              {"dry_run": True},  False),
    ("[2b] body dry_run=False",          "",              {"dry_run": False}, False),
    ("[9]  ?dry_run=true, body {}",      "dry_run=true",  {},                 False),
    ("[9]  no flag, body {}",            "",              {},                 False),
    ("[9]  ?dry_run=false, body {}",     "dry_run=false", {},                 False),
    # test_conv_fork.py [4] — merge, default=True
    ("[4]  ?dry_run=false",              "dry_run=false", {},                 True),
    ("[4]  body dry_run=False",          "",              {"dry_run": False}, True),
    ("[4]  no flag at all",              "",              {},                 True),
    ("[4]  ?dry_run=true",               "dry_run=true",  {},                 True),
    ("[4]  ?dry_run=flase (typo)",       "dry_run=flase", {},                 True),
    ("[4]  body True beats ?false",      "dry_run=false", {"dry_run": True},  True),
]

SHIPPED_SRC = func_source("_dry_run_from")
SHIPPED = compile_fn(SHIPPED_SRC)


def oracle_vector(fn):
    return [fn(req(q), dict(b), default=d) for _, q, b, d in ORACLE]


BASE = oracle_vector(SHIPPED)


# ---------------------------------------------------------------------------
# The mutants.
# ---------------------------------------------------------------------------

MUTANTS = [
    ("MM6  read the body only (the shipped defect 843bf9d fixed)",
     'raw = str(request.query_params.get("dry_run", "")).strip().lower()',
     'raw = ""'),

    ("M-a  raw not in ('false','0','no')  ->  raw == 'true'",
     'return raw not in ("false", "0", "no")',
     'return raw == "true"'),

    ("M-b  drop '0' and 'no' from the commit tuple",
     'return raw not in ("false", "0", "no")',
     'return raw not in ("false",)'),

    ("M-c  drop the empty-raw early return (empty means DRY)",
     '    if not raw:\n        return default\n',
     ''),

    ("M-d  bool(body[...])  ->  body[...] is not False",
     'return bool(body["dry_run"])',
     'return body["dry_run"] is not False'),

    ("M-e  presence test  ->  is not None",
     'if "dry_run" in body:',
     'if body.get("dry_run") is not None:'),

    ("M-f  drop .strip() from the query read",
     'str(request.query_params.get("dry_run", "")).strip().lower()',
     'str(request.query_params.get("dry_run", "")).lower()'),

    ("M-g  ignore `default`, always dry when absent",
     '    if not raw:\n        return default\n',
     '    if not raw:\n        return True\n'),

    ("M-h  query beats body (precedence flip)",
     '    if "dry_run" in body:\n        return bool(body["dry_run"])\n'
     '    raw = str(request.query_params.get("dry_run", "")).strip().lower()\n'
     '    if not raw:\n        return default\n'
     '    return raw not in ("false", "0", "no")',
     '    raw = str(request.query_params.get("dry_run", "")).strip().lower()\n'
     '    if raw:\n        return raw not in ("false", "0", "no")\n'
     '    if "dry_run" in body:\n        return bool(body["dry_run"])\n'
     '    return default'),
]


def test_which_mutants_the_shipped_suites_kill():
    print()
    print("[A5-15] mutation survival of _dry_run_from against the shipped oracle")
    print()
    print("       the oracle (what the two suites actually drive):")
    for (label, q, b, d), v in zip(ORACLE, BASE):
        print(f"         {label:<32} default={str(d):<5} -> "
              f"{'DRY ' if v else 'LIVE'}")

    for label, needle, repl in MUTANTS:
        print()
        n = SHIPPED_SRC.count(needle)
        print(f"   {label}")
        print(f"       needle occurrences = {n}")
        if n != 1:
            FAILED.append(f"{label}: needle matched {n} lines, NOT 1 — "
                          f"mutation ABORTED, no result reported")
            print("       ABORTED — not a mutation, nothing reported")
            continue
        mutant = compile_fn(SHIPPED_SRC.replace(needle, repl))
        vec = oracle_vector(mutant)
        diffs = [ORACLE[i][0] for i in range(len(BASE)) if vec[i] != BASE[i]]
        if diffs:
            print(f"       RED   — distinguished by: {diffs}")
        else:
            print("       GREEN — SURVIVES: agrees with the shipped function "
                  "on every tested triple")
            SURVIVED.append(label)
            # Show, for a surviving mutant, an input that DOES differ, so the
            # survival is a real coverage hole and not a no-op edit.
            extra = [("?dry_run=1", "dry_run=1", {}, False),
                     ("?dry_run=TRUE", "dry_run=TRUE", {}, False),
                     ("?dry_run=yes", "dry_run=yes", {}, False),
                     ("?dry_run=0", "dry_run=0", {}, True),
                     ("?dry_run=no", "dry_run=no", {}, True),
                     ("?dry_run=", "dry_run=", {}, False),
                     ("?dry_run= true ", "dry_run=%20true%20", {}, False),
                     ("?dry_run= false ", "dry_run=%20false%20", {}, True),
                     ("?dry_run=FALSE", "dry_run=FALSE", {}, True),
                     ("body {'dry_run': None}", "", {"dry_run": None}, True),
                     ("body {'dry_run': ''}", "", {"dry_run": ""}, True),
                     ("body {'dry_run': 0}", "", {"dry_run": 0}, True),
                     ("body {'dry_run': 'false'}", "", {"dry_run": "false"}, False)]
            shown = 0
            for lab, q, b, d in extra:
                a = SHIPPED(req(q), dict(b), default=d)
                m = mutant(req(q), dict(b), default=d)
                if a != m:
                    print(f"           differs on {lab} (default={d}): "
                          f"shipped={'DRY' if a else 'LIVE'} "
                          f"mutant={'DRY' if m else 'LIVE'}")
                    shown += 1
            if shown == 0:
                print("           ...and no probe distinguishes it either — "
                      "an equivalent mutant, not a coverage hole")

    print()
    check(len(SURVIVED) > 0,
          f"F27: {len(SURVIVED)} mutation(s) of _dry_run_from survive the two "
          f"shipped suites: {SURVIVED}")


ALL = [test_which_mutants_the_shipped_suites_kill]

if __name__ == "__main__":
    print("=" * 74)
    print("AREA 5 hostile pass #2 — _dry_run_from mutation survival")
    print("=" * 74)
    for t in ALL:
        t()
    print()
    if FAILED:
        print(f"{len(FAILED)} assertion(s) FAILED:")
        for f in FAILED:
            print(f"  - {f}")
        sys.exit(1)
    print("done")
