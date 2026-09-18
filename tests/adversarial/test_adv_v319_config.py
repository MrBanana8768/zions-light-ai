"""AREA 5 hostile pass #2: config parsing, the admin dry_run, the boolean
dialects, and the AST detector's own blind spots.

NEW FILE. Nothing here edits a shipped module. Every case that exercises
shipped code does it by loading the REAL function's source out of the real
file and exec'ing that one function definition verbatim in a namespace whose
only other contents are stubs the function needs. The extraction is by NAME
through `ast`, and each case prints the source it ran, so a reader can check
that the bytes under test are the shipped bytes and not a paraphrase.

Why not import `compactor.main`? It pulls chromadb / transformers / fastembed,
which the machine running this does not have, and the adversarial docker stack
attacks the HTTP surface rather than a single function's parsing table. The
parsing table is what is wrong here, so it is what gets tested.

Run:  python tests/adversarial/test_adv_v319_config.py
      (also pytest-collectable: every case is a test_* function)
"""

from __future__ import annotations

import ast
import os
import sys
import textwrap
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


# ---------------------------------------------------------------------------
# The extractor. Pulls one top-level def out of a file, verbatim.
# ---------------------------------------------------------------------------

def load_function(path: Path, name: str, ns: dict):
    """exec the real source of `path`'s top-level `name` into `ns`.

    ast.get_source_segment gives the EXACT characters from the file, so what
    runs is the shipped function and not a transcription of it.
    """
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(path))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            seg = ast.get_source_segment(src, node)
            assert seg is not None, f"no source segment for {name}"
            exec(compile(textwrap.dedent(seg), str(path), "exec"), ns)
            return ns[name], node.lineno
    raise AssertionError(f"{path}:{name} not found — the function moved or was renamed")


# ===========================================================================
# [A5-1]  _dry_run_from: the parsing table, and where a LIVE RUN falls out
# ===========================================================================

def test_dry_run_from_parsing_table():
    print()
    print("[A5-1] _dry_run_from(request, body, *, default) — every input shape")

    from starlette.requests import Request

    ns: dict = {}
    fn, lineno = load_function(MAIN, "_dry_run_from", ns)
    print(f"       loaded compactor/main.py:{lineno} verbatim")

    def req(qs: str) -> Request:
        return Request({
            "type": "http", "method": "POST", "path": "/admin/x",
            "headers": [], "query_string": qs.encode(),
        })

    # (query string, body, what MERGE does, what COMPACT does)
    # merge default=True (dry is safe), compact default=False (LIVE is the
    # documented contract).  True = a plan.  False = a WRITE.
    cases = [
        # ---- the three shapes block [9] of test_admin_compact.py covers ----
        ("dry_run=true", {}, True, True),
        ("", {}, True, False),
        ("dry_run=false", {}, False, False),
        # ---- everything it does not ----
        ("dry_run=1", {}, True, True),
        ("dry_run=TRUE", {}, True, True),
        ("dry_run=yes", {}, True, True),
        ("dry_run=ture", {}, True, True),          # value typo -> safe
        ("dry_run=", {}, True, False),             # EMPTY VALUE -> default
        ("dry_run", {}, True, False),              # bare flag  -> default
        ("dryrun=true", {}, True, False),          # KEY typo   -> default
        ("dry-run=true", {}, True, False),         # KEY typo   -> default
        ("dry_run=false&dry_run=true", {}, True, True),   # LAST wins
        ("dry_run=true&dry_run=false", {}, False, False),  # LAST wins
        ("dry_run=true&dry_run=", {}, True, False),        # LAST wins: empty
        ("dry_run=off", {}, True, True),           # 'off' means DRY here
        ("dry_run=no", {}, False, False),          # 'no' means COMMIT here
        ("dry_run=0", {}, False, False),
        ("dry_run= false ", {}, False, False),     # stripped, so commit
        # ---- body wins, in the BODY's dialect, which is bool() ----
        ("", {"dry_run": True}, True, True),
        ("", {"dry_run": False}, False, False),
        ("", {"dry_run": None}, False, False),     # <-- merge goes LIVE
        ("", {"dry_run": ""}, False, False),       # <-- merge goes LIVE
        ("", {"dry_run": 0}, False, False),        # <-- merge goes LIVE
        ("", {"dry_run": []}, False, False),       # <-- merge goes LIVE
        ("", {"dry_run": "false"}, True, True),    # <-- compact goes DRY
        ("", {"dry_run": "0"}, True, True),        # <-- compact goes DRY
        ("", {"dry_run": "no"}, True, True),
        ("dry_run=true", {"dry_run": None}, False, False),  # body beats query
    ]

    print()
    print("       query / body                          merge   compact")
    print("       " + "-" * 62)
    bad = []
    for qs, body, want_merge, want_compact in cases:
        got_m = fn(req(qs), body, default=True)
        got_c = fn(req(qs), body, default=False)
        shown = f"?{qs}" if qs else "(none)"
        if body:
            shown += f" + {body}"
        m = "DRY " if got_m else "LIVE"
        c = "DRY " if got_c else "LIVE"
        flag = "" if (got_m == want_merge and got_c == want_compact) else "  <-- UNEXPECTED"
        print(f"       {shown:<38}{m}    {c}{flag}")
        if got_m != want_merge or got_c != want_compact:
            bad.append((qs, body, got_m, got_c))

    check(not bad, f"the table above is the behaviour actually observed ({bad})")

    # THE FINDINGS, asserted so a fix turns them red.
    print()
    check(fn(req("dry_run="), {}, default=False) is False,
          "F1: ?dry_run= (empty value) on /compact is a LIVE RUN")
    check(fn(req("dry_run"), {}, default=False) is False,
          "F2: ?dry_run with no '=' on /compact is a LIVE RUN")
    check(fn(req("dryrun=true"), {}, default=False) is False,
          "F3: a typo in the KEY on /compact is a LIVE RUN — the docstring's "
          "'anything else, INCLUDING A TYPO, leaves the caller in the safe "
          "direction' holds for the VALUE only")
    check(fn(req("dry_run=true&dry_run="), {}, default=False) is False,
          "F4: ?dry_run=true&dry_run= discards the true and runs LIVE")
    check(fn(req(""), {"dry_run": None}, default=True) is False,
          "F5: {'dry_run': null} makes /merge-into COMMIT — the merge "
          "endpoint takes the COMPACT endpoint's default")
    check(fn(req(""), {"dry_run": "false"}, default=False) is True,
          "F6: {'dry_run': 'false'} makes /compact a DRY RUN, while "
          "?dry_run=false on the same endpoint COMMITS — two dialects "
          "inside the one function written to end dialect divergence")
    check(fn(req("dry_run=no"), {}, default=True) is False,
          "F7: ?dry_run=no COMMITS here, while FastAPI's own bool parser on "
          "the sibling /admin/cleanup-test-conversations reads 'no' as "
          "dry_run=False, i.e. also commit — but ?dry_run=off is DRY here "
          "and commit there")
    check(fn(req("dry_run=off"), {}, default=True) is True,
          "F7b: ...and 'off' is the token where the two siblings disagree")


# ===========================================================================
# [A5-2]  supervisord's boolean dialect: four env vars that can kill PID 1
# ===========================================================================

def _supervisor_boolean():
    """The REAL supervisor.datatypes.boolean, extracted from the installed
    package's source. Imported normally it drags in `grp`, which does not
    exist off-Unix; extracting the function and its two constants runs the
    shipped bytes with no import at all.
    """
    import importlib.util
    spec = importlib.util.find_spec("supervisor.datatypes")
    if spec is None or spec.origin is None:
        return None, None
    p = Path(spec.origin)
    src = p.read_text(encoding="utf-8")
    tree = ast.parse(src)
    ns: dict = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in ("TRUTHY_STRINGS", "FALSY_STRINGS")
            for t in node.targets
        ):
            exec(compile(ast.Module([node], []), str(p), "exec"), ns)
    fn, _ = load_function(p, "boolean", ns)
    return fn, p


def test_supervisord_boolean_gates_are_a_boot_failure():
    print()
    print("[A5-2] autostart=%(ENV_x)s — supervisord's boolean(), which RAISES")

    boolean, origin = _supervisor_boolean()
    if boolean is None:
        print("   SKIP  supervisor is not installed here; see the report for the "
              "source-quoted reasoning (a SKIP IS NOT A PASS)")
        FAILED.append("A5-2 could not run: supervisor not installed")
        return
    print(f"       extracted boolean() from {origin}")

    # The four env vars fed straight into `autostart=` in supervisord.conf.
    gates = ["STT_ENABLED", "TTS_ENABLED", "COMPACTOR_SELFTEST_ON_BOOT",
             "COMPACTOR_BACKUP_ENABLED", "WEBUIDB_SYNC_ENABLED"]
    conf = (ROOT / "supervisord.conf").read_text(encoding="utf-8")
    for g in gates:
        check(f"autostart=%(ENV_{g})s" in conf,
              f"supervisord.conf really gates a program on ENV_{g}")

    print()
    print("       value          supervisord boolean()")
    print("       " + "-" * 48)
    raises = []
    for v in ["true", "false", "True", "TRUE", "1", "0", "yes", "no", "on",
              "off", "true ", " true", "true\n", "enabled", "", "2", "y", "t",
              "disabled", "False."]:
        try:
            r = repr(boolean(v))
        except Exception as e:
            r = f"RAISES {type(e).__name__}"
            raises.append(v)
        print(f"       {v!r:<14} {r}")

    check("enabled" in raises and "" in raises and "true " in raises
          and "y" in raises and "2" in raises,
          "F8: a value outside {yes,true,on,1,no,false,off,0} RAISES inside "
          "supervisord's CONFIG LOAD. `exec supervisord` is the container's "
          "PID 1, so STT_ENABLED=enabled (or a trailing space, from a RunPod "
          "template field) is a container that never starts — vLLM, "
          "OpenWebUI and the compactor all down, nothing logged past the "
          "supervisord parse error")
    check("true " in raises and " true" in raises,
          "F8b: boolean() lowercases but does NOT strip, so whitespace "
          "around a correct value is fatal")


# ===========================================================================
# [A5-3]  STT_ENABLED=1 — supervisord starts it, the self-test stops seeing it
# ===========================================================================

def test_stt_enabled_dialect_divergence_hides_a_dead_service():
    print()
    print("[A5-3] STT_ENABLED/TTS_ENABLED: supervisord's dialect vs selftest's")

    boolean, _ = _supervisor_boolean()
    if boolean is None:
        FAILED.append("A5-3 could not run: supervisor not installed")
        print("   SKIP  supervisor not installed (A SKIP IS NOT A PASS)")
        return

    # selftest.py's rule, extracted from its own line rather than retyped.
    sel_src = (ROOT / "compactor" / "selftest.py").read_text(encoding="utf-8")
    line = next(l for l in sel_src.splitlines() if l.startswith("STT_ENABLED ="))
    print(f"       selftest.py: {line.strip()}")
    check('.strip().lower() == "true"' in line,
          "selftest's rule is an exact-'true' comparison")

    def selftest_rule(v):
        return v.strip().lower() == "true"

    print()
    print("       STT_ENABLED    supervisord starts stt?   selftest probes it?")
    print("       " + "-" * 62)
    divergent = []
    for v in ["true", "TRUE", "1", "yes", "on", "false", "0", "no", "off"]:
        try:
            sup = boolean(v)
        except Exception:
            sup = "BOOT FAILURE"
        sel = selftest_rule(v)
        mark = ""
        if sup is True and sel is False:
            divergent.append(v)
            mark = "  <-- STARTED BUT NEVER PROBED"
        print(f"       {v!r:<14} {str(sup):<24}  {sel}{mark}")

    check(divergent == ["1", "yes", "on"],
          f"F9: STT_ENABLED in {{1, yes, on}} starts the Whisper service and "
          f"removes its probe from the boot self-test (got {divergent})")

    # And the self-test's denominator moves with the check list, so the
    # missing probe cannot be seen in the report either.
    check("\"total\": len(checks)" in sel_src,
          "F9b: selftest's summary total is len(checks) — a dropped probe "
          "changes 7/7 to 5/5 and the status stays 'pass', so the operator's "
          "green line is the same green line")


# ===========================================================================
# [A5-4]  The shell dialect: WEBUI_DB_LOCAL, and the default that is dangerous
# ===========================================================================

def test_shell_boolean_dialect_and_the_dangerous_default():
    print()
    print("[A5-4] entrypoint.sh — `[ \"$X\" = \"true\" ]`, byte equality")

    entry = (ROOT / "entrypoint.sh").read_text(encoding="utf-8")

    check('export WEBUI_DB_LOCAL="${WEBUI_DB_LOCAL:-true}"' in entry,
          "WEBUI_DB_LOCAL DEFAULTS TO true — and true is the value that moves "
          "the live database off /data. The default IS the dangerous value")
    check('if [ "${WEBUI_DB_LOCAL}" = "true" ]; then' in entry,
          "...and it is tested with untrimmed, case-sensitive byte equality")
    check("WEBUI_DB_LOCAL" not in (ROOT / "runpod.env.template").read_text(encoding="utf-8"),
          "F10: WEBUI_DB_LOCAL is a RunPod template variable by entrypoint's "
          "own account and is NOT documented in runpod.env.template, so the "
          "one boolean whose wrong value moves her chat history is the one an "
          "operator has no spelling for")

    def sh_eq_true(v):
        return v == "true"

    print()
    print("       WEBUI_DB_LOCAL   local-disk move happens?")
    print("       " + "-" * 46)
    for v in ["true", "True", "TRUE", "1", "yes", "on", "true ", " true", "false"]:
        m = "  <-- means FALSE" if (not sh_eq_true(v) and v.strip().lower() in
                                    ("true", "1", "yes", "on")) else ""
        print(f"       {v!r:<16} {sh_eq_true(v)}{m}")
    check(not sh_eq_true("True") and not sh_eq_true("1") and not sh_eq_true("true "),
          "F11: True / 1 / 'true ' all silently mean false (known M9 item). "
          "Direction is fail-safe for the v3.1.4.x series and inverted at the "
          "step-6 migration, whose whole instruction is to set it to true")

    # The sibling with the OPPOSITE spelling, four hundred lines apart.
    check('if [ "${WEBUI_DB_ALLOW_EMPTY_START}" != "true" ]; then' in entry,
          "the sibling flag uses != \"true\", so its typo direction is refuse")


# ===========================================================================
# [A5-5]  The AST detector's gaps, demonstrated on real shipped shapes
# ===========================================================================

def _tsrc_gap() -> str:
    return (ROOT / "compactor" / "test_envcfg.py").read_text(encoding="utf-8")


def test_ast_detector_blind_spots():
    print()
    print("[A5-5] test_envcfg's _unsoftened_env_conversions — what it cannot see")

    sys.path.insert(0, str(ROOT / "compactor"))
    import test_envcfg as T  # the real detector

    # 1. The shipped tree really is clean for the shapes it DOES catch.
    offenders = []
    for p in T._shipped_modules():
        offenders += T._unsoftened_env_conversions(
            p.read_text(encoding="utf-8", errors="replace"),
            p.relative_to(T.ROOT).as_posix())
    check(offenders == [],
          f"baseline: the 26 scanned modules are clean for the four caught "
          f"shapes (got {offenders})")

    # 2. Two hops. STATED gap, but this is the shape that is one edit away
    #    from existing: main.py already writes `TOKENIZE_WARN_INTERVAL_S =
    #    float(_env_int(...))`, i.e. a conversion over a conversion.
    two_hop = '''
import os
raw = os.environ.get("MAX_MODEL_LEN", "32768")
trimmed = raw.strip()
MAX_MODEL_LEN = int(trimmed)
'''
    eq(T._unsoftened_env_conversions(two_hop, "<two-hop>"), [],
       "GAP 1: env -> raw -> trimmed -> int(trimmed) is INVISIBLE")

    # 3. An unannotated string helper. tts/ and stt/ both annotate `_env`
    #    `-> str`, which is the only reason sites 8-10 were catchable. Drop
    #    the annotation and the identical defect disappears.
    unannotated = '''
import os
def _env(name, default):
    v = os.environ.get(name)
    return v if v is not None and v != "" else default
TTS_PORT = int(_env("TTS_PORT", "9001"))
'''
    eq(T._unsoftened_env_conversions(unannotated, "<unannotated>"), [],
       "GAP 2: the EXACT defect 91d8463 fixed in tts/server.py is invisible "
       "again if the helper loses its `-> str`")

    # 4. A converter outside int/float/complex, over an env value, at module
    #    scope, in a module main.py imports. Each of these raises on a string
    #    an operator can type, and each is therefore the same dead container.
    other_converters = '''
import json, os, re, datetime
from pathlib import Path
PINS = json.loads(os.environ.get("COMPACTOR_PIN_RULES", "[]"))
SKIP = re.compile(os.environ.get("COMPACTOR_SKIP_PATTERN", ".^"))
SINCE = datetime.datetime.fromisoformat(os.environ.get("COMPACTOR_SINCE", "2026-01-01"))
RATIO = os.environ.get("COMPACTOR_RATIO", "1/2")
NUM, DEN = (int(x) for x in RATIO.split("/"))
BITS = bytes.fromhex(os.environ.get("COMPACTOR_KEY", "00"))
'''
    hits = T._unsoftened_env_conversions(other_converters, "<other>")
    eq(hits, [],
       "GAP 3: json.loads / re.compile / fromisoformat / bytes.fromhex over "
       "an env value are all INVISIBLE, and every one of them raises on a "
       "value an operator can type")

    # 5. NOT a stated gap, and this is the finding: a generator/comprehension
    #    binding. _env_tainted_names only walks ast.Assign / ast.AnnAssign,
    #    so a name bound by `for`, by a walrus, by tuple-unpacking, or by
    #    `with ... as` is never tainted.
    comprehension = '''
import os
PORTS = [int(p) for p in os.environ.get("COMPACTOR_PORTS", "8080").split(",")]
'''
    eq(T._unsoftened_env_conversions(comprehension, "<comprehension>"), [],
       "GAP 4 (UNSTATED): a comprehension target is never tainted, so "
       "`[int(p) for p in os.environ.get(...).split(',')]` is invisible")

    # NOT a gap at HEAD. c0ff9db ("close the detector's bypasses") landed
    # after 91d8463 and taints a walrus target, so this shape IS caught.
    # Recorded because the report must not credit the detector with less
    # than it does, or more.
    walrus = '''
import os
if (raw := os.environ.get("MAX_MODEL_LEN")):
    MAX_MODEL_LEN = int(raw)
'''
    check(T._unsoftened_env_conversions(walrus, "<walrus>") != [],
          "NOT-A-GAP: a walrus binding IS tainted at HEAD (c0ff9db)")

    tupled = '''
import os
HOST, PORT = os.environ.get("COMPACTOR_ADDR", "0.0.0.0:8080").split(":")
PORT_I = int(PORT)
'''
    eq(T._unsoftened_env_conversions(tupled, "<tuple-unpack>"), [],
       "GAP 6 (UNSTATED): tuple-unpacking an env read taints NEITHER name — "
       "_env_tainted_names only adds targets that are a bare ast.Name")

    # NOT gaps at HEAD either — kwargs and dotted callees were closed by
    # c0ff9db. Asserted in the POSITIVE direction so that a regression of
    # that hardening turns this file red.
    kwarg = '''
import os
TIMEOUT = int(x=os.environ.get("COMPACTOR_TIMEOUT", "30"))
'''
    check(T._unsoftened_env_conversions(kwarg, "<kwarg>") != [],
          "NOT-A-GAP: int(x=...) IS caught at HEAD")

    dotted = '''
import os
import builtins
PORT = builtins.int(os.environ.get("COMPACTOR_PORT", "8080"))
'''
    check(T._unsoftened_env_conversions(dotted, "<dotted>") != [],
          "NOT-A-GAP: a dotted converter IS caught at HEAD")

    # THE GAP LIST IS WRONG ABOUT ITSELF. The docstring still says, at HEAD:
    #   "A try that catches the wrong exception (`except OSError:` around an
    #    int()) reads as softened here but is not."
    # It does not. _VALUE_GUARDS is {ValueError, TypeError, Exception,
    # BaseException}, so OSError does not soften and the site IS reported.
    # The detector is stricter than its own documentation, and the gap list
    # is the part nothing tests.
    wrong_exc = '''
import os
try:
    PORT = int(os.environ.get("COMPACTOR_PORT", "8080"))
except OSError:
    PORT = 8080
'''
    hits_we = T._unsoftened_env_conversions(wrong_exc, "<wrong-exc>")
    check(hits_we != [],
          "F19: the stated gap 'except OSError reads as softened here but is "
          "not' is FALSE at HEAD — the site IS reported "
          f"({hits_we}). A gap list nothing tests is a docstring that can "
          "argue the next reader into the wrong conclusion in either "
          "direction")
    check("reads as softened here but is not" in _tsrc_gap(),
          "F19b: ...and the claim is still in the shipped docstring")

    # 9. SCOPE. _shipped_modules globs *.py NON-RECURSIVELY in four fixed
    #    directories. A package subdirectory under compactor/ is unscanned,
    #    and so is log-sweep.py at the repo root.
    check(not any(p.name == "log-sweep.py" for p in T._shipped_modules()),
          "GAP 10: log-sweep.py sits at the repo root and is not scanned")
    _tsrc = Path(T.__file__).read_text(encoding="utf-8")
    _shipped_src = ast.get_source_segment(
        _tsrc,
        next(n for n in ast.parse(_tsrc).body
             if isinstance(n, ast.FunctionDef) and n.name == "_shipped_modules"))
    check('d.glob("*.py")' in (_shipped_src or ""),
          "GAP 10b: the glob is `d.glob('*.py')` — NOT rglob, so any "
          "sub-package added under compactor/ is unscanned from the day it "
          "is created, which is the same 'a list is only as complete as the "
          "last person to think about it' the section was written against")

    # 10. THE REAL, CURRENTLY-SHIPPING INSTANCE. The detector's own CONTROL
    #     says "every directory Dockerfile COPYs a .py from belongs here".
    #     That is the wrong criterion: scripts/ reaches the pod by VOLUME,
    #     not by COPY — entrypoint.sh's own refusal banner tells the
    #     operator to run /data/scripts/recover-webui-db.py — so the
    #     criterion excludes exactly the directory the operator types
    #     commands from during an incident.
    ms = ROOT / "scripts" / "merge-conversations.py"
    hits = T._unsoftened_env_conversions(
        ms.read_text(encoding="utf-8"), "scripts/merge-conversations.py")
    check(hits != [],
          f"F20: scripts/merge-conversations.py carries a LIVE raw env "
          f"conversion the detector would flag if it looked: {hits}")
    line = ms.read_text(encoding="utf-8").splitlines()[73]
    check('int(os.environ.get("COMPACTOR_PORT", "8080"))' in line,
          f"F20b: it is `{line.strip()}` — evaluated at argparse-SETUP time, "
          "so COMPACTOR_PORT=8O80 makes the tool die before --port can "
          "override it. COMPACTOR_PORT is a Dockerfile ENV and a RunPod "
          "template variable, and RUNBOOK_MEMORY_IDENTITY.md:115 is the "
          "un-fork step that runs this script")
    check(not any(p.name == "merge-conversations.py" for p in T._shipped_modules()),
          "F20c: ...and scripts/ is outside SHIPPED_DIRS, so [7] is green "
          "over it")
    entry = (ROOT / "entrypoint.sh").read_text(encoding="utf-8")
    check("/data/scripts/recover-webui-db.py" in entry,
          "F20d: entrypoint.sh's own boot-refusal banner sends the operator "
          "to /data/scripts/, which the Dockerfile never COPYs — the "
          "CONTROL's completeness criterion cannot see it")


# ===========================================================================
# [A5-6]  degrade.guard's TTL: job 2's new guard cannot see the disk fill
# ===========================================================================

def test_degrade_ttl_makes_job2s_new_guard_unable_to_fire():
    print()
    print("[A5-6] degrade.guard's TTL vs 'the disk can fill in between'")

    os.environ["COMPACTOR_DEGRADE_CHECK_TTL_S"] = "10"
    os.environ["COMPACTOR_MIN_FREE_MB_WRITES"] = "200"
    sys.path.insert(0, str(ROOT / "compactor"))
    for m in ("degrade", "envcfg", "logsetup"):
        sys.modules.pop(m, None)
    import degrade

    eq(degrade._CHECK_TTL_S, 10.0, "the shipped default TTL is 10 s")

    free = {"mb": 5000.0}
    calls = []
    degrade._free_mb = lambda path: (calls.append(path), free["mb"])[1]
    degrade._reset_cache_for_tests()

    # t0: the request path's check (_tail_store_blocked).
    eq(degrade.guard("async memory tail"), True, "t0 request path: allowed")
    # the disk fills one second later, while job 1 indexes and job 2 waits on
    # a vLLM extraction call.
    free["mb"] = 3.0
    eq(degrade.guard("async memory tail"), True,
       "t0+1s _async_tail's top check: still ALLOWED from the cached reading")
    eq(degrade.guard("fact extraction tail"), True,
       "F12: t0+3s job 2's NEW guard: ALSO ALLOWED — it answers from the "
       "same cached tuple, so the guard added for 'the disk can fill between "
       "that check and this write' cannot observe a disk that fills inside "
       "the TTL window")
    eq(degrade.guard("hierarchy rollup"), True,
       "...and so does job 3's, which has had the same guard since v3.1.8")
    eq(len(calls), 1,
       "one statvfs for all four checks — the comment's 'this is a tuple "
       "read, not a second statvfs' is exactly why it cannot fire")

    # It only fires once the TTL has expired, i.e. only when the tail takes
    # LONGER than COMPACTOR_DEGRADE_CHECK_TTL_S from the request-path check.
    degrade._cache = (degrade.time.monotonic() - 0.001,
                      degrade._cache[1], degrade._cache[2])
    eq(degrade.guard("fact extraction tail"), False,
       "past the TTL it does fire — so the window it closes starts at "
       "t0 + 10 s, where t0 is the REQUEST-path reading, not _async_tail's")

    # And the label is decoration: guard() ignores it for the decision.
    degrade._reset_cache_for_tests()
    free["mb"] = 3.0
    labels = ["async memory tail", "fact extraction tail", "hierarchy rollup",
              "persona auto-capture", "not a real label at all"]
    answers = {l: degrade.guard(l) for l in labels}
    check(set(answers.values()) == {False},
          "F13: guard()'s answer is label-INDEPENDENT — every label reads the "
          "one cached tuple. [E10c]'s `lambda op: op != blocked_label` "
          "constructs a state the shipped code cannot be in, so it proves the "
          "guard is CALLED with that label, not that job 2 refuses under real "
          "disk pressure mid-tail")

    # The label collision the sweep did leave behind.
    main_src = MAIN.read_text(encoding="utf-8")
    n = main_src.count('degrade.guard("async memory tail")')
    eq(n, 2,
       "F14: 'async memory tail' is used by TWO call sites (_tail_store_blocked "
       "and _async_tail), so their debug lines are indistinguishable and a "
       "label-selective test patch blocks both at once. Job 2's own label "
       "'fact extraction tail' is unique — that half is clean")
    eq(main_src.count('degrade.guard("fact extraction tail")'), 1,
       "job 2's label occurs exactly once")


# ===========================================================================
# [A5-16]  The ordering hazard _PESSIMISTIC_SUMMARY_SCALE was fixed for,
#          checked mechanically over every module-level env read in the tree
# ===========================================================================

def test_no_module_level_env_call_precedes_its_definition():
    print()
    print("[A5-16] every module-scope env_* call is defined ABOVE its use")

    sys.path.insert(0, str(ROOT / "compactor"))
    import test_envcfg as T

    HELPERS = {"env_int", "env_float", "env_window_s",
               "_env_int", "_env_float", "_window_s"}
    problems = []
    checked = 0
    for p in T._shipped_modules():
        rel = p.relative_to(T.ROOT).as_posix()
        src = p.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=rel)
        # where each helper name becomes bound in this module
        bound: dict[str, int] = {}
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in HELPERS:
                bound[node.name] = node.lineno
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for a in node.names:
                    nm = a.asname or a.name
                    if nm in HELPERS:
                        bound[nm] = node.lineno
        # module-scope calls only: walk top-level statements, skipping defs
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            for n in ast.walk(node):
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) \
                        and n.func.id in HELPERS:
                    checked += 1
                    where = bound.get(n.func.id)
                    if where is None:
                        problems.append(f"{rel}:{n.lineno}: {n.func.id} is not "
                                        f"bound anywhere at module scope")
                    elif where > n.lineno:
                        problems.append(f"{rel}:{n.lineno}: calls {n.func.id} "
                                        f"defined below at line {where} — "
                                        f"NameError at import")
    check(checked >= 40,
          f"CONTROL: the scan reached {checked} module-scope env_* calls "
          f"(a scan that found none would pass for the worst reason)")
    eq(problems, [],
       "the ordering hazard 91d8463 avoided at main.py:159 does not exist "
       "anywhere else in the shipped tree")
    # And the specific line the commit is about.
    src = MAIN.read_text(encoding="utf-8")
    check('_PESSIMISTIC_SUMMARY_SCALE = env_float(' in src,
          "main.py:159 uses envcfg.env_float (imported at line 50), not the "
          "local _env_float defined at line 186")
    check("from envcfg import env_float" in src,
          "...and that import is present")
    check("from envcfg import env_float\n" in src and "env_int" not in
          src.split("from envcfg import env_float")[0].split("\n")[-1],
          "main.py imports env_float ONLY — a future module-scope env_int "
          "call in main.py would be a NameError, which is the same hazard "
          "one name over")


# ===========================================================================
# [A5-17]  One env var, two converters, and a comment asserting they agree
# ===========================================================================

def test_tokenize_warn_interval_is_read_as_int_here_and_float_there():
    print()
    print("[A5-17] COMPACTOR_TOKENIZE_WARN_INTERVAL_S: int in main, float in "
          "summarizer")

    msrc = MAIN.read_text(encoding="utf-8")
    ssrc = (ROOT / "compactor" / "summarizer.py").read_text(encoding="utf-8")

    check('TOKENIZE_WARN_INTERVAL_S = float(_env_int('
          '"COMPACTOR_TOKENIZE_WARN_INTERVAL_S", 300))' in msrc,
          "main.py reads it through _env_int and then widens to float")
    check('TOKENIZE_WARN_INTERVAL_S = env_float('
          '"COMPACTOR_TOKENIZE_WARN_INTERVAL_S", 300)' in ssrc,
          "summarizer.py reads the SAME variable through env_float")
    check("Same env var and default main.py reads" in ssrc
          and "deliberately, not independently" in ssrc
          and "an operator setting this once should govern every /tokenize"
              in ssrc,
          "F28: summarizer.py's own comment ASSERTS the two agree — 'Same "
          "env var and default main.py reads ... deliberately, not "
          "independently tuned: an operator setting this once should govern "
          "every /tokenize dependency in the process'")
    _l693 = msrc.splitlines()[692]
    check("(main.py:693, TOKENIZE_WARN_INTERVAL_S)" in ssrc
          and "TOKENIZE_WARN_INTERVAL_S" not in _l693,
          "F28c: ...and the comment's own line citation is stale — it says "
          f"main.py:693 and that line is `{_l693.strip()}` (inside the "
          "image-stripping helper). The constant is at main.py:761. The one "
          "pointer a reader would follow to check the agreement claim lands "
          "on unrelated code")

    sys.path.insert(0, str(ROOT / "compactor"))
    for m in ("envcfg",):
        sys.modules.pop(m, None)
    import envcfg

    def main_rule(v):
        os.environ["COMPACTOR_TOKENIZE_WARN_INTERVAL_S"] = v
        # main._env_int is byte-identical to envcfg.env_int (both checked in
        # test_envcfg [3]); using envcfg's avoids importing main.
        return float(envcfg.env_int("COMPACTOR_TOKENIZE_WARN_INTERVAL_S", 300))

    def summarizer_rule(v):
        os.environ["COMPACTOR_TOKENIZE_WARN_INTERVAL_S"] = v
        return envcfg.env_float("COMPACTOR_TOKENIZE_WARN_INTERVAL_S", 300)

    print()
    print("       value      main.py   summarizer.py")
    print("       " + "-" * 42)
    diverge = []
    for v in ["300", "0.5", "60.5", "1e3", "600"]:
        a, b = main_rule(v), summarizer_rule(v)
        mark = "  <-- DISAGREE" if a != b else ""
        print(f"       {v!r:<10} {a:<9} {b}{mark}")
        if a != b:
            diverge.append(v)
    os.environ.pop("COMPACTOR_TOKENIZE_WARN_INTERVAL_S", None)
    check(diverge == ["0.5", "60.5", "1e3"],
          f"F28b: any non-integer spelling takes effect in summarizer.py and "
          f"silently reverts to 300 in main.py (diverging on {diverge}). "
          f"test_envcfg [4] asserts bgwork and tailhealth agree about their "
          f"shared window; this pair has the same shape and no such test")


ALL = [
    test_dry_run_from_parsing_table,
    test_supervisord_boolean_gates_are_a_boot_failure,
    test_stt_enabled_dialect_divergence_hides_a_dead_service,
    test_shell_boolean_dialect_and_the_dangerous_default,
    test_ast_detector_blind_spots,
    test_degrade_ttl_makes_job2s_new_guard_unable_to_fire,
    test_no_module_level_env_call_precedes_its_definition,
    test_tokenize_warn_interval_is_read_as_int_here_and_float_there,
]

if __name__ == "__main__":
    print("=" * 74)
    print("AREA 5 hostile pass #2 — config / dry_run / booleans / detector")
    print("=" * 74)
    for t in ALL:
        t()
    print()
    if FAILED:
        print(f"{len(FAILED)} assertion(s) FAILED:")
        for f in FAILED:
            print(f"  - {f}")
        sys.exit(1)
    print("all assertions held (each one encodes a finding; see the report)")
