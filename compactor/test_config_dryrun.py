"""compactor/test_config_dryrun.py — _dry_run_from's full parsing table.

v3.1.9 HIGH #1 and #2 (hostile pass 2 on 843bf9d, findings/hostile2-config.md).
_dry_run_from unified the two ADMIN ENDPOINTS that read dry_run (admin_merge,
admin_compact) so the rule could not be applied at one and missed at the
other again. It turned out to have left two more asymmetries inside itself:

  HIGH #1 — the QUERY STRING treated a PRESENT-BUT-AMBIGUOUS value exactly
  like an ABSENT one and returned `default`. On /compact, default is a live
  run, so `?dry_run=` (what `curl ".../compact?dry_run=$FLAG"` sends when
  $FLAG is unset), a bare `?dry_run`, a misspelled key (`?dryrun=true`,
  `?dry-run=true`), and `?dry_run=true&dry_run=` (Starlette's
  QueryParams.get is last-wins, so the empty repeat silently discarded the
  `true`) were all LIVE RUNS — up to 200 vLLM calls, a rewritten state file,
  an advanced watermark — for an operator who typed a flag at all, however
  malformed.

  HIGH #2 — the BODY used `bool(body["dry_run"])`, Python truthiness, so
  every JSON-falsy value present under the key (`null`, `""`, `0`, `[]` — all
  of what `{"dry_run": $FLAG}` through jq or envsubst produces from an unset
  variable) meant COMMIT, while `{"dry_run": "false"}` — a non-empty string —
  meant DRY: the opposite of what `?dry_run=false` does on the very same
  endpoint. Two dialects inside the one function written to end dialect
  divergence.

GATE REVIEW round (same day, same function) — two more holes of the
identical shape, found by re-reading the first fix rather than the shipped
defect:

  GATE (a) — the query fix above still read
  `request.query_params.get("dry_run")`, which is Starlette's last-wins
  accessor, so `?dry_run=true&dry_run=false` read "false" and COMMITTED — an
  operator who sent one true and one false in the same request got the
  write, not the safer answer. Fixed by reading every repeated value with
  `getlist` and requiring ALL of them to be a commit token before the query
  source says commit.

  GATE (b) — the body branch returned immediately, so `{"dry_run": false}`
  with `?dry_run=true` on the SAME request committed without the query
  string ever being consulted — an explicit dry in the query was silently
  overridden by the body. Fixed by evaluating both sources that are actually
  present to a verdict and combining them: commit only if EVERY present
  source says commit; any present source saying dry makes the whole call
  dry; `default` is read only when neither source is present at all.

This file is the truth table both findings were proved against, run directly
against the real `main._dry_run_from` (not a lookalike), at BOTH `default`
values, because the fix's rule is "ambiguity resolves to dry independent of
default" and a table exercised at only one default cannot tell "always dry"
from "dry because default happened to be dry".

It also closes v3.1.9 MEDIUM #10 (four `_dry_run_from` mutations survived
both shipped suites: dropping '0'/'no' from the commit tuple, two body-dialect
corrections, and dropping .strip() from the query read) — every row below
that carries a HIGH #1/#2 or MEDIUM #10 tag is the row that mutation needed to
survive and now cannot, because the assertion is exact rather than directional.

No server, no model, no network:
    python test_config_dryrun.py
"""

import os
import sys
import tempfile

_TMP_ROOT = tempfile.mkdtemp(prefix="compactor-test-config-dryrun-")
os.environ["COMPACTOR_STORAGE_ROOT"] = _TMP_ROOT
os.environ["MODEL_REPO"] = "test-model"
os.environ["VLLM_URL"] = "http://stub:8000"
os.environ["COMPACTOR_RAG_ENABLED"] = "false"

import main  # noqa: E402
from starlette.datastructures import QueryParams  # noqa: E402

FAILED = []


def check(cond, label):
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}")
        FAILED.append(label)


class _FakeRequest:
    """The only thing _dry_run_from reads off `request` is `.query_params`,
    so a real starlette QueryParams on a duck-typed stand-in exercises the
    exact same parsing Starlette hands the endpoint — no TestClient, no app,
    no event loop needed for a pure parsing table."""

    def __init__(self, qs=""):
        self.query_params = QueryParams(qs)


def dry(qs="", body=None, *, default):
    return main._dry_run_from(_FakeRequest(qs), body if body is not None else {},
                               default=default)


# ---------------------------------------------------------------------------
# [A] Query-string dialect, at both defaults (merge's True/dry-default and
# compact's False/live-default). Absent is the ONLY case that reads
# `default`; every present-but-ambiguous case must be True (dry) at both.
# ---------------------------------------------------------------------------

print("[A] query-string dialect")
print()

# (query string,                    dry @ default=True,  dry @ default=False,  tag)
QUERY_CASES = [
    ("dry_run=true",                 True,  True,  None),
    ("",                              True,  False, "absent reads `default`"),
    ("dry_run=false",                False, False, None),
    ("dry_run=1",                    True,  True,  None),
    ("dry_run=TRUE",                 True,  True,  None),
    ("dry_run=yes",                  True,  True,  None),
    ("dry_run=ture",                 True,  True,  "typo VALUE: safe direction (unchanged)"),
    ("dry_run=",                     True,  True,  "HIGH #1: was `default` (LIVE on /compact)"),
    ("dry_run",                      True,  True,  "HIGH #1: bare flag, was `default`"),
    ("dryrun=true",                  True,  True,  "HIGH #1: misspelled key, was `default`"),
    ("dry-run=true",                 True,  True,  "HIGH #1: misspelled key (hyphen), was `default`"),
    ("dry_run=false&dry_run=true",   True,  True,  "not every value is a commit token ('true' isn't)"),
    ("dry_run=true&dry_run=false",   True,  True,  "GATE (a): was LIVE via last-wins ('false' picked); "
                                                    "now DRY - 'true' in the list is not a commit token"),
    ("dry_run=true&dry_run=",        True,  True,  "HIGH #1: last-wins empty, was `default`"),
    ("dry_run=false&dry_run=false",  False, False, "GATE (a) CONTROL: repeated but AGREEING commit "
                                                    "tokens still commit - this is not 'always dry now'"),
    ("dry_run=off",                  True,  True,  None),
    ("dry_run=no",                   False, False, "MEDIUM #10 (M-b): 'no' must still commit"),
    ("dry_run=0",                    False, False, "MEDIUM #10 (M-b): '0' must still commit"),
    ("dry_run=%20false",             False, False, "MEDIUM #10 (M-f): leading space still strips"),
]

for qs, want_t, want_f, tag in QUERY_CASES:
    label = f"?{qs or '(none)'}"
    if tag:
        label += f"  [{tag}]"
    got_t = dry(qs, default=True)
    got_f = dry(qs, default=False)
    check(got_t is want_t,
          f"{label}  @ default=True -> {'DRY' if want_t else 'LIVE'} "
          f"(got {'DRY' if got_t else 'LIVE'})")
    check(got_f is want_f,
          f"{label}  @ default=False -> {'DRY' if want_f else 'LIVE'} "
          f"(got {'DRY' if got_f else 'LIVE'})")

# CONTROL: this table is not "everything is dry now" — decisive commit
# tokens above (dry_run=false/no/0, and the last-wins case) all resolved to
# LIVE, so the guard can still say yes.
check(any(not want_f for _, _, want_f, _ in QUERY_CASES),
      "CONTROL: at least one query case still resolves LIVE")


# ---------------------------------------------------------------------------
# [B] Body dialect — now the SAME vocabulary as the query string, at both
# defaults. Absent is the only case that reads `default`.
# ---------------------------------------------------------------------------

print()
print("[B] body dialect")
print()

# (body,                    dry @ default=True, dry @ default=False, tag)
BODY_CASES = [
    ({"dry_run": True},      True,  True,  None),
    ({"dry_run": False},     False, False, None),
    ({"dry_run": "true"},    True,  True,  None),
    ({"dry_run": "TRUE"},    True,  True,  None),
    ({"dry_run": "false"},   False, False, "HIGH #2: was DRY (bool('false') truthy); now unified"),
    ({"dry_run": "0"},       False, False, "HIGH #2: string '0' now also commits, like the query"),
    ({"dry_run": "no"},      False, False, "HIGH #2: string 'no' now also commits"),
    ({"dry_run": None},      True,  True,  "HIGH #2: was LIVE (bool(None) is False); now dry"),
    ({"dry_run": ""},        True,  True,  "HIGH #2: was LIVE (bool('') is False); now dry"),
    ({"dry_run": 0},         True,  True,  "HIGH #2: was LIVE (bool(0) is False); now dry"),
    ({"dry_run": []},        True,  True,  "HIGH #2: was LIVE (bool([]) is False); now dry"),
    ({"dry_run": {"n": 1}},  True,  True,  "not a recognised shape: dry, not a crash"),
    ({"dry_run": 1},         True,  True,  "int 1 is not bool True: dry, not a silent coercion"),
    ({},                     True,  False, "absent from body falls through to the query/default"),
]

for body, want_t, want_f, tag in BODY_CASES:
    label = f"{body!r}"
    if tag:
        label += f"  [{tag}]"
    got_t = dry(body=body, default=True)
    got_f = dry(body=body, default=False)
    check(got_t is want_t,
          f"{label}  @ default=True -> {'DRY' if want_t else 'LIVE'} "
          f"(got {'DRY' if got_t else 'LIVE'})")
    check(got_f is want_f,
          f"{label}  @ default=False -> {'DRY' if want_f else 'LIVE'} "
          f"(got {'DRY' if got_f else 'LIVE'})")

# CONTROL: a real commit is still reachable through the body too.
check(any(not want_f for _, _, want_f, _ in BODY_CASES),
      "CONTROL: at least one body case still resolves LIVE")


# ---------------------------------------------------------------------------
# [C] Cross-source combination (GATE (b)). Before the gate fix, the body
# branch returned immediately and the query string was never consulted once
# a body was present at all — so a body saying commit silently beat a query
# saying dry, with no trace either happened. The rule now: commit only if
# EVERY present source says commit; any present source saying dry makes the
# whole call dry. This replaces the old "body always wins" precedence
# entirely — MEDIUM #10 (M-h), the precedence-flip mutation, is superseded
# by this table rather than re-asserted on its own terms, because "body
# wins" is no longer the rule to protect.
# ---------------------------------------------------------------------------

print()
print("[C] cross-source combination: commit only if EVERY present source agrees")
print()

# (query string,   body,                  dry@default=True, dry@default=False, tag)
CROSS_CASES = [
    ("dry_run=false", {},                  False, False, "CONTROL: query commit ALONE still commits"),
    ("",               {"dry_run": False}, False, False, "CONTROL: body commit ALONE still commits"),
    ("dry_run=false", {"dry_run": False},  False, False, "CONTROL: both say commit -> still commits"),
    ("dry_run=true",  {"dry_run": False},  True,  True,  "GATE (b): was LIVE (body returned first, "
                                                          "query never consulted); now DRY"),
    ("dry_run=false", {"dry_run": True},   True,  True,  "body dry beats query commit (unchanged "
                                                          "direction, now via the general rule)"),
    ("dry_run=true",  {"dry_run": True},   True,  True,  "both say dry -> dry"),
]

for qs, body, want_t, want_f, tag in CROSS_CASES:
    label = f"query={qs or '(none)'!r} body={body!r}  [{tag}]"
    got_t = dry(qs, body, default=True)
    got_f = dry(qs, body, default=False)
    check(got_t is want_t,
          f"{label}  @ default=True -> {'DRY' if want_t else 'LIVE'} (got {'DRY' if got_t else 'LIVE'})")
    check(got_f is want_f,
          f"{label}  @ default=False -> {'DRY' if want_f else 'LIVE'} (got {'DRY' if got_f else 'LIVE'})")

# CONTROL: this table is not "two sources present means always dry" either —
# the three commit rows above prove a real commit is still reachable through
# every combination of "who is present", not just through a single source.
check(any(not want_f for _, _, _, want_f, _ in CROSS_CASES),
      "CONTROL: at least one cross-source case still resolves LIVE")


if FAILED:
    print()
    print(f"{len(FAILED)} assertion(s) failed:")
    for label in FAILED:
        print(f"  - {label}")
    sys.exit(1)
print()
print("All _dry_run_from parsing-table tests passed.")
