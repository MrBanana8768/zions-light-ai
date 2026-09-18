"""Hostile pass #2, AREA 6: the AST detector as an ORACLE.

c0ff9db widened compactor/test_envcfg.py's detector from 4 shapes to 15 and
wrote a gap list "so the next person does not over-trust it", with everything
in it "measured, not guessed".

This file measures the gap list itself. It calls the SHIPPED
`_unsoftened_env_conversions` -- no copy, no re-implementation -- over sixteen
env-conversion shapes a working developer would plausibly write, and asserts
that every shape the gap list does NOT name is caught.

    python tests/adversarial/test_adv_v319_envcfg_oracle.py

Result at the time of writing: twelve of sixteen bypass the detector. Six of
the twelve are named in the gap list (fair warning, correctly given). Six are
NOT, and one of those six is the exact spelling the commit message claims to
have closed ("element-wise tuple unpacking").

Nothing in the shipped tree has any of these shapes today, so this is about
the oracle's honesty rather than a live defect: the detector is presented as
un-shippable-defect enforcement plus a complete, measured gap list, and the
gap list is not complete.
"""

from __future__ import annotations

import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
os.environ.setdefault("MODEL_REPO", "")
sys.path.insert(0, str(ROOT / "compactor"))

import test_envcfg as T  # noqa: E402


# Each entry: label -> (source, is_named_in_the_gap_list)
CASES: dict[str, tuple[str, bool]] = {
    # ---- shapes the gap list DOES name. Listed so the file proves the gap
    # list is accurate where it speaks, not only incomplete where it does not.
    "two hops through an `or` default": ("""
import os
_raw = os.environ.get("COMPACTOR_TIMEOUT", "30")
TIMEOUT_RAW = _raw or "30"
TIMEOUT = int(TIMEOUT_RAW)
""", True),
    "an unannotated _env helper": ("""
import os
def _env(name, default=""):
    return os.environ.get(name) or default
PORT = int(_env("STT_PORT", "9000"))
""", True),
    "a quoted annotation on the helper": ("""
import os
def _env(name: str, default: str) -> "str":
    return os.environ.get(name) or default
PORT = int(_env("PORT", "9000"))
""", True),
    "a try that catches but never rebinds": ("""
import os
try:
    RETRIES = int(os.environ.get("RETRIES", "3"))
except ValueError:
    pass
""", True),
    "json.loads, a converter outside int/float/complex": ("""
import os, json
ROUTES = json.loads(os.environ.get("ROUTES", "{}"))
""", True),
    "int() on a bare attribute of an env-derived object": ("""
import os
class _C:
    raw = os.environ.get("PORT", "9000")
CFG = _C()
PORT = int(CFG.raw)
""", True),

    # ---- shapes the gap list does NOT name.
    "a comma-separated env list through a comprehension": ("""
import os
PORTS = [int(p) for p in os.environ.get("PORTS", "9000,9001").split(",")]
""", False),
    "the same through a for loop": ("""
import os
PORTS = []
for p in os.environ.get("PORTS", "9000").split(","):
    PORTS.append(int(p))
""", False),
    "tuple unpacking whose RHS is a CALL, not a tuple literal": ("""
import os
_host, _port = os.environ.get("ADDR", "h:1").split(":")
PORT = int(_port)
""", False),
    "an env-defaulted function parameter": ("""
import os
def connect(timeout: str = os.environ.get("TIMEOUT", "5")):
    return int(timeout)
""", False),
    "an augmented assignment binding": ("""
import os
raw = ""
raw += os.environ.get("PORT", "9000")
PORT = int(raw)
""", False),
    "map(int, ...) -- no int() Call node holds an env argument": ("""
import os
PORTS = list(map(int, os.environ.get("PORTS", "9000").split(",")))
""", False),

    # ---- controls. These MUST be caught, or nothing above means anything.
    "CONTROL direct os.environ.get conversion": ("""
import os
PORT = int(os.environ.get("PORT", "9000"))
""", False),
    "CONTROL the house-style .strip() chain": ("""
import os
def _env(name: str, default: str) -> str:
    return os.environ.get(name) or default
PORT = int(_env("STT_PORT", "9000").strip())
""", False),
    "CONTROL a dict(os.environ) snapshot subscript": ("""
import os
CFG = dict(os.environ)
PORT = int(CFG["PORT"])
""", False),
    "CONTROL a walrus binding": ("""
import os
if (raw := os.environ.get("PORT")):
    PORT = int(raw)
""", False),
}

_CONTROLS = {k for k in CASES if k.startswith("CONTROL")}


def _scan(src: str) -> list[str]:
    return T._unsoftened_env_conversions(src, "<adv>")


def test_the_controls_are_still_caught():
    """If the detector stopped firing entirely, every case below would read as
    a bypass and this file would report sixteen findings and no defect."""
    missed = [k for k in _CONTROLS if not _scan(CASES[k][0])]
    assert not missed, (
        f"the detector no longer reports shapes it is documented to catch: "
        f"{sorted(missed)}. Read nothing into the rest of this file."
    )


def test_every_bypass_is_either_caught_or_named_in_the_gap_list():
    """A gap list is only worth trusting if it is complete.

    The file's own words: "WHAT IT DOES NOT CATCH, stated so the next person
    does not over-trust it. Everything in this list was measured, not
    guessed." These six shapes are neither caught nor listed.
    """
    undocumented = [
        label for label, (src, named) in CASES.items()
        if label not in _CONTROLS and not named and not _scan(src)
    ]
    assert not undocumented, (
        "shapes that bypass the detector and are NOT in its gap list:\n  "
        + "\n  ".join(undocumented)
        + "\nThe sharpest is the tuple unpacking: c0ff9db's message says it "
          "closed 'element-wise tuple unpacking', and _env_tainted_names only "
          "handles it when BOTH sides are Tuple/List literals of equal length. "
          "`_host, _port = os.environ.get(...).split(':')` -- the commoner "
          "spelling -- taints neither name."
    )


def test_the_named_gaps_really_are_gaps():
    """The reverse check: a gap list that names a shape the detector actually
    catches is stale, and a stale gap list gets trimmed by someone who then
    trims a live entry with it."""
    stale = [
        label for label, (src, named) in CASES.items()
        if named and _scan(src)
    ]
    assert not stale, (
        f"the gap list names shapes the detector DOES catch (so the list is "
        f"stale): {stale}"
    )


if __name__ == "__main__":
    print("shape-by-shape:")
    for _label, (_src, _named) in CASES.items():
        _hits = _scan(_src)
        _tag = "caught " if _hits else ("gap-listed" if _named else "BYPASS ")
        print(f"  {_tag}  {_label}")
    print()
    _fails = []
    for _name, _fn in sorted(globals().items()):
        if not (_name.startswith("test_") and callable(_fn)):
            continue
        try:
            _fn()
            print(f"  PASS  {_name}")
        except AssertionError as _e:
            print(f"  FAIL  {_name}")
            print(f"        {_e}")
            _fails.append(_name)
    print()
    print(f"{len(_fails)} oracle defect(s) reproduced.")
    sys.exit(1 if _fails else 0)
