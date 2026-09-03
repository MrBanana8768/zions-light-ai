"""
compactor.envcfg — shared, crash-safe environment-variable parsing.

THE DEFECT THIS EXISTS TO CLOSE (V314_BACKLOG R30 / V317_PLAN Stage 5 #13).
Roughly 35 sites across facts.py, summarizer.py, backup.py, pgarchive.py,
retrieval.py, webuidb.py, degrade.py, alert.py, selftest.py and bgwork.py
read configuration as a bare `int(os.environ.get(...))` or
`float(os.environ.get(...))` — some softened with `or default`, which only
rescues an EMPTY string, not a bad one. Every one of those sites is an
import-time crash on a typo, and every one of these modules is imported at
main.py module scope (directly, or transitively via main -> facts ->
retrieval, etc.), so the failure is not "one feature degrades" — it is a
container that will not boot, with a traceback nobody connects to a config
line. `MAX_MODEL_LEN=32K` demonstrates it concretely: main.py's own copy of
that read was fixed in v3.1.7 (main._env_int), but facts.py and
summarizer.py each kept an independent bare `int(...)` reading the SAME
variable, so the exact same typo still stops the boot through either of
them. That is this project's recurring defect — a fix applied at one call
site and missed at its siblings — and the reason this is a shared helper
instead of 35 individual try/excepts.

THE CONTRACT, and it is deliberately narrow:

  * An unset, blank, or unparseable value returns the DEFAULT. NEVER a
    raise. This is the one rule that must hold everywhere: a typo in
    runpod.env must degrade the affected knob to its default, not take
    down the process. That much is not a judgment call.

  * `env_int` / `env_float` do NOT police range. An explicit `0`, a
    negative, `nan` or `inf` is returned exactly as given. This mirrors
    main._env_float's contract (see its docstring, corrected in v3.1.7 for
    the same reason): callers disagree about what a legal range is. A
    budget fraction of 0 silently disables its whole layer and is
    almost certainly a mistake; COMPACTOR_MAX_RETAINED_IMAGES takes 0 as a
    meaningful "keep none"; PGARCHIVE_ALLOW_SHRINK-style flags aren't even
    numeric. One shared rule that rejected all non-positive values would
    quietly break every knob in the second group to fix the first, so the
    range decision is left to the caller — which is where the 35 call
    sites already made that decision, in most cases via a downstream
    `max(floor, ...)` (backup.MIN_KEEP, pgarchive.MIN_KEEP, ...) that this
    change leaves untouched.

  * `env_window_s` is the one exception, carried over unchanged from
    bgwork._window_s / tailhealth._window_s (both pre-existing, both
    written for exactly this "degrade window" shape): it additionally
    rejects non-positive AND non-finite values, because for a health
    degrade window specifically, 0 doesn't mean "off" the way it does for
    e.g. MAX_RETAINED_IMAGES — it means "always true", the always-on
    warning the window exists to prevent, and `inf` means the same thing
    by a different route (`math.isfinite` catches it; `nan` already fails
    `> 0`). See bgwork._window_s's docstring for the full incident this
    was fixed for. This is the only site-specific policy folded into this
    module, because it already existed in two places and the alternative
    was a third copy.

WHAT THIS MODULE DOES NOT DO: log. Logging is not configured this early —
every one of these reads happens at module import, before logsetup runs —
so a value that fails to parse and falls back is invisible here. That is
the same tradeoff main._env_int already made and documented: "a bad value
is logged nowhere because logging is not configured this early... an
operator who set a value that did not take will see it in /health/full's
config block." This module does not change that; it only generalizes it.
Nothing stops a caller from logging AFTER its own module's logger exists,
but none of the 35 sites this change touches previously did that either,
so none gained it here — that would be a behavior change beyond the scope
of "stop crashing on a typo."

NOTHING IN THIS PACKAGE MAY IMPORT `main` FROM HERE. That is precisely why
bgwork._window_s and tailhealth._window_s were separate, duplicated
functions instead of one shared helper: main.py imports both bgwork and
tailhealth, so neither can import main back without a cycle. This module
has zero imports from the compactor package for the same reason — main.py,
facts.py, summarizer.py, backup.py, pgarchive.py, retrieval.py, webuidb.py,
degrade.py, alert.py, selftest.py and bgwork.py can all import IT with no
risk of a cycle, however the graph among the others shifts later.
"""

from __future__ import annotations

import math
import os


def env_int(name: str, default: int) -> int:
    """Read an int-valued env var. Unset, blank, or unparseable -> default.

    `os.environ.get` returns `""` (not the default) when the var is set to
    an empty string, which is what .env files do for opt-in blanks — treat
    that the same as unset. A value that is present but not a valid int
    (`5O` for `50`, a stray quote, a trailing comment) returns the default
    too, rather than raising: see the module docstring for why that is the
    one non-negotiable rule here.
    """
    v = os.environ.get(name, "")
    if not v.strip():
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def env_float(name: str, default: float) -> float:
    """Read a float-valued env var. Unset, blank, or unparseable -> default.

    Same contract as env_int, and the same non-policing of range: an
    explicit `0`, a negative, `nan` or `inf` is returned as given. See the
    module docstring — callers disagree about what a legal range is, so a
    caller that needs a positive (or otherwise restricted) value must say
    so itself, the way env_window_s does for degrade windows.
    """
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def env_window_s(name: str, default: float) -> float:
    """Read a degrade-window seconds value from the environment, safely.

    Identical in behavior to bgwork._window_s and tailhealth._window_s
    (both predate this module and both must keep matching it — there is a
    test asserting the two agree). Beyond env_float's contract, this also
    rejects a non-positive or non-finite result:

      * NON-POSITIVE. `0` or a negative parses fine and then silently
        disables whatever "recently" check the window drives (typically
        `since <= window_s`, which becomes never-true), which is the exact
        regression these windows exist to catch, arriving quietly through
        the one config value whose whole job is to prevent it.
      * NON-FINITE. `inf` parses, satisfies `v > 0`, and then pins the
        "recently" flag True from the first qualifying event until the
        process restarts — an always-on warning is a warning nobody reads,
        the same failure shape from the other direction. `math.isfinite`
        catches it; `nan` already fails `v > 0` on its own, so this just
        makes that rejection explicit instead of incidental.
    """
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) and v > 0 else default
