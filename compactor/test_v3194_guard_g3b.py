"""v3.1.9.4, lane v3194-guard, G3b (hostile pass #14's "not demonstrated"
list): the reuse window check's `_standin_reserve` (main.py, `compact_
if_needed`) adds a fixed `+128` to absorb the current-time line's own
reserve — a line decided LATER, in `chat_completions`, after this
function returns, so the exact figure is not available here to subtract
(see that `+128`'s own comment). p14 measured the line's real reserve at
"~97-105 tokens" and left "whether the slack can go negative" as not
demonstrated: it depends on the LONGEST line `_format_time_line` can ever
produce, across the longest weekday/month names AND whatever timezone
abbreviation a real IANA zone can hand back, which varies by the exact
calendar day (DST transitions change a zone's `tzname()` abbreviation
mid-year).

THIS FILE DEMONSTRATES (not a fix — none was needed): a brute-force scan
of every IANA zone this image ships (`zoneinfo.available_timezones()`,
498 zones on the reference image) at every day of a leap year (to catch
every DST abbreviation change) finds a worst case of 99 tokens
(`Antarctica/Macquarie`, 2024-09-11, "Wednesday" + "September" — the two
longest names in their tables — landing on AEST, "[Current date and
time: Wednesday, September 11, 2024, 10:34 PM AEST (UTC+10:00)]"), 29
tokens under the fixed 128 the window check absorbs. The slack does NOT
go negative; the check is correct as shipped. Runs in a few seconds
(498 zones x 366 days), cheap enough to pin as a permanent regression: if
a future tzdata update, a table edit (`_WEEKDAYS`/`_MONTHS`), or a change
to `_TIME_LINE_JOIN_SLACK` ever pushes the worst case past what the fixed
absorption assumes, this test is the thing that notices.

    python test_v3194_guard_g3b.py
"""

import datetime as dt
import os
import sys
import tempfile
from zoneinfo import ZoneInfo, available_timezones

os.environ.setdefault("MODEL_REPO", "test-model")
os.environ.setdefault("VLLM_URL", "http://stub:8000")
os.environ["COMPACTOR_RAG_ENABLED"] = "false"
os.environ["COMPACTOR_STORAGE_ROOT"] = tempfile.mkdtemp(prefix="g3bguard-")

import memory  # noqa: E402

memory.ensure_storage_layout()

import main  # noqa: E402

FAILED: list[str] = []


def check(cond, label):
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"FAIL {label}")
        FAILED.append(label)


# The fixed absorption compact_if_needed's window check relies on (main.py,
# `_standin_reserve = _standin_rendered + _fresh_reserve + 128`) — a bare
# literal there, not a named constant, so pinned here by its own value with
# this comment as the cross-reference. If that literal ever moves, this
# test's own margin check (not the worst-case measurement) is what would
# need updating to match — the worst case itself is a fact about
# _format_time_line and this image's tzdata, independent of it.
_FIXED_ABSORPTION = 128

print("\n[G3b] the fixed +128 the reuse window check absorbs for the "
      "time-line reserve covers every IANA zone's worst case")

zones = sorted(available_timezones())
if len(zones) < 50:
    # Windows without the `tzdata` pip package (the interpreter this repo's
    # own rules call "for iteration") has no zone database at all —
    # zoneinfo.available_timezones() returns empty, and this test answers a
    # question about worst-case ZONE ABBREVIATIONS that an empty database
    # cannot ask. Same skip convention test_disconnect_uvloop.py uses for a
    # comparable "wrong platform for this question" case. Final results are
    # Linux/Docker, where the shipped image's real tzdata is present.
    print(f"  SKIP: only {len(zones)} zone(s) available — no real tzdata on "
          f"this interpreter. Run in the container: docker compose -f "
          f"docker-compose.tests.yml run --rm unit-tests")
    sys.exit(3)

_YEAR = 2024  # a leap year, so Feb 29 is covered too
_worst_reserve = -1
_worst_line = None
_worst_zone = None
_worst_date = None

_start = dt.date(_YEAR, 1, 1)
for _zname in zones:
    try:
        _z = ZoneInfo(_zname)
    except Exception:
        continue
    _d = _start
    while _d.year == _YEAR:
        _now = dt.datetime(_d.year, _d.month, _d.day, 12, 34, tzinfo=dt.timezone.utc)
        _line = main._format_time_line(_now, _z)
        _r = main._time_line_token_reserve(_line)
        if _r > _worst_reserve:
            _worst_reserve, _worst_line, _worst_zone, _worst_date = _r, _line, _zname, _now
        _d += dt.timedelta(days=1)

print(f"  worst case: reserve={_worst_reserve} zone={_worst_zone!r} "
      f"date={_worst_date} line={_worst_line!r}")

check(
    _worst_reserve <= _FIXED_ABSORPTION,
    f"*** G3b: the worst real-world time-line reserve ({_worst_reserve}) "
    f"stays under the window check's fixed absorption ({_FIXED_ABSORPTION}) "
    f"— {_FIXED_ABSORPTION - _worst_reserve} token(s) of margin to spare "
    f"(zone={_worst_zone!r}, line={_worst_line!r})",
)
check(
    _worst_reserve < 110,
    f"the worst case ({_worst_reserve}) is close to p14's own "
    f"'~97-105 tokens' estimate, not a surprise far outside it",
)

# CONTROL: the join slack alone (no zone/weekday/month variation) is a real,
# positive contributor, not zero — i.e. this measurement is not accidentally
# ignoring _TIME_LINE_JOIN_SLACK.
_ctrl_line = main._format_time_line(
    dt.datetime(2024, 1, 1, 0, 0, tzinfo=dt.timezone.utc), dt.timezone.utc
)
_ctrl_reserve = main._time_line_token_reserve(_ctrl_line)
check(
    _ctrl_reserve < _worst_reserve,
    f"CONTROL: a plain UTC line on an ordinary day ({_ctrl_reserve}) reserves "
    f"less than the worst case found above ({_worst_reserve}) — the scan is "
    f"actually finding a WORSE case, not just measuring the floor",
)


if FAILED:
    print(f"\n{len(FAILED)} check(s) FAILED:")
    for f in FAILED:
        print(f"  - {f}")
    sys.exit(1)
print("\nall checks passed")
