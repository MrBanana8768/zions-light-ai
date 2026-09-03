"""
CPU-only tests for compactor.tailhealth (v3.1.4) — the counter behind the
memory_tail block in /health/full.

What this pins, and the mutation each case kills:

    consecutive_skips not reset by a store       -> [4]
    skipped_recently on the CUMULATIVE count     -> [5] the stale skip
    trimmed totals not accumulated               -> [3]
    note() silent on a skip                      -> [2]
    a store counted as a skip (or vice versa)    -> [1]/[2]

Run: python test_tailhealth.py
"""

import os
import sys
import time

import tailhealth


def assert_eq(a, b, label):
    if a != b:
        print(f"FAIL {label}: expected {b!r}, got {a!r}")
        sys.exit(1)
    print(f"  ok   {label}")


def assert_true(c, label):
    if not c:
        print(f"FAIL {label}")
        sys.exit(1)
    print(f"  ok   {label}")


print("[0] a fresh process has counted nothing")
tailhealth._reset_for_tests()
s = tailhealth.snapshot()
assert_eq(s["stored"], 0, "nothing stored")
assert_eq(s["skipped"], 0, "nothing skipped")
assert_eq(s["consecutive_skips"], 0, "no streak")
assert_eq(s["seconds_since_last_skip"], None, "None, not 0 — it never happened")
assert_eq(s["skipped_recently"], False, "not degraded")
assert_eq(s["last_skip_outcome"], None, "no last skip")
assert_eq(s["trim_retention"], None, "None, not 0.0 — nothing has been measured")
assert_eq(set(s["outcomes"]), set(tailhealth.OUTCOMES), "every outcome label is present at 0")
assert_true(all(v == 0 for v in s["outcomes"].values()), "...at 0")
assert_eq(s["skip_window_s"], tailhealth.SKIP_DEGRADE_WINDOW_S, "window echoed for the reader")

print()
print("[1] a verbatim store")
tailhealth._reset_for_tests()
r = tailhealth.note(tailhealth.STORED, raw_chars=1200, kept_chars=1200)
assert_eq(r, None, "note() has nothing to add to the log line for a store")
s = tailhealth.snapshot()
assert_eq(s["stored"], 1, "stored 1")
assert_eq(s["skipped"], 0, "skipped 0")
assert_eq(s["outcomes"]["stored"], 1, "outcome counted under its label")
assert_eq((s["raw_chars"], s["kept_chars"]), (1200, 1200), "raw and kept both counted")
assert_eq((s["trimmed_raw_chars"], s["trimmed_kept_chars"]), (0, 0),
          "a verbatim store does not touch the trim totals")
assert_eq(s["trim_retention"], None, "...so retention is still unmeasured")
assert_eq(s["skipped_recently"], False, "a store is not a skip")

print()
print("[2] a skip")
tailhealth._reset_for_tests()
r = tailhealth.note(tailhealth.SKIPPED_NO_BOUNDARY, raw_chars=812, kept_chars=0)
assert_true(r is not None and "1 consecutive" in r,
            "note() returns the streak for the caller's log line")
s = tailhealth.snapshot()
assert_eq(s["skipped"], 1, "skipped 1")
assert_eq(s["stored"], 0, "stored 0")
assert_eq(s["consecutive_skips"], 1, "streak 1")
assert_eq(s["last_skip_outcome"], "skipped_no_boundary", "the machine label, not a reason string")
assert_eq((s["raw_chars"], s["kept_chars"]), (812, 0), "raw counted; kept is 0 for a skip")
assert_true(s["seconds_since_last_skip"] is not None and s["seconds_since_last_skip"] < 5.0,
            "the age clock started")
assert_eq(s["skipped_recently"], True, "a skip just now IS recent")

print()
print("[3] a trimmed store measures retention")
tailhealth._reset_for_tests()
tailhealth.note(tailhealth.STORED_TRIMMED, raw_chars=1000, kept_chars=800)
tailhealth.note(tailhealth.STORED_TRIMMED, raw_chars=3000, kept_chars=200)
tailhealth.note(tailhealth.STORED, raw_chars=500, kept_chars=500)
s = tailhealth.snapshot()
assert_eq((s["trimmed_raw_chars"], s["trimmed_kept_chars"]), (4000, 1000),
          "trim totals cover the trimmed stores only")
assert_eq(s["trim_retention"], 0.25, "retention = trimmed kept / trimmed raw")
assert_eq((s["raw_chars"], s["kept_chars"]), (4500, 1500), "overall totals include everything")
assert_eq(s["stored"], 3, "all three stored")
assert_eq(s["outcomes"]["stored_trimmed"], 2, "two under the trimmed label")

print()
print("[4] the streak resets on a store; the cumulative count does not")
tailhealth._reset_for_tests()
for _ in range(3):
    r = tailhealth.note(tailhealth.SKIPPED_TOO_SHORT, raw_chars=100, kept_chars=0)
assert_true("3 consecutive" in (r or ""), "third skip reports a streak of 3")
assert_eq(tailhealth.snapshot()["consecutive_skips"], 3, "streak 3")
tailhealth.note(tailhealth.STORED, raw_chars=900, kept_chars=900)
s = tailhealth.snapshot()
assert_eq(s["consecutive_skips"], 0, "one store resets the streak")
assert_eq(s["skipped"], 3, "...but the cumulative skip count is the record and stays")
assert_eq(s["last_skip_outcome"], "skipped_too_short", "...and the last skip is still named")

print()
print("[5] skipped_recently is windowed — a stale skip does not pin 'degraded'")
tailhealth._reset_for_tests()
tailhealth.note(tailhealth.SKIPPED_HOLED, raw_chars=100, kept_chars=0)
during = tailhealth.snapshot(window_s=0.05)
time.sleep(0.12)  # past the 0.05 s window
after = tailhealth.snapshot(window_s=0.05)
assert_eq(during["skipped_recently"], True, "recent while inside the window")
assert_eq(after["skipped_recently"], False, "clears itself once the window rolls over")
assert_eq(after["skipped"], 1, "the cumulative count is NOT reset")
assert_eq(after["consecutive_skips"], 1, "nor the streak — only recency ages out")
assert_eq(after["skip_window_s"], 0.05, "the window in force is echoed")

print()
print("[6] the snapshot is a copy")
tailhealth._reset_for_tests()
tailhealth.note(tailhealth.STORED, raw_chars=10, kept_chars=10)
s = tailhealth.snapshot()
s["outcomes"]["stored"] = 999
s["stored"] = 999
assert_eq(tailhealth.snapshot()["outcomes"]["stored"], 1, "mutating the snapshot changes nothing")

print()
print("[7] an unknown label is counted, not raised — this runs in a request's finally")
tailhealth._reset_for_tests()
r = tailhealth.note("skipped_something_new", raw_chars=5, kept_chars=0)
s = tailhealth.snapshot()
assert_eq(s["outcomes"].get("skipped_something_new"), 1, "counted under its own name")
assert_eq(s["skipped"], 1, "an unknown outcome is a skip, never a silent store")
# ...AND LOSSY BY DEFAULT. LOSSY_SKIP_OUTCOMES is computed by exclusion
# rather than by naming the five lossy labels, precisely so that a label
# this module has never heard of degrades health instead of being waved
# through. An outcome added to main.py and forgotten here is the shape of
# every silent-skip defect this module was written to end, so the default
# has to be the loud one. Without this the "by exclusion" construction can
# be replaced with an `and outcome in OUTCOMES` guard and no test notices.
assert_true(s["skipped_recently"],
            "an unknown outcome is treated as LOSSY, so health still degrades")
assert_true("skipped_something_new" not in tailhealth.OUTCOMES,
            "fixture: the label really is one the module does not know")
assert_true(r is not None, "and reported as one")

print()
print("[8] negative or odd char counts cannot drive a total backwards")
tailhealth._reset_for_tests()
tailhealth.note(tailhealth.STORED, raw_chars=-50, kept_chars=-50)
s = tailhealth.snapshot()
assert_eq((s["raw_chars"], s["kept_chars"]), (0, 0), "clamped at 0")

print()
print("[9] the degrade window survives a bad env value")
# A typo here used to be a BOOT failure: tailhealth is imported at main.py
# module scope, and `float(os.environ.get(...) or 300)` rescues only the
# empty string, so `30O` in runpod.env raised ValueError at import and the
# compactor never started. A non-positive value was worse than a crash - it
# parsed, and then made `skipped_recently` (`since <= window`) permanently
# False, so /health/full reported ok while the memory tail was skipping.
# That is this module's entire reason for existing, switched off by one
# config line nobody would look at twice.
_saved = os.environ.get("COMPACTOR_TAIL_SKIP_DEGRADE_WINDOW_S")
try:
    # 'inf' is the dangerous one: it PARSES, satisfies `v > 0`, and then pins
    # skipped_recently True from the first skip until restart — a warning that
    # is always on is a warning nobody reads, which is the failure the window
    # exists to prevent, arriving through the one knob meant to prevent it.
    for _bad in ("abc", "", "   ", "30O", "0", "-5", "inf", "Infinity",
                 "1e400", "nan", None):
        if _bad is None:
            os.environ.pop("COMPACTOR_TAIL_SKIP_DEGRADE_WINDOW_S", None)
        else:
            os.environ["COMPACTOR_TAIL_SKIP_DEGRADE_WINDOW_S"] = _bad
        try:
            _v = tailhealth._window_s("COMPACTOR_TAIL_SKIP_DEGRADE_WINDOW_S", 300.0)
            assert_eq(_v, 300.0, f"{_bad!r} -> the 300s default, no raise")
        except SystemExit:
            raise
        except Exception as e:
            assert_true(False, f"{_bad!r} -> raised {type(e).__name__}")
    # A real value must still be honoured, or the checks above could pass
    # against a function that ignores the environment entirely.
    os.environ["COMPACTOR_TAIL_SKIP_DEGRADE_WINDOW_S"] = "45.5"
    assert_eq(tailhealth._window_s("COMPACTOR_TAIL_SKIP_DEGRADE_WINDOW_S", 300.0),
              45.5, "a valid override is still honoured")
    # snapshot's explicit window_s must go through the SAME guard. It used to
    # bypass it, so window_s=0 made `since <= 0` true for a skip that had just
    # happened: 0 was MORE alarming than the default while -5 was less.
    tailhealth._reset_for_tests()
    tailhealth.note(tailhealth.SKIPPED_EMPTY, raw_chars=10, kept_chars=0)
    for _w in (0, -5, float("inf"), float("nan"), "abc"):
        assert_eq(tailhealth.snapshot(window_s=_w)["skip_window_s"], 300.0,
                  f"snapshot(window_s={_w!r}) falls back to the default")
    assert_eq(tailhealth.snapshot(window_s=45.5)["skip_window_s"], 45.5,
              "a valid explicit window is still honoured")
finally:
    os.environ.pop("COMPACTOR_TAIL_SKIP_DEGRADE_WINDOW_S", None)
    if _saved is not None:
        os.environ["COMPACTOR_TAIL_SKIP_DEGRADE_WINDOW_S"] = _saved

print()
print("[10] SKIPPED_EMPTY does not make skipped_recently true (v3.1.7, R27)")
# She hit Stop before the first token: nothing to memorize, so this is the
# one skip label that carries no loss. Keying the degrade decision off ANY
# skip pinned /health/full degraded on this alone — the release's own figure
# is 51 of 63 skips in one window were manual stops.
tailhealth._reset_for_tests()
tailhealth.note(tailhealth.SKIPPED_EMPTY, raw_chars=10, kept_chars=0)
s = tailhealth.snapshot()
assert_eq(s["skipped"], 1, "still counted — the record is intact")
assert_eq(s["last_skip_outcome"], "skipped_empty", "and named")
assert_true(s["seconds_since_last_skip"] is not None, "the general clock started")
assert_eq(s["skipped_recently"], False, "but an empty skip alone is not 'recently lossy'")
assert_eq(s["seconds_since_last_lossy_skip"], None, "no lossy skip has ever happened")

print()
print("[11] a lossy skip alongside empty skips still degrades")
tailhealth._reset_for_tests()
tailhealth.note(tailhealth.SKIPPED_EMPTY, raw_chars=0, kept_chars=0)
tailhealth.note(tailhealth.SKIPPED_NO_BOUNDARY, raw_chars=50, kept_chars=0)
tailhealth.note(tailhealth.SKIPPED_EMPTY, raw_chars=0, kept_chars=0)
s = tailhealth.snapshot()
assert_eq(s["skipped"], 3, "all three counted")
assert_eq(s["last_skip_outcome"], "skipped_empty", "the most recent skip of any kind")
assert_eq(s["skipped_recently"], True,
          "a real loss happened in the window, even though the LAST skip did not")
assert_true(s["seconds_since_last_lossy_skip"] is not None
            and s["seconds_since_last_lossy_skip"] < 5.0,
            "the lossy clock reflects the middle (lossy) skip")

print()
print("[12] LOSSY_SKIP_OUTCOMES excludes only SKIPPED_EMPTY")
assert_eq(tailhealth.SKIPPED_EMPTY not in tailhealth.LOSSY_SKIP_OUTCOMES, True,
           "the one label with no loss")
for _o in tailhealth.OUTCOMES:
    if _o in tailhealth.STORING_OUTCOMES or _o == tailhealth.SKIPPED_EMPTY:
        continue
    assert_true(_o in tailhealth.LOSSY_SKIP_OUTCOMES, f"{_o} is lossy")

print()
print("[13] note() does not raise on a non-numeric char count (v3.1.7, R28)")
# `int(None)` and `int('bad')` both raise TypeError/ValueError. note() runs in
# the request path's `finally`, where a bookkeeping error must not become a
# second failure — the module docstring's own claim, previously honoured only
# for the outcome label.
tailhealth._reset_for_tests()
for _raw, _kept in ((None, 0), ("not-a-number", 0), (5, None), (5, "also-bad")):
    r = tailhealth.note(tailhealth.SKIPPED_TOO_SHORT, raw_chars=_raw, kept_chars=_kept)
    assert_true(r is not None, f"note({_raw!r}, {_kept!r}) did not raise and returned a streak")
s = tailhealth.snapshot()
assert_eq(s["skipped"], 4, "all four calls counted despite the bad inputs")
assert_eq(sum(s["outcomes"].values()), s["stored"] + s["skipped"],
          "outcomes reconcile with stored+skipped even after coercion failures")

print()
print("[14] the outcome tally and stored/skipped never disagree (v3.1.7, R28)")
# Belt and braces on the reconciliation test_saturation.py relies on: mix
# clean and dirty calls, storing and skipping, and check the invariant holds
# throughout, not just at the end.
tailhealth._reset_for_tests()
_calls = [
    (tailhealth.STORED, 100, 100),
    (tailhealth.SKIPPED_EMPTY, None, 0),
    (tailhealth.SKIPPED_NO_BOUNDARY, "bad", 0),
    (tailhealth.STORED_TRIMMED, 500, 200),
    (tailhealth.SKIPPED_TOO_SHORT, 30, None),
]
for _outcome, _raw, _kept in _calls:
    tailhealth.note(_outcome, raw_chars=_raw, kept_chars=_kept)
    _s = tailhealth.snapshot()
    assert_eq(sum(_s["outcomes"].values()), _s["stored"] + _s["skipped"],
              f"reconciled after {_outcome}")

print()
print("All tailhealth tests passed.")
