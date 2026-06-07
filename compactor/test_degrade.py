"""
CPU-only Tier-1 tests for compactor.degrade (V2.3 Theme 2).

Disk-pressure write-gating: when free space drops below the threshold,
new-memory writes are paused (guard() → False) but the check fails OPEN if
free space can't be read. Caching avoids statvfs storms.

Run: python test_degrade.py
"""

import os
import sys

os.environ["COMPACTOR_MIN_FREE_MB_WRITES"] = "200"
os.environ["COMPACTOR_DEGRADE_CHECK_TTL_S"] = "0"  # disable TTL for deterministic tests

import degrade  # noqa: E402


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


def _set_free(mb):
    """Force _free_mb to report a fixed value + clear the cache."""
    degrade._free_mb = lambda path: mb
    degrade._reset_cache_for_tests()


# ---------------------------------------------------------------------------

def test_allows_writes_with_ample_space():
    print("\n[test] writes_allowed: plenty of free space → allowed")
    _set_free(10_000)
    allowed, free = degrade.writes_allowed()
    assert_eq(allowed, True, "allowed when 10 GB free")
    assert_eq(free, 10_000, "free reported")


def test_blocks_writes_under_threshold():
    print("\n[test] writes_allowed: below threshold → blocked")
    _set_free(50)  # < 200
    allowed, free = degrade.writes_allowed()
    assert_eq(allowed, False, "blocked when 50 MB free")


def test_exactly_at_threshold_allows():
    print("\n[test] writes_allowed: exactly at threshold → allowed (>=)")
    _set_free(200)
    allowed, _ = degrade.writes_allowed()
    assert_eq(allowed, True, "200 == threshold is allowed")


def test_fails_open_when_unreadable():
    print("\n[test] writes_allowed: unreadable free space → fails OPEN (inf)")
    _set_free(float("inf"))
    allowed, free = degrade.writes_allowed()
    assert_eq(allowed, True, "inf free → allowed (fail open)")


def test_guard_matches_writes_allowed():
    print("\n[test] guard() mirrors writes_allowed()")
    _set_free(10_000)
    assert_eq(degrade.guard("op"), True, "guard True when allowed")
    _set_free(10)
    assert_eq(degrade.guard("op"), False, "guard False when blocked")


def test_write_state_shape_allowed():
    print("\n[test] write_state: allowed shape")
    _set_free(5_000)
    st = degrade.write_state()
    assert_eq(st["new_memory_writes"], "allowed", "state allowed")
    assert_eq(st["min_free_mb"], 200, "threshold echoed")
    assert_eq(st["free_mb"], 5000.0, "free reported")


def test_write_state_shape_paused():
    print("\n[test] write_state: paused shape under pressure")
    _set_free(10)
    st = degrade.write_state()
    assert_eq(st["new_memory_writes"], "paused", "state paused")


def test_write_state_free_mb_none_when_infinite():
    print("\n[test] write_state: free_mb is None when undeterminable")
    _set_free(float("inf"))
    st = degrade.write_state()
    assert_eq(st["free_mb"], None, "inf → None in report")
    assert_eq(st["new_memory_writes"], "allowed", "still allowed")


def test_cache_ttl_avoids_recheck():
    print("\n[test] caching: within TTL, a changed disk reading is not re-read")
    # Enable a long TTL for this test only
    os.environ["COMPACTOR_DEGRADE_CHECK_TTL_S"] = "60"
    import importlib
    importlib.reload(degrade)
    calls = {"n": 0}

    def counting_free(path):
        calls["n"] += 1
        return 10_000
    degrade._free_mb = counting_free
    degrade._reset_cache_for_tests()

    degrade.writes_allowed()
    degrade.writes_allowed()
    degrade.writes_allowed()
    assert_eq(calls["n"], 1, "only one statvfs within TTL window")
    # Restore TTL=0 + reload so later imports elsewhere aren't affected
    os.environ["COMPACTOR_DEGRADE_CHECK_TTL_S"] = "0"
    importlib.reload(degrade)


def test_transition_logging_does_not_crash():
    print("\n[test] state transitions (block→clear→block) don't error")
    _set_free(10)      # block (logs warning)
    degrade.writes_allowed()
    _set_free(10_000)  # clear (logs info)
    degrade.writes_allowed()
    _set_free(10)      # block again
    degrade.writes_allowed()
    print("  ok   transitions handled")


def _all():
    return [
        test_allows_writes_with_ample_space,
        test_blocks_writes_under_threshold,
        test_exactly_at_threshold_allows,
        test_fails_open_when_unreadable,
        test_guard_matches_writes_allowed,
        test_write_state_shape_allowed,
        test_write_state_shape_paused,
        test_write_state_free_mb_none_when_infinite,
        test_cache_ttl_avoids_recheck,
        test_transition_logging_does_not_crash,
    ]


if __name__ == "__main__":
    for t in _all():
        t()
    print("\nAll degrade smoke tests passed.")
