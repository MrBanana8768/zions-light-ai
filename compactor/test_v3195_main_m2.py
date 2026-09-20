"""
v3195-main M2 -- the new degeneracy rules' fire counts reach /health/full
(SP\\V3195_MAIN_BRIEF.md M2, SP\\p18a-findings.md F6: "three of textclean's
four public functions have zero production callers... the module that
measures decoration measures nothing in production").

Also covers the M3 side effect on the SAME endpoint: count_tokens' new
chat-template-fallback fields pass through health.tokenizer_state() without
any health.py contract change (main.tokenizer_state() is spread with **st;
see that function's own docstring).

    python test_v3195_main_m2.py
"""

import asyncio
import os
import sys

os.environ.setdefault("MODEL_REPO", "")

import health  # noqa: E402
import main  # noqa: E402

_FAILED = False


def check(cond, label):
    global _FAILED
    status = "ok  " if cond else "FAIL"
    print(f"  {status} {label}")
    if not cond:
        _FAILED = True


def full():
    return asyncio.run(health.gather_health_full("http://127.0.0.1:9", 4096))


print("[1] checks.degeneracy_rules exists and carries both new rule names")
body = full()
dr = body["checks"].get("degeneracy_rules") or {}
check(bool(dr), "checks.degeneracy_rules is present")
check(dr.get("available") is True, f"reported available ({dr})")
check("symbol_wall" in dr and "word_novelty" in dr,
      f"both new rule names are present  [{dr}]")
check(isinstance(dr.get("symbol_wall"), int) and isinstance(dr.get("word_novelty"), int),
      "both counts are plain integers")

print()
print("[2] a nonzero count never moves `status` (visibility only, same "
      "doctrine as tokenizer/reuse/budget_margin/truncated_summaries)")
# Force at least one fire and re-read.
wall = "".join(chr(0x2200 + i) for i in range(45)) * 6  # 45 distinct, dense
main._reply_degenerate_verdict_uncached("prefix text here. " * 40 + wall)
body2 = full()
dr2 = body2["checks"].get("degeneracy_rules") or {}
check(dr2.get("symbol_wall", 0) >= 1,
      f"symbol_wall now nonzero ({dr2.get('symbol_wall')})")
check(body2["status"] in ("ok", "degraded"),
      f"status is still a normal value ({body2['status']!r}), not forced by this")
check(
    not any("degeneracy" in r.lower() or "symbol_wall" in r.lower()
            or "word_novelty" in r.lower() for r in body2["status_reasons"]),
    "a nonzero degeneracy-rule count is not itself a status_reason "
    "(visibility only, matching tokenizer/reuse/budget_margin doctrine)"
)

print()
print("[3] checks.tokenizer carries the new M3 fields via the existing "
      "**st spread -- no health.py contract change needed for these three")
tok = body["checks"].get("tokenizer")
check(tok is not None and tok.get("available") is True,
      f"checks.tokenizer is present and available  [{tok}]")
for key in ("chat_template_fallback_streak", "chat_template_fallback_total",
            "chat_template_degraded_since"):
    check(key in tok, f"checks.tokenizer carries {key!r}")

print()
print("[4] main not loaded in this process -> a clean 'unavailable', not a "
      "crash (same contract as the sibling _*_state functions)")
import sys as _sys
_saved = _sys.modules.pop("main")
try:
    st = health._degeneracy_rule_state()
    check(st == {"available": False, "reason": "main is not loaded in this process"},
          f"clean unavailable dict when main is not in sys.modules  [{st}]")
finally:
    _sys.modules["main"] = _saved

print()
if _FAILED:
    print("SOME v3195-main M2 CHECKS FAILED")
    sys.exit(1)
print("All v3195-main M2 checks passed.")
