"""2026-09-25 RCA (her chat): the hard-budget guard over-shed. Production cut
7,869,389 tokens to 12,813 against a 20,672-token limit (38% of the window
left unused), dropping 4,191 old turn(s) and keeping only 4 messages, with
the retained images sitting right before the recent turns.

DIAGNOSIS: `_enforce_hard_budget`'s round loop and `_shed_last_resort` both
stop shedding the INSTANT a real measurement says "this now fits" — the
P12-5 branch's own floor (`_aligned_recent_tail` / `KEEP_RECENT_TURNS`,
default 4) is a MINIMUM the guard must never shed below, not a target it
fills the window to. Once cutting down to (or, on a bad per-item estimate,
past) that floor already fits, nothing downstream ever asks "would some of
what was just shed still fit?" — the shed loop only ever proves "shed
enough", never "shed too much". On a payload many multiples of the window,
the fixed-size floor is nearly always far smaller than the limit allows, so
most of the window goes unused every time this path is taken.

THE FIX: `_fill_back_dropped_turns`, called once after `_shed_last_resort`.
It diffs the final `msgs` against the ORIGINAL input by identity (every
shedding site removes turns from a contiguous prefix, oldest-first, so the
missing set is always a contiguous prefix in original order — no per-site
bookkeeping needed), groups adjacent (user, assistant) pairs so an add can
never leave the array mid-pair, and adds groups back newest-first while
each one still measures (via the SAME `_measure` closure, vLLM's own
/tokenize) under the limit. Scoped to TURNS only (never a dropped memory
block or the stand-in — see its own docstring), and only when the last
measurement was a REAL /tokenize count, never a local-estimate fallback.

This file reproduces the incident's SHAPE — thousands of old turns, 2
images near the tail, one large injected-memory block, a system prompt —
against a small limit, with local/exact token counts monkeypatched to the
CORRECTED calibration (hostile review round 2): the served tokenizer is
mistral_common-backed, not the transformers-converted one this codebase's
2026-08-28 comment measured. It never reads low, only high on a
SINGLE-message call, by a fixed per-call framing cost worst on short
content (the hostile reviewer's own bucketed scale figures from
`floor.py`); a whole-list call is accurate. An image's flat local estimate
still reads roughly half its true cost — that figure is about
`IMAGE_TOKEN_ESTIMATE` vs a real vision tile, unrelated to the text
tokenizer swap, and unchanged by this round. No real tokenizer, no her
data — every message here is synthetic.

    python test_rca_20260925_guard_fillback.py
"""

import os
import sys
import tempfile

os.environ.pop("MODEL_REPO", None)
os.environ["COMPACTOR_STORAGE_ROOT"] = tempfile.mkdtemp(prefix="rca0925-fillback-")
os.environ["COMPACTOR_RAG_ENABLED"] = "false"
os.environ["MAX_MODEL_LEN"] = "32768"

import main  # noqa: E402

FAILED: list[str] = []


def check(cond, label, detail=""):
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"FAIL {label}{': ' + detail if detail else ''}")
        FAILED.append(label)


# =============================================================================
# CORRECTED calibration (hostile review round 2): the served tokenizer is
# mistral_common-backed, not the transformers-converted one this file's
# first cut modeled. It never reads low — a WHOLE-LIST count_tokens(msgs)
# call is accurate (prod "token scale 1.00x" on ordinary content); a
# SINGLE-message count_tokens([m]) call reads HIGH by a fixed per-call
# framing cost, worst on short content — the hostile reviewer's own
# bucketed scale figures (floor.py): 0.63 for content this short, up to
# 0.98 for longer content. Images are unaffected by that recalibration (a
# separate, still-open concern about the flat IMAGE_TOKEN_ESTIMATE vs a
# real vision tile's cost, not the text tokenizer swap) and keep their own
# figures.
# =============================================================================
TRUE_IMG = 8192
LOCAL_IMG = 4096


def _scale_for_true(true_n):
    if true_n <= 6:
        return 0.63
    if true_n <= 15:
        return 0.77
    if true_n <= 40:
        return 0.89
    return 0.98


def _true_one(m):
    content = m.get("content")
    if isinstance(content, list):
        return TRUE_IMG
    text = content if isinstance(content, str) else ""
    return max(1, len(text) // 4)


def _local_single(m):
    content = m.get("content")
    if isinstance(content, list):
        return LOCAL_IMG
    true_n = _true_one(m)
    return max(true_n, int(round(true_n / _scale_for_true(true_n))))


def fake_count_tokens(messages):
    """The local tokenizer: accurate on a whole list, high on a singleton."""
    if len(messages) == 1:
        return _local_single(messages[0])
    return sum(_true_one(m) for m in messages)


def fake_count_tokens_exact(messages, add_generation_prompt=None):
    if not messages:
        return 0
    return sum(_true_one(m) for m in messages)


_real_count_tokens = main.count_tokens
_real_count_tokens_exact = main.count_tokens_exact
main.count_tokens = fake_count_tokens
main.count_tokens_exact = fake_count_tokens_exact


def _build_incident_shaped_payload(n_pairs=2095):
    system_prompt = {"role": "system", "content": "You are a helpful assistant. " * 20}
    memory_block = {
        "role": "system",
        "content": "[INJECTED MEMORY]\n" + ("fact: something happened. " * 200),
    }
    assistant_filler = "The walls glowed with intricate patterns. " * 90
    user_filler = "Tell me more about that please." * 3
    old_turns = []
    for i in range(n_pairs):
        old_turns.append({"role": "user", "content": f"[{i}] {user_filler}"})
        old_turns.append({"role": "assistant", "content": f"[{i}] {assistant_filler}"})
    # 2 images near the tail of the old span, before the recent window —
    # matches the incident ("the retained images sit right before the
    # recent turns").
    old_turns += [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:x"}}]},
        {"role": "assistant", "content": "Nice picture."},
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:x"}}]},
        {"role": "assistant", "content": "Another one."},
    ]
    recent = [
        {"role": "user", "content": "recent question 1"},
        {"role": "assistant", "content": "recent answer 1"},
        {"role": "user", "content": "recent question 2"},
        {"role": "assistant", "content": "recent answer 2"},
    ]
    return [system_prompt, memory_block] + old_turns + recent


# =============================================================================
# [0] WIRING: _enforce_hard_budget actually calls _fill_back_dropped_turns.
#     A mutant that removes the call site (but leaves the function itself
#     untouched) still passes a purely-behavioral check on a fixture where
#     other mechanisms (the pre-cut's own smaller remainder, the P12-5
#     branch's own exact-verified stop) happen to land close to the limit
#     anyway — measured: on THIS file's own [1] fixture, that happened. So
#     the wiring itself is checked directly, by spying on the module-level
#     name _enforce_hard_budget actually calls.
# =============================================================================
print("\n[0] _enforce_hard_budget actually calls _fill_back_dropped_turns")

_real_fill_back = main._fill_back_dropped_turns
_fill_back_calls = {"n": 0}


def _spy_fill_back(*a, **kw):
    _fill_back_calls["n"] += 1
    return _real_fill_back(*a, **kw)


main._fill_back_dropped_turns = _spy_fill_back
_wiring_payload = _build_incident_shaped_payload(n_pairs=50)
main._enforce_hard_budget(list(_wiring_payload), limit=5000, protect_system=1, report={})
check(
    _fill_back_calls["n"] >= 1,
    "*** WIRING: _enforce_hard_budget calls _fill_back_dropped_turns at "
    "least once per invocation",
    f"calls observed: {_fill_back_calls['n']}",
)
main._fill_back_dropped_turns = _real_fill_back


# =============================================================================
# [1] THE REPRODUCTION: the guard must fit AND must fill up to the limit,
#     not stop at whatever the fixed-size floor happens to hold.
# =============================================================================
print("\n[1] the incident-shaped synthetic payload: the guard fills up to "
      "the limit instead of stopping at the floor")

LIMIT = 20672
payload = _build_incident_shaped_payload()
true_total_before = fake_count_tokens_exact(payload)
check(
    true_total_before > 90 * LIMIT,
    "fixture: the payload is many multiples of the limit, like the "
    "incident (7.8M tokens against a 20,672-token limit)",
    f"true total {true_total_before} vs limit {LIMIT}",
)

report: dict = {}
result = main._enforce_hard_budget(list(payload), limit=LIMIT, protect_system=1, report=report)
final_true = fake_count_tokens_exact(result)
kept_turns = [m for m in result if m.get("role") != "system"]

check(report.get("fits") is True, "the guard fits the payload", f"{report}")
check(final_true <= LIMIT, "and never exceeds the VERIFIED limit",
      f"{final_true} vs {LIMIT}")

# X = 10%: this fixture's own granularity. Each restorable (user, assistant)
# exchange costs ~1,014 TRUE tokens here (user ~99//4=24, assistant
# ~3960//4=990) — about 4.9% of a 20,672-token limit. Once fewer than one
# full exchange of headroom remains, the guard cannot fill any further
# without a partial-message trim (which this fix deliberately does not do —
# see _fill_back_dropped_turns's own docstring: it restores whole exchanges
# only). 10% covers roughly two exchanges' width plus the
# _FILLBACK_MEASURE_CAP (8) bound on how many are even tried.
MARGIN_X_PCT = 10
used_pct = 100 * final_true / LIMIT
check(
    used_pct >= 100 - MARGIN_X_PCT,
    f"*** THE FIX: the guard fills to within {MARGIN_X_PCT}% of the limit "
    f"({used_pct:.1f}% used) — not 38% empty, like the incident",
    f"final {final_true} / limit {LIMIT} = {used_pct:.1f}% used",
)
check(
    len(kept_turns) > main.KEEP_RECENT_TURNS,
    f"more than the bare {main.KEEP_RECENT_TURNS}-turn floor survives — the "
    f"incident kept exactly 4",
    f"kept {len(kept_turns)} turns",
)

# The two images are still present and still sit ahead of the final plain
# recent turns — the fix never reorders anything, and never touches image
# retention.
image_positions = [i for i, m in enumerate(result) if isinstance(m.get("content"), list)]
plain_recent_positions = [
    i for i, m in enumerate(result)
    if isinstance(m.get("content"), str) and m.get("content", "").startswith("recent")
]
check(
    bool(image_positions),
    "the retained images are still present in the forwarded result",
    f"positions {image_positions}",
)
check(
    (not image_positions) or (not plain_recent_positions)
    or max(image_positions) < min(plain_recent_positions),
    "the images still sit ahead of the final recent turns — never reordered",
    f"images {image_positions}, recent {plain_recent_positions}",
)

# Chronological order is preserved: every kept turn's own index tag is
# non-decreasing left to right (fill-back reconstructs original order
# exactly; it must never shuffle anything).
import re  # noqa: E402

tags = []
for m in kept_turns:
    c = m.get("content")
    if isinstance(c, str):
        mo = re.match(r"\[(\d+)\]", c)
        if mo:
            tags.append(int(mo.group(1)))
check(
    tags == sorted(tags),
    "turns remain in original chronological order — nothing was reordered",
    f"tags: {tags}",
)


# =============================================================================
# [2] CONTROL: when /tokenize is down at the final measurement, fill-back
#     does nothing (never adds content back on an unverified estimate).
# =============================================================================
print("\n[2] CONTROL: fill-back stays off when the last measurement is not "
      "a real /tokenize count")


def _always_local(messages, add_generation_prompt=None):
    return None  # /tokenize refused


main.count_tokens_exact = _always_local
report_down: dict = {}
result_down = main._enforce_hard_budget(
    list(payload), limit=LIMIT, protect_system=1, report=report_down
)
check(
    report_down.get("counted_by", "").startswith("the local tokenizer"),
    "fixture: the final measurement really did fall back to the local counter",
    f"{report_down}",
)
main.count_tokens_exact = fake_count_tokens_exact


# =============================================================================
# [3] direct unit test of _fill_back_dropped_turns: never exceeds the limit,
#     never restores a dropped MEMORY block, respects the measurement cap.
# =============================================================================
print("\n[3] _fill_back_dropped_turns, called directly")

shed_pairs = 500
above_floor_kept = 3
recent_tail = [
    {"role": "user", "content": f"[9000] kept"},
    {"role": "assistant", "content": f"[9001] kept"},
]
original_turns = []
for i in range(shed_pairs):
    original_turns.append({"role": "user", "content": f"[{i}] u " + "z" * 40})
    original_turns.append({"role": "assistant", "content": f"[{i}] a " + "z" * 400})
original = [{"role": "system", "content": "sys"}] + original_turns + recent_tail

# msgs: only the recent tail survived (simulating the over-shed floor case).
msgs = [{"role": "system", "content": "sys"}] + recent_tail
running = sum(_true_one(m) for m in msgs)
calls = {"n": 0}


def _measure(ms):
    calls["n"] += 1
    return sum(_true_one(m) for m in ms), "vLLM's /tokenize"


limit3 = running + 3 * (_true_one(original_turns[-2]) + _true_one(original_turns[-1])) + 50
new_msgs, new_running, new_counter, new_dropped = main._fill_back_dropped_turns(
    original, msgs, running, "vLLM's /tokenize", dropped=shed_pairs * 2,
    limit=limit3, measure=_measure,
)
check(new_running <= limit3, "direct call: never exceeds the limit",
      f"{new_running} vs {limit3}")
check(
    calls["n"] <= main._FILLBACK_MEASURE_CAP,
    f"direct call: bounded at _FILLBACK_MEASURE_CAP ({main._FILLBACK_MEASURE_CAP}) "
    f"measurements",
    f"calls made: {calls['n']}",
)
restored_tags = []
for m in new_msgs:
    c = m.get("content")
    if isinstance(c, str):
        mo = re.match(r"\[(\d+)\] a", c)
        if mo:
            restored_tags.append(int(mo.group(1)))
check(
    bool(restored_tags),
    "direct call: at least one previously-shed exchange was restored",
    f"restored assistant tags: {restored_tags}",
)
check(
    restored_tags == sorted(restored_tags)
    and restored_tags[-1] == shed_pairs - 1,
    "direct call: restored newest-shed-first, ending at the most recent "
    "shed exchange, in original order",
    f"restored: {restored_tags}",
)
check(
    new_dropped == shed_pairs * 2 - 2 * len(restored_tags),
    "direct call: the dropped counter is decremented by exactly what was "
    "restored",
    f"dropped={new_dropped}, restored pairs={len(restored_tags)}",
)


main.count_tokens = _real_count_tokens
main.count_tokens_exact = _real_count_tokens_exact


if FAILED:
    print(f"\n{len(FAILED)} check(s) FAILED:")
    for f in FAILED:
        print(f"  - {f}")
    sys.exit(1)
print("\nall checks passed")
