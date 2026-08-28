"""
CPU-only Tier-1 tests for the context-budget guards.

Regression cover for the 2026-08-13 production failure, where the component
whose job is to keep requests inside the context window overflowed it:

  1. A long conversation packed EVERY older turn into one summarization prompt
     -> that call exceeded MAX_MODEL_LEN -> 400.
  2. Compaction caught the 400 and "degraded" by forwarding the ORIGINAL
     oversized messages.
  3. Memory injection then added 100 facts + retrieved exchanges on top.
  4. The real chat request 400'd: "maximum context length is 32768 tokens...
     your prompt contains at least 32769 input tokens".

So: _chunk_to_budget bounds the summarizer's input, and _enforce_hard_budget is
the final pre-flight that guarantees what we forward can actually be served.

The last two sections drive the endpoint rather than calling the guard
directly, because the fixes they cover are load-bearing on the CALL SITE and
nothing that calls a function can see whether its call site is right.

The final section covers what happens when the guard loses — when the request
goes out anyway and vLLM refuses it. On 2026-08-24 23:49 that was 139.9s of
compaction, a context-length 400 at 33,127 tokens, and then no reply, no
indexed exchange, no facts and no episodic write — while openwebui.log and
compactor.log both recorded HTTP 200 and the only trace of the destroyed turn
was two unattributed WARNING lines. A budget that can be exceeded needs its
failure to be legible, so those tests assert the failure is visible to the USER
and attributable in the LOG.

Run inside the compactor image or any container with the requirements:
    python test_budget_guard.py
"""

import asyncio
import inspect
import io
import json
import logging
import os
import re
import shutil
import sys
import tempfile
from unittest.mock import patch

# Small, predictable window. No MODEL_REPO -> char/4 token estimator.
os.environ.pop("MODEL_REPO", None)
os.environ["MAX_MODEL_LEN"] = "1000"
os.environ["COMPACTOR_GENERATION_RESERVE"] = "200"   # HARD_INPUT_LIMIT = 800
os.environ["COMPACTOR_SUMMARY_MAX_TOKENS"] = "100"
os.environ["COMPACTOR_SUMMARY_INPUT_RESERVE"] = "100"
os.environ["COMPACTOR_KEEP_RECENT_TURNS"] = "4"      # even — the blocker's trigger
# The endpoint-level tests at the bottom of this file exercise the real memory
# layers, which read and write the storage volume. Redirect it to a scratch
# directory BEFORE importing anything that resolves a module-level path
# (memory.STORAGE_ROOT, retrieval.CHROMA_PATH), or those tests scribble facts
# and personas into the deployment's /data. RAG off so retrieval never reaches
# for fastembed/chromadb: episodic recall is not what this file is about, and
# both are installed in the production image, so leaving it on would mean
# building an ONNX embedder to run a budget test.
_TMP_ROOT = tempfile.mkdtemp(prefix="compactor-test-budget-guard-")
os.environ["COMPACTOR_STORAGE_ROOT"] = _TMP_ROOT
os.environ["COMPACTOR_RAG_ENABLED"] = "false"

import dedup  # noqa: E402
import facts  # noqa: E402
import logsetup  # noqa: E402
import main  # noqa: E402
import memory  # noqa: E402
import persona  # noqa: E402
import retrieval  # noqa: E402
import summarizer  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

# Belt and braces alongside COMPACTOR_RAG_ENABLED=false: latch the module
# unavailable outright, so a stray import order or a future default flip cannot
# turn one of these tests into a several-second model download.
# Mirrors test_retrieval._force_unavailable / test_import_guard.
retrieval._available = False
retrieval._embedder = None
retrieval._chroma_collection = None

memory.ensure_storage_layout()

# client=127.0.0.1 and raise_server_exceptions=False mirror test_import_guard —
# the established shape for endpoint-level tests here. The false is what makes
# an unhandled exception inside the handler come back as a 500 status we can
# assert on, instead of propagating and aborting the run somewhere unrelated.
client = TestClient(
    main.app, client=("127.0.0.1", 12345), raise_server_exceptions=False
)


def assert_eq(actual, expected, label):
    if actual != expected:
        print(f"FAIL {label}: expected {expected!r}, got {actual!r}")
        sys.exit(1)
    print(f"  ok   {label}")


def assert_true(cond, label):
    if not cond:
        print(f"FAIL {label}")
        sys.exit(1)
    print(f"  ok   {label}")


def user(text):
    return {"role": "user", "content": text}


def big(role, approx_tokens):
    # char/4 estimator -> 4 chars ~= 1 token
    return {"role": role, "content": "x" * (approx_tokens * 4)}


# The caller's own system prompt: index 0, the persona. Deliberately 200 chars
# — under the guard's 400-char trim floor — so "index 0 survived" can be an
# equality check on the exact string rather than a prefix match.
PERSONA = "PERSONA-SENTINEL " + "p" * 183


def injected(tag, chars=600):
    """One injected memory block, shaped like what inject_system_block emits:
    a system message in the leading system run, after the persona."""
    return {"role": "system", "content": tag + "m" * max(0, chars - len(tag))}


class _CaptureLogs(logging.Handler):
    """Keeps the guard's LogRecords so a test can assert on the LEVEL.

    Text alone is not enough here: the 2026-08-27 line was WARNING-shaped
    success prose ("hard budget enforced: 28054 -> 26565 tokens (limit
    24576)") over a payload the guard had just measured as 2k too large. The
    level is the part that decides whether anyone ever looks."""

    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


def _guard_with_logs(msgs, limit=None, protect=None):
    """_enforce_hard_budget with main's logger tapped. -> (out, records).

    `protect` is passed positionally as the third argument only when a test
    supplies one, so the default-argument tests below exercise the real
    two-argument and one-argument call shapes an existing caller uses."""
    handler = _CaptureLogs()
    lg = logging.getLogger("compactor")
    lg.addHandler(handler)
    try:
        if protect is not None:
            out = main._enforce_hard_budget(msgs, limit, protect)
        elif limit is None:
            out = main._enforce_hard_budget(msgs)
        else:
            out = main._enforce_hard_budget(msgs, limit)
    finally:
        lg.removeHandler(handler)
    return out, handler.records


def _budget_line(records):
    """The guard's single verdict record — WARNING when it fit, ERROR when it
    did not. None if it never spoke (i.e. it had nothing to do)."""
    for r in records:
        if "hard budget" in r.getMessage():
            return r
    return None


_SYS_DROPPED_RE = re.compile(r"dropped (\d+) injected block\(s\) entirely")
_TRIMMED_RE = re.compile(r"trimmed (\d+) injected block\(s\)")
_TURNS_DROPPED_RE = re.compile(r"dropped (\d+) old turn\(s\)")


def _sys_dropped(record):
    m = _SYS_DROPPED_RE.search(record.getMessage()) if record is not None else None
    return int(m.group(1)) if m else 0


def _trimmed(record):
    m = _TRIMMED_RE.search(record.getMessage()) if record is not None else None
    return int(m.group(1)) if m else 0


def _turns_dropped(record):
    m = _TURNS_DROPPED_RE.search(record.getMessage()) if record is not None else None
    return int(m.group(1)) if m else 0


def _systems(msgs):
    """The system messages, in order — the guard's protection is positional,
    so order is the thing under test, not membership."""
    return [m for m in msgs if m.get("role") == "system"]


def _fingerprint(msgs):
    """A readable stand-in for the whole payload: role, length, and the
    sentinel prefix of every message. Comparing two of these says exactly what
    the guard shed differently without dumping kilobytes of filler into the
    failure line."""
    return [
        (m.get("role"), len(main._message_text(m)), main._message_text(m)[:24])
        for m in msgs
    ]


def test_hard_limit_configured():
    print("\n[test] hard budget config")
    assert_eq(main.HARD_INPUT_LIMIT, 800, "HARD_INPUT_LIMIT = MAX_MODEL_LEN - reserve")


def test_under_budget_is_untouched():
    print("\n[test] _enforce_hard_budget — under budget passes through unchanged")
    msgs = [{"role": "system", "content": "persona"}, user("hi")]
    out = main._enforce_hard_budget(msgs)
    assert_eq(out, msgs, "identical list returned")


def _assert_template_valid(out, label_prefix):
    """The Mistral-family invariants the rc6 review found violated: the first
    non-system message must be a USER turn, and non-system roles must
    alternate strictly."""
    roles = [m["role"] for m in out if m.get("role") != "system"]
    assert_true(roles, f"{label_prefix}: at least one non-system turn survives")
    assert_eq(roles[0], "user", f"{label_prefix}: first non-system turn is user")
    for a, b in zip(roles, roles[1:]):
        assert_true(a != b, f"{label_prefix}: roles alternate ({a}->{b})")


def test_the_production_scenario():
    print("\n[test] _enforce_hard_budget — the 2026-08-13 overflow scenario")
    # Oversized injected memory (facts+RAG+summary) on top of a long history —
    # exactly the shape that produced "at least 32769 input tokens". 11 turns:
    # user-first alternation ending on the NEW USER TURN, as real traffic does
    # (the rc6 review caught the old fixture ending on an assistant turn —
    # itself template-invalid).
    msgs = [
        {"role": "system", "content": "P" * (300 * 4)},   # persona
        {"role": "system", "content": "F" * (400 * 4)},   # injected memory
    ] + [big("user" if i % 2 == 0 else "assistant", 100) for i in range(11)]

    before = main.count_tokens(msgs)
    assert_true(before > main.HARD_INPUT_LIMIT, f"starts over budget ({before})")

    out = main._enforce_hard_budget(msgs)
    after = main.count_tokens(out)
    assert_true(after <= main.HARD_INPUT_LIMIT, f"ends within budget ({after})")

    print("\n[test] _enforce_hard_budget — the newest turn is never dropped")
    assert_eq(out[-1]["content"], msgs[-1]["content"], "last turn preserved intact")

    print("\n[test] _enforce_hard_budget — role alternation survives shedding")
    # The rc6 review's confirmed HIGH: stopping mid-pair left the conversation
    # assistant-first, manufacturing the Mistral 400 the guard exists to stop.
    _assert_template_valid(out, "post-shed")


def test_per_request_limit():
    print("\n[test] _enforce_hard_budget — per-request limit parameter")
    # A client-requested max_tokens shrinks the effective input limit; the
    # guard must honor the caller-supplied limit, not the module constant.
    msgs = [{"role": "system", "content": "S" * (100 * 4)}] + [
        big("user" if i % 2 == 0 else "assistant", 60) for i in range(11)
    ]
    out = main._enforce_hard_budget(msgs, 300)
    assert_true(main.count_tokens(out) <= 300, "honors the tighter explicit limit")
    _assert_template_valid(out, "tight-limit")


def test_tokenization_cost_is_bounded():
    print("\n[test] _enforce_hard_budget — full-list tokenizations are O(1), not O(drops)")
    # The rc6 review's other confirmed HIGH: the old loop re-tokenized the
    # ENTIRE message list once per dropped message (O(N^2) on the event loop).
    # Now: one prescreen (no tokenizer), one entry count, per-message counts,
    # and a bounded number of verification counts.
    msgs = [{"role": "system", "content": "S" * (50 * 4)}] + [
        big("user" if i % 2 == 0 else "assistant", 40) for i in range(41)
    ]  # ~1700 tokens; needs ~25 drops to fit 800
    calls = {"full": 0}
    orig = main.count_tokens

    def counting(m):
        if len(m) > 1:
            calls["full"] += 1
        return orig(m)

    main.count_tokens = counting
    try:
        out = main._enforce_hard_budget(msgs)
    finally:
        main.count_tokens = orig
    assert_true(main.count_tokens(out) <= main.HARD_INPUT_LIMIT, "fits after shedding")
    assert_true(
        calls["full"] <= 8,
        f"full-list tokenizations bounded (got {calls['full']}, want <=8)",
    )
    _assert_template_valid(out, "many-drops")


def test_split_messages_keeps_user_first():
    print("\n[test] split_messages — compaction window starts on a USER turn (the blocker)")
    # The rc6 review's BLOCKER: with even KEEP_RECENT_TURNS (4) and a real
    # request's ODD non-system count, keep_recent always began with an
    # assistant turn — every successful compaction emitted a template-invalid
    # conversation. Latent since V1, shielded by the summarize-overflow bug.
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(7):  # u,a,u,a,u,a,u — history pairs + the new user turn
        msgs.append({"role": "user" if i % 2 == 0 else "assistant", "content": f"t{i}"})
    system_msgs, to_summarize, keep_recent = main.split_messages(msgs)
    assert_eq(len(system_msgs), 1, "system preserved")
    assert_eq(keep_recent[0]["role"], "user", "keep window starts with a user turn")
    assert_eq(
        len(to_summarize) + len(keep_recent), 7, "no non-system turn lost in the split"
    )
    roles = [m["role"] for m in keep_recent]
    for a, b in zip(roles, roles[1:]):
        assert_true(a != b, f"keep window alternates ({a}->{b})")

    print("\n[test] split_messages — short conversations untouched")
    short = [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]
    s, t, k = main.split_messages(short)
    assert_eq((len(s), len(t), len(k)), (1, 0, 1), "under-threshold passthrough")


def test_pathological_single_huge_turn():
    print("\n[test] _enforce_hard_budget — one enormous turn still terminates")
    # A single user message larger than the whole window. We cannot drop it
    # (it is the newest turn), so the guard must trim blocks, give up, and
    # RETURN rather than spin forever.
    msgs = [
        {"role": "system", "content": "S" * (900 * 4)},
        big("user", 2000),
    ]
    out = main._enforce_hard_budget(msgs)
    assert_true(isinstance(out, list) and len(out) >= 1, "returned a list, did not hang")
    assert_true(out[-1]["role"] == "user", "the user's own turn survives")


# ---------------------------------------------------------------------------
# v3.1 fix B — the guard could not drop injected memory, and said so at WARNING
#
# 2026-08-27, the phantom conversation (msgs=1):
#   22:14  over budget (26499>24576) but no older turns to summarize
#   22:14  hard budget enforced: 28054 -> 26565 tokens (limit 24576);
#          dropped 0 old turn(s), trimmed 5 injected block(s)
# 26,565 is ABOVE 24,576. Halving was the only thing the guard could do to a
# system message, and injected memory IS a system message — so the layer most
# able to overshoot was the one it could least touch. It shed what it could,
# logged like a success, and forwarded an oversized request.
# ---------------------------------------------------------------------------

def test_injected_blocks_are_droppable():
    print("\n[test] _enforce_hard_budget — injected system blocks can be dropped entirely")
    # The 22:14 shape: ONE user turn and a pile of injected memory. No older
    # turns to shed, and trimming bottoms out at the 400-char floor.
    msgs = (
        [{"role": "system", "content": PERSONA}]
        + [injected(f"BLOCK{i}:") for i in range(14)]
        + [user("what did we decide about the ranger?")]
    )
    before = main.count_tokens(msgs)
    assert_true(before > main.HARD_INPUT_LIMIT, f"starts over budget ({before})")

    out, records = _guard_with_logs(msgs)
    after = main.count_tokens(out)
    assert_true(
        after <= main.HARD_INPUT_LIMIT,
        f"now fits ({after} <= {main.HARD_INPUT_LIMIT})",
    )

    line = _budget_line(records)
    assert_true(line is not None, "the guard logged a verdict")
    assert_true(_sys_dropped(line) > 0, f"sys_dropped is counted in the log: {line.getMessage()}")
    assert_true(
        len([m for m in out if m["role"] == "system"]) < 15,
        "fewer system messages came out than went in",
    )

    # The one system message that must NEVER be dropped. Everything after
    # index 0 is memory we injected; index 0 is who the model thinks it is.
    assert_eq(out[0]["role"], "system", "a system message is still first")
    assert_eq(out[0]["content"], PERSONA, "index 0 — the caller's persona — survives untouched")
    assert_eq(out[-1]["content"], msgs[-1]["content"], "the user's own turn survives intact")


def test_nothing_dropped_when_already_under_budget():
    print("\n[test] _enforce_hard_budget — a payload that already fits is untouched")
    # Deliberately ABOVE the prescreen's limit//2 cutoff, so this reaches the
    # real counting path instead of the cheap early return that
    # test_under_budget_is_untouched exercises. The new dropping stage lives
    # past that gate, and a guard that sheds from a request that already fits
    # is silently deleting memory for nothing.
    msgs = (
        [{"role": "system", "content": PERSONA}]
        + [injected("FACTS:", 800), injected("RECALL:", 800)]
        + [big("user", 100), big("assistant", 100), big("user", 100)]
    )
    est = main._fast_token_estimate(msgs)
    total = main.count_tokens(msgs)
    assert_true(
        est > main.HARD_INPUT_LIMIT // 2,
        f"past the prescreen ({est} > {main.HARD_INPUT_LIMIT // 2})",
    )
    assert_true(total <= main.HARD_INPUT_LIMIT, f"but under the limit ({total})")

    out, records = _guard_with_logs(msgs)
    assert_eq(out, msgs, "same messages returned, nothing shed")
    assert_eq(_budget_line(records), None, "and the guard stayed silent")


def test_alternation_survives_the_dropping_stage():
    print("\n[test] _enforce_hard_budget — alternation still repaired after blocks are dropped")
    # The rc6 blocker was an assistant-first conversation handed to a Mistral
    # template — the 400 the guard exists to prevent, manufactured by the
    # guard. The new dropping stage runs AFTER the repair, so it must only
    # ever remove system messages and never disturb the turn the repair left
    # in front.
    msgs = (
        [{"role": "system", "content": PERSONA}]
        + [injected(f"BLOCK{i}:") for i in range(10)]
        + [big("user" if i % 2 == 0 else "assistant", 100) for i in range(11)]
    )
    out, records = _guard_with_logs(msgs)
    assert_true(main.count_tokens(out) <= main.HARD_INPUT_LIMIT, "fits after shedding")
    assert_true(_sys_dropped(_budget_line(records)) > 0, "the dropping stage actually ran")
    _assert_template_valid(out, "post-drop")
    assert_eq(out[0]["content"], PERSONA, "persona still first")
    assert_eq(out[-1]["content"], msgs[-1]["content"], "newest turn intact")


class FramingTokenizer:
    """A tokenizer whose chat template costs a fixed amount PER CALL — the BOS
    and generation prompt every real template emits once, however many messages
    it is given.

    That per-call constant is why the guard cannot trust arithmetic: it lands
    in count_tokens([m]) for every message, so each per-message cost is an
    OVER-estimate of that message's marginal contribution, and `running` drifts
    BELOW the truth as messages are shed. The verification round then finds the
    list still over budget with droppable blocks in hand — the one state where
    the give-up condition's new `len(sys_idxs) <= 1` clause decides whether the
    guard keeps working or forwards an oversized request.

    encode() keeps this file's 4-chars-per-token convention."""

    FRAMING = "F" * 160  # 40 tokens of framing per call

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return self.FRAMING + "".join(main._message_text(m) for m in messages)

    def encode(self, text):
        return list(range(len(text) // 4))


def test_give_up_waits_for_the_last_droppable_block():
    print("\n[test] _enforce_hard_budget — does not give up while injected blocks remain")
    # Turns are exhausted (one user turn, never dropped) and trimming is
    # exhausted (the 32-trim cap is reached), which is exactly the old give-up
    # condition. With injected blocks still droppable it must NOT stop there.
    orig_tok = main._tokenizer
    try:
        main._tokenizer = FramingTokenizer()
        msgs = (
            [{"role": "system", "content": PERSONA}]
            + [injected(f"BLOCK{i}:") for i in range(40)]
            + [user("does any of this actually fit?")]
        )
        before = main.count_tokens(msgs)
        assert_true(before > main.HARD_INPUT_LIMIT, f"starts far over budget ({before})")

        out, records = _guard_with_logs(msgs)
        after = main.count_tokens(out)
        assert_true(
            after <= main.HARD_INPUT_LIMIT,
            f"kept shedding until it fit ({after} <= {main.HARD_INPUT_LIMIT})",
        )
        line = _budget_line(records)
        assert_true(line is not None, "the guard logged a verdict")
        assert_eq(line.levelno, logging.WARNING, "and it is the success verdict, not ERROR")
        assert_true(_sys_dropped(line) > 0, "blocks were dropped rather than given up on")
        assert_eq(out[0]["content"], PERSONA, "index 0 still survives the long grind")
        assert_eq(out[-1]["content"], msgs[-1]["content"], "as does the user's turn")
    finally:
        main._tokenizer = orig_tok


def test_unfittable_payload_reports_error_not_success():
    print("\n[test] _enforce_hard_budget — an unfittable payload logs ERROR, not WARNING")
    # One user turn larger than the entire limit. It is the newest turn, so it
    # is never dropped and the request goes out anyway — but this is the guard
    # failing at its only job, and vLLM's 400 is the expected next event. It
    # has to read as a failure in the log, at a level someone alerts on.
    msgs = [{"role": "system", "content": PERSONA}, big("user", 2000)]
    out, records = _guard_with_logs(msgs)

    over = main.count_tokens(out) - main.HARD_INPUT_LIMIT
    assert_true(over > 0, f"genuinely does not fit ({over} token(s) over)")
    assert_eq(out[-1]["content"], msgs[-1]["content"], "forwarded with the newest turn intact")

    line = _budget_line(records)
    assert_true(line is not None, "the guard logged a verdict")
    assert_eq(line.levelno, logging.ERROR, "reported at ERROR")
    assert_true(
        f"{over} token(s) over" in line.getMessage(),
        f"names the shortfall: {line.getMessage()}",
    )
    assert_true(
        not any(
            r.levelno <= logging.WARNING and "hard budget enforced" in r.getMessage()
            for r in records
        ),
        "and does not also claim the budget was enforced",
    )


def test_success_path_still_logs_warning():
    print("\n[test] _enforce_hard_budget — shedding that WORKS stays at WARNING")
    # The counterweight to the ERROR branch: routine shedding is routine, and
    # promoting it would bury the line that actually means something.
    msgs = [{"role": "system", "content": PERSONA}] + [
        big("user" if i % 2 == 0 else "assistant", 100) for i in range(11)
    ]
    out, records = _guard_with_logs(msgs)
    assert_true(main.count_tokens(out) <= main.HARD_INPUT_LIMIT, "fits after shedding")
    line = _budget_line(records)
    assert_true(line is not None, "the guard logged a verdict")
    assert_eq(line.levelno, logging.WARNING, "success stays at WARNING")
    assert_true("hard budget enforced" in line.getMessage(), "with the enforced wording")


# ---------------------------------------------------------------------------
# v3.1 fix B2 — protect_system: the guard may spend what WE injected, never
# what the CALLER sent.
#
# The first cut of the dropping stage protected index 0 alone. A caller that
# sends two system messages (persona + a per-request instruction, which the
# OpenAI-compatible API allows and clients do) had its SECOND one deleted
# outright once injected memory ran out — content the pre-v3.1 code would at
# worst have halved, and in the only case that reaches (one user turn larger
# than the whole budget) deleting it does not even achieve the fit.
#
# A separate review then found the same hole twenty lines earlier: the TRIM
# loop picked the largest system block by content length with no protection at
# all, so it could halve the caller's persona while the drop loop below was
# guarded. Halving a persona mid-sentence is a quieter version of deleting it —
# the model still receives something that looks like instructions.
#
# The call site counts protect_system on the ORIGINAL client array, before
# compaction or injection:  sum(1 for m in messages if m["role"] == "system").
# ---------------------------------------------------------------------------

# The caller's SECOND system message: a per-request instruction sitting behind
# the persona. 1000 chars — big enough that a naive largest-block trim would
# reach for it, so "survives byte-for-byte" is a claim about both stages.
CALLER2 = "CALLER-SECOND-SENTINEL " + "c" * 977


def test_regression_callers_second_system_message_is_never_dropped():
    print("\n[test] _enforce_hard_budget — REGRESSION: caller system #2 survives (prior gate B3)")
    # The exact failing case from the prior gate:
    #     B3. caller #2 survives verbatim: False
    #         system messages out: 1 (in 2)
    # Two caller system messages, injected memory on top, and a user turn
    # larger than the entire budget so the guard runs every stage to
    # exhaustion. Pre-fix, the drop loop ran down to sys_idxs[-1] == the
    # caller's own second message and deleted it — while still ending 2,488
    # over, so it bought nothing.
    msgs = (
        [{"role": "system", "content": PERSONA}, {"role": "system", "content": CALLER2}]
        + [injected(f"BLOCK{i}:") for i in range(3)]
        + [big("user", 2000)]
    )
    before = main.count_tokens(msgs)
    assert_true(before > main.HARD_INPUT_LIMIT, f"starts over budget ({before})")

    out, records = _guard_with_logs(msgs, None, 2)

    sys_out = _systems(out)
    assert_eq(len(sys_out), 2, "system messages out: 2 (in 5) — only what we injected was spent")
    assert_eq(sys_out[0]["content"], PERSONA, "caller #1 survives byte-for-byte")
    assert_eq(sys_out[1]["content"], CALLER2, "caller #2 survives byte-for-byte")

    # Spent first, and spent fully: nothing we injected reached the model.
    line = _budget_line(records)
    assert_true(line is not None, "the guard logged a verdict")
    assert_eq(_sys_dropped(line), 3, "all 3 injected blocks were dropped")
    assert_true(
        not any("BLOCK" in m["content"] for m in sys_out),
        "no injected block survived alongside the caller's messages",
    )
    assert_eq(out[-1]["content"], msgs[-1]["content"], "the user's own turn survives intact")


def test_trim_loop_will_not_halve_a_protected_persona():
    print("\n[test] _enforce_hard_budget — the TRIM loop skips the caller's blocks too")
    # Constructed so a naive "largest block" pick takes the persona: it is
    # 2000 chars against the injected block's 600, and both are over the
    # 400-char trim floor. The only correct pick is the injected one.
    persona = "BIG-PERSONA-SENTINEL " + "b" * 1979
    mem = injected("MEM:")
    assert_true(
        len(persona) > len(mem["content"]),
        "fixture: the persona IS the largest system block (a naive pick takes it)",
    )
    msgs = [
        {"role": "system", "content": persona},
        mem,
        big("user", 180),
    ]
    before = main.count_tokens(msgs)
    assert_true(before > main.HARD_INPUT_LIMIT, f"starts over budget ({before})")

    out, records = _guard_with_logs(msgs, None, 1)
    after = main.count_tokens(out)
    assert_true(after <= main.HARD_INPUT_LIMIT, f"the trim achieved the fit ({after})")

    sys_out = _systems(out)
    assert_eq(len(sys_out), 2, "both system messages still present — trim, not drop")
    assert_eq(sys_out[0]["content"], persona, "the persona is NOT halved")
    assert_true(
        "trimmed to fit" not in sys_out[0]["content"],
        "and carries no truncation marker",
    )
    assert_true(
        "trimmed to fit" in sys_out[1]["content"],
        "the injected block took the cut instead",
    )
    assert_true(
        len(sys_out[1]["content"]) < len(mem["content"]),
        "and actually got shorter",
    )
    line = _budget_line(records)
    assert_eq(_trimmed(line), 1, "exactly one block was trimmed")
    assert_eq(_sys_dropped(line), 0, "and nothing needed dropping")


def test_protect_system_defaults_to_one():
    print("\n[test] _enforce_hard_budget — protect_system defaults to 1 for existing callers")
    # Every caller written before this parameter existed passes one or two
    # positional arguments. The default has to reproduce the old behaviour
    # exactly: index 0 protected, everything after it spendable.
    def fixture():
        return (
            [{"role": "system", "content": PERSONA}]
            + [injected(f"BLOCK{i}:") for i in range(3)]
            + [big("user", 2000)]
        )

    omitted, rec_omitted = _guard_with_logs(fixture())
    explicit, rec_explicit = _guard_with_logs(fixture(), None, 1)

    assert_eq(_fingerprint(omitted), _fingerprint(explicit), "omitting the parameter == passing 1")
    assert_true(omitted == explicit, "byte-identical, not merely the same shape")
    assert_eq(
        _sys_dropped(_budget_line(rec_omitted)),
        _sys_dropped(_budget_line(rec_explicit)),
        "and sheds exactly the same amount",
    )

    # Behaviour pins the default BELOW 2: at 2 an injected block would survive
    # that the guard needed to spend.
    sys_out = _systems(omitted)
    assert_eq(len(sys_out), 1, "exactly one system message survives — so the default is not 2")
    assert_eq(sys_out[0]["content"], PERSONA, "and it is the persona, kept as it always was")
    assert_eq(_sys_dropped(_budget_line(rec_omitted)), 3, "all 3 injected blocks spent")

    # Behaviour CANNOT pin it below 1: both loops clamp with max(1, ...), so a
    # default of 0 is indistinguishable at runtime from a default of 1. The
    # written contract is still 1 — a caller reading the signature must see the
    # protection, not infer it from a clamp twenty lines further down — so this
    # last one asserts the literal.
    sig = inspect.signature(main._enforce_hard_budget)
    assert_true("protect_system" in sig.parameters, "the parameter is named protect_system")
    assert_eq(sig.parameters["protect_system"].default, 1, "and its declared default is 1")


def test_zero_caller_system_messages_still_protects_index_zero():
    print("\n[test] _enforce_hard_budget — max(1, protect_system) with a system-less client array")
    # A client can send a bare conversation with no system message at all;
    # compaction (the summary block) and memory injection then ADD them, so
    # the call site's count is 0 while the final list has several. Without the
    # max(1, ...) clamp both loops would run to zero system messages and, in
    # the trim loop, chew the compaction summary in half on the way. The
    # summary is 800 chars here — the largest block, and a naive pick's first
    # target.
    summary = "SUMMARY-SENTINEL " + "s" * 783
    msgs = [
        {"role": "system", "content": summary},
        injected("MEM0:"),
        injected("MEM1:"),
        big("user", 2000),
    ]
    out, records = _guard_with_logs(msgs, None, 0)

    sys_out = _systems(out)
    assert_true(len(sys_out) >= 1, "the guard did not strip every system message")
    assert_eq(out[0]["role"], "system", "index 0 of the final list is still a system message")
    assert_eq(out[0]["content"], summary, "and it survives byte-for-byte — neither dropped nor trimmed")
    assert_true(len(out) >= 2, "the conversation was not reduced to nothing")
    assert_eq(out[-1]["content"], msgs[-1]["content"], "the user's turn is still there")
    assert_eq(_sys_dropped(_budget_line(records)), 2, "both injected blocks were spent instead")


def test_injected_blocks_remain_fully_spendable():
    print("\n[test] _enforce_hard_budget — protection does not make injected memory sticky")
    # The counterweight to the four tests above: guarding the caller's blocks
    # must not cost the guard its reach. With one caller system message, every
    # injected block is still trimmable AND droppable, all the way to zero.
    msgs = (
        [{"role": "system", "content": PERSONA}]
        + [injected(f"MEM{i}:") for i in range(5)]
        + [big("user", 2000)]
    )
    out, records = _guard_with_logs(msgs, None, 1)

    line = _budget_line(records)
    assert_true(line is not None, "the guard logged a verdict")
    assert_eq(_trimmed(line), 5, "every injected block was trimmed")
    assert_eq(_sys_dropped(line), 5, "and then every one of them was dropped")

    sys_out = _systems(out)
    assert_eq(len(sys_out), 1, "one system message left")
    assert_eq(sys_out[0]["content"], PERSONA, "the caller's, untouched")
    assert_true(
        not any("MEM" in main._message_text(m) for m in out),
        "no trace of injected memory in the forwarded payload",
    )


def test_protected_unfittable_payload_still_forwards_at_error():
    print("\n[test] _enforce_hard_budget — protection does not turn into a refusal to send")
    # The failure mode the protection could plausibly introduce: with two
    # caller system messages the drop loop now stops early, so the guard has
    # LESS it may shed. It must still return the payload and forward it — the
    # newest turn is never dropped, vLLM's 400 is the expected next event, and
    # the log line is the only warning anyone gets. Assert the LEVEL: the
    # 2026-08-27 line was WARNING-shaped success prose over an oversized
    # payload, which is why nobody looked.
    msgs = [
        {"role": "system", "content": PERSONA},
        {"role": "system", "content": CALLER2},
        big("user", 2000),
    ]
    out, records = _guard_with_logs(msgs, None, 2)

    over = main.count_tokens(out) - main.HARD_INPUT_LIMIT
    assert_true(over > 0, f"genuinely does not fit ({over} token(s) over)")
    assert_true(isinstance(out, list), "a payload was returned, not an exception")
    assert_eq(out[-1]["content"], msgs[-1]["content"], "forwarded with the newest turn intact")
    assert_eq(len(_systems(out)), 2, "and both caller system messages still aboard")

    line = _budget_line(records)
    assert_true(line is not None, "the guard logged a verdict")
    assert_eq(line.levelno, logging.ERROR, "reported at ERROR, not WARNING")
    assert_true(
        f"{over} token(s) over" in line.getMessage(),
        f"naming the shortfall: {line.getMessage()}",
    )
    assert_true(
        not any(
            r.levelno <= logging.WARNING and "hard budget enforced" in r.getMessage()
            for r in records
        ),
        "and never claims the budget was enforced",
    )


def test_alternation_repair_survives_protected_shedding():
    print("\n[test] _enforce_hard_budget — rc6 alternation repair still holds with protect_system=2")
    # (a) The repair path itself. Sized so the turn-drop loop reaches the fit
    #     after an ODD number of drops, leaving the conversation
    #     assistant-first — the rc6 BLOCKER's exact shape. The repair must
    #     still shed the stray assistant turn with the new protection in
    #     place, and must not reach for a caller system message to do it.
    msgs = (
        [
            {"role": "system", "content": PERSONA},
            {"role": "system", "content": CALLER2[:1000]},
        ]
        + [injected("MEM0:"), injected("MEM1:")]
        + [big("user" if i % 2 == 0 else "assistant", 40) for i in range(11)]
    )
    out, records = _guard_with_logs(msgs, None, 2)
    assert_true(main.count_tokens(out) <= main.HARD_INPUT_LIMIT, "fits after shedding")
    _assert_template_valid(out, "protected-repair")

    turns = [m for m in out if m.get("role") != "system"]
    assert_eq(len(turns), 3, "three turns survive — alternation is a real constraint here")
    line = _budget_line(records)
    # 7 drops get it under the limit and leave an assistant in front; the 8th
    # is the repair. Arithmetic alone stops at 7, so this pins the repair ran.
    assert_eq(_turns_dropped(line), 8, "the repair contributed the 8th drop")
    sys_out = _systems(out)
    assert_eq(sys_out[0]["content"], PERSONA, "caller #1 untouched by the repair")
    assert_eq(sys_out[1]["content"], CALLER2[:1000], "caller #2 untouched by the repair")

    # (b) The same invariant when ALL THREE stages run: turns to exhaustion,
    #     then every injected block trimmed, then every one of them dropped.
    #     The two later stages only ever touch system messages, so the turn the
    #     turn-stage left in front must still be there — and must still be a
    #     user turn. The final turn is oversized so the turn stage cannot
    #     reach the fit on its own and the other two are forced to run.
    msgs = (
        [
            {"role": "system", "content": PERSONA},
            {"role": "system", "content": CALLER2[:1000]},
        ]
        + [injected(f"MEM{i}:") for i in range(6)]
        + [big("user" if i % 2 == 0 else "assistant", 40) for i in range(10)]
        + [big("user", 480)]
    )
    out, records = _guard_with_logs(msgs, None, 2)
    assert_true(main.count_tokens(out) <= main.HARD_INPUT_LIMIT, "fits after all three stages")
    _assert_template_valid(out, "protected-all-stages")

    line = _budget_line(records)
    assert_true(_turns_dropped(line) > 0, "the turn stage ran")
    assert_true(_trimmed(line) > 0, "the trim stage ran")
    assert_eq(_trimmed(line), 6, "every injected block trimmed")
    assert_eq(_sys_dropped(line), 6, "then every injected block dropped")
    sys_out = _systems(out)
    assert_eq(len(sys_out), 2, "and it stopped at the caller's two")
    assert_eq(sys_out[0]["content"], PERSONA, "caller #1 verbatim")
    assert_eq(sys_out[1]["content"], CALLER2[:1000], "caller #2 verbatim")
    assert_eq(out[-1]["content"], msgs[-1]["content"], "newest turn intact")


# ---------------------------------------------------------------------------
# v3.1 fix B2, the CALL SITE — the half of the fix no direct test can reach
#
# A mutation matrix over _enforce_hard_budget's protect_system work killed
# twelve mutations and left one alive:
#
#     M9  main.py call site  caller_system -> literal 1   SURVIVED
#
# Every test above this line passes protect_system in itself, so none of them
# can tell whether the endpoint computes it or hands over a constant. The
# signature default is 1, so with the call site broken the protection silently
# reverts to pre-fix behaviour — the caller's second system message gets halved
# and then deleted — while all 154 assertions above still print ok.
#
# The call site is the fix:
#
#     caller_system = sum(1 for m in messages if m.get("role") == "system")
#     body["messages"] = await run_in_threadpool(
#         _enforce_hard_budget, body["messages"], effective_limit, caller_system
#     )
#
# `messages` there is the ORIGINAL client array — before compaction's summary
# block and before memory injection, both of which ADD system messages. So the
# number is deliberately smaller than the count of system messages in the list
# the guard receives, and that gap is what these two tests pin: a literal 1 is
# wrong, and so is counting `body["messages"]`.
#
# Both drive the real endpoint. vLLM is stubbed at httpx.AsyncClient, which is
# also the only place the forwarded payload can be observed from outside the
# handler — the endpoint returns vLLM's response, not its own request.
# ---------------------------------------------------------------------------

# One fact, seeded on disk, is enough to make memory injection fire for real —
# it adds a third system message between the caller's blocks and the
# conversation, which is exactly the layer the guard is allowed to spend.
FACT_SENTINEL = "INJECTED-FACT-SENTINEL"
FACT_TEXT = FACT_SENTINEL + " " + "f" * 600


class _StubResponse:
    """A minimally well-formed non-streaming completion, so the handler runs
    its normal 200 path instead of one of the degraded branches."""

    status_code = 200
    text = ""

    def json(self):
        return {
            "id": "stub",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
        }


class _StubVLLM:
    """Stands in for httpx.AsyncClient on the request path, and records the
    JSON body the endpoint actually POSTs upstream.

    That recorded body is the assertion surface: it is the payload vLLM would
    have templated, after compaction, injection, the budget guard and both
    merge passes. Nothing the handler returns reveals it."""

    sent: list[dict] = []

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None, **kwargs):
        _StubVLLM.sent.append(json)
        return _StubResponse()

    def stream(self, *args, **kwargs):
        raise AssertionError("these tests drive the non-streaming path only")

    async def aclose(self):
        pass


def _swallow_tail(coro, label=None):
    """Stand-in for main._fire_and_forget.

    The post-response memory tail extracts facts, embeds and rolls up
    summaries — none of which this file is testing, all of which would write
    to the scratch volume and reach for vLLM. Closing the coroutine also keeps
    Python from warning that it was never awaited."""
    # `label` mirrors main._fire_and_forget's signature. Named explicitly
    # rather than swallowed by **kwargs: a stub that accepts anything can
    # never fail when the real signature moves, and this file's stub going
    # stale is what a signature change looks like when it breaks.
    try:
        coro.close()
    except Exception:
        pass


def _post_chat(messages, conv_id):
    """One real POST /v1/chat/completions with vLLM stubbed.

    -> (response, forwarded_body, log_records). forwarded_body is None if the
    handler never reached the upstream call."""
    _StubVLLM.sent.clear()
    handler = _CaptureLogs()
    lg = logging.getLogger("compactor")
    lg.addHandler(handler)
    try:
        with patch.object(main.httpx, "AsyncClient", _StubVLLM), \
             patch.object(main, "_fire_and_forget", _swallow_tail):
            r = client.post(
                "/v1/chat/completions",
                json={"model": "stub-model", "messages": messages, "stream": False},
                headers={"X-Conversation-Id": conv_id},
            )
    finally:
        lg.removeHandler(handler)
    return r, (_StubVLLM.sent[-1] if _StubVLLM.sent else None), handler.records


def _system_text(msgs):
    """All system content in the forwarded payload, joined.

    _merge_adjacent_system_messages runs AFTER the guard and collapses the
    caller's two blocks into one message, so "did CALLER2 survive" is a
    substring question about the merged text, not a question about which index
    it landed at. The strings are sentinels, so a substring hit is still an
    exact-bytes claim about that block's content."""
    return "\n\n".join(
        main._message_text(m) for m in msgs if m.get("role") == "system"
    )


def test_endpoint_forwards_both_caller_system_messages():
    print("\n[test] POST /v1/chat/completions — the caller's system messages reach vLLM (M9)")
    # The property, end to end: the guard may spend the memory WE injected and
    # may not spend what the CLIENT sent. Two caller system messages, one
    # injected facts block, and a final user turn larger than the entire
    # budget — so the guard runs every stage to exhaustion and cannot reach the
    # fit whatever it sheds. That last part is what makes this test decisive:
    # if the payload could be made to fit by dropping the injected block alone,
    # a broken call site would stop there too and the mutation would live.
    cid = "m9-endpoint"
    facts.save_facts(cid, [{"text": FACT_TEXT, "added_turn": 1, "last_used": 100}])
    msgs = [
        {"role": "system", "content": PERSONA},
        {"role": "system", "content": CALLER2},
        big("user", 2000),
    ]
    assert_eq(len(_systems(msgs)), 2, "fixture: the CLIENT sent two system messages")

    r, forwarded, records = _post_chat(msgs, cid)
    assert_eq(r.status_code, 200, f"the request completed (body: {r.text[:200]!r})")
    assert_true(forwarded is not None, "the request reached the vLLM stub")

    # Preconditions, asserted rather than assumed — the two claims below are
    # only about the call site if injection really happened and the guard
    # really spent it. Without these, a fixture that quietly stopped injecting
    # would leave this test green and testing nothing.
    line = _budget_line(records)
    assert_true(line is not None, "the guard ran on the request path and logged a verdict")
    assert_true(
        _sys_dropped(line) >= 1,
        f"injected memory was there to spend, and was spent: {line.getMessage()}",
    )

    sys_text = _system_text(forwarded["messages"])
    assert_true(PERSONA in sys_text, "caller system #1 reached the model byte-for-byte")
    # THE ASSERTION M9 BREAKS. With a literal 1 at the call site the trim loop
    # halves CALLER2 twice and the drop loop then deletes it outright — and
    # still does not achieve the fit, so it buys nothing.
    assert_true(CALLER2 in sys_text, "caller system #2 reached the model byte-for-byte")
    assert_true(
        "trimmed to fit" not in sys_text,
        "and neither of them was halved on the way",
    )
    assert_true(
        FACT_SENTINEL not in sys_text,
        "the injected facts block was spent instead — protection is not stickiness",
    )
    assert_eq(
        forwarded["messages"][-1]["content"],
        msgs[-1]["content"],
        "the user's own turn was forwarded intact",
    )


def test_call_site_passes_the_callers_system_count():
    print("\n[test] the call site computes protect_system rather than passing a literal (M9)")
    # The narrower companion: replace the guard with a recorder and read the
    # positional arguments the endpoint hands it. Where the test above asserts
    # the PROPERTY, this one names the defect — the number itself — so a
    # failure points at the call site instead of at the payload.
    cid = "m9-spy"
    facts.save_facts(cid, [{"text": FACT_TEXT, "added_turn": 1, "last_used": 100}])
    msgs = [
        {"role": "system", "content": PERSONA},
        {"role": "system", "content": CALLER2},
        user("what did we decide about the ranger?"),
    ]
    assert_eq(len(_systems(msgs)), 2, "fixture: the CLIENT sent two system messages")

    seen = {}

    def recorder(messages, limit=None, protect_system=None, report=None):
        seen["messages"] = list(messages)
        seen["limit"] = limit
        seen["protect_system"] = protect_system
        seen["report"] = report
        return messages

    with patch.object(main, "_enforce_hard_budget", recorder):
        r, forwarded, _records = _post_chat(msgs, cid)

    assert_eq(r.status_code, 200, f"the request completed (body: {r.text[:200]!r})")
    assert_true("messages" in seen, "the guard was called on the request path")
    assert_eq(seen["limit"], main.HARD_INPUT_LIMIT, "with the request's effective limit")
    # 2, not 1 (the signature default, i.e. the M9 mutation) and not 3 (the
    # count of what was actually passed, i.e. counting the wrong array).
    assert_eq(
        seen["protect_system"], 2,
        "protect_system == the system messages the CLIENT sent",
    )
    assert_eq(
        len(_systems(seen["messages"])), 3,
        "...while the list handed to the guard carries three — injection ran first",
    )
    assert_true(
        any(FACT_SENTINEL in main._message_text(m) for m in seen["messages"]),
        "and the third one is the injected block, not a second caller message",
    )
    # v3.1 D4: the fourth argument. The margin's blast radius is decided by
    # whether the rejection path can tell a 400 the guard PREDICTED from one
    # that surprised it, and it can only tell if the call site hands the guard
    # somewhere to write that down. A None here is the whole mechanism dead.
    assert_true(
        isinstance(seen["report"], dict),
        f"the call site passes a report dict for the guard's verdict: "
        f"{seen['report']!r}",
    )


class NoChatTemplate:
    """A tokenizer that loads fine and then refuses to template — the shape the
    served model actually has. `coder3101/Cydonia-24B-v4.3-vision-heretic`
    carries no chat_template.jinja, so apply_chat_template raises before jinja2
    is even reached. encode() keeps this file's 4-chars-per-token convention."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        raise ValueError("tokenizer.chat_template is not set")

    def encode(self, text):
        return list(range(len(text) // 4))


def test_tier2_fallback_is_not_silent():
    print("\n[test] count_tokens — the tier-2 fallback announces itself")
    # v3.1 P0-0 / F60. count_tokens has three tiers and tier 1 has never run in
    # production: jinja2 was absent from the compactor venv AND the served
    # model carries no chat template. Tier 2 caught that with a bare
    # `except Exception:` and no log statement, so a total loss of counting
    # accuracy presented as normal operation for months — in the component
    # whose only job is to know how big things are. Tier 3 (:132) at least
    # warned. This asserts tier 2 now does too.
    orig_tok = main._tokenizer
    root = logging.getLogger()
    orig_stream = root.handlers[0].stream
    buf = io.StringIO()
    try:
        logsetup._reset_log_once_for_tests()
        main._tokenizer = NoChatTemplate()
        root.handlers[0].stream = buf

        msgs = [big("user", 40), big("assistant", 40), big("user", 40)]
        n = main.count_tokens(msgs)
        out = buf.getvalue()
        assert_eq(n, 3 * (40 + 4), "still returns a usable per-message count")
        assert_true("WARNING" in out, "reported at WARNING, matching tier 3")
        assert_true("ValueError" in out, "names the exception type")
        assert_true("UNDERCOUNT" in out.upper(), "names the consequence, not just the error")

        print("\n[test] count_tokens — the tier-2 warning does not repeat")
        # _enforce_hard_budget calls count_tokens once for the whole list and
        # then once per message, so a line per call would make the fix its own
        # denial of service. Once per process, via logsetup.log_once.
        buf.truncate(0)
        buf.seek(0)
        for _ in range(5):
            main.count_tokens(msgs)
        assert_eq(buf.getvalue(), "", "silent after the first line")
    finally:
        main._tokenizer = orig_tok
        root.handlers[0].stream = orig_stream
        logsetup._reset_log_once_for_tests()


def test_chunk_to_budget():
    print("\n[test] _chunk_to_budget — splits oversized input for the summarizer")
    turns = [big("user", 100) for _ in range(10)]   # ~1000 tokens total
    batches = main._chunk_to_budget(turns, 300)
    assert_true(len(batches) > 1, f"split into multiple batches ({len(batches)})")
    assert_eq(sum(len(b) for b in batches), 10, "no turn lost across batches")
    for b in batches:
        # Each batch fits, unless it is a single oversized turn (allowed).
        assert_true(
            main.count_tokens(b) <= 300 or len(b) == 1,
            "batch within budget (or a lone oversized turn)",
        )

    print("\n[test] _chunk_to_budget — small input stays one batch")
    assert_eq(len(main._chunk_to_budget([user("hi"), user("there")], 1000)), 1, "single batch")

    print("\n[test] _chunk_to_budget — a lone oversized turn is not dropped")
    batches = main._chunk_to_budget([big("user", 5000)], 100)
    assert_eq(len(batches), 1, "one batch")
    assert_eq(len(batches[0]), 1, "the turn is kept, not silently discarded")


# ---------------------------------------------------------------------------
# When the guard loses: what the user sees, and what the log says
# ---------------------------------------------------------------------------
#
# The guard forwards an over-budget payload rather than dropping the user's
# newest turn, and says so at ERROR. vLLM then rejects it. Everything below is
# about that second half.
#
# Why it is in THIS file: a context-length 400 is the budget guard being wrong,
# and these assert the two things the 2026-08-24 record could not supply — a
# user who is told their message failed, and a log line that names the
# conversation and both token counts.


def ctx_400(actual_tokens):
    """A vLLM context-length 400 body, in the production shape.

    Shares its wording with test_modality.ctx_400 because both are parsed by
    the same two functions; the window here is this file's 1000-token one so
    the fixture stays internally consistent."""
    return (
        '{"error":{"message":"This model\'s maximum context length is 1000 '
        'tokens. However, you requested 0 output tokens and your prompt '
        'contains ' + str(actual_tokens) + ' input tokens, for a total of ... '
        '(parameter=input_tokens)"}}'
    )


# A 400 that is NOT about size. Deliberately not the "not a multimodal model"
# body: that one flips main._backend_multimodal for the life of the process and
# would silently strip images out of every later test in this file.
OTHER_400 = '{"error":{"message":"only user and assistant roles are supported!"}}'


class _StubStreamResponse:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body

    async def aread(self):
        return self._body.encode()

    async def aiter_raw(self):
        raise AssertionError("the error branch must not read the body as a stream")
        yield b""   # pragma: no cover - makes this an async generator


class _StubStreamCM:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *exc):
        return False


class _StubRejectedResponse:
    """What the non-streaming path gets back: a real status and a parseable
    error body, since that path reads r.json() before it looks at the status."""

    def __init__(self, status_code, body):
        self.status_code = status_code
        self.text = body

    def json(self):
        return json.loads(self.text)


class _StubVLLMRefusing(_StubVLLM):
    """_StubVLLM that refuses every request, on both halves of the handler.

    `status` and `body` are class attributes so a test sets the rejection it
    wants before posting."""

    status = 400
    body = ""

    def stream(self, method, url, json=None, **kwargs):
        _StubVLLM.sent.append(json)
        return _StubStreamCM(_StubStreamResponse(self.status, self.body))

    async def post(self, url, json=None, **kwargs):
        _StubVLLM.sent.append(json)
        return _StubRejectedResponse(self.status, self.body)


def _post_rejected(messages, conv_id, *, status=400, body=None, stream=True,
                   exact_tokens=None):
    """POST one chat completion that vLLM refuses.

    -> (response, log records, list of coroutines handed to _fire_and_forget).

    That third value is the point of the helper: "was this turn memorized" is
    not visible in the response or in the log, and a rejected request must not
    reach the memory tail at all.

    _BUDGET_MARGIN is saved and restored. _note_backend_rejection moves it, it
    is a module global with no reset, and a test that left it moved would
    silently shrink the budget for every test that runs after it.
    """
    body = ctx_400(1234) if body is None else body
    _StubVLLM.sent.clear()
    tails = []

    def _record_tail(coro, label=None):
        tails.append(coro)
        try:
            coro.close()
        except Exception:
            pass

    handler = _CaptureLogs()
    lg = logging.getLogger("compactor")
    lg.addHandler(handler)
    margin_before = main._BUDGET_MARGIN
    stub = type("_S", (_StubVLLMRefusing,), {"status": status, "body": body})
    try:
        with patch.object(main.httpx, "AsyncClient", stub), \
             patch.object(main, "_fire_and_forget", _record_tail), \
             patch.object(main, "count_tokens_exact", lambda _m: exact_tokens):
            r = client.post(
                "/v1/chat/completions",
                json={"model": "stub-model", "messages": messages, "stream": stream},
                headers={"X-Conversation-Id": conv_id},
            )
    finally:
        lg.removeHandler(handler)
        main._BUDGET_MARGIN = margin_before
    return r, handler.records, tails


def _sse_chunks(response):
    """The parsed `data:` payloads of an SSE body, [DONE] excluded."""
    out = []
    for block in response.text.split("\n\n"):
        for line in block.split("\n"):
            if not line.startswith("data: "):
                continue
            payload = line[6:].strip()
            if payload and payload != "[DONE]":
                out.append(json.loads(payload))
    return out


def _assistant_text(chunks):
    return "".join(
        c.get("choices", [{}])[0].get("delta", {}).get("content") or ""
        for c in chunks
    )


def _rejection_line(records):
    """The ERROR record for a lost turn, or None.

    Matched on level as well as text: the pre-v3.1 line existed, at WARNING,
    in a file the runbook did not send anyone to. The level is the fix."""
    for rec in records:
        if rec.levelno != logging.ERROR:
            continue
        msg = rec.getMessage()
        if "REQUEST REJECTED by vLLM" in msg or "vLLM FAILED this request" in msg:
            return rec
    return None


def test_context_400_tells_the_user_the_message_did_not_go_through():
    print("\n[test] stream 400 — the user is told the turn failed, not given a reply")
    r, _records, _tails = _post_rejected([user("what did we decide?")], "rej-visible")

    # The condition that makes everything else necessary, asserted rather than
    # assumed: StreamingResponse commits its status before vLLM answers, so the
    # rejection cannot be expressed as an HTTP status on this path.
    assert_eq(r.status_code, 200, "the stream had already committed HTTP 200")

    chunks = _sse_chunks(r)
    text = _assistant_text(chunks)
    assert_true(
        main._REJECTED_PREAMBLE in text,
        f"the reply leads with the outcome: {text[:80]!r}",
    )
    assert_true(
        "nothing about this turn was saved to memory" in text,
        "and says the turn was not remembered either",
    )
    assert_true(
        main.MODEL_RESTART_MESSAGE not in text,
        "and does NOT claim the backend is restarting — it answered, refusing",
    )
    assert_true("[DONE]" in r.text, "the stream still terminates cleanly")


def test_context_400_is_typed_so_a_client_can_tell_it_from_a_reply():
    print("\n[test] stream 400 — finish_reason 'error' + an error object (INCIDENT A5)")
    # A5: the old pair ended finish_reason "stop", which to OpenWebUI is an
    # ordinary successful completion whose text happens to read like an
    # apology. Nothing downstream could distinguish "model replied" from
    # "request rejected" — so the transcript kept the turn and memory did not.
    r, _records, _tails = _post_rejected([user("what did we decide?")], "rej-typed")
    chunks = _sse_chunks(r)
    finals = [c for c in chunks if c["choices"][0].get("finish_reason")]
    assert_eq(len(finals), 1, "exactly one terminal chunk")
    assert_eq(finals[0]["choices"][0]["finish_reason"], "error", "finish_reason is 'error'")
    err = finals[0].get("error") or {}
    assert_eq(err.get("code"), "context_length_exceeded", "the error names the cause")
    assert_eq(err.get("type"), "invalid_request_error", "typed as a rejection, not an outage")


def test_context_400_logs_an_error_naming_the_conversation_and_both_counts():
    print("\n[test] stream 400 — one ERROR carrying conv_id, our count and vLLM's")
    r, records, _tails = _post_rejected([user("what did we decide?")], "rej-logged")
    line = _rejection_line(records)
    assert_true(line is not None, "a rejection was logged at ERROR")
    msg = line.getMessage()
    assert_true("conv=rej-logged" in msg, f"the line names the conversation: {msg[:120]!r}")
    assert_true("we measured" in msg and "the local tokenizer" in msg,
                "it names our own count AND which counter produced it")
    assert_true("vLLM counted 1,234" in msg, "it carries vLLM's true count")
    assert_true("UNDERCOUNTED" in msg, "and states which way the gap runs")
    assert_true("no access log will show this" in msg,
                "and warns that the 200 upstream is not evidence of success")
    assert_true("produced no reply" in msg, "and says the turn is gone")
    # The 2026-08-24 log had no such line at all, so a passing assertion above
    # must not be satisfiable by an ordinary success.
    assert_eq(r.status_code, 200, "still a 200 on the wire — that is the problem")


def test_the_log_line_names_tokenize_when_tokenize_answered():
    print("\n[test] stream 400 — the count's SOURCE is reported, not assumed")
    # count_tokens and count_tokens_exact disagree by ~50% on assistant
    # content, so "we measured N" is ambiguous until the line says which one
    # measured it — and the reader is trying to locate an undercount.
    _r, records, _tails = _post_rejected(
        [user("what did we decide?")], "rej-source", exact_tokens=777,
    )
    msg = _rejection_line(records).getMessage()
    assert_true("we measured 777 tokens with vLLM's /tokenize" in msg,
                f"the line names /tokenize as the source: {msg[:160]!r}")


def test_a_rejected_turn_is_never_memorized():
    print("\n[test] stream 400 — no memory tail fires for a turn that produced nothing")
    _r, _records, tails = _post_rejected([user("what did we decide?")], "rej-no-tail")
    assert_eq(len(tails), 0, "the memory tail was not fired")


def test_retry_is_promised_only_when_the_rejection_taught_us_something():
    print("\n[test] stream 400 — 'send it again' only when the margin actually moved")
    # Observed 2026-08-27: three consecutive failures moved the margin +127
    # each while it needed ~5250, and the same "send it again" advice produced
    # the same failure. Advice that cannot work is a lie with a friendly face.
    margin_before = main._BUDGET_MARGIN
    try:
        main._BUDGET_MARGIN = 0
        r, _records, _t = _post_rejected([user("hi")], "rej-retry")
        first = _assistant_text(_sse_chunks(r))
        assert_true(main.CONTEXT_OVERFLOW_RETRY.strip() in first,
                    "a rejection that tightened the budget invites a resend")

        # Now pin the margin at the cap so the same rejection can teach nothing.
        main._BUDGET_MARGIN = main.MAX_MODEL_LEN // 4
        r, _records, _t = _post_rejected([user("hi")], "rej-no-retry")
        second = _assistant_text(_sse_chunks(r))
        assert_true(main.CONTEXT_OVERFLOW_NO_RETRY.strip() in second,
                    "a rejection that taught nothing says so instead")
        assert_true(main.CONTEXT_OVERFLOW_RETRY.strip() not in second,
                    "and does not invite a resend that would fail identically")
    finally:
        main._BUDGET_MARGIN = margin_before


def test_a_non_size_400_does_not_blame_the_context_window():
    print("\n[test] stream 400 — a non-size rejection gets the generic text")
    r, _records, _t = _post_rejected([user("hi")], "rej-other", body=OTHER_400)
    text = _assistant_text(_sse_chunks(r))
    assert_true(main._REJECTED_PREAMBLE in text, "still leads with the outcome")
    assert_true("too large for the model's context window" not in text,
                "but does not invent a size problem")
    assert_true(main.MODEL_RESTART_MESSAGE not in text,
                "and does not claim an outage either")
    code = _sse_chunks(r)[-1]["error"]["code"]
    assert_eq(code, "backend_rejected", "typed as a generic rejection")


def test_a_backend_5xx_on_the_stream_is_also_logged_as_a_lost_turn():
    print("\n[test] stream 500 — logged at ERROR, and NOT called a rejection")
    r, records, tails = _post_rejected(
        [user("hi")], "rej-5xx", status=500, body='{"error":"internal"}',
    )
    line = _rejection_line(records)
    assert_true(line is not None, "the lost turn is logged at ERROR")
    msg = line.getMessage()
    assert_true("vLLM FAILED this request (HTTP 500)" in msg,
                f"a backend fault is not described as a rejection: {msg[:120]!r}")
    assert_eq(len(tails), 0, "and nothing is memorized")
    # The 5xx text is still the restart message: on a 5xx the backend really is
    # unhealthy, so that message is true. Only the 4xx case was the lie.
    assert_true(main.MODEL_RESTART_MESSAGE in _assistant_text(_sse_chunks(r)),
                "the user is told the backend is unavailable")


def test_nonstream_400_logs_the_loss_and_skips_the_memory_tail():
    print("\n[test] non-stream 400 — relayed as a 400, logged at ERROR, not memorized")
    # This path always handed the client vLLM's real status, which is why the
    # invisible failure was the stream path. What it did NOT do was say which
    # conversation lost a turn — and it fired the memory tail before it looked
    # at the status at all (v3.1 F20).
    r, records, tails = _post_rejected(
        [user("what did we decide?")], "rej-nonstream", stream=False,
    )
    assert_eq(r.status_code, 400, "vLLM's own status is relayed verbatim")
    line = _rejection_line(records)
    assert_true(line is not None, "the lost turn is logged at ERROR")
    msg = line.getMessage()
    assert_true("conv=rej-nonstream" in msg, "naming the conversation")
    assert_true("vLLM counted 1,234" in msg, "and carrying vLLM's true count")
    assert_true("no access log will show this" not in msg,
                "and NOT claiming the 200 caveat, which is false on this path")
    assert_eq(len(tails), 0, "the memory tail was not fired for a refused turn")


# ---------------------------------------------------------------------------
# The tail's inputs: what the extractor and dedup are told
# ---------------------------------------------------------------------------


def _tail_spies(conv_id, touched, injected, *, extracted="- A new fact."):
    """Run one _async_tail with the two vLLM-facing calls replaced by spies.

    -> (extraction kwargs+args seen, dedup kwargs seen). Summaries and episodic
    indexing are off; this is only about what the tail hands its collaborators.
    """
    seen = {}

    async def spy_extract(_client, _url, _model, _user, _asst, existing, **kwargs):
        seen["existing"] = existing
        seen["extract_kwargs"] = kwargs
        return [extracted]

    async def spy_dedup(_client, _url, _model, combined, **kwargs):
        seen["dedup_kwargs"] = kwargs
        return combined, 0

    kwargs = {} if injected is None else {"injected_facts": injected}
    with patch.object(facts, "extract_facts_from_exchange", spy_extract), \
         patch.object(dedup, "dedup_facts", spy_dedup), \
         patch.object(summarizer, "enabled", lambda: False):
        asyncio.run(main._async_tail(
            conv_id, touched, "and then?", "Lyra drew her bow.", 9,
            [{"role": "user", "content": "and then?"}], **kwargs,
        ))
    return seen


def _oversized_store(n=8, chars=1200):
    """A fact store comfortably past COMPACTOR_MAX_FACTS_TOKENS, so
    select_for_injection really is narrower than the whole thing."""
    return [
        {"text": f"F{i} " + "s" * chars, "added_turn": i, "last_used": 1748000000 + i}
        for i in range(1, n + 1)
    ]


def test_extraction_is_handed_the_injected_subset_not_the_whole_store():
    print("\n[test] handoff — the extractor sees what the MODEL saw, not the store")
    # facts.py now trims its own input, so the whole store no longer overflows
    # the window. It is still the wrong list: the trim it would apply is a
    # second, later opinion about which facts matter. Handing it the injected
    # subset means "already known" means the same thing on both sides of the
    # exchange.
    store = _oversized_store()
    injected = facts.select_for_injection(store)
    assert_true(
        0 < len(injected) < len(store),
        f"fixture: injection is really narrower than the store "
        f"({len(injected)} of {len(store)})",
    )
    seen = _tail_spies("tail-injected", store, injected)
    assert_eq(len(seen["existing"]), len(injected),
              "the extractor was handed the injected subset")
    assert_true(seen["existing"] is injected, "and the very list the request path built")


def test_extraction_is_bounded_even_for_a_caller_that_passes_nothing():
    print("\n[test] handoff — the default narrows too, so no caller can pass the store")
    # injected_facts is keyword-only with a default so its arrival breaks no
    # caller. The default has to be select_for_injection(store), not the store:
    # a default that reintroduces the defect for un-updated callers is not a
    # default, it is the defect with a nicer signature.
    store = _oversized_store()
    seen = _tail_spies("tail-default", store, None)
    assert_eq(len(seen["existing"]), len(facts.select_for_injection(store)),
              "an omitted injected_facts still yields the bounded set")
    assert_true(len(seen["existing"]) < len(store), "and not the whole store")


def test_extraction_and_dedup_are_told_which_conversation():
    print("\n[test] handoff — conv_id reaches facts.extract and dedup.dedup_facts")
    # facts: without it a failed extraction logs "conv=?" and a lost memory
    # cannot be traced to the turn that lost it.
    # dedup: without it the refusal memo is disabled outright, so every pass
    # re-asks the model clusters it has already refused to merge (I-6).
    cid = "tail-conv-id"
    store = [
        {"text": "The ranger is Lyra.", "added_turn": 1, "last_used": 1748000000},
        {"text": "Lyra carries a yew bow.", "added_turn": 2, "last_used": 1748000001},
    ]
    # On disk as well as in hand: the tail re-reads the store inside its lock,
    # and inline dedup only runs when that re-read plus the new fact make two.
    facts.save_facts(cid, store)
    seen = _tail_spies(cid, store, store)
    assert_eq(seen["extract_kwargs"].get("conv_id"), cid,
              "the extractor is told the conversation")
    assert_true("dedup_kwargs" in seen, "fixture: inline dedup actually ran")
    assert_eq(seen["dedup_kwargs"].get("conv_id"), cid,
              "dedup is told the conversation")


def test_the_request_path_hands_the_tail_the_list_it_injected():
    print("\n[test] handoff — the call site passes injected_facts, end to end")
    # The narrower companion to the two above: they assert what _async_tail
    # does with the argument, this asserts the endpoint supplies it. Without
    # this, a call site that silently stopped passing it would fall back to the
    # default and every assertion above would still be green.
    cid = "tail-callsite"
    store = _oversized_store()
    facts.save_facts(cid, store)
    seen = {}

    def recorder(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs

        async def _noop():
            return None

        return _noop()

    with patch.object(main, "_async_tail", recorder):
        r, _forwarded, _records = _post_chat([user("and then?")], cid)

    assert_eq(r.status_code, 200, f"the request completed (body: {r.text[:200]!r})")
    passed = seen.get("kwargs", {}).get("injected_facts")
    assert_true(passed is not None, "injected_facts was passed by keyword")
    assert_true(0 < len(passed) < len(store),
                f"and it is the bounded subset ({len(passed)} of {len(store)})")
    assert_eq(len(seen["args"][1]), len(store),
              "while touched_facts is still the WHOLE store, so LRU keeps its signal")


# =====================================================================
# v3.1 A14 — the prescreen, which is the last place on the request path where
# an irreversible forward-without-measuring decision is made.
# =====================================================================


def _guard_watching_tokenize(msgs, limit):
    """_enforce_hard_budget with count_tokens_exact replaced by a spy that
    answers None. -> (out, records, call_count).

    The spy answering None rather than a number is deliberate: it keeps the
    guard on exactly the arithmetic it uses today, so the only thing this
    helper changes is that "did anything measure this payload" becomes
    observable. Nothing else in the package can answer that question."""
    calls = []

    def _spy(m):
        calls.append(len(m))
        return None

    handler = _CaptureLogs()
    lg = logging.getLogger("compactor")
    lg.addHandler(handler)
    try:
        with patch.object(main, "count_tokens_exact", _spy):
            out = main._enforce_hard_budget(msgs, limit)
    finally:
        lg.removeHandler(handler)
    return out, handler.records, len(calls)


def test_the_prescreen_measures_the_payloads_it_used_to_wave_through():
    """v3.1 A14. The divisor was 2, and the 2x margin's stated safety argument
    was a chars-per-LOCAL-token measurement — the oracle P0-0c discredited.

    Stated in the unit that actually charges: skipping is safe when
    chars/vLLM-token >= 4/D. D=2 bets that no payload ever prices below 2.0
    chars per vLLM token, and the worst assistant turn measured on 2026-08-28
    came in at 17,930/8,988 = 1.995 — through the break-even. D=8 puts the
    condition at 0.5 chars per vLLM token, which no text can price below, so
    the skip is safe on content rather than on a hope about content."""
    print("\n[test] A14 — a mid-band payload is now measured, not assumed")
    # HARD_INPUT_LIMIT is 800 here. limit//8 = 100, limit//2 = 400. Six turns
    # of 120 chars estimate at 6 * (30 + 4) = 204 — comfortably inside the band
    # the old divisor waved through and the new one measures.
    band = [big("user" if i % 2 == 0 else "assistant", 30) for i in range(6)]
    est = main._fast_token_estimate(band)
    assert_true(100 < est <= 400, f"the fixture is in the disputed band ({est})")
    _out, _records, calls = _guard_watching_tokenize(band, 800)
    assert_true(calls >= 1, f"the guard asked what it actually costs ({calls} calls)")

    print("\n[test] A14 — and the small payloads the prescreen exists for still skip")
    small = [big("user", 15)]
    assert_true(main._fast_token_estimate(small) <= 100, "fixture is under limit//8")
    _out, _records, calls = _guard_watching_tokenize(small, 800)
    assert_eq(calls, 0, "no /tokenize round trip for a payload 8x under the limit")

    print("\n[test] A14 — the boundary is limit//8 exactly")
    # Pinning the divisor rather than the behaviour: a future edit that moves it
    # back toward 2 should fail here and read the comment above.
    at = [{"role": "user", "content": "x" * 384}]        # 96 + 4 = 100
    over = [{"role": "user", "content": "x" * 388}]      # 97 + 4 = 101
    assert_eq(main._fast_token_estimate(at), 100, "fixture sits on the boundary")
    assert_eq(main._fast_token_estimate(over), 101, "and one token past it")
    assert_eq(_guard_watching_tokenize(at, 800)[2], 0, "at the boundary: skipped")
    assert_eq(_guard_watching_tokenize(over, 800)[2], 1, "one past it: measured")


# =====================================================================
# v3.1 A9 — which counter made the decision, said out loud.
# =====================================================================


def _records_containing(records, needle):
    return [r for r in records if needle in r.getMessage()]


def test_the_scale_line_names_the_counter_when_tokenize_answers():
    print("\n[test] A9 — the scale line names /tokenize as the source")
    msgs = [big("user" if i % 2 == 0 else "assistant", 30) for i in range(6)]
    handler = _CaptureLogs()
    lg = logging.getLogger("compactor")
    lg.addHandler(handler)
    try:
        with patch.object(main, "count_tokens_exact", lambda _m: 300):
            main._enforce_hard_budget(msgs, 800)
    finally:
        lg.removeHandler(handler)
    hits = _records_containing(handler.records, "token scale")
    assert_true(bool(hits), "the guard reported a scale at all")
    assert_true("counted by vLLM's /tokenize" in hits[0].getMessage(),
                f"and named the counter: {hits[0].getMessage()[:160]!r}")


def test_the_scale_line_speaks_when_tokenize_REFUSES():
    """The whole of A9 in one assertion.

    This line used to be gated on `exact is not None and |scale-1| > 0.05`,
    which made the only counter-naming line in the package unreachable in
    precisely the state it would diagnose: when /tokenize refuses, scale is
    forced to exactly 1.0, the shed runs on a tokenizer that reads 34-51% low,
    and the closing line is textually identical in shape to the healthy case —
    at HTTP 200. That is the 2026-08-28 signature exactly."""
    print("\n[test] A9 — /tokenize refusing is REPORTED, not inferred from silence")
    msgs = [big("user" if i % 2 == 0 else "assistant", 30) for i in range(6)]
    _out, records, _calls = _guard_watching_tokenize(msgs, 800)
    hits = _records_containing(records, "token scale unavailable")
    assert_true(bool(hits), "the degraded case says it is degraded")
    assert_eq(hits[0].levelno, logging.WARNING,
              "at WARNING — an INFO gated to zero is what this replaces")
    assert_true("UNCORRECTED" in hits[0].getMessage(),
                "and says the number below it is uncorrected")


def test_the_shed_line_names_the_counter_and_the_margin():
    print("\n[test] A9/A10 — the verdict line carries its provenance")
    # Every number in this line came from one of two counters that disagree by
    # up to 51% in the direction that overflows, and for a week of diagnosis it
    # said which one: never. `limit` is already NET of _BUDGET_MARGIN, so a
    # reader comparing it against HARD_INPUT_LIMIT could not see why they
    # differed either.
    margin_before = main._BUDGET_MARGIN
    try:
        main._BUDGET_MARGIN = 100
        msgs = [{"role": "system", "content": PERSONA}] + [
            big("user" if i % 2 == 0 else "assistant", 150) for i in range(8)
        ]
        _out, records, _calls = _guard_watching_tokenize(msgs, 800)
        line = _budget_line(records)
        assert_true(line is not None, "the guard reached its verdict line")
        msg = line.getMessage()
        assert_true("counted by the local tokenizer" in msg,
                    f"the line names the counter: {msg[:200]!r}")
        assert_true("margin 100" in msg,
                    f"and the margin that explains the limit: {msg[:200]!r}")
        assert_true("(limit 700" in msg,
                    f"which is the limit net of that margin: {msg[:200]!r}")
    finally:
        main._BUDGET_MARGIN = margin_before


# =====================================================================
# v3.1 A13 — a /tokenize outage that starts in hour six is still reportable.
# =====================================================================


def _tokenize_state():
    return (
        main._tokenize_fail_streak,
        main._tokenize_degraded_since,
        dict(main._tokenize_last_warn),
        main.TOKENIZE_WARN_INTERVAL_S,
    )


def _restore_tokenize_state(state):
    (main._tokenize_fail_streak, main._tokenize_degraded_since,
     last, main.TOKENIZE_WARN_INTERVAL_S) = state
    main._tokenize_last_warn.clear()
    main._tokenize_last_warn.update(last)


def _reset_tokenize_state():
    main._tokenize_fail_streak = 0
    main._tokenize_degraded_since = None
    main._tokenize_last_warn.clear()


def test_a_tokenize_outage_is_reportable_more_than_once_per_process():
    """v3.1 A13. This was logsetup.log_once("count_tokens_exact.http") — ONE
    line per process, over a set that is deliberately never cleared, shared by
    all four callers.

    The aggravator is that the likeliest first spender is BENIGN: a 400 here is
    usually the chat template refusing an assistant-final list, which the
    summarizer hands it on any conversation long enough to compact. A structural
    400 in minute two permanently silenced the report of a broken endpoint in
    hour six."""
    saved = _tokenize_state()
    try:
        print("\n[test] A13 — the same failure class is rate-limited, not spent")
        _reset_tokenize_state()
        main.TOKENIZE_WARN_INTERVAL_S = 3600.0
        handler = _CaptureLogs()
        lg = logging.getLogger("compactor")
        lg.addHandler(handler)
        try:
            main._note_tokenize_failure("http.400", "returned HTTP 400")
            main._note_tokenize_failure("http.400", "returned HTTP 400")
        finally:
            lg.removeHandler(handler)
        assert_eq(len(_records_containing(handler.records, "/tokenize degraded")), 1,
                  "two identical failures inside the window: one line")

        print("\n[test] A13 — a benign 400 cannot spend a transport failure's signal")
        # The key encodes the failure CLASS. Under log_once, one key covered
        # every call site and every status for the life of the process.
        handler = _CaptureLogs()
        lg.addHandler(handler)
        try:
            main._note_tokenize_failure("error.ConnectError", "unreachable")
        finally:
            lg.removeHandler(handler)
        assert_eq(len(_records_containing(handler.records, "/tokenize degraded")), 1,
                  "a different class still gets its line")

        print("\n[test] A13 — the outage is a readable FACT, not just a log line")
        h = main.tokenize_health()
        assert_eq(h["consecutive_failures"], 3, "three failures counted")
        assert_true(not h["ok"], "and the dependency reports itself as not ok")
        assert_true(h["degraded_since"] is not None, "with a start time")

        print("\n[test] A13 — recovery is announced, which a one-shot cannot do")
        handler = _CaptureLogs()
        lg.addHandler(handler)
        try:
            main._note_tokenize_success()
        finally:
            lg.removeHandler(handler)
        assert_eq(len(_records_containing(handler.records, "answering again")), 1,
                  "the recovery line exists")
        assert_true(main.tokenize_health()["ok"], "and the state is clean again")

        print("\n[test] A13 — a success while healthy says nothing at all")
        handler = _CaptureLogs()
        lg.addHandler(handler)
        try:
            main._note_tokenize_success()
        finally:
            lg.removeHandler(handler)
        assert_eq(len(handler.records), 0, "no line per healthy call")

        print("\n[test] A13 — the real call site uses it (an outage hours in still warns)")
        # Through count_tokens_exact, not the helper: the defect was at the CALL
        # SITE. Interval 0 stands in for "hours later".
        _reset_tokenize_state()
        main.TOKENIZE_WARN_INTERVAL_S = 0.0

        def _boom(*a, **kw):
            raise main.httpx.ConnectError("connection refused")

        handler = _CaptureLogs()
        lg.addHandler(handler)
        try:
            with patch.object(main.httpx, "post", _boom):
                first = main.count_tokens_exact([user("hi")])
                second = main.count_tokens_exact([user("hi")])
        finally:
            lg.removeHandler(handler)
        assert_eq((first, second), (None, None), "both calls fall back")
        assert_eq(len(_records_containing(handler.records, "/tokenize degraded")), 2,
                  "and both are reported — under log_once the second was silent")
    finally:
        _restore_tokenize_state(saved)


# =====================================================================
# v3.1 A10 — the release path needs a call site or it is dead code.
# =====================================================================


class _StubStreamAccepted:
    """A 200 SSE response, so the streaming handler runs its normal path."""

    status_code = 200

    async def aread(self):
        return b""

    async def aiter_raw(self):
        yield (
            b'data: {"choices":[{"index":0,"delta":{"content":"ok"},'
            b'"finish_reason":null}]}\n\n'
        )
        yield (
            b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
        )
        yield b"data: [DONE]\n\n"


class _StubVLLMAccepting(_StubVLLM):
    def stream(self, method, url, json=None, **kwargs):
        _StubVLLM.sent.append(json)
        return _StubStreamCM(_StubStreamAccepted())


def _post_accepted(messages, conv_id, *, stream):
    """One chat completion vLLM ACCEPTS, on either path. -> response."""
    _StubVLLM.sent.clear()
    with patch.object(main.httpx, "AsyncClient", _StubVLLMAccepting), \
         patch.object(main, "_fire_and_forget", _swallow_tail):
        return client.post(
            "/v1/chat/completions",
            json={"model": "stub-model", "messages": messages, "stream": stream},
            headers={"X-Conversation-Id": conv_id},
        )


def test_an_accepted_request_counts_toward_releasing_the_margin():
    """v3.1 A10, the call-site half.

    _note_backend_accepted can be perfect and still change nothing if no path
    calls it — and the rejection half has had a call site since v3.0.5 while
    the release half had none, which is exactly how the margin came to be
    monotonic. Both response paths are asserted because the streaming one is
    the path production takes."""
    saved = (main._BUDGET_MARGIN, main.BUDGET_MARGIN_RELEASE_AFTER,
             main._budget_ok_streak)
    try:
        for stream in (False, True):
            label = "streaming" if stream else "non-streaming"
            print(f"\n[test] A10 — an accepted {label} request releases the margin")
            main.BUDGET_MARGIN_RELEASE_AFTER = 2
            main._budget_ok_streak = 0
            main._BUDGET_MARGIN = 4000
            r = _post_accepted([user("hello")], f"a10-{label}", stream=stream)
            assert_eq(r.status_code, 200, f"{label}: the stub was served")
            assert_eq(main._BUDGET_MARGIN, 4000, f"{label}: one success is not two")
            _post_accepted([user("hello again")], f"a10-{label}", stream=stream)
            assert_eq(main._BUDGET_MARGIN, 2000, f"{label}: the second releases")
    finally:
        (main._BUDGET_MARGIN, main.BUDGET_MARGIN_RELEASE_AFTER,
         main._budget_ok_streak) = saved


def test_a_rejected_request_does_NOT_count_toward_release():
    print("\n[test] A10 — a 400 is not evidence the margin can be released")
    saved = (main._BUDGET_MARGIN, main.BUDGET_MARGIN_RELEASE_AFTER,
             main._budget_ok_streak)
    try:
        main.BUDGET_MARGIN_RELEASE_AFTER = 2
        main._budget_ok_streak = 0
        main._BUDGET_MARGIN = 4000
        # _post_rejected restores _BUDGET_MARGIN itself, so the assertion here
        # is about the STREAK: the counter that decides the next release.
        _post_rejected([user("hi")], "a10-rejected")
        _post_rejected([user("hi")], "a10-rejected")
        assert_eq(main._budget_ok_streak, 0, "two rejections banked nothing")
    finally:
        (main._BUDGET_MARGIN, main.BUDGET_MARGIN_RELEASE_AFTER,
         main._budget_ok_streak) = saved


# =====================================================================
# v3.1 D3 — injected memory is bounded as a FRACTION of the window, and the
# guard spends all of it before it will forward a payload it knows will 400.
#
# The hole these cover is not that any one layer is too big. Facts are capped
# at COMPACTOR_MAX_FACTS_TOKENS, retrieval at COMPACTOR_MAX_RETRIEVAL_TOKENS,
# each summary chunk at its own generation ceiling — and their SUM was capped
# nowhere, against a limit none of them could see. On 2026-08-28 a request with
# msgs=2, source=hash and lastturn=0 was handed 95 facts and 3 retrieval hits,
# could not be compacted ("no older turns to summarize"), could not be shed,
# and was rejected. No reply, no facts, no episodic write, no retry — for four
# hours.
# =====================================================================


def _blocks(*specs):
    """[(priority, label, text)] in SEND order, as the request path builds it.

    Text is `label` padded with the label's first letter, so a failure line
    says which layer survived without printing a wall of filler."""
    return [
        (prio, label, label + ":" + label[0] * max(0, chars - len(label) - 1))
        for prio, label, chars in specs
    ]


def test_the_injection_bound_drops_lowest_priority_first():
    print("\n[test] _bound_injected_blocks — retrieval goes before facts, facts before the summary")
    # Four layers, each individually within its own module's cap, summing to
    # roughly 4x the budget. Priority orders DROPPING; the survivors must come
    # back in SEND order, because inject_system_block's ordering contract
    # ("original system -> facts -> retrieved -> summary") is what the model
    # reads, and a bound that reorders the memory is a different bug.
    blocks = _blocks(
        (main._INJECT_PRIORITY_PERSONA, "persona", 400),
        (main._INJECT_PRIORITY_FACTS, "facts", 1200),
        (main._INJECT_PRIORITY_RETRIEVAL, "retrieval", 1200),
        (main._INJECT_PRIORITY_SUMMARY, "summary", 1200),
    )
    # char/4 tokenizer, and no /tokenize in this environment, so the cost is
    # the pessimistic 2x of the local count — the same worst-case undercount
    # the summarizer assumes when it cannot check itself.
    local = sum(main.count_tokens([{"role": "system", "content": t}]) for _, _, t in blocks)
    budget = int(local * main._PESSIMISTIC_SUMMARY_SCALE) // 2

    kept, dropped, cost = main._bound_injected_blocks(blocks, budget)

    assert_eq(dropped, ["retrieval", "facts"], "the two lowest priorities went first")
    assert_eq(
        [t.split(":")[0] for t in kept], ["persona", "summary"],
        "the survivors are the two highest priorities, still in SEND order — "
        "the summary was appended last and comes back second, because priority "
        "orders dropping and not sending",
    )
    assert_true(
        cost > budget,
        f"the reported cost is what BLEW the budget ({cost} > {budget}), not "
        f"what survived — '0 tokens against 32' tells an operator nothing",
    )


def test_the_injection_bound_never_drops_the_last_layer():
    print("\n[test] _bound_injected_blocks — a bound that rounds to zero is a refusal, not a bound")
    # A budget of zero. Every layer is over it, so a naive loop empties the
    # list — and "she remembers me from the first message" is the product, not
    # a nice-to-have. The floor is one layer, which is at most that layer's own
    # cap; the hard-budget guard is what takes it to nothing when the window
    # genuinely demands it (see the last-resort test below).
    blocks = _blocks(
        (main._INJECT_PRIORITY_PERSONA, "persona", 400),
        (main._INJECT_PRIORITY_FACTS, "facts", 1200),
        (main._INJECT_PRIORITY_RETRIEVAL, "retrieval", 1200),
    )
    kept, dropped, _cost = main._bound_injected_blocks(blocks, 0)
    assert_eq(len(kept), 1, "one layer survives a zero budget")
    assert_eq(kept[0].split(":")[0], "persona", "and it is the highest-priority one")
    assert_eq(dropped, ["retrieval", "facts"], "the rest are named as dropped")

    print("\n[test] _bound_injected_blocks — a sum that fits is not touched, and costs no /tokenize call")
    small = _blocks((main._INJECT_PRIORITY_FACTS, "facts", 40))
    calls = []

    def _counting_exact(msgs, *a, **k):
        calls.append(msgs)
        return None

    with patch.object(main, "count_tokens_exact", _counting_exact):
        kept, dropped, _cost = main._bound_injected_blocks(small, main.HARD_INPUT_LIMIT)
    assert_eq(len(kept), 1, "the block is injected")
    assert_eq(dropped, [], "nothing dropped")
    assert_eq(
        calls, [],
        "and nothing was measured: if the PESSIMISTIC local sum already fits, "
        "no measurement can change the outcome",
    )


def test_a_prior_assistant_turn_is_what_makes_a_request_a_conversation():
    print("\n[test] _has_conversational_history — the marker FRONTEND_SPEC §15 asks the client for")
    assert_eq(main._has_conversational_history([user("write a title")]), False,
              "a one-message background call has no history")
    assert_eq(
        main._has_conversational_history(
            [{"role": "system", "content": PERSONA}, user("hello")]
        ),
        False,
        "neither does a first turn carrying a system prompt",
    )
    assert_eq(
        main._has_conversational_history(
            [user("hi"), {"role": "assistant", "content": "hello"}, user("again")]
        ),
        True,
        "a prior assistant turn is the thing that distinguishes the two",
    )


def test_a_request_with_no_prior_assistant_turn_gets_the_narrow_budget():
    print("\n[test] POST /v1/chat/completions — task traffic does not receive the whole store")
    # End to end, because the fraction is chosen at the CALL SITE and nothing
    # that calls _bound_injected_blocks can see whether its caller picked the
    # right one. Same conversation, same stored memory, posted twice: once with
    # a prior assistant turn and once without.
    cid = "d3-no-history"
    persona.save_persona(cid, "PERSONA-STORE-SENTINEL " + "q" * 80, source="admin")
    facts.save_facts(
        cid,
        [
            {"text": FACT_SENTINEL + "-" + str(i) + " " + "f" * 120,
             "added_turn": 1, "last_used": 100}
            for i in range(3)
        ],
    )
    # No system message from the client: auto_capture must not overwrite the
    # stored persona, and text_to_inject only returns a persona the request is
    # not already carrying.
    with_history = [
        user("what did we decide?"),
        {"role": "assistant", "content": "we decided."},
        user("and now?"),
    ]
    no_history = [user("Generate a concise 3-5 word title for this chat")]

    _r, forwarded, _records = _post_chat(with_history, cid)
    assert_true(forwarded is not None, "the conversation request reached the stub")
    sys_text = _system_text(forwarded["messages"])
    assert_true("PERSONA-STORE-SENTINEL" in sys_text, "conversation: persona injected")
    assert_true(
        FACT_SENTINEL in sys_text,
        f"conversation: facts injected too — half the window is available to "
        f"memory here: {sys_text[:200]!r}",
    )

    _r, forwarded, records = _post_chat(no_history, cid)
    assert_true(forwarded is not None, "the task request reached the stub")
    sys_text = _system_text(forwarded["messages"])
    assert_true(
        FACT_SENTINEL not in sys_text,
        f"task traffic: the accumulated facts were NOT injected: {sys_text[:200]!r}",
    )
    assert_true(
        "PERSONA-STORE-SENTINEL" in sys_text,
        "but the bound is a bound, not a refusal — the top layer still went",
    )
    line = next(
        (r for r in records if "injected memory over budget" in r.getMessage()), None
    )
    assert_true(line is not None, "and the drop is reported, not silent")
    assert_eq(line.levelno, logging.WARNING, "at WARNING — this is memory the model will not see")
    assert_true(
        "NO prior assistant turn" in line.getMessage(),
        f"naming why the narrow budget applied: {line.getMessage()}",
    )


def test_the_guard_sheds_injection_to_nothing_before_forwarding_an_oversized_payload():
    print("\n[test] _enforce_hard_budget — the last resort spends every injected block")
    # The state the round loop can exit in: still over, with injected memory
    # still in hand. /tokenize is pinned one token above the limit, so every
    # verification round disagrees with the arithmetic and the six-round budget
    # runs out with blocks left. Before v3.1 D3 the guard forwarded there, on
    # the reasoning that the newest turn is never dropped — true of TURNS, and
    # not transferable to memory the model will never get to read. The 400 that
    # follows costs the user's typed message; dropping the memory costs
    # nothing the 400 was not already going to cost.
    saved_margin = main._BUDGET_MARGIN
    main._BUDGET_MARGIN = 0
    try:
        msgs = (
            [{"role": "system", "content": PERSONA}]
            + [injected(f"BLOCK{i}:") for i in range(10)]
            + [user("does any of this fit?")]
        )
        with patch.object(
            main, "count_tokens_exact", lambda _m: main.HARD_INPUT_LIMIT + 1
        ):
            out, records = _guard_with_logs(msgs)
        line = _budget_line(records)
        assert_true(line is not None, "the guard logged a verdict")
        assert_eq(line.levelno, logging.ERROR, "it could not fit, so it is an ERROR")
        assert_eq(
            _sys_dropped(line), 10,
            f"every injected block was spent, not just the ones the rounds "
            f"reached: {line.getMessage()}",
        )
        assert_true(
            _trimmed(line) < 10,
            f"and the rounds alone did not get there — {_trimmed(line)} "
            f"trim(s) in six rounds is what the old code stopped at",
        )
        assert_eq(len(_systems(out)), 1, "one system message survives: the caller's")
        assert_eq(out[0]["content"], PERSONA, "and it is untouched")
        assert_eq(out[-1]["content"], msgs[-1]["content"], "as is the newest turn")
        assert_true(
            "Nothing injected remains" in line.getMessage(),
            f"the line says WHAT is left, so nobody re-diagnoses this as a "
            f"memory problem: {line.getMessage()}",
        )
    finally:
        main._BUDGET_MARGIN = saved_margin


def test_the_guard_stops_as_soon_as_nothing_is_sheddable():
    print("\n[test] _enforce_hard_budget — six /tokenize calls over a payload it cannot change")
    # The old give-up test was `len(idxs) <= 1 and trimmed >= 32 and
    # len(sys_idxs) <= 1`. With protect_system=2 the last clause can never
    # hold, so an unfittable payload burned the full six rounds — six exact
    # measurements of a list that had not changed since the first.
    calls = []

    def _counting_exact(msgs, *a, **k):
        calls.append(len(msgs))
        return main.HARD_INPUT_LIMIT * 3

    msgs = [
        {"role": "system", "content": PERSONA},
        {"role": "system", "content": CALLER2},
        big("user", 2000),
    ]
    with patch.object(main, "count_tokens_exact", _counting_exact):
        _out, records = _guard_with_logs(msgs, None, 2)
    assert_eq(
        len(calls), 2,
        "one ground truth and one verification — nothing was sheddable after that",
    )
    line = _budget_line(records)
    assert_eq(line.levelno, logging.ERROR, "still reported as the failure it is")


def test_the_guard_reports_what_it_decided():
    print("\n[test] _enforce_hard_budget — the report dict, on all three exits")
    saved_margin = main._BUDGET_MARGIN
    main._BUDGET_MARGIN = 0
    try:
        report = {}
        main._enforce_hard_budget([user("hi")], main.HARD_INPUT_LIMIT, 1, report)
        assert_eq(report["fits"], True, "prescreen skip: fits")
        assert_true("prescreen" in report["counted_by"], "and says nothing was measured")

        report = {}
        main._enforce_hard_budget(
            [{"role": "system", "content": PERSONA}, big("user", 100)],
            main.HARD_INPUT_LIMIT, 1, report,
        )
        assert_eq(report["fits"], True, "measured and under the limit: fits")

        report = {}
        main._enforce_hard_budget(
            [{"role": "system", "content": PERSONA}, big("user", 2000)],
            main.HARD_INPUT_LIMIT, 1, report,
        )
        assert_eq(report["fits"], False, "measured and over the limit: does NOT fit")
        assert_true(
            report["measured"] > report["limit"],
            "with the numbers that say so, not just the verdict",
        )
    finally:
        main._BUDGET_MARGIN = saved_margin


# =====================================================================
# v3.1 D4 — _BUDGET_MARGIN is a module global, so what it is allowed to LEARN
# is what decides its blast radius.
#
# 2026-08-28: one conversation sent two messages whose own content did not fit
# the window. The guard shed everything it was permitted to shed, logged
# "hard budget FAILED to fit ... still 16417 over", forwarded, and took the 400
# it had just predicted. 16,384 + 16,417 = 32,801, which is exactly what vLLM
# reported — so the calibration "learned" the number the guard had already
# measured, and latched the margin to its MAX_MODEL_LEN//4 ceiling of 8,192 for
# EVERY conversation in the process. Four hours later a real conversation was
# running at "limit 8192, margin 8192" and shedding on every request.
# =====================================================================


def test_a_400_the_guard_predicted_does_not_widen_the_margin():
    print("\n[test] _note_backend_rejection — a predicted rejection teaches nothing")
    saved = (main._BUDGET_MARGIN, main._budget_ok_streak)
    try:
        main._BUDGET_MARGIN = 0
        handler = _CaptureLogs()
        lg = logging.getLogger("compactor")
        lg.addHandler(handler)
        try:
            tightened = main._note_backend_rejection(
                ctx_400(2500), 800, guard_measured_overflow=True
            )
        finally:
            lg.removeHandler(handler)
        assert_eq(tightened, False, "nothing was learned")
        assert_eq(main._BUDGET_MARGIN, 0, "and the margin did not move")
        line = next(
            (r for r in handler.records
             if "NOT widening the budget margin" in r.getMessage()),
            None,
        )
        assert_true(line is not None, "the refusal is reported, not silent")
        assert_true(
            not any("Tightening the hard limit" in r.getMessage()
                    for r in handler.records),
            "and it did not also claim to have tightened",
        )
    finally:
        (main._BUDGET_MARGIN, main._budget_ok_streak) = saved


def test_a_400_that_surprised_the_guard_still_widens_the_margin():
    print("\n[test] _note_backend_rejection — the counterfactual: a surprise still calibrates")
    # The same body and the same limit. The ONLY difference is whether the
    # guard had already measured this payload as over. Without this assertion
    # the test above passes just as well against a margin that never moves at
    # all, which would delete the /tokenize-outage backstop entirely.
    saved = (main._BUDGET_MARGIN, main._budget_ok_streak)
    try:
        main._BUDGET_MARGIN = 0
        tightened = main._note_backend_rejection(
            ctx_400(2500), 800, guard_measured_overflow=False
        )
        assert_eq(tightened, True, "an unexpected 400 is evidence and is used")
        assert_true(main._BUDGET_MARGIN > 0, "the margin advanced")
    finally:
        (main._BUDGET_MARGIN, main._budget_ok_streak) = saved


def test_one_unfittable_conversation_does_not_narrow_the_window_for_the_others():
    print("\n[test] POST /v1/chat/completions — the D4 property, end to end")
    # The call-site half. _note_backend_rejection can be perfect and still
    # change nothing if the endpoint never tells it what the guard decided —
    # and the guard's verdict is computed ~200 lines from the rejection path,
    # which is exactly the distance A8's defect survived at.
    saved = (main._BUDGET_MARGIN, main._budget_ok_streak)
    try:
        main._BUDGET_MARGIN = 0
        msgs = [{"role": "system", "content": PERSONA}, big("user", 2000)]
        _r, records, _tails = _post_rejected(
            msgs, "d4-unfittable", body=ctx_400(2500)
        )
        assert_true(
            any("hard budget FAILED to fit" in r.getMessage() for r in records),
            "precondition: the guard knew this payload did not fit",
        )
        assert_true(
            any("NOT widening the budget margin" in r.getMessage() for r in records),
            "so the rejection that followed was not treated as new information",
        )
        assert_true(
            not any("Tightening the hard limit" in r.getMessage() for r in records),
            "and the process-wide margin was left alone for everyone else",
        )
    finally:
        (main._BUDGET_MARGIN, main._budget_ok_streak) = saved


def _all_tests():
    return [
        test_hard_limit_configured,
        test_under_budget_is_untouched,
        test_the_production_scenario,
        test_per_request_limit,
        test_tokenization_cost_is_bounded,
        test_split_messages_keeps_user_first,
        test_pathological_single_huge_turn,
        test_injected_blocks_are_droppable,
        test_nothing_dropped_when_already_under_budget,
        test_alternation_survives_the_dropping_stage,
        test_give_up_waits_for_the_last_droppable_block,
        test_unfittable_payload_reports_error_not_success,
        test_success_path_still_logs_warning,
        test_regression_callers_second_system_message_is_never_dropped,
        test_trim_loop_will_not_halve_a_protected_persona,
        test_protect_system_defaults_to_one,
        test_zero_caller_system_messages_still_protects_index_zero,
        test_injected_blocks_remain_fully_spendable,
        test_protected_unfittable_payload_still_forwards_at_error,
        test_alternation_repair_survives_protected_shedding,
        test_endpoint_forwards_both_caller_system_messages,
        test_call_site_passes_the_callers_system_count,
        test_tier2_fallback_is_not_silent,
        test_chunk_to_budget,
        test_context_400_tells_the_user_the_message_did_not_go_through,
        test_context_400_is_typed_so_a_client_can_tell_it_from_a_reply,
        test_context_400_logs_an_error_naming_the_conversation_and_both_counts,
        test_the_log_line_names_tokenize_when_tokenize_answered,
        test_a_rejected_turn_is_never_memorized,
        test_retry_is_promised_only_when_the_rejection_taught_us_something,
        test_a_non_size_400_does_not_blame_the_context_window,
        test_a_backend_5xx_on_the_stream_is_also_logged_as_a_lost_turn,
        test_nonstream_400_logs_the_loss_and_skips_the_memory_tail,
        test_extraction_is_handed_the_injected_subset_not_the_whole_store,
        test_extraction_is_bounded_even_for_a_caller_that_passes_nothing,
        test_extraction_and_dedup_are_told_which_conversation,
        test_the_request_path_hands_the_tail_the_list_it_injected,
        test_the_prescreen_measures_the_payloads_it_used_to_wave_through,
        test_the_scale_line_names_the_counter_when_tokenize_answers,
        test_the_scale_line_speaks_when_tokenize_REFUSES,
        test_the_shed_line_names_the_counter_and_the_margin,
        test_a_tokenize_outage_is_reportable_more_than_once_per_process,
        test_an_accepted_request_counts_toward_releasing_the_margin,
        test_a_rejected_request_does_NOT_count_toward_release,
        test_the_injection_bound_drops_lowest_priority_first,
        test_the_injection_bound_never_drops_the_last_layer,
        test_a_prior_assistant_turn_is_what_makes_a_request_a_conversation,
        test_a_request_with_no_prior_assistant_turn_gets_the_narrow_budget,
        test_the_guard_sheds_injection_to_nothing_before_forwarding_an_oversized_payload,
        test_the_guard_stops_as_soon_as_nothing_is_sheddable,
        test_the_guard_reports_what_it_decided,
        test_a_400_the_guard_predicted_does_not_widen_the_margin,
        test_a_400_that_surprised_the_guard_still_widens_the_margin,
        test_one_unfittable_conversation_does_not_narrow_the_window_for_the_others,
    ]


if __name__ == "__main__":
    try:
        for t in _all_tests():
            t()
        print("\nAll budget-guard tests passed.")
    finally:
        shutil.rmtree(_TMP_ROOT, ignore_errors=True)
