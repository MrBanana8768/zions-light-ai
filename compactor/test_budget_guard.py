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

The last section drives the guard through the real endpoint rather than calling
it directly, because the protect_system fix is load-bearing on its CALL SITE and
nothing that calls the function can see whether the call site is right.

Run inside the compactor image or any container with the requirements:
    python test_budget_guard.py
"""

import inspect
import io
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

import facts  # noqa: E402
import logsetup  # noqa: E402
import main  # noqa: E402
import memory  # noqa: E402
import retrieval  # noqa: E402

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


def _swallow_tail(coro):
    """Stand-in for main._fire_and_forget.

    The post-response memory tail extracts facts, embeds and rolls up
    summaries — none of which this file is testing, all of which would write
    to the scratch volume and reach for vLLM. Closing the coroutine also keeps
    Python from warning that it was never awaited."""
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

    def recorder(messages, limit=None, protect_system=None):
        seen["messages"] = list(messages)
        seen["limit"] = limit
        seen["protect_system"] = protect_system
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
    ]


if __name__ == "__main__":
    try:
        for t in _all_tests():
            t()
        print("\nAll budget-guard tests passed.")
    finally:
        shutil.rmtree(_TMP_ROOT, ignore_errors=True)
