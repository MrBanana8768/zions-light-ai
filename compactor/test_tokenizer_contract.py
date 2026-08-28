"""Tier-1.5 tests: the compactor's budget arithmetic, measured against a REAL
tokenizer served over vLLM's /tokenize contract.

READ THIS BEFORE YOU TRUST A GREEN RUN
======================================

What this suite proves
----------------------
That the compactor ASKS the server what a payload costs instead of trusting its
own estimate; that it reads the answer out of the field vLLM actually puts it
in; that the scale-correction arithmetic in `_chunk_to_budget` and
`_enforce_hard_budget` lands where it claims to when a real tokenizer is the
judge; and that when /tokenize lies, hangs, errors or disappears, the compactor
degrades in a way that is visible and still serves the user.

What this suite does NOT prove
------------------------------
* **Nothing about the production numbers.** The fixture's tokenizer is
  HuggingFaceTB/SmolLM2-135M-Instruct — a 49k byte-level BPE. Cydonia-24B is
  tokenized by mistral_common over tekken.json. The counts differ. A payload
  that fits here may not fit on the pod, and vice versa. Do not quote a number
  from this suite as a production budget figure.
* **Nothing about GENERATION_RESERVE.** Whether 16384 is the right reserve is a
  property of how long THIS model's replies run (measured 7,513-11,347 tokens
  on the production conversation). No fixture can tell you that.
* **Nothing about vLLM itself.** The fixture reimplements two endpoints; it is
  not vLLM. Its shapes were dumped from vllm==0.10.0's own protocol module, but
  the deployed stack pins vllm==0.24.0 (Dockerfile:78). See the README.

Why it exists
-------------
`compactor/test_smoke.py` says, in its own opening docstring, that it runs with
"no vLLM, no GPU, no real tokenizer" and that "count_tokens falls back to the
char/4 estimator". Every budget test in this repo has inherited that. So the
suite has been asserting the budget arithmetic against the estimator that was
wrong, which is why a 23% payload / 34-51% assistant-content undercount reached
production twice (INCIDENT_2026-08-24.md, INCIDENT_2026-08-28.md) with a full
green board.

Several tests below are TEETH CHECKS: they re-run the pre-fix behaviour and
assert that it OVERFLOWS when measured by the server. A suite that only asserts
the fixed path passes just as happily on code that never had the fix.

WHAT THIS SUITE MISSED, AND WHY (D7, added 2026-08-29)
------------------------------------------------------
It was built to catch /tokenize contract failures. It ran green twice, shipped,
and four hours later D1 — /tokenize 400s on an assistant-final message list —
took compaction down in production.

It missed it because every request it made ended on a USER turn:

* `conversation()` was the only generator, and it ends on the new user turn.
* `truth()` sent `add_generation_prompt` unset, so the server defaulted it True
  and the fixture cheerfully tokenized assistant-final lists that real vLLM
  refuses. Section 3 measured summarizer batches through a request production
  could not make.
* the fixture had no shape validation at all, so there was nothing to hit.

Three places agreed on a contract nothing in the tree sends. Sections 8-10 close
that: 8 covers the assistant-final shape end to end, 9 covers the FALLBACK path
rather than only the measured one (both of the 2026-08-29 defects lived in
fallbacks), and 10 enumerates every module in the tree that POSTs to /tokenize
and drives each shape it sends — failing if a third client appears.

STILL WEAKER THAN IT LOOKS
--------------------------
Stated plainly rather than left for the next incident to find:

* The scale-correction assertions are only as hostile as the tokenizer PAIR.
  Test [5] says so for content categories; it is equally true of [6], [7], [21]
  and [22], whose thresholds all rest on gpt2-vs-SmolLM2 diverging.
* Nothing here exercises `format_retrieval_block`'s `count_tokens` hook, because
  no caller passes one — the retrieval budget runs on `_estimate_tokens` in
  production and this suite says nothing about it.
* `_reported_prompt_tokens` is checked against the fixture's wording, and the
  fixture's wording was dumped from vllm 0.10.0 while the pod pins 0.24.0. See
  [16] and the README's finding.
* The tests assert on LOG SUBSTRINGS in several places. A log line whose numbers
  go stale still passes those (see the note [21] can emit).

How to run
----------
    docker compose -f docker-compose.tokenizer-contract.yml \
        up --build --exit-code-from contract-tests

Or, against a fixture you started yourself:

    docker compose -f docker-compose.tokenizer-contract.yml up -d tokenize-fixture
    VLLM_URL=http://localhost:18000 FIXTURE_URL=http://localhost:18000 \
        python compactor/test_tokenizer_contract.py

With no fixture reachable this module SKIPS and exits 0, so the existing
CPU-only suite is unaffected. It is deliberately not silent about skipping.
"""

import asyncio
import os
import pathlib
import re
import sys
import tempfile

# --- environment must be set before `import main` ------------------------
# main.py reads MAX_MODEL_LEN / VLLM_URL / MODEL_REPO at import time.
_TMP_STATE = tempfile.mkdtemp(prefix="contract-state-")
os.environ.setdefault("COMPACTOR_STORAGE_ROOT", _TMP_STATE)
os.environ.setdefault("VLLM_URL", "http://localhost:18000")
os.environ.setdefault("MODEL_REPO", "fixture-model")
os.environ.setdefault("MAX_MODEL_LEN", "32768")
os.environ.setdefault("COMPACTOR_GENERATION_RESERVE", "16384")

FIXTURE_URL = os.environ.get("FIXTURE_URL", os.environ["VLLM_URL"]).rstrip("/")
LOCAL_TOKENIZER = os.environ.get("CONTRACT_LOCAL_TOKENIZER", "gpt2").strip()

import httpx  # noqa: E402


def _skip(reason: str) -> None:
    print("=" * 72)
    print("SKIPPED: test_tokenizer_contract.py")
    print(f"  reason: {reason}")
    print()
    print("  This suite needs the vLLM-shaped tokenizer fixture. Start it with:")
    print("    docker compose -f docker-compose.tokenizer-contract.yml \\")
    print("        up --build --exit-code-from contract-tests")
    print()
    print("  Skipping it means the budget code is currently only covered by")
    print("  char/4 assertions — the estimator that took production down on")
    print("  2026-08-24 and 2026-08-28. This is not a neutral skip.")
    print("=" * 72)
    sys.exit(0)


try:
    _info = httpx.get(f"{FIXTURE_URL}/_fixture/info", timeout=3.0)
    _info.raise_for_status()
    FIXTURE_INFO = _info.json()
except Exception as e:  # noqa: BLE001
    _skip(f"{FIXTURE_URL}/_fixture/info unreachable ({type(e).__name__}: {e})")

import logsetup  # noqa: E402
import main  # noqa: E402

FIXTURE_MAX_LEN = int(FIXTURE_INFO["max_model_len"])

# A STALE FIXTURE IMAGE IS NOT A SKIP.
#
# The test runner mounts the repo, so editing a test takes effect immediately;
# the fixture is a BUILT IMAGE, so editing the server does not. That asymmetry
# is invisible from the test side, and it is how a harness ends up green against
# a server that cannot produce the failure it is checking for — the D1 group
# below asserts that an assistant-final /tokenize request is REFUSED, and a
# fixture built before 2026-08-29 answers it 200. Fail loudly rather than pass
# vacuously.
_REQUIRED_FIXTURE_FEATURES = {"assistant_final_400", "shapes"}
_missing = _REQUIRED_FIXTURE_FEATURES - set(FIXTURE_INFO.get("features") or [])
if _missing:
    print("=" * 72)
    print("STALE FIXTURE IMAGE — refusing to run")
    print(f"  {FIXTURE_URL} does not advertise: {', '.join(sorted(_missing))}")
    print()
    print("  This suite asserts that /tokenize REFUSES an assistant-final")
    print("  message list (D1, the defect that took compaction down four hours")
    print("  after the v3.1 deploy). A fixture without that check answers 200,")
    print("  so those assertions would pass while proving nothing — which is")
    print("  precisely the failure D7 is about. Rebuild it:")
    print()
    print("    docker compose -f docker-compose.tokenizer-contract.yml \\")
    print("        up --build --exit-code-from contract-tests")
    print("=" * 72)
    sys.exit(1)

_FAILURES: list[str] = []
_NOTES: list[str] = []


# ------------------------------------------------------------------ helpers


def ok(label: str, extra: str = "") -> None:
    print(f"  ok   {label}{(' — ' + extra) if extra else ''}")


def fail(label: str, detail: str) -> None:
    _FAILURES.append(f"{label}: {detail}")
    print(f"  FAIL {label} — {detail}")


def check(cond: bool, label: str, detail: str = "", extra: str = "") -> None:
    if cond:
        ok(label, extra)
    else:
        fail(label, detail or "condition was false")


def note(text: str) -> None:
    _NOTES.append(text)
    print(f"  NOTE {text}")


def set_local_tokenizer(name: str | None) -> str:
    """Choose which tokenizer `main.count_tokens` uses locally.

    Deliberately distinct from the fixture server's. On 2026-08-28 the two
    sides of this comparison were transformers-converted-tekken (local) and
    mistral_common (vLLM); here they are gpt2 (local) and SmolLM2 (server).
    Different vocabularies, same failure mode.
    """
    if not name:
        main._tokenizer = None
        main.MODEL_REPO = None
        return "char/4 estimator (no tokenizer)"
    from transformers import AutoTokenizer

    main._tokenizer = AutoTokenizer.from_pretrained(name)
    main.MODEL_REPO = os.environ.get("MODEL_REPO", "fixture-model")
    tier = "chat-template" if main._tokenizer.chat_template else "encode()+4 (tier 2)"
    return f"{name} via {tier}"


def fixture(path: str, payload: dict | None = None) -> dict:
    if payload is None:
        r = httpx.get(f"{FIXTURE_URL}{path}", timeout=10.0)
    else:
        r = httpx.post(f"{FIXTURE_URL}{path}", json=payload, timeout=30.0)
    r.raise_for_status()
    return r.json()


def set_mode(**kw) -> None:
    fixture("/_fixture/mode", kw)


def reset_fixture() -> None:
    fixture("/_fixture/reset", {})
    logsetup._reset_log_once_for_tests()


def truth(messages: list[dict]) -> int:
    """The server's count, taken directly — never through main, so a test can
    still measure ground truth while main is being lied to.

    `add_generation_prompt` is decided here the same way `count_tokens_exact`
    decides it, and that is not a convenience. Until 2026-08-29 this helper sent
    the field unset, the server defaulted it True, and the fixture happily
    tokenized an assistant-final list that real vLLM refuses with a 400. Every
    "measured on the server" assertion in section 3 runs on batches that
    routinely end on an assistant turn, so the harness was measuring those
    batches through a request production could not make. That is D7 in one
    helper: the contract it imagined, not the one the code exercises.
    """
    agp = bool(messages) and messages[-1].get("role") != "assistant"
    r = httpx.post(
        f"{FIXTURE_URL}/tokenize",
        json={
            "model": "fixture-model",
            "messages": messages,
            "add_generation_prompt": agp,
        },
        timeout=60.0,
    )
    r.raise_for_status()
    return int(r.json()["count"])


def truth_prompt(text: str) -> int:
    """The server's count for the COMPLETION shape — `{"model", "prompt"}`.

    A different request than `truth`: no chat template, no generation prompt,
    `add_special_tokens` defaulting True rather than False. This is the shape
    `summarizer._count_tokens` sends, and nothing in this suite measured it
    before 2026-08-29.
    """
    r = httpx.post(
        f"{FIXTURE_URL}/tokenize",
        json={"model": "fixture-model", "prompt": text},
        timeout=60.0,
    )
    r.raise_for_status()
    return int(r.json()["count"])


def shapes() -> list[dict]:
    """Every /tokenize request the fixture has seen since the last reset."""
    return fixture("/_fixture/shapes").get("shapes") or []


class CaptureLogs:
    """Collect records from every compactor logger for the duration of a block."""

    def __init__(self) -> None:
        self.records: list = []

    def __enter__(self):
        import logging

        outer = self

        class _H(logging.Handler):
            def emit(self, record):
                outer.records.append(record)

        self._h = _H()
        self._root = logging.getLogger("compactor")
        self._root.addHandler(self._h)
        self._prev_level = self._root.level
        self._root.setLevel(logging.DEBUG)
        # main's own logger may not be under "compactor." — attach there too.
        self._mainlog = main.logger
        self._mainlog.addHandler(self._h)
        return self

    def __exit__(self, *a):
        self._root.removeHandler(self._h)
        self._root.setLevel(self._prev_level)
        self._mainlog.removeHandler(self._h)
        return False

    def text(self) -> str:
        return "\n".join(r.getMessage() for r in self.records)

    def at_least(self, level: str) -> str:
        import logging

        want = getattr(logging, level)
        return "\n".join(r.getMessage() for r in self.records if r.levelno >= want)


# ------------------------------------------------------- content generators
#
# These reproduce the SHAPE of the production content, not its text. Prose is
# where tokenizers agree; a test written on prose passes and proves nothing.

# INCIDENT_2026-08-28: one assistant reply carried 1,710 U+2501 and 441 U+2500.
DECORATIVE_REPLY = (
    "━" * 1710
    + "\n"
    + "─" * 441
    + "\nHere is the summary you asked for, rendered as a table.\n"
)

EMOJI_BLOCK = "".join("\U0001f9ed\U0001f6e0\U0001f4ca✨\U0001f525" for _ in range(400))

CJK_BLOCK = "".join("神の光を見てください。" for _ in range(400))

PROSE_BLOCK = (
    "The quick brown fox jumps over the lazy dog, and the dog, being lazy, "
    "does not object to this arrangement in the slightest degree. "
) * 40


def decorative_turn(role: str = "assistant") -> dict:
    return {"role": role, "content": DECORATIVE_REPLY}


def conversation(n_pairs: int, assistant_content: str = DECORATIVE_REPLY) -> list[dict]:
    """The GUARD's shape: a full payload ending on the user's new turn."""
    msgs: list[dict] = [{"role": "system", "content": "You are a helpful assistant."}]
    for i in range(n_pairs):
        msgs.append({"role": "user", "content": f"Question {i}: tell me about the plan."})
        msgs.append({"role": "assistant", "content": assistant_content})
    msgs.append({"role": "user", "content": "And what should I do next?"})
    return msgs


def old_turns(n_pairs: int, assistant_content: str = DECORATIVE_REPLY) -> list[dict]:
    """The SUMMARIZER's shape: a slice of completed exchanges, so it ends on an
    assistant reply and carries no system message.

    This is what `split_messages` hands `summarize()` — `to_summarize` is
    `non_system[:-KEEP_RECENT_TURNS]`, sliced at a user boundary, so the turn
    before that boundary is the assistant's. Every summarize() in the production
    logs of 2026-08-28 and 2026-08-29 measured a list of exactly this shape, and
    every one of them got a 400 for it.
    """
    msgs: list[dict] = []
    for i in range(n_pairs):
        msgs.append({"role": "user", "content": f"Question {i}: tell me about the plan."})
        msgs.append({"role": "assistant", "content": assistant_content})
    assert msgs[-1]["role"] == "assistant"
    return msgs


# ================================================================= 1. CONTRACT


def test_tokenize_answers_the_shape_the_compactor_reads():
    print("\n[1] /tokenize speaks the contract main.count_tokens_exact depends on")
    reset_fixture()
    msgs = [{"role": "user", "content": "hello"}]
    raw = httpx.post(
        f"{FIXTURE_URL}/tokenize", json={"model": "fixture-model", "messages": msgs}, timeout=30.0
    ).json()
    for field in ("count", "max_model_len", "tokens"):
        check(field in raw, f"response carries `{field}` (vLLM TokenizeResponse)")
    n = main.count_tokens_exact(msgs)
    check(
        isinstance(n, int) and n > 0,
        "count_tokens_exact returns the server's integer count",
        f"got {n!r}",
        extra=f"{n} tokens",
    )
    check(n == raw["count"], "it reads .count, not len(.tokens) or anything else")
    check(
        main.count_tokens_exact([]) == 0,
        "empty message list short-circuits to 0 without a round trip",
    )


def test_tokenize_does_not_context_check():
    print("\n[2] /tokenize answers for payloads far over the window (vLLM does too)")
    reset_fixture()
    huge = [{"role": "user", "content": DECORATIVE_REPLY * 20}]
    n = main.count_tokens_exact(huge)
    check(
        isinstance(n, int) and n > FIXTURE_MAX_LEN,
        "a payload over max_model_len still gets a count, not a 400",
        f"got {n!r}",
        extra=f"{n} tokens vs window {FIXTURE_MAX_LEN}",
    )
    # serving_engine.py:606-609 returns early for TokenizeChatRequest — the
    # guard depends on this: it must be able to measure something too big to
    # send, or it can never decide how much to shed.


# ============================================== 2. WHERE THE ESTIMATOR LIES


def _ratio(msgs: list[dict]) -> tuple[int, int, float]:
    local = main.count_tokens(msgs)
    exact = truth(msgs)
    return local, exact, (exact / local if local else float("inf"))


def test_prose_is_where_the_two_agree():
    print("\n[3] CONTROL: on prose the local count is roughly right")
    reset_fixture()
    local, exact, r = _ratio([{"role": "assistant", "content": PROSE_BLOCK}])
    check(
        0.6 <= r <= 1.8,
        "prose ratio is unremarkable",
        f"ratio {r:.2f}x (local {local}, server {exact})",
        extra=f"local {local} vs server {exact} = {r:.2f}x",
    )
    note(
        "This is the control. It is why a budget test written on prose passes "
        "while production burns: the two tokenizers agree on prose."
    )


def test_box_drawing_is_where_the_estimator_lies():
    print("\n[4] THE 2026-08-28 BUG: box-drawing content, measured both ways")
    reset_fixture()
    msgs = [decorative_turn()]
    local, exact, r = _ratio(msgs)
    print(
        f"       1710x U+2501 + 441x U+2500  ->  local {local}  server {exact}  "
        f"scale {r:.2f}x"
    )
    check(
        r >= 1.5,
        "the local count is materially wrong on decorative rules",
        f"ratio only {r:.2f}x — the fixture is not reproducing the divergence",
        extra=f"{r:.2f}x undercount",
    )
    # And the consequence, stated as the test that would have caught it:
    # pick a limit the local count clears and the server does not.
    limit = (local + exact) // 2
    check(
        local <= limit < exact,
        "there is a limit the estimator clears and the server rejects",
        f"local {local}, exact {exact}, limit {limit}",
        extra=f"limit {limit}: estimator says FITS, server says OVER by {exact - limit}",
    )
    note(
        "Any assertion written against count_tokens alone would call this "
        "payload safe. That is the whole shape of the outage."
    )


def test_the_hostile_set_reproduces_the_divergence():
    """Report every hostile category; fail only if NONE of them diverge.

    The per-category ratio is a property of the FIXTURE PAIR (gpt2 locally vs
    SmolLM2 on the server), not of the compactor, so a per-category threshold
    would be asserting something this suite has no business asserting.
    Measured 2026-08-28 with the default pair:

        box-drawing  1.97x     <- reproduces the production class
        CJK          1.47x     <- reproduces it
        emoji        1.00x     <- does NOT. Both sides are byte-level BPEs
                                  and price these code points alike.

    So: emoji divergence is attested by the production measurement in
    INCIDENT_2026-08-28.md and NOT by this suite. Do not read a green run as
    cover for emoji-heavy content. Swapping either tokenizer changes these
    numbers; the assertion below is only that the pair can still reproduce the
    bug class at all.
    """
    print("\n[5] the hostile content set: which categories reproduce the divergence")
    reset_fixture()
    ratios: dict[str, tuple[int, int, float]] = {}
    for label, block in (
        ("box-drawing", DECORATIVE_REPLY),
        ("emoji", EMOJI_BLOCK),
        ("CJK", CJK_BLOCK),
        ("prose (control)", PROSE_BLOCK),
    ):
        local, exact, r = _ratio([{"role": "assistant", "content": block}])
        ratios[label] = (local, exact, r)
        print(f"       {label:<18} local {local:>6}   server {exact:>6}   {r:.2f}x")

    hostile = {k: v for k, v in ratios.items() if "control" not in k}
    worst = max(r for _, _, r in hostile.values())
    check(
        worst >= 1.5,
        "at least one hostile category still reproduces the undercount",
        f"best ratio was only {worst:.2f}x — this tokenizer pair can no longer "
        f"reproduce the 2026-08-28 bug class, so the suite below is not "
        f"testing what it claims. Change FIXTURE_TOKENIZER or "
        f"CONTRACT_LOCAL_TOKENIZER until it can.",
        extra=f"worst-case {worst:.2f}x",
    )
    flat = [k for k, (_, _, r) in hostile.items() if r < 1.2]
    if flat:
        note(
            f"these hostile categories did NOT diverge with this tokenizer "
            f"pair: {', '.join(flat)}. They are covered by the production "
            f"measurements in INCIDENT_2026-08-28.md, not by this suite."
        )


# ================================================== 3. SUMMARIZER BATCHES FIT


def _summarizer_budget() -> int:
    return min(
        main.MAX_MODEL_LEN,
        max(256, main.MAX_MODEL_LEN - main.SUMMARY_MAX_TOKENS - main.SUMMARY_INPUT_RESERVE),
    )


def test_summarizer_batches_actually_fit_the_server():
    print("\n[6] every batch _chunk_to_budget packs FITS, measured on the server")
    reset_fixture()
    budget = _summarizer_budget()
    turns = conversation(14)[1:]  # drop the system message: summarize() gets turns
    local = main.count_tokens(turns)
    exact = truth(turns)
    scale = exact / local if local else 1.0
    batches = main._chunk_to_budget(turns, budget, scale)
    print(f"       budget {budget}, {len(turns)} turns, scale {scale:.2f}x -> {len(batches)} batches")
    worst = 0
    for i, b in enumerate(batches):
        measured = truth(b)
        worst = max(worst, measured)
        if measured > budget:
            fail(
                "every batch fits",
                f"batch {i} measures {measured} > budget {budget}",
            )
            break
    else:
        ok("every batch fits", f"largest batch {worst} <= budget {budget}")


def test_unscaled_batching_would_have_overflowed():
    print("\n[7] TEETH: the same batching with scale=1.0 overflows the server")
    reset_fixture()
    budget = _summarizer_budget()
    turns = conversation(14)[1:]
    batches = main._chunk_to_budget(turns, budget, 1.0)  # the pre-v3.1 call
    over = [(i, truth(b)) for i, b in enumerate(batches)]
    biggest = max(n for _, n in over)
    check(
        biggest > budget,
        "unscaled batches exceed the budget when a real tokenizer measures them",
        f"largest unscaled batch was {biggest} <= budget {budget}; the fixture "
        f"is not hostile enough for this teeth check to mean anything",
        extra=f"largest unscaled batch {biggest} vs budget {budget} "
        f"(+{biggest - budget})",
    )
    note(
        "This is the assertion the CPU-only suite structurally cannot make: it "
        "has no second opinion to measure the batch against."
    )


# ====================================================== 4. THE GUARD'S RESULT


def test_guard_result_measures_under_the_limit_on_the_server():
    print("\n[8] _enforce_hard_budget's OUTPUT fits, measured on the server")
    reset_fixture()
    limit = 8000
    msgs = conversation(12)
    before = truth(msgs)
    with CaptureLogs() as logs:
        out = main._enforce_hard_budget(msgs, limit=limit, protect_system=1)
    after = truth(out)
    print(f"       {before} -> {after} tokens (limit {limit}), {len(msgs)} -> {len(out)} messages")
    check(
        after <= limit,
        "the shed payload genuinely fits",
        f"guard returned {after} tokens against a limit of {limit}",
        extra=f"{before} -> {after} <= {limit}",
    )
    check(
        out[-1] is msgs[-1] or out[-1] == msgs[-1],
        "the newest turn was never dropped",
    )
    check(
        "hard budget" in logs.text(),
        "the guard said what it did",
        f"no hard-budget line in: {logs.text()[:300]!r}",
    )


def test_guard_without_tokenize_lands_over_the_limit():
    print("\n[9] TEETH: the pre-v3.1 guard (local count only) still overflows")
    reset_fixture()
    limit = 8000
    msgs = conversation(12)
    saved = main.count_tokens_exact
    try:
        main.count_tokens_exact = lambda m: None  # the world before P0-0c
        out = main._enforce_hard_budget(msgs, limit=limit, protect_system=1)
    finally:
        main.count_tokens_exact = saved
    after = truth(out)
    print(f"       local-only guard returned {len(out)} messages measuring {after} (limit {limit})")
    check(
        after > limit,
        "a guard that trusts only the local count certifies an oversized payload",
        f"result measured {after} <= {limit}; the divergence is too small here "
        f"for this teeth check to be meaningful",
        extra=f"{after} > {limit} — over by {after - limit}",
    )
    note(
        "This is 2026-08-28 in one assertion: '150050 -> 15155 FITS' was true "
        "only in the arithmetic the guard was doing."
    )


def test_guard_makes_a_bounded_number_of_tokenize_calls():
    print("\n[10] the guard consults /tokenize a BOUNDED number of times")
    reset_fixture()
    msgs = conversation(30)  # 61 messages
    main._enforce_hard_budget(msgs, limit=8000, protect_system=1)
    calls = fixture("/_fixture/stats").get("tokenize", 0)
    print(f"       {len(msgs)} messages -> {calls} /tokenize call(s)")
    check(
        calls <= 8,
        "bounded round trips (1 ground-truth + <=6 verification rounds)",
        f"{calls} calls for {len(msgs)} messages",
        extra=f"{calls} calls for {len(msgs)} messages",
    )
    check(
        calls < len(msgs),
        "never one call per message (main.py:846-853 cost discipline)",
        f"{calls} calls for {len(msgs)} messages",
    )


# ============================================ 5. WHEN /tokenize IS UNREACHABLE


def test_unreachable_tokenize_falls_back_warns_and_still_serves():
    print("\n[11] /tokenize unreachable: falls back, warns, still returns a payload")
    reset_fixture()
    saved_url = main.VLLM_URL
    try:
        # A closed port on localhost — a genuine connection failure, not a
        # mocked one. This is the path the code's except-branch handles.
        main.VLLM_URL = "http://127.0.0.1:1"
        logsetup._reset_log_once_for_tests()
        with CaptureLogs() as logs:
            n = main.count_tokens_exact([{"role": "user", "content": "hello"}])
            out = main._enforce_hard_budget(conversation(8), limit=8000, protect_system=1)
        check(n is None, "count_tokens_exact returns None rather than raising", f"got {n!r}")
        warned = logs.at_least("WARNING")
        check(
            "unreachable" in warned or "/tokenize" in warned,
            "the fallback is announced at WARNING, naming /tokenize",
            f"warnings were: {warned[:300]!r}",
        )
        check(
            "undercount" in warned or "under-count" in warned,
            "the warning says WHAT the fallback costs, not just that it happened",
            f"warnings were: {warned[:300]!r}",
        )
        check(isinstance(out, list) and out, "the guard still returns a usable payload")
        check(out[-1]["role"] == "user", "and still keeps the newest turn")
    finally:
        main.VLLM_URL = saved_url
        logsetup._reset_log_once_for_tests()


def test_tokenize_http_error_and_garbage_are_both_survived():
    print("\n[12] /tokenize 500, and /tokenize 200-with-no-count")
    for mode, kw, label in (
        ("http_error", {"status": 500}, "HTTP 500"),
        ("http_error", {"status": 400}, "HTTP 400"),
        ("garbage", {}, "200 with no `count` field"),
    ):
        reset_fixture()
        set_mode(tokenize_mode=mode, **kw)
        logsetup._reset_log_once_for_tests()
        n = main.count_tokens_exact([{"role": "user", "content": "hello"}])
        check(n is None, f"{label} -> None (not a crash, not a bogus number)", f"got {n!r}")
        out = main._enforce_hard_budget(conversation(6), limit=8000, protect_system=1)
        check(isinstance(out, list) and out, f"{label} -> the guard still serves")
    reset_fixture()


def test_tokenize_hang_is_bounded_by_the_read_timeout():
    print("\n[13] /tokenize hangs: the 10s read timeout bounds the damage")
    import time as _t

    reset_fixture()
    set_mode(tokenize_mode="hang", delay=20.0)
    logsetup._reset_log_once_for_tests()
    t0 = _t.monotonic()
    n = main.count_tokens_exact([{"role": "user", "content": "hello"}])
    dt = _t.monotonic() - t0
    print(f"       returned {n!r} after {dt:.1f}s")
    check(n is None, "a hung /tokenize returns None", f"got {n!r}")
    check(
        dt < 15.0,
        "and does so inside the configured read timeout, not the request's",
        f"took {dt:.1f}s",
        extra=f"{dt:.1f}s",
    )
    reset_fixture()


# ================================= 6. WHEN /tokenize IS PLAUSIBLE BUT WRONG


def test_a_wrong_low_count_is_believed_and_the_result_is_oversized():
    print("\n[14] /tokenize returns a plausible but LOW count (the 0.5x lie)")
    reset_fixture()
    limit = 8000
    msgs = conversation(12)
    set_mode(tokenize_mode="wrong", factor=0.5)
    with CaptureLogs() as logs:
        out = main._enforce_hard_budget(msgs, limit=limit, protect_system=1)
    reset_fixture()  # back to honest, so `truth` is truth again
    actual = truth(out)
    print(f"       guard believed it fit; server actually charges {actual} against {limit}")
    check(
        actual > limit,
        "a lying /tokenize produces an oversized payload — as it must",
        f"result measured {actual} <= {limit}; the injected lie did not bite, so "
        f"this test is not measuring what it claims",
        extra=f"{actual} > {limit}",
    )
    check(
        "hard budget enforced" in logs.text(),
        "and the guard reports SUCCESS, because it has no better source",
        f"log was: {logs.text()[:300]!r}",
    )
    note(
        "The guard is exactly as honest as /tokenize. There is no arithmetic "
        "fix for this — only the calibration path (_BUDGET_MARGIN, P0-0b), "
        "which learns from vLLM's 400 rather than from /tokenize."
    )


def test_a_wrong_high_count_over_sheds_but_never_drops_the_newest_turn():
    print("\n[15] /tokenize returns a plausible but HIGH count (the 3x lie)")
    reset_fixture()
    msgs = conversation(12)
    set_mode(tokenize_mode="wrong", factor=3.0)
    with CaptureLogs() as logs:
        out = main._enforce_hard_budget(msgs, limit=8000, protect_system=1)
    reset_fixture()
    check(isinstance(out, list) and len(out) >= 1, "the guard terminates")
    check(
        out[-1]["content"] == msgs[-1]["content"],
        "the message the user just typed survives an over-shed",
        f"last message was {out[-1].get('content','')[:60]!r}",
    )
    check(
        any(m.get("role") == "system" for m in out),
        "the caller's protected system message survives an over-shed",
    )
    check(
        len(out) < len(msgs),
        "and it did over-shed, so this test exercised the path",
        f"{len(msgs)} -> {len(out)} messages",
        extra=f"{len(msgs)} -> {len(out)} messages",
    )
    if "FAILED to fit" in logs.text():
        note("over-shed ended at ERROR — expected when the lie exceeds what can be shed")


# ============================================ 7. THE 400 THE GUARD EXISTS FOR


def test_chat_completions_rejects_an_oversized_payload_recognisably():
    print("\n[16] the backend's context-length 400 is recognised by main")
    reset_fixture()
    msgs = [{"role": "user", "content": DECORATIVE_REPLY * 12}]
    r = httpx.post(
        f"{FIXTURE_URL}/v1/chat/completions",
        json={"model": "fixture-model", "messages": msgs, "max_tokens": 512},
        timeout=60.0,
    )
    check(r.status_code == 400, "an oversized request is refused", f"status {r.status_code}")
    body = r.text
    check(
        main._is_context_overflow(body),
        "main._is_context_overflow classifies it as a window overflow",
        f"body was {body[:300]!r}",
    )
    reported = main._reported_prompt_tokens(body)
    style = FIXTURE_INFO.get("error_style")
    if reported is None:
        note(
            f"FINDING: with FIXTURE_ERROR_STYLE={style} (the wording vllm 0.10.0 "
            f"emits: 'you requested N tokens (M in the messages, K in the "
            f"completion)'), main._reported_prompt_tokens returns None — its "
            f"regex _CTX_OVERFLOW_RE matches 'prompt contains (at least )?N "
            f"input tokens', which that wording does not contain. The "
            f"calibration in P0-0b therefore learns nothing from such a "
            f"rejection. Verify which wording vllm==0.24.0 (Dockerfile:78) "
            f"actually emits before treating this as a defect or as fine."
        )
    else:
        ok("_reported_prompt_tokens parsed the true size", f"{reported} tokens")


def test_chat_completions_serves_a_payload_the_guard_approved():
    print("\n[17] end to end: guard -> backend, and the backend accepts it")
    reset_fixture()
    limit = 8000
    out = main._enforce_hard_budget(conversation(12), limit=limit, protect_system=1)
    # max_tokens is set so the BACKEND'S OWN GATE is tight against exactly this
    # limit: vLLM refuses when token_num + max_tokens > max_model_len, so with
    # max_tokens = window - limit the request is served if and only if the
    # prompt really came in at or under `limit`.
    #
    # It used to be a flat 256 against a 32,768 window, which meant the backend
    # would have accepted a payload four times the size the guard promised. A
    # 200 proved the fixture was up, not that the guard was right. This is the
    # production relationship — limit is HARD_INPUT_LIMIT, the remainder is
    # GENERATION_RESERVE — rather than a number chosen to pass.
    r = httpx.post(
        f"{FIXTURE_URL}/v1/chat/completions",
        json={
            "model": "fixture-model",
            "messages": out,
            "max_tokens": FIXTURE_MAX_LEN - limit,
        },
        timeout=60.0,
    )
    check(
        r.status_code == 200,
        "what the guard approved, the backend serves — with the window's whole "
        "remainder reserved for generation",
        f"status {r.status_code}: {r.text[:300]!r}",
        extra=f"prompt_tokens {r.json().get('usage', {}).get('prompt_tokens')}"
        if r.status_code == 200
        else "",
    )


# ==================================== 8. D1: THE 400 THIS HARNESS DID NOT CATCH
#
# The harness existed to catch /tokenize contract failures. It ran green twice
# and shipped, and four hours later an assistant-final message list 400'd
# /tokenize in production and took compaction down.
#
# It missed it because every request it made ended on a USER turn. `truth()`
# sent the flag unset; `conversation()` was the only generator; the fixture
# answered an assistant-final list with a 200 that real vLLM has never given.
# Three separate places all agreed on a contract nothing in the tree sends.


def test_fixture_reproduces_the_assistant_final_refusal():
    print("\n[18] the fixture now REFUSES an assistant-final /tokenize, as vLLM does")
    reset_fixture()
    slice_ = old_turns(2)

    def post(msgs, **flags):
        return httpx.post(
            f"{FIXTURE_URL}/tokenize",
            json={"model": "fixture-model", "messages": msgs, **flags},
            timeout=30.0,
        )

    r = post(slice_, add_generation_prompt=True)
    check(
        r.status_code == 400,
        "add_generation_prompt=True on an assistant-final list is refused",
        f"status {r.status_code}; the fixture is not reproducing D1 at all, so "
        f"every assertion below it is vacuous",
    )
    body = r.text
    check(
        "add_generation_prompt" in body and "continue_final_message" in body,
        "with vLLM's own wording, which is what the log will carry",
        f"body was {body[:300]!r}",
        extra="'Consider using continue_final_message instead.'",
    )

    r_off = post(slice_, add_generation_prompt=False)
    check(
        r_off.status_code == 200 and isinstance(r_off.json().get("count"), int),
        "the same list with the flag off is counted normally",
        f"status {r_off.status_code}: {r_off.text[:200]!r}",
        extra=f"{r_off.json().get('count')} tokens" if r_off.status_code == 200 else "",
    )

    # Prove the refusal is about the FLAG AND THE SHAPE and not about the
    # content — otherwise a green [19] could be an artifact of this payload.
    set_mode(assistant_final_400=False)
    r_disabled = post(slice_, add_generation_prompt=True)
    reset_fixture()
    check(
        r_disabled.status_code == 200,
        "and with the check disabled the identical request succeeds",
        f"status {r_disabled.status_code}; the 400 above came from something "
        f"other than the shape check, so it is not evidence of D1",
    )

    # The sibling refusal, aimed at the wrong fix: keeping
    # add_generation_prompt=True and adding continue_final_message=True.
    r_both = post(conversation(1), add_generation_prompt=True, continue_final_message=True)
    reset_fixture()
    check(
        r_both.status_code == 400,
        "setting BOTH flags is refused too — the plausible wrong fix for D1",
        f"status {r_both.status_code}",
    )
    note(
        "the both-flags wording is reconstructed from vLLM's chat utils, NOT "
        "dumped from the image like the assistant-final one (which is quoted "
        "from the production log). No test asserts its text."
    )


def test_count_tokens_exact_measures_an_assistant_final_slice():
    print("\n[19] D1 REGRESSION: count_tokens_exact on the summarizer's own shape")
    reset_fixture()
    logsetup._reset_log_once_for_tests()
    slice_ = old_turns(3)
    with CaptureLogs() as logs:
        n = main.count_tokens_exact(slice_)
    st = fixture("/_fixture/stats")
    recorded = [s for s in shapes() if s["kind"] == "chat"]
    check(
        isinstance(n, int) and n > 0,
        "an assistant-final slice gets a real count",
        f"got {n!r} — this is D1 exactly: the measurement the summarizer needs "
        f"most is the one it cannot get",
        extra=f"{n} tokens",
    )
    check(
        st.get("tokenize.refused", 0) == 0,
        "because no 400 was provoked at all",
        f"fixture stats: {st!r}",
    )
    check(
        bool(recorded) and recorded[0]["add_generation_prompt"] is False,
        "the flag was decided FROM THE MESSAGES, not left True by the caller",
        f"the request the fixture actually received was {recorded!r}",
    )
    check(
        "degraded" not in logs.at_least("WARNING"),
        "and nothing announced a degradation",
        f"warnings: {logs.at_least('WARNING')[:300]!r}",
    )
    check(
        n == truth(slice_),
        "and the number is the server's number for that shape",
        f"count_tokens_exact said {n}, the server says {truth(slice_)}",
    )

    # --- TEETH: the pre-3a65aa1 call, with the flag left to the caller -------
    reset_fixture()
    logsetup._reset_log_once_for_tests()
    with CaptureLogs() as logs2:
        n_bad = main.count_tokens_exact(slice_, add_generation_prompt=True)
    warned = logs2.at_least("WARNING")
    check(
        n_bad is None,
        "forcing add_generation_prompt=True on the same slice loses the count",
        f"got {n_bad!r}; the fixture is not biting and this teeth check proves "
        f"nothing",
        extra="None — the exact production failure",
    )
    check(
        fixture("/_fixture/stats").get("tokenize.refused", 0) == 1,
        "because the server refused it, once",
    )
    check(
        "HTTP 400" in warned,
        "the loss is announced at WARNING with the status",
        f"warnings: {warned[:300]!r}",
    )
    check(
        "continue_final_message" in warned,
        "and the log carries vLLM's own wording — the whole diagnosis, in the "
        "line a human reads",
        f"warnings: {warned[:400]!r}",
    )
    reset_fixture()
    note(
        "[19] is the test that did not exist on 2026-08-28. Nothing about it "
        "needed production to happen first: the shape it sends is the shape "
        "split_messages has always produced."
    )


def test_summarize_survives_an_assistant_final_slice_end_to_end():
    print("\n[20] D1 END TO END: summarize() over a slice ending on the assistant")
    reset_fixture()
    logsetup._reset_log_once_for_tests()
    turns = old_turns(14)

    async def _run():
        async with httpx.AsyncClient() as c:
            return await main.summarize(c, turns)

    with CaptureLogs() as logs:
        out = asyncio.run(_run())
    text = logs.text()
    st = fixture("/_fixture/stats")
    print(f"       {len(turns)} turns -> summary of {len(out)} chars; stats {st}")
    check(
        isinstance(out, str) and out.strip(),
        "summarize returns a summary rather than degrading",
        f"got {out!r}",
    )
    check(
        st.get("tokenize.refused", 0) == 0,
        "not one /tokenize request in the whole run was refused",
        f"fixture stats: {st!r} — the 400 is back",
    )
    check(
        "summarize: token scale" in text,
        "the batches were sized by a MEASURED scale",
        f"log was: {text[:400]!r}",
    )
    check(
        "/tokenize did not answer" not in text,
        "and not by the fallback — the measurement reached the one path that "
        "needs it",
        f"log was: {text[:400]!r}",
    )
    check(
        st.get("chat_completions", 0) >= 1,
        "and the summarization calls were actually issued",
        f"fixture stats: {st!r}",
    )


# ================================ 9. D2: THE FALLBACK NOTHING EVER EXERCISED
#
# D2 shipped a scale=1.0 fallback in main.summarize(). The harness measured the
# batches on the MEASURED path and never on the fallback path, so a defective
# fallback was invisible. Both of the defects found in production four hours
# after the v3.1 deploy lived in fallbacks.


def test_the_summarize_fallback_sizes_batches_pessimistically_and_they_fit():
    print("\n[21] D2 REGRESSION: /tokenize down -> batches sized PESSIMISTICALLY")
    reset_fixture()
    logsetup._reset_log_once_for_tests()
    turns = old_turns(14)
    # The honest ratio for this slice, measured BEFORE /tokenize is taken away.
    local = main.count_tokens(turns)
    honest = truth(turns) / local if local else 1.0

    seen: dict = {}
    real_chunk = main._chunk_to_budget

    def _spy(t, budget, scale=1.0):
        out = real_chunk(t, budget, scale)
        # summarize() also chunks during its REDUCE rounds; the first call is
        # the map split, which is the one sized by the fallback scale.
        seen.setdefault("scale", scale)
        seen.setdefault("budget", budget)
        seen.setdefault("batches", out)
        return out

    # /tokenize is down; /v1/chat/completions is NOT, so summarize() runs its
    # whole real path with only the measurement missing. That is the production
    # condition, not a simulation of it.
    set_mode(tokenize_mode="http_error", status=500)
    main._chunk_to_budget = _spy
    try:

        async def _run():
            async with httpx.AsyncClient() as c:
                return await main.summarize(c, turns)

        with CaptureLogs() as logs:
            asyncio.run(_run())
    finally:
        main._chunk_to_budget = real_chunk
        reset_fixture()  # honest again, so `truth` is truth

    scale = seen.get("scale")
    budget = seen.get("budget")
    batches = seen.get("batches") or []
    print(
        f"       honest ratio for this slice {honest:.2f}x; the fallback chose "
        f"{scale}x; {len(batches)} batch(es) against budget {budget}"
    )
    check(scale is not None, "the batching call was observed", f"recorded {seen!r}")
    check(
        scale is not None and scale > 1.0,
        "the fallback is NOT scale=1.0 — that is D2",
        f"scale was {scale!r}, which asserts the local tokenizer is right at "
        f"the exact moment we have just discovered we cannot check it",
        extra=f"{scale}x",
    )
    check(
        scale is not None and scale >= honest,
        "and it is at least the ratio the server actually charges on this slice",
        f"fallback scale {scale} is BELOW the measured ratio {honest:.2f}x for "
        f"this content, so the fallback is optimistic. Either the pessimistic "
        f"constant is too small, or this tokenizer pair diverges harder than "
        f"the constant was set for — check which before dismissing it",
        extra=f"{scale}x >= {honest:.2f}x measured",
    )
    worst = max((truth(b) for b in batches), default=0)
    check(
        batches and worst <= budget,
        "and every batch the fallback packed FITS, measured on the server",
        f"largest fallback batch measured {worst} against budget {budget}",
        extra=f"largest {worst} <= {budget}",
    )

    # --- TEETH: the same slice, batched the way D2 batched it ---------------
    bad = real_chunk(turns, budget, 1.0)
    worst_bad = max(truth(b) for b in bad)
    check(
        worst_bad > budget,
        "the scale=1.0 fallback D2 shipped with does NOT fit",
        f"largest unscaled batch was {worst_bad} <= budget {budget}; the fixture "
        f"pair is not hostile enough for this teeth check to mean anything",
        extra=f"{worst_bad} > {budget} (+{worst_bad - budget}) in {len(bad)} batch(es)",
    )

    warned = logs.at_least("WARNING")
    check(
        "did not answer" in warned,
        "the fallback announces itself at WARNING",
        f"warnings: {warned[:300]!r}",
    )
    # The fallback WARNING must quote the scale it ACTUALLY applied. It used to
    # read "UNCORRECTED (scale 1.0)" while applying _PESSIMISTIC_SUMMARY_SCALE:
    # correct arithmetic, pre-fix diagnostic. D7 reported it and could not fix
    # it (main.py was owned elsewhere); the gate fixed it, so this is now an
    # assertion rather than a note. It is the A9 doctrine applied to A9's own
    # line — a log that cannot distinguish the fixed case from the broken one
    # is the failure, whichever half is wrong.
    check(
        "scale 1.0" not in warned and f"{scale:.2f}x" in warned,
        "and it quotes the scale it actually applied, not the pre-fix one",
        f"the fallback applied {scale}x but its WARNING says: {warned[:300]!r}",
        extra=f"log names {scale:.2f}x",
    )


# ============================ 10. EVERY /tokenize CLIENT, AND EVERY SHAPE IT SENDS


def _summarizer_count(text: str) -> int:
    import summarizer

    async def _ask():
        async with httpx.AsyncClient() as c:
            return await summarizer._count_tokens(c, FIXTURE_URL, "fixture-model", text)

    return asyncio.run(_ask())


def test_the_summarizer_modules_own_tokenize_client():
    print("\n[22] the THIRD shape: summarizer._count_tokens sends {model, prompt}")
    import summarizer

    reset_fixture()
    logsetup._reset_log_once_for_tests()
    text = DECORATIVE_REPLY
    n = _summarizer_count(text)
    seen = [s for s in shapes() if s["kind"] == "completion"]
    server = truth_prompt(text)
    check(
        n == server,
        "it reads .count out of the completion-shaped response",
        f"got {n}, the server says {server}",
        extra=f"{n} tokens",
    )
    check(
        bool(seen),
        "and the request really was COMPLETION-shaped, not chat-shaped",
        f"the fixture saw {shapes()!r}",
    )
    check(
        fixture("/_fixture/stats").get("tokenize.refused", 0) == 0,
        "this shape cannot hit the assistant-final 400 — no template is applied",
    )

    # --- and its fallback, which is a DIFFERENT fallback from main's --------
    reset_fixture()
    set_mode(tokenize_mode="http_error", status=500)
    logsetup._reset_log_once_for_tests()
    with CaptureLogs() as logs:
        n_down = _summarizer_count(text)
    reset_fixture()
    ceiling = summarizer._pessimistic_tokens(text)
    print(f"       /tokenize down -> {n_down} (ceiling {ceiling}, server would say {server})")
    check(
        n_down == ceiling,
        "with /tokenize down it returns the pessimistic CEILING, never an estimate",
        f"got {n_down}, expected the {summarizer._WORST_TOKENS_PER_CHAR}/char "
        f"ceiling of {ceiling}",
    )
    check(
        n_down > server,
        "and that ceiling really is above what the server charges for this text",
        f"ceiling {n_down} <= server {server}: the fallback would UNDER-split. "
        f"_WORST_TOKENS_PER_CHAR is set from production measurements, so a "
        f"failure here is most likely this tokenizer pair, not the constant — "
        f"but check before dismissing it",
        extra=f"{n_down} > {server}",
    )
    check(
        "tokens/char" in logs.at_least("WARNING"),
        "and the fallback says what it is doing",
        f"warnings: {logs.at_least('WARNING')[:300]!r}",
    )
    note(
        "summarizer.py holds its own httpx client, its own timeout, its own "
        "fallback and its own log keys — main.count_tokens_exact's fixes do not "
        "reach it. The module says why (an import cycle). It is a second "
        "implementation of the same contract and it needs its own coverage; "
        "before today it had none."
    )


def test_every_tokenize_client_in_the_tree_is_enumerated_and_covered():
    print("\n[23] enumerate every /tokenize client, and cover every shape it sends")
    reset_fixture()

    # --- who calls /tokenize at all -----------------------------------------
    url_literal = re.compile(r'f"\{[A-Za-z_][A-Za-z_0-9]*\}/tokenize"')
    here = pathlib.Path(__file__).resolve().parent
    found: dict[str, list[int]] = {}
    for p in sorted(here.glob("*.py")):
        if p.name.startswith("test_"):
            continue
        for i, line in enumerate(
            p.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        ):
            if url_literal.search(line):
                found.setdefault(p.name, []).append(i)
    for name, lines in sorted(found.items()):
        print(f"       {name}: {', '.join(str(x) for x in lines)}")
    known = {"main.py", "summarizer.py"}
    check(
        set(found) == known,
        "exactly two modules POST to /tokenize, and this suite covers both",
        f"found {sorted(found)} but expected {sorted(known)}. A NEW /tokenize "
        f"client is a new request shape, a new timeout and a new fallback, and "
        f"none of them are covered here until someone adds a case. That is how "
        f"D1 and D2 shipped.",
        extra=f"{', '.join(sorted(found))}",
    )

    # --- and every shape they send, driven through the real call sites ------
    reset_fixture()
    main.count_tokens_exact(conversation(2))  # guard / _sent_token_size shape
    main.count_tokens_exact(old_turns(2))  # main.summarize's slice shape
    _summarizer_count("a short piece of rollup input")  # summarizer.py's shape
    observed = {
        (s["kind"], s["add_generation_prompt"], s["last_role"]) for s in shapes()
    }
    for s in sorted(observed, key=str):
        print(f"       sent: kind={s[0]} add_generation_prompt={s[1]} last_role={s[2]}")
    want = {
        ("chat", True, "user"),
        ("chat", False, "assistant"),
        ("completion", None, None),
    }
    check(
        want <= observed,
        "all three shapes the tree sends were exercised against the fixture",
        f"missing {sorted(want - observed, key=str)}",
        extra=f"{len(observed)} distinct shape(s)",
    )
    check(
        fixture("/_fixture/stats").get("tokenize.refused", 0) == 0,
        "and not one of them was refused",
        f"fixture stats: {fixture('/_fixture/stats')!r}",
    )
    reset_fixture()
    note(
        "shapes NOT sent by any call site today, and therefore NOT covered: a "
        "system-final message list; a multimodal content array (the fixture has "
        "no vision tower and drops image parts); `tools=`; "
        "`continue_final_message=True`. If a call site starts sending one, [23] "
        "fails on the client count before the shape can go untested."
    )


# ============================================================== 11. THE REGIMES


def _all_tests():
    return [
        test_tokenize_answers_the_shape_the_compactor_reads,
        test_tokenize_does_not_context_check,
        test_prose_is_where_the_two_agree,
        test_box_drawing_is_where_the_estimator_lies,
        test_the_hostile_set_reproduces_the_divergence,
        test_summarizer_batches_actually_fit_the_server,
        test_unscaled_batching_would_have_overflowed,
        test_guard_result_measures_under_the_limit_on_the_server,
        test_guard_without_tokenize_lands_over_the_limit,
        test_guard_makes_a_bounded_number_of_tokenize_calls,
        test_unreachable_tokenize_falls_back_warns_and_still_serves,
        test_tokenize_http_error_and_garbage_are_both_survived,
        test_tokenize_hang_is_bounded_by_the_read_timeout,
        test_a_wrong_low_count_is_believed_and_the_result_is_oversized,
        test_a_wrong_high_count_over_sheds_but_never_drops_the_newest_turn,
        test_chat_completions_rejects_an_oversized_payload_recognisably,
        test_chat_completions_serves_a_payload_the_guard_approved,
        test_fixture_reproduces_the_assistant_final_refusal,
        test_count_tokens_exact_measures_an_assistant_final_slice,
        test_summarize_survives_an_assistant_final_slice_end_to_end,
        test_the_summarize_fallback_sizes_batches_pessimistically_and_they_fit,
        test_the_summarizer_modules_own_tokenize_client,
        test_every_tokenize_client_in_the_tree_is_enumerated_and_covered,
    ]


def main_() -> int:
    print("=" * 72)
    print("tokenizer-contract suite")
    print(f"  fixture      : {FIXTURE_URL}")
    print(f"  server tok   : {FIXTURE_INFO['tokenizer']} (vocab {FIXTURE_INFO['vocab_size']})")
    print(f"  window       : {FIXTURE_MAX_LEN}")
    print(f"  error style  : {FIXTURE_INFO['error_style']}")
    desc = set_local_tokenizer(LOCAL_TOKENIZER or None)
    print(f"  local count  : {desc}")
    print()
    print("  The two tokenizers are DIFFERENT on purpose. That difference is the")
    print("  bug class. Absolute numbers below say nothing about Cydonia-24B.")
    print("=" * 72)

    for t in _all_tests():
        try:
            t()
        except Exception as e:  # noqa: BLE001
            import traceback

            fail(t.__name__, f"raised {type(e).__name__}: {e}")
            traceback.print_exc()

    print("\n" + "=" * 72)
    if _NOTES:
        print("NOTES")
        for n in _NOTES:
            print(f"  - {n}")
        print()
    print("REMINDER: this validates the CONTRACT and the WIRING. The fixture's")
    print("tokenizer is not Cydonia-24B's. A green run is NOT evidence that the")
    print("production budget numbers are right.")
    print("=" * 72)

    if _FAILURES:
        print(f"\n{len(_FAILURES)} FAILURE(S):")
        for f in _FAILURES:
            print(f"  - {f}")
        return 1
    print("\nAll tokenizer-contract tests passed.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main_())
    finally:
        import shutil

        shutil.rmtree(_TMP_STATE, ignore_errors=True)
