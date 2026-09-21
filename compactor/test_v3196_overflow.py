"""
v3196-overflow — she must never see CONTEXT_OVERFLOW_MESSAGE.

Covers the three items of SP\\V3196_OVERFLOW_BRIEF.md, all in compactor/main.py:

  1. The hard-budget guard already measures a payload as over the window
     after shedding everything it may shed, and the request path used to
     forward it to vLLM anyway ("forward best effort") — spending the
     compaction, the memory injection and a round trip on a 400 that was
     already certain. `chat_completions` now refuses BEFORE the vLLM call
     whenever the guard measured a REAL overflow (not merely "fits the
     window but not with room for the current-time line", which must still
     be sent), and answers with the real numbers instead of a bare "too
     large" — see main._guard_predicted_overflow_reply.
  2. vLLM's own 400 for a payload the guard believed fit reports the ACTUAL
     prompt size — an exact measurement of our undercount. The request path
     now re-sheds against the just-widened `_BUDGET_MARGIN` and resends
     ONCE, transparently, before ever telling the user anything failed.
  3. `_BUDGET_MARGIN` is persisted (memory.atomic_write_json /
     memory.read_json) and reloaded at import, so a redeploy does not
     re-learn the correction by failing on her again.

Every fixture here is synthetic: repeated filler characters and literal
sentinel strings, no real conversation, fact or persona text. Counts, ids
and hashes only, per SP\\V3195_SHARED.md and SP\\V3196_OVERFLOW_BRIEF.md.

Run: python test_v3196_overflow.py
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from unittest.mock import patch

# Small, predictable window, matching test_budget_guard.py's own numbers so
# the two files' fixtures stay comparable. No MODEL_REPO -> char/4 estimator.
os.environ.pop("MODEL_REPO", None)
os.environ["MAX_MODEL_LEN"] = "1000"
os.environ["COMPACTOR_GENERATION_RESERVE"] = "200"   # HARD_INPUT_LIMIT = 800
# 50, not the shipped default (256): at this file's tiny MAX_MODEL_LEN=1000,
# the shipped default alone (250 > GENERATION_RESERVE's own 200) would make
# the "real ceiling" (MAX_MODEL_LEN - floor = 744) TIGHTER than the POLICY
# limit (HARD_INPUT_LIMIT = 800) — the inverse of the production relationship
# this lane's item 1 rework is about (MAX_MODEL_LEN 32768, GENERATION_RESERVE
# 12000, real ceiling ~32512 vs. policy ~20768: real is much LOOSER). With
# floor=50 the real ceiling (950) sits comfortably above the policy limit
# (800), so section [3] below can construct a payload that exceeds the
# policy but still fits the real window — the exact shape three real pod
# requests took on 2026-09-18/19/21 and the first cut of this lane refused
# by mistake. The shipped default (256) is still pinned exactly, separately,
# in its own subprocess in section [3b] alongside the coordinator's own
# production numbers.
os.environ["COMPACTOR_MIN_GENERATION_FLOOR"] = "50"
os.environ["COMPACTOR_SUMMARY_MAX_TOKENS"] = "100"
os.environ["COMPACTOR_SUMMARY_INPUT_RESERVE"] = "100"
os.environ["COMPACTOR_RAG_ENABLED"] = "false"

_TMP_ROOT = tempfile.mkdtemp(prefix="compactor-test-v3196-overflow-")
os.environ["COMPACTOR_STORAGE_ROOT"] = _TMP_ROOT

import main  # noqa: E402
import memory  # noqa: E402
import retrieval  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

retrieval._available = False
retrieval._embedder = None
retrieval._chroma_collection = None

# R22 (test_budget_guard.py's own fix, mirrored here): count_tokens_exact and
# count_text_tokens_exact open a REAL httpx.post to VLLM_URL/tokenize, which
# nothing in this file ever points at a live server. Fail it fast instead of
# eating a ~4.2s dead-TCP timeout per call.
_real_httpx_post = main.httpx.post


def _fail_tokenize_fast(url, *args, **kwargs):
    if "/tokenize" in url:
        raise main.httpx.ConnectError(
            "connection refused (stubbed for test speed — see R22)"
        )
    return _real_httpx_post(url, *args, **kwargs)  # pragma: no cover


main.httpx.post = _fail_tokenize_fast
memory.ensure_storage_layout()

client = TestClient(main.app, client=("127.0.0.1", 12345), raise_server_exceptions=False)

_FAILED = False


def check(cond, label):
    global _FAILED
    print(f"  {'ok  ' if cond else 'FAIL'} {label}")
    if not cond:
        _FAILED = True


def user(text):
    return {"role": "user", "content": text}


def big_user(approx_tokens):
    return {"role": "user", "content": "Item. " + "w" * (approx_tokens * 4)}


PERSONA = "PERSONA-SENTINEL " + "p" * 183


def ctx_400(actual_tokens, limit_tokens=1000):
    return (
        '{"error":{"message":"This model\'s maximum context length is '
        + str(limit_tokens) + ' tokens. However, you requested 0 output '
        'tokens and your prompt contains ' + str(actual_tokens) + ' input '
        'tokens, for a total of ... (parameter=input_tokens)"}}'
    )


OTHER_400 = '{"error":{"message":"only user and assistant roles are supported!"}}'


class _CaptureLogs(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


def _swallow_tail(coro, label=None):
    try:
        coro.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# A stub httpx.AsyncClient whose response is SCRIPTED per upstream call, by
# call index (0 = the first attempt, 1 = the one automatic retry, ...). This
# is the one thing test_budget_guard.py's own stubs cannot do — its
# _StubVLLMRefusing answers every call identically, which is exactly why item
# 2's retry cannot be exercised through it (see fix-3196-overflow.md).
# ---------------------------------------------------------------------------

class _StubStreamResp:
    def __init__(self, status_code, body=""):
        self.status_code = status_code
        self._body = body

    async def aread(self):
        return self._body.encode()

    async def aiter_raw(self):
        yield (
            b'data: {"choices":[{"index":0,"delta":{"content":"ok"},'
            b'"finish_reason":null}]}\n\n'
        )
        yield b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
        yield b"data: [DONE]\n\n"


class _StubStreamCM:
    def __init__(self, resp):
        self._resp = resp

    async def __aenter__(self):
        return self._resp

    async def __aexit__(self, *exc):
        return False


class _StubNonStreamResp:
    def __init__(self, status_code, body=None):
        self.status_code = status_code
        if body is None:
            body = json.dumps({
                "id": "stub",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }],
            })
        self.text = body

    def json(self):
        return json.loads(self.text)


class _SequencedVLLM:
    """`script` is [(status_code, body_str), ...], one entry per upstream
    call this test expects; extra calls beyond the script's length repeat
    its last entry, and a test that must never see more calls than the
    script has entries asserts that directly against `.sent`."""

    sent: list[dict] = []
    script: list[tuple[int, str]] = []

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def aclose(self):
        pass

    @classmethod
    def _next(cls):
        i = len(cls.sent)
        idx = min(i, len(cls.script) - 1)
        return cls.script[idx]

    def stream(self, method, url, json=None, **kw):
        status, body = self._next()
        _SequencedVLLM.sent.append(json)
        return _StubStreamCM(_StubStreamResp(status, body))

    async def post(self, url, json=None, **kw):
        status, body = self._next()
        _SequencedVLLM.sent.append(json)
        return _StubNonStreamResp(status, None if status < 400 else body)


def _sse_chunks(response):
    out = []
    for block in response.text.split("\n\n"):
        for line in block.split("\n"):
            if line.startswith("data: "):
                payload = line[6:].strip()
                if payload and payload != "[DONE]":
                    out.append(json.loads(payload))
    return out


def _assistant_text(chunks):
    return "".join(
        c.get("choices", [{}])[0].get("delta", {}).get("content") or ""
        for c in chunks
    )


def _post(messages, conv_id, script, *, stream, tails=None, save_margin_target=None,
          extra=None):
    """POST one chat completion with the upstream response(s) scripted.

    -> (response, sent_bodies, log_records). `_SequencedVLLM.sent` is reset
    first and returned by reference (copied) so callers can assert call
    counts. `extra`, if given, is merged into the request JSON (e.g.
    `{"max_tokens": 500}`) — item 1's clamp behaviour is what this exists
    for."""
    _SequencedVLLM.sent = []
    _SequencedVLLM.script = script
    handler = _CaptureLogs()
    lg = logging.getLogger("compactor")
    lg.addHandler(handler)

    def _record_tail(coro, label=None):
        if tails is not None:
            tails.append(coro)
        try:
            coro.close()
        except Exception:
            pass

    body = {"model": "stub-model", "messages": messages, "stream": stream}
    if extra:
        body.update(extra)
    try:
        with patch.object(main.httpx, "AsyncClient", _SequencedVLLM), \
             patch.object(main, "_fire_and_forget", _record_tail):
            r = client.post(
                "/v1/chat/completions",
                json=body,
                headers={"X-Conversation-Id": conv_id},
            )
    finally:
        lg.removeHandler(handler)
    return r, list(_SequencedVLLM.sent), handler.records


def _reset_margin():
    main._BUDGET_MARGIN = 0
    main._budget_ok_streak = 0


# ---------------------------------------------------------------------------
# Subprocess probe helpers, moved up (originally written for section [8]'s
# mutation checks only) so section [3] can ALSO use them: pinning the exact
# production numbers (MAX_MODEL_LEN=32768, GENERATION_RESERVE=12000) needs
# its own process — this file's usual MAX_MODEL_LEN=1000 window is read once
# at `main` import and cannot be changed mid-process for one section only.
# ---------------------------------------------------------------------------

_REPO_DIR = os.path.dirname(os.path.abspath(__file__))


def _mutated_copy(needle, replacement=None, must_count=1):
    """Copy this directory's *.py files into a fresh temp dir with `needle`
    replaced by `replacement` in main.py — exactly `must_count` time(s) — and
    return that directory. Asserts the mutated file still parses.

    `needle` may also be a list of (needle, replacement, must_count) triples,
    applied in order, each checked for its own exact count — for a mutation
    that only breaks something when two cooperating lines change together.

    `needle=None` (both None) copies the tree UNMODIFIED — used by section
    [3]'s production-shape probes, which need their own process (a
    different MAX_MODEL_LEN) but no mutation at all."""
    dst = tempfile.mkdtemp(prefix="compactor-test-v3196-mutant-")
    for name in os.listdir(_REPO_DIR):
        if name.endswith(".py"):
            shutil.copy(os.path.join(_REPO_DIR, name), os.path.join(dst, name))
    if needle is None:
        return dst
    main_path = os.path.join(dst, "main.py")
    with open(main_path, encoding="utf-8") as f:
        src = f.read()
    edits = (
        needle if isinstance(needle, list)
        else [(needle, replacement, must_count)]
    )
    for _needle, _repl, _count in edits:
        count = src.count(_needle)
        if count != _count:
            raise AssertionError(
                f"mutation needle appeared {count} time(s), expected "
                f"{_count}: {_needle[:80]!r}"
            )
        src = src.replace(_needle, _repl, _count)
    with open(main_path, "w", encoding="utf-8") as f:
        f.write(src)
    import ast
    ast.parse(src)  # must still be valid Python
    return dst


def _run_probe_against(dst, probe_body, env_overrides=None):
    """Run `probe_body` (a script using `main` and this file's own helpers,
    re-imported inline) against the copy in a subprocess, isolated from this
    process's already-imported `main` module. -> (CompletedProcess).

    `env_overrides` replaces the default MAX_MODEL_LEN/GENERATION_RESERVE."""
    storage = tempfile.mkdtemp(prefix="compactor-test-v3196-mutant-storage-")
    env = dict(os.environ)
    env["COMPACTOR_STORAGE_ROOT"] = storage
    env["MAX_MODEL_LEN"] = "1000"
    env["COMPACTOR_GENERATION_RESERVE"] = "200"
    env["COMPACTOR_RAG_ENABLED"] = "false"
    env.pop("MODEL_REPO", None)
    if env_overrides:
        env.update(env_overrides)
    script = (
        "import sys; sys.path.insert(0, " + repr(dst) + ")\n"
        + probe_body
    )
    try:
        return subprocess.run(
            [sys.executable, "-c", script], env=env, capture_output=True,
            text=True, timeout=60,
        )
    finally:
        shutil.rmtree(storage, ignore_errors=True)


# ===========================================================================
print("[1] item 1 — a genuinely unfittable payload never reaches vLLM")
# ===========================================================================
# One system message (protect_system=1 covers it -- nothing else is
# injected: no facts saved, RAG off) plus one user turn alone bigger than
# the whole window. _enforce_hard_budget has nothing left to shed once the
# newest turn and the caller's own system prompt are all that remain --
# exactly the "true floor" item 1's brief section describes.
for stream in (False, True):
    label = "stream" if stream else "non-stream"
    _reset_margin()
    tails = []
    BIG = [
        {"role": "system", "content": PERSONA},
        big_user(1400),  # ~5600 chars -> ~1400 local tokens, well over 800
    ]
    r, sent, records = _post(
        BIG, f"unfittable-{label}", script=[(200, "")], stream=stream, tails=tails,
    )
    check(len(sent) == 0,
          f"[{label}] vLLM was never called ({len(sent)} call(s))")
    check("hard budget FAILED to fit" in "\n".join(rc.getMessage() for rc in records),
          f"[{label}] the guard's own ERROR line still fired")
    check(
        any("refusing to forward to vLLM" in rc.getMessage() for rc in records),
        f"[{label}] the pre-flight refusal is itself logged at ERROR",
    )
    if stream:
        check(r.status_code == 200, f"[{label}] SSE opens 200 (status is in the error object)")
        chunks = _sse_chunks(r)
        text = _assistant_text(chunks)
        err = chunks[-1].get("error") if chunks else None
        check(err is not None and err.get("code") == "context_length_exceeded",
              f"[{label}] typed so a client can tell it from a real reply")
    else:
        check(r.status_code == 400, f"[{label}] a real 400, not a relayed vLLM body")
        text = r.json()["error"]["message"]
        check(r.json()["error"]["code"] == "context_length_exceeded",
              f"[{label}] typed as context_length_exceeded")
    check(main._REJECTED_PREAMBLE in text, f"[{label}] leads with the outcome")
    check("Send it again" not in text,
          f"[{label}] no retry is promised for a payload nothing can shrink")
    check(
        any(ch.isdigit() for ch in text) and "tokens" in text,
        f"[{label}] the reply names real numbers, not a bare 'too large'",
    )
    check(len(tails) == 0, f"[{label}] nothing was memorized (no tail fired)")


# ===========================================================================
print("\n[2] item 1 — CONTROL: an ordinary request still reaches vLLM")
# ===========================================================================
for stream in (False, True):
    label = "stream" if stream else "non-stream"
    _reset_margin()
    r, sent, _records = _post(
        [user("hello")], f"ordinary-{label}", script=[(200, "")], stream=stream,
    )
    check(len(sent) == 1, f"[{label}] CONTROL: an easy request still reaches vLLM once")
    check(r.status_code == 200, f"[{label}] CONTROL: and succeeds")


# ===========================================================================
print("\n[3] item 1 REWORK — exceeding the POLICY limit (GENERATION_RESERVE) "
      "no longer refuses; only exceeding the REAL ceiling (MAX_MODEL_LEN "
      "minus the floor) does")
# ===========================================================================
# The coordinator's correction, in miniature: the first cut of this lane
# refused whenever `_enforce_hard_budget` missed its POLICY shedding target
# (`effective_limit`, GENERATION_RESERVE folded in). Real pod evidence
# (2026-09-18/19/21) showed vLLM accepting payloads that missed that target
# by a wide margin, because vLLM only rejects on MAX_MODEL_LEN itself — see
# MIN_GENERATION_FLOOR's own comment in main.py for the vLLM 0.19.0 source
# this is read from. `_enforce_hard_budget` is replaced with a scripted
# stand-in so the exact boundary (`_fit_request_for_vllm`'s own `measured +
# MIN_GENERATION_FLOOR + _BUDGET_MARGIN > MAX_MODEL_LEN`) can be tested
# precisely — it reads ONLY `report["measured"]`, not `report["fits"]` or
# `report["limit"]` (the POLICY-relative fields), which this section proves
# by choosing a `measured` that fails the old policy check but passes the
# new real one.

class _ScriptedGuard:
    def __init__(self, fits, measured, limit):
        self.fits, self.measured, self.limit = fits, measured, limit
        self.calls = 0

    def __call__(self, messages, limit=None, protect_system=1, report=None,
                 reserve=0, standin_protected=True):
        self.calls += 1
        if report is not None:
            report.update({
                "limit": self.limit, "measured": self.measured,
                "fits": self.fits, "counted_by": "scripted",
                "dropped_turns": 3, "trimmed_blocks": 1, "dropped_blocks": 2,
            })
        return list(messages)


_reset_margin()
check(main.MIN_GENERATION_FLOOR == 50,
      "fixture: this file's env override took (see the top-of-file comment "
      "for why 50, not the shipped 256, at this file's tiny MAX_MODEL_LEN)")
POLICY = main.HARD_INPUT_LIMIT               # 800 in this file's env
REAL_CEILING = main.MAX_MODEL_LEN - main.MIN_GENERATION_FLOOR   # 950
check(REAL_CEILING > POLICY,
      f"fixture: the real ceiling ({REAL_CEILING}) sits above the policy "
      f"limit ({POLICY}) — the production relationship, only reachable "
      f"here because of the env override just above")

# Exceeds the policy limit (so the OLD code refused this) but still comfortably
# inside the real ceiling.
guard = _ScriptedGuard(fits=False, measured=POLICY + 40, limit=POLICY - 50)
with patch.object(main, "_enforce_hard_budget", guard):
    r, sent, records = _post(
        [user("what time is it")], "policy-exceeded-forwarded",
        script=[(200, "")], stream=False,
    )
check(guard.calls == 1, "fixture: the scripted guard ran on the request path")
check(len(sent) == 1,
      f"measured ({guard.measured}) is OVER the policy limit ({POLICY}) but "
      f"under the real ceiling ({REAL_CEILING}) — vLLM IS still called "
      f"({len(sent)} call(s)), where the first cut of this lane refused")
check(r.status_code == 200, "and the request succeeds")
check(
    not any("refusing to forward to vLLM" in rc.getMessage() for rc in records),
    "and no pre-flight refusal is logged for it",
)

# Exceeds the real ceiling too -- genuinely nothing left to do.
guard2 = _ScriptedGuard(fits=False, measured=REAL_CEILING + 10, limit=POLICY - 50)
with patch.object(main, "_enforce_hard_budget", guard2):
    r2, sent2, _records2 = _post(
        [user("what time is it")], "real-ceiling-exceeded",
        script=[(200, "")], stream=False,
    )
check(len(sent2) == 0,
      f"measured ({guard2.measured}) is over the REAL ceiling "
      f"({REAL_CEILING}) — vLLM is NOT called ({len(sent2)} call(s))")
check(r2.status_code == 400, "and the caller gets the honest pre-flight 400")

# The clamp: an explicit max_tokens that would combine with `measured` to
# exceed MAX_MODEL_LEN is reduced, not left to cause a guaranteed vLLM 400.
_clamped_before = main.max_tokens_clamp_state()["clamped"]
guard3 = _ScriptedGuard(fits=True, measured=POLICY - 40, limit=POLICY)
with patch.object(main, "_enforce_hard_budget", guard3):
    r3, sent3, records3 = _post(
        [user("hi")], "explicit-max-tokens-clamped",
        script=[(200, "")], stream=False,
        extra={"max_tokens": main.MAX_MODEL_LEN},  # absurdly large on purpose
    )
check(len(sent3) == 1, "the clamp does not refuse -- it still forwards")
_sent_max_tokens = sent3[0].get("max_tokens")
_expected_room = main.MAX_MODEL_LEN - guard3.measured
check(_sent_max_tokens == _expected_room,
      f"max_tokens {main.MAX_MODEL_LEN} -> {_sent_max_tokens} (expected "
      f"{_expected_room} = MAX_MODEL_LEN - measured), not sent unclamped")
check(
    any("clamping max_tokens" in rc.getMessage() and rc.levelno == logging.WARNING
        for rc in records3),
    "the clamp is logged at WARNING with the numbers",
)
check(main.max_tokens_clamp_state()["clamped"] == _clamped_before + 1,
      "and counted, for /health/full")

# CONTROL: no explicit max_tokens -> nothing is invented. vLLM's own
# get_max_tokens() already auto-bounds generation to MAX_MODEL_LEN - input
# when max_tokens is absent (see MIN_GENERATION_FLOOR's comment) -- adding
# one here would only add a NEW failure mode a real client's silence never
# had.
guard4 = _ScriptedGuard(fits=True, measured=POLICY - 40, limit=POLICY)
with patch.object(main, "_enforce_hard_budget", guard4):
    r4, sent4, _records4 = _post(
        [user("hi")], "no-max-tokens-untouched", script=[(200, "")], stream=False,
    )
check(len(sent4) == 1 and "max_tokens" not in sent4[0],
      "CONTROL: a request with no max_tokens is forwarded without one")


# ===========================================================================
print("\n[3b] item 1 REWORK — pinning the PRODUCTION shape exactly (own "
      "subprocess: MAX_MODEL_LEN=32768, GENERATION_RESERVE=12000, the "
      "shipped MIN_GENERATION_FLOOR default of 256)")
# ===========================================================================
# The coordinator's own numbers, run against the REAL (unmutated,
# uninstrumented) source in its own process -- this file's usual
# MAX_MODEL_LEN=1000 is fixed at `main` import and cannot be swapped in for
# one section only. `_enforce_hard_budget` is scripted here too (same
# reasoning as [3] above: pins the EXACT numbers the coordinator cited --
# input ~23,200, policy ~20,674 -- rather than hoping a real fixture lands on
# them), inside a subprocess that also drives the real /v1/chat/completions
# endpoint end to end.

_PROD_PROBE = r"""
import json
import main
from fastapi.testclient import TestClient
from unittest.mock import patch
client = TestClient(main.app, client=("127.0.0.1", 12345), raise_server_exceptions=False)

MEASURED = 23200
POLICY = 20674   # what the pod's own log reported as the enforced limit

class _Guard:
    calls = 0
    def __call__(self, messages, limit=None, protect_system=1, report=None,
                 reserve=0, standin_protected=True):
        _Guard.calls += 1
        if report is not None:
            report.update({"limit": POLICY, "measured": MEASURED, "fits": False,
                            "counted_by": "scripted", "dropped_turns": 1200,
                            "trimmed_blocks": 0, "dropped_blocks": 3})
        return list(messages)

class _Resp:
    status_code = 200
    text = ""
    def json(self):
        return {"id": "x", "choices": [{"index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop"}]}

class _Client:
    sent = []
    def __init__(self, *a, **k): pass
    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False
    async def post(self, url, json=None, **kw):
        _Client.sent.append(json)
        return _Resp()
    async def aclose(self): pass

with patch.object(main, "_enforce_hard_budget", _Guard()), \
     patch.object(main.httpx, "AsyncClient", _Client), \
     patch.object(main, "_fire_and_forget", lambda coro, label=None: coro.close()):
    extra = {"max_tokens": __REQ_MAX_TOKENS__} if __REQ_MAX_TOKENS__ else {}
    r = client.post("/v1/chat/completions",
                     json={"model": "m", "messages": [{"role": "user", "content": "hi"}],
                           "stream": False, **extra},
                     headers={"X-Conversation-Id": "prod-shape"})
print("STATUS", r.status_code)
print("CALLS", len(_Client.sent))
print("SENT_MAX_TOKENS", (_Client.sent[0] or {}).get("max_tokens") if _Client.sent else None)
"""

_prod_env = {"MAX_MODEL_LEN": "32768", "COMPACTOR_GENERATION_RESERVE": "12000"}
_dst_prod = _mutated_copy(None)

def _run_prod_probe(max_tokens):
    env = dict(_prod_env)
    env["COMPACTOR_MIN_GENERATION_FLOOR"] = "256"
    script = _PROD_PROBE.replace("__REQ_MAX_TOKENS__", repr(max_tokens))
    return _run_probe_against(_dst_prod, script, env_overrides=env)

# The production shape itself, no explicit max_tokens (the common case --
# OpenWebUI does not send one by default): MUST be forwarded.
_res_prod = _run_prod_probe(None)
check(_res_prod.returncode == 0,
      f"probe ran cleanly (stderr: {_res_prod.stderr[-400:]!r})")
check("STATUS 200" in _res_prod.stdout,
      f"the production shape (measured=23200, policy=20674, MAX_MODEL_LEN="
      f"32768) is FORWARDED, not refused (stdout: {_res_prod.stdout!r})")
check("CALLS 1" in _res_prod.stdout, "exactly one upstream call")
check("SENT_MAX_TOKENS None" in _res_prod.stdout,
      "no max_tokens invented when the client sent none")

# Same shape, but the client asks for more room than 32768 - 23200 = 9568
# actually has: MUST be clamped, not refused and not sent unclamped.
_res_clamp = _run_prod_probe(15000)
check(_res_clamp.returncode == 0,
      f"probe ran cleanly (stderr: {_res_clamp.stderr[-400:]!r})")
check("STATUS 200" in _res_clamp.stdout,
      f"still forwarded, not refused (stdout: {_res_clamp.stdout!r})")
check("SENT_MAX_TOKENS 9568" in _res_clamp.stdout,
      f"max_tokens 15000 -> 9568 (= 32768 - 23200), not sent unclamped "
      f"(stdout: {_res_clamp.stdout!r})")

shutil.rmtree(_dst_prod, ignore_errors=True)


# ===========================================================================
print("\n[3c] item 1 REWORK — CONTROL: input ALONE over MAX_MODEL_LEN is "
      "still refused before the call, even in the production-scale window")
# ===========================================================================

_CONTROL_PROBE = _PROD_PROBE.replace("MEASURED = 23200", "MEASURED = 33000")

def _run_control_probe():
    env = dict(_prod_env)
    env["COMPACTOR_MIN_GENERATION_FLOOR"] = "256"
    script = _CONTROL_PROBE.replace("__REQ_MAX_TOKENS__", "None")
    dst = _mutated_copy(None)
    try:
        return _run_probe_against(dst, script, env_overrides=env)
    finally:
        shutil.rmtree(dst, ignore_errors=True)

_res_ctrl = _run_control_probe()
check(_res_ctrl.returncode == 0,
      f"probe ran cleanly (stderr: {_res_ctrl.stderr[-400:]!r})")
check("STATUS 400" in _res_ctrl.stdout,
      f"CONTROL: 33,000 measured tokens alone exceeds MAX_MODEL_LEN (32768) "
      f"-- refused before the call (stdout: {_res_ctrl.stdout!r})")
check("CALLS 0" in _res_ctrl.stdout, "vLLM was never called")


# ===========================================================================
print("\n[3d] item 1 REWORK — the config trap: COMPACTOR_GENERATION_RESERVE "
      "below MIN_GENERATION_FLOOR must never invert the policy limit and "
      "the serve limit")
# ===========================================================================
# The coordinator's second correction. An operator who sets
# GENERATION_RESERVE below the shipped MIN_GENERATION_FLOOR default (256)
# has explicitly accepted a SMALLER reply budget than the floor alone would
# enforce -- and a flat floor ignored that choice, so the guard could
# certify a payload as fitting its own (looser) policy target while
# `_fit_request_for_vllm` still refused to serve it. `_serve_floor` closes
# this by construction: floor = min(MIN_GENERATION_FLOOR, MAX_MODEL_LEN -
# effective_limit), so the serve limit can never sit BELOW the policy limit
# `_enforce_hard_budget` was actually handed.

print("  [3d-i] the invariant itself, swept across the combinations that "
      "used to invert")
for max_model_len, reserve, req_max_tokens, floor_default in (
    (1000, 200, 0, 256),    # this file's own trap: reserve(200) < floor(256)
    (32768, 100, 0, 256),   # production scale, an aggressively small reserve
    (4096, 1, 0, 256),      # near-zero reserve -- floor must not overrule it
    (1000, 16384, 0, 256),  # reserve far above MAX_MODEL_LEN itself
    (1000, 200, 900, 256),  # a client's own huge max_tokens dominates reserve
    (1000, 5000, 0, 50),    # floor SMALLER than reserve -- floor still binds
):
    _saved_floor = main.MIN_GENERATION_FLOOR
    _saved_mml = main.MAX_MODEL_LEN
    try:
        main.MIN_GENERATION_FLOOR = floor_default
        main.MAX_MODEL_LEN = max_model_len
        effective_limit = min(
            max_model_len,
            max(256, max_model_len - max(reserve, req_max_tokens)),
        )
        floor = main._serve_floor(effective_limit)
        serve_limit = max_model_len - floor
        check(
            serve_limit >= effective_limit,
            f"MAX_MODEL_LEN={max_model_len} reserve={reserve} "
            f"req_max_tokens={req_max_tokens} floor_default={floor_default}: "
            f"serve_limit ({serve_limit}) >= policy limit ({effective_limit}) "
            f"-- floor resolved to {floor}",
        )
    finally:
        main.MIN_GENERATION_FLOOR = _saved_floor
        main.MAX_MODEL_LEN = _saved_mml

print("  [3d-ii] end to end: a payload the guard certifies as fitting its "
      "policy target is never refused by the serve check, even under this "
      "file's own trap ratio (GENERATION_RESERVE=200 < the SHIPPED "
      "MIN_GENERATION_FLOOR default of 256)")
_inv_probe = r"""
import main
from fastapi.testclient import TestClient
from unittest.mock import patch
client = TestClient(main.app, client=("127.0.0.1", 12345), raise_server_exceptions=False)

class _Guard:
    def __call__(self, messages, limit=None, protect_system=1, report=None,
                 reserve=0, standin_protected=True):
        # 790: just under the 800-token POLICY limit (main.HARD_INPUT_LIMIT
        # at MAX_MODEL_LEN=1000/GENERATION_RESERVE=200 -- "fits" as far as
        # the guard is concerned), and NOT tied to `limit` itself (`limit`
        # here is effective_limit - _time_reserve, narrowed by the
        # current-time line's own reserve, which would blur the exact
        # boundary this probe needs to cross). The CORRECT floor (200, from
        # min(256, 1000-800)) leaves this comfortably under MAX_MODEL_LEN
        # (790+200=990); the MUTANT flat floor (256) does not
        # (790+256=1046) -- the exact gap this mutation check needs to be
        # observable at all.
        if report is not None:
            report.update({"limit": limit, "measured": 790, "fits": True,
                            "counted_by": "scripted", "dropped_turns": 0,
                            "trimmed_blocks": 0, "dropped_blocks": 0})
        return list(messages)

class _Resp:
    status_code = 200
    text = ""
    def json(self):
        return {"id": "x", "choices": [{"index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop"}]}

class _Client:
    sent = []
    def __init__(self, *a, **k): pass
    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False
    async def post(self, url, json=None, **kw):
        _Client.sent.append(json)
        return _Resp()
    async def aclose(self): pass

with patch.object(main, "_enforce_hard_budget", _Guard()), \
     patch.object(main.httpx, "AsyncClient", _Client), \
     patch.object(main, "_fire_and_forget", lambda coro, label=None: coro.close()):
    r = client.post("/v1/chat/completions",
                     json={"model": "m", "messages": [{"role": "user", "content": "hi"}],
                           "stream": False},
                     headers={"X-Conversation-Id": "inversion-trap"})
print("STATUS", r.status_code)
print("CALLS", len(_Client.sent))
"""
_dst_inv = _mutated_copy(None)
try:
    _res_inv = _run_probe_against(
        _dst_inv, _inv_probe,
        env_overrides={
            "MAX_MODEL_LEN": "1000",
            "COMPACTOR_GENERATION_RESERVE": "200",
            "COMPACTOR_MIN_GENERATION_FLOOR": "256",
        },
    )
finally:
    shutil.rmtree(_dst_inv, ignore_errors=True)
check(_res_inv.returncode == 0,
      f"probe ran cleanly (stderr: {_res_inv.stderr[-400:]!r})")
check("STATUS 200" in _res_inv.stdout and "CALLS 1" in _res_inv.stdout,
      f"a payload the guard reports as fitting its policy limit (measured "
      f"790, under the 800-token policy limit, under the trap ratio "
      f"reserve=200 < floor=256) is forwarded, not refused by a floor that "
      f"would otherwise sit BELOW the policy limit (744 < 800) (stdout: "
      f"{_res_inv.stdout!r})")


# ===========================================================================
print("\n[4] item 2 — a rejection that TAUGHT the calibration something is "
      "retried ONCE, transparently")
# ===========================================================================
for stream in (False, True):
    label = "stream" if stream else "non-stream"
    _reset_margin()
    tails = []
    # First response: vLLM reports the TRUE size (1234) against the 800-token
    # budget we sent -- an undercount the guard did not predict (the REAL
    # _enforce_hard_budget ran here, on an ordinary small message, and
    # reported fits=True; nothing about this shape trips the item-1
    # pre-flight). Second response: 200 -- what a real vLLM would answer
    # once the retry re-shed against the just-widened margin.
    r, sent, records = _post(
        [user("hello")], f"retry-succeeds-{label}",
        script=[(400, ctx_400(1234)), (200, "")],
        stream=stream, tails=tails,
    )
    check(len(sent) == 2, f"[{label}] exactly two upstream calls: the original + one retry")
    check(r.status_code == 200, f"[{label}] the caller sees a normal 200")
    if stream:
        chunks = _sse_chunks(r)
        text = _assistant_text(chunks)
        check(text == "ok", f"[{label}] the client received ONLY the retry's real content ({text!r})")
        check(
            main._REJECTED_PREAMBLE not in r.text,
            f"[{label}] no failure text ever reached the client — the first "
            f"attempt's rejection was never yielded",
        )
    else:
        check(
            r.json().get("choices", [{}])[0].get("message", {}).get("content") == "ok",
            f"[{label}] the caller's JSON body is the retry's real reply",
        )
    check(main._BUDGET_MARGIN > 0, f"[{label}] the calibration learned from the surprise")
    check(
        any("re-shedding against the just-widened budget margin" in rc.getMessage()
            for rc in records),
        f"[{label}] the retry is logged, not silent",
    )
    check(len(tails) == 1, f"[{label}] the memory tail ran exactly ONCE for this turn")


# ===========================================================================
print("\n[5] item 2 — exactly ONE retry, even when the retry ALSO fails and "
      "ALSO would have taught the calibration something")
# ===========================================================================
for stream in (False, True):
    label = "stream" if stream else "non-stream"
    _reset_margin()
    # The SECOND body reports an even bigger true count (2000) than the
    # first (1234) -- by the raw arithmetic this WOULD widen the margin
    # again (tightened=True a second time), so if the cap were only
    # `tightened`-gated instead of `_attempt == 0`-gated this would retry
    # forever on a stub that keeps moving the goalposts. It must not.
    r, sent, records = _post(
        [user("hello")], f"retry-cap-{label}",
        script=[(400, ctx_400(1234)), (400, ctx_400(2000))],
        stream=stream,
    )
    check(len(sent) == 2, f"[{label}] exactly two upstream calls, never three")
    check(
        sum(1 for rc in records
            if "re-shedding against the just-widened budget margin" in rc.getMessage()) == 1,
        f"[{label}] the retry log line appears exactly once",
    )
    if stream:
        check(r.status_code == 200, f"[{label}] SSE still opens 200")
        text = _assistant_text(_sse_chunks(r))
        check(main._REJECTED_PREAMBLE in text, f"[{label}] the honest failure reaches the client")
    else:
        check(r.status_code == 400, f"[{label}] the caller gets vLLM's real 400, relayed")


# ===========================================================================
print("\n[6] item 2 — CONTROLS: no retry when it could not help")
# ===========================================================================
_reset_margin()
r, sent, _records = _post(
    [user("hello")], "no-retry-not-context", script=[(400, OTHER_400), (200, "")],
    stream=True,
)
check(len(sent) == 1, "a non-size 400 is not retried (re-shedding cannot fix a role error)")

main._BUDGET_MARGIN = main.MAX_MODEL_LEN // 4  # already at the cap
r, sent, _records = _post(
    [user("hello")], "no-retry-not-tightened",
    script=[(400, ctx_400(1234)), (200, "")], stream=True,
)
check(len(sent) == 1,
      "a rejection that teaches the calibration nothing (margin already "
      "capped) is not retried — resending the SAME array would fail identically")
_reset_margin()


# ===========================================================================
print("\n[7] item 3 — the margin persists across a restart")
# ===========================================================================
check(main._BUDGET_MARGIN_STATE_PATH.parent == memory.storage_root(),
      "fixture: the margin file lives under this process's storage root")

_reset_margin()
# <= MAX_MODEL_LEN // 4 (250 with this file's MAX_MODEL_LEN=1000), or
# _load_budget_margin's own documented clamp would legitimately cut it down
# and this "round trip" check would be testing the clamp instead of the
# round trip.
main._save_budget_margin(200)
check(main._load_budget_margin() == 200, "round trip: save then load returns the same value")

# "unreadable means start clean" -- the same contract facts.py/persona.py use.
main._BUDGET_MARGIN_STATE_PATH.write_text("{not json", encoding="utf-8")
check(main._load_budget_margin() == 0, "a corrupt file loads as 0, not a crash")

memory.atomic_write_json(main._BUDGET_MARGIN_STATE_PATH, {"margin": True})
check(main._load_budget_margin() == 0,
      "a bool is not accepted as a margin (bool is an int subclass in Python)")

memory.atomic_write_json(main._BUDGET_MARGIN_STATE_PATH, {"margin": -5})
check(main._load_budget_margin() == 0, "a negative value loads as 0")

memory.atomic_write_json(main._BUDGET_MARGIN_STATE_PATH, {"margin": "lots"})
check(main._load_budget_margin() == 0, "a non-numeric value loads as 0")

memory.atomic_write_json(main._BUDGET_MARGIN_STATE_PATH,
                          {"margin": main.MAX_MODEL_LEN * 10})
check(main._load_budget_margin() == main.MAX_MODEL_LEN // 4,
      "an oversized stale value is clamped to MAX_MODEL_LEN // 4, not trusted")

if not main._BUDGET_MARGIN_STATE_PATH.exists():
    pass
main._BUDGET_MARGIN_STATE_PATH.unlink(missing_ok=True)
check(main._load_budget_margin() == 0, "an absent file loads as 0 (a fresh volume)")


def _read_margin_file():
    return memory.read_json(main._BUDGET_MARGIN_STATE_PATH, default=None)


print("\n[7b] item 3 — the two mutation points actually write to disk")
_reset_margin()
main._BUDGET_MARGIN_STATE_PATH.unlink(missing_ok=True)
tightened = main._note_backend_rejection(
    ctx_400(2500), main.HARD_INPUT_LIMIT, guard_measured_overflow=False,
)
check(tightened, "fixture: a genuine undercount widened the margin")
on_disk = _read_margin_file()
check(on_disk is not None and on_disk.get("margin") == main._BUDGET_MARGIN,
      f"the widening was persisted immediately ({on_disk} vs in-memory "
      f"{main._BUDGET_MARGIN})")

main._budget_ok_streak = main.BUDGET_MARGIN_RELEASE_AFTER - 1
main._note_backend_accepted()
on_disk2 = _read_margin_file()
check(on_disk2 is not None and on_disk2.get("margin") == main._BUDGET_MARGIN,
      f"the release was ALSO persisted ({on_disk2} vs in-memory "
      f"{main._BUDGET_MARGIN})")
_reset_margin()
main._save_budget_margin(0)


print("\n[7c] item 3 — a fresh process reloads the persisted margin at import")
_subprocess_probe = r"""
import os, sys, json
os.environ.setdefault("MODEL_REPO", "")
sys.path.insert(0, {path!r})
import main
print(main._BUDGET_MARGIN)
"""


def _import_margin_in_subprocess(storage_root, margin_value=None):
    env = dict(os.environ)
    env["COMPACTOR_STORAGE_ROOT"] = storage_root
    env["MAX_MODEL_LEN"] = "1000"
    env.pop("MODEL_REPO", None)
    os.makedirs(storage_root, exist_ok=True)
    if margin_value is not None:
        with open(os.path.join(storage_root, "budget_margin.json"), "w",
                   encoding="utf-8") as f:
            json.dump({"margin": margin_value}, f)
    script = _subprocess_probe.format(path=os.path.dirname(os.path.abspath(__file__)))
    proc = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True,
        timeout=60,
    )
    return proc


_sub_root = tempfile.mkdtemp(prefix="compactor-test-v3196-sub-")
try:
    # <= MAX_MODEL_LEN // 4 (250, matching the subprocess's own
    # MAX_MODEL_LEN=1000) so this proves the reload, not the clamp.
    proc = _import_margin_in_subprocess(_sub_root, margin_value=123)
    check(proc.returncode == 0,
          f"subprocess import ran cleanly (stderr: {proc.stderr[-400:]!r})")
    check(proc.stdout.strip() == "123",
          f"a persisted margin is reloaded at import (got {proc.stdout.strip()!r})")
finally:
    shutil.rmtree(_sub_root, ignore_errors=True)

_sub_root2 = tempfile.mkdtemp(prefix="compactor-test-v3196-sub2-")
try:
    proc2 = _import_margin_in_subprocess(_sub_root2, margin_value=None)
    check(proc2.returncode == 0,
          f"subprocess import ran cleanly with no state file (stderr: "
          f"{proc2.stderr[-400:]!r})")
    check(proc2.stdout.strip() == "0",
          f"a fresh volume with no state file imports at margin 0 (got "
          f"{proc2.stdout.strip()!r})")
finally:
    shutil.rmtree(_sub_root2, ignore_errors=True)


# ===========================================================================
print("\n[8] mutation checks — break the fix on a COPY, confirm these tests "
      "go red for the right reason")
# ===========================================================================

# --- Mutation 1a: item 1's REAL refuse condition (_fit_request_for_vllm)
# disabled -- run against the production-scale CONTROL from section [3c]
# (input alone over MAX_MODEL_LEN), the exact shape that must never reach
# vLLM.
_dst1a = _mutated_copy(
    "    if measured + floor + _BUDGET_MARGIN > MAX_MODEL_LEN:",
    "    if False:  # MUTANT",
)
_env_1a = dict(_prod_env)
_env_1a["COMPACTOR_MIN_GENERATION_FLOOR"] = "256"
_res1a = _run_probe_against(
    _dst1a, _CONTROL_PROBE.replace("__REQ_MAX_TOKENS__", "None"),
    env_overrides=_env_1a,
)
check(
    _res1a.returncode == 0 and "STATUS 200" in _res1a.stdout
    and "CALLS 1" in _res1a.stdout,
    "MUTATION 1a: disabling _fit_request_for_vllm's refuse condition makes "
    "the endpoint forward the CONTROL case (33,000 measured tokens alone, "
    "over MAX_MODEL_LEN=32768) to the stub instead of refusing it — this "
    "test's own [3c] checks ('CONTROL ... refused before the call', "
    "'vLLM was never called') are exactly what would have caught this in "
    f"the real suite (probe stdout: {_res1a.stdout.strip()!r}, stderr "
    f"tail: {_res1a.stderr[-300:]!r})"
)
shutil.rmtree(_dst1a, ignore_errors=True)

# --- Mutation 1b: the max_tokens CLAMP disabled -- run against the
# production-scale clamp case from section [3b] (measured=23200,
# max_tokens=15000 requested, MAX_MODEL_LEN=32768: 23200+15000=38200 would
# be a guaranteed vLLM 400 if sent unclamped).
_dst1b = _mutated_copy("        if requested > room:", "        if False:  # MUTANT")
_env_1b = dict(_prod_env)
_env_1b["COMPACTOR_MIN_GENERATION_FLOOR"] = "256"
_res1b = _run_probe_against(
    _dst1b, _PROD_PROBE.replace("__REQ_MAX_TOKENS__", "15000"),
    env_overrides=_env_1b,
)
_unclamped = "SENT_MAX_TOKENS 15000" in _res1b.stdout
check(
    _res1b.returncode == 0 and _unclamped,
    "MUTATION 1b: disabling the clamp condition sends max_tokens=15000 "
    "UNCLAMPED beside a 23,200-token prompt against a 32,768-token window "
    "(38,200 total -- a guaranteed vLLM 400 by vLLM's own token_num + "
    "max_tokens > max_model_len check) instead of clamping it to 9,568 -- "
    "this test's own [3b] check ('max_tokens 15000 -> 9568 ... not sent "
    f"unclamped') is what would have caught this (probe stdout: "
    f"{_res1b.stdout.strip()!r}, stderr tail: {_res1b.stderr[-300:]!r})"
)
shutil.rmtree(_dst1b, ignore_errors=True)

# --- Mutation 1c: the inversion fix (_serve_floor's own `min`) removed --
# run against the [3d-ii] scenario (GENERATION_RESERVE=200 below the
# shipped MIN_GENERATION_FLOOR default of 256): a payload the guard
# reports as fitting its own policy limit (800) must not be refused by a
# flat 256-token floor that would put the serve limit (744) BELOW it.
_dst1c = _mutated_copy(
    "    return min(MIN_GENERATION_FLOOR, MAX_MODEL_LEN - effective_limit)",
    "    return MIN_GENERATION_FLOOR  # MUTANT: the old flat floor, inversion restored",
)
_res1c = _run_probe_against(
    _dst1c, _inv_probe,
    env_overrides={
        "MAX_MODEL_LEN": "1000",
        "COMPACTOR_GENERATION_RESERVE": "200",
        "COMPACTOR_MIN_GENERATION_FLOOR": "256",
    },
)
check(
    _res1c.returncode == 0 and "STATUS 400" in _res1c.stdout
    and "CALLS 0" in _res1c.stdout,
    "MUTATION 1c: reverting _serve_floor to the flat MIN_GENERATION_FLOOR "
    "(no min() against the policy reserve) re-inverts the two limits and "
    "refuses a payload the guard just certified as fitting its own policy "
    "target (measured 790, under the 800-token policy limit; the mutant "
    "flat floor of 256 makes 790+256=1046 exceed MAX_MODEL_LEN=1000) -- "
    "this test's own [3d-ii] check ('forwarded, not refused by a floor that "
    f"would otherwise sit BELOW the policy limit') is what would have "
    f"caught this (probe stdout: {_res1c.stdout.strip()!r}, stderr tail: "
    f"{_res1c.stderr[-300:]!r})"
)
shutil.rmtree(_dst1c, ignore_errors=True)

# --- Mutation 2: item 2's one-retry cap removed. TWO cooperating lines
# enforce "exactly one retry" in the streaming path: the loop's own
# `for _attempt in range(2):` bound (a `continue` on the LAST index of a
# `range(2)` loop just ends the loop, same as falling through — so this
# alone already caps total calls at 2) AND the retry-eligibility check's
# `_attempt == 0` clause. Widening the range ALONE proved nothing in an
# earlier draft of this test (the eligibility check still refused to retry
# on attempt 1, so the call count stayed at 2 regardless) — which is
# reassuring about the real code's defense in depth, but means demonstrating
# what the SECOND line protects against requires loosening both together.
# Each needle's exact indentation is what makes it match only the streaming
# occurrence, not the non-streaming path's own copies of both lines a few
# hundred lines down.
_dst2 = _mutated_copy([
    (
        "                    for _attempt in range(2):",
        "                    for _attempt in range(10):",
        1,
    ),
    (
        "                                if (\n"
        "                                    _attempt == 0\n"
        "                                    and r.status_code < 500\n"
        "                                    and _is_context_overflow(err_body)\n"
        "                                    and tightened\n"
        "                                ):",
        "                                if (\n"
        "                                    True\n"
        "                                    and r.status_code < 500\n"
        "                                    and _is_context_overflow(err_body)\n"
        "                                    and tightened\n"
        "                                ):",
        1,
    ),
])
_probe2 = r"""
import json
import main
from fastapi.testclient import TestClient
from unittest.mock import patch
client = TestClient(main.app, client=("127.0.0.1", 12345), raise_server_exceptions=False)

def ctx_400(n):
    return ('{"error":{"message":"This model\'s maximum context length is '
            '2000000 tokens. However, you requested 0 output tokens and your '
            'prompt contains ' + str(n) + ' input tokens, for a total of ... '
            '(parameter=input_tokens)"}}')

class _StreamResp:
    def __init__(self, n): self.status_code = 400; self._n = n
    async def aread(self): return ctx_400(self._n).encode()

class _CM:
    def __init__(self, r): self._r = r
    async def __aenter__(self): return self._r
    async def __aexit__(self, *a): return False

class _Client:
    sent = 0
    def __init__(self, *a, **k): pass
    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False
    async def aclose(self): pass
    def stream(self, method, url, json=None, **kw):
        _Client.sent += 1
        # actual size keeps growing every call, so an UNCAPPED retry loop
        # never converges and never stops on its own -- only the two
        # mutated-away lines used to stop it. The window is large (2M/1
        # here) so _BUDGET_MARGIN's own MAX_MODEL_LEN // 4 ceiling (500k)
        # is nowhere near reached by these numbers -- growth alone must
        # not be the thing that eventually halts this loop.
        return _CM(_StreamResp(1_999_800 + _Client.sent * 1000))

with patch.object(main.httpx, "AsyncClient", _Client), \
     patch.object(main, "_fire_and_forget", lambda coro, label=None: coro.close()):
    r = client.post("/v1/chat/completions",
                     json={"model": "m", "messages": [{"role": "user", "content": "hi"}],
                           "stream": True},
                     headers={"X-Conversation-Id": "mutant-2"})
print("CALLS", _Client.sent)
"""
_res2 = _run_probe_against(
    _dst2, _probe2,
    env_overrides={"MAX_MODEL_LEN": "2000000", "COMPACTOR_GENERATION_RESERVE": "200"},
)
import re as _re_mod
_m2 = _re_mod.search(r"CALLS (\d+)", _res2.stdout)
_calls2 = int(_m2.group(1)) if _m2 else -1
check(_res2.returncode == 0 and _calls2 > 2,
      "MUTATION 2: widening the streaming attempt budget AND removing the "
      "`_attempt == 0` gate together, on an ever-growing rejection, makes "
      "the retry loop keep going past the ONE-retry budget "
      f"({_calls2} upstream call(s), expected exactly 2 with the real fix) "
      "— this test's own [5] check 'exactly two upstream calls, never "
      f"three' is what would have caught this (stderr tail: "
      f"{_res2.stderr[-300:]!r})")
shutil.rmtree(_dst2, ignore_errors=True)

# --- Mutation 3: item 3's persistence load is short-circuited to 0 ---
_dst3 = _mutated_copy(
    "    return min(v, MAX_MODEL_LEN // 4)",
    "    return 0  # MUTANT",
    must_count=1,
)
_probe3 = r"""
import main
main._save_budget_margin(555)
print("LOADED", main._load_budget_margin())
"""
_res3 = _run_probe_against(_dst3, _probe3)
_caught3 = "LOADED 0" in _res3.stdout
check(_res3.returncode == 0 and _caught3,
      "MUTATION 3: short-circuiting _load_budget_margin to always return 0 "
      "makes a saved margin of 555 read back as 0 — this test's own [7] "
      "'round trip: save then load returns the same value' check is what "
      f"would have caught this (probe stdout: {_res3.stdout.strip()!r})")
shutil.rmtree(_dst3, ignore_errors=True)


print()
if _FAILED:
    print("SOME v3196-overflow CHECKS FAILED")
    sys.exit(1)
print("All v3196-overflow checks passed.")
shutil.rmtree(_TMP_ROOT, ignore_errors=True)
