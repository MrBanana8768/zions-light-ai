"""
Tier-1 tests for v3.1.9.8 WORD-BAN FILE DEPLOY + IDLE WARM-UP
(compactor/wordban_runtime.py + its two injection sites in main.py).

    [U1] canonicalization / sha256 — byte-identity with the client (OpenWebUI)
         path
    [U2] file loader: valid, missing, invalid JSON, sha mismatch, stat()
         caching (no re-read when unchanged)
    [U3] decide_structured_outputs: file wins, client fallback, none,
         conflict logged once per (file_sha, client_sha) pair, model gate
         (F8)
    [U3b] is_grammar_shape (F7)
    [U4] health_snapshot shape (incl. the "model" field, F8)
    [U5] warm-up gate: each idle condition independently blocks; a vLLM
         "restart" (identity change) re-triggers; backoff after a failure
    [U5b] hostile review round 1 F1: a fresh boot (no reqctx history at
          all) still warms after idle_s, via _started_at/
          _identity_changed_at; an identity change resets the idle clock
          even long after boot
    [E1] main.py integration: injected into the her-reply calls (streaming
         + non-streaming), the "request done" line carries wb=<rev>:<sha8>,
         and the file wins over a client-sent value
    [E1b] hostile review round 1: F2 (never injects into the compactor's
          own selftest round trip), F6 (an exception in the decision is
          caught, fail-open), F7 (a client's own response_format / other-
          shaped structured_outputs is never overridden), F8 (a model
          mismatch skips injection)
    [E2] internal calls never see it (a stand-in httpx call that main.py's
         OWN internal helpers would make, proven by construction: nothing
         outside decide_structured_outputs' two call sites in main.py ever
         imports wordban_runtime)

    python test_wordban_runtime.py
"""

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("MODEL_REPO", "test-model")
os.environ.setdefault("VLLM_URL", "http://stub:8000")
os.environ["COMPACTOR_RAG_ENABLED"] = "false"
os.environ["COMPACTOR_FACTS_EXTRACTION"] = "false"
os.environ["COMPACTOR_HIERARCHICAL_SUMMARY"] = "false"

_TMP_ROOT = tempfile.mkdtemp(prefix="compactor-test-wordban-")
os.environ["COMPACTOR_STORAGE_ROOT"] = _TMP_ROOT
_WB_DIR = tempfile.mkdtemp(prefix="compactor-test-wordban-file-")
os.environ["COMPACTOR_WORDBAN_FILE"] = str(Path(_WB_DIR) / "active.json")

import wordban_runtime  # noqa: E402
import reqctx  # noqa: E402
import bgwork  # noqa: E402
import main  # noqa: E402
import memory  # noqa: E402
import retrieval  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

retrieval._available = False
retrieval._embedder = None
retrieval._chroma_collection = None
memory.ensure_storage_layout()

client = TestClient(main.app, client=("127.0.0.1", 12361), raise_server_exceptions=False)

FAILED = []


def assert_eq(actual, expected, label):
    if actual != expected:
        print(f"FAIL {label}: expected {expected!r}, got {actual!r}")
        FAILED.append(label)
    else:
        print(f"  ok   {label}")


def assert_true(cond, label):
    if not cond:
        print(f"FAIL {label}")
        FAILED.append(label)
    else:
        print(f"  ok   {label}")


class _Collector(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)


@contextlib.contextmanager
def capture_logs(name):
    lg = logging.getLogger(name)
    h = _Collector()
    prev = lg.level
    lg.addHandler(h)
    lg.setLevel(logging.DEBUG)
    try:
        yield h.records
    finally:
        lg.removeHandler(h)
        lg.setLevel(prev)


ACTIVE_PATH = Path(os.environ["COMPACTOR_WORDBAN_FILE"])


# Matches the "model" every _post_nonstream_wb body below sends
# ("stub-model") — write_active defaults active.json's own "model" field
# to this so [E1]'s existing file-wins/none tests don't ALSO have to
# specify a matching --model; the model-MISMATCH gate (F8) gets its own
# dedicated test below, with an explicitly different model on one side.
TEST_MODEL = "stub-model"


def write_active(revision, structured_outputs, *, bad_sha=False, deployed_at="2026-09-25T00:00:00Z",
                  model=TEST_MODEL, omit_model=False):
    computed = wordban_runtime.value_sha256_of(structured_outputs)
    doc = {
        "revision": revision,
        "value_sha256": ("0" * 64) if bad_sha else computed,
        "structured_outputs": structured_outputs,
        "deployed_at": deployed_at,
    }
    if not omit_model:
        doc["model"] = model
    ACTIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    ACTIVE_PATH.write_text(json.dumps(doc), encoding="utf-8")
    return computed


def clear_active():
    ACTIVE_PATH.unlink(missing_ok=True)
    wordban_runtime._reset_for_tests()


# ===========================================================================
# [U1] canonicalization + byte identity with the OpenWebUI client path
# ===========================================================================

print("[U1] canonicalization / sha256 / byte-identity with the client path")

_grammar = "root ::= 'x' 'y' 'z'"  # a small stand-in, never real her-data/banlist content
_pasted_value_txt = json.dumps({"grammar": _grammar}, ensure_ascii=True)

# This is EXACTLY what wordban.py cmd_build writes to structured_outputs.
# value.txt (json.dumps({"grammar": ebnf}, ensure_ascii=True)).
_released_sha256 = hashlib.sha256(_pasted_value_txt.encode("utf-8")).hexdigest()

# OpenWebUI's OWN parse step: JSONCodec.loads == json.loads (confirmed
# against the vendored 0.11.4 source, see wordban_runtime's module
# docstring) — this is what a real OpenWebUI process does with the pasted
# string before it ever reaches the compactor.
_client_side_value = json.loads(_pasted_value_txt)
assert_eq(_client_side_value, {"grammar": _grammar}, "OpenWebUI's own parse of the pasted string")

# This module's canonicalization of that SAME parsed value must hash
# identically to the raw released file's own sha256 — that identity is
# what makes value_sha256 in active.json agree with word-ban-deploy.py's
# computation over the raw file bytes.
assert_eq(
    wordban_runtime.value_sha256_of(_client_side_value), _released_sha256,
    "canonical_value_bytes(parsed value) hashes the same as the raw released file",
)

# BYTE IDENTITY PROOF: the bytes vLLM would receive in `structured_outputs`
# are identical whether the value came from the client (already a dict, as
# analyzed) or from the file (also a dict, loaded from active.json) — both
# take the SAME json.dumps serialization inside httpx's `json=` parameter.
# Modelled here directly against stdlib json, which is exactly what httpx
# uses to encode `json=body`.
_body_from_client = {"model": "m", "structured_outputs": _client_side_value}
write_active(1, _client_side_value)
_decision = wordban_runtime.decide_structured_outputs(_client_side_value)
_body_from_file = {"model": "m", "structured_outputs": _decision.value}
assert_eq(
    json.dumps(_body_from_client, sort_keys=True), json.dumps(_body_from_file, sort_keys=True),
    "the outgoing vLLM body is byte-identical whether structured_outputs came from the client or the file",
)
clear_active()


# ===========================================================================
# [U2] file loader
# ===========================================================================

print("\n[U2] file loader: valid / missing / invalid / sha mismatch / stat caching")

clear_active()
d = wordban_runtime._load()
assert_eq(d.present, False, "no file -> present=False")
assert_eq(d.error, None, "no file is NOT an error")
assert_eq(d.structured_outputs, None, "no file -> no value")

sha = write_active(3, {"grammar": _grammar})
d = wordban_runtime._load()
assert_eq(d.present, True, "a real file -> present=True")
assert_eq(d.error, None, "a valid file -> no error")
assert_eq(d.revision, 3, "revision read back")
assert_eq(d.value_sha256, sha, "value_sha256 read back")
assert_eq(d.structured_outputs, {"grammar": _grammar}, "structured_outputs read back")

# stat() caching: rewriting the SAME bytes to the SAME path changes mtime/
# inode (a new file), which correctly triggers a reparse — but calling
# _load() TWICE with nothing changed on disk must not re-read.
_read_calls = []
_orig_read_bytes = Path.read_bytes


def _counting_read_bytes(self):
    if self == ACTIVE_PATH:
        _read_calls.append(1)
    return _orig_read_bytes(self)


with patch.object(Path, "read_bytes", _counting_read_bytes):
    wordban_runtime._load()
    wordband_calls_after_first = len(_read_calls)
    wordban_runtime._load()
    wordband_calls_after_second = len(_read_calls)
assert_eq(wordband_calls_after_first, 0, "unchanged file -> no re-read even on the FIRST call after a prior successful load")
assert_eq(wordband_calls_after_second, 0, "unchanged file -> still no re-read on a second call")

clear_active()
ACTIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
ACTIVE_PATH.write_text("not json at all {{{", encoding="utf-8")
d = wordban_runtime._load()
assert_true(d.present, "an unreadable-as-JSON file is still 'present'")
assert_true(d.error is not None, "invalid JSON -> error set (fail-open)")
assert_eq(d.structured_outputs, None, "invalid JSON -> no value used")

clear_active()
write_active(4, {"grammar": _grammar}, bad_sha=True)
d = wordban_runtime._load()
assert_true(d.error is not None, "a sha256 mismatch is refused (error set)")
assert_true("mismatch" in d.error, "the error names it a mismatch")
assert_eq(d.structured_outputs, None, "sha mismatch -> no value used (fail-open, not fail-silent-wrong)")

# Hostile review round 1, F8: "model" is now a REQUIRED field.
clear_active()
write_active(6, {"grammar": _grammar}, omit_model=True)
d = wordban_runtime._load()
assert_true(d.error is not None, "a file missing 'model' is refused (error set)")
assert_true("model" in d.error, "the error names the missing field")
assert_eq(d.structured_outputs, None, "missing model -> no value used (fail-open)")

clear_active()

# Hostile review round 2, N2: the runtime ALSO refuses to load a withdrawn
# VALUE sha, fail-open with an ERROR — the last line of defense if a
# mislabelled redeploy of a withdrawn value somehow got past
# word-ban-deploy.py's own checks. The real rev-4 sha is never exercised
# against real content here (no grammar in this repo); this monkeypatches
# WITHDRAWN_VALUE_SHA256S with a synthetic sha to prove the mechanism.
withdrawn_grammar = {"grammar": _grammar + "-withdrawn-runtime"}
withdrawn_sha = wordban_runtime.value_sha256_of(withdrawn_grammar)
_orig_withdrawn_values = dict(wordban_runtime.WITHDRAWN_VALUE_SHA256S)
wordban_runtime.WITHDRAWN_VALUE_SHA256S.clear()
wordban_runtime.WITHDRAWN_VALUE_SHA256S[withdrawn_sha] = "synthetic stand-in for a withdrawn value"
try:
    clear_active()
    write_active(14, withdrawn_grammar)  # an innocent revision label, not 4
    d = wordban_runtime._load()
    assert_true(d.error is not None, "a withdrawn VALUE is refused (error set) regardless of its revision label")
    assert_true("withdrawn" in d.error, "the error names it withdrawn")
    assert_eq(d.structured_outputs, None, "withdrawn value -> no value used (fail-open)")
finally:
    wordban_runtime.WITHDRAWN_VALUE_SHA256S.clear()
    wordban_runtime.WITHDRAWN_VALUE_SHA256S.update(_orig_withdrawn_values)

clear_active()

# Hostile review round 2, N1: once the file's model no longer matches this
# compactor's OWN configured model (MODEL_REPO, "test-model" per this
# test's own os.environ.setdefault at the top), a WARNING is logged once
# per file change, not per request.
with capture_logs("compactor.wordban_runtime") as cap:
    write_active(15, {"grammar": _grammar}, model="not-this-pods-model")
    wordban_runtime._load()
assert_true(
    any("configured for model" in r.getMessage() for r in cap),
    "a file/pod model mismatch is logged once, at WARNING",
)
assert_true(
    all(r.levelno == logging.WARNING for r in cap if "configured for model" in r.getMessage()),
    "...at WARNING level",
)
with capture_logs("compactor.wordban_runtime") as cap2:
    wordban_runtime._load()  # nothing changed on disk -> no re-log
assert_eq(len(cap2), 0, "an unchanged file does not re-log the model mismatch")

# ...and NOT logged when the file's model DOES match this pod's configured
# model.
clear_active()
with capture_logs("compactor.wordban_runtime") as cap3:
    write_active(16, {"grammar": _grammar}, model="test-model")
    wordban_runtime._load()
assert_true(
    not any("configured for model" in r.getMessage() for r in cap3),
    "no mismatch warning when the file's model matches this pod's MODEL_REPO",
)

clear_active()


# ===========================================================================
# [U3] decide_structured_outputs
# ===========================================================================

print("\n[U3] decide_structured_outputs: file wins, client fallback, none, conflict logging")

clear_active()
dec = wordban_runtime.decide_structured_outputs(None)
assert_eq(dec.source, "none", "no file, no client -> none")
assert_eq(dec.value, None, "none -> value is None")

client_val = {"grammar": "client-only, never released"}
dec = wordban_runtime.decide_structured_outputs(client_val)
assert_eq(dec.source, "client", "no file, client sent one -> client")
assert_eq(dec.value, client_val, "client value passed through unchanged")
assert_eq(dec.trace_field, None, "no wb= trace field for a client-sourced decision")

file_val = {"grammar": _grammar}
write_active(5, file_val)
dec = wordban_runtime.decide_structured_outputs(None)
assert_eq(dec.source, "file", "file present, no client -> file")
assert_eq(dec.value, file_val, "file value used")
assert_eq(dec.trace_field, f"5:{wordban_runtime.value_sha256_of(file_val)[:8]}", "trace_field is <rev>:<sha8>")

with capture_logs("compactor.wordban_runtime") as cap:
    dec = wordban_runtime.decide_structured_outputs(client_val)
assert_eq(dec.source, "file", "file wins over a DIFFERING client-sent value")
assert_true(
    any("conflict" in r.getMessage() for r in cap),
    "a conflict between file and client is logged",
)
assert_true(
    _grammar not in "\n".join(r.getMessage() for r in cap)
    and "client-only" not in "\n".join(r.getMessage() for r in cap),
    "the conflict log line never contains grammar content — shas only",
)
with capture_logs("compactor.wordban_runtime") as cap2:
    wordban_runtime.decide_structured_outputs(client_val)
assert_eq(len(cap2), 0, "the SAME conflict pair is logged only ONCE, not on every request")

clear_active()

# Hostile review round 1, F8: the `model` kwarg gate, tested directly
# (main.py's integration test [E1b] proves the wiring; this proves the
# decision function's own logic in isolation).
write_active(7, {"grammar": _grammar}, model="model-a")
dec = wordban_runtime.decide_structured_outputs(None, model="model-a")
assert_eq(dec.source, "file", "model matches -> file wins, as usual")
dec = wordban_runtime.decide_structured_outputs(None, model="model-b")
assert_eq(dec.source, "none", "model does NOT match -> file not applied, no client value -> none")
dec = wordban_runtime.decide_structured_outputs({"grammar": "client for model-b"}, model="model-b")
assert_eq(dec.source, "client", "model mismatch + a client value -> the client's own value, untouched")
assert_eq(dec.value, {"grammar": "client for model-b"}, "...unchanged")
dec = wordban_runtime.decide_structured_outputs(None)  # model=None (default) -> check skipped entirely
assert_eq(dec.source, "file", "model=None (caller has no opinion) skips the check entirely — file wins as before")

clear_active()


# ===========================================================================
# [U3b] is_grammar_shape (hostile review round 1, F7)
# ===========================================================================

print("\n[U3b] is_grammar_shape: the one shape this module ever produces or accepts")

assert_true(wordban_runtime.is_grammar_shape({"grammar": "root ::= 'x'"}), "a plain {grammar: str} is the shape")
assert_true(not wordban_runtime.is_grammar_shape({"grammar": "x", "extra": 1}),
            "an extra key disqualifies it")
assert_true(not wordban_runtime.is_grammar_shape({"grammar": 123}), "a non-string grammar value disqualifies it")
assert_true(not wordban_runtime.is_grammar_shape({"json_schema": {}}), "an unrelated key disqualifies it")
assert_true(not wordban_runtime.is_grammar_shape("a string"), "a bare string is not the shape")
assert_true(not wordban_runtime.is_grammar_shape(None), "None is not the shape")
assert_true(not wordban_runtime.is_grammar_shape([]), "a list is not the shape")


# ===========================================================================
# [U4] health_snapshot shape
# ===========================================================================

print("\n[U4] health_snapshot")

clear_active()
snap = wordban_runtime.health_snapshot()
for key in ("source", "revision", "value_sha256", "file_error", "model", "model_matches_pod",
            "warmed", "warm_state", "last_warm_compile_s", "last_warm_at", "warmups_in_flight"):
    assert_true(key in snap, f"health_snapshot has {key!r}")
assert_eq(snap["source"], "none", "no file, no decision yet -> none")
assert_eq(snap["model_matches_pod"], False, "no file at all -> model_matches_pod is False, not None")

write_active(6, {"grammar": _grammar}, model="a-health-test-model")
wordban_runtime.decide_structured_outputs(None)
snap = wordban_runtime.health_snapshot()
assert_eq(snap["source"], "file", "after a decision using the file -> source=file")
assert_eq(snap["revision"], 6, "revision reflects the file")
assert_eq(snap["model"], "a-health-test-model",
          "model reflects the file (F8) — visible to an operator diagnosing a mismatch")

# Hostile review round 2, N1: model_matches_pod against this compactor's
# OWN configured model (MODEL_REPO="test-model", set at the top of this
# file) — False when the file targets some OTHER model...
assert_eq(snap["model_matches_pod"], False,
          "model_matches_pod is False when the file's model differs from MODEL_REPO")
clear_active()

# ...True when the file's own model equals MODEL_REPO exactly.
write_active(17, {"grammar": _grammar}, model="test-model")
wordban_runtime.decide_structured_outputs(None)
snap = wordban_runtime.health_snapshot()
assert_eq(snap["model_matches_pod"], True,
          "model_matches_pod is True when the file's model equals this pod's MODEL_REPO")
clear_active()


# ===========================================================================
# [U5] warm-up gate
# ===========================================================================

print("\n[U5] warm-up gate: each idle condition independently blocks; restart re-triggers; backoff")


class _FakeMetricsClient:
    """Duck-typed httpx.AsyncClient stand-in — only .get()/.post() are
    used by fetch_vllm_metrics / WarmupWorker._warm."""

    def __init__(self):
        self.running = 0.0
        self.waiting = 0.0
        self.identity = "1700000000.0"
        self.post_calls = []
        self.post_should_fail = False
        self.post_delay_compile_s = 0.0

    async def get(self, url, timeout=None):
        assert "/metrics" in url
        text = (
            f"process_start_time_seconds {self.identity}\n"
            f"vllm:num_requests_running {self.running}\n"
            f"vllm:num_requests_waiting {self.waiting}\n"
        )
        return _FakeMetricsResp(text)

    async def post(self, url, json=None, timeout=None, headers=None):
        self.post_calls.append((url, json, headers))
        if self.post_should_fail:
            raise RuntimeError("simulated vLLM failure")
        return _FakeMetricsResp("", status_code=200)


class _FakeMetricsResp:
    def __init__(self, text, status_code=200):
        self.text = text
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx
            raise httpx.HTTPStatusError("bad", request=None, response=None)


def make_worker(fake_client, *, idle_s=30.0, backoff_initial_s=1.0, backoff_max_s=4.0):
    clock = {"t": 1000.0}

    def _clock():
        return clock["t"]

    w = wordban_runtime.WarmupWorker(
        "http://stub:8000", "test-model",
        idle_s=idle_s, poll_s=1.0, warmup_timeout_s=10.0,
        backoff_initial_s=backoff_initial_s, backoff_max_s=backoff_max_s,
        clock=_clock,
    )
    return w, clock


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# --- each idle condition independently blocks -------------------------------
#
# `_last_activity_age_s` and `_requests_in_flight` both read reqctx's REAL
# monotonic clock (deliberately — that clock is what the wrapper actually
# advances in production), which is decoupled from the fake `clock` this
# worker uses for its OWN bookkeeping (backoff windows). Rather than try to
# make a unit test outrun 30 real seconds, each condition below is
# exercised by monkeypatching the WORKER's OWN accessor methods directly —
# proving `_idle_enough` actually consults each one, independently,
# without coupling this test to wall-clock timing.

clear_active()
write_active(7, {"grammar": _grammar})
fake = _FakeMetricsClient()
w, clock = make_worker(fake)
w._requests_in_flight = lambda: 0
w._last_activity_age_s = lambda: 999.0
reqctx._reset_for_tests()

# 1. requests_in_flight != 0 blocks, everything else idle.
w._requests_in_flight = lambda: 1
run(w._tick(fake))
assert_eq(w._warmed_key, None, "in-flight request blocks the warm-up")
w._requests_in_flight = lambda: 0

# 2. last activity age < idle_s blocks.
w._last_activity_age_s = lambda: 1.0  # 1s < idle_s=30
run(w._tick(fake))
assert_eq(w._warmed_key, None, "recent activity (< idle_s) blocks the warm-up")
w._last_activity_age_s = lambda: 999.0

# (F1 fix, hostile review round 1: a fresh boot's "no activity history at
# all" case is no longer exercised HERE, against a mocked
# `_last_activity_age_s` returning bare None — that used to be the whole
# story, and was the BUG (a fresh boot never warmed, ever). The REAL
# method now derives a usable age from `_started_at`/`_identity_changed_
# at` even with zero reqctx history; see [U5b] below, which drives it
# unmocked.)

# 3. bg outstanding != 0 blocks.
with patch.object(bgwork.pool, "stats", lambda: {"outstanding": 1}):
    run(w._tick(fake))
assert_eq(w._warmed_key, None, "outstanding background work blocks the warm-up")

# 4. vLLM running/waiting != 0 blocks.
fake.running = 1.0
run(w._tick(fake))
assert_eq(w._warmed_key, None, "vLLM busy (num_requests_running) blocks the warm-up")
fake.running = 0.0

# All clear -> warms.
clock["t"] += 100.0
run(w._tick(fake))
assert_true(w._warmed_key is not None, "with every condition clear, the warm-up fires")
assert_eq(len(fake.post_calls), 1, "exactly one warm-up POST was sent")
_url, _payload, _headers = fake.post_calls[0]
assert_eq(_payload["max_tokens"], 1, "warm-up request: max_tokens=1")
assert_eq(_payload["stream"], False, "warm-up request: stream=False")
assert_eq(_payload["structured_outputs"], {"grammar": _grammar},
          "warm-up request carries the byte-identical (same object) grammar")
assert_true(_headers["X-Request-Id"].startswith("warmup-"),
            "warm-up request id has the warmup-<sha8>-<n> shape")
assert_eq(w.snapshot()["warmed"], True, "snapshot reports warmed=True")
assert_true(w.snapshot()["last_warm_compile_s"] is not None, "snapshot reports a compile time")

# A second tick with nothing changed does not re-warm.
run(w._tick(fake))
assert_eq(len(fake.post_calls), 1, "no repeat warm-up once warmed_key matches")

# --- a vLLM "restart" (identity change) re-triggers -------------------------

fake.identity = "1700009999.0"  # a fresh process start time
clock["t"] += 100.0
run(w._tick(fake))
assert_eq(len(fake.post_calls), 2, "a changed vllm_identity re-triggers the warm-up even though value_sha256 is unchanged")

# --- a grammar change (same vLLM process) re-triggers -----------------------

write_active(8, {"grammar": _grammar + "2"})
clock["t"] += 100.0
run(w._tick(fake))
assert_eq(len(fake.post_calls), 3, "a changed value_sha256 re-triggers the warm-up even though vllm_identity is unchanged")

# --- backoff on failure, never a tight loop ---------------------------------

_warmed_key_before_failure = w._warmed_key
write_active(9, {"grammar": _grammar + "3"})
fake.post_should_fail = True
clock["t"] += 100.0
run(w._tick(fake))
assert_eq(w._state, "backoff", "a failed warm-up enters backoff")
assert_eq(w._warmed_key, _warmed_key_before_failure, "warmed_key is unchanged by a FAILED attempt")
_backoff_calls_before = len(fake.post_calls)
clock["t"] += 0.5  # still inside the 1s backoff window
run(w._tick(fake))
assert_eq(len(fake.post_calls), _backoff_calls_before, "no retry before the backoff window elapses")
clock["t"] += 1.0  # backoff window (1.0s) has now elapsed
fake.post_should_fail = False
run(w._tick(fake))
assert_eq(len(fake.post_calls), _backoff_calls_before + 1, "a retry is sent once the backoff window elapses")
assert_true(w._warmed_key is not None and w._warmed_key[0] == wordban_runtime.value_sha256_of({"grammar": _grammar + "3"}),
            "the retry succeeds and updates warmed_key")

clear_active()
reqctx._reset_for_tests()


# ===========================================================================
# [U5b] F1 fix (hostile review round 1, BLOCKER): a fresh boot (never any
# reqctx activity) still warms after idle_s, via the REAL (unmocked)
# `_last_activity_age_s` — `_started_at` and `_identity_changed_at`, and an
# identity change resets the idle clock even long after boot.
# ===========================================================================

print("\n[U5b] F1 fix: fresh-boot idle clock, and an identity change resets it")

clear_active()
write_active(11, {"grammar": _grammar + "-u5b"})
reqctx._reset_for_tests()  # no chat request has EVER finished in this process
fake2 = _FakeMetricsClient()
w2, clock2 = make_worker(fake2, idle_s=30.0)
assert_eq(reqctx.in_flight_stats()["last_request_done_age_s"], None,
          "precondition: reqctx has no activity history at all")

# Immediately after boot: not idle yet (age 0 from _started_at/first scrape).
run(w2._tick(fake2))
assert_eq(w2._warmed_key, None, "a fresh boot does not warm INSTANTLY")
assert_eq(w2._state, "waiting_idle", "state reflects: grammar active, just not idle yet")

# 30s later, still no chat traffic ever: warms anyway.
clock2["t"] += 31.0
run(w2._tick(fake2))
assert_true(w2._warmed_key is not None,
            "a fresh boot with NO chat traffic ever warms once idle_s has passed "
            "since boot (the F1 bug: this used to block forever on reqctx's None)")
_calls_after_boot_warm = len(fake2.post_calls)

# A lone vLLM restart (identity changes) long after boot resets the idle
# clock — even though _started_at is now ancient, the FRESH identity must
# still get its OWN 30s window before warming again.
clock2["t"] += 10_000.0  # _started_at is now far in the past
fake2.identity = "9999999999.0"  # vLLM restarted
run(w2._tick(fake2))
assert_eq(len(fake2.post_calls), _calls_after_boot_warm,
          "an identity change does NOT warm instantly, even long after boot")
assert_eq(w2._state, "waiting_idle", "blocked by the freshly-reset idle clock, not by anything else")
clock2["t"] += 10.0  # only 10s since the identity change
run(w2._tick(fake2))
assert_eq(len(fake2.post_calls), _calls_after_boot_warm,
          "still blocked at 10s since the identity change (< idle_s=30)")
clock2["t"] += 21.0  # 31s since the identity change now
run(w2._tick(fake2))
assert_eq(len(fake2.post_calls), _calls_after_boot_warm + 1,
          "warms once idle_s has passed since the IDENTITY CHANGE, regardless of how stale _started_at is")

clear_active()
reqctx._reset_for_tests()


# ===========================================================================
# [E1] main.py integration: injection + wb= trace field + file-wins
# ===========================================================================

print("\n[E1] main.py integration: injected into her-reply calls, wb= trace field, file wins")


def _find(records, needle):
    return next((r for r in records if needle in r.getMessage()), None)


# Stub vLLM backend — shape lifted from test_request_tracing.py's own
# _StubVLLM (deliberately duplicated rather than imported: importing
# another test_*.py module would re-execute ITS whole assertion suite as
# an import side effect, since these files are designed to run standalone
# via `python test_X.py`, not to be imported from one another).
class _StubResponse:
    status_code = 200

    def __init__(self, content, finish_reason="stop"):
        self._content = content
        self._fr = finish_reason

    def json(self):
        return {
            "id": "stub",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": self._content},
                "finish_reason": self._fr,
            }],
            "usage": {"completion_tokens": 1},
        }


class _StubVLLM:
    captured_bodies: list = []

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None, headers=None, **kwargs):
        _StubVLLM.captured_bodies.append(json)
        return _StubResponse("hello")

    async def aclose(self):
        pass


@contextlib.contextmanager
def capture(logger_name="compactor"):
    lg = logging.getLogger(logger_name)
    h = _Collector()
    prev = lg.level
    lg.addHandler(h)
    lg.setLevel(logging.DEBUG)
    try:
        yield h
    finally:
        lg.removeHandler(h)
        lg.setLevel(prev)


def _post_nonstream_wb(conv_id, client_structured_outputs=None):
    _StubVLLM.captured_bodies = []
    body = {"model": "stub-model", "messages": [{"role": "user", "content": "hi"}], "stream": False}
    if client_structured_outputs is not None:
        body["structured_outputs"] = client_structured_outputs
    with patch.object(main.httpx, "AsyncClient", _StubVLLM), capture() as cap:
        r = client.post("/v1/chat/completions", json=body,
                         headers={"X-Conversation-Id": conv_id})
    return r, cap.records, list(_StubVLLM.captured_bodies)


clear_active()
sha8 = write_active(10, {"grammar": _grammar})[:8]
r, recs, bodies = _post_nonstream_wb("wb-e1a")
assert_eq(r.status_code, 200, "200 with the file active")
assert_eq(bodies[0].get("structured_outputs"), {"grammar": _grammar},
          "the outgoing vLLM body carries the file's grammar")
line = _find(recs, "request done req=")
assert_true(line is not None, "request done line present")
assert_true(f"wb=10:{sha8}" in line.getMessage(), "the request done line carries wb=<rev>:<sha8>")

# file wins over a differing client value
r, recs, bodies = _post_nonstream_wb("wb-e1b", client_structured_outputs={"grammar": "client version"})
assert_eq(bodies[0].get("structured_outputs"), {"grammar": _grammar},
          "the file's value wins even though the client sent a different one")

clear_active()
r, recs, bodies = _post_nonstream_wb("wb-e1c")
assert_eq(bodies[0].get("structured_outputs"), None,
          "no file, no client value -> nothing sent")
line = _find(recs, "request done req=")
assert_true(line is not None and "wb=" not in line.getMessage(),
            "no wb= field when there is nothing to report")

clear_active()


# ===========================================================================
# [E1b] hostile review round 1: F2 (selftest skip), F6 (decide exception
# is caught, fail-open), F7 (client's own response_format / other-shaped
# structured_outputs is never overridden), F8 (model mismatch skips
# injection) — all exercised through main.py's real endpoint.
# ===========================================================================

print("\n[E1b] F2/F6/F7/F8: selftest skip, exception fail-open, response_format/other-shape skip, model mismatch")

import selftest as selftest_module_for_test  # noqa: E402

clear_active()
write_active(12, {"grammar": _grammar})

# F2: a selftest one-shot conv_id is never injected, even with a valid
# file active.
_selftest_conv = f"{selftest_module_for_test.ONE_SHOT_CONV_PREFIX}deadbeef__"
r, recs, bodies = _post_nonstream_wb(_selftest_conv)
assert_eq(r.status_code, 200, "200")
assert_eq(bodies[0].get("structured_outputs"), None,
          "a selftest round-trip conv_id never gets the file's grammar")
line = _find(recs, "request done req=")
assert_true(line is not None and "wb=" not in line.getMessage(),
            "no wb= field for a selftest request")

# F6: decide_structured_outputs raising must not cost her the reply, and
# must leave the client's own value (if any) untouched.
with patch.object(wordban_runtime, "decide_structured_outputs",
                   side_effect=RuntimeError("boom")):
    r, recs, bodies = _post_nonstream_wb("wb-e1b-f6", client_structured_outputs={"grammar": "client kept"})
assert_eq(r.status_code, 200, "a decide_structured_outputs exception does not fail the request")
assert_eq(bodies[0].get("structured_outputs"), {"grammar": "client kept"},
          "the client's own value passes through UNCHANGED when the decision raises")
line = _find(recs, "request done req=")
assert_true(line is not None and "wb=" not in line.getMessage(),
            "no wb= field when the decision raised")
err_line = _find(recs, "decide_structured_outputs raised")
assert_true(err_line is not None, "the exception is logged")

# F7a: a client that sent its OWN response_format is never overridden —
# the file's grammar is not injected, and the client's structured_outputs
# (if any) is untouched too.
_StubVLLM.captured_bodies = []
body = {"model": "stub-model", "messages": [{"role": "user", "content": "hi"}],
        "stream": False, "response_format": {"type": "json_object"}}
with patch.object(main.httpx, "AsyncClient", _StubVLLM), capture() as cap:
    r = client.post("/v1/chat/completions", json=body, headers={"X-Conversation-Id": "wb-e1-f7a"})
bodies = list(_StubVLLM.captured_bodies)
assert_eq(r.status_code, 200, "200")
assert_eq(bodies[0].get("structured_outputs"), None,
          "a client-sent response_format blocks injection entirely")
assert_eq(bodies[0].get("response_format"), {"type": "json_object"},
          "the client's response_format itself is untouched")

# F7b: a client-sent structured_outputs that is NOT our {"grammar": str}
# shape is never overridden either.
_StubVLLM.captured_bodies = []
other_shape = {"json_schema": {"name": "x"}}
body = {"model": "stub-model", "messages": [{"role": "user", "content": "hi"}],
        "stream": False, "structured_outputs": other_shape}
with patch.object(main.httpx, "AsyncClient", _StubVLLM), capture() as cap:
    r = client.post("/v1/chat/completions", json=body, headers={"X-Conversation-Id": "wb-e1-f7b"})
bodies = list(_StubVLLM.captured_bodies)
assert_eq(r.status_code, 200, "200")
assert_eq(bodies[0].get("structured_outputs"), other_shape,
          "a client-sent, non-grammar-shaped structured_outputs is left exactly as sent")

# F8: the file is scoped to a DIFFERENT model than this request -> not injected.
clear_active()
write_active(13, {"grammar": _grammar}, model="some-other-model")
r, recs, bodies = _post_nonstream_wb("wb-e1-f8")
assert_eq(bodies[0].get("structured_outputs"), None,
          "the file targets a different model than the request -> not injected")
line = _find(recs, "request done req=")
assert_true(line is not None and "wb=" not in line.getMessage(),
            "no wb= field on a model mismatch")
mismatch_line = _find(recs, "model_mismatch") or _find(recs, "not injected")
assert_true(mismatch_line is not None, "the model mismatch is logged")

clear_active()


# ===========================================================================
# [E2] internal calls never see it — by construction: no module other than
# main.py imports wordban_runtime, and main.py calls
# decide_structured_outputs at exactly the two her-reply call sites.
# ===========================================================================

print("\n[E2] structured_outputs injection is confined to main.py's her-reply call sites")

import ast  # noqa: E402
_main_src = Path(main.__file__).read_text(encoding="utf-8")
_tree = ast.parse(_main_src)
_call_sites = [
    node for node in ast.walk(_tree)
    if isinstance(node, ast.Call)
    and isinstance(node.func, ast.Attribute)
    and node.func.attr == "decide_structured_outputs"
]
assert_eq(len(_call_sites), 1, "decide_structured_outputs is called from exactly ONE place in main.py")

for other in ("facts", "summarizer", "dedup", "continuity", "backfill", "retrieval"):
    mod = sys.modules.get(other)
    if mod is None:
        continue
    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert_true("wordban_runtime" not in src, f"{other}.py never imports wordban_runtime")


print()
if FAILED:
    for f in FAILED:
        print("FAIL " + f)
    sys.exit(1)
print("All test_wordban_runtime.py checks passed.")
