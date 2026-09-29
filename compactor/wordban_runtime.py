"""
compactor.wordban_runtime — v3.1.9.8 WORD-BAN FILE DEPLOY + IDLE WARM-UP.

WHY. The owner has been pasting the word-ban grammar (`scripts/wordban/`,
the `structured_outputs` custom parameter) directly into OpenWebUI's
Advanced Params for each model, and OpenWebUI forwards it, unmodified, on
every request that model makes. The UI is being replaced (V4: Matrix), so
the grammar has to move to a FILE the compactor injects instead — this
module owns that file, the injection decision, and the background
warm-up that keeps the first REAL request from paying xgrammar's
~110-170s cold-compile cost.

FILE. `COMPACTOR_WORDBAN_FILE` (default `/data/zl-ops/word-ban/active.json`,
a sibling of the other on-pod `/data/...` paths the same way STORAGE_ROOT/
CONTROL_ROOT/PENDING_ROOT are — see memory.STORAGE_ROOT / continuity.
CONTROL_ROOT / continuity.PENDING_ROOT for the pattern this follows).
Written atomically by `scripts/word-ban-deploy.py`, never by this process.
Shape: `{"revision": N, "value_sha256": "...", "structured_outputs": <the
exact value the OpenWebUI custom param would carry>, "deployed_at": "..."}`.

THE BYTE-IDENTITY REQUIREMENT. xgrammar's compile cache is keyed by the
grammar STRING vLLM actually receives — a warm-up that sends anything else
compiles a cache entry her real request will never hit, which is worse
than no warm-up (it burns the ~150s AND leaves her waiting anyway). So the
file's `structured_outputs` value must produce, once injected into a vLLM
request body, BYTE-IDENTICAL JSON to what OpenWebUI produces from the
pasted param today. Traced all the way through (v3.1.9, no code here,
report-only):

  - `scripts/wordban/wordban.py cmd_build` writes the pasted value as
    `json.dumps({"grammar": ebnf}, ensure_ascii=True)` — ONE line of text,
    one top-level key.
  - OpenWebUI 0.11.x stores that exact string as
    `model.params.custom_params["structured_outputs"]`. On every request
    for that model, `apply_params_to_form_data`
    (open_webui/utils/payload.py) does
    `custom_params[key] = JSONCodec.loads(value)` for any custom-param
    value that is a `str` — and `JSONCodec.loads` is a bare `json.loads`
    (open_webui/utils/misc.py's `vendor_misc.py` copy, confirmed against
    the vendored 0.11.4 source at
    `~/zl-ops/word-ban/work/owui-0.11.4-src`) — then
    `form_data["structured_outputs"] = value` (a **dict**, `{"grammar":
    "<ebnf>"}`, not a string) because the key was absent from `form_data`.
  - So THE COMPACTOR RECEIVES `structured_outputs` as a Python dict, already
    parsed, in the incoming `/v1/chat/completions` JSON body — never a
    string. `compactor/main.py` today does not touch this key at all (grep
    "structured_outputs" main.py: no hits before this change), so it is
    forwarded to vLLM completely unmodified as part of `json=body`
    (`httpx.AsyncClient.post(..., json=body)`, which serializes with
    stdlib `json.dumps` under the hood).

  Therefore: this module stores `structured_outputs` in `active.json` as
  that SAME parsed dict shape (not the raw pasted string), and
  `decide_structured_outputs` hands the CALLER that exact dict to set on
  `body["structured_outputs"]`. Injecting it takes the same path through
  `httpx`'s `json=` serialization that a client-sent value already takes
  today — same object shape, same serializer, same bytes on the wire.
  `test_wordban_runtime.py`'s `test_byte_identical_to_client_path` proves
  this directly: it builds a value the way `wordban.py` does, feeds it
  through OpenWebUI's OWN parse step (`json.loads`, matching JSONCodec),
  round-trips it through this module's canonicalization, and asserts the
  final `json.dumps` bytes vLLM would receive are identical byte-for-byte
  either way.

CANONICALIZATION (the "value_sha256" this module and
`word-ban-deploy.py` must agree on): `json.dumps(structured_outputs,
ensure_ascii=True).encode("utf-8")` — the exact call `wordban.py
cmd_build` makes to produce `structured_outputs.value.txt` in the first
place. A released value FILE's sha256 (computed over its raw bytes by
`wordban.py`/`word-ban-deploy.py`) and this function's sha256 of the
PARSED-then-reserialized value therefore agree, because the one shape
this ever holds (a single-key dict, string value) round-trips losslessly
through `json.loads` -> `json.dumps(ensure_ascii=True)`.

INJECTION. `decide_structured_outputs` is called from
`main._chat_completions_impl`, ONLY at the two her-reply vLLM call sites
(the streaming and non-streaming POSTs), never from an internal call
(facts/summarize/dedup/digest/tokenize each build their own httpx calls
elsewhere and never run through this function). The FILE WINS over a
client-sent value when both are present and differ — logged once per
distinct (file_sha, client_sha) pair, content-free.

OpenWebUI's background task calls (title/tags/follow-ups/autocomplete —
`main._is_openwebui_task_conv_id`) run through these SAME two call sites.
Today, the pasted custom param applies to them too (OpenWebUI attaches
`custom_params` to every request it sends for a model, task or chat —
nothing in `apply_params_to_form_data` special-cases task traffic). This
module keeps that: `decide_structured_outputs` does not look at conv_id
or task-ness at all, so the file (or client value) applies uniformly,
exactly as the pasted param did. Narrowing it to "chat only" would be a
BEHAVIOUR CHANGE from what is live today, and nothing about it looked
"clearly wrong" enough to make unasked (a title that also cannot contain
the banned word is, if anything, more consistent with the owner's intent
than one that can).

FAIL-OPEN. An invalid, missing or unreadable file gives up ONE thing:
the ban is not enforced from the file for any request in flight while it
is broken. The client-sent value (if the client still sends one) is used
instead; with neither, no `structured_outputs` is sent at all — the
model runs unbanned rather than the request failing. This is deliberate:
a config-file typo must never turn into a lost reply. It logs ERROR once
per file change and `/health/full`'s `word_ban.file_error` names it.

WARM-UP. See `WarmupWorker` below.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import httpx

import logsetup
import reqctx
from envcfg import env_bool, env_float

logger = logging.getLogger("compactor.wordban_runtime")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _default_file() -> Path:
    # A sibling of memory.STORAGE_ROOT (/data/openwebui/compactor) and
    # continuity.CONTROL_ROOT/PENDING_ROOT (both /data/compactor-*) in
    # SPIRIT — "derive a /data path the same way the others do" — but NOT
    # nested under any of them: the word-ban file is deployed by a SEPARATE
    # operator tool (`word-ban-deploy.py`, run by hand on the pod, never by
    # the compactor) into `/data/zl-ops/word-ban/`, the same directory the
    # owner already pastes `word-ban-structured_outputs.value.txt` FROM
    # (`~/zl-ops/word-ban/` on the build machine, `/data/zl-ops/word-ban/`
    # on the pod — see OPERATIONS.md). Matching that existing convention
    # is more useful here than matching STORAGE_ROOT's, which the backup/
    # restore machinery owns and this file must stay outside of for the
    # same reason CONTROL_ROOT/PENDING_ROOT do (a restore must not silently
    # roll the deployed grammar back).
    #
    # v3.1.9.10: follows ZL_HOME (the same env var ops/zlops/podenv.py and
    # ops/bootstrap.sh read, default /data/zl-ops), so a pod whose ZL_HOME is
    # elsewhere cannot read or warm from another pod's deployed grammar.
    return Path(os.environ.get("ZL_HOME") or "/data/zl-ops") / "word-ban" / "active.json"


WORDBAN_FILE = Path(os.environ.get("COMPACTOR_WORDBAN_FILE", "") or _default_file())

WARMUP_ENABLED = env_bool("COMPACTOR_WORDBAN_WARMUP", True)
IDLE_S = env_float("COMPACTOR_WORDBAN_IDLE_S", 30.0)

# How often the warm-up loop polls conditions. Cheap: a stat() on the
# file plus (when a warm-up might actually be due) one /metrics scrape.
# Not configurable via its own env var — 5s is fixed by the owner's ask
# ("poll cheaply, e.g. every 5 s") and there is no operational reason to
# tune it independently of IDLE_S.
POLL_S = 5.0

# vLLM read timeout for the warm-up request itself. The owner's ask is
# ">= 400 s" — generous headroom over the worst measured cold compile
# (rev 8: 174.2 s) because a warm-up that times out early just means one
# more real request pays the compile instead, which is strictly worse
# than waiting.
WARMUP_TIMEOUT_S = env_float("COMPACTOR_WORDBAN_WARMUP_TIMEOUT_S", 400.0)

# Backoff on a failed warm-up attempt (vLLM unreachable, a non-2xx, a
# timeout): 60s then 5 minutes, never a tight loop. Resets to the initial
# value the moment a warm-up SUCCEEDS or the target (warmed_key) changes
# out from under a backoff — a new grammar deserves a fresh attempt
# immediately, not whatever backoff the OLD grammar's failures left behind.
BACKOFF_INITIAL_S = env_float("COMPACTOR_WORDBAN_WARMUP_BACKOFF_S", 60.0)
BACKOFF_MAX_S = env_float("COMPACTOR_WORDBAN_WARMUP_BACKOFF_MAX_S", 300.0)


# ---------------------------------------------------------------------------
# Canonicalization + sha256 — shared by the file loader, the injector and
# the warm-up (one function, per the owner's ask: "Use the same
# serialization path as the injection (share one function)").
# ---------------------------------------------------------------------------

def canonical_value_bytes(structured_outputs: Any) -> bytes:
    """See the module docstring's CANONICALIZATION section. Must match
    `wordban.py cmd_build`'s `json.dumps({"grammar": ebnf},
    ensure_ascii=True)` byte-for-byte for the released value's sha256 to
    verify — `word-ban-deploy.py` computes `value_sha256` the same way
    (over the raw released file's bytes, which for the one shape this
    ever holds is identical to this function's output)."""
    return json.dumps(structured_outputs, ensure_ascii=True).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def value_sha256_of(structured_outputs: Any) -> str:
    return sha256_hex(canonical_value_bytes(structured_outputs))


def configured_model() -> str | None:
    """The compactor's OWN configured model — `MODEL_REPO`, the same env
    var `main.py` reads at its own module scope (`MODEL_REPO = os.environ.
    get("MODEL_REPO")`, main.py line ~108) and passes to `wordban_runtime.
    init(VLLM_URL, MODEL_REPO or "")` at startup. Read directly from the
    environment here, not from `main.MODEL_REPO` or `worker._model`,
    because main.py imports wordban_runtime (never the other way around)
    — importing main from here would be circular, and the env var is the
    one shared source of truth both modules already read independently.
    Returns None when unset or blank (never "" — callers treat that as
    "nothing to compare against", same as an absent file)."""
    return os.environ.get("MODEL_REPO") or None


# Hostile review round 2, N2: a value this module must refuse to load
# regardless of what `revision` its active.json claims — kept in sync BY
# HAND with scripts/word-ban-deploy.py's OWN copy of this same set (see
# that script's module docstring for why it is hardcoded there rather than
# read from scripts/wordban/ at runtime; the reasoning is identical here:
# this module ships alone in the compactor image — see Dockerfile's `COPY
# compactor/wordban_runtime.py` — with no scripts/wordban/ checkout to read
# a live withdrawn-list from). `word-ban-deploy.py`'s OWN check (matching
# by revision LABEL only, in WITHDRAWN_REVISIONS) is not enough on its own:
# a mislabelled redeploy of rev 4's bytes under "--revision 5" would sail
# straight through label-only matching and land in active.json, at which
# point THIS is the last line of defense before the compactor loads it.
# rev 4 traps the model (see word-ban-deploy.py's WITHDRAWN_REVISIONS[4]
# for the detail) — this is deliberately checked at LOAD time (fail-open,
# ERROR-logged, same as every other _parse_and_validate refusal) rather
# than trusted just because it is already on disk.
WITHDRAWN_VALUE_SHA256S = {
    "6b052ac1a4aa9be2e3c0428f6229a60b83c8f36fc07f06df65dac0da24d5c740":
        "revision 4's released value — it traps the model, regardless of "
        "what revision label active.json claims for it.",
}


def is_grammar_shape(value: Any) -> bool:
    """Hostile review round 1, F7: is `value` the ONE shape this module
    ever produces or accepts — a dict with EXACTLY one key, `"grammar"`,
    whose value is a string? Used to tell "a client sent the SAME kind of
    thing we would" (safe to compare/override) from "a client asked vLLM
    for some OTHER structured output" (a JSON-schema `response_format`,
    a differently-shaped `structured_outputs`, `guided_json`, ...) —
    injecting our grammar ON TOP of that is not a conflict to log and
    move on from, it is a second guided-decoding mechanism on the same
    request, which vLLM refuses outright (measured: a 400, "count>1").
    main.py's injection site skips entirely (leaves the client's own
    value untouched) whenever this returns False."""
    return (
        isinstance(value, dict)
        and set(value.keys()) == {"grammar"}
        and isinstance(value.get("grammar"), str)
    )


# ---------------------------------------------------------------------------
# File loader — lazy, stat()-gated, cached in memory.
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class _FileState:
    # None means "no file / never successfully parsed". A stat_key of
    # None also means "have not looked yet OR the file was absent last
    # time we looked" — _load() always re-stats regardless, so this only
    # ever suppresses a redundant *log line*, never a redundant read.
    stat_key: tuple | None = None
    present: bool = False
    revision: int | None = None
    value_sha256: str | None = None
    structured_outputs: Any = None
    deployed_at: str | None = None
    # Hostile review round 1, F8: the file applies to ONE model — the one
    # word-ban-deploy.py was told to deploy for (default coder3101/
    # Cydonia-24B-v4.3-vision-heretic, banlist.yaml's own `model:`).
    # Required (not optional) — see _parse_and_validate.
    model: str | None = None
    # Human-readable reason the file could not be used, or None if it
    # parsed and validated cleanly (including "no file present", which is
    # NOT an error — see `file_error` vs `present` in health_snapshot()).
    error: str | None = None


_state = _FileState()


def _stat_key(st: os.stat_result) -> tuple:
    # mtime+size+inode, per the owner's ask. st_mtime_ns (not st_mtime) so
    # two writes inside the same float-rounded second are still told
    # apart — word-ban-deploy.py's rollback/deploy cycle in a test or a
    # fast manual redo could otherwise land in the same second.
    return (st.st_mtime_ns, st.st_size, st.st_ino)


def _parse_and_validate(raw: bytes) -> _FileState:
    """Never raises — returns a `_FileState` with `.error` set instead.
    Called only when the file's stat() signature changed, so a bad parse
    here is exactly the "once per file change" the owner asked for; the
    caller (`_load`) is what turns that into the single log line."""
    new = _FileState(present=True)
    try:
        doc = json.loads(raw)
        if not isinstance(doc, dict):
            raise ValueError("active.json does not contain a JSON object")
        if ("revision" not in doc or "value_sha256" not in doc
                or "structured_outputs" not in doc or "model" not in doc):
            raise ValueError(
                "active.json is missing one of revision/value_sha256/"
                "structured_outputs/model"
            )
        revision = doc["revision"]
        if not isinstance(revision, int):
            raise ValueError(f"revision must be an integer, got {type(revision).__name__}")
        value_sha256 = doc["value_sha256"]
        if not isinstance(value_sha256, str) or not value_sha256:
            raise ValueError("value_sha256 must be a non-empty string")
        model = doc["model"]
        if not isinstance(model, str) or not model:
            raise ValueError("model must be a non-empty string")
        so = doc["structured_outputs"]
        computed = value_sha256_of(so)
        if computed != value_sha256:
            # Never log the content, only the shas (module contract).
            raise ValueError(
                f"value_sha256 mismatch: file says {value_sha256[:12]}…, "
                f"computed {computed[:12]}… over the canonicalized value"
            )
        if value_sha256 in WITHDRAWN_VALUE_SHA256S:
            # N2: refuse a withdrawn VALUE regardless of the revision label
            # active.json claims for it — see WITHDRAWN_VALUE_SHA256S.
            # Fail-open (same as every other refusal here): the ban is not
            # enforced from the file until this is fixed, which is strictly
            # safer than loading a grammar known to trap the model.
            raise ValueError(
                f"value_sha256 {value_sha256[:12]}… is a WITHDRAWN value "
                f"({WITHDRAWN_VALUE_SHA256S[value_sha256]}) — refusing to "
                f"load it, regardless of revision={revision!r}"
            )
        new.revision = revision
        new.value_sha256 = value_sha256
        new.structured_outputs = so
        new.model = model
        new.deployed_at = doc.get("deployed_at")
    except Exception as e:  # noqa: BLE001 — fail-open, see module docstring
        new.error = f"{type(e).__name__}: {e}"
    return new


def _load() -> _FileState:
    """Re-read `WORDBAN_FILE` if its stat() signature changed since the
    last call; otherwise return the cached `_FileState` untouched. AT
    MOST ONE stat() PER CALL, per the owner's ask — the caller
    (`decide_structured_outputs`) calls this exactly once per request."""
    global _state
    try:
        st = WORDBAN_FILE.stat()
    except OSError:
        # "No file" is not an error (module docstring) — only log the
        # TRANSITION into this state, once, not every request that finds
        # it still absent.
        if _state.present or _state.stat_key is not None:
            logger.info(
                f"word-ban file {WORDBAN_FILE} is no longer present; "
                f"falling back to the client-sent value if present, else none"
            )
        _state = _FileState()  # present=False, error=None
        return _state
    key = _stat_key(st)
    if key == _state.stat_key:
        return _state
    try:
        raw = WORDBAN_FILE.read_bytes()
    except OSError as e:
        new = _FileState(stat_key=key, present=True, error=f"{type(e).__name__}: {e}")
        logger.error(
            f"word-ban file {WORDBAN_FILE} exists but could not be read "
            f"({new.error}); the ban is NOT enforced from the file for any "
            f"request until this is fixed — falling back to the client-sent "
            f"value if present, else none"
        )
        _state = new
        return _state
    new = _parse_and_validate(raw)
    new.stat_key = key
    if new.error:
        logger.error(
            f"word-ban file {WORDBAN_FILE} is invalid ({new.error}); the ban "
            f"is NOT enforced from the file for any request until this is "
            f"fixed — falling back to the client-sent value if present, "
            f"else none"
        )
    else:
        logger.info(
            f"word-ban file loaded: revision={new.revision} "
            f"value_sha256={new.value_sha256[:12]}…"
        )
        # N1: the file's own model vs THIS POD's configured model
        # (MODEL_REPO) — distinct from decide_structured_outputs's
        # per-REQUEST model check (that one compares against
        # body.get("model") on every chat turn; this one is a static,
        # once-per-file-change check against what this compactor process
        # itself is configured to serve). Naturally fires once per file
        # change: this whole branch only runs when the file's stat()
        # signature changed (see the guard at the top of this function).
        pod_model = configured_model()
        if pod_model is not None and pod_model != new.model:
            logger.warning(
                f"word-ban file is deployed for model={new.model!r}, but "
                f"this compactor is configured for model={pod_model!r} "
                f"(MODEL_REPO) — real chat traffic on this pod names "
                f"{pod_model!r}, so decide_structured_outputs's per-request "
                f"model check (F8) will not inject this file's grammar "
                f"unless the client itself asks for {new.model!r}"
            )
    _state = new
    return _state


def _reset_for_tests() -> None:
    """Test helper only — never called from production code."""
    global _state, _last_decision
    _state = _FileState()
    _last_decision = None


# ---------------------------------------------------------------------------
# Injection decision
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class WordBanDecision:
    value: Any               # the structured_outputs value to send, or None
    source: str               # "file" | "client" | "none"
    revision: int | None = None
    value_sha256: str | None = None

    @property
    def trace_field(self) -> str | None:
        """The `wb=<revision>:<sha8>` token for the "request done" line,
        or None when there is nothing to report (source is not "file")."""
        if self.source != "file" or self.revision is None or not self.value_sha256:
            return None
        return f"{self.revision}:{self.value_sha256[:8]}"


# The most recent decide_structured_outputs() outcome, for /health/full's
# `word_ban.source` — a per-request fact ("file" | "client" | "none"),
# not a property of the file alone (a valid file always reports "file"
# here too, since decide_structured_outputs never returns anything else
# once the file validates). Before the first chat request in a fresh
# process, falls back to "file" or "none" from the file's own state alone
# (see health_snapshot) — there is no request yet to have decided anything.
_last_decision: "WordBanDecision | None" = None


def decide_structured_outputs(
    client_value: Any, model: str | None = None
) -> WordBanDecision:
    """The ONE decision point for both her-reply vLLM call sites
    (main._chat_completions_impl, streaming and non-streaming). Reads the
    file (at most one stat() — see `_load`), and returns what to send:

      - file valid, model matches -> the file's value, source="file",
                                      ALWAYS (wins over a client value)
      - file valid, model DIFFERS
        (hostile review round 1, F8) -> as if the file did not apply to
                                      THIS request at all: the client's
                                      value if any, else none. Logged
                                      once per distinct (deployed_model,
                                      requested_model) pair, at INFO —
                                      this is normal multi-model
                                      operation, not a fault.
      - file invalid/absent,
        client sent one           -> the client's value, source="client"
      - neither                   -> None, source="none"

    `model` is `body.get("model")` from the incoming request — pass None
    (the default) to skip the model check entirely (every existing
    caller/test that has no opinion on it keeps working unchanged).

    Logs a WARNING, once per distinct (file_sha, client_sha) pair, when
    both are present, the model matches, and they differ. Never logs
    content — shas only.
    """
    global _last_decision
    st = _load()
    if st.error is None and st.structured_outputs is not None:
        if model is not None and st.model is not None and model != st.model:
            pair_key = f"wordban.model_mismatch.{st.model}.{model}"
            if logsetup.log_once(pair_key):
                logger.info(
                    f"word-ban file is deployed for model={st.model!r}; "
                    f"this request is model={model!r} — not injected "
                    f"(this is expected if more than one model is served "
                    f"through this endpoint)"
                )
            decision = (
                WordBanDecision(client_value, "client")
                if client_value is not None
                else WordBanDecision(None, "none")
            )
            _last_decision = decision
            return decision
        if client_value is not None:
            client_sha = value_sha256_of(client_value)
            if client_sha != st.value_sha256:
                pair_key = (
                    f"wordban.conflict.{st.value_sha256[:16]}.{client_sha[:16]}"
                )
                if logsetup.log_once(pair_key):
                    logger.warning(
                        f"structured_outputs conflict: the deployed file "
                        f"(revision {st.revision}, sha256 "
                        f"{st.value_sha256[:16]}…) overrides a client-sent "
                        f"value (sha256 {client_sha[:16]}…) — the file "
                        f"always wins"
                    )
        decision = WordBanDecision(
            st.structured_outputs, "file", st.revision, st.value_sha256
        )
    elif client_value is not None:
        decision = WordBanDecision(client_value, "client")
    else:
        decision = WordBanDecision(None, "none")
    _last_decision = decision
    return decision


# ---------------------------------------------------------------------------
# vLLM identity + /metrics probing (shared by the health snapshot and the
# warm-up loop)
# ---------------------------------------------------------------------------

_PROCESS_START_RE = re.compile(
    r"^process_start_time_seconds(?:\{[^}]*\})?\s+([0-9.eE+-]+)\s*$", re.M
)
_NUM_RUNNING_RE = re.compile(
    r"^vllm:num_requests_running(?:\{[^}]*\})?\s+([0-9.eE+-]+)\s*$", re.M
)
_NUM_WAITING_RE = re.compile(
    r"^vllm:num_requests_waiting(?:\{[^}]*\})?\s+([0-9.eE+-]+)\s*$", re.M
)


def _parse_metric(pattern: re.Pattern, text: str) -> float | None:
    m = pattern.search(text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


@dataclasses.dataclass(frozen=True)
class VllmMetrics:
    identity: str | None       # a value stable per vLLM process, or None
    num_running: float | None
    num_waiting: float | None
    identity_source: str       # "process_start_time_seconds" | "unavailable"


async def fetch_vllm_metrics(client: httpx.AsyncClient, vllm_url: str) -> VllmMetrics | None:
    """GET `<vllm_url>/metrics` and pull the three numbers the warm-up
    gate needs. Returns None on any failure (unreachable, non-200,
    unparseable) — callers treat that exactly like "vLLM is not idle":
    never warm on a probe we can't trust.

    IDENTITY: `process_start_time_seconds` is a standard `prometheus_
    client` ProcessCollector gauge (library-level, not vLLM-specific) —
    vLLM's OpenAI server mounts its metrics via `prometheus_client.
    make_asgi_app()`, which registers the default collectors including
    this one, so it is present whenever `/metrics` is reachable at all,
    library version permitting. It changes exactly when the vLLM PROCESS
    restarts (a fresh PID => a fresh process start time => a cold
    xgrammar cache) and stays fixed across load, which is exactly the
    "stable per vLLM process" the owner asked for.

    KNOWN GAP (report this — do not silently assume): the owner's ask
    also says "if that's unavailable, use something else stable per
    vLLM process (check what vLLM 0.19 exposes in the image; don't
    guess)". This build environment has no GPU pod and no network
    access to the real image to confirm vLLM 0.19's /metrics contents
    beyond the prometheus_client standard above, so no fallback metric
    is guessed at here. If `process_start_time_seconds` is ever absent
    on the real pod, `identity` comes back None, `identity_source` comes
    back "unavailable", and the warm-up loop (by design, see
    `WarmupWorker._tick`) treats every poll as "no known warmed_key to
    compare against" and never fires automatically in that mode — it
    fails toward NOT warming rather than mis-warming, and
    `/health/full`'s `word_ban.warm_state` shows "disabled" so this is
    visible, not silent. This must be confirmed against the real image
    before relying on automatic re-warm-on-restart in production.
    """
    try:
        r = await client.get(f"{vllm_url}/metrics", timeout=5.0)
        r.raise_for_status()
    except httpx.HTTPError:
        return None
    text = r.text
    start = _parse_metric(_PROCESS_START_RE, text)
    running = _parse_metric(_NUM_RUNNING_RE, text)
    waiting = _parse_metric(_NUM_WAITING_RE, text)
    if start is None:
        return VllmMetrics(None, running, waiting, "unavailable")
    # Stringify rather than keep the float: two scrapes of the SAME
    # process must compare equal, and a float's repr is stable for a
    # value vLLM itself only ever sets once (at process start) and never
    # updates, so this is just a display/compare convenience, not a
    # precision concern.
    return VllmMetrics(repr(start), running, waiting, "process_start_time_seconds")


# ---------------------------------------------------------------------------
# Warm-up
# ---------------------------------------------------------------------------

class WarmupWorker:
    """Background loop (started at app startup, per main.lifespan) that
    sends a one-token max_tokens=1 request straight to vLLM whenever the
    grammar is active, has not yet been warmed on the CURRENT vLLM
    process, and the process has been genuinely idle for
    `COMPACTOR_WORDBAN_IDLE_S` (default 30s).

    Tracks `warmed_key = (value_sha256, vllm_identity)` — a warm-up is
    due whenever the file names an active grammar AND that pair differs
    from the last one this process successfully warmed. This covers both
    triggers the owner named: the grammar changing (value_sha256 differs
    for the SAME vllm_identity) and vLLM restarting (vllm_identity
    differs for the SAME value_sha256, i.e. the xgrammar cache is cold
    again even though the deployed grammar hasn't moved).
    """

    def __init__(
        self,
        vllm_url: str,
        model: str,
        *,
        idle_s: float = IDLE_S,
        poll_s: float = POLL_S,
        warmup_timeout_s: float = WARMUP_TIMEOUT_S,
        backoff_initial_s: float = BACKOFF_INITIAL_S,
        backoff_max_s: float = BACKOFF_MAX_S,
        clock=time.monotonic,
        sleep=None,
    ) -> None:
        self._vllm_url = vllm_url.rstrip("/")
        self._model = model
        self._idle_s = idle_s
        self._poll_s = poll_s
        self._warmup_timeout_s = warmup_timeout_s
        self._backoff_initial_s = backoff_initial_s
        self._backoff_max_s = backoff_max_s
        self._clock = clock
        self._sleep = sleep  # set post-construction (asyncio.sleep) — see run()
        self._warmed_key: tuple[str, str] | None = None
        self._state = "idle"  # idle | waiting_idle | warming | backoff | disabled | vllm_unreachable
        self._backoff_s = backoff_initial_s
        self._backoff_until: float | None = None
        self._in_flight = 0
        self._last_compile_s: float | None = None
        self._last_warm_at: float | None = None  # wall-clock (time.time()), for reporting
        self._warmup_n = 0
        self._stopped = False
        # Hostile review round 1, F1 (BLOCKER): a fresh boot has never
        # finished a chat request, so `_last_activity_age_s()` returns
        # None — which used to block idleness FOREVER on that signal
        # alone, so a pod that restarted with no chat traffic yet (the
        # common case right after a deploy/restart) never warmed at all;
        # she paid the cold compile on her first real reply regardless of
        # how long the pod had been sitting idle.
        #
        # Fixed by giving the idle gate TWO MORE clocks, either of which
        # can satisfy it on its own once no chat request has ever run:
        #   - `_started_at`: this WORKER's own process clock, set once,
        #     here. Covers "fresh boot, never any traffic" — 30s after
        #     THIS process starts polling, not "never".
        #   - `_identity_changed_at`: set in `_tick` every time the
        #     scraped `vllm_identity` differs from the last one this
        #     worker saw (including the very FIRST successful scrape —
        #     see `_tick`), so a LONE vLLM restart (compactor process
        #     untouched, so `_started_at` is stale) still gets its own
        #     fresh 30s idle window rather than inheriting one that may
        #     have expired ages ago.
        # `_last_activity_age_s` (below) takes the MINIMUM of all three
        # ages that are not None — the SMALLEST age is the most recent
        # "something happened" signal, and idleness must be judged
        # against the most recent one, not the stalest.
        self._started_at = clock()
        self._identity_changed_at: float | None = None
        self._last_identity_seen: str | None = None

    # -- polled inputs, each independently overridable for tests --------

    async def _bg_outstanding(self) -> int:
        import bgwork
        return bgwork.pool.stats()["outstanding"]

    def _requests_in_flight(self) -> int:
        return reqctx.in_flight_stats()["requests_in_flight"]

    def _last_activity_age_s(self) -> float | None:
        """The SMALLEST of three ages, each None when that clock has
        nothing to report yet:

          - seconds since `reqctx`'s own `_last_done_at` (the last chat
            reply finished) — None until the first chat request in this
            process's life completes;
          - seconds since THIS worker started polling (`_started_at`,
            set once in `__init__`) — never None once the worker exists;
          - seconds since the last observed `vllm_identity` CHANGE
            (`_identity_changed_at`, set in `_tick`, including the first
            successful /metrics scrape ever seen) — None until that first
            scrape lands.

        Hostile review round 1, F1 (BLOCKER): taking ONLY the first of
        these meant a fresh boot with no chat traffic yet — the ordinary
        shape right after a restart — never satisfied "idle" at all,
        because `None` is not `>= idle_s`; the warm-up simply never fired
        and she paid the cold compile on her first real reply, the exact
        thing this feature exists to prevent. The other two clocks give
        idleness a basis to judge by even when reqctx has never seen a
        request: 30s after this worker starts polling (a fresh compactor
        boot), or 30s after vLLM's own identity last changed (a vLLM-only
        restart, compactor untouched — `_started_at` would be stale
        there, so it alone is not enough either).

        The MINIMUM (not e.g. the max, or "any one is enough") is
        correct: the most RECENT signal of "something happened" is what
        idleness must be measured against — a clock that has been quiet
        for an hour says nothing if a different one changed 2 seconds
        ago.
        """
        ages = [reqctx.in_flight_stats()["last_request_done_age_s"]]
        now = self._clock()
        ages.append(now - self._started_at)
        if self._identity_changed_at is not None:
            ages.append(now - self._identity_changed_at)
        present = [a for a in ages if a is not None]
        return min(present) if present else None

    # -- the gate ---------------------------------------------------------

    async def _idle_enough(self, metrics: VllmMetrics) -> bool:
        """All four conditions from the owner's ask, each independently
        blocking (mutation-check target: dropping any ONE of these must
        fail a test). `metrics` is the SAME scrape `_tick` already used
        to read `vllm_identity` — one /metrics fetch per poll cycle, not
        two."""
        if self._requests_in_flight() != 0:
            return False
        last_age = self._last_activity_age_s()
        if last_age is None or last_age < self._idle_s:
            return False
        try:
            outstanding = await self._bg_outstanding()
        except Exception:
            # bgwork unavailable (should not happen outside a test double)
            # — fail toward NOT warming, same doctrine as everywhere else
            # in this loop: a false "idle" risks colliding a warm-up with
            # real background traffic; a false "not idle" just delays it.
            return False
        if outstanding != 0:
            return False
        if (metrics.num_running or 0) + (metrics.num_waiting or 0) != 0:
            return False
        return True

    # -- one poll cycle -----------------------------------------------------

    async def _tick(self, http_client: httpx.AsyncClient) -> None:
        now = self._clock()
        if self._backoff_until is not None and now < self._backoff_until:
            self._state = "backoff"
            return
        st = _load()
        if st.error is not None or st.structured_outputs is None:
            # No active grammar to warm (never deployed, or the file is
            # currently broken — fail-open already covers the request
            # path; there's nothing to pre-warm here either way).
            self._state = "idle"
            return
        metrics = await fetch_vllm_metrics(http_client, self._vllm_url)
        if metrics is None:
            # Hostile review round 1, F5: distinct from "disabled" below —
            # this is "vLLM did not answer /metrics at all" (down,
            # restarting, unreachable), a TRANSIENT condition an operator
            # should read differently from "this vLLM build has no usable
            # identity metric".
            self._state = "vllm_unreachable"
            return
        if metrics.identity is None:
            # KNOWN GAP (see fetch_vllm_metrics docstring): with no
            # reliable per-process identity we cannot tell "same process,
            # already warmed" from "restarted, cold cache" apart, so we
            # never auto-fire here — visible via warm_state="disabled",
            # never silent.
            self._state = "disabled"
            return
        # Hostile review round 1, F1: the identity clock feeding
        # `_last_activity_age_s`'s idle gate — updated on EVERY tick that
        # reaches here, whether or not a warm-up ends up firing, so a
        # LONE vLLM restart (identity changes, compactor process
        # untouched) gets its own fresh 30s idle window the instant it is
        # first observed, INCLUDING the very first successful scrape this
        # worker ever makes (`_last_identity_seen is None` the first
        # time through).
        if metrics.identity != self._last_identity_seen:
            self._identity_changed_at = now
            self._last_identity_seen = metrics.identity
        key = (st.value_sha256, metrics.identity)
        if key == self._warmed_key:
            self._state = "idle"
            return
        self._state = "waiting_idle"
        if not await self._idle_enough(metrics):
            return
        await self._warm(http_client, st, key)

    async def _warm(self, http_client: httpx.AsyncClient, st: _FileState, key: tuple[str, str]) -> None:
        self._state = "warming"
        self._in_flight += 1
        self._warmup_n += 1
        n = self._warmup_n
        sha8 = st.value_sha256[:8]
        started = self._clock()
        logger.info(
            f"word-ban warm-up starting: revision={st.revision} "
            f"value_sha256={sha8}… vllm_identity={key[1]!r} (n={n})"
        )
        payload = {
            # Hostile review round 1, F8: the file's OWN model
            # (`st.model`, required — see `_parse_and_validate`), not
            # this worker's `self._model` (the compactor's configured
            # MODEL_REPO) — the warm-up must compile the grammar against
            # the SAME model the injection actually targets, or it warms
            # xgrammar's cache for a model nobody's real request uses.
            # `self._model` is kept only as a defensive fallback; `st.
            # model` is never None once the file has loaded without error
            # (required field).
            "model": st.model or self._model,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
            "stream": False,
            "structured_outputs": st.structured_outputs,
        }
        try:
            r = await http_client.post(
                f"{self._vllm_url}/v1/chat/completions",
                json=payload,
                timeout=self._warmup_timeout_s,
                headers={"X-Request-Id": f"warmup-{sha8}-{n}"},
            )
            r.raise_for_status()
        except Exception as e:
            elapsed = self._clock() - started
            logger.warning(
                f"word-ban warm-up FAILED after {elapsed:.1f}s "
                f"({type(e).__name__}: {e}); revision={st.revision} "
                f"value_sha256={sha8}… — backing off {self._backoff_s:.0f}s"
            )
            self._backoff_until = self._clock() + self._backoff_s
            self._backoff_s = min(self._backoff_s * 5.0, self._backoff_max_s)
            self._state = "backoff"
            return
        finally:
            self._in_flight -= 1
        elapsed = self._clock() - started
        self._last_compile_s = elapsed
        self._last_warm_at = time.time()
        self._warmed_key = key
        self._backoff_s = self._backoff_initial_s
        self._backoff_until = None
        self._state = "idle"
        logger.info(
            f"word-ban warm-up done: revision={st.revision} "
            f"value_sha256={sha8}… compile_s={elapsed:.1f}"
        )

    # -- run loop -------------------------------------------------------

    async def run(self) -> None:
        sleep = self._sleep or asyncio.sleep
        async with httpx.AsyncClient() as client:
            while not self._stopped:
                try:
                    await self._tick(client)
                except Exception:
                    logger.exception("word-ban warm-up loop: unhandled error in one cycle")
                await sleep(self._poll_s)

    def stop(self) -> None:
        self._stopped = True

    # -- health snapshot --------------------------------------------------

    def snapshot(self) -> dict:
        return {
            "warmed": self._warmed_key is not None,
            "warm_state": self._state,
            "last_warm_compile_s": (
                round(self._last_compile_s, 1) if self._last_compile_s is not None else None
            ),
            "last_warm_at": (
                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self._last_warm_at))
                if self._last_warm_at is not None else None
            ),
            "warmups_in_flight": self._in_flight,
        }


# Process-wide singleton, mirroring bgwork.pool's pattern. Constructed
# lazily (main.py provides VLLM_URL/MODEL_REPO only after its own module
# scope runs) via `init()`, called once from main.lifespan.
worker: WarmupWorker | None = None


def init(vllm_url: str, model: str) -> WarmupWorker:
    global worker
    worker = WarmupWorker(vllm_url, model)
    return worker


def health_snapshot() -> dict:
    """For /health/full's `word_ban` key. Never raises — an exception
    here must not take the whole health endpoint down with it (same
    doctrine as bgwork.pool.stats()'s call site in health.py)."""
    st = _state  # read only; never triggers a reload from the health path
    if _last_decision is not None:
        source = _last_decision.source
    else:
        # No chat request has run `decide_structured_outputs` yet in this
        # process (fresh boot) — state what the file alone implies.
        source = "file" if (st.present and st.error is None) else "none"
    out = {
        "source": source,
        "revision": st.revision,
        "value_sha256": st.value_sha256,
        "file_error": st.error,
        # Hostile review round 1, F8: the model this file's grammar
        # applies to — so an operator seeing `source != "file"` for real
        # chat traffic can tell "the file is broken" (file_error set)
        # apart from "the file is fine, just deployed for a different
        # model than this request" (this field disagrees with the
        # request's own `model`).
        "model": st.model,
        # N1: whether the deployed file's model equals THIS pod's own
        # configured model (MODEL_REPO) — a bool, always present, so an
        # operator does not have to cross-reference "model" against their
        # own runpod.env by hand. False both when there is no valid file
        # (st.model is None — nothing to claim a match) and when
        # configured_model() itself is unset (nothing to compare against,
        # so "matches" cannot be asserted true).
        "model_matches_pod": st.model is not None and st.model == configured_model(),
    }
    if not WARMUP_ENABLED:
        out.update(
            warmed=False, warm_state="disabled", last_warm_compile_s=None,
            last_warm_at=None, warmups_in_flight=0,
        )
    elif worker is None:
        out.update(
            warmed=False, warm_state="disabled", last_warm_compile_s=None,
            last_warm_at=None, warmups_in_flight=0,
        )
    else:
        out.update(worker.snapshot())
    return out
