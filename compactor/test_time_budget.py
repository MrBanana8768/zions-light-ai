"""The time line is paid for inside the window, never on top of it.

v3.1.9 (V4_ROADMAP.md section 1.1 item 1; trap (d) of the feature brief).

The line is added to the forwarded payload AFTER the hard-budget guard has
decided the array — after the guard, the merges and the template-tail repair,
because each of those can change which message is the newest user turn. A line
added after the guard is a line the guard never counted, so the guard is handed
a limit REDUCED by the line's reserve, and the reserve is an upper bound on
what the line can cost: its UTF-8 byte length (no tokenizer this project uses
spends fewer than one byte per token) plus the separator and a fixed slack for
the join. And when the guard reports the payload does NOT fit — the caller's own
prompt and newest turn already overflow — no line is added at all: it could only
make a request vLLM is about to refuse larger.

The counter here is deterministic (one token per UTF-8 byte of text, plus 4 per
message) and is installed as BOTH the exact and the local counter, so the guard
and this file's own measurement agree by construction and the property asserted
is the arithmetic, not a tokenizer.

  [1] the guard is handed effective_limit minus the reserve, and the full limit
      when there is no line (feature off, task traffic)
  [2] a sweep of payload sizes straddling the limit: every forwarded payload,
      line included, measures inside the window the guard enforces, and the
      line is present on every one of them
  [3] a payload the guard cannot fit carries no line; CONTROL, one that fits does

    python test_time_budget.py
"""

import datetime as dt
import json
import logging
import os
import sys
import tempfile
from unittest.mock import patch

os.environ.setdefault("MODEL_REPO", "test-model")
os.environ.setdefault("VLLM_URL", "http://stub:8000")
os.environ["COMPACTOR_RAG_ENABLED"] = "false"
os.environ["COMPACTOR_FACTS_EXTRACTION"] = "false"
os.environ["COMPACTOR_STORAGE_ROOT"] = tempfile.mkdtemp(prefix="time-budget-")
os.environ["MAX_MODEL_LEN"] = "4096"
os.environ["COMPACTOR_GENERATION_RESERVE"] = "1024"
os.environ.pop("COMPACTOR_TIMEZONE", None)
os.environ.pop("TZ", None)
os.environ.pop("COMPACTOR_TIME_INJECTION", None)

import memory  # noqa: E402

memory.ensure_storage_layout()

import httpx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402

FAILED: list[str] = []
NEEDLE = "Current date and time"
T0 = dt.datetime(2026, 9, 14, 16, 41, tzinfo=dt.timezone.utc)


def check(cond, label):
    print(("  ok   " if cond else "FAIL ") + label)
    if not cond:
        FAILED.append(label)


def tokens(msgs) -> int:
    """One token per UTF-8 byte of each message's text, plus 4 per message."""
    return sum(len(main._message_text(m).encode("utf-8")) + 4 for m in msgs)


main.count_tokens_exact = lambda ms, *a, **k: tokens(ms)
main.count_tokens = lambda ms: tokens(ms)

EFFECTIVE = min(main.MAX_MODEL_LEN,
                max(256, main.MAX_MODEL_LEN - max(main.GENERATION_RESERVE, 0)))
check(EFFECTIVE == 3072 and main._BUDGET_MARGIN == 0,
      f"fixture: effective limit {EFFECTIVE}, no learned margin")


class _Backend:
    bodies: list[dict] = []

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None, **kw):
        _Backend.bodies.append(json)
        return httpx.Response(200, json={"choices": [{"index": 0, "message": {
            "role": "assistant", "content": "A reply."}, "finish_reason": "stop"}]},
            request=httpx.Request("POST", url))

    async def aclose(self):
        pass


async def _no_compaction(messages, conv_id, stored_turns_out=None):
    return list(messages)


_guard_limits: list[int] = []
_guard_reserves: list[int] = []
_guard_reports: list[dict] = []
_real_guard = main._enforce_hard_budget


def _spy_guard(
    msgs, limit=None, protect_system=1, report=None, reserve=0,
    standin_protected=True,
):
    # `reserve` (hostile pass #5 F3) and `standin_protected` (P14-1,
    # hostile pass #14, lane v3194-guard) are real parameters of the guard
    # now, not spy-only additions: passed through so this file still
    # exercises the guard's actual behaviour instead of silently dropping
    # an argument the request path sends and testing a call it never
    # makes (this file patches compact_if_needed to `_no_compaction`,
    # which never reports a stand-in, so `standin_protected` always
    # arrives True here — forwarded anyway, on the same principle).
    _guard_limits.append(limit)
    _guard_reserves.append(reserve)
    out = _real_guard(msgs, limit, protect_system, report, reserve, standin_protected)
    _guard_reports.append(dict(report or {}))
    return out


class _Lines(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, r):
        self.records.append(r)

    def text(self):
        return "\n".join(r.getMessage() for r in self.records)


client = TestClient(main.app, client=("127.0.0.1", 12345), raise_server_exceptions=False)


def post(msgs, conv="budget-conv", enabled=True):
    _Backend.bodies = []
    lg = logging.getLogger("compactor")
    cap = _Lines()
    prev = lg.level
    lg.addHandler(cap)
    lg.setLevel(logging.DEBUG)
    with patch.object(main.httpx, "AsyncClient", _Backend), \
         patch.object(main, "compact_if_needed", _no_compaction), \
         patch.object(main, "_enforce_hard_budget", _spy_guard), \
         patch.object(main, "TIME_INJECTION_ENABLED", enabled, create=True), \
         patch.object(main, "_now_utc", lambda: T0, create=True), \
         patch.object(main, "_fire_and_forget",
                      lambda coro, label=None: (coro.close(), True)[1]):
        r = client.post("/v1/chat/completions",
                        json={"model": "test-model", "messages": msgs, "stream": False},
                        headers={"X-Conversation-Id": conv})
    lg.removeHandler(cap)
    lg.setLevel(prev)
    fwd = _Backend.bodies[-1]["messages"] if _Backend.bodies else []
    return r.status_code, fwd, cap.text()


def convo(n_exchanges, newest_pad):
    out = [{"role": "system", "content": "You are a patient assistant."}]
    for i in range(n_exchanges):
        out.append({"role": "user", "content": f"Tell me about item {i}. " + "x" * 150})
        # 150 characters of NON-repeating filler (v3.1.9.2). This was "y" * 150,
        # which reply_is_degenerate correctly flags as a repetition loop; since
        # v3.1.9.2 the forwarded window replaces flagged replies before the
        # guard measures them, so the sweep never needed to shed. Same length,
        # so the token arithmetic of the sweep is unchanged.
        _fill = " ".join(f"w{i}x{j}" for j in range(60))[:150]
        out.append({"role": "assistant", "content": f"Item {i} is a box. " + _fill})
    out.append({"role": "user", "content": "Which is largest? " + "z" * newest_pad})
    return out


fmt = getattr(main, "_format_time_line", None)
reserve_fn = getattr(main, "_time_line_token_reserve", None)
check(fmt is not None and reserve_fn is not None,
      "main._format_time_line and main._time_line_token_reserve exist")
LINE = fmt(T0, dt.timezone.utc) if fmt else ""
RESERVE = reserve_fn(LINE) if reserve_fn else 0
check(RESERVE >= len((LINE + "\n\n").encode("utf-8")) + 4,
      f"the reserve ({RESERVE}) covers the line and its separator in bytes "
      f"({len((LINE + chr(10) * 2).encode())}) with slack")


# ===========================================================================
print("[1] the limit the guard is handed")
_guard_limits.clear()
status, fwd, _ = post(convo(3, 10))
check(status == 200 and _guard_limits == [EFFECTIVE - RESERVE],
      f"feature on: guard limit {_guard_limits} == {EFFECTIVE} - {RESERVE}")
_guard_limits.clear()
post(convo(3, 10), enabled=False)
check(_guard_limits == [EFFECTIVE], f"CONTROL feature off: the full limit ({_guard_limits})")
_guard_limits.clear()
post([{"role": "user", "content": "### Task:\nGenerate a concise title."}],
     conv="9d1c2b3a-0000-4000-8000-00000000abcdtitle_generation")
check(_guard_limits == [EFFECTIVE],
      f"task traffic (no line) is not charged a reserve ({_guard_limits})")
_guard_limits.clear()
post([{"role": "system", "content": "You are a patient assistant."},
      {"role": "assistant", "content": "Hello."}], conv="no-user-turn")
check(_guard_limits == [EFFECTIVE],
      f"a payload with no user message (nothing to date) is not charged a reserve "
      f"({_guard_limits})")


# ===========================================================================
print("[2] a sweep across the limit")
over, missing, shed_seen, fits_exactly = [], [], 0, 0
for pad in range(0, 420, 3):
    msgs = convo(9, pad)
    status, fwd, _ = post(msgs, conv=f"sweep-{pad}")
    if status != 200 or not fwd:
        missing.append((pad, status))
        continue
    n = tokens(fwd)
    if n > EFFECTIVE:
        over.append((pad, n))
    if json.dumps(fwd).count(NEEDLE) != 1:
        missing.append((pad, "no line"))
    if len(fwd) < len(msgs):
        shed_seen += 1
    if EFFECTIVE - RESERVE < n:
        fits_exactly += 1
check(shed_seen >= 20, f"fixture: the guard shed turns on {shed_seen} sweep point(s)")
check(not missing, f"every sweep payload was forwarded with exactly one line ({missing[:5]})")
check(not over,
      f"no forwarded payload, line included, exceeds the {EFFECTIVE}-token window "
      f"({len(over)} over: {over[:5]})")
print(f"  note {fits_exactly} payload(s) landed inside the reserve band, i.e. would have "
      f"been at risk had the line not been reserved")


# ===========================================================================
print("[3] a payload that cannot fit is not made larger")
BIG = [{"role": "system", "content": "You are a patient assistant."},
       {"role": "user", "content": "Item. " + "w" * (EFFECTIVE + 400)}]
_guard_reports.clear()
status, fwd, big_log = post(BIG, conv="too-big")
check(_guard_reports and _guard_reports[-1].get("fits") is False,
      f"fixture: the guard measured this payload as not fitting ({_guard_reports[-1:]})")
check(fwd and json.dumps(fwd).count(NEEDLE) == 0,
      "no line was added to a payload already over the window")
check("hard budget FAILED to fit" in big_log,
      "fixture: this genuine overflow (well past the reserve too) still logs "
      "the guard's ERROR — F3 narrows the false case, it does not silence the real one")
_guard_reports.clear()
status, fwd, _ = post(convo(2, 5), conv="fits")
check(_guard_reports and _guard_reports[-1].get("fits") is True
      and json.dumps(fwd).count(NEEDLE) == 1,
      "CONTROL: a payload that fits carries its line")


# ===========================================================================
print("[4] the reserve band: nothing sheddable, but the REAL window is fine (F3)")
# hostile pass #5 F3. One system message (protected, protect_system=1) and
# one user turn (never dropped): nothing here is sheddable. A payload in this
# shape that measures over `EFFECTIVE - RESERVE` (the guard's narrowed limit)
# but not over `EFFECTIVE` itself (the real window) fits fine and vLLM will
# accept it — but before this fix the guard's own ERROR fired anyway, because
# it only ever compared against the narrowed limit. The soak
# (test_soak_conversation.py) and the adversarial suite both treat that ERROR
# as a real failure.
sysm = {"role": "system", "content": "You are a patient assistant."}


def band_msgs(gap):
    base = tokens([sysm, {"role": "user", "content": ""}])
    return [sysm, {"role": "user", "content": "q" * (EFFECTIVE - gap - base)}]


check(RESERVE >= 2, f"fixture: the reserve ({RESERVE}) leaves room for a 1-token band")
main._BUDGET_MARGIN = 0
_guard_reports.clear()
status, fwd, band_log = post(band_msgs(RESERVE - 1), conv="band-tight")
check(status == 200 and fwd, f"fixture: the request completed ({status})")
check(_guard_reports and _guard_reports[-1].get("fits") is False,
      f"fixture: the reserve-narrowed guard call reports fits=False "
      f"({_guard_reports[-1:]})")
check(tokens(fwd) <= EFFECTIVE,
      f"the forwarded payload fits the real {EFFECTIVE}-token window "
      f"({tokens(fwd)})")
check(json.dumps(fwd).count(NEEDLE) == 0,
      "no line was added — nothing reserved the room for it")
check("hard budget FAILED to fit" not in band_log,
      f"*** F3: no false ERROR for a payload that fits the real window: {band_log!r}")
check("but not with room left for the current-time line" in band_log,
      f"an INFO line explains why the line was skipped: {band_log!r}")
check(band_log.count("payload fits the") == 1, "said once for this conversation")
_guard_reports.clear()
status, fwd, band_log2 = post(band_msgs(RESERVE - 1), conv="band-tight")
check(status == 200 and "payload fits the" not in band_log2
      and "hard budget FAILED" not in band_log2,
      "a second request on the SAME conversation does not repeat the INFO line "
      "(log_once is keyed per conversation) and still logs no ERROR")

# CONTROL: a different conversation in the same band DOES get its own line.
_guard_reports.clear()
status, fwd, band_log3 = post(band_msgs(RESERVE - 1), conv="band-tight-other")
check(status == 200 and "payload fits the" in band_log3,
      "CONTROL: a DIFFERENT conversation in the same band gets its own INFO line")

# CONTROL: comfortably inside the reserve — the line is added, no special log.
_guard_reports.clear()
status, fwd, ok_log = post(band_msgs(RESERVE + 40), conv="band-fits")
check(status == 200 and json.dumps(fwd).count(NEEDLE) == 1,
      "CONTROL: with room for the reserve, the line is added as usual")
check("hard budget FAILED" not in ok_log and "payload fits the" not in ok_log,
      "CONTROL: neither the ERROR nor the reserve-band INFO fires when there "
      "was room to begin with")

# CONTROL (repeats [3] with the log now inspected): a payload well past the
# reserve too still gets the real ERROR — F3 narrows the false case, it does
# not silence the guard when the request truly will not fit.
check("hard budget FAILED to fit" in big_log,
      "CONTROL: a genuine overflow (see [3] BIG) still logs the real ERROR")


print()
if FAILED:
    print(f"FAILED: {len(FAILED)} check(s)")
    for f in FAILED:
        print(f"  - {f}")
    sys.exit(1)
print("All time-budget checks passed.")
