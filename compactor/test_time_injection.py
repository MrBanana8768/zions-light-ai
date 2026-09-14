"""The model is told the real current date and time — and only the model is.

v3.1.9, V4_ROADMAP.md section 1.1 item 1. Nothing in the prompt carried
wall-clock time, so asked what time it was the model invented one, and the
inventions reached the fact store as "facts". One line now rides at the head
of the NEWEST user message of the payload the compactor forwards:

    [Current date and time: Monday, September 14, 2026, 9:41 AM MST (UTC-07:00)]

WHAT THIS FILE PINS, section by section (budget and memory have their own
files, test_time_budget.py and test_time_memory.py):

  [1] the wording, byte for byte, across zones, DST, noon/midnight, and a zone
      whose abbreviation is numeric
  [2] COMPACTOR_TIMEZONE resolution (TZ is NOT read): an invalid zone never
      stops the process importing, falls back to UTC, and says so loudly
      exactly once; no tzdata at all still yields UTC
 [2b] HER BROWSER'S ZONE: the `User timezone:` line OpenWebUI renders from
      {{CURRENT_TIMEZONE}} into the LEADING system message wins; an unrendered
      placeholder or an invalid name falls back (env, then UTC); the label in
      a user or assistant turn, or in a later system message, is ignored; two
      zones give the right local times; the system prompt is forwarded
      unchanged
  [3] COMPACTOR_TIME_INJECTION parsing
  [4] the real route, non-streaming and streaming: exactly one line, in the
      newest user message, never in the leading system block, never as an
      extra message; a content-list turn gets a text PART; no user message
      does not crash; a continuation payload still dates the newest user turn
  [5] PREFIX STABILITY: two requests a minute apart, and a request with the
      feature off, forward byte-identical messages everywhere except the
      newest user message
  [6] task traffic (OpenWebUI title/tags/follow-ups, and repeat task traffic
      on a stable id) gets no line, and the memory tail classifies it exactly
      as it did before
  [7] chat commands never reach the backend and store nothing dated
  [8] the memory tail is handed the ORIGINAL request and the reply, identical
      with the feature on and off, for both paths; backfill too
  [9] the injection has exactly one call site, inside chat_completions, so
      no admin handler or replay path can carry it
 [10] /health/full's config names the source (browser/env/utc), the zone and
      the line the model is shown

    python test_time_injection.py
"""

import ast
import asyncio
import datetime as dt
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("MODEL_REPO", "test-model")
os.environ.setdefault("VLLM_URL", "http://stub:8000")
os.environ["COMPACTOR_RAG_ENABLED"] = "false"
os.environ["COMPACTOR_FACTS_EXTRACTION"] = "false"
os.environ["COMPACTOR_STORAGE_ROOT"] = tempfile.mkdtemp(prefix="time-inject-")
# The zone this process runs in is the default, so the route sections below
# know exactly what line to expect.
os.environ.pop("COMPACTOR_TIMEZONE", None)
os.environ.pop("TZ", None)
os.environ.pop("COMPACTOR_TIME_INJECTION", None)

import memory  # noqa: E402

memory.ensure_storage_layout()

import httpx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import envcfg  # noqa: E402
import main  # noqa: E402
import summarizer  # noqa: E402
import tailhealth  # noqa: E402

FAILED: list[str] = []
NEEDLE = "Current date and time"


def check(cond, label):
    print(("  ok   " if cond else "FAIL ") + label)
    if not cond:
        FAILED.append(label)


def need(name):
    """The feature's API, or a recorded FAIL — never a traceback that hides
    every later check (this file must be readable against unmodified code)."""
    obj = getattr(main, name, None)
    if obj is None:
        check(False, f"main.{name} exists")
    return obj


try:
    from zoneinfo import ZoneInfo
    ZoneInfo("America/Phoenix")
    HAVE_TZDATA = True
except Exception:
    HAVE_TZDATA = False
if not HAVE_TZDATA and sys.platform != "win32":
    # The production image and the unit image both carry /usr/share/zoneinfo.
    # A Linux run without it is an environment defect, not a skip.
    check(False, "fixture: tzdata is present on this Linux platform")

UTC = dt.timezone.utc
T0 = dt.datetime(2026, 9, 14, 16, 41, 7, tzinfo=UTC)   # a Monday


class _Lines(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, r):
        self.records.append(r)

    def text(self):
        return "\n".join(r.getMessage() for r in self.records)


# ===========================================================================
print("[1] the wording")
fmt = need("_format_time_line")
if fmt:
    check(fmt(T0, UTC) ==
          "[Current date and time: Monday, September 14, 2026, 4:41 PM UTC]",
          f"UTC, minute precision, seconds dropped: {fmt(T0, UTC)!r}")
    check(fmt(dt.datetime(2026, 9, 14, 0, 5, tzinfo=UTC), UTC).endswith(", 12:05 AM UTC]"),
          "just after midnight reads 12:05 AM, not 0:05")
    check(fmt(dt.datetime(2026, 9, 14, 12, 0, tzinfo=UTC), UTC).endswith(", 12:00 PM UTC]"),
          "noon reads 12:00 PM")
    check(fmt(dt.datetime(2026, 12, 31, 23, 59, tzinfo=UTC), UTC) ==
          "[Current date and time: Thursday, December 31, 2026, 11:59 PM UTC]",
          "weekday and month names are fixed English, not the process locale")
    if HAVE_TZDATA:
        check(fmt(T0, ZoneInfo("America/Phoenix")) ==
              "[Current date and time: Monday, September 14, 2026, 9:41 AM MST (UTC-07:00)]",
              f"America/Phoenix: abbreviation AND offset: {fmt(T0, ZoneInfo('America/Phoenix'))!r}")
        check("MDT (UTC-06:00)" in fmt(T0, ZoneInfo("America/Denver"))
              and "MST (UTC-07:00)" in fmt(dt.datetime(2026, 1, 12, 16, 41, tzinfo=UTC),
                                            ZoneInfo("America/Denver")),
              "DST: America/Denver is MDT in September and MST in January")
        check(fmt(T0, ZoneInfo("Asia/Kolkata")) ==
              "[Current date and time: Monday, September 14, 2026, 10:11 PM IST (UTC+05:30)]",
              "a half-hour offset is written in full")
        _sp = fmt(T0, ZoneInfo("America/Sao_Paulo"))
        check(_sp == "[Current date and time: Monday, September 14, 2026, 1:41 PM UTC-03:00]",
              f"a numeric abbreviation ('-03') is not printed twice: {_sp!r}")
        check(fmt(dt.datetime(2026, 9, 14, 23, 30, tzinfo=UTC), ZoneInfo("Pacific/Auckland"))
              .startswith("[Current date and time: Tuesday, September 15, 2026, 11:30 AM NZST"),
              "the DATE is the zone's date, not UTC's (Auckland is already Tuesday)")
    for probe in (T0, dt.datetime(2026, 2, 1, 3, 4, tzinfo=UTC)):
        line = fmt(probe, UTC)
        check("\n" not in line and line.startswith("[") and line.endswith("]")
              and line.count(NEEDLE) == 1 and len(line.encode()) < 100,
              f"one bracketed line, under 100 bytes: {line!r}")


# ===========================================================================
print("[2] timezone resolution")
_resolve_impl = need("_resolve_time_zone")


def resolve(environ):
    """_resolve_time_zone, with a RAISE recorded as a named FAIL: raising is
    the defect this section exists for, and a traceback would name nothing
    and hide every later check. The stand-in result fails every check below
    (a +13:00 zone that is not called UTC)."""
    try:
        return _resolve_impl(environ)
    except Exception as e:
        check(False, f"_resolve_time_zone({environ}) never raises "
                     f"({type(e).__name__}: {e})")
        return dt.timezone(dt.timedelta(hours=13)), "<raised>", "<raised>", None


if _resolve_impl:
    z, name, source, err = resolve({})
    check(z.utcoffset(T0) == dt.timedelta(0) and name == "UTC" and err is None,
          f"nothing set -> UTC, no error ({name!r}, {source!r}, {err!r})")
    z, name, source, err = resolve({"COMPACTOR_TIMEZONE": "   "})
    check(name == "UTC" and err is None, "a blank COMPACTOR_TIMEZONE is unset, not invalid")
    z, name, source, err = resolve({"COMPACTOR_TIMEZONE": "Mars/Olympus_Mons"})
    check(z.utcoffset(T0) == dt.timedelta(0) and name == "UTC"
          and err and "Mars/Olympus_Mons" in err,
          f"an invalid name -> UTC, and the error names what was set: {err!r}")
    z, name, source, err = resolve({"COMPACTOR_TIMEZONE": "../../etc/passwd"})
    check(name == "UTC" and err, f"a path-shaped name is refused, not opened: {err!r}")
    z, name, source, err = resolve({"COMPACTOR_TIMEZONE": "Mars/Olympus_Mons",
                                    "TZ": "America/Phoenix"})
    check(name == "UTC" and err,
          "an invalid COMPACTOR_TIMEZONE falls back to UTC, not to TZ")
    if HAVE_TZDATA:
        z, name, source, err = resolve({"COMPACTOR_TIMEZONE": " America/Phoenix "})
        check(name == "America/Phoenix" and source == "COMPACTOR_TIMEZONE" and err is None
              and z.utcoffset(T0.replace(tzinfo=None)) == dt.timedelta(hours=-7),
              f"COMPACTOR_TIMEZONE is trimmed and used ({name!r}, {source!r})")
        z, name, source, err = resolve({"TZ": "America/Phoenix"})
        check(name == "UTC" and err is None,
              "TZ is NOT a source: the owner's order is browser, COMPACTOR_TIMEZONE, UTC "
              f"({name!r}, {source!r})")
        z, name, source, err = resolve({"COMPACTOR_TIMEZONE": "Asia/Tokyo",
                                        "TZ": "America/Phoenix"})
        check(name == "Asia/Tokyo", "COMPACTOR_TIMEZONE is used whatever TZ says")

    # No tzdata anywhere (the Windows interpreter, a slim image): UTC must still
    # work, because it never needed a database.
    import zoneinfo as _zi

    def _no_db(key):
        raise _zi.ZoneInfoNotFoundError(f"No time zone found with key {key}")

    with patch.object(main, "ZoneInfo", _no_db, create=True):
        z, name, source, err = resolve({"COMPACTOR_TIMEZONE": "UTC"})
        check(name == "UTC" and err is None and z.utcoffset(T0) == dt.timedelta(0),
              f"no tzdata: COMPACTOR_TIMEZONE=UTC is still UTC, not an error ({err!r})")
        z, name, source, err = resolve({"COMPACTOR_TIMEZONE": "America/Phoenix"})
        check(name == "UTC" and err, "no tzdata: a real zone falls back to UTC, recorded")

    # hostile pass #5 F1: the shipped image (Ubuntu 24.04, no tzdata-legacy)
    # resolves a zone's CURRENT name but not its "backward" links -
    # SP\p5-a\tzprobe.out measured America/Phoenix OK, US/Arizona FAIL, on
    # that exact image. Simulated here (rather than depending on this test
    # host's own tzdata) so the check holds on Windows and on a host that DID
    # install tzdata-legacy: only the split matters, not which names.
    def _legacy_split(key):
        if key in main._LEGACY_ZONE_ALIASES.values():
            return _zi.ZoneInfo(key)
        raise _zi.ZoneInfoNotFoundError(f"No time zone found with key {key}")

    if HAVE_TZDATA:
        with patch.object(main, "ZoneInfo", _legacy_split, create=True):
            z, name, source, err = resolve({"COMPACTOR_TIMEZONE": "US/Arizona"})
            check(name == "UTC" and err and "America/Phoenix" in err
                  and "resolves" in err,
                  f"a backward-link name's error suggests the canonical "
                  f"replacement that resolves here: {err!r}")
            z, name, source, err = resolve({"COMPACTOR_TIMEZONE": "Mars/Olympus_Mons"})
            check(name == "UTC" and err and "did you mean" not in err,
                  f"CONTROL: a name with no known alias gets no suggestion: {err!r}")
            z, name, source, err = resolve({"COMPACTOR_TIMEZONE": "Asia/Calcutta"})
            check(name == "UTC" and err and "Asia/Kolkata" in err,
                  f"a second alias, same table: {err!r}")

# Import never fails on a bad zone, and the operator hears about it ONCE, at
# ERROR, from the first chat request if nothing announced it at boot.
_CHILD = r'''
import logging, os, sys, tempfile
os.environ.update({"MODEL_REPO": "test-model", "VLLM_URL": "http://stub:8000",
    "COMPACTOR_RAG_ENABLED": "false", "COMPACTOR_FACTS_EXTRACTION": "false",
    "COMPACTOR_STORAGE_ROOT": tempfile.mkdtemp(prefix="time-boot-"),
    "COMPACTOR_TIMEZONE": "Mars/Olympus_Mons"})
import memory; memory.ensure_storage_layout()
import main
recs = []
class H(logging.Handler):
    def emit(self, r): recs.append(r)
logging.getLogger("compactor").addHandler(H())
main._announce_time_zone()
main._announce_time_zone()
errs = [r for r in recs if r.levelno >= logging.ERROR and "Mars/Olympus_Mons" in r.getMessage()]
print("RESULT", len(errs), main.time_injection_state()["fallback_timezone"],
      main.current_time_line().endswith(" UTC]"))
'''
p = subprocess.run([sys.executable, "-c", _CHILD], capture_output=True, text=True,
                   encoding="utf-8", errors="replace", timeout=300,
                   cwd=os.path.dirname(os.path.abspath(__file__)),
                   env=dict(os.environ, PYTHONIOENCODING="utf-8"))
res = next((ln.split()[1:] for ln in p.stdout.splitlines() if ln.startswith("RESULT")), None)
check(p.returncode == 0 and res is not None,
      f"an invalid COMPACTOR_TIMEZONE does not stop main importing (rc={p.returncode}"
      f"{'' if res else ', stderr tail: ' + p.stderr.strip()[-300:]!r})")
check(res == ["1", "UTC", "True"],
      f"it is announced at ERROR exactly once, and the process runs on UTC ({res})")


# ===========================================================================
print("[3] COMPACTOR_TIME_INJECTION")
env_bool = getattr(envcfg, "env_bool", None)
check(env_bool is not None, "envcfg.env_bool exists")
if env_bool:
    for raw, want in (("false", False), ("0", False), ("no", False), ("off", False),
                      (" FALSE\n", False), ("Off", False),
                      ("true", True), ("1", True), ("yes", True), ("on", True),
                      ("", True), ("   ", True)):
        with patch.dict(os.environ, {"X_TIME_BOOL": raw}):
            check(env_bool("X_TIME_BOOL", True) is want, f"{raw!r} -> {want}")
    with patch.dict(os.environ, {"X_TIME_BOOL": "flase"}):
        check(env_bool("X_TIME_BOOL", True) is True and env_bool("X_TIME_BOOL", False) is False,
              "an unrecognised spelling is the DEFAULT, never a raise (envcfg's contract)")
    os.environ.pop("X_TIME_BOOL", None)
    check(env_bool("X_TIME_BOOL", True) is True, "unset -> default")
check(getattr(main, "TIME_INJECTION_ENABLED", None) is True,
      "the feature is ON by default")


# ===========================================================================
print("[2b] her browser's zone, from the leading system message")
_bz_impl = need("_browser_time_zone")
_rz_impl = need("_request_time_zone")


def _no_raise(fn, stand_in):
    """A raise from the zone readers is recorded as a named FAIL, never a
    traceback that hides the rest of the file (mutation M10a hit exactly that)."""
    if fn is None:
        return None

    def wrapped(msgs):
        try:
            return fn(msgs)
        except Exception as e:
            check(False, f"{fn.__name__} never raises ({type(e).__name__}: {e})")
            return stand_in
    return wrapped


bz = _no_raise(_bz_impl, ("<raised>", "<raised>", "<raised>"))
rz = _no_raise(_rz_impl, (None, "<raised>", "<raised>", "<raised>"))
LABEL = getattr(main, "TIME_ZONE_PROMPT_LABEL", None)
check(LABEL == "User timezone:", f"the documented label is exactly 'User timezone:' ({LABEL!r})")
PERSONA = "You are a patient assistant.\nUser timezone: {}\nSpeak plainly."


def sysmsg(zone_text):
    return {"role": "system", "content": PERSONA.format(zone_text)}


if bz and rz and HAVE_TZDATA:
    z, name, err = bz([sysmsg("America/Phoenix"), {"role": "user", "content": "hi"}])
    check(name == "America/Phoenix" and err is None
          and z.utcoffset(T0.replace(tzinfo=None)) == dt.timedelta(hours=-7),
          f"the rendered line in the leading system message is read ({name!r}, {err!r})")
    z, name, err = bz([{"role": "system", "content": "Persona.\r\nUser timezone: Asia/Tokyo\r\n"}])
    check(name == "Asia/Tokyo", "a CRLF system prompt is read too")
    z, name, err = bz([sysmsg("{{CURRENT_TIMEZONE}}")])
    # "not a rendered time zone", not the generic invalid-name reason: the two
    # need different fixes (the client does not render variables, vs a typo),
    # and the operator reads this string in the log and in /health/full.
    check(z is None and err and "not a rendered time zone" in err,
          f"an unrendered placeholder is refused, and SAYS it is unrendered ({err!r})")
    z, name, err = bz([sysmsg("Mars/Olympus_Mons")])
    check(z is None and err and "Mars/Olympus_Mons" in err, f"an invalid name is refused ({err!r})")
    z, name, err = bz([sysmsg("")])
    check(z is None and err, "an empty rendered value is refused")
    check(bz([{"role": "system", "content": "No zone here."}]) == (None, None, None)
          and bz([]) == (None, None, None),
          "no label: no zone and no error, an ordinary request")
    # THE SPOOFS. Only messages[0], and only when it is a system message.
    for label, msgs in (
        ("a user turn", [{"role": "system", "content": "Persona."},
                         {"role": "user", "content": "User timezone: Asia/Tokyo"}]),
        ("an assistant turn", [{"role": "system", "content": "Persona."},
                               {"role": "assistant", "content": "User timezone: Asia/Tokyo"},
                               {"role": "user", "content": "hi"}]),
        ("a user turn with NO system message",
         [{"role": "user", "content": "User timezone: Asia/Tokyo"}]),
        ("a second system message", [{"role": "system", "content": "Persona."},
                                     {"role": "system", "content": "User timezone: Asia/Tokyo"},
                                     {"role": "user", "content": "hi"}]),
        # A VALID zone after the label, so a label that is not anchored to the
        # start of a line would actually be read (an invalid one would be
        # refused anyway and prove nothing - mutation N1c).
        ("the middle of a system-prompt line",
         [{"role": "system", "content": "Format example - User timezone: Asia/Tokyo"}]),
    ):
        check(bz(msgs) == (None, None, None),
              f"the label in {label} is ignored entirely ({bz(msgs)})")

    # PRECEDENCE: browser, then COMPACTOR_TIMEZONE, then UTC.
    env_on = dict(_TIME_ZONE=ZoneInfo("Asia/Tokyo"), TIME_ZONE_NAME="Asia/Tokyo",
                  _TIME_ZONE_SOURCE="COMPACTOR_TIMEZONE", _TIME_ZONE_ERROR=None)
    env_bad = dict(_TIME_ZONE=dt.timezone.utc, TIME_ZONE_NAME="UTC",
                   _TIME_ZONE_SOURCE="COMPACTOR_TIMEZONE", _TIME_ZONE_ERROR="bad name")
    env_unset = dict(_TIME_ZONE=dt.timezone.utc, TIME_ZONE_NAME="UTC",
                     _TIME_ZONE_SOURCE="default", _TIME_ZONE_ERROR=None)

    def with_env(env, msgs):
        ps = [patch.object(main, k, v) for k, v in env.items()]
        for p_ in ps:
            p_.start()
        try:
            return rz(msgs)
        finally:
            for p_ in ps:
                p_.stop()

    r_ = with_env(env_on, [sysmsg("America/Phoenix")])
    check(r_[1:3] == ("America/Phoenix", "browser"), f"browser beats COMPACTOR_TIMEZONE ({r_[1:]})")
    r_ = with_env(env_on, [sysmsg("{{CURRENT_TIMEZONE}}")])
    check(r_[1:3] == ("Asia/Tokyo", "env") and r_[3],
          f"unrendered placeholder -> COMPACTOR_TIMEZONE, reason kept ({r_[1:]})")
    r_ = with_env(env_on, [sysmsg("Mars/Olympus_Mons")])
    check(r_[1:3] == ("Asia/Tokyo", "env"), f"invalid browser zone -> COMPACTOR_TIMEZONE ({r_[1:]})")
    r_ = with_env(env_on, [{"role": "user", "content": "User timezone: America/Phoenix"}])
    check(r_[1:3] == ("Asia/Tokyo", "env") and r_[3] is None,
          f"a user-turn spoof leaves COMPACTOR_TIMEZONE in force ({r_[1:]})")
    r_ = with_env(env_unset, [sysmsg("{{CURRENT_TIMEZONE}}")])
    check(r_[1:3] == ("UTC", "utc"), f"no browser zone and no COMPACTOR_TIMEZONE -> UTC ({r_[1:]})")
    r_ = with_env(env_bad, [{"role": "user", "content": "hi"}])
    check(r_[1:3] == ("UTC", "utc"), f"an INVALID COMPACTOR_TIMEZONE is not 'env' ({r_[1:]})")
    r_ = with_env(env_unset, [sysmsg("Asia/Kolkata")])
    check(r_[1:3] == ("Asia/Kolkata", "browser"), f"browser zone with no env set ({r_[1:]})")
elif not HAVE_TZDATA:
    print("  NOTE [2b] zone checks need tzdata; they run on Linux (the unit image)")


# ===========================================================================
# The route harness: a stub backend that records every body it is sent.
# ===========================================================================

class _Backend:
    bodies: list[dict] = []
    reply = "An ordinary reply about item seven."

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None, **kw):
        _Backend.bodies.append(json)
        return httpx.Response(200, json={"choices": [{"index": 0, "message": {
            "role": "assistant", "content": _Backend.reply}, "finish_reason": "stop"}]},
            request=httpx.Request("POST", url))

    def stream(self, method, url, json=None, **kw):
        _Backend.bodies.append(json)
        return _StreamCM(_Backend.reply)

    async def aclose(self):
        pass


class _StreamResp:
    status_code = 200

    def __init__(self, text):
        self._chunks = [
            f"data: {json.dumps({'choices': [{'delta': {'content': text}}]})}\n\n".encode(),
            f"data: {json.dumps({'choices': [{'delta': {}, 'finish_reason': 'stop'}]})}\n\n".encode(),
            b"data: [DONE]\n\n",
        ]

    async def aread(self):
        return b""

    async def aiter_raw(self):
        for c in self._chunks:
            yield c


class _StreamCM:
    def __init__(self, text):
        self._r = _StreamResp(text)

    async def __aenter__(self):
        return self._r

    async def __aexit__(self, *exc):
        return False


client = TestClient(main.app, client=("127.0.0.1", 12345), raise_server_exceptions=False)
_tails: list[dict] = []


def _spy_tail(conv_id, text, **kw):
    _tails.append({"conv_id": conv_id, "text": text, **kw})
    return main.TailDecision(False, "", tailhealth.SKIPPED_EMPTY, "spy", 0)


def post(msgs, conv="time-conv", stream=False, enabled=True, at=T0, spy_tail=True,
         extra=None):
    """POST through the real route. Returns (status, forwarded chat bodies, log)."""
    _Backend.bodies = []
    cap = _Lines()
    lg = logging.getLogger("compactor")
    prev = lg.level
    lg.addHandler(cap)
    lg.setLevel(logging.DEBUG)
    patches = [patch.object(main.httpx, "AsyncClient", _Backend),
               patch.object(main, "TIME_INJECTION_ENABLED", enabled, create=True),
               patch.object(main, "_now_utc", lambda: at, create=True),
               patch.object(main, "_fire_and_forget",
                            lambda coro, label=None: (coro.close(), True)[1])]
    if spy_tail:
        patches.append(patch.object(main, "_run_memory_tail", _spy_tail))
    try:
        for p_ in patches:
            p_.start()
        body = {"model": "test-model", "messages": json.loads(json.dumps(msgs)),
                "stream": stream}
        body.update(extra or {})
        headers = {"X-Conversation-Id": conv} if conv else {}
        r = client.post("/v1/chat/completions", json=body, headers=headers)
        _ = r.text
    finally:
        for p_ in reversed(patches):
            p_.stop()
        lg.removeHandler(cap)
        lg.setLevel(prev)
    return r.status_code, list(_Backend.bodies), cap.text()


def count_line(payload_msgs) -> int:
    return json.dumps(payload_msgs).count(NEEDLE)


def newest_user(msgs):
    return next((m for m in reversed(msgs) if m.get("role") == "user"), None)


LINE0 = (need("_format_time_line") or (lambda *_: "<missing>"))(T0, UTC)
CONVO = [
    {"role": "system", "content": "You are a patient assistant."},
    {"role": "user", "content": "Tell me about item one."},
    {"role": "assistant", "content": "Item one is a blue box."},
    {"role": "user", "content": "And item two?"},
]


# ===========================================================================
print("[4] the real route")
for stream in (False, True):
    tag = "stream" if stream else "non-stream"
    status, bodies, log = post(CONVO, conv=f"route-{tag}", stream=stream)
    fwd = bodies[-1]["messages"] if bodies else []
    nu = newest_user(fwd) or {}
    check(status == 200 and len(bodies) == 1, f"{tag}: fixture: HTTP {status}, one backend call")
    check(nu.get("content") == f"{LINE0}\n\nAnd item two?",
          f"{tag}: the newest user message is the line, a blank line, then her text: "
          f"{str(nu.get('content'))[:120]!r}")
    check(count_line(fwd) == 1, f"{tag}: exactly one time line in the whole payload")
    check(fwd and fwd[0].get("role") == "system" and NEEDLE not in json.dumps(fwd[0]),
          f"{tag}: the leading system block does not carry it")
    check(len(fwd) == len(CONVO) and [m["role"] for m in fwd] == [m["role"] for m in CONVO],
          f"{tag}: no message was added; roles still alternate "
          f"({[m.get('role') for m in fwd]})")
    status_off, bodies_off, _ = post(CONVO, conv=f"route-{tag}-off", stream=stream,
                                     enabled=False)
    fwd_off = bodies_off[-1]["messages"] if bodies_off else []
    check(status_off == 200 and count_line(fwd_off) == 0
          and (newest_user(fwd_off) or {}).get("content") == "And item two?",
          f"{tag} CONTROL: COMPACTOR_TIME_INJECTION=false forwards her text untouched")

# HER BROWSER'S ZONE through the real route: two zones, two local times, and
# the system prompt forwarded byte-for-byte (the model can read the line too).
if HAVE_TZDATA:
    for zone_name, want in (
        ("America/Phoenix", "Monday, September 14, 2026, 9:41 AM MST (UTC-07:00)"),
        ("Asia/Tokyo", "Tuesday, September 15, 2026, 1:41 AM JST (UTC+09:00)"),
    ):
        sent_msgs = [sysmsg(zone_name)] + CONVO[1:]
        status, bodies, _ = post(sent_msgs, conv=f"route-browser-{zone_name.replace('/', '-')}")
        fwd = bodies[-1]["messages"] if bodies else []
        check(status == 200 and str((newest_user(fwd) or {}).get("content", "")).startswith(
                  f"[Current date and time: {want}]\n\n"),
              f"browser zone {zone_name}: {str((newest_user(fwd) or {}).get('content'))[:90]!r}")
        check(bool(fwd) and fwd[0].get("content") == sent_msgs[0]["content"],
              f"browser zone {zone_name}: the system prompt, label line included, is forwarded unchanged")
    status, bodies, _ = post(CONVO[:-1] + [{"role": "user", "content":
                                            "User timezone: Asia/Tokyo\nAnd item two?"}],
                             conv="route-spoof")
    fwd = bodies[-1]["messages"] if bodies else []
    check(str((newest_user(fwd) or {}).get("content", "")).startswith(LINE0 + "\n\n"),
          "a user who TYPES 'User timezone: Asia/Tokyo' does not move the clock (still UTC)")

# A content-list turn (an image, or any client that sends parts).
LIST_TURN = [{"type": "text", "text": "What is in this picture?"},
             {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]
with patch.object(main, "backend_is_multimodal", lambda: True):
    status, bodies, _ = post(CONVO[:-1] + [{"role": "user", "content": LIST_TURN}],
                             conv="route-list")
fwd = bodies[-1]["messages"] if bodies else []
c = (newest_user(fwd) or {}).get("content")
check(isinstance(c, list) and len(c) == 3
      and c[0] == {"type": "text", "text": LINE0 + "\n\n"} and c[1:] == LIST_TURN,
      f"content-list turn: a leading TEXT PART is prepended, the parts she sent are "
      f"untouched: {json.dumps(c)[:160]}")
check(count_line(fwd) == 1, "content-list turn: exactly one line")

# Two user turns in a row are merged by the route before forwarding; the line
# must be at the head of the merged message, once.
status, bodies, _ = post(CONVO + [{"role": "user", "content": "Also item three."}],
                         conv="route-merge")
fwd = bodies[-1]["messages"] if bodies else []
c = str((newest_user(fwd) or {}).get("content"))
check(c.startswith(LINE0 + "\n\n") and count_line(fwd) == 1 and "Also item three." in c,
      f"merged consecutive user turns: one line, at the head: {c[:140]!r}")

# No user message at all: the same answer with the feature on and off, and no
# crash (the tail repair adds continue_final_message, the backend still called).
NO_USER = [{"role": "system", "content": "You are a patient assistant."},
           {"role": "assistant", "content": "Hello."}]
s_on, b_on, log_on = post(NO_USER, conv="route-nouser")
s_off, b_off, _ = post(NO_USER, conv="route-nouser-off", enabled=False)
check(s_on == s_off and s_on < 500 and "Traceback" not in log_on,
      f"no user message: HTTP {s_on} on, {s_off} off, no crash")
check(all(count_line(b["messages"]) == 0 for b in b_on)
      and [b["messages"] for b in b_on] == [b["messages"] for b in b_off],
      "no user message: nothing is injected anywhere")

# A continuation payload (assistant-final with content): the newest USER turn
# is interior; it is still the one dated.
CONT = CONVO + [{"role": "assistant", "content": "Item two is"}]
status, bodies, _ = post(CONT, conv="route-cont")
fwd = bodies[-1]["messages"] if bodies else []
check(status == 200 and bodies and bodies[-1].get("continue_final_message") is True
      and fwd[-1] == CONT[-1]
      and (newest_user(fwd) or {}).get("content") == f"{LINE0}\n\nAnd item two?"
      and count_line(fwd) == 1,
      "continuation: the assistant tail is untouched, the newest user turn is dated")


# ===========================================================================
print("[5] prefix stability: only the newest user message ever differs")
T1 = T0 + dt.timedelta(minutes=1, seconds=13)
LONG = [{"role": "system", "content": "You are a patient assistant."}]
for i in range(1, 9):
    LONG += [{"role": "user", "content": f"Tell me about item {i}."},
             {"role": "assistant", "content": f"Item {i} is a box of size {i}."}]
LONG.append({"role": "user", "content": "Which item is largest?"})
_, b_a, _ = post(LONG, conv="prefix-a", at=T0)
_, b_b, _ = post(LONG, conv="prefix-a", at=T1)
_, b_off, _ = post(LONG, conv="prefix-a", enabled=False)
A, B, OFF = (b[-1]["messages"] if b else [] for b in (b_a, b_b, b_off))
idx = max((i for i, m in enumerate(A) if m.get("role") == "user"), default=-1)
check(len(A) == len(B) == len(OFF) and idx == len(A) - 1,
      "fixture: three payloads of equal shape, newest user turn last")
check(A[:idx] == B[:idx] == OFF[:idx],
      "every message before the newest user turn is byte-identical a minute later "
      "AND with the feature off — the prompt prefix vLLM caches is untouched")
check(A[idx] != B[idx] and A[idx]["content"].endswith("Which item is largest?")
      and B[idx]["content"].endswith("Which item is largest?"),
      "CONTROL: the newest user turn does change with the minute")
_ja, _jb = json.dumps(A), json.dumps(B)
_common = next((k for k in range(min(len(_ja), len(_jb))) if _ja[k] != _jb[k]),
               min(len(_ja), len(_jb)))
_start = len(json.dumps(A[:idx])) - 1
check(_common >= _start,
      f"serialized, the two payloads first differ at byte {_common}, at or after the "
      f"newest user message starts (byte {_start}) of {len(_ja)}")


# ===========================================================================
print("[6] task traffic")
TASK_REQ = [{"role": "user", "content": "### Task:\nGenerate a concise title for the chat.\n"
             "### Chat History:\nUSER: tell me about item one"}]
for suffix in ("title_generation", "tags_generation", "follow_up_generation",
               "emoji_generation", "query_generation", "autocomplete_generation"):
    _, bodies, _ = post(TASK_REQ, conv=f"3f2a9c1e-7b1d-4c55-9e0a-1d2b3c4d5e6f{suffix}")
    check(bodies and count_line(bodies[-1]["messages"]) == 0,
          f"OpenWebUI {suffix} traffic (header {{{{CHAT_ID}}}}{{{{TASK}}}}) is not dated")
_, bodies, _ = post(TASK_REQ, conv="3f2a9c1e-7b1d-4c55-9e0a-1d2b3c4d5e6f")
check(bodies and count_line(bodies[-1]["messages"]) == 1,
      "CONTROL: the same single-message array on the chat's own id (a real first turn) IS dated")

# Repeat task traffic on a STABLE id: the tail's own classifier.
REPEAT = "repeat-task-conv"
st = summarizer.load_state(REPEAT)
st["turns_seen"] = main.TASK_TRAFFIC_MIN_POSITION + 2
summarizer.save_state(REPEAT, st)
check(main._is_repeat_task_traffic(REPEAT, TASK_REQ),
      "fixture: the tail's classifier calls this request repeat task traffic")
_tails.clear()
_, bodies, _ = post(TASK_REQ, conv=REPEAT)
check(bodies and count_line(bodies[-1]["messages"]) == 0,
      "repeat task traffic on a stable id is not dated")
_, bodies, _ = post(CONVO, conv=REPEAT)
check(bodies and count_line(bodies[-1]["messages"]) == 1,
      "CONTROL: a real exchange on that id (it has an assistant turn) is dated")

# The classification the TAIL makes is identical with the feature on and off:
# the real _run_memory_tail, a stubbed-out tail coroutine, the outcome counted.


def _outcome(conv, msgs, enabled):
    before = tailhealth.snapshot()
    post(msgs, conv=conv, enabled=enabled, spy_tail=False)
    after = tailhealth.snapshot()
    moved = sorted(k for k, v in after["outcomes"].items()
                   if v != before["outcomes"].get(k, 0))
    return moved, {k: after[k] - before[k] for k in ("stored", "skipped")}


_o_on, _o_off = _outcome(REPEAT, TASK_REQ, True), _outcome(REPEAT, TASK_REQ, False)
check(_o_on == _o_off == ([tailhealth.SKIPPED_TASK_TRAFFIC], {"stored": 0, "skipped": 1}),
      f"repeat task traffic: the tail counts one {tailhealth.SKIPPED_TASK_TRAFFIC}, "
      f"on and off ({_o_on} / {_o_off})")
check(_outcome("fresh-first-turn", TASK_REQ, True)[1]
      == _outcome("fresh-first-turn-2", TASK_REQ, False)[1] == {"stored": 1, "skipped": 0},
      "a real first turn: stored, on and off")


# ===========================================================================
print("[6b] a hash-identity collision must not read as task traffic (F2)")
# hostile pass #5 F2. Under hash identity (production today) a brand-new
# chat's conv_id is sha256(system|||first_user[:512]) — the SAME id an OLDER
# chat gets if it opened with the same line. _is_repeat_task_traffic cannot
# tell "task call, forever" from "new chat, unlucky opener" from history
# alone; only the message's SHAPE can (_looks_like_openwebui_task_prompt).
# An ordinary, non-task-shaped opener must still be dated even though it
# collides, by id, with an older and already-deep conversation. The opener
# text below is a synthetic placeholder, not a real message — deliberately
# generic so it stays that way; only its SHAPE (not task-shaped) matters
# to this test.
OPENER_SYS = "You are a patient assistant."
_opener = [{"role": "system", "content": OPENER_SYS},
           {"role": "user", "content": "What is on the schedule today?"}]
_coll_id, _coll_src = memory.resolve_conv_id({}, _opener, body={})
check(_coll_src == "hash", f"fixture: this request resolves by hash ({_coll_src})")
_coll_st = summarizer.load_state(_coll_id)
_coll_st["turns_seen"] = main.TASK_TRAFFIC_MIN_POSITION + 2   # an older, deep chat
summarizer.save_state(_coll_id, _coll_st)
check(main._is_repeat_task_traffic(_coll_id, _opener),
      "fixture: history alone marks this id as repeat task traffic")
_, bodies, _ = post(_opener, conv=None)
check(bodies and count_line(bodies[-1]["messages"]) == 1,
      "F2: a real opener that hash-collides with an older, deep chat IS dated "
      "(the message's shape, not history alone, gates the dating skip)")
_, bodies, _ = post([{"role": "system", "content": OPENER_SYS},
                     {"role": "user", "content": "What is on the schedule tomorrow?"}], conv=None)
check(bodies and count_line(bodies[-1]["messages"]) == 1,
      "CONTROL: a different opener (no collision, a fresh id) is dated too")
# CONTROL: genuine task-shaped text on an equally "deep" colliding id is
# still not dated — the shape check narrows the skip, it does not remove it.
_task_opener = [{"role": "system", "content": OPENER_SYS}] + TASK_REQ
_task_coll_id, _ = memory.resolve_conv_id({}, _task_opener, body={})
_tst = summarizer.load_state(_task_coll_id)
_tst["turns_seen"] = main.TASK_TRAFFIC_MIN_POSITION + 2
summarizer.save_state(_task_coll_id, _tst)
check(main._looks_like_openwebui_task_prompt(_task_opener),
      "fixture: this one IS task-shaped")
_, bodies, _ = post(_task_opener, conv=None)
check(bodies and count_line(bodies[-1]["messages"]) == 0,
      "CONTROL: task-SHAPED text on an equally deep colliding id is still not dated")


# ===========================================================================
print("[7] chat commands")
import facts  # noqa: E402

for enabled in (True, False):
    conv = f"cmd-conv-{enabled}"
    status, bodies, _ = post(CONVO[:-1] + [{"role": "user", "content":
                                            "/remember item nine is green"}],
                             conv=conv, enabled=enabled)
    stored = facts.load_facts(conv)
    check(status == 200 and bodies == [],
          f"/remember (time injection {'on' if enabled else 'off'}): answered without "
          f"the backend")
    check(stored and all(NEEDLE not in f.get("text", "") for f in stored)
          and any("item nine is green" in f.get("text", "") for f in stored),
          f"/remember stored the fact as typed: {[f.get('text') for f in stored]}")


# ===========================================================================
print("[8] the memory tail is handed the original request, on both paths")
for stream in (False, True):
    tag = "stream" if stream else "non-stream"
    seen = {}
    for enabled in (True, False):
        _tails.clear()
        status, bodies, _ = post(CONVO, conv=f"tail-{tag}", stream=stream, enabled=enabled)
        check(len(_tails) == 1, f"{tag}: fixture: the tail ran once ({len(_tails)})")
        t = _tails[-1] if _tails else {}
        seen[enabled] = {k: t.get(k) for k in ("text", "finished", "truncated", "holed",
                                               "last_user_text", "turn_index", "messages")}
    on = seen.get(True) or {}
    check(NEEDLE not in json.dumps(on.get("messages")) and on.get("last_user_text") == "And item two?",
          f"{tag}: the tail's messages and last_user_text carry no line "
          f"({on.get('last_user_text')!r})")
    check(on.get("text") == _Backend.reply,
          f"{tag}: the tail judges the reply, and only the reply")
    check(seen.get(True) == seen.get(False),
          f"{tag}: everything the tail is handed is identical with the feature on and off")

_bf: list = []


async def _spy_backfill(conv_id, messages, *a, **k):
    _bf.append(json.loads(json.dumps(messages)))
    return False


with patch.object(main.backfill, "start_backfill_if_needed", _spy_backfill):
    post(CONVO, conv="backfill-conv")
check(len(_bf) == 1 and NEEDLE not in json.dumps(_bf) and _bf[0] == CONVO,
      "backfill (which replays the history into facts and summaries) gets the original array")


# ===========================================================================
print("[9] one call site, inside chat_completions")
SRC = Path(main.__file__).read_text(encoding="utf-8")
tree = ast.parse(SRC)
calls: list[str] = []
for fn in ast.walk(tree):
    if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
        for node in ast.walk(fn):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "_inject_time_line"):
                calls.append(fn.name)
check(calls == ["chat_completions"],
      f"_inject_time_line is called from chat_completions and nowhere else ({calls}) — "
      f"so /admin handlers, backfill, the rollup and summarization never add it")
_admin = [fn.name for fn in ast.walk(tree)
          if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
          and fn.name.startswith("admin_")
          and any(isinstance(n, ast.Name) and n.id in ("current_time_line",
                                                       "_time_line_for_request")
                  for n in ast.walk(fn))]
check(_admin == [], f"no admin handler reads the time line ({_admin})")


# ===========================================================================
print("[10] /health/full config")
import health  # noqa: E402

post(CONVO, conv="health-utc")
rep = asyncio.run(health.gather_health_full("http://127.0.0.1:9", 1000))
ti = (rep.get("config") or {}).get("time_injection") or {}
check(ti.get("enabled") is True and ti.get("last_source") == "utc"
      and ti.get("last_timezone") == "UTC" and ti.get("fallback_timezone") == "UTC"
      and ti.get("fallback_error") is None and ti.get("prompt_label") == "User timezone:",
      f"config.time_injection names the source, the zone and whether it is on: {ti}")
check(str(ti.get("current_line", "")).startswith("[Current date and time: ")
      and str(ti.get("current_line", "")).endswith(" UTC]"),
      f"and the exact line the model is shown right now: {ti.get('current_line')!r}")
if HAVE_TZDATA:
    post([sysmsg("America/Phoenix")] + CONVO[1:], conv="health-browser")
    ti = (asyncio.run(health.gather_health_full("http://127.0.0.1:9", 1000)).get("config")
          or {}).get("time_injection") or {}
    check(ti.get("last_source") == "browser" and ti.get("last_timezone") == "America/Phoenix"
          and " (UTC-07:00)]" in str(ti.get("current_line")),
          f"after a browser-zoned request it shows source=browser and her zone: {ti}")
    post([sysmsg("{{CURRENT_TIMEZONE}}")] + CONVO[1:], conv="health-placeholder")
    ti = (asyncio.run(health.gather_health_full("http://127.0.0.1:9", 1000)).get("config")
          or {}).get("time_injection") or {}
    check(ti.get("last_source") == "utc" and "CURRENT_TIMEZONE" in str(ti.get("last_browser_error")),
          f"after an unrendered placeholder: source=utc and the reason ({ti.get('last_browser_error')!r})")


print()
if FAILED:
    print(f"FAILED: {len(FAILED)} check(s)")
    for f in FAILED:
        print(f"  - {f}")
    sys.exit(1)
print("All time-injection checks passed.")
