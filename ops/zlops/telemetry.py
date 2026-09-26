"""Telemetry: what `zl dash` shows, and what `zl health`/`zl bundle` reuse.

Every collector in this module answers one question about the pod (is the
compactor healthy, is vLLM keeping up, is the SSH session itself alive) by
reading something that is ALREADY there - `/health/full`, `supervisorctl`,
`nvidia-smi`, `/proc`, `/data` - never by re-deriving a verdict a repair
script or the compactor already owns. `interpret_health` is the one place
that turns `/health/full`'s raw JSON into an operator verdict, because that
document's shape has changed release to release (see OPERATIONS.md "Reading
/health/full") and every field access here tolerates absence by construction
- tested directly against a v3.1.6.1-shaped document, which has none of
`checks.hierarchy`, `checks.reuse`, `checks.budget_margin`,
`checks.truncated_summaries`, or `checks.ledger`.

Two ways to use this module:

  - `collect_snapshot(options)` - one synchronous pass, used by
    `zl dash --json`/`--once` and by `zl bundle`'s `meta/telemetry.json`.
    Every collector runs once (twice, `rate_gap_s` apart, for the two that
    need a rate: vLLM throughput and host CPU%) under its own timeout; a
    failing collector's section is `{"data": None, "error": "..."}`, never
    an exception that aborts the whole snapshot.

  - `Hub` - one background thread per collector, each on its own cadence,
    each enforcing its own hard timeout, writing into a lock-protected
    dict. `dash.py`'s UI thread only ever reads `Hub.snapshot()`, a shallow
    copy taken under the lock - it never calls a collector directly, so a
    stalled `/data` `du` or an unreachable vLLM can never block a keypress
    or a redraw.

Every collector accepts its I/O (an HTTP fetch function, a `subprocess.run`,
a filesystem root) as an optional parameter with a real default, so tests
inject a fake without touching PATH or the real filesystem - the same
seam `compactor/test_*.py` already uses for `_utc_stamp`-style patching,
just as a function parameter instead of a monkeypatch.

Style: stdlib only. No bare `except:` - every catch names the exception type
it expects and records `error` in the caller's structure; nothing here ever
lets a collector's failure crash the dashboard.
"""

from __future__ import annotations

import http.client
import json
import math
import os
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import xmlrpc.client
from collections import deque
from pathlib import Path
from typing import Any, Callable

from . import podenv

# =============================================================================
# Shared thresholds
# =============================================================================

# Backup-age thresholds, defined once and shared by interpret_health() (the
# `zl health`/status-line verdict) and the Backups panel (the raw hours),
# rather than each re-deriving its own copy - see COMMANDS.md/OPERATIONS.md
# "newest backup must be under about 30 hours" and the memory.md rule
# against two copies of one number drifting apart.
BACKUP_WARN_HOURS = 30.0
BACKUP_CRIT_HOURS = 36.0


def backup_age_level(age_hours: float | None) -> str:
    """"ok"/"warn"/"crit" for a backup's age in hours. None (no backup
    observed at all) is crit, same as an age past the crit threshold.
    """
    if age_hours is None:
        return "crit"
    if age_hours >= BACKUP_CRIT_HOURS:
        return "crit"
    if age_hours >= BACKUP_WARN_HOURS:
        return "warn"
    return "ok"


# A panel's data is shown "stale" (dashboard convention, not a compactor
# concept) once it is older than this multiple of its own collection cadence.
STALE_MULTIPLE = 3.0


# =============================================================================
# Small generic helpers
# =============================================================================


def _call_with_timeout(fn: Callable[[], Any], timeout_s: float) -> tuple[Any, Exception | None]:
    """Run `fn` on a daemon thread and wait up to `timeout_s`.

    Returns (value, None) on success, (None, exception) on failure or
    timeout. `fn` itself keeps running past the timeout on its own thread
    (Python has no safe way to kill a thread) - callers never see it again,
    but nothing here blocks past `timeout_s`. Used for anything that reads
    `/data` (MooseFS can stall) or the filesystem generally, where the
    called function has no timeout parameter of its own (unlike
    `subprocess.run(timeout=...)` or `urlopen(timeout=...)`, which are used
    directly instead).
    """
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["value"] = fn()
        except Exception as exc:  # noqa: BLE001 - recorded, never swallowed silently
            box["error"] = exc

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        return None, TimeoutError(f"timed out after {timeout_s:.0f}s")
    if "error" in box:
        return None, box["error"]
    return box.get("value"), None


def tail_bytes(path: Path, max_bytes: int = 65536) -> str:
    """The last `max_bytes` of `path`, decoded leniently. Raises OSError if
    the file cannot be read at all (missing, permission) - callers decide
    whether that is a fault or just "this pod doesn't run that service".
    """
    size = path.stat().st_size
    with path.open("rb") as f:
        if size > max_bytes:
            f.seek(size - max_bytes)
        data = f.read()
    return data.decode("utf-8", "replace")


def _parse_float(raw: str) -> float | None:
    raw = raw.strip()
    if raw in ("+Inf", "Inf", "inf", "+inf"):
        return math.inf
    if raw in ("-Inf", "-inf"):
        return -math.inf
    if raw.lower() == "nan":
        return math.nan
    try:
        return float(raw)
    except ValueError:
        return None


# =============================================================================
# interpret_health - the seam both `zl health` and the dashboard use
# =============================================================================


def _findings_status(findings: list[dict]) -> str:
    order = {"ok": 0, "info": 0, "warn": 1, "crit": 2}
    worst = 0
    for f in findings:
        worst = max(worst, order.get(f.get("level"), 0))
    return ("ok", "warn", "crit")[worst]


def interpret_health(doc: dict | None, now: float) -> dict:
    """Turn one `/health/full` document into `{"status", "reasons",
    "findings"}` (the fixed seam contract - do not add or rename keys).

    Rules (brief, implemented independently of each other so a v3.1.6.1
    document lacking most `checks.*` keys still gets whatever partial
    picture it can offer):
      1. `stats.unreadable` counts must all be 0 - crit otherwise.
      2. newest backup age: warn >= 30h, crit >= 36h or no backup at all.
      3. `checks.sqlite_journal.hot` - crit.
      4. `status == "degraded"` - warn, one finding per `status_reasons`.
      5. `status == "down"`, or the endpoint unreachable (`doc is None`) -
         crit.
      6. `checks.hierarchy` catch-up verdict "stuck" (or "unknown", which
         OPERATIONS.md says also degrades from v3.1.9) - warn.
      7. `checks.ledger.degraded` (no such field exists in the codebase
         today - grep-confirmed; kept because the brief pins the rule and a
         future release may add it) - crit.

    `checks.reuse`/`checks.budget_margin`/`checks.truncated_summaries` are
    visibility-only in the compactor's own doctrine (OPERATIONS.md says so
    explicitly for each) - they never move `status` here either; the
    dashboard surfaces their raw numbers in a separate panel instead.
    """
    findings: list[dict] = []

    def add(level: str, text: str) -> None:
        findings.append({"level": level, "text": text})

    if doc is None:
        add("crit", "compactor /health/full unreachable")
        return {"status": "crit", "reasons": ["compactor /health/full unreachable"], "findings": findings}

    if not isinstance(doc, dict):
        text = f"compactor /health/full returned a non-object body ({type(doc).__name__})"
        add("crit", text)
        return {"status": "crit", "reasons": [text], "findings": findings}

    # Rule 5 (down/unreachable) and rule 4 (degraded).
    raw_status = doc.get("status")
    status_reasons = doc.get("status_reasons")
    status_reasons = status_reasons if isinstance(status_reasons, list) else []
    if raw_status == "down":
        add("crit", "compactor status: down (storage broken)")
    elif raw_status == "degraded":
        if status_reasons:
            for r in status_reasons:
                add("warn", str(r))
        else:
            add("warn", "compactor status: degraded (no reasons given)")
    elif raw_status == "ok":
        add("ok", "compactor status: ok")
    elif raw_status is None:
        add("warn", "compactor /health/full response has no 'status' field")
    else:
        add("warn", f"compactor status: unrecognized value {raw_status!r}")

    # Rule 1: unreadable memory counts.
    stats = doc.get("stats")
    if isinstance(stats, dict):
        unreadable = stats.get("unreadable")
        if isinstance(unreadable, dict) and unreadable:
            bad = {k: v for k, v in unreadable.items() if isinstance(v, (int, float)) and v > 0}
            if bad:
                add("crit", f"unreadable memory files: {bad}")
            else:
                add("ok", "unreadable memory files: all 0")
        # Missing/not-a-dict (older releases): nothing to check, tolerated.

    checks = doc.get("checks")
    checks = checks if isinstance(checks, dict) else {}

    # Rule 3: sqlite_journal hot.
    sj = checks.get("sqlite_journal")
    if isinstance(sj, dict):
        if sj.get("hot"):
            where = sj.get("path") or sj.get("db_path") or ""
            add("crit", f"sqlite journal is hot: {where}".rstrip())
        elif sj.get("ok") is False:
            add("warn", f"sqlite journal probe reported not-ok: {sj.get('error') or sj}")

    # Rule 6: hierarchy catch-up stuck/unknown.
    hierarchy = checks.get("hierarchy")
    if isinstance(hierarchy, dict):
        entries = hierarchy.get("catching_up_all")
        if not isinstance(entries, list):
            single = hierarchy.get("catching_up")
            entries = [single] if isinstance(single, dict) else []
        stuck = [e for e in entries if isinstance(e, dict) and e.get("verdict") == "stuck"]
        unknown = [e for e in entries if isinstance(e, dict) and e.get("verdict") == "unknown"]
        if stuck:
            ids = [e.get("conv_id") or e.get("id") or "?" for e in stuck]
            add("warn", f"hierarchy catch-up stuck for {len(stuck)} conversation(s): {ids}")
        if unknown:
            add("warn", f"hierarchy catch-up unknown (no evidence yet) for {len(unknown)} conversation(s)")

    # Rule 7: ledger degraded (speculative field - see docstring).
    ledger = checks.get("ledger")
    if isinstance(ledger, dict) and ledger.get("degraded"):
        add("crit", f"ledger degraded: {ledger.get('reason') or ledger}")

    # Rule 2: newest backup age.
    backups = doc.get("backups")
    if isinstance(backups, dict):
        if backups.get("error"):
            add("crit", f"backup status unobservable: {backups['error']}")
        else:
            latest_mtime = backups.get("latest_mtime")
            count = backups.get("count")
            if not latest_mtime or count in (0, None):
                add("crit", "no backup archives found")
            else:
                age_h = (now - latest_mtime) / 3600.0
                level = backup_age_level(age_h)
                text = f"newest backup age: {age_h:.1f}h (latest: {backups.get('latest')})"
                add(level, text)
    # backups missing entirely: not expected on any real release, tolerated.

    status = _findings_status(findings)
    reasons = [f["text"] for f in findings if f["level"] in ("warn", "crit")]
    return {"status": status, "reasons": reasons, "findings": findings}


def fetch_health_full(
    base_url: str, timeout_s: float = 5.0, fetch: Callable[[str, float], bytes] | None = None
) -> dict:
    """GET `{base_url}/health/full` and parse it as JSON. A 503 ("down")
    still carries a diagnostic body, so an HTTPError's body is read too -
    only a transport failure (connection refused, DNS, timeout) raises.
    `fetch(url, timeout_s) -> bytes` is injectable for tests.
    """
    url = f"{base_url}/health/full"
    if fetch is not None:
        body = fetch(url, timeout_s)
    else:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310 - localhost admin endpoint
                body = resp.read()
        except urllib.error.HTTPError as exc:
            body = exc.read()
    if isinstance(body, str):
        body = body.encode("utf-8")
    return json.loads(body.decode("utf-8", "replace"))


def collect_health(url: str | None = None, timeout_s: float = 5.0, fetch=None) -> dict:
    """One `/health/full` fetch plus its interpretation, as one collector
    result (the doc alone is not enough - the dashboard's Health panel also
    shows `checks.reuse`/`checks.budget_margin`/`checks.truncated_summaries`
    straight from `doc`, none of which interpret_health folds into status).
    """
    doc = fetch_health_full(url or podenv.COMPACTOR_URL, timeout_s, fetch=fetch)
    return {"doc": doc, "interpreted": interpret_health(doc, time.time())}


# =============================================================================
# Prometheus text-exposition parser (hand-rolled: stdlib only, no
# prometheus_client dependency on the pod's compactor-venv)
# =============================================================================


class Sample:
    """One Prometheus sample: a metric name, its label set, and a value.

    A plain class (not a dataclass with slots) so `labels` stays a normal
    mutable dict without frozen-dataclass hashing surprises - samples are
    never hashed or compared, only iterated.
    """

    __slots__ = ("name", "labels", "value")

    def __init__(self, name: str, labels: dict[str, str], value: float) -> None:
        self.name = name
        self.labels = labels
        self.value = value

    def __repr__(self) -> str:  # pragma: no cover - debugging aid only
        return f"Sample({self.name!r}, {self.labels!r}, {self.value!r})"


def parse_prometheus_text(text: str) -> dict[str, list[Sample]]:
    """Parse Prometheus text exposition format by hand (no dependency): `#
    HELP`/`# TYPE` comments are skipped, label values are unescaped
    (`\\\\`, `\\"`, `\\n`), values accept `+Inf`/`-Inf`/`NaN`, and a name
    with no `{...}` (a bare gauge/counter line) is handled too. Returns
    `{metric_name: [Sample, ...]}` - a name can repeat (one series per
    label combination, e.g. one per `le` bucket, or one per engine).
    """
    out: dict[str, list[Sample]] = {}
    for raw_line in text.splitlines():
        line = raw_line.rstrip("\r")
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        i = 0
        n = len(line)
        start = i
        while i < n and (line[i].isalnum() or line[i] in "_:"):
            i += 1
        name = line[start:i]
        if not name:
            continue
        labels: dict[str, str] = {}
        if i < n and line[i] == "{":
            i += 1
            while i < n and line[i] != "}":
                while i < n and line[i] in " ,":
                    i += 1
                if i >= n or line[i] == "}":
                    break
                kstart = i
                while i < n and line[i] != "=":
                    i += 1
                key = line[kstart:i].strip()
                i += 1  # skip '='
                if i < n and line[i] == '"':
                    i += 1
                    buf: list[str] = []
                    while i < n and line[i] != '"':
                        ch = line[i]
                        if ch == "\\" and i + 1 < n:
                            nxt = line[i + 1]
                            if nxt == "n":
                                buf.append("\n")
                            elif nxt in ('"', "\\"):
                                buf.append(nxt)
                            else:
                                buf.append(nxt)
                            i += 2
                        else:
                            buf.append(ch)
                            i += 1
                    i += 1  # skip closing quote
                    if key:
                        labels[key] = "".join(buf)
                while i < n and line[i] == ",":
                    i += 1
            if i < n and line[i] == "}":
                i += 1
        while i < n and line[i] == " ":
            i += 1
        rest = line[i:].split()
        if not rest:
            continue
        value = _parse_float(rest[0])
        if value is None:
            continue
        out.setdefault(name, []).append(Sample(name, labels, value))
    return out


# =============================================================================
# Rates and histogram quantiles from two scrapes
# =============================================================================


def counter_rate(prev_value: float, prev_ts: float, cur_value: float, cur_ts: float) -> float | None:
    """Per-second rate between two counter samples. A counter that WENT
    DOWN (vLLM restarted, the counter reset to 0) is treated as if it
    started counting from 0 at `prev_ts` - the delta is `cur_value` alone,
    never negative.
    """
    dt = cur_ts - prev_ts
    if dt <= 0:
        return None
    delta = cur_value if cur_value < prev_value else (cur_value - prev_value)
    return delta / dt


def histogram_quantile_from_buckets(
    prev_buckets: dict[float, float] | None, cur_buckets: dict[float, float] | None, q: float
) -> float | None:
    """Estimate the `q` quantile (0..1) from the DELTA between two
    cumulative-bucket snapshots (`{le: cumulative_count}`), not the
    lifetime histogram - a live dashboard cares about the last window, not
    "since boot". Linear interpolation within the bucket that contains the
    target rank, same as PromQL's `histogram_quantile`.

    A Prometheus bucket count is ALREADY cumulative (`le="0.5"` counts
    every observation `le="0.1"` counts, plus more) - so `cur[le] -
    prev[le]`, taken at the SAME `le`, is itself the cumulative count for
    this window at that `le`. It must never be summed across `le`s as if
    each were an independent per-bin count (that double/triple-counts:
    summing 50+90+100 when the true window total is 100 observations). The
    per-`le` delta sequence is walked in increasing `le` order instead,
    exactly like reading the raw histogram itself, with the LAST `le`'s
    delta as the window total. A single `le` whose delta went down (a
    counter reset) is treated the same way `counter_rate` treats one - the
    new value alone, floored at the previous `le`'s value so the sequence
    stays non-decreasing even if only some buckets reset.
    """
    if not prev_buckets or not cur_buckets:
        return None
    les = sorted(set(prev_buckets) | set(cur_buckets))
    cum: list[tuple[float, float]] = []
    running = 0.0
    for le in les:
        p = prev_buckets.get(le, 0.0)
        c = cur_buckets.get(le, 0.0)
        d = c if c < p else (c - p)
        running = max(running, d)  # never decreasing across le, even under a partial reset
        cum.append((le, running))
    total = cum[-1][1] if cum else 0.0
    if total <= 0:
        return None
    target = q * total
    prev_le = 0.0
    prev_cum = 0.0
    for le, d in cum:
        if d >= target:
            bin_count = d - prev_cum
            if bin_count <= 0 or math.isinf(le):
                # Can't interpolate a zero-width bin or past the last finite
                # bucket; report the lower edge of where the rank landed.
                return prev_le
            frac = (target - prev_cum) / bin_count
            return prev_le + frac * (le - prev_le)
        prev_le = le
        prev_cum = d
    last_finite = [le for le in les if not math.isinf(le)]
    return last_finite[-1] if last_finite else None


# =============================================================================
# nvidia-smi
# =============================================================================


def _na(raw: str) -> float | None:
    raw = raw.strip()
    if raw in ("[N/A]", "N/A", ""):
        return None
    try:
        return float(raw)
    except ValueError:
        return None


_NVIDIA_SMI_FIELDS = (
    "name",
    "utilization_gpu_pct",
    "memory_used_mb",
    "memory_total_mb",
    "temperature_c",
    "power_draw_w",
    "power_limit_w",
    "clock_sm_mhz",
)


def parse_nvidia_smi_csv(text: str) -> list[dict]:
    """Parse `nvidia-smi --query-gpu=... --format=csv,noheader,nounits`
    output: one row per GPU, `[N/A]` fields (common for power/clocks on
    some cards/drivers) become None rather than crashing a float() call.
    """
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < len(_NVIDIA_SMI_FIELDS):
            continue
        row: dict[str, Any] = {"name": parts[0]}
        for field, raw in zip(_NVIDIA_SMI_FIELDS[1:], parts[1:]):
            row[field] = _na(raw)
        out.append(row)
    return out


def collect_gpu(timeout_s: float = 3.0, run: Callable[..., Any] = subprocess.run) -> list[dict]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit,clocks.sm",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = run(cmd, capture_output=True, text=True, timeout=timeout_s, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError("nvidia-smi not found (no GPU runtime attached)") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"nvidia-smi timed out after {timeout_s:.0f}s") from exc
    if proc.returncode != 0:
        raise RuntimeError(f"nvidia-smi exited {proc.returncode}: {(proc.stderr or '').strip()[:200]}")
    return parse_nvidia_smi_csv(proc.stdout)


# =============================================================================
# supervisord: XML-RPC over its unix socket (cheap, no process fork), with a
# `supervisorctl status` subprocess fallback (works even if the RPC
# interface is disabled/unreachable, but costs a fork+exec every poll - so
# the Hub polls it at a slower cadence than the RPC path; see
# `CADENCE_S`/`Hub._tick`'s services special-case below). Measured on a dev
# box: polling `supervisorctl status` every 5s cost ~7% of a core (the
# hostile review's own number); the RPC path over a unix socket is a single
# syscall round trip, no fork - see the brief's hand-off report for the
# measured before/after.
# =============================================================================

_SUP_LINE_RE = re.compile(r"^(?P<name>\S+)\s+(?P<state>\S+)\s*(?P<rest>.*)$")
_SUP_PID_RE = re.compile(r"\bpid (\d+)\b")
_SUP_UPTIME_RE = re.compile(r"\buptime ([0-9:]+(?:days?,\s*[0-9:]+)?)")

# supervisord.conf: `[unix_http_server] file=/var/run/supervisor.sock`.
# Read from the conf file (not hardcoded) so a future path change is picked
# up automatically; falls back to the documented default if the conf can't
# be read (e.g. a dev container with no supervisord at all).
SUPERVISORD_CONF_PATHS = (
    Path("/etc/supervisor/conf.d/supervisord.conf"),  # Dockerfile:357 - the real pod path
    Path(__file__).resolve().parents[2] / "supervisord.conf",  # repo checkout, for dev/tests
)
DEFAULT_SUPERVISOR_SOCKET = Path("/var/run/supervisor.sock")


def read_supervisor_socket_path(conf_paths: tuple[Path, ...] = SUPERVISORD_CONF_PATHS) -> Path:
    """Parse `[unix_http_server] file=...` out of supervisord.conf (a tiny,
    line-oriented INI - the stdlib `configparser` chokes on supervisord's
    `%(ENV_X)s` interpolation syntax elsewhere in the file, so this is a
    deliberately narrow hand-rolled scan, not a general INI parser).
    """
    for conf_path in conf_paths:
        try:
            text = conf_path.read_text()
        except OSError:
            continue
        section = None
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith((";", "#")):
                continue
            if line.startswith("[") and line.endswith("]"):
                section = line.lower()
                continue
            if section == "[unix_http_server]" and line.startswith("file"):
                _, _, value = line.partition("=")
                value = value.strip()
                if value:
                    return Path(value)
    return DEFAULT_SUPERVISOR_SOCKET


class _UnixStreamHTTPConnection(http.client.HTTPConnection):
    """`http.client.HTTPConnection` that connects over an AF_UNIX socket
    instead of TCP - the only piece stdlib is missing for XML-RPC-over-a-
    unix-socket (supervisor itself ships the equivalent as
    `supervisor.xmlrpc.UnixStreamHTTPConnection`; reimplemented here rather
    than depending on the `supervisor` package being importable from the
    dashboard's own interpreter, which is not guaranteed).
    """

    def __init__(self, socket_path: str, timeout: float = 5.0) -> None:
        super().__init__("localhost", timeout=timeout)
        self._socket_path = socket_path

    def connect(self) -> None:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect(self._socket_path)
        self.sock = sock


class _UnixStreamTransport(xmlrpc.client.Transport):
    def __init__(self, socket_path: str, timeout: float = 5.0) -> None:
        super().__init__()
        self._socket_path = socket_path
        self._timeout = timeout

    def make_connection(self, host: str) -> _UnixStreamHTTPConnection:  # noqa: ARG002 - host unused (fixed socket)
        return _UnixStreamHTTPConnection(self._socket_path, timeout=self._timeout)


def make_supervisor_rpc_proxy(socket_path: Path | None = None, timeout_s: float = 5.0) -> Any:
    """A `xmlrpc.client.ServerProxy` talking to supervisord's RPC interface
    over its unix socket. Kept as its own function (not inlined into
    `collect_services_rpc`) so tests can monkeypatch/inject a fake here
    instead of needing a real supervisord socket.
    """
    socket_path = socket_path or read_supervisor_socket_path()
    transport = _UnixStreamTransport(str(socket_path), timeout=timeout_s)
    return xmlrpc.client.ServerProxy("http://localhost/RPC2", transport=transport)


def _services_from_rpc_infos(infos: list[dict]) -> list[dict]:
    out = []
    for info in infos:
        name = info.get("name") or "?"
        group = info.get("group") or name
        full_name = name if group in (None, name) else f"{group}:{name}"
        desc = str(info.get("description") or "")
        uptime_m = _SUP_UPTIME_RE.search(desc)
        pid = info.get("pid") or None
        out.append(
            {
                "name": full_name,
                "state": str(info.get("statename") or "?"),
                "pid": int(pid) if pid else None,
                "uptime": uptime_m.group(1) if uptime_m else None,
                "detail": desc,
            }
        )
    return out


def collect_services_rpc(
    socket_path: Path | None = None, timeout_s: float = 5.0, proxy: Any = None
) -> list[dict]:
    """`supervisor.getAllProcessInfo()` over the RPC socket - no fork, one
    unix socket round trip. `proxy` is injectable for tests (anything with
    a `.supervisor.getAllProcessInfo()` attribute); real callers leave it
    None and get `make_supervisor_rpc_proxy()`.
    """
    proxy = proxy if proxy is not None else make_supervisor_rpc_proxy(socket_path, timeout_s)
    infos = proxy.supervisor.getAllProcessInfo()
    return _services_from_rpc_infos(list(infos))


def parse_supervisorctl_status(text: str) -> list[dict]:
    """Parse `supervisorctl status` lines: `name STATE [pid N, uptime
    H:MM:SS]` (RUNNING), `name STOPPED Not started`, `name FATAL <reason>`,
    `name EXITED <timestamp>`, `name BACKOFF <reason>`, etc. Program names
    with a `group:name` form (event listeners) are kept as one token.
    """
    out = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        m = _SUP_LINE_RE.match(line)
        if not m:
            continue
        rest = m.group("rest").strip()
        pid_m = _SUP_PID_RE.search(rest)
        uptime_m = _SUP_UPTIME_RE.search(rest)
        out.append(
            {
                "name": m.group("name"),
                "state": m.group("state"),
                "pid": int(pid_m.group(1)) if pid_m else None,
                "uptime": uptime_m.group(1) if uptime_m else None,
                "detail": rest,
            }
        )
    return out


def _collect_services_subprocess(timeout_s: float, run: Callable[..., Any]) -> list[dict]:
    try:
        proc = run(["supervisorctl", "status"], capture_output=True, text=True, timeout=timeout_s, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError("supervisorctl not found") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"supervisorctl timed out after {timeout_s:.0f}s") from exc
    # supervisorctl's own exit code is 3 when any program isn't RUNNING -
    # that is data, not a tool failure; only treat other codes as a fault.
    if proc.returncode not in (0, 3):
        raise RuntimeError(f"supervisorctl exited {proc.returncode}: {(proc.stderr or '').strip()[:200]}")
    return parse_supervisorctl_status(proc.stdout)


# Exceptions a broken/unreachable RPC socket can raise: missing file, refused
# connection, permission, a timeout mid-call, or supervisord answering with a
# malformed/faulted XML-RPC response. All of them mean "fall back", never
# "crash the collector".
_RPC_FAILURE_TYPES = (OSError, EOFError, xmlrpc.client.Error, http.client.HTTPException)


def collect_services(
    timeout_s: float = 5.0,
    run: Callable[..., Any] = subprocess.run,
    rpc_proxy: Any = None,
    socket_path: Path | None = None,
    prefer_rpc: bool = True,
) -> dict:
    """Service states from supervisord. Tries the RPC socket first (one
    unix round trip, no fork - see the module section header above for
    why); falls back to `supervisorctl status` (a subprocess) only if the
    RPC path fails, and says which path it used in `"source"` so `Hub`
    can poll the (expensive) fallback at a slower cadence than the
    (cheap) RPC path. `rpc_proxy` is injectable for tests; `prefer_rpc`
    lets a caller (or a test) force the subprocess path outright.
    """
    source = "rpc"
    services: list[dict] | None = None
    if prefer_rpc:
        try:
            services = collect_services_rpc(socket_path=socket_path, timeout_s=timeout_s, proxy=rpc_proxy)
        except _RPC_FAILURE_TYPES:
            services = None
    if services is None:
        services = _collect_services_subprocess(timeout_s, run)
        source = "subprocess"
    for svc in services:
        svc["unexpected"] = svc["state"] != "RUNNING" and svc["name"] not in podenv.ONE_SHOT_OR_OPTIONAL
    return {"services": services, "source": source}


# =============================================================================
# /proc/net/tcp[6] and SSH_CONNECTION
# =============================================================================

TCP_STATE_ESTABLISHED = "01"
TCP_STATE_LISTEN = "0A"


def _parse_hex_ip_port(token: str, v6: bool) -> tuple[str, int] | None:
    ip_hex, _, port_hex = token.partition(":")
    if not port_hex:
        return None
    try:
        port = int(port_hex, 16)
        raw = bytes.fromhex(ip_hex)
    except ValueError:
        return None
    try:
        if v6:
            if len(raw) != 16:
                return None
            groups = [raw[i : i + 4][::-1] for i in range(0, 16, 4)]
            ip = socket.inet_ntop(socket.AF_INET6, b"".join(groups))
        else:
            if len(raw) != 4:
                return None
            ip = socket.inet_ntop(socket.AF_INET, raw[::-1])
    except (OSError, ValueError):
        return None
    return ip, port


def parse_proc_net_tcp(text: str, v6: bool = False) -> list[dict]:
    """Parse `/proc/net/tcp` or `/proc/net/tcp6`: hex-encoded little-endian
    addresses/ports, one connection per data row (the header row is
    skipped). `state` is the raw two-hex-digit code (`01` = ESTABLISHED,
    `0A` = LISTEN - see `TCP_STATE_*`).
    """
    out = []
    lines = text.splitlines()
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        local = _parse_hex_ip_port(parts[1], v6)
        remote = _parse_hex_ip_port(parts[2], v6)
        if local is None or remote is None:
            continue
        out.append(
            {
                "local_ip": local[0],
                "local_port": local[1],
                "remote_ip": remote[0],
                "remote_port": remote[1],
                "state": parts[3].upper(),
            }
        )
    return out


def parse_ssh_connection(value: str | None) -> dict | None:
    """Parse the `SSH_CONNECTION` env var: `"client_ip client_port
    server_ip server_port"`. Returns None for an unset/malformed value (a
    RunPod Web Terminal shell has no SSH_CONNECTION at all).
    """
    if not value:
        return None
    parts = value.split()
    if len(parts) != 4:
        return None
    client_ip, client_port, server_ip, server_port = parts
    try:
        return {
            "client_ip": client_ip,
            "client_port": int(client_port),
            "server_ip": server_ip,
            "server_port": int(server_port),
        }
    except ValueError:
        return None


def _proc_stat_info(pid: int, proc_root: Path) -> dict | None:
    """`comm`/`ppid`/`starttime_ticks` from `/proc/<pid>/stat`, parsed from
    the LAST `)` so a `comm` containing spaces or parens (rare, but valid)
    doesn't desynchronize the field count.
    """
    try:
        raw = (proc_root / str(pid) / "stat").read_text()
    except OSError:
        return None
    lparen = raw.find("(")
    rparen = raw.rfind(")")
    if lparen == -1 or rparen == -1 or rparen < lparen:
        return None
    comm = raw[lparen + 1 : rparen]
    rest = raw[rparen + 2 :].split()
    if len(rest) < 20:
        return None
    try:
        ppid = int(rest[1])
        starttime_ticks = int(rest[19])
    except (ValueError, IndexError):
        return None
    return {"comm": comm, "ppid": ppid, "starttime_ticks": starttime_ticks}


def _system_boot_time(proc_root: Path) -> float | None:
    try:
        text = (proc_root / "stat").read_text()
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("btime "):
            try:
                return float(line.split()[1])
            except (ValueError, IndexError):
                return None
    return None


def _clock_ticks_per_sec() -> int:
    try:
        return os.sysconf("SC_CLK_TCK")
    except (ValueError, AttributeError, OSError):
        return 100


def find_sshd_ancestor(
    pid: int | None = None, max_hops: int = 32, proc_root: Path = Path("/proc")
) -> dict | None:
    """Walk this process's parents looking for the per-connection `sshd`
    that owns this login shell's session (not the master listening on the
    socket). Returns `{"pid", "comm", "ppid", "starttime_ticks"}`, or None
    within `max_hops` - which is the normal case for a RunPod Web Terminal
    shell (no sshd anywhere in its ancestry).
    """
    pid = os.getpid() if pid is None else pid
    seen: set[int] = set()
    for _ in range(max_hops):
        if pid in seen or pid <= 1:
            return None
        seen.add(pid)
        info = _proc_stat_info(pid, proc_root)
        if info is None:
            return None
        if info["comm"] == "sshd":
            return {"pid": pid, **info}
        pid = info["ppid"]
    return None


def session_start_and_duration(
    now: float | None = None, proc_root: Path = Path("/proc"), pid: int | None = None
) -> dict:
    now = time.time() if now is None else now
    sshd = find_sshd_ancestor(pid=pid, proc_root=proc_root)
    if sshd is None:
        return {"in_ssh": False}
    btime = _system_boot_time(proc_root)
    if btime is None:
        return {"in_ssh": True, "error": "could not read /proc/stat btime"}
    hz = _clock_ticks_per_sec()
    start_epoch = btime + sshd["starttime_ticks"] / hz
    return {"in_ssh": True, "start_epoch": start_epoch, "duration_s": max(0.0, now - start_epoch)}


def collect_ssh_session(
    now: float | None = None,
    environ: dict[str, str] | None = None,
    tcp_path: Path = Path("/proc/net/tcp"),
    tcp6_path: Path = Path("/proc/net/tcp6"),
    proc_root: Path = Path("/proc"),
) -> dict:
    """The "telemetry for the SSH connection" itself: who's connected,
    since when, how many connections share the port, whether sshd is even
    listening, and whether this session is inside tmux (so it survives the
    connection dying).
    """
    now = time.time() if now is None else now
    env = environ if environ is not None else os.environ
    conn = parse_ssh_connection(env.get("SSH_CONNECTION"))
    port = conn["server_port"] if conn else 22
    session = session_start_and_duration(now=now, proc_root=proc_root)

    established_ips: list[str] = []
    listening = False
    for path, v6 in ((tcp_path, False), (tcp6_path, True)):
        try:
            text = path.read_text()
        except OSError:
            continue
        for row in parse_proc_net_tcp(text, v6=v6):
            if row["local_port"] != port:
                continue
            if row["state"] == TCP_STATE_LISTEN:
                listening = True
            elif row["state"] == TCP_STATE_ESTABLISHED:
                established_ips.append(row["remote_ip"])

    return {
        "connection": conn,
        "port": port,
        "session": session,
        "established_count": len(established_ips),
        "established_client_ips": sorted(set(established_ips)),
        "listening": listening,
        "in_tmux": bool(env.get("TMUX")),
    }


# =============================================================================
# Terminal round-trip time (DSR probe)
# =============================================================================

_DSR_RE = re.compile(r"\x1b\[(\d+);(\d+)R")


def find_dsr_reply(buf: str) -> tuple[int, int, int, int] | None:
    """Find a cursor-position report (`ESC[row;colR`, the terminal's reply
    to a Device Status Report request `ESC[6n`) anywhere in `buf`. Returns
    `(row, col, start, end)` - `start`/`end` (end exclusive) so a caller
    reading a raw input stream can splice the reply OUT before treating the
    rest of `buf` as keystrokes; the reply itself must never reach a
    keypress handler.
    """
    m = _DSR_RE.search(buf)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), m.start(), m.end()


def probe_terminal_rtt(timeout_s: float = 1.0) -> float | None:
    """Best-effort terminal round-trip probe OUTSIDE curses (used by
    `--once`/`collect_snapshot` on a real TTY). Writes `ESC[6n`, measures
    how long stdin takes to produce a DSR reply. Returns milliseconds, or
    None if stdin/stdout isn't a TTY, termios is unavailable, or nothing
    came back in time - never raises. `dash.py`'s live loop does NOT call
    this: it reads the reply through its own curses input loop instead, so
    the same byte is never both an RTT sample and a keystroke.
    """
    try:
        import select
        import termios
        import tty
    except ImportError:
        return None
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return None
    fd = sys.stdin.fileno()
    try:
        old = termios.tcgetattr(fd)
    except termios.error:
        return None
    try:
        tty.setcbreak(fd)
        start = time.monotonic()
        sys.stdout.write("\x1b[6n")
        sys.stdout.flush()
        buf = ""
        deadline = start + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            ready, _, _ = select.select([fd], [], [], remaining)
            if not ready:
                return None
            chunk = os.read(fd, 64).decode("ascii", "ignore")
            if not chunk:
                return None
            buf += chunk
            if find_dsr_reply(buf):
                return (time.monotonic() - start) * 1000.0
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        except termios.error:
            pass


# =============================================================================
# vLLM: /metrics (Prometheus) with a vllm.log fallback
# =============================================================================

# Exposition names verified against the real image (not trusted from memory
# - the brief's own list turned out to be missing the automatic `_total`
# suffix prometheus_client adds to every Counter):
#
#   docker run --rm --entrypoint bash c1295894dd58 -c \
#     'grep -n "vllm:" /opt/vllm-venv/lib/python3*/site-packages/vllm/v1/metrics/loggers.py'
#
# confirms these `name=` values in vllm/v1/metrics/loggers.py (vLLM 0.24,
# v1 engine): gauges `vllm:num_requests_running`, `vllm:num_requests_waiting`,
# `vllm:kv_cache_usage_perc` (exposed AS GIVEN - Gauge does not suffix);
# counters `vllm:prompt_tokens`, `vllm:generation_tokens`,
# `vllm:prefix_cache_queries`, `vllm:prefix_cache_hits`, `vllm:num_preemptions`.
# Then, separately:
#     grep -n "_total\|class Counter" \
#       /opt/vllm-venv/lib/python3*/site-packages/prometheus_client/metrics.py
# confirms prometheus_client's Counter unconditionally appends `_total` to
# the sample it emits (`self._name + '_total'`) unless the given name
# already ends in `_total` (none of vLLM's do) - so every counter above is
# actually exposed with a `_total` suffix, e.g. `vllm:prompt_tokens_total`.
# Histograms (`vllm:time_to_first_token_seconds`,
# `vllm:inter_token_latency_seconds`) expose `_bucket{le=...}`, `_sum`,
# `_count` per prometheus_client's standard Histogram behaviour (not
# separately grepped - that suffixing is prometheus_client's documented,
# unconditional contract, the same mechanism just verified for Counter).
VLLM_GAUGE_RUNNING = "vllm:num_requests_running"
VLLM_GAUGE_WAITING = "vllm:num_requests_waiting"
VLLM_GAUGE_KV_CACHE_PCT = "vllm:kv_cache_usage_perc"
VLLM_COUNTER_PROMPT_TOKENS = "vllm:prompt_tokens_total"
VLLM_COUNTER_GENERATION_TOKENS = "vllm:generation_tokens_total"
VLLM_COUNTER_PREFIX_QUERIES = "vllm:prefix_cache_queries_total"
VLLM_COUNTER_PREFIX_HITS = "vllm:prefix_cache_hits_total"
VLLM_COUNTER_PREEMPTIONS = "vllm:num_preemptions_total"
VLLM_HIST_TTFT_BUCKET = "vllm:time_to_first_token_seconds_bucket"
VLLM_HIST_ITL_BUCKET = "vllm:inter_token_latency_seconds_bucket"


def _sum_series(samples: dict[str, list[Sample]], name: str) -> float | None:
    rows = samples.get(name)
    if not rows:
        return None
    return sum(r.value for r in rows)


def _avg_series(samples: dict[str, list[Sample]], name: str) -> float | None:
    rows = samples.get(name)
    if not rows:
        return None
    return sum(r.value for r in rows) / len(rows)


def _bucket_series(samples: dict[str, list[Sample]], name: str) -> dict[float, float] | None:
    rows = samples.get(name)
    if not rows:
        return None
    out: dict[float, float] = {}
    for r in rows:
        le_raw = r.labels.get("le")
        if le_raw is None:
            continue
        le = _parse_float(le_raw)
        if le is None:
            continue
        out[le] = out.get(le, 0.0) + r.value
    return out or None


def extract_vllm_metrics(samples: dict[str, list[Sample]]) -> dict:
    """Pull the panel-relevant numbers out of one `/metrics` scrape.
    Gauges/counters/buckets only - a single scrape can't produce a rate or
    a quantile; `vllm_rates()`/`histogram_quantile_from_buckets()` need two
    of these, `rate_gap_s` apart.
    """
    return {
        "running": _sum_series(samples, VLLM_GAUGE_RUNNING),
        "waiting": _sum_series(samples, VLLM_GAUGE_WAITING),
        "kv_cache_usage_pct": (
            v * 100.0 if (v := _avg_series(samples, VLLM_GAUGE_KV_CACHE_PCT)) is not None else None
        ),
        VLLM_COUNTER_PROMPT_TOKENS: _sum_series(samples, VLLM_COUNTER_PROMPT_TOKENS),
        VLLM_COUNTER_GENERATION_TOKENS: _sum_series(samples, VLLM_COUNTER_GENERATION_TOKENS),
        VLLM_COUNTER_PREFIX_QUERIES: _sum_series(samples, VLLM_COUNTER_PREFIX_QUERIES),
        VLLM_COUNTER_PREFIX_HITS: _sum_series(samples, VLLM_COUNTER_PREFIX_HITS),
        VLLM_COUNTER_PREEMPTIONS: _sum_series(samples, VLLM_COUNTER_PREEMPTIONS),
        "ttft_buckets": _bucket_series(samples, VLLM_HIST_TTFT_BUCKET),
        "itl_buckets": _bucket_series(samples, VLLM_HIST_ITL_BUCKET),
    }


def vllm_rates(prev: dict, prev_ts: float, cur: dict, cur_ts: float) -> dict:
    def rate(key: str) -> float | None:
        pv, cv = prev.get(key), cur.get(key)
        if pv is None or cv is None:
            return None
        return counter_rate(pv, prev_ts, cv, cur_ts)

    prefix_hit_rate = None
    pq0, pq1 = prev.get(VLLM_COUNTER_PREFIX_QUERIES), cur.get(VLLM_COUNTER_PREFIX_QUERIES)
    ph0, ph1 = prev.get(VLLM_COUNTER_PREFIX_HITS), cur.get(VLLM_COUNTER_PREFIX_HITS)
    if None not in (pq0, pq1, ph0, ph1):
        dq = pq1 if pq1 < pq0 else (pq1 - pq0)
        dh = ph1 if ph1 < ph0 else (ph1 - ph0)
        if dq > 0:
            prefix_hit_rate = max(0.0, min(1.0, dh / dq))

    preempt_delta = None
    pe0, pe1 = prev.get(VLLM_COUNTER_PREEMPTIONS), cur.get(VLLM_COUNTER_PREEMPTIONS)
    if pe0 is not None and pe1 is not None:
        preempt_delta = pe1 if pe1 < pe0 else (pe1 - pe0)

    return {
        "prompt_tokens_per_s": rate(VLLM_COUNTER_PROMPT_TOKENS),
        "generation_tokens_per_s": rate(VLLM_COUNTER_GENERATION_TOKENS),
        "prefix_cache_hit_rate": prefix_hit_rate,
        "preemptions_delta": preempt_delta,
    }


def vllm_latency_quantiles(prev_buckets: dict | None, cur_buckets: dict | None) -> dict:
    return {
        "p50_s": histogram_quantile_from_buckets(prev_buckets, cur_buckets, 0.50),
        "p95_s": histogram_quantile_from_buckets(prev_buckets, cur_buckets, 0.95),
    }


# The tail of a vllm.log INFO line, e.g.:
#   Engine 000: Avg prompt throughput: 719.7 tokens/s, Avg generation
#   throughput: 6.0 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache
#   usage: 0.6%, Prefix cache hit rate: 11.5%
# "Preemptions: N" is inserted before "GPU KV cache usage" only when
# num_preemptions > 0 (loggers.py's own conditional append) - verified in
# the same grep above.
_VLLM_LOG_LINE_RE = re.compile(
    r"Engine (?P<engine>\d+): "
    r"Avg prompt throughput: (?P<prompt_tps>[\d.]+) tokens/s, "
    r"Avg generation throughput: (?P<gen_tps>[\d.]+) tokens/s, "
    r"Running: (?P<running>\d+) reqs, "
    r"Waiting: (?P<waiting>\d+) reqs, "
    r"(?:Preemptions: (?P<preemptions>\d+), )?"
    r"GPU KV cache usage: (?P<kv>[\d.]+)%, "
    r"Prefix cache hit rate: (?P<prefix_hit>[\d.]+)%"
)

# The line's own timestamp, e.g. `(APIServer pid=115) INFO 09-22 20:27:19
# [loggers.py:259] Engine 000: Avg prompt throughput: ...` - verified against
# both c1295894dd58 and prod's f3fafa3e1949 (same vLLM 0.19.0 build in both).
# No year field, so `parse_vllm_log_timestamp` infers it from `now`.
_VLLM_LOG_TS_RE = re.compile(r"\b(?P<mon>\d{2})-(?P<day>\d{2}) (?P<hh>\d{2}):(?P<mm>\d{2}):(?P<ss>\d{2})\b")

# A vllm.log line older than this is never shown as if it were live - the
# fallback exists for when /metrics is unreachable, and a stale number
# presented as current is worse than an honest "no recent line".
VLLM_LOG_STALE_S = 30.0


def parse_vllm_log_timestamp(line: str, now: float | None = None) -> float | None:
    """Parse the `MM-DD HH:MM:SS` timestamp vLLM's own logger prefixes every
    line with, into an epoch float. There is no year field, so the year is
    inferred from `now`: if treating the parsed month/day/time as THIS year
    would put it more than a day in the future (the log rolled over a
    year boundary since it was written), the previous year is used instead.
    """
    m = _VLLM_LOG_TS_RE.search(line)
    if not m:
        return None
    now = time.time() if now is None else now
    import datetime as _dt

    now_dt = _dt.datetime.fromtimestamp(now, _dt.timezone.utc)
    for year in (now_dt.year, now_dt.year - 1):
        try:
            dt = _dt.datetime(
                year,
                int(m.group("mon")),
                int(m.group("day")),
                int(m.group("hh")),
                int(m.group("mm")),
                int(m.group("ss")),
                tzinfo=_dt.timezone.utc,
            )
        except ValueError:
            continue
        epoch = dt.timestamp()
        if epoch <= now + 86400:
            return epoch
    return None


def parse_vllm_log_line(line: str, now: float | None = None) -> dict | None:
    m = _VLLM_LOG_LINE_RE.search(line)
    if not m:
        return None
    d = m.groupdict()
    line_epoch = parse_vllm_log_timestamp(line, now=now)
    return {
        "engine": int(d["engine"]),
        "prompt_tokens_per_s": float(d["prompt_tps"]),
        "generation_tokens_per_s": float(d["gen_tps"]),
        "running": int(d["running"]),
        "waiting": int(d["waiting"]),
        "preemptions": int(d["preemptions"]) if d["preemptions"] else None,
        "kv_cache_usage_pct": float(d["kv"]),
        "prefix_cache_hit_rate_pct": float(d["prefix_hit"]),
        "line_epoch": line_epoch,
        "line_age_s": (now if now is not None else time.time()) - line_epoch if line_epoch is not None else None,
        "stale": (
            ((now if now is not None else time.time()) - line_epoch) > VLLM_LOG_STALE_S
            if line_epoch is not None
            else True  # no timestamp at all -> never trust it as live
        ),
    }


def newest_vllm_log_line(log_dir: Path | None = None, max_bytes: int = 65536, now: float | None = None) -> dict | None:
    log_dir = log_dir or podenv.LOG_DIR
    path = log_dir / "vllm.log"
    try:
        text = tail_bytes(path, max_bytes)
    except OSError:
        return None
    for line in reversed(text.splitlines()):
        parsed = parse_vllm_log_line(line, now=now)
        if parsed:
            return parsed
    return None


def _default_http_get(url: str, timeout_s: float) -> bytes:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310 - localhost only
        return resp.read()


def collect_vllm(
    url: str | None = None,
    timeout_s: float = 3.0,
    fetch: Callable[[str, float], bytes] = _default_http_get,
    log_dir: Path | None = None,
) -> dict:
    """One instantaneous vLLM sample: `/metrics` if reachable (`source":
    "metrics"`), else the newest usable line in `vllm.log` (`"source":
    "log"`, panel labelled "from log" by the UI). Rates/quantiles need two
    of these - see `vllm_rates`/`vllm_latency_quantiles`.
    """
    vllm_url = url or podenv.VLLM_URL
    try:
        body = fetch(f"{vllm_url}/metrics", timeout_s)
        text = body.decode("utf-8", "replace") if isinstance(body, (bytes, bytearray)) else body
        samples = parse_prometheus_text(text)
        return {"source": "metrics", "metrics": extract_vllm_metrics(samples)}
    except Exception as exc:  # noqa: BLE001 - network/parse failure, falls back
        log_line = newest_vllm_log_line(log_dir)
        if log_line is None:
            raise RuntimeError(
                f"/metrics unreachable ({type(exc).__name__}: {exc}) and vllm.log has no usable line"
            ) from exc
        return {"source": "log", "log": log_line, "metrics_error": f"{type(exc).__name__}: {exc}"}


# =============================================================================
# Host: load, memory, CPU%, root free space, process count
# =============================================================================


def _read_loadavg(path: Path = Path("/proc/loadavg")) -> list[float] | None:
    try:
        parts = path.read_text().split()
    except OSError:
        return None
    try:
        return [float(parts[0]), float(parts[1]), float(parts[2])]
    except (ValueError, IndexError):
        return None


def _read_meminfo(path: Path = Path("/proc/meminfo")) -> dict | None:
    try:
        text = path.read_text()
    except OSError:
        return None
    out: dict[str, int] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, _, rest = line.partition(":")
        tokens = rest.strip().split()
        if not tokens:
            continue
        try:
            out[key] = int(tokens[0]) * 1024  # kB -> bytes
        except ValueError:
            continue
    return out or None


_CPU_FIELDS = ("user", "nice", "system", "idle", "iowait", "irq", "softirq", "steal", "guest", "guest_nice")


def _read_proc_stat_cpu(path: Path = Path("/proc/stat")) -> dict | None:
    try:
        text = path.read_text()
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("cpu "):
            parts = line.split()[1:]
            nums = []
            for p in parts:
                try:
                    nums.append(int(p))
                except ValueError:
                    break
            if not nums:
                return None
            return dict(zip(_CPU_FIELDS, nums))
    return None


def cpu_percent_from_stat(a: dict | None, b: dict | None) -> float | None:
    """CPU busy % between two `/proc/stat` "cpu " samples (deltas, not
    lifetime averages)."""
    if not a or not b:
        return None
    a_idle = a.get("idle", 0) + a.get("iowait", 0)
    b_idle = b.get("idle", 0) + b.get("iowait", 0)
    a_total = sum(a.values())
    b_total = sum(b.values())
    dt = b_total - a_total
    if dt <= 0:
        return None
    busy = dt - (b_idle - a_idle)
    return max(0.0, min(100.0, busy / dt * 100.0))


def collect_host(
    loadavg_path: Path = Path("/proc/loadavg"),
    meminfo_path: Path = Path("/proc/meminfo"),
    stat_path: Path = Path("/proc/stat"),
    root_path: str = "/",
    proc_dir: Path = Path("/proc"),
) -> dict:
    statv = os.statvfs(root_path)
    try:
        proc_count = sum(1 for n in os.listdir(proc_dir) if n.isdigit())
    except OSError:
        proc_count = None
    return {
        "loadavg": _read_loadavg(loadavg_path),
        "meminfo": _read_meminfo(meminfo_path),
        "cpu_times": _read_proc_stat_cpu(stat_path),
        "root_free_bytes": statv.f_frsize * statv.f_bavail,
        "root_total_bytes": statv.f_frsize * statv.f_blocks,
        "proc_count": proc_count,
    }


# =============================================================================
# /data usage (du, never df - df reports the whole MooseFS cluster)
# =============================================================================


def collect_data_usage(
    timeout_s: float = 60.0,
    data_dir: Path | None = None,
    run: Callable[..., Any] = subprocess.run,
) -> dict:
    data_dir = data_dir or podenv.DATA
    try:
        names = sorted(p.name for p in data_dir.iterdir())
    except OSError as exc:
        raise RuntimeError(f"cannot list {data_dir}: {exc}") from exc

    deadline = time.monotonic() + timeout_s
    entries: list[dict] = []
    errors: dict[str, str] = {}
    total_kb = 0
    for name in names:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            errors[name] = "skipped: over the du time budget"
            continue
        try:
            proc = run(
                ["du", "-sk", str(data_dir / name)],
                capture_output=True,
                text=True,
                timeout=max(0.1, remaining),
                check=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("du not found") from exc
        except subprocess.TimeoutExpired:
            errors[name] = "du timed out"
            continue
        if proc.returncode != 0:
            errors[name] = (proc.stderr or "").strip()[:200]
            continue
        try:
            kb = int(proc.stdout.split()[0])
        except (ValueError, IndexError):
            errors[name] = "unparseable du output"
            continue
        entries.append({"name": name, "kb": kb})
        total_kb += kb

    pct_of_volume = None
    vol_gb_raw = os.environ.get("ZL_VOLUME_GB", "").strip()
    if vol_gb_raw:
        try:
            vol_gb = float(vol_gb_raw)
            if vol_gb > 0:
                pct_of_volume = (total_kb / 1024 / 1024) / vol_gb * 100.0
        except ValueError:
            pass

    return {
        "entries": entries,
        "total_kb": total_kb,
        "errors": errors,
        "pct_of_volume": pct_of_volume,
        "note": "df /data would report the whole shared MooseFS cluster, not a quota - du only",
    }


# =============================================================================
# Backups
# =============================================================================


# compactor/backup.py: `_ARCHIVE_PREFIX = "zions-backup-"`, `_ARCHIVE_SUFFIX =
# ".tar.gz"`, stamp `time.strftime("%Y%m%d-%H%M%S", time.gmtime())`. A
# `.tar.gz.partial` (the in-progress atomic-publish temp name,
# backup.py:1276) or a `zions-backup-<stamp>-<mkdtemp suffix>` STAGING
# directory (backup.py:1607) must never count as "the newest backup" - a
# dashboard that shows a .partial's mtime as done work would tell the
# operator a backup completed when it hasn't (or hasn't yet been verified).
# Only the exact final archive name counts.
_BACKUP_ARCHIVE_RE = re.compile(r"^zions-backup-\d{8}-\d{6}\.tar\.gz$")


def collect_backups(backups_dir: Path | None = None, timeout_s: float = 5.0) -> dict:
    backups_dir = backups_dir or podenv.BACKUPS

    def _scan() -> list[tuple[str, float, int]]:
        out = []
        for p in backups_dir.iterdir():
            if not p.is_file() or not _BACKUP_ARCHIVE_RE.match(p.name):
                continue
            try:
                st = p.stat()
            except OSError:
                continue
            out.append((p.name, st.st_mtime, st.st_size))
        return out

    entries, err = _call_with_timeout(_scan, timeout_s)
    if err is not None:
        raise RuntimeError(str(err)) from err
    if not entries:
        return {"latest": None, "latest_mtime": None, "count": 0, "total_bytes": 0}
    entries.sort(key=lambda t: t[1])
    latest = entries[-1]
    return {
        "latest": latest[0],
        "latest_mtime": latest[1],
        "count": len(entries),
        "total_bytes": sum(t[2] for t in entries),
    }


# =============================================================================
# Log signal (privacy-filtered tail of the error/warning logs)
# =============================================================================

LOG_SIGNAL_FILES = (
    "compactor.log",
    "compactor-error.log",
    "vllm.log",
    "vllm-error.log",
    "openwebui.log",
    "openwebui-error.log",
    "supervisord.log",
)

_LOG_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?")
_LOG_TRACEBACK_RE = re.compile(r"Traceback \(most recent call last\)")

_LEVEL_WORDS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
_SIGNAL_LEVELS = ("WARNING", "ERROR", "CRITICAL")

# Every level check below is ANCHORED TO THE LEVEL FIELD of that file's own
# format, never a substring search over the whole line (a substring search
# would match a WARNING-shaped word inside a piece of DATA too, and more
# importantly a naive scan can't tell "this line HAS no level field" from
# "the level field says INFO" - the difference between showing an EngineCore
# progress line with no level at all and correctly leaving it out).
#
# Formats verified against real bundles under
# /home/drew/pod-exports/*/bundle*/zla-bundle-*/logs/zla/ (line SHAPES only):
#   compactor.log      "TS,mmm <logger> LEVEL message"                (python logging)
#   compactor-error.log "LEVEL:     message"                          (uvicorn)
#   vllm.log/-error.log "(Component pid=N) LEVEL MM-DD HH:MM:SS [...] message"
#                        or "(Component pid=N) LEVEL:     message"    (uvicorn sub-lines)
#                        or "(Component pid=N) <no level at all>"     (EngineCore raw prints)
#   openwebui*.log      "TS.mmm | LEVEL    | logger:func:line - message"  (loguru)
#   supervisord.log     "TS,mmm LEVEL message" with LEVEL truncated to 4
#                        chars (CRIT/WARN/ERRO/INFO/DEBU)

_LEVEL_COMPACTOR_APP_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}[.,]\d+\s+\S+\s+(DEBUG|INFO|WARNING|ERROR|CRITICAL)\b"
)
_LEVEL_COLON_RE = re.compile(r"^(?:\([^)]*\)\s+)?(DEBUG|INFO|WARNING|ERROR|CRITICAL):")
_LEVEL_VLLM_RE = re.compile(r"^\([^)]*\)\s+(DEBUG|INFO|WARNING|ERROR|CRITICAL)\b")
_LEVEL_LOGURU_RE = re.compile(
    r"^\S+ \S+ \|\s*(DEBUG|INFO|WARNING|ERROR|CRITICAL|TRACE|SUCCESS)\s*\|"
)
_LEVEL_SUPERVISORD_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2},\d+\s+(CRIT|ERRO|WARN|INFO|DEBU)\b"
)
_SUPERVISORD_LEVEL_MAP = {"CRIT": "CRITICAL", "ERRO": "ERROR", "WARN": "WARNING", "INFO": "INFO", "DEBU": "DEBUG"}


def _line_level_field(line: str, fname: str) -> str | None:
    """The LEVEL this line's own format says it is (one of `_LEVEL_WORDS`,
    or a loguru-only word like TRACE/SUCCESS), or None if this line's
    format carries no level field at all (an EngineCore raw print, a
    traceback continuation frame, ...). `fname` picks which format to
    anchor to; an unrecognised `fname` tries every anchor (never a bare
    substring search).
    """
    if fname == "compactor.log":
        m = _LEVEL_COMPACTOR_APP_RE.match(line)
        return m.group(1) if m else None
    if fname == "compactor-error.log":
        m = _LEVEL_COLON_RE.match(line)
        return m.group(1) if m else None
    if fname in ("vllm.log", "vllm-error.log"):
        m = _LEVEL_VLLM_RE.match(line) or _LEVEL_COLON_RE.match(line)
        return m.group(1) if m else None
    if fname in ("openwebui.log", "openwebui-error.log"):
        m = _LEVEL_LOGURU_RE.match(line)
        return m.group(1) if m else None
    if fname == "supervisord.log":
        m = _LEVEL_SUPERVISORD_RE.match(line)
        return _SUPERVISORD_LEVEL_MAP[m.group(1)] if m else None
    # Unknown source: try every anchor rather than falling back to a
    # substring scan.
    for fn in (_LEVEL_COMPACTOR_APP_RE, _LEVEL_COLON_RE, _LEVEL_VLLM_RE, _LEVEL_LOGURU_RE):
        m = fn.match(line)
        if m:
            return m.group(1)
    m = _LEVEL_SUPERVISORD_RE.match(line)
    return _SUPERVISORD_LEVEL_MAP[m.group(1)] if m else None


def is_signal_line(line: str, fname: str = "") -> bool:
    """WARNING/ERROR/CRITICAL by the line's OWN level field, or a
    traceback header (`Traceback (most recent call last):` - the one line
    of a traceback that is always safe to show and never itself a data
    value).
    """
    level = _line_level_field(line, fname)
    if level in _SIGNAL_LEVELS:
        return True
    return bool(_LOG_TRACEBACK_RE.search(line))


def _log_line_level(line: str, fname: str = "") -> str:
    level = _line_level_field(line, fname)
    if level == "CRITICAL" or _LOG_TRACEBACK_RE.search(line):
        return "crit"
    if level == "ERROR":
        return "crit"
    if level == "WARNING":
        return "warn"
    return "info"


def extract_log_timestamp(line: str) -> str | None:
    m = _LOG_TS_RE.match(line)
    return m.group(0) if m else None


# A shown line never runs past this many characters (before the "[file] TS "
# prefix dash.py adds) - long enough to be useful, short enough that one
# noisy line can't push everything else off the panel.
LOG_LINE_MAX_CHARS = 240


def collect_log_signal(log_dir: Path | None = None, max_lines: int = 8) -> dict:
    """The privacy-scrubbed tail of the ERROR/WARNING/CRITICAL/traceback
    lines across every log the dashboard watches. Every line that survives
    `is_signal_line`'s level-field check is ALSO run through
    `privacy.scrub_log_line` before it is kept - the one scrubber `zl
    bundle` uses too (see `privacy.py`'s module docstring for why a
    substring line-drop filter used to hide most of this panel).
    """
    from . import privacy

    log_dir = log_dir or podenv.LOG_DIR
    entries: list[dict] = []
    file_errors: dict[str, str] = {}
    for fname in LOG_SIGNAL_FILES:
        path = log_dir / fname
        try:
            text = tail_bytes(path)
        except OSError as exc:
            file_errors[fname] = f"{type(exc).__name__}: {exc}"
            continue
        for line in text.splitlines()[-4000:]:
            if not is_signal_line(line, fname):
                continue
            safe = privacy.scrub_log_line(line, fname)
            if safe is None:
                continue
            if len(safe) > LOG_LINE_MAX_CHARS:
                safe = safe[:LOG_LINE_MAX_CHARS] + "…"
            entries.append(
                {
                    "file": fname,
                    "timestamp": extract_log_timestamp(safe),
                    "level": _log_line_level(safe, fname),
                    "text": safe,
                }
            )
    entries = entries[-max_lines:]
    return {"entries": entries, "file_errors": file_errors}


# =============================================================================
# collect_snapshot - one synchronous pass (--once / --json / meta/telemetry.json)
# =============================================================================


def collect_snapshot(options: dict | None = None) -> dict:
    """One JSON-serialisable snapshot of everything the dashboard shows.
    Nothing here raises: a failing collector's section is `{"data": None,
    "error": "..."}`. `rate_gap_s` (default 2.0) controls how long the two
    samples that vLLM throughput and host CPU% need are spaced apart -
    tests pass a small value to stay fast.
    """
    options = dict(options or {})
    rate_gap_s = float(options.get("rate_gap_s", 2.0))
    started = time.time()

    def run(fn: Callable[[], Any], timeout_s: float) -> dict:
        value, err = _call_with_timeout(fn, timeout_s)
        return {"data": value, "error": (str(err) if err else None), "collected_at": time.time()}

    out: dict[str, dict] = {
        "health": run(collect_health, 5.0),
        "services": run(collect_services, 5.0),
        "gpu": run(collect_gpu, 3.0),
        "backups": run(collect_backups, 5.0),
        "ssh": run(collect_ssh_session, 3.0),
        "log": run(collect_log_signal, 5.0),
        "data_usage": run(collect_data_usage, 60.0),
    }

    host0, host0_err = _call_with_timeout(collect_host, 3.0)
    vllm0, vllm0_err = _call_with_timeout(collect_vllm, 3.0)
    t0 = time.time()
    time.sleep(max(0.0, rate_gap_s))
    host1, host1_err = _call_with_timeout(collect_host, 3.0)
    vllm1, vllm1_err = _call_with_timeout(collect_vllm, 3.0)
    t1 = time.time()

    if host1_err is not None:
        out["host"] = {"data": None, "error": str(host1_err), "collected_at": t1}
    else:
        data = dict(host1)
        data["cpu_percent"] = (
            cpu_percent_from_stat(host0.get("cpu_times"), host1.get("cpu_times"))
            if (host0_err is None and host0)
            else None
        )
        out["host"] = {"data": data, "error": None, "collected_at": t1}

    if vllm1_err is not None:
        out["vllm"] = {"data": None, "error": str(vllm1_err), "collected_at": t1}
    else:
        data = dict(vllm1)
        if vllm1.get("source") == "metrics" and vllm0_err is None and vllm0 and vllm0.get("source") == "metrics":
            m0, m1 = vllm0["metrics"], vllm1["metrics"]
            data["rates"] = vllm_rates(m0, t0, m1, t1)
            data["ttft_quantiles_s"] = vllm_latency_quantiles(m0.get("ttft_buckets"), m1.get("ttft_buckets"))
            data["itl_quantiles_s"] = vllm_latency_quantiles(m0.get("itl_buckets"), m1.get("itl_buckets"))
        out["vllm"] = {"data": data, "error": None, "collected_at": t1}

    # RTT: only meaningful on a real, interactive TTY, and only when asked for.
    if options.get("no_rtt"):
        out["rtt"] = {"data": None, "error": "disabled (--no-rtt)", "collected_at": t1}
    else:
        rtt_ms, rtt_err = _call_with_timeout(probe_terminal_rtt, 2.0)
        if rtt_err is not None or rtt_ms is None:
            out["rtt"] = {"data": None, "error": "not available outside an interactive terminal", "collected_at": t1}
        else:
            out["rtt"] = {"data": {"last_ms": rtt_ms}, "error": None, "collected_at": t1}

    out["_meta"] = {"collected_at": t1, "duration_s": t1 - started, "rate_gap_s": rate_gap_s}
    return out


# =============================================================================
# Hub - the live, background-thread orchestrator dash.py drives
# =============================================================================

CADENCE_S = {
    "health": 5.0,
    "services": 5.0,
    "gpu": 2.0,
    "vllm": 2.0,
    "host": 2.0,
    "data_usage": 300.0,
    "backups": 30.0,
    "ssh": 5.0,
    "log": 5.0,
}

# NOTE: there is deliberately no per-collector outer timeout wrapper here
# anymore (there used to be, spawning a fresh thread-with-join every tick -
# see `Hub._tick`'s docstring). Every collector that can genuinely block
# without its own bound (subprocess `timeout=`, `urlopen(timeout=...)`)
# already carries one at the point it does I/O; the one exception,
# `collect_data_usage`'s directory listing, is exactly the `/data`-stall
# case `Hub`'s single-thread-per-collector design exists to make safe
# WITHOUT a timeout (see `STALL_MULTIPLE` below).

HISTORY_LEN = 120


class PanelState:
    __slots__ = ("data", "error", "collected_at", "call_started_at")

    def __init__(self) -> None:
        self.data: Any = None
        self.error: str | None = None
        self.collected_at: float | None = None
        # Set while a call to this collector is in flight, cleared the
        # instant it returns - `None` the rest of the time. Lets the UI
        # show "stalled Ns" for a collector whose single long-lived worker
        # thread is still blocked deep inside a call (e.g. a `/data`
        # `iterdir()` stuck on a MooseFS stall) without EVER needing a
        # second thread to watch the first one.
        self.call_started_at: float | None = None


# A collector still running past this many times its own cadence is shown
# as "stalled", not just "stale" - stale means "the last successful read is
# old"; stalled means "a read has been in flight this whole time and never
# came back", which `_is_stale`-style age-of-last-good-data alone can't
# distinguish (the dashboard used to just keep showing old numbers with no
# sign anything was wrong).
STALL_MULTIPLE = 2.0


class Hub:
    """Owns exactly ONE long-lived background thread per collector, for the
    entire life of the dashboard - never a new thread per tick. Each
    thread loops on its own cadence from `CADENCE_S`, calling its
    collector directly (no per-call wrapper thread either): if a call
    blocks past its cadence (a `/data` stall is the case that matters -
    MooseFS can hang a plain `iterdir()`/`stat()` with no timeout
    parameter to give it), that collector's own thread simply stays
    blocked in it - not scheduled again until it returns - and the panel
    shows "stalled Ns" (computed from `call_started_at`, not a second
    watchdog thread). Nothing here ever spawns an additional thread past
    `start()`, so a stuck collector costs exactly one blocked daemon
    thread for the life of the process, never a growing pile of them, and
    `stop()` never joins - every thread is daemon=True, so process exit is
    never blocked on one, stuck or not.

    The UI thread only ever calls `snapshot()`/`history()` (both take the
    lock briefly and return copies) - it never calls a collector directly,
    so a slow `/data` `du` or an unreachable vLLM can never stall a
    keypress or a redraw.
    """

    def __init__(self, options: dict | None = None) -> None:
        self.options = dict(options or {})
        self._lock = threading.Lock()
        self._state = {name: PanelState() for name in CADENCE_S}
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()
        self._vllm_prev: tuple[dict, float] | None = None
        self._host_prev: tuple[dict, float] | None = None
        self._mt_baseline: dict | None = None
        # Per-collector override of the NEXT wait (falls back to
        # CADENCE_S[name] when absent). Only "services" ever sets this
        # today - see `_tick`'s services special-case: the RPC path is
        # cheap enough for the normal cadence, the `supervisorctl`
        # subprocess fallback is not, so it backs off to a slower one.
        self._next_wait: dict[str, float] = {}
        self._history: dict[str, deque] = {
            "gpu_util": deque(maxlen=HISTORY_LEN),
            "gen_tps": deque(maxlen=HISTORY_LEN),
            "kv_cache_pct": deque(maxlen=HISTORY_LEN),
            "rtt_ms": deque(maxlen=HISTORY_LEN),
        }

    def start(self) -> None:
        for name in CADENCE_S:
            t = threading.Thread(target=self._loop, args=(name,), daemon=True, name=f"zl-dash-{name}")
            self._threads.append(t)
            t.start()

    def stop(self) -> None:
        """Signal every collector thread to stop after its current tick.
        Deliberately never joins: a thread blocked inside a stalled `/data`
        call may never return, and this dashboard must never hang program
        exit waiting for one. Every thread is daemon=True, so the process
        exits regardless of whether any of them are still blocked.
        """
        self._stop.set()

    def _loop(self, name: str) -> None:
        first = True
        while not self._stop.is_set():
            if not first:
                wait_s = self._next_wait.get(name, CADENCE_S[name])
                if self._stop.wait(wait_s):
                    break
            first = False
            self._tick(name)

    def _tick(self, name: str) -> None:
        fn = getattr(self, f"_collect_{name}")
        start = time.time()
        with self._lock:
            self._state[name].call_started_at = start
        try:
            value = fn()
            err: Exception | None = None
        except Exception as exc:  # noqa: BLE001 - recorded, never crashes the loop or the dashboard
            value = None
            err = exc
        now = time.time()
        with self._lock:
            st = self._state[name]
            st.collected_at = now
            st.call_started_at = None
            if err is not None:
                st.error = str(err)
                # Keep the last good `data` on a transient failure - the UI
                # shows it stale-red rather than blanking a working panel.
            else:
                st.data = value
                st.error = None
        if name == "services" and err is None and isinstance(value, dict):
            # RPC is cheap enough for the normal cadence; the subprocess
            # fallback costs a fork+exec, so it backs off to CADENCE_S
            # tripled (15s at the current 5s default) rather than polling
            # it at the same rate as the free path.
            self._next_wait["services"] = (
                CADENCE_S["services"] * 3.0 if value.get("source") == "subprocess" else CADENCE_S["services"]
            )

    # --- individual collectors, each wrapping the module-level function ---

    def _collect_health(self) -> dict:
        result = collect_health()
        outcomes = ((result.get("doc") or {}).get("memory_tail") or {}).get("outcomes")
        result["memory_tail_deltas"] = self._memory_tail_deltas(outcomes)
        return result

    def _memory_tail_deltas(self, outcomes: Any) -> dict:
        if not isinstance(outcomes, dict):
            return {}
        if self._mt_baseline is None:
            self._mt_baseline = dict(outcomes)
            return {k: 0 for k in outcomes}
        if any(outcomes.get(k, 0) < self._mt_baseline.get(k, 0) for k in outcomes):
            # A counter went backwards: the compactor restarted. Re-baseline
            # so "since dashboard start" never shows a negative number.
            self._mt_baseline = dict(outcomes)
            return {k: 0 for k in outcomes}
        return {k: outcomes.get(k, 0) - self._mt_baseline.get(k, 0) for k in outcomes}

    def _collect_services(self) -> dict:
        return collect_services()

    def _collect_gpu(self) -> list[dict]:
        gpus = collect_gpu()
        if gpus:
            util = gpus[0].get("utilization_gpu_pct")
            if util is not None:
                with self._lock:
                    self._history["gpu_util"].append(util)
        return gpus

    def _collect_vllm(self) -> dict:
        sample = collect_vllm()
        now = time.time()
        data = dict(sample)
        if sample.get("source") == "metrics":
            metrics = sample["metrics"]
            kv = metrics.get("kv_cache_usage_pct")
            if kv is not None:
                with self._lock:
                    self._history["kv_cache_pct"].append(kv)
            if self._vllm_prev is not None:
                prev_metrics, prev_ts = self._vllm_prev
                rates = vllm_rates(prev_metrics, prev_ts, metrics, now)
                data["rates"] = rates
                data["ttft_quantiles_s"] = vllm_latency_quantiles(
                    prev_metrics.get("ttft_buckets"), metrics.get("ttft_buckets")
                )
                data["itl_quantiles_s"] = vllm_latency_quantiles(
                    prev_metrics.get("itl_buckets"), metrics.get("itl_buckets")
                )
                gen_tps = rates.get("generation_tokens_per_s")
                if gen_tps is not None:
                    with self._lock:
                        self._history["gen_tps"].append(gen_tps)
            self._vllm_prev = (metrics, now)
        else:
            self._vllm_prev = None
        return data

    def _collect_host(self) -> dict:
        raw = collect_host()
        now = time.time()
        data = dict(raw)
        if self._host_prev is not None:
            prev_cpu, prev_ts = self._host_prev
            data["cpu_percent"] = cpu_percent_from_stat(prev_cpu, raw.get("cpu_times"))
        else:
            data["cpu_percent"] = None
        self._host_prev = (raw.get("cpu_times"), now)
        return data

    def _collect_data_usage(self) -> dict:
        return collect_data_usage()

    def _collect_backups(self) -> dict:
        return collect_backups()

    def _collect_ssh(self) -> dict:
        return collect_ssh_session()

    def _collect_log(self) -> dict:
        return collect_log_signal()

    # --- reads (UI thread) ---

    def record_rtt_sample(self, ms: float) -> None:
        """The dashboard's curses input loop calls this directly (it owns
        the DSR probe/reply itself - see dash.py) so the sample lands in
        the same history store as every other sparkline.
        """
        with self._lock:
            self._history["rtt_ms"].append(ms)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                name: {
                    "data": st.data,
                    "error": st.error,
                    "collected_at": st.collected_at,
                    "call_started_at": st.call_started_at,
                }
                for name, st in self._state.items()
            }

    def history(self) -> dict[str, list[float]]:
        with self._lock:
            return {k: list(v) for k, v in self._history.items()}
