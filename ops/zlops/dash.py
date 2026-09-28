"""`zl dash` — a live terminal telemetry dashboard for use INSIDE an SSH
session on the pod (the owner's own words: "a telemetry dashboard for the
ssh connection"). No new port, no web server: this reads the same things an
operator would `curl`/`supervisorctl status`/`nvidia-smi` by hand, and shows
them live in the terminal the SSH session already has.

Two halves, deliberately separated so the UI is testable without a real
terminal:

  - `render_frame(snapshot, history, width, now, opts) -> [(text, attr), ...]`
    is a PURE function: given one telemetry snapshot (the shape
    `telemetry.Hub.snapshot()`/`telemetry.collect_snapshot()` both produce)
    and a width, it returns the exact lines to print - no line longer than
    `width`, no curses call anywhere in it. `--once`/`--json` and the
    curses drawer both go through this one function, so what you see live
    is provably the same rendering the tests exercise.
  - Everything below `_run_curses` drives an actual terminal: starts
    `telemetry.Hub`'s background collector threads, reads keys (and the
    dashboard's own terminal round-trip probe reply) through one input
    loop, and repaints only on a data change or once per `--interval`
    (the CPU-cost requirement: this dashboard should cost under 2% CPU -
    see the module docstring's own report in the brief for the measured
    number).

`main(argv) -> int` is the fixed seam `zl dash` dispatches to.
"""

from __future__ import annotations

import argparse
import curses
import hashlib
import json
import locale
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from . import podenv, telemetry

MIN_COLS = 80
MIN_LINES = 24
TWO_COLUMN_MIN_WIDTH = 120
SPARK_WIDTH = 36

# =============================================================================
# Small formatting helpers (pure; used only by the panel builders below)
# =============================================================================


def _fmt_bytes(n: float | None) -> str:
    if n is None:
        return "?"
    n = float(n)
    for unit in ("B", "K", "M", "G", "T"):
        if abs(n) < 1024.0 or unit == "T":
            return f"{n:.1f}{unit}" if unit != "B" else f"{n:.0f}{unit}"
        n /= 1024.0
    return f"{n:.1f}T"


def _fmt_kb(kb: float | None) -> str:
    if kb is None:
        return "?"
    return _fmt_bytes(kb * 1024)


def _fmt_pct(v: float | None, digits: int = 1) -> str:
    return "?" if v is None else f"{v:.{digits}f}%"


def _fmt_pct01(v: float | None, digits: int = 1) -> str:
    return "?" if v is None else f"{v * 100:.{digits}f}%"


def _fmt_rate(v: float | None) -> str:
    return "?" if v is None else f"{v:.1f}/s"


def _fmt_ms_from_s(v: float | None) -> str:
    return "?" if v is None else f"{v * 1000:.0f}ms"


def _fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    seconds = int(seconds)
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}d {hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _clip(text: str, width: int) -> str:
    if width <= 0:
        return ""
    return text[:width]


def _phoenix_local_time(epoch: float) -> str:
    """America/Phoenix has no DST (fixed UTC-7 year-round), so a zoneinfo
    failure (no tzdata installed - possible on a minimal image) falls back
    to a hardcoded UTC-7 offset rather than losing the "her time" header
    entirely.
    """
    try:
        from zoneinfo import ZoneInfo

        dt = datetime.fromtimestamp(epoch, ZoneInfo("America/Phoenix"))
    except Exception:  # noqa: BLE001 - zoneinfo/tzdata missing, any reason
        dt = datetime.fromtimestamp(epoch, timezone(timedelta(hours=-7)))
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _utc_time(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _supports_unicode() -> bool:
    enc = (sys.stdout.encoding or "") if hasattr(sys.stdout, "encoding") else ""
    if "utf" in enc.lower():
        return True
    try:
        return "utf" in (locale.getpreferredencoding() or "").lower()
    except (ValueError, locale.Error):
        return False


_SPARK_UNICODE = "▁▂▃▄▅▆▇█"
_SPARK_ASCII = " .:-=+*#"


def sparkline(values: list[float | None], width: int, ascii_only: bool = False) -> str:
    """Render the last `width` samples of `values` as a one-line spark
    graph. A value below the observed min/max range in this window still
    renders (clamped) rather than raising; None samples render as a gap.
    """
    ramp = _SPARK_ASCII if ascii_only else _SPARK_UNICODE
    if width <= 0:
        return ""
    window = values[-width:] if values else []
    finite = [v for v in window if v is not None]
    if not finite:
        return ""
    lo, hi = min(finite), max(finite)
    out = []
    for v in window:
        if v is None:
            out.append(" ")
            continue
        if hi <= lo:
            level = 0
        else:
            level = int((v - lo) / (hi - lo) * (len(ramp) - 1))
            level = max(0, min(len(ramp) - 1, level))
        out.append(ramp[level])
    return "".join(out)


def compute_build_id(main_py_path: Path | None = None) -> str | None:
    """sha256 of the compactor's `main.py` as it exists on THIS running
    container's overlay right now, first 12 hex chars - a quick "what code
    is actually loaded" fingerprint for the header, independent of any
    version string main.py may or may not print accurately.
    """
    path = main_py_path or (podenv.COMPACTOR_PKG / "main.py")
    try:
        data = path.read_bytes()
    except OSError:
        return None
    return hashlib.sha256(data).hexdigest()[:12]


# =============================================================================
# Panel builders — each takes its slice of the snapshot and returns
# [(text, attr), ...]. `attr` is a semantic tag ("ok"/"warn"/"crit"/"stale"/
# "dim"/"bold"/"header"/"help"/"normal"), translated to real styling by the
# curses drawer or by --once's ANSI mapper — never a curses call here.
# =============================================================================


def _age_s(entry: dict, now: float) -> float | None:
    ts = (entry or {}).get("collected_at")
    return None if ts is None else max(0.0, now - ts)


def _is_stale(age: float | None, cadence_s: float) -> bool:
    return age is not None and age > telemetry.STALE_MULTIPLE * cadence_s


def _stalled_s(entry: dict, now: float, cadence_s: float) -> float | None:
    """Seconds a collector's call has been in flight, if it has been
    running long enough to call out - `entry["call_started_at"]` is set
    the instant `Hub._tick` calls the collector and cleared the instant it
    returns, so a non-None value here (past the threshold) means "this
    panel's single worker thread is still blocked inside its call right
    now", distinct from "the last successful read is merely old" (that's
    `_is_stale`). Never confuse the two in the UI: a stalled collector's
    LAST GOOD DATA is still shown (so the panel isn't blank), just marked
    with how long the current call has been stuck.
    """
    started = (entry or {}).get("call_started_at")
    if started is None:
        return None
    age = max(0.0, now - started)
    return age if age > telemetry.STALL_MULTIPLE * cadence_s else None


def _title(name: str, age: float | None, stale: bool, stalled_s: float | None = None) -> tuple[str, str]:
    if stalled_s is not None:
        suffix = f"(stalled {stalled_s:.0f}s)"
        return (f"-- {name} {suffix} --", "crit")
    if age is None:
        suffix = "(no data yet)"
    else:
        suffix = f"(age {age:.0f}s)" + (" STALE" if stale else "")
    return (f"-- {name} {suffix} --", "stale" if stale else "header")


def _panel_health(entry: dict, now: float) -> list[tuple[str, str]]:
    age = _age_s(entry, now)
    stale = _is_stale(age, telemetry.CADENCE_S["health"])
    stalled = _stalled_s(entry, now, telemetry.CADENCE_S["health"])
    lines = [_title("Health", age, stale, stalled)]
    err = entry.get("error")
    data = entry.get("data")
    if err:
        lines.append((f"  error: {err}", "crit"))
        return lines
    if not data:
        lines.append(("  (no data yet)", "dim"))
        return lines
    interp = data.get("interpreted") or {}
    status = interp.get("status", "?")
    # Round-4 addendum (v3.1.9.8): /health/full's own `requests_in_flight`,
    # shown right on the header/status line -- HIDDEN entirely when the
    # field is absent (v3.1.9 compactor does not report it at all; a
    # missing field must never be shown as 0).
    doc = data.get("doc") or {}
    request_tracing = doc.get("request_tracing") or {}
    inflight = request_tracing.get("requests_in_flight")
    status_line = f"  status: {status}"
    if isinstance(inflight, int):
        status_line += f"   in_flight={inflight}"
    lines.append((status_line, status if status in ("ok", "warn", "crit") else "normal"))
    # Boot-time word-ban warm-up (coordinator addendum, v3.1.9.8): shown
    # only when the field is present at all (v3.1.9/pre-warm-up compactors
    # never report it -- must never be shown as some default). Read
    # tolerantly (`doc.get("word_ban", {})`, the shape is not committed
    # yet): `warm_state` is one of `idle | waiting_idle | warming |
    # backoff | disabled | vllm_unreachable`, marked warn for the two
    # failure-shaped states and neutral for everything else -- an
    # UNKNOWN future value is shown raw rather than crashing or being
    # hidden.
    word_ban = doc.get("word_ban")
    if isinstance(word_ban, dict):
        warm_state = word_ban.get("warm_state")
        if warm_state is not None:
            attr = "warn" if warm_state in ("backoff", "vllm_unreachable") else "normal"
            lines.append((f"  word_ban.warm_state: {warm_state}", attr))
    findings = list(interp.get("findings") or [])
    # crit > warn > info/ok, before truncating - a low-priority finding
    # must never push a crit one off the shown list (the review's own
    # "the cut to 6 findings can hide crit ones" finding).
    _LEVEL_ORDER = {"crit": 0, "warn": 1, "info": 2, "ok": 3}
    findings.sort(key=lambda f: _LEVEL_ORDER.get(f.get("level"), 4))
    shown, hidden = findings[:6], findings[6:]
    for f in shown:
        lines.append((f"  [{f.get('level', '?'):>4}] {f.get('text', '')}", f.get("level", "normal")))
    if hidden:
        hidden_crit = sum(1 for f in hidden if f.get("level") == "crit")
        # By construction (crit sorts first) a hidden crit can only happen
        # if there are more than 6 crits, which the "+N more" line itself
        # flags loudly rather than silently dropping.
        attr = "crit" if hidden_crit else "dim"
        lines.append((f"  +{len(hidden)} more finding(s){f' ({hidden_crit} crit!)' if hidden_crit else ''}", attr))
    checks = doc.get("checks") or {}
    hierarchy = checks.get("hierarchy")
    if isinstance(hierarchy, dict):
        lines.append((f"  hierarchy catching_up: {hierarchy.get('catching_up')}", "dim"))
    reuse = checks.get("reuse")
    if isinstance(reuse, dict):
        lines.append(
            (
                f"  reuse attempted={reuse.get('attempted')} succeeded={reuse.get('succeeded')}"
                f" last_reason={reuse.get('last_reason')} declined_recently={reuse.get('declined_recently')}",
                "dim",
            )
        )
    margin = checks.get("budget_margin")
    if isinstance(margin, dict):
        lines.append((f"  budget_margin={margin.get('margin')} ceiling={margin.get('ceiling')}", "dim"))
    trunc = checks.get("truncated_summaries")
    if isinstance(trunc, dict):
        lines.append(
            (f"  truncated_summaries hierarchy={trunc.get('hierarchy')} compaction={trunc.get('compaction')}", "dim")
        )
    deltas = data.get("memory_tail_deltas") or {}
    nonzero = {k: v for k, v in deltas.items() if v}
    if nonzero:
        lines.append((f"  memory_tail since dash start: {nonzero}", "warn"))
    # v3.1.9.9 Cloudflare Tunnel: "ready"/"not ready"/"unknown"/"disabled"
    # when /health/full reports a `tunnel` key at all - HIDDEN entirely on
    # an older compactor that predates it (telemetry.tunnel_status_word
    # returns None there), never shown as some default.
    tunnel_status = data.get("tunnel_status")
    if tunnel_status is not None:
        attr = "warn" if tunnel_status == "not ready" else "normal"
        line = f"  tunnel: {tunnel_status}"
        # Round-11 review (LOW): "unknown" alone (cloudflared's own /ready
        # probe was unreachable/not yet answered) tells the operator
        # nothing about WHY - the compactor's probe error text (e.g. a
        # ConnectError) is right there on `tunnel.error` and is safe to
        # show (it's connector/network diagnostic text, never her chat
        # content) - truncated the same way `_gpu_line`'s own nvidia-smi
        # failure text is, so one runaway error string can't blow up the
        # panel.
        if tunnel_status == "unknown":
            tunnel_doc = doc.get("tunnel")
            error = tunnel_doc.get("error") if isinstance(tunnel_doc, dict) else None
            if error:
                line += f"  ({str(error)[:160]})"
        lines.append((line, attr))
    return lines


def _panel_services(entry: dict, now: float, health_doc: dict | None = None) -> list[tuple[str, str]]:
    age = _age_s(entry, now)
    stale = _is_stale(age, telemetry.CADENCE_S["services"])
    stalled = _stalled_s(entry, now, telemetry.CADENCE_S["services"])
    lines = [_title("Services", age, stale, stalled)]
    if entry.get("error"):
        lines.append((f"  error: {entry['error']}", "crit"))
        return lines
    services = (entry.get("data") or {}).get("services") or []
    if not services:
        lines.append(("  (no data yet)", "dim"))
    for s in services:
        name = s.get("name", "?")
        state = s.get("state", "?")
        # v3.1.9.9: `cloudflared` sits in ONE_SHOT_OR_OPTIONAL (an EXITED
        # cloudflared with no token set is correct), but a token IS set
        # (tunnel.configured) and cloudflared isn't RUNNING anyway is a
        # real fault ONE_SHOT_OR_OPTIONAL membership alone would hide -
        # see telemetry.cloudflared_fault's own docstring.
        tunnel_warn = telemetry.cloudflared_fault(health_doc, state) if name == "cloudflared" else None
        if tunnel_warn:
            attr, marker = "warn", "!"
        else:
            attr = "crit" if s.get("unexpected") else ("ok" if state == "RUNNING" else "dim")
            marker = "!" if s.get("unexpected") else " "
        line = f" {marker} {name:<16} {state:<9} pid={s.get('pid')} uptime={s.get('uptime')}"
        if tunnel_warn:
            line += f"  <-- {tunnel_warn}"
        lines.append((line, attr))
    return lines


def _panel_gpu(entry: dict, now: float, history: dict, ascii_only: bool) -> list[tuple[str, str]]:
    age = _age_s(entry, now)
    stale = _is_stale(age, telemetry.CADENCE_S["gpu"])
    stalled = _stalled_s(entry, now, telemetry.CADENCE_S["gpu"])
    lines = [_title("GPU", age, stale, stalled)]
    if entry.get("error"):
        lines.append((f"  {entry['error']}", "warn"))
        return lines
    gpus = entry.get("data") or []
    if not gpus:
        lines.append(("  (no data yet)", "dim"))
        return lines
    for i, g in enumerate(gpus):
        mem_used = g.get("memory_used_mb")
        mem_total = g.get("memory_total_mb")
        mem_used_s = _fmt_bytes(mem_used * 1024 * 1024) if mem_used is not None else "?"
        mem_total_s = _fmt_bytes(mem_total * 1024 * 1024) if mem_total is not None else "?"
        lines.append(
            (
                f"  [{i}] {g.get('name', '?')}  util={_fmt_pct(g.get('utilization_gpu_pct'))}"
                f" mem={mem_used_s}/{mem_total_s}"
                f" temp={g.get('temperature_c')}C power={g.get('power_draw_w')}/{g.get('power_limit_w')}W"
                f" clk={g.get('clock_sm_mhz')}MHz",
                "normal",
            )
        )
    spark = sparkline(history.get("gpu_util") or [], SPARK_WIDTH, ascii_only)
    if spark:
        lines.append((f"  util history: {spark}", "dim"))
    return lines


def _is_word_ban_warming(word_ban: dict | None) -> bool:
    """True when the boot-time word-ban warm-up request (coordinator
    addendum, v3.1.9.8) is plausibly the reason vLLM shows running/waiting
    activity -- `warm_state == "warming"`, or `warmups_in_flight > 0` (the
    warm-up is NOT counted in `request_tracing.requests_in_flight`, only
    here). Read tolerantly: the shape is not committed yet.
    """
    if not isinstance(word_ban, dict):
        return False
    if word_ban.get("warm_state") == "warming":
        return True
    n = word_ban.get("warmups_in_flight")
    return isinstance(n, (int, float)) and n > 0


def _panel_vllm(
    entry: dict, now: float, history: dict, ascii_only: bool, word_ban: dict | None = None,
) -> list[tuple[str, str]]:
    age = _age_s(entry, now)
    stale = _is_stale(age, telemetry.CADENCE_S["vllm"])
    stalled = _stalled_s(entry, now, telemetry.CADENCE_S["vllm"])
    lines = [_title("vLLM", age, stale, stalled)]
    if entry.get("error"):
        lines.append((f"  {entry['error']}", "crit"))
        return lines
    data = entry.get("data")
    if not data:
        lines.append(("  (no data yet)", "dim"))
        return lines
    warmup_note = " (word-ban warm-up)" if _is_word_ban_warming(word_ban) else ""
    if data.get("source") == "log":
        log = data.get("log") or {}
        # The log fallback's numbers are only as fresh as the LINE they came
        # from, not as fresh as when the dashboard happened to read the
        # file - a vllm.log line can sit unchanged for a while if the
        # engine is idle, or vLLM itself may be wedged. Never presented as
        # live once stale (brief: "never present stale numbers as live").
        line_age = log.get("line_age_s")
        is_stale = bool(log.get("stale", True))
        age_text = f"line age {line_age:.0f}s" if line_age is not None else "line has no timestamp"
        stale_text = " STALE - not live" if is_stale else " (live)"
        lines.append((f"  [from log] (/metrics unreachable) {age_text}{stale_text}", "stale" if is_stale else "warn"))
        row_attr = "stale" if is_stale else "normal"
        lines.append(
            (
                f"  running={log.get('running')} waiting={log.get('waiting')}{warmup_note}"
                f" prompt={_fmt_rate(log.get('prompt_tokens_per_s'))} gen={_fmt_rate(log.get('generation_tokens_per_s'))}",
                row_attr,
            )
        )
        lines.append(
            (
                f"  kv_cache={_fmt_pct(log.get('kv_cache_usage_pct'))}"
                f" prefix_hit={_fmt_pct(log.get('prefix_cache_hit_rate_pct'))}"
                f" preemptions={log.get('preemptions')}",
                row_attr,
            )
        )
        return lines
    metrics = data.get("metrics") or {}
    lines.append(
        (
            f"  running={metrics.get('running')} waiting={metrics.get('waiting')}{warmup_note}"
            f" kv_cache={_fmt_pct(metrics.get('kv_cache_usage_pct'))}",
            "normal",
        )
    )
    rates = data.get("rates") or {}
    lines.append(
        (
            f"  prompt={_fmt_rate(rates.get('prompt_tokens_per_s'))}"
            f" gen={_fmt_rate(rates.get('generation_tokens_per_s'))}"
            f" prefix_hit={_fmt_pct01(rates.get('prefix_cache_hit_rate'))}"
            f" preempt+={rates.get('preemptions_delta')}",
            "normal",
        )
    )
    ttft = data.get("ttft_quantiles_s") or {}
    itl = data.get("itl_quantiles_s") or {}
    lines.append(
        (
            f"  TTFT p50={_fmt_ms_from_s(ttft.get('p50_s'))} p95={_fmt_ms_from_s(ttft.get('p95_s'))}"
            f"   ITL p50={_fmt_ms_from_s(itl.get('p50_s'))} p95={_fmt_ms_from_s(itl.get('p95_s'))}",
            "normal",
        )
    )
    spark = sparkline(history.get("gen_tps") or [], SPARK_WIDTH, ascii_only)
    if spark:
        lines.append((f"  gen tok/s: {spark}", "dim"))
    spark2 = sparkline(history.get("kv_cache_pct") or [], SPARK_WIDTH, ascii_only)
    if spark2:
        lines.append((f"  kv cache%: {spark2}", "dim"))
    return lines


def _panel_ssh(entry: dict, now: float, history: dict, ascii_only: bool, no_rtt: bool) -> list[tuple[str, str]]:
    age = _age_s(entry, now)
    stale = _is_stale(age, telemetry.CADENCE_S["ssh"])
    stalled = _stalled_s(entry, now, telemetry.CADENCE_S["ssh"])
    lines = [_title("SSH session", age, stale, stalled)]
    if entry.get("error"):
        lines.append((f"  {entry['error']}", "crit"))
        return lines
    data = entry.get("data")
    if not data:
        lines.append(("  (no data yet)", "dim"))
        return lines
    conn = data.get("connection")
    if conn:
        lines.append(
            (
                f"  client {conn['client_ip']}:{conn['client_port']}"
                f" -> server {conn['server_ip']}:{conn['server_port']}",
                "normal",
            )
        )
    else:
        lines.append(("  RunPod Web Terminal (no SSH)", "dim"))
    session = data.get("session") or {}
    if session.get("in_ssh"):
        lines.append((f"  session duration: {_fmt_duration(session.get('duration_s'))}", "normal"))
    lines.append(
        (
            f"  established on port {data.get('port')}: {data.get('established_count')}"
            f"  clients: {', '.join(data.get('established_client_ips') or []) or '-'}",
            "normal",
        )
    )
    lines.append(
        ("  sshd listening: yes" if data.get("listening") else "  sshd listening: NO",
         "ok" if data.get("listening") else "warn")
    )
    if data.get("in_tmux"):
        lines.append(("  tmux: yes", "ok"))
    else:
        lines.append(
            (
                "  tmux: NOT in tmux -- this dashboard dies with the connection"
                " (zl tools --apply installs tmux)",
                "warn",
            )
        )
    if no_rtt:
        lines.append(("  RTT: disabled (--no-rtt)", "dim"))
    else:
        rtt_hist = [v for v in (history.get("rtt_ms") or []) if v is not None]
        if rtt_hist:
            last, avg, mx = rtt_hist[-1], sum(rtt_hist) / len(rtt_hist), max(rtt_hist)
            spark = sparkline(rtt_hist, SPARK_WIDTH, ascii_only)
            lines.append(
                (f"  RTT (SSH+terminal) last={last:.0f}ms avg={avg:.0f}ms max={mx:.0f}ms  {spark}", "normal")
            )
        else:
            lines.append(("  RTT (SSH+terminal): no sample yet", "dim"))
    return lines


def _panel_host_data_backups(
    host_entry: dict, data_entry: dict, backups_entry: dict, now: float
) -> list[tuple[str, str]]:
    age = _age_s(host_entry, now)
    stale = _is_stale(age, telemetry.CADENCE_S["host"])
    stalled = _stalled_s(host_entry, now, telemetry.CADENCE_S["host"])
    lines = [_title("Host / data / backups", age, stale, stalled)]

    if host_entry.get("error"):
        lines.append((f"  host error: {host_entry['error']}", "crit"))
    else:
        hd = host_entry.get("data")
        if hd:
            mem = hd.get("meminfo") or {}
            total, avail = mem.get("MemTotal"), mem.get("MemAvailable")
            used = _fmt_bytes(total - avail) if (total is not None and avail is not None) else "?"
            lines.append(
                (
                    f"  load: {hd.get('loadavg')}  cpu: {_fmt_pct(hd.get('cpu_percent'))}"
                    f"  mem used: {used} / {_fmt_bytes(total)}",
                    "normal",
                )
            )
            lines.append(
                (
                    f"  / free: {_fmt_bytes(hd.get('root_free_bytes'))} / {_fmt_bytes(hd.get('root_total_bytes'))}"
                    f"  procs: {hd.get('proc_count')}",
                    "normal",
                )
            )
        else:
            lines.append(("  (no data yet)", "dim"))

    if data_entry.get("error"):
        lines.append((f"  /data error: {data_entry['error']}", "crit"))
    else:
        du = data_entry.get("data")
        if du:
            pct = f" ({du['pct_of_volume']:.1f}% of ZL_VOLUME_GB)" if du.get("pct_of_volume") is not None else ""
            age_data = _age_s(data_entry, now)
            age_suffix = f"  (age {age_data:.0f}s)" if age_data is not None else ""
            lines.append(
                (
                    f"  /data usage (du, NOT a quota - whole cluster): {_fmt_kb(du.get('total_kb'))}{pct}{age_suffix}",
                    "normal",
                )
            )
            for e in sorted((du.get("entries") or []), key=lambda e: -e.get("kb", 0))[:5]:
                lines.append((f"    {e.get('name')}: {_fmt_kb(e.get('kb'))}", "dim"))
        else:
            lines.append(("  /data usage: (no data yet)", "dim"))

    if backups_entry.get("error"):
        lines.append((f"  backups error: {backups_entry['error']}", "crit"))
    else:
        bk = backups_entry.get("data")
        if bk:
            if bk.get("latest_mtime"):
                age_h = (now - bk["latest_mtime"]) / 3600.0
                level = telemetry.backup_age_level(age_h)
                lines.append(
                    (f"  newest backup: {bk.get('latest')}  age={age_h:.1f}h  count={bk.get('count')}", level)
                )
            else:
                lines.append(("  no backup archives found", "crit"))
        else:
            lines.append(("  backups: (no data yet)", "dim"))
    return lines


def _panel_log(entry: dict, now: float, show: bool) -> list[tuple[str, str]]:
    age = _age_s(entry, now)
    stale = _is_stale(age, telemetry.CADENCE_S["log"])
    stalled = _stalled_s(entry, now, telemetry.CADENCE_S["log"])
    lines = [_title("Log signal (l to toggle)", age, stale, stalled)]
    if not show:
        lines.append(("  (hidden — press l to show)", "dim"))
        return lines
    if entry.get("error"):
        lines.append((f"  {entry['error']}", "crit"))
        return lines
    data = entry.get("data") or {}
    entries = data.get("entries") or []
    if not entries:
        lines.append(("  (no ERROR/CRITICAL/WARNING/Traceback lines in the tail)", "dim"))
    for e in entries:
        ts = e.get("timestamp") or "?"
        lines.append((f"  [{e.get('file')}] {ts} {e.get('text')}", e.get("level", "normal")))
    for fname, msg in (data.get("file_errors") or {}).items():
        lines.append((f"  ({fname}: {msg})", "dim"))
    return lines


def _header(now_epoch: float, opts: dict) -> list[tuple[str, str]]:
    pod_id = opts.get("pod_id") or "?"
    gpu_name = opts.get("gpu_name") or "?"
    build_id = opts.get("build_id") or "?"
    interval = opts.get("interval") or 0.0
    paused = " [PAUSED]" if opts.get("paused") else ""
    return [
        (f"zl dash — pod {pod_id}  gpu {gpu_name}  compactor build {build_id}", "header"),
        (
            f"local (Phoenix) {_phoenix_local_time(now_epoch)}   UTC {_utc_time(now_epoch)}"
            f"   refresh={interval:.0f}s{paused}",
            "header",
        ),
    ]


def _footer() -> list[tuple[str, str]]:
    return [("q quit   p/space pause   r refresh   j/k scroll   PgUp/PgDn   l log panel   ? help", "dim")]


_HELP_TEXT = [
    "zl dash — help",
    "",
    "q          quit (restores the terminal)",
    "p, space   pause/resume the live view (collectors keep running)",
    "r          redraw now",
    "j / k      scroll down / up one line",
    "PgDn/PgUp  scroll a page",
    "l          toggle the log-signal panel",
    "?          toggle this help",
    "",
    "Panels: Health, Services, GPU, vLLM, SSH session, Host/data/backups,",
    "Log signal. Each panel shows its own data age and turns red ('STALE')",
    "once it is older than 3x its own collection interval.",
    "",
    "Press any key to close this help.",
]


def _help_frame(width: int) -> list[tuple[str, str]]:
    return [(_clip(t, width), "help") for t in _HELP_TEXT]


def _merge_columns(left: list[tuple[str, str]], right: list[tuple[str, str]], col_w: int) -> list[tuple[str, str]]:
    """Side-by-side merge for the >=120-column layout. Each output line
    carries ONE attr (the (text, attr) contract has no per-span styling),
    chosen from the left half unless it's blank — a deliberate, documented
    simplification, not a bug: two-column mode is a visual convenience, and
    losing color on an occasional blank-left/colored-right line costs
    nothing a test checks for (no line exceeding width, no exception).
    """
    n = max(len(left), len(right))
    out = []
    for i in range(n):
        lt, la = left[i] if i < len(left) else ("", "normal")
        rt, ra = right[i] if i < len(right) else ("", "normal")
        lt_c = lt[:col_w].ljust(col_w)
        attr = la if lt.strip() else ra
        out.append((f"{lt_c} | {rt}", attr))
    return out


def render_frame(snapshot: dict, history: dict, width: int, now: float, opts: dict) -> list[tuple[str, str]]:
    """Pure: one telemetry snapshot + a terminal width -> the exact lines
    to print, each already clipped to `width`. No curses call anywhere in
    this function or anything it calls - this is what makes the UI
    testable without a terminal (--once uses it directly; the curses
    drawer only adds color and a scroll window on top).
    """
    width = max(1, width)
    if opts.get("help"):
        return _help_frame(width)

    ascii_only = bool(opts.get("ascii_only"))
    no_rtt = bool(opts.get("no_rtt"))

    lines: list[tuple[str, str]] = []
    lines.extend(_header(now, opts))
    lines.append(("", "normal"))

    # word_ban/tunnel both live under /health/full's own doc - pulled out
    # here, tolerantly, once, so the vLLM panel can label a running/waiting
    # count as the boot-time warm-up rather than her real traffic, and the
    # Services panel can tell "cloudflared disabled by config" apart from
    # "tunnel configured but cloudflared is down" (telemetry.cloudflared_fault).
    _health_doc = ((snapshot.get("health", {}) or {}).get("data") or {}).get("doc") or {}
    if not isinstance(_health_doc, dict):
        _health_doc = {}
    word_ban = _health_doc.get("word_ban")

    health = _panel_health(snapshot.get("health", {}), now)
    services = _panel_services(snapshot.get("services", {}), now, _health_doc)
    gpu = _panel_gpu(snapshot.get("gpu", {}), now, history, ascii_only)
    hostdata = _panel_host_data_backups(
        snapshot.get("host", {}), snapshot.get("data_usage", {}), snapshot.get("backups", {}), now
    )
    vllm = _panel_vllm(snapshot.get("vllm", {}), now, history, ascii_only, word_ban)
    ssh = _panel_ssh(snapshot.get("ssh", {}), now, history, ascii_only, no_rtt)
    log = _panel_log(snapshot.get("log", {}), now, bool(opts.get("show_log", True)))

    blank = ("", "normal")
    left = health + [blank] + services + [blank] + gpu + [blank] + hostdata
    right = vllm + [blank] + ssh

    if width >= TWO_COLUMN_MIN_WIDTH:
        col_w = (width - 3) // 2
        lines.extend(_merge_columns(left, right, col_w))
        lines.append(blank)
        lines.extend(log)
    else:
        lines.extend(left)
        lines.append(blank)
        lines.extend(right)
        lines.append(blank)
        lines.extend(log)

    lines.append(blank)
    lines.extend(_footer())

    return [(_clip(text, width), attr) for text, attr in lines]


def too_small_frame(width: int, height: int) -> list[tuple[str, str]]:
    msg = f"terminal too small ({width}x{height}); need at least {MIN_COLS}x{MIN_LINES}"
    return [(_clip(msg, max(1, width)), "crit")]


# =============================================================================
# --once / --json
# =============================================================================

_ANSI = {
    "ok": "\x1b[32m",
    "warn": "\x1b[33m",
    "crit": "\x1b[31m",
    "stale": "\x1b[31m",
    "header": "\x1b[1;36m",
    "bold": "\x1b[1m",
    "dim": "\x1b[2m",
    "help": "\x1b[36m",
    "normal": "",
}
_ANSI_RESET = "\x1b[0m"


def _apply_ansi(text: str, attr: str, no_color: bool) -> str:
    if no_color or not text:
        return text
    code = _ANSI.get(attr, "")
    return f"{code}{text}{_ANSI_RESET}" if code else text


def _terminal_width_default() -> int:
    col = os.environ.get("COLUMNS", "").strip()
    if col.isdigit() and int(col) > 0:
        return int(col)
    return 100


def _gpu_name_from_snapshot(snapshot: dict) -> str | None:
    gpus = (snapshot.get("gpu") or {}).get("data") or []
    return gpus[0].get("name") if gpus else None


def _base_opts(args: argparse.Namespace, snapshot: dict) -> dict:
    return {
        "paused": False,
        "show_log": True,
        "help": False,
        "ascii_only": not _supports_unicode(),
        "interval": args.interval,
        "pod_id": podenv.RUNPOD_POD_ID or "?",
        "gpu_name": _gpu_name_from_snapshot(snapshot),
        "build_id": compute_build_id(),
        "no_rtt": args.no_rtt,
    }


def _run_once(args: argparse.Namespace) -> int:
    width = args.width or _terminal_width_default()
    snapshot = telemetry.collect_snapshot({"rate_gap_s": 2.0, "no_rtt": args.no_rtt})
    now = snapshot.get("_meta", {}).get("collected_at", time.time())
    opts = _base_opts(args, snapshot)
    lines = render_frame(snapshot, {}, width, now, opts)
    # Round-4 cosmetic: ANSI color escapes belong on a real terminal only --
    # `--once`'s output redirected to a file/pipe (`zl dash --once > f`,
    # `| grep`, a log capture, ...) must come out as plain text, the same
    # as any other well-behaved CLI (color auto-disabled when stdout is
    # not a tty), not literal `\x1b[...]` bytes. `--no-color` still forces
    # it off explicitly either way.
    no_color = args.no_color or not sys.stdout.isatty()
    print("# zl dash --once: rates need two samples, so two were taken ~2s apart")
    for text, attr in lines:
        print(_apply_ansi(text, attr, no_color))
    return 0


def _run_json(args: argparse.Namespace) -> int:
    snapshot = telemetry.collect_snapshot({"rate_gap_s": 2.0, "no_rtt": args.no_rtt})
    # json.dumps serialises a non-finite bucket edge (float('inf') as a
    # dict key from a `le="+Inf"` Prometheus label) using Python's
    # documented, if non-strict, "Infinity"/"NaN" tokens - fine for this
    # tool's own consumers (`zl bundle`, this dashboard, a human eyeballing
    # it), and round-trips through Python's own json.loads.
    print(json.dumps(snapshot, indent=2, sort_keys=True, default=str))
    return 0


# =============================================================================
# curses driver
# =============================================================================

_MONO_FALLBACK: dict[str, int] = {}
_COLOR_PAIR: dict[str, int] = {}


def _init_mono_fallback() -> None:
    _MONO_FALLBACK.update(
        {
            "ok": curses.A_NORMAL,
            "warn": curses.A_BOLD,
            "crit": curses.A_REVERSE,
            "stale": curses.A_REVERSE,
            "header": curses.A_BOLD | curses.A_UNDERLINE,
            "bold": curses.A_BOLD,
            "dim": curses.A_DIM,
            "help": curses.A_BOLD,
            "normal": curses.A_NORMAL,
        }
    )


def _init_colors() -> bool:
    try:
        curses.start_color()
        curses.use_default_colors()
    except curses.error:
        return False
    if not curses.has_colors():
        return False
    pairs = [
        ("ok", curses.COLOR_GREEN),
        ("warn", curses.COLOR_YELLOW),
        ("crit", curses.COLOR_RED),
        ("stale", curses.COLOR_RED),
        ("header", curses.COLOR_CYAN),
        ("help", curses.COLOR_CYAN),
    ]
    ok = True
    for i, (name, fg) in enumerate(pairs, start=1):
        try:
            curses.init_pair(i, fg, -1)
            _COLOR_PAIR[name] = i
        except curses.error:
            ok = False
    return ok


def _curses_attr(attr: str, color_ok: bool, no_color: bool) -> int:
    if no_color or not color_ok:
        return _MONO_FALLBACK.get(attr, curses.A_NORMAL)
    pair = _COLOR_PAIR.get(attr)
    base = curses.color_pair(pair) if pair else curses.A_NORMAL
    if attr == "header":
        base |= curses.A_BOLD
    if attr == "dim":
        base |= curses.A_DIM
    return base


def _safe_addstr(win, y: int, x: int, text: str, attr: int) -> None:
    try:
        win.addstr(y, x, text, attr)
    except curses.error:
        pass  # bottom-right cell / off-screen write; never fatal


class _InputReader:
    """Reads curses input a batch at a time and pulls a DSR reply
    (`ESC[row;colR`) out of the raw stream BEFORE anything else sees it, so
    the terminal's answer to our own RTT probe is never misdelivered to a
    keypress handler. Everything that isn't part of (or a possible prefix
    of) a DSR reply is returned as a plain key: a single character, or a
    curses special-key int (arrows, KEY_RESIZE, ...) for anything
    `keypad(True)` already translated.
    """

    def __init__(self, win) -> None:
        self.win = win
        self._buf = ""
        self._buf_started = 0.0

    def poll(self, now: float) -> tuple[list[Any], tuple[int, int] | None]:
        keys: list[Any] = []
        dsr: tuple[int, int] | None = None
        while True:
            ch = self.win.getch()
            if ch == -1:
                break
            if 0 <= ch < 256:
                if not self._buf:
                    self._buf_started = now
                self._buf += chr(ch)
                found = telemetry.find_dsr_reply(self._buf)
                if found:
                    row, col, start, end = found
                    for c in self._buf[:start]:
                        keys.append(c)
                    self._buf = self._buf[end:]
                    dsr = (row, col)
            else:
                self._flush(keys)
                keys.append(ch)
        if self._buf and not self._looks_like_prefix(self._buf):
            self._flush(keys)
        elif self._buf and (now - self._buf_started) > 0.1:
            # A lone ESC (or an abandoned partial sequence) with nothing
            # more arriving - don't hold it forever.
            self._flush(keys)
        return keys, dsr

    def _flush(self, keys: list[Any]) -> None:
        for c in self._buf:
            keys.append(c)
        self._buf = ""

    @staticmethod
    def _looks_like_prefix(buf: str) -> bool:
        if buf == "\x1b":
            return True
        if buf.startswith("\x1b["):
            return all(c.isdigit() or c == ";" for c in buf[2:])
        return False


def _send_dsr_probe() -> None:
    try:
        os.write(1, b"\x1b[6n")
    except OSError:
        pass


def _draw(stdscr, lines: list[tuple[str, str]], scroll: int, height: int, width: int, color_ok: bool, no_color: bool) -> int:
    max_scroll = max(0, len(lines) - height)
    scroll = max(0, min(scroll, max_scroll))
    stdscr.erase()
    for i, (text, attr) in enumerate(lines[scroll : scroll + height]):
        _safe_addstr(stdscr, i, 0, text, _curses_attr(attr, color_ok, no_color))
    stdscr.refresh()
    return scroll


def _curses_main(stdscr, args: argparse.Namespace) -> int:
    try:
        curses.curs_set(0)
    except curses.error:
        pass  # some terminals (or TERM values) don't support cursor visibility control - not fatal
    stdscr.nodelay(True)
    stdscr.keypad(True)
    _init_mono_fallback()
    color_ok = False if args.no_color else _init_colors()

    hub = telemetry.Hub({"no_rtt": args.no_rtt})
    hub.start()
    reader = _InputReader(stdscr)
    build_id = compute_build_id()
    state = {"paused": False, "show_log": True, "help": False, "scroll": 0}
    last_drawn: Any = None
    last_draw_time = 0.0
    last_rtt_probe = 0.0
    rtt_pending_since: float | None = None

    try:
        while True:
            now = time.time()
            keys, dsr = reader.poll(now)

            if dsr is not None and rtt_pending_since is not None:
                hub.record_rtt_sample((time.monotonic() - rtt_pending_since) * 1000.0)
                rtt_pending_since = None

            force_redraw = False
            quit_requested = False
            for k in keys:
                if k in ("q", "Q"):
                    quit_requested = True
                elif k in ("p", " "):
                    state["paused"] = not state["paused"]
                elif k == "r":
                    force_redraw = True
                elif k == "l":
                    state["show_log"] = not state["show_log"]
                elif k == "?":
                    state["help"] = not state["help"]
                elif k == "j" or k == curses.KEY_DOWN:
                    state["scroll"] += 1
                elif k == "k" or k == curses.KEY_UP:
                    state["scroll"] = max(0, state["scroll"] - 1)
                elif k == curses.KEY_NPAGE:
                    state["scroll"] += 10
                elif k == curses.KEY_PPAGE:
                    state["scroll"] = max(0, state["scroll"] - 10)
                elif state["help"]:
                    state["help"] = False  # any other key closes the help overlay

            if quit_requested:
                break

            if not args.no_rtt and rtt_pending_since is None and (now - last_rtt_probe) >= 5.0:
                _send_dsr_probe()
                rtt_pending_since = time.monotonic()
                last_rtt_probe = now
            elif rtt_pending_since is not None and (time.monotonic() - rtt_pending_since) > 2.0:
                rtt_pending_since = None  # the probe just timed out; try again later

            height, width = stdscr.getmaxyx()
            if height < MIN_LINES or width < MIN_COLS:
                stdscr.erase()
                for i, (text, attr) in enumerate(too_small_frame(width, height)):
                    _safe_addstr(stdscr, i, 0, text, _curses_attr(attr, color_ok, args.no_color))
                stdscr.refresh()
                time.sleep(0.1)
                continue

            if state["paused"] and not force_redraw:
                time.sleep(0.05)
                continue

            due = (now - last_draw_time) >= args.interval
            if not (due or force_redraw or keys):
                time.sleep(0.03)
                continue

            snapshot = hub.snapshot()
            history = hub.history()
            opts = {
                "paused": state["paused"],
                "show_log": state["show_log"],
                "help": state["help"],
                "ascii_only": not _supports_unicode(),
                "interval": args.interval,
                "pod_id": podenv.RUNPOD_POD_ID or "?",
                "gpu_name": _gpu_name_from_snapshot(snapshot),
                "build_id": build_id,
                "no_rtt": args.no_rtt,
            }
            lines = render_frame(snapshot, history, width, now, opts)
            frame_key = (tuple(lines), width, height, state["scroll"])
            if frame_key != last_drawn or force_redraw:
                state["scroll"] = _draw(stdscr, lines, state["scroll"], height, width, color_ok, args.no_color)
                last_drawn = frame_key
                last_draw_time = now
            else:
                time.sleep(0.03)
    finally:
        hub.stop()
        try:
            curses.curs_set(1)
        except curses.error:
            pass
    return 0


def _run_curses(args: argparse.Namespace) -> int:
    """Drive the live curses UI, but never let a terminal that can't
    support it (TERM=dumb, no terminfo entry, a `curs_set`/`initscr`
    failure, ...) crash the whole command. `curses.wrapper` itself raises
    straight through (it only guarantees `endwin()` runs on the way out,
    not that `initscr()` succeeds in the first place) - anything other
    than an interactive Ctrl-C falls back to one `--once`-style frame with
    an explanation, so `zl dash` over a truly minimal terminal still shows
    SOMETHING instead of a bare traceback or a silent non-zero exit.
    """
    try:
        return curses.wrapper(_curses_main, args)
    except KeyboardInterrupt:
        return 0
    except curses.error as exc:
        print(f"zl dash: live view unavailable in this terminal ({exc}); falling back to --once", file=sys.stderr)
        return _run_once(args)


# =============================================================================
# CLI entry point
# =============================================================================


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="zl dash", description="Live terminal telemetry dashboard for the pod.")
    p.add_argument("--interval", type=float, default=2.0, help="seconds between redraws (default 2)")
    p.add_argument("--once", action="store_true", help="collect once, print one plain-text frame, exit")
    p.add_argument("--json", action="store_true", help="print collect_snapshot() as JSON and exit")
    p.add_argument("--width", type=int, default=None, help="frame width for --once (default $COLUMNS or 100)")
    p.add_argument("--no-rtt", action="store_true", help="disable the terminal round-trip probe")
    p.add_argument("--no-color", action="store_true", help="disable ANSI/curses color")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv) if argv is not None else sys.argv[1:])

    once = args.once
    as_json = args.json
    if not once and not as_json:
        try:
            is_tty = sys.stdout.isatty()
        except (ValueError, AttributeError):
            is_tty = False
        if not is_tty:
            once = True  # brief: non-TTY stdout with neither flag behaves as --once

    if as_json:
        return _run_json(args)
    if once:
        return _run_once(args)

    try:
        return _run_curses(args)
    except Exception as exc:  # noqa: BLE001 - never let a live-dashboard crash dump a raw traceback into her SSH session
        # `_run_curses` already handles the common case (curses.error) by
        # falling back to --once itself; this is the last-resort backstop
        # for anything else a broken terminal can raise (e.g. TERM unset
        # entirely) - still fall back to a plain frame rather than just
        # printing an error and giving up, so the operator sees the data.
        print(f"zl dash: {type(exc).__name__}: {exc}; falling back to --once", file=sys.stderr)
        try:
            return _run_once(args)
        except Exception as exc2:  # noqa: BLE001 - truly last resort
            print(f"zl dash: --once fallback also failed: {type(exc2).__name__}: {exc2}", file=sys.stderr)
            return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
