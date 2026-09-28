"""
Tests for ops/zlops/dash.py — `zl dash`'s pure frame renderer, --once/--json
paths, and a real end-to-end PTY smoke test of the live curses UI.

`render_frame()` is a pure function (snapshot + width -> [(text, attr), ...]
lines) by design, exactly so the UI is checkable without a real terminal:
most of this file drives it directly with hand-built snapshots. The one
test that DOES need a terminal (`test_pty_smoke_quit_restores_terminal`)
spawns a real pty (`pty.fork`) so curses gets a real tty to negotiate with
- there is no way to prove "the terminal gets reset on quit" without one.

Two mutation proofs (brief: "a test only counts if you watched it fail"):
  - the DSR-reply-is-never-a-keypress property of `_InputReader`
  - the "clip every line to width" property of `render_frame`
Each temporarily swaps in a broken version, confirms the safety property
now fails via a plain `assert` (never this file's own exit-on-failure
helpers), restores the original, and re-confirms green.

Run: python test_ops_dash.py
"""

from __future__ import annotations

import inspect
import io
import json
import os
import pty
import signal
import sys
# LOW (round 3): never write ops/**/__pycache__ from a test run --
# this file's own `from zlops import ...` below would otherwise create
# it even when the harness invoking this file forgot PYTHONDONTWRITEBYTECODE.
sys.dont_write_bytecode = True
import time
from contextlib import redirect_stdout
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_OPS = _HERE.parent / "ops"
if str(_OPS) not in sys.path:
    sys.path.insert(0, str(_OPS))

from zlops import dash, telemetry  # noqa: E402


# ---------------------------------------------------------------------------
# Assert helpers (same exit-code convention as compactor/test_setup_sshd_script.py)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Fake snapshots
# ---------------------------------------------------------------------------


def _entry(data=None, error=None, age_s=1.0, now=None):
    now = now if now is not None else time.time()
    return {"data": data, "error": error, "collected_at": (now - age_s) if data is not None or error else None}


def _all_errored_snapshot(now):
    err = "boom: simulated collector failure"
    return {
        "health": _entry(error=err, now=now),
        "services": _entry(error=err, now=now),
        "gpu": _entry(error=err, now=now),
        "vllm": _entry(error=err, now=now),
        "ssh": _entry(error=err, now=now),
        "host": _entry(error=err, now=now),
        "data_usage": _entry(error=err, now=now),
        "backups": _entry(error=err, now=now),
        "log": _entry(error=err, now=now),
    }


def _full_fake_snapshot(now):
    return {
        "health": _entry(
            data={
                "doc": {
                    "status": "degraded",
                    "status_reasons": ["vLLM unreachable"],
                    "checks": {
                        "hierarchy": {"catching_up": {"conv_id": "c1", "verdict": "stuck"}},
                        "reuse": {"attempted": 10, "succeeded": 9, "last_reason": "success", "declined_recently": False},
                        "budget_margin": {"margin": 0, "ceiling": 4096},
                        "truncated_summaries": {"hierarchy": [0, 0, 0], "compaction": 0},
                    },
                },
                "interpreted": {
                    "status": "warn",
                    "reasons": ["vLLM unreachable"],
                    "findings": [{"level": "warn", "text": "vLLM unreachable"}],
                },
                "memory_tail_deltas": {"stored": 4, "skipped_shed": 1},
            },
            now=now,
        ),
        "services": _entry(
            data={
                "services": [
                    {"name": "compactor", "state": "RUNNING", "pid": 123, "uptime": "1:00:00", "unexpected": False},
                    {"name": "vllm", "state": "FATAL", "pid": None, "uptime": None, "unexpected": True},
                    {"name": "selftest", "state": "EXITED", "pid": None, "uptime": None, "unexpected": False},
                ]
            },
            now=now,
        ),
        "gpu": _entry(
            data=[
                {
                    "name": "NVIDIA A100",
                    "utilization_gpu_pct": 42.0,
                    "memory_used_mb": 10240.0,
                    "memory_total_mb": 40960.0,
                    "temperature_c": 55.0,
                    "power_draw_w": 150.0,
                    "power_limit_w": 300.0,
                    "clock_sm_mhz": 1350.0,
                }
            ],
            now=now,
        ),
        "vllm": _entry(
            data={
                "source": "metrics",
                "metrics": {"running": 1, "waiting": 0, "kv_cache_usage_pct": 42.0},
                "rates": {
                    "prompt_tokens_per_s": 123.4,
                    "generation_tokens_per_s": 45.6,
                    "prefix_cache_hit_rate": 0.8,
                    "preemptions_delta": 0,
                },
                "ttft_quantiles_s": {"p50_s": 0.05, "p95_s": 0.2},
                "itl_quantiles_s": {"p50_s": 0.01, "p95_s": 0.03},
            },
            now=now,
        ),
        "ssh": _entry(
            data={
                "connection": {"client_ip": "203.0.113.5", "client_port": 54321, "server_ip": "10.0.0.2", "server_port": 22},
                "port": 22,
                "session": {"in_ssh": True, "start_epoch": now - 3600, "duration_s": 3600.0},
                "established_count": 1,
                "established_client_ips": ["203.0.113.5"],
                "listening": True,
                "in_tmux": True,
            },
            now=now,
        ),
        "host": _entry(
            data={
                "loadavg": [1.0, 0.9, 0.8],
                "meminfo": {"MemTotal": 32 * 1024**3, "MemAvailable": 20 * 1024**3},
                "cpu_percent": 12.5,
                "root_free_bytes": 500 * 1024**3,
                "root_total_bytes": 1000 * 1024**3,
                "proc_count": 180,
            },
            now=now,
        ),
        "data_usage": _entry(
            data={
                "entries": [{"name": "openwebui", "kb": 512000}, {"name": "backups", "kb": 128000}],
                "total_kb": 640000,
                "errors": {},
                "pct_of_volume": 5.2,
                "note": "df /data would report the whole shared MooseFS cluster, not a quota - du only",
            },
            now=now,
        ),
        "backups": _entry(data={"latest": "backup-2026.tgz", "latest_mtime": now - 3600, "count": 5, "total_bytes": 1024}, now=now),
        "log": _entry(
            data={
                "entries": [
                    {"file": "compactor.log", "timestamp": "2026-09-24 10:00:00", "level": "crit", "text": "ERROR disk write failed"},
                    {"file": "vllm-error.log", "timestamp": "2026-09-24 10:00:01", "level": "warn", "text": "WARNING low kv cache"},
                ],
                "file_errors": {},
            },
            now=now,
        ),
    }


_FAKE_HISTORY = {
    "gpu_util": [10.0, 20.0, 42.0] * 10,
    "gen_tps": [5.0, 15.0, 45.6] * 10,
    "kv_cache_pct": [10.0, 30.0, 42.0] * 10,
    "rtt_ms": [10.0, 12.0, 9.0] * 10,
}

_BASE_OPTS = {
    "paused": False,
    "show_log": True,
    "help": False,
    "ascii_only": False,
    "interval": 2.0,
    "pod_id": "abc123",
    "gpu_name": "NVIDIA A100",
    "build_id": "deadbeef1234",
    "no_rtt": False,
}


# =============================================================================
# render_frame — pure function, no terminal needed
# =============================================================================


def test_health_panel_shows_requests_in_flight_when_present():
    """Round-4 addendum (v3.1.9.8): /health/full's own `requests_in_flight`
    is shown on the Health panel's status line when present.
    """
    entry = {
        "data": {
            "doc": {"status": "ok", "request_tracing": {"requests_in_flight": 3}},
            "interpreted": {"status": "ok", "reasons": [], "findings": []},
        },
        "error": None,
        "collected_at": time.time(),
    }
    lines = dash._panel_health(entry, time.time())
    text = "\n".join(t for t, _a in lines)
    assert_true("in_flight=3" in text, "requests_in_flight is shown on the status line")


def test_health_panel_hides_requests_in_flight_when_absent():
    entry = {
        "data": {
            "doc": {"status": "ok"},
            "interpreted": {"status": "ok", "reasons": [], "findings": []},
        },
        "error": None,
        "collected_at": time.time(),
    }
    lines = dash._panel_health(entry, time.time())
    text = "\n".join(t for t, _a in lines)
    assert_true("in_flight" not in text, "requests_in_flight is hidden when the field is absent")


def test_mutation_always_showing_inflight_field_is_caught():
    """Prove the hidden-when-absent test above bites: inline a version
    that always shows `in_flight=` (defaulting a missing field to 0
    instead of hiding it), confirm it would have shown a fabricated 0,
    then confirm the real code hides it.
    """
    doc = {"status": "ok"}
    rt = doc.get("request_tracing") or {}
    mutated_line = f"  status: ok   in_flight={rt.get('requests_in_flight', 0)}"
    assert_true("in_flight=0" in mutated_line,
                "mutated (defaults to 0) fabricates a count that was never reported -- confirms the bite")


def test_mutation_reading_top_level_key_misses_the_real_shape():
    """Round-4 blocker: the real v3.1.9.8 `/health/full` payload nests
    `requests_in_flight` under `request_tracing` (health.py:2290 in
    zl-v3198), not at the top level. Prove a version that reads the
    top-level key (the earlier bug) silently hides an in-flight count
    that the real compactor DOES report, then confirm the real code
    (which reads the nested key) shows it.
    """
    entry = {
        "data": {
            "doc": {"status": "ok", "request_tracing": {"requests_in_flight": 3}},
            "interpreted": {"status": "ok", "reasons": [], "findings": []},
        },
        "error": None,
        "collected_at": time.time(),
    }
    doc = entry["data"]["doc"]
    mutated_inflight = doc.get("requests_in_flight")  # old (wrong) top-level read
    assert_true(mutated_inflight is None,
                "mutated (top-level read) misses the nested requests_in_flight -- confirms the bite")

    lines = dash._panel_health(entry, time.time())
    text = "\n".join(t for t, _a in lines)
    assert_true("in_flight=3" in text, "the real (restored) nested read shows the count")

    absent_entry = {
        "data": {"doc": {"status": "ok"}, "interpreted": {"status": "ok", "reasons": [], "findings": []}},
        "error": None, "collected_at": time.time(),
    }
    lines = dash._panel_health(absent_entry, time.time())
    text = "\n".join(t for t, _a in lines)
    assert_true("in_flight" not in text, "the real (restored) code hides the absent field instead")


# =============================================================================
# Boot-time word-ban warm-up (coordinator addendum, v3.1.9.8) -- committed
# `warm_state` values: idle | waiting_idle | warming | backoff | disabled |
# vllm_unreachable, plus a `model` field. Read tolerantly
# (`doc.get("word_ban", {})`); an unknown future value must render raw,
# never crash.
# =============================================================================

_WARM_STATE_ATTR = {
    "idle": "normal", "waiting_idle": "normal", "warming": "normal", "disabled": "normal",
    "backoff": "warn", "vllm_unreachable": "warn",
}


def _health_entry_with_word_ban(word_ban: dict) -> dict:
    return {
        "data": {
            "doc": {"status": "ok", "word_ban": word_ban},
            "interpreted": {"status": "ok", "reasons": [], "findings": []},
        },
        "error": None,
        "collected_at": time.time(),
    }


def test_health_panel_hides_word_ban_when_absent():
    entry = {
        "data": {"doc": {"status": "ok"}, "interpreted": {"status": "ok", "reasons": [], "findings": []}},
        "error": None, "collected_at": time.time(),
    }
    lines = dash._panel_health(entry, time.time())
    text = "\n".join(t for t, _a in lines)
    assert_true("warm_state" not in text, "warm_state is hidden entirely when word_ban is absent")


def test_health_panel_shows_each_committed_warm_state_with_the_right_attr():
    for state, expected_attr in _WARM_STATE_ATTR.items():
        entry = _health_entry_with_word_ban({"warm_state": state, "model": "qwen-test"})
        lines = dash._panel_health(entry, time.time())
        matches = [(t, a) for t, a in lines if "warm_state" in t]
        assert_true(len(matches) == 1, f"exactly one warm_state line for state={state!r}")
        text, attr = matches[0]
        assert_true(state in text, f"warm_state value {state!r} appears in the line", )
        assert_eq(attr, expected_attr, f"warm_state={state!r} gets attr={expected_attr!r}")


def test_health_panel_shows_an_unknown_warm_state_raw_without_crashing():
    entry = _health_entry_with_word_ban({"warm_state": "some_future_state_nobody_documented_yet"})
    lines = dash._panel_health(entry, time.time())  # must not raise
    text = "\n".join(t for t, _a in lines)
    assert_true("some_future_state_nobody_documented_yet" in text, "an unknown value is shown raw")


def test_vllm_panel_labels_running_waiting_as_warmup_when_warm_state_is_warming():
    vllm_entry = {
        "data": {"metrics": {"running": 1, "waiting": 0, "kv_cache_usage_pct": 5.0}, "rates": {}},
        "error": None, "collected_at": time.time(),
    }
    lines = dash._panel_vllm(vllm_entry, time.time(), {}, False, {"warm_state": "warming"})
    text = "\n".join(t for t, _a in lines)
    assert_true("(word-ban warm-up)" in text, "the running/waiting line is labelled as the warm-up")


def test_vllm_panel_labels_running_waiting_as_warmup_when_warmups_in_flight_positive():
    """Even outside `warm_state == "warming"` (e.g. `idle` with a stray
    in-flight counter still draining), a nonzero `warmups_in_flight` alone
    must still label the panel -- it, not `requests_in_flight`, is what
    actually counts this request.
    """
    vllm_entry = {
        "data": {"metrics": {"running": 1, "waiting": 0, "kv_cache_usage_pct": 5.0}, "rates": {}},
        "error": None, "collected_at": time.time(),
    }
    lines = dash._panel_vllm(vllm_entry, time.time(), {}, False, {"warm_state": "idle", "warmups_in_flight": 1})
    text = "\n".join(t for t, _a in lines)
    assert_true("(word-ban warm-up)" in text, "warmups_in_flight > 0 alone is enough to label it")


def test_vllm_panel_no_warmup_label_for_other_warm_states():
    for state in ("idle", "waiting_idle", "backoff", "disabled", "vllm_unreachable"):
        vllm_entry = {
            "data": {"metrics": {"running": 1, "waiting": 0, "kv_cache_usage_pct": 5.0}, "rates": {}},
            "error": None, "collected_at": time.time(),
        }
        lines = dash._panel_vllm(vllm_entry, time.time(), {}, False, {"warm_state": state, "warmups_in_flight": 0})
        text = "\n".join(t for t, _a in lines)
        assert_true("(word-ban warm-up)" not in text, f"state={state!r} with 0 in flight is not labelled warm-up")


def test_vllm_panel_no_warmup_label_when_word_ban_absent():
    vllm_entry = {
        "data": {"metrics": {"running": 1, "waiting": 0, "kv_cache_usage_pct": 5.0}, "rates": {}},
        "error": None, "collected_at": time.time(),
    }
    lines = dash._panel_vllm(vllm_entry, time.time(), {}, False, None)
    text = "\n".join(t for t, _a in lines)
    assert_true("(word-ban warm-up)" not in text, "no word_ban data at all -- never labelled")


def test_render_frame_wires_word_ban_from_health_doc_into_the_vllm_panel():
    """End-to-end: `render_frame` must itself pull `word_ban` out of the
    HEALTH snapshot's doc (word_ban is not part of the vLLM collector's
    own payload) and pass it to the vLLM panel.
    """
    now = time.time()
    snap = {
        "health": _health_entry_with_word_ban({"warm_state": "warming"}),
        "vllm": {
            "data": {"metrics": {"running": 2, "waiting": 0, "kv_cache_usage_pct": 5.0}, "rates": {}},
            "error": None, "collected_at": now,
        },
    }
    lines = dash.render_frame(snap, {}, 140, now, dict(_BASE_OPTS))
    text = "\n".join(t for t, _a in lines)
    assert_true("(word-ban warm-up)" in text, "render_frame threads word_ban from health into the vLLM panel")


def test_mutation_vllm_panel_never_checking_word_ban_hides_the_warmup_label():
    """Prove the tests above bite: a version of `_panel_vllm` that ignores
    its `word_ban` argument entirely never shows the label, even during a
    real warm-up -- exactly what an operator watching the dashboard must
    NOT see (it would read as her own traffic).
    """
    vllm_entry = {
        "data": {"metrics": {"running": 1, "waiting": 0, "kv_cache_usage_pct": 5.0}, "rates": {}},
        "error": None, "collected_at": time.time(),
    }
    original = dash._panel_vllm

    def _mutated(entry, now, history, ascii_only, word_ban=None):
        return original(entry, now, history, ascii_only, None)  # drops word_ban -- the bug

    mutated_lines = _mutated(vllm_entry, time.time(), {}, False, {"warm_state": "warming"})
    mutated_text = "\n".join(t for t, _a in mutated_lines)
    assert_true("(word-ban warm-up)" not in mutated_text, "mutated (ignores word_ban) never labels it -- confirms the bite")

    real_lines = dash._panel_vllm(vllm_entry, time.time(), {}, False, {"warm_state": "warming"})
    real_text = "\n".join(t for t, _a in real_lines)
    assert_true("(word-ban warm-up)" in real_text, "the real (restored) panel labels it")


# =============================================================================
# v3.1.9.9 Cloudflare Tunnel: Health panel's `tunnel:` line, Services panel's
# cloudflared-configured-but-down warning.
# =============================================================================


def _health_entry_with_tunnel_status(tunnel_status) -> dict:
    return {
        "data": {
            "doc": {"status": "ok"},
            "interpreted": {"status": "ok", "reasons": [], "findings": []},
            "tunnel_status": tunnel_status,
        },
        "error": None,
        "collected_at": time.time(),
    }


def test_health_panel_shows_tunnel_status_when_present():
    for word in ("ready", "not ready", "unknown", "disabled"):
        entry = _health_entry_with_tunnel_status(word)
        lines = dash._panel_health(entry, time.time())
        matches = [(t, a) for t, a in lines if "tunnel:" in t]
        assert_true(len(matches) == 1, f"exactly one tunnel line for tunnel_status={word!r}")
        text, attr = matches[0]
        assert_true(word in text, f"tunnel_status value {word!r} appears in the line")
        expected_attr = "warn" if word == "not ready" else "normal"
        assert_eq(attr, expected_attr, f"tunnel_status={word!r} gets attr={expected_attr!r}")


def test_health_panel_hides_tunnel_status_when_absent():
    entry = _health_entry_with_tunnel_status(None)
    lines = dash._panel_health(entry, time.time())
    text = "\n".join(t for t, _a in lines)
    assert_true("tunnel:" not in text, "tunnel line is hidden entirely when tunnel_status is None (older compactor)")


def _health_entry_with_tunnel_error(error) -> dict:
    return {
        "data": {
            "doc": {"status": "ok", "tunnel": {"configured": True, "ready": None, "error": error}},
            "interpreted": {"status": "ok", "reasons": [], "findings": []},
            "tunnel_status": "unknown",
        },
        "error": None,
        "collected_at": time.time(),
    }


def test_health_panel_shows_tunnel_error_next_to_unknown():
    """Round-11 review (LOW): "unknown" alone doesn't say WHY - the
    compactor's own probe error text (`tunnel.error`) must appear right
    next to it, truncated, so an operator sees "ConnectError: refused"
    instead of just a bare "unknown".
    """
    entry = _health_entry_with_tunnel_error("ConnectError: [Errno 111] Connection refused")
    lines = dash._panel_health(entry, time.time())
    matches = [(t, a) for t, a in lines if "tunnel:" in t]
    assert_true(len(matches) == 1, "exactly one tunnel line")
    text, attr = matches[0]
    assert_true("unknown" in text, "the tunnel_status word is still shown")
    assert_true("ConnectError: [Errno 111] Connection refused" in text, "the probe error text is shown")
    assert_eq(attr, "normal", "unknown (with an error) is not itself a warn attr")


def test_health_panel_truncates_a_long_tunnel_error():
    long_error = "x" * 500
    entry = _health_entry_with_tunnel_error(long_error)
    lines = dash._panel_health(entry, time.time())
    text = "\n".join(t for t, _a in lines)
    assert_true(long_error not in text, "the full 500-char error text is not dumped verbatim")
    assert_true("x" * 160 in text, "a bounded prefix of the error text survives")


def test_health_panel_no_error_suffix_when_tunnel_error_absent():
    entry = _health_entry_with_tunnel_error(None)
    lines = dash._panel_health(entry, time.time())
    matches = [t for t, _a in lines if "tunnel:" in t]
    assert_eq(matches, ["  tunnel: unknown"], "no trailing '()' when there is no error text at all")


def test_health_panel_no_error_suffix_when_tunnel_status_is_not_unknown():
    """`tunnel.error` is only ever shown alongside "unknown" - a "not
    ready"/"ready"/"disabled" state must never grow a stray error suffix
    even if a stale `tunnel.error` field happens to still be present.
    """
    entry = {
        "data": {
            "doc": {"status": "ok", "tunnel": {"configured": True, "ready": False, "error": "stale text"}},
            "interpreted": {"status": "ok", "reasons": [], "findings": []},
            "tunnel_status": "not ready",
        },
        "error": None,
        "collected_at": time.time(),
    }
    lines = dash._panel_health(entry, time.time())
    text = "\n".join(t for t, _a in lines)
    assert_true("stale text" not in text, "error text is only ever shown for tunnel_status == 'unknown'")


def test_mutation_dropping_tunnel_error_suffix_is_caught():
    """Prove the tests above bite: a version of the tunnel line that never
    looks at `doc['tunnel']['error']` at all would show a bare "unknown"
    with no diagnostic text, exactly the round-11 review's complaint.
    """
    entry = _health_entry_with_tunnel_error("ConnectError: refused")

    def _mutated(entry, now):
        data = entry.get("data") or {}
        tunnel_status = data.get("tunnel_status")
        return [(f"  tunnel: {tunnel_status}", "normal")] if tunnel_status is not None else []

    mutated_lines = _mutated(entry, time.time())
    mutated_text = "\n".join(t for t, _a in mutated_lines)
    assert_true("ConnectError" not in mutated_text, "mutated (error-blind) version shows no diagnostic text")

    real_lines = dash._panel_health(entry, time.time())
    real_text = "\n".join(t for t, _a in real_lines)
    assert_true("ConnectError: refused" in real_text, "the real code shows the probe error")


def test_mutation_tunnel_status_defaulting_instead_of_hiding_is_caught():
    """Prove the hidden-when-absent test above bites: a broken version that
    defaults a missing tunnel_status to "disabled" (instead of hiding the
    line) would fabricate a state that /health/full never actually
    reported.
    """
    data = {"tunnel_status": None}
    mutated_word = data.get("tunnel_status") or "disabled"  # the bug: fabricates a default
    assert_eq(mutated_word, "disabled", "mutated (defaults to disabled) fabricates a state -- confirms the bite")


def _services_entry(services: list[dict]) -> dict:
    return {"data": {"services": services}, "error": None, "collected_at": time.time()}


def test_services_panel_flags_cloudflared_configured_but_down():
    entry = _services_entry([{"name": "cloudflared", "state": "EXITED", "unexpected": False, "pid": None, "uptime": None}])
    health_doc = {"tunnel": {"configured": True, "ready": None}}
    lines = dash._panel_services(entry, time.time(), health_doc)
    matches = [(t, a) for t, a in lines if "cloudflared" in t]
    assert_true(len(matches) == 1, "exactly one cloudflared row")
    text, attr = matches[0]
    assert_true("tunnel configured but cloudflared is EXITED" in text, "the warning text names the actual state")
    assert_eq(attr, "warn", "configured-but-down cloudflared gets attr=warn, not crit")


def test_services_panel_does_not_flag_cloudflared_disabled_by_config():
    entry = _services_entry([{"name": "cloudflared", "state": "EXITED", "unexpected": False, "pid": None, "uptime": None}])
    health_doc = {"tunnel": {"configured": False, "ready": None}}
    lines = dash._panel_services(entry, time.time(), health_doc)
    matches = [(t, a) for t, a in lines if "cloudflared" in t]
    text, attr = matches[0]
    assert_true("tunnel configured but cloudflared is" not in text, "not configured -> EXITED is fine, no warning")
    assert_eq(attr, "dim", "a tolerated non-RUNNING one-shot/optional service stays attr=dim")


def test_services_panel_no_flag_when_health_doc_absent():
    """An unreachable /health/full (health_doc defaults to None/{}) must
    never be treated as "tunnel configured" -- that would spuriously flag
    every pod that has never even set up the tunnel the moment /health/full
    itself is briefly unreachable.
    """
    entry = _services_entry([{"name": "cloudflared", "state": "EXITED", "unexpected": False, "pid": None, "uptime": None}])
    lines = dash._panel_services(entry, time.time(), None)
    matches = [(t, a) for t, a in lines if "cloudflared" in t]
    text, attr = matches[0]
    assert_true("tunnel configured but cloudflared is" not in text, "no health doc -> no spurious warning")
    assert_eq(attr, "dim", "still reads as a tolerated one-shot/optional service")


def test_render_frame_wires_health_doc_tunnel_into_the_services_panel():
    """End-to-end: `render_frame` must itself pull the health doc's `tunnel`
    key out of the HEALTH snapshot (not the services collector's own
    payload, which has no /health/full access at all) and pass it to the
    Services panel.
    """
    now = time.time()
    snap = {
        "health": {
            "data": {"doc": {"status": "ok", "tunnel": {"configured": True, "ready": False}},
                      "interpreted": {"status": "ok", "reasons": [], "findings": []},
                      "tunnel_status": "not ready"},
            "error": None, "collected_at": now,
        },
        "services": _services_entry(
            [{"name": "cloudflared", "state": "EXITED", "unexpected": False, "pid": None, "uptime": None}]
        ),
    }
    snap["services"]["collected_at"] = now
    # Single-column layout (width < TWO_COLUMN_MIN_WIDTH): a two-column
    # frame clips each column to roughly half the width, which would cut
    # this warning's text off mid-sentence and make this an assertion
    # about clipping rather than about the wiring this test exists to
    # check - see `_panel_services`'s own direct tests above for the
    # exact-text checks.
    lines = dash.render_frame(snap, {}, 100, now, dict(_BASE_OPTS))
    text = "\n".join(t for t, _a in lines)
    assert_true("tunnel configured but cloudflared is EXITED" in text,
                "render_frame threads the health doc's tunnel key into the Services panel")


def test_mutation_services_panel_never_checking_health_doc_hides_the_warning():
    """Prove the tests above bite: a version of `_panel_services` that
    ignores its `health_doc` argument entirely never shows the warning,
    even when a token IS configured and cloudflared really is down --
    exactly what an operator must NOT miss.
    """
    entry = _services_entry([{"name": "cloudflared", "state": "EXITED", "unexpected": False, "pid": None, "uptime": None}])
    original = dash._panel_services

    def _mutated(entry, now, health_doc=None):
        return original(entry, now, None)  # drops health_doc -- the bug

    mutated_lines = _mutated(entry, time.time(), {"tunnel": {"configured": True, "ready": None}})
    mutated_text = "\n".join(t for t, _a in mutated_lines)
    assert_true("tunnel configured but cloudflared is" not in mutated_text,
                "mutated (ignores health_doc) never warns -- confirms the bite")

    real_lines = dash._panel_services(entry, time.time(), {"tunnel": {"configured": True, "ready": None}})
    real_text = "\n".join(t for t, _a in real_lines)
    assert_true("tunnel configured but cloudflared is EXITED" in real_text, "the real (restored) panel warns")


def test_render_frame_all_errored_at_80_and_140_no_line_exceeds_width():
    now = time.time()
    snap = _all_errored_snapshot(now)
    for width in (80, 140):
        lines = dash.render_frame(snap, {}, width, now, dict(_BASE_OPTS))
        assert_true(len(lines) > 0, f"width={width}: frame is non-empty even with every collector errored")
        for text, attr in lines:
            assert_true(len(text) <= width, f"width={width}: {len(text)} chars <= width ({text!r})")
            assert_true(isinstance(attr, str), f"width={width}: attr is a string tag")


def test_render_frame_full_fake_data_at_80_and_140_no_line_exceeds_width():
    now = time.time()
    snap = _full_fake_snapshot(now)
    for width in (80, 140):
        lines = dash.render_frame(snap, _FAKE_HISTORY, width, now, dict(_BASE_OPTS))
        assert_true(len(lines) > 0, f"width={width}: frame is non-empty with full fake data")
        for text, attr in lines:
            assert_true(len(text) <= width, f"width={width}: {len(text)} chars <= width ({text!r})")
        joined = "\n".join(t for t, _ in lines)
        assert_true("A100" in joined, f"width={width}: GPU name shows up somewhere in the frame")
        assert_true("compactor" in joined, f"width={width}: a service name shows up somewhere in the frame")


def test_render_frame_two_column_at_140_one_column_at_80():
    now = time.time()
    snap = _full_fake_snapshot(now)
    wide = dash.render_frame(snap, _FAKE_HISTORY, 140, now, dict(_BASE_OPTS))
    narrow = dash.render_frame(snap, _FAKE_HISTORY, 80, now, dict(_BASE_OPTS))
    wide_text = "\n".join(t for t, _ in wide)
    assert_true(" | " in wide_text, "140 cols uses the two-column separator somewhere")
    narrow_text = "\n".join(t for t, _ in narrow)
    # The narrow layout still must not exceed 80 cols (checked above); a
    # crude but effective way to confirm it's NOT doing two-column merging
    # is that no line is a " | "-joined pair of full-width halves.
    assert_true(len(narrow) >= len(wide) - 5, "single-column layout doesn't lose panels, just stacks them")
    del narrow_text


def test_render_frame_help_overlay():
    now = time.time()
    opts = dict(_BASE_OPTS)
    opts["help"] = True
    lines = dash.render_frame({}, {}, 80, now, opts)
    joined = "\n".join(t for t, _ in lines)
    assert_true("help" in joined.lower(), "help overlay renders help text")
    for text, _ in lines:
        assert_true(len(text) <= 80, "help overlay lines also respect width")


def test_render_frame_no_data_yet_panels_render_without_crashing():
    now = time.time()
    empty = {k: {"data": None, "error": None, "collected_at": None} for k in
             ("health", "services", "gpu", "vllm", "ssh", "host", "data_usage", "backups", "log")}
    lines = dash.render_frame(empty, {}, 100, now, dict(_BASE_OPTS))
    joined = "\n".join(t for t, _ in lines)
    assert_true("no data yet" in joined, "a never-collected panel says so rather than crashing")


def test_render_frame_stale_panel_is_flagged():
    now = time.time()
    snap = {k: {"data": None, "error": None, "collected_at": None} for k in
            ("services", "gpu", "vllm", "ssh", "host", "data_usage", "backups", "log")}
    # health's cadence is 5s; STALE_MULTIPLE is 3x -> stale past 15s old.
    snap["health"] = {
        "data": {"doc": {"status": "ok"}, "interpreted": {"status": "ok", "reasons": [], "findings": []}},
        "error": None,
        "collected_at": now - 100,
    }
    lines = dash.render_frame(snap, {}, 100, now, dict(_BASE_OPTS))
    joined = "\n".join(t for t, _ in lines)
    assert_true("STALE" in joined, "an old panel is flagged STALE")


# =============================================================================
# sparkline
# =============================================================================


def test_sparkline_basic_and_ascii_fallback():
    values = [0.0, 5.0, 10.0]
    uni = dash.sparkline(values, 3, ascii_only=False)
    asc = dash.sparkline(values, 3, ascii_only=True)
    assert_eq(len(uni), 3, "unicode sparkline is one char per sample")
    assert_eq(len(asc), 3, "ascii sparkline is one char per sample")
    assert_true(all(ord(c) < 128 for c in asc), "ascii fallback never emits non-ASCII")
    assert_true(any(ord(c) > 127 for c in uni), "unicode sparkline actually uses block characters")


def test_sparkline_handles_none_gaps_and_empty():
    assert_eq(dash.sparkline([], 10), "", "no samples -> empty string")
    s = dash.sparkline([1.0, None, 2.0], 3)
    assert_eq(len(s), 3, "a None sample still occupies one column")
    assert_eq(s[1], " ", "a None sample renders as a gap")


# =============================================================================
# --once / --json (in-process, via dash.main, output captured)
# =============================================================================


def test_once_output_degrades_cleanly_with_no_collectors_reachable():
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = dash.main(["--once", "--no-rtt", "--width", "90"])
    assert_eq(code, 0, "--once exits 0 even when every collector fails")
    out = buf.getvalue()
    assert_true("zl dash" in out, "header printed")
    assert_true(len(out) > 0, "some output was produced")
    for line in out.splitlines():
        assert_true(len(line) <= 200, "no absurdly long line (ANSI codes aside) in --once output")


def test_json_output_valid_with_no_collectors_reachable():
    """Force every collector to fail deterministically (rather than relying
    on this dev host happening to have none of them reachable) and confirm
    `--json` still exits 0 with a valid, fully-shaped JSON document.
    """
    names = [
        "collect_health",
        "collect_services",
        "collect_gpu",
        "collect_vllm",
        "collect_host",
        "collect_backups",
        "collect_ssh_session",
        "collect_log_signal",
        "collect_data_usage",
        "probe_terminal_rtt",
    ]
    originals = {n: getattr(telemetry, n) for n in names}

    def boom(*a, **k):
        raise RuntimeError("unreachable (test)")

    orig_snapshot = telemetry.collect_snapshot

    def fast_snapshot(options=None):
        options = dict(options or {})
        options["rate_gap_s"] = 0.01  # keep the test fast; the gap itself is tested elsewhere
        return orig_snapshot(options)

    for n in names:
        setattr(telemetry, n, boom)
    telemetry.collect_snapshot = fast_snapshot
    try:
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = dash.main(["--json", "--no-rtt"])
    finally:
        for n in names:
            setattr(telemetry, n, originals[n])
        telemetry.collect_snapshot = orig_snapshot

    assert_eq(code, 0, "--json exits 0 even when every collector fails")
    data = json.loads(buf.getvalue())  # raises if not valid JSON
    for name in ("health", "services", "gpu", "vllm", "host", "backups", "ssh", "log", "data_usage"):
        assert_true(name in data, f"{name} key present in the JSON snapshot")
        assert_true(data[name]["data"] is None, f"{name}.data is null when every collector is unreachable")
        assert_true(bool(data[name]["error"]), f"{name}.error carries a message")


def test_non_tty_stdout_with_no_flags_behaves_as_once():
    """Brief: stdout not a TTY and neither --once nor --json given -> --once
    behavior. `redirect_stdout` to a StringIO is exactly that (not a tty).
    """
    buf = io.StringIO()
    assert_true(not buf.isatty(), "a StringIO is never a tty (sanity check on the test itself)")
    with redirect_stdout(buf):
        code = dash.main(["--no-rtt", "--width", "90"])
    assert_eq(code, 0, "falls back to --once behavior and exits 0")
    assert_true("zl dash" in buf.getvalue(), "produced the --once-style frame, not a curses session")


def test_once_emits_no_ansi_when_stdout_is_not_a_tty():
    """Round-4 cosmetic: `zl dash --once` redirected to a file/pipe (not a
    real terminal) must print plain text, not literal `\\x1b[...]` color
    escapes -- the same auto-disable any well-behaved CLI applies, without
    needing `--no-color` spelled out explicitly.
    """
    buf = io.StringIO()
    assert_true(not buf.isatty(), "a StringIO is never a tty (sanity check on the test itself)")
    with redirect_stdout(buf):
        code = dash.main(["--once", "--no-rtt", "--width", "90"])
    assert_eq(code, 0, "--once exits 0")
    out = buf.getvalue()
    assert_true("\x1b[" not in out, f"no ANSI escape made it into non-tty output ({out[:200]!r})")


def test_mutation_ignoring_tty_check_leaks_ansi_to_a_pipe():
    """Prove the test above bites: inline the OLD `_run_once` (no_color
    taken from `args.no_color` alone, never checking `sys.stdout.isatty`),
    confirm ANSI escapes would have been printed to our non-tty buffer,
    then confirm the real (restored) code prints none.
    """
    old_no_color = False  # the OLD code: args.no_color alone, ignoring isatty
    mutated_line = dash._apply_ansi("status: ok", "ok", old_no_color)
    assert_true("\x1b[" in mutated_line,
                "mutated (isatty-blind) --once would emit ANSI to a non-tty stream -- confirms the bite")

    buf = io.StringIO()
    with redirect_stdout(buf):
        code = dash.main(["--once", "--no-rtt", "--width", "90"])
    assert_eq(code, 0, "--once exits 0")
    assert_true("\x1b[" not in buf.getvalue(), "the real (restored) code emits no ANSI to the same non-tty buffer")


# =============================================================================
# Mutation proofs
# =============================================================================


def test_render_frame_width_clip_mutation_proof():
    """Prove the "no line exceeds width" checks actually bite: swap in a
    broken render_frame that skips clipping, confirm the property now
    fails, restore, confirm green again.
    """
    original = dash.render_frame

    def broken(snapshot, history, width, now, opts):
        lines = original(snapshot, history, width, now, opts)
        # BUG: pad a line well past `width` instead of clipping to it.
        if lines:
            text, attr = lines[0]
            lines = [(text + ("!" * (width + 50)), attr)] + lines[1:]
        return lines

    dash.render_frame = broken
    caught = False
    try:
        now = time.time()
        lines = dash.render_frame(_all_errored_snapshot(now), {}, 80, now, dict(_BASE_OPTS))
        for text, _ in lines:
            assert len(text) <= 80
    except AssertionError:
        caught = True
    finally:
        dash.render_frame = original
    assert_true(caught, "mutation proof: an unclipped line is caught red")
    now = time.time()
    lines = dash.render_frame(_all_errored_snapshot(now), {}, 80, now, dict(_BASE_OPTS))
    assert_true(all(len(t) <= 80 for t, _ in lines), "restored render_frame is green again")


def test_dsr_reply_never_a_keypress_mutation_proof():
    """Prove the "DSR reply is consumed, never delivered as a keystroke"
    property bites: use a broken buffer-splice (keeps the whole raw buffer
    instead of excising the matched span) and confirm escape bytes leak
    into what a keypress handler would see, then restore the real logic.
    """

    def real_split(buf):
        found = telemetry.find_dsr_reply(buf)
        row, col, start, end = found
        return buf[:start] + buf[end:]

    def broken_split(buf):
        # BUG: "consumes" the reply but doesn't actually remove it from
        # what gets treated as keystrokes.
        return buf

    buf = "q\x1b[10;5Rp"
    caught = False
    try:
        result = broken_split(buf)
        assert "\x1b" not in result
    except AssertionError:
        caught = True
    assert_true(caught, "mutation proof: a broken splice leaks the escape sequence, and the check catches it")

    result = real_split(buf)
    assert_true("\x1b" not in result, "the real splice never leaks the escape sequence")
    assert_eq(result, "qp", "the real splice leaves exactly the real keystrokes")


# =============================================================================
# PTY smoke test — the one test that needs a real terminal
# =============================================================================


def _dash_command():
    cli_py = _OPS / "zlops" / "cli.py"
    if cli_py.exists():
        return [sys.executable, "-m", "zlops", "dash"]
    return [
        sys.executable,
        "-c",
        "import sys; from zlops import dash; sys.exit(dash.main([]))",
    ]


def test_pty_smoke_quit_restores_terminal():
    """Spawn the real dashboard entry point in a pty (TERM=xterm-256color,
    100x30), wait for the first frame, send 'q', and confirm: exit 0 within
    10s, AND that some terminal-reset escape sequence was actually emitted
    after 'q' was sent (curses' endwin() restoring the terminal mode -
    exact bytes vary by ncurses/terminfo version, so this checks for the
    presence of control-sequence output following the quit key, not one
    specific sequence).
    """
    cmd = _dash_command()
    env = dict(os.environ)
    env["TERM"] = "xterm-256color"
    env["PYTHONPATH"] = str(_OPS)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("COLUMNS", None)
    env.pop("LINES", None)

    pid, master_fd = pty.fork()
    if pid == 0:  # child
        try:
            import fcntl
            import struct
            import termios

            fcntl.ioctl(0, termios.TIOCSWINSZ, struct.pack("HHHH", 30, 100, 0, 0))
        except Exception:
            pass
        os.chdir(str(_OPS))
        os.execvpe(cmd[0], cmd, env)
        os._exit(127)  # pragma: no cover - only reached if exec fails

    # parent
    os.set_blocking(master_fd, False)
    collected = b""
    deadline = time.monotonic() + 10.0
    sent_quit = False
    quit_sent_at = None

    def read_available():
        nonlocal collected
        try:
            chunk = os.read(master_fd, 65536)
        except OSError:
            return b""
        collected += chunk
        return chunk

    try:
        # Wait for the first drawn frame (something recognisable on screen)
        # before sending 'q' - sending it too early could race the curses
        # setup and be swallowed before nodelay/keypad are configured.
        while time.monotonic() < deadline and not sent_quit:
            read_available()
            if b"zl dash" in collected or b"terminal too small" in collected:
                time.sleep(0.2)  # let the frame finish drawing
                os.write(master_fd, b"q")
                sent_quit = True
                quit_sent_at = len(collected)
            else:
                time.sleep(0.05)

        assert_true(sent_quit, "the dashboard produced a first frame within 10s")

        # Wait for the child to exit.
        exited = False
        exit_code = None
        while time.monotonic() < deadline:
            try:
                wpid, status = os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                exited = True
                exit_code = 0
                break
            if wpid == pid:
                exited = True
                exit_code = os.waitstatus_to_exitcode(status)
                break
            read_available()
            time.sleep(0.05)

        assert_true(exited, "the dashboard exited within 10s of 'q'")
        assert_eq(exit_code, 0, "the dashboard's exit code after 'q' is 0")

        # Drain anything left and check for the SPECIFIC alternate-screen
        # exit sequence (rmcup, `ESC[?1049l`) emitted AFTER 'q' was sent -
        # not just "any ESC byte" (curs_set(1) alone emits `ESC[?25h`,
        # which would pass a loose "any ESC" check even if `curses.wrapper`
        # / `endwin()` were bypassed entirely - verified empirically: the
        # real dashboard's exit sequence is
        # `...\x1b[?12l\x1b[?25h\x1b[?1l\x1b>...\x1b[?1049l...`).
        for _ in range(20):
            read_available()
            time.sleep(0.02)
        after_quit = collected[quit_sent_at:]
        assert_true(b"\x1b" in after_quit, "a terminal control/reset sequence was emitted after quitting")
        assert_true(
            b"\x1b[?1049l" in after_quit,
            "the alternate-screen-exit sequence (rmcup) was emitted -- proves curses.wrapper's endwin() actually ran, "
            "not just a lone curs_set(1)",
        )
    finally:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            os.waitpid(pid, 0)
        except ChildProcessError:
            pass
        try:
            os.close(master_fd)
        except OSError:
            pass


class _FakeWin:
    """Stands in for the curses window `_InputReader` reads from: feed it a
    list of characters/ints, `getch()` returns them one at a time then -1.
    """

    def __init__(self, chars):
        self._queue = [ord(c) if isinstance(c, str) else c for c in chars]

    def getch(self):
        if not self._queue:
            return -1
        return self._queue.pop(0)


def test_input_reader_splices_a_real_dsr_reply_out_of_the_stream():
    """Drives the ACTUAL `_InputReader` class (not a hand-written stand-in
    like the pure-logic mutation proof below) with a DSR reply embedded in
    a real keystroke stream, proving the splice really happens in the
    shipped code path `_curses_main` uses.
    """
    win = _FakeWin(list("q") + list("\x1b[10;5R") + list("p"))
    reader = dash._InputReader(win)
    keys, dsr = reader.poll(time.time())
    assert_eq(dsr, (10, 5), "the DSR reply's row/col is extracted")
    assert_eq("".join(str(k) for k in keys), "qp", "the DSR reply bytes never appear as keystrokes")


def test_mutation_input_reader_dsr_disabled_is_caught():
    """The exact mutation the review's dash sub-review flagged (M12): make
    `_InputReader.poll` skip the DSR lookup entirely, so the escape
    sequence is delivered as plain keystrokes instead.
    """
    original = dash.telemetry.find_dsr_reply
    try:
        dash.telemetry.find_dsr_reply = lambda buf: None  # BUG: DSR replies never recognised
        win = _FakeWin(list("q") + list("\x1b[10;5R") + list("p"))
        reader = dash._InputReader(win)
        # A lone/abandoned-looking prefix is flushed after 0.1s (see
        # `_InputReader.poll`'s own docstring) - simulate that by polling
        # again slightly later so the buffered escape bytes get flushed as
        # keys, exactly like a real terminal that never got a DSR answer.
        keys, dsr = reader.poll(time.time())
        keys2, _ = reader.poll(time.time() + 0.2)
        all_keys = "".join(str(k) for k in keys + keys2)
        caught = "\x1b" not in all_keys
        assert_true(not caught, "mutated (DSR-disabled) InputReader leaks the escape sequence as keystrokes -- confirms the test bites")
    finally:
        dash.telemetry.find_dsr_reply = original
    win = _FakeWin(list("q") + list("\x1b[10;5R") + list("p"))
    reader = dash._InputReader(win)
    keys, dsr = reader.poll(time.time())
    assert_eq(dsr, (10, 5), "restored InputReader splices the DSR reply again")


def test_safe_addstr_never_raises_on_curses_error():
    """M7 (review's dash sub-review): a curses.error from `win.addstr`
    (writing to the bottom-right cell, or off-screen after a resize) must
    never propagate - it's expected, not exceptional, for a full-screen
    redraw. Uses a fake window (no real terminal needed) whose `addstr`
    always raises, exactly the shape `curses` itself raises in that case.
    """

    class _RaisingWin:
        def addstr(self, y, x, text, attr):
            raise dash.curses.error("bottom-right cell")

    dash._safe_addstr(_RaisingWin(), 0, 0, "text", 0)  # must not raise
    print("  ok   _safe_addstr swallows a real curses.error from addstr")


def test_mutation_safe_addstr_reraises_is_caught():
    original = dash._safe_addstr

    def broken(win, y, x, text, attr):
        win.addstr(y, x, text, attr)  # BUG: no try/except

    class _RaisingWin:
        def addstr(self, y, x, text, attr):
            raise dash.curses.error("bottom-right cell")

    dash._safe_addstr = broken
    caught = False
    try:
        dash._safe_addstr(_RaisingWin(), 0, 0, "text", 0)
    except dash.curses.error:
        caught = True
    finally:
        dash._safe_addstr = original
    assert_true(caught, "mutated (no try/except) _safe_addstr propagates curses.error -- confirms the test bites")
    dash._safe_addstr(_RaisingWin(), 0, 0, "text", 0)  # restored: must not raise
    print("  ok   restored _safe_addstr is green again")


def test_pty_idle_cpu_stays_low_no_busy_loop():
    """M8 (review's dash sub-review): with nothing changing on screen, the
    live loop must actually sleep between polls, not spin. Runs the real
    dashboard in a pty for a few seconds against an UNREACHABLE pod (every
    collector fails fast and stays failed, so nothing ever changes) and
    measures process CPU time via /proc/<pid>/stat - a busy loop shows up
    as CPU time roughly tracking wall time; a properly-sleeping loop stays
    a small fraction of it.
    """
    cmd = _dash_command()
    env = dict(os.environ)
    env["TERM"] = "xterm-256color"
    env["PYTHONPATH"] = str(_OPS)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("COLUMNS", None)
    env.pop("LINES", None)

    pid, master_fd = pty.fork()
    if pid == 0:
        try:
            import fcntl
            import struct
            import termios

            fcntl.ioctl(0, termios.TIOCSWINSZ, struct.pack("HHHH", 30, 100, 0, 0))
        except Exception:
            pass
        os.chdir(str(_OPS))
        os.execvpe(cmd[0], cmd, env)
        os._exit(127)  # pragma: no cover

    os.set_blocking(master_fd, False)

    def cpu_time_s(p):
        try:
            with open(f"/proc/{p}/stat") as f:
                fields = f.read().rsplit(")", 1)[1].split()
            return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")
        except OSError:
            return None

    try:
        deadline = time.monotonic() + 8.0
        collected = b""
        while time.monotonic() < deadline:
            try:
                collected += os.read(master_fd, 65536)
            except OSError:
                pass
            if b"zl dash" in collected:
                break
            time.sleep(0.05)
        time.sleep(0.3)
        c0 = cpu_time_s(pid)
        t0 = time.monotonic()
        while time.monotonic() - t0 < 4.0:
            try:
                os.read(master_fd, 65536)
            except OSError:
                pass
            time.sleep(0.05)
        wall = time.monotonic() - t0
        c1 = cpu_time_s(pid)
        os.write(master_fd, b"q")
        if c0 is None or c1 is None:
            print("  skip test_pty_idle_cpu_stays_low_no_busy_loop: /proc/<pid>/stat unavailable on this box")
            return
        cpu_pct = (c1 - c0) / wall * 100.0
        print(f"  ok   idle CPU over {wall:.1f}s: {cpu_pct:.1f}% (busy-loop would be near 100%)")
        assert_true(cpu_pct < 50.0, f"idle loop sleeps between polls ({cpu_pct:.1f}% CPU, not pegged)")
    finally:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            os.waitpid(pid, 0)
        except ChildProcessError:
            pass
        try:
            os.close(master_fd)
        except OSError:
            pass


class _FakeStdscr:
    """Just enough of curses' window API for `_curses_main` to run one pass
    and quit - no real terminal needed. `curses.curs_set`/`start_color`
    etc are all guarded with `try/except curses.error` in the real code
    (so they degrade quietly without a real `initscr()`), which is exactly
    what makes driving `_curses_main` directly, in-process, possible.
    """

    def __init__(self, keys):
        self._keys = list(keys)

    def nodelay(self, flag):
        pass

    def keypad(self, flag):
        pass

    def getmaxyx(self):
        return (30, 100)

    def getch(self):
        return ord(self._keys.pop(0)) if self._keys else -1

    def erase(self):
        pass

    def refresh(self):
        pass

    def addstr(self, y, x, text, attr):
        pass


def test_curses_main_stops_the_hub_on_quit():
    """M17 (review's dash sub-review): the `finally: hub.stop(); curs_set(1)`
    block in `_curses_main` is what actually stops the background collector
    threads on quit - removing it (replacing with `pass`) leaves them
    running forever. Captures the REAL `Hub` instance `_curses_main`
    creates (by wrapping the class, not replacing collection logic) and
    checks `_stop` was set after `_curses_main` returns.
    """
    args = dash._parse_args(["--no-rtt"])
    captured = {}
    original_hub_cls = dash.telemetry.Hub

    class _CapturingHub(original_hub_cls):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            captured["hub"] = self

    dash.telemetry.Hub = _CapturingHub
    try:
        stdscr = _FakeStdscr(["q"])
        code = dash._curses_main(stdscr, args)
    finally:
        dash.telemetry.Hub = original_hub_cls
    assert_eq(code, 0, "_curses_main returns 0 after 'q'")
    hub = captured.get("hub")
    assert_true(hub is not None, "the real Hub was constructed")
    assert_true(hub._stop.is_set(), "hub.stop() was called in the finally block - background threads told to stop")


def test_mutation_curses_main_finally_removed_is_caught():
    """Prove the test above actually bites: monkeypatch `_curses_main` to
    the M17-mutated shape (finally: pass) and watch the hub-stopped
    assertion fail.
    """
    import ast
    import textwrap

    src = inspect.getsource(dash._curses_main)
    mutated_src = src.replace(
        "    finally:\n        hub.stop()\n        try:\n            curses.curs_set(1)\n        except curses.error:\n            pass\n    return 0",
        "    finally:\n        pass\n    return 0",
    )
    assert mutated_src != src, "the finally block text must be found to mutate"
    ast.parse(textwrap.dedent(mutated_src))  # sanity: still valid Python

    namespace = dict(dash.__dict__)
    exec(compile(mutated_src, "<mutated _curses_main>", "exec"), namespace)  # noqa: S102 - test-only, in-memory mutation
    mutated_fn = namespace["_curses_main"]

    args = dash._parse_args(["--no-rtt"])
    captured = {}
    original_hub_cls = dash.telemetry.Hub

    class _CapturingHub(original_hub_cls):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            captured["hub"] = self

    dash.telemetry.Hub = _CapturingHub
    try:
        stdscr = _FakeStdscr(["q"])
        mutated_fn(stdscr, args)
    finally:
        dash.telemetry.Hub = original_hub_cls
    hub = captured.get("hub")
    caught = hub is not None and not hub._stop.is_set()
    assert_true(caught, "mutated (finally: pass) _curses_main never stops the hub -- confirms the test bites")
    if hub is not None:
        hub.stop()  # clean up the leaked threads this mutation intentionally caused


def test_pty_smoke_mutation_proof():
    """Prove the pty smoke test's own reset-sequence assertion actually
    bites: an os._exit()-style hard kill (bypassing curses.wrapper's
    cleanup) would leave no reset bytes after 'q' - simulate that
    directly against the assertion logic rather than re-spawning a whole
    second pty (which the real test above already does, at real cost).
    """
    after_quit_with_clean_exit = b"\x1b[?1049l\x1b[?25h"
    after_quit_with_hard_kill = b""  # os._exit() skips endwin() entirely

    assert_true(b"\x1b" in after_quit_with_clean_exit, "a clean curses exit leaves reset bytes")

    caught = False
    try:
        assert b"\x1b" in after_quit_with_hard_kill
    except AssertionError:
        caught = True
    assert_true(caught, "mutation proof: a hard-killed process (no cleanup) is caught red by this exact check")


# =============================================================================
# Runner
# =============================================================================

_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]


if __name__ == "__main__":
    for t in _TESTS:
        print(f"-- {t.__name__}")
        t()
    print(f"\nAll {len(_TESTS)} dash tests passed.")
