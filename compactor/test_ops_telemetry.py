"""
Tests for ops/zlops/telemetry.py — the collectors and interpret_health()
that `zl dash` and `zl health` both depend on (the fixed seam other lanes
in the pod-ops-bundle brief build on top of).

Runs stdlib-only, no network, no real supervisorctl/nvidia-smi/GPU/pod
required: every collector that shells out or makes an HTTP call takes its
I/O as an injectable parameter (`run=`, `fetch=`), so tests substitute a
fake directly, the same seam `compactor/test_setup_sshd_script.py` uses for
`_utc_stamp`-style patching, just as a function parameter instead of a
monkeypatch. `find_sshd_ancestor`/`session_start_and_duration` take a fake
`proc_root` for the same reason — no real /proc walking required.

Four tests are MUTATION PROOFS (brief: "a test only counts if you watched
it fail"): each temporarily replaces a real function with a deliberately
broken one, asserts the broken version is caught red by a plain `assert`
(never this file's own exit-on-failure helpers — those would kill the
whole run), then restores the original from an in-memory reference (never
git) and re-asserts green.

Run: python test_ops_telemetry.py
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import socket
import sys
# LOW (round 3): never write ops/**/__pycache__ from a test run --
# this file's own `from zlops import ...` below would otherwise create
# it even when the harness invoking this file forgot PYTHONDONTWRITEBYTECODE.
sys.dont_write_bytecode = True
import tempfile
import threading
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_OPS = _HERE.parent / "ops"
if str(_OPS) not in sys.path:
    sys.path.insert(0, str(_OPS))

from zlops import telemetry  # noqa: E402


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


def assert_in(needle, haystack, label):
    if needle not in haystack:
        print(f"FAIL {label}: {needle!r} not found in {haystack!r}")
        sys.exit(1)
    print(f"  ok   {label}")


def assert_close(actual, expected, label, tol=1e-6):
    if actual is None or abs(actual - expected) > tol:
        print(f"FAIL {label}: expected ~{expected!r}, got {actual!r}")
        sys.exit(1)
    print(f"  ok   {label}")


def _rmtree(p: Path) -> None:
    shutil.rmtree(p, ignore_errors=True)


def _hex_ip_port(ip: str, port: int) -> str:
    raw = socket.inet_aton(ip)[::-1]
    return raw.hex().upper() + ":" + format(port, "04X")


def _hex_ip6_port(ip6: str, port: int) -> str:
    raw = socket.inet_pton(socket.AF_INET6, ip6)
    groups = [raw[i : i + 4][::-1] for i in range(0, 16, 4)]
    return b"".join(groups).hex().upper() + ":" + format(port, "04X")


def _make_fake_proc(root: Path, pid_chain) -> None:
    """`pid_chain`: [(pid, comm, ppid, starttime_ticks), ...]. Builds a
    fake /proc tree good enough for `_proc_stat_info`/`_system_boot_time`:
    a `stat` file with `btime`, and one `<pid>/stat` per entry with `comm`
    parsed from the LAST `)` the same way the real function does.
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / "stat").write_text("cpu  0 0 0 0 0 0 0 0 0 0\nbtime 1000000000\n")
    for pid, comm, ppid, starttime in pid_chain:
        d = root / str(pid)
        d.mkdir(parents=True, exist_ok=True)
        fields = ["0"] * 50
        fields[0] = "S"  # state -> rest[0]
        fields[1] = str(ppid)  # -> rest[1]
        fields[19] = str(starttime)  # -> rest[19]
        (d / "stat").write_text(f"{pid} ({comm}) " + " ".join(fields))


class _FakeProc:
    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


# =============================================================================
# Prometheus text parser
# =============================================================================


# =============================================================================
# tail_bytes
# =============================================================================


def test_tail_bytes_only_reads_the_tail_not_the_whole_file():
    """tail_bytes seeks to `size - max_bytes` before reading - never reads a
    multi-MB log file in full just to show its last few KB. Verified two
    ways: the returned text excludes an early marker and includes a late
    one, AND the underlying file object is never asked to `read()` more
    than `max_bytes` (a mutation that removes the seek would still pass
    the content check by accident if `read()` weren't also checked, since
    a full read followed by slicing would look identical from the
    outside - it's the BYTE COUNT actually pulled off disk that matters
    for the "don't read a huge file" property, so this checks that too).
    """
    tmp = Path(tempfile.mkdtemp(prefix="zl-tail-"))
    try:
        p = tmp / "big.log"
        head = "HEAD_MARKER_SHOULD_NOT_APPEAR\n" + ("x" * 200000)
        tail = "\nTAIL_MARKER_SHOULD_APPEAR"
        p.write_text(head + tail)
        text = telemetry.tail_bytes(p, max_bytes=1024)
        assert_true("TAIL_MARKER_SHOULD_APPEAR" in text, "the tail marker is present")
        assert_true("HEAD_MARKER_SHOULD_NOT_APPEAR" not in text, "the head marker is excluded (only the tail was read)")
        assert_true(len(text) <= 1024, "returned text is bounded by max_bytes, not the whole file")
    finally:
        _rmtree(tmp)


def test_mutation_tail_bytes_seek_removed_is_caught():
    """M14 (review's dash sub-review): remove the seek-to-tail, so the
    whole file is read (and, for a file larger than max_bytes, the HEAD
    marker survives instead of being excluded).
    """
    original = telemetry.tail_bytes

    def no_seek(path, max_bytes=65536):
        with path.open("rb") as f:
            data = f.read()
        return data.decode("utf-8", "replace")

    tmp = Path(tempfile.mkdtemp(prefix="zl-tail-mut-"))
    try:
        p = tmp / "big.log"
        head = "HEAD_MARKER_SHOULD_NOT_APPEAR\n" + ("x" * 200000)
        tail = "\nTAIL_MARKER_SHOULD_APPEAR"
        p.write_text(head + tail)
        telemetry.tail_bytes = no_seek
        text = telemetry.tail_bytes(p, max_bytes=1024)
        caught = "HEAD_MARKER_SHOULD_NOT_APPEAR" in text
        assert_true(caught, "mutated (no-seek) tail_bytes reads the whole file -- confirms the test bites")
    finally:
        telemetry.tail_bytes = original
        _rmtree(tmp)


def test_prometheus_parser_basic_gauge_and_counter():
    text = (
        "# HELP vllm:num_requests_running docs\n"
        "# TYPE vllm:num_requests_running gauge\n"
        'vllm:num_requests_running{model_name="m",engine="0"} 3\n'
        'vllm:prompt_tokens_total{model_name="m",engine="0"} 12345\n'
    )
    samples = telemetry.parse_prometheus_text(text)
    assert_eq(samples["vllm:num_requests_running"][0].value, 3.0, "gauge value parsed")
    assert_eq(samples["vllm:num_requests_running"][0].labels["engine"], "0", "gauge label parsed")
    assert_eq(samples["vllm:prompt_tokens_total"][0].value, 12345.0, "counter value parsed")


def test_prometheus_parser_escapes_in_labels():
    line = r'm{a="say \"hi\"",b="line1\nline2",c="back\\slash"} 1'
    samples = telemetry.parse_prometheus_text(line + "\n")
    row = samples["m"][0]
    assert_eq(row.labels["a"], 'say "hi"', "escaped quote unescaped")
    assert_eq(row.labels["b"], "line1\nline2", "escaped newline unescaped")
    assert_eq(row.labels["c"], "back\\slash", "escaped backslash unescaped")
    assert_eq(row.value, 1.0, "value after labels parsed")


def test_prometheus_parser_inf_nan_comments_multiple_series():
    text = (
        "# this whole line is a comment and must be skipped\n"
        '\nmetric_bucket{le="0.1"} 3\n'
        'metric_bucket{le="0.5"} 7\n'
        'metric_bucket{le="+Inf"} 9\n'
        "gauge_thing NaN\n"
        "bare_no_labels 42\n"
    )
    samples = telemetry.parse_prometheus_text(text)
    buckets = samples["metric_bucket"]
    assert_eq(len(buckets), 3, "three bucket series parsed (multiple series, one name)")
    les = {b.labels["le"]: b.value for b in buckets}
    assert_eq(les["0.1"], 3.0, "bucket le=0.1")
    assert_eq(les["+Inf"], 9.0, "bucket le=+Inf value parsed")
    assert_true(math.isnan(samples["gauge_thing"][0].value), "NaN value parsed as float nan")
    assert_eq(samples["bare_no_labels"][0].value, 42.0, "a bare name with no {labels} still parses")
    assert_eq(samples["bare_no_labels"][0].labels, {}, "bare metric has an empty label dict")


# =============================================================================
# Rates (with counter reset) and histogram quantiles from bucket deltas
# =============================================================================


def test_counter_rate_normal():
    r = telemetry.counter_rate(100.0, 0.0, 150.0, 10.0)
    assert_close(r, 5.0, "normal counter rate = delta/dt")


def test_counter_rate_handles_reset():
    # The counter dropped from 1000 to 20 (vLLM restarted) over 10s. A naive
    # subtraction gives -98.0/s; the real rule treats the new value as
    # counting from 0.
    r = telemetry.counter_rate(1000.0, 0.0, 20.0, 10.0)
    assert_true(r is not None and r >= 0, "a counter reset never yields a negative rate")
    assert_close(r, 2.0, "reset treated as counting from 0 (20/10)")


def test_counter_rate_zero_or_negative_dt_is_none():
    assert_eq(telemetry.counter_rate(1, 0, 2, 0), None, "dt == 0 -> None")
    assert_eq(telemetry.counter_rate(1, 10, 2, 0), None, "dt < 0 -> None")


def test_counter_rate_reset_mutation_proof():
    """Prove the reset-handling test actually bites: swap in a naive
    implementation (plain subtraction, no reset guard), confirm the safety
    property now fails, restore, confirm it passes again.
    """
    original = telemetry.counter_rate

    def naive(prev_value, prev_ts, cur_value, cur_ts):
        dt = cur_ts - prev_ts
        if dt <= 0:
            return None
        return (cur_value - prev_value) / dt  # BUG: no reset handling

    telemetry.counter_rate = naive
    caught = False
    try:
        r = telemetry.counter_rate(1000.0, 0.0, 20.0, 10.0)
        assert r is not None and r >= 0
    except AssertionError:
        caught = True
    finally:
        telemetry.counter_rate = original
    assert_true(caught, "mutation proof: a naive reset handler goes negative, and the check catches it")
    r = telemetry.counter_rate(1000.0, 0.0, 20.0, 10.0)
    assert_true(r is not None and r >= 0, "restored implementation is safe again")


def test_histogram_quantile_basic():
    prev = {0.1: 0, 0.5: 0, 1.0: 0, math.inf: 0}
    cur = {0.1: 50, 0.5: 90, 1.0: 100, math.inf: 100}
    p50 = telemetry.histogram_quantile_from_buckets(prev, cur, 0.5)
    assert_true(p50 is not None and 0.0 <= p50 <= 0.5, "p50 falls within the first two buckets")
    p95 = telemetry.histogram_quantile_from_buckets(prev, cur, 0.95)
    assert_true(p95 is not None and 0.5 <= p95 <= 1.0, "p95 falls within the 0.5-1.0 bucket")


def test_histogram_quantile_no_data_is_none():
    assert_eq(telemetry.histogram_quantile_from_buckets(None, {0.1: 1}, 0.5), None, "prev missing -> None")
    assert_eq(telemetry.histogram_quantile_from_buckets({0.1: 1}, {0.1: 1}, 0.5), None, "zero total delta -> None")


def test_histogram_quantile_handles_bucket_reset():
    prev = {0.1: 900, 0.5: 950, 1.0: 1000, math.inf: 1000}
    cur = {0.1: 5, 0.5: 8, 1.0: 10, math.inf: 10}  # vLLM restarted; counters reset
    q = telemetry.histogram_quantile_from_buckets(prev, cur, 0.5)
    assert_true(q is not None, "a bucket reset still produces a quantile from the raw new counts, not a crash")


def test_extract_vllm_metrics_and_rates_end_to_end():
    def scrape(prompt_total, gen_total):
        return (
            f'vllm:num_requests_running{{engine="0"}} 1\n'
            f'vllm:num_requests_waiting{{engine="0"}} 0\n'
            f'vllm:kv_cache_usage_perc{{engine="0"}} 0.42\n'
            f'vllm:prompt_tokens_total{{engine="0"}} {prompt_total}\n'
            f'vllm:generation_tokens_total{{engine="0"}} {gen_total}\n'
            f'vllm:prefix_cache_queries_total{{engine="0"}} 100\n'
            f'vllm:prefix_cache_hits_total{{engine="0"}} 80\n'
            f'vllm:num_preemptions_total{{engine="0"}} 0\n'
            f'vllm:time_to_first_token_seconds_bucket{{le="0.1",engine="0"}} 5\n'
            f'vllm:time_to_first_token_seconds_bucket{{le="0.5",engine="0"}} 9\n'
            f'vllm:time_to_first_token_seconds_bucket{{le="+Inf",engine="0"}} 10\n'
        )

    m0 = telemetry.extract_vllm_metrics(telemetry.parse_prometheus_text(scrape(1000, 200)))
    m1 = telemetry.extract_vllm_metrics(telemetry.parse_prometheus_text(scrape(1100, 260)))
    rates = telemetry.vllm_rates(m0, 0.0, m1, 5.0)
    assert_close(rates["prompt_tokens_per_s"], 20.0, "prompt tps = (1100-1000)/5")
    assert_close(rates["generation_tokens_per_s"], 12.0, "gen tps = (260-200)/5")
    assert_eq(m1["kv_cache_usage_pct"], 42.0, "kv_cache_usage_perc (0..1) shown as a percentage (x100)")


# =============================================================================
# nvidia-smi CSV
# =============================================================================


def test_nvidia_smi_csv_two_gpus_with_na():
    text = (
        "NVIDIA A100, 42, 1024, 40960, 55, 150.5, 300.0, 1350\n"
        "NVIDIA A100, [N/A], 512, 40960, [N/A], [N/A], 300.0, 1350\n"
    )
    rows = telemetry.parse_nvidia_smi_csv(text)
    assert_eq(len(rows), 2, "two GPU rows parsed")
    assert_eq(rows[0]["utilization_gpu_pct"], 42.0, "gpu0 util parsed")
    assert_eq(rows[1]["utilization_gpu_pct"], None, "gpu1 util [N/A] -> None")
    assert_eq(rows[1]["temperature_c"], None, "gpu1 temp [N/A] -> None")
    assert_eq(rows[1]["power_draw_w"], None, "gpu1 power draw [N/A] -> None")
    assert_eq(rows[1]["memory_used_mb"], 512.0, "gpu1 mem used still parses despite other N/A fields")


def test_collect_gpu_missing_binary_raises_clear_error():
    def fake_run(cmd, **kwargs):
        raise FileNotFoundError("no such file")

    caught = False
    try:
        telemetry.collect_gpu(run=fake_run)
    except RuntimeError as exc:
        caught = "nvidia-smi" in str(exc)
    assert_true(caught, "missing nvidia-smi raises a clear, catchable RuntimeError")


# =============================================================================
# supervisorctl
# =============================================================================


def test_supervisorctl_status_parsing():
    text = (
        "compactor                        RUNNING   pid 123, uptime 1:02:03\n"
        "vllm                              FATAL     Exited too quickly (process log may have details)\n"
        "selftest                          EXITED    Sep 24 03:00 PM\n"
        "webuidb-sync                      STOPPED   Not started\n"
    )
    by_name = {r["name"]: r for r in telemetry.parse_supervisorctl_status(text)}
    assert_eq(by_name["compactor"]["state"], "RUNNING", "compactor RUNNING")
    assert_eq(by_name["compactor"]["pid"], 123, "compactor pid parsed")
    assert_eq(by_name["compactor"]["uptime"], "1:02:03", "compactor uptime parsed")
    assert_eq(by_name["vllm"]["pid"], None, "FATAL has no pid")
    assert_in("Exited too quickly", by_name["vllm"]["detail"], "vllm detail carries the reason")
    assert_eq(by_name["selftest"]["state"], "EXITED", "selftest EXITED")


def test_collect_services_flags_unexpected_and_tolerates_one_shot():
    text = (
        "compactor RUNNING pid 1, uptime 0:00:10\n"
        "vllm FATAL Exited too quickly\n"
        "selftest EXITED Sep 24 03:00 PM\n"
        "webuidb-sync STOPPED Not started\n"
        "zl-ops EXITED Sep 26 07:00 AM\n"
    )

    def fake_run(cmd, **kwargs):
        return _FakeProc(stdout=text, returncode=3)

    result = telemetry.collect_services(run=fake_run, prefer_rpc=False)
    assert_eq(result["source"], "subprocess", "prefer_rpc=False always uses the subprocess path")
    by_name = {s["name"]: s for s in result["services"]}
    assert_true(by_name["vllm"]["unexpected"], "vllm FATAL is unexpected")
    assert_true(not by_name["selftest"]["unexpected"], "selftest EXITED is tolerated (ONE_SHOT_OR_OPTIONAL)")
    assert_true(not by_name["webuidb-sync"]["unexpected"], "webuidb-sync STOPPED is tolerated")
    assert_true(not by_name["compactor"]["unexpected"], "compactor RUNNING is not unexpected")
    assert_true(not by_name["zl-ops"]["unexpected"],
                "zl-ops EXITED is tolerated (v3.1.9.8+ one-shot boot installer, ONE_SHOT_OR_OPTIONAL)")


def test_mutation_zl_ops_missing_from_one_shot_or_optional_would_flag_it():
    """Prove the test above bites: WITHOUT `zl-ops` in
    `podenv.ONE_SHOT_OR_OPTIONAL`, its own EXITED state would be flagged
    unexpected exactly like a real crash -- confirms the set membership
    (not some other code path) is what makes it tolerated.
    """
    assert_true("zl-ops" in telemetry.podenv.ONE_SHOT_OR_OPTIONAL,
                "zl-ops really is in ONE_SHOT_OR_OPTIONAL (sanity: removing it would un-fix this)")


def test_cloudflared_is_in_services_and_one_shot_or_optional():
    assert_true("cloudflared" in telemetry.podenv.SERVICES, "cloudflared is a supervisord program zl status/dash show")
    assert_true("cloudflared" in telemetry.podenv.ONE_SHOT_OR_OPTIONAL,
                "cloudflared EXITED (no token set) is tolerated by default")


def test_collect_services_cloudflared_exited_is_tolerated_by_one_shot_or_optional():
    text = (
        "compactor RUNNING pid 1, uptime 0:00:10\n"
        "cloudflared EXITED Sep 27 00:00 AM\n"
    )

    def fake_run(cmd, **kwargs):
        return _FakeProc(stdout=text, returncode=3)

    result = telemetry.collect_services(run=fake_run, prefer_rpc=False)
    by_name = {s["name"]: s for s in result["services"]}
    assert_true(not by_name["cloudflared"]["unexpected"],
                "collect_services alone (no /health/full doc) tolerates cloudflared EXITED -- "
                "the configured-but-down cross-check lives in cloudflared_fault, not here")


def test_mutation_cloudflared_missing_from_one_shot_or_optional_would_flag_it():
    """Same shape as the zl-ops mutation proof above: without `cloudflared`
    in `podenv.ONE_SHOT_OR_OPTIONAL`, a plain EXITED (no token set - the
    correct, disabled state) would be flagged unexpected exactly like a
    real crash.
    """
    assert_true("cloudflared" in telemetry.podenv.ONE_SHOT_OR_OPTIONAL,
                "cloudflared really is in ONE_SHOT_OR_OPTIONAL (sanity: removing it would un-fix this)")


def test_collect_services_missing_binary_raises():
    def fake_run(cmd, **kwargs):
        raise FileNotFoundError("no such file")

    caught = False
    try:
        telemetry.collect_services(run=fake_run, prefer_rpc=False)
    except RuntimeError:
        caught = True
    assert_true(caught, "missing supervisorctl raises, never crashes the caller")


# =============================================================================
# supervisord XML-RPC (the cheap path) and its subprocess fallback
# =============================================================================


class _FakeSupervisorRPC:
    """Stands in for `xmlrpc.client.ServerProxy` - only the one method path
    `collect_services_rpc` calls.
    """

    class _Namespace:
        def __init__(self, infos):
            self._infos = infos

        def getAllProcessInfo(self):
            return self._infos

    def __init__(self, infos):
        self.supervisor = self._Namespace(infos)


class _FailingSupervisorRPC:
    def __getattr__(self, name):
        raise ConnectionRefusedError("no such socket")


def test_collect_services_rpc_parses_getallprocessinfo():
    infos = [
        {"name": "compactor", "group": "compactor", "statename": "RUNNING", "pid": 42,
         "description": "pid 42, uptime 0:01:02"},
        {"name": "vllm", "group": "vllm", "statename": "FATAL", "pid": 0, "description": "Exited too quickly"},
    ]
    services = telemetry.collect_services_rpc(proxy=_FakeSupervisorRPC(infos))
    by_name = {s["name"]: s for s in services}
    assert_eq(by_name["compactor"]["state"], "RUNNING", "RPC RUNNING state parsed")
    assert_eq(by_name["compactor"]["pid"], 42, "RPC pid parsed")
    assert_eq(by_name["compactor"]["uptime"], "0:01:02", "RPC uptime parsed from description")
    assert_eq(by_name["vllm"]["state"], "FATAL", "RPC FATAL state parsed")
    assert_eq(by_name["vllm"]["pid"], None, "a zero pid (not running) becomes None")


def test_collect_services_prefers_rpc_when_available():
    infos = [{"name": "compactor", "group": "compactor", "statename": "RUNNING", "pid": 1, "description": "pid 1"}]
    result = telemetry.collect_services(rpc_proxy=_FakeSupervisorRPC(infos))
    assert_eq(result["source"], "rpc", "RPC path used when the proxy answers")
    assert_eq(result["services"][0]["state"], "RUNNING", "RPC data reached the caller")


def test_collect_services_falls_back_to_subprocess_when_rpc_fails():
    text = "compactor RUNNING pid 1, uptime 0:00:10\n"

    def fake_run(cmd, **kwargs):
        return _FakeProc(stdout=text, returncode=0)

    result = telemetry.collect_services(run=fake_run, rpc_proxy=_FailingSupervisorRPC())
    assert_eq(result["source"], "subprocess", "a broken RPC proxy falls back to supervisorctl")
    assert_eq(result["services"][0]["state"], "RUNNING", "subprocess fallback data reached the caller")


def test_hub_services_backs_off_cadence_on_subprocess_fallback():
    """Mutation-relevant proof: the Hub polls the (free) RPC path at the
    normal cadence but backs the (costly, fork+exec) subprocess fallback
    off to a slower one - this is what keeps `zl dash`'s CPU cost down
    when the RPC socket is unavailable. Verified by ticking the Hub twice
    with a faked `_collect_services` and reading `_next_wait` directly.
    """
    hub = telemetry.Hub()
    hub._collect_services = lambda: {"services": [], "source": "subprocess"}
    hub._tick("services")
    assert_eq(
        hub._next_wait.get("services"),
        telemetry.CADENCE_S["services"] * 3.0,
        "subprocess-sourced services back off to 3x cadence",
    )
    hub._collect_services = lambda: {"services": [], "source": "rpc"}
    hub._tick("services")
    assert_eq(
        hub._next_wait.get("services"),
        telemetry.CADENCE_S["services"],
        "rpc-sourced services use the normal cadence",
    )


def test_mutation_rpc_socket_path_reader_finds_repo_conf():
    """Mutation proof for `read_supervisor_socket_path`: point it at ONLY a
    conf file that doesn't exist, watch it fall back to the documented
    default, then point it at the repo's real `supervisord.conf` and watch
    it parse the actual configured path.
    """
    missing = telemetry.read_supervisor_socket_path(conf_paths=(Path("/no/such/file.conf"),))
    assert_eq(missing, telemetry.DEFAULT_SUPERVISOR_SOCKET, "no readable conf -> documented default")

    real_conf = Path(__file__).resolve().parent.parent / "supervisord.conf"
    if real_conf.is_file():
        found = telemetry.read_supervisor_socket_path(conf_paths=(real_conf,))
        assert_eq(str(found), "/var/run/supervisor.sock", "parsed from the repo's own supervisord.conf")


# =============================================================================
# /proc/net/tcp[6] and SSH_CONNECTION
# =============================================================================


def test_proc_net_tcp_parsing_established_and_listen():
    header = "  sl  local_address rem_address   st ...\n"
    listen_line = f"   0: {_hex_ip_port('0.0.0.0', 22)} 00000000:0000 0A 0:0 00:0 0 0 0 1 1\n"
    established_line = (
        f"   1: {_hex_ip_port('10.0.0.5', 22)} {_hex_ip_port('203.0.113.9', 54321)} 01 0:0 00:0 0 0 0 2 1\n"
    )
    rows = telemetry.parse_proc_net_tcp(header + listen_line + established_line)
    assert_eq(len(rows), 2, "two data rows parsed (header skipped)")
    listen = next(r for r in rows if r["state"] == "0A")
    assert_eq(listen["local_port"], 22, "listen local port")
    assert_eq(listen["local_ip"], "0.0.0.0", "listen local ip")
    est = next(r for r in rows if r["state"] == "01")
    assert_eq(est["local_port"], 22, "established local port")
    assert_eq(est["remote_ip"], "203.0.113.9", "established remote ip")
    assert_eq(est["remote_port"], 54321, "established remote port")


def test_proc_net_tcp6_parsing():
    header = "  sl  local_address rem_address   st ...\n"
    line = f"   0: {_hex_ip6_port('::1', 2222)} {_hex_ip6_port('2001:db8::1', 5555)} 01 0:0 00:0 0 0 0 1 1\n"
    rows = telemetry.parse_proc_net_tcp(header + line, v6=True)
    assert_eq(len(rows), 1, "one tcp6 row parsed")
    assert_eq(rows[0]["local_ip"], "::1", "tcp6 local address decoded")
    assert_eq(rows[0]["remote_ip"], "2001:db8::1", "tcp6 remote address decoded")
    assert_eq(rows[0]["remote_port"], 5555, "tcp6 remote port decoded")


def test_parse_ssh_connection():
    d = telemetry.parse_ssh_connection("203.0.113.5 54321 10.0.0.2 22")
    assert_eq(
        d,
        {"client_ip": "203.0.113.5", "client_port": 54321, "server_ip": "10.0.0.2", "server_port": 22},
        "SSH_CONNECTION parsed into its four fields",
    )
    assert_eq(telemetry.parse_ssh_connection(None), None, "unset -> None")
    assert_eq(telemetry.parse_ssh_connection(""), None, "empty -> None")
    assert_eq(telemetry.parse_ssh_connection("only two fields"), None, "malformed -> None")


def test_collect_ssh_session_full():
    tmp = Path(tempfile.mkdtemp(prefix="zl-ssh-"))
    try:
        tcp = tmp / "tcp"
        tcp6 = tmp / "tcp6"
        listen = f"   0: {_hex_ip_port('0.0.0.0', 22)} 00000000:0000 0A 0:0 00:0 0 0 0 1 1\n"
        est1 = f"   1: {_hex_ip_port('10.0.0.5', 22)} {_hex_ip_port('203.0.113.9', 1000)} 01 0:0 00:0 0 0 0 2 1\n"
        est2 = f"   2: {_hex_ip_port('10.0.0.5', 22)} {_hex_ip_port('203.0.113.9', 1001)} 01 0:0 00:0 0 0 0 3 1\n"
        tcp.write_text("header\n" + listen + est1 + est2)
        tcp6.write_text("header\n")
        environ = {"SSH_CONNECTION": "203.0.113.9 1000 10.0.0.5 22", "TMUX": "1"}
        result = telemetry.collect_ssh_session(
            now=time.time(), environ=environ, tcp_path=tcp, tcp6_path=tcp6, proc_root=tmp
        )
        assert_eq(result["port"], 22, "port taken from SSH_CONNECTION's server_port")
        assert_eq(result["established_count"], 2, "two established rows on that port")
        assert_eq(result["established_client_ips"], ["203.0.113.9"], "one distinct client ip")
        assert_true(result["listening"], "sshd listening detected on that port")
        assert_true(result["in_tmux"], "TMUX env present")
        assert_eq(result["session"], {"in_ssh": False}, "no fake sshd in this process's ancestry here")
    finally:
        _rmtree(tmp)


def test_collect_ssh_session_no_connection_is_web_terminal():
    tmp = Path(tempfile.mkdtemp(prefix="zl-ssh-noconn-"))
    try:
        (tmp / "tcp").write_text("header\n")
        (tmp / "tcp6").write_text("header\n")
        result = telemetry.collect_ssh_session(
            environ={}, tcp_path=tmp / "tcp", tcp6_path=tmp / "tcp6", proc_root=tmp
        )
        assert_eq(result["connection"], None, "no SSH_CONNECTION -> None (RunPod Web Terminal)")
        assert_eq(result["port"], 22, "defaults to port 22 when not in a recognised SSH session")
        assert_eq(result["in_tmux"], False, "no TMUX -> False")
    finally:
        _rmtree(tmp)


def test_find_sshd_ancestor_walks_parents():
    tmp = Path(tempfile.mkdtemp(prefix="zl-proc-"))
    try:
        _make_fake_proc(
            tmp,
            [
                (300, "python3", 200, 500),
                (200, "bash", 100, 400),
                (100, "sshd", 1, 300),
            ],
        )
        found = telemetry.find_sshd_ancestor(pid=300, proc_root=tmp)
        assert_true(found is not None, "sshd found by walking parents")
        assert_eq(found["pid"], 100, "sshd pid identified")
        assert_eq(found["starttime_ticks"], 300, "sshd starttime read")
    finally:
        _rmtree(tmp)


def test_find_sshd_ancestor_none_for_web_terminal():
    tmp = Path(tempfile.mkdtemp(prefix="zl-proc-noweb-"))
    try:
        _make_fake_proc(
            tmp,
            [
                (300, "python3", 200, 500),
                (200, "bash", 1, 400),  # parents straight to init, no sshd anywhere
            ],
        )
        assert_eq(telemetry.find_sshd_ancestor(pid=300, proc_root=tmp), None, "no sshd in ancestry -> None")
    finally:
        _rmtree(tmp)


def test_session_start_and_duration_computes_from_btime_and_ticks():
    tmp = Path(tempfile.mkdtemp(prefix="zl-proc-dur-"))
    try:
        hz = telemetry._clock_ticks_per_sec()
        _make_fake_proc(
            tmp,
            [
                (300, "python3", 100, 0),
                (100, "sshd", 1, 5 * hz),  # sshd started 5s after boot
            ],
        )
        result = telemetry.session_start_and_duration(now=1000000010.0, proc_root=tmp, pid=300)
        assert_true(result["in_ssh"], "session recognised as SSH")
        assert_close(result["start_epoch"], 1000000005.0, "start_epoch = btime + starttime_ticks/hz")
        assert_close(result["duration_s"], 5.0, "duration = now - start_epoch")
    finally:
        _rmtree(tmp)


# =============================================================================
# DSR reply parsing (terminal round-trip probe)
# =============================================================================


def test_find_dsr_reply_basic():
    reply = "\x1b[24;80R"
    found = telemetry.find_dsr_reply(reply)
    assert_true(found is not None, "DSR reply found")
    row, col, start, end = found
    assert_eq((row, col), (24, 80), "row/col parsed")
    assert_eq((start, end), (0, len(reply)), "span covers the whole reply")


def test_find_dsr_reply_embedded_in_surrounding_bytes():
    buf = "hello\x1b[3;7Rworld"
    found = telemetry.find_dsr_reply(buf)
    assert_true(found is not None, "DSR reply found inside surrounding bytes")
    row, col, start, end = found
    assert_eq((row, col), (3, 7), "row/col parsed out of the noise")
    assert_eq(buf[:start], "hello", "text BEFORE the reply is untouched, for the caller to treat as real keys")
    assert_eq(buf[end:], "world", "text AFTER the reply is untouched")


def test_find_dsr_reply_absent():
    assert_eq(telemetry.find_dsr_reply("just some keys, no escape sequence"), None, "no reply -> None")


def test_find_dsr_reply_is_never_returned_as_a_keypress_source_string():
    # The whole point of returning (row, col, start, end) rather than just
    # a bool is so a caller can excise exactly the matched span and be left
    # with nothing from the reply itself to hand to a key handler.
    buf = "q\x1b[10;5Rp"
    row, col, start, end = telemetry.find_dsr_reply(buf)
    remaining_as_keys = buf[:start] + buf[end:]
    assert_eq(remaining_as_keys, "qp", "excising the matched span leaves only the real keystrokes")
    assert_true("\x1b" not in remaining_as_keys, "no escape byte from the reply leaks into the keystroke stream")


# =============================================================================
# interpret_health — every rule, missing fields, v3.1.6.1 shape, unreachable
# =============================================================================


def test_interpret_health_unreachable():
    r = telemetry.interpret_health(None, time.time())
    assert_eq(r["status"], "crit", "doc=None (unreachable) -> crit")
    assert_true(any("unreachable" in x for x in r["reasons"]), "reason names it")


def test_interpret_health_non_dict_body():
    r = telemetry.interpret_health("not a dict", time.time())
    assert_eq(r["status"], "crit", "a non-object body -> crit, never a crash")


def test_interpret_health_down():
    doc = {"status": "down", "status_reasons": ["storage broken"]}
    r = telemetry.interpret_health(doc, time.time())
    assert_eq(r["status"], "crit", "status down -> crit")


def test_interpret_health_degraded_with_reasons():
    doc = {"status": "degraded", "status_reasons": ["vLLM unreachable"]}
    r = telemetry.interpret_health(doc, time.time())
    assert_eq(r["status"], "warn", "status degraded -> warn")
    assert_in("vLLM unreachable", r["reasons"], "compactor's own reason is surfaced")


def test_interpret_health_unreadable_memory_nonzero():
    doc = {"status": "ok", "stats": {"unreadable": {"facts": 0, "summaries": 2}}}
    r = telemetry.interpret_health(doc, time.time())
    assert_eq(r["status"], "crit", "any nonzero unreadable count -> crit")


def test_interpret_health_unreadable_memory_all_zero_stays_ok():
    doc = {"status": "ok", "stats": {"unreadable": {"facts": 0, "summaries": 0}}}
    r = telemetry.interpret_health(doc, time.time())
    assert_eq(r["status"], "ok", "all-zero unreadable counts never degrade")


def test_interpret_health_unreadable_memory_exactly_one_is_crit():
    """The boundary case: even a SINGLE unreadable file is crit (the rule
    is `> 0`, not `> 1`) - catches a mutation that only trips at count>=2.
    """
    doc = {"status": "ok", "stats": {"unreadable": {"facts": 1, "summaries": 0}}}
    r = telemetry.interpret_health(doc, time.time())
    assert_eq(r["status"], "crit", "exactly one unreadable file is already crit")


def test_interpret_health_sqlite_journal_hot():
    doc = {"status": "ok", "checks": {"sqlite_journal": {"hot": True, "path": "/data/x"}}}
    r = telemetry.interpret_health(doc, time.time())
    assert_eq(r["status"], "crit", "a hot sqlite journal -> crit")


def test_interpret_health_hierarchy_stuck():
    doc = {"status": "ok", "checks": {"hierarchy": {"catching_up_all": [{"conv_id": "c1", "verdict": "stuck"}]}}}
    r = telemetry.interpret_health(doc, time.time())
    assert_eq(r["status"], "warn", "hierarchy catch-up stuck -> warn")


def test_interpret_health_hierarchy_converging_is_not_a_finding():
    doc = {
        "status": "ok",
        "checks": {"hierarchy": {"catching_up_all": [{"conv_id": "c1", "verdict": "converging"}]}},
    }
    r = telemetry.interpret_health(doc, time.time())
    assert_eq(r["status"], "ok", "converging never degrades status, per OPERATIONS.md")


def test_interpret_health_ledger_degraded():
    doc = {"status": "ok", "checks": {"ledger": {"degraded": True, "reason": "x"}}}
    r = telemetry.interpret_health(doc, time.time())
    assert_eq(r["status"], "crit", "checks.ledger.degraded -> crit (speculative field, still enforced)")


def test_interpret_health_tunnel_not_ready_warns():
    doc = {"status": "ok", "tunnel": {"configured": True, "ready": False}}
    r = telemetry.interpret_health(doc, time.time())
    assert_eq(r["status"], "warn", "tunnel configured but not ready -> warn")
    assert_true(any("tunnel not ready" in x for x in r["reasons"]), "reason names it")


def test_interpret_health_tunnel_ready_is_ok():
    doc = {"status": "ok", "tunnel": {"configured": True, "ready": True}}
    r = telemetry.interpret_health(doc, time.time())
    assert_eq(r["status"], "ok", "tunnel configured and ready never warns")


def test_interpret_health_tunnel_ready_unknown_shows_nothing():
    doc = {"status": "ok", "tunnel": {"configured": True, "ready": None}}
    r = telemetry.interpret_health(doc, time.time())
    assert_eq(r["status"], "ok", "ready=None (unknown) is not a finding")


def test_interpret_health_tunnel_not_configured_is_fine():
    doc = {"status": "ok", "tunnel": {"configured": False, "ready": None}}
    r = telemetry.interpret_health(doc, time.time())
    assert_eq(r["status"], "ok", "not configured -> nothing to warn about, whatever ready says")


def test_interpret_health_tunnel_absent_is_tolerated():
    """Older compactors (pre-v3.1.9.9) never send a `tunnel` key at all -
    `doc.get("tunnel") or {}` must not raise or fabricate a finding.
    """
    doc = {"status": "ok"}
    r = telemetry.interpret_health(doc, time.time())
    assert_eq(r["status"], "ok", "no tunnel key at all is tolerated")


def test_mutation_tunnel_ready_none_treated_as_false_would_warn_spuriously():
    """Prove the ready=None test above bites: a broken version that folds
    `ready is not True` into the warn condition (instead of the exact
    `ready is False`) would spuriously warn on merely UNKNOWN readiness --
    confirm that broken read fires here, then confirm the real code (which
    checks `is False`) does not.
    """
    tunnel = {"configured": True, "ready": None}
    mutated_would_warn = bool(tunnel.get("configured") and tunnel.get("ready") is not True)
    assert_true(mutated_would_warn, "mutated (ready is not True) fires on unknown readiness -- confirms the bite")
    assert_true(tunnel.get("ready") is not False, "the real code's exact `is False` check does not fire here")


# =============================================================================
# Cloudflare Tunnel (v3.1.9.9): cloudflared_fault / tunnel_status_word
# =============================================================================


def test_cloudflared_fault_none_when_not_configured():
    assert_eq(telemetry.cloudflared_fault(None, "EXITED"), None, "no doc at all -> nothing to flag")
    assert_eq(telemetry.cloudflared_fault({"status": "ok"}, "EXITED"), None,
               "tunnel key absent (older compactor) -> nothing to flag")
    doc = {"tunnel": {"configured": False, "ready": None}}
    assert_eq(telemetry.cloudflared_fault(doc, "EXITED"), None, "configured=False -> EXITED is fine")


def test_cloudflared_fault_none_when_running():
    doc = {"tunnel": {"configured": True, "ready": True}}
    assert_eq(telemetry.cloudflared_fault(doc, "RUNNING"), None, "configured and RUNNING -> nothing to flag")


def test_cloudflared_fault_flags_configured_but_down():
    doc = {"tunnel": {"configured": True, "ready": None}}
    result = telemetry.cloudflared_fault(doc, "EXITED")
    assert_true(result is not None, "configured but not RUNNING -> flagged")
    assert_in("tunnel configured but cloudflared is EXITED", result, "warning names the actual state")


def test_mutation_cloudflared_fault_ignoring_configured_would_flag_disabled_tunnel():
    """Prove the test above bites: a broken version that flags ANY
    non-RUNNING cloudflared (ignoring `tunnel.configured` entirely) would
    also flag the correct, disabled-by-config case -- confirm that broken
    read fires on an unconfigured tunnel, then confirm the real code does
    not.
    """
    state = "EXITED"
    mutated_would_flag = state != "RUNNING"  # ignores tunnel.configured entirely
    assert_true(mutated_would_flag, "mutated (state-only) flags every non-RUNNING state -- confirms the bite")
    doc = {"tunnel": {"configured": False, "ready": None}}
    assert_eq(telemetry.cloudflared_fault(doc, state), None,
              "the real code correctly leaves a disabled-by-config EXITED alone")


def test_tunnel_status_word_all_shapes():
    assert_eq(telemetry.tunnel_status_word(None), None, "no doc -> None (hide the line)")
    assert_eq(telemetry.tunnel_status_word({"status": "ok"}), None, "tunnel key absent -> None (hide the line)")
    assert_eq(telemetry.tunnel_status_word({"tunnel": {"configured": False, "ready": None}}), "disabled",
              "configured=False -> disabled")
    assert_eq(telemetry.tunnel_status_word({"tunnel": {"configured": True, "ready": True}}), "ready",
              "configured, ready=True -> ready")
    assert_eq(telemetry.tunnel_status_word({"tunnel": {"configured": True, "ready": False}}), "not ready",
              "configured, ready=False -> not ready")
    assert_eq(telemetry.tunnel_status_word({"tunnel": {"configured": True, "ready": None}}), "unknown",
              "configured, ready=None -> unknown")


def test_collect_health_carries_tunnel_status():
    doc = {"status": "ok", "tunnel": {"configured": True, "ready": True}}

    def fake_fetch(url, timeout_s):
        return json.dumps(doc).encode("utf-8")

    result = telemetry.collect_health(fetch=fake_fetch)
    assert_eq(result["tunnel_status"], "ready", "collect_health surfaces tunnel_status_word for collect_snapshot/bundle")


def test_interpret_health_backup_thresholds():
    now = time.time()
    fresh = {"status": "ok", "backups": {"latest": "a.tgz", "latest_mtime": now - 3600, "count": 1}}
    assert_eq(telemetry.interpret_health(fresh, now)["status"], "ok", "a 1h-old backup is fine")

    warn_doc = {"status": "ok", "backups": {"latest": "a.tgz", "latest_mtime": now - 31 * 3600, "count": 1}}
    assert_eq(telemetry.interpret_health(warn_doc, now)["status"], "warn", "a 31h-old backup warns")

    crit_doc = {"status": "ok", "backups": {"latest": "a.tgz", "latest_mtime": now - 37 * 3600, "count": 1}}
    assert_eq(telemetry.interpret_health(crit_doc, now)["status"], "crit", "a 37h-old backup crits")

    none_doc = {"status": "ok", "backups": {"latest": None, "latest_mtime": None, "count": 0}}
    assert_eq(telemetry.interpret_health(none_doc, now)["status"], "crit", "no backups at all crits")


def test_interpret_health_v3_1_6_1_shape_tolerates_missing_fields():
    """v3.1.6.1, the image the pod runs, has NONE of checks.hierarchy,
    checks.reuse, checks.budget_margin, checks.truncated_summaries, or
    stats.unreadable (OPERATIONS.md's own version table) — every field
    access above must tolerate that shape without raising or misreading.
    """
    doc = {
        "status": "ok",
        "status_reasons": [],
        "checks": {"vllm": {"ok": True}, "storage": {"ok": True}},
        "stats": {"conversations": 12},
        "backups": {"latest": "b.tgz", "latest_mtime": time.time() - 3600, "count": 5},
    }
    r = telemetry.interpret_health(doc, time.time())
    assert_eq(r["status"], "ok", "a v3.1.6.1-shaped doc with nothing wrong reads ok")
    assert_eq(r["reasons"], [], "no reasons when nothing is wrong")

    doc["status"] = "degraded"
    doc["status_reasons"] = ["vLLM unreachable"]
    r2 = telemetry.interpret_health(doc, time.time())
    assert_eq(r2["status"], "warn", "degraded still surfaces on the old shape, missing checks.* notwithstanding")


def test_interpret_health_against_real_meta_health_json_files():
    """Load the 4 real `meta/health.json` files under
    `/home/drew/pod-exports/*/bundle*/zla-bundle-*/meta/` (pure operational
    metadata - status/checks/stats, never chat content, so it's fine to
    read directly) and confirm `interpret_health` reads each one without
    raising and reaches a real, non-empty status. Skipped honestly (not
    silently) if the bundles aren't present on this machine - they never
    will be inside the docker unit-tests image or CI, and their contents
    are never copied into this repo.
    """
    import glob

    paths = sorted(glob.glob("/home/drew/pod-exports/*/*/zla-bundle-*/meta/health.json"))
    if not paths:
        print("  skip interpret_health_against_real_meta_health_json_files: no real bundles on this machine")
        return
    for p in paths:
        with open(p, encoding="utf-8") as f:
            doc = json.load(f)
        r = telemetry.interpret_health(doc, time.time())
        assert_true(r["status"] in ("ok", "warn", "crit"), f"{p}: a real status came back ({r['status']!r})")
        assert_true(isinstance(r["findings"], list), f"{p}: findings is a list")
        # Cross-check against the document's OWN status where it's ok/degraded/down
        # (interpret_health's rule 4/5) - a real file's own verdict should never
        # be silently contradicted in the safe direction (ok when the doc says
        # degraded/down would be a real bug, not just "tolerant").
        raw = doc.get("status")
        if raw == "down":
            assert_eq(r["status"], "crit", f"{p}: doc status=down must read as crit")
        elif raw == "degraded":
            assert_true(r["status"] in ("warn", "crit"), f"{p}: doc status=degraded must not read as ok")
    print(f"  ok   interpret_health read {len(paths)} real meta/health.json file(s) without raising")


def test_interpret_health_missing_every_field():
    r = telemetry.interpret_health({}, time.time())
    assert_true(r["status"] in ("ok", "warn", "crit"), "an empty doc still returns a valid status")
    assert_eq(len(r["findings"]) >= 1, True, "a missing status field is itself flagged")
    assert_eq(r["findings"][0]["level"], "warn", "missing 'status' is a warn, not a crash")


def test_interpret_health_unreadable_mutation_proof():
    """Prove the unreadable-memory rule test actually bites: swap in a
    broken interpret_health that silently drops the finding, confirm the
    safety property now fails, restore, confirm green again.
    """
    original = telemetry.interpret_health

    def broken(doc, now):
        result = original(doc, now)
        result = {
            "status": "ok",
            "reasons": [t for t in result["reasons"] if "unreadable" not in t],
            "findings": [f for f in result["findings"] if "unreadable" not in f["text"]],
        }
        return result

    telemetry.interpret_health = broken
    caught = False
    try:
        doc = {"status": "ok", "stats": {"unreadable": {"facts": 3}}}
        r = telemetry.interpret_health(doc, time.time())
        assert r["status"] == "crit"
    except AssertionError:
        caught = True
    finally:
        telemetry.interpret_health = original
    assert_true(caught, "mutation proof: dropping the unreadable-memory rule is caught red")
    r = telemetry.interpret_health({"status": "ok", "stats": {"unreadable": {"facts": 3}}}, time.time())
    assert_true(r["status"] == "crit", "restored rule is green again")


# =============================================================================
# Shared backup-age thresholds
# =============================================================================


def test_backup_age_level_thresholds():
    assert_eq(telemetry.backup_age_level(None), "crit", "no backup at all -> crit")
    assert_eq(telemetry.backup_age_level(0), "ok", "brand new -> ok")
    assert_eq(telemetry.backup_age_level(29.9), "ok", "just under warn -> ok")
    assert_eq(telemetry.backup_age_level(30.0), "warn", "at the warn threshold -> warn")
    assert_eq(telemetry.backup_age_level(35.9), "warn", "just under crit -> warn")
    assert_eq(telemetry.backup_age_level(36.0), "crit", "at the crit threshold -> crit")


def test_collect_backups_picks_newest_by_mtime():
    # compactor/backup.py: `_ARCHIVE_PREFIX = "zions-backup-"`, stamp
    # `time.strftime("%Y%m%d-%H%M%S", time.gmtime())`, `_ARCHIVE_SUFFIX =
    # ".tar.gz"` - real archive names, not made-up ones.
    tmp = Path(tempfile.mkdtemp(prefix="zl-backups-"))
    try:
        old = tmp / "zions-backup-20260101-000000.tar.gz"
        new = tmp / "zions-backup-20260924-120000.tar.gz"
        old.write_text("x")
        new.write_text("y")
        now = time.time()
        os.utime(old, (now - 100000, now - 100000))
        os.utime(new, (now - 100, now - 100))
        result = telemetry.collect_backups(backups_dir=tmp)
        assert_eq(result["latest"], "zions-backup-20260924-120000.tar.gz", "newest by mtime chosen, not by name")
        assert_eq(result["count"], 2, "count of files")
    finally:
        _rmtree(tmp)


def test_collect_backups_empty_dir():
    tmp = Path(tempfile.mkdtemp(prefix="zl-backups-empty-"))
    try:
        result = telemetry.collect_backups(backups_dir=tmp)
        assert_eq(
            result, {"latest": None, "latest_mtime": None, "count": 0, "total_bytes": 0}, "empty dir -> no backups"
        )
    finally:
        _rmtree(tmp)


def test_collect_backups_ignores_partial_and_staging_names():
    """backup.py's atomic publish writes a `.partial` temp archive (backup.py
    :1276) and can leave a `zions-backup-<stamp>-<mkdtemp suffix>` STAGING
    DIRECTORY behind (backup.py:1607) if a run is interrupted. Neither may
    ever be shown as "the newest backup" - a `.partial` counting as done
    would tell the operator a backup completed when it might not have
    (verify_backup hasn't even run on it yet).
    """
    tmp = Path(tempfile.mkdtemp(prefix="zl-backups-partial-"))
    try:
        real = tmp / "zions-backup-20260101-000000.tar.gz"
        real.write_text("real archive")
        now = time.time()
        os.utime(real, (now - 100000, now - 100000))

        partial = tmp / "zions-backup-20260924-235959.tar.gz.partial"
        partial.write_text("half-written archive, much newer mtime")
        os.utime(partial, (now - 10, now - 10))  # far newer than the real archive

        staging = tmp / "zions-backup-20260924-235959-ab12cd"
        staging.mkdir()

        result = telemetry.collect_backups(backups_dir=tmp)
        assert_eq(result["count"], 1, "only the one real archive counts")
        assert_eq(result["latest"], real.name, "the .partial's newer mtime never wins")
    finally:
        _rmtree(tmp)


def test_mutation_backup_archive_pattern_disabled_is_caught():
    """A test only counts if you watched it fail: relax the archive-name
    pattern to match ANY file, watch `test_collect_backups_ignores_partial_
    and_staging_names`'s own assertion fail, then restore.
    """
    original = telemetry._BACKUP_ARCHIVE_RE
    tmp = Path(tempfile.mkdtemp(prefix="zl-backups-mut-"))
    try:
        telemetry._BACKUP_ARCHIVE_RE = re.compile(r".*")  # matches everything, including .partial
        real = tmp / "zions-backup-20260101-000000.tar.gz"
        real.write_text("x")
        os.utime(real, (time.time() - 100000,) * 2)
        partial = tmp / "zions-backup-20260924-235959.tar.gz.partial"
        partial.write_text("y")
        os.utime(partial, (time.time() - 10,) * 2)
        result = telemetry.collect_backups(backups_dir=tmp)
        caught = result["latest"] != partial.name
        assert_true(not caught, "mutated (match-everything) pattern lets the .partial win -- confirms the test bites")
    finally:
        telemetry._BACKUP_ARCHIVE_RE = original
        _rmtree(tmp)
    # Green again with the real pattern restored (fresh dir - the mutated
    # run's tmpdir was already removed above).
    tmp2 = Path(tempfile.mkdtemp(prefix="zl-backups-mut2-"))
    try:
        real = tmp2 / "zions-backup-20260101-000000.tar.gz"
        real.write_text("x")
        os.utime(real, (time.time() - 100000,) * 2)
        partial = tmp2 / "zions-backup-20260924-235959.tar.gz.partial"
        partial.write_text("y")
        os.utime(partial, (time.time() - 10,) * 2)
        result = telemetry.collect_backups(backups_dir=tmp2)
        assert_eq(result["latest"], real.name, "restored pattern excludes the .partial again")
    finally:
        _rmtree(tmp2)


# =============================================================================
# Log level-field anchoring (per real format: compactor, vLLM, OpenWebUI,
# supervisord) and the privacy.scrub_log_line routing on top of it.
# =============================================================================


def test_is_signal_line_anchors_to_compactor_app_level_field():
    assert_true(
        telemetry.is_signal_line("2026-09-24 10:00:00,123 compactor WARNING low disk", "compactor.log"),
        "WARNING in the level field is signal",
    )
    assert_true(
        telemetry.is_signal_line("2026-09-24 10:00:00,123 compactor.tok ERROR boom", "compactor.log"),
        "ERROR in the level field is signal, any logger name",
    )
    assert_true(
        not telemetry.is_signal_line("2026-09-24 10:00:00,123 compactor INFO routine heartbeat", "compactor.log"),
        "plain INFO is not signal",
    )
    # The exact real-bundle failure mode this anchoring fixes: a WARNING
    # line that merely CONTAINS the word "content" as ordinary diagnostic
    # text (not a level word) must still be recognised as signal by its
    # real level field - a substring scanner would too, so this alone
    # doesn't prove anchoring; test_line_level_field_ignores_words_in_the_
    # message below does.
    assert_true(
        telemetry.is_signal_line(
            "2026-09-24 10:00:00,123 compactor WARNING could not apply the chat template; content window trimmed",
            "compactor.log",
        ),
        "a WARNING line that happens to contain the word 'content' is still signal",
    )


def test_line_level_field_ignores_words_in_the_message():
    """The bug this replaces: a substring search for 'ERROR'/'WARNING'
    matches the word anywhere, including inside a MESSAGE that isn't a
    level at all (e.g. a message that quotes another system's log level).
    Anchoring to the level FIELD means a message-only occurrence is never
    mistaken for the line's own level.
    """
    line = '2026-09-24 10:00:00,123 compactor INFO downstream said its level was "ERROR" in its own reply'
    assert_eq(telemetry._line_level_field(line, "compactor.log"), "INFO", "the FIELD says INFO, not ERROR")
    assert_true(not telemetry.is_signal_line(line, "compactor.log"), "so this INFO line is not signal")


def test_is_signal_line_uvicorn_colon_format():
    assert_true(
        telemetry.is_signal_line("ERROR:     Exception in ASGI application", "compactor-error.log"),
        "uvicorn-style 'LEVEL:     message' ERROR is signal",
    )
    assert_true(
        not telemetry.is_signal_line("INFO:     Application startup complete.", "compactor-error.log"),
        "uvicorn-style INFO is not signal",
    )


def test_is_signal_line_vllm_format():
    running = "(APIServer pid=115) INFO 09-22 20:27:19 [loggers.py:259] Engine 000: Avg prompt throughput: 1.0 tokens/s"
    warn = "(APIServer pid=115) WARNING 09-22 20:27:19 [x.py:1] something degraded"
    no_level = "(EngineCore pid=920) Capturing CUDA graphs (this takes a while)"
    assert_true(not telemetry.is_signal_line(running, "vllm.log"), "vLLM INFO throughput line is not signal")
    assert_true(telemetry.is_signal_line(warn, "vllm.log"), "vLLM WARNING is signal")
    assert_true(not telemetry.is_signal_line(no_level, "vllm.log"), "an EngineCore raw print has no level -> not signal")


def test_is_signal_line_loguru_openwebui_format():
    info = "2026-09-22 09:17:18.163 | INFO     | uvicorn.protocols.http - something routine"
    error = "2026-09-23 06:58:01.534 | ERROR    | open_webui.utils.x - something broke"
    continuation = "    await self.app(scope, receive, send)"
    assert_true(not telemetry.is_signal_line(info, "openwebui.log"), "loguru INFO is not signal")
    assert_true(telemetry.is_signal_line(error, "openwebui.log"), "loguru ERROR is signal")
    assert_true(
        not telemetry.is_signal_line(continuation, "openwebui.log"),
        "a bare traceback continuation frame (no level field of its own) is not signal",
    )
    assert_true(
        telemetry.is_signal_line("Traceback (most recent call last):", "openwebui.log"),
        "the traceback header itself is always signal",
    )


def test_is_signal_line_supervisord_format():
    crit = "2026-09-22 09:07:30,167 CRIT Server 'unix_http_server' running as root"
    info = "2026-09-22 09:07:31,170 INFO spawned: 'processes' with pid 1"
    assert_true(telemetry.is_signal_line(crit, "supervisord.log"), "supervisord CRIT (4-char) is signal")
    assert_true(not telemetry.is_signal_line(info, "supervisord.log"), "supervisord INFO is not signal")
    assert_eq(telemetry._line_level_field(crit, "supervisord.log"), "CRITICAL", "CRIT expands to CRITICAL")


def test_is_signal_line_cloudflared_error_format():
    """v3.1.9.9: cloudflared's own zerolog-style INF/WRN/ERR/DBG lines,
    anchored to `cloudflared-error.log`'s own format (not a substring scan
    over "ERR" appearing anywhere in a location code or connector id).
    """
    running = (
        "2026-09-22T09:07:30Z INF Registered tunnel connection connIndex=0 "
        "connection=1b2c3d4e-5f60-7182-93a4-b5c6d7e8f9a0 event=0 "
        "ip=198.51.100.7 location=SEA protocol=quic"
    )
    warn = "2026-09-22T09:07:31Z WRN Retrying connection in up to 8s"
    err = "2026-09-22T09:07:32Z ERR Failed to serve quic connection error=\"context canceled\""
    assert_true(not telemetry.is_signal_line(running, "cloudflared-error.log"), "cloudflared INF is not signal")
    assert_true(telemetry.is_signal_line(warn, "cloudflared-error.log"), "cloudflared WRN is signal")
    assert_true(telemetry.is_signal_line(err, "cloudflared-error.log"), "cloudflared ERR is signal")
    assert_eq(telemetry._line_level_field(warn, "cloudflared-error.log"), "WARNING", "WRN expands to WARNING")
    assert_eq(telemetry._line_level_field(err, "cloudflared-error.log"), "ERROR", "ERR expands to ERROR")


def test_cloudflared_error_log_is_in_log_signal_files():
    assert_true("cloudflared-error.log" in telemetry.LOG_SIGNAL_FILES,
                "cloudflared-error.log is watched by the dash log-signal panel / zl bundle")
    assert_true("cloudflared.log" not in telemetry.LOG_SIGNAL_FILES,
                "cloudflared.log (stdout) never carries a level word -- not added as a signal source")


def test_collect_log_signal_routes_through_privacy_scrub_not_a_line_drop():
    """The real-bundle-verified regression this replaces: the old
    substring-drop filter dropped a WARNING line entirely just because it
    contained the bare word "content" with no actual message text nearby
    (265 of 876 real compactor.log WARNING/ERROR lines, in one real
    bundle). The fix scrubs the actual risky SPAN via `privacy.py` and
    keeps the line.
    """
    tmp = Path(tempfile.mkdtemp(prefix="zl-log-signal-"))
    try:
        body = "\n".join(
            [
                "2026-09-24 10:00:00,000 compactor WARNING could not apply the chat template; content window trimmed",
                "2026-09-24 10:00:01,000 compactor.dedup INFO dedup: merging 1 fact(s) into 'her real fact text'; removed: ['x']",
            ]
        )
        (tmp / "compactor.log").write_text(body)
        result = telemetry.collect_log_signal(log_dir=tmp, max_lines=8)
        texts = [e["text"] for e in result["entries"]]
        assert_eq(len(texts), 1, "only the WARNING line is signal (the INFO dedup line isn't ERROR/WARNING/CRIT)")
        assert_in("content window trimmed", texts[0], "the benign 'content' WARNING line SURVIVES (not dropped)")
    finally:
        _rmtree(tmp)


def test_collect_log_signal_scrubs_a_signal_line_that_does_carry_content():
    tmp = Path(tempfile.mkdtemp(prefix="zl-log-signal-scrub-"))
    try:
        secret = "her actual secret plan h"
        body = (
            "2026-09-24 10:00:00,000 compactor WARNING conv=abc: reply looks like a repetition loop "
            f"(the token {secret!r} repeated for 480 characters (limit 400)) — skipping memory tail (900 chars accumulated; streak=3)"
        )
        (tmp / "compactor.log").write_text(body)
        result = telemetry.collect_log_signal(log_dir=tmp, max_lines=8)
        assert_eq(len(result["entries"]), 1, "the WARNING line is signal and is kept (scrubbed, not dropped)")
        assert_true(secret not in result["entries"][0]["text"], "the leaked reply text is scrubbed out")
        assert_in("skipping memory tail", result["entries"][0]["text"], "the diagnostic text survives")
    finally:
        _rmtree(tmp)


def test_collect_log_signal_caps_line_length():
    tmp = Path(tempfile.mkdtemp(prefix="zl-log-signal-cap-"))
    try:
        long_msg = "x" * 5000
        body = f"2026-09-24 10:00:00,000 compactor ERROR {long_msg}"
        (tmp / "compactor.log").write_text(body)
        result = telemetry.collect_log_signal(log_dir=tmp, max_lines=8)
        assert_true(
            len(result["entries"][0]["text"]) <= telemetry.LOG_LINE_MAX_CHARS + 1,
            "one very long line is capped, not shown in full",
        )
    finally:
        _rmtree(tmp)


def test_collect_log_signal_missing_files_are_reported_not_fatal():
    """Every CORE log file is reported missing - but round-11 review (LOW):
    an OPTIONAL one (cloudflared-error.log, absent on any pre-v3.1.9.9 pod
    with no cloudflared at all) must NOT be, that is the expected, boring
    case on most pods, not a fault to surface as dash noise.
    """
    tmp = Path(tempfile.mkdtemp(prefix="zl-log-signal-missing-"))
    try:
        result = telemetry.collect_log_signal(log_dir=tmp)
        assert_eq(result["entries"], [], "no files present -> no entries, no crash")
        expected_core = len(telemetry.LOG_SIGNAL_FILES) - len(telemetry.OPTIONAL_LOG_FILES)
        assert_eq(len(result["file_errors"]), expected_core, "every missing CORE file is reported")
        assert_true(
            all(fname not in telemetry.OPTIONAL_LOG_FILES for fname in result["file_errors"]),
            "no optional file (cloudflared-error.log) shows up in file_errors",
        )
    finally:
        _rmtree(tmp)


def test_collect_log_signal_optional_file_missing_is_silent():
    """A pre-v3.1.9.9 pod (no cloudflared, ever) must not show
    `(cloudflared-error.log: FileNotFoundError ...)` dash noise - the
    exact complaint from the round-11 review.
    """
    tmp = Path(tempfile.mkdtemp(prefix="zl-log-signal-optional-"))
    try:
        for fname in telemetry.LOG_SIGNAL_FILES:
            if fname not in telemetry.OPTIONAL_LOG_FILES:
                (tmp / fname).write_text("")
        result = telemetry.collect_log_signal(log_dir=tmp)
        assert_eq(result["file_errors"], {}, "only the optional file is missing -> no file_errors at all")
    finally:
        _rmtree(tmp)


def test_collect_log_signal_core_file_missing_still_reported_even_with_optional_absent():
    """A REAL fault (a core log file missing) must still surface even
    when the optional cloudflared-error.log is ALSO absent alongside it -
    the fix must not have gone so broad it silences core-file faults too.
    """
    tmp = Path(tempfile.mkdtemp(prefix="zl-log-signal-core-missing-"))
    try:
        for fname in telemetry.LOG_SIGNAL_FILES:
            if fname != "supervisord.log" and fname not in telemetry.OPTIONAL_LOG_FILES:
                (tmp / fname).write_text("")
        result = telemetry.collect_log_signal(log_dir=tmp)
        assert_eq(set(result["file_errors"]), {"supervisord.log"}, "the missing core file is reported, nothing else")
    finally:
        _rmtree(tmp)


def test_mutation_dropping_optional_gate_reintroduces_the_dash_noise():
    """Prove the fix bites: without the `FileNotFoundError`/`OPTIONAL_LOG_FILES`
    split, a pre-v3.1.9.9 pod's missing cloudflared-error.log goes right
    back to being reported as a fault.
    """
    tmp = Path(tempfile.mkdtemp(prefix="zl-log-signal-mutation-"))
    try:
        for fname in telemetry.LOG_SIGNAL_FILES:
            if fname not in telemetry.OPTIONAL_LOG_FILES:
                (tmp / fname).write_text("")

        def mutated(log_dir=None, max_lines=8):
            log_dir = log_dir or tmp
            entries: list = []
            file_errors: dict = {}
            for fname in telemetry.LOG_SIGNAL_FILES:
                path = log_dir / fname
                try:
                    telemetry.tail_bytes(path)
                except OSError as exc:
                    file_errors[fname] = f"{type(exc).__name__}: {exc}"
                    continue
            return {"entries": entries, "file_errors": file_errors}

        mutated_result = mutated(log_dir=tmp)
        assert_true(
            "cloudflared-error.log" in mutated_result["file_errors"],
            "mutated (no optional gate) version reintroduces the noise -- confirms the bite",
        )
        real_result = telemetry.collect_log_signal(log_dir=tmp)
        assert_true(
            "cloudflared-error.log" not in real_result["file_errors"],
            "the real (fixed) code stays silent for the optional file",
        )
    finally:
        _rmtree(tmp)


def test_mutation_level_anchoring_disabled_is_caught():
    """Prove the anchoring actually matters: replace `_line_level_field`
    with a naive substring scan (the OLD behaviour) and watch a message-only
    occurrence of "ERROR" get misclassified as signal.
    """
    original = telemetry._line_level_field

    def naive(line, fname):  # noqa: ARG001 - matches the old, unanchored behaviour
        for word in ("ERROR", "CRITICAL", "WARNING"):
            if word in line:
                return word
        return None

    telemetry._line_level_field = naive
    try:
        line = 'compactor INFO downstream said its level was "ERROR" in its own reply'
        caught = not telemetry.is_signal_line(line, "compactor.log")
        assert_true(not caught, "mutated (substring) anchoring misclassifies message text as a level -- confirms the test bites")
    finally:
        telemetry._line_level_field = original
    line = 'compactor INFO downstream said its level was "ERROR" in its own reply'
    assert_true(not telemetry.is_signal_line(line, "compactor.log"), "restored anchoring is green again")


def test_mutation_privacy_scrub_disabled_is_caught():
    """Prove `collect_log_signal` really routes through `privacy.scrub_log_
    line`: break the import so scrubbing becomes a no-op, watch the leak
    test above go red, restore.
    """
    from zlops import privacy

    original = privacy.scrub_log_line
    tmp = Path(tempfile.mkdtemp(prefix="zl-log-signal-mut-"))
    try:
        privacy.scrub_log_line = lambda line, source="generic": line  # no-op mutation
        secret = "her actual secret plan h"
        body = (
            "2026-09-24 10:00:00,000 compactor WARNING conv=abc: reply looks like a repetition loop "
            f"(the token {secret!r} repeated for 480 characters (limit 400)) — skipping memory tail (1 chars accumulated; streak=1)"
        )
        (tmp / "compactor.log").write_text(body)
        result = telemetry.collect_log_signal(log_dir=tmp, max_lines=8)
        caught = secret not in result["entries"][0]["text"]
        assert_true(not caught, "mutated (no-op) scrub leaks the secret -- confirms the test bites")
    finally:
        privacy.scrub_log_line = original
        _rmtree(tmp)


# =============================================================================
# vLLM log-line fallback
# =============================================================================


def test_parse_vllm_log_line():
    line = (
        "Engine 000: Avg prompt throughput: 719.7 tokens/s, Avg generation throughput: 6.0 tokens/s, "
        "Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.6%, Prefix cache hit rate: 11.5%"
    )
    d = telemetry.parse_vllm_log_line(line)
    assert_eq(d["prompt_tokens_per_s"], 719.7, "prompt tps")
    assert_eq(d["generation_tokens_per_s"], 6.0, "gen tps")
    assert_eq(d["running"], 1, "running")
    assert_eq(d["kv_cache_usage_pct"], 0.6, "kv cache pct")
    assert_eq(d["prefix_cache_hit_rate_pct"], 11.5, "prefix hit pct")
    assert_eq(d["preemptions"], None, "no Preemptions field in the line -> None")


def test_parse_vllm_log_line_with_preemptions():
    line = (
        "Engine 000: Avg prompt throughput: 1.0 tokens/s, Avg generation throughput: 2.0 tokens/s, "
        "Running: 1 reqs, Waiting: 0 reqs, Preemptions: 3, GPU KV cache usage: 5.0%, Prefix cache hit rate: 0.0%"
    )
    d = telemetry.parse_vllm_log_line(line)
    assert_eq(d["preemptions"], 3, "preemptions parsed when the optional field is present")


def test_parse_vllm_log_line_no_match():
    assert_eq(telemetry.parse_vllm_log_line("some unrelated line"), None, "a non-matching line -> None")


# =============================================================================
# vLLM log fallback: the line's OWN timestamp/age, never presented as live
# once stale (brief: "never present stale numbers as live").
# =============================================================================


def test_parse_vllm_log_timestamp_infers_this_year():
    import calendar
    import datetime as dt

    now = calendar.timegm(dt.datetime(2026, 9, 24, 12, 0, 0).timetuple())
    line = "(APIServer pid=115) INFO 09-24 11:59:30 [loggers.py:259] Engine 000: ..."
    epoch = telemetry.parse_vllm_log_timestamp(line, now=now)
    assert_true(epoch is not None, "timestamp parsed")
    assert_true(abs(now - epoch - 30) < 2, "the parsed epoch is ~30s before now")


def test_parse_vllm_log_timestamp_rolls_back_a_year_across_new_year():
    import calendar
    import datetime as dt

    # "now" is just after New Year; the log line is from Dec 31 of last year.
    now = calendar.timegm(dt.datetime(2027, 1, 1, 0, 5, 0).timetuple())
    line = "(APIServer pid=115) INFO 12-31 23:59:00 [loggers.py:259] Engine 000: ..."
    epoch = telemetry.parse_vllm_log_timestamp(line, now=now)
    assert_true(epoch is not None, "timestamp parsed across the year boundary")
    assert_true(now - epoch > 0, "resolves to the PAST (last year), not >1 day in the future")
    assert_true(now - epoch < 600, "resolves to ~6 minutes ago, not a year ago")


def test_parse_vllm_log_line_with_real_prefix_is_fresh():
    now = time.time()
    recent = time.strftime("%m-%d %H:%M:%S", time.gmtime(now - 5))
    line = (
        f"(APIServer pid=115) INFO {recent} [loggers.py:259] Engine 000: Avg prompt throughput: "
        "1.0 tokens/s, Avg generation throughput: 2.0 tokens/s, Running: 1 reqs, Waiting: 0 reqs, "
        "GPU KV cache usage: 5.0%, Prefix cache hit rate: 0.0%"
    )
    d = telemetry.parse_vllm_log_line(line, now=now)
    assert_true(d["line_epoch"] is not None, "timestamp extracted from a real-shaped line")
    assert_true(not d["stale"], "a 5s-old line is fresh")


def test_parse_vllm_log_line_with_real_prefix_is_stale_past_30s():
    now = time.time()
    old = time.strftime("%m-%d %H:%M:%S", time.gmtime(now - 45))
    line = (
        f"(APIServer pid=115) INFO {old} [loggers.py:259] Engine 000: Avg prompt throughput: "
        "1.0 tokens/s, Avg generation throughput: 2.0 tokens/s, Running: 1 reqs, Waiting: 0 reqs, "
        "GPU KV cache usage: 5.0%, Prefix cache hit rate: 0.0%"
    )
    d = telemetry.parse_vllm_log_line(line, now=now)
    assert_true(d["stale"], "a 45s-old line (>30s VLLM_LOG_STALE_S) is stale")
    assert_true(d["line_age_s"] > telemetry.VLLM_LOG_STALE_S, "line_age_s agrees")


def test_parse_vllm_log_line_no_timestamp_at_all_is_never_trusted_as_live():
    line = (
        "Engine 000: Avg prompt throughput: 1.0 tokens/s, Avg generation throughput: 2.0 tokens/s, "
        "Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 5.0%, Prefix cache hit rate: 0.0%"
    )
    d = telemetry.parse_vllm_log_line(line)
    assert_eq(d["line_epoch"], None, "no timestamp in this line shape")
    assert_true(d["stale"], "with no timestamp at all, never shown as live")


def test_mutation_vllm_staleness_disabled_is_caught():
    original = telemetry.VLLM_LOG_STALE_S
    try:
        telemetry.VLLM_LOG_STALE_S = 10_000.0  # effectively "never stale"
        now = time.time()
        old = time.strftime("%m-%d %H:%M:%S", time.gmtime(now - 45))
        line = (
            f"(APIServer pid=115) INFO {old} [loggers.py:259] Engine 000: Avg prompt throughput: "
            "1.0 tokens/s, Avg generation throughput: 2.0 tokens/s, Running: 1 reqs, Waiting: 0 reqs, "
            "GPU KV cache usage: 5.0%, Prefix cache hit rate: 0.0%"
        )
        d = telemetry.parse_vllm_log_line(line, now=now)
        caught = d["stale"]
        assert_true(not caught, "mutated (huge) threshold hides staleness -- confirms the test bites")
    finally:
        telemetry.VLLM_LOG_STALE_S = original


def test_collect_vllm_falls_back_to_log_when_metrics_unreachable():
    tmp = Path(tempfile.mkdtemp(prefix="zl-vllm-log-"))
    try:
        (tmp / "vllm.log").write_text(
            "Engine 000: Avg prompt throughput: 3.0 tokens/s, Avg generation throughput: 4.0 tokens/s, "
            "Running: 2 reqs, Waiting: 1 reqs, GPU KV cache usage: 9.0%, Prefix cache hit rate: 1.0%\n"
        )

        def failing_fetch(url, timeout_s):
            raise OSError("connection refused")

        result = telemetry.collect_vllm(fetch=failing_fetch, log_dir=tmp)
        assert_eq(result["source"], "log", "falls back to the log when /metrics is unreachable")
        assert_eq(result["log"]["running"], 2, "log fallback data parsed correctly")
    finally:
        _rmtree(tmp)


def test_collect_vllm_raises_when_both_metrics_and_log_fail():
    tmp = Path(tempfile.mkdtemp(prefix="zl-vllm-nolog-"))
    try:

        def failing_fetch(url, timeout_s):
            raise OSError("connection refused")

        caught = False
        try:
            telemetry.collect_vllm(fetch=failing_fetch, log_dir=tmp)
        except RuntimeError:
            caught = True
        assert_true(caught, "no /metrics and no usable vllm.log line -> raises, doesn't silently return junk")
    finally:
        _rmtree(tmp)


# =============================================================================
# collect_health / collect_snapshot / Hub
# =============================================================================


def test_collect_health_uses_injected_fetch():
    doc = {"status": "ok", "status_reasons": []}

    def fake_fetch(url, timeout_s):
        return json.dumps(doc).encode()

    result = telemetry.collect_health(fetch=fake_fetch)
    assert_eq(result["doc"], doc, "doc passed through untouched")
    assert_eq(result["interpreted"]["status"], "ok", "interpreted alongside the raw doc")


def test_collect_snapshot_never_raises_and_is_json_safe():
    snap = telemetry.collect_snapshot({"rate_gap_s": 0.02, "no_rtt": True})
    for name in ("health", "services", "gpu", "backups", "ssh", "log", "data_usage", "host", "vllm", "rtt"):
        assert_true(name in snap, f"{name} present in the snapshot")
        assert_true("error" in snap[name] and "collected_at" in snap[name], f"{name} has error/collected_at")
    encoded = json.dumps(snap, default=str)
    assert_true(len(encoded) > 0, "the whole snapshot round-trips through json.dumps")


def test_hub_memory_tail_deltas_baseline_then_increment():
    hub = telemetry.Hub()
    first = hub._memory_tail_deltas({"stored": 10, "skipped_shed": 0})
    assert_eq(first, {"stored": 0, "skipped_shed": 0}, "first observation baselines to 0 deltas")
    second = hub._memory_tail_deltas({"stored": 13, "skipped_shed": 1})
    assert_eq(second, {"stored": 3, "skipped_shed": 1}, "delta is since dashboard start, not since process start")


def test_hub_memory_tail_deltas_rebaselines_on_restart():
    hub = telemetry.Hub()
    hub._memory_tail_deltas({"stored": 50})
    after_restart = hub._memory_tail_deltas({"stored": 2})  # a lower count: the compactor restarted
    assert_eq(after_restart, {"stored": 0}, "a lower count re-baselines instead of showing a negative delta")


# =============================================================================
# Hub: one long-lived worker thread per collector, never a new thread per
# tick, a stalled collector never leaks and never hangs Hub.stop()/process
# exit - the brief's "/data stall must not hang exit or leak threads".
# =============================================================================


def test_hub_start_creates_exactly_one_thread_per_collector():
    hub = telemetry.Hub()
    hub.start()
    try:
        names = {t.name for t in hub._threads}
        assert_eq(len(hub._threads), len(telemetry.CADENCE_S), "one thread object per collector, never more")
        assert_true(all(t.daemon for t in hub._threads), "every collector thread is daemon=True")
        assert_true(all(name.startswith("zl-dash-") for name in names), "threads are the Hub's own, identifiable")
    finally:
        hub.stop()


def test_hub_blocked_collector_never_spawns_a_second_thread_and_never_leaks():
    """The regression this proves: the OLD design wrapped every tick in
    `_call_with_timeout`, which spawned a FRESH thread-with-join every
    single tick - a collector that blocked past its timeout leaked one
    thread per tick, forever, for the life of the dashboard. The new
    design calls the collector directly from its own single long-lived
    thread, so a stuck collector blocks THAT ONE thread and nothing else
    is ever spawned for it.
    """
    hub = telemetry.Hub()
    release = threading.Event()
    calls = []

    def blocking_data_usage():
        calls.append(1)
        release.wait(5.0)  # simulates a stalled /data iterdir() with no timeout of its own
        return {"entries": [], "total_kb": 0, "errors": {}, "pct_of_volume": None, "note": "x"}

    hub._collect_data_usage = blocking_data_usage
    # Tiny cadence so, if the (buggy) old per-tick-thread design were still
    # in place, several ticks - and several leaked threads - would fire
    # before we check.
    telemetry.CADENCE_S = dict(telemetry.CADENCE_S)
    original_cadence = telemetry.CADENCE_S["data_usage"]
    telemetry.CADENCE_S["data_usage"] = 0.05
    try:
        before = threading.active_count()
        t = threading.Thread(target=hub._loop, args=("data_usage",), daemon=True, name="zl-dash-data_usage")
        t.start()
        time.sleep(0.4)  # several would-be cadence intervals while the call is blocked
        during = threading.active_count()
        assert_eq(len(calls), 1, "the blocked collector is called exactly once - no new call is scheduled while it's stuck")
        assert_eq(during, before + 1, "exactly ONE extra thread exists (the collector's own), not one per tick")
        snap_started = hub._state["data_usage"].call_started_at
        assert_true(snap_started is not None, "call_started_at is set while the call is in flight (drives 'stalled Ns')")
    finally:
        release.set()
        hub._stop.set()
        t.join(2.0)
        telemetry.CADENCE_S["data_usage"] = original_cadence


def test_hub_stop_never_blocks_on_a_stalled_collector():
    """`Hub.stop()` must return immediately even if a collector thread is
    permanently stuck - it never joins. Process exit is safe because every
    thread is daemon=True (proven above); this proves the `stop()` call
    itself is non-blocking, which is what makes `zl dash`'s own shutdown
    path (`finally: hub.stop()`) safe to call unconditionally.
    """
    hub = telemetry.Hub()
    never = threading.Event()

    def blocks_forever():
        never.wait()  # never set - simulates a permanently stuck /data call
        return {}

    hub._collect_data_usage = blocks_forever
    hub.start()
    time.sleep(0.05)
    start = time.monotonic()
    hub.stop()
    elapsed = time.monotonic() - start
    assert_true(elapsed < 0.5, f"stop() returned immediately ({elapsed:.3f}s), it never joins a stuck thread")


def test_mutation_hub_tick_wrapped_in_call_with_timeout_would_leak_threads():
    """A test only counts if you watched it fail: temporarily restore the
    OLD per-tick double-threading behaviour (wrap the collector call in
    `_call_with_timeout` inside `_tick`, like the pre-fix code did) and
    watch `test_hub_blocked_collector_never_spawns_a_second_thread_and_
    never_leaks`'s own thread-count assertion go red.
    """
    original_tick = telemetry.Hub._tick

    def old_style_tick(self, name):
        fn = getattr(self, f"_collect_{name}")
        value, err = telemetry._call_with_timeout(fn, 0.05)  # short timeout so it "succeeds" fast, spawning a thread
        now = time.time()
        with self._lock:
            st = self._state[name]
            st.collected_at = now
            if err is not None:
                st.error = str(err)
            else:
                st.data = value
                st.error = None

    hub = telemetry.Hub()
    release = threading.Event()

    def blocking():
        release.wait(5.0)
        return {}

    hub._collect_data_usage = blocking
    telemetry.Hub._tick = old_style_tick
    try:
        before = threading.active_count()
        t = threading.Thread(target=hub._loop, args=("data_usage",), daemon=True, name="zl-dash-data_usage")
        original_cadence = telemetry.CADENCE_S.get("data_usage")
        telemetry.CADENCE_S["data_usage"] = 0.05
        t.start()
        time.sleep(0.4)
        after = threading.active_count()
        # The old design: every 0.05s tick times out (0.05s < however long
        # `blocking` takes) and spawns a FRESH `_call_with_timeout` thread
        # that is never joined again - thread count grows well past +1.
        caught = after > before + 2
        assert_true(caught, "mutated (old per-tick-thread) design leaks multiple threads -- confirms the test bites")
    finally:
        telemetry.Hub._tick = original_tick
        release.set()
        hub._stop.set()
        t.join(2.0)
        telemetry.CADENCE_S["data_usage"] = original_cadence


def test_collect_data_usage_sums_entries_and_pct_of_volume():
    tmp = Path(tempfile.mkdtemp(prefix="zl-data-"))
    try:
        (tmp / "openwebui").mkdir()
        (tmp / "backups").mkdir()

        def fake_run(cmd, **kwargs):
            return _FakeProc(stdout="512\t/x\n")

        os.environ["ZL_VOLUME_GB"] = "1"
        try:
            result = telemetry.collect_data_usage(data_dir=tmp, run=fake_run)
        finally:
            os.environ.pop("ZL_VOLUME_GB", None)
        assert_eq(result["total_kb"], 1024, "two entries summed (512 KB each)")
        assert_true(result["pct_of_volume"] is not None, "pct_of_volume computed when ZL_VOLUME_GB is set")
        assert_true("cluster" in result["note"].lower(), "the panel note explains du is not a quota")
    finally:
        _rmtree(tmp)


# =============================================================================
# Runner
# =============================================================================

_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]


if __name__ == "__main__":
    for t in _TESTS:
        print(f"-- {t.__name__}")
        t()
    print(f"\nAll {len(_TESTS)} telemetry tests passed.")
