#!/usr/bin/env python3
"""Measure the three hot paths v3.1.9 intends to change, before it changes them.

    docker compose -f docker-compose.bench.yml run --rm --build bench
    docker compose -f docker-compose.bench.yml run --rm bench --only B
    docker compose -f docker-compose.bench.yml run --rm bench --baseline /out/before.json

WHY THIS EXISTS. Going into v3.1.9 the tree held exactly four performance
numbers, all of them in one comment above `_redact_degenerate_turns`'s call
site, all of them stopping at 170 turns, against a production conversation
that is at ~1,150 assistant replies and growing ~2 MB/day. Optimising against
an extrapolation is how a release ships something slower than what it
replaced. Every v3.1.9 change is supposed to land with a measured
before/after, and this is the thing that measures it.

WHAT IS DELIBERATELY NOT MEASURED. There is no GPU here, no vLLM and no
model. Generation is not the subject: the subject is the compactor's own CPU
and its own disk IO, plus - for path A - the NUMBER of backend calls it
makes, which is the quantity v3.1.9.1 actually changes and the one that
multiplies out to the 117 seconds the user waits. The backend is stubbed at
zero latency and the call count is reported as its own column, so a reader
can multiply by whatever per-call latency the pod is showing that week rather
than inheriting one baked in here. Where a path genuinely cannot be measured
without a backend, this says so instead of printing a number.

THE THREE PATHS

  A  compact_if_needed            the inline compaction the user waits for.
                                  Measured BOTH ways: with a stored hierarchy
                                  to reuse (this branch's HEAD) and without
                                  (v3.1.8). The two are the same code - the
                                  diff between 9ba5903 and ced4520 shows
                                  `stored_turns` is 0 whenever there is
                                  nothing stored and `fresh_input` is then
                                  exactly `text_only`, which IS v3.1.8's
                                  argument to summarize() - so the no-state
                                  run is the v3.1.8 baseline, not an
                                  approximation of it.

  B  _redact_degenerate_turns     the whole-history scan that runs on every
                                  rollup. Three sweeps, because one sweep
                                  cannot separate turn count from reply
                                  length and that separation is the entire
                                  question. See benchfixtures/conversation.py.

  C  _is_repeat_task_traffic      a blocking summary-state read taken on the
                                  event loop inside the streaming generator's
                                  `finally`. Measured as a read, and as an
                                  event-loop stall, because those are two
                                  different costs and only the second one
                                  reaches other users' requests.

READING THE OUTPUT. Every table row carries N, the median, and the spread
(min..max), because a single sample from a shared machine is not a
measurement. Sweeps also print a log-log slope: 1.0 is linear in the swept
variable, 2.0 is quadratic. The JSON block at the end is the diffable form -
run this before a change, keep the file, and pass it back with --baseline
afterwards to get a delta column.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import statistics
import sys
import tempfile
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
# compactor/ first: main.py and its siblings import each other by bare name
# (`import summarizer`), so they have to be importable the way the process
# that ships them imports them.
for _p in (os.path.join(_ROOT, "compactor"), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Environment, set BEFORE main is imported.
#
# main.py reads its whole configuration at import time, so anything set after
# the import is ignored - the same reason scripts/run-tests.py gives one
# process to each suite. Everything here is either a stub address (nothing is
# dialled) or a scratch directory; the BUDGET constants are deliberately left
# at their production defaults, because a benchmark run against different
# budgets is a benchmark of a different program.
# ---------------------------------------------------------------------------
_STATE_DIR = os.environ.get("COMPACTOR_STORAGE_ROOT") or tempfile.mkdtemp(
    prefix="zla-bench-"
)
os.environ["COMPACTOR_STORAGE_ROOT"] = _STATE_DIR
os.environ.setdefault("MODEL_REPO", "bench-model")
os.environ.setdefault("VLLM_URL", "http://stub:8000")
os.environ["COMPACTOR_RAG_ENABLED"] = "false"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import memory  # noqa: E402

memory.ensure_storage_layout()

import main  # noqa: E402
import summarizer  # noqa: E402

from benchfixtures import conversation as fx  # noqa: E402

# RESOLVE THE TOKENIZER ONCE, then pin it.
#
# Not tidiness - correctness. `get_tokenizer` caches only a SUCCESS: on a
# miss the module global stays None, the `if _tokenizer is not None`
# short-circuit at the top never fires, and every subsequent call retries
# AutoTokenizer.from_pretrained and formats a multi-line warning.
# count_tokens is called once per request AND once per message inside
# _chunk_to_budget, so on a 2,300-message array that is 2,300 retries and
# 2,300 log lines inside one compaction. Leaving that in would put a large
# constant under every path-A number that the pod - where the tokenizer
# resolves on the first call and is cached - does not pay.
#
# The retry cost is not hidden by this, it is measured on its own in
# path_a_tokenizer_miss, which puts the real function back to do it.
_REAL_GET_TOKENIZER = main.get_tokenizer
_PINNED_TOKENIZER = _REAL_GET_TOKENIZER()
main.get_tokenizer = lambda: _PINNED_TOKENIZER


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = min(len(s) - 1, max(0, int(round((len(s) - 1) * p))))
    return s[k]


class Cell:
    """One measured (median, spread, N) with the samples kept.

    The median and not the mean, and the spread printed rather than folded
    into a standard deviation: these runs share a machine with whatever else
    is on it, so the distribution is one-sided - a sample is never faster
    than the work, only slower than it. A mean is dragged by the tail, a
    median is not, and min..max is the honest statement of how much tail
    there was.
    """

    def __init__(self, samples_ms: list[float]):
        self.samples = samples_ms
        self.n = len(samples_ms)
        self.median = statistics.median(samples_ms) if samples_ms else 0.0
        self.lo = min(samples_ms) if samples_ms else 0.0
        self.hi = max(samples_ms) if samples_ms else 0.0
        self.p90 = _pct(samples_ms, 0.90)

    def as_json(self) -> dict:
        return {
            "n": self.n,
            "median_ms": round(self.median, 4),
            "min_ms": round(self.lo, 4),
            "max_ms": round(self.hi, 4),
            "p90_ms": round(self.p90, 4),
        }


def measure(fn, *, runs: int, warmup: int = 1, budget_s: float = 8.0) -> Cell:
    """Run `fn` until `runs` samples or `budget_s` of wall clock, min 3.

    The budget exists so the slow cells at the top of a sweep cannot turn a
    baseline into a coffee break - a benchmark nobody re-runs after each
    change is not a benchmark, it is a one-off measurement with extra steps.
    N is reported per row so a 3-sample cell is never mistaken for a
    7-sample one.
    """
    for _ in range(warmup):
        fn()
    out: list[float] = []
    started = time.perf_counter()
    while len(out) < runs:
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1000.0)
        if len(out) >= 3 and (time.perf_counter() - started) > budget_s:
            break
    return Cell(out)


def loglog_slope(xs: list[float], ys: list[float]) -> float | None:
    """Least-squares slope of log(y) against log(x).

    This is the whole "is it linear or is it super-linear" answer in one
    number, which is why it is printed next to every sweep rather than left
    for the reader to eyeball off four rows. 1.0 means doubling the swept
    variable doubles the time; 2.0 means it quadruples it.
    """
    pts = [
        (x, y) for x, y in zip(xs, ys) if x > 0 and y > 0
    ]
    if len(pts) < 2:
        return None
    import math

    lx = [math.log(x) for x, _ in pts]
    ly = [math.log(y) for _, y in pts]
    mx = sum(lx) / len(lx)
    my = sum(ly) / len(ly)
    den = sum((x - mx) ** 2 for x in lx)
    if den == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(lx, ly)) / den


# ---------------------------------------------------------------------------
# The environment the numbers were taken in.
#
# Printed first and stored in the JSON because the single most common way a
# before/after comparison lies is that the two halves ran on different
# machines - or, on this project, in different CONTAINERS, where the cgroup
# CPU quota is the number that matters and `nproc` is not.
# ---------------------------------------------------------------------------

def _read(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read().strip()
    except OSError:
        return ""


def _cpu_model() -> str:
    for line in _read("/proc/cpuinfo").splitlines():
        if line.lower().startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _cgroup_cpu() -> str:
    v2 = _read("/sys/fs/cgroup/cpu.max")
    if v2:
        return f"cgroup v2 cpu.max={v2}"
    quota = _read("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    period = _read("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if quota:
        return f"cgroup v1 quota={quota} period={period}"
    return "no cgroup cpu limit found"


def _fs_of(path: str) -> str:
    """Which filesystem the summary state actually lands on.

    Path C is about a disk read, and the production volume is MooseFS. A
    number taken on overlayfs and a number taken on a network mount are not
    comparable, so the mount type travels with the measurement.
    """
    best = ("", "")
    for line in _read("/proc/mounts").splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        mnt, typ = parts[1], parts[2]
        if path.startswith(mnt) and len(mnt) >= len(best[0]):
            best = (mnt, typ)
    return f"{best[1] or 'unknown'} at {best[0] or '?'}"


def environment(label: str) -> dict:
    code = ""
    try:
        with open(os.path.join(_ROOT, "compactor", "main.py"), "rb") as fh:
            code = hashlib.sha256(fh.read()).hexdigest()[:12]
    except OSError:
        pass
    try:
        affinity = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        affinity = os.cpu_count() or 0
    tok = main.get_tokenizer()
    return {
        "label": label,
        "when": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count() or 0,
        "cpu_affinity": affinity,
        "cgroup_cpu": _cgroup_cpu(),
        "in_container": os.path.exists("/.dockerenv"),
        # CONTENTION, on the record. This engine is shared - another agent
        # taking it for a CPU-inference run while a sweep is in flight would
        # inflate every row and nothing in the output would say so. Sampled
        # at the start and again at the end (see `loadavg_end`); if the two
        # differ much, or either is a large fraction of cpu_count, the run
        # was not taken on a quiet machine and the numbers are ceilings
        # rather than measurements.
        "loadavg_start": _read("/proc/loadavg")[:14],
        "state_dir": _STATE_DIR,
        "state_fs": _fs_of(_STATE_DIR),
        "main_py_sha256_12": code,
        # WHICH COUNTER. count_tokens has a four-tier ladder and the tier in
        # effect changes path A's CPU cost by an order of magnitude. On the
        # pod the transformers tokenizer loads (and its chat template does
        # not, so it runs per-message encode+4); in an offline container
        # with no weights there is no tokenizer at all and the estimator is
        # chars/4. Saying which one ran is the difference between a number
        # and a number-shaped thing.
        "count_tokens_tier": (
            "transformers tokenizer" if tok is not None else "chars/4 estimator"
        ),
        "hierarchical_summary_enabled": summarizer.enabled(),
        "config": {
            "MAX_MODEL_LEN": main.MAX_MODEL_LEN,
            "GENERATION_RESERVE": main.GENERATION_RESERVE,
            "HARD_INPUT_LIMIT": main.HARD_INPUT_LIMIT,
            "TARGET_TOKENS": main.TARGET_TOKENS,
            "KEEP_RECENT_TURNS": main.KEEP_RECENT_TURNS,
            "MAX_SUMMARY_CALLS_PER_REQUEST": main.MAX_SUMMARY_CALLS_PER_REQUEST,
            "SUMMARY_MAX_TOKENS": main.SUMMARY_MAX_TOKENS,
            "L1_CHUNK_SIZE": summarizer.L1_CHUNK_SIZE,
            "FINGERPRINT_TAIL_TURNS": summarizer._FINGERPRINT_TAIL_TURNS,
        },
    }


# ---------------------------------------------------------------------------
# Backend stub
# ---------------------------------------------------------------------------

class Backend:
    """Stands in for vLLM and COUNTS what it was asked to do.

    The call count is the headline for path A, not the wall clock: on the pod
    each of these is tens of seconds of generation, and v3.1.9.1's whole
    claim is that it issues fewer of them. A stub that returned instantly and
    was not counted would make the optimisation look like it did nothing.

    `latency_ms` is available but defaults to 0 so the reported wall clock is
    the compactor's OWN cost with nothing modelled mixed into it. The
    modelled end-to-end is computed separately, from the call count, and is
    labelled as modelled everywhere it appears.
    """

    def __init__(self, latency_ms: float = 0.0, scale: float = 1.5):
        self.latency_ms = latency_ms
        self.scale = scale
        self.reset()

    def reset(self) -> None:
        self.calls = 0
        self.turns_sent = 0
        self.chars_sent = 0

    async def summarize_once(self, client, turns: list[dict]) -> str:
        self.calls += 1
        self.turns_sent += len(turns)
        self.chars_sent += sum(len(main._message_text(m)) for m in turns)
        if self.latency_ms:
            await asyncio.sleep(self.latency_ms / 1000.0)
        # ~1,000 tokens of prose, which is what SUMMARY_MAX_TOKENS permits.
        # The length is load-bearing for the reduce phase: it re-chunks the
        # partials to the same budget, so a one-line stub would make a
        # multi-round reduce impossible to observe.
        return "SUMMARY. " + ("summarised context. " * 200)

    def count_tokens_exact(self, messages, add_generation_prompt=None):
        """Stand in for vLLM's /tokenize.

        NOT None. Returning None is the degraded path - it puts summarize()
        on the pessimistic 2.0x fallback, which over-splits batches and would
        make this benchmark measure the outage rather than the steady state.
        `scale` is the ratio the pod actually shows between the local
        estimator and vLLM's own count on this model's content (the local one
        reads 23-51% low on assistant text); it is a declared parameter, and
        it is the only modelled quantity inside a measured number here.
        """
        if not messages:
            return 0
        return int(main.count_tokens(messages) * self.scale)


# ---------------------------------------------------------------------------
# Path A - compact_if_needed
# ---------------------------------------------------------------------------

def _store_state(conv_id: str, state: dict) -> None:
    summarizer.save_state(conv_id, state)


def path_a(args, backend: Backend) -> dict:
    """The inline compaction the user waits for, three ways per fixture size.

      v3.1.8 (no reuse)        conv_id=None. This is not an approximation of
                               v3.1.8: the diff between 9ba5903 and ced4520
                               makes `stored_turns` 0 and `fresh_input`
                               exactly `text_only` on this branch, which is
                               v3.1.8's whole argument to summarize().
      v3.1.9.1 (lagging)       the hierarchy covers only the oldest span -
                               the shape the pod was actually in when the
                               117 seconds was measured (40 of 56 covered).
      v3.1.9.1 (caught up)     the hierarchy covers everything but the last
                               chunk-and-a-bit, which is where maybe_rollup
                               sits when nothing has frozen it. This is the
                               case the optimisation is FOR, and it is the
                               one that changes what compaction does rather
                               than only how long it takes.

    `did_nothing` is as important as the timing. Over
    MAX_SUMMARY_CALLS_PER_REQUEST batches, summarize() deliberately refuses
    and hands everything back verbatim, so a long conversation gets ZERO
    backend calls and ZERO compaction, and the hard-budget guard downstream
    sheds turns instead. A row with 0 calls is therefore either very fast or
    completely ineffective, and only this column says which.
    """
    rows = []
    # One loop for the whole path. asyncio.run() builds and tears down an
    # event loop per call, which is ~1 ms of pure harness that the pod - one
    # long-lived loop - never pays. Measuring it would put a floor under
    # every path-A number and hide exactly the differences being looked for.
    loop = asyncio.new_event_loop()
    try:
        for n_msgs in args.a_sizes:
            # n_msgs counts messages including the system turn, so sizes can
            # be stated the way the pod states them ("61 messages").
            n_ex = max(1, (n_msgs - 1) // 2)
            msgs = fx.conversation(
                n_ex, spread=True, assistant_chars=args.assistant_chars
            )
            chars = sum(len(main._message_text(m)) for m in msgs)
            local = main.count_tokens(msgs)
            n_nonsystem = len([m for m in msgs if m.get("role") != "system"])
            _, to_summarize, _keep = main.split_messages(msgs)
            chunk = summarizer.L1_CHUNK_SIZE

            def _state_for(name: str, covered: int) -> str | None:
                covered = (covered // chunk) * chunk
                if covered <= 0:
                    return None
                conv = f"bench_a_{name}_{n_ex}"
                # The tiers are bounded by construction (l1 drops ten chunks
                # at a time into an l2 chapter, an L3 refresh empties l2), so
                # a realistic long-conversation state is ten L1 chunks, a few
                # chapters and one theme - NOT one chunk per twenty turns.
                # Writing the unbounded shape would inflate every read in
                # path C and every format_summary_block here.
                n_l1 = min(10, max(1, covered // chunk))
                _store_state(
                    conv,
                    fx.summary_state(
                        conv,
                        l1_chunks=n_l1,
                        l2_chapters=5 if covered > 10 * chunk else 0,
                        l3=covered > 10 * chunk,
                        last_turn=covered,
                    ),
                )
                return conv

            # HEALTHY means the watermark is one unrolled chunk behind the
            # array, because that is what maybe_rollup actually converges to:
            # _needs_l1_rollup fires as soon as (position - watermark) >=
            # L1_CHUNK_SIZE and the drain loops, so the steady state is a
            # remainder under one chunk. FROZEN@N is the other shape this
            # release keeps finding - a hierarchy that stopped advancing
            # early (a looping model froze it for 14 consecutive turns in the
            # soak, watermark still 0) and never recovered. At small message
            # counts the two coincide, and the 'cover' column says so rather
            # than the row pretending to be two measurements.
            healthy = ((n_nonsystem - chunk) // chunk) * chunk
            variants = [
                ("v3.1.8 (no reuse)", None),
                (f"v3.1.9.1 frozen@{args.a_covered_turns}",
                 _state_for("lag", min(args.a_covered_turns, len(to_summarize)))),
                ("v3.1.9.1 healthy",
                 _state_for("full", max(0, healthy))),
            ]

            for variant, conv in variants:
                seen: dict = {}

                def once(conv=conv, seen=seen):
                    backend.reset()
                    out = loop.run_until_complete(
                        main.compact_if_needed(list(msgs), conv)
                    )
                    seen["calls"] = backend.calls
                    seen["turns_sent"] = backend.turns_sent
                    seen["out_tokens"] = main.count_tokens(out)
                    seen["out_msgs"] = len(out)

                cell = measure(once, runs=args.runs, budget_s=args.cell_budget_s)
                out_tokens = seen.get("out_tokens", 0)
                rows.append({
                    "messages": len(msgs),
                    "variant": variant,
                    "chars": chars,
                    "local_tokens": local,
                    "modelled_vllm_tokens": int(local * backend.scale),
                    "offered_turns": len(to_summarize),
                    "stored_covers_turns": _covered_of(conv),
                    "backend_calls": seen.get("calls", 0),
                    "turns_sent_to_backend": seen.get("turns_sent", 0),
                    "out_tokens": out_tokens,
                    "out_messages": seen.get("out_msgs", 0),
                    # Did compaction achieve anything at all? Over the call
                    # cap it returns the input untouched, which is fast and
                    # useless, and the guard sheds afterwards.
                    "did_nothing": out_tokens >= local,
                    "still_over_target": out_tokens > main.TARGET_TOKENS,
                    "cell": cell.as_json(),
                    "modelled_end_to_end_s": round(
                        seen.get("calls", 0) * args.llm_call_ms / 1000.0
                        + cell.median / 1000.0,
                        2,
                    ),
                })
    finally:
        loop.close()
    return {"rows": rows}


def _covered_of(conv: str | None) -> int:
    if not conv:
        return 0
    try:
        return summarizer._highest_chunk_turn(summarizer.load_state(conv))
    except Exception:
        return 0


def path_a_tokenizer_miss(args) -> dict:
    """What count_tokens costs when the tokenizer will not load.

    Not an idle curiosity, and not this benchmark's own artefact: on a MISS
    `get_tokenizer` leaves the module global at None, so the guard at the top
    of it (`if _tokenizer is not None`) never short-circuits and every single
    call retries `AutoTokenizer.from_pretrained` and formats a warning.
    count_tokens is called once per request plus ONCE PER MESSAGE inside
    _chunk_to_budget, so on a long history that is hundreds of retries per
    compaction. It is latent on the pod because the tokenizer does load - and
    it is exactly the shape that turns a tokenizer-cache problem into a
    latency outage, so the cost is worth having on the record.
    """
    msgs = fx.conversation(30, spread=True)
    pinned = measure(lambda: main.count_tokens(msgs), runs=args.runs, budget_s=3.0)
    saved = main.get_tokenizer
    try:
        # The real function, with the module global still None: exactly the
        # state a pod whose tokenizer will not load is in.
        main.get_tokenizer = _REAL_GET_TOKENIZER
        natural = measure(
            lambda: main.count_tokens(msgs), runs=max(3, args.runs), budget_s=6.0
        )
    finally:
        main.get_tokenizer = saved
    return {
        "messages": len(msgs),
        "tokenizer_loaded": _PINNED_TOKENIZER is not None,
        "with_retry_on_every_call": natural.as_json(),
        "short_circuited": pinned.as_json(),
    }


# ---------------------------------------------------------------------------
# Path B - _redact_degenerate_turns
# ---------------------------------------------------------------------------

def _sweep_turns(args, *, growth: bool, sizes: list[int], tag: str,
                 assistant_chars: int) -> dict:
    biggest = fx.conversation(
        max(sizes), assistant_chars=assistant_chars, growth=growth
    )
    rows = []
    for n in sizes:
        # A PREFIX of the same conversation, not a fresh fixture: the
        # generator is one seeded RNG consumed in order, so turn i is the
        # same text at every n. Regenerating per size would let content vary
        # with n, which is the confound this whole sweep exists to remove.
        msgs = biggest[: 1 + 2 * n]
        chars = sum(
            len(main._message_text(m)) for m in msgs if m.get("role") == "assistant"
        )
        flagged = sum(
            1
            for m in msgs
            if m.get("role") == "assistant"
            and main.reply_is_degenerate(main._message_text(m))
        )
        cell = measure(
            lambda msgs=msgs: main._redact_degenerate_turns(msgs),
            runs=args.runs,
            budget_s=args.cell_budget_s,
        )
        rows.append({
            "turns": n,
            "assistant_chars": chars,
            "mean_reply_chars": chars // max(1, n),
            "flagged": flagged,
            "cell": cell.as_json(),
            "ms_per_turn": round(cell.median / max(1, n), 4),
            "us_per_kchar": round(cell.median * 1000.0 / max(1, chars / 1000.0), 3),
        })
    return {
        "tag": tag,
        "rows": rows,
        "slope_vs_turns": loglog_slope(
            [r["turns"] for r in rows], [r["cell"]["median_ms"] for r in rows]
        ),
    }


def path_b(args) -> dict:
    out: dict = {}

    # B1. Turn count, FIXED reply length. If the claimed super-linearity is
    # real it has to show up here, because this is the only sweep in which
    # nothing but the turn count moves.
    out["B1_turns_fixed_length"] = _sweep_turns(
        args, growth=False, sizes=args.b_turn_sizes, tag="turns @ fixed reply length",
        assistant_chars=args.assistant_chars,
    )

    # B2. Reply length at a FIXED turn count of 170 - the largest of the four
    # numbers in the comment, so the two sweeps meet at a point the existing
    # claim also covers.
    rows = []
    for ln in args.b_length_sizes:
        msgs = fx.conversation(170, assistant_chars=ln)
        chars = sum(
            len(main._message_text(m)) for m in msgs if m.get("role") == "assistant"
        )
        flagged = sum(
            1
            for m in msgs
            if m.get("role") == "assistant"
            and main.reply_is_degenerate(main._message_text(m))
        )
        cell = measure(
            lambda msgs=msgs: main._redact_degenerate_turns(msgs),
            runs=args.runs,
            budget_s=args.cell_budget_s,
        )
        rows.append({
            "reply_chars": ln,
            "turns": 170,
            "assistant_chars": chars,
            "flagged": flagged,
            "cell": cell.as_json(),
            "ms_per_turn": round(cell.median / 170.0, 4),
        })
    out["B2_length_fixed_turns"] = {
        "tag": "reply length @ 170 turns",
        "rows": rows,
        "slope_vs_length": loglog_slope(
            [r["reply_chars"] for r in rows], [r["cell"]["median_ms"] for r in rows]
        ),
    }

    # B3. The suspected artefact, reproduced on purpose: reply length rising
    # with position, normalised so that at 170 turns this fixture and B1's
    # push the SAME total volume through the detector. If B3 bends and B1
    # does not, the bend is the fixture.
    out["B3_growth_artefact"] = _sweep_turns(
        args, growth=True, sizes=args.b_growth_sizes,
        tag="turns @ reply length rising with position",
        assistant_chars=args.assistant_chars,
    )

    # B4. The planned fix, measured rather than assumed: scan only the turns
    # the rollup can actually consume - L1_CHUNK_SIZE (20) plus
    # _FINGERPRINT_TAIL_TURNS (64) = 84 turns, i.e. the last 168 messages.
    bound = summarizer.L1_CHUNK_SIZE + summarizer._FINGERPRINT_TAIL_TURNS
    biggest = fx.conversation(
        max(args.b_turn_sizes), assistant_chars=args.assistant_chars
    )
    rows = []
    for n in args.b_turn_sizes:
        msgs = biggest[: 1 + 2 * n]
        tail = msgs[-(2 * bound):] if n > bound else msgs
        cell = measure(
            lambda tail=tail: main._redact_degenerate_turns(tail),
            runs=args.runs,
            budget_s=args.cell_budget_s,
        )
        rows.append({
            "turns": n,
            "scanned_turns": min(n, bound),
            "cell": cell.as_json(),
        })
    out["B4_bounded_scan"] = {
        "tag": f"proposed fix: scan only the last {bound} turns",
        "bound_turns": bound,
        "rows": rows,
    }
    return out


# ---------------------------------------------------------------------------
# Path C - the blocking state read on the event loop
# ---------------------------------------------------------------------------

async def _tick_probe(seconds: float, work=None, every_ms: float = 50.0) -> list[float]:
    """How late a 1 ms timer runs while `work` is being done on the same loop.

    This is what "blocking the event loop" costs measured in the only unit
    that matters to a second user: how long their coroutine sat ready and
    unscheduled. A ticker that is supposed to wake every millisecond and
    wakes 40 ms late has told you that somebody else's request waited 40 ms,
    and it has told you without needing a second HTTP client in the picture.
    """
    lateness: list[float] = []
    stop = False
    target = 0.001

    async def ticker():
        prev = time.perf_counter()
        while not stop:
            await asyncio.sleep(target)
            now = time.perf_counter()
            lateness.append((now - prev - target) * 1000.0)
            prev = now

    async def driver():
        deadline = time.perf_counter() + seconds
        while time.perf_counter() < deadline:
            await asyncio.sleep(every_ms / 1000.0)
            if work is not None:
                work()

    t = asyncio.create_task(ticker())
    await driver()
    stop = True
    await asyncio.sleep(0.005)
    t.cancel()
    try:
        await t
    except asyncio.CancelledError:
        pass
    return lateness


def path_c(args) -> dict:
    conv = "bench_c_state"
    state = fx.summary_state(conv, last_turn=2300)
    _store_state(conv, state)
    path = summarizer.summary_path(conv)
    size = os.path.getsize(path)

    small = "bench_c_small"
    _store_state(
        small,
        fx.summary_state(small, l1_chunks=1, l2_chapters=0, l3=False,
                         tail_fp=8, last_turn=20),
    )

    out: dict = {
        "state_file": str(path),
        "state_bytes_full_tiers": size,
        "state_bytes_small": os.path.getsize(summarizer.summary_path(small)),
        "state_fs": _fs_of(str(path)),
    }

    # C1. The read itself.
    out["C1_load_state"] = {
        "full_tiers": measure(
            lambda: summarizer.load_state(conv), runs=max(args.runs, 21),
            budget_s=args.cell_budget_s
        ).as_json(),
        "small": measure(
            lambda: summarizer.load_state(small), runs=max(args.runs, 21),
            budget_s=args.cell_budget_s
        ).as_json(),
        "save_state_full_tiers": measure(
            lambda: summarizer.save_state(conv, dict(state)),
            runs=max(args.runs, 11), budget_s=args.cell_budget_s
        ).as_json(),
    }

    # C2. The guarded function, both branches. The docstring claims the cost
    # is "never on the hot path of an ongoing conversation", and that claim
    # is checkable: _has_conversational_history short-circuits before the
    # read, so a conversation array should cost nothing and a history-less
    # one (OpenWebUI's title/tag/follow-up traffic) should cost a read.
    real_msgs = fx.conversation(30, spread=True)
    task_msgs = [{"role": "user", "content": "Generate a concise title."}]
    out["C2_is_repeat_task_traffic"] = {
        "with_history_short_circuit": measure(
            lambda: main._is_repeat_task_traffic(conv, real_msgs),
            runs=max(args.runs, 21), budget_s=args.cell_budget_s
        ).as_json(),
        "history_less_reads_disk": measure(
            lambda: main._is_repeat_task_traffic(conv, task_msgs),
            runs=max(args.runs, 21), budget_s=args.cell_budget_s
        ).as_json(),
        "history_less_answer": main._is_repeat_task_traffic(conv, task_msgs),
        "with_history_answer": main._is_repeat_task_traffic(conv, real_msgs),
    }

    # C3. Event-loop stall. Baseline first, then the same loop with the real
    # read on it, then with an injected stall standing in for a slow volume.
    # The injected values are NOT measurements of MooseFS - nothing here can
    # measure MooseFS - they are the transfer function: given a read that
    # takes D, this is what every other request on the loop pays.
    async def stalls() -> dict:
        res: dict = {}
        base = await _tick_probe(args.stall_seconds, None)
        res["idle_loop"] = _lateness_json(base)
        res["real_read_every_50ms"] = _lateness_json(
            await _tick_probe(
                args.stall_seconds, lambda: main._is_repeat_task_traffic(conv, task_msgs)
            )
        )
        # THE WRITE, NOT THE READ, and this is the row that reorders the
        # priorities. _is_repeat_task_traffic is one read taken only on
        # history-less requests; maybe_rollup's save_state is a tempfile +
        # fsync + rename taken on the event loop on EVERY turn that moves the
        # position, which is every turn. Measuring only what the task asked
        # for would have left the larger of the two unmeasured.
        res["real_save_state_every_50ms"] = _lateness_json(
            await _tick_probe(
                args.stall_seconds, lambda: summarizer.save_state(conv, dict(state))
            )
        )
        # The whole per-turn sequence the background tail takes on the loop:
        # _rollup_hierarchy's load, maybe_rollup's load, maybe_rollup's save.
        res["per_turn_tail_sequence_every_50ms"] = _lateness_json(
            await _tick_probe(args.stall_seconds, lambda: (
                summarizer.load_state(conv),
                summarizer.load_state(conv),
                summarizer.save_state(conv, dict(state)),
            ))
        )
        res["injected"] = []
        for d_ms in args.stall_injections:
            lat = await _tick_probe(
                args.stall_seconds,
                lambda d=d_ms: time.sleep(d / 1000.0),
            )
            res["injected"].append({"stall_ms": d_ms, **_lateness_json(lat)})
        # And the fix, measured rather than asserted: the same stall taken
        # off the loop. If this row is not at baseline, run_in_threadpool is
        # not the answer and v3.1.9 needs a different one.
        worst = max(args.stall_injections)

        async def offloaded():
            from starlette.concurrency import run_in_threadpool

            return await run_in_threadpool(time.sleep, worst / 1000.0)

        holder: dict = {}

        def kick():
            holder["t"] = asyncio.ensure_future(offloaded())

        res["injected_offloaded"] = {
            "stall_ms": worst,
            **_lateness_json(await _tick_probe(args.stall_seconds, kick)),
        }
        return res

    out["C3_event_loop_stall"] = asyncio.run(stalls())

    # C4. How many blocking store operations the per-turn background tail
    # takes ON THE EVENT LOOP. This is the count, not a timing: the timings
    # are C1's. It is here because _is_repeat_task_traffic is not the only
    # one, and an optimisation aimed only at it would move about a third of
    # the cost.
    out["C4_blocking_store_ops_per_turn"] = _blocking_op_inventory()
    return out


def _lateness_json(xs: list[float]) -> dict:
    return {
        "samples": len(xs),
        "median_ms": round(statistics.median(xs), 4) if xs else 0.0,
        "p99_ms": round(_pct(xs, 0.99), 4),
        "max_ms": round(max(xs), 4) if xs else 0.0,
    }


def _blocking_op_inventory() -> list[dict]:
    """Where the per-turn tail touches the store synchronously on the loop.

    Hand-verified against this tree rather than derived at runtime, because
    the thing worth recording is which CALL SITE it is - a runtime counter
    would say "three reads" and leave the next reader to find them again.
    bgwork.pool.submit uses asyncio.create_task, so everything the rollup
    coroutine does runs on the event loop, not in a worker.
    """
    return [
        {
            "site": "main._is_repeat_task_traffic -> summarizer.load_state",
            "called_from": "_run_memory_tail, in the streaming generator's finally",
            "per_turn": "only when the array carries NO assistant turn",
            "op": "read",
        },
        {
            "site": "main._rollup_hierarchy -> summarizer.load_state (before)",
            "called_from": "_fire_and_forget -> bgwork.pool.submit -> "
                           "asyncio.create_task (event loop)",
            "per_turn": "every turn the tail runs",
            "op": "read",
        },
        {
            "site": "summarizer.maybe_rollup -> load_state",
            "called_from": "same coroutine, under conv_lock",
            "per_turn": "every turn the tail runs",
            "op": "read",
        },
        {
            "site": "summarizer.maybe_rollup -> save_state (when changed)",
            "called_from": "same coroutine, under conv_lock",
            "per_turn": "every turn that advances the position or the anchor",
            "op": "write (tempfile + fsync + rename)",
        },
    ]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _eff(r: dict) -> str:
    """no-op / part / ok, and the distinction is the whole point.

    "Fast" and "did nothing" look identical in a wall-clock column. The
    over-cap branch of summarize() returns the input untouched in
    milliseconds, which is the fastest row in the table and the worst
    outcome in it - the guard downstream then sheds real turns out of her
    conversation. A benchmark that only printed milliseconds would rank it
    first.
    """
    if r.get("did_nothing"):
        return "no-op"
    return "part" if r.get("still_over_target") else "ok"


def _fmt(cell: dict) -> str:
    return (
        f"{cell['median_ms']:>10.3f}  "
        f"({cell['min_ms']:.3f}..{cell['max_ms']:.3f}, N={cell['n']})"
    )


def _delta(now: float, before: float | None) -> str:
    if before is None or before <= 0:
        return ""
    pct = (now - before) / before * 100.0
    return f"   [{pct:+.1f}% vs baseline]"


def report(result: dict, baseline: dict | None) -> None:
    env = result["env"]
    p = result["params"]
    print("=" * 78)
    print("v3.1.9 COMPACTION BASELINE")
    print("=" * 78)
    print(f"  label            {env['label']}")
    print(f"  when             {env['when']}")
    print(f"  python           {env['python']}  on  {env['platform']}")
    print(f"  cpu              {env['cpu_model']}")
    print(f"                   {env['cpu_count']} online, "
          f"{env['cpu_affinity']} usable, {env['cgroup_cpu']}")
    print(f"  container        {env['in_container']}")
    print(f"  load average     {env.get('loadavg_start', '?')} at start, "
          f"{env.get('loadavg_end', '?')} at end "
          f"(over {env.get('wall_seconds', 0)} s)")
    print(f"  state dir        {env['state_dir']}  ({env['state_fs']})")
    print(f"  compactor/main.py sha256[:12]  {env['main_py_sha256_12']}")
    print(f"  token counter    {env['count_tokens_tier']}")
    print(f"  config           TARGET_TOKENS={env['config']['TARGET_TOKENS']} "
          f"HARD_INPUT_LIMIT={env['config']['HARD_INPUT_LIMIT']} "
          f"MAX_SUMMARY_CALLS={env['config']['MAX_SUMMARY_CALLS_PER_REQUEST']}")
    print(f"  fixture          seed={p['seed']} median reply="
          f"{p['assistant_chars']} chars, runs<={p['runs']}, "
          f"cell budget={p['cell_budget_s']}s")
    print(f"  backend          STUBBED, zero latency; /tokenize modelled at "
          f"{p['tokenize_scale']}x local")
    if baseline:
        print(f"  baseline         {baseline['env']['label']} "
              f"({baseline['env']['when']}, main.py "
              f"{baseline['env']['main_py_sha256_12']})")
        if baseline["env"]["cpu_model"] != env["cpu_model"]:
            print("  !! BASELINE WAS TAKEN ON A DIFFERENT CPU - the delta "
                  "column is not a measurement")
    print()

    b_index = {}
    if baseline:
        b_index = _index(baseline)

    if "A" in result["paths"]:
        a = result["paths"]["A"]
        print("-" * 78)
        print("A. compact_if_needed - the compaction the user waits for")
        print("-" * 78)
        print(f"{'msgs':>5} {'variant':<21} {'in tok':>8} {'offer':>6} "
              f"{'cover':>6} {'LLM':>4} {'sent':>5} {'out tok':>8} {'eff':>4} "
              f"{'compactor ms (median)':>32}")
        for r in a["rows"]:
            key = f"A|{r['messages']}|{r['variant']}"
            print(f"{r['messages']:>5} {r['variant']:<21} "
                  f"{r['local_tokens']:>8} {r['offered_turns']:>6} "
                  f"{r['stored_covers_turns']:>6} "
                  f"{r['backend_calls']:>4} {r['turns_sent_to_backend']:>5} "
                  f"{r['out_tokens']:>8} "
                  f"{_eff(r):>5} "
                  f"{_fmt(r['cell']):>32}"
                  f"{_delta(r['cell']['median_ms'], b_index.get(key))}")
        print()
        print("  'cover'  turns the stored hierarchy already covers")
        print("  'LLM'    backend calls issued; 'sent' turns handed to them")
        print("  'eff'    what compaction achieved:")
        print("             no-op  it refused (over the "
              "MAX_SUMMARY_CALLS cap) and returned")
        print("                    the input untouched - fast, and useless; the "
              "hard-budget")
        print("                    guard sheds turns downstream instead")
        print("             part   it shrank the payload but it is STILL over "
              "TARGET_TOKENS")
        print("             ok     under TARGET_TOKENS")
        print(f"  Modelled end-to-end at "
              f"{p['llm_call_ms']/1000:.1f}s per backend call "
              f"(ARITHMETIC, not a measurement):")
        for r in a["rows"]:
            print(f"    {r['messages']:>5} msgs  {r['variant']:<21} "
                  f"{r['modelled_end_to_end_s']:>8.2f} s"
                  + ("   [but compaction did nothing]" if r["did_nothing"] else ""))
        print()
        tm = result["paths"].get("A_tokenizer_miss")
        if tm:
            print("  count_tokens when the tokenizer will not load "
                  f"({tm['messages']} messages, tokenizer_loaded="
                  f"{tm['tokenizer_loaded']}):")
            print(f"    retrying from_pretrained every call  "
                  f"{_fmt(tm['with_retry_on_every_call'])}")
            print(f"    short-circuited                      "
                  f"{_fmt(tm['short_circuited'])}")
            print()

    if "B" in result["paths"]:
        b = result["paths"]["B"]
        print("-" * 78)
        print("B. _redact_degenerate_turns - the whole-history scan, every rollup")
        print("-" * 78)
        for key in ("B1_turns_fixed_length", "B3_growth_artefact"):
            s = b[key]
            print(f"  {key}  [{s['tag']}]")
            print(f"    {'turns':>6} {'mean reply':>11} {'ms/turn':>9} "
                  f"{'total ms (median)':>34}")
            for r in s["rows"]:
                bk = f"{key}|{r['turns']}"
                print(f"    {r['turns']:>6} {r['mean_reply_chars']:>11} "
                      f"{r['ms_per_turn']:>9.4f} {_fmt(r['cell']):>34}"
                      f"{_delta(r['cell']['median_ms'], b_index.get(bk))}")
            sl = s["slope_vs_turns"]
            print(f"    log-log slope vs turns: "
                  f"{'n/a' if sl is None else f'{sl:.2f}'}   "
                  f"(1.00 = linear, 2.00 = quadratic)")
            flagged = sum(r["flagged"] for r in s["rows"])
            print(f"    replies the detector flagged: {flagged} "
                  f"(must be 0, or a shorter code path was measured)")
            print()
        s = b["B2_length_fixed_turns"]
        print(f"  B2_length_fixed_turns  [{s['tag']}]")
        print(f"    {'reply chars':>12} {'ms/turn':>9} {'total ms (median)':>34}")
        for r in s["rows"]:
            bk = f"B2|{r['reply_chars']}"
            print(f"    {r['reply_chars']:>12} {r['ms_per_turn']:>9.4f} "
                  f"{_fmt(r['cell']):>34}"
                  f"{_delta(r['cell']['median_ms'], b_index.get(bk))}")
        sl = s["slope_vs_length"]
        print(f"    log-log slope vs reply length: "
              f"{'n/a' if sl is None else f'{sl:.2f}'}")
        print()
        s = b["B4_bounded_scan"]
        print(f"  B4_bounded_scan  [{s['tag']}]")
        print(f"    {'history':>8} {'scanned':>8} {'total ms (median)':>34}")
        for r in s["rows"]:
            print(f"    {r['turns']:>8} {r['scanned_turns']:>8} "
                  f"{_fmt(r['cell']):>34}")
        print()

    if "C" in result["paths"]:
        c = result["paths"]["C"]
        print("-" * 78)
        print("C. the blocking summary-state read on the event loop")
        print("-" * 78)
        print(f"  state file        {c['state_file']}")
        print(f"  bytes             {c['state_bytes_full_tiers']} at full tier "
              f"fill, {c['state_bytes_small']} for a young conversation")
        print(f"  filesystem        {c['state_fs']}")
        print()
        print("  C1 the read/write themselves")
        for k, v in c["C1_load_state"].items():
            print(f"    {k:<22} {_fmt(v)}")
        print()
        print("  C2 _is_repeat_task_traffic, both branches")
        cc = c["C2_is_repeat_task_traffic"]
        print(f"    with history (short-circuit)  {_fmt(cc['with_history_short_circuit'])}"
              f"  -> {cc['with_history_answer']}")
        print(f"    history-less (reads disk)     {_fmt(cc['history_less_reads_disk'])}"
              f"  -> {cc['history_less_answer']}")
        print()
        print("  C3 what a blocking call does to every other request on the loop")
        print("     (1 ms ticker; lateness = how long a ready coroutine waited)")
        s = c["C3_event_loop_stall"]
        for name in ("idle_loop", "real_read_every_50ms",
                     "real_save_state_every_50ms",
                     "per_turn_tail_sequence_every_50ms"):
            v = s.get(name)
            if not v:
                continue
            print(f"    {name:<34} median {v['median_ms']:>8.3f}  "
                  f"p99 {v['p99_ms']:>8.3f}  max {v['max_ms']:>9.3f} ms")
        for v in s["injected"]:
            print(f"    injected stall {v['stall_ms']:>5} ms"
                  f"{'':<13} median {v['median_ms']:>8.3f}  "
                  f"p99 {v['p99_ms']:>8.3f}  max {v['max_ms']:>9.3f} ms")
        v = s["injected_offloaded"]
        print(f"    same {v['stall_ms']} ms via run_in_threadpool"
              f"{'':<8} median {v['median_ms']:>8.3f}  p99 {v['p99_ms']:>8.3f}  "
              f"max {v['max_ms']:>9.3f} ms")
        print("    (the injected rows are the TRANSFER FUNCTION for a slow "
              "volume, not a")
        print("     measurement of one: nothing here can measure MooseFS. A "
              "read that takes")
        print("     D delays every other ready coroutine by up to D, and the "
              "last row is")
        print("     what taking it off the loop does.)")
        print()
        print("  C4 blocking store operations the per-turn tail takes ON the loop")
        for op in c["C4_blocking_store_ops_per_turn"]:
            print(f"    {op['op']:<6} {op['site']}")
            print(f"           {op['per_turn']}")
        print()

    print("=" * 78)
    print("BENCH-JSON-BEGIN")
    print(json.dumps(result, indent=1, sort_keys=True))
    print("BENCH-JSON-END")


def _index(result: dict) -> dict:
    """Flatten a result into {key: median_ms} so two runs can be diffed.

    Keys are content-addressed by what was measured (path, size, variant),
    never by row order, because a sweep that gains a size must still diff
    against the sizes it shares with the baseline.
    """
    out: dict = {}
    paths = result.get("paths") or {}
    for r in (paths.get("A") or {}).get("rows", []):
        out[f"A|{r['messages']}|{r['variant']}"] = r["cell"]["median_ms"]
    b = paths.get("B") or {}
    for key in ("B1_turns_fixed_length", "B3_growth_artefact", "B4_bounded_scan"):
        for r in (b.get(key) or {}).get("rows", []):
            out[f"{key}|{r['turns']}"] = r["cell"]["median_ms"]
    for r in (b.get("B2_length_fixed_turns") or {}).get("rows", []):
        out[f"B2|{r['reply_chars']}"] = r["cell"]["median_ms"]
    return out


# ---------------------------------------------------------------------------

_T0 = time.time()


def _ints(s: str) -> list[int]:
    return [int(x) for x in s.replace(",", " ").split() if x.strip()]


def main_cli(argv: list[str] | None = None) -> int:
    global _T0
    _T0 = time.time()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default="ABC",
                    help="subset of paths to run, e.g. B or AC (default ABC)")
    ap.add_argument("--runs", type=int, default=7,
                    help="samples per cell (default 7); the median is reported")
    ap.add_argument("--cell-budget-s", type=float, default=8.0,
                    help="wall-clock ceiling per cell after 3 samples")
    ap.add_argument("--label", default=os.environ.get("BENCH_LABEL", "baseline"),
                    help="name for this run, carried into the JSON")
    ap.add_argument("--json", default="", help="also write the JSON to this path")
    ap.add_argument("--baseline", default="",
                    help="a previous --json file; adds a delta column")
    ap.add_argument("--assistant-chars", type=int,
                    default=fx.MEDIAN_ASSISTANT_CHARS,
                    help="median assistant reply length (production: 5248)")
    ap.add_argument("--a-sizes", type=_ints, default=[61, 85, 121, 301, 2301],
                    help="path A fixture sizes in MESSAGES. 61 is the pod's "
                         "measured 117-second case; 85 is the conversation "
                         "that is live today; 2301 is the dormant one")
    ap.add_argument("--a-covered-turns", type=int, default=40,
                    help="turns the stored hierarchy already covers in the "
                         "v3.1.9.1 row (the pod's shape was 40 of 56)")
    ap.add_argument("--b-turn-sizes", type=_ints,
                    default=[20, 40, 85, 170, 340, 600, 900, 1200])
    ap.add_argument("--b-length-sizes", type=_ints,
                    default=[656, 1312, 2624, 5248, 10496, 20992])
    ap.add_argument("--b-growth-sizes", type=_ints, default=[20, 40, 85, 170, 340])
    ap.add_argument("--llm-call-ms", type=float, default=29250.0,
                    help="per-backend-call latency used ONLY for the modelled "
                         "end-to-end line (default 29250 = the pod's 117 s "
                         "over 4 calls)")
    ap.add_argument("--tokenize-scale", type=float, default=1.5,
                    help="modelled ratio of vLLM /tokenize to the local "
                         "estimator on this model's content")
    ap.add_argument("--stall-seconds", type=float, default=2.0)
    ap.add_argument("--stall-injections", type=_ints, default=[1, 5, 25, 100, 500])
    args = ap.parse_args(argv)

    backend = Backend(latency_ms=0.0, scale=args.tokenize_scale)
    main._summarize_once = backend.summarize_once
    main.count_tokens_exact = backend.count_tokens_exact

    result = {
        "schema": "zla-bench/1",
        "env": environment(args.label),
        "params": {
            "seed": fx.SEED,
            "runs": args.runs,
            "cell_budget_s": args.cell_budget_s,
            "assistant_chars": args.assistant_chars,
            "llm_call_ms": args.llm_call_ms,
            "tokenize_scale": args.tokenize_scale,
            "a_sizes": args.a_sizes,
            "b_turn_sizes": args.b_turn_sizes,
            "b_length_sizes": args.b_length_sizes,
            "b_growth_sizes": args.b_growth_sizes,
            "stall_injections": args.stall_injections,
        },
        "paths": {},
    }

    only = args.only.upper()
    if "A" in only:
        result["paths"]["A"] = path_a(args, backend)
        result["paths"]["A_tokenizer_miss"] = path_a_tokenizer_miss(args)
    if "B" in only:
        result["paths"]["B"] = path_b(args)
    if "C" in only:
        result["paths"]["C"] = path_c(args)

    result["env"]["loadavg_end"] = _read("/proc/loadavg")[:14]
    result["env"]["wall_seconds"] = round(time.time() - _T0, 1)

    baseline = None
    if args.baseline:
        try:
            with open(args.baseline, "r", encoding="utf-8") as fh:
                baseline = json.load(fh)
        except (OSError, ValueError) as e:
            print(f"could not read baseline {args.baseline}: {e}", file=sys.stderr)

    report(result, baseline)

    if args.json:
        try:
            with open(args.json, "w", encoding="utf-8") as fh:
                json.dump(result, fh, indent=1, sort_keys=True)
            print(f"\nwrote {args.json}")
        except OSError as e:
            print(f"could not write {args.json}: {e}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main_cli())
