"""ADVERSARIAL: what 22a51ed left on the event loop, + degenerate inputs.

  docker compose -f docker-compose.tests.yml run --rm --entrypoint /bin/bash \
    unit-tests -c 'cp -r /src /work && cd /work/compactor && \
    /opt/compactor-venv/bin/python /work/tests/adversarial/test_adv_v319_loop.py'

22a51ed wrapped THREE state-IO sites and argued the change from a benchmark:
"3.8-4.9 ms of BLOCKING loop time on every turn the tail runs ... the transfer
function into request lateness measured 1:1". C1 asks whether the sweep
reached its own siblings.
"""

import asyncio
import os
import statistics
import sys
import tempfile
import time

os.environ.setdefault("MODEL_REPO", "test-model")
os.environ.setdefault("VLLM_URL", "http://stub:8000")
os.environ["COMPACTOR_RAG_ENABLED"] = "false"
os.environ["COMPACTOR_STORAGE_ROOT"] = tempfile.mkdtemp(prefix="adv-loop-")
os.environ["COMPACTOR_TARGET_TOKENS"] = "500"

sys.path.insert(0, "/work/compactor")
import memory  # noqa: E402

memory.ensure_storage_layout()
import main  # noqa: E402
import summarizer  # noqa: E402
from starlette.concurrency import run_in_threadpool  # noqa: E402

BROKEN: list[str] = []


def broke(cond, label):
    if cond:
        print(f"  *** BROKE: {label}")
        BROKEN.append(label)
    else:
        print(f"  (held)   {label}")


CONV = "loop_conv"

# A state file at the module's DOCUMENTED capacity: 9 L1 (500 tok), 4 L2
# (1200 tok), 1 L3 (2000 tok) -- what a long-running conversation carries.
st = summarizer._empty_state(CONV)
st["l1"] = [{"text": "s" * 2000, "first_turn": i * 20 + 1,
             "last_turn": (i + 1) * 20} for i in range(9)]
st["l2"] = [{"text": "c" * 4800, "first_turn": 1, "last_turn": 180}
            for _ in range(4)]
st["l3"] = {"text": "t" * 8000, "first_turn": 1, "last_turn": 180}
st["tail_fp"] = ["%064x" % i for i in range(40)]
st["turns_seen"] = 180
st["last_summarized_turn"] = 180
summarizer.save_state(CONV, st)
size = summarizer.summary_path(CONV).stat().st_size
print(f"\nstate file: {size / 1024:.1f} KB (9 L1 / 4 L2 / 1 L3, the documented max)")

t = []
for _ in range(40):
    a = time.perf_counter()
    summarizer.load_state(CONV)
    t.append((time.perf_counter() - a) * 1000)
print(f"load_state: mean {statistics.mean(t):.2f} ms  "
      f"p95 {sorted(t)[int(len(t) * .95)]:.2f} ms   (tmpfs; /data is MooseFS)")

print()
print("=" * 74)
print("C1  THE SWEEP MISSED ITS OWN SIBLINGS -- INCLUDING THE HOTTEST ONE")
print("=" * 74)
print("""
Every bare load_state/save_state still executing on the event loop inside an
async def, after the commit that exists to remove them:
""")
SITES = [
    ("main.py:5391", "chat_completions", "sstate = summarizer.load_state(conv_id)",
     "EVERY request with a conv_id -- the summary-injection read"),
    ("main.py:1690", "compact_if_needed", "_st = summarizer.load_state(conv_id)",
     "ADDED BY ced4520 IN THIS BRANCH, one commit before the sweep"),
    ("main.py:6828", "admin_compact", "_state = summarizer.load_state(conv_id)",
     "INSIDE `async with conv_lock`"),
    ("main.py:6833", "admin_compact", "summarizer.save_state(conv_id, _state)",
     "INSIDE `async with conv_lock` -- the fsync+rename half"),
    ("main.py:6861", "admin_compact", "prev = summarizer.load_state(...)",
     "once per drain iteration, up to max_calls=200"),
    ("main.py:6870", "admin_compact", "now = summarizer.load_state(...)",
     "once per drain iteration, up to max_calls=200"),
    ("main.py:6682", "admin_compact", "before = summarizer.load_state(conv_id)", ""),
    ("main.py:6877", "admin_compact", "after = summarizer.load_state(conv_id)", ""),
    ("main.py:6152", "admin_get_summary", "return summarizer.load_state(conv_id)", ""),
    ("main.py:6004", "admin_conversation_summary", "summarizer.load_state(conv_id)", ""),
    ("commands.py:727", "_handle_why", "summarizer_module.load_state(conv_id)",
     "the `why` chat command, on the request path"),
]
for where, fn, code, why in SITES:
    print(f"    {where:<18} async {fn:<26} {code}")
    if why:
        print(f"    {'':18}   ^ {why}")

print("""
main.py:1690-1710 is the sharpest: the commit wrapped format_summary_block --
a pure in-memory render with NO io -- and left the DISK READ three lines above
it bare.

    _st = summarizer.load_state(conv_id)              # <- disk, on the loop
    ...
    stored_text = await run_in_threadpool(            # <- no disk, in a thread
        summarizer.format_summary_block, _st, ...)
""")


async def heartbeat(stop, lateness, period=0.005):
    nxt = time.perf_counter()
    while not stop.is_set():
        nxt += period
        d = nxt - time.perf_counter()
        if d > 0:
            await asyncio.sleep(d)
        else:
            await asyncio.sleep(0)
        lateness.append((time.perf_counter() - nxt) * 1000)


async def measure(bare: bool, n=120):
    stop, late = asyncio.Event(), []
    hb = asyncio.create_task(heartbeat(stop, late))
    await asyncio.sleep(0.05)
    late.clear()
    for _ in range(n):
        if bare:
            summarizer.load_state(CONV)          # main.py:5391 / 1690
        else:
            await run_in_threadpool(summarizer.load_state, CONV)
        await asyncio.sleep(0)
    stop.set()
    await hb
    late.sort()
    return statistics.mean(late), late[int(len(late) * .99)], late[-1]


b = asyncio.run(measure(True))
w = asyncio.run(measure(False))
print(f"  120 loads ON THE LOOP (as shipped): mean {b[0]:.2f} ms  "
      f"p99 {b[1]:.2f} ms  max {b[2]:.2f} ms")
print(f"  120 loads IN A THREAD (as wrapped): mean {w[0]:.2f} ms  "
      f"p99 {w[1]:.2f} ms  max {w[2]:.2f} ms")
broke(b[1] > w[1] * 1.5,
      f"C1: the two sites on the per-request path were not wrapped; loop "
      f"lateness p99 {b[1]:.2f} ms against {w[1]:.2f} ms wrapped -- the same "
      f"regression 22a51ed measured at its three chosen sites, still live at "
      f"the two that run on EVERY request")


print()
print("=" * 74)
print("C2  A LONE SURROGATE FREEZES A CONVERSATION'S HIERARCHY FOREVER")
print("=" * 74)
print("""
atomic_write_json uses json.dump(..., ensure_ascii=False) onto a utf-8 stream.
A lone surrogate (which survives JSON decoding of a client body, and which
python's json module emits happily) raises UnicodeEncodeError at ENCODE time --
after the temp file is open and before os.replace. save_state then raises, and
maybe_rollup's caller logs and moves on.
""")
CONV2 = "surrogate_conv"
bad = summarizer._empty_state(CONV2)
bad["l1"] = [{"text": "scene \ud800 end", "first_turn": 1, "last_turn": 20}]
bad["last_summarized_turn"] = 20
err = None
try:
    summarizer.save_state(CONV2, bad)
except Exception as e:
    err = f"{type(e).__name__}: {e}"
print(f"  save_state -> {err or 'wrote OK'}")
exists = summarizer.summary_path(CONV2).exists()
print(f"  state file on disk afterwards: {exists}")
orphans = list(summarizer.summary_path(CONV2).parent.glob("*.tmp"))
print(f"  orphan temp files left behind: {len(orphans)}")
broke(err is not None and not exists,
      f"C2: one lone surrogate anywhere in the state makes EVERY subsequent "
      f"save_state for that conversation raise ({err}); the rollup is "
      f"discarded each turn and the hierarchy stops advancing permanently, "
      f"while /health stays green")

print()
print("=" * 74)
print("C3  DEGENERATE MESSAGE ARRAYS THROUGH THE NEW SUBSTITUTION")
print("=" * 74)
CALLS = []


async def _spy(client, to_summarize):
    CALLS.append(len(to_summarize))
    return "S", []


main.summarize = _spy
CONV3 = "degen_conv"
s3 = summarizer._empty_state(CONV3)
s3["l1"] = [{"text": "COVERS-1-40", "first_turn": 1, "last_turn": 40}]
s3["last_summarized_turn"] = 40
summarizer.save_state(CONV3, s3)

CASES = {
    "assistant-first, no user turn":
        [{"role": "assistant", "content": "hi " + "w " * 200}] * 60,
    "empty role strings":
        [{"role": "", "content": "x " * 200} for _ in range(60)],
    "system message in the middle":
        [{"role": "user", "content": "u " * 200}] * 30
        + [{"role": "system", "content": "mid"}]
        + [{"role": "assistant", "content": "a " * 200}] * 30,
    "50k empty messages":
        [{"role": "user", "content": ""} for _ in range(50000)],
    "content is None":
        [{"role": "user", "content": None} for _ in range(60)],
}
for name, msgs in CASES.items():
    CALLS.clear()
    t0 = time.perf_counter()
    try:
        out = asyncio.run(main.compact_if_needed(list(msgs), CONV3))
        dt = (time.perf_counter() - t0) * 1000
        n_in = len([m for m in msgs if m.get("role") != "system"])
        n_out = len(out)
        sub = any("COVERS-1-40" in str(m.get("content", "")) for m in out)
        print(f"  {name:<32} in={n_in:<6} out={n_out:<6} "
              f"substituted={sub!s:<5} {dt:7.1f} ms")
    except Exception as e:
        print(f"  {name:<32} RAISED {type(e).__name__}: {str(e)[:60]}")

print()
print("=" * 74)
print(f"BREAKS REPRODUCED: {len(BROKEN)}")
for x in BROKEN:
    print(f"  - {x}")
print("=" * 74)
