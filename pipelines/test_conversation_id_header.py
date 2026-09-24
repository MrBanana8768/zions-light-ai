"""
Tests for the OpenWebUI filter: chat_id propagation + history cap.

This file runs OUTSIDE the compactor image — the filter is pasted into
OpenWebUI's admin UI, so it is never imported by anything else and would
otherwise ship completely untested. It is also the only component that can
silently destroy her memory: cap the history while conv_id is still
hash-derived and every fact, embedding and summary is orphaned under an id
nothing references again.

So the interlock (never truncate without a stamped chat_id) is the single
most important assertion here, and it is mutation-tested along with the
rest.

    python test_conversation_id_header.py
"""

import asyncio
import importlib.util
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "owui_filter", Path(__file__).with_name("conversation_id_header.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

FAILED = []


def check(cond, label):
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}")
        FAILED.append(label)


def convo(n_exchanges, system=1):
    """system message(s) + alternating user/assistant."""
    msgs = [{"role": "system", "content": f"persona {i}"} for i in range(system)]
    for i in range(n_exchanges):
        msgs.append({"role": "user", "content": f"u{i}"})
        msgs.append({"role": "assistant", "content": f"a{i}"})
    return msgs


def run(body, metadata, max_turns=0):
    f = _mod.Filter()
    f.valves.max_turns = max_turns
    return asyncio.run(f.inlet(body=body, __metadata__=metadata))


print("\n[1] chat_id propagation (the original job)")
b = run({"messages": convo(2)}, {"chat_id": "abc-123"})
check(b.get("metadata", {}).get("chat_id") == "abc-123", "chat_id stamped into body.metadata")
b = run({"messages": convo(2), "metadata": {"other": "keep me"}}, {"chat_id": "x"})
check(b["metadata"].get("other") == "keep me", "an existing metadata field is not clobbered")
b = run({"messages": convo(2)}, None)
check("metadata" not in b, "no metadata from OpenWebUI -> body untouched")

print("\n[2] THE INTERLOCK — never truncate without a stamped chat_id")
# This is the memory-wipe guard. With no chat_id the compactor falls back to
# sha256(system|||first_user), so trimming the first user message mints a new
# conv_id and strands everything under the old one.
big = convo(200)
b = run({"messages": list(big)}, None, max_turns=10)
check(len(b["messages"]) == len(big), "no metadata + cap ON -> history NOT truncated")
b = run({"messages": list(big)}, {"no_chat_id": True}, max_turns=10)
check(len(b["messages"]) == len(big), "metadata without chat_id + cap ON -> NOT truncated")

print("\n[3] capping, once the stamp succeeded")
b = run({"messages": convo(200)}, {"chat_id": "c"}, max_turns=0)
check(len(b["messages"]) == 401, "max_turns=0 is off by default -> nothing dropped")
b = run({"messages": convo(200)}, {"chat_id": "c"}, max_turns=100)
kept = b["messages"]
n_sys = sum(1 for m in kept if m["role"] == "system")
n_turn = len(kept) - n_sys
check(n_turn <= 100, f"non-system messages capped ({n_turn} <= 100)")
check(kept[-1]["content"] == "a199", "the newest message survives")
check(n_sys == 1, "the system message survives")
check(kept[n_sys]["role"] == "user", "the window opens on a USER turn, not an assistant")

print("\n[4] system messages are never dropped, wherever they sit")
msgs = convo(50, system=3)
msgs.insert(40, {"role": "system", "content": "mid-conversation system"})
b = run({"messages": msgs}, {"chat_id": "c"}, max_turns=10)
check(sum(1 for m in b["messages"] if m["role"] == "system") == 4,
      "all 4 system messages kept, including the interior one")

print("\n[5] short conversations and degenerate input")
short = convo(3)
b = run({"messages": list(short)}, {"chat_id": "c"}, max_turns=100)
check(b["messages"] == short, "under the cap -> byte-identical, no rewrite")
b = run({"messages": []}, {"chat_id": "c"}, max_turns=10)
check(b["messages"] == [], "empty message list -> no crash")
only_sys = [{"role": "system", "content": "s"}]
b = run({"messages": list(only_sys)}, {"chat_id": "c"}, max_turns=1)
check(b["messages"] == only_sys, "system-only -> unchanged")
# all-assistant tail would leave nothing after the alternation walk
b = run({"messages": [{"role": "system", "content": "s"}]
                     + [{"role": "assistant", "content": f"a{i}"} for i in range(20)]},
        {"chat_id": "c"}, max_turns=5)
check(len(b["messages"]) == 21, "a window that would empty out -> forwarded untouched")

print("\n[6] the DEFAULT is off - installing this filter changes nothing")
# Constructed with no valve set, the way OpenWebUI first loads it. A non-zero
# default would truncate before anyone had verified chat_id propagation -
# springing the exact trap the docstring spends a paragraph on.
_f = _mod.Filter()
check(_f.valves.max_turns == 0, "max_turns defaults to 0")
_b = asyncio.run(_f.inlet(body={"messages": convo(200)}, __metadata__={"chat_id": "c"}))
check(len(_b["messages"]) == 401, "a freshly-installed filter forwards the full history")

print("\n[7] the cap counts TURNS, not total messages")
# With many system messages the two differ. A conversation at exactly the cap
# must not be truncated because system messages push the TOTAL over it - that
# would drop real turns to make room for the persona, at a boundary nobody
# would think to test.
msgs = convo(50, system=10)          # 10 system + 100 turns
check(len([m for m in msgs if m["role"] != "system"]) == 100, "fixture: exactly 100 turns")
b = run({"messages": list(msgs)}, {"chat_id": "c"}, max_turns=100)
check(len(b["messages"]) == len(msgs), "100 turns under a 100 cap -> nothing dropped")

# And ORDER must survive. Counting total messages instead of turns does not
# always drop anything - when turns[-N:] still returns everything, the only
# visible damage is that system messages get hoisted to the front. An
# interior system message moving is a silent rewrite of the conversation, so
# assert identity, not just length.
msgs2 = convo(49, system=4)
msgs2.insert(50, {"role": "system", "content": "interior"})   # 5 system + 98 turns = 103
check(len(msgs2) > 100 and len([m for m in msgs2 if m["role"] != "system"]) < 100,
      "fixture: total over the cap, turns under it")
b = run({"messages": list(msgs2)}, {"chat_id": "c"}, max_turns=100)
check(b["messages"] == msgs2, "under the turn cap -> order preserved, interior system not hoisted")

print("\n[8] a filter must never raise")
b = run({"messages": "not a list"}, {"chat_id": "c"}, max_turns=10)
check(b["messages"] == "not a list", "malformed messages -> body returned unchanged")
b = run({}, {"chat_id": "c"}, max_turns=10)
check(b.get("metadata", {}).get("chat_id") == "c", "no messages key -> still stamps chat_id")
# A valve that cannot be int()-ed raises INSIDE inlet, which is the only way
# to prove the try/except is load-bearing rather than decorative.
_f = _mod.Filter()
_f.valves.max_turns = "not a number"
_b = asyncio.run(_f.inlet(body={"messages": convo(200)}, __metadata__={"chat_id": "c"}))
check(len(_b["messages"]) == 401, "an unparseable valve -> body returned, no exception escapes")

print()
if FAILED:
    for f in FAILED:
        print("FAIL " + f)
    sys.exit(1)
print("All OpenWebUI filter tests passed.")
