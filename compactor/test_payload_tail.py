"""
The payload forwarded to vLLM must have a template-valid tail.

PRODUCTION, 2026-08-29 22:38:46. vLLM refused a generation outright:

    ValueError: Cannot set `add_generation_prompt` to True when the last
    message is from the assistant. Consider using `continue_final_message`
    instead.

and the compactor logged "this turn produced no reply, no facts and no
episodic write, and nothing retries it". Five times that day.

Every flag guard in main.py lived in count_tokens_exact - the path that
MEASURES a payload. Nothing guarded the payload actually sent. The two paths
hit the same template and the same three refusal rules, and only one of them
was defended. That is the same fix-one-site-miss-the-sibling shape as the two
/tokenize outages, except here the sibling was a different function entirely.

The array reaches that shape by cascade, not client error: a stream that dies
mid-reply leaves an EMPTY assistant turn in the client's history, OpenWebUI
resends the whole array next turn, and the empty turn is now final. One dead
stream poisons the following turn too.

    python test_payload_tail.py
"""

import os
import sys

os.environ.setdefault("MODEL_REPO", "test-model")
os.environ.setdefault("VLLM_URL", "http://stub:8000")

import main  # noqa: E402

FAILED = []


def check(label, body, want_last_role, want_cfm, want_len=None):
    note, was_invalid = main._repair_template_invalid_tail(body)
    msgs = body["messages"]
    last = msgs[-1].get("role") if msgs else None
    cfm = body.get("continue_final_message")
    agp = body.get("add_generation_prompt")
    if last != want_last_role:
        FAILED.append(f"{label}: last role {last!r}, wanted {want_last_role!r}")
        return
    if bool(cfm) != want_cfm:
        FAILED.append(f"{label}: continue_final_message={cfm}, wanted {want_cfm}")
        return
    if cfm and agp:
        FAILED.append(f"{label}: both flags set, which vLLM refuses outright")
        return
    if want_len is not None and len(msgs) != want_len:
        FAILED.append(f"{label}: {len(msgs)} message(s), wanted {want_len}")
        return
    print(f"  ok   {label}  (last={last}, cfm={cfm}, n={len(msgs)})")


U = {"role": "user", "content": "what did we decide about the trip?"}
A = {"role": "assistant", "content": "we decided on the coast"}
EMPTY = {"role": "assistant", "content": ""}
BLANK = {"role": "assistant", "content": "   " + chr(10) + "  "}
S = {"role": "system", "content": "be kind"}

print("[1] the production shape: an empty assistant turn left by a dead stream")
check("user, assistant, user, EMPTY", {"messages": [U, A, U, dict(EMPTY)]},
      "user", False, want_len=3)
check("with a system turn too", {"messages": [S, U, A, U, dict(EMPTY)]},
      "user", False, want_len=4)
check("whitespace-only, not just empty", {"messages": [U, A, U, dict(BLANK)]},
      "user", False, want_len=3)
check("several dead streams in a row",
      {"messages": [U, A, U, dict(EMPTY), dict(BLANK), dict(EMPTY)]},
      "user", False, want_len=3)

print()
print("[2] a REAL assistant-final list is a continuation, not junk")
# Dropping this would discard the very text the user asked to continue.
check("user, assistant", {"messages": [U, dict(A)]}, "assistant", True,
      want_len=2)
check("system, user, assistant", {"messages": [S, U, dict(A)]}, "assistant",
      True, want_len=3)

print()
print("[3] an ordinary turn is left completely alone")
b = {"messages": [U, A, U]}
check("user-final", b, "user", False, want_len=3)
if "continue_final_message" in b or "add_generation_prompt" in b:
    FAILED.append("a healthy payload had template flags added to it")
else:
    print("  ok   no template flags were added to a healthy payload")

print()
print("[4] the conversation can never be emptied - and never sent EMPTY")
# A lone assistant turn has no user turn to fall back to. Dropping it would
# send vLLM an empty array, trading a 400 for a different 400. But keeping
# it EMPTY with continue_final_message set is ALSO a 400: verified against
# vLLM 0.19's full template stack in the production image, empty string
# content is refused ("Assistant message must have either content or
# tool_calls") while whitespace-only content is accepted. So the kept tail
# must carry at least a space. The first version of this repair shipped the
# refused shape; the probe caught it.
b4a = {"messages": [dict(EMPTY)]}
check("a lone empty assistant turn", b4a, "assistant", True, want_len=1)
if not (b4a["messages"][-1].get("content") or ""):
    FAILED.append(
        "the kept lone assistant tail is still EMPTY - vLLM refuses that "
        "shape even with continue_final_message (verified 2026-08-30)"
    )
else:
    print("  ok   the kept tail carries content vLLM verifiably accepts")
b4b = {"messages": [S, dict(EMPTY)]}
check("system + lone empty assistant", b4b, "assistant", True, want_len=2)
if not (b4b["messages"][-1].get("content") or ""):
    FAILED.append("the system+lone kept tail is still EMPTY")
else:
    print("  ok   same for the system-prefixed variant")

print()
print("[5] a stale flag from the client is never left to collide")
# vLLM refuses when BOTH are true. If the client sent continue_final_message
# and the tail is user-final, it must not survive.
b = {"messages": [U, A, U], "continue_final_message": True}
check("client sent a stale continue_final_message", b, "user", False)

print()
print("[6] an already-valid payload is not reported as a repair")
# The WARNING this drives says the request "would have been rejected by
# vLLM". Firing it on a payload vLLM would have accepted teaches the
# operator to ignore the line that does matter.
_n, _invalid = main._repair_template_invalid_tail(
    {"messages": [U, dict(A)], "continue_final_message": True}
)
if _invalid:
    FAILED.append(
        "an assistant-final payload the client already flagged with "
        "continue_final_message was reported as having been invalid"
    )
else:
    print("  ok   a client-flagged continuation is not reported as invalid")
_n2, _invalid2 = main._repair_template_invalid_tail(
    {"messages": [U, A, U, {"role": "assistant", "content": ""}]}
)
if not _invalid2:
    FAILED.append("a genuinely broken tail was not reported as invalid")
else:
    print("  ok   a dead-stream tail IS reported as invalid")

print()
print("[7] INTERIOR empty assistant turns - the 2026-08-30 06:41 production shape")
# She cancels a stream (0 chars) -> OpenWebUI stores an empty assistant turn
# -> she types a NEW message (not a regenerate) -> the payload is USER-final
# with the dead turn INTERIOR, and the template refuses the whole request:
# "Invalid assistant message: role='assistant' content=''". Four of these in
# the 08-28..08-30 window, ~2 dead turns a day. The tail repair only shed
# empties while they were LAST, which covers the regenerate flow its
# docstring describes and misses this one entirely.
#
# Verified against vLLM 0.19's own template stack (2026-08-30): this exact
# 6-message array with "" is REFUSED and with " " is ACCEPTED (24 tokens).
b7 = {"messages": [S, U, A, dict(U), dict(EMPTY), dict(U)]}
_n7, _inv7 = main._repair_template_invalid_tail(b7)
m7 = b7["messages"]
if len(m7) != 6:
    FAILED.append(
        f"an interior turn was DROPPED ({len(m7)} of 6 left) - that splices "
        f"two user turns together and breaks the alternation the template "
        f"also requires, trading one 400 for another"
    )
elif (m7[4].get("content") or "") == "":
    FAILED.append(
        "the interior empty assistant turn was left empty - vLLM refuses "
        "that payload outright (production 06:41:06)"
    )
elif not _inv7:
    FAILED.append("an interior repair was not reported as a real repair")
else:
    print("  ok   interior empty turn space-filled, all 6 turns preserved")

# Several of them, and mixed with a trailing one.
b7b = {"messages": [S, U, dict(EMPTY), dict(U), dict(BLANK), dict(U)]}
main._repair_template_invalid_tail(b7b)
if any(
    m.get("role") == "assistant" and isinstance(m.get("content"), str)
    and m["content"] == ""
    for m in b7b["messages"]
):
    FAILED.append("a second interior empty turn survived the sweep")
else:
    print("  ok   multiple interior empties are all repaired")

# A multimodal assistant turn reads as text-empty but may carry an image.
# Filling it would be one thing; DESTROYING the image would be worse than
# the 400, so list content must be left exactly alone.
_img = {"role": "assistant", "content": [{"type": "image_url", "image_url": {"url": "data:x"}}]}
b7c = {"messages": [S, U, dict(_img), dict(U)]}
main._repair_template_invalid_tail(b7c)
if b7c["messages"][2].get("content") != _img["content"]:
    FAILED.append("a multimodal assistant turn was rewritten - an image may have been destroyed")
else:
    print("  ok   multimodal (list) content is never rewritten")

# And a healthy conversation is still untouched.
b7d = {"messages": [S, U, A, dict(U)]}
_n7d, _inv7d = main._repair_template_invalid_tail(b7d)
if _n7d is not None or len(b7d["messages"]) != 4:
    FAILED.append(f"a healthy payload was modified: note={_n7d!r}")
else:
    print("  ok   a healthy conversation is left alone")

print()
if FAILED:
    for f in FAILED:
        print("FAIL " + f)
    sys.exit(1)
print("All payload-tail tests passed.")
