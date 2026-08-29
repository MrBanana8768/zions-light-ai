"""
The /tokenize request body — the flags, and only the flags.

This test exists because the fix it guards took production down TWICE and
nothing in the estate caught either time.

  2026-08-29 morning. count_tokens_exact sent add_generation_prompt=True on a
  message list ending with an assistant turn. vLLM's Mistral template refuses:
      "Cannot set `add_generation_prompt` to True when the last message is
       from the assistant. Consider using `continue_final_message` instead."
  The summarizer measures a slice of OLD turns, which routinely ends on an
  assistant reply, so every compaction fell back to a counter reading up to
  51% low. Compaction died; the guard shed 80+ turns per request.

  2026-08-29 afternoon. The fix set add_generation_prompt=False and stopped
  there. The template has a SECOND guard:
      "Expected last role User or Tool (or Assistant with prefix or
       continue_final_message set to True)"
  Same outage, new error string, inside the hour.

The gate then proved the estate was blind to it: reintroducing the defect left
the 30-file suite at PASS=30. A fix that has taken production down twice and
has no test is a fix waiting to be reverted by someone tidying up.

So this asserts the BODY, per message shape, rather than any downstream
behaviour. It needs no server, no model and no network.

    python test_tokenize_flags.py
"""

import os
import sys
from unittest.mock import patch

os.environ.setdefault("MODEL_REPO", "test-model")
os.environ.setdefault("VLLM_URL", "http://stub:8000")

import main  # noqa: E402


class _Resp:
    status_code = 200
    text = '{"count": 7}'

    @staticmethod
    def json():
        return {"count": 7}


def body_for(messages, **kw):
    """Run the real count_tokens_exact and return the JSON it would POST."""
    seen = {}

    def _post(url, json=None, timeout=None, **_):
        seen["url"] = url
        seen["json"] = json
        return _Resp()

    with patch.object(main.httpx, "post", _post):
        main.count_tokens_exact(messages, **kw)
    return seen.get("json")


def check(label, messages, want_agp, want_cfm, **kw):
    b = body_for(messages, **kw)
    if b is None:
        print(f"FAIL {label}: no request was made at all")
        sys.exit(1)
    agp = b.get("add_generation_prompt")
    cfm = b.get("continue_final_message")
    if agp != want_agp or cfm != want_cfm:
        print(f"FAIL {label}: add_generation_prompt={agp} "
              f"continue_final_message={cfm}, wanted {want_agp}/{want_cfm}")
        sys.exit(1)
    # vLLM refuses the request outright if both are true — a third refusal
    # rule, and one this code must make unreachable rather than merely avoid.
    if agp and cfm:
        print(f"FAIL {label}: both flags true, which vLLM rejects "
              f"('Cannot set both')")
        sys.exit(1)
    print(f"  ok   {label}  (agp={agp} cfm={cfm})")


U = {"role": "user", "content": "hello"}
A = {"role": "assistant", "content": "hi there"}
S = {"role": "system", "content": "be kind"}
T = {"role": "tool", "content": "result"}

print("[1] the shape that took production down: an assistant-final list")
# This is what the SUMMARIZER sends on every compaction. Both flags wrong here
# is the whole outage.
check("user, assistant", [U, A], False, True)
check("system, user, assistant", [S, U, A], False, True)
check("a long slice ending on an assistant turn", [S] + [U, A] * 20, False, True)

print()
print("[2] the shape the GUARD sends: ending on the user's new turn")
check("user only", [U], True, False)
check("system, user", [S, U], True, False)
check("user, assistant, user", [U, A, U], True, False)

print()
print("[3] shapes that are neither")
check("system only", [S], True, False)
check("tool-final", [U, A, T], True, False)

print()
print("[4] an explicit caller override is honoured, and stays consistent")
check("forced True", [U], True, False, add_generation_prompt=True)
check("forced False on a user-final list", [U], False, True,
      add_generation_prompt=False)

print()
print("[5] an empty list costs nothing")
if body_for([]) is not None:
    print("FAIL an empty list should short-circuit to 0 without a request")
    sys.exit(1)
print("  ok   no request is made for an empty list")

print()
print("[6] the flags are always complements, over every shape above")
# Stated separately because it is the invariant, not an accident of the cases:
# vLLM rejects both-true, and the template rejects both-false on an
# assistant-final list. There is no shape where the same value is right twice.
for label, msgs in (
    ("[U]", [U]), ("[S]", [S]), ("[U,A]", [U, A]), ("[S,U,A]", [S, U, A]),
    ("[U,A,U]", [U, A, U]), ("[U,A,T]", [U, A, T]),
):
    b = body_for(msgs)
    if b["add_generation_prompt"] == b["continue_final_message"]:
        print(f"FAIL {label}: flags are equal "
              f"({b['add_generation_prompt']}); they must be complements")
        sys.exit(1)
print("  ok   add_generation_prompt and continue_final_message never agree")

print()
print("All /tokenize flag tests passed.")
