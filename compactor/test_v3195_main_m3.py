"""
v3195-main M3 -- count_tokens's tier 1 is the THIRD site of the two-flag
rule (SP\\p18c-findings.md, the count_tokens finding; SP\\V3195_MAIN_BRIEF.md
M3). `count_tokens_exact` (main.py, ~line 1504) and tokens.py both pick
`add_generation_prompt`/`continue_final_message` from whether the message
list ends on an assistant turn; `count_tokens`'s tier 1
(`tok.apply_chat_template(...)`, ~line 1618 before this fix) passed
`add_generation_prompt=True` unconditionally and never passed
`continue_final_message` at all -- so any assistant-final list raised on
tier 1 and fell to the crude `encode()+4` fallback, on every call, forever.

No real tokenizer is available on this machine (offline; see
SP\\p18a-findings.md's own "Not demonstrated" section), so this test stands
up a FAKE tokenizer object that reproduces the real one's documented
refusal rule (measured against the shipped image in SP\\p18c-findings.md,
top-of-file ACCEPT/REFUSE table) closely enough to exercise the FLAG-
SELECTION LOGIC in count_tokens, which is what this fix changes. It does
not claim to reproduce real token counts -- only the shape of the bug and
the fix.

    python test_v3195_main_m3.py
"""

import os
import sys

os.environ.setdefault("MODEL_REPO", "coder-fake-for-test")

import main  # noqa: E402

_FAILED = False


def check(cond, label):
    global _FAILED
    status = "ok  " if cond else "FAIL"
    print(f"  {status} {label}")
    if not cond:
        _FAILED = True


class _FakeTok:
    """Reproduces the two REAL refusal rules measured in SP\\p18c-findings.md
    against the shipped vLLM/mistral_common stack:
      - add_generation_prompt=True on an assistant-final list raises
        ("Consider using continue_final_message instead").
      - neither flag set on a non-assistant-final list raises ("Expected
        last role User or Tool...").
    Rendering is a simple deterministic stand-in (role tag + word count),
    NOT a claim about real tokenizer output -- it exists so tier 1 and
    tier 2 produce genuinely different, comparable counts for the same
    input, which is what the "before/after" measurement below needs.
    """

    def __init__(self):
        self.calls = []  # (add_generation_prompt, continue_final_message)

    def apply_chat_template(self, messages, tokenize=False,
                             add_generation_prompt=False,
                             continue_final_message=False):
        self.calls.append((add_generation_prompt, continue_final_message))
        last_role = messages[-1].get("role") if messages else None
        if add_generation_prompt and last_role == "assistant" and not continue_final_message:
            raise ValueError(
                "The last message in the conversation is already an "
                "assistant message. Consider using `continue_final_message` "
                "instead."
            )
        if (not add_generation_prompt and not continue_final_message
                and last_role != "assistant"):
            raise ValueError(
                "Expected last role User or Tool (or Assistant with prefix "
                "or continue_final_message set to True)"
            )
        parts = []
        for m in messages:
            parts.append(f"<{m.get('role')}>")
            parts.extend(str(m.get("content", "")).split())
        if add_generation_prompt:
            parts.append("<assistant_prompt>")
        return " ".join(parts)

    def encode(self, text):
        return text.split()


def msg(role, text):
    return {"role": role, "content": text}


print("[1] reproduction: an assistant-final ONE-message list is exactly "
      "the shape the guard's per-message cost table builds "
      "(main.py:7816-ish: count_tokens([m]) for each m)")
tok = _FakeTok()
_orig_get_tokenizer = main.get_tokenizer
main.get_tokenizer = lambda: tok
try:
    a_msg = msg("assistant", "the quick brown fox jumps over the lazy dog")
    u_msg = msg("user", "the quick brown fox jumps over the lazy dog")
    s_msg = msg("system", "the quick brown fox jumps over the lazy dog")

    n_assistant = main.count_tokens([a_msg])
    n_user = main.count_tokens([u_msg])
    n_system = main.count_tokens([s_msg])

    # Tier-1 succeeded means apply_chat_template was actually called with a
    # flag combination that did not raise for THIS message's role -- check
    # by replaying the fake tokenizer's own call log rather than trusting
    # count_tokens' return value alone (a raise is swallowed by design; the
    # call log is the only place the attempt itself is visible).
    check(len(tok.calls) == 3,
          f"three apply_chat_template attempts were made (got {len(tok.calls)})")

    # Same content, same word count, priced through THREE roles. If tier 1
    # ran for all three, the counts should be equal (identical content,
    # identical template overhead per this fake's own rendering rule) --
    # tier 1 does not distinguish role in the render length here beyond the
    # role tag, which is 1 token either way. If the assistant one instead
    # fell to tier 2 (encode()+4, no role tag, no template awareness), it
    # would read differently from the other two for the SAME content.
    check(n_user == n_system,
          f"user and system price identically for identical content "
          f"({n_user} vs {n_system}) -- both go through tier 1")
    # NOT equal to n_user: correctly-flagged tier 1 renders an
    # assistant-final list WITHOUT add_generation_prompt's trailing
    # "<assistant_prompt>" marker (there is nothing left to prompt for --
    # the assistant turn is already the last message), so this fake's own
    # render is exactly 1 token SHORTER than the user/system case, which
    # both DO get that marker. That is real template behaviour, not a bug:
    # count_tokens_exact's own docstring says the two flags exist because
    # the two shapes genuinently differ. The number that matters is that
    # tier 1 ran at all (see [2]/[3] below) -- before this fix it never did
    # for an assistant-final list, regardless of what it would have priced.
    check(n_assistant == n_user - 1,
          f"THE FIX: assistant-final now goes through TIER 1 (the real "
          f"template), not the tier-2 fallback -- it prices 1 token less "
          f"than the user/system case (no add_generation_prompt marker to "
          f"render), not the arbitrary encode()+4 number tier 2 would have "
          f"given ({n_assistant} vs {n_user})")

    print()
    print("[2] the actual flags sent for each role (this is what changed)")
    # calls[0] is a_msg's attempt, calls[1] is u_msg's, calls[2] is s_msg's
    a_agp, a_cfm = tok.calls[0]
    u_agp, u_cfm = tok.calls[1]
    check(a_agp is False and a_cfm is True,
          f"assistant-final list: add_generation_prompt=False, "
          f"continue_final_message=True (got agp={a_agp}, cfm={a_cfm})")
    check(u_agp is True and u_cfm is False,
          f"user-final list: add_generation_prompt=True, "
          f"continue_final_message=False (got agp={u_agp}, cfm={u_cfm}) -- "
          f"unchanged from before this fix, matching count_tokens_exact's "
          f"own sibling logic")

    print()
    print("[3] the per-message cost table the guard actually builds "
          "(main.py's per = [count_tokens([m]) for m in msgs]) -- measured "
          "before/after, on a realistic mixed-role window")
    window = (
        [msg("system", "you are a careful assistant who answers plainly")]
        + [msg("user" if i % 2 == 0 else "assistant",
               "the reply covers the point in a few plain sentences " * (1 + i % 3))
           for i in range(12)]
    )
    tok2 = _FakeTok()
    main.get_tokenizer = lambda: tok2
    per_message = [main.count_tokens([m]) for m in window]
    n_assistant_turns = sum(1 for m in window if m.get("role") == "assistant")
    assistant_calls = [c for c in tok2.calls]
    # Every call must have been an ATTEMPT that either succeeded (recorded
    # in tok2.calls) with the role-appropriate flags. Count how many of the
    # attempts used continue_final_message=True (assistant-shaped) vs
    # add_generation_prompt=True (everything else) and confirm it matches
    # the window's own role mix -- this is the "every assistant turn is
    # priced by the fallback" claim, inverted: after the fix, every
    # assistant turn is priced by tier 1 like everything else.
    cfm_count = sum(1 for (_, cfm) in assistant_calls if cfm)
    check(cfm_count == n_assistant_turns,
          f"{cfm_count} of {len(assistant_calls)} tier-1 attempts used "
          f"continue_final_message=True, matching the window's "
          f"{n_assistant_turns} assistant turn(s) exactly -- before this "
          f"fix this was 0 regardless of how many assistant turns existed, "
          f"because the flag was never sent")
finally:
    main.get_tokenizer = _orig_get_tokenizer

print()
print("[4] CONTROL: a list that does NOT end on assistant is unaffected "
      "(add_generation_prompt=True, continue_final_message=False, exactly "
      "as before this fix)")
tok3 = _FakeTok()
main.get_tokenizer = lambda: tok3
try:
    su_list = [msg("system", "be careful"), msg("user", "hello there friend")]
    n = main.count_tokens(su_list)
    check(len(tok3.calls) == 1, "exactly one apply_chat_template attempt")
    agp, cfm = tok3.calls[0]
    check(agp is True and cfm is False,
          f"unchanged: add_generation_prompt=True, "
          f"continue_final_message=False (got agp={agp}, cfm={cfm})")
    check(isinstance(n, int) and n > 0, f"a real count was returned ({n})")
finally:
    main.get_tokenizer = _orig_get_tokenizer

print()
print("[5] CONTROL: empty message list does not crash "
      "(messages[-1] guarded)")
tok4 = _FakeTok()
main.get_tokenizer = lambda: tok4
try:
    n_empty = main.count_tokens([])
    check(isinstance(n_empty, int), f"empty list returns an int ({n_empty})")
finally:
    main.get_tokenizer = _orig_get_tokenizer

print()
print("[6] visibility: a GENUINE tier-2 fallback is counted and reaches "
      "tokenizer_state(), separate from the /tokenize HTTP counter")


class _AlwaysRaisesTok:
    def apply_chat_template(self, *a, **k):
        raise ValueError("synthetic: the template refuses this shape")

    def encode(self, text):
        return text.split()


tok5 = _AlwaysRaisesTok()
main.get_tokenizer = lambda: tok5
try:
    before_total = main.tokenizer_state()["chat_template_fallback_total"]
    before_http = main._tokenize_fail_streak
    main.count_tokens([msg("user", "anything at all")])
    st = main.tokenizer_state()
    check(st["chat_template_fallback_total"] == before_total + 1,
          f"chat_template_fallback_total incremented "
          f"({before_total} -> {st['chat_template_fallback_total']})")
    check(st["chat_template_fallback_streak"] >= 1,
          f"chat_template_fallback_streak is nonzero "
          f"({st['chat_template_fallback_streak']})")
    check(main._tokenize_fail_streak == before_http,
          f"the UNRELATED /tokenize HTTP counter is untouched "
          f"(_tokenize_fail_streak stayed {before_http}) -- this failure "
          f"never made an HTTP call at all")
finally:
    main.get_tokenizer = _orig_get_tokenizer
    main._chat_template_fallback_streak = 0
    main._chat_template_fallback_total = 0
    main._chat_template_degraded_since = None
    main._chat_template_last_warn_at = None

print()
if _FAILED:
    print("SOME v3195-main M3 CHECKS FAILED")
    sys.exit(1)
print("All v3195-main M3 checks passed.")
