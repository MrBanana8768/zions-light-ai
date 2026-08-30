"""FULL-pipeline tail-shape verification: vLLM 0.19's MistralTokenizer
wrapping transformers' MistralCommonTokenizer, exactly as the server builds
it. This is the complete template stack the production 400s came from -
prep (rules 1-2) plus the transformers/mistral_common layer (rule 3 and
whatever empty-content behaviour exists).
"""
import inspect
import shutil
import sys

from transformers.tokenization_mistral_common import MistralCommonTokenizer
import vllm.tokenizers.mistral as vtm

TEKKEN = ("/opt/vllm-venv/lib/python3.12/site-packages/"
          "mistral_common/data/tekken_240911.json")

# Give from_pretrained an HF-shaped local dir holding only tekken.json.
import os
os.makedirs("/tmp/tok", exist_ok=True)
shutil.copy(TEKKEN, "/tmp/tok/tekken.json")

mct = None
errs = []
for attempt, fn in (
    ("from_pretrained('/tmp/tok')",
     lambda: MistralCommonTokenizer.from_pretrained("/tmp/tok")),
    ("MistralCommonTokenizer('/tmp/tok/tekken.json')",
     lambda: MistralCommonTokenizer("/tmp/tok/tekken.json")),
    ("MistralCommonTokenizer(tokenizer_path=...)",
     lambda: MistralCommonTokenizer(tokenizer_path="/tmp/tok/tekken.json")),
):
    try:
        mct = fn()
        print("constructed via", attempt)
        break
    except Exception as e:
        errs.append(f"{attempt} -> {type(e).__name__}: {str(e)[:100]}")
if mct is None:
    print("could not construct MistralCommonTokenizer:")
    for e in errs:
        print("  ", e)
    sys.exit(1)

tok = vtm.MistralTokenizer(mct)
print("vLLM MistralTokenizer wrapped; version:", getattr(tok, "version", "?"))

U = {"role": "user", "content": "tell me about the coast"}
A = {"role": "assistant", "content": "we talked about the coast and"}
AE = {"role": "assistant", "content": ""}
AW = {"role": "assistant", "content": "   "}
S = {"role": "system", "content": "be kind"}

CASES = [
    ("healthy user-final, agp=True",
     [S, U, A, U], dict(add_generation_prompt=True, continue_final_message=False),
     "ACCEPT"),
    ("assistant-final WITH content, cfm=True (repair continuation)",
     [S, U, A], dict(add_generation_prompt=False, continue_final_message=True),
     "ACCEPT"),
    ("assistant-final EMPTY, cfm=True",
     [S, U, AE], dict(add_generation_prompt=False, continue_final_message=True),
     "UNKNOWN"),
    ("assistant-final WHITESPACE, cfm=True",
     [S, U, AW], dict(add_generation_prompt=False, continue_final_message=True),
     "UNKNOWN"),
    ("LONE empty assistant, cfm=True (repair case [4] exactly)",
     [AE], dict(add_generation_prompt=False, continue_final_message=True),
     "UNKNOWN"),
    ("system + lone empty assistant, cfm=True",
     [S, AE], dict(add_generation_prompt=False, continue_final_message=True),
     "UNKNOWN"),
    ("system-final (repair leaves untouched)",
     [U, A, S], dict(add_generation_prompt=True, continue_final_message=False),
     "UNKNOWN"),
    ("OUTAGE 1: assistant-final, agp=True",
     [S, U, A], dict(add_generation_prompt=True, continue_final_message=False),
     "REFUSE"),
    ("OUTAGE 2: assistant-final, neither flag",
     [S, U, A], dict(add_generation_prompt=False, continue_final_message=False),
     "REFUSE"),
    ("rule 1: both flags",
     [S, U, A], dict(add_generation_prompt=True, continue_final_message=True),
     "REFUSE"),
]

print()
worry = 0
for label, msgs, kw, expect in CASES:
    try:
        out = tok.apply_chat_template([dict(m) for m in msgs], tools=None, **kw)
        got = f"ACCEPTED ({len(out)} tokens)"
        ok = expect in ("ACCEPT", "UNKNOWN")
    except Exception as e:
        got = f"REFUSED {type(e).__name__}: {str(e)[:150]}"
        ok = expect in ("REFUSE", "UNKNOWN")
    flag = "   " if ok else ">>> UNEXPECTED"
    if not ok:
        worry += 1
    print(f"{flag}[{label}] expected {expect}")
    print(f"      {got}")

print()
print("unexpected outcomes:", worry)
