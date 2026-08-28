# Tokenizer-contract fixture

A CPU-only, vLLM-**shaped** tokenizer server, so the compactor's budget code can
be tested against a real tokenizer instead of against the estimator that was
wrong.

## Why

`compactor/test_smoke.py` opens with this, verbatim:

> Exercises the pure-Python code paths (no vLLM, no GPU, no real tokenizer).
> The HuggingFace tokenizer load is skipped by leaving MODEL_REPO unset, so
> count_tokens falls back to the char/4 estimator.

Every budget test in the repo inherits that. So the suite has only ever asserted
the budget arithmetic **against char/4** — the estimator that under-counted the
production payload by 23% and assistant content by 34-51%. The suite was, by
construction, incapable of catching the bug that took production down on
2026-08-24 and again on 2026-08-28, and it was fully green on both days.

This fixture is the second opinion the suite never had.

## Run it

```
docker compose -f docker-compose.tokenizer-contract.yml \
    up --build --exit-code-from contract-tests
```

Against a fixture you keep running, iterating on the tests:

```
docker compose -f docker-compose.tokenizer-contract.yml up -d tokenize-fixture
docker compose -f docker-compose.tokenizer-contract.yml run --rm contract-tests
```

From the host, by hand (the fixture publishes `18000:8000`):

```
curl -s localhost:18000/_fixture/info
curl -s -X POST localhost:18000/tokenize -H 'content-type: application/json' \
     -d '{"model":"fixture-model","messages":[{"role":"user","content":"hi"}]}'
```

Swap either side of the comparison:

```
# local count = char/4, the regime test_smoke.py runs in
docker compose -f docker-compose.tokenizer-contract.yml \
    run --rm -e CONTRACT_LOCAL_TOKENIZER= contract-tests
```

With no fixture reachable, `compactor/test_tokenizer_contract.py` prints a loud
SKIP and exits 0. The existing CPU-only suite is untouched.

## What was chosen, and what it cost

The compactor depends on exactly two backend contracts
(`compactor/main.py:371-373`, `:459`):

```python
r = httpx.post(
    f"{VLLM_URL}/tokenize",
    json={"model": MODEL_REPO, "messages": messages},
    ...
n = r.json().get("count")
```

`{"model", "messages"} -> .count`. Fidelity to **that** is the whole point; a
server with a differently-shaped `/tokenize` tests nothing useful.

| Option | `/tokenize` shape | Verdict |
| --- | --- | --- |
| **vLLM CPU release image** (`public.ecr.aws/q9t5s3a7/vllm-cpu-release-repo`) | exact, by definition | **Cannot run on this host.** Measured below. |
| **vLLM CUDA image** (`vllm/vllm-openai:latest`) | exact | 8.63 GB compressed, and needs a GPU. Out. |
| **llama.cpp server** | `/tokenize` takes `{"content": "..."}` and returns `{"tokens": [...]}` — no `messages`, no `count` | Wrong shape. Would test nothing. |
| **TGI** | `/tokenize` takes `{"inputs": "..."}`, returns a token array | Wrong shape. |
| **Ollama** | no tokenize endpoint at all | Out. |
| **Purpose-built stub** (this) | reimplemented from vLLM's own protocol module | **Chosen.** |

### Why vLLM's own CPU build could not be used — measured, 2026-08-28

The prebuilt CPU wheel is compiled with AVX-512. This host has AVX2 only:

```
$ docker run --rm alpine:3 sh -c "grep -m1 flags /proc/cpuinfo | tr ' ' '\n' \
      | grep -E '^(avx512f|avx2|amx_tile)$' | sort -u"
avx2
```

The image pulls and Python imports fine. The extension module does not:

```
$ docker run --rm --entrypoint python3 public.ecr.aws/q9t5s3a7/vllm-cpu-release-repo:v0.10.0 \
      -c "import vllm; print('vllm', vllm.__version__)"
vllm 0.10.0

$ docker run --name p --entrypoint python3 public.ecr.aws/q9t5s3a7/vllm-cpu-release-repo:v0.10.0 -c "
print('a: import vllm', flush=True); import vllm
print('b: import _C custom ops', flush=True); import vllm._C
print('c: import api_server', flush=True); import vllm.entrypoints.openai.api_server"
a: import vllm
b: import _C custom ops
$ docker inspect p --format '{{.State.ExitCode}}'
132                      # 128 + 4 = SIGILL, illegal instruction
```

And end to end, serving a 135M model:

```
$ docker run -d --name probe -p 18000:8000 -e VLLM_CPU_KVCACHE_SPACE=2 \
      public.ecr.aws/q9t5s3a7/vllm-cpu-release-repo:v0.10.0 \
      --model HuggingFaceTB/SmolLM2-135M-Instruct --max-model-len 2048
$ docker ps -a --filter name=probe --format '{{.Status}}'
Exited (132) 42 seconds ago
```

Building vLLM CPU from source with `VLLM_CPU_DISABLE_AVX512=true` would work,
but that is a multi-hour compile per CI image refresh for a fixture whose job is
to be run on every change. That trade is not worth taking, and a fixture
expensive enough to skip is a fixture that gets skipped.

**Sizes, measured:** vLLM CPU 4.04 GB on disk (1.29 GB compressed pull) versus
380 MB for this fixture and 379 MB for its test runner.

### What the stub gives up, and what it keeps

It gives up being vLLM. It keeps the shape, and the shape was dumped from
vLLM's own source rather than guessed:

```
$ docker run --rm --entrypoint python3 public.ecr.aws/q9t5s3a7/vllm-cpu-release-repo:v0.10.0 -c "
from vllm.entrypoints.openai.protocol import TokenizeChatRequest, TokenizeResponse
print(sorted(TokenizeChatRequest.model_fields)); print(sorted(TokenizeResponse.model_fields))"
['add_generation_prompt', 'add_special_tokens', 'chat_template', 'chat_template_kwargs',
 'continue_final_message', 'messages', 'mm_processor_kwargs', 'model',
 'return_token_strs', 'tools']
['count', 'max_model_len', 'token_strs', 'tokens']
```

Two behaviours were copied from `vllm/entrypoints/openai/serving_engine.py`:

* `/tokenize` performs **no** context-length validation (`:606-609` returns
  early for `TokenizeChatRequest`). The fixture matches — the guard has to be
  able to measure a payload too large to send, or it can never decide how much
  to shed.
* `/v1/chat/completions` rejects when `token_num + max_tokens > max_model_len`
  (`:619-631`), with the message reproduced verbatim.

## Honest limits

Read these before citing a green run.

1. **The tokenizer is not Cydonia-24B's.** It is
   `HuggingFaceTB/SmolLM2-135M-Instruct`, a 49k byte-level BPE. Production is
   mistral_common over `tekken.json`. **Every absolute number this suite prints
   is meaningless for the production budget.** What is validated is the
   contract and the wiring: that the compactor asks the server, reads `.count`,
   that its scale-corrected arithmetic actually lands under the limit when a
   real tokenizer measures it, and that it degrades honestly when the endpoint
   lies or vanishes.
2. **`GENERATION_RESERVE` is not validated here.** Whether 16384 is right is a
   property of how long *this model's* replies run (7,513-11,347 tokens,
   measured on the production conversation). No fixture can settle that.
3. **Version gap.** The shapes were verified against `vllm==0.10.0` (the CPU
   release image). The deployed stack pins `vllm==0.24.0` (`Dockerfile:78`).
   The endpoints are believed stable across that range; that was **not**
   verified from this machine, and it is the single largest fidelity gap here.
4. **No vision.** `/tokenize` here sees only text parts. `IMAGE_TOKEN_ESTIMATE`
   (4096, roughly half a real Mistral3 tile per `main.py:889-895`) is not
   exercised.
5. **Emoji do not diverge with the default tokenizer pair** — measured 1.00x,
   because gpt2 and SmolLM2 are both byte-level BPEs that price those code
   points alike. Box-drawing (1.97x) and CJK (1.47x) do. The suite prints all
   three and fails only if none of them diverge. Emoji cost is attested by the
   production measurement in `INCIDENT_2026-08-28.md`, not by this suite.

## Fault injection

```
POST /_fixture/mode   {"tokenize_mode": "...", "factor": 0.5, "status": 400, "delay": 15.0}
```

| mode | behaviour |
| --- | --- |
| `ok` | normal |
| `wrong` | returns `int(true_count * factor)` — plausible, and wrong. `0.5` reproduces the production undercount |
| `http_error` | returns `status` with a vLLM-ish error body |
| `garbage` | 200 with no `count` key |
| `hang` | sleeps `delay` s, longer than the compactor's 10 s read timeout |

`GET /_fixture/stats` returns per-endpoint call counts, so a test can assert the
guard consults `/tokenize` a **bounded** number of times rather than once per
message — the cost discipline `main.py:846-853` depends on.
`POST /_fixture/reset` clears both. `GET /_fixture/info` reports the baked
tokenizer, window and error style.

`FIXTURE_ERROR_STYLE=legacy` emits the older
`"your prompt contains at least N input tokens"` wording instead of vLLM
0.10.0's. This matters: see the finding below.

## Measured results, 2026-08-28

Full run, 17 test groups, both regimes green.

| | char/4 local (test_smoke's regime) | gpt2 local (tier-2 `encode()+4`) |
| --- | --- | --- |
| prose (control) | 0.88x | 1.02x |
| box-drawing, 1710x U+2501 + 441x U+2500 | **6.34x** (556 -> 3523) | **1.97x** (1786 -> 3523) |
| worst hostile category | 11.17x | 1.97x |
| unscaled summarizer batch vs 29696 budget | 49223 (**+19527 over**) | 49223 (**+19527 over**) |
| pre-fix guard result vs 8000 limit | 42185 (**+34185 over**) | 14081 (**+6081 over**) |
| fixed guard result vs 8000 limit | 7055 (fits) | 7055 (fits) |
| `/tokenize` calls for a 62-message payload | 2 | 2 |

The two "over" rows are the point. They are teeth checks: they re-run the
pre-v3.1 behaviour and assert it overflows **when a real tokenizer measures the
result**. A suite that only asserts the fixed path passes just as happily on
code that never had the fix.

## One finding this fixture surfaced

`main._reported_prompt_tokens` (`main.py:196`) parses the true prompt size out
of vLLM's rejection with:

```python
_CTX_OVERFLOW_RE = re.compile(r"prompt contains (?:at least )?(\d+) input tokens")
```

vLLM 0.10.0 does not emit that wording. `serving_engine.py:625-631` emits:

> `This model's maximum context length is N tokens. However, you requested M tokens (K in the messages, J in the completion). Please reduce the length of the messages or completion.`

Measured against the fixture in both styles:

* `FIXTURE_ERROR_STYLE=legacy` -> `_reported_prompt_tokens` returns `41934`. Parses.
* `FIXTURE_ERROR_STYLE=v010` -> returns `None`.

`_is_context_overflow` still classifies both correctly (it looks only for
`"maximum context length"`), so the user-facing message is right either way.
But the P0-0b calibration learns `_BUDGET_MARGIN` from that parsed number, so
under the v0.10 wording it would learn nothing from a rejection while logging
as though it had. **This is not yet a confirmed production defect** — the
deployed pin is `vllm==0.24.0`, whose wording was not verified from this
machine. Confirm it against the pod before treating it as either a bug or a
non-issue. The fixture reproduces both wordings so the answer is one env var
away.
