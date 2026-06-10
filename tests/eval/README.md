# Quality eval (`tests/eval/`)

The fourth kind of test, beyond the three tiers in [TESTING.md](../../TESTING.md):
a **quality eval**. The tiers prove the code is correct (Tier-1), the deploy is
live (Tier-2 boot self-test), and end-to-end scenarios work (Tier-3). None of
them answer *"is the output actually good?"* — that's what this measures.

Today it covers **speech-to-text (V3.2)**: transcribe known audio clips through
the running Whisper service and score each against its reference transcript
with **Word Error Rate (WER)**.

## Pieces

| File | What | Runs |
|---|---|---|
| `wer.py` | WER metric (word-level Levenshtein over normalized tokens) | pure stdlib |
| `test_wer.py` | Tier-1 unit tests for the metric | any machine, no audio |
| `stt_eval.py` | transcribes `fixtures/*.wav` and scores WER vs `*.txt` | against a **live** STT service |
| `fixtures/` | operator-supplied `<name>.wav` + `<name>.txt` pairs | — |

## Run

```bash
# Unit-test the metric (CPU, no deps):
python tests/eval/test_wer.py

# Score real clips against a running service (needs httpx):
STT_URL=http://127.0.0.1:9000 python tests/eval/stt_eval.py
# or a pod:  STT_URL=https://<POD>-9000.proxy.runpod.net python tests/eval/stt_eval.py
# stricter:  WER_THRESHOLD=0.15 STT_URL=... python tests/eval/stt_eval.py
```

`stt_eval.py` exits `0` if every fixture is at/under `WER_THRESHOLD` (default
`0.30`) — or if there are no fixtures yet — and `1` if any clip exceeds it.

## Adding fixtures

Real speech clips are **operator-supplied** — synthetic silence (what the boot
self-test uses for a liveness probe) can't measure accuracy. See
[`fixtures/README.md`](fixtures/README.md) for the format. A handful of short,
clear clips (5–15s) plus one noisy/accented clip is a good starting set.

> Not baked into the image: `tests/` is excluded via `.dockerignore`. This is a
> developer/operator harness, run against a deployment.
