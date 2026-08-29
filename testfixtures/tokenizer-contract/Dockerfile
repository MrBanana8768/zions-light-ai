# The vLLM-shaped tokenizer fixture. CPU-only, no model weights, no GPU.
#
# Size discipline: this exists so CI can run a REAL tokenizer against the
# compactor's budget code. It must stay small enough that nobody is tempted to
# skip it. Measured 2026-08-28: ~700 MB, versus 4.04 GB on disk for
# public.ecr.aws/q9t5s3a7/vllm-cpu-release-repo:v0.10.0 — which then died with
# SIGILL on an AVX2-only host anyway. See README.md.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/hf \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Bake the tokenizer into the image so the container needs no network at run
# time. A fixture that reaches the internet on every CI start is a fixture that
# fails on the day the network is flaky and gets deleted the week after.
ARG FIXTURE_TOKENIZER=HuggingFaceTB/SmolLM2-135M-Instruct
ENV FIXTURE_TOKENIZER=${FIXTURE_TOKENIZER}
RUN python -c "\
from transformers import AutoTokenizer; \
import os; \
t = AutoTokenizer.from_pretrained(os.environ['FIXTURE_TOKENIZER']); \
assert t.chat_template, 'fixture tokenizer must carry a chat_template — /tokenize takes messages'; \
print('baked', os.environ['FIXTURE_TOKENIZER'], 'vocab', t.vocab_size)"

COPY fixture_server.py /app/fixture_server.py

EXPOSE 8000
CMD ["uvicorn", "fixture_server:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "warning"]
