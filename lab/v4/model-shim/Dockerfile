# V4 lab: tiny shim in front of llama.cpp's server giving it the two
# vLLM-shaped endpoints compactor/main.py actually calls. See shim.py for
# why this exists and README/LAB.md "Model parity gaps" for what it does
# NOT cover (structured_outputs / grammar-constrained decoding).
FROM python:3.11-slim-bookworm

# Build context is the zl-v4lab REPO ROOT (see docker-compose.lab.yml) so
# this can share lab/v4/guard.py with the bot and compactor images, the same
# way they share it.
WORKDIR /work
COPY lab/v4/model-shim/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY lab/v4/model-shim/shim.py lab/v4/guard.py ./

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["python3", "shim.py"]
