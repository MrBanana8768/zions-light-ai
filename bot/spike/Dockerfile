# Phase-1 spike lab image. NOT the bot image — throwaway tooling used to
# answer the go/no-go questions in V4_MATRIX_CLIENT.md. Built and run only
# inside the zlaspike docker network against the copied dev chat-app stack.
FROM python:3.11-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
        libolm-dev \
        libolm3 \
        build-essential \
        postgresql-client \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python3"]
