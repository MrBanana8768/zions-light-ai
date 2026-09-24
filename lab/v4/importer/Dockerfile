# V4 lab importer: vodozemac (not libolm) for Megolm encryption + the
# appservice masquerade, matching the pattern already proven working in
# /home/drew/zl-ops/v4/lab/val_import.py (V1/V2/V5/V6/V7 in that file's
# docstring). Build context is the zl-v4lab REPO ROOT.
FROM python:3.11-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
        libolm-dev libolm3 build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir aiohttp vodozemac

WORKDIR /work
COPY lab/v4/importer/import_history.py lab/v4/guard.py ./

ENV PYTHONUNBUFFERED=1
# No ENTRYPOINT -- v4lab-import.sh passes the full "python3 import_history.py
# ..." command via `docker compose run`, so it isn't prefixed twice.
CMD ["python3", "import_history.py", "--help"]
