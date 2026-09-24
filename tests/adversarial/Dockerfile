# The adversary. Deliberately NOT the compactor image.
#
# Same discipline as tests/integration: this suite must not be able to import
# `main`, because a test that reaches inside can accidentally assert on
# internals and stop being an attack on the PUBLIC surface. It gets an HTTP
# client, a test runner, and the tools an attacker actually needs.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# curl and procps are for the cases that go outside pytest: killing the
# compactor mid-write, watching it come back, poking a raw socket.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        procps \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        pytest>=8 \
        httpx>=0.27 \
        hypothesis>=6.100

WORKDIR /repo
# -p no:cacheprovider because /repo is read-only; without it every run emits
# two warnings about failing to write .pytest_cache, and noise in a summary is
# how a real finding gets skimmed past.
# -x is NOT set: an adversarial run should report every break it finds, not
# stop at the first.
# The path is in CMD, NOT in ENTRYPOINT, and that distinction cost real
# time. With `pytest tests/adversarial` baked into ENTRYPOINT, a path
# passed to `docker compose run` is APPENDED rather than substituted, so
# `run --rm adversary tests/adversarial/test_adv_input.py` quietly ran the
# whole directory AND that file — every adversary's suite against one
# compactor. That is what produced a container exiting 255 and took Docker
# Desktop down mid-session. Now an argument REPLACES the default target,
# which is what everyone assumed it did all along.
ENTRYPOINT ["python", "-m", "pytest", "-p", "no:cacheprovider"]
CMD ["tests/adversarial", "-v", "--tb=short", "-rA"]
