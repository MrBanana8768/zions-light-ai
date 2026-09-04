# Runner for the Tier-3 integration suite (docker-compose.integration.yml).
#
# Deliberately tiny, and deliberately NOT the compactor image. This suite's
# whole discipline is that it imports no compactor code and hits the public
# API as a black box; giving it an image that could import `main` would make
# that discipline a convention instead of a fact. Its requirements.txt is two
# lines for the same reason.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# The repo is mounted read-only at /repo by the compose file; pytest.ini and
# conftest.py live beside the tests, so rootdir resolves from there.
WORKDIR /repo
# -p no:cacheprovider because /repo is mounted read-only: without it pytest
# emits two warnings per run about failing to write .pytest_cache, which is
# noise that trains people to skim the summary — and the summary is where the
# SKIP-versus-PASS distinction this project cares about lives.
ENTRYPOINT ["python", "-m", "pytest", "tests/integration", "-p", "no:cacheprovider"]
CMD ["-v", "--tb=short"]
