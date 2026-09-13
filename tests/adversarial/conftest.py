"""Shared fixtures for the adversarial suite.

Deliberately thin. The whole point of this suite is to attack the PUBLIC
surface, so anything that makes an attack more convenient than it would be
for a real client belongs here only if a real client could do it too.
"""

import os
import pathlib

import httpx
import pytest

# test_adv_v319_{reuse,loop,webuidb}.py are standalone DEMONSTRATION SCRIPTS,
# not pytest suites, despite the test_ prefix and living in the directory
# this stack's Dockerfile CMD collects (["tests/adversarial", ...]). Each
# does `sys.path.insert(0, "/work/compactor")` and expects `main`/`summarizer`
# /`webuidb` importable from there — a layout that exists only after the
# UNIT image's documented `cp -r /src /work && cd /work/compactor` dance
# (see each file's own module docstring for the exact invocation). Under
# THIS (adversarial) stack /work does not exist — its own Dockerfile only
# pip-installs and sets WORKDIR /repo, and the only mount is ./:/repo:ro — so
# `import memory` fails at collection and pytest reports a collection error
# for a file nothing here is meant to run automatically. Ignored rather than
# renamed off the test_ prefix, so the documented `python
# tests/adversarial/test_adv_v319_*.py` invocation in each file's own
# docstring keeps working unchanged.
collect_ignore_glob = ["test_adv_v319_*.py"]

BASE_URL = os.environ.get("ZIONS_TEST_BASE_URL", "http://127.0.0.1:8080").rstrip("/")
FIXTURE_URL = os.environ.get("FIXTURE_URL", "http://vllm-fixture:8000").rstrip("/")
MODEL = os.environ.get("ZIONS_TEST_MODEL", "fixture-model")
TIMEOUT = float(os.environ.get("ZIONS_TEST_TIMEOUT", "300"))
FINDINGS = pathlib.Path("/findings")


@pytest.fixture
def client():
    with httpx.Client(base_url=BASE_URL, timeout=TIMEOUT) as c:
        yield c


@pytest.fixture
def fixture_client():
    """Talks to the vLLM stand-in, for injecting backend faults."""
    with httpx.Client(base_url=FIXTURE_URL, timeout=30.0) as c:
        yield c


@pytest.fixture(autouse=True)
def _reset_fixture_mode(fixture_client):
    """Every adversarial case starts from a clean backend.

    Fault injection is global state on the fixture. Within one project that is
    safe (nothing else is running), but a case that dies mid-fault would
    otherwise poison every case after it and produce a cascade of failures
    that look like findings and are not.

    ALL of set_mode's whitelist, not two of eight. This used to reset only
    tokenize_mode and reply_chars, which was sufficient only by luck of
    gating: factor/status/delay are read solely inside branches keyed off
    tokenize_mode (so resetting that one key neutralised them too),
    reply_seq and reply_looping ride along with reply_chars (dead once
    reply_chars is back to 0), and assistant_final_400 — which IS read
    unconditionally by the fixture — happened to never be set to anything
    but its own default by any case in this directory. The next case that
    sets assistant_final_400 or reply_looping without also setting every
    other key would otherwise leak its mode to every later case in this
    process, silently. Defaults mirror fixture_server.py's own _MODE dict.
    """
    yield
    try:
        fixture_client.post("/_fixture/mode", json={
            "tokenize_mode": "ok",
            "reply_chars": 0,
            "reply_seq": 0,
            "reply_looping": True,
            "factor": 0.5,
            "status": 400,
            "delay": 15.0,
            "assistant_final_400": True,
        })
    except Exception:
        pass


def record(name: str, detail: str) -> None:
    """Write a finding to the host-mounted directory.

    On disk immediately, and not at the end: several cases here kill the
    compactor or the container on purpose, and a finding that only exists in
    a process that is about to die is not a finding.
    """
    try:
        FINDINGS.mkdir(parents=True, exist_ok=True)
        p = FINDINGS / f"{name}.md"
        with p.open("a", encoding="utf-8") as fh:
            fh.write(detail.rstrip() + "\n\n")
    except OSError:
        pass
