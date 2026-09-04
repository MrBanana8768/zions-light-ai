"""Shared fixtures for the adversarial suite.

Deliberately thin. The whole point of this suite is to attack the PUBLIC
surface, so anything that makes an attack more convenient than it would be
for a real client belongs here only if a real client could do it too.
"""

import os
import pathlib

import httpx
import pytest

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
    """
    yield
    try:
        fixture_client.post("/_fixture/mode", json={"tokenize_mode": "ok",
                                                    "reply_chars": 0})
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
