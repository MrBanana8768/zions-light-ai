"""Adversarial suite: PERSISTENT STATE.

Goal: corrupt, truncate, contradict and poison the compactor's stored memory
and see whether it (a) reports the damage or silently reads it as EMPTY, and
(b) whether any path turns a read error into permanent data loss.

WHY SOME FINDINGS ARE NOT IN THIS FILE. The adversary container is deliberately
HTTP-only (see docker-compose.adversarial.yml: it shares the compactor's netns
for loopback, but does NOT mount the store volume). Damage that has to be
written straight onto a memory file — the wrong-SHAPE-reads-as-empty family
that is the heart of this report — cannot be built from here. Those are
reproduced by tests/adversarial/findings/_scratch_state/*.py (which run inside
the compactor container, where /data is writable) and are written up in
tests/adversarial/findings/state-01..07-*.md. Everything reproducible over the
PUBLIC admin API lives here.

The negative tests below (test_finding_*) assert the CURRENT, defective
behaviour so the suite is green while the break is live; each docstring states
what a fix would change (which would flip that assertion). The positive tests
(test_sound_*) assert behaviour that is correct today and must stay correct.

Every test builds its own damage on a fresh conv_id and best-effort forgets it,
so the file runs start to finish, repeatedly, without polluting the store.
"""

import json
import uuid

import pytest

from conftest import BASE_URL, MODEL, record  # noqa: F401


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _cid(tag: str) -> str:
    return f"advstate-{tag}-{uuid.uuid4().hex[:10]}"


def _bundle(facts=None, summary_state=None, episodic=None) -> dict:
    """A minimal, VALID v2.1 import bundle. Callers poison one field."""
    return {
        "version": "v2.1",
        "exported_at": 0,
        "source_conv_id": "adv-src",
        "facts": [] if facts is None else facts,
        "summary_state": {} if summary_state is None else summary_state,
        "episodic": [] if episodic is None else episodic,
    }


def _import(client, bundle, target, overwrite=False):
    # Send with the stdlib encoder (allow_nan=True), not httpx's json= (which
    # is allow_nan=False and cannot serialise NaN). A real attacker crafting a
    # bundle by hand is under no such restriction, and the compactor parses
    # NaN happily — so the raw-content path is the faithful one.
    payload = {"bundle": bundle, "target_conv_id": target, "overwrite": overwrite}
    return client.post(
        "/admin/conversations/import",
        content=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )


def _forget(client, conv_id):
    try:
        client.delete(f"/admin/conversations/{conv_id}/facts")
    except Exception:
        pass


def _health(client) -> dict:
    return client.get("/health/full").json()


# A fact record that is valid JSON and passes the non-empty-text gate, but
# whose numeric field is a type load_facts' int() coercion cannot handle.
# json allows NaN by default, so it survives the HTTP round-trip.
POISON_FACTS = {
    "added_turn_dict": {"text": "POISON", "added_turn": {}, "last_used": 0},
    "added_turn_nan": {"text": "POISON", "added_turn": float("nan"), "last_used": 0},
    "last_used_list": {"text": "POISON", "added_turn": 0, "last_used": [1]},
}


# ---------------------------------------------------------------------------
# F-S6 — import writes fact records without validation; a malformed one bricks
# every later read of that conversation's facts and permanently stalls fact
# extraction, while import and /health both report success.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(POISON_FACTS))
def test_finding_import_malformed_fact_bricks_facts_reads(client, name):
    """CONFIRMED F-S6 (state-04).

    Current: import returns 200, then GET /facts returns 500 forever, because
    load_facts does int({}) / int(NaN) / int([1]) and raises — a
    non-StoreUnreadable exception that no handler expects.

    A fix (validate bundle facts before save_facts, OR coerce defensively in
    load_facts) would make import 400 the bundle, or make GET /facts 200 with
    the record dropped. Either flips the `== 500` assertion below.
    """
    conv = _cid("fs6-" + name)
    rec = POISON_FACTS[name]
    r = _import(client, _bundle(facts=[rec]), conv)
    assert r.status_code == 200, f"import unexpectedly rejected: {r.text}"
    assert r.json()["imported"]["facts"] == 1

    got = client.get(f"/admin/conversations/{conv}/facts")
    record(
        "state-F-S6-import-poison",
        f"[{name}] import={r.status_code} imported=1; "
        f"GET /facts -> {got.status_code} (expect 500 = the break). "
        f"body={got.text[:120]!r}",
    )
    assert got.status_code == 500, (
        "F-S6 appears fixed: a malformed imported fact no longer 500s the "
        "facts read. Update this test to assert the graceful behaviour."
    )

    _forget(client, conv)


@pytest.mark.parametrize("name", list(POISON_FACTS))
def test_finding_unreadable_facts_do_not_affect_health_status(client, name):
    """ADVFIX REWRITE (P8 gate) — F-S6's health gap is FIXED; this now pins
    the correct behaviour. Was: CONFIRMED F-S6 / health gap (state-04,
    state-01) — `gather_health_full` built `status`/`status_reasons` from
    storage, vLLM, disk-pressure, background shedding, memory-tail skipping
    and /tokenize, but never from `stats.unreadable`, so a corrupt/unreadable
    memory file registered in the body (`stats.unreadable.facts` moved) but
    contributed NOTHING to the top-line status or the Docker HEALTHCHECK.

    Proven fixed live for all three POISON_FACTS shapes
    (`findings/state-F-S6-import-poison.md`): `status_reasons` now carries
    "unreadable memory on disk: N facts. Those conversations are not being
    read and must not be written over; see stats.unreadable." and
    `health.status` is "degraded". This test does NOT assert the top-line
    value stays "degraded" specifically (unrelated tail-skip noise from other
    traffic in this shared-process stack can independently move `status`); it
    asserts the precise fix: the unreadable count rises AND a status_reason
    now references it.

    FAILS IF: stats.unreadable.facts stops rising on this poison, or
    status_reasons stops mentioning "unreadable" for a nonzero
    stats.unreadable.facts.
    """
    conv = _cid("fs6h-" + name)
    r = _import(client, _bundle(facts=[POISON_FACTS[name]]), conv)
    assert r.status_code == 200

    h = _health(client)
    unreadable = h.get("stats", {}).get("unreadable", {}).get("facts")
    reasons = h.get("status_reasons", [])
    mentions_unreadable = any(
        ("unreadable" in s.lower() or "corrupt" in s.lower() or conv in s)
        for s in reasons
    )
    record(
        "state-F-S6-import-poison",
        f"[{name}] health.status={h.get('status')!r} "
        f"stats.unreadable.facts={unreadable} "
        f"reasons_mention_unreadable={mentions_unreadable} "
        f"reasons={reasons!r}",
    )
    assert isinstance(unreadable, int) and unreadable >= 1, (
        "the poison did not register in stats.unreadable.facts — the body "
        "signal is gone too; re-baseline."
    )
    # The FIX: the unreadable layer now has a voice in status_reasons.
    assert mentions_unreadable, (
        "GOOD NEWS did not fully land: stats.unreadable.facts rose but no "
        "status_reason mentions 'unreadable'/'corrupt'/the conv id — the "
        "health gap F-S6 named is back. reasons=%r" % (reasons,)
    )

    _forget(client, conv)


def test_finding_import_summary_nan_chunk_500s_summary_read(client):
    """CONFIRMED (state-03 NaN note): a summary_state whose chunk carries a
    NaN turn-label imports 200 but makes GET /summary 500 — json parses NaN in,
    FastAPI cannot serialise it out. _validate_bundle only checks that
    summary_state is a dict; it does not look inside.

    A fix (reject non-finite numbers on read, or in _validate_bundle) would
    make GET /summary 200 or the import 400.
    """
    conv = _cid("nan-sum")
    ss = {
        "l1": [{"text": "chunk", "first_turn": float("nan"),
                "last_turn": float("nan")}],
        "l2": [], "l3": None,
        "last_summarized_turn": 0, "turns_seen": 0, "tail_fp": [],
    }
    r = _import(client, _bundle(summary_state=ss), conv)
    assert r.status_code == 200, f"import rejected: {r.text}"
    got = client.get(f"/admin/conversations/{conv}/summary")
    record(
        "state-F-S5-summary-nan",
        f"import={r.status_code}; GET /summary -> {got.status_code} "
        f"(expect 500). body={got.text[:100]!r}",
    )
    assert got.status_code == 500, (
        "summary NaN chunk no longer 500s — the read now tolerates or rejects "
        "non-finite turn labels. Update this test."
    )
    _forget(client, conv)


def test_finding_import_over_reports_fact_count(client):
    """Noted (state-04): the import response counts the bundle's list LENGTH,
    not how many records survive a read. A bundle of six junk-but-harmless
    entries reports imported.facts == 6 while load_facts returns 1.
    """
    conv = _cid("overcount")
    junk = ["a string", 123, None, {"no_text_key": 1},
            {"text": "", "added_turn": "x"},
            {"text": "the only real one", "added_turn": 5, "last_used": 5}]
    r = _import(client, _bundle(facts=junk), conv)
    assert r.status_code == 200
    reported = r.json()["imported"]["facts"]
    got = client.get(f"/admin/conversations/{conv}/facts")
    survived = len(got.json()["facts"]) if got.status_code == 200 else None
    record(
        "state-import-overcount",
        f"imported.facts reported={reported}, actually readable={survived}",
    )
    assert reported == 6 and survived == 1, (
        "import fact-count reporting changed; re-baseline this observation."
    )
    _forget(client, conv)


# ---------------------------------------------------------------------------
# Positive controls — behaviour that is CORRECT today and must stay correct.
# ---------------------------------------------------------------------------

def test_sound_import_round_trip(client):
    """A clean bundle round-trips: facts land and read back."""
    conv = _cid("rt")
    facts = [{"text": "sky is teal", "added_turn": 1, "last_used": 1, "pin": False},
             {"text": "code is 4242", "added_turn": 2, "last_used": 2, "pin": True}]
    r = _import(client, _bundle(facts=facts), conv)
    assert r.status_code == 200, r.text
    got = client.get(f"/admin/conversations/{conv}/facts")
    assert got.status_code == 200
    texts = {f["text"] for f in got.json()["facts"]}
    assert {"sky is teal", "code is 4242"} <= texts
    _forget(client, conv)


def test_sound_import_refuses_overwrite_of_live_conversation(client):
    """overwrite=False must refuse a target that already has state, leaving
    the existing facts untouched. This is the guard that stops an import
    silently wiping a live conversation."""
    conv = _cid("live")
    first = [{"text": "ORIGINAL fact", "added_turn": 1, "last_used": 1}]
    assert _import(client, _bundle(facts=first), conv).status_code == 200

    second = [{"text": "REPLACEMENT fact", "added_turn": 1, "last_used": 1}]
    r = _import(client, _bundle(facts=second), conv, overwrite=False)
    assert r.status_code == 400, f"overwrite guard did not fire: {r.text}"

    got = client.get(f"/admin/conversations/{conv}/facts").json()["facts"]
    assert any("ORIGINAL" in f["text"] for f in got)
    assert not any("REPLACEMENT" in f["text"] for f in got)
    _forget(client, conv)


@pytest.mark.parametrize("bad", [
    ({"version": "v9.9", "facts": [], "summary_state": {}, "episodic": []}),
    ({"version": "v2.1", "facts": []}),                       # missing keys
    ({"version": "v2.1", "facts": {}, "summary_state": {}, "episodic": []}),  # facts not a list
    ({"version": "v2.1", "facts": [], "summary_state": [1], "episodic": []}),  # summary not a dict
])
def test_sound_import_rejects_malformed_bundle_shape(client, bad):
    """_validate_bundle rejects the wrong bundle-level shapes with 400."""
    conv = _cid("badbundle")
    r = _import(client, bad, conv)
    assert r.status_code == 400, f"expected 400, got {r.status_code}: {r.text}"


def test_sound_health_full_has_unreadable_block(client):
    """The unreadable accounting exists and is shaped as documented — this is
    the mechanism that F-S1/4/7 slip past (they never raise, so they never
    increment it), so the block itself must at least be present and typed."""
    h = _health(client)
    stats = h.get("stats", {})
    assert "unreadable" in stats
    for layer in ("facts", "episodic", "summaries"):
        assert layer in stats["unreadable"]
        assert isinstance(stats["unreadable"][layer], int)
