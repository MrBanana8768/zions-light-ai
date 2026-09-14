"""Adversarial suite: CONCURRENCY.

Races, interleavings and contention against the compactor's per-conversation
locking. Every case here must be able to FAIL on a serialised system — a test
that would pass if the whole server took one global mutex is not a
concurrency test, it is a smoke test with threads. Each case states, in its
docstring, what would make it fail.

Vocabulary used throughout:

  * "the tail" is the fire-and-forget post-response work
    (main._async_tail via bgwork.pool): episodic indexing, fact extraction,
    summary rollup. It completes AFTER the HTTP response, so anything a
    client does immediately after a reply races it.
  * "settle" means: poll /health/full until the background pool reports no
    outstanding work. Without it every assertion below would be racing the
    thing it is measuring rather than the thing under test.

Rates, not single observations. A race seen once in 500 is still a race, and
several cases here report a FAILURE RATE over many attempts rather than
asserting on one.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import os
import time
import uuid

import httpx
import pytest

from conftest import BASE_URL, MODEL, record

TIMEOUT = float(os.environ.get("ZIONS_TEST_TIMEOUT", "300"))
TAIL_WAIT = float(os.environ.get("ZIONS_TEST_TAIL_WAIT", "120"))

# Repetition counts. Overridable so a reviewer can turn the dial up without
# editing the file; the defaults are the numbers these findings were actually
# measured at.
REPS = int(os.environ.get("ADV_RACE_REPS", "60"))
REPS_LONG = int(os.environ.get("ADV_RACE_REPS_LONG", "120"))


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def new_conv(tag: str = "") -> str:
    """A conv_id no other case will touch. Sanitised charset only — memory.py
    strips anything outside [A-Za-z0-9_-] and length-caps at 64, and a case
    that tripped that would be testing the sanitiser by accident."""
    return f"advrace-{tag}-{uuid.uuid4().hex[:12]}"[:64]


def chat_body(user_turns: list[str], *, stream: bool = False) -> dict:
    """A conversation array ending on a user turn.

    `user_turns` are the USER texts; assistant turns are interleaved so the
    array has the alternation the chat template requires. The compactor
    derives its `turn_index` seed from len(messages), so the LENGTH of this
    array is load-bearing for the ordering attacks below.
    """
    messages: list[dict] = [{"role": "system", "content": "you are a test"}]
    for i, t in enumerate(user_turns):
        messages.append({"role": "user", "content": t})
        if i < len(user_turns) - 1:
            messages.append(
                {"role": "assistant", "content": f"prior assistant turn {i}"}
            )
    return {"model": MODEL, "messages": messages, "stream": stream}


def say(c: httpx.Client, conv: str, text: str, **kw) -> httpx.Response:
    """A FIRST-turn shaped request: system + one user turn, no assistant turn."""
    return c.post(
        "/v1/chat/completions",
        json=chat_body([text], **kw),
        headers={"X-Conversation-Id": conv},
    )


def say_hist(c: httpx.Client, conv: str, text: str, **kw) -> httpx.Response:
    """A request that CARRIES CONVERSATIONAL HISTORY, i.e. one assistant turn
    before the new user turn.

    Use this for every repeated request on one conversation. main.py's
    `_is_repeat_task_traffic` (v3.1.8 N4b) skips the memory tail for any
    request with NO assistant turn arriving on a conv_id whose recorded
    position is already past the opening — that is how it declines to
    memorise OpenWebUI's title/tag/follow-up calls. A load generator that
    sends bare single-turn arrays therefore has its tails skipped rather than
    run, and every attack built on it measures nothing. Measured here: 56 of
    64 piled requests came back `skipped_task_traffic` and the pool never
    grew a queue.
    """
    return c.post(
        "/v1/chat/completions",
        json=chat_body(["opening user turn", text], **kw),
        headers={"X-Conversation-Id": conv},
    )


def say_turns(c: httpx.Client, conv: str, turns: list[str], **kw):
    return c.post(
        "/v1/chat/completions",
        json=chat_body(turns, **kw),
        headers={"X-Conversation-Id": conv},
    )


def reply_text(r: httpx.Response) -> str:
    try:
        return r.json()["choices"][0]["message"]["content"]
    except Exception:
        return ""


def health(c: httpx.Client) -> dict:
    return c.get("/health/full").json()


def bg(c: httpx.Client) -> dict:
    return health(c).get("background_work") or {}


def settle(c: httpx.Client, timeout: float = TAIL_WAIT) -> bool:
    """Block until the background pool has drained. False on timeout.

    The single most important helper in this file. The tail is
    fire-and-forget, so an assertion made the instant a response returns is
    an assertion about a store that is still being written.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if int(bg(c).get("outstanding", 1)) == 0:
                # Two consecutive zeroes: `outstanding` drops in the pool's
                # done-callback, which can run a tick before the write it
                # performed is visible to a separate reader.
                time.sleep(0.2)
                if int(bg(c).get("outstanding", 1)) == 0:
                    return True
        except Exception:
            pass
        time.sleep(0.1)
    return False


def export(c: httpx.Client, conv: str) -> dict:
    r = c.get(f"/admin/conversations/{conv}/export")
    r.raise_for_status()
    return r.json()


def episodic_rows(c: httpx.Client, conv: str) -> list[dict]:
    return export(c, conv).get("episodic") or []


def turn_indices(c: httpx.Client, conv: str) -> list[int]:
    out = []
    for row in episodic_rows(c, conv):
        try:
            out.append(int(row.get("turn_index")))
        except (TypeError, ValueError):
            pass
    return sorted(out)


def inventory(c: httpx.Client, conv: str) -> dict:
    r = c.get(f"/admin/conversations/{conv}")
    r.raise_for_status()
    return r.json()


def facts_of(c: httpx.Client, conv: str) -> list[dict]:
    r = c.get(f"/admin/conversations/{conv}/facts")
    r.raise_for_status()
    return r.json().get("facts") or []


def fact_texts(c: httpx.Client, conv: str) -> set[str]:
    return {f.get("text", "") for f in facts_of(c, conv)}


def parallel(fns):
    """Run zero-argument callables AT THE SAME TIME; collect (ok, value).

    Threads, not asyncio: the point is to be N INDEPENDENT CLIENTS, and the
    compactor's whole defence is a per-conversation asyncio lock on ITS event
    loop. Driving it from one client's event loop would let the test's own
    scheduling pick the interleaving.
    """
    out = []
    with cf.ThreadPoolExecutor(max_workers=max(2, len(fns))) as ex:
        futs = [ex.submit(f) for f in fns]
        for f in futs:
            try:
                out.append((True, f.result()))
            except Exception as e:  # noqa: BLE001 — the exception IS a result
                out.append((False, e))
    return out


def barrier_parallel(fns):
    """Like `parallel`, but every callable is held at a barrier and released
    together, so the requests hit the server inside the same few
    milliseconds instead of being staggered by thread startup."""
    import threading

    b = threading.Barrier(len(fns))

    def wrap(f):
        def inner():
            b.wait()
            return f()
        return inner

    return parallel([wrap(f) for f in fns])


def fresh_client() -> httpx.Client:
    return httpx.Client(base_url=BASE_URL, timeout=TIMEOUT)


# ---------------------------------------------------------------------------
# A corpus of MUTUALLY UNRELATED facts.
#
# This is not decoration, it is a correctness requirement, and getting it wrong
# cost a full round of false positives. Templated payloads
# ("merged payload 0-0 concerning pangolins", "... 0-1 concerning pangolins")
# are near-duplicates to an embedding model, so the INLINE DEDUP that runs at
# the end of every memory tail clusters and merges them. Measured on the
# adv-race stack, against a merge that had just landed 40 templated facts:
#
#   dedup pass: 41 fact(s) in, 10 candidate cluster(s), 10 LLM call(s),
#               10 merge(s), 30 fact(s) removed
#
# 30 of the 40 gone, with `archived: []` — dedup REMOVES, it does not archive.
# Read without the log line that is indistinguishable from a lost update, and
# it is not one: it is dedup doing its job on input that deserved it.
#
# So every fact a race test plants has to be something dedup will keep.
_DISTINCT_FACTS = [
    "the reactor coolant valve is rated to nine hundred kilopascals",
    "her grandmother was born in Trondheim during a blackout",
    "the office kettle descales itself every fourteen days",
    "sourdough starter number four is named Gerald",
    "the loft hatch sticks unless you lift it from the left",
    "parking permit renewal falls due on the ninth of March",
    "the cat will only drink from a running tap",
    "his passport expires the week before the Lisbon trip",
    "the boiler pressure gauge reads low but is calibrated wrong",
    "chapter eleven of the manuscript was cut entirely",
    "the neighbour's fence encroaches by eleven centimetres",
    "she plays viola, not violin, and minds the distinction",
    "the archive tapes are stored at fifteen degrees Celsius",
    "flight BA283 lands at terminal five, not three",
    "the piano needs tuning after every heating season",
    "rosemary will not survive in that clay soil",
    "the server room key is held by facilities, not security",
    "her thesis defence is scheduled for a Tuesday morning",
    "the van's clutch was replaced at ninety thousand miles",
    "anchovies are the one ingredient he refuses outright",
    "the fire drill assembly point moved to the south car park",
    "that recording was made on a borrowed Neumann microphone",
    "the greenhouse thermostat drifts two degrees high in summer",
    "his brother trained as a farrier before switching to law",
    "the lease break clause requires ninety days of notice",
    "the darkroom timer counts down in half seconds",
    "she took up open water swimming after the injury",
    "the attic insulation was topped up two winters ago",
    "the supplier in Leeds discontinued the brass fittings",
    "his hearing aid needs a size ten twelve battery",
    "the recipe fails above two thousand metres of altitude",
    "the wedding photographs were lost when the drive failed",
    "the allotment has a standpipe but no electricity",
    "she reads the paper back to front, obituaries first",
    "the telescope mount needs polar alignment each session",
    "the smoke alarm in the hall chirps at low battery",
    "his first car was a diesel estate with no radio",
    "the border collie is deaf in one ear from birth",
    "the pension statement arrives in October each year",
    "the roof slates are Welsh, and no longer quarried",
    "she is allergic to the preservative in tinned olives",
    "the departmental printer only duplexes on A4",
    "the harbour gates close two hours either side of low tide",
    "the manuscript's dedication names a teacher, not a relative",
    "the freezer drawer runs warmer than the others",
    "his running shoes need replacing every six hundred kilometres",
    "the well water tests high for iron but not nitrate",
    "the string quartet lost its cellist to an orchestra post",
]


def distinct_facts(n: int, salt: str) -> list[str]:
    """`n` facts that inline dedup will not cluster.

    `salt` keeps two conversations' corpora textually distinct without making
    the members of either near-duplicates of each other: the salt goes on the
    END of an already-unrelated sentence, so it separates conversations
    without pulling the sentences within one conversation together.
    """
    assert n <= len(_DISTINCT_FACTS), (
        f"only {len(_DISTINCT_FACTS)} distinct facts available; asking for {n} "
        f"would force templated near-duplicates and dedup would eat them"
    )
    return [f"{_DISTINCT_FACTS[i]} ({salt})" for i in range(n)]


def rate_line(name: str, hits: int, reps: int) -> str:
    pct = (100.0 * hits / reps) if reps else 0.0
    return f"{name}: {hits}/{reps} attempts ({pct:.1f}%)"


# ---------------------------------------------------------------------------
# 1. Same conv_id, many clients — turn index integrity
# ---------------------------------------------------------------------------

def test_parallel_chats_do_not_collide_turn_indices(client):
    """N simultaneous chats on ONE conversation must produce N episodic rows
    with N DISTINCT turn indices.

    WHAT WOULD MAKE THIS FAIL: retrieval._next_turn_index allocates from the
    store's current maximum. If index_exchange did not hold conv_lock (or if
    the lock were per-request rather than per-conv) two tails would read the
    same maximum and allocate the same ordinal. Duplicate ordinals are not
    cosmetic — admin_compact rebuilds the transcript BY SLOT from them.

    Each request carries a DISTINCT final user turn on purpose: _doc_id is
    content-addressed since v3.1 D1, so N byte-identical exchanges collapse
    to one row by design and would hide the very collision being hunted.
    """
    n = 12
    conv = new_conv("par")
    clients = [fresh_client() for _ in range(n)]
    try:
        fns = [
            (lambda i=i: say_hist(clients[i], conv, f"parallel probe number {i}"))
            for i in range(n)
        ]
        results = barrier_parallel(fns)
    finally:
        for c in clients:
            c.close()

    bad = [v for ok, v in results if not ok or v.status_code != 200]
    assert not bad, f"non-200 under parallel load on one conv: {bad[:3]}"
    assert settle(client), "pool never drained"

    idx = turn_indices(client, conv)
    rows = episodic_rows(client, conv)
    dupes = sorted({i for i in idx if idx.count(i) > 1})
    detail = (
        f"conv={conv} n={n} rows={len(rows)} indices={idx} duplicates={dupes}"
    )
    if dupes or len(rows) != n:
        record("race-turn-index", f"## parallel chats on one conv\n\n{detail}")
    assert not dupes, f"duplicate turn_index under parallel chat: {detail}"
    assert len(rows) == n, (
        f"exchanges vanished under parallel chat on one conv: {detail}"
    )


def test_parallel_chats_leave_a_consistent_position(client):
    """The summarizer's position counter must not be pushed past the
    conversation by concurrent rollups.

    turns_seen is read-modify-written per rollup under conv_lock. N
    concurrent tails on one conv each call maybe_rollup with THEIR OWN
    snapshot of the client array. If the position were advanced once per
    call rather than once per new exchange, turns_seen would run ahead of
    the exchanges the store actually holds, and every later chunk label
    would name turns the summary does not cover.

    WHAT WOULD MAKE THIS FAIL: an over-count (position > 2 * exchanges + a
    small allowance) or a stall at zero while exchanges accumulated.
    """
    n = 10
    conv = new_conv("pos")
    clients = [fresh_client() for _ in range(n)]
    try:
        barrier_parallel([
            (lambda i=i: say_hist(clients[i], conv, f"position probe {i}"))
            for i in range(n)
        ])
    finally:
        for c in clients:
            c.close()
    assert settle(client)

    inv = inventory(client, conv)
    seen = int((inv.get("summary") or {}).get("turns_seen") or 0)
    rows = len(episodic_rows(client, conv))
    detail = f"conv={conv} n={n} turns_seen={seen} episodic_rows={rows}"
    record("race-position", f"## parallel chats, position\n\n{detail}")
    # Every request sent a 2-turn array, so no serial ordering can put the
    # position above 2 turns per exchange.
    assert seen <= 2 * n, f"position ran ahead of the conversation: {detail}"
    assert seen > 0, f"position never advanced at all: {detail}"


# ---------------------------------------------------------------------------
# 2. Admin forget vs the tail
# ---------------------------------------------------------------------------

def _residue(c: httpx.Client, conv: str) -> dict:
    inv = inventory(c, conv)
    return {
        "facts": inv.get("facts", {}).get("count"),
        "episodic": (inv.get("episodic") or {}).get("indexed_exchanges"),
        "turns_seen": (inv.get("summary") or {}).get("turns_seen"),
        "l1": (inv.get("summary") or {}).get("l1_chunks"),
    }


@pytest.mark.parametrize("reps", [REPS])
def test_admin_forget_races_the_in_flight_tail(client, reps):
    """DELETE /admin/conversations/{id}/facts fired while the previous
    turn's tail is still writing.

    THE ASYMMETRY BEING ATTACKED. The chat command /forget goes through
    commands._handle_forget, which drains the background pool BEFORE the
    wipe, drains again AFTER it, re-reads every layer from disk, and wipes a
    second time if anything survived. The admin endpoint calls
    main._clear_all_memory bare: no drain, no re-read, no retry. Its
    docstring says "Forget ALL memory for a conversation".

    _clear_all_memory holds conv_lock, but the tail takes conv_lock THREE
    SEPARATE TIMES (episodic index, then facts, then the rollup inside
    maybe_rollup) precisely so it does not hold it across an LLM call. A
    wipe that lands in either gap deletes what job 1 wrote and then watches
    jobs 2 and 3 write the same exchange back.

    WHAT WOULD MAKE THIS FAIL: memory belonging to the forgotten exchange
    present after the wipe returned 200 with its counters. Reported as a
    rate over `reps` attempts.
    """
    hits = []
    for i in range(reps):
        conv = new_conv(f"af{i}")
        r = say(client, conv, f"remember that attempt {i} used the colour teal")
        assert r.status_code == 200, r.text
        # No settle: the whole point is to arrive while the tail is in flight.
        d = client.delete(f"/admin/conversations/{conv}/facts")
        assert d.status_code == 200, f"admin forget failed: {d.status_code} {d.text}"
        wipe = d.json()
        assert settle(client), "pool never drained"
        after = _residue(client, conv)
        left = {k: v for k, v in after.items() if v}
        if left:
            hits.append({"conv": conv, "wipe_said": wipe, "left": left})

    line = rate_line("admin forget left memory behind", len(hits), reps)
    if hits:
        record(
            "race-admin-forget",
            "## CONFIRMED: admin forget is not drained against the tail\n\n"
            f"{line}\n\n"
            "DELETE /admin/conversations/{id}/facts returned 200 with wipe "
            "counters, and memory for the just-forgotten exchange was on "
            "disk once the tail finished. The chat command /forget drains "
            "the pool and re-verifies; this endpoint does neither.\n\n"
            "First three:\n```json\n"
            + json.dumps(hits[:3], indent=2)
            + "\n```\n",
        )
    print("\n" + line)
    assert not hits, (
        line
        + f" — example: {json.dumps(hits[0], indent=2) if hits else ''}"
    )


@pytest.mark.parametrize("reps", [REPS])
def test_chat_forget_command_races_the_in_flight_tail(client, reps):
    """The same attack against the PROTECTED sibling.

    /forget as a chat command drains, wipes, drains, verifies, retries. This
    case exists so the finding above is attributable to the missing drain
    and not to the attack being unlucky: same timing, same conversation
    shape, different endpoint.

    WHAT WOULD MAKE THIS FAIL: residue after the command answered — which
    would mean the drain/verify/retry does not close the window it claims
    to.
    """
    hits = []
    for i in range(reps):
        conv = new_conv(f"cf{i}")
        r = say(client, conv, f"remember that attempt {i} used the colour ochre")
        assert r.status_code == 200
        f = say(client, conv, "/forget")
        assert f.status_code == 200, f.text
        assert settle(client)
        after = _residue(client, conv)
        left = {k: v for k, v in after.items() if v}
        if left:
            hits.append({"conv": conv, "said": reply_text(f), "left": left})

    line = rate_line("/forget command left memory behind", len(hits), reps)
    if hits:
        record(
            "race-chat-forget",
            f"## /forget (chat command) residue\n\n{line}\n\n```json\n"
            + json.dumps(hits[:3], indent=2)
            + "\n```\n",
        )
    print("\n" + line)
    assert not hits, line


# ---------------------------------------------------------------------------
# 3. /remember vs the tail — the documented race's siblings
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("reps", [REPS])
def test_remember_acknowledged_facts_survive_the_tail(client, reps):
    """A fact the compactor said "Remembered:" must still be there next turn.

    This is the race that already shipped once (commands._handle_remember's
    comment). The fix was to hold conv_lock across the load-modify-write.
    This fires /remember into the window where the PREVIOUS turn's tail is
    doing its own load-modify-write of the same file.

    WHAT WOULD MAKE THIS FAIL: the tail's save landing on top of the
    /remember, so the acknowledged fact is absent once everything settles.
    """
    hits = []
    for i in range(reps):
        conv = new_conv(f"rm{i}")
        secret = f"the codeword for attempt {i} is zarquon{i}"
        r = say(client, conv, f"attempt {i}: tell me something")
        assert r.status_code == 200
        # Straight into the tail's window.
        rr = say(client, conv, f"/remember {secret}")
        assert rr.status_code == 200, rr.text
        ack = reply_text(rr)
        assert "Remembered" in ack, f"unexpected /remember reply: {ack!r}"
        assert settle(client)
        texts = fact_texts(client, conv)
        if secret not in texts:
            hits.append({"conv": conv, "ack": ack, "facts": sorted(texts)})

    line = rate_line("acknowledged /remember fact lost", len(hits), reps)
    if hits:
        record(
            "race-remember",
            "## CONFIRMED: /remember acknowledged a fact that is not stored\n\n"
            f"{line}\n\n```json\n" + json.dumps(hits[:3], indent=2) + "\n```\n",
        )
    print("\n" + line)
    assert not hits, line


def test_concurrent_remembers_all_land(client):
    """K simultaneous /remember on one conversation, each acknowledged with a
    running count, must produce K facts.

    WHAT WOULD MAKE THIS FAIL: a lost update. Each handler does
    load -> append -> save under conv_lock; if the lock did not cover the
    READ (the shape of the bug the file's own comment describes) two
    handlers would both load the same list and the second save would drop
    the first's fact — while both replies said "Remembered:".
    """
    k = 16
    conv = new_conv("crm")
    secrets = [f"concurrent codeword {i} is quux{i}" for i in range(k)]
    clients = [fresh_client() for _ in range(k)]
    try:
        results = barrier_parallel([
            (lambda i=i: say(clients[i], conv, f"/remember {secrets[i]}"))
            for i in range(k)
        ])
    finally:
        for c in clients:
            c.close()
    acks = [reply_text(v) for ok, v in results if ok]
    assert all("Remembered" in a for a in acks), f"some /remember failed: {acks}"
    assert settle(client)

    texts = fact_texts(client, conv)
    missing = [s for s in secrets if s not in texts]
    detail = (
        f"conv={conv} k={k} stored={len(texts)} missing={missing}\n"
        f"acks={acks}"
    )
    if missing:
        record(
            "race-remember-concurrent",
            "## CONFIRMED: concurrent /remember lost an acknowledged fact\n\n"
            + detail,
        )
    assert not missing, f"lost update across concurrent /remember: {detail}"


# ---------------------------------------------------------------------------
# 2b. Admin forget vs a tail that has not started yet
# ---------------------------------------------------------------------------

# The vLLM stand-in is one uvicorn process. Past roughly a dozen simultaneous
# connections it starts refusing, the compactor relays 503, and the run stops
# measuring the compactor and starts measuring the fixture. Every burst below
# goes out in waves of this size.
WAVE = 8


def _pile_wave(conv: str, tag: str):
    """One wave of WAVE simultaneous history-carrying requests on ONE conv."""
    cs = [fresh_client() for _ in range(WAVE)]
    try:
        return barrier_parallel([
            (lambda i=i: say_hist(
                cs[i], conv, f"{tag} pile turn {i} {uuid.uuid4().hex[:8]}"
            ))
            for i in range(WAVE)
        ])
    finally:
        for c in cs:
            c.close()


def _backlog(client: httpx.Client, tag: str, target: int = 18,
             max_waves: int = 14) -> int:
    """Fill bgwork's queue with tails belonging to ONE OTHER conversation.

    bgwork._run awaits the concurrency semaphore BEFORE the coroutine, so a
    tail submitted past MAX_CONCURRENT (4) has executed ZERO LINES — it holds
    a facts snapshot and nothing else. commands._settle_background_work
    documents exactly that state; this manufactures it, so a tail submitted
    after this call is PARKED rather than racing.

    Why one conversation rather than many: tails on DISTINCT conversations
    retire as fast as the requests arrive and the queue never grows (measured:
    outstanding never exceeded 4 over 32 requests across 32 convs). Tails on
    ONE conversation serialise on that conv's lock, so the queue grows by
    roughly half of every wave (measured: 8 -> 33 over 8 waves of 8).

    Returns the outstanding count reached.
    """
    filler = new_conv(f"{tag}pile")
    out = 0
    for _w in range(max_waves):
        _pile_wave(filler, tag)
        out = int(bg(client).get("outstanding", 0))
        if out >= target:
            break
    return out


@pytest.mark.parametrize("reps", [max(5, REPS // 4)])
def test_admin_forget_against_a_queued_tail(client, reps):
    """The same missing drain, driven to its worst case.

    The pool's concurrency cap is 4. Fill it, then chat on the victim and
    DELETE the victim's memory immediately. The victim's tail is parked on
    the semaphore having run no lines at all, so the wipe deletes nothing of
    it — and then the tail writes the whole exchange (episodic row, extracted
    facts, summary position) into a conversation the operator has been told
    is forgotten.

    WHAT WOULD MAKE THIS FAIL: episodic rows or facts for the wiped exchange
    present afterwards. The chat command's twin drains the pool first and is
    immune; this endpoint does neither drain nor verify.
    """
    hits = []
    inconclusive = 0
    for i in range(reps):
        victim = new_conv(f"vic{i}")
        queued = _backlog(client, tag=f"vf{i}")
        # The victim's turn goes in LAST, so its tail is submitted behind the
        # backlog and cannot have run a line when the wipe arrives.
        r = say(client, victim, f"victim turn {i}: the passphrase is hollyhock{i}")
        if r.status_code != 200:
            inconclusive += 1
            continue
        at_wipe = int(bg(client).get("outstanding", 0))
        d = client.delete(f"/admin/conversations/{victim}/facts")
        assert d.status_code == 200, d.text
        wipe = d.json()
        assert settle(client, timeout=TAIL_WAIT * 2), "pool never drained"
        after = _residue(client, victim)
        left = {k: v for k, v in after.items() if v}
        if at_wipe <= 4:
            # The backlog drained before the wipe went out; this attempt did
            # not actually manufacture the interleaving it is testing.
            inconclusive += 1
            continue
        if left:
            hits.append({
                "conv": victim, "queued_before": queued,
                "outstanding_at_wipe": at_wipe,
                "wipe_said": wipe, "left": left,
            })

    line = (
        rate_line("admin forget vs QUEUED tail left memory", len(hits), reps)
        + f" ({inconclusive} attempt(s) could not build a backlog and were "
        f"not counted either way)"
    )
    if hits:
        record(
            "race-admin-forget",
            "## CONFIRMED (worst case): admin forget vs a tail that has not "
            "started\n\n"
            f"{line}\n\n"
            "The pool's 4 concurrency slots were filled with filler "
            "conversations before the victim's turn, so the victim's tail was "
            "parked on the semaphore with zero lines executed when "
            "DELETE /admin/conversations/<id>/facts ran. The wipe reported its "
            "counters; the tail then wrote the exchange back.\n\n"
            "```json\n" + json.dumps(hits[:3], indent=2) + "\n```\n",
        )
    print("\n" + line)
    assert not hits, line


# ---------------------------------------------------------------------------
# 3b. Admin vs admin — merge runs in a THREADPOOL and holds no lock
# ---------------------------------------------------------------------------

def _remember_many(c: httpx.Client, conv: str, texts: list[str]) -> None:
    """Seed facts THROUGH /remember, one round trip each.

    Use this only where the acknowledgement itself is under test. For plain
    setup use _seed_facts, which is one request instead of len(texts).
    """
    for t in texts:
        r = say(c, conv, f"/remember {t}")
        assert r.status_code == 200, r.text
        assert "Remembered" in reply_text(r), reply_text(r)


def _seed_facts(c: httpx.Client, conv: str, texts: list[str]) -> None:
    """Plant a fact list in ONE request, via the import endpoint.

    Setup only. Seeding twenty facts through /remember is twenty round trips
    at roughly a second each, which put the rate-measuring cases below out of
    reach — and a race that needs thirty attempts to show cannot be measured
    if each attempt costs a minute of setup.
    """
    now = int(time.time())
    r = c.post(
        "/admin/conversations/import",
        json={
            "bundle": {
                "version": "v2.1",
                "exported_at": now,
                "source_conv_id": conv,
                "facts": [
                    {"text": t, "added_turn": 1, "last_used": now, "pin": False}
                    for t in texts
                ],
                "summary_state": {},
                "episodic": [],
            },
            "target_conv_id": conv,
            "overwrite": True,
        },
    )
    assert r.status_code == 200, f"seed failed: {r.status_code} {r.text[:300]}"


def test_two_merges_into_one_conversation_lose_facts(client):
    """Two merges into the SAME destination, at the same time, many times.

    portability.merge_conversation is called through run_in_threadpool, so it
    runs on a WORKER THREAD. Its only mutual exclusion is a single
    `memory.conv_lock(dst).locked()` probe — an asyncio lock, which a thread
    can neither hold nor wait on — followed by an unsynchronised
    `load_facts -> _merge_fact_lists -> save_facts`. The comment above that
    probe says it "cannot await the lock - it can only refuse to write
    underneath a holder". Two merges hold nothing, so neither refuses.

    Two threads, no lock, one file, read-modify-write: whichever saves second
    writes a list built from a read that predates the other's save.

    The window is short (a JSON read, a list fold, an atomic write), so this
    is measured as a RATE over many attempts rather than asserted on one.

    WHAT WOULD MAKE THIS FAIL: dst missing facts that a merge reported as
    `facts_added` — a lost update with HTTP 200 and a count on both sides.
    """
    reps = REPS
    n = 20
    hits = []
    for i in range(reps):
        dst = new_conv(f"mdst{i}")
        src_a = new_conv(f"msrca{i}")
        src_b = new_conv(f"msrcb{i}")
        # Distinct, not templated (see _DISTINCT_FACTS), and drawn from
        # DISJOINT halves of the corpus so neither merge can be credited with
        # the other's facts.
        a_texts = [f"{t} (merge-a-{i})" for t in _DISTINCT_FACTS[:n]]
        b_texts = [f"{t} (merge-b-{i})" for t in _DISTINCT_FACTS[n:2 * n]]
        _seed_facts(client, src_a, a_texts)
        _seed_facts(client, src_b, b_texts)
        _seed_facts(client, dst, ["the destination seed fact for this run"])

        ca, cb = fresh_client(), fresh_client()
        try:
            res = barrier_parallel([
                (lambda: ca.post(
                    f"/admin/conversations/{src_a}/merge-into/{dst}",
                    json={"dry_run": False},
                )),
                (lambda: cb.post(
                    f"/admin/conversations/{src_b}/merge-into/{dst}",
                    json={"dry_run": False},
                )),
            ])
        finally:
            ca.close()
            cb.close()
        assert settle(client)

        bodies = []
        for ok, v in res:
            assert ok, f"merge raised: {v}"
            bodies.append({
                "status": v.status_code,
                "body": v.json() if v.status_code < 500 else v.text[:300],
            })
        # A merge that REFUSED (400/409) lost nothing and is not a hit; the
        # refusal is the guard working. Only a 200 that reported facts_added
        # and then has none of them counts.
        got = fact_texts(client, dst)
        missing = []
        for b, texts in zip(bodies, (a_texts, b_texts)):
            if b["status"] != 200 or not isinstance(b["body"], dict):
                continue
            if not int(b["body"].get("facts_added") or 0):
                continue
            missing += [t for t in texts if t not in got]
        if missing:
            hits.append({
                "dst": dst,
                "responses": bodies,
                "missing": len(missing),
                "of": 2 * n,
                "example_missing": missing[:3],
                "stored": len(got),
            })

    line = rate_line("concurrent merges lost acknowledged facts", len(hits), reps)
    if hits:
        record(
            "race-merge",
            "## CONFIRMED: concurrent merge-into loses facts it reported "
            "adding\n\n"
            "`portability.merge_conversation` runs on a threadpool worker and "
            "holds NO lock - its `conv_lock(dst).locked()` probe is an asyncio "
            "lock a thread can neither take nor wait on. Two merges into one "
            "destination both pass the probe and both do an unsynchronised "
            "load-modify-write of the same file.\n\n"
            f"{line}\n\n```json\n"
            + json.dumps(hits[:3], indent=2) + "\n```\n",
        )
    print("\n" + line)
    assert not hits, line


def test_merge_into_a_conversation_with_a_queued_tail(client):
    """merge-into a conversation whose tail has not started.

    The `locked()` probe is a snapshot taken on a worker thread. A tail parked
    on bgwork's semaphore holds no lock yet, so the probe passes; the tail then
    takes the lock and writes. Which of the two writes lands second is not
    ordered by anything.

    WHAT WOULD MAKE THIS FAIL: merged facts absent after everything settles,
    with the merge having answered 200 and named a count.
    """
    reps = max(4, REPS // 6)
    hits = []
    for i in range(reps):
        src = new_conv(f"qsrc{i}")
        dst = new_conv(f"qdst{i}")
        texts = distinct_facts(20, f"merge-queued-{i}")
        _seed_facts(client, src, texts)

        _backlog(client, tag=f"qf{i}")
        d = say(client, dst, f"destination turn {i}")
        assert d.status_code in (200, 503), d.text

        m = client.post(
            f"/admin/conversations/{src}/merge-into/{dst}", json={"dry_run": False}
        )
        assert settle(client)
        if m.status_code == 200:
            got = fact_texts(client, dst)
            missing = [t for t in texts if t not in got]
            if missing:
                hits.append({
                    "dst": dst, "merge": m.json(),
                    "missing": len(missing), "of": len(texts),
                })
        elif m.status_code not in (400, 409):
            hits.append({
                "dst": dst, "unexpected_status": m.status_code, "body": m.text[:300],
            })

    line = rate_line("merge lost facts against a queued tail", len(hits), reps)
    if hits:
        record(
            "race-merge",
            f"## merge-into vs a queued tail\n\n{line}\n\n```json\n"
            + json.dumps(hits[:3], indent=2) + "\n```\n",
        )
    print("\n" + line)
    assert not hits, line


# ---------------------------------------------------------------------------
# 5. Shedding — what happens to work the pool drops
# ---------------------------------------------------------------------------

def test_shed_work_is_counted_as_stored(client):
    """Drive bgwork past its outstanding ceiling and compare the health
    surface against the store.

    main._run_memory_tail calls `tailhealth.note(STORED)` and THEN
    `_fire_and_forget`, which DISCARDS bgwork.pool.submit's boolean. A tail
    the pool sheds is therefore counted as `stored` in
    /health/full.memory_tail.outcomes for an exchange that never reached any
    memory layer.

    The project's doctrine is that a silent skip is the cardinal sin, and
    tailhealth.py exists because 63 skipped exchanges reported as `ok`. This
    is the same disagreement with the opposite sign: the counter claims work
    that did not happen.

    WHAT WOULD MAKE THIS FAIL: `stored` growing by more than the number of
    exchanges the store actually holds, while shedding occurred. On a pool
    that never sheds the two agree, and this case says it could not provoke
    one rather than passing quietly.
    """
    conv = new_conv("shed")
    # Measured growth is roughly half a wave of backlog per wave, so the
    # 64-task ceiling needs about eighteen. Twenty-four so the shed is a
    # sustained condition rather than a single edge event.
    waves = 24
    n = waves * WAVE
    before_stored = int(
        health(client).get("memory_tail", {}).get("outcomes", {}).get("stored", 0)
    )
    before_shed = int(bg(client).get("shed", 0))

    # ONE conversation on purpose: every tail contends for the same conv_lock,
    # so outstanding work piles up far faster than it retires. Sent in small
    # waves rather than one 128-way burst, because the vLLM stand-in is a
    # single uvicorn and a burst that makes IT fail measures the wrong thing.
    res = []
    peak = 0
    for w in range(waves):
        # say_hist, not say: without an assistant turn in the array every
        # request after the first is `skipped_task_traffic` and fires no tail
        # at all, so the queue never grows and the ceiling is never reached.
        res += _pile_wave(conv, f"burst{w}")
        peak = max(peak, int(bg(client).get("outstanding", 0)))
    ok200 = sum(1 for ok, v in res if ok and getattr(v, "status_code", 0) == 200)
    errs = [
        (ok, getattr(v, "status_code", repr(v)[:120]))
        for ok, v in res
        if not ok or getattr(v, "status_code", 0) != 200
    ]
    assert settle(client, timeout=TAIL_WAIT * 4), "pool never drained after burst"

    after = health(client)
    after_stored = int(after.get("memory_tail", {}).get("outcomes", {}).get("stored", 0))
    after_bg = after.get("background_work") or {}
    shed = int(after_bg.get("shed", 0)) - before_shed
    stored_delta = after_stored - before_stored
    rows = len(episodic_rows(client, conv))

    detail = (
        f"conv={conv}\n"
        f"requests sent: {n}, HTTP 200: {ok200}, other: {errs[:5]}\n"
        f"peak background_work.outstanding observed: {peak} "
        f"(ceiling {after_bg.get('max_outstanding')})\n"
        f"memory_tail.outcomes.stored delta: {stored_delta}\n"
        f"background_work.shed delta: {shed}\n"
        f"episodic rows actually in the store: {rows}\n"
        f"health status after: {after.get('status')!r} "
        f"reasons={after.get('status_reasons')}\n"
    )
    print("\n" + detail)
    record("race-shedding", f"## shedding vs the health counters\n\n{detail}")

    if shed == 0:
        pytest.skip(
            "could not provoke a shed with this burst; the ceiling is "
            f"{after_bg.get('max_outstanding')} and the tails retired too "
            "fast for the queue to reach it"
        )
    assert stored_delta <= rows, (
        "/health/full counted memory-tail work that never reached the store:\n"
        + detail
    )


# ---------------------------------------------------------------------------
# 6. Import vs a queued tail — what the locked() guard cannot see
# ---------------------------------------------------------------------------

def test_import_lands_under_a_queued_tail(client):
    """Import a bundle into a conversation whose tail is parked on the pool's
    semaphore.

    portability.import_conversation refuses when `conv_lock(target).locked()`
    — and its own comment names the hazard it is defending against: "an
    extraction tail that read facts before the import ran, is parked on its
    vLLM call while holding conv_lock, and writes that pre-import snapshot
    back the moment it returns."

    A tail parked on bgwork's SEMAPHORE is in the same state minus the lock:
    it holds a pre-import facts snapshot and has executed no lines. locked()
    is False, so the guard admits the import — the exact situation the guard
    exists to refuse, entered through the door it does not watch.

    WHAT WOULD MAKE THIS FAIL: bundle facts absent after everything settles,
    with the import having answered 200 and `overwrote_existing`.
    """
    reps = max(4, REPS // 6)
    hits = []
    inconclusive = 0
    for i in range(reps):
        target = new_conv(f"imp{i}")
        payload = distinct_facts(20, f"import-{i}")
        bundle = {
            "version": "v2.1",
            "exported_at": int(time.time()),
            "source_conv_id": new_conv(f"impsrc{i}"),
            "facts": [
                {"text": t, "added_turn": 1, "last_used": int(time.time()), "pin": False}
                for t in payload
            ],
            "summary_state": {},
            "episodic": [],
        }
        _backlog(client, tag=f"if{i}")
        r = say(client, target, f"target turn {i} before the import")
        if r.status_code != 200:
            inconclusive += 1
            continue
        at_import = int(bg(client).get("outstanding", 0))
        imp = client.post(
            "/admin/conversations/import",
            json={"bundle": bundle, "target_conv_id": target, "overwrite": True},
        )
        assert settle(client, timeout=TAIL_WAIT * 2)
        if at_import <= 4:
            inconclusive += 1
            continue
        if imp.status_code == 400:
            # The guard refused. That is the behaviour it advertises.
            continue
        assert imp.status_code == 200, f"{imp.status_code} {imp.text[:300]}"
        got = fact_texts(client, target)
        missing = [t for t in payload if t not in got]
        if missing:
            hits.append({
                "target": target, "outstanding_at_import": at_import,
                "import_said": imp.json(), "missing": len(missing),
                "of": len(payload),
            })

    line = (
        rate_line("import lost bundle facts to a queued tail", len(hits), reps)
        + f" ({inconclusive} attempt(s) could not build a backlog)"
    )
    if hits:
        record(
            "race-import",
            "## import vs a tail queued on the pool semaphore\n\n"
            "`import_conversation`'s only mutual exclusion is "
            "`conv_lock(target).locked()`. A tail parked on bgwork's "
            "semaphore holds no lock and has run no lines, so the guard "
            "admits the import and the tail writes afterwards.\n\n"
            f"{line}\n\n```json\n" + json.dumps(hits[:3], indent=2) + "\n```\n",
        )
    print("\n" + line)
    assert not hits, line


# ---------------------------------------------------------------------------
# 7. Admin compact while chat advances the same conversation
# ---------------------------------------------------------------------------

def _grow(client: httpx.Client, conv: str, exchanges: int) -> None:
    """Drive a conversation forward `exchanges` turns, one at a time, with the
    array growing the way a real client's does."""
    turns: list[str] = []
    for i in range(exchanges):
        turns = turns + [f"turn {i} of the growing conversation about topic{i}"]
        r = say_turns(client, conv, turns)
        assert r.status_code in (200, 503), r.text
        settle(client, timeout=TAIL_WAIT)


def test_compact_while_chat_advances_the_position(client):
    """POST /admin/.../compact concurrently with chat on the same conversation.

    Two things in admin_compact are not held against the chat path:

      1. It CLEARS the chat path's window anchor (`tail_fp`, `head_fp`,
         `window_turns`) under conv_lock and then releases it. A chat rollup
         arriving afterwards finds no anchor and falls into
         _observed_position's no-anchor branch, which either holds the
         position or advances it by _ASSUMED_NEW_TURNS — neither of which is
         measured against the array in hand.
      2. Its drain loop reads `last_summarized_turn` UNLOCKED on both sides
         of each maybe_rollup call, so a concurrent chat rollup between the
         two reads is indistinguishable from progress by its own call. The
         reported `rollup_calls` and `stopped_because` then describe an
         interleaving rather than the endpoint's own work.

    WHAT WOULD MAKE THIS FAIL: a 5xx, a position that exceeds what the
    conversation can support, or a summary whose chunk labels reach past the
    recorded position.
    """
    conv = new_conv("cmp")
    _grow(client, conv, 24)
    assert settle(client)

    before = inventory(client, conv)
    ca, cb = fresh_client(), fresh_client()
    try:
        res = barrier_parallel([
            (lambda: ca.post(
                f"/admin/conversations/{conv}/compact",
                json={"dry_run": False, "max_calls": 40},
            )),
            (lambda: say_turns(
                cb, conv,
                [f"turn {i} of the growing conversation about topic{i}"
                 for i in range(24)] + ["a turn arriving during the compaction"],
            )),
        ])
    finally:
        ca.close()
        cb.close()
    assert settle(client, timeout=TAIL_WAIT * 2)

    statuses = [getattr(v, "status_code", repr(v)[:120]) for ok, v in res]
    bodies = []
    for ok, v in res:
        if ok and getattr(v, "status_code", 0) == 200:
            try:
                bodies.append(v.json())
            except Exception:
                bodies.append("<not json>")
        else:
            bodies.append(getattr(v, "text", repr(v))[:400])

    after = inventory(client, conv)
    state = client.get(f"/admin/conversations/{conv}/summary").json()
    chunk_max = 0
    for tier in ("l1", "l2"):
        for ch in state.get(tier) or []:
            try:
                chunk_max = max(chunk_max, int(ch.get("last_turn") or 0))
            except (TypeError, ValueError):
                pass
    pos = int((after.get("summary") or {}).get("turns_seen") or 0)

    detail = (
        f"conv={conv}\n"
        f"statuses: {statuses}\n"
        f"before: {json.dumps(before.get('summary'), indent=2)}\n"
        f"after:  {json.dumps(after.get('summary'), indent=2)}\n"
        f"compact/chat bodies: {json.dumps(bodies, indent=2)[:1500]}\n"
        f"highest chunk last_turn: {chunk_max}; turns_seen: {pos}\n"
    )
    print("\n" + detail)
    record("race-compact", f"## compact concurrent with chat\n\n{detail}")

    assert all(
        (isinstance(s, int) and s < 500) for s in statuses
    ), f"5xx while compacting during a live chat:\n{detail}"
    assert chunk_max <= pos, (
        "a summary chunk is labelled past the conversation's recorded "
        f"position — the position and the labels disagree:\n{detail}"
    )


# ---------------------------------------------------------------------------
# 8. Streaming interleavings
# ---------------------------------------------------------------------------

def _abort_stream(conv: str, text: str, after_chunks: int) -> str:
    """Open a streaming chat and hang up after `after_chunks` SSE chunks.

    after_chunks=0 means: disconnect before reading a single byte of the
    body — the earliest point a client can vanish once the request is in.
    """
    c = fresh_client()
    try:
        with c.stream(
            "POST", "/v1/chat/completions",
            json=chat_body(["opening user turn", text], stream=True),
            headers={"X-Conversation-Id": conv},
        ) as r:
            if r.status_code != 200:
                return f"status={r.status_code}"
            n = 0
            for _ in r.iter_lines():
                n += 1
                if n >= after_chunks > 0:
                    break
        return f"aborted after {n} chunk(s)"
    except Exception as e:  # noqa: BLE001 — an abort can surface as one
        return f"{type(e).__name__}: {e}"
    finally:
        c.close()


def test_concurrent_streams_and_aborts_on_one_conversation(client):
    """Many streams on one conversation, aborted at every stage, while other
    streams start on the same conversation.

    An aborted stream still has a partially accumulated reply. main's
    accumulator decides whether that partial text enters memory
    (decide_memory_tail), and the disconnect handling runs while OTHER
    streams on the same conv_id are mid-flight and contending for the same
    conv_lock.

    WHAT WOULD MAKE THIS FAIL: a 5xx, a request that never returns, a store
    the compactor can no longer read afterwards, or duplicate turn indices
    left by the aborted turns.
    """
    conv = new_conv("stream")
    stages = [0, 1, 2, 3, 5, 8, 13, 21]
    results = []
    for _round in range(3):
        results += [
            v for ok, v in barrier_parallel([
                (lambda s=s, r=_round: _abort_stream(
                    conv, f"stream probe r{r} s{s} about widgets{s}", s
                ))
                for s in stages
            ])
        ]
    assert settle(client, timeout=TAIL_WAIT * 2)

    # The store must still be readable, and consistent.
    inv = inventory(client, conv)
    idx = turn_indices(client, conv)
    dupes = sorted({i for i in idx if idx.count(i) > 1})
    detail = (
        f"conv={conv}\n"
        f"abort outcomes: {results}\n"
        f"inventory: {json.dumps(inv, indent=2)}\n"
        f"turn indices: {idx}\nduplicates: {dupes}\n"
    )
    print("\n" + detail)
    record("race-streams", f"## concurrent streams with aborts\n\n{detail}")
    assert "status=5" not in " ".join(results), f"5xx from a stream:\n{detail}"
    assert not dupes, f"aborted streams left duplicate turn indices:\n{detail}"
    # A conversation that cannot be read back is the failure this guards.
    assert inv.get("facts", {}).get("count") is not None, (
        f"facts unreadable after concurrent stream aborts:\n{detail}"
    )


# ---------------------------------------------------------------------------
# 9. Duplicate and near-duplicate conversation identity
# ---------------------------------------------------------------------------

def test_byte_identical_requests_sent_twice_at_once(client):
    """The same request, byte for byte, twice, at the same time.

    _doc_id is content-addressed, so two identical exchanges are ONE row by
    design. What must not happen is the store disagreeing with itself: two
    rows sharing an index, a fact stored twice, or the position advancing by
    two exchanges for one.

    WHAT WOULD MAKE THIS FAIL: more than one episodic row, a duplicated
    fact, or turns_seen advanced as though two distinct exchanges occurred.
    """
    conv = new_conv("dup")
    ca, cb = fresh_client(), fresh_client()
    try:
        res = barrier_parallel([
            (lambda: say(ca, conv, "the identical duplicated request text")),
            (lambda: say(cb, conv, "the identical duplicated request text")),
        ])
    finally:
        ca.close()
        cb.close()
    assert all(ok and v.status_code in (200, 503) for ok, v in res), res
    assert settle(client)

    rows = episodic_rows(client, conv)
    idx = turn_indices(client, conv)
    facts_list = [f.get("text") for f in facts_of(client, conv)]
    pos = int((inventory(client, conv).get("summary") or {}).get("turns_seen") or 0)
    detail = (
        f"conv={conv}\nrows={len(rows)} indices={idx}\n"
        f"facts={facts_list}\nturns_seen={pos}\n"
    )
    print("\n" + detail)
    record("race-duplicate", f"## byte-identical concurrent requests\n\n{detail}")
    assert len(facts_list) == len(set(facts_list)), (
        f"a fact was stored twice by two identical concurrent requests:\n{detail}"
    )
    assert len(idx) == len(set(idx)), f"duplicate turn indices:\n{detail}"


def test_conv_ids_that_the_sanitiser_folds_together(client):
    """Clients whose conv_ids DIFFER but which `memory._sanitize` folds into
    one, writing at the same time.

    `_sanitize` strips everything outside `[A-Za-z0-9_-]` and then truncates
    to 64 characters. So two clients can hold ids they believe are distinct
    and land on one store without either being told. Three shapes here:

      * CASE: `Alpha` and `alpha` survive sanitisation unchanged, so they are
        genuinely different stores and must stay separate.
      * PUNCTUATION: `x.` and `x` both reduce to `x`.
      * LENGTH: two 70-character ids sharing their first 64 both reduce to
        that prefix — the shape a client generating long composite ids would
        actually produce.

    All ASCII, deliberately: httpx encodes header values as ASCII and RAISES
    on a non-ASCII conv_id, so a test using one measures the client rather
    than the compactor. (My first version did exactly that and reported a
    false positive.)

    WHAT WOULD MAKE THIS FAIL: an acknowledged fact missing from the store
    its id folds to — a lost update between two clients that do not know they
    have collided — or the case-distinct pair sharing a store.
    """
    stem = uuid.uuid4().hex[:10]
    upper = f"advraceCASE{stem}"
    lower = f"advracecase{stem}"
    plain = f"advracepunct{stem}"
    punct = f"advracepunct{stem}."          # folds to `plain`
    long_stem = "advracelong" + "z" * (64 - 11 - 10) + stem   # exactly 64 chars
    long_a = long_stem + "AAAAAA"           # both truncate to long_stem
    long_b = long_stem + "BBBBBB"

    markers = {
        upper: "upper case identity marker",
        lower: "lower case identity marker",
        plain: "plain punctuation marker",
        punct: "trailing dot marker",
        long_a: "long id A marker",
        long_b: "long id B marker",
    }
    cs = {cid: fresh_client() for cid in markers}
    try:
        res = barrier_parallel([
            (lambda cid=cid, m=m: (cid, say(cs[cid], cid, f"/remember {m}")))
            for cid, m in markers.items()
        ])
    finally:
        for c in cs.values():
            c.close()
    assert settle(client)

    # Every request must actually have been SENT and acknowledged, or the
    # assertions below are about a request that never happened.
    sent = {}
    for ok, v in res:
        assert ok, f"a /remember raised client-side rather than being sent: {v!r}"
        cid, r = v
        assert r.status_code == 200, f"{cid}: {r.status_code} {r.text[:200]}"
        sent[cid] = reply_text(r)
        assert "Remembered" in sent[cid], f"{cid}: {sent[cid]!r}"

    up, lo = fact_texts(client, upper), fact_texts(client, lower)
    pl = fact_texts(client, plain)
    lg = fact_texts(client, long_stem)
    detail = (
        f"acknowledgements: {json.dumps(sent, indent=2)}\n"
        f"upper={upper}: {sorted(up)}\n"
        f"lower={lower}: {sorted(lo)}\n"
        f"plain={plain} (also written as {punct!r}): {sorted(pl)}\n"
        f"long fold target={long_stem}: {sorted(lg)}\n"
    )
    print("\n" + detail)
    record("race-identity", f"## conv_id folding under concurrent writes\n\n{detail}")

    assert "upper case identity marker" in up, f"case-distinct id lost a fact:\n{detail}"
    assert "lower case identity marker" in lo, f"case-distinct id lost a fact:\n{detail}"
    assert not (up & lo), f"ids differing only in case shared a store:\n{detail}"
    # Both writers of each COLLIDING pair were told "Remembered". Both facts
    # must therefore be in the single store their ids fold to; losing one is a
    # lost update between clients that never knew they collided.
    for marker in ("plain punctuation marker", "trailing dot marker"):
        assert marker in pl, (
            f"a fact acknowledged under a colliding conv_id is not stored "
            f"({marker!r}):\n{detail}"
        )
    for marker in ("long id A marker", "long id B marker"):
        assert marker in lg, (
            f"a fact acknowledged under a length-truncated conv_id is not "
            f"stored ({marker!r}):\n{detail}"
        )


# ---------------------------------------------------------------------------
# 10. Deadlock hunting
# ---------------------------------------------------------------------------

def test_opposing_merges_do_not_deadlock(client):
    """merge A->B and merge B->A at the same time.

    commands._handle_retire is the only place that takes two conv_locks and
    it sorts the pair, so it cannot self-deadlock. merge_conversation takes
    none at all — it runs on a threadpool worker and probes `locked()`. This
    checks the pair for the failure a lock-ordering mistake produces: a
    request that never returns.

    WHAT WOULD MAKE THIS FAIL: either call not answering inside the timeout.
    """
    a, b = new_conv("da"), new_conv("db")
    _seed_facts(client, a, distinct_facts(10, "deadlock-a"))
    _seed_facts(client, b, [f"{t} (deadlock-b)" for t in _DISTINCT_FACTS[10:20]])

    ca, cb = (httpx.Client(base_url=BASE_URL, timeout=60.0) for _ in range(2))
    t0 = time.time()
    try:
        res = barrier_parallel([
            (lambda: ca.post(
                f"/admin/conversations/{a}/merge-into/{b}", json={"dry_run": False}
            )),
            (lambda: cb.post(
                f"/admin/conversations/{b}/merge-into/{a}", json={"dry_run": False}
            )),
        ])
    finally:
        ca.close()
        cb.close()
    elapsed = time.time() - t0
    detail = (
        f"a={a} b={b} elapsed={elapsed:.1f}s\n"
        + "\n".join(
            f"{'ok' if ok else 'RAISED'}: "
            f"{getattr(v, 'status_code', repr(v)[:200])} "
            f"{getattr(v, 'text', '')[:200]}"
            for ok, v in res
        )
    )
    print("\n" + detail)
    record("race-deadlock", f"## opposing merges\n\n{detail}")
    assert all(ok for ok, _ in res), f"a merge never returned:\n{detail}"
    assert elapsed < 55, f"opposing merges took {elapsed:.1f}s — possible lock stall:\n{detail}"


def test_forget_while_another_conversation_holds_the_pool(client):
    """/forget drains the pool PROCESS-WIDE, so a forget on conversation A
    waits on conversation B's tail.

    commands._settle_background_work says so explicitly and calls it bounded.
    This checks the bound holds rather than becoming an unbounded wait when
    the pool is full of other conversations' work.

    WHAT WOULD MAKE THIS FAIL: /forget not answering, or answering after a
    delay long enough to be a user-visible hang.
    """
    victim = new_conv("fwait")
    say(client, victim, "the thing to be forgotten")
    _backlog(client, tag="fw")
    t0 = time.time()
    r = say(client, victim, "/forget")
    elapsed = time.time() - t0
    detail = (
        f"conv={victim} status={r.status_code} elapsed={elapsed:.1f}s\n"
        f"reply={reply_text(r)!r}\n"
    )
    print("\n" + detail)
    record("race-forget-drain", f"## /forget waiting on a full pool\n\n{detail}")
    assert r.status_code == 200, detail
    assert settle(client)
    left = {k: v for k, v in _residue(client, victim).items() if v}
    assert not left, f"/forget left memory behind under pool pressure: {left}\n{detail}"


# ---------------------------------------------------------------------------
# 11. Command storm — every mutating chat command against one conversation
# ---------------------------------------------------------------------------

def test_command_storm_on_one_conversation(client):
    """Fire every mutating command at one conversation at once, with chat
    turns interleaved.

    /remember, /forget <substring>, /pin, /unpin and /tidy each hold conv_lock
    across their own load-modify-write, and the tail holds it across a
    different one. This is the combined interleaving nobody writes a test for:
    the commands are individually correct and the question is whether the SET
    of them is.

    WHAT WOULD MAKE THIS FAIL: a 5xx, a request that never returns, a fact
    acknowledged by /remember and not forgotten by anything that is
    nevertheless absent, or a store that cannot be read back afterwards.
    """
    conv = new_conv("storm")
    # Distinct, not templated. The first version of this case used
    # "storm keeper 0 about lighthouses" ... "storm keeper 5 ...", and the
    # concurrent /dedup clustered all six and replaced them with ONE merged
    # canonical — which, against a fixture with no model weights, is the
    # canned reply string. Every keeper "vanished", and none of it was a race.
    # See RACE-00.
    keep = [f"{t} (storm-keep)" for t in _DISTINCT_FACTS[:6]]
    # These carry a shared, distinctive word so `/forget scaffolding` has a
    # substring to match; they are otherwise unrelated to each other.
    doomed = [f"{t} — filed under scaffolding" for t in _DISTINCT_FACTS[20:24]]
    _seed_facts(client, conv, keep + doomed)

    ops = []
    cs = [fresh_client() for _ in range(14)]
    ops.append(lambda: say_hist(cs[0], conv, "/remember storm latecomer alpha"))
    ops.append(lambda: say_hist(cs[1], conv, "/remember storm latecomer beta"))
    ops.append(lambda: say_hist(cs[2], conv, "/forget scaffolding"))
    ops.append(lambda: say_hist(cs[3], conv, "/pin lighthouses"))
    ops.append(lambda: say_hist(cs[4], conv, "/unpin lighthouses"))
    ops.append(lambda: say_hist(cs[5], conv, "/tidy"))
    ops.append(lambda: say_hist(cs[6], conv, "a chat turn during the storm one"))
    ops.append(lambda: say_hist(cs[7], conv, "a chat turn during the storm two"))
    ops.append(lambda: cs[8].get(f"/admin/conversations/{conv}"))
    ops.append(lambda: cs[9].get(f"/admin/conversations/{conv}/facts"))
    ops.append(lambda: cs[10].get(f"/admin/conversations/{conv}/export"))
    ops.append(lambda: cs[11].post(f"/admin/conversations/{conv}/archive"))
    # restore_all explicitly: since v3.1.9 (hostile pass 4, F3) an empty body
    # is a 400, which would take the restore out of the race it is here for.
    ops.append(lambda: cs[12].post(f"/admin/conversations/{conv}/restore", json={"restore_all": True}))
    ops.append(lambda: cs[13].post(f"/admin/conversations/{conv}/dedup"))
    try:
        res = barrier_parallel(ops)
    finally:
        for c in cs:
            c.close()
    assert settle(client, timeout=TAIL_WAIT * 2)

    statuses = [getattr(v, "status_code", f"RAISED {v!r}"[:120]) for ok, v in res]
    texts = fact_texts(client, conv)
    archived = client.get(f"/admin/conversations/{conv}/archive").json().get("archived") or []
    archived_texts = {a.get("text") for a in archived}
    # A keeper may legitimately have been moved to the ARCHIVE by the
    # concurrent /tidy or archive pass — archived is recoverable and is not
    # loss. Gone from both is loss.
    lost = [k for k in keep if k not in texts and k not in archived_texts]
    detail = (
        f"conv={conv}\nstatuses: {statuses}\n"
        f"active facts: {sorted(texts)}\n"
        f"archived: {sorted(archived_texts)}\n"
        f"keepers lost from BOTH stores: {lost}\n"
    )
    print("\n" + detail)
    record("race-command-storm", f"## command storm on one conv\n\n{detail}")
    assert not any(
        isinstance(s, int) and s >= 500 for s in statuses
    ), f"5xx during the command storm:\n{detail}"
    assert all(ok for ok, _ in res), f"a request never returned:\n{detail}"
    assert not lost, f"a fact vanished from both active and archive:\n{detail}"


# ---------------------------------------------------------------------------
# 12. Backup runs on a THREAD while the store is being written
# ---------------------------------------------------------------------------

def test_backup_while_the_store_is_being_written(client):
    """POST /admin/backups during sustained write load.

    main.admin_run_backup calls `asyncio.to_thread(backup_module.run_once)`, so
    the archiver walks the storage tree on a WORKER THREAD while the event
    loop's tails are calling `memory.atomic_write_json` — which creates a
    `<name>.<rand>.tmp` sibling and then `os.replace`s it. An archiver that
    enumerates the directory and then opens what it found can be handed a name
    that no longer exists.

    WHAT WOULD MAKE THIS FAIL: the backup erroring, or
    `GET /admin/backups/verify` refusing the archive it just produced.
    """
    conv = new_conv("bkp")
    cb = fresh_client()
    try:
        # Load on one side, backup on the other, released together.
        def _load():
            out = []
            for w in range(6):
                out += _pile_wave(conv, f"bk{w}")
            return out

        res = barrier_parallel([
            _load,
            (lambda: cb.post("/admin/backups")),
        ])
    finally:
        cb.close()
    assert settle(client, timeout=TAIL_WAIT * 2)

    backup_res = res[1][1]
    ok = res[1][0]
    body = (
        backup_res.text[:800] if ok and hasattr(backup_res, "text") else repr(backup_res)[:400]
    )
    status = getattr(backup_res, "status_code", None)
    ver = client.get("/admin/backups/verify")
    detail = (
        f"conv={conv}\nbackup status={status}\nbackup body={body}\n"
        f"verify status={ver.status_code} body={ver.text[:600]}\n"
    )
    print("\n" + detail)
    record("race-backup", f"## backup during write load\n\n{detail}")
    assert ok, f"the backup request raised:\n{detail}"
    assert status is not None and status < 500, (
        f"backup 5xx'd while the store was being written:\n{detail}"
    )
    assert ver.status_code < 500, f"verify 5xx'd:\n{detail}"


# ---------------------------------------------------------------------------
# 13. Documented-sound behaviour, hammered
# ---------------------------------------------------------------------------

def test_dedup_holds_the_lock_against_remember(client):
    """`/admin/.../dedup` holds conv_lock across an LLM round trip. A
    /remember arriving during it must queue behind it and survive.

    This is a CHECKED-AND-SOUND case: it exists to prove the lock scope that
    IS correct is correct, so that the failures reported elsewhere in this
    file are attributable to the paths that lack it rather than to the
    attack.

    WHAT WOULD MAKE THIS FAIL: the acknowledged fact missing, or dedup's
    `after` list (computed before the /remember landed) being written over
    the top of it.
    """
    reps = max(6, REPS // 4)
    hits = []
    for i in range(reps):
        conv = new_conv(f"dd{i}")
        _remember_many(
            client, conv,
            distinct_facts(8, f"dedup-seed-{i}")
            # ONE deliberate near-duplicate, so dedup has real work and holds
            # the lock across an actual LLM call rather than returning at once.
            + [f"{_DISTINCT_FACTS[0]} (dedup-seed-{i}) restated"],
        )
        assert settle(client)
        secret = f"the dedup-race codeword for {i} is thimble{i}"
        ca, cb = fresh_client(), fresh_client()
        try:
            res = barrier_parallel([
                (lambda: ca.post(f"/admin/conversations/{conv}/dedup")),
                (lambda: say_hist(cb, conv, f"/remember {secret}")),
            ])
        finally:
            ca.close()
            cb.close()
        assert settle(client)
        acked = any(
            ok and hasattr(v, "text") and "Remembered" in reply_text(v)
            for ok, v in res
        )
        if acked and secret not in fact_texts(client, conv):
            hits.append({
                "conv": conv,
                "statuses": [getattr(v, "status_code", repr(v)[:80]) for _, v in res],
                "facts": sorted(fact_texts(client, conv)),
            })

    line = rate_line("dedup overwrote an acknowledged /remember", len(hits), reps)
    if hits:
        record(
            "race-dedup",
            f"## dedup vs /remember\n\n{line}\n\n```json\n"
            + json.dumps(hits[:3], indent=2) + "\n```\n",
        )
    print("\n" + line)
    assert not hits, line


# ---------------------------------------------------------------------------
# 14. The same unsynchronised write, reached through its other doors
# ---------------------------------------------------------------------------

def test_merge_and_import_into_one_destination(client):
    """`merge-into` and `import` landing on the same conversation at once.

    Neither takes `conv_lock`; each only PROBES it. `import_conversation` runs
    to completion on the event loop and so cannot interleave with itself —
    that is the property its own comment relies on — but
    `merge_conversation` runs on a threadpool worker, so its
    `load_facts -> save_facts` can straddle the import entirely. Each probe
    looks for a lock that neither of them takes.

    This is RACE-02's mechanism reached through a different pair of endpoints,
    included because a fix that only serialises merge against merge would
    leave it standing.

    WHAT WOULD MAKE THIS FAIL: the import's own bundle absent after the import
    answered 200 with its counts.
    """
    reps = max(10, REPS // 2)
    n = 15
    hits = []
    for i in range(reps):
        dst = new_conv(f"mixdst{i}")
        src = new_conv(f"mixsrc{i}")
        merge_texts = [f"{t} (merge-side-{i})" for t in _DISTINCT_FACTS[:n]]
        import_texts = [f"{t} (import-side-{i})" for t in _DISTINCT_FACTS[n:2 * n]]
        _seed_facts(client, src, merge_texts)
        _seed_facts(client, dst, ["the destination seed fact"])
        now = int(time.time())
        bundle = {
            "version": "v2.1", "exported_at": now, "source_conv_id": src,
            "facts": [
                {"text": t, "added_turn": 1, "last_used": now, "pin": False}
                for t in import_texts
            ],
            "summary_state": {}, "episodic": [],
        }
        ca, cb = fresh_client(), fresh_client()
        try:
            res = barrier_parallel([
                (lambda: ca.post(
                    f"/admin/conversations/{src}/merge-into/{dst}",
                    json={"dry_run": False},
                )),
                (lambda: cb.post(
                    "/admin/conversations/import",
                    json={"bundle": bundle, "target_conv_id": dst,
                          "overwrite": True},
                )),
            ])
        finally:
            ca.close()
            cb.close()
        assert settle(client)

        m, imp = res[0][1], res[1][1]
        got = fact_texts(client, dst)
        # An import with overwrite=true REPLACES the fact list wholesale, so
        # merge facts absent after a successful import is that endpoint's
        # documented behaviour, not a race. Only the reverse direction — the
        # import's own bundle missing after it answered 200 — is unambiguous.
        missing = []
        if getattr(imp, "status_code", 0) == 200:
            missing = [t for t in import_texts if t not in got]
        if missing:
            hits.append({
                "dst": dst,
                "merge_status": getattr(m, "status_code", None),
                "import_status": getattr(imp, "status_code", None),
                "import_said": (
                    imp.json() if getattr(imp, "status_code", 0) == 200
                    else getattr(imp, "text", "")[:200]
                ),
                "missing_from_bundle": len(missing), "of": n,
                "stored": len(got),
            })

    line = rate_line("merge clobbered an acknowledged import", len(hits), reps)
    if hits:
        record(
            "race-merge",
            "## merge-into vs import on one destination\n\n"
            "Neither endpoint takes conv_lock; each only probes it. The merge "
            "runs on a threadpool worker so its load-modify-write straddles "
            "the import, which runs atomically on the event loop.\n\n"
            f"{line}\n\n```json\n" + json.dumps(hits[:3], indent=2) + "\n```\n",
        )
    print("\n" + line)
    assert not hits, line


def test_merge_against_a_remember_holding_the_lock(client):
    """`merge-into` racing a `/remember` on the destination.

    `/remember` DOES hold conv_lock across its load-modify-write — that fix is
    the one the file's own comment describes shipping after a user watched a
    remembered fact disappear. The merge only PROBES that lock, from a thread,
    so the probe is a snapshot of one instant and the merge's own
    read-modify-write is covered by nothing. If the probe lands while the
    command holds the lock the merge refuses with a 400 (the guard working);
    a moment either side and both write.

    WHAT WOULD MAKE THIS FAIL: the acknowledged /remember fact missing after
    a merge that answered 200 — the same user-visible symptom as the original
    incident, arriving through an operator action instead of a tail.
    """
    reps = REPS
    n = 15
    hits = []
    refused = 0
    for i in range(reps):
        dst = new_conv(f"mrdst{i}")
        src = new_conv(f"mrsrc{i}")
        _seed_facts(client, src, [f"{t} (merge-{i})" for t in _DISTINCT_FACTS[:n]])
        _seed_facts(client, dst, ["the destination seed fact"])
        secret = f"the remember-vs-merge codeword for run {i} is saffron{i}"
        ca, cb = fresh_client(), fresh_client()
        try:
            res = barrier_parallel([
                (lambda: ca.post(
                    f"/admin/conversations/{src}/merge-into/{dst}",
                    json={"dry_run": False},
                )),
                (lambda: say(cb, dst, f"/remember {secret}")),
            ])
        finally:
            ca.close()
            cb.close()
        assert settle(client)
        m, rem = res[0][1], res[1][1]
        if getattr(m, "status_code", 0) in (400, 409):
            refused += 1
            continue
        acked = "Remembered" in reply_text(rem)
        if acked and secret not in fact_texts(client, dst):
            hits.append({
                "dst": dst,
                "merge_status": getattr(m, "status_code", None),
                "remember_said": reply_text(rem),
                "facts_now": sorted(fact_texts(client, dst))[:5],
            })

    line = (
        rate_line("merge overwrote an acknowledged /remember", len(hits), reps)
        + f" ({refused} merge(s) refused with 400/409 — the probe working)"
    )
    if hits:
        record(
            "race-merge",
            f"## merge-into vs /remember\n\n{line}\n\n```json\n"
            + json.dumps(hits[:3], indent=2) + "\n```\n",
        )
    print("\n" + line)
    assert not hits, line


# ---------------------------------------------------------------------------
# 15. The extractor's duplicate-suppression list is a PRE-REQUEST snapshot
# ---------------------------------------------------------------------------

def _dup_counts(texts: list[str]) -> dict:
    seen: dict[str, int] = {}
    for t in texts:
        seen[t] = seen.get(t, 0) + 1
    return {t: n for t, n in seen.items() if n > 1}


def test_concurrent_turns_re_extract_facts_the_others_already_stored(client):
    """A DIFFERENTIAL test: the same N turns, run serially and run at once.

    `_facts_tail` hands the extractor its duplicate-suppression list from
    `touched_facts` — the snapshot the REQUEST PATH loaded, long before the
    lock was taken — and not from the `facts.load_facts()` it performs one
    line later under that lock:

        extraction_facts = facts.select_for_injection(
            touched_facts, max_tokens=facts._MAX_FACTS_TOKENS)
        ...
        combined = _merge_touched(facts.load_facts(conv_id), touched_facts) \\
                   + new_entries

    `facts._EXTRACTION_SYSTEM_PROMPT` says "Do NOT restate facts already in the
    EXISTING FACTS list below", and that list is the only duplicate
    suppression the extraction call has. Serially it is accurate: turn N's
    snapshot contains what turn N-1 stored. Concurrently every in-flight turn
    holds the SAME pre-burst snapshot, so each one is told the store is
    emptier than it is and each re-extracts what the others are storing.
    `new_entries` is then appended to `combined` with no membership check, so
    the duplicates land as separate rows.

    Evidence this is not hypothetical — a conversation from an earlier run of
    this suite, after piled concurrent turns:

        "facts": [
          {"text": "Acknowledged. (fixture reply ...)", "last_used": 1788553953},
          {"text": "Acknowledged. (fixture reply ...)", "last_used": 1788553900},
          {"text": "Acknowledged. (fixture reply ...)", "last_used": ...}, ...
        ]

    byte-identical rows, in one store.

    THE DIFFERENTIAL IS THE POINT. Duplicate rows on their own could be a
    property of the extractor or of the fixture's constant reply. Running the
    identical workload serially — where the same reply produces the same
    candidate fact and the suppression list IS current — is the control. If
    the serial run stays clean and the concurrent one duplicates, the
    duplication is the interleaving.

    WHAT WOULD MAKE THIS FAIL: the concurrent run holding more duplicate fact
    rows than the serial one.
    """
    n = 8
    seq_conv = new_conv("dupseq")
    for i in range(n):
        r = say_hist(client, seq_conv, f"serial probe {i} regarding item{i}")
        assert r.status_code in (200, 503), r.text
        assert settle(client), "pool never drained between serial turns"
    seq_texts = [f.get("text", "") for f in facts_of(client, seq_conv)]

    par_conv = new_conv("duppar")
    cs = [fresh_client() for _ in range(n)]
    try:
        barrier_parallel([
            (lambda i=i: say_hist(cs[i], par_conv, f"serial probe {i} regarding item{i}"))
            for i in range(n)
        ])
    finally:
        for c in cs:
            c.close()
    assert settle(client, timeout=TAIL_WAIT * 2)
    par_texts = [f.get("text", "") for f in facts_of(client, par_conv)]

    seq_dupes = _dup_counts(seq_texts)
    par_dupes = _dup_counts(par_texts)
    seq_extra = sum(v - 1 for v in seq_dupes.values())
    par_extra = sum(v - 1 for v in par_dupes.values())
    detail = (
        f"same {n} turns, serial vs concurrent\n"
        f"serial   conv={seq_conv} rows={len(seq_texts)} "
        f"duplicate-rows={seq_extra} {json.dumps(seq_dupes)[:400]}\n"
        f"parallel conv={par_conv} rows={len(par_texts)} "
        f"duplicate-rows={par_extra} {json.dumps(par_dupes)[:400]}\n"
    )
    print("\n" + detail)
    record(
        "race-duplicate-extraction",
        "## concurrent turns re-extract each other's facts\n\n" + detail,
    )
    assert par_extra <= seq_extra, (
        "concurrent turns produced duplicate fact rows the same workload run "
        "serially did not — the extractor's suppression list is the "
        "pre-request snapshot, not the list read under the lock:\n" + detail
    )


# ---------------------------------------------------------------------------
# 16. Two threads writing ONE vector store
# ---------------------------------------------------------------------------

def _build_exchanges(client: httpx.Client, conv: str, n: int) -> int:
    """Drive `n` real exchanges onto `conv` so it has EPISODIC rows to merge.

    Facts can be planted in one request; episodic rows cannot — they are
    written by the tail, from an actual exchange.
    """
    for i in range(n):
        r = say_hist(client, conv, f"episodic seed {i} about subject{i}")
        assert r.status_code in (200, 503), r.text
        assert settle(client)
    return len(episodic_rows(client, conv))


def test_merge_writes_chroma_from_a_thread_while_chat_indexes(client):
    """`merge-into` importing episodic rows on a WORKER THREAD while the
    memory tail indexes exchanges on the EVENT LOOP.

    `retrieval.index_exchange` is called under conv_lock from `_async_tail`,
    which runs on the event loop. `retrieval.import_indexed_exchange` is
    called from `portability.merge_conversation`, which runs on a threadpool
    worker and holds no lock. Both write the SAME ChromaDB collection — the
    per-conversation lock partitions the JSON files, but the vector store is
    one shared object, so it is not a partition at all for this pair.

    `admin_compact` adds a third participant: it reads the same collection
    through `run_in_threadpool(retrieval.export_indexed_exchanges, ...)`.

    WHAT WOULD MAKE THIS FAIL: a 5xx from any of the three, a store that can
    no longer be read back (`/export` or `/admin/conversations/<id>` failing,
    or reporting `indexed_exchanges: null`, which is what this codebase
    returns when a layer is unreadable rather than empty), or duplicate turn
    indices left behind by the two writers.
    """
    src = new_conv("chsrc")
    dst = new_conv("chdst")
    n_src = _build_exchanges(client, src, 6)
    assert n_src > 0, "the source conversation has no episodic rows to merge"
    _build_exchanges(client, dst, 2)

    # Load on the loop, merge on a thread, release together.
    ca, cb = fresh_client(), fresh_client()
    try:
        def _load():
            out = []
            for w in range(4):
                out += _pile_wave(dst, f"ch{w}")
            return out

        res = barrier_parallel([
            _load,
            (lambda: ca.post(
                f"/admin/conversations/{src}/merge-into/{dst}",
                json={"dry_run": False},
            )),
            (lambda: cb.get(f"/admin/conversations/{dst}/export")),
        ])
    finally:
        ca.close()
        cb.close()
    assert settle(client, timeout=TAIL_WAIT * 3)

    merge_r, export_r = res[1][1], res[2][1]
    statuses = {
        "merge": getattr(merge_r, "status_code", repr(merge_r)[:120]),
        "concurrent_export": getattr(export_r, "status_code", repr(export_r)[:120]),
    }
    # The store must still be READABLE, and must distinguish empty from
    # unreadable — this codebase reports null for the latter, deliberately.
    inv = inventory(client, dst)
    idx = turn_indices(client, dst)
    dupes = sorted({i for i in idx if idx.count(i) > 1})
    detail = (
        f"src={src} ({n_src} episodic rows) dst={dst}\n"
        f"statuses: {json.dumps(statuses)}\n"
        f"merge body: {getattr(merge_r, 'text', '')[:500]}\n"
        f"dst inventory after: {json.dumps(inv, indent=2)}\n"
        f"dst turn indices: {idx}\nduplicates: {dupes}\n"
    )
    print("\n" + detail)
    record("race-chroma", f"## merge + indexing + export on one collection\n\n{detail}")

    for name, s in statuses.items():
        assert isinstance(s, int) and s < 500, f"{name} 5xx'd:\n{detail}"
    assert (inv.get("episodic") or {}).get("indexed_exchanges") is not None, (
        "the episodic layer is UNREADABLE after concurrent writes from a "
        "worker thread and the event loop (null means unreadable here, not "
        f"empty):\n{detail}"
    )
    assert inv.get("facts", {}).get("count") is not None, (
        f"the facts layer is unreadable after concurrent writes:\n{detail}"
    )
    assert not dupes, (
        f"two writers on one collection left duplicate turn indices:\n{detail}"
    )
