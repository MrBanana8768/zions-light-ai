"""Tests for scripts/wordban/ - the tool that builds, verifies and releases the word ban (the xgrammar grammar
pasted into the model's `structured_outputs` custom parameter).

What is pinned here:
  * the generator is deterministic (two builds in separate processes, different hash seeds: identical bytes);
  * the verifier FAILS on a trapped grammar: revision 4's angel trap (an accented letter after "angel" walked
    into a state where only combining marks were allowed and the reply could not end) is rebuilt from a
    synthetic banlist with the old 'taint' behaviour, and both the DFA check and `verify --no-image` catch it;
    with today's 'dead' behaviour the same banlist verifies clean;
  * the real banlist has no trap, and the three prefixes that trapped revision 4 on 2026-09-23 (angelĂ,
    Angelø, angeléñÖs) are refused at the accented letter with a normal continuation left;
  * a word added to the YAML is tested automatically (derived must-block cases) and blocked;
  * the e2e probe sentences are built from the banlist's own test lists, not hardcoded, and agree with the
    tool's own automaton;
  * the corpus loader walks each chat's branch table-first and prefers the `output` text;
  * `release` refuses without a full, passing verify of the exact artifact;
  * the pod check script's small grammar interpreter judges text exactly like the tool's own automaton.
Everything is synthetic; no model, no tokenizer, no network, no docker (the token-level half of `verify`
needs the image and is run on the host, see scripts/wordban/README.md).

    python test_wordban_tool.py
"""
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import types
from pathlib import Path

_HERE = Path(__file__).resolve().parent
WB = _HERE.parent / "scripts" / "wordban"
sys.path.insert(0, str(WB))
import wb_core as core  # noqa: E402
import wordban as W  # noqa: E402
import wb_cmds  # noqa: E402

_FAILS = []


def assert_eq(actual, expected, label):
    if actual != expected:
        print(f"FAIL {label}: expected {expected!r}, got {actual!r}")
        _FAILS.append(label)
        return
    print(f"  ok   {label}")


def assert_true(cond, label, detail=""):
    if not cond:
        print(f"FAIL {label}" + (f" ({detail})" if detail else ""))
        _FAILS.append(label)
        return
    print(f"  ok   {label}")


def run_tool(args, env=None):
    e = dict(os.environ)
    e.update(env or {})
    e["WORDBAN_HOME"] = e.get("WORDBAN_HOME") or tempfile.mkdtemp(prefix="wb-home-")
    p = subprocess.run([sys.executable, str(WB / "wordban.py")] + args, capture_output=True, text=True, env=e)
    return p.returncode, p.stdout + p.stderr


# A small synthetic banlist: the angel family as revision 4 had it (Angelo still an exception), plus a separator
# rule, so the builds stay fast. `mode` switches the accented-continuation behaviour.
def mini_banlist(mode="dead", mark_cap=2, extra_rules=""):
    return f"""
revision: 99
model: test/model
unicode_version: "0.0.0"
rollback: {{99: "test fixture - not a real rollback note"}}
folding:
  latin_blocks: [[0x00C0, 0x024F], [0x1E00, 0x1EFF], [0xFF21, 0xFF5A]]
  greek_blocks: [[0x0370, 0x03FF], [0x1F00, 0x1FFF]]
  extra: {{"ø": o, "Ø": O}}
  accented_continuation: {mode}
  fffd: one_run_per_word
  mark_cap: {mark_cap}
  mark_cap_hebrew: null
separators:
  ALL: [" ", "-", ".", "_", "*"]
families:
  angel:
    rules:
      - {{id: ANGEL, stem: angel, alone: ban, other: ban, allow_words: [a, o, eno, enos],
         context: {{after: "los ", allow_words: [a, o, eno, enos, es]}}}}
      - {{id: A-N-G-E-L, match: angel, spaced: ALL}}
{extra_rules}
tests:
  must_block: [[angel, Angels, "angelĂ", "Angelø", "angeléñÖs", "a-n-g-e-l"]]
  must_pass: [[Angela, Angelo, Los Angeles, evangelical, "the angle"]]
  trap_regressions: ["the angelĂ", "and Angelø", "our angeléñÖs"]
"""


def write(tmp, name, text):
    p = Path(tmp) / name
    p.write_text(text, encoding="utf-8")
    return p


# ===========================================================================
def test_generator_is_deterministic():
    tmp = Path(tempfile.mkdtemp(prefix="wb-det-"))
    try:
        outs = []
        for seed in ("1", "2"):
            d = tmp / f"b{seed}"
            rc, out = run_tool(["build", "--here", "--banlist", str(WB / "banlist.yaml"), "--build-dir", str(d)],
                               env={"PYTHONHASHSEED": seed})
            assert_eq(rc, 0, f"build of the real banlist succeeds (hash seed {seed})")
            outs.append(d)
        g1, g2 = (outs[0] / "wb_ban.ebnf").read_bytes(), (outs[1] / "wb_ban.ebnf").read_bytes()
        assert_true(len(g1) > 100_000 and g1 == g2, "two builds in separate processes are byte-identical",
                    f"{len(g1)} vs {len(g2)} bytes")
        v = json.loads((outs[0] / "structured_outputs.value.txt").read_text(encoding="utf-8"))
        assert_eq(v["grammar"].encode(), g1, "the paste value is exactly {\"grammar\": <the grammar>}")
        man = json.loads((outs[0] / "manifest.json").read_text(encoding="utf-8"))
        assert_eq(man["dfa_trap_states"], 0, "the real banlist builds with no trap state")
        return outs[0]
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def test_real_banlist_has_no_trap_and_the_live_trap_prefixes_are_refused(build_dir):
    ebnf = (build_dir / "wb_ban.ebnf").read_text(encoding="utf-8")
    A = core.Automaton(ebnf)
    traps, _prev, _ = A.trap_states()
    assert_eq(traps, [], "no reachable state without an ordinary way to finish (character level)")
    for prefix in ("a small angelĂ", "RU and Angelø", "our angeléñÖs"):
        s, at = A.run(prefix)
        assert_true(s is None, f"{prefix!r} is refused")
        assert_true(at is not None and at <= prefix.index("gel") + 3, f"{prefix!r} is refused by the accented letter at the latest")
        s0, _ = A.run(prefix[:at])
        assert_true(s0 is not None and s0 not in traps, f"just before the refusal in {prefix!r} a normal continuation remains")
    # revision 6: the stem itself is refused (no name exceptions), glued and split forms too; real words pass
    for t in ("angel", "theangel", "an-gel", "g\u200bold", "g\u043eld", "angeI", "candleflame", "flamenco", "los ángeles"):
        assert_true(not A.accepts(t), f"{t!r} is refused (rev 6)")
    for t in ("evangelical", "strangely", "marigold", "inflammation", "flammable", "Los Angeles",
              "\U0001f468\u200d\U0001f469\u200d\U0001f467", "e.g. old", "gol de Messi"):
        assert_true(A.accepts(t), f"{t!r} passes (rev 6)")
    s0, _ = A.run("Los Angel")
    assert_true(s0 is not None and not A.accepts("Los Angel\u0301es"), "no combining mark right after a banned stem")
    rc, out = run_tool(["verify", "--no-image", "--fuzz", "2000", "--build-dir", str(build_dir)])
    assert_eq(rc, 0, "verify --no-image passes on the real banlist")
    assert_true("VERIFY PASSED" in out and "reference definition agrees" in out,
                "the local suite ran: artifact automaton, lists, reference cross-check, fuzz", out[-800:])


def test_yellow_light_glued_carve_outs_are_gone(build_dir):
    # rev 7 FINAL (owner decision D1, 2026-09-25): "ban EVERY word starting with 'yellow light': NO carve-outs" -
    # this REPLACES decision B. Documenting what the grammar does NOW, for every form (word-start spaced/glued,
    # hyphenated, and the glued-INSIDE-another-word form): lighter(s)/lighthouse(s)/lightning are BANNED, same as
    # lighting/lights/light's/every other continuation - there is no exception left anywhere, and no funnel (the
    # phrase has no live continuation at all once "yellow"+JOIN+"light" is spelled).
    ebnf = (build_dir / "wb_ban.ebnf").read_text(encoding="utf-8")
    A = core.Automaton(ebnf)
    for t in ("yellow lighter", "yellow lighters", "yellow lighthouse", "yellow lighthouses", "yellow lightning",
              "yellow-lighter", "yellow-lighthouse", "yellowlighter", "yellowlighthouse", "yellowlightning",
              "theyellowlighter", "theyellowlighters", "theyellowlighthouse", "theyellowlighthouses",
              "theyellowlightning", "THEYELLOWLIGHTER", "TheYellowLighthouse", "guardianyellowlighter",
              "theyellowlight", "THEYELLOWLIGHT", "theyellowlighting", "theyellowlights", "theyellowlight's",
              "theyellowlightly", "guardianyellowlight"):
        assert_true(not A.accepts(t), f"{t!r} is BANNED (D1: no carve-out anywhere, including the glued-inside form)")


def test_verifier_fails_on_revision_4s_angel_trap():
    tmp = Path(tempfile.mkdtemp(prefix="wb-trap-"))
    try:
        # the revision-4 behaviour, and revision 4's shape (marks a transparent self-loop, no cap)
        bl = write(tmp, "trap.yaml", mini_banlist(mode="taint", mark_cap="null"))
        spec, ebnf = core.build(bl)
        assert_true(len(spec.dfa_trap_states()) > 0, "the DFA check finds the trap in the rebuilt revision-4 behaviour")
        A = core.Automaton(ebnf)
        traps, prev, _ = A.trap_states()
        s, at = A.run("the angelĂ")
        assert_true(at is None and s in traps, "'the angelĂ' leads into a trap state (as live on 2026-09-23)")
        examples = [A.example_prefix(t, prev).lower() for t in traps]
        assert_true(any(e.startswith("angel") for e in examples), "a trap example starts with the angel stem", examples)
        d = tmp / "build"
        rc, out = run_tool(["build", "--here", "--banlist", str(bl), "--build-dir", str(d)])
        assert_eq(rc, 0, "a trapped grammar in revision 4's shape still builds (so the verifier can be shown it)")
        assert_true("trap" in out.lower(), "build warns about the trap", out[-400:])
        rc, out = run_tool(["verify", "--no-image", "--fuzz", "500", "--build-dir", str(d)])
        assert_eq(rc, 1, "verify exits non-zero on the trapped grammar")
        assert_true("[FAIL] no-trap property" in out, "the no-trap check is the one that fails", out[-1200:])
        assert_true("[FAIL] trap regressions" in out, "the trap-regression prefixes fail too", out[-1200:])
        # with the mark cap on, build refuses outright: a state with no way on at all cannot be emitted
        bl2 = write(tmp, "trap2.yaml", mini_banlist(mode="taint", mark_cap=2))
        rc, out = run_tool(["build", "--here", "--banlist", str(bl2), "--build-dir", str(tmp / "b2")])
        assert_true(rc != 0 and "REFUSED" in out, "build refuses to emit a trap under the mark cap", out[-400:])
        # the same banlist with today's behaviour is clean
        bl3 = write(tmp, "dead.yaml", mini_banlist(mode="dead"))
        d3 = tmp / "b3"
        rc, out = run_tool(["build", "--here", "--banlist", str(bl3), "--build-dir", str(d3)])
        assert_eq(rc, 0, "the 'dead' behaviour builds")
        rc, out = run_tool(["verify", "--no-image", "--fuzz", "2000", "--build-dir", str(d3)])
        assert_eq(rc, 0, "and verifies clean: an accented continuation is simply not allowed")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_a_word_added_to_the_yaml_is_tested_and_blocked():
    tmp = Path(tempfile.mkdtemp(prefix="wb-add-"))
    try:
        bl = write(tmp, "add.yaml", mini_banlist(extra_rules="""  zorbex:
    rules:
      - {id: ZORBEX, match: zorbex}"""))
        spec, ebnf = core.build(bl)
        derived = core.derived_must_block(spec)
        forms = [f for rid, f in derived if rid == "ZORBEX"]
        assert_eq(forms, ["zorbex", "Zorbex", "ZORBEX"], "the new rule appears in the must-block cases by itself")
        A = core.Automaton(ebnf)
        assert_true(all(not A.accepts(f) for f in forms + ["a zorbexian", "zörbex"]), "and the grammar blocks it")
        assert_true(A.accepts("zorb") and A.accepts("bzorbex"), "a shorter word and a non-word-start pass")
        d = tmp / "b"
        run_tool(["build", "--here", "--banlist", str(bl), "--build-dir", str(d)])
        rc, out = run_tool(["verify", "--no-image", "--fuzz", "500", "--build-dir", str(d)])
        n_listed = 6
        assert_true(rc == 0 and f"{n_listed + len(derived)}/{n_listed + len(derived)} blocked" in out,
                    "verify counts the derived cases in must-block", out[-900:])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_corpus_loader_walks_the_branch_table_first_and_prefers_output():
    tmp = Path(tempfile.mkdtemp(prefix="wb-corpus-"))
    try:
        db = tmp / "webui.db"
        c = sqlite3.connect(db)
        c.execute("create table chat (id text, chat json, current_message_id text)")
        c.execute("create table chat_message (id text, chat_id text, role text, parent_id text, content json, "
                  "output json, created_at bigint)")
        hist = {"history": {"currentId": "m3", "messages": {
            "m1": {"id": "m1", "parentId": None, "role": "user", "content": "hello one", "timestamp": 1},
            "m2": {"id": "m2", "parentId": "m1", "role": "assistant", "content": "json copy of two", "timestamp": 2},
            "x9": {"id": "x9", "parentId": "m1", "role": "assistant", "content": "an old regeneration", "timestamp": 2},
        }}}
        c.execute("insert into chat values ('c1', ?, 'm4')", (json.dumps(hist),))
        rows = [("c1-m2", "assistant", "m1", json.dumps("table copy of two"),
                 json.dumps([{"type": "message", "content": [{"type": "output_text", "text": "OUTPUT of two"}]}]), 2),
                ("c1-m3", "user", "m2", json.dumps("three, only in the table"), None, 3),
                ("c1-m4", "assistant", "m3", json.dumps("four"), json.dumps(None), 4)]
        for mid, role, parent, content, output, ts in rows:
            c.execute("insert into chat_message values (?, 'c1', ?, ?, ?, ?, ?)", (mid, role, parent, content, output, ts))
        c.commit()
        c.close()
        got = [m["t"] for m in core.load_corpus(db)]
        assert_eq(got, ["hello one", "OUTPUT of two", "three, only in the table", "four"],
                  "branch from chat.current_message_id, table rows first, output text preferred, off-branch left out")
        got_all = [m["t"] for m in core.load_corpus(db, scope="all")]
        assert_true("an old regeneration" in got_all, "scope='all' adds the other stored messages")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_corpus_sample_rate_reaches_the_floor_in_recent_mode():
    # rev 7 round 2 (coordinator decision): do NOT lower the decode-step sample floor (100,000) - raise the
    # SAMPLING RATE in 'recent' mode instead, so the sample count still reaches it: rate =
    # min(1.0, (floor * 1.2) / total_tokens), capped at 1.0. 'full' mode keeps the given rate unchanged.
    assert_eq(core.CORPUS_SAMPLE_FLOOR, 100000, "the floor itself is unchanged")
    assert_eq(core.corpus_sample_rate("full", 5_000_000, 0.03), 0.03,
              "full mode keeps the given --corpus-sample rate, never recomputed")
    got = core.corpus_sample_rate("recent", 240_479, 0.03)
    assert_true(abs(got - min(1.0, 120_000 / 240_479)) < 1e-9,
                "recent mode raises the rate to floor*1.2/total_tokens", got)
    assert_true(got * 240_479 >= core.CORPUS_SAMPLE_FLOOR * 0.99,
                "the raised rate, applied to the actual token count, reaches the floor (allowing for rounding)")
    assert_eq(core.corpus_sample_rate("recent", 50, 0.03), 1.0,
              "a tiny corpus: the rate is capped at 1.0 (every prefix sampled), never above")
    assert_eq(core.corpus_sample_rate("recent", 0, 0.03), 0.03,
              "zero tokens: falls back to the given rate rather than dividing by zero")


def test_compile_gate_median_logic_with_fake_timings():
    # rev 8 fix round (hostile-review fix 3): the first-compile gate must not fail on ONE slow outlier - an
    # unchanged grammar measured 144-174s across 9 real compiles against the 180s limit. These are FAKE timing
    # numbers only (no real compilation happens here) exercising the pure decision functions in wb_core.py.
    LIMIT = 180

    # a fast first sample: done after 1 sample, no second/third needed, and it passes.
    assert_eq(core.compile_gate_need_second_sample(150, LIMIT), False,
              "a first compile under the limit needs no second sample")
    assert_true(core.compile_gate_verdict([150], LIMIT), "a single fast sample passes on its own")

    # a slow first sample needs a second.
    assert_eq(core.compile_gate_need_second_sample(190, LIMIT), True,
              "a first compile over the limit needs a second sample")

    # two samples that AGREE (both over the limit): no third needed, and the median (their average) still fails.
    assert_eq(core.compile_gate_need_third_sample(190, 185, LIMIT), False,
              "two samples on the same side of the limit are not a tie")
    assert_true(not core.compile_gate_verdict([190, 185], LIMIT),
                "two samples both over the limit: the gate fails (this is the real regression case)")

    # two samples that DISAGREE (one over, one under the limit): that is the tie a third sample must break.
    assert_eq(core.compile_gate_need_third_sample(190, 170, LIMIT), True,
              "one sample over and one under the limit is exactly the tie a third sample breaks")

    # the 9-real-compile scenario this fix is modeled on: a slow first sample (174s, over 180? no - use a
    # genuinely over-limit outlier at 190s) followed by two fast, in-range samples - the median of all three
    # must be comfortably under the limit, so the gate passes despite the one slow outlier.
    samples = [190, 150, 160]
    assert_true(core.compile_gate_need_second_sample(samples[0], LIMIT), "outlier alone would need a 2nd sample")
    assert_true(core.compile_gate_need_third_sample(samples[0], samples[1], LIMIT),
                "190 (over) vs 150 (under) disagree: a third sample is needed to break the tie")
    assert_true(core.compile_gate_verdict(samples, LIMIT),
                "median([190,150,160]) = 160, under the 180s limit: PASS despite the 190s outlier")

    # the mirror-image regression case: two genuinely slow samples plus one fast one must still fail, so the
    # gate cannot be gamed by a single lucky fast compile either.
    slow_samples = [190, 195, 150]
    assert_true(not core.compile_gate_verdict(slow_samples, LIMIT),
                "median([190,195,150]) = 190, over the 180s limit: FAIL - a real regression is still caught")

    # a boundary check: a median exactly AT the limit passes ("exceeds" the limit is a strict greater-than).
    assert_true(core.compile_gate_verdict([180], LIMIT), "a median exactly at the limit does not 'exceed' it")
    assert_true(not core.compile_gate_verdict([180.01], LIMIT), "a median a hair over the limit fails")

    # rev 8 fix round 2 (Opus re-review): an extra sample under 10% of the first sample is IMPLAUSIBLE for a
    # real cold compile of the same grammar - almost certainly a cache hit, not genuine work. This must FAIL
    # the gate outright, never silently get dropped so the remaining (fast) samples pass on a falsely light
    # median. [190, 0.002, 0.002] is exactly the bug this closes: a real 190s outlier whose "extra samples"
    # were actually served from xgrammar's own grammar cache (~0.002s, as measured on the real backend) rather
    # than a genuine cold recompile - the OLD gate's median([190, 0.002, 0.002]) = 0.002s would have PASSED.
    cache_hit_samples = [190, 0.002, 0.002]
    assert_eq(core.compile_gate_implausible_samples(cache_hit_samples), [0.002, 0.002],
              "both extra samples are under 10% of the first (190s) sample: implausible cache hits")
    assert_true(not core.compile_gate_verdict(cache_hit_samples, LIMIT),
                "[190, 0.002, 0.002]: the gate FAILS on the implausible samples, not a falsely light median")

    # the mirror, legitimate case: a genuine slow first compile (over the limit) followed by two REAL cold
    # compiles that happen to be fast (not cache hits - comfortably over 10% of the first sample) must PASS on
    # their median, exactly as with `samples` above but restated here against the new implausibility check to
    # confirm it does not fire on legitimate cold samples.
    legit_outlier_then_cold = [190, 145, 155]
    assert_eq(core.compile_gate_implausible_samples(legit_outlier_then_cold), [],
              "145s and 155s are both well over 10% of the 190s first sample: not implausible")
    assert_true(core.compile_gate_verdict(legit_outlier_then_cold, LIMIT),
                "median([190,145,155]) = 155, under the 180s limit, and no sample is implausible: PASS")

    # a single implausible sample (not both) is still enough to fail the gate outright.
    assert_true(not core.compile_gate_verdict([190, 170, 0.002], LIMIT),
                "even one implausible extra sample (0.002s) fails the gate, regardless of the others")


def test_recent_messages_picks_the_newest_chat_and_last_n():
    # rev 7 round 2 (owner): "I only pull the most recent chat for behavior, so only the most recent replies
    # should be scanned." recent_messages() must (a) pick the chat with the NEWEST message, not the newest chat
    # row or an arbitrary one, (b) walk THAT chat's current branch table-first (same as load_corpus), and
    # (c) return only the last N messages of it, in chronological order.
    tmp = Path(tempfile.mkdtemp(prefix="wb-recent-"))
    try:
        db = tmp / "webui.db"
        c = sqlite3.connect(db)
        c.execute("create table chat (id text, chat json, current_message_id text)")
        c.execute("create table chat_message (id text, chat_id text, role text, parent_id text, content json, "
                  "output json, created_at bigint)")
        # chat "old": an older conversation, newest message ts=100
        c.execute("insert into chat values ('old', '{}', 'm2')")
        for mid, role, parent, ts in [("old-m1", "user", None, 90), ("old-m2", "assistant", "m1", 100)]:
            c.execute("insert into chat_message values (?, 'old', ?, ?, ?, null, ?)",
                      (mid, role, parent, json.dumps(f"old text {mid}"), ts))
        # chat "new": the most recently active one, newest message ts=500 - table-first branch walk, table rows
        # first with a JSON-history-only message filling a gap, off-branch message excluded. Table row ids are
        # prefixed with the chat id (OpenWebUI convention, stripped by _chat_messages); history/pointer ids and
        # parentIds are bare (unprefixed), matching chat.current_message_id and history.currentId.
        hist = {"history": {"currentId": "m3", "messages": {
            "m2h": {"id": "m2h", "parentId": "m1", "role": "assistant",
                   "content": "history-only filler", "timestamp": 400},
        }}}
        c.execute("insert into chat values ('new', ?, 'm3')", (json.dumps(hist),))
        rows = [("new-m1", "user", None, "new text 1", 300),
                ("new-m3", "assistant", "m2h", "new text 3 (newest)", 500),
                ("new-off", "assistant", "m1", "off-branch, must not appear", 350)]
        for mid, role, parent, text, ts in rows:
            c.execute("insert into chat_message values (?, 'new', ?, ?, ?, null, ?)",
                      (mid, role, parent, json.dumps(text), ts))
        c.commit()
        c.close()
        msgs, chat_id, chat_hash, newest_ts = core.recent_messages(db, n=200)
        assert_eq(chat_id, "new", "the chat with the newest message (ts=500) is picked, not the newer chat ROW")
        assert_eq(chat_hash, core.sha256_bytes(b"new"), "the stamp gets a sha256 of the chat id, not the id itself")
        assert_eq(newest_ts, 500, "the newest message timestamp is returned too (for the corpus-age check)")
        got = [m["t"] for m in msgs]
        assert_eq(got, ["new text 1", "history-only filler", "new text 3 (newest)"],
                  "table-first branch walk (JSON history fills the gap), off-branch excluded, chronological order")
        msgs2, _cid2, _h2, _nts2 = core.recent_messages(db, n=2)
        assert_eq([m["t"] for m in msgs2], ["history-only filler", "new text 3 (newest)"],
                  "n slices the LAST n messages of the branch, keeping order")

        class _Args:
            corpus = str(db)
            recent = 2
            full_corpus = False
        cm = W.load_corpus_mode(db, _Args())
        assert_eq(cm["stamp"]["mode"], "recent", "load_corpus_mode records mode='recent'")
        assert_eq(cm["stamp"]["n"], 2, "load_corpus_mode records the requested n")
        assert_eq(cm["stamp"]["chat_id_sha256"], core.sha256_bytes(b"new"),
                  "load_corpus_mode records the chat id's HASH, never the id itself")
        assert_eq(cm["stamp"]["messages"], 2, "load_corpus_mode records the message count actually used")
        assert_true("new" not in json.dumps(list(cm["stamp"].values())),
                    "the raw chat id never appears among the stamp's VALUES (a key named 'newest_message_ts' "
                    "containing the substring 'new' is fine - only checking values, not key names)")

        class _ArgsFull:
            corpus = str(db)
            full_corpus = True
        cmf = W.load_corpus_mode(db, _ArgsFull())
        assert_eq(cmf["stamp"]["mode"], "full", "--full-corpus opts back into the old whole-corpus behaviour")
        assert_eq(cmf["stamp"]["messages"], len(core.load_corpus(db)), "full mode scans every chat's branch")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_recent_messages_skips_test_chats_and_prefers_most_messages():
    # rev 7 FINAL hostile-review fix 6: "skip test chats when picking the 'newest chat' ... prefer the chat with
    # the most messages among those active in the last 24h; document the rule." A chat whose id/title contains
    # "selftest"/"itest" (this tool's own e2e/pod-check test conversation ids) is never picked, even if it has
    # the single newest message; among the remaining chats active within 24h of each other, the one with the
    # MOST messages wins, not simply the one with the single newest timestamp.
    tmp = Path(tempfile.mkdtemp(prefix="wb-recent-skip-"))
    try:
        db = tmp / "webui.db"
        c = sqlite3.connect(db)
        c.execute("create table chat (id text, title text, chat json, current_message_id text)")
        c.execute("create table chat_message (id text, chat_id text, role text, parent_id text, content json, "
                  "output json, created_at bigint)")
        now = 1_000_000
        # a test chat with the single NEWEST message - must be skipped
        c.execute("insert into chat values ('itest-wordban-check', 'itest-wordban-check', '{}', 'm1')")
        c.execute("insert into chat_message values ('itest-wordban-check-m1', 'itest-wordban-check', 'user', "
                  "null, ?, null, ?)", (json.dumps("test probe text"), now))
        # a real chat, active in the same window, with only 1 message
        c.execute("insert into chat values ('quiet', 'a quiet chat', '{}', 'q1')")
        c.execute("insert into chat_message values ('quiet-q1', 'quiet', 'user', null, ?, null, ?)",
                  (json.dumps("one message"), now - 100))
        # a real chat, active in the same 24h window, with the MOST messages - must win
        c.execute("insert into chat values ('busy', 'a busy chat', '{}', 'b3')")
        for i, ts in enumerate([now - 500, now - 400, now - 300]):
            c.execute("insert into chat_message values (?, 'busy', 'user', ?, ?, null, ?)",
                      (f"busy-b{i+1}", f"b{i}" if i else None, json.dumps(f"busy text {i}"), ts))
        c.commit()
        c.close()
        msgs, chat_id, _hash, _ts = core.recent_messages(db, n=200)
        assert_eq(chat_id, "busy", "the test chat is skipped, and among the rest the one with the most messages wins")
        assert_eq(len(msgs), 3, "all 3 of the busy chat's messages are returned")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_release_accepts_recent_mode_and_refuses_no_corpus_check():
    # rev 7 round 2 (owner): `release` accepts a recent-mode corpus stamp with no --allow-no-corpus needed; it
    # still refuses a stamp where no corpus check ran at all.
    tmp = Path(tempfile.mkdtemp(prefix="wb-rel-recent-"))
    try:
        bl = write(tmp, "b.yaml", mini_banlist())
        d = tmp / "b"
        run_tool(["build", "--here", "--banlist", str(bl), "--build-dir", str(d)])
        run_tool(["verify", "--no-image", "--fuzz", "300", "--build-dir", str(d)])
        st = json.loads((d / "verify-stamp.json").read_text(encoding="utf-8"))
        st.update(full=True, tool_sha256=W.tool_sha())
        st["rows"] = st.get("rows", []) + [{"check": "funnel gate: every state offers >= 21 natural ways on",
                                            "pass": True, "detail": ""}]
        st["image_facts"] = {"facts": {"first_compile_s": 1.0}}
        out_dir = tmp / "out"
        # no corpus check at all: still refused, exactly as before
        st["corpus"] = False
        st["corpus_mode"] = None
        (d / "verify-stamp.json").write_text(json.dumps(st), encoding="utf-8")
        rc, out = run_tool(["release", "--build-dir", str(d), "--out-dir", str(out_dir), "--allow-no-e2e"])
        assert_true(rc != 0 and "REFUSED" in out and "ran without --corpus" in out,
                    "no corpus check at all: still refused without --allow-no-corpus", out[-300:])
        # a recent-mode corpus stamp: accepted with NO --allow-no-corpus flag
        st["corpus"] = True
        st["corpus_mode"] = {"mode": "recent", "n": 200, "chat_id_sha256": "0" * 64, "messages": 37}
        (d / "verify-stamp.json").write_text(json.dumps(st), encoding="utf-8")
        rc, out = run_tool(["release", "--build-dir", str(d), "--out-dir", str(out_dir), "--allow-no-e2e"])
        assert_true(rc == 0, "a recent-mode corpus stamp is accepted with no --allow-no-corpus", out[-500:])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_release_refuses_without_a_full_passing_verify():
    tmp = Path(tempfile.mkdtemp(prefix="wb-rel-"))
    try:
        bl = write(tmp, "b.yaml", mini_banlist())
        d = tmp / "b"
        run_tool(["build", "--here", "--banlist", str(bl), "--build-dir", str(d)])
        out_dir = tmp / "out"
        rc, out = run_tool(["release", "--build-dir", str(d), "--out-dir", str(out_dir), "--allow-no-e2e"])
        assert_true(rc != 0 and "REFUSED" in out and "no verify stamp" in out, "no stamp: refused", out[-300:])
        run_tool(["verify", "--no-image", "--fuzz", "300", "--build-dir", str(d)])
        rc, out = run_tool(["release", "--build-dir", str(d), "--out-dir", str(out_dir), "--allow-no-e2e"])
        assert_true(rc != 0 and "--no-image" in out, "a local-only verify is not enough", out[-300:])
        st = json.loads((d / "verify-stamp.json").read_text(encoding="utf-8"))
        st.update(full=True, corpus=True, tool_sha256="0" * 64)
        (d / "verify-stamp.json").write_text(json.dumps(st), encoding="utf-8")
        rc, out = run_tool(["release", "--build-dir", str(d), "--out-dir", str(out_dir), "--allow-no-e2e"])
        assert_true(rc != 0 and "different tool code" in out, "a stamp written by other tool code: refused", out[-300:])
        st.update(full=True, corpus=True)
        (d / "verify-stamp.json").write_text(json.dumps(st), encoding="utf-8")
        (d / "wb_ban.ebnf").write_text((d / "wb_ban.ebnf").read_text(encoding="utf-8") + "\n", encoding="utf-8")
        rc, out = run_tool(["release", "--build-dir", str(d), "--out-dir", str(out_dir), "--allow-no-e2e"])
        assert_true(rc != 0 and "does not match its manifest" in out, "an artifact edited after the build: refused",
                    out[-300:])
        assert_true(not out_dir.exists() or not any(out_dir.iterdir()), "nothing was written by a refused release")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_pod_check_reports_a_truncated_paste_without_a_traceback(build_dir):
    src = (WB / "check_template.py").read_text(encoding="utf-8")
    ebnf = (build_dir / "wb_ban.ebnf").read_text(encoding="utf-8")
    import hashlib
    src = (src.replace("__REVISION__", "6").replace("__SHA__", hashlib.sha256(ebnf.encode()).hexdigest())
           .replace("__MODEL__", '"m/x"').replace("__PROMPT__", '"p"').replace("__ALLOWED__", "[]"))
    tmp = Path(tempfile.mkdtemp(prefix="wb-chk-"))
    try:
        chk = tmp / "word-ban-check.py"
        chk.write_text(src, encoding="utf-8")
        value = json.dumps({"grammar": ebnf})
        lines = ebnf.splitlines()
        cases = {
            "cut in the middle of the JSON": value[:len(value) // 2],
            "cut at a rule boundary, JSON re-closed": json.dumps({"grammar": "\n".join(lines[:len(lines) // 2]) + "\n"}),
            "cut inside a char class, JSON re-closed": json.dumps({"grammar": ebnf[:ebnf.index("[", len(ebnf) // 2) + 5]}),
        }
        for name, saved in cases.items():
            db = tmp / "webui.db"
            db.unlink(missing_ok=True)
            c = sqlite3.connect(db)
            c.execute("create table model (id text, params json)")
            c.execute("insert into model values ('m/x', ?)", (json.dumps({"custom_params": {"structured_outputs": saved}}),))
            c.commit()
            c.close()
            p = subprocess.run([sys.executable, str(chk), str(db)], capture_output=True, text=True, timeout=60)
            outp = p.stdout + p.stderr
            assert_true(p.returncode == 1 and "Traceback" not in outp and "truncated or corrupt" in outp,
                        f"pod check, {name}: the friendly message, exit 1, no traceback", outp[-400:])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_e2e_probes_are_built_from_the_banlist_and_agree_with_the_automaton(build_dir):
    bl = core.load_yaml(WB / "banlist.yaml")
    spec, ebnf = core.build(WB / "banlist.yaml")
    probes = W.build_probes(spec, bl)
    T = bl.get("tests") or {}
    mp = W.flat(T.get("must_pass"))
    mb = W.flat(T.get("must_block")) + [f for _, f in core.derived_must_block(spec)]
    STALE = ("The choir sang; Angela smiled in Los Angeles, and the marigold glowed.",
              "A white flamingo, flamenco and inflammation.")                  # the old hand-written probes
    texts = {t for t, _ in probes}
    assert_true(not (texts & set(STALE)), "the old hand-written probe sentences are gone", texts & set(STALE))
    got_true, got_false = {t for t, e in probes if e}, {t for t, e in probes if not e}
    assert_true(set(mp) <= got_true, "every must_pass entry is its own True probe", sorted(set(mp) - got_true)[:5])
    assert_true(set(mb) <= got_false, "every must_block entry is its own False probe", sorted(set(mb) - got_false)[:5])
    A = core.Automaton(ebnf)                                  # the same local automaton `verify` judges with
    bad = [(t, e, A.accepts(t)) for t, e in probes if A.accepts(t) != e]
    assert_eq(bad, [], f"all {len(probes)} probes (built from the real banlist) agree with the tool's own automaton")


def test_pod_check_interpreter_agrees_with_the_tool(build_dir):
    src = (WB / "check_template.py").read_text(encoding="utf-8")
    head = src[:src.index("try:\n    con = sqlite3.connect")]
    head = (head.replace("__REVISION__", "0").replace("__SHA__", "x").replace("__MODEL__", '"m"')
            .replace("__PROMPT__", '""').replace("__ALLOWED__", "[]"))
    ns = {}
    exec(compile(head, "word-ban-check.py", "exec"), ns)
    ebnf = (build_dir / "wb_ban.ebnf").read_text(encoding="utf-8")
    G, A = ns["Grammar"](ebnf), core.Automaton(ebnf)
    bl = core.load_yaml(WB / "banlist.yaml")
    texts = [str(x) for lst in bl["tests"]["must_block"] + bl["tests"]["must_pass"] for x in lst]
    texts += ["The choir sang in Los Angeles.", "the angelĂ", "flamingo🦩 and gold", "e" + "́" * 3]
    bad = [t for t in texts if (G.refused_at(t) is None) != A.accepts(t)]
    assert_eq(bad, [], f"the pod check's interpreter and the tool agree on {len(texts)} texts")


def test_cycle_runs_steps_in_order_and_stops_on_failure():
    tmp = Path(tempfile.mkdtemp(prefix="wb-cycle-"))
    order = []

    def fake_build(a):
        order.append("build")
        return 0

    def fake_verify(a):
        order.append("verify")
        return 0

    def fake_e2e(a):
        order.append("e2e")
        return 0

    def fake_diff(a):
        order.append("diff")
        return 0

    def fake_notes(rev, base_ref, new_bl_path, verify_line, e2e_line, diff_text, out_path, **_kw):
        order.append("notes")
        out_path.write_text("draft", encoding="utf-8")
        return out_path

    def fake_resolve(rev, out_dir, build_dir=None):
        return "git:deadbeef:scripts/wordban/banlist.yaml"

    orig = dict(cmd_build=W.cmd_build, cmd_verify=W.cmd_verify, cmd_e2e=wb_cmds.cmd_e2e, cmd_diff=wb_cmds.cmd_diff,
               write_notes=wb_cmds.write_notes, resolve=wb_cmds.resolve_release_banlist, HOME=W.HOME)
    try:
        W.cmd_build, W.cmd_verify = fake_build, fake_verify
        wb_cmds.cmd_e2e, wb_cmds.cmd_diff = fake_e2e, fake_diff
        wb_cmds.write_notes, wb_cmds.resolve_release_banlist = fake_notes, fake_resolve
        W.HOME = tmp
        (tmp / "build7").mkdir()
        (tmp / "build7" / "banlist.yaml").write_text("revision: 7\n", encoding="utf-8")
        args = types.SimpleNamespace(corpus="db.sqlite", base_release=None, no_e2e=False, out_dir=str(tmp),
                                     build_dir=str(tmp / "build7"), image="img", tokenizer="tok",
                                     recent=200, full_corpus=False, e2e_both=False)
        rc = wb_cmds.cmd_cycle(args)
        assert_eq(rc, 0, "cycle succeeds (rc 0) when every step is stubbed to succeed")
        assert_eq(order, ["build", "verify", "e2e", "diff", "notes"], "steps a-e run in order")

        # --no-e2e: step (c) is skipped entirely
        order.clear()
        args.no_e2e = True
        rc = wb_cmds.cmd_cycle(args)
        assert_eq(rc, 0, "cycle succeeds with --no-e2e")
        assert_eq(order, ["build", "verify", "diff", "notes"], "--no-e2e skips step (c) but keeps the rest in order")

        # stop-on-failure: a failing step stops every later step, with a clear message
        order.clear()
        args.no_e2e = False

        def fail_verify(a):
            order.append("verify")
            return 1
        W.cmd_verify = fail_verify
        rc = wb_cmds.cmd_cycle(args)
        assert_true(rc != 0, "cycle fails (non-zero) when a step fails")
        assert_eq(order, ["build", "verify"], "no step after the failing one runs")
    finally:
        W.cmd_build, W.cmd_verify = orig["cmd_build"], orig["cmd_verify"]
        wb_cmds.cmd_e2e, wb_cmds.cmd_diff = orig["cmd_e2e"], orig["cmd_diff"]
        wb_cmds.write_notes, wb_cmds.resolve_release_banlist = orig["write_notes"], orig["resolve"]
        W.HOME = orig["HOME"]
        shutil.rmtree(tmp, ignore_errors=True)


def test_notes_draft_generation_from_synthetic_banlists():
    tmp = Path(tempfile.mkdtemp(prefix="wb-notes-"))
    try:
        old = write(tmp, "old.yaml", mini_banlist())
        new_extra = "  gold:\n    rules:\n      - {id: GOLDX, match: goldx}\n"
        new_text = mini_banlist(extra_rules=new_extra).replace(
            'must_pass: [[Angela, Angelo, Los Angeles, evangelical, "the angle"]]',
            'must_pass: [[Angela, Angelo, Los Angeles, evangelical, "the angle", "a new pass"]]')
        new = write(tmp, "new.yaml", new_text)
        _la, _eb, sa = wb_cmds.load_side(str(old))
        sb, _ = core.build(new)
        rules = wb_cmds.diff_rules_data(sa, sb)
        added = [r for r in rules if r[0] == "+"]
        assert_true(any(r[1] == "GOLDX" for r in added), "a new rule shows as a + rule (mechanical, from the diff)",
                    added)
        tdiff = wb_cmds.diff_tests_data(sa, sb)
        assert_true("must_pass" in tdiff and "a new pass" in tdiff["must_pass"][0],
                    "a new must_pass entry shows as added in tests.must_pass", tdiff.get("must_pass"))
        out_path = tmp / "notes.md"
        wb_cmds.write_notes(99, str(old), new, "VERIFY PASSED: 5/5 checks", "E2E PASSED", "her corpus (2 messages)...",
                            out_path)
        text = out_path.read_text(encoding="utf-8")
        assert_true("GOLDX" in text, "the notes draft lists the new rule", text[:2000])
        assert_true("a new pass" in text, "the notes draft lists the new must_pass entry", text[:2000])
        assert_true("VERIFY PASSED: 5/5 checks" in text, "the notes draft includes the verify summary line")
        assert_true("E2E PASSED" in text, "the notes draft includes the e2e summary line")
        assert_true("revision 99" in text, "the notes draft names the revision")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_resolve_release_banlist_resolution():
    tmp = Path(tempfile.mkdtemp(prefix="wb-resolve-"))
    try:
        bl_bytes = mini_banlist().encode()
        h = core.sha256_bytes(bl_bytes)
        (tmp / "release.v5.json").write_text(json.dumps({"banlist_sha256": h}), encoding="utf-8")
        bdir = tmp / "build5"
        bdir.mkdir()
        (bdir / "banlist.yaml").write_bytes(bl_bytes)
        got = wb_cmds.resolve_release_banlist(5, tmp)
        assert_eq(got, str(bdir / "banlist.yaml"), "an exact revision match resolves via the conventional build<N> dir")
        got2 = wb_cmds.resolve_release_banlist(7, tmp)
        assert_eq(got2, str(bdir / "banlist.yaml"), "no release.v7.json: falls back to the newest release below it (v5)")
        (tmp / "release.v9.json").write_text(json.dumps({"banlist_sha256": "0" * 64}), encoding="utf-8")
        try:
            wb_cmds.resolve_release_banlist(9, tmp)
            assert_true(False, "resolve_release_banlist raises when nothing resolves (not a silent guess)")
        except ValueError as e:
            assert_true("REFUSED" in str(e), "the failure is loud (says REFUSED)", str(e))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_live_revision_defaults_the_diff_base():
    # rev 7 FINAL (coordinator): a top-level `live_revision:` (the revision actually pasted on prod) defaults
    # `diff`'s and `cycle`'s comparison base, instead of always assuming revision-1 - the exact gap that let v6
    # (built, released as a file, but never pasted) get used as a rollback/diff target when v5 was still live.
    assert_eq(wb_cmds.default_base_revision({"live_revision": 5}, 7), 5,
              "live_revision: present -> used, even though it is not revision-1")
    assert_eq(wb_cmds.default_base_revision({}, 7), 6,
              "live_revision: absent -> falls back to the previous behaviour (revision - 1)")
    assert_eq(wb_cmds.default_base_revision({"live_revision": 5}, 6), 5,
              "live_revision: still wins even when it DOES equal revision - 1")

    tmp = Path(tempfile.mkdtemp(prefix="wb-live-rev-"))
    try:
        bl_bytes = mini_banlist().encode()          # revision: 99, per mini_banlist()
        h = core.sha256_bytes(bl_bytes)
        (tmp / "release.v5.json").write_text(json.dumps({"banlist_sha256": h}), encoding="utf-8")
        bdir = tmp / "build5"
        bdir.mkdir()
        (bdir / "banlist.yaml").write_bytes(bl_bytes)
        new_text = mini_banlist().replace("revision: 99", "revision: 99\nlive_revision: 5")
        new_bl = write(tmp, "new.yaml", new_text)
        args = types.SimpleNamespace(old=None, new=str(new_bl), out_dir=str(tmp), corpus=None, show_forms=False)
        wb_cmds.cmd_diff(args)
        assert_eq(args.old, str(bdir / "banlist.yaml"),
                  "diff with `old` omitted resolves it via live_revision, not revision - 1 (which would be 98)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


_LIKE_GOOD = """
revision: 99
model: test/model
unicode_version: "0.0.0"
folding: {latin_blocks: [[0x00C0, 0x024F]], greek_blocks: [[0x0370, 0x03FF]], extra: {}, accented_continuation: dead,
         fffd: one_run_per_word, mark_cap: 2, mark_cap_hebrew: null}
separators:
  ALL: [" ", "-"]
  SPLIT: ["-", "_"]
families:
  foo:
    rules:
      - {id: FOO, match: foo}
      - {id: FOO-SPACED, match: foo, spaced: ALL}
      - {id: FOO-SPLIT, match: foo, split: SPLIT}
      - {id: FOO-IN, match: foo, where: inside}
  bar:
    rules:
      - {id: BAR, match: bar, like: foo}
      - {id: BAR-SPACED, match: bar, spaced: ALL, like: foo}
      - {id: BAR-SPLIT, match: bar, split: SPLIT, like: foo}
      - {id: BAR-IN, match: bar, where: inside, like: foo}
tests: {}
"""


def test_like_consistency_check_catches_a_mismatch():
    tmp = Path(tempfile.mkdtemp(prefix="wb-like-"))
    try:
        good = write(tmp, "good.yaml", _LIKE_GOOD)
        bl = core.load_yaml(good)
        assert_eq(W.like_mismatches(bl), [], "matching split:/spaced: set names and a glued form: no mismatch")
        bad_text = _LIKE_GOOD.replace('SPLIT: ["-", "_"]', 'SPLIT: ["-", "_"]\n  OTHER: ["~"]').replace(
            "- {id: BAR-SPLIT, match: bar, split: SPLIT, like: foo}",
            "- {id: BAR-SPLIT, match: bar, split: OTHER, like: foo}")
        bad = write(tmp, "bad.yaml", bad_text)
        bl2 = core.load_yaml(bad)
        bad_out = W.like_mismatches(bl2)
        assert_true(bad_out, "a family using a different split separator-set name than its `like:` target is caught",
                    bad_out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_run_step_coerces_a_string_sys_exit_into_a_failing_rc():
    # rev 7 hostile-review fix 4: sys.exit("boom") sets SystemExit.code to the STRING "boom", not an int; the old
    # `rc = 0 if e.code in (None, 0) else e.code` then fed that string straight to `int(rc or 0)` at the end of
    # run_step, raising ValueError instead of reporting a step failure.
    tmp = Path(tempfile.mkdtemp(prefix="wb-runstep-"))
    orig_home = W.HOME
    try:
        W.HOME = tmp

        def fn():
            sys.exit("boom")
        rc, dt, logp = wb_cmds.run_step(999, "string-exit-test", fn)
        assert_eq(rc, 1, "a string sys.exit(...) is reported as a step failure (rc 1), not a crash")
        assert_true(dt >= 0, "run_step still returns a duration")
        assert_true("boom" in logp.read_text(encoding="utf-8"), "the sys.exit message is in the step log", logp)

        def fn_int():
            sys.exit(3)
        rc2, _dt2, _logp2 = wb_cmds.run_step(999, "int-exit-test", fn_int)
        assert_eq(rc2, 3, "an integer sys.exit(N) still becomes rc N")

        def fn_none():
            sys.exit()
        rc3, _dt3, _logp3 = wb_cmds.run_step(999, "none-exit-test", fn_none)
        assert_eq(rc3, 0, "a bare sys.exit() still becomes rc 0")
    finally:
        W.HOME = orig_home
        shutil.rmtree(tmp, ignore_errors=True)


def test_like_multiset_machinery_still_works_without_the_yl_wf_pair():
    # rev 7 round 2: white_flame (and the yellow_light<->white_flame targeted pairing check) was removed - it
    # cost a real grammar-size fraction (7 rules) for zero behavioural effect, since "flame" is already
    # unconditionally banned by FLAM on its own. The real banlist no longer declares any `like:` at all; the
    # generic multiset machinery (like_shape/like_targets/like_mismatches) stays for a future family, checked
    # here against the real banlist (no declared `like:`, so nothing to mismatch) and against a synthetic one
    # (still exercised by test_like_consistency_check_catches_a_mismatch above).
    bl = core.load_yaml(WB / "banlist.yaml")
    assert_true("white_flame" not in (bl.get("families") or {}), "white_flame was removed from the real banlist")
    assert_eq(W.like_targets(bl), {}, "the real banlist declares no `like:` any more")
    assert_eq(W.like_mismatches(bl), [], "no `like:` declared means nothing to mismatch")


def test_check_template_db_resolution_and_outcomes():
    # rev 7 hostile-review fix 3: db resolution precedence (argv > DATABASE_URL > WEBUI_DB_LOCAL-derived), clean
    # errors (no traceback) for a bad DATABASE_URL / missing file, and the third "cut inside an unfinished word"
    # outcome, distinct from both "no banned word" and "BANNED WORD".
    tmpl = (WB / "check_template.py").read_text(encoding="utf-8")
    tmp = Path(tempfile.mkdtemp(prefix="wb-check3-"))
    try:
        # tiny real grammar so the interpreter has something to judge: "zorbex" is banned, everything else passes
        bl = write(tmp, "b.yaml", mini_banlist(extra_rules="""  zorbex:
    rules:
      - {id: ZORBEX, match: zorbex}"""))
        spec, ebnf = core.build(bl)
        import hashlib
        sha = hashlib.sha256(ebnf.encode()).hexdigest()
        src = (tmpl.replace("__REVISION__", "99").replace("__SHA__", sha).replace("__MODEL__", '"test/model"')
               .replace("__PROMPT__", '"p"').replace("__ALLOWED__", "[]"))
        assert_true("max_tokens\": 900" in src, "max_tokens was raised to ~900 (fix 3)")
        chk = tmp / "word-ban-check.py"
        chk.write_text(src, encoding="utf-8")

        def make_db(name, value=None, model_id="test/model"):
            p = tmp / name
            p.unlink(missing_ok=True)
            c = sqlite3.connect(p)
            c.execute("create table model (id text, params json)")
            if value is not None:
                c.execute("insert into model values (?, ?)",
                          (model_id, json.dumps({"custom_params": {"structured_outputs": value}})))
            c.commit()
            c.close()
            return p

        val = json.dumps({"grammar": ebnf})
        argv_db = make_db("argv.db", val)

        def run(argv_extra, env_extra):
            e = dict(os.environ)
            for k in ("DATABASE_URL", "WEBUI_DB_LOCAL"):
                e.pop(k, None)
            e.update(env_extra or {})
            p = subprocess.run([sys.executable, str(chk)] + (argv_extra or []), capture_output=True, text=True,
                              env=e, timeout=60)
            return p.returncode, p.stdout + p.stderr

        # these only exercise db RESOLUTION (the source line printed first); the actual A/B/C requests then fail
        # with "connection refused" since no vLLM/compactor is listening in this test - that is expected here and
        # is not what these assertions check.
        rc, out = run([str(argv_db)], {})
        assert_true(f"db: {argv_db} (sys.argv[1])" in out, "sys.argv[1] takes precedence", out[:400])

        db_url_db = make_db("dburl.db", val)
        rc, out = run([], {"DATABASE_URL": f"sqlite:///{db_url_db}"})
        assert_true("DATABASE_URL" in out.splitlines()[0], "DATABASE_URL is used when argv is absent", out[:400])

        rc, out = run([], {"DATABASE_URL": "postgres://x/y"})
        assert_true(rc != 0 and "Traceback" not in out and "not a plain sqlite" in out,
                    "a non-sqlite DATABASE_URL is reported clearly, not silently ignored, no traceback", out)

        rc, out = run([], {"DATABASE_URL": f"sqlite:///{db_url_db}?mode=ro"})
        assert_true(rc != 0 and "Traceback" not in out and "not a plain sqlite" in out,
                    "a DATABASE_URL with a ?query suffix is reported clearly, not silently ignored", out)

        # rev 7 FINAL hostile-review fix 6: WEBUI_LOCAL_DB / WEBUI_SNAPSHOT_DB / DATA_DIR override the defaults,
        # exactly like v3.1.9's entrypoint.sh (entrypoint.sh:279-280) - tested directly via env vars, no patching.
        env_local_db = make_db("env-local.db", val)
        rc, out = run([], {"WEBUI_LOCAL_DB": str(env_local_db)})
        assert_true(f"db: {env_local_db}" in out, "WEBUI_LOCAL_DB overrides the default local path", out[:400])
        env_snap_db = make_db("env-snap.db", val)
        rc, out = run([], {"WEBUI_DB_LOCAL": "false", "WEBUI_SNAPSHOT_DB": str(env_snap_db)})
        assert_true(f"db: {env_snap_db}" in out, "WEBUI_SNAPSHOT_DB overrides the default snapshot path", out[:400])
        data_dir = tmp / "datadir"
        data_dir.mkdir()
        make_db("datadir/webui.db", val)
        rc, out = run([], {"WEBUI_DB_LOCAL": "false", "DATA_DIR": str(data_dir)})
        assert_true(f"db: {data_dir / 'webui.db'}" in out,
                    "DATA_DIR relocates the snapshot default (<DATA_DIR>/webui.db)", out[:400])

        # WEBUI_DB_LOCAL-derived: patch LOCAL_DB/SNAPSHOT_DB to tmp paths so the test doesn't touch /var or /data
        local_db = make_db("local.db", val)
        snap_db = make_db("snap.db", val)
        src2 = src.replace(
            'LOCAL_DB = os.environ.get("WEBUI_LOCAL_DB") or "/var/lib/openwebui/webui.db"',
            f'LOCAL_DB = "{local_db}"').replace(
            'SNAPSHOT_DB = os.environ.get("WEBUI_SNAPSHOT_DB") or (os.environ.get("DATA_DIR") or '
            '"/data/openwebui") + "/webui.db"',
            f'SNAPSHOT_DB = "{snap_db}"')
        assert_true(f'LOCAL_DB = "{local_db}"' in src2, "the LOCAL_DB patch actually matched the source text")
        chk2 = tmp / "word-ban-check2.py"
        chk2.write_text(src2, encoding="utf-8")

        def run2(env_extra):
            e = dict(os.environ)
            for k in ("DATABASE_URL", "WEBUI_DB_LOCAL"):
                e.pop(k, None)
            e.update(env_extra or {})
            p = subprocess.run([sys.executable, str(chk2)], capture_output=True, text=True, env=e, timeout=60)
            return p.returncode, p.stdout + p.stderr
        rc, out = run2({})
        assert_true(f"db: {local_db}" in out, "WEBUI_DB_LOCAL unset defaults to true (the local db)", out[:400])
        rc, out = run2({"WEBUI_DB_LOCAL": "true"})
        assert_true(f"db: {local_db}" in out, "WEBUI_DB_LOCAL=true uses the local db", out[:400])
        rc, out = run2({"WEBUI_DB_LOCAL": "FALSE"})
        assert_true(f"db: {snap_db}" in out, "WEBUI_DB_LOCAL=FALSE (case-folded) uses the snapshot", out[:400])
        rc, out = run2({"WEBUI_DB_LOCAL": "sideways"})
        assert_true(rc != 0 and "Traceback" not in out and "neither a recognised true" in out,
                    "an unrecognised WEBUI_DB_LOCAL is a clear message, not a crash", out)

        missing = tmp / "does-not-exist.db"
        rc, out = run([str(missing)], {})
        assert_true(rc != 0 and "Traceback" not in out, "a missing db file is a clear message, not a traceback", out)

        corrupt = tmp / "corrupt.db"
        corrupt.write_bytes(b"not a sqlite file at all")
        rc, out = run([str(corrupt)], {})
        assert_true(rc != 0 and "Traceback" not in out, "a corrupt db file is a clear message, not a traceback", out)

        # the third outcome: use the REAL banlist (its FLAM stem: alone/other ban, allow_prefixes mab/b) so a
        # length cutoff mid "flamm" (not yet "flammable"/"flammability", not yet dead either - a live, unfinished
        # continuation) is genuinely NOT ACCEPTING and NEVER REFUSED: exactly "cut inside an unfinished word",
        # distinct from both a plain pass and a banned word.
        import hashlib as _hashlib
        _spec, real_ebnf_text = core.build(WB / "banlist.yaml")
        real_sha = _hashlib.sha256(real_ebnf_text.encode()).hexdigest()
        real_bl = core.load_yaml(WB / "banlist.yaml")
        real_src = (tmpl.replace("__REVISION__", str(real_bl["revision"])).replace("__SHA__", real_sha)
                   .replace("__MODEL__", json.dumps(real_bl["model"])).replace("__PROMPT__", '"p"')
                   .replace("__ALLOWED__", "[]"))
        chk_real = tmp / "word-ban-check-real.py"
        chk_real.write_text(real_src, encoding="utf-8")
        real_val = json.dumps({"grammar": real_ebnf_text})
        real_db2 = make_db("real.db", real_val, model_id=real_bl["model"])
        stub = """import sys, json, io, urllib.request, runpy
class R(io.BytesIO):
    def __enter__(s): return s
    def __exit__(s, *a): pass
def fake(req, timeout=None):
    return R(json.dumps({"choices": [{"message": {"content": "a sentence with flamm"},
                                       "finish_reason": "length"}]}).encode())
urllib.request.urlopen = fake
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
"""
        stubp = tmp / "stub_run.py"
        stubp.write_text(stub, encoding="utf-8")
        p = subprocess.run([sys.executable, str(stubp), str(chk_real), str(real_db2)], capture_output=True,
                           text=True, timeout=120)
        out = p.stdout + p.stderr
        assert_true("cut inside an unfinished word" in out, "the third outcome is reported distinctly", out[-2000:])
        assert_true("BANNED WORD" not in out, "an unfinished-but-unrefused tail is never reported as a banned word",
                    out[-2000:])
        assert_true(p.returncode == 1,
                    "overall FAIL (A's control needs a banned word, which this stub never writes) but not because "
                    "'cut inside an unfinished word' was itself treated as a banned word", out[-800:])

        # rev 7 FINAL hostile-review fix 2 (MEDIUM): the trap-shape detector itself (rev-4-style trap simulation:
        # a long unfinished word, or one holding combining marks/U+FFFD - the exact shape the 2026-09-23 trap
        # produced). Tested directly against the helper functions (execing just the head, like
        # test_pod_check_interpreter_agrees_with_the_tool does) - constructing a REAL reachable grammar state
        # that is both "mid an unfinished word" and "40+ chars" is not generally possible against a trap-free
        # grammar (that is the whole point of the no-trap design), so the unit-testable surface is the detector.
        head = src[:src.index("try:\n    con = sqlite3.connect")]
        head = head.replace("__REVISION__", "99").replace("__SHA__", "x").replace("__MODEL__", '"m"').replace(
            "__PROMPT__", '""').replace("__ALLOWED__", "[]")
        ns = {}
        exec(compile(head, "word-ban-check.py", "exec"), ns)
        looks_like_a_trap, trailing_unfinished_word = ns["looks_like_a_trap"], ns["trailing_unfinished_word"]
        assert_eq(trailing_unfinished_word("a sentence with flamm"), "flamm", "the trailing unfinished word alone")
        assert_true(not looks_like_a_trap("flamm"), "a short, plain unfinished word is NOT trap-shaped")
        assert_true(looks_like_a_trap("a" * 41), "a run of 41 plain letters IS trap-shaped (over the 40-char limit)")
        assert_true(not looks_like_a_trap("a" * 40), "exactly 40 plain letters is NOT over the limit")
        assert_true(looks_like_a_trap("áb"), "a short tail WITH a combining mark IS trap-shaped")
        assert_true(looks_like_a_trap("a�b"), "a short tail WITH U+FFFD IS trap-shaped")
        # rev 7 FINAL (targeted review, fix 2): the REAL bug - trailing_unfinished_word() used to stop at the
        # first non-alnum/non-WORDCHAR character, so a tail ENDING in combining marks or U+FFFD (exactly the
        # rev-4 trap's own shape) produced an EMPTY string, and looks_like_a_trap("") is never trap-shaped -
        # silently defeating the whole check on the one input it exists to catch.
        mark_tail = "a" + ("\u0301" * 300)
        assert_eq(trailing_unfinished_word("prefix text " + mark_tail), mark_tail,
                  "a trailing run of combining marks is NOT silently dropped from the extracted tail any more")
        assert_true(looks_like_a_trap(trailing_unfinished_word("prefix text " + mark_tail)),
                    "...and is correctly detected as trap-shaped once extracted")
        fffd_tail = "a\ufffd\ufffd\ufffd"
        assert_eq(trailing_unfinished_word("prefix " + fffd_tail), fffd_tail,
                  "a trailing run of U+FFFD is NOT silently dropped from the extracted tail either")

        # the coordinator's own reproduction: angel\u0102 + 300 combining marks against the REAL rev-4 grammar
        # (wb_ban.v4.ebnf) must FAIL as a trap; a normal mid-word cut against the same grammar stays non-trap.
        v4 = Path("~/zl-ops/word-ban/wb_ban.v4.ebnf").expanduser()
        if v4.exists():
            G4 = ns["Grammar"](v4.read_text(encoding="utf-8"))
            trap_text = "angel\u0102" + ("\u0301" * 300)
            at = G4.refused_at(trap_text)
            tail = trailing_unfinished_word(trap_text)
            assert_true(at == len(trap_text), "the rev-4 grammar never outright refuses this text (it is the "
                        "trap: incomplete, not accepting, never refused)", at)
            assert_true(looks_like_a_trap(tail), "angel\u0102 + 300 combining marks against wb_ban.v4.ebnf IS trap-shaped")
            normal_text = "a sentence about something ordinary"
            normal_at = G4.refused_at(normal_text)
            if normal_at is not None:
                normal_tail = trailing_unfinished_word(normal_text[:normal_at])
                assert_true(not looks_like_a_trap(normal_tail), "an ordinary mid-word cut is NOT trap-shaped",
                            normal_tail)
        else:
            print(f"  (skipped: {v4} not found on this machine)")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_cmd_e2e_uses_build_probes_and_build_probes_dedupes():
    # cmd_e2e must build probes from the banlist (W.build_probes), never a hardcoded list - the exact mistake
    # logged in memory.md ("wordban e2e probes were hardcoded and went stale when rev 6 changed the rules").
    src = (WB / "wb_cmds.py").read_text(encoding="utf-8")
    assert_true("W.build_probes(spec, bl)" in src, "cmd_e2e calls W.build_probes(spec, bl)")
    bl = core.load_yaml(WB / "banlist.yaml")
    spec, _ = core.build(WB / "banlist.yaml")
    probes = W.build_probes(spec, bl)
    keys = [(t, e) for t, e in probes]
    assert_eq(len(keys), len(set(keys)), "build_probes has no duplicate (text, expected) pairs")
    T = bl.get("tests") or {}
    mp0 = W.flat(T.get("must_pass"))[0]
    assert_eq(probes[0], [mp0, True], "the first probe is the first must_pass entry, unmoved by de-dup (order kept)")


if __name__ == "__main__":
    bd = test_generator_is_deterministic()
    try:
        test_real_banlist_has_no_trap_and_the_live_trap_prefixes_are_refused(bd)
        test_yellow_light_glued_carve_outs_are_gone(bd)
        test_e2e_probes_are_built_from_the_banlist_and_agree_with_the_automaton(bd)
        test_pod_check_interpreter_agrees_with_the_tool(bd)
        test_pod_check_reports_a_truncated_paste_without_a_traceback(bd)
    finally:
        shutil.rmtree(bd.parent, ignore_errors=True)
    test_verifier_fails_on_revision_4s_angel_trap()
    test_a_word_added_to_the_yaml_is_tested_and_blocked()
    test_corpus_loader_walks_the_branch_table_first_and_prefers_output()
    test_corpus_sample_rate_reaches_the_floor_in_recent_mode()
    test_compile_gate_median_logic_with_fake_timings()
    test_recent_messages_picks_the_newest_chat_and_last_n()
    test_recent_messages_skips_test_chats_and_prefers_most_messages()
    test_release_accepts_recent_mode_and_refuses_no_corpus_check()
    test_release_refuses_without_a_full_passing_verify()
    test_cycle_runs_steps_in_order_and_stops_on_failure()
    test_notes_draft_generation_from_synthetic_banlists()
    test_resolve_release_banlist_resolution()
    test_live_revision_defaults_the_diff_base()
    test_like_consistency_check_catches_a_mismatch()
    test_like_multiset_machinery_still_works_without_the_yl_wf_pair()
    test_run_step_coerces_a_string_sys_exit_into_a_failing_rc()
    test_check_template_db_resolution_and_outcomes()
    test_cmd_e2e_uses_build_probes_and_build_probes_dedupes()
    if _FAILS:
        print(f"\n{len(_FAILS)} FAILURE(S): {_FAILS}")
        sys.exit(1)
    print("\nAll wordban tool tests passed.")
