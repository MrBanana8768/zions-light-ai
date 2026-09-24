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
from pathlib import Path

_HERE = Path(__file__).resolve().parent
WB = _HERE.parent / "scripts" / "wordban"
sys.path.insert(0, str(WB))
import wb_core as core  # noqa: E402

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


def test_pod_check_interpreter_agrees_with_the_tool(build_dir):
    src = (WB / "check_template.py").read_text(encoding="utf-8")
    head = src[:src.index("con = sqlite3.connect")]
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


if __name__ == "__main__":
    bd = test_generator_is_deterministic()
    try:
        test_real_banlist_has_no_trap_and_the_live_trap_prefixes_are_refused(bd)
        test_pod_check_interpreter_agrees_with_the_tool(bd)
        test_pod_check_reports_a_truncated_paste_without_a_traceback(bd)
    finally:
        shutil.rmtree(bd.parent, ignore_errors=True)
    test_verifier_fails_on_revision_4s_angel_trap()
    test_a_word_added_to_the_yaml_is_tested_and_blocked()
    test_corpus_loader_walks_the_branch_table_first_and_prefers_output()
    test_release_refuses_without_a_full_passing_verify()
    if _FAILS:
        print(f"\n{len(_FAILS)} FAILURE(S): {_FAILS}")
        sys.exit(1)
    print("\nAll wordban tool tests passed.")
