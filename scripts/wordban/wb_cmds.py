"""wordban subcommands e2e, diff and release (build, verify and tokenizer live in wordban.py)."""
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import unicodedata
from pathlib import Path

import wb_core as core
import wordban as W

REPO = W.HERE.parent.parent


# ============================================================================================================
# e2e
# ============================================================================================================
TRIM_TABLES = ("chat", "chat_message", "message", "chatidtag", "memory", "file", "chat_file", "shared_chat",
               "feedback", "prompt_history", "document", "knowledge", "knowledge_file", "note", "channel_message")


def cmd_e2e(args):
    build_dir, man = W.load_build(args.build_dir)
    bl = core.load_yaml(build_dir / "banlist.yaml")
    model = bl.get("model")
    tokd = Path(args.tokenizer).expanduser().resolve()
    priv = W.private_dir()
    results = {"grammar_sha256": man["grammar_sha256"], "runs": []}
    ok = True
    try:
        db = W.copy_db(args.webui_db, priv)
        c = sqlite3.connect(db)
        for t in TRIM_TABLES:                   # her chats are not needed: users, models and config are kept
            try:
                c.execute(f"delete from {t}")
            except sqlite3.Error:
                pass
        c.commit()
        c.execute("vacuum")
        c.close()
        db.rename(priv / "e2e.db")
        shutil.copyfile(build_dir / W.VALUE, priv / "value.txt")
        spec = core.Spec(bl)
        probes = [["The choir sang; Angela smiled in Los Angeles, and the marigold glowed.", True],
                  [" ".join(f for _, f in core.derived_must_block(spec)[:3]), False],
                  ["A white flamingo, flamenco and inflammation.", True]]
        (priv / "probes.json").write_text(json.dumps(probes), encoding="utf-8")
        compactors = [("image", "/opt/compactor")]
        if args.compactor_ref:
            src = priv / "compactor-src"
            src.mkdir()
            tar = subprocess.run(["git", "-C", str(REPO), "-c", "safe.directory=*", "archive", args.compactor_ref,
                                  "compactor"], check=True, capture_output=True).stdout
            subprocess.run(["tar", "-x", "-C", str(src)], input=tar, check=True)
            compactors.insert(0, (args.compactor_ref, "/priv/compactor-src/compactor"))
        os.chmod(priv, 0o755)
        for p in priv.rglob("*"):
            os.chmod(p, 0o755 if p.is_dir() else 0o644)
        for label, cdir in compactors:
            W.say(f"== compactor {label} ({cdir})")
            cmd = ["docker", "run", "--rm", "--name", f"wb-e2e-{os.getpid()}-{label.replace('.', '')}",
                   "-v", f"{W.HERE}:/wb:ro", "-v", f"{tokd}:/tok:ro", "-v", f"{priv}:/priv:ro",
                   "--entrypoint", "bash", args.image, "/wb/e2e/inner.sh", cdir, model]
            out = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL).stdout
            run = {"compactor": label, "edit": None, "chats": [], "requests": []}
            for line in out.splitlines():
                if line.startswith(("E2E-HEALTH", "E2E-COMPACTOR")):
                    W.say("  " + line)
                elif line.startswith("E2E-EDIT "):
                    run["edit"] = json.loads(line[9:])
                elif line.startswith("E2E-CHAT "):
                    run["chats"].append(json.loads(line[9:]))
                elif line.startswith("E2E-STUB "):
                    run["requests"].append(json.loads(line[9:]))
            checks = []
            e = run["edit"] or {}
            checks.append(("model update through OpenWebUI's own API saved the value unchanged",
                           e.get("http") == 200 and e.get("identical_to_file")))
            gr = [r for r in run["requests"] if r.get("tag", "").startswith("[wb-e2e B")]
            checks.append(("both ban-phase chats (plain + streaming) reached the vLLM port", len(gr) == 2))
            checks.append(("the grammar arrived byte-identical (sha256)",
                           bool(gr) and all(r["grammar_sha256"] == man["grammar_sha256"] for r in gr)))
            checks.append(("vLLM 0.19's own request validation + xgrammar compile accept it",
                           bool(gr) and all(r["vllm"].get("vllm_accepts_request") for r in gr)))
            checks.append(("probe sentences judged as expected by the compiled grammar",
                           bool(gr) and all(p["accepted"] == p["expected"] for r in gr for p in r["vllm"].get("probes", []))))
            checks.append(("no bad_words left in the forwarded body", bool(gr) and all(r["bad_words"] is None for r in gr)))
            for name, good in checks:
                W.say(f"  [{'PASS' if good else 'FAIL'}] {name}")
                ok = ok and bool(good)
            if gr:
                W.say(f"  compile inside the stub: {[r['vllm'].get('compile_s') for r in gr]} s; "
                      f"grammar {gr[0]['grammar_chars']:,} chars")
            if not all(g for _, g in checks):
                W.say("  (raw) " + out[-3000:])
            run["checks"] = [{"check": n, "pass": bool(g)} for n, g in checks]
            results["runs"].append(run)
    finally:
        shutil.rmtree(priv, ignore_errors=True)
    results["pass"] = ok
    results["when"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (build_dir / W.E2E_STAMP).write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    W.say(f"\nE2E {'PASSED' if ok else 'FAILED'}")
    return 0 if ok else 1


# ============================================================================================================
# diff
# ============================================================================================================
def load_side(ref):
    """a banlist .yaml, git:REV[:path], a grammar .ebnf, or a paste value .txt -> (label, ebnf, spec|None)"""
    if ref.startswith("git:"):
        rest = ref[4:]
        rev, _, path = rest.partition(":")
        path = path or "scripts/wordban/banlist.yaml"
        data = subprocess.run(["git", "-C", str(REPO), "show", f"{rev}:{path}"], check=True,
                              capture_output=True).stdout
        tmp = Path(tempfile.mkdtemp()) / "banlist.yaml"
        tmp.write_bytes(data)
        ref = str(tmp)
    p = Path(ref).expanduser()
    if p.suffix in (".yaml", ".yml"):
        spec, ebnf = core.build(p)
        return str(ref), ebnf, spec
    text = p.read_text(encoding="utf-8")
    if p.suffix == ".txt" or text.lstrip().startswith("{"):
        text = json.loads(text)["grammar"]
    return str(ref), text, None


def blocked_words(A, text, limit=400):
    """the words the grammar stops, found by running it, masking the word it stops at, and running again"""
    out = []
    for _ in range(limit):
        s, at = A.run(text)
        if at is None:
            if s is not None and A.acc[s]:
                return out
            at = len(text) - 1
        i = at
        wc = lambda ch: ch.isalnum() or unicodedata.category(ch) in core.MARK_CATS or ch in "_'’-"
        if not wc(text[i]):
            j = i - 1
            while j >= 0 and not wc(text[j]) and i - j < 4:
                j -= 1
            i = j if j >= 0 and wc(text[j]) else i
        a = b = i
        while a > 0 and wc(text[a - 1]):
            a -= 1
        while b < len(text) and wc(text[b]):
            b += 1
        b = max(b, i + 1)
        out.append(text[a:b])
        text = text[:a] + " Qzx " + text[b:]
    return out


def cmd_diff(args):
    la, ea, sa = load_side(args.old)
    lb, eb, sb = load_side(args.new)
    W.say(f"OLD {la}: grammar {len(ea):,} B sha256 {core.sha256_bytes(ea.encode())[:16]}")
    W.say(f"NEW {lb}: grammar {len(eb):,} B sha256 {core.sha256_bytes(eb.encode())[:16]}")
    if sa and sb:
        ra = {r["id"]: (f, r) for f, r in sa.rules}
        rb = {r["id"]: (f, r) for f, r in sb.rules}
        for k in rb:
            if k not in ra:
                W.say(f"  + rule {k} ({rb[k][0]}): {json.dumps(rb[k][1], ensure_ascii=False)}")
        for k in ra:
            if k not in rb:
                W.say(f"  - rule {k} ({ra[k][0]}): {json.dumps(ra[k][1], ensure_ascii=False)}")
            elif ra[k][1] != rb[k][1]:
                W.say(f"  ~ rule {k}: {json.dumps(ra[k][1], ensure_ascii=False)}\n"
                      f"          -> {json.dumps(rb[k][1], ensure_ascii=False)}")
        for key in ("accented_continuation", "mark_cap", "mark_cap_hebrew", "fffd"):
            if sa.bl.get("folding", {}).get(key) != sb.bl.get("folding", {}).get(key):
                W.say(f"  ~ folding.{key}: {sa.bl['folding'].get(key)} -> {sb.bl['folding'].get(key)}")
    A, B = core.Automaton(ea), core.Automaton(eb)
    ta, tb = A.trap_states()[0], B.trap_states()[0]
    W.say(f"  trap states (character level): OLD {len(ta)}, NEW {len(tb)}")
    tests = []
    for s in (sa, sb):
        if s:
            T = s.bl.get("tests") or {}
            tests += W.flat(T.get("must_block")) + W.flat(T.get("must_pass")) + [f for _, f in core.derived_must_block(s)]
    changed = [(t, A.accepts(t), B.accepts(t)) for t in dict.fromkeys(tests)]
    changed = [c for c in changed if c[1] != c[2]]
    W.say(f"  listed test strings whose verdict changes: {len(changed)}")
    for t, a, b in changed[:80]:
        W.say(f"    {'allowed -> BLOCKED' if a else 'blocked -> ALLOWED'}  {t!r}")
    if args.corpus:
        priv = W.private_dir()
        try:
            db = W.copy_db(args.corpus, priv)
            msgs = core.load_corpus(db)
            db.unlink()
            ref = core.Reference(sb or sa) if (sb or sa) else None
            fam_of = {r["id"]: f for f, r in (sb or sa).rules} if (sb or sa) else {}
            newly, freed = {}, {}
            forms_n, forms_f = {}, {}
            t0 = time.time()
            for m in msgs:
                wa = blocked_words(A, m["t"])
                wb = blocked_words(B, m["t"])
                ca, cb = {}, {}
                for w in wa:
                    ca[w] = ca.get(w, 0) + 1
                for w in wb:
                    cb[w] = cb.get(w, 0) + 1
                for w in set(ca) | set(cb):
                    d = cb.get(w, 0) - ca.get(w, 0)
                    if not d:
                        continue
                    fam = "other"
                    if ref:
                        h = ref.hits(w)
                        fam = fam_of.get(h[0][2], h[0][2]) if h else "other"
                    key = (m["role"], fam)
                    if d > 0:
                        newly[key] = newly.get(key, 0) + d
                        forms_n[w] = forms_n.get(w, 0) + d
                    else:
                        freed[key] = freed.get(key, 0) - d
                        forms_f[w] = forms_f.get(w, 0) - d
            W.say(f"  her corpus ({len(msgs)} messages, {time.time() - t0:.0f}s), occurrences whose verdict changes "
                  "(counts only):")
            W.say(f"    newly BLOCKED: {dict(sorted(newly.items()))}")
            W.say(f"    newly ALLOWED: {dict(sorted(freed.items()))}")
            if args.show_forms:
                W.say("    forms newly blocked (terminal only, never saved): "
                      + ", ".join(f"{w!r}x{n}" for w, n in sorted(forms_n.items(), key=lambda x: -x[1])[:60]))
                W.say("    forms newly allowed (terminal only, never saved): "
                      + ", ".join(f"{w!r}x{n}" for w, n in sorted(forms_f.items(), key=lambda x: -x[1])[:60]))
        finally:
            shutil.rmtree(priv, ignore_errors=True)
    return 0


# ============================================================================================================
# release
# ============================================================================================================
def cmd_release(args):
    build_dir, man = W.load_build(args.build_dir)
    sp = build_dir / W.STAMP
    if not sp.exists():
        sys.exit("REFUSED: no verify stamp for this build - run `wordban.py verify` (full, with --corpus) first")
    stamp = json.loads(sp.read_text(encoding="utf-8"))
    why = []
    if stamp.get("grammar_sha256") != man["grammar_sha256"] or stamp.get("value_sha256") != man["value_sha256"]:
        why.append("the verify stamp is for a different artifact")
    if not stamp.get("pass"):
        why.append("verify did not pass")
    if stamp.get("tool_sha256") != W.tool_sha():
        why.append("the verify stamp was written by different tool code (edit, then verify again)")
    if stamp.get("image") != args.image:
        why.append(f"verify ran on image {stamp.get('image')}, not {args.image}")
    funnel = [r for r in stamp.get("rows", []) if r["check"].startswith("funnel gate")]
    if stamp.get("full") and not funnel:
        why.append("the verify stamp has no funnel-gate result (older tool)")
    elif funnel and not funnel[0]["pass"]:
        why.append("the funnel gate failed (a state narrower than allowed)")
    if not stamp.get("full"):
        why.append("verify ran with --no-image (the token-level proof did not run)")
    if not stamp.get("corpus") and not args.allow_no_corpus:
        why.append("verify ran without --corpus (no false-block measurement)")
    if why:
        sys.exit("REFUSED: " + "; ".join(why))
    e2e = None
    ep = build_dir / W.E2E_STAMP
    if ep.exists():
        e2e = json.loads(ep.read_text(encoding="utf-8"))
        if e2e.get("grammar_sha256") != man["grammar_sha256"]:
            e2e = None
    if e2e is None and not args.allow_no_e2e:
        sys.exit("REFUSED: no passing e2e stamp for this artifact - run `wordban.py e2e` (or pass --allow-no-e2e)")
    if e2e is not None and not e2e.get("pass"):
        sys.exit("REFUSED: e2e failed for this artifact")
    bl = core.load_yaml(build_dir / "banlist.yaml")
    rev = man["revision"]
    out = Path(args.out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    value = (build_dir / W.VALUE).read_bytes()
    grammar = (build_dir / W.GRAMMAR).read_bytes()
    targets = {f"word-ban-structured_outputs.value.v{rev}.txt": value, f"wb_ban.v{rev}.ebnf": grammar}
    for name, data in targets.items():
        p = out / name
        if p.exists() and p.read_bytes() != data and not args.force:
            sys.exit(f"REFUSED: {p} already exists with different content (a published revision {rev}); "
                     "bump `revision:` in the banlist, or pass --force")
    # archive the current main report under its own revision before replacing it
    for main, pat in (("word-ban.md", "word-ban.v{}.md"), ("word-ban-check.py", "word-ban-check.v{}.py")):
        mp = out / main
        if mp.exists():
            m = re.search(r"revision (\d+)", mp.read_text(encoding="utf-8")[:4000])
            if m and int(m.group(1)) != rev and not (out / pat.format(m.group(1))).exists():
                shutil.copyfile(mp, out / pat.format(m.group(1)))
    written = {}

    def put(name, data):
        p = out / name
        if p.exists():
            os.chmod(p, 0o644)
            p.unlink()
        p.write_bytes(data)
        written[name] = core.sha256_bytes(data)
    for name, data in targets.items():
        put(name, data)
    put("word-ban-structured_outputs.value.txt", value)
    tmpl = (W.HERE / "check_template.py").read_text(encoding="utf-8")
    chk = bl.get("check") or {}
    src = (tmpl.replace("__REVISION__", str(rev)).replace("__SHA__", man["grammar_sha256"])
           .replace("__MODEL__", json.dumps(bl.get("model")))
           .replace("__PROMPT__", json.dumps(chk.get("prompt", ""), ensure_ascii=False))
           .replace("__ALLOWED__", json.dumps(chk.get("allowed", []), ensure_ascii=False)))
    compile(src, "word-ban-check.py", "exec")
    put("word-ban-check.py", src.encode())
    put(f"word-ban-check.v{rev}.py", src.encode())
    prompt_line = " ".join(str(bl.get("prompt_line", "")).split())
    put(f"word-ban-prompt-line.v{rev}.txt", (prompt_line + "\n").encode())
    notes = Path(args.notes).read_text(encoding="utf-8") if args.notes else ""
    report = render_report(rev, man, stamp, e2e, bl, prompt_line, written, out, notes)
    put("word-ban.md", report.encode())
    put(f"word-ban.v{rev}.md", report.encode())
    rel = {"revision": rev, "released": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "files": written,
           "grammar_sha256": man["grammar_sha256"], "banlist_sha256": man["banlist_sha256"],
           "verify": {"pass": stamp["pass"], "when": stamp["when"]}, "e2e": None if e2e is None else e2e["pass"]}
    put(f"release.v{rev}.json", json.dumps(rel, indent=1).encode())
    W.say(f"released revision {rev} into {out}:")
    for k, v in written.items():
        W.say(f"  {k:48} sha256 {v}")
    return 0


def render_report(rev, man, stamp, e2e, bl, prompt_line, written, out, notes):
    rows = [r for r in stamp["rows"] if r["check"] != "corpus facts"]
    facts = next((json.loads(r["detail"]) for r in stamp["rows"] if r["check"] == "corpus facts"), None)
    withdrawn = {int(k): v for k, v in (bl.get("withdrawn_revisions") or {}).items()}
    prev = sorted((int(m.group(1)) for p in out.glob("word-ban-structured_outputs.value.v*.txt")
                   for m in [re.search(r"\.v(\d+)\.txt$", p.name)]
                   if m and int(m.group(1)) < rev and int(m.group(1)) not in withdrawn), reverse=True)
    L = [f"# Word ban, revision {rev}", "",
         f"Built by `scripts/wordban/wordban.py` from `scripts/wordban/banlist.yaml` (sha256 "
         f"`{man['banlist_sha256'][:16]}…`), verified {stamp['when']}"
         + (f", end-to-end checked {e2e['when']}" if e2e else "") + ". Every file below was written by "
         "`wordban.py release`, which refuses to run unless the full verification passed on this exact artifact.", ""]
    if notes:
        L += [notes.rstrip(), ""]
    L += ["## Steps for the owner", "",
          "### Step 1: Replace the ban value (OpenWebUI)", "",
          f"1. Open `word-ban-structured_outputs.value.txt` in Notepad. It is one line of **{man['value_bytes']:,}** "
          f"characters, sha256 `{man['value_sha256']}`. Press **Ctrl+A**, then **Ctrl+C**.",
          f"2. Go to **Admin Panel → Settings → Models → {bl.get('model')} → Advanced Params → Show**.",
          "3. In the existing **`structured_outputs`** row, replace the value with the clipboard. Leave the other rows "
          "as they are.", "4. Click **Save & Update**.", "",
          "### Step 2: The system-prompt line (same editor)", "",
          "Replace the existing \"… things cannot be typed …\" paragraph with this (also in "
          f"`word-ban-prompt-line.v{rev}.txt`):", "", "> " + prompt_line, "",
          "### Step 3: The on-pod check (RunPod web terminal)", "",
          "1. Open `word-ban-check.py` in Notepad and copy all of it.",
          "2. In the pod terminal, type `cat > /tmp/word-ban-check.py`, press Enter, paste, then press **Ctrl+D**.",
          "3. Run `python3 /tmp/word-ban-check.py`.", "",
          f"A pass: the first line says `= revision {rev}, pasted intact`; **A** reports a banned word; **B** and "
          "**C** report `no banned word`; the last line is `RESULT: PASS`. The first request with the new grammar "
          f"waits once while vLLM compiles it (measured {stamp.get('image_facts', {}).get('facts', {}).get('first_compile_s', '?')} s "
          "on the build machine, CPU).", "",
          "### Rollback", "",
          (f"Paste `word-ban-structured_outputs.value.v{prev[0]}.txt` (or an earlier one) into the row, or delete the row."
           if prev else "Delete the `structured_outputs` row."), ""]
    for k, why in sorted(withdrawn.items()):
        L += [f"**Never roll back to revision {k}:** {why}", ""]
    L += [
          "## Verification", "", "| check | result |", "|---|---|"]
    for r in rows:
        L.append(f"| {r['check']} | {'**PASS**' if r['pass'] else '**FAIL**'} {r['detail'].replace('|', '/')} |")
    if e2e:
        for run in e2e["runs"]:
            for c in run["checks"]:
                L.append(f"| e2e, compactor {run['compactor']}: {c['check']} | {'**PASS**' if c['pass'] else '**FAIL**'} |")
    if facts:
        L += ["", "## Her corpus (counts only)", "",
              f"- messages on the current branch of every chat: {facts['messages']}",
              f"- messages with a banned form: {facts['with_banned_form']}",
              f"- banned-form occurrences by family: {facts['occurrences']}",
              f"- watched words (not banned): {facts['watch']}"]
    L += ["", "## Files", "", "| file | sha256 |", "|---|---|"]
    L += [f"| `{k}` | `{v}` |" for k, v in written.items()]
    L += ["", f"Grammar: {man['grammar_bytes']:,} B, {man['dfa_states']} states, {man['classes']} character classes, "
          f"Unicode {man['unicode_version']}. Rules: {', '.join(man['rules'])}.", ""]
    return "\n".join(L)


# ============================================================================================================
def add_parsers(sub, common):
    p = sub.add_parser("e2e", help="byte-identity through real OpenWebUI + the compactor (both versions)")
    p.add_argument("--webui-db", required=True, help="a webui.db backup (read-only; a trimmed copy is used)")
    p.add_argument("--compactor-ref", default="v3.1.9", help="git ref of the compactor source to run as well as "
                   "the image's own (empty = only the image's)")
    common(p)
    p = sub.add_parser("diff", help="what changes between two revisions")
    p.add_argument("old")
    p.add_argument("new")
    p.add_argument("--corpus")
    p.add_argument("--show-forms", action="store_true", help="print the changed word forms (terminal only)")
    p = sub.add_parser("release", help="write the paste value, check script, prompt line and report")
    p.add_argument("--out-dir", default=str(W.HOME))
    p.add_argument("--notes", help="a markdown file to put at the top of the report")
    p.add_argument("--allow-no-corpus", action="store_true")
    p.add_argument("--allow-no-e2e", action="store_true")
    p.add_argument("--force", action="store_true", help="overwrite an existing vN file with different content")
    common(p)


def dispatch(name):
    return {"e2e": cmd_e2e, "diff": cmd_diff, "release": cmd_release}[name]
