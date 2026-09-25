"""wordban subcommands e2e, diff, release and cycle (build, verify and tokenizer live in wordban.py)."""
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import types
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
    results = {"grammar_sha256": man["grammar_sha256"], "tool_sha256": W.tool_sha(), "image": args.image,
              "image_tag": W.DEFAULT_IMAGE_TAG if args.image == W.DEFAULT_IMAGE else
                          "unknown (--image != DEFAULT_IMAGE)", "runs": []}
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
        probes = W.build_probes(spec, bl)          # from the banlist's own test lists - never stale
        (priv / "probes.json").write_text(json.dumps(probes), encoding="utf-8")
        # rev 7 round 2 (owner): cycle's e2e step runs only prod's compactor by default (not also the image's own
        # bundled one) - `--e2e-both` (cycle) / `skip_image_compactor=False` (this function, default) restores
        # the old "run both" behaviour. The standalone `e2e` command still defaults to both, unchanged.
        compactors = [] if getattr(args, "skip_image_compactor", False) else [("image", "/opt/compactor")]
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
            mism = [p for r in gr for p in r["vllm"].get("probes", []) if p["accepted"] != p["expected"]]
            checks.append(("probe sentences judged as expected by the compiled grammar", bool(gr) and not mism))
            # rev 7 (Opus review follow-up): a run that silently returned fewer probe results than were sent (a
            # stub crash mid-way, e.g.) must FAIL - `mism` alone can't see that, since zero results is zero
            # mismatches too.
            short = [r for r in gr if len(r["vllm"].get("probes", [])) < len(probes)]
            checks.append(("every probe sent got a result back (none silently dropped)", bool(gr) and not short))
            checks.append(("no bad_words left in the forwarded body", bool(gr) and all(r["bad_words"] is None for r in gr)))
            for name, good in checks:
                W.say(f"  [{'PASS' if good else 'FAIL'}] {name}")
                ok = ok and bool(good)
            for p in mism[:20]:
                t = p["text"]
                ctx = t if len(t) <= 84 else t[:40] + " … " + t[-40:]     # ~40 chars either side, same width as
                W.say(f"    MISMATCH text={ctx!r} expected={p['expected']} accepted={p['accepted']}")  # blocked_words
            if short:
                W.say(f"    DROPPED: {len(probes)} probes sent, as few as {min(len(r['vllm'].get('probes', [])) for r in gr)} came back")
            if gr:
                W.say(f"  compile inside the stub: {[r['vllm'].get('compile_s') for r in gr]} s; "
                      f"grammar {gr[0]['grammar_chars']:,} chars")
            if not all(g for _, g in checks):
                W.say("  (raw) " + out[-3000:])
            run["checks"] = [{"check": n, "pass": bool(g)} for n, g in checks]
            # rev 7 (Opus review follow-up): counts and mismatches only in the persisted stamp - not a full dump
            # of every probe's result (these are synthetic banlist strings, not her data, but "counts only" is
            # the posture everywhere else in this tool, so it applies here too on explicit instruction).
            for r in run["requests"]:
                pr = (r.get("vllm") or {}).get("probes")
                if pr is not None:
                    r["vllm"]["probes"] = {"sent": len(probes), "count": len(pr),
                                           "mismatches": [p for p in pr if p["accepted"] != p["expected"]]}
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
def resolve_release_banlist(rev, out_dir, build_dir=None):
    """the source banlist.yaml behind revision `rev`'s RELEASED grammar, as a ref `diff`'s load_side() accepts
    (`git:<sha>:scripts/wordban/banlist.yaml`, or a plain path) - for `cycle`'s diff-against-the-last-release
    step. `release` never copies banlist.yaml into $WORDBAN_HOME (see cmd_release: only the value, the grammar,
    the check script and the report), only release.vN.json's `banlist_sha256`, so this reconstructs it:
      1. a same-revision build dir still holding that exact banlist.yaml (the caller's `build_dir`, else the
         conventional `<out_dir>/build<rev>` and, last, the default `<out_dir>/build` - both real layouts seen
         in this repo's history);
      2. failing that, the git commit whose scripts/wordban/banlist.yaml blob hashes to that banlist_sha256
         (walked with `git log --follow`, oldest-history-safe);
      3. failing both, a loud, explicit refusal - never a silent guess.
    `rev` may be an exact revision (needs `out_dir/release.v<rev>.json`) or, if that is missing, the search
    falls back to the newest `release.vN.json` with N < rev."""
    out_dir = Path(out_dir).expanduser()
    rp = out_dir / f"release.v{rev}.json"
    if not rp.exists():
        cands = sorted((int(m.group(1)), p) for p in out_dir.glob("release.v*.json")
                       for m in [re.search(r"release\.v(\d+)\.json$", p.name)] if m and int(m.group(1)) < rev)
        if not cands:
            raise ValueError(f"resolve_release_banlist: no release.v{rev}.json, and no release.vN.json with "
                             f"N < {rev}, in {out_dir}")
        rev, rp = cands[-1]
    rel = json.loads(rp.read_text(encoding="utf-8"))
    want = rel.get("banlist_sha256")
    if not want:
        raise ValueError(f"resolve_release_banlist: {rp} has no banlist_sha256")
    for cand in (build_dir, out_dir / f"build{rev}", out_dir / "build"):
        if not cand:
            continue
        bp = Path(cand).expanduser() / "banlist.yaml"
        if bp.exists() and core.sha256_bytes(bp.read_bytes()) == want:
            return str(bp)
    log = subprocess.run(["git", "-C", str(REPO), "-c", "safe.directory=*", "log", "--format=%H", "--follow",
                          "--", "scripts/wordban/banlist.yaml"], capture_output=True, text=True)
    for sha in log.stdout.splitlines():
        blob = subprocess.run(["git", "-C", str(REPO), "-c", "safe.directory=*", "show",
                               f"{sha}:scripts/wordban/banlist.yaml"], capture_output=True)
        if blob.returncode == 0 and core.sha256_bytes(blob.stdout) == want:
            return f"git:{sha}:scripts/wordban/banlist.yaml"
    raise ValueError(f"resolve_release_banlist: REFUSED - no build dir and no git commit has a "
                     f"scripts/wordban/banlist.yaml matching banlist_sha256 {want[:16]}… "
                     f"(release.v{rev}.json). Pass the build dir that still holds that revision's banlist.yaml "
                     "copy, or find the commit by hand.")


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
    """the words the grammar stops, as (word, context) pairs: run it, mask what it stops at, run again. A refused
    invisible character (soft hyphen, zero-width space ...) is reported and removed on its own."""
    out = []
    wc = lambda ch: ch.isalnum() or unicodedata.category(ch) in core.MARK_CATS or ch in "_'’-"
    for _ in range(limit):
        s, at = A.run(text)
        if at is None:
            if s is not None and A.acc[s]:
                return out
            at = len(text) - 1
        i = at
        if unicodedata.category(text[i]) == "Cf":
            out.append((f"U+{ord(text[i]):04X}", text[max(0, i - 40):i + 1]))
            text = text[:i] + text[i + 1:]
            continue
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
        out.append((text[a:b], text[max(0, a - 40):b]))
        text = text[:a] + " " * (b - a) + text[b:]
    return out


def default_base_revision(bl, rev):
    """rev 7 FINAL (coordinator): default the diff/notes comparison base to `live_revision:` when the banlist
    declares one - the revision actually PASTED on prod (which can lag `revision:` by any number of built-but-
    never-pasted ones, e.g. v6) - falling back to the previous behaviour (rev - 1) when it is absent."""
    live = bl.get("live_revision")
    return int(live) if live is not None else rev - 1


def cmd_diff(args):
    if args.old is None:
        new_bl = core.load_yaml(Path(args.new).expanduser()) if Path(args.new).suffix in (".yaml", ".yml") else None
        if new_bl is None:
            sys.exit("REFUSED: `old` was omitted, and `new` is not a banlist.yaml path, so live_revision: cannot "
                     "be resolved from it - pass `old` explicitly")
        base_rev = default_base_revision(new_bl, new_bl.get("revision"))
        args.old = resolve_release_banlist(base_rev, args.out_dir)
        W.say(f"(old omitted: defaulting to live_revision {base_rev} -> {args.old})")
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
            cm = W.load_corpus_mode(db, args)
            msgs = cm["msgs"]
            db.unlink()
            ref = core.Reference(sb or sa) if (sb or sa) else None
            fam_of = {r["id"]: f for f, r in (sb or sa).rules} if (sb or sa) else {}
            newly, freed = {}, {}
            forms_n, forms_f = {}, {}
            t0 = time.time()
            for m in msgs:
                wa = blocked_words(A, m["t"])
                wb = blocked_words(B, m["t"])
                ca, cb, ctx = {}, {}, {}
                for w, c in wa:
                    ca[w] = ca.get(w, 0) + 1
                    ctx.setdefault(w, c)
                for w, c in wb:
                    cb[w] = cb.get(w, 0) + 1
                    ctx.setdefault(w, c)
                for w in set(ca) | set(cb):
                    d = cb.get(w, 0) - ca.get(w, 0)
                    if not d:
                        continue
                    fam = "other"
                    if ref:
                        h = ref.hits(ctx[w]) or ref.hits(w)   # the word in its context (a phrase needs it)
                        fam = fam_of.get(h[-1][2], h[-1][2]) if h else "other"
                    key = (m["role"], fam)
                    if d > 0:
                        newly[key] = newly.get(key, 0) + d
                        forms_n[w] = forms_n.get(w, 0) + d
                    else:
                        freed[key] = freed.get(key, 0) - d
                        forms_f[w] = forms_f.get(w, 0) - d
            W.say(f"  her corpus ({cm['desc']}: {len(msgs)} messages, {time.time() - t0:.0f}s), occurrences whose "
                  "verdict changes (counts only):")
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
    if e2e is not None and e2e.get("tool_sha256") != W.tool_sha():
        sys.exit("REFUSED: the e2e stamp was written by different tool code (edit, then e2e again)")
    bl = core.load_yaml(build_dir / "banlist.yaml")
    rev = man["revision"]
    # rev 7 hostile-review fix 7: rollback is now keyed by revision (`rollback: {N: "..."}`, was a single
    # string) - refuse BEFORE writing anything if the CURRENT revision has no entry.
    rollback_map = bl.get("rollback") or {}
    rollback_map = {int(k): v for k, v in rollback_map.items()}
    if rev not in rollback_map:
        sys.exit(f"REFUSED: banlist.yaml has no rollback[{rev}] entry for the current revision - add one "
                  "(e.g. rollback: {" + f"{rev}" + ': "vN, which ..."}) before releasing')
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
    # rev 7 hostile-review fix 7: rollback is keyed by revision now (`rollback: {N: "..."}`) - cmd_release
    # already refused before writing anything if rev has no entry, so this is just the lookup for that revision.
    rollback_note = {int(k): v for k, v in (bl.get("rollback") or {}).items()}.get(rev)
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
          # rev 7 FINAL (hostile-review fix 1, HIGH): `rollback[rev]` (hand-written per revision, KEYED by
          # revision since fix 7) is now the PRIMARY, TRUSTED statement of the rollback target, not a mere
          # "(or ...)" caveat tacked onto `prev[0]` - `prev[0]` is just the highest-numbered value FILE FOUND ON
          # DISK under --out-dir, which is NOT necessarily what is actually live on the model (v6 was built and
          # released as a file but never pasted - prod stayed on v5 - so a purely file-presence-based `prev[0]`
          # would have wrongly named v6, a file nobody ever pasted, as the rollback target). `prev[0]` is used
          # only as a fallback "paste the last release file found" when no `rollback[rev]` note exists at all.
          ("Paste " + rollback_note + " into the row, or delete the row."
           if rollback_note else
           (f"Paste `word-ban-structured_outputs.value.v{prev[0]}.txt` into the row, or delete the row."
            if prev else "Delete the `structured_outputs` row."))
          + (" Never " + ", ".join(f"v{k}" for k in sorted(withdrawn)) + "." if withdrawn else ""), ""]
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
# cycle
# ============================================================================================================
def run_step(rev, name, fn):
    """run one cycle step, its full output tee'd to ~/zl-ops/word-ban/work/logs/cycle-rev<N>-<step>.txt as well
    as the terminal; returns (rc, seconds, log path). `fn` takes no arguments and returns an int (0 = ok).

    verify's image half and e2e's docker runs use `subprocess.call`/`subprocess.run` with no explicit stdout, so
    their output goes straight to file descriptor 1/2, bypassing a Python-level `sys.stdout` swap entirely (a
    reassigned sys.stdout object is invisible to an inherited OS fd). To log the FULL output - not just this
    process's own print()s - fd 1 and 2 are duplicated to a pipe for the duration of the step, with a reader
    thread copying every byte to both the real terminal and the log file."""
    logp = W.HOME / "work" / "logs" / f"cycle-rev{rev}-{name}.txt"
    logp.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    sys.stdout.flush()
    sys.stderr.flush()
    real_out, real_err = os.dup(1), os.dup(2)
    r, w = os.pipe()
    os.dup2(w, 1)
    os.dup2(w, 2)
    os.close(w)

    def pump(lf):
        while True:
            chunk = os.read(r, 65536)
            if not chunk:
                return
            os.write(real_out, chunk)
            lf.write(chunk)
            lf.flush()
    with open(logp, "wb") as lf:
        th = threading.Thread(target=pump, args=(lf,), daemon=True)
        th.start()
        try:
            rc = fn()
        except SystemExit as e:                     # sys.exit(...) inside a called cmd_* - treat like a return code
            # rev 7 hostile-review fix 4: e.code may be None (success), an int, or a STRING/other message
            # (sys.exit("boom") sets .code to that string, printed to stderr by the interpreter - not an int).
            # Coerce it into a failing integer rc instead of feeding a non-numeric value to int(rc or 0) below.
            if e.code is None or e.code == 0:
                rc = 0
            elif isinstance(e.code, int):
                rc = e.code
            else:
                print(f"STEP FAILED (sys.exit message): {e.code!r}")
                rc = 1
        except Exception as e:                       # noqa: BLE001 - report, don't crash the tee/log bookkeeping
            print(f"EXCEPTION: {e!r}")
            rc = 1
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(real_out, 1)   # fd 1/2 point back at the terminal; that was the pipe's last writer, so its
            os.dup2(real_err, 2)   # read end now drains to EOF - but real_out/real_err must stay open until the
            th.join(timeout=5)     # pump thread (which still uses them) has actually finished draining it
            os.close(real_out)
            os.close(real_err)
            os.close(r)
    return int(rc or 0), time.time() - t0, logp


def diff_rules_data(sa, sb):
    """(kind, rule id, family, detail) for every + / - / ~ rule between two built Specs - the same comparison
    cmd_diff prints, factored out so cycle's notes draft can render it without re-parsing printed text."""
    ra = {r["id"]: (f, r) for f, r in sa.rules} if sa else {}
    rb = {r["id"]: (f, r) for f, r in sb.rules} if sb else {}
    out = []
    for k in rb:
        if k not in ra:
            out.append(("+", k, rb[k][0], rb[k][1]))
    for k in ra:
        if k not in rb:
            out.append(("-", k, ra[k][0], ra[k][1]))
        elif ra[k][1] != rb[k][1]:
            out.append(("~", k, ra[k][0], (ra[k][1], rb[k][1])))
    return out


def diff_tests_data(sa, sb):
    """{list name -> (added, removed)} for tests.must_block/must_pass/collateral/gaps between two banlists.
    These are synthetic banlist entries, never corpus data, so printing them is fine (see README, Privacy)."""
    out = {}
    for name in ("must_block", "must_pass", "collateral", "gaps"):
        a = set(W.flat((sa.bl.get("tests") or {}).get(name))) if sa else set()
        b = set(W.flat((sb.bl.get("tests") or {}).get(name))) if sb else set()
        added, removed = sorted(b - a), sorted(a - b)
        if added or removed:
            out[name] = (added, removed)
    return out


def diff_struct_data(sa, sb):
    """rev 7 hostile-review fix 6: structural fields the rule/test diff above doesn't touch - separator sets
    (added/removed set names, and added/removed characters within a set that exists on both sides), folding.*
    (confusables, lookalikes, forbidden_chars and the rest), rollback (keyed by revision), check.prompt/allowed (the
    pod-check prompt) and funnels (min_width, the documented list). [(label, old, new)] for anything that
    differs; both sides may be None (no base to compare against)."""
    out = []
    ba = sa.bl if sa else {}
    bb = sb.bl if sb else {}
    sepa, sepb = (ba.get("separators") or {}), (bb.get("separators") or {})
    for name in sorted(set(sepa) | set(sepb)):
        va, vb = set(sepa.get(name) or []), set(sepb.get(name) or [])
        if name not in sepa:
            out.append((f"separators.{name}", None, sorted(vb)))
        elif name not in sepb:
            out.append((f"separators.{name}", sorted(va), None))
        elif va != vb:
            out.append((f"separators.{name}", sorted(va), sorted(vb)))
    fa, fb = (ba.get("folding") or {}), (bb.get("folding") or {})
    for key in sorted(set(fa) | set(fb)):
        if fa.get(key) != fb.get(key):
            out.append((f"folding.{key}", fa.get(key), fb.get(key)))
    if ba.get("rollback") != bb.get("rollback"):
        out.append(("rollback", ba.get("rollback"), bb.get("rollback")))
    ca, cb = (ba.get("check") or {}), (bb.get("check") or {})
    for key in ("prompt", "allowed"):
        if ca.get(key) != cb.get(key):
            out.append((f"check.{key}", ca.get(key), cb.get(key)))
    fua, fub = (ba.get("funnels") or {}), (bb.get("funnels") or {})
    if fua.get("min_width") != fub.get("min_width"):
        out.append(("funnels.min_width", fua.get("min_width"), fub.get("min_width")))
    if fua.get("documented") != fub.get("documented"):
        out.append(("funnels.documented", fua.get("documented"), fub.get("documented")))
    return out


def render_rule_diff_md(sa, sb):
    L = []
    for kind, rid, fam, detail in diff_rules_data(sa, sb):
        if kind == "~":
            old, new = detail
            L.append(f"- `~` rule `{rid}` ({fam}): `{json.dumps(old, ensure_ascii=False)}` -> "
                     f"`{json.dumps(new, ensure_ascii=False)}`")
        else:
            L.append(f"- `{kind}` rule `{rid}` ({fam}): `{json.dumps(detail, ensure_ascii=False)}`")
    for name, (added, removed) in diff_tests_data(sa, sb).items():
        L.append(f"- `tests.{name}`: {len(added)} added, {len(removed)} removed"
                 + (f"; added: {added}" if added else "") + (f"; removed: {removed}" if removed else ""))
    for label, old, new in diff_struct_data(sa, sb):
        L.append(f"- `~` {label}: `{json.dumps(old, ensure_ascii=False)}` -> `{json.dumps(new, ensure_ascii=False)}`")
    return L or ["- (no rule, test-list or structural change detected)"]


def _newly_blocked_total(diff_text):
    """sum of the {(role, family): count} dict printed after "newly BLOCKED:" in a diff's corpus section -
    counts only, never her data (the diff log itself never prints text, only these dict counts)."""
    m = re.search(r"newly BLOCKED:\s*(\{.*\})", diff_text)
    if not m:
        return None
    try:
        d = eval(m.group(1), {"__builtins__": {}})              # a literal {(role,fam): int, ...} dict, trusted
    except Exception:                                            # noqa: BLE001
        return None
    return sum(d.values()) if isinstance(d, dict) else None


def write_notes(rev, base_ref, new_bl_path, verify_line, e2e_line, diff_text, out_path, bl=None, verify_stamp=None,
                base_label=None):
    """the cycle notes draft (cycle step e). Two parts: a PLAIN-ENGLISH summary for the owner up front (rev 7
    FINAL hostile-review fix 5 - what changed, the newly-blocked count on the recent corpus, the prod
    first-compile wait, the paste steps, the rollback - deliberately NO one-off operational reminders here, the
    tool does not carry those), then the mechanically-generated detail below it (from
    diff_rules_data/diff_tests_data/diff_struct_data, the same comparison `diff` itself prints - a strong DRAFT,
    not a drift-proof guarantee: it can still miss a banlist field none of those
    functions compare). Meant to be read and edited by a human before it is passed to `release --notes`."""
    sb, _eb = core.build(new_bl_path)
    sa = None
    la = base_ref
    if base_ref:
        try:
            la, _eb, sa = load_side(base_ref)
        except Exception as e:                      # noqa: BLE001 - a notes draft without "what changed" is still useful
            print(f"(no structural diff: {e})")
    la = base_label or la
    bl = bl or core.load_yaml(new_bl_path)
    rollback_note = {int(k): v for k, v in (bl.get("rollback") or {}).items()}.get(rev)
    compile_s = ((verify_stamp or {}).get("image_facts") or {}).get("facts", {}).get("first_compile_s")
    newly = _newly_blocked_total(diff_text)
    rule_diff = render_rule_diff_md(sa, sb)
    families_touched = sorted({m.group(1) for line in rule_diff for m in [re.search(r"\((\w+)\):", line)] if m}
                              ) if sa else []
    S = [f"# Word ban, revision {rev} - notes for the owner", "", "## Plain English", ""]
    S.append(f"**What changed vs {la or 'the previous release'}:** " + (
        f"{len(rule_diff)} rule/test/structural change(s)" + (f" touching {', '.join(families_touched)}"
        if families_touched else "") + " - see \"What changed (detail)\" below for the exact list."
        if rule_diff and rule_diff != ["- (no rule, test-list or structural change detected)"]
        else "no rule/test/structural changes detected vs that revision."))
    S.append("")
    S.append("**Newly blocked, on the recent corpus (counts only, no text):** "
             + (f"{newly} additional occurrence(s) her current chat's recent messages would now trigger that "
                f"{la or 'the previous revision'} did not." if newly is not None
                else "not found in the diff log (see the Corpus diff section below)."))
    S.append("")
    S.append("**First compile (once per vLLM restart):** " + (
        f"~{compile_s:.0f}s measured on the build machine (CPU); prod has historically compiled at roughly "
        "0.73x that (rev 6: 63s prod vs 86s local), so expect roughly 70-100s on the pod, once, the first time "
        "a request needs the new grammar. Neither hop (compactor v3.1.9, OpenWebUI 0.11.4) enforces a read "
        "timeout, so a slow compile is slow but will not fail the request." if compile_s
        else "not available (verify stamp missing image_facts)."))
    S.append("")
    S.append("**Paste steps:** replace `structured_outputs` with `word-ban-structured_outputs.value.txt` and the "
             "system-prompt line with `word-ban-prompt-line.v" + str(rev) + ".txt` (Admin Panel -> Settings -> "
             "Models -> the model -> Advanced Params), then run `word-ban-check.py` on the pod - see `release`'s "
             "full report (`word-ban.md`) for the exact numbered steps.")
    S.append("")
    S.append("**Rollback:** " + (rollback_note or f"paste the previous release's value file (v{rev - 1}, if it "
             "was ever actually pasted - check what is LIVE, not just what was built)") + ".")
    S.append("")
    L = S + ["## What changed (detail)" + (f" (vs `{la}`)" if la else ""), ""] + rule_diff + [""]
    L += ["## Corpus diff (counts only)", "", "```", diff_text.strip(), "```", ""]
    L += ["## Verify", "", verify_line, ""]
    L += ["## E2E", "", e2e_line, ""]
    out_path.write_text("\n".join(L), encoding="utf-8")
    return out_path


def cmd_cycle(args):
    """build -> verify (full, with corpus) -> e2e (against HEAD) -> diff (against the last release) -> a notes
    draft (a strong draft of "what changed", from the rule/test-list/structural diff - not a drift-proof
    guarantee; see write_notes). Stops at the first failing step. Never releases (memory.md rule 4: a hostile
    review comes first)."""
    bl = core.load_yaml(W.HERE / "banlist.yaml")
    rev = bl.get("revision")
    out_dir = Path(args.out_dir).expanduser() if getattr(args, "out_dir", None) else W.HOME
    build_dir = Path(args.build_dir).expanduser()
    work = W.HOME / "work"
    work.mkdir(parents=True, exist_ok=True)
    say = W.say
    say(f"cycle: revision {rev}, build dir {build_dir}")
    timings = {}
    step_results = {}

    def build_step():
        a = types.SimpleNamespace(banlist=str(W.HERE / "banlist.yaml"), build_dir=str(build_dir), here=False,
                                  image=args.image, tokenizer=args.tokenizer)
        return W.cmd_build(a)

    def verify_step():
        a = types.SimpleNamespace(build_dir=str(build_dir), corpus=args.corpus, no_image=False, fuzz=40000,
                                  walk_steps=120000, image=args.image, tokenizer=args.tokenizer,
                                  recent=args.recent, full_corpus=args.full_corpus)
        return W.cmd_verify(a)

    def e2e_step():
        # rev 7 (owner, 2026-09-24): production runs v3.1.9 (not HEAD) - cycle's e2e step runs against the same
        # compactor ref e2e itself defaults to (v3.1.9), so a work-in-progress wordban change is checked against
        # what prod actually runs, not against unrelated in-flight compactor work on this branch.
        # rev 7 round 2 (owner): by default this ALSO skips the image's own bundled compactor - only prod's -
        # unless --e2e-both is given.
        a = types.SimpleNamespace(build_dir=str(build_dir), webui_db=args.corpus, compactor_ref="v3.1.9",
                                  image=args.image, tokenizer=args.tokenizer,
                                  skip_image_compactor=not args.e2e_both)
        return cmd_e2e(a)

    def diff_step():
        # rev 7 FINAL (coordinator): --base-release, if given, always wins; otherwise default to the banlist's
        # own `live_revision:` (what is actually pasted on prod) via default_base_revision(), falling back to
        # rev-1 only when live_revision: is absent - e.g. v6 was built but never pasted, so a real release
        # still on v5 must diff against v5, not v6.
        base_rev = args.base_release or default_base_revision(bl, rev)
        try:
            old = resolve_release_banlist(base_rev, out_dir)
        except ValueError as e:
            print(str(e))
            return 1
        a = types.SimpleNamespace(old=old, new=str(build_dir / "banlist.yaml"), corpus=args.corpus, show_forms=False,
                                  recent=args.recent, full_corpus=args.full_corpus)
        step_results["diff_old"] = old
        step_results["diff_old_label"] = f"v{base_rev}"
        return cmd_diff(a)

    def notes_step():
        vp = build_dir / W.STAMP
        v = json.loads(vp.read_text(encoding="utf-8")) if vp.exists() else {}
        n_f = sum(1 for r in v.get("rows", []) if not r["pass"])
        verify_line = (f"VERIFY {'PASSED' if v.get('pass') else 'FAILED'}: "
                       f"{len(v.get('rows', [])) - n_f}/{len(v.get('rows', []))} checks") if v else "VERIFY: not run"
        ep = build_dir / W.E2E_STAMP
        e = json.loads(ep.read_text(encoding="utf-8")) if (ep.exists() and not args.no_e2e) else None
        e2e_line = (f"E2E {'PASSED' if e.get('pass') else 'FAILED'}") if e is not None else \
            ("E2E: skipped (--no-e2e)" if args.no_e2e else "E2E: not run")
        diff_log = W.HOME / "work" / "logs" / f"cycle-rev{rev}-diff.txt"     # step (d)'s own tee'd log
        diff_full = diff_log.read_text(encoding="utf-8") if diff_log.exists() else ""
        i = diff_full.find("her corpus (")
        diff_text = diff_full[i:] if i >= 0 else diff_full
        out_path = work / f"notes-rev{rev}.md"
        write_notes(rev, step_results.get("diff_old"), build_dir / "banlist.yaml", verify_line, e2e_line,
                   diff_text, out_path, bl=bl, verify_stamp=v, base_label=step_results.get("diff_old_label"))
        step_results["notes_path"] = out_path
        step_results["verify_line"] = verify_line
        step_results["e2e_line"] = e2e_line
        print(f"wrote {out_path}")
        return 0
    steps = [("build", build_step), ("verify", verify_step)]
    if not args.no_e2e:
        steps.append(("e2e", e2e_step))
    steps += [("diff", diff_step), ("notes", notes_step)]
    for name, fn in steps:
        say(f"== cycle step: {name}")
        rc, dt, logp = run_step(rev, name, fn)
        timings[name] = dt
        say(f"   {name}: {'ok' if rc == 0 else 'FAILED (rc=' + str(rc) + ')'} in {dt:.1f}s - log {logp}")
        if rc != 0:
            say(f"\ncycle STOPPED at step {name!r} (rc={rc}). Fix the underlying cause (never weaken a check to "
                f"make this pass) and re-run cycle. Full output: {logp}")
            return 1
    say("\n== cycle summary")
    say(f"  revision {rev}, build dir {build_dir}")
    for name, dt in timings.items():
        say(f"  step {name}: {dt:.1f}s")
    say(f"  notes draft: {step_results.get('notes_path')}")
    say(f"  {step_results.get('verify_line', '')}")
    if args.no_e2e:
        say("  e2e: SKIPPED (--no-e2e) - release still needs a passing e2e stamp, or --allow-no-e2e")
    else:
        say(f"  {step_results.get('e2e_line', '')}")
    say(f"\nNext: wordban.py release --build-dir {build_dir} --notes {step_results.get('notes_path')}")
    say("hostile review required before release (memory.md rule 4)")
    return 0


# ============================================================================================================
def add_parsers(sub, common):
    p = sub.add_parser("e2e", help="byte-identity through real OpenWebUI + the compactor (both versions)")
    p.add_argument("--webui-db", required=True, help="a webui.db backup (read-only; a trimmed copy is used)")
    p.add_argument("--compactor-ref", default="v3.1.9", help="git ref of the compactor source to run as well as "
                   "the image's own (empty = only the image's)")
    common(p)
    p = sub.add_parser("diff", help="what changes between two revisions")
    p.add_argument("old", nargs="?", help="omit to default to the NEW banlist's own `live_revision:` (what is "
                   "actually pasted on prod), resolved via the OUT_DIR release files; falls back to `new`'s "
                   "revision - 1 if `live_revision:` is absent")
    p.add_argument("new")
    p.add_argument("--out-dir", default=str(W.HOME), help="where release.vN.json / buildN live, to resolve "
                   "`old` when it is omitted (defaulting to live_revision)")
    p.add_argument("--corpus")
    p.add_argument("--recent", type=int, default=200, help="with --corpus: only the most recently active chat's "
                   "last N messages (owner decision, rev 7 round 2; default 200)")
    p.add_argument("--full-corpus", action="store_true", help="with --corpus: scan every chat's whole current "
                   "branch instead (the old default)")
    p.add_argument("--show-forms", action="store_true", help="print the changed word forms (terminal only)")
    p = sub.add_parser("release", help="write the paste value, check script, prompt line and report")
    p.add_argument("--out-dir", default=str(W.HOME))
    p.add_argument("--notes", help="a markdown file to put at the top of the report")
    p.add_argument("--allow-no-corpus", action="store_true")
    p.add_argument("--allow-no-e2e", action="store_true")
    p.add_argument("--force", action="store_true", help="overwrite an existing vN file with different content")
    common(p)
    p = sub.add_parser("cycle", help="build -> verify -> e2e -> diff -> a notes draft, in one run")
    p.add_argument("--corpus", required=True, help="a webui.db backup (read-only; copied privately by each step)")
    p.add_argument("--base-release", type=int, help="revision to diff against (default: the newest below this one)")
    p.add_argument("--no-e2e", action="store_true", help="skip the e2e step (release will still need it, or "
                   "--allow-no-e2e)")
    p.add_argument("--recent", type=int, default=200, help="with --corpus: only the most recently active chat's "
                   "last N messages, for verify's and diff's corpus checks (owner decision, rev 7 round 2; "
                   "default 200)")
    p.add_argument("--full-corpus", action="store_true", help="scan every chat's whole current branch instead "
                   "(the old default)")
    p.add_argument("--e2e-both", action="store_true", help="also run the image's own bundled compactor in the "
                   "e2e step (default: prod's compactor only, --compactor-ref v3.1.9)")
    p.add_argument("--out-dir", default=str(W.HOME), help="where release.vN.json / build<N> live, for the diff step")
    common(p)


def dispatch(name):
    return {"e2e": cmd_e2e, "diff": cmd_diff, "release": cmd_release, "cycle": cmd_cycle}[name]
