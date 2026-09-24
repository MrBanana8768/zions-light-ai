#!/usr/bin/env python3
"""wordban: the word ban on the model's output, as a reproducible tool.

    wordban.py build    [--banlist banlist.yaml] [--build-dir DIR]
    wordban.py verify   [--build-dir DIR] [--corpus webui.db] [--image IMG] [--tokenizer DIR] [--no-image]
    wordban.py e2e      [--build-dir DIR] --webui-db webui.db [--compactor-ref v3.1.9]
    wordban.py diff     OLD NEW [--corpus webui.db] [--show-forms]
    wordban.py release  [--build-dir DIR] [--out-dir ~/zl-ops/word-ban] [--notes FILE]
    wordban.py tokenizer [--tokenizer DIR]

See README.md in this directory. The grammar is an xgrammar EBNF grammar that goes into the model's
`structured_outputs` custom parameter in OpenWebUI; vLLM then refuses, token by token, any continuation that
would spell a banned word. Nothing here touches the pod: every check runs locally (CPU), and `verify` / `e2e`
run vLLM 0.19's own code from the published image.

PRIVACY: `--corpus` / `--webui-db` are copied (read-only source) into a private temporary directory that is
deleted when the command ends. Only counts are printed. Nothing from them is ever written to the repo."""
import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import wb_core as core  # noqa: E402

DEFAULT_IMAGE = "angreg/zions-light-ai@sha256:c1295894dd585784611c6833b1d4c396880ac8723e6b9b46531a5aa846cb8a65"
HOME = Path(os.environ.get("WORDBAN_HOME", "~/zl-ops/word-ban")).expanduser()
TOKENIZER_REPO = "coder3101/Cydonia-24B-v4.3-vision-heretic"
TOKENIZER_REV = "c1e236fb254bec28003e40da1f7e7a0bd426945f"
TOKENIZER_FILES = ["tekken.json", "params.json", "config.json", "generation_config.json"]
TEKKEN_SHA_PREFIX = "6e2501687ccd"
GRAMMAR = "wb_ban.ebnf"
VALUE = "structured_outputs.value.txt"
MANIFEST = "manifest.json"
STAMP = "verify-stamp.json"
E2E_STAMP = "e2e-stamp.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tool_sha():
    """one sha256 over the tool's own code: a verify stamp is only trusted by the code that wrote it"""
    h = hashlib.sha256()
    for p in sorted(list(HERE.glob("*.py")) + list(HERE.glob("e2e/*"))):
        if p.is_file():
            h.update(p.relative_to(HERE).as_posix().encode() + b"\x00" + p.read_bytes() + b"\x00")
    return h.hexdigest()


def say(*a):
    print(*a, flush=True)


# ============================================================================================================
# build
# ============================================================================================================
def in_image_python(args, argv, mounts):
    """re-run this script inside the image (its vLLM venv has the pinned Unicode version)."""
    cmd = ["docker", "run", "--rm", "--name", f"wb-{argv[0]}-{os.getpid()}", "--user", f"{os.getuid()}:{os.getgid()}",
           "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "HOME=/tmp", "-v", f"{HERE}:/wb:ro"]
    for src, dst, mode in mounts:
        cmd += ["-v", f"{src}:{dst}:{mode}"]
    cmd += ["--entrypoint", "/opt/vllm-venv/bin/python", args.image, "/wb/wordban.py"] + argv
    return subprocess.call(cmd)


def cmd_build(args):
    bl_path = Path(args.banlist).resolve()
    bl = core.load_yaml(bl_path)
    want = str(bl.get("unicode_version", ""))
    out = Path(args.build_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    if want and unicodedata.unidata_version != want and not args.here:
        say(f"local Python has Unicode {unicodedata.unidata_version}, the banlist pins {want}: building inside the image")
        return in_image_python(args, ["build", "--here", "--banlist", "/bl/" + bl_path.name, "--build-dir", "/out"],
                               [(bl_path.parent, "/bl", "ro"), (out, "/out", "rw")])
    t0 = time.time()
    try:
        spec, ebnf = core.build(bl_path)
    except ValueError as e:
        say(f"REFUSED: {e}")
        return 1
    value = json.dumps({"grammar": ebnf}, ensure_ascii=True)
    for f in (GRAMMAR, VALUE, STAMP, E2E_STAMP):
        (out / f).unlink(missing_ok=True)
    (out / GRAMMAR).write_text(ebnf, encoding="utf-8")
    (out / VALUE).write_text(value, encoding="utf-8")
    (out / "banlist.yaml").write_bytes(bl_path.read_bytes())
    traps = spec.dfa_trap_states()
    man = {
        "revision": spec.revision, "banlist_sha256": spec.source_sha,
        "grammar_sha256": sha(out / GRAMMAR), "grammar_bytes": len(ebnf.encode()),
        "value_sha256": sha(out / VALUE), "value_bytes": len(value.encode()),
        "dfa_states": spec.dfa_n, "dfa_states_before_minimising": spec.n_raw, "classes": len(spec.classes),
        "rules": [r["id"] for _, r in spec.rules], "unicode_version": unicodedata.unidata_version,
        "python": sys.version.split()[0], "dfa_trap_states": len(traps), "built_s": round(time.time() - t0, 1),
    }
    (out / MANIFEST).write_text(json.dumps(man, indent=1, ensure_ascii=False), encoding="utf-8")
    say(f"built revision {spec.revision}: {spec.dfa_n} states ({spec.n_raw} before minimising), "
        f"{len(spec.classes)} character classes, grammar {man['grammar_bytes']:,} B sha256 {man['grammar_sha256']}")
    say(f"  paste value {out / VALUE}: {man['value_bytes']:,} B sha256 {man['value_sha256']}")
    if traps:
        say(f"  WARNING: {len(traps)} DFA state(s) with no normal exit (a trap) - verify will fail")
    return 0


def load_build(build_dir):
    d = Path(build_dir).expanduser().resolve()
    man = json.loads((d / MANIFEST).read_text(encoding="utf-8"))
    for f, k in ((GRAMMAR, "grammar_sha256"), (VALUE, "value_sha256")):
        if sha(d / f) != man[k]:
            sys.exit(f"{d / f} does not match its manifest (edited after build?) - run build again")
    return d, man


# ============================================================================================================
# verify
# ============================================================================================================
class Results:
    def __init__(self):
        self.rows = []

    def add(self, name, ok, detail=""):
        self.rows.append({"check": name, "pass": bool(ok), "detail": detail})
        say(f"  [{'PASS' if ok else 'FAIL'}] {name}{': ' + detail if detail else ''}")

    @property
    def ok(self):
        return all(r["pass"] for r in self.rows)


def flat(lists):
    """a test list (or list of lists) -> strings; a YAML scalar that did not stay a string (true, no, 1.0 ...) is
    an error in the banlist, not something to guess about"""
    out = []
    for x in lists or []:
        for y in (x if isinstance(x, list) else [x]):
            if not isinstance(y, str):
                sys.exit(f"banlist test entry {y!r} is not a string - quote it in the YAML")
            out.append(y)
    return out


def fuzz_strings(spec, n, seed=11):
    """strings built from pieces of the banned words, the allowed words, accents, separators and marks"""
    rnd = random.Random(seed)
    words = []
    base = {}
    for rid, f in core.derived_must_block(spec):
        base.setdefault(rid, f.lower())
    for _, r in spec.rules:
        w = base.get(r["id"]) or "".join(ch for ch in r.get("stem", r.get("match", "")) if ch.isalpha())
        words.append(w)
        for a in (r.get("allow_words") or []) + (r.get("allow_prefixes") or []) + \
                 ((r.get("context") or {}).get("allow_words") or []):
            words.append(w + a)
    words += ["los ", "white ", "the ", "evangel", "tri", "Michel", "mari", "in", "en", "re", "a", "o", "s", "es",
              "ed", "ing", "Qz", "ruch", "rich", "ruin", "rue", "Los Angel", "LOS ANGELES", "strange", "change",
              "michelangelo", "marigold", "inflam", "enflam", "proinflam", "candle", "pure", "ev", "str", "flamm",
              "flammab", "ruach", "רוח", "ר", "ו", "ח", "αγγελ", "Αγγελ"]
    for e in spec.escapes:
        words += core.Reference._words(e["trie"])
    acc = ("áàâäãåāăąǎạảéèêëēĕėęěẹíìîïīĭįǐóòôöõøōŏőǒọúùûüūŭůűųǔñńņňçćĉċčșşśŝšğĝġģĺļľłďđÀÁÂÄÅÉÈÊËÍÏÓÖØÚÜÑÇĂŐŠ"
           "аеосрухіјАЕОСРНТХΑΕΟΙΝΚο" + "I1|0")
    seps = [" ", "-", "'", "’", "_", ".", "*", "(", ")", "\n", "\t", " ", "  ", "", "", "", "", ",", "!", "1", "0",
            "·", "—", "`", "|", "/", "**", "​", "‌", "‍", "­", "⁠", "\U0001f468‍",
            "‍\U0001f469", "״", "־"]
    marks = ["́", "̈", "̂", "ּ", "ָ", "ׁ"]
    out = []
    for _ in range(n):
        s = []
        for _k in range(rnd.randint(1, 4)):
            w = rnd.choice(words)
            if rnd.random() < 0.5:
                w = w[:rnd.randint(1, len(w))] + (rnd.choice(words) if rnd.random() < 0.3 else "")
            w = "".join((rnd.choice(acc) if (rnd.random() < 0.12 and c.isalpha()) else
                         (c.upper() if rnd.random() < 0.25 else c)) for c in w)
            if rnd.random() < 0.15:
                i = rnd.randint(0, len(w))
                w = w[:i] + rnd.choice(marks) * rnd.randint(1, 5) + w[i:]
            if rnd.random() < 0.15:
                i = rnd.randint(0, len(w))
                w = w[:i] + rnd.choice(seps) + w[i:]
            s.append(w)
            s.append(rnd.choice(seps))
        out.append("".join(s))
    return out


def private_dir():
    base = HOME / "private"
    base.mkdir(parents=True, exist_ok=True)
    os.chmod(base, 0o700)
    return Path(tempfile.mkdtemp(prefix="wordban-", dir=base))


def copy_db(src, dst_dir):
    dst = Path(dst_dir) / "webui.db"
    shutil.copyfile(src, dst)                      # the source is only read
    os.chmod(dst, 0o600)
    return dst


def local_checks(spec_path, build_dir, man, res, fuzz_n=40000):
    """everything that needs no tokenizer: the artifact's own automaton, the test lists, the reference."""
    bl = core.load_yaml(spec_path)
    ebnf = (build_dir / GRAMMAR).read_text(encoding="utf-8")
    if unicodedata.unidata_version == str(bl.get("unicode_version", "")):
        spec, again = core.build(spec_path)
        res.add("artifact = banlist (rebuilt here, byte-identical)", again == ebnf,
                f"sha256 {core.sha256_bytes(again.encode())[:16]} vs {man['grammar_sha256'][:16]}")
    else:
        spec = core.Spec(bl, spec_path.read_bytes())
        say(f"  (rebuild comparison skipped: this Python has Unicode {unicodedata.unidata_version}, "
            f"the banlist pins {bl.get('unicode_version')}; the image run repeats it)")
    ref = core.Reference(spec)
    t0 = time.time()
    A = core.Automaton(ebnf)
    say(f"  automaton read back from the grammar: {A.n} states, {A.n_intervals} code-point intervals "
        f"({time.time() - t0:.1f}s)")
    traps, prev, _ = A.trap_states()
    res.add("no-trap property, character level (every reachable state has an ordinary way to finish)", not traps,
            f"{len(traps)} trap state(s)" + ("; e.g. after " + ", ".join(repr(A.example_prefix(s, prev)) for s in traps[:6])
                                              if traps else f" among {A.n} states"))
    T = bl.get("tests") or {}
    mb = flat(T.get("must_block")) + [f for _, f in core.derived_must_block(spec)]
    mp = flat(T.get("must_pass"))
    bad = [t for t in mb if A.accepts(t)]
    res.add("must-block (character level)", not bad, f"{len(mb) - len(bad)}/{len(mb)} blocked" +
            (f"; ALLOWED: {bad[:10]}" if bad else ""))
    bad = [t for t in mp if not A.accepts(t)]
    res.add("must-pass (character level)", not bad, f"{len(mp) - len(bad)}/{len(mp)} allowed" +
            (f"; BLOCKED: {bad[:10]}" if bad else ""))
    dis = [t for t in mb + mp if ref.blocked(t) == A.accepts(t)]
    res.add("reference definition agrees on every listed case", not dis, f"{len(mb) + len(mp) - len(dis)}/{len(mb) + len(mp)}"
            + (f"; disagree: {dis[:10]}" if dis else ""))
    # the trap regressions: wherever the grammar stops them, a normal continuation must remain
    tl = []
    for t in flat(T.get("trap_regressions")):
        s, at = A.run(t)
        if s is None:
            s, _ = A.run(t[:at])
        tl.append((t, s in traps))
    res.add("trap regressions (angelĂ, Angelø, angeléñÖs ...): normal exit remains", not any(x for _, x in tl),
            f"{len(tl)} prefixes" + (f"; TRAPPED: {[t for t, x in tl if x]}" if any(x for _, x in tl) else ""))
    # combining-mark runs
    cap, caph = spec.mark_cap, spec.mark_cap_hebrew
    mr = []
    if not cap:
        for k in range(0, 7):
            mr.append(("e", k, True, A.accepts("x e" + "́" * k + " y")))
    if cap:
        for base, mk, lim in (("e", "́", cap), ("a", "̈", cap), ("ש", "ָ", caph or cap)):
            for k in range(0, 7):
                want = k <= lim
                got = A.accepts("x " + base + mk * k + " y")
                mr.append((base, k, want, got))
        mixed = A.accepts("x שָּ́ y")                # 2 Hebrew + 1 other = 3 > 2
        mr.append(("mixed", 3, False, mixed))
    bad = [m for m in mr if m[2] != m[3]]
    res.add(f"combining-mark cap (<= {cap} in a row" + (f", <= {caph} Hebrew points" if caph else "") + ")", not bad,
            f"{len(mr)} runs" + (f"; wrong: {bad}" if bad else ""))
    # differential fuzz: the artifact's automaton vs the independent reference
    t0 = time.time()
    fz = fuzz_strings(spec, fuzz_n)
    dis = [t for t in fz if ref.blocked(t) == A.accepts(t)]
    res.add(f"differential fuzz, grammar vs reference definition ({fuzz_n:,} strings)", not dis,
            f"{fuzz_n - len(dis):,}/{fuzz_n:,} agree ({time.time() - t0:.0f}s)" + (f"; e.g. {dis[:8]}" if dis else ""))
    return spec, ref, A, bl


def cmd_verify(args):
    build_dir, man = load_build(args.build_dir)
    spec_path = build_dir / "banlist.yaml"
    res = Results()
    say(f"verify revision {man['revision']}: grammar sha256 {man['grammar_sha256']}")
    say("== local (the grammar's own automaton, the reference definition)")
    spec, ref, A, bl = local_checks(spec_path, build_dir, man, res, fuzz_n=args.fuzz)
    image_part = None
    priv = None
    try:
        if not args.no_image:
            priv = private_dir()
            T = bl.get("tests") or {}
            tests = {
                "must_block": flat(T.get("must_block")) + [f for _, f in core.derived_must_block(spec)],
                "must_pass": flat(T.get("must_pass")), "collateral": flat(T.get("collateral")),
                "gaps": flat(T.get("gaps")), "token_paths": T.get("token_paths") or [],
                "trap_regressions": flat(T.get("trap_regressions")),
                "accent": core.accent_cases(spec, T.get("accent_words") or []),
                "mark_cap": spec.mark_cap, "mark_cap_hebrew": spec.mark_cap_hebrew,
                "funnels": bl.get("funnels") or {},
            }
            rnd = random.Random(7)
            cps = [c for c in range(0x20, 0x110000) if not (0xD800 <= c <= 0xDFFF)]
            sample = rnd.sample(cps, 3000) + [ord(c) for c in "하配иτά檢ьıмৰ正の時é😇—“”‘’✅━☼▶👾ñäáü"]
            tests["random_unicode"] = [t for t in ("x " + chr(cp) + " x" for cp in sample) if not ref.blocked(t)]
            (priv / "tests.json").write_text(json.dumps(tests, ensure_ascii=False), encoding="utf-8")
            corpus_args = []
            if args.corpus:
                say("== her corpus (READ-ONLY source copied to a private directory; counts only)")
                db = copy_db(args.corpus, priv)
                corpus_stats(db, spec, ref, A, bl, res, priv)
                corpus_args = ["--corpus", "/priv/corpus.json"]
                db.unlink()
            say("== image: vLLM 0.19's own xgrammar backend + the real tokenizer (CPU)")
            tokd = Path(args.tokenizer).expanduser().resolve()
            if not (tokd / "tekken.json").exists():
                sys.exit(f"no tokenizer at {tokd} - run: wordban.py tokenizer --tokenizer {tokd}")
            cmd = ["docker", "run", "--rm", "--name", f"wb-verify-{os.getpid()}", "--network", "none",
                   "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp",
                   "-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1", "-e", "PYTHONUNBUFFERED=1",
                   "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "VLLM_LOGGING_LEVEL=WARNING",
                   "-v", f"{HERE}:/wb:ro", "-v", f"{tokd}:/tok:ro", "-v", f"{build_dir}:/build:ro",
                   "-v", f"{priv}:/priv:rw", "--entrypoint", "/opt/vllm-venv/bin/python", args.image,
                   "/wb/wb_image.py", "verify", "--grammar", "/build/" + GRAMMAR, "--banlist", "/build/banlist.yaml",
                   "--tests", "/priv/tests.json", "--out", "/priv/image-results.json",
                   "--walk-steps", str(args.walk_steps)] + corpus_args
            rc = subprocess.call(cmd)
            rp = priv / "image-results.json"
            if not rp.exists():
                res.add("image run", False, f"no results (exit {rc})")
            else:
                image_part = json.loads(rp.read_text(encoding="utf-8"))
                for r in image_part["rows"]:
                    res.rows.append(r)
                image_part.pop("rows")
    finally:
        if priv:
            shutil.rmtree(priv, ignore_errors=True)
    stamp = {"grammar_sha256": man["grammar_sha256"], "value_sha256": man["value_sha256"],
             "banlist_sha256": man["banlist_sha256"], "revision": man["revision"], "image": args.image,
             "tool_sha256": tool_sha(),
             "full": not args.no_image, "corpus": bool(args.corpus), "pass": res.ok, "rows": res.rows,
             "image_facts": image_part, "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (build_dir / STAMP).write_text(json.dumps(stamp, indent=1, ensure_ascii=False), encoding="utf-8")
    n_f = sum(1 for r in res.rows if not r["pass"])
    say(f"\nVERIFY {'PASSED' if res.ok else 'FAILED'}: {len(res.rows) - n_f}/{len(res.rows)} checks"
        + ("" if not args.no_image else "  (local part only: --no-image; release needs the full run)"))
    return 0 if res.ok else 1


def corpus_stats(db, spec, ref, A, bl, res, priv):
    import re
    msgs = core.load_corpus(db)
    by = {"assistant": 0, "user": 0}
    fam_of = {r["id"]: fam for fam, r in spec.rules}
    fam_of["MARKS"] = "combining-mark cap"
    hits = {"assistant": {}, "user": {}}
    with_hit = {"assistant": 0, "user": 0}
    masked = []
    t0 = time.time()
    for m in msgs:
        by[m["role"]] += 1
        h = ref.hits(m["t"])
        if h:
            with_hit[m["role"]] += 1
            for _s, _e, rid in h:
                f = fam_of.get(rid, rid)
                hits[m["role"]][f] = hits[m["role"]].get(f, 0) + 1
        words = [x for x in h if x[2] != "MARKS"]
        marks = [x for x in h if x[2] == "MARKS"]
        masked.append({"role": m["role"], "t": ref.mask(m["t"]) if h else m["t"], "had_hit": bool(words),
                       "orig": ref.mask_spans(m["t"], marks) if (words and marks) else (m["t"] if words else None),
                       "mark_hit": ref.mask_spans(m["t"], words) if marks else None})
    say(f"  {len(msgs)} messages on the current branch of every chat ({by['assistant']} replies, {by['user']} hers); "
        f"reference pass {time.time() - t0:.0f}s")
    say(f"  replies with a banned form: {with_hit['assistant']}; her messages with one: {with_hit['user']}")
    say(f"  banned-form occurrences by family: replies {hits['assistant']}; hers {hits['user']}")
    # character level: the masked text must pass the grammar's automaton (false blocks), the rest must not
    fb = sum(1 for m in masked if not A.accepts(m["t"]))
    miss = sum(1 for m in masked if m["had_hit"] and A.accepts(m["orig"]))
    res.add("her corpus, character level: 0 false blocks after masking the banned forms", fb == 0,
            f"{len(masked) - fb}/{len(masked)} messages accepted")
    res.add("her corpus, character level: every message with a banned word is refused", miss == 0,
            f"{sum(1 for m in masked if m['had_hit']) - miss}/{sum(1 for m in masked if m['had_hit'])} refused")
    watch = {}
    pats = [(name, re.compile(rx)) for name, rx in (bl.get("watch") or {}).items()]
    for name, _ in pats:
        watch[name] = {"assistant": 0, "user": 0}
    for m in msgs:                  # the text as written: an accented evasion (Angelâ) is not the allowed word
        for name, c in pats:
            watch[name][m["role"]] += len(c.findall(m["t"]))
    say("  watched words (not banned; counts only): " + "; ".join(f"{k}: replies {v['assistant']}, hers {v['user']}"
                                                                  for k, v in watch.items()))
    (priv / "corpus.json").write_text(json.dumps(masked, ensure_ascii=False), encoding="utf-8")
    res.rows.append({"check": "corpus facts", "pass": True, "detail": json.dumps(
        {"messages": by, "with_banned_form": with_hit, "occurrences": hits, "watch": watch})})


# ============================================================================================================
# tokenizer
# ============================================================================================================
def cmd_tokenizer(args):
    import urllib.request
    d = Path(args.tokenizer).expanduser()
    d.mkdir(parents=True, exist_ok=True)
    base = f"https://huggingface.co/{TOKENIZER_REPO}/resolve/{TOKENIZER_REV}/"
    for f in TOKENIZER_FILES:
        p = d / f
        if not p.exists() or p.stat().st_size == 0:
            say(f"downloading {base + f}")
            with urllib.request.urlopen(base + f, timeout=120) as r:
                p.write_bytes(r.read())
        say(f"  {f}: {p.stat().st_size:,} B")
    (d / "consolidated.safetensors").touch()   # empty marker: vLLM's auto mode then picks the Mistral tokenizer
    h = sha(d / "tekken.json")
    ok = h.startswith(TEKKEN_SHA_PREFIX)
    say(f"tekken.json sha256 {h} {'(the pinned tokenizer)' if ok else '(UNEXPECTED - not the pinned tokenizer)'}")
    return 0 if ok else 1


# ============================================================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, build=True):
        if build:
            p.add_argument("--build-dir", default=str(HOME / "build"))
        p.add_argument("--image", default=DEFAULT_IMAGE)
        p.add_argument("--tokenizer", default=str(HOME / "tokenizer"))
    p = sub.add_parser("build", help="banlist.yaml -> grammar + paste value")
    p.add_argument("--banlist", default=str(HERE / "banlist.yaml"))
    p.add_argument("--here", action="store_true", help="build with this Python even if its Unicode version differs")
    common(p)
    p = sub.add_parser("verify", help="the full verification suite (non-zero exit on any failure)")
    p.add_argument("--corpus", help="a webui.db (backup) to measure false blocks on; read-only, copied first")
    p.add_argument("--no-image", action="store_true", help="only the local checks (no tokenizer / docker)")
    p.add_argument("--fuzz", type=int, default=40000)
    p.add_argument("--walk-steps", type=int, default=120000)
    common(p)
    p = sub.add_parser("tokenizer", help="download the pinned Tekken tokenizer files")
    common(p, build=False)
    for extra in ("e2e", "diff", "release"):
        pass
    import wb_cmds                                              # noqa: E402  (e2e, diff, release)
    wb_cmds.add_parsers(sub, common)
    args = ap.parse_args(argv)
    fn = {"build": cmd_build, "verify": cmd_verify, "tokenizer": cmd_tokenizer}.get(args.cmd) or wb_cmds.dispatch(args.cmd)
    return fn(args)


if __name__ == "__main__":
    sys.exit(main())
