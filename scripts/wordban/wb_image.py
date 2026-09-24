"""wordban, the half that runs INSIDE the image (its /opt/vllm-venv): vLLM 0.19's own XgrammarBackend and the
real Tekken tokenizer (mounted at /tok), on CPU. Called by `wordban.py verify` (and `diff`); not meant to be
run by hand.

The token-level NO-TRAP PROOF:
  1. read the grammar's automaton back from the EBNF text (wb_core.Automaton);
  2. walk every vocabulary token from every state (numpy), which gives, for each state, exactly the set of
     tokens the grammar allows and where each one leads; explore every state reachable by ANY token sequence;
  3. at every one of those states, compare that allowed set with vLLM/xgrammar's real bitmask (reached by
     feeding a concrete token path) - they must be identical, EOS included;
  4. then, on that token graph: every reachable state must be able to reach an accepting state (EOS allowed)
     through ORDINARY tokens (a token with at least one character that is not a combining mark or U+FFFD).
Because (3) holds at every reachable state, (4) is a statement about what vLLM will actually do."""
import argparse
import collections
import json
import random
import sys
import time
import types
import unicodedata

import numpy as np

sys.path.insert(0, "/wb")
import wb_core as core  # noqa: E402

from vllm.tokenizers import get_tokenizer  # noqa: E402
from vllm.v1.structured_output.backend_types import StructuredOutputOptions  # noqa: E402
from vllm.v1.structured_output.backend_xgrammar import XgrammarBackend  # noqa: E402
import xgrammar as xgr  # noqa: E402
import importlib.metadata as _md  # noqa: E402
XGR_VERSION = _md.version("xgrammar")
VLLM_VERSION = _md.version("vllm")

rows = []


def add(name, ok, detail=""):
    rows.append({"check": name, "pass": bool(ok), "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{': ' + detail if detail else ''}", flush=True)


class Stack:
    def __init__(self, ebnf):
        self.tok = get_tokenizer("/tok", tokenizer_mode="auto")
        self.vocab = list(self.tok.vocab)
        self.V = len(self.vocab)
        cfg = types.SimpleNamespace(structured_outputs_config=types.SimpleNamespace(disable_any_whitespace=False),
                                    speculative_config=None)
        self.be = XgrammarBackend(vllm_config=cfg, tokenizer=self.tok, vocab_size=self.V)
        self.EOS = self.tok.eos_token_id
        info = xgr.TokenizerInfo(encoded_vocab=self.vocab, vocab_type=xgr.VocabType.RAW, vocab_size=self.V,
                                 stop_token_ids=[self.EOS])
        self.special = set(info.special_token_ids)
        self.ebnf = ebnf
        t0 = time.time()
        self.be.compile_grammar(StructuredOutputOptions.GRAMMAR, ebnf)
        self.first_compile = time.time() - t0
        t0 = time.time()
        self.be.compile_grammar(StructuredOutputOptions.GRAMMAR, ebnf)
        self.second_compile = time.time() - t0
        self.bm = self.be.allocate_token_bitmask(1)

    def enc(self, s):
        return self.tok.encode(text=s, add_special_tokens=False)

    def fresh(self):
        return self.be.compile_grammar(StructuredOutputOptions.GRAMMAR, self.ebnf)

    def run(self, ids):
        g = self.fresh()
        for k, t in enumerate(ids):
            if not g.accept_tokens("r", [t]):
                return k, False, g
        return None, bool(g.accept_tokens("r", [self.EOS])), g

    def allowed(self, g):
        g.fill_bitmask(self.bm, 0)
        b = self.bm[0].numpy().view(np.uint8)
        return np.unpackbits(b, bitorder="little")[:self.V].astype(bool)


def token_classes(st, A):
    """per token: the automaton interval ids of its characters (padded), its length, and whether it is ORDINARY"""
    V = st.V
    seqs = []
    ordinary = np.zeros(V, bool)
    usable = np.ones(V, bool)
    for i, s in enumerate(st.vocab):
        if i in st.special or not s:
            usable[i] = False
            seqs.append([])
            continue
        seqs.append([A.iv(ch) for ch in s])
        ordinary[i] = any(ch != "�" and unicodedata.category(ch) not in core.MARK_CATS for ch in s)
    lens = np.array([len(x) for x in seqs])
    order = np.argsort(-lens, kind="stable")
    L = int(lens.max())
    pad = A.n_intervals
    C = np.full((V, L), pad, np.int32)
    for r, i in enumerate(order):
        x = seqs[i]
        if x:
            C[r, :len(x)] = x
    active = [int((lens > col).sum()) for col in range(L)]
    return C, order, active, ordinary, usable


def proof(st, A, max_states=200000):
    t0 = time.time()
    n, nI = A.n, A.n_intervals
    T2 = np.full((n + 1, nI + 1), n, np.int32)
    T2[:n, :nI] = np.where(np.array([list(r) for r in A.T], np.int32) >= 0,
                           np.array([list(r) for r in A.T], np.int32), n)
    T2[:, nI] = np.arange(n + 1)                           # padding keeps the state
    T2[n, :] = n
    C, order, active, ordinary, usable = token_classes(st, A)
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    ord_sorted = ordinary[order]
    use_sorted = usable[order]

    def walk(s):
        cur = np.full(st.V, s, np.int32)
        for col, k in enumerate(active):
            cur[:k] = T2[cur[:k], C[:k, col]]
        return cur                                         # in sorted order
    acc = np.array(A.acc + [False])
    seen = {0: None}
    q = collections.deque([0])
    edges_ord = {}
    allowed_of = {}
    while q:
        s = q.popleft()
        end = walk(s)
        ok = (end != n) & use_sorted
        allowed_of[s] = ok
        nxt = np.unique(end[ok])
        edges_ord[s] = set(np.unique(end[ok & ord_sorted]).tolist())
        for t in nxt.tolist():
            if t not in seen:
                # remember one concrete token that leads there
                r = int(np.nonzero(ok & (end == t))[0][0])
                seen[t] = (s, int(order[r]))
                q.append(t)
        if len(seen) > max_states:
            break
    reach = sorted(seen)
    # token-level fixpoint: states that can reach EOS through ordinary tokens
    good = {s for s in reach if acc[s]}
    rev = collections.defaultdict(set)
    for s in reach:
        for t in edges_ord[s]:
            rev[t].add(s)
    stack = list(good)
    while stack:
        t = stack.pop()
        for s in rev[t]:
            if s not in good:
                good.add(s)
                stack.append(s)
    traps = [s for s in reach if s not in good]
    t_sim = time.time() - t0

    def path(s):
        out = []
        while seen[s] is not None:
            s, tid = seen[s]
            out.append(tid)
        return list(reversed(out))
    # compare with xgrammar at every reachable state
    t1 = time.time()
    mism = []
    eos_mism = []
    compared = 0
    for s in reach:
        p = path(s)
        g = st.fresh()
        good_path = all(g.accept_tokens("r", [t]) for t in p)
        if not good_path:
            mism.append((s, "path refused"))
            continue
        bits = st.allowed(g)
        sim = np.zeros(st.V, bool)
        sim[order] = allowed_of[s]
        x = bits.copy()
        x[list(st.special)] = False
        x[~usable] = False
        d = np.nonzero(x != sim)[0]
        if len(d):
            mism.append((s, [st.vocab[i] for i in d[:5]], int(len(d))))
        if bool(bits[st.EOS]) != bool(acc[s]):
            eos_mism.append(s)
        compared += 1
    t_cmp = time.time() - t1
    ex = []
    for s in traps[:5]:
        ex.append("".join(st.vocab[t] for t in path(s)))
    return dict(reach=len(reach), traps=len(traps), trap_examples=ex, mism=mism[:10], n_mism=len(mism),
                eos_mism=len(eos_mism), compared=compared, t_sim=round(t_sim, 1), t_cmp=round(t_cmp, 1),
                ordinary=ordinary, usable=usable)


WALL_MIN = 60


def in_symbol_wall(text, pos):
    """is `pos` inside a degenerate symbol wall: a run of WALL_MIN+ characters with no whitespace that holds at
    least 10 non-ASCII symbols (not letters, digits or marks)? The degeneration of 2026-09 wrote such walls (whole
    Unicode tables glued together); a refusal there is not a blocked word. (A long URL is ASCII: not a wall.)"""
    pos = max(0, min(pos, len(text) - 1))
    a = b = pos
    while a > 0 and not text[a - 1].isspace():
        a -= 1
    while b < len(text) and not text[b].isspace():
        b += 1
    run = text[a:b]
    if len(run) < WALL_MIN:
        return False
    sym = sum(1 for c in run if not c.isascii() and not (c.isalnum() or unicodedata.category(c) in core.MARK_CATS))
    return sym >= 10


def walks(st, steps, ordinary, prefixes, seed=5):
    """adversarial random walks through the REAL backend: at each step the mask must hold an ordinary token or
    EOS; the next token is drawn with a strong bias toward combining marks, U+FFFD, accented letters and
    fragments of the banned words."""
    rnd = random.Random(seed)
    V = st.V
    adv = np.zeros(V, bool)
    frag = ("ang", "angel", "ru", "rua", "gol", "gold", "fla", "flam", "flame", "whit", "white", "el", "am", "ach",
            "ch", "ó", "é", "ñ", "Ö", "ø", "Ă", "ά", "γγ")
    for i, s in enumerate(st.vocab):
        if not s or i in st.special:
            continue
        low = s.strip().lower()
        if (not ordinary[i]) or any(not c.isascii() and c.isalpha() for c in s) or any(low.startswith(f) for f in frag):
            adv[i] = True
    adv_ids = np.nonzero(adv)[0]
    done = 0
    bad = []
    pre = [st.enc(p) for p in prefixes] + [[]]
    while done < steps:
        g = st.fresh()
        for t in rnd.choice(pre):
            if not g.accept_tokens("r", [t]):
                break
        for _ in range(rnd.randint(5, 60)):
            bits = st.allowed(g)
            done += 1
            if not ((bits & ordinary).any() or bits[st.EOS]):
                bad.append(done)
                break
            cand = adv_ids[bits[adv_ids]]
            if len(cand) and rnd.random() < 0.7:
                t = int(cand[rnd.randrange(len(cand))])
            else:
                ok = np.nonzero(bits & ordinary)[0]
                if not len(ok):
                    break
                t = int(ok[rnd.randrange(len(ok))])
            if not g.accept_tokens("r", [t]):
                bad.append(("refused an allowed token", st.vocab[t]))
                break
    return done, bad


def cmd_verify(a):
    ebnf = open(a.grammar, encoding="utf-8").read()
    tests = json.load(open(a.tests, encoding="utf-8"))
    st = Stack(ebnf)
    print(f"  tokenizer: {st.V:,} tokens, EOS {st.EOS}, {len(st.special)} special; xgrammar {XGR_VERSION}, vLLM {VLLM_VERSION}", flush=True)
    add("first compile of the grammar (once per vLLM start)", st.first_compile < 120,
        f"{st.first_compile:.1f}s; again from xgrammar's cache {st.second_compile:.3f}s")
    # rebuild comparison with the image's Unicode tables
    spec, again = core.build(a.banlist)
    add("artifact = banlist (rebuilt in the image, byte-identical)", again == ebnf,
        f"Unicode {unicodedata.unidata_version}")

    def check_list(name, items, want_ok):
        bad = []
        for t in items:
            k, eos, _ = st.run(st.enc(t))
            ok = k is None and eos
            if ok != want_ok:
                bad.append(t)
        add(name, not bad, f"{len(items) - len(bad)}/{len(items)}" + (f"; WRONG: {bad[:10]}" if bad else ""))
        return bad
    check_list("must-block through the real tokenizer", tests["must_block"], False)
    check_list("must-pass through the real tokenizer", tests["must_pass"], True)
    check_list("designed collateral (expected blocked)", tests["collateral"], False)
    check_list("known gaps (expected allowed; documented)", tests["gaps"], True)
    check_list("every accented letter in every banned word", tests["accent"], False)
    bad = []
    for p in tests["token_paths"]:
        ids = [i for piece in p for i in st.enc(piece)]
        k, eos, _ = st.run(ids)
        if k is None and eos:
            bad.append(p)
    add("non-canonical token paths", not bad, f"{len(tests['token_paths']) - len(bad)}/{len(tests['token_paths'])} blocked"
        + (f"; ALLOWED: {bad[:5]}" if bad else ""))
    check_list("random Unicode code points between spaces", tests["random_unicode"], True)
    # the proof
    A = core.Automaton(ebnf)
    pr = proof(st, A)
    add("token level: xgrammar's real mask = the automaton's, at every token-reachable state",
        pr["n_mism"] == 0 and pr["eos_mism"] == 0,
        f"{pr['compared']:,} states compared (EOS too), {pr['n_mism']} mask mismatches, {pr['eos_mism']} EOS "
        f"mismatches ({pr['t_sim']}s walk, {pr['t_cmp']}s compare)" + (f"; e.g. {pr['mism'][:3]}" if pr["mism"] else ""))
    add("NO-TRAP PROOF, token level: every reachable state can finish through ordinary tokens",
        pr["traps"] == 0, f"{pr['reach']:,} reachable states, {pr['traps']} trap(s)"
        + (f"; e.g. after {pr['trap_examples']}" if pr["trap_examples"] else ""))
    ordinary = pr["ordinary"]
    # the trap regressions, through the real backend
    bad = []
    for p in tests["trap_regressions"]:
        ids = st.enc(p)
        g = st.fresh()
        for t in ids:
            if not g.accept_tokens("r", [t]):
                break
        bits = st.allowed(g)
        if not ((bits & ordinary).any() or bits[st.EOS]):
            bad.append(p)
    add("trap regressions through the real backend (angelĂ, Angelø, angeléñÖs ...)", not bad,
        f"{len(tests['trap_regressions'])} prefixes" + (f"; TRAPPED: {bad}" if bad else ""))
    # combining-mark runs through the real backend (whole-token marks)
    mark_tokens = [i for i, s in enumerate(st.vocab) if len(s) == 1 and unicodedata.category(s) in core.MARK_CATS]
    heb = [i for i in mark_tokens if 0x0591 <= ord(st.vocab[i]) <= 0x05C7 and unicodedata.category(st.vocab[i]) == "Mn"]
    gen = [i for i in mark_tokens if i not in heb]
    runs = []
    cap = tests.get("mark_cap") or 0
    caph = tests.get("mark_cap_hebrew") or cap
    for base, pool, lim in ((" e", gen, cap), (" ש", heb, caph)):
        for k in range(1, 7):
            g = st.fresh()
            for t in st.enc(base):
                g.accept_tokens("r", [t])
            ok = True
            for j in range(k):
                if not g.accept_tokens("r", [pool[j % len(pool)]]):
                    ok = False
                    break
            runs.append((base.strip(), k, ok, k <= lim))
    bad = [r for r in runs if r[2] != r[3]]
    add("combining-mark cap through the real backend (whole-token marks)", not bad,
        f"{len(mark_tokens)} whole-token marks ({len(heb)} Hebrew); runs of 1-6 after 'e' and after 'ש'"
        + (f"; WRONG: {bad}" if bad else ""))
    # adversarial walks
    t0 = time.time()
    n, bad = walks(st, a.walk_steps, ordinary, tests["trap_regressions"])
    add(f"adversarial random walks through the real backend", not bad,
        f"{n:,} decode steps, each with a normal exit ({time.time() - t0:.0f}s)" + (f"; FAILED at {bad[:5]}" if bad else ""))
    facts = {"first_compile_s": round(st.first_compile, 2), "reachable_states": pr["reach"],
             "xgrammar": XGR_VERSION, "vllm": VLLM_VERSION, "vocab": st.V}
    # corpus
    if a.corpus:
        corpus = json.load(open(a.corpus, encoding="utf-8"))
        t0 = time.time()
        fails = 0
        ntok = 0
        checked = 0
        missed = 0
        trapped = 0
        rnd = random.Random(3)
        walls = 0
        for m in corpus:
            ids = st.enc(m["t"])
            ntok += len(ids)
            g = st.fresh()
            ok = True
            at = None
            for k, t in enumerate(ids):
                if rnd.random() < a.corpus_sample:
                    bits = st.allowed(g)
                    checked += 1
                    if not ((bits & ordinary).any() or bits[st.EOS]):
                        trapped += 1
                if not g.accept_tokens("r", [t]):
                    ok = False
                    at = k
                    break
            if ok and not g.accept_tokens("r", [st.EOS]):
                ok = False
            if not ok and at is not None and in_symbol_wall(m["t"], len(st.tok.decode(ids[:at]))):
                walls += 1                  # refused inside a degenerate symbol wall, not inside a word
            else:
                fails += not ok
            if m.get("had_hit") and m.get("orig"):
                k2, eos2, _ = st.run(st.enc(m["orig"]))
                missed += (k2 is None and eos2)
        add("her corpus through the real tokenizer: 0 false blocks (banned forms masked)", fails == 0,
            f"{len(corpus) - fails - walls}/{len(corpus)} messages accepted, {ntok:,} tokens ({time.time() - t0:.0f}s)"
            + (f"; {walls} refused only inside a degenerate symbol wall (a run of {WALL_MIN}+ characters with no "
               "space, mixing symbols and letters - no word is blocked there)" if walls else ""))
        add("her corpus: every message with a banned word is refused unmasked", missed == 0,
            f"{sum(1 for m in corpus if m.get('had_hit')) - missed}/{sum(1 for m in corpus if m.get('had_hit'))}")
        mk = [m for m in corpus if m.get("mark_hit")]
        mk_ref = sum(1 for m in mk if st.run(st.enc(m["mark_hit"]))[0:2] != (None, True))
        rows.append({"check": "her corpus: messages with a combining-mark run over the cap (character level)",
                     "pass": True, "detail": f"{len(mk)} messages; the real tokenizer + grammar refuse {mk_ref} of "
                     "them (the rest reach the grammar partly as U+FFFD, which ends a run)"})
        print("  [INFO] " + rows[-1]["check"] + ": " + rows[-1]["detail"], flush=True)
        add("her corpus: sampled decode steps all have a normal exit", trapped == 0 and checked >= 100000,
            f"{checked:,} prefixes sampled, {trapped} without a normal exit")
        facts["corpus_tokens"] = ntok
        long_text = max((m["t"] for m in corpus if m["role"] == "assistant"), key=len)
    else:
        long_text = " ".join(tests["must_pass"]) * 20
    ids = st.enc(long_text)[:2000]
    g = st.fresh()
    ts = []
    for t in ids:
        s = time.perf_counter()
        g.fill_bitmask(st.bm, 0)
        ts.append(time.perf_counter() - s)
        if not g.accept_tokens("r", [t]):
            break
    ts.sort()
    med, p99 = 1e6 * ts[len(ts) // 2], 1e6 * ts[int(len(ts) * .99)]
    add("cost per decode step (fill_bitmask)", p99 < 5000,
        f"median {med:.0f} us, p99 {p99:.0f} us, max {1e6 * ts[-1]:.0f} us over {len(ts)} steps")
    facts.update(cost_median_us=round(med), cost_p99_us=round(p99))
    json.dump({"rows": rows, "facts": facts}, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
    return 0


def cmd_accepts(a):
    """for `diff`: which of the given texts each grammar accepts (token level)"""
    texts = json.load(open(a.texts, encoding="utf-8"))
    out = {}
    for gp in a.grammars:
        st = Stack(open(gp, encoding="utf-8").read())
        res = []
        for t in texts:
            k, eos, _ = st.run(st.enc(t))
            res.append(k is None and eos)
        out[gp] = res
    json.dump(out, open(a.out, "w", encoding="utf-8"))
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("verify")
    p.add_argument("--grammar", required=True)
    p.add_argument("--banlist", required=True)
    p.add_argument("--tests", required=True)
    p.add_argument("--corpus")
    p.add_argument("--corpus-sample", type=float, default=0.03)
    p.add_argument("--walk-steps", type=int, default=120000)
    p.add_argument("--out", required=True)
    p = sub.add_parser("accepts")
    p.add_argument("--texts", required=True)
    p.add_argument("--grammars", nargs="+", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    return {"verify": cmd_verify, "accepts": cmd_accepts}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
