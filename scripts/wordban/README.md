# wordban: the word ban on her replies

The ban is one xgrammar grammar, pasted into the model's **`structured_outputs`** custom parameter in OpenWebUI
(Admin Panel → Settings → Models → the model → Advanced Params). OpenWebUI forwards it with every request. The
compactor passes the body through unchanged. vLLM 0.19 compiles the grammar with xgrammar and masks, token by
token, every continuation that would spell a banned word. Nothing in the image changes.

This directory turns that grammar into a declared, reproducible, verified artifact:

```
banlist.yaml ──build──▶ wb_ban.ebnf + structured_outputs.value.txt (+ manifest.json)
                 │
             verify  (local: the artifact's own automaton, a second independent definition, fuzz, her corpus;
                 │    image: vLLM 0.19's xgrammar backend + the real tokenizer, the token-level no-trap proof)
               e2e   (real OpenWebUI → real compactor → a recording vLLM front half; byte identity)
                 │
             release ──▶ ~/zl-ops/word-ban/: the value (vN), the pod check script, the prompt line, a report
```

`release` refuses to run unless `verify` passed, in full (the image half and her corpus included), on the exact
built bytes. It also refuses unless `e2e` passed, or `--allow-no-e2e` is given.

## Files

| file | what |
|---|---|
| `banlist.yaml` | **The ban.** Families of rules, folding options, the combining-mark cap, the prompt line, the pod-check prompt, the test lists. |
| `wordban.py` | The CLI: `build`, `verify`, `tokenizer`; it loads `wb_cmds.py` for `e2e`, `diff`, `release`. |
| `wb_core.py` | Pure Python. Banlist → character-class DFA → EBNF. The EBNF → automaton reader that every check uses. The character-level trap check. The **reference** definition, an independent backtracking matcher. The corpus loader. |
| `wb_image.py` | Runs inside the image: the xgrammar checks and the token-level proof. |
| `e2e/` | Stub vLLM front half, OpenWebUI driver, and the in-container script for `e2e`. |
| `check_template.py` | Template for `word-ban-check.py`, the on-pod check. It judges replies with the grammar saved on the pod, so it never goes stale. |
| `compactor/test_wordban_tool.py` | Unit tests: determinism; the rev-4 trap fixture must fail; a new word gets tested; corpus walk; release refusals (including a stamp from other tool code); the pod interpreter; a truncated paste in the pod check. |

## Workflow

```bash
cd ~/code/zions-light-ai/scripts/wordban
python3 wordban.py tokenizer                                   # once: the pinned Tekken files -> ~/zl-ops/word-ban/tokenizer
python3 wordban.py build                                       # -> ~/zl-ops/word-ban/build/
python3 wordban.py verify --corpus /home/drew/pod-exports/<day>/<backup>/webui.db
python3 wordban.py e2e    --webui-db /home/drew/pod-exports/<day>/<backup>/webui.db
python3 wordban.py diff   git:HEAD~1 banlist.yaml --corpus <webui.db>     # what the change does, in counts
python3 wordban.py release --notes notes.md                    # -> ~/zl-ops/word-ban/
```

- `$WORDBAN_HOME` (default `~/zl-ops/word-ban`) holds `build/`, `tokenizer/` and the private scratch directory.
- `--image` defaults to the published image digest that the pod runs.
- **Every command needs Linux with docker**, except `build --here`, `verify --no-image` and `diff`.
- The build depends on the Unicode tables of the Python that runs it. The banlist pins `unicode_version: "15.0.0"`, the version in the image's vLLM venv and in the unit-test image. When the local Python differs, `build` re-runs itself inside the image, so every machine produces the same bytes.

**Getting the tokenizer.** `wordban.py tokenizer` downloads `tekken.json`, `params.json`, `config.json` and `generation_config.json`:
- source: `coder3101/Cydonia-24B-v4.3-vision-heretic` at the pinned revision `c1e236fb…`, from huggingface.co (public, no token);
- it checks the tekken sha (`6e2501687ccd…`);
- it adds an empty `consolidated.safetensors`. That marker makes vLLM's `tokenizer_mode="auto"` pick the Mistral tokenizer, as on the pod.

## Adding or changing a word

Usually it is one line under `families:` in `banlist.yaml`, then `build`, `verify` and `release`. Pattern syntax:

| syntax | means |
|---|---|
| `gold` | letters: folded and case-insensitive (`gōld`, `ＧＯＬＤ`, `Gołd`). Greek folds its accents (`ἄ` = `α`). |
| `{L}` / `{L?}` | any one letter / optionally one letter |
| `{SEP:G}` / `{SEP:WF}*` / `{SEP:SP}+` | one character of separator set G / any run of set WF (or none) / a run of one or more |
| `[l1]` | one of these characters |
| `where: word_start` (default) / `anywhere` / `inside` | a word start is after a non-letter, or a lower→Upper camelCase step; `inside` = not at a word start |
| `spaced: ALL` | a run of separators between every letter (`g-o-l-d`, `g, o, l, d`) |
| `split: SPLIT` | a run of separators at one place inside the word (`an-gel`, `**G**old`) |
| `match: <pattern>` | banned outright |
| `allow_in_words: [evangel, ...]` | a word that BEGINS with one of these (complete by the end of the banned letters) is exempt: `evangelical`, `marigold`, `inflammation` |
| `context: {after: "los ", plain: true, allow_words: [es]}` | right after the context, the rule becomes a stem with these endings, in plain ASCII letters only (`Los Angeles`, not `los ángeles`) |
| `stem: <pattern>` + `alone`, `other`, `allow_words`, `allow_prefixes` | banned, except the listed whole words (plain letters only) or prefixes of other words |
| `fffd: transparent` | U+FFFD runs (points and accents the tokenizer has no whole token for) are skipped inside the pattern |

**Folding options** (`folding:`): `confusables` (Cyrillic/Greek letters that look Latin), `lookalikes` (`I`/`1`/`|` for `l`, `0` for `o`), both applied only to the letters of the banned patterns; `forbidden_chars` (refused everywhere); `zwj_not_between_letters`; `marks_after_stem: refuse`; `mark_cap`.

**Examples:**
- `{id: ZORBEX, match: zorbex}` bans every word that starts with "zorbex".
- `flamed` was banned in revision 5 by removing `ed` from the flam stem's allowed words.
- Revision 6 bans the stems glued to the end of other words (`theangel`, `puregold`, `candleflame`) with `where: anywhere` / `inside` and an `allow_in_words` list of the real words that contain them.

`verify` adds each rule's own spelling to must-block automatically (lower, Title and UPPER case). Also add real near-misses to `tests.must_block` and legitimate words to `tests.must_pass`.

**Accented letters after a stem are refused** (`accented_continuation: dead`). After `angel`, `flam`, `ru` and so on, an accented form of a letter that an allowed word reads (`Angelâ`, `flamé`, `angeléñÖs`) is simply not allowed; the model writes another word.
- Revision 4 instead let such a letter continue while remembering it (`taint`), and then refused both the word end and every letter. That left only combining marks allowed, and the reply could not end: the **trap** of 2026-09-23.
- `taint` still exists only so the tests can rebuild that trap.
- A stem whose continuation could never complete is refused at build time.

**Funnels** (revision 6). An allowed word that begins with banned letters (`flamenco` after `flame`, `Angela` after `angel`) corners the model once it has written those letters: in revision 5 the state after `" flame"` offered 4 plain tokens and 208 combining marks. Rules:
- keep an exception only when it has real use or a strong reason (revision 6 keeps `Los Angeles` and `flammable`/`flammability` only);
- no combining mark right after a banned stem (`marks_after_stem: refuse`);
- `funnels:` in the banlist: `verify` measures, in every token-reachable state, the NATURAL escape width (printable ASCII / Latin-1 letters / common punctuation tokens that lead somewhere the reply can still finish). Every state must offer at least `min_width` (21, revision 3's narrowest), except the states after the documented prefixes (every letter case), which must offer at least their `floor` and carry a reason. `release` refuses on any funnel regression.

## What `verify` proves

**Local, on the artifact.** Nothing here trusts the generator's internal DFA. The automaton is read back from the EBNF text.
- **No-trap, character level.** Every reachable state can reach an accepting state through ordinary characters: not a combining mark, not U+FFFD, not unassigned or private-use.
- Must-block and must-pass lists, plus the derived cases.
- The trap regressions.
- The combining-mark cap.
- 40,000 fuzzed strings on which the grammar and the independent reference definition must agree.
- With `--corpus`: her messages. The current branch of every chat is walked table-first, using the `output` text; masked text must pass, and every message with a banned word must be refused. Only counts are printed.

**Image: vLLM 0.19's own `XgrammarBackend`, the real tokenizer, CPU.**
- **Rebuild.** The rebuild in the image is byte-identical.
- **Timing.** First-compile time, and the cost per decode step.
- **Test lists.** Every list goes through the real tokenizer, plus collateral, gaps, every folded variant of every letter of the `accent_words`, non-canonical token paths, and random Unicode.
- **Token-level no-trap proof.**
  1. Every vocabulary token is walked from every state. This gives each state's exact allowed-token set, and every state reachable by any token sequence.
  2. At every one of those states, the set is compared with xgrammar's real bitmask, EOS included; they must be identical.
  3. On that token graph, every reachable state must reach EOS through ordinary tokens.
- **Funnel gate** (above), plus the natural-token trap count and the number of states with no word end and no EOS.
- **Adversarial random walks** through the real backend, biased toward marks, U+FFFD, accented letters and ban fragments. Every step must offer an ordinary token or EOS.
- With `--corpus`: her messages through the real tokenizer, with sampled decode steps checked for a normal exit.

## Known limits (by design; listed in `tests.gaps` / `tests.collateral`)

- **Byte-fallback characters reach the grammar as U+FFFD.** In a letter-only rule, one run of them per word may stand for one letter or an accent (`Ŕuach`, `gȍld`). Two unseen letters in one word pass (`ŔÙach`, `Ｇｏｌｄ`).
- **Right after a banned stem, U+FFFD is refused.** So `Angela💔` and `flamingo🦩` without a space are blocked.
- **The mark cap counts whole-token marks only.** There are 154 single-mark tokens. A byte-fallback mark is U+FFFD and ends the run, so runs of such marks are not capped.
- **Synonyms and paraphrase are not blocked.** The prompt line covers intent.
- **Lower-case Spanish `los angeles`** reads as the place name (the context rule cannot tell them apart); `los ángeles` is blocked.
- **Space splits** are banned for gold (`g old`, `go ld`) but not for angel or flame (`an gel` would block "an gelato"); markup (`g<b></b>old`), HTML entities, math-alphanumeric and circled letters, leetspeak beyond `I`/`1`/`|`/`0`, and near-spellings (`anjel`) pass.
- **Control tokens inside a word.** xgrammar sees Mistral's control tokens (`<unk>`, `[INST]` ...) as ordinary text, and vLLM drops them when it detokenises: `g` + `<unk>` + `old` shows as "gold". The model would have to sample a control token in mid-word; no grammar change can see this.
- **Continue response.** The grammar starts again at its root on every request, so a word split across OpenWebUI's "continue response" boundary is not seen.
- **Speculative decoding.** The token-level proof assumes the pod's configuration (no speculative decoding, so no rollback or jump-forward); re-run `verify` if `VLLM_EXTRA_ARGS` ever enables it.

## Privacy

- **Private copies.** `--corpus` and `--webui-db` are copied into a mode-0700 directory under `$WORDBAN_HOME/private/`, and it is deleted when the command ends.
- **Output.** Only counts are printed. `diff --show-forms` prints changed word forms to the terminal and never writes them.
- **Repo.** No file in the repo contains her data. Test fixtures are synthetic.
