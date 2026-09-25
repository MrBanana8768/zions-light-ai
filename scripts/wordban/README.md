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
# or, for one revision at a time, build -> verify -> e2e -> diff -> a notes draft in a single run:
python3 wordban.py cycle --corpus <webui.db> --build-dir ~/zl-ops/word-ban/buildN

# AFTER PASTING revision N onto the model (Step 1 of the release report), set live_revision: N in banlist.yaml.
# `diff`/`cycle` then default their "what changed" comparison to THAT revision, not revision-1 - so the default
# answer is always "what changed vs what she is actually seeing", even when a revision was built and released
# as a file but never pasted (e.g. v6): `wordban.py diff banlist.yaml` (old omitted) resolves to live_revision.
```

`cycle` (revision 7) automates the loop above: it builds into `--build-dir`, runs the full `verify` (image + corpus),
runs `e2e` against `--compactor-ref v3.1.9` (the same default `e2e` itself uses - what prod actually runs; round 2:
by default this is the ONLY compactor e2e runs - `--e2e-both` also runs the image's own bundled one), diffs
the new banlist against `live_revision:` if the banlist declares one (else the previous revision), found under
`--out-dir` (`--base-release` overrides either default) (counts only), and writes a plain-English-plus-mechanical
notes draft (`~/zl-ops/word-ban/work/notes-revN.md` - the "what changed" section comes straight from the rule/
test-list/structural diff (`diff_rules_data`/`diff_tests_data`/`diff_struct_data`), never hand-written prose - a
STRONG DRAFT, not a drift-proof guarantee: it is only as complete as that diff is, so still read it before using
it). Each step's full output is logged to `~/zl-ops/word-ban/work/logs/cycle-revN-<step>.txt`. It stops at the
first failing step and never releases: a hostile review still has to happen first, by hand, exactly as before.
`--no-e2e` skips the e2e step (`release` still needs a passing e2e stamp of its own, or `--allow-no-e2e`).

**`--recent N` (default 200), rev 7 round 2 (owner decision):** "I only pull the most recent chat for behavior,
so only the most recent replies should be scanned." `verify`, `diff` and `cycle` default to loading only the
MOST RECENTLY ACTIVE chat in `--corpus` (the chat with the newest message, by timestamp) - walked table-first
exactly like the full corpus loader - and taking its last `N` messages; every corpus-based check (character-level
false blocks and refusals, the real-tokenizer pass, sampled decode steps, and `diff`'s counts) uses only these.
`--full-corpus` opts back into the old behaviour (every chat's whole current branch). The verify stamp records
which mode ran (`corpus_mode`: `{"mode": "recent", "n": N, "chat_id_sha256": ..., "messages": ...}` or
`{"mode": "full", "messages": ...}` - only a HASH of the chat id, never the id or her data) and the message
count; `release` accepts either mode with no extra flag, and still refuses a stamp with no corpus check at all.
Nothing about the image-half token-level proof, the funnel gate, mask equality, or the test lists changes.

- `$WORDBAN_HOME` (default `~/zl-ops/word-ban`) holds `build/`, `tokenizer/` and the private scratch directory.
- `--image` (`DEFAULT_IMAGE` in `wordban.py`) is pinned to the digest of the TAG the pod actually deploys
  (`DEFAULT_IMAGE_TAG`, e.g. `v3.1.9-cu12`) - **update both on every deploy**, or verify/e2e prove something
  other than what's live. `verify` warns loudly if `--image` differs from `DEFAULT_IMAGE`, and the verify/e2e
  stamps record the image tag as well as the digest.
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
| `{SEP:R}` / `{SEP:PJOIN}*` / `{SEP:SP}+` | one character of separator set R / any run of set PJOIN (or none) / a run of one or more |
| `[l1]` | one of these characters |
| `where: word_start` (default) / `anywhere` / `inside` | a word start is after a non-letter, or a lower→Upper camelCase step; `inside` = not at a word start |
| `spaced: ALL` | a run of separators between every letter (`g-o-l-d`, `g, o, l, d`) |
| `split: SPLIT` | a run of separators at one place inside the word (`an-gel`, `**G**old`) |
| `match: <pattern>` | banned outright |
| `allow_in_words: [evangel, ...]` | a word that BEGINS with one of these (complete by the end of the banned letters) is exempt: `evangelical`, `marigold`, `inflammation` |
| `context: {after: "los ", plain: true, allow_words: [es]}` | right after the context, the rule becomes a stem with these endings, in plain ASCII letters only (`Los Angeles`, not `los ángeles`) |
| `not_after: <pattern>` | (rev 7) the opposite of `context.after` - the rule does NOT fire when the word is directly preceded by `<pattern>`. `G OLD`'s `not_after: "[0123456789]{SEP:SP?}"` keeps a "g" right after a digit (glued or with one space: `100g`, `200 g`) from reading as a split "gold" (it's the unit gram) |
| `like: <family or rule id>` | (rev 7) declares that this family's separator policy must match the named family's IN KIND - `verify` checks a MULTISET of every separator-set NAME the family's rules reference anywhere (`match:`/`stem:` pattern text, `context.after`, `not_after`, `split:`, `spaced:`, `phrase_split`'s `inner_sep`/`outer_sep`), not "last rule wins", plus whether it has a glued `where: inside`/`anywhere` form. Not currently declared by any family (its one user, `white_flame`, was removed in round 2 - see Known limits); kept for a future pair |
| `phrase_split: [w1, w2]` + `inner_sep`, `outer_sep` | (rev 7) a TWO-word phrase where a split is also allowed INSIDE either word ("yel-low light", "yellow lig-ht", "yel low light"): one variant per split position in `w1` (with the mandatory `outer_sep` boundary run before `w2`), and one per split position in `w2` (with `outer_sep` after `w1`). Used by `yellow_light`'s `*-WORDSPLIT` rule |
| `stem: <pattern>` + `alone`, `other`, `allow_words`, `allow_prefixes` | banned, except the listed whole words (plain letters only) or prefixes of other words |
| `fffd: transparent` | U+FFFD runs (points and accents the tokenizer has no whole token for) are skipped inside the pattern |

**Folding options** (`folding:`): `confusables` (Cyrillic/Greek letters that look Latin), `lookalikes` (`I`/`1`/`|`/`ʟ` for `l`, `0` for `o`, `ɑ` for `a` - rev 7 fix 2), both applied only to the letters of the banned patterns; `forbidden_chars` (refused everywhere); `zwj_not_between_letters`; `marks_after_stem: refuse`; `mark_cap`.

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
- keep an exception only when it has real use or a strong reason (revision 6 keeps `Los Angeles`, `flammable`/`flammability` (owner) and `flambé`/`flamboyant` (2 real uses));
- no combining mark right after a banned stem (`marks_after_stem: refuse`);
- `funnels:` in the banlist: `verify` measures, in every token-reachable state, the NATURAL escape width (printable ASCII / Latin-1 letters / common punctuation tokens that lead somewhere the reply can still finish). Every state must offer at least `min_width` (21, revision 3's narrowest), except the states after the documented prefixes (every letter case), which must offer at least their `floor` and carry a reason. `release` refuses on any funnel regression.

## What `verify` proves

**Local, on the artifact.** Nothing here trusts the generator's internal DFA. The automaton is read back from the EBNF text.
- **No-trap, character level.** Every reachable state can reach an accepting state through ordinary characters: not a combining mark, not U+FFFD, not unassigned or private-use.
- Must-block and must-pass lists, plus the derived cases.
- The trap regressions.
- The combining-mark cap.
- 40,000 fuzzed strings on which the grammar and the independent reference definition must agree.
- With `--corpus`: her messages - by default (rev 7 round 2) only the most recently active chat's last `--recent` messages (200), walked table-first using the `output` text; `--full-corpus` scans every chat's whole current branch instead. Masked text must pass, and every message with a banned word must be refused. Only counts are printed.

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
- With `--corpus`: her messages (same recent-by-default / `--full-corpus` scope as above) through the real tokenizer, with sampled decode steps checked for a normal exit.

## Known limits (by design; listed in `tests.gaps` / `tests.collateral`)

- **Byte-fallback characters reach the grammar as U+FFFD.** In a letter-only rule, one run of them per word may stand for one letter or an accent (`Ŕuach`, `gȍld`). Two unseen letters in one word pass (`ŔÙach`, `Ｇｏｌｄ`).
- **Right after a banned stem, U+FFFD is refused.** So `Angela💔` and `flamingo🦩` without a space are blocked.
- **The mark cap counts whole-token marks only.** There are 154 single-mark tokens. A byte-fallback mark is U+FFFD and ends the run, so runs of such marks are not capped.
- **Synonyms and paraphrase are not blocked.** The prompt line covers intent.
- (rev 6 only; fixed in rev 7) `yellow light` glued after lower-case letters (`theyellowlight`) used to pass. Rev
  7 bans it (`where: inside`, like the flame family's `FLAM-IN`) per the owner's literal wording ("glued to
  another word").
- **"yellow light" is banned only when JOINED (owner decision A), and PASSES by design across a sentence/clause
  break.** `"yellow. light"`, `"yellow, light"`, `"yellow; light"`, `"yellow: light"`, `"Yellow + light = cream"`,
  a line break that follows punctuation or is followed by a list marker (`- `/`* `/`1.`) or a blank line, all
  PASS - this REPLACES the rev-7-draft behaviour (fixed by the hostile review) where `,`/`+`/NBSP/tab/newline
  turned ordinary sentence punctuation into a false block ("yellow, light", "e.g." NBSP "old"). Only a genuinely
  JOINED form (space/NBSP/tab run, a single line break, a hyphen/underscore/dot/slash/etc, markup, or a split
  inside either word) is banned. Owner decision B: inside the phrase, "yellow light/lights/light's/lighting" are
  banned; "yellow lighter(s)/lighthouse(s)/lightning" PASS (they describe a yellow object) - a documented funnel
  (`funnels.documented`), since only those three words are a live way on after "yellow light".
- **"white flame" needs no phrase rule of its own.** Owner decision A originally asked for a `white_flame`
  family with the same joined-vs-break design as `yellow_light`, "for consistency" - but "flame" alone is
  already unconditionally banned in every context by the `flame` family (any occurrence, any separator, no
  escape except flammable/flambé/flamboyant), so a joined/break phrase rule for "white flame" can never change
  any verdict: every string that spells "white flame" already contains the banned word "flame" by itself. It
  cost 7 rules (a real fraction of the grammar, part of why the first compile went over the tool's 120s limit)
  for zero behavioural effect, so round 2 removed the family (and the `like:` pairing it existed to declare).
  Every test that used to name "white flame" still passes, blocked by `flame` alone.
- **NBSP (U+00A0) and narrow NBSP (U+202F) behave exactly like a plain space everywhere**; tab and newline are
  never in-word SPLIT characters for angel/gold/flame any more (`e.g.\told`, `e.g.\nold`, `100 g\nold` all pass) -
  fixed after the rev-7-draft added them and created false blocks. A run of other exotic whitespace (e.g. EM
  SPACE U+2003) also does not split a word (it never did for angel/flame; it briefly did for a SPLIT-set change
  in the rev-7 draft, since reverted).
- **Lower-case Spanish `los angeles`** reads as the place name (the context rule cannot tell them apart); `los ángeles` is blocked.
- **Space splits** are banned for gold (`g old`, `go ld`) but not for angel or flame (`an gel` would block "an gelato"); markup (`g<b></b>old`), HTML entities, math-alphanumeric and circled letters, leetspeak beyond `I`/`1`/`|`/`0`, and near-spellings (`anjel`) pass.
- **Control tokens inside a word.** xgrammar sees Mistral's control tokens (`<unk>`, `[INST]` ...) as ordinary text, and vLLM drops them when it detokenises: `g` + `<unk>` + `old` shows as "gold". The model would have to sample a control token in mid-word; no grammar change can see this.
- **Continue response.** The grammar starts again at its root on every request, so a word split across OpenWebUI's "continue response" boundary is not seen.
- **Speculative decoding.** The token-level proof assumes the pod's configuration (no speculative decoding, so no rollback or jump-forward); re-run `verify` if `VLLM_EXTRA_ARGS` ever enables it.

## Privacy

- **Private copies.** `--corpus` and `--webui-db` are copied into a mode-0700 directory under `$WORDBAN_HOME/private/`, and it is deleted when the command ends.
- **Output.** Only counts are printed. `diff --show-forms` prints changed word forms to the terminal and never writes them.
- **Repo.** No file in the repo contains her data. Test fixtures are synthetic.
