# V4 Local Lab

Runs the whole Element front end end to end, on this machine, with no GPU
and nothing live: Synapse + Postgres + Element Web, a tiny CPU model behind
an OpenAI-compatible server, the real compactor, and a minimal bot -- all
loaded with **copies** of her real chat history and memory. It is the
standard harness for testing front-end integrations fully locally, front to
back, per the owner's standing rule.

## Hard safety rule (enforced in code)

Nothing here may ever talk to the live pod or the live Synapse. Every
component that is given a URL (the bot, the model shim, the compactor's
entrypoint, `create_room.py`, the importer) calls `lab/v4/guard.py` before
doing anything else, and **refuses to start** (exit 1, clear message)
unless every configured URL resolves to localhost, a docker-compose
service name, or a private address. `chat.revelationsaints.com` and
`*.proxy.runpod.net` are explicitly denylisted regardless of what they
resolve to. This is not just a doc: `python3 lab/v4/guard.py <name>
NAME=URL` can be run standalone, and every service's own startup does
exactly that.

Data safety: her data is only ever **copied**, never read live.
`scripts/v4lab-copy-data.sh` copies `webui.db` and the `compactor/` store
from the read-only pod-export (`/home/drew/pod-exports/2026-09-23/
backup-1355Z/`) into `/home/drew/scratch/v4lab/data/` (outside this repo,
gitignored) and never writes to the source. Every lab script that touches
the source treats it read-only; the importer refuses to run against a path
under `pod-exports/` directly.

## One-command up / down

```bash
bash lab/v4/scripts/v4lab-up.sh      # idempotent: build+start everything,
                                      # create @her/@owner/@bot, create
                                      # her E2EE room
bash lab/v4/scripts/v4lab-status.sh  # container + health-endpoint check
bash lab/v4/scripts/v4lab-down.sh    # stop containers; keeps volumes/data
bash lab/v4/scripts/v4lab-reset.sh   # wipe Matrix-side state back to
                                      # fresh (keeps her data copies;
                                      # --purge-data also deletes those)
```

All container and project names start `v4lab-` / `v4lab`.

## Logging in

- Element: **http://localhost:18009**
- Synapse (for admin/API use): **http://localhost:18008**
- Users: `her`, `owner`, `bot` on homeserver `localhost`. Registration is
  **closed**; these three are created by the admin CLI
  (`scripts/v4lab-create-users.sh`, run by `v4lab-up.sh`).
- Passwords, the Synapse admin token, and the created room id live in
  `lab/v4/.env.lab` -- **generated with random values on first
  `v4lab-up.sh` run, gitignored, never in git.** `cat lab/v4/.env.lab` to
  read them.
- Element will likely ask a fresh device to "Confirm your digital
  identity" against @her's existing identity (created by a previous
  device/run). Close it (X) and choose **"I'll verify later"** -- never
  "Can't confirm? -> reset your digital identity", whose own text warns
  "You will lose any message history that's stored only on the server."
- If Element asks for a recovery key to restore Secure Backup: it is at
  `lab/v4/importer/out/her_recovery_secret.b64` (created by the importer
  the first time it runs, as a lab stand-in for Element's own Secure
  Backup setup -- gitignored, never printed to any log).

## Re-importing / resetting

```bash
bash lab/v4/scripts/v4lab-copy-data.sh          # (re)copy her data
bash lab/v4/scripts/v4lab-import.sh              # full import, resumable
bash lab/v4/scripts/v4lab-import.sh --last 200   # fast partial import
bash lab/v4/scripts/v4lab-import.sh --count-only # just report branch length
```

The importer is crash-safe and resumable: its journal
(`lab/v4/importer/out/journal.db`) tracks every turn's state
(`planned` -> `encrypted` -> `sent`); re-running continues where it left
off and never re-encrypts an already-encrypted turn.

To start over: `v4lab-reset.sh` (Matrix-side state only), then
`v4lab-up.sh`, then `v4lab-import.sh`.

## Logs

```bash
docker compose -f lab/v4/docker-compose.lab.yml -p v4lab logs -f v4lab-bot
docker compose -f lab/v4/docker-compose.lab.yml -p v4lab logs -f v4lab-compactor
docker compose -f lab/v4/docker-compose.lab.yml -p v4lab logs -f v4lab-model
```
`v4lab-status.sh` also prints `/health` and `/health/full` from the
compactor.

## Architecture

```
 Element Web (:18009) --E2EE--> Synapse (:18008) --> Postgres
                                     ^
 v4lab-bot (mautrix, pinned device, Pg crypto store) --/sync--┘
    │ POST /v1/chat/completions, X-Conversation-Id: ea1494ea...
    v
 v4lab-compactor (:18080, THIS BRANCH's compactor/, real code)
    │ COMPACTOR_STORAGE_ROOT = a COPY of her compactor/ store
    v
 v4lab-model-shim (:8000 internal) --proxies /v1/chat/completions--> v4lab-model (llama.cpp server)
    └── also serves /tokenize in vLLM's {"count": N} shape, via a local
        HF tokenizer for MODEL_REPO (same technique the compactor itself
        uses for its own local-fallback counter)
```

## The model, and its API parity

**Chosen model:** `Qwen/Qwen2.5-0.5B-Instruct` (tokenizer/HF repo) served
as `Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M` (llama.cpp `-hf` auto-download,
~400 MB). Small enough to prefill and generate on CPU in seconds; carries a
real chat template.

**Why not vLLM CPU:** checked first. `vllm/vllm-openai` on Docker Hub ships
only `cuXXX`/`rocm`/`aarch64` tags -- there is no pullable CPU image.
Building `Dockerfile.cpu` from source was judged impractical for a lab
(a from-source vLLM CPU build is a substantial, slow, fragile build step,
not proportionate to what this lab needs). Documented here instead of
attempted silently.

**What runs instead:** llama.cpp's own server, which is already
OpenAI-compatible (`/v1/chat/completions`, including streaming, byte-for-
byte in the shapes the compactor reads) plus a small shim
(`lab/v4/model-shim/shim.py`) that adds the ONE thing llama.cpp's server
does not have in vLLM's shape: `POST /tokenize` returning `{"count": N}`
(llama.cpp's own `/tokenize` returns `{"tokens": [...]}` with no count).
The shim answers this with the real HF tokenizer for `MODEL_REPO`
(`AutoTokenizer`, rendering the chat template first when the request
carries `messages`), so the count is exact, not an estimate.

**Parity gaps, and what is NOT covered:**
- `structured_outputs` (vLLM's grammar-constrained decoding, used by the
  word-ban work) has no equivalent wired up. The shim strips the field and
  logs a warning once; the request still succeeds, just without grammar
  enforcement. The word-ban grammar file at `/home/drew/zl-ops/word-ban/`
  was not exercised in this lab.
- The shim's `/v1/models` synthesizes a minimal response if llama.cpp's own
  is unreachable, rather than truly proxying model metadata.
- No GPU-specific behavior (quantization flags, `GPU_MEMORY_UTILIZATION`,
  etc.) is exercised, obviously.

## Lab-appropriate budgets (compactor overrides)

Set in `docker-compose.lab.yml`'s `v4lab-compactor` environment, smaller
than production so CPU prefill/summarize finishes in a reasonable time on
a 0.5B model with an 8k context:

| Variable | Lab | Why |
|---|---|---|
| `COMPACTOR_MAX_RETAINED_IMAGES` | `0` | the model is text-only |
| `COMPACTOR_KEEP_RECENT_TURNS` | `4` | code default |
| `COMPACTOR_SUMMARY_MAX_TOKENS` | `384` | smaller model, smaller summaries |
| `COMPACTOR_MAX_FACTS_TOKENS` | `800` | smaller injection budget overall |
| `COMPACTOR_INJECT_FACTS_TOKENS` | `200` | paired with the above |
| `COMPACTOR_MAX_RETRIEVAL_TOKENS` | `800` | paired with the above |
| `COMPACTOR_SUMMARY_BLOCK_MAX_TOKENS` | `1500` | keeps the injected block small on an 8k context |
| `COMPACTOR_SELFTEST_ON_BOOT` | `false` | not needed for the lab |
| `COMPACTOR_BACKUP_ENABLED` | `false` | no backup daemon needed in an ephemeral lab |
| `MODEL_REPO` | `Qwen/Qwen2.5-0.5B-Instruct` | so token counting matches the served model |

## The importer (L3)

`lab/v4/importer/import_history.py` walks her branch **table-first**, from
the newest leaf: `chat.current_message_id` **lags** and was observed
pointing at a `chat_message` row that no longer exists at all on this real
copy, so the importer derives the newest leaf itself (a leaf is a row no
other row's `parent_id` names; the tip is the newest leaf by
`created_at`), then walks `parent_id` back to the root. For assistant
rows it prefers the `output` field's text over `content`, per the design.

Each turn is Megolm-encrypted with `vodozemac`, sent via the lab's
appservice masquerading as `@her`/`@bot` with the original `ts`, under a
Megolm session rotated every 40 messages; each session's key is uploaded
to (and read back from) her key backup **before** it is used to encrypt
anything (crash-safe: see the file's docstring). Turns over ~28,000
characters (her longest is 85,005 bytes; Synapse's default event-size
limit refuses that in one event) are split at the last line boundary and
sent as several events under the same session.

**Import count (13:55Z copy):** table-first branch length **3,929** turns
(the task's own estimate was "about 3,950" -- consistent). All 3,929 were
imported (`sent_this_run` 3,873 on the run that finished it, after an
earlier run had already sent 56 before hitting the event-size issue that
`split_body()` fixed). `--last N` is supported for a fast partial import;
one full run was completed as required.

## L4 verification (Playwright)

`lab/v4/scripts/verify_element.py`, run inside the existing `bounce-pw`
image on **host networking** (`--network host`, so `http://localhost:18009`
resolves the same way it would for a human, and the page's origin is
literally `localhost` -- required for WebCrypto's secure-context check;
resolving Element by its compose service name instead gives a non-secure
origin and Element refuses to run at all).

It logs in as `@her`, handles the safe "confirm your digital identity ->
I'll verify later" path (never the destructive "reset your digital
identity", whose own text warns about losing server-only history), opens
her room, scrolls to the top (patient: real pagination over ~3,929 turns
takes real wall-clock time, not a few wheel events), and reports the tile
count and whether the beginning-of-room marker was reached. It then sends
a live message and waits for the bot's streamed reply to arrive and
stabilize.

Run it:
```bash
source lab/v4/.env.lab
docker run --rm --network host \
  -e ELEMENT_URL=http://localhost:18009/ \
  -e HER_PASSWORD="${LAB_HER_PASSWORD}" \
  -e HER_SCREENSHOT_DIR=/shots \
  -v "$(pwd)/lab/v4/scripts:/pw:ro" \
  -v "$(pwd)/lab/v4/importer/out:/out:ro" \
  -v /home/drew/scratch/v4lab/shots:/shots \
  bounce-pw:latest python3 /pw/verify_element.py
```
Screenshots (if `HER_SCREENSHOT_DIR` is set) are for the operator's own
debugging only. **Never prints her message text** -- only counts, ids and
lengths; grep the script for `body` to audit that.

To confirm the compactor logged the mapped conversation id:
```bash
docker logs v4lab-compactor 2>&1 | grep 'source=header'
```

## Where this differs from production

- **Model:** a 0.5B CPU model behind llama.cpp + a shim, not the
  production GPU model behind vLLM. See "parity gaps" above.
- **No apiauth/gate/capture system.** V4_DESIGN.md's byte-exact gate
  (G1-G6, the capture, `seq` pairing) is production-only future work not
  yet in `compactor/` on this branch; this lab runs the compactor exactly
  as it exists today, pointed at a copy of her memory and the tiny model.
  Nothing in this lab depends on or exercises that gate.
- **No streaming split-by-encrypted-size (H-8).** The bot's edits are not
  chunked at the production 40,000-byte budget; the lab's short replies
  from a 0.5B model haven't needed it. The importer DOES split oversized
  historical turns (see above), but as separate sent messages, not as the
  production edit-based scheme.
- **No cross-signing bootstrap or full Secure Backup UX** on the importer
  side -- it creates a bare key-backup version (stand-in for "Element
  already set this up"), not full 4S/SSSS. A real first Element login
  bootstraps its own cross-signing on top of that, which is what
  `verify_element.py` actually exercises.
- **No `!` command-specific bot logic beyond pass-through** -- the
  compactor parses `!` commands from the message text itself
  (`commands.parse_command`), so the bot doesn't need any special-casing,
  but this was not exhaustively tested per-command.
- **Synapse rate-limit override, registration-closed admin flow, and the
  appservice masquerade** are all real, not stubbed -- these match how
  chat-app's own dev stack and the production runbook already work.
- **chat-app itself was not modified.** Its `docker-compose.yml`,
  `synapse/homeserver.template.yaml` and `element/config.json` were read
  as reference/adaptation source for this lab's own
  `lab/v4/{docker-compose.lab.yml,synapse/homeserver.lab.yaml,
  element/config.lab.json}`; no changes were needed in the chat-app repo.

## Known gaps / things that don't work yet

- The model-shim's `structured_outputs` (word-ban grammar) stripping is
  untested end-to-end -- documented as a gap, not silently ignored.
- `verify_element.py`'s selectors are pinned to this Element build's
  actual DOM (much of it hashed CSS-module classes, not the classic
  `mx_Foo` names); a future Element bump may need reselecting. The script
  favors text/role-based locators over CSS classes for exactly this
  reason where it could.
- The bot does not implement the production 40,000-byte streaming-edit
  split (see above) -- fine for a 0.5B model's short replies, would need
  adding if this lab is later used with a bigger local model.
