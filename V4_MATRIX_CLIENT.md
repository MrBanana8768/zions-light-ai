# V4.0 — Element as the client: the Matrix bot

The first piece of V4. It replaces OpenWebUI with Element, served from the
existing `chat-app` deployment at `chat.revelationsaints.com`, by adding a
bot that sits between a Matrix room and the compactor.

## Why this, and why first

OpenWebUI stores each conversation as one JSON document in one SQLite row and
rewrites the whole row on every message. At about 3,800 messages that row is
55 to 106 MB, rewritten per send under SQLite's single-writer lock, on a RunPod
network volume that stalls. The results in September 2026 were constant
"database is locked" failures, a chat tree with 29 snapped parent links, a
"current message" pointer that moved onto a 09-05 message, and her memory
forking five times because the conversation's identity was a hash of the system
prompt and her first message.

Matrix stores a message as one event, appended once and never rewritten.
A send writes kilobytes regardless of how long the conversation is. The
homeserver owns the history; no client can re-root or overwrite it. Element
pages history on demand. The room id gives a conversation identity that never
changes. That removes the whole class of failure rather than one instance of
it, and it retires `webui.db`, so the planned database move becomes
unnecessary.

## Decisions (owner, 2026-09-22)

| Question | Decision |
|---|---|
| Homeserver | the existing `chat-app`: Synapse on Postgres, Hetzner, Terraform-managed, persistent volume, push-to-deploy |
| Encryption | rooms are end-to-end encrypted |
| Users | one now; design for several |
| Her history | **imported, non-negotiable**, faithfully: her messages from her account, the AI's from the bot's, original timestamps, encrypted |
| Host uptime | the Hetzner box stays up permanently |
| Where the code lives | this repository, `bot/`; `chat-app` runs its image as one compose service |
| Reaching the compactor | HTTPS through the RunPod proxy, authenticated by PR #30's API key |

## Architecture

```
Element (browser)  --E2EE-->  Synapse + Postgres        (Hetzner, chat-app)
                                   |  /sync, encrypted events
                              zla-bot  (Hetzner, chat-app compose service)
                                   |  decrypts; builds the OpenAI request
                                   |  HTTPS + Authorization: Bearer <key>
                              compactor  (RunPod pod, :8080 via proxy)
                                   |
                                 vLLM
```

- **Library: mautrix-python.** Actively maintained, used by the mautrix
  bridges in production, supports both a normal logged-in device with E2EE and
  a Synapse application service, and keeps its crypto store in Postgres
  (`PgCryptoStore`), which is already on the persistent volume. The bot's
  encryption identity therefore survives a box rebuild. Encryption is via
  python-olm, which is what every Python Matrix client uses today.
- **Live mode: the bot is an ordinary Matrix user with one device.** It syncs,
  decrypts, replies. No application-service encryption, which is still
  experimental in Synapse.
- **Import mode: an application service**, used once. An appservice can act as
  users in its namespace and set an event's timestamp (the `ts` parameter),
  which is the only supported way to make imported history carry its original
  dates. It encrypts as each user it acts for.
- **Conversation id.** The bot sends `X-Conversation-Id: mx-<first 32 hex of
  sha256(room id)>`. Room ids contain characters the compactor strips; a hash
  keeps the id stable, short and collision-safe. The id is recorded per room.
- **Full-resend mode first.** Today the compactor depends on the client
  resending the whole history (its turn index is the array length; see
  `FRONTEND_SPEC.md` section 15). The bot keeps a mirror of each room's
  decrypted transcript in its own Postgres schema, derived from room events and
  never written back to the room, and sends all of it, exactly as OpenWebUI
  did. Bounded windows wait for the server-side turn sequence in V4 proper.
- **Model settings move to the bot.** The system prompt, temperature, `min_p`,
  `repetition_penalty` and any other sampling parameters currently held in
  OpenWebUI's model configuration become bot configuration.
- **Replies stream** as an initial message plus edits at a modest interval, with
  a typing indicator while the model works.
- **Images** are downloaded, decrypted and passed as image content parts.
- **Slash commands** pass through unchanged; the compactor already handles them.
- **Failure is visible.** If the compactor or the pod is down, the bot says so
  in the room, and the message is not silently dropped.

### What end-to-end encryption does and does not buy here

It protects the conversation from anyone with access to the Hetzner database
or disks who is not the bot. The bot decrypts by design, keeps a plaintext
mirror, and forwards plaintext to the compactor, which stores memory in
plaintext on the pod. The bot is the trust boundary; its host and its crypto
store must be treated accordingly.

## The central risk: importing encrypted history she can still read

The faithful import must encrypt each of her imported messages as her, from a
device the importer creates. The keys for those messages must reach every
device she will ever use, including ones she logs in on after the importer is
gone. If they do not, the history imports successfully and then shows as
"unable to decrypt" forever.

The way through is server-side key backup: the importer uploads the session
keys for everything it encrypts to her account's key backup, so any of her
devices that unlocks backup can read the history. That needs her backup set up
first, and the importer given access to it for the duration of the import.

This is the one part of the plan that has to be proved before anything else is
built. Phase 1 exists to prove it or kill it.

## Phase 1 result (2026-09-22): go, with corrections

The spike (`bot/spike/`) ran on a copy of the `chat-app` dev stack with
synthetic data:

- **Encrypted round trip:** works. The bot sent `X-Conversation-Id: mx-<32 hex>`
  and a bearer key to a stub compactor, and replied encrypted.
- **Faithful import:** 4,000 messages, two senders, each encrypted as its own
  sender with its original timestamp, in 327 s with no throttling once the
  settings below are applied.
- **Readable afterwards, in real Element:** Element Web created the backup and
  showed the recovery key. The importer imported 600 messages using only the
  backup's public key. All of the user's devices were then deleted. Element Web
  on a brand-new device restored with that recovery key and showed all 600,
  each from the right sender and with its original date.
- **The bot survives restarts and rebuilds:** it keeps the same device and
  identity key, and keeps decrypting.

What changed in the design as a result:

1. **The importer never creates a backup version and never holds backup
   secrets.** Her Element creates the backup. The importer reads the current
   version's public key and uploads to it. There is no recovery-key handoff.
2. **Neither mautrix-python nor matrix-nio implements key backup.** About 150
   lines on top of python-olm's `PkEncryption` do it (`bot/spike/lib/backup.py`).
3. **The importer is separate code from the live bot.** It manages Megolm
   sessions directly rather than through `OlmMachine`.
4. **The bot's transcript mirror is seeded by the importer,** which holds the
   plaintext, keyed by event id. The bot never needs to decrypt imported
   history, which its own device could not do anyway.
5. **The import must be resumable.** Transaction ids are scoped to the access
   token, so a naive re-run after a crash duplicates every message. The
   importer records its progress durably and resumes from it.
6. **Synapse settings for the import only:** an application-service
   registration with `rate_limited: false`, and relaxed `rc_message` limits.
   Both are removed once the import is done.
7. **The bot's `device_id` is pinned in its configuration,** not generated by
   the server, so a rebuild comes back as the same device.

What she will see, and must be told before cutover:

- **Every imported message carries Element's grey ⓘ icon.** Its tooltip reads
  "the authenticity of this encrypted message can't be guaranteed on this
  device". It is the same icon Element shows on any history restored from key
  backup. Messages sent after cutover will not have it. Removing it from the
  imported history would mean giving the importer her cross-signing key, which
  this plan deliberately avoids.
- **Setting up Secure Backup in Element has a "type the key back in" step.**
  If it is skipped, the account ends up with a backup she can never restore
  from, and nothing warns her. The cutover runbook must walk through it.
- **After a restore, Element showed a "you have unverified sessions" banner.**
  Its source is not yet understood; Phase 2 must find out and remove it.

## Phases

Each phase is a lane plus a hostile review, as for every compactor release.

**Phase 0 — prerequisites.**
- PR #30 (API-key gate) rebased onto the current compactor line, reviewed and
  shipped. Until then port 8080 is public with no authentication.
- Port 8080 stays exposed only for the bot, behind the key.

**Phase 1 — spike, on the local `chat-app` dev stack, synthetic data only.**
Go / no-go for the design. Must demonstrate:
1. an encrypted room round trip: a user sends, the bot decrypts, calls a stub
   compactor, replies encrypted, the user reads it;
2. a faithful import of a synthetic history of about 4,000 messages, with
   original timestamps, encrypted as each sender;
3. **decryptability after the fact**: the importer's device is deleted, the
   user logs in on a brand-new device, restores key backup, and every imported
   message decrypts;
4. the bot's crypto store survives a restart and a rebuild of its container.

**Phase 2 — the live bot.** Full-resend mode against a stub compactor, then
against a test conversation on the real compactor. Streaming, images, commands,
failure notices, several rooms and users, the request log confirming
`source=header` with the derived id.

**Phase 3 — the importer at scale.** OpenWebUI export to import, on synthetic
data at her real size, then a dry run from a copy of her real chat into a
throwaway room that is destroyed afterwards.

**Phase 4 — cutover**, as a runbook:
1. v3.1.9.4 or later live on the pod, which makes memory merges safe.
2. Import her history into her room.
3. Merge every memory id that holds part of her memory into the room's
   conversation id, while she is not chatting.
4. She logs in to Element, restores key backup, and sees her whole history.
5. First message; verify from the log.
6. OpenWebUI stays up read-only as the archive until she is settled, then is
   retired, and `webui.db` with it.

## Not in this piece

Bounded windows and the server-side turn sequence; the memory surfaces from
`FRONTEND_SPEC.md` ("what it was given", editing facts) beyond what slash
commands already offer; voice messages; federation. And the compactor's own
stores (the episodic SQLite and the fact files) stay on the network volume.
They have one writer and are a smaller problem than `webui.db`, but they are
still on a disk that stalls.
