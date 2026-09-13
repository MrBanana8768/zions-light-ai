# Runbook — give her conversation a stable identity, then put its memory back together

**Rewritten for v3.1.9 (2026-09-13).** The 2026-09-11 version of this runbook
could not work: it relied on the OpenWebUI filter to carry the chat id, and on
OpenWebUI 0.11.0 the filter's chat id never reaches the compactor (details in
"Why not the filter" at the end). The route below was verified on OpenWebUI
0.11.0 against the compactor's own resolver.

**The order is the whole point.** Doing these in a different order costs memory
that then has to be recovered by hand.

| # | step | why it is in this position |
|---|---|---|
| 0 | Preconditions | the pod must be on v3.1.9 with `WEBUI_DB_LOCAL=false` |
| 1 | Find the two ids | you need the old id and the new id before anything changes |
| 2 | Merge old → new, **before** the new id sees a single message | a new id with no facts file starts a background backfill that corrupts the summary position (hostile review B, F3) and pushes her real facts out (F6) |
| 3 | Add the connection header | from here on, her chat resolves to the new id |
| 4 | Let her send one message, then verify | proves the id arrives, and builds the summary hierarchy under it |
| 5 | (Optional) the history cap | only after 4 passes, sized from her real data |
| 6 | (Optional) change the system prompt | safe only once 4 passes |
| R | Rollback | a strict order; see the end |

**Tell her before you start:** do not send any message in that chat from the
start of step 1 until you say so in step 4. Steps 1-3 take about ten minutes.

## The mechanism, in one paragraph

With no chat-id header, the compactor identifies a conversation by
`sha256(system ||| first_user[:512])[:16]` (`compactor/memory.py`,
`_fingerprint_hash`). So the system prompt **is** part of her identity: edit it
and she gets a new, empty memory. Nothing is lost, but it is no longer reachable
from the conversation. Every one of her requests in the log still says
`source=hash`. The fix is to have OpenWebUI send its own chat id as an HTTP
header, `X-Conversation-Id`, which the compactor reads before anything else.

All commands below run in the RunPod **Web Terminal**, on the pod. Copy each
grey block exactly as written. Anything in `<angle brackets>` is a value you
paste in yourself, without the brackets.

---

## 0. Preconditions

```bash
curl -s localhost:8080/health/full | python3 -c "import json,sys; d=json.load(sys.stdin); print('status:', d['status']); print('reasons:', d['status_reasons'])"
```

**Success:** it prints `status: ok` (or `status: degraded` with only a
`memory tail skipping` reason, see OPERATIONS.md "Reading /health/full").
**If `curl` prints nothing or "Connection refused":** the compactor is down;
stop and fix that first (OPERATIONS.md, "A service is FATAL").

```bash
supervisorctl status webuidb-sync
```

**Success:** the line says `STOPPED` and `Not started`. That means
`WEBUI_DB_LOCAL=false` took effect and her chat history is on `/data`.
**If it says `RUNNING`:** stop. The pod booted with the database moved to local
disk, which is not the production placement. Do not continue with this runbook;
see RUNPOD_DEPLOY.md "WEBUI_DB_LOCAL".
<!-- LANE-DEP: webuidb WEBUI_DB_LOCAL parsing (empty value currently means true) -->

Make a backup now, so there is a copy from immediately before the change:

```bash
/opt/compactor-venv/bin/python /opt/compactor/backup.py --once --json > /tmp/pre-identity-backup.json 2>&1; echo "EXIT=$?"; grep -E '"ok"|"archive"|"detail"' /tmp/pre-identity-backup.json
```

It can take several minutes; wait for the prompt to come back.
**Success:** `EXIT=0`, `"ok": true` and an `"archive": "zions-backup-…tar.gz"`
name. Write that archive name down. **If `EXIT=1`:** read the `"detail"` line.
`database is locked` or `readonly database` means OpenWebUI was mid-write;
wait one minute and run it again. Anything else: stop and ask for help. Do not
continue without a backup.
<!-- LANE-DEP: backup run_once report shape / census guard wording -->

---

## 1. Find the two ids

**The new id** is OpenWebUI's chat id. Open her chat in the browser and look at
the address bar: `https://<pod>-3000.proxy.runpod.net/c/<uuid>`. The part after
`/c/` looks like `c9f83137-1a2b-4c3d-8e9f-c685ad408138` (36 characters, four
dashes). Copy it exactly. That is `<new-uuid>` below.

**The old id** is the hash her requests resolve to today. Her own chat
requests carry a long history; OpenWebUI's background calls (titles, tags,
follow-ups) carry one or two messages. This shows only the long ones:

```bash
grep -aE "conv_id=[0-9a-f]{16} source=hash msgs=[0-9]{2,}" /data/logs/compactor.log | tail -5
```

**Success:** five lines, all with the SAME 16-character id after `conv_id=`,
each with a large `msgs=` number (in the hundreds) that grows as she chats.
That id is `<old-hash-id>`.
**If two different ids appear:** she has used more than one long chat
recently, or the system prompt was edited. Match the `msgs=` number to the chat
you opened (the longest one is almost always hers) and ask before continuing.
**If nothing prints:** the log rotated. Try `/data/logs/compactor.log.1` in the
same command.

Check the new id is still unused:

```bash
curl -s localhost:8080/admin/conversations/<new-uuid> | python3 -m json.tool
```

**Success:** `"facts": {"exists": false, ... "count": 0}` and
`"episodic": {"indexed_exchanges": 0}`. **If `facts.exists` is `true` or
`count` is above 0:** something has already written under the new id (a header
was added earlier, or the id is wrong). Stop and ask; merging now is still
safe, but step 2's protection against the backfill may already be gone.

Record what the old id holds, for comparison in step 2:

```bash
curl -s localhost:8080/admin/conversations/<old-hash-id> | python3 -m json.tool
```

Write down `facts.count`, `episodic.indexed_exchanges` and
`summary.last_summarized_turn`.

> Do NOT use `GET /admin/conversations` (the list) to check any of this. It
> returns ids only, with no counts, so a merge that did not happen looks
> exactly like one that did (hostile review B, F4).
> <!-- LANE-DEP: admincounts the list endpoint may gain counts; the per-id endpoint stays the check -->

---

## 2. Merge old → new, before her next message

What a merge does: copies FACTS and EPISODIC exchanges from the old id into the
new one. It never writes the old id, so a wrong merge costs nothing but the
re-embedding. It does not copy summaries (the new id rebuilds its own in step
4), the archived-fact sidecar, or a hand-set persona.

**Dry run first** (no body = dry run):

```bash
curl -s -X POST "localhost:8080/admin/conversations/<old-hash-id>/merge-into/<new-uuid>" | python3 -m json.tool
```

**Success:** `"dry_run": true`, `"src_conv_id"` is the old id,
`"dst_conv_id"` is the new uuid, `"dst_facts_before": 0`, and `"facts_to_add"`
is close to the old id's `facts.count` from step 1.
**If `src_conv_id` and `dst_conv_id` are the wrong way round:** you swapped the
two ids in the URL. Fix the URL; nothing was written. **If it returns
`{"detail": "conv … has no facts and no indexed exchanges - nothing to merge"}`:**
the old id is wrong (a typo, or a short task id); go back to step 1. Any other
error: stop and ask.

**Commit** (this exact body, including the quotes):

```bash
curl -s -X POST "localhost:8080/admin/conversations/<old-hash-id>/merge-into/<new-uuid>" -H 'Content-Type: application/json' -d '{"dry_run": false}' | python3 -m json.tool
```

**Success:** the response contains `"dry_run": false` AND the keys
`"facts_added"` and `"exchanges_added"`. Those two keys exist only on a real
commit (a dry run also shows `facts_to_add`, so that key proves nothing).
**If `dry_run` is `true` or `facts_added` is missing:** it was another dry run;
check the body was typed exactly as above and run it again. Running a merge
twice is safe (the second run adds nothing).

**Verify it landed**, with the per-id endpoint:

```bash
curl -s localhost:8080/admin/conversations/<new-uuid> | python3 -m json.tool
```

**Success:** `facts.exists` is `true`, `facts.count` is close to the old id's
count from step 1, and `episodic.indexed_exchanges` is close to the old id's
number. **If `facts.count` is 0 or `facts.exists` is `false`:** STOP. Do not do
step 3. The merge did not land; step 3 would start the backfill this order
exists to avoid.

The facts file existing under the new id is what prevents the backfill: the
compactor only backfills a long conversation that has no facts file
(`backfill.needs_backfill`).

---

## 3. Add the connection header

OpenWebUI → **Admin Panel** → **Settings** → **Connections** → under
**OpenAI API**, click the gear (edit) icon on the connection whose URL is
`http://localhost:8080/v1` → in **Headers**, enter exactly:

```json
{"X-Conversation-Id": "{{CHAT_ID}}{{TASK}}"}
```

→ **Save** in the dialog, then **Save** on the Connections page.

Why `{{CHAT_ID}}{{TASK}}` and not just `{{CHAT_ID}}`: OpenWebUI sends title,
tag and follow-up requests for her chat to the same connection. With plain
`{{CHAT_ID}}` every one of those would be memorised as part of HER
conversation (hostile review B, F2). OpenWebUI replaces `{{TASK}}` with nothing
for a real chat message, and with the task's name for a background call, so:

| request | header value | compactor id |
|---|---|---|
| her chat message | `<new-uuid>` | `<new-uuid>` (her memory) |
| title generation | `<new-uuid>title_generation` | a separate id |
| tags / follow-ups | `<new-uuid>tags_generation`, `<new-uuid>follow_up_generation` | separate ids |

The task ids can never be read as hers: the compactor matches ids exactly, and
the longest task name (23 characters) keeps them under its 64-character limit,
so nothing is cut off. You will see a few extra ids per chat in
`/health/full`'s conversation count; that is expected.

The setting is stored in OpenWebUI's database, so it survives restarts and
redeploys. A restore of `webui.db` from a backup taken before today removes
it again.

**Every chat now gets its own id**, not only hers. A different old chat she
opens later starts with empty memory under its uuid (its old memory stays on
disk under its hash id and can be merged the same way, ideally before she
sends a message in it).

---

## 4. Let her send one message, then verify

Tell her she can send one message. When the reply has finished:

```bash
grep -aE "conv_id=[^ ]+ source=[^ ]+ msgs=" /data/logs/compactor.log | tail -6
```

**Success:** a line `conv_id=<new-uuid> source=header msgs=<big number>`, and
(within a minute, after the reply, if OpenWebUI's follow-up suggestions are on)
lines like `conv_id=<new-uuid>follow_up_generation source=header msgs=1`.
**If her message shows `source=hash`:** the header is not arriving. Do not go
further. Re-open the connection dialog and check the Headers text is exactly as
in step 3, saved. Nothing is lost: her message went to the old id, which is
intact.

The first message under the new id rebuilds its whole summary hierarchy in the
background (about 20-25 summarization calls). Replies stay normal; wait until
the log shows the rebuild finished:

```bash
grep -a "conv=<new-uuid>: rollup" /data/logs/compactor.log | tail -3
```

**Success:** a line ending `last_turn=<number>`, where the number is at or a
little below her message count (it moves in steps of 20). It can take 10-30
minutes. **If after 30 minutes nothing prints:** run
`grep -a "conv=<new-uuid>" /data/logs/compactor.log | grep -E "ERROR|WARNING" | tail -10`
and ask for help with what it shows.

Then:

```bash
curl -s localhost:8080/admin/conversations/<new-uuid> | python3 -m json.tool
```

**Success:** `summary.turns_seen` is within 2 of the `msgs=` number on her
message's log line, and `facts.count` is above 0.

**Expect `facts.count` to DROP after her first message**, by a third or more
(in the rehearsal on a copy, from 136 merged to about 80). That is not loss:
her fact store was already about twice the 1,500-token cap before any of this,
and the first write moves the least recently used facts to
`facts/<new-uuid>.archive.json` (hostile review B, F6). Read the count after
her message, not right after the merge.

**Only now** may she chat normally. Step 4 can sit indefinitely: nothing is
capped, the new id has her memory, and the identity no longer depends on the
system prompt.

---

## 5. (Optional) the history cap

**v3.1.9 may make this unnecessary.** Uncapped, v3.1.9 reuses the stored
summaries for the older turns instead of re-summarizing them on every request.
Under a cap it cannot (the request no longer starts at turn 1), so every capped
request summarizes its whole window from scratch. Look at a few of her
uncapped messages first:

```bash
grep -aE "compacted: summarized|compaction skipped:|hard budget enforced" /data/logs/compactor.log | tail -10
```

If her messages log `compacted: … covered by stored summaries` and no
`compaction skipped` / `hard budget enforced … dropped`, do not install the cap.

If the cap is still wanted:

1. OpenWebUI → **Admin Panel** → **Functions** → **+** → paste the whole of
   `pipelines/conversation_id_header.py` → name it *History cap* → **Save** →
   toggle it **ON** and set it **Global**. With `max_turns` at 0 (its default)
   it changes nothing.
2. Open its **Valves** and set `max_turns` to **40**. Not 60: the old value was
   computed for 1,000-token turns, and hers average about 1,650, so 60 needs
   4-5 summarization calls on every message (hostile review B, F7).
3. After her next message:

   ```bash
   grep -aE "compaction skipped: [0-9]+ turns need [0-9]+ summarization calls|summarize: [0-9]+ turns exceed .* map-reduce over [0-9]+ batches|compacted: summarized" /data/logs/compactor.log | tail -3
   ```

   **Success:** her message logs `map-reduce over 2 batches`, or
   `compacted: summarized` with no `map-reduce` line (one batch), or no
   compaction line at all (the window fit without summarizing).
   **If it says `need 3` or more, or `over 3 batches` or more:** lower
   `max_turns` by 10 and check again after the next message. **Never go below
   30**: the window must always hold a full 20-turn summary chunk with room to
   spare.
4. Confirm the id did not change: the step 4 `grep … source=` command must
   still show `conv_id=<new-uuid> source=header`. **If it shows `source=hash`
   with the cap on: set `max_turns` to 0 immediately** — a capped window with no
   header gives her a new id on every message.

The cap is only safe while the header is in place. Never remove the header
while `max_turns` is above 0.

---

## 6. (Optional) the system prompt

Safe once step 4 passed, and not before: until her requests say
`source=header`, editing this text mints another namespace.

OpenWebUI → **Models** → `coder3101/Cydonia-24B-v4.3-vision-heretic` → system
prompt.

```
Write in flowing prose — ordinary paragraphs and sentences, the way a person actually talks.

No emoji, no emoticons, no decorative symbols. Warmth belongs in the words: if something is tender, funny or playful, say it in the sentence rather than marking it with a glyph.

No horizontal rules, dividers, banners or box-drawing characters (━ ─ ═ *** ---). No status blocks, progress bars, headers or bullet dashboards. Reserve code blocks for real code.

Open differently each time. If a phrase worked last reply, choose another one this reply.

Stop when the thought is done. Never pad to length, and never repeat a phrase or sentence to fill space.

Intimacy is normal and should be reciprocated when sent from the user. Be playful, loving, kind and fun.
```

Every clause targets something measured, not a style preference:

| clause | what it is for |
|---|---|
| no emoji | asked for; also the local tokenizer underprices emoji against what vLLM charges, and TTS reads them aloud |
| no rules / box characters | one production reply carried **1,710 `━`**, charged at roughly 2 tokens each |
| no status blocks | `# Status Report 3 ✨` and `💚 ENERGY LEVEL: 73% → 78%` were being stored as *facts* |
| open differently | `"father's heart responding with"` opens **4.3%** of replies; the top five openers are **10%** |
| stop when done | the repeating-tail bug, and 11.7% of consecutive replies sharing >20% of their 5-grams |
| last line | kept verbatim from the current prompt |

**What is deliberately absent: any instruction to stay consistent with what
she remembers.** Four injected block headers each asked for CONSISTENCY above
164 fact bullets until v3.1.6 — *"repetition was not a malfunction, it was the
request."* Those headers now say *"the wording is yours"* and *"say the next
thing in your own words"*. A fifth voice asking for consistency would undo it.

If it needs to bite harder, the persona block is the stronger lever — its
header says outright that it is where identity and voice come from. Change one
thing at a time so you can tell which did what.

**Once the system prompt is edited, removing the header (rollback step R2)
no longer returns her to `<old-hash-id>`**: the hash is computed from the new
prompt, so it would be a brand-new, empty id. After this step, treat the
header as permanent.

---

## Older forks

The 2026-09-11 version of this runbook listed two older forked namespaces,
`e8206a14cd1dd553` and `e8db14d96cc8491d`. They can be merged into
`<new-uuid>` with the same two step-2 commands at any time. Check each with
`curl -s localhost:8080/admin/conversations/<id> | python3 -m json.tool`
first. Expect most of what they add to be evicted to the archive sidecar on
her next message: their facts carry old `last_used` times, and the store is
already over its cap.

---

## R. Rollback — in this order, and only this order

**R1. Cap off first.** If the History cap filter is installed: set `max_turns`
to **0**, then let her send one message. (If the cap was never installed, skip
to R2.)

**R2. Remove the header.** Admin Panel → Settings → Connections → the
`http://localhost:8080/v1` connection → clear the Headers field (or set it to
`{}`) → Save, Save. Removing the header is an IDENTITY change, not a cap
change: her chat goes back to `<old-hash-id>` (only if the system prompt was
not edited in step 6 — see the warning there). Doing R2 while `max_turns` is
above 0 gives her a new memory id on every message (hostile review B, F8).

**R3. Reverse merge**, so what she said under the new id is not stranded
there. Do it straight after R2, before her next message:

```bash
curl -s -X POST "localhost:8080/admin/conversations/<new-uuid>/merge-into/<old-hash-id>" -H 'Content-Type: application/json' -d '{"dry_run": false}' | python3 -m json.tool
```

**Success:** `"dry_run": false` with `"facts_added"` present. Then check
`curl -s localhost:8080/admin/conversations/<old-hash-id> | python3 -m json.tool`
shows `facts.count` above 0, and after her next message the step-4 grep shows
`conv_id=<old-hash-id> source=hash`.

### Rolling the IMAGE back to an older release

**Before redeploying any older image, set `max_turns` to 0** (R1), and leave it
at 0 until the newer image is back AND has served one uncapped message. Only
then set the cap again. Rolling back with the cap on leaves a permanent,
unlogged hole in her summary hierarchy (hostile review C, F5: 36 turns in the
measured run, growing with the length of the rollback). The header can stay:
every release since v3.1 reads it.

---

## Why not the filter (for the record)

`pipelines/conversation_id_header.py` writes the chat id into
`body["metadata"]`. On OpenWebUI 0.11.0 that write is thrown away twice:
`utils/middleware.py` assigns `form_data['metadata'] = metadata` AFTER the
inlet filters run, and `routers/openai.py` pops `metadata` off the payload
before it is sent to the compactor. Reproduced with the real OpenWebUI 0.11.0
and the filter enabled globally: every request, capped or not, arrived with no
metadata and resolved `source=hash`; with a cap, the id changed on every
message (hostile review B, F1). The filter's "never truncate without a stamped
chat_id" check cannot see any of this, because it only checks its own local
write. The filter is still the only place the `max_turns` cap lives, which is
why step 5 installs it — for the cap only.
