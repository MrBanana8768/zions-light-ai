# Runbook — stop her memory forking, then put it back together

**Written 2026-09-11.** Three changes, and **the order is the whole point**.
Doing them in a different order costs memory that then has to be recovered
again.

| # | change | why it is in this position |
|---|---|---|
| 1 | Install the chat-ID filter | until this lands, *any* prompt edit mints a new memory |
| 2 | Merge the orphaned halves | there is a stable destination to merge *into* only after 1 |
| 3 | Change the system prompt | safe only after 1, because the prompt feeds the identity hash |

## The mechanism, in one paragraph

With no chat-ID header, the compactor identifies a conversation by
`sha256(system ||| first_user[:512])[:16]` — `memory.py:70`. So the system
prompt **is** part of her identity. Edit it and she gets a new memory
namespace; nothing is lost, but it is no longer reachable from the
conversation. That has already happened twice here, and every request in the
log still says `source=hash`.

---

## 1. The chat-ID fix

`pipelines/conversation_id_header.py` is an **OpenWebUI Function (Filter
type)**, not a Pipelines-server plugin. Its own header carries the full
installation runbook; this is the short form.

1. OpenWebUI → **Settings → Admin → Functions**
2. **+** → paste the entire file → name it *Conversation ID propagation*
3. **Save**, then toggle **ON** (globally, or per-model)
4. Send one message, then verify:

```bash
grep -a "conv_id=" /data/logs/compactor.log | tail -5
```

**You must see `source=body_metadata.chat_id`.** If it still says
`source=hash`, stop — the filter is not reaching the compactor, and everything
below depends on it.

5. **Note the new conv_id.** It is OpenWebUI's chat UUID. Everything from
   before this install lives under the old 16-hex hash ids and is not
   reachable from it yet.

> **Do NOT set the `max_turns` valve yet.** It needs v3.1.7 — R23 and R12 are
> the two defects that made capping unsafe, and capping before the merge is
> what would make the gap permanent. Step 10 of the file's own runbook says
> the same thing.

---

## 2. The merge

**In this deployment there are TWO orphaned namespaces, not one:**

| conv_id | holds | last written |
|---|---|---|
| `e8206a14cd1dd553` | 174 facts, watermark 2111, 6 L1 chunks | 2026-09-10 20:55 |
| `e8db14d96cc8491d` | 198 facts | currently live |

Both merge into the new UUID from step 1. Run them one at a time and verify
between.

### What a merge does and does not do

* **Merges** facts and episodic exchanges.
* **Does not merge** summaries — the destination derives its own hierarchy,
  and folding the source's in would double-count the narrative.
* **Never writes the source.** A merge that comes out wrong costs nothing but
  the re-embedding.
* **Dry run by default.**

### Path A — the endpoint (needs v3.1.5 or later on the pod)

`POST /admin/conversations/<src>/merge-into/<dst>` does not exist on v3.1.4 —
neither the route nor `portability.merge_conversation`. It arrives in
**v3.1.5**, which is step 1 of `DEPLOY_SERIES.md` and the smallest step there.

```bash
curl -s -X POST "localhost:8080/admin/conversations/<src>/merge-into/<dst>" | python3 -m json.tool
```

```bash
curl -s -X POST "localhost:8080/admin/conversations/<src>/merge-into/<dst>" -H 'Content-Type: application/json' -d '{"dry_run": false}' | python3 -m json.tool
```

**Use the BODY form, not `?dry_run=false`.** On v3.1.5 and v3.1.6.1 the query
form is read but silently ignored (R4, fixed in v3.1.7): it returns HTTP 200
with plausible counts and changes nothing. **Read the counts on the second
command, not just its status.**

### Path B — no deploy

Runs the real `merge_conversation` out of a clone of a later tag, against the
live store, while the pod keeps running v3.1.4. It does not vendor a copy of
the logic.

```bash
git clone -b v3.1.8 <repo-url> /data/hotfix
```

**The compactor must be stopped.** `memory.conv_lock` is an `asyncio.Lock` —
it excludes coroutines inside one process and nothing else, so a memory tail
writing the same facts file concurrently loses whichever write lands second.
The script refuses to run while anything answers on the compactor's port
rather than racing it.

```bash
supervisorctl stop compactor
```

```bash
/opt/compactor-venv/bin/python /data/hotfix/scripts/merge-conversations.py --src e8206a14cd1dd553 --dst <new-uuid>
```

```bash
/opt/compactor-venv/bin/python /data/hotfix/scripts/merge-conversations.py --src e8206a14cd1dd553 --dst <new-uuid> --apply
```

```bash
supervisorctl start compactor
```

It backs up the destination's facts file before writing, and prints the undo
path. Repeat for `e8db14d96cc8491d`.

### Verify before going further

```bash
curl -s localhost:8080/admin/conversations/<new-uuid> | python3 -m json.tool
```

The fact count should be close to the sum of what the two old ids held. **If
it is 0, stop** — do not set `max_turns`, because capping is what would make
the remaining gap permanent.

---

## 3. The system prompt

Safe once step 1 is verified, and not before: until `source` stops being
`hash`, editing this text mints another namespace.

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
