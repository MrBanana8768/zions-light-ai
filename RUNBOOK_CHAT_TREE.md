# Runbook — her chat shows old history, or stops part-way

The symptom: OpenWebUI opens her conversation and the history
ends weeks ago, or at some arbitrary point, while the messages since then
appear to be gone. They are not gone. This happened twice, on 2026-09-22 and
2026-09-23, and both times every message was still in the database.

Run everything below in the pod's Web Terminal. There is no `sqlite3` command
in the image; the database commands use OpenWebUI's own Python.

---

## What breaks, in one paragraph

This image's OpenWebUI stores a conversation twice: as a tree of messages
inside the chat's JSON document, and as rows in a `chat_message` table —
which is in fact the copy OpenWebUI itself prefers when it can (it only
falls back to the JSON to fill in what the table's own links cannot
resolve, then writes that back into the table too). The screen is drawn by
starting at the chat's `current_message_id` and walking parent links
backwards, and that SAME walk, from the SAME table-preferring copy, is what
gets sent to the model for a saved chat — this is not just a display bug.
Two things can go wrong. A **pointer** can be set to an old message
mid-history, and then the screen (and the model) sees only that message and
its ancestors. A **parent link** can be missing, and then the walk stops
there, orphaning everything newer. Both have the same appearance and
neither loses data.

## The two causes, and how to tell them apart

| Cause | Fingerprint |
|---|---|
| **A stale browser tab.** A tab still holding an old view posts its own copy of the history back when she sends. `merge_history` keeps every message either tab knows about (nothing is deleted), but the stale tab's own old pointer wins outright as long as that old message is still there — which it always is. | The pointer lands on the **same** old message as a previous incident, and few or no links are broken. |
| **A failed write.** The network volume (MooseFS) stalls, a save fails half-way, and a message is stored while its parent is not. | Many broken links, spread over weeks, each at a different time. Usually accompanied by `database is locked` and `disk I/O error` in `openwebui-error.log`. |

On 2026-09-22 there were 29 broken links, from failed writes. On 2026-09-23
there were 3, and the pointer had returned to the very same 09-05 message —
a stale tab.

The permanent fixes are getting the database off the network volume, and
replacing this storage model entirely (see `V4_MATRIX_CLIENT.md`).

---

## 0. Get the script

Copy this repo onto the pod, matching every other operator script here —
there is no ssh/scp/rsync in the image, but git is:

```bash
git clone --depth 1 --branch <tag-or-branch> \
    https://github.com/MrBanana8768/zions-light-ai.git /opt/zl-repo
```

Every command below runs from that clone, not from a lone copied file — the
script needs nothing from the compactor package (only the Python standard
library), so `/app/venv/bin/python` (OpenWebUI's own venv, used below since
this touches `webui.db`) and `/opt/compactor-venv/bin/python` both work.

## 1. Close every tab first

**Before touching anything, close her chat on every device** — phone, tablet,
a second browser, anything left open on a screen somewhere. A tab that is
open when you repair will undo the repair the next time she sends.

If you cannot be sure you have them all, sign the other sessions out in
OpenWebUI (Settings → Account → Sessions). Their saves then fail instead of
overwriting her history.

## 2. Diagnose (read-only, changes nothing)

```bash
/app/venv/bin/python /opt/zl-repo/scripts/repair-chat-tree.py /data/openwebui/webui.db
```

It defaults to HER chat id — it never guesses "the largest chat" (she has
clones close enough in size that a bigger throwaway one could silently win
that guess). Pass `--chat <id-or-prefix>` for anything else, and `--json`
for a machine-readable form.

Read the summary and the `verification` line. `roots 1` with a branch close
to the message count means the tree is healthy. Any orphan re-links printed
are what the table above uses to tell the two causes apart — each one shows
its own timestamp and where it would attach. The dry run's own exit code
says whether there is anything to do at all: `0` means nothing is pending,
`3` means something is (re-run with `--apply` once ready), `1` means look
at the output before doing anything (see step 4's encoded-text case).

## 3. Repair

```bash
/app/venv/bin/python /opt/zl-repo/scripts/repair-chat-tree.py /data/openwebui/webui.db --apply
```

It takes its OWN backup first — a plain copy, timestamped, only after
checking there is room for one — refuses to write unless every check
passes, unless the database is free of other processes and any
`-wal`/`-journal` sidecar (see "Never do these" below), and unless the chat
is unchanged since it read it. Success ends with the restored branch
length, the flat list unchanged, and `integrity: ok`. If something needs a
second look afterward, `--restore <stamp>` (the timestamp it printed when it
backed up) puts the exact original bytes back — no need to have kept a
separate copy yourself.

**If it reports `encoded text left: <more than 0>`,** a previous repair
with the retired v1 script left messages stored in their encoded form (see
"Never do these"). Clean that first, then repair:

```bash
/app/venv/bin/python /opt/zl-repo/scripts/fix-encoded-messages.py /data/openwebui/webui.db "$(ls -d /data/forensics/pre-tree-repair-*/webui.db | tail -1)" --chat <her chat id>
```

(add `--apply` once the dry run verifies), where the second path is a copy
of the database from **before** that earlier repair — this script only
restores a message from a genuine pre-repair snapshot; it cannot invent one.
If no such snapshot exists (check `/data/forensics/`), say so rather than
guessing: those messages' original text may not be recoverable this way.

## 4. Bring it back

```bash
supervisorctl start openwebui && sleep 20 && curl -s -o /dev/null -w "%{http_code}\n" -m 10 http://127.0.0.1:3000/api/config && supervisorctl start backup
```

Wait for `200`, then hard-refresh her chat (Ctrl+F5) and confirm the history
reaches back to the beginning.

## 5. Afterwards

- If the fingerprint was a stale tab, the repair holds only as long as that
  tab stays closed.
- If it was failed writes, check the volume and the error log:
  ```bash
  L=${LOG_DIR:-/data/logs}; [ -d /var/log/zla ] && L=/var/log/zla; grep -ac "database is locked" $L/openwebui-error.log; ls -la /data/openwebui/webui.db*
  ```
  A `webui.db-journal` file that persists means RUNBOOK_DB_JOURNAL.md, not
  this runbook.

---

## Never do these

- Do not run the repair with OpenWebUI running, a tab open, or a
  `-wal`/`-journal` sidecar present — the script refuses all three itself,
  but do not work around that refusal.
- Do not delete `webui.db-journal`; see RUNBOOK_DB_JOURNAL.md.
- Do not use the retired first version of the repair script (before the one
  in this repo). It copied message text in its stored encoded form
  (messages then render with a literal quote and `\n`), required a parent
  to be strictly earlier than its reply when OpenWebUI stamps both in the
  same second, and rebuilt the legacy flat `messages` list with the whole
  branch, which nearly doubled the row OpenWebUI rewrites on every send and
  roughly doubled how long each send held the database lock.
- Do not pick a chat by size ("whichever is biggest"). Both scripts here
  default to her actual chat id for exactly that reason.

## The other script that USED to be here

An earlier draft of this runbook pointed at `close-stale-backfills.py` for
the separate, unrelated v3.1.9.4 backfill-record hazard. That script is
retired, not carried into this repo: it had no stale/is-running check (so
it could close a backfill that was genuinely in progress), no
compactor-liveness check, wrote the closed state as `"complete"` rather
than `"abandoned"`, ignored the attempt cap, and took no backup of its own.
`scripts/backfill-records.py` supersedes it and closes all five gaps — use
that instead, from the same clone:

```bash
/opt/compactor-venv/bin/python /opt/zl-repo/scripts/backfill-records.py --store /data/openwebui/compactor
/opt/compactor-venv/bin/python /opt/zl-repo/scripts/backfill-records.py --store /data/openwebui/compactor --apply
```

One correction to how that hazard was described here before: it is NOT
that v3.1.9.4 "evicts established facts" by re-running a backfill over
them. Checked directly against `compactor/backfill.py`: a resumed backfill
with facts already on disk MERGES its extraction with them, and a fresh
backfill against a conversation that already has facts is refused outright
(`"backfill refused — N fact(s) already on disk"`) — v3.1.9.4 fixed the
actual pre-v3.1.9.4 bug (a stale record with live facts used to be ignored
forever, silently). The real hazard `backfill-records.py` exists for is
GPU contention: every stale `in_progress` or backed-off `failed` record
already on the volume resumes the moment its conversation is next used,
all at once, on the very first upgrade past v3.1.9.3 — competing background
vLLM extraction calls against her live chat on the pod's one GPU, not fact
loss. See CHANGELOG.md's v3.1.9.6 entry and OPERATIONS.md for the numbers
verified on a real backup.
