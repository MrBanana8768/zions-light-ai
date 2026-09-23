# Runbook — her chat shows old history, or stops part-way

The symptom: OpenWebUI opens her conversation and the history
ends weeks ago, or at some arbitrary point, while the messages since then
appear to be gone. They are not gone. This happened twice, on 2026-09-22 and
2026-09-23, and both times every message was still in the database.

Run everything below in the pod's Web Terminal. There is no `sqlite3` command
in the image; the database commands use OpenWebUI's own Python.

---

## What breaks, in one paragraph

OpenWebUI 0.11 stores a conversation twice: as a tree of messages inside the
chat's JSON document, and as rows in a `chat_message` table. The screen is
drawn by starting at the chat's `current_message_id` and walking parent links
backwards. Two things can go wrong. A **pointer** can be set to an old message
mid-history, and then the screen shows only that message and its ancestors.
A **parent link** can be missing, and then the walk stops there, orphaning
everything newer. Both have the same appearance and neither loses data.

## The two causes, and how to tell them apart

| Cause | Fingerprint |
|---|---|
| **A stale browser tab.** A tab still holding an old view posts its own copy of the history back when she sends. It overwrites the pointer, and OpenWebUI's `merge_history` lets it override stored messages. | The pointer lands on the **same** old message as a previous incident, and few or no links are broken. |
| **A failed write.** The network volume (MooseFS) stalls, a save fails half-way, and a message is stored while its parent is not. | Many broken links, spread over weeks, each at a different time. Usually accompanied by `database is locked` and `disk I/O error` in `openwebui-error.log`. |

On 2026-09-22 there were 29 broken links, from failed writes. On 2026-09-23
there were 3, and the pointer had returned to the very same 09-05 message —
a stale tab.

The permanent fixes are getting the database off the network volume, and
replacing this storage model entirely (see `V4_MATRIX_CLIENT.md`).

---

## 1. Close every tab first

**Before touching anything, close her chat on every device** — phone, tablet,
a second browser, anything left open on a screen somewhere. A tab that is
open when you repair will undo the repair the next time she sends.

If you cannot be sure you have them all, sign the other sessions out in
OpenWebUI (Settings → Account → Sessions). Their saves then fail instead of
overwriting her history.

## 2. Diagnose (read-only, changes nothing)

```bash
/app/venv/bin/python /data/scripts/repair-chat-tree.py /data/openwebui/webui.db
```

Read the first two lines and the last four. `roots 1` with a branch close to
the message count means the tree is healthy. A small `before: the branch the
UI would show is N message(s)` confirms the break. The dry run also prints
every link it would restore, with timestamps, which is how you tell the two
causes apart using the table above.

## 3. Stop the writers and keep a copy

```bash
supervisorctl stop openwebui backup && timeout 15 supervisorctl status openwebui backup
```

```bash
D=/data/forensics/pre-repair-$(date -u +%Y%m%d-%H%M%S); mkdir -p "$D" && cp -p /data/openwebui/webui.db "$D"/ && ls -la "$D"
```

## 4. Repair

```bash
/app/venv/bin/python /data/scripts/repair-chat-tree.py /data/openwebui/webui.db --apply
```

It refuses to write unless every check passes, unless the database is free of
other processes, and unless the chat is unchanged since it read it. Success
ends with the restored branch length, the flat list unchanged, and
`integrity: ok`.

**If it reports `no encoded text left in the history: False`,** a previous
repair with the retired v1 script left messages stored in their encoded form.
Clean that first, then repair:

```bash
/app/venv/bin/python /data/scripts/fix-encoded-messages.py /data/openwebui/webui.db "$(ls -d /data/forensics/pre-tree-repair-*/webui.db | tail -1)"
```

(add `--apply` once the dry run verifies), where the second path is a copy of
the database from **before** that earlier repair.

## 5. Bring it back

```bash
supervisorctl start openwebui && sleep 20 && curl -s -o /dev/null -w "%{http_code}\n" -m 10 http://127.0.0.1:3000/api/config && supervisorctl start backup
```

Wait for `200`, then hard-refresh her chat (Ctrl+F5) and confirm the history
reaches back to the beginning.

## 6. Afterwards

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

- Do not run the repair with OpenWebUI running, or with a tab open.
- Do not delete `webui.db-journal`; see RUNBOOK_DB_JOURNAL.md.
- Do not use the retired first version of the repair script. It copied message
  text in its stored encoded form (messages then render with a literal quote
  and `\n`), required a parent to be strictly earlier than its reply when
  OpenWebUI stamps both in the same second, and rebuilt the legacy flat
  `messages` list with the whole branch, which nearly doubled the row
  OpenWebUI rewrites on every send and roughly doubled how long each send
  held the database lock.

## The other script here

`close-stale-backfills.py` is unrelated to the tree, and belongs with a
v3.1.9.4 deploy. That release decides whether to backfill a conversation's
facts from its backfill record before checking whether facts already exist, so
an interrupted record re-extracts the whole history on her next message and
can evict established facts. Run it, with the compactor stopped, before
deploying v3.1.9.4:

```bash
/opt/compactor-venv/bin/python /data/scripts/close-stale-backfills.py /data/openwebui/compactor
```

It only closes records for conversations that already have facts, keeps the
original record inside the new one, and deletes nothing.
