"""
title: Zion's Light AI — history cap (its chat_id stamp is discarded by OpenWebUI 0.11.0)
author: Zion's Light AI project
author_url: https://github.com/MrBanana8768/zions-light-ai
funding_url: https://github.com/MrBanana8768/zions-light-ai
version: 1.1.1
required_open_webui_version: 0.4.0
license: same as parent project

OpenWebUI Function (Filter type). Read "WHAT THIS FILTER CAN AND CANNOT DO"
before installing it: the first job this file was written for does not work
on the OpenWebUI this project ships.

== WHAT THIS FILTER CAN AND CANNOT DO (v3.1.9, verified on OpenWebUI 0.11.0) ==

  1. It CANNOT give the compactor a stable conversation id.

     inlet() writes chat_id into body["metadata"]. On OpenWebUI 0.11.0 that
     write never leaves OpenWebUI, twice over: utils/middleware.py runs the
     inlet filters and THEN assigns `form_data['metadata'] = metadata`,
     replacing whatever a filter put there, and routers/openai.py does
     `payload.pop('metadata', None)` before the request is POSTed to the
     compactor. Reproduced with the real 0.11.0 and this filter enabled
     globally: every request, capped or not, arrived with no metadata and the
     compactor logged `source=hash` (hostile review of v3.1.7, reviewer B,
     F1). The metadata stamp in inlet() is DEAD CODE on this OpenWebUI. It is
     left in place only because it is harmless and its tests pin it; do not
     read it as doing anything.

     The route that works is configuration, not code: in OpenWebUI, Admin
     Panel -> Settings -> Connections -> the compactor connection
     (http://localhost:8080/v1) -> Headers:

         {"X-Conversation-Id": "{{CHAT_ID}}{{TASK}}"}

     `{{TASK}}` is not decoration. OpenWebUI sends its title, tag and
     follow-up generation for a chat through the same connection with the
     same chat_id; plain `{{CHAT_ID}}` would memorize every one of those as
     part of her conversation (reviewer B, F2). `{{TASK}}` is empty for a real
     chat message and the task name for a background call, so those land on
     `<uuid>title_generation` and similar ids instead. The full, ordered
     procedure - merge FIRST, then the header, then verify - is
     RUNBOOK_MEMORY_IDENTITY.md in the repository root. Follow that, not this
     docstring.

  2. It CAN cap how many turns OpenWebUI resends (`max_turns`). That is the
     only reason to install it, and only after the header route above is
     verified: the compactor log must show `source=header` for her chat.

== WHY THE CAP EXISTS ==

Measured in production, 2026-09-01, on a conversation at 664 messages:

    hard budget enforced: 1,128,842 -> 17,744 tokens (limit 20768)
                          dropped 651 old turn(s)

OpenWebUI re-sends the FULL history with every message. At 1.13M tokens
the compactor's hard-budget guard was discarding 651 of 659 turns on
every request, so the model received about eight turns plus a fixed
memory block. The user's report was "it just gives the last response
again". It was not repeating itself; it was being starved.

Memory is NOT affected by capping, PROVIDED the conversation id does not
depend on the payload. Facts, the episodic index and the L1/L2/L3 summaries
are keyed on conv_id and live in the compactor's own storage.

v3.1.9 may make the cap unnecessary: uncapped, the compactor reuses its
stored summaries for the older turns instead of re-summarizing them. Under a
cap it cannot (the window no longer starts at turn 1), so every capped
request summarizes its whole window from scratch. Check the uncapped logs
first (RUNBOOK_MEMORY_IDENTITY.md step 5).

== THE ORDERING TRAP - NOT ENFORCED BY THIS CODE ==

Without a stable id, the compactor derives conv_id from
`sha256(system ||| first_user[:512])`. Cap the history and the FIRST USER
MESSAGE in the payload changes on every exchange - which changes the hash,
which mints a brand-new conv_id on EVERY message. Truncating on a
hash-derived conv_id is a memory fork per turn.

An earlier version of this docstring said this trap was "ENFORCED IN CODE"
because inlet refuses to truncate when it did not stamp chat_id. That was
false, and it is the sentence most likely to hurt her: the check only looks
at this filter's own local write, which always succeeds and is then thrown
away by OpenWebUI (point 1). The refusal still means one true thing -
OpenWebUI had no chat_id for this request, so the header would have been
empty too - but it cannot tell whether the header is configured, and so it
cannot tell whether capping is safe. Only the compactor log can:

    grep -aE "conv_id=[^ ]+ source=[^ ]+ msgs=" /data/logs/compactor.log | tail -5

Her chat must show `source=header`. If it shows `source=hash` while
max_turns is above 0, set max_turns to 0 IMMEDIATELY.

== INSTALLING IT, FOR THE CAP ONLY ==

Prerequisite: RUNBOOK_MEMORY_IDENTITY.md steps 0-4 done, and her chat logging
`source=header`.

1. OpenWebUI: Admin Panel -> Functions -> "+"
2. Paste this entire file. Name it "History cap". Save.
3. Toggle it ON and set it Global. With max_turns at 0 (the default) it
   changes nothing.
4. Set the max_turns valve to 40 - NOT 60. The old 60 was computed for turns
   of ~1,000 tokens; hers average ~1,650, and at 60 every message needs 4-5
   summarization calls (reviewer B, F7). Valve units are NON-SYSTEM messages
   (40 = ~20 exchanges).
5. After her next message:
     grep -aE "compaction skipped: [0-9]+ turns need [0-9]+ summarization calls|summarize: [0-9]+ turns exceed .* map-reduce over [0-9]+ batches|compacted: summarized" /data/logs/compactor.log | tail -3
   Success: "map-reduce over 2 batches" (or 1), or "compacted: summarized"
   with no map-reduce line. If it needs 3 or more calls/batches, lower
   max_turns by 10 and check again. Never below 30: the window must hold a
   full 20-turn L1 chunk with room to spare.

== ROLLBACK - IN THIS ORDER ==

1. max_turns -> 0, and let her send one message.
2. Only then remove the connection header. Removing the header while
   max_turns is above 0 sends a capped window with no id: a new conv_id on
   every message (reviewer B, F8). Removing the header is an IDENTITY change,
   not a cap change.
3. Reverse-merge <uuid> -> <old-hash-id> (RUNBOOK_MEMORY_IDENTITY.md R3), so
   what she said under the uuid is not stranded there. Use the JSON BODY
   form, and confirm the response carries `facts_added` (only a real commit
   has it; a dry run's `facts_to_add` proves nothing):

     curl -s -X POST "localhost:8080/admin/conversations/<uuid>/merge-into/<old-hash-id>" -H 'Content-Type: application/json' -d '{"dry_run": false}'

Rolling the compactor IMAGE back to an older release: set max_turns to 0
BEFORE redeploying the older image, and leave it at 0 until the newer image is
back and has served one uncapped message. Rolling back with the cap on leaves
a permanent, unlogged hole in her summary hierarchy (hostile review of v3.1.7,
reviewer C, F5).

Disabling or deleting this filter only removes the cap. It does not change
the conversation id - on OpenWebUI 0.11.0 it never did.
"""

from pydantic import BaseModel, Field
from typing import Any, Optional


class Filter:
    class Valves(BaseModel):
        # Priority controls ordering when multiple filters are enabled.
        # 0 = neutral; lower runs earlier.
        priority: int = 0

        max_turns: int = Field(
            default=0,
            description=(
                "Max NON-SYSTEM messages to forward (40 = ~20 exchanges). "
                "0 disables capping. Do not enable until the compactor log "
                "shows source=header for this chat (the X-Conversation-Id "
                "connection header, RUNBOOK_MEMORY_IDENTITY.md). This filter "
                "cannot set the id itself on OpenWebUI 0.11.0 - see its "
                "docstring."
            ),
        )

    def __init__(self):
        self.type = "filter"
        self.name = "Conversation ID propagation"
        self.valves = self.Valves()

    def _cap(self, messages: list, max_turns: int) -> list:
        """The last `max_turns` non-system messages, with every system
        message kept and alternation left intact.

        SYSTEM MESSAGES ARE NEVER DROPPED, wherever they sit. They carry the
        persona; losing one to a history cap would change who she is, which
        is a far worse failure than a long payload.

        THE FIRST KEPT TURN IS ALWAYS A USER TURN. A window that opens on an
        assistant message is a shape the Mistral template refuses outright
        ("Expected last role User or Tool..." on the mirror-image case), and
        it costs the compactor a repair pass on every request. Cutting one
        extra message is free; handing downstream a broken shape is not.

        The newest message is never dropped — it is the one she just typed.
        """
        system = [m for m in messages if isinstance(m, dict) and m.get("role") == "system"]
        turns = [m for m in messages if isinstance(m, dict) and m.get("role") != "system"]
        if max_turns <= 0 or len(turns) <= max_turns:
            return messages

        kept = turns[-max_turns:]
        # Walk forward off any leading assistant turn(s).
        while kept and kept[0].get("role") == "assistant":
            kept.pop(0)
        if not kept:
            return messages  # degenerate; forward untouched rather than empty
        return system + kept

    async def inlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __metadata__: Optional[dict] = None,
        **kwargs: Any,
    ) -> dict:
        """Called by OpenWebUI before the request is forwarded.

        Every failure path returns the body UNCHANGED. A filter that raises
        breaks her chat outright, which is worse than any problem this
        filter solves.
        """
        try:
            if not __metadata__:
                return body  # no chat_id: the header would be empty too, so do not cap

            chat_id = __metadata__.get("chat_id")
            if not chat_id:
                return body

            # (1) Stamp chat_id into body.metadata. DEAD ON OpenWebUI 0.11.0:
            # middleware.py replaces form_data['metadata'] after the inlet
            # filters run and openai.py pops it before the POST, so the
            # compactor never sees this (hostile review v3.1.7, B/F1). Kept
            # because it is harmless and pinned by the tests; the working id
            # route is the X-Conversation-Id connection header.
            meta = body.get("metadata")
            if not isinstance(meta, dict):
                meta = {}
                body["metadata"] = meta
            meta["chat_id"] = str(chat_id)

            # (2) Cap the history. Reachable only when OpenWebUI supplied a
            # chat_id. That is NOT proof the compactor receives an id: this
            # check sees only the local stamp above, which OpenWebUI 0.11.0
            # discards. Capping is safe only while the compactor log shows
            # source=header for this chat; with source=hash every capped
            # request mints a new conv_id. See the docstring.
            max_turns = int(getattr(self.valves, "max_turns", 0) or 0)
            if max_turns > 0:
                messages = body.get("messages")
                if isinstance(messages, list) and messages:
                    body["messages"] = self._cap(messages, max_turns)

            return body
        except Exception:
            # Deliberately silent and total. There is no logger here worth
            # depending on, and a traceback out of inlet() is a dead chat.
            return body

    # outlet is the response-side hook; nothing to add on the way back.
