"""
title: Zion's Light AI — Conversation ID propagation + history cap
author: Zion's Light AI project
author_url: https://github.com/MrBanana8768/zions-light-ai
funding_url: https://github.com/MrBanana8768/zions-light-ai
version: 1.1.0
required_open_webui_version: 0.4.0
license: same as parent project

OpenWebUI Function (Filter type) that does two things, in this order and
for one reason each:

  1. Propagates OpenWebUI's internal chat_id to the context-compactor so
     it has a STABLE conv_id for memory (facts, RAG, hierarchical
     summaries) instead of a hash of the conversation's opening.

  2. Optionally caps how many turns OpenWebUI resends, so the compactor
     is not handed the entire transcript on every single message.

== WHY (2) EXISTS ==

Measured in production, 2026-09-01, on a conversation at 664 messages:

    hard budget enforced: 1,128,842 -> 17,744 tokens (limit 20768)
                          dropped 651 old turn(s)

OpenWebUI re-sends the FULL history with every message. At 1.13M tokens
the compactor's hard-budget guard was discarding 651 of 659 turns on
every request, so the model received about eight turns plus a fixed
memory block — and answered nearly the same prompt every time. The user's
report was "it just gives the last response again". It was not repeating
itself; it was being starved.

Every mechanism downstream — the budget guard, the summarization cap, two
full tokenizations per request that pushed replies to 53-60 minutes — is
compensation for a payload that should never have been that large. This
valve removes the cause instead of compensating for it.

Memory is NOT affected by capping. Facts, the episodic index and the
L1/L2/L3 summaries are keyed on conv_id and live in the compactor's own
storage. They are exactly what makes trimming the transcript safe.

== THE ORDERING TRAP, ENFORCED IN CODE ==

Do not cap history until chat_id propagation is verified working.

Without (1), the compactor derives conv_id from
`sha256(system ||| first_user[:512])`. Cap the history and the FIRST USER
MESSAGE in the payload changes — which changes the hash, which mints a
brand-new conv_id and orphans every fact, embedding and summary under the
old one. Truncating on a hash-derived conv_id is a memory wipe with extra
steps.

So `max_turns` defaults to 0 (off), and `inlet` REFUSES to truncate on
any request where it did not successfully stamp chat_id. Both halves of
that are deliberate: the default means installing this filter changes
nothing until you opt in, and the refusal means a request that somehow
arrives without metadata cannot fork the conversation even if the valve
is on.

== INSTALLATION ==

1. In OpenWebUI: Settings → Admin → Functions
2. Click "+" to add a new function
3. Paste this entire file
4. Name: "Conversation ID propagation"
5. Save, then toggle ON globally (or per-model)
6. VERIFY before going further:
     grep -a "conv_id=" /data/logs/compactor.log | tail -5
   You must see `source=body_metadata.chat_id`. If it still says
   `source=hash`, stop here — the filter is not reaching the compactor,
   and turning on max_turns would fork her memory.
7. ONLY THEN, set the `max_turns` valve (100 is a reasonable start —
   that is 100 non-system messages, so ~50 exchanges).

== ROLLBACK ==

Set max_turns back to 0. Nothing persists; the next request sends the
full history again.
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
                "Max NON-SYSTEM messages to forward (100 = ~50 exchanges). "
                "0 disables capping. Do not enable until the compactor log "
                "shows source=body_metadata.chat_id — see this filter's "
                "docstring for why."
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
                return body  # nothing to propagate, and so nothing to cap

            chat_id = __metadata__.get("chat_id")
            if not chat_id:
                return body

            # (1) Stamp the stable id. This must succeed before (2) may run.
            meta = body.get("metadata")
            if not isinstance(meta, dict):
                meta = {}
                body["metadata"] = meta
            meta["chat_id"] = str(chat_id)

            # (2) Cap the history. Reachable ONLY below a successful stamp:
            # with chat_id set, conv_id no longer depends on the payload's
            # first user message, so trimming it cannot fork her memory.
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
