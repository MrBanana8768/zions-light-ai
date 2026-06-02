"""
title: Zion's Light AI — Conversation ID propagation
author: Zion's Light AI project
author_url: https://github.com/MrBanana8768/zions-light-ai
funding_url: https://github.com/MrBanana8768/zions-light-ai
version: 1.0.0
required_open_webui_version: 0.4.0
license: same as parent project

OpenWebUI Function (Filter type) that propagates OpenWebUI's internal
chat_id to the context-compactor so it can use it as a stable
conv_id for memory operations (facts, RAG, hierarchical summaries).

Without this filter installed, the compactor falls back to a SHA256 hash
of the conversation's opening fingerprint. The hash works but has
subtle failure modes (two conversations opening identically would
collide; edits to the system prompt invalidate the ID). This filter
gives the compactor a stable, OpenWebUI-native identifier instead.

== Installation ==

1. In OpenWebUI: Settings → Admin → Functions
2. Click "+" to add a new function
3. Paste this entire file
4. Name: "Conversation ID propagation"
5. Save, then toggle ON globally (or per-model)
6. Verify in the compactor logs:
     supervisorctl tail vllm | grep conv_id
   You should see `source=body_metadata.chat_id` instead of `source=hash`.

== How it works ==

OpenWebUI's Function filters can mutate the request body via the inlet
hook before it's forwarded to the LLM/middleware. We can't easily inject
HTTP headers from in-process Functions (that would require the separate
Pipelines server), but the body is fully mutable.

We write OpenWebUI's chat_id (from `__metadata__`) into
`body["metadata"]["chat_id"]`. The compactor's `resolve_conv_id`
inspects this field as its second-preference source (after the
X-Conversation-Id header, before the hash fallback).
"""

from pydantic import BaseModel
from typing import Any, Optional


class Filter:
    class Valves(BaseModel):
        # Priority controls ordering when multiple filters are enabled.
        # 0 = neutral; lower runs earlier. No real reason to change this
        # filter's priority — it's purely additive.
        priority: int = 0

    def __init__(self):
        self.type = "filter"
        self.name = "Conversation ID propagation"
        self.valves = self.Valves()

    async def inlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __metadata__: Optional[dict] = None,
        **kwargs: Any,
    ) -> dict:
        """Called by OpenWebUI before forwarding the request to the LLM.
        Receives the OpenAI-format body and OpenWebUI's request metadata.

        We extract chat_id from metadata (OpenWebUI's internal primary
        key for the conversation) and embed it into body.metadata.chat_id
        so the compactor downstream can resolve it as the conv_id.
        """
        if not __metadata__:
            return body  # nothing to propagate

        chat_id = __metadata__.get("chat_id")
        if not chat_id:
            return body

        # Ensure body has a metadata dict, then write chat_id without
        # clobbering any other fields a different filter may have set.
        meta = body.get("metadata")
        if not isinstance(meta, dict):
            meta = {}
            body["metadata"] = meta
        meta["chat_id"] = str(chat_id)

        return body

    # outlet is the response-side hook; we don't need it for conv_id
    # propagation since the response doesn't need any conv_id info added.
    # Leaving it out per OpenWebUI's "only define the hooks you use" pattern.
