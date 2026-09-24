"""One case, so the stack itself is proven before anything attacks it.

An adversarial suite that reports "everything is broken" because the
deployment never came up is worse than useless: it burns the reviewer's
trust on a result that was never about the code.
"""


def test_the_stack_is_up_and_is_the_current_tree(client):
    r = client.get("/health/full")
    assert r.status_code == 200, f"compactor not answering: {r.status_code}"
    body = r.json()
    assert body.get("checks", {}).get("vllm", {}).get("ok") is True, (
        f"the vLLM stand-in is not reachable from the compactor: {body.get('checks')}"
    )
    # v3.1.8 vocabulary. If this is missing the stack is serving stale code and
    # every finding below would be about a build nobody has - the exact trap
    # docker-compose.integration.yml carries a long comment about.
    outcomes = body.get("memory_tail", {}).get("outcomes", {})
    assert "skipped_task_traffic" in outcomes, (
        "this compactor predates v3.1.8 (no skipped_task_traffic outcome); "
        "recreate it with --force-recreate before trusting anything here"
    )
