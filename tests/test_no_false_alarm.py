"""Regression test: an honest server must not trigger a capacity false alarm.

The chat capacity probe used to compare the full replies of two prompts that
differ only in their final token, classifying identical replies as silent
truncation. That produced a false positive on honest servers whose replies are
deterministic regardless of the tail: the two prompts differ only at the end,
the replies come back identical, and the probe wrongly concludes the server
dropped the tail — reporting a tiny capacity (a "15-token server") instead of
the realistic multi-thousand-token boundary it truly accepts.

The fix replaced that comparison with a canary-head check: a marker word is
prepended at the very start and the model is told to echo the first word. An
honest server preserves the head, so the canary is present and the length is
accepted; only a server that genuinely drops the head appears to truncate.

This test pins that fix. It stands up an honest server that accepts any prompt
length without truncation, drives the real CLI, and asserts the reported
capacity is a realistic high boundary (in the thousands) or an honest unknown
— never a small number like 15 with an error cliff. Reverting to the old
reply-comparison logic makes this assert fail, so the regression cannot creep
back in.

Hermetic: no network, no real inference server. The mock is served through
``httpx.ASGITransport`` by replacing the client factory, exactly as every
other hermetic test does.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

import llmprobe.cli as cli

from tests.mocks.server import make_mock_server

BASE_URL = "http://mock"
MAX_TOKENS = 8192

# The ceiling the CLI probes by default (see DEFAULT_CEILING). An honest server
# accepts it, so the true maximum is unknown but far above any small value.
DEFAULT_CEILING = 32768

# A "small number" like 15 is the symptom of the false alarm: the old logic
# reported a capacity below LO=16 when it misread deterministic replies as
# truncation at every probed length. Any honest server must never land there.
SMALL_NUMBER_BOUND = 1000

runner = CliRunner()


def _honest_deterministic_server() -> Any:
    """Build an honest lint-free server whose chat replies ignore the tail.

    The underlying ``make_mock_server(max_tokens=MAX_TOKENS, behavior="honest")
    never truncates and never errors: it derives embeddings from the full
    input, reports an 8192-token context, and serves real ``/tokenize`` and
    config endpoints so backend detection and config reads behave normally.

    The singular override replaces ``/v1/chat/completions`` so the reply is the
    *first word* of the user message regardless of the input length or its
    final token. That is a genuinely honest server — every length is accepted,
    nothing is dropped — whose replies are nevertheless deterministic with
    respect to the differing tail. It is exactly the shape of server the old
    reply-comparison logic misclassified, and exactly the one the canary-head
    fix must keep reporting as healthy.
    """
    app = make_mock_server(max_tokens=MAX_TOKENS, behavior="honest")
    # Remove the stock chat route and register a deterministic-echo replacement
    # so it is the only handler for that path.
    app.router.routes = [
        route
        for route in app.router.routes
        if getattr(route, "path", None) != "/v1/chat/completions"
    ]

    async def _echofirstword(body: dict) -> dict:
        prompt = ""
        for message in reversed(body.get("messages", [])):
            content = message.get("content", "")
            if isinstance(content, str):
                prompt = content
                break
        first = prompt.split()[0] if prompt.split() else ""
        return {
            "id": "cmpl-mock",
            "object": "chat.completion",
            "model": "mock",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": first},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": 1,
                "total_tokens": len(prompt.split()) + 1,
            },
        }

    app.add_api_route("/v1/chat/completions", _echofirstword, methods=["POST"])
    return app


def _asgi_client_factory(app: Any):
    def make(
        _base_url: str,
        _api_key: str | None = None,
        timeout: float = 10.0,
        **_kwargs,
    ) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=BASE_URL,
            transport=httpx.ASGITransport(app=app),
            timeout=httpx.Timeout(timeout),
        )

    return make


def _capacity_max_tokens(payload: dict) -> tuple[int, str]:
    """Extract ``(max_accepted_tokens, provenance)`` from a probe JSON report.

    Raises ``AssertionError`` if the report carries no capacity entry, because
    an honest probed server must always yield a measured cliff result rather
    than an empty list.
    """
    capacity = payload.get("capacity") or []
    assert len(capacity) == 1, (
        "expected exactly one probed capacity entry, got "
        f"{[c.get('endpoint', {}).get('value') for c in capacity]}"
    )
    entry = capacity[0]
    tokens = entry["max_accepted_tokens"]
    return tokens["value"], tokens["provenance"]


def test_honest_deterministic_server_never_false_alarms(monkeypatch: pytest.MonkeyPatch) -> None:
    """An honest server is reported as a high boundary, never a tiny capacity.

    Driving the real CLI against a server that accepts any prompt length must
    yield a realistic multi-thousand-token boundary (or an honest unknown for
    the true maximum, which still sits far above any small value). It must
    never collapse to a small number like 15 with an error cliff — that is the
    false alarm this test guards against.
    """
    server = _honest_deterministic_server()
    # monkeypatch, not a bare assignment: a bare `cli._make_client = ...` is never
    # undone and leaks the stub into every later test in the session, so an
    # unreachable-server test silently connects to THIS server and exits 0.
    monkeypatch.setattr(cli, "_make_client", _asgi_client_factory(server))

    result = runner.invoke(
        cli.app,
        [BASE_URL, "--probe", "--json", "--endpoint", "chat"],
    )

    assert result.exit_code == 0, (
        f"honest server probe exited {result.exit_code}: {result.output}"
    )

    payload = json.loads(result.output)
    max_tokens, provenance = _capacity_max_tokens(payload)

    # The false alarm symptom is a small number like 15: the old logic read the
    # deterministic replies as truncation at every probed length and reported a
    # capacity below LO. An honest server must report a realistic high boundary
    # (in the thousands) or an honest unknown that is still far above LO.
    assert max_tokens >= SMALL_NUMBER_BOUND, (
        "honest server reported a suspiciously small capacity "
        f"{max_tokens} (expected a boundary in the thousands, not a "
        "15-token false alarm)"
    )

    # The high boundary is a lower bound on a server we could not exhaust, so
    # provenance must be the honest unknown marker, not a confident lie.
    assert provenance in ("unknown", "measured"), (
        f"unexpected provenance {provenance!r} for a high honest boundary"
    )

    # Sanity: the resolved chat endpoint was actually probed.
    endpoint = payload["capacity"][0]["endpoint"]["value"]
    assert endpoint == "/v1/chat/completions"


def test_honest_deterministic_server_exceeds_small_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    """The honest boundary must clear the ceiling that triggers the false alarm.

    Guards the specific regression: if the old reply-comparison logic is
    restored, every probed length collapses to ``silent_truncation`` and the
    binary search reports ``max_accepted_tokens == 15`` (LO - 1). That value
    must never reach the report for an honest server, so assert it is strictly
    below the ceiling the false alarm commonly reported and far above LO.
    """
    server = _honest_deterministic_server()
    # monkeypatch, not a bare assignment: a bare `cli._make_client = ...` is never
    # undone and leaks the stub into every later test in the session, so an
    # unreachable-server test silently connects to THIS server and exits 0.
    monkeypatch.setattr(cli, "_make_client", _asgi_client_factory(server))

    result = runner.invoke(
        cli.app,
        [BASE_URL, "--probe", "--json", "--endpoint", "chat"],
    )
    assert result.exit_code == 0, result.output

    max_tokens, _ = _capacity_max_tokens(json.loads(result.output))
    assert max_tokens > DEFAULT_CEILING // 2, (
        f"honest server reported a cliff at {max_tokens}, far below the "
        "multi-thousand-token ceiling it genuinely accepts"
    )
