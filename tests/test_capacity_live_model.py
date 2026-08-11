"""Regression safety net: the capacity probe must use the real model name.

The probe threads the config's ``model_id`` into every request (see
``llmprobe/probes/capacity.py:probe_capacity(..., model=...)`` and the call
site in ``llmprobe/cli.py``). If that plumbing is reverted to a hardcoded
mock-derived name (``chat-mock`` / ``embed-mock`` / ``vllm-probe``), a real
server that only answers for its actual model will 404 every probe request and
the probe will fabricate a ``hard_error``/``UNKNOWN`` cliff instead of a real
measurement.

This test drives the scripted mock with ``required_model`` set so that it
returns HTTP 404 for ANY model name except the configured one. A healthy probe
that honours the caller-supplied model gets a genuine result; one that hardcodes
a mock name turns the whole probe into 404-driven errors and fails here.

Hermetic: no network, no real inference server — only the mock server through
``httpx.ASGITransport``.
"""

from __future__ import annotations

import httpx
import pytest

from llmprobe.models import Backend, CliffBehavior, Endpoint, Provenance
from llmprobe.probes.capacity import probe_capacity

from tests.mocks.server import make_mock_server

BASE_URL = "http://mock"

EMBEDDINGS = "/v1/embeddings"

CHAT = "/v1/chat/completions"

EMBED_ENDPOINT = Endpoint.EMBEDDINGS

CHAT_ENDPOINT = Endpoint.CHAT

# The model the mock server is configured to accept; any other name is 404.
REAL_MODEL = "dspark"

# The model names the probe used to hardcode before the lp-pyv fix. Each must
# be refused by the mock (proving the test is non-vacuous) and would be sent if
# the code regressed to a hardcoded value.
HARDCODED_LEAKS = ("chat-mock", "embed-mock", "vllm-probe", "default")


def _client(app) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url=BASE_URL)


@pytest.mark.asyncio
async def test_probe_uses_real_model_name_not_hardcoded_embedding() -> None:
    """An embeddings probe with a real model name must succeed, not 404.

    The server answers only for ``REAL_MODEL`` and 404s every other name. If
    the probe hardcodes a mock-derived name instead of the supplied
    ``model``, every embedding request is rejected and the probe must report a
    hard-error cliff rather than the honest server's accepted ceiling.
    """
    server = make_mock_server(
        max_tokens=8192, behavior="honest", required_model=REAL_MODEL
    )

    async with _client(server) as client:
        result = await probe_capacity(
            client,
            BASE_URL,
            EMBED_ENDPOINT,
            ceiling=8192,
            backend=Backend.LLAMACPP,
            model=REAL_MODEL,
        )

    assert result is not None
    assert result.cliff_behavior == CliffBehavior.ACCEPTED
    assert result.max_accepted_tokens == 8192
    assert result.max_accepted_source == Provenance.UNKNOWN


@pytest.mark.asyncio
async def test_probe_uses_real_model_name_not_hardcoded_chat() -> None:
    """A chat probe with a real model name must reach a measured cliff, not 404.

    Server answers only for ``REAL_MODEL``. A probe that regresses to a
    hardcoded chat model name gets 404 on every request, so the search cannot
    accept a single length and reports an ``UNKNOWN``/no measured boundary
    instead of the configured hard-error cliff at ``max_tokens``.
    """
    server = make_mock_server(
        max_tokens=512, behavior="hard_error", required_model=REAL_MODEL
    )

    async with _client(server) as client:
        result = await probe_capacity(
            client,
            BASE_URL,
            CHAT_ENDPOINT,
            ceiling=32768,
            backend=Backend.LLAMACPP,
            model=REAL_MODEL,
        )

    assert result is not None
    assert result.max_accepted_tokens == 512
    assert result.cliff_behavior == CliffBehavior.HARD_ERROR
    assert result.max_accepted_source == Provenance.MEASURED


@pytest.mark.parametrize("leak", HARDCODED_LEAKS)
def test_mock_refuses_hardcoded_model_leak(leak: str) -> None:
    """The mock is non-vacuous: every model the probe used to hardcode is a 404.

    This proves the two probe tests above would genuinely turn red if the code
    regressed to a hardcoded mock name — the mock does refuse those names, so
    passing ``REAL_MODEL`` is the only way to reach a real measurement.
    """
    from fastapi.testclient import TestClient

    server = make_mock_server(
        max_tokens=512, behavior="honest", required_model=REAL_MODEL
    )
    client = TestClient(server)

    emb = client.post(
        f"{BASE_URL}{EMBEDDINGS}",
        json={"input": "some prompt", "model": leak},
    )
    assert emb.status_code == 404
    assert emb.json()["error"]["type"] == "model_not_found"

    chat = client.post(
        f"{BASE_URL}{CHAT}",
        json={
            "model": leak,
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert chat.status_code == 404

    good = client.post(
        f"{BASE_URL}{EMBEDDINGS}",
        json={"input": "some prompt", "model": REAL_MODEL},
    )
    assert good.status_code == 200
