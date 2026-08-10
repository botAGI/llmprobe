"""Tests for :mod:`llmprobe.tokens`.

The README promises an _exact_ token count via ``/tokenize`` when the server
exposes that endpoint. These tests drive the scripted mock server through
``httpx.ASGITransport`` — no network, no real inference server — and verify
that :func:`make_prompt_of_exactly` genuinely reaches ``/tokenize`` and only
claims exactness when the count was verified against it.
"""

from __future__ import annotations

import httpx
import pytest

from llmprobe.models import Backend
from llmprobe.tokens import make_prompt_of_exactly

from tests.mocks.server import make_mock_server

BASE_URL = "http://mock"


def _client(app) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url=BASE_URL)


@pytest.mark.asyncio
async def test_exact_count_via_tokenize() -> None:
    """When the server exposes a working ``/tokenize`` endpoint the prompt is
    verified against it, so the result must claim exactness.
    """
    server = make_mock_server(max_tokens=512, behavior="hard_error")
    async with _client(server) as client:
        prompt, exact = await make_prompt_of_exactly(
            client, BASE_URL, 17, backend=Backend.LLAMACPP
        )
    assert exact is True
    assert len(prompt.split()) == 17


@pytest.mark.asyncio
async def test_estimate_when_tokenzier_absent() -> None:
    """When ``/tokenize`` is unavailable (404) the exact-count fallback cannot
    run, so the length is an estimate and ``exact`` must be ``False`` — never
    a confident claim about a count we could not verify.
    """
    server = make_mock_server(
        max_tokens=512, behavior="hard_error", tokenize_enabled=False
    )
    async with _client(server) as client:
        prompt, exact = await make_prompt_of_exactly(
            client, BASE_URL, 17, backend=Backend.LLAMACPP
        )
    assert exact is False
    assert len(prompt.split()) <= 17
