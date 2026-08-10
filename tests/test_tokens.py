"""Tests for :mod:`llmprobe.tokens`.

The README promises an _exact_ token count via ``/tokenize`` when the server
exposes that endpoint. These tests drive the scripted mock server through
``httpx.ASGITransport`` — no network, no real inference server — and verify
that :func:`make_prompt_of_exactly` genuinely reaches ``/tokenize`` and only
claims exactness when the count was verified against it. Hermetic: no network,
no real inference server.
"""

from __future__ import annotations

import httpx

from llmprobe.models import Backend
from llmprobe.tokens import make_prompt_of_exactly

from tests.mocks.server import make_mock_server

BASE_URL = "http://mock"


def _client(app) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url=BASE_URL)


async def _count_tokens(app, prompt: str) -> int:
    """Tokenize ``prompt`` against the same mock server to independently
    verify the count returned by ``make_prompt_of_exactly``."""
    async with _client(app) as client:
        resp = await client.post(f"{BASE_URL}/tokenize", json={"content": prompt})
        if resp.status_code != 200:
            return -1
        return len(resp.json().get("tokens", []))


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


async def test_exact_count_is_server_confirmed() -> None:
    """When ``/tokenize`` is available ``make_prompt_of_exactly`` returns a
    prompt that tokenizes to exactly ``n`` on the server itself, and flags it
    ``exact=True``. The count is the server's own tokenizer's answer, not a
    guess.
    """
    server = make_mock_server(max_tokens=1024, behavior="honest")
    async with _client(server) as client:
        prompt, exact = await make_prompt_of_exactly(
            client, BASE_URL, 42, backend=Backend.LLAMACPP
        )
    count = await _count_tokens(server, prompt)
    assert exact is True
    assert count == 42


async def test_larger_exact_count() -> None:
    """Exactness holds for larger request sizes, not just small ones."""
    server = make_mock_server(max_tokens=2048, behavior="honest")
    async with _client(server) as client:
        prompt, exact = await make_prompt_of_exactly(
            client, BASE_URL, 257, backend=Backend.LLAMACPP
        )
    count = await _count_tokens(server, prompt)
    assert exact is True
    assert count == 257


async def test_estimate_when_tokenizer_absent() -> None:
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


async def test_zero_tokens_never_claimed_exact() -> None:
    """n=0 cannot be produced as an exact word count by the filler strategy
    (every candidate prompt has at least one token), so the result must be
    honest: ``exact=False`` rather than a fabricated claim of verification.
    """
    server = make_mock_server(max_tokens=1024, behavior="honest")
    async with _client(server) as client:
        prompt, exact = await make_prompt_of_exactly(
            client, BASE_URL, 0, backend=Backend.LLAMACPP
        )
    assert prompt == "token"
    assert exact is False
