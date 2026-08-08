"""Tests for llmprobe.tokens.

Hermetic: no network. Uses httpx.MockTransport against a scripted /tokenize
that counts whitespace-separated words, matching tests/mocks/server.py.
"""

from __future__ import annotations

import json

import httpx
import pytest

from llmprobe.models import Backend
from llmprobe.tokens import MAX_ATTEMPTS, make_prompt_of_exactly

BASE_URL = "http://tokens.test"


def _client(token_handler) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path != "/tokenize":
            return httpx.Response(404, text="not found")
        return token_handler(request)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _count_words_handler(requests: list[int]):
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(1)
        content = json.loads(request.content).get("content", "")
        return httpx.Response(200, json={"tokens": content.split()})

    return handler


@pytest.mark.asyncio
async def test_exact_777_is_verified() -> None:
    requests: list[int] = []
    client = _client(_count_words_handler(requests))
    try:
        prompt, exact = await make_prompt_of_exactly(
            client, BASE_URL, 777, Backend.LLAMACPP
        )
    finally:
        await client.aclose()

    assert exact is True
    assert len(prompt.split()) == 777
    assert len(requests) <= MAX_ATTEMPTS


@pytest.mark.asyncio
async def test_missing_tokenize_returns_approximate() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="not found")

    client = _client(handler)
    try:
        prompt, exact = await make_prompt_of_exactly(
            client, BASE_URL, 777, Backend.VLLM
        )
    finally:
        await client.aclose()

    assert exact is False
    assert len(prompt.split()) == 777


@pytest.mark.asyncio
async def test_adjust_loop_caps_at_max_attempts() -> None:
    # A tokenizer that never reports the target: the loop must stop after
    # at most MAX_ATTEMPTS requests instead of spinning forever.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"tokens": ["x"] * 100})

    requests: list[int] = []

    def counting_handler(request: httpx.Request) -> httpx.Response:
        requests.append(1)
        return handler(request)

    client = _client(counting_handler)
    try:
        prompt, exact = await make_prompt_of_exactly(
            client, BASE_URL, 777, Backend.GENERIC
        )
    finally:
        await client.aclose()

    assert exact is False
    assert len(requests) <= MAX_ATTEMPTS
    assert len(requests) == MAX_ATTEMPTS
