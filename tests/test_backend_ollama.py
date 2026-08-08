"""Tests for the Ollama backend adapter.

Hermetic: no network, no real inference server. Fixtures in
``tests/fixtures/`` are served through an ``httpx.MockTransport``.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from llmprobe.backends.ollama import detect, read_config
from llmprobe.models import Backend, Finding, Provenance, Severity

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def _make_client(show_fixture: str) -> httpx.AsyncClient:
    """Build a client that serves the Ollama fixtures over a mock transport."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json=_load("ollama_tags.json"))
        if request.url.path == "/api/ps":
            return httpx.Response(200, json=_load("ollama_ps.json"))
        if request.url.path == "/api/show":
            return httpx.Response(200, json=_load(show_fixture))
        return httpx.Response(404)

    return httpx.AsyncClient(
        base_url="http://ollama.test", transport=httpx.MockTransport(handler)
    )


@pytest.mark.asyncio
async def test_detect_recognizes_ollama_shape() -> None:
    client = _make_client("ollama_show_mismatch.json")
    try:
        score = await detect(client, "http://ollama.test")
        assert score == 1.0
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_detect_rejects_non_ollama_shape() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"id": "mock"}],
            },
        )

    client = httpx.AsyncClient(
        base_url="http://ownerless.test",
        transport=httpx.MockTransport(handler),
    )
    try:
        score = await detect(client, "http://ownerless.test")
        assert score == 0.0
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_mismatch_emits_context_downgrade_finding() -> None:
    client = _make_client("ollama_show_mismatch.json")
    try:
        config, findings = await read_config(client, "http://ollama.test")
    finally:
        await client.aclose()

    assert config.backend == Backend.OLLAMA
    assert config.model_id == "llama3:8b"
    assert config.n_ctx_total == 4096
    assert config.sources["n_ctx_total"] == Provenance.MEASURED

    assert findings == [
        Finding(
            severity=Severity.MISMATCH,
            code="OLLAMA_CTX_DOWNGRADE",
            advertised=32768,
            measured=4096,
            message=(
                "model 'llama3:8b' is loaded with a smaller context (4096) "
                "than its trained context (32768)"
            ),
        )
    ]


@pytest.mark.asyncio
async def test_equal_values_produce_no_finding() -> None:
    client = _make_client("ollama_show_equal.json")
    try:
        config, findings = await read_config(client, "http://ollama.test")
    finally:
        await client.aclose()

    assert config.n_ctx_total == 4096
    assert findings == []


@pytest.mark.asyncio
async def test_running_model_match_is_case_insensitive_and_trims_whitespace() -> None:
    """Model names with extra whitespace or different case still match."""

    tags = {"models": [{"name": "  Llama3:8B  "}]}
    ps = {"models": [{"name": "llama3:8b", "context_length": 4096}]}
    show = {
        "model_info": {
            "general.arch": "llama",
            "llama.context_length": 32768,
        }
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json=tags)
        if request.url.path == "/api/ps":
            return httpx.Response(200, json=ps)
        if request.url.path == "/api/show":
            assert json.loads(request.read()).get("name") == "llama3:8b"
            return httpx.Response(200, json=show)
        return httpx.Response(404)

    client = httpx.AsyncClient(
        base_url="http://ollama.test", transport=httpx.MockTransport(handler)
    )
    try:
        config, findings = await read_config(client, "http://ollama.test")
    finally:
        await client.aclose()

    assert config.model_id == "llama3:8b"
    assert config.n_ctx_total == 4096
    assert findings == [
        Finding(
            severity=Severity.MISMATCH,
            code="OLLAMA_CTX_DOWNGRADE",
            advertised=32768,
            measured=4096,
            message=(
                "model 'llama3:8b' is loaded with a smaller context (4096) "
                "than its trained context (32768)"
            ),
        )
    ]
