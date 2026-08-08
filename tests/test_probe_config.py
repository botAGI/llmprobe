"""Tests for llmprobe.probes.config (adapter selection).

Hermetic: no network, no real inference server. Backend fixtures from
``tests/fixtures/`` are served through an ``httpx.MockTransport``.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from llmprobe.models import Backend, Provenance
from llmprobe.probes.config import read_effective_config

FIXTURES = Path(__file__).parent / "fixtures"
BASE_URL = "http://probe.test"


def _load_json(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def _load_text(name: str) -> str:
    return (FIXTURES / name).read_text()


def _client(routes: dict[str, object]) -> httpx.AsyncClient:
    """Build a client serving canned responses by URL path.

    ``routes`` maps a path to either a ``dict`` (returned as JSON 200) or a
    ``tuple`` ``(status_code, body)``. Unlisted paths return 404.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        entry = routes.get(request.url.path)
        if entry is None:
            return httpx.Response(404, text="not found")
        if isinstance(entry, dict):
            return httpx.Response(200, json=entry)
        status, body = entry
        content_type = "text/plain" if isinstance(body, str) else "application/json"
        return httpx.Response(status, text=str(body), headers={"content-type": content_type})

    return httpx.AsyncClient(base_url=BASE_URL, transport=httpx.MockTransport(handler))


async def _read(routes: dict[str, object], claimed_ctx: int | None = None):
    """Run read_effective_config and always close the client."""
    client = _client(routes)
    try:
        return await read_effective_config(client, BASE_URL, claimed_ctx)
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_vllm_adapter_wins_over_generic_on_vllm_fixtures() -> None:
    routes = {
        "/metrics": (200, _load_text("vllm_metrics.txt")),
        "/v1/models": _load_json("vllm_models.json"),
    }
    config, findings = await _read(routes)

    assert config.backend == Backend.VLLM
    assert config.n_ctx_total == 8192
    assert config.sources["n_ctx_total"] == Provenance.READ
    assert findings == []


@pytest.mark.asyncio
async def test_llamacpp_adapter_wins_over_generic_on_llamacpp_fixtures() -> None:
    routes = {
        "/props": _load_json("llamacpp_props.json"),
    }
    config, findings = await _read(routes)

    assert config.backend == Backend.LLAMACPP
    assert config.n_ctx_total == 32768
    assert config.n_ctx_per_slot == 8192
    assert config.total_slots == 4
    assert findings == []


@pytest.mark.asyncio
async def test_tie_between_llamacpp_and_ollama_resolves_to_llamacpp() -> None:
    # Both llama.cpp (/props with build_info) and Ollama (/api/tags) signals
    # are present, so both adapters detect at confidence 1.0. The fixed
    # priority order must deterministically select llamacpp.
    routes = {
        "/props": _load_json("llamacpp_props.json"),
        "/api/tags": _load_json("ollama_tags.json"),
    }
    client = _client(routes)
    try:
        config, _findings = await read_effective_config(client, BASE_URL, None)
        assert config.backend == Backend.LLAMACPP
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_tie_resolution_is_stable_across_repeated_runs() -> None:
    routes = {
        "/props": _load_json("llamacpp_props.json"),
        "/api/tags": _load_json("ollama_tags.json"),
    }
    results: set[str] = set()
    for _ in range(20):
        client = _client(routes)
        try:
            config, _ = await read_effective_config(client, BASE_URL, None)
        finally:
            await client.aclose()
        results.add(config.backend.value)

    assert results == {"llamacpp"}


@pytest.mark.asyncio
async def test_matching_nothing_falls_back_to_generic_without_raising() -> None:
    routes = {}  # every probe path returns 404.
    config, findings = await _read(routes)

    assert config.backend == Backend.GENERIC
    assert config.sources["n_ctx_total"] == Provenance.UNKNOWN
    assert findings == []
