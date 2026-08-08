"""Tests for llmprobe.backends.llamacpp.

Hermetic: no network. The adapter is driven through ``httpx.MockTransport``
with responses derived from ``tests/fixtures/llamacpp_props.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx

from llmprobe.backends.llamacpp import detect, read_config
from llmprobe.models import Backend, Provenance

BASE_URL = "http://llamaserver"

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _load_props() -> dict:
    with open(FIXTURE_DIR / "llamacpp_props.json", encoding="utf-8") as fh:
        return json.load(fh)


def _client(props: dict | None = None, slots_status: int = 200) -> httpx.AsyncClient:
    props = props if props is not None else _load_props()

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/props":
            return httpx.Response(200, json=props)
        if path == "/slots":
            if slots_status == 200:
                return httpx.Response(200, json=[])
            return httpx.Response(slots_status, json={"error": "no slots"})
        return httpx.Response(404, json={})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=BASE_URL)


async def test_detect_returns_one_for_llamacpp() -> None:
    async with _client() as client:
        assert await detect(client, BASE_URL) == 1.0


async def test_detect_returns_zero_when_build_info_missing() -> None:
    props = _load_props()
    props.pop("build_info")
    async with _client(props=props) as client:
        assert await detect(client, BASE_URL) == 0.0


async def test_read_config_parses_total_slots() -> None:
    async with _client() as client:
        config = await read_config(client, BASE_URL)
        assert config.total_slots == 4
        assert config.sources["total_slots"] == Provenance.READ


async def test_read_config_parses_n_ctx_per_slot_with_read_provenance() -> None:
    async with _client() as client:
        config = await read_config(client, BASE_URL)
        assert config.n_ctx_per_slot == 8192
        assert config.sources["n_ctx_per_slot"] == Provenance.READ


async def test_n_ubatch_is_none_with_unknown_provenance() -> None:
    async with _client() as client:
        config = await read_config(client, BASE_URL)
        assert config.n_ubatch is None
        assert config.sources["n_ubatch"] == Provenance.UNKNOWN


async def test_n_batch_is_none_with_unknown_provenance() -> None:
    async with _client() as client:
        config = await read_config(client, BASE_URL)
        assert config.n_batch is None
        assert config.sources["n_batch"] == Provenance.UNKNOWN


async def test_slots_501_does_not_raise() -> None:
    async with _client(slots_status=501) as client:
        config = await read_config(client, BASE_URL)
        assert config is not None
        assert config.total_slots == 4


async def test_effective_config_backend_is_llamacpp() -> None:
    async with _client() as client:
        config = await read_config(client, BASE_URL)
        assert config.backend == Backend.LLAMACPP
        assert config.model_id == "mock/llama-3.1-8b"


async def test_every_numeric_field_has_provenance() -> None:
    async with _client() as client:
        config = await read_config(client, BASE_URL)
        numeric_fields = [
            "n_ctx_total",
            "n_ctx_per_slot",
            "n_batch",
            "n_ubatch",
            "total_slots",
        ]
        for field in numeric_fields:
            assert field in config.sources
            assert isinstance(config.sources[field], Provenance)


async def test_read_config_tolerates_extra_unknown_field() -> None:
    props = _load_props()
    props["some_future_unknown_field"] = {"nested": [1, 2, 3]}
    async with _client(props=props) as client:
        config = await read_config(client, BASE_URL)
        assert config is not None
        assert config.backend == Backend.LLAMACPP
        assert config.total_slots == 4
        assert config.n_ctx_per_slot == 8192


async def test_read_config_tolerates_missing_field() -> None:
    props = _load_props()
    props.pop("model")
    async with _client(props=props) as client:
        config = await read_config(client, BASE_URL)
        assert config is not None
        assert config.backend == Backend.LLAMACPP
        assert config.model_id == ""
        assert config.total_slots == 4
