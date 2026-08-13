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
from llmprobe.probes.config import read_effective_config

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


async def test_read_config_ignores_extra_unknown_keys() -> None:
    props = _load_props()
    props["some_future_field"] = "irrelevant"
    props["another_unknown"] = 123
    async with _client(props=props) as client:
        config = await read_config(client, BASE_URL)
        assert config is not None
        assert config.backend == Backend.LLAMACPP
        assert config.total_slots == 4
        assert config.n_ctx_per_slot == 8192


async def test_read_config_tolerates_nested_unknown_field() -> None:
    props = _load_props()
    props["some_future_unknown_field"] = {"nested": [1, 2, 3]}
    async with _client(props=props) as client:
        config = await read_config(client, BASE_URL)
        assert config is not None
        assert config.backend == Backend.LLAMACPP
        assert config.total_slots == 4
        assert config.n_ctx_per_slot == 8192


async def test_read_config_tolerates_missing_model() -> None:
    props = _load_props()
    props.pop("model")
    async with _client(props=props) as client:
        config = await read_config(client, BASE_URL)
        assert config is not None
        assert config.backend == Backend.LLAMACPP
        assert config.model_id == ""
        assert config.total_slots == 4


async def test_read_config_defaults_missing_required_fields() -> None:
    props = _load_props()
    props.pop("default_generation_settings")
    props.pop("total_slots")
    props.pop("model")
    async with _client(props=props) as client:
        config = await read_config(client, BASE_URL)
        assert config.total_slots is None
        assert config.n_ctx_per_slot is None
        assert config.model_id == ""
        assert config.sources["n_ctx_total"] == Provenance.UNKNOWN
        assert "total_slots" not in config.sources
        assert "n_ctx_per_slot" not in config.sources


async def test_read_config_handles_non_dict_props_payload() -> None:
    async with _client(props=["not", "a", "dict"]) as client:
        config = await read_config(client, BASE_URL)
        assert config.total_slots is None
        assert config.n_ctx_per_slot is None
        assert config.model_id == ""


async def test_read_config_handles_non_dict_default_settings() -> None:
    props = _load_props()
    props["default_generation_settings"] = ["not", "a", "dict"]
    async with _client(props=props) as client:
        config = await read_config(client, BASE_URL)
        assert config.n_ctx_per_slot is None
        assert "n_ctx_per_slot" not in config.sources
        assert config.total_slots == 4


async def test_integration_llamacpp_compatible_path_with_disabled_slots() -> None:
    """Full llama.cpp-compatible flow against a mock with 501 on disabled slots.

    Integration: this drives the real orchestration path
    (``read_effective_config``: detect every backend concurrently, select the
    most confident, then read its config) rather than calling the adapter's
    ``detect``/``read_config`` in isolation. The mock answers in llama.cpp
    shape — ``/props`` without ``n_batch``/``n_ubatch``, ``/slots`` present,
    and ``501`` on disabled slots — and we assert every response is handled
    correctly: the server is detected as llama.cpp, the config is read without
    raising, and the fields llama.cpp never advertises stay ``None`` with
    ``UNKNOWN`` provenance.
    """
    props = _load_props()
    assert "n_ubatch" not in props
    assert "n_batch" not in props

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/props":
            return httpx.Response(200, json=props)
        if path == "/slots":
            return httpx.Response(501, json={"error": "slots disabled"})
        return httpx.Response(404, json={})

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url=BASE_URL
    )
    try:
        config, findings = await read_effective_config(client, BASE_URL, None)
    finally:
        await client.aclose()

    assert config.backend == Backend.LLAMACPP
    assert config.total_slots == 4
    assert config.n_ctx_per_slot == 8192
    assert config.n_ubatch is None
    assert config.sources["n_ubatch"] == Provenance.UNKNOWN
    assert config.n_batch is None
    assert config.sources["n_batch"] == Provenance.UNKNOWN
    assert findings == []


async def test_total_context_is_not_invented_by_multiplying_slots() -> None:
    """Per-slot context times slot count is not the total, and is not knowable.

    Modern llama.cpp defaults to a unified KV cache, where every slot sees the
    whole context instead of owning a slice of it. Verified against a live
    b9049 server started with ``--ctx-size 8192``: the log reads
    ``n_parallel = 4 and kv_unified = true``, ``llama_context: n_ctx = 8192``
    and four slots each reporting ``n_ctx = 8192`` -- so the honest total is
    8192, while the multiplication reported 32768, inflating it fourfold.

    ``kv_unified`` is absent from ``/props`` (checked key by key on that
    server), so the answer cannot be read and must not be guessed: a confident
    guess is exactly what this tool exists to refuse.
    """
    async with _client() as client:
        config = await read_config(client, BASE_URL)

    assert config.n_ctx_per_slot is not None, "fixture must expose a per-slot context"
    assert config.total_slots is not None, "fixture must expose a slot count"
    product = config.n_ctx_per_slot * config.total_slots
    assert config.n_ctx_total != product or config.n_ctx_total is None, (
        "total context was invented by multiplying per-slot context by slots"
    )
    assert config.sources["n_ctx_total"] is Provenance.UNKNOWN
