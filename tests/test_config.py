"""Tests for the parallelism handling in llmprobe.probes.config.

Hermetic: no network. Backend responses are served through an
``httpx.MockTransport`` built from ``tests/fixtures/``.

The README guarantees "Effective config vs. what you passed": an absent
``--parallel`` flag is a server *default*, not ``1``. llama.cpp's documented
default is 4 slots, so the total ``n_ctx`` is divided by 4 to yield the honest
per-slot context rather than being treated as a single-slot value.
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


def _client(routes: dict[str, object]) -> httpx.AsyncClient:
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


async def _read(routes: dict[str, object]):
    client = _client(routes)
    try:
        return await read_effective_config(client, BASE_URL, None)
    finally:
        await client.aclose()


def _props_without_total_slots() -> dict:
    props = _load_json("llamacpp_props.json")
    props.pop("total_slots")
    return props


@pytest.mark.asyncio
async def test_absent_total_slots_defaults_to_parallel_four() -> None:
    # A llama.cpp server that omits total_slots still runs --parallel with its
    # documented default of 4. read_effective_config must not report the
    # per-slot n_ctx as if it were a single-slot total.
    routes = {"/props": _props_without_total_slots()}
    config, findings = await _read(routes)

    assert config.backend == Backend.LLAMACPP
    assert config.n_ctx_per_slot == 8192
    assert config.total_slots == 4
    assert config.n_ctx_total == 8192 * 4 == 32768
    assert config.sources["total_slots"] == Provenance.INFERRED
    assert config.sources["n_ctx_total"] == Provenance.INFERRED
    assert findings == []


@pytest.mark.asyncio
async def test_per_slot_context_is_total_divided_by_parallel() -> None:
    # The honest per-slot context is the total divided by the parallelism, not
    # the raw total. With the default of 4 the derived per-slot value must be
    # n_ctx_total // 4.
    routes = {"/props": _props_without_total_slots()}
    config, _findings = await _read(routes)

    assert config.n_ctx_total is not None
    assert config.total_slots == 4
    assert config.n_ctx_per_slot == config.n_ctx_total // config.total_slots


@pytest.mark.asyncio
async def test_advertised_total_slots_is_respected() -> None:
    # When the server does advertise total_slots, its value is authoritative
    # and the default is not applied.
    routes = {"/props": _load_json("llamacpp_props.json")}
    config, _findings = await _read(routes)

    assert config.backend == Backend.LLAMACPP
    assert config.total_slots == 4
    assert config.sources["total_slots"] == Provenance.READ
    # The slot count is authoritative; the total context derived from it was
    # not. A live b9049 server with --ctx-size 8192 logs `kv_unified = true`
    # and gives all four slots the whole 8192, so the old product of 32768 was
    # four times the truth, and /props does not publish kv_unified to tell them
    # apart.
    assert config.n_ctx_total is None
    assert config.sources["n_ctx_total"] == Provenance.UNKNOWN
