"""Tests for llmprobe.backends.vllm.

Hermetic: no network. Uses httpx.MockTransport serving the recorded fixtures
in tests/fixtures/ (no real inference server).
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from llmprobe.backends.vllm import (
    _parse_vllm_metrics,
    detect,
    extract_prompt_tokens,
    read_config,
)
from llmprobe.models import Backend, Provenance
from llmprobe.probes.capacity import probe_capacity

from tests.mocks.server import make_mock_server

FIXTURES = Path(__file__).parent / "fixtures"
BASE_URL = "http://vllm.test"
EMBEDDINGS = "/v1/embeddings"


def _fixture_text(name: str) -> str:
    return (FIXTURES / name).read_text()


def _client(routes: dict[str, object]) -> httpx.AsyncClient:
    """Build a client that serves canned responses by URL path.

    ``routes`` maps a path to an ``(int, str)`` ``(status_code, body)`` tuple.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        entry = routes.get(request.url.path)
        if entry is None:
            return httpx.Response(404, text="not found")
        status, body = entry
        return httpx.Response(status, text=body)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_detect_high_on_vllm_metrics() -> None:
    metrics = _fixture_text("vllm_metrics.txt")
    client = _client({"/metrics": (200, metrics)})
    try:
        score = await detect(client, BASE_URL)
    finally:
        await client.aclose()
    assert score > 0.8


@pytest.mark.asyncio
async def test_detect_high_on_realistic_float_metrics() -> None:
    # Live vLLM instances emit gauge values as floats (e.g. 3.0, 1.0), not
    # integers. Detection must not depend on an integer-looking sample, and
    # must trigger on the vllm: namespace alone even when only a single gauge
    # is exposed.
    metrics = "\n".join(
        [
            "# HELP vllm:num_requests_running Requests currently running.",
            "# TYPE vllm:num_requests_running gauge",
            "vllm:num_requests_running 3.0",
        ]
    )
    client = _client({"/metrics": (200, metrics)})
    try:
        score = await detect(client, BASE_URL)
    finally:
        await client.aclose()
    assert score > 0.8


@pytest.mark.asyncio
async def test_detect_high_on_cache_config_info_labels() -> None:
    # Detection must hold even when the only vllm: line carries a label block
    # attached to the metric name, which is how vLLM emits cache_config_info.
    metrics = (
        'vllm:cache_config_info{block_size="16",cache_dtype="auto",'
        'gpu_memory_utilization="0.9",num_gpu_blocks="4096",'
        'num_cpu_blocks="256",swap_space_bytes="4294967296"} 1.0'
    )
    client = _client({"/metrics": (200, metrics)})
    try:
        score = await detect(client, BASE_URL)
    finally:
        await client.aclose()
    assert score > 0.8


@pytest.mark.asyncio
async def test_detect_high_on_live_full_dump() -> None:
    # Regression: a real live vLLM /metrics dump is hundreds of lines wide and
    # mixes labeled lines (labels attached to the name, no space before '{'),
    # float gauges, counters and +Inf histogram buckets. Detection must hold on
    # this full realistic form and must not depend on any single metric name.
    metrics = _fixture_text("vllm_metrics_live.txt")
    client = _client({"/metrics": (200, metrics)})
    try:
        score = await detect(client, BASE_URL)
    finally:
        await client.aclose()
    assert score > 0.8


@pytest.mark.asyncio
async def test_detect_zero_on_empty_metrics_body() -> None:
    # A live run must not raise when /metrics returns an empty body or one that
    # carries only comment/blank lines: detection resolves to non-vLLM instead
    # of crashing on a parsing error.
    metrics = "\n".join(
        [
            "# HELP num_requests_running Requests currently running.",
            "# TYPE num_requests_running gauge",
        ]
    )
    client = _client({"/metrics": (200, metrics)})
    try:
        score = await detect(client, BASE_URL)
    finally:
        await client.aclose()
    assert score == 0.0


@pytest.mark.asyncio
async def test_detect_zero_on_whitespace_metrics_body() -> None:
    client = _client({"/metrics": (200, "\n\n   \n  # only a comment\n")})
    try:
        score = await detect(client, BASE_URL)
    finally:
        await client.aclose()
    assert score == 0.0


@pytest.mark.asyncio
async def test_read_config_empty_metrics_is_backend_agnostic() -> None:
    # read_config must not raise a parsing error when the metrics body is empty
    # or comment-only; the config is still built from /v1/models.
    models = _fixture_text("vllm_models.json")
    metrics = "# HELP vllm:num_requests_running Requests running.\n"
    client = _client(
        {
            "/v1/models": (200, models),
            "/metrics": (200, metrics),
        }
    )
    try:
        config = await read_config(client, BASE_URL)
    finally:
        await client.aclose()

    assert config.backend == Backend.VLLM
    assert config.n_ctx_total == 8192
    assert config.sources["n_ctx_total"] == Provenance.READ


@pytest.mark.asyncio
async def test_read_config_keeps_max_model_len_on_live_dump() -> None:
    # The reason detection matters: on the live dump the server must resolve to
    # the vLLM backend so max_model_len is preserved as n_ctx_total. Falling
    # back to generic would drop it to None with provenance UNKNOWN.
    models = _fixture_text("vllm_models.json")
    metrics = _fixture_text("vllm_metrics_live.txt")
    client = _client(
        {
            "/v1/models": (200, models),
            "/metrics": (200, metrics),
        }
    )
    try:
        config = await read_config(client, BASE_URL)
    finally:
        await client.aclose()

    assert config.backend == Backend.VLLM
    assert config.n_ctx_total == 8192
    assert config.sources["n_ctx_total"] == Provenance.READ


@pytest.mark.asyncio
async def test_detect_zero_on_plain_text_metrics() -> None:
    # A metrics endpoint that is reachable but carries no vllm: namespace is
    # not a vLLM server, regardless of how many data lines it exposes.
    plain = "\n".join(
        [
            "# HELP num_requests_running Requests running.",
            "# TYPE num_requests_running gauge",
            "num_requests_running 3.0",
        ]
    )
    client = _client({"/metrics": (200, plain)})
    try:
        score = await detect(client, BASE_URL)
    finally:
        await client.aclose()
    assert score == 0.0


@pytest.mark.asyncio
async def test_detect_zero_on_llamacpp_metrics() -> None:
    # llama.cpp-style metrics carry no vllm: prefix.
    llamacpp_metrics = "\n".join(
        [
            'llamacpp:llm_prompt_tokens_total 17',
            'llamacpp:llm_tokens_predicted_total 31',
            'llamacpp:slots_idle 0',
        ]
    )
    client = _client({"/metrics": (200, llamacpp_metrics)})
    try:
        score = await detect(client, BASE_URL)
    finally:
        await client.aclose()
    assert score == 0.0


@pytest.mark.asyncio
async def test_detect_zero_when_metrics_missing() -> None:
    client = _client({})
    try:
        score = await detect(client, BASE_URL)
    finally:
        await client.aclose()
    assert score == 0.0


@pytest.mark.asyncio
async def test_read_config_maps_max_model_len_to_n_ctx_total() -> None:
    models = _fixture_text("vllm_models.json")
    metrics = _fixture_text("vllm_metrics.txt")
    client = _client(
        {
            "/v1/models": (200, models),
            "/metrics": (200, metrics),
        }
    )
    try:
        config = await read_config(client, BASE_URL)
    finally:
        await client.aclose()

    assert config.backend == Backend.VLLM
    assert config.model_id == "meta-llama/Llama-3-8B-Instruct"
    assert config.n_ctx_total == 8192
    assert config.sources["n_ctx_total"] == Provenance.READ


def test_parse_vllm_metrics_empty_string_returns_no_metrics() -> None:
    # A metrics endpoint that returns an empty body must be handled without
    # raising and must not be misidentified as vLLM.
    result = _parse_vllm_metrics("")
    assert result == {}


def test_parse_vllm_metrics_comments_only_returns_no_metrics() -> None:
    # A body containing only HELP/TYPE comment lines carries no vllm: samples,
    # so it must be handled without raising and must not be misidentified.
    comments_only = "\n".join(
        [
            "# HELP num_requests_running Requests running.",
            "# TYPE num_requests_running gauge",
        ]
    )
    result = _parse_vllm_metrics(comments_only)
    assert result == {}


def test_extract_prompt_tokens_reads_usage_count() -> None:
    # vLLM reports usage.prompt_tokens on its responses; the exact count the
    # server itself reported must be read back verbatim.
    payload = {
        "object": "list",
        "data": [{"object": "embedding", "embedding": [0.1], "index": 0}],
        "usage": {"prompt_tokens": 16, "total_tokens": 16},
    }
    assert extract_prompt_tokens(payload) == 16


def test_extract_prompt_tokens_none_when_missing() -> None:
    assert extract_prompt_tokens({}) is None
    assert extract_prompt_tokens({"usage": {}}) is None
    assert extract_prompt_tokens(None) is None


def test_extract_prompt_tokens_none_when_not_int() -> None:
    payload = {"usage": {"prompt_tokens": "16"}}
    assert extract_prompt_tokens(payload) is None


@pytest.mark.asyncio
async def test_read_config_derives_slots_from_cache_config() -> None:
    # The cache_config_info metric exposes the KV-cache geometry
    # (num_gpu_blocks=6400, block_size=16); total_slots and n_ctx_per_slot must
    # be derived from it rather than left unknown.
    models = _fixture_text("vllm_models.json")
    metrics = _fixture_text("vllm_metrics_live.txt")
    client = _client(
        {
            "/v1/models": (200, models),
            "/metrics": (200, metrics),
        }
    )
    try:
        config = await read_config(client, BASE_URL)
    finally:
        await client.aclose()

    assert config.total_slots == 6400
    assert config.n_ctx_per_slot == 16
    assert config.sources["total_slots"] == Provenance.INFERRED
    assert config.sources["n_ctx_per_slot"] == Provenance.INFERRED


@pytest.mark.asyncio
async def test_read_config_slots_default_zero_when_no_cache_config() -> None:
    # When the metrics carry no cache_config_info label we still report a
    # default of 0 with inferred provenance rather than an unknown marker.
    models = _fixture_text("vllm_models.json")
    metrics = "\n".join(
        [
            "# HELP vllm:num_requests_running Requests currently running.",
            "# TYPE vllm:num_requests_running gauge",
            "vllm:num_requests_running 2.0",
        ]
    )
    client = _client(
        {
            "/v1/models": (200, models),
            "/metrics": (200, metrics),
        }
    )
    try:
        config = await read_config(client, BASE_URL)
    finally:
        await client.aclose()

    assert config.total_slots == 0
    assert config.n_ctx_per_slot == 0
    assert config.sources["total_slots"] == Provenance.INFERRED
    assert config.sources["n_ctx_per_slot"] == Provenance.INFERRED


@pytest.mark.asyncio
async def test_read_config_slots_default_zero_when_metrics_empty() -> None:
    models = _fixture_text("vllm_models.json")
    client = _client(
        {
            "/v1/models": (200, models),
            "/metrics": (200, ""),
        }
    )
    try:
        config = await read_config(client, BASE_URL)
    finally:
        await client.aclose()

    assert config.total_slots == 0
    assert config.n_ctx_per_slot == 0
    assert config.sources["total_slots"] == Provenance.INFERRED
    assert config.sources["n_ctx_per_slot"] == Provenance.INFERRED


@pytest.mark.asyncio
async def test_vllm_probe_uses_prompt_tokens_exact_count() -> None:
    """A vLLM server that reports usage.prompt_tokens drives an exact count.

    The mock server carries no /tokenize endpoint (tokenize_enabled=False), so
    the /tokenize path cannot vouch for exactness; only the vLLM
    usage.prompt_tokens field can. If the probe used that exact count the result
    must report token_count_exact=True; if it silently fell back to an
    approximation it would be False and the claimed exactness would be lost.
    """
    server = make_mock_server(
        max_tokens=512, behavior="hard_error", tokenize_enabled=False
    )
    transport = httpx.ASGITransport(app=server)
    async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as client:
        result = await probe_capacity(
            client,
            BASE_URL,
            EMBEDDINGS,
            ceiling=32768,
            backend=Backend.VLLM,
        )
    assert result.token_count_exact is True
