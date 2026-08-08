"""Tests for llmprobe.backends.vllm.

Hermetic: no network. Uses httpx.MockTransport serving the recorded fixtures
in tests/fixtures/ (no real inference server).
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from llmprobe.backends.vllm import _parse_vllm_metrics, detect, read_config
from llmprobe.models import Backend, Provenance

FIXTURES = Path(__file__).parent / "fixtures"
BASE_URL = "http://vllm.test"


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
