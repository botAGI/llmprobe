"""Tests for the generic OpenAI-compatible fallback backend adapter.

Hermetic: no network. Uses ``httpx.MockTransport`` serving only
``GET /v1/models`` — the sole endpoint the generic adapter consults.
"""

from __future__ import annotations

import httpx
import pytest

from llmprobe.backends.generic import (
    MODELS_HTTP_ERROR_CODE,
    MODELS_UNREACHABLE_CODE,
    detect,
    read_config,
)
from llmprobe.models import Backend, Provenance, Severity

BASE_URL = "http://generic.test"
NUMERIC_FIELDS = [
    "n_ctx_total",
    "n_ctx_per_slot",
    "n_batch",
    "n_ubatch",
    "total_slots",
]


def _client(models_status: int = 200, body: object | None = None) -> httpx.AsyncClient:
    """Build a client that serves a canned ``/v1/models`` response only."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            if models_status == 200:
                return httpx.Response(200, json=body or {"object": "list", "data": []})
            return httpx.Response(models_status, json={"error": "bad"})
        return httpx.Response(404, json={})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_detect_returns_low_confidence() -> None:
    """Detect never raises and stays strictly below any specific adapter."""
    client = _client()
    try:
        score = await detect(client, BASE_URL)
    finally:
        await client.aclose()
    assert score == 0.1
    assert score < 0.9


@pytest.mark.asyncio
async def test_detect_does_not_touch_network() -> None:
    """Detect returns regardless of how the server responds or whether routes exist."""
    client = _client(models_status=500)
    try:
        score = await detect(client, BASE_URL)
    finally:
        await client.aclose()
    assert score == 0.1


@pytest.mark.asyncio
async def test_read_config_never_raises_against_v1_models_only_server() -> None:
    """Since /v1/models is the only endpoint we use, this must never raise."""
    client = _client(body={"object": "list", "data": [{"id": "mock-model"}]})
    try:
        config, findings = await read_config(client, BASE_URL)
    finally:
        await client.aclose()
    assert config.backend == Backend.GENERIC
    assert config.model_id == "mock-model"
    assert findings == []


@pytest.mark.asyncio
async def test_read_config_survives_malformed_models_body() -> None:
    """A malformed body yields empty model id and an ERROR finding, no raise."""
    client = _client(models_status=200, body={"unexpected": True})
    try:
        config, findings = await read_config(client, BASE_URL)
    finally:
        await client.aclose()
    assert config.backend == Backend.GENERIC
    assert config.model_id == ""
    assert findings == []


@pytest.mark.asyncio
async def test_read_config_reports_non_200_http_error_finding() -> None:
    """A non-200 /v1/models response is surfaced as an ERROR finding."""
    client = _client(models_status=500, body=None)
    try:
        config, findings = await read_config(client, BASE_URL)
    finally:
        await client.aclose()
    assert config.model_id == ""
    assert len(findings) == 1
    finding = findings[0]
    assert finding.severity == Severity.ERROR
    assert finding.code == MODELS_HTTP_ERROR_CODE
    assert finding.advertised == 500
    assert "500" in finding.message


@pytest.mark.asyncio
async def test_read_config_reports_timeout_finding() -> None:
    """A transport timeout on /v1/models is surfaced as an ERROR finding."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            raise httpx.ReadTimeout("timed out", request=request)
        return httpx.Response(404, json={})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        config, findings = await read_config(client, BASE_URL)
    finally:
        await client.aclose()
    assert config.model_id == ""
    assert len(findings) == 1
    finding = findings[0]
    assert finding.severity == Severity.ERROR
    assert finding.code == MODELS_UNREACHABLE_CODE
    assert finding.advertised is None


@pytest.mark.asyncio
async def test_all_numeric_fields_none_with_unknown_provenance() -> None:
    """Every numeric capacity field is None and carries provenance UNKNOWN."""
    client = _client(body={"object": "list", "data": [{"id": "mock"}]})
    try:
        config, _ = await read_config(client, BASE_URL)
    finally:
        await client.aclose()
    for field in NUMERIC_FIELDS:
        assert getattr(config, field) is None
        assert config.sources[field] == Provenance.UNKNOWN


@pytest.mark.asyncio
async def test_every_numeric_field_has_provenance_entry() -> None:
    """Every numeric field is present in sources and is a valid Provenance."""
    client = _client(body={"object": "list", "data": [{"id": "mock"}]})
    try:
        config, _ = await read_config(client, BASE_URL)
    finally:
        await client.aclose()
    for field in NUMERIC_FIELDS:
        assert field in config.sources
        assert isinstance(config.sources[field], Provenance)
