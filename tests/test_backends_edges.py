"""Degradation edge-case tests for the backend adapters.

Hermetic: no network, no real inference server. The adapter-selection and
effective-config read (:func:`llmprobe.probes.config.read_effective_config`)
is driven through an ``httpx.MockTransport`` that reproduces each degraded
server condition.

These tests pin the robustness contract: a live probe must never crash a
backend (no traceback, no unhandled exception) when the server returns broken
JSON, drops fields, serves empty bodies, answers 500 on every endpoint, or
times out. The tool must degrade gracefully: it selects the most honest
adapter it can, returns a valid config, and any finding it surfaces is
``severity == ERROR`` (the process exit-code contract in
:class:`llmprobe.models.ProbeReport`). A degraded server must never surface a
crash or an unexpected non-ERROR finding.
"""

from __future__ import annotations

import httpx

from llmprobe.models import Backend, Severity
from llmprobe.probes.config import read_effective_config

BASE_URL = "http://edges.test"


Route = tuple[int, "str | dict"]


def _client(routes: dict[str, Route]) -> httpx.AsyncClient:
    """Build a client serving canned responses by URL path.

    ``routes`` maps a path to a ``(status_code, body)`` tuple; unlisted paths
    return 404. A ``str`` body is served as-is (so invalid JSON can be
    simulated); a ``dict`` body is serialised to JSON.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        entry = routes.get(request.url.path)
        if entry is None:
            return httpx.Response(404, text="not found")
        status, body = entry
        if isinstance(body, dict):
            return httpx.Response(status, json=body)
        return httpx.Response(status, text=body)

    return httpx.AsyncClient(base_url=BASE_URL, transport=httpx.MockTransport(handler))


async def _read(routes: dict[str, Route]):
    """Run the effective-config read and always close the client."""
    client = _client(routes)
    try:
        return await read_effective_config(client, BASE_URL, None)
    finally:
        await client.aclose()


def _assert_only_error_findings(findings) -> None:
    """Pin the contract: every finding is ERROR severity (never a crash)."""
    for finding in findings:
        assert finding.severity == Severity.ERROR, (
            f"degraded server surfaced a non-ERROR finding: "
            f"{finding.code!r} with severity {finding.severity.value}"
        )


async def test_props_broken_json_degrades_to_error_finding() -> None:
    """A /props serving broken JSON must not crash — it degrades to ERROR.

    Broken ``/props`` defeats llama.cpp detection; combined with a malformed
    ``/v1/models`` (the generic fallback's only endpoint) the probe must
    surface an ERROR finding, not raise.
    """
    config, findings = await _read(
        {
            "/props": (200, "this is { not valid json"),
            "/v1/models": (200, "also not json"),
        }
    )
    _assert_only_error_findings(findings)
    assert config.backend == Backend.GENERIC
    assert len(findings) == 1
    assert findings[0].code == "GENERIC_MODELS_HTTP_ERROR"
    assert "invalid JSON" in findings[0].message


async def test_v1_models_missing_data_field_does_not_crash() -> None:
    """A /v1/models body that omits ``data`` must not crash vLLM's reader.

    vLLM reads ``data[0].max_model_len``; with no ``data`` field the honest
    result is an empty model id and unknown context, not a crash or a
    fabricated value.
    """
    config, findings = await _read(
        {
            "/metrics": (200, "vllm:num_requests_running 1.0"),
            "/v1/models": (200, {"object": "list"}),
        }
    )
    _assert_only_error_findings(findings)
    assert findings == []
    assert config.backend == Backend.VLLM
    assert config.model_id == ""
    assert config.n_ctx_total is None


async def test_empty_metrics_degrades_to_generic_without_error() -> None:
    """An empty /metrics body must not crash and degrades to generic.

    vLLM detection depends on the ``vllm:`` namespace in /metrics; an empty
    body defeats it and the probe honestly falls back to generic rather than
    raising on a parse error or inventing a vLLM attribution.
    """
    config, findings = await _read(
        {
            "/metrics": (200, ""),
            "/v1/models": (200, {"object": "list", "data": []}),
        }
    )
    _assert_only_error_findings(findings)
    assert findings == []
    assert config.backend == Backend.GENERIC


async def test_http_500_on_all_endpoints_yields_error_finding() -> None:
    """A server answering 500 on every endpoint yields an ERROR, not a crash.

    No backend can detect over a 500 wall; the generic fallback surfaces the
    failed read as an ERROR finding (which drives exit code 2) instead of
    raising.
    """
    config, findings = await _read(
        {
            "/props": (500, "error"),
            "/metrics": (500, "error"),
            "/api/tags": (500, "error"),
            "/v1/models": (500, {"error": "boom"}),
        }
    )
    _assert_only_error_findings(findings)
    assert config.backend == Backend.GENERIC
    assert len(findings) == 1
    assert findings[0].code == "GENERIC_MODELS_HTTP_ERROR"
    assert findings[0].advertised == 500


async def test_connection_timeout_yields_error_finding() -> None:
    """A connection timeout on every request yields ERROR, not a crash.

    When no request can complete, no backend can detect the server; the
    generic fallback surfaces the unreachable read as an ERROR finding (exit
    code 2) rather than raising.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("connection timed out", request=request)

    client = httpx.AsyncClient(
        base_url=BASE_URL, transport=httpx.MockTransport(handler)
    )
    try:
        config, findings = await read_effective_config(client, BASE_URL, None)
    finally:
        await client.aclose()

    _assert_only_error_findings(findings)
    assert config.backend == Backend.GENERIC
    assert len(findings) == 1
    assert findings[0].code == "GENERIC_MODELS_UNREACHABLE"
