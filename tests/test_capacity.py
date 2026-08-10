"""Tests for :mod:`llmprobe.probes.capacity`.

Hermetic: drives the scripted mock server from ``tests/mocks/server.py``
through ``httpx.ASGITransport`` — no network, no real inference server.
"""

from __future__ import annotations

import httpx
import pytest

from llmprobe.models import Backend, CliffBehavior, Provenance
from llmprobe.probes.capacity import probe_capacity

from tests.mocks.server import make_mock_server

BASE_URL = "http://mock"

EMBEDDINGS = "/v1/embeddings"


def _client(app) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url=BASE_URL)


@pytest.mark.asyncio
async def test_hard_error_server_cliff() -> None:
    """A hard_error server: max accepted is exactly max_tokens; cliff is HARD_ERROR."""
    server = make_mock_server(max_tokens=512, behavior="hard_error")
    async with _client(server) as client:
        result = await probe_capacity(
            client, BASE_URL, EMBEDDINGS, ceiling=32768, backend=Backend.LLAMACPP
        )
    assert result.max_accepted_tokens == 512
    assert result.cliff_behavior == CliffBehavior.HARD_ERROR
    assert result.probe_requests_used <= 40


@pytest.mark.asyncio
async def test_silent_truncation_server_cliff() -> None:
    """A silent_truncation server MUST be detected — the whole point of the module.

    HTTP always returns 200, so only the vector comparison can reveal the
    silently dropped tail. If this case is not caught the product does not work.
    """
    server = make_mock_server(max_tokens=512, behavior="silent_truncation")
    async with _client(server) as client:
        result = await probe_capacity(
            client, BASE_URL, EMBEDDINGS, ceiling=32768, backend=Backend.LLAMACPP
        )
    assert result.max_accepted_tokens == 512
    assert result.cliff_behavior == CliffBehavior.SILENT_TRUNCATION


@pytest.mark.asyncio
async def test_honest_server_accepts_ceiling() -> None:
    """An honest server never errors nor truncates: everything up to ceiling is accepted."""
    server = make_mock_server(max_tokens=8192, behavior="honest")
    async with _client(server) as client:
        result = await probe_capacity(
            client, BASE_URL, EMBEDDINGS, ceiling=8192, backend=Backend.LLAMACPP
        )
    assert result.max_accepted_tokens == 8192
    assert result.cliff_behavior == CliffBehavior.ACCEPTED


@pytest.mark.asyncio
async def test_status_only_classifier_is_insufficient() -> None:
    """Prove the detector is not vacuous: a status-only classifier cannot
    distinguish silent truncation from honest.

    A server that silently truncates and an honest server both return HTTP 200
    for an oversized input. Any classifier that looks only at the status code
    therefore reports BOTH as ``accepted`` — failing to surface case 2. Only a
    classifier that inspects the response body (the vector comparison) can tell
    them apart.
    """
    trunc = make_mock_server(max_tokens=512, behavior="silent_truncation")
    honest = make_mock_server(max_tokens=512, behavior="honest")

    async def status_only_max(app) -> tuple[int, str]:
        # A naive status-only probe: the largest n that still returns HTTP 200.
        # Both servers 200 for every n, so both "accept" far beyond the cliff.
        n = 512 + 100
        async with _client(app) as c:
            resp = await c.post(
                f"{BASE_URL}{EMBEDDINGS}",
                json={"input": " ".join(["tok"] * n), "model": "mock"},
            )
        if resp.status_code != 200:
            return n - 1, "hard_error"
        return n, "accepted"

    trunc_result = await status_only_max(trunc)
    honest_result = await status_only_max(honest)

    # Both returned HTTP 200 for an oversized input, so the status-only rule
    # reports both identically — it cannot detect silent truncation at all.
    assert trunc_result == honest_result
    assert trunc_result[1] == "accepted"
    assert honest_result[1] == "accepted"
    assert trunc_result[0] > 512  # both "accept" beyond the real cliff


@pytest.mark.asyncio
async def test_below_lo_capacity_is_reported_as_unmeasured() -> None:
    """A server that rejects every probed length (capacity below LO) must not
    report a never-probed integer as ``measured``.

    The binary search only probes lengths ``>= LO``; when every probe is
    rejected the ``max_accepted_tokens`` value falls below LO and was never
    actually probed. Its provenance MUST be ``unknown`` so the report does not
    claim a measurement it does not have.
    """
    server = make_mock_server(max_tokens=10, behavior="hard_error")
    async with _client(server) as client:
        result = await probe_capacity(
            client, BASE_URL, EMBEDDINGS, ceiling=32768, backend=Backend.LLAMACPP
        )
    assert result.max_accepted_source == Provenance.UNKNOWN
    assert result.cliff_behavior == CliffBehavior.HARD_ERROR


@pytest.mark.asyncio
async def test_token_count_exact_when_tokenizer_available() -> None:
    """When the server exposes a working ``/tokenize`` endpoint the probed
    lengths are verifiable, so ``token_count_exact`` must be ``True``.
    """
    server = make_mock_server(max_tokens=512, behavior="hard_error")
    async with _client(server) as client:
        result = await probe_capacity(
            client, BASE_URL, EMBEDDINGS, ceiling=32768, backend=Backend.LLAMACPP
        )
    assert result.token_count_exact is True


@pytest.mark.asyncio
async def test_token_count_is_estimate_when_tokenizer_unavailable() -> None:
    """When ``/tokenize`` is unavailable (the exact-count fallback cannot run)
    the lengths are nominal estimates and ``token_count_exact`` must be
    ``False`` — never a confident guess about a count we could not verify.
    """
    server = make_mock_server(
        max_tokens=512, behavior="hard_error", tokenize_enabled=False
    )
    async with _client(server) as client:
        result = await probe_capacity(
            client, BASE_URL, EMBEDDINGS, ceiling=32768, backend=Backend.LLAMACPP
        )
    assert result.token_count_exact is False


@pytest.mark.asyncio
async def test_vllm_prompt_tokens_yields_exact_count() -> None:
    """For a vLLM backend that reports a matching ``usage.prompt_tokens`` the
    probed lengths are verifiable against the server's own count, so
    ``token_count_exact`` must be ``True``.

    This is the README promise: 'Exact count via usage.prompt_tokens (vLLM)'.
    The mock server reports ``usage.prompt_tokens`` equal to the exact length
    we asked for; the probe must trust that field and report the count as
    exact rather than falling back to an approximation.
    """
    server = make_mock_server(max_tokens=512, behavior="hard_error")
    async with _client(server) as client:
        result = await probe_capacity(
            client, BASE_URL, EMBEDDINGS, ceiling=32768, backend=Backend.VLLM
        )
    assert result.max_accepted_tokens == 512
    assert result.cliff_behavior == CliffBehavior.HARD_ERROR
    assert result.token_count_exact is True
