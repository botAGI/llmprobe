"""Tests for :mod:`llmprobe.probes.capacity`.

Hermetic: drives the scripted mock server from ``tests/mocks/server.py``
through ``httpx.ASGITransport`` — no network, no real inference server.
"""

from __future__ import annotations

import re
from pathlib import Path

import httpx
import pytest

from llmprobe.models import Backend, CliffBehavior, Endpoint, Provenance
from llmprobe.probes.capacity import probe_capacity

from tests.mocks.server import make_mock_server

BASE_URL = "http://mock"

GOLDEN_DIR = Path(__file__).parent / "golden"

EMBEDDINGS = "/v1/embeddings"

CHAT = "/v1/chat/completions"

EMBED_ENDPOINT = Endpoint.EMBEDDINGS

CHAT_ENDPOINT = Endpoint.CHAT


def _client(app) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url=BASE_URL)


@pytest.mark.asyncio
async def test_hard_error_server_cliff() -> None:
    """A hard_error server: max accepted is exactly max_tokens; cliff is HARD_ERROR."""
    server = make_mock_server(max_tokens=512, behavior="hard_error")
    async with _client(server) as client:
        result = await probe_capacity(
            client, BASE_URL, EMBED_ENDPOINT, ceiling=32768, backend=Backend.LLAMACPP
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
            client, BASE_URL, EMBED_ENDPOINT, ceiling=32768, backend=Backend.LLAMACPP
        )
    assert result.max_accepted_tokens == 512
    assert result.cliff_behavior == CliffBehavior.SILENT_TRUNCATION


@pytest.mark.asyncio
async def test_honest_server_accepts_ceiling() -> None:
    """An honest server never errors nor truncates: everything up to ceiling is accepted."""
    server = make_mock_server(max_tokens=8192, behavior="honest")
    async with _client(server) as client:
        result = await probe_capacity(
            client, BASE_URL, EMBED_ENDPOINT, ceiling=8192, backend=Backend.LLAMACPP
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
            client, BASE_URL, EMBED_ENDPOINT, ceiling=32768, backend=Backend.LLAMACPP
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
            client, BASE_URL, EMBED_ENDPOINT, ceiling=32768, backend=Backend.LLAMACPP
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
            client, BASE_URL, EMBED_ENDPOINT, ceiling=32768, backend=Backend.LLAMACPP
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
            client, BASE_URL, EMBED_ENDPOINT, ceiling=32768, backend=Backend.VLLM
        )
    assert result.max_accepted_tokens == 512
    assert result.cliff_behavior == CliffBehavior.HARD_ERROR
    assert result.token_count_exact is True


@pytest.mark.asyncio
async def test_chat_hard_error_server_cliff() -> None:
    """A chat hard_error server: max accepted is exactly max_tokens; cliff is HARD_ERROR."""
    server = make_mock_server(max_tokens=512, behavior="hard_error")
    async with _client(server) as client:
        result = await probe_capacity(
            client, BASE_URL, CHAT_ENDPOINT, ceiling=32768, backend=Backend.LLAMACPP
        )
    assert result.max_accepted_tokens == 512
    assert result.cliff_behavior == CliffBehavior.HARD_ERROR
    assert result.max_accepted_source == Provenance.MEASURED


@pytest.mark.asyncio
async def test_chat_silent_truncation_server_cliff() -> None:
    """A chat silent_truncation server MUST be detected via the head canary.

    The chat endpoint always returns HTTP 200; only the canary check can reveal
    that the head was silently dropped beyond the limit. If this case is not
    caught the chat capacity probe does not work.
    """
    server = make_mock_server(max_tokens=512, behavior="silent_truncation")
    async with _client(server) as client:
        result = await probe_capacity(
            client, BASE_URL, CHAT_ENDPOINT, ceiling=32768, backend=Backend.LLAMACPP
        )
    assert result.max_accepted_tokens == 512
    assert result.cliff_behavior == CliffBehavior.SILENT_TRUNCATION
    assert result.max_accepted_source == Provenance.MEASURED


@pytest.mark.asyncio
async def test_chat_honest_server_accepts_ceiling() -> None:
    """A chat honest server never errors nor truncates: ceiling is accepted."""
    server = make_mock_server(max_tokens=8192, behavior="honest")
    async with _client(server) as client:
        result = await probe_capacity(
            client, BASE_URL, CHAT_ENDPOINT, ceiling=8192, backend=Backend.LLAMACPP
        )
    assert result.max_accepted_tokens == 8192
    assert result.cliff_behavior == CliffBehavior.ACCEPTED


@pytest.mark.asyncio
async def test_silent_truncation_two_prompts_different_final_token() -> None:
    """The README promise: two prompts differing only in their final token
    expose silent truncation.

    A silent_truncation server derives its embedding only from the first
    ``max_tokens`` tokens, so two oversized prompts that differ ONLY in their
    final token come back identical — the differing tail was silently dropped.
    The SAME two prompts against an honest server must DIFFER, otherwise the
    two-prompt method would be vacuous: if honestly-different prompts always
    produced identical vectors it could never surface the truncation.
    """
    n = 512 + 64  # above the cliff, so the differing tail is the only distinction

    final_a = "llmprobeFinalA"
    final_b = "llmprobeFinalB"
    prompt_a = " ".join(["tok"] * (n - 1) + [final_a])
    prompt_b = " ".join(["tok"] * (n - 1) + [final_b])
    assert prompt_a.split()[-1] != prompt_b.split()[-1]

    async def embed(app, prompt: str) -> list[float]:
        async with _client(app) as client:
            resp = await client.post(
                f"{BASE_URL}{EMBEDDINGS}",
                json={"input": prompt, "model": "embed-mock"},
            )
        assert resp.status_code == 200
        return list(resp.json()["data"][0]["embedding"])

    trunc = make_mock_server(max_tokens=512, behavior="silent_truncation")
    honest = make_mock_server(max_tokens=512, behavior="honest")

    trunc_a = await embed(trunc, prompt_a)
    trunc_b = await embed(trunc, prompt_b)
    honest_a = await embed(honest, prompt_a)
    honest_b = await embed(honest, prompt_b)

    # Silent truncation discards the differing tail -> the two embeddings are
    # identical regardless of the different final token.
    assert trunc_a == trunc_b
    # Honest parsing preserves the differing tail -> the two embeddings differ,
    # proving the two-prompt method is a meaningful (non-vacuous) signal.
    assert honest_a != honest_b


@pytest.mark.asyncio
async def test_silent_truncation_distinguishes_last_token() -> None:
    """The embedding comparison must be sensitive to the final token.

    README promise: two prompts differing only in their last token are the
    probe for silent truncation. The comparison is only trustworthy if the
    server's embedding genuinely reflects the last token of a short prompt —
    if it always discarded or ignored the tail, differing final tokens would
    collapse to identical vectors and the truncation check would be vacuous.

    We keep the prompt length *below* the cliff so a silent_truncation server
    (which only drops tokens beyond ``max_tokens``) keeps the whole input, and
    the only difference between the two requests is the final token. The
    embeddings MUST differ, proving the mock can distinguish last tokens and
    therefore that an identical result on an oversized prompt really means the
    tail was truncated.
    """
    server = make_mock_server(max_tokens=512, behavior="silent_truncation")

    async def embed(prompt: str) -> list[float]:
        async with _client(server) as client:
            resp = await client.post(
                f"{BASE_URL}{EMBEDDINGS}",
                json={"input": prompt, "model": "embed-mock"},
            )
        assert resp.status_code == 200
        return list(resp.json()["data"][0]["embedding"])

    n = 256  # well below the 512 cliff, so no tail is dropped
    final_a = "llmprobeDistA"
    final_b = "llmprobeDistB"
    prompt_a = " ".join(["tok"] * (n - 1) + [final_a])
    prompt_b = " ".join(["tok"] * (n - 1) + [final_b])
    assert prompt_a.split()[-1] != prompt_b.split()[-1]

    emb_a = await embed(prompt_a)
    emb_b = await embed(prompt_b)

    assert emb_a != emb_b


@pytest.mark.asyncio
async def test_binary_search_probe_request_count_is_logarithmic() -> None:
    """The capacity probe MUST binary-search over input length, not scan
    linearly: it must locate a cliff deep inside ``[LO, ceiling]`` in a number
    of probe requests that scales with ``log2(ceiling)``, not with the cliff
    position.

    README promises "binary search to the actual cliff, per endpoint". A linear
    scan from ``LO`` would cost O(cliff) requests; a binary probe must cost
    O(log2(ceiling)). We place the cliff at an awkward, non-power-of-two depth
    to rule out any luck-of-the-grid pass and assert the probe used a strictly
    sub-linear number of requests.
    """
    cliff = 5003
    server = make_mock_server(max_tokens=cliff, behavior="hard_error")
    async with _client(server) as client:
        result = await probe_capacity(
            client, BASE_URL, EMBED_ENDPOINT, ceiling=32768, backend=Backend.LLAMACPP
        )
    assert result.max_accepted_tokens == cliff
    assert result.cliff_behavior == CliffBehavior.HARD_ERROR
    # Binary search over [16, 32768] needs ~15 classifications plus the
    # ceiling pre-probe and the post-cliff confirmation. Linear scanning from
    # LO would need (5003 - 16) * 2 = ~9974 requests. Anything far below that
    # proves logarithmic (binary), not linear, probing.
    assert result.probe_requests_used < 100


def test_golden_silent_truncation_matches_expected_format() -> None:
    """The committed golden report must stay in the expected capability-card
    format: a titled markdown table whose data rows carry a provenance marker
    in the Source column, plus the Findings and Fix sections.

    This guards the README promise that every reported value carries a
    provenance marker (``read``/``measured``/``inferred``/``unknown``); a row
    rendered without one breaks that promise and must fail here.
    """
    text = (GOLDEN_DIR / "silent-truncation.md").read_text()

    assert text.startswith("# Capability Report — ")
    assert "| Property | Claimed | Measured | Source | Verdict |" in text

    data_rows = [
        line
        for line in text.splitlines()
        if line.startswith("| ")
        and "| Property |" not in line
        and "| --- " not in line
    ]
    assert data_rows, "expected at least one table data row"

    row = re.compile(
        r"^\| .+ \| .+ \| .+ \| "
        r"(?P<source>read|measured|inferred|unknown) \| .+ \|$"
    )
    for data in data_rows:
        m = row.match(data)
        assert m, f"table row rendered without a provenance marker: {data!r}"

    assert "## Findings" in text
    assert "## Fix" in text
    assert "silent_truncation" in text


@pytest.mark.asyncio
async def test_unreachable_server_raises_http_error() -> None:
    """An unreachable server must surface as an ``httpx.HTTPError``, never a
    fabricated ``hard_error`` verdict.

    README promises: a transport failure is reported to stderr and the process
    exits ``2`` (see cli.py, which turns ``httpx.HTTPError`` into exit code 2).
    That promise only holds if the capacity probe does not swallow a transport
    failure into a ``hard_error`` classification. Here the transport drops every
    request (unreachable host), so ``probe_capacity`` must propagate the error
    for the caller to exit ``2`` — it must not pretend to have measured a cliff.
    """
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused", request=request)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport, base_url=BASE_URL
    ) as client:
        with pytest.raises(httpx.HTTPError):
            await probe_capacity(
                client,
                BASE_URL,
                EMBED_ENDPOINT,
                ceiling=32768,
                backend=Backend.LLAMACPP,
            )


@pytest.mark.asyncio
async def test_endpoint_selection_routes_probe_to_chat_path() -> None:
    """``Endpoint.CHAT`` must route the probe to the chat path, not embeddings.

    The README promises ``--endpoint chat`` exercises the chat endpoint. When
    the caller selects ``CHAT`` the probe must send its requests to the
    ``/v1/chat/completions`` path; the resolved path must be reflected in the
    reported ``CapacityResult.endpoint`` so the honest value is recorded.
    """
    server = make_mock_server(max_tokens=512, behavior="hard_error")
    async with _client(server) as client:
        result = await probe_capacity(
            client, BASE_URL, Endpoint.CHAT, ceiling=32768, backend=Backend.LLAMACPP
        )
    assert result.endpoint == CHAT
    assert result.max_accepted_tokens == 512
    assert result.cliff_behavior == CliffBehavior.HARD_ERROR


@pytest.mark.asyncio
async def test_endpoint_selection_routes_probe_to_embeddings_path() -> None:
    """``Endpoint.EMBEDDINGS`` must route the probe to the embeddings path.

    An explicit ``EMBEDDINGS`` selection must be honoured directly and the
    resolved path must be reported as the exercised endpoint in the result.
    """
    server = make_mock_server(max_tokens=512, behavior="hard_error")
    async with _client(server) as client:
        result = await probe_capacity(
            client, BASE_URL, Endpoint.EMBEDDINGS, ceiling=32768, backend=Backend.VLLM
        )
    assert result.endpoint == EMBEDDINGS
    assert result.max_accepted_tokens == 512
    assert result.cliff_behavior == CliffBehavior.HARD_ERROR


@pytest.mark.asyncio
async def test_endpoint_auto_resolves_per_backend() -> None:
    """``Endpoint.AUTO`` must resolve to the backend's default probe path.

    The README promises the default ``auto`` selection resolves per backend and
    is not silently collapsed to embeddings for every backend. Each backend's
    default path must be the path actually exercised and reported.
    """
    from llmprobe.backends import DEFAULT_PROBE_ENDPOINTS

    for backend in Backend:
        server = make_mock_server(max_tokens=512, behavior="hard_error")
        async with _client(server) as client:
            result = await probe_capacity(
                client, BASE_URL, Endpoint.AUTO, ceiling=32768, backend=backend
            )
        assert result.endpoint == DEFAULT_PROBE_ENDPOINTS[backend], (
            f"auto did not resolve to {backend}'s default probe path"
        )
