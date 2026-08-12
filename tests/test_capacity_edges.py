"""Edge-case robustness tests for :mod:`llmprobe.probes.capacity`.

Hermetic: no network, no real inference server. The probe is driven through
``httpx.MockTransport`` (for the degraded-response cases the scripted mock in
``tests/mocks/server.py`` cannot express — empty bodies, zero-length vectors,
torn connections) and through :func:`tests.mocks.server.make_mock_server` (for
the honest/degenerate-server the factory already models).

Every test pins a robustness contract: a live probe must never crash on a
degenerate server. It must either degrade to an honest classification or, for a
torn transport, propagate the ``httpx.HTTPError`` — never fabricate a ``measured``
value it could not obtain.
"""

from __future__ import annotations

import json

import httpx
import pytest

from llmprobe.models import Backend, CliffBehavior, Endpoint, Provenance
from llmprobe.probes.capacity import probe_capacity

from tests.mocks.server import make_mock_server

BASE_URL = "http://edge.test"

EMBEDDINGS = "/v1/embeddings"

EMBED_ENDPOINT = Endpoint.EMBEDDINGS

CHAT = "/v1/chat/completions"

CHAT_ENDPOINT = Endpoint.CHAT

# The chat probe detects silent truncation via a distinctive canary marker
# prepended to the prompt head. Kept in sync with
# ``llmprobe.probes.capacity._CANARY``.
CANARY = "ZQX7"


def _tokenize_response(request: httpx.Request) -> httpx.Response:
    """Mirror the scripted mock's ``/tokenize``: one token per word.

    The capacity probe verifies exact counts against ``/tokenize``; without a
    working handler the ``_n_token_prompt`` exact-count path could not run and
    every test would trivially report ``token_count_exact=False``.
    """
    body = json.loads(request.read())
    content = body.get("content", "")
    if isinstance(content, list):
        content = " ".join(str(x) for x in content)
    return httpx.Response(200, json={"tokens": ["tok"] * len(str(content).split())})


def _embeddings_client(embed_handler) -> httpx.AsyncClient:
    """Client whose ``/tokenize`` works and whose embeddings obey a handler.

    ``embed_handler`` is called with the request and must return an
    ``httpx.Response`` (or raise) for :data:`EMBEDDINGS`; ``/tokenize`` is
    always served a valid one-token-per-word payload so the probe can verify
    exact counts.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/tokenize":
            return _tokenize_response(request)
        if request.url.path == EMBEDDINGS:
            return embed_handler(request)
        return httpx.Response(404, text="not found")

    return httpx.AsyncClient(
        base_url=BASE_URL, transport=httpx.MockTransport(handler)
    )


def _chat_client(chat_handler) -> httpx.AsyncClient:
    """Client whose ``/tokenize`` works and whose chat replies obey a handler.

    ``chat_handler`` is called with the request and must return an
    ``httpx.Response`` (or raise) for :data:`CHAT`; ``/tokenize`` is always
    served a valid one-token-per-word payload so the probe can build exact
    ``n``-token prompts.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/tokenize":
            return _tokenize_response(request)
        if request.url.path == CHAT:
            return chat_handler(request)
        return httpx.Response(404, text="not found")

    return httpx.AsyncClient(
        base_url=BASE_URL, transport=httpx.MockTransport(handler)
    )


@pytest.mark.asyncio
async def test_empty_200_body_degrades_to_hard_error() -> None:
    """An HTTP 200 with an empty body must not crash the probe.

    The two-prompt embedding POSTs both return ``200`` with no body. The probe
    cannot read any embedding, so it must honestly classify the length as
    ``hard_error`` (the length could not be verified as accepted) rather than
    raise on the unparsable body. If the ``ValueError`` guard in the embedding
    reader were ever removed, ``resp.json()`` on an empty body would raise and
    this test would go red.
    """
    def embed(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"")

    async with _embeddings_client(embed) as client:
        result = await probe_capacity(
            client, BASE_URL, EMBED_ENDPOINT, ceiling=64, backend=Backend.LLAMACPP, model="mock"
        )

    assert result.cliff_behavior == CliffBehavior.HARD_ERROR


@pytest.mark.asyncio
async def test_zero_length_embedding_does_not_crash() -> None:
    """An embeddings endpoint returning a zero-length vector must not crash.

    A broken server may answer 200 with an empty ``embedding`` array, making
    the cosine comparison degenerate (zero-norm vectors). The comparison must
    guard against division by zero and return a valid ``CapacityResult``, never
    raise a ``ZeroDivisionError``. Removing that guard would crash the probe on
    every probe length and fail this test.
    """
    def embed(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"embedding": []}], "model": "mock"})

    async with _embeddings_client(embed) as client:
        result = await probe_capacity(
            client, BASE_URL, EMBED_ENDPOINT, ceiling=64, backend=Backend.LLAMACPP, model="mock"
        )

    assert isinstance(result.cliff_behavior, CliffBehavior)


@pytest.mark.asyncio
async def test_mid_response_teardown_propagates_http_error() -> None:
    """A connection torn mid-response must surface as ``httpx.HTTPError``.

    When the server drops the connection while streaming the embedding, the
    probe must propagate the transport failure (``httpx.HTTPError``) so the
    caller can report an unreachable/degraded server and exit ``2`` — it must
    NOT swallow it into a fabricated ``hard_error`` verdict the probe could not
    actually measure. A ``RemoteProtocolError`` IS what httpx raises when a
    peer closes mid-message, so ``pytest.raises(httpx.HTTPError)`` pins the
    propagation promise. If the probe ever caught transport errors and returned
    a ``hard_error`` result instead, no ``HTTPError`` would be raised and this
    test would go red.
    """
    def embed(request: httpx.Request) -> httpx.Response:
        raise httpx.RemoteProtocolError(
            "peer closed connection without sending complete message body"
        )

    async with _embeddings_client(embed) as client:
        with pytest.raises(httpx.HTTPError):
            await probe_capacity(
                client, BASE_URL, EMBED_ENDPOINT, ceiling=64, backend=Backend.LLAMACPP, model="mock"
            )


@pytest.mark.asyncio
async def test_ceiling_below_search_lower_bound_is_honest() -> None:
    """A ceiling below the search floor must not crash or loop forever.

    The binary search only probes lengths ``>= LO``; when the caller passes a
    ``ceiling`` below that floor the search space is empty/inverted. The probe
    must terminate immediately and report the ceiling honestly as a lower-bound
    estimate (``max_accepted_source == UNKNOWN``) rather than hang or claim a
    measured maximum it never searched for. Removing the guard that returns when
    the ceiling is accepted would let the inverted-range loop spin and fail this
    test.
    """
    server = make_mock_server(max_tokens=8192, behavior="honest")
    transport = httpx.ASGITransport(app=server)
    async with httpx.AsyncClient(
        transport=transport, base_url=BASE_URL
    ) as client:
        result = await probe_capacity(
            client, BASE_URL, EMBED_ENDPOINT, ceiling=8, backend=Backend.LLAMACPP, model="mock"
        )

    assert result.max_accepted_source == Provenance.UNKNOWN
    assert result.max_accepted_tokens == 8
    assert result.probe_requests_used <= 4


@pytest.mark.asyncio
async def test_server_accepting_everything_reports_unmeasured() -> None:
    """A server that accepts everything must not claim a measured maximum.

    When the ceiling itself is accepted the true capacity lies above it, so the
    reported ceiling is only a lower bound — a probe could not measure the real
    maximum. It MUST be flagged ``max_accepted_source == UNKNOWN`` (never
    ``measured``) and ``cliff_behavior == ACCEPTED``.
    """
    server = make_mock_server(max_tokens=32768, behavior="honest")
    transport = httpx.ASGITransport(app=server)
    async with httpx.AsyncClient(
        transport=transport, base_url=BASE_URL
    ) as client:
        result = await probe_capacity(
            client, BASE_URL, EMBED_ENDPOINT, ceiling=32768, backend=Backend.LLAMACPP, model="mock"
        )

    assert result.max_accepted_tokens == 32768
    assert result.cliff_behavior == CliffBehavior.ACCEPTED
    assert result.max_accepted_source == Provenance.UNKNOWN


@pytest.mark.asyncio
async def test_marker_variant_case_and_whitespace_is_not_silent_truncation() -> None:
    """A server that fully accepts the input and echoes the marker with varied
    case or spacing must not be misclassified as silent truncation.

    The chat probe detects silent truncation by normalising the reply (stripping
    surrounding whitespace, folding to uppercase) and looking for the canary
    marker as a substring: a server that drops the prompt head loses the canary,
    so its absence signals truncation. A server that fully accepts the input
    echoes the marker and must be reported as ``accepted`` — the normalisation
    is what makes matching robust to case and whitespace, not a strict equality
    against a verbatim echo. If the check ever reverted to a case/whitespace-
    sensitive exact match, this test would go red.
    """
    def chat(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": f"  {CANARY.lower()} "}}]},
        )

    async with _chat_client(chat) as client:
        result = await probe_capacity(
            client, BASE_URL, CHAT_ENDPOINT, ceiling=64, backend=Backend.LLAMACPP, model="mock"
        )

    assert result is not None
    assert result.cliff_behavior in (
        CliffBehavior.ACCEPTED,
    )
    assert result.cliff_behavior != CliffBehavior.SILENT_TRUNCATION


@pytest.mark.asyncio
async def test_chat_marker_absent_in_calibration_returns_unknown() -> None:
    """A chat server that cannot echo the calibration marker reports UNKNOWN.

    Before measuring any cliff, the chat probe sends a short calibration input
    whose head carries the ``ZQX7`` marker and verifies the reply echoes it.
    When the server answers 200 but the marker is absent from the reply (the
    marker-echo mechanism is untrustworthy even for a certainly-accepted short
    input), the probe must report an honest UNKNOWN result — ``max_accepted_
    source == UNKNOWN`` with no measured boundary — rather than a confident
    ``silent_truncation`` verdict it could not reliably obtain.
    """
    def chat(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "wrongword"}}]},
        )

    async with _chat_client(chat) as client:
        result = await probe_capacity(
            client,
            BASE_URL,
            CHAT_ENDPOINT,
            ceiling=64,
            backend=Backend.LLAMACPP,
            model="mock",
        )

    assert result is not None
    assert result.max_accepted_source == Provenance.UNKNOWN
    assert result.max_accepted_tokens == 0
