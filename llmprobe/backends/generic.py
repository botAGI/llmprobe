"""Generic OpenAI-compatible fallback backend adapter.

Last-resort adapter used when no more specific backend (llama.cpp, vLLM,
Ollama) matches the server. Imports only from :mod:`llmprobe.models`.

This adapter is deliberately conservative: it trusts nothing it cannot read
directly. It only ever queries the OpenAI-compatible ``GET /v1/models``
endpoint to learn the advertised model id. All capacities (context length,
batch sizes, slot counts) are reported as ``None`` with provenance
``UNKNOWN`` rather than being invented.

``detect`` always returns ``0.1`` so that any specific adapter — even one
with imperfect confidence (e.g. vLLM at ``0.95``) — wins the detection
round and assigns the concrete backend when one exists.
"""

from __future__ import annotations

import asyncio
import httpx

from llmprobe.models import (
    Backend,
    EffectiveConfig,
    Finding,
    Provenance,
    Severity,
    redact_userinfo,
)

GENERIC_DETECT_CONFIDENCE = 0.1

#: Finding codes emitted when the sole generic endpoint cannot be read.
MODELS_UNREACHABLE_CODE = "GENERIC_MODELS_UNREACHABLE"
MODELS_HTTP_ERROR_CODE = "GENERIC_MODELS_HTTP_ERROR"

#: Retry budget for transient network failures on the sole generic endpoint.
#: A single blip (timeout, dropped connection) must not be mistaken for a hard
#: boundary, so ``/v1/models`` is retried a few times with exponential backoff
#: before the unreachable finding is surrendered.
MODELS_NETWORK_ATTEMPTS = 3
MODELS_NETWORK_BACKOFF_SECONDS = (0.5, 1.0, 2.0)


async def _get_with_retry(
    client: httpx.AsyncClient, url: str
) -> httpx.Response:
    """GET ``url`` retrying transient network failures with backoff.

    Only ``httpx.TransportError`` (connection refused, timeout, dropped
    connection mid-flight) is retried — an HTTP status response is not a
    network failure and is returned as-is on the first attempt. The retry
    budget and backoff schedule honour :data:`MODELS_NETWORK_ATTEMPTS` and
    :data:`MODELS_NETWORK_BACKOFF_SECONDS`. If every attempt fails, the last
    exception is re-raised so the caller can surface it as a finding.
    """
    last_error: httpx.TransportError | None = None
    for attempt in range(MODELS_NETWORK_ATTEMPTS):
        try:
            return await client.get(url, timeout=client.timeout)
        except httpx.TransportError as exc:
            last_error = exc
            if attempt < MODELS_NETWORK_ATTEMPTS - 1:
                await asyncio.sleep(MODELS_NETWORK_BACKOFF_SECONDS[attempt])
    assert last_error is not None
    raise last_error


async def detect(client: httpx.AsyncClient, base_url: str) -> float:
    """Return a low confidence (0.1) that ``base_url`` is generic.

    Always ``0.1`` regardless of the server, so any specific adapter that
    returns a higher confidence takes precedence. Used only as a last resort.
    """
    return GENERIC_DETECT_CONFIDENCE


async def read_config(
    client: httpx.AsyncClient, base_url: str
) -> tuple[EffectiveConfig, list[Finding]]:
    """Read the only value a generic server is willing to admit.

    ``GET /v1/models`` yields the advertised model id (provenance ``read``
    when present, otherwise ``unknown``). Every capacity field stays ``None``
    with provenance ``UNKNOWN`` because a generic OpenAI-compatible server
    gives us no reliable measurement of them.

    Returned alongside the config is a list of findings. A non-200 response
    or a transport failure while reading ``/v1/models`` is surfaced as an
    ERROR finding rather than swallowed silently or allowed to raise, so the
    caller can report exactly why the probe failed.
    """
    base = base_url.rstrip("/")
    display = redact_userinfo(base)
    model_id = ""
    findings: list[Finding] = []
    try:
        resp = await _get_with_retry(client, f"{base}/v1/models")
    except httpx.HTTPError as exc:
        findings.append(
            Finding(
                severity=Severity.ERROR,
                code=MODELS_UNREACHABLE_CODE,
                message=redact_userinfo(
                    f"failed to reach {display}/v1/models: {exc}"
                ),
            )
        )
        payload = {}
    else:
        if resp.status_code != 200:
            findings.append(
                Finding(
                    severity=Severity.ERROR,
                    code=MODELS_HTTP_ERROR_CODE,
                    advertised=resp.status_code,
                    message=(
                        f"{display}/v1/models returned HTTP {resp.status_code}"
                    ),
                )
            )
            payload = {}
        else:
            try:
                payload = resp.json()
            except ValueError:
                findings.append(
                    Finding(
                        severity=Severity.ERROR,
                        code=MODELS_HTTP_ERROR_CODE,
                        message=f"{display}/v1/models returned invalid JSON",
                    )
                )
                payload = {}

    data = payload.get("data") if isinstance(payload, dict) else None
    entry = data[0] if isinstance(data, list) and data else {}
    if isinstance(entry, dict):
        model_id = str(entry.get("id") or "")

    sources: dict[str, Provenance] = {
        "n_ctx_total": Provenance.UNKNOWN,
        "n_ctx_per_slot": Provenance.UNKNOWN,
        "n_batch": Provenance.UNKNOWN,
        "n_ubatch": Provenance.UNKNOWN,
        "total_slots": Provenance.UNKNOWN,
    }

    config = EffectiveConfig(
        backend=Backend.GENERIC,
        model_id=model_id,
        n_ctx_total=None,
        n_ctx_per_slot=None,
        n_batch=None,
        n_ubatch=None,
        total_slots=None,
        sources=sources,
    )
    return config, findings
