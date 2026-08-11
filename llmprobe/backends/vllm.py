"""vLLM backend adapter.

Talks to a running vLLM server's OpenAI-compatible HTTP surface plus its
Prometheus metrics endpoint to read the configuration vLLM actually exposes.

vLLM (unlike llama.cpp) advertises ``max_model_len`` on ``/v1/models`` and
serves Prometheus metrics under ``/metrics`` with a ``vllm:`` prefix, which is
the signal we use for detection. No value is fabricated: fields vLLM does not
expose are reported as provenance ``unknown``.

Capacity fields: the real bound on concurrent sequences is vLLM's
``max_num_seqs`` scheduler value, not the KV-cache geometry. Standard vLLM
builds only expose ``vllm:cache_config_info`` (``block_size``, ``num_gpu_blocks``)
over HTTP and do not advertise ``max_num_seqs``; when we can read a
``vllm:max_num_seqs`` metric we derive slots from it and the per-slot context
from ``max_model_len // max_num_seqs``. When it is unavailable we honestly
report ``total_slots`` and ``n_ctx_per_slot`` as ``None`` with provenance
``unknown`` rather than substitute ``block_size`` / ``num_gpu_blocks``.
"""

from __future__ import annotations

from typing import Any

import httpx

from llmprobe.models import Backend, EffectiveConfig, Provenance


VLLM_PREFIX = "vllm:"
VLLM_DETECT_CONFIDENCE = 0.95


def extract_prompt_tokens(payload: Any) -> int | None:
    """Return ``usage.prompt_tokens`` from a vLLM API response.

    vLLM reports ``usage.prompt_tokens`` on its chat and embeddings responses;
    this reads the exact number the server itself reported. Returns ``None``
    when the field is absent or not an integer — we never fabricate a count.
    """
    try:
        usage = payload["usage"]
        tokens = usage["prompt_tokens"]
    except (KeyError, TypeError):
        return None
    if not isinstance(tokens, int):
        return None
    return tokens


def _metric_lines(metrics_text: str | None) -> list[str]:
    """Return non-empty, non-comment lines from a Prometheus text dump.

    ``None`` and empty/only-comment bodies are treated the same: there are no
    valid metric lines.
    """
    if not metrics_text:
        return []
    return [
        line.strip()
        for line in metrics_text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _to_float(value: str) -> float | None:
    """Best-effort parse of a Prometheus sample value to a float."""
    try:
        return float(value)
    except ValueError:
        return None


def _parse_vllm_metrics(metrics_text: str) -> dict[str, Any]:
    """Parse the vLLM metrics fields we care about.

    vLLM's Prometheus endpoint namespaces every metric with the ``vllm:``
    prefix. The presence of any such line is itself the detection signal; we
    never require a specific metric to be present, because live servers vary
    by version and some metrics may be disabled.

    Returns a dict with:

    * ``is_vllm`` — True if any metric line carries the ``vllm:`` prefix.
    * ``num_requests_running`` — float value of ``vllm:num_requests_running``
      when present (Prometheus gauges are emitted as floats).
    * ``max_num_seqs`` — int value of ``vllm:max_num_seqs`` when present. This
      is the scheduler's concurrency bound; the true slot count. It is
      deliberately distinct from the KV-cache geometry labels.
    """
    lines = _metric_lines(metrics_text)
    if not lines:
        # Empty body or only comment/blank lines: there are no valid metric
        # samples, so we report an empty metrics dict rather than risk a
        # parsing failure downstream.
        return {}
    result: dict[str, Any] = {"is_vllm": False}
    for line in lines:
        if not line.startswith(VLLM_PREFIX):
            continue
        # The vllm: namespace alone identifies the server.
        result["is_vllm"] = True
        body = line[len(VLLM_PREFIX) :]
        name_part, _, value = body.partition(" ")
        # Drop any Prometheus label block from the metric name before matching.
        name = name_part.split("{", 1)[0].rstrip()
        value = value.strip()
        if name == "num_requests_running":
            parsed = _to_float(value)
            if parsed is not None:
                result["num_requests_running"] = parsed
        elif name == "max_num_seqs":
            parsed = _to_float(value)
            if parsed is not None:
                result["max_num_seqs"] = int(parsed)
    return result


def _parse_max_num_seqs(parsed: dict[str, Any]) -> int | None:
    """Return the scheduler concurrency bound, or ``None`` when unavailable.

    Only a positive integer ``vllm:max_num_seqs`` sample is honoured. Anything
    else (absent metric, non-finite, zero) yields ``None`` so the caller can
    fall back to an honest ``unknown`` instead of a bogus estimate.
    """
    raw = parsed.get("max_num_seqs")
    if not isinstance(raw, int) or raw <= 0:
        return None
    return raw


async def detect(client: httpx.AsyncClient, base_url: str) -> float:
    """Confidence (0.0..1.0) that ``base_url`` is a vLLM server.

    Detection is the ``vllm:`` prefix appearing in the server's ``/metrics``
    output. A missing or non-vLLM ``/metrics`` yields ``0.0``; a vLLM-style
    ``/metrics`` yields a high confidence > 0.8.
    """
    try:
        resp = await client.get(f"{base_url}/metrics")
    except httpx.HTTPError:
        return 0.0
    if resp.status_code != 200:
        return 0.0
    parsed = _parse_vllm_metrics(resp.text)
    if parsed.get("is_vllm"):
        return VLLM_DETECT_CONFIDENCE
    return 0.0


async def read_config(
    client: httpx.AsyncClient, base_url: str
) -> EffectiveConfig:
    """Read the config vLLM advertises.

    * ``GET /v1/models`` — ``data[0].max_model_len`` becomes ``n_ctx_total``
      with provenance ``read`` (vLLM exposes this; llama.cpp does not).
    * ``GET /metrics`` — parse ``vllm:num_requests_running`` and the optional
      ``vllm:max_num_seqs`` scheduler bound when present (best effort).

    vLLM does not expose batch sizes over its HTTP API, so ``n_batch`` and
    ``n_ubatch`` stay ``None`` with provenance ``unknown``.

    ``total_slots`` and ``n_ctx_per_slot`` are derived from the scheduler's
    concurrency bound ``max_num_seqs``: when it is readable over HTTP,
    ``total_slots = max_num_seqs`` and ``n_ctx_per_slot = max_model_len //
    max_num_seqs`` (provenance ``inferred``). Standard vLLM builds do not expose
    ``max_num_seqs`` over HTTP, in which case both fields are reported as
    ``None`` with provenance ``unknown`` — ``block_size``/``num_gpu_blocks``
    (KV-cache geometry) are never substituted for them.
    """
    models_resp = await client.get(f"{base_url}/v1/models")
    try:
        models_resp.raise_for_status()
        models = models_resp.json()
    except (httpx.HTTPError, ValueError):
        models = {}
    data = models.get("data") or []
    entry = data[0] if data else {}
    model_id = str(entry.get("id", ""))
    max_model_len = entry.get("max_model_len")

    sources: dict[str, Provenance] = {
        "n_ctx_total": (
            Provenance.READ if max_model_len is not None else Provenance.UNKNOWN
        ),
        "n_ctx_per_slot": Provenance.UNKNOWN,
        "n_batch": Provenance.UNKNOWN,
        "n_ubatch": Provenance.UNKNOWN,
        "total_slots": Provenance.UNKNOWN,
    }

    metrics_text = ""
    try:
        metrics_resp = await client.get(f"{base_url}/metrics")
        if metrics_resp.status_code == 200:
            metrics_text = metrics_resp.text
    except httpx.HTTPError:
        metrics_text = ""
    parsed = _parse_vllm_metrics(metrics_text)

    max_num_seqs = _parse_max_num_seqs(parsed)
    n_ctx_per_slot: int | None = None
    total_slots: int | None = None
    if max_num_seqs is not None and max_model_len is not None:
        total_slots = max_num_seqs
        n_ctx_per_slot = max_model_len // max_num_seqs
        sources["total_slots"] = Provenance.INFERRED
        sources["n_ctx_per_slot"] = Provenance.INFERRED

    return EffectiveConfig(
        backend=Backend.VLLM,
        model_id=model_id,
        n_ctx_total=max_model_len,
        n_ctx_per_slot=n_ctx_per_slot,
        total_slots=total_slots,
        sources=sources,
    )
