"""vLLM backend adapter.

Talks to a running vLLM server's OpenAI-compatible HTTP surface plus its
Prometheus metrics endpoint to read the configuration vLLM actually exposes.

vLLM (unlike llama.cpp) advertises ``max_model_len`` on ``/v1/models`` and
serves Prometheus metrics under ``/metrics`` with a ``vllm:`` prefix, which is
the signal we use for detection. No value is fabricated: fields vLLM does not
expose are reported as provenance ``unknown``.
"""

from __future__ import annotations

from typing import Any

import httpx

from llmprobe.models import Backend, EffectiveConfig, Provenance


VLLM_PREFIX = "vllm:"
VLLM_DETECT_CONFIDENCE = 0.95


def _metric_lines(metrics_text: str) -> list[str]:
    """Return non-empty, non-comment lines from a Prometheus text dump."""
    return [
        line.strip()
        for line in metrics_text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _labels(text: str) -> dict[str, str]:
    """Parse the ``{key="value",...}`` label block of a metric line."""
    if "{" not in text or "}" not in text:
        return {}
    start = text.index("{")
    end = text.index("}", start)
    raw = text[start + 1 : end]
    labels: dict[str, str] = {}
    for part in raw.split(","):
        part = part.strip()
        if "=" not in part:
            continue
        key, _, value = part.partition("=")
        labels[key.strip()] = value.strip().strip('"')
    return labels


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
    * ``cache_config_info`` — labels of ``vllm:cache_config_info`` when present.
    """
    result: dict[str, Any] = {"is_vllm": False}
    for line in _metric_lines(metrics_text):
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
        elif name == "cache_config_info":
            result["cache_config_info"] = _labels(body)
    return result


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
    * ``GET /metrics`` — parse ``vllm:num_requests_running`` and
      ``vllm:cache_config_info`` labels when present (best effort).

    vLLM does not expose per-slot context, batch sizes, or a fixed slot count
    over its HTTP API, so those fields stay unset with provenance ``unknown``.
    """
    models_resp = await client.get(f"{base_url}/v1/models")
    models_resp.raise_for_status()
    models = models_resp.json()
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
    _parse_vllm_metrics(metrics_text)

    return EffectiveConfig(
        backend=Backend.VLLM,
        model_id=model_id,
        n_ctx_total=max_model_len,
        sources=sources,
    )
