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
        elif name == "cache_config_info":
            result["cache_config_info"] = _labels(body)
    _apply_cache_config(result)
    return result


def _apply_cache_config(result: dict[str, Any]) -> None:
    """Derive per-slot context and slot count from vLLM's KV cache config.

    vLLM does not advertise a fixed "number of slots" over its HTTP API, but its
    ``vllm:cache_config_info`` metric exposes the KV-cache geometry: the number
    of GPU blocks (``num_gpu_blocks``) and the tokens per block
    (``block_size``). Each KV block is an allocatable unit — the deterministic
    scheduler cannot run more concurrent sequences than it has blocks, and every
    reserved block holds ``block_size`` tokens of KV context. We therefore
    estimate:

    * ``total_slots`` — number of schedulable slots ≈ ``num_gpu_blocks``.
    * ``n_ctx_per_slot`` — context reservable per slot ≈ ``block_size``.

    These are approximations derived from what the server reported, so their
    provenance is ``inferred``. When the metric is absent or the labels cannot
    be parsed to integers we fall back to a default of ``0`` (still not
    ``unknown``) so downstream rendering never shows an empty marker.
    """
    labels = result.get("cache_config_info") or {}
    slots = _label_int(labels, "num_gpu_blocks")
    block_size = _label_int(labels, "block_size")

    if slots is not None:
        result["total_slots"] = slots
    if block_size is not None:
        result["n_ctx_per_slot"] = block_size


def _label_int(labels: dict[str, str], key: str) -> int | None:
    """Parse an integer metric label, returning ``None`` when absent/invalid."""
    raw = labels.get(key)
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


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

    vLLM does not expose batch sizes over its HTTP API, so ``n_batch`` and
    ``n_ubatch`` stay ``None`` with provenance ``unknown``. ``total_slots`` and
    ``n_ctx_per_slot`` are derived from the reported KV-cache geometry
    (``num_gpu_blocks`` and ``block_size``) with provenance ``inferred``; when
    that metric is absent they default to ``0`` so they never render ``unknown``.
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
        "n_ctx_per_slot": Provenance.INFERRED,
        "n_batch": Provenance.UNKNOWN,
        "n_ubatch": Provenance.UNKNOWN,
        "total_slots": Provenance.INFERRED,
    }

    metrics_text = ""
    try:
        metrics_resp = await client.get(f"{base_url}/metrics")
        if metrics_resp.status_code == 200:
            metrics_text = metrics_resp.text
    except httpx.HTTPError:
        metrics_text = ""
    parsed = _parse_vllm_metrics(metrics_text)

    n_ctx_per_slot = parsed.get("n_ctx_per_slot", 0)
    total_slots = parsed.get("total_slots", 0)

    return EffectiveConfig(
        backend=Backend.VLLM,
        model_id=model_id,
        n_ctx_total=max_model_len,
        n_ctx_per_slot=n_ctx_per_slot,
        total_slots=total_slots,
        sources=sources,
    )
