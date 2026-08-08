"""Ollama backend adapter.

Detects an Ollama server, reads its effective config, and reports a context
downgrade when a loaded model's actual context length is smaller than its
trained context length.

Ollama endpoints used:

* ``GET /api/tags`` — list of installed models (used to detect the server by
  shape and to choose a model to probe).
* ``GET /api/ps`` — models currently loaded / RUNNING. A running model reports
  the ACTUAL context length it was loaded with (``context_length``).
* ``POST /api/show`` — model details; ``model_info["*.context_length"]`` is
  the TRAINED context length baked into the model file.

The point of this adapter: when the trained context (e.g. 32768) is larger
than the loaded context (e.g. 4096) the server silently serves a smaller
context than the model supports. That downgrade is surfaced as a
``OLLAMA_CTX_DOWNGRADE`` MISMATCH finding.
"""

from __future__ import annotations

from typing import Any

import httpx

from llmprobe.models import Backend, EffectiveConfig, Finding, Provenance, Severity


async def _get_json(
    client: httpx.AsyncClient, url: str
) -> dict[str, Any] | None:
    try:
        resp = await client.get(url)
    except httpx.HTTPError:
        return None
    if resp.status_code != 200:
        return None
    try:
        payload = resp.json()
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


async def _post_json(
    client: httpx.AsyncClient, url: str, body: dict[str, Any]
) -> dict[str, Any] | None:
    try:
        resp = await client.post(url, json=body)
    except httpx.HTTPError:
        return None
    if resp.status_code != 200:
        return None
    try:
        payload = resp.json()
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def _is_tags_shape(payload: dict[str, Any]) -> bool:
    """Return True if ``payload`` matches the Ollama ``/api/tags`` shape."""
    models = payload.get("models")
    if not isinstance(models, list):
        return False
    if "object" in payload:
        # OpenAI-style /v1/models shape, not Ollama.
        return False
    if not models:
        return True
    first = models[0]
    return isinstance(first, dict) and "name" in first


async def detect(client: httpx.AsyncClient, base_url: str) -> float:
    """Return 1.0 if ``base_url`` serves the Ollama ``/api/tags`` shape.

    Returns 0.0 when the server is unreachable, does not answer ``/api/tags``,
    or the payload does not match Ollama's model-list shape.
    """
    payload = await _get_json(client, f"{base_url}/api/tags")
    if payload is None:
        return 0.0
    return 1.0 if _is_tags_shape(payload) else 0.0


def _norm(name: str | None) -> str | None:
    """Normalize a model name for comparisons: trim whitespace, lowercase."""
    if not isinstance(name, str):
        return None
    stripped = name.strip()
    return stripped.lower() if stripped else None


def _model_names(payload: dict[str, Any]) -> list[str]:
    models = payload.get("models")
    if not isinstance(models, list):
        return []
    names: list[str] = []
    for item in models:
        if isinstance(item, dict) and isinstance(item.get("name"), str):
            names.append(item["name"].strip())
    return names


def _first_running_model(ps: dict[str, Any] | None, names: list[str]) -> str | None:
    if ps is None:
        return None
    models = ps.get("models")
    if not isinstance(models, list):
        return None
    normalized = {_norm(n) for n in names if _norm(n) is not None}
    for item in models:
        if not isinstance(item, dict):
            continue
        norm = _norm(item.get("name"))
        if norm is not None and norm in normalized:
            return item["name"]
    return None


def _int_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _loaded_context(ps_model: dict[str, Any] | None) -> int | None:
    if not ps_model:
        return None
    return _int_value(ps_model.get("context_length"))


def _trained_context(show: dict[str, Any] | None) -> int | None:
    """Extract the trained context length from ``model_info``.

    Ollama's ``model_info`` uses keys of the form ``"<arch>.context_length"``
    (e.g. ``llama.context_length``). We match the ``*.context_length`` path
    and return the largest such value found.
    """
    if show is None:
        return None
    model_info = show.get("model_info")
    if not isinstance(model_info, dict):
        return None
    values: list[int] = []
    for key, value in model_info.items():
        if isinstance(key, str) and key.endswith(".context_length"):
            parsed = _int_value(value)
            if parsed is not None:
                values.append(parsed)
    return max(values) if values else None


async def read_config(
    client: httpx.AsyncClient, base_url: str
) -> tuple[EffectiveConfig, list[Finding]]:
    """Read the effective config and findings from an Ollama server.

    Returns a tuple of ``(EffectiveConfig, list[Finding])``. The config's
    ``n_ctx_total`` is the ACTUAL context the running model was loaded with
    (measured via ``/api/ps``). When the trained context from ``/api/show``
    exceeds the loaded context, a ``OLLAMA_CTX_DOWNGRADE`` finding is emitted.
    """
    findings: list[Finding] = []

    tags = await _get_json(client, f"{base_url}/api/tags")
    names = _model_names(tags) if tags is not None else []
    if not names:
        return EffectiveConfig(
            backend=Backend.OLLAMA,
            model_id="",
            n_ctx_total=None,
            sources={"n_ctx_total": Provenance.UNKNOWN},
        ), findings

    ps = await _get_json(client, f"{base_url}/api/ps")
    running = _first_running_model(ps, names)
    model_id = running if running is not None else names[0]

    loaded_ctx: int | None = None
    if running is not None:
        ps_model: dict[str, Any] | None = None
        if ps is not None:
            running_norm = _norm(running)
            for item in ps.get("models", []):
                if isinstance(item, dict) and _norm(item.get("name")) == running_norm:
                    ps_model = item
                    break
        loaded_ctx = _loaded_context(ps_model)

        show = await _post_json(
            client, f"{base_url}/api/show", {"name": running}
        )
        trained_ctx = _trained_context(show)

        if (
            trained_ctx is not None
            and loaded_ctx is not None
            and trained_ctx > loaded_ctx
        ):
            findings.append(
                Finding(
                    severity=Severity.MISMATCH,
                    code="OLLAMA_CTX_DOWNGRADE",
                    advertised=trained_ctx,
                    measured=loaded_ctx,
                    message=(
                        f"model '{model_id}' is loaded with a smaller context "
                        f"({loaded_ctx}) than its trained context ({trained_ctx})"
                    ),
                )
            )

    sources: dict[str, Provenance] = {
        "n_ctx_total": (
            Provenance.MEASURED if loaded_ctx is not None else Provenance.UNKNOWN
        )
    }
    cfg = EffectiveConfig(
        backend=Backend.OLLAMA,
        model_id=model_id,
        n_ctx_total=loaded_ctx,
        sources=sources,
    )
    return cfg, findings
