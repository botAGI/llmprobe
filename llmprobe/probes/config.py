"""Adapter selection and effective-config reads.

Runs every backend's ``detect`` concurrently, picks the most confident
adapter (breaking ties deterministically by a fixed priority order), then
reads the configuration from the winning backend and merges any findings it
produced. Imports only from :mod:`llmprobe.models` and
:mod:`llmprobe.backends`.
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

import httpx

from llmprobe.backends import generic, llamacpp, ollama, vllm
from llmprobe.models import Backend, EffectiveConfig, Finding

# Detection priority: when two adapters report equal confidence we rely on a
# fixed order rather than on completion order, so the winner is deterministic.
_ADAPTER_PRIORITY: dict[Backend, int] = {
    Backend.LLAMACPP: 0,
    Backend.VLLM: 1,
    Backend.OLLAMA: 2,
    Backend.GENERIC: 3,
}

# (backend, module) pairs mirroring _ADAPTER_PRIORITY for concurrent dispatch.
_SELECTABLE_ADAPTERS: dict[Backend, Any] = {
    Backend.LLAMACPP: llamacpp,
    Backend.VLLM: vllm,
    Backend.OLLAMA: ollama,
    Backend.GENERIC: generic,
}

ReadConfigResult = EffectiveConfig | tuple[EffectiveConfig, list[Finding]]


def _detect_adapter(
    adapter: Callable[[httpx.AsyncClient, str], Awaitable[float]],
) -> Callable[[httpx.AsyncClient, str], Awaitable[float]]:
    """Guard a backend ``detect`` so a failing probe never blocks selection.

    A backend that raises while probing the server is treated as a "no match"
    (confidence 0) rather than aborting the whole selection round.
    """

    async def guarded(client: httpx.AsyncClient, base_url: str) -> float:
        try:
            score = await adapter(client, base_url)
        except Exception:
            return 0.0
        return score if isinstance(score, (int, float)) else 0.0

    return guarded


async def _select_backend(
    client: httpx.AsyncClient, base_url: str
) -> Backend:
    """Run every adapter's ``detect`` concurrently and return the winner.

    The winner is the adapter with the highest confidence; ties are resolved
    by the fixed priority order in :data:`_ADAPTER_PRIORITY` (llamacpp >
    vllm > ollama > generic). The result is independent of the order in which
    the concurrent probes happen to complete.
    """
    base = base_url.rstrip("/")

    async def _probe(backend: Backend) -> tuple[Backend, float]:
        module = _SELECTABLE_ADAPTERS[backend]
        guarded = _detect_adapter(module.detect)
        score = await guarded(client, base)
        return backend, score

    results = await asyncio.gather(
        *(_probe(backend) for backend in _SELECTABLE_ADAPTERS)
    )

    def _key(item: tuple[Backend, float]) -> tuple[float, int]:
        backend, score = item
        return score, -_ADAPTER_PRIORITY[backend]

    winner, _ = max(results, key=_key)
    return winner


def _merge_findings(
    result: ReadConfigResult,
) -> tuple[EffectiveConfig, list[Finding]]:
    """Normalise a ``read_config`` result into ``(config, findings)``."""
    if isinstance(result, tuple):
        config, extra = result
        return config, list(extra)
    return result, []


async def read_effective_config(
    client: httpx.AsyncClient,
    base_url: str,
    claimed_ctx: int | None,
) -> tuple[EffectiveConfig, list[Finding]]:
    """Detect the backend and read the effective configuration.

    ``claimed_ctx`` is accepted for API compatibility with downstream probes
    that pass a caller-claimed context; adapter selection itself never depends
    on it. The winning adapter's ``read_config`` is invoked and any findings
    it emitted are returned alongside the configuration.
    """
    backend = await _select_backend(client, base_url)
    base = base_url.rstrip("/")
    module = _SELECTABLE_ADAPTERS[backend]
    result = await module.read_config(client, base)
    return _merge_findings(result)
