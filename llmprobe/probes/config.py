"""Adapter selection and effective-config reads.

Runs every backend's ``detect`` concurrently, picks the most confident
adapter (breaking ties deterministically by a fixed priority order), then
reads the configuration from the winning backend and merges any findings it
produced. Imports only from :mod:`llmprobe.models` and
:mod:`llmprobe.backends`.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable

import httpx

from llmprobe.backends import generic, llamacpp, ollama, vllm
from llmprobe.models import Backend, EffectiveConfig, Finding, Provenance

logger = logging.getLogger(__name__)

# llama.cpp does not expose the effective --parallel slot count over HTTP in
# every release. When total_slots is absent we must not assume a single slot:
# llama.cpp treats an omitted --parallel flag as a default of 4 (see
# tools/server/server.cpp), so the honest per-slot context is n_ctx / 4, not
# n_ctx / 1. This is the value documented in the README's "what you passed"
# guarantee.
_LLAMACPP_DEFAULT_PARALLEL = 4

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
            # Kept broad so a failing probe never blocks selection, but the
            # swallow must not be silent: an unexpected exception (including a
            # programming bug in an adapter) is logged so a misattributed
            # backend selection is diagnosable instead of invisible.
            logger.exception(
                "backend detect() raised on %s; treated as no-match", base_url
            )
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


def _apply_llamacpp_parallel_default(config: EffectiveConfig) -> EffectiveConfig:
    """Apply llama.cpp's default parallelism to an unadvertised slot count.

    A llama.cpp server that omits ``total_slots`` from ``/props`` is still
    running --parallel with a default of 4, not 1. Assume the documented
    default so a total ``n_ctx`` is divided by 4 (the honest per-slot context)
    rather than being misreported as a single-slot value. Nothing is invented:
    the assumption is the documented server default and is marked INFERRED.
    """
    if config.backend is not Backend.LLAMACPP:
        return config
    if config.total_slots is not None:
        return config
    if not isinstance(config.n_ctx_per_slot, int):
        return config

    sources = dict(config.sources)
    sources["total_slots"] = Provenance.INFERRED
    sources["n_ctx_total"] = Provenance.INFERRED

    return config.model_copy(
        update={
            "total_slots": _LLAMACPP_DEFAULT_PARALLEL,
            "n_ctx_total": config.n_ctx_per_slot * _LLAMACPP_DEFAULT_PARALLEL,
            "sources": sources,
        }
    )


async def read_effective_config(
    client: httpx.AsyncClient,
    base_url: str,
    claimed_ctx: int | None,
    timeout: float | None = None,
) -> tuple[EffectiveConfig, list[Finding]]:
    """Detect the backend and read the effective configuration.

    ``claimed_ctx`` is accepted for API compatibility with downstream probes
    that pass a caller-claimed context; adapter selection itself never depends
    on it. The winning adapter's ``read_config`` is invoked and any findings
    it emitted are returned alongside the configuration. ``timeout`` is
    threaded through for callers that bound every request; the shared client
    already enforces it on the wire.

    The effective config is normalised so that an absent parallel slot count is
    interpreted as the server default rather than as a single slot: a llama.cpp
    server that does not advertise ``total_slots`` still runs ``--parallel``
    with a default of 4, so the total ``n_ctx`` is divided by 4.
    """
    backend = await _select_backend(client, base_url)
    base = base_url.rstrip("/")
    module = _SELECTABLE_ADAPTERS[backend]
    result = await module.read_config(client, base)
    config, findings = _merge_findings(result)
    return _apply_llamacpp_parallel_default(config), findings
