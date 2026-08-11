"""Probe adapter for the llama.cpp HTTP server (``llama-server``).

Only imports from :mod:`llmprobe.models` — no other llmprobe modules.

Every reported value carries a provenance marker. llama.cpp does NOT expose
``n_batch`` or ``n_ubatch`` over HTTP, so both are left ``None`` with
provenance ``UNKNOWN`` rather than being invented.
"""

from __future__ import annotations

from typing import Any

import httpx

from llmprobe.models import Backend, EffectiveConfig, Provenance


def _base(base_url: str) -> str:
    return base_url.rstrip("/")


async def detect(client: httpx.AsyncClient, base_url: str) -> float:
    """Return a confidence (0..1) that ``base_url`` is a llama.cpp server.

    llama.cpp's ``/props`` response carries a ``build_info`` key that other
    OpenAI-compatible servers do not. Its presence is our detection signal.
    """
    try:
        resp = await client.get(f"{_base(base_url)}/props")
        resp.raise_for_status()
        payload = resp.json()
    except (httpx.HTTPError, ValueError):
        return 0.0

    if isinstance(payload, dict) and "build_info" in payload:
        return 1.0
    return 0.0


async def read_config(client: httpx.AsyncClient, base_url: str) -> EffectiveConfig:
    """Read the effective configuration advertised by a llama.cpp server.

    ``default_generation_settings.n_ctx`` is PER-SLOT. ``n_batch`` and
    ``n_ubatch`` are not exposed over HTTP, so they stay ``None`` with
    provenance ``UNKNOWN``.
    """
    base = _base(base_url)

    resp = await client.get(f"{base}/props", timeout=10.0)
    resp.raise_for_status()
    payload = resp.json()

    # The payload is expected to be a JSON object. A live server returning an
    # unexpected shape (array, scalar, plain whitespace) must not crash a probe:
    # treat anything that is not a dict as if it were an empty one so every
    # field below falls back to its default.
    props = payload if isinstance(payload, dict) else {}

    default_settings = props.get("default_generation_settings")
    if not isinstance(default_settings, dict):
        default_settings = {}
    n_ctx_per_slot = default_settings.get("n_ctx")
    total_slots = props.get("total_slots")
    model_id = str(props.get("model") or "")

    merge: dict[str, Provenance] = {
        "n_batch": Provenance.UNKNOWN,
        "n_ubatch": Provenance.UNKNOWN,
    }

    if total_slots is not None:
        merge["total_slots"] = Provenance.READ
    if n_ctx_per_slot is not None:
        merge["n_ctx_per_slot"] = Provenance.READ

    n_ctx_total: int | None = None
    if isinstance(n_ctx_per_slot, int) and isinstance(total_slots, int):
        n_ctx_total = n_ctx_per_slot * total_slots
        merge["n_ctx_total"] = Provenance.INFERRED
    else:
        merge["n_ctx_total"] = Provenance.UNKNOWN

    # /slots is optional: with --no-slots the server returns 501. Tolerate it.
    try:
        slots_resp = await client.get(f"{base}/slots", timeout=10.0)
        if slots_resp.status_code == 200:
            _source_slot_ctx(slots_resp.json(), merge)
    except httpx.HTTPError:
        pass

    return EffectiveConfig(
        backend=Backend.LLAMACPP,
        model_id=model_id,
        n_ctx_total=n_ctx_total,
        n_ctx_per_slot=n_ctx_per_slot,
        n_batch=None,
        n_ubatch=None,
        total_slots=total_slots,
        sources=merge,
    )


def _source_slot_ctx(payload: Any, merge: dict[str, Provenance]) -> None:
    """Use the first active slot's ``n_ctx`` as a cross-check when available.

    llama.cpp's ``/slots`` returns per-slot state; when a slot exposes its own
    ``n_ctx`` (READ, not derived) it is a direct observation and is at least as
    trustworthy as the per-slot default. Used only to refine the reported
    per-slot context; otherwise a no-op.
    """
    if not isinstance(payload, list) or not payload:
        return
    first = payload[0]
    if not isinstance(first, dict):
        return
    slot_ctx = first.get("n_ctx")
    if isinstance(slot_ctx, int):
        merge["n_ctx_per_slot"] = Provenance.READ
