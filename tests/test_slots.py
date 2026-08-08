"""Tests for :mod:`llmprobe.probes.slots`.

Hermetic: pure computation over ``EffectiveConfig``, no network.
"""

from __future__ import annotations

from llmprobe.models import Backend, EffectiveConfig, Finding, Provenance, Severity
from llmprobe.probes.slots import check_slots


def _config(
    *,
    n_ctx_total: int | None,
    n_ctx_per_slot: int | None,
    total_slots: int | None,
) -> EffectiveConfig:
    sources: dict[str, Provenance] = {}
    for field, value in (
        ("n_ctx_total", n_ctx_total),
        ("n_ctx_per_slot", n_ctx_per_slot),
        ("total_slots", total_slots),
    ):
        if value is not None:
            sources[field] = Provenance.READ
    return EffectiveConfig(
        backend=Backend.LLAMACPP,
        model_id="test-model",
        n_ctx_total=n_ctx_total,
        n_ctx_per_slot=n_ctx_per_slot,
        total_slots=total_slots,
        sources=sources,
    )


def test_consistent_values_produce_no_finding() -> None:
    config = _config(n_ctx_total=8192, n_ctx_per_slot=2048, total_slots=4)
    assert check_slots(config, claimed_ctx=2048) == []


def test_claimed_ctx_mismatch_emits_finding() -> None:
    config = _config(n_ctx_total=8192, n_ctx_per_slot=2048, total_slots=4)
    assert check_slots(config, claimed_ctx=8192) == [
        Finding(
            severity=Severity.MISMATCH,
            code="CTX_PER_SLOT_MISMATCH",
            advertised=8192,
            measured=2048,
            message=(
                "derived per-slot context (2048) disagrees with "
                "reported per-slot context (2048)"
            ),
        )
    ]


def test_unknown_total_slots_produces_no_finding() -> None:
    config = _config(n_ctx_total=8192, n_ctx_per_slot=8192, total_slots=None)
    assert check_slots(config, claimed_ctx=2048) == []


def test_unknown_total_ctx_produces_no_finding() -> None:
    config = _config(n_ctx_total=None, n_ctx_per_slot=2048, total_slots=4)
    assert check_slots(config, claimed_ctx=8192) == []


def test_regression_llamacpp_absent_parallel_defaults_to_four() -> None:
    # Regression test: llama.cpp treats an absent --parallel flag as 4, NOT 1.
    # See server.cpp lines 151-155. A server configured with n_ctx=8192 across
    # 4 slots exposes n_ctx_total=8192 and total_slots=4, but a caller who
    # assumes a single slot claims claimed_ctx=8192. The derived per-slot
    # context is 8192 // 4 = 2048, so we must emit a mismatch with
    # measured == 2048.
    config = _config(n_ctx_total=8192, n_ctx_per_slot=2048, total_slots=4)
    findings = check_slots(config, claimed_ctx=8192)
    assert len(findings) == 1
    assert findings[0].code == "CTX_PER_SLOT_MISMATCH"
    assert findings[0].measured == 2048
