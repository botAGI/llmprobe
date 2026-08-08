"""Tests for llmprobe.models — shared pydantic contract."""

import pytest
from pydantic import ValidationError

from llmprobe.models import (
    Backend,
    CapacityResult,
    CliffBehavior,
    EffectiveConfig,
    Finding,
    ProbeReport,
    Provenance,
    Severity,
)


def _effective_config(sources_ok: bool = True) -> EffectiveConfig:
    sources = (
        {
            "n_ctx_total": Provenance.READ,
            "n_ctx_per_slot": Provenance.UNKNOWN,
            "n_batch": Provenance.MEASURED,
            "n_ubatch": Provenance.INFERRED,
            "total_slots": Provenance.READ,
        }
        if sources_ok
        else {}
    )
    return EffectiveConfig(
        backend=Backend.VLLM,
        model_id="test/model",
        n_ctx_total=4096,
        n_ctx_per_slot=2048,
        n_batch=512,
        n_ubatch=128,
        total_slots=8,
        sources=sources,
    )


def _finding(severity: Severity) -> Finding:
    return Finding(
        severity=severity,
        code=f"TEST-{severity.value}",
        message=f"test {severity.value}",
    )


def _report(findings: list[Finding]) -> ProbeReport:
    return ProbeReport(
        base_url="http://localhost:8080",
        config=_effective_config(),
        capacity=[
            CapacityResult(
                endpoint="/v1/completions",
                max_accepted_tokens=4096,
                cliff_behavior=CliffBehavior.ACCEPTED,
                probe_requests_used=1,
            )
        ],
        findings=findings,
    )


@pytest.mark.parametrize(
    "model",
    [
        _effective_config(),
        _finding(Severity.INFO),
        _report([]),
    ],
)
def test_all_models_round_trip(model):
    """Every model round-trips through model_dump_json and back unchanged."""
    dumped = model.model_dump_json()
    reloaded = type(model).model_validate_json(dumped)
    assert reloaded == model


def test_every_numeric_field_has_provenance():
    """Every numeric field in EffectiveConfig has a provenance entry."""
    config = _effective_config()
    numeric_fields = [
        "n_ctx_total",
        "n_ctx_per_slot",
        "n_batch",
        "n_ubatch",
        "total_slots",
    ]
    for field in numeric_fields:
        assert field in config.sources
        assert isinstance(config.sources[field], Provenance)


def test_exit_code_zero_when_no_findings():
    assert _report([]).exit_code == 0
    assert _report([_finding(Severity.INFO)]).exit_code == 0


def test_exit_code_one_on_mismatch():
    assert _report([_finding(Severity.MISMATCH)]).exit_code == 1
    assert _report([_finding(Severity.INFO), _finding(Severity.MISMATCH)]).exit_code == 1


def test_exit_code_two_on_error():
    assert _report([_finding(Severity.ERROR)]).exit_code == 2
    assert _report([_finding(Severity.MISMATCH), _finding(Severity.ERROR)]).exit_code == 2


def test_numeric_fields_are_optional():
    config = _effective_config()
    config.n_ctx_total = None
    dumped = config.model_dump_json()
    reloaded = EffectiveConfig.model_validate_json(dumped)
    assert reloaded.n_ctx_total is None
