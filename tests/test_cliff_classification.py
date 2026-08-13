"""The cliff must be classified against what the server itself reported.

Three defects are guarded here, all found by driving the real CLI against real
llama.cpp servers (b9049, one with the default ``n_ubatch``, one with
``-b/-ub 8192``) rather than against a mock:

1. A HEALTHY server false-alarmed. Measured capacity lands two tokens below the
   configured context (BOS/EOS), so ``measured < claimed`` fired a MISMATCH and
   exit 1 on a correctly configured server. A gate that reddens on healthy
   servers is worse than no gate.
2. A BROKEN server stayed silent without ``--claimed-ctx``. The server reports
   its own per-slot context via ``/props``; the shortfall was measurable without
   the operator knowing any number, which is the entire premise of the tool.
3. The remedy echoed the broken value. ``--batch-size 512 --ubatch-size 512``
   was advised for a server broken *by* ``ubatch=512``, because the value came
   from the measured cliff instead of the desired context.
"""

from __future__ import annotations

from llmprobe.cli import _capacity_findings
from llmprobe.models import (
    Backend,
    CapacityResult,
    CliffBehavior,
    EffectiveConfig,
    Finding,
    Provenance,
    Severity,
)
from llmprobe.report import _fix_lines
from llmprobe.models import ProbeReport


def _config(
    *,
    ctx_per_slot: int | None,
    source: Provenance = Provenance.READ,
    backend: Backend = Backend.LLAMACPP,
) -> EffectiveConfig:
    sources = {} if ctx_per_slot is None else {"n_ctx_per_slot": source}
    return EffectiveConfig(
        backend=backend,
        model_id="m",
        n_ctx_per_slot=ctx_per_slot,
        sources=sources,
    )


def _cap(measured: int, behavior: CliffBehavior = CliffBehavior.HARD_ERROR):
    return CapacityResult(
        endpoint="/v1/embeddings",
        max_accepted_tokens=measured,
        cliff_behavior=behavior,
        probe_requests_used=34,
    )


def test_healthy_server_two_tokens_short_is_not_a_mismatch() -> None:
    """8190 measured against 8192 configured is BOS/EOS overhead, not a defect.

    Live control: llama.cpp b9049 started with ``-b 8192 -ub 8192`` accepts a
    2002-token embedding request and reports ``n_ctx_per_slot=8192``; the probe
    measures 8190 because the tokenizer adds two special tokens. Emitting a
    MISMATCH here reddens a correctly configured server.
    """
    findings = _capacity_findings(8192, _cap(8190), _config(ctx_per_slot=8192))
    assert findings == [], f"false alarm on a healthy server: {findings}"


def test_broken_server_is_caught_without_any_claimed_ctx() -> None:
    """The server's own reported context is baseline enough to fail the gate.

    Live control: the same image with the default ``n_ubatch`` reports
    ``n_ctx_per_slot=8192`` and hard-errors past 510 tokens. The operator does
    not know the true ceiling — that is why they run the tool — so the finding
    must come from the server's own numbers, with no ``--claimed-ctx`` given.
    """
    findings = _capacity_findings(None, _cap(510), _config(ctx_per_slot=8192))
    assert len(findings) == 1, "a 16x shortfall below the server's own context"
    assert findings[0].severity == Severity.MISMATCH
    assert findings[0].advertised == 8192
    assert findings[0].measured == 510


def test_cause_is_only_named_when_the_server_reported_its_context() -> None:
    """Without a read context there is nothing to blame the batch ceiling for.

    ``UBATCH_CEILING`` asserts a cause. It is only defensible when the server
    itself said it was configured for more than it delivers; with only an
    operator-supplied number, the honest code says capacity fell short and
    names no cause.
    """
    named = _capacity_findings(None, _cap(510), _config(ctx_per_slot=8192))
    assert named[0].code == "UBATCH_CEILING"

    unnamed = _capacity_findings(8192, _cap(510), _config(ctx_per_slot=None))
    assert len(unnamed) == 1
    assert unnamed[0].code != "UBATCH_CEILING", (
        "named a batch ceiling without reading the server's context"
    )


def test_inferred_context_is_not_used_as_a_baseline() -> None:
    """Only a value the server actually reported may accuse it.

    A derived ``n_ctx_per_slot`` is our own arithmetic; comparing a measurement
    against our own guess and calling the difference a server defect is exactly
    the confident guess this tool exists to refuse.
    """
    findings = _capacity_findings(
        None, _cap(510), _config(ctx_per_slot=8192, source=Provenance.INFERRED)
    )
    assert findings == [], "an inferred context must not accuse the server"


def _report(finding: Finding, backend: Backend = Backend.LLAMACPP) -> ProbeReport:
    return ProbeReport(
        base_url="http://h",
        config=_config(ctx_per_slot=8192, backend=backend),
        capacity=[_cap(510)],
        findings=[finding],
    )


def test_remedy_never_echoes_the_broken_value() -> None:
    """Advising ``--ubatch-size 512`` to a server broken by 512 is active harm.

    The remedy must restore the context the server was configured for, so the
    number comes from ``advertised`` (8192), never from the measured cliff.
    """
    fix = "\n".join(
        _fix_lines(
            _report(
                Finding(
                    severity=Severity.MISMATCH,
                    code="UBATCH_CEILING",
                    advertised=8192,
                    measured=510,
                    message="requests past 510 tokens are hard_error",
                )
            )
        )
    )
    assert "512" not in fix, f"remedy echoes the broken ceiling: {fix}"
    assert "8192" in fix, f"remedy does not restore the configured context: {fix}"


def test_remedy_is_not_offered_for_a_backend_that_lacks_those_flags() -> None:
    """``--ubatch-size`` is llama.cpp's; handing it to Ollama is nonsense advice.

    Ollama takes ``num_ctx``/``OLLAMA_CONTEXT_LENGTH``; printing llama.cpp
    server flags there is a confident guess dressed as a remedy.
    """
    fix = "\n".join(
        _fix_lines(
            _report(
                Finding(
                    severity=Severity.MISMATCH,
                    code="UBATCH_CEILING",
                    advertised=8192,
                    measured=510,
                    message="requests past 510 tokens are silently truncated",
                ),
                backend=Backend.OLLAMA,
            )
        )
    )
    assert "--ubatch-size" not in fix, f"llama.cpp flags advised to Ollama: {fix}"
