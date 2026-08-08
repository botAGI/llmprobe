"""Tests for llmprobe.report — JSON + markdown capability card rendering.

Hermetic: no network, no real inference server. Uses golden files checked
into ``tests/golden/`` (one clean report, one with a silent-truncation
finding).

The module's point is honesty: the markdown capability card carries a
provenance marker on every table row. The regex assertion below enforces
that no row can render without a marker.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

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
from llmprobe.report import to_json, to_markdown

GOLDEN_DIR = Path(__file__).parent / "golden"

# A table data row rendered by this module has exactly five cells, and the
# fourth cell (Source) MUST be one of the provenance markers. Data rows are
# the lines beginning with "| " that are not the header or the separator.
_ROW = re.compile(r"^\| .+ \| .+ \| .+ \| (?P<source>read|measured|inferred|unknown) \| .+ \|$")


def _sources() -> dict[str, Provenance]:
    return {
        "n_ctx_total": Provenance.READ,
        "n_ctx_per_slot": Provenance.READ,
        "n_batch": Provenance.MEASURED,
        "n_ubatch": Provenance.INFERRED,
        "total_slots": Provenance.READ,
    }


def _clean_report() -> ProbeReport:
    return ProbeReport(
        base_url="http://localhost:8080",
        config=EffectiveConfig(
            backend=Backend.LLAMACPP,
            model_id="mock/llama-3.1-8b",
            n_ctx_total=8192,
            n_ctx_per_slot=2048,
            n_batch=512,
            n_ubatch=128,
            total_slots=4,
            sources=_sources(),
        ),
        capacity=[
            CapacityResult(
                endpoint="/completion",
                max_accepted_tokens=8192,
                cliff_behavior=CliffBehavior.ACCEPTED,
                probe_requests_used=3,
            )
        ],
        findings=[],
    )


def _silent_report() -> ProbeReport:
    return ProbeReport(
        base_url="http://localhost:8080",
        config=EffectiveConfig(
            backend=Backend.LLAMACPP,
            model_id="mock/llama-3.1-8b",
            n_ctx_total=8192,
            n_ctx_per_slot=2048,
            n_batch=512,
            n_ubatch=128,
            total_slots=4,
            sources=_sources(),
        ),
        capacity=[
            CapacityResult(
                endpoint="/completion",
                max_accepted_tokens=7168,
                cliff_behavior=CliffBehavior.SILENT_TRUNCATION,
                probe_requests_used=5,
            )
        ],
        findings=[
            Finding(
                severity=Severity.MISMATCH,
                code="UBATCH_CEILING",
                advertised=8192,
                measured=7168,
                message="requests past 7168 tokens are silently truncated",
            )
        ],
    )


def _table_data_rows(markdown: str) -> list[str]:
    """Return every non-header, non-separator table row."""
    rows: list[str] = []
    for line in markdown.splitlines():
        if not line.startswith("| "):
            continue
        if line.startswith("| Property |"):
            continue
        if line.startswith("| --- "):
            continue
        rows.append(line)
    return rows


def _golden(name: str) -> str:
    return (GOLDEN_DIR / name).read_text()


def test_clean_report_matches_golden() -> None:
    md = to_markdown(_clean_report())
    assert md == _golden("clean.md")


def test_silent_truncation_report_matches_golden() -> None:
    md = to_markdown(_silent_report())
    assert md == _golden("silent-truncation.md")


def test_every_table_data_row_has_a_provenance_marker() -> None:
    """THE point of this module: no row can render without a marker."""
    for report in (_clean_report(), _silent_report()):
        md = to_markdown(report)
        rows = _table_data_rows(md)
        assert rows, "expected at least one table data row"
        for row in rows:
            assert _ROW.match(row), (
                f"table row rendered without a provenance marker: {row!r}"
            )


def test_unknown_model_id_renders_source_unknown_not_read() -> None:
    """An unread model id must not be labelled ``read`` (honesty rule).

    A generic backend that cannot learn a model id reports ``model_id=""``.
    The report must surface that as ``unknown`` rather than claiming the
    server told us the value.
    """
    cfg = EffectiveConfig(
        backend=Backend.GENERIC,
        model_id="",
        sources={},
    )
    report = ProbeReport(base_url="http://x", config=cfg, capacity=[])
    md = to_markdown(report)
    model_row = next(
        row for row in _table_data_rows(md) if row.startswith("| model |")
    )
    assert "| model | unknown | unknown | unknown | unknown |" == model_row


def test_no_fix_section_when_nothing_wrong() -> None:
    assert "## Fix" not in to_markdown(_clean_report())


def test_fix_section_emits_batch_ubatch_flags_from_measured_cliff() -> None:
    md = to_markdown(_silent_report())
    assert "--batch-size 8192 --ubatch-size 8192" in md
    assert "## Fix" in md


def test_to_json_round_trips_through_model() -> None:
    report = _clean_report()
    dumped = to_json(report)
    reloaded = ProbeReport.model_validate_json(dumped)
    assert reloaded == report


def test_to_json_is_compact_json() -> None:
    dumped = to_json(_clean_report())
    json.loads(dumped)  # must be valid JSON
