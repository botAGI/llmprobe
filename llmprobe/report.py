"""Render a :class:`ProbeReport` as JSON or as a markdown capability card.

Pure formatting only: no I/O, no network, and no imports from any llmprobe
module other than :mod:`llmprobe.models`. The markdown card is the artifact
people paste into issues, so it must be tight and honest — every table row
carries a provenance marker (read / measured / inferred / unknown). A row
without a marker is a bug.
"""

from __future__ import annotations

from llmprobe.models import (
    CliffBehavior,
    ProbeReport,
    Provenance,
)

_HEADER_ROW = "| Property | Claimed | Measured | Source | Verdict |"
_SEPARATOR_ROW = "| --- | --- | --- | --- | --- |"

_CEILING_CODE_SUBSTRINGS = ("BATCH", "UBATCH", "CEILING")


def _is_ceiling_finding(code: str) -> bool:
    """True if a finding code describes a batch/ubatch context ceiling."""
    upper = code.upper()
    return any(sub in upper for sub in _CEILING_CODE_SUBSTRINGS)


def _next_power_of_two(value: int) -> int:
    """Smallest power of two >= value (values must be positive)."""
    n = 1
    while n < value:
        n <<= 1
    return n


def to_json(report: ProbeReport) -> str:
    """Render the report as compact JSON.

    This is a pass-through over the pydantic model; it exists so callers do
    not reach into serialization details and so the golden JSON is produced
    through a single, reviewable code path.
    """
    return report.model_dump_json()


def _fmt(value: object) -> str:
    """Format a cell value, rendering missing values honestly as ``unknown``."""
    if value is None:
        return "unknown"
    return str(value)


def _config_numeric_row(
    label: str,
    value: int | None,
    provenance: Provenance,
) -> str:
    claimed = _fmt(value)
    return (
        f"| {label} | {claimed} | {claimed} | {provenance.value} | "
        f"{'ok' if value is not None else 'unknown'} |"
    )


def _capacity_rows(report: ProbeReport) -> list[str]:
    rows: list[str] = []
    for cap in report.capacity:
        ceil = _cliff_verdict(cap.cliff_behavior)
        rows.append(
            f"| max input tokens ({cap.endpoint}) | unknown | "
            f"{_fmt(cap.max_accepted_tokens)} | {Provenance.MEASURED.value} | ok |"
        )
        rows.append(
            f"| cliff behaviour ({cap.endpoint}) | unknown | "
            f"{cap.cliff_behavior.value} | {Provenance.MEASURED.value} | {ceil} |"
        )
    return rows


def _cliff_verdict(behavior: CliffBehavior) -> str:
    if behavior == CliffBehavior.ACCEPTED:
        return "ok"
    if behavior == CliffBehavior.SILENT_TRUNCATION:
        return "truncated"
    return "error"


def _finding_lines(report: ProbeReport) -> list[str]:
    if not report.findings:
        return []
    lines = ["## Findings", ""]
    for finding in report.findings:
        adv = _fmt(finding.advertised)
        meas = _fmt(finding.measured)
        lines.append(
            f"- **[{finding.severity.value}] {finding.code}**: "
            f"advertised={adv} vs measured={meas} — {finding.message}"
        )
    return lines


def _fix_lines(report: ProbeReport) -> list[str]:
    """Emit the exact llama.cpp remedy flags for batching-ceiling findings.

    The Fix section is printed only when at least one finding indicates a
    batch/ubatch context ceiling. The remedy value ``N`` is derived from the
    measured cliff rounded up to the next power of two. When there is nothing
    wrong, the section is omitted entirely.
    """
    ceiling = [f for f in report.findings if _is_ceiling_finding(f.code)]
    if not ceiling:
        return []

    ceiling.sort(key=lambda f: f.code)
    measured = ceiling[0].measured
    if not isinstance(measured, int) or measured <= 0:
        measured = ceiling[0].advertised
    if not isinstance(measured, int) or measured <= 0:
        return []

    n = _next_power_of_two(measured)
    return [
        "## Fix",
        "",
        f"--batch-size {n} --ubatch-size {n}",
    ]


def to_markdown(report: ProbeReport) -> str:
    """Render the report as a tight, honest markdown capability card."""
    rows: list[str] = [
        f"# Capability Report — {report.base_url}",
        "",
        _HEADER_ROW,
        _SEPARATOR_ROW,
    ]

    config = report.config
    rows.append(
        f"| backend | {config.backend.value} | {config.backend.value} | "
        f"{Provenance.READ.value} | ok |"
    )
    model = config.model_id if config.model_id else "unknown"
    rows.append(
        f"| model | {model} | {model} | {Provenance.READ.value} | "
        f"{'ok' if config.model_id else 'unknown'} |"
    )
    rows.append(
        _config_numeric_row(
            "context (total)",
            config.n_ctx_total,
            config.sources.get("n_ctx_total", Provenance.UNKNOWN),
        )
    )
    rows.append(
        _config_numeric_row(
            "context (per slot)",
            config.n_ctx_per_slot,
            config.sources.get("n_ctx_per_slot", Provenance.UNKNOWN),
        )
    )
    rows.append(
        _config_numeric_row(
            "slots",
            config.total_slots,
            config.sources.get("total_slots", Provenance.UNKNOWN),
        )
    )
    rows.extend(_capacity_rows(report))

    parts = ["\n".join(rows)]
    findings = _finding_lines(report)
    if findings:
        parts.append("\n".join(findings))
    fix = _fix_lines(report)
    if fix:
        parts.append("\n".join(fix))

    return "\n\n".join(parts) + "\n"
