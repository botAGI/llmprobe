"""Render a :class:`ProbeReport` as JSON or as a markdown capability card.

Pure formatting only: no network, and no imports from any llmprobe module
other than :mod:`llmprobe.models`. The one disk read is our own version,
looked up once at import from package metadata / ``pyproject.toml`` so the
card can carry that static value. The markdown card is the artifact people
paste into issues, so it must be tight and honest — every table row carries a
provenance marker (read / measured / inferred / unknown). A row without a
marker is a bug.

The card also names the tool version and the (UTC) measurement time in the
header so a pasted card is self-describing.

The JSON form must uphold the same promise: every reported value is emitted
as a ``{"value": ..., "provenance": ...}`` object so that a machine consumer
can tell a measured number from an honest unknown. A value emitted without a
provenance key is a bug.
"""

from __future__ import annotations

import json
import tomllib
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from llmprobe.models import (
    CliffBehavior,
    ProbeReport,
    Provenance,
)


def _tool_version() -> str:
    """Return the installed llmprobe version, or ``unknown`` if it is hidden.

    Prefers the ``version`` key in this repository's ``pyproject.toml`` (the
    source of truth when running from a checkout), falling back to installed
    package metadata, then to the honest ``unknown`` marker.
    """
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    if pyproject.exists():
        try:
            with pyproject.open("rb") as fh:
                data = tomllib.load(fh)
        except (tomllib.TOMLDecodeError, OSError):
            data = {}
        version = data.get("project", {}).get("version")
        if version:
            return str(version)
    try:
        return metadata.version("llmprobe")
    except metadata.PackageNotFoundError:
        return "unknown"


_VERSION = _tool_version()

_HEADER_ROW = "| Property | Claimed | Measured | Source | Verdict |"
_SEPARATOR_ROW = "| --- | --- | --- | --- | --- |"

_CEILING_CODE_SUBSTRINGS = ("BATCH", "UBATCH", "CEILING")

# Markdown meta-characters that can inject structure (headings, emphasis,
# code, links, HTML, table separators) when a user- or server-provided string
# is interpolated verbatim into the card. Controlled internal identifiers
# (finding codes, enum values) are intentionally not treated as untrusted, so
# the set deliberately leaves ``-``/``.`` untouched.
_MD_INJECTION_CHARS = set("\\`*_[]()#+!|<>~{}")


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


def _pv(value: Any, provenance: Provenance) -> dict[str, Any]:
    """Bundle a reported value with its provenance marker.

    Every leaf of the JSON report is emitted as ``{"value": ..., "provenance":
    ...}`` so the "provenance on every value" promise from the README holds for
    the machine-readable form too, not just the markdown card.
    """
    return {"value": value, "provenance": provenance.value}


def _config_provenance(config: Any, field: str) -> Provenance:
    """Provenance for a config field, defaulting honestly to ``unknown``.

    Backends record provenance per-field in ``config.sources``; a field with no
    recorded source is reported ``unknown`` rather than silently guessed.
    """
    return config.sources.get(field, Provenance.UNKNOWN)


def _to_json_object(report: ProbeReport) -> dict[str, Any]:
    """Build the provenance-augmented JSON representation of a report."""
    config = report.config
    config_json: dict[str, Any] = {
        "backend": _pv(config.backend.value, Provenance.READ),
        "model_id": _pv(
            config.model_id,
            Provenance.READ if config.model_id else Provenance.UNKNOWN,
        ),
        "n_ctx_total": _pv(
            config.n_ctx_total, _config_provenance(config, "n_ctx_total")
        ),
        "n_ctx_per_slot": _pv(
            config.n_ctx_per_slot, _config_provenance(config, "n_ctx_per_slot")
        ),
        "n_batch": _pv(
            config.n_batch, _config_provenance(config, "n_batch")
        ),
        "n_ubatch": _pv(
            config.n_ubatch, _config_provenance(config, "n_ubatch")
        ),
        "total_slots": _pv(
            config.total_slots, _config_provenance(config, "total_slots")
        ),
    }

    capacity = [
        {
            "endpoint": _pv(cap.endpoint, Provenance.READ),
            "max_accepted_tokens": _pv(
                cap.max_accepted_tokens, cap.max_accepted_source
            ),
            "token_count_exact": _pv(
                cap.token_count_exact, Provenance.MEASURED
            ),
            "cliff_behavior": _pv(
                cap.cliff_behavior.value, Provenance.MEASURED
            ),
            "probe_requests_used": _pv(
                cap.probe_requests_used, Provenance.MEASURED
            ),
        }
        for cap in report.capacity
    ]

    findings = [
        {
            "severity": _pv(f.severity.value, Provenance.INFERRED),
            "code": _pv(f.code, Provenance.READ),
            "advertised": _pv(
                f.advertised,
                f.advertised_source if f.advertised is not None else Provenance.UNKNOWN,
            ),
            "measured": _pv(
                f.measured,
                f.measured_source if f.measured is not None else Provenance.UNKNOWN,
            ),
            "message": _pv(f.message, Provenance.READ),
        }
        for f in report.findings
    ]

    return {
        "base_url": _pv(report.base_url, Provenance.READ),
        "config": config_json,
        "capacity": capacity,
        "findings": findings,
    }


def to_json(report: ProbeReport) -> str:
    """Render the report as compact JSON with provenance on every value.

    Unlike a bare pydantic dump, every reported leaf is wrapped as
    ``{"value": ..., "provenance": ...}`` so a machine consumer can always tell
    a measured number from an honest unknown. The structure is produced through
    a single reviewable code path; callers never reach into serialization
    details.
    """
    return json.dumps(_to_json_object(report), separators=(",", ":"))


def _fmt(value: object) -> str:
    """Format a cell value, rendering missing values honestly as ``unknown``."""
    if value is None:
        return "unknown"
    return str(value)


def _esc(value: object) -> str:
    """Escape untrusted text before it is interpolated into markdown.

    Backslash-escapes every markdown meta-character so a user- or
    server-provided string (model name, base URL, endpoint, finding message)
    cannot inject headings, emphasis, inline code, links, HTML, or break out
    of a table cell. ``None`` renders as the honest ``unknown`` marker.
    """
    if value is None:
        return "unknown"
    out: list[str] = []
    for ch in str(value):
        if ch in _MD_INJECTION_CHARS:
            out.append("\\")
        out.append(ch)
    return "".join(out)


def _sanitize_base_url(url: str) -> str:
    """Strip URL-embedded credentials so API keys are never printed.

    A base URL may carry a secret in its userinfo slot, e.g.
    ``http://sk-1234@localhost:8080`` or ``http://user:token@host/v1``. The
    report is pasted into issues and logs, so the credentials must not leak.
    URLs without userinfo are returned unchanged.
    """
    parts = urlsplit(url if "://" in url else f"//{url}")
    if not (parts.username or parts.password):
        return url
    hostname = parts.hostname or ""
    port = f":{parts.port}" if parts.port else ""
    cleaned = urlunsplit(
        (parts.scheme, f"{hostname}{port}", parts.path, parts.query, parts.fragment)
    )
    return cleaned if cleaned else url


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
        endpoint = _esc(cap.endpoint)
        rows.append(
            f"| max input tokens ({endpoint}) | unknown | "
            f"{_fmt(cap.max_accepted_tokens)} | {cap.max_accepted_source.value} | ok |"
        )
        token_exact = "exact" if cap.token_count_exact else "estimate"
        rows.append(
            f"| token count ({endpoint}) | unknown | {token_exact} | "
            f"{Provenance.MEASURED.value} | ok |"
        )
        rows.append(
            f"| cliff behaviour ({endpoint}) | unknown | "
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
        adv = _esc(finding.advertised)
        meas = _esc(finding.measured)
        adv_marker = (
            f" ({finding.advertised_source.value})"
            if finding.advertised is not None
            else ""
        )
        meas_marker = (
            f" ({finding.measured_source.value})"
            if finding.measured is not None
            else ""
        )
        lines.append(
            f"- **[{finding.severity.value}] {finding.code}**: "
            f"advertised={adv}{adv_marker} vs measured={meas}{meas_marker} — "
            f"{_esc(finding.message)}"
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


def to_markdown(
    report: ProbeReport,
    *,
    version: str = _VERSION,
    measured_at: datetime | None = None,
) -> str:
    """Render the report as a tight, honest markdown capability card.

    The card title is followed by a header line naming the tool ``version``
    and the ``measured_at`` measurement time in UTC, so a pasted card is
    self-describing. When ``measured_at`` is omitted the current UTC time is
    used; callers that need a deterministic card (tests, recorded fixtures)
    pass an explicit value.
    """
    if measured_at is None:
        measured_at = datetime.now(timezone.utc)
    ts = measured_at.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows: list[str] = [
        f"# Capability Report — {_esc(_sanitize_base_url(report.base_url))}",
        "",
        f"llmprobe {version} · measured {ts} (UTC)",
        "",
        _HEADER_ROW,
        _SEPARATOR_ROW,
    ]

    config = report.config
    backend = _esc(config.backend.value)
    rows.append(
        f"| backend | {backend} | {backend} | "
        f"{Provenance.READ.value} | ok |"
    )
    model = _esc(config.model_id) if config.model_id else "unknown"
    model_source = (
        Provenance.READ.value if config.model_id else Provenance.UNKNOWN.value
    )
    rows.append(
        f"| model | {model} | {model} | {model_source} | "
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
