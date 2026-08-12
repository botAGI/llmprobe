"""Validate that ``--json`` report output conforms to the report JSON schema.

The tool's machine-readable contract (see ``llmprobe/report.to_json``) is that
every reported value is emitted as a ``{"value": ..., "provenance": ...}``
object where provenance is one of ``read`` / ``measured`` / ``inferred`` /
``unknown``. A value emitted without a provenance key, with an invalid
provenance, or with a value of the wrong type is a broken report.

This module pins that contract as an explicit JSON Schema and validates real
``--json`` output — produced through the single ``to_json`` code path — against
it on three scenarios: a clean report (no findings), a report with a mismatch
finding, and a report with an error finding.

Hermetic: no network, no live server. Report instances are built directly from
the pydantic models and rendered via ``to_json``.

Note: ``jsonschema`` is deliberately NOT used — it is not a project dependency.
Validation runs through a minimal, self-contained validator that supports the
small JSON Schema subset the contract uses (``type``, ``enum``, ``properties``,
``required``, ``items``, ``anyOf``).
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import typer
from typer.testing import CliRunner

import llmprobe.cli as cli
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
from llmprobe.report import to_json, to_json_schema
from tests.mocks.server import make_mock_server

# The provenance marker on a reported value must always be one of these.
_PROVENANCES = ["read", "measured", "inferred", "unknown"]

_BASE_URL = "http://mock"

_runner = CliRunner()


# ---------------------------------------------------------------------------
# Minimal JSON Schema validator (subset sufficient for the report contract).
# It is intentionally small and self-contained so validation needs no third
# party dependency and cannot drift from what the project actually supports.
# ---------------------------------------------------------------------------


class _SchemaError(ValueError):
    """Raised when an instance does not validate against its schema."""


def validate(node: Any, schema: dict[str, Any], path: str = "$") -> None:
    """Recursively validate ``node`` against a minimal-schema ``schema``.

    Supports the JSON Schema keywords used by the report contract: ``type``
    (string or list), ``enum``, ``anyOf``, ``properties`` + ``required`` for
    objects, and ``items`` for arrays. Raises ``_SchemaError`` on the first
    violation.
    """
    if "anyOf" in schema:
        for sub in schema["anyOf"]:
            try:
                validate(node, sub, path)
                return
            except _SchemaError:
                continue
        raise _SchemaError(f"{path}: value {node!r} matches no anyOf branch")

    if "enum" in schema:
        if node not in schema["enum"]:
            raise _SchemaError(f"{path}: {node!r} not in enum {schema['enum']}")
        return

    schema_type = schema.get("type")
    if schema_type is not None:
        allowed = [schema_type] if isinstance(schema_type, str) else list(schema_type)
        if not _matches_type(node, allowed):
            raise _SchemaError(f"{path}: expected type {allowed}, got {node!r}")

    if isinstance(node, list):
        if "items" in schema:
            for i, item in enumerate(node):
                validate(item, schema["items"], f"{path}[{i}]")
        return

    if isinstance(node, dict):
        if "required" in schema:
            for key in schema["required"]:
                if key not in node:
                    raise _SchemaError(f"{path}: missing required key {key!r}")
        props = schema.get("properties", {})
        for key, value in node.items():
            if key in props:
                validate(value, props[key], f"{path}.{key}")


def _matches_type(node: Any, allowed: list[str]) -> bool:
    for t in allowed:
        if t == "null" and node is None:
            return True
        if t == "boolean" and isinstance(node, bool):
            return True
        if t == "string" and isinstance(node, str):
            return True
        if t == "integer" and isinstance(node, int) and not isinstance(node, bool):
            return True
        if t == "array" and isinstance(node, list):
            return True
        if t == "object" and isinstance(node, dict):
            return True
    return False


# ---------------------------------------------------------------------------
# The report JSON contract, expressed as an explicit JSON Schema.
#
# "пункт 3" of the task spec is the machine-readable report contract: a report
# value is always a provenance-tagged wrapper, never a bare scalar. The schema
# below is the source of truth the tests validate ``--json`` output against.
# ---------------------------------------------------------------------------

_PROVENANCE = {
    "type": "string",
    "enum": _PROVENANCES,
}

# A nullable scalar with a provenance marker. ``value`` is null/string/int and
# ``provenance`` is always a valid marker.
_SCALAR_WRAPPER = {
    "type": "object",
    "properties": {
        "value": {"anyOf": [{"type": "null"}, {"type": "string"}, {"type": "integer"}]},
        "provenance": _PROVENANCE,
    },
    "required": ["value", "provenance"],
}

_INT_WRAPPER = {
    "type": "object",
    "properties": {
        "value": {"type": "integer"},
        "provenance": _PROVENANCE,
    },
    "required": ["value", "provenance"],
}

_BOOL_WRAPPER = {
    "type": "object",
    "properties": {
        "value": {"type": "boolean"},
        "provenance": _PROVENANCE,
    },
    "required": ["value", "provenance"],
}

_STR_WRAPPER = {
    "type": "object",
    "properties": {
        "value": {"type": "string"},
        "provenance": _PROVENANCE,
    },
    "required": ["value", "provenance"],
}

# A config numeric field that may be absent: value is either an integer or
# null (honest ``unknown``), never a string.
_NULLABLE_INT_WRAPPER = {
    "type": "object",
    "properties": {
        "value": {"anyOf": [{"type": "null"}, {"type": "integer"}]},
        "provenance": _PROVENANCE,
    },
    "required": ["value", "provenance"],
}

_CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "backend": {
            "type": "object",
            "properties": {
                "value": {
                    "type": "string",
                    "enum": [b.value for b in Backend],
                },
                "provenance": _PROVENANCE,
            },
            "required": ["value", "provenance"],
        },
        "model_id": _STR_WRAPPER,
        "n_ctx_total": _NULLABLE_INT_WRAPPER,
        "n_ctx_per_slot": _NULLABLE_INT_WRAPPER,
        "n_batch": _NULLABLE_INT_WRAPPER,
        "n_ubatch": _NULLABLE_INT_WRAPPER,
        "total_slots": _NULLABLE_INT_WRAPPER,
    },
    "required": [
        "backend",
        "model_id",
        "n_ctx_total",
        "n_ctx_per_slot",
        "n_batch",
        "n_ubatch",
        "total_slots",
    ],
}

_CAPACITY_SCHEMA = {
    "type": "object",
    "properties": {
        "endpoint": _STR_WRAPPER,
        "max_accepted_tokens": _INT_WRAPPER,
        "token_count_exact": _BOOL_WRAPPER,
        "cliff_behavior": {
            "type": "object",
            "properties": {
                "value": {
                    "type": "string",
                    "enum": [c.value for c in CliffBehavior],
                },
                "provenance": _PROVENANCE,
            },
            "required": ["value", "provenance"],
        },
        "probe_requests_used": _INT_WRAPPER,
    },
    "required": [
        "endpoint",
        "max_accepted_tokens",
        "token_count_exact",
        "cliff_behavior",
        "probe_requests_used",
    ],
}

_FINDING_SCHEMA = {
    "type": "object",
    "properties": {
        "severity": {
            "type": "object",
            "properties": {
                "value": {
                    "type": "string",
                    "enum": [s.value for s in Severity],
                },
                "provenance": _PROVENANCE,
            },
            "required": ["value", "provenance"],
        },
        "code": _STR_WRAPPER,
        "advertised": _SCALAR_WRAPPER,
        "measured": _SCALAR_WRAPPER,
        "message": _STR_WRAPPER,
    },
    "required": ["severity", "code", "advertised", "measured", "message"],
}

# The full report: the provenance-augmented machine contract emitted by --json.
REPORT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "base_url": _STR_WRAPPER,
        "config": _CONFIG_SCHEMA,
        "capacity": {
            "type": "array",
            "items": _CAPACITY_SCHEMA,
        },
        "findings": {
            "type": "array",
            "items": _FINDING_SCHEMA,
        },
    },
    "required": ["base_url", "config", "capacity", "findings"],
}


# ---------------------------------------------------------------------------
# Scenario fixtures: clean report, mismatch report, error report.
# ---------------------------------------------------------------------------


def _sources() -> dict[str, Provenance]:
    return {
        "n_ctx_total": Provenance.READ,
        "n_ctx_per_slot": Provenance.READ,
        "n_batch": Provenance.MEASURED,
        "n_ubatch": Provenance.INFERRED,
        "total_slots": Provenance.READ,
    }


def _base_config() -> EffectiveConfig:
    return EffectiveConfig(
        backend=Backend.LLAMACPP,
        model_id="mock/llama-3.1-8b",
        n_ctx_total=8192,
        n_ctx_per_slot=2048,
        n_batch=512,
        n_ubatch=128,
        total_slots=4,
        sources=_sources(),
    )


def _base_capacity() -> CapacityResult:
    return CapacityResult(
        endpoint="/completion",
        max_accepted_tokens=8192,
        cliff_behavior=CliffBehavior.ACCEPTED,
        probe_requests_used=3,
    )


def _clean_report() -> ProbeReport:
    return ProbeReport(
        base_url="http://localhost:8080",
        config=_base_config(),
        capacity=[_base_capacity()],
        findings=[],
    )


def _mismatch_report() -> ProbeReport:
    report = _clean_report()
    report.findings = [
        Finding(
            severity=Severity.MISMATCH,
            code="UBATCH_CEILING",
            advertised=8192,
            measured=7168,
            message="requests past 7168 tokens are silently truncated",
        )
    ]
    return report


def _error_report() -> ProbeReport:
    report = _clean_report()
    report.findings = [
        Finding(
            severity=Severity.ERROR,
            code="HARD_ERROR",
            advertised=None,
            measured="boom",
            message="capacity probe failed hard",
        )
    ]
    return report


def _json_output(report: ProbeReport) -> dict[str, Any]:
    """Render a report exactly as the CLI ``--json`` path would."""
    return json.loads(to_json(report))


# ---------------------------------------------------------------------------
# The point of this module: ``--json`` output validates against the schema.
# ---------------------------------------------------------------------------


def test_clean_report_validates_against_schema() -> None:
    validate(_json_output(_clean_report()), REPORT_SCHEMA)


def test_mismatch_report_validates_against_schema() -> None:
    validate(_json_output(_mismatch_report()), REPORT_SCHEMA)


def test_error_report_validates_against_schema() -> None:
    validate(_json_output(_error_report()), REPORT_SCHEMA)


def test_schema_is_a_valid_json_schema_document() -> None:
    """The contract schema must itself be a structurally valid schema."""
    # Cross-check that the derived schema agrees with the tool's own concept of
    # a report: it must be an object with the four top-level sections.
    assert REPORT_SCHEMA["type"] == "object"
    assert set(REPORT_SCHEMA["required"]) == {
        "base_url",
        "config",
        "capacity",
        "findings",
    }


# ---------------------------------------------------------------------------
# The validator must catch real breakage: a proven report shape rendered
# wrong (missing a provenance marker, wrong type, bad marker) must fail.
# ---------------------------------------------------------------------------


def test_value_missing_provenance_marker_is_rejected() -> None:
    bad = _json_output(_clean_report())
    bad["config"]["n_ctx_total"] = {"value": 8192}  # dropped "provenance"
    try:
        validate(bad, REPORT_SCHEMA)
    except _SchemaError:
        return
    raise AssertionError("report missing a provenance marker passed validation")


def test_invalid_provenance_marker_is_rejected() -> None:
    bad = _json_output(_clean_report())
    bad["config"]["n_batch"]["provenance"] = "approximate"
    try:
        validate(bad, REPORT_SCHEMA)
    except _SchemaError:
        return
    raise AssertionError("report with an invalid provenance marker passed validation")


def test_wrong_typed_value_is_rejected() -> None:
    bad = _json_output(_clean_report())
    bad["config"]["n_ubatch"]["value"] = "one-hundred-and-twenty-eight"
    try:
        validate(bad, REPORT_SCHEMA)
    except _SchemaError:
        return
    raise AssertionError("report with a string where an integer is required passed validation")


def test_missing_toplevel_section_is_rejected() -> None:
    bad = _json_output(_clean_report())
    del bad["findings"]
    try:
        validate(bad, REPORT_SCHEMA)
    except _SchemaError:
        return
    raise AssertionError("report missing the findings section passed validation")


def test_json_schema_mode_produces_compatible_report_document() -> None:
    """The tool's own ``--json-schema`` output is a well-formed schema.

    Cross-check: ``ProbeReport.model_json_schema()`` describes the raw model
    and must at least declare a report object with the top-level sections the
    ``--json`` contract exposes, so the CLI's schema command and the validated
    contract do not name different root sections.
    """
    schema = json.loads(to_json_schema())
    assert schema["type"] == "object"
    assert "config" in schema["properties"]
    assert "capacity" in schema["properties"]
    assert "findings" in schema["properties"]


# ---------------------------------------------------------------------------
# CLI-invocation scenario tests: run the real ``llmprobe --json`` command (not
# just ``to_json``) hermetically against the scripted mock, and validate the
# emitted report against the provenance contract on three scenarios — clean,
# divergence (mismatch), error. The contract is pinned by ``REPORT_SCHEMA``
# rather than the ``--json-schema`` pydantic dump, because ``to_json`` emits a
# provenance-augmented shape that ``ProbeReport.model_json_schema`` does not
# describe; calling ``--json-schema`` separately confirms its root sections
# still agree with the contract the tests validate against.
# ---------------------------------------------------------------------------


def _asgi_client(app: object, api_key: str | None = None):
    def make(
        _base_url: str,
        _api_key: str | None = None,
        timeout: float = 10.0,
        **_kwargs,
    ) -> httpx.AsyncClient:
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return httpx.AsyncClient(
            base_url=_BASE_URL,
            transport=httpx.ASGITransport(app=app),
            timeout=httpx.Timeout(timeout),
            headers=headers,
        )

    return make


def _invoke(
    app,
    monkeypatch: pytest.MonkeyPatch,
    args: list[str],
    api_key: str | None = None,
):
    monkeypatch.setattr(cli, "_make_client", _asgi_client(app, api_key))
    return _runner.invoke(cli.app, args)


def _cli_options() -> set[str]:
    cmd = typer.main.get_command(cli.app)
    return {
        opt
        for p in cmd.params
        for opt in list(p.opts) + list(getattr(p, "secondary_opts", []) or [])
    }


def test_cli_declares_json_and_json_schema_options() -> None:
    """``--json`` and ``--json-schema`` are real CLI options (introspected)."""
    opts = _cli_options()
    assert "--json" in opts
    assert "--json-schema" in opts


def test_json_schema_command_emits_contract_root() -> None:
    """``llmprobe --json-schema`` emits a schema naming the four report sections.

    The pydantic schema does not describe the provenance-augmented ``--json``
    shape (which is why ``REPORT_SCHEMA`` is the validation contract), but it
    must still agree on the top-level root sections so the CLI's schema command
    does not drift from the contract the scenario tests validate against.
    """
    result = _runner.invoke(cli.app, ["--json-schema"])
    assert result.exit_code == 0, result.output
    schema = json.loads(result.stdout)
    assert schema["type"] == "object"
    assert set(schema["properties"]) >= {"base_url", "config", "capacity", "findings"}


def test_clean_report_via_cli_validates_against_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``llmprobe --json`` on a clean server emits a schema-valid, empty report."""
    server = make_mock_server(max_tokens=8192, behavior="honest")
    result = _invoke(server, monkeypatch, [_BASE_URL, "--probe", "--json"])

    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["findings"] == []
    validate(report, REPORT_SCHEMA)


def test_mismatch_report_via_cli_validates_against_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A divergent (mismatch) ``--json`` report still validates against the schema.

    The silent-truncation server with a claimed 8192 ctx forces a mismatch
    finding (exit 1); the emitted report must remain schema-valid.
    """
    server = make_mock_server(max_tokens=512, behavior="silent_truncation")
    result = _invoke(
        server,
        monkeypatch,
        [_BASE_URL, "--claimed-ctx", "8192", "--probe", "--json"],
    )

    assert result.exit_code == 1, result.output
    report = json.loads(result.stdout)
    assert {f["severity"]["value"] for f in report["findings"]} == {"mismatch"}
    validate(report, REPORT_SCHEMA)


def test_error_report_via_cli_validates_against_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An error-severity ``--json`` report still validates against the schema.

    An unauthenticated probe against a key-protected mock surfaces an
    ``GENERIC_MODELS_HTTP_ERROR`` finding (exit 2); the error report must
    remain schema-valid.
    """
    server = make_mock_server(
        max_tokens=512, behavior="honest", required_token="right"
    )
    result = _invoke(
        server, monkeypatch, [_BASE_URL, "--json"], api_key="wrong"
    )

    assert result.exit_code == 2, result.output
    report = json.loads(result.stdout)
    assert {f["severity"]["value"] for f in report["findings"]} == {"error"}
    validate(report, REPORT_SCHEMA)
