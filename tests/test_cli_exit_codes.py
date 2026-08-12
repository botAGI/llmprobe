"""Tests for the CLI process exit-code contract.

The CLI derives its process exit code from the outcome of a probe:

* a clean result (no errors, no mismatch findings) exits 0;
* a MISMATCH finding (the server disagrees with ``--claimed-ctx``) exits 1;
* any error path (unreachable server, HTTP error, invalid endpoint) exits 2.

Hermetic: no network, no real inference server. The CLI is driven through
``typer.testing.CliRunner``; where a live server is needed the scripted mock
from ``tests/mocks/server.py`` is served through ``httpx.ASGITransport`` by
replacing the client factory. The ``-k exit_code`` selection targets only
exit-code behavior.
"""

from __future__ import annotations

import json

import httpx
import pytest
import typer
from typer.testing import CliRunner

import llmprobe.cli as cli

from tests.mocks.server import make_mock_server

BASE_URL = "http://mock"

runner = CliRunner()


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
            base_url=BASE_URL,
            transport=httpx.ASGITransport(app=app),
            timeout=httpx.Timeout(timeout),
            headers=headers,
        )

    return make


def _invoke(
    app: object,
    monkeypatch: pytest.MonkeyPatch,
    args: list[str],
):
    monkeypatch.setattr(cli, "_make_client", _asgi_client(app))
    return runner.invoke(cli.app, args)


def test_clean_result_exit_code_is_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """A clean probe with no error/mismatch findings exits 0."""
    server = make_mock_server(max_tokens=8192, behavior="honest")
    result = _invoke(server, monkeypatch, [BASE_URL, "--probe", "--json"])
    assert result.exit_code == 0, result.output

    report = json.loads(result.stdout)
    assert report["findings"] == []


def test_mismatch_result_exit_code_is_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """A MISMATCH finding (claimed ctx above the measured cliff) exits 1."""
    server = make_mock_server(max_tokens=512, behavior="silent_truncation")
    result = _invoke(
        server, monkeypatch, [BASE_URL, "--claimed-ctx", "8192", "--probe", "--json"]
    )
    assert result.exit_code == 1, result.output

    report = json.loads(result.stdout)
    severities = {f["severity"]["value"] for f in report["findings"]}
    assert "mismatch" in severities


_REAL_CLIENT_FACTORY = cli._make_client


def test_unreachable_server_exit_code_is_two(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unreachable server exits 2 with no report on stdout.

    The real (network-bound) client factory is pinned explicitly because a
    prior test's ASGI client can leak into ``cli._make_client`` under asyncio
    auto-mode, which would otherwise serve an in-process transport instead of
    genuinely failing to connect.
    """
    monkeypatch.setattr(cli, "_make_client", _REAL_CLIENT_FACTORY)
    result = runner.invoke(cli.app, ["http://127.0.0.1:1", "--json"])
    assert result.exit_code == 2
    assert result.stdout == ""


def test_http_error_exit_code_is_two(monkeypatch: pytest.MonkeyPatch) -> None:
    """An HTTP server error surfaces as an error finding => exit 2."""
    server = make_mock_server(max_tokens=512, behavior="honest", required_token="right")
    result = _invoke(server, monkeypatch, [BASE_URL, "--json"])
    assert result.exit_code == 2, result.output

    report = json.loads(result.stdout)
    errors = [f for f in report["findings"] if f["severity"]["value"] == "error"]
    assert any(f["code"]["value"] == "GENERIC_MODELS_HTTP_ERROR" for f in errors)


def test_invalid_endpoint_exit_code_is_two() -> None:
    """An invalid ``--endpoint`` selection is rejected with exit code 2.

    The ``--endpoint`` option is typed against the ``Endpoint`` enum, so typer
    rejects an out-of-range value while parsing the option, before any probe
    runs. The CLA is: a malformed endpoint is an error, never a silent default.
    """
    result = runner.invoke(cli.app, [BASE_URL, "--endpoint", "bogus", "--json"])
    assert result.exit_code == 2


def test_endpoint_parse_error_exit_code_is_two(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ValueError during endpoint coercion inside main exits 2.

    When ``endpoint`` reaches the probe path as a non-``Endpoint`` value (for
    example a caller invoking ``main`` directly, bypassing typer's option
    coercion), the resulting ``ValueError`` must be caught and turned into an
    explicit ``typer.Exit(code=2)`` rather than leaking a traceback.
    """
    with pytest.raises(typer.Exit) as excinfo:
        cli.main(BASE_URL, endpoint="bogus")
    assert excinfo.value.exit_code == 2
