"""Tests for the three CLI process exit codes, driven on mocks.

Hermetic: no network, no real inference server. The CLI contract is:

* ``0`` — everything matched: no mismatch and no error findings.
* ``1`` — a MISMATCH finding: claimed context is above the measured cliff.
* ``2`` — an ERROR finding or an unreachable/failed server.

The three cases are exercised through ``typer.testing.CliRunner`` against the
scripted mock from ``tests/mocks/server.py`` served via an in-process
``httpx.ASGITransport`` seam (case 2 needs an affirmative MISMATCH finding); the
unreachable-server case uses the real HTTP client against a dead local port.
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
    """Client factory seam: route every request to the in-process mock app."""

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


def test_all_cli_options_are_declared() -> None:
    """The exit-code-relevant options must be declared on the CLI.

    Introspected from click params (never by substring-matching --help output,
    which wraps by terminal width and breaks on CI) and reproduced via the
    process-exit contract documented in the module docstring.
    """
    cmd = typer.main.get_command(cli.app)
    opts = {
        opt
        for p in cmd.params
        for opt in list(p.opts) + list(getattr(p, "secondary_opts", []) or [])
    }
    assert "--json" in opts
    assert "--probe" in opts
    assert "--claimed-ctx" in opts


def test_exit_code_0_on_match(monkeypatch: pytest.MonkeyPatch) -> None:
    """Claimed context equal to the measured cliff => exit code 0.

    A 512-token silent-truncation server claimed exactly at its 512-token
    limit produces NO mismatch and NO error, so llmprobe must succeed with
    exit code 0 and an empty findings list.
    """
    server = make_mock_server(max_tokens=512, behavior="silent_truncation")
    result = _invoke(
        server,
        monkeypatch,
        [BASE_URL, "--claimed-ctx", "512", "--probe", "--json"],
    )
    assert result.exit_code == 0, result.output

    report = json.loads(result.stdout)
    assert report["findings"] == []


def test_exit_code_1_on_claimed_vs_measured_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Claimed context above the measured cliff => exit code 1.

    A 512-token silent-truncation server claimed at 8192 cannot honour that
    context past 512 tokens. The measured cliff below the claim is a MISMATCH,
    so the process must exit 1 — "the server is up but disagrees with you".
    """
    server = make_mock_server(max_tokens=512, behavior="silent_truncation")
    result = _invoke(
        server,
        monkeypatch,
        [BASE_URL, "--claimed-ctx", "8192", "--probe", "--json"],
    )
    assert result.exit_code == 1, result.output

    report = json.loads(result.stdout)
    severities = {f["severity"]["value"] for f in report["findings"]}
    assert "mismatch" in severities


def test_exit_code_2_on_unreachable_server() -> None:
    """An unreachable server => exit code 2 with no report.

    When the origin cannot be reached at all, llmprobe treats it as an
    unrecoverable error: it prints a diagnostic to stderr and exits 2, never
    emitting a (possibly empty, misleading) report on stdout.
    """
    result = runner.invoke(cli.app, ["http://127.0.0.1:1", "--json"])
    assert result.exit_code == 2
    assert result.stdout == ""
