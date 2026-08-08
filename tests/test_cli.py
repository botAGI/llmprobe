"""Tests for ``llmprobe.cli`` — the typer entrypoint.

Hermetic: no network, no real inference server. The CLI is driven through
``typer.testing.CliRunner``; where a live server is needed the scripted mock
from ``tests/mocks/server.py`` is served through ``httpx.ASGITransport`` by
replacing the client factory. CLI options are asserted by introspecting click
params, never by substring-matching ``--help`` output (which wraps by terminal
width and breaks on CI).
"""

from __future__ import annotations

import json

import httpx
import pytest
import typer
from typer.testing import CliRunner

import llmprobe.cli as cli
from llmprobe.models import CliffBehavior
from llmprobe.probes.capacity import DEFAULT_CEILING

from tests.mocks.server import make_mock_server

BASE_URL = "http://mock"

runner = CliRunner()


def _asgi_client(app: object):
    def make(_base_url: str, timeout: float = 10.0) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=BASE_URL,
            transport=httpx.ASGITransport(app=app),
            timeout=httpx.Timeout(timeout),
        )

    return make


def _invoke(app, monkeypatch: pytest.MonkeyPatch, args: list[str]):
    monkeypatch.setattr(cli, "_make_client", _asgi_client(app))
    return runner.invoke(cli.app, args)


def _all_cli_options() -> set[str]:
    cmd = typer.main.get_command(cli.app)
    return {opt for p in cmd.params for opt in p.opts}


def test_cli_options_are_declared() -> None:
    opts = _all_cli_options()
    assert "--claimed-ctx" in opts
    assert "--probe" in opts
    assert "--json" in opts
    assert "--endpoint" in opts
    assert "--timeout" in opts


def test_safe_is_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--safe`` is the default: no inference load is sent unless ``--probe``.

    The ``--probe/--safe`` boolean option defaults to ``False`` (safe), and
    with the default flags the report carries an empty ``capacity`` list,
    proving no probe requests were issued.
    """
    cmd = typer.main.get_command(cli.app)
    probe_param = next(p for p in cmd.params if "--probe" in p.opts)
    assert probe_param.default is False

    server = make_mock_server(max_tokens=512, behavior="honest")
    result = _invoke(server, monkeypatch, [BASE_URL, "--json"])
    assert result.exit_code == 0

    report = json.loads(result.stdout)
    assert report["capacity"] == []


def test_silent_truncation_mock_exits_1_when_claimed_ctx_mismatches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 512-token silent-truncation server with a claimed 8192 ctx => exit 1."""
    server = make_mock_server(max_tokens=512, behavior="silent_truncation")
    result = _invoke(
        server, monkeypatch, [BASE_URL, "--claimed-ctx", "8192", "--probe", "--json"]
    )
    assert result.exit_code == 1

    report = json.loads(result.stdout)
    severities = {f["severity"] for f in report["findings"]}
    assert "mismatch" in severities


def test_probe_reports_the_measured_cliff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Against the silent-truncation mock the probe surfaces the true cliff."""
    server = make_mock_server(max_tokens=512, behavior="silent_truncation")
    result = _invoke(server, monkeypatch, [BASE_URL, "--probe", "--json"])
    assert result.exit_code == 0

    report = json.loads(result.stdout)
    assert report["capacity"] != []
    cap = report["capacity"][0]
    assert cap["max_accepted_tokens"] == 512
    assert cap["cliff_behavior"] == CliffBehavior.SILENT_TRUNCATION.value


def test_honest_mock_probes_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    """An honest server with no claimed ctx and probing => exit 0, no findings."""
    server = make_mock_server(max_tokens=8192, behavior="honest")
    result = _invoke(server, monkeypatch, [BASE_URL, "--probe", "--json"])
    assert result.exit_code == 0

    report = json.loads(result.stdout)
    assert report["findings"] == []


def test_probe_ceiling_uses_claimed_context_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The probe searches up to the CLI's own ceiling (not the advertised ctx)."""
    server = make_mock_server(max_tokens=512, behavior="honest")
    result = _invoke(server, monkeypatch, [BASE_URL, "--probe", "--json"])
    assert result.exit_code == 0

    report = json.loads(result.stdout)
    cap = report["capacity"][0]
    assert cap["max_accepted_tokens"] == DEFAULT_CEILING
    assert cap["cliff_behavior"] == CliffBehavior.ACCEPTED.value


def test_unreachable_server_exits_2() -> None:
    """A server that cannot be reached => exit 2 with no report."""
    result = runner.invoke(cli.app, ["http://127.0.0.1:1", "--json"])
    assert result.exit_code == 2


def test_timeout_is_threaded_into_every_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--timeout`` is forwarded to the client factory bound to each request."""
    captured: dict[str, float] = {}

    def capturing_client(
        _base_url: str, timeout: float = cli.DEFAULT_TIMEOUT
    ) -> httpx.AsyncClient:
        captured["timeout"] = timeout
        return httpx.AsyncClient(
            base_url=BASE_URL,
            transport=httpx.ASGITransport(
                app=make_mock_server(max_tokens=512, behavior="honest")
            ),
            timeout=httpx.Timeout(timeout),
        )

    monkeypatch.setattr(cli, "_make_client", capturing_client)
    result = runner.invoke(cli.app, [BASE_URL, "--timeout", "3.5", "--json"])
    assert result.exit_code == 0
    assert captured["timeout"] == 3.5
