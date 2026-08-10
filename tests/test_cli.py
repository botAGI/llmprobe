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
from llmprobe.backends import DEFAULT_PROBE_ENDPOINTS
from llmprobe.models import Backend, CliffBehavior
from llmprobe.probes.capacity import DEFAULT_CEILING

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
    app,
    monkeypatch: pytest.MonkeyPatch,
    args: list[str],
    api_key: str | None = None,
):
    monkeypatch.setattr(cli, "_make_client", _asgi_client(app, api_key))
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
    assert "--api-key" in opts
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


def test_api_key_is_sent_and_never_leaks_into_card(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Bearer api-key authenticates the probe and stays out of the card.

    Reproduces + fixes the security question: the endpoint's api-key must not
    arrive on stdout. The mock demands ``Bearer sup3rs3cr3t``; with the key
    the probe succeeds, and the key string never appears in the JSON output.
    """
    secret = "sup3rs3cr3t"
    server = make_mock_server(
        max_tokens=512, behavior="honest", required_token=secret
    )
    result = _invoke(
        server, monkeypatch, [BASE_URL, "--probe", "--json"], api_key=secret
    )
    assert result.exit_code == 0, result.output
    assert secret not in result.output

    report = json.loads(result.stdout)
    assert report["findings"] == []


def test_api_key_wrong_rejects_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """A wrong/missing key must reject loudly, not silently pass."""
    server = make_mock_server(
        max_tokens=512, behavior="honest", required_token="right"
    )
    result = _invoke(
        server, monkeypatch, [BASE_URL, "--json"], api_key="wrong"
    )
    assert result.exit_code == 2
    report = json.loads(result.stdout)
    errors = [f for f in report["findings"] if f["severity"] == "error"]
    assert any(f["code"] == "GENERIC_MODELS_HTTP_ERROR" for f in errors)


def test_url_embedded_api_key_is_redacted_from_card() -> None:
    """A key in the base URL must not leak into the card header."""
    server = make_mock_server(max_tokens=512, behavior="honest")
    url = "http://sup3rs3cr3t@mock"
    result = _invoke(server, monkeypatch := pytest.MonkeyPatch(), [url])
    assert result.exit_code == 0
    assert "sup3rs3cr3t" not in result.output
    assert "# Capability Report — http://mock" in result.output


def test_timeout_is_threaded_into_every_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--timeout`` is forwarded to the client factory bound to each request."""
    captured: dict[str, float] = {}

    def capturing_client(
        _base_url: str,
        _api_key: str | None = None,
        timeout: float = cli.DEFAULT_TIMEOUT,
        **_kwargs,
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


def test_auto_endpoint_resolves_per_backend() -> None:
    """``--endpoint auto`` resolves to each backend's default probe path.

    The README promises that the default ``auto`` selection resolves per
    backend. ``auto`` must not silently collapse to embeddings for every
    backend: an explicit backend-aware default must be used instead, and an
    explicit ``chat``/``embeddings`` choice is never overridden.
    """
    for backend in Backend:
        resolved = cli._resolve_path(cli.Endpoint.AUTO, backend)
        assert resolved == DEFAULT_PROBE_ENDPOINTS[backend], (
            f"auto did not resolve to {backend}'s default probe path"
        )

    assert cli._resolve_path(cli.Endpoint.CHAT, Backend.LLAMACPP) == (
        "/v1/chat/completions"
    )
    assert cli._resolve_path(cli.Endpoint.EMBEDDINGS, Backend.VLLM) == (
        "/v1/embeddings"
    )


def test_auto_endpoints_are_distinct_per_backend() -> None:
    """Per-backend auto defaults must not all be identical.

    If every backend collapsed to the same endpoint the ``auto`` resolution
    would be meaningless — the whole point is a mapping that differs by the
    detected backend.
    """
    paths = {backend: cli._resolve_path(cli.Endpoint.AUTO, backend) for backend in Backend}
    assert len(set(paths.values())) > 1
