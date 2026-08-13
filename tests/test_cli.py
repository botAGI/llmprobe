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
import threading
import time

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


# The real (network-bound) client factory. ``monkeypatch`` can leak a prior
# test's ASGI client into ``cli._make_client`` under asyncio auto-mode, so a
# test that must exercise the real timeout path pins this factory explicitly
# instead of trusting ``cli._make_client`` at call time.
_REAL_CLIENT_FACTORY = cli._make_client


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
    return {
        opt
        for p in cmd.params
        for opt in list(p.opts) + list(getattr(p, "secondary_opts", []) or [])
    }


def test_cli_options_are_declared() -> None:
    opts = _all_cli_options()
    assert "--claimed-ctx" in opts
    assert "--probe" in opts
    assert "--safe" in opts
    assert "--json" in opts
    assert "--endpoint" in opts
    assert "--api-key" in opts
    assert "--timeout" in opts
    assert "--json-schema" in opts
    assert "--max-requests" in opts
    assert "--verbose" in opts


def test_json_schema_prints_valid_json_and_exits_zero() -> None:
    """``--json-schema`` prints the report's JSON schema and exits without probing.

    No base URL is required: the schema is derived purely from the pydantic
    model and printed to stdout as valid JSON with exit code 0, matching the
    acceptance check ``llmprobe --json-schema``.
    """
    result = runner.invoke(cli.app, ["--json-schema"])
    assert result.exit_code == 0, result.output

    schema = json.loads(result.stdout)
    assert schema["title"] == "ProbeReport"
    assert schema["type"] == "object"
    assert "config" in schema["properties"]
    assert "findings" in schema["properties"]


def test_clean_result_exits_0(monkeypatch: pytest.MonkeyPatch) -> None:
    """A clean probe with no mismatch or error findings => exit code 0.

    An honest server probed with no ``--claimed-ctx`` produces no findings, so
    the derived process exit code must be 0. This is the happy-path contract:
    llmprobe only fails the process when it actually finds a mismatch or error.
    """
    server = make_mock_server(max_tokens=8192, behavior="honest")
    result = _invoke(server, monkeypatch, [BASE_URL, "--probe", "--json"])
    assert result.exit_code == 0, result.output

    report = json.loads(result.stdout)
    assert report["findings"] == []


def test_mismatch_result_exits_1(monkeypatch: pytest.MonkeyPatch) -> None:
    """A MISMATCH finding (claimed ctx above the measured cliff) => exit code 1.

    A 512-token silent-truncation server with a claimed 8192-ctx contract
    provably cannot honour that context past 512 tokens. The measured cliff
    below the claim is surfaced as a MISMATCH finding, and the derived exit
    code must be 1 — signalling "the server is up but disagrees with you".
    """
    server = make_mock_server(max_tokens=512, behavior="silent_truncation")
    result = _invoke(
        server, monkeypatch, [BASE_URL, "--claimed-ctx", "8192", "--probe", "--json"]
    )
    assert result.exit_code == 1, result.output

    report = json.loads(result.stdout)
    severities = {f["severity"]["value"] for f in report["findings"]}
    assert "mismatch" in severities


def test_unavailable_server_exits_2() -> None:
    """An unreachable/failed server => exit code 2 with no report.

    When the origin cannot be reached at all, llmprobe treats it as an
    unrecoverable error: it prints a diagnostic to stderr and exits 2, never
    emitting a (possibly empty, misleading) report.
    """
    result = runner.invoke(cli.app, ["http://127.0.0.1:1", "--json"])
    assert result.exit_code == 2
    assert result.stdout == ""


def test_safe_is_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--safe`` is the default: no inference load is sent unless ``--probe``.

    The ``--safe/--probe`` boolean option defaults to ``True`` (safe), and
    with the default flags the report carries an empty ``capacity`` list,
    proving no probe requests were issued.
    """
    cmd = typer.main.get_command(cli.app)
    safe_param = next(p for p in cmd.params if "--safe" in p.opts)
    assert safe_param.default is True

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
    severities = {f["severity"]["value"] for f in report["findings"]}
    assert "mismatch" in severities


def test_probe_reports_the_measured_cliff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Against the silent-truncation mock the probe surfaces the true cliff."""
    server = make_mock_server(max_tokens=512, behavior="silent_truncation")
    result = _invoke(server, monkeypatch, [BASE_URL, "--probe", "--json"])
    assert result.exit_code == 0

    report = json.loads(result.stdout)
    assert report["capacity"] != []
    cap = report["capacity"][0]
    assert cap["max_accepted_tokens"]["value"] == 512
    assert cap["cliff_behavior"]["value"] == CliffBehavior.SILENT_TRUNCATION.value


def test_honest_mock_probes_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    """An honest server with no claimed ctx and probing => exit 0, no findings."""
    server = make_mock_server(max_tokens=8192, behavior="honest")
    result = _invoke(server, monkeypatch, [BASE_URL, "--probe", "--json"])
    assert result.exit_code == 0

    report = json.loads(result.stdout)
    assert report["findings"] == []


def test_verbose_traces_each_probe_to_stderr_without_touching_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--verbose`` logs each probe to stderr; stdout stays machine-readable JSON.

    The traces must carry the probed input length, the classification verdict,
    and an elapsed response time, and must go ONLY to stderr so the ``--json``
    report on stdout remains valid JSON (no trailing probe lines mixed in).
    """
    server = make_mock_server(max_tokens=512, behavior="silent_truncation")
    result = _invoke(
        server,
        monkeypatch,
        [BASE_URL, "--probe", "--verbose", "--json"],
    )
    assert result.exit_code == 0

    # stdout must still be pure JSON — verbose output never pollutes the report.
    report = json.loads(result.stdout)
    assert report["capacity"] != []
    assert report["capacity"][0]["cliff_behavior"]["value"] == (
        CliffBehavior.SILENT_TRUNCATION.value
    )

    # stderr must carry per-probe trace lines with length, verdict, and elapsed.
    assert result.stderr, "verbose mode produced no probe trace output on stderr"
    assert "probe:" in result.stderr
    assert "length=" in result.stderr
    assert "verdict=" in result.stderr
    assert "elapsed=" in result.stderr


def test_default_emits_no_verbose_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without ``--verbose`` no probe trace lines are written to stderr."""
    server = make_mock_server(max_tokens=512, behavior="silent_truncation")
    result = _invoke(server, monkeypatch, [BASE_URL, "--probe", "--json"])
    assert result.exit_code == 0
    assert "probe:" not in result.stderr


def test_probe_ceiling_uses_claimed_context_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The probe searches up to the CLI's own ceiling (not the advertised ctx)."""
    server = make_mock_server(max_tokens=512, behavior="honest")
    result = _invoke(server, monkeypatch, [BASE_URL, "--probe", "--json"])
    assert result.exit_code == 0

    report = json.loads(result.stdout)
    cap = report["capacity"][0]
    assert cap["max_accepted_tokens"]["value"] == DEFAULT_CEILING
    assert cap["cliff_behavior"]["value"] == CliffBehavior.ACCEPTED.value


def test_claimed_ctx_matching_measured_cliff_exits_0(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``--claimed-ctx`` equal to the measured cliff => exit 0 (no mismatch).

    The mismatch check only fires when the measured ``max_accepted_tokens`` is
    strictly below the claimed context. When the claim exactly matches what the
    server truncates at, nothing is a mismatch and the process must succeed.
    """
    server = make_mock_server(max_tokens=512, behavior="silent_truncation")
    result = _invoke(
        server, monkeypatch, [BASE_URL, "--claimed-ctx", "512", "--probe", "--json"]
    )
    assert result.exit_code == 0

    report = json.loads(result.stdout)
    cap = report["capacity"][0]
    assert cap["max_accepted_tokens"]["value"] == 512
    assert report["findings"] == []


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
    errors = [
        f for f in report["findings"] if f["severity"]["value"] == "error"
    ]
    assert any(
        f["code"]["value"] == "GENERIC_MODELS_HTTP_ERROR" for f in errors
    )


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


def test_endpoint_chat_resolves_to_chat_completions() -> None:
    """``--endpoint chat`` resolves ``_resolve_path`` to the chat completions path.

    The README promises ``--endpoint chat`` exercises the chat endpoint. The
    explicit ``CHAT`` choice, like ``EMBEDDINGS``, must be honoured directly
    across every backend and never overridden by the backend default.
    """
    for backend in Backend:
        assert cli._resolve_path(cli.Endpoint.CHAT, backend) == (
            "/v1/chat/completions"
        )


def test_explicit_endpoint_sends_probe_traffic_without_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Selecting ``--endpoint chat`` enables inference even without ``--probe``.

    Naming an explicit endpoint (CHAT/EMBEDDINGS) is an instruction to probe
    that endpoint, so probe traffic is sent despite the ``--safe`` default.
    Only the default ``auto`` selection respects ``--safe`` suppression (see
    ``test_safe_is_the_default``).
    """
    server = make_mock_server(max_tokens=512, behavior="honest")
    result = _invoke(
        server, monkeypatch, [BASE_URL, "--endpoint", "chat", "--json"]
    )
    assert result.exit_code == 0, result.output

    report = json.loads(result.stdout)
    assert report["capacity"] != []
    assert report["capacity"][0]["endpoint"]["value"] == "/v1/chat/completions"


def test_endpoint_chat_drives_the_chat_capacity_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--endpoint chat`` must genuinely exercise the chat completions cliff.

    The README promises ``--endpoint chat`` probes the chat endpoint. That is
    only true if the selected endpoint flows all the way into
    ``probe_capacity`` — the reported capacity must carry the
    ``/v1/chat/completions`` path, not an embeddings path. This is an
    end-to-end wiring check (not just ``_resolve_path`` in isolation).
    """
    server = make_mock_server(max_tokens=512, behavior="honest")
    result = _invoke(
        server, monkeypatch, [BASE_URL, "--endpoint", "chat", "--probe", "--json"]
    )
    assert result.exit_code == 0, result.output

    report = json.loads(result.stdout)
    assert report["capacity"] != []
    assert report["capacity"][0]["endpoint"]["value"] == "/v1/chat/completions"


def test_auto_endpoints_are_distinct_per_backend() -> None:
    """Per-backend auto defaults must not all be identical.

    If every backend collapsed to the same endpoint the ``auto`` resolution
    would be meaningless — the whole point is a mapping that differs by the
    detected backend.
    """
    paths = {backend: cli._resolve_path(cli.Endpoint.AUTO, backend) for backend in Backend}
    assert len(set(paths.values())) > 1


def test_short_timeout_fails_fast_against_slow_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A short ``--timeout`` against a slow server fails fast instead of hanging.

    Uses the REAL client factory (not the ASGI seam) against a local socket
    server that sleeps before replying, so the httpx read timeout is genuinely
    exercised rather than short-circuited by an in-process transport.
    """
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class _Slow(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            time.sleep(2.0)
            body = b'{"object":"list","data":[{"id":"mock"}]}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # silence per-request logging
            pass

    monkeypatch.setattr(cli, "_make_client", _REAL_CLIENT_FACTORY)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Slow)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}"
        start = time.monotonic()
        result = runner.invoke(cli.app, [url, "--timeout", "0.1"])
        elapsed = time.monotonic() - start
    finally:
        server.shutdown()
        thread.join()

    # The first config read exceeds the 0.1s timeout and fails fast.
    assert result.exit_code == 2
    assert elapsed < 2.0, f"expected fail-fast, took {elapsed:.2f}s"
