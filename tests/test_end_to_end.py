"""End-to-end test — one test that proves the product.

A server that claims an 8192-token context but silently truncates incoming
prompts at 512 tokens must be caught and reported with a remedy. This single
test drives the real CLI end to end (via ``typer.testing.CliRunner``) against
the scripted ``silent_truncation`` mock and asserts that the rendered
capability card exposes the mismatch, the measured cliff, and a Fix.

Hermetic: no network, no real inference server. The mock is served through
``httpx.ASGITransport`` by replacing the client factory, exactly as every
other hermetic test does.
"""

from __future__ import annotations

import httpx
import pytest
from typer.testing import CliRunner

import llmprobe.cli as cli

from tests.mocks.server import make_mock_server

BASE_URL = "http://mock"
MAX_TOKENS = 512
CLAIMED_CTX = 8192

_PROVENANCE_MARKERS = ("read", "measured", "inferred", "unknown")

runner = CliRunner()


def _asgi_client(app: object):
    def make(_base_url: str) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=BASE_URL,
            transport=httpx.ASGITransport(app=app),
            timeout=httpx.Timeout(10.0),
        )

    return make


def _invoke(app, monkeypatch: pytest.MonkeyPatch, args: list[str]):
    monkeypatch.setattr(cli, "_make_client", _asgi_client(app))
    return runner.invoke(cli.app, args)


def _table_data_rows(stdout: str) -> list[str]:
    """Return only the table *data* rows of the rendered markdown card.

    Skips the heading, the header row, and the separator row; every remaining
    ``| ... |`` line is a data row that must carry a provenance marker.
    """
    rows: list[str] = []
    in_table = False
    for line in stdout.splitlines():
        if not line.startswith("| "):
            in_table = False
            continue
        if not in_table:
            in_table = True
            continue
        if set(line.replace("|", "").replace("-", "").strip()) == set():
            continue
        rows.append(line)
    return rows


def test_end_to_end_silent_truncation_is_caught_and_has_a_fix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The full product catches a lying server and points at the remedy."""
    server = make_mock_server(max_tokens=MAX_TOKENS, behavior="silent_truncation")

    result = _invoke(
        server,
        monkeypatch,
        [BASE_URL, "--claimed-ctx", str(CLAIMED_CTX), "--probe"],
    )

    stdout = result.stdout

    # A 512-token silent truncation vs a claimed 8192 ctx is a mismatch.
    assert result.exit_code == 1

    # The measured cliff is surfaced.
    assert str(MAX_TOKENS) in stdout

    # The card names the silent truncation behaviour explicitly.
    assert "silent_truncation" in stdout or "truncated" in stdout

    # A Fix section with the exact llama.cpp remedy flags is present.
    assert "--batch-size" in stdout
    assert "--ubatch-size" in stdout

    # Every table data row in the rendered card carries a provenance marker.
    assert "## Findings" in stdout
    data_rows = _table_data_rows(stdout)
    assert data_rows, "expected at least one table data row in the card"
    for row in data_rows:
        assert any(marker in row for marker in _PROVENANCE_MARKERS), (
            f"table data row lacks a provenance marker: {row!r}"
        )
