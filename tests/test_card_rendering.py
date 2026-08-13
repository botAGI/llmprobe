"""What reaches the operator's terminal must be what the report says.

``to_markdown`` is well covered, but the card is printed through a rich
``Console``, and rich reads ``[...]`` as its own markup. The severity label —
the single most important word in a finding — was being consumed on the way
out: the report said ``**[mismatch] UBATCH_CEILING**`` and the terminal showed
``** UBATCH_CEILING**``. Every markdown test passed the whole time, because
none of them looked at what the CLI actually printed.
"""

from __future__ import annotations

import httpx
import pytest
from typer.testing import CliRunner

import llmprobe.cli as cli

from tests.mocks.server import make_mock_server

BASE_URL = "http://mock"

runner = CliRunner()


def _asgi_client(app: object):
    def make(
        _base_url: str,
        _api_key: str | None = None,
        timeout: float = 10.0,
        **_kwargs,
    ) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url=BASE_URL,
            timeout=timeout,
        )

    return make


def test_severity_label_survives_to_the_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``[mismatch]`` must appear on stdout, not be eaten as console markup."""
    server = make_mock_server(max_tokens=512, behavior="silent_truncation")
    monkeypatch.setattr(cli, "_make_client", _asgi_client(server))

    result = runner.invoke(
        cli.app, [BASE_URL, "--probe", "--claimed-ctx", "8192", "--endpoint", "chat"]
    )

    assert "## Findings" in result.stdout, result.stdout
    assert "[mismatch]" in result.stdout, (
        f"severity label lost between report and terminal: {result.stdout}"
    )


def test_card_lines_are_not_reflowed_by_the_terminal_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Table rows must stay on one line so the card can be piped or pasted.

    A report wrapped at the terminal width stops being a markdown table: rows
    break mid-cell and the result no longer renders anywhere it is pasted.
    """
    server = make_mock_server(max_tokens=512, behavior="silent_truncation")
    monkeypatch.setattr(cli, "_make_client", _asgi_client(server))

    result = runner.invoke(
        cli.app, [BASE_URL, "--probe", "--claimed-ctx", "8192", "--endpoint", "chat"]
    )

    for line in result.stdout.splitlines():
        if line.startswith("| "):
            assert line.rstrip().endswith("|"), f"table row was wrapped: {line!r}"
