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

import json
import subprocess
import sys
from pathlib import Path

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
    def make(
        _base_url: str,
        _api_key: str | None = None,
        timeout: float = 10.0,
        **_kwargs,
    ) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=BASE_URL,
            transport=httpx.ASGITransport(app=app),
            timeout=httpx.Timeout(timeout),
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


def test_end_to_end_embedding_truncation_boundary_is_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fake embeddings server that truncates at max_tokens is caught exactly.

    Against a locally raised fake embeddings server that genuinely discards
    input past ``max_tokens`` (the tail-collapse two-prompt method), the real
    CLI driven end to end must report a measured boundary of EXACTLY
    ``max_tokens`` -- never off by one, never an approximate estimate. A
    boundary one token away (``max_tokens-1`` or ``max_tokens+1``), a
    non-exact token count, or a misclassified cliff all mean the probe got the
    edge of the truncation wrong, and this test must turn red.
    """
    truncation_limit = 384
    server = make_mock_server(max_tokens=truncation_limit, behavior="silent_truncation")

    result = _invoke(
        server,
        monkeypatch,
        [BASE_URL, "--endpoint", "embeddings", "--probe", "--json"],
    )

    assert result.exit_code == 1, result.output

    payload = json.loads(result.output)

    # The probe must have exercised exactly the embeddings endpoint only.
    entries = payload["capacity"]
    assert [entry["endpoint"] for entry in entries] == ["/v1/embeddings"]

    # The measured boundary must be EXACTLY the truncation limit.
    entry = entries[0]
    assert entry["max_accepted_tokens"] == truncation_limit, (
        f"boundary off: expected {truncation_limit}, got "
        f"{entry['max_accepted_tokens']} (max_accepted_source="
        f"{entry['max_accepted_source']})"
    )
    assert entry["max_accepted_source"] == "measured", (
        "the boundary must be measured, not guessed"
    )
    assert entry["token_count_exact"] is True, (
        "the boundary token count must be verified exactly via /tokenize"
    )
    assert entry["cliff_behavior"] == "silent_truncation", (
        "the fake server silently drops the tail; the probe must say so"
    )

    # The lying server is surfaced as a finding with the exact measured edge.
    assert "## Findings" in result.stdout or payload["findings"]


def test_end_to_end_honest_server_verdict_is_accepted_without_false_alarms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A server that does not truncate is reported as accepted, no false alarms.

    The fake server is started without truncation (honest behavior) and the
    real CLI is driven against it. The probe verdict must be ``accepted`` and
    the report must carry no findings: an honest server that accepts every
    length through and beyond the probe ceiling must never raise a
    silent-truncation or mismatch false alarm, and the process must exit 0.
    """
    server = make_mock_server(max_tokens=CLAIMED_CTX, behavior="honest")

    result = _invoke(server, monkeypatch, [BASE_URL, "--probe", "--json"])

    assert result.exit_code == 0, result.output

    payload = json.loads(result.output)

    # No false alarms: an honest server produces no findings at all.
    assert payload["findings"] == []

    # The probe verdict for every exercised endpoint is ``accepted`` — the
    # server never truncated, never errored, and accepted every length it was
    # asked about.
    assert payload["capacity"], "expected the capacity probe to report"
    for entry in payload["capacity"]:
        verdict = entry["cliff_behavior"]["value"]
        assert verdict == "accepted", (
            f"honest server was flagged {verdict!r}, expected 'accepted' "
            "(false alarm)"
        )


_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run(args: list[str], timeout: int = 300) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def test_pip_install_then_help_is_zero(tmp_path: Path) -> None:
    """The wheel builds, installs into a clean venv, and the CLI answers help."""
    venv_dir = tmp_path / "venv"
    _run([sys.executable, "-m", "venv", "--system-site-packages", str(venv_dir)]).check_returncode()

    venv_python = venv_dir / "bin" / "python"
    venv_llmprobe = venv_dir / "bin" / "llmprobe"

    install = _run(
        [str(venv_python), "-m", "pip", "install", "--no-deps", str(_PROJECT_ROOT)]
    )
    assert install.returncode == 0, install.stdout + install.stderr
    assert venv_llmprobe.exists(), "llmprobe console script was not installed"

    help_run = _run([str(venv_llmprobe), "--help"])
    assert help_run.returncode == 0, help_run.stdout + help_run.stderr
    assert "Usage:" in help_run.stdout
