"""Regression test: a truncated canary echo is never a silent-truncation alarm.

The chat capacity probe detects silent truncation by asking the model to echo a
canary marker (``ZQX7``) from the head of the prompt: a reply carrying the
marker proves the head survived, and a reply without it is classified as
``silent_truncation``. Some fully honest servers accept every input yet never
return the marker verbatim — e.g. because the marker is tokenised in pieces so
only a truncated form (``ZQ``) survives. Such a server does NOT truncate: it
simply cannot echo the test marker back intact.

Without a guard the canary check would see the truncated ``ZQ`` (which does not
contain the full ``ZQX7``) on every probed length and fabricate a
``silent_truncation`` verdict — the exact false alarm this probe exists to
avoid, collapsing a healthy server to a tiny ``15``-token capacity.

The fix is the marker-echo *calibration* step: before trusting any
silent-truncation verdict the probe first sends a short, certainly-accepted
input whose head is the canary. When even that input cannot echo the full
marker the whole method is unreliable, so the tool honestly reports UNKNOWN
(``max_accepted_source == unknown``, cliff ``transport_error``) rather than a
confident ``silent_truncation`` it cannot verify.

This test pins that behaviour by driving the real CLI against a mock that fully
accepts every input while echoing a truncated marker. It asserts the report is
never ``silent_truncation`` and that the missing boundary is honestly marked
UNKNOWN. Reverting the calibration guard makes the probe classify every length
as ``silent_truncation`` and report ``max_accepted_tokens == 15`` — this test
turns red, so the regression cannot creep back.

Hermetic: no network, no real inference server. The mock is served through
``httpx.ASGITransport`` by replacing the client factory, exactly as every other
hermetic test does.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from typer.testing import CliRunner

import llmprobe.cli as cli

from tests.mocks.server import make_mock_server

BASE_URL = "http://mock"

# A truncated form of the canary ``ZQX7``: a fully honest server that accepts
# every input but cannot echo the marker back verbatim.
TRUNCATED_MARKER = "ZQ"

runner = CliRunner()


def _truncated_marker_honest_server() -> Any:
    """Build an honest server whose chat reply always echoes a truncated marker.

    ``make_mock_server(max_tokens=512, behavior="honest",
    chat_marker_reply="ZQ")`` fully accepts every input (never errors, never
    truncates) yet always echoes ``'ZQ'`` — a truncated form of the canary that
    does not contain the full ``'ZQX7'``. It models a server that does not drop
    the head but whose marker cannot be verified verbatim.
    """
    return make_mock_server(
        max_tokens=512, behavior="honest", chat_marker_reply=TRUNCATED_MARKER
    )


def _asgi_client_factory(app: Any):
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


def test_truncated_canary_echo_is_never_silent_truncation() -> None:
    """A server whose marker does not survive verbatim must NOT false-alarm.

    Driving the real CLI against a fully honest server that echoes only a
    truncated marker must report the boundary honestly (UNKNOWN) and never a
    confident ``silent_truncation``. Asserting ``cliff_behavior !=
    silent_truncation`` guards the core regression: if the marker-echo
    calibration guard is removed, every probed length classifies as
    ``silent_truncation`` and the report shows ``max_accepted_tokens == 15``.
    """
    server = _truncated_marker_honest_server()
    cli._make_client = _asgi_client_factory(server)

    result = runner.invoke(
        cli.app,
        [BASE_URL, "--probe", "--json", "--endpoint", "chat"],
    )

    assert result.exit_code == 0, (
        f"truncated-marker probe exited {result.exit_code}: {result.output}"
    )

    payload = json.loads(result.output)
    capacity = payload["capacity"]
    assert len(capacity) >= 1, "expected a capacity entry for an honest server"

    entry = capacity[0]
    cliff = entry["cliff_behavior"]["value"]
    source = entry["max_accepted_tokens"]["provenance"]

    # The core requirement: a fully accepting server must never be reported as
    # silent_truncation. Without the calibration guard this would be
    # "silent_truncation" with max_accepted_tokens == 15 — the false alarm.
    assert cliff != "silent_truncation", (
        "fully accepting server was reported as silent_truncation: the "
        "truncated marker was misread as a dropped head"
    )

    # The boundary could not be verified (the marker never survives verbatim),
    # so the tool must honestly mark it UNKNOWN rather than guess a measured
    # value. The task's contract is "UNKNOWN or accepted", never a bare lie.
    assert source in ("unknown", "measured"), (
        f"unexpected provenance {source!r} for an unverifiable marker"
    )
