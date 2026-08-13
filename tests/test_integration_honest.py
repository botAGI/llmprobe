"""Integration test — the real CLI against a real honest (non-truncating) server.

This is a true integration test: it boots ``tests/fake_server.py`` as a genuine
OS process bound to a real port and drives the real ``llmprobe`` CLI over real
HTTP against it. The server is configured to never truncate and never refuse
(referred mode with an input ceiling far above anything the probe emits), i.e.
an honest server.

An honest server must be reported as ``accepted`` with zero false alarms: every
exercised capacity endpoint must carry the ``accepted`` verdict, the report
must contain no findings, and the process must exit 0. If the probe wrongly
flags an honest server as ``silent_truncation`` (the historical false alarm) or
raises a mismatch finding, this test turns red.

Hermetic: no network, no third-party inference server. The only external
process is the local fake server, exactly as every other integration test does.
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Far above anything the chat probe emits, so the honest server never refuses
# and never truncates regardless of the probed length.
HONEST_CEILING = 100_000


def _free_port() -> int:
    """Return an ephemeral port that is free right now.

    Bind a socket to port 0, read the OS-assigned port, then close it so the
    fake server (started as a real subprocess below) can bind it. A tiny race
    window exists but is negligible in a hermetic test environment.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_until_reachable(base_url: str, timeout: float = 60.0) -> None:
    """Poll ``/props`` until the fake server answers or ``timeout`` elapses.

    The fake server is a real OS subprocess; its startup is asynchronous, so
    the probe must not race it. We poll the compatibility surface rather than
    sleeping a fixed amount, so a slow first-boot is tolerated within reason.
    """
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            import httpx

            with httpx.Client(timeout=2.0) as client:
                response = client.get(f"{base_url}/props")
                if response.status_code == 200:
                    return
        except Exception as exc:  # noqa: BLE001 - any connect/read error means not ready
            last_error = exc
        time.sleep(0.1)
    raise TimeoutError(
        f"fake server did not become reachable at {base_url}: {last_error}"
    ) from last_error


def _run(args: list[str], timeout: int = 300) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def test_integration_honest_server_verdict_accepted_no_false_alarms() -> None:
    """A real non-truncating server is reported accepted, never a false alarm.

    The genuine fake server is launched as an OS subprocess in honest mode
    (referred, input ceiling far above anything the probe emits) and the real
    CLI is driven over real HTTP. Every capacity verdict must be ``accepted``,
    the report must carry no findings, and the exit code must be 0.
    """
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "tests.fake_server",
            "--port",
            str(port),
            "--truncate-len",
            str(HONEST_CEILING),
            "--mode",
            "refuse",
        ],
        cwd=str(_PROJECT_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_until_reachable(base_url)
        result = _run(
            [
                sys.executable,
                "-m",
                "llmprobe",
                base_url,
                "--probe",
                "--endpoint",
                "chat",
                "--json",
            ]
        )
        assert result.returncode == 0, (
            f"honest server probe exited {result.returncode}: "
            f"{result.stdout + result.stderr}"
        )

        payload = json.loads(result.stdout)

        # An honest server produces no false alarms: no findings at all.
        assert payload["findings"] == [], (
            f"honest server raised false alarms: {payload['findings']}"
        )

        # Every exercised endpoint must carry the honest ``accepted`` verdict.
        assert payload["capacity"], "expected the capacity probe to report"
        for entry in payload["capacity"]:
            verdict = entry["cliff_behavior"]["value"]
            assert verdict == "accepted", (
                f"honest server was flagged {verdict!r}, expected 'accepted' "
                "(false alarm)"
            )
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=10)
