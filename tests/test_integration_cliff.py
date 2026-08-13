"""Integration test against the real fake truncating server.

The neighbouring task ``lp-fego`` added ``tests/fake_server.py``, a real HTTP
server that silently truncates incoming prompts past a configured token limit.
This test boots that server as a genuine OS subprocess bound to an ephemeral
port and drives the real ``llmprobe`` CLI over real HTTP against it.

A server that silently truncates at 4096 tokens is the exact lie a capacity
probe exists to catch. The probe must:

* report the ``silent_truncation`` verdict — never ``accepted`` or ``unknown``,
  because a truncating server silently discards input and must be named;
* measure a boundary (the largest accepted length) within 5% of the true 4096
  cut — never an overconfident or fabricated guess.

Hermetic: the only network traffic is to the local fake server subprocess on
``127.0.0.1``; nothing reaches the outside world.
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# The fake server truncates at this many tokens; the probe must find the
# boundary near this value, not an approximation.
TRUNCATION_CTX = 4096
# The measured boundary must land within this fraction of the true cut.
_TRUNCATION_TOLERANCE = 0.05


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
    import httpx

    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
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


def _run_cli(base_url: str) -> subprocess.CompletedProcess[str]:
    """Drive the real CLI against ``base_url`` and return the result.

    Uses the chat endpoint because the fake server's silent truncation drops
    the head of an oversized prompt, which is what the canary-based chat probe
    detects. ``--json`` returns a parseable capability card.
    """
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "llmprobe",
            base_url,
            "--endpoint",
            "chat",
            "--json",
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


def test_fake_server_silent_truncation_at_4096_reports_boundary_within_five_percent() -> None:
    """The real fake truncating server is caught, measured, and named.

    Booting ``tests/fake_server.py --truncate-len 4096 --mode silent`` as an
    OS process and probing it with the real CLI must surface a measured
    boundary within 5% of 4096 and the ``silent_truncation`` verdict —
    the same lie a real deployment would present.
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
            str(TRUNCATION_CTX),
            "--mode",
            "silent",
        ],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_until_reachable(base_url)
        result = _run_cli(base_url)
        assert result.returncode == 0, result.stdout + result.stderr

        payload = json.loads(result.stdout)
        assert payload["capacity"], "expected the capacity probe to report"
        for entry in payload["capacity"]:
            verdict = entry["cliff_behavior"]["value"]
            assert verdict == "silent_truncation", (
                f"fake server was classified {verdict!r}, expected "
                "'silent_truncation'"
            )

            boundary = entry["max_accepted_tokens"]["value"]
            tolerance = TRUNCATION_CTX * _TRUNCATION_TOLERANCE
            lower = TRUNCATION_CTX - tolerance
            upper = TRUNCATION_CTX + tolerance
            assert lower <= boundary <= upper, (
                f"measured boundary {boundary} deviates from the true "
                f"4096-token cut by more than {_TRUNCATION_TOLERANCE:.0%}"
            )
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=10)
