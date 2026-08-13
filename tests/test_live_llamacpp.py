"""The founding incident, against two real llama.cpp servers.

Everything else in this suite talks to a mock we wrote, which can only prove
that the code agrees with our own idea of how a server behaves. This module
talks to two real ``llama.cpp`` servers that differ in exactly one flag, and it
is the only place where "measured, not claimed" is true of llmprobe itself.

Run it with ``scripts/live_control.sh``, which starts both servers, exports the
two URLs and tears them down. Without those variables the module skips: a test
that silently passes when its subject is absent is worse than no test.

The pair is deliberately falsifiable in both directions. A detector that flags
everything would pass the broken case and fail the healthy one; a detector that
flags nothing would do the reverse. Only a real one passes both.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

BROKEN_URL = os.environ.get("LLMPROBE_LIVE_BROKEN_URL")
HEALTHY_URL = os.environ.get("LLMPROBE_LIVE_HEALTHY_URL")

pytestmark = pytest.mark.skipif(
    not (BROKEN_URL and HEALTHY_URL),
    reason=(
        "live control not configured; run scripts/live_control.sh to start a "
        "llama.cpp pair and set LLMPROBE_LIVE_BROKEN_URL / _HEALTHY_URL"
    ),
)


def _probe(url: str, *extra: str) -> tuple[int, dict]:
    """Run the installed CLI against ``url`` and return (exit code, report)."""
    proc = subprocess.run(
        [sys.executable, "-m", "llmprobe", url, "--probe", "--json",
         "--endpoint", "embeddings", *extra],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert proc.stdout, f"no report on stdout (exit {proc.returncode}): {proc.stderr}"
    return proc.returncode, json.loads(proc.stdout)


def _v(field: object) -> object:
    """Unwrap a reported field: every JSON value carries its own provenance."""
    return field["value"] if isinstance(field, dict) and "value" in field else field


def _max_input(report: dict) -> int:
    embeddings = [
        c for c in report["capacity"] if "embeddings" in str(_v(c["endpoint"]))
    ]
    assert embeddings, f"no embeddings capacity in report: {report['capacity']}"
    return int(_v(embeddings[0]["max_accepted_tokens"]))


def test_default_ubatch_server_is_caught_without_being_told_the_number() -> None:
    """A server capped by n_ubatch must fail the gate on its own numbers.

    This is the incident the tool exists for: the server is configured for
    8192 tokens per slot and says so on /props, but the physical batch caps
    every request far below that, and n_ubatch is not in /props at all. No
    --claimed-ctx is passed, because an operator who knew the real ceiling
    would have no reason to probe for it.
    """
    code, report = _probe(BROKEN_URL)

    measured = _max_input(report)
    ctx = int(_v(report["config"]["n_ctx_per_slot"]))
    assert measured * 2 < ctx, (
        f"expected a capacity far below the configured context, "
        f"measured {measured} against {ctx}"
    )

    findings = [
        f for f in report["findings"] if _v(f["severity"]) == "mismatch"
    ]
    assert findings, f"a {ctx // measured}x shortfall produced no finding"
    assert _v(findings[0]["advertised"]) == ctx, (
        "the accusation must quote the server's own reported context"
    )
    assert code == 1, f"a capacity mismatch must exit 1 for CI gating, got {code}"


def test_healthy_server_passes_clean_and_the_remedy_is_never_the_broken_value(
) -> None:
    """The same image with -b/-ub raised must produce no finding at all.

    Guards the false alarm: capacity lands a couple of tokens below the
    configured context because of BOS/EOS, and reporting that as a defect
    reddens a correctly configured server.
    """
    code, report = _probe(HEALTHY_URL, "--claimed-ctx", "8192")

    measured = _max_input(report)
    ctx = int(_v(report["config"]["n_ctx_per_slot"]))
    assert measured >= ctx - 64, (
        f"healthy server measured {measured}, far below its configured {ctx}"
    )
    assert report["findings"] == [], f"false alarm on a healthy server: {report}"
    assert code == 0, f"a healthy server must exit 0, got {code}"


def test_remedy_does_not_repeat_the_ceiling_that_caused_the_defect() -> None:
    """The Fix printed for the broken server must raise the batch, not pin it.

    The remedy used to be derived from the measured cliff, so a server broken
    by ubatch=512 was advised to set --ubatch-size 512. Asserting on the
    rendered card keeps that specific harm from returning.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "llmprobe", BROKEN_URL, "--probe",
         "--endpoint", "embeddings"],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    card = proc.stdout
    assert "## Fix" in card, f"a mismatch must come with a remedy: {card}"

    fix = card.split("## Fix", 1)[1]
    code, report = _probe(BROKEN_URL)
    measured = _max_input(report)
    ceiling = 1
    while ceiling < measured:
        ceiling *= 2
    assert f"--ubatch-size {ceiling}" not in fix, (
        f"remedy pins the batch to the broken ceiling {ceiling}: {fix}"
    )
    assert str(_v(report["config"]["n_ctx_per_slot"])) in fix, (
        f"remedy does not restore the configured context: {fix}"
    )
