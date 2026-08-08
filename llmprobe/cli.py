"""typer entrypoint for llmprobe.

Wires together configuration reads (:mod:`llmprobe.probes.config`), per-slot
arithmetic checks (:mod:`llmprobe.probes.slots`), and the capacity cliff probe
(:mod:`llmprobe.probes.capacity`) into a single CLI command. Emits either a
markdown capability card or JSON via :mod:`llmprobe.report` and derives the
process exit code from the severity of any findings.

Imports only from :mod:`llmprobe.models`, the probe modules, and
:mod:`llmprobe.report`.
"""

from __future__ import annotations

import asyncio
from enum import Enum
from typing import Annotated

import httpx
import typer
from rich.console import Console

from llmprobe.probes.capacity import DEFAULT_CEILING, probe_capacity
from llmprobe.probes.config import read_effective_config
from llmprobe.probes.slots import check_slots
from llmprobe.report import to_json, to_markdown
from llmprobe.models import (
    CapacityResult,
    CliffBehavior,
    Finding,
    ProbeReport,
    Severity,
)

app = typer.Typer(
    name="llmprobe",
    help=(
        "Point it at a local inference server: reports what it can actually "
        "do, measured, not claimed."
    ),
    add_completion=False,
)

_console = Console()


class Endpoint(str, Enum):
    """Which inference endpoint the capacity probe should exercise."""

    EMBEDDINGS = "embeddings"
    CHAT = "chat"
    AUTO = "auto"


def _resolve_path(endpoint: Endpoint) -> str:
    """Map an ``Endpoint`` selection onto a concrete probe path."""
    if endpoint is Endpoint.CHAT:
        return "/v1/chat/completions"
    return "/v1/embeddings"


def _capacity_findings(
    reference: int | None,
    cap: CapacityResult,
) -> list[Finding]:
    """Surface a measured cliff below the claimed context as a finding.

    A cliff (silent truncation or hard error) found strictly below the context
    the caller was led to believe is a mismatch worth reporting. When there is
    no reference context, or the measured cliff is not below it, nothing is
    emitted.
    """
    if cap.cliff_behavior not in (
        CliffBehavior.SILENT_TRUNCATION,
        CliffBehavior.HARD_ERROR,
    ):
        return []
    if reference is None or cap.max_accepted_tokens >= reference:
        return []
    return [
        Finding(
            severity=Severity.MISMATCH,
            code="UBATCH_CEILING",
            advertised=reference,
            measured=cap.max_accepted_tokens,
            message=(
                f"requests past {cap.max_accepted_tokens} tokens are "
                f"{cap.cliff_behavior.value}"
            ),
        )
    ]


def _make_client(base_url: str) -> httpx.AsyncClient:
    """Create a fresh client bound to ``base_url``.

    This factory is the test seam: hermetic tests replace it with a client
    wired to an ``ASGITransport`` over the mock server.
    """
    return httpx.AsyncClient(base_url=base_url, timeout=httpx.Timeout(10.0))


async def _assert_reachable(client: httpx.AsyncClient) -> None:
    """Raise ``httpx.HTTPError`` if the server cannot be reached at all.

    A reachability probe is required because adapter detection treats a
    completely unreachable server as an empty ``generic`` match and its config
    read swallows the connection error, so ``read_effective_config`` would
    otherwise never raise. A single origin GET succeeds against any reachable
    server (whatever its status code) and raises only on a transport failure.
    """
    await client.get("")


async def probe(
    base_url: str,
    claimed_ctx: int | None,
    do_probe: bool,
    endpoint: Endpoint,
) -> ProbeReport:
    """Run the configured read and optional capacity probe, then assemble a report."""
    async with _make_client(base_url) as client:
        await _assert_reachable(client)
        config, findings = await read_effective_config(
            client, base_url, claimed_ctx
        )
        findings.extend(check_slots(config, claimed_ctx))

        capacity: list[CapacityResult] = []
        if do_probe:
            reference = (
                claimed_ctx if claimed_ctx is not None else config.n_ctx_total
            )
            cap = await probe_capacity(
                client,
                base_url,
                _resolve_path(endpoint),
                ceiling=DEFAULT_CEILING,
                backend=config.backend,
            )
            capacity.append(cap)
            findings.extend(_capacity_findings(reference, cap))

        return ProbeReport(
            base_url=base_url,
            config=config,
            capacity=capacity,
            findings=findings,
        )


@app.command()
def main(
    base_url: Annotated[
        str, typer.Argument(help="Base URL of the inference server.")
    ],
    claimed_ctx: Annotated[
        int | None,
        typer.Option(
            "--claimed-ctx",
            help=(
                "Context you believe the server has; enables mismatch checking."
            ),
        ),
    ] = None,
    probe_flag: Annotated[
        bool,
        typer.Option(
            "--probe/--safe",
            help=(
                "--probe sends inference load to find the real capacity cliff; "
                "--safe (the default) reads configuration only."
            ),
        ),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit JSON instead of the markdown card."),
    ] = False,
    endpoint: Annotated[
        Endpoint,
        typer.Option("--endpoint", help="Which endpoint to probe."),
    ] = Endpoint.AUTO,
) -> None:
    """Probe ``BASE_URL`` and report what the server can actually do."""
    try:
        report = asyncio.run(probe(base_url, claimed_ctx, probe_flag, endpoint))
    except httpx.HTTPError as exc:
        typer.echo(f"llmprobe: unreachable or failed server: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    if json_output:
        typer.echo(to_json(report))
    else:
        _console.print(to_markdown(report))
    raise typer.Exit(code=report.exit_code)
