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
import os
import re
from enum import Enum
from typing import Annotated

import httpx
import typer
from rich.console import Console

from llmprobe.backends import DEFAULT_PROBE_ENDPOINTS
from llmprobe.probes.capacity import DEFAULT_CEILING, probe_capacity
from llmprobe.probes.config import read_effective_config
from llmprobe.probes.slots import check_slots
from llmprobe.report import to_json, to_markdown
from llmprobe.models import (
    Backend,
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

_API_KEY_ENV = "LLMPROBE_API_KEY"

# A base URL may carry its secrets inline (https://user:pass@host). That userinfo
# block is a credential and must never be echoed into output or logs.
_USERINFO_RE = re.compile(r"(//[^/@]+@)")

#: Default per-request timeout in seconds (applied to every HTTP request).
DEFAULT_TIMEOUT = 10.0


class Endpoint(str, Enum):
    """Which inference endpoint the capacity probe should exercise."""

    EMBEDDINGS = "embeddings"
    CHAT = "chat"
    AUTO = "auto"


def redact_base_url(base_url: str) -> str:
    """Return ``base_url`` with any inline credentials stripped.

    A caller may embed an API key in the URL (``https://key@host``). That
    secret must not reach the card or logs, so the ``userinfo`` segment is
    removed for display while the connection URL is left untouched.
    """
    return _USERINFO_RE.sub("//", base_url)


def _resolve_path(endpoint: Endpoint, backend: Backend) -> str:
    """Map an ``Endpoint`` selection onto a concrete probe path.

    ``AUTO`` is resolved against the detected ``backend`` using the per-backend
    default probe endpoint (see :data:`llmprobe.backends.DEFAULT_PROBE_ENDPOINTS`),
    so the endpoint actually exercised matches the backend type rather than
    always defaulting to embeddings. Explicit ``CHAT`` / ``EMBEDDINGS`` choices
    are honoured directly and never overridden.
    """
    if endpoint is Endpoint.CHAT:
        return "/v1/chat/completions"
    if endpoint is Endpoint.EMBEDDINGS:
        return "/v1/embeddings"
    if endpoint is Endpoint.AUTO:
        return DEFAULT_PROBE_ENDPOINTS[backend]
    raise ValueError(f"unknown endpoint: {endpoint!r}")


def _capacity_findings(
    claimed_ctx: int | None,
    cap: CapacityResult,
) -> list[Finding]:
    """Surface a measured cliff below the claimed context as a mismatch finding.

    A cliff (silent truncation or hard error) found strictly below the context
    the caller claimed with ``--claimed-ctx`` is a mismatch worth failing on.
    When there is no ``claimed_ctx``, or the measured cliff is not below it,
    nothing is emitted. Comparing ``claimed_ctx`` against the measured
    ``cap.max_accepted_tokens`` is the mismatch check that drives exit code 1.
    """
    if claimed_ctx is None:
        return []
    if cap.cliff_behavior not in (
        CliffBehavior.SILENT_TRUNCATION,
        CliffBehavior.HARD_ERROR,
    ):
        return []
    if cap.max_accepted_tokens >= claimed_ctx:
        return []
    return [
        Finding(
            severity=Severity.MISMATCH,
            code="UBATCH_CEILING",
            advertised=claimed_ctx,
            measured=cap.max_accepted_tokens,
            message=(
                f"requests past {cap.max_accepted_tokens} tokens are "
                f"{cap.cliff_behavior.value}"
            ),
        )
    ]


def _make_client(
    base_url: str, api_key: str | None = None, timeout: float = DEFAULT_TIMEOUT
) -> httpx.AsyncClient:
    """Create a fresh client bound to ``base_url``.

    This factory is the test seam: hermetic tests replace it with a client
    wired to an ``ASGITransport`` over the mock server. The ``timeout`` is
    applied to every HTTP request the client issues, so callers thread it
    through to bound all requests. When an ``api_key`` is given it is attached
    as a ``Bearer`` token on the ``Authorization`` header only — the key is
    never placed in the URL, body, or report, so it cannot leak into the card
    or logs.
    """
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    return httpx.AsyncClient(
        base_url=base_url,
        timeout=httpx.Timeout(timeout),
        headers=headers,
    )


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
    timeout: float = DEFAULT_TIMEOUT,
    api_key: str | None = None,
) -> ProbeReport:
    """Run the configured read and optional capacity probe, then assemble a report."""
    async with _make_client(base_url, api_key, timeout) as client:
        await _assert_reachable(client)
        config, findings = await read_effective_config(
            client, base_url, claimed_ctx, timeout
        )
        findings.extend(check_slots(config, claimed_ctx))

        capacity: list[CapacityResult] = []
        if do_probe:
            cap = await probe_capacity(
                client,
                base_url,
                _resolve_path(endpoint, config.backend),
                ceiling=DEFAULT_CEILING,
                backend=config.backend,
                timeout=timeout,
            )
            capacity.append(cap)
            # Compare --claimed-ctx against the measured max_accepted_tokens;
            # a claimed_ctx mismatch is surfaced as a MISMATCH finding => exit 1.
            findings.extend(_capacity_findings(claimed_ctx, cap))

        report_url = redact_base_url(base_url)
        return ProbeReport(
            base_url=report_url,
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
    timeout: Annotated[
        float,
        typer.Option(
            "--timeout",
            help=(
                "Per-request timeout in seconds applied to every HTTP request."
            ),
        ),
    ] = DEFAULT_TIMEOUT,
    api_key: Annotated[
        str | None,
        typer.Option(
            "--api-key",
            envvar=_API_KEY_ENV,
            help=(
                "Bearer token sent as the Authorization header. Never echoed "
                "into the card or logs."
            ),
        ),
    ] = None,
) -> None:
    """Probe ``BASE_URL`` and report what the server can actually do."""
    try:
        report = asyncio.run(
            probe(base_url, claimed_ctx, probe_flag, endpoint, timeout, api_key)
        )
    except httpx.HTTPError as exc:
        typer.echo(
            f"llmprobe: unreachable or failed server: "
            f"{redact_base_url(str(exc))}",
            err=True,
        )
        raise typer.Exit(code=2) from exc

    if json_output:
        typer.echo(to_json(report))
    else:
        _console.print(to_markdown(report))
    raise typer.Exit(code=report.exit_code)
