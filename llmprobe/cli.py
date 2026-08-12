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
import sys
from typing import Annotated

import httpx
import typer
from rich.console import Console

from llmprobe.backends import DEFAULT_PROBE_ENDPOINTS
from llmprobe.probes.capacity import DEFAULT_CEILING, probe_capacity
from llmprobe.probes.config import read_effective_config
from llmprobe.probes.slots import check_slots
from llmprobe.report import to_json, to_json_schema, to_markdown
from llmprobe.models import (
    Backend,
    CapacityResult,
    CliffBehavior,
    Endpoint,
    Finding,
    ProbeReport,
    Severity,
    redact_userinfo,
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

#: Default per-request timeout in seconds (applied to every HTTP request).
#: Generous by design: probing long inputs can exceed a short timeout, and the
#: capacity probe scales this base up proportionally to the prompt's token
#: count.
DEFAULT_TIMEOUT = 10.0


def redact_base_url(base_url: str) -> str:
    """Return ``base_url`` with any inline credentials stripped.

    A caller may embed an API key in the URL (``https://key@host``). That
    secret must not reach the card or logs, so the ``userinfo`` segment is
    removed for display while the connection URL is left untouched.
    """
    return redact_userinfo(base_url)


def _coerce_endpoint(endpoint: Endpoint | str) -> Endpoint:
    """Normalize a raw ``--endpoint`` value into an ``Endpoint`` member.

    The README documents ``--endpoint chat`` (and ``embeddings`` / ``auto``) as
    valid selections. Because the enum is a ``str``-backed ``Endpoint``, a
    string value such as ``'chat'`` is coerced here to ``Endpoint.CHAT`` so the
    CLI accepts the documented spelling rather than demanding ``Endpoint.CHAT``.

    ``'choice'`` is accepted as a synonym for ``auto`` (the deliberate default
    that resolves per backend), matching the prose use of "choice" in the README.
    """
    if isinstance(endpoint, Endpoint):
        return endpoint
    if isinstance(endpoint, str):
        value = endpoint.lower()
        if value == "choice":
            return Endpoint.AUTO
        try:
            return Endpoint(value)
        except ValueError:
            raise ValueError(
                f"unknown endpoint: {endpoint!r} (expected one of "
                f"{[e.value for e in Endpoint]})"
            ) from None
    raise ValueError(f"unknown endpoint: {endpoint!r}")


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
    safe: bool = True,
    endpoint: Endpoint | str = Endpoint.AUTO,
    timeout: float = DEFAULT_TIMEOUT,
    api_key: str | None = None,
    *,
    chat: bool = False,
) -> ProbeReport:
    """Run the configured read and optional capacity probe, then assemble a report.

    ``endpoint`` selects which endpoint to exercise and may be given either as
    an ``Endpoint`` member or as its documented string spelling (``"chat"``,
    ``"embeddings"``, ``"auto"``); matching strings are coerced to the
    ``Endpoint`` member. For the README-promised ``--endpoint chat`` the
    ``chat=True`` shorthand is an equivalent, explicit way to select the chat
    endpoint (``Endpoint.CHAT``); when given, it takes precedence.
    """
    endpoint = Endpoint.CHAT if chat else _coerce_endpoint(endpoint)
    async with _make_client(base_url, api_key, timeout) as client:
        try:
            await _assert_reachable(client)
        except httpx.HTTPError:
            raise
        config, findings = await read_effective_config(
            client, base_url, claimed_ctx, timeout, endpoint
        )
        findings.extend(check_slots(config, claimed_ctx))

        capacity: list[CapacityResult] = []
        # Selecting an explicit endpoint (CHAT/EMBEDDINGS) requests inference on
        # that endpoint, so probe traffic is sent even without --probe. Only the
        # default AUTO selection honours the --safe/--probe suppression.
        effective_safe = safe and endpoint is Endpoint.AUTO
        cap = await probe_capacity(
            client,
            base_url,
            endpoint,
            ceiling=DEFAULT_CEILING,
            backend=config.backend,
            model=config.model_id,
            timeout=timeout,
            safe=effective_safe,
        )
        if cap is not None:
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
        str | None,
        typer.Argument(
            help=(
                "Base URL of the inference server. Required unless --json-schema "
                "is given."
            ),
        ),
    ] = None,
    json_schema: Annotated[
        bool,
        typer.Option(
            "--json-schema",
            help=(
                "Print the JSON schema of the report (from the pydantic model) "
                "to stdout and exit without probing."
            ),
        ),
    ] = False,
    claimed_ctx: Annotated[
        int | None,
        typer.Option(
            "--claimed-ctx",
            help=(
                "Context you believe the server has; enables mismatch checking."
            ),
        ),
    ] = None,
    safe: Annotated[
        bool,
        typer.Option(
            "--safe/--probe",
            help=(
                "--safe (the default) reads configuration only; "
                "--probe sends inference load to find the real capacity cliff."
            ),
        ),
    ] = True,
    no_safe: Annotated[
        bool,
        typer.Option(
            "--no-safe",
            help=(
                "--no-safe is an alias for --probe: send inference load to "
                "find the real capacity cliff instead of reading config only."
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
    if json_schema:
        typer.echo(to_json_schema())
        raise typer.Exit(code=0)

    if base_url is None:
        raise typer.BadParameter("BASE_URL is required unless --json-schema is given")

    try:
        report = asyncio.run(
            probe(
                base_url,
                claimed_ctx,
                safe=safe and not no_safe,
                endpoint=endpoint,
                timeout=timeout,
                api_key=api_key,
            )
        )
    except httpx.HTTPError as exc:
        typer.echo(
            f"llmprobe: unreachable or failed server: "
            f"{redact_base_url(str(exc))}",
            err=True,
        )
        sys.exit(2)
    except ValueError as exc:
        typer.echo(f"llmprobe: invalid endpoint: {exc}", err=True)
        sys.exit(2)

    if json_output:
        typer.echo(to_json(report))
    else:
        _console.print(to_markdown(report))
    raise typer.Exit(code=report.exit_code)
