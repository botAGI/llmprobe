"""Shared pydantic v2 contracts for llmprobe.

Pure data models only: no I/O, no network, no imports from other llmprobe
modules. Every reported value carries a provenance marker
(see the ``Provenance`` enum and ``EffectiveConfig.sources``).
"""

from enum import Enum
from typing import Optional
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, Field


def redact_userinfo(url: str) -> str:
    """Strip ``userinfo`` (credentials) from a URL for display.

    A base URL may carry a secret in its userinfo slot, e.g.
    ``http://sk-1234@host:8080`` or ``http://user:token@host/v1``. Pasted into
    issues, JSON, and logs, such credentials must never be printed. This pure
    string helper removes the ``username:password@`` segment for display while
    leaving URLs without userinfo untouched. The request URL used on the wire
    is unaffected — only the human/machine-facing form is redacted.
    """
    parts = urlsplit(url if "://" in url else f"//{url}")
    if not (parts.username or parts.password):
        return url
    hostname = parts.hostname or ""
    port = f":{parts.port}" if parts.port else ""
    cleaned = urlunsplit(
        (parts.scheme, f"{hostname}{port}", parts.path, parts.query, parts.fragment)
    )
    return cleaned if cleaned else url


class Backend(str, Enum):
    """Inference backends llmprobe is able to probe."""

    LLAMACPP = "llamacpp"
    VLLM = "vllm"
    OLLAMA = "ollama"
    GENERIC = "generic"


class Endpoint(str, Enum):
    """Which inference endpoint a capacity probe should exercise."""

    EMBEDDINGS = "embeddings"
    CHAT = "chat"
    AUTO = "auto"


class Provenance(str, Enum):
    """How a reported value was obtained.

    - READ: the server told us the value directly.
    - MEASURED: we probed the value ourselves.
    - INFERRED: derived from other values.
    - UNKNOWN: we could not tell.
    """

    READ = "read"
    MEASURED = "measured"
    INFERRED = "inferred"
    UNKNOWN = "unknown"


class EffectiveConfig(BaseModel):
    """The configuration effectively in force on a probed server.

    Numeric fields are Optional because a backend may not expose them. Each
    numeric field MUST have a corresponding provenance entry in ``sources``.
    """

    backend: Backend
    model_id: str
    n_ctx_total: Optional[int] = None
    n_ctx_per_slot: Optional[int] = None
    n_batch: Optional[int] = None
    n_ubatch: Optional[int] = None
    total_slots: Optional[int] = None
    n_ctx: Optional[int] = None

    sources: dict[str, Provenance] = Field(
        default_factory=dict,
        description=(
            "Provenance marker for each numeric field name, including n_ctx."
        ),
    )


class CliffBehavior(str, Enum):
    """How the server behaves when context length is exceeded."""

    ACCEPTED = "accepted"
    SILENT_TRUNCATION = "silent_truncation"
    HARD_ERROR = "hard_error"


class CapacityResult(BaseModel):
    """Outcome of probing a single endpoint's real capacity.

    ``max_accepted_tokens`` carries its own provenance because a binary search
    that rejects every probed length (real capacity below :data:`LO`) cannot
    report an accepted length, and one that accepts the ceiling cannot know the
    true maximum (it lies above ``ceiling``); in both cases the caller must be
    able to see that the integer is a lower-bound estimate rather than a
    measured observation.

    ``token_count_exact`` records whether the probed lengths were verified to
    be exactly ``max_accepted_tokens`` tokens via a live ``/tokenize`` endpoint.
    When tokenization is unavailable the prompts are nominal estimates and this
    flag is ``False`` — an honest "we could not verify the count" signal rather
    than a confident guess.
    """

    endpoint: str
    max_accepted_tokens: int
    max_accepted_source: Provenance = Provenance.MEASURED
    token_count_exact: bool = True
    cliff_behavior: CliffBehavior
    probe_requests_used: int


class Severity(str, Enum):
    """Severity of a finding."""

    INFO = "info"
    MISMATCH = "mismatch"
    ERROR = "error"


class Finding(BaseModel):
    """A single discrepancy (or note) discovered during a probe.

    ``advertised`` is whatever value the server or the caller led us to
    believe, so its provenance is ``read`` unless stated otherwise; ``measured``
    is always the value we probed or derived ourselves, hence ``measured``.
    Both carry an explicit provenance marker so the "provenance on every value"
    guarantee holds even outside the config/capacity tables. When the
    corresponding value is ``None`` the source marker is irrelevant.
    """

    severity: Severity
    code: str
    advertised: str | int | None = None
    advertised_source: Provenance = Provenance.READ
    measured: str | int | None = None
    measured_source: Provenance = Provenance.MEASURED
    message: str


class ProbeReport(BaseModel):
    """Full result of probing a base_url."""

    base_url: str
    config: EffectiveConfig
    capacity: list[CapacityResult]
    findings: list[Finding] = Field(default_factory=list)

    @property
    def exit_code(self) -> int:
        """Process exit code derived from findings.

        2 if any finding has severity ERROR,
        else 1 if any finding has severity MISMATCH,
        else 0.
        """
        if any(f.severity == Severity.ERROR for f in self.findings):
            return 2
        if any(f.severity == Severity.MISMATCH for f in self.findings):
            return 1
        return 0
