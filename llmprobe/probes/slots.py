"""Per-slot context arithmetic checks.

Pure computation over the effective configuration reported by a server.
Imports only from :mod:`llmprobe.models` — no I/O, no network.
"""

from __future__ import annotations

from llmprobe.models import (
    EffectiveConfig,
    Finding,
    Severity,
)


def check_slots(
    config: EffectiveConfig,
    claimed_ctx: int | None,
) -> list[Finding]:
    """Verify per-slot context against the server's total context.

    When both ``total_slots`` and ``n_ctx_total`` are known we derive the
    per-slot context as ``n_ctx_total // total_slots``. If that derived value
    disagrees with the per-slot context the server reported, or with the
    ``claimed_ctx`` the caller was told, we emit a ``CTX_PER_SLOT_MISMATCH``
    finding. ``advertised`` carries the caller's claimed value and
    ``measured`` the value derived from the server's own totals.

    If either ``total_slots`` or ``n_ctx_total`` is unknown we cannot derive a
    per-slot figure and emit no findings for this check.
    """
    if config.total_slots is None or config.n_ctx_total is None:
        return []
    if config.total_slots <= 0:
        return []

    derived_per_slot = config.n_ctx_total // config.total_slots

    reported = config.n_ctx_per_slot
    reported_mismatch = derived_per_slot != reported
    claimed_mismatch = claimed_ctx is not None and derived_per_slot != claimed_ctx
    mismatched = reported_mismatch or claimed_mismatch
    if not mismatched:
        return []

    reported_part = (
        f", reported per-slot context ({reported})"
        if reported_mismatch and reported is not None
        else ""
    )
    claimed_part = (
        f", claimed context ({claimed_ctx})"
        if claimed_mismatch and claimed_ctx is not None
        else ""
    )
    return [
        Finding(
            severity=Severity.MISMATCH,
            code="CTX_PER_SLOT_MISMATCH",
            advertised=claimed_ctx,
            measured=derived_per_slot,
            message=(
                f"derived per-slot context ({derived_per_slot}) disagrees with "
                f"the total-slot arithmetic{reported_part}{claimed_part}"
            ),
        )
    ]
