"""Exact-length prompt construction.

Two strategies, both honest about what we actually verified:

* If the server exposes ``POST /tokenize`` (both llama.cpp and vLLM do), we
  build a prompt by repeating a filler word, re-tokenize to verify the count,
  and adjust across at most :data:`MAX_ATTEMPTS` iterations until the prompt
  tokenizes to EXACTLY ``n`` tokens. We return ``exact=True`` because we
  verified the count ourselves.
* If ``/tokenize`` returns 404, we repeat a calibrated single-token ASCII
  filler (``" the"``) and return ``exact=False``. The caller must be able to
  see that the length is approximate — we never claim exactness we did not
  verify.

Only imports from :mod:`llmprobe.models`.
"""

from __future__ import annotations

import httpx

from llmprobe.models import Backend

MAX_ATTEMPTS = 5

_FILLER = "token"
_APPROX_FILLER = " the"


def _base(base_url: str) -> str:
    return base_url.rstrip("/")


def _approximate(n: int) -> str:
    """A prompt of ``n`` presumed-token words using a single-token filler.

    Not verified against the server, so callers must treat the length as
    approximate (``exact=False``).
    """
    return (_APPROX_FILLER * max(0, n)).strip()


async def _tokenize(
    client: httpx.AsyncClient, base_url: str, prompt: str
) -> int | None:
    """Return the server-reported token count for ``prompt``.

    Returns ``None`` when the endpoint is absent (404) or unreadable — we do
    not fabricate a count.
    """
    try:
        resp = await client.post(f"{_base(base_url)}/tokenize", json={"content": prompt})
    except httpx.HTTPError:
        return None
    if resp.status_code != 200:
        return None
    try:
        payload = resp.json()
    except ValueError:
        return None
    tokens = payload.get("tokens") if isinstance(payload, dict) else None
    if not isinstance(tokens, list):
        return None
    return len(tokens)


async def make_prompt_of_exactly(
    client: httpx.AsyncClient,
    base_url: str,
    n: int,
    backend: Backend,
) -> tuple[str, bool]:
    """Build a prompt that aims for exactly ``n`` tokens.

    Returns ``(prompt, exact)`` where ``exact`` is ``True`` only when the
    count was verified to be exactly ``n`` against a live ``/tokenize``
    endpoint. When the endpoint is unavailable we fall back to an estimate and
    return ``exact=False`` so the caller never mistakes an approximation for a
    measured value.

    ``backend`` is accepted for API symmetry; the request shape sent to
    ``/tokenize`` is stable across the supported backends.
    """
    base = _base(base_url)

    probe = await _tokenize(client, base, _FILLER)
    if probe is None or probe <= 0:
        # Strategy 2: no verifiable tokenizer, calibrated estimate.
        return _approximate(n), False

    # Strategy 1: calibrate then adjust. One tokenize call per iteration,
    # never more than MAX_ATTEMPTS total (probe included).
    reps = max(1, round(n / probe))
    requests = 1
    while requests < MAX_ATTEMPTS:
        prompt = " ".join([_FILLER] * reps)
        got = await _tokenize(client, base, prompt)
        requests += 1
        if got is None:
            return prompt, False
        if got == n:
            return prompt, True
        per_rep = got / reps if reps else 1.0
        reps = max(1, round(reps + (n - got) / per_rep))

    prompt = " ".join([_FILLER] * reps)
    return prompt, False
