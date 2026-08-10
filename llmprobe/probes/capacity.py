"""Capacity cliff detection via binary search over input length.

The core product: given an inference endpoint, find the largest input length
the server genuinely accepts, and report *how* it fails beyond that point.

We never trust an HTTP 200 by itself. For ``/v1/embeddings`` we send TWO
prompts of the same length ``n`` that differ only in their FINAL token and
compare the returned vectors. A server that silently drops the tail returns
identical vectors (cosine similarity effectively 1.0) regardless of the
differing tail — we call that ``SILENT_TRUNCATION``. Only when the two
responses genuinely differ do we accept the length. A 4xx/5xx is
``HARD_ERROR``. For ``/v1/chat/completions`` we plant a unique canary at the
start and ask the model to repeat the first word; an absent canary reveals
head truncation.

Imports only from :mod:`llmprobe.models`.
"""

from __future__ import annotations

import math

import httpx

from llmprobe.backends.vllm import extract_prompt_tokens
from llmprobe.models import Backend, CapacityResult, CliffBehavior, Provenance
from llmprobe.tokens import _tokenize

LO = 16
DEFAULT_CEILING = 32768
COSINE_SIMILARITY_THRESHOLD = 0.9999

_FILLER = "tok"
_FINAL_A = "llmprobeFinalA"
_FINAL_B = "llmprobeFinalB"
_CANARY = "llmprobeCanary"


def _n_token_prompt(n: int, final: str) -> str:
    """Build a prompt of ``n`` presumed single-token words ending in ``final``.

    Each whitespace-delimited word is treated as one token. No tokenizer is
    invoked — the count is a nominal estimate, acceptable here because the
    binary-search *classification* depends on the server's own responses, not
    on the prompt's exact token count.
    """
    if n <= 1:
        return final
    return " ".join([_FILLER] * (n - 1) + [final])


async def _exact_tokenization_available(client: httpx.AsyncClient, base_url: str) -> bool:
    """Return ``True`` when we can trust ``_n_token_prompt`` counts as exact.

    The prompts built by :func:`_n_token_prompt` are exactly ``n`` tokens only
    if every marker word it can contain is itself a single token. We verify all
    four markers against the live ``/tokenize`` endpoint in one request; any
    word that tokenizes to more (or fewer) than one token makes the count a
    nominal estimate, for which we must report ``exact=False``.
    """
    probe = " ".join([_FILLER, _FINAL_A, _FINAL_B, _CANARY])
    count = await _tokenize(client, base_url.rstrip("/"), probe)
    return count == 4


async def _vllm_prompt_tokens_exact(
    client: httpx.AsyncClient, base_url: str, endpoint: str
) -> bool:
    """For vLLM, verify a probe length against the server's own count.

    vLLM reports ``usage.prompt_tokens`` on its responses, so the server itself
    is the source of truth for the exact count. We send one known-length probe
    and trust the response only when ``usage.prompt_tokens`` equals the exact
    length we asked for (exported via :func:`extract_prompt_tokens`); anything
    else means the response carried no usable count and we report an estimate.
    """
    n = LO
    base = base_url.rstrip("/")
    body: dict = {"input": " ".join([_FILLER] * n), "model": "vllm-probe"}
    try:
        resp = await client.post(f"{base}{endpoint}", json=body)
    except httpx.HTTPError:
        return False
    if resp.status_code != 200:
        return False
    try:
        payload = resp.json()
    except ValueError:
        return False
    reported = extract_prompt_tokens(payload)
    return reported == n


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


async def _post_embed(
    client: httpx.AsyncClient, base_url: str, prompt: str
) -> list[float] | None:
    """POST one embedding; return the vector on 200 else ``None``."""
    base = base_url.rstrip("/")
    try:
        resp = await client.post(
            f"{base}/v1/embeddings", json={"input": prompt, "model": "embed-mock"}
        )
    except httpx.HTTPError:
        return None
    if resp.status_code != 200:
        return None
    try:
        payload = resp.json()
    except ValueError:
        return None
    try:
        return list(payload["data"][0]["embedding"])
    except (KeyError, IndexError, TypeError):
        return None


async def _embed_classify(
    client: httpx.AsyncClient, base_url: str, n: int, requests: list[int]
) -> str:
    """Classify length ``n`` against ``/v1/embeddings``.

    Two identical-length prompts differing only in the final token. A 4xx/5xx
    is ``hard_error``; effectively identical vectors are ``silent_truncation``;
    genuinely different vectors are ``accepted``.
    """
    a = _n_token_prompt(n, _FINAL_A)
    b = _n_token_prompt(n, _FINAL_B)
    requests[0] += 1
    va = await _post_embed(client, base_url, a)
    requests[0] += 1
    vb = await _post_embed(client, base_url, b)
    if va is None or vb is None:
        return "hard_error"
    if _cosine(va, vb) > COSINE_SIMILARITY_THRESHOLD:
        return "silent_truncation"
    return "accepted"


async def _chat_classify(
    client: httpx.AsyncClient, base_url: str, n: int, requests: list[int]
) -> str:
    """Classify length ``n`` against ``/v1/chat/completions`` via a head canary.

    A unique canary is planted at the very start of the prompt and the model is
    asked to repeat the first word. If the canary is absent from the reply the
    head was truncated (``silent_truncation``). A 4xx/5xx is ``hard_error``.

    The canary must be the FIRST token of the canary message, so that
    "the first word of this prompt" is unambiguous; the instruction that asks
    the model to repeat it is sent as a separate preceding message so the
    server's notion of input length tracks only the canary payload.
    """
    base = base_url.rstrip("/")
    body = f"{_CANARY} " + _n_token_prompt(max(n - 1, 0), _FINAL_A)
    requests[0] += 1
    try:
        resp = await client.post(
            f"{base}/v1/chat/completions",
            json={
                "model": "mock",
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Repeat the first word of this prompt exactly."
                        ),
                    },
                    {"role": "user", "content": body},
                ],
            },
        )
    except httpx.HTTPError:
        return "hard_error"
    if resp.status_code != 200:
        return "hard_error"
    try:
        content = resp.json()["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError):
        return "hard_error"
    if _CANARY not in str(content):
        return "silent_truncation"
    return "accepted"


_OUTCOME_TO_CLIFF = {
    "hard_error": CliffBehavior.HARD_ERROR,
    "silent_truncation": CliffBehavior.SILENT_TRUNCATION,
    "accepted": CliffBehavior.ACCEPTED,
}


async def probe_capacity(
    client: httpx.AsyncClient,
    base_url: str,
    endpoint: str,
    ceiling: int = DEFAULT_CEILING,
    backend: Backend = Backend.GENERIC,
) -> CapacityResult:
    """Determine the largest accepted input length and how the server fails.

    Binary-search over input length ``n`` in ``[LO, ceiling]``. Returns a
    :class:`~llmprobe.models.CapacityResult` with the largest ``n`` classified
    ``ACCEPTED`` as ``max_accepted_tokens`` and the outcome of the first
    non-accepted length as ``cliff_behavior``. When every probed length is
    accepted (including ``ceiling``), ``cliff_behavior`` is ``ACCEPTED``.

    When every probed length in ``[LO, ceiling]`` is rejected, no length was
    measured as accepted; ``max_accepted_tokens`` carries a lower-bound value
    that was never probed and ``max_accepted_source`` is set to ``UNKNOWN`` so
    the report does not claim a measurement it does not have.

    ``backend`` is accepted for API symmetry; the request shapes we send are
    stable across the supported backends.
    """
    requests = [0]

    # Can we verify our prompt lengths against the server's own tokenizer? If
    # not, every reported count is a nominal estimate and token_count_exact must
    # be False — we never claim a verified count we could not obtain. vLLM is
    # the exception: it reports its own exact count via usage.prompt_tokens, so
    # we verify against that field instead of /tokenize.
    if backend == Backend.VLLM:
        exact = await _vllm_prompt_tokens_exact(client, base_url, endpoint)
    else:
        exact = await _exact_tokenization_available(client, base_url)

    async def classify(n: int) -> str:
        if endpoint.rstrip("/").endswith("/chat/completions"):
            return await _chat_classify(client, base_url, n, requests)
        return await _embed_classify(client, base_url, n, requests)

    # If the ceiling itself is accepted there is no cliff within range.
    if await classify(ceiling) == "accepted":
        return CapacityResult(
            endpoint=endpoint,
            max_accepted_tokens=ceiling,
            token_count_exact=exact,
            cliff_behavior=CliffBehavior.ACCEPTED,
            probe_requests_used=requests[0],
        )

    lo, hi = LO, ceiling
    while lo <= hi:
        mid = (lo + hi) // 2
        outcome = await classify(mid)
        if outcome == "accepted":
            lo = mid + 1
        else:
            hi = mid - 1

    max_accepted = hi
    cliff_outcome = await classify(max_accepted + 1)
    return CapacityResult(
        endpoint=endpoint,
        max_accepted_tokens=max_accepted,
        max_accepted_source=(
            Provenance.UNKNOWN
            if max_accepted < LO
            else Provenance.MEASURED
        ),
        token_count_exact=exact,
        cliff_behavior=_OUTCOME_TO_CLIFF[cliff_outcome],
        probe_requests_used=requests[0],
    )
