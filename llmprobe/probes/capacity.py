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

import logging
import math
from collections.abc import Awaitable, Callable

import httpx

from llmprobe.backends.vllm import extract_prompt_tokens
from llmprobe.models import Backend, CapacityResult, CliffBehavior, Provenance
from llmprobe.tokens import MAX_ATTEMPTS, _tokenize

logger = logging.getLogger(__name__)

LO = 16
DEFAULT_CEILING = 32768
COSINE_SIMILARITY_THRESHOLD = 0.9999

# Surface a single, greppable flag when identical embeddings confirm that the
# server silently discarded the differing tail. The value is a measured fact:
# the two prompts were verified to differ only in the final token and the
# server still returned identical vectors.
_TRUNCATION_FLAG = "silent_truncation"

_FILLER = "tok"
_FINAL_A = "llmprobeFinalA"
_FINAL_B = "llmprobeFinalB"
_CANARY = "llmprobeCanary"


async def _n_token_prompt(
    client: httpx.AsyncClient,
    base_url: str,
    n: int,
    final: str,
    timeout: httpx.Timeout,
) -> str:
    """Build a prompt of exactly ``n`` tokens ending in ``final``.

    Starts from ``n-1`` filler words plus ``final`` and verifies the count
    against the live ``/tokenize`` endpoint (via :func:`_tokenize`), adjusting
    the filler count until the server reports exactly ``n`` tokens. This is the
    README's "exact count via /tokenize" contract: the count is whatever the
    server's own tokenizer says, not a guess. When ``/tokenize`` is unavailable
    we fall back to the nominal estimate (``n-1`` fillers + ``final``) so the
    caller can report the length as approximate instead of claiming an exact
    count it could not verify.
    """
    if n <= 1:
        return final
    base = base_url.rstrip("/")
    reps = n - 1
    for _ in range(MAX_ATTEMPTS):
        prompt = " ".join([_FILLER] * reps + [final])
        got = await _tokenize(client, base, prompt, timeout=timeout)
        if got is None or got == n:
            return prompt
        per_rep = got / (reps + 1) if reps + 1 else 1.0
        reps = max(1, round(reps + (n - got) / per_rep))
    return " ".join([_FILLER] * reps + [final])


async def _exact_tokenization_available(
    client: httpx.AsyncClient, base_url: str, timeout: httpx.Timeout
) -> bool:
    """Return ``True`` when we can trust ``_n_token_prompt`` counts as exact.

    The prompts built by :func:`_n_token_prompt` are exactly ``n`` tokens only
    if every marker word it can contain is itself a single token. We verify all
    four markers against the live ``/tokenize`` endpoint in one request; any
    word that tokenizes to more (or fewer) than one token makes the count a
    nominal estimate, for which we must report ``exact=False``.
    """
    probe = " ".join([_FILLER, _FINAL_A, _FINAL_B, _CANARY])
    count = await _tokenize(
        client, base_url.rstrip("/"), probe, timeout=timeout
    )
    return count == 4


async def _vllm_prompt_tokens_exact(
    client: httpx.AsyncClient, base_url: str, endpoint: str, timeout: httpx.Timeout
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
    resp = await client.post(f"{base}{endpoint}", json=body, timeout=timeout)
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
    client: httpx.AsyncClient,
    base_url: str,
    endpoint: str,
    prompt: str,
    timeout: httpx.Timeout,
) -> list[float] | None:
    """POST one embedding to ``endpoint``; return the vector on 200 else ``None``."""
    base = base_url.rstrip("/")
    resp = await client.post(
        f"{base}{endpoint}",
        json={"input": prompt, "model": "embed-mock"},
        timeout=timeout,
    )
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


def _assert_differ_only_in_final_token(a: str, b: str) -> None:
    """Raise ``ValueError`` if ``a`` and ``b`` do not differ only in the final token.

    The two-prompt method (:func:`_embed_classify`) is only meaningful when the
    two prompts are otherwise identical, so an identical rather than merely
    near-identical pair would make the ``silent_truncation`` verdict vacuous.
    We compare the strings before any request is sent: a malformed pair fails
    loudly instead of yielding a misleading classification.
    """
    if a == b:
        raise ValueError("embedding probe prompts are identical (no differing tail)")
    a_tail = a.rsplit(" ", 1)[-1]
    b_tail = b.rsplit(" ", 1)[-1]
    a_head = a[: -len(a_tail)] if a_tail else a
    b_head = b[: -len(b_tail)] if b_tail else b
    if a_head != b_head:
        raise ValueError(
            "embedding probe prompts differ beyond the final token "
            "(two-prompt silent-truncation check is not valid)"
        )


async def _embed_classify(
    client: httpx.AsyncClient,
    base_url: str,
    endpoint: str,
    n: int,
    requests: list[int],
    timeout: httpx.Timeout,
) -> str:
    """Classify length ``n`` against ``/v1/embeddings``.

    Two identical-length prompts differing only in the final token. A 4xx/5xx
    is ``hard_error``; effectively identical vectors are ``silent_truncation``;
    genuinely different vectors are ``accepted``.
    """
    a = await _n_token_prompt(client, base_url, n, _FINAL_A, timeout)
    b = await _n_token_prompt(client, base_url, n, _FINAL_B, timeout)
    _assert_differ_only_in_final_token(a, b)
    requests[0] += 1
    va = await _post_embed(client, base_url, endpoint, a, timeout)
    requests[0] += 1
    vb = await _post_embed(client, base_url, endpoint, b, timeout)
    if va is None or vb is None:
        return "hard_error"
    if _cosine(va, vb) > COSINE_SIMILARITY_THRESHOLD:
        logger.info(
            "%s confirmed: prompt tail silently discarded (a==b at length %d); "
            "differing final token produced identical embeddings",
            _TRUNCATION_FLAG,
            n,
        )
        return "silent_truncation"
    return "accepted"


async def _chat_classify(
    client: httpx.AsyncClient,
    base_url: str,
    n: int,
    requests: list[int],
    timeout: httpx.Timeout,
) -> str:
    """Classify length ``n`` against ``/v1/chat/completions`` via a head canary.

    A unique canary is planted at the very start of the prompt and the model is
    asked to repeat the first word. If the canary is absent from the reply the
    head was truncated (``silent_truncation``). A 4xx/5xx is ``hard_error``.

    The canary must be the FIRST token of the canary payload, so that "the
    first word of this prompt" is unambiguous. The full canary prompt is sent
    as the single user message so the server's notion of input length tracks
    exactly the bytes we probe.
    """
    base = base_url.rstrip("/")
    tail = await _n_token_prompt(
        client, base_url, max(n - 1, 0), _FINAL_A, timeout
    )
    body = f"{_CANARY} " + tail
    requests[0] += 1
    resp = await client.post(
        f"{base}/v1/chat/completions",
        json={
            "model": "chat-mock",
            "messages": [{"role": "user", "content": body}],
        },
        timeout=timeout,
    )
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

Classifier = Callable[[int], Awaitable[str]]


async def _probe_lo(
    classify: Classifier, lo: int, hi: int
) -> tuple[int, int]:
    """Probe a midpoint and, if accepted, raise the lower bound.

    ``mid`` is classified via ``classify`` (which dispatches to
    ``_embed_classify``/``_chat_classify``). When accepted the cliff lies above
    ``mid`` so ``lo`` is raised past it; otherwise ``mid`` becomes the new
    exclusive upper bound. Returns the updated ``(lo, hi)``.
    """
    mid = (lo + hi) // 2
    if await classify(mid) == "accepted":
        return mid + 1, hi
    return lo, mid - 1


async def _probe_hi(classify: Classifier, n: int) -> str:
    """Classify a single candidate length ``n`` above the accepted boundary.

    ``n`` (``max_accepted + 1``) is the first length past the largest accepted
    one — the cliff. Classifying it via ``classify`` (which dispatches to
    ``_embed_classify``/``_chat_classify``) yields the outcome that becomes
    ``cliff_behavior``. Returns the raw outcome string.
    """
    return await classify(n)


async def _binary_search(
    classify,
    endpoint: str,
    ceiling: int,
    exact: bool,
    requests: list[int],
) -> CapacityResult:
    """Binary-search over input length ``n`` in ``[LO, ceiling]``.

    The ``classify`` callable returns ``"accepted"``/``"silent_truncation"``/
    ``"hard_error"`` for a given length and must increment ``requests[0]`` for
    every request it issues. Returns a :class:`~llmprobe.models.CapacityResult`
    with the largest ``n`` classified ``ACCEPTED`` as ``max_accepted_tokens``
    and the outcome of the first non-accepted length as ``cliff_behavior``.
    When every probed length is accepted (including ``ceiling``),
    ``cliff_behavior`` is ``ACCEPTED``.

    When every probed length in ``[LO, ceiling]`` is rejected, no length was
    measured as accepted; ``max_accepted_tokens`` carries a lower-bound value
    that was never probed and ``max_accepted_source`` is set to ``UNKNOWN`` so
    the report does not claim a measurement it does not have.
    """
    # If the ceiling itself is accepted there is no cliff within range.
    if await classify(ceiling) == "accepted":
        return CapacityResult(
            endpoint=endpoint,
            max_accepted_tokens=ceiling,
            token_count_exact=exact,
            cliff_behavior=CliffBehavior.ACCEPTED,
            probe_requests_used=requests[0],
        )

    # Narrow ``[LO, ceiling]`` with a binary search over length ``n``: at each
    # step probe the midpoint and move the accepted boundary accordingly. The
    # search is logarithmic in the ceiling, never linear in the cliff position.
    lo, hi = LO, ceiling
    while lo <= hi:
        lo, hi = await _probe_lo(classify, lo, hi)

    max_accepted = hi
    cliff_outcome = await _probe_hi(classify, max_accepted + 1)
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


async def _binary_search_chat(
    client: httpx.AsyncClient,
    base_url: str,
    endpoint: str,
    ceiling: int,
    exact: bool,
    requests: list[int],
    timeout: httpx.Timeout,
) -> CapacityResult:
    """Binary-search the ``/v1/chat/completions`` cliff.

    Runs the shared binary-search loop against :func:`_chat_classify`, which
    probes each length via a head canary rather than the two-prompt embedding
    method used for the embeddings endpoint.
    """
    async def classify(n: int) -> str:
        return await _chat_classify(client, base_url, n, requests, timeout)

    return await _binary_search(classify, endpoint, ceiling, exact, requests)


async def probe_capacity(
    client: httpx.AsyncClient,
    base_url: str,
    endpoint: str,
    ceiling: int = DEFAULT_CEILING,
    backend: Backend = Backend.GENERIC,
    timeout: float | None = None,
) -> CapacityResult:
    """Determine the largest accepted input length and how the server fails.

    Dispatches to a per-endpoint binary search: a head-canary search over
    ``/v1/chat/completions`` and the two-prompt search over
    ``/v1/embeddings``.

    ``backend`` is accepted for API symmetry; the request shapes we send are
    stable across the supported backends. ``timeout``, when given, bounds every
    HTTP request the probe issues (defaulting to the client's configured
    timeout when omitted).
    """
    per_request = (
        httpx.Timeout(timeout) if timeout is not None else client.timeout
    )
    requests = [0]

    # Can we verify our prompt lengths against the server's own tokenizer? If
    # not, every reported count is a nominal estimate and token_count_exact must
    # be False — we never claim a verified count we could not obtain. vLLM is
    # the exception: it reports its own exact count via usage.prompt_tokens, so
    # we verify against that field instead of /tokenize.
    if backend == Backend.VLLM:
        exact = await _vllm_prompt_tokens_exact(
            client, base_url, endpoint, per_request
        )
    else:
        exact = await _exact_tokenization_available(client, base_url, per_request)

    if endpoint.rstrip("/").endswith("/chat/completions"):
        return await _binary_search_chat(
            client, base_url, endpoint, ceiling, exact, requests, per_request
        )

    async def classify(n: int) -> str:
        return await _embed_classify(
            client, base_url, endpoint, n, requests, per_request
        )

    return await _binary_search(classify, endpoint, ceiling, exact, requests)
