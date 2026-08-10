"""Backend registry.

Adapter probe logic lives in sibling modules (``llamacpp``, ``vllm``,
``ollama``, ``generic``); this package holds only the data shared across the
detection flow. The ``--endpoint auto`` resolution in :mod:`llmprobe.cli`
uses :data:`DEFAULT_PROBE_ENDPOINTS` to map a detected backend onto the probe
path it will actually exercise.
"""

from llmprobe.models import Backend

# Default capacity-probe path per backend. ``--endpoint auto`` resolves to
# this when the backend is detected. The choice reflects the most meaningful
# (and safest) probe target for each backing service: llama.cpp's embedding
# endpoint is the cliff we can measure, while chat is appropriate for vLLM.
DEFAULT_PROBE_ENDPOINTS: dict[Backend, str] = {
    Backend.LLAMACPP: "/v1/embeddings",
    Backend.VLLM: "/v1/chat/completions",
    Backend.OLLAMA: "/v1/chat/completions",
    Backend.GENERIC: "/v1/embeddings",
}

__all__ = ["DEFAULT_PROBE_ENDPOINTS"]
