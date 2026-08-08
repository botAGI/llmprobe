"""Scripted mock inference server.

A configurable FastAPI app used by every other test. Provides a factory that
emits a server whose embedding endpoint behaves in one of three ways so that
tests can prove llmprobe's measurement logic detects each case.
"""

from __future__ import annotations

import hashlib

from fastapi import FastAPI, Response

EMBED_DIM = 384


def _split_words(text: str) -> list[str]:
    return text.split()


def _embed(tokens: list[str], limit: int | None) -> list[float]:
    effective = tokens if limit is None else tokens[:limit]
    base = "\x1f".join(effective).encode("utf-8")
    vector: list[float] = []
    for i in range(EMBED_DIM):
        digest = hashlib.sha256(base + str(i).encode("utf-8")).digest()
        value = (int.from_bytes(digest[:8], "big") / 2**64) * 2.0 - 1.0
        vector.append(round(value, 12))
    return vector


def make_mock_server(
    max_tokens: int,
    behavior: str,
    backend: str = "llamacpp",
) -> FastAPI:
    """Build a scripted mock inference server.

    ``behavior`` is one of:

    * ``honest`` — embeddings derived from the FULL input text.
    * ``silent_truncation`` — always HTTP 200, but embedding derived only from
      the first ``max_tokens`` tokens (tail silently dropped).
    * ``hard_error`` — HTTP 500 when input token count exceeds ``max_tokens``.
    """
    if behavior not in ("honest", "silent_truncation", "hard_error"):
        raise ValueError(f"unknown behavior: {behavior!r}")

    app = FastAPI(title="mock-llamaserver")

    @app.get("/props")
    def props() -> dict:
        # llama.cpp shape. Deliberately omits n_batch / n_ubatch.
        return {
            "total_slots": 1,
            "default_generation_settings": {"n_ctx": max_tokens},
            "build_info": {"build": 0, "commit": "mock", "version": "0.0.0"},
        }

    @app.get("/v1/models")
    def models() -> dict:
        return {"object": "list", "data": [{"id": "mock"}]}

    @app.post("/tokenize")
    def tokenize(body: dict) -> dict:
        content = body.get("content", "")
        if isinstance(content, (list, tuple)):
            content = " ".join(str(x) for x in content)
        return {"tokens": _split_words(str(content))}

    @app.post("/v1/embeddings")
    def embeddings(body: dict, response: Response) -> dict:
        raw = body.get("input", "")
        inputs: list[str]
        if isinstance(raw, str):
            inputs = [raw]
        else:
            inputs = [str(x) for x in list(raw)]

        for item in inputs:
            tokens = _split_words(item)
            if behavior == "hard_error" and len(tokens) > max_tokens:
                response.status_code = 500
                return {
                    "error": {
                        "message": "prompt too long",
                        "type": "invalid_request_error",
                        "code": 500,
                    }
                }

        results = []
        for item in inputs:
            tokens = _split_words(item)
            limit = max_tokens if behavior == "silent_truncation" else None
            results.append(
                {
                    "object": "embedding",
                    "embedding": _embed(tokens, limit),
                    "index": len(results),
                }
            )

        return {
            "object": "list",
            "data": results,
            "model": "mock",
            "usage": {
                "prompt_tokens": sum(len(_split_words(x)) for x in inputs),
                "total_tokens": sum(len(_split_words(x)) for x in inputs),
            },
        }

    @app.post("/v1/chat/completions")
    def chat_completions(body: dict) -> dict:
        messages = body.get("messages", [])
        prompt = ""
        for message in reversed(messages):
            content = message.get("content", "")
            if isinstance(content, str):
                prompt = content
                break
        tokens = _split_words(prompt)
        if behavior == "silent_truncation":
            tokens = tokens[:max_tokens]
        reply = tokens[0] if tokens else ""
        return {
            "id": "cmpl-mock",
            "object": "chat.completion",
            "model": "mock",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": reply},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(tokens),
                "completion_tokens": 1,
                "total_tokens": len(tokens) + 1,
            },
        }

    return app
