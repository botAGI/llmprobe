"""Scripted mock inference server.

A configurable FastAPI app used by every other test. Provides a factory that
emits a server whose embedding endpoint behaves in one of three ways so that
tests can prove llmprobe's measurement logic detects each case.
"""

from __future__ import annotations

import hashlib

from fastapi import FastAPI, HTTPException, Request, Response

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
    required_token: str | None = None,
    tokenize_enabled: bool = True,
) -> FastAPI:
    """Build a scripted mock inference server.

    ``behavior`` is one of:

    * ``honest`` — embeddings derived from the FULL input text.
    * ``silent_truncation`` — always HTTP 200, but embedding derived only from
      the first ``max_tokens`` tokens (tail silently dropped).
    * ``hard_error`` — HTTP 500 when input token count exceeds ``max_tokens``.

    When ``required_token`` is set, every request must carry
    ``Authorization: Bearer <required_token>``; anything missing or wrong
    returns 401. Detection and config-read endpoints are gated too, so an
    unauthenticated probe cannot silently fall back to an empty config.

    When ``tokenize_enabled`` is ``False``, the ``/tokenize`` endpoint returns
    404 so a probe cannot verify exact token counts and must report the count as
    an estimate (``token_count_exact=False``).
    """
    if behavior not in ("honest", "silent_truncation", "hard_error"):
        raise ValueError(f"unknown behavior: {behavior!r}")

    app = FastAPI(title="mock-llamaserver")

    def _authorized(request: Request) -> bool:
        return request.headers.get("Authorization") == f"Bearer {required_token}"

    def _guard(request: Request, response: Response) -> bool:
        if not required_token:
            return True
        if not _authorized(request):
            response.status_code = 401
            return False
        return True

    @app.get("/props")
    def props(request: Request, response: Response) -> dict:
        if not _guard(request, response):
            return {
                "error": {
                    "message": "unauthorized",
                    "type": "authentication_error",
                }
            }
        # llama.cpp shape. Deliberately omits n_batch / n_ubatch.
        return {
            "total_slots": 1,
            "default_generation_settings": {"n_ctx": max_tokens},
            "build_info": {"build": 0, "commit": "mock", "version": "0.0.0"},
        }

    @app.get("/v1/models")
    def models(request: Request, response: Response) -> dict:
        if not _guard(request, response):
            return {
                "object": "list",
                "data": [],
            }
        return {"object": "list", "data": [{"id": "mock"}]}

    @app.post("/tokenize")
    def tokenize(request: Request, response: Response, body: dict) -> dict:
        if not tokenize_enabled:
            raise HTTPException(status_code=404, detail="Not Found")
        if not _guard(request, response):
            return {"tokens": []}
        content = body.get("content", "")
        if isinstance(content, (list, tuple)):
            content = " ".join(str(x) for x in content)
        return {"tokens": _split_words(str(content))}

    @app.post("/v1/embeddings")
    def embeddings(request: Request, response: Response, body: dict) -> dict:
        if not _guard(request, response):
            return {
                "object": "list",
                "data": [],
                "usage": {"prompt_tokens": 0, "total_tokens": 0},
            }
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
    def chat_completions(request: Request, response: Response, body: dict) -> dict:
        if not _guard(request, response):
            return {
                "id": "cmpl-mock",
                "object": "chat.completion",
                "model": "mock",
                "choices": [],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
        messages = body.get("messages", [])
        prompt = ""
        for message in reversed(messages):
            content = message.get("content", "")
            if isinstance(content, str):
                prompt = content
                break
        tokens = _split_words(prompt)
        # An honest server's reply reflects the full prompt (so the differing
        # tail token produces a differing reply). A silently truncating server
        # drops the tail beyond ``max_tokens`` and replies from the retained
        # head only, so two prompts that differ only in a dropped tail token
        # come back identical — the signal the two-prompt method detects. An
        # oversized prompt is a hard error.
        if behavior == "hard_error" and len(tokens) > max_tokens:
            response.status_code = 500
            return {
                "error": {
                    "message": "prompt too long",
                    "type": "invalid_request_error",
                    "code": 500,
                }
            }
        if behavior == "silent_truncation" and len(tokens) > max_tokens:
            # Only the first ``max_tokens`` tokens survive; the reply draws on
            # the retained head, never the silently-dropped tail.
            reply = tokens[:max_tokens][0] if tokens and max_tokens > 0 else ""
        else:
            reply = prompt
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
