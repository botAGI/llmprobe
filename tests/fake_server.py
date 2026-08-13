"""Fake HTTP server that truncates request input at a configured length.

Run it manually to sanity-check truncation behavior against a real client:

    python -m tests.fake_server --port 8765 --truncate-len 8 --mode refuse
    python -m tests.fake_server --port 8765 --truncate-len 8 --mode silent

It exposes a ``POST /echo`` endpoint that mirrors the input it receives. Two
modes are supported, chosen at server start:

* ``refuse`` — when the request payload is longer than ``--truncate-len``,
  respond with HTTP 400 and an error body instead of echoing.
* ``silent`` — truncate the payload to ``--truncate-len`` characters and echo
  the truncated version with HTTP 200, without indicating anything was dropped.

It also serves a llama.cpp-shaped compatibility surface so a probe can observe
server-declared limits and per-slot state:

* ``GET /props`` — server properties; deliberately omits ``n_ubatch`` (and
  ``n_batch``) so a probe cannot rely on them being present.
* ``GET /slots`` — per-slot state as a list. With ``--no-slots`` the endpoint
  answers ``501`` (slots disabled), exactly like a real llama.cpp server run
  without slots.
* ``GET /v1/models`` — the advertised model id, so a probe can resolve the
  model it should request.
* ``POST /tokenize`` — counts space-separated words as tokens, matching the
  capacity probe's exact-token-count contract.
* ``POST /v1/chat/completions`` — silently truncates an oversized prompt to
  ``--truncate-len`` tokens, dropping the head (a real silent truncation a
  capacity probe must catch).

``--truncate-len`` is expressed in tokens for the chat endpoint (each
space-separated word counts as one token). The ``/echo`` endpoint truncates by
characters as documented; both share the same limit value so the manual
``--mode silent`` sanity check and the probed chat endpoint truncate at the
same configured boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
import uvicorn

RequestModel = dict

DEFAULT_MAX_TOKENS = 4096

# The model advertised to a probe. Kept parallel to DEFAULT_MAX_TOKENS so the
# server presents a coherent (if fictional) llama.cpp identity.
MODEL_ID = "fake-4096"


def _count_tokens(text: str) -> int:
    """Return the number of whitespace-separated words in ``text``.

    The capacity probe builds prompts from single-word tokens (filler words, a
    canary marker) and verifies counts against ``/tokenize``, so a word-based
    count keeps this fake exactly consistent with the probe's own model: each
    word a probe emits counts as exactly one token.
    """
    return len(text.split())

EMBED_DIM = 384


def _split_words(text: str) -> list[str]:
    return text.split()


def _embed(tokens: list[str]) -> list[float]:
    """A deterministic embedding derived from the FULL token list.

    Any change to the input — including a single differing tail token — changes
    the vector, so two prompts that differ only in their final token produce
    distinct embeddings. This is the honest counterpart to a silently
    truncating server, which would return identical vectors regardless of the
    dropped tail.
    """
    base = "\x1f".join(tokens).encode("utf-8")
    vector: list[float] = []
    for i in range(EMBED_DIM):
        digest = hashlib.sha256(base + str(i).encode("utf-8")).digest()
        value = (int.from_bytes(digest[:8], "big") / 2**64) * 2.0 - 1.0
        vector.append(round(value, 12))
    return vector


def _input_items(body: RequestModel) -> list[str]:
    raw = body.get("input", "")
    if isinstance(raw, str):
        return [raw]
    return [str(x) for x in list(raw)]


class FakeServer:
    """A truncating HTTP server with a llama.cpp-shaped compatibility surface.

    ``truncate_len`` is the maximum number of tokens (space-separated words)
    allowed before the probed endpoints (``/v1/chat/completions`` and
    ``/v1/embeddings``) discard the excess — for ``silent`` mode the tail is
    kept and the head silently dropped, for ``refuse`` the block below applies.
    ``no_slots`` makes ``/slots`` answer ``501``.
    """

    def __init__(
        self,
        truncate_len: int,
        mode: str,
        no_slots: bool = False,
        total_slots: int = 1,
    ) -> None:
        if truncate_len < 0:
            raise ValueError(
                f"truncate_len must be non-negative, got {truncate_len}"
            )
        if mode not in ("refuse", "silent"):
            raise ValueError(f"unknown mode: {mode!r}")
        if total_slots <= 0:
            raise ValueError(f"total_slots must be positive, got {total_slots}")
        self._truncate_len = truncate_len
        self._mode = mode
        self._no_slots = no_slots
        self._total_slots = total_slots

    def build(self) -> FastAPI:
        app = FastAPI(title="fake-truncating-server")

        @app.post("/echo")
        async def echo(request: Request) -> dict:
            body: RequestModel = await request.json()
            raw = body.get("input", "")
            text = raw if isinstance(raw, str) else str(raw)
            if self._mode == "refuse" and len(text) > self._truncate_len:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "input_length": len(text),
                        "truncate_len": self._truncate_len,
                        "message": "input exceeds length limit",
                    },
                )
            truncated = text[: self._truncate_len]
            return {
                "mode": self._mode,
                "truncate_len": self._truncate_len,
                "input": truncated,
                "input_length": len(truncated),
            }

        @app.get("/props")
        def props() -> dict:
            # llama.cpp shape. Deliberately omits n_batch / n_ubatch.
            return {
                "model": MODEL_ID,
                "total_slots": self._total_slots,
                "default_generation_settings": {
                    "n_ctx": DEFAULT_MAX_TOKENS,
                },
                "build_info": {
                    "build": 0,
                    "commit": "fake",
                    "version": "0.0.0",
                },
            }

        @app.get("/slots")
        def slots(response: Response) -> list | dict:
            if self._no_slots:
                # --no-slots mode: a real llama.cpp server answers 501 here.
                response.status_code = 501
                return {
                    "error": {
                        "message": "slots disabled",
                        "type": "server_error",
                        "code": 501,
                    }
                }
            return [
                {
                    "id": index,
                    "state": 1,
                    "n_ctx": DEFAULT_MAX_TOKENS,
                    "is_processing": False,
                }
                for index in range(self._total_slots)
            ]

        @app.get("/v1/models")
        def models() -> dict:
            return {"object": "list", "data": [{"id": MODEL_ID}]}

        @app.post("/tokenize")
        def tokenize(body: dict) -> dict:
            content = body.get("content", "")
            if isinstance(content, (list, tuple)):
                content = " ".join(str(x) for x in content)
            return {"tokens": _split_words(str(content))}

        @app.post("/v1/embeddings")
        def embeddings(body: dict) -> dict:
            results = []
            for item in _input_items(body):
                tokens = _split_words(item)
                # Silent truncation: derivation uses only the first
                # ``truncate_len`` tokens. Two prompts that differ only in their
                # final token therefore collide once that token is dropped.
                effective = tokens[: self._truncate_len] if self._mode == "silent" else tokens
                results.append(
                    {
                        "object": "embedding",
                        "embedding": _embed(effective),
                        "index": len(results),
                    }
                )
            return {
                "object": "list",
                "data": results,
                "model": MODEL_ID,
                "usage": {
                    "prompt_tokens": sum(_count_tokens(x) for x in _input_items(body)),
                    "total_tokens": sum(_count_tokens(x) for x in _input_items(body)),
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
            # Silent truncation keeps only the LAST ``truncate_len`` tokens, so
            # the head — which carries the probe's canary marker — is dropped on
            # an oversized prompt and the reply echoes a retained filler instead.
            retained = (
                tokens[-self._truncate_len :] if self._mode == "silent" else tokens
            )
            reply = retained[0] if retained else ""
            return {
                "id": "chatcmpl-fake",
                "object": "chat.completion",
                "model": MODEL_ID,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": reply},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": _count_tokens(prompt),
                    "completion_tokens": 1,
                    "total_tokens": _count_tokens(prompt) + 1,
                },
            }

        return app


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="fake_server",
        description="HTTP server that truncates input at a configured length.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="bind host")
    parser.add_argument("--port", type=int, default=8765, help="bind port")
    parser.add_argument(
        "--truncate-len",
        type=int,
        required=True,
        help="maximum token (word) count preserved by the probed endpoints",
    )
    parser.add_argument(
        "--mode",
        choices=("refuse", "silent"),
        required=True,
        help="truncation behavior",
    )
    parser.add_argument(
        "--total-slots",
        type=int,
        default=1,
        help="number of llama.cpp slots to advertise in /props",
    )
    parser.add_argument(
        "--no-slots",
        action="store_true",
        help="make /slots answer 501 (slots disabled)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    server = FakeServer(
        args.truncate_len,
        args.mode,
        no_slots=args.no_slots,
        total_slots=args.total_slots,
    )
    uvicorn.run(server.build(), host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
