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

The truncation endpoint accepts any JSON payload; only its ``input`` field is
inspected.
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

    ``truncate_len`` is the maximum number of characters allowed in the
    ``input`` field. ``mode`` selects refusal vs silent truncation (see module
    docstring). ``no_slots`` makes ``/slots`` answer ``501``.
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
        help="maximum input length in characters",
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
