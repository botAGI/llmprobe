"""Fake HTTP server that truncates request input at a configured length.

Run it manually to sanity-check truncation behavior against a real client:

    python -m tests.fake_server --port 8765 --limit 8 --mode refuse
    python -m tests.fake_server --port 8765 --limit 8 --mode silent

It exposes a single ``POST /echo`` endpoint that mirrors the input it
receives. Two modes are supported, chosen at server start:

* ``refuse`` — when the request payload is longer than ``--limit``, respond
  with HTTP 400 and an error body instead of echoing.
* ``silent`` — truncate the payload to ``--limit`` bytes and echo the
  truncated version with HTTP 200, without indicating anything was dropped.

The endpoint accepts any JSON payload; only its ``input`` field is inspected.
"""

from __future__ import annotations

import argparse
import sys

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import uvicorn

RequestModel = dict


class FakeServer:
    """A truncating HTTP server.

    ``limit`` is the maximum number of characters allowed in the ``input``
    field. ``mode`` selects refusal vs silent truncation (see module docstring).
    """

    def __init__(self, limit: int, mode: str) -> None:
        if limit < 0:
            raise ValueError(f"limit must be non-negative, got {limit}")
        if mode not in ("refuse", "silent"):
            raise ValueError(f"unknown mode: {mode!r}")
        self._limit = limit
        self._mode = mode

    def build(self) -> FastAPI:
        app = FastAPI(title="fake-truncating-server")

        @app.post("/echo")
        async def echo(request: Request) -> dict:
            body: RequestModel = await request.json()
            raw = body.get("input", "")
            text = raw if isinstance(raw, str) else str(raw)
            if self._mode == "refuse" and len(text) > self._limit:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "input_length": len(text),
                        "limit": self._limit,
                        "message": "input exceeds length limit",
                    },
                )
            truncated = text[: self._limit]
            return {
                "mode": self._mode,
                "limit": self._limit,
                "input": truncated,
                "input_length": len(truncated),
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
        "--limit",
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    server = FakeServer(args.limit, args.mode)
    uvicorn.run(server.build(), host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
