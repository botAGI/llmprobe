# llmprobe

**Point it at a running local inference server. It tells you what that server can actually do — measured, not claimed.**

Your `llama-server` says it serves 8192 tokens of context. Your embedding endpoint
accepts a 2000-token document and returns HTTP 200. Both statements can be true while
the last 1500 tokens were silently thrown away.

That is not hypothetical. On llama.cpp master today, enabling embeddings makes the
server clamp `n_batch` down to `n_ubatch` — whose default is **512** — and print a
warning nobody reads ([`tools/server/server.cpp:145-149`](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/server.cpp)).
A pooled embedding model must fit the whole sequence in one physical batch, so
everything above 512 tokens either errors or gets truncated. And `n_ubatch` is not
exposed over HTTP at all: it is absent from `/props`. You cannot read it. You can only
measure it.

## What it checks

- **Real maximum input** — binary search to the actual cliff, per endpoint
  (`/v1/embeddings` and `/v1/chat/completions` fail at different limits).
- **Silent truncation** — the dangerous case. Two prompts differing only in their last
  token: if the embeddings come back identical, the tail was discarded.
- **Effective config vs. what you passed** — an absent flag is a *default*, not `1`
  (llama.cpp's `--parallel` defaults to 4, so your per-slot context is `n_ctx / 4`).
- **Provenance on every value** — `read`, `measured`, `inferred`, or `unknown`.
  A confident guess is worse than an honest "unknown".

## Usage

```bash
uvx llmprobe http://localhost:8080            # safe: read config only
uvx llmprobe http://localhost:8080 --probe    # send traffic, find the real cliff
uvx llmprobe http://localhost:8080 --probe --claimed-ctx 8192   # exit 1 on mismatch
```

Exit codes: `0` clean, `1` advertised capacity does not match measured, `2` error.
That makes it usable as a CI gate, not just a one-off report.

## Supported backends

llama.cpp (`llama-server`), vLLM, Ollama, and a generic OpenAI-compatible fallback.

## Status

Early. v0 covers config read, capacity cliff, and per-slot context.

## License

MIT
