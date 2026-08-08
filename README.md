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
uvx llmprobe http://localhost:8080 --json      # emit machine-readable JSON
uvx llmprobe http://localhost:8080 --endpoint chat   # probe the chat endpoint (default auto)
uvx llmprobe http://localhost:8080 --timeout 30      # per-request timeout in seconds
```

Exit codes: `0` clean, `1` advertised capacity does not match measured, `2` error.
That makes it usable as a CI gate, not just a one-off report.

By default llmprobe reads configuration only (`--safe`); pass `--probe` to send
inference traffic. `--endpoint` selects which endpoint the capacity probe
exercises (`embeddings`, `chat`, or `auto`, the default that resolves per
backend). Every HTTP request carries a per-request timeout (default 10s,
overridable with `--timeout`) so an unresponsive server fails fast instead of
hanging the process.

## Error handling and security

- **Unreachable server**: llmprobe verifies the server is reachable before
  adapter detection. A transport failure is reported to stderr and the process
  exits `2` — it never pretends an empty "generic" match is a healthy server.
- **Failing backends**: a backend that cannot be probed is treated as a non-match
  rather than aborting detection; one working adapter always wins.
- **Timeouts**: `--timeout` bounds every HTTP request. This keeps a hung or
  slowly-draining server from stalling a CI step indefinitely.
- **Provenance on every value**: `read`, `measured`, `inferred`, or `unknown`.
  A confident guess is worse than an honest "unknown", and no value is
  fabricated to make a report look complete.

## Known limitations

Be clear-eyed about what a run can and cannot tell you:

- **Generic OpenAI-compatible servers** advertise no reliable capacity over
  HTTP, so llmprobe reports every capacity field as `unknown` and only a model
  id with provenance `read`. That is not a bug — it is the honest answer the
  protocol permits. Run `--probe` if you need a measured number.
- **llama.cpp does not expose `n_batch` or `n_ubatch` over HTTP**, so both are
  reported as `unknown` (provenance `unknown`). They cannot be read; they can
  only be probed.
- **Backend detection is heuristic.** It keys on server-specific response
  shapes (e.g. llama.cpp's `/props` `build_info`). A server that imitates one
  of those shapes may be classified as that backend.
- **The capacity probe measures input length, not throughput.** It reports the
  largest *input* a server genuinely accepts and how it fails beyond that —
  not latency, tokens/sec, or concurrency.
- **Model-facing requests in the probe are mocks.** The probe posts requests
  for a model id like `embed-mock` / `mock`; a server that validates the model
  id strictly may reject them and be reported as a hard error.

## Troubleshooting live runs

- **Exit code `2` with an "unreachable or failed server" message**: llmprobe
  could not reach the server at all (transport failure) or a request raised an
  HTTP error. Check that the server is up, reachable from this host, and that
  the port is open before re-running.
- **The process hangs on a dead server**: it should not. Every request carries
  a per-request timeout (default 10s, `--timeout` to override). If a run still
  appears stuck, the server is accepting connections but never responding.
- **Capacity fields come back `unknown`**: you are talking to an
  OpenAI-compatible or llama.cpp server where the value is not exposed over
  HTTP. Use `--probe` to measure the real cliff instead of reading config.
- **A specific backend is reported as generic**: detection found no specific
  signature. Confirm the server is running its native API (not a proxy that
  reshapes `/props`, `/v1/models`, or similar endpoints llmprobe inspects).
- **A probe reports a hard error at the very first length**: the endpoint
  rejected the probe request — often a strict model-id check (see known
  limitations) or a server that requires authentication on that endpoint.

When diagnosing, prefer `--json` (machine-readable report with provenance on
every value) over eyeballing the markdown card.

## Supported backends

llama.cpp (`llama-server`), vLLM, Ollama, and a generic OpenAI-compatible fallback.

## Status

Early. v0 covers config read, capacity cliff, and per-slot context.

## License

MIT
