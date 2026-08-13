# llmprobe

[![Version](https://img.shields.io/badge/version-0.1.0-blue)](https://github.com/botAGI/llmprobe/releases)

[Русская версия](README.ru.md)

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

Where the server does tell you, llmprobe reads it rather than guessing: on
`/completion` llama.cpp returns `truncated` and `tokens_evaluated`, and those
are used directly. The blind spot this tool exists for is the embeddings path,
where no such field exists and the batch ceiling is invisible over HTTP.

## What it checks

- **Real maximum input** — binary search to the actual cliff, per endpoint
  (`/v1/embeddings` and `/v1/chat/completions` fail at different limits).
- **Silent truncation** — the dangerous case. Two prompts differing only in their last
  token: if the embeddings come back identical, the tail was discarded.
- **Effective config vs. what you passed** — an absent flag is a *default*, not `1`
  (llama.cpp's `--parallel` defaults to 4, so your per-slot context is `n_ctx / 4`).
- **Provenance on every value** — `read`, `measured`, `inferred`, or `unknown`.
  A confident guess is worse than an honest "unknown".

## Examples of output

Both reports below are verbatim output against two real `llama.cpp` b9049
servers, started from the same image with the same model and differing in
exactly one flag. `scripts/live_control.sh` reproduces the pair, and
`tests/test_live_llamacpp.py` asserts both outcomes.

**A server capped below the context it advertises.** Started with
`--embeddings --ctx-size 8192` and nothing else, so `n_ubatch` keeps its
default of 512. `/props` reports 8192 tokens per slot and says nothing about
`n_ubatch`; the server hard-errors past 510 tokens (510 of content plus BOS and
EOS fills the 512 batch). No `--claimed-ctx` was passed — the shortfall is
measured against the server's own reported context:

```text
# Capability Report — http://127.0.0.1:18081

llmprobe 0.1.0 · measured 2026-08-13T15:23:17Z (UTC)

| Property | Claimed | Measured | Source | Verdict |
| --- | --- | --- | --- | --- |
| backend | llamacpp | llamacpp | read | ok |
| model | unknown | unknown | unknown | unknown |
| context (total) | unknown | unknown | unknown | unknown |
| context (per slot) | 8192 | 8192 | read | ok |
| slots | 4 | 4 | read | ok |
| max input tokens (/v1/embeddings) | unknown | 510 | measured | ok |
| token count (/v1/embeddings) | unknown | estimate | measured | ok |
| cliff behaviour (/v1/embeddings) | unknown | hard_error | measured | error |
| probe requests used (/v1/embeddings) | unknown | 34 | measured | ok |
| requests spent | unknown | 34 | measured | ok |
| measured at | unknown | 2026-08-13T15:23:17Z | measured | ok |

## Findings

- **[mismatch] UBATCH_CEILING**: advertised=8192 (read) vs measured=510 (measured) — requests past 510 tokens are hard\_error

## Fix

--batch-size 8192 --ubatch-size 8192
```

Exit code `1`.

**The same image, the same model, with `-b 8192 -ub 8192`.** Capacity now
matches the configured context, and the two-token gap is BOS and EOS:

```text
# Capability Report — http://127.0.0.1:18082

llmprobe 0.1.0 · measured 2026-08-13T15:26:43Z (UTC)

| Property | Claimed | Measured | Source | Verdict |
| --- | --- | --- | --- | --- |
| backend | llamacpp | llamacpp | read | ok |
| model | unknown | unknown | unknown | unknown |
| context (total) | unknown | unknown | unknown | unknown |
| context (per slot) | 8192 | 8192 | read | ok |
| slots | 4 | 4 | read | ok |
| max input tokens (/v1/embeddings) | unknown | 8190 | measured | ok |
| token count (/v1/embeddings) | unknown | estimate | measured | ok |
| cliff behaviour (/v1/embeddings) | unknown | hard_error | measured | error |
| probe requests used (/v1/embeddings) | unknown | 34 | measured | ok |
| requests spent | unknown | 34 | measured | ok |
| measured at | unknown | 2026-08-13T15:26:43Z | measured | ok |
```

Exit code `0`, no findings. The pair is what makes the detector falsifiable: a
tool that flagged everything would pass the first server and fail the second.

## What has been verified live, and what has not

At the brand of "measured, not claimed", this table is mandatory rather than
modest.

| Backend | Config read | Capacity probe | How |
| --- | --- | --- | --- |
| llama.cpp | live | **live, both directions** | b9049, real GGUF, the pair above |
| vLLM | live | live | a running server, 1M-context model |
| Ollama | mock only | mock only | recorded responses, no live run yet |
| generic OpenAI-compatible | mock only | mock only | recorded responses |

Silent truncation is the dangerous case, and llama.cpp does not exhibit it on
embeddings — it hard-errors, which is the honest failure. The detector's
silent-truncation path is therefore exercised against mocks and against
servers that truncate; treat that path as less proven than the hard-error one.

## Install

Not on PyPI yet, so install it from the repository:

```bash
uvx --from git+https://github.com/botAGI/llmprobe llmprobe http://localhost:8080
```

or, to get the `llmprobe` command on your PATH:

```bash
pip install git+https://github.com/botAGI/llmprobe
```

Every example below uses the installed `llmprobe` command.

## Usage

```bash
llmprobe http://localhost:8080            # safe: read config only
llmprobe http://localhost:8080 --probe    # send traffic, find the real cliff
llmprobe http://localhost:8080 --probe --claimed-ctx 8192   # exit 1 on mismatch
llmprobe http://localhost:8080 --json      # emit machine-readable JSON
llmprobe http://localhost:8080 --endpoint chat   # probe the chat endpoint (default auto)
llmprobe http://localhost:8080 --timeout 30      # per-request timeout in seconds
```

Exit codes: `0` clean, `1` advertised capacity does not match measured, `2` error.
That makes it usable as a CI gate, not just a one-off report.

Inference is off by default: llmprobe reads configuration only. This is the
`safe` mode. `--probe` sends inference traffic to find the real capacity cliff;
`--safe` is the default state of the `--probe/--safe` pair, so you never need
to pass it explicitly. Selecting an explicit endpoint with `--endpoint choice`
(`chat` or `embeddings`) also enables inference on that endpoint, even without
`--probe` — naming the endpoint you want to measure implies you want it probed.
Only the default `auto` selection respects `--safe` suppression. `--endpoint`
chooses which endpoint the capacity probe exercises (`embeddings`, `chat`, or
`auto`, the default that resolves per backend). Every HTTP request carries a
per-request timeout (default 10s, overridable with `--timeout`) so an
unresponsive server fails fast instead of hanging the process.

## Quick start

Point it at a running server and read what the server actually exposes. These
two commands take you from a first look to an automated capacity check. Each
flag below is a real CLI option — verify against `llmprobe --help`.

1. **Safe run** — read configuration only, send no inference traffic, and get
   a report of what the server advertises and what llmprobe can verify as
   `read`:

   ```bash
   llmprobe http://localhost:8080
   ```

2. **Production run that aborts on a mismatch** — send probe traffic to find
   the real capacity cliff, and exit `1` (failing a CI gate) if the measured
   ceiling is below the context you claimed:

   ```bash
   llmprobe http://localhost:8080 --probe --claimed-ctx 8192
   ```

   Exit code `1` means "advertised capacity does not match measured"; `0` is
   clean. Use this as a gate to stop a deploy that would silently truncate
   prompts beyond the server's real limit.

## Using as a CI gate

llmprobe returns a process exit code from the severity of its findings, which
is what makes it usable as a gate in a CI pipeline rather than just a
sidecar report. The codes are stable and documented:

| Code | Meaning |
| --- | --- |
| `0` | clean — no mismatch found |
| `1` | advertised capacity does not match measured (a `--claimed-ctx` mismatch surfaced as a `MISMATCH` finding) |
| `2` | error — server unreachable, transport failure, or a request raised an HTTP error |

A gate step runs llmprobe against your server, and the job fails if the exit
code is non-zero. Below is a GitHub Actions workflow step that runs the probe
against a local inference server and checks the return code explicitly. It
passes `--claimed-ctx` so that a mismatch (measured ceiling below what you
advertise) aborts the pipeline instead of shipping a server that silently
truncates prompts.

```yaml
- name: Probe inference server capacity (CI gate)
  run: |
    llmprobe http://localhost:8080 --probe --claimed-ctx 8192
  # llmprobe exits 0 when measured capacity matches the claim and
  # 1 when it does not. Any non-zero exit fails this step.
```

A few notes on making the gate honest and non-flaky:

- **Pass `--claimed-ctx`.** Without it llmprobe still probes, but nothing is
  ever compared, so the gate can never detect a mismatch. The exit code only
  becomes meaningful as a gate when you state the context you actually serve.
- **Set `--timeout`.** Every HTTP request already carries a per-request
  timeout (default 10s) so a hung server fails the step fast instead of
  stalling the job until CI's own timeout kills it.
- **Treat exit code `2` as a hard stop, not a retry.** It means the server was
  unreachable or rejected the probe — gating on the unknown rather than
  guessing success is the honest behaviour.
- **Do not rely on `--json` output parsing for gating.** The JSON report is
  for humans and tooling to inspect; the exit code is the single stable, CI-
  relevant contract.

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

## How llmprobe counts tokens

The token counts that drive the capacity probe are never invented. llmprobe
uses a server-side source of truth when one is available and clearly labels an
approximation when it is not:

- **Exact count via `/tokenize` (llama.cpp).** llama.cpp serves
  `POST /tokenize`, which returns the token sequence for a given string.
  llmprobe builds a probe prompt, re-tokenizes it against the live server, and
  adjusts it until the server reports exactly the length we asked for. The
  count is whatever the server's own tokenizer says — not a guess.
- **Exact count via `usage.prompt_tokens` (vLLM).** vLLM reports
  `usage.prompt_tokens` on its chat and embeddings responses; llmprobe reads
  the number the server itself reported.
- **Estimate when neither source is available.** If `/tokenize` is absent or
  unreadable and the response carries no usable `usage.prompt_tokens`, llmprobe
  repeats a calibrated single-token filler and treats the length as an
  approximation. The result is marked `estimated` so nobody mistakes a guess
  for a measured value.

The distinction exists for the same reason provenance exists everywhere in
this tool: a length confirmed by the server's own tokenizer is a fact, a
length built from guesswork filler is a model, and a report must always let
you tell the two apart.

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

## Limitations

The tool only measures the server's capacity and configuration. It does **not**
and cannot do the following:

- **Does not support streaming response generation.** The behaviour of
  endpoints is measured, not streaming protocols (`text/event-stream`);
  generation `tokens/s` throughput and first-token latency are not measured.
- **Does not measure response quality.** The tool does not assess the
  correctness, meaningfulness, or usefulness of generated text — only whether
  the server accepted the request and how much context it actually processed.
- **Capacity/slots metrics only.** The report is limited to properties such as
  context size, number of slots, and maximum number of input tokens. Other
  characteristics (throughput, stability under load, behaviour of different
  models) remain out of scope.
- **The `ok`/`truncated`/`mismatch` verdicts refer to capacity, not to the
  model's "correctness".** A clean configuration (`ok`) does not guarantee
  sensible answers — it only means there is no silent context loss.

"Everything `ok`" should not be read as "the server works well". It means only
this: the claimed capacity was confirmed by measurement.

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
- **The probe sends the model id it read from the server.** When the server
  reports no model id, the literal `unknown` goes out instead, and a server
  that validates model ids strictly will reject that and be reported as a hard
  error. Pinning a wrong id here once produced a confident "maximum input: 15
  tokens" from what was really an HTTP 404 on every request.

## Status

Early, and honest about it. Config read, capacity cliff and per-slot context
work; the llama.cpp path is verified against real servers in both directions,
vLLM against a running server, and Ollama only against recorded responses. Not
published to PyPI yet — install from the repository as shown above.

## License

MIT

