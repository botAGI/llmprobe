# llmprobe

[![Version](https://img.shields.io/badge/version-0.1.0-blue)](https://pypi.org/project/llmprobe/)

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
three commands take you from install to an automated capacity check. Each
flag below is a real CLI option — verify against `llmprobe --help`.

1. **Install.**

   ```bash
   uvx llmprobe
   ```

2. **Safe run** — read configuration only, send no inference traffic, and get
   a report of what the server advertises and what llmprobe can verify as
   `read`:

   ```bash
   uvx llmprobe http://localhost:8080
   ```

3. **Production run that aborts on a mismatch** — send probe traffic to find
   the real capacity cliff, and exit `1` (failing a CI gate) if the measured
   ceiling is below the context you claimed:

   ```bash
   uvx llmprobe http://localhost:8080 --probe --claimed-ctx 8192
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
    uvx llmprobe http://localhost:8080 --probe --claimed-ctx 8192
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

## Использование как гейта в CI

Чтобы использовать llmprobe как шлюз в пайплайне CI, достаточно запустить его
в шаге workflow и проверить код возврата. llmprobe завершает работу с `0`,
когда измеренная ёмкость совпадает с заявленной, и с `1` при несовпадении.
Любой ненулевой код возврата остановит шаг и, как следствие, весь job.

Ниже — пример шага GitHub Actions workflow, который запускает llmprobe против
локального inference-сервера с `--json` и проверяет код возврата. `--json`
выдаёт машинночитаемый отчёт на stdout (его можно сохранить как артефакт или
разобрать в следующем шаге), а решение о том, пройден ли шлюз, принимается по
коду возврата — это единственный стабильный для CI контракт. Передача
`--claimed-ctx` делает проверку осмысленной: без неё измеренная ёмкость
ни с чем не сравнивается, и шлюз не сможет обнаружить несоответствие.

```yaml
- name: Check inference server capacity (CI gate)
  run: |
    uvx llmprobe http://localhost:8080 --json --probe --claimed-ctx 8192 \
      > llmprobe-report.json
    exit_code=$?
    # llmprobe exits 0 when measured capacity matches the claim,
    # 1 when it does not, and 2 on error. Preserve the report artifact
    # and fail the step if the exit code is non-zero.
    if [ "$exit_code" -ne 0 ]; then
      echo "llmprobe gate failed with exit code $exit_code"
      exit "$exit_code"
    fi
```

Проверьте каждый флаг против `llmprobe --help`, прежде чем включать шаг в
workflow. Все права на описанное здесь поведение проверяемы командами:
`llmprobe http://localhost:8080` возвращает `0` на чистом сервере, `1` при
несовпадении ёмкости и `2` при ошибке (недоступный сервер, транспортный сбой).
Отчёт в `llmprobe-report.json` содержит провенанс на каждом значении и
предназначен для инспекции; полагайтесь не на разбор JSON, а на код возврата,
который и является контрактом для CI.

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

## Examples of output

Below are examples of real tool output. Every report consists of a `Capability
Report` table, where for each property the claimed (Claimed) and measured
(Measured) values, the value source (Source), and the verdict (Verdict) are given.

Example report for a server with a correct configuration (`tests/golden/clean.md`):

```text
# Capability Report — http://localhost:8080

| Property | Claimed | Measured | Source | Verdict |
| --- | --- | --- | --- | --- |
| backend | llamacpp | llamacpp | read | ok |
| model | mock/llama-3.1-8b | mock/llama-3.1-8b | read | ok |
| context (total) | 8192 | 8192 | read | ok |
| context (per slot) | 2048 | 2048 | read | ok |
| slots | 4 | 4 | read | ok |
| max input tokens (/completion) | unknown | 8192 | measured | ok |
| cliff behaviour (/completion) | unknown | accepted | measured | ok |
```

Example report that reveals a problem — the measured ceiling is below the
claimed one, and excess tokens are silently discarded (`tests/golden/silent-truncation.md`):

```text
# Capability Report — http://localhost:8080

| Property | Claimed | Measured | Source | Verdict |
| --- | --- | --- | --- | --- |
| backend | llamacpp | llamacpp | read | ok |
| model | mock/llama-3.1-8b | mock/llama-3.1-8b | read | ok |
| context (total) | 8192 | 8192 | read | ok |
| context (per slot) | 2048 | 2048 | read | ok |
| slots | 4 | 4 | read | ok |
| max input tokens (/completion) | unknown | 7168 | measured | ok |
| cliff behaviour (/completion) | unknown | silent_truncation | measured | truncated |

## Findings

- **[mismatch] UBATCH_CEILING**: advertised=8192 vs measured=7168 — requests past 7168 tokens are silently truncated

## Fix

--batch-size 8192 --ubatch-size 8192
```

In the second example, the distinction between the `ok` and `truncated`
verdicts is important: it is `truncated` that means the server accepted the
request but dropped the tail of the context without reporting an error.

## Tool limitations

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

## Status

Early. v0 covers config read, capacity cliff, and per-slot context.

## License

MIT

