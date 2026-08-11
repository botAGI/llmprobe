# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-08-11

### Added

- **Silent truncation detector** — detects when the tail of a request is silently
  discarded by the server instead of being reported back to the caller.
- **llama.cpp/vLLM/Ollama/generic adapters** — support for multiple backends
  through a shared interface, with a fallback for OpenAI-compatible systems.
- **Capability card with provenance markers** — every report carries a marker
  (`read`, `measured`, `inferred`, `unknown`) showing where a value came from.
- **Exit codes 0/1/2** — explicit completion codes for success, error, and
  undetermined results, so calling scripts can tell the outcomes apart.

### Known limitations

- **Live verification only against vLLM.** Adapter behavior was verified "live"
  only against a real vLLM server (recorded snapshot of responses —
  `tests/fixtures/vllm_metrics_live.txt`).
- **llama.cpp and Ollama are covered by mocks.** The `llamacpp` and `ollama`
  adapters have only been tested through `httpx.MockTransport` with recorded
  fixtures (`tests/fixtures/llamacpp_props.json`,
  `tests/fixtures/ollama_*.json`). They have not yet been checked against a real
  server, so their actual behavior may differ from what is claimed.

## [Unreleased]

### Added

- Initial release preparation — publishing setup and release infrastructure.
