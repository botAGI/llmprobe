# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Initial release preparation** — first public release of llmprobe, a CLI
  that measures what a running local inference server can actually do rather
  than what it claims.
- **Capacity probing** — binary search to the real maximum input for each
  endpoint, and detection of silent truncation where the tail of a prompt is
  silently discarded.
- **Per-slot context reporting** — effective per-slot context derived from the
  actual server config, not from the values you passed (a missing flag is a
  default, not `1`).
- **Multiple backends** — support for llama.cpp (`llama-server`), vLLM, Ollama,
  and a generic OpenAI-compatible fallback.
- **Provenance on every value** — each reported value carries a marker
  (`read`, `measured`, `inferred`, or `unknown`) so you can tell what the
  server told us from what we probed.
