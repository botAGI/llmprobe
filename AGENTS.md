# Rules for agents working in this repo

## Scope
- Touch ONLY the files your task names. Nothing else, ever.
- Do NOT create files the task did not ask for.
- Do NOT commit runtime state, caches, editor files, `CLAUDE.md`, or `.runtime/`.
- One task = one focused commit. No "chore: runtime cruft" commits.

## Definition of done
A task is done only when its acceptance test passes:
```
python -m pytest <the test file named in the task> -q
```
If the test does not pass, the task is not done. Do not claim otherwise.

## Code style
- Python 3.11+, type hints on public functions.
- `httpx` for HTTP (async), `pydantic` v2 for models, `typer` for CLI, `rich` for output.
- No new dependencies beyond those unless the task explicitly says so.
- Keep modules small and single-purpose; no cross-module imports except from `llmprobe/models.py`.

## Honesty rules (this is the whole point of the product)
- Every reported value carries a provenance marker: `read` (server told us),
  `measured` (we probed it), `inferred` (derived), `estimated` (approximated
  because no exact source was available), `unknown` (we could not tell).
- NEVER fabricate a value to make output look complete. `unknown` is a valid answer
  and is more useful than a confident guess.
- Do not soften or delete a test that fails — fix the code.

## Testing
- Tests are hermetic: no network, no real inference server. Use the mock server
  in `tests/mocks/server.py` and recorded fixtures in `tests/fixtures/`.
- CLI options are asserted by introspecting click params, NEVER by substring-matching
  `--help` output (help text wraps by terminal width and breaks on CI):
  `typer.main.get_command(app)` → `{opt for p in cmd.params for opt in p.opts}`.
