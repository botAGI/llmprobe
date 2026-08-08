# Review Findings — llmprobe

Reviewer role. Objective: find defects (code/docstring divergence, unhandled
branches, tests that bypass the production path, values without provenance)
rather than write new features. Every finding lists file, line, and a way to
reproduce. Fixes are limited to what was found, minimal in scope.

Scope reviewed: `llmprobe/cli.py`, `llmprobe/models.py`, `llmprobe/tokens.py`,
`llmprobe/probes/*.py`, plus the adapters in `llmprobe/backends/*.py` that the
CLI reaches at runtime.

Regression gate (run before and after any fix):

```
python3 -m pytest -q
```

Baseline: 84 passed. After the one fix below: 84 passed (no regressions).

---

## Status summary

| # | Severity | Finding | Status |
|---|----------|---------|--------|
| 1 | High | Unhandled `JSONDecodeError` in vLLM adapter on malformed `/v1/models` | Fixed |
| 2 | Medium | Capacity cliff fabricates a never-probed "measured" max when even the minimum probe length is rejected | Documented |
| 3 | Medium | `llmprobe/tokens.py` is dead code — never used by the production capacity path | Documented |
| 4 | Low | Slot mismatch context check message omits the claimed value that triggered it | Documented |

---

## Finding 1 (FIXED) — Unhandled `JSONDecodeError` in vLLM adapter

- File: `llmprobe/backends/vllm.py:126-128`
- Severity: High (unhandled exception in a live run)

### What is wrong

`read_config` calls:

```python
models_resp = await client.get(f"{base_url}/v1/models")
models_resp.raise_for_status()
models = models_resp.json()
```

When the server answers `GET /v1/models` with HTTP 200 but a non-JSON body,
`models_resp.json()` raises `json.JSONDecodeError`. This is not an
`httpx.HTTPError`, so `cli.py:205` (`except httpx.HTTPError`) does not catch it
and the CLI crashes with a raw traceback instead of the graceful exit-2 path.

Detection can select the vLLM adapter (via `/metrics`) even when the
`/v1/models` payload is malformed, so this branch is reachable in a live run.

### How to reproduce

1. Stand up a server that serves `vllm:`-prefixed Prometheus metrics on
   `/metrics` (so the vLLM adapter is selected) but returns `200` with a
   non-JSON body on `/v1/models`.
2. Run `llmprobe --probe <base_url>`.
3. A raw `JSONDecodeError` traceback is printed; no report is produced.

Confirmed with a `httpx.MockTransport` harness routing `/metrics` to
`vllm:num_requests_running 0` and `/v1/models` to `this is not json`:

```
UNHANDLED EXCEPTION type= JSONDecodeError -> Expecting value: line 1 column 1 (char 0)
```

### Fix (applied)

Guard the parse, matching the defensive pattern already used by the generic and
ollama adapters, and fall back to an empty payload so the report still emits
with honest provenance:

```python
models_resp = await client.get(f"{base_url}/v1/models")
try:
    models_resp.raise_for_status()
    models = models_resp.json()
except (httpx.HTTPError, ValueError):
    models = {}
```

After the fix the adapter returns the report with `n_ctx_total` provenance
`unknown` instead of crashing, consistent with the codebase's honesty rule
("`unknown` is a valid answer").

No existing test regressed (`tests/test_backend_vllm.py`, `tests/test_cli.py`,
`tests/test_probe_config.py` all pass). No new tests added in order to keep the
change minimal; the repro is scripted above.

---

## Finding 2 — Capacity cliff reports a never-probed max as "measured"

- File: `llmprobe/probes/capacity.py:184-199`
- Severity: Medium (honesty/provenance violation, not a crash)

### What is wrong

The binary search probes only lengths in `[LO, ceiling]` with `LO = 16`.
If the server rejects every probed length (real capacity is below 16), the
search drives `hi` below `LO`:

```python
lo, hi = LO, ceiling          # 16, 32768
while lo <= hi:
    mid = (lo + hi) // 2
    ...
    else:
        hi = mid - 1

max_accepted = hi             # can fall to 15
```

`max_accepted` then equals `15`, a length that was never probed (only lengths
`>= 16` were sent). `report.py:72-73` renders this value with provenance
`MEASURED`:

```
| max input tokens (...) | unknown | 15 | measured | ok |
```

So llmprobe reports "measured 15" when in fact every probed length was rejected
and the true maximum is only known to be *below 16*.

### How to reproduce

Confirmed with `make_mock_server(max_tokens=10, behavior="hard_error")` (a
server that rejects every length if a hard cliff exists below 16) and
`probe_capacity(..., ceiling=32768)`:

```
max_accepted_tokens = 15   LO = 16   # 15 was never sent to the server
```

### Suggested fix (not applied — semantics require a model decision)

`CapacityResult.max_accepted_tokens` is a plain `int` with no provenance field,
so an exact "unmeasured" value cannot be marked honestly without extending the
model. Two options, left to maintainers:

1. When `max_accepted < LO`, report `max_accepted_tokens = LO` (a length that
   was actually probed) alongside the cliff behavior — this keeps the value a
   real observation and never claims an unprobed length.
2. Extend `CapacityResult` with a provenance/`accepted` flag so the report can
   say "below 16 (not measured)" instead of a confident integer.

---

## Finding 3 — `llmprobe/tokens.py` is dead code

- File: `llmprobe/tokens.py`
- Severity: Medium (maintenance/consistency, no live crash)

### What is wrong

`make_prompt_of_exactly` is implemented and fully tested
(`tests/test_tokens.py`) but is never imported by any production module. The
production capacity probe (`llmprobe/probes/capacity.py`) instead builds
prompts with `_n_token_prompt` (capacity.py:37-45), which assumes each
whitespace-delimited word is exactly one token and never verifies the count.

`capacity.py`'s own module docstring claims its prompts "mirror the `/tokenize`
contract the tokenizer verifies" (capacity.py:40-41), yet no tokenizer is ever
invoked on that path. The verified-tokenizer machinery in `tokens.py` exists
and is tested but is not wired in, so the "exact length" guarantee the token
module advertises is never exercised by the real capacity probe.

### How to reproduce

`grep -rn "make_prompt_of_exactly" llmprobe/` returns only the definition in
`tokens.py`; the only callers are its tests.

### Suggested fix (not applied)

Either wire `make_prompt_of_exactly` into `probe_capacity`, or remove the
unused module. Since installing it into the binary search changes probe request
budgets and golden outputs, this is left as a deliberate, separate change
rather than folded into this review.

---

## Finding 4 — Context mismatch message omits the claiming trigger

- File: `llmprobe/probes/slots.py:45-55`
- Severity: Low (report clarity, no crash)

### What is wrong

`check_slots` sets `mismatched = True` in either of two ways: the derived
per-slot context disagrees with the reported per-slot context, OR it disagrees
with the caller-claimed `claimed_ctx`. But the finding message only ever
mentions the reported per-slot context:

```
f"derived per-slot context ({derived_per_slot}) disagrees with "
f"reported per-slot context ({config.n_ctx_per_slot})"
```

When the mismatch comes solely from the `claimed_ctx` branch (reported
per-slot context is `None` or consistent), the message describes a comparison
against `config.n_ctx_per_slot` that did not actually cause the finding, and
the `advertised`/`measured` fields do carry `claimed_ctx` but the prose does
not explain why.

### How to reproduce

Call `check_slots` with a config where `total_slots` and `n_ctx_total` derive a
per-slot value that matches `config.n_ctx_per_slot` but differs from a supplied
`claimed_ctx`. A `CTX_PER_SLOT_MISMATCH` finding is emitted whose message
cites only the reported per-slot context, not the (equal) reported value that
would appear consistent from the message text alone.

### Suggested fix (not applied)

Include `claimed_ctx` in the message when it is the disambiguating trigger, or
state both compared values clearly. Minor and left as a hardening suggestion.

---

## Notes on what was checked and found clean

- `_OUTCOME_TO_CLIFF[cliff_outcome]` in `capacity.py:198` cannot `KeyError`:
  the classifier only ever returns `accepted` / `hard_error` /
  `silent_truncation`, all present in the mapping.
- Adapter `detect` functions are wrapped by `_detect_adapter`
  (`config.py:40-56`), so a raising probe never blocks selection.
- Provenance markers: the report never prints a numeric row without a marker;
  unreadable fields fall back to `unknown` rather than a confident guess.
- Exit-code derivation (`models.py:102-114`) matches the CLI's use.
