# Review Findings — llmprobe

Reviewer role. Objective: find defects rather than write new features. Every
finding lists file, line, and a way to reproduce. Fixes are limited to what
was found, minimal in scope.

This review is scoped to one defect family, as assigned: **silent exception
swallowing** (`except` without logging or with `pass`), **default values that
mask an error**, and **functions returning `None` instead of an explicit
failure indicator**. It covers the production path the CLI exercises:
`llmprobe/probes/*.py`, `llmprobe/backends/*.py`, `llmprobe/tokens.py`. The
report shows the code being honest about what it could not read rather than
guessing.

Prior reviews (beads lp-atp, lp-hb8, lp-6ur) on unhandled exceptions,
provenance of capacity lower bounds, and dead code are resolved on `main`;
this document supersedes the previous `REVIEW_FINDINGS.md` and reports only
the findings of the assigned silent-swallowing review.

Regression gate (run before and after any fix):

```
python3 -m pytest -q
```

Baseline: 144 passed. After the fixes below: 144 passed (no regressions).

---

## Status summary

| # | Severity | Finding | Status |
|---|----------|---------|--------|
| 1 | Medium | `/slots` read failure swallowed with bare `pass` (no logging) and JSON parse error left unhandled | Fixed |
| 2 | Medium | Backend `detect` swallow-all (`except Exception`) is silent SAM-visible: any error is masked as "no match" with no trace | Fixed |
| 3 | Low | Ollama `_get_json`/`_post_json` return `None` for both "no data" and "probe failed", so a failed read is reported as `unknown` config with no error finding | Documented — fix not applied |

---

## Finding 1 (FIXED) — llama.cpp `/slots` read failure swallowed with bare `pass`

- File: `llmprobe/backends/llamacpp.py` (`read_config`, `/slots` block)
- Severity: Medium (silent exception swallow)

### What is wrong

The optional `/slots` cross-check is the single bare `except ... : pass` in the
codebase:

```python
try:
    slots_resp = await client.get(f"{base}/slots", timeout=client.timeout)
    if slots_resp.status_code == 200:
        _source_slot_ctx(slots_resp.json(), merge)
except httpx.HTTPError:
    pass
```

A transport failure reaching `/slots` (connection reset, timeout, DNS) is
silently discarded. The per-slot cross-check / provenance refinement is
skipped without a single log line, so an operator has no way to know a server
that *does* serve `/slots` had its contribution dropped. The `/slots` read is
the only place a non-fatal failure is swallowed in the llama.cpp adapter.

Separately, the `except` clause only catches `httpx.HTTPError`; if `/slots`
returns HTTP 200 with an unparsable body, `slots_resp.json()` raises
`ValueError` which is **not** caught and propagates up through `read_config`,
crashing the whole probe. Both defects sit in the same optional-read block and
both are "the optional `/slots` lookup did not yield a value".

### How to reproduce

1. `read_config` against a llama.cpp server whose `/slots` transport fails
   (e.g. a handler that raises `httpx.ConnectError` for the `/slots` path only).
2. `read_config` returns normally; nothing is logged; the slot cross-check is
   silently skipped.
3. Point `/slots` at a 200 body that is not valid JSON (e.g. `"not json"`).
   `read_config` raises `ValueError` and the probe aborts.

### Fix (applied)

Log the swallowed failure and tolerate the JSON-parse error in the same
optional-read block — both are the optional `/slots` lookup not yielding a
value, and both must never crash a probe nor vanish silently:

```python
try:
    slots_resp = await client.get(f"{base}/slots", timeout=client.timeout)
    if slots_resp.status_code == 200:
        _source_slot_ctx(slots_resp.json(), merge)
except (httpx.HTTPError, ValueError):
    logger.exception("GET %s/slots failed; per-slot cross-check skipped", base)
```

`read_config` now surfaces the reason for the skipped cross-check via the
logger while still never raising on this optional read. No existing test
regressed (`tests/test_backend_llamacpp.py`, including
`test_slots_501_does_not_raise`, all pass).

---

## Finding 2 (FIXED) — Backend `detect` swallow-all is silent

- File: `llmprobe/probes/config.py` (`_detect_adapter`)
- Severity: Medium (silent exception swallow + misattribution hazard)

### What is wrong

Adapter selection wraps every backend's `detect` in a `try/except Exception`
that returns confidence `0.0` on **any** exception:

```python
try:
    score = await adapter(client, base_url)
except Exception:
    return 0.0
```

The breadth is intentional (a failing probe must never block selection), but
the swallow is silent. This is the one catch-all in the codebase that also
absorbs programmer errors (a `KeyError`, `TypeError`, or `AttributeError`
introduced into any adapter's `detect`), converts them into "no match", and
thereby lets a *different* adapter win the selection round. The whole report
is then attributed to the wrong backend — with no log line, no finding, and
no provenance marker indicating anything went wrong during selection. Per the
README's honesty principle this should at least be diagnosable.

### How to reproduce

1. Introduce a transient bug (e.g. `undefined_var` / a `KeyError`) inside any
   adapter's `detect`.
2. Run the CLI.
3. Selection silently falls through to another adapter and emits a plausible
   but wrong report; nothing indicates the failing adapter raised.

### Fix (applied)

Keep the same robust control flow (selection must never block) but stop the
swallow being silent: log the exception with the failing `base_url` before
returning `0.0`:

```python
except Exception:
    logger.exception(
        "backend detect() raised on %s; treated as no-match", base_url
    )
    return 0.0
```

A genuine detection bug is now visible in the log for diagnostics. Behavior is
otherwise unchanged — no test regressed (`tests/test_config.py`,
`tests/test_probe_config.py`, `tests/test_cli.py` all pass).

---

## Finding 3 (DOCUMENTED, not fixed) — Ollama helpers return `None` for both "no data" and "failed"

- File: `llmprobe/backends/ollama.py` (`_get_json`, `_post_json`, `read_config`)
- Severity: Low (None-return conflates failure with absence; no code change)

### What is wrong

`_get_json` / `_post_json` return `None` for three indistinguishable causes: a
transport failure, a non-200 status, and an unparsable/`ValueError` body:

```python
async def _get_json(client, url):
    try:
        resp = await client.get(url)
    except httpx.HTTPError:
        return None
    if resp.status_code != 200:
        return None
    try:
        payload = resp.json()
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None
```

`read_config` then treats `None` as "no online model" and returns an
`EffectiveConfig` with `model_id=""`, `n_ctx_total=None`, provenance
`UNKNOWN`, and — crucially — **no `ERROR` finding**. A live server whose
`/api/tags` fails on the wire is therefore reported as a perfectly normal
"unknown config" rather than as a failed read. This is the inverse of the
generic adapter, which emits an explicit `ERROR` finding (code
`GENERIC_MODELS_UNREACHABLE`) when its model endpoint cannot be read. The
report is not *untrue* (provenance is honestly `unknown`), but the failure
signal is masked.

The `None` return is judged an accepted, low-severity compromise: the CLI
already performs a reachability pre-flight (`cli.py:_assert_reachable`), so a
fully unreachable server raises before `read_config` runs; the residual risk
is only a partial failure (reachable `/` but failed `/api/tags`), which is an
edge case. Fixing it properly (returning a findings list from the helpers or
distinguishing "no models" from "failed") is a larger change than the minimal
scope of this review, so it is documented here rather than applied.

### How to reproduce

Serve `GET /api/tags` as a transport error while `/` (the reachability probe)
succeeds, then run `read_effective_config`. It returns an empty `unknown`
config with zero findings instead of surfacing an error.

---

## Notes on what was checked and found clean

- `tokens._tokenize`, `capacity._post_embed`, `capacity._post_chat` return
  `None` on unreadable responses, but their docstrings make the contract
  explicit (transport failures propagate; `None` means "non-200 or unparsable")
  and callers treat `None` honestly (`hard_error`), so this is an explicit
  failure indicator, not a masked failure.
- The remaining `except ValueError` / `except (KeyError, ...)` guards across
  `vllm.py`, `ollama.py`, `capacity.py`, and `tokens.py` all either return an
  explicit `None` handled honestly or fall back to an explicit provenance
  `unknown`; none swallow without a traceable response.
- `cli.py`'s single `except httpx.HTTPError` prints the failure to stderr and
  exits with code 2 — a surfaced, not swallowed, error.
- `report.py`'s `except (TOMLDecodeError, OSError)` / `PackageNotFoundError`
  fall back to the honest `"unknown"` version marker rather than a guess.
