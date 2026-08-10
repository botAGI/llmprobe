# Capability Report — http://localhost:8080

| Property | Claimed | Measured | Source | Verdict |
| --- | --- | --- | --- | --- |
| backend | llamacpp | llamacpp | read | ok |
| model | mock/llama-3.1-8b | mock/llama-3.1-8b | read | ok |
| context (total) | 8192 | 8192 | read | ok |
| context (per slot) | 2048 | 2048 | read | ok |
| slots | 4 | 4 | read | ok |
| max input tokens (/completion) | unknown | 7168 | measured | ok |
| token count (/completion) | unknown | exact | measured | ok |
| cliff behaviour (/completion) | unknown | silent_truncation | measured | truncated |

## Findings

- **[mismatch] UBATCH_CEILING**: advertised=8192 (read) vs measured=7168 (measured) — requests past 7168 tokens are silently truncated

## Fix

--batch-size 8192 --ubatch-size 8192
