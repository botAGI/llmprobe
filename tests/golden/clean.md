# Capability Report — http://localhost:8080

llmprobe 0.1.0 · measured 2026-08-11T07:38:33Z (UTC)

| Property | Claimed | Measured | Source | Verdict |
| --- | --- | --- | --- | --- |
| backend | llamacpp | llamacpp | read | ok |
| model | mock/llama-3.1-8b | mock/llama-3.1-8b | read | ok |
| context (total) | 8192 | 8192 | read | ok |
| context (per slot) | 2048 | 2048 | read | ok |
| slots | 4 | 4 | read | ok |
| max input tokens (/completion) | unknown | 8192 | measured | ok |
| token count (/completion) | unknown | exact | measured | ok |
| cliff behaviour (/completion) | unknown | accepted | measured | ok |
| probe requests used (/completion) | unknown | 3 | measured | ok |
