---
name: qwen-npu-orchestrator
description: Entry point for the Qwen3.5-0.8B NPU pipeline. ALWAYS load this first, before any other skill in this pipeline. It tells you which single next skill to load — do not load the other stage skills speculatively, only load the one this points you to.
---

# Orchestrator — read this, then load exactly ONE other skill

This is a 4-stage pipeline. Each stage is its own skill folder (sibling
directories to this one: `01-download-model`, `02-patch-neumann-inverse`,
`03-convert-litert`, `04-qnn-npu-compile`). Each stage skill is self
contained — it does not assume you've read the others. **Load only the one
skill matching your current stage, not all four**, to save context.

## Step 1: check state

```bash
mkdir -p work/logs
if [ -f work/state.json ]; then
  cat work/state.json
else
  echo '{"stage":"download","stage_status":"pending","stage_attempts":0,"artifacts":{},"blockers":[],"last_updated":""}' > work/state.json
  cat work/state.json
fi
```

## Step 2: map `stage` field to the skill folder to load next

| `state.json` "stage" value | skill folder to load           |
|-----------------------------|---------------------------------|
| `download`                  | `01-download-model`             |
| `patch`                     | `02-patch-neumann-inverse`       |
| `convert_litert`             | `03-convert-litert`             |
| `npu_compile`                | `04-qnn-npu-compile`            |

If `stage_status` is `"blocked"`: **do not proceed automatically.** Print
the `blockers` array contents and stop — a human needs to look at this.

If `stage_status` is `"done"` and `stage` is `npu_compile`: the whole
pipeline is finished. Print `work/SUMMARY.md` if it exists and stop.

Otherwise: load the ONE matching skill above and follow it exactly. Every
rule that applies across all stages (never delete `~/.qwen` or `work/`
outside of an explicit instruction, never mark a stage `done` without its
Verify command printing `PASS`, stop after 3 failed identical retries and
set `blocked`) is repeated inside each stage skill — you don't need to
re-read this file for those, they're self-contained.
