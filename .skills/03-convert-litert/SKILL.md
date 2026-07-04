---
name: qwen-npu-03-convert-litert
description: Stage 3 of the Qwen3.5-0.8B NPU pipeline — convert the patched PyTorch model to a .tflite LiteRT graph. Load only when state.json says stage=convert_litert.
---

# Stage 3 — Convert to LiteRT

This is the hardest stage in the whole pipeline and the one most likely to
need real debugging. Try the easy path first; only fall back to the manual
path if it genuinely fails.

## Step 3.1 — easy path

```bash
pip install --break-system-packages -q litert-torch 2>&1 | tail -5
python3 -c "
exec(open('work/patches/apply_patch.py').read())
from litert_torch.generative.utility import export_hf
export_hf.convert(
    checkpoint_path='work/model_raw',
    output_path='work/qwen35_0.8b_patched.tflite',
)
print('EXPORT_HF_OK')
" 2>&1 | tee work/logs/stage3_attempt1.log
```
If this prints `EXPORT_HF_OK` and the output file exists and is non-empty:
skip straight to **Verify** below.

If it raises `ImportError`, `AttributeError`, or anything mentioning
"unsupported" / "unknown architecture": stop retrying this exact command —
do not try more than 2 total attempts at Step 3.1 — and move to Step 3.2.

## Step 3.2 — manual path (only if 3.1 failed)

Do ONE web search:
`"litert-torch generative API custom attention building block example"`
Read the top 2-3 results for the CURRENT building-block class names (these
change between versions — don't use names from memory). Then write
`work/patches/litert_model_wrapper.py` that:
1. Loads `work/model_raw` with `transformers`, with the Stage 2 patch
   applied (`exec(open("work/patches/apply_patch.py").read())` before
   loading the model).
2. Wraps the forward pass with fixed static shapes (e.g. batch=1,
   seq_len=128 — LiteRT requires static shapes, no dynamic dims).
3. Calls whatever top-level `litert_torch` conversion function the search
   results show (the name/signature may not be `export_hf.convert` — use
   what current docs actually show).

Each time you hit a NEW error message in this step, do exactly one targeted
web search about that specific error message before attempting a fix. Do
not attempt more than 2 different fixes for the same error before searching
again. Stop entirely (see "On block" below) after 5 total attempts counting
both 3.1 and 3.2.

## Verify

```bash
python3 -c "
import os
p = 'work/qwen35_0.8b_patched.tflite'
assert os.path.exists(p), 'file missing'
assert os.path.getsize(p) > 0, 'file is empty'
print('PASS')
"
```

## On PASS
```bash
python3 -c "
import json
s = json.load(open('work/state.json'))
s['artifacts']['tflite_path'] = 'work/qwen35_0.8b_patched.tflite'
s['stage'] = 'npu_compile'
s['stage_status'] = 'done'
json.dump(s, open('work/state.json','w'), indent=2)
"
```
Then load the `00-orchestrator` skill again.

## On block (5 attempts exhausted)
```bash
python3 -c "
import json
s = json.load(open('work/state.json'))
s['stage_status'] = 'blocked'
s['blockers'].append('Stage 3: <paste the last real error text here>')
json.dump(s, open('work/state.json','w'), indent=2)
"
```
Stop and end your turn — this stage genuinely needs a human if you're stuck
here, don't keep spinning.
