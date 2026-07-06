---
name: qwen-npu-03-convert-litert
description: Stage 3 of the Qwen3.5-0.8B NPU pipeline — convert the patched PyTorch model to a .tflite LiteRT graph, using a third-party reference implementation of GatedDeltaNet support for litert-torch as a starting point. Load only when state.json says stage=convert_litert.
---

# Stage 3 — Convert to LiteRT

litert-torch has no official GatedDeltaNet building block. A third party
already built one:
`https://github.com/byepv2493k-ship-it/litert-torch-qwen3.5`

**Use it as a source of patch files to adapt, not as a working pipeline to
run as-is.** Its `requirements_conda.txt` pins `ai-edge-litert-nightly`,
`ai-edge-quantizer-nightly`, and a specific nightly PyTorch build — nightly
package indices don't keep old builds around, so those exact pins are very
likely gone or broken by now. **Do not `pip install -r requirements_conda.txt`
from that repo.** Install current versions yourself (Step 3.1) and expect
to have to reconcile API drift between "when that repo was written" and "now"
(Step 3.3).

## Step 3.1 — get current, non-stale dependency versions

```bash
mkdir -p work/vendor
git clone --depth 1 https://github.com/byepv2493k-ship-it/litert-torch-qwen3.5 work/vendor/litert-torch-qwen3.5
cat work/vendor/litert-torch-qwen3.5/QWEN_LITERT_SETUP.md   # read once for context, don't follow its exact install commands

# Do NOT use work/vendor/.../requirements_conda.txt. Instead find litert-torch's
# OWN current pinned/recommended versions from its real repo:
```
Do one web search: `"google-ai-edge/litert-torch" requirements OR pyproject current torch version`
— find what version of `torch` litert-torch itself currently expects (this
changes over time; don't assume a number from memory or from the vendored
repo's stale pin). Then:
```bash
pip install --break-system-packages -q -U "torch==<version found above>" \
  || pip install --break-system-packages -q -U torch   # fallback to latest if the exact pin is unavailable
pip install --break-system-packages -q -U ai-edge-litert litert-torch ai-edge-quantizer
python3 -c "import litert_torch, torch; print('litert_torch OK, torch', torch.__version__)"
```
If this import fails, do one web search for the exact error message before
trying anything else — package names/split between `litert-torch` and
`ai-edge-torch` may have changed again since this skill was written.

## Step 3.2 — locate where the four custom files need to live

The vendored repo's files are meant to sit inside litert-torch's own
`generative/examples/` tree, alongside its other model examples (e.g. gemma).
Find where that actually is in YOUR installed package, and look at an
existing example for the current expected file layout/base classes —
don't assume the vendored repo's relative paths still match:
```bash
INSTALLED_ROOT=$(python3 -c "import litert_torch, os; print(os.path.dirname(litert_torch.__file__))")
echo "$INSTALLED_ROOT"
ls "$INSTALLED_ROOT/generative/examples/" 2>/dev/null
# Look at one existing example's file layout (pick whichever one exists, e.g. gemma):
find "$INSTALLED_ROOT/generative/examples" -maxdepth 2 -iname "*.py" | head -20
```
Compare that layout against what's in the vendored repo:
```bash
find work/vendor/litert-torch-qwen3.5 -iname "model_config.py" -o -iname "gated_deltanet.py" -o -iname "qwen3_5.py" -o -iname "convert_v3_5_to_tflite.py"
```
Create `"$INSTALLED_ROOT/generative/examples/qwen35/"` (or whatever the
existing examples' naming convention actually is — match it, don't invent a
different one) and copy the four files there.

## Step 3.3 — reconcile API drift (expected, budget real time for this)

The four files were written against some prior version of `litert_torch`'s
internal APIs (base attention classes, config dataclass fields, the
converter entrypoint signature). Do NOT assume they still match. For each
copied file:
```bash
python3 -c "import ast; ast.parse(open('<copied_file>').read())"  # at least confirm it's syntactically valid Python first
```
Then try importing each one directly and read the traceback:
```bash
python3 -c "
import sys
sys.path.insert(0, '$INSTALLED_ROOT/generative/examples/qwen35')
import model_config
import gated_deltanet
import qwen3_5
print('IMPORTS_OK')
"
```
**If/when this fails** (expected — don't treat one import error as a
blocker), the traceback will point at a specific missing class/attribute.
Do ONE targeted web search for that exact missing name plus "litert-torch"
or "ai-edge-torch", find its current name/location, and do a search-and-replace
in the copied file. Repeat per error. Cap: if you're still fixing import
errors after 8 distinct fixes, stop this approach and fall back to Step 3.5
(minimal-scaffold fallback) rather than continuing indefinitely — this
specific reconciliation is allowed to take real iteration, but not unbounded
iteration.

One known historical issue already documented in the vendored repo's own
notes, so you don't have to rediscover it: it says it removed a hard import
of `torchao.quantization.pt2e.quantize_pt2e` to fix a TorchAO compatibility
break on newer nightly builds. If you hit a similar `torchao` import error,
the fix pattern is the same — make the import lazy/conditional rather than
top-level, don't try to pin an old torchao version to match it.

## Step 3.4 — apply the Stage 2 Neumann patch to THIS code path too

**Important: this is a separate code path from Stage 2's patch.** Stage 2
patched the HuggingFace/`fla` runtime implementation (used for the CPU
correctness sanity check). This vendored `gated_deltanet.py` is a *different*,
independent reimplementation used specifically for `torch.export` tracing —
patching Stage 2's target does not affect this file. Repeat the same
grep-and-patch procedure from Stage 2, but against this file:
```bash
grep -n "solve_triangular\|solve_tril\|triangular" "$INSTALLED_ROOT/generative/examples/qwen35/gated_deltanet.py"
```
Find the function doing the exact inverse, then edit that file directly
(not a monkeypatch this time, since you're editing your own copy) to call
`neumann_chunk_inverse` from `work/patches/gdn_neumann_patch.py` (copy that
file next to `gated_deltanet.py` and import it) instead of the exact solve.
Keep the same parameters: `neumann_n=3, residual_s=8, chunk_size_check=64`.

## Step 3.5 — run the conversion

```bash
mkdir -p work/litert_output
python3 -m litert_torch.generative.examples.qwen35.convert_v3_5_to_tflite \
  --checkpoint_path=work/model_raw \
  --output_path=work/litert_output/ \
  --quantize=dynamic_int8 \
  --kv_cache_max_len=2048 \
  2>&1 | tee work/logs/stage3_convert.log
```
(Flags come from the vendored repo's documented usage — check
`--help` on the module first in case the signature also drifted:
`python3 -m litert_torch.generative.examples.qwen35.convert_v3_5_to_tflite --help`)

## Step 3.6 — fallback if 3.1–3.5 don't converge

If you're still blocked after Step 3.3's 8-fix cap, or the conversion script
itself has drifted too far to run: fall back to writing a minimal manual
scaffold from scratch using `litert_torch`'s Generative API building blocks
directly (search `"litert-torch generative API custom attention building
block example"` for current class names), using the vendored repo's
`gated_deltanet.py` purely as *reference math* (it already shows the
GatedDeltaNet recurrence structure, even if its exact class hierarchy no
longer matches). This is real engineering effort, not a quick fix — if you
reach this fallback, log clearly in `work/logs` that you did, so the next
run (or a human) knows the easy path didn't work.

## Verify

```bash
python3 -c "
import glob, os
files = glob.glob('work/litert_output/*.tflite')
nonempty = [f for f in files if os.path.getsize(f) > 0]
assert nonempty, 'no non-empty .tflite in work/litert_output'
print('PASS')
print('FILES:', nonempty)
"
```

## On PASS
```bash
python3 -c "
import json, glob
s = json.load(open('work/state.json'))
tflite_files = glob.glob('work/litert_output/*.tflite')
s['artifacts']['tflite_path'] = tflite_files[0]
s['artifacts']['vendor_repo_dir'] = 'work/vendor/litert-torch-qwen3.5'
s['stage'] = 'npu_compile'
s['stage_status'] = 'done'
json.dump(s, open('work/state.json','w'), indent=2)
"
```
Then load the `00-orchestrator` skill again.

## On block
```bash
python3 -c "
import json
s = json.load(open('work/state.json'))
s['stage_status'] = 'blocked'
s['blockers'].append('Stage 3: <paste exact error/state here — note whether you were stuck in 3.3 API reconciliation or 3.6 manual fallback>')
json.dump(s, open('work/state.json','w'), indent=2)
"
```
Stop and end your turn.
