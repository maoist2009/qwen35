---
name: qwen-npu-02-patch-neumann-inverse
description: Stage 2 of the Qwen3.5-0.8B NPU pipeline — replace GatedDeltaNet's exact triangular-solve matrix inversion with a validated MatMul-only Neumann-series approximation. Load only when state.json says stage=patch.
---

# Stage 2 — Patch the matrix inversion

Two files sit next to this SKILL.md in the same folder: `gdn_neumann_patch.py`
and `test_questions.txt`. **Copy them, do not rewrite them.** The algorithm
in `gdn_neumann_patch.py` is already numerically validated (worst relative
error 1e-5 to 1e-7 vs. the exact inverse at chunk_size=64) — you do not need
to derive it from any paper, and you should not modify the math in it.

## Step 2.1 — copy and self-test the provided code

```bash
mkdir -p work/patches
cp "$(dirname "$0" 2>/dev/null || echo .)/gdn_neumann_patch.py" work/patches/ 2>/dev/null \
  || cp ./gdn_neumann_patch.py work/patches/ 2>/dev/null \
  || echo "COPY THE FILE MANUALLY from this skill folder to work/patches/gdn_neumann_patch.py — do not retype it"
cp ./test_questions.txt work/patches/ 2>/dev/null || true
python3 work/patches/gdn_neumann_patch.py
```
Must print `PASS`. If it fails, you copied the file wrong or your `torch`
install is broken (`python3 -c "import torch; print(torch.__version__)"`)
— do not debug the algorithm itself, it's pre-validated.

## Step 2.2 — find the exact triangular-solve call site

Run exactly these commands (this searches YOUR installed package versions,
not a general web search):
```bash
python3 -c "import transformers, os; print(os.path.dirname(transformers.__file__))"
python3 -c "import fla, os; print(os.path.dirname(fla.__file__))" 2>&1 || echo "fla not installed"
grep -rln "solve_triangular\|solve_tril" "$(python3 -c 'import transformers,os;print(os.path.dirname(transformers.__file__))')" 2>/dev/null
grep -rln "solve_triangular\|solve_tril" "$(python3 -c 'import fla,os;print(os.path.dirname(fla.__file__))' 2>/dev/null)" 2>/dev/null
```
Open whichever file(s) get printed. Find the `def <name>(...)` a few lines
above the matched line. Write down the exact **module import path** (e.g.
`fla.ops.utils.solve_tril`) and exact **function name** (e.g. `solve_tril`).

**If both grep commands print nothing:** do one web search —
`"Qwen3.5" GatedDeltaNet transformers modeling source triangular solve` —
find which file it's in from the search results, then grep that specific
file by path instead of the whole package.

## Step 2.3 — write the patch entrypoint

Fill in your two values from Step 2.2 into `<module.path>` and
`<function_name>` below, then save as `work/patches/apply_patch.py`:
```python
import sys
sys.path.insert(0, "work/patches")
from gdn_neumann_patch import neumann_chunk_inverse
import <module.path> as _target_mod

def _patched(A, *args, **kwargs):
    return neumann_chunk_inverse(A, neumann_n=3, residual_s=8, chunk_size_check=64)

_target_mod.<function_name> = _patched
print("PATCH_APPLIED:", "<module.path>.<function_name>")
```

**Verify:**
```bash
python3 -c "exec(open('work/patches/apply_patch.py').read())"
```
Must print a line starting with `PATCH_APPLIED:`. `AttributeError` or
`ImportError` means Step 2.2's values are wrong — go back and re-check the
grep output character-for-character, don't guess a fix.

## Step 2.4 — mechanical correctness check (no subjective judgment needed)

```bash
python3 -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained('work/model_raw')
model = AutoModelForCausalLM.from_pretrained('work/model_raw', torch_dtype=torch.float32)
model.eval()

lines = open('work/patches/test_questions.txt').read().strip().split(chr(10))
passed = 0
for line in lines:
    prompt, expect = line.split('|')
    ids = tok(prompt, return_tensors='pt')
    out = model.generate(**ids, max_new_tokens=8, do_sample=False)
    text = tok.decode(out[0], skip_special_tokens=True)
    ok = expect.lower() in text.lower()
    passed += ok
    print(f'{\"OK \" if ok else \"MISS\"} | {prompt!r} -> {text!r}')
print(f'SCORE: {passed}/{len(lines)}')
"
```
This is the BASELINE score (unpatched). Record the number. Now re-run the
identical script but insert `exec(open("work/patches/apply_patch.py").read())`
right after the imports, before loading the tokenizer/model. Compare the two
`SCORE:` lines.

**Decision rule — purely numeric, no judgment call:**
- patched_score >= baseline_score - 1 → **PASS**, go to "On PASS" below.
- patched_score < baseline_score - 1 → edit `apply_patch.py`, change
  `residual_s=8` to `residual_s=16`, rerun once. Still failing after that →
  block (see below), do not keep guessing parameters past this one retry.

## On PASS
```bash
python3 -c "
import json
s = json.load(open('work/state.json'))
s['artifacts']['patched_model_dir'] = 'work/model_raw'
s['artifacts']['patch_entrypoint'] = 'work/patches/apply_patch.py'
s['stage'] = 'convert_litert'
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
s['blockers'].append('Stage 2: baseline=<N> patched=<N> even after residual_s=16')
json.dump(s, open('work/state.json','w'), indent=2)
"
```
Stop and end your turn.
