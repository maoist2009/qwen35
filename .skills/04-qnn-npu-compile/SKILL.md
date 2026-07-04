---
name: qwen-npu-04-qnn-npu-compile
description: Stage 4 of the Qwen3.5-0.8B NPU pipeline — install qsc-cli/QAIRT SDK and AOT-compile the .tflite into an SM8750 NPU binary. Load only when state.json says stage=npu_compile.
---

# Stage 4 — QNN SDK install + NPU compile

`acquire_qnn_sdk.sh` sits next to this SKILL.md. **Copy it into `work/` and
run it, do not rewrite it** — it already handles the qsc-cli self-update
wait and the idempotent-login bug.

## Step 4.1 — get the SDK

```bash
cp ./acquire_qnn_sdk.sh work/ 2>/dev/null || echo "copy acquire_qnn_sdk.sh from this skill folder to work/ manually"
chmod +x work/acquire_qnn_sdk.sh
export QSC_EMAIL="${QSC_EMAIL:?set as a GitHub Actions secret, never hardcode}"
export QSC_PASSWORD="${QSC_PASSWORD:?set as a GitHub Actions secret, never hardcode}"
bash work/acquire_qnn_sdk.sh
source "$HOME/litert_workspace/.env"
echo "QNN_SDK_ROOT=$QNN_SDK_ROOT"
```
If this script exits non-zero, read its `[FATAL]` line — it already tells
you exactly what failed (deb install, self-update timeout, login, download,
or unzip). Do not improvise a different fix than what the message says;
if the message doesn't map to an obvious retry, block (see below).

## Step 4.2 — AOT compile to SM8750

**This task targets SM8750 (Snapdragon 8 Elite, previous generation) — not
SM8850 (Snapdragon 8 Elite Gen 5). These are different chips. Do not
substitute one for the other even if a doc example uses SM8850.**

```bash
pip install --break-system-packages -q ai-edge-litert
mkdir -p work/npu_output

for arch in x86_64-linux-clang x86_64-linux-ubuntu; do
  if [ -d "$QAIRT_ROOT/bin/$arch" ]; then
    export PATH="$QAIRT_ROOT/bin/$arch:$PATH"
    export LD_LIBRARY_PATH="$QAIRT_ROOT/lib/$arch:$LD_LIBRARY_PATH"
    break
  fi
done

python3 -c "
from ai_edge_litert.aot import aot_compile as aot_lib
from ai_edge_litert.aot.vendors.qualcomm import target as qnn_target

t = qnn_target.Target(qnn_target.SocModel.SM8750)
aot_lib.aot_compile('work/qwen35_0.8b_patched.tflite', output_dir='work/npu_output', target=[t])
print('COMPILE_CALL_RETURNED')
" 2>&1 | tee work/logs/stage4_compile.log

echo "--- checking /tmp/*.error (Qualcomm's compiler often fails silently from Python's view) ---"
ls /tmp/*.error 2>/dev/null && cat /tmp/*.error || echo "no error logs found"
```

## Verify (mechanical — file must exist AND be non-empty)

```bash
python3 -c "
import glob, os
files = glob.glob('work/npu_output/*')
nonempty = [f for f in files if os.path.getsize(f) > 0]
assert nonempty, 'no non-empty output files in work/npu_output'
print('PASS')
print('OUTPUT_FILES:', nonempty)
"
```

**If this fails:** read `work/logs/stage4_compile.log` and the `/tmp/*.error`
output above. If the error text names an unsupported op, that means the
Stage 2 patch didn't actually make it into the exported `.tflite` — the
most common cause is that Stage 3's export loaded the model fresh without
first running `apply_patch.py`. Go back to Stage 3, confirm
`exec(open("work/patches/apply_patch.py").read())` runs BEFORE the model
load in that stage's conversion script, then redo Stage 3 and Stage 4.
Otherwise, block with the exact error text — don't guess further SDK config
changes.

## On PASS
```bash
python3 -c "
import json
s = json.load(open('work/state.json'))
s['artifacts']['npu_output_dir'] = 'work/npu_output'
s['stage_status'] = 'done'
json.dump(s, open('work/state.json','w'), indent=2)
"
cat > work/SUMMARY.md <<'EOF'
# Pipeline summary
- Model: Qwen/Qwen3.5-0.8B
- Patch: Neumann-series approximation (see work/patches/apply_patch.py)
- Stage 2 baseline/patched scores: see work/logs
- Stage 3 conversion path: see work/logs/stage3_attempt1.log
- Final NPU artifacts: work/npu_output/
EOF
echo "PIPELINE COMPLETE"
```
Stop — the task is done.

## On block
```bash
python3 -c "
import json
s = json.load(open('work/state.json'))
s['stage_status'] = 'blocked'
s['blockers'].append('Stage 4: <paste the exact error text here>')
json.dump(s, open('work/state.json','w'), indent=2)
"
```
Stop and end your turn.
