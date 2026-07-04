---
name: qwen-npu-01-download-model
description: Stage 1 of the Qwen3.5-0.8B NPU pipeline — download the base model from Hugging Face. Load only when state.json says stage=download.
---

# Stage 1 — Download the model

**Do not use `git clone`, `wget`, `curl`, or `huggingface-cli` (that command
is deprecated). Use `hf download` — it is already installed in this
environment and handles resume/retry/integrity itself.**

The exact repo id is `Qwen/Qwen3.5-0.8B` (verified real, released 2026-03-02,
~1.77GB, do not substitute a different id or add a suffix like `-Instruct`).

```bash
mkdir -p work/model_raw
hf download Qwen/Qwen3.5-0.8B --local-dir work/model_raw --format json > work/logs/hf_download.json
cat work/logs/hf_download.json
```
`--format json` gives machine-parseable output instead of a human progress
bar — read it, don't guess success from absence of errors alone.

**If `hf: command not found`:**
```bash
pip install --break-system-packages -q -U huggingface_hub
hf --version
```
Then retry the download command above once.

**If you get a 403 / gated-model error:** this specific repo is not gated
(it's a public Apache-licensed release), so a 403 here means something else
is wrong (bad token env var, wrong repo id typo). Run
`hf auth whoami` to check auth state and `hf download Qwen/Qwen3.5-0.8B --dry-run`
to see what it thinks it would fetch, before trying anything else. Do not
add `--token` guesses.

## Verify (must print PASS before advancing)

```bash
python3 -c "
import json, os
d = 'work/model_raw'
assert os.path.exists(f'{d}/config.json'), 'missing config.json'
idx = f'{d}/model.safetensors.index.json'
if os.path.exists(idx):
    files = set(json.load(open(idx))['weight_map'].values())
    for f in files:
        p = f'{d}/{f}'
        assert os.path.getsize(p) > 0, f'{p} is empty'
else:
    assert os.path.getsize(f'{d}/model.safetensors') > 0
print('PASS')
"
```

## On PASS
```bash
python3 -c "
import json
s = json.load(open('work/state.json'))
s['artifacts']['model_dir'] = 'work/model_raw'
s['stage'] = 'patch'
s['stage_status'] = 'done'
json.dump(s, open('work/state.json','w'), indent=2)
print('state.json updated -> stage=patch')
"
```
Then load the `00-orchestrator` skill again to pick up the next stage skill.

## On repeated failure
If the exact same error happens 3 times in a row, do not keep retrying:
```bash
python3 -c "
import json
s = json.load(open('work/state.json'))
s['stage_status'] = 'blocked'
s['blockers'].append('Stage 1 download: <paste the exact error text here>')
json.dump(s, open('work/state.json','w'), indent=2)
"
```
Stop and end your turn.
