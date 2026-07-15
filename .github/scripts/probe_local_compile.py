#!/usr/bin/env python3
"""Discovery probe: prove the local qairt-converter 2.42.0.251225 can compile an
INT4-QDQ (w4a16) ONNX graph into a qnn_context_binary — the exact case that FAILED
in the cloud (job jgl1xo4e5: "INT4 QDQ not loadable"). Uses a tiny synthetic model
with the same 3-input C++ contract as DreamLite CLIP (input_embedding/position_ids/
attention_mask -> hidden) so the compile flags transfer verbatim to the real clip.
"""
import os, sys, json, subprocess, shutil
import numpy as np
import torch
import torch.nn as nn

WORK = os.environ.get("GITHUB_WORKSPACE", os.getcwd())
OUT = os.path.join(WORK, "probe_out")
os.makedirs(OUT, exist_ok=True)

QAIRT_ROOT = os.environ.get("QAIRT_ROOT") or os.environ.get("QNN_SDK_ROOT")
assert QAIRT_ROOT and os.path.isdir(QAIRT_ROOT), f"QAIRT_ROOT missing: {QAIRT_ROOT}"

# locate qairt-converter
conv = None
for arch in ("x86_64-linux-clang", "x86_64-linux-ubuntu"):
    p = os.path.join(QAIRT_ROOT, "bin", arch, "qairt-converter")
    if os.path.exists(p):
        conv = p
        os.environ["PATH"] = os.path.join(QAIRT_ROOT, "bin", arch) + ":" + os.environ["PATH"]
        lib = os.path.join(QAIRT_ROOT, "lib", arch)
        os.environ["LD_LIBRARY_PATH"] = lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")
        break
assert conv, "qairt-converter not found under QAIRT_ROOT/bin"
print(f"[probe] qairt-converter = {conv}")

def run(cmd):
    print(f"[probe] $ {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    print("---- stdout ----"); print(r.stdout)
    print("---- stderr ----"); print(r.stderr)
    print(f"[probe] rc={r.returncode}")
    return r

# 1) dump help (pin exact flags for context binary + input_list format)
run([conv, "--help"])
run([conv, "--help-all"] if False else [conv, "--help"])

# 2) tiny clip-like model
class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(8, 8)
        self.ln = nn.LayerNorm(8)
    def forward(self, input_embedding, position_ids, attention_mask):
        x = self.lin(input_embedding) + position_ids.permute(1, 0, 2)[:, 0:1, :] * 0.0
        x = x * attention_mask.unsqueeze(-1)
        return self.ln(x)

m = Tiny()
m.eval()
N = 4
emb = torch.randn(1, N, 8)
pos = torch.randint(0, N, (3, N), dtype=torch.int32)
mask = torch.ones(1, N)
fp32 = os.path.join(OUT, "tiny_fp32.onnx")
torch.onnx.export(
    m, (emb, pos, mask), fp32,
    input_names=["input_embedding", "position_ids", "attention_mask"],
    output_names=["hidden"],
    opset_version=18, dynamic_axes=None,
)
print(f"[probe] exported {fp32} ({os.path.getsize(fp32)} bytes)")

# 3) AIMET w4a16 quantize -> QDQ onnx (INT4 wt / INT16 act)
import onnx
from aimet_onnx.quantsim import QuantSim
try:
    from aimet.common.defs import QuantScheme
except Exception:
    from aimet_onnx.defs import QuantScheme

dummy = {
    "input_embedding": np.random.randn(1, N, 8).astype(np.float32),
    "position_ids": np.random.randint(0, N, (3, N), dtype=np.int32),
    "attention_mask": np.ones((1, N), dtype=np.float32),
}
onnx_model = onnx.load(fp32)
sim = QuantSim(model=onnx_model, dummy_input=dummy,
               quant_scheme=QuantScheme.post_training_tf,
               default_param_bw=4, default_activation_bw=16, config_file=None)
def fwd(session, _dummy):
    session.run(None, dummy)
sim.compute_encodings(fwd, None)
qpath = os.path.join(OUT, "tiny_q")
sim.save(qpath)
print(f"[probe] AIMET w4a16 saved -> {qpath}.onnx/.encodings/.data")
for ext in (".onnx", ".encodings", ".data"):
    f = qpath + ext
    print(f"   {f}: {os.path.getsize(f) if os.path.exists(f) else 'MISSING'} bytes")

# 4) input_list (random shapes; one line, all inputs)
il = os.path.join(OUT, "input_list.txt")
with open(il, "w") as f:
    f.write("input_embedding 1,4,8 position_ids 3,4 attention_mask 1,4\n")
print(f"[probe] input_list:\n" + open(il).read())

# 5) local compile -> qnn_context_binary
binp = os.path.join(OUT, "tiny.bin")
r = run([conv,
         "--input_network", qpath + ".onnx",
         "--output_path", binp,
         "--input_list", il,
         "--target_runtime", "qnn_context_binary",
         "--quantize_io"])
if os.path.exists(binp) and os.path.getsize(binp) > 0:
    print(f"[probe] SUCCESS tiny.bin = {os.path.getsize(binp)} bytes")
else:
    print("[probe] COMPILE FAILED (see qairt-converter output above)")
    # try alternate input_list format (colon) for the record
    il2 = os.path.join(OUT, "input_list2.txt")
    with open(il2, "w") as f:
        f.write("input_embedding:1,4,8\nposition_ids:3,4\nattention_mask:1,4\n")
    run([conv, "--input_network", qpath + ".onnx", "--output_path",
         os.path.join(OUT, "tiny2.bin"), "--input_list", il2,
         "--target_runtime", "qnn_context_binary", "--quantize_io"])
print("[probe] DONE")
