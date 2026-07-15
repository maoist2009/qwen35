#!/usr/bin/env python3
"""Cloud convert DreamLite CLIP (Qwen3VL text tower) -> clip.bin via qai_hub.

Designed to run in CI (GitHub Actions). Steps:
  1. Pull text_encoder from HuggingFace (DREAMLITE_REPO, subfolder=text_encoder).
  2. Export a clean fp32 ONNX, weights externalized to clip.onnx.data
     (avoids the 2 GiB protobuf limit; clip weights ~3.7 GB).
  3. qai_hub cloud quantize w4a16 (INT4 weights / INT16 acts).
     (Optional: fold quantize into the compiler via DREAMLITE_QUANT_IN_COMPILE=1,
      using --quantize_full_type w4a16, if the standalone quantize job wedges.)
  4. qai_hub cloud compile -> qnn_context_binary (--quantize_io).
  5. Download build/dreamlite/clip.bin.

CLIP input contract (C++ loader):
  input_embedding [1,N,2048] float32   (token_emb lookup result)
  position_ids     [3,N]      int32     (mRoPE, must stay int32)
  attention_mask   [N]        float32
  -> hidden        [1,N,2048] float32   (last_hidden_state)

Auth: QAI_HUB_API_TOKEN env (repo secret). No S3 IP-pin patch (Actions = Azure/US).
"""
import os
import sys
import time
import shutil
import traceback

import numpy as np
import torch
import onnx

# ---- config from env ----
REPO = os.environ.get("DREAMLITE_REPO", "carlofkl/DreamLite-base")
OUT = os.environ.get("DREAMLITE_OUT", "build/dreamlite")
N = int(os.environ.get("DREAMLITE_TEXT_SEQ_LEN", "512"))
H = int(os.environ.get("DREAMLITE_HIDDEN", "2048"))
NUM_CALIB = int(os.environ.get("DREAMLITE_CALIB_SAMPLES", "100"))
PRECISION = os.environ.get("DREAMLITE_PRECISION", "w4a16").strip().lower()
QUANT_IN_COMPILE = os.environ.get("DREAMLITE_QUANT_IN_COMPILE", "").strip() in ("1", "true", "yes")
TARGETS = [s.strip() for s in os.environ.get("DREAMLITE_TARGETS", "sm8750").split(",") if s.strip()]
# Default optimization level 1: level 3 (the QAIRT default) hit the cloud
# compile time limit ("Time limit exceeded in compilation step") on the
# 3.7 GB text tower. Level 1 trades some inference perf for a much shorter
# compile; the converter + in-compiler w4a16 quantizer already pass at any
# level. Lower the cost of a re-run; bump to 2/3 only once it lands.
COMPILE_OPTS = os.environ.get(
    "DREAMLITE_COMPILE_OPTS",
    "--target_runtime qnn_context_binary --quantize_io --qnn_context_binary_optimization_level 1",
)
DTYPE = torch.float32

SOC_TO_DEVICES = {
    "sm8750": ["Samsung Galaxy S25 (Family)", "Snapdragon 8 Elite QRD", "Snapdragon 8 Elite (Family)"],
}


def eprint(*a):
    print(*a, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- #
# text tower load + fp32 ONNX export
# --------------------------------------------------------------------------- #
def load_text_encoder(repo):
    from transformers import Qwen3VLForConditionalGeneration

    eprint(f"[load] text_encoder from {repo} (subfolder=text_encoder, bf16)")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        repo, subfolder="text_encoder", torch_dtype=torch.bfloat16
    )
    model.eval()
    qvlm = model.model
    if hasattr(qvlm, "visual"):
        del qvlm.visual  # reclaim vision-tower RAM; clip = text tower only
    return qvlm.language_model


# --------------------------------------------------------------------------- #
# mRoPE rotary embedding -> QNN-supported ops
#
# HF's Qwen3VLTextRotaryEmbedding computes cos/sin in-graph, but its
# apply_interleaved_mrope uses strided slice-assignment which the dynamo ONNX
# exporter lowers to ScatterND@18 / GatherNd -- neither accepted by QAIRT
# 2.45.0. We replicate the exact same math with only supported ops
# (MatMul + Concat + constant Gather + Cos/Sin + Transpose), verified
# bit-identical to HF (tmp/verify_mrope.py). position_ids [3,N] stays the
# graph input, so the C++ loader contract is unchanged.
# --------------------------------------------------------------------------- #
def build_mrope_gather_index(mrope_section, half):
    out = np.arange(3 * half).reshape(3, half)  # out[section, k] = section*half + k
    res = out[0].copy()
    idx_h = np.arange(1, mrope_section[1] * 3, 3)
    res[idx_h] = out[1][idx_h]
    idx_w = np.arange(2, mrope_section[2] * 3, 3)
    res[idx_w] = out[2][idx_w]
    return torch.from_numpy(res.astype(np.int64))  # [half]


def compute_mrope(position_ids, inv_freq, gather_idx):
    pos = position_ids.reshape(3, -1).to(torch.float32)          # [3, N]
    half = inv_freq.shape[0]
    inv4 = inv_freq.reshape(1, 1, -1, 1).to(torch.float32).expand(3, 1, -1, 1)
    pose = pos.unsqueeze(1).unsqueeze(2)                          # [3,1,1,N]
    freqs = torch.matmul(inv4, pose).transpose(2, 3)              # [3,1,N,half]
    f0, f1, f2 = freqs[0], freqs[1], freqs[2]
    stacked = torch.cat([f0, f1, f2], dim=-1)                     # [1,N,3*half]
    gi = gather_idx.reshape(1, 1, -1).expand(1, position_ids.shape[-1], -1)
    freqs_t = torch.gather(stacked, -1, gi)                       # [1,N,half]
    emb = torch.cat([freqs_t, freqs_t], dim=-1)                   # [1,N,2*half]
    return emb.cos(), emb.sin()


class MropeRotary(torch.nn.Module):
    def __init__(self, inv_freq, mrope_section):
        super().__init__()
        self.register_buffer("inv_freq", inv_freq.clone().to(torch.float32), persistent=False)
        half = inv_freq.shape[0]
        self.register_buffer(
            "gather_idx", build_mrope_gather_index(mrope_section, half), persistent=False
        )

    def forward(self, hidden_states, position_ids):
        c, s = compute_mrope(position_ids, self.inv_freq, self.gather_idx)
        return c.to(hidden_states.dtype), s.to(hidden_states.dtype)


class TextEncoderExport(torch.nn.Module):
    def __init__(self, lm):
        super().__init__()
        self.lm = lm
        # Swap the in-graph mRoPE (unsupported ScatterND@18/GatherNd) for the
        # faithful, QNN-supported-ops version above. position_ids [3,N] input
        # and the hidden output are unchanged.
        inv_freq = lm.rotary_emb.inv_freq.clone().to(torch.float32)
        mrope_section = list(lm.rotary_emb.mrope_section)
        lm.rotary_emb = MropeRotary(inv_freq, mrope_section)
        # Constant lower-triangular causal matrix [N,N]; combined in-graph with
        # the 1D attention_mask to build the 4D mask (see forward). Avoids
        # HF's create_causal_mask, which emits an unsupported GatherND.
        self.register_buffer(
            "causal_const", torch.tril(torch.ones(N, N, dtype=torch.float32)), persistent=False
        )

    def forward(self, input_embedding, position_ids, attention_mask):
        pos = position_ids.unsqueeze(1)      # [3,N] -> [3,1,N]
        # Build the 4D causal+padding mask in-graph from the 1D attention_mask
        # using only supported ops (Mul). HF's create_causal_mask would emit an
        # unsupported GatherND; passing a ready 4D mask makes it early-exit.
        am = attention_mask.reshape(1, N)    # [1,N]
        m4d = (self.causal_const * am)[None, None]  # [1,1,N,N]
        out = self.lm(
            inputs_embeds=input_embedding,
            position_ids=pos,
            attention_mask=m4d,
        )
        return out.last_hidden_state


def export_clip(onnx_path):
    lm = load_text_encoder(REPO)
    te = TextEncoderExport(lm).eval()
    # in-place fp32 cast (param-by-param .data reassignment) avoids a 2x peak
    with torch.no_grad():
        for p in te.parameters():
            if p.dtype != torch.float32:
                p.data = p.data.to(torch.float32)
    eprint("[cast] text_encoder -> fp32 (in-place)")

    dummy = (
        torch.zeros(1, N, H, dtype=DTYPE),
        torch.zeros(3, N, dtype=torch.int32),
        torch.ones(N, dtype=DTYPE),
    )
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    torch.onnx.export(
        te, dummy, onnx_path,
        input_names=["input_embedding", "position_ids", "attention_mask"],
        output_names=["hidden"],
        dynamic_axes=None, opset_version=int(os.environ.get("DREAMLITE_OPSET", "18")), do_constant_folding=True,
    )
    eprint(f"[onnx] graph={os.path.getsize(onnx_path)} bytes")
    eprint("[onnx] export returned; pinning ir_version")

    # pin ir_version 10 -> 8 (qai-hub rejects ir_version >= 10); 1-byte edit,
    # external weights in .data untouched.
    with open(onnx_path, "r+b") as f:
        head = f.read(2)
        if head[0] == 0x08 and head[1] == 0x0A:
            f.seek(1)
            f.write(b"\x08")
            eprint("[pin] ir_version 10 -> 8 (byte edit)")
        else:
            eprint(f"[pin] unexpected header {head.hex()}; left as-is")

    data = onnx_path + ".data"
    if os.path.exists(data):
        eprint(f"[onnx] weights={os.path.getsize(data)} bytes (external)")


def build_calibration():
    return {
        "input_embedding": [np.random.randn(1, N, H).astype(np.float32) for _ in range(NUM_CALIB)],
        "position_ids": [np.zeros((3, N), dtype=np.int32) for _ in range(NUM_CALIB)],
        "attention_mask": [np.ones(N, dtype=np.float32) for _ in range(NUM_CALIB)],
    }


# --------------------------------------------------------------------------- #
# qai_hub helpers
# --------------------------------------------------------------------------- #
def api_retry(fn, *a, tries=10, base=20, **k):
    import qai_hub
    import requests

    # qai_hub.client only exposes InternalError / RateLimitedError / UserError /
    # APIException / Error (no ReadTimeout); build the transient set from what
    # actually exists so a renamed/missing attr can never crash the retry setup.
    transient = set()
    for name in ("InternalError", "RateLimitedError", "APIException"):
        cls = getattr(qai_hub.client, name, None)
        if cls is not None:
            transient.add(cls)
    transient.add(ConnectionError)
    transient.add(TimeoutError)
    for rn in ("ReadTimeout", "ConnectionError", "Timeout"):
        cls = getattr(requests.exceptions, rn, None)
        if cls is not None:
            transient.add(cls)
    transient = tuple(transient)

    last = None
    for i in range(tries):
        try:
            return fn(*a, **k)
        except transient as e:
            last = e
            if i == tries - 1:
                break
            d = min(base * 2 ** i, 300)
            eprint(f"[retry] {getattr(fn, '__name__', fn)} transient {type(e).__name__}: {e}; wait {d}s")
            time.sleep(d)
        except Exception as e:
            eprint(f"[error] {getattr(fn, '__name__', fn)} fatal: {e}")
            raise
    raise last


def status_of(job):
    st = job.get_status()
    s = getattr(st, "state", None)  # qai_hub 0.52: JobStatus.state (State enum)
    if s is not None:
        return s
    s = getattr(st, "status", None)
    return s


def _statename(st):
    return getattr(st, "name", st)


def wait_done(job, label):
    st = status_of(job)
    while _statename(st) not in ("SUCCESS", "FAILED"):
        time.sleep(20)
        st = status_of(job)
    eprint(f"[{label}] status={_statename(st)} job={job.job_id}")
    return st


def resolve_device(client, soc):
    if os.environ.get("DREAMLITE_DEVICE"):
        return client.get_device(os.environ["DREAMLITE_DEVICE"])
    cands = SOC_TO_DEVICES.get(soc, [soc])
    avail = {d.name: d for d in client.get_devices()}
    for c in cands:
        if c in avail:
            return avail[c]
    raise RuntimeError(f"[device] no candidate for {soc}; available={list(avail)}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    import qai_hub

    token = os.environ.get("QAI_HUB_API_TOKEN")
    if not token:
        raise RuntimeError("QAI_HUB_API_TOKEN not set")
    client = qai_hub.Client(config=qai_hub.ClientConfig(token))

    onnx_dir = os.path.join(OUT, "clip_onnx")
    onnx_path = os.path.join(onnx_dir, "clip.onnx")
    bin_path = os.path.join(OUT, "clip.bin")
    os.makedirs(OUT, exist_ok=True)

    if os.environ.get("DREAMLITE_SKIP_EXPORT") in ("1", "true", "yes") and os.path.exists(onnx_path) and os.path.exists(onnx_path + ".data"):
        eprint("[export] SKIP (DREAMLITE_SKIP_EXPORT set, onnx present)")
    else:
        export_clip(onnx_path)
    calib = build_calibration()

    resume_quant = os.environ.get("DREAMLITE_QUANT_JOB")
    qjob_id = "<in-compile>"
    qmodel = None
    if resume_quant:
        eprint(f"[resume] reusing quantize job {resume_quant}")
        qjob = client.get_job(resume_quant)
        st = wait_done(qjob, "quant")
        if _statename(st) != "SUCCESS":
            raise RuntimeError(f"[quant] resume job {resume_quant} ended {_statename(st)}")
        qmodel = qjob.get_target_model()
        qjob_id = resume_quant
        device = resolve_device(client, TARGETS[0])
        eprint(f"[device] {device.name}")
    else:
        model = api_retry(client.upload_model, onnx_dir)
        eprint(f"[upload] model_id={model.model_id}")

        if PRECISION == "w4a16":
            w_dt, a_dt = qai_hub.QuantizeDtype.INT4, qai_hub.QuantizeDtype.INT16
        elif PRECISION == "w8a16":
            w_dt, a_dt = qai_hub.QuantizeDtype.INT8, qai_hub.QuantizeDtype.INT16
        else:
            raise RuntimeError(f"unknown precision {PRECISION}")

        device = resolve_device(client, TARGETS[0])
        eprint(f"[device] {device.name}")

        if not QUANT_IN_COMPILE:
            qjob = api_retry(
                client.submit_quantize_job,
                model=model, calibration_data=calib,
                weights_dtype=w_dt, activations_dtype=a_dt,
            )
            qjob_id = qjob.job_id
            eprint(f"[quant] job={qjob.job_id} ({w_dt}/{a_dt})")
            st = wait_done(qjob, "quant")
            if _statename(st) != "SUCCESS":
                raise RuntimeError(f"[quant] job {qjob.job_id} ended {_statename(st)}")
            qmodel = qjob.get_target_model()

    if QUANT_IN_COMPILE:
        opts = f"{COMPILE_OPTS} --quantize_full_type {PRECISION}"
        cjob = api_retry(
            client.submit_compile_job, model=model, input_specs=None,
            device=device, name="dreamlite_clip_w4a16_ptq",
            options=opts, calibration_data=calib,
        )
    else:
        cjob = api_retry(
            client.submit_compile_job, model=qmodel, input_specs=None,
            device=device, name="dreamlite_clip_compile", options=COMPILE_OPTS,
        )
    eprint(f"[compile] job={cjob.job_id}")
    st = wait_done(cjob, "compile")
    if _statename(st) != "SUCCESS":
        raise RuntimeError(f"[compile] job {cjob.job_id} ended {_statename(st)}")

    src = cjob.get_target_model().download(bin_path, timeout=200)
    shutil.copy(src, bin_path)
    eprint(f"[done] clip.bin -> {bin_path} ({os.path.getsize(bin_path)} bytes)")
    eprint(f"[jobs] compile_job={cjob.job_id} quant_job={qjob_id}")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.exit(1)
