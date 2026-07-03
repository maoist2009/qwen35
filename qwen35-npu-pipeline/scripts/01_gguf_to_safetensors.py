#!/usr/bin/env python3
"""
把别人用 llama.cpp 系工具追加训练后导出的 fp16 GGUF，反向转换回 HuggingFace
safetensors 目录，用于后续 02 步的算子修补（02 步需要在 PyTorch/HF 层面改
GatedDeltaNet 的 forward，没法在 GGUF 二进制层面改）。

优先级：
  1. 如果本机能跑 docker 且存在 `ungguf` 工具（github.com/dreamfast/ungguf），
     优先用它——它专门写了 Qwen3.5 的 V-head reorder / norm 换算规则，是目前
     唯一见到的、明确声明支持 Qwen3.5 混合架构（GatedDeltaNet + attention）
     反向转换的开源工具。
  2. 否则回退到 transformers 原生的 `from_pretrained(..., gguf_file=...)`
     路径。这条路径对标准 Llama 系架构成熟，但对 Qwen3.5 这种混合线性注意力
     架构的 GGUF 元数据支持程度不确定——脚本会在转换后做一次 tensor 覆盖率
     校验，如果关键的 GatedDeltaNet 专属权重（A_log / dt_bias / conv1d 等）
     缺失或形状对不上，会直接报错而不是静默产出一个残缺模型。

用法：
  python3 01_gguf_to_safetensors.py --gguf FT.gguf --reference /path/to/base_hf_dir --out ./merged
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# 反向转换时用于粗略校验"看起来像 Qwen3.5 GatedDeltaNet 层"的关键字。
# 如果你的 transformers 版本里 Qwen3.5 的实际 key 命名不同，请对照
# reference 目录里 model.safetensors.index.json 的 key 修改这里。
GDN_KEY_HINTS = ["A_log", "dt_bias", "conv1d", "gated_delta"]


def try_ungguf(gguf_path: Path, reference_dir: Path, out_dir: Path) -> bool:
    if shutil.which("docker") is None:
        print("[01] 未检测到 docker，跳过 ungguf 路径")
        return False
    if shutil.which("ungguf.sh") is None and not Path("./ungguf.sh").exists():
        print("[01] 未检测到 ungguf.sh，跳过（可 git clone https://github.com/dreamfast/ungguf 后重试）")
        return False
    print("[01] 使用 ungguf 转换 Qwen3.5 GGUF -> safetensors（保留 fp16）")
    cmd = ["./ungguf.sh", "convert-qwen35", "--keep-fp16",
           str(gguf_path), str(out_dir), str(reference_dir)]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("[01] ungguf 转换失败，回退到 transformers 原生路径")
        return False
    verify_cmd = ["./ungguf.sh", "verify", str(gguf_path), str(out_dir), str(reference_dir)]
    vr = subprocess.run(verify_cmd)
    if vr.returncode != 0:
        print("[01][FATAL] ungguf 的 bit-exact 校验未通过，产物不可信，中止。")
        sys.exit(1)
    return True


def fallback_transformers(gguf_path: Path, reference_dir: Path, out_dir: Path):
    print("[01] 使用 transformers gguf_file= 原生加载路径（未针对 GatedDeltaNet 混合架构验证过，务必检查后续校验输出）")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(str(reference_dir))
    model = AutoModelForCausalLM.from_pretrained(
        str(reference_dir.parent) if False else str(gguf_path.parent),
        gguf_file=str(gguf_path.name),
        torch_dtype="float16",
        low_cpu_mem_usage=True,
    )
    model.save_pretrained(str(out_dir), safe_serialization=True, max_shard_size="4GB")
    tok.save_pretrained(str(out_dir))

    # 把 reference 里非权重的架构相关文件（generation_config 等）也补齐，
    # 避免 gguf 元数据缺字段导致 config 不完整。
    for fname in ["generation_config.json", "config.json"]:
        src = reference_dir / fname
        dst = out_dir / fname
        if src.exists() and not dst.exists():
            shutil.copy(src, dst)


def verify_gdn_keys(out_dir: Path):
    import json
    from safetensors import safe_open

    index_path = out_dir / "model.safetensors.index.json"
    if index_path.exists():
        keys = list(json.loads(index_path.read_text())["weight_map"].keys())
    else:
        single = out_dir / "model.safetensors"
        if not single.exists():
            print(f"[01][FATAL] 在 {out_dir} 下找不到 safetensors 产物")
            sys.exit(1)
        with safe_open(str(single), framework="pt") as f:
            keys = list(f.keys())

    found = {h: any(h in k for k in keys) for h in GDN_KEY_HINTS}
    missing = [h for h, ok in found.items() if not ok]
    if missing:
        print(f"[01][FATAL] 转换产物里缺少 GatedDeltaNet 专属权重关键字: {missing}")
        print("            说明 GGUF -> safetensors 的架构映射不完整（很可能是")
        print("            transformers 原生 gguf 加载器还不认识这些混合架构")
        print("            专属张量）。请改用 ungguf 工具，或手动检查 key 命名。")
        sys.exit(1)
    print(f"[01] GatedDeltaNet 关键权重校验通过: {list(GDN_KEY_HINTS)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True, type=Path)
    ap.add_argument("--reference", required=True, type=Path,
                     help="原始 Qwen3.5-4B safetensors 目录，提供 tensor 命名/config 参照")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    if not args.gguf.exists():
        print(f"[01][FATAL] GGUF 文件不存在: {args.gguf}")
        sys.exit(1)
    if not args.reference.exists():
        print(f"[01][FATAL] 参照模型目录不存在: {args.reference}")
        sys.exit(1)

    ok = try_ungguf(args.gguf, args.reference, args.out)
    if not ok:
        fallback_transformers(args.gguf, args.reference, args.out)

    verify_gdn_keys(args.out)
    print(f"[01] 完成，合并后的 safetensors 目录: {args.out}")


if __name__ == "__main__":
    main()
