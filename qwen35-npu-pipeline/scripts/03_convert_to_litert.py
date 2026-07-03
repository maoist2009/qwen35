#!/usr/bin/env python3
"""
把打过纽曼补丁的 Qwen3.5-4B 转成 LiteRT (.tflite)。

!!! 这是整条流水线里确定性最低的一步，务必先读完这段说明 !!!

LiteRT Torch（原 ai-edge-torch）的 Generative API 目前对 NPU 通道是 alpha
状态，且官方 building block 库里没有 GatedDeltaNet / 线性注意力的现成模板
——目前公开的示例只覆盖标准 Transformer（Gemma、TinyLlama、Llama 等标准
softmax attention）。Qwen3.5 是 75% GatedDeltaNet + 25% 标准 attention 的
混合架构，这意味着：

  路径 A（尝试自动导出）：先试 litert_torch 的 export_hf 工具，如果它内部
  已经支持 qwen3_5 / qwen3_next 的 model_type，可以直接吃 HF 权重转出来。
  这是最省事的路径，但不保证成功——如果 export_hf 不认识这个架构，会在
  图追踪(torch.export)阶段直接报错。

  路径 B（手写脚手架，大概率需要你接着改）：用 Generative API 的构建块
  （TransformerBlock / attention_utils）手工搭一个和 Qwen3.5 config 对齐
  的模型定义，再把 01/02 步产出的权重按 key 名映射进去。这个工作量不小，
  本脚本只生成骨架和权重映射表，具体每层的 forward 组装需要你对照
  transformers 里 Qwen3.5 的 modeling 源码来写。

脚本默认先尝试路径 A；失败后打印路径 B 的骨架文件位置，不会伪造一个假的
.tflite 文件出来。
"""
import argparse
import json
import sys
from pathlib import Path

SKELETON_TEMPLATE = '''\
"""
自动生成的手写转换骨架（路径 B）。
export_hf 无法直接识别当前模型架构时使用这个文件继续。

TODO（需要你手动完成，无法自动生成，因为依赖具体 transformers 源码版本）：
  1. 对照 transformers 里 Qwen3.5 的 modeling_qwen3_5.py，把每个
     DecoderLayer 拆成 "GatedDeltaNet 层" 和 "标准 Attention 层" 两类，
     按 config.layer_types 里的顺序（默认每 4 层 1 个 full_attention）
     组装。
  2. GatedDeltaNet 层的 recurrent/chunk-wise 计算，用
     litert_torch.generative 里能追踪、能导出的算子重写（矩阵乘、
     逐元素乘加、按本 pipeline 02 步补丁的纽曼级数求逆）——不要在这里
     再引入 fla 库的 Triton kernel，Triton 内核没法被 torch.export
     追踪导出成 LiteRT 图。
  3. 用 weight_map.json（本脚本已生成）把 safetensors 里的权重按 key
     加载进你手写的模块。
  4. 用 litert_torch.convert(model, sample_inputs, quant_config=...)
     导出 .tflite。

本文件只是占位骨架，不能直接运行。
"""
raise NotImplementedError(
    "路径 B 骨架未完成，请按上面 TODO 对照 transformers 源码手工实现。"
)
'''


def try_export_hf(model_dir: Path, out_path: Path, apply_patch_path: Path) -> bool:
    try:
        # 必须先执行补丁，让 fla / transformers 内的三角求逆函数被替换掉，
        # 再进行任何涉及该函数的 tracing / export。
        import runpy
        print(f"[03] 应用算子补丁: {apply_patch_path}")
        runpy.run_path(str(apply_patch_path))
    except Exception as e:
        print(f"[03][FATAL] 补丁应用失败，中止（不能在未打补丁的原始模型上继续导出）: {e}")
        sys.exit(1)

    try:
        from litert_torch.generative.utility import export_hf  # 包路径以实际版本为准
    except ImportError as e:
        print(f"[03] 未找到 litert_torch.generative.utility.export_hf（{e}），路径 A 不可用")
        return False

    try:
        print(f"[03] 尝试 export_hf({model_dir} -> {out_path})")
        export_hf.convert(
            checkpoint_path=str(model_dir),
            output_path=str(out_path),
            quantization_recipe="dynamic_wi8_afp32",  # 与论文 W4A16/W8A16 不同，先用官方默认recipe验证链路，
                                                        # 链路打通后再按论文 Appendix G 换成更激进的量化配置
        )
    except Exception as e:
        print(f"[03] export_hf 转换失败（大概率是架构不识别）: {e}")
        return False

    return out_path.exists()


def write_skeleton(model_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    skeleton_path = out_dir / "manual_convert_skeleton.py"
    skeleton_path.write_text(SKELETON_TEMPLATE)

    # 生成权重 key 列表，方便路径 B 手写时对照
    weight_map_path = out_dir / "weight_map.json"
    try:
        from safetensors import safe_open
        index_path = model_dir / "model.safetensors.index.json"
        if index_path.exists():
            keys = list(json.loads(index_path.read_text())["weight_map"].keys())
        else:
            with safe_open(str(model_dir / "model.safetensors"), framework="pt") as f:
                keys = list(f.keys())
        weight_map_path.write_text(json.dumps(sorted(keys), indent=2))
    except Exception as e:
        print(f"[03] 生成 weight_map.json 失败（不影响主流程报错逻辑）: {e}")

    print(f"[03][FATAL] 自动导出（路径 A）失败。已生成手写转换骨架：")
    print(f"            {skeleton_path}")
    print(f"            {weight_map_path}")
    print("            请按骨架文件里的 TODO 完成路径 B，再重新跑本脚本"
          "（或直接跑你写好的转换脚本产出同名 .tflite 文件）。")
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    apply_patch_path = args.model_dir / "apply_patch.py"
    if not apply_patch_path.exists():
        print(f"[03][FATAL] 找不到 {apply_patch_path}，说明 02 步没有正常跑完，中止。")
        sys.exit(1)

    ok = try_export_hf(args.model_dir, args.out, apply_patch_path)
    if not ok:
        write_skeleton(args.model_dir, args.out.parent / "manual_convert_skeleton")

    print(f"[03] 完成: {args.out}")


if __name__ == "__main__":
    main()
