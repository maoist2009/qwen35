#!/usr/bin/env python3
"""
把 patches/gdn_neumann_patch.py 里的 MatMul-only 近似求逆，接到 Qwen3.5
GatedDeltaNet 实际调用三角求逆的位置。

!!! 关于"自动探测"的诚实说明 !!!
GatedDeltaNet 的 chunk-wise 三角求逆，具体落在哪个函数里，取决于 Qwen3.5
是用 transformers 自带的 modeling_qwen3_5.py 实现，还是走
flash-linear-attention (fla) 库的 fla.ops.gated_delta_rule.chunk 路径。
这两条路径在不同 transformers/fla 版本里都出现过，且函数名不保证稳定
（我见过 solve_tril / chunk_scaled_dot_kkt_fwd / attn_inv 等不同命名）。

所以这个脚本不会"假装"自动打对了补丁——它会：
  1. 尝试几个已知的候选导入路径；
  2. 打印它找到的函数签名和源码前几行，让你确认这确实是三角求逆；
  3. 只有你确认后才真正打补丁（或者用 --yes 跳过确认，用于 CI/非交互场景，
     但强烈建议第一次手动跑一遍确认）；
  4. 如果一个候选都没找到，直接报错退出并打印手动打补丁的位置提示，
     而不是静默放过、产出一个其实没被加速的模型。

补丁本身不修改权重（safetensors 原样拷贝），只是把 PATCHED_MODEL_DIR 里
放一份 conversion_recipe.json + 一个 apply_patch.py 入口，03 步转换前会
先 import 这个入口来完成猴子补丁，再做 torch.export。
"""
import argparse
import importlib
import inspect
import json
import shutil
import sys
from pathlib import Path

CANDIDATE_TARGETS = [
    ("fla.ops.utils.solve_tril", "solve_tril"),
    ("fla.ops.gated_delta_rule.chunk", "chunk_scaled_dot_kkt_fwd"),
    ("fla.ops.common.chunk_h", "chunk_fwd_h"),
    ("transformers.models.qwen3_5.modeling_qwen3_5", "solve_tril"),
    ("transformers.models.qwen3_next.modeling_qwen3_next", "solve_tril"),
]


def find_candidates():
    found = []
    for mod_name, attr_name in CANDIDATE_TARGETS:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        if hasattr(mod, attr_name):
            fn = getattr(mod, attr_name)
            try:
                src = inspect.getsource(fn)
            except (OSError, TypeError):
                src = "<无法获取源码>"
            found.append((mod_name, attr_name, src))
    return found


def write_patch_entrypoint(out_dir: Path, mod_name: str, attr_name: str,
                            neumann_n: int, residual_s: int, chunk_size: int):
    entry = out_dir / "apply_patch.py"
    entry.write_text(f'''"""
自动生成：在模型 export/转换之前 import 这个文件即可完成猴子补丁。
目标: {mod_name}.{attr_name}
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "patches"))
from gdn_neumann_patch import neumann_chunk_inverse
import {mod_name} as _target_mod

_ORIG = _target_mod.{attr_name}

def _patched_solve(A, *args, **kwargs):
    # 注意：原函数签名可能带额外参数（比如 output_dtype、cu_seqlens 等），
    # 这里透传 *args/**kwargs 但不使用，如果原函数依赖这些参数控制数值行为，
    # 需要你按实际签名调整。这是脚手架，不是保证正确的最终代码。
    return neumann_chunk_inverse(
        A,
        neumann_n={neumann_n},
        residual_s={residual_s},
        chunk_size_check={chunk_size},
    )

_target_mod.{attr_name} = _patched_solve
print(f"[apply_patch] 已将 {mod_name}.{attr_name} 替换为纽曼级数近似 "
      f"(N={neumann_n}, S={residual_s}, chunk_size<= {chunk_size})")
''')
    print(f"[02] 已生成补丁入口: {entry}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--chunk-size", type=int, default=64)
    ap.add_argument("--neumann-n", type=int, default=3)
    ap.add_argument("--residual-s", type=int, default=8)
    ap.add_argument("--yes", action="store_true", help="非交互模式，自动选第一个候选")
    args = ap.parse_args()

    if not args.model_dir.exists():
        print(f"[02][FATAL] 模型目录不存在: {args.model_dir}")
        sys.exit(1)

    print("[02] 探测三角求逆函数的可能位置...")
    candidates = find_candidates()
    if not candidates:
        print("[02][FATAL] 一个候选都没找到。请手动定位 Qwen3.5 GatedDeltaNet 实现里")
        print("            做 chunk-wise 三角求逆的函数（通常在 fla 库或")
        print("            transformers 的 modeling_qwen3_5.py / modeling_qwen3_next.py")
        print("            里搜索 'solve_triangular' / 'forward substitution' / ")
        print("            'tril' 关键字），然后把 patches/gdn_neumann_patch.py 里的")
        print("            neumann_chunk_inverse 手动接进去。")
        sys.exit(1)

    print(f"[02] 找到 {len(candidates)} 个候选：")
    for i, (mod_name, attr_name, src) in enumerate(candidates):
        print(f"  [{i}] {mod_name}.{attr_name}")
        preview = "\n".join(src.splitlines()[:8])
        print(f"      源码前几行:\n{preview}\n      ...")

    if args.yes:
        choice = 0
    else:
        raw = input(f"选择要打补丁的候选序号 [0-{len(candidates)-1}] (回车默认0，Ctrl+C 取消): ").strip()
        choice = int(raw) if raw else 0

    mod_name, attr_name, _ = candidates[choice]

    print(f"[02] 拷贝模型目录 {args.model_dir} -> {args.out}")
    if args.out.exists():
        shutil.rmtree(args.out)
    shutil.copytree(args.model_dir, args.out)

    write_patch_entrypoint(args.out, mod_name, attr_name,
                            args.neumann_n, args.residual_s, args.chunk_size)

    recipe = {
        "target_module": mod_name,
        "target_attr": attr_name,
        "neumann_n": args.neumann_n,
        "residual_s": args.residual_s,
        "chunk_size": args.chunk_size,
        "source_paper": "arXiv:2606.06034 (reproduced independently, not official code)",
    }
    (args.out / "conversion_recipe.json").write_text(json.dumps(recipe, indent=2, ensure_ascii=False))
    print(f"[02] 完成。补丁配置写入 {args.out / 'conversion_recipe.json'}")
    print("[02] 提醒：03 步转换脚本必须在 import 模型代码之前先 import 这个 apply_patch.py，"
          "否则补丁不会生效，导出的还是原始前向替换求逆版本。")


if __name__ == "__main__":
    main()
