#!/usr/bin/env python3
"""
用 ai_edge_litert.aot 把 .tflite 编译成针对 Qualcomm SoC 的 QNN 二进制包。
需要预先装好 Qualcomm QNN SDK 并设置 QNN_SDK_ROOT（见 00_setup_env.sh）。

参考: Google AI Edge 官方博客
"Unlocking Peak Performance on Qualcomm NPU with LiteRT"
https://developers.googleblog.com/unlocking-peak-performance-on-qualcomm-npu-with-litert/
"""
import argparse
import glob
import os
import sys
from pathlib import Path

# Snapdragon 8 Elite (上一代) = SM8750；Snapdragon 8 Elite Gen 5 = SM8850。
# 两代芯片、两个不同 SoC 型号，别混用——用之前先确认你的目标设备到底是哪一代。


def _setup_qairt_env(qairt_root: Path):
    """按参考脚本的做法探测 arch 子目录，把 bin/lib 加进 PATH/LD_LIBRARY_PATH。"""
    os.environ["QAIRT_ROOT"] = str(qairt_root)
    for arch_subdir in ("x86_64-linux-clang", "x86_64-linux-ubuntu"):
        if (qairt_root / "bin" / arch_subdir).exists():
            os.environ["PATH"] = f"{qairt_root / 'bin' / arch_subdir}:{os.environ.get('PATH', '')}"
            os.environ["LD_LIBRARY_PATH"] = f"{qairt_root / 'lib' / arch_subdir}:{os.environ.get('LD_LIBRARY_PATH', '')}"
            print(f"[04] QAIRT arch 子目录: {arch_subdir}")
            return
    print(f"[04][WARN] 在 {qairt_root}/bin 下没找到已知的 arch 子目录（探测了 "
          f"x86_64-linux-clang / x86_64-linux-ubuntu），如果编译报找不到工具链，"
          f"手动检查实际目录名并对照修改本函数。")


def _check_error_logs():
    """QNN 底层编译失败经常不让 Python 侧抛异常，而是在 /tmp 留错误日志，
    这里显式抓出来打印，避免'看起来跑完了其实是失败的'。"""
    error_files = glob.glob("/tmp/*.error")
    if not error_files:
        return
    print("\n[04] 检测到 /tmp 下的编译错误日志：")
    for ef in error_files:
        print(f"--- {ef} ---")
        try:
            print(Path(ef).read_text()[:4000])
        except OSError:
            print("(无法读取)")
    print("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite", required=True, type=Path)
    ap.add_argument("--soc", default="SM8850",
                     help="目标 SoC 型号（SocModel 枚举名）。8 Elite(上一代)=SM8750，"
                          "8 Elite Gen5=SM8850，先确认清楚设备是哪一代再填。")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    if not args.tflite.exists():
        print(f"[04][FATAL] 找不到 .tflite 文件: {args.tflite}")
        print("            说明 03 步没有产出真实模型（可能落到了手写骨架分支），")
        print("            请先完成 03 步再来跑这一步。")
        sys.exit(1)
    if args.tflite.stat().st_size == 0:
        print(f"[04][FATAL] .tflite 文件存在但大小为 0 字节: {args.tflite}，说明 03 步是假成功，中止。")
        sys.exit(1)

    qnn_sdk_root = os.environ.get("QNN_SDK_ROOT")
    if not qnn_sdk_root:
        print("[04][FATAL] 未设置 QNN_SDK_ROOT。")
        print("            可以手动去 https://qpm.qualcomm.com/ 下载，")
        print("            也可以跑 scripts/00b_acquire_qnn_sdk.sh 自动获取，")
        print("            然后 source $WORKDIR/.env 把变量带进当前 shell。")
        sys.exit(1)
    _setup_qairt_env(Path(qnn_sdk_root))

    try:
        from ai_edge_litert.aot import aot_compile as aot_lib
        from ai_edge_litert.aot.vendors.qualcomm import target as qnn_target
    except ImportError as e:
        print(f"[04][FATAL] 无法导入 ai_edge_litert.aot（{e}）。")
        print("            这个包名/API 可能随版本变化，请核对：")
        print("            https://ai.google.dev/edge/litert")
        sys.exit(1)

    soc_enum = getattr(qnn_target.SocModel, args.soc, None)
    if soc_enum is None:
        available = [x for x in dir(qnn_target.SocModel) if not x.startswith("_")]
        print(f"[04][FATAL] SoC 型号 {args.soc} 不在 qnn_target.SocModel 里。")
        print(f"            可用选项: {available}")
        sys.exit(1)

    args.out.mkdir(parents=True, exist_ok=True)
    target = qnn_target.Target(soc_enum)

    print(f"[04] AOT 编译 {args.tflite} -> target={args.soc} ...")
    print("[04] 这个过程可能需要 30 分钟到 1 小时，取决于模型大小和机器性能。")
    try:
        compiled_models = aot_lib.aot_compile(str(args.tflite), output_dir=str(args.out), target=[target])
    except Exception as e:
        print(f"[04][WARN] aot_compile 抛出异常，但不代表一定没产出任何文件，继续做产物校验: {e}")
        compiled_models = None

    # 高通底层编译失败经常不通过 Python 异常传上来，而是留错误日志在 /tmp，
    # 必须显式检查，不能只信 Python 侧“没报错”。
    _check_error_logs()

    # 落盘 + 零字节校验（"假成功"检测：文件存在但是空的，说明模型里有 NPU
    # 不认识的算子——如果你看到这里报错，先确认 02 步的补丁是不是真的生效了，
    # 是不是还有别的算子没被 QNN 的约 90 个支持算子覆盖到）。
    if compiled_models is not None:
        try:
            for i, m in enumerate(compiled_models):
                dst = args.out / f"qwen35_4b_{args.soc}_{i}.bin"
                data = getattr(m, "serialized_bytes", None) or getattr(m, "bytes", None)
                if data is None:
                    print(f"[04][WARN] 无法从编译结果中提取字节数据，返回对象: {m!r}")
                    continue
                dst.write_bytes(data)
                print(f"[04] 写出: {dst}")
        except TypeError:
            print(f"[04][WARN] compiled_models 不是预期的可迭代结构，原样打印供排查: {compiled_models!r}")

    produced = list(args.out.glob("*.tflite")) + list(args.out.glob("*.bin"))
    valid = [f for f in produced if f.stat().st_size > 0]
    if not valid:
        print(f"[04][FATAL] AOT 编译实际失败：{args.out} 下没有任何非空产物文件。")
        print("            这通常意味着模型图里还有 NPU 不支持的算子（比如没打上")
        print("            纽曼补丁的 GatedDeltaNet 三角求逆），回去检查 02/03 步。")
        sys.exit(1)

    print(f"[04] 完成，有效产物: {[str(f) for f in valid]}")
    print("[04] 后续：把产物和原始 .tflite 一起打包进 Google Play AI Pack，")
    print("            或直接用 LiteRT-LM / Compiled Model API 在设备上加载测试。")


if __name__ == "__main__":
    main()
