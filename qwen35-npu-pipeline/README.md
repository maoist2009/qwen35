# Qwen3.5-4B → Neumann 补丁 GatedDeltaNet → LiteRT → QNN(8 Elite) 一条龙脚本

## 目录结构
```
qwen35-npu-pipeline/
├── run_pipeline.sh              # 总入口
├── scripts/
│   ├── 00_setup_env.sh          # 装依赖 + 检查 QNN SDK（手动获取路径）
│   ├── 00b_acquire_qnn_sdk.sh   # (可选) 用 qsc-cli 自动获取 QNN SDK
│   ├── 01_gguf_to_safetensors.py# (可选) fp16 GGUF 回代合并
│   ├── 02_patch_gdn_neumann.py  # 探测并接入纽曼级数求逆补丁
│   ├── 03_convert_to_litert.py  # 转 .tflite（不确定性最高的一步）
│   └── 04_qnn_aot_compile.py    # QNN AOT 编译到 SM8850
└── patches/
    └── gdn_neumann_patch.py     # 纽曼级数近似求逆的核心数学实现（已自测）
```

## 快速开始

**方式一：手动装好 QNN SDK**
```bash
export BASE_MODEL_DIR=/path/to/Qwen3.5-4B-safetensors
export FT_GGUF_PATH=/path/to/someone_finetuned.fp16.gguf   # 没有就不设置
export QNN_SDK_ROOT=/path/to/qairt/<version>                # 手动去 qpm.qualcomm.com 下载装好
bash run_pipeline.sh
```

**方式二：用 `00b` 自动获取 QNN SDK（基于第三方 qsc-cli 工具，见下方安全说明）**
```bash
export BASE_MODEL_DIR=/path/to/Qwen3.5-4B-safetensors
export QSC_EMAIL=you@example.com
export QSC_PASSWORD='...'          # 只走环境变量，绝对不要写进脚本/仓库
export AUTO_ACQUIRE_QNN_SDK=1
export QNN_TARGET_SOC=SM8850        # 8 Elite Gen5；上一代 8 Elite 用 SM8750
bash run_pipeline.sh
```

出错会在对应 step 停下（`set -euo pipefail`），可以用 `FROM_STEP=N bash run_pipeline.sh` 从某一步续跑。

**关于 `00b_acquire_qnn_sdk.sh` 的信任边界**：它依赖的 `qsc-cli.deb` 来自个人维护的
GitHub release（`maoist2009/qwen35`），不是高通官方分发渠道。这条自动化路径省事，
但本质是在信任一个第三方打包的二进制会被 `sudo dpkg -i` 装到系统里。如果你介意，
用方式一，自己去 `qpm.qualcomm.com` 走官方下载。

## 我能保证 / 不能保证什么

**已验证、有把握的部分：**
- `patches/gdn_neumann_patch.py` 里的纽曼级数截断+对角掩码+并行残差修正算法，
  我在本机跑了数值自测（对比 `torch.linalg.solve_triangular` 精确解），
  chunk=64、N=3、S=8 时相对误差在 1e-5～1e-7 量级，溢出保护（chunk>64 时拒绝执行）
  也验证生效。这部分代码可以直接用。
- GGUF 反向合并（01）、QNN AOT 编译调用方式（04）、环境依赖安装（00），
  都是照着查到的官方文档 / 现成工具（`dreamfast/ungguf`、Google AI Edge 官方博客）
  写的，命令和 API 调用方式是真实存在的。

**没有把握、需要你在真机/真环境上继续调的部分：**
1. **02 步的"自动探测三角求逆函数位置"**：不同 transformers/fla 版本里，
   GatedDeltaNet 具体在哪个函数做三角求逆，命名不保证稳定。脚本会打印候选
   源码让你确认，找不到会直接报错退出，不会静默放过。
2. **03 步是整条链路最薄弱的环节**：LiteRT Torch 的 Generative API 对 NPU
   还是 alpha 状态，且没有 GatedDeltaNet 的官方 building block。脚本先尝试
   `export_hf` 自动导出，失败后会生成一个手写转换骨架
   （`manual_convert_skeleton/`），但骨架本身标了 `NotImplementedError`，
   需要你对照当时的 `transformers` 源码把每层手工组装出来——这部分工作量
   我没法代劳，因为依赖你实际环境里的库版本、config 字段名。
3. **论文本身没有开源代码**，`gdn_neumann_patch.py` 是我按论文公式独立复现的，
   不是作者原始实现，数值行为方向一致但没有和官方代码做过 bit-level 对比。

## 建议的验证顺序
1. 先跑 `python3 patches/gdn_neumann_patch.py` 确认自测通过（已经帮你测过一次了）。
2. 有真机/开发环境后，先只做 00+02，在 CPU 上跑一次原始 vs 补丁后的模型做
   困惑度对比，确认补丁没有明显精度损失，再往下走 03/04。
3. 03 步大概率要花时间手写，建议先拿一个小很多的、非 MoE 的验证模型
   （比如同架构但更小的 Qwen3.5-0.8B）跑通链路，再上 4B。
