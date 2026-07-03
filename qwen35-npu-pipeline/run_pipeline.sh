#!/usr/bin/env bash
# ============================================================================
# Qwen3.5-4B  ->  Neumann-inverse patched GatedDeltaNet  ->  LiteRT  ->  QNN(SM8850)
# ============================================================================
# 每一步都可以单独重跑（用 --from-step 指定），中间产物落盘，出错就停，
# 不会静默跳过、也不会用假数据继续往下走。
#
# 已知的不确定性（脚本里也会在对应步骤打印同样的警告）：
#   1. GatedDeltaNet 的三角求逆具体在 flash-linear-attention 的哪个函数里，
#      因 fla 版本而异；02 步会先做自动探测，探测不到会停下来让你手动确认。
#   2. LiteRT Torch (ai-edge-torch 继任者) 目前 NPU 通道是 alpha，且没有
#      GatedDeltaNet 的官方 building block；03 步生成的是可编辑脚手架，
#      大概率需要你根据实际 transformers 源码再调一轮。
#   3. QNN AOT 编译需要预先装好 Qualcomm QNN SDK 并设置 QNN_SDK_ROOT，
#      这个 SDK 只能去高通官网下载，无法 pip 安装，脚本只做存在性检查。
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="${WORKDIR:-$SCRIPT_DIR/work}"
mkdir -p "$WORKDIR"

# ---------- 可配置参数（也可以用环境变量覆盖）----------
BASE_MODEL_DIR="${BASE_MODEL_DIR:-}"          # 原始 Qwen3.5-4B safetensors 目录（必填）
FT_GGUF_PATH="${FT_GGUF_PATH:-}"              # 别人追加训练导出的 fp16 GGUF（选填，不填则跳过回代）
OUTPUT_DIR="${OUTPUT_DIR:-$WORKDIR/output}"
CHUNK_SIZE="${CHUNK_SIZE:-64}"                # 论文验证过 64 最稳；128 会数值溢出，脚本会拦截
NEUMANN_N="${NEUMANN_N:-3}"                   # 截断阶数
RESIDUAL_S="${RESIDUAL_S:-8}"                 # 残差修正阶数
QNN_TARGET_SOC="${QNN_TARGET_SOC:-SM8850}"    # 8 Elite Gen5
FROM_STEP="${FROM_STEP:-0}"

mkdir -p "$OUTPUT_DIR"

log() { echo -e "\n\033[1;36m[pipeline] $*\033[0m"; }
die() { echo -e "\033[1;31m[FATAL] $*\033[0m" >&2; exit 1; }

[[ -z "$BASE_MODEL_DIR" ]] && die "必须设置 BASE_MODEL_DIR（原始 Qwen3.5-4B safetensors 目录）"
[[ ! -d "$BASE_MODEL_DIR" ]] && die "BASE_MODEL_DIR 不存在: $BASE_MODEL_DIR"

if [[ "$CHUNK_SIZE" -gt 64 && "$NEUMANN_N" -ge 3 ]]; then
  die "chunk_size=$CHUNK_SIZE 且 N=$NEUMANN_N 时纽曼级数高阶项会在 FP16/INT16 溢出" \
      "（论文附录实测 128x128/N=3 最坏情况达到 341376）。请把 CHUNK_SIZE 降到 64，" \
      "或者自行在 02 脚本里加 block-wise 拆分。"
fi

STEP=0
run_step () {
  STEP=$((STEP + 1))
  if [[ "$STEP" -lt "$FROM_STEP" ]]; then
    log "跳过 step $STEP（FROM_STEP=$FROM_STEP）: $1"
    return
  fi
  log "step $STEP: $1"
  shift
  "$@"
}

run_step "环境检查与依赖安装" \
  bash "$SCRIPT_DIR/scripts/00_setup_env.sh"

if [[ "${AUTO_ACQUIRE_QNN_SDK:-0}" == "1" ]]; then
  [[ -z "${QSC_EMAIL:-}" || -z "${QSC_PASSWORD:-}" ]] && \
    die "AUTO_ACQUIRE_QNN_SDK=1 需要先 export QSC_EMAIL / QSC_PASSWORD"
  run_step "自动获取 QNN SDK (qsc-cli)" \
    bash "$SCRIPT_DIR/scripts/00b_acquire_qnn_sdk.sh"
  # shellcheck disable=SC1091
  [[ -f "$HOME/litert_workspace/.env" ]] && source "$HOME/litert_workspace/.env"
else
  log "AUTO_ACQUIRE_QNN_SDK!=1，假设你已经手动设置了 QNN_SDK_ROOT" \
      "（或者单独跑一次 scripts/00b_acquire_qnn_sdk.sh 再 source 它的 .env）"
fi

MERGED_MODEL_DIR="$WORKDIR/merged_safetensors"
if [[ -n "$FT_GGUF_PATH" ]]; then
  run_step "把追加训练的 fp16 GGUF 回代合并进 safetensors" \
    python3 "$SCRIPT_DIR/scripts/01_gguf_to_safetensors.py" \
      --gguf "$FT_GGUF_PATH" \
      --reference "$BASE_MODEL_DIR" \
      --out "$MERGED_MODEL_DIR"
else
  log "未提供 FT_GGUF_PATH，跳过 GGUF 回代，直接使用 BASE_MODEL_DIR"
  MERGED_MODEL_DIR="$BASE_MODEL_DIR"
fi

PATCHED_MODEL_DIR="$WORKDIR/patched_model"
run_step "修补 GatedDeltaNet 矩阵求逆算子（纽曼级数截断+掩码+残差修正）" \
  python3 "$SCRIPT_DIR/scripts/02_patch_gdn_neumann.py" \
    --model-dir "$MERGED_MODEL_DIR" \
    --out "$PATCHED_MODEL_DIR" \
    --chunk-size "$CHUNK_SIZE" \
    --neumann-n "$NEUMANN_N" \
    --residual-s "$RESIDUAL_S"

TFLITE_PATH="$WORKDIR/qwen35_4b_patched.tflite"
run_step "转换为 LiteRT (.tflite)，MatMul-only 算子应全部落在 QNN 支持集合内" \
  python3 "$SCRIPT_DIR/scripts/03_convert_to_litert.py" \
    --model-dir "$PATCHED_MODEL_DIR" \
    --out "$TFLITE_PATH"

run_step "QNN AOT 编译到 $QNN_TARGET_SOC，产出可部署包" \
  python3 "$SCRIPT_DIR/scripts/04_qnn_aot_compile.py" \
    --tflite "$TFLITE_PATH" \
    --soc "$QNN_TARGET_SOC" \
    --out "$OUTPUT_DIR"

log "全部完成，产物在: $OUTPUT_DIR"
