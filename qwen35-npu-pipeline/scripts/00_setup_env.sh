#!/usr/bin/env bash
# 安装/检查依赖。分两类：
#   (a) pip 能装的（gguf 反量化、transformers、litert-torch、ai-edge-litert）
#   (b) 必须手动装的（Qualcomm QNN SDK —— 不在任何公开 pip/npm 源上，
#       需要登录 https://qpm.qualcomm.com 或高通开发者官网下载）
set -euo pipefail

log() { echo -e "\033[1;36m[setup] $*\033[0m"; }
warn() { echo -e "\033[1;33m[warn]  $*\033[0m"; }

log "检查 Python / pip"
command -v python3 >/dev/null || { echo "缺少 python3"; exit 1; }
python3 -m pip --version >/dev/null || { echo "缺少 pip"; exit 1; }

log "安装 GGUF/权重处理相关依赖"
python3 -m pip install --break-system-packages -q -U \
  "transformers>=4.48" \
  "gguf>=0.13.0" \
  "safetensors" \
  "accelerate" \
  "sentencepiece" \
  "numpy"

log "安装 LiteRT / AI Edge 转换相关依赖（包名可能随版本变化，失败请查 https://ai.google.dev/edge/litert 确认最新包名）"
python3 -m pip install --break-system-packages -q -U \
  "ai-edge-litert" \
  "litert-torch" \
  "ai-edge-quantizer" \
  || warn "litert 系包安装失败：这些是活跃开发中的库，包名/可用性可能已变化，建议改从 https://github.com/google-ai-edge/litert-torch 源码装"

log "检查 Qualcomm QNN SDK（这个必须手动下载，脚本不负责安装）"
if [[ -z "${QNN_SDK_ROOT:-}" ]]; then
  warn "未设置 QNN_SDK_ROOT 环境变量。"
  warn "请前往 Qualcomm Package Manager (QPM) 下载 QNN SDK，安装后执行："
  warn '    export QNN_SDK_ROOT=/path/to/qairt/<version>'
  warn "04 步（QNN AOT 编译）会在缺失该变量时直接报错退出，而不是假装成功。"
else
  [[ -d "$QNN_SDK_ROOT" ]] || warn "QNN_SDK_ROOT=$QNN_SDK_ROOT 但目录不存在，请检查路径"
  log "QNN_SDK_ROOT = $QNN_SDK_ROOT"
fi

log "环境检查完成"
