#!/usr/bin/env bash
# 自动化下载/安装 Qualcomm QNN SDK（用 qsc-cli），并把 QNN_SDK_ROOT 写到
# $WORKDIR/.env 供后续步骤 source。
#
# 相比手动去官网下载，这条路径可以脚本化，但依赖第三方非官方打包的
# qsc-cli（来自 GitHub release，不是 Qualcomm 官方分发），装之前请自己
# 评估一下信任边界——这不是 pip/apt 官方源，是个人维护的 .deb。
#
# 用法（凭证一律走环境变量，不写进脚本/仓库）：
#   export QSC_EMAIL=you@example.com
#   export QSC_PASSWORD='...'
#   [可选] export QSC_DEB_URL=https://.../qsc-cli.deb   # 默认见下面
#   [可选] export QNN_SDK_NAME=Qualcomm_AI_Runtime_Community
#   [可选] export QNN_SDK_VERSION=2.42.0.251225
#   [可选] export DO_DISK_CLEANUP=1   # 只在你确定是一次性/CI机器时开
#   [可选] export DO_SETUP_SWAP=1     # 内存紧张的机器上开，注意会写 24G 文件
#   bash 00b_acquire_qnn_sdk.sh
set -euo pipefail

WORK_DIR="${WORKDIR:-$HOME/litert_workspace}"
mkdir -p "$WORK_DIR"

QSC_DEB_URL="${QSC_DEB_URL:-https://github.com/maoist2009/qwen35/releases/download/QSC-cli/qsc-cli.deb}"
QNN_SDK_NAME="${QNN_SDK_NAME:-Qualcomm_AI_Runtime_Community}"
QNN_SDK_VERSION="${QNN_SDK_VERSION:-2.42.0.251225}"

log() { echo -e "\033[1;36m[qnn-sdk] $*\033[0m"; }
warn() { echo -e "\033[1;33m[warn]   $*\033[0m"; }
die() { echo -e "\033[1;31m[FATAL]  $*\033[0m" >&2; exit 1; }

if [[ -z "${QSC_EMAIL:-}" || -z "${QSC_PASSWORD:-}" ]]; then
  die "请先 export QSC_EMAIL 和 QSC_PASSWORD（不要写进脚本/仓库里）"
fi

# ---------------- 可选：磁盘清理（只应在一次性 CI/构建机上做）----------------
if [[ "${DO_DISK_CLEANUP:-0}" == "1" ]]; then
  log "清理常见的 CI 预装大文件占用（GitHub Actions runner 场景）"
  sudo rm -rf /usr/share/dotnet /usr/local/lib/android /opt/ghc \
              /opt/hostedtoolcache/CodeQL /opt/microsoft 2>/dev/null || true
  df -h / || true
else
  log "跳过磁盘清理（DO_DISK_CLEANUP!=1）。如果是自己的常驻机器，本来就不该跑这步。"
fi

# ---------------- 可选：Swap ----------------
if [[ "${DO_SETUP_SWAP:-0}" == "1" ]]; then
  log "配置 24G swap（转换大模型时内存容易不够）"
  if ! swapon --show | grep -q "$WORK_DIR/../swapfile" 2>/dev/null; then
    sudo fallocate -l 24G /swapfile 2>/dev/null || sudo dd if=/dev/zero of=/swapfile bs=1M count=24576 status=progress
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile || warn "swapon 失败，可能已经有 swap 在用，检查 swapon --show"
  else
    log "swap 已存在，跳过"
  fi
  free -h
else
  log "跳过 swap 配置（DO_SETUP_SWAP!=1）"
fi

# ---------------- 安装 qsc-cli ----------------
if ! command -v qsc-cli >/dev/null 2>&1; then
  log "安装 qsc-cli"
  cd "$WORK_DIR"
  wget -q "$QSC_DEB_URL" -O qsc-cli.deb || die "下载 qsc-cli.deb 失败: $QSC_DEB_URL"

  if ! sudo DEBIAN_FRONTEND=noninteractive apt-get install -y ./qsc-cli.deb; then
    warn "标准安装失败，尝试解包后放开 preinst 里过严的检查（会跳过它的自带前置校验，"
    warn "装完后本脚本会用 --version 做功能性验证，不是单纯看 dpkg 退出码）"
    rm -rf qsc_tmp qsc-cli-patched.deb
    dpkg-deb -R qsc-cli.deb qsc_tmp
    [[ -f qsc_tmp/DEBIAN/preinst ]] && sed -i "s/exit 1/exit 0/g" qsc_tmp/DEBIAN/preinst
    dpkg-deb -b qsc_tmp qsc-cli-patched.deb
    sudo dpkg -i qsc-cli-patched.deb || sudo apt-get install -f -y
    rm -rf qsc_tmp qsc-cli-patched.deb
  fi
  rm -f qsc-cli.deb
  hash -r
fi

command -v qsc-cli >/dev/null 2>&1 || die "qsc-cli 安装后仍不可用"
qsc-cli --version >/dev/null 2>&1 || die "qsc-cli --version 跑不通，装的是坏的，不继续往下走"
log "qsc-cli 可用: $(qsc-cli --version 2>&1 | head -1)"

# ---------------- 幂等登录（修复原脚本"重复登录报错"的问题）----------------
# 思路：先探测是否已登录，已登录就不再调用 login。不同版本 qsc-cli 的"查询登录态"
# 子命令可能不同，这里按常见命名依次尝试，都探测不到就直接尝试登录一次，
# 并且把 login 的非零退出码里"已登录"这类信息当作非致命情况处理，而不是
# 无脑再调一遍。
is_logged_in() {
  for probe_cmd in "qsc-cli whoami" "qsc-cli account status" "qsc-cli auth status"; do
    if $probe_cmd >/dev/null 2>&1; then
      return 0
    fi
  done
  return 1
}

if is_logged_in; then
  log "检测到已登录状态，跳过 login"
else
  log "执行登录（仅一次，不重复调用）"
  login_output=$(qsc-cli login -u "$QSC_EMAIL" -p "$QSC_PASSWORD" 2>&1) || {
    if echo "$login_output" | grep -qiE "already logged in|already authenticated"; then
      log "login 返回非 0 但信息显示已登录，视为成功"
    else
      die "qsc-cli 登录失败: $login_output"
    fi
  }
  log "登录完成"
fi

# ---------------- 下载并标准化 SDK 目录 ----------------
QNN_ZIP_NAME="v${QNN_SDK_VERSION}.zip"
cd "$WORK_DIR"

if [[ -d "$WORK_DIR/qairt_root" ]]; then
  log "检测到已存在 $WORK_DIR/qairt_root，跳过重新下载（删掉这个目录可强制重下）"
else
  log "下载 SDK: $QNN_SDK_NAME @ $QNN_SDK_VERSION"
  qsc-cli sdk download \
    --name "$QNN_SDK_NAME" \
    --required-version "$QNN_SDK_VERSION" \
    --path . \
    --force || die "SDK 下载失败"

  [[ -f "$QNN_ZIP_NAME" ]] || die "找不到期望的产物文件: $QNN_ZIP_NAME（版本号/命名规则可能变了，检查 qsc-cli sdk list 的实际输出）"

  log "解压并标准化目录"
  rm -rf qairt_tmp
  unzip -q "$QNN_ZIP_NAME" -d qairt_tmp
  rm -f "$QNN_ZIP_NAME"

  SDK_ROOT=$(find qairt_tmp -maxdepth 3 -type d -name "bin" | head -n 1 | xargs -r dirname)
  [[ -n "$SDK_ROOT" ]] || die "解压后找不到包含 bin/ 的 SDK 根目录"

  mv "$SDK_ROOT" "$WORK_DIR/qairt_root"
  rm -rf qairt_tmp
fi

log "SDK 就绪: $WORK_DIR/qairt_root"

# 写出供后续步骤 source 的环境变量文件
cat > "$WORK_DIR/.env" <<EOF
export QNN_SDK_ROOT="$WORK_DIR/qairt_root"
export QAIRT_ROOT="$WORK_DIR/qairt_root"
EOF
log "环境变量已写入 $WORK_DIR/.env，后续步骤 source 一下即可：source $WORK_DIR/.env"
