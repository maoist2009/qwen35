#!/usr/bin/env bash
# Idempotent QNN SDK acquisition via qsc-cli. Copy this file into work/ and
# run it — do not rewrite it. Credentials come from env vars only.
set -euo pipefail

WORK_DIR="${WORKDIR:-$HOME/litert_workspace}"
mkdir -p "$WORK_DIR"

QSC_DEB_URL="${QSC_DEB_URL:-https://github.com/maoist2009/qwen35/releases/download/QSC-cli/qsc-cli.deb}"
QNN_SDK_NAME="${QNN_SDK_NAME:-Qualcomm_AI_Runtime_Community}"
QNN_SDK_VERSION="${QNN_SDK_VERSION:-2.42.0.251225}"

log() { echo "[qnn-sdk] $*"; }
die() { echo "[FATAL] $*" >&2; exit 1; }

[[ -z "${QSC_EMAIL:-}" || -z "${QSC_PASSWORD:-}" ]] && die "export QSC_EMAIL and QSC_PASSWORD first"

if ! command -v qsc-cli >/dev/null 2>&1; then
  log "installing qsc-cli"
  cd "$WORK_DIR"
  wget -q "$QSC_DEB_URL" -O qsc-cli.deb || die "download qsc-cli.deb failed"
  if ! sudo DEBIAN_FRONTEND=noninteractive apt-get install -y ./qsc-cli.deb; then
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
command -v qsc-cli >/dev/null 2>&1 || die "qsc-cli still not available after install"

log "checking version, may trigger self-update to a newer release"
qsc-cli --version || true
if ! qsc-cli sdk --help >/dev/null 2>&1; then
  log "waiting up to 5 minutes for self-update to finish..."
  sleep 300
fi
qsc-cli --version
qsc-cli sdk --help >/dev/null 2>&1 || die "qsc-cli sdk subcommand still missing after waiting; self-update did not complete"

# idempotent login — check before calling, don't call login twice blindly
if qsc-cli whoami >/dev/null 2>&1; then
  log "already logged in"
else
  login_output=$(qsc-cli login -u "$QSC_EMAIL" -p "$QSC_PASSWORD" 2>&1) || {
    if echo "$login_output" | grep -qiE "already logged in|already authenticated"; then
      log "login returned nonzero but says already logged in, treating as success"
    else
      die "login failed: $login_output"
    fi
  }
  log "login complete"
fi

cd "$WORK_DIR"
if [[ -d "$WORK_DIR/qairt_root" ]]; then
  log "qairt_root already present, skipping download (delete it to force redownload)"
else
  QNN_ZIP_NAME="v${QNN_SDK_VERSION}.zip"
  qsc-cli sdk download --name "$QNN_SDK_NAME" --required-version "$QNN_SDK_VERSION" --path . --force \
    || die "SDK download failed"
  [[ -f "$QNN_ZIP_NAME" ]] || die "expected zip $QNN_ZIP_NAME not found after download"
  rm -rf qairt_tmp
  unzip -q "$QNN_ZIP_NAME" -d qairt_tmp
  rm -f "$QNN_ZIP_NAME"
  SDK_ROOT=$(find qairt_tmp -maxdepth 3 -type d -name "bin" | head -n 1 | xargs -r dirname)
  [[ -n "$SDK_ROOT" ]] || die "could not find bin/ directory after unzip"
  mv "$SDK_ROOT" "$WORK_DIR/qairt_root"
  rm -rf qairt_tmp
fi

cat > "$WORK_DIR/.env" <<EOF
export QNN_SDK_ROOT="$WORK_DIR/qairt_root"
export QAIRT_ROOT="$WORK_DIR/qairt_root"
EOF
log "done. source $WORK_DIR/.env to pick up QNN_SDK_ROOT/QAIRT_ROOT"
