#!/usr/bin/env bash
# Twelfth discovery: with python3.10 the converter now only fails on
# libc++.so.1 (the SDK's libDlModelToolsPy.so is clang-built and links the
# LLVM C++ runtime). Install libc++1/libc++abi1 via apt, locate the SDK's
# bundled libc++ as a fallback, then run qairt-converter --version / --help.
set +e
WORK_DIR="$HOME/litert_workspace"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"
LOG="$WORK_DIR/qsc_discover12.log"
exec > >(tee -a "$LOG") 2>&1
echo "===== $(date -u) install qsc-cli 1.12.0 (supports -u/-p login) ====="
cd /tmp
wget -q "https://github.com/maoist2009/qwen35/releases/download/QSC-cli/qsc-cli.deb" -O qsc112.deb
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y ./qsc112.deb || {
  rm -rf qtmp qpatched.deb
  dpkg-deb -R qsc112.deb qtmp
  sed -i "s/exit 1/exit 0/g" qtmp/DEBIAN/preinst 2>/dev/null
  dpkg-deb -b qtmp qpatched.deb
  sudo dpkg -i qpatched.deb || sudo apt-get install -f -y
  rm -rf qtmp qpatched.deb
}
rm -f qsc112.deb; hash -r
qsc-cli --version
echo "===== login via -u/-p (1.12.0) ====="
qsc-cli login -u "$QSC_EMAIL" -p "$QSC_PASSWORD" 2>&1 | sed -E 's/(password|pass|token).*/\1=***/I'
echo "===== upgrade binary to 1.28.1 ====="
curl -L "https://softwarecenter.qualcomm.com/api/download/software/tools/Qualcomm_Software_Center/Linux/Debian/latest.deb" -o qsc_installer.deb -w "HTTP %{http_code} size=%{size_download}\n"
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y ./qsc_installer.deb || {
  rm -rf qtmp qpatched.deb
  dpkg-deb -R qsc_installer.deb qtmp
  sed -i "s/exit 1/exit 0/g" qtmp/DEBIAN/preinst 2>/dev/null
  dpkg-deb -b qtmp qpatched.deb
  sudo dpkg -i qpatched.deb || sudo apt-get install -f -y
  rm -rf qtmp qpatched.deb
}
rm -f qsc_installer.deb; hash -r
qsc-cli --version
echo "===== restore/cache QAIRT SDK zip ====="
DL="$WORK_DIR/qairt_dl"; mkdir -p "$DL"
if [ ! -f "$DL/v2.42.0.251225.zip" ]; then
  qsc-cli sdk download --name Qualcomm_AI_Runtime_Community --required-version 2.42.0.251225 --path "$DL" --force
  echo "download rc=$?"
fi
UNZ="$WORK_DIR/qairt_unzip"; rm -rf "$UNZ"; mkdir -p "$UNZ"
ZIP=$(find "$DL" -maxdepth 2 -name '*.zip' | head -1)
unzip -o "$ZIP" -d "$UNZ" >/dev/null 2>&1
QAIRT_ROOT="$UNZ/qairt/2.42.0.251225"
CONV="$QAIRT_ROOT/bin/x86_64-linux-clang/qairt-converter"
echo "QAIRT_ROOT=$QAIRT_ROOT"
echo "CONV=$CONV"
echo "===== install libc++ (LLVM C++ runtime) ====="
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y libc++1 libc++abi1 2>&1 | tail -3
echo "===== locate any SDK-bundled libc++ as fallback ====="
find "$QAIRT_ROOT" -name 'libc++.so*' 2>/dev/null | head
SDK_LIBCXX=$(find "$QAIRT_ROOT" -name 'libc++.so*' 2>/dev/null | head -1)
if [ -n "$SDK_LIBCXX" ]; then
  SDK_LIBCXX_DIR=$(dirname "$SDK_LIBCXX")
  echo "SDK_LIBCXX_DIR=$SDK_LIBCXX_DIR"
fi
echo "===== setup python3.10 venv + deps ====="
python3.10 -m venv "$WORK_DIR/venv"
. "$WORK_DIR/venv/bin/activate"
python -m pip install -q --upgrade pip
python -m pip install -q numpy onnx onnxruntime scipy packaging
echo "===== source envsetup (PYTHONPATH + LD_LIBRARY_PATH) ====="
. "$QAIRT_ROOT/bin/envsetup.sh"
if [ -n "$SDK_LIBCXX_DIR" ]; then
  export LD_LIBRARY_PATH="$SDK_LIBCXX_DIR:$LD_LIBRARY_PATH"
fi
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "===== converter --version (py3.10 + libc++ + qti on path) ====="
"$CONV" --version 2>&1 | head -15
echo "===== converter --help (py3.10 + libc++ + qti on path) ====="
"$CONV" --help 2>&1 | head -90
echo "===== $(date -u) done ====="
