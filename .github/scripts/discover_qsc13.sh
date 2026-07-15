#!/usr/bin/env bash
# Thirteenth discovery: same as discover12 (python3.10 venv + qti on PYTHONPATH
# + libc++1 for the clang-built libDlModelToolsPy.so), but the libc++ apt install
# no longer collides with QSC 1.28.1's background dependencies_installer.sh. We
# wait for the dpkg frontend lock to be free and retry, so libc++.so.1 actually
# lands on disk before we run the converter.
set +e
wait_apt_lock() {
  local n=0
  while fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 || fuser /var/lib/dpkg/lock >/dev/null 2>&1; do
    n=$((n+1))
    if [ "$n" -gt 120 ]; then echo "WARN: apt lock still held after ~240s, proceeding anyway"; return 0; fi
    sleep 2
  done
  echo "apt lock free"
}
apt_install() {
  # install with dpkg-lock wait + one retry
  wait_apt_lock
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y "$@" 2>&1 | tail -4 || {
    echo "first attempt failed, waiting and retrying..."
    sleep 10; wait_apt_lock
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y "$@" 2>&1 | tail -4
  }
}
WORK_DIR="$HOME/litert_workspace"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"
LOG="$WORK_DIR/qsc_discover13.log"
exec > >(tee -a "$LOG") 2>&1
echo "===== $(date -u) install qsc-cli 1.12.0 (supports -u/-p login) ====="
cd /tmp
wget -q "https://github.com/maoist2009/qwen35/releases/download/QSC-cli/qsc-cli.deb" -O qsc112.deb
apt_install ./qsc112.deb || {
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
apt_install ./qsc_installer.deb || {
  rm -rf qtmp qpatched.deb
  dpkg-deb -R qsc_installer.deb qtmp
  sed -i "s/exit 1/exit 0/g" qtmp/DEBIAN/preinst 2>/dev/null
  dpkg-deb -b qtmp qpatched.deb
  sudo dpkg -i qpatched.deb || sudo apt-get install -f -y
  rm -rf qtmp qpatched.deb
}
rm -f qsc_installer.deb; hash -r
qsc-cli --version
echo "===== settle: let QSC 1.28.1 background dependencies_installer.sh release the dpkg lock ====="
sleep 15; wait_apt_lock
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
echo "===== install libc++ (LLVM C++ runtime) — guarded against dpkg lock ====="
apt_install libc++1 libc++abi1
echo "libc++.so.1 location:"; find / -name 'libc++.so.1' 2>/dev/null | head
echo "===== setup python3.10 venv + deps ====="
python3.10 -m venv "$WORK_DIR/venv"
. "$WORK_DIR/venv/bin/activate"
python -m pip install -q --upgrade pip
python -m pip install -q numpy onnx onnxruntime scipy packaging
echo "===== source envsetup (PYTHONPATH + LD_LIBRARY_PATH) ====="
. "$QAIRT_ROOT/bin/envsetup.sh"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "===== converter --version (py3.10 + libc++ + qti on path) ====="
"$CONV" --version 2>&1 | head -15
echo "===== converter --help (py3.10 + libc++ + qti on path) ====="
"$CONV" --help 2>&1 | head -90
echo "===== $(date -u) done ====="
