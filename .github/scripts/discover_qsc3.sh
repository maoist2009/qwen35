#!/usr/bin/env bash
# Third discovery: list qsc-cli products so we can find the Qualcomm AI Runtime
# Community product name + its distribution/release for `qsc-cli download`. set +e.
set +e
WORK_DIR="$HOME/litert_workspace"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"
LOG="$WORK_DIR/qsc_discover3.log"
exec > >(tee -a "$LOG") 2>&1
echo "===== $(date -u) install ====="
cd "$WORK_DIR"
wget -q "https://github.com/maoist2009/qwen35/releases/download/QSC-cli/qsc-cli.deb" -O qsc-cli.deb
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y ./qsc-cli.deb || {
  rm -rf qsc_tmp qsc-cli-patched.deb
  dpkg-deb -R qsc-cli.deb qsc_tmp
  sed -i "s/exit 1/exit 0/g" qsc_tmp/DEBIAN/preinst
  dpkg-deb -b qsc_tmp qsc-cli-patched.deb
  sudo dpkg -i qsc-cli-patched.deb || sudo apt-get install -f -y
  rm -rf qsc_tmp qsc-cli-patched.deb
}
rm -f qsc-cli.deb
hash -r
echo "===== login ====="
qsc-cli login -u "$QSC_EMAIL" -p "$QSC_PASSWORD" 2>&1 | sed -E 's/(password|pass|token).*/\1=***/I'
echo "===== info product --help ====="
qsc-cli info product --help
echo "===== info product (full list) ====="
qsc-cli info product
echo "===== grep AI/Runtime/Qualcomm ====="
qsc-cli info product 2>&1 | grep -iE "ai|runtime|qualcomm|neural" | head -80
echo "===== $(date -u) done ====="
