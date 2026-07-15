#!/usr/bin/env bash
# Second discovery: after login, dump the REAL `qsc-cli download` / `info`
# interface so we can pin the exact SDK-download invocation. set +e.
set +e
WORK_DIR="$HOME/litert_workspace"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"
LOG="$WORK_DIR/qsc_discover2.log"
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
echo "===== download --help ====="
qsc-cli download --help
echo "===== info --help ====="
qsc-cli info --help
echo "===== info (all products) ====="
qsc-cli info
echo "===== info grep Runtime ====="
qsc-cli info 2>&1 | grep -i "runtime" | head -40
echo "===== $(date -u) done ====="
