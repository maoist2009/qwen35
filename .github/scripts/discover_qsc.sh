#!/usr/bin/env bash
# Discovery: install qsc-cli.deb and dump its REAL interface so we can pin the
# exact sdk-download subcommand. set +e so we capture every probe even on error.
set +e
WORK_DIR="$HOME/litert_workspace"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"
LOG="$WORK_DIR/qsc_discover.log"
exec > >(tee -a "$LOG") 2>&1
echo "===== $(date -u) start ====="
echo "===== install qsc-cli.deb ====="
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
echo "which qsc-cli: $(command -v qsc-cli)"
echo "===== qsc-cli --version ====="
qsc-cli --version
echo "===== qsc-cli --help (full) ====="
qsc-cli --help
echo "===== probe update/self-update subcommands ====="
qsc-cli update --help 2>&1
qsc-cli self-update --help 2>&1
qsc-cli upgrade --help 2>&1
echo "===== qsc-cli sdk --help (pre-login) ====="
qsc-cli sdk --help 2>&1
echo "===== login ====="
qsc-cli login -u "$QSC_EMAIL" -p "$QSC_PASSWORD" 2>&1 | sed -E 's/(password|pass|token).*/\1=***/I'
echo "===== qsc-cli --help (post-login, full) ====="
qsc-cli --help
echo "===== qsc-cli sdk --help (post-login) ====="
qsc-cli sdk --help 2>&1
echo "===== qsc-cli sdk download --help (post-login) ====="
qsc-cli sdk download --help 2>&1
echo "===== wait 60s, re-probe sdk (allow async self-update) ====="
sleep 60
qsc-cli --version
qsc-cli sdk --help 2>&1
echo "===== $(date -u) done ====="
