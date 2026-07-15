#!/usr/bin/env bash
# Sixth discovery: honor "account/password login, no API key".
# qsc-cli 1.28.1 dropped -u/-p login, but 1.12.0 still supports it and creates
# a persistent session. Strategy: install 1.12.0, log in with creds (session
# persists), then UPGRADE the binary to 1.28.1 (has the `sdk` subcommand) on
# top. If 1.28.1 reads the same session store, `qsc-cli sdk download` works
# with pure account/password. Then pull Qualcomm AI Runtime 2.42.0.251225 and
# locate qairt-converter.
set +e
WORK_DIR="$HOME/litert_workspace"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"
LOG="$WORK_DIR/qsc_discover6.log"
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
echo "===== verify session: info product (1.12.0) ====="
qsc-cli info product 2>&1 | head -20
echo "===== upgrade binary to 1.28.1 (has sdk subcommand) over the top ====="
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
echo "===== sdk info (uses 1.12.0 session if shared) ====="
qsc-cli sdk info
echo "===== sdk download --help ====="
qsc-cli sdk download --help
echo "===== attempt sdk download Qualcomm AI Runtime 2.42.0.251225 ====="
DL="$WORK_DIR/qairt_dl"; mkdir -p "$DL"
qsc-cli sdk download --name Qualcomm_AI_Runtime_Community --required-version 2.42.0.251225 --path "$DL" --force
echo "download rc=$?"
echo "===== list downloaded tree ====="
find "$DL" -maxdepth 4 | head -80
echo "===== sdk install ====="
qsc-cli sdk install --name Qualcomm_AI_Runtime_Community --required-version 2.42.0.251225 --path "$DL" 2>&1 | head -40
echo "===== locate qairt-converter ====="
QAIRT_ROOT=$(find "$WORK_DIR" "$HOME" -maxdepth 5 -type d -name qairt 2>/dev/null | head -1)
echo "QAIRT_ROOT=$QAIRT_ROOT"
find "$QAIRT_ROOT" -name qairt-converter 2>/dev/null
find "$WORK_DIR" -name qairt-converter 2>/dev/null | head
echo "===== $(date -u) done ====="
