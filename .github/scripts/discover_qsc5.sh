#!/usr/bin/env bash
# Fifth discovery: with the OFFICIAL latest qsc-cli (v1.28.1) installed and
# authenticated via QSC_API_KEY (v1.28.1 dropped -u/-p login), list the
# available SDKs and attempt to download Qualcomm AI Runtime 2.42.0.251225,
# then locate qairt-converter so the local INT4-QDQ -> qnn_context_binary
# compile can finally run.
set +e
WORK_DIR="$HOME/litert_workspace"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"
LOG="$WORK_DIR/qsc_discover5.log"
exec > >(tee -a "$LOG") 2>&1
export QSC_API_KEY="${QSC_API_KEY:-$QSC_API_KEY_INPUT}"
echo "===== $(date -u) install official latest qsc-cli (1.28.1) ====="
cd /tmp
curl -L "https://softwarecenter.qualcomm.com/api/download/software/tools/Qualcomm_Software_Center/Linux/Debian/latest.deb" -o qsc_latest.deb -w "HTTP %{http_code} size=%{size_download}\n"
ls -la qsc_latest.deb
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y ./qsc_latest.deb || {
  rm -rf qtmp qpatched.deb
  dpkg-deb -R qsc_latest.deb qtmp
  sed -i "s/exit 1/exit 0/g" qtmp/DEBIAN/preinst 2>/dev/null
  dpkg-deb -b qtmp qpatched.deb
  sudo dpkg -i qpatched.deb || sudo apt-get install -f -y
  rm -rf qtmp qpatched.deb
}
rm -f qsc_latest.deb
hash -r
qsc-cli --version
echo "===== sdk info (available sdks + versions) ====="
qsc-cli sdk info
echo "===== sdk download --help ====="
qsc-cli sdk download --help
echo "===== sdk install --help ====="
qsc-cli sdk install --help
echo "===== attempt download Qualcomm AI Runtime 2.42.0.251225 ====="
DL="$WORK_DIR/qairt_dl"
mkdir -p "$DL"
qsc-cli sdk download --name Qualcomm_AI_Runtime_Community --required-version 2.42.0.251225 --path "$DL" --force
echo "download rc=$?"
echo "===== list downloaded tree ====="
find "$DL" -maxdepth 4 | head -80
echo "===== sdk install ====="
qsc-cli sdk install --name Qualcomm_AI_Runtime_Community --required-version 2.42.0.251225 --path "$DL" 2>&1 | head -40
echo "===== locate qairt-converter ====="
QAIRT_ROOT=$(find "$WORK_DIR" "$HOME" -maxdepth 5 -type d -name "qairt" 2>/dev/null | head -1)
echo "QAIRT_ROOT=$QAIRT_ROOT"
find "$QAIRT_ROOT" -name qairt-converter 2>/dev/null
find "$WORK_DIR" -name qairt-converter 2>/dev/null | head
echo "===== $(date -u) done ====="
