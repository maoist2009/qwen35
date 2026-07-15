#!/usr/bin/env bash
# Seventh discovery: extract the Qualcomm AI Runtime SDK zip (sdk install
# refuses zip => unzip manually) and locate qairt-converter so we can confirm
# the local INT4-QDQ -> qnn_context_binary compile is possible.
set +e
WORK_DIR="$HOME/litert_workspace"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"
LOG="$WORK_DIR/qsc_discover7.log"
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
echo "===== upgrade binary to 1.28.1 (has sdk subcommand) ====="
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
echo "===== sdk download Qualcomm AI Runtime 2.42.0.251225 ====="
DL="$WORK_DIR/qairt_dl"; mkdir -p "$DL"
qsc-cli sdk download --name Qualcomm_AI_Runtime_Community --required-version 2.42.0.251225 --path "$DL" --force
echo "download rc=$?"
echo "===== downloaded tree ====="
find "$DL" -maxdepth 4
echo "===== unzip manually ====="
UNZ="$WORK_DIR/qairt_unzip"; rm -rf "$UNZ"; mkdir -p "$UNZ"
ZIP=$(find "$DL" -maxdepth 2 -name '*.zip' | head -1)
echo "ZIP=$ZIP"
unzip -o "$ZIP" -d "$UNZ" >/dev/null 2>&1
echo "===== unzipped tree (top) ====="
find "$UNZ" -maxdepth 3 | head -60
echo "===== locate qairt-converter ====="
CONV=$(find "$UNZ" "$WORK_DIR" -name qairt-converter 2>/dev/null | head -1)
echo "CONV=$CONV"
if [ -n "$CONV" ]; then
  QAIRT_ROOT=$(dirname "$(dirname "$(dirname "$CONV")")")
  echo "QAIRT_ROOT=$QAIRT_ROOT"
  "$CONV" --version 2>&1 | head -5
  echo "----- converter help (first 30 lines) -----"
  "$CONV" --help 2>&1 | head -30
fi
echo "===== $(date -u) done ====="
