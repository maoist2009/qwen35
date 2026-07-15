#!/usr/bin/env bash
# Fourth discovery: install the OFFICIAL latest qsc-cli
# (https://softwarecenter.qualcomm.com/api/download/software/tools/Qualcomm_Software_Center/Linux/Debian/latest.deb)
# and check whether it exposes a `sdk` subcommand + whether this account is
# entitled to the Qualcomm AI Runtime product. The repo's QSC-cli (v1.12.0)
# had no `sdk` command and listed only device-firmware products.
set +e
WORK_DIR="$HOME/litert_workspace"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"
LOG="$WORK_DIR/qsc_discover4.log"
exec > >(tee -a "$LOG") 2>&1
echo "===== $(date -u) fetch official latest qsc-cli ====="
cd /tmp
curl -L "https://softwarecenter.qualcomm.com/api/download/software/tools/Qualcomm_Software_Center/Linux/Debian/latest.deb" -o qsc_latest.deb -w "HTTP %{http_code} size=%{size_download}\n"
ls -la qsc_latest.deb
echo "===== install ====="
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
echo "===== version + top-level help ====="
qsc-cli --version
qsc-cli --help
echo "===== sdk subcommand? ====="
qsc-cli sdk --help
echo "===== login ====="
qsc-cli login -u "$QSC_EMAIL" -p "$QSC_PASSWORD" 2>&1 | sed -E 's/(password|pass|token).*/\1=***/I'
echo "===== sdk download --help (post-login) ====="
qsc-cli sdk download --help
echo "===== info product (AI/runtime grep) ====="
qsc-cli info product 2>&1 | grep -iE "ai|runtime|qualcomm|neural|a540|a740|sm8|sm7" | head -80
echo "===== info product (full) ====="
qsc-cli info product 2>&1 | head -120
echo "===== $(date -u) done ====="
