#!/usr/bin/env bash
# Fifteenth discovery: the converter now RUNS (libc++ + yaml/protobuf fixed in
# discover13/14). Confirm the exact flags needed for the real clip.bin build by
# dumping the FULL --help and grepping for the quantize / input_list /
# target_runtime tokens. Also run --version via the real flag if it exists.
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
LOG="$WORK_DIR/qsc_discover15.log"
exec > >(tee -a "$LOG") 2>&1
echo "===== $(date -u) skip qsc-cli reinstall (SDK zip cached); rely on cache ====="
echo "===== restore QAIRT SDK zip from cache ====="
DL="$WORK_DIR/qairt_dl"; mkdir -p "$DL"
if [ ! -f "$DL/v2.42.0.251225.zip" ]; then
  echo "CACHE MISS — would need qsc-cli; aborting"
  exit 2
fi
UNZ="$WORK_DIR/qairt_unzip"; rm -rf "$UNZ"; mkdir -p "$UNZ"
ZIP=$(find "$DL" -maxdepth 2 -name '*.zip' | head -1)
unzip -o "$ZIP" -d "$UNZ" >/dev/null 2>&1
QAIRT_ROOT="$UNZ/qairt/2.42.0.251225"
CONV="$QAIRT_ROOT/bin/x86_64-linux-clang/qairt-converter"
echo "QAIRT_ROOT=$QAIRT_ROOT"
echo "CONV=$CONV"
echo "===== setup python3.10 venv + deps ====="
python3.10 -m venv "$WORK_DIR/venv"
. "$WORK_DIR/venv/bin/activate"
python -m pip install -q --upgrade pip
python -m pip install -q numpy onnx onnxruntime scipy packaging pyyaml protobuf flatbuffers sympy
echo "===== source envsetup ====="
. "$QAIRT_ROOT/bin/envsetup.sh"
echo "===== FULL --help (capture to file, then grep key flags) ====="
"$CONV" --help > /tmp/conv_help.txt 2>&1
echo "--- grep: target_runtime / backend / soc_model ---"
grep -nE -- '--target_runtime|--target_backend|--target_soc_model' /tmp/conv_help.txt || echo "(none)"
echo "--- grep: input_list ---"
grep -nE -- '--input_list' /tmp/conv_help.txt || echo "(none)"
echo "--- grep: quantize ---"
grep -nE -- '--quantize' /tmp/conv_help.txt || echo "(none)"
echo "--- grep: input_network / output_path ---"
grep -nE -- '--input_network|--output_path' /tmp/conv_help.txt || echo "(none)"
echo "===== $(date -u) done ====="
