#!/usr/bin/env bash
# Acquire the Qualcomm AI Runtime (QAIRT) SDK via qsc-cli 1.12.0 for SM8750.
#
# This is a DISCOVERY step: the exact qsc-cli install method, login syntax,
# and download subcommand are not pinned anywhere in the repo (Phase-0 BLOCKER
# per docs/dreamlite_npu_plan.md). Every external-tool step is verbose (set -x)
# EXCEPT the qsc login, whose command line + output are masked so the
# credential (QSC_EMAIL / QSC_PASS) never lands in the Actions log.
#
# On success it exports QNN_SDK_ROOT / QAIRT_ROOT to $GITHUB_ENV for downstream
# compile steps. On any failure it dumps `qsc --help` / subcommand help so the
# real syntax is visible in the run log.
set -e
echo "==> acquire_qairt_sdk start; uname=$(uname -a)"

# ---- install qsc-cli (method unknown -> try candidates, verbose) ----
set -x
pip install --quiet qsc-cli 2>&1 | tail -5 || true
command -v qsc || qsc --version 2>/dev/null || echo "[probe] qsc-cli absent after 'pip install qsc-cli'"
pip install --quiet qsc 2>&1 | tail -5 || true
pip install --quiet qualcomm-software-center 2>&1 | tail -5 || true
set +x
echo "==> qsc install candidates done; probing presence"
set -x
command -v qsc || echo "[probe] qsc still not on PATH"
qsc --version 2>&1 | head -20 || true
qsc --help 2>&1 | head -60 || true
set +x

# ---- login (CREDENTIAL: do NOT echo command line; mask output) ----
set +x
echo "==> qsc login (credential masked)"
if command -v qsc >/dev/null 2>&1; then
  qsc login "$QSC_EMAIL" "$QSC_PASS" 2>&1 \
    | sed -E 's/(password|passwd|token|=).*/\1=***/I' | head -40 || true
else
  echo "[probe] qsc not found; cannot login"
fi
set -x

# ---- discover download subcommand (verbose, no creds) ----
echo "==> qsc subcommand discovery"
qsc download --help 2>&1 | head -60 || true
qsc workspace --help 2>&1 | head -60 || true
qsc info --help 2>&1 | head -40 || true

# ---- attempt the download (best-guess subcommands) ----
echo "==> attempting download (Qualcomm_AI_Runtime_Community, QAIRT 2.42.0.251225)"
qsc download Qualcomm_AI_Runtime_Community 2>&1 | tail -20 || true
qsc download --product Qualcomm_AI_Runtime_Community --version 2.42.0.251225 2>&1 | tail -20 || true
qsc workspace download Qualcomm_AI_Runtime_Community 2>&1 | tail -20 || true

# ---- locate the SDK + qairt-converter ----
echo "==> locating qairt-converter / QNN_SDK_ROOT"
QC=$(find / -type f -name qairt-converter 2>/dev/null | head -5)
echo "qairt-converter candidates: $QC"
for b in $QC; do
  d=$(dirname "$b"); echo "SDK bin dir: $d"; ls -la "$d" | head -20
  echo "--- qairt-converter --help ---"
  "$b" --help 2>&1 | head -40 || true
done
find / -type d \( -iname "QNN" -o -iname "*qnn*" -o -iname "*qualcomm*ai*runtime*" \) 2>/dev/null | head -20

# export for downstream steps (if found)
if [ -n "$QC" ]; then
  SDK=$(dirname "$(echo "$QC" | head -1)")
  echo "QNN_SDK_ROOT=$SDK" >> "$GITHUB_ENV"
  echo "QAIRT_ROOT=$SDK" >> "$GITHUB_ENV"
  echo "[probe] exported QNN_SDK_ROOT=$SDK"
fi
echo "==> acquire_qairt_sdk done"
