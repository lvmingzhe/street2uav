#!/usr/bin/env bash
# Download the same-label control checkpoints (HyperFLAW's DeepLabV3 + MobileNetV2 trained on street labels,
# seeds 42, 1, 2; 52 MB each) from the GitHub release of this repository and check their SHA-256.
#   bash scripts/download_checkpoints.sh
set -euo pipefail
cd "$(dirname "$0")/.."
BASE="https://github.com/lvmingzhe/street2uav/releases/download/v1.0"
for s in 42 1 2; do
  f="checkpoints/control_s$s.pth"
  [ -f "$f" ] || curl -L --fail -o "$f" "$BASE/control_s$s.pth"
done
(cd checkpoints && sha256sum -c SHA256SUMS --ignore-missing)
