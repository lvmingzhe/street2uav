#!/usr/bin/env bash
# Table 1: evaluate the released checkpoints (final recipe, three seeds; same-label control, three seeds) with and
# without test-time augmentation, corrected night pairing. Re-entrant: existing results are kept.
#   bash scripts/table1.sh /path/to/DATA        (results in runs/table1/, summary printed at the end)
# Time on one RTX 4080 SUPER: about 25 min per final checkpoint with TTA, 6 min without; the control is faster.
set -euo pipefail
DATA=${1:?usage: bash scripts/table1.sh DATA}
cd "$(dirname "$0")/.."
OUT=runs/table1
mkdir -p "$OUT"
evaluate() {  # name checkpoint [--tta]
  [ -f "$OUT/$1.json" ] || python -m street2uav.evaluate --ckpt "$2" --data_root "$DATA" --out "$OUT/$1.json" "${@:3}"
}
for s in 42 1 2; do
  evaluate "final_tta_s$s" "checkpoints/final_s$s.pth" --tta
  evaluate "final_single_s$s" "checkpoints/final_s$s.pth"
done
for s in 42 1 2; do  # control checkpoints: bash scripts/download_checkpoints.sh
  if [ -f "checkpoints/control_s$s.pth" ]; then
    evaluate "control_tta_s$s" "checkpoints/control_s$s.pth" --tta
    evaluate "control_single_s$s" "checkpoints/control_s$s.pth"
  fi
done
python scripts/summarize.py "$OUT"/*.json
