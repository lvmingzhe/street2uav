#!/usr/bin/env bash
# Train the final recipe with three seeds, then score each model on the street validation split and the test sets
# (TTA). Re-entrant: finished steps are skipped.
#   bash scripts/train_final.sh /path/to/DATA   (checkpoints and results in runs/final/)
# Time on one RTX 4080 SUPER: about 97 min of training + 30 min of evaluation per seed.
set -euo pipefail
DATA=${1:?usage: bash scripts/train_final.sh DATA}
cd "$(dirname "$0")/.."
OUT=runs/final
mkdir -p "$OUT"
for s in 42 1 2; do
  [ -f "$OUT/final_s$s.pth" ] || python -m street2uav.train --data_root "$DATA" --seed "$s" --out "$OUT/final_s$s.pth"
  [ -f "$OUT/val_tta_s$s.json" ] || python -m street2uav.validate --ckpt "$OUT/final_s$s.pth" --data_root "$DATA" \
    --tta --out "$OUT/val_tta_s$s.json"
  [ -f "$OUT/test_tta_s$s.json" ] || python -m street2uav.evaluate --ckpt "$OUT/final_s$s.pth" --data_root "$DATA" \
    --tta --out "$OUT/test_tta_s$s.json"
done
python scripts/summarize.py "$OUT"/test_tta_s*.json
