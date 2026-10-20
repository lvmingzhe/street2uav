#!/usr/bin/env bash
# Table 3: one-change ablations of the final model (seed 42). Each setting is trained, then scored on the street
# validation split and on the test sets, both with TTA (the "Single pass" row is the final model without TTA).
# Re-entrant: finished steps are skipped. Pass setting names to run a subset:
#   bash scripts/table3.sh /path/to/DATA [final linear_head ...]     (results in runs/table3/)
# Time on one RTX 4080 SUPER: 1.5-2.5 h per setting; fine-tuned RADIO about 6.3 h (8 s per iteration). The
# "+ AeroScapes labels" setting needs DATA/AeroScapes (see README).
set -euo pipefail
DATA=${1:?usage: bash scripts/table3.sh DATA [setting ...]}
shift
cd "$(dirname "$0")/.."
OUT=runs/table3
mkdir -p "$OUT"
FIXED=(--total_iters 2540)  # same number of iterations as the final recipe (10 x 2034 // 8), sampling with replacement
declare -A ARGS=(
  [final]=""
  [linear_head]="--head linear"
  [dinov2_s630]="--backbone dinov2_vitl14 --input_size 630"
  [finetuned_radio]="--backbone_lr 1e-5 --micro_batch 2"
  [short_side_432]="--input_size 432"
  [square_720]="--no-keep_aspect"
  [no_augmentation]="--aug none"
  [acdc_only]="--sources acdc ${FIXED[*]}"
  [cityscapes_only]="--sources cityscapes ${FIXED[*]}"
  [no_acdc_night]="--acdc_weathers fog rain ${FIXED[*]}"
  [plus_aeroscapes]="--sources acdc cityscapes aeroscapes ${FIXED[*]}"
)
ORDER=(final linear_head dinov2_s630 finetuned_radio short_side_432 square_720 no_augmentation acdc_only
       cityscapes_only no_acdc_night plus_aeroscapes)
[ $# -gt 0 ] && ORDER=("$@")
for name in "${ORDER[@]}"; do
  ck="$OUT/$name.pth"
  # shellcheck disable=SC2086
  [ -f "$ck" ] || python -m street2uav.train --data_root "$DATA" --seed 42 --out "$ck" ${ARGS[$name]}
  [ -f "$OUT/${name}_val.json" ] || python -m street2uav.validate --ckpt "$ck" --data_root "$DATA" --tta \
    --out "$OUT/${name}_val.json"
  [ -f "$OUT/${name}_test.json" ] || python -m street2uav.evaluate --ckpt "$ck" --data_root "$DATA" --tta \
    --out "$OUT/${name}_test.json"
done
if [ -f "$OUT/final.pth" ] && [ ! -f "$OUT/single_pass_test.json" ]; then
  python -m street2uav.validate --ckpt "$OUT/final.pth" --data_root "$DATA" --out "$OUT/single_pass_val.json"
  python -m street2uav.evaluate --ckpt "$OUT/final.pth" --data_root "$DATA" --out "$OUT/single_pass_test.json"
fi
python - "$OUT" <<'EOF'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
print("| setting | Val | Car | Drone | Night |\n|---|---|---|---|---|")
for t in sorted(out.glob("*_test.json")):
    name = t.name[:-len("_test.json")]
    r, v = json.load(open(t)), out / f"{name}_val.json"
    val = json.load(open(v))["val"]["miou"] if v.exists() else float("nan")
    cells = [val, r["car"]["miou"], r["drone"]["miou"], r["drone"]["weather"]["night"]]
    print(f"| {name} | " + " | ".join(f"{c:.2f}" for c in cells) + " |")
EOF
