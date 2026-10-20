# Release code vs. the pipeline used for the paper

Checked on 2026-10-19 in the paper's environment (Python 3.9, PyTorch 1.10.1, CUDA 11.3, one RTX 4080 SUPER).

## Test sets

`street2uav.evaluate` against HyperFLAW's official evaluation code (corrected night pairing). Each run compares 21
metrics (joint; car and drone mIoU; car and drone mIoU per weather; car and drone IoU of road, building, vegetation,
person and car) at the precision printed by the official evaluator's log (joint: 3 decimals in %; the others: 4-5
decimals as fractions). The JSON files in this folder hold the full results and confusion matrices.

| checkpoint | TTA | equal | joint | car | drone | drone night |
|---|---|---|---|---|---|---|
| `final_s42.pth` | yes | 21 / 21 | 70.623 | 76.489 | 65.969 | 52.91 |
| `final_s42.pth` | no | 21 / 21 | 68.908 | 74.853 | 63.479 | 51.67 |
| `final_s1.pth` | yes | 21 / 21 | 70.586 | 76.638 | 65.097 | 53.32 |
| `final_s2.pth` | yes | 21 / 21 | 70.607 | 76.588 | 66.128 | 53.69 |
| `control_s42.pth` | yes | 21 / 21 | 45.929 | 57.448 | 26.495 | 22.84 |
| `control_s42.pth` | no | 21 / 21 | 44.044 | 55.917 | 24.147 | 21.34 |
| frozen DINOv2-L, s = 630 (Table 3) | yes | 21 / 21 | 65.895 | 72.731 | 58.805 | 45.05 |
| linear head (Table 3) | yes | 21 / 21 | 69.558 | 75.285 | 67.006 | 52.82 |

The 2-decimal values in `../reference_official_evaluator.json` were rounded twice (log, then summary) and can differ
from these by 0.01 (control with TTA: drone 26.495 is listed there as 26.50).

## Street validation

`street2uav.validate` against the paper's validation script, `final_s42.pth` (mIoU %; clear / fog / rain / night):

| | release code | paper |
|---|---|---|
| single pass | 74.89 (71.34 / 80.49 / 66.18 / 63.56) | 74.89 (71.34 / 80.49 / 66.18 / 63.56) |
| TTA | 76.20 (72.75 / 82.23 / 67.85 / 64.44) | 76.20 (72.75 / 82.23 / 67.85 / 64.44) |

## Training

One epoch of the final recipe with seed 42, with the original training script and with `street2uav.train` (same data
order, augmentations and initialization; GPU training is not bit-reproducible):

| script | train loss | street val. mIoU | clear / fog / rain / night | minutes |
|---|---|---|---|---|
| original | 0.299 | 63.37 | 60.20 / 62.86 / 55.09 / 57.18 | 8.6 |
| release | 0.299 | 63.37 | 60.20 / 62.86 / 55.10 / 57.18 | 8.6 |
