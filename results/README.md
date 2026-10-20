# Results

`reference_official_evaluator.json` holds the paper's numbers as computed during the project with HyperFLAW's own
evaluation code (`run_adapt.sh` in evaluation mode, corrected night pairing, batch size 1) and the paper's street
validation script:

- `test`: every released checkpoint (`final_s42`, `final_s1`, `final_s2`, `control_s42`, `control_s1`,
  `control_s2`) with (`tta`) and without (`single`) test-time augmentation, plus the DINOv2-L and linear-head
  ablations (seed 42, TTA); joint, car and drone mIoU, per-weather and per-class IoU (%);
- `street_validation_miou`: street validation mIoU of the final checkpoints;
- `three_seed_mean_std`: mean and sample standard deviation over seeds 42, 1 and 2 (Table 1);
- `table3_ablation_seed42`: all rows of Table 3.

`release_code/` holds the outputs of this repository's `evaluate.py` and `validate.py` for the released checkpoints,
including the confusion matrices, and `release_code/comparison.md` compares them with the reference numbers.
