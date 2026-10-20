# Corrected night pairing of the FA-R-XL test split

The test split file distributed with FA-R-XL (`FLYAWARE-R-XL/splits/heterogeneous/test/test_user.json`, SHA-256
`736ad78ea00f6d145acb797c48ac84b56e3ffd7186c63028ea077fc741050742`, identical to the copy in the current
`FLYAWARE-R-XL.zip`, SHA-256 `84d615ba3e961fc8f5acd0e4b5f97e1604c12e38ee2d8168308fbafcb76a9bf5`) pairs images and
label maps by position. The 112 clear, 50 fog and 46 rain test images are paired with the label map of the same file
name. None of the 50 night images is: the night label list is the sorted night image list rotated by 35 positions,
so `test/night/00000.png` is scored against the label map `test/night/00036.png`, and so on. HyperFLAW's data loader
(`dataset/target/flyaware_r_xl.py`, commit `682b58d`) reads the label path from this list, so the official evaluation
scores every night image against the label map of a different image.

We pair every night image with its same-name label map and leave everything else unchanged.

| file | content |
|---|---|
| `test_user_nightfix.json` | corrected split (SHA-256 `21c030066436c053b3fbacbee9f468d7bae7d2946632352402eb41732f7987de`); it differs from the official file only in the 50 night label paths |
| `night_pairs.csv` | the 50 night images with the official and the corrected label map and their file-index offset |
| `edge_alignment.json` | result of the model-free check below |
| `overlays/<name>.jpg` | each night image with its same-name label map (left) and with its official partner (right), colours at 50% opacity; images brightened for display |
| `scripts/make_corrected_split.py` | rebuilds the two files above from the official split and checks both hashes |
| `scripts/edge_alignment_check.py` | the model-free check |
| `scripts/make_overlays.py` | renders the overlays |

```bash
python night_fix/scripts/make_corrected_split.py --data_root DATA --out_dir /tmp/nightfix   # byte-identical rebuild
python night_fix/scripts/edge_alignment_check.py --data_root DATA --out /tmp/edge_alignment.json
python night_fix/scripts/make_overlays.py --data_root DATA --out_dir /tmp/night_overlays
```

## Model-free check

Does a label map's class boundary follow the image's edges? For each image, the score of a candidate label map is the
mean image gradient magnitude on the label map's class boundaries divided by the mean gradient magnitude of the image
(about 1 for unrelated boundaries). Each night image is ranked against all 50 night label maps:

| | same-name label map | official partner |
|---|---|---|
| ranked first (of 50) | 47 / 50 | 1 / 50 |
| median rank | 1 | 29.5 |
| mean score | 1.441 | 1.035 (all pairs: 1.052) |

The same-name map scores higher than the official partner for 49 of the 50 night images. As a control, 110 of the 112
clear images, which the official split pairs by file name, rank their own label map first. The one night
exception, `00042`, shows a dense block of tall buildings whose many window edges favour label maps with many
boundaries; visually, its same-name label map (almost only buildings) matches the image and its official partner
(the label map of `00024`: a straight road with large vegetation areas) does not.

## Effect

With the corrected pairing, the drone night mIoU of HyperFLAW's released model rises from 12.66% to 28.38%, and that
of our HyperFLAW run from 12.67% to 30.16% (drone mIoU 35.44% to 39.91%). The car test split is not affected. All
results of the paper and of this repository use the corrected pairing unless stated otherwise;
`python -m street2uav.evaluate --pairing official` scores with the official file.

The overlays contain FA-R-XL images (© 2024 G. Rizzoli, M. Caligiuri, D. Shenaj, F. Barbato, P. Zanuttigh,
CC BY-NC-SA 4.0), shared under the same license.
