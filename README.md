# Learning Adverse-Weather UAV Semantic Segmentation from Street Labels with a Vision Foundation Model

Code, model weights and the corrected night test split for the paper of the same title (under review).

A frozen [C-RADIOv4-SO400M](https://huggingface.co/nvidia/C-RADIOv4-SO400M) backbone with a 0.89M-parameter
convolutional decoder is trained on labeled street images only (ACDC fog/rain/night and Cityscapes clear) and
segments drone images in all weathers, without any drone training image. On the ACDC + FA-R-XL benchmark of
[HyperFLAW](https://github.com/LTTM/HyperFLAW) (mIoU %, corrected night pairing):

| Method | Labels | Drone images in training | Joint | Car | Drone | Clear | Fog | Rain | Night |
|---|---|---|---|---|---|---|---|---|---|
| HyperFLAW, released model | synthetic | unlabeled | 23.76 | 27.46 | 39.32 | 41.32 | 41.68 | 43.68 | 28.38 |
| HyperFLAW network, street labels, + TTA (3 seeds) | ACDC + Cityscapes | none | 45.36 | 57.02 | 26.73 | 29.32 | 21.69 | 32.54 | 21.03 |
| **Ours + TTA** (`checkpoints/final_s42.pth`) | ACDC + Cityscapes | none | **70.62** | **76.49** | **65.97** | **71.43** | **69.41** | **62.64** | **52.91** |
| Ours + TTA (3 seeds) | ACDC + Cityscapes | none | 70.61 | 76.57 | 65.73 | 70.75 | 69.20 | 62.76 | 53.31 |

Joint: mIoU over the car and drone test sets together (HyperFLAW's metric); Clear to Night: drone images. The
adverse-weather drone test images of FA-R-XL are synthetic translations of clear frames. The test split shipped with
FA-R-XL pairs each night image with the label map of another image; we pair them by file name
([night_fix/](night_fix/README.md)).

## Contents

| path | content |
|---|---|
| `street2uav/` | model (`model.py`), data and splits (`data.py`), training (`train.py`), street validation (`validate.py`), test evaluation with HyperFLAW's metric (`evaluate.py`), inference on any images (`predict.py`), same-label control (`control.py`) |
| `checkpoints/` | decoder weights of the final recipe, seeds 42 (the paper's selected model), 1 and 2 (3.5 MB each); the same-label control checkpoints are attached to the [v1.0 release](https://github.com/lvmingzhe/street2uav/releases/tag/v1.0) (`bash scripts/download_checkpoints.sh`) |
| `night_fix/` | corrected FA-R-XL test split, list of the 50 night pairs, overlays and a model-free alignment check |
| `scripts/` | commands for Tables 1 and 3, training of the final recipe, result summaries |
| `results/` | the paper's numbers from HyperFLAW's official evaluator, and the release code's results for comparison |
| `third_party/` | HyperFLAW's DeepLabV3 + MobileNetV2 (`hyperflaw/`, unchanged), used by the same-label control; a patch that runs DINOv2 on Python 3.9 and PyTorch 1.10 |

The federated self-training probe of the paper (Section 5.3, Table 2 and the "+ federated self-training" row of
Table 1) was run inside the HyperFLAW code base and is not part of this release.

## Setup

```bash
conda env create -f environment.yml && conda activate street2uav
```

The paper used Python 3.9, PyTorch 1.10.1 with CUDA 11.3, timm 0.9.16 and einops 0.8.2 on one RTX 4080 SUPER (32 GB).
Run all commands from the repository root.

**Backbone.** RADIO is loaded through `torch.hub` from [NVlabs/RADIO](https://github.com/NVlabs/RADIO) at commit
`fbd19ec` (the version used for the paper); the first run downloads the code from GitHub and the weights
`c-radio_v4-so400m_half.pth.tar` (1.2 GB, SHA-256 `d02697ede20f2716c4db12a9d3dab1c5a9b47d15a16cdcd46a69d08062b77aaf`)
from Hugging Face. Without network access at run time, clone the repository at that commit, download the weights, and
pass `--radio_repo /path/to/RADIO --radio_version /path/to/c-radio_v4-so400m_half.pth.tar` to every command. The
RADIO code and weights are distributed by NVIDIA under their own licenses (NVIDIA Source Code License for RADIO;
NVIDIA Open Model License for C-RADIOv4) and are not included here. The DINOv2 ablation loads
`facebookresearch/dinov2` at commit `7b187bd`, which needs Python >= 3.10 and PyTorch >= 2.0; in the paper's
environment, apply `third_party/dinov2_py39_torch110.patch` to a clone at that commit and pass `--dinov2_repo PATH`.

**Data.** Download `HyperFLAW-ACDC.zip` and `FLYAWARE-R-XL.zip` from the dataset links of the
[HyperFLAW repository](https://github.com/LTTM/HyperFLAW) and unzip both into one directory `DATA`:

```
DATA/HyperFLAW-ACDC/{rgb_anon,gt,splits}     ACDC fog/rain/night + Cityscapes clear (training, street validation, car test)
DATA/FLYAWARE-R-XL/{rgb,gt,splits}           drone test images (never used for training)
DATA/AeroScapes/data/{images,masks}/{train,val}   optional, only for the "+ AeroScapes labels" ablation
```

The archives used for the paper have SHA-256 `87c61bdcba019f1377ec2cba26dc33bb2e382c4e459e0684dea1ded594611120`
(HyperFLAW-ACDC.zip) and `84d615ba3e961fc8f5acd0e4b5f97e1604c12e38ee2d8168308fbafcb76a9bf5` (FLYAWARE-R-XL.zip).
AeroScapes is read in the layout of [dronefreak/Aeroscapes](https://huggingface.co/datasets/dronefreak/Aeroscapes).

## Evaluate

```bash
python -m street2uav.evaluate --ckpt checkpoints/final_s42.pth --data_root DATA --tta --out final_s42_tta.json
```

This prints joint 70.62, car 76.49 and drone 65.97 mIoU (drone clear 71.43, fog 69.41, rain 62.64, night 52.91) in
about 25 minutes on one RTX 4080 SUPER. Without `--tta` (a single pass at short side 720): joint 68.91, car 74.85,
drone 63.48, night 51.67. `--pairing official` scores the drone night images against the label maps of the split file
shipped with FA-R-XL. `bash scripts/table1.sh DATA` evaluates all released checkpoints with and without TTA.

`evaluate.py` re-implements HyperFLAW's evaluation (`StreamSegMetrics`, batch size 1, native resolution): car labels
are Cityscapes label ids mapped to the 19 train ids; on the drone images, the 19-class prediction is merged into the
five drone classes (sidewalk counts as road, terrain as vegetation, rider as person, truck and bus as car, all other
classes as wrong) and nearest-resized to the label size when it differs (night images are 2304 x 1296 with 3840 x 2160
labels); mIoU is averaged over the classes present in the labels, and the joint score is the mIoU of the summed car
and drone confusion matrices. On the released checkpoints it reproduces the numbers of HyperFLAW's official evaluator
([results/](results/README.md)).

To segment your own images (19 Cityscapes classes; colour maps and train-id maps are written to `preds/`):

```bash
python -m street2uav.predict --ckpt checkpoints/final_s42.pth --inputs my_images/ --out_dir preds --tta
```

## Train

```bash
python -m street2uav.train --data_root DATA --out runs/my_final_s42.pth --seed 42
python -m street2uav.validate --ckpt runs/my_final_s42.pth --data_root DATA --tta
python -m street2uav.evaluate --ckpt runs/my_final_s42.pth --data_root DATA --tta
```

The defaults are the final recipe: 2034 street training images (every 10th image of each source and weather is held
out for validation), random 16:9 crops covering 50-100% of the image height resized to 720 x 1280, horizontal flips,
colour jitter, short side 720 with the aspect ratio kept, 10 epochs of batch 8, AdamW (learning rate 5e-4, weight
decay 0.05) with cosine decay; only the decoder is trained (about 97 minutes). The street validation split is the
only signal used for model selection; `train.py` checks that no training or validation image is a test image.
Training is not bit-reproducible on GPU: seeds 42, 1 and 2 of the recipe give 65.73 ± 0.55 drone mIoU with TTA.

| script | what it runs |
|---|---|
| `scripts/train_final.sh DATA` | the final recipe with seeds 42, 1 and 2, then validation and test with TTA |
| `scripts/table3.sh DATA [setting ...]` | Table 3: linear head, frozen DINOv2-L, fine-tuned RADIO, short side 432, square input, no augmentation, ACDC only, Cityscapes only, no ACDC night images, + AeroScapes labels |
| `python -m street2uav.train --arch control --lr 3e-4 --epochs 40 ...` | the same-label control: HyperFLAW's DeepLabV3 + MobileNetV2 trained on the same street labels |

## License

This repository is released under [CC BY-NC-SA 4.0](LICENSE), the license of HyperFLAW, whose evaluation protocol,
data splits and network code (`third_party/hyperflaw/`, unchanged) it builds on. The overlays in `night_fix/overlays/`
contain FA-R-XL images (CC BY-NC-SA 4.0). The datasets and the backbones are subject to their own licenses.

## Acknowledgements

We thank the authors of HyperFLAW, FA-R-XL, RADIO, DINOv2, ACDC, Cityscapes and AeroScapes for releasing their code,
models and data.
