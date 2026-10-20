"""Datasets, splits and label encodings.

Expected layout under --data_root (the HyperFLAW data release, https://github.com/LTTM/HyperFLAW):
  HyperFLAW-ACDC/   ACDC fog/rain/night + Cityscapes clear (rgb_anon/, gt/, splits/)     <- HyperFLAW-ACDC.zip
  FLYAWARE-R-XL/    real drone test images (rgb/, gt/, splits/)                           <- FLYAWARE-R-XL.zip
  AeroScapes/       optional, only for the "+ AeroScapes labels" ablation: data/images/{train,val}/*.jpg and
                    data/masks/{train,val}/*.png as in https://huggingface.co/datasets/dronefreak/Aeroscapes

Training (labeled street images only): ACDC train fog/rain/night (`*_gt_labelTrainIds.png`) and HyperFLAW's
Cityscapes clear train subset (`*_gtFine_labelIds.png`). Every 10th image of each source/weather group (sorted by path)
is held out as the street validation split, the only signal used for model selection; it always contains the held-out
images of all groups, whatever the training sources. Train and validation frame ids are checked to be disjoint from
both test splits. The drone test images are never read for training.

Tests (as in HyperFLAW's evaluation): car = HyperFLAW-ACDC/splits/heterogeneous/test/test_user.json (Cityscapes val
clear 214 + ACDC val fog 100 / rain 100 / night 106); drone = FA-R-XL test (258 images), with the corrected night
pairing by default (night_fix/test_user_nightfix.json; "official" = the split file shipped with the dataset).
"""
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

RELEASE_ROOT = Path(__file__).resolve().parents[1]
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
WEATHERS = ("clear", "fog", "rain", "night")
CLASSES = ("road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign", "vegetation",
           "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle")

# Cityscapes labelIds -> 19 train ids (HyperFLAW dataset/target/acdc.py), other ids -> 255 (ignored)
LABELS2TRAIN = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13,
                27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
LUT = np.full(256, 255, dtype=np.uint8)
for _k, _v in LABELS2TRAIN.items():
    LUT[_k] = _v

# FA-R-XL (HyperFLAW dataset/target/flyaware_r_xl.py): RGB label colours (order = raw id) -> train ids; colours not in
# the list and background/clutter -> 255. 19-class predictions are merged into the 5 drone classes with LABELS2COARSE
# (18 = "other", never present in the drone labels).
# building, road, static car, tree, low vegetation, moving car, human, background/clutter
FA_COLORS = np.array([[128, 0, 0], [128, 64, 128], [192, 0, 192], [0, 128, 0], [128, 128, 0], [64, 0, 128],
                      [64, 64, 0], [0, 0, 0]], dtype=np.uint8)
FA_LABELS2TRAIN = {0: 2, 1: 0, 2: 13, 3: 8, 4: 8, 5: 13, 6: 11, 7: 255}
FA_LUT = np.full(256, 255, dtype=np.uint8)
for _k, _v in FA_LABELS2TRAIN.items():
    FA_LUT[_k] = _v
LABELS2COARSE = {0: 0, 1: 0, 2: 2, 3: 18, 4: 18, 5: 18, 6: 18, 7: 18, 8: 8, 9: 8, 10: 18, 11: 11, 12: 11, 13: 13,
                 14: 13, 15: 13, 16: 18, 17: 18, 18: 18}
DRONE_CLASSES = {0: "road", 2: "building", 8: "vegetation", 11: "person", 13: "car"}

# AeroScapes (optional ablation): coarse groups of the 19 train ids, as merged by the drone evaluator.
# road, building, vegetation, person, car, other
GROUPS = [[0, 1], [2], [8, 9], [11, 12], [13, 14, 15], [3, 4, 5, 6, 7, 10, 16, 17, 18]]
GROUP_OF = np.zeros(19, dtype=np.int64)
for _g, _members in enumerate(GROUPS):
    GROUP_OF[_members] = _g
# AeroScapes ids: 0 background, 1 person, 2 bike, 3 car, 4 drone, 5 boat, 6 animal, 7 obstacle, 8 construction,
# 9 vegetation, 10 road, 11 sky. Bike and sky are "other" for the evaluator too; ambiguous classes are ignored.
AERO_LUT = np.full(256, 255, dtype=np.uint8)
for _k, _v in {1: 3, 2: 5, 3: 4, 8: 1, 9: 2, 10: 0, 11: 5}.items():
    AERO_LUT[_k] = _v


def frame_id(path):
    name = Path(str(path)).name
    for suffix in ("_rgb_anon.png", "_leftImg8bit.png", ".png", ".jpg"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def car_split_file(root):
    return Path(root) / "HyperFLAW-ACDC/splits/heterogeneous/test/test_user.json"


def drone_split_file(root, pairing="corrected"):
    if pairing == "corrected":
        return RELEASE_ROOT / "night_fix/test_user_nightfix.json"
    return Path(root) / "FLYAWARE-R-XL/splits/heterogeneous/test/test_user.json"


def street_index(root):
    """-> {group: [(rgb, label, weather, kind)]}, each group sorted by path; kind = label encoding."""
    acdc = Path(root) / "HyperFLAW-ACDC"
    groups = {}
    for w in ("fog", "rain", "night"):
        rows = []
        for rgb in sorted((acdc / "rgb_anon" / w / "train").rglob("*_rgb_anon.png")):
            rel = rgb.relative_to(acdc / "rgb_anon" / w / "train")
            gt = acdc / "gt" / w / "train" / rel.parent / f"{frame_id(rgb)}_gt_labelTrainIds.png"
            if gt.is_file():
                rows.append((str(rgb), str(gt), w, "trainid"))
        groups[f"acdc_{w}"] = rows
    rows = []
    for rgb in sorted((acdc / "rgb_anon/clear_from_cityscapes/train").glob("*_leftImg8bit.png")):
        gt = acdc / "gt/clear_from_cityscapes/train" / f"{frame_id(rgb)}_gtFine_labelIds.png"
        if gt.is_file():
            rows.append((str(rgb), str(gt), "clear", "labelid"))
    groups["cityscapes_clear"] = rows
    if not all(groups.values()):
        raise SystemExit(f"street images not found under {acdc}: {({k: len(v) for k, v in groups.items()})}")
    return groups


def street_split(root):
    """(train, val): every 10th image of each group is validation. Aborts if any image is a test image."""
    train, val = [], []
    for rows in street_index(root).values():
        val += rows[::10]
        train += [r for i, r in enumerate(rows) if i % 10]
    test_ids = set()
    for split in (car_split_file(root), drone_split_file(root, "official")):
        for user in json.load(open(split))["user_data"].values():
            test_ids |= {frame_id(x) for x in user["x"]}
    overlap = {frame_id(r[0]) for r in train + val} & test_ids
    if overlap:
        raise SystemExit(f"ABORT: {len(overlap)} training/validation images are test images, "
                         f"e.g. {sorted(overlap)[:3]}")
    return train, val


def select_train(rows, sources=("acdc", "cityscapes"), acdc_weathers=("fog", "rain", "night")):
    rows = [r for r in rows if ("cityscapes" if r[3] == "labelid" else "acdc") in sources]
    return [r for r in rows if r[3] != "trainid" or r[2] in acdc_weathers]


def aero_rows(root, split, frac=1.0):
    """AeroScapes rows (rgb, mask, 'clear', 'aero'); frac < 1 keeps a fixed subset (seed 0, independent of --seed)."""
    base = Path(root) / "AeroScapes/data"
    rows = []
    for rgb in sorted((base / "images" / split).glob("*.jpg")):
        mask = base / "masks" / split / f"{rgb.stem}.png"
        if mask.is_file():
            rows.append((str(rgb), str(mask), "clear", "aero"))
    if not rows:
        raise SystemExit(f"AeroScapes {split} not found under {base}")
    if frac < 1.0:
        keep = sorted(random.Random(0).sample(range(len(rows)), max(1, round(frac * len(rows)))))
        rows = [rows[i] for i in keep]
    return rows


def load_label(path, kind):
    a = np.array(Image.open(path), dtype=np.uint8)
    if kind == "aero":
        return AERO_LUT[a]
    return LUT[a] if kind == "labelid" else a


def encode_drone_label(path):
    """FA-R-XL RGB label -> (H, W) uint8 train ids, 255 = ignored (HyperFLAW FLYAWARERXL._encode_segmap)."""
    rgb = np.array(Image.open(path).convert("RGB"))
    raw = np.full(rgb.shape[:2], len(FA_COLORS), dtype=np.uint8)
    for i, c in enumerate(FA_COLORS):
        raw[np.all(rgb == c, axis=2)] = i
    return FA_LUT[raw]


def test_rows(root, domain, pairing="corrected"):
    """[(rgb, label, weather)] of the car or drone test split, in split-file order."""
    if domain == "car":
        split, base, weather_of = car_split_file(root), Path(root) / "HyperFLAW-ACDC", {
            "clear_from_cityscapes": "clear", "fog": "fog", "rain": "rain", "night": "night"}
        img_dir = lab_dir = base
    else:
        split, base, weather_of = drone_split_file(root, pairing), Path(root) / "FLYAWARE-R-XL", {
            "day": "clear", "fog": "fog", "rain": "rain", "night": "night"}
        img_dir, lab_dir = base / "rgb", base / "gt"
    d = json.load(open(split))["user_data"]["test_user"]
    return [(str(img_dir / x), str(lab_dir / y), weather_of[x.split("/")[1]]) for x, y in zip(d["x"], d["y"])]


def to_input(img):
    """PIL image -> ImageNet-normalized (3, H, W) tensor."""
    return T.functional.normalize(T.functional.to_tensor(img), MEAN, STD)


class TrainSet(Dataset):
    """Random crops with the target aspect ratio, resized to crop_hw.
    aug='basic': crop scale in [0.5, 1] of the image height, horizontal flip, colour jitter (final recipe);
    aug='none': central crop at full height when possible."""

    def __init__(self, rows, crop_hw=(720, 1280), aug="basic"):
        self.rows, self.crop_hw, self.aug = rows, crop_hw, aug
        self.jitter = T.ColorJitter(0.3, 0.3, 0.3, 0.05)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        rgb_p, gt_p, _, kind = self.rows[i]
        img = Image.open(rgb_p).convert("RGB")
        lab = Image.fromarray(load_label(gt_p, kind))
        ch, cw = self.crop_hw
        W, H = img.size
        if self.aug == "basic":
            s = random.uniform(0.5, 1.0)
            h = min(H, int(round(s * H)))
            wdt = min(W, int(round(h * cw / ch)))
            h = int(round(wdt * ch / cw))
            x0, y0 = random.randint(0, W - wdt), random.randint(0, H - h)
        else:
            wdt = min(W, int(round(H * cw / ch)))
            h = int(round(wdt * ch / cw))
            x0, y0 = (W - wdt) // 2, (H - h) // 2
        box = (x0, y0, x0 + wdt, y0 + h)
        img = img.crop(box).resize((cw, ch), Image.BILINEAR)
        lab = lab.crop(box).resize((cw, ch), Image.NEAREST)
        if self.aug == "basic":
            if random.random() < 0.5:
                img, lab = img.transpose(Image.FLIP_LEFT_RIGHT), lab.transpose(Image.FLIP_LEFT_RIGHT)
            img = self.jitter(img)
        return to_input(img), torch.from_numpy(np.array(lab, dtype=np.int64)), int(kind == "aero")
