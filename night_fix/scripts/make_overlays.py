"""Visual check of the night pairing: each night test image with its same-name label map (left) and with the label map
the official split pairs it with (right), label colours at 50% opacity. Night images are brightened for display only.
The images are rendered locally from the dataset (they are not redistributed here).
  python night_fix/scripts/make_overlays.py --data_root DATA --out_dir night_overlays
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

SIZE = (960, 540)
FA_COLORS = np.array([[128, 0, 0], [128, 64, 128], [192, 0, 192], [0, 128, 0], [128, 128, 0], [64, 0, 128],
                      [64, 64, 0], [0, 0, 0]], dtype=np.uint8)
SHOW = {0: (70, 70, 70), 1: (128, 64, 128), 2: (0, 0, 142), 3: (107, 142, 35), 4: (107, 142, 35), 5: (0, 0, 142),
        6: (220, 20, 60)}  # building, road, car, vegetation, vegetation, car, person (Cityscapes colours)


def brighten(rgb):
    x = (rgb.astype(np.float64) / 255.0) ** 0.55
    return (255 * (x / max(np.percentile(x, 99.5), 1e-6)).clip(0, 1)).astype(np.uint8)


def overlay(rgb, label_path):
    lab = np.asarray(Image.open(label_path).convert("RGB").resize(SIZE, Image.NEAREST))
    out = rgb.astype(np.float64)
    for i, colour in SHOW.items():
        m = np.all(lab == FA_COLORS[i], axis=2)
        out[m] = 0.5 * out[m] + 0.5 * np.array(colour)
    return out.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    base = Path(a.data_root) / "FLYAWARE-R-XL"
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    d = json.load(open(base / "splits/heterogeneous/test/test_user.json"))["user_data"]["test_user"]
    for x, y in zip(d["x"], d["y"]):
        if "/night/" not in x:
            continue
        rgb = brighten(np.asarray(Image.open(base / "rgb" / x).convert("RGB").resize(SIZE, Image.BILINEAR)))
        pair = np.concatenate([overlay(rgb, base / "gt" / x), np.full((SIZE[1], 12, 3), 255, np.uint8),
                               overlay(rgb, base / "gt" / y)], axis=1)
        Image.fromarray(pair).save(out / f"{Path(x).stem}.jpg", quality=88)
        print(f"{x}: left = label {x}, right = official partner {y}")


if __name__ == "__main__":
    main()
