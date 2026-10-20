"""Segment your own images: writes <name>_trainid.png (19 Cityscapes train ids) and <name>_color.png (Cityscapes
colours) for every input image.
  python -m street2uav.predict --ckpt checkpoints/final_s42.pth --inputs img1.jpg dir_of_images/ --out_dir preds --tta
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import checkpoint, data

PALETTE = np.array([(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153), (153, 153, 153),
                    (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
                    (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)],
                   dtype=np.uint8)
SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--inputs", nargs="+", required=True, help="image files or directories")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tta", action="store_true", help="flip x short sides s and 4s/3")
    ap.add_argument("--radio_version", default=checkpoint.RADIO_VERSION)
    ap.add_argument("--radio_repo", default=checkpoint.RADIO_REPO)
    ap.add_argument("--dinov2_repo", default=checkpoint.DINOV2_REPO)
    a = ap.parse_args()
    files = []
    for p in map(Path, a.inputs):
        files += sorted(f for f in p.iterdir() if f.suffix.lower() in SUFFIXES) if p.is_dir() else [p]
    model, _ = checkpoint.load(a.ckpt, tta=a.tta, radio_version=a.radio_version, radio_repo=a.radio_repo,
                               dinov2_repo=a.dinov2_repo)
    model = model.to("cuda")
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for f in files:
            x = data.to_input(Image.open(f).convert("RGB")).unsqueeze(0).cuda()
            pred = model(x).argmax(1)[0].cpu().numpy().astype(np.uint8)
            Image.fromarray(pred).save(out / f"{f.stem}_trainid.png")
            Image.fromarray(PALETTE[pred]).save(out / f"{f.stem}_color.png")
            print(f"{f} -> {out / (f.stem + '_color.png')}", flush=True)


if __name__ == "__main__":
    main()
