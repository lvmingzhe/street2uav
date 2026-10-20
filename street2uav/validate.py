"""Street validation mIoU, the only signal used for model selection (never a test image).

Images and labels are resized to half their native size; mIoU over the 19 classes, overall and per weather.
  python -m street2uav.validate --ckpt checkpoints/final_s42.pth --data_root DATA [--tta]
"""
import argparse
import json

import numpy as np
import torch
from PIL import Image

from . import checkpoint, data
from .metrics import confusion, miou


@torch.no_grad()
def validate(model, rows, device, scale=0.5):
    was_training = model.training
    model.eval()
    cms = {w: np.zeros((19, 19), dtype=np.int64) for w in data.WEATHERS}
    for rgb_p, gt_p, w, kind in rows:
        img = Image.open(rgb_p).convert("RGB")
        W, H = img.size
        size = (int(W * scale), int(H * scale))
        x = data.to_input(img.resize(size, Image.BILINEAR)).unsqueeze(0).to(device)
        lab = np.array(Image.fromarray(data.load_label(gt_p, kind)).resize(size, Image.NEAREST), dtype=np.int64)
        pred = model(x).argmax(1)[0].cpu().numpy()
        cms[w] += confusion(lab, pred)
    model.train(was_training)
    out = {"miou": round(miou(sum(cms.values())), 2)}
    out.update({f"miou_{w}": round(miou(cm), 2) for w, cm in cms.items() if cm.sum() > 0})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--tta", action="store_true", help="flip x short sides s and 4s/3")
    ap.add_argument("--radio_version", default=checkpoint.RADIO_VERSION)
    ap.add_argument("--radio_repo", default=checkpoint.RADIO_REPO)
    ap.add_argument("--dinov2_repo", default=checkpoint.DINOV2_REPO)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    model, _ = checkpoint.load(a.ckpt, tta=a.tta, radio_version=a.radio_version, radio_repo=a.radio_repo,
                               dinov2_repo=a.dinov2_repo)
    _, val_rows = data.street_split(a.data_root)
    res = {"ckpt": a.ckpt, "tta": a.tta, "n_images": len(val_rows),
           "val": validate(model.to("cuda"), val_rows, "cuda")}
    print(json.dumps(res, indent=2))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
