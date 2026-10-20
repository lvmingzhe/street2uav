"""Test-set evaluation with HyperFLAW's metric (stand-alone re-implementation of its official evaluation).

Car test: 19-class confusion matrix at native resolution. Drone test (FA-R-XL): 19-class predictions merged into the
5 drone classes (road, building, vegetation, person, car; other classes -> "other") and nearest-resized to the label
size when it differs from the image size. Joint: mIoU of the summed car + drone matrices (HyperFLAW's mean over the two
test sets). mIoU over the classes present in the labels; per-weather mIoU from per-weather matrices. Batch size 1, fp32.

  python -m street2uav.evaluate --ckpt checkpoints/final_s42.pth --data_root DATA --tta --out results/final_s42.json
  --pairing official   scores the drone night images against the label maps paired by the dataset's split file
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from . import checkpoint, data
from .metrics import class_iou, confusion, miou

COARSE_LUT = torch.tensor([data.LABELS2COARSE[i] for i in range(19)])
CAR_CLASSES = {0: "road", 2: "building", 8: "vegetation", 11: "person", 13: "car"}  # reported for the street images


def first(batch):
    return batch[0]


class TestSet(Dataset):
    def __init__(self, rows, domain):
        self.rows, self.domain = rows, domain

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        rgb_p, lab_p, weather = self.rows[i]
        x = data.to_input(Image.open(rgb_p).convert("RGB"))
        if self.domain == "car":
            lab = data.LUT[np.array(Image.open(lab_p), dtype=np.uint8)]
        else:
            lab = data.encode_drone_label(lab_p)
        return x, torch.from_numpy(lab.astype(np.int64)), weather


@torch.no_grad()
def evaluate_domain(model, rows, domain, device, num_workers=4, log_every=50):
    """-> (total confusion matrix, {weather: confusion matrix})."""
    cm = np.zeros((19, 19), dtype=np.int64)
    cm_w = {w: np.zeros((19, 19), dtype=np.int64) for w in data.WEATHERS}
    loader = DataLoader(TestSet(rows, domain), batch_size=1, num_workers=num_workers, collate_fn=first)
    t0 = time.time()
    for i, (x, lab, weather) in enumerate(loader, 1):
        _, pred = model(x.unsqueeze(0).to(device)).max(dim=1)  # (1, H, W)
        if pred.shape[1:] != lab.shape:
            pred = F.interpolate(pred.unsqueeze(0).double(), lab.shape, mode="nearest").squeeze(0).long()
        pred = pred[0].cpu()
        if domain == "drone":
            pred = COARSE_LUT[pred]
        c = confusion(lab.numpy(), pred.numpy())
        cm += c
        cm_w[weather] += c
        if i % log_every == 0 or i == len(rows):
            print(f"[{domain}] {i}/{len(rows)}  {time.time() - t0:.0f}s", flush=True)
    return cm, cm_w


def summarize(cm, cm_w, names):
    return {"miou": miou(cm), "weather": {w: miou(c) for w, c in cm_w.items() if c.sum() > 0},
            "class_iou": class_iou(cm, names)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--tta", action="store_true", help="flip x short sides s and 4s/3 (the paper's default)")
    ap.add_argument("--pairing", default="corrected", choices=["corrected", "official"],
                    help="drone night labels: same-name label maps (corrected) or the dataset's split file")
    ap.add_argument("--domains", nargs="+", default=["car", "drone"], choices=["car", "drone"])
    ap.add_argument("--radio_version", default=checkpoint.RADIO_VERSION)
    ap.add_argument("--radio_repo", default=checkpoint.RADIO_REPO)
    ap.add_argument("--dinov2_repo", default=checkpoint.DINOV2_REPO)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = False  # as HyperFLAW's evaluation
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    model, ck = checkpoint.load(a.ckpt, tta=a.tta, radio_version=a.radio_version, radio_repo=a.radio_repo,
                                dinov2_repo=a.dinov2_repo)
    model = model.to(device)
    res = {"ckpt": a.ckpt, "config": ck["config"], "tta": a.tta, "pairing": a.pairing}
    cms = {}
    for domain in a.domains:
        rows = data.test_rows(a.data_root, domain, a.pairing)
        cm, cm_w = evaluate_domain(model, rows, domain, device, a.num_workers)
        cms[domain] = (cm, cm_w)
        names = CAR_CLASSES if domain == "car" else data.DRONE_CLASSES
        res[domain] = {"n_images": len(rows), **summarize(cm, cm_w, names)}
    if len(cms) == 2:
        res["joint"] = miou(cms["car"][0] + cms["drone"][0])
    res["confusion"] = {d: {"all": cm.tolist(), **{w: c.tolist() for w, c in cm_w.items()}}
                        for d, (cm, cm_w) in cms.items()}
    short = {k: v for k, v in res.items() if k != "confusion"}
    print(json.dumps(short, indent=2))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
