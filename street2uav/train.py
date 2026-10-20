"""Supervised training on labeled street images (the defaults are the paper's final recipe).

Final recipe: frozen C-RADIOv4-SO400M + conv decoder, ACDC train fog/rain/night + Cityscapes clear (2034 images), random
crops resized to 720 x 1280, short side 720 with the aspect ratio kept, basic augmentation, 10 epochs of batch 8,
AdamW (lr 5e-4, weight decay 0.05), cosine schedule, cross-entropy. The street validation split is scored after every
epoch; no test image is read.

  python -m street2uav.train --data_root DATA --out checkpoints/my_final_s42.pth --seed 42

Ablations (Table 3 of the paper; scripts/ablations.sh has the exact commands):
  --head linear                      linear head instead of the conv decoder
  --backbone dinov2_vitl14           frozen DINOv2-L/14 (use --input_size 630 for the same patch grid as RADIO at 720)
  --backbone_lr 1e-5 --micro_batch 2 fine-tune all RADIO weights (the decoder keeps --lr); the backbone stays in eval
                                     mode and uses gradient checkpointing; saved next to the checkpoint as
                                     <name>.radio_ft.pth
  --input_size 432 / --no-keep_aspect / --aug none
  --sources acdc | cityscapes | acdc cityscapes aeroscapes, --acdc_weathers fog rain, with --total_iters 2540 for the
                                     same number of iterations as the final recipe (sampling with replacement)
  --arch control --lr 3e-4 --epochs 40   the same-label control (HyperFLAW's DeepLabV3 + MobileNetV2, fully trained)

AeroScapes labels are coarse (road / building / vegetation / person / car / other); they supervise the 19-class output
through group log-probabilities (logsumexp of the member classes' log-softmax), with the groups the drone evaluator
uses.
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from . import checkpoint, data
from .backbones import DINOV2_REPO, RADIO_REPO, RADIO_VERSION
from .validate import validate


def seg_loss(logits, y, coarse):
    """Cross-entropy averaged over the valid pixels of the batch; coarse (AeroScapes) samples are scored on group
    log-probabilities."""
    if not coarse.any():
        return F.cross_entropy(logits, y, ignore_index=255)
    logp = F.log_softmax(logits, 1)
    total, count = logits.new_zeros(()), 0
    fine = ~coarse
    if fine.any():
        total = total + F.nll_loss(logp[fine], y[fine], ignore_index=255, reduction="sum")
        count += int((y[fine] != 255).sum())
    glogp = torch.stack([torch.logsumexp(logp[coarse][:, g], 1) for g in data.GROUPS], 1)
    total = total + F.nll_loss(glogp, y[coarse], ignore_index=255, reduction="sum")
    count += int((y[coarse] != 255).sum())
    return total / max(count, 1)


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out", required=True, help="checkpoint path; the training log is written to <out>.json")
    ap.add_argument("--sources", nargs="+", default=["acdc", "cityscapes"],
                    choices=["acdc", "cityscapes", "aeroscapes"])
    ap.add_argument("--acdc_weathers", nargs="+", default=["fog", "rain", "night"], choices=["fog", "rain", "night"],
                    help="ACDC training conditions kept (the validation split always holds all of them)")
    ap.add_argument("--drone_frac", type=float, default=1.0, help="fraction of AeroScapes train (fixed subset)")
    ap.add_argument("--mix_drone", type=float, default=0.5, help="share of AeroScapes samples with street + drone")
    ap.add_argument("--total_iters", type=int, default=0, help="train for n iterations, sampling with replacement")
    ap.add_argument("--log_every", type=int, default=254, help="logging interval with --total_iters")
    ap.add_argument("--arch", default="segmodel", choices=["segmodel", "control"])
    ap.add_argument("--backbone", default="radio", choices=["radio", "dinov2_vitl14"])
    ap.add_argument("--head", default="conv", choices=["conv", "linear"])
    ap.add_argument("--backbone_lr", type=float, default=0.0, help=">0 fine-tunes RADIO at this learning rate")
    ap.add_argument("--micro_batch", type=int, default=0, help="gradient accumulation over micro-batches of m images")
    ap.add_argument("--radio_version", default=RADIO_VERSION, help="RADIO version name or local weights file")
    ap.add_argument("--radio_repo", default=RADIO_REPO, help="NVlabs/RADIO[:ref] or a local clone")
    ap.add_argument("--dinov2_repo", default=DINOV2_REPO, help="facebookresearch/dinov2[:ref] or a local clone")
    ap.add_argument("--input_size", type=int, default=720, help="short side fed to the backbone")
    ap.add_argument("--keep_aspect", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--crop_h", type=int, default=720)
    ap.add_argument("--crop_w", type=int, default=1280)
    ap.add_argument("--aug", default="basic", choices=["basic", "none"])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--val_every", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    return ap


def main():
    a = build_parser().parse_args()
    assert not (a.backbone_lr and (a.backbone != "radio" or a.arch != "segmodel")), \
        "--backbone_lr fine-tunes RADIO only"
    assert not (a.micro_batch and "aeroscapes" in a.sources), "--micro_batch supports the street (fine-label) loss only"

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = torch.device("cuda")
    train_all, val_rows = data.street_split(a.data_root)  # the same validation split for every configuration
    train_rows = data.select_train(train_all, a.sources, a.acdc_weathers)
    n_street = len(train_rows)
    drone_rows = data.aero_rows(a.data_root, "train", a.drone_frac) if "aeroscapes" in a.sources else []
    train_rows += drone_rows
    print(f"train {len(train_rows)} images (street {n_street}, AeroScapes {len(drone_rows)}), "
          f"street validation {len(val_rows)} (disjoint from the test splits)", flush=True)
    dataset = data.TrainSet(train_rows, (a.crop_h, a.crop_w), a.aug)
    if a.total_iters:
        if n_street and drone_rows:
            w = [(1 - a.mix_drone) / n_street] * n_street + [a.mix_drone / len(drone_rows)] * len(drone_rows)
        else:
            w = [1.0] * len(train_rows)
        sampler = WeightedRandomSampler(w, num_samples=a.total_iters * a.batch_size, replacement=True,
                                        generator=torch.Generator().manual_seed(a.seed))
        loader = DataLoader(dataset, batch_size=a.batch_size, sampler=sampler, num_workers=a.num_workers,
                            pin_memory=True, drop_last=True)
    else:
        loader = DataLoader(dataset, batch_size=a.batch_size, shuffle=True, num_workers=a.num_workers,
                            pin_memory=True, drop_last=True)

    cfg = {"arch": a.arch, "backbone": a.backbone, "head": a.head, "input_size": a.input_size,
           "keep_aspect": a.keep_aspect, "num_classes": 19}
    model = checkpoint.build_model(cfg, radio_version=a.radio_version, radio_repo=a.radio_repo,
                                   dinov2_repo=a.dinov2_repo).to(device)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable parameters {sum(p.numel() for p in params):,}", flush=True)
    groups = [{"params": params, "lr": a.lr}]
    if a.backbone_lr > 0:  # fine-tune all RADIO weights (kept in eval mode by SegModel.train)
        radio = model.backbone
        for p in radio.parameters():
            p.requires_grad_(True)
        radio.model.set_grad_checkpointing(True)
        model.trainable_backbone = True
        groups.append({"params": list(radio.parameters()), "lr": a.backbone_lr})
        print(f"fine-tuned backbone parameters {sum(p.numel() for p in groups[1]['params']):,}", flush=True)
    opt = torch.optim.AdamW(groups, lr=a.lr, weight_decay=a.weight_decay)

    def forward_backward(x, y, k):
        """Gradients of one batch; returns the loss. --micro_batch keeps the loss of the whole batch: cross-entropy
        summed over the micro-batches' valid pixels / valid pixels of the batch."""
        opt.zero_grad(set_to_none=True)
        if not a.micro_batch:
            loss = seg_loss(model(x), y, k)
            loss.backward()
            return loss.item()
        n_valid, total_loss = max(int((y != 255).sum()), 1), 0.0
        for xs, ys in zip(x.split(a.micro_batch), y.split(a.micro_batch)):
            loss = F.cross_entropy(model(xs), ys, ignore_index=255, reduction="sum") / n_valid
            loss.backward()
            total_loss += loss.item()
        return total_loss

    total = a.total_iters if a.total_iters else a.epochs * len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: 0.5 * (1 + math.cos(math.pi * it / total)))
    history, t0 = [], time.time()
    if a.total_iters:  # one pass over the sampler, validation at the end
        run, n_drone = 0.0, 0
        for it, (x, y, k) in enumerate(loader, 1):
            x, y, k = x.to(device, non_blocking=True), y.to(device, non_blocking=True), k.to(device)
            run += forward_backward(x, y, k.bool())
            opt.step()
            sched.step()
            n_drone += int(k.sum())
            if it % a.log_every == 0 or it == total:
                rec = {"iter": it, "train_loss": round(run / a.log_every, 4),
                       "drone_share": round(n_drone / (a.log_every * a.batch_size), 3),
                       "minutes": round((time.time() - t0) / 60, 1)}
                if it == total:
                    rec["val"] = validate(model, val_rows, device)
                history.append(rec)
                print(json.dumps(rec), flush=True)
                run, n_drone = 0.0, 0
    for ep in range(1, (0 if a.total_iters else a.epochs) + 1):
        run = 0.0
        for x, y, k in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            run += forward_backward(x, y, k.bool().to(device))
            opt.step()
            sched.step()
        rec = {"epoch": ep, "train_loss": round(run / len(loader), 4), "minutes": round((time.time() - t0) / 60, 1)}
        if ep % a.val_every == 0 or ep == a.epochs:
            rec["val"] = validate(model, val_rows, device)
        history.append(rec)
        print(json.dumps(rec), flush=True)

    out = Path(a.out)
    if a.backbone_lr > 0:
        ft = out.with_suffix(".radio_ft.pth")
        torch.save({"base": a.radio_version, "radio_model_state": model.backbone.state_dict()}, ft)
        cfg["radio_finetuned"] = ft.name
        print(f"saved the fine-tuned backbone {ft}", flush=True)
    checkpoint.save(out, model, cfg, vars(a))
    json.dump({"args": vars(a), "config": cfg, "n_train": len(train_rows), "n_train_street": n_street,
               "n_train_drone": len(drone_rows), "n_val": len(val_rows), "history": history,
               "final_val": history[-1].get("val")}, open(out.with_suffix(".json"), "w"), indent=2)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
