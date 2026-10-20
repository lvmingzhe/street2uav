"""Checkpoint format: {"state_dict": trainable weights, "config": model configuration,
"train_args": train.py arguments}.

config keys: arch ("segmodel" | "control"), backbone ("radio" | "dinov2_vitl14"), head ("conv" | "linear"),
input_size (short side), keep_aspect, num_classes, and radio_finetuned (file name of the fine-tuned RADIO weights next
to the checkpoint, only with train.py --backbone_lr).
"""
from pathlib import Path

import torch

from .backbones import DINOV2_REPO, RADIO_REPO, RADIO_VERSION
from .control import CNNSegControl
from .model import SegModel, tta_scales


def build_model(cfg, tta=False, radio_version=RADIO_VERSION, radio_repo=RADIO_REPO, dinov2_repo=DINOV2_REPO,
                imagenet_init=True, ckpt_dir=None):
    """Model of a config; tta=True enables the fixed test-time augmentation (flip x short sides s and 4s/3)."""
    if cfg["arch"] == "control":
        scales = tta_scales(cfg["input_size"], 16) if tta else ()
        return CNNSegControl(cfg["num_classes"], cfg["input_size"], cfg["keep_aspect"], imagenet_init=imagenet_init,
                             tta_flip=tta, tta_scales=scales)
    if cfg.get("radio_finetuned"):
        radio_version = str(Path(ckpt_dir or ".") / cfg["radio_finetuned"])
    patch = 14 if cfg["backbone"].startswith("dinov2") else 16
    return SegModel(cfg["backbone"], cfg["head"], cfg["input_size"], cfg["keep_aspect"], cfg["num_classes"],
                    radio_version=radio_version, radio_repo=radio_repo, dinov2_repo=dinov2_repo,
                    tta_flip=tta, tta_scales=tta_scales(cfg["input_size"], patch) if tta else ())


def save(path, model, cfg, train_args):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": cfg, "train_args": train_args}, path)


def load(path, tta=False, radio_version=RADIO_VERSION, radio_repo=RADIO_REPO, dinov2_repo=DINOV2_REPO):
    """-> (model in eval mode, checkpoint dict). The backbone weights come from torch.hub (or radio_version)."""
    ck = torch.load(path, map_location="cpu")
    model = build_model(ck["config"], tta=tta, radio_version=radio_version, radio_repo=radio_repo,
                        dinov2_repo=dinov2_repo, imagenet_init=False, ckpt_dir=Path(path).parent)
    model.load_state_dict(ck["state_dict"], strict=True)
    return model.eval(), ck
