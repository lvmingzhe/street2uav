"""Frozen vision foundation model + small trainable decoder (Section 3 of the paper).

Input: an image tensor normalized with the ImageNet mean and std, at any resolution. The image is resized so that its
short side is `input_size` (aspect ratio kept, long side rounded to a multiple of the patch size; with
keep_aspect=False it is resized to input_size x input_size), the backbone produces patch tokens on a grid of
H'/p x W'/p, the decoder predicts 19-class logits on that grid, and the logits are upsampled bilinearly to the input
resolution.

Decoders:
  conv   (final model): 1x1 conv (C -> 256), ReLU, 3x3 conv (256 -> 256), ReLU, 1x1 conv (256 -> 19); 0.89M parameters
  linear (ablation):    1x1 conv (C -> 19)
Test-time augmentation (evaluation only): softmax averaged over horizontal flips x several short sides; the output is
then the log of the averaged probabilities.

The backbone is not a registered submodule, so state_dict() holds the decoder only (3.5 MB for the final model).
"""
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import DINOV2_REPO, RADIO_REPO, RADIO_VERSION, load_dinov2, load_radio

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
BACKBONES = {"radio": (16, 1152), "dinov2_vitl14": (14, 1024)}  # patch size, token dimension


class Decoder(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 19, mode: str = "conv", mid: int = 256):
        super().__init__()
        if mode not in ("conv", "linear"):
            raise ValueError(f"unknown decoder: {mode}")
        self.mode = mode
        if mode == "linear":
            self.cls = nn.Conv2d(in_dim, num_classes, kernel_size=1)
        else:
            self.proj = nn.Conv2d(in_dim, mid, kernel_size=1)
            self.conv3 = nn.Conv2d(mid, mid, kernel_size=3, padding=1)
            self.cls = nn.Conv2d(mid, num_classes, kernel_size=1)
            self.relu = nn.ReLU(inplace=True)

    def forward(self, tokens: torch.Tensor, size) -> torch.Tensor:
        if self.mode == "linear":
            x = self.cls(tokens)
        else:
            x = self.cls(self.relu(self.conv3(self.relu(self.proj(tokens)))))
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class SegModel(nn.Module):
    def __init__(self, backbone: str = "radio", head: str = "conv", input_size: int = 720, keep_aspect: bool = True,
                 num_classes: int = 19, radio_version: str = RADIO_VERSION, radio_repo: str = RADIO_REPO,
                 dinov2_repo: str = DINOV2_REPO, trainable_backbone: bool = False, tta_flip: bool = False,
                 tta_scales: Sequence[int] = ()):
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError(f"unknown backbone: {backbone}")
        self.backbone_name = backbone
        self.patch, dim = BACKBONES[backbone]
        assert input_size % self.patch == 0, f"input_size must be a multiple of the patch size ({self.patch})"
        self.input_size, self.keep_aspect = input_size, keep_aspect
        self.trainable_backbone = trainable_backbone
        self.tta_flip, self.tta_scales = tta_flip, tuple(int(s) for s in tta_scales)
        if backbone.startswith("dinov2"):
            bb = load_dinov2(backbone, dinov2_repo)
        else:
            bb = load_radio(radio_version, radio_repo)
        self._backbone = {"model": bb}  # not registered: keeps the backbone out of state_dict() and of .train()
        self.head = Decoder(dim, num_classes, head)
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1), persistent=False)

    @property
    def backbone(self) -> nn.Module:
        return self._backbone["model"]

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self._backbone["model"] = self.backbone.to(*args, **kwargs)
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()  # always: RADIO's training mode adds stochastic position-embedding behaviour
        return self

    def target_hw(self, h: int, w: int, size: int):
        p = self.patch
        if not self.keep_aspect:
            return size, size
        if h <= w:
            return size, max(p, int(round(w * size / h / p)) * p)
        return max(p, int(round(h * size / w / p)) * p), size

    def tokens(self, x: torch.Tensor, size: Optional[int] = None) -> torch.Tensor:
        """Patch tokens (B, C, H'/p, W'/p) of an ImageNet-normalized batch."""
        th, tw = self.target_hw(x.shape[-2], x.shape[-1], size or self.input_size)
        if next(self.backbone.parameters()).device != x.device:  # e.g. after .cuda(), which skips the backbone
            self._backbone["model"] = self.backbone.to(x.device)
        with torch.set_grad_enabled(self.trainable_backbone and torch.is_grad_enabled()):
            if self.backbone_name.startswith("dinov2"):  # DINOv2 takes the ImageNet-normalized image
                if x.shape[-2:] != (th, tw):
                    x = F.interpolate(x, size=(th, tw), mode="bilinear", align_corners=False)
                feats = self.backbone.forward_features(x)["x_norm_patchtokens"]
            else:  # RADIO takes [0, 1] RGB and normalizes internally
                x01 = x * self.std + self.mean
                if x01.shape[-2:] != (th, tw):
                    x01 = F.interpolate(x01, size=(th, tw), mode="bilinear", align_corners=False)
                out = self.backbone(x01)  # RadioOutput(summary, features)
                feats = out.features if hasattr(out, "features") else out[1]
        B, N, C = feats.shape
        return feats.transpose(1, 2).reshape(B, C, th // self.patch, tw // self.patch)

    def forward_single(self, x: torch.Tensor, size: Optional[int] = None) -> torch.Tensor:
        return self.head(self.tokens(x, size), size=x.shape[-2:])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Logits (B, 19, H, W); in eval mode with TTA, log of the averaged probabilities."""
        if self.training or not (self.tta_flip or self.tta_scales):
            return self.forward_single(x)
        probs = []
        for size in (self.tta_scales or (self.input_size,)):
            for flip in ((False, True) if self.tta_flip else (False,)):
                out = self.forward_single(torch.flip(x, dims=[3]) if flip else x, size)
                probs.append(F.softmax(torch.flip(out, dims=[3]) if flip else out, dim=1))
        return torch.log(torch.stack(probs).mean(0) + 1e-8)


def tta_scales(input_size: int, patch: int = 16):
    """The fixed test-time scales: the training short side and 4/3 of it, rounded to a multiple of the patch size
    (720 -> 720, 960)."""
    return input_size, round(input_size * 4 / 3 / patch) * patch
