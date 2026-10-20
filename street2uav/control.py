"""Same-label control (Table 1): HyperFLAW's own network, DeepLabV3 + MobileNetV2, trained on the same street labels
with the same input pipeline as SegModel.

The network (third_party/hyperflaw, unchanged) starts from ImageNet-initialized MobileNetV2 weights as in HyperFLAW and
is fully trained. Every image uses the same batch-normalization branch, i.e. ordinary BN without weather conditioning.
The image is resized so that its short side is `input_size` (aspect ratio kept, long side a multiple of 16), and the
logits are resized back to the input resolution. Test-time augmentation as in SegModel.
"""
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from third_party.hyperflaw.deeplabv3 import _deeplabv3_mobilenetv2, deeplabv3_mobilenetv2
from third_party.hyperflaw.mobilenetv2_conditional import MobileNetV2

NUM_CONDITIONS = 5  # HyperFLAW's weather branches; only branch 1 is used


class CNNSegControl(nn.Module):
    def __init__(self, num_classes: int = 19, input_size: int = 720, keep_aspect: bool = True,
                 imagenet_init: bool = True, tta_flip: bool = False, tta_scales: Sequence[int] = ()):
        super().__init__()
        assert input_size % 16 == 0
        if imagenet_init:  # downloads the ImageNet MobileNetV2 weights used by HyperFLAW
            self.net = deeplabv3_mobilenetv2(num_classes, num_conditions=NUM_CONDITIONS)
        else:  # the same network without the download, for loading a trained checkpoint
            self.net = _deeplabv3_mobilenetv2(MobileNetV2(width_mult=1, in_channels=3, num_conditions=NUM_CONDITIONS),
                                              num_classes, NUM_CONDITIONS)
        self.input_size, self.keep_aspect = input_size, keep_aspect
        self.tta_flip, self.tta_scales = tta_flip, tuple(int(s) for s in tta_scales)

    def target_hw(self, h: int, w: int, size: int):
        h, w, size = int(h), int(w), int(size)
        if not self.keep_aspect:
            return size, size
        if h <= w:
            return size, max(16, int(round(w * size / h / 16)) * 16)
        return max(16, int(round(h * size / w / 16)) * 16), size

    def forward_single(self, x: torch.Tensor, size: Optional[int] = None) -> torch.Tensor:
        h, w = x.shape[-2:]
        th, tw = self.target_hw(h, w, size or self.input_size)
        xi = F.interpolate(x, size=(th, tw), mode="bilinear", align_corners=False) if (th, tw) != (h, w) else x
        out = self.net(xi, [1] * x.shape[0])["out"]  # one BN branch for every image
        return F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False) if (th, tw) != (h, w) else out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training or not (self.tta_flip or self.tta_scales):
            return self.forward_single(x)
        probs = []
        for size in (self.tta_scales or (self.input_size,)):
            for flip in ((False, True) if self.tta_flip else (False,)):
                out = self.forward_single(torch.flip(x, dims=[3]) if flip else x, size)
                probs.append(F.softmax(torch.flip(out, dims=[3]) if flip else out, dim=1))
        return torch.log(torch.stack(probs).mean(0) + 1e-8)
