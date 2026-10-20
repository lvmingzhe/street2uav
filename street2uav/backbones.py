"""Vision foundation model backbones, loaded once per process and kept frozen.

RADIO: C-RADIOv4-SO400M through the torch.hub entry of NVlabs/RADIO. `version` is either the registered name
"c-radio_v4-so400m" (weights downloaded from https://huggingface.co/nvidia/C-RADIOv4-SO400M) or a local copy of
c-radio_v4-so400m_half.pth.tar; `repo` is "NVlabs/RADIO[:ref]" on GitHub or the path of a local clone.
A RADIO backbone fine-tuned by train.py (--backbone_lr) is saved as <checkpoint>.radio_ft.pth and is loaded on top of
the version it started from.
DINOv2: ViT-L/14 through the torch.hub entry of facebookresearch/dinov2 (GitHub or a local clone).
"""
import inspect
import os

import torch

RADIO_REPO = "NVlabs/RADIO:fbd19ec1e68483482d7d96a59cb639880a8b33ed"  # the RADIO commit used for the paper
RADIO_VERSION = "c-radio_v4-so400m"
DINOV2_REPO = "facebookresearch/dinov2:7b187bd4df8efce2cbcbbb67bd01532c19bf4c9c"  # Python >= 3.10, PyTorch >= 2.0;
# for the paper's environment apply third_party/dinov2_py39_torch110.patch to a clone and pass its path

_CACHE = {}


def _drop_weights_only(fn):
    def wrapped(*args, **kwargs):
        kwargs.pop("weights_only", None)
        return fn(*args, **kwargs)
    return wrapped


# The RADIO hub entry passes weights_only=False, which PyTorch < 1.13 does not accept (the paper used PyTorch 1.10).
if "weights_only" not in inspect.signature(torch.load).parameters:
    torch.load = _drop_weights_only(torch.load)
if "weights_only" not in inspect.signature(torch.hub.load_state_dict_from_url).parameters:
    torch.hub.load_state_dict_from_url = _drop_weights_only(torch.hub.load_state_dict_from_url)


def _hub_load(repo, entry, **kwargs):
    if os.path.isdir(repo):
        return torch.hub.load(repo, entry, source="local", **kwargs)
    return torch.hub.load(repo, entry, source="github", skip_validation=True, **kwargs)


def load_radio(version=RADIO_VERSION, repo=RADIO_REPO):
    key = ("radio", str(version))
    if key not in _CACHE:
        if str(version).endswith(".radio_ft.pth"):
            ft = torch.load(version, map_location="cpu")
            model = _hub_load(repo, "radio_model", version=str(ft["base"]), progress=True)
            model.load_state_dict(ft["radio_model_state"], strict=True)
        else:
            model = _hub_load(repo, "radio_model", version=str(version), progress=True)
        _CACHE[key] = model
    return freeze(_CACHE[key])


def load_dinov2(name="dinov2_vitl14", repo=DINOV2_REPO):
    key = ("dinov2", name)
    if key not in _CACHE:
        _CACHE[key] = _hub_load(repo, name, pretrained=True)
    return freeze(_CACHE[key])


def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)
    return model.eval()
