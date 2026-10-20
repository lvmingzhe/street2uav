"""DeepLabV3 + MobileNetV2 with condition-specific batch normalization, copied unchanged from HyperFLAW
(https://github.com/LTTM/HyperFLAW, commit 682b58d, 02_adaptation/src/modules/), licensed CC BY-NC-SA 4.0.
Used only for the same-label control of the paper (street2uav/control.py), with a single normalization branch."""
from .deeplabv3 import deeplabv3_mobilenetv2  # noqa: F401
