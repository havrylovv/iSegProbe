"""Basic non-learnable upsamplers."""

import torch.nn.functional as F

from core.model.upsamplers import BaseUpsampler


class IdentityUpsampler(BaseUpsampler):
    """Identity upsampler that does not change the input tensor."""

    def __init__(self):
        super().__init__()

    def forward(self, source, guidance):
        return source


class NearestUpsampler(BaseUpsampler):
    def __init__(self):
        super().__init__()

    def forward(self, source, guidance):
        _, _, h, w = guidance.shape
        return F.interpolate(source, (h, w), mode="nearest")


class BilinearUpsampler(BaseUpsampler):
    def __init__(self):
        super().__init__()

    def forward(self, source, guidance):
        _, _, h, w = guidance.shape
        return F.interpolate(source, (h, w), mode="bilinear", align_corners=True)


class BicubicUpsampler(BaseUpsampler):
    def __init__(self):
        super().__init__()

    def forward(self, source, guidance):
        _, _, h, w = guidance.shape
        return F.interpolate(source, (h, w), mode="bicubic")
