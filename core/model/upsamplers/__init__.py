from abc import ABC, abstractmethod

import torch.nn as nn


class BaseUpsampler(nn.Module, ABC):
    """Base class for upsampling modules."""

    @abstractmethod
    def forward(self, source, guidance):
        pass


from .basic_upsamplers import (
    BicubicUpsampler,
    BilinearUpsampler,
    IdentityUpsampler,
    NearestUpsampler,
)
from .JBUFeatUp import JBUFeatUpUpsampler
from .LiFT import LiFTUpsampler
from .LoftUp import LoftUpUpsampler

# used to load upsamplers from config
UPSAMPLER_REGISTRY = {
    "identity": IdentityUpsampler,
    "nearest": NearestUpsampler,
    "bilinear": BilinearUpsampler,
    "bicubic": BicubicUpsampler,
    "jbu_featup": JBUFeatUpUpsampler,
    "lift": LiFTUpsampler,
    "loftup": LoftUpUpsampler,
}

__all__ = [
    "BicubicUpsampler",
    "BilinearUpsampler",
    "IdentityUpsampler",
    "NearestUpsampler",
    "JBUFeatUpUpsampler",
    "LiFTUpsampler",
    "LoftUpUpsampler",
    "BaseUpsampler",
]
