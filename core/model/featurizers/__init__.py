from abc import ABC

from torch import nn


class BaseFeaturizer(ABC, nn.Module):
    """Base class for all featurizers."""

    def forward(self, x, additional_features=None, **kwargs):
        """Featurize the input data."""
        raise NotImplementedError


from .DINO import DINOFeaturizer
from .DINOv2 import DINOv2Featurizer
from .MaskCLIP import MaskCLIPFeaturizer
from .simple_ViT import SimpleViTFeaturizer

__all__ = [
    "MaskCLIPFeaturizer",
    "DINOv2Featurizer",
    "SimpleViTFeaturizer",
    "DINOFeaturizer",
]
