"""Base class for segmentation head."""

from abc import abstractmethod

import torch.nn as nn


class BaseClassifierHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.classifier = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    @abstractmethod
    def forward(self, x):
        pass
