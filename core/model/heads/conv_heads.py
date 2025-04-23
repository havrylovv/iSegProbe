"""Different convolutional segmentation heads."""

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from .base_head import BaseClassifierHead


class SimpleClassifierHead(BaseClassifierHead):
    """Single 1x1 conv layer."""

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__(in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.classifier(x)
        return output


class SimpleConvSegHead(BaseClassifierHead):
    """Several 1x1 conv layers."""

    def __init__(self, in_channels: int, num_layers: int, num_classes: int) -> None:
        super().__init__(in_channels, num_classes)

        self.num_layers = num_layers
        self.convs = []

        for i in range(num_layers):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels,
                    out_channels=self.in_channels,
                    kernel_size=1,
                    stride=1,
                )
            )
        self.convs = nn.Sequential(*self.convs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.convs(x)
        out = self.classifier(x)
        return out


class ConvSegHead(BaseClassifierHead):
    """Several 3x3 conv layers, followed by a 1x1 conv layer."""

    def __init__(self, in_channels: int, num_layers: int, num_classes: int) -> None:
        super().__init__(in_channels, num_classes)

        self.num_layers = num_layers
        self.convs = []

        for i in range(num_layers):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels,
                    out_channels=self.in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
        self.convs = nn.Sequential(*self.convs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.convs(x)
        out = self.classifier(x)
        return out
