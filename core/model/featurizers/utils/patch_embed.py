"""
Patch Embedding Module.
Source: https://github.com/uncbiag/SimpleClick/blob/71eee559fefdd22076e6a973792ba874a0cee762/isegm/model/modeling/models_vit.py#L78
"""

from typing import Tuple

import torch
from torch import nn


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        img_size: Tuple[int, int] = (224, 224),
        patch_size: Tuple[int, int] = (16, 16),
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: nn.Module = None,
        flatten: bool = True,
    ) -> None:
        super().__init__()
        self.in_chans = in_chans
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x
