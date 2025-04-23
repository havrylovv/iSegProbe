"""
PCA decomposition of features. Used for visualization.
Adapted from: https://github.com/mhamilton723/FeatUp
"""

from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA


class TorchPCA(object):
    """PCA implementation using PyTorch."""

    def __init__(self, n_components: int) -> None:
        self.n_components = n_components

    def fit(self, X: torch.Tensor) -> "TorchPCA":
        self.mean_ = X.mean(dim=0)
        unbiased = X - self.mean_.unsqueeze(0)
        U, S, V = torch.pca_lowrank(
            unbiased, q=self.n_components, center=False, niter=4
        )
        self.components_ = V.T
        self.singular_values_ = S
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        t0 = X - self.mean_.unsqueeze(0)
        projected = t0 @ self.components_.T
        return projected


def pca(
    image_feats_list: List[torch.Tensor],
    dim: int = 3,
    fit_pca: Union[TorchPCA, PCA] = None,
    use_torch_pca: bool = True,
    max_samples: int = None,
) -> Tuple[List[torch.Tensor], Union[TorchPCA, PCA]]:
    """Apply PCA to a list of image features."""
    device = image_feats_list[0].device

    def flatten(tensor, target_size=None):
        if target_size is not None and fit_pca is None:
            tensor = F.interpolate(tensor, (target_size, target_size), mode="bilinear")
        B, C, H, W = tensor.shape
        return (
            tensor.permute(1, 0, 2, 3)
            .reshape(C, B * H * W)
            .permute(1, 0)
            .detach()
            .cpu()
        )

    if len(image_feats_list) > 1 and fit_pca is None:
        target_size = image_feats_list[0].shape[2]
    else:
        target_size = None

    flattened_feats = []
    for feats in image_feats_list:
        flattened_feats.append(flatten(feats, target_size))
    x = torch.cat(flattened_feats, dim=0)

    # Subsample the data if max_samples is set and the number of samples exceeds max_samples
    if max_samples is not None and x.shape[0] > max_samples:
        indices = torch.randperm(x.shape[0])[:max_samples]
        x = x[indices]

    if fit_pca is None:
        if use_torch_pca:
            fit_pca = TorchPCA(n_components=dim).fit(x)
        else:
            fit_pca = PCA(n_components=dim).fit(x)

    reduced_feats = []
    for feats in image_feats_list:
        x_red = fit_pca.transform(flatten(feats))
        if isinstance(x_red, np.ndarray):
            x_red = torch.from_numpy(x_red)
        x_red -= x_red.min(dim=0, keepdim=True).values
        x_red /= x_red.max(dim=0, keepdim=True).values
        B, C, H, W = feats.shape
        reduced_feats.append(x_red.reshape(B, H, W, dim).permute(0, 3, 1, 2).to(device))

    return reduced_feats, fit_pca
