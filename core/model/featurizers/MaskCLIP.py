"""MaskCLIP Featurizer, adapter for injection of additional features."""

import os

import torch
from torch import nn

from core.utils.log import logger

from .maskclip import clip


class MaskCLIPFeaturizer(nn.Module):
    """MaskCLIP Featurizer. `feats_injection_mode` indicates how optional click features are injected.
    Args:
        model_name (str, optional): Name of the CLIP model to load.
        feats_injection_mode (str, optional): Mode for injecting additional features.
            Options:
                - "no_injeciton",
                - "before_backbone": after patch embedding and before main backbone part,
                - "after_backbone": after main backbone part.
    """

    def __init__(
        self, model_name: str = "ViT-B/16", feats_injection_mode: str = "no_injection"
    ) -> None:

        super().__init__()
        self.feats_injection_mode = feats_injection_mode
        self.model, self.preprocess = clip.load(
            model_name,
            download_root=os.getenv(
                "TORCH_HOME", os.path.join(os.path.expanduser("~"), ".cache", "torch")
            ),
        )
        logger.info(f"Loaded checkpoint for MaskCLIP: {model_name}")

        self.model.eval()
        self.patch_size = self.model.visual.patch_size

    def forward(
        self, x: torch.Tensor, additional_features: torch.Tensor = None
    ) -> torch.Tensor:
        x = x.to(self.model.dtype)  # might require to convert to HalfTensor
        b, _, input_size_h, input_size_w = x.shape

        patch_h = input_size_h // self.patch_size
        patch_w = input_size_w // self.patch_size

        if (
            additional_features is not None
            and self.feats_injection_mode == "before_backbone"
        ):
            # patch embedding
            x = self.model.visual.conv1(x)  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

            # feats injection
            assert (
                x.shape == additional_features.shape
            ), f"x.shape: {x.shape}, additional_features.shape: {additional_features.shape}"
            x += additional_features

            # forward pass
            return self.forward_without_path_embed(x, (input_size_h, input_size_w))

        features = self.model.get_patch_encodings(x).to(torch.float32)

        if (
            additional_features is not None
            and self.feats_injection_mode == "after_backbone"
        ):
            assert (
                features.shape == additional_features.shape
            ), f"features.shape: {features.shape}, additional_features.shape: {additional_features.shape}"
            features += additional_features

        return features.reshape(b, patch_h, patch_w, -1).permute(0, 3, 1, 2)

    def forward_without_path_embed(self, x, orig_image_hw: tuple = (224, 224)):
        """Forward pass through the model without path embedding."""
        input_size_h, input_size_w = orig_image_hw
        b = x.shape[0]

        patch_h = input_size_h // self.patch_size
        patch_w = input_size_w // self.patch_size
        features = self.model.encode_projected_patches(x, orig_image_hw).to(
            torch.float32
        )

        return features.reshape(b, patch_h, patch_w, -1).permute(0, 3, 1, 2)
