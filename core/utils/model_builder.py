"""Class to load model components."""

from typing import Dict

import torch.nn as nn

from core.model.featurizers import *
from core.model.heads import HEAD_REGISTRY, BaseClassifierHead
from core.model.upsamplers import UPSAMPLER_REGISTRY, BaseUpsampler
from core.utils.log import logger


class ModelBuilder:
    """Class to load different components of interactive segmentaiton model."""

    def __init__(self) -> None:
        pass

    def load_featurizer(
        self,
        type: str,
        params: Dict,
        freeze: bool = True,
    ) -> nn.Module:
        """Loads the featurizer based on the type and parameters provided."""
        type = type.lower()

        if type == "mask_clip":
            if hasattr(params, "model_name"):
                if params["model_name"] != "ViT-B/16":
                    raise ValueError(
                        f"Currently unsupported model_name for MaskCLIP: {params['model_name']}"
                    )
            backbone = MaskCLIPFeaturizer(**params)
        elif type == "dinov2":
            backbone = DINOv2Featurizer(**params)
        elif type == "vit":
            backbone = DINOFeaturizer(**params)
        elif type == "simple_vit":
            backbone = SimpleViTFeaturizer(
                image_size=params["img_size"],
                patch_size=params["patch_size"],
                dim=params["embed_dim"],
                depth=params["depth"],
                heads=params["heads"],
                mlp_dim=params["mlp_dim"],
                channels=params["channels"],
                dim_head=params["dim_head"],
            )
        else:
            raise ValueError(f"Unsupported backbone type: {type}")

        if freeze:
            for param in backbone.parameters():
                param.requires_grad = False

        return backbone

    def load_upsampler(
        self, type: str, params: Dict = None, freeze: bool = True
    ) -> BaseUpsampler:
        """Loads the upsampler based on the type and parameters provided."""
        type = type.lower()
        if type not in UPSAMPLER_REGISTRY:
            raise ValueError(f"Unsupported upsampler type: {type}")

        # Retrieve the upsampler class from the registry
        upsampler_cls = UPSAMPLER_REGISTRY[type]
        upsampler = upsampler_cls(**params) if params else upsampler_cls()

        if freeze:
            for param in upsampler.parameters():
                param.requires_grad = False

        logger.info(f"UPSAMPLER: Loaded {upsampler.__class__.__name__}")
        return upsampler

    def load_head(
        self, type: str, params: Dict, freeze: Dict = False
    ) -> BaseClassifierHead:
        if type not in HEAD_REGISTRY:
            raise ValueError(f"Unsupported head type: {type}")

        # Retrieve the head class from the registry
        head_cls = HEAD_REGISTRY[type]

        # Instantiate the head class with the parameters
        head = head_cls(**params)
        logger.info(f"HEAD: Loaded {head.__class__.__name__}")

        if freeze:
            for param in head.parameters():
                param.requires_grad = False

        return head

    def load_neck(self, type: str, params: Dict, freeze=False) -> nn.Module:
        raise NotImplementedError(
            "Neck loading is not implemented yet. Please implement the neck loading logic."
        )
