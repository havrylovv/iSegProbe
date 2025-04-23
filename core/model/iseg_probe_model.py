"""Interactive segmentaiton model for probing VFMs and upsamplers."""

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from core.model.iseg_base_model import iSegBaseModel
from core.utils.log import logger
from core.utils.serialization import serialize

from ..utils.model_builder import ModelBuilder
from .featurizers.utils import PatchEmbed


class iSegProbeModel(iSegBaseModel):
    """
    Model for probing VFMs and upsamlers. It has two architectures:
    1. `backbone_upsampler_head`: backbone -> upsampler -> head
    2. `backbone_neck_head`: backbone -> neck -> head
    All model components are loaded through `ModelBuilder` class.

    Args:
        backbone_cfg (Dict): Backbone configuration.
        head_cfg (Dict): Head configuration.
        embed_coords_cfg (Dict): Configuration for embedding coordinates module.
        neck_cfg (Dict): Neck configuration.
        upsampler_cfg (Dict): Upsampler configuration.
        save_cfg (Dict): Configuration for saving model components.
        architecture (str): Architecture type. Options are 'backbone_upsampler_head' or 'backbone_neck_head',
        model_builder (ModelBuilder): Model builder instance.
    """

    @serialize
    def __init__(
        self,
        backbone_cfg: Dict = None,
        head_cfg: Dict = None,
        embed_coords_cfg: Dict = None,
        neck_cfg: Dict = None,
        upsampler_cfg: Dict = None,
        save_cfg: Dict = None,
        architecture: str = "backbone_upsampler_head",
        model_builder: ModelBuilder = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.save_cfg = save_cfg
        self.architecture = architecture
        self.embed_coords_type = embed_coords_cfg["type"]
        self.upsampler_type = upsampler_cfg["type"] if upsampler_cfg else None
        self.model_builder = model_builder

        assert (
            backbone_cfg is not None
            and head_cfg is not None
            and embed_coords_cfg is not None
        ), "backbone, head and embed_coords configurations must be provided"

        assert self.architecture in [
            "backbone_upsampler_head",
            "backbone_neck_head",
        ], f"Unknown architecture: {self.architecture}"

        # load frozen modules: backbone, upsampler
        self.backbone = self.model_builder.load_featurizer(
            backbone_cfg["type"], backbone_cfg["params"], freeze=True
        )
        self.upsampler = (
            self.model_builder.load_upsampler(
                upsampler_cfg["type"], upsampler_cfg["params"], freeze=True
            )
            if upsampler_cfg
            else self.model_builder.load_upsampler("bilinear", None, freeze=True)
        )

        # load trainable modules: neck, head, embed_coords
        self.neck = (
            self.model_builder.load_neck(
                neck_cfg["type"],
                neck_cfg["params"],
                upsampler=self.upsampler,
                freeze=False,
            )
            if neck_cfg
            else nn.Identity()
        )

        self.head = self.model_builder.load_head(
            head_cfg["type"], head_cfg["params"], freeze=False
        )

        if self.embed_coords_type == "patchEmbed":
            self.embed_coords = PatchEmbed(
                img_size=embed_coords_cfg["params"]["img_size"],
                patch_size=embed_coords_cfg["params"]["patch_size"],
                in_chans=3 if self.with_prev_mask else 2,
                embed_dim=embed_coords_cfg["params"]["embed_dim"],
            )
        elif self.embed_coords_type == "simple_vit":
            self.embed_coords = self.model_builder.load_featurizer(
                "simple_vit", embed_coords_cfg["params"], freeze=False
            )
        else:
            raise ValueError(f"Unknown embed_coords_type: {self.embed_coords_type}")

        # count number of parameters
        self._count_parameters()

    def backbone_forward(
        self, image: torch.Tensor, coord_features: torch.Tensor = None
    ) -> Dict:
        coord_features = self.embed_coords(coord_features)

        backbone_features = self.backbone(image, coord_features)

        if self.architecture == "backbone_upsampler_head":
            backbone_features = self.upsampler(source=backbone_features, guidance=image)

            if (
                self.upsampler_type != "identity"
                and image.size()[2:] != backbone_features.size()[2:]
            ):
                backbone_features = nn.functional.interpolate(
                    backbone_features,
                    size=image.size()[2:],
                    mode="bilinear",
                    align_corners=True,
                )
        elif self.architecture == "backbone_neck_head":
            backbone_features = self.neck(backbone_features, guidance=image)

        output = self.head(backbone_features)
        return {"instances": output, "instances_aux": None}

    def get_lowres_highres_feats(
        self, image: torch.Tensor, points: torch.Tensor
    ) -> Tuple:
        """Extracts low and high resolution features from the image.
        Used in `get_save_feats_callback` during evaluation.

        Args:
            image (torch.Tensor): 3 or 4 channel image tensor (if 4, last channel is previous mask)
            points (torch.Tensor)

        Returns:
            tuple: A tuple containing the image and features dictionary. Image dict contains potentially useful data for visualization.
            The features dictionary contains the extracted features.
            Combination of Low and High resolution features are then used for PCA reduction and visualization.
            We compute PCA jointly for corresponding `LowRes` and `HighRes`features. Thus, it is necessary to return pairs of features.
        """

        image, prev_mask = self.prepare_input(image)
        coord_features = self.get_coord_features(image, prev_mask, points)
        coord_features = self.maps_transform(coord_features)
        lr_coord_features_embedded = self.embed_coords(coord_features)

        lr_feats = self.backbone(image, lr_coord_features_embedded)
        hr_feats = self.upsampler(source=lr_feats, guidance=image)
        feats = {"LowRes": lr_feats, "HighRes": hr_feats}

        if self.upsampler.__class__.__name__ in ["IdentityUpsampler", "LiFTUpsampler"]:
            # iterate though the features and upsample them
            for key, value in feats.items():
                if "HighRes" in key:
                    feats[key] = nn.functional.interpolate(
                        value,
                        size=image.size()[2:],
                        mode="bilinear",
                        align_corners=True,
                    )
        image_dict = {"coord_features": coord_features}

        return image_dict, feats

    def _count_parameters(self):
        params_count = {
            "backbone (M)": round(
                sum(p.numel() for p in self.backbone.parameters()) / 1e6, 2
            ),
            "head (M)": round(sum(p.numel() for p in self.head.parameters()) / 1e6, 2),
            "embed_coords (k)": round(
                sum(p.numel() for p in self.embed_coords.parameters()) / 1e3, 2
            ),
            "neck (M)": round(sum(p.numel() for p in self.neck.parameters()) / 1e6, 2),
            "trainable (M)": round(
                sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6, 2
            ),
            "total (M)": round(sum(p.numel() for p in self.parameters()) / 1e6, 2),
        }

        if isinstance(self.upsampler, nn.Module):
            params_count["upsampler (k)"] = round(
                sum(p.numel() for p in self.upsampler.parameters()) / 1e3, 2
            )

        logger.info(f"PARAMETERS COUNT: {params_count}")

    def get_state_dict_to_save(self):
        """
        Returns a filtered state dictionary based on the configuration provided in `self.save_cfg`.
        Defines which parts of the model should be saved in checkpoints.
        This avoids reppetitive saving heavy frozen modules.

        - Keys of `self.save_cfg` are model blocks.
        - Values can be:
            - `True` (include all submodules),
            - `False` (exclude entirely),
            - A dictionary with keys 'save' (boolean) and 'exclude' (list of submodules to ignore).

        Example:
        ```
            save_cfg = {
                'backbone': False,
                'neck': {
                    'save': True,
                    'exclude': ['mlp']
                },
                'head': True,
            }
        ```
        """
        state_dict = self.state_dict()
        if not self.save_cfg:
            return state_dict  # No filtering needed if save_cfg is not defined

        def is_included(param_name):
            """
            Checks if a parameter should be included based on `save_cfg` rules.
            """
            parts = param_name.split(".")  # Example: ['neck', 'upsampler', 'weight']
            cfg = self.save_cfg  # Start from the root config

            for i, part in enumerate(parts):
                if isinstance(cfg, dict):
                    # Check if the current module is explicitly excluded
                    if "exclude" in cfg and part in cfg["exclude"]:
                        return False

                    # Move deeper into the structure if possible
                    cfg = cfg.get(part, None)

                    # If cfg is False, it means this entire part is excluded
                    if cfg is False:
                        return False

                    # If cfg is None, it means the module is not explicitly defined, assume saving
                    if cfg is None:
                        return True

                    # If cfg is a dictionary and has "save": False, exclude this block
                    if isinstance(cfg, dict) and not cfg.get("save", False):
                        return False

            return True  # If we pass all checks, the parameter should be saved

        # Filter state_dict based on is_included function
        return {k: v for k, v in state_dict.items() if is_included(k)}
