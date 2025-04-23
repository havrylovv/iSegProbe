"""Custom augmentations for image processing. Used during training phase."""

import random
from typing import Dict, List, Tuple

import cv2
import numpy as np
from albumentations import DualTransform, ImageOnlyTransform
from albumentations.augmentations import functional as F
from albumentations.core.serialization import SERIALIZABLE_REGISTRY
from albumentations.core.transforms_interface import to_tuple

from core.utils.misc import (
    clamp_bbox,
    expand_bbox,
    get_bbox_from_mask,
    get_labels_with_sizes,
)


class UniformRandomResize(DualTransform):
    def __init__(
        self,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        interpolation: int = cv2.INTER_LINEAR,
        always_apply: bool = False,
        p: float = 1.0,
    ) -> None:
        super().__init__(always_apply, p)
        self.scale_range = scale_range
        self.interpolation = interpolation

    def get_params_dependent_on_targets(self, params: Dict) -> Dict:
        scale = random.uniform(*self.scale_range)
        height = int(round(params["image"].shape[0] * scale))
        width = int(round(params["image"].shape[1] * scale))
        return {"new_height": height, "new_width": width}

    def apply(
        self,
        img: np.ndarray,
        new_height: int = 0,
        new_width: int = 0,
        interpolation: int = cv2.INTER_LINEAR,
        **params
    ) -> np.ndarray:
        return F.resize(
            img, height=new_height, width=new_width, interpolation=interpolation
        )

    def apply_to_keypoint(
        self, keypoint: Tuple, new_height: int = 0, new_width: int = 0, **params
    ) -> Tuple:
        scale_x = new_width / params["cols"]
        scale_y = new_height / params["rows"]
        return F.keypoint_scale(keypoint, scale_x, scale_y)

    def get_transform_init_args_names(self) -> Tuple[str, str]:
        return "scale_range", "interpolation"

    @property
    def targets_as_params(self) -> List[str]:
        return ["image"]


class ZoomIn(DualTransform):
    def __init__(
        self,
        height: int,
        width: int,
        bbox_jitter: int = 0.1,
        expansion_ratio: int = 1.4,
        min_crop_size: int = 200,
        min_area: int = 100,
        always_resize: bool = False,
        always_apply: bool = False,
        p: float = 0.5,
    ) -> None:
        super(ZoomIn, self).__init__(always_apply, p)
        self.height = height
        self.width = width
        self.bbox_jitter = to_tuple(bbox_jitter)
        self.expansion_ratio = expansion_ratio
        self.min_crop_size = min_crop_size
        self.min_area = min_area
        self.always_resize = always_resize

    def apply(
        self, img: np.ndarray, selected_object, bbox: Tuple, **params
    ) -> np.ndarray:
        if selected_object is None:
            if self.always_resize:
                img = F.resize(img, height=self.height, width=self.width)
            return img

        rmin, rmax, cmin, cmax = bbox
        img = img[rmin : rmax + 1, cmin : cmax + 1]
        img = F.resize(img, height=self.height, width=self.width)

        return img

    def apply_to_mask(
        self, mask: np.ndarray, selected_object, bbox: Tuple, **params
    ) -> np.ndarray:
        if selected_object is None:
            if self.always_resize:
                mask = F.resize(
                    mask,
                    height=self.height,
                    width=self.width,
                    interpolation=cv2.INTER_NEAREST,
                )
            return mask

        rmin, rmax, cmin, cmax = bbox
        mask = mask[rmin : rmax + 1, cmin : cmax + 1]
        if isinstance(selected_object, tuple):
            layer_indx, mask_id = selected_object
            obj_mask = mask[:, :, layer_indx] == mask_id
            new_mask = np.zeros_like(mask)
            new_mask[:, :, layer_indx][obj_mask] = mask_id
        else:
            obj_mask = mask == selected_object
            new_mask = mask.copy()
            new_mask[np.logical_not(obj_mask)] = 0

        new_mask = F.resize(
            new_mask,
            height=self.height,
            width=self.width,
            interpolation=cv2.INTER_NEAREST,
        )
        return new_mask

    def get_params_dependent_on_targets(self, params: Dict) -> Dict:
        instances = params["mask"]

        is_mask_layer = len(instances.shape) > 2
        candidates = []
        if is_mask_layer:
            for layer_indx in range(instances.shape[2]):
                labels, areas = get_labels_with_sizes(instances[:, :, layer_indx])
                candidates.extend(
                    [
                        (layer_indx, obj_id)
                        for obj_id, area in zip(labels, areas)
                        if area > self.min_area
                    ]
                )
        else:
            labels, areas = get_labels_with_sizes(instances)
            candidates = [
                obj_id for obj_id, area in zip(labels, areas) if area > self.min_area
            ]

        selected_object = None
        bbox = None
        if candidates:
            selected_object = random.choice(candidates)
            if is_mask_layer:
                layer_indx, mask_id = selected_object
                obj_mask = instances[:, :, layer_indx] == mask_id
            else:
                obj_mask = instances == selected_object

            bbox = get_bbox_from_mask(obj_mask)

            if isinstance(self.expansion_ratio, tuple):
                expansion_ratio = random.uniform(*self.expansion_ratio)
            else:
                expansion_ratio = self.expansion_ratio

            bbox = expand_bbox(bbox, expansion_ratio, self.min_crop_size)
            bbox = self._jitter_bbox(bbox)
            bbox = clamp_bbox(bbox, 0, obj_mask.shape[0] - 1, 0, obj_mask.shape[1] - 1)

        return {"selected_object": selected_object, "bbox": bbox}

    def _jitter_bbox(self, bbox: Tuple) -> Tuple:
        rmin, rmax, cmin, cmax = bbox
        height = rmax - rmin + 1
        width = cmax - cmin + 1
        rmin = int(rmin + random.uniform(*self.bbox_jitter) * height)
        rmax = int(rmax + random.uniform(*self.bbox_jitter) * height)
        cmin = int(cmin + random.uniform(*self.bbox_jitter) * width)
        cmax = int(cmax + random.uniform(*self.bbox_jitter) * width)

        return rmin, rmax, cmin, cmax

    def apply_to_bbox(self, bbox, **params):
        raise NotImplementedError

    def apply_to_keypoint(self, keypoint, **params):
        raise NotImplementedError

    @property
    def targets_as_params(self) -> List[str]:
        return ["mask"]

    def get_transform_init_args_names(self) -> Tuple[str]:
        return (
            "height",
            "width",
            "bbox_jitter",
            "expansion_ratio",
            "min_crop_size",
            "min_area",
            "always_resize",
        )


def remove_image_only_transforms(sdict: Dict) -> Dict:
    if not "transforms" in sdict:
        return sdict

    keep_transforms = []
    for tdict in sdict["transforms"]:
        cls = SERIALIZABLE_REGISTRY[tdict["__class_fullname__"]]
        if "transforms" in tdict:
            keep_transforms.append(remove_image_only_transforms(tdict))
        elif not issubclass(cls, ImageOnlyTransform):
            keep_transforms.append(tdict)
    sdict["transforms"] = keep_transforms

    return sdict
