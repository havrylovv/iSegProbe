"""Transform for zooming in on a region of interest in an image based on user clicks."""

from typing import List, Tuple

import torch

from core.inference.clicker import Click
from core.utils.misc import clamp_bbox, expand_bbox, get_bbox_from_mask, get_bbox_iou

from .base_transform import BaseTransform


class ZoomIn(BaseTransform):
    def __init__(
        self,
        target_size: int = 400,
        skip_clicks: int = 1,
        expansion_ratio: float = 1.4,
        min_crop_size: int = 200,
        recompute_thresh_iou: float = 0.5,
        prob_thresh: float = 0.50,
    ) -> None:
        super().__init__()
        self.target_size = target_size
        self.min_crop_size = min_crop_size
        self.skip_clicks = skip_clicks
        self.expansion_ratio = expansion_ratio
        self.recompute_thresh_iou = recompute_thresh_iou
        self.prob_thresh = prob_thresh

        self._input_image_shape = None
        self._prev_probs = None  # previous prediction probabilities
        self._object_roi = None  # (rmin, rmax, cmin, cmax) - min/max row/col indices
        self._roi_image = None  # actual image crop

    def transform(
        self, image_nd: torch.Tensor, clicks_lists: List[List[Click]]
    ) -> Tuple[torch.Tensor, List[List[Click]]]:
        """image_nd is a list of images. During inference, model might be called on image and flipped image,
        with further aggregation of results."""
        transformed_image = []
        transformed_clicks_lists = []
        for bindx in range(len(clicks_lists)):
            new_image_nd, new_clicks_lists = self._transform(
                image_nd[bindx].unsqueeze(0), [clicks_lists[bindx]]
            )
            transformed_image.append(new_image_nd)
            transformed_clicks_lists.append(new_clicks_lists[0])
        return torch.cat(transformed_image, dim=0), transformed_clicks_lists

    def _transform(
        self, image_nd: torch.Tensor, clicks_lists: List[List[Click]]
    ) -> Tuple[torch.Tensor, List[List[Click]]]:

        assert image_nd.shape[0] == 1 and len(clicks_lists) == 1
        self.image_changed = False

        clicks_list = clicks_lists[0]
        if len(clicks_list) <= self.skip_clicks:
            return image_nd, clicks_lists

        self._input_image_shape = image_nd.shape

        current_object_roi = None
        if self._prev_probs is not None:
            current_pred_mask = (self._prev_probs > self.prob_thresh)[0, 0]
            if current_pred_mask.sum() > 0:
                current_object_roi = get_object_roi(
                    current_pred_mask,
                    clicks_list,
                    self.expansion_ratio,
                    self.min_crop_size,
                )

        if current_object_roi is None:
            if self.skip_clicks >= 0:
                return image_nd, clicks_lists
            else:
                current_object_roi = 0, image_nd.shape[2] - 1, 0, image_nd.shape[3] - 1

        update_object_roi = False
        if self._object_roi is None:
            update_object_roi = True
        elif not check_object_roi(self._object_roi, clicks_list):
            update_object_roi = True
        elif (
            get_bbox_iou(current_object_roi, self._object_roi)
            < self.recompute_thresh_iou
        ):
            update_object_roi = True

        if update_object_roi:
            self._object_roi = current_object_roi
            self.image_changed = True
        self._roi_image = get_roi_image_nd(image_nd, self._object_roi, self.target_size)
        tclicks_lists = [self._transform_clicks(clicks_list)]
        return self._roi_image.to(image_nd.device), tclicks_lists

    def inv_transform(self, prob_map: torch.Tensor) -> torch.Tensor:
        new_prob_maps = []
        for bindx in range(prob_map.shape[0]):
            new_prob_map = self._inv_transform(prob_map[bindx].unsqueeze(0))
            new_prob_maps.append(new_prob_map)
        return torch.cat(new_prob_maps, dim=0)

    def _inv_transform(self, prob_map: torch.Tensor) -> torch.Tensor:
        if self._object_roi is None:
            self._prev_probs = prob_map.cpu().numpy()
            return prob_map

        assert prob_map.shape[0] == 1
        rmin, rmax, cmin, cmax = self._object_roi
        prob_map = torch.nn.functional.interpolate(
            prob_map,
            size=(rmax - rmin + 1, cmax - cmin + 1),
            mode="bilinear",
            align_corners=True,
        )

        if self._prev_probs is not None:
            new_prob_map = torch.zeros(
                *self._prev_probs.shape, device=prob_map.device, dtype=prob_map.dtype
            )
            new_prob_map[:, :, rmin : rmax + 1, cmin : cmax + 1] = prob_map
        else:
            new_prob_map = prob_map

        self._prev_probs = new_prob_map.cpu().numpy()

        return new_prob_map

    def check_possible_recalculation(self) -> bool:
        if (
            self._prev_probs is None
            or self._object_roi is not None
            or self.skip_clicks > 0
        ):
            return False

        pred_mask = (self._prev_probs > self.prob_thresh)[0, 0]
        if pred_mask.sum() > 0:
            possible_object_roi = get_object_roi(
                pred_mask, [], self.expansion_ratio, self.min_crop_size
            )
            image_roi = (
                0,
                self._input_image_shape[2] - 1,
                0,
                self._input_image_shape[3] - 1,
            )
            if get_bbox_iou(possible_object_roi, image_roi) < 0.50:
                return True
        return False

    def get_state(self) -> Tuple:
        roi_image = self._roi_image.cpu() if self._roi_image is not None else None
        return (
            self._input_image_shape,
            self._object_roi,
            self._prev_probs,
            roi_image,
            self.image_changed,
        )

    def set_state(self, state: Tuple) -> None:
        (
            self._input_image_shape,
            self._object_roi,
            self._prev_probs,
            self._roi_image,
            self.image_changed,
        ) = state

    def reset(self) -> None:
        self._input_image_shape = None
        self._object_roi = None
        self._prev_probs = None
        self._roi_image = None
        self.image_changed = False

    def _transform_clicks(self, clicks_list: List[Click]) -> List[Click]:
        if self._object_roi is None:
            return clicks_list

        rmin, rmax, cmin, cmax = self._object_roi
        crop_height, crop_width = self._roi_image.shape[2:]

        transformed_clicks = []
        for click in clicks_list:
            new_r = crop_height * (click.coords[0] - rmin) / (rmax - rmin + 1)
            new_c = crop_width * (click.coords[1] - cmin) / (cmax - cmin + 1)
            transformed_clicks.append(click.copy(coords=(new_r, new_c)))
        return transformed_clicks


def get_object_roi(
    pred_mask: torch.Tensor,
    clicks_list: List[Click],
    expansion_ratio: float,
    min_crop_size: float,
) -> Tuple[int, int, int, int]:
    pred_mask = pred_mask.copy()

    for click in clicks_list:
        if click.is_positive:
            pred_mask[int(click.coords[0]), int(click.coords[1])] = 1

    bbox = get_bbox_from_mask(pred_mask)
    bbox = expand_bbox(bbox, expansion_ratio, min_crop_size)
    h, w = pred_mask.shape[0], pred_mask.shape[1]
    bbox = clamp_bbox(bbox, 0, h - 1, 0, w - 1)

    return bbox


def get_roi_image_nd(
    image_nd: torch.Tensor, object_roi: Tuple[int, int, int, int], target_size: int
) -> torch.Tensor:
    rmin, rmax, cmin, cmax = object_roi

    height = rmax - rmin + 1
    width = cmax - cmin + 1

    if isinstance(target_size, tuple):
        new_height, new_width = target_size
    else:
        scale = target_size / max(height, width)
        new_height = int(round(height * scale))
        new_width = int(round(width * scale))

    with torch.no_grad():
        roi_image_nd = image_nd[:, :, rmin : rmax + 1, cmin : cmax + 1]
        roi_image_nd = torch.nn.functional.interpolate(
            roi_image_nd,
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=True,
        )

    return roi_image_nd


def check_object_roi(
    object_roi: Tuple[int, int, int, int], clicks_list: List[Click]
) -> bool:
    for click in clicks_list:
        if click.is_positive:
            if click.coords[0] < object_roi[0] or click.coords[0] >= object_roi[1]:
                return False
            if click.coords[1] < object_roi[2] or click.coords[1] >= object_roi[3]:
                return False

    return True
