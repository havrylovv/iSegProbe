"""Class for iterative generation of synthetic clicks during evalutation phase."""

from copy import deepcopy
from typing import List, Tuple

import cv2
import numpy as np


class Click:
    def __init__(
        self, is_positive: bool, coords: Tuple[int, int], indx: int = None
    ) -> None:
        self.is_positive = is_positive
        self.coords = coords
        self.indx = indx

    @property
    def coords_and_indx(self) -> Tuple[Tuple[int, int], int]:
        return (*self.coords, self.indx)

    def copy(self, **kwargs) -> "Click":
        self_copy = deepcopy(self)
        for k, v in kwargs.items():
            setattr(self_copy, k, v)
        return self_copy


class Clicker(object):
    def __init__(
        self,
        gt_mask: np.ndarray = None,
        init_clicks: List[Click] = None,
        ignore_label: int = -1,
        click_indx_offset: int = 0,
    ) -> None:
        self.click_indx_offset = click_indx_offset
        if gt_mask is not None:
            self.gt_mask = gt_mask == 1
            self.not_ignore_mask = gt_mask != ignore_label
        else:
            self.gt_mask = None

        self.reset_clicks()

        if init_clicks is not None:
            for click in init_clicks:
                self.add_click(click)

    def make_next_click(self, pred_mask: np.ndarray) -> None:
        assert self.gt_mask is not None
        click = self._get_next_click(pred_mask)
        self.add_click(click)

    def get_clicks(self, clicks_limit: int = None) -> List[Click]:
        return self.clicks_list[:clicks_limit]

    def _get_next_click(self, pred_mask: np.ndarray, padding: bool = True) -> Click:
        fn_mask = np.logical_and(
            np.logical_and(self.gt_mask, np.logical_not(pred_mask)),
            self.not_ignore_mask,
        )
        fp_mask = np.logical_and(
            np.logical_and(np.logical_not(self.gt_mask), pred_mask),
            self.not_ignore_mask,
        )

        if padding:
            fn_mask = np.pad(fn_mask, ((1, 1), (1, 1)), "constant")
            fp_mask = np.pad(fp_mask, ((1, 1), (1, 1)), "constant")

        fn_mask_dt = cv2.distanceTransform(fn_mask.astype(np.uint8), cv2.DIST_L2, 0)
        fp_mask_dt = cv2.distanceTransform(fp_mask.astype(np.uint8), cv2.DIST_L2, 0)

        if padding:
            fn_mask_dt = fn_mask_dt[1:-1, 1:-1]
            fp_mask_dt = fp_mask_dt[1:-1, 1:-1]

        fn_mask_dt = fn_mask_dt * self.not_clicked_map
        fp_mask_dt = fp_mask_dt * self.not_clicked_map

        fn_max_dist = np.max(fn_mask_dt)
        fp_max_dist = np.max(fp_mask_dt)

        is_positive = fn_max_dist > fp_max_dist
        if is_positive:
            coords_y, coords_x = np.where(fn_mask_dt == fn_max_dist)  # coords is [y, x]
        else:
            coords_y, coords_x = np.where(fp_mask_dt == fp_max_dist)  # coords is [y, x]

        return Click(is_positive=is_positive, coords=(coords_y[0], coords_x[0]))

    def add_click(self, click: Click) -> None:
        coords = click.coords

        click.indx = self.click_indx_offset + self.num_pos_clicks + self.num_neg_clicks
        if click.is_positive:
            self.num_pos_clicks += 1
        else:
            self.num_neg_clicks += 1

        self.clicks_list.append(click)
        if self.gt_mask is not None:
            self.not_clicked_map[coords[0], coords[1]] = False

    def _remove_last_click(self) -> None:
        click = self.clicks_list.pop()
        coords = click.coords

        if click.is_positive:
            self.num_pos_clicks -= 1
        else:
            self.num_neg_clicks -= 1

        if self.gt_mask is not None:
            self.not_clicked_map[coords[0], coords[1]] = True

    def reset_clicks(self) -> None:
        if self.gt_mask is not None:
            self.not_clicked_map = np.ones_like(self.gt_mask, dtype=bool)

        self.num_pos_clicks = 0
        self.num_neg_clicks = 0

        self.clicks_list = []

    def get_state(self) -> List[Click]:
        return deepcopy(self.clicks_list)

    def set_state(self, state: List[Click]) -> None:
        self.reset_clicks()
        for click in state:
            self.add_click(click)

    def __len__(self) -> int:
        return len(self.clicks_list)
