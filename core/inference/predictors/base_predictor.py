"""Base class for interactive segmentation predictors."""

from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from core.inference.clicker import Click, Clicker
from core.inference.transforms import (
    AddHorizontalFlip,
    BaseTransform,
    LimitLongestSide,
    SigmoidForPred,
)
from core.model.iseg_base_model import iSegBaseModel


class BasePredictor(object):
    def __init__(
        self,
        model: iSegBaseModel,
        device: torch.device,
        net_clicks_limit: int = None,
        with_flip: bool = False,
        zoom_in: BaseTransform = None,
        max_size: int = None,
        **kwargs
    ) -> None:
        self.with_flip = with_flip
        self.net_clicks_limit = net_clicks_limit
        self.original_image = None
        self.device = device
        self.zoom_in = zoom_in
        self.prev_prediction = None
        self.model_indx = 0
        self.click_models = None
        self.net_state_dict = None

        if isinstance(model, tuple):
            self.net, self.click_models = model
        else:
            self.net = model

        self.to_tensor = transforms.ToTensor()

        self.transforms = [zoom_in] if zoom_in is not None else []
        if max_size is not None:
            self.transforms.append(LimitLongestSide(max_size=max_size))
        self.transforms.append(SigmoidForPred())
        if with_flip:
            self.transforms.append(AddHorizontalFlip())

    def set_input_image(self, image: Union[torch.Tensor, np.ndarray], **kwargs) -> None:
        if not isinstance(image, torch.Tensor):
            image_nd = self.to_tensor(image)
        else:
            image_nd = image
        for transform in self.transforms:
            transform.reset()
        self.original_image = image_nd.to(self.device)
        if len(self.original_image.shape) == 3:
            self.original_image = self.original_image.unsqueeze(0)
        self.prev_prediction = torch.zeros_like(self.original_image[:, :1, :, :])

    def get_prediction(
        self, clicker: Clicker, prev_mask: torch.Tensor = None
    ) -> np.ndarray:
        clicks_list = clicker.get_clicks()

        if self.click_models is not None:
            model_indx = (
                min(
                    clicker.click_indx_offset + len(clicks_list), len(self.click_models)
                )
                - 1
            )
            if model_indx != self.model_indx:
                self.model_indx = model_indx
                self.net = self.click_models[model_indx]

        input_image = self.original_image
        if prev_mask is None:
            prev_mask = self.prev_prediction
        if hasattr(self.net, "with_prev_mask") and self.net.with_prev_mask:
            input_image = torch.cat((input_image, prev_mask), dim=1)
        image_nd, clicks_lists, is_image_changed = self.apply_transforms(
            input_image, [clicks_list]
        )
        pred_logits = self._get_prediction(
            image_nd, clicks_lists, is_image_changed
        )  # [B,1,H,W], already interpolated to training size, e.g. 448

        prediction = F.interpolate(
            pred_logits, mode="bilinear", align_corners=True, size=image_nd.size()[2:]
        )

        for t in reversed(self.transforms):
            prediction = t.inv_transform(
                prediction
            )  # [B,1,OrigH, OrigW] - reverse zoomin is usually applied here

        if self.zoom_in is not None and self.zoom_in.check_possible_recalculation():
            return self.get_prediction(clicker)

        self.prev_prediction = prediction
        return prediction.cpu().numpy()[0, 0]

    def get_lowres_highres_feats(
        self, clicker: Clicker, prev_mask: torch.Tensor = None
    ) -> Tuple[Dict, Dict]:
        """Called in evaluation phase if feats_callback is specified in evaluate_sample().
        Returns two dictionaries: (1) with auxiliary image info and (2) with low and high resolution (before and after upsampling) features.
        """
        if not hasattr(self.net, "get_lowres_highres_feats"):
            raise ValueError(
                "Model does not support lowres-highres features extraction. It should have get_lowres_highres_feats() function. "
            )

        clicks_list = clicker.get_clicks()

        if self.click_models is not None:
            model_indx = (
                min(
                    clicker.click_indx_offset + len(clicks_list), len(self.click_models)
                )
                - 1
            )
            if model_indx != self.model_indx:
                self.model_indx = model_indx
                self.net = self.click_models[model_indx]

        input_image = self.original_image.clone()

        if prev_mask is None:
            prev_mask = self.prev_prediction
        if hasattr(self.net, "with_prev_mask") and self.net.with_prev_mask:
            input_image = torch.cat((input_image, prev_mask), dim=1)
        image_nd, clicks_lists, is_image_changed = self.apply_transforms(
            input_image, [clicks_list]
        )

        points_nd = self.get_points_nd(clicks_lists)
        img_dict, feats = self.net.get_lowres_highres_feats(image_nd, points_nd)

        return img_dict, feats

    def _get_prediction(self, image_nd, clicks_lists, is_image_changed):
        points_nd = self.get_points_nd(clicks_lists)
        return self.net(image_nd, points_nd)["instances"]

    def _batch_infer(self, batch_image_tensor, batch_clickers, prev_mask=None):
        if prev_mask is None:
            prev_mask = self.prev_prediction

        if hasattr(self.net, "with_prev_mask") and self.net.with_prev_mask:
            input_image = torch.cat((batch_image_tensor, prev_mask), dim=1)

        clicks_lists = [clicker.get_clicks() for clicker in batch_clickers]
        image_nd, clicks_lists, is_image_changed = self.apply_transforms(
            input_image, clicks_lists
        )
        points_nd = self.get_points_nd(clicks_lists)
        pred_logits = self.net(image_nd, points_nd)["instances"]
        prediction = F.interpolate(
            pred_logits, mode="bilinear", align_corners=True, size=image_nd.size()[2:]
        )

        for t in reversed(self.transforms):
            prediction = t.inv_transform(prediction)

        self.prev_prediction = prediction
        return prediction.cpu().numpy()[:, 0]

    def _get_transform_states(self):
        return [x.get_state() for x in self.transforms]

    def _set_transform_states(self, states):
        assert len(states) == len(self.transforms)
        for state, transform in zip(states, self.transforms):
            transform.set_state(state)

    def apply_transforms(
        self, image_nd: torch.Tensor, clicks_lists: List[Click]
    ) -> Tuple:
        is_image_changed = False
        for t in self.transforms:
            image_nd, clicks_lists = t.transform(image_nd, clicks_lists)
            is_image_changed |= t.image_changed

        return image_nd, clicks_lists, is_image_changed

    def get_points_nd(self, clicks_lists: List[List[Click]]) -> torch.Tensor:
        total_clicks = []
        num_pos_clicks = [
            sum(x.is_positive for x in clicks_list) for clicks_list in clicks_lists
        ]
        num_neg_clicks = [
            len(clicks_list) - num_pos
            for clicks_list, num_pos in zip(clicks_lists, num_pos_clicks)
        ]
        num_max_points = max(num_pos_clicks + num_neg_clicks)
        if self.net_clicks_limit is not None:
            num_max_points = min(self.net_clicks_limit, num_max_points)
        num_max_points = max(1, num_max_points)

        for clicks_list in clicks_lists:
            clicks_list = clicks_list[: self.net_clicks_limit]
            pos_clicks = [
                click.coords_and_indx for click in clicks_list if click.is_positive
            ]
            pos_clicks = pos_clicks + (num_max_points - len(pos_clicks)) * [
                (-1, -1, -1)
            ]

            neg_clicks = [
                click.coords_and_indx for click in clicks_list if not click.is_positive
            ]
            neg_clicks = neg_clicks + (num_max_points - len(neg_clicks)) * [
                (-1, -1, -1)
            ]
            total_clicks.append(pos_clicks + neg_clicks)

        return torch.tensor(total_clicks, device=self.device)

    def get_states(self) -> Dict:
        return {
            "transform_states": self._get_transform_states(),
            "prev_prediction": self.prev_prediction.clone(),
        }

    def set_states(self, states: Dict) -> None:
        self._set_transform_states(states["transform_states"])
        self.prev_prediction = states["prev_prediction"]
