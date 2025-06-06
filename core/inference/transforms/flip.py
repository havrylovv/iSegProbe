"""Flip transform for images and clicks."""

from typing import Dict, List, Tuple

import torch

from core.inference.clicker import Click

from .base_transform import BaseTransform


class AddHorizontalFlip(BaseTransform):
    def transform(
        self, image_nd: torch.Tensor, clicks_lists: List[List[Click]]
    ) -> Tuple[torch.Tensor, List[List[Click]]]:
        assert len(image_nd.shape) == 4
        image_nd = torch.cat([image_nd, torch.flip(image_nd, dims=[3])], dim=0)

        image_width = image_nd.shape[3]
        clicks_lists_flipped = []
        for clicks_list in clicks_lists:
            clicks_list_flipped = [
                click.copy(coords=(click.coords[0], image_width - click.coords[1] - 1))
                for click in clicks_list
            ]
            clicks_lists_flipped.append(clicks_list_flipped)
        clicks_lists = clicks_lists + clicks_lists_flipped

        return image_nd, clicks_lists

    def inv_transform(self, prob_map: torch.Tensor) -> torch.Tensor:
        assert len(prob_map.shape) == 4 and prob_map.shape[0] % 2 == 0
        num_maps = prob_map.shape[0] // 2
        prob_map, prob_map_flipped = prob_map[:num_maps], prob_map[num_maps:]

        return 0.5 * (prob_map + torch.flip(prob_map_flipped, dims=[3]))

    def get_state(self) -> None:
        return None

    def set_state(self, state: Dict) -> None:
        pass

    def reset(self) -> None:
        pass
