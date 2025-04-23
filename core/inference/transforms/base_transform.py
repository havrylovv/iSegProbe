"""Base transform class used during evaluation phase."""

from typing import Dict, List, Tuple

import torch

from core.inference.clicker import Click


class BaseTransform(object):
    def __init__(self) -> None:
        self.image_changed = False

    def transform(
        self, image_nd: torch.Tensor, clicks_lists: List[List[Click]]
    ) -> torch.Tensor:
        raise NotImplementedError

    def inv_transform(self, prob_map: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    def get_state(self) -> Dict:
        raise NotImplementedError

    def set_state(self, state: Dict) -> None:
        raise NotImplementedError


class SigmoidForPred(BaseTransform):
    def transform(
        self, image_nd: torch.Tensor, clicks_lists: List[List[Click]]
    ) -> Tuple[torch.Tensor, List[List[Click]]]:
        return image_nd, clicks_lists

    def inv_transform(self, prob_map: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(prob_map)

    def reset(self) -> None:
        pass

    def get_state(self) -> None:
        return None

    def set_state(self, state: Dict) -> None:
        pass
