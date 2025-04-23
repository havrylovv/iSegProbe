"""Additional optimizers that are be used for BRS (Backpropagation Refinement Scheme) evaluation regime."""

from typing import Callable, Dict, Tuple

import numpy as np
import torch

from core.training.metrics import _compute_iou

from .brs_losses import BRSMaskLoss


class BaseOptimizer:
    def __init__(
        self,
        optimizer_params: Dict,
        prob_thresh: float = 0.49,
        reg_weight: float = 1e-3,
        min_iou_diff: float = 0.01,
        brs_loss: torch.nn.Module = BRSMaskLoss(),
        with_flip: bool = False,
        flip_average: bool = False,
        **kwargs
    ) -> None:
        self.brs_loss = brs_loss
        self.optimizer_params = optimizer_params
        self.prob_thresh = prob_thresh
        self.reg_weight = reg_weight
        self.min_iou_diff = min_iou_diff
        self.with_flip = with_flip
        self.flip_average = flip_average

        self.best_prediction = None
        self._get_prediction_logits = None
        self._opt_shape = None
        self._best_loss = None
        self._click_masks = None
        self._last_mask = None
        self.device = None

    def init_click(
        self,
        get_prediction_logits: Callable,
        pos_mask: torch.Tensor,
        neg_mask: torch.Tensor,
        device: str,
        shape: Tuple = None,
    ) -> None:
        self.best_prediction = None
        self._get_prediction_logits = get_prediction_logits
        self._click_masks = (pos_mask, neg_mask)
        self._opt_shape = shape
        self._last_mask = None
        self.device = device

    def __call__(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        opt_params = torch.from_numpy(x).float().to(self.device)
        opt_params.requires_grad_(True)

        with torch.enable_grad():
            opt_vars, reg_loss = self.unpack_opt_params(opt_params)
            result_before_sigmoid = self._get_prediction_logits(*opt_vars)
            result = torch.sigmoid(result_before_sigmoid)

            pos_mask, neg_mask = self._click_masks
            if self.with_flip and self.flip_average:
                result, result_flipped = torch.chunk(result, 2, dim=0)
                result = 0.5 * (result + torch.flip(result_flipped, dims=[3]))
                pos_mask, neg_mask = (
                    pos_mask[: result.shape[0]],
                    neg_mask[: result.shape[0]],
                )

            loss, f_max_pos, f_max_neg = self.brs_loss(result, pos_mask, neg_mask)
            loss = loss + reg_loss

        f_val = loss.detach().cpu().numpy()
        if self.best_prediction is None or f_val < self._best_loss:
            self.best_prediction = result_before_sigmoid.detach()
            self._best_loss = f_val

        if f_max_pos < (1 - self.prob_thresh) and f_max_neg < self.prob_thresh:
            return [f_val, np.zeros_like(x)]

        current_mask = result > self.prob_thresh
        if self._last_mask is not None and self.min_iou_diff > 0:
            diff_iou = _compute_iou(current_mask, self._last_mask)
            if len(diff_iou) > 0 and diff_iou.mean() > 1 - self.min_iou_diff:
                return [f_val, np.zeros_like(x)]
        self._last_mask = current_mask

        loss.backward()
        f_grad = opt_params.grad.cpu().numpy().ravel().astype(np.float)

        return [f_val, f_grad]

    def unpack_opt_params(self, opt_params) -> NotImplementedError:
        raise NotImplementedError


class InputOptimizer(BaseOptimizer):
    def unpack_opt_params(
        self, opt_params: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        opt_params = opt_params.view(self._opt_shape)
        if self.with_flip:
            opt_params_flipped = torch.flip(opt_params, dims=[3])
            opt_params = torch.cat([opt_params, opt_params_flipped], dim=0)
        reg_loss = self.reg_weight * torch.sum(opt_params**2)

        return (opt_params,), reg_loss


class ScaleBiasOptimizer(BaseOptimizer):
    def __init__(
        self, *args, scale_act: str = None, reg_bias_weight: float = 10.0, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.scale_act = scale_act
        self.reg_bias_weight = reg_bias_weight

    def unpack_opt_params(
        self, opt_params: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        scale, bias = torch.chunk(opt_params, 2, dim=0)
        reg_loss = self.reg_weight * (
            torch.sum(scale**2) + self.reg_bias_weight * torch.sum(bias**2)
        )

        if self.scale_act == "tanh":
            scale = torch.tanh(scale)
        elif self.scale_act == "sin":
            scale = torch.sin(scale)

        return (1 + scale, bias), reg_loss
