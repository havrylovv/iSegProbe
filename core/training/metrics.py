"""Training metrics."""

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from core.utils import misc


class TrainMetric(object):
    def __init__(self, pred_outputs, gt_outputs):
        self.pred_outputs = pred_outputs
        self.gt_outputs = gt_outputs

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def get_epoch_value(self):
        raise NotImplementedError

    def reset_epoch_stats(self):
        raise NotImplementedError

    def log_states(self, sw, tag_prefix, global_step):
        pass

    @property
    def name(self):
        return type(self).__name__


class AdaptiveIoU(TrainMetric):
    def __init__(
        self,
        init_thresh: float = 0.4,
        thresh_step: float = 0.025,
        thresh_beta: float = 0.99,
        iou_beta: float = 0.9,
        ignore_label: float = -1,
        from_logits: bool = True,
        pred_output: str = "instances",
        gt_output: str = "instances",
    ) -> None:
        super().__init__(pred_outputs=(pred_output,), gt_outputs=(gt_output,))
        self._ignore_label = ignore_label
        self._from_logits = from_logits
        self._iou_thresh = init_thresh
        self._thresh_step = thresh_step
        self._thresh_beta = thresh_beta
        self._iou_beta = iou_beta
        self._ema_iou = 0.0
        self._epoch_iou_sum = 0.0
        self._epoch_batch_count = 0

    def update(self, pred: torch.Tensor, gt: torch.Tensor) -> None:
        gt_mask = gt > 0.5
        if self._from_logits:
            pred = torch.sigmoid(pred)

        gt_mask_area = torch.sum(gt_mask, dim=(1, 2)).detach().cpu().numpy()
        if np.all(gt_mask_area == 0):
            return

        ignore_mask = gt == self._ignore_label
        max_iou = _compute_iou(pred > self._iou_thresh, gt_mask, ignore_mask).mean()
        best_thresh = self._iou_thresh
        for t in [best_thresh - self._thresh_step, best_thresh + self._thresh_step]:
            temp_iou = _compute_iou(pred > t, gt_mask, ignore_mask).mean()
            if temp_iou > max_iou:
                max_iou = temp_iou
                best_thresh = t

        self._iou_thresh = (
            self._thresh_beta * self._iou_thresh + (1 - self._thresh_beta) * best_thresh
        )
        self._ema_iou = self._iou_beta * self._ema_iou + (1 - self._iou_beta) * max_iou
        self._epoch_iou_sum += max_iou
        self._epoch_batch_count += 1

    def get_epoch_value(self) -> float:
        if self._epoch_batch_count > 0:
            return self._epoch_iou_sum / self._epoch_batch_count
        else:
            return 0.0

    def reset_epoch_stats(self) -> None:
        self._epoch_iou_sum = 0.0
        self._epoch_batch_count = 0

    def log_states(self, sw: SummaryWriter, tag_prefix: str, global_step: int) -> None:
        sw.add_scalar(
            tag=tag_prefix + "_ema_iou", value=self._ema_iou, global_step=global_step
        )
        sw.add_scalar(
            tag=tag_prefix + "_iou_thresh",
            value=self._iou_thresh,
            global_step=global_step,
        )

    @property
    def iou_thresh(self) -> float:
        return self._iou_thresh


def _compute_iou(
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    ignore_mask: bool = None,
    keep_ignore: bool = False,
) -> np.ndarray:
    if ignore_mask is not None:
        pred_mask = torch.where(ignore_mask, torch.zeros_like(pred_mask), pred_mask)

    reduction_dims = misc.get_dims_with_exclusion(gt_mask.dim(), 0)
    union = (
        torch.mean((pred_mask | gt_mask).float(), dim=reduction_dims)
        .detach()
        .cpu()
        .numpy()
    )
    intersection = (
        torch.mean((pred_mask & gt_mask).float(), dim=reduction_dims)
        .detach()
        .cpu()
        .numpy()
    )
    nonzero = union > 0

    iou = intersection[nonzero] / union[nonzero]
    if not keep_ignore:
        return iou
    else:
        result = np.full_like(intersection, -1)
        result[nonzero] = iou
        return result
