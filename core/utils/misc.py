"""Miscellaneous utility functions."""

import importlib.util
import random
from os import environ
from pathlib import Path
from types import ModuleType
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig

from core.model import iSegBaseModel

from .log import logger


def load_module(script_path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location("model_script", script_path)
    model_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_script)

    return model_script


def get_dims_with_exclusion(dim, exclude=None):
    dims = list(range(dim))
    if exclude is not None:
        dims.remove(exclude)

    return dims


def save_checkpoint(
    net: iSegBaseModel,
    checkpoints_path: Path,
    epoch: str = None,
    prefix: str = "",
    verbose: bool = True,
    multi_gpu: bool = False,
) -> None:
    if epoch is None:
        checkpoint_name = "last_checkpoint.pth"
    else:
        checkpoint_name = f"{epoch:03d}.pth"

    if prefix:
        checkpoint_name = f"{prefix}_{checkpoint_name}"

    if not checkpoints_path.exists():
        checkpoints_path.mkdir(parents=True)

    checkpoint_path = checkpoints_path / checkpoint_name
    if verbose:
        logger.info(f"Save checkpoint to {str(checkpoint_path)}")

    net = net.module if multi_gpu else net

    if hasattr(net, "get_state_dict_to_save"):
        state_dict = (
            net.get_state_dict_to_save()
        )  # check if model has custom function get_state_dict_to_save()
    else:
        state_dict = net.state_dict()

    torch.save({"state_dict": state_dict, "config": net._config}, str(checkpoint_path))


def get_bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


def expand_bbox(bbox: Tuple, expand_ratio: float, min_crop_size: int = None) -> Tuple:
    rmin, rmax, cmin, cmax = bbox
    rcenter = 0.5 * (rmin + rmax)
    ccenter = 0.5 * (cmin + cmax)
    height = expand_ratio * (rmax - rmin + 1)
    width = expand_ratio * (cmax - cmin + 1)
    if min_crop_size is not None:
        height = max(height, min_crop_size)
        width = max(width, min_crop_size)

    rmin = int(round(rcenter - 0.5 * height))
    rmax = int(round(rcenter + 0.5 * height))
    cmin = int(round(ccenter - 0.5 * width))
    cmax = int(round(ccenter + 0.5 * width))

    return rmin, rmax, cmin, cmax


def clamp_bbox(
    bbox: Tuple, rmin: float, rmax: float, cmin: float, cmax: float
) -> Tuple:
    return (
        max(rmin, bbox[0]),
        min(rmax, bbox[1]),
        max(cmin, bbox[2]),
        min(cmax, bbox[3]),
    )


def get_bbox_iou(b1, b2):
    h_iou = get_segments_iou(b1[:2], b2[:2])
    w_iou = get_segments_iou(b1[2:4], b2[2:4])
    return h_iou * w_iou


def get_segments_iou(s1: Tuple, s2: Tuple) -> float:
    a, b = s1
    c, d = s2
    intersection = max(0, min(b, d) - max(a, c) + 1)
    union = max(1e-6, max(b, d) - min(a, c) + 1)
    return intersection / union


def get_labels_with_sizes(x: np.ndarray) -> Tuple[list, list]:
    obj_sizes = np.bincount(x.flatten())
    labels = np.nonzero(obj_sizes)[0].tolist()
    labels = [x for x in labels if x != 0]
    return labels, obj_sizes[labels].tolist()


def seed_all(seed: int, log=True) -> None:
    if seed == -1:
        return

    import os

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = (
        ":4096:8"  #':16:8'  # fix CuBLAS for CUDA >= 10.2
    )
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(mode=True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if log:
        logger.info(f"Set seed to: {seed}")


def seed_worker(worker_id):
    """Used to set the seed for each worker in a DataLoader."""
    logger.info(f"Seeded Dataloader's worker #{worker_id}")
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device(cfg: DictConfig) -> torch.device:
    if cfg.training.distributed:
        device = torch.device("cuda")
        cfg.training.gpu_ids = [cfg.training.gpu_ids[cfg.training.local_rank]]
        torch.cuda.set_device(cfg.training.gpu_ids[0])
    else:
        if cfg.training.multi_gpu:
            environ["CUDA_VISIBLE_DEVICES"] = cfg.training.gpus
            ngpus = torch.cuda.device_count()
            assert ngpus >= cfg.training.ngpus
        device = torch.device(f"cuda:{cfg.training.gpu_ids[0]}")
    return device


class Lambda(nn.Module):
    def __init__(self, func):
        """Converts single-argument function to an nn.Module."""
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)
