"""Default configurations for defining experiments."""

from functools import partial
from typing import Dict, Tuple

import albumentations as A
from easydict import EasyDict as edict
from omegaconf import DictConfig
from torch.optim.lr_scheduler import LRScheduler, MultiStepLR

from core.data import iSegBaseDataset
from core.data.datasets import SBDDataset
from core.data.points_sampler import MultiPointSampler
from core.data.transforms import UniformRandomResize
from core.training.losses import NormalizedFocalLossSigmoid
from core.utils.model_builder import ModelBuilder

model_builder = ModelBuilder()


def get_loss_cfg(cfg: DictConfig) -> edict:
    """Get the loss configuration."""
    loss_cfg = edict()
    loss_cfg.instance_loss = NormalizedFocalLossSigmoid(alpha=0.5, gamma=2)
    loss_cfg.instance_loss_weight = 1.0
    return loss_cfg


def get_sbd_train_val_datasets(
    cfg: DictConfig,
) -> Tuple[iSegBaseDataset, iSegBaseDataset]:
    """Get the training and validation datasets for SBD dataset."""
    cfg.dataloader.batch_size = (
        32 if cfg.dataloader.batch_size < 1 else cfg.dataloader.batch_size
    )
    cfg.dataloader.val_batch_size = cfg.dataloader.batch_size
    crop_size = cfg.training_params.crop_size

    train_augmentator = A.Compose(
        [
            UniformRandomResize(scale_range=(0.75, 1.25)),
            A.Flip(),
            A.RandomRotate90(),
            A.ShiftScaleRotate(
                shift_limit=0.03,
                scale_limit=0,
                rotate_limit=(-3, 3),
                border_mode=0,
                p=0.75,
            ),
            A.PadIfNeeded(
                min_height=crop_size[0], min_width=crop_size[1], border_mode=0
            ),
            A.RandomCrop(*crop_size),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.25, 0.25), contrast_limit=(-0.15, 0.4), p=0.75
            ),
            A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.75),
        ],
        p=1.0,
    )

    val_augmentator = A.Compose(
        [
            UniformRandomResize(scale_range=(0.75, 1.25)),
            A.PadIfNeeded(
                min_height=crop_size[0], min_width=crop_size[1], border_mode=0
            ),
            A.RandomCrop(*crop_size),
        ],
        p=1.0,
    )

    points_sampler = MultiPointSampler(
        cfg.training_params.num_max_points,
        prob_gamma=0.80,
        merge_objects_prob=0.15,
        max_num_merged_objects=2,
    )

    trainset = SBDDataset(
        cfg.DATASETS.SBD_PATH,
        split="train",
        augmentator=train_augmentator,
        min_object_area=80,
        keep_background_prob=0.01,
        points_sampler=points_sampler,
        samples_scores_path="./assets/sbd_samples_weights.pkl",
        samples_scores_gamma=1.25,
    )

    valset = SBDDataset(
        cfg.DATASETS.SBD_PATH,
        split="val",
        augmentator=val_augmentator,
        min_object_area=80,
        points_sampler=points_sampler,
        epoch_len=500,
    )
    return trainset, valset


def get_optimizer_cfg(cfg: DictConfig) -> Tuple[str, Dict]:
    optimizer_name = "adam"
    optimizer_params = {"lr": 5e-5, "betas": (0.9, 0.999), "eps": 1e-8}

    return optimizer_name, optimizer_params


def get_lr_scheduler(cfg: DictConfig) -> LRScheduler:
    lr_scheduler = partial(
        MultiStepLR, milestones=cfg.training_params.lr_milestones, gamma=0.1
    )
    return lr_scheduler
