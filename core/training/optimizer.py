"""Training optimizers."""

import math
from typing import Dict

import torch
from torch.optim import Optimizer

import core.utils.lr_decay as lrd
from core.model import iSegBaseModel
from core.utils.log import logger


def get_optimizer(model: iSegBaseModel, opt_name: str, opt_kwargs: Dict) -> Optimizer:
    params = []
    base_lr = opt_kwargs["lr"]
    for name, param in model.named_parameters():
        param_group = {"params": [param]}
        if not param.requires_grad:
            params.append(param_group)
            continue

        if not math.isclose(getattr(param, "lr_mult", 1.0), 1.0):
            logger.info(f'Applied lr_mult={param.lr_mult} to "{name}" parameter.')
            param_group["lr"] = param_group.get("lr", base_lr) * param.lr_mult

        params.append(param_group)

    optimizer = {
        "sgd": torch.optim.SGD,
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
    }[opt_name.lower()](params, **opt_kwargs)

    return optimizer


def get_optimizer_with_layerwise_decay(
    model: iSegBaseModel, opt_name: str, opt_kwargs: Dict
) -> Optimizer:
    # build optimizer with layer-wise lr decay (lrd)
    lr = opt_kwargs["lr"]
    param_groups = lrd.param_groups_lrd(
        model,
        lr,
        weight_decay=0.02,
        no_weight_decay_list=model.backbone.no_weight_decay(),
        layer_decay=0.75,
    )
    optimizer = {
        "sgd": torch.optim.SGD,
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
    }[opt_name.lower()](param_groups, **opt_kwargs)

    return optimizer
