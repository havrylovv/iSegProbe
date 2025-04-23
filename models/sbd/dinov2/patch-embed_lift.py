"""
Backbone: DINOv2
Click encoder: PatchEmbed
Upsampler: LiFT
Click Features Injection: before backbone (early injection)
"""

from easydict import EasyDict as edict
from omegaconf import DictConfig

from core.model.iseg_probe_model import iSegProbeModel
from core.training.metrics import AdaptiveIoU
from core.training.trainer import iSegTrainer
from core.utils.log import init_wandb
from core.utils.misc import get_device, seed_all
from models.defaults import (
    get_loss_cfg,
    get_lr_scheduler,
    get_optimizer_cfg,
    get_sbd_train_val_datasets,
    model_builder,
)

global MODEL_NAME


def define_modules_cfg(cfg: DictConfig) -> edict:
    """Core function to be filled by the user. It defines architecture and configs for modules of the `iSegProbeModel` model.
    Returns:
        (dict): A dictionary containing the model architecture and configurations for each module.
    """
    # unique model name
    MODEL_NAME = "sbd_dinov2_lift_convhead_patchembed_earlyinject_224"
    # model architecture, options: backbone_upsampler_head, backbone_neck_head
    ARCHITECTURE = "backbone_upsampler_head"

    backbone_cfg = dict(
        type="dinov2",
        params=dict(
            feats_injection_mode="before_backbone",
        ),
    )

    embed_coords_cfg = dict(
        type="patchEmbed",
        params=dict(
            img_size=cfg.training_params.crop_size, patch_size=(14, 14), embed_dim=384
        ),
    )

    head_cfg = dict(
        type="convhead",
        params=dict(
            in_channels=384,
            num_layers=2,
            num_classes=1,
        ),
    )

    upsampler_cfg = dict(
        type="lift",
        params=dict(
            lift_path=cfg.UPSAMPLERS.LIFT,
            n_dim=384,
            patch=14,
        ),
    )
    neck_cfg = None

    # define which parts of the model to save during training
    # do not save frozen modules
    save_cfg = dict(
        embed_coords=True,
        backbone=False,
        upsampler=False,
        head=True,
    )

    return edict(
        {
            "backbone": backbone_cfg,
            "upsampler": upsampler_cfg,
            "head": head_cfg,
            "embed_coords": embed_coords_cfg,
            "neck": neck_cfg,
            "save": save_cfg,
            "architecture": ARCHITECTURE,
        }
    )


def init_model(cfg: DictConfig):
    """Initialize the model. Core information is defined through `define_modules_cfg` function."""
    modules_cfg = define_modules_cfg(cfg)

    model = iSegProbeModel(
        backbone_cfg=modules_cfg.backbone,
        head_cfg=modules_cfg.head,
        embed_coords_cfg=modules_cfg.embed_coords,
        neck_cfg=modules_cfg.neck,
        upsampler_cfg=modules_cfg.upsampler,
        save_cfg=modules_cfg.save,
        architecture=modules_cfg.architecture,
        model_builder=model_builder,
        use_disks=True,
        norm_radius=5,
        with_prev_mask=True,
    )

    device = get_device(cfg)
    model.to(device)

    return model


def train(model, cfg: DictConfig) -> None:
    """Initialize the trainer and start training. Core components could be defined by
    either importing default configurations from `models/defaults.py` or overriding them manually.
    """

    loss_cfg = get_loss_cfg(cfg)
    trainset, valset = get_sbd_train_val_datasets(cfg)
    optimizer_name, optimizer_params = get_optimizer_cfg(cfg)
    lr_scheduler = get_lr_scheduler(cfg)

    trainer = iSegTrainer(
        model,
        cfg,
        loss_cfg,
        trainset,
        valset,
        optimizer=optimizer_name,
        optimizer_params=optimizer_params,
        layerwise_decay=False,
        lr_scheduler=lr_scheduler,
        checkpoint_interval=[
            tuple(x) for x in cfg.training_params.checkpoint_interval
        ],  # convert to list of tuples
        image_dump_interval=300,
        metrics=[AdaptiveIoU()],
        max_interactive_points=cfg.training_params.num_max_points,
        max_num_next_clicks=3,
        seed=cfg.training.seed,
    )

    trainer.run(
        num_epochs=cfg.training_params.epochs,
        validation=cfg.training_params.do_validation,
    )


def main(cfg: DictConfig):
    """This function is the entry point for the training script."""
    seed_all(cfg.training.seed)
    init_wandb(cfg)
    model = init_model(cfg)
    train(model, cfg)
