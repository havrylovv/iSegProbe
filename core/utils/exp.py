"""Utility functions for initializing and managing experiments."""

import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union

import torch
import yaml
from easydict import EasyDict as edict
from omegaconf import DictConfig, OmegaConf

from .distributed import get_world_size, synchronize
from .log import add_logging, logger


def init_experiment(train_cfg: DictConfig, model_name: str) -> DictConfig:
    model_path = Path(train_cfg.exp.model_path)
    ftree = get_model_family_tree(model_path, model_name=model_name)

    if ftree is None:
        print(
            'Models can only be located in the "models" directory in the root of the repository'
        )
        sys.exit(1)

    cfg = load_config(model_path)
    update_config(cfg, train_cfg)

    cfg.training.distributed = train_cfg.training.distributed
    cfg.training.local_rank = train_cfg.training.local_rank
    if cfg.training.distributed:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        if train_cfg.dataloader.workers > 0:
            torch.multiprocessing.set_start_method("forkserver", force=True)

    experiments_path = Path(cfg.EXPS_PATH)
    exp_parent_path = experiments_path / "/".join(ftree)
    exp_parent_path.mkdir(parents=True, exist_ok=True)

    if cfg.training.resume_exp:
        exp_path = find_resume_exp(exp_parent_path, cfg.training.resume_exp)
    else:
        last_exp_indx = find_last_exp_indx(exp_parent_path)
        exp_name = f"{last_exp_indx:03d}"
        if cfg.exp.name:
            exp_name += "_" + cfg.exp.name
        exp_path = exp_parent_path / exp_name
        synchronize()
        if cfg.training.local_rank == 0:
            exp_path.mkdir(parents=True)

    cfg.EXP_PATH = exp_path
    cfg.CHECKPOINTS_PATH = exp_path / "checkpoints"
    cfg.VIS_PATH = exp_path / "vis"
    cfg.LOGS_PATH = exp_path / "logs"

    if cfg.training.local_rank == 0:
        cfg.LOGS_PATH.mkdir(exist_ok=True)
        cfg.CHECKPOINTS_PATH.mkdir(exist_ok=True)
        cfg.VIS_PATH.mkdir(exist_ok=True)

        dst_script_path = exp_path / (
            model_path.stem
            + datetime.strftime(datetime.today(), "_%Y-%m-%d-%H-%M-%S.py")
        )
        shutil.copy(model_path, dst_script_path)

    synchronize()

    if cfg.training.gpus != "":
        gpu_ids = [int(id) for id in cfg.training.gpus.split(",")]
    else:
        gpu_ids = list(range(max(cfg.training.ngpus, get_world_size())))
        cfg.training.gpus = ",".join([str(id) for id in gpu_ids])

    cfg.training.gpu_ids = gpu_ids
    cfg.training.ngpus = len(gpu_ids)
    cfg.training.multi_gpu = cfg.training.ngpus > 1

    if cfg.training.local_rank == 0:
        add_logging(cfg.LOGS_PATH, prefix="train_")
        logger.info(f"Number of GPUs: {cfg.training.ngpus}")
        if cfg.training.distributed:
            logger.info(f"Multi-Process Multi-GPU Distributed Training")

        logger.info("Run experiment with config:")
        logger.info(OmegaConf.to_yaml(cfg))

    return cfg


def get_model_family_tree(
    model_path: str, terminate_name: str = "models", model_name: str = None
) -> Union[None, List[str]]:
    if model_name is None:
        model_name = model_path.stem
    family_tree = [model_name]
    for x in model_path.parents:
        if x.stem == terminate_name:
            break
        family_tree.append(x.stem)
    else:
        return None

    return family_tree[::-1]


def find_last_exp_indx(exp_parent_path: str) -> int:
    indx = 0
    for x in exp_parent_path.iterdir():
        if not x.is_dir():
            continue

        exp_name = x.stem
        if exp_name[:3].isnumeric():
            indx = max(indx, int(exp_name[:3]) + 1)

    return indx


def find_resume_exp(exp_parent_path: str, exp_pattern: str) -> str:
    candidates = sorted(exp_parent_path.glob(f"{exp_pattern}*"))
    if len(candidates) == 0:
        print(
            f'No experiments could be found that satisfies the pattern = "*{exp_pattern}"'
        )
        sys.exit(1)
    elif len(candidates) > 1:
        print("More than one experiment found:")
        for x in candidates:
            print(x)
        sys.exit(1)
    else:
        exp_path = candidates[0]
        print(f'Continue with experiment "{exp_path}"')

    return exp_path


def update_config(main_cfg: DictConfig, cfg: DictConfig) -> None:
    for key, value in cfg.items():
        # Skip if key (case-insensitive) exists in main config
        if key.lower() in (k.lower() for k in main_cfg.keys()):
            continue

        if isinstance(value, DictConfig):
            # If it's a nested DictConfig, merge recursively
            main_cfg[key] = OmegaConf.create({})
            update_config(main_cfg[key], value)
        else:
            main_cfg[key] = value


def load_config(model_path: str) -> DictConfig:
    model_name = model_path.stem
    config_path = model_path.parent / (model_name + ".yaml")

    if config_path.exists():
        cfg = load_config_file(config_path)
    else:
        cfg = dict()

    # start search for '/configs/main_cfg.yaml' from the model's parent directory
    config_parent = model_path.parent

    # traverse upwards from the model's parent directory until we reach the root
    while config_parent != config_parent.parent:  # If we reach the root directory
        search_path = config_parent / "configs" / "main_cfg.yaml"

        if search_path.exists():
            local_config = load_config_file(search_path, model_name=model_name)
            cfg.update({k: v for k, v in local_config.items() if k not in cfg})
            break

        config_parent = config_parent.parent

    return OmegaConf.create(cfg)


def load_config_file(
    config_path: str, model_name: str = None, return_edict: bool = False
) -> Union[Dict, edict]:
    """Loads yaml config file and returns it as a dictionary or EasyDict."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    if "SUBCONFIGS" in cfg:
        if model_name is not None and model_name in cfg["SUBCONFIGS"]:
            cfg.update(cfg["SUBCONFIGS"][model_name])
        del cfg["SUBCONFIGS"]

    return edict(cfg) if return_edict else cfg
