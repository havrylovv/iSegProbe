"""Main training script."""

import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from core.utils.exp import init_experiment
from core.utils.misc import load_module


@hydra.main(config_path="configs/", config_name="train_cfg", version_base="1.1")
def main(train_cfg: DictConfig) -> None:
    model_script = load_module(train_cfg.exp.model_path)
    model_base_name = getattr(model_script, "MODEL_NAME", None)
    train_cfg.training.distributed = "WORLD_SIZE" in os.environ
    cfg = init_experiment(train_cfg, model_base_name)

    torch.multiprocessing.set_sharing_strategy("file_system")

    # save config
    hydra_cfg_path = os.path.join(cfg.EXP_PATH, "hydra_config.yaml")
    with open(hydra_cfg_path, "w") as f:
        OmegaConf.save(cfg, f)

    model_script.main(cfg)


if __name__ == "__main__":
    main()
