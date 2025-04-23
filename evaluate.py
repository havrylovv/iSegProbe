"""Main evaluation script."""

from pathlib import Path

import hydra
import numpy as np
import wandb
from omegaconf import DictConfig, OmegaConf

from core.inference import utils
from core.inference.evaluation import evaluate_dataset
from core.inference.predictors import get_predictor
from core.inference.utils import (
    get_checkpoints_list_and_logs_path,
    get_device,
    get_prediction_vis_callback,
    get_predictor_and_zoomin_params,
    get_save_feats_callback,
    get_wandb_run_name,
    save_iou_analysis_data,
    save_results,
    update_eval_and_load_data_configs,
)
from core.model.featurizers.utils import interpolate_pos_embed_inference
from core.utils.exp import logger
from core.utils.log import buffering_handler
from core.utils.misc import seed_all


@hydra.main(config_path="configs/", config_name="eval_cfg", version_base="1.1")
def main(eval_cfg: DictConfig) -> None:
    seed_all(0)
    # collect all logs in a buffer for potential logging to wandb
    logger.addHandler(buffering_handler)
    eval_cfg, main_cfg = update_eval_and_load_data_configs(eval_cfg)
    device = get_device(eval_cfg)
    checkpoints_list, logs_path, logs_prefix = get_checkpoints_list_and_logs_path(
        eval_cfg, main_cfg
    )
    logs_path.mkdir(parents=True, exist_ok=True)

    if eval_cfg.wandb:
        if eval_cfg.checkpoint == "":
            raise ValueError(
                "Checkpoint path is required for logging to wandb. \
                    Logging of entire experiments to wandb is not supported."
            )

        if type(eval_cfg.datasets) == str:
            num_datasets = len(eval_cfg.datasets.split(","))
        elif type(eval_cfg.datasets) == list:
            num_datasets = len(eval_cfg.datasets)

        if num_datasets > 1:
            raise ValueError("Logging to wandb is not supported for multiple datasets.")

    single_model_eval = len(checkpoints_list) == 1
    assert (
        not eval_cfg.iou_analysis if not single_model_eval else True
    ), "Can't perform IoU analysis for multiple checkpoints"

    print_header = True  # for evaluation table
    target_iou = eval_cfg.target_iou

    for dataset_name in eval_cfg.datasets.split(","):
        dataset = utils.get_dataset(dataset_name, main_cfg)

        for checkpoint_path in checkpoints_list:
            logger.info(f"Evaluating model from {checkpoint_path}")

            model = utils.load_is_model(
                checkpoint_path,
                device,
                eval_cfg.eval_ritm,
            )
            predictor_params, zoomin_params = get_predictor_and_zoomin_params(
                eval_cfg, dataset_name, eval_ritm=eval_cfg.eval_ritm
            )

            logger.info(f"Use ZoomIn with params: {zoomin_params}")

            # For SimpleClick models, we usually need to interpolate the positional embedding
            if not eval_cfg.eval_ritm:
                interpolate_pos_embed_inference(
                    model.backbone, zoomin_params["target_size"], device
                )
            predictor = get_predictor(
                model,
                eval_cfg.mode,
                device,
                prob_thresh=eval_cfg.thresh,
                predictor_params=predictor_params,
                zoom_in_params=zoomin_params,
            )
            # callback for visualizing predictions
            vis_callback = (
                get_prediction_vis_callback(logs_path, dataset_name, eval_cfg.thresh)
                if eval_cfg.vis_preds
                else None
            )
            # callback for operations with features
            if eval_cfg.save_feats:
                feats_callback = get_save_feats_callback(
                    logs_path,
                    dataset_name,
                    eval_cfg.save_feats_folder_name,
                    exec_for_n_imgs=eval_cfg.save_feats_for_n_imgs,
                )
            else:
                feats_callback = None

            eval_cfg.target_iou = target_iou
            dataset_results = evaluate_dataset(
                dataset,
                predictor,
                pred_thr=eval_cfg.thresh,
                max_iou_thr=eval_cfg.target_iou,
                min_clicks=eval_cfg.min_n_clicks,
                max_clicks=eval_cfg.n_clicks,
                callback=vis_callback,
                feats_callback=feats_callback,
            )

            if eval_cfg.iou_analysis:
                save_iou_analysis_data(
                    eval_cfg,
                    dataset_name,
                    logs_path,
                    logs_prefix,
                    dataset_results,
                    model_name=eval_cfg.model_name,
                )

            row_name = eval_cfg.mode if single_model_eval else checkpoint_path.stem
            results = save_results(
                model,
                eval_cfg,
                row_name,
                dataset_name,
                logs_path,
                logs_prefix,
                dataset_results,
                save_ious=single_model_eval and eval_cfg.save_ious,
                single_model_eval=single_model_eval,
                print_header=print_header,
            )
            print_header = False

            if eval_cfg.wandb:
                wandb_config = eval_cfg
                wandb_config = OmegaConf.to_container(wandb_config, resolve=True)
                wandb_config["main_cfg"] = dict(main_cfg)
                wandb_config["zoomin_params"] = zoomin_params

                wandb.init(
                    project=eval_cfg.wandb_project,
                    name=get_wandb_run_name(eval_cfg),
                    config=wandb_config,
                    dir=eval_cfg.wandb_dir,
                )
                wandb.log(results)

                # Log mIoU@X as a table and create line plot
                miou_table = wandb.Table(columns=["k (Clicks)", "mIoU@k"])
                assert len(results["clicks_list"]) == len(results["miou_list"])
                if not isinstance(results["clicks_list"], list):
                    results["clicks_list"] = list(results["clicks_list"])
                    results["miou_list"] = list(results["miou_list"])

                X = results["clicks_list"]
                Y = results["miou_list"]

                for i in range(len(X)):
                    miou_table.add_data(
                        int(X[i]), Y[i]
                    )  # NOTE: wandb does not accept float values for X

                wandb.log(
                    {
                        "mIoU_vs_Clicks": wandb.plot.line(
                            miou_table,
                            "k (Clicks)",
                            "mIoU@k",
                            title="mIoU Given k Clicks",
                        )
                    }
                )

                # Log NoC@X
                noc_table = wandb.Table(columns=["X (IoU, %)", "NoC@X"])
                for X in [80, 85, 90]:
                    noc_table.add_data(X, results[f"NoC@{X}%"])

                wandb.log(
                    {
                        "NoC_vs_IoU": wandb.plot.line(
                            noc_table,
                            "X (IoU, %)",
                            "NoC@X",
                            title="Number of Clicks to Reach X% IoU",
                        )
                    }
                )

                # write all captured logs from buffering_handler to wandb output.log file (ONLY locally, not on wandb server)
                output_log_path = Path(wandb.run.dir) / "output.log"
                with open(output_log_path, "a") as f:
                    for log_entry in buffering_handler.buffer:
                        f.write(log_entry + "\n")
                    buffering_handler.buffer = []

                wandb.finish()

    # uncomment the following lines for memory analysis
    # print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    # print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
    # print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))


if __name__ == "__main__":
    main()
