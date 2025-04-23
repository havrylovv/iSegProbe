"""Various utility functions for evaluation."""

import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from pprint import pprint
from typing import Callable, Dict, List, Tuple, Union

import cv2
import numpy as np
import torch
from omegaconf import DictConfig

from core.data.base_dataset import iSegBaseDataset
from core.data.datasets import *
from core.inference import utils
from core.model import iSegBaseModel
from core.utils.exp import load_config_file
from core.utils.log import logger
from core.utils.serialization import load_model
from core.utils.viz import draw_points, draw_probmap, draw_with_blend_and_clicks


def get_time_metrics(
    all_ious: List[np.ndarray], elapsed_time: float
) -> Tuple[float, float]:
    n_images = len(all_ious)
    n_clicks = sum(map(len, all_ious))

    mean_spc = elapsed_time / n_clicks
    mean_spi = elapsed_time / n_images

    return mean_spc, mean_spi


def load_is_model(
    checkpoint: str,
    device: torch.device,
    eval_ritm: bool,
    **kwargs,
) -> Union[iSegBaseModel, Tuple[iSegBaseModel, List[iSegBaseModel]]]:
    if isinstance(checkpoint, (str, Path)):
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=False)
        logger.info("Load checkpoint from: %s" % checkpoint)
    else:
        state_dict = checkpoint

    if isinstance(state_dict, list):
        model = load_single_is_model(state_dict[0], device, eval_ritm, **kwargs)
        models = [
            load_single_is_model(x, device, eval_ritm, **kwargs) for x in state_dict
        ]

        return model, models
    else:
        return load_single_is_model(state_dict, device, eval_ritm, **kwargs)


def load_single_is_model(
    state_dict: Dict[str, torch.Tensor],
    device: torch.device,
    eval_ritm: bool,
    **kwargs,
) -> iSegBaseModel:
    _config = state_dict["config"]

    pprint(_config)
    model = load_model(_config, eval_ritm, **kwargs)

    current_state_dict = model.state_dict()
    new_state_dict = state_dict["state_dict"]
    current_state_dict.update(new_state_dict)
    msg = model.load_state_dict(current_state_dict, strict=False)
    logger.info(f"Load Following Weights: {new_state_dict.keys()}")
    logger.info(f"Loading Message: {msg}")

    for param in model.parameters():
        param.requires_grad = False
    model.to(device)
    model.eval()

    return model


def get_dataset(dataset_name: str, main_cfg: DictConfig) -> iSegBaseDataset:
    if dataset_name == "GrabCut":
        dataset = GrabCutDataset(main_cfg.DATASETS.GRABCUT_PATH)
    elif dataset_name == "Berkeley":
        dataset = BerkeleyDataset(main_cfg.DATASETS.BERKELEY_PATH)
    elif dataset_name == "DAVIS":
        dataset = DavisDataset(main_cfg.DATASETS.DAVIS_PATH)
    elif dataset_name == "SBD":
        dataset = SBDEvaluationDataset(main_cfg.DATASETS.SBD_PATH)
    elif dataset_name == "SBD_Train":
        dataset = SBDEvaluationDataset(main_cfg.DATASETS.SBD_PATH, split="train")
    elif dataset_name == "PascalVOC":
        dataset = PascalVocDataset(main_cfg.DATASETS.PASCALVOC_PATH, split="test")
    elif dataset_name == "COCO_MVal":
        dataset = DavisDataset(main_cfg.DATASETS.COCO_MVAL_PATH)
    else:
        raise NotImplementedError(f"Dataset key: {dataset_name} is not found.")

    return dataset


def get_iou(
    gt_mask: np.ndarray, pred_mask: np.ndarray, ignore_label: int = -1
) -> float:
    ignore_gt_mask_inv = gt_mask != ignore_label
    obj_gt_mask = gt_mask == 1

    intersection = np.logical_and(
        np.logical_and(pred_mask, obj_gt_mask), ignore_gt_mask_inv
    ).sum()
    union = np.logical_and(
        np.logical_or(pred_mask, obj_gt_mask), ignore_gt_mask_inv
    ).sum()

    return intersection / union


def compute_noc_metric(
    all_ious: List[np.ndarray], iou_thrs: List[float], max_clicks: int = 20
) -> Tuple[List[float], List[float], List[int]]:
    def _get_noc(iou_arr, iou_thr):
        vals = iou_arr >= iou_thr
        return np.argmax(vals) + 1 if np.any(vals) else max_clicks

    noc_list = []
    noc_list_std = []
    over_max_list = []
    for iou_thr in iou_thrs:
        scores_arr = np.array(
            [_get_noc(iou_arr, iou_thr) for iou_arr in all_ious], dtype=np.int_
        )

        score = scores_arr.mean()
        score_std = scores_arr.std()
        over_max = (scores_arr == max_clicks).sum()

        noc_list.append(score)
        noc_list_std.append(score_std)
        over_max_list.append(over_max)

    return noc_list, noc_list_std, over_max_list


def find_checkpoint(weights_folder: str, checkpoint_name: str) -> str:
    weights_folder = Path(weights_folder)
    if ":" in checkpoint_name:
        model_name, checkpoint_name = checkpoint_name.split(":")
        models_candidates = [
            x for x in weights_folder.glob(f"{model_name}*") if x.is_dir()
        ]
        assert len(models_candidates) == 1
        model_folder = models_candidates[0]
    else:
        model_folder = weights_folder

    if checkpoint_name.endswith(".pth"):
        if Path(checkpoint_name).exists():
            checkpoint_path = checkpoint_name
        else:
            checkpoint_path = weights_folder / checkpoint_name
    else:
        model_checkpoints = list(model_folder.rglob(f"{checkpoint_name}*.pth"))
        assert len(model_checkpoints) == 1
        checkpoint_path = model_checkpoints[0]

    return str(checkpoint_path)


def get_results_table(
    noc_list: List[float],
    over_max_list: List[int],
    brs_type: str,
    dataset_name: str,
    mean_spc: float,
    elapsed_time: float,
    iou_first: float,
    n_clicks: int = 20,
    model_name: str = None,
    upsampler_type: str = None,
    single_model_eval: bool = True,
) -> Tuple[str, str, Dict]:
    """Table with evaluation results that is genereted for each evaluation run."""
    # construct table header
    upsampler_header = ""
    upsampler_row = ""

    upsampler_header += f'{"Upsampler Type":^20}|'

    if upsampler_type is not None:
        upsampler_row += f"{upsampler_type:^20}|"
    else:
        upsampler_row += f'{"":^20}|'

    # if single_model_eval: log BRS type, otherwise log name of checkpoint
    brs_type_or_ckpts = f'{"BRS Type":^13}|' if single_model_eval else f'{"Ckpt":^13}|'

    table_header = (
        f"|{upsampler_header}" + brs_type_or_ckpts + f'{"Dataset":^11}|'
        f'{"NoC@80%":^9}|'
        f'{"NoC@85%":^9}|'
        f'{"NoC@90%":^9}|'
        f'{"IoU@1":^9}|'
        f'{">="+str(n_clicks)+"@85%":^9}|'
        f'{">="+str(n_clicks)+"@90%":^9}|'
        f'{"SPC,s":^7}|'
        f'{"Time":^9}|'
    )

    row_width = len(table_header)

    header = (
        f"Eval results for model: {model_name}\n"
        if single_model_eval and (model_name is not None)
        else ""
    )
    header += "-" * row_width + "\n"
    header += table_header + "\n" + "-" * row_width

    # construct table row with results
    eval_time = str(timedelta(seconds=int(elapsed_time)))
    table_row = f"|{upsampler_row}{brs_type:^13}|{dataset_name:^11}|"
    table_row += f"{noc_list[0]:^9.2f}|"
    table_row += f"{noc_list[1]:^9.2f}|" if len(noc_list) > 1 else f'{"?":^9}|'
    table_row += f"{noc_list[2]:^9.2f}|" if len(noc_list) > 2 else f'{"?":^9}|'
    table_row += f"{iou_first:^9.2f}|"
    table_row += f"{over_max_list[1]:^9}|" if len(noc_list) > 1 else f'{"?":^9}|'
    table_row += f"{over_max_list[2]:^9}|" if len(noc_list) > 2 else f'{"?":^9}|'
    table_row += f"{mean_spc:^7.3f}|{eval_time:^9}|"

    # additionally save results in a dictionary
    results_dict = {
        "NoC@80%": noc_list[0],
        "NoC@85%": noc_list[1] if len(noc_list) > 1 else -1,
        "NoC@90%": noc_list[2] if len(noc_list) > 2 else -1,
        f">={n_clicks}@85%": over_max_list[1] if len(noc_list) > 1 else -1,
        f">={n_clicks}@90%": over_max_list[2] if len(noc_list) > 2 else -1,
        "SPC,s": mean_spc,
        "Time": eval_time,
    }

    return header, table_row, results_dict


def update_eval_and_load_data_configs(
    eval_cfg: DictConfig,
) -> Tuple[DictConfig, DictConfig]:
    """Updates the evaluation configuration and loads the data configuration (paths to pre-trained models and datasets)."""

    if (eval_cfg.iou_analysis or eval_cfg.print_ious) and eval_cfg.min_n_clicks <= 1:
        eval_cfg.target_iou = 1.01
    else:
        eval_cfg.target_iou = max(0.8, eval_cfg.target_iou)

    main_cfg = load_config_file(eval_cfg.main_cfg_path, return_edict=True)
    main_cfg.EXPS_PATH = Path(main_cfg.EXPS_PATH)

    if eval_cfg.logs_path == "":
        eval_cfg.logs_path = main_cfg.EXPS_PATH / "evaluation_logs"
    else:
        eval_cfg.logs_path = Path(eval_cfg.logs_path)

    return eval_cfg, main_cfg


def get_device(eval_cfg: DictConfig) -> torch.device:
    if eval_cfg.cpu:
        return torch.device("cpu")
    else:
        return torch.device(f"cuda:{eval_cfg.gpus.split(',')[0]}")


def get_predictor_and_zoomin_params(
    eval_cfg: DictConfig,
    dataset_name: str,
    apply_zoom_in: bool = True,
    eval_ritm: bool = False,
) -> Tuple[Dict, Dict]:
    """Load predictor parameters and zoom-in parameters for the evaluation."""
    predictor_params = {}

    if eval_cfg.clicks_limit is not None:
        if eval_cfg.clicks_limit == -1:
            eval_cfg.clicks_limit = eval_cfg.n_clicks
        predictor_params["net_clicks_limit"] = eval_cfg.clicks_limit

    zoom_in_params = None
    if apply_zoom_in and eval_ritm:
        if eval_cfg.eval_mode == "cvpr":
            zoom_in_params = {"target_size": 600 if dataset_name == "DAVIS" else 400}
        elif eval_cfg.eval_mode.startswith("fixed"):
            crop_size = int(eval_cfg.eval_mode[5:])
            zoom_in_params = {"skip_clicks": -1, "target_size": (crop_size, crop_size)}
        else:
            raise NotImplementedError

    if apply_zoom_in and not eval_ritm:
        if eval_cfg.eval_mode == "cvpr":
            zoom_in_params = {
                "skip_clicks": -1,
                "target_size": (672, 672) if dataset_name == "DAVIS" else (448, 448),
            }
        elif eval_cfg.eval_mode.startswith("fixed"):
            crop_size = eval_cfg.eval_mode.split(",")
            crop_size_h = int(crop_size[0][5:])
            crop_size_w = crop_size_h
            if len(crop_size) == 2:
                crop_size_w = int(crop_size[1])
            zoom_in_params = {
                "skip_clicks": -1,
                "target_size": (crop_size_h, crop_size_w),
            }
        else:
            raise NotImplementedError

    return predictor_params, zoom_in_params


def get_checkpoints_list_and_logs_path(
    eval_cfg: DictConfig, main_cfg: DictConfig
) -> Tuple[str, str, str]:
    """Get the list of checkpoints and the path to save logs based on the evaluation configuration."""
    logs_prefix = ""
    if eval_cfg.exp_path:
        rel_exp_path = eval_cfg.exp_path
        checkpoint_prefix = ""
        if ":" in rel_exp_path:
            rel_exp_path, checkpoint_prefix = rel_exp_path.split(":")

        exp_path_prefix = main_cfg.EXPS_PATH / rel_exp_path
        candidates = list(exp_path_prefix.parent.glob(exp_path_prefix.stem + "*"))
        assert len(candidates) == 1, "Invalid experiment path."
        exp_path = candidates[0]
        checkpoints_list = sorted(
            (exp_path / "checkpoints").glob(checkpoint_prefix + "*.pth"), reverse=True
        )
        assert len(checkpoints_list) > 0, "Couldn't find any checkpoints."

        if checkpoint_prefix:
            if len(checkpoints_list) == 1:
                logs_prefix = checkpoints_list[0].stem
            else:
                logs_prefix = f"all_{checkpoint_prefix}"
        else:
            logs_prefix = "all_checkpoints"

        logs_path = eval_cfg.logs_path / exp_path.relative_to(main_cfg.EXPS_PATH)
    else:
        checkpoints_list = [
            Path(
                utils.find_checkpoint(
                    main_cfg.INTERACTIVE_MODELS_PATH, eval_cfg.checkpoint
                )
            )
        ]
        logs_path = eval_cfg.logs_path / "others" / checkpoints_list[0].stem

    return checkpoints_list, logs_path, logs_prefix


def save_results(
    model: iSegBaseModel,
    eval_cfg: DictConfig,
    row_name: str,
    dataset_name: str,
    logs_path: str,
    logs_prefix: str,
    dataset_results: Tuple[List[np.ndarray], float],
    save_ious: bool = False,
    print_header: bool = True,
    single_model_eval: bool = False,
) -> Dict:
    """Saves the evaluation results to a file and prints them to the console. Returns all results in a dictionary."""
    results = {}
    all_ious, elapsed_time = dataset_results
    mean_spc, mean_spi = utils.get_time_metrics(all_ious, elapsed_time)

    iou_thrs = np.arange(0.8, min(0.95, eval_cfg.target_iou) + 0.001, 0.05).tolist()
    noc_list, noc_list_std, over_max_list = utils.compute_noc_metric(
        all_ious, iou_thrs=iou_thrs, max_clicks=eval_cfg.n_clicks
    )
    iou_first = np.array([ious[0] for ious in all_ious]).mean(0)
    row_name = (
        "last" if row_name == "last_checkpoint" else row_name
    )  # for evaluation table
    model_name = (
        str(logs_path.relative_to(eval_cfg.logs_path)) + ":" + logs_prefix
        if logs_prefix
        else logs_path.stem
    )

    upsampler_type = (
        model.upsampler.__class__.__name__
        if hasattr(model, "upsampler")
        else "No Upsampler"
    )

    header, table_row, metrics_dict = utils.get_results_table(
        noc_list,
        over_max_list,
        row_name,
        dataset_name,
        mean_spc,
        elapsed_time,
        iou_first,
        eval_cfg.n_clicks,
        model_name,
        upsampler_type,
        single_model_eval,
    )

    results.update(metrics_dict)

    # if to include mean ious after every click to evaluation table
    if eval_cfg.print_ious:
        min_num_clicks = min(len(x) for x in all_ious)
        mean_ious = np.array([x[:min_num_clicks] for x in all_ious]).mean(axis=0)
        miou_str = " ".join(
            [
                f"mIoU@{click_id}={mean_ious[click_id - 1]:.2%};"
                for click_id in [_ for _ in range(1, 21)]
                if click_id <= min_num_clicks
            ]
        )
        table_row += "; " + miou_str

        mean_ious = [round(mean_ious[i] * 100, 2) for i in range(len(mean_ious))]

        results.update(
            {
                f"mIoU@{click_id}": mean_ious[click_id - 1]
                for click_id in range(1, 21)
                if click_id <= min_num_clicks
            }
        )
        # for wandb logging as a plot
        miou_list = [
            mean_ious[click_id - 1]
            for click_id in range(1, 21)
            if click_id <= min_num_clicks
        ]
        clicks_list = [
            click_id for click_id in range(1, 21) if click_id <= min_num_clicks
        ]
        results["miou_list"] = miou_list
        results["clicks_list"] = clicks_list
    else:
        target_iou_int = int(eval_cfg.target_iou * 100)
        if target_iou_int not in [80, 85, 90]:
            noc_list, _, over_max_list = utils.compute_noc_metric(
                all_ious, iou_thrs=[eval_cfg.target_iou], max_clicks=eval_cfg.n_clicks
            )
            table_row += f" NoC@{eval_cfg.target_iou:.1%} = {noc_list[0]:.2f};"
            table_row += (
                f" >={eval_cfg.n_clicks}@{eval_cfg.target_iou:.1%} = {over_max_list[0]}"
            )

            results.update(
                {
                    f"NoC@{eval_cfg.target_iou:.1%}": round(noc_list[0], 2),
                    f">={eval_cfg.n_clicks}@{eval_cfg.target_iou:.1%}": over_max_list[
                        0
                    ],
                }
            )

    if print_header:
        print(header)
    print(table_row)

    # if to save all ious as pickle file for further analysis
    if save_ious:
        ious_path = logs_path / "ious" / (logs_prefix if logs_prefix else "")
        ious_path.mkdir(parents=True, exist_ok=True)
        with open(
            ious_path
            / f"{dataset_name}_{eval_cfg.eval_mode}_{eval_cfg.mode}_{eval_cfg.n_clicks}.pkl",
            "wb",
        ) as fp:
            pickle.dump(all_ious, fp)

    name_prefix = ""
    if logs_prefix:
        name_prefix = logs_prefix + "_"
        if not single_model_eval:
            name_prefix += f"{dataset_name}_"

    log_path = (
        logs_path
        / f"{name_prefix}{eval_cfg.eval_mode}_{eval_cfg.mode}_{eval_cfg.n_clicks}.txt"
    )
    if log_path.exists():
        with open(log_path, "a") as f:
            f.write(table_row + "\n")
    else:
        with open(log_path, "w") as f:
            if print_header:
                f.write(header + "\n")
            f.write(table_row + "\n")

    return results


def save_iou_analysis_data(
    eval_cfg: DictConfig,
    dataset_name: str,
    logs_path: str,
    logs_prefix: str,
    dataset_results: str,
    model_name: str = None,
) -> None:
    """Saves the IoU analysis data to a pickle file for further analysis."""
    all_ious, _ = dataset_results

    name_prefix = ""
    if logs_prefix:
        name_prefix = logs_prefix + "_"
    name_prefix += dataset_name + "_"
    if model_name is None:
        model_name = (
            str(logs_path.relative_to(eval_cfg.logs_path)) + ":" + logs_prefix
            if logs_prefix
            else logs_path.stem
        )

    pkl_path = (
        logs_path
        / f"plots/{name_prefix}{eval_cfg.eval_mode}_{eval_cfg.mode}_{eval_cfg.n_clicks}.pickle"
    )
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with pkl_path.open("wb") as f:
        pickle.dump(
            {
                "dataset_name": dataset_name,
                "model_name": f"{model_name}_{eval_cfg.mode}",
                "all_ious": all_ious,
            },
            f,
        )


def get_prediction_vis_callback(
    logs_path: str, dataset_name: str, prob_thresh: float
) -> Callable:
    """Callback function to visualize predictions (clicks and masks) during evaluation."""
    save_path = logs_path / "predictions_vis" / dataset_name
    save_path.mkdir(parents=True, exist_ok=True)

    # get path to save IoU
    iou_save_path = logs_path / "predictions_vis" / dataset_name / "iou"
    iou_save_path.mkdir(parents=True, exist_ok=True)

    def callback(image, gt_mask, pred_probs, sample_id, click_indx, clicks_list):
        sample_path = save_path / f"{sample_id}_{click_indx}.jpg"
        prob_map = draw_probmap(
            pred_probs
        )  # NOTE: currently not used, but you can also save the probability map if needed
        image_with_mask = draw_with_blend_and_clicks(
            image, pred_probs > prob_thresh, alpha=0.5, clicks_list=clicks_list
        )

        # calculate IoU and save it as json file
        pred_mask = pred_probs > prob_thresh
        iou = utils.get_iou(gt_mask, pred_mask) * 100
        iou_dict = {
            "iou": iou,
        }
        with open(iou_save_path / f"{sample_id}_{click_indx}.json", "w") as f:
            json.dump(iou_dict, f)

        # save image with mask
        cv2.imwrite(str(sample_path), image_with_mask[:, :, ::-1])

        # save GT mask if not saved yet
        if click_indx == 0:
            cv2.imwrite(
                str(save_path / f"{sample_id}_gt.jpg"), gt_mask.astype(np.uint8) * 255
            )

    return callback


def get_save_feats_callback(
    logs_path: str, dataset_name: str, save_folder_name: str, exec_for_n_imgs: int = 10
) -> Callable:
    """Callback function to save raw features during evaluation."""
    current_time = datetime.now().strftime("%Y-%m-%d_%H:%M")
    save_path = (
        logs_path / "feats" / dataset_name / f"{save_folder_name}_{current_time}"
    )
    save_imgs_path = save_path / "images"

    save_path.mkdir(parents=True, exist_ok=True)
    save_imgs_path.mkdir(parents=True, exist_ok=True)

    def callback(image, feats, sample_id, click_indx, clicks_list):
        # keep only the first click and limit the number of images to save
        if sample_id >= exec_for_n_imgs or click_indx >= 1:
            return None

        # save feats as .pth file
        for k, v in feats.items():
            torch.save(v, str(save_path / f"{sample_id}_{click_indx}_{k}.pth"))

        # draw clicks on image
        if isinstance(image, dict):
            image = image["image"]

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if clicks_list is not None and len(clicks_list) > 0:
            pos_points = [click.coords for click in clicks_list if click.is_positive]
            neg_points = [
                click.coords for click in clicks_list if not click.is_positive
            ]

            image = draw_points(image, pos_points, color=(0, 255, 0), radius=6)
            image = draw_points(image, neg_points, color=(255, 0, 0), radius=6)

        # additionally save images with clicks, for better correspondence with features
        cv2.imwrite(str(save_imgs_path / f"{sample_id}_{click_indx}_image.jpg"), image)

    return callback


def get_wandb_run_name(eval_cfg: DictConfig) -> str:
    if eval_cfg.wandb_name:
        return eval_cfg.wandb_name
