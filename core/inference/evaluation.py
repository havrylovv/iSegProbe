"""Evaluation of samples and datasets."""

from copy import deepcopy
from time import time
from typing import Callable, List, Tuple

import numpy as np
import torch

from core.data.base_dataset import iSegBaseDataset
from core.inference import utils
from core.inference.clicker import Click, Clicker
from core.inference.predictors import BasePredictor

try:
    get_ipython()
    from tqdm import tqdm_notebook as tqdm
except NameError:
    from tqdm import tqdm


def evaluate_dataset(
    dataset: iSegBaseDataset, predictor: BasePredictor, **kwargs
) -> Tuple[List[np.ndarray], float]:
    all_ious = []
    start_time = time()
    for index in tqdm(range(len(dataset)), leave=False):
        sample = dataset.get_sample(index)
        for object_id in sample.objects_ids:
            _, sample_ious, _ = evaluate_sample(
                sample.image,
                sample.gt_mask(object_id),
                predictor,
                sample_id=index,
                **kwargs,
            )
            all_ious.append(sample_ious)
    end_time = time()
    elapsed_time = end_time - start_time
    return all_ious, elapsed_time


def evaluate_sample(
    image: np.ndarray,
    gt_mask: np.ndarray,
    predictor: BasePredictor,
    max_iou_thr: float,
    pred_thr: float = 0.49,
    min_clicks: int = 1,
    max_clicks: int = 20,
    sample_id: int = None,
    callback: Callable = None,
    feats_callback: Callable = None,
) -> Tuple[List[Click], np.ndarray, np.ndarray]:
    clicker = Clicker(gt_mask=gt_mask)
    pred_mask = np.zeros_like(gt_mask)
    ious_list = []
    with torch.no_grad():
        predictor.set_input_image(image)

        for click_indx in range(max_clicks):

            clicker.make_next_click(pred_mask)
            # should be located before get_prediction, as it changes the state of the predictor
            if feats_callback is not None:
                _, feats = predictor.get_lowres_highres_feats(deepcopy(clicker))
                feats_callback(image, feats, sample_id, click_indx, clicker.clicks_list)

            pred_probs = predictor.get_prediction(clicker)
            pred_mask = pred_probs > pred_thr

            if callback is not None:
                callback(
                    image,
                    gt_mask,
                    pred_probs,
                    sample_id,
                    click_indx,
                    clicker.clicks_list,
                )

            iou = utils.get_iou(gt_mask, pred_mask)
            ious_list.append(iou)

            if iou >= max_iou_thr and click_indx + 1 >= min_clicks:
                break

        return clicker.clicks_list, np.array(ious_list, dtype=np.float32), pred_probs


def get_points_nd(clicks_lists: List[List[Click]]) -> List[np.ndarray]:
    total_clicks = []
    num_pos_clicks = [
        sum(x.is_positive for x in clicks_list) for clicks_list in clicks_lists
    ]
    num_neg_clicks = [
        len(clicks_list) - num_pos
        for clicks_list, num_pos in zip(clicks_lists, num_pos_clicks)
    ]
    num_max_points = max(num_pos_clicks + num_neg_clicks)
    num_max_points = max(1, num_max_points)

    for clicks_list in clicks_lists:
        pos_clicks = [
            click.coords_and_indx for click in clicks_list if click.is_positive
        ]
        pos_clicks = pos_clicks + (num_max_points - len(pos_clicks)) * [(-1, -1, -1)]

        neg_clicks = [
            click.coords_and_indx for click in clicks_list if not click.is_positive
        ]
        neg_clicks = neg_clicks + (num_max_points - len(neg_clicks)) * [(-1, -1, -1)]
        total_clicks.append(pos_clicks + neg_clicks)
    return total_clicks
