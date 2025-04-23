"""Visualize precomputed low-res and high-res feature maps using PCA decomposition.
See arguments description for details. 

Usage example:
    python isbench/plots/plot_features.py \
    --lr_feats_path /path/to/lr_feats \
    --hr_feats_paths /path/to/hr_feats1,/path/to/hr_feats2 \
    --feats_captions Caption1, Caption2 \
    --img_ids 0,1,2 \
    --save_dir /path/to/save_dir \
    --title "Feature Visualization" \
    --seed 1
"""

import argparse
import datetime
import os
from pathlib import Path
from typing import List, Tuple

import matplotlib
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import PILToTensor

from core.utils.viz import plot_feats

# Set the backend to Agg for non-GUI environments
matplotlib.use("Agg")


def load_features(
    img_ids: List[int], lr_path: str, hr_paths: List[str]
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[torch.Tensor]]]:
    """Load low-res and high-res features from specified paths.
    NOTE: take features with first click only."""
    images = []
    lr_feats_list = []
    hr_feats_lists = []

    for img_id in img_ids:
        image_path = Path(lr_path) / "images" / f"{img_id}_0_image.jpg"
        image = Image.open(image_path)
        image_tensor = PILToTensor()(image).unsqueeze(0)
        images.append(image_tensor)

        lr_feat_path = Path(lr_path) / f"{img_id}_0_LowRes.pth"
        lr_feats_list.append(torch.load(lr_feat_path)[0])

        hr_feats = []
        for path in hr_paths:
            hr_feat_path = Path(path) / f"{img_id}_0_HighRes.pth"
            hr_feats.append(torch.load(hr_feat_path)[0])
        hr_feats_lists.append(hr_feats)

    return images, lr_feats_list, hr_feats_lists


def main(args: argparse.Namespace) -> None:
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Parse input paths and captions
    lr_path = (
        args.lr_feats_path
    )  # Assume comparison is always done against same low-res features!
    hr_paths = args.hr_feats_paths.split(",")
    captions = args.feats_captions.split(",")
    img_ids = list(map(int, args.img_ids.split(",")))

    # Load features
    images, lr_feats_list, hr_feats_lists = load_features(img_ids, lr_path, hr_paths)

    # Construct save path
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = Path(args.save_dir) / f"feats_plot_{timestamp}.jpg"
    os.makedirs(args.save_dir, exist_ok=True)

    # Plot features
    plot_feats(
        images=images,
        lr_feats_list=lr_feats_list,
        hr_feats_lists=hr_feats_lists,
        save_path=save_path,
        plot_title=args.title,
        feats_captions=captions,
    )
    print(f"Saved plot to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize Low-res and High-res feature maps for comparison via PCA decomposition. \
        Supports simultaneous visualization of multiple feature types for multiple images. \
        Requires pre-computed features saved in the specified folders (this can be done by setting `save_feats=True` in eval_cfg.yaml)."
    )
    parser.add_argument(
        "--lr_feats_path",
        type=str,
        required=True,
        help="Path to folder with saved low-res features. Features are saved during evaluation by setting `save_feats` to True in eval_cfg.yaml.",
    )
    parser.add_argument(
        "--hr_feats_paths",
        type=str,
        required=True,
        help="Comma-separated list of paths to folders with saved high-res features. Features are saved during evaluation by setting `save_feats` to True in eval_cfg.yaml.",
    )
    parser.add_argument(
        "--feats_captions",
        type=str,
        required=True,
        help="Comma-separated list of captions for feature types (e.g., LowRes, Bilinear, ...).",
    )
    parser.add_argument(
        "--img_ids",
        type=str,
        default="0",
        help="Comma-separated list of image IDs to visualize.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=".",
        help="Directory to save the resulting image.",
    )
    parser.add_argument("--title", type=str, default=None, help="Optional plot title.")
    parser.add_argument("--seed", type=int, default=317, help="Random seed.")

    args = parser.parse_args()
    main(args)
