"""Functions for visualizing images and features."""

from functools import lru_cache
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from core.inference.clicker import Click
from core.utils.pca_features import pca


@lru_cache(maxsize=16)
def get_palette(num_cls: int) -> np.ndarray:
    """Generate a color palette for the given number of classes."""
    palette = np.zeros(3 * num_cls, dtype=np.int32)

    for j in range(0, num_cls):
        lab = j
        i = 0

        while lab > 0:
            palette[j * 3 + 0] |= ((lab >> 0) & 1) << (7 - i)
            palette[j * 3 + 1] |= ((lab >> 1) & 1) << (7 - i)
            palette[j * 3 + 2] |= ((lab >> 2) & 1) << (7 - i)
            i = i + 1
            lab >>= 3

    return palette.reshape((-1, 3))


def draw_probmap(x: np.ndarray) -> np.ndarray:
    return cv2.applyColorMap((x * 255).astype(np.uint8), cv2.COLORMAP_HOT)


def draw_points(
    image: np.ndarray, points: List, color: Tuple[int, int, int], radius: int = 3
) -> np.ndarray:
    """Draw points (user clicks) on an image."""
    image = image.copy()
    for p in points:
        if p[0] < 0:
            continue
        if len(p) == 3:
            pradius = {0: 8, 1: 6, 2: 4}[p[2]] if p[2] < 3 else 2
        else:
            pradius = radius
        image = cv2.circle(image, (int(p[1]), int(p[0])), pradius, color, -1)

    return image


def draw_with_blend_and_clicks(
    img: np.ndarray,
    mask: np.ndarray = None,
    alpha: float = 0.6,
    clicks_list: List[Click] = None,
    pos_color: Tuple[int, int, int] = (0, 255, 0),
    neg_color: Tuple[int, int, int] = (255, 0, 0),
    radius: int = 4,
    use_color_mask: bool = False,
) -> np.ndarray:
    """Draw the image with a mask and user clicks.
    By default, darkens the image outside the mask, draws yellowish contours and user clicks.
    If `use_color_mask` is True, draws the mask in color and darkens the image outside the mask.
    This regime is used for the interactive demo.
    """
    result = img.copy()

    # Darken the entire image slightly
    darkend = cv2.addWeighted(result, alpha, np.zeros_like(result), 0, 0)

    if mask is not None:
        # Create a yellowish contour around the mask
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(result, contours, -1, (255, 255, 100), thickness=2)

        mask_region = (mask > 0).astype(np.uint8)
        # Merge the image mask with the darkened background
        result = (
            darkend * (1 - mask_region[:, :, np.newaxis])
            + result * mask_region[:, :, np.newaxis]
        )

        if use_color_mask:
            palette = get_palette(np.max(mask) + 1)
            rgb_mask = palette[mask.astype(np.uint8)]
            result = result * (1 - alpha * mask_region[:, :, None]) + rgb_mask * (
                alpha * mask_region[:, :, None]
            )

        result = result.astype(np.uint8)

    if clicks_list is not None and len(clicks_list) > 0:
        pos_points = [click.coords for click in clicks_list if click.is_positive]
        neg_points = [click.coords for click in clicks_list if not click.is_positive]

        result = draw_points(result, pos_points, pos_color, radius=radius)
        result = draw_points(result, neg_points, neg_color, radius=radius)

    return result


def plot_feats(
    images: list[torch.Tensor],
    lr_feats_list: list[torch.Tensor],
    hr_feats_lists: list[list[torch.Tensor]],
    save_path: str,
    feats_captions: list[str],
    plot_title: str = None,
):
    """Plot low-res and high-res features using PCA."""

    font_size = 45
    font_style = "DejaVu Sans"

    assert (
        len(images) == len(lr_feats_list) == len(hr_feats_lists)
    ), "Mismatch in the length of input lists"

    num_images = len(images)
    num_hr_feats = len(hr_feats_lists[0])

    fig, axes = plt.subplots(
        num_images, num_hr_feats + 2, figsize=(6 * (num_hr_feats + 2), 6 * num_images)
    )
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    if plot_title is not None:
        fig.suptitle(plot_title, fontsize=16)

    if num_images == 1:
        axes = [axes]

    for i in range(num_images):
        image = images[i]
        lr_feats = lr_feats_list[i]
        hr_feats_list = hr_feats_lists[i]

        assert len(image.shape) == 4, f"Expected 4D tensor for image, got {image.shape}"

        min_hw = min(image.shape[2], image.shape[3])
        image = torch.nn.functional.interpolate(
            image, size=(min_hw, min_hw), mode="bilinear", align_corners=False
        )

        axes[i][0].imshow(image[0].permute(1, 2, 0).detach().cpu())
        if i == 0:
            axes[i][0].set_title("Image", fontsize=font_size, fontname=font_style)

        # basis for the PCA used for all the features
        fit_pca = None

        # Compute PCA for the pair of features
        hr_feats = [hr_feats.unsqueeze(0) for hr_feats in hr_feats_list]
        # map the features to the same size
        [lr_feats_pca, *hr_feats_pca], _ = pca(
            [lr_feats.unsqueeze(0), *hr_feats], fit_pca=fit_pca
        )

        # clap the values between 0 and 1
        lr_feats_pca = torch.clamp(lr_feats_pca, 0, 1)
        for j, hr_feats in enumerate(hr_feats_pca):
            hr_feats_pca[j] = torch.clamp(hr_feats[0], 0, 1)

        axes[i][1].imshow(lr_feats_pca[0].permute(1, 2, 0).detach().cpu())
        if i == 0:
            axes[i][1].set_title(
                f"{feats_captions[0]}", fontsize=font_size, fontname=font_style
            )
        for j, hr_feats in enumerate(hr_feats_pca):
            axes[i][j + 2].imshow(hr_feats.permute(1, 2, 0).detach().cpu())
            if i == 0:
                axes[i][j + 2].set_title(
                    f"{feats_captions[j+1]}", fontsize=font_size, fontname=font_style
                )

        remove_axes(axes[i])

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def _remove_axes(ax: plt.Axes) -> None:
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    ax.set_xticks([])
    ax.set_yticks([])


def remove_axes(axes) -> None:
    if len(axes.shape) == 2:
        for ax1 in axes:
            for ax in ax1:
                _remove_axes(ax)
    else:
        for ax in axes:
            _remove_axes(ax)
